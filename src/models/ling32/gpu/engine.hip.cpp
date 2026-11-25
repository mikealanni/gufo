#include "src/models/ling32/gpu/engine.hpp"

#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/ling32/gpu/kernels.hip.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::ling32::gpu {
namespace {

using gufo::core::GgmlType;
using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;
namespace qfn = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kBatchTokens = 8;
constexpr std::uint32_t kPrefillTile = 4096;
constexpr std::uint32_t kPrefillKeyCap = 32768;
constexpr std::uint32_t kRoutePad = 16;
constexpr std::uint32_t kMtpBackfill = 32;

bool MoeVecSupported(int type) {
  return type == 8 || type == 12 || type == 13 || type == 14;
}

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr)
    *error = message;
  return false;
}

std::string P(std::uint32_t layer, const char* name) {
  return "blk." + std::to_string(layer) + "." + name;
}

struct DeviceTensor {
  void* data{nullptr};
  std::size_t bytes{0};
  GgmlType type{GgmlType::kF16};
  std::uint32_t rows{0};
  std::uint32_t cols{0};
  std::uint32_t experts{1};
};

struct LayerWeights {
  DeviceTensor attn_norm;
  DeviceTensor ffn_norm;
  DeviceTensor attn_q;
  DeviceTensor attn_k;
  DeviceTensor attn_v;
  DeviceTensor conv_q;
  DeviceTensor conv_k;
  DeviceTensor conv_v;
  DeviceTensor ssm_f_a;
  DeviceTensor ssm_g_a;
  DeviceTensor ssm_beta;
  DeviceTensor ssm_dt;
  DeviceTensor ssm_a;
  DeviceTensor ssm_norm;
  DeviceTensor attn_kv_a;
  DeviceTensor attn_kv_norm;
  DeviceTensor attn_k_b;
  DeviceTensor attn_v_b;
  DeviceTensor attn_gate;
  DeviceTensor attn_output;
  DeviceTensor ffn_gate;
  DeviceTensor ffn_up;
  DeviceTensor ffn_down;
  DeviceTensor router;
  DeviceTensor router3;
  DeviceTensor router_bias;
  DeviceTensor expert_gate;
  DeviceTensor expert_up;
  DeviceTensor expert_down;
  DeviceTensor shared_gate;
  DeviceTensor shared_up;
  DeviceTensor shared_down;
  DeviceTensor kda_stack;
  DeviceTensor wk16;
  DeviceTensor wv16;
  DeviceTensor shared_gu;
  DeviceTensor eh_proj;
  DeviceTensor enorm;
  DeviceTensor hnorm;
  DeviceTensor layer_out_norm;
};

struct DevBuf {
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& other) noexcept
      : data(other.data), raw_base(other.raw_base), bytes(other.bytes) {
    other.raw_base = nullptr;
    other.data = nullptr;
    other.bytes = 0;
  }
  DevBuf& operator=(DevBuf&& other) noexcept {
    if (this != &other) {
      Release();
      data = other.data;
      raw_base = other.raw_base;
      bytes = other.bytes;
      other.data = nullptr;
      other.raw_base = nullptr;
      other.bytes = 0;
    }
    return *this;
  }
  void Release() {
    if (raw_base == nullptr)
      return;
    auto& g = Guards();
    for (std::size_t i = 0; i < g.size(); ++i) {
      if (g[i].base == raw_base) {
        g.erase(g.begin() + static_cast<std::ptrdiff_t>(i));
        break;
      }
    }
    hipFree(raw_base);
    raw_base = nullptr;
    data = nullptr;
  }
  ~DevBuf() { Release(); }

  struct GuardEntry {
    const char* name;
    std::uint8_t* base;
    std::size_t size;
  };
  static std::vector<GuardEntry>& Guards() {
    static std::vector<GuardEntry> g;
    return g;
  }
  static constexpr std::size_t kGuardBytes = 8192;

  [[nodiscard]] bool Alloc(std::size_t count, std::size_t elem,
                           std::string* error, const char* what) {
    bytes = count * elem;
    if (bytes == 0)
      return true;
    static const bool guard = std::getenv("LING_GUARD") != nullptr;
    const std::size_t total = guard ? bytes + 2 * kGuardBytes : bytes;
    void* raw = nullptr;
    if (hipMalloc(&raw, total) != hipSuccess) {
      bytes = 0;
      return Fail(error, std::string("hipMalloc failed for ") + what);
    }
    if (guard) {
      hipMemset(raw, 0xA5, total);
      Guards().push_back({what, static_cast<std::uint8_t*>(raw), total});
      data = static_cast<std::uint8_t*>(raw) + kGuardBytes;
      raw_base = raw;
    } else {
      data = raw;
      raw_base = raw;
    }
    if (std::getenv("LING_LOG_ALLOC") != nullptr) {
      std::fprintf(stderr, "[alloc] %-14s %p +%zu (end %p)\n", what, data,
                   bytes, static_cast<char*>(data) + bytes);
    }
    return true;
  }

  static bool CheckGuards(const char* when) {
    bool ok = true;
    std::vector<std::uint8_t> host(DevBuf::kGuardBytes);
    for (const auto& g : Guards()) {
      for (int side = 0; side < 2; ++side) {
        const std::uint8_t* src =
            side == 0 ? g.base : g.base + g.size - DevBuf::kGuardBytes;
        hipMemcpy(host.data(), src, DevBuf::kGuardBytes, hipMemcpyDeviceToHost);
        for (std::size_t i = 0; i < host.size(); ++i) {
          if (host[i] != 0xA5) {
            std::fprintf(
                stderr, "[guard] OVERFLOW %s of '%s' at +%zu (%s) value %02x\n",
                side == 0 ? "before" : "after", g.name, i, when, host[i]);
            ok = false;
            break;
          }
        }
      }
    }
    return ok;
  }

  [[nodiscard]] float* F() const { return static_cast<float*>(data); }
  [[nodiscard]] std::int32_t* I() const {
    return static_cast<std::int32_t*>(data);
  }

  void* data{nullptr};
  void* raw_base{nullptr};
  std::size_t bytes{0};
};

int TypeCode(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return 0;
    case GgmlType::kQ4_0:
      return 2;
    case GgmlType::kQ8_0:
      return 8;
    case GgmlType::kQ4_K:
      return 12;
    case GgmlType::kQ5_K:
      return 13;
    case GgmlType::kQ6_K:
      return 14;
    default:
      return -1;
  }
}

qfn::WeightType WeightTypeOf(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return qfn::WeightType::kF32;
    case GgmlType::kF16:
      return qfn::WeightType::kF16;
    case GgmlType::kQ8_0:
      return qfn::WeightType::kQ8_0;
    case GgmlType::kQ4_K:
      return qfn::WeightType::kQ4_K;
    case GgmlType::kQ5_K:
      return qfn::WeightType::kQ5_K;
    case GgmlType::kQ6_K:
      return qfn::WeightType::kQ6_K;
    default:
      return qfn::WeightType::kF32;
  }
}

DeviceTensor MakeTensor(const GgufTensorInfo& t, void* device) {
  DeviceTensor out;
  out.data = device;
  out.bytes = t.size_bytes;
  out.type = t.type;
  out.cols =
      t.dimensions.empty() ? 0 : static_cast<std::uint32_t>(t.dimensions[0]);
  out.rows = 1;
  for (std::size_t d = 1; d < t.dimensions.size(); ++d) {
    out.rows *= static_cast<std::uint32_t>(t.dimensions[d]);
  }
  if (t.dimensions.size() > 2) {
    out.experts = static_cast<std::uint32_t>(t.dimensions[2]);
    out.rows /= out.experts;
  }
  return out;
}

}  // namespace

static const bool kProf = std::getenv("LING_PROF") != nullptr;
static double g_prof[4];
static int g_prof_n;
static std::chrono::steady_clock::time_point g_prof_t;
#define PROF_BEGIN()                               \
  do {                                             \
    if (kProf) {                                   \
      hipStreamSynchronize(stream);                \
      g_prof_t = std::chrono::steady_clock::now(); \
    }                                              \
  } while (0)
#define PROF_END(i)                                                 \
  do {                                                              \
    if (kProf) {                                                    \
      hipStreamSynchronize(stream);                                 \
      g_prof[i] += std::chrono::duration<double, std::milli>(       \
                       std::chrono::steady_clock::now() - g_prof_t) \
                       .count();                                    \
    }                                                               \
  } while (0)
#define PROF_REPORT()                                                    \
  do {                                                                   \
    if (kProf && ++g_prof_n % 16 == 0) {                                 \
      std::fprintf(stderr, "[prof/tok ms] kda=%.2f mla=%.2f ffn=%.2f\n", \
                   g_prof[0] / g_prof_n, g_prof[1] / g_prof_n,           \
                   g_prof[2] / g_prof_n);                                \
    }                                                                    \
  } while (0)

struct Model::Impl {
  std::unique_ptr<GgufReader> reader;
  Config config;
  hipStream_t stream{nullptr};
  std::unordered_map<std::string, DeviceTensor> tensors;
  std::vector<LayerWeights> layers;
  LayerWeights mtp;
  DeviceTensor token_embd;
  DeviceTensor output_norm;
  DeviceTensor output;
  std::size_t resident{0};
  std::uint32_t n_layers{0};
};

struct Session::Impl {
  struct LayerState {
    DevBuf conv;
    DevBuf state;
    DevBuf cache;
    DevBuf ck_state;
    DevBuf ck_hist;
    DevBuf ck_raw;
    DevBuf pool;
  };

  std::shared_ptr<Model> model;
  hipStream_t stream{nullptr};
  std::uint32_t max_context{0};
  std::uint32_t position{0};
  std::uint32_t mtp_position{0};
  std::int32_t mtp_prev_token{-1};
  bool mtp_has_prev{false};
  bool mtp_enabled{false};
  std::vector<float> logits;
  std::vector<float> mtp_logits;
  std::vector<LayerState> layers;

  DevBuf token;
  DevBuf q8;
  DevBuf x;
  DevBuf xn;
  DevBuf block;
  DevBuf fin;
  DevBuf ffn;
  DevBuf logits_dev;
  DevBuf q;
  DevBuf k;
  DevBuf v;
  DevBuf oq;
  DevBuf ok;
  DevBuf ov;
  DevBuf fa;
  DevBuf ga;
  DevBuf beta;
  DevBuf qall;
  DevBuf kvall;
  DevBuf qrope;
  DevBuf kpe;
  DevBuf keyrow;
  DevBuf scores;
  DevBuf qabs;
  DevBuf qvec;
  DevBuf ctxlat;
  DevBuf pos_dev;
  DevBuf pos_mtp;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graph_exec{nullptr};
  int decode_calls{0};
  DevBuf pm;
  DevBuf pl;
  DevBuf pacc;
  DevBuf vhead;
  DevBuf gate;
  DevBuf attn;
  DevBuf attn_proj;
  DevBuf kda_out;
  DevBuf kda_so;
  DevBuf qkvfg;
  hipblasHandle_t blas{nullptr};
  DevBuf pf_kn;
  DevBuf mtp_bf;
  DevBuf lat_sel;
  DevBuf sel_scores;
  DevBuf sel_idx;
  DevBuf sel_pos;
  DevBuf spec_h;
  DevBuf spec_logits_dev;
  DevBuf sv_prev;
  std::vector<float> spec_logits_host;
  std::vector<std::int32_t> spec_tokens;
  std::uint32_t spec_pos0{0};
  std::uint32_t spec_n{0};
  DevBuf pf_k16;
  DevBuf pf_v16;
  DevBuf pf_q16;
  DevBuf pf_q8;
  DevBuf pf_vv;
  DevBuf pf_lat16;
  DevBuf pf_fin16;
  DevBuf pf_x3;
  DevBuf pf_act16;
  DevBuf pf_gate_e;
  DevBuf pf_up_e;
  DevBuf pf_down_e;
  DevBuf pf_router;
  DevBuf pf_counts;
  DevBuf pf_bounds;
  DevBuf pf_cursors;
  DevBuf pf_rows_token;
  DevBuf pf_rows_slot;
  DevBuf pf_tiles48;
  DevBuf pf_tiles64;
  DevBuf pf_tiles128;
  std::vector<std::uint32_t> pf_host_counts;
  std::vector<std::int32_t> pf_host_tiles;
  DevBuf sgu;
  DevBuf mtp_h;
  DevBuf mtp_prev_h;
  DevBuf mtp_pending_h;
  DevBuf mtp_draft_h;
  DevBuf mtp_embd;
  DevBuf mtp_e_norm;
  DevBuf mtp_h_norm;
  DevBuf mtp_eh;
  DevBuf mtp_cur;
  DevBuf mtp_logits_dev;
  DevBuf router_logits;
  DevBuf ids;
  DevBuf weights;
  DevBuf gf;
  DevBuf uf;
  DevBuf act;
  DevBuf part;
  DevBuf sg;
  DevBuf su;
  DevBuf shared_act;
  DevBuf shared_part;
  std::unique_ptr<qfn::NpuGemm> npu;
  int npu_a{-1};
  int npu_c[2]{-1, -1};
  std::vector<std::array<int, 2>> npu_sh;
  std::uint64_t npu_ticket{0};
  static constexpr int kQkvSlices = 12;
  int npu_qa[2]{-1, -1};
  std::vector<int> npu_qc[2];
  std::vector<std::vector<int>> npu_qkv;
  std::uint64_t npu_qticket[2]{0, 0};
  DevBuf x1;
  DevBuf dense_gate;
  DevBuf dense_up;
  DevBuf dense_act;
  DevBuf ex_g;
  DevBuf ex_u;
  DevBuf ex_d;

  const Config& config() const { return model->impl_->config; }

  bool Dense(const DeviceTensor& w, const float* in, float* out,
             std::uint32_t rows, std::uint32_t cols, std::uint32_t tokens,
             std::string* error) {
    const int type = TypeCode(w.type);
    if (type < 0) {
      return Fail(error, "unsupported dense weight type " +
                             std::to_string(static_cast<int>(w.type)));
    }
    int rc = -1;
    if (w.type == GgmlType::kF32) {
      qfn::SmallGemm(w.data, qfn::WeightType::kF32, in, out, tokens, rows, cols,
                     stream);
      return true;
    }
    if (w.type == GgmlType::kQ8_0) {
      if (tokens <= 8) {
        if (qfn_mmq_quantize_q8_1(in, q8.F(), static_cast<int>(tokens),
                                  static_cast<int>(cols), stream) != 0) {
          return Fail(error, "input quantization failed");
        }
        rc = qfn_mmq_q8_0_dense_vec_preq(
            w.data, nullptr, q8.F(), out, static_cast<int>(rows),
            static_cast<int>(tokens), static_cast<int>(cols), stream);
      } else {
        rc = qfn_mmq_q8_0_dense(w.data, in, out, static_cast<int>(rows),
                                static_cast<int>(tokens),
                                static_cast<int>(cols), stream);
      }
    } else {
      rc = qfn_mmq_dense_vec(type, w.data, in, out, static_cast<int>(rows),
                             static_cast<int>(cols), static_cast<int>(tokens),
                             stream);
    }
    if (rc != 0)
      return Fail(error, "dense projection failed");
    return true;
  }

  void KdaBlock(const LayerWeights& w, std::uint32_t layer, const float* xn,
                float* block) {
    const Config& c = config();
    const std::uint32_t di = c.KdaDim();
    const std::uint32_t dh = c.kda_head_dim;
    float* qkvfg_p = qkvfg.F();
    Dense(w.kda_stack, xn, qkvfg_p, 5U * di, c.hidden_size, 1, nullptr);
    KdaConv(qkvfg_p, qkvfg_p + di, qkvfg_p + 2 * di,
            static_cast<const float*>(w.conv_q.data),
            static_cast<const float*>(w.conv_k.data),
            static_cast<const float*>(w.conv_v.data), layers[layer].conv.F(),
            oq.F(), ok.F(), ov.F(), di, c.ssm_conv_kernel, stream);
    float* fa_p = qkvfg_p + 3 * di;
    float* ga_p = qkvfg_p + 4 * di;
    Dense(w.ssm_beta, xn, beta.F(), c.num_heads, c.hidden_size, 1, nullptr);
    KdaDecode(oq.F(), ok.F(), ov.F(), fa_p, ga_p, beta.F(),
              static_cast<const float*>(w.ssm_dt.data),
              static_cast<const float*>(w.ssm_a.data),
              static_cast<const float*>(w.ssm_norm.data),
              layers[layer].state.F(), kda_so.F(), kda_out.F(), c.num_heads,
              c.gate_lower_bound, c.rms_eps, stream);
    Dense(w.attn_output, kda_out.F(), attn_proj.F(), c.hidden_size, di, 1,
          nullptr);
    CopyRow(attn_proj.F(), block, c.hidden_size, stream);
  }

  static bool ValidateOn() {
    static const bool v = std::getenv("LING_VALIDATE") != nullptr;
    return v;
  }

  [[nodiscard]] bool CheckFloats(const char* what, const float* dev,
                                 std::size_t count, std::uint32_t layer,
                                 std::uint32_t pos0) {
    hipStreamSynchronize(stream);
    std::vector<float> host(count);
    if (hipMemcpy(host.data(), dev, count * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      std::fprintf(stderr, "[validate] %s copy failed layer %u pos0 %u\n", what,
                   layer, pos0);
      return false;
    }
    float peak = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
      const bool forced_inf =
          host[i] == INFINITY && std::string(what) == "sel_scores";
      if (!std::isfinite(host[i]) && !forced_inf) {
        std::fprintf(stderr,
                     "[validate] NON-FINITE %s layer %u pos0 %u at %zu\n", what,
                     layer, pos0, i);
        return false;
      }
      peak = std::max(peak, std::fabs(host[i]));
    }
    if (peak > 6.0e4F) {
      std::fprintf(stderr, "[validate] LARGE %s=%g layer %u pos0 %u\n", what,
                   peak, layer, pos0);
    }
    return true;
  }

  [[nodiscard]] bool CheckHalves(const char* what, const void* dev,
                                 std::size_t count, std::uint32_t layer,
                                 std::uint32_t pos0) {
    hipStreamSynchronize(stream);
    std::vector<std::uint16_t> host(count);
    if (hipMemcpy(host.data(), dev, count * 2, hipMemcpyDeviceToHost) !=
        hipSuccess) {
      std::fprintf(stderr, "[validate] %s copy failed layer %u pos0 %u\n", what,
                   layer, pos0);
      return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
      if ((host[i] & 0x7C00U) == 0x7C00U) {
        std::fprintf(stderr,
                     "[validate] F16 INF/NAN %s layer %u pos0 %u at %zu\n",
                     what, layer, pos0, i);
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool CheckInts(const char* what, const std::int32_t* dev,
                               std::size_t count, std::int32_t lo,
                               std::int32_t hi, std::uint32_t layer,
                               std::uint32_t pos0) {
    hipStreamSynchronize(stream);
    std::vector<std::int32_t> host(count);
    hipMemcpy(host.data(), dev, count * sizeof(std::int32_t),
              hipMemcpyDeviceToHost);
    for (std::size_t i = 0; i < count; ++i) {
      if (host[i] < lo || host[i] >= hi) {
        std::fprintf(stderr,
                     "[validate] OUT OF RANGE %s[%zu]=%d not in [%d,%d) layer "
                     "%u pos0 %u\n",
                     what, i, host[i], lo, hi, layer, pos0);
        return false;
      }
    }
    return true;
  }

  // Latent rows the attention of rows [pos0, pos0 + n) reads: the whole cache
  // while the prefix fits the sparse budget, else sink + recent + top blocks
  // followed by the unpooled tail and the new rows. `probe` is the absorbed
  // query ([heads][576]) the blocks are scored with.
  const float* SelectLatents(std::uint32_t layer, std::uint32_t pos0,
                             std::uint32_t n, const float* probe,
                             bool need_probe, std::function<void()> make_probe,
                             std::uint32_t* prefix_rows,
                             std::uint32_t* total_rows) {
    float* cache = layers[layer].cache.F();
    const std::uint32_t nbf = pos0 / kPoolBlockRows;
    if (nbf <= kSparseBlocks) {
      *prefix_rows = pos0;
      *total_rows = pos0 + n;
      return cache;
    }
    if (need_probe)
      make_probe();
    SparseSelect(layers[layer].pool.F(), probe, sel_scores.F(), sel_idx.I(),
                 nbf, kSparseBlocks, kSparseRecent, stream);
    if (ValidateOn()) {
      (void)CheckFloats("sel_scores", sel_scores.F(), nbf, layer, pos0);
      (void)CheckInts("sel_idx", sel_idx.I(), kSparseBlocks, 0,
                      static_cast<std::int32_t>(nbf), layer, pos0);
    }
    const std::uint32_t tail_start = nbf * kPoolBlockRows;
    *prefix_rows = kSparseBlocks * kPoolBlockRows + (pos0 - tail_start);
    *total_rows = *prefix_rows + n;
    GatherLatents(cache, sel_idx.I(), lat_sel.F(), kSparseBlocks, tail_start,
                  *total_rows, stream);
    if (std::getenv("LING_NO_SYNC_SPARSE") == nullptr)
      hipStreamSynchronize(stream);
    return lat_sel.F();
  }

  void UpdatePool(std::uint32_t layer, std::uint32_t pos0, std::uint32_t n) {
    const std::uint32_t first = pos0 / kPoolBlockRows;
    const std::uint32_t last = (pos0 + n) / kPoolBlockRows;
    if (last > first && layers[layer].pool.data != nullptr)
      PoolUpdate(layers[layer].cache.F(), layers[layer].pool.F(), first,
                 last - first, stream);
  }

  void MlaBlock(const LayerWeights& w, std::uint32_t layer, const float* xn,
                float* block, const std::int32_t* posp,
                std::uint32_t pos_host) {
    const Config& c = config();
    const std::uint32_t nh = c.num_heads;
    const std::uint32_t qk = c.key_length_mla;
    const std::uint32_t rope = c.rope_dimension;
    const std::uint32_t nope = c.QkNope();
    const std::uint32_t kv = c.kv_lora_rank;
    const std::uint32_t vd = c.MlaVDim();
    const std::uint32_t key = kv + rope;

    Dense(w.attn_q, xn, qall.F(), nh * qk, c.hidden_size, 1, nullptr);
    Dense(w.attn_kv_a, xn, kvall.F(), key, c.hidden_size, 1, nullptr);
    qfn::RmsNormRows(kvall.F(), static_cast<const float*>(w.attn_kv_norm.data),
                     kvall.F(), 1, kv, 1, c.rms_eps, stream);
    float* cache = layers[layer].cache.F();
    RopeHeads(kvall.F() + kv, cache + kv, 1, rope, 0, rope, 0, c.freq_base,
              stream, 0, posp, key);
    StoreRow(kvall.F(), cache, posp, kv, key, stream);
    RopeHeads(qall.F(), qvec.F() + kv, nh, qk, nope, rope, 0, c.freq_base,
              stream, key, posp, 0);
    HeadGemvQ8(w.attn_k_b.data, qall.F(), qvec.F(), nh, kv, nope, qk, key,
               stream);
    const float scale = 1.0F / std::sqrt(static_cast<float>(qk));
    UpdatePool(layer, pos_host, 1);
    std::uint32_t prefix = 0;
    std::uint32_t total = 0;
    const float* lat =
        SelectLatents(layer, pos_host, 1, qvec.F(), false, {}, &prefix, &total);
    const std::int32_t* rows_ptr = posp;
    if (lat != cache) {
      SetI32(sel_pos.I(), static_cast<std::int32_t>(prefix), stream);
      rows_ptr = sel_pos.I();
    }
    MlaDecode(qvec.F(), lat, pm.F(), pl.F(), pacc.F(), ctxlat.F(), rows_ptr,
              scale, stream);
    HeadGemvQ8(w.attn_v_b.data, ctxlat.F(), attn.F(), nh, vd, kv, kv, vd,
               stream);
    Dense(w.attn_gate, xn, gate.F(), nh, c.hidden_size, 1, nullptr);
    MlaGate(attn.F(), gate.F(), attn.F(), nh, vd, stream);
    Dense(w.attn_output, attn.F(), block, c.hidden_size, nh * vd, 1, nullptr);
  }

  void DenseFfn(const LayerWeights& w, const float* fin, float* out) {
    const Config& c = config();
    const std::uint32_t ff = c.dense_ff;
    Dense(w.ffn_gate, fin, dense_gate.F(), ff, c.hidden_size, 1, nullptr);
    Dense(w.ffn_up, fin, dense_up.F(), ff, c.hidden_size, 1, nullptr);
    SwigluPlain(dense_gate.F(), dense_up.F(), dense_act.F(), ff, stream);
    Dense(w.ffn_down, dense_act.F(), out, c.hidden_size, ff, 1, nullptr);
  }

  bool MoeFfn(const LayerWeights& w, const float* fin, float* out,
              std::uint32_t layer, std::string* error) {
    const Config& c = config();
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t eff = c.expert_ff;
    Dense(w.router, fin, router_logits.F(), c.expert_count, h, 1, nullptr);
    RouterLing(router_logits.F(), static_cast<const float*>(w.router_bias.data),
               ids.I(), weights.F(), c.expert_count, c.expert_groups,
               c.expert_groups_used, c.expert_used, c.expert_weights_scale,
               stream);

    const int gate_type = TypeCode(w.expert_gate.type);
    const int up_type = TypeCode(w.expert_up.type);
    const int down_type = TypeCode(w.expert_down.type);
    if (gate_type < 0 || up_type < 0 || down_type < 0) {
      return Fail(error, "unsupported expert weight type");
    }
    const std::uint32_t used = c.expert_used;
    const std::uint32_t ge = eff * h;
    const std::uint32_t de = h * eff;
    bool quantized =
        MoeVecSupported(gate_type) && MoeVecSupported(up_type) &&
        MoeVecSupported(down_type) &&
        qfn_mmq_moe_vec(gate_type, w.expert_gate.data, fin, ids.I(), gf.F(),
                        static_cast<int>(eff), static_cast<int>(h), 1,
                        static_cast<int>(c.expert_count),
                        static_cast<int>(used), stream) == 0;
    quantized = quantized &&
                qfn_mmq_moe_vec(up_type, w.expert_up.data, fin, ids.I(), uf.F(),
                                static_cast<int>(eff), static_cast<int>(h), 1,
                                static_cast<int>(c.expert_count),
                                static_cast<int>(used), stream) == 0;
    if (quantized) {
      if (c.swiglu_clamp_exp[layer] > 1.0e-6F) {
        SwigluClamp(gf.F(), uf.F(), act.F(), used * eff,
                    c.swiglu_clamp_exp[layer], stream);
      } else {
        SwigluPlain(gf.F(), uf.F(), act.F(), used * eff, stream);
      }
      quantized =
          qfn_mmq_moe_vec(down_type, w.expert_down.data, act.F(), ids.I(),
                          part.F(), static_cast<int>(h), static_cast<int>(eff),
                          static_cast<int>(used),
                          static_cast<int>(c.expert_count), 1, stream) == 0;
    }
    if (!quantized) {
      DequantExperts(w.expert_gate.data, ids.I(), ex_g.F(), used, ge,
                     w.expert_gate.bytes / c.expert_count, gate_type, stream);
      DequantExperts(w.expert_up.data, ids.I(), ex_u.F(), used, ge,
                     w.expert_up.bytes / c.expert_count, up_type, stream);
      DequantExperts(w.expert_down.data, ids.I(), ex_d.F(), used, de,
                     w.expert_down.bytes / c.expert_count, down_type, stream);
      for (std::uint32_t slot = 0; slot < used; ++slot) {
        qfn::SmallGemm(ex_g.F() + static_cast<std::size_t>(slot) * ge,
                       qfn::WeightType::kF32, fin,
                       gf.F() + static_cast<std::size_t>(slot) * eff, 1, eff, h,
                       stream);
        qfn::SmallGemm(ex_u.F() + static_cast<std::size_t>(slot) * ge,
                       qfn::WeightType::kF32, fin,
                       uf.F() + static_cast<std::size_t>(slot) * eff, 1, eff, h,
                       stream);
      }
      if (c.swiglu_clamp_exp[layer] > 1.0e-6F) {
        SwigluClamp(gf.F(), uf.F(), act.F(), used * eff,
                    c.swiglu_clamp_exp[layer], stream);
      } else {
        SwigluPlain(gf.F(), uf.F(), act.F(), used * eff, stream);
      }
      for (std::uint32_t slot = 0; slot < used; ++slot) {
        qfn::SmallGemm(ex_d.F() + static_cast<std::size_t>(slot) * de,
                       qfn::WeightType::kF32,
                       act.F() + static_cast<std::size_t>(slot) * eff,
                       part.F() + static_cast<std::size_t>(slot) * h, 1, h, eff,
                       stream);
      }
    }

    float* sg_p = sg.F();
    float* su_p = su.F();
    if (w.shared_gu.data != nullptr) {
      Dense(w.shared_gu, fin, sgu.F(), 2U * eff, h, 1, nullptr);
      sg_p = sgu.F();
      su_p = sgu.F() + eff;
    } else {
      Dense(w.shared_gate, fin, sg_p, eff, h, 1, nullptr);
      Dense(w.shared_up, fin, su_p, eff, h, 1, nullptr);
    }
    if (c.swiglu_clamp_shexp[layer] > 1.0e-6F) {
      SwigluClamp(sg_p, su_p, shared_act.F(), eff, c.swiglu_clamp_shexp[layer],
                  stream);
    } else {
      SwigluPlain(sg_p, su_p, shared_act.F(), eff, stream);
    }
    Dense(w.shared_down, shared_act.F(), shared_part.F(), h, eff, 1, nullptr);
    MoeCombine(part.F(), weights.F(), shared_part.F(), out, c.expert_used, h,
               stream);
    return true;
  }

  bool EncodeDecode(std::string* error) {
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    qfn::EmbedTokens(m.token_embd.data, WeightTypeOf(m.token_embd.type),
                     token.I(), x.F(), 1, c.hidden_size, 1, stream);

    for (std::uint32_t l = 0; l < m.n_layers; ++l) {
      const LayerWeights& w = m.layers[l];
      qfn::RmsNormRows(x.F(), static_cast<const float*>(w.attn_norm.data),
                       xn.F(), 1, c.hidden_size, 1, c.rms_eps, stream);
      PROF_BEGIN();
      if (c.IsKda(l)) {
        KdaBlock(w, l, xn.F(), block.F());
        PROF_END(0);
      } else {
        MlaBlock(w, l, xn.F(), block.F(), pos_dev.I(), position);
        PROF_END(1);
      }
      AddTo(x.F(), block.F(), c.hidden_size, stream);

      qfn::RmsNormRows(x.F(), static_cast<const float*>(w.ffn_norm.data),
                       fin.F(), 1, c.hidden_size, 1, c.rms_eps, stream);
      PROF_BEGIN();
      if (c.IsDense(l)) {
        DenseFfn(w, fin.F(), ffn.F());
      } else if (!MoeFfn(w, fin.F(), ffn.F(), l, error)) {
        return false;
      }
      PROF_END(2);
      AddTo(x.F(), ffn.F(), c.hidden_size, stream);
    }
    PROF_REPORT();

    qfn::RmsNormRows(x.F(), static_cast<const float*>(m.output_norm.data),
                     xn.F(), 1, c.hidden_size, 1, c.rms_eps, stream);
    hipMemcpyAsync(mtp_h.F(), xn.F(), c.hidden_size * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    if (!Dense(m.output, xn.F(), logits_dev.F(), c.vocab_size, c.hidden_size, 1,
               error)) {
      return false;
    }
    return true;
  }

  bool RunDecode(std::string* error) {
    static const bool kGraphs = std::getenv("LING_GRAPH") != nullptr;
    ++decode_calls;
    if (!kGraphs || kProf || decode_calls < 3)
      return EncodeDecode(error);
    if (graph_exec == nullptr) {
      if (hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) !=
          hipSuccess) {
        return Fail(error, "graph capture begin failed");
      }
      const bool ok = EncodeDecode(error);
      if (hipStreamEndCapture(stream, &graph) != hipSuccess || !ok) {
        return Fail(error, "graph capture failed");
      }
      if (hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0) !=
          hipSuccess) {
        return Fail(error, "graph instantiate failed");
      }
    }
    if (hipGraphLaunch(graph_exec, stream) != hipSuccess)
      return Fail(error, "graph launch failed");
    return true;
  }

  bool Forward(std::int32_t token_id, std::string* error) {
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    if (c.HasMtp() && mtp_enabled && mtp_has_prev &&
        !MtpStep(mtp_prev_token, mtp_pending_h.F(), position - 1, position - 1,
                 position, false, false, error)) {
      return false;
    }
    SetI32(token.I(), token_id, stream);
    SetI32(pos_dev.I(), static_cast<std::int32_t>(position), stream);
    if (!RunDecode(error))
      return false;
    if (hipMemcpyAsync(logits.data(), logits_dev.F(),
                       logits.size() * sizeof(float), hipMemcpyDeviceToHost,
                       stream) != hipSuccess) {
      return Fail(error, "logit download failed");
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
      return Fail(error, "stream synchronization failed");
    }
    if (c.HasMtp()) {
      hipMemcpyAsync(mtp_pending_h.F(), mtp_prev_h.F(),
                     c.hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
                     stream);
      hipMemcpyAsync(mtp_prev_h.F(), mtp_h.F(), c.hidden_size * sizeof(float),
                     hipMemcpyDeviceToDevice, stream);
      mtp_prev_token = token_id;
      mtp_has_prev = true;
    }
    ++position;
    return true;
  }

  bool MoeFfnBatch(const LayerWeights& w, const float* fin, float* out,
                   std::uint32_t layer, std::uint32_t n, std::string* error) {
    const Config& c = config();
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t eff = c.expert_ff;
    const std::uint32_t used = c.expert_used;
    Dense(w.shared_gate, fin, sg.F(), eff, h, n, nullptr);
    Dense(w.shared_up, fin, su.F(), eff, h, n, nullptr);
    if (c.swiglu_clamp_shexp[layer] > 1.0e-6F) {
      SwigluClamp(sg.F(), su.F(), shared_act.F(),
                  static_cast<std::size_t>(n) * eff,
                  c.swiglu_clamp_shexp[layer], stream);
    } else {
      SwigluPlain(sg.F(), su.F(), shared_act.F(),
                  static_cast<std::size_t>(n) * eff, stream);
    }
    Dense(w.shared_down, shared_act.F(), shared_part.F(), h, eff, n, nullptr);

    const int gate_type = TypeCode(w.expert_gate.type);
    const int up_type = TypeCode(w.expert_up.type);
    const int down_type = TypeCode(w.expert_down.type);
    if (gate_type < 0 || up_type < 0 || down_type < 0)
      return Fail(error, "unsupported expert weight type");

    for (std::uint32_t t = 0; t < n; ++t) {
      const std::size_t go = static_cast<std::size_t>(t) * used * eff;
      const std::size_t po = static_cast<std::size_t>(t) * used * h;
      const float* xt = fin + static_cast<std::size_t>(t) * h;
      float* ot = out + static_cast<std::size_t>(t) * h;
      Dense(w.router, xt, router_logits.F(), c.expert_count, h, 1, nullptr);
      RouterLing(router_logits.F(),
                 static_cast<const float*>(w.router_bias.data), ids.I(),
                 weights.F(), c.expert_count, c.expert_groups,
                 c.expert_groups_used, used, c.expert_weights_scale, stream);

      bool quantized =
          MoeVecSupported(gate_type) && MoeVecSupported(up_type) &&
          MoeVecSupported(down_type) &&
          qfn_mmq_moe_vec(gate_type, w.expert_gate.data, xt, ids.I(),
                          gf.F() + go, static_cast<int>(eff),
                          static_cast<int>(h), 1,
                          static_cast<int>(c.expert_count),
                          static_cast<int>(used), stream) == 0;
      quantized =
          quantized &&
          qfn_mmq_moe_vec(up_type, w.expert_up.data, xt, ids.I(), uf.F() + go,
                          static_cast<int>(eff), static_cast<int>(h), 1,
                          static_cast<int>(c.expert_count),
                          static_cast<int>(used), stream) == 0;
      if (quantized) {
        if (c.swiglu_clamp_exp[layer] > 1.0e-6F) {
          SwigluClamp(gf.F() + go, uf.F() + go, act.F() + go, used * eff,
                      c.swiglu_clamp_exp[layer], stream);
        } else {
          SwigluPlain(gf.F() + go, uf.F() + go, act.F() + go, used * eff,
                      stream);
        }
        quantized =
            qfn_mmq_moe_vec(down_type, w.expert_down.data, act.F() + go,
                            ids.I(), part.F() + po, static_cast<int>(h),
                            static_cast<int>(eff), static_cast<int>(used),
                            static_cast<int>(c.expert_count), 1, stream) == 0;
      }
      if (!quantized) {
        const std::uint32_t ge = eff * h;
        const std::uint32_t de = h * eff;
        DequantExperts(w.expert_gate.data, ids.I(), ex_g.F(), used, ge,
                       w.expert_gate.bytes / c.expert_count, gate_type, stream);
        DequantExperts(w.expert_up.data, ids.I(), ex_u.F(), used, ge,
                       w.expert_up.bytes / c.expert_count, up_type, stream);
        DequantExperts(w.expert_down.data, ids.I(), ex_d.F(), used, de,
                       w.expert_down.bytes / c.expert_count, down_type, stream);
        for (std::uint32_t slot = 0; slot < used; ++slot) {
          qfn::SmallGemm(ex_g.F() + static_cast<std::size_t>(slot) * ge,
                         qfn::WeightType::kF32, xt,
                         gf.F() + go + static_cast<std::size_t>(slot) * eff, 1,
                         eff, h, stream);
          qfn::SmallGemm(ex_u.F() + static_cast<std::size_t>(slot) * ge,
                         qfn::WeightType::kF32, xt,
                         uf.F() + go + static_cast<std::size_t>(slot) * eff, 1,
                         eff, h, stream);
        }
        if (c.swiglu_clamp_exp[layer] > 1.0e-6F) {
          SwigluClamp(gf.F() + go, uf.F() + go, act.F() + go, used * eff,
                      c.swiglu_clamp_exp[layer], stream);
        } else {
          SwigluPlain(gf.F() + go, uf.F() + go, act.F() + go, used * eff,
                      stream);
        }
        for (std::uint32_t slot = 0; slot < used; ++slot) {
          qfn::SmallGemm(ex_d.F() + static_cast<std::size_t>(slot) * de,
                         qfn::WeightType::kF32,
                         act.F() + go + static_cast<std::size_t>(slot) * eff,
                         part.F() + po + static_cast<std::size_t>(slot) * h, 1,
                         h, eff, stream);
        }
      }
      MoeCombine(part.F() + po, weights.F(),
                 shared_part.F() + static_cast<std::size_t>(t) * h, ot, used, h,
                 stream);
    }
    return true;
  }

  bool QuantTiled(const float* x, std::uint32_t n, std::uint32_t k) {
    if (k % 32 != 0)
      return false;
    qfn::QuantizeQ8Tiled(x, pf_q8.data, n, k, stream);
    return true;
  }

  // Projection of the activations last handed to QuantTiled.
  void Proj(const DeviceTensor& w, const float* x, float* out,
            std::uint32_t rows, std::uint32_t cols, std::uint32_t n) {
    static const bool kNoW8 = std::getenv("LING_NO_W8A8") != nullptr;
    if (n <= 8) {
      Dense(w, x, out, rows, cols, n, nullptr);
      return;
    }
    if (!kNoW8 && w.type == GgmlType::kQ8_0 &&
        qfn::W8A8Gemm(w.data, pf_q8.data, out, n, rows, cols, stream)) {
      return;
    }
    Dense(w, x, out, rows, cols, n, nullptr);
  }

  bool Sgemm(const void* w, const float* x, float* out, std::uint32_t m,
             std::uint32_t n, std::uint32_t k, std::string* error) {
    static const bool kBlas = std::getenv("LING_HIPBLAS") != nullptr;
    if (!kBlas) {
      SgemmF32(static_cast<const float*>(w), x, out, m, n, k, stream);
      return true;
    }
    const float alpha = 1.0F;
    const float beta0 = 0.0F;
    if (hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(m),
                     static_cast<int>(n), static_cast<int>(k), &alpha,
                     static_cast<const float*>(w), static_cast<int>(k), x,
                     static_cast<int>(k), &beta0, out,
                     static_cast<int>(m)) != HIPBLAS_STATUS_SUCCESS) {
      return Fail(error, "hipblas sgemm failed");
    }
    return true;
  }

  bool MoeSmall(const LayerWeights& w, const float* fin, float* out,
                std::uint32_t layer, std::uint32_t n, std::string* error) {
    const Config& c = config();
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t eff = c.expert_ff;
    const std::uint32_t used = c.expert_used;
    const int gt = TypeCode(w.expert_gate.type);
    const int ut = TypeCode(w.expert_up.type);
    const int dt = TypeCode(w.expert_down.type);
    if (!MoeVecSupported(gt) || !MoeVecSupported(ut) || !MoeVecSupported(dt))
      return Fail(error, "small MoE needs vec-supported expert types");
    Dense(w.router, fin, pf_router.F(), c.expert_count, h, n, nullptr);
    RouterLing(pf_router.F(), static_cast<const float*>(w.router_bias.data),
               ids.I(), weights.F(), c.expert_count, c.expert_groups,
               c.expert_groups_used, used, c.expert_weights_scale, stream, n);
    if (qfn_mmq_moe_vec(gt, w.expert_gate.data, fin, ids.I(), gf.F(),
                        static_cast<int>(eff), static_cast<int>(h),
                        static_cast<int>(n), static_cast<int>(c.expert_count),
                        static_cast<int>(used), stream) != 0 ||
        qfn_mmq_moe_vec(ut, w.expert_up.data, fin, ids.I(), uf.F(),
                        static_cast<int>(eff), static_cast<int>(h),
                        static_cast<int>(n), static_cast<int>(c.expert_count),
                        static_cast<int>(used), stream) != 0) {
      return Fail(error, "small MoE gate/up failed");
    }
    const std::uint32_t count = n * used * eff;
    if (c.swiglu_clamp_exp[layer] > 1.0e-6F)
      SwigluClamp(gf.F(), uf.F(), act.F(), count, c.swiglu_clamp_exp[layer],
                  stream);
    else
      SwigluPlain(gf.F(), uf.F(), act.F(), count, stream);
    if (qfn_mmq_moe_vec(dt, w.expert_down.data, act.F(), ids.I(), part.F(),
                        static_cast<int>(h), static_cast<int>(eff),
                        static_cast<int>(n * used),
                        static_cast<int>(c.expert_count), 1, stream) != 0) {
      return Fail(error, "small MoE down failed");
    }
    Dense(w.shared_gate, fin, sg.F(), eff, h, n, nullptr);
    Dense(w.shared_up, fin, su.F(), eff, h, n, nullptr);
    if (c.swiglu_clamp_shexp[layer] > 1.0e-6F)
      SwigluClamp(sg.F(), su.F(), shared_act.F(), n * eff,
                  c.swiglu_clamp_shexp[layer], stream);
    else
      SwigluPlain(sg.F(), su.F(), shared_act.F(), n * eff, stream);
    Dense(w.shared_down, shared_act.F(), shared_part.F(), h, eff, n, nullptr);
    MoeCombineBatchF32(part.F(), weights.F(), shared_part.F(), out, n, used, h,
                       stream);
    return true;
  }

  std::size_t kPrefillMin() const { return kPrefillTile; }
  bool npu_has_qkv{false};
  bool npu_act_live{false};

  bool NpuAllocActivations(bool qkv, std::string* err) {
    if (!qkv) {
      npu_a = npu->NewBuffer(npu->ABytes(), err);
      for (int i = 0; i < 2; ++i)
        npu_c[i] = npu->NewBuffer(npu->CBytes(), err);
      npu_act_live = npu_a >= 0 && npu_c[0] >= 0 && npu_c[1] >= 0;
      return npu_act_live;
    }
    for (int b = 0; b < 2; ++b) {
      npu_qa[b] = npu->NewBuffer(npu->ABytes(), err);
      if (npu_qa[b] < 0)
        return false;
      npu_qc[b].clear();
      for (int i = 0; i < kQkvSlices; ++i) {
        const int id = npu->NewBuffer(npu->CBytes(), err);
        if (id < 0)
          return false;
        npu_qc[b].push_back(id);
      }
    }
    return true;
  }

  void NpuReleaseActivations() {
    if (npu == nullptr || !npu_act_live)
      return;
    hipStreamSynchronize(stream);
    npu->Wait(stream);
    hipStreamSynchronize(stream);
    npu->ReleaseBuffer(npu_a);
    npu->ReleaseBuffer(npu_c[0]);
    npu->ReleaseBuffer(npu_c[1]);
    npu_a = npu_c[0] = npu_c[1] = -1;
    for (int b = 0; b < 2; ++b) {
      if (npu_qa[b] >= 0)
        npu->ReleaseBuffer(npu_qa[b]);
      npu_qa[b] = -1;
      for (const int id : npu_qc[b])
        npu->ReleaseBuffer(id);
      npu_qc[b].clear();
    }
    npu_act_live = false;
  }

  bool NpuEnsureActivations() {
    if (npu == nullptr || npu_act_live)
      return npu != nullptr;
    std::string err;
    if (!NpuAllocActivations(false, &err) ||
        (npu_has_qkv && !NpuAllocActivations(true, &err))) {
      std::fprintf(stderr, "NPU activation buffers failed: %s\n", err.c_str());
      NpuReleaseActivations();
      npu->MarkFailed();
      return false;
    }
    npu_act_live = true;
    return true;
  }

  bool NpuReady(std::uint32_t layer, std::uint32_t n) const {
    return npu != nullptr && npu_act_live && npu->Healthy() &&
           n == npu->tokens() && layer < npu_sh.size() && npu_sh[layer][0] >= 0;
  }

  void SetupNpu() {
    const char* dir = std::getenv("GUFO_NPU_DIR");
    if (dir == nullptr || dir[0] == '\0')
      return;
    const Config& c = config();
    const std::uint32_t h = c.hidden_size;
    qfn::NpuGemm::Config cfg;
    cfg.xclbin =
        std::string(dir) + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
    cfg.insts = std::string(dir) + "/runtime_mk/insts_" +
                std::to_string(kPrefillTile) + "x" + std::to_string(h) +
                "x1024_128x64x128.txt";
    cfg.tokens = kPrefillTile;
    cfg.k = h;
    std::string err;
    npu = qfn::NpuGemm::Create(cfg, &err);
    if (npu == nullptr) {
      std::fprintf(stderr, "NPU off: %s\n", err.c_str());
      return;
    }
    bool ok = NpuAllocActivations(false, &err);
    const Model::Impl& m = *model->impl_;
    npu_sh.assign(m.n_layers, {-1, -1});
    std::uint32_t staged = 0;
    for (std::uint32_t l = 0; ok && l < m.n_layers; ++l) {
      const LayerWeights& w = m.layers[l];
      if (c.IsDense(l) || w.shared_gate.data == nullptr ||
          w.shared_gate.type != GgmlType::kQ8_0 ||
          w.shared_up.type != GgmlType::kQ8_0 || c.expert_ff > npu->n())
        continue;
      const DeviceTensor* src[2] = {&w.shared_gate, &w.shared_up};
      for (int i = 0; i < 2; ++i) {
        const int id = npu->NewWeightBuffer(npu->BBytes(), &err);
        if (id < 0) {
          ok = false;
          break;
        }
        qfn::NpuEncodeWeights(src[i]->data, h, 0, c.expert_ff, h, 0, h,
                              npu->Device(id), stream);
        npu_sh[l][i] = id;
      }
      ++staged;
    }
    npu_qkv.assign(m.n_layers, {});
    std::uint32_t qkv_staged = 0;
    if (ok && c.KdaDim() == 4U * npu->n()) {
      npu_has_qkv = true;
      ok = NpuAllocActivations(true, &err);
      for (std::uint32_t l = 0; ok && l < m.n_layers; ++l) {
        const LayerWeights& w = m.layers[l];
        if (!c.IsKda(l) || w.attn_q.type != GgmlType::kQ8_0 ||
            w.attn_k.type != GgmlType::kQ8_0 ||
            w.attn_v.type != GgmlType::kQ8_0)
          continue;
        const DeviceTensor* src[3] = {&w.attn_q, &w.attn_k, &w.attn_v};
        for (int t = 0; ok && t < 3; ++t) {
          for (std::uint32_t sl = 0; ok && sl < 4; ++sl) {
            const int id = npu->NewWeightBuffer(npu->BBytes(), &err);
            ok = id >= 0;
            if (!ok)
              break;
            qfn::NpuEncodeWeights(src[t]->data, h, sl * npu->n(), npu->n(), h,
                                  0, h, npu->Device(id), stream);
            npu_qkv[l].push_back(id);
          }
        }
        ++qkv_staged;
      }
    }
    hipStreamSynchronize(stream);
    if (!ok) {
      std::fprintf(stderr, "NPU off: buffers %s\n", err.c_str());
      npu.reset();
      npu_sh.clear();
      return;
    }
    for (const auto& ids : npu_sh)
      for (const int id : ids)
        if (id >= 0)
          npu->FinalizeWeights(id);
    for (const auto& ids : npu_qkv)
      for (const int id : ids)
        npu->FinalizeWeights(id);
    std::fprintf(stderr, "NPU: shared gate/up of %u layers, KDA q/k/v of %u\n",
                 staged, qkv_staged);
  }

  bool MoePrefill(const LayerWeights& w, const float* fin, float* out,
                  std::uint32_t layer, std::uint32_t n, std::string* error) {
    const Config& c = config();
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t eff = c.expert_ff;
    const std::uint32_t used = c.expert_used;
    const std::uint32_t experts = c.expert_count;
    const std::size_t slots = static_cast<std::size_t>(n) * used;

    NarrowRows(fin, static_cast<__half*>(pf_fin16.data),
               static_cast<std::size_t>(n) * h, stream);
    const bool npu_sh_on = NpuReady(layer, n);
    const auto shared_expert = [&]() {
      if (npu_sh_on) {
        npu->Wait(stream, npu_ticket);
        qfn::NpuDecodeOutput(npu->Staged(stream, npu_c[0]), sg.F(), n, npu->n(),
                             eff, 0, false, stream, eff);
        qfn::NpuDecodeOutput(npu->Staged(stream, npu_c[1]), su.F(), n, npu->n(),
                             eff, 0, false, stream, eff);
      } else {
        QuantTiled(fin, n, h);
        Proj(w.shared_gate, fin, sg.F(), eff, h, n);
        Proj(w.shared_up, fin, su.F(), eff, h, n);
      }
      if (c.swiglu_clamp_shexp[layer] > 1.0e-6F) {
        SwigluClamp(sg.F(), su.F(), shared_act.F(),
                    static_cast<std::uint32_t>(n * eff),
                    c.swiglu_clamp_shexp[layer], stream);
      } else {
        SwigluPlain(sg.F(), su.F(), shared_act.F(),
                    static_cast<std::uint32_t>(n * eff), stream);
      }
      QuantTiled(shared_act.F(), n, eff);
      Proj(w.shared_down, shared_act.F(), shared_part.F(), h, eff, n);
    };
    if (npu_sh_on) {
      qfn::NpuEncodeActivations(static_cast<const __half*>(pf_fin16.data), h, n,
                                h, 0, h, npu->Device(npu_a), stream, false);
      npu->Submit(stream, npu_a, npu_sh[layer].data(), npu_c, 2);
      npu_ticket = npu->Ticket();
    } else {
      shared_expert();
    }

    if (w.router3.data != nullptr) {
      SplitHiLo(fin, static_cast<__half*>(pf_x3.data), n, h, false, stream);
      if (!qfn::DenseHalfWeightGemm(static_cast<const __half*>(w.router3.data),
                                    static_cast<const __half*>(pf_x3.data),
                                    pf_router.F(), n, experts, 3U * h, stream))
        return Fail(error, "router GEMM failed");
    } else if (!Sgemm(w.router.data, fin, pf_router.F(), experts, n, h,
                      error)) {
      return false;
    }
    RouterLing(pf_router.F(), static_cast<const float*>(w.router_bias.data),
               ids.I(), weights.F(), experts, c.expert_groups,
               c.expert_groups_used, used, c.expert_weights_scale, stream, n);
    if (ValidateOn()) {
      if (!CheckFloats("router_logits", pf_router.F(),
                       static_cast<std::size_t>(n) * experts, layer,
                       position) ||
          !CheckInts("ids", ids.I(), static_cast<std::size_t>(n) * used, 0,
                     static_cast<std::int32_t>(experts), layer, position)) {
        return Fail(error, "validation failed in router");
      }
    }
    qfn::ExpertCounts(ids.I(), static_cast<std::uint32_t*>(pf_counts.data), n,
                      experts, used, stream);
    const auto bound = [&](std::uint32_t r) {
      return static_cast<std::uint32_t>((slots + 15U * experts) / r + experts);
    };
    const std::uint32_t cap48 = bound(48);
    const std::uint32_t cap64 = bound(64);
    const std::uint32_t cap128 = bound(128);
    static const bool kHostTiles = std::getenv("LING_DEVICE_TILES") == nullptr;
    std::uint32_t n48 = cap48;
    std::uint32_t n64 = cap64;
    std::uint32_t n128 = cap128;
    if (kHostTiles) {
      hipMemcpyAsync(pf_host_counts.data(), pf_counts.data,
                     experts * sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                     stream);
      hipStreamSynchronize(stream);
      const std::size_t third = pf_host_tiles.size() / 3;
      std::int32_t* h48 = pf_host_tiles.data();
      std::int32_t* h64 = h48 + third;
      std::int32_t* h128 = h64 + third;
      n48 = n64 = n128 = 0;
      for (std::uint32_t e = 0; e < experts; ++e) {
        const std::uint32_t padded =
            (pf_host_counts[e] + kRoutePad - 1) / kRoutePad * kRoutePad;
        for (std::uint32_t j = 0; j * 48 < padded; ++j)
          h48[n48++] = static_cast<std::int32_t>(e | (j << 16));
        for (std::uint32_t j = 0; j * 64 < padded; ++j)
          h64[n64++] = static_cast<std::int32_t>(e | (j << 16));
        for (std::uint32_t j = 0; j * 128 < padded; ++j)
          h128[n128++] = static_cast<std::int32_t>(e | (j << 16));
      }
      hipMemcpyAsync(pf_tiles48.data, h48, n48 * 4, hipMemcpyHostToDevice,
                     stream);
      hipMemcpyAsync(pf_tiles64.data, h64, n64 * 4, hipMemcpyHostToDevice,
                     stream);
      hipMemcpyAsync(pf_tiles128.data, h128, n128 * 4, hipMemcpyHostToDevice,
                     stream);
    } else {
      BuildTiles(static_cast<const std::uint32_t*>(pf_counts.data), experts,
                 pf_tiles48.I(), cap48, pf_tiles64.I(), cap64, pf_tiles128.I(),
                 cap128, stream);
    }
    qfn::RoutedCompact(ids.I(),
                       static_cast<const std::uint32_t*>(pf_counts.data),
                       pf_bounds.I(), pf_cursors.I(), pf_rows_token.I(),
                       pf_rows_slot.I(), n, used, experts, stream);

    const auto tiles_for = [&](GgmlType t, std::uint32_t* rows,
                               std::uint32_t* count) -> const std::int32_t* {
      if (t == GgmlType::kQ8_0) {
        *rows = 64;
        *count = static_cast<std::uint32_t>(n64);
        return pf_tiles64.I();
      }
      *rows = 48;
      *count = static_cast<std::uint32_t>(n48);
      return pf_tiles48.I();
    };
    std::uint32_t rows = 0;
    std::uint32_t count = 0;
    const auto* x16 = static_cast<const __half*>(pf_fin16.data);
    const std::int32_t* tl = nullptr;
    const bool fused = w.expert_gate.type == GgmlType::kQ4_K &&
                       w.expert_up.type == GgmlType::kQ4_K &&
                       c.swiglu_clamp_exp[layer] <= 1.0e-6F;
    if (fused) {
      if (!qfn::RoutedGatedF16Gemm(
              w.expert_gate.data, w.expert_up.data, qfn::WeightType::kQ4_K, x16,
              pf_tiles128.I(), static_cast<std::uint32_t>(n128), 128,
              pf_bounds.I(), pf_rows_token.I(), pf_rows_slot.I(),
              static_cast<__half*>(pf_act16.data), eff, h, stream)) {
        return Fail(error, "routed gate/up projection failed");
      }
    } else {
      tl = tiles_for(w.expert_gate.type, &rows, &count);
      if (!qfn::RoutedF16Gemm(
              w.expert_gate.data, WeightTypeOf(w.expert_gate.type), x16, tl,
              count, rows, pf_bounds.I(), pf_rows_token.I(), pf_rows_slot.I(),
              nullptr, pf_gate_e.F(), nullptr, eff, h, stream)) {
        return Fail(error, "routed gate projection failed");
      }
      tl = tiles_for(w.expert_up.type, &rows, &count);
      if (!qfn::RoutedF16Gemm(w.expert_up.data, WeightTypeOf(w.expert_up.type),
                              x16, tl, count, rows, pf_bounds.I(),
                              pf_rows_token.I(), pf_rows_slot.I(), nullptr,
                              pf_up_e.F(), nullptr, eff, h, stream)) {
        return Fail(error, "routed up projection failed");
      }
      SwigluHalf(pf_gate_e.F(), pf_up_e.F(),
                 static_cast<__half*>(pf_act16.data), slots * eff,
                 c.swiglu_clamp_exp[layer], stream);
    }
    tl = tiles_for(w.expert_down.type, &rows, &count);
    if (!qfn::RoutedF16Gemm(
            w.expert_down.data, WeightTypeOf(w.expert_down.type),
            static_cast<const __half*>(pf_act16.data), tl, count, rows,
            pf_bounds.I(), pf_rows_slot.I(), pf_rows_slot.I(), nullptr, nullptr,
            static_cast<__half*>(pf_down_e.data), h, eff, stream)) {
      return Fail(error, "routed down projection failed");
    }
    if (ValidateOn()) {
      if (!CheckHalves("act16", pf_act16.data, slots * eff, layer, position) ||
          !CheckHalves("down_e", pf_down_e.data, slots * h, layer, position)) {
        return Fail(error, "validation failed in experts");
      }
    }
    if (npu_sh_on)
      shared_expert();
    MoeCombineBatch(static_cast<const __half*>(pf_down_e.data), weights.F(),
                    shared_part.F(), out, n, used, h, stream);
    return true;
  }

  bool QkvReady(std::uint32_t layer) const {
    return npu != nullptr && npu_act_live && npu->Healthy() &&
           layer < npu_qkv.size() && !npu_qkv[layer].empty();
  }

  void QkvQueue(int bank, std::uint32_t layer, std::uint32_t n) {
    const Config& c = config();
    const std::uint32_t h = c.hidden_size;
    const LayerWeights& w = model->impl_->layers[layer];
    qfn::RmsNormRows(x.F(), static_cast<const float*>(w.attn_norm.data), xn.F(),
                     n, h, 1, c.rms_eps, stream);
    NarrowRows(xn.F(), static_cast<__half*>(pf_fin16.data),
               static_cast<std::size_t>(n) * h, stream);
    qfn::NpuEncodeActivations(static_cast<const __half*>(pf_fin16.data), h, n,
                              h, 0, h, npu->Device(npu_qa[bank]), stream,
                              false);
    npu->Submit(stream, npu_qa[bank], npu_qkv[layer].data(),
                npu_qc[bank].data(), kQkvSlices);
    npu_qticket[bank] = npu->Ticket();
  }

  void QkvFinish(int bank, std::uint32_t n) {
    const std::uint32_t di = config().KdaDim();
    npu->Wait(stream, npu_qticket[bank]);
    float* out[3] = {q.F(), k.F(), v.F()};
    for (int t = 0; t < 3; ++t)
      for (std::uint32_t sl = 0; sl < 4; ++sl)
        qfn::NpuDecodeOutput(npu->Staged(stream, npu_qc[bank][t * 4 + sl]),
                             out[t], n, npu->n(), di, sl * npu->n(), false,
                             stream, 0);
  }

  bool ForwardPair(const std::int32_t* tokens, std::string* error) {
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t T = npu->tokens();
    const std::uint32_t pos0 = position;
    const auto t0 = std::chrono::steady_clock::now();
    if (hipMemcpyAsync(token.I(), tokens, 2U * T * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream) != hipSuccess)
      return Fail(error, "token upload failed");
    qfn::EmbedTokens(m.token_embd.data, WeightTypeOf(m.token_embd.type),
                     token.I(), x.F(), T, h, 1, stream);
    qfn::EmbedTokens(m.token_embd.data, WeightTypeOf(m.token_embd.type),
                     token.I() + T, x1.F(), T, h, 1, stream);
    bool prepared[2] = {false, false};
    const auto prefetch = [&](int bank, std::uint32_t layer) {
      prepared[bank] = false;
      if (layer >= m.n_layers || !QkvReady(layer))
        return;
      QkvQueue(bank, layer, T);
      prepared[bank] = true;
    };
    prefetch(0, 0);
    std::swap(x, x1);
    prefetch(1, 0);
    std::swap(x, x1);
    for (std::uint32_t l = 0; l < m.n_layers; ++l) {
      for (int bank = 0; bank < 2; ++bank) {
        if (bank == 1)
          std::swap(x, x1);
        const bool ok =
            PrefillLayer(l, T, pos0 + static_cast<std::uint32_t>(bank) * T,
                         false, prepared[bank] ? bank : -1, error);
        if (!ok) {
          if (bank == 1)
            std::swap(x, x1);
          return false;
        }
        prefetch(bank, l + 1);
        if (bank == 1)
          std::swap(x, x1);
      }
    }
    if (!FinishBatch(tokens, T, error))
      return false;
    std::swap(x, x1);
    const bool ok = FinishBatch(tokens + T, T, error);
    std::swap(x, x1);
    if (std::getenv("LING_PF_TIMING") != nullptr) {
      std::fprintf(stderr, "[prefill pair n=%u pos=%u] %.1f ms\n", 2U * T, pos0,
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count());
    }
    return ok && npu->Healthy();
  }

  bool PrefillLayer(std::uint32_t l, std::uint32_t n, std::uint32_t pos0,
                    bool spec, int qkv_bank, std::string* error) {
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t di = c.KdaDim();
    const std::uint32_t nh = c.num_heads;
    const std::uint32_t qk = c.key_length_mla;
    const std::uint32_t kv = c.kv_lora_rank;
    const std::uint32_t vd = c.MlaVDim();
    const std::uint32_t key = kv + c.rope_dimension;
    const std::uint32_t ff = c.dense_ff;
    const LayerWeights& w = m.layers[l];
    qfn::RmsNormRows(x.F(), static_cast<const float*>(w.attn_norm.data), xn.F(),
                     n, h, 1, c.rms_eps, stream);
    PROF_BEGIN();
    if (c.IsKda(l)) {
      QuantTiled(xn.F(), n, h);
      if (qkv_bank >= 0) {
        QkvFinish(qkv_bank, n);
      } else {
        Proj(w.attn_q, xn.F(), q.F(), di, h, n);
        Proj(w.attn_k, xn.F(), k.F(), di, h, n);
        Proj(w.attn_v, xn.F(), v.F(), di, h, n);
      }
      if (spec) {
        const std::size_t row_bytes =
            static_cast<std::size_t>(n) * di * sizeof(float);
        hipMemcpyAsync(layers[l].ck_hist.data, layers[l].conv.data,
                       layers[l].conv.bytes, hipMemcpyDeviceToDevice, stream);
        hipMemcpyAsync(layers[l].ck_raw.F(), q.F(), row_bytes,
                       hipMemcpyDeviceToDevice, stream);
        hipMemcpyAsync(layers[l].ck_raw.F() + 8U * di, k.F(), row_bytes,
                       hipMemcpyDeviceToDevice, stream);
        hipMemcpyAsync(layers[l].ck_raw.F() + 16U * di, v.F(), row_bytes,
                       hipMemcpyDeviceToDevice, stream);
      }
      KdaConvBatch(q.F(), k.F(), v.F(),
                   static_cast<const float*>(w.conv_q.data),
                   static_cast<const float*>(w.conv_k.data),
                   static_cast<const float*>(w.conv_v.data), layers[l].conv.F(),
                   oq.F(), ok.F(), ov.F(), di, n, stream);
      Proj(w.ssm_f_a, xn.F(), fa.F(), di, h, n);
      Proj(w.ssm_g_a, xn.F(), ga.F(), di, h, n);
      if (!Sgemm(w.ssm_beta.data, xn.F(), beta.F(), nh, n, h, error))
        return false;
      KdaRecurBatch(oq.F(), ok.F(), ov.F(), fa.F(), ga.F(), beta.F(),
                    static_cast<const float*>(w.ssm_dt.data),
                    static_cast<const float*>(w.ssm_a.data),
                    static_cast<const float*>(w.ssm_norm.data),
                    layers[l].state.F(), kda_so.F(), kda_out.F(), n, nh,
                    c.gate_lower_bound, c.rms_eps, stream,
                    spec ? layers[l].ck_state.F() : nullptr);
      QuantTiled(kda_out.F(), n, di);
      Proj(w.attn_output, kda_out.F(), block.F(), h, di, n);
      PROF_END(0);
    } else {
      QuantTiled(xn.F(), n, h);
      Proj(w.attn_q, xn.F(), qall.F(), nh * qk, h, n);
      Proj(w.attn_kv_a, xn.F(), kvall.F(), key, h, n);
      if (!Sgemm(w.attn_gate.data, xn.F(), gate.F(), nh, n, h, error))
        return false;
      float* cache = layers[l].cache.F();
      MlaKvPrep(kvall.F(), static_cast<const float*>(w.attn_kv_norm.data),
                cache, pos0, n, c.rms_eps, c.freq_base, stream);
      RopeQBatch(qall.F(), n, nh, qk, c.QkNope(), pos0, c.freq_base, stream);
      UpdatePool(l, pos0, n);
      if (spec && std::getenv("LING_SPEC_PREFILL_MLA") == nullptr) {
        const float scale = 1.0F / std::sqrt(static_cast<float>(qk));
        for (std::uint32_t i = 0; i < n; ++i) {
          const float* qrow = qall.F() + static_cast<std::size_t>(i) * nh * qk;
          HeadGemvQ8(w.attn_k_b.data, qrow, qvec.F(), nh, kv, c.QkNope(), qk,
                     key, stream);
          ProbeRope(qrow, qvec.F(), stream);
          std::uint32_t prefix = 0;
          std::uint32_t total = 0;
          const float* lat = SelectLatents(l, pos0 + i, 1, qvec.F(), false, {},
                                           &prefix, &total);
          SetI32(pos_dev.I(), static_cast<std::int32_t>(pos0 + i), stream);
          const std::int32_t* rows_ptr = pos_dev.I();
          if (lat != layers[l].cache.F()) {
            SetI32(sel_pos.I(), static_cast<std::int32_t>(prefix), stream);
            rows_ptr = sel_pos.I();
          }
          MlaDecode(qvec.F(), lat, pm.F(), pl.F(), pacc.F(), ctxlat.F(),
                    rows_ptr, scale, stream);
          HeadGemvQ8(w.attn_v_b.data, ctxlat.F(),
                     attn.F() + static_cast<std::size_t>(i) * nh * vd, nh, vd,
                     kv, kv, vd, stream);
        }
      } else {
        std::uint32_t prefix = 0;
        std::uint32_t rows = 0;
        const float* lat = SelectLatents(
            l, pos0, n, qvec.F(), true,
            [&]() {
              const float* last =
                  qall.F() + static_cast<std::size_t>(n - 1) * nh * qk;
              HeadGemvQ8(w.attn_k_b.data, last, qvec.F(), nh, kv, c.QkNope(),
                         qk, key, stream);
              ProbeRope(last, qvec.F(), stream);
            },
            &prefix, &rows);
        LatentToHalf(lat, static_cast<__half*>(pf_lat16.data), rows, stream);
        if (!qfn::DenseHalfWeightGemm(static_cast<const __half*>(w.wk16.data),
                                      static_cast<const __half*>(pf_lat16.data),
                                      pf_kn.F(), rows, nh * c.QkNope(), kv,
                                      stream) ||
            !qfn::DenseHalfWeightGemm(static_cast<const __half*>(w.wv16.data),
                                      static_cast<const __half*>(pf_lat16.data),
                                      pf_vv.F(), rows, nh * vd, kv, stream)) {
          return Fail(error, "MLA K/V decompression failed");
        }
        BuildKvHalf(pf_kn.F(), pf_vv.F(), lat,
                    static_cast<__half*>(pf_k16.data),
                    static_cast<__half*>(pf_v16.data), rows, stream);
        NarrowRows(qall.F(), static_cast<__half*>(pf_q16.data),
                   static_cast<std::size_t>(n) * nh * qk, stream);
        MlaWmmaAttention(static_cast<const __half*>(pf_q16.data),
                         static_cast<const __half*>(pf_k16.data),
                         static_cast<const __half*>(pf_v16.data), attn.F(), n,
                         prefix, 1.0F / std::sqrt(static_cast<float>(qk)),
                         stream);
        if (ValidateOn() &&
            (!CheckFloats("attn_out", attn.F(),
                          static_cast<std::size_t>(n) * nh * vd, l, pos0) ||
             !CheckHalves("k16", pf_k16.data,
                          static_cast<std::size_t>(rows) * 32 * 192, l,
                          pos0))) {
          return Fail(error, "validation failed in attention");
        }
      }
      MlaGate(attn.F(), gate.F(), attn.F(), n * nh, vd, stream);
      QuantTiled(attn.F(), n, nh * vd);
      Proj(w.attn_output, attn.F(), block.F(), h, nh * vd, n);
      PROF_END(1);
    }
    AddTo(x.F(), block.F(), static_cast<std::size_t>(n) * h, stream);
    qfn::RmsNormRows(x.F(), static_cast<const float*>(w.ffn_norm.data), fin.F(),
                     n, h, 1, c.rms_eps, stream);
    PROF_BEGIN();
    if (c.IsDense(l)) {
      QuantTiled(fin.F(), n, h);
      Proj(w.ffn_gate, fin.F(), dense_gate.F(), ff, h, n);
      Proj(w.ffn_up, fin.F(), dense_up.F(), ff, h, n);
      SwigluPlain(dense_gate.F(), dense_up.F(), dense_act.F(),
                  static_cast<std::size_t>(n) * ff, stream);
      QuantTiled(dense_act.F(), n, ff);
      Proj(w.ffn_down, dense_act.F(), ffn.F(), h, ff, n);
    } else if (!(spec ? MoeSmall(w, fin.F(), ffn.F(), l, n, error)
                      : MoePrefill(w, fin.F(), ffn.F(), l, n, error))) {
      return false;
    }
    PROF_END(2);
    AddTo(x.F(), ffn.F(), static_cast<std::size_t>(n) * h, stream);
    DumpLayer(l, n);
    if (ValidateOn() &&
        !CheckFloats("x", x.F(), static_cast<std::size_t>(n) * h, l, pos0)) {
      return Fail(error, "validation failed after layer");
    }
    return true;
  }

  bool ForwardPrefill(const std::int32_t* tokens, std::uint32_t n,
                      std::string* error, bool spec = false) {
    if (n == 0 || n > kPrefillTile || (spec && n > 8))
      return Fail(error, "invalid prefill tile size");
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t di = c.KdaDim();
    const std::uint32_t nh = c.num_heads;
    const std::uint32_t qk = c.key_length_mla;
    const std::uint32_t kv = c.kv_lora_rank;
    const std::uint32_t vd = c.MlaVDim();
    const std::uint32_t key = kv + c.rope_dimension;
    const std::uint32_t ff = c.dense_ff;
    const std::uint32_t pos0 = position;
    const auto pf_t0 = std::chrono::steady_clock::now();

    if (hipMemcpyAsync(token.I(), tokens,
                       static_cast<std::size_t>(n) * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream) != hipSuccess) {
      return Fail(error, "token upload failed");
    }
    qfn::EmbedTokens(m.token_embd.data, WeightTypeOf(m.token_embd.type),
                     token.I(), x.F(), n, h, 1, stream);
    for (std::uint32_t l = 0; l < m.n_layers; ++l) {
      if (!PrefillLayer(l, n, pos0, spec, -1, error))
        return false;
    }
    if (kProf)
      std::fprintf(stderr, "[pf prof ms] kda=%.0f mla=%.0f ffn=%.0f\n",
                   g_prof[0], g_prof[1], g_prof[2]);
    const bool ok =
        spec ? FinishSpec(tokens, n, error) : FinishBatch(tokens, n, error);
    if (std::getenv("LING_GUARD") != nullptr) {
      hipStreamSynchronize(stream);
      DevBuf::CheckGuards("prefill chunk");
    }
    if (std::getenv("LING_PF_TIMING") != nullptr) {
      std::fprintf(stderr, "[prefill n=%u pos=%u] %.1f ms\n", n, pos0,
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - pf_t0)
                       .count());
    }
    return ok;
  }

  void DumpLayer(std::uint32_t l, std::uint32_t n) {
    static const bool kDump = std::getenv("LING_DEBUG_LAYERS") != nullptr;
    if (!kDump)
      return;
    hipStreamSynchronize(stream);
    const std::uint32_t h = config().hidden_size;
    std::vector<float> row(h);
    hipMemcpy(row.data(), x.F() + static_cast<std::size_t>(n - 1) * h,
              h * sizeof(float), hipMemcpyDeviceToHost);
    double sum = 0.0;
    for (float v : row)
      sum += static_cast<double>(v) * v;
    std::fprintf(stderr, "layer %u last-row norm %.5f x0=%.5f x1=%.5f\n", l,
                 std::sqrt(sum), row[0], row[1]);
  }

  bool ForwardBatch(const std::int32_t* tokens, std::uint32_t n,
                    std::string* error) {
    if (n == 0 || n > kBatchTokens)
      return Fail(error, "invalid prefill batch size");
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t di = c.KdaDim();
    const std::uint32_t nh = c.num_heads;
    const std::uint32_t qk = c.key_length_mla;
    const std::uint32_t rope_dim = c.rope_dimension;
    const std::uint32_t nope = c.QkNope();
    const std::uint32_t kv = c.kv_lora_rank;
    const std::uint32_t vd = c.MlaVDim();
    const std::uint32_t key = kv + rope_dim;
    const std::uint32_t ff = c.dense_ff;

    if (hipMemcpyAsync(token.I(), tokens,
                       static_cast<std::size_t>(n) * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream) != hipSuccess) {
      return Fail(error, "token upload failed");
    }
    qfn::EmbedTokens(m.token_embd.data, WeightTypeOf(m.token_embd.type),
                     token.I(), x.F(), n, h, 1, stream);

    for (std::uint32_t l = 0; l < m.n_layers; ++l) {
      const LayerWeights& w = m.layers[l];
      qfn::RmsNormRows(x.F(), static_cast<const float*>(w.attn_norm.data),
                       xn.F(), n, h, 1, c.rms_eps, stream);
      if (c.IsKda(l)) {
        Dense(w.attn_q, xn.F(), q.F(), di, h, n, nullptr);
        Dense(w.attn_k, xn.F(), k.F(), di, h, n, nullptr);
        Dense(w.attn_v, xn.F(), v.F(), di, h, n, nullptr);
        for (std::uint32_t t = 0; t < n; ++t) {
          KdaConv(q.F() + t * di, k.F() + t * di, v.F() + t * di,
                  static_cast<const float*>(w.conv_q.data),
                  static_cast<const float*>(w.conv_k.data),
                  static_cast<const float*>(w.conv_v.data), layers[l].conv.F(),
                  oq.F() + t * di, ok.F() + t * di, ov.F() + t * di, di,
                  c.ssm_conv_kernel, stream);
        }
        Dense(w.ssm_f_a, xn.F(), fa.F(), di, h, n, nullptr);
        Dense(w.ssm_g_a, xn.F(), ga.F(), di, h, n, nullptr);
        Dense(w.ssm_beta, xn.F(), beta.F(), nh, h, n, nullptr);
        for (std::uint32_t t = 0; t < n; ++t) {
          KdaStep(oq.F() + t * di, ok.F() + t * di, ov.F() + t * di,
                  fa.F() + t * di, ga.F() + t * di, beta.F() + t * nh,
                  static_cast<const float*>(w.ssm_dt.data),
                  static_cast<const float*>(w.ssm_a.data),
                  static_cast<const float*>(w.ssm_norm.data),
                  layers[l].state.F(), kda_out.F() + t * di, c.num_heads,
                  c.kda_head_dim, c.gate_lower_bound, c.rms_eps, stream);
        }
        Dense(w.attn_output, kda_out.F(), block.F(), h, di, n, nullptr);
      } else {
        Dense(w.attn_q, xn.F(), qall.F(), nh * qk, h, n, nullptr);
        Dense(w.attn_kv_a, xn.F(), kvall.F(), key, h, n, nullptr);
        const float scale = 1.0F / std::sqrt(static_cast<float>(qk));
        for (std::uint32_t t = 0; t < n; ++t) {
          const std::uint32_t pos = position + t;
          qfn::RmsNormRows(kvall.F() + t * key,
                           static_cast<const float*>(w.attn_kv_norm.data),
                           kvall.F() + t * key, 1, kv, 1, c.rms_eps, stream);
          RopeHeads(kvall.F() + t * key + kv, kpe.F(), 1, rope_dim, 0, rope_dim,
                    pos, c.freq_base, stream);
          RopeHeads(qall.F() + static_cast<std::size_t>(t) * nh * qk, qrope.F(),
                    nh, qk, nope, rope_dim, pos, c.freq_base, stream);
          CopyRow(kvall.F() + t * key, keyrow.F(), kv, stream);
          CopyRow(kpe.F(), keyrow.F() + kv, rope_dim, stream);
          AppendRow(keyrow.F(), layers[l].cache.F(), pos, key, stream);
          const float* cache = layers[l].cache.F();
          for (std::uint32_t head = 0; head < nh; ++head) {
            const auto* kb =
                static_cast<const std::uint8_t*>(w.attn_k_b.data) +
                static_cast<std::size_t>(head) * w.attn_k_b.bytes / nh;
            Dense({const_cast<std::uint8_t*>(kb), 0, w.attn_k_b.type, kv, nope,
                   1},
                  qall.F() + static_cast<std::size_t>(t) * nh * qk + head * qk,
                  qabs.F(), kv, nope, 1, nullptr);
            CopyRow(qabs.F(), qvec.F(), kv, stream);
            CopyRow(qrope.F() + head * rope_dim, qvec.F() + kv, rope_dim,
                    stream);
            MlaScores(qvec.F(), cache,
                      scores.F() + static_cast<std::size_t>(head) * max_context,
                      1, pos + 1, key, scale, stream);
            MlaGather(scores.F() + static_cast<std::size_t>(head) * max_context,
                      cache, ctxlat.F() + static_cast<std::size_t>(head) * kv,
                      1, pos + 1, key, kv, stream);
            const auto* vb =
                static_cast<const std::uint8_t*>(w.attn_v_b.data) +
                static_cast<std::size_t>(head) * w.attn_v_b.bytes / nh;
            Dense(
                {const_cast<std::uint8_t*>(vb), 0, w.attn_v_b.type, vd, kv, 1},
                ctxlat.F() + static_cast<std::size_t>(head) * kv,
                attn.F() + (static_cast<std::size_t>(t) * nh + head) * vd, vd,
                kv, 1, nullptr);
          }
        }
        Dense(w.attn_gate, xn.F(), gate.F(), nh, h, n, nullptr);
        for (std::uint32_t t = 0; t < n; ++t) {
          MlaGate(attn.F() + static_cast<std::size_t>(t) * nh * vd,
                  gate.F() + static_cast<std::size_t>(t) * nh,
                  attn.F() + static_cast<std::size_t>(t) * nh * vd, nh, vd,
                  stream);
        }
        Dense(w.attn_output, attn.F(), block.F(), h, nh * vd, n, nullptr);
      }
      AddTo(x.F(), block.F(), static_cast<std::size_t>(n) * h, stream);
      qfn::RmsNormRows(x.F(), static_cast<const float*>(w.ffn_norm.data),
                       fin.F(), n, h, 1, c.rms_eps, stream);
      if (c.IsDense(l)) {
        Dense(w.ffn_gate, fin.F(), dense_gate.F(), ff, h, n, nullptr);
        Dense(w.ffn_up, fin.F(), dense_up.F(), ff, h, n, nullptr);
        SwigluPlain(dense_gate.F(), dense_up.F(), dense_act.F(),
                    static_cast<std::size_t>(n) * ff, stream);
        Dense(w.ffn_down, dense_act.F(), ffn.F(), h, ff, n, nullptr);
      } else if (!MoeFfnBatch(w, fin.F(), ffn.F(), l, n, error)) {
        return false;
      }
      AddTo(x.F(), ffn.F(), static_cast<std::size_t>(n) * h, stream);
      DumpLayer(l, n);
    }

    return FinishBatch(tokens, n, error);
  }

  bool FinishSpec(const std::int32_t* tokens, std::uint32_t n,
                  std::string* error) {
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    const std::uint32_t h = c.hidden_size;
    qfn::RmsNormRows(x.F(), static_cast<const float*>(m.output_norm.data),
                     xn.F(), n, h, 1, c.rms_eps, stream);
    hipMemcpyAsync(spec_h.F(), xn.F(),
                   static_cast<std::size_t>(n) * h * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    if (!Dense(m.output, xn.F(), spec_logits_dev.F(), c.vocab_size, h, n,
               error))
      return false;
    spec_logits_host.resize(static_cast<std::size_t>(n) * c.vocab_size);
    if (hipMemcpyAsync(spec_logits_host.data(), spec_logits_dev.F(),
                       spec_logits_host.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
      return Fail(error, "verify logits download failed");
    }
    spec_pos0 = position;
    spec_n = n;
    spec_tokens.assign(tokens, tokens + n);
    return true;
  }

  bool CommitSpec(std::uint32_t keep, std::string* error) {
    const Config& c = config();
    const std::uint32_t h = c.hidden_size;
    const std::uint32_t di = c.KdaDim();
    if (spec_n == 0 || keep >= spec_n)
      return Fail(error, "invalid commit");
    if (keep + 1 < spec_n) {
      const std::size_t state_floats = static_cast<std::size_t>(c.num_heads) *
                                       c.kda_head_dim * c.kda_head_dim;
      for (std::uint32_t l = 0; l < config().TrunkLayers(); ++l) {
        if (!c.IsKda(l))
          continue;
        hipMemcpyAsync(
            layers[l].state.F(), layers[l].ck_state.F() + keep * state_floats,
            state_floats * sizeof(float), hipMemcpyDeviceToDevice, stream);
        KdaHistRestore(layers[l].ck_hist.F(), layers[l].ck_raw.F(),
                       layers[l].conv.F(), di, keep, stream);
      }
    }
    position = spec_pos0 + keep + 1;
    std::memcpy(
        logits.data(),
        spec_logits_host.data() + static_cast<std::size_t>(keep) * c.vocab_size,
        c.vocab_size * sizeof(float));
    const std::size_t hb = h * sizeof(float);
    hipMemcpyAsync(mtp_h.F(), spec_h.F() + static_cast<std::size_t>(keep) * h,
                   hb, hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(mtp_prev_h.F(), mtp_h.F(), hb, hipMemcpyDeviceToDevice,
                   stream);
    if (keep >= 1) {
      hipMemcpyAsync(mtp_pending_h.F(),
                     spec_h.F() + static_cast<std::size_t>(keep - 1) * h, hb,
                     hipMemcpyDeviceToDevice, stream);
    } else {
      hipMemcpyAsync(mtp_pending_h.F(), sv_prev.F(), hb,
                     hipMemcpyDeviceToDevice, stream);
    }
    for (std::uint32_t j = 1; j <= keep; ++j) {
      if (!MtpStep(spec_tokens[j],
                   spec_h.F() + static_cast<std::size_t>(j - 1) * h,
                   spec_pos0 + j, spec_pos0 + j, spec_pos0 + j + 1, false,
                   false, error)) {
        return false;
      }
    }
    mtp_prev_token = spec_tokens[keep];
    mtp_has_prev = true;
    spec_n = 0;
    return hipStreamSynchronize(stream) == hipSuccess;
  }

  bool FinishBatch(const std::int32_t* tokens, std::uint32_t n,
                   std::string* error) {
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    const std::uint32_t h = c.hidden_size;
    qfn::RmsNormRows(x.F(), static_cast<const float*>(m.output_norm.data),
                     xn.F(), n, h, 1, c.rms_eps, stream);
    std::uint32_t bf_count = 0;
    if (c.HasMtp() && mtp_enabled && n >= 3) {
      bf_count = std::min<std::uint32_t>(kMtpBackfill, n - 2);
      hipMemcpyAsync(mtp_bf.F(),
                     xn.F() + static_cast<std::size_t>(n - 2 - bf_count) * h,
                     static_cast<std::size_t>(bf_count) * h * sizeof(float),
                     hipMemcpyDeviceToDevice, stream);
    }
    if (c.HasMtp()) {
      if (n >= 2) {
        hipMemcpyAsync(mtp_pending_h.F(),
                       xn.F() + static_cast<std::size_t>(n - 2) * h,
                       h * sizeof(float), hipMemcpyDeviceToDevice, stream);
      }
      hipMemcpyAsync(mtp_h.F(), xn.F() + static_cast<std::size_t>(n - 1) * h,
                     h * sizeof(float), hipMemcpyDeviceToDevice, stream);
      hipMemcpyAsync(mtp_prev_h.F(), mtp_h.F(), h * sizeof(float),
                     hipMemcpyDeviceToDevice, stream);
    }
    if (!Dense(m.output, xn.F() + static_cast<std::size_t>(n - 1) * h,
               logits_dev.F(), c.vocab_size, h, 1, error)) {
      return false;
    }
    if (hipMemcpyAsync(logits.data(), logits_dev.F(),
                       logits.size() * sizeof(float), hipMemcpyDeviceToHost,
                       stream) != hipSuccess) {
      return Fail(error, "logit download failed");
    }
    if (hipStreamSynchronize(stream) != hipSuccess)
      return Fail(error, "stream synchronization failed");
    if (std::getenv("LING_DEBUG_LAYERS") != nullptr) {
      std::vector<std::pair<float, int>> top;
      for (std::size_t i = 0; i < logits.size(); ++i)
        top.emplace_back(logits[i], static_cast<int>(i));
      std::partial_sort(
          top.begin(), top.begin() + 5, top.end(),
          [](const auto& a, const auto& b) { return a.first > b.first; });
      std::fprintf(stderr, "top:");
      for (int i = 0; i < 5; ++i)
        std::fprintf(stderr, " %d(%.3f)", top[i].second, top[i].first);
      std::fprintf(stderr, "\n");
    }
    for (std::uint32_t j = 0; j < bf_count; ++j) {
      const std::uint32_t idx = n - 1 - bf_count + j;
      if (!MtpStep(tokens[idx], mtp_bf.F() + static_cast<std::size_t>(j) * h,
                   position + idx, position + idx, position + idx + 1, false,
                   false, error)) {
        return false;
      }
    }
    if (c.HasMtp()) {
      mtp_prev_token = tokens[n - 1];
      mtp_has_prev = true;
    }
    position += n;
    return true;
  }

  bool MtpStep(std::int32_t token_id, const float* h_prev,
               std::uint32_t rope_pos, std::uint32_t cache_index,
               std::uint32_t context_rows, bool update_h, bool emit_logits,
               std::string* error) {
    const Config& c = config();
    const Model::Impl& m = *model->impl_;
    const LayerWeights& w = m.mtp;
    const std::uint32_t layer = c.MtpLayer();
    const std::uint32_t h = c.hidden_size;

    SetI32(token.I(), token_id, stream);
    qfn::EmbedTokens(m.token_embd.data, WeightTypeOf(m.token_embd.type),
                     token.I(), mtp_embd.F(), 1, h, 1, stream);
    qfn::RmsNormRows(mtp_embd.F(), static_cast<const float*>(w.enorm.data),
                     mtp_e_norm.F(), 1, h, 1, c.rms_eps, stream);
    qfn::RmsNormRows(h_prev, static_cast<const float*>(w.hnorm.data),
                     mtp_h_norm.F(), 1, h, 1, c.rms_eps, stream);
    hipMemcpyAsync(mtp_eh.F(), mtp_e_norm.F(), h * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(mtp_eh.F() + h, mtp_h_norm.F(), h * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    if (!Dense(w.eh_proj, mtp_eh.F(), mtp_cur.F(), h, 2U * h, 1, error)) {
      return false;
    }

    qfn::RmsNormRows(mtp_cur.F(), static_cast<const float*>(w.attn_norm.data),
                     xn.F(), 1, h, 1, c.rms_eps, stream);
    SetI32(pos_mtp.I(), static_cast<std::int32_t>(rope_pos), stream);
    MlaBlock(w, layer, xn.F(), block.F(), pos_mtp.I(), rope_pos);
    AddTo(mtp_cur.F(), block.F(), h, stream);

    qfn::RmsNormRows(mtp_cur.F(), static_cast<const float*>(w.ffn_norm.data),
                     fin.F(), 1, h, 1, c.rms_eps, stream);
    if (!MoeFfn(w, fin.F(), ffn.F(), layer, error)) {
      return false;
    }
    AddTo(mtp_cur.F(), ffn.F(), h, stream);

    qfn::RmsNormRows(mtp_cur.F(),
                     static_cast<const float*>(w.layer_out_norm.data), xn.F(),
                     1, h, 1, c.rms_eps, stream);
    if (update_h) {
      hipMemcpyAsync(mtp_h.F(), xn.F(), h * sizeof(float),
                     hipMemcpyDeviceToDevice, stream);
    }
    if (!emit_logits) {
      return hipStreamSynchronize(stream) == hipSuccess ||
             Fail(error, "MTP stream synchronization failed");
    }
    if (!Dense(m.output, xn.F(), mtp_logits_dev.F(), c.vocab_size, h, 1,
               error)) {
      return false;
    }
    if (hipMemcpyAsync(mtp_logits.data(), mtp_logits_dev.F(),
                       mtp_logits.size() * sizeof(float), hipMemcpyDeviceToHost,
                       stream) != hipSuccess) {
      return Fail(error, "MTP logit download failed");
    }
    return hipStreamSynchronize(stream) == hipSuccess ||
           Fail(error, "MTP stream synchronization failed");
  }

  bool MtpAppend(std::int32_t token_id, std::string* error) {
    if (!MtpStep(token_id, mtp_draft_h.F(), mtp_position, mtp_position,
                 mtp_position + 1, true, true, error)) {
      return false;
    }
    hipMemcpyAsync(mtp_draft_h.F(), mtp_h.F(),
                   config().hidden_size * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    ++mtp_position;
    return true;
  }
};

Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

bool Session::Append(std::span<const std::int32_t> tokens,
                     std::string* error_msg) {
  if (impl_->position + tokens.size() > impl_->max_context) {
    return Fail(error_msg, "context length exceeded");
  }
  std::size_t at = 0;
  while (at < tokens.size()) {
    const std::size_t remaining = tokens.size() - at;
    static const bool kOldPrefill = std::getenv("LING_OLD_PREFILL") != nullptr;
    if (remaining >= 2 && !kOldPrefill) {
      if (remaining >= impl_->kPrefillMin())
        impl_->NpuEnsureActivations();
      if (impl_->npu != nullptr && impl_->npu->Healthy() &&
          !impl_->npu_qkv.empty() && remaining >= 2U * impl_->npu->tokens()) {
        if (!impl_->ForwardPair(tokens.data() + at, error_msg))
          return false;
        at += 2U * impl_->npu->tokens();
        continue;
      }
      const std::size_t tile = std::min<std::size_t>(kPrefillTile, remaining);
      if (!impl_->ForwardPrefill(tokens.data() + at,
                                 static_cast<std::uint32_t>(tile), error_msg)) {
        return false;
      }
      at += tile;
      continue;
    }
    const std::size_t count =
        std::min<std::size_t>(kBatchTokens, tokens.size() - at);
    if (count == 1) {
      impl_->NpuReleaseActivations();
      if (!impl_->Forward(tokens[at], error_msg))
        return false;
    } else if (!impl_->ForwardBatch(tokens.data() + at,
                                    static_cast<std::uint32_t>(count),
                                    error_msg)) {
      return false;
    }
    at += count;
  }
  return true;
}

bool Session::MtpAppend(std::int32_t token, std::string* error_msg) {
  impl_->mtp_enabled = true;
  return impl_->MtpAppend(token, error_msg);
}

bool Session::Verify(std::span<const std::int32_t> tokens,
                     std::vector<float>* logits, std::string* error_msg) {
  if (tokens.empty() || tokens.size() > 8)
    return Fail(error_msg, "verify width must be 1..8");
  if (impl_->position + tokens.size() > impl_->max_context)
    return Fail(error_msg, "context length exceeded");
  if (!impl_->ForwardPrefill(tokens.data(),
                             static_cast<std::uint32_t>(tokens.size()),
                             error_msg, true)) {
    return false;
  }
  *logits = impl_->spec_logits_host;
  return true;
}

bool Session::Commit(std::uint32_t keep, std::string* error_msg) {
  return impl_->CommitSpec(keep, error_msg);
}

void Session::BeginDraft() {
  const std::size_t hb = impl_->config().hidden_size * sizeof(float);
  hipMemcpyAsync(impl_->sv_prev.F(), impl_->mtp_h.F(), hb,
                 hipMemcpyDeviceToDevice, impl_->stream);
  hipMemcpyAsync(impl_->mtp_draft_h.F(), impl_->mtp_h.F(), hb,
                 hipMemcpyDeviceToDevice, impl_->stream);
  impl_->mtp_position = impl_->position;
}

void Session::EnableMtp() {
  impl_->mtp_enabled = true;
}

void Session::ResetMtp() {
  hipMemcpyAsync(impl_->mtp_draft_h.F(), impl_->mtp_pending_h.F(),
                 impl_->config().hidden_size * sizeof(float),
                 hipMemcpyDeviceToDevice, impl_->stream);
  impl_->mtp_position = impl_->position == 0 ? 0 : impl_->position - 1;
}

const std::vector<float>& Session::Logits() const noexcept {
  return impl_->logits;
}

const std::vector<float>& Session::MtpLogits() const noexcept {
  return impl_->mtp_logits;
}

std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

void Session::Reset() {
  for (auto& layer : impl_->layers) {
    if (layer.conv.bytes != 0)
      hipMemset(layer.conv.data, 0, layer.conv.bytes);
    if (layer.state.bytes != 0)
      hipMemset(layer.state.data, 0, layer.state.bytes);
    if (layer.cache.bytes != 0)
      hipMemset(layer.cache.data, 0, layer.cache.bytes);
  }
  impl_->position = 0;
  impl_->mtp_position = 0;
  impl_->mtp_prev_token = -1;
  impl_->mtp_has_prev = false;
  hipMemset(impl_->mtp_h.data, 0, impl_->mtp_h.bytes);
  hipMemset(impl_->mtp_prev_h.data, 0, impl_->mtp_prev_h.bytes);
  hipMemset(impl_->mtp_pending_h.data, 0, impl_->mtp_pending_h.bytes);
  hipMemset(impl_->mtp_draft_h.data, 0, impl_->mtp_draft_h.bytes);
}

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& path,
                                   std::string* error_msg) {
  auto model = std::shared_ptr<Model>(new Model());
  auto& impl = *(model->impl_ = std::make_unique<Impl>());
  impl.reader = GgufReader::OpenFile(path, error_msg);
  if (!impl.reader)
    return nullptr;
  auto config = ParseConfig(*impl.reader, error_msg);
  if (!config)
    return nullptr;
  impl.config = *config;
  impl.n_layers = impl.config.TrunkLayers();
  if (hipSetDevice(0) != hipSuccess) {
    Fail(error_msg, "hipSetDevice failed");
    return nullptr;
  }
  if (hipStreamCreate(&impl.stream) != hipSuccess) {
    Fail(error_msg, "hipStreamCreate failed");
    return nullptr;
  }
  if (qfn_mmq_init(0) != 0) {
    Fail(error_msg, "qfn_mmq_init failed");
    return nullptr;
  }

  gufo::hip::TensorUpload uploader(*impl.reader);
  for (const auto& tensor : impl.reader->GetTensors()) {
    const std::string name(tensor.name);
    void* device = nullptr;
    if (hipMalloc(&device, tensor.size_bytes) != hipSuccess) {
      Fail(error_msg, "hipMalloc failed for " + name);
      return nullptr;
    }
    if (!uploader.Upload(tensor, device, error_msg)) {
      hipFree(device);
      return nullptr;
    }
    impl.tensors.emplace(name, MakeTensor(tensor, device));
    impl.resident += tensor.size_bytes;
  }

  for (auto& entry : impl.tensors) {
    const std::string& name = entry.first;
    DeviceTensor& t = entry.second;
    const std::string mtp_prefix =
        "blk." + std::to_string(impl.config.MtpLayer()) + ".";
    if (t.type == GgmlType::kQ8_0 && name.rfind(mtp_prefix, 0) != 0)
      continue;
    if (t.type == GgmlType::kQ4_0 && name.rfind(mtp_prefix, 0) == 0) {
      const std::size_t blocks =
          static_cast<std::size_t>(t.rows) * t.cols * t.experts / 32;
      void* buf = nullptr;
      if (hipMalloc(&buf, blocks * 34) != hipSuccess) {
        Fail(error_msg, "hipMalloc failed for " + name);
        return nullptr;
      }
      EncodeQ8(t.data, buf, blocks, 2, impl.stream);
      if (hipStreamSynchronize(impl.stream) != hipSuccess) {
        Fail(error_msg, "Q8_0 re-encode failed for " + name);
        return nullptr;
      }
      hipFree(t.data);
      t.data = buf;
      t.bytes = blocks * 34;
      t.type = GgmlType::kQ8_0;
      continue;
    }
    if (name.find("_exps.") != std::string::npos || t.type == GgmlType::kF32)
      continue;
    const int type = TypeCode(t.type);
    if (type < 0)
      continue;
    const std::size_t elements =
        static_cast<std::size_t>(t.rows) * t.cols * t.experts;
    const std::size_t old_bytes = t.bytes;
    const std::size_t bytes = elements * sizeof(float);
    void* f32 = nullptr;
    if (hipMalloc(&f32, bytes) != hipSuccess) {
      Fail(error_msg, "hipMalloc failed for dequantized " + name);
      return nullptr;
    }
    DequantRows(t.data, static_cast<float*>(f32),
                static_cast<std::uint32_t>(elements), type, impl.stream);
    if (hipStreamSynchronize(impl.stream) != hipSuccess) {
      Fail(error_msg, "dequantization failed for " + name);
      return nullptr;
    }
    hipFree(t.data);
    t.data = f32;
    t.bytes = bytes;
    t.type = GgmlType::kF32;
    impl.resident += bytes - old_bytes;
  }

  const auto get = [&](const std::string& name) -> DeviceTensor& {
    auto it = impl.tensors.find(name);
    if (it == impl.tensors.end())
      std::abort();
    return it->second;
  };
  impl.token_embd = get("token_embd.weight");
  impl.output_norm = get("output_norm.weight");
  impl.output = get("output.weight");
  impl.layers.resize(impl.n_layers);
  for (std::uint32_t l = 0; l < impl.n_layers; ++l) {
    auto& w = impl.layers[l];
    w.attn_norm = get(P(l, "attn_norm.weight"));
    w.ffn_norm = get(P(l, "ffn_norm.weight"));
    w.attn_output = get(P(l, "attn_output.weight"));
    if (impl.config.IsKda(l)) {
      w.attn_q = get(P(l, "attn_q.weight"));
      w.attn_k = get(P(l, "attn_k.weight"));
      w.attn_v = get(P(l, "attn_v.weight"));
      w.conv_q = get(P(l, "ssm_conv1d_q.weight"));
      w.conv_k = get(P(l, "ssm_conv1d_k.weight"));
      w.conv_v = get(P(l, "ssm_conv1d_v.weight"));
      w.ssm_f_a = get(P(l, "ssm_f_a.weight"));
      w.ssm_g_a = get(P(l, "ssm_g_a.weight"));
      w.ssm_beta = get(P(l, "ssm_beta.weight"));
      w.ssm_dt = get(P(l, "ssm_dt.bias"));
      w.ssm_a = get(P(l, "ssm_a"));
      w.ssm_norm = get(P(l, "ssm_norm.weight"));
    } else {
      w.attn_q = get(P(l, "attn_q.weight"));
      w.attn_kv_a = get(P(l, "attn_kv_a_mqa.weight"));
      w.attn_kv_norm = get(P(l, "attn_kv_a_norm.weight"));
      w.attn_k_b = get(P(l, "attn_k_b.weight"));
      w.attn_v_b = get(P(l, "attn_v_b.weight"));
      w.attn_gate = get(P(l, "attn_gate.weight"));
    }
    if (impl.config.IsDense(l)) {
      w.ffn_gate = get(P(l, "ffn_gate.weight"));
      w.ffn_up = get(P(l, "ffn_up.weight"));
      w.ffn_down = get(P(l, "ffn_down.weight"));
    } else {
      w.router = get(P(l, "ffn_gate_inp.weight"));
      w.router_bias = get(P(l, "exp_probs_b.bias"));
      w.expert_gate = get(P(l, "ffn_gate_exps.weight"));
      w.expert_up = get(P(l, "ffn_up_exps.weight"));
      w.expert_down = get(P(l, "ffn_down_exps.weight"));
      w.shared_gate = get(P(l, "ffn_gate_shexp.weight"));
      w.shared_up = get(P(l, "ffn_up_shexp.weight"));
      w.shared_down = get(P(l, "ffn_down_shexp.weight"));
    }
  }
  const auto to_q8 = [&](DeviceTensor& t) -> bool {
    if (t.type != GgmlType::kQ6_K)
      return true;
    const std::size_t elements =
        static_cast<std::size_t>(t.rows) * t.cols * t.experts;
    const std::size_t blocks = elements / 32;
    void* buf = nullptr;
    if (hipMalloc(&buf, blocks * 34) != hipSuccess)
      return Fail(error_msg, "hipMalloc failed for Q8_0 expert re-encode");
    EncodeQ8(t.data, buf, blocks, 14, impl.stream);
    if (hipStreamSynchronize(impl.stream) != hipSuccess)
      return Fail(error_msg, "Q8_0 expert re-encode failed");
    hipFree(t.data);
    impl.resident += blocks * 34 - t.bytes;
    t.data = buf;
    t.bytes = blocks * 34;
    t.type = GgmlType::kQ8_0;
    return true;
  };
  for (std::uint32_t l = 0; l < impl.n_layers; ++l) {
    auto& w = impl.layers[l];
    if (impl.config.IsDense(l))
      continue;
    if (!to_q8(w.expert_gate) || !to_q8(w.expert_up) || !to_q8(w.expert_down))
      return nullptr;
  }
  for (std::uint32_t l = 0; l < impl.n_layers; ++l) {
    auto& w = impl.layers[l];
    if (!impl.config.IsMla(l) || w.attn_k_b.type != GgmlType::kQ8_0)
      continue;
    void* wk = nullptr;
    void* wv = nullptr;
    const std::size_t bytes = std::size_t{4096} * 512 * sizeof(__half);
    if (hipMalloc(&wk, bytes) != hipSuccess ||
        hipMalloc(&wv, bytes) != hipSuccess) {
      return nullptr;
    }
    BuildKvbHalf(w.attn_k_b.data, w.attn_v_b.data, static_cast<__half*>(wk),
                 static_cast<__half*>(wv), impl.stream);
    w.wk16 = {wk, bytes, GgmlType::kF16, 4096, 512, 1};
    w.wv16 = {wv, bytes, GgmlType::kF16, 4096, 512, 1};
  }
  if (hipStreamSynchronize(impl.stream) != hipSuccess)
    return nullptr;
  const auto stack = [&](std::initializer_list<DeviceTensor*> parts,
                         DeviceTensor* out) -> bool {
    std::size_t total = 0;
    for (const DeviceTensor* part : parts) {
      if (part->type != GgmlType::kQ8_0 || part->cols != (*parts.begin())->cols)
        return true;
      total += part->bytes;
    }
    void* buf = nullptr;
    if (hipMalloc(&buf, total) != hipSuccess)
      return Fail(error_msg, "hipMalloc failed for stacked weights");
    std::size_t off = 0;
    std::uint32_t rows = 0;
    for (DeviceTensor* part : parts) {
      if (hipMemcpyAsync(static_cast<std::uint8_t*>(buf) + off, part->data,
                         part->bytes, hipMemcpyDeviceToDevice,
                         impl.stream) != hipSuccess) {
        return Fail(error_msg, "stacked weight copy failed");
      }
      off += part->bytes;
      rows += part->rows;
    }
    if (hipStreamSynchronize(impl.stream) != hipSuccess)
      return Fail(error_msg, "stacked weight copy failed");
    off = 0;
    for (DeviceTensor* part : parts) {
      hipFree(part->data);
      part->data = static_cast<std::uint8_t*>(buf) + off;
      off += part->bytes;
    }
    *out = **parts.begin();
    out->data = buf;
    out->bytes = total;
    out->rows = rows;
    return true;
  };
  for (std::uint32_t l = 0; l < impl.n_layers; ++l) {
    auto& w = impl.layers[l];
    if (impl.config.IsKda(l) &&
        !stack({&w.attn_q, &w.attn_k, &w.attn_v, &w.ssm_f_a, &w.ssm_g_a},
               &w.kda_stack)) {
      return nullptr;
    }
    if (!impl.config.IsDense(l) &&
        !stack({&w.shared_gate, &w.shared_up}, &w.shared_gu)) {
      return nullptr;
    }
    if (!impl.config.IsDense(l) && w.router.type == GgmlType::kF32 &&
        std::getenv("LING_F32_ROUTER") == nullptr) {
      const std::uint32_t rk = impl.config.hidden_size;
      const std::size_t bytes =
          static_cast<std::size_t>(impl.config.expert_count) * 3U * rk *
          sizeof(__half);
      void* buf = nullptr;
      if (hipMalloc(&buf, bytes) != hipSuccess) {
        if (error_msg)
          *error_msg = "hipMalloc failed for router hi/lo copy";
        return nullptr;
      }
      SplitHiLo(static_cast<const float*>(w.router.data),
                static_cast<__half*>(buf), impl.config.expert_count, rk, true,
                impl.stream);
      w.router3.data = buf;
      w.router3.bytes = bytes;
      w.router3.type = GgmlType::kF16;
    }
  }
  hipStreamSynchronize(impl.stream);
  if (impl.config.HasMtp()) {
    const std::uint32_t l = impl.config.MtpLayer();
    LayerWeights& w = impl.mtp;
    w.attn_norm = get(P(l, "attn_norm.weight"));
    w.ffn_norm = get(P(l, "ffn_norm.weight"));
    w.attn_output = get(P(l, "attn_output.weight"));
    w.attn_q = get(P(l, "attn_q.weight"));
    w.attn_kv_a = get(P(l, "attn_kv_a_mqa.weight"));
    w.attn_kv_norm = get(P(l, "attn_kv_a_norm.weight"));
    w.attn_k_b = get(P(l, "attn_k_b.weight"));
    w.attn_v_b = get(P(l, "attn_v_b.weight"));
    w.attn_gate = get(P(l, "attn_gate.weight"));
    w.router = get(P(l, "ffn_gate_inp.weight"));
    w.router_bias = get(P(l, "exp_probs_b.bias"));
    w.expert_gate = get(P(l, "ffn_gate_exps.weight"));
    w.expert_up = get(P(l, "ffn_up_exps.weight"));
    w.expert_down = get(P(l, "ffn_down_exps.weight"));
    w.shared_gate = get(P(l, "ffn_gate_shexp.weight"));
    w.shared_up = get(P(l, "ffn_up_shexp.weight"));
    w.shared_down = get(P(l, "ffn_down_shexp.weight"));
    w.eh_proj = get(P(l, "nextn.eh_proj.weight"));
    w.enorm = get(P(l, "nextn.enorm.weight"));
    w.hnorm = get(P(l, "nextn.hnorm.weight"));
    w.layer_out_norm = get(P(l, "layer_output_norm.weight"));
  }
  return model;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  auto session = std::unique_ptr<Session>(new Session());
  auto& s = *session->impl_;
  s.model = shared_from_this();
  s.stream = impl_->stream;
  s.max_context = std::min(max_context, impl_->config.context_length);
  const Config& c = impl_->config;
  const std::uint32_t di = c.KdaDim();
  const std::uint32_t key = c.kv_lora_rank + c.rope_dimension;
  const std::uint32_t ff = c.dense_ff;
  const std::uint32_t eff = c.expert_ff;
  const std::uint32_t h = c.hidden_size;

  const auto alloc = [&](DevBuf& b, std::uint32_t n, const char* what) {
    return b.Alloc(n, sizeof(float), error_msg, what);
  };
  const auto balloc = [&](DevBuf& b, std::uint32_t n, const char* what) {
    return b.Alloc(static_cast<std::size_t>(n) * kPrefillTile, sizeof(float),
                   error_msg, what);
  };
  if (!balloc(s.x, h, "x") || !balloc(s.x1, h, "x1") ||
      !balloc(s.xn, h, "xn") || !balloc(s.block, h, "block") ||
      !balloc(s.fin, h, "fin") || !balloc(s.ffn, h, "ffn") ||
      !alloc(s.logits_dev, c.vocab_size, "logits") || !balloc(s.q, di, "q") ||
      !balloc(s.k, di, "k") || !balloc(s.v, di, "v") ||
      !balloc(s.oq, di, "oq") || !balloc(s.ok, di, "ok") ||
      !balloc(s.ov, di, "ov") || !balloc(s.fa, di, "fa") ||
      !balloc(s.ga, di, "ga") || !balloc(s.beta, c.num_heads, "beta") ||
      !balloc(s.qall, c.num_heads * c.key_length_mla, "qall") ||
      !balloc(s.kvall, key, "kvall") ||
      !balloc(s.qrope, c.num_heads * c.rope_dimension, "qrope") ||
      !balloc(s.kpe, c.rope_dimension, "kpe") ||
      !alloc(s.keyrow, key, "keyrow") ||
      !alloc(s.scores, c.num_heads * s.max_context, "scores") ||
      !alloc(s.qabs, c.kv_lora_rank, "qabs") ||
      !alloc(s.qvec, c.num_heads * key, "qvec") ||
      !s.pos_dev.Alloc(1, sizeof(std::int32_t), error_msg, "pos") ||
      !s.pos_mtp.Alloc(1, sizeof(std::int32_t), error_msg, "pos_mtp") ||
      !alloc(s.pm, kMlaMaxSplitCount * c.num_heads, "pm") ||
      !alloc(s.pl, kMlaMaxSplitCount * c.num_heads, "pl") ||
      !alloc(s.pacc, kMlaMaxSplitCount * c.num_heads * c.kv_lora_rank,
             "pacc") ||
      !alloc(s.ctxlat, c.num_heads * c.kv_lora_rank, "ctxlat") ||
      !alloc(s.vhead, c.num_heads * c.MlaVDim(), "vhead") ||
      !balloc(s.gate, c.num_heads, "gate") ||
      !balloc(s.attn, c.num_heads * c.MlaVDim(), "attn") ||
      !balloc(s.attn_proj, h, "attn_proj") ||
      !balloc(s.kda_out, di, "kda_out") || !balloc(s.kda_so, di, "kda_so") ||
      !balloc(s.qkvfg, 5U * di, "qkvfg") || !balloc(s.sgu, 2U * eff, "sgu") ||
      !alloc(s.mtp_h, h, "mtp_h") || !alloc(s.mtp_prev_h, h, "mtp_prev_h") ||
      !alloc(s.mtp_pending_h, h, "mtp_pending_h") ||
      !alloc(s.mtp_draft_h, h, "mtp_draft_h") ||
      !alloc(s.mtp_embd, h, "mtp_embd") ||
      !alloc(s.mtp_e_norm, h, "mtp_e_norm") ||
      !alloc(s.mtp_h_norm, h, "mtp_h_norm") ||
      !alloc(s.mtp_eh, 2U * h, "mtp_eh") || !alloc(s.mtp_cur, h, "mtp_cur") ||
      !alloc(s.mtp_logits_dev, c.vocab_size, "mtp_logits") ||
      !alloc(s.router_logits, c.expert_count, "router") ||
      !balloc(s.gf, c.expert_used * eff, "gf") ||
      !balloc(s.uf, c.expert_used * eff, "uf") ||
      !balloc(s.act, c.expert_used * eff, "act") ||
      !balloc(s.part, c.expert_used * h, "part") || !balloc(s.sg, eff, "sg") ||
      !balloc(s.su, eff, "su") || !balloc(s.shared_act, eff, "shared_act") ||
      !balloc(s.shared_part, h, "shared_part") ||
      !balloc(s.dense_gate, ff, "dense_gate") ||
      !balloc(s.dense_up, ff, "dense_up") ||
      !balloc(s.dense_act, ff, "dense_act") ||
      !alloc(s.ex_g, c.expert_used * eff * h, "expert gate dequant") ||
      !alloc(s.ex_u, c.expert_used * eff * h, "expert up dequant") ||
      !alloc(s.ex_d, c.expert_used * h * eff, "expert down dequant")) {
    return nullptr;
  }
  const std::uint32_t max_k =
      std::max({c.hidden_size, di, 2U * c.hidden_size, c.dense_ff});
  if (!s.token.Alloc(2U * kPrefillTile, sizeof(std::int32_t), error_msg,
                     "token") ||
      !s.q8.Alloc(qfn_mmq_q8_1_bytes(static_cast<int>(kBatchTokens),
                                     static_cast<int>(max_k)),
                  1, error_msg, "q8")) {
    return nullptr;
  }
  const std::size_t slots =
      static_cast<std::size_t>(kPrefillTile) * c.expert_used;
  const std::size_t tile_cap = slots / 16 + c.expert_count + 16;
  const std::size_t key_cap =
      std::min<std::size_t>(s.max_context, kPrefillKeyCap);
  if (hipblasCreate(&s.blas) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(s.blas, s.stream) != HIPBLAS_STATUS_SUCCESS) {
    Fail(error_msg, "hipblas init failed");
    return nullptr;
  }
  if (!s.ids.Alloc(slots, sizeof(std::int32_t), error_msg, "ids") ||
      !s.weights.Alloc(slots, sizeof(float), error_msg, "weights") ||
      !s.pf_kn.Alloc(key_cap * 4096, sizeof(float), error_msg, "pf_kn") ||
      !s.pf_q8.Alloc(
          qfn::Q8TiledBytes(kPrefillTile, std::max({h, di, c.dense_ff})) +
              (1U << 20),
          1, error_msg, "pf_q8") ||
      !s.pf_k16.Alloc(key_cap * 32 * 192, sizeof(__half), error_msg,
                      "pf_k16") ||
      !s.pf_v16.Alloc(key_cap * 32 * 128, sizeof(__half), error_msg,
                      "pf_v16") ||
      !s.pf_q16.Alloc(static_cast<std::size_t>(kPrefillTile) * 32 * 192,
                      sizeof(__half), error_msg, "pf_q16") ||
      !s.spec_h.Alloc(8U * h, sizeof(float), error_msg, "spec_h") ||
      !s.spec_logits_dev.Alloc(8U * c.vocab_size, sizeof(float), error_msg,
                               "spec_logits") ||
      !s.sv_prev.Alloc(h, sizeof(float), error_msg, "sv_prev") ||
      !s.lat_sel.Alloc(key_cap * 576, sizeof(float), error_msg, "lat_sel") ||
      !s.sel_scores.Alloc(4096, sizeof(float), error_msg, "sel_scores") ||
      !s.sel_idx.Alloc(kSparseBlocks, sizeof(std::int32_t), error_msg,
                       "sel_idx") ||
      !s.sel_pos.Alloc(1, sizeof(std::int32_t), error_msg, "sel_pos") ||
      !s.mtp_bf.Alloc(static_cast<std::size_t>(kMtpBackfill) * h, sizeof(float),
                      error_msg, "mtp_bf") ||
      !s.pf_vv.Alloc(key_cap * 4096, sizeof(float), error_msg, "pf_vv") ||
      !s.pf_lat16.Alloc(key_cap * 512, sizeof(__half), error_msg, "pf_lat16") ||
      !s.pf_fin16.Alloc(static_cast<std::size_t>(kPrefillTile) * h,
                        sizeof(__half), error_msg, "pf_fin16") ||
      !s.pf_x3.Alloc(static_cast<std::size_t>(kPrefillTile) * 3U * h,
                     sizeof(__half), error_msg, "pf_x3") ||
      !s.pf_act16.Alloc(slots * eff, sizeof(__half), error_msg, "pf_act16") ||
      !s.pf_gate_e.Alloc(slots * eff, sizeof(float), error_msg, "pf_gate_e") ||
      !s.pf_up_e.Alloc(slots * eff, sizeof(float), error_msg, "pf_up_e") ||
      !s.pf_down_e.Alloc(slots * h, sizeof(float), error_msg, "pf_down_e") ||
      !s.pf_router.Alloc(
          static_cast<std::size_t>(kPrefillTile) * c.expert_count,
          sizeof(float), error_msg, "pf_router") ||
      !s.pf_counts.Alloc(c.expert_count, sizeof(std::uint32_t), error_msg,
                         "pf_counts") ||
      !s.pf_bounds.Alloc(c.expert_count + 1, sizeof(std::int32_t), error_msg,
                         "pf_bounds") ||
      !s.pf_cursors.Alloc(c.expert_count, sizeof(std::int32_t), error_msg,
                          "pf_cursors") ||
      !s.pf_rows_token.Alloc(qfn::RoutedCompactRows(slots, c.expert_count),
                             sizeof(std::int32_t), error_msg,
                             "pf_rows_token") ||
      !s.pf_rows_slot.Alloc(qfn::RoutedCompactRows(slots, c.expert_count),
                            sizeof(std::int32_t), error_msg, "pf_rows_slot") ||
      !s.pf_tiles48.Alloc(tile_cap, sizeof(std::int32_t), error_msg,
                          "tiles48") ||
      !s.pf_tiles64.Alloc(tile_cap, sizeof(std::int32_t), error_msg,
                          "tiles64") ||
      !s.pf_tiles128.Alloc(tile_cap, sizeof(std::int32_t), error_msg,
                           "tiles128")) {
    return nullptr;
  }
  s.pf_host_counts.resize(c.expert_count);
  s.pf_host_tiles.resize(3 * tile_cap);
  s.logits.resize(c.vocab_size);
  s.mtp_logits.resize(c.vocab_size);

  s.layers.resize(impl_->config.num_layers);
  for (std::uint32_t l = 0; l < impl_->config.num_layers; ++l) {
    auto& state = s.layers[l];
    if (c.IsKda(l)) {
      if (!state.conv.Alloc(3 * (c.ssm_conv_kernel - 1) * di, sizeof(float),
                            error_msg, "conv") ||
          !state.state.Alloc(static_cast<std::size_t>(c.num_heads) *
                                 c.kda_head_dim * c.kda_head_dim,
                             sizeof(float), error_msg, "kda state") ||
          !state.ck_state.Alloc(8U * static_cast<std::size_t>(c.num_heads) *
                                    c.kda_head_dim * c.kda_head_dim,
                                sizeof(float), error_msg, "kda ckpt") ||
          !state.ck_hist.Alloc(3U * (c.ssm_conv_kernel - 1) * di, sizeof(float),
                               error_msg, "conv ckpt") ||
          !state.ck_raw.Alloc(3U * 8U * di, sizeof(float), error_msg,
                              "raw ckpt")) {
        return nullptr;
      }
    } else {
      if (!state.cache.Alloc(static_cast<std::size_t>(s.max_context) * key,
                             sizeof(float), error_msg, "mla cache") ||
          !state.pool.Alloc(
              (static_cast<std::size_t>(s.max_context) / kPoolBlockRows + 2) *
                  2 * key,
              sizeof(float), error_msg, "mla pool")) {
        return nullptr;
      }
    }
  }
  s.SetupNpu();
  session->Reset();
  if (s.max_context >= kPrefillTile) {
    std::vector<std::int32_t> warm(2U * kPrefillTile, 1);
    std::string warm_error;
    if (s.npu != nullptr && !s.npu_qkv.empty() &&
        s.max_context >= 2U * kPrefillTile) {
      s.ForwardPair(warm.data(), &warm_error);
      hipStreamSynchronize(s.stream);
      session->Reset();
    }
    s.ForwardPrefill(warm.data(), kPrefillTile, &warm_error);
    hipStreamSynchronize(s.stream);
    session->Reset();
  }
  return session;
}

const Config& Model::config() const noexcept {
  return impl_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return impl_->resident;
}

}  // namespace gufo::models::ling32::gpu
