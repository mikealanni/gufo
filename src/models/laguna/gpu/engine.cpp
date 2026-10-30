#include "src/models/laguna/gpu/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/gpu/device_model.hpp"
#include "src/models/laguna/gpu/kernels/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models::laguna::gpu {
namespace {

constexpr std::uint32_t kVecBatch = 8;
constexpr std::uint32_t kVecExpertRows = 32;
constexpr std::uint32_t kMaxTopK = 16;
constexpr std::uint32_t kMaxExperts = 256;
constexpr std::uint32_t kRoutePadRows = 16;
constexpr std::uint32_t kPairTileRows = 128;
constexpr std::uint32_t kPlainTileRows = 48;
constexpr std::uint32_t kQ8TileRows = 64;
constexpr std::uint32_t kRoutedTileMin = 16;
constexpr std::uint32_t kRoutedTileMax = 80;
constexpr std::uint32_t kRoutedTileStep = 16;
constexpr std::uint32_t kWarmTokens = 64;
// Routed-expert activations are stored divided by this (F16 range); the
// combine multiplies it back.
constexpr float kRoutedActScale = 64.0F;

using core::GgmlType;

models::QwenTensorRef TensorRef(const DeviceTensor& t) {
  models::QwenTensorRef ref;
  ref.data = t.data;
  ref.type = t.type;
  ref.num_elements = static_cast<std::size_t>(t.cols) * t.rows;
  return ref;
}

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr)
    *error_msg = message;
  return false;
}

#define LAGUNA_HIP(call)                                                   \
  do {                                                                     \
    const hipError_t status_ = (call);                                     \
    if (status_ != hipSuccess) {                                           \
      return Fail(error_msg,                                               \
                  std::string(#call) + ": " + hipGetErrorString(status_)); \
    }                                                                      \
  } while (false)

template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  bool Allocate(std::size_t count) {
    Reset();
    if (hipMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) !=
        hipSuccess) {
      ptr_ = nullptr;
      return false;
    }
    count_ = count;
    return true;
  }
  void Reset() {
    if (ptr_ != nullptr)
      (void)hipFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }
  T* get() const noexcept { return ptr_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
  T* ptr_{nullptr};
  std::size_t count_{0};
};

int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1U));
  for (std::uint32_t cols = kRoutedTileMin; cols < kRoutedTileMax;
       cols += kRoutedTileStep) {
    if (cols >= 2 * mean)
      return static_cast<int>(cols);
  }
  return static_cast<int>(kRoutedTileMax);
}

bool DenseVecType(GgmlType type) {
  return type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
         type == GgmlType::kQ5_1 || type == GgmlType::kQ6_K;
}

std::uint32_t TileCapacityFor(std::size_t slots, std::uint32_t experts) {
  return static_cast<std::uint32_t>(
      (slots + std::size_t{experts} * (kRoutePadRows - 1)) / kRoutePadRows +
      experts + 1);
}

}  // namespace

struct ModelState {
  EngineOptions options;
  std::shared_ptr<core::GgufReader> reader;
  Config config;
  std::unique_ptr<DeviceModel> device;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
  std::vector<std::uint32_t> dflash_taps;
  std::uint32_t dflash_window{0};
  std::uint32_t drafter_block{0};
  bool keep_reader{false};
};

struct Session::Impl {
  std::shared_ptr<ModelState> model;
  hipStream_t stream{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t tile{0};
  std::uint32_t position{0};
  std::uint32_t logit_rows{1};
  float* h_logits{nullptr};
  std::int32_t* h_tokens{nullptr};
  std::uint32_t* h_pos{nullptr};
  std::int32_t* h_tiles{nullptr};
  __half* xh{nullptr};
  hipEvent_t ready{nullptr};
  std::uint32_t* h_counts{nullptr};

  std::vector<DeviceBuffer<__half>> k_cache, v_cache;
  std::vector<std::uint32_t> ring;
  std::unique_ptr<hip::QwenDFlashGpuExecutor> dflash;
  DeviceBuffer<float> tap_capture;
  std::uint32_t tap_capture_rows{0};
  std::uint32_t tap_capture_pos{0};
  std::uint32_t tap_pending{0}, tap_pending_pos{0}, tap_pending_first{0};
  bool inject_taps{true};
  DeviceBuffer<std::int32_t> tokens;
  DeviceBuffer<std::uint32_t> pos;
  DeviceBuffer<float> x, xn, q, k, v, gate, attn, o, router;
  DeviceBuffer<float> gate_e, up_e, down_e, sh_gate, sh_up, sh_out;
  DeviceBuffer<float> expert_scale, expert_const, dense_gate, dense_up,
      d_logits;
  DeviceBuffer<float> topk, last;
  DeviceBuffer<std::int32_t> ids;
  DeviceBuffer<std::uint32_t> counts;
  DeviceBuffer<std::int32_t> route_bounds, route_cursors, rows_token, rows_slot;
  DeviceBuffer<std::int32_t> tiles_pair, tiles_down;
  DeviceBuffer<__half> act_half, qh;

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
    if (xh != nullptr)
      (void)hipFree(xh);
    if (ready != nullptr)
      (void)hipEventDestroy(ready);
    if (h_logits != nullptr)
      (void)hipHostFree(h_logits);
    if (h_tokens != nullptr)
      (void)hipHostFree(h_tokens);
    if (h_pos != nullptr)
      (void)hipHostFree(h_pos);
    if (h_tiles != nullptr)
      (void)hipHostFree(h_tiles);
    if (h_counts != nullptr)
      (void)hipHostFree(h_counts);
  }

  bool Init(std::uint32_t max_context, std::string* error_msg);
  bool Dense(const DeviceTensor& w, const float* in, float* out,
             std::uint32_t n, std::string* error_msg);
  bool Experts(const DeviceTensor& w, const float* in, float* out,
               std::uint32_t rows, std::uint32_t used, std::uint32_t n_tokens,
               std::string* error_msg);
  bool Attention(std::uint32_t il, std::uint32_t n, std::string* error_msg);
  bool DenseFfn(const DeviceLayer& layer, std::uint32_t n,
                std::string* error_msg);
  bool Moe(const DeviceLayer& layer, std::uint32_t n, std::string* error_msg);
  bool RoutedExperts(const DeviceLayer& layer, std::uint32_t n, bool* scaled,
                     std::string* error_msg);
  bool Forward(std::span<const std::int32_t> tokens, bool all_rows,
               std::uint32_t prefix_end, std::string* error_msg);
  bool DflashInject(std::uint32_t rows, std::string* error_msg);
};

bool Session::Impl::Init(std::uint32_t max_context, std::string* error_msg) {
  const Config& c = model->config;
  if (c.expert_used > kMaxTopK || c.expert_count > kMaxExperts) {
    return Fail(error_msg, "Laguna: expert counts exceed the router limit");
  }
  capacity = max_context;
  tile = std::max<std::uint32_t>(1, model->options.prefill_tile);
  LAGUNA_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  LAGUNA_HIP(hipEventCreateWithFlags(&ready, hipEventDisableTiming));
  if (qfn_mmq_init(0) != 0)
    return Fail(error_msg, "mmq initialization failed");

  const std::size_t kv_row = static_cast<std::size_t>(c.num_kv) * c.head_dim;
  const std::uint32_t max_heads =
      *std::max_element(c.heads.begin(), c.heads.end());
  const std::size_t q_row = static_cast<std::size_t>(max_heads) * c.head_dim;
  const std::size_t slots = static_cast<std::size_t>(tile) * c.expert_used;
  k_cache.resize(c.num_layers);
  v_cache.resize(c.num_layers);
  ring.assign(c.num_layers, 0);
  bool ok = true;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    std::uint32_t rows = capacity;
    if (c.IsSwa(il)) {
      ring[il] = c.sliding_window + tile;
      rows = std::min(capacity, ring[il]);
      if (rows == capacity)
        ring[il] = 0;
    }
    ok = ok && k_cache[il].Allocate(kv_row * rows) &&
         v_cache[il].Allocate(kv_row * rows);
  }
  ok = ok && tokens.Allocate(tile) && pos.Allocate(4) &&
       x.Allocate(std::size_t{tile} * c.hidden_size) &&
       xn.Allocate(std::size_t{tile} * c.hidden_size) &&
       q.Allocate(std::size_t{tile} * q_row) &&
       qh.Allocate(std::size_t{tile} * q_row) &&
       k.Allocate(std::size_t{tile} * kv_row) &&
       v.Allocate(std::size_t{tile} * kv_row) &&
       gate.Allocate(std::size_t{tile} * max_heads) &&
       attn.Allocate(std::size_t{tile} * q_row) &&
       o.Allocate(std::size_t{tile} * c.hidden_size) &&
       router.Allocate(std::size_t{tile} * c.expert_count) &&
       gate_e.Allocate(slots * c.expert_ff) &&
       up_e.Allocate(slots * c.expert_ff) &&
       down_e.Allocate(slots * c.hidden_size) && topk.Allocate(slots) &&
       ids.Allocate(slots) && counts.Allocate(c.expert_count) &&
       route_bounds.Allocate(kMaxExperts + 1) &&
       route_cursors.Allocate(kMaxExperts) &&
       rows_token.Allocate(
           qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       rows_slot.Allocate(
           qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       tiles_pair.Allocate(TileCapacityFor(slots, c.expert_count)) &&
       tiles_down.Allocate(TileCapacityFor(slots, c.expert_count)) &&
       act_half.Allocate(slots * c.expert_ff) && expert_scale.Allocate(slots) &&
       expert_const.Allocate(slots) &&
       sh_gate.Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_up.Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_out.Allocate(std::size_t{tile} * c.hidden_size) &&
       dense_gate.Allocate(std::size_t{tile} * c.dense_ff) &&
       dense_up.Allocate(std::size_t{tile} * c.dense_ff) &&
       last.Allocate(std::size_t{kMaxVerifyRows} * c.hidden_size) &&
       d_logits.Allocate(std::size_t{kMaxVerifyRows} * c.vocab_size);
  ok = ok &&
       hipMalloc(reinterpret_cast<void**>(&xh),
                 std::size_t{tile} * c.hidden_size * sizeof(__half)) ==
           hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_logits),
                     std::size_t{kMaxVerifyRows} * c.vocab_size *
                         sizeof(float)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tokens),
                     std::size_t{tile} * sizeof(std::int32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_pos),
                     4 * sizeof(std::uint32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_counts),
                     kMaxExperts * sizeof(std::uint32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tiles),
                     2 * std::size_t{TileCapacityFor(slots, c.expert_count)} *
                         sizeof(std::int32_t)) == hipSuccess;
  if (!ok)
    return Fail(error_msg, "Laguna: session allocation failed");
  {
    const std::vector<float> efill(slots, kRoutedActScale);
    LAGUNA_HIP(hipMemcpy(expert_const.get(), efill.data(),
                         efill.size() * sizeof(float), hipMemcpyHostToDevice));
  }
  std::fill(h_logits, h_logits + std::size_t{kMaxVerifyRows} * c.vocab_size,
            0.0F);
  return true;
}

bool Session::Impl::Dense(const DeviceTensor& w, const float* in, float* out,
                          std::uint32_t n, std::string* error_msg) {
  if (w.type == GgmlType::kQ8_0) {
    if (qfn_mmq_q8_0_dense(w.data, in, out, static_cast<int>(w.rows),
                           static_cast<int>(n), static_cast<int>(w.cols),
                           stream) != 0) {
      return Fail(error_msg, "Q8_0 dense projection failed");
    }
    return true;
  }
  if (!DenseVecType(w.type))
    return Fail(error_msg, "unsupported dense weight type");
  for (std::uint32_t r0 = 0; r0 < n; r0 += kVecBatch) {
    const std::uint32_t rows = std::min(kVecBatch, n - r0);
    const float* src = in + static_cast<std::size_t>(r0) * w.cols;
    float* dst = out + static_cast<std::size_t>(r0) * w.rows;
    if (qfn_mmq_dense_vec(static_cast<int>(w.type), w.data, src, dst,
                          static_cast<int>(w.rows), static_cast<int>(w.cols),
                          static_cast<int>(rows), stream) != 0) {
      return Fail(error_msg, "quantized dense projection failed");
    }
  }
  return true;
}

bool Session::Impl::Experts(const DeviceTensor& w, const float* in, float* out,
                            std::uint32_t rows, std::uint32_t used,
                            std::uint32_t n_tokens, std::string* error_msg) {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(rows);
  const int U = static_cast<int>(used);
  if (rows * used > kVecExpertRows) {
    return Fail(error_msg, "Laguna: wide expert batch on the vector path");
  }
  (void)n_tokens;
  if (qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, in, ids.get(), out, M,
                      K, T, E, U, stream, nullptr, nullptr) != 0) {
    return Fail(error_msg, "routed expert projection failed");
  }
  return true;
}

bool Session::Impl::Attention(std::uint32_t il, std::uint32_t n,
                              std::string* error_msg) {
  const Config& c = model->config;
  const DeviceLayer& L = model->device->layers()[il];
  const std::uint32_t heads = c.Heads(il);
  const std::uint32_t kv_row = c.num_kv * c.head_dim;
  const bool swa = c.IsSwa(il);
  const RopeTables& rope = model->device->rope(swa);

  if (!Dense(L.attn_q, xn.get(), q.get(), n, error_msg) ||
      !Dense(L.attn_k, xn.get(), k.get(), n, error_msg) ||
      !Dense(L.attn_v, xn.get(), v.get(), n, error_msg)) {
    return false;
  }
  rocm::Q8Rows(L.attn_gate.data, xn.get(), gate.get(), n, L.attn_gate.rows,
               L.attn_gate.cols, stream);
  rocm::RmsNormRows(q.get(), static_cast<const float*>(L.attn_q_norm.data),
                    q.get(), n, heads * c.head_dim, heads, c.rms_eps, stream);
  rocm::RmsNormRows(k.get(), static_cast<const float*>(L.attn_k_norm.data),
                    k.get(), n, kv_row, c.num_kv, c.rms_eps, stream);
  rocm::RopeTable(q.get(), n, heads, c.head_dim, rope.n_dims, rope.inv_freq,
                  rope.mscale, pos.get(), stream);
  rocm::RopeTable(k.get(), n, c.num_kv, c.head_dim, rope.n_dims, rope.inv_freq,
                  rope.mscale, pos.get(), stream);
  rocm::StoreKv(k.get(), v.get(), k_cache[il].get(), v_cache[il].get(), n,
                kv_row, pos.get(), ring[il], stream);
  rocm::ToHalf(q.get(), qh.get(), std::size_t{n} * heads * c.head_dim, stream);
  // Four real bugs found and fixed in AttentionWmma/AttentionWmmaKernelLaguna
  // (shape guard; token stride missing *2; missing softplus(gate) multiply
  // the scalar reference applies but WMMA silently dropped; head address
  // using the tile width (kWmmaGroup=8) instead of the real per-kv_head
  // stride (heads/kv_heads=6) as the memory stride, which only worked by
  // coincidence for kv_head==0). Verified via `gufo bench
  // --validate-prefill` and a per-layer scalar-vs-WMMA trace
  // (LAGUNA_WMMA_DEBUG=1): every group-6 layer now matches scalar to
  // fp16-precision-level noise only. See laguna-port-state.md for the full
  // debugging trail, including why n=16 alone still fails top1_match (a
  // pre-existing near-tie numeric sensitivity, not a WMMA bug -- confirmed
  // by reproducing the same divergence with WMMA fully disabled).
  constexpr bool kWmmaAttentionEnabled = true;
  const bool use_wmma =
      kWmmaAttentionEnabled &&
      rocm::AttentionWmma(qh.get(), k_cache[il].get(), v_cache[il].get(),
                          gate.get(), attn.get(), n, pos.get(), heads, c.num_kv,
                          c.head_dim, swa ? c.sliding_window : 0, ring[il],
                          stream);
  if (!use_wmma) {
    rocm::Attention(q.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
                    attn.get(), n, pos.get(), heads, c.num_kv, c.head_dim,
                    swa ? c.sliding_window : 0, ring[il], stream);
  }
  if (use_wmma && std::getenv("LAGUNA_WMMA_DEBUG") != nullptr) {
    DeviceBuffer<float> scalar_attn;
    scalar_attn.Allocate(std::size_t{n} * heads * c.head_dim);
    rocm::Attention(q.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
                    scalar_attn.get(), n, pos.get(), heads, c.num_kv,
                    c.head_dim, swa ? c.sliding_window : 0, ring[il], stream);
    LAGUNA_HIP(hipStreamSynchronize(stream));
    std::vector<float> wmma_host(std::size_t{n} * heads * c.head_dim);
    std::vector<float> scalar_host(wmma_host.size());
    LAGUNA_HIP(hipMemcpy(wmma_host.data(), attn.get(),
                         wmma_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    LAGUNA_HIP(hipMemcpy(scalar_host.data(), scalar_attn.get(),
                         scalar_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    for (std::uint32_t t = 0; t < n; ++t) {
      for (std::uint32_t h = 0; h < heads; ++h) {
        double max_abs = 0.0;
        double sum_abs_scalar = 0.0;
        const std::size_t base = (std::size_t{t} * heads + h) * c.head_dim;
        for (std::uint32_t d = 0; d < c.head_dim; ++d) {
          const double diff =
              std::fabs(wmma_host[base + d] - scalar_host[base + d]);
          max_abs = std::max(max_abs, diff);
          sum_abs_scalar += std::fabs(scalar_host[base + d]);
        }
        std::fprintf(stderr,
                     "[LAGUNA_WMMA_DEBUG] layer=%u token=%u head=%u "
                     "max_abs_diff=%.6f scalar_sum_abs=%.6f "
                     "wmma[0..3]=%.4f,%.4f,%.4f,%.4f "
                     "scalar[0..3]=%.4f,%.4f,%.4f,%.4f\n",
                     il, t, h, max_abs, sum_abs_scalar, wmma_host[base + 0],
                     wmma_host[base + 1], wmma_host[base + 2],
                     wmma_host[base + 3], scalar_host[base + 0],
                     scalar_host[base + 1], scalar_host[base + 2],
                     scalar_host[base + 3]);
      }
    }
  }
  if (!Dense(L.attn_output, attn.get(), o.get(), n, error_msg))
    return false;
  rocm::AddInPlace(x.get(), o.get(), std::size_t{n} * c.hidden_size, stream);
  return true;
}

bool Session::Impl::DenseFfn(const DeviceLayer& layer, std::uint32_t n,
                             std::string* error_msg) {
  const Config& c = model->config;
  if (!Dense(layer.ffn_gate, xn.get(), dense_gate.get(), n, error_msg) ||
      !Dense(layer.ffn_up, xn.get(), dense_up.get(), n, error_msg)) {
    return false;
  }
  rocm::Swiglu(dense_gate.get(), dense_up.get(), std::size_t{n} * c.dense_ff,
               stream);
  if (!Dense(layer.ffn_down, dense_gate.get(), o.get(), n, error_msg))
    return false;
  rocm::AddInPlace(x.get(), o.get(), std::size_t{n} * c.hidden_size, stream);
  return true;
}

bool Session::Impl::RoutedExperts(const DeviceLayer& layer, std::uint32_t n,
                                  bool* scaled, std::string* error_msg) {
  using qwen38_flash_next::rocm::WeightType;
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  const std::uint32_t experts = c.expert_count;
  const DeviceTensor& ge = layer.gate_exps;
  const DeviceTensor& ue = layer.up_exps;
  const DeviceTensor& de = layer.down_exps;
  const bool gated = ge.type == ue.type &&
                     (ge.type == GgmlType::kQ4_K || ge.type == GgmlType::kQ5_K);
  *scaled = gated;
  const std::uint32_t pair_rows = gated ? kPairTileRows : kPlainTileRows;
  const std::uint32_t down_rows =
      de.type == GgmlType::kQ8_0 ? kQ8TileRows : kPlainTileRows;

  qwen38_flash_next::rocm::ExpertCounts(ids.get(), counts.get(), n, experts,
                                        used, stream);
  LAGUNA_HIP(hipMemcpyAsync(h_counts, counts.get(),
                            experts * sizeof(std::uint32_t),
                            hipMemcpyDeviceToHost, stream));
  LAGUNA_HIP(hipEventRecord(ready, stream));
  LAGUNA_HIP(hipEventSynchronize(ready));
  std::uint32_t pair_tiles = 0;
  std::uint32_t down_tiles = 0;
  std::int32_t* host_pair = h_tiles;
  std::int32_t* host_down =
      h_tiles + TileCapacityFor(static_cast<std::size_t>(tile) * used, experts);
  for (std::uint32_t e = 0; e < experts; ++e) {
    const std::uint32_t padded =
        (h_counts[e] + kRoutePadRows - 1) / kRoutePadRows * kRoutePadRows;
    for (std::uint32_t j = 0; j * pair_rows < padded; ++j) {
      host_pair[pair_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    for (std::uint32_t j = 0; j * down_rows < padded; ++j) {
      host_down[down_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  if (pair_tiles == 0)
    return Fail(error_msg, "routed experts: no tiles");
  LAGUNA_HIP(hipMemcpyAsync(tiles_pair.get(), host_pair,
                            pair_tiles * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  LAGUNA_HIP(hipMemcpyAsync(tiles_down.get(), host_down,
                            down_tiles * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  qwen38_flash_next::rocm::RoutedCompact(
      ids.get(), counts.get(), route_bounds.get(), route_cursors.get(),
      rows_token.get(), rows_slot.get(), n, used, experts, stream);
  rocm::NarrowRows(xn.get(), xh, n, c.hidden_size, c.hidden_size, stream);

  using qwen38_flash_next::rocm::RoutedF16Gemm;
  using qwen38_flash_next::rocm::RoutedGatedF16Gemm;
  const auto gate_type = static_cast<WeightType>(ge.type);
  const auto down_type = static_cast<WeightType>(de.type);
  bool ok;
  if (gated) {
    ok = RoutedGatedF16Gemm(ge.data, ue.data, gate_type, xh, tiles_pair.get(),
                            pair_tiles, pair_rows, route_bounds.get(),
                            rows_token.get(), rows_slot.get(), act_half.get(),
                            ge.rows, ge.cols, stream, 1.0F / kRoutedActScale);
  } else {
    ok = RoutedF16Gemm(ge.data, gate_type, xh, tiles_pair.get(), pair_tiles,
                       pair_rows, route_bounds.get(), rows_token.get(),
                       rows_slot.get(), nullptr, gate_e.get(), nullptr, ge.rows,
                       ge.cols, stream) &&
         RoutedF16Gemm(ue.data, static_cast<WeightType>(ue.type), xh,
                       tiles_pair.get(), pair_tiles, pair_rows,
                       route_bounds.get(), rows_token.get(), rows_slot.get(),
                       gate_e.get(), nullptr, act_half.get(), ue.rows, ue.cols,
                       stream);
  }
  if (!ok)
    return Fail(error_msg, "routed gate/up projection failed");
  if (!RoutedF16Gemm(de.data, down_type, act_half.get(), tiles_down.get(),
                     down_tiles, down_rows, route_bounds.get(), rows_slot.get(),
                     rows_slot.get(), nullptr, down_e.get(), nullptr, de.rows,
                     de.cols, stream)) {
    return Fail(error_msg, "routed down projection failed");
  }
  return true;
}

bool Session::Impl::Moe(const DeviceLayer& layer, std::uint32_t n,
                        std::string* error_msg) {
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  const bool wide = n * used > kVecExpertRows;

  rocm::RouterF32(static_cast<const float*>(layer.router.data), xn.get(),
                  router.get(), n, c.expert_count, c.hidden_size, stream);
  rocm::SigmoidBiasTopK(router.get(),
                        static_cast<const float*>(layer.router_bias.data),
                        ids.get(), topk.get(), n, c.expert_count, used,
                        c.expert_weights_norm, c.expert_weights_scale, stream);

  if (!Dense(layer.gate_shexp, xn.get(), sh_gate.get(), n, error_msg) ||
      !Dense(layer.up_shexp, xn.get(), sh_up.get(), n, error_msg)) {
    return false;
  }
  rocm::Swiglu(sh_gate.get(), sh_up.get(), std::size_t{n} * c.shared_ff,
               stream);
  if (!Dense(layer.down_shexp, sh_gate.get(), sh_out.get(), n, error_msg))
    return false;

  bool scaled = false;
  if (wide) {
    if (!RoutedExperts(layer, n, &scaled, error_msg))
      return false;
  } else {
    const DeviceTensor& ge = layer.gate_exps;
    const DeviceTensor& ue = layer.up_exps;
    const bool paired =
        ge.type == ue.type && ge.rows == ue.rows && ge.cols == ue.cols;
    if (paired) {
      if (qfn_mmq_moe_vec(static_cast<int>(ge.type), ge.data, xn.get(),
                          ids.get(), gate_e.get(), static_cast<int>(ge.rows),
                          static_cast<int>(ge.cols), static_cast<int>(n),
                          static_cast<int>(ge.experts), static_cast<int>(used),
                          stream, ue.data, up_e.get()) != 0) {
        return Fail(error_msg, "paired expert projection failed");
      }
    } else if (!Experts(ge, xn.get(), gate_e.get(), n, used, n, error_msg) ||
               !Experts(ue, xn.get(), up_e.get(), n, used, n, error_msg)) {
      return false;
    }
    // Non-Q8_0 down projections read the half-precision sum of each Q8_1
    // block, which overflows for very large rows: keep rows inside range.
    scaled = layer.down_exps.type != GgmlType::kQ8_0;
    if (scaled) {
      rocm::SwigluScaled(gate_e.get(), up_e.get(), expert_scale.get(), n * used,
                         c.expert_ff, stream);
    } else {
      rocm::Swiglu(gate_e.get(), up_e.get(),
                   static_cast<std::size_t>(n) * used * c.expert_ff, stream);
    }
    if (!Experts(layer.down_exps, gate_e.get(), down_e.get(), n * used, 1, n,
                 error_msg)) {
      return false;
    }
  }
  const float* expert_scale_ptr =
      wide ? expert_const.get() : expert_scale.get();
  rocm::MoeCombine(x.get(), down_e.get(), topk.get(),
                   scaled ? expert_scale_ptr : nullptr, sh_out.get(), nullptr,
                   n, used, c.hidden_size, stream);
  return true;
}

bool Session::Impl::Forward(std::span<const std::int32_t> input, bool all_rows,
                            std::uint32_t prefix_end,
                            std::string* error_msg) {
  const Config& c = model->config;
  const DeviceModel& dm = *model->device;
  const auto n = static_cast<std::uint32_t>(input.size());
  std::copy(input.begin(), input.end(), h_tokens);
  h_pos[0] = position;
  logit_rows = all_rows ? n : 1;
  LAGUNA_HIP(hipMemcpyAsync(tokens.get(), h_tokens, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  LAGUNA_HIP(hipMemcpyAsync(pos.get(), h_pos, 4 * sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream));
  rocm::EmbedQ8(dm.token_embd().data, tokens.get(), x.get(), n, c.hidden_size,
                stream);
  std::uint32_t tap_rows = 0, tap_first = 0;
  if (dflash != nullptr) {
    // Only the drafter's sliding window matters: capture the rows ending the
    // session prefix and skip forwards that finish far before it.
    const std::uint32_t end = position + n;
    if (end + model->dflash_window >= prefix_end) {
      tap_first = end > model->dflash_window ? end - model->dflash_window : 0;
      tap_rows = end - tap_first;
      tap_capture_pos = position + tap_first;
    }
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& L = dm.layers()[il];
    rocm::RmsNormRows(x.get(), static_cast<const float*>(L.attn_norm.data),
                      xn.get(), n, c.hidden_size, 1, c.rms_eps, stream);
    if (!Attention(il, n, error_msg))
      return false;
    rocm::RmsNormRows(x.get(), static_cast<const float*>(L.ffn_norm.data),
                      xn.get(), n, c.hidden_size, 1, c.rms_eps, stream);
    if (!(L.is_moe ? Moe(L, n, error_msg) : DenseFfn(L, n, error_msg)))
      return false;
    if (tap_rows != 0 &&
        std::find(model->dflash_taps.begin(), model->dflash_taps.end(),
                  il + 1) != model->dflash_taps.end()) {
      const auto tap = static_cast<std::size_t>(
          std::find(model->dflash_taps.begin(), model->dflash_taps.end(),
                    il + 1) -
          model->dflash_taps.begin());
      rocm::CaptureTap(
          tap_capture.get(), x.get() + static_cast<std::size_t>(tap_first) *
                                        c.hidden_size,
          tap_rows, c.hidden_size,
          static_cast<std::size_t>(model->dflash_taps.size()) * c.hidden_size,
          tap * c.hidden_size, stream);
    }
  }
  tap_capture_rows = tap_rows;
  const std::uint32_t head_rows = all_rows ? n : 1;
  const float* head_in =
      x.get() + (all_rows ? 0 : std::size_t{n - 1} * c.hidden_size);
  rocm::RmsNormRows(head_in, static_cast<const float*>(dm.output_norm().data),
                    last.get(), head_rows, c.hidden_size, 1, c.rms_eps, stream);
  if (!Dense(dm.output(), last.get(), d_logits.get(), head_rows, error_msg))
    return false;
  LAGUNA_HIP(
      hipMemcpyAsync(h_logits, d_logits.get(),
                     std::size_t{head_rows} * c.vocab_size * sizeof(float),
                     hipMemcpyDeviceToHost, stream));
  LAGUNA_HIP(hipStreamSynchronize(stream));
  position += n;
  if (tap_capture_rows != 0) {
    tap_pending = tap_capture_rows;
    tap_pending_pos = tap_capture_pos;
    tap_pending_first = 0;
    tap_capture_rows = 0;
    if (inject_taps && !DflashInject(tap_pending, error_msg))
      return false;
  }
  return true;
}

Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

bool Session::Append(std::span<const std::int32_t> tokens,
                     std::string* error_msg) {
  if (tokens.empty())
    return Fail(error_msg, "Laguna: no tokens to append");
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Laguna: context capacity exceeded");
  }
  const std::uint32_t prefix_end = impl_->position +
                                   static_cast<std::uint32_t>(tokens.size());
  for (std::size_t off = 0; off < tokens.size();) {
    const std::size_t n =
        std::min<std::size_t>(tokens.size() - off, impl_->tile);
    if (!impl_->Forward(tokens.subspan(off, n), false, prefix_end, error_msg))
      return false;
    off += n;
  }
  return true;
}

bool Session::AppendAll(std::span<const std::int32_t> tokens,
                        std::string* error_msg) {
  if (tokens.empty() || tokens.size() > kMaxVerifyRows) {
    return Fail(error_msg, "Laguna: unsupported verification width");
  }
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Laguna: context capacity exceeded");
  }
  impl_->inject_taps = false;
  const bool ok =
      impl_->Forward(tokens, true,
                     impl_->position + static_cast<std::uint32_t>(tokens.size()),
                     error_msg);
  impl_->inject_taps = true;
  return ok;
}

void Session::Rewind(std::uint32_t position) noexcept {
  if (position < impl_->position)
    impl_->position = position;
}

std::span<const float> Session::Logits() const noexcept {
  const auto vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{impl_->logit_rows - 1} * vocab, vocab};
}

std::span<const float> Session::Logits(std::uint32_t row) const noexcept {
  const auto vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{row} * vocab, vocab};
}

std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

std::uint32_t Session::Capacity() const noexcept {
  return impl_->capacity;
}

void Session::Reset() noexcept {
  impl_->position = 0;
  if (impl_->dflash != nullptr)
    impl_->dflash->Reset();
}

hip::QwenDFlashGpuExecutor* Session::DflashExecutor() const noexcept {
  return impl_->dflash.get();
}

bool Session::DflashInject(std::uint32_t rows, std::string* error_msg) {
  return impl_->DflashInject(rows, error_msg);
}

bool Session::Impl::DflashInject(std::uint32_t rows, std::string* error_msg) {
  if (dflash == nullptr)
    return true;
  if (rows > tap_pending) {
    tap_pending = 0;
    return Fail(error_msg, "Laguna: drafter capture out of range");
  }
  if (rows == 0)
    return true;
  if (!dflash->InjectTargetContextDevice(tap_capture.get() +
                                            static_cast<std::size_t>(
                                                tap_pending_first) *
                                                model->dflash_taps.size() *
                                                model->config.hidden_size,
                                        tap_pending_pos, rows)) {
    tap_pending = 0;
    return Fail(error_msg, "Laguna: drafter context injection failed");
  }
  tap_pending -= rows;
  tap_pending_pos += rows;
  tap_pending_first += rows;
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& b : impl_->k_cache)
    total += b.bytes();
  for (const auto& b : impl_->v_cache)
    total += b.bytes();
  return total;
}

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const EngineOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> model(new Model());
  auto impl = std::make_shared<ModelState>();
  impl->options = options;
  std::string err;
  impl->reader = core::GgufReader::OpenFile(model_path, &err);
  if (!impl->reader) {
    Fail(error_msg, "failed to open GGUF: " + err);
    return nullptr;
  }
  const auto config = ParseConfig(*impl->reader, &err);
  if (!config) {
    Fail(error_msg, "failed to parse Laguna config: " + err);
    return nullptr;
  }
  impl->config = *config;
  if (!options.dflash_model_path.empty())
    impl->keep_reader = true;
  impl->device = DeviceModel::Upload(*impl->reader, impl->config, &err);
  if (!impl->device) {
    Fail(error_msg, "failed to upload Laguna weights: " + err);
    return nullptr;
  }
  if (!options.dflash_model_path.empty()) {
    auto dflash_reader =
        core::GgufReader::OpenFile(options.dflash_model_path, &err);
    if (!dflash_reader) {
      Fail(error_msg, "failed to open the drafter: " + err);
      return nullptr;
    }
    std::string dflash_err;
    impl->dflash_model = hip::QwenDFlashGpuModel::Create(
        std::move(dflash_reader), impl->config.hidden_size,
        impl->config.vocab_size, impl->config.num_layers,
        TensorRef(impl->device->token_embd()),
        TensorRef(impl->device->output()), &dflash_err);
    if (!impl->dflash_model) {
      Fail(error_msg, "failed to load the drafter: " + dflash_err);
      return nullptr;
    }
    const auto& dcfg = impl->dflash_model->GetDFlashConfig();
    impl->dflash_taps = dcfg.target_layer_ids;
    impl->drafter_block =
        std::min<std::uint32_t>(dcfg.block_size, Session::kMaxVerifyRows);
    impl->dflash_window = dcfg.sliding_window == 0
                              ? options.max_context
                              : std::min<std::uint32_t>(
                                    dcfg.sliding_window, options.max_context);
    if (!impl->dflash_taps.empty() &&
        impl->dflash_taps.back() >= impl->config.num_layers) {
      Fail(error_msg, "the drafter taps exceed the Laguna layer count");
      return nullptr;
    }
  }
  // Release the GGUF mmap now that all weights are on the device: keeping it
  // alive pins ~model-size host pages and trips the SVM/resident-memory limit.
  if (!impl->keep_reader)
    impl->reader.reset();
  model->state_ = std::move(impl);
  return model;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  std::unique_ptr<Session> session(new Session());
  session->impl_->model = state_;
  if (!session->impl_->Init(max_context, error_msg))
    return nullptr;
  if (HasDflash()) {
    session->impl_->dflash = hip::QwenDFlashGpuExecutor::Create(
        state_->dflash_model,
        std::min<std::uint32_t>(max_context, state_->options.max_context),
        error_msg);
    if (!session->impl_->dflash) {
      Fail(error_msg, "Laguna: drafter executor creation failed");
      return nullptr;
    }
    const std::size_t taps = state_->dflash_taps.size();
    if (taps != 0 &&
        !session->impl_->tap_capture.Allocate(std::size_t{state_->dflash_window +
                                                         Session::kMaxVerifyRows} *
                                              taps * state_->config.hidden_size)) {
      Fail(error_msg, "Laguna: drafter capture allocation failed");
      return nullptr;
    }
  }
  const std::int32_t warm[kWarmTokens] = {};
  for (const std::uint32_t width : {1U, kWarmTokens}) {
    if (width > max_context)
      break;
    if (!session->impl_->Forward({warm, width}, false, width, error_msg)) {
      return nullptr;
    }
    session->impl_->position = 0;
    if (session->impl_->dflash != nullptr) {
      session->impl_->dflash->Reset();
      session->impl_->tap_pending = 0;
      session->impl_->tap_pending_pos = 0;
      session->impl_->tap_pending_first = 0;
      session->impl_->tap_capture_rows = 0;
    }
  }
  return session;
}

const Config& Model::config() const noexcept {
  return state_->config;
}

bool Model::HasDflash() const noexcept {
  return state_->dflash_model != nullptr;
}

std::uint32_t Model::DrafterBlockSize() const noexcept {
  return state_->drafter_block;
}

std::size_t Model::ResidentBytes() const noexcept {
  return state_->device->resident_bytes();
}

}  // namespace gufo::models::laguna::gpu
