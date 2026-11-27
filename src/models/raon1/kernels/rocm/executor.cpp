#include "src/models/raon1/kernels/rocm/executor.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/raon1/kernels/rocm/raon.hpp"

namespace qfn = gufo::models::qwen38_flash_next::rocm;

// The current bank's buffers; a pair run gives each bank its own host thread.
#define sc_ (B().sc)
#define pf_ (B().pf)
#define mla_ (B().mla)
#define xh_ (B().xh)
#define mmq_scratch_ (B().mmq)
#define mmq_scratch_bytes_ (B().mmq_bytes)
#define mla_cap_ (B().mla_cap)
#define stream_ (B().stream)

namespace gufo::models::raon1::rocm {

thread_local Executor::Bank* Executor::cur_bank_ = nullptr;

namespace {

using gufo::core::GgmlType;

constexpr std::size_t kUploadChunk = 64U << 20;

std::string P(std::uint32_t layer, const char* name) {
  return "blk." + std::to_string(layer) + "." + name;
}

int TypeCode(GgmlType t) {
  switch (t) {
    case GgmlType::kQ8_0:
      return 8;
    case GgmlType::kQ4_K:
      return 12;
    case GgmlType::kQ6_K:
      return 14;
    case GgmlType::kQ5_0:
      return 6;
    default:
      return -1;
  }
}

float Bf16(std::uint16_t h) {
  std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

float Half(float bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits);
  const std::uint32_t sign = (u & 0x8000U) << 16U;
  const std::uint32_t exp = (u >> 10U) & 0x1FU;
  const std::uint32_t man = u & 0x3FFU;
  std::uint32_t f = sign;
  if (exp == 0) {
    if (man != 0) {
      std::uint32_t e = 113U;
      std::uint32_t m = man;
      while ((m & 0x400U) == 0) {
        m <<= 1U;
        --e;
      }
      f |= (e << 23U) | ((m & 0x3FFU) << 13U);
    }
  } else if (exp == 0x1FU) {
    f |= 0x7F800000U | (man << 13U);
  } else {
    f |= ((exp + 112U) << 23U) | (man << 13U);
  }
  float out = 0.0F;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

void DequantRowHost(const core::GgufTensorInfo& t, std::uint64_t row,
                    float* out) {
  std::uint64_t rows = 1;
  for (std::size_t d = 1; d < t.dimensions.size(); ++d) {
    rows *= std::max<std::uint64_t>(t.dimensions[d], 1);
  }
  const std::size_t rb = static_cast<std::size_t>(t.size_bytes / rows);
  const char* src = static_cast<const char*>(t.data) + row * rb;
  const std::size_t k = t.dimensions[0];
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out, k);
      break;
    case GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, out, k);
      break;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, out, k);
      break;
    case GgmlType::kQ5_0: {
      const auto* p = reinterpret_cast<const std::uint8_t*>(src);
      for (std::size_t b = 0; b < k / 32; ++b) {
        std::uint16_t dh = 0;
        std::memcpy(&dh, p + b * 22, 2);
        std::uint32_t qh = 0;
        std::memcpy(&qh, p + b * 22 + 2, 4);
        const float d = Half(dh);
        for (std::size_t j = 0; j < 16; ++j) {
          const std::uint32_t q0 =
              (p[b * 22 + 6 + j] & 0x0FU) | (((qh >> j) & 1U) << 4U);
          const std::uint32_t q1 =
              (p[b * 22 + 6 + j] >> 4U) | (((qh >> (j + 16)) & 1U) << 4U);
          out[b * 32 + 2 * j] = d * (static_cast<float>(q0) - 16.0F);
          out[b * 32 + 2 * j + 1] = d * (static_cast<float>(q1) - 16.0F);
        }
      }
      break;
    }
    default:
      std::abort();
  }
}

}  // namespace

Executor::Scratch::~Scratch() {
  const float* ptrs[] = {
      x,    y,          pre,     post, comb,  attn_in, block, ffn_in, ffn_out,
      head, logits,     q,       k,    v,     fa,      gproj, beta,   delta,
      g2,   decay,      rda_out, qr,   mla_q, mla_qe,  kv,    qabs,   ctx,
      gate, moe_logits, weights, gf,   uf,    act,     dpart, parts,  mla_part};
  for (const float* p : ptrs)
    hipFree(const_cast<float*>(p));
  hipFree(ids);
  hipFree(argmax);
}

Executor::Session::~Session() {
  for (float* p : snap_conv) {
    if (p != nullptr)
      (void)hipFree(p);
  }
  for (float* p : snap_state) {
    if (p != nullptr)
      (void)hipFree(p);
  }
  for (auto& l : layers) {
    hipFree(l.conv);
    hipFree(l.state);
    hipFree(l.lat);
    hipFree(l.extra);
    if (l.pool != nullptr)
      hipFree(l.pool);
  }
}

Executor::~Executor() {
  for (auto& it : tensors_)
    hipFree(it.second.data);
  for (auto& bank : banks_) {
    for (auto& s : bank->slots) {
      if (s.helper.joinable())
        s.helper.join();
    }
    if (bank->mmq != nullptr)
      hipFree(bank->mmq);
    if (bank->stream != nullptr)
      hipStreamDestroy(bank->stream);
  }
}

std::unique_ptr<Executor> Executor::Create(const std::string& path,
                                           const ExecutorOptions& options,
                                           std::string* error) {
  // Blocking sync: host threads sleep in stream/event waits instead of
  // spinning a core each (decode syncs every token, prefill every layer).
  if (std::getenv("GUFO_RAON1_SPIN") == nullptr)
    (void)hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
  auto e = std::unique_ptr<Executor>(new Executor());
  e->options_ = options;
  e->reader_ = core::GgufReader::OpenFile(path, error);
  if (!e->reader_)
    return nullptr;
  auto cfg = ParseConfig(*e->reader_, error);
  if (!cfg)
    return nullptr;
  e->config_ = *cfg;
  e->layers_ = options.layers != 0
                   ? std::min(options.layers, e->config_.num_layers)
                   : e->config_.num_layers;
  if (const char* v = std::getenv("GUFO_RAON1_SPARSE")) {
    e->sparse_.on = true;
    int k = 0, w = 0, d = 0;
    const int got = std::sscanf(v, "%d,%d,%d", &k, &w, &d);
    if (got >= 1 && k > 0)
      e->sparse_.k = std::min(k, 128);
    if (got >= 2 && w > 0)
      e->sparse_.window = w;
    if (got >= 3 && d > 0)
      e->sparse_.min_depth = d;
  }
  e->banks_.push_back(std::make_unique<Bank>());
  if (hipStreamCreate(&e->banks_[0]->stream) != hipSuccess) {
    if (error)
      *error = "hipStreamCreate failed";
    return nullptr;
  }
  if (!e->Resolve(error) || !e->AllocScratch(error))
    return nullptr;
  if (!e->AllocPrefill(true, error) || !e->BuildHalfWeights(error))
    return nullptr;
  if (!options.dflash_layers.empty()) {
    if (!e->AllocCapture(*e->banks_[0], error))
      return nullptr;
    e->tap_input_ = std::getenv("GUFO_RAON1_TAP_INPUT") != nullptr;
  }
  if (!e->SetupNpu(error))
    return nullptr;
  return e;
}

const DeviceTensor* Executor::Find(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

std::size_t Executor::ResidentBytes() const noexcept {
  std::size_t bytes = mmq_scratch_bytes_;
  for (const auto& it : tensors_)
    bytes += it.second.bytes;
  return bytes;
}

bool Executor::UploadImpl(const std::string& name,
                          const core::GgufTensorInfo& t, std::string* error) {
  const auto cols = static_cast<std::uint32_t>(t.dimensions[0]);
  std::uint64_t count = 1;
  for (std::size_t d = 1; d < t.dimensions.size(); ++d)
    count *= t.dimensions[d];
  const auto rows = static_cast<std::uint32_t>(count);
  const std::size_t row_bytes = static_cast<std::size_t>(t.size_bytes / count);
  DeviceTensor d;
  d.cols = cols;
  d.rows = rows;
  d.row_bytes = row_bytes;
  d.bytes = t.size_bytes;

  const int code = TypeCode(t.type);
  static const bool use_f16 = std::getenv("GUFO_RAON1_GPU_F16") != nullptr;
  const bool convert_f16 = use_f16 && code >= 0 && name.rfind("blk.", 0) == 0 &&
                           name.find("exps") == std::string::npos;
  static const bool f16_trace =
      std::getenv("GUFO_RAON1_GPU_F16_TRACE") != nullptr;
  if (f16_trace) {
    std::fprintf(stderr,
                 "UPLOAD %s t=%d code=%d f16=%d src=%zu ptr=%p "
                 "dims=%zu,%zu,%zu,%zu rb=%zu\n",
                 name.c_str(), static_cast<int>(t.type), code,
                 static_cast<int>(convert_f16),
                 static_cast<std::size_t>(t.size_bytes),
                 static_cast<const void*>(t.data),
                 t.dimensions.size() > 0 ? t.dimensions[0] : 0,
                 t.dimensions.size() > 1 ? t.dimensions[1] : 0,
                 t.dimensions.size() > 2 ? t.dimensions[2] : 0,
                 t.dimensions.size() > 3 ? t.dimensions[3] : 0, row_bytes);
  }
  if (convert_f16) {
    d.type = 15;
    d.row_bytes = cols * sizeof(__half);
    d.bytes = static_cast<std::size_t>(count) * cols * sizeof(__half);
    if (hipMalloc(&d.data, d.bytes) != hipSuccess) {
      if (error)
        *error = "hipMalloc failed for " + name;
      return false;
    }
    constexpr std::size_t kRowsPerChunk = 4096;
    std::vector<float> row(cols);
    std::vector<__half> chunk(kRowsPerChunk * cols);
    for (std::uint64_t r0 = 0; r0 < count; r0 += kRowsPerChunk) {
      const std::size_t rows = static_cast<std::size_t>(
          std::min<std::uint64_t>(kRowsPerChunk, count - r0));
      for (std::size_t r = 0; r < rows; ++r) {
        DequantRowHost(t, r0 + r, row.data());
        float max_abs = 0.0F;
        for (const float v : row) {
          if (!std::isfinite(v)) {
            if (f16_trace) {
              std::fprintf(stderr, "NONFIN %s row=%llu\n", name.c_str(),
                           static_cast<unsigned long long>(r0 + r));
            }
            break;
          }
          max_abs = std::max(max_abs, std::abs(v));
        }
        if (f16_trace && max_abs > 60000.0F) {
          std::fprintf(stderr, "BIG %s row=%llu max=%g\n", name.c_str(),
                       static_cast<unsigned long long>(r0 + r), max_abs);
        }
        __half* dst = chunk.data() + r * cols;
        for (std::uint32_t i = 0; i < cols; ++i)
          dst[i] = __float2half_rn(row[i]);
      }
      const std::size_t offset = static_cast<std::size_t>(r0) * cols;
      if (hipMemcpy(static_cast<char*>(d.data) + offset * sizeof(__half),
                    chunk.data(), rows * cols * sizeof(__half),
                    hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(d.data);
        if (error)
          *error = "upload failed for " + name;
        return false;
      }
    }
  } else if (code >= 0) {
    d.type = code;
    if (hipMalloc(&d.data, t.size_bytes + 4096) != hipSuccess) {
      if (error)
        *error = "hipMalloc failed for " + name;
      return false;
    }
    if (hipMemcpy(d.data, t.data, t.size_bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
      hipFree(d.data);
      if (error)
        *error = "upload failed for " + name;
      return false;
    }
  } else if (t.type == GgmlType::kF32 || t.type == GgmlType::kBF16) {
    d.type = 0;
    const std::size_t elems = static_cast<std::size_t>(count) * cols;
    std::vector<float> host(elems);
    if (t.type == GgmlType::kF32) {
      std::memcpy(host.data(), t.data, elems * sizeof(float));
    } else {
      const auto* h = static_cast<const std::uint16_t*>(t.data);
      for (std::size_t i = 0; i < elems; ++i)
        host[i] = Bf16(h[i]);
    }
    if (hipMalloc(&d.data, host.size() * sizeof(float)) != hipSuccess ||
        hipMemcpy(d.data, host.data(), host.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      hipFree(d.data);
      if (error)
        *error = "upload failed for " + name;
      return false;
    }
  } else {
    if (error)
      *error = "unsupported type " + std::to_string(static_cast<int>(t.type)) +
               " for " + name;
    return false;
  }
  tensors_.emplace(name, d);
  if (f16_trace) {
    std::fprintf(stderr, "DEV %s ptr=%p bytes=%zu\n", name.c_str(), d.data,
                 d.bytes);
  }
  return true;
}

Executor::Prefill::~Prefill() {
  void* ptrs[] = {x16,     act16,   down_e,   gate_e,     up_e,
                  counts,  bounds,  cursors,  rows_token, rows_slot,
                  tiles48, tiles64, tiles128, situ};
  for (void* p : ptrs) {
    if (p != nullptr)
      (void)hipFree(p);
  }
}

bool Executor::AllocPrefill(bool first, std::string* error) {
  for (std::uint32_t l = config_.leading_dense; l < layers_; ++l) {
    const auto* g = Find(P(l, "ffn_gate_exps.weight"));
    const auto* u = Find(P(l, "ffn_up_exps.weight"));
    const auto* d = Find(P(l, "ffn_down_exps.weight"));
    if (g == nullptr || u == nullptr || d == nullptr || g->type != 12 ||
        u->type != 12 || (d->type != 8 && d->type != 6)) {
      return true;
    }
  }
  pf_ = std::make_unique<Prefill>();
  const std::size_t n = kMaxBatch;
  const std::size_t slots = n * config_.expert_used;
  const std::size_t experts = config_.expert_count;
  const std::size_t rows = qfn::RoutedCompactRows(slots, experts);
  pf_->tile_cap = (slots + 15 * experts) / 48 + experts;
  auto alloc = [&](void** p, std::size_t bytes) {
    return hipMalloc(p, bytes) == hipSuccess;
  };
  const std::size_t ff = config_.expert_ff;
  if (!alloc(reinterpret_cast<void**>(&pf_->x16),
             n * config_.hidden_size * 2) ||
      !alloc(reinterpret_cast<void**>(&pf_->act16), slots * ff * 2) ||
      !alloc(reinterpret_cast<void**>(&pf_->down_e),
             slots * config_.hidden_size * 2) ||
      !alloc(reinterpret_cast<void**>(&pf_->gate_e), slots * ff * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->up_e), slots * ff * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->counts), experts * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->bounds), (experts + 1) * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->cursors), experts * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->rows_token), rows * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->rows_slot), rows * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->tiles48), pf_->tile_cap * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->tiles64), pf_->tile_cap * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->tiles128), pf_->tile_cap * 4) ||
      !alloc(reinterpret_cast<void**>(&pf_->situ), 2 * sizeof(float))) {
    if (error)
      *error = "prefill scratch alloc failed";
    return false;
  }
  {
    const float pair[2] = {config_.situ_beta, config_.situ_linear_beta};
    hipMemcpy(pf_->situ, pair, sizeof(pair), hipMemcpyHostToDevice);
  }
  pf_->host_counts.resize(experts);
  pf_->host_tiles.resize(pf_->tile_cap * 2);
  for (std::uint32_t l = config_.leading_dense; first && l < layers_; ++l) {
    const auto* w = Find(P(l, "ffn_down_exps.weight"));
    if (w == nullptr || w->type != 6)
      continue;
    const std::size_t blocks = w->bytes / 22;
    void* q8 = nullptr;
    if (hipMalloc(&q8, blocks * 34) != hipSuccess) {
      if (error)
        *error = "Q8 down copy alloc failed";
      return false;
    }
    Q5ToQ8(w->data, q8, blocks, stream_);
    down_q8_[l] = q8;
  }
  mla_ = std::make_unique<MlaPrefill>();
  const std::size_t heads = config_.num_heads;
  auto alloc2 = [&](void** p, std::size_t bytes) {
    return hipMalloc(p, bytes) == hipSuccess;
  };
  mla_cap_ =
      std::min<std::uint32_t>(options_.max_context, first ? 131072U : 32768U);
  if (!alloc2(reinterpret_cast<void**>(&mla_->lat16), kMlaBlock * 512 * 2) ||
      !alloc2(reinterpret_cast<void**>(&mla_->knope),
              kMlaBlock * heads * 128 * 4) ||
      !alloc2(reinterpret_cast<void**>(&mla_->vexp),
              kMlaBlock * heads * 128 * 4) ||
      !alloc2(reinterpret_cast<void**>(&mla_->k16),
              static_cast<std::size_t>(mla_cap_) * heads * 192 * 2) ||
      !alloc2(reinterpret_cast<void**>(&mla_->v16),
              static_cast<std::size_t>(mla_cap_) * heads * 128 * 2) ||
      !alloc2(reinterpret_cast<void**>(&mla_->q16),
              static_cast<std::size_t>(kMaxBatch) * heads * 192 * 2) ||
      !alloc2(reinterpret_cast<void**>(&mla_->scores),
              (static_cast<std::size_t>(options_.max_context) / 64 + 2) * 4) ||
      !alloc2(reinterpret_cast<void**>(&mla_->sel), 128 * 4) ||
      !alloc2(reinterpret_cast<void**>(&mla_->kpos),
              (static_cast<std::size_t>(kMlaBlock) * 8) * 4) ||
      !alloc2(reinterpret_cast<void**>(&mla_->lat_g),
              static_cast<std::size_t>(kMlaBlock) * 8 * 512 * 4) ||
      !alloc2(reinterpret_cast<void**>(&mla_->extra_g),
              static_cast<std::size_t>(kMlaBlock) * 8 * 64 * 4)) {
    if (error)
      *error = "MLA prefill scratch alloc failed";
    return false;
  }
  if (!first) {
    mla_->kbt = banks_[0]->mla->kbt;
    mla_->vbh = banks_[0]->mla->vbh;
  } else
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!config_.IsMla(l))
        continue;
      const auto* kb = Find(P(l, "attn_k_b.weight"));
      const auto* vb = Find(P(l, "attn_v_b.weight"));
      if (kb == nullptr || vb == nullptr) {
        if (error)
          *error = "missing MLA k_b/v_b";
        return false;
      }
      __half* kbt = nullptr;
      __half* vbh = nullptr;
      __half* tmp = nullptr;
      if (!alloc2(reinterpret_cast<void**>(&kbt), heads * 128 * 512 * 2) ||
          !alloc2(reinterpret_cast<void**>(&vbh), heads * 128 * 512 * 2) ||
          !alloc2(reinterpret_cast<void**>(&tmp), heads * 128 * 512 * 2)) {
        if (error)
          *error = "MLA weight copy alloc failed";
        return false;
      }
      qfn::DequantRowsToHalf(kb->data, static_cast<std::uint32_t>(kb->type),
                             128, heads * 512, tmp, stream_);
      TransposeKb(tmp, kbt, stream_);
      qfn::DequantRowsToHalf(vb->data, static_cast<std::uint32_t>(vb->type),
                             512, heads * 128, vbh, stream_);
      hipStreamSynchronize(stream_);
      hipFree(tmp);
      mla_->kbt[l] = kbt;
      mla_->vbh[l] = vbh;
    }
  if (first) {
    constexpr std::uint32_t kFullKvTokens = 98304U;
    full_cap_ = std::min<std::uint32_t>(options_.max_context, kFullKvTokens);
    full_.assign(layers_, FullKv{});
  }
  hipStreamSynchronize(stream_);
  return true;
}

bool Executor::BuildHalfWeights(std::string* error) {
  std::size_t max_k = 0;
  std::size_t total = 0;
  for (const auto& [name, w] : tensors_) {
    if (w.type != 8 && w.type != 12 && w.type != 14)
      continue;
    if (w.cols % 64 != 0 || name.find("exps") != std::string::npos ||
        name.find("_k_b") != std::string::npos ||
        name.find("_v_b") != std::string::npos || name.rfind("blk.", 0) != 0) {
      continue;
    }
    const std::size_t rows = w.bytes / w.row_bytes;
    __half* dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&dev), rows * w.cols * 2) !=
        hipSuccess) {
      if (error)
        *error = "F16 weight alloc failed";
      return false;
    }
    qfn::DequantRowsToHalf(w.data, static_cast<std::uint32_t>(w.type), w.cols,
                           rows, dev, banks_[0]->stream);
    half_w_[w.data] = dev;
    max_k = std::max<std::size_t>(max_k, w.cols);
    total += rows * w.cols * 2;
  }
  hipStreamSynchronize(banks_[0]->stream);
  half_max_k_ = max_k;
  if (!AllocXh(error))
    return false;
  if (std::getenv("GUFO_RAON1_VERBOSE") != nullptr) {
    std::fprintf(stderr, "raon1: %zu F16 dense weights, %.2f GiB\n",
                 half_w_.size(), static_cast<double>(total) / (1ULL << 30));
  }
  return true;
}

bool Executor::AllocXh(std::string* error) {
  if (hipMalloc(reinterpret_cast<void**>(&xh_),
                static_cast<std::size_t>(kMaxBatch) * half_max_k_ * 2) !=
      hipSuccess) {
    if (error)
      *error = "F16 activation alloc failed";
    return false;
  }
  return true;
}

bool Executor::SetupNpu(std::string* error) {
  const char* dir = std::getenv("GUFO_NPU_DIR");
  if (dir == nullptr || dir[0] == '\0' || std::getenv("GUFO_RAON1_NO_NPU"))
    return true;
  const std::uint32_t k = config_.hidden_size;
  if (config_.KdaDim() != static_cast<std::uint32_t>(kNpuSlices) * 1024U)
    return true;
  qfn::NpuGemm::Config cfg;
  cfg.xclbin =
      std::string(dir) + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
  cfg.insts = std::string(dir) + "/runtime_mk/insts_" +
              std::to_string(kMaxBatch) + "x" + std::to_string(k) +
              "x1024_128x64x128.txt";
  cfg.tokens = kMaxBatch;
  cfg.k = k;
  std::string err;
  npu_ = qfn::NpuGemm::Create(cfg, &err);
  if (npu_ == nullptr) {
    std::fprintf(stderr, "raon1: NPU off: %s\n", err.c_str());
    return true;
  }
  npu_layers_.assign(layers_, {});
  const char* names[kNpuProj] = {"attn_q.weight", "attn_k.weight",
                                 "attn_v.weight", "ssm_g.weight"};
  std::uint32_t staged = 0;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!config_.IsRda(l))
      continue;
    bool ok = true;
    std::vector<int> ids;
    for (int t = 0; ok && t < kNpuProj; ++t) {
      const auto* w = Find(P(l, names[t]));
      if (w == nullptr || w->cols != k || w->rows != config_.KdaDim() ||
          (w->type != 14 && w->type != 8 && w->type != 12)) {
        ok = false;
        break;
      }
      for (int sl = 0; sl < kNpuSlices; ++sl) {
        const int id = npu_->NewWeightBuffer(npu_->BBytes(), &err);
        if (id < 0) {
          ok = false;
          break;
        }
        qfn::NpuEncodeWeightsQuant(w->data, static_cast<std::uint32_t>(w->type),
                                   k, sl * 1024U, 1024U, k, 0, k,
                                   npu_->Device(id), banks_[0]->stream);
        ids.push_back(id);
      }
    }
    if (ok) {
      npu_layers_[l] = std::move(ids);
      ++staged;
    }
  }
  npu_lin_.assign(layers_, {});
  const char* parts_env = std::getenv("GUFO_RAON1_NPU_PARTS");
  const std::string parts = parts_env != nullptr ? parts_env : "gate,shared";
  const bool whole = !parts.empty();
  const bool want_out = parts.find("out") != std::string::npos;
  const bool want_gate = parts.find("gate") != std::string::npos;
  const bool want_shared = parts.find("shared") != std::string::npos;
  std::string lin_err;
  for (std::uint32_t l = 0; whole && l < layers_; ++l) {
    auto& nl = npu_lin_[l];
    const auto* wo = Find(P(l, "attn_output.weight"));
    if (want_out && wo != nullptr && wo->cols % k == 0 && wo->rows == k)
      NpuLinSetup(nl.out, *wo, wo->cols, static_cast<int>(k), &lin_err);
    if (want_gate && config_.IsMla(l)) {
      const auto* wg = Find(P(l, "attn_gate.weight"));
      if (wg != nullptr && wg->cols == k)
        NpuLinSetup(nl.gate, *wg, k, static_cast<int>(wg->rows), &lin_err);
    }
    if (want_shared && !config_.IsDense(l)) {
      const auto* sg = Find(P(l, "ffn_gate_shexp.weight"));
      const auto* su = Find(P(l, "ffn_up_shexp.weight"));
      if (sg != nullptr && su != nullptr && sg->cols == k) {
        NpuLinSetup(nl.shg, *sg, k, static_cast<int>(sg->rows), &lin_err);
        NpuLinSetup(nl.shu, *su, k, static_cast<int>(su->rows), &lin_err);
      }
    }
  }
  hipStreamSynchronize(banks_[0]->stream);
  for (const auto& ids : npu_layers_)
    for (const int id : ids)
      npu_->FinalizeWeights(id);
  for (const auto& nl : npu_lin_)
    for (const NpuLin* lin : {&nl.out, &nl.gate, &nl.shg, &nl.shu})
      for (const int id : lin->w)
        npu_->FinalizeWeights(id);
  if (!AllocNpuBank(*banks_[0], error))
    return false;
  std::fprintf(stderr, "raon1: NPU q/k/v/g of %u RDA layers\n", staged);
  return true;
}

bool Executor::NpuLinSetup(NpuLin& lin, const DeviceTensor& w,
                           std::uint32_t k_total, int n_out,
                           std::string* error) {
  const std::uint32_t kh = config_.hidden_size;
  if (k_total % kh != 0 || (w.type != 14 && w.type != 8 && w.type != 12)) {
    return false;
  }
  lin.halves = static_cast<int>(k_total / kh);
  lin.slices = (n_out + 1023) / 1024;
  lin.n_out = n_out;
  lin.w.clear();
  std::string err;
  for (int h = 0; h < lin.halves; ++h) {
    for (int sl = 0; sl < lin.slices; ++sl) {
      const int id = npu_->NewWeightBuffer(npu_->BBytes(), &err);
      if (id < 0) {
        lin.w.clear();
        if (error)
          *error = err;
        return false;
      }
      const std::uint32_t row0 = static_cast<std::uint32_t>(sl) * 1024U;
      const std::uint32_t valid = std::min<std::uint32_t>(
          1024U, static_cast<std::uint32_t>(n_out) - row0);
      const std::uint32_t cols = (valid + 127U) / 128U * 128U;
      const void* src = w.data;
      std::uint32_t r0 = row0;
      void* tmp = nullptr;
      if (cols != valid) {
        const std::size_t rb = w.row_bytes;
        if (hipMalloc(&tmp, rb * cols) != hipSuccess)
          return false;
        hipMemsetAsync(tmp, 0, rb * cols, banks_[0]->stream);
        hipMemcpyAsync(tmp, static_cast<const char*>(w.data) + row0 * rb,
                       rb * valid, hipMemcpyDeviceToDevice, banks_[0]->stream);
        src = tmp;
        r0 = 0;
      }
      qfn::NpuEncodeWeightsQuant(src, static_cast<std::uint32_t>(w.type),
                                 k_total, r0, cols, kh, h * kh, k_total,
                                 npu_->Device(id), banks_[0]->stream);
      if (tmp != nullptr) {
        hipStreamSynchronize(banks_[0]->stream);
        hipFree(tmp);
      }
      lin.w.push_back(id);
    }
  }
  return true;
}

Executor::NpuOp Executor::NpuLinBegin(const NpuLin& lin, int n, std::size_t ldx,
                                      const __half* x16) {
  Bank& b = B();
  const std::uint32_t kh = config_.hidden_size;
  std::vector<NpuRun> runs;
  for (int h = 0; h < lin.halves; ++h) {
    const int a = h == 0 ? b.npu_a : b.npu_a2;
    qfn::NpuEncodeActivations(x16, ldx, n, kh, h * kh,
                              static_cast<std::uint32_t>(lin.halves) * kh,
                              npu_->Device(a), stream_, false);
    runs.push_back({a, lin.w.data() + h * lin.slices,
                    b.npu_c.data() + h * lin.slices, lin.slices});
  }
  return NpuStart(runs);
}

void Executor::NpuLinEnd(const NpuLin& lin, const NpuOp& op, float* out,
                         std::size_t pitch, int n) {
  Bank& b = B();
  NpuFinish(op);
  for (int h = 0; h < lin.halves; ++h) {
    for (int sl = 0; sl < lin.slices; ++sl) {
      const std::uint32_t valid = std::min<std::uint32_t>(
          1024U, static_cast<std::uint32_t>(lin.n_out) - sl * 1024U);
      qfn::NpuDecodeOutput(npu_->Staged(stream_, b.npu_c[h * lin.slices + sl]),
                           out, n, npu_->n(), pitch, sl * npu_->n(), h > 0,
                           stream_, valid == 1024U ? 0U : valid);
    }
  }
}

bool Executor::AllocNpuBank(Bank& bank, std::string* error) {
  if (npu_ == nullptr)
    return true;
  std::string err;
  bank.npu_a = npu_->NewBuffer(npu_->ABytes(), &err);
  bank.npu_a2 = npu_->NewBuffer(npu_->ABytes(), &err);
  for (auto& s : bank.slots) {
    hipEventCreateWithFlags(&s.event, hipEventDisableTiming);
    if (hipHostMalloc(reinterpret_cast<void**>(&s.flag), sizeof(int),
                      hipHostMallocCoherent | hipHostMallocMapped) !=
        hipSuccess) {
      if (error)
        *error = "NPU flag alloc failed";
      return false;
    }
    *s.flag = 0;
  }
  for (int i = 0; bank.npu_a >= 0 && i < kNpuProj * kNpuSlices; ++i) {
    const int id = npu_->NewBuffer(npu_->CBytes(), &err);
    if (id < 0) {
      bank.npu_a = -1;
      break;
    }
    bank.npu_c.push_back(id);
  }
  if (bank.npu_a < 0 && error)
    *error = "NPU buffers: " + err;
  return bank.npu_a >= 0;
}

Executor::NpuOp Executor::NpuStart(const std::vector<NpuRun>& runs) {
  Bank& b = B();
  const int slot = b.next_slot;
  b.next_slot = (b.next_slot + 1) % Bank::kNpuSlots;
  auto& s = b.slots[slot];
  if (s.helper.joinable())
    s.helper.join();
  ++s.epoch;
  hipEventRecord(s.event, stream_);
  std::vector<NpuRun> copy = runs;
  qfn::NpuGemm* npu = npu_.get();
  std::mutex* mutex = &npu_mutex_;
  hipEvent_t event = s.event;
  int* flag = s.flag;
  const int epoch = s.epoch;
  s.helper = std::thread([=] {
    hipEventSynchronize(event);
    std::uint64_t ticket = 0;
    {
      std::lock_guard<std::mutex> lock(*mutex);
      for (const auto& r : copy)
        ticket = npu->SubmitHost(r.a, r.w, r.c, r.count);
    }
    npu->WaitHost(ticket);
    __atomic_store_n(flag, epoch, __ATOMIC_RELEASE);
  });
  return {slot, epoch};
}

void Executor::NpuFinish(const NpuOp& op) {
  WaitFlag(B().slots[op.slot].flag, op.epoch, stream_);
}

bool Executor::NpuQkvgBegin(std::uint32_t l, int n, NpuOp* op) {
  Bank& b = B();
  const std::uint32_t H = config_.hidden_size;
  qfn::NpuEncodeActivations(sc_->a16, H, n, H, 0, H, npu_->Device(b.npu_a),
                            stream_, false);
  *op = NpuStart({{b.npu_a, npu_layers_[l].data(), b.npu_c.data(),
                   kNpuProj * kNpuSlices}});
  return true;
}

void Executor::NpuQkvgEnd(const NpuOp& op, int n) {
  Bank& b = B();
  NpuFinish(op);
  float* outs[kNpuProj] = {sc_->q, sc_->k, sc_->v, sc_->g2};
  const std::size_t inner = config_.KdaDim();
  for (int t = 0; t < kNpuProj; ++t) {
    for (int sl = 0; sl < kNpuSlices; ++sl) {
      qfn::NpuDecodeOutput(npu_->Staged(stream_, b.npu_c[t * kNpuSlices + sl]),
                           outs[t], n, npu_->n(), inner, sl * npu_->n(), false,
                           stream_, 0);
    }
  }
}

bool Executor::AllocCapture(Bank& bank, std::string* error) {
  const std::size_t width = options_.dflash_layers.size() * config_.HcDim();
  if (hipMalloc(reinterpret_cast<void**>(&bank.capture),
                static_cast<std::size_t>(kMaxBatch) * width * sizeof(float)) !=
      hipSuccess) {
    if (error)
      *error = "drafter capture alloc failed";
    return false;
  }
  return true;
}

bool Executor::AddBank(std::string* error) {
  auto bank = std::make_unique<Bank>();
  Bank* raw = bank.get();
  banks_.push_back(std::move(bank));
  if (hipStreamCreateWithFlags(&raw->stream, hipStreamNonBlocking) !=
      hipSuccess) {
    if (error)
      *error = "hipStreamCreate failed";
    return false;
  }
  Bank* previous = cur_bank_;
  cur_bank_ = raw;
  const bool ok =
      AllocScratch(error) && AllocPrefill(false, error) && AllocXh(error) &&
      (options_.dflash_layers.empty() || AllocCapture(*raw, error)) &&
      AllocNpuBank(*raw, error);
  cur_bank_ = previous;
  return ok;
}

bool Executor::PrefillMoe(std::uint32_t l, int n, std::string* error) {
  const std::uint32_t H = config_.hidden_size;
  const std::uint32_t use = config_.expert_used;
  const std::uint32_t experts = config_.expert_count;
  const std::uint32_t ff = config_.expert_ff;
  const std::size_t slots = static_cast<std::size_t>(n) * use;
  const __half* x16 = sc_->f16;
  qfn::ExpertCounts(sc_->ids, pf_->counts, n, experts, use, stream_);
  const std::uint32_t n48 =
      static_cast<std::uint32_t>((slots + 15U * experts) / 48 + experts);
  const std::uint32_t n64 =
      static_cast<std::uint32_t>((slots + 15U * experts) / 64 + experts);
  const std::uint32_t n128 =
      static_cast<std::uint32_t>((slots + 15U * experts) / 128 + experts);
  BuildTiles(pf_->counts, static_cast<int>(experts), pf_->tiles48,
             static_cast<int>(n48), pf_->tiles64, static_cast<int>(n64),
             pf_->tiles128, static_cast<int>(n128), stream_);
  qfn::RoutedCompact(sc_->ids, pf_->counts, pf_->bounds, pf_->cursors,
                     pf_->rows_token, pf_->rows_slot, n, use, experts, stream_);
  const auto& wg = *Find(P(l, "ffn_gate_exps.weight"));
  const auto& wu = *Find(P(l, "ffn_up_exps.weight"));
  const auto& wd = *Find(P(l, "ffn_down_exps.weight"));
  static const int gu_rows = [] {
    const char* v = std::getenv("GUFO_RAON1_GU_TILE");
    return v != nullptr ? std::atoi(v) : 128;
  }();
  if (!qfn::RoutedQ4KSituGemmF16(
          wg.data, wu.data, x16, gu_rows == 128 ? pf_->tiles128 : pf_->tiles48,
          gu_rows == 128 ? n128 : n48, gu_rows, pf_->bounds, pf_->rows_token,
          pf_->rows_slot, pf_->act16, ff, H, pf_->situ, stream_)) {
    if (error)
      *error = "routed gate/up GEMM failed";
    return false;
  }
  const void* down = wd.type == 6 ? down_q8_.at(l) : wd.data;
  static const bool wide_down = std::getenv("GUFO_RAON1_DOWN_WIDE") != nullptr;
  if (!(wide_down ? qfn::RoutedF16GemmQ8Wide(down, pf_->act16, pf_->tiles128,
                                             n128, pf_->bounds, pf_->rows_slot,
                                             pf_->rows_slot, nullptr, nullptr,
                                             pf_->down_e, H, ff, stream_)
                  : qfn::RoutedF16Gemm(down, qfn::WeightType::kQ8_0, pf_->act16,
                                       pf_->tiles64, n64, 64, pf_->bounds,
                                       pf_->rows_slot, pf_->rows_slot, nullptr,
                                       nullptr, pf_->down_e, H, ff, stream_))) {
    if (error)
      *error = "routed down GEMM failed";
    return false;
  }
  MoeCombineHalf(pf_->down_e, sc_->weights, sc_->ffn_out, H, use, n, stream_);
  return true;
}

bool Executor::Upload(const std::string& name, std::string* error) {
  if (tensors_.find(name) != tensors_.end())
    return true;
  const auto* t = reader_->FindTensor(name);
  if (t == nullptr) {
    if (error)
      *error = "missing tensor " + name;
    return false;
  }
  return UploadImpl(name, *t, error);
}

bool Executor::Resolve(std::string* error) {
  if (!Upload("token_embd.weight", error) ||
      !Upload("output_norm.weight", error) || !Upload("output.weight", error)) {
    return false;
  }
  for (std::uint32_t l = 0; l < layers_; ++l) {
    const char* common[] = {"attn_norm.weight",     "ffn_norm.weight",
                            "hc_attn_base.weight",  "hc_attn_fn.weight",
                            "hc_attn_scale.weight", "hc_ffn_base.weight",
                            "hc_ffn_fn.weight",     "hc_ffn_scale.weight"};
    for (const char* n : common) {
      if (!Upload(P(l, n), error))
        return false;
    }
    if (config_.IsRda(l)) {
      const char* rda[] = {"attn_q.weight",       "attn_k.weight",
                           "attn_v.weight",       "ssm_conv1d_q.weight",
                           "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
                           "ssm_f_a.weight",      "ssm_f_b.weight",
                           "ssm_dt.bias",         "ssm_a",
                           "ssm_beta.weight",     "ssm_rda.weight",
                           "ssm_norm.weight",     "ssm_g.weight",
                           "attn_output.weight"};
      for (const char* n : rda) {
        if (!Upload(P(l, n), error))
          return false;
      }
    } else {
      const char* mla[] = {"attn_q_a.weight",       "attn_q_a_norm.weight",
                           "attn_q_b.weight",       "attn_kv_a_mqa.weight",
                           "attn_kv_a_norm.weight", "attn_k_b.weight",
                           "attn_v_b.weight",       "attn_gate.weight",
                           "attn_output.weight"};
      for (const char* n : mla) {
        if (!Upload(P(l, n), error))
          return false;
      }
    }
    if (config_.IsDense(l)) {
      const char* dense[] = {"ffn_gate.weight", "ffn_up.weight",
                             "ffn_down.weight"};
      for (const char* n : dense) {
        if (!Upload(P(l, n), error))
          return false;
      }
    } else {
      const char* moe[] = {"ffn_gate_inp.weight",  "exp_probs_b.bias",
                           "ffn_gate_exps.weight", "ffn_up_exps.weight",
                           "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                           "ffn_up_shexp.weight",  "ffn_down_shexp.weight"};
      for (const char* n : moe) {
        if (!Upload(P(l, n), error))
          return false;
      }
    }
  }
  return true;
}

bool Executor::AllocScratch(std::string* error) {
  sc_ = std::make_unique<Scratch>();
  const auto H = config_.hidden_size;
  const auto ff = std::max(config_.expert_ff, config_.dense_ff);
  const auto inner = config_.KdaDim();
  const auto vocab = config_.vocab_size;
  const auto head_dim = config_.MlaHeadKey();
  const auto rank = config_.kv_lora_rank;
  const auto use = config_.expert_used;
  auto alloc = [&](float*& p, std::size_t n, const char* name,
                   std::size_t rows = kMaxBatch) {
    if (hipMalloc(&p, n * rows * sizeof(float)) != hipSuccess) {
      if (error)
        *error = std::string("scratch alloc failed: ") + name;
      return false;
    }
    return true;
  };
  if (!alloc(sc_->x, config_.HcDim(), "x") ||
      !alloc(sc_->y, config_.HcDim(), "y") ||
      !alloc(sc_->pre, config_.hc_count, "pre") ||
      !alloc(sc_->post, config_.hc_count, "post") ||
      !alloc(sc_->comb, config_.hc_count * config_.hc_count, "comb") ||
      !alloc(sc_->attn_in, H, "attn_in") || !alloc(sc_->block, H, "block") ||
      !alloc(sc_->ffn_in, H, "ffn_in") || !alloc(sc_->ffn_out, H, "ffn_out") ||
      !alloc(sc_->head, H, "head") ||
      !alloc(sc_->logits, vocab, "logits", kHeadRows) ||
      !alloc(sc_->q, inner, "q") || !alloc(sc_->k, inner, "k") ||
      !alloc(sc_->v, inner, "v") || !alloc(sc_->fa, 128, "fa") ||
      !alloc(sc_->gproj, inner, "gproj") ||
      !alloc(sc_->beta, config_.num_heads, "beta") ||
      !alloc(sc_->delta, config_.num_heads, "delta") ||
      !alloc(sc_->g2, inner, "g2") || !alloc(sc_->decay, inner, "decay") ||
      !alloc(sc_->rda_out, inner, "rda_out") ||
      !alloc(sc_->qr, config_.q_lora_rank, "qr") ||
      !alloc(sc_->mla_q, config_.MlaQDim(), "mla_q") ||
      !alloc(sc_->mla_qe,
             static_cast<std::size_t>(config_.num_heads) *
                 config_.qk_rope_head_dim,
             "mla_qe") ||
      !alloc(sc_->kv, config_.q_lora_rank, "kv") ||
      !alloc(sc_->qabs, config_.num_heads * rank, "qabs") ||
      !alloc(sc_->ctx, config_.num_heads * rank, "ctx") ||
      !alloc(sc_->gate, config_.MlaVDim(), "gate") ||
      !alloc(sc_->moe_logits, config_.expert_count, "moe_logits") ||
      !alloc(sc_->weights, use, "weights") || !alloc(sc_->gf, ff, "gf") ||
      !alloc(sc_->uf, ff, "uf") || !alloc(sc_->act, ff, "act") ||
      !alloc(sc_->dpart, H, "dpart") ||
      !alloc(sc_->mla_part,
             static_cast<std::size_t>(config_.num_heads) * kMlaPartSplits *
                 (rank + 2),
             "mla_part", kMaxVerify) ||
      !alloc(sc_->parts, static_cast<std::size_t>(use) * H, "parts")) {
    return false;
  }
  if (hipMalloc(&sc_->ids, use * kMaxBatch * sizeof(int)) != hipSuccess ||
      hipMalloc(&sc_->argmax, kHeadRows * sizeof(int)) != hipSuccess ||
      hipMalloc(&sc_->tok, kMaxBatch * sizeof(int)) != hipSuccess ||
      hipMalloc(&sc_->a16, static_cast<std::size_t>(kMaxBatch) * H * 2) !=
          hipSuccess ||
      hipMalloc(&sc_->f16, static_cast<std::size_t>(kMaxBatch) * H * 2) !=
          hipSuccess) {
    if (error)
      *error = "scratch alloc failed: ids";
    return false;
  }
  mmq_scratch_bytes_ = glm53_flash::rocm::MatvecScratchBytes(
      std::max(8, kMaxBatch * 6), std::max<std::uint32_t>(H, config_.dense_ff));
  if (hipMalloc(&mmq_scratch_, mmq_scratch_bytes_) != hipSuccess) {
    if (error)
      *error = "mmq scratch alloc failed";
    return false;
  }
  (void)head_dim;
  return true;
}

std::unique_ptr<Executor::Session> Executor::CreateSession(std::string* error) {
  auto s = std::make_unique<Session>();
  static std::atomic<std::uint64_t> next_id{1};
  s->id = next_id.fetch_add(1);
  const auto inner = config_.KdaDim();
  const auto taps = config_.ssm_conv_kernel - 1;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    LayerState st;
    if (config_.IsRda(l)) {
      if (hipMalloc(&st.conv, 3 * taps * inner * sizeof(float)) != hipSuccess ||
          hipMalloc(&st.state, static_cast<std::size_t>(config_.num_heads) *
                                   config_.kda_head_dim * config_.kda_head_dim *
                                   sizeof(float)) != hipSuccess) {
        if (error)
          *error = "RDA state alloc failed";
        return nullptr;
      }
      hipMemsetAsync(st.conv, 0, 3 * taps * inner * sizeof(float), stream_);
      hipMemsetAsync(st.state, 0,
                     static_cast<std::size_t>(config_.num_heads) *
                         config_.kda_head_dim * config_.kda_head_dim *
                         sizeof(float),
                     stream_);
    } else {
      const std::size_t cap =
          static_cast<std::size_t>(options_.max_context) * config_.kv_lora_rank;
      if (hipMalloc(&st.lat, cap * sizeof(float)) != hipSuccess ||
          hipMalloc(&st.extra, static_cast<std::size_t>(options_.max_context) *
                                   config_.qk_rope_head_dim * sizeof(float)) !=
              hipSuccess) {
        if (error)
          *error = "MLA cache alloc failed";
        return nullptr;
      }
      if (sparse_.on &&
          hipMalloc(&st.pool,
                    (static_cast<std::size_t>(options_.max_context) / 64 + 2) *
                        2 * config_.kv_lora_rank * sizeof(float)) !=
              hipSuccess) {
        if (error)
          *error = "MLA pool alloc failed";
        return nullptr;
      }
    }
    s->layers.push_back(st);
  }
  return s;
}

void Executor::PreqBegin(const float* x, int n, int k) {
  static const bool on = std::getenv("GUFO_RAON1_PREQ") == nullptr ||
                         std::getenv("GUFO_RAON1_PREQ")[0] != '0';
  preq_x_ = nullptr;
  if (!on || n > 8)
    return;
  const std::size_t need = glm53_flash::rocm::MatvecScratchBytes(n, k);
  if (preq_bytes_ < need) {
    if (preq_buf_ != nullptr)
      (void)hipFree(preq_buf_);
    preq_buf_ = nullptr;
    preq_bytes_ = 0;
    if (hipMalloc(&preq_buf_, need) != hipSuccess) {
      preq_buf_ = nullptr;
      return;
    }
    preq_bytes_ = need;
  }
  if (glm53_flash::rocm::QuantizeActivations(x, n, k, preq_buf_, stream_) != 0)
    return;
  preq_x_ = x;
  preq_n_ = n;
  preq_k_ = k;
}

bool Executor::Matvec(const DeviceTensor& w, const float* x, float* y, int m,
                      int n, std::string* error) {
  int rc = -1;
  if (preq_x_ != nullptr && x == preq_x_ && n == preq_n_ &&
      static_cast<int>(w.cols) == preq_k_ && w.type != 0 && w.type != 15 &&
      n < kHalfGemmMin) {
    rc = glm53_flash::rocm::MatvecQuantized(w.type, w.data, preq_buf_, y, m, n,
                                            static_cast<int>(w.cols), stream_);
    if (rc != 0) {
      if (error)
        *error = "quantized matvec failed";
      return false;
    }
    return true;
  }
  if (n >= kHalfGemmMin && xh_ != nullptr && half_w_.count(w.data) != 0) {
    const __half* x16 = xh_;
    if (x == B().hint_src[0] && B().hint_buf[0] != nullptr) {
      x16 = B().hint_buf[0];
    } else if (x == B().hint_src[1] && B().hint_buf[1] != nullptr) {
      x16 = B().hint_buf[1];
    } else {
      ToHalf(x, xh_, static_cast<std::size_t>(n) * w.cols, stream_);
    }
    rc = qfn::DenseHalfWeightGemmWide(half_w_.at(w.data), x16, y, n, m, w.cols,
                                      stream_)
             ? 0
             : -1;
  } else if (w.type == 15) {
    rc = 0;
    for (int t = 0; t < n && rc == 0; ++t) {
      rc = rocm::F16Matvec(static_cast<const __half*>(w.data),
                           x + static_cast<std::size_t>(t) * w.cols,
                           y + static_cast<std::size_t>(t) * m, m,
                           static_cast<int>(w.cols), stream_);
    }
  } else if (w.type == 0 && n >= 16) {
    rc = rocm::F32Gemm(static_cast<const float*>(w.data), x, y, m, n,
                       static_cast<int>(w.cols), stream_);
  } else if (w.type == 0 && n > 1) {
    rc = rocm::F32SmallGemm(static_cast<const float*>(w.data), x, y, m, n,
                            static_cast<int>(w.cols), stream_);
  } else {
    rc = glm53_flash::rocm::Gemm(w.type, w.data, x, y, m, n, w.cols,
                                 mmq_scratch_, stream_);
  }
  static const bool trace = std::getenv("GUFO_RAON1_GPU_TRACE") != nullptr;
  if (trace) {
    hipStreamSynchronize(stream_);
    std::fprintf(stderr,
                 "MATVEC t=%d m=%d n=%d k=%u rc=%d gpu=%s w=%p x=%p y=%p "
                 "s=%p wb=%zu\n",
                 w.type, m, n, w.cols, rc, hipGetErrorString(hipGetLastError()),
                 w.data, x, y, mmq_scratch_, w.bytes);
  }
  if (rc != 0) {
    if (error)
      *error = "matvec failed (type " + std::to_string(w.type) + ", m " +
               std::to_string(m) + ", k " + std::to_string(w.cols) + ")";
    return false;
  }
  return true;
}

bool Executor::MatvecRows(const DeviceTensor& w, std::uint64_t row0, int m,
                          const float* x, float* y, std::string* error) {
  void* base = static_cast<char*>(w.data) + row0 * w.row_bytes;
  const int rc = w.type == 15
                     ? rocm::F16Matvec(static_cast<const __half*>(base), x, y,
                                       m, static_cast<int>(w.cols), stream_)
                     : glm53_flash::rocm::Matvec(w.type, base, x, y, m, 1,
                                                 w.cols, mmq_scratch_, stream_);
  if (rc != 0) {
    if (error)
      *error = "row matvec failed";
    return false;
  }
  return true;
}

bool Executor::F32Matvec(const DeviceTensor& w, const float* x, float* y, int m,
                         int n, std::string* error) {
  int rc = 0;
  if (w.type == 0 && n >= 16) {
    rc = rocm::F32Gemm(static_cast<const float*>(w.data), x, y, m, n,
                       static_cast<int>(w.cols), stream_);
    n = 0;
  } else if (w.type == 0 && n > 1) {
    rc = rocm::F32SmallGemm(static_cast<const float*>(w.data), x, y, m, n,
                            static_cast<int>(w.cols), stream_);
    n = 0;
  }
  for (int t = 0; t < n && rc == 0; ++t) {
    const float* xt = x + static_cast<std::size_t>(t) * w.cols;
    float* yt = y + static_cast<std::size_t>(t) * m;
    rc = w.type == 15
             ? rocm::F16Matvec(static_cast<const __half*>(w.data), xt, yt, m,
                               static_cast<int>(w.cols), stream_)
             : rocm::F32Matvec(static_cast<const float*>(w.data), xt, yt, m,
                               static_cast<int>(w.cols), stream_);
  }
  if (rc != 0) {
    if (error)
      *error = "F32 matvec failed";
    return false;
  }
  return true;
}

void Executor::ResetSession(Session& session) {
  session.position = 0;
  InvalidateFull();
  for (std::uint32_t l = 0; l < layers_; ++l) {
    auto& st = session.layers[l];
    if (config_.IsRda(l)) {
      const std::size_t inner = config_.KdaDim();
      const std::size_t taps = config_.ssm_conv_kernel - 1;
      hipMemsetAsync(st.conv, 0, 3 * taps * inner * sizeof(float), stream_);
      hipMemsetAsync(st.state, 0,
                     static_cast<std::size_t>(config_.num_heads) *
                         config_.kda_head_dim * config_.kda_head_dim *
                         sizeof(float),
                     stream_);
    } else {
      const std::size_t lat =
          static_cast<std::size_t>(options_.max_context) * config_.kv_lora_rank;
      const std::size_t extra = static_cast<std::size_t>(options_.max_context) *
                                config_.qk_rope_head_dim;
      hipMemsetAsync(st.lat, 0, lat * sizeof(float), stream_);
      hipMemsetAsync(st.extra, 0, extra * sizeof(float), stream_);
    }
  }
}

bool Executor::Evaluate(Session& session, std::int32_t token,
                        std::string* error) {
  return EvaluateBatch(session, std::span<const std::int32_t>(&token, 1),
                       error);
}

bool Executor::EvaluateBatch(Session& session,
                             std::span<const std::int32_t> tokens,
                             std::string* error) {
  logit_bank_ = 0;
  if (tokens.size() >= static_cast<std::size_t>(kMoeGemmMin))
    EnsureFull(session.position + static_cast<std::uint32_t>(tokens.size()));
  if (!RunBatch(session, tokens, error, session.position, nullptr, nullptr,
                true)) {
    return false;
  }
  session.position += static_cast<std::uint32_t>(tokens.size());
  return true;
}

struct Executor::LayerGate {
  std::mutex mutex;
  std::condition_variable cv;
  std::uint32_t progress{0};
  bool done{false};
  std::vector<hipEvent_t> events;

  bool Consume(std::uint32_t layer, hipStream_t stream) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return progress > layer || done; });
    if (progress <= layer)
      return false;
    hipStreamWaitEvent(stream, events[layer], 0);
    return true;
  }
  void Produce(std::uint32_t layer, hipStream_t stream) {
    hipEventRecord(events[layer], stream);
    {
      std::lock_guard<std::mutex> lock(mutex);
      progress = layer + 1;
    }
    cv.notify_all();
  }
  void Finish() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      done = true;
    }
    cv.notify_all();
  }
};

bool Executor::EnsureFull(std::uint32_t tokens) {
  if (full_.empty() || tokens > full_cap_)
    return false;
  constexpr std::uint32_t kStep = 8192;
  const std::uint32_t want =
      std::min<std::uint32_t>(full_cap_, (tokens + kStep - 1) / kStep * kStep);
  const std::size_t heads = config_.num_heads;
  bool grew = false;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!config_.IsMla(l) || full_[l].cap >= want)
      continue;
    if (!grew) {
      for (auto& bank : banks_)
        hipStreamSynchronize(bank->stream);
      grew = true;
    }
    auto& f = full_[l];
    if (f.k != nullptr)
      hipFree(f.k);
    if (f.v != nullptr)
      hipFree(f.v);
    f = FullKv{};
    if (hipMalloc(reinterpret_cast<void**>(&f.k),
                  static_cast<std::size_t>(want) * heads * 192 * 2) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&f.v),
                  static_cast<std::size_t>(want) * heads * 128 * 2) !=
            hipSuccess) {
      if (f.k != nullptr)
        hipFree(f.k);
      f = FullKv{};
      return false;
    }
    f.cap = want;
  }
  return true;
}

int Executor::PipelineBanks(std::size_t tokens) const {
  if (const char* v = std::getenv("GUFO_RAON1_BANKS"))
    return std::clamp(std::atoi(v), 1, 4);
  (void)tokens;
  return 2;
}

bool Executor::CanPair(const Session& session, int tokens, int banks) const {
  if (pf_ == nullptr || mla_ == nullptr || banks < 2 ||
      tokens < banks * kMoeGemmMin || tokens > banks * kMaxBatch) {
    return false;
  }
  if (sparse_.on &&
      session.position >= static_cast<std::uint32_t>(sparse_.min_depth)) {
    return session.position + static_cast<std::uint32_t>(tokens) <=
           options_.max_context;
  }
  return session.position + static_cast<std::uint32_t>(tokens) <=
         std::min<std::uint32_t>(options_.max_context,
                                 full_.empty() ? 32768U : full_cap_);
}

bool Executor::EvaluatePipelined(Session& session,
                                 std::span<const std::int32_t> tokens,
                                 int banks, std::string* error) {
  if (!CanPair(session, static_cast<int>(tokens.size()), banks)) {
    if (error)
      *error = "pipeline does not fit";
    return false;
  }
  while (static_cast<int>(banks_.size()) < banks) {
    if (!AddBank(error))
      return false;
  }
  EnsureFull(session.position + static_cast<std::uint32_t>(tokens.size()));
  const std::uint32_t p = session.position;
  const int total = static_cast<int>(tokens.size());
  std::vector<int> sizes(banks, total / banks);
  for (int i = 0; i < total % banks; ++i)
    ++sizes[i];
  std::vector<std::unique_ptr<LayerGate>> gates;
  for (int i = 0; i + 1 < banks; ++i) {
    gates.push_back(std::make_unique<LayerGate>());
    gates.back()->events.resize(layers_);
    for (auto& ev : gates.back()->events)
      hipEventCreateWithFlags(&ev, hipEventDisableTiming);
  }
  std::vector<std::string> errors(banks);
  std::vector<char> oks(banks, 0);
  std::vector<std::thread> workers;
  std::uint32_t offset = 0;
  std::vector<std::uint32_t> starts(banks);
  for (int i = 0; i < banks; ++i) {
    starts[i] = offset;
    offset += static_cast<std::uint32_t>(sizes[i]);
  }
  auto run = [&](int i) {
    LayerGate* in = i > 0 ? gates[i - 1].get() : nullptr;
    LayerGate* out = i + 1 < banks ? gates[i].get() : nullptr;
    const bool ok =
        RunBatch(session, tokens.subspan(starts[i], sizes[i]), &errors[i],
                 p + starts[i], in, out, i + 1 == banks);
    oks[i] = ok;
    if (out != nullptr)
      out->Finish();
  };
  for (int i = 1; i < banks; ++i) {
    workers.emplace_back([&, i] {
      cur_bank_ = banks_[i].get();
      run(i);
      cur_bank_ = nullptr;
    });
  }
  run(0);
  for (auto& w : workers)
    w.join();
  for (int i = 0; i < banks; ++i)
    hipStreamSynchronize(banks_[i]->stream);
  for (auto& g : gates)
    for (auto ev : g->events)
      hipEventDestroy(ev);
  for (int i = 0; i < banks; ++i) {
    if (!oks[i]) {
      if (error)
        *error = errors[i];
      return false;
    }
  }
  last_pipeline_.clear();
  for (int i = 0; i < banks; ++i) {
    last_pipeline_.push_back(
        {i, p + starts[i], static_cast<std::uint32_t>(sizes[i])});
  }
  session.position = p + static_cast<std::uint32_t>(total);
  logit_bank_ = banks - 1;
  return true;
}

bool Executor::RunBatch(Session& session, std::span<const std::int32_t> tokens,
                        std::string* error, std::uint32_t pos0, LayerGate* in,
                        LayerGate* out, bool head) {
  const int n = static_cast<int>(tokens.size());
  if (n < 1 || n > kMaxBatch) {
    if (error)
      *error = "batch size out of range";
    return false;
  }
  for (const auto token : tokens) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= config_.vocab_size) {
      if (error)
        *error = "token out of range";
      return false;
    }
  }
  const auto H = config_.hidden_size;
  const auto hc = config_.hc_count;
  const auto hd = config_.kda_head_dim;
  const std::size_t inner = config_.KdaDim();
  const auto rank = config_.kv_lora_rank;
  const auto use = config_.expert_used;
  const std::size_t hcdim = config_.HcDim();
  const std::size_t qdim = config_.MlaQDim();
  const std::size_t qe_dim =
      static_cast<std::size_t>(config_.num_heads) * config_.qk_rope_head_dim;
  const std::size_t abs_dim =
      static_cast<std::size_t>(config_.num_heads) * rank;

  const auto* embd = Find("token_embd.weight");
  if (embd == nullptr) {
    if (error)
      *error = "missing token_embd.weight";
    return false;
  }
  hipMemcpyAsync(sc_->tok, tokens.data(), static_cast<std::size_t>(n) * 4,
                 hipMemcpyHostToDevice, stream_);
  qfn::EmbedTokensQuant(embd->data, static_cast<std::uint32_t>(embd->type),
                        sc_->tok, sc_->x, n, H, hc, stream_);
  float* xcur = sc_->x;
  float* ycur = sc_->y;

  auto get = [&](std::uint32_t l, const char* nm) -> const DeviceTensor& {
    return *Find(P(l, nm));
  };
  auto norm = [&](float* x, const DeviceTensor& w, int width, int rows,
                  float eps) {
    RmsNormRows(x, static_cast<const float*>(w.data), x, width, rows, eps,
                stream_);
  };

  // Norm that also writes the F16 copy the dense GEMMs and the NPU encoder
  // consume, remembered as a hint so they skip their own conversion.
  auto norm_half = [&](float* x, const DeviceTensor& w, int width, int rows,
                       float eps, __half* y16, int slot) {
    RmsNormRows(x, static_cast<const float*>(w.data), x, width, rows, eps,
                stream_, y16);
    B().hint_src[slot] = x;
    B().hint_buf[slot] = y16;
  };

  auto hc_select = [&](const DeviceTensor& fn, const DeviceTensor& scale,
                       const DeviceTensor& base, float* y) {
    if (n >= kHalfGemmMin) {
      if (HcDots(xcur, static_cast<const float*>(fn.data), sc_->dpart, n,
                 stream_) != 0) {
        return -1;
      }
      return HcSelectPre(
          xcur, sc_->dpart, static_cast<const float*>(scale.data),
          static_cast<const float*>(base.data), config_.hc_eps,
          config_.hc_magnitude, y, sc_->pre, sc_->post, sc_->comb, stream_, n);
    }
    if (HcDotsPart(xcur, static_cast<const float*>(fn.data), sc_->dpart, n,
                   stream_) != 0) {
      return -1;
    }
    return HcSelectPre(xcur, sc_->dpart, static_cast<const float*>(scale.data),
                       static_cast<const float*>(base.data), config_.hc_eps,
                       config_.hc_magnitude, y, sc_->pre, sc_->post, sc_->comb,
                       stream_, n, kHcPartChunks);
  };

  auto capture_tap = [&](std::uint32_t l, bool at_input) {
    if (B().capture == nullptr || at_input != tap_input_)
      return;
    const auto& ids = options_.dflash_layers;
    for (std::size_t j = 0; j < ids.size(); ++j) {
      if (ids[j] != l)
        continue;
      hipMemcpy2DAsync(B().capture + j * hcdim,
                       ids.size() * hcdim * sizeof(float), xcur,
                       hcdim * sizeof(float), hcdim * sizeof(float), n,
                       hipMemcpyDeviceToDevice, stream_);
    }
  };
  B().cap_rows = B().capture != nullptr ? n : 0;
  B().cap_pos = pos0;

  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (in != nullptr && !in->Consume(l, stream_)) {
      if (error)
        *error = "first bank failed";
      return false;
    }
    capture_tap(l, true);
    const auto& ha_fn = get(l, "hc_attn_fn.weight");
    const auto& ha_scale = get(l, "hc_attn_scale.weight");
    const auto& ha_base = get(l, "hc_attn_base.weight");
    if (hc_select(ha_fn, ha_scale, ha_base, sc_->attn_in) != 0) {
      if (error)
        *error = "hc_attn failed";
      return false;
    }
    norm_half(sc_->attn_in, get(l, "attn_norm.weight"), H, n, config_.rms_eps,
              sc_->a16, 0);

    auto& st = session.layers[l];
    if (config_.IsRda(l)) {
      const bool on_npu = npu_ != nullptr && n == static_cast<int>(kMaxBatch) &&
                          npu_->Healthy() && l < npu_layers_.size() &&
                          !npu_layers_[l].empty() && B().npu_a >= 0;
      NpuOp qkvg_op;
      if (on_npu) {
        NpuQkvgBegin(l, n, &qkvg_op);
      } else if ((PreqBegin(sc_->attn_in, n, static_cast<int>(H)), false) ||
                 !Matvec(get(l, "attn_q.weight"), sc_->attn_in, sc_->q, inner,
                         n, error) ||
                 !Matvec(get(l, "attn_k.weight"), sc_->attn_in, sc_->k, inner,
                         n, error) ||
                 !Matvec(get(l, "attn_v.weight"), sc_->attn_in, sc_->v, inner,
                         n, error) ||
                 !Matvec(get(l, "ssm_g.weight"), sc_->attn_in, sc_->g2, inner,
                         n, error)) {
        return false;
      }
      if (!Matvec(get(l, "ssm_f_a.weight"), sc_->attn_in, sc_->fa, 128, n,
                  error) ||
          !Matvec(get(l, "ssm_f_b.weight"), sc_->fa, sc_->gproj, inner, n,
                  error) ||
          !F32Matvec(get(l, "ssm_beta.weight"), sc_->attn_in, sc_->beta,
                     config_.num_heads, n, error) ||
          !F32Matvec(get(l, "ssm_rda.weight"), sc_->attn_in, sc_->delta,
                     config_.num_heads, n, error)) {
        return false;
      }
      PreqEnd();
      if (on_npu)
        NpuQkvgEnd(qkvg_op, n);
      if (recording_ && !rec_.empty() && n <= kMaxVerify) {
        auto& r = rec_[l];
        const int cnt = n;
        const float* srcs[7] = {sc_->q,  sc_->k,    sc_->v,    sc_->gproj,
                                sc_->g2, sc_->beta, sc_->delta};
        float* dsts[7] = {r.q, r.k, r.v, r.gproj, r.g2, r.beta, r.delta};
        const int counts[7] = {cnt * static_cast<int>(inner),
                               cnt * static_cast<int>(inner),
                               cnt * static_cast<int>(inner),
                               cnt * static_cast<int>(inner),
                               cnt * static_cast<int>(inner),
                               cnt * static_cast<int>(config_.num_heads),
                               cnt * static_cast<int>(config_.num_heads)};
        RecordOperands(srcs, dsts, counts, stream_);
      }
      if (RdaPrefill(
              static_cast<const float*>(get(l, "ssm_conv1d_q.weight").data),
              static_cast<const float*>(get(l, "ssm_conv1d_k.weight").data),
              static_cast<const float*>(get(l, "ssm_conv1d_v.weight").data),
              sc_->q, sc_->k, sc_->v, st.conv, sc_->gproj,
              static_cast<const float*>(get(l, "ssm_dt.bias").data),
              static_cast<const float*>(get(l, "ssm_a").data), sc_->beta,
              sc_->delta, sc_->g2,
              static_cast<const float*>(get(l, "ssm_norm.weight").data),
              st.state, sc_->rda_out, n, config_.num_heads,
              config_.kda_gate_lower_bound, config_.rms_eps, stream_) != 0) {
        if (error)
          *error = "RDA failed";
        return false;
      }
      if (NpuOn(n) && l < npu_lin_.size() && npu_lin_[l].out.ok()) {
        const auto& lin = npu_lin_[l].out;
        ToHalf(sc_->rda_out, xh_, static_cast<std::size_t>(n) * lin.halves * H,
               stream_);
        NpuLinEnd(
            lin,
            NpuLinBegin(lin, n, static_cast<std::size_t>(lin.halves) * H, xh_),
            sc_->block, H, n);
      } else if (!Matvec(get(l, "attn_output.weight"), sc_->rda_out, sc_->block,
                         H, n, error)) {
        return false;
      }
    } else {
      PreqBegin(sc_->attn_in, n, static_cast<int>(H));
      if (!Matvec(get(l, "attn_q_a.weight"), sc_->attn_in, sc_->qr,
                  config_.q_lora_rank, n, error)) {
        PreqEnd();
        return false;
      }
      norm(sc_->qr, get(l, "attn_q_a_norm.weight"), config_.q_lora_rank, n,
           config_.latent_norm_eps);
      if (!Matvec(get(l, "attn_q_b.weight"), sc_->qr, sc_->mla_q, qdim, n,
                  error) ||
          !Matvec(get(l, "attn_kv_a_mqa.weight"), sc_->attn_in, sc_->kv,
                  config_.q_lora_rank, n, error)) {
        PreqEnd();
        return false;
      }
      PreqEnd();
      const auto& kb = get(l, "attn_k_b.weight");
      const auto& vb = get(l, "attn_v_b.weight");
      KvNorm(sc_->kv,
             static_cast<const float*>(get(l, "attn_kv_a_norm.weight").data),
             st.lat + pos0 * rank, st.extra + pos0 * config_.qk_rope_head_dim,
             n, config_.q_lora_rank, rank, config_.qk_rope_head_dim,
             config_.latent_norm_eps, stream_);
      if (!full_.empty()) {
        std::lock_guard<std::mutex> lock(full_mutex_);
        full_[l].len = std::min<std::size_t>(full_[l].len, pos0);
      }
      if (sparse_.on && st.pool != nullptr) {
        const int first_block = static_cast<int>(pos0 / 64);
        const int last_block = static_cast<int>((pos0 + n - 1) / 64);
        PoolUpdate(st.lat, st.pool, first_block, last_block - first_block + 1,
                   static_cast<int>(pos0 + n), stream_);
      }
      const bool use_sparse = sparse_.on && st.pool != nullptr &&
                              n >= kMoeGemmMin && mla_ != nullptr &&
                              static_cast<int>(pos0) >= sparse_.min_depth;
      if (use_sparse) {
        const std::size_t heads = config_.num_heads;
        const int local_start =
            pos0 > static_cast<std::uint32_t>(sparse_.window)
                ? static_cast<int>((pos0 - sparse_.window) / 64 * 64)
                : 0;
        const int sink_rows = std::min(sparse_.sink, local_start);
        const int lo = sink_rows / 64;
        const int score_end = local_start / 64;
        const int kk = std::max(0, std::min(sparse_.k, score_end - lo));
        const int nk =
            sink_rows + kk * 64 + static_cast<int>(pos0) - local_start + n;
        if (kk > 0 && nk <= static_cast<int>(kMlaBlock) * 8) {
          QNopeMean(sc_->mla_q, sc_->dpart, n, static_cast<int>(heads),
                    static_cast<int>(config_.qk_nope_head_dim),
                    static_cast<int>(config_.qk_rope_head_dim), stream_);
          if (glm53_flash::rocm::MatvecBatchedTokens(
                  kb.type, kb.data, sc_->dpart, sc_->qabs, rank, kb.cols,
                  static_cast<int>(heads), 8, mmq_scratch_, stream_) != 0) {
            if (error)
              *error = "sparse k_b matvec failed";
            return false;
          }
          SparseSelect(sc_->mla_q, sc_->dpart, sc_->qabs, st.pool, mla_->scores,
                       mla_->sel, mla_->kpos, n, lo, score_end, kk, sink_rows,
                       local_start, static_cast<int>(pos0), 8, stream_);
          for (int b0 = 0; b0 < nk; b0 += static_cast<int>(kMlaBlock)) {
            const int cnt = std::min<int>(kMlaBlock, nk - b0);
            GatherKv(st.lat, st.extra, mla_->kpos + b0, cnt, mla_->lat16,
                     mla_->extra_g, stream_);
            if (!qfn::DenseHalfWeightGemmWide(mla_->kbt.at(l), mla_->lat16,
                                              mla_->knope, cnt, heads * 128,
                                              rank, stream_) ||
                !qfn::DenseHalfWeightGemmWide(mla_->vbh.at(l), mla_->lat16,
                                              mla_->vexp, cnt, heads * 128,
                                              rank, stream_)) {
              if (error)
                *error = "MLA K/V expansion failed";
              return false;
            }
            BuildKvHalf(mla_->knope, mla_->extra_g,
                        mla_->k16 + static_cast<std::size_t>(b0) * heads * 192,
                        cnt, stream_);
            ToHalf(mla_->vexp,
                   mla_->v16 + static_cast<std::size_t>(b0) * heads * 128,
                   static_cast<std::size_t>(cnt) * heads * 128, stream_);
          }
          ToHalf(sc_->mla_q, mla_->q16, static_cast<std::size_t>(n) * qdim,
                 stream_);
          MlaWmmaAttend(mla_->q16, mla_->k16, mla_->v16, sc_->rda_out, n,
                        static_cast<int>(pos0), stream_, mla_->kpos, nk);
        } else {
          if (error)
            *error = "sparse attention does not fit";
          return false;
        }
      } else if (n >= kMoeGemmMin && mla_ != nullptr &&
                 (pos0 + n <= mla_cap_ ||
                  (!full_.empty() && pos0 + n <= full_[l].cap))) {
        const std::size_t tot = pos0 + n;
        const std::size_t heads = config_.num_heads;
        __half* kfull = mla_->k16;
        __half* vfull = mla_->v16;
        std::size_t from = 0;
        if (!full_.empty() && tot <= full_[l].cap) {
          std::lock_guard<std::mutex> lock(full_mutex_);
          auto& f = full_[l];
          kfull = f.k;
          vfull = f.v;
          from = f.owner == session.id ? std::min<std::size_t>(f.len, pos0) : 0;
          f.owner = session.id;
          f.len = tot;
        }
        for (std::size_t b0 = from; b0 < tot; b0 += kMlaBlock) {
          const std::size_t cnt = std::min<std::size_t>(kMlaBlock, tot - b0);
          ToHalf(st.lat + b0 * rank, mla_->lat16, cnt * rank, stream_);
          if (!qfn::DenseHalfWeightGemmWide(mla_->kbt.at(l), mla_->lat16,
                                            mla_->knope, cnt, heads * 128, rank,
                                            stream_) ||
              !qfn::DenseHalfWeightGemmWide(mla_->vbh.at(l), mla_->lat16,
                                            mla_->vexp, cnt, heads * 128, rank,
                                            stream_)) {
            if (error)
              *error = "MLA K/V expansion failed";
            return false;
          }
          BuildKvHalf(mla_->knope, st.extra + b0 * config_.qk_rope_head_dim,
                      kfull + b0 * heads * 192, cnt, stream_);
          ToHalf(mla_->vexp, vfull + b0 * heads * 128, cnt * heads * 128,
                 stream_);
        }
        ToHalf(sc_->mla_q, mla_->q16, static_cast<std::size_t>(n) * qdim,
               stream_);
        MlaWmmaAttend(mla_->q16, kfull, vfull, sc_->rda_out, n,
                      static_cast<int>(pos0), stream_);
      } else {
        if (n <= kMaxVerify) {
          MlaQeGather(sc_->mla_q, sc_->mla_qe, sc_->dpart,
                      static_cast<int>(config_.num_heads),
                      static_cast<int>(config_.qk_nope_head_dim),
                      static_cast<int>(config_.qk_rope_head_dim), stream_, n);
          if (glm53_flash::rocm::MatvecBatchedTokens(
                  kb.type, kb.data, sc_->dpart, sc_->qabs, rank, kb.cols,
                  static_cast<int>(config_.num_heads), n, mmq_scratch_,
                  stream_) != 0) {
            if (error)
              *error = "k_b matvec failed";
            return false;
          }
        } else {
          for (int t = 0; t < n; ++t) {
            float* mq = sc_->mla_q + t * qdim;
            MlaQeGather(mq, sc_->mla_qe + t * qe_dim, sc_->dpart,
                        static_cast<int>(config_.num_heads),
                        static_cast<int>(config_.qk_nope_head_dim),
                        static_cast<int>(config_.qk_rope_head_dim), stream_);
            if (glm53_flash::rocm::MatvecBatched(
                    kb.type, kb.data, sc_->dpart, sc_->qabs + t * abs_dim, rank,
                    kb.cols, static_cast<int>(config_.num_heads), mmq_scratch_,
                    stream_) != 0) {
              if (error)
                *error = "k_b matvec failed";
              return false;
            }
          }
        }
        bool attended = false;
        if (sparse_.on && st.pool != nullptr && n <= kMaxVerify &&
            mla_ != nullptr && static_cast<int>(pos0) >= sparse_.min_depth) {
          const int local_start =
              pos0 > static_cast<std::uint32_t>(sparse_.window)
                  ? static_cast<int>((pos0 - sparse_.window) / 64 * 64)
                  : 0;
          const int sink_rows = std::min(sparse_.sink, local_start);
          const int lo = sink_rows / 64;
          const int score_end = local_start / 64;
          const int kk = std::max(0, std::min(sparse_.k, score_end - lo));
          const int nk =
              sink_rows + kk * 64 + static_cast<int>(pos0) - local_start + n;
          if (kk > 0 && nk <= static_cast<int>(kMlaBlock) * 8) {
            SparseSelect(nullptr, nullptr, sc_->qabs, st.pool, mla_->scores,
                         mla_->sel, mla_->kpos, n, lo, score_end, kk, sink_rows,
                         local_start, static_cast<int>(pos0), n, stream_);
            GatherKvFloat(st.lat, st.extra, mla_->kpos, nk, mla_->lat_g,
                          mla_->extra_g, stream_);
            if (MlaAttendSplit(sc_->qabs, sc_->mla_qe, mla_->lat_g,
                               mla_->extra_g, sc_->mla_part, sc_->ctx,
                               static_cast<int>(config_.num_heads), rank,
                               static_cast<int>(config_.qk_rope_head_dim),
                               nk - n, n, stream_) != 0) {
              if (error)
                *error = "MLA attention failed";
              return false;
            }
            attended = true;
          }
        }
        for (int t0 = 0; !attended && t0 < n; t0 += kMaxVerify) {
          const int cnt = std::min<int>(kMaxVerify, n - t0);
          if (MlaAttendSplit(sc_->qabs + t0 * abs_dim,
                             sc_->mla_qe + t0 * qe_dim, st.lat, st.extra,
                             sc_->mla_part, sc_->ctx + t0 * abs_dim,
                             static_cast<int>(config_.num_heads), rank,
                             static_cast<int>(config_.qk_rope_head_dim),
                             static_cast<int>(pos0) + t0, cnt, stream_) != 0) {
            if (error)
              *error = "MLA attention failed";
            return false;
          }
        }
        if (n <= kMaxVerify) {
          if (glm53_flash::rocm::MatvecBatchedTokens(
                  vb.type, vb.data, sc_->ctx, sc_->rda_out, config_.v_head_dim,
                  vb.cols, static_cast<int>(config_.num_heads), n, mmq_scratch_,
                  stream_) != 0) {
            if (error)
              *error = "v_b matvec failed";
            return false;
          }
        } else {
          for (int t = 0; t < n; ++t) {
            if (glm53_flash::rocm::MatvecBatched(
                    vb.type, vb.data, sc_->ctx + t * abs_dim,
                    sc_->rda_out + t * inner, config_.v_head_dim, vb.cols,
                    static_cast<int>(config_.num_heads), mmq_scratch_,
                    stream_) != 0) {
              if (error)
                *error = "v_b matvec failed";
              return false;
            }
          }
        }
      }
      if (NpuOn(n) && l < npu_lin_.size() && npu_lin_[l].gate.ok()) {
        const auto& lin = npu_lin_[l].gate;
        NpuLinEnd(lin, NpuLinBegin(lin, n, H, sc_->a16), sc_->gate,
                  config_.MlaVDim(), n);
      } else if (!Matvec(get(l, "attn_gate.weight"), sc_->attn_in, sc_->gate,
                         config_.MlaVDim(), n, error)) {
        return false;
      }
      VectorMul(sc_->rda_out, sc_->gate, sc_->rda_out, n * config_.MlaVDim(),
                stream_);
      if (NpuOn(n) && l < npu_lin_.size() && npu_lin_[l].out.ok()) {
        const auto& lin = npu_lin_[l].out;
        ToHalf(sc_->rda_out, xh_, static_cast<std::size_t>(n) * lin.halves * H,
               stream_);
        NpuLinEnd(
            lin,
            NpuLinBegin(lin, n, static_cast<std::size_t>(lin.halves) * H, xh_),
            sc_->block, H, n);
      } else if (!Matvec(get(l, "attn_output.weight"), sc_->rda_out, sc_->block,
                         H, n, error)) {
        return false;
      }
    }

    HcApply(xcur, sc_->block, sc_->post, sc_->comb, ycur, stream_, n);
    std::swap(xcur, ycur);
    const auto& hf_fn = get(l, "hc_ffn_fn.weight");
    const auto& hf_scale = get(l, "hc_ffn_scale.weight");
    const auto& hf_base = get(l, "hc_ffn_base.weight");
    hc_select(hf_fn, hf_scale, hf_base, sc_->ffn_in);
    norm_half(sc_->ffn_in, get(l, "ffn_norm.weight"), H, n, config_.rms_eps,
              sc_->f16, 1);

    if (config_.IsDense(l)) {
      if (!Matvec(get(l, "ffn_gate.weight"), sc_->ffn_in, sc_->gf,
                  config_.dense_ff, n, error) ||
          !Matvec(get(l, "ffn_up.weight"), sc_->ffn_in, sc_->uf,
                  config_.dense_ff, n, error)) {
        return false;
      }
      SituAct(sc_->gf, sc_->uf, sc_->act, n * config_.dense_ff,
              config_.situ_beta, config_.situ_linear_beta, stream_);
      if (!Matvec(get(l, "ffn_down.weight"), sc_->act, sc_->ffn_out, H, n,
                  error)) {
        return false;
      }
    } else {
      if (!F32Matvec(get(l, "ffn_gate_inp.weight"), sc_->ffn_in,
                     sc_->moe_logits, config_.expert_count, n, error)) {
        return false;
      }
      MoeRoute(sc_->moe_logits,
               static_cast<const float*>(get(l, "exp_probs_b.bias").data),
               config_.expert_weights_scale, sc_->ids, sc_->weights,
               config_.expert_count, use, config_.expert_weights_norm, stream_,
               n);
      const bool shared_npu = NpuOn(n) && l < npu_lin_.size() &&
                              npu_lin_[l].shg.ok() && npu_lin_[l].shu.ok() &&
                              npu_lin_[l].shg.slices + npu_lin_[l].shu.slices <=
                                  static_cast<int>(B().npu_c.size());
      NpuOp shared_op;
      if (shared_npu) {
        const auto& g = npu_lin_[l].shg;
        const auto& u = npu_lin_[l].shu;
        qfn::NpuEncodeActivations(sc_->f16, H, n, H, 0, H,
                                  npu_->Device(B().npu_a), stream_, false);
        shared_op = NpuStart(
            {{B().npu_a, g.w.data(), B().npu_c.data(), g.slices},
             {B().npu_a, u.w.data(), B().npu_c.data() + g.slices, u.slices}});
      }
      if (n >= kMoeGemmMin && pf_ != nullptr) {
        if (!PrefillMoe(l, n, error))
          return false;
      } else {
        const auto& wg = get(l, "ffn_gate_exps.weight");
        const auto& wu = get(l, "ffn_up_exps.weight");
        const auto& wd = get(l, "ffn_down_exps.weight");
        auto expert_matvec = [&](const DeviceTensor& w, const float* x,
                                 float* y, int m, int k, bool x_per_expert) {
          if (w.type == 15) {
            for (int t = 0; t < n; ++t) {
              const std::size_t xs = x_per_expert
                                         ? use * static_cast<std::size_t>(k)
                                         : static_cast<std::size_t>(k);
              const int rc = rocm::F16MatvecExperts(
                  static_cast<const __half*>(w.data), x + t * xs,
                  y + static_cast<std::size_t>(t) * use * m, m, k,
                  static_cast<int>(use), sc_->ids + t * use, x_per_expert,
                  stream_);
              if (rc != 0)
                return rc;
            }
            return 0;
          }
          constexpr int kExpertGroup = 4;
          const std::size_t xs = x_per_expert
                                     ? use * static_cast<std::size_t>(k)
                                     : static_cast<std::size_t>(k);
          for (int t0 = 0; t0 < n; t0 += kExpertGroup) {
            const int rc = glm53_flash::rocm::MatvecExpertsTokens(
                w.type, w.data, x + t0 * xs,
                y + static_cast<std::size_t>(t0) * use * m, m, k,
                config_.expert_count, use, std::min(kExpertGroup, n - t0),
                sc_->ids + t0 * use, x_per_expert, mmq_scratch_, stream_);
            if (rc != 0)
              return rc;
          }
          return 0;
        };
        const auto rc1 = expert_matvec(wg, sc_->ffn_in, sc_->gf,
                                       config_.expert_ff, H, false);
        const auto rc2 = expert_matvec(wu, sc_->ffn_in, sc_->uf,
                                       config_.expert_ff, H, false);
        if (rc1 != 0 || rc2 != 0) {
          if (error)
            *error = "MoE gate/up matvec failed";
          return false;
        }
        SituAct(sc_->gf, sc_->uf, sc_->act,
                n * static_cast<int>(use) * static_cast<int>(config_.expert_ff),
                config_.situ_beta, config_.situ_linear_beta, stream_);
        if (expert_matvec(wd, sc_->act, sc_->parts, H, config_.expert_ff,
                          true) != 0) {
          if (error)
            *error = "MoE down matvec failed";
          return false;
        }
        for (int t = 0; t < n; ++t) {
          MoeCombine(sc_->parts + static_cast<std::size_t>(t) * use * H,
                     sc_->weights + t * use, sc_->ffn_out + t * H, H, use,
                     stream_);
        }
      }
      if (shared_npu) {
        const auto& g = npu_lin_[l].shg;
        const auto& u = npu_lin_[l].shu;
        NpuFinish(shared_op);
        for (int sl = 0; sl < g.slices; ++sl) {
          const std::uint32_t valid = std::min<std::uint32_t>(
              1024U, static_cast<std::uint32_t>(g.n_out) - sl * 1024U);
          qfn::NpuDecodeOutput(npu_->Staged(stream_, B().npu_c[sl]), sc_->gf, n,
                               npu_->n(), config_.expert_ff, sl * npu_->n(),
                               false, stream_, valid == 1024U ? 0U : valid);
        }
        for (int sl = 0; sl < u.slices; ++sl) {
          const std::uint32_t valid = std::min<std::uint32_t>(
              1024U, static_cast<std::uint32_t>(u.n_out) - sl * 1024U);
          qfn::NpuDecodeOutput(npu_->Staged(stream_, B().npu_c[g.slices + sl]),
                               sc_->uf, n, npu_->n(), config_.expert_ff,
                               sl * npu_->n(), false, stream_,
                               valid == 1024U ? 0U : valid);
        }
      } else if ((PreqBegin(sc_->ffn_in, n, static_cast<int>(H)), false) ||
                 !Matvec(get(l, "ffn_gate_shexp.weight"), sc_->ffn_in, sc_->gf,
                         config_.expert_ff, n, error) ||
                 !Matvec(get(l, "ffn_up_shexp.weight"), sc_->ffn_in, sc_->uf,
                         config_.expert_ff, n, error)) {
        PreqEnd();
        return false;
      }
      PreqEnd();
      SituAct(sc_->gf, sc_->uf, sc_->act, n * config_.expert_ff,
              config_.situ_beta, config_.situ_linear_beta, stream_);
      if (!Matvec(get(l, "ffn_down_shexp.weight"), sc_->act, sc_->dpart, H, n,
                  error)) {
        return false;
      }
      AddVectors(sc_->ffn_out, sc_->dpart, sc_->ffn_out, n * H, stream_);
    }
    HcApply(xcur, sc_->ffn_out, sc_->post, sc_->comb, ycur, stream_, n);
    std::swap(xcur, ycur);
    capture_tap(l, false);
    if (out != nullptr)
      out->Produce(l, stream_);
  }

  const int hr = std::min(n, static_cast<int>(kHeadRows));
  if (head && options_.head) {
    for (int t = 0; t < hr; ++t) {
      HeadMeanNorm(xcur + (n - hr + t) * hcdim,
                   static_cast<const float*>(Find("output_norm.weight")->data),
                   sc_->head + t * H, H, hc, config_.rms_eps, stream_);
    }
    if (!Matvec(*Find("output.weight"), sc_->head, sc_->logits,
                config_.vocab_size, hr, error)) {
      return false;
    }
    ArgmaxRows(sc_->logits, hr, config_.vocab_size, sc_->argmax, stream_);
  }
  if (head)
    last_rows_ = hr;
  return true;
}

bool Executor::SaveState(Session& session, std::string* error) {
  const std::size_t inner = config_.KdaDim();
  const std::size_t conv_bytes = 3 * (config_.ssm_conv_kernel - 1) * inner * 4;
  const std::size_t state_bytes = static_cast<std::size_t>(config_.num_heads) *
                                  config_.kda_head_dim * config_.kda_head_dim *
                                  4;
  if (session.snap_conv.empty()) {
    session.snap_conv.assign(layers_, nullptr);
    session.snap_state.assign(layers_, nullptr);
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!config_.IsRda(l))
        continue;
      if (hipMalloc(reinterpret_cast<void**>(&session.snap_conv[l]),
                    conv_bytes) != hipSuccess ||
          hipMalloc(reinterpret_cast<void**>(&session.snap_state[l]),
                    state_bytes) != hipSuccess) {
        if (error)
          *error = "state snapshot alloc failed";
        return false;
      }
    }
  }
  if (rec_.empty()) {
    rec_.assign(layers_, Rec{});
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!config_.IsRda(l))
        continue;
      auto& r = rec_[l];
      float** ptrs[] = {&r.q, &r.k, &r.v, &r.gproj, &r.g2};
      for (float** p : ptrs) {
        if (hipMalloc(reinterpret_cast<void**>(p), kMaxVerify * inner * 4) !=
            hipSuccess) {
          if (error)
            *error = "operand record alloc failed";
          return false;
        }
      }
      float** small[] = {&r.beta, &r.delta};
      for (float** p : small) {
        if (hipMalloc(reinterpret_cast<void**>(p),
                      kMaxVerify * config_.num_heads * 4) != hipSuccess) {
          if (error)
            *error = "operand record alloc failed";
          return false;
        }
      }
    }
  }
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!config_.IsRda(l))
      continue;
    hipMemcpyAsync(session.snap_conv[l], session.layers[l].conv, conv_bytes,
                   hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(session.snap_state[l], session.layers[l].state, state_bytes,
                   hipMemcpyDeviceToDevice, stream_);
  }
  session.snap_position = session.position;
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error) {
  if (session.snap_conv.empty() || rec_.empty() || keep == 0 ||
      keep > static_cast<std::uint32_t>(kMaxVerify)) {
    if (error)
      *error = "rollback without a saved state";
    return false;
  }
  const std::size_t inner = config_.KdaDim();
  const std::size_t conv_bytes = 3 * (config_.ssm_conv_kernel - 1) * inner * 4;
  const std::size_t state_bytes = static_cast<std::size_t>(config_.num_heads) *
                                  config_.kda_head_dim * config_.kda_head_dim *
                                  4;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!config_.IsRda(l))
      continue;
    auto& st = session.layers[l];
    hipMemcpyAsync(st.conv, session.snap_conv[l], conv_bytes,
                   hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(st.state, session.snap_state[l], state_bytes,
                   hipMemcpyDeviceToDevice, stream_);
    const auto& r = rec_[l];
    auto get = [&](const char* nm) -> const DeviceTensor& {
      return *Find(P(l, nm));
    };
    if (RdaPrefill(
            static_cast<const float*>(get("ssm_conv1d_q.weight").data),
            static_cast<const float*>(get("ssm_conv1d_k.weight").data),
            static_cast<const float*>(get("ssm_conv1d_v.weight").data), r.q,
            r.k, r.v, st.conv, r.gproj,
            static_cast<const float*>(get("ssm_dt.bias").data),
            static_cast<const float*>(get("ssm_a").data), r.beta, r.delta, r.g2,
            static_cast<const float*>(get("ssm_norm.weight").data), st.state,
            sc_->rda_out, static_cast<int>(keep), config_.num_heads,
            config_.kda_gate_lower_bound, config_.rms_eps, stream_) != 0) {
      if (error)
        *error = "rollback replay failed";
      return false;
    }
  }
  session.position = session.snap_position + keep;
  return true;
}

std::vector<std::int32_t> Executor::GreedyRows(std::uint32_t rows) {
  std::vector<std::int32_t> out(rows, 0);
  hipMemcpyAsync(out.data(), LS()->argmax, rows * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);
  return out;
}

bool Executor::ReadLogitRows(std::uint32_t first, std::uint32_t count,
                             float* dst) {
  if (first + count > static_cast<std::uint32_t>(last_rows_))
    return false;
  hipMemcpyAsync(
      dst, LS()->logits + static_cast<std::size_t>(first) * config_.vocab_size,
      static_cast<std::size_t>(count) * config_.vocab_size * 4,
      hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);
  return true;
}

bool Executor::WriteLogitRow(std::uint32_t row, const float* src) {
  if (row >= static_cast<std::uint32_t>(kHeadRows))
    return false;
  std::int32_t best = 0;
  for (std::uint32_t i = 1; i < config_.vocab_size; ++i)
    best = src[i] > src[best] ? static_cast<std::int32_t>(i) : best;
  hipMemcpyAsync(
      LS()->logits + static_cast<std::size_t>(row) * config_.vocab_size, src,
      static_cast<std::size_t>(config_.vocab_size) * 4, hipMemcpyHostToDevice,
      stream_);
  hipMemcpyAsync(LS()->argmax + row, &best, sizeof(best), hipMemcpyHostToDevice,
                 stream_);
  hipStreamSynchronize(stream_);
  last_rows_ = std::max<int>(last_rows_, static_cast<int>(row) + 1);
  return true;
}

const DeviceTensor* Executor::TokenEmbedding() const {
  return Find("token_embd.weight");
}

const DeviceTensor* Executor::Output() const {
  return Find("output.weight");
}

bool Executor::Logits(Session&, float* dst, std::string* error) {
  if (hipMemcpyAsync(dst,
                     LS()->logits + static_cast<std::size_t>(last_rows_ - 1) *
                                        config_.vocab_size,
                     config_.vocab_size * sizeof(float), hipMemcpyDeviceToHost,
                     stream_) != hipSuccess) {
    if (error)
      *error = "logit copy failed";
    return false;
  }
  hipStreamSynchronize(stream_);
  return true;
}

std::vector<float> Executor::Logits(Session&) {
  std::vector<float> out(config_.vocab_size);
  if (hipMemcpy(out.data(),
                LS()->logits + static_cast<std::size_t>(last_rows_ - 1) *
                                   config_.vocab_size,
                config_.vocab_size * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return {};
  }
  return out;
}

std::int32_t Executor::Argmax(Session&, std::string*) {
  std::int32_t out = 0;
  hipMemcpy(&out, LS()->argmax + (last_rows_ - 1), sizeof(out),
            hipMemcpyDeviceToHost);
  return out;
}

std::size_t Executor::SnapshotBytes(const Session& session) const {
  const std::size_t inner = config_.KdaDim();
  const std::size_t conv = 3 * (config_.ssm_conv_kernel - 1) * inner * 4;
  const std::size_t state = static_cast<std::size_t>(config_.num_heads) *
                            config_.kda_head_dim * config_.kda_head_dim * 4;
  const std::size_t mla = static_cast<std::size_t>(session.position) *
                          (config_.kv_lora_rank + config_.qk_rope_head_dim) * 4;
  std::size_t total = 0;
  for (std::uint32_t l = 0; l < layers_; ++l)
    total += config_.IsRda(l) ? conv + state : mla;
  return total;
}

bool Executor::SaveSnapshot(const Session& session, std::span<std::uint8_t> out,
                            std::string* error) {
  if (out.size() != SnapshotBytes(session)) {
    if (error)
      *error = "snapshot buffer size mismatch";
    return false;
  }
  const std::size_t inner = config_.KdaDim();
  const std::size_t conv = 3 * (config_.ssm_conv_kernel - 1) * inner * 4;
  const std::size_t state = static_cast<std::size_t>(config_.num_heads) *
                            config_.kda_head_dim * config_.kda_head_dim * 4;
  const std::size_t lat =
      static_cast<std::size_t>(session.position) * config_.kv_lora_rank * 4;
  const std::size_t rope =
      static_cast<std::size_t>(session.position) * config_.qk_rope_head_dim * 4;
  const auto t_wait = std::chrono::steady_clock::now();
  hipStreamSynchronize(stream_);
  const auto t_copy = std::chrono::steady_clock::now();
  std::uint8_t* dst = out.data();
  auto copy = [&](const void* src, std::size_t bytes) {
    if (bytes == 0)
      return true;
    const bool ok = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
                                   stream_) == hipSuccess;
    dst += bytes;
    return ok;
  };
  for (std::uint32_t l = 0; l < layers_; ++l) {
    const auto& st = session.layers[l];
    const bool ok = config_.IsRda(l)
                        ? copy(st.conv, conv) && copy(st.state, state)
                        : copy(st.lat, lat) && copy(st.extra, rope);
    if (!ok) {
      if (error)
        *error = "snapshot copy failed";
      return false;
    }
  }
  const bool done = hipStreamSynchronize(stream_) == hipSuccess;
  if (std::getenv("RAON1_SNAP_LOG") != nullptr)
    std::fprintf(stderr, "raon1: snapshot wait-for-gpu %.1f ms, copy %.1f ms\n",
                 std::chrono::duration<double, std::milli>(t_copy - t_wait)
                     .count(),
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_copy)
                     .count());
  return done;
}

bool Executor::RestoreSnapshot(Session& session, std::uint32_t position,
                               std::span<const std::uint8_t> in,
                               std::string* error) {
  session.position = position;
  InvalidateFull();
  if (in.size() != SnapshotBytes(session) || position > options_.max_context) {
    if (error)
      *error = "snapshot payload size mismatch";
    return false;
  }
  const std::size_t inner = config_.KdaDim();
  const std::size_t conv = 3 * (config_.ssm_conv_kernel - 1) * inner * 4;
  const std::size_t state = static_cast<std::size_t>(config_.num_heads) *
                            config_.kda_head_dim * config_.kda_head_dim * 4;
  const std::size_t lat =
      static_cast<std::size_t>(position) * config_.kv_lora_rank * 4;
  const std::size_t rope =
      static_cast<std::size_t>(position) * config_.qk_rope_head_dim * 4;
  hipStreamSynchronize(stream_);
  const std::uint8_t* src = in.data();
  auto copy = [&](void* dst, std::size_t bytes) {
    if (bytes == 0)
      return true;
    const bool ok =
        hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) == hipSuccess;
    src += bytes;
    return ok;
  };
  for (std::uint32_t l = 0; l < layers_; ++l) {
    auto& st = session.layers[l];
    const bool ok = config_.IsRda(l)
                        ? copy(st.conv, conv) && copy(st.state, state)
                        : copy(st.lat, lat) && copy(st.extra, rope);
    if (!ok) {
      if (error)
        *error = "snapshot restore failed";
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::raon1::rocm
