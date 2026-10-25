#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

#include <fcntl.h>
#include <hipblas/hipblas.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/rocm/embed.hpp"
#include "src/models/glm53_flash/kernels/rocm/kda.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"
#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"
#include "src/models/glm53_flash/kernels/rocm/moe.hpp"
#include "src/models/glm53_flash/kernels/rocm/npu_decode.hpp"
#include "src/models/glm53_flash/kernels/rocm/router.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

extern "C" int ds4_mmq_init(int device);

namespace gufo::models::glm53_flash::rocm {

namespace {

using core::GgmlType;
namespace qfn = gufo::models::qwen38_flash_next::rocm;

constexpr std::size_t kTailMargin = 4096;
constexpr std::size_t kUploadChunk = std::size_t{128} << 20;
constexpr std::size_t kMatvecScratchBytes = 1 << 20;
constexpr int kHidden = kHcHidden;
constexpr int kStreams = kHcStreams;
constexpr int kDenseFf = 12288;
constexpr int kExpertFf = 2048;
constexpr int kQkvSlices = 3 * kKdaInner / 1024;

constexpr int kShexpSlices = 2 * kExpertFf / 1024;
constexpr int kQLora = 1536;
// Routed expert tile classes: experts with big buckets take wider token tiles
// (fewer weight decodes); 64-row paired gate/up tiles are not supported.
constexpr std::uint32_t kGateWideMin = 96;
constexpr std::uint32_t kGateWideRows = 256;
constexpr std::uint32_t kGateNarrowRows = 128;
constexpr std::uint32_t kDownWideMin = 129;
constexpr std::uint32_t kDownWideRows = 192;
constexpr std::uint32_t kDownNarrowRows = 128;
constexpr std::uint32_t kPadRows = 16;
// Batches at least this wide use the W8A8 GEMMs.
constexpr int kWideBatch = 32;
constexpr int kMaxExpertTokens = 8;
constexpr int kFfnScratch = std::max(kDenseFf, kMoeUsed* kExpertFf);
constexpr int kMlaQ = kDsaHeads * 256;
constexpr float kRmsEps = 1e-5F;
constexpr float kKNormEps = 1e-6F;

int TypeCode(GgmlType t) {
  switch (t) {
    case GgmlType::kF32:
      return kTypeF32;
    case GgmlType::kQ8_0:
      return kTypeQ8_0;
    case GgmlType::kQ5_K:
      return kTypeQ5_K;
    case GgmlType::kQ6_K:
      return kTypeQ6_K;
    case GgmlType::kIQ2_XXS:
      return kTypeIQ2_XXS;
    case GgmlType::kIQ2_S:
      return kTypeIQ2_S;
    default:
      return -1;
  }
}

template<typename T>
bool Alloc(T** ptr, std::size_t count, std::string* error) {
  if (hipMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)) !=
      hipSuccess) {
    if (error != nullptr) {
      *error =
          "hipMalloc failed (" + std::to_string(count * sizeof(T)) + " bytes)";
    }
    *ptr = nullptr;
    return false;
  }
  if (hipMemset(*ptr, 0, count * sizeof(T)) != hipSuccess) {
    if (error != nullptr)
      *error = "hipMemset failed";
    return false;
  }
  return true;
}

std::string P(std::uint32_t layer, const char* name) {
  return "blk." + std::to_string(layer) + "." + name;
}

}  // namespace

struct Executor::LayerWeights {
  bool kda{false};
  bool dense{false};
  const DeviceTensor *hc_attn_fn{}, *hc_attn_scale{}, *hc_attn_base{},
      *attn_norm{}, *hc_ffn_fn{}, *hc_ffn_scale{}, *hc_ffn_base{}, *ffn_norm{};
  // KDA
  const DeviceTensor *attn_q{}, *attn_k{}, *attn_v{}, *conv_q{}, *conv_k{},
      *conv_v{}, *ssm_f_a{}, *ssm_f_b{}, *ssm_beta{}, *ssm_g_a{}, *ssm_g_b{},
      *ssm_dt{}, *ssm_a{}, *ssm_norm{}, *attn_output{};
  // DSA
  const DeviceTensor *q_a{}, *q_a_norm{}, *q_b{}, *kv_a{}, *kv_a_norm{}, *k_b{},
      *v_b{}, *idx_q_b{}, *idx_k{}, *idx_k_norm_w{}, *idx_k_norm_b{},
      *idx_gate{}, *idx_ape{}, *idx_proj{};
  // FFN
  const DeviceTensor *ffn_gate{}, *ffn_up{}, *ffn_down{};
  const DeviceTensor *router{}, *router_bias{}, *exp_gate{}, *exp_up{},
      *exp_down{}, *sh_gate{}, *sh_up{}, *sh_down{};
};

struct Executor::Scratch {
  void* matvec{nullptr};
  float* x[2]{nullptr, nullptr};
  float *y{}, *post{}, *comb{}, *block{}, *ffn_out{};
  // KDA
  float *kq{}, *kk{}, *kv{}, *decay{}, *beta{}, *fa{}, *fproj{}, *braw{}, *ga{},
      *g2{}, *o{};
  // DSA
  float *qr{}, *q{}, *lat{}, *iq{}, *ik{}, *gt{}, *wp{}, *scores{}, *qabs{},
      *ctx{}, *vout{}, *v_heads{};
  void* head_half{};
  int* sel{};
  // FFN
  float *gate{}, *up{}, *act{};
  int* tok_ids{};
  float *res_last{}, *hc_mixes{}, *rlogits{}, *wts{}, *parts{}, *sg{}, *su{},
      *sact{}, *shared{};
  int* ids{};
  // Head
  float *head_in{}, *logits{}, *dflash_capture{};
  int* argmax{};
  int* pos_dev{};
  char* q8_arena{};
  std::size_t q8_arena_bytes{0};
  // Routed-expert F16 route (prefill).
  void *x_half{}, *act_half{}, *parts_half{};
  std::uint32_t* counts{};
  std::int32_t *bounds{}, *cursors{}, *rows_token{}, *rows_slot{},
      *tiles_pair{}, *tiles_down{};
  std::uint32_t* host_counts{};
  std::int32_t* host_tiles{};
  std::uint32_t tile_capacity{0};
  long long score_stride{0};
  std::vector<void*> all;
};

Executor::Session::~Session() {
  for (auto& log : kda_log) {
    for (void* p :
         {static_cast<void*>(log.raw_q), static_cast<void*>(log.raw_k),
          static_cast<void*>(log.raw_v), static_cast<void*>(log.decay),
          static_cast<void*>(log.beta)}) {
      if (p != nullptr)
        hipFree(p);
    }
  }
  for (float* p : saved_conv) {
    if (p != nullptr)
      hipFree(p);
  }
  for (float* p : saved_state) {
    if (p != nullptr)
      hipFree(p);
  }
  for (auto& l : layers) {
    for (void* p :
         {static_cast<void*>(l.conv), static_cast<void*>(l.state),
          static_cast<void*>(l.cache.lat_half), static_cast<void*>(l.cache.ik),
          static_cast<void*>(l.cache.gate),
          static_cast<void*>(l.cache.pooled)}) {
      if (p != nullptr)
        hipFree(p);
    }
  }
}

Executor::~Executor() {
  hipDeviceSynchronize();
  DropGraph();
  if (capture_stream_ != nullptr)
    hipStreamDestroy(capture_stream_);
  if (staging_ != nullptr)
    hipHostFree(staging_);
  for (auto& [key, pair] : hc_fn_split_) {
    hipFree(pair.first);
    hipFree(pair.second);
  }
  for (std::uint16_t* p : router_hi_)
    if (p != nullptr)
      hipFree(p);
  for (std::uint16_t* p : router_lo_)
    if (p != nullptr)
      hipFree(p);
  for (const auto* table :
       {npu_tab_[0], npu_tab_[1], npu_sh_tab_[0], npu_sh_tab_[1]})
    if (table != nullptr)
      hipFree(const_cast<std::uint8_t**>(table));
  if (blas_ != nullptr)
    hipblasDestroy(static_cast<hipblasHandle_t>(blas_));
  if (q8_fallback_ != nullptr)
    hipFree(q8_fallback_);
  for (auto& [name, t] : tensors_) {
    if (t.q8_owned && t.q8 != nullptr)
      hipFree(t.q8);
    if (t.f16 != nullptr)
      hipFree(t.f16);
    if (t.data != nullptr)
      hipFree(t.data);
  }
  if (scratch_) {
    for (void* p : scratch_->all) {
      hipFree(p);
    }
    if (scratch_->host_counts != nullptr)
      hipHostFree(scratch_->host_counts);
    if (scratch_->host_tiles != nullptr)
      hipHostFree(scratch_->host_tiles);
  }
}

// Streams a tensor to the device through a pinned staging buffer with large
// preads. Copying straight from the mapping faults the file in 128 KiB at a
// time, which both crawls and thrashes the page cache against the device
// allocations on a unified-memory machine.
bool Executor::CopyToDevice(void* dst, const core::GgufTensorInfo& t,
                            std::string* error) {
  const char* src = static_cast<const char*>(t.data);
  int fd = -1;
  std::uint64_t file_offset = 0;
  for (const auto& region : reader_->GetMappedRegions()) {
    const char* base = static_cast<const char*>(region.data);
    if (src >= base && src < base + region.size) {
      fd = region.file_descriptor;
      file_offset = static_cast<std::uint64_t>(src - base);
      break;
    }
  }
  if (fd < 0) {
    if (hipMemcpy(dst, src, t.size_bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
      if (error != nullptr)
        *error = "upload failed for " + std::string(t.name);
      return false;
    }
    return true;
  }
  if (staging_ == nullptr &&
      hipHostMalloc(&staging_, kUploadChunk, hipHostMallocDefault) !=
          hipSuccess) {
    if (error != nullptr)
      *error = "pinned staging allocation failed";
    return false;
  }
  for (std::size_t done = 0; done < t.size_bytes;) {
    const std::size_t n = std::min(kUploadChunk, t.size_bytes - done);
    std::size_t got = 0;
    while (got < n) {
      const ssize_t r = pread(fd, static_cast<char*>(staging_) + got, n - got,
                              static_cast<off_t>(file_offset + done + got));
      if (r <= 0) {
        if (error != nullptr)
          *error = "read failed for " + std::string(t.name);
        return false;
      }
      got += static_cast<std::size_t>(r);
    }
    if (hipMemcpy(static_cast<char*>(dst) + done, staging_, n,
                  hipMemcpyHostToDevice) != hipSuccess) {
      if (error != nullptr)
        *error = "upload failed for " + std::string(t.name);
      return false;
    }
    posix_fadvise(fd, static_cast<off_t>(file_offset + done),
                  static_cast<off_t>(n), POSIX_FADV_DONTNEED);
    done += n;
  }
  return true;
}

bool Executor::Upload(const std::string& name, std::string* error) {
  const auto* t = reader_->FindTensor(name);
  if (t == nullptr) {
    if (error != nullptr)
      *error = "missing tensor " + name;
    return false;
  }
  const int code = TypeCode(t->type);
  if (code < 0) {
    if (error != nullptr)
      *error = "unsupported tensor type for " + name;
    return false;
  }
  DeviceTensor d;
  d.type = code;
  d.k = static_cast<std::uint32_t>(t->dimensions[0]);
  d.rows = t->dimensions.size() > 1
               ? static_cast<std::uint32_t>(t->dimensions[1])
               : 1;
  d.experts = t->dimensions.size() > 2
                  ? static_cast<std::uint32_t>(t->dimensions[2])
                  : 1;
  d.slab_bytes = t->size_bytes / d.experts;
  if (hipMalloc(&d.data, t->size_bytes + kTailMargin) != hipSuccess) {
    if (error != nullptr)
      *error = "hipMalloc failed for " + name;
    return false;
  }
  if (!CopyToDevice(d.data, *t, error)) {
    hipFree(d.data);
    return false;
  }
  if (d.experts == 1 && name.rfind("blk.", 0) == 0 && d.k % 32 == 0) {
    if (d.type == kTypeQ8_0) {
      d.q8 = d.data;
    } else if (d.type == kTypeQ6_K || d.type == kTypeQ5_K) {
      const std::size_t bytes =
          static_cast<std::size_t>(d.rows) * (d.k / 32) * 34;
      if (hipMalloc(&d.q8, bytes + kTailMargin) == hipSuccess) {
        d.q8_owned = true;
        gufo::models::qwen38_flash_next::rocm::RequantToQ8_0(
            d.data, static_cast<std::uint32_t>(t->type), d.k, d.rows, d.q8,
            stream_);
        hipStreamSynchronize(stream_);
      } else {
        d.q8 = nullptr;
      }
    }
  }
  if (d.experts > 1 && d.type == kTypeQ6_K &&
      (name.find("attn_k_b") != std::string::npos ||
       name.find("attn_v_b") != std::string::npos)) {
    const std::size_t rows_total = static_cast<std::size_t>(d.rows) * d.experts;
    std::vector<__half> half(rows_total * d.k);
    std::vector<float> row(d.k);
    const std::size_t row_bytes = t->size_bytes / rows_total;
    for (std::size_t r = 0; r < rows_total; ++r) {
      quant::DequantizeQ6_K(static_cast<const char*>(t->data) + r * row_bytes,
                            row.data(), d.k);
      for (std::uint32_t i = 0; i < d.k; ++i) {
        half[r * d.k + i] = __float2half(row[i]);
      }
    }
    if (hipMalloc(&d.f16, half.size() * sizeof(__half)) == hipSuccess) {
      hipMemcpy(d.f16, half.data(), half.size() * sizeof(__half),
                hipMemcpyHostToDevice);
    }
  }
  tensors_.emplace(name, d);
  return true;
}

const DeviceTensor* Executor::Find(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

std::unique_ptr<Executor> Executor::Create(const std::string& path,
                                           const ExecutorOptions& options,
                                           std::string* error) {
  std::unique_ptr<Executor> ex(new Executor());
  ex->reader_ = core::GgufReader::OpenFile(path, error);
  if (!ex->reader_)
    return nullptr;
  auto config = ParseConfig(*ex->reader_, error);
  if (!config)
    return nullptr;
  ex->config_ = *config;
  ex->options_ = options;
#ifdef GUFO_NPU
  if (const char* dir = std::getenv("GUFO_NPU_DIR");
      dir != nullptr && dir[0] != '\0') {
    qfn::NpuGemm::Config cfg;
    cfg.xclbin =
        std::string(dir) + "/runtime_k/final_4096xKx1024_128x64x128.xclbin";
    cfg.insts =
        std::string(dir) + "/runtime_k/insts_4096x4096x1024_128x64x128.txt";
    cfg.k = kHidden;
    std::string npu_error;
    ex->npu_ = qfn::NpuGemm::Create(cfg, &npu_error);
    if (ex->npu_ == nullptr)
      std::fprintf(stderr, "NPU off: %s\n", npu_error.c_str());
  }
#endif
  ex->profile_ = std::getenv("GUFO_GLM53_PROFILE") != nullptr;
  ex->layers_ = options.layers == 0
                    ? ex->config_.TrunkLayers()
                    : std::min(options.layers, ex->config_.TrunkLayers());
  if (options.top_pools == 0 || options.top_pools > kDsaTopPools) {
    if (error != nullptr)
      *error = "top_pools out of range";
    return nullptr;
  }
  if (hipSetDevice(0) != hipSuccess) {
    if (error != nullptr)
      *error = "no HIP device";
    return nullptr;
  }
  {
    hipblasHandle_t handle = nullptr;
    if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
      if (error != nullptr)
        *error = "hipblas init failed";
      return nullptr;
    }
    ex->blas_ = handle;
  }
  if (ds4_mmq_init(0) != 0) {
    if (error != nullptr)
      *error = "mmq init failed";
    return nullptr;
  }
  const bool full = ex->layers_ == ex->config_.TrunkLayers();
  const bool head = options.head && full;
  ex->options_.head = head;
  for (const char* name : {"token_embd.weight"}) {
    if (ex->reader_->FindTensor(name) == nullptr) {
      if (error != nullptr)
        *error = std::string("missing tensor ") + name;
      return nullptr;
    }
  }
  if (!options.dflash_layers.empty()) {
    if (options.dflash_rows == 0) {
      if (error != nullptr)
        *error = "drafter taps need dflash_rows";
      return nullptr;
    }
    for (const auto tap : options.dflash_layers) {
      if (tap == 0 || tap > ex->layers_) {
        if (error != nullptr)
          *error = "drafter tap layer exceeds the loaded trunk";
        return nullptr;
      }
      ex->dflash_taps_.push_back(tap - 1);
    }
  }
  if (!ex->Upload("token_embd.weight", error))
    return nullptr;
  ex->token_embd_dev_ = ex->tensors_.at("token_embd.weight");
  if (head) {
    for (const char* name : {"output_norm.weight", "output.weight"}) {
      if (!ex->Upload(name, error))
        return nullptr;
    }
  }
  for (std::uint32_t layer = 0; layer < ex->layers_; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    for (const auto& t : ex->reader_->GetTensors()) {
      if (t.name.substr(0, prefix.size()) == prefix) {
        if (!ex->Upload(std::string(t.name), error))
          return nullptr;
      }
    }
  }
  if (!ex->Resolve(error) || !ex->AllocScratch(error) ||
      !ex->SetupRouter(error) || !ex->SetupNpu(error))
    return nullptr;
  return ex;
}

bool Executor::SetupRouter(std::string* error) {
  router_hi_.assign(layers_, nullptr);
  router_lo_.assign(layers_, nullptr);
  const std::size_t count = std::size_t{kMoeExperts} * kHidden;
  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    const LayerWeights& w = weights_[layer];
    if (w.dense || w.router == nullptr || w.router->type != kTypeF32)
      continue;
    void* hi = nullptr;
    void* lo = nullptr;
    if (hipMalloc(&hi, count * 2) != hipSuccess ||
        hipMalloc(&lo, count * 2) != hipSuccess) {
      if (error != nullptr)
        *error = "router split allocation failed";
      return false;
    }
    router_hi_[layer] = static_cast<std::uint16_t*>(hi);
    router_lo_[layer] = static_cast<std::uint16_t*>(lo);
    SplitF16(w.router->f32(), router_hi_[layer], router_lo_[layer],
             static_cast<long long>(count), nullptr);
  }
  const std::size_t fn_count = std::size_t{kHcMixes} * kStreams * kHidden;
  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    const LayerWeights& w = weights_[layer];
    for (const DeviceTensor* fn : {w.hc_attn_fn, w.hc_ffn_fn}) {
      void* hi = nullptr;
      void* lo = nullptr;
      if (hipMalloc(&hi, fn_count * 2) != hipSuccess ||
          hipMalloc(&lo, fn_count * 2) != hipSuccess) {
        if (error != nullptr)
          *error = "hc split allocation failed";
        return false;
      }
      SplitF16(fn->f32(), static_cast<std::uint16_t*>(hi),
               static_cast<std::uint16_t*>(lo),
               static_cast<long long>(fn_count), nullptr, kHcMixFnScale);
      hc_fn_split_[fn->f32()] = {static_cast<std::uint16_t*>(hi),
                                 static_cast<std::uint16_t*>(lo)};
    }
  }
  return hipDeviceSynchronize() == hipSuccess;
}

bool Executor::SetupNpu(std::string*) {
  if (npu_ == nullptr)
    return true;
  const auto drop = [&](const char* why, const std::string& detail) {
    std::fprintf(stderr, "NPU off: %s %s\n", why, detail.c_str());
    npu_.reset();
    npu_qkv_.clear();
    return true;
  };
  if (bank_b_.x[0] == nullptr)
    return drop("chunk is not the NPU row count", "");
  std::string npu_error;
  for (int bank = 0; bank < 2; ++bank) {
    npu_a_[bank][0] = npu_->NewBuffer(npu_->ABytes(), &npu_error);
    if (npu_a_[bank][0] < 0)
      return drop("buffers", npu_error);
    for (int i = 0; i < kQkvSlices; ++i) {
      const int id = npu_->NewBuffer(npu_->CBytes(), &npu_error);
      if (id < 0)
        return drop("buffers", npu_error);
      npu_c_[bank].push_back(id);
    }
    for (int i = 0; i < kShexpSlices; ++i) {
      const int id = npu_->NewBuffer(npu_->CBytes(), &npu_error);
      if (id < 0)
        return drop("buffers", npu_error);
      npu_sh_c_[bank].push_back(id);
    }
  }
  for (int bank = 0; bank < 2; ++bank) {
    std::vector<const std::uint8_t*> table;
    for (const int id : npu_c_[bank])
      table.push_back(
          static_cast<const std::uint8_t*>(npu_->Staged(nullptr, id)));
    const std::size_t bytes = table.size() * sizeof(table[0]);
    void* dev = nullptr;
    if (hipMalloc(&dev, bytes) != hipSuccess ||
        hipMemcpy(dev, table.data(), bytes, hipMemcpyHostToDevice) !=
            hipSuccess)
      return drop("stream table", "");
    npu_tab_[bank] = static_cast<const std::uint8_t**>(dev);
    std::vector<const std::uint8_t*> sh_table;
    for (const int id : npu_sh_c_[bank])
      sh_table.push_back(
          static_cast<const std::uint8_t*>(npu_->Staged(nullptr, id)));
    const std::size_t sh_bytes = sh_table.size() * sizeof(sh_table[0]);
    void* sh_dev = nullptr;
    if (hipMalloc(&sh_dev, sh_bytes) != hipSuccess ||
        hipMemcpy(sh_dev, sh_table.data(), sh_bytes, hipMemcpyHostToDevice) !=
            hipSuccess)
      return drop("stream table", "");
    npu_sh_tab_[bank] = static_cast<const std::uint8_t**>(sh_dev);
  }
  npu_sh_.assign(layers_, {});
  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    const LayerWeights& w = weights_[layer];
    if (w.dense || w.sh_gate == nullptr || w.sh_up == nullptr ||
        w.sh_gate->rows != kExpertFf || w.sh_up->rows != kExpertFf)
      continue;
    std::vector<int> ids;
    for (const DeviceTensor* t : {w.sh_gate, w.sh_up}) {
      for (std::uint32_t slice = 0; slice < kExpertFf / npu_->n(); ++slice) {
        const int id = npu_->NewBuffer(npu_->BBytes(), &npu_error);
        if (id < 0)
          return drop("weights", npu_error);
        qfn::NpuEncodeWeightsQuant(t->data, static_cast<std::uint32_t>(t->type),
                                   kHidden, slice * npu_->n(), npu_->n(),
                                   kHidden, 0, kHidden, npu_->Device(id),
                                   nullptr);
        ids.push_back(id);
      }
    }
    npu_sh_[layer] = std::move(ids);
  }
  npu_qkv_.assign(layers_, {});
  std::uint32_t staged = 0;
  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    const LayerWeights& w = weights_[layer];
    if (!w.kda || w.attn_q->rows != kKdaInner || w.attn_k->rows != kKdaInner ||
        w.attn_v->rows != kKdaInner)
      continue;
    std::vector<int> ids;
    for (const DeviceTensor* t : {w.attn_q, w.attn_k, w.attn_v}) {
      for (std::uint32_t slice = 0; slice < kKdaInner / npu_->n(); ++slice) {
        const int id = npu_->NewBuffer(npu_->BBytes(), &npu_error);
        if (id < 0)
          return drop("weights", npu_error);
        qfn::NpuEncodeWeightsQuant(t->data, static_cast<std::uint32_t>(t->type),
                                   kHidden, slice * npu_->n(), npu_->n(),
                                   kHidden, 0, kHidden, npu_->Device(id),
                                   nullptr);
        ids.push_back(id);
      }
    }
    npu_qkv_[layer] = std::move(ids);
    ++staged;
  }
  if (hipDeviceSynchronize() != hipSuccess)
    return drop("upload", "");
  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    for (const int id : npu_qkv_[layer])
      npu_->FinalizeWeights(id);
    for (const int id : npu_sh_[layer])
      npu_->FinalizeWeights(id);
    if (!npu_sh_[layer].empty()) {
      const LayerWeights& sw = weights_[layer];
      for (const DeviceTensor* t : {sw.sh_gate, sw.sh_up}) {
        auto* owned = const_cast<DeviceTensor*>(t);
        if (owned->q8_owned && owned->q8 != nullptr) {
          q8_fallback_bytes_ = std::max<std::size_t>(
              q8_fallback_bytes_,
              static_cast<std::size_t>(owned->rows) * (owned->k / 32) * 34);
          hipFree(owned->q8);
          owned->q8 = nullptr;
          owned->q8_owned = false;
        }
      }
    }
    if (npu_qkv_[layer].empty())
      continue;
    const LayerWeights& w = weights_[layer];
    for (const DeviceTensor* t : {w.attn_q, w.attn_k, w.attn_v}) {
      auto* owned = const_cast<DeviceTensor*>(t);
      if (owned->q8_owned && owned->q8 != nullptr) {
        q8_fallback_bytes_ = std::max<std::size_t>(
            q8_fallback_bytes_,
            static_cast<std::size_t>(owned->rows) * (owned->k / 32) * 34);
        hipFree(owned->q8);
        owned->q8 = nullptr;
        owned->q8_owned = false;
      }
    }
  }
  if (!npu_qkv_.empty()) {
    const std::vector<int>& warm =
        npu_qkv_[0].empty() ? npu_qkv_[1] : npu_qkv_[0];
    if (!warm.empty()) {
      npu_->Submit(stream_, npu_a_[0][0], warm.data(), npu_c_[0].data(), 1);
      npu_->Wait(stream_);
      hipStreamSynchronize(stream_);
    }
  }
  if (q8_fallback_bytes_ != 0 &&
      hipMalloc(&q8_fallback_, q8_fallback_bytes_ + kTailMargin) !=
          hipSuccess) {
    q8_fallback_ = nullptr;
    q8_fallback_bytes_ = 0;
  }
  std::fprintf(stderr, "NPU: KDA q/k/v of %u layers\n", staged);
  return true;
}

bool Executor::Resolve(std::string* error) {
  weights_.resize(layers_);
  for (std::uint32_t l = 0; l < layers_; ++l) {
    LayerWeights& w = weights_[l];
    w.kda = config_.IsKda(l);
    w.dense = config_.IsDense(l);
    bool ok = true;
    const auto get = [&](const DeviceTensor*& slot, const char* name) {
      slot = Find(P(l, name));
      if (slot == nullptr) {
        ok = false;
        if (error != nullptr && error->empty()) {
          *error = "missing tensor " + P(l, name);
        }
      }
    };
    get(w.hc_attn_fn, "hc_attn_fn.weight");
    get(w.hc_attn_scale, "hc_attn_scale.weight");
    get(w.hc_attn_base, "hc_attn_base.weight");
    get(w.attn_norm, "attn_norm.weight");
    get(w.hc_ffn_fn, "hc_ffn_fn.weight");
    get(w.hc_ffn_scale, "hc_ffn_scale.weight");
    get(w.hc_ffn_base, "hc_ffn_base.weight");
    get(w.ffn_norm, "ffn_norm.weight");
    get(w.attn_output, "attn_output.weight");
    if (w.kda) {
      get(w.attn_q, "attn_q.weight");
      get(w.attn_k, "attn_k.weight");
      get(w.attn_v, "attn_v.weight");
      get(w.conv_q, "ssm_conv1d_q.weight");
      get(w.conv_k, "ssm_conv1d_k.weight");
      get(w.conv_v, "ssm_conv1d_v.weight");
      get(w.ssm_f_a, "ssm_f_a.weight");
      get(w.ssm_f_b, "ssm_f_b.weight");
      get(w.ssm_beta, "ssm_beta.weight");
      get(w.ssm_g_a, "ssm_g_a.weight");
      get(w.ssm_g_b, "ssm_g_b.weight");
      get(w.ssm_dt, "ssm_dt.bias");
      get(w.ssm_a, "ssm_a");
      get(w.ssm_norm, "ssm_norm.weight");
    } else {
      get(w.q_a, "attn_q_a.weight");
      get(w.q_a_norm, "attn_q_a_norm.weight");
      get(w.q_b, "attn_q_b.weight");
      get(w.kv_a, "attn_kv_a_mqa.weight");
      get(w.kv_a_norm, "attn_kv_a_norm.weight");
      get(w.k_b, "attn_k_b.weight");
      get(w.v_b, "attn_v_b.weight");
      get(w.idx_q_b, "indexer.attn_q_b.weight");
      get(w.idx_k, "indexer.attn_k.weight");
      get(w.idx_k_norm_w, "indexer.k_norm.weight");
      get(w.idx_k_norm_b, "indexer.k_norm.bias");
      get(w.idx_gate, "indexer_compressor_gate.weight");
      get(w.idx_ape, "indexer_compressor_ape.weight");
      get(w.idx_proj, "indexer.proj.weight");
    }
    if (w.dense) {
      get(w.ffn_gate, "ffn_gate.weight");
      get(w.ffn_up, "ffn_up.weight");
      get(w.ffn_down, "ffn_down.weight");
    } else {
      get(w.router, "ffn_gate_inp.weight");
      get(w.router_bias, "exp_probs_b.bias");
      get(w.exp_gate, "ffn_gate_exps.weight");
      get(w.exp_up, "ffn_up_exps.weight");
      get(w.exp_down, "ffn_down_exps.weight");
      get(w.sh_gate, "ffn_gate_shexp.weight");
      get(w.sh_up, "ffn_up_shexp.weight");
      get(w.sh_down, "ffn_down_shexp.weight");
    }
    if (!ok)
      return false;
  }
  return true;
}

bool Executor::AllocScratch(std::string* error) {
  scratch_ = std::make_unique<Scratch>();
  Scratch& s = *scratch_;
  const std::size_t c = options_.prefill_chunk;
  const std::size_t pools = options_.max_context / kDsaPool + 1;
  bool ok = true;
  const auto take = [&](auto** ptr, std::size_t count) {
    if (!ok)
      return;
    ok = Alloc(ptr, count, error);
    if (ok)
      s.all.push_back(static_cast<void*>(*ptr));
  };
  take(reinterpret_cast<char**>(&s.matvec), kMatvecScratchBytes);
  take(&s.x[0], c * kStreams * kHidden);
  take(&s.x[1], c * kStreams * kHidden);
  take(&s.y, c * kHidden);
  take(&s.post, c * kStreams);
  take(&s.comb, c * kStreams * kStreams);
  take(&s.block, c * kHidden);
  take(&s.ffn_out, c * kHidden);
  take(&s.kq, c * kKdaInner);
  take(&s.kk, c * kKdaInner);
  take(&s.kv, c * kKdaInner);
  take(&s.decay, c * kKdaInner);
  take(&s.beta, c * kKdaHeads);
  take(&s.fa, c * kKdaHeadDim);
  take(&s.fproj, c * kKdaInner);
  take(&s.braw, c * kKdaHeads);
  take(&s.ga, c * kKdaHeadDim);
  take(&s.g2, c * kKdaInner);
  take(&s.o, c * kKdaInner);
  take(&s.qr, c * kQLora);
  take(&s.q, c * kMlaQ);
  take(&s.lat, c * kDsaLatent);
  take(&s.iq, c * kDsaIndexHeads * kDsaIndexDim);
  take(&s.ik, c * kDsaIndexDim);
  take(&s.gt, c * kDsaIndexDim);
  take(&s.wp, c * kDsaIndexHeads);
  s.score_stride = static_cast<long long>(pools);
  take(&s.scores, c * pools);
  take(&s.qabs, c * kDsaHeads * kDsaLatent);
  take(&s.ctx, std::size_t{kDsaHeads} * kDsaLatent);
  take(reinterpret_cast<std::uint16_t**>(&s.head_half),
       c * kDsaHeads * kDsaLatent);
  take(&s.v_heads, c * kMlaQ);
  take(&s.vout, c * kMlaQ);
  take(&s.sel, c * kDsaTopPools);
  take(&s.gate, c * kFfnScratch);
  take(&s.up, c * kFfnScratch);
  take(&s.act, c * kFfnScratch);
  take(&s.hc_mixes, c * kHcPreScratch);
  take(&s.res_last, kStreams * kHidden);
  take(&s.tok_ids, c);
  if (npu_ != nullptr && npu_->tokens() == c) {
    take(&bank_b_.x[0], c * kStreams * kHidden);
    take(&bank_b_.x[1], c * kStreams * kHidden);
    take(&bank_b_.y, c * kHidden);
    take(&bank_b_.post, c * kStreams);
    take(&bank_b_.comb, c * kStreams * kStreams);
    take(&bank_b_.hc_mixes, c * kHcPreScratch);
  }
  take(&s.rlogits, c * kMoeExperts);
  take(&s.wts, c * kMoeUsed);
  take(&s.ids, c * kMoeUsed);
  take(&s.parts, c * kMoeUsed * kHidden);
  take(&s.sg, c * kExpertFf);
  take(&s.su, c * kExpertFf);
  take(&s.sact, c * kExpertFf);
  take(&s.shared, c * kHidden);
  const std::size_t slots = c * kMoeUsed;
  const std::size_t compact_rows =
      gufo::models::qwen38_flash_next::rocm::RoutedCompactRows(slots,
                                                               kMoeExperts);
  s.tile_capacity = static_cast<std::uint32_t>(
      (slots + std::size_t{kMoeExperts} * (kPadRows - 1)) / kPadRows +
      kMoeExperts + 1);
  take(reinterpret_cast<std::uint16_t**>(&s.x_half), c * kHidden);
  take(reinterpret_cast<std::uint16_t**>(&s.act_half), slots * kExpertFf);
  take(reinterpret_cast<std::uint16_t**>(&s.parts_half), slots * kHidden);
  take(&s.counts, kMoeExperts);
  take(&s.bounds, kMoeExperts + 1);
  take(&s.cursors, kMoeExperts);
  take(&s.rows_token, compact_rows);
  take(&s.rows_slot, compact_rows);
  take(&s.tiles_pair, s.tile_capacity);
  take(&s.tiles_down, s.tile_capacity);
  if (ok && (hipHostMalloc(reinterpret_cast<void**>(&s.host_counts),
                           kMoeExperts * sizeof(std::uint32_t)) != hipSuccess ||
             hipHostMalloc(reinterpret_cast<void**>(&s.host_tiles),
                           2 * s.tile_capacity * sizeof(std::int32_t)) !=
                 hipSuccess)) {
    ok = false;
    if (error != nullptr)
      *error = "pinned routing buffers failed";
  }
  s.q8_arena_bytes = c * std::size_t{160} * 1024;
  take(&s.q8_arena, s.q8_arena_bytes);
  take(&s.head_in, std::size_t{options_.logit_rows} * kHidden);
  take(&s.argmax, options_.logit_rows);
  take(&s.pos_dev, 1);
  if (options_.dflash_rows != 0) {
    take(&s.dflash_capture, std::size_t{options_.dflash_rows} *
                                options_.dflash_layers.size() * kHidden);
  }
  take(&s.logits, std::size_t{options_.logit_rows} * config_.vocab_size);
  return ok;
}

std::unique_ptr<Executor::Session> Executor::CreateSession(std::string* error) {
  auto s = std::make_unique<Session>();
  s->id = ++session_counter_;
  s->layers.resize(layers_);
  bool ok = true;
  const std::size_t ctx = options_.max_context;
  for (std::uint32_t l = 0; ok && l < layers_; ++l) {
    LayerState& st = s->layers[l];
    if (weights_[l].kda) {
      ok =
          Alloc(&st.conv, 3 * 3 * kKdaInner, error) &&
          Alloc(&st.state,
                static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim * kKdaHeadDim,
                error);
    } else {
      st.cache.ring = static_cast<int>(options_.prefill_chunk) + kDsaPool;
      ok = Alloc(reinterpret_cast<float**>(&st.cache.lat_half),
                 ctx * kDsaLatent / 2, error) &&
           Alloc(&st.cache.ik,
                 std::size_t{static_cast<std::size_t>(st.cache.ring)} *
                     kDsaIndexDim,
                 error) &&
           Alloc(&st.cache.gate,
                 std::size_t{static_cast<std::size_t>(st.cache.ring)} *
                     kDsaIndexDim,
                 error) &&
           Alloc(&st.cache.pooled, (ctx / kDsaPool + 1) * kDsaIndexDim, error);
    }
  }
  return ok ? std::move(s) : nullptr;
}

Executor::Act Executor::Prep(const float* x, int tokens, int k) const {
  if (tokens < kWideBatch) {
    return {x, nullptr};
  }
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  const std::size_t bytes = (qfn::Q8TiledBytes(tokens, k) + 255) / 256 * 256;
  if (arena_used_ + bytes > scratch_->q8_arena_bytes) {
    return {x, nullptr};
  }
  char* at = scratch_->q8_arena + arena_used_;
  arena_used_ += bytes;
  qfn::QuantizeQ8Tiled(x, at, tokens, k, stream_);
  return {x, at};
}

void Executor::Proj(const DeviceTensor& w, const Act& x, float* out,
                    int tokens) const {
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  if (x.q != nullptr && w.q8 != nullptr &&
      qfn::W8A8Gemm(w.q8, x.q, out, tokens, w.rows, w.k, stream_)) {
    return;
  }
  if (x.q != nullptr && w.q8 == nullptr && q8_fallback_ != nullptr &&
      w.type == kTypeQ6_K && w.experts == 1 && w.k % 32 == 0 &&
      static_cast<std::size_t>(w.rows) * (w.k / 32) * 34 <=
          q8_fallback_bytes_) {
    qfn::RequantToQ8_0(w.data, static_cast<std::uint32_t>(GgmlType::kQ6_K), w.k,
                       w.rows, q8_fallback_, stream_);
    if (qfn::W8A8Gemm(q8_fallback_, x.q, out, tokens, w.rows, w.k, stream_)) {
      return;
    }
  }
  Gemm(w.type, w.data, x.f, out, static_cast<int>(w.rows), tokens,
       static_cast<int>(w.k), scratch_->matvec, stream_);
}

// out[h][t][r] = sum_k W_h[r][k] * in[h][t][k] for the 64 head slabs of a
// stacked [64][rows][k] tensor: one strided-batched GEMM over F16 copies of
// the slabs and the F16 activations.
void Executor::HeadGemm(const DeviceTensor& w, void* out, bool half_out, int T,
                        int k, int rows) const {
  const Scratch& b = *scratch_;
  const auto handle = static_cast<hipblasHandle_t>(blas_);
  hipblasSetStream(handle, stream_);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  hipblasGemmStridedBatchedEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, rows, T, k, &alpha, w.f16,
      HIPBLAS_R_16F, k, static_cast<long long>(rows) * k, b.head_half,
      HIPBLAS_R_16F, k, static_cast<long long>(T) * k, &beta, out,
      half_out ? HIPBLAS_R_16F : HIPBLAS_R_32F, rows,
      static_cast<long long>(T) * rows, kDsaHeads, HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT);
}

void Executor::KdaBlock(std::uint32_t layer, Session& s, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  LayerState& st = s.layers[layer];
  arena_used_ = 0;
  const Act y = Prep(b.y, T, kHidden);
  if (qkv_bank_ >= 0) {
    NpuQkvFinish(qkv_bank_);
  } else {
    Proj(*w.attn_q, y, b.kq, T);
    Proj(*w.attn_k, y, b.kk, T);
    Proj(*w.attn_v, y, b.kv, T);
  }
  const bool record = recording_ && !s.kda_log.empty() &&
                      T <= static_cast<int>(options_.logit_rows);
  if (record) {
    const std::size_t bytes =
        static_cast<std::size_t>(T) * kKdaInner * sizeof(float);
    hipMemcpyAsync(s.kda_log[layer].raw_q, b.kq, bytes, hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(s.kda_log[layer].raw_k, b.kk, bytes, hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(s.kda_log[layer].raw_v, b.kv, bytes, hipMemcpyDeviceToDevice,
                   stream_);
  }
  if (qkv_bank_ >= 0) {
    KdaConvStreams(npu_tab_[qkv_bank_], w.conv_q->f32(), w.conv_k->f32(),
                   w.conv_v->f32(), st.conv, b.kq, b.kk, b.kv, T, stream_);
  } else {
    KdaConv(b.kq, b.kk, b.kv, st.conv, w.conv_q->f32(), w.conv_k->f32(),
            w.conv_v->f32(), stream_, T);
  }
  Proj(*w.ssm_f_a, y, b.fa, T);
  Proj(*w.ssm_f_b, Prep(b.fa, T, kKdaHeadDim), b.fproj, T);
  Proj(*w.ssm_beta, y, b.braw, T);
  KdaGates(b.fproj, w.ssm_dt->f32(), w.ssm_a->f32(), b.braw, b.decay, b.beta,
           stream_, T);
  if (record) {
    hipMemcpyAsync(s.kda_log[layer].decay, b.decay,
                   static_cast<std::size_t>(T) * kKdaInner * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(s.kda_log[layer].beta, b.beta,
                   static_cast<std::size_t>(T) * kKdaHeads * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
  }
  KdaRecur(b.kq, b.kk, b.kv, b.decay, b.beta, st.state, b.o, stream_, T);
  Proj(*w.ssm_g_a, y, b.ga, T);
  Proj(*w.ssm_g_b, Prep(b.ga, T, kKdaHeadDim), b.g2, T);
  KdaOut(b.o, w.ssm_norm->f32(), b.g2, stream_, T);
  Proj(*w.attn_output, Prep(b.o, T, kKdaInner), b.block, T);
}

void Executor::DsaBlock(std::uint32_t layer, Session& s, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  LayerState& st = s.layers[layer];
  arena_used_ = 0;
  const Act y = Prep(b.y, T, kHidden);
  Proj(*w.q_a, y, b.qr, T);
  RmsNorm(b.qr, w.q_a_norm->f32(), kQLora, kRmsEps, stream_, T);
  const Act qr = Prep(b.qr, T, kQLora);
  Proj(*w.q_b, qr, b.q, T);
  Proj(*w.kv_a, y, b.lat, T);
  RmsNorm(b.lat, w.kv_a_norm->f32(), kDsaLatent, kRmsEps, stream_, T);
  Proj(*w.idx_q_b, qr, b.iq, T);
  Proj(*w.idx_k, y, b.ik, T);
  LayerNorm(b.ik, w.idx_k_norm_w->f32(), w.idx_k_norm_b->f32(), kDsaIndexDim,
            kKNormEps, stream_, T);
  Proj(*w.idx_gate, y, b.gt, T);
  Proj(*w.idx_proj, y, b.wp, T);
  Mark(5);
  if (T >= 3) {
    const int pos0 = static_cast<int>(s.position);
    const int top = static_cast<int>(options_.top_pools);
    DsaAppendBatch(st.cache, b.lat, b.ik, b.gt, w.idx_ape->f32(), pos0, T,
                   stream_);
    DsaScoresBatch(b.iq, b.wp, st.cache.pooled, pos0, T, b.score_stride,
                   b.scores, stream_);
    Mark(9);
    DsaSelectBatch(b.scores, b.score_stride, pos0, T, top, b.sel, stream_);
    Mark(6);
    // attn_k_b / attn_v_b are 64 stacked per-head slabs: one GEMM per head
    // over all tokens, in the head-major layout the attention kernel reads.
    HeadMajor(b.q, static_cast<std::uint16_t*>(b.head_half), T, kDsaHeads, 256,
              stream_);
    std::uint16_t* q_half = reinterpret_cast<std::uint16_t*>(b.v_heads);
    HeadGemm(*w.k_b, q_half, true, T, 256, kDsaLatent);
    Mark(7);
    DsaAttendBatch(q_half, st.cache.lat_half, b.sel, pos0, T, top, 1.0F / 16.0F,
                   static_cast<std::uint16_t*>(b.head_half), stream_);
    Mark(8);
    HeadGemm(*w.v_b, b.v_heads, true, T, kDsaLatent, 256);
    TokenMajor(reinterpret_cast<const std::uint16_t*>(b.v_heads), b.vout, T,
               kDsaHeads, 256, stream_);
    Mark(7);
  } else if (T == 1) {
    const int top = static_cast<int>(options_.top_pools);
    DsaAppendDev(st.cache, b.lat, b.ik, b.gt, w.idx_ape->f32(), b.pos_dev,
                 stream_);
    DsaScoresDev(b.iq, b.wp, st.cache.pooled,
                 static_cast<int>(options_.max_context / kDsaPool), b.scores,
                 b.pos_dev, stream_);
    DsaSelectDev(b.scores, top, b.sel, b.pos_dev, stream_);
    MatvecBatched(w.k_b->type, w.k_b->data, b.q, b.qabs, kDsaLatent, 256,
                  kDsaHeads, b.matvec, stream_);
    DsaAttendDev(b.qabs, st.cache.lat_half, b.sel, top, 1.0F / 16.0F, b.ctx,
                 b.pos_dev, stream_);
    MatvecBatched(w.v_b->type, w.v_b->data, b.ctx, b.vout, 256, kDsaLatent,
                  kDsaHeads, b.matvec, stream_);
  } else {
    for (int t = 0; t < T; ++t) {
      const int pos = static_cast<int>(s.position) + t;
      DsaAppend(st.cache, b.lat + static_cast<std::size_t>(t) * kDsaLatent,
                b.ik + static_cast<std::size_t>(t) * kDsaIndexDim,
                b.gt + static_cast<std::size_t>(t) * kDsaIndexDim,
                w.idx_ape->f32(), pos, stream_);
      const int npool = (pos + 1) / kDsaPool;
      const int nsel = std::min<int>(npool, options_.top_pools);
      DsaScores(
          b.iq + static_cast<std::size_t>(t) * kDsaIndexHeads * kDsaIndexDim,
          b.wp + static_cast<std::size_t>(t) * kDsaIndexHeads, st.cache.pooled,
          npool, b.scores, stream_);
      DsaSelect(b.scores, npool, static_cast<int>(options_.top_pools), b.sel,
                stream_);
      MatvecBatched(w.k_b->type, w.k_b->data,
                    b.q + static_cast<std::size_t>(t) * kMlaQ, b.qabs,
                    kDsaLatent, 256, kDsaHeads, b.matvec, stream_);
      DsaAttend(b.qabs, st.cache.lat_half, b.sel, nsel, pos, 1.0F / 16.0F,
                b.ctx, stream_);
      MatvecBatched(w.v_b->type, w.v_b->data, b.ctx,
                    b.vout + static_cast<std::size_t>(t) * kMlaQ, 256,
                    kDsaLatent, kDsaHeads, b.matvec, stream_);
    }
  }
  Proj(*w.attn_output, Prep(b.vout, T, kMlaQ), b.block, T);
}

// Prefill experts on the Flash-Next route: expert-sorted row compaction, the
// paired gate/up F16 GEMM with clamped SwiGLU, then the down GEMM, whose F16
// rows MoeCombineF16 sums with the shared expert.
void Executor::RoutedExpertsF16(const LayerWeights& w, int T) const {
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  const Scratch& b = *scratch_;
  const std::uint32_t slots = static_cast<std::uint32_t>(T) * kMoeUsed;
  qfn::ExpertCounts(b.ids, b.counts, T, kMoeExperts, kMoeUsed, stream_);
  hipMemcpyAsync(b.host_counts, b.counts, kMoeExperts * sizeof(std::uint32_t),
                 hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);
  std::int32_t* host_pair = b.host_tiles;
  std::int32_t* host_down = b.host_tiles + b.tile_capacity;
  std::uint32_t pair_tiles = 0;
  std::uint32_t down_tiles = 0;
  const auto add = [&](std::int32_t* list, std::uint32_t& n, std::uint32_t lo,
                       std::uint32_t hi, std::uint32_t rows) {
    for (int e = 0; e < kMoeExperts; ++e) {
      const std::uint32_t count = b.host_counts[e];
      if (count < lo || count >= hi)
        continue;
      const std::uint32_t padded = (count + kPadRows - 1) / kPadRows * kPadRows;
      for (std::uint32_t j = 0; j < (padded + rows - 1) / rows; ++j) {
        list[n++] = static_cast<std::int32_t>(e | (j << 16));
      }
    }
  };
  constexpr std::uint32_t kAll = ~0U;
  add(host_pair, pair_tiles, kGateWideMin, kAll, kGateWideRows);
  const std::uint32_t gate_wide_tiles = pair_tiles;
  add(host_pair, pair_tiles, 0, kGateWideMin, kGateNarrowRows);
  add(host_down, down_tiles, kDownWideMin, kAll, kDownWideRows);
  const std::uint32_t down_wide_tiles = down_tiles;
  add(host_down, down_tiles, 0, kDownWideMin, kDownNarrowRows);
  hipMemcpyAsync(b.tiles_pair, host_pair, pair_tiles * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  hipMemcpyAsync(b.tiles_down, host_down, down_tiles * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  qfn::RoutedCompact(b.ids, b.counts, b.bounds, b.cursors, b.rows_token,
                     b.rows_slot, T, kMoeUsed, kMoeExperts, stream_);
  const auto weight_type = [](int code) {
    return code == kTypeIQ2_S ? qfn::WeightType::kIQ2_S
                              : qfn::WeightType::kIQ2_XXS;
  };
  const auto gate_pair = [&](std::uint32_t offset, std::uint32_t count,
                             std::uint32_t rows) {
    if (count == 0)
      return;
    qfn::RoutedGatedF16Gemm(
        w.exp_gate->data, w.exp_up->data, weight_type(w.exp_gate->type),
        static_cast<const __half*>(b.x_half), b.tiles_pair + offset, count,
        rows, b.bounds, b.rows_token, b.rows_slot,
        static_cast<__half*>(b.act_half), kExpertFf, kHidden, stream_);
  };
  gate_pair(0, gate_wide_tiles, kGateWideRows);
  gate_pair(gate_wide_tiles, pair_tiles - gate_wide_tiles, kGateNarrowRows);
  const auto down = [&](std::uint32_t offset, std::uint32_t count,
                        std::uint32_t rows) {
    if (count == 0)
      return;
    qfn::RoutedF16Gemm(
        w.exp_down->data, weight_type(w.exp_down->type),
        static_cast<const __half*>(b.act_half), b.tiles_down + offset, count,
        rows, b.bounds, b.rows_slot, b.rows_slot, nullptr, nullptr,
        static_cast<__half*>(b.parts_half), kHidden, kExpertFf, stream_);
  };
  down(0, down_wide_tiles, kDownWideRows);
  down(down_wide_tiles, down_tiles - down_wide_tiles, kDownNarrowRows);
  (void)slots;
}

void Executor::NpuShexpQueue(std::uint32_t layer) const {
  const Scratch& b = *scratch_;
  const int bank = shexp_bank_;
  qfn::NpuEncodeActivations(static_cast<const __half*>(b.x_half), kHidden,
                            npu_->tokens(), kHidden, 0, kHidden,
                            npu_->Device(npu_a_[bank][0]), stream_, false);
  npu_->Submit(stream_, npu_a_[bank][0], npu_sh_[layer].data(),
               npu_sh_c_[bank].data(), kShexpSlices);
  npu_sh_ticket_[bank] = npu_->Ticket();
}

void Executor::NpuShexpFinish() const {
  const Scratch& b = *scratch_;
  const int bank = shexp_bank_;
  npu_->Wait(stream_, npu_sh_ticket_[bank]);
  NpuDecodeFused(npu_sh_tab_[bank], b.sg, b.su, b.su,
                 static_cast<int>(npu_->tokens()), kShexpSlices,
                 kShexpSlices / 2, kExpertFf, stream_);
}

void Executor::FfnBlock(std::uint32_t layer, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  arena_used_ = 0;
  const Act y = Prep(b.y, T, kHidden);
  if (w.dense) {
    Proj(*w.ffn_gate, y, b.gate, T);
    Proj(*w.ffn_up, y, b.up, T);
    SwigluClamp(b.gate, b.up, b.act, T * kDenseFf, stream_);
    Proj(*w.ffn_down, Prep(b.act, T, kDenseFf), b.ffn_out, T);
    return;
  }
  if (T >= kWideBatch && w.router->type == kTypeF32) {
    RouterGemm(b.y, router_hi_[layer], router_lo_[layer], b.rlogits, T,
               stream_);
  } else {
    Proj(*w.router, Act{b.y, nullptr}, b.rlogits, T);
  }
  MoeRoute(b.rlogits, w.router_bias->f32(), config_.expert_weights_scale, b.ids,
           b.wts, stream_, T);
  const bool shexp_npu = shexp_bank_ >= 0 && T > 1 &&
                         T == static_cast<int>(npu_->tokens()) &&
                         layer < npu_sh_.size() && !npu_sh_[layer].empty();
  if (T <= kMaxExpertTokens) {
    if (w.exp_gate->type == w.exp_up->type) {
      MatvecExpertsPairTokens(w.exp_gate->type, w.exp_gate->data,
                              w.exp_up->data, b.y, b.gate, b.up, kExpertFf,
                              kHidden, kMoeExperts, kMoeUsed, T, b.ids,
                              b.matvec, stream_);
    } else {
      MatvecExpertsTokens(w.exp_gate->type, w.exp_gate->data, b.y, b.gate,
                          kExpertFf, kHidden, kMoeExperts, kMoeUsed, T, b.ids,
                          false, b.matvec, stream_);
      MatvecExpertsTokens(w.exp_up->type, w.exp_up->data, b.y, b.up, kExpertFf,
                          kHidden, kMoeExperts, kMoeUsed, T, b.ids, false,
                          b.matvec, stream_);
    }
    SwigluClamp(b.gate, b.up, b.act, T * kMoeUsed * kExpertFf, stream_);
    MatvecExpertsTokens(w.exp_down->type, w.exp_down->data, b.act, b.parts,
                        kHidden, kExpertFf, kMoeExperts, kMoeUsed, T, b.ids,
                        true, b.matvec, stream_);
  } else {
    if (!res_half_)
      ToHalf(b.y, b.x_half, static_cast<long long>(T) * kHidden, stream_);
    if (shexp_npu)
      NpuShexpQueue(layer);
    RoutedExpertsF16(w, T);
  }
  if (shexp_npu) {
    NpuShexpFinish();
  } else {
    Proj(*w.sh_gate, y, b.sg, T);
    Proj(*w.sh_up, y, b.su, T);
  }
  SwigluClamp(b.sg, b.su, b.sact, T * kExpertFf, stream_);
  Proj(*w.sh_down, Prep(b.sact, T, kExpertFf), b.shared, T);
  if (T <= kMaxExpertTokens) {
    MoeCombine(b.parts, b.wts, b.shared, b.ffn_out, stream_, T);
  } else if (!res_half_) {
    MoeCombineF16(b.parts_half, b.wts, b.shared, b.ffn_out, stream_, T);
  }
}

bool Executor::EmbedRows(const std::int32_t* tokens, int T, float* dst,
                         std::string* error) {
  const auto* embd = reader_->FindTensor("token_embd.weight");
  const std::size_t row_bytes = embd->size_bytes / embd->dimensions[1];
  for (int t = 0; t < T; ++t) {
    if (tokens[t] < 0 ||
        static_cast<std::uint64_t>(tokens[t]) >= embd->dimensions[1]) {
      if (error != nullptr)
        *error = "token out of range";
      return false;
    }
  }
  hipMemcpyAsync(scratch_->tok_ids, tokens,
                 static_cast<std::size_t>(T) * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  EmbedTokens(token_embd_dev_.data, static_cast<std::uint32_t>(embd->type),
              row_bytes, scratch_->tok_ids, dst, res_half_, T, stream_);
  hipStreamSynchronize(stream_);
  return true;
}

int Executor::FfnRows(std::uint32_t layer, int T, int logit_rows) const {
  if (layer + 1 != layers_ || !options_.head || options_.layers != 0)
    return T;
  if (std::find(dflash_taps_.begin(), dflash_taps_.end(), layer) !=
      dflash_taps_.end())
    return T;
  return logit_rows;
}

void* Executor::ResidualAt(float* base, std::size_t rows_offset) const {
  const std::size_t elems = rows_offset * kStreams * kHidden;
  return res_half_ ? static_cast<void*>(reinterpret_cast<std::uint16_t*>(base) +
                                        elems)
                   : static_cast<void*>(base + elems);
}

void Executor::HcPostStep(const void* x, const float* block, void* next,
                          int T) const {
  const Scratch& b = *scratch_;
  if (res_half_) {
    HcPost(static_cast<const std::uint16_t*>(x), block, b.post, b.comb,
           static_cast<std::uint16_t*>(next), stream_, T);
  } else {
    HcPost(static_cast<const float*>(x), block, b.post, b.comb,
           static_cast<float*>(next), stream_, T);
  }
}

void Executor::RunLayer(std::uint32_t layer, Session& session, int T, int& cur,
                        bool attn_prepared, int ffn_rows) {
  Scratch& b = *scratch_;
  const LayerWeights& w = weights_[layer];
  if (!attn_prepared) {
    HcPreStep(ResidualAt(b.x[cur], 0), *w.hc_attn_fn, w.hc_attn_scale->f32(),
              w.hc_attn_base->f32(), w.attn_norm->f32(), T);
  }
  Mark(0);
  if (w.kda) {
    KdaBlock(layer, session, T);
    Mark(1);
  } else {
    DsaBlock(layer, session, T);
    Mark(2);
  }
  HcPostStep(ResidualAt(b.x[cur], 0), b.block, ResidualAt(b.x[1 - cur], 0), T);
  cur = 1 - cur;

  if (ffn_rows > 0) {
    const std::size_t first = static_cast<std::size_t>(T - ffn_rows);
    HcPreStep(ResidualAt(b.x[cur], first), *w.hc_ffn_fn, w.hc_ffn_scale->f32(),
              w.hc_ffn_base->f32(), w.ffn_norm->f32(), ffn_rows);
    Mark(0);
    FfnBlock(layer, ffn_rows);
    Mark(w.dense ? 3 : 4);
    if (res_half_ && !w.dense && ffn_rows > 1) {
      HcPostMoe(static_cast<const std::uint16_t*>(ResidualAt(b.x[cur], first)),
                b.parts_half, b.wts, b.shared, b.post, b.comb,
                static_cast<std::uint16_t*>(ResidualAt(b.x[1 - cur], first)),
                stream_, ffn_rows);
    } else {
      HcPostStep(ResidualAt(b.x[cur], first), b.ffn_out,
                 ResidualAt(b.x[1 - cur], first), ffn_rows);
    }
    cur = 1 - cur;
  }
  if (!dflash_taps_.empty()) {
    const auto tap = std::find(dflash_taps_.begin(), dflash_taps_.end(), layer);
    if (tap != dflash_taps_.end()) {
      const int rows = std::min<int>(T, options_.dflash_rows);
      const long long stride =
          static_cast<long long>(dflash_taps_.size()) * kHidden;
      const long long offset =
          static_cast<long long>(tap - dflash_taps_.begin()) * kHidden;
      if (res_half_) {
        MeanStreams(reinterpret_cast<const std::uint16_t*>(b.x[cur]), T - rows,
                    rows, b.dflash_capture, stride, offset, stream_);
      } else {
        MeanStreams(b.x[cur], T - rows, rows, b.dflash_capture, stride, offset,
                    stream_);
      }
      capture_rows_ = static_cast<std::uint32_t>(rows);
      capture_position_ =
          session.position + static_cast<std::uint32_t>(T - rows);
    }
  }
  Mark(0);
}

void Executor::FinishHead(float* residual, int T, int logit_rows) {
  Scratch& b = *scratch_;
  if (res_half_) {
    HalfToFloat(reinterpret_cast<const std::uint16_t*>(residual) +
                    static_cast<std::size_t>(T - 1) * kStreams * kHidden,
                b.res_last, kStreams * kHidden, stream_);
    last_residual_ = b.res_last;
  } else {
    last_residual_ =
        residual + static_cast<std::size_t>(T - 1) * kStreams * kHidden;
  }
  if (logit_rows > 0 && options_.head) {
    const DeviceTensor& norm = *Find("output_norm.weight");
    const DeviceTensor& out = *Find("output.weight");
    const std::size_t first = static_cast<std::size_t>(T - logit_rows);
    if (res_half_) {
      HeadNorm(static_cast<const std::uint16_t*>(ResidualAt(residual, first)),
               norm.f32(), b.head_in, stream_, logit_rows);
    } else {
      HeadNorm(static_cast<const float*>(ResidualAt(residual, first)),
               norm.f32(), b.head_in, stream_, logit_rows);
    }
    Proj(out, Act{b.head_in, nullptr}, b.logits, logit_rows);
    last_logit_rows_ = logit_rows;
  }
}

bool Executor::Forward(Session& session, const std::int32_t* tokens, int T,
                       int logit_rows, std::string* error) {
  if (T <= 0 || T > static_cast<int>(options_.prefill_chunk)) {
    if (error != nullptr)
      *error = "bad chunk size";
    return false;
  }
  if (session.position + static_cast<std::uint32_t>(T) > options_.max_context) {
    if (error != nullptr)
      *error = "context full";
    return false;
  }
  Scratch& b = *scratch_;
  int cur = 0;
  res_half_ = T >= kWideBatch && !std::getenv("GUFO_GLM53_F32_RESIDUAL");
  if (!EmbedRows(tokens, T, b.x[cur], error))
    return false;

  if (T == 1) {
    const int pos = static_cast<int>(session.position);
    hipMemcpy(b.pos_dev, &pos, sizeof(int), hipMemcpyHostToDevice);
  }
  bool cancelled = false;
  const auto run = [&]() {
    profile_mark_ = std::chrono::steady_clock::now();
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      if (T > 1 && Cancelled()) {
        cancelled = true;
        return;
      }
      RunLayer(layer, session, T, cur, false, FfnRows(layer, T, logit_rows));
    }
    FinishHead(b.x[cur], T, logit_rows);
  };
  const bool graphable = T == 1 && !recording_ && !profile_;
  if (!graphable || !graph_warm_) {
    run();
    graph_warm_ = graph_warm_ || graphable;
  } else {
    if (graph_exec_ == nullptr || graph_session_ != session.id ||
        graph_logit_rows_ != logit_rows) {
      DropGraph();
      hipGraph_t graph = nullptr;
      if (capture_stream_ == nullptr) {
        hipStreamCreateWithFlags(&capture_stream_, hipStreamNonBlocking);
      }
      const hipStream_t launch_stream = stream_;
      stream_ = capture_stream_;
      hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
      run();
      const hipError_t ended = hipStreamEndCapture(stream_, &graph);
      stream_ = launch_stream;
      const hipError_t made =
          ended == hipSuccess
              ? hipGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0)
              : ended;
      if (made != hipSuccess) {
        if (error != nullptr)
          *error = std::string("HIP graph capture failed: ") +
                   hipGetErrorString(made);
        return false;
      }
      hipGraphDestroy(graph);
      graph_session_ = session.id;
      graph_logit_rows_ = logit_rows;
    }
    hipGraphLaunch(graph_exec_, stream_);
    last_residual_ = b.x[0];
    if (logit_rows > 0 && options_.head) {
      last_logit_rows_ = logit_rows;
    }
    if (!dflash_taps_.empty()) {
      capture_rows_ = 1;
      capture_position_ = session.position;
    }
  }
  if (cancelled) {
    hipStreamSynchronize(stream_);
    if (error != nullptr)
      *error = "cancelled";
    return false;
  }
  session.position += static_cast<std::uint32_t>(T);
  if (hipStreamSynchronize(stream_) != hipSuccess ||
      hipGetLastError() != hipSuccess) {
    if (error != nullptr)
      *error = "HIP error during forward";
    return false;
  }
  return true;
}

void Executor::DropGraph() {
  if (graph_exec_ != nullptr) {
    hipGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  graph_session_ = 0;
}

void Executor::HcPreStep(const void* x, const DeviceTensor& fn,
                         const float* scale, const float* base,
                         const float* norm_w, int T) const {
  const Scratch& b = *scratch_;
  if (res_half_) {
    const auto split = hc_fn_split_.find(fn.f32());
    HcPre(static_cast<const std::uint16_t*>(x), fn.f32(), scale, base, norm_w,
          config_.hc_eps, b.y, b.post, b.comb, b.hc_mixes, stream_, T,
          static_cast<std::uint16_t*>(b.x_half),
          split != hc_fn_split_.end() ? split->second.first : nullptr,
          split != hc_fn_split_.end() ? split->second.second : nullptr);
  } else {
    HcPre(static_cast<const float*>(x), fn.f32(), scale, base, norm_w,
          config_.hc_eps, b.y, b.post, b.comb, b.hc_mixes, stream_, T);
  }
}

void Executor::Mark(int bucket) const {
  if (!profile_)
    return;
  hipStreamSynchronize(stream_);
  const auto now = std::chrono::steady_clock::now();
  profile_seconds_[bucket] +=
      std::chrono::duration<double>(now - profile_mark_).count();
  profile_mark_ = now;
}

void Executor::PrintProfile() {
  if (!profile_)
    return;
  static const char* kNames[] = {
      "hc",       "kda",        "dsa",          "ffn-dense",  "ffn-moe",
      "dsa:proj", "dsa:select", "dsa:headgemm", "dsa:attend", "dsa:scores"};
  double total = 0.0;
  for (const double v : profile_seconds_)
    total += v;
  for (int i = 0; i < 10; ++i) {
    if (profile_seconds_[i] == 0.0)
      continue;
    std::printf("  %-10s %7.3f s %5.1f%%\n", kNames[i], profile_seconds_[i],
                100.0 * profile_seconds_[i] / std::max(total, 1e-9));
  }
  std::fill(std::begin(profile_seconds_), std::end(profile_seconds_), 0.0);
}

bool Executor::Evaluate(Session& session, std::int32_t token,
                        std::string* error) {
  return Forward(session, &token, 1, 1, error);
}

void Executor::NpuQkvQueue(int bank, int T) {
  const Scratch& b = *scratch_;
  const int a = npu_a_[bank][0];
  qfn::NpuEncodeActivations(static_cast<const __half*>(b.x_half), kHidden,
                            static_cast<std::uint32_t>(T), kHidden, 0, kHidden,
                            npu_->Device(a), stream_, false);
  npu_->Submit(stream_, a, npu_qkv_[current_layer_].data(), npu_c_[bank].data(),
               kQkvSlices);
  npu_ticket_[bank] = npu_->Ticket();
}

void Executor::NpuQkvFinish(int bank) const {
  npu_->Wait(stream_, npu_ticket_[bank]);
}

void Executor::SwapBank() {
  Scratch& b = *scratch_;
  std::swap(b.x[0], bank_b_.x[0]);
  std::swap(b.x[1], bank_b_.x[1]);
  std::swap(b.y, bank_b_.y);
  std::swap(b.post, bank_b_.post);
  std::swap(b.comb, bank_b_.comb);
  std::swap(b.hc_mixes, bank_b_.hc_mixes);
}

bool Executor::ForwardPair(Session& session, const std::int32_t* tokens,
                           int count, int logit_rows, std::string* error) {
  const int T = count / 2;
  if (npu_ == nullptr || !npu_->Healthy() ||
      T != static_cast<int>(npu_->tokens()) || count != 2 * T ||
      T > static_cast<int>(options_.prefill_chunk) ||
      session.position + static_cast<std::uint32_t>(count) >
          options_.max_context ||
      logit_rows > T) {
    if (error != nullptr)
      *error = "pair batch not supported";
    return false;
  }
  Scratch& b = *scratch_;
  const std::uint32_t pos0 = session.position;
  res_half_ = !std::getenv("GUFO_GLM53_F32_RESIDUAL");
  if (!EmbedRows(tokens, T, b.x[0], error))
    return false;
  SwapBank();
  const bool second = EmbedRows(tokens + T, T, b.x[0], error);
  SwapBank();
  if (!second)
    return false;
  int cur[2] = {0, 0};
  bool prepared[2] = {false, false};
  const auto prefetch = [&](int bank, std::uint32_t layer) {
    prepared[bank] = false;
    if (layer >= layers_ || npu_qkv_[layer].empty())
      return;
    const LayerWeights& w = weights_[layer];
    HcPreStep(ResidualAt(b.x[cur[bank]], 0), *w.hc_attn_fn,
              w.hc_attn_scale->f32(), w.hc_attn_base->f32(), w.attn_norm->f32(),
              T);
    current_layer_ = layer;
    NpuQkvQueue(bank, T);
    prepared[bank] = true;
  };
  prefetch(0, 0);
  SwapBank();
  prefetch(1, 0);
  SwapBank();
  profile_mark_ = std::chrono::steady_clock::now();
  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    if (Cancelled()) {
      hipStreamSynchronize(stream_);
      if (error != nullptr)
        *error = "cancelled";
      return false;
    }
    for (int bank = 0; bank < 2; ++bank) {
      if (bank == 1)
        SwapBank();
      session.position = pos0 + static_cast<std::uint32_t>(bank * T);
      qkv_bank_ = prepared[bank] ? bank : -1;
      shexp_bank_ = bank;
      RunLayer(layer, session, T, cur[bank], prepared[bank],
               FfnRows(layer, T, bank == 1 ? logit_rows : 0));
      qkv_bank_ = -1;
      shexp_bank_ = -1;
      prefetch(bank, layer + 1);
      if (bank == 1)
        SwapBank();
    }
  }
  SwapBank();
  session.position = pos0 + static_cast<std::uint32_t>(T);
  FinishHead(b.x[cur[1]], T, logit_rows);
  SwapBank();
  session.position = pos0 + static_cast<std::uint32_t>(count);
  if (hipStreamSynchronize(stream_) != hipSuccess ||
      hipGetLastError() != hipSuccess) {
    if (error != nullptr)
      *error = "HIP error during forward";
    return false;
  }
  if (!npu_->Healthy()) {
    if (error != nullptr)
      *error = "NPU run timed out; the batch result is discarded";
    return false;
  }
  return true;
}

bool Executor::Prefill(Session& session, std::span<const std::int32_t> tokens,
                       std::string* error, std::size_t chunk_size,
                       const ForwardHook& after_forward) {
  if (tokens.empty()) {
    if (error != nullptr)
      *error = "empty prompt";
    return false;
  }
  const std::size_t chunk =
      chunk_size == 0
          ? options_.prefill_chunk
          : std::min<std::size_t>(chunk_size, options_.prefill_chunk);
  const bool pairs = npu_ != nullptr && npu_->Healthy() &&
                     chunk == options_.prefill_chunk && chunk == npu_->tokens();
  for (std::size_t at = 0; at < tokens.size();) {
    if (pairs && tokens.size() - at >= 2 * chunk) {
      const bool last = at + 2 * chunk == tokens.size();
      if (!ForwardPair(session, tokens.data() + at, static_cast<int>(2 * chunk),
                       last ? 1 : 0, error) ||
          (after_forward && !after_forward(error))) {
        return false;
      }
      at += 2 * chunk;
      continue;
    }
    const std::size_t n = std::min(chunk, tokens.size() - at);
    const bool last = at + n == tokens.size();
    if (!Forward(session, tokens.data() + at, static_cast<int>(n), last ? 1 : 0,
                 error) ||
        (after_forward && !after_forward(error))) {
      return false;
    }
    at += n;
  }
  return true;
}

bool Executor::Verify(Session& session, std::span<const std::int32_t> tokens,
                      std::uint32_t rows, std::string* error) {
  if (rows == 0 || rows > options_.logit_rows || rows > tokens.size()) {
    if (error != nullptr)
      *error = "bad verification row count";
    return false;
  }
  return Forward(session, tokens.data(), static_cast<int>(tokens.size()),
                 static_cast<int>(rows), error);
}

std::vector<std::int32_t> Executor::ArgmaxRows(std::uint32_t rows) {
  std::vector<std::int32_t> out(rows);
  ::gufo::models::glm53_flash::rocm::ArgmaxRows(
      scratch_->logits, static_cast<int>(rows),
      static_cast<int>(config_.vocab_size), scratch_->argmax, stream_);
  hipMemcpyAsync(out.data(), scratch_->argmax, rows * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);
  return out;
}

bool Executor::SaveState(Session& session, std::string* error) {
  if (session.saved_conv.empty()) {
    session.saved_conv.assign(layers_, nullptr);
    session.saved_state.assign(layers_, nullptr);
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!weights_[l].kda)
        continue;
      if (!Alloc(&session.saved_conv[l], 3 * 3 * kKdaInner, error) ||
          !Alloc(
              &session.saved_state[l],
              static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim * kKdaHeadDim,
              error)) {
        return false;
      }
    }
  }
  if (session.kda_log.empty()) {
    session.kda_log.resize(layers_);
    const std::size_t cap = options_.logit_rows;
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!weights_[l].kda)
        continue;
      auto& log = session.kda_log[l];
      if (!Alloc(&log.raw_q, cap * kKdaInner, error) ||
          !Alloc(&log.raw_k, cap * kKdaInner, error) ||
          !Alloc(&log.raw_v, cap * kKdaInner, error) ||
          !Alloc(&log.decay, cap * kKdaInner, error) ||
          !Alloc(&log.beta, cap * kKdaHeads, error)) {
        return false;
      }
    }
  }
  session.saved_position = session.position;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!weights_[l].kda)
      continue;
    hipMemcpyAsync(session.saved_conv[l], session.layers[l].conv,
                   3 * 3 * kKdaInner * sizeof(float), hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(session.saved_state[l], session.layers[l].state,
                   static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim *
                       kKdaHeadDim * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
  }
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error) {
  if (session.kda_log.empty() || keep == 0 || keep > options_.logit_rows) {
    if (error != nullptr)
      *error = "nothing recorded to roll back to";
    return false;
  }
  RestoreState(session);
  Scratch& b = *scratch_;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!weights_[l].kda)
      continue;
    const LayerWeights& w = weights_[l];
    const auto& log = session.kda_log[l];
    const std::size_t bytes =
        static_cast<std::size_t>(keep) * kKdaInner * sizeof(float);
    hipMemcpyAsync(b.kq, log.raw_q, bytes, hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(b.kk, log.raw_k, bytes, hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(b.kv, log.raw_v, bytes, hipMemcpyDeviceToDevice, stream_);
    KdaConv(b.kq, b.kk, b.kv, session.layers[l].conv, w.conv_q->f32(),
            w.conv_k->f32(), w.conv_v->f32(), stream_, static_cast<int>(keep));
    KdaRecur(b.kq, b.kk, b.kv, log.decay, log.beta, session.layers[l].state,
             b.o, stream_, static_cast<int>(keep));
  }
  session.position = session.saved_position + keep;
  if (hipStreamSynchronize(stream_) != hipSuccess) {
    if (error != nullptr)
      *error = "HIP error during rollback";
    return false;
  }
  return true;
}

void Executor::RestoreState(Session& session) {
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!weights_[l].kda || session.saved_conv.empty())
      continue;
    hipMemcpyAsync(session.layers[l].conv, session.saved_conv[l],
                   3 * 3 * kKdaInner * sizeof(float), hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(session.layers[l].state, session.saved_state[l],
                   static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim *
                       kKdaHeadDim * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
  }
  hipStreamSynchronize(stream_);
}

const float* Executor::DflashCapture() const {
  return scratch_->dflash_capture;
}

const DeviceTensor* Executor::TokenEmbedding() const {
  return token_embd_dev_.data != nullptr ? &token_embd_dev_ : nullptr;
}

const DeviceTensor* Executor::Output() const {
  return Find("output.weight");
}

std::vector<float> Executor::Logits(const Session&) {
  std::vector<float> out(config_.vocab_size);
  hipMemcpy(out.data(),
            scratch_->logits + static_cast<std::size_t>(last_logit_rows_ - 1) *
                                   config_.vocab_size,
            out.size() * sizeof(float), hipMemcpyDeviceToHost);
  return out;
}

std::vector<float> Executor::Residual(const Session&) {
  std::vector<float> out(kStreams * kHidden);
  hipMemcpy(out.data(), last_residual_, out.size() * sizeof(float),
            hipMemcpyDeviceToHost);
  return out;
}

bool Executor::ReadLogitRows(std::uint32_t first, std::uint32_t count,
                             float* destination) {
  if (count == 0 ||
      first + count > static_cast<std::uint32_t>(last_logit_rows_)) {
    return false;
  }
  return hipMemcpy(destination,
                   scratch_->logits +
                       static_cast<std::size_t>(first) * config_.vocab_size,
                   static_cast<std::size_t>(count) * config_.vocab_size *
                       sizeof(float),
                   hipMemcpyDeviceToHost) == hipSuccess;
}

void Executor::ResetSession(Session& session) {
  for (std::uint32_t l = 0; l < layers_; ++l) {
    LayerState& st = session.layers[l];
    if (st.conv != nullptr) {
      hipMemset(st.conv, 0, 3 * 3 * kKdaInner * sizeof(float));
      hipMemset(st.state, 0,
                static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim *
                    kKdaHeadDim * sizeof(float));
    }
  }
  session.position = 0;
  session.saved_position = 0;
  hipDeviceSynchronize();
}

namespace {

constexpr std::size_t kKdaConvBytes = 3 * 3 * kKdaInner * sizeof(float);
constexpr std::size_t kKdaStateBytes = static_cast<std::size_t>(kKdaHeads) *
                                       kKdaHeadDim * kKdaHeadDim *
                                       sizeof(float);
constexpr std::size_t kLatRowBytes = kDsaLatent * sizeof(std::uint16_t);
constexpr std::size_t kIndexRowBytes = kDsaIndexDim * sizeof(float);

}  // namespace

std::uint64_t Executor::SnapshotBytes(const Session& session) const {
  const std::uint64_t pos = session.position;
  std::uint64_t total = 0;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (weights_[l].kda) {
      total += kKdaConvBytes + kKdaStateBytes;
    } else {
      total += pos * kLatRowBytes + (pos / kDsaPool) * kIndexRowBytes +
               2 * (pos % kDsaPool) * kIndexRowBytes;
    }
  }
  return total;
}

bool Executor::SaveSnapshot(const Session& session, std::span<std::uint8_t> out,
                            std::string* error) {
  if (out.size() != SnapshotBytes(session)) {
    if (error != nullptr)
      *error = "snapshot buffer has the wrong size";
    return false;
  }
  const std::size_t pos = session.position;
  const std::size_t tail = pos % kDsaPool;
  std::uint8_t* at = out.data();
  const auto copy = [&](const void* device, std::size_t bytes) {
    if (bytes == 0)
      return true;
    if (Cancelled()) {
      if (error != nullptr)
        *error = "cancelled";
      return false;
    }
    if (hipMemcpy(at, device, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
      if (error != nullptr)
        *error = "snapshot copy failed";
      return false;
    }
    at += bytes;
    return true;
  };
  hipStreamSynchronize(stream_);
  for (std::uint32_t l = 0; l < layers_; ++l) {
    const LayerState& st = session.layers[l];
    if (weights_[l].kda) {
      if (!copy(st.conv, kKdaConvBytes) || !copy(st.state, kKdaStateBytes))
        return false;
      continue;
    }
    if (!copy(st.cache.lat_half, pos * kLatRowBytes) ||
        !copy(st.cache.pooled, (pos / kDsaPool) * kIndexRowBytes))
      return false;
    for (std::size_t i = 0; i < tail; ++i) {
      const std::size_t slot = (pos - tail + i) % st.cache.ring;
      if (!copy(st.cache.ik + slot * kDsaIndexDim, kIndexRowBytes))
        return false;
    }
    for (std::size_t i = 0; i < tail; ++i) {
      const std::size_t slot = (pos - tail + i) % st.cache.ring;
      if (!copy(st.cache.gate + slot * kDsaIndexDim, kIndexRowBytes))
        return false;
    }
  }
  return true;
}

bool Executor::RestoreSnapshot(Session& session, std::uint32_t position,
                               std::span<const std::uint8_t> in,
                               std::string* error) {
  session.position = position;
  if (in.size() != SnapshotBytes(session) || position > options_.max_context) {
    session.position = 0;
    if (error != nullptr)
      *error = "snapshot does not fit this session";
    return false;
  }
  const std::size_t pos = position;
  const std::size_t tail = pos % kDsaPool;
  const std::uint8_t* at = in.data();
  const auto copy = [&](void* device, std::size_t bytes) {
    if (bytes == 0)
      return true;
    if (Cancelled()) {
      if (error != nullptr)
        *error = "cancelled";
      return false;
    }
    if (hipMemcpy(device, at, bytes, hipMemcpyHostToDevice) != hipSuccess) {
      if (error != nullptr)
        *error = "snapshot restore failed";
      return false;
    }
    at += bytes;
    return true;
  };
  bool ok = true;
  for (std::uint32_t l = 0; ok && l < layers_; ++l) {
    LayerState& st = session.layers[l];
    if (weights_[l].kda) {
      ok = copy(st.conv, kKdaConvBytes) && copy(st.state, kKdaStateBytes);
      continue;
    }
    ok = copy(st.cache.lat_half, pos * kLatRowBytes) &&
         copy(st.cache.pooled, (pos / kDsaPool) * kIndexRowBytes);
    for (std::size_t i = 0; ok && i < tail; ++i) {
      const std::size_t slot = (pos - tail + i) % st.cache.ring;
      ok = copy(st.cache.ik + slot * kDsaIndexDim, kIndexRowBytes);
    }
    for (std::size_t i = 0; ok && i < tail; ++i) {
      const std::size_t slot = (pos - tail + i) % st.cache.ring;
      ok = copy(st.cache.gate + slot * kDsaIndexDim, kIndexRowBytes);
    }
  }
  if (!ok) {
    session.position = 0;
    return false;
  }
  session.saved_position = 0;
  return true;
}

}  // namespace gufo::models::glm53_flash::rocm
