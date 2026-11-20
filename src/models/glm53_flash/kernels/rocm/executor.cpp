#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

#include <fcntl.h>
#include <hipblas/hipblas.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/rocm/embed.hpp"
#include "src/models/glm53_flash/kernels/rocm/kda.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"
#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"
#include "src/models/glm53_flash/kernels/rocm/moe.hpp"
#include "src/models/glm53_flash/kernels/rocm/mtp.hpp"
#include "src/models/glm53_flash/kernels/rocm/npu_decode.hpp"
#include "src/models/glm53_flash/kernels/rocm/router.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

extern "C" int ds4_mmq_init(int device);
extern "C" int ds4_mmq_q4_K_dense_preq(const void*, const void*, std::size_t,
                                       float*, int, int, int, hipStream_t);
extern "C" int ds4_mmq_q6_K_dense_preq(const void*, const void*, std::size_t,
                                       float*, int, int, int, hipStream_t);
extern "C" std::size_t ds4_mmq_q8_1_mmq_bytes(int, int);
extern "C" int ds4_mmq_quantize_q8_1_mmq(const float*, void*, int, int, int,
                                         hipStream_t);

namespace gufo::models::glm53_flash::rocm {

namespace {

using core::GgmlType;
namespace qfn = gufo::models::qwen38_flash_next::rocm;
namespace qfnk = gufo::models::qwen38_flash_next;

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
// Padding rows are skipped in the kernel (live_tok_tiles), so a wider tile
// costs fewer launches, not more MMA work: measured neutral at 2048-token
// banks (pp4096 718 vs 714), keep the tuning proven on 4096-token chunks.
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

// Expert weight layouts RoutedGatedF16Gemm (gate/up pair) and RoutedF16Gemm
// (down) can read. Everything else falls back to the chunked decode matvec.
bool RoutedF16Supports(int code) {
  return code == kTypeQ3_K || code == kTypeQ4_K || code == kTypeQ5_K ||
         code == kTypeIQ2_S || code == kTypeIQ2_XXS;
}

bool RoutedF16DownSupports(int code) {
  return code == kTypeQ3_K || code == kTypeQ4_K || code == kTypeQ5_K ||
         code == kTypeQ8_0 || code == kTypeIQ2_S || code == kTypeIQ2_XXS;
}
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
    case GgmlType::kQ3_K:
      return kTypeQ3_K;
    case GgmlType::kQ4_K:
      return kTypeQ4_K;
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

// Device (GTT) memory accounting, printed with GUFO_MEM_REPORT=1. Strix Halo
// has no VRAM: every byte below competes with the page cache and every other
// process for the same 120 GB, so the split is worth knowing when a run fails
// to load or the NPU's memory gate trips.
enum class Gtt : int {
  kWeights = 0,
  kMirrors,
  kScratch,
  kSession,
  kOther,
  kCount
};
std::uint64_t gtt_bytes[static_cast<int>(Gtt::kCount)]{};
Gtt gtt_current{Gtt::kOther};

class GttScope {
public:
  explicit GttScope(Gtt kind) : previous_(gtt_current) { gtt_current = kind; }
  ~GttScope() { gtt_current = previous_; }

private:
  Gtt previous_;
};

void GttCount(std::size_t bytes) {
  gtt_bytes[static_cast<int>(gtt_current)] += bytes;
}

constexpr const char* kGttNames[] = {"weights", "mirrors", "scratch", "session",
                                     "other"};

void GttReport() {
  double total = 0.0;
  for (int i = 0; i < static_cast<int>(Gtt::kCount); ++i) {
    total += static_cast<double>(gtt_bytes[i]) / (1 << 30);
  }
  std::fprintf(stderr, "GTT: total %.1f GiB", total);
  for (int i = 0; i < static_cast<int>(Gtt::kCount); ++i) {
    if (gtt_bytes[i] != 0) {
      std::fprintf(stderr, ", %s %.1f", kGttNames[i],
                   static_cast<double>(gtt_bytes[i]) / (1 << 30));
    }
  }
  std::fprintf(stderr, "\n");
}

template<typename T>
bool Alloc(T** ptr, std::size_t count, std::string* error) {
  GttCount(count * sizeof(T));
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
  // MTP (NEXTN) block 45 only: the fused embedding+hidden projection and the
  // nextn norms. Set by Resolve for the draft block; null everywhere else.
  const DeviceTensor *nextn_eh_proj{}, *nextn_enorm{}, *nextn_hnorm{},
      *nextn_shn{};
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
  // MTP draft block: the trunk hidden capture (one row per token of the last
  // forward), the packed 8192-wide block inputs and per-row temporaries.
  float *mtp_hidden{}, *mtp_concat{}, *mtp_emb{}, *mtp_tmp{}, *mtp_pack1{},
      *mtp_emb1{};
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
  if (saved_slab != nullptr)
    hipFree(saved_slab);
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
  hipFree(mtp_carry);
  hipFree(mtp_mcarry);
}
Executor::~Executor() {
  hipDeviceSynchronize();
  if (profile_)
    PrintProfile();
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
  hipFree(verification_ids_);
  hipFree(verification_scratch_ids_);
  hipFree(verification_candidates_);
  hipFree(verification_penalties_);
  if (verification_candidates_host_ != nullptr)
    hipHostFree(verification_candidates_host_);
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
  int shard = -1;
  const auto& regions = reader_->GetMappedRegions();
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    const char* base = static_cast<const char*>(region.data);
    if (src >= base && src < base + region.size) {
      fd = region.file_descriptor;
      file_offset = static_cast<std::uint64_t>(src - base);
      shard = static_cast<int>(i);
      break;
    }
  }
  // Fast path: the shared parallel reader keeps host residency at its staging
  // buffers (O_DIRECT) and overlaps the disk read with the device copy.
  if (uploader_ != nullptr && shard >= 0) {
    if (!uploader_->Copy(static_cast<std::uint32_t>(shard), file_offset,
                         t.size_bytes, dst, error)) {
      return false;
    }
    return uploader_->Finish(error);
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
  return UploadImpl(name, name, *reader_, error);
}

bool Executor::UploadFromMtp(const std::string& name, const char* key,
                             std::string* error) {
  return UploadImpl(name, key != nullptr ? key : name, *mtp_reader_, error);
}

bool Executor::UploadImpl(const std::string& name, const std::string& key,
                          const core::GgufReader& reader, std::string* error) {
  const auto* t = reader.FindTensor(name);
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
  gtt_bytes[static_cast<int>(Gtt::kWeights)] += t->size_bytes;
  if (hipMalloc(&d.data, t->size_bytes + kTailMargin) != hipSuccess) {
    if (error != nullptr)
      *error = "hipMalloc failed for " + name;
    return false;
  }
  if (!CopyToDevice(d.data, *t, error)) {
    hipFree(d.data);
    return false;
  }
  // Dense-block Q8_0 mirrors for the quantized GEMMs. The embedding table is
  // included only when it needs re-encoding (a Q4_K table; the embedding and
  // drafter readers know Q6_K/Q8_0 natively), so a Q6_K embedding stays
  // untouched for the DFlash2 drafter.
  const bool embd_needs_q8 = name == "token_embd.weight" &&
                             d.type != kTypeQ6_K && d.type != kTypeQ8_0 &&
                             d.type != kTypeF32;
  if (d.experts == 1 && (name.rfind("blk.", 0) == 0 || embd_needs_q8) &&
      d.k % 32 == 0) {
    if (d.type == kTypeQ8_0) {
      d.q8 = d.data;
    } else if ((d.type == kTypeQ6_K || d.type == kTypeQ5_K ||
                d.type == kTypeQ4_K) &&
               (embd_needs_q8 || !std::getenv("GUFO_GLM53_NO_MIRROR"))) {
      const std::size_t bytes =
          static_cast<std::size_t>(d.rows) * (d.k / 32) * 34;
      gtt_bytes[static_cast<int>(Gtt::kMirrors)] += bytes;
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
    // A Q4_K embedding (RCO) has no embedding-kernel decoder: replace the
    // table with the Q8_0 mirror so both the embedding kernel and the DFlash2
    // drafter read one format. (Native-Q4_K pass-through is pending; the
    // drafter lookup faults on it and re-encoding noise kills acceptance.)
    if (embd_needs_q8 && d.q8_owned && d.q8 != nullptr) {
      hipFree(d.data);
      d.data = d.q8;
      d.type = kTypeQ8_0;
      d.q8 = nullptr;
      d.q8_owned = false;
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
    gtt_bytes[static_cast<int>(Gtt::kMirrors)] += half.size() * sizeof(__half);
    if (hipMalloc(&d.f16, half.size() * sizeof(__half)) == hipSuccess) {
      hipMemcpy(d.f16, half.data(), half.size() * sizeof(__half),
                hipMemcpyHostToDevice);
    }
  }
  tensors_.emplace(key, d);
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
  if (ex->reader_ != nullptr) {
    std::string upload_error;
    ex->uploader_ = gufo::hip::WeightUpload::Create(
        ex->reader_->GetMappedRegions(), &upload_error);
    // A null uploader is not fatal: CopyToDevice falls back to the pread path.
  }
  if (!ex->reader_)
    return nullptr;
  auto config = ParseConfig(*ex->reader_, error);
  if (!config)
    return nullptr;
  ex->config_ = *config;
  // kMoeExperts is compile-time 152 for RCO port
  ex->options_ = options;
#ifdef GUFO_NPU
  bool npu_ok = true;
  const char* npu_dir = std::getenv("GUFO_NPU_DIR");
  if (npu_dir != nullptr && npu_dir[0] != '\0') {
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::size_t kb = 0;
    while (meminfo >> key >> kb) {
      if (key == "MemAvailable:")
        break;
      meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (kb < 32ULL * 1024 * 1024) {
      npu_ok = false;
      std::fprintf(stderr, "NPU off: low memory (%zu MB available)\n",
                   kb / 1024);
    }
  }
  if (npu_ok && npu_dir != nullptr && npu_dir[0] != '\0') {
    // Runtime-M/K program: one XCLBIN serves every (M, K); the pair bank row
    // count selects the instruction file. 2048 lets a 4096-token prompt run as
    // one two-bank pair; GUFO_GLM53_NPU_M=4096 restores the wide-bank form.
    std::size_t npu_m = 2048;
    if (const char* m_env = std::getenv("GUFO_GLM53_NPU_M")) {
      const int m_req = std::atoi(m_env);
      if (m_req == 4096 || m_req == 2048)
        npu_m = static_cast<std::size_t>(m_req);
    }
    qfn::NpuGemm::Config cfg;
    cfg.xclbin = std::string(npu_dir) +
                 "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
    cfg.insts = std::string(npu_dir) + "/runtime_mk/insts_" +
                std::to_string(npu_m) + "x4096x1024_128x64x128.txt";
    cfg.tokens = static_cast<std::uint32_t>(npu_m);
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
  if (!options.mtp_model_path.empty() && !ex->LoadMtp(error))
    return nullptr;
  if (!ex->Resolve(error) || !ex->AllocScratch(error) ||
      !ex->SetupRouter(error) || !ex->SetupNpu(error))
    return nullptr;
  if (std::getenv("GUFO_MEM_REPORT") != nullptr)
    GttReport();
  return ex;
}

bool Executor::LoadMtp(std::string* error) {
  mtp_reader_ = core::GgufReader::OpenFile(options_.mtp_model_path, error);
  if (mtp_reader_ == nullptr)
    return false;
  static constexpr const char* kBlock[] = {"attn_norm.weight",
                                           "attn_q_a.weight",
                                           "attn_q_a_norm.weight",
                                           "attn_q_b.weight",
                                           "attn_kv_a_mqa.weight",
                                           "attn_kv_a_norm.weight",
                                           "attn_k_b.weight",
                                           "attn_v_b.weight",
                                           "attn_output.weight",
                                           "indexer.attn_q_b.weight",
                                           "indexer.attn_k.weight",
                                           "indexer.k_norm.weight",
                                           "indexer.k_norm.bias",
                                           "indexer.proj.weight",
                                           "indexer_compressor_gate.weight",
                                           "indexer_compressor_ape.weight",
                                           "ffn_norm.weight",
                                           "ffn_gate_inp.weight",
                                           "exp_probs_b.bias",
                                           "ffn_gate_exps.weight",
                                           "ffn_up_exps.weight",
                                           "ffn_down_exps.weight",
                                           "ffn_gate_shexp.weight",
                                           "ffn_up_shexp.weight",
                                           "ffn_down_shexp.weight",
                                           "nextn.eh_proj.weight",
                                           "nextn.enorm.weight",
                                           "nextn.hnorm.weight",
                                           "nextn.shared_head_norm.weight"};
  // The sidecar names every block tensor `blk.45.*`, and the executor keeps
  // those keys as-is: block 45 is layer `layers_`, one past the trunk.
  const std::string p = "blk." + std::to_string(layers_) + ".";
  for (const char* suffix : kBlock) {
    const std::string full = p + suffix;
    if (!UploadFromMtp(full, full.c_str(), error))
      return false;
  }
  if (!UploadFromMtp("token_embd.weight", "mtp_token_embd.weight", error) ||
      !UploadFromMtp("output.weight", "mtp_output.weight", error)) {
    return false;
  }
  mtp_.active = true;
  mtp_.layer = layers_;
  mtp_.token_embd = tensors_.at("mtp_token_embd.weight");
  mtp_.output = tensors_.at("mtp_output.weight");
  return true;
}

bool Executor::SetupRouter(std::string* error) {
  router_hi_.assign(layers_, nullptr);
  router_lo_.assign(layers_, nullptr);
  const std::size_t count = static_cast<std::size_t>(experts()) * kHidden;
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
  const char* whole_env = std::getenv("GUFO_NPU_WHOLE");
  if (whole_env != nullptr && whole_env[0] != '\0' && whole_env[0] != '0') {
    // Second instruction stream: the same kernel, reduction length
    // kExpertFf, for the shared expert's down projection.
    std::string sd_error;
    npu_sd_prog_ = npu_->AddProgram(
        std::string(std::getenv("GUFO_NPU_DIR")) + "/runtime_mk/insts_" +
            std::to_string(npu_->tokens()) + "x" + std::to_string(kExpertFf) +
            "x1024_128x64x128.txt",
        &sd_error);
    if (npu_sd_prog_ < 0) {
      std::fprintf(stderr, "NPU whole off: %s\n", sd_error.c_str());
    } else {
      constexpr std::uint32_t kSdSlices = kHidden / 1024;
      for (int bank = 0; bank < 2; ++bank) {
        npu_sd_a_[bank] = npu_->NewBuffer(npu_->ABytes(kExpertFf), &sd_error);
        if (npu_sd_a_[bank] < 0)
          return drop("whole buffers", sd_error);
        for (std::uint32_t i = 0; i < kSdSlices; ++i) {
          const int id = npu_->NewBuffer(npu_->CBytes(), &sd_error);
          if (id < 0)
            return drop("whole buffers", sd_error);
          npu_sd_c_[bank].push_back(id);
        }
        std::vector<const std::uint8_t*> table;
        for (const int id : npu_sd_c_[bank])
          table.push_back(
              static_cast<const std::uint8_t*>(npu_->Staged(nullptr, id)));
        void* dev = nullptr;
        if (hipMalloc(&dev, table.size() * sizeof(table[0])) != hipSuccess ||
            hipMemcpy(dev, table.data(), table.size() * sizeof(table[0]),
                      hipMemcpyHostToDevice) != hipSuccess)
          return drop("whole stream table", "");
        npu_sd_tab_[bank] = static_cast<const std::uint8_t**>(dev);
      }
      npu_sd_.assign(layers_, {});
      std::uint32_t sd_staged = 0;
      for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        const LayerWeights& w = weights_[layer];
        if (w.dense || w.sh_down == nullptr || w.sh_down->rows != kHidden ||
            w.sh_down->k != kExpertFf)
          continue;
        std::vector<int> ids;
        for (std::uint32_t slice = 0; slice < kSdSlices; ++slice) {
          const int id = npu_->NewBuffer(npu_->BBytes(kExpertFf), &sd_error);
          if (id < 0)
            return drop("whole weights", sd_error);
          qfn::NpuEncodeWeightsQuant(
              w.sh_down->data, static_cast<std::uint32_t>(w.sh_down->type),
              kExpertFf, slice * npu_->n(), npu_->n(), kExpertFf, 0, kExpertFf,
              npu_->Device(id), nullptr);
          ids.push_back(id);
        }
        npu_sd_[layer] = std::move(ids);
        ++sd_staged;
      }
      if (hipDeviceSynchronize() != hipSuccess)
        return drop("whole upload", "");
      for (auto& ids : npu_sd_)
        for (const int id : ids)
          npu_->FinalizeWeights(id);
      npu_whole_ = true;
      std::fprintf(stderr, "NPU: whole (sh_down) of %u layers\n", sd_staged);
    }
  }
  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    for (const int id : npu_qkv_[layer])
      npu_->FinalizeWeights(id);
    for (const int id : npu_sh_[layer])
      npu_->FinalizeWeights(id);
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
  weights_.resize(layers_ + (mtp_.active ? 1 : 0));
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
  if (mtp_.active) {
    // The draft block is a plain-residual DSA + MoE layer: no hyper-
    // connection mixers, and it is only ever run one token at a time.
    const std::uint32_t l = mtp_.layer;
    LayerWeights& w = weights_[mtp_.layer];
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
    get(w.attn_norm, "attn_norm.weight");
    get(w.attn_output, "attn_output.weight");
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
    get(w.ffn_norm, "ffn_norm.weight");
    get(w.router, "ffn_gate_inp.weight");
    get(w.router_bias, "exp_probs_b.bias");
    get(w.exp_gate, "ffn_gate_exps.weight");
    get(w.exp_up, "ffn_up_exps.weight");
    get(w.exp_down, "ffn_down_exps.weight");
    get(w.sh_gate, "ffn_gate_shexp.weight");
    get(w.sh_up, "ffn_up_shexp.weight");
    get(w.sh_down, "ffn_down_shexp.weight");
    get(w.nextn_eh_proj, "nextn.eh_proj.weight");
    get(w.nextn_enorm, "nextn.enorm.weight");
    get(w.nextn_hnorm, "nextn.hnorm.weight");
    get(w.nextn_shn, "nextn.shared_head_norm.weight");
    if (!ok)
      return false;
  }
  return true;
}

bool Executor::AllocScratch(std::string* error) {
  const GttScope scope(Gtt::kScratch);
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
  if (npu_ != nullptr && npu_->tokens() <= c) {
    take(&bank_b_.x[0], c * kStreams * kHidden);
    take(&bank_b_.x[1], c * kStreams * kHidden);
    take(&bank_b_.y, c * kHidden);
    take(&bank_b_.post, c * kStreams);
    take(&bank_b_.comb, c * kStreams * kStreams);
    take(&bank_b_.hc_mixes, c * kHcPreScratch);
  }
  const std::size_t n_experts = static_cast<std::size_t>(experts());
  take(&s.rlogits, c * n_experts);
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
                                                               n_experts);
  s.tile_capacity = static_cast<std::uint32_t>(
      (slots + n_experts * (kPadRows - 1)) / kPadRows + n_experts + 1);
  take(reinterpret_cast<std::uint16_t**>(&s.x_half), c * kHidden);
  take(reinterpret_cast<std::uint16_t**>(&s.act_half), slots * kExpertFf);
  take(reinterpret_cast<std::uint16_t**>(&s.parts_half), slots * kHidden);
  take(&s.counts, n_experts);
  take(&s.bounds, n_experts + 1);
  take(&s.cursors, n_experts);
  take(&s.rows_token, compact_rows);
  take(&s.rows_slot, compact_rows);
  take(&s.tiles_pair, s.tile_capacity);
  take(&s.tiles_down, s.tile_capacity);
  if (ok && (hipHostMalloc(reinterpret_cast<void**>(&s.host_counts),
                           n_experts * sizeof(std::uint32_t)) != hipSuccess ||
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
  if (mtp_.active) {
    // EmbedTokens replicates every row across the four hyper-connection
    // streams (4 * kHidden floats per token); the packed block rows are
    // 2 * kHidden wide and cover a whole pair chunk.
    take(&s.mtp_hidden, 2 * c * kHidden);
    take(&s.mtp_concat, 2 * c * kHidden);
    take(&s.mtp_emb, 4 * c * kHidden);
    take(&s.mtp_tmp, kHidden);
    take(&s.mtp_pack1, 2 * kHidden);
    take(&s.mtp_emb1, 4 * kHidden);
  }
  return ok;
}

std::unique_ptr<Executor::Session> Executor::CreateSession(std::string* error) {
  const GttScope scope(Gtt::kSession);
  auto s = std::make_unique<Session>();
  s->id = ++session_counter_;
  s->layers.resize(weights_.size());
  bool ok = true;
  const std::size_t ctx = options_.max_context;
  for (std::uint32_t l = 0; ok && l < s->layers.size(); ++l) {
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
  if (ok && mtp_.active) {
    ok = Alloc(&s->mtp_carry, kHidden, error) &&
         Alloc(&s->mtp_mcarry, kHidden, error);
    if (ok) {
      // A zero carry makes the first advance write the zero-padded entry 0,
      // exactly as prefill priming leaves the never-written cache row 0.
      hipMemset(s->mtp_carry, 0, kHidden * sizeof(float));
      LayerState& st = s->layers[mtp_.layer];
      hipMemset(st.cache.lat_half, 0, kDsaLatent * sizeof(std::uint16_t));
      hipMemset(st.cache.pooled, 0, kDsaIndexDim * sizeof(float));
    }
  }
  if (ok && std::getenv("GUFO_MEM_REPORT") != nullptr)
    GttReport();
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
  if (w.q8 == nullptr && w.experts == 1 && tokens >= kWideBatch &&
      w.k % 256 == 0 && (w.type == kTypeQ4_K || w.type == kTypeQ6_K)) {
    const int slot = w.type == kTypeQ4_K ? 0 : 1;
    const int ggml_code = w.type == kTypeQ4_K ? 12 : 14;
    if (x.mmq[slot] == nullptr) {
      const std::size_t bytes =
          (ds4_mmq_q8_1_mmq_bytes(tokens, static_cast<int>(w.k)) + 255) / 256 *
          256;
      if (arena_used_ + bytes <= scratch_->q8_arena_bytes &&
          ds4_mmq_quantize_q8_1_mmq(x.f, scratch_->q8_arena + arena_used_,
                                    tokens, static_cast<int>(w.k), ggml_code,
                                    stream_) == 0) {
        x.mmq[slot] = scratch_->q8_arena + arena_used_;
        x.mmq_bytes[slot] = bytes;
        arena_used_ += bytes;
      }
    }
    if (x.mmq[slot] != nullptr) {
      const int rc =
          slot == 0
              ? ds4_mmq_q4_K_dense_preq(w.data, x.mmq[slot], x.mmq_bytes[slot],
                                        out, static_cast<int>(w.rows), tokens,
                                        static_cast<int>(w.k), stream_)
              : ds4_mmq_q6_K_dense_preq(w.data, x.mmq[slot], x.mmq_bytes[slot],
                                        out, static_cast<int>(w.rows), tokens,
                                        static_cast<int>(w.k), stream_);
      if (rc == 0)
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
  Mark(1);
  if (qkv_bank_ >= 0) {
    KdaConvStreams(npu_tab_[qkv_bank_], w.conv_q->f32(), w.conv_k->f32(),
                   w.conv_v->f32(), st.conv, b.kq, b.kk, b.kv, T, stream_);
  } else {
    KdaConv(b.kq, b.kk, b.kv, st.conv, w.conv_q->f32(), w.conv_k->f32(),
            w.conv_v->f32(), stream_, T);
  }
  Mark(10);
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
  Mark(11);
  Proj(*w.ssm_g_a, y, b.ga, T);
  Proj(*w.ssm_g_b, Prep(b.ga, T, kKdaHeadDim), b.g2, T);
  KdaOut(b.o, w.ssm_norm->f32(), b.g2, stream_, T);
  Proj(*w.attn_output, Prep(b.o, T, kKdaInner), b.block, T);
  Mark(12);
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
  const int n_experts = experts();
  qfn::ExpertCounts(b.ids, b.counts, T, n_experts, kMoeUsed, stream_);
  hipMemcpyAsync(b.host_counts, b.counts,
                 static_cast<std::size_t>(n_experts) * sizeof(std::uint32_t),
                 hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);
  std::int32_t* host_pair = b.host_tiles;
  std::int32_t* host_down = b.host_tiles + b.tile_capacity;
  std::uint32_t pair_tiles = 0;
  std::uint32_t down_tiles = 0;
  const auto add = [&](std::int32_t* list, std::uint32_t& n, std::uint32_t lo,
                       std::uint32_t hi, std::uint32_t rows) {
    for (int e = 0; e < n_experts; ++e) {
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
  // The routed F16 GEMMs cap row-tile width per weight type: the paired
  // gate/up kernel is only proven at BN 128 for K-quants, and the plain one
  // rejects BN > 48 for Q4_K/Q5_K and BN > 64 for Q8_0. Launching a wider
  // tile silently does nothing, so K-quants run a single narrow map.
  const bool gate_wide_ok = w.exp_gate->type == kTypeIQ2_XXS ||
                            w.exp_gate->type == kTypeIQ2_S ||
                            w.exp_gate->type == kTypeQ3_K;
  const bool down_wide_ok = w.exp_down->type != kTypeQ4_K &&
                            w.exp_down->type != kTypeQ5_K &&
                            w.exp_down->type != kTypeQ8_0;
  const std::uint32_t down_wide_rows =
      w.exp_down->type == kTypeQ4_K || w.exp_down->type == kTypeQ5_K
          ? 48
          : (w.exp_down->type == kTypeQ8_0
                 ? 64
                 : (w.exp_down->type == kTypeQ3_K ? 128 : kDownWideRows));
  if (gate_wide_ok) {
    add(host_pair, pair_tiles, kGateWideMin, kAll, kGateWideRows);
    add(host_pair, pair_tiles, 0, kGateWideMin, kGateNarrowRows);
  } else {
    add(host_pair, pair_tiles, 0, kAll, kGateNarrowRows);
  }
  const std::uint32_t gate_wide_tiles = pair_tiles;
  if (down_wide_ok) {
    add(host_down, down_tiles, kDownWideMin, kAll, down_wide_rows);
    add(host_down, down_tiles, 0, kDownWideMin, kDownNarrowRows);
  } else {
    add(host_down, down_tiles, 0, kAll, down_wide_rows);
  }
  const std::uint32_t down_wide_tiles = down_tiles;
  hipMemcpyAsync(b.tiles_pair, host_pair, pair_tiles * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  hipMemcpyAsync(b.tiles_down, host_down, down_tiles * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  qfn::RoutedCompact(b.ids, b.counts, b.bounds, b.cursors, b.rows_token,
                     b.rows_slot, T, kMoeUsed, n_experts, stream_);
  const auto weight_type = [](int code) {
    switch (code) {
      case kTypeQ3_K:
        return qfn::WeightType::kQ3_K;
      case kTypeQ4_K:
        return qfn::WeightType::kQ4_K;
      case kTypeQ5_K:
        return qfn::WeightType::kQ5_K;
      case kTypeQ8_0:
        return qfn::WeightType::kQ8_0;
      case kTypeIQ2_S:
        return qfn::WeightType::kIQ2_S;
      default:
        return qfn::WeightType::kIQ2_XXS;
    }
  };
  const auto gate_pair = [&](std::uint32_t offset, std::uint32_t count,
                             std::uint32_t rows) {
    if (count == 0)
      return;
    if (!qfn::RoutedGatedF16Gemm(
            w.exp_gate->data, w.exp_up->data, weight_type(w.exp_gate->type),
            static_cast<const __half*>(b.x_half), b.tiles_pair + offset, count,
            rows, b.bounds, b.rows_token, b.rows_slot,
            static_cast<__half*>(b.act_half), kExpertFf, kHidden, stream_, 1.0F,
            true)) {
      std::fprintf(
          stderr,
          "glm53: paired gate/up GEMM launch failed (type=%d rows=%u)\n",
          w.exp_gate->type, rows);
    }
  };
  gate_pair(0, gate_wide_tiles, gate_wide_ok ? kGateWideRows : kGateNarrowRows);
  gate_pair(gate_wide_tiles, pair_tiles - gate_wide_tiles, kGateNarrowRows);
  const auto down = [&](std::uint32_t offset, std::uint32_t count,
                        std::uint32_t rows) {
    if (count == 0)
      return;
    if (!qfn::RoutedF16Gemm(
            w.exp_down->data, weight_type(w.exp_down->type),
            static_cast<const __half*>(b.act_half), b.tiles_down + offset,
            count, rows, b.bounds, b.rows_slot, b.rows_slot, nullptr, nullptr,
            static_cast<__half*>(b.parts_half), kHidden, kExpertFf, stream_)) {
      std::fprintf(stderr, "glm53: down GEMM launch failed (type=%d rows=%u)\n",
                   w.exp_down->type, rows);
    }
  };
  down(0, down_wide_tiles, down_wide_rows);
  down(down_wide_tiles, down_tiles - down_wide_tiles, kDownNarrowRows);
  (void)slots;
}

// Expert FFN for layouts the routed F16 GEMM does not know (Q3_K): the decode
// matvec is limited to kMaxExpertTokens columns per launch, so prefill runs it
// in groups of that many tokens. Slower than the routed GEMM but numerically
// the same path decode already uses.
void Executor::RoutedExpertsMatvec(const LayerWeights& w, int T) const {
  const Scratch& b = *scratch_;
  const bool pair = w.exp_gate->type == w.exp_up->type;
  // Q3_K has no fused routed GEMM, so prefill runs the decode matvec. Keep
  // the chunk at one token: the mmvq n>1 path accumulates in a different order
  // and drifts (cosine 0.97 at n=8, 0.77 across a 64-token batch), which
  // would trade quality for speed. One-token chunks are bit-stable vs decode.
  constexpr int kChunk = 1;
  for (int t0 = 0; t0 < T; t0 += kChunk) {
    const int n = std::min<int>(kChunk, T - t0);
    const std::size_t x_off = static_cast<std::size_t>(t0) * kHidden;
    const std::size_t slot_off = static_cast<std::size_t>(t0) * kMoeUsed;
    const std::size_t act_off = slot_off * kExpertFf;
    const std::size_t part_off = slot_off * kHidden;
    if (pair) {
      MatvecExpertsPairTokens(w.exp_gate->type, w.exp_gate->data,
                              w.exp_up->data, b.y + x_off, b.gate + act_off,
                              b.up + act_off, kExpertFf, kHidden, experts(),
                              kMoeUsed, n, b.ids + slot_off, b.matvec, stream_);
    } else {
      MatvecExpertsTokens(w.exp_gate->type, w.exp_gate->data, b.y + x_off,
                          b.gate + act_off, kExpertFf, kHidden, experts(),
                          kMoeUsed, n, b.ids + slot_off, false, b.matvec,
                          stream_);
      MatvecExpertsTokens(w.exp_up->type, w.exp_up->data, b.y + x_off,
                          b.up + act_off, kExpertFf, kHidden, experts(),
                          kMoeUsed, n, b.ids + slot_off, false, b.matvec,
                          stream_);
    }
    SwigluClamp(b.gate + act_off, b.up + act_off, b.act + act_off,
                n * kMoeUsed * kExpertFf, stream_);
    MatvecExpertsTokens(w.exp_down->type, w.exp_down->data, b.act + act_off,
                        b.parts + part_off, kHidden, kExpertFf, experts(),
                        kMoeUsed, n, b.ids + slot_off, true, b.matvec, stream_);
  }
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

void Executor::NpuShDownQueue(std::uint32_t layer, int T) const {
  const Scratch& b = *scratch_;
  const int bank = shexp_bank_;
  // Encode activations half-precision: encode kernels read F16. Safe by
  // stream order: the routed GEMMs that read x_half were launched before
  // this overwrite, and the NPU copies x_half before any later stage runs.
  ToHalf(b.sact, b.x_half, static_cast<std::size_t>(T) * kExpertFf, stream_);
  qfn::NpuEncodeActivations(static_cast<const __half*>(b.x_half), kExpertFf,
                            static_cast<std::uint32_t>(T), kExpertFf, 0,
                            kExpertFf, npu_->Device(npu_sd_a_[bank]), stream_,
                            false);
  npu_->Submit(stream_, npu_sd_a_[bank], npu_sd_[layer].data(),
               npu_sd_c_[bank].data(), kHidden / npu_->n(), npu_sd_prog_);
  npu_sd_ticket_[bank] = npu_->Ticket();
}

void Executor::NpuShDownFinish(int T) const {
  const int bank = shexp_bank_;
  npu_->Wait(stream_, npu_sd_ticket_[bank]);
  NpuDecodeFused(npu_sd_tab_[bank], scratch_->shared, nullptr, nullptr, T,
                 kHidden / npu_->n(), kHidden / npu_->n(), kHidden, stream_);
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
               static_cast<int>(config_.expert_count), stream_);
  } else {
    Proj(*w.router, Act{b.y, nullptr}, b.rlogits, T);
  }
  Mark(15);
  MoeRoute(b.rlogits, w.router_bias->f32(), config_.expert_weights_scale, b.ids,
           b.wts, experts(), stream_, T);
  const bool shexp_npu = shexp_bank_ >= 0 && T > 1 &&
                         T == static_cast<int>(npu_->tokens()) &&
                         layer < npu_sh_.size() && !npu_sh_[layer].empty();
  // The expert-sorted F16 GEMM only knows the IQ2/Q4_K/Q5_K/IQ4/IQ3_S layouts.
  // A Q3_K expert stack (the 152-expert RCO export) has no routed GEMM yet, so
  // it runs the decode matvec in groups of kMaxExpertTokens instead.
  const bool routed_f16 = RoutedF16Supports(w.exp_gate->type) &&
                          RoutedF16Supports(w.exp_up->type) &&
                          RoutedF16DownSupports(w.exp_down->type);
  const bool use_matvec = T <= kMaxExpertTokens || !routed_f16;
  if (use_matvec) {
    RoutedExpertsMatvec(w, T);
  } else {
    if (!res_half_)
      ToHalf(b.y, b.x_half, static_cast<long long>(T) * kHidden, stream_);
    if (shexp_npu)
      NpuShexpQueue(layer);
    RoutedExpertsF16(w, T);
  }
  Mark(13);
  if (shexp_npu) {
    NpuShexpFinish();
  } else {
    Proj(*w.sh_gate, y, b.sg, T);
    Proj(*w.sh_up, y, b.su, T);
  }
  SwigluClamp(b.sg, b.su, b.sact, T * kExpertFf, stream_);
  const bool shdown_npu = shexp_npu && npu_whole_ && layer < npu_sd_.size() &&
                          !npu_sd_[layer].empty();
  if (shdown_npu)
    NpuShDownQueue(layer, T);
  else
    Proj(*w.sh_down, Prep(b.sact, T, kExpertFf), b.shared, T);
  if (shdown_npu)
    NpuShDownFinish(T);
  if (use_matvec) {
    MoeCombine(b.parts, b.wts, b.shared, b.ffn_out, stream_, T);
  } else if (!res_half_) {
    MoeCombineF16(b.parts_half, b.wts, b.shared, b.ffn_out, stream_, T);
  }
  Mark(14);
}

bool Executor::EmbedRows(const std::int32_t* tokens, int T, float* dst,
                         std::string* error) {
  return EmbedRowsFrom(token_embd_dev_, tokens, T, dst, error);
}

bool Executor::EmbedRowsFrom(const DeviceTensor& embd,
                             const std::int32_t* tokens, int T, float* dst,
                             std::string* error) {
  // A Q4_K table has no embedding-kernel decoder, so the kernel reads the
  // Q8_0 mirror built at load; `data` itself stays native for the drafter.
  const bool q4_embd = embd.type == kTypeQ4_K;
  const std::uint32_t embd_type =
      q4_embd ? kTypeQ8_0 : static_cast<std::uint32_t>(embd.type);
  const void* const embd_data = q4_embd ? embd.q8 : embd.data;
  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(
      static_cast<gufo::core::GgmlType>(embd_type), embd.k);
  for (int t = 0; t < T; ++t) {
    if (tokens[t] < 0 || static_cast<std::uint64_t>(tokens[t]) >= embd.rows) {
      if (error != nullptr)
        *error = "token out of range";
      return false;
    }
  }
  hipMemcpyAsync(scratch_->tok_ids, tokens,
                 static_cast<std::size_t>(T) * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  EmbedTokens(embd_data, embd_type, row_bytes, scratch_->tok_ids, dst,
              res_half_, T, stream_);
  hipStreamSynchronize(stream_);
  return true;
}

bool Executor::MtpPrimeChunk(Session& session, const std::int32_t* tok, int n,
                             std::size_t cap_base, std::string* error) {
  const bool carry = session.mtp_position > 0;
  const int rows = carry ? n : n - 1;
  const std::uint32_t pos0 =
      carry ? session.mtp_position : session.mtp_position + 1;
  if (rows > 0 && !MtpPrimeEntries(session, tok + (carry ? 0 : 1), rows, pos0,
                                   carry, cap_base, error)) {
    return false;
  }
  session.mtp_position = pos0 + static_cast<std::uint32_t>(rows);
  hipMemcpyAsync(session.mtp_carry,
                 scratch_->mtp_hidden + (cap_base + n - 1) * kHidden,
                 kHidden * sizeof(float), hipMemcpyDeviceToDevice, stream_);
  return true;
}

bool Executor::MtpPrimeEntries(Session& session, const std::int32_t* tokens,
                               int rows, std::uint32_t pos0, bool use_carry,
                               std::size_t cap_base, std::string* error) {
  if (rows <= 0)
    return true;
  const LayerWeights& w = weights_[mtp_.layer];
  Scratch& b = *scratch_;
  // The packed rows are F32, so embed without the wide-batch F16 streams.
  const bool half_streams = res_half_;
  res_half_ = false;
  if (!EmbedRowsFrom(mtp_.token_embd, tokens, rows, b.mtp_emb, error))
    return false;
  res_half_ = half_streams;
  MtpPack(b.mtp_emb, b.mtp_hidden + cap_base * kHidden, use_carry ? 0 : -1,
          use_carry ? session.mtp_carry : nullptr, w.nextn_enorm->f32(),
          w.nextn_hnorm->f32(), kRmsEps, b.mtp_concat, rows, stream_);
  arena_used_ = 0;
  const Act x = Prep(b.mtp_concat, rows, 2 * kHidden);
  Proj(*w.nextn_eh_proj, x, b.y, rows);
  RmsNorm(b.y, w.attn_norm->f32(), kHidden, kRmsEps, stream_, rows);
  arena_used_ = 0;
  const Act y = Prep(b.y, rows, kHidden);
  Proj(*w.kv_a, y, b.lat, rows);
  RmsNorm(b.lat, w.kv_a_norm->f32(), kDsaLatent, kRmsEps, stream_, rows);
  Proj(*w.idx_k, y, b.ik, rows);
  LayerNorm(b.ik, w.idx_k_norm_w->f32(), w.idx_k_norm_b->f32(), kDsaIndexDim,
            kKNormEps, stream_, rows);
  Proj(*w.idx_gate, y, b.gt, rows);
  LayerState& st = session.layers[mtp_.layer];
  if (DsaAppendBatch(st.cache, b.lat, b.ik, b.gt, w.idx_ape->f32(),
                     static_cast<int>(pos0), rows, stream_) != 0) {
    if (error != nullptr)
      *error = "mtp cache append failed";
    return false;
  }
  return hipGetLastError() == hipSuccess;
}

bool Executor::MtpDraftStep(Session& session, std::int32_t token, bool first,
                            std::string* error) {
  if (!mtp_.active) {
    if (error != nullptr)
      *error = "no MTP block loaded";
    return false;
  }
  const LayerWeights& w = weights_[mtp_.layer];
  Scratch& b = *scratch_;
  res_half_ = false;
  if (!EmbedRowsFrom(mtp_.token_embd, &token, 1, b.mtp_emb1, error))
    return false;
  MtpPack(b.mtp_emb1, nullptr, 0,
          first ? session.mtp_carry : session.mtp_mcarry, w.nextn_enorm->f32(),
          w.nextn_hnorm->f32(), kRmsEps, b.mtp_pack1, 1, stream_);
  arena_used_ = 0;
  const Act x = Prep(b.mtp_pack1, 1, 2 * kHidden);
  Proj(*w.nextn_eh_proj, x, b.y, 1);
  RmsNorm(b.y, w.attn_norm->f32(), kHidden, kRmsEps, stream_, 1);
  const int pos = static_cast<int>(session.mtp_position);
  hipMemcpy(b.pos_dev, &pos, sizeof(int), hipMemcpyHostToDevice);
  DsaBlock(mtp_.layer, session, 1);
  MtpVecAdd(b.y, b.block, b.y, kHidden, 1, stream_);
  hipMemcpyAsync(b.mtp_tmp, b.y, kHidden * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_);
  RmsNorm(b.y, w.ffn_norm->f32(), kHidden, kRmsEps, stream_, 1);
  FfnBlock(mtp_.layer, 1);
  MtpVecAdd(b.mtp_tmp, b.ffn_out, b.mtp_tmp, kHidden, 1, stream_);
  hipMemcpyAsync(session.mtp_mcarry, b.mtp_tmp, kHidden * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_);
  hipMemcpyAsync(b.head_in, b.mtp_tmp, kHidden * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_);
  RmsNorm(b.head_in, w.nextn_shn->f32(), kHidden, kRmsEps, stream_, 1);
  Proj(mtp_.output, Act{b.head_in, nullptr}, b.logits, 1);
  last_logit_rows_ = 1;
  ++session.mtp_position;
  if (hipGetLastError() != hipSuccess) {
    if (error != nullptr)
      *error = "MTP draft step failed";
    return false;
  }
  hipStreamSynchronize(stream_);
  return true;
}

bool Executor::MtpRefine(Session& session, std::span<const std::int32_t> chain,
                         std::uint32_t keep, std::string* error) {
  if (!mtp_.active || keep == 0)
    return true;
  const std::uint32_t base = session.position - keep;
  if (chain.size() > 1 && keep > 1) {
    if (!MtpPrimeEntries(session, chain.data() + 1, static_cast<int>(keep) - 1,
                         base + 1, false, 0, error)) {
      return false;
    }
  }
  session.mtp_position = base + keep;
  hipMemcpyAsync(
      session.mtp_carry,
      scratch_->mtp_hidden + static_cast<std::size_t>(keep - 1) * kHidden,
      kHidden * sizeof(float), hipMemcpyDeviceToDevice, stream_);
  return hipGetLastError() == hipSuccess;
}

bool Executor::MtpAdvance(Session& session, std::int32_t token,
                          std::string* error) {
  if (!mtp_.active)
    return true;
  if (!MtpPrimeEntries(session, &token, 1, session.mtp_position, true, 0,
                       error)) {
    return false;
  }
  ++session.mtp_position;
  const int rows = mtp_capture_rows_ > 0 ? mtp_capture_rows_ : 1;
  hipMemcpyAsync(
      session.mtp_carry,
      scratch_->mtp_hidden + static_cast<std::size_t>(rows - 1) * kHidden,
      kHidden * sizeof(float), hipMemcpyDeviceToDevice, stream_);
  return hipGetLastError() == hipSuccess;
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
    if (res_half_ && !w.dense && ffn_rows > 1 &&
        (RoutedF16Supports(w.exp_gate->type) &&
         RoutedF16Supports(w.exp_up->type) &&
         RoutedF16DownSupports(w.exp_down->type))) {
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
  if (mtp_.active && options_.head) {
    // MTP feed: llama.cpp's glm-dsa trunk exports the post-`output_norm`
    // hidden as `h_nextn` (which the draft block then normalizes with
    // `hnorm`), so capture RMS(output_norm) hidden for every row here.
    const DeviceTensor& norm = *Find("output_norm.weight");
    if (res_half_) {
      HeadNorm(reinterpret_cast<const std::uint16_t*>(residual), norm.f32(),
               scratch_->mtp_hidden + mtp_capture_row0_ * kHidden, stream_, T);
    } else {
      HeadNorm(residual, norm.f32(),
               scratch_->mtp_hidden + mtp_capture_row0_ * kHidden, stream_, T);
    }
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
  mtp_capture_row0_ = 0;
  mtp_capture_rows_ = T;
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
      "hc",         "kda:qkv",    "dsa",        "ffn-dense",
      "ffn-moe",    "dsa:proj",   "dsa:select", "dsa:headgemm",
      "dsa:attend", "dsa:scores", "kda:conv",   "kda:recur",
      "kda:wo",     "moe:routed", "moe:shexp",  "moe:router"};
  double total = 0.0;
  for (const double v : profile_seconds_)
    total += v;
  for (int i = 0; i < 16; ++i) {
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
  mtp_capture_row0_ = static_cast<std::size_t>(T);
  FinishHead(b.x[cur[1]], T, logit_rows);
  if (mtp_.active) {
    // Capture the first bank's hidden as well: its rows are the first half
    // of the pair's capture.
    SwapBank();
    {
      const DeviceTensor& norm = *Find("output_norm.weight");
      if (res_half_) {
        HeadNorm(reinterpret_cast<const std::uint16_t*>(b.x[cur[0]]),
                 norm.f32(), scratch_->mtp_hidden, stream_, T);
      } else {
        HeadNorm(b.x[cur[0]], norm.f32(), scratch_->mtp_hidden, stream_, T);
      }
    }
    SwapBank();
    mtp_capture_row0_ = 0;
    mtp_capture_rows_ = 2 * T;
  }
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
  // Two-bank pair: process 2 * NPU-bank-rows per call while at least a full
  // pair remains, so every prompt of 2 * npu rows or more runs paired
  // (2048-row banks cover 4096-token prompts and their tails at depth).
  const std::size_t pair_rows =
      npu_ != nullptr && npu_->Healthy() && chunk == options_.prefill_chunk
          ? std::min<std::size_t>(npu_->tokens(), options_.prefill_chunk)
          : 0;
  for (std::size_t at = 0; at < tokens.size();) {
    if (pair_rows != 0 && tokens.size() - at >= 2 * pair_rows) {
      const bool last = at + 2 * pair_rows == tokens.size();
      if (!ForwardPair(session, tokens.data() + at,
                       static_cast<int>(2 * pair_rows), last ? 1 : 0, error) ||
          (after_forward && !after_forward(error))) {
        return false;
      }
      if (mtp_.active &&
          !(MtpPrimeChunk(session, tokens.data() + at,
                          static_cast<int>(pair_rows), 0, error) &&
            MtpPrimeChunk(session, tokens.data() + at + pair_rows,
                          static_cast<int>(pair_rows),
                          static_cast<int>(pair_rows), error))) {
        return false;
      }
      at += 2 * pair_rows;
      continue;
    }
    const std::size_t n = std::min(chunk, tokens.size() - at);
    const bool last = at + n == tokens.size();
    if (!Forward(session, tokens.data() + at, static_cast<int>(n), last ? 1 : 0,
                 error) ||
        (after_forward && !after_forward(error))) {
      return false;
    }
    if (mtp_.active && !MtpPrimeChunk(session, tokens.data() + at,
                                      static_cast<int>(n), 0, error)) {
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

namespace {
// Bounded per-row penalty workspace of the compact verification path.
constexpr std::size_t kCompactPenaltyLimit = 4096;
}  // namespace

bool Executor::CanCompactVerification(const sampling::SamplerState& sampler) {
  const auto& config = sampler.config();
  return config.uses_random_sampling() && config.top_k > 0 &&
         std::size_t(config.top_k) <= qfnk::kMtpCandidates &&
         config.min_keep <= qfnk::kMtpCandidates &&
         sampler.penalties().size() + 8 <= kCompactPenaltyLimit &&
         !sampler.NeedsConstraintMask();
}

bool Executor::SampledCandidates(
    std::vector<std::vector<sampling::LogitCandidate>>* out,
    const sampling::SamplerState& sampler, std::span<const std::int32_t> drafts,
    std::string* error) {
  const std::size_t rows = drafts.size();
  if (rows == 0 || !CanCompactVerification(sampler) ||
      sampler.penalties().size() + rows + 8 > kCompactPenaltyLimit) {
    if (error != nullptr)
      *error = "sampled verification is not compact-eligible";
    return false;
  }
  const auto vocab = static_cast<std::uint32_t>(config_.vocab_size);
  if (verification_ids_ == nullptr) {
    const auto ids = 7 * qfn::MtpCandidateWorkspaceSize(vocab);
    if (!Alloc(&verification_ids_, ids, error) ||
        !Alloc(&verification_scratch_ids_, ids, error) ||
        !Alloc(&verification_candidates_,
               7 * (qfnk::kMtpCandidates + kCompactPenaltyLimit), error) ||
        hipHostMalloc(reinterpret_cast<void**>(&verification_candidates_host_),
                      7 * (qfnk::kMtpCandidates + kCompactPenaltyLimit) *
                          sizeof(sampling::LogitCandidate),
                      hipHostMallocDefault) != hipSuccess) {
      if (error != nullptr && error->empty())
        *error = "verification candidate allocation failed";
      return false;
    }
    verification_penalty_host_.reserve(7 * kCompactPenaltyLimit);
  }
  verification_rows_.resize(std::max(verification_rows_.size(), rows));
  auto tentative = sampler.WithoutConstraint();
  for (std::size_t first = 0; first < rows; first += 7) {
    const auto count =
        static_cast<std::uint32_t>(std::min<std::size_t>(7, rows - first));
    auto& penalties = verification_penalty_host_;
    penalties.clear();
    qfn::GreedyPenaltyRows batch{};
    for (std::uint32_t row = 0; row < count; ++row) {
      const auto counts = tentative.penalties();
      penalties.insert(penalties.end(), counts.begin(), counts.end());
      batch.offsets[row + 1] = static_cast<std::uint32_t>(penalties.size());
      if (row + 1 < count)
        tentative.Accept(static_cast<sampling::TokenId>(drafts[first + row]));
    }
    if (!penalties.empty()) {
      if (penalties.size() > verification_penalty_capacity_) {
        const auto capacity =
            std::max(penalties.size(), 2 * verification_penalty_capacity_);
        sampling::TokenPenalty* allocated = nullptr;
        if (hipMalloc(&allocated, capacity * sizeof(*allocated)) !=
            hipSuccess) {
          if (error != nullptr)
            *error = "verification penalty allocation failed";
          return false;
        }
        hipFree(verification_penalties_);
        verification_penalties_ = allocated;
        verification_penalty_capacity_ = capacity;
      }
      batch.penalties = verification_penalties_;
      if (hipMemcpyAsync(verification_penalties_, penalties.data(),
                         penalties.size() * sizeof(penalties.front()),
                         hipMemcpyHostToDevice, stream_) != hipSuccess) {
        if (error != nullptr)
          *error = "verification penalty upload failed";
        return false;
      }
    }
    qfn::VerificationCandidates(scratch_->logits + first * vocab, batch,
                                verification_ids_, verification_scratch_ids_,
                                verification_candidates_, count, vocab,
                                stream_);
    const auto total = count * qfnk::kMtpCandidates +
                       static_cast<std::size_t>(penalties.size());
    if (hipMemcpyAsync(verification_candidates_host_, verification_candidates_,
                       total * sizeof(sampling::LogitCandidate),
                       hipMemcpyDeviceToHost, stream_) != hipSuccess ||
        hipStreamSynchronize(stream_) != hipSuccess) {
      if (error != nullptr)
        *error = "verification candidate download failed";
      return false;
    }
    tentative.Accept(static_cast<sampling::TokenId>(drafts[first + count - 1]));
    for (std::uint32_t row = 0; row < count; ++row) {
      const auto begin = row * qfnk::kMtpCandidates + batch.offsets[row];
      auto& kept = verification_rows_[first + row];
      kept.assign(verification_candidates_host_ + begin,
                  verification_candidates_host_ + begin + qfnk::kMtpCandidates +
                      batch.offsets[row + 1] - batch.offsets[row]);
    }
  }
  out->assign(verification_rows_.begin(),
              verification_rows_.begin() + static_cast<std::ptrdiff_t>(rows));
  return true;
}

bool Executor::SaveState(Session& session, std::string* error) {
  if (session.saved_conv.empty()) {
    session.saved_conv.assign(layers_, nullptr);
    session.saved_state.assign(layers_, nullptr);
    constexpr std::size_t kConvFloats = 3 * 3 * kKdaInner;
    constexpr std::size_t kStateFloats =
        static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim * kKdaHeadDim;
    std::size_t kda_layers = 0;
    for (std::uint32_t l = 0; l < layers_; ++l)
      kda_layers += weights_[l].kda ? 1 : 0;
    float* slab = nullptr;
    if (!Alloc(&slab, kda_layers * (kConvFloats + kStateFloats), error))
      return false;
    session.saved_slab = slab;
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!weights_[l].kda)
        continue;
      session.saved_conv[l] = slab;
      session.saved_state[l] = slab + kConvFloats;
      slab += kConvFloats + kStateFloats;
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
  session.mtp_position = 0;
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
  std::uint64_t total = 0;
  for (std::uint32_t l = 0; l < session.layers.size(); ++l) {
    if (weights_[l].kda) {
      total += kKdaConvBytes + kKdaStateBytes;
    } else {
      const std::uint64_t rows = mtp_rows(l, session);
      total += rows * kLatRowBytes + (rows / kDsaPool) * kIndexRowBytes +
               2 * (rows % kDsaPool) * kIndexRowBytes;
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
  for (std::uint32_t l = 0; l < session.layers.size(); ++l) {
    const LayerState& st = session.layers[l];
    if (weights_[l].kda) {
      if (!copy(st.conv, kKdaConvBytes) || !copy(st.state, kKdaStateBytes))
        return false;
      continue;
    }
    const std::size_t rows = static_cast<std::size_t>(mtp_rows(l, session));
    const std::size_t tail = rows % kDsaPool;
    if (!copy(st.cache.lat_half, rows * kLatRowBytes) ||
        !copy(st.cache.pooled, (rows / kDsaPool) * kIndexRowBytes))
      return false;
    for (std::size_t i = 0; i < tail; ++i) {
      const std::size_t slot = (rows - tail + i) % st.cache.ring;
      if (!copy(st.cache.ik + slot * kDsaIndexDim, kIndexRowBytes))
        return false;
    }
    for (std::size_t i = 0; i < tail; ++i) {
      const std::size_t slot = (rows - tail + i) % st.cache.ring;
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
  for (std::uint32_t l = 0; ok && l < session.layers.size(); ++l) {
    LayerState& st = session.layers[l];
    if (weights_[l].kda) {
      ok = copy(st.conv, kKdaConvBytes) && copy(st.state, kKdaStateBytes);
      continue;
    }
    const std::size_t rows = static_cast<std::size_t>(mtp_rows(l, session));
    const std::size_t tail = rows % kDsaPool;
    ok = copy(st.cache.lat_half, rows * kLatRowBytes) &&
         copy(st.cache.pooled, (rows / kDsaPool) * kIndexRowBytes);
    for (std::size_t i = 0; ok && i < tail; ++i) {
      const std::size_t slot = (rows - tail + i) % st.cache.ring;
      ok = copy(st.cache.ik + slot * kDsaIndexDim, kIndexRowBytes);
    }
    for (std::size_t i = 0; ok && i < tail; ++i) {
      const std::size_t slot = (rows - tail + i) % st.cache.ring;
      ok = copy(st.cache.gate + slot * kDsaIndexDim, kIndexRowBytes);
    }
  }
  if (!ok) {
    session.position = 0;
    return false;
  }
  session.mtp_position = position;
  session.saved_position = 0;
  return true;
}

}  // namespace gufo::models::glm53_flash::rocm
