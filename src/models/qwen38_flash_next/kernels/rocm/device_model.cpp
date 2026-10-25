#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

/// Bytes one row of `type` occupies for a row length of `ne0`. Used to check
/// that every device tensor's allocation matches the shape and type the loader
/// believes it uploaded: a tensor whose bytes were reinterpreted under another
/// type's layout still loads and still runs, and returns wrong numbers.
std::size_t RowBytes(core::GgmlType type, std::size_t ne0) {
  switch (type) {
    case core::GgmlType::kF32:
      return ne0 * 4;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      return ne0 * 2;
    case core::GgmlType::kQ8_0:
      return ne0 / 32 * 34;
    case core::GgmlType::kQ5_1:
      return ne0 / 32 * 24;
    case core::GgmlType::kIQ4_NL:
      return ne0 / 32 * 18;
    case core::GgmlType::kIQ3_S:
      return ne0 / 256 * 110;
    case core::GgmlType::kIQ4_XS:
      return ne0 / 256 * 136;
    case core::GgmlType::kQ4_K:
      return ne0 / 256 * 144;
    case core::GgmlType::kQ5_K:
      return ne0 / 256 * 176;
    case core::GgmlType::kQ6_K:
      return ne0 / 256 * 210;
    default:
      return 0;
  }
}

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

bool WholeNpuEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("GUFO_NPU_WHOLE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::size_t& max_dequant_elems;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  bool expand_inject{false};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Dense Q4_K/Q5_K/Q5_1 matrices run wide batches through an F16 copy, which
  /// needs a weight scratch and F16 activation rows of their width.
  void NoteDequant(const DeviceTensor& d) {
    if (d.experts > 1 ||
        (d.type != core::GgmlType::kQ4_K && d.type != core::GgmlType::kQ5_K &&
         d.type != core::GgmlType::kQ5_1)) {
      return;
    }
    max_dequant_elems = std::max<std::size_t>(
        max_dequant_elems, static_cast<std::size_t>(d.rows) * d.cols);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
  }

  /// Abort on a tensor whose allocation disagrees with its declared type and
  /// shape. Permanent: the failure this catches is silent, so it must be loud.
  void CheckExtent(const DeviceTensor& d, std::string_view name,
                   std::size_t bytes, core::GgmlType gguf_type) {
    if (d.data == nullptr) {
      return;
    }
    const std::size_t row = RowBytes(d.type, d.cols);
    const std::size_t want =
        row == 0 ? 0
                 : row * static_cast<std::size_t>(d.rows) *
                       static_cast<std::size_t>(
                           std::max<std::uint32_t>(d.experts, 1));
    if (row == 0 || bytes < want) {
      std::fprintf(stderr,
                   "device tensor extent mismatch: %s\n"
                   "  gguf type : %s\n  dev type  : %s\n"
                   "  rows x cols x experts : %u x %u x %u\n"
                   "  bytes     : %zu   expected : %zu\n",
                   std::string(name).c_str(), core::ToString(gguf_type).data(),
                   core::ToString(d.type).data(), d.rows, d.cols, d.experts,
                   bytes, want);
      std::fflush(stderr);
      std::abort();
    }
    if (d.type != gguf_type) {
      std::fprintf(stderr,
                   "device type differs from GGUF type: %s  %s -> %s  "
                   "(%zu bytes, %u x %u)\n",
                   std::string(name).c_str(), core::ToString(gguf_type).data(),
                   core::ToString(d.type).data(), bytes, d.rows, d.cols);
      std::fflush(stderr);
      std::abort();
    }
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    RecordAllocation(t.name, ptr, size + kTailMargin);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)GuardedMemsetAsync(GUFO_SITE, static_cast<std::uint8_t*>(ptr) + size,
                             0, kTailMargin, nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    CheckExtent(d, t.name, size, t.type);
    NoteDequant(d);
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      RecordAllocation("mtp-split", dst.data, part_bytes + kTailMargin);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (GuardedMemcpy2D(GUFO_SITE, dst.data, row_bytes, src, 2 * row_bytes,
                          row_bytes, t.rows,
                          hipMemcpyDeviceToDevice) != hipSuccess ||
          GuardedMemset(GUFO_SITE,
                        static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                        kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    ForgetAllocation(combined.data, GUFO_SITE);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  /// Tensors sharing one input are stacked into a single GEMV. The K-quant
  /// types are included so a requantized trunk keeps the fused projection
  /// instead of silently splitting into one GEMV per part.
  static bool StackableType(core::GgmlType type) {
    switch (type) {
      case core::GgmlType::kF32:
      case core::GgmlType::kQ8_0:
      case core::GgmlType::kQ4_K:
      case core::GgmlType::kQ5_K:
      case core::GgmlType::kQ6_K:
        return true;
      default:
        return false;
    }
  }

  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          !StackableType(type)) {
        Fail("stacked upload needs F32, Q8_0 or a K-quant tensor of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0) {
      (void)GuardedMemsetAsync(GUFO_SITE,
                               static_cast<std::uint8_t*>(ptr) + size, 0,
                               kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    // Only F32 is narrowed. A K-quant part is already a compact byte stream, so
    // dividing its size by sizeof(float) and relabelling the result F16 would
    // reinterpret its bytes as float and corrupt every row. Quantized parts
    // are concatenated verbatim and keep their own type.
    if (type != core::GgmlType::kF32) {
      void* packed = nullptr;
      if (hipMalloc(&packed, size + kTailMargin) != hipSuccess) {
        Fail("hipMalloc failed for packed tensor");
        return d;
      }
      allocations.push_back(packed);
      RecordAllocation("stacked", packed, size + kTailMargin);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                         static_cast<std::uint8_t*>(packed) + offset, error)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
      d.data = packed;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      NoteDequant(d);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    RecordAllocation("stacked-f16", half,
                     count * sizeof(std::uint16_t) + kTailMargin);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)GuardedMemsetAsync(GUFO_SITE,
                             static_cast<std::uint8_t*>(half) + count * 2, 0,
                             kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up),
            expand_inject && m.inject.type == core::GgmlType::kQ8_0
                ? ExpandQ8ToF32(m.inject)
                : Copy(m.inject)};
  }

  /// A Q8_0 trunk inject is expanded to F32 (exact: an F16 scale times an
  /// int8 code fits a float), so the mixer keeps its fused inject epilogue
  /// instead of a second norm and a quantized projection per block.
  DeviceTensor ExpandQ8ToF32(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t count = static_cast<std::size_t>(t.ElementCount());
    std::vector<float> values(count);
    gufo::quant::DequantizeQ8_0(t.data, values.data(), count);
    const std::size_t size = count * sizeof(float);
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name));
      return d;
    }
    allocations.push_back(ptr);
    RecordAllocation(t.name, ptr, size + kTailMargin);
    bytes += size + kTailMargin;
    if (GuardedMemcpy(GUFO_SITE, ptr, values.data(), size,
                      hipMemcpyHostToDevice) != hipSuccess ||
        GuardedMemset(GUFO_SITE, static_cast<std::uint8_t*>(ptr) + size, 0,
                      kTailMargin) != hipSuccess) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = ptr;
    d.type = core::GgmlType::kF32;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    CheckExtent(d, t.name, size, core::GgmlType::kF32);
    return d;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || !StackableType(t->type) ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }
};

}  // namespace

int DeviceModel::NpuProgramFor(std::uint32_t rows,
                               std::uint32_t* run_tokens) const noexcept {
  std::uint32_t run = 4096;
  if (rows < 4096) {
    if (!npu_m_runtime_) {
      *run_tokens = 0;
      return -1;
    }
    run = std::clamp<std::uint32_t>((rows + 511) / 512 * 512, 1024, 4096);
  }
  *run_tokens = run;
  return npu_m_program_[run / 512];
}

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    ForgetAllocation(p, GUFO_SITE);
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // The output head now runs its Q6_K rows through the small GEMV, but the
  // embedding lookup and the routed expert kernels still cannot read it.
  // Reject those before allocating device weights.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
#ifdef GUFO_NPU
  // Wake the NPU before any heavy GPU work starts: resuming it re-initializes
  // the SMU and PSP, which the GPU shares. A heartbeat then keeps it awake.
  std::unique_ptr<NpuGemm> early_npu;
  if (const char* dir = std::getenv("GUFO_NPU_DIR");
      dir != nullptr && dir[0] != '\0') {
    NpuGemm::Config cfg;
    // The runtime-M program serves any group size with one XCLBIN; it needs
    // an instruction file per size, so it is used only when all are present.
    const std::string mk = std::string(dir) + "/runtime_mk/";
    const auto mk_insts = [&](std::uint32_t rows) {
      return mk + "insts_" + std::to_string(rows) + "x2560x1024_128x64x128.txt";
    };
    bool runtime_m =
        std::getenv("GUFO_NPU_FIXED_M") == nullptr &&
        std::filesystem::exists(mk + "final_4096xKx1024_128x64x128.xclbin");
    for (std::uint32_t rows = 1024; rows <= 4096 && runtime_m; rows += 512) {
      runtime_m = std::filesystem::exists(mk_insts(rows));
    }
    if (runtime_m) {
      cfg.xclbin = mk + "final_4096xKx1024_128x64x128.xclbin";
      cfg.insts = mk_insts(4096);
    } else {
      cfg.xclbin = std::string(dir) + "/final_4096x2560x1024_128x64x128.xclbin";
      cfg.insts = std::string(dir) + "/insts_4096x2560x1024_128x64x128.txt";
    }
    std::string npu_error;
    early_npu = NpuGemm::Create(cfg, &npu_error);
    if (early_npu == nullptr) {
      std::fprintf(stderr, "NPU off: %s\n", npu_error.c_str());
    } else if (runtime_m) {
      bool ok = true;
      for (std::uint32_t rows = 1024; rows < 4096 && ok; rows += 512) {
        const int id = early_npu->AddProgram(mk_insts(rows), &npu_error);
        ok = id >= 1;
        m->npu_m_program_[rows / 512] = id;
      }
      m->npu_m_runtime_ = ok;
      if (!ok) {
        std::fprintf(stderr, "NPU runtime-M programs unavailable: %s\n",
                     npu_error.c_str());
      }
    }
  }
#endif
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{
      *stager,           conversions,     m->allocations_,       m->bytes_,
      m->max_half_cols_, m->max_q8_cols_, m->max_dequant_elems_, error_msg};
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_ =
      w.output.data == w.token_embd.data ? m->token_embd_ : up.Copy(w.output);
  m->hc_head_ = up.Mixer(w.hc_head);
  m->layers_.reserve(w.layers.size());
  up.expand_inject = true;
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    // The sidecar has its own shard index; reuse the target's readers and
    // staging pool.
    up.shard_base = shard_count;
    up.expand_inject = false;
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, c.count, "device_model.cpp:378", "upload", "f32",
                      c.count, 1, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    ForgetAllocation(c.source, GUFO_SITE);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  // Decode reads each large dense matrix once per round, so its bytes set the
  // round's cost: a 4-bit (Q4_1) copy cuts them by 41% and a 5-bit copy by 29%.
  // The copies re-quantize 8-bit weights and change the output.
  // GUFO_DECODE_COPY picks the trade: "balanced" (default) copies the
  // projections that tolerate 4 bits, "q4" and "q5" copy every large matrix,
  // "off" decodes from the exact Q8_0 weights.
  const char* copy_env = std::getenv("GUFO_DECODE_COPY");
  const std::string copy_mode = copy_env != nullptr ? copy_env : "balanced";
  const bool balanced = copy_mode == "balanced";
  const bool q4_on = balanced || copy_mode == "q4";
  const bool q5_on = q4_on || copy_mode == "q5";
  if (q4_on || q5_on) {
    const std::size_t blk_bytes = q4_on ? 20 : 24;
    const auto dec_type = q4_on ? core::GgmlType::kQ4_1 : core::GgmlType::kQ5_1;
    constexpr std::size_t kMinElements = std::size_t{2560} * 6144;
    std::size_t copied = 0;
    const auto add = [&](DeviceTensor& t, std::size_t min_elements) {
      if (t.type != core::GgmlType::kQ8_0 || t.data == nullptr ||
          t.cols % 32 != 0 || std::size_t{t.rows} * t.cols < min_elements) {
        return;
      }
      const std::size_t blocks = std::size_t{t.rows} * t.cols / 32;
      void* copy = nullptr;
      if (hipMalloc(&copy, blocks * blk_bytes + kTailMargin) != hipSuccess) {
        return;  // the tensor keeps its Q8_0 decode path
      }
      m->allocations_.push_back(copy);
      RecordAllocation(q4_on ? "decode_q4_1" : "decode_q5_1", copy,
                       blocks * blk_bytes + kTailMargin);
      m->bytes_ += blocks * blk_bytes + kTailMargin;
      copied += blocks * blk_bytes;
      const bool done = q4_on
                            ? RequantQ8_0ToQ4_1(t.data, copy, blocks, nullptr)
                            : RequantQ8_0ToQ5_1(t.data, copy, blocks, nullptr);
      if (done) {
        t.decode_data = copy;
        t.decode_type = dec_type;
      }
    };
    // The four 320 x 10240 hyper-connection projections of every layer are
    // 14% of the bytes decode reads.
    constexpr std::size_t kHcElements = std::size_t{320} * 10240;
    // GUFO_DECODE_COPY_CLASSES narrows the copies to a comma-separated subset
    // of ssm_in, ssm_out, attn_qkv, attn_out, hc and shexp.
    const char* classes = std::getenv("GUFO_DECODE_COPY_CLASSES");
    if (classes == nullptr && balanced) {
      classes = "ssm_in,ssm_out,attn_qkv,shexp";
    }
    const auto wants = [&](const char* name) {
      return classes == nullptr || std::strstr(classes, name) != nullptr;
    };
    for (auto& l : m->layers_) {
      if (wants("ssm_in"))
        add(l.ssm_in, kMinElements);
      if (wants("ssm_out"))
        add(l.ssm_out, kMinElements);
      if (wants("attn_qkv"))
        add(l.attn_qkv, kMinElements);
      if (wants("attn_out"))
        add(l.attn_out, kMinElements);
      if (q4_on) {
        if (wants("hc")) {
          for (DeviceMixer* mixer : {&l.hc_attn, &l.hc_ffn}) {
            add(mixer->down, kHcElements);
            add(mixer->up, kHcElements);
          }
        }
        if (wants("shexp")) {
          add(l.shexp_up, 1);
          add(l.shexp_gate, 1);
          add(l.shexp_down, 1);
        }
      }
    }
    if (hipDeviceSynchronize() != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg = "decode weight conversion failed";
      }
      return nullptr;
    }
    std::fprintf(stderr, "decode %s copies: %.2f GiB\n",
                 q4_on ? "Q4_1" : "Q5_1",
                 static_cast<double>(copied) / (1024.0 * 1024.0 * 1024.0));
  }
#ifdef GUFO_NPU
  // GUFO_NPU_DIR names a directory holding the NPU kernel pair
  // (final_4096x2560x1024_128x64x128.xclbin and the matching insts file).
  if (early_npu != nullptr) {
    std::string npu_error;
    auto npu = std::move(early_npu);
    {
      constexpr std::uint32_t kRows = 16384;
      // GUFO_NPU_SLICES sets how many 1024-row slices the NPU computes (the
      // last rows of ssm_in); the GPU does the rest.
      std::uint32_t kSlices = 6;
      if (const char* sl = std::getenv("GUFO_NPU_SLICES"); sl != nullptr) {
        kSlices =
            std::clamp(static_cast<std::uint32_t>(std::atoi(sl)), 1U, 12U);
      }
      const std::uint32_t kZRow0 = kRows - kSlices * npu->n();
      std::size_t staged = 0;
      std::uint32_t layers_on = 0;
      // GUFO_NPU_LAYERS limits how many linear-attention layers use the NPU.
      std::uint32_t layer_limit = 0xFFFFFFFFu;
      if (const char* limit = std::getenv("GUFO_NPU_LAYERS");
          limit != nullptr) {
        layer_limit = static_cast<std::uint32_t>(std::atoi(limit));
      }
      for (auto& l : m->layers_) {
        if (layers_on >= layer_limit) {
          break;
        }
        if (l.ssm_in.data == nullptr ||
            l.ssm_in.type != core::GgmlType::kQ8_0 ||
            l.ssm_in.rows != kZRow0 + kSlices * npu->n() ||
            l.ssm_in.cols != npu->k()) {
          continue;
        }
        std::vector<int> ids;
        for (std::uint32_t i = 0; i < kSlices; ++i) {
          const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
          if (id < 0) {
            break;
          }
          NpuEncodeWeights(l.ssm_in.data, l.ssm_in.cols, kZRow0 + i * npu->n(),
                           npu->n(), npu->k(), 0, npu->k(), npu->Device(id),
                           nullptr);
          ids.push_back(id);
          staged += npu->BBytes();
        }
        if (hipDeviceSynchronize() != hipSuccess) {
          ids.clear();
        }
        for (int id : ids) {
          npu->FinalizeWeights(id);
        }
        if (ids.size() == kSlices) {
          l.npu_z = std::move(ids);
          ++layers_on;
        }
      }
      // GUFO_NPU_GATE_SLICES sets how many 1024-row slices of attn_qkv (two
      // heads' query and gate rows each) the NPU computes in the attention
      // layers.
      std::uint32_t gate_slices = 12;
      if (const char* gs = std::getenv("GUFO_NPU_GATE_SLICES"); gs != nullptr) {
        gate_slices = std::min(static_cast<std::uint32_t>(std::atoi(gs)), 12U);
      }
      std::uint32_t gate_layers = 0;
      for (auto& l : m->layers_) {
        if (gate_slices == 0 || l.attn_qkv.empty() ||
            l.attn_qkv.data == nullptr ||
            l.attn_qkv.type != core::GgmlType::kQ8_0 ||
            l.attn_qkv.rows != 13312 || l.attn_qkv.cols != npu->k()) {
          continue;
        }
        std::vector<int> ids;
        for (std::uint32_t i = 0; i < gate_slices; ++i) {
          const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
          if (id < 0) {
            break;
          }
          NpuEncodeWeights(l.attn_qkv.data, l.attn_qkv.cols, i * npu->n(),
                           npu->n(), npu->k(), 0, npu->k(), npu->Device(id),
                           nullptr);
          ids.push_back(id);
          staged += npu->BBytes();
        }
        if (hipDeviceSynchronize() != hipSuccess) {
          ids.clear();
        }
        for (int id : ids) {
          npu->FinalizeWeights(id);
        }
        if (ids.size() == gate_slices) {
          l.npu_gate = std::move(ids);
          ++gate_layers;
        }
      }
      if (gate_layers != 0) {
        std::fprintf(stderr, "NPU: attention query/gate of %u layers\n",
                     gate_layers);
      }
      // GUFO_NPU_SHEXP=0 keeps the shared expert gate and up projections on
      // the GPU.
      const char* shexp_env = std::getenv("GUFO_NPU_SHEXP");
      std::uint32_t shexp_layers = 0;
      for (auto& l : m->layers_) {
        if ((shexp_env != nullptr && shexp_env[0] == '0') ||
            l.shexp_gate.data == nullptr || l.shexp_up.data == nullptr ||
            l.shexp_gate.type != core::GgmlType::kQ8_0 ||
            l.shexp_up.type != core::GgmlType::kQ8_0 ||
            l.shexp_gate.cols != npu->k() || l.shexp_up.cols != npu->k() ||
            l.shexp_gate.rows != l.shexp_up.rows ||
            l.shexp_gate.rows > npu->n()) {
          continue;
        }
        std::vector<int> ids;
        for (const DeviceTensor* w : {&l.shexp_gate, &l.shexp_up}) {
          const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
          if (id < 0) {
            break;
          }
          NpuEncodeWeights(w->data, w->cols, 0, w->rows, npu->k(), 0, npu->k(),
                           npu->Device(id), nullptr);
          ids.push_back(id);
          staged += npu->BBytes();
        }
        if (hipDeviceSynchronize() != hipSuccess) {
          ids.clear();
        }
        for (int id : ids) {
          npu->FinalizeWeights(id);
        }
        if (ids.size() == 2) {
          l.npu_shexp = std::move(ids);
          ++shexp_layers;
        }
      }
      if (shexp_layers != 0) {
        std::fprintf(stderr, "NPU: shared expert of %u layers\n", shexp_layers);
      }
      std::uint32_t whole_layers = 0;
      if (WholeNpuEnabled()) {
        for (auto& l : m->layers_) {
          if (l.linear && l.ssm_in.data != nullptr &&
              l.ssm_in.type == core::GgmlType::kQ8_0 &&
              l.ssm_in.rows == 16 * npu->n() && l.ssm_in.cols == npu->k()) {
            std::vector<int> ids;
            for (std::uint32_t i = 0; i < 16; ++i) {
              const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
              if (id < 0)
                break;
              NpuEncodeWeights(l.ssm_in.data, l.ssm_in.cols,
                               static_cast<std::uint32_t>(i) * npu->n(),
                               npu->n(), npu->k(), 0, npu->k(), npu->Device(id),
                               nullptr);
              ids.push_back(id);
              staged += npu->BBytes();
            }
            if (hipDeviceSynchronize() != hipSuccess)
              ids.clear();
            for (int id : ids)
              npu->FinalizeWeights(id);
            if (ids.size() == 16) {
              l.npu_ssm_all = std::move(ids);
              ++whole_layers;
            }
          }
        }
        for (auto& l : m->layers_) {
          if (!l.linear && l.attn_qkv.data != nullptr &&
              l.attn_qkv.type == core::GgmlType::kQ8_0 &&
              l.attn_qkv.rows == 13 * npu->n() && l.attn_qkv.cols == npu->k()) {
            std::vector<int> ids;
            for (std::uint32_t i = 0; i < 13; ++i) {
              const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
              if (id < 0)
                break;
              NpuEncodeWeights(l.attn_qkv.data, l.attn_qkv.cols,
                               static_cast<std::uint32_t>(i) * npu->n(),
                               npu->n(), npu->k(), 0, npu->k(), npu->Device(id),
                               nullptr);
              ids.push_back(id);
              staged += npu->BBytes();
            }
            if (hipDeviceSynchronize() != hipSuccess)
              ids.clear();
            for (int id : ids)
              npu->FinalizeWeights(id);
            if (ids.size() == 13) {
              l.npu_qkv_all = std::move(ids);
              ++whole_layers;
            }
          }
        }
      }
      if (hipDeviceSynchronize() != hipSuccess) {
        if (error_msg != nullptr) {
          *error_msg = "NPU weight staging failed";
        }
        return nullptr;
      }
      std::fprintf(stderr, "NPU: z projection of %u layers, %.2f GiB staged\n",
                   layers_on,
                   static_cast<double>(staged) / (1024.0 * 1024.0 * 1024.0));
      if (layers_on != 0 || whole_layers != 0) {
        m->npu_ = std::move(npu);
      }
    }
  }
#endif
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
