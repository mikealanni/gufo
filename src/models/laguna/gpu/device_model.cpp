#include "src/models/laguna/gpu/device_model.hpp"

#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::laguna::gpu {
namespace {

constexpr std::size_t kCopyChunk = std::size_t{256} << 20;
constexpr std::uint32_t kQ8Block = 32;
constexpr std::size_t kQ8Bytes = 34;
constexpr std::uint32_t kQ6KBlock = 256;
constexpr std::size_t kQ6KBytes = 210;
constexpr float kPi = 3.14159265358979323846F;

/// Releases the mapped file pages of an uploaded range so host memory holds
/// only the chunk in flight.
void DropPages(const void* data, std::size_t bytes) {
  constexpr std::uintptr_t kPage = 4096;
  const auto begin =
      (reinterpret_cast<std::uintptr_t>(data) + kPage - 1) & ~(kPage - 1);
  const auto end =
      (reinterpret_cast<std::uintptr_t>(data) + bytes) & ~(kPage - 1);
  if (end > begin) {
    (void)madvise(reinterpret_cast<void*>(begin), end - begin, MADV_DONTNEED);
  }
}

void SetError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr)
    *error_msg = message;
}

std::string P(std::uint32_t layer, const char* name) {
  return "blk." + std::to_string(layer) + "." + name;
}

std::uint16_t FloatToHalf(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

class Uploader {
public:
  Uploader(const core::GgufReader& reader, hip::WeightUpload* stager,
           std::vector<void*>* allocations, std::size_t* bytes,
           std::string* error)
      : stager_(stager),
        regions_(reader.GetMappedRegions().begin(),
                 reader.GetMappedRegions().end()),
        allocations_(allocations),
        bytes_(bytes),
        error_(error) {
    for (const auto& t : reader.GetTensors()) {
      tensors_.emplace(std::string(t.name), &t);
    }
  }

  bool Finish() {
    std::string error;
    if (!stager_->Finish(&error)) {
      SetError(error_, "weight upload failed: " + error);
      return false;
    }
    return true;
  }

  /// Uploads a tensor. Q6_K and BF16 matrices become Q8_0 when `to_q8`.
  bool Bind(DeviceTensor* out, const std::string& name, bool to_q8 = false) {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      SetError(error_, "missing tensor: " + name);
      return false;
    }
    const auto& t = *it->second;
    out->cols = static_cast<std::uint32_t>(t.dimensions[0]);
    out->rows = t.dimensions.size() > 1
                    ? static_cast<std::uint32_t>(t.dimensions[1])
                    : 1;
    out->experts = t.dimensions.size() > 2
                       ? static_cast<std::uint32_t>(t.dimensions[2])
                       : 1;
    out->type = t.type;
    if (to_q8 &&
        (t.type == core::GgmlType::kQ6_K || t.type == core::GgmlType::kBF16)) {
      out->type = core::GgmlType::kQ8_0;
      return ReencodeToQ8(out, t);
    }
    return Stage(out, t);
  }

  bool Floats(float** out, const std::vector<float>& values) {
    DeviceTensor t;
    if (!Alloc(&t, values.size() * sizeof(float)))
      return false;
    if (hipMemcpy(t.data, values.data(), values.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      SetError(error_, "device copy failed");
      return false;
    }
    *out = static_cast<float*>(t.data);
    return true;
  }

private:
  bool Alloc(DeviceTensor* out, std::size_t bytes) {
    if (hipMalloc(&out->data, bytes) != hipSuccess) {
      SetError(error_, "device allocation failed (" + std::to_string(bytes) +
                           " bytes)");
      return false;
    }
    allocations_->push_back(out->data);
    *bytes_ += bytes;
    return true;
  }

  /// Queues a tensor on the O_DIRECT staging pipeline, bypassing the page
  /// cache; the destination is valid after Finish().
  bool Stage(DeviceTensor* out, const core::GgufTensorInfo& t) {
    const auto* at = static_cast<const char*>(t.data);
    for (std::uint32_t shard = 0; shard < regions_.size(); ++shard) {
      const auto* base = static_cast<const char*>(regions_[shard].data);
      if (at >= base && at + t.size_bytes <= base + regions_[shard].size) {
        if (!Alloc(out, t.size_bytes))
          return false;
        std::string error;
        if (!stager_->Copy(shard, static_cast<std::uint64_t>(at - base),
                           t.size_bytes, out->data, &error)) {
          SetError(error_, "weight upload failed: " + error);
          return false;
        }
        return true;
      }
    }
    return Copy(out, t.data, t.size_bytes);
  }

  bool Copy(DeviceTensor* out, const void* src, std::size_t bytes) {
    if (!Alloc(out, bytes))
      return false;
    for (std::size_t off = 0; off < bytes; off += kCopyChunk) {
      const std::size_t n = std::min(kCopyChunk, bytes - off);
      if (hipMemcpy(static_cast<char*>(out->data) + off,
                    static_cast<const char*>(src) + off, n,
                    hipMemcpyHostToDevice) != hipSuccess) {
        SetError(error_, "device copy failed");
        return false;
      }
      DropPages(static_cast<const char*>(src) + off, n);
    }
    return true;
  }

  static void DequantRow(const core::GgufTensorInfo& t, std::uint64_t row,
                         std::uint32_t cols, float* values) {
    if (t.type == core::GgmlType::kQ6_K) {
      const std::size_t src_row = cols / kQ6KBlock * kQ6KBytes;
      quant::DequantizeQ6_K(static_cast<const char*>(t.data) + row * src_row,
                            values, cols);
      return;
    }
    const auto* src = reinterpret_cast<const std::uint16_t*>(t.data) +
                      row * static_cast<std::size_t>(cols);
    for (std::uint32_t i = 0; i < cols; ++i)
      values[i] = Bf16ToFloat(src[i]);
  }

  bool ReencodeToQ8(DeviceTensor* out, const core::GgufTensorInfo& t) {
    const std::uint64_t n_rows =
        static_cast<std::uint64_t>(out->rows) * out->experts;
    const std::size_t dst_row = out->cols / kQ8Block * kQ8Bytes;
    if (!Alloc(out, dst_row * n_rows))
      return false;
    const std::uint64_t rows_per_chunk =
        std::max<std::uint64_t>(1, kCopyChunk / dst_row);
    const std::size_t src_row_bytes = t.size_bytes / n_rows;
    std::vector<char> host(rows_per_chunk * dst_row);
    const unsigned max_workers = std::thread::hardware_concurrency();
    const unsigned workers = std::min(8U, std::max(1U, max_workers));
    const bool log_progress = std::getenv("LAGUNA_LOAD_LOG") != nullptr;
    for (std::uint64_t r0 = 0; r0 < n_rows; r0 += rows_per_chunk) {
      const std::uint64_t n = std::min(rows_per_chunk, n_rows - r0);
      if (log_progress) {
        std::fprintf(stderr,
                     "[laguna] reencode %.*s: %llu/%llu rows (%u workers)\n",
                     static_cast<int>(t.name.size()), t.name.data(),
                     static_cast<unsigned long long>(r0),
                     static_cast<unsigned long long>(n_rows), workers);
      }
      std::vector<std::thread> pool;
      for (unsigned w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
          std::vector<float> values(out->cols);
          for (std::uint64_t r = n * w / workers; r < n * (w + 1) / workers;
               ++r) {
            DequantRow(t, r0 + r, out->cols, values.data());
            char* dst = host.data() + r * dst_row;
            for (std::uint32_t b = 0; b < out->cols / kQ8Block; ++b) {
              const float* v = values.data() + b * kQ8Block;
              float amax = 0.0F;
              for (std::uint32_t i = 0; i < kQ8Block; ++i) {
                amax = std::max(amax, std::fabs(v[i]));
              }
              const float scale = amax / 127.0F;
              const float inv = scale > 0.0F ? 1.0F / scale : 0.0F;
              const std::uint16_t d = FloatToHalf(scale);
              char* block = dst + b * kQ8Bytes;
              std::memcpy(block, &d, sizeof(d));
              for (std::uint32_t i = 0; i < kQ8Block; ++i) {
                block[2 + i] = static_cast<char>(
                    static_cast<signed char>(std::lround(v[i] * inv)));
              }
            }
          }
        });
      }
      for (auto& th : pool)
        th.join();
      if (log_progress) {
        std::fprintf(stderr,
                     "[laguna] reencode %.*s: chunk %llu-%llu done (%.1f GB)\n",
                     static_cast<int>(t.name.size()), t.name.data(),
                     static_cast<unsigned long long>(r0),
                     static_cast<unsigned long long>(r0 + n),
                     static_cast<double>(*bytes_) / 1e9);
      }
      if (hipMemcpy(static_cast<char*>(out->data) + r0 * dst_row, host.data(),
                    n * dst_row, hipMemcpyHostToDevice) != hipSuccess) {
        SetError(error_, "device copy failed");
        return false;
      }
      DropPages(static_cast<const char*>(t.data) + r0 * src_row_bytes,
                n * src_row_bytes);
    }
    return true;
  }

  std::unordered_map<std::string, const core::GgufTensorInfo*> tensors_;
  hip::WeightUpload* stager_;
  std::vector<core::GgufMappedRegion> regions_;
  std::vector<void*>* allocations_;
  std::size_t* bytes_;
  std::string* error_;
};

/// Effective inverse frequency of every rotated pair, matching the CPU
/// reference: plain rope, or YaRN blending of interpolated and extrapolated
/// frequencies.
std::vector<float> InvFreq(const Config& c, bool swa) {
  const std::uint32_t n_dims = swa ? c.rope_dims_swa : c.rope_dims_full;
  const float base = swa ? c.rope_base_swa : c.rope_base;
  const float freq_scale = 1.0F / c.yarn_factor;
  float low = 0.0F;
  float high = 0.0F;
  if (!swa) {
    const auto corr_dim = [&](float beta) {
      return static_cast<float>(n_dims) *
             std::log(static_cast<float>(c.yarn_orig_ctx) /
                      (beta * 2.0F * kPi)) /
             (2.0F * std::log(base));
    };
    low = std::max(0.0F, std::floor(corr_dim(c.yarn_beta_fast)));
    high = std::min(static_cast<float>(n_dims - 1),
                    std::ceil(corr_dim(c.yarn_beta_slow)));
  }
  std::vector<float> out(n_dims / 2);
  for (std::uint32_t i = 0; i < out.size(); ++i) {
    const float extrap =
        std::pow(base, -static_cast<float>(2 * i) / static_cast<float>(n_dims));
    float f = extrap;
    if (!swa) {
      const float interp = freq_scale * extrap;
      const float y =
          (static_cast<float>(i) - low) / std::max(0.001F, high - low);
      const float ramp = 1.0F - std::min(1.0F, std::max(0.0F, y));
      f = interp * (1.0F - ramp) + extrap * ramp;
    }
    out[i] = f;
  }
  return out;
}

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const core::GgufReader& reader,
                                                 const Config& config,
                                                 std::string* error_msg) {
  std::unique_ptr<DeviceModel> model(new DeviceModel());
  model->config_ = config;
  const auto mapped = reader.GetMappedRegions();
  const std::vector<core::GgufMappedRegion> shards(mapped.begin(),
                                                   mapped.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager)
    return nullptr;
  Uploader up(reader, stager.get(), &model->allocations_, &model->bytes_,
              error_msg);

  if (!up.Bind(&model->token_embd_, "token_embd.weight") ||
      !up.Bind(&model->output_norm_, "output_norm.weight") ||
      !up.Bind(&model->output_, "output.weight")) {
    return nullptr;
  }
  if (model->token_embd_.type != core::GgmlType::kQ8_0) {
    SetError(error_msg, "Laguna: token_embd must be Q8_0");
    return nullptr;
  }
  model->rope_full_.n_dims = config.rope_dims_full;
  model->rope_full_.mscale = config.yarn_attn_factor;
  model->rope_swa_.n_dims = config.rope_dims_swa;
  model->rope_swa_.mscale = 1.0F;
  if (!up.Floats(&model->rope_full_.inv_freq, InvFreq(config, false)) ||
      !up.Floats(&model->rope_swa_.inv_freq, InvFreq(config, true))) {
    return nullptr;
  }

  model->layers_.resize(config.num_layers);
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    auto& L = model->layers_[il];
    L.is_moe = il >= config.leading_dense;
    if (std::getenv("LAGUNA_LOAD_LOG") != nullptr)
      std::fprintf(stderr, "[laguna] layer %u start (%.1f GB)\n", il,
                   static_cast<double>(model->bytes_) / 1e9);
    bool ok = up.Bind(&L.attn_norm, P(il, "attn_norm.weight")) &&
              up.Bind(&L.attn_q, P(il, "attn_q.weight")) &&
              up.Bind(&L.attn_k, P(il, "attn_k.weight")) &&
              up.Bind(&L.attn_v, P(il, "attn_v.weight")) &&
              up.Bind(&L.attn_gate, P(il, "attn_gate.weight")) &&
              up.Bind(&L.attn_output, P(il, "attn_output.weight")) &&
              up.Bind(&L.attn_q_norm, P(il, "attn_q_norm.weight")) &&
              up.Bind(&L.attn_k_norm, P(il, "attn_k_norm.weight")) &&
              up.Bind(&L.ffn_norm, P(il, "ffn_norm.weight"));
    if (std::getenv("LAGUNA_LOAD_LOG") != nullptr)
      std::fprintf(stderr, "[laguna] layer %u attn done (%.1f GB)\n", il,
                   static_cast<double>(model->bytes_) / 1e9);
    if (ok && !L.is_moe) {
      ok = up.Bind(&L.ffn_gate, P(il, "ffn_gate.weight")) &&
           up.Bind(&L.ffn_up, P(il, "ffn_up.weight")) &&
           up.Bind(&L.ffn_down, P(il, "ffn_down.weight"));
    }
    if (ok && L.is_moe) {
      ok = up.Bind(&L.router, P(il, "ffn_gate_inp.weight")) &&
           up.Bind(&L.router_bias, P(il, "exp_probs_b.bias")) &&
           up.Bind(&L.gate_exps, P(il, "ffn_gate_exps.weight"), true) &&
           up.Bind(&L.up_exps, P(il, "ffn_up_exps.weight"), true);
      if (std::getenv("LAGUNA_LOAD_LOG") != nullptr)
        std::fprintf(stderr, "[laguna] layer %u gate/up done (%.1f GB)\n", il,
                     static_cast<double>(model->bytes_) / 1e9);
      if (ok)
        ok = up.Bind(&L.down_exps, P(il, "ffn_down_exps.weight"), true);
      if (std::getenv("LAGUNA_LOAD_LOG") != nullptr)
        std::fprintf(stderr, "[laguna] layer %u down done (%.1f GB)\n", il,
                     static_cast<double>(model->bytes_) / 1e9);
      if (ok)
        ok = up.Bind(&L.gate_shexp, P(il, "ffn_gate_shexp.weight")) &&
             up.Bind(&L.up_shexp, P(il, "ffn_up_shexp.weight")) &&
             up.Bind(&L.down_shexp, P(il, "ffn_down_shexp.weight"));
    }
    if (!ok)
      return nullptr;
    if (std::getenv("LAGUNA_LOAD_LOG") != nullptr) {
      std::fprintf(stderr, "[laguna] uploaded layer %u (%.1f GB)\n", il,
                   static_cast<double>(model->bytes_) / 1e9);
    }
  }
  if (!up.Finish())
    return nullptr;
  return model;
}

}  // namespace gufo::models::laguna::gpu
