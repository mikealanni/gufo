#ifndef GUFO_MODELS_LAGUNA_GPU_DEVICE_MODEL_HPP_
#define GUFO_MODELS_LAGUNA_GPU_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/runtime/config.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::laguna::gpu {

/// One weight resident in device memory, still in its GGUF encoding.
/// `rows` is ne[1] and `experts` ne[2], so M and the expert count stay apart.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{1};
  std::uint32_t experts{1};
};

struct DeviceLayer {
  bool is_moe{false};
  DeviceTensor attn_norm, attn_q, attn_k, attn_v, attn_gate, attn_output;
  DeviceTensor attn_q_norm, attn_k_norm;
  DeviceTensor ffn_norm;
  DeviceTensor ffn_gate, ffn_up, ffn_down;
  DeviceTensor router, router_bias;
  DeviceTensor gate_exps, up_exps, down_exps;
  DeviceTensor gate_shexp, up_shexp, down_shexp;
  // Shared-expert gate/up weights staged on the NPU (block-float, one buffer
  // each - npu->n()=1024 already matches shared_ff exactly, no chunking
  // needed). Empty when NPU offload is unavailable or this layer's shapes
  // don't fit.
  std::vector<int> npu_shexp;
  // QKV staged on the NPU: attn_q chunked into npu->n()=1024-row blocks (up
  // to kNpuMaxSlices for the widest, 72-head layers), attn_k/attn_v each a
  // single block (their row count, num_kv*head_dim, is already 1024).
  std::vector<int> npu_q, npu_k, npu_v;
};

/// Per-layer-type rope: effective inverse frequencies of the rotated pairs.
struct RopeTables {
  float* inv_freq{nullptr};
  std::uint32_t n_dims{0};
  float mscale{1.0F};
};

/// The whole model uploaded to the GPU. Routed experts stored as Q6_K or BF16
/// are re-encoded as Q8_0, the nearest type the routed kernels read.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const core::GgufReader& reader, const Config& config,
      std::string* error_msg);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceTensor& output_norm() const noexcept { return output_norm_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  const RopeTables& rope(bool swa) const noexcept {
    return swa ? rope_swa_ : rope_full_;
  }
  std::size_t resident_bytes() const noexcept { return bytes_; }
  // NPU shared-expert offload (prefill only; fixed M=4096, no runtime-M
  // sweep - see laguna-port-state.md for why this is scoped down from
  // Flint's full runtime-M/ping-pong-bank version). nullptr when
  // GUFO_NPU_DIR is unset or the K=3072 program failed to load.
  qwen38_flash_next::rocm::NpuGemm* npu() const noexcept { return npu_.get(); }

private:
  DeviceModel() = default;
  bool StageNpuWeights();

  Config config_;
  DeviceTensor token_embd_, output_, output_norm_;
  std::vector<DeviceLayer> layers_;
  RopeTables rope_full_, rope_swa_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::unique_ptr<qwen38_flash_next::rocm::NpuGemm> npu_;
};

}  // namespace gufo::models::laguna::gpu

#endif  // GUFO_MODELS_LAGUNA_GPU_DEVICE_MODEL_HPP_
