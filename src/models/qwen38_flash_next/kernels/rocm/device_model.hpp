#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  /// A Q5_1 re-encoding of a large Q8_0 matrix that the decode GEMV reads in
  /// place of `data` (prefill keeps the Q8_0 rows); null when there is none.
  void* decode_data{nullptr};
  /// Type of `decode_data`: Q4_1, or Q5_1 for GUFO_DECODE_COPY=q5.
  core::GgmlType decode_type{core::GgmlType::kQ5_1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceMixer {
  DeviceTensor norm;
  DeviceTensor down;
  DeviceTensor up;
  DeviceTensor inject;
};

struct DeviceLayer {
  bool linear{false};
  DeviceMixer hc_attn;
  DeviceMixer hc_ffn;

  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_norm, ssm_out;
  /// qkv and gate rows stacked ([hidden -> conv channels + value dim]) when
  /// both are Q8_0; then ssm_qkv/ssm_gate are empty.
  DeviceTensor ssm_in;
  /// NPU buffer ids of the six 1024-row slices of the z half of ssm_in (rows
  /// 10240..16383) in block-float tile layout; empty when the NPU is off.
  std::vector<int> npu_z;
  /// NPU buffer ids of the 1024-row slices of the attention gate rows (four
  /// heads each) of attn_qkv; empty when unused.
  std::vector<int> npu_gate;
  /// NPU buffer ids of the shared expert gate and up weights (640 rows each,
  /// in 1024-row slices with the unused rows left unwritten); empty when
  /// unused.
  std::vector<int> npu_shexp;
  /// Whole ssm_in slices (16) staged only when GUFO_NPU_WHOLE=1.
  std::vector<int> npu_ssm_all;
  /// Whole attn_qkv slices (13) staged only when GUFO_NPU_WHOLE=1.
  std::vector<int> npu_qkv_all;
  /// alpha and beta rows stacked: [hidden -> 2 * v_heads] F16.
  DeviceTensor ssm_alpha_beta;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm,
      indexer_q, indexer_k, indexer_q_norm, indexer_k_norm;
  /// [q|gate ; k ; v] rows stacked when all are Q8_0; then attn_q/k/v are
  /// empty.
  DeviceTensor attn_qkv;
  DeviceTensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv,
      ple_conv1d;
  /// Router rows followed by the shared-expert gate row:
  /// [hidden -> num_experts + 1] F16.
  DeviceTensor router;
  DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps, shexp_gate, shexp_up,
      shexp_down;
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_fc_embedding, nextn_fc_hidden;
  DeviceMixer nextn_head;
};

/// The trunk (and optionally the MTP draft block) uploaded to the GPU. The
/// n-gram table is never uploaded: it is read from disk per token.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors from the same open files used to bind their metadata.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, const core::GgufReader* mtp_reader,
      std::string* error_msg = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceMixer& hc_head() const noexcept { return hc_head_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  /// The NPU GEMM engine, or null when the NPU is not in use.
  NpuGemm* npu() const noexcept { return npu_.get(); }
  /// Instruction program and padded token count of an NPU group of `rows`
  /// rows (at most one full group). With the runtime-M program the group runs
  /// at the smallest multiple of 512 that fits (at least 1024); otherwise only
  /// a full group is served and the program id is -1.
  int NpuProgramFor(std::uint32_t rows,
                    std::uint32_t* run_tokens) const noexcept;
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 matrices (decode activation quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }
  /// Largest dense Q4_K/Q5_K/Q5_1 matrix, in elements (0 when there is none).
  [[nodiscard]] std::size_t max_dequant_elems() const noexcept {
    return max_dequant_elems_;
  }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceMixer hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  std::vector<void*> allocations_;
  std::unique_ptr<NpuGemm> npu_;
  /// Program id per padded group size / 512 (-1 when the runtime-M streams
  /// are not loaded); index 8 is the full 4096-row group, program 0.
  int npu_m_program_[9]{-1, -1, -1, -1, -1, -1, -1, -1, 0};
  bool npu_m_runtime_{false};
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
  std::size_t max_dequant_elems_{0};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
