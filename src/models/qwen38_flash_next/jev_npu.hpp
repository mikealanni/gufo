#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_NPU_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_NPU_HPP_

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/jev_model.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

/// NPU-resident jev scorer. Same contract as JevModel but runs the whole
/// judge pass on the XDNA2 NPU (resident GEMMs) with GPU glue, in a single
/// M=512 prefill instead of 512 autoregressive GEMVs. Returns {} if the NPU
/// path is unavailable (no XRT, shape mismatch, unhealthy executor).
class JevNpuModel {
public:
  ~JevNpuModel();
  JevNpuModel(const JevNpuModel&) = delete;
  JevNpuModel& operator=(const JevNpuModel&) = delete;

  [[nodiscard]] static std::unique_ptr<JevNpuModel> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  [[nodiscard]] JevBinaryVerdict Answer(std::string_view question,
                                        std::string_view context) const;

  /// Answers up to 4 independent questions in one NPU pass (see
  /// jev_npu.hip.cpp's kJevBatchMax/RunBatch) -- far cheaper per-question
  /// than repeated Answer() calls, since the NPU program and every staging
  /// buffer already process a fixed M=512 rows regardless of how many are
  /// real tokens. Returns one verdict per input, same order; empty on
  /// failure of the whole batch.
  [[nodiscard]] std::vector<JevBinaryVerdict> AnswerBatch(
      std::span<const std::pair<std::string_view, std::string_view>> items)
      const;

private:
  struct Impl;
  explicit JevNpuModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_NPU_HPP_
