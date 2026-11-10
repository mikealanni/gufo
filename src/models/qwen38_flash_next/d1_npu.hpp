#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_D1_NPU_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_D1_NPU_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "src/models/lfm3/weights.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

/// d1 decision backbone (LFM3 layout) with every projection on the XDNA2 NPU
/// and the rest on the GPU. One prefill over at most 512 prompt tokens; the
/// logits at the requested answer slots are returned for a chosen vocabulary
/// subset. Matrices must be Q4_0; the embedding Q6_K or Q4_0. The GGUF reader
/// must outlive the model (tensors borrow its mapping).
class D1NpuModel {
public:
  ~D1NpuModel();
  D1NpuModel(const D1NpuModel&) = delete;
  D1NpuModel& operator=(const D1NpuModel&) = delete;

  [[nodiscard]] static std::unique_ptr<D1NpuModel> Create(
      const core::GgufReader& reader, std::string* error = nullptr);

  [[nodiscard]] const lfm3::Weights& weights() const;

  /// `groups` is empty for a lone question (full bidirectional global
  /// attention) or one tag per token (-1 state, 1.. question ids).
  /// `sequences` lists ascending start offsets of independent sequences
  /// packed into the one pass (empty = a single sequence); tokens never see,
  /// convolve with or rotate relative to another sequence.
  /// `logits` receives slots.size() * vocabulary.size() floats.
  [[nodiscard]] bool Forward(std::span<const std::uint32_t> tokens,
                             std::span<const std::int32_t> groups,
                             std::span<const std::uint32_t> sequences,
                             std::span<const std::uint32_t> slots,
                             std::span<const std::uint32_t> vocabulary,
                             std::span<float> logits) const;

private:
  struct Impl;
  explicit D1NpuModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_D1_NPU_HPP_
