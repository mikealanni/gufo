#ifndef GUFO_MODELS_LING32_RUNTIME_MANIFEST_HPP_
#define GUFO_MODELS_LING32_RUNTIME_MANIFEST_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling32/runtime/config.hpp"

namespace gufo::models::ling32 {

enum class Slot {
  kTokenEmbd,
  kOutput,
  kOutputNorm,
  kAttnNorm,
  kFfnNorm,
  kKdaQ,
  kKdaK,
  kKdaV,
  kKdaOutput,
  kKdaA,
  kKdaBeta,
  kKdaDt,
  kKdaConvQ,
  kKdaConvK,
  kKdaConvV,
  kKdaFA,
  kKdaGA,
  kKdaNorm,
  kMlaQ,
  kMlaKvA,
  kMlaKvANorm,
  kMlaKB,
  kMlaVB,
  kMlaGate,
  kMlaOutput,
  kDenseGate,
  kDenseUp,
  kDenseDown,
  kRouter,
  kRouterBias,
  kExpertGate,
  kExpertUp,
  kExpertDown,
  kSharedGate,
  kSharedUp,
  kSharedDown,
  kNextnEhProj,
  kNextnEnorm,
  kNextnHnorm,
  kNextnSharedHeadNorm,
};

struct ManifestEntry {
  Slot slot;
  std::uint32_t layer;
  const core::GgufTensorInfo* tensor;
};

struct Manifest {
  std::vector<ManifestEntry> entries;
  std::uint64_t float_bytes{0};
  std::uint64_t quantized_bytes{0};
  std::uint64_t expert_bytes{0};
  std::uint64_t mtp_bytes{0};

  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return float_bytes + quantized_bytes;
  }
};

[[nodiscard]] bool IsFloatSlot(Slot slot) noexcept;

[[nodiscard]] std::optional<Manifest> BuildManifest(
    const Config& config, const core::GgufReader& reader,
    std::string* error_msg);

}  // namespace gufo::models::ling32

#endif  // GUFO_MODELS_LING32_RUNTIME_MANIFEST_HPP_
