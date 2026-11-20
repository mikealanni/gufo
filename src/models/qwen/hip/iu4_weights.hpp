#ifndef GUFO_MODELS_QWEN_HIP_IU4_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN_HIP_IU4_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

enum class Iu4Role : std::uint8_t {
  kAttnQ,
  kAttnK,
  kAttnV,
  kAttnO,
  kSsmQkv,
  kSsmZ,
  kSsmA,
  kSsmB,
  kSsmOut,
  kFfnGate,
  kFfnUp,
  kFfnDown,
  kCount,
};

// One rotated symmetric 4-bit projection resident on the device. `tiles` and
// `scales` are in the layout documented in ops/iu4.hpp; `awq` is the f32 per
// input-channel divide vector (null for the lm_head, whose input is not AWQ
// scaled).
struct Iu4Projection {
  void* tiles{nullptr};
  std::uint16_t* scales{nullptr};
  float* awq{nullptr};
  std::uint32_t rows{0};
  std::uint32_t cols{0};
  [[nodiscard]] bool valid() const noexcept { return tiles != nullptr; }
};

// Device-resident copy of the rotated 4-bit projections of a Qwen3.5-family
// artifact (the published "mq4" container). Projections are converted at load
// time without requantizing: payload nibbles and group scales are preserved
// bit for bit, only the byte order changes.
class Iu4Weights {
public:
  Iu4Weights() = default;
  Iu4Weights(const Iu4Weights&) = delete;
  Iu4Weights& operator=(const Iu4Weights&) = delete;
  ~Iu4Weights();

  // Loads every projection of `layers` layers plus lm_head. Returns false and
  // fills `error` when the file is malformed, a tensor is missing, or a group
  // has a zero point other than -8 scales.
  bool Load(const std::string& path, std::uint32_t layers, std::string* error);
  void Release() noexcept;

  [[nodiscard]] const Iu4Projection& Find(std::uint32_t layer,
                                          Iu4Role role) const noexcept;
  [[nodiscard]] const Iu4Projection& LmHead() const noexcept {
    return lm_head_;
  }
  [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }
  [[nodiscard]] std::size_t device_bytes() const noexcept {
    return device_bytes_;
  }

private:
  std::vector<Iu4Projection> projections_;
  Iu4Projection lm_head_;
  std::uint32_t layers_{0};
  std::size_t device_bytes_{0};
};

}  // namespace gufo::hip
#endif
#endif
