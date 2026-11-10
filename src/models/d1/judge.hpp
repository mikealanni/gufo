#ifndef GUFO_MODELS_D1_JUDGE_HPP_
#define GUFO_MODELS_D1_JUDGE_HPP_
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/d1/model.hpp"

namespace gufo::models::d1 {
struct JudgeVerdict {
  float yes{0.0F};
  float no{0.0F};
};
/// Yes/no judge over a d1 decision model: each (question, context) pair is a
/// `noul` question over the context as its state. Pairs with the same context
/// share one forward pass, packed up to 16 at a time. The forward itself is
/// injected (NPU backbone in serving, the FP32 host oracle in tests).
class Judge {
public:
  /// Logits for `slots` over `vocabulary`, row-major slots x vocabulary.
  using ForwardFn = std::function<bool(
      std::span<const std::uint32_t> tokens,
      std::span<const std::int32_t> groups,
      std::span<const std::uint32_t> slots,
      std::span<const std::uint32_t> vocabulary, std::span<float> logits)>;

  static std::unique_ptr<Judge> Create(const core::GgufReader& reader,
                                       ForwardFn forward,
                                       std::uint32_t max_tokens,
                                       std::string* error = nullptr);
  ~Judge();
  /// One verdict per item, same order; empty on failure.
  std::vector<JudgeVerdict> Answer(
      std::span<const std::pair<std::string_view, std::string_view>> items)
      const;

private:
  Judge();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::models::d1
#endif
