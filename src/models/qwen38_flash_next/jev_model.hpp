#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_MODEL_HPP_

#include <memory>
#include <string>
#include <string_view>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

/// Probability mass jev assigns to the two answers at the reply position.
struct JevBinaryVerdict {
  float yes{0.0F};
  float no{0.0F};
};

/// Zero-shot binary judge ("jev") backed by a small dense Qwen3 instruct GGUF.
///
/// The model reads a question plus optional context and is scored once at the
/// answer position: `yes` and `no` are the normalized probabilities of the
/// vocabulary entries that spell each word, not free-form generation.
class JevModel {
public:
  ~JevModel();
  JevModel(const JevModel&) = delete;
  JevModel& operator=(const JevModel&) = delete;

  /// Loads a dense Qwen3 GGUF (for example Qwen3-0.6B). Fails for any other
  /// architecture or an unsupported tensor layout.
  [[nodiscard]] static std::unique_ptr<JevModel> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Scores the question with `context` as preceding assistant reasoning.
  [[nodiscard]] JevBinaryVerdict Answer(std::string_view question,
                                        std::string_view context) const;

private:
  struct Impl;
  explicit JevModel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_MODEL_HPP_
