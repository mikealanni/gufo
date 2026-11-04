#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_INLINE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_INLINE_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

/// True when `sentence` is a complete self-posed yes/no check, such as
/// "Should I check the cache first?" or "Is this the root cause?".
///
/// Questions addressed to the reader ("do you want ...") and statements are
/// rejected, so a false positive stays a no-op rather than a wrong answer.
[[nodiscard]] bool IsJevSelfQuestion(std::string_view sentence);

/// Streaming detector for self-posed yes/no questions inside a `<think>` block.
///
/// Feed incrementally decoded reasoning text one piece at a time. `Push`
/// returns the question that just completed at a `?`, normalized to a single
/// line, or nothing. The scanner is cheap and bounded: it keeps only the tail
/// of the current sentence and ignores text inside code fences and inline
/// code, where a `?` is syntax rather than a question.
class JevInlineScanner {
public:
  /// Appends one decoded piece and returns the self-posed question it
  /// completed, if any. A returned question is never returned again.
  [[nodiscard]] std::optional<std::string> Push(std::string_view piece);

  /// Drops buffered text, for example when reasoning closes.
  void Reset();

private:
  void Append(char c);
  void AdvanceBacktick(char c);
  void EndSentence();

  std::string tail_;
  std::size_t backticks_{0};
  char pending_period_{0};
  bool in_fence_{false};
  bool in_inline_code_{false};
  bool overflowed_{false};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_INLINE_HPP_
