#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_FINISH_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_FINISH_HPP_

#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// One rule that fired on a finished turn. `score` is the rule's confidence
/// that the model stopped before completing its work, in (0, 1].
struct JevFinishHit {
  std::string rule;
  std::string reason;
  float score{0.0F};
};

/// Verdict for one model turn. Silence means finished: `unfinished` is false
/// unless a rule reached `kJevFinishThreshold`.
struct JevFinishVerdict {
  bool unfinished{false};
  std::string reason;
  float score{0.0F};
};

/// Minimum rule score for `JudgeFinish` to report an unfinished turn. Rules
/// below it still appear in `EvaluateFinishRules` for calibration.
inline constexpr float kJevFinishThreshold = 0.6F;

/// Every rule that fires for the turn, strongest first. A turn that ends in a
/// tool call is never unfinished and returns no hits.
[[nodiscard]] std::vector<JevFinishHit> EvaluateFinishRules(
    std::string_view task, std::string_view response, bool had_tool_call,
    std::string_view finish_reason);

/// Cheap rule-based check that the model stopped before finishing: an unclosed
/// code fence, a sentence cut off mid-way, an announced next action that never
/// ran, or an empty answer. Precision first: ambiguous endings stay finished.
[[nodiscard]] JevFinishVerdict JudgeFinish(std::string_view task,
                                           std::string_view response,
                                           bool had_tool_call,
                                           std::string_view finish_reason);

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_FINISH_HPP_
