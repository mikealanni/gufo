#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GOAL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GOAL_HPP_

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// One tool call the assistant made: id, tool name, the shell command (shell
/// tools) and the file it touches (edit/write tools).
struct JevGoalCall {
  std::string_view id;
  std::string_view name;
  std::string_view command;
  std::string_view path;
};

/// Role-neutral view of one conversation message. Views must outlive the call.
struct JevGoalMessage {
  enum class Role { kUser, kAssistant, kTool };
  Role role{Role::kUser};
  std::string_view text;
  std::vector<JevGoalCall> calls;  // assistant messages
  std::string_view tool_call_id;   // tool messages
  std::string_view tool_name;      // tool messages (fallback to the call name)
};

/// Result of scanning a request for the magic phrase. `index` is the message
/// that carries the latest trigger; the evidence window starts there.
struct JevGoalSpec {
  bool active{false};
  std::string text;  // goal text with the phrase stripped
  std::size_t index{0};
};

/// What the tracker wants to do when the model hands control back.
enum class JevGoalAction { kNone, kDone, kContinue, kNudge };

struct JevGoalDecision {
  JevGoalAction action{JevGoalAction::kNone};
  std::size_t items{0};
  std::size_t open{0};
  std::string rule;     // which rule produced the action (empty for done)
  std::string reason;   // short log reason
  std::string message;  // user turn to append for continue/nudge
};

/// Max automatic continuations the tracker adds to one request chain.
inline constexpr int kJevGoalMaxContinues = 3;

/// True when `text` contains one of the accepted trigger phrases, tolerant of
/// case, punctuation, whitespace and apostrophes ("Jev, make sure it's done.").
[[nodiscard]] bool HasJevGoalTrigger(std::string_view text);

/// `text` without the trigger phrases and the separators around them. A pure
/// function of `text`, so a conversation renders the same bytes on every
/// request. An empty result becomes "Continue." so no empty turn is rendered.
[[nodiscard]] std::string StripJevGoalTrigger(std::string_view text);

/// Scans the RAW (unstripped) user messages. Active when any user message
/// carries the trigger; the goal text is the stripped latest trigger message,
/// or the previous non-empty user message when that one is only the phrase.
[[nodiscard]] JevGoalSpec DetectJevGoal(
    std::span<const JevGoalMessage> messages);

/// Decision for a turn that ended with no tool call. `history` holds the
/// conversation before the final assistant turn (stripped or raw: only tool
/// calls and results are read), `final_text` is that turn's text and
/// `continues` the automatic continuations already spent on this chain.
/// Pure rules, no model call; silence (kDone) when nothing is open.
[[nodiscard]] JevGoalDecision EvaluateJevGoal(
    const JevGoalSpec& spec, std::span<const JevGoalMessage> history,
    std::string_view final_text, int continues);

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GOAL_HPP_
