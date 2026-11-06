#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GATE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GATE_HPP_

#include <cstddef>
#include <string_view>
#include <vector>

#include "src/core/reasoning.hpp"

namespace gufo::models::qwen38_flash_next {

/// Reasoning effort the pre-gate chose for one request, before decode starts.
struct JevGateDecision {
  ReasoningEffort effort{ReasoningEffort::kMedium};
  bool needs_thinking{true};
};

/// Rule-based effort pre-gate for the request whose pending turn is `text`:
/// the last user message with any attached log, error or code context.
///
/// Trivial lookups collapse to `kLow` without thinking, anything carrying
/// stack-trace, diff or debugging signal stays at `kXHigh`, and ambiguity lands
/// on `kMedium`. The gate never raises above the caller's default and only
/// fills a default here: explicit `reasoning_effort`/`enable_thinking` from
/// the client bypasses it in the HTTP layer.
[[nodiscard]] JevGateDecision ClassifyJevRequest(std::string_view text);

/// One tool result the model is about to read: the tool that ran, the shell
/// command when it was a shell tool, and the text it returned.
struct JevToolResult {
  std::string_view tool;
  std::string_view command;
  std::string_view output;
};

/// Whether the pending agent turn may run at low reasoning effort.
struct JevAgentDecision {
  bool lower{false};
  std::string_view reason;
};

/// Per-turn policy for coding-agent loops. `turns` lists the tool-result
/// groups of the conversation in order; the last group is the pending turn.
/// A turn is lowered only when every result in it is routine (a read, listing,
/// search, edit or inspection command that returned no failure signal) and
/// fewer than `kJevAgentMaxRun` routine turns directly precede it. The result
/// can only lower effort: callers never raise above their own setting.
[[nodiscard]] JevAgentDecision ClassifyJevAgentTurn(
    const std::vector<std::vector<JevToolResult>>& turns);

inline constexpr std::size_t kJevAgentMaxRun = 5;

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GATE_HPP_
