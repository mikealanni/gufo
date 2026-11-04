#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GATE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GATE_HPP_

#include <string_view>

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

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_JEV_GATE_HPP_
