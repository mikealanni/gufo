#ifndef GUFO_CLI_JUDGE_HPP_
#define GUFO_CLI_JUDGE_HPP_

#include <span>

namespace gufo::cli {

/// Standalone resident-judge run: loads a small GGUF, scores one question (or
/// the built-in calibration battery) on the NPU, and prints per-call timings.
[[nodiscard]] int RunJudge(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_JUDGE_HPP_
