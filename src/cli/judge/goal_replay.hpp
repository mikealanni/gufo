#ifndef GUFO_CLI_JUDGE_GOAL_REPLAY_HPP_
#define GUFO_CLI_JUDGE_GOAL_REPLAY_HPP_

#include <string>

namespace gufo::cli {

/// Replays exported opencode sessions (one JSONL per session) through the
/// goal tracker and scores each turn-end verdict against what the user did
/// next. Prints precision/recall, per-rule counts, false continues and the
/// per-decision time.
int RunGoalReplay(const std::string& dir, bool verbose);

}  // namespace gufo::cli

#endif  // GUFO_CLI_JUDGE_GOAL_REPLAY_HPP_
