#include "src/models/qwen38_flash_next/jev_gate.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen38_flash_next {
namespace {

// Request bodies routinely carry pasted logs; classification only needs the
// head of the text plus coarse size signals, so pattern scanning is bounded.
constexpr std::size_t kScanLimit = 64 * 1024;
constexpr std::size_t kTrivialMaxChars = 160;
constexpr std::size_t kHardMinChars = 2000;
constexpr std::size_t kDiffFilesForXHigh = 2;

// Any one of these marks a request whose answer depends on evidence already in
// the prompt; the model has to reason over it rather than recall it.
constexpr std::array<std::string_view, 16> kStackTraceMarkers{{
    "traceback (most recent call last)",
    "panic:",
    "segmentation fault",
    "core dumped",
    "abort()",
    "caused by:",
    "exception:",
    "assertionerror",
    "undefined reference",
    "runtime error",
    "stack trace",
    "segfault",
    "kernel panic",
    "error[",
    "fatal error",
    "\tat ",
}};

// A file:line reference is where a compiler or test harness pointed the user.
constexpr std::array<std::string_view, 10> kErrorLocations{{
    ".cpp:",
    ".hpp:",
    ".cc:",
    ".py:",
    ".rs:",
    ".go:",
    ".java:",
    ".js:",
    ".ts:",
    ".rb:",
}};

constexpr std::array<std::string_view, 10> kDebugPhrases{{
    "fix this",
    "fix the bug",
    "why is",
    "why does",
    "doesn't work",
    "does not work",
    "not working",
    "won't start",
    "root cause",
    "misbehav",
}};

constexpr std::array<std::string_view, 6> kDebugWords{{
    "debug",
    "broken",
    "regression",
    "repro",
    "failing",
    "crashing",
}};

// Requests that edit or grow code need the full budget even without a trace.
constexpr std::array<std::string_view, 5> kWorkPhrases{{
    "add support",
    "change the",
    "update the",
    "write a",
}};

constexpr std::array<std::string_view, 8> kWorkWords{{
    "implement",
    "refactor",
    "migrate",
    "optimize",
    "optimise",
    "fix",
    "port",
    "rename",
}};

// Recall-shaped asks: the answer is a lookup, not a derivation.
constexpr std::array<std::string_view, 9> kLookupPhrases{{
    "what is",
    "what's",
    "what are",
    "what does",
    "what do",
    "meaning of",
    "definition of",
    "how many",
    "how much",
}};

constexpr std::array<std::string_view, 11> kLookupWords{{
    "where",
    "when",
    "which",
    "translate",
    "list",
    "show",
    "convert",
    "spell",
    "define",
    "summarize",
    "version",
}};

constexpr std::array<std::string_view, 12> kSmallTalk{{
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "yes",
    "no",
    "nice",
    "great",
    "cool",
}};

std::string FoldAscii(std::string_view text) {
  const std::size_t limit = std::min(text.size(), kScanLimit);
  std::string folded;
  folded.reserve(limit);
  for (std::size_t i = 0; i < limit; ++i) {
    const char c = text[i];
    folded.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a')
                                          : c);
  }
  return folded;
}

bool IsWordChar(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

bool HasPhrase(std::string_view text, std::string_view phrase) {
  return text.find(phrase) != std::string_view::npos;
}

bool HasWord(std::string_view text, std::string_view word) {
  for (std::size_t pos = text.find(word); pos != std::string_view::npos;
       pos = text.find(word, pos + 1)) {
    const bool left_clear =
        pos == 0 || !IsWordChar(static_cast<char>(text[pos - 1]));
    const std::size_t next = pos + word.size();
    const bool right_clear =
        next >= text.size() || !IsWordChar(static_cast<char>(text[next]));
    if (left_clear && right_clear) {
      return true;
    }
  }
  return false;
}

bool HasAny(std::string_view text, std::span<const std::string_view> phrases) {
  return std::any_of(
      phrases.begin(), phrases.end(),
      [text](std::string_view phrase) { return HasPhrase(text, phrase); });
}

bool HasAnyWord(std::string_view text,
                std::span<const std::string_view> words) {
  return std::any_of(words.begin(), words.end(), [text](std::string_view word) {
    return HasWord(text, word);
  });
}

std::size_t Count(std::string_view text, std::string_view needle) {
  std::size_t count = 0;
  for (std::size_t pos = text.find(needle); pos != std::string_view::npos;
       pos = text.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

bool HasCode(std::string_view text) {
  return HasPhrase(text, "```") || HasPhrase(text, "~~~") ||
         HasPhrase(text, "diff --git") || HasPhrase(text, "<function=");
}

bool IsSmallTalk(std::string_view text) {
  const std::size_t begin = text.find_first_not_of(" \t\n\r!?.");
  if (begin == std::string_view::npos) {
    return true;
  }
  const std::size_t end = text.find_last_not_of(" \t\n\r!?.");
  const std::string_view trimmed = text.substr(begin, end - begin + 1);
  return std::any_of(
      kSmallTalk.begin(), kSmallTalk.end(),
      [trimmed](std::string_view phrase) { return trimmed == phrase; });
}

bool IsTrivial(std::string_view folded) {
  if (folded.size() > kTrivialMaxChars || HasCode(folded)) {
    return false;
  }
  return IsSmallTalk(folded) || HasPhrase(folded, "?") ||
         HasAny(folded, kLookupPhrases) || HasAnyWord(folded, kLookupWords);
}

bool IsHard(std::string_view folded, bool code) {
  return folded.size() >= kHardMinChars || HasAny(folded, kStackTraceMarkers) ||
         HasAny(folded, kErrorLocations) || HasAny(folded, kDebugPhrases) ||
         HasAnyWord(folded, kDebugWords) ||
         Count(folded, "diff --git") >= kDiffFilesForXHigh ||
         (code &&
          (HasAny(folded, kWorkPhrases) || HasAnyWord(folded, kWorkWords)));
}

}  // namespace

JevGateDecision ClassifyJevRequest(std::string_view text) {
  const std::string folded = FoldAscii(text);
  const bool code = HasCode(folded);
  if (IsHard(folded, code)) {
    return {ReasoningEffort::kXHigh, true};
  }
  if (IsTrivial(folded)) {
    return {ReasoningEffort::kLow, false};
  }
  return {ReasoningEffort::kMedium, true};
}

namespace {

constexpr std::size_t kAgentShellOutputMax = 4000;
constexpr std::size_t kAgentReadHead = 160;
constexpr std::size_t kAgentOtherHead = 400;

constexpr std::array<std::string_view, 17> kAgentFailWords{{
    "error",
    "failed",
    "failure",
    "traceback",
    "exception",
    "fatal",
    "panic",
    "segfault",
    "assertion",
    "no such file",
    "not found",
    "permission denied",
    "undefined reference",
    "timed out",
    "timeout",
    "command not found",
    "exit code",
}};

constexpr std::array<std::string_view, 7> kAgentReadTools{{
    "read",
    "filesystem_read_multiple_files",
    "glob",
    "grep",
    "list",
    "ls",
    "filesystem_directory_tree",
}};

constexpr std::array<std::string_view, 7> kAgentLightTools{{
    "todowrite",
    "todoread",
    "edit",
    "write",
    "memory_store_context",
    "memory_multi_search_context",
    "get_goal",
}};

// First words of read-only shell inspection commands.
constexpr std::array<std::string_view, 29> kAgentInspect{{
    "ls",    "cat",   "head", "tail", "wc",   "stat", "pwd",   "echo",
    "find",  "grep",  "rg",   "tree", "file", "nl",   "du",    "df",
    "which", "sleep", "test", "[",    "date", "ps",   "pgrep", "git",
    "sed",   "awk",   "sort", "uniq", "diff",
}};

constexpr std::array<std::string_view, 8> kAgentGitRead{{
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "ls-files",
    "rev-parse",
    "remote",
}};

bool AgentWordChar(char c) noexcept {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

std::string AgentFold(std::string_view text) {
  std::string out(text.size(), '\0');
  std::transform(text.begin(), text.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

// True when `folded` holds a failure word on word boundaries. "exit code" only
// counts when a non-zero digit follows.
bool AgentHasFailure(std::string_view folded) {
  for (const auto word : kAgentFailWords) {
    for (std::size_t pos = folded.find(word); pos != std::string_view::npos;
         pos = folded.find(word, pos + 1)) {
      const bool left = pos == 0 || !AgentWordChar(folded[pos - 1]);
      const std::size_t next = pos + word.size();
      const bool right = next >= folded.size() || !AgentWordChar(folded[next]);
      if (!left || !right)
        continue;
      if (word != "exit code")
        return true;
      if (next + 1 < folded.size() && folded[next] == ' ' &&
          folded[next + 1] >= '1' && folded[next + 1] <= '9') {
        return true;
      }
    }
  }
  return false;
}

std::string_view AgentHead(std::string_view text, std::size_t limit) {
  return text.substr(0, std::min(text.size(), limit));
}

std::string_view FirstWord(std::string_view text, std::size_t* end) {
  std::size_t i = 0;
  while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])))
    ++i;
  const std::size_t start = i;
  if (i < text.size() && text[i] == '[') {
    *end = i + 1;
    return text.substr(i, 1);
  }
  while (i < text.size() && AgentWordChar(text[i]))
    ++i;
  *end = i;
  return text.substr(start, i - start);
}

bool AgentInList(std::string_view word,
                 std::span<const std::string_view> list) {
  return std::find(list.begin(), list.end(), word) != list.end();
}

// A shell command that only looks at the machine: optional `cd X &&` prefix,
// then a read-only first word. Pipes and redirects are not inspected further;
// the output failure scan covers a failing pipeline.
bool AgentInspectCommand(std::string_view command) {
  std::size_t end = 0;
  std::string_view word = FirstWord(command, &end);
  if (word == "cd") {
    std::size_t sep = command.find("&&", end);
    std::size_t sep_len = 2;
    const std::size_t semi = command.find(';', end);
    if (sep == std::string_view::npos ||
        (semi != std::string_view::npos && semi < sep)) {
      sep = semi;
      sep_len = 1;
    }
    if (sep == std::string_view::npos)
      return false;
    command.remove_prefix(sep + sep_len);
    word = FirstWord(command, &end);
  }
  if (!AgentInList(word, kAgentInspect))
    return false;
  if (word == "git") {
    std::string_view rest = command.substr(end);
    std::size_t sub_end = 0;
    const std::string_view sub = FirstWord(rest, &sub_end);
    return AgentInList(sub, kAgentGitRead);
  }
  if (word == "sed") {
    return command.substr(end).find("-n") != std::string_view::npos;
  }
  if (word == "awk" || word == "sort" || word == "uniq" || word == "diff")
    return true;
  return true;
}

bool AgentRoutine(const JevToolResult& result) {
  if (result.tool == "bash") {
    if (!AgentInspectCommand(result.command) ||
        result.output.size() >= kAgentShellOutputMax) {
      return false;
    }
    return !AgentHasFailure(AgentFold(result.output));
  }
  if (AgentInList(result.tool, kAgentReadTools)) {
    return !AgentHasFailure(
        AgentFold(AgentHead(result.output, kAgentReadHead)));
  }
  if (AgentInList(result.tool, kAgentLightTools)) {
    return !AgentHasFailure(
        AgentFold(AgentHead(result.output, kAgentOtherHead)));
  }
  return false;
}

bool AgentTurnRoutine(const std::vector<JevToolResult>& turn) {
  return !turn.empty() && std::all_of(turn.begin(), turn.end(), AgentRoutine);
}

}  // namespace

JevAgentDecision ClassifyJevAgentTurn(
    const std::vector<std::vector<JevToolResult>>& turns) {
  if (turns.empty())
    return {false, "no-tool-result"};
  if (!AgentTurnRoutine(turns.back()))
    return {false, "tool-result-not-routine"};
  std::size_t run = 0;
  for (std::size_t i = turns.size() - 1; i-- > 0;) {
    if (!AgentTurnRoutine(turns[i]))
      break;
    ++run;
  }
  if (run >= kJevAgentMaxRun)
    return {false, "run-limit"};
  return {true, "routine-tool-result"};
}

}  // namespace gufo::models::qwen38_flash_next
