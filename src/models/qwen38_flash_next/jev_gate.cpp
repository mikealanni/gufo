#include "src/models/qwen38_flash_next/jev_gate.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

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

}  // namespace gufo::models::qwen38_flash_next
