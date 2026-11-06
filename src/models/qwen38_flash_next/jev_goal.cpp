#include "src/models/qwen38_flash_next/jev_goal.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <map>
#include <vector>

namespace gufo::models::qwen38_flash_next {
namespace {

// Accepted trigger phrases, lowercase words separated by one space. Matching
// ignores case, punctuation between words and apostrophes inside words.
constexpr std::array<std::string_view, 8> kTriggers{{
    "jev make sure its done",
    "jev make sure it is done",
    "jev make sure its finished",
    "jev make sure it is finished",
    "jev finish it",
    "jev finish this",
    "jev keep going until done",
    "jev keep going until its done",
}};

// Rules measured on 4,957 real turn ends (see jev-gate-plan.md): the ones
// whose "next user message was a continue" rate is at or below the base rate
// stay off. Flip to re-enable for calibration.
constexpr bool kRuleLastFailure = false;  // 5 of 42 fires followed by continue
constexpr bool kRuleLoopNudge = false;    // 5 of 21
constexpr bool kRuleOpenTest = false;     // 6 of 22
constexpr bool kRuleOpenPush = false;     // 0 of 1
constexpr bool kRuleOpenFile = true;
constexpr std::size_t kMaxItems = 8;
constexpr std::size_t kMaxClauseChars = 240;
constexpr std::size_t kItemChars = 80;
constexpr std::size_t kFailureWindow = 4096;
constexpr std::size_t kLoopWindow = 60;
constexpr int kLoopFailures = 3;
constexpr std::size_t kAnswerMinChars = 10;
constexpr std::size_t kClaimWindow = 400;
constexpr std::size_t kIntentWindow = 220;

bool IsAlnum(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

char Lower(char c) {
  return c >= 'A' && c <= 'Z' ? static_cast<char>(c | 0x20) : c;
}

// Length of an apostrophe at `i` (ASCII or UTF-8 curly), else 0.
std::size_t ApostropheLen(std::string_view text, std::size_t i) {
  if (i >= text.size())
    return 0;
  if (text[i] == '\'')
    return 1;
  if (i + 2 < text.size() && static_cast<unsigned char>(text[i]) == 0xE2 &&
      static_cast<unsigned char>(text[i + 1]) == 0x80 &&
      (static_cast<unsigned char>(text[i + 2]) == 0x99 ||
       static_cast<unsigned char>(text[i + 2]) == 0x98)) {
    return 3;
  }
  return 0;
}

bool IsSeparator(std::string_view text, std::size_t i) {
  return !IsAlnum(text[i]) && ApostropheLen(text, i) == 0;
}

std::string Fold(std::string_view text) {
  std::string out(text);
  for (auto& c : out)
    c = Lower(c);
  return out;
}

// End of the match of `phrase` at `pos`, or npos.
std::size_t MatchPhraseAt(std::string_view text, std::size_t pos,
                          std::string_view phrase) {
  std::size_t i = pos;
  std::size_t word_start = 0;
  bool first = true;
  while (word_start < phrase.size()) {
    std::size_t word_end = phrase.find(' ', word_start);
    if (word_end == std::string_view::npos)
      word_end = phrase.size();
    if (!first) {
      const std::size_t sep_start = i;
      while (i < text.size() && IsSeparator(text, i))
        ++i;
      if (i == sep_start)
        return std::string_view::npos;
    }
    first = false;
    for (std::size_t w = word_start; w < word_end; ++w) {
      while (const std::size_t a = ApostropheLen(text, i))
        i += a;
      if (i >= text.size() || Lower(text[i]) != phrase[w])
        return std::string_view::npos;
      ++i;
    }
    word_start = word_end + 1;
  }
  if (i < text.size() && IsAlnum(text[i]))
    return std::string_view::npos;
  return i;
}

// First trigger in `text` at or after `from`: {begin, end} or {npos, npos}.
std::pair<std::size_t, std::size_t> FindTrigger(std::string_view text,
                                                std::size_t from) {
  for (std::size_t pos = from; pos < text.size(); ++pos) {
    if ((text[pos] | 0x20) != 'j')
      continue;
    if (pos > 0 && IsAlnum(text[pos - 1]))
      continue;
    for (const auto phrase : kTriggers) {
      const std::size_t end = MatchPhraseAt(text, pos, phrase);
      if (end != std::string_view::npos)
        return {pos, end};
    }
  }
  return {std::string_view::npos, std::string_view::npos};
}

bool IsTrailingSeparator(char c) {
  return c == ' ' || c == '\t' || c == ',' || c == '.' || c == ';' ||
         c == ':' || c == '!' || c == '?';
}

std::string StripOnce(std::string_view text, bool* changed) {
  std::string out;
  std::size_t from = 0;
  *changed = false;
  while (true) {
    const auto [begin, end] = FindTrigger(text, from);
    if (begin == std::string_view::npos)
      break;
    *changed = true;
    out.append(text.substr(from, begin - from));
    std::size_t stop = end;
    while (stop < text.size() && IsTrailingSeparator(text[stop]))
      ++stop;
    from = stop;
  }
  out.append(text.substr(from));
  return out;
}

std::string Trim(std::string_view text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end &&
         (std::isspace(static_cast<unsigned char>(text[begin])) ||
          text[begin] == ',' || text[begin] == ';')) {
    ++begin;
  }
  while (end > begin &&
         (std::isspace(static_cast<unsigned char>(text[end - 1])) ||
          text[end - 1] == ',' || text[end - 1] == ';')) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

// ---------------------------------------------------------------- evidence

bool Contains(std::string_view folded, std::string_view needle) {
  return folded.find(needle) != std::string_view::npos;
}

bool ContainsWord(std::string_view folded, std::string_view word) {
  for (std::size_t pos = folded.find(word); pos != std::string_view::npos;
       pos = folded.find(word, pos + 1)) {
    const bool left = pos == 0 || !IsAlnum(folded[pos - 1]);
    const std::size_t next = pos + word.size();
    const bool right = next >= folded.size() || !IsAlnum(folded[next]);
    if (left && right)
      return true;
  }
  return false;
}

// True when the marker at `pos` is a benign count ("0 failed", "no errors",
// "failures: 0").
bool BenignCount(std::string_view folded, std::size_t pos,
                 std::size_t marker_len) {
  std::size_t b = pos;
  while (b > 0 && folded[b - 1] == ' ')
    --b;
  if (b >= 1 && folded[b - 1] == '0' && (b == 1 || !IsAlnum(folded[b - 2])))
    return true;
  if (b >= 2 && folded.substr(b - 2, 2) == "no" &&
      (b == 2 || !IsAlnum(folded[b - 3]))) {
    return true;
  }
  std::size_t a = pos + marker_len;
  while (a < folded.size() && (folded[a] == ':' || folded[a] == ' ' ||
                               folded[a] == '=' || folded[a] == 's')) {
    ++a;
  }
  return a < folded.size() && folded[a] == '0' &&
         (a + 1 >= folded.size() || !IsAlnum(folded[a + 1]));
}

bool HasMarker(std::string_view folded, std::string_view marker,
               bool check_benign) {
  for (std::size_t pos = folded.find(marker); pos != std::string_view::npos;
       pos = folded.find(marker, pos + 1)) {
    if (!check_benign || !BenignCount(folded, pos, marker.size()))
      return true;
  }
  return false;
}

constexpr std::array<std::string_view, 12> kHardFailure{{
    "traceback (most recent",
    "error:",
    "fatal:",
    "assertionerror",
    "panic:",
    "undefined reference",
    "command not found",
    "no such file or directory",
    "syntax error",
    "npm err",
    "make: ***",
    "segmentation fault",
}};

constexpr std::array<std::string_view, 3> kCountFailure{{
    "failed",
    "failure",
    "errors",
}};

std::string_view TailOf(std::string_view text, std::size_t limit) {
  return text.size() > limit ? text.substr(text.size() - limit) : text;
}

// Whether a command's output reads as a failure. Text only: the server never
// sees exit codes.
bool LooksFailed(std::string_view output) {
  const std::string folded = Fold(TailOf(output, kFailureWindow));
  for (const auto marker : kHardFailure) {
    if (Contains(folded, marker))
      return true;
  }
  for (const auto marker : kCountFailure) {
    if (HasMarker(folded, marker, true))
      return true;
  }
  return false;
}

std::string FirstFailureLine(std::string_view output) {
  const std::string_view tail = TailOf(output, kFailureWindow);
  std::size_t begin = 0;
  while (begin < tail.size()) {
    std::size_t end = tail.find('\n', begin);
    if (end == std::string_view::npos)
      end = tail.size();
    const std::string_view line = tail.substr(begin, end - begin);
    if (LooksFailed(line)) {
      std::string out = Trim(line.substr(0, 120));
      return out;
    }
    begin = end + 1;
  }
  return {};
}

// Strips a leading `cd X &&` / `cd X;` so the real command is classified.
std::string_view StripCd(std::string_view command) {
  while (true) {
    std::size_t i = 0;
    while (i < command.size() &&
           std::isspace(static_cast<unsigned char>(command[i])))
      ++i;
    if (command.substr(i, 3) != "cd ")
      return command.substr(i);
    const std::size_t amp = command.find("&&", i);
    const std::size_t semi = command.find(';', i);
    const std::size_t sep = std::min(amp, semi);
    if (sep == std::string_view::npos)
      return command.substr(i);
    command.remove_prefix(sep + (sep == amp ? 2 : 1));
  }
}

enum class CommandKind { kOther, kTest, kBuild, kCommit, kPush };

CommandKind ClassifyCommand(std::string_view command) {
  const std::string folded = Fold(StripCd(command));
  if (Contains(folded, "git commit"))
    return CommandKind::kCommit;
  if (Contains(folded, "git push"))
    return CommandKind::kPush;
  if (ContainsWord(folded, "pytest") || ContainsWord(folded, "ctest") ||
      ContainsWord(folded, "jest") || ContainsWord(folded, "vitest") ||
      ContainsWord(folded, "mocha") || ContainsWord(folded, "rspec") ||
      Contains(folded, "unittest") || Contains(folded, "go test") ||
      Contains(folded, "cargo test") || Contains(folded, "npm test") ||
      Contains(folded, "npm run test") || Contains(folded, "pnpm test") ||
      Contains(folded, "make test") || Contains(folded, "make check") ||
      Contains(folded, "gradle test") || Contains(folded, "mvn test") ||
      ContainsWord(folded, "test") || ContainsWord(folded, "tests")) {
    return CommandKind::kTest;
  }
  if (Contains(folded, "build") || Contains(folded, "cmake") ||
      ContainsWord(folded, "make") || ContainsWord(folded, "ninja") ||
      ContainsWord(folded, "tsc") || Contains(folded, "cargo check") ||
      Contains(folded, "compile") || ContainsWord(folded, "gcc") ||
      ContainsWord(folded, "g++") || ContainsWord(folded, "clang") ||
      ContainsWord(folded, "mvn") || ContainsWord(folded, "gradle")) {
    return CommandKind::kBuild;
  }
  return CommandKind::kOther;
}

bool IsShell(std::string_view tool) {
  return tool == "bash" || tool == "shell";
}

// A shell command that creates or changes files (redirect, in-place edit,
// copy/move, patch, installs). Counts as write evidence like an edit tool.
bool ShellWrites(std::string_view command) {
  const std::string folded = Fold(StripCd(command));
  for (std::size_t i = 0; i < folded.size(); ++i) {
    if (folded[i] != '>')
      continue;
    const bool fd_dup = i + 1 < folded.size() && folded[i + 1] == '&';
    const bool devnull = folded.compare(i, 10, ">/dev/null") == 0 ||
                         folded.compare(i, 11, "> /dev/null") == 0;
    if (!fd_dup && !devnull && !(i > 0 && folded[i - 1] == '-'))
      return true;
  }
  return Contains(folded, "sed -i") || Contains(folded, "tee ") ||
         ContainsWord(folded, "mv") || ContainsWord(folded, "cp") ||
         Contains(folded, "git apply") || ContainsWord(folded, "patch") ||
         ContainsWord(folded, "mkdir") || ContainsWord(folded, "touch") ||
         Contains(folded, "pip install") || Contains(folded, "npm install") ||
         Contains(folded, "cargo add") || Contains(folded, "git add") ||
         Contains(folded, "git checkout") || Contains(folded, "git mv") ||
         Contains(folded, "chmod");
}

// Delegating to a sub-agent or teammate is work the tracker cannot see into.
bool IsDelegationTool(std::string_view tool) {
  return tool == "task" || tool.starts_with("team_");
}

bool IsWriteTool(std::string_view tool) {
  return tool == "edit" || tool == "write" || tool == "multiedit" ||
         tool == "patch" || tool == "apply_patch" ||
         tool == "filesystem_write_file" || tool == "filesystem_edit_file";
}

struct CallRecord {
  std::string_view name;
  std::string_view command;
  std::string_view path;
  std::string_view id;
  std::string_view result;
  bool has_result{false};
  mutable int kind_cache{-1};
  mutable int wrote_cache{-1};

  CommandKind kind() const {
    if (kind_cache < 0) {
      kind_cache = static_cast<int>(IsShell(name) ? ClassifyCommand(command)
                                                  : CommandKind::kOther);
    }
    return static_cast<CommandKind>(kind_cache);
  }

  bool wrote() const {
    if (wrote_cache < 0) {
      wrote_cache = IsWriteTool(name) || IsDelegationTool(name) ||
                    (IsShell(name) && ShellWrites(command));
    }
    return wrote_cache != 0;
  }
};

std::vector<CallRecord> CollectCalls(std::span<const JevGoalMessage> history,
                                     std::size_t from) {
  constexpr std::size_t kIdLookback = 64;
  std::vector<CallRecord> records;
  for (std::size_t i = from; i < history.size(); ++i) {
    const auto& message = history[i];
    if (message.role == JevGoalMessage::Role::kAssistant) {
      for (const auto& call : message.calls) {
        CallRecord record;
        record.name = call.name;
        record.command = call.command;
        record.path = call.path;
        record.id = call.id;
        records.push_back(record);
      }
    } else if (message.role == JevGoalMessage::Role::kTool) {
      const std::size_t stop =
          records.size() > kIdLookback ? records.size() - kIdLookback : 0;
      for (std::size_t j = records.size(); j-- > stop;) {
        if (records[j].id == message.tool_call_id && !records[j].has_result) {
          records[j].result = message.text;
          records[j].has_result = true;
          break;
        }
      }
    }
  }
  return records;
}

// ------------------------------------------------------------------- goals

enum class ItemKind { kChange, kFile, kTest, kBuild, kCommit, kPush, kAnswer };

struct Item {
  ItemKind kind{ItemKind::kChange};
  std::string text;
  std::string target;  // file basename for kFile
  const char* rule{"open-change"};
};

constexpr std::array<std::string_view, 17> kChangeVerbs{{
    "fix",
    "add",
    "implement",
    "create",
    "write",
    "change",
    "update",
    "modify",
    "refactor",
    "rename",
    "remove",
    "delete",
    "patch",
    "rewrite",
    "make",
    "build",
    "wire",
}};

constexpr std::array<std::string_view, 24> kAnswerVerbs{{
    "explain",     "describe", "tell",     "summarize", "summarise", "why",
    "what",        "how",      "list",     "show",      "check",     "review",
    "investigate", "find",     "look",     "read",      "analyze",   "analyse",
    "compare",     "debug",    "diagnose", "report",    "which",     "where",
}};

constexpr std::array<std::string_view, 24> kFileExtensions{{
    "cpp", "hpp", "h",  "c",    "cc",   "py",    "js",  "ts",
    "tsx", "jsx", "md", "json", "toml", "yaml",  "yml", "sh",
    "txt", "rs",  "go", "css",  "html", "cmake", "hip", "java",
}};

std::string ShortText(std::string_view text) {
  std::string out = Trim(text.substr(0, kItemChars));
  for (auto& c : out) {
    if (c == '\n')
      c = ' ';
  }
  return out;
}

bool StartsWithWordOf(std::string_view folded,
                      std::span<const std::string_view> words) {
  std::size_t i = 0;
  while (i < folded.size() && !IsAlnum(folded[i]))
    ++i;
  std::size_t j = i;
  while (j < folded.size() && IsAlnum(folded[j]))
    ++j;
  const std::string_view first = folded.substr(i, j - i);
  return std::find(words.begin(), words.end(), first) != words.end();
}

bool HasWordOf(std::string_view folded,
               std::span<const std::string_view> words) {
  return std::any_of(words.begin(), words.end(), [&](std::string_view w) {
    return ContainsWord(folded, w);
  });
}

std::string PathToken(std::string_view clause) {
  std::size_t i = 0;
  while (i < clause.size()) {
    while (i < clause.size() &&
           !(IsAlnum(clause[i]) || clause[i] == '/' || clause[i] == '_' ||
             clause[i] == '.' || clause[i] == '-')) {
      ++i;
    }
    std::size_t j = i;
    while (j < clause.size() &&
           (IsAlnum(clause[j]) || clause[j] == '/' || clause[j] == '_' ||
            clause[j] == '.' || clause[j] == '-')) {
      ++j;
    }
    std::string_view token = clause.substr(i, j - i);
    while (!token.empty() && token.back() == '.')
      token.remove_suffix(1);
    const std::size_t dot = token.rfind('.');
    if (dot != std::string_view::npos && dot + 1 < token.size() && dot > 0) {
      const std::string ext = Fold(token.substr(dot + 1));
      if (std::find(kFileExtensions.begin(), kFileExtensions.end(), ext) !=
          kFileExtensions.end()) {
        return std::string(token);
      }
    }
    i = j + 1;
  }
  return {};
}

std::string Basename(std::string_view path) {
  const std::size_t slash = path.rfind('/');
  return std::string(slash == std::string_view::npos ? path
                                                     : path.substr(slash + 1));
}

void SplitClauses(std::string_view text, std::vector<std::string>* out) {
  std::string current;
  const auto flush = [&] {
    std::string trimmed = Trim(current);
    if (trimmed.size() >= 3)
      out->push_back(std::move(trimmed));
    current.clear();
  };
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    const bool sentence_end =
        (c == '.' || c == '!' || c == '?') &&
        (i + 1 >= text.size() ||
         std::isspace(static_cast<unsigned char>(text[i + 1])));
    if (c == '\n' || c == ';' || sentence_end) {
      if (c == '?')
        current.push_back(c);
      flush();
      continue;
    }
    current.push_back(c);
  }
  flush();
}

// The verb opens the clause or follows a connector ("... and commit",
// "then push", "remember to commit"), so a noun use does not count.
bool Imperative(std::string_view folded, std::string_view verb) {
  constexpr std::array<std::string_view, 8> kBefore{{
      "and ",
      "then ",
      "also ",
      "to ",
      ", ",
      "please ",
      "finally ",
      "&& ",
  }};
  if (folded.substr(0, verb.size()) == verb)
    return true;
  for (const auto before : kBefore) {
    std::string needle(before);
    needle += verb;
    if (folded.find(needle) != std::string_view::npos)
      return true;
  }
  return false;
}

std::vector<Item> ExtractItems(std::string_view goal) {
  // Pasted logs can run to hundreds of KB; the ask is at either end.
  constexpr std::size_t kGoalEnds = 1500;
  std::string bounded;
  if (goal.size() > 2 * kGoalEnds) {
    bounded.assign(goal.substr(0, kGoalEnds));
    bounded.push_back('\n');
    bounded.append(goal.substr(goal.size() - kGoalEnds));
    goal = bounded;
  }
  std::vector<std::string> clauses;
  SplitClauses(goal, &clauses);
  std::vector<Item> items;
  bool test = false;
  bool build = false;
  bool commit = false;
  bool push = false;
  bool answer = false;
  for (const auto& clause : clauses) {
    if (items.size() >= kMaxItems)
      break;
    if (clause.size() > kMaxClauseChars)
      continue;  // pasted logs and code, not an instruction
    const std::string folded = Fold(clause);
    const bool negated_git =
        Contains(folded, "no git") || Contains(folded, "not push") ||
        Contains(folded, "no push") || Contains(folded, "dont push") ||
        Contains(folded, "don't push") || Contains(folded, "not commit") ||
        Contains(folded, "no commit") || Contains(folded, "dont commit") ||
        Contains(folded, "don't commit") ||
        Contains(folded, "without commit") || Contains(folded, "without push");
    const bool wants_test =
        Contains(folded, "test") &&
        (ContainsWord(folded, "run") || ContainsWord(folded, "rerun") ||
         ContainsWord(folded, "pass") || ContainsWord(folded, "passes") ||
         ContainsWord(folded, "passing") || ContainsWord(folded, "green") ||
         ContainsWord(folded, "verify") || ContainsWord(folded, "ensure") ||
         Contains(folded, "make sure") || folded.starts_with("test"));
    const bool wants_build =
        (ContainsWord(folded, "build") || ContainsWord(folded, "builds") ||
         ContainsWord(folded, "compile") || ContainsWord(folded, "compiles")) &&
        (ContainsWord(folded, "run") || ContainsWord(folded, "ensure") ||
         ContainsWord(folded, "verify") || Contains(folded, "make sure") ||
         ContainsWord(folded, "pass") || ContainsWord(folded, "succeed") ||
         ContainsWord(folded, "succeeds") || ContainsWord(folded, "works") ||
         ContainsWord(folded, "clean") || folded.starts_with("build") ||
         folded.starts_with("compile"));
    const bool wants_commit = !negated_git && Imperative(folded, "commit");
    const bool wants_push = !negated_git && Imperative(folded, "push");
    const bool change =
        StartsWithWordOf(folded, kChangeVerbs) ||
        (ContainsWord(folded, "please") && HasWordOf(folded, kChangeVerbs) &&
         !wants_test && !wants_build);
    const bool reads =
        StartsWithWordOf(folded, kAnswerVerbs) || clause.back() == '?';
    const std::string path = PathToken(clause);
    if (change && !path.empty()) {
      items.push_back(
          {ItemKind::kFile, ShortText(clause), Basename(path), "open-file"});
    } else if (change && !wants_test && !wants_build && !wants_commit &&
               !wants_push) {
      items.push_back(
          {ItemKind::kChange, ShortText(clause), {}, "open-change"});
    } else if (reads && !change && !answer) {
      answer = true;
      items.push_back(
          {ItemKind::kAnswer, ShortText(clause), {}, "open-answer"});
    }
    if (wants_test && !test) {
      test = true;
      items.push_back({ItemKind::kTest, "run the tests", {}, "open-test"});
    }
    if (wants_build && !build) {
      build = true;
      items.push_back({ItemKind::kBuild, "build", {}, "open-build"});
    }
    if (wants_commit && !commit) {
      commit = true;
      items.push_back({ItemKind::kCommit, "commit", {}, "open-commit"});
    }
    if (wants_push && !push) {
      push = true;
      items.push_back({ItemKind::kPush, "push", {}, "open-push"});
    }
  }
  return items;
}

bool CommandSucceeded(const CallRecord& record) {
  return record.has_result && !LooksFailed(record.result);
}

bool ItemClosed(const Item& item, const std::vector<CallRecord>& calls,
                std::string_view final_text) {
  switch (item.kind) {
    case ItemKind::kChange:
      return std::any_of(calls.begin(), calls.end(),
                         [](const CallRecord& c) { return c.wrote(); });
    case ItemKind::kFile:
      return std::any_of(calls.begin(), calls.end(), [&](const CallRecord& c) {
        return (IsWriteTool(c.name) &&
                c.path.find(item.target) != std::string_view::npos) ||
               (c.wrote() && !IsWriteTool(c.name) &&
                c.command.find(item.target) != std::string_view::npos) ||
               IsDelegationTool(c.name);
      });
    case ItemKind::kTest:
    case ItemKind::kBuild: {
      const CommandKind want = item.kind == ItemKind::kTest
                                   ? CommandKind::kTest
                                   : CommandKind::kBuild;
      for (auto it = calls.rbegin(); it != calls.rend(); ++it) {
        if (it->kind() == want)
          return CommandSucceeded(*it);
      }
      return false;
    }
    case ItemKind::kCommit:
    case ItemKind::kPush: {
      const CommandKind want = item.kind == ItemKind::kCommit
                                   ? CommandKind::kCommit
                                   : CommandKind::kPush;
      for (auto it = calls.rbegin(); it != calls.rend(); ++it) {
        if (it->kind() != want || !it->has_result)
          continue;
        const std::string folded = Fold(TailOf(it->result, kFailureWindow));
        if (!LooksFailed(it->result) && !Contains(folded, "nothing to commit"))
          return true;
      }
      return false;
    }
    case ItemKind::kAnswer:
      return final_text.size() >= kAnswerMinChars;
  }
  return true;
}

constexpr std::array<std::string_view, 13> kIntent{{
    "i'll next",
    "i will next",
    "let me now",
    "next, i will",
    "next i will",
    "i'll now",
    "i will now",
    "now i will",
    "now i'll",
    "next, i'll",
    "next i'll",
    "i'm going to now",
    "let me start by",
}};

constexpr std::array<std::string_view, 17> kDoneWords{{
    "done",
    "complete",
    "completed",
    "finished",
    "implemented",
    "fixed",
    "all set",
    "ready",
    "working",
    "works",
    "passes",
    "passing",
    "successfully",
    "resolved",
    "created",
    "added",
    "updated",
}};

constexpr std::array<std::string_view, 12> kAcknowledgesProblem{{
    "fail",
    "error",
    "can't",
    "cannot",
    "unable",
    "blocked",
    "couldn't",
    "could not",
    "not working",
    "doesn't",
    "does not",
    "still",
}};

std::string FoldApostrophes(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    if (const std::size_t a = ApostropheLen(text, i)) {
      out.push_back('\'');
      i += a;
      continue;
    }
    out.push_back(Lower(text[i]));
    ++i;
  }
  return out;
}

}  // namespace

bool HasJevGoalTrigger(std::string_view text) {
  return FindTrigger(text, 0).first != std::string_view::npos;
}

std::string StripJevGoalTrigger(std::string_view text) {
  if (FindTrigger(text, 0).first == std::string_view::npos)
    return std::string(text);
  std::string current(text);
  for (int round = 0; round < 4; ++round) {
    bool changed = false;
    std::string next = StripOnce(current, &changed);
    current = std::move(next);
    if (!changed)
      break;
  }
  std::string trimmed = Trim(current);
  return trimmed.empty() ? "Continue." : trimmed;
}

JevGoalSpec DetectJevGoal(std::span<const JevGoalMessage> messages) {
  JevGoalSpec spec;
  std::size_t trigger = messages.size();
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (messages[i].role == JevGoalMessage::Role::kUser &&
        HasJevGoalTrigger(messages[i].text)) {
      trigger = i;
    }
  }
  if (trigger == messages.size())
    return spec;
  spec.active = true;
  spec.index = trigger;
  for (std::size_t i = trigger + 1; i-- > 0;) {
    if (messages[i].role != JevGoalMessage::Role::kUser)
      continue;
    std::string stripped = StripJevGoalTrigger(messages[i].text);
    if (stripped != "Continue.") {
      spec.text = std::move(stripped);
      break;
    }
  }
  return spec;
}

JevGoalDecision EvaluateJevGoal(const JevGoalSpec& spec,
                                std::span<const JevGoalMessage> history,
                                std::string_view final_text, int continues) {
  JevGoalDecision decision;
  if (!spec.active)
    return decision;
  decision.action = JevGoalAction::kDone;
  const auto calls = CollectCalls(history, spec.index);
  std::vector<Item> items = ExtractItems(spec.text);
  decision.items = items.size();

  // Loop: the same call failing repeatedly in the recent window.
  {
    std::map<std::string_view, int> seen;
    const std::size_t begin =
        calls.size() > kLoopWindow ? calls.size() - kLoopWindow : 0;
    for (std::size_t i = begin; i < calls.size(); ++i) {
      if (IsShell(calls[i].name))
        ++seen[calls[i].command];
    }
    std::map<std::string_view, int> failures;
    for (std::size_t i = begin; i < calls.size(); ++i) {
      const auto& call = calls[i];
      if (!IsShell(call.name) || !call.has_result ||
          seen[call.command] < kLoopFailures || !LooksFailed(call.result)) {
        continue;
      }
      if (kRuleLoopNudge && ++failures[call.command] >= kLoopFailures) {
        decision.action = JevGoalAction::kNudge;
        decision.rule = "loop";
        decision.reason = "same-command-failed";
        decision.message =
            "Same command failed 3 times; change approach. Last failure: " +
            FirstFailureLine(call.result);
        break;
      }
    }
  }

  std::vector<std::string> open;
  const char* open_rule = nullptr;
  for (const auto& item : items) {
    if ((item.kind == ItemKind::kTest && !kRuleOpenTest) ||
        (item.kind == ItemKind::kPush && !kRuleOpenPush) ||
        (item.kind == ItemKind::kFile && !kRuleOpenFile)) {
      continue;
    }
    if (!ItemClosed(item, calls, final_text)) {
      open.push_back(item.text);
      if (!open_rule)
        open_rule = item.rule;
    }
  }
  decision.open = open.size();

  // The last test/build run after the goal failed and nothing later passed.
  std::string last_failure;
  for (auto it = calls.rbegin(); it != calls.rend(); ++it) {
    if (it->kind() != CommandKind::kTest && it->kind() != CommandKind::kBuild)
      continue;
    if (it->has_result && LooksFailed(it->result))
      last_failure = FirstFailureLine(it->result);
    break;
  }

  const std::string claim_text =
      FoldApostrophes(TailOf(final_text, kClaimWindow));
  const std::string intent_text =
      FoldApostrophes(TailOf(final_text, kIntentWindow));
  const bool claims_done = std::any_of(
      kDoneWords.begin(), kDoneWords.end(), [&](std::string_view word) {
        return ContainsWord(claim_text, word) || Contains(claim_text, word);
      });
  const bool acknowledges = std::any_of(
      kAcknowledgesProblem.begin(), kAcknowledgesProblem.end(),
      [&](std::string_view word) { return Contains(claim_text, word); });
  std::string intent;
  for (const auto phrase : kIntent) {
    if (Contains(intent_text, phrase)) {
      intent = std::string(phrase);
      break;
    }
  }
  const bool empty_turn = final_text.size() < kAnswerMinChars;

  if (decision.action == JevGoalAction::kNudge) {
    // A loop only matters when the model stopped without owning the failure.
    if (continues >= kJevGoalMaxContinues || acknowledges)
      decision.action = JevGoalAction::kDone;
    return decision;
  }
  if (continues >= kJevGoalMaxContinues)
    return decision;

  // Precision first: a turn that honestly reports a blocker or hands a
  // question back stays finished. The tracker acts on false completion claims,
  // announced steps that never ran, and empty turns.
  std::string rule;
  std::string reason;
  if (!open.empty() && (claims_done || empty_turn)) {
    rule = open_rule;
    reason = claims_done ? "claims-done-without-evidence" : "empty-turn";
  } else if (kRuleLastFailure && !last_failure.empty() && claims_done &&
             !acknowledges) {
    rule = "last-failure";
    reason = "claims-done-but-last-test-or-build-failed";
  } else if (!intent.empty()) {
    rule = "intent";
    reason = "announced-next-step";
  } else {
    return decision;
  }
  decision.action = JevGoalAction::kContinue;
  decision.rule = rule;
  decision.reason = reason;
  decision.message = "Continue.";
  if (!open.empty()) {
    decision.message += " Still open:";
    for (std::size_t i = 0; i < open.size(); ++i)
      decision.message += (i ? "; " : " ") + open[i];
    decision.message += ".";
  }
  if (!last_failure.empty())
    decision.message += " Last failure: " + last_failure + ".";
  if (open.empty() && last_failure.empty())
    decision.message += " Do the step you announced instead of stopping.";
  return decision;
}

}  // namespace gufo::models::qwen38_flash_next
