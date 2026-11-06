#include "src/models/qwen38_flash_next/jev_finish.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstring>

namespace gufo::models::qwen38_flash_next {
namespace {

// A prose paragraph shorter than this carries too little to judge by its end.
constexpr std::size_t kMinProseChars = 40;
// The unpunctuated-ending rule is weak, so it also wants a long paragraph.
constexpr std::size_t kMinWeakProseChars = 80;
// A task this long deserves more than a few characters back.
constexpr std::size_t kLongTaskChars = 600;
constexpr std::size_t kTinyAnswerChars = 20;
constexpr std::size_t kMinPlanItems = 2;

constexpr float kScoreLength = 1.0F;
constexpr float kScoreEmpty = 0.9F;
constexpr float kScoreFence = 0.95F;
constexpr float kScoreOpenEnd = 0.85F;
constexpr float kScoreStopword = 0.85F;
constexpr float kScoreAnnounce = 0.8F;
constexpr float kScorePlan = 0.7F;
constexpr float kScoreBrace = 0.65F;
constexpr float kScoreTiny = 0.45F;
constexpr float kScoreBare = 0.62F;
constexpr float kScoreBareItem = 0.45F;
// A list item this long that stops without punctuation is a cut sentence.
constexpr std::size_t kMinItemWords = 10;

// Words that cannot end a sentence: the text after them is missing.
constexpr std::array<std::string_view, 40> kDanglingWords{{
    "the", "a",    "an",    "and",   "or",      "but",    "of",   "to",
    "in",  "for",  "with",  "that",  "which",   "is",     "are",  "as",
    "by",  "on",   "at",    "from",  "because", "if",     "then", "so",
    "not", "this", "these", "those", "its",     "their",  "your", "my",
    "our", "will", "would", "can",   "could",   "should", "has",  "have",
}};

// Openers of a sentence that promises an action the model never takes. The
// text after "let me" is checked separately so "let me know" stays finished.
constexpr std::array<std::string_view, 17> kAnnouncements{{
    "i'll now ",
    "i will now ",
    "now i'll ",
    "now i will ",
    "next, i'll ",
    "next i'll ",
    "next, i will ",
    "next i will ",
    "now let me ",
    "let's now ",
    "i'm going to ",
    "i am going to ",
    "first, i'll ",
    "first i'll ",
    "i'll start ",
    "i'll begin ",
    "i'll go ahead ",
}};

constexpr std::array<std::string_view, 6> kLetMeBenign{{
    "know",
    "help",
    "be ",
    "say",
    "clarify",
    "just ",
}};

// A line addressed to the reader hands the turn over on purpose, even when it
// ends in a colon or promises an action ("refresh the page and tell me:").
constexpr std::array<std::string_view, 14> kHandoffWords{{
    "you",
    "your",
    "you're",
    "you've",
    "you'll",
    "please",
    "tell",
    "reply",
    "paste",
    "share",
    "send",
    "confirm",
    "provide",
    "try",
}};

constexpr std::array<std::string_view, 8> kPlanHeaders{{
    "steps:",
    "plan:",
    "the plan:",
    "here's the plan:",
    "here is the plan:",
    "my plan:",
    "approach:",
    "todo:",
}};

constexpr std::array<std::string_view, 12> kCompletionWords{{
    "done",
    "completed",
    "fixed",
    "implemented",
    "created",
    "updated",
    "added",
    "removed",
    "passes",
    "passed",
    "resolved",
    "finished",
}};

std::string Lower(std::string_view text) {
  std::string out(text);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string Upper(std::string_view text) {
  std::string out(text);
  for (auto& c : out) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string_view TrimRight(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.remove_suffix(1);
  }
  return text;
}

std::string_view TrimLeft(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.remove_prefix(1);
  }
  return text;
}

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

bool IsFenceLine(std::string_view line) {
  return StartsWith(TrimLeft(line), "```");
}

std::vector<std::string_view> SplitLines(std::string_view text) {
  std::vector<std::string_view> lines;
  while (!text.empty()) {
    const std::size_t nl = text.find('\n');
    lines.push_back(text.substr(0, nl));
    if (nl == std::string_view::npos) {
      break;
    }
    text.remove_prefix(nl + 1);
  }
  return lines;
}

std::size_t CountFences(const std::vector<std::string_view>& lines) {
  return static_cast<std::size_t>(
      std::count_if(lines.begin(), lines.end(),
                    [](std::string_view line) { return IsFenceLine(line); }));
}

// Strips a markdown list marker or heading prefix; sets `listy` when one was
// present because list items and headings legitimately lack final punctuation.
std::string_view StripMarker(std::string_view line, bool* listy) {
  line = TrimLeft(line);
  *listy = false;
  if (line.size() > 1 && (line[0] == '-' || line[0] == '*' || line[0] == '+') &&
      line[1] == ' ') {
    *listy = true;
    return TrimLeft(line.substr(2));
  }
  if (!line.empty() && std::isdigit(static_cast<unsigned char>(line[0])) != 0) {
    std::size_t i = 0;
    while (i < line.size() &&
           std::isdigit(static_cast<unsigned char>(line[i])) != 0) {
      ++i;
    }
    if (i + 1 < line.size() && (line[i] == '.' || line[i] == ')') &&
        line[i + 1] == ' ') {
      *listy = true;
      return TrimLeft(line.substr(i + 2));
    }
  }
  if (!line.empty() && (line[0] == '#' || line[0] == '|' || line[0] == '>')) {
    *listy = true;
  }
  return line;
}

bool IsTerminal(char c) {
  if (static_cast<unsigned char>(c) >= 0x80) {
    return true;  // CJK punctuation, emoji, quotes: cannot tell, stay silent.
  }
  return std::strchr(".!?;:)]}\"'`*_~|>", c) != nullptr;
}

std::string_view LastWord(std::string_view text) {
  text = TrimRight(text);
  std::size_t i = text.size();
  while (i > 0 && (std::isalpha(static_cast<unsigned char>(text[i - 1])) != 0 ||
                   text[i - 1] == '\'')) {
    --i;
  }
  return text.substr(i);
}

std::size_t CountWords(std::string_view text) {
  std::size_t count = 0;
  bool in_word = false;
  for (const char c : text) {
    const bool word = std::isspace(static_cast<unsigned char>(c)) == 0;
    count += word && !in_word;
    in_word = word;
  }
  return count;
}

// A last token with digits, slashes or dots is a path, URL, number or name, so
// the missing closing punctuation is style, not truncation.
bool LooksLikeToken(std::string_view body) {
  std::size_t i = body.size();
  while (i > 0 && std::isspace(static_cast<unsigned char>(body[i - 1])) == 0) {
    --i;
  }
  const std::string_view word = body.substr(i);
  return std::any_of(word.begin(), word.end(), [](char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '/' ||
           c == '.' || c == '_' || c == '-' || c == '@' || c == '#';
  });
}

bool IsDangling(std::string_view word) {
  const std::string lowered = Lower(word);
  if (word.size() > 1 && word == std::string_view(Upper(word))) {
    return false;  // "money coming IN": an acronym or label, not a cut word.
  }
  return std::any_of(kDanglingWords.begin(), kDanglingWords.end(),
                     [&](std::string_view w) { return w == lowered; });
}

// The last non-empty line outside code, and whether the response ends in a
// closed code block instead of prose.
struct Tail {
  std::string_view last_line;
  bool ends_in_code{false};
  bool empty{true};
};

Tail FindTail(const std::vector<std::string_view>& lines) {
  Tail tail;
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    if (TrimRight(*it).empty()) {
      continue;
    }
    tail.empty = false;
    tail.ends_in_code = IsFenceLine(*it);
    tail.last_line = TrimRight(*it);
    return tail;
  }
  return tail;
}

std::string LastSentence(std::string_view line) {
  const std::string lowered = Lower(TrimLeft(line));
  std::size_t start = 0;
  for (std::size_t i = 0; i + 1 < lowered.size(); ++i) {
    if ((lowered[i] == '.' || lowered[i] == '!' || lowered[i] == '?') &&
        lowered[i + 1] == ' ') {
      start = i + 2;
    }
  }
  std::string sentence = lowered.substr(start);
  // Drop markdown emphasis and list markers around the opener.
  sentence.erase(std::remove(sentence.begin(), sentence.end(), '*'),
                 sentence.end());
  bool listy = false;
  const std::string_view stripped = StripMarker(sentence, &listy);
  return std::string(stripped);
}

bool IsHandoff(std::string_view line) {
  const std::string lowered = Lower(line);
  for (const std::string_view imperative :
       {"run ", "refresh ", "open ", "reply ", "restart ", "reload "}) {
    if (StartsWith(lowered, imperative)) {
      return true;
    }
  }
  std::size_t i = 0;
  while (i < lowered.size()) {
    while (i < lowered.size() &&
           std::isalpha(static_cast<unsigned char>(lowered[i])) == 0 &&
           lowered[i] != '\'') {
      ++i;
    }
    const std::size_t start = i;
    while (i < lowered.size() &&
           (std::isalpha(static_cast<unsigned char>(lowered[i])) != 0 ||
            lowered[i] == '\'')) {
      ++i;
    }
    const std::string_view word =
        std::string_view(lowered).substr(start, i - start);
    if (word.empty()) {
      continue;
    }
    if (std::any_of(kHandoffWords.begin(), kHandoffWords.end(),
                    [&](std::string_view h) { return h == word; })) {
      return true;
    }
  }
  return false;
}

bool Announces(const std::string& sentence) {
  for (const auto opener : kAnnouncements) {
    if (StartsWith(sentence, opener)) {
      return true;
    }
  }
  if (StartsWith(sentence, "let me ")) {
    const std::string_view rest = std::string_view(sentence).substr(7);
    return std::none_of(
        kLetMeBenign.begin(), kLetMeBenign.end(),
        [&](std::string_view b) { return StartsWith(rest, b); });
  }
  return false;
}

// Net open brackets in the last closed code block: a block that closes while
// still nested was cut by the model, not by the fence.
int OpenDepthOfLastBlock(const std::vector<std::string_view>& lines) {
  std::size_t close = lines.size();
  while (close > 0 && !IsFenceLine(lines[close - 1])) {
    --close;
  }
  if (close == 0) {
    return 0;
  }
  std::size_t open = close - 1;
  while (open > 0 && !IsFenceLine(lines[open - 1])) {
    --open;
  }
  if (open == 0) {
    return 0;
  }
  int depth = 0;
  for (std::size_t i = open; i + 1 < close; ++i) {
    for (const char c : lines[i]) {
      depth += (c == '{' || c == '(' || c == '[') -
               (c == '}' || c == ')' || c == ']');
    }
  }
  return depth;
}

bool HasPlanWithoutExecution(const std::vector<std::string_view>& lines,
                             std::string_view response) {
  const std::string lowered = Lower(response);
  for (const auto done : kCompletionWords) {
    if (lowered.find(done) != std::string::npos) {
      return false;
    }
  }
  for (std::size_t i = 0; i < lines.size(); ++i) {
    const std::string header = Lower(TrimRight(TrimLeft(lines[i])));
    std::string bare = header;
    bare.erase(std::remove(bare.begin(), bare.end(), '*'), bare.end());
    const bool is_header =
        std::any_of(kPlanHeaders.begin(), kPlanHeaders.end(),
                    [&](std::string_view h) { return StartsWith(bare, h); });
    if (!is_header) {
      continue;
    }
    std::size_t items = 0;
    bool first_person = false;
    for (std::size_t j = i + 1; j < lines.size(); ++j) {
      bool listy = false;
      const std::string item = Lower(StripMarker(lines[j], &listy));
      if (TrimRight(lines[j]).empty()) {
        continue;
      }
      if (!listy) {
        return false;  // Prose follows the list, so the plan was acted on.
      }
      ++items;
      first_person = first_person || StartsWith(item, "i will") ||
                     StartsWith(item, "i'll") || StartsWith(item, "we will");
    }
    return items >= kMinPlanItems && first_person;
  }
  return false;
}

void Add(std::vector<JevFinishHit>& hits, const char* rule, const char* reason,
         float score) {
  hits.push_back({rule, reason, score});
}

}  // namespace

std::vector<JevFinishHit> EvaluateFinishRules(std::string_view task,
                                              std::string_view response,
                                              bool had_tool_call,
                                              std::string_view finish_reason) {
  std::vector<JevFinishHit> hits;
  if (had_tool_call) {
    return hits;
  }
  if (finish_reason == "length") {
    Add(hits, "length", "stopped at the token limit", kScoreLength);
  }
  response = TrimRight(response);
  if (TrimLeft(response).empty()) {
    if (finish_reason == "stop") {
      Add(hits, "empty", "empty answer", kScoreEmpty);
    }
    std::sort(hits.begin(), hits.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    return hits;
  }
  const auto lines = SplitLines(response);
  if (CountFences(lines) % 2 == 1) {
    Add(hits, "fence", "unclosed code fence", kScoreFence);
  }
  const Tail tail = FindTail(lines);
  if (tail.ends_in_code) {
    if (OpenDepthOfLastBlock(lines) >= 2) {
      Add(hits, "brace", "code block closes with unbalanced brackets",
          kScoreBrace);
    }
  } else if (!tail.empty && CountFences(lines) % 2 == 0) {
    bool listy = false;
    const std::string_view body = StripMarker(tail.last_line, &listy);
    const char last = tail.last_line.back();
    const bool handoff = IsHandoff(body);
    if (!handoff && (last == ':' || last == ',' || last == '(' || last == '[' ||
                     last == '{')) {
      Add(hits, "open_end", "last line ends in an open punctuation mark",
          kScoreOpenEnd);
    } else if (!IsTerminal(last)) {
      if (body.size() >= kMinProseChars && IsDangling(LastWord(body))) {
        Add(hits, "stopword", "last sentence ends on a dangling word",
            kScoreStopword);
      } else if (LooksLikeToken(body)) {
        // Ends on a path, number or name: stay silent.
      } else if (!listy && body.size() >= kMinWeakProseChars) {
        Add(hits, "bare", "last paragraph has no closing punctuation",
            kScoreBare);
      } else if (listy && CountWords(body) >= kMinItemWords) {
        Add(hits, "bare_item", "last list item has no closing punctuation",
            kScoreBareItem);
      }
    }
    if (!handoff && Announces(LastSentence(tail.last_line))) {
      Add(hits, "announce", "announces a next action that never ran",
          kScoreAnnounce);
    }
    if (HasPlanWithoutExecution(lines, response)) {
      Add(hits, "plan", "plan listed with no execution after it", kScorePlan);
    }
  }
  if (task.size() >= kLongTaskChars && response.size() < kTinyAnswerChars) {
    Add(hits, "tiny", "answer far shorter than the task", kScoreTiny);
  }
  std::sort(hits.begin(), hits.end(),
            [](const auto& a, const auto& b) { return a.score > b.score; });
  return hits;
}

JevFinishVerdict JudgeFinish(std::string_view task, std::string_view response,
                             bool had_tool_call,
                             std::string_view finish_reason) {
  const auto hits =
      EvaluateFinishRules(task, response, had_tool_call, finish_reason);
  JevFinishVerdict verdict;
  if (hits.empty() || hits.front().score < kJevFinishThreshold) {
    if (!hits.empty()) {
      verdict.score = hits.front().score;
    }
    return verdict;
  }
  verdict.unfinished = true;
  verdict.reason = hits.front().reason;
  verdict.score = hits.front().score;
  return verdict;
}

}  // namespace gufo::models::qwen38_flash_next
