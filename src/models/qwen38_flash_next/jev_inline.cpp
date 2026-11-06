#include "src/models/qwen38_flash_next/jev_inline.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <span>
#include <string>
#include <string_view>

namespace gufo::models::qwen38_flash_next {
namespace {

// A self-posed check is one short sentence. Longer trailing text means the
// boundary was missed, so the safest action is to stay silent.
constexpr std::size_t kMaxQuestionChars = 240;
// Bounded streaming window: any sentence longer than this cannot be a usable
// yes/no check either, so it is dropped instead of buffered.
constexpr std::size_t kMaxSentenceChars = 512;

// First-person or demonstrative openers that make a question self-posed. The
// observed model traces phrase these as "Should I ..." far more often than
// "Is X the cause ...", so both shapes are covered.
constexpr std::array<std::string_view, 25> kSelfCues{{
    "should i",  "should we",  "shall i",    "is it",     "is this",
    "is that",   "is there",   "are there",  "are these", "do i",
    "do we",     "do these",   "does this",  "does it",   "does that",
    "did i",     "have i",     "has it",     "can i",     "can we",
    "will this", "would this", "could this", "am i",      "was it",
}};

// A question about the reader is the model talking to the user, not to itself.
constexpr std::array<std::string_view, 5> kReaderWords{{
    "you",
    "your",
    "yours",
    "yourself",
    "user",
}};

bool IsWordChar(char c) noexcept {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

std::string FoldAscii(std::string_view text) {
  std::string folded;
  folded.reserve(text.size());
  for (const char c : text) {
    folded.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a')
                                          : c);
  }
  return folded;
}

bool HasBounded(std::string_view text, std::string_view phrase) {
  for (std::size_t pos = text.find(phrase); pos != std::string_view::npos;
       pos = text.find(phrase, pos + 1)) {
    const bool left_clear =
        pos == 0 || !IsWordChar(static_cast<char>(text[pos - 1]));
    const std::size_t next = pos + phrase.size();
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
      [text](std::string_view phrase) { return HasBounded(text, phrase); });
}

// Filler the model puts before a question; the cue must follow it directly.
constexpr std::array<std::string_view, 12> kLeadIns{{
    "wait",
    "hmm",
    "so",
    "but",
    "and",
    "or",
    "okay",
    "ok",
    "actually",
    "now",
    "then",
    "alternatively",
}};

// Open questions cannot be answered Yes/No.
constexpr std::array<std::string_view, 8> kWhWords{{
    "what",
    "which",
    "where",
    "why",
    "how",
    "when",
    "who",
    "whose",
}};

// A cue with fewer words after it ("is that true?", "is it?") carries no
// content, so jev would be judging nothing.
constexpr std::size_t kMinWordsAfterCue = 3;

bool StartsWithWord(std::string_view text,
                    std::span<const std::string_view> words) {
  for (const auto word : words) {
    if (text.size() >= word.size() && text.compare(0, word.size(), word) == 0 &&
        (text.size() == word.size() ||
         !IsWordChar(static_cast<char>(text[word.size()])))) {
      return true;
    }
  }
  return false;
}

std::size_t CountWords(std::string_view text) {
  std::size_t count = 0;
  bool in_word = false;
  for (const char c : text) {
    const bool word = IsWordChar(c);
    count += word && !in_word;
    in_word = word;
  }
  return count;
}

std::string_view StripLeadIns(std::string_view text) {
  for (bool again = true; again;) {
    again = false;
    while (!text.empty() && !IsWordChar(text.front())) {
      text.remove_prefix(1);
    }
    for (const auto lead : kLeadIns) {
      if (StartsWithWord(text, std::span(&lead, 1))) {
        text.remove_prefix(lead.size());
        again = true;
        break;
      }
    }
  }
  return text;
}

std::string Normalize(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  bool space = false;
  for (const char c : text) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      space = true;
      continue;
    }
    if (space && !out.empty()) {
      out.push_back(' ');
    }
    space = false;
    out.push_back(c);
  }
  return out;
}

}  // namespace

bool IsJevSelfQuestion(std::string_view sentence) {
  const std::string text = Normalize(sentence);
  if (text.size() < 8 || text.size() > kMaxQuestionChars) {
    return false;
  }
  if (text.back() != '?') {
    return false;
  }
  const std::string folded = FoldAscii(text);
  if (HasAny(folded, kReaderWords)) {
    return false;
  }
  std::string_view rest = StripLeadIns(folded);
  if (StartsWithWord(rest, kWhWords)) {
    return false;
  }
  for (const auto cue : kSelfCues) {
    if (!StartsWithWord(rest, std::span(&cue, 1))) {
      continue;
    }
    const std::string_view tail = rest.substr(cue.size());
    return CountWords(tail) >= kMinWordsAfterCue;
  }
  return false;
}

void JevInlineScanner::Reset() {
  tail_.clear();
  backticks_ = 0;
  pending_period_ = 0;
  in_fence_ = false;
  in_inline_code_ = false;
  overflowed_ = false;
}

void JevInlineScanner::EndSentence() {
  tail_.clear();
  overflowed_ = false;
}

// A run of backticks is deferred until the following character shows whether
// it is inline code or a fence; both hide `?` from the detector.
void JevInlineScanner::AdvanceBacktick(char c) {
  if (c == '`') {
    ++backticks_;
    return;
  }
  if (backticks_ >= 3) {
    in_fence_ = !in_fence_;
    tail_.clear();
  } else if (backticks_ > 0) {
    in_inline_code_ = !in_inline_code_;
  }
  backticks_ = 0;
}

void JevInlineScanner::Append(char c) {
  if (tail_.size() >= kMaxSentenceChars) {
    overflowed_ = true;
    tail_.clear();
    return;
  }
  tail_.push_back(c);
}

std::optional<std::string> JevInlineScanner::Push(std::string_view piece) {
  std::optional<std::string> result;
  for (const char c : piece) {
    AdvanceBacktick(c);
    if (backticks_ > 0 || in_fence_) {
      continue;
    }
    if (in_inline_code_) {
      // Keep inline code in the question text for jev's prompt, but a `?`
      // there is syntax, not a sentence end.
      Append(c == '\n' ? ' ' : c);
      continue;
    }
    if (pending_period_ != 0) {
      const char period = pending_period_;
      pending_period_ = 0;
      if (std::isspace(static_cast<unsigned char>(c)) != 0) {
        EndSentence();
      } else {
        Append(period);
      }
    }
    if (c == '.') {
      pending_period_ = '.';
      continue;
    }
    if (c == '?') {
      if (!overflowed_) {
        std::string candidate = tail_;
        candidate.push_back('?');
        if (IsJevSelfQuestion(candidate)) {
          result = Normalize(candidate);
        }
      }
      EndSentence();
      continue;
    }
    if (c == '!') {
      EndSentence();
      continue;
    }
    if (c == '\n') {
      tail_.push_back(' ');
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(c)) != 0 && tail_.empty()) {
      continue;
    }
    Append(c);
  }
  return result;
}

}  // namespace gufo::models::qwen38_flash_next
