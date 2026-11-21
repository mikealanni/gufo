#ifndef GUFO_SERVER_LOGPROBS_HPP_
#define GUFO_SERVER_LOGPROBS_HPP_

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/cli/serve/sampling_request.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {

struct DecodedLogprob {
  sampling::TokenId id{0};
  std::string token;
  double logprob{0};
  std::optional<std::string> raw_bytes;
};

/// Sampled token first, then the unique top candidates in descending order,
/// as in vLLM v0.32.0's append_logprobs_for_next_position.
struct DecodedLogprobs {
  std::vector<DecodedLogprob> entries;
  std::size_t source_bytes{0};
  [[nodiscard]] std::size_t StorageBytes() const {
    std::size_t bytes =
        sizeof(*this) + entries.capacity() * sizeof(DecodedLogprob);
    for (const auto& entry : entries)
      bytes += entry.token.capacity() +
               (entry.raw_bytes ? entry.raw_bytes->capacity() : 0);
    return bytes;
  }
};

/// A borrowed event; text-only consumers retain the existing string-view API.
struct TokenPiece {
  std::string_view text;
  const DecodedLogprobs* logprobs{nullptr};
  TokenPiece(std::string_view value,
             const DecodedLogprobs* probabilities = nullptr)
      : text(value), logprobs(probabilities) {}
  TokenPiece(const std::string& value) : text(value) {}
  TokenPiece(const char* value) : text(value) {}
  operator std::string_view() const { return text; }
};

inline std::optional<SamplingRequestError> ParseLogprobs(
    const json::Value& body, bool chat, sampling::SamplingConfig* config) {
  config->logprobs.reset();
  const auto* enabled = body.find("logprobs");
  if (chat) {
    if (enabled && !enabled->is_null() && !enabled->is_bool())
      return SamplingRequestError{"'logprobs' must be a boolean",
                                  "invalid_logprobs"};
    std::uint32_t count = 0;
    if (auto error = detail::ReadSamplingInteger(body, "top_logprobs", 0U, 20U,
                                                 &count, true))
      return error;
    if (enabled && enabled->as_bool())
      config->logprobs = count;
    else if (count)
      return SamplingRequestError{"'top_logprobs' requires 'logprobs': true",
                                  "invalid_top_logprobs"};
  } else if (enabled && !enabled->is_null()) {
    std::uint32_t count = 0;
    if (auto error = detail::ReadSamplingInteger(body, "logprobs", 0U, 20U,
                                                 &count, true))
      return error;
    config->logprobs = count;
  }
  return std::nullopt;
}

/// Correct byte-fallback tokens against the last four sampled token pieces,
/// independently for every alternative (vLLM v0.32.0 LogprobsProcessor).
class LogprobDecoder {
public:
  template<typename Decode>
  DecodedLogprobs DecodeToken(const sampling::TokenLogprobs& values,
                              Decode decode) {
    DecodedLogprobs result;
    const auto raw = decode(values.selected.token);
    result.source_bytes = raw.size();
    const auto append = [&](const sampling::TokenLogprob& entry) {
      const auto bytes = decode(entry.token);
      auto text = Correct(bytes);
      auto original =
          text == bytes ? std::nullopt : std::optional(std::string(bytes));
      result.entries.push_back(
          {entry.token, std::move(text), entry.logprob, std::move(original)});
    };
    append(values.selected);
    for (const auto& entry : values.top)
      if (entry.token != values.selected.token)
        append(entry);
    history_.push_back(raw);
    if (history_.size() > 4)
      history_.erase(history_.begin());
    return result;
  }

private:
  std::string Correct(std::string_view raw) const {
    constexpr std::string_view replacement = "\xEF\xBF\xBD";
    auto text = core::Utf8Decoder{}.Push(raw, true);
    if (!text.ends_with(replacement))
      return text;
    for (std::size_t count = 1; count <= history_.size(); ++count) {
      const auto begin = history_.size() - count;
      std::string combined;
      for (auto i = begin; i < history_.size(); ++i)
        combined += history_[i];
      combined += raw;
      auto decoded = core::Utf8Decoder{}.Push(combined, true);
      if (decoded.ends_with(replacement))
        continue;
      auto clean_end = history_.size();
      while (clean_end > begin && core::Utf8Decoder{}
                                      .Push(history_[clean_end - 1], true)
                                      .ends_with(replacement))
        --clean_end;
      std::string clean;
      for (auto i = begin; i < clean_end; ++i)
        clean += history_[i];
      clean = core::Utf8Decoder{}.Push(clean, true);
      std::size_t common = 0;
      while (common < clean.size() && common < decoded.size() &&
             clean[common] == decoded[common])
        ++common;
      // Keep the substring on a Unicode scalar boundary.
      while (common < decoded.size() && common > 0 &&
             (static_cast<unsigned char>(decoded[common]) & 0xc0) == 0x80)
        --common;
      return decoded.substr(common);
    }
    return {};
  }
  std::vector<std::string> history_;
};

inline json::Value LogprobJson(const DecodedLogprob& value) {
  auto result = json::Value::object();
  result["token"] = value.token;
  result["logprob"] = std::max(value.logprob, -9999.0);
  result["bytes"] = json::Value::array();
  for (const unsigned char byte :
       value.raw_bytes ? *value.raw_bytes : value.token)
    result["bytes"].push_back(static_cast<int>(byte));
  return result;
}

inline json::Value ChatLogprobs(std::span<const DecodedLogprobs> tokens,
                                std::uint32_t count) {
  auto result = json::Value::object();
  result["content"] = json::Value::array();
  for (const auto& token : tokens) {
    auto entry = LogprobJson(token.entries.at(0));
    entry["top_logprobs"] = json::Value::array();
    // The selected token is reported separately and may rank outside top K.
    const auto& selected = token.entries.front();
    auto next = token.entries.begin() + 1;
    bool used_selected = false;
    for (std::size_t i = 0;
         i < std::min<std::size_t>(count, token.entries.size()); ++i) {
      const bool take_selected =
          !used_selected &&
          (next == token.entries.end() || selected.logprob > next->logprob ||
           (selected.logprob == next->logprob && selected.id < next->id));
      entry["top_logprobs"].push_back(
          LogprobJson(take_selected ? selected : *next++));
      used_selected |= take_selected;
    }
    result["content"].push_back(std::move(entry));
  }
  return result;
}

inline std::size_t UnicodeLength(std::string_view text) {
  return std::count_if(text.begin(), text.end(), [](unsigned char byte) {
    return (byte & 0xc0) != 0x80;
  });
}

inline json::Value CompletionLogprobs(std::span<const DecodedLogprobs> tokens,
                                      std::size_t& offset) {
  auto result = json::Value::object();
  for (const auto* name :
       {"tokens", "token_logprobs", "top_logprobs", "text_offset"})
    result[name] = json::Value::array();
  for (const auto& token : tokens) {
    const auto& selected = token.entries.at(0);
    result["tokens"].push_back(selected.token);
    result["token_logprobs"].push_back(std::max(selected.logprob, -9999.0));
    result["text_offset"].push_back(offset);
    offset += UnicodeLength(selected.token);
    auto top = json::Value::object();
    // The map can represent only one value per spelling. Entries are
    // selected-first, followed by alternatives in descending probability.
    for (const auto& entry : token.entries)
      if (!top.contains(entry.token))
        top[entry.token] = std::max(entry.logprob, -9999.0);
    result["top_logprobs"].push_back(std::move(top));
  }
  return result;
}

inline json::Value CompletionLogprobs(std::span<const DecodedLogprobs> tokens) {
  std::size_t offset = 0;
  return CompletionLogprobs(tokens, offset);
}

/// A range in the UTF-8-normalized, unfiltered model output.
struct TextSourceSpan {
  std::size_t begin;
  std::size_t size;
};

/// Join filtered output spans to generated tokens by source position, never
/// by matching text (a tool argument can repeat the visible answer verbatim).
/// A token crossing a text/framing boundary belongs to its first visible
/// text span. Pure reasoning, tool markup and hidden stop tokens are omitted.
class VisibleLogprobs {
public:
  explicit VisibleLogprobs(std::uint32_t count) : count_(count) {}

  void Push(const TokenPiece& piece, bool final = false) {
    // Stop matching may hold several sampled pieces before releasing their
    // text.
    if (piece.logprobs && piece.logprobs->source_bytes)
      tokens_.push_back(
          {.input_begin = input_bytes_,
           .input_end = input_bytes_ + piece.logprobs->source_bytes,
           .values = *piece.logprobs});
    input_bytes_ += piece.logprobs ? piece.logprobs->source_bytes : 0;
    (void)decoder_.Push(piece.text, final, [&](auto input, auto output) {
      const auto end = consumed_bytes_ + input;
      for (auto i = mapped_; i < tokens_.size() && tokens_[i].input_begin < end;
           ++i) {
        auto& token = tokens_[i];
        if (!token.mapped)
          token.begin = decoded_bytes_;
        token.end = decoded_bytes_ + output;
        token.mapped = true;
      }
      while (mapped_ < tokens_.size() && tokens_[mapped_].input_end <= end)
        ++mapped_;
      consumed_bytes_ = end;
      decoded_bytes_ += output;
    });
  }

  json::Value Take(std::span<const TextSourceSpan> spans) {
    auto result = json::Value::array();
    for (const auto& span : spans) {
      if (!span.size)
        continue;
      // Unmapped tokens are only the trailing incomplete UTF-8 scalar.
      const auto mapped_end =
          tokens_.begin() + mapped_ +
          (mapped_ < tokens_.size() && tokens_[mapped_].mapped ? 1 : 0);
      auto it = std::lower_bound(
          tokens_.begin(), mapped_end, span.begin,
          [](const auto& token, auto begin) { return token.end <= begin; });
      for (; it != mapped_end && it->begin < span.begin + span.size; ++it) {
        if (it->emitted)
          continue;
        auto entries = ChatLogprobs(std::span(&it->values, 1), count_);
        result.push_back(entries.find("content")->items().front());
        it->emitted = true;
      }
    }
    return result;
  }

private:
  struct Token {
    std::size_t input_begin, input_end;
    DecodedLogprobs values;
    std::size_t begin{0}, end{0};
    bool mapped{false}, emitted{false};
  };
  std::uint32_t count_;
  core::Utf8Decoder decoder_;
  std::vector<Token> tokens_;
  std::size_t input_bytes_{0}, consumed_bytes_{0}, decoded_bytes_{0},
      mapped_{0};
};

}  // namespace gufo::server
#endif  // GUFO_SERVER_LOGPROBS_HPP_
