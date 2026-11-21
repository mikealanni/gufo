#include "src/cli/serve/logprobs.hpp"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "src/cli/serve/responses_logprobs.hpp"

int main() {
  using namespace gufo::server;
  // Independent byte-fallback fixture: alternatives use preceding sampled
  // context, never another alternative at the same position.
  const std::vector<std::string> vocabulary{"\xc3", "\xa9", "A", "\xaa", "🙂"};
  LogprobDecoder decoder;
  const auto decode = [&](auto id) { return vocabulary.at(id); };
  const auto first = decoder.DecodeToken({{0, -1}, {{2, -2}, {4, -3}}}, decode);
  assert(first.entries[0].token.empty());
  assert(first.entries[1].token == "A");
  const auto second =
      decoder.DecodeToken({{1, -0.1}, {{3, -2}, {1, -0.1}}}, decode);
  assert(second.entries.size() == 2);
  assert(second.entries[0].token == "é" && second.entries[1].token == "ê");
  const std::vector<DecodedLogprobs> sequence{first, second};
  const auto chat = ChatLogprobs(sequence, 1);
  assert(chat.find("content")->items()[1].find("bytes")->dump() == "[169]");
  assert(chat.find("content")->items()[0].find("top_logprobs")->size() == 1);
  const auto completion = CompletionLogprobs(sequence, 7);
  assert(completion.find("text_offset")->dump() == "[7,7]");
  assert(UnicodeLength("aé🙂") == 3);
  // vLLM keeps the selected token in completion maps even with zero requested
  // alternatives; very negative values use OpenAI's JSON-safe sentinel.
  const DecodedLogprobs unlikely{{{9, "x", -INFINITY}}, 1};
  assert(CompletionLogprobs(std::span(&unlikely, 1))
             .find("top_logprobs")
             ->dump() == R"([{"x":-9999}])");
  assert(ChatLogprobs(std::span(&unlikely, 1), 0)
             .find("content")
             ->items()[0]
             .find("top_logprobs")
             ->size() == 0);
  const auto disabled =
      gufo::json::parse(R"({"logprobs":null,"top_logprobs":null})");
  gufo::sampling::SamplingConfig config;
  assert(!ParseLogprobs(disabled, true, &config) && !config.logprobs);
  for (const auto* body :
       {"{}", R"({"logprobs":false})", R"({"top_logprobs":0})"}) {
    config.logprobs = 20;
    assert(!ParseLogprobs(gufo::json::parse(body), true, &config));
    assert(!config.logprobs);
  }
  assert(
      !ParseLogprobs(gufo::json::parse(R"({"logprobs":true})"), true, &config));
  assert(config.logprobs == 0);
  for (const auto* body :
       {"{}", R"({"include":null})", R"({"include":[],"top_logprobs":20})",
        R"({"include":["reasoning.encrypted_content"]})"}) {
    config.logprobs = 20;
    assert(!ParseResponsesLogprobs(gufo::json::parse(body), &config));
    assert(!config.logprobs);
  }
  for (const auto* body :
       {R"({"include":"message.output_text.logprobs"})",
        R"({"include":[null]})", R"({"include":[false]})",
        R"({"top_logprobs":-1})", R"({"top_logprobs":21})",
        R"({"top_logprobs":1.5})", R"({"top_logprobs":true})",
        R"({"top_logprobs":"3"})"})
    assert(ParseResponsesLogprobs(gufo::json::parse(body), &config));
  assert(!ParseResponsesLogprobs(
      gufo::json::parse(
          R"({"include":["message.output_text.logprobs"],"top_logprobs":null})"),
      &config));
  assert(config.logprobs == 0);

  // Two tokens contribute to one Unicode scalar; neither is lost or repeated.
  VisibleLogprobs response(0);
  response.Push(TokenPiece("\xc3", &first));
  response.Push(TokenPiece("\xa9", &second));
  const TextSourceSpan accent{0, 2};
  const auto accent_tokens = response.Take(std::span(&accent, 1));
  assert(accent_tokens.size() == 2);
  assert(accent_tokens.items()[0].member_str("token").empty());
  assert(accent_tokens.items()[1].member_str("token") == "é");
  assert(response.Take(std::span(&accent, 1)).empty());

  // A malformed prefix is replaced before the next ASCII token. Their source
  // ranges must not be merged just because both decoded in the same Push.
  VisibleLogprobs malformed(0);
  malformed.Push(TokenPiece("\xe2", &first));
  malformed.Push(TokenPiece("A", &unlikely));
  const TextSourceSpan ascii{3, 1};
  const auto ascii_tokens = malformed.Take(std::span(&ascii, 1));
  assert(ascii_tokens.size() == 1);
  assert(ascii_tokens.items()[0].member_str("token") == "x");
  const TextSourceSpan replacement{0, 3};
  assert(malformed.Take(std::span(&replacement, 1)).size() == 1);
  malformed.Push(TokenPiece("", &second));
  malformed.Push(TokenPiece(""), true);
  assert(malformed.Take(std::span(&ascii, 1)).empty());

  // A possible stop prefix is initially held. When it fails to match,
  // the released bytes still belong to their original tokens.
  VisibleLogprobs held(0);
  const DecodedLogprobs a{{{1, "a", -1}}, 1}, b{{{2, "b", -2}}, 1};
  held.Push(TokenPiece("", &a));
  held.Push(TokenPiece("ab", &b));
  const TextSourceSpan pair{0, 2};
  const auto released = held.Take(std::span(&pair, 1));
  assert(released.size() == 2);
  assert(released.items()[0].member_str("token") == "a");
  assert(released.items()[1].member_str("token") == "b");
  assert(chat.find("content")->items()[0].find("bytes")->dump() == "[195]");

  VisibleLogprobs truncated(0);
  truncated.Push(TokenPiece("\xe2", &first));
  truncated.Push(TokenPiece(""), true);
  assert(truncated.Take(std::span(&replacement, 1)).size() == 1);
}
