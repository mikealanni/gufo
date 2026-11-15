#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/d1/model.hpp"
#include "src/models/lfm3/decision_forward.hpp"
#include "src/models/qwen/control_tokens.hpp"

// Usage: d1_decide_check MODEL.gguf REQUEST.json [pack|single]
// Host oracle for the text-only d1 decision path: packs the questions,
// runs the FP32 forward and prints the answers as JSON.
int main(int argc, char** argv) {
  using namespace gufo;
  using namespace gufo::models;
  if (argc < 3) {
    std::cerr
        << "usage: d1_decide_check MODEL.gguf REQUEST.json [pack|single]\n";
    return 2;
  }
  const bool single = argc > 3 && std::string(argv[3]) == "single";
  std::string error;
  auto reader = core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << error << '\n';
    return 1;
  }
  auto weights = lfm3::Weights::Bind(*reader, &error);
  auto tokenizer = lfm3::Tokenizer::Load(*reader, &error);
  if (!weights || !tokenizer) {
    std::cerr << error << '\n';
    return 1;
  }
  std::string text;
  {
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f)
      return 1;
    char buf[4096];
    for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
      text.append(buf, n);
    std::fclose(f);
  }
  const auto request = json::parse(text);
  const auto* state = request.find("state");
  const auto* qs = request.find("questions");
  std::vector<d1::Question> questions;
  for (const auto& [name, value] : qs->members()) {
    auto q = d1::ParseQuestion(*tokenizer, name, value);
    q.temperature =
        reader->GetMetadataFloat32("lfm3.decision.temperature." + q.type)
            .value_or(1);
    questions.push_back(std::move(q));
  }
  std::string prefix(tokenizer->Piece(tokenizer->Bos()));
  prefix += tokenization::kImStart;
  prefix += "user\n";
  if (state && !state->is_null())
    prefix += d1::Plain(*tokenizer, state->str(), weights->config.vocab) +
              "\n\n\nQUESTION:\n";
  const std::string header = state && !state->is_null() ? "QUESTION:\n" : "";
  std::vector<std::string> suffixes;
  for (const auto& q : questions)
    suffixes.push_back(d1::Suffix(
        *tokenizer, q, d1::Plain(*tokenizer, q.body, weights->config.vocab)));
  std::vector<std::uint32_t> vocabulary;
  std::unordered_map<std::uint32_t, std::uint32_t> columns;
  for (auto& q : questions)
    for (auto& forms : q.readout)
      for (auto& id : forms) {
        auto [at, added] = columns.emplace(id, vocabulary.size());
        if (added)
          vocabulary.push_back(id);
        id = at->second;
      }
  auto answers = json::Value::object();
  const auto start = std::chrono::steady_clock::now();
  std::size_t tokens_run = 0;
  const std::size_t set = single ? 1 : 16;
  for (std::size_t first = 0; first < questions.size(); first += set) {
    const auto count = std::min(set, questions.size() - first);
    auto own = std::span(questions).subspan(first, count);
    auto prompt = d1::Pack(*tokenizer, prefix, header,
                           std::span(suffixes).subspan(first, count), own);
    const auto groups = count > 1 ? std::span<const std::int32_t>(prompt.groups)
                                  : std::span<const std::int32_t>();
    const auto logits = lfm3::DecisionForward(*weights, prompt.tokens, groups,
                                              prompt.slots, vocabulary);
    tokens_run += prompt.tokens.size();
    for (std::size_t i = 0; i < count; ++i)
      answers[own[i].name] = d1::ReadAnswer(
          own[i],
          std::span(logits).subspan(i * vocabulary.size(), vocabulary.size()));
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  std::cout << answers.dump() << "\ntokens " << tokens_run << " ms " << ms
            << '\n';
  return 0;
}
