#include <iostream>

#include "src/core/gguf_reader.hpp"
#include "src/models/d1/judge.hpp"
#include "src/models/lfm3/decision_forward.hpp"

// Usage: d1_trace_check MODEL.gguf QUESTION CONTEXT
// One yes/no decision on the FP32 host oracle (use GUFO_D1_TRACE=1).
int main(int argc, char** argv) {
  using namespace gufo;
  if (argc < 4) return 2;
  std::string error;
  auto reader = core::GgufReader::OpenFile(argv[1], &error);
  auto weights = reader ? models::lfm3::Weights::Bind(*reader, &error) : std::nullopt;
  if (!weights) { std::cerr << error << '\n'; return 1; }
  auto judge = models::d1::Judge::Create(
      *reader,
      [&](auto tokens, auto groups, auto slots, auto vocabulary, auto logits) {
        std::cerr << "tokens " << tokens.size() << " first";
        for (std::size_t i = 0; i < 6 && i < tokens.size(); ++i) std::cerr << ' ' << tokens[i];
        std::cerr << " slot " << slots[0] << '\n';
        const auto out = models::lfm3::DecisionForward(*weights, tokens, groups, slots, vocabulary);
        std::copy(out.begin(), out.end(), logits.begin());
        std::cerr << "logits " << out[0] << ' ' << out[1] << '\n';
        return true;
      }, 512, &error);
  const std::pair<std::string_view, std::string_view> item{argv[2], argv[3]};
  const auto v = judge->Answer(std::span(&item, 1));
  std::cout << "yes=" << v[0].yes << " no=" << v[0].no << '\n';
}
