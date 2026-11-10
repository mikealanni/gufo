#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/decision_forward.hpp"
#include "src/models/lfm3/reference.hpp"

// Usage: decision_forward_check MODEL.gguf
// Causal DecisionForward must reproduce the scalar Reference::Step logits.
int main(int argc, char** argv) {
  using namespace gufo::models::lfm3;
  if (argc < 2) {
    std::cerr << "usage: decision_forward_check MODEL.gguf\n";
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << error << '\n';
    return 1;
  }
  auto weights = Weights::Bind(*reader, &error);
  if (!weights) {
    std::cerr << error << '\n';
    return 1;
  }
  const std::vector<std::uint32_t> tokens = {1234, 5678, 910, 11121, 3141,
                                             5926, 5358, 9793};
  Reference reference(*weights, 64);
  std::vector<float> expected(weights->config.vocab);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    reference.Step(tokens[i], i + 1 == tokens.size()
                                  ? std::span<float>(expected)
                                  : std::span<float>());
  const std::uint32_t slot = tokens.size() - 1;
  const auto actual = DecisionForward(*weights, tokens, {},
                                      std::span(&slot, 1), {}, true);
  double worst = 0, scale = 0;
  std::size_t top_a = 0, top_e = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    worst = std::max(worst, double(std::abs(actual[i] - expected[i])));
    scale = std::max(scale, double(std::abs(expected[i])));
    if (actual[i] > actual[top_a]) top_a = i;
    if (expected[i] > expected[top_e]) top_e = i;
  }
  std::cout << "worst |dlogit| " << worst << " of max |logit| " << scale
            << ", argmax actual " << top_a << " expected " << top_e << '\n';
  return (worst < 1e-3 * std::max(1.0, scale) && top_a == top_e) ? 0 : 1;
}
