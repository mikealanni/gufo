#include <iostream>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/reference.hpp"
#include "src/models/lfm3/tokenizer.hpp"

// Usage: lfm3_generate_check MODEL.gguf PROMPT N
// Greedy chat reply from the scalar reference, to compare with the NPU path.
int main(int argc, char** argv) {
  using namespace gufo;
  if (argc < 4)
    return 2;
  std::string error;
  auto reader = core::GgufReader::OpenFile(argv[1], &error);
  auto weights =
      reader ? models::lfm3::Weights::Bind(*reader, &error) : std::nullopt;
  auto tokenizer =
      reader ? models::lfm3::Tokenizer::Load(*reader, &error) : nullptr;
  if (!weights || !tokenizer) {
    std::cerr << error << '\n';
    return 1;
  }
  const std::vector<std::pair<std::string, std::string>> messages = {
      {"user", argv[2]}};
  const auto ids =
      tokenizer->Encode(models::lfm3::Tokenizer::Render(messages, false));
  const int count = std::atoi(argv[3]);
  models::lfm3::Reference reference(*weights, 4096);
  std::vector<float> logits(weights->config.vocab);
  std::vector<std::uint32_t> out;
  for (std::size_t i = 0; i < ids.size(); ++i)
    reference.Step(ids[i], i + 1 == ids.size() ? std::span<float>(logits)
                                               : std::span<float>());
  for (int k = 0; k < count; ++k) {
    std::uint32_t best = 0;
    for (std::uint32_t v = 1; v < logits.size(); ++v)
      if (logits[v] > logits[best])
        best = v;
    out.push_back(best);
    reference.Step(best, std::span<float>(logits));
  }
  std::cout << "prompt tokens " << ids.size() << "\nids:";
  for (auto id : out)
    std::cout << ' ' << id;
  std::cout << '\n' << tokenizer->Decode(out) << '\n';
}
