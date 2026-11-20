// Next-token NLL of the Qwen27B prefill on real text, one prefill per sample:
// a context window of `length` tokens is prefilled and the log probability of
// the token that follows it in the corpus is read from the final logits. Two
// runs with the same arguments (for example with and without
// GUFO_IU4_ARTIFACT) see identical contexts, so their per-sample NLL can be
// compared directly. Usage:
//   qwen27b_prefill_nll <model.gguf> <corpus.txt> <samples> <length> [seed]
// Prints one line per sample ("sample <i> nll <x> top1 <id>") and a summary.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {
std::string ReadFile(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr)
    return {};
  std::string out;
  char buf[1 << 16];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
    out.append(buf, n);
  std::fclose(f);
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> <corpus.txt> <samples> <length> "
                 "[seed]\n",
                 argv[0]);
    return 2;
  }
  const std::size_t samples = std::strtoul(argv[3], nullptr, 10);
  const std::size_t length = std::strtoul(argv[4], nullptr, 10);
  std::uint64_t state = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 1;
  state = state * 0x9E3779B97F4A7C15ull + 0x1234567ull;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  auto executor = gufo::hip::QwenGpuExecutor::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), &error,
      length + 64);
  if (executor == nullptr) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const std::string text = ReadFile(argv[2]);
  const auto tokens = executor->GetTokenizer().Encode(text);
  if (tokens.size() < length + 2) {
    std::fprintf(stderr, "corpus has %zu tokens, need %zu\n", tokens.size(),
                 length + 2);
    return 1;
  }
  std::printf("corpus tokens %zu\n", tokens.size());
  const auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  double total = 0.0;
  for (std::size_t i = 0; i < samples; ++i) {
    const std::size_t start = next() % (tokens.size() - length - 1);
    executor->Reset();
    (void)executor->ForwardPromptBatch(
        std::span(tokens).subspan(start, length));
    const auto logits = executor->CopyLastLogits();
    double maximum = *std::ranges::max_element(logits), sum = 0.0;
    for (const float v : logits)
      sum += std::exp(static_cast<double>(v) - maximum);
    const std::size_t target = tokens[start + length];
    const double nll = maximum + std::log(sum) - logits[target];
    const auto top1 = static_cast<std::size_t>(
        std::ranges::max_element(logits) - logits.begin());
    total += nll;
    std::printf("sample %zu nll %.6f top1 %zu target %zu\n", i, nll, top1,
                target);
    std::fflush(stdout);
  }
  std::printf("mean nll %.6f over %zu samples\n", total / samples, samples);
  return 0;
}
