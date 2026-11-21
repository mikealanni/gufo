#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/inkling/gpu/engine.hpp"
#include "src/models/qwen/tokenizer.hpp"

int main(int argc, char** argv) {
  constexpr int kTop = 5;
  constexpr std::uint32_t kContext = 8192;
  if (argc >= 3 && std::string(argv[2]) == "--text") {
    constexpr int kGenerate = 64;
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    if (!tok) {
      std::fprintf(stderr, "tokenizer failed: %s\n", err.c_str());
      return 1;
    }
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session = model->CreateSession(kContext, &err);
    const auto ids =
        tok->Encode(argc > 3 ? argv[3] : "The capital of France is", {});
    std::vector<std::int32_t> in(ids.begin(), ids.end());
    std::printf("prompt tokens: %zu\n", in.size());
    (void)session->Append(in, &err);
    for (int i = 0; i < kGenerate; ++i) {
      const auto logits = session->Logits();
      const auto best = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      std::printf("%s", tok->DecodeTokenCopy(
                               static_cast<gufo::tokenization::TokenId>(best))
                            .c_str());
      std::fflush(stdout);
      (void)session->Append(std::span(&best, 1), &err);
    }
    std::printf("\n");
    return 0;
  }
  if (argc < 3)
    return 2;
  std::string error;
  auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(kContext, &error);
  const bool batch = std::getenv("INKLING_BATCH") != nullptr;
  std::vector<std::int32_t> toks;
  for (int i = 2; i < argc; ++i)
    toks.push_back(std::atoi(argv[i]));
  const std::size_t step = batch ? toks.size() : 1;
  for (std::size_t i = 0; i < toks.size(); i += step) {
    if (!session->Append(std::span(toks).subspan(i, step), &error)) {
      std::fprintf(stderr, "append failed: %s\n", error.c_str());
      return 1;
    }
    const auto logits = session->Logits();
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kTop, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("pos %zu:", i + step - 1);
    for (int k = 0; k < kTop; ++k)
      std::printf(" %d(%.4f)", idx[k], logits[idx[k]]);
    std::printf("\n");
  }
  if (const char* env = std::getenv("INKLING_BENCH")) {
    constexpr int kDecode = 32;
    constexpr int kBenchToken = 1000;
    const int n = std::atoi(env);
    session->Reset();
    std::vector<std::int32_t> prompt(n, kBenchToken);
    const auto t0 = std::chrono::steady_clock::now();
    (void)session->Append(prompt, &error);
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < kDecode; ++i) {
      const std::int32_t one = kBenchToken;
      (void)session->Append(std::span(&one, 1), &error);
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("pp %d: %.1f tok/s, tg: %.2f tok/s\n", n,
                n / std::chrono::duration<double>(t1 - t0).count(),
                kDecode / std::chrono::duration<double>(t2 - t1).count());
  }
  return 0;
}
