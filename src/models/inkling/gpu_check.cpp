#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/inkling/gpu/engine.hpp"
#include "src/models/qwen/tokenizer.hpp"

int main(int argc, char** argv) {
  constexpr int kTop = 5;
  const std::uint32_t kContext =
      std::getenv("INKLING_CTX") != nullptr
          ? static_cast<std::uint32_t>(std::atoi(std::getenv("INKLING_CTX")))
          : 8192U;
  if (argc >= 3 && std::string(argv[2]) == "--text") {
    const int kGenerate = std::getenv("INKLING_GEN") != nullptr
                              ? std::atoi(std::getenv("INKLING_GEN"))
                              : 64;
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    if (!tok) {
      std::fprintf(stderr, "tokenizer failed: %s\n", err.c_str());
      return 1;
    }
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session = model->CreateSession(kContext, &err);
    std::string prompt_text = argc > 3 ? argv[3] : "The capital of France is";
    if (const char* pf = std::getenv("INKLING_PROMPT_FILE")) {
      std::ifstream pin(pf);
      std::stringstream pss;
      pss << pin.rdbuf();
      prompt_text = pss.str();
    }
    const auto ids = tok->Encode(prompt_text, {});
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
  if (argc >= 6 && std::string(argv[2]) == "--windows") {
    // --windows <text file> <count> <len> <out.bin>: final-token logits of
    // count prose windows
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int count = std::atoi(argv[4]);
    const int len = std::atoi(argv[5]);
    std::ofstream out(argv[6], std::ios::binary);
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session =
        model->CreateSession(static_cast<std::uint32_t>(len) + 16, &err);
    for (int w = 0; w < count; ++w) {
      session->Reset();
      std::vector<std::int32_t> win;
      for (int i = 0; i < len; ++i)
        win.push_back(static_cast<std::int32_t>(
            ids[(static_cast<std::size_t>(w) * len + i) % ids.size()]));
      if (!session->Append(win, &err))
        return 1;
      const auto l = session->Logits();
      out.write(reinterpret_cast<const char*>(l.data()),
                static_cast<std::streamsize>(l.size() * sizeof(float)));
    }
    return 0;
  }
  if (argc >= 4 && std::string(argv[2]) == "--verify") {
    // --verify <text file> <k>: AppendAll(k rows) must match k sequential single-token steps
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int k = argc > 4 ? std::atoi(argv[4]) : 2;
    constexpr int kPrefix = 300;
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session = model->CreateSession(2048, &err);
    std::vector<std::int32_t> prefix;
    for (int i = 0; i < kPrefix; ++i) prefix.push_back(static_cast<std::int32_t>(ids[i]));
    std::vector<std::int32_t> next;
    for (int i = 0; i < k; ++i) next.push_back(static_cast<std::int32_t>(ids[kPrefix + i]));
    (void)session->Append(prefix, &err);
    std::vector<std::vector<float>> seq;
    for (int i = 0; i < k; ++i) {
      (void)session->Append(std::span(&next[i], 1), &err);
      const auto l = session->Logits();
      seq.emplace_back(l.begin(), l.end());
    }
    session->Reset();
    (void)session->Append(prefix, &err);
    if (!session->AppendAll(next, &err)) {
      std::fprintf(stderr, "AppendAll failed: %s\n", err.c_str());
      return 1;
    }
    for (int i = 0; i < k; ++i) {
      const auto l = session->Logits(static_cast<std::uint32_t>(i));
      double maxd = 0.0;
      int a0 = 0, a1 = 0;
      for (std::size_t j = 0; j < l.size(); ++j) {
        maxd = std::max(maxd, std::fabs(static_cast<double>(l[j]) - seq[i][j]));
        if (l[j] > l[a0]) a0 = static_cast<int>(j);
        if (seq[i][j] > seq[i][a1]) a1 = static_cast<int>(j);
      }
      std::printf("row %d: max |dlogit| %.4f, argmax batch %d seq %d\n", i, maxd, a0, a1);
    }
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
    std::vector<std::int32_t> prompt;
    if (const char* file = std::getenv("INKLING_BENCH_FILE")) {
      std::string err;
      auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
      auto tok =
          gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
      std::ifstream in(file);
      std::stringstream ss;
      ss << in.rdbuf();
      const auto ids = tok->Encode(ss.str(), {});
      for (int i = 0; i < n; ++i)
        prompt.push_back(static_cast<std::int32_t>(
            ids[static_cast<std::size_t>(i) % ids.size()]));
    } else {
      prompt.assign(n, kBenchToken);
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!session->Append(prompt, &error)) {
      std::fprintf(stderr, "bench prefill failed: %s\n", error.c_str());
      return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < kDecode; ++i) {
      const auto logits = session->Logits();
      const auto best = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      if (!session->Append(std::span(&best, 1), &error)) {
        std::fprintf(stderr, "bench decode failed: %s\n", error.c_str());
        return 1;
      }
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("pp %d: %.1f tok/s, tg: %.2f tok/s\n", n,
                n / std::chrono::duration<double>(t1 - t0).count(),
                kDecode / std::chrono::duration<double>(t2 - t1).count());
  }
  return 0;
}
