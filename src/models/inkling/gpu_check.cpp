#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "src/models/inkling/gpu/engine.hpp"

int main(int argc, char** argv) {
  constexpr int kTop = 5;
  constexpr std::uint32_t kContext = 8192;
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
  return 0;
}
