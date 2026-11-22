#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "src/models/inkling/reference.hpp"

int main(int argc, char** argv) {
  constexpr int kTop = 5;
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.gguf token...\n", argv[0]);
    return 2;
  }
  std::string error;
  auto model = gufo::models::inkling::ReferenceModel::Open(argv[1], &error);
  if (!model) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession();
  for (int i = 2; i < argc; ++i) {
    const auto& logits = model->Evaluate(*session, std::atoi(argv[i]));
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kTop, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("pos %d:", i - 2);
    for (int k = 0; k < kTop; ++k)
      std::printf(" %d(%.4f)", idx[k], logits[idx[k]]);
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
