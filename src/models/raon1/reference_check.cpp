#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "src/models/raon1/reference.hpp"

int main(int argc, char** argv) {
  constexpr int kTop = 5;
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.gguf token...\n", argv[0]);
    return 2;
  }
  std::string error;
  auto model = gufo::models::raon1::ReferenceModel::Open(argv[1], &error);
  if (!model) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession();
  FILE* dump = nullptr;
  if (const char* path = std::getenv("GUFO_DUMP_LOGITS"))
    dump = std::fopen(path, "wb");
  for (int i = 2; i < argc; ++i) {
    const auto& logits = model->Evaluate(*session, std::atoi(argv[i]));
    if (dump)
      std::fwrite(logits.data(), sizeof(float), logits.size(), dump);
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
  if (dump)
    std::fclose(dump);
  return 0;
}
