#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/raon1/kernels/rocm/executor.hpp"

static void GemmBench() {
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  const std::size_t shapes[][3] = {{5120, 2560, 1024}, {2560, 5120, 1024},
                                   {1088, 2560, 1024}, {2560, 1088, 1024},
                                   {576, 2560, 1024},  {7680, 576, 1024}};
  for (const auto& sh : shapes) {
    for (const std::size_t pad : {0UL}) {
      const std::size_t m = sh[0], k = sh[1], n = sh[2];
      const std::size_t ldx = k + pad;
      __half *w, *x;
      float* y;
      hipMalloc(&w, m * k * 2);
      hipMalloc(&x, n * ldx * 2);
      hipMalloc(&y, n * m * 4);
      hipMemset(w, 0x3c, m * k * 2);
      hipMemset(x, 0x3c, n * ldx * 2);
      for (int i = 0; i < 3; ++i)
        qfn::DenseHalfWeightGemm(w, x, y, n, m, k, nullptr, pad ? ldx : 0);
      hipDeviceSynchronize();
      hipEvent_t a, b;
      hipEventCreate(&a);
      hipEventCreate(&b);
      hipEventRecord(a);
      for (int i = 0; i < 10; ++i)
        qfn::DenseHalfWeightGemm(w, x, y, n, m, k, nullptr, pad ? ldx : 0);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms;
      hipEventElapsedTime(&ms, a, b);
      ms /= 10;
      std::printf("m=%zu k=%zu n=%zu pad=%zu %.3f ms %.1f TFLOPS\n", m, k, n,
                  pad, ms, 2.0 * m * k * n / (ms * 1e9));
      hipFree(w);
      hipFree(x);
      hipFree(y);
    }
  }
}

int main(int argc, char** argv) {
  if (std::getenv("GUFO_GEMM_BENCH") != nullptr) {
    GemmBench();
    return 0;
  }
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s MODEL TOKEN [TOKEN...]\n", argv[0]);
    return 1;
  }
  std::string error;
  gufo::models::raon1::rocm::ExecutorOptions options;
  if (const char* layers = std::getenv("GUFO_RAON1_GPU_LAYERS")) {
    options.layers = static_cast<std::uint32_t>(std::atoi(layers));
  }
  auto executor =
      gufo::models::raon1::rocm::Executor::Create(argv[1], options, &error);
  if (!executor) {
    std::fprintf(stderr, "create failed: %s\n", error.c_str());
    return 1;
  }
  auto session = executor->CreateSession(&error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }
  FILE* dump = nullptr;
  if (const char* path = std::getenv("GUFO_GPU_DUMP_LOGITS")) {
    dump = std::fopen(path, "wb");
  }
  if (const char* batch = std::getenv("GUFO_GPU_BATCH")) {
    const int width = std::atoi(batch);
    std::vector<std::int32_t> ids;
    for (int i = 2; i < argc; ++i)
      ids.push_back(std::atoi(argv[i]));
    for (std::size_t i = 0; i < ids.size(); i += width) {
      const std::size_t len = std::min<std::size_t>(width, ids.size() - i);
      if (!executor->EvaluateBatch(
              *session, std::span<const std::int32_t>(ids.data() + i, len),
              &error)) {
        std::fprintf(stderr, "batch failed: %s\n", error.c_str());
        return 1;
      }
    }
    const auto logits = executor->Logits(*session);
    if (dump != nullptr) {
      std::fwrite(logits.data(), sizeof(float), logits.size(), dump);
      std::fclose(dump);
    }
    std::printf("batch argmax %d\n", executor->Argmax(*session, &error));
    return 0;
  }
  for (int i = 2; i < argc; ++i) {
    const int token = std::atoi(argv[i]);
    if (!executor->Evaluate(*session, token, &error)) {
      std::fprintf(stderr, "evaluate failed: %s\n", error.c_str());
      return 1;
    }
    const auto logits = executor->Logits(*session);
    if (dump != nullptr) {
      std::fwrite(logits.data(), sizeof(float), logits.size(), dump);
    }
    std::vector<std::pair<float, int>> top;
    for (std::size_t t = 0; t < logits.size(); ++t)
      top.emplace_back(logits[t], static_cast<int>(t));
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      std::greater<>());
    std::printf("pos %u:", session->position - 1);
    for (int k = 0; k < 5; ++k) {
      std::printf(" %d(%.4f)", top[k].second, top[k].first);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  if (dump != nullptr)
    std::fclose(dump);
  return 0;
}
