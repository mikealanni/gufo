// NPU timing probe at routed-expert shapes: one 1024-column slice and a chain
// of three, for M = 512 / 1024 tokens and K = 2048 / 3072. Operands are random
// bytes (timing only). Run only through a binary named gufo* with the fabric
// clock held and GUFO_NPU_DIR pointing at tools/npu.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace rocm = gufo::models::qwen38_flash_next::rocm;

int main() {
  const char* dirp = std::getenv("GUFO_NPU_DIR");
  if (dirp == nullptr) {
    std::fprintf(stderr, "set GUFO_NPU_DIR\n");
    return 2;
  }
  const std::string dir = dirp;
  hipStream_t stream;
  (void)hipStreamCreate(&stream);
  struct Case {
    std::uint32_t m, k;
  };
  const Case cases[] = {{512, 2048}, {512, 3072}, {1024, 2048}, {1024, 3072}};
  for (const Case& cs : cases) {
    std::string error;
    rocm::NpuGemm::Config cfg;
    cfg.xclbin = dir + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
    cfg.insts = dir + "/runtime_mk/insts_" + std::to_string(cs.m) + "x" +
                std::to_string(cs.k) + "x1024_128x64x128.txt";
    cfg.tokens = cs.m;
    cfg.k = cs.k;
    cfg.n = 1024;
    auto npu = rocm::NpuGemm::Create(cfg, &error);
    if (!npu) {
      std::printf("M=%u K=%u: create failed: %s\n", cs.m, cs.k, error.c_str());
      continue;
    }
    const int a = npu->NewBuffer(npu->ABytes(), &error);
    std::vector<int> b, c;
    for (int i = 0; i < 3; ++i) {
      b.push_back(npu->NewBuffer(npu->BBytes(), &error));
      c.push_back(npu->NewBuffer(npu->CBytes(), &error));
    }
    std::mt19937 rng(3);
    auto fill = [&](int id, std::size_t bytes) {
      auto* p = static_cast<std::uint8_t*>(npu->Host(id));
      for (std::size_t i = 0; i < bytes; ++i)
        p[i] = static_cast<std::uint8_t>(rng());
    };
    fill(a, npu->ABytes());
    for (int i = 0; i < 3; ++i) {
      fill(b[i], npu->BBytes());
      npu->FinalizeWeights(b[i]);
    }
    for (int slices : {1, 3}) {
      double best = 1e30;
      for (int run = 0; run < 12; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!npu->Submit(stream, a, b.data(), c.data(), slices)) {
          std::printf("submit failed\n");
          return 1;
        }
        (void)npu->Wait(stream);
        (void)hipStreamSynchronize(stream);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        if (run >= 2)
          best = std::min(best, ms);
      }
      const double flops =
          2.0 * cs.m * cs.k * 1024.0 * slices;
      std::printf(
          "M=%4u K=%u N=%4d: %.3f ms per dispatch (%d slice%s)  %.1f TFLOPS  "
          "A %.2f MB B %.2f MB\n",
          cs.m, cs.k, 1024 * slices, best, slices, slices > 1 ? "s" : "",
          flops / best / 1e9, npu->ABytes() / 1e6, npu->BBytes() / 1e6 * slices);
    }
  }
  return 0;
}
