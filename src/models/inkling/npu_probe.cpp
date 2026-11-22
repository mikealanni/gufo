// NPU feasibility probe for Inkling dense projections: attn_q of layer 0 as
// three 1024-column slices of a K=2304 runtime-M/K block-float GEMM, checked
// against a CPU reference and timed. Run only through a binary named gufo* with
// the fabric clock held and GUFO_NPU_DIR pointing at tools/npu (holding
// runtime_mk/).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace rocm = gufo::models::qwen38_flash_next::rocm;

namespace {
constexpr std::uint32_t kTokens = 2048;
constexpr std::uint32_t kK = 2304;
constexpr std::uint32_t kSlice = 1024;
constexpr std::uint32_t kSlices = 3;
constexpr int kRuns = 10;
constexpr int kCheckTokens = 4;
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::getenv("GUFO_NPU_DIR") == nullptr) {
    std::fprintf(stderr, "usage: GUFO_NPU_DIR=<tools/npu> %s model.gguf\n",
                 argv[0]);
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader)
    return 1;
  const auto* wq = reader->FindTensor("blk.0.attn_q.weight");
  if (wq == nullptr || wq->type != gufo::core::GgmlType::kQ8_0)
    return 1;
  const std::uint32_t rows = static_cast<std::uint32_t>(wq->dimensions[1]);

  const std::string dir = std::getenv("GUFO_NPU_DIR");
  rocm::NpuGemm::Config cfg;
  cfg.xclbin = dir + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
  cfg.insts = dir + "/runtime_mk/insts_2048x2304x1024_128x64x128.txt";
  cfg.tokens = kTokens;
  cfg.k = kK;
  cfg.n = kSlice;
  auto npu = rocm::NpuGemm::Create(cfg, &error);
  if (!npu) {
    std::fprintf(stderr, "NpuGemm::Create failed: %s\n", error.c_str());
    return 1;
  }
  hipStream_t stream;
  (void)hipStreamCreate(&stream);

  void* d_q8 = nullptr;
  (void)hipMalloc(&d_q8, wq->size_bytes);
  (void)hipMemcpy(d_q8, wq->data, wq->size_bytes, hipMemcpyHostToDevice);

  const int a_id = npu->NewBuffer(npu->ABytes(), &error);
  const int a_lo = npu->NewBuffer(npu->ABytes(), &error);
  std::vector<int> b_ids;
  std::vector<int> c_ids;
  std::vector<int> c_lo;
  for (std::uint32_t i = 0; i < kSlices; ++i) {
    const int b = npu->NewBuffer(npu->BBytes(), &error);
    rocm::NpuEncodeWeights(d_q8, kK, i * kSlice, kSlice, kK, 0, kK,
                           npu->Device(b), stream);
    b_ids.push_back(b);
    c_ids.push_back(npu->NewBuffer(npu->CBytes(), &error));
    c_lo.push_back(npu->NewBuffer(npu->CBytes(), &error));
  }
  (void)hipStreamSynchronize(stream);
  for (const int b : b_ids)
    npu->FinalizeWeights(b);

  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<__half> host_x(static_cast<std::size_t>(kTokens) * kK);
  for (auto& v : host_x)
    v = __float2half(dist(rng));
  __half* d_x = nullptr;
  (void)hipMalloc(&d_x, host_x.size() * sizeof(__half));
  (void)hipMemcpy(d_x, host_x.data(), host_x.size() * sizeof(__half),
                  hipMemcpyHostToDevice);
  float* d_out = nullptr;
  (void)hipMalloc(&d_out, static_cast<std::size_t>(kTokens) * kSlice * kSlices *
                              sizeof(float));

  double best_ms = 1e30;
  for (int run = 0; run < kRuns; ++run) {
    const auto t0 = std::chrono::steady_clock::now();
    rocm::NpuEncodeActivations(d_x, kK, kTokens, kK, 0, kK, npu->Device(a_id),
                               stream);
    if (!npu->Submit(stream, a_id, b_ids.data(), c_ids.data(),
                     static_cast<int>(kSlices))) {
      std::fprintf(stderr, "submit failed\n");
      return 1;
    }
    (void)npu->Wait(stream);
    for (std::uint32_t i = 0; i < kSlices; ++i) {
      rocm::NpuDecodeOutput(npu->Staged(stream, c_ids[i]), d_out, kTokens,
                            kSlice, kSlice * kSlices, i * kSlice, false,
                            stream);
    }
    (void)hipStreamSynchronize(stream);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    best_ms = std::min(best_ms, ms);
    std::printf("run %d: %.2f ms (encode+3 slices+wait+decode)\n", run, ms);
  }

  auto evaluate = [&](const char* label) {
    std::vector<float> out(static_cast<std::size_t>(kTokens) * kSlice *
                           kSlices);
    (void)hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    // CPU reference: Q8_0 rows, F16 activations, first kCheckTokens tokens
    const auto* q8 = static_cast<const std::uint8_t*>(wq->data);
    double max_rel = 0.0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    const std::size_t row_bytes = kK / 32 * 34;
    for (std::uint32_t r = 0; r < kSlice * kSlices && r < rows; ++r) {
      std::vector<float> w(kK);
      for (std::uint32_t b = 0; b < kK / 32; ++b) {
        const std::uint8_t* blk = q8 + r * row_bytes + b * 34;
        __half hh;
        std::memcpy(&hh, blk, 2);
        const float d = __half2float(hh);
        for (int i = 0; i < 32; ++i)
          w[b * 32 + i] =
              d * static_cast<float>(static_cast<std::int8_t>(blk[2 + i]));
      }
      for (int t = 0; t < kCheckTokens; ++t) {
        double acc = 0.0;
        for (std::uint32_t k = 0; k < kK; ++k)
          acc += static_cast<double>(w[k]) *
                 __half2float(host_x[static_cast<std::size_t>(t) * kK + k]);
        const double got =
            out[static_cast<std::size_t>(t) * kSlice * kSlices + r];
        sq_err += (got - acc) * (got - acc);
        sq_ref += acc * acc;
        max_rel =
            std::max(max_rel, std::fabs(got - acc) / (std::fabs(acc) + 1.0));
      }
    }
    std::printf("%s: rel RMS error %.5f; max rel %.4f\n", label,
                std::sqrt(sq_err / sq_ref), max_rel);
  };
  // hi pass only, then hi+lo
  for (std::uint32_t i = 0; i < kSlices; ++i) {
    rocm::NpuDecodeOutput(npu->Staged(stream, c_ids[i]), d_out, kTokens, kSlice,
                          kSlice * kSlices, i * kSlice, false, stream);
  }
  (void)hipStreamSynchronize(stream);
  evaluate("hi only");
  for (std::uint32_t i = 0; i < kSlices; ++i) {
    rocm::NpuDecodeOutput(npu->Staged(stream, c_lo[i]), d_out, kTokens, kSlice,
                          kSlice * kSlices, i * kSlice, true, stream);
  }
  (void)hipStreamSynchronize(stream);
  evaluate("hi + lo");
  const double flops = 2.0 * kTokens * kK * kSlice * kSlices;
  std::printf("best %.2f ms = %.2f TFLOPs\n", best_ms, flops / best_ms / 1e9);
  return 0;
}
