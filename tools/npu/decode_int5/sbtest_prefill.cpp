// AGENT-7: big-M prefill (ATB) GEMM probe.
//
// Times a full LFM3-1.7B prefill chunk (M rows) through the asymmetric-tile
// buffering GEMM path: 16 layers x (conv/attn projections + fused-FFN pair),
// with synthetic weights. Purpose: establish end-to-end prefill throughput
// before the layer math is wired in.
//
//   sbtest-prefill <programs-dir> [chunk_rows] [iters]
//
// Programs dir must hold:
//   final_atb_k2048.xclbin, final_atb_k7680.xclbin (core ELFs, one per K class)
//   insts_atb_1024x<K>x<N>.txt
#include <cstdio>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "src/npu/runtime/runtime.hpp"
#include "src/npu/runtime/atb_layout.hpp"

using gufo::npu::runtime::NpuRuntime;
using gufo::npu::runtime::Program;
using gufo::npu::runtime::WeightBuffer;

// CPU reference for the ATB GEMM: Y = X @ W^T, X is M x K, W is the model layout
// N x K. So Y[m][n] = sum_k X[m][k] * W[n][k]. AVX2 dot product, parallel over m.
static void RefMatmulXWT(const float* X, const float* W, float* Y,
                         std::uint32_t M, std::uint32_t N, std::uint32_t K) {
#pragma omp parallel for schedule(static)
  for (std::uint32_t m = 0; m < M; ++m) {
    const float* xr = X + std::size_t{m} * K;
    for (std::uint32_t n = 0; n < N; ++n) {
      const float* wn = W + std::size_t{n} * K;
      __m256 acc = _mm256_setzero_ps();
      std::uint32_t k = 0;
      for (; k + 8 <= K; k += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(xr + k), _mm256_loadu_ps(wn + k), acc);
      alignas(32) float t[8];
      _mm256_store_ps(t, acc);
      float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
      for (; k < K; ++k) s += xr[k] * wn[k];
      Y[std::size_t{m} * N + n] = s;
    }
  }
}



int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tools/npu/decode_int5/runtime";
  const std::uint32_t M = argc > 2 ? std::atoi(argv[2]) : 1024;
  const int iters = argc > 3 ? std::atoi(argv[3]) : 3;
  const std::uint32_t mrows = M / 16;
  std::string err;

  using Clock = std::chrono::steady_clock;
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  NpuRuntime rt;  // K = 2048 class
  if (!rt.Init(dir + "/final_atb_k2048.xclbin", &err)) {
    std::fprintf(stderr, "init k2048: %s\n", err.c_str());
    return 1;
  }
  NpuRuntime rt2;  // K = 7680 class (down projection)
  if (!rt2.Init(dir + "/final_atb_k7680.xclbin", &err)) {
    std::fprintf(stderr, "init k7680: %s\n", err.c_str());
    return 1;
  }

  // Real shapes (N multiple of 128 per dispatch; gu = 15360 is split in two).
  struct Real {
    const char* name;
    std::uint32_t K, N;
    bool second;
  };
  const std::vector<Real> ops = {{"conv_in", 2048, 6144, false},
                                 {"conv_out", 2048, 2048, false},
                                 {"qkv", 2048, 3072, false},
                                 {"o", 2048, 2048, false},
                                 {"gu_a", 2048, 8192, false},
                                 {"gu_b", 2048, 7168, false},
                                 {"dn", 7680, 2048, true}};

  const bool skip_dn = std::getenv("GUFO_PREFILL_SKIP_DN") != nullptr;
  std::vector<Program> progs(ops.size());
  std::vector<WeightBuffer> wbs(ops.size());
  std::vector<std::vector<float>> wmat(ops.size());  // model-layout [N][K]
  auto rtp = [&](std::size_t i) -> NpuRuntime& { return ops[i].second ? rt2 : rt; };

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  for (std::size_t i = 0; i < ops.size(); ++i) {
    const auto& o = ops[i];
    if (skip_dn && o.second) continue;
    char insts[256];
    std::snprintf(insts, sizeof(insts), "%s/insts_atb_%ux%ux%u.txt", dir.c_str(), M, o.K, o.N);
    NpuRuntime& r = rtp(i);
    progs[i] = r.MakeProgram(o.K, o.N, insts, std::size_t{M} * o.N / 8 * 9, &err, 0,
                             mrows, /*a_group=*/3, /*c_group=*/5);
    if (!err.empty()) {
      std::fprintf(stderr, "%s: %s\n", o.name, err.c_str());
      return 1;
    }
    // Random weights in model layout [N rows][K cols].
    wmat[i].resize(std::size_t{o.N} * o.K);
    for (auto& v : wmat[i]) v = uni(rng);
    wbs[i] = r.PackWeightAtb(wmat[i].data(), o.N, o.K, /*b_group=*/4);
  }

  std::printf("prefill probe: chunk M=%u, %zu GEMM shapes, %d iters, %d omp threads\n", M,
              ops.size(), iters, omp_get_max_threads());
  std::vector<float> x(std::size_t{M} * 7680), y(std::size_t{M} * 15360, 0.0f);
  for (auto& v : x) v = uni(rng);
  // Isolate the host halves.
  {
    const std::size_t ia = 4;  // gu_a: K=2048 N=8192
    auto t0 = Clock::now();
    for (int k = 0; k < 20; ++k)
      rtp(ia).EncodeAAtb(x.data(), M, ops[ia].K, progs[ia]);
    auto t1 = Clock::now();
    for (int k = 0; k < 20; ++k)
      rtp(ia).DecodeCAtb(y.data(), M, ops[ia].N, progs[ia]);
    auto t2 = Clock::now();
    std::vector<std::uint8_t> dst(std::size_t{M} * ops[ia].N / 8 * 9);
    auto t3 = Clock::now();
    for (int k = 0; k < 20; ++k)
      std::memcpy(dst.data(), progs[ia].ch, dst.size());
    auto t4 = Clock::now();
    std::printf("host: encA %.3f ms  decC %.3f ms  memcpyC %.3f ms (N=%u)\n",
                ms(t0, t1) / 20.0, ms(t1, t2) / 20.0, ms(t3, t4) / 20.0, ops[ia].N);
  }


  // Numerics: compare the NPU output against the CPU reference Y = X @ W^T
  // with random X and W. bfp16 should give a few percent relative L2 error;
  // a wrong B-orientation would give ~sqrt(2) (uncorrelated sums).
  {
    std::vector<float> ref(std::size_t{M} * 15360);
    for (std::size_t i = 0; i < ops.size(); ++i) {
      if (skip_dn && ops[i].second) continue;
      const std::uint32_t K = ops[i].K, N = ops[i].N;
      RefMatmulXWT(x.data(), wmat[i].data(), ref.data(), M, N, K);
      rtp(i).GemmAtb(progs[i], wbs[i], x.data(), M, y.data(), nullptr);
      double num = 0, den = 0, mx = 0;
      for (std::size_t e = 0; e < std::size_t{M} * N; ++e) {
        const double d = y[e] - ref[e];
        num += d * d;
        den += double(ref[e]) * ref[e];
        mx = std::fabs(d) > mx ? std::fabs(d) : mx;
      }
      const double rel = den > 0 ? std::sqrt(num / den) : 1.0;
      std::printf("verify %-9s K=%-5u N=%-6u relL2 %.4f  maxabs %.3f  %s\n", ops[i].name,
                  K, N, rel, mx, rel < 0.08 ? "PASS" : "FAIL");
    }
  }

  // Debug: identity composition. X[m][k] = (m==k), W model-layout identity
  // W[n][k] = (n==k). Then Y[m][n] = sum_k X[m][k] W[n][k] = (m==n).
  // Where the 1s land reveals any A/B/C block-mapping permutation.
  if (std::getenv("GUFO_PREFILL_DEBUG")) {
    const std::size_t i = 3;  // "o": K=N=2048
    const std::uint32_t K = ops[i].K, N = ops[i].N;
    std::fill(x.begin(), x.end(), 0.f);
    for (std::uint32_t m = 0; m < M; ++m) x[std::size_t{m} * K + m] = 1.f;
    std::fill(wmat[i].begin(), wmat[i].end(), 0.f);
    for (std::uint32_t n = 0; n < N; ++n) wmat[i][std::size_t{n} * K + n] = 1.f;
    wbs[i] = rtp(i).PackWeightAtb(wmat[i].data(), N, K, 4);
    rtp(i).GemmAtb(progs[i], wbs[i], x.data(), M, y.data(), nullptr);
    std::printf("identity map (row -> cols with y==1):\n");
    for (std::uint32_t m = 0; m < 12; ++m) {
      std::printf("  m=%u:", m);
      for (std::uint32_t n = 0; n < N; ++n)
        if (std::fabs(y[std::size_t{m} * N + n]) > 0.5f) std::printf(" %u", n);
      std::printf("\n");
    }
    // Debug2: invert the raw stream with the harness reference functions at
    // several L1_block_m hypotheses to find the true C dispatch mapping.
    if (std::atoi(std::getenv("GUFO_PREFILL_DEBUG")) >= 2) {
      // Tagged identity: C[m][m] = m+1 (exact in bfp16ebs8 for m < 128*8).
      std::fill(wmat[i].begin(), wmat[i].end(), 0.f);
      for (std::uint32_t n = 0; n < N; ++n) wmat[i][std::size_t{n} * K + n] = float(n + 1);
      wbs[i] = rtp(i).PackWeightAtb(wmat[i].data(), N, K, 4);
      rtp(i).GemmAtb(progs[i], wbs[i], x.data(), M, y.data(), nullptr);
      const std::size_t nbytes = std::size_t{M} * N / 8 * 9;
      std::vector<std::uint8_t> raw(nbytes);
      std::memcpy(raw.data(), progs[i].ch, nbytes);
      // Decode with the harness reference codec (index i: group = i/8, col-in = i%8).
      auto z = gufo::npu::atb::bfp16ebs8ToFloat(int(nbytes), raw.data(), 0);
      std::printf("raw C stream scan (m -> stream loc):\n");
      std::size_t shown = 0, tot = 0;
      std::FILE* mapf = std::fopen("/tmp/opencode/c_map.txt", "w");
      for (std::size_t i = 0; i < z.size(); ++i) {
        if (std::fabs(z[i]) < 0.5f) continue;
        const std::size_t g = i / 8, cib = i % 8;
        const std::size_t b = g / 8, rib = g % 8;
        const std::size_t m = std::size_t(z[i]) - 1;
        std::fprintf(mapf, "%zu %zu %zu %zu %zu\n", m, b, b / (N / 8), b % (N / 8), rib * 8 + cib);
        ++tot;
      }
      std::fclose(mapf);
      std::printf("  wrote %zu entries to /tmp/opencode/c_map.txt\n", tot);
      return 0;
    }
    return 0;
  }

  std::vector<double> acc(ops.size(), 0.0);

  std::vector<gufo::npu::runtime::NpuRuntime::Timings> tm(ops.size());
  // Real LFM3-1.7B layout: 24 layers, 13 conv (0,1,3,5,...,23) + 11 attention.
  const int layers = 24;
  const bool is_conv[24] = {true, true, false, true, false, true, false, true,
                            false, true, false, true, false, true, false, true,
                            false, true, false, true, false, true, false, true};


  for (int it = 0; it < iters; ++it) {
    const auto t0 = Clock::now();
    for (int l = 0; l < layers; ++l) {
      const auto idx = [&](const char* nm) {
        for (std::size_t i = 0; i < ops.size(); ++i)
          if (!std::strcmp(ops[i].name, nm)) return i;
        return std::size_t{0};
      };
      const std::size_t i1 = is_conv[l] ? idx("conv_in") : idx("qkv");
      const std::size_t i2 = is_conv[l] ? idx("conv_out") : idx("o");
      const std::size_t ig = idx("gu_a"), igb = idx("gu_b"), idn = idx("dn");
      for (std::size_t i : {i1, i2, ig, igb, idn}) {
        if (skip_dn && ops[i].second) continue;
        const auto s0 = Clock::now();
        rtp(i).GemmAtb(progs[i], wbs[i], x.data(), M, y.data(), &tm[i]);
        acc[i] += ms(s0, Clock::now());
      }
    }
    const auto t1 = Clock::now();
    const double secs = ms(t0, t1) / 1000.0;
    std::printf("iter %d: %.2f ms / %u tokens = %.0f tok/s\n", it, secs * 1000.0, M,
                M / secs);
  }
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (skip_dn && ops[i].second) continue;
    const auto& t = tm[i];
    std::printf("  %-9s K=%-5u N=%-6u %7.2f ms  enc %6.2f wait %6.2f dec %6.2f\n",
                ops[i].name, ops[i].K, ops[i].N, acc[i] / iters, t.enc / iters,
                t.wait / iters, t.dec / iters);
  }
  return 0;
}
