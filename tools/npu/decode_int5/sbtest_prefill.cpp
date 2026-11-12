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
#include <omp.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "src/npu/runtime/runtime.hpp"

using gufo::npu::runtime::NpuRuntime;
using gufo::npu::runtime::Program;
using gufo::npu::runtime::WeightBuffer;

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
  auto rtp = [&](std::size_t i) -> NpuRuntime& { return ops[i].second ? rt2 : rt; };

  for (std::size_t i = 0; i < ops.size(); ++i) {
    const auto& o = ops[i];
    if (skip_dn && o.second) continue;
    char insts[256], xcl[256];
    std::snprintf(insts, sizeof(insts), "%s/insts_atb_%ux%ux%u.txt", dir.c_str(), M, o.K, o.N);
    NpuRuntime& r = rtp(i);
    progs[i] = r.MakeProgram(o.K, o.N, insts, std::size_t{M} * o.N / 8 * 9, &err, 0,
                             mrows, /*a_group=*/3, /*c_group=*/5);
    if (!err.empty()) {
      std::fprintf(stderr, "%s: %s\n", o.name, err.c_str());
      return 1;
    }
    // Synthetic weights (timing only).
    std::vector<float> w(std::size_t{o.K} * o.N, 0.0625f);
    wbs[i] = r.PackWeightAtb(w.data(), o.K, o.N, /*b_group=*/4);
  }
  std::printf("prefill probe: chunk M=%u, %zu GEMM shapes, %d iters, %d omp threads\n", M,
              ops.size(), iters, omp_get_max_threads());
  std::vector<float> x(std::size_t{M} * 7680, 1.0f), y(std::size_t{M} * 15360, 0.0f);
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


  // Numerics sanity: with W = 0.0625 (exact in bfp16) and x = 1, every output is
  // K * 0.0625. Any layout/indexing bug shows up as a stray value here.
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (skip_dn && ops[i].second) continue;
    const double want = ops[i].K * 0.0625;
    rtp(i).GemmAtb(progs[i], wbs[i], x.data(), M, y.data(), nullptr);
    std::size_t bad = 0;
    for (std::uint32_t r = 0; r < M; ++r)
      for (std::uint32_t c = 0; c < ops[i].N; ++c)
        if (std::fabs(y[std::size_t(r) * ops[i].N + c] - want) > 1e-3 * want) ++bad;
    std::printf("verify %-9s want %8.2f bad %zu/%u\n", ops[i].name, want, bad,
                M * ops[i].N);
  }
  std::vector<double> acc(ops.size(), 0.0);
  std::vector<gufo::npu::runtime::NpuRuntime::Timings> tm(ops.size());
  const int layers = 16;
  // LFM3: conv at layer 1,2 then alternating conv/attn; 9 conv + 7 attn.
  const bool is_conv[16] = {true, true, false, true, false, true, false, true,
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
