// WP10 bisect: same packed weights through the m64 prefill program once and
// through the m16 program 4x. Compares decoded outputs AND raw on-device C
// wire bytes (device corruption vs host decode indexing).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <random>
#include <string>
#include <vector>

#include "src/npu/runtime/runtime.hpp"

using gufo::npu::runtime::NpuRuntime;

// Compare only the written wire bytes: per 16384 B strip and 4096 B row slice,
// the kernel writes 4 x 576 B at offsets 0..2304.
static int ValidDiff(const std::uint8_t* a, const std::uint8_t* b, std::size_t cbytes) {
  int d = 0;
  for (std::size_t s = 0; s < cbytes / 16384; ++s)
    for (int r = 0; r < 4; ++r)
      for (int g = 0; g < 4 * 576; ++g) {
        const std::size_t o = s * 16384 + r * 4096 + g;
        d += a[o] != b[o];
      }
  return d;
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tools/npu/decode_int5/runtime";
  const std::uint32_t K = 2048, N = 2048, n = 64;
  std::string err;
  NpuRuntime rt;
  if (!rt.Init(dir + "/final_16x64x128_ffn.xclbin", &err)) {
    std::fprintf(stderr, "init: %s\n", err.c_str());
    return 1;
  }
  auto p16 = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048.txt", 0, &err);
  if (!err.empty()) { std::fprintf(stderr, "p16: %s\n", err.c_str()); return 1; }
  auto p64 = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
  if (!err.empty()) { std::fprintf(stderr, "p64: %s\n", err.c_str()); return 1; }

  std::mt19937 gen(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> W(std::size_t(N) * K), X(std::size_t(n) * K);
  for (auto& v : W) v = nd(gen) * 0.02f;
  for (auto& v : X) v = nd(gen);
  auto wb = rt.PackWeight(W.data(), N, K);

  std::vector<float> y64(std::size_t(n) * N), y16(std::size_t(n) * N);
  const std::size_t cbytes = std::size_t(N / 128) * 16384;
  {  // p64 self-consistency: two identical dispatches must produce identical C.
    rt.Gemm(p64, wb, X.data(), n, y64.data());
    std::vector<std::uint8_t> cA(p64.ch, p64.ch + cbytes);
    rt.Gemm(p64, wb, X.data(), n, y64.data());
    int d = 0, first = -1;
    for (std::size_t i = 0; i < cbytes; ++i)
      if (cA[i] != p64.ch[i]) { ++d; if (first < 0) first = (int)i; }
    std::printf("p64 self raw diff: %d bytes first %d (s=%zu r=%d g=%d)\n", d, first,
                first > 0 ? (std::size_t)first / 16384 : 0, first > 0 ? (first % 4096) / 1024 : 0,
                first % 576);
  }
  rt.Gemm(p64, wb, X.data(), n, y64.data());
  std::vector<std::uint8_t> c64(p64.ch, p64.ch + cbytes);
  {  // first-dispatch-is-wrong test on a FRESH p16 and a FRESH p64 + warmup fix
    auto p16c = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048.txt", 0, &err);
    auto p64c = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
    {  // WRONG-C probe: poison p64c.ch, watch every other program's C after one dispatch.
      std::memset(p64c.ch, 0xAA, cbytes);
      std::vector<std::pair<std::string, const std::uint8_t*>> others{{"p64", p64.ch}, {"p16c", p16c.ch}};
      std::vector<std::string> before;
      for (auto& o : others) {
        std::string sig;
        for (int q = 0; q < 64; ++q) sig += std::to_string(o.second[q * 512]);
        before.push_back(sig);
      }
      rt.Gemm(p64c, wb, X.data(), n, y64.data());
      for (std::size_t i = 0; i < others.size(); ++i) {
        std::string sig;
        for (int q = 0; q < 64; ++q) sig += std::to_string(others[i].second[q * 512]);
        if (sig != before[i]) std::printf("WRONG-C: %s CHANGED on fresh p64c dispatch\n", others[i].first.c_str());
      }
      int same = 0;
      for (std::size_t i = 0; i < cbytes; ++i) same += p64c.ch[i] == 0xAA;
      std::printf("WRONG-C: fresh p64c still 0xAA bytes: %d / %zu (0=all written)\n", same, cbytes);
      std::vector<std::uint8_t> d1(p64c.ch, p64c.ch + cbytes);
      std::printf("WRONG-C d1-vs-truth: %d\n", ValidDiff(d1.data(), c64.data(), cbytes));
    }
    p64c = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
    if (getenv("GUFO_OTHER_RUN_BETWEEN")) rt.Gemm(p64, wb, X.data(), n, y64.data());
    rt.Gemm(p64c, wb, X.data(), n, y64.data());  // dispatch 1 (suspect)
    std::vector<std::uint8_t> d1(p64c.ch, p64c.ch + cbytes);
    rt.Gemm(p64c, wb, X.data(), n, y64.data());  // dispatch 2
    std::printf("FRESH d1-vs-d2: %d | d1-vs-truth: %d | d2-vs-truth: %d\n",
                ValidDiff(d1.data(), p64c.ch, cbytes), ValidDiff(d1.data(), c64.data(), cbytes),
                ValidDiff(p64c.ch, c64.data(), cbytes));
    // warmup pattern: dummy dispatch, then measure 8
    rt.Gemm(p64c, wb, X.data(), n, y64.data());
    auto p64d = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
    rt.Gemm(p64d, wb, X.data(), n, y64.data());  // warmup of p64d
    rt.Gemm(p64d, wb, X.data(), n, y64.data());
    std::printf("WARMED fresh p64: %d bytes vs ref\n", ValidDiff(d1.data(), p64d.ch, cbytes));
  }
  {  // alternating p64 programs (engine-like): fresh program each, run same gemm
    auto p64b = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
    auto wb2 = rt.PackWeight(W.data(), N, K);
    std::vector<std::uint8_t> ref;
    int bad = 0, tot = 0;
    for (int t = 0; t < 40; ++t) {
      if (const char* su = getenv("GUFO_SETTLE_US")) usleep(atoi(su));
      rt.Gemm(t % 2 ? p64b : p64, t % 2 ? wb2 : wb, X.data(), n, y64.data());
      if (t == 0) ref.assign(p64.ch, p64.ch + cbytes);
      const std::uint8_t* cur = t % 2 ? p64b.ch : p64.ch;
      const int d = ValidDiff(ref.data(), cur, cbytes);
      ++tot; bad += d ? 1 : 0;
      std::printf("alt t%d: %d\n", t, d);
    }
    std::printf("ALT: %d/%d corrupt\n", bad, tot);
    // p16-before-p64 alternation
    bad = 0;
    for (int t = 0; t < 8; ++t) {
      rt.Gemm(p16, wb, X.data(), 16, y16.data());
      rt.Gemm(p64, wb, X.data(), n, y64.data());
      bad += ValidDiff(ref.data(), p64.ch, cbytes) ? 1 : 0;
    }
    std::printf("P16BEFORE: %d/8 corrupt\n", bad);
  }
  {  // 10x p64 self-consistency
    rt.Gemm(p64, wb, X.data(), n, y64.data());
    std::vector<std::uint8_t> ref(p64.ch, p64.ch + cbytes);
    int bad = 0;
    for (int t = 0; t < 10; ++t) {
      rt.Gemm(p64, wb, X.data(), n, y64.data());
      const int d = ValidDiff(ref.data(), p64.ch, cbytes);
      if (d) ++bad;
      std::printf("t%d: %d\n", t, d);
    }
    std::printf("p64 self: %d/10 corrupt\n", bad);
  }
  rt.Gemm(p64, wb, X.data(), n, y64.data());
  {  // warmup FIX test: fresh p64e, dummy dispatch, real dispatch, compare to warm p64
    auto p64e = rt.MakeProgram(K, N, dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
    if (getenv("GUFO_OTHER_RUN_BETWEEN")) rt.Gemm(p64, wb, X.data(), n, y64.data());
    rt.Gemm(p64e, wb, X.data(), n, y64.data());   // warmup (throwaway)
    rt.Gemm(p64e, wb, X.data(), n, y64.data());   // measured
    std::printf("WARMUP-FIX p64e 2nd-run vs warm p64 C: %d bytes\n",
                ValidDiff(c64.data(), p64e.ch, cbytes));
  }

  for (std::uint32_t mb = 0; mb < 4; ++mb) {
    rt.Gemm(p16, wb, X.data() + std::size_t(mb) * 16 * K, 16,
            y16.data() + std::size_t(mb) * 16 * N);
    // raw wire compare: m64 holds mb's 576 B wire at mb*576 inside each 4096 B
    // aie-row slice; p16 holds its rows at offset 0 of the same slice.
    int d = 0, first_s = -1, first_r = -1, first_g = -1;
    for (std::size_t s = 0; s < N / 128; ++s)
      for (std::uint32_t r = 0; r < 4; ++r)
        for (int g = 0; g < 576; ++g) {
          const std::uint8_t a = c64[s * 16384 + r * 4096 + mb * 576 + g];
          const std::uint8_t b = p16.ch[s * 16384 + r * 4096 + g];
          if (a != b && ++d == 1) { first_s = (int)s; first_r = (int)r; first_g = g; }
        }
    std::printf("mb%u raw C wire diff: %d bytes%s%s%s", mb, d, d ? " first " : "",
                d ? (std::to_string(first_s) + "/" + std::to_string(first_r) + "/" +
                     std::to_string(first_g)).c_str()
                  : "",
                "\n");
  }

  {
    FILE* f = fopen("/tmp/opencode/y64.bin", "wb");
    fwrite(y64.data(), 4, y64.size(), f); fclose(f);
    f = fopen("/tmp/opencode/y16.bin", "wb");
    fwrite(y16.data(), 4, y16.size(), f); fclose(f);
  }
  double worst = 0;
  std::size_t worst_i = 0;
  for (std::size_t i = 0; i < y64.size(); ++i) {
    const double d = std::fabs(y64[i] - y16[i]);
    if (d > worst) { worst = d; worst_i = i; }
  }
  double num = 0, den = 0;
  for (std::size_t i = 0; i < y64.size(); ++i) {
    num += (double)(y64[i] - y16[i]) * (y64[i] - y16[i]);
    den += (double)y16[i] * y16[i];
  }
  std::printf("m64 vs m16: rel L2 %g max abs %g at row %zu col %zu (y64 %g y16 %g)\n",
              std::sqrt(num / den), worst, worst_i / N, worst_i % N, y64[worst_i], y16[worst_i]);
  static int rc[4][16];
  for (auto& row : rc) for (auto& v : row) v = 0;
  for (std::size_t i = 0; i < y64.size(); ++i)
    if (std::fabs(y64[i] - y16[i]) > 0.01)
      ++rc[(i / N) / 16][(i % N) / 128];
  for (int mb = 0; mb < 4; ++mb) {
    std::printf("mb%d:", mb);
    for (int s = 0; s < 16; ++s) std::printf(" %3d", rc[mb][s]);
    std::printf("\n");
  }
  return 0;
}
