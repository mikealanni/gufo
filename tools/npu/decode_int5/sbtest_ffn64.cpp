// WP10 bisect: Ffn64 (gu64 + C-shuffle + dn64) vs the fused m16 Ffn on the SAME
// weights and inputs. Any mismatch is in the m64 FFN path (shuffle or programs).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "src/npu/runtime/runtime.hpp"

using namespace gufo::npu::runtime;

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tools/npu/decode_int5/runtime";
  std::string err;
  NpuRuntime rt;
  if (!rt.Init(dir + "/final_16x64x128_ffn.xclbin", &err)) {
    std::fprintf(stderr, "init: %s\n", err.c_str());
    return 1;
  }
  // Programs (mirrors engine Create).
  auto p_ffn = rt.MakeProgram(2048, 2048, dir + "/insts_ffn.txt",
                              std::size_t{136} * kStripBytes, &err);
  if (!err.empty()) { std::fprintf(stderr, "p_ffn: %s\n", err.c_str()); return 1; }
  auto p_gu64 = rt.MakeProgram(2048, 15360, dir + "/insts_ffngu64.txt", 0, &err, 0, 4);
  if (!err.empty()) { std::fprintf(stderr, "p_gu64: %s\n", err.c_str()); return 1; }
  auto p_dn64 = rt.MakeProgram(7680, 2048, dir + "/insts_plain_7680x2048_m64.txt", 0, &err, 0, 4);
  if (!err.empty()) { std::fprintf(stderr, "p_dn64: %s\n", err.c_str()); return 1; }

  std::mt19937 gen(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> G(std::size_t(15360) * 2048), D(std::size_t(2048) * 7680);
  for (auto& v : G) v = nd(gen) * 0.02f;
  for (auto& v : D) v = nd(gen) * 0.02f;

  // Fused weight buffer exactly like the engine: PackMatInt5 gu then down at gu_bytes.
  const std::size_t gu_bytes = std::size_t{120} * 32 * 4 * kQuarterBytes;
  const std::size_t dn_bytes = std::size_t{16} * 120 * 4 * kQuarterBytes;
  WeightBuffer m;
  m.bytes = gu_bytes + dn_bytes;
  m.bo = xrt::bo(rt.device(), m.bytes, XRT_BO_FLAGS_HOST_ONLY, rt.kernel().group_id(3));
  m.host = m.bo.map<std::uint8_t*>();
  std::memset(m.host, 0, m.bytes);
  PackMatInt5(G.data(), 15360, 2048, m.host);
  PackMatInt5(D.data(), 2048, 7680, m.host + gu_bytes);
  m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto wb_dn = rt.PackWeight(D.data(), 2048, 7680);

  const std::uint32_t n = 64;
  std::vector<float> X(std::size_t(n) * 2048);
  for (auto& v : X) v = nd(gen);
  std::vector<float> y16(std::size_t(n) * 2048, 0.f), y64(std::size_t(n) * 2048, 0.f);
  rt.Ffn(p_ffn, m, X.data(), 16, y16.data());
  rt.Ffn64(m, p_gu64, wb_dn, p_dn64, X.data(), n, y64.data());
  const std::vector<std::uint8_t> c1(p_gu64.ch, p_gu64.ch + std::size_t(120) * 16384);
  if (getenv("GUFO_SEQ2")) {  // minimal: Ffn64 twice back-to-back, same weights.
    std::vector<float> y64b(std::size_t(n) * 2048, 0.f);
    rt.Ffn64(m, p_gu64, wb_dn, p_dn64, X.data(), n, y64b.data());
    double s0 = 0, sden = 0;
    for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) { s0 += double(y64b[i] - y64[i]) * (y64b[i] - y64[i]); sden += double(y64[i]) * y64[i]; }
    std::printf("SEQ2 same-weights Ffn64 1 vs 2: rel %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
    int dup = 0;
    for (std::size_t i = 0; i < std::size_t(120) * 16384; ++i) dup += c1[i] != p_gu64.ch[i];
    std::printf("SEQ2 gu C 1 vs 2: %d diff\n", dup);
    auto cmp = [&](const std::vector<float>& a, const std::vector<float>& b, const char* tag) {
      double s0 = 0, sden = 0;
      for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) { s0 += double(a[i] - b[i]) * (a[i] - b[i]); sden += double(a[i]) * a[i]; }
      std::printf("%s: rel %.6f\n", tag, std::sqrt(s0 / (sden + 1e-30)));
    };
    std::vector<float> y16b(std::size_t(n) * 2048, 0.f);
    // (a) dn weights change, gu unchanged.
    for (auto& v : D) v = nd(gen) * 0.02f;
    PackMatInt5(D.data(), 2048, 7680, m.host + gu_bytes);
    m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto wb_dn2 = rt.PackWeight(D.data(), 2048, 7680);
    rt.Ffn(p_ffn, m, X.data(), 16, y16b.data());
    auto ah1 = std::make_unique<std::uint8_t[]>(552960);
    std::memcpy(ah1.get(), p_dn64.ah, 552960);  // staged A of call#1
    rt.Ffn64(m, p_gu64, wb_dn2, p_dn64, X.data(), n, y64b.data());
    {
      int dd = 0;
      for (int i = 0; i < 552960; ++i) dd += ah1[i] != p_dn64.ah[i];
      std::printf("DN-change: p_dn.ah diff bytes vs call#1: %d\n", dd);
      auto dec = [&](int row, int col) -> float {
        const int t = col / 64, c64 = col % 64, mb = row >> 4, rb = row & 15;
        const int sc = c64 / 8, i8 = c64 % 8;
        const int gi = (rb & 7) + ((rb >> 3) & 1) * 8 + sc * 16;
        const std::uint8_t* g = p_dn64.ah + (std::size_t(t) * 4 + mb) * 1152 + std::size_t(gi) * 9;
        float v[8];
        DecGroup(g, v);
        return v[i8];
      };
      double s0 = 0, sden = 0;
      for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 2048; ++c) {
          double acc = 0;
          for (int j = 0; j < 7680; ++j) acc += double(dec(r, j)) * double(D[std::size_t(c) * 7680 + j]);
          const double b2 = y64b[std::size_t(r) * 2048 + c];
          s0 += (acc - b2) * (acc - b2);
          sden += acc * acc;
        }
      std::printf("DN-change: y64b vs CPU A@D^T: rel %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
    }
    cmp(y64, y64b, "DN-change call2 vs call1 y (0 => down used STALE weights)");
    {  // plain-gemm rebind test on m64 programs with a FRESH weight BO.
      auto p16r = rt.MakeProgram(2048, 2048, dir + "/insts_plain_2048x2048.txt", 0, &err);
      auto p64r = rt.MakeProgram(2048, 2048, dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
      std::vector<float> Wc(std::size_t(2048) * 2048);
      for (auto& v : Wc) v = nd(gen) * 0.02f;
      auto wbc = rt.PackWeight(Wc.data(), 2048, 2048);
      std::vector<float> xa(std::size_t(n) * 2048);
      for (auto& v : xa) v = nd(gen);
      std::vector<float> y64r(std::size_t(n) * 2048, 0.f), y16r(std::size_t(n) * 2048, 0.f);
      rt.Gemm(p16r, wbc, xa.data(), 16, y16r.data());  // warm p16r
      rt.Gemm(p64r, wbc, xa.data(), n, y64r.data());   // warm p64r
      rt.Gemm(p64r, wbc, xa.data(), n, y64r.data());
      rt.Gemm(p16r, wbc, xa.data(), 16, y16r.data());
      cmp(y16r, y64r, "WARMED p64 vs p16 (expect 0)");
      // fresh BO, both programs warm:
      std::vector<float> Wd(std::size_t(2048) * 2048);
      for (auto& v : Wd) v = nd(gen) * 0.02f;
      auto wbd = rt.PackWeight(Wd.data(), 2048, 2048);
      rt.Gemm(p64r, wbd, xa.data(), n, y64r.data());
      rt.Gemm(p16r, wbd, xa.data(), 16, y16r.data());
      cmp(y16r, y64r, "REBIND-BETWEEN-dispatch p64 vs p16 (expect 0)");
    }
    {  // fresh down program, same weights: p_dn64 state after reuse suspect
      auto p_dnB = rt.MakeProgram(7680, 2048, dir + "/insts_plain_7680x2048_m64.txt", 0, &err, 0, 4);
      std::vector<float> y64d(std::size_t(n) * 2048, 0.f);
      rt.Ffn64(m, p_gu64, wb_dn2, p_dnB, X.data(), n, y64d.data());
      cmp(y16b, y64d, "DN-change FRESH p_dn: Ffn64 vs Ffn");
      int dd2 = 0;
      for (int i = 0; i < 552960; ++i) dd2 += p_dn64.ah[i] != p_dnB.ah[i];
      std::printf("DN-change: p_dn64.ah vs fresh p_dnB.ah diff: %d\n", dd2);
      std::vector<float> y64e(std::size_t(n) * 2048, 0.f);
      rt.Ffn64(m, p_gu64, wb_dn2, p_dnB, X.data(), n, y64e.data());
      cmp(y64d, y64e, "fresh p_dn twice (expect 0)");
      // fresh m16 fused program on repacked m: is STALE p_ffn the wrong one?
      auto p_ffn2 = rt.MakeProgram(2048, 2048, dir + "/insts_ffn.txt",
                                  std::size_t{136} * kStripBytes, &err);
      rt.Ffn(p_ffn2, m, X.data(), 16, y64d.data());  // warm
      rt.Ffn(p_ffn2, m, X.data(), 16, y64d.data());
      std::printf("DN-change: fresh-p_ffn vs old-p_ffn y16: ");
      cmp(y16b, y64d, "->");
      rt.Ffn64(m, p_gu64, wb_dn2, p_dn64, X.data(), n, y64d.data());
      cmp(y64d, y16b, "FRESH-p_ffn vs Ffn64 (expect 0 if Ffn64 right)");
    }
    {  // gu program got same weights again: its C must equal call#1 C in valid region.
      int v = 0, tot = 0;
      for (std::size_t sN = 0; sN < 120; ++sN)
        for (int r = 0; r < 4; ++r)
          for (int mb = 0; mb < 4; ++mb)
            for (int b = 0; b < 288; ++b) {
              ++tot;
              const std::size_t o = sN * 16384 + r * 4096 + mb * 288 + b;  // per DecodeC?? valid layout
              v += c1[o] != p_gu64.ch[o];
            }
      std::printf("DN-change: gu C valid-byte diff vs call#1: %d/%d\n", v, tot);
    }
    // (b) gu weights change, dn unchanged.
    for (auto& v : G) v = nd(gen) * 0.02f;
    PackMatInt5(G.data(), 15360, 2048, m.host);
    m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    rt.Ffn(p_ffn, m, X.data(), 16, y16b.data());
    rt.Ffn64(m, p_gu64, wb_dn2, p_dn64, X.data(), n, y64b.data());
    cmp(y16b, y64b, "GU-change Ffn64 vs Ffn");
    return 0;
  }

  // Compare rows 0..15 (both valid), then row structure of 16..63 vs self-consistency.
  double num = 0, den = 0;
  int badrow = -1;
  for (std::uint32_t r = 0; r < 16; ++r)
    for (std::uint32_t c = 0; c < 2048; ++c) {
      const double a = y16[std::size_t(r) * 2048 + c], b = y64[std::size_t(r) * 2048 + c];
      num += (a - b) * (a - b);
      den += a * a;
      if (badrow < 0 && std::fabs(a - b) > 1e-2 * (std::fabs(a) + 1e-6)) badrow = int(r * 2048 + c);
    }
  std::printf("Ffn64 vs Ffn rows0-15: rel %.6f first-bad %d\n", std::sqrt(num / (den + 1e-30)), badrow);

  // SECOND call with DIFFERENT weights (engine-like sequence): catches stale-arg reuse.
  for (auto& v : G) v = nd(gen) * 0.02f;
  for (auto& v : D) v = nd(gen) * 0.02f;
  PackMatInt5(G.data(), 15360, 2048, m.host);
  PackMatInt5(D.data(), 2048, 7680, m.host + gu_bytes);
  m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto wb_dn2 = rt.PackWeight(D.data(), 2048, 7680);
  std::vector<float> y16b(std::size_t(n) * 2048, 0.f), y64b(std::size_t(n) * 2048, 0.f);
  rt.Ffn(p_ffn, m, X.data(), 16, y16b.data());
  rt.Ffn64(m, p_gu64, wb_dn2, p_dn64, X.data(), n, y64b.data());
  const std::size_t gub = std::size_t(120) * 16384;
  std::vector<std::uint8_t> c2(p_gu64.ch, p_gu64.ch + gub);
  int dup = 0;
  for (std::size_t i = 0; i < gub; ++i) dup += c2[i] != c1[i];
  std::printf("gu C call#2 vs call#1: %d diff bytes (stale check)\n", dup);
  double s0 = 0, sden = 0;
  for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) { s0 += double(y64b[i] - y64[i]) * (y64b[i] - y64[i]); sden += double(y64[i]) * y64[i]; }
  std::printf("y64 call#2 vs call#1 (A-weights? rel): %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
  num = 0; den = 0;
  for (std::uint32_t r = 0; r < 16; ++r)
    for (std::uint32_t c = 0; c < 2048; ++c) {
      const double a2 = y16b[std::size_t(r) * 2048 + c], b2 = y64b[std::size_t(r) * 2048 + c];
      num += (a2 - b2) * (a2 - b2);
      den += a2 * a2;
    }
  std::printf("Ffn64 call#2 vs Ffn: rel %.6f\n", std::sqrt(num / (den + 1e-30)));

  // call#3 + m16 self-consistency to attribute the breakage.
  std::vector<float> y16c(std::size_t(n) * 2048, 0.f), y64c(std::size_t(n) * 2048, 0.f);
  rt.Ffn(p_ffn, m, X.data(), 16, y16c.data());
  double s1 = 0, s2 = 0;
  for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) { s1 += double(y16b[i] - y16c[i]) * (y16b[i] - y16c[i]); s2 += double(y16c[i]) * y16c[i]; }
  std::printf("Ffn(m16) call2 vs call3: rel %.6f\n", std::sqrt(s1 / (s2 + 1e-30)));
  rt.Ffn64(m, p_gu64, wb_dn2, p_dn64, X.data(), n, y64c.data());
  s1 = 0; s2 = 0;
  for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) { s1 += double(y64c[i] - y16c[i]) * (y64c[i] - y16c[i]); s2 += double(y16c[i]) * y16c[i]; }
  std::printf("Ffn64 call#3 vs Ffn call#3: rel %.6f\n", std::sqrt(s1 / (s2 + 1e-30)));
  // row0 exactness of gu C shuffle coverage: which mb blocks touched
  double num2 = 0, den2 = 0;
  for (std::size_t i = std::size_t(16) * 2048; i < y64.size(); ++i) num2 += double(y64[i]) * y64[i];
  std::printf("y64 rows16-63 energy: %.3f (expect >0)\n", std::sqrt(num2 / (y64.size() - 16 * 2048)));
  return 0;
}
