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
#include <thread>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <memory>
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

  // ----- AGENT-8: swiglu-in-gu-flush (cfg28) chained probe -----------------
  // gu programs (stage N=8192/7168 -> act N=4096/3584, C half-width on device)
  // -> byte-shuffle act wire into dn A -> plain dn GEMM. Verifies act against
  // CPU silu(X.G^T)*(X.U^T) and dn(act) against act.Wd^T. Scaffold shuffle is
  // host-side; it proves the wire formats chain so the copy can move to the
  // device (or be elided by tap permutation) later.
  if (std::getenv("GUFO_PREFILL_SWI")) {
    NpuRuntime rt3;  // swiglu-flush program (N=8192 design; mode 6 also has _b)
    if (!rt3.Init(dir + "/final_atb_swi_k2048.xclbin", &err)) {
      std::fprintf(stderr, "init swi: %s\n", err.c_str());
      return 1;
    }
    const std::uint32_t HID = 2048, INT = 7680;
    const std::uint32_t na = 4096, nb = 3584;  // act halves (stage 8192 / 7168)
    const int swimode = std::atoi(std::getenv("GUFO_PREFILL_SWI"));
    // mode 6 = FULLC xclbin (kernel emits act twice into a FULL-width C wire):
    // first half of the C buffer is the clean halved act stream.
    const std::size_t cfmul = swimode >= 6 ? 2 : 1;
    // mode 6 artifacts are per-N (full-width C does not fit a 15360 design),
    // so the 7168 program runs on a second runtime instance.
    std::unique_ptr<NpuRuntime> rt3b;
    if (swimode >= 6 || swimode == 1) {  // per-N swi xclbins (stage 8192 / 7168)
      rt3b = std::make_unique<NpuRuntime>();
      if (!rt3b->Init(dir + "/final_atb_swi_b_k2048.xclbin", &err)) {
        std::fprintf(stderr, "init swi_b: %s\n", err.c_str());
        return 1;
      }
    }
    NpuRuntime& rpb = rt3b ? *rt3b : rt3;
    char pth[512];
    std::snprintf(pth, sizeof pth, "%s/insts_atb_swi_%ux2048x8192.txt", dir.c_str(), M);
    Program pa =
        rt3.MakeProgram(HID, 2 * na, pth, cfmul * std::size_t{M} * na / 8 * 9, &err, 0, mrows, 3,
                        5);
    std::snprintf(pth, sizeof pth, "%s/insts_atb_swi_%ux2048x7168.txt", dir.c_str(), M);
    Program pb =
        rpb.MakeProgram(HID, 2 * nb, pth, cfmul * std::size_t{M} * nb / 8 * 9, &err, 0, mrows, 3,
                        5);
    std::snprintf(pth, sizeof pth, "%s/insts_atb_%ux7680x2048.txt", dir.c_str(), M);
    Program pdn = rt2.MakeProgram(INT, HID, pth, std::size_t{M} * HID / 8 * 9, &err, 0, mrows, 3, 5);
    if (!err.empty()) {
      std::fprintf(stderr, "swi programs: %s\n", err.c_str());
      return 1;
    }

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> xs(std::size_t{M} * HID), G(std::size_t{INT} * HID),
        U(std::size_t{INT} * HID), Wd(std::size_t{HID} * INT);
    for (auto& v : xs) v = uni(rng);
    for (auto& v : G) v = uni(rng);
    for (auto& v : U) v = uni(rng);
    for (auto& v : Wd) v = uni(rng);
    if (swimode == 2) {  // act == 0 expected: G = U = 0
      std::fill(G.begin(), G.end(), 0.f);
      std::fill(U.begin(), U.end(), 0.f);
    }

    // cfg28 stage pairing: C-stream units are 8x8 blocks ordered
    // (sbr,sbc,br,bc); consecutive vecs (bc0,bc1) cover stage cols
    // 16*sbc+0..7 and 8..15 -> gate/up must interleave at 8-col granularity:
    // stage row s = 16q+r: r<8 gate row of model col c0+8q+r, r>=8 up row.
    auto pack_swi = [&](std::uint32_t c0, std::uint32_t n) {
      std::vector<float> ws(std::size_t{2} * n * HID);
      for (std::uint32_t s = 0; s < 2 * n; ++s) {
        const std::uint32_t q = s >> 4, r = s & 15;
        const float* src =
            (r < 8 ? G.data() : U.data()) + std::size_t{c0 + q * 8 + (r & 7)} * HID;
        std::memcpy(ws.data() + std::size_t{s} * HID, src, std::size_t{HID} * 4);
      }
      return ws;
    };
    const std::vector<float> wsa = pack_swi(0, na), wsb = pack_swi(na, nb);
    WeightBuffer wa = rt3.PackWeightAtb(wsa.data(), 2 * na, HID, 4);
    WeightBuffer wwb = rpb.PackWeightAtb(wsb.data(), 2 * nb, HID, 4);
    WeightBuffer wd = rt2.PackWeightAtb(Wd.data(), HID, INT, 4);

    // A is identical for both swi programs: encode once, mirror the stream.
    rt3.EncodeAAtb(xs.data(), M, HID, pa);
    std::memcpy(pb.ah, pa.ah, std::size_t{M} * HID / 8 * 9);

    auto dispatch = [&](NpuRuntime& r, Program& p, WeightBuffer& w) {
      WeightBuffer& ww = r.WeightSlot(w, p);
      p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      ww.run.start();
      ww.run.wait();
      if (!p.warmed) {  // first C write is lost on this stack
        p.warmed = true;
        ww.run.start();
        ww.run.wait();
      }
      p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    };
    const auto t_swi0 = Clock::now();
    dispatch(rt3, pa, wa);
    dispatch(rpb, pb, wwb);
    const double swi_ms = ms(t_swi0, Clock::now());

    // Diagnostic: scale-byte histogram of act streams + reference hexdump of a
    // VERIFIED plain C stream (plain gu_a on rt) for byte-pattern comparison.
    if (true) {
      int hist[256] = {0};
      const std::size_t ba = std::size_t{M} * na / 8 * 9, bb = std::size_t{M} * nb / 8 * 9;
      for (std::size_t b = 0; b < ba / 9; ++b) hist[pa.ch[b * 9]]++;
      for (std::size_t b = 0; b < bb / 9; ++b) hist[pb.ch[b * 9]]++;
      std::printf("act scale-byte hist (top):");
      for (int pass = 0; pass < 4; ++pass) {
        int best = -1;
        for (int e = 0; e < 256; ++e)
          if (hist[e] > 0 && (best < 0 || hist[e] > hist[best])) best = e;
        if (best < 0) break;
        std::printf(" e%d:%d", best, hist[best]);
        hist[best] = 0;
      }
      std::printf("\n");
    }

    // Diagnostic: is the act wire non-zero at all, and where?
    {
      auto scan = [](const std::uint8_t* p, std::size_t n) {
        std::size_t nz = 0;
        for (std::size_t i = 0; i < n; ++i) nz += p[i] != 0;
        return nz;
      };
      const std::size_t ba = std::size_t{M} * na / 8 * 9, bb = std::size_t{M} * nb / 8 * 9;
      std::printf("act wire: A %zu/%zu nonzero bytes  B %zu/%zu\n", scan(pa.ch, ba), ba,
                  scan(pb.ch, bb), bb);
      std::printf("act A raw hex (first 72 B):");
      for (int i = 0; i < 72; ++i) std::printf(" %02x", pa.ch[i]);
      std::printf("\nact B raw hex (first 72 B):");
      for (int i = 0; i < 72; ++i) std::printf(" %02x", pb.ch[i]);
      std::printf("\n");
    }
    // Plain (verified) gu_a C for byte-pattern reference.
    if (true) {
      char ip[512];
      std::snprintf(ip, sizeof ip, "%s/insts_atb_%ux2048x8192.txt", dir.c_str(), M);
      Program pg =
          rt.MakeProgram(HID, 2 * na, ip, std::size_t{M} * 2 * na / 8 * 9, &err, 0, mrows, 3, 5);
      rt.EncodeAAtb(xs.data(), M, HID, pg);
      WeightBuffer wg = rt.PackWeightAtb(wsa.data(), 2 * na, HID, 4);
      WeightBuffer& wwg = rt.WeightSlot(wg, pg);
      pg.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      wwg.run.start();
      wwg.run.wait();
      wwg.run.start();
      wwg.run.wait();
      pg.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      {
        const std::size_t nb2 = std::size_t{M} * 2 * na / 8 * 9;
        std::size_t nd = 0;
        for (std::size_t i = 0; i < nb2; ++i) nd += (pg.ch[i] != pa.ch[i]);
        std::printf("plain-gu C vs swi-act wire BYTES DIFF: %zu of %zu\n", nd, nb2);
      }
      std::printf("plain gu C raw hex (first 72 B):");
      for (int i = 0; i < 72; ++i) std::printf(" %02x", pg.ch[i]);
      std::printf("\n");
      if (swimode == 2) return 0;
    }
    if (swimode == 2) return 0;

    // act wire -> logical act[M][INT] (cfg28 stage-pairing tile map):
    // program P, act col jp, row r -> v8 block
    //   o64=jp&63, bcb=o64>>3, mp=r>>9, rcore=(r>>7)&3, r_in=r&127,
    //   s=64*((r_in>>6)&1)+16*((r_in>>4)&3)+2*bcb+((r_in>>3)&1),
    //   v8=((mp*np+(jp>>9))*8+((jp>>6)&7))*4096 + rcore*1024 + s*8 + (r_in&7)
    auto act_block = [](std::uint32_t r, std::uint32_t jp, std::uint32_t np) {
      const std::uint32_t o64 = jp & 63, bcb = o64 >> 3;
      const std::uint32_t mp = r >> 9, rcore = (r >> 7) & 3, r_in = r & 127;
      const std::uint32_t s =
          64 * ((r_in >> 6) & 1) + 16 * ((r_in >> 4) & 3) + 2 * bcb + ((r_in >> 3) & 1);
      return (std::size_t(mp * np + (jp >> 9)) * 8 + ((jp >> 6) & 7)) * 4096 +
             std::size_t(rcore) * 1024 + s * 8 + (r_in & 7);
    };
    std::vector<float> za = gufo::npu::atb::bfp16ebs8ToFloat(
        int(cfmul * std::size_t{M} * na / 8 * 9), static_cast<std::uint8_t*>(pa.ch), 0);
    std::vector<float> zb = gufo::npu::atb::bfp16ebs8ToFloat(
        int(cfmul * std::size_t{M} * nb / 8 * 9), static_cast<std::uint8_t*>(pb.ch), 0);
    // mode 6 (SWI_DUP full-C): per full tile of 2048 wire blocks (= 128 act
    // vecs), act vec q occupies wire blocks 16q+8..16q+15.
    const auto hw = [&](std::size_t h) {
      return swimode >= 6
                 ? (h >> 10) * 2048 + 16 * ((h >> 3) & 127) + 8 + (h & 7)
                 : h;
    };
    std::vector<float> act(std::size_t{M} * INT), gact(std::size_t{M} * INT),
        uact(std::size_t{M} * INT);
    RefMatmulXWT(xs.data(), G.data(), gact.data(), M, INT, HID);
    RefMatmulXWT(xs.data(), U.data(), uact.data(), M, INT, HID);
    if (swimode == 4) {  // passthrough: recover true layout by matching blocks
      RefMatmulXWT(xs.data(), G.data(), gact.data(), M, INT, HID);
      
      
      
      const std::size_t nba = std::size_t{M} * na / 8 * 9, nbb = std::size_t{M} * nb / 8 * 9;
      std::FILE* f = std::fopen("/tmp/opencode/swi_match.txt", "w");
      std::printf("matching sample blocks (kind r c0 -> err)...\n");
      for (int t = 0; t < 120; ++t) {
        const bool prog = t >= 60;
        const auto& z = prog ? zb : za;
        const std::uint32_t cbase = prog ? na : 0;
        const std::size_t nblk = (prog ? nbb : nba) / 9;
        const std::size_t blk = nblk / 600 * t + 3;
        if (blk * 8 + 8 > z.size()) continue;
        const float* v = z.data() + blk * 8;
        if (!(std::fabs(v[0]) > 1.f)) continue;  // skip zero blocks
        // candidate kinds: 0 = row-major (r fixed, c..c+7), 1 = col (c fixed, r..r+7)
        float best[2] = {1e30f, 1e30f};
        std::uint32_t bc[2] = {0xFFFFFFFFu, 0xFFFFFFFFu};
        for (std::uint32_t r = 0; r < M; ++r) {
          const float* g = gact.data() + std::size_t{r} * INT + cbase;
          for (std::uint32_t c = 0; c + 8 <= (prog ? nb : na); c += 1) {
            float e = 0, m0 = std::fabs(v[0]) + 1;
            for (int i = 0; i < 8; ++i) {
              const float d = std::fabs(v[i] - g[c + i]);
              e = std::max(e, d / m0);
            }
            if (e < best[0]) { best[0] = e; bc[0] = r * 16384 + c; }
          }
        }
        for (std::uint32_t c = 0; c + 8 <= (prog ? nb : na); ++c) {
          for (std::uint32_t r = 0; r + 8 < M; r += 4) {
            float e = 0, m0 = std::fabs(v[0]) + 1;
            for (int i = 0; i < 8; ++i) {
              const float d = std::fabs(v[i] - gact[std::size_t{r + i} * INT + cbase + c]);
              e = std::max(e, d / m0);
            }
            if (e < best[1]) { best[1] = e; bc[1] = r * 16384 + c; }
          }
        }
        std::printf("blk %6zu p%d: rowkind err %.3f @(r%u,c%u)   colkind err %.3f @(r%u,c%u)\n",
                    blk, int(prog), best[0], bc[0] >> 14, bc[0] & 16383, best[1], bc[1] >> 14,
                    bc[1] & 16383);
        std::fprintf(f, "%zu %d %.4f %u %.4f %u\n", blk, int(prog), best[0], bc[0], best[1],
                     bc[1]);
      }
      std::fclose(f);
      return 0;
    }
    if (swimode == 6) {  // where is data in the full C buffers?
      auto qcount = [](const std::uint8_t* p, std::size_t q) {
        std::size_t nz = 0;
        for (std::size_t i = 0; i < q; ++i) nz += p[i] != 0;
        return nz;
      };
      const std::size_t qa = cfmul * std::size_t{M} * na / 8 * 9 / 4,
                      qb = cfmul * std::size_t{M} * nb / 8 * 9 / 4;
      std::printf("pa.ch nz per quarter:");
      for (int i = 0; i < 4; ++i) std::printf(" %zu", qcount(static_cast<std::uint8_t*>(pa.ch) + i * qa, qa));
      std::printf("\npb.ch nz per quarter:");
      for (int i = 0; i < 4; ++i) std::printf(" %zu", qcount(static_cast<std::uint8_t*>(pb.ch) + i * qb, qb));
      std::printf("\n");
    }
    double num = 0, den = 0;
    double tn[8][8] = {{0}}, td[8][8] = {{0}};
    for (std::uint32_t r = 0; r < M; ++r)
      for (std::uint32_t c = 0; c < INT; ++c) {
        const std::uint32_t np = (c < na ? na : nb) / 512;
        const float v = c < na ? za[hw(act_block(r, c, np)) * 8 + (c & 7)]
                              : zb[hw(act_block(r, c - na, np)) * 8 + (c & 7)];
        act[std::size_t{r} * INT + c] = v;
        const float g = gact[std::size_t{r} * INT + c];
        const float ref = g / (1.f + std::exp(-g)) * uact[std::size_t{r} * INT + c];
        const double d = v - ref;
        num += d * d;
        den += double(ref) * ref;
        if (c < na && (c & 15) == 0) {
          const double dd = v - double(ref);
          tn[r >> 7][(c >> 9) & 7] += dd * dd;
          td[r >> 7][(c >> 9) & 7] += double(ref) * double(ref);
        }
      }
    {
      std::printf("per-tile relL2 (rows mp0-7, cols c/512 0-7):\n");
      for (int rr = 0; rr < 8; ++rr) {
        std::printf("mp%d", rr);
        for (int cc = 0; cc < 8; ++cc)
          std::printf(" %6.3f", td[rr][cc] ? std::sqrt(tn[rr][cc] / td[rr][cc]) : -1.0);
        std::printf("\n");
      }
    }
    {
      const double rel = std::sqrt(num / den);
      std::printf("verify act(swiglu NPU) relL2 %.4f  %s\n", rel,                  rel < 0.08 ? "PASS" : "FAIL");
      if (swimode == 5) {  // passthrough kernel: compare against GATE only
        num = den = 0;
        long ngood = 0, nbad = 0, ninf = 0;
        for (std::uint32_t r = 0; r < M; ++r)
          for (std::uint32_t c = 0; c < INT; ++c) {
            const std::uint32_t np = (c < na ? na : nb) / 512;
            const float v = c < na
                                ? za[act_block(r, c, np) * 8 + (c & 7)]
                                : zb[act_block(r, c - na, np) * 8 + (c & 7)];
            const double g = double(gact[std::size_t{r} * INT + c]);
            if (!(std::fabs(v) < 1e9)) { ++ninf; ++nbad; continue; }
            if (std::fabs(v - g) <= 0.12 * std::fabs(g) + 2e-2)
              ++ngood;
            else
              ++nbad;
          }
        std::printf("passthrough cells: good %ld bad %ld (huge %ld) of %d\n", ngood, nbad,
                    ninf, int(M) * INT);
        {  // positional structure of good cells
          long tot[4][8] = {{0}}, good[4][8] = {{0}};
          for (std::uint32_t r = 0; r < M; r += 2)
            for (std::uint32_t c = 0; c < na; c += 9) {
              const std::size_t blk = act_block(r, c, 8);
              const float v = za[blk * 8 + (c & 7)];
              const double g = gact[std::size_t{r} * INT + c];
              const std::uint32_t isg = std::fabs(v) < 1e9 &&
                                        std::fabs(v - g) <= 0.12 * std::fabs(g) + 2e-2;
              const std::uint32_t r_in = r & 127;
              ++tot[0][blk & 7];
              good[0][blk & 7] += isg;
              ++tot[1][(((r_in >> 6) & 1) << 1) | ((r_in >> 3) & 1)];
              good[1][(((r_in >> 6) & 1) << 1) | ((r_in >> 3) & 1)] += isg;
              ++tot[2][(c >> 6) & 7];
              good[2][(c >> 6) & 7] += isg;
              ++tot[3][(r >> 7) & 3];
              good[3][(r >> 7) & 3] += isg;
            }
          const char* nm[4] = {"blk&7", "half,row8", "corecol", "rcore"};
          for (int h = 0; h < 4; ++h) {
            std::printf("good%% by %s:", nm[h]);
            for (int i = 0; i < 8; ++i)
              if (tot[h][i]) std::printf(" %d:%d", i, (int)(good[h][i] * 100 / tot[h][i]));
            std::printf("\n");
          }
        }
        {  // multiset fingerprint: is the wire carrying gate values AT ALL?
          std::vector<float> a(za), b(zb), g = gact;
          a.insert(a.end(), b.begin(), b.end());
          std::sort(a.begin(), a.end());
          std::sort(g.begin(), g.end());
          double n2 = 0, d2 = 0;
          for (std::size_t i = 0; i < g.size(); ++i) {
            const double d = double(a[i]) - double(g[i]);
            n2 += d * d;
            d2 += double(g[i]) * double(g[i]);
          }
          std::printf("multiset(sorted) rel-diff %.4f  act[min %.1f max %.1f] gate[min %.1f max %.1f]\n",
                      std::sqrt(n2 / d2), a.front(), a.back(), g.front(), g.back());
        }
        std::printf("good-density map (row=64 tok, col=64 ch; digit=good %%):\n");
        for (std::uint32_t r = 0; r < M; r += 64) {
          std::printf("r%4u", r);
          for (std::uint32_t c = 0; c < INT; c += 64) {
            int good = 0, tot = 0;
            for (std::uint32_t i = 0; i < 64; i += 3)
              for (std::uint32_t j = 0; j < 64; j += 7) {
                const std::uint32_t cc = c + j;
                const std::uint32_t np = (cc < na ? na : nb) / 512;
                const float v = cc < na ? za[act_block(r + i, cc, np) * 8 + (cc & 7)]
                                         : zb[act_block(r + i, cc - na, np) * 8 + (cc & 7)];
                ++tot;
                if (std::fabs(v) < 1e9 &&
                    std::fabs(v - gact[std::size_t{r + i} * INT + cc]) <=
                        0.12f * std::fabs(gact[std::size_t{r + i} * INT + cc]) + 2e-2f)
                  ++good;
              }
            std::printf("%d", good * 10 / tot);
          }
          std::printf("\n");
        }
        return 0;
      }
    }

    // Byte-shuffle the act wire (pa || pb blocks) into dn's A stream blocks.
    std::vector<std::uint8_t> awire(std::size_t{M} * INT / 8 * 9);
    for (std::uint32_t r = 0; r < M; ++r)
      for (std::uint32_t c = 0; c < INT; c += 8) {
        const std::uint32_t np = (c < na ? na : nb) / 512;
        const std::uint8_t* src =
            (c < na ? pa.ch : pb.ch) + hw(act_block(r, c < na ? c : c - na, np)) * 9;
        // dn A block: tile (r>>7, c>>6) of [M/128 x 120] 8x8 2x1 layout.
        const std::size_t ablk =
            (std::size_t(r >> 7) * (INT / 64) + (c >> 6)) * 1024 +
            (std::size_t(((r & 127) >> 4) * 8 + ((c & 63) >> 3)) * 2 +
             ((r & 127) >> 3 & 1)) *
                    8 +
            (r & 7);
        std::memcpy(awire.data() + ablk * 9, src, 9);
      }
    if (swimode == 10) {  // cfg31: PLAIN gu pair -> on-core swiglu reshape -> dn
      std::snprintf(pth, sizeof pth, "%s/insts_atb_%ux2048x8192.txt", dir.c_str(), M);
      Program qa = rt.MakeProgram(HID, 2 * na, pth, std::size_t{M} * 2 * na / 8 * 9, &err, 0,
                                  mrows, 3, 5);
      std::snprintf(pth, sizeof pth, "%s/insts_atb_%ux2048x7168.txt", dir.c_str(), M);
      Program qb = rt.MakeProgram(HID, 2 * nb, pth, std::size_t{M} * 2 * nb / 8 * 9, &err, 0,
                                  mrows, 3, 5);
      rt.EncodeAAtb(xs.data(), M, HID, qa);
      std::memcpy(qb.ah, qa.ah, std::size_t{M} * HID / 8 * 9);
      WeightBuffer wqa = rt.PackFfnGuAtb(G.data(), U.data(), na, HID, 4);
      WeightBuffer wqb = rt.PackFfnGuAtb(G.data() + std::size_t{na} * HID, U.data() + std::size_t{na} * HID, nb, HID, 4);
      const auto t_g0 = Clock::now();
      dispatch(rt, qa, wqa);
      dispatch(rt, qb, wqb);
      const double gu_ms = ms(t_g0, Clock::now());
      double gu_dev_ms;  // device-only: warmed dispatch, start+wait, no syncs
      {
        WeightBuffer& sa = rt.WeightSlot(wqa, qa);
        WeightBuffer& sb = rt.WeightSlot(wqb, qb);
        const auto t_d0 = Clock::now();
        sa.run.start();
        sa.run.wait();
        sb.run.start();
        sb.run.wait();
        gu_dev_ms = ms(t_d0, Clock::now());
      }
      if (std::getenv("GUFO_PMAP")) {  // where do gate/up live on the plain wire?
        const std::size_t nq = std::size_t{M} * 2 * na / 8 * 9;
        std::vector<float> zq = gufo::npu::atb::bfp16ebs8ToFloat(
            int(nq), static_cast<std::uint8_t*>(qa.ch), 0);
        long good[2][2] = {{0}};
        long tot = 0;
        for (std::uint32_t r = 0; r < M; r += 4)
          for (std::uint32_t c = 0; c < na; c += 8) {
            const std::size_t p = hw(act_block(r, c, 8));
            for (int d = 0; d < 2; ++d) {  // candidate: gate at slot half d
              const std::size_t b0 = p - 8 + d * 8;
              const float v = zq[b0 * 8 + (c & 7)];
              const double g = gact[std::size_t{r} * INT + c];
              good[d][0] += std::fabs(v - g) <= 0.12 * std::fabs(g) + 2e-2;
              const float w = zq[(b0 + 8) * 8 + (c & 7)];
              const double u = uact[std::size_t{r} * INT + c];
              good[d][1] += std::fabs(w - u) <= 0.12 * std::fabs(u) + 2e-2;
            }
            ++tot;
          }
        std::printf("plain wire match (of %ld): gate@half0 %ld up@half1 %ld | gate@half1 %ld up@half0 %ld\n",
                    tot, good[0][0], good[0][1], good[1][0], good[1][1]);
        return 0;
      }
      NpuRuntime rt6;
      if (!rt6.Init(dir + "/final_swresh_k2048.xclbin", &err)) {
        std::fprintf(stderr, "init swresh: %s\n", err.c_str());
        return 1;
      }
      std::snprintf(pth, sizeof pth, "%s/insts_swresh_%ux7680x2048.txt", dir.c_str(), M);
      Program prg = rt6.MakeProgram(INT, HID, pth, std::size_t{M} * INT / 8 * 9, &err, 0,
                                    mrows, 3, 5);
      prg.a = qa.c;
      WeightBuffer fake;
      fake.bo = qb.c;
      fake.bytes = std::size_t{M} * 2 * nb / 8 * 9;
      WeightBuffer& wr = rt6.WeightSlot(fake, prg);
      if (std::getenv("SWI31_TAG")) {  // permutation x-ray: tag each wire block
        const std::size_t nqb = std::size_t{M} * 2 * na / 8;  // gate wire blocks
        auto* qb8 = static_cast<std::uint8_t*>(qa.ch);
        for (std::size_t b = 0; b < nqb; ++b) {  // tag: 0xA9 | (block idx+1) 24 bit
          const std::uint32_t tv = static_cast<std::uint32_t>(b + 1);
          qb8[b * 9 + 0] = 0xA9;
          qb8[b * 9 + 1] = tv & 0xff;
          qb8[b * 9 + 2] = (tv >> 8) & 0xff;
          qb8[b * 9 + 3] = (tv >> 16) & 0xff;
          for (int i = 4; i < 9; ++i) qb8[b * 9 + i] = 0x11 * (i + b);
        }
        qa.c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wr.run.start();
        wr.run.wait();
        prg.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto* o8 = static_cast<const std::uint8_t*>(prg.c.map());
        const std::size_t nob = std::size_t{M} * INT / 8;  // out blocks
        long ok = 0;
        for (std::size_t b = 0; b < nob; ++b) ok += (o8[b * 9] == 0xA9);
        std::printf("TAG: out blocks with wire tag: %ld/%zu\n", ok, nob);
        {  // full-kernel triage: zeros vs garbage in first blocks
          long z = 0;
          for (std::size_t b = 0; b < nob; ++b) {
            bool allz = true;
            for (int i = 0; i < 9; ++i) allz &= (o8[b * 9 + i] == 0);
            z += allz;
          }
          std::printf("TAG: zero blocks: %ld/%zu; b0 bytes:", z, nob);
          for (int i = 0; i < 18; ++i) std::printf(" %02x", o8[i]);
          std::printf(" | b16:");
          for (int i = 0; i < 9; ++i) std::printf(" %02x", o8[16 * 9 + i]);
          std::printf("\n");
        }
        std::printf("outblk->src(wire blk):");  // full map of first tile(s)
        for (std::size_t b = 0; b < 32; ++b) {
          const std::uint32_t tv =
              o8[b * 9 + 1] | (o8[b * 9 + 2] << 8) | (o8[b * 9 + 3] << 16);
          const std::uint32_t src = tv ? tv - 1 : 0xFFFFFFFFu;
          std::printf(" b%zu<-%x", b, src);
        }
        std::printf("\n");
        // expected for RAW passthrough: out blk 8n+e? vec push => blocks 8n..8n+7
        // all from slot(n) half 0: s(n) = ((rg>>2)<<6)|((rg&3)<<4)|((kq)<<1)|p
        std::printf("expected: b8n..8n+7 <- slot n (gate half), n=(rg*8+kq)*2+p\n");
        for (int rg = 0; rg < 2; ++rg)
          for (int kq = 0; kq < 8; ++kq)
            for (int p = 0; p < 2; ++p) {
              const int n = (rg * 8 + kq) * 2 + p;
              const int sl = (rg >> 2 << 6) | (rg & 3) << 4 | (kq << 1) | p;
              std::printf(" n%d->s%d", n, sl);
            }
        std::printf("\n");
        return 0;
      }
      wr.run.start();
      wr.run.wait();
      if (std::getenv("SWI31_PRE")) {  // poison output: 0xAB = "never written"
        auto *m8 = static_cast<std::uint8_t *>(prg.c.map());
        const std::size_t ob = std::size_t{M} * INT / 8 * 9;
        memset(m8, 0xAB, ob);
        prg.c.sync(XCL_BO_SYNC_BO_TO_DEVICE, ob, 0);
      }
      if (const char* bu = std::getenv("SWI31_BURST")) {  // K dispatches with no host work between
        const int nb = std::atoi(bu);
        if (const char* ps = std::getenv("SWI31_PRESYNC")) {  // td | fd | both
          const std::string m = ps;
          if (m == "td" || m == "both") prg.c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
          if (m == "fd" || m == "both") prg.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }
        const char* gu = std::getenv("SWI31_GAPUS");
        const int gapUs = gu ? std::atoi(gu) : 0;
        for (int r = 0; r < nb; ++r) {
          if (gapUs && r) std::this_thread::sleep_for(std::chrono::microseconds(gapUs));
          wr.run.start();
          wr.run.wait();
        }
        prg.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto* q = static_cast<const std::uint8_t*>(prg.c.map());
        int zt = 0;
        for (std::size_t t = 0; t < 960; ++t) {
          bool allz = true;
          for (std::size_t i = 0; i < 1024 * 9 && allz; ++i) allz = q[t * 1024 * 9 + i] == 0;
          zt += allz;
        }
        std::printf("BURST %d: zero tiles %d\n", nb, zt);
        return 0;
      }
      if (const char* rep = std::getenv("SWI31_REP")) {  // same inputs, N dispatches: per-run zero tiles
        const int nrep = std::atoi(rep);
        const std::size_t tb = 1024 * 9;
        const char* gapEnv = std::getenv("SWI31_GAP");
        const int gapMs = gapEnv ? std::atoi(gapEnv) : 0;
        for (int r = 0; r < nrep; ++r) {
          if (gapMs) std::this_thread::sleep_for(std::chrono::milliseconds(gapMs));
          wr.run.start();
          wr.run.wait();
          prg.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          const auto* q = static_cast<const std::uint8_t*>(prg.c.map());
          int zt = 0;
          for (std::size_t t = 0; t < 960; ++t) {
            bool allz = true;
            for (std::size_t i = 0; i < tb && allz; ++i) allz = q[t * tb + i] == 0;
            zt += allz;
          }
          std::printf("REP %d: zero tiles %d\n", r, zt);
        }
      }
      const auto t_r0 = Clock::now();
      wr.run.start();
      wr.run.wait();
      const double resh_ms = ms(t_r0, Clock::now());
      prg.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      // act check in dn-A order vs torch act
      std::vector<float> zo = gufo::npu::atb::bfp16ebs8ToFloat(
          int(std::size_t{M} * INT / 8 * 9), static_cast<std::uint8_t*>(prg.c.map()), 0);
      double an = 0, ad = 0;
      const bool rawref = std::getenv("SWI31_RAW") != nullptr;
      for (std::uint32_t r = 0; r < M; ++r)
        for (std::uint32_t c = 0; c < INT; c += 8) {
          const std::size_t ablk =
              (std::size_t(r >> 7) * (INT / 64) + (c >> 6)) * 1024 +
              (std::size_t(((r & 127) >> 4) * 8 + ((c & 63) >> 3)) * 2 +
               ((r & 127) >> 3 & 1)) *
                      8 +
              (r & 7);
          for (std::uint32_t e = 0; e < 8; ++e) {
            const double v = zo[ablk * 8 + e];
            const double w =
                rawref ? gact[std::size_t{r} * INT + c + e]
                       : (std::getenv("SWI31_RAW3")
                              ? double(gact[std::size_t{r} * INT + c + e]) *
                                    double(uact[std::size_t{r} * INT + c + e])
                              : act[std::size_t{r} * INT + c + e]);
            an += (v - w) * (v - w);
            ad += w * w;
          }
        }
      std::printf("verify act(swresh)    relL2 %.4f  (plain gu pair %.2f ms, gu device-only %.2f ms, swresh %.2f ms)\n",
                  std::sqrt(an / ad), gu_ms, gu_dev_ms, resh_ms);
      if (std::getenv("SWI31_ZMAP")) {  // which dn tiles came back all-zero?
        std::vector<int> zt;
        std::size_t zab = 0;  // tiles still 0xAB = never written by measured run
        for (std::size_t t = 0; t < std::size_t{M} * INT / 8 / 1024; ++t) {
          bool z = true, p = true;
          const auto *p8 = static_cast<const std::uint8_t*>(prg.c.map()) + t * 1024 * 9;
          for (std::size_t i = 0; i < 1024 * 9; ++i) {
            z &= (p8[i] == 0);
            p &= (p8[i] == 0xAB);
          }
          if (z) zt.push_back(int(t));
          zab += p;
        }
        std::printf("unwritten(0xAB) tiles: %zu\n", zab);
        std::printf("zero tiles: %zu of %zu:", zt.size(), std::size_t{M} * INT / 8 / 1024);
        int prev = -2, from = -1;
        for (int v : zt) {
          if (v != prev + 1) {
            if (from >= 0) std::printf(" %d..%d", from, prev);
            from = v;
          }
          prev = v;
        }
        if (from >= 0) std::printf(" %d..%d", from, prev);
        std::printf("\n");
        return 0;
      }
      if (std::getenv("SWI31_WHERE")) {  // exact RAW copy check: o[n] == wire[slot(n)]
        std::vector<float> zq = gufo::npu::atb::bfp16ebs8ToFloat(
            int(std::size_t{M} * 2 * na / 8 * 9), static_cast<std::uint8_t*>(qa.ch), 0);
        long exact = 0, tot = 0;
        std::printf("mismatch(n,slot):");
        for (std::uint32_t r = 0; r < 128; ++r)
          for (std::uint32_t c8 = 0; c8 < 8; ++c8) {
            const std::size_t ablk =
                (std::size_t(r >> 7) * (INT / 64) + 0) * 1024 +
                (std::size_t(((r & 127) >> 4) * 8 + c8) * 2 + ((r & 127) >> 3 & 1)) * 8 +
                (r & 7);
            const std::size_t gbase = hw(act_block(r, c8 * 8, 8)) - 8;  // gate block
            const bool ok =
                std::fabs(zo[ablk * 8 + 3] - zq[gbase * 8 + 3]) <=
                    0.15f * std::fabs(zq[gbase * 8 + 3]) + 2e-2f &&
                std::fabs(zo[ablk * 8 + 5] - zq[gbase * 8 + 5]) <=
                    0.15f * std::fabs(zq[gbase * 8 + 5]) + 2e-2f;
            ++tot;
            if (ok)
              ++exact;
            else if (tot < 40) {
              long mb = 9999;
              const float ref3 = zo[ablk * 8 + 3], ref5 = zo[ablk * 8 + 5];
              for (std::size_t b = 0; b < 8192; ++b)
                if (std::fabs(zq[b * 8 + 3] - ref3) < 0.13f * (std::fabs(ref3) + 2e-2f) &&
                    std::fabs(zq[b * 8 + 5] - ref5) < 0.13f * (std::fabs(ref5) + 2e-2f)) {
                  mb = long(b);
                  break;
                }
              std::printf(" (%u,%u:e%+ld,g%+ld)", r, c8,
                          std::int64_t(ablk) - std::int64_t(gbase),
                          std::int64_t(ablk) - mb);
            }
          }
        std::printf("\nRAW copy exact: %ld/%ld\n", exact, tot);
        return 0;
      }
      std::snprintf(pth, sizeof pth, "%s/insts_atb_%ux7680x2048.txt", dir.c_str(), M);
      Program pdn2 =
          rt2.MakeProgram(INT, HID, pth, std::size_t{M} * HID / 8 * 9, &err, 0, mrows, 3, 5);
      rt2.RebindA(pdn2, prg.c);
      WeightBuffer& wd2 = rt2.WeightSlot(wd, pdn2);
      wd2.run.start();
      wd2.run.wait();
      const auto t_t0 = Clock::now();
      wd2.run.start();
      wd2.run.wait();
      const double dn2_ms = ms(t_t0, Clock::now());
      pdn2.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::vector<float> y(std::size_t{M} * HID);
      rt2.DecodeCAtb(y.data(), M, HID, pdn2);
      std::vector<float> yref(std::size_t{M} * HID);
      RefMatmulXWT(act.data(), Wd.data(), yref.data(), M, HID, INT);
      double tn = 0, td = 0;
      for (std::size_t e = 0; e < y.size(); ++e) {
        const double d = double(y[e]) - double(yref[e]);
        tn += d * d;
        td += double(yref[e]) * double(yref[e]);
      }
      std::printf("verify dn(swresh-A)   relL2 %.4f  %s   (%.2f ms, zero host encode)\n",
                  std::sqrt(tn / td), tn / td < 0.08 * 0.08 ? "PASS" : "FAIL", dn2_ms);
      std::printf("CHAIN TOTAL: %.2f ms (gu %.2f + swresh %.2f + dn %.2f)\n",
                  gu_ms + resh_ms + dn2_ms, gu_ms, resh_ms, dn2_ms);
      return 0;
    }
    if (swimode == 8) {  // cfg30: on-device reshape pass -> dn; zero host encode.
      NpuRuntime rt5;
      if (!rt5.Init(dir + "/final_swiresh_k2048.xclbin", &err)) {
        std::fprintf(stderr, "init swiresh: %s\n", err.c_str());
        return 1;
      }
      std::snprintf(pth, sizeof pth, "%s/insts_swiresh_%ux7680x2048.txt", dir.c_str(), M);
      Program prg = rt5.MakeProgram(INT, HID, pth, std::size_t{M} * INT / 8 * 9, &err, 0,
                                    mrows, 3, 5);
      // Bind A := pa.C wire, B := pb.C wire (via the weight-slot arg), C := out.
      prg.a = pa.c;
      WeightBuffer fake;
      fake.bo = pb.c;
      fake.bytes = std::size_t{M} * nb / 8 * 9 * 2;
      WeightBuffer& wr = rt5.WeightSlot(fake, prg);
      wr.run.start();
      wr.run.wait();  // warm: first C write is lost on this stack
      const auto t_r0 = Clock::now();
      wr.run.start();
      wr.run.wait();
      const double resh_ms = ms(t_r0, Clock::now());
      prg.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::size_t nbad = 0;
      {
        const std::uint8_t* g = static_cast<const std::uint8_t*>(prg.c.map());
        for (std::size_t b = 0; b < awire.size(); ++b) nbad += (g[b] != awire[b]);
      }
      std::printf("verify reshape BYTES vs host shuffle: %s (%zu bad of %zu, %.2f ms)\n",
                  nbad ? "FAIL" : "PASS", nbad, awire.size(), resh_ms);
      // dn GEMM reading the DEVICE-PRODUCED A wire (no encode, no A sync).
      std::snprintf(pth, sizeof pth, "%s/insts_atb_%ux7680x2048.txt", dir.c_str(), M);
      Program pdn2 =
          rt2.MakeProgram(INT, HID, pth, std::size_t{M} * HID / 8 * 9, &err, 0, mrows, 3, 5);
      rt2.RebindA(pdn2, prg.c);
      WeightBuffer& wd2 = rt2.WeightSlot(wd, pdn2);
      wd2.run.start();
      wd2.run.wait();  // warm
      const auto t_t0 = Clock::now();
      wd2.run.start();
      wd2.run.wait();
      const double dn2_ms = ms(t_t0, Clock::now());
      pdn2.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::vector<float> y(std::size_t{M} * HID);
      rt2.DecodeCAtb(y.data(), M, HID, pdn2);
      std::vector<float> yref(std::size_t{M} * HID);
      RefMatmulXWT(act.data(), Wd.data(), yref.data(), M, HID, INT);
      double tn = 0, td = 0;
      for (std::size_t e = 0; e < y.size(); ++e) {
        const double d = double(y[e]) - double(yref[e]);
        tn += d * d;
        td += double(yref[e]) * double(yref[e]);
      }
      std::printf("verify dn(dev-A)      relL2 %.4f  %s   (%.2f ms, zero host encode)\n",
                  std::sqrt(tn / td), tn / td < 0.08 * 0.08 ? "PASS" : "FAIL", dn2_ms);
      return 0;
    }
    if (swimode == 7) {  // cfg29: dn runs DIRECTLY from the gu programs' C wires
      // via the DMA tap program — no host encode, no A sync, activations never
      // leave device memory.
      NpuRuntime rt4;
      if (!rt4.Init(dir + "/final_dn_tap_k2048.xclbin", &err)) {
        std::fprintf(stderr, "init dn_tap: %s\n", err.c_str());
        return 1;
      }
      std::snprintf(pth, sizeof pth, "%s/insts_dntap_%ux7680x2048.txt", dir.c_str(), M);
      // 4-arg program: (A1=3, A2=4, B=5, C=6); 3-arg ladder variant: (A=3, B=4, C=5)
      const bool t3 = std::getenv("GUFO_TAP3") != nullptr;
      Program pdn2 = rt4.MakeProgram(INT, HID, pth, std::size_t{M} * HID / 8 * 9, &err, 0,
                                    mrows, 3, t3 ? 5 : 6);
      WeightBuffer wd4 = rt4.PackWeightAtb(Wd.data(), HID, INT, t3 ? 4 : 5);
      if (std::getenv("GUFO_TAP_SCRATCH")) {  // bring-up: in-xclbin BOs, tap only
        xrt::bo a1 = rt4.MakeBo(18 << 20, 3);
        rt4.RebindA(pdn2, a1,
                    t3 ? xrt::bo()
                       : (std::getenv("GUFO_TAP_SAMEBO") ? a1 : rt4.MakeBo(5 << 20, 4)));
      } else if (!t3)
        rt4.RebindA(pdn2, pa.c, pb.c);  // A streams = gu C output BOs
      WeightBuffer& w2 = rt4.WeightSlot(wd4, pdn2);
      w2.run.start();
      w2.run.wait();  // warm: first C write is lost
      const auto t_t0 = Clock::now();
      w2.run.start();
      w2.run.wait();
      const double tap_ms = ms(t_t0, Clock::now());
      pdn2.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::vector<float> y(std::size_t{M} * HID);
      rt4.DecodeCAtb(y.data(), M, HID, pdn2);
      std::vector<float> yref(std::size_t{M} * HID);
      RefMatmulXWT(act.data(), Wd.data(), yref.data(), M, HID, INT);
      double tn = 0, td = 0;
      for (std::size_t e = 0; e < y.size(); ++e) {
        const double d = double(y[e]) - double(yref[e]);
        tn += d * d;
        td += double(yref[e]) * double(yref[e]);
      }
      std::printf("verify dn(A-tap NPU)  relL2 %.4f  %s   (%.2f ms, zero host encode)\n",
                  std::sqrt(tn / td), tn / td < 0.08 * 0.08 ? "PASS" : "FAIL", tap_ms);
      return 0;
    }
    std::memcpy(pdn.ah, awire.data(), awire.size());
    const auto t_dn0 = Clock::now();
    dispatch(rt2, pdn, wd);
    const double dn_ms = ms(t_dn0, Clock::now());
    std::vector<float> y(std::size_t{M} * HID);
    rt2.DecodeCAtb(y.data(), M, HID, pdn);
    std::vector<float> yref(std::size_t{M} * HID);
    RefMatmulXWT(act.data(), Wd.data(), yref.data(), M, HID, INT);
    num = den = 0;
    for (std::size_t e = 0; e < y.size(); ++e) {
      const double d = double(y[e]) - double(yref[e]);
      num += d * d;
      den += double(yref[e]) * double(yref[e]);
    }
    std::printf("verify dn(act_npu)   relL2 %.4f  %s\n", std::sqrt(num / den),
                num / den < 0.08 * 0.08 ? "PASS" : "FAIL");
    std::printf("timings: gu pair (2 dispatches) %.2f ms   dn %.2f ms\n", swi_ms, dn_ms);
    return 0;
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

  // Device floor: with A already resident (verify ran each op), measure pure
  // submit+wait per shape. Chunk floor = sum over the layer mix.
  if (std::atoi(std::getenv("GUFO_PREFILL_DEBUG") ? std::getenv("GUFO_PREFILL_DEBUG")
                                                  : "0") < 2) {
    std::printf("device floor (submit+wait, A resident):\n");
    double total_floor = 0;
    for (std::size_t i = 0; i < ops.size(); ++i) {
      if (skip_dn && ops[i].second) continue;
      rtp(i).GemmAtb(progs[i], wbs[i], x.data(), M, y.data(), nullptr);  // warm
      const auto f0 = Clock::now();
      constexpr int kRep = 10;
      for (int k = 0; k < kRep; ++k) rtp(i).DispatchAtb(progs[i], wbs[i]);
      const double dms = ms(f0, Clock::now()) / kRep;
      const int per_chunk = std::strcmp(ops[i].name, "conv_in") == 0 ||
                                    std::strcmp(ops[i].name, "conv_out") == 0
                                ? 13
                                : std::strcmp(ops[i].name, "qkv") == 0 ||
                                          std::strcmp(ops[i].name, "o") == 0
                                      ? 11
                                      : 24;
      total_floor += dms * per_chunk;
      std::printf("  %-9s %7.2f ms x%2d\n", ops[i].name, dms, per_chunk);
    }
    std::printf("  chunk floor %.1f ms = %.0f tok/s GEMM-only\n", total_floor,
                1024000.0 / total_floor);
  }

  std::vector<double> acc(ops.size(), 0.0);

  // Device-pipeline experiment: queue op i+1 (encode+start) while op i is still
  // running on the device, deferred wait. No C sync/decode: measures the
  // ceiling of host/device overlap for the scaffold chain.
  if (std::getenv("GUFO_PREFILL_PIPE")) {
    const bool is_conv[24] = {true, true, false, true, false, true, false, true,
                             false, true, false, true, false, true, false, true,
                             false, true, false, true, false, true, false, true};
    const auto idx = [&](const char* nm) {
      for (std::size_t i = 0; i < ops.size(); ++i)
        if (!std::strcmp(ops[i].name, nm)) return i;
      return std::size_t{0};
    };
    std::vector<std::size_t> seq;
    for (int l = 0; l < 24; ++l) {
      seq.push_back(is_conv[l] ? idx("conv_in") : idx("qkv"));
      seq.push_back(is_conv[l] ? idx("conv_out") : idx("o"));
      seq.push_back(idx("gu_a"));
      seq.push_back(idx("gu_b"));
      if (!skip_dn) seq.push_back(idx("dn"));
    }
    for (std::size_t i : seq)  // ensure pinned+warm (first C write lost)
      rtp(i).GemmAtb(progs[i], wbs[i], x.data(), M, y.data(), nullptr);
    for (int it = 0; it < iters; ++it) {
      const auto t0 = Clock::now();
      WeightBuffer* pend = nullptr;
      for (std::size_t i : seq) {
        WeightBuffer& w = rtp(i).WeightSlot(wbs[i], progs[i]);
        rtp(i).EncodeAAtb(x.data(), M, ops[i].K, progs[i]);
        progs[i].a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        rtp(i).SlotRun(w);
        rtp(i).AttachTo(w.run, w, progs[i]);
        w.run.start();
        if (pend) {
          pend->run.wait();
          pend = nullptr;
        }
        pend = &w;
      }
      if (pend) pend->run.wait();
      const double d = ms(t0, Clock::now());
      std::printf("pipe iter %d: %.2f ms = %.0f tok/s\n", it, d, 1000.0 * M / d);
    }
    return 0;
  }

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
