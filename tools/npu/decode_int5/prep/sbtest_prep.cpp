// Attention prep probe, two dispatches on a synthetic QKV GEMM result (C wire):
//   sbtest-prep <xclbin> <insts_kv> <insts_q> [reps]
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

static void EncGroup(const float* v, uint8_t* out) {
  unsigned max_exp = 0;
  for (int i = 0; i < 8; ++i) { uint32_t u; std::memcpy(&u, &v[i], 4); max_exp = std::max(max_exp, (u >> 23) & 0xFFu); }
  uint32_t ib = max_exp >= 7 ? (260u - max_exp) << 23 : 0u;
  float inv; std::memcpy(&inv, &ib, 4);
  out[0] = (uint8_t)max_exp;
  for (int i = 0; i < 8; ++i) { float q = std::fmin(127.0f, std::fmax(-128.0f, std::nearbyint(v[i] * inv))); out[1 + i] = (uint8_t)(int8_t)q; }
}
static void DecGroup(const uint8_t* p, float* v) {
  float mult = std::ldexp(1.0f, (int)p[0] - 127) / 64.0f;
  for (int i = 0; i < 8; ++i) v[i] = (float)(int8_t)p[1 + i] * mult;
}
static uint16_t Bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float BfF(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }


static void MakeTab(uint8_t* tab8, int pos0, float theta) {
  uint16_t* tab = (uint16_t*)tab8;
  for (int br = 0; br < 2; ++br) for (int ib = 0; ib < 8; ++ib) for (int row = 0; row < 8; ++row) for (int col = 0; col < 8; ++col) {
    float pos = (float)(pos0 + br * 8 + row);
    int i = ib * 8 + col;
    float ang = pos * std::pow(theta, -2.0f * i / 128.f);
    tab[((br * 8 + ib) * 2) * 64 + row * 8 + col] = Bf16(std::cos(ang));
    tab[((br * 8 + ib) * 2 + 1) * 64 + row * 8 + col] = Bf16(std::sin(ang));
  }
}
static void RefNR(const float* xd, const std::vector<float>& w, int pos0, float eps, float theta, float* out) {
  for (int t = 0; t < 16; ++t) {
    const float* xr = &xd[t * 128];
    double ss = 0; for (int d = 0; d < 128; ++d) ss += (double)xr[d] * xr[d];
    float sc = 1.f / std::sqrt((float)(ss / 128) + eps), pos = (float)(pos0 + t), y[128];
    for (int d = 0; d < 128; ++d) y[d] = xr[d] * sc * w[d];
    for (int i = 0; i < 64; ++i) {
      float ang = pos * std::pow(theta, -2.0f * i / 128.f), c = std::cos(ang), sn = std::sin(ang);
      out[t * 128 + i] = y[i] * c - y[i + 64] * sn;
      out[t * 128 + i + 64] = y[i + 64] * c + y[i] * sn;
    }
  }
}



static std::vector<char> ReadAll(const char* p) { std::ifstream f(p, std::ios::binary | std::ios::ate); size_t n = f.tellg(); f.seekg(0); std::vector<char> v(n); f.read(v.data(), n); return v; }

int main(int argc, char** argv) {
  const int NT = 8, W = NT * 8, reps = argc > 4 ? atoi(argv[4]) : 5, Q0 = 100;
  const size_t TILE = 18432, PIECE = 1536, TE = 5120, PASS = 8 * 98304, PCOL = 98304, CH = 18432;
  const size_t QKV_BYTES = (size_t)8 * 3 * 32 * TILE, TAB_BYTES = (size_t)W * TE;
  const size_t PASS_BYTES = (NT - 1) * PASS + 8 * PCOL, KVW = (size_t)(W + 2) * CH, KN_BYTES = (size_t)W * 6144;
  const float eps = 1e-5f, theta = 1000000.f;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> wq(128), wk(128);
  for (auto& v : wq) v = (0.5f + std::fabs(nd(rng))) * 0.1275f;
  for (auto& v : wk) v = 0.5f + std::fabs(nd(rng));
  std::vector<uint8_t> QKV(QKV_BYTES, 0), TAB(TAB_BYTES, 0);
  std::vector<std::vector<float>> xd((size_t)NT * 24);
  for (int mt = 0; mt < NT; ++mt) for (int nt = 0; nt < 24; ++nt) {
    auto& X = xd[(size_t)mt * 24 + nt];
    X.resize(128 * 128);
    const size_t tile = (size_t)((mt / 4) * 3 + nt / 8) * 32 + (nt % 8) * 4 + (mt % 4);
    uint8_t* t = &QKV[tile * TILE];
    for (int r = 0; r < 128; ++r) for (int c = 0; c < 128; ++c) X[r * 128 + c] = nd(rng) * (c % 37 == 0 ? 4.f : 1.f) * (1.f + r % 3);
    for (int sbr = 0; sbr < 8; ++sbr) for (int sbc = 0; sbc < 8; ++sbc) for (int br = 0; br < 2; ++br) for (int bc = 0; bc < 2; ++bc) for (int rib = 0; rib < 8; ++rib) {
      int row = sbr * 16 + br * 8 + rib, col = sbc * 16 + bc * 8;
      uint8_t* g = t + (((sbr * 8 + sbc) * 4) + br * 2 + bc) * 72 + rib * 9;
      EncGroup(&X[row * 128 + col], g);
      DecGroup(g, &X[row * 128 + col]);
    }
  }
  // key history chunks Q0-2, Q0-1 (kv window slots 0, 1) and chunk 0 (sink), random K blocks
  std::vector<uint8_t> KVH(KVW, 0), C0(CH, 0);
  std::vector<float> khist(2 * 4 * 16 * 128), kc0(4 * 16 * 128);
  auto rand_chunk = [&](uint8_t* base, float* kf, int r) {
    for (int kb = 0; kb < 2; ++kb) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
      float v8[8]; for (int i = 0; i < 8; ++i) v8[i] = nd(rng) * 1.5f;
      uint8_t* g = base + (size_t)((r * 0) + (kb * 16 + ks)) * 72 + rib * 9;
      EncGroup(v8, g);
      DecGroup(g, &kf[(kb * 8 + rib) * 128 + ks * 8]);
    }
  };
  auto piece_store = [&](uint8_t* chunk_store, int r, const uint8_t* chunk) { for (int e = 0; e < 3; ++e) std::memcpy(chunk_store + e * 6144 + r * PIECE, chunk + e * 1536, 1536); };
  for (int h = 0; h < 2; ++h) for (int r = 0; r < 4; ++r) {
    uint8_t ch[4608] = {0};
    rand_chunk(ch, &khist[((size_t)h * 4 + r) * 16 * 128], r);
    piece_store(&KVH[(size_t)h * CH], r, ch);
  }
  for (int r = 0; r < 4; ++r) { uint8_t ch[4608] = {0}; rand_chunk(ch, &kc0[(size_t)r * 16 * 128], r); piece_store(C0.data(), r, ch); }
  for (int w = 0; w < W; ++w) {
    uint8_t* te = &TAB[(size_t)w * TE];
    MakeTab(te, (Q0 + w) * 16, theta);
    uint16_t* wp = (uint16_t*)(te + 4096);
    for (int d = 0; d < 128; ++d) { wp[d] = Bf16(wq[d]); wp[128 + d] = Bf16(wk[d]); }
    *(uint16_t*)(te + 4608) = Bf16(eps);
    int32_t h[4] = {(Q0 + w) * 16, 16, (Q0 + w), 0};
    std::memcpy(te + 4864, h, 16);
  }
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(argv[1])};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  auto mk = [&](const std::vector<char>& iw) { xrt::bo b(dev, iw.size(), XRT_BO_FLAGS_CACHEABLE, kern.group_id(1)); std::memcpy(b.map<void*>(), iw.data(), iw.size()); b.sync(XCL_BO_SYNC_BO_TO_DEVICE); return b; };
  auto iw_kv = ReadAll(argv[2]), iw_q = ReadAll(argv[3]);
  xrt::bo ib_kv = mk(iw_kv), ib_q = mk(iw_q);
  xrt::bo b_qkv(dev, QKV.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo b_tab(dev, TAB.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
  xrt::bo b_pass(dev, PASS_BYTES, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
  xrt::bo b_kv(dev, KVW, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(6));
  xrt::bo b_kvw(b_kv, (size_t)W * CH, 2 * CH);
  xrt::bo b_kn(dev, KN_BYTES, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(7));
  xrt::bo b_c0(dev, std::max(KN_BYTES, CH), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(7));
  std::memcpy(b_qkv.map<void*>(), QKV.data(), QKV.size());
  std::memcpy(b_tab.map<void*>(), TAB.data(), TAB.size());
  std::memset(b_pass.map<void*>(), 0, PASS_BYTES);
  std::memcpy(b_kv.map<void*>(), KVH.data(), KVW);
  std::memset(b_kn.map<void*>(), 0, KN_BYTES);
  std::memset(b_c0.map<void*>(), 0, std::max(KN_BYTES, CH));
  std::memcpy(b_c0.map<void*>(), C0.data(), CH);
  for (auto* b : {&b_qkv, &b_tab, &b_pass, &b_kv, &b_kn, &b_c0}) b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto run_wait = [&](xrt::run run) -> int {
    auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    for (;;) { auto st = run.state(); if (st == ERT_CMD_STATE_COMPLETED) return 0; if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED && st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) { std::printf("run ended in state %d\n", (int)st); return 3; } if (std::chrono::steady_clock::now() > t_end) { std::printf("timeout, state %d\n", (int)st); return 4; } }
  };
  auto t0 = std::chrono::steady_clock::now();
  if (int e = run_wait(kern(3, ib_kv, (unsigned)(iw_kv.size() / 4), b_qkv, b_tab, b_pass, b_kvw, b_kn))) return e;
  double kv_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  b_kvw.sync(XCL_BO_SYNC_BO_FROM_DEVICE); b_kn.sync(XCL_BO_SYNC_BO_FROM_DEVICE); b_kv.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint8_t* KVo = b_kv.map<uint8_t*>();
  const uint8_t* KN = b_kn.map<uint8_t*>();
  // key bound per kv head: max |k|^2 over the window and the history chunks
  float kb[4];
  std::vector<float> kdec((size_t)(W + 2) * 4 * 16 * 128);  // [chunk slot][r][token][d]
  for (int slot = 0; slot < W + 2; ++slot) for (int r = 0; r < 4; ++r) {
    uint8_t ch[4608];
    for (int e = 0; e < 3; ++e) std::memcpy(ch + e * 1536, KVo + (size_t)slot * CH + e * 6144 + r * PIECE, 1536);
    for (int kb2 = 0; kb2 < 2; ++kb2) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) DecGroup(ch + (kb2 * 16 + ks) * 72 + rib * 9, &kdec[((size_t)(slot * 4 + r) * 16 + kb2 * 8 + rib) * 128 + ks * 8]);
  }
  for (int r = 0; r < 4; ++r) {
    double mx = 0;
    for (int slot = 0; slot < W + 2; ++slot) for (int t = 0; t < 16; ++t) { double ss = 0; for (int d = 0; d < 128; ++d) { float v = kdec[((size_t)(slot * 4 + r) * 16 + t) * 128 + d]; ss += (double)v * v; } mx = std::max(mx, ss); }
    kb[r] = (float)std::sqrt(mx) * 1.02f;
  }
  for (int w = 0; w < W; ++w) std::memcpy(&TAB[(size_t)w * TE + 4992], kb, 16);
  std::memcpy(b_tab.map<void*>(), TAB.data(), TAB.size());
  b_tab.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  double best = 1e9;
  for (int rp = 0; rp < reps; ++rp) {
    auto t1 = std::chrono::steady_clock::now();
    if (int e = run_wait(kern(3, ib_q, (unsigned)(iw_q.size() / 4), b_qkv, b_tab, b_pass, b_kv, b_c0))) return e;
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
  }
  b_pass.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint8_t* C = b_pass.map<uint8_t*>();
  double worst = 0, wd = 0, wr = 0; long wb = 0, wmt = 0;
  double nq = 0, dq = 0, mdiff = 0, mref_max = 0, msum = 0; long hdr_bad = 0, mcount = 0;
  std::vector<float> ref(16 * 128), sub(16 * 128);
  for (int mt = 0; mt < NT; ++mt) for (int c = 0; c < 8; ++c) for (int r = 0; r < 4; ++r) {
    const int w = mt * 8 + c, q = Q0 + w;
    auto stripe = [&](int nt, float* dst) { for (int t = 0; t < 16; ++t) for (int d = 0; d < 128; ++d) dst[t * 128 + d] = xd[(size_t)mt * 24 + nt][(c * 16 + t) * 128 + d]; };
    uint8_t qb[9216];
    for (int u = 0; u < 6; ++u) std::memcpy(qb + u * 1536, C + (size_t)mt * PASS + (size_t)c * PCOL + (size_t)(1 + u) * 6144 + r * PIECE, 1536);
    const uint8_t* hp = C + (size_t)mt * PASS + (size_t)c * PCOL + r * PIECE;
    int32_t hh[4]; std::memcpy(hh, hp, 16);
    if (hh[0] != q * 16 || hh[1] != 16 || hh[2] != q || hh[3] != 0) ++hdr_bad;
    const float* Md = (const float*)(hp + 128);
    for (int hq = 0; hq < 4; ++hq) {
      stripe(r * 4 + hq, sub.data());
      RefNR(sub.data(), wq, q * 16, eps, theta, ref.data());
      for (int br = 0; br < 2; ++br) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
        float d8[8]; DecGroup(qb + ((hq * 2 + br) * 16 + ks) * 72 + rib * 9, d8);
        const float* rr = &ref[(br * 8 + rib) * 128 + ks * 8];
        for (int i = 0; i < 8; ++i) { double df = std::fabs((double)d8[i] - rr[i]); if (!(df <= worst)) { worst = df; wb = hq * 1000000 + br * 100000 + ks * 100 + rib; wmt = mt * 100 + c * 10 + r; wd = d8[i]; wr = rr[i]; } nq += (double)(d8[i] - rr[i]) * (d8[i] - rr[i]); dq += (double)rr[i] * rr[i]; }
      }
      for (int t = 0; t < 16; ++t) {
        const int pos = q * 16, last = pos + t, lo = std::max(0, last - 31);
        double mx = -1e30, qn2 = 0;
        for (int d = 0; d < 128; ++d) qn2 += (double)ref[t * 128 + d] * ref[t * 128 + d];
        auto score = [&](int tok) {
          const float* kr;
          const int qc = tok / 16;
          if (tok < 8) kr = &kc0[((size_t)r * 16 + tok) * 128];
          else if (qc < Q0) kr = &khist[(((size_t)(qc - (Q0 - 2)) * 4 + r) * 16 + tok % 16) * 128];
          else kr = &kdec[(((size_t)(qc - Q0 + 2) * 4 + r) * 16 + tok % 16) * 128];
          double sc = 0; for (int d = 0; d < 128; ++d) sc += (double)ref[t * 128 + d] * kr[d];
          return sc;
        };
        for (int tok = lo; tok <= last; ++tok) mx = std::max(mx, score(tok));
        for (int tok = 0; tok < std::min(8, lo); ++tok) mx = std::max(mx, score(tok));
        const double bound = std::sqrt(qn2) * kb[r];
        const double mref = std::min(std::max(mx, bound - 100.0), bound);
        const double got = Md[hq * 16 + t];
        mdiff = std::max(mdiff, std::fabs(got - mref)); mref_max = std::max(mref_max, std::fabs(mref)); msum += std::fabs(got - mref); ++mcount;
      }
    }
  }
  { const float* d = (const float*)(C + 384); std::printf("dbg yh %g %g | yl %g %g | y %g %g | ns %g %g\n", d[0], d[8], d[64], d[72], d[128], d[136], d[192], d[200]); }
  std::printf("worst Q elem: diff %g dev %g ref %g at hq/br/ks/rib code %ld (mt*100+c*10+r %ld)\n", worst, wd, wr, wb, wmt);
  { uint8_t qb0[9216]; for (int u = 0; u < 6; ++u) std::memcpy(qb0 + u * 1536, C + (size_t)(1 + u) * 6144, 1536); float d8[8]; DecGroup(qb0, d8); std::printf("q blk0 row0: "); for (int i = 0; i < 8; ++i) std::printf("%g ", d8[i]); std::printf("(exp byte %d)\n", qb0[0]); }
  std::printf("prep NT=%d: kv dispatch %.3f ms, q dispatch %.4f ms; Q rel err %.5f; header mismatches %ld; M max abs diff %.4f (mean %.4f, |M| max %.2f)\n", NT, kv_ms, best, std::sqrt(nq / dq), hdr_bad, mdiff, msum / mcount, mref_max);
  return 0;
}
