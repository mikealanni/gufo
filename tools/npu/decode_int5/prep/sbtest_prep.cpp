// Attention prep probe fed from a synthetic QKV GEMM result (C wire): sbtest-prep <xclbin> <insts> [reps]; NT=8 token tiles.
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


int main(int argc, char** argv) {
  const int NT = 8, reps = argc > 3 ? atoi(argv[3]) : 10;
  const size_t TILE = 18432, PIECE = 1536, TE = 5120, PASS = 8 * 98304, PCOL = 98304;
  const size_t QKV_BYTES = (size_t)8 * 3 * 32 * TILE, TAB_BYTES = (size_t)NT * 8 * TE;
  const size_t Q_BYTES = (NT - 1) * PASS + 8 * PCOL, KV_BYTES = (size_t)NT * 8 * 18432, KN_BYTES = (size_t)NT * 8 * 6144;
  const float eps = 1e-5f, theta = 1000000.f;
  const int pos_base = 4096;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> wq(128), wk(128);
  for (auto& v : wq) v = (0.5f + std::fabs(nd(rng))) * 0.1275f;
  for (auto& v : wk) v = 0.5f + std::fabs(nd(rng));
  std::vector<uint8_t> QKV(QKV_BYTES, 0), TAB(TAB_BYTES, 0), OUT(Q_BYTES + KV_BYTES + KN_BYTES, 0);
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
  for (int g = 0; g < NT * 8; ++g) {
    uint8_t* te = &TAB[(size_t)g * TE];
    MakeTab(te, pos_base + g * 16, theta);
    uint16_t* wp = (uint16_t*)(te + 4096);
    for (int d = 0; d < 128; ++d) { wp[d] = Bf16(wq[d]); wp[128 + d] = Bf16(wk[d]); }
    *(uint16_t*)(te + 4608) = Bf16(eps);
  }
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(argv[1])};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
  size_t isz = f.tellg(); f.seekg(0);
  std::vector<char> iw(isz); f.read(iw.data(), isz);
  xrt::bo ib(dev, isz, XRT_BO_FLAGS_CACHEABLE, kern.group_id(1));
  std::memcpy(ib.map<void*>(), iw.data(), isz); ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo ba(dev, QKV.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bb(dev, TAB.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bc(dev, OUT.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memcpy(ba.map<void*>(), QKV.data(), QKV.size());
  std::memcpy(bb.map<void*>(), TAB.data(), TAB.size());
  std::memset(bc.map<void*>(), 0, OUT.size());
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE); bb.sync(XCL_BO_SYNC_BO_TO_DEVICE); bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = kern(3, ib, (unsigned)(isz / 4), ba, bb, bc);
    auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    for (;;) { auto st = run.state(); if (st == ERT_CMD_STATE_COMPLETED) break; if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED && st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) { std::printf("run ended in state %d\n", (int)st); return 3; } if (std::chrono::steady_clock::now() > t_end) { std::printf("timeout, state %d\n", (int)st); return 4; } }
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint8_t* C = bc.map<uint8_t*>();
  double nq = 0, dq = 0, nk = 0, dk = 0, nv = 0, dv = 0, ntv = 0, dtv = 0, nkn = 0, dkn = 0;
  long tail_k_bad = 0;
  std::vector<float> ref(16 * 128), sub(16 * 128);
  for (int mt = 0; mt < NT; ++mt) for (int c = 0; c < 8; ++c) for (int r = 0; r < 4; ++r) {
    const int g = mt * 8 + c;
    auto stripe = [&](int nt, float* dst) { for (int t = 0; t < 16; ++t) for (int d = 0; d < 128; ++d) dst[t * 128 + d] = xd[(size_t)mt * 24 + nt][(c * 16 + t) * 128 + d]; };
    uint8_t qb[9216];
    for (int u = 0; u < 6; ++u) std::memcpy(qb + u * 1536, C + (size_t)mt * PASS + (size_t)c * PCOL + (size_t)(1 + u) * 6144 + r * PIECE, 1536);
    for (int hq = 0; hq < 4; ++hq) {
      stripe(r * 4 + hq, sub.data());
      RefNR(sub.data(), wq, pos_base + g * 16, eps, theta, ref.data());
      for (int br = 0; br < 2; ++br) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
        float d8[8]; DecGroup(qb + ((hq * 2 + br) * 16 + ks) * 72 + rib * 9, d8);
        const float* rr = &ref[(br * 8 + rib) * 128 + ks * 8];
        for (int i = 0; i < 8; ++i) { nq += (double)(d8[i] - rr[i]) * (d8[i] - rr[i]); dq += (double)rr[i] * rr[i]; }
      }
    }
    uint8_t ch[4608];
    for (int e = 0; e < 3; ++e) std::memcpy(ch + e * 1536, C + Q_BYTES + (size_t)g * 18432 + e * 6144 + r * PIECE, 1536);
    stripe(16 + r, sub.data());
    RefNR(sub.data(), wk, pos_base + g * 16, eps, theta, ref.data());
    for (int br = 0; br < 2; ++br) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
      float d8[8]; DecGroup(ch + (br * 16 + ks) * 72 + rib * 9, d8);
      const float* rr = &ref[(br * 8 + rib) * 128 + ks * 8];
      for (int i = 0; i < 8; ++i) { nk += (double)(d8[i] - rr[i]) * (d8[i] - rr[i]); dk += (double)rr[i] * rr[i]; }
    }
    stripe(16 + r, sub.data());
    RefNR(sub.data(), wk, pos_base + g * 16, eps, theta, ref.data());
    {
      const float* kn = (const float*)(C + Q_BYTES + KV_BYTES + (size_t)g * 6144 + r * PIECE);
      for (int t = 0; t < 16; ++t) {
        double ss = 0; for (int d = 0; d < 128; ++d) ss += (double)ref[t * 128 + d] * ref[t * 128 + d];
        double got = kn[(t / 8) * 64 + (t % 8) * 8];
        nkn += (got - ss) * (got - ss); dkn += ss * ss;
      }
    }
    stripe(20 + r, sub.data());
    for (int tb = 0; tb < 2; ++tb) {
      uint8_t tl[4608];
      for (int e = 0; e < 3; ++e) std::memcpy(tl + e * 1536, C + (size_t)mt * PASS + (size_t)c * PCOL + (size_t)(7 + tb * 3 + e) * 6144 + r * PIECE, 1536);
      if (std::memcmp(tl, ch + tb * 1152, 1152) != 0) ++tail_k_bad;
      const uint16_t* vd = (const uint16_t*)(tl + 1152);
      for (int db = 0; db < 16; ++db) for (int tt = 0; tt < 8; ++tt) for (int d = 0; d < 8; ++d) {
        double rr = sub[(tb * 8 + tt) * 128 + db * 8 + d], got = BfF(vd[(db * 8 + tt) * 8 + d]);
        ntv += (got - rr) * (got - rr); dtv += rr * rr;
      }
    }
    for (int br = 0; br < 2; ++br) for (int db = 0; db < 16; ++db) for (int gg = 0; gg < 8; ++gg) {
      float d8[8]; DecGroup(ch + 2304 + (br * 16 + db) * 72 + gg * 9, d8);
      for (int i8 = 0; i8 < 8; ++i8) { double rr = sub[(br * 8 + i8) * 128 + db * 8 + gg]; nv += (d8[i8] - rr) * (d8[i8] - rr); dv += rr * rr; }
    }
  }
  std::printf("prep NT=%d (%d tokens): rel err Q %.5f K %.5f V %.5f tailV %.5f |k|^2 %.5f; tail K mismatches %ld; best %.4f ms\n", NT, NT * 128, std::sqrt(nq / dq), std::sqrt(nk / dk), std::sqrt(nv / dv), std::sqrt(ntv / dtv), std::sqrt(nkn / dkn), tail_k_bad, best);
  return 0;
}
