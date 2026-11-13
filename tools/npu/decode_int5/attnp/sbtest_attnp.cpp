// Prefill attention probe: one 128-row pass (8 columns x 16 tokens), causal prefix per column.
//   gufo-attnp <xclbin> <insts_attnp_MC.txt> <MC> <pos0> [reps=10]   (L = pos0+128, needs L <= MC*16)
#include <immintrin.h>
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

int main(int argc, char** argv) {
  const char* xclbin = argv[1];
  const char* insts = argv[2];
  const int mc = atoi(argv[3]);
  const int pos0 = atoi(argv[4]);
  const int reps = argc > 5 ? atoi(argv[5]) : 10;
  const int HD = 128, KVH = 4, QH = 16, NR = 128;
  const int L = pos0 + NR;
  if (L > mc * 16) { std::printf("L %d > mc*16\n", L); return 2; }
  const int qchunks = mc;
  const size_t UNIT = 4 * 1536, PCOL = 16 * UNIT;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0, 1);
  const float sc = 1.4426950408889634f / std::sqrt((float)HD);
  std::vector<float> Q((size_t)QH * NR * HD), K((size_t)KVH * L * HD), V((size_t)KVH * L * HD);
  for (auto& x : Q) x = nd(rng) * sc * 1.5f;
  for (auto& x : K) x = nd(rng) * 1.5f;
  for (auto& x : V) x = nd(rng);
  std::vector<float> Qd = Q, Kd = K, Vd = V;
  std::vector<uint8_t> A(8 * PCOL, 0), Bbuf((size_t)mc * 3 * UNIT, 0);
  for (int r = 0; r < KVH; ++r) {
    float kmax = 0;
    for (int t = 0; t < L; ++t) { float s2 = 0; for (int d = 0; d < HD; ++d) s2 += K[((size_t)r * L + t) * HD + d] * K[((size_t)r * L + t) * HD + d]; kmax = std::max(kmax, std::sqrt(s2)); }
    for (int c = 0; c < 8; ++c) {
      const int pos = pos0 + 16 * c, n = 16;
      uint8_t* core = &A[c * PCOL];
      auto unit = [&](int u) { return core + (size_t)u * UNIT + (size_t)r * 1536; };
      int32_t h[4] = {pos, n, (L - 1) / 16, 0};
      std::memcpy(unit(0), h, 16);
      float* mrow = (float*)(unit(0) + 128);
      std::vector<uint8_t> qb(9216);
      for (int rb = 0; rb < 8; ++rb) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
        int i = rb * 8 + rib, hq = r * 4 + i / 16, t = c * 16 + i % 16;
        float* src = &Q[((size_t)hq * NR + t) * HD + ks * 8];
        uint8_t* o = &qb[((size_t)(rb * 16 + ks) * 8 + rib) * 9];
        EncGroup(src, o);
        DecGroup(o, &Qd[((size_t)hq * NR + t) * HD + ks * 8]);
      }
      for (int u = 0; u < 6; ++u) std::memcpy(unit(1 + u), &qb[u * 1536], 1536);
      for (int i = 0; i < 64; ++i) {
        int hq = r * 4 + i / 16, t = c * 16 + i % 16;
        float qn = 0; for (int d = 0; d < HD; ++d) qn += Q[((size_t)hq * NR + t) * HD + d] * Q[((size_t)hq * NR + t) * HD + d];
        mrow[i] = std::sqrt(qn) * kmax;
      }
    }
  }
  for (int r = 0; r < KVH; ++r)
    for (int q = 0; q < qchunks; ++q) {
      uint8_t chunk[4608] = {0};
      for (int kb = 0; kb < 2; ++kb) {
        for (int ks = 0; ks < 16; ++ks) for (int tt = 0; tt < 8; ++tt) {
          int tok = q * 16 + kb * 8 + tt;
          if (tok >= L) continue;
          float* src = &K[((size_t)r * L + tok) * HD + ks * 8];
          uint8_t* o = chunk + ((size_t)(kb * 16 + ks) * 8 + tt) * 9;
          EncGroup(src, o); DecGroup(o, &Kd[((size_t)r * L + tok) * HD + ks * 8]);
        }
        for (int db = 0; db < 16; ++db) for (int g = 0; g < 8; ++g) {
          float v8[8], d8[8];
          for (int tt = 0; tt < 8; ++tt) { int tok = q * 16 + kb * 8 + tt; v8[tt] = tok < L ? V[((size_t)r * L + tok) * HD + db * 8 + g] : 0.f; }
          uint8_t* o = chunk + 2304 + ((size_t)(kb * 16 + db) * 8 + g) * 9;
          EncGroup(v8, o); DecGroup(o, d8);
          for (int tt = 0; tt < 8; ++tt) { int tok = q * 16 + kb * 8 + tt; if (tok < L) Vd[((size_t)r * L + tok) * HD + db * 8 + g] = d8[tt]; }
        }
      }
      for (int j = 0; j < 3; ++j) std::memcpy(&Bbuf[((size_t)q * 3 + j) * UNIT + (size_t)r * 1536], chunk + j * 1536, 1536);
    }
  for (int c = 0; c < 8; ++c) {
    const int pos = pos0 + 16 * c, total = pos + 16, t0 = pos & ~7;
    for (int r = 0; r < KVH; ++r) {
      uint8_t tl[4608] = {0};
      for (int tb = 0; tb < 3; ++tb) {
        uint8_t* dst = tl;
        std::memset(tl, 0, sizeof tl);
        int tokbase = t0 + tb * 8;
        if (tokbase < total) {
          for (int ks = 0; ks < 16; ++ks) for (int tt = 0; tt < 8; ++tt) {
            if (tokbase + tt >= total) continue;
            float v8[8]; for (int i = 0; i < 8; ++i) v8[i] = K[((size_t)r * L + tokbase + tt) * HD + ks * 8 + i];
            EncGroup(v8, dst + (ks * 8 + tt) * 9);
          }
          uint16_t* vd = (uint16_t*)(dst + 1152);
          for (int db = 0; db < 16; ++db) for (int tt = 0; tt < 8; ++tt) for (int d = 0; d < 8; ++d) {
            if (tokbase + tt >= total) continue;
            float v = V[((size_t)r * L + tokbase + tt) * HD + db * 8 + d];
            uint32_t u; std::memcpy(&u, &v, 4); vd[(db * 8 + tt) * 8 + d] = (uint16_t)(u >> 16);
          }
        }
        for (int e = 0; e < 3; ++e) std::memcpy(&A[c * PCOL + (size_t)(7 + tb * 3 + e) * UNIT + (size_t)r * 1536], tl + e * 1536, 1536);
      }
    }
  }
  const size_t c_bytes = (size_t)8 * 34 * 4 * 1024;
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(xclbin)};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  std::ifstream f(insts, std::ios::binary | std::ios::ate);
  size_t isz = f.tellg(); f.seekg(0);
  std::vector<char> iw(isz); f.read(iw.data(), isz);
  xrt::bo ib(dev, isz, XRT_BO_FLAGS_CACHEABLE, kern.group_id(1));
  std::memcpy(ib.map<void*>(), iw.data(), isz); ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo ba(dev, A.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bb(dev, Bbuf.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bc(dev, c_bytes, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memcpy(ba.map<void*>(), A.data(), A.size()); std::memcpy(bb.map<void*>(), Bbuf.data(), Bbuf.size());
  std::memset(bc.map<void*>(), 0, c_bytes);
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE); bb.sync(XCL_BO_SYNC_BO_TO_DEVICE); bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = kern(3, ib, (unsigned)(isz / 4), ba, bb, bc);
    { auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(20); for (;;) { auto st = run.state(); if (st == ERT_CMD_STATE_COMPLETED) break; if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED && st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) { std::printf("run ended in state %d\n", (int)st); return 3; } if (std::chrono::steady_clock::now() > t_end) { std::printf("timeout, state %d\n", (int)st); return 4; } } }
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint8_t* C = bc.map<uint8_t*>();
  double num = 0, den = 0, maxabs = 0;
  for (int c = 0; c < 8; ++c) for (int r = 0; r < KVH; ++r) {
    std::vector<double> O((size_t)64 * HD, 0.0), Ls(64, 0.0);
    for (int i = 0; i < 34; ++i) {
      const float* e = (const float*)(C + (((size_t)c * 34 + i) * 4 + r) * 1024);
      if (i < 32) {
        for (int k = 0; k < 256; ++k) { int tile = i * 4 + k / 64, lane = k % 64, rb = tile / 16, db = tile % 16; O[(size_t)(rb * 8 + lane / 8) * HD + db * 8 + lane % 8] += e[k]; }
      } else {
        for (int k = 0; k < 256; ++k) { int tile = (i - 32) * 4 + k / 64, lane = k % 64; Ls[tile * 8 + lane / 8] += e[k]; }
      }
    }
    for (int i = 0; i < 64; ++i) {
      int hq = r * 4 + i / 16, tl = i % 16, t = c * 16 + tl, lim = pos0 + t;
      std::vector<double> s(lim + 1);
      double mx = -1e30;
      for (int tok = 0; tok <= lim; ++tok) { double d = 0; for (int x = 0; x < HD; ++x) d += (double)Qd[((size_t)hq * NR + t) * HD + x] * Kd[((size_t)r * L + tok) * HD + x]; s[tok] = d; mx = std::max(mx, d); }
      double sum = 0; for (auto& v : s) { v = std::exp2(v - mx); sum += v; }
      for (int x = 0; x < HD; ++x) {
        double ref = 0; for (int tok = 0; tok <= lim; ++tok) ref += s[tok] * Vd[((size_t)r * L + tok) * HD + x];
        ref /= sum;
        double got = Ls[i] > 0 ? O[(size_t)i * HD + x] / Ls[i] : 0.0;
        num += (got - ref) * (got - ref); den += ref * ref; maxabs = std::max(maxabs, std::fabs(got - ref));
      }
    }
  }
  std::printf("attnp pos0=%d L=%d mc=%d: rel err %.5f max abs %.5f; best %.4f ms\n", pos0, L, mc, std::sqrt(num / den), maxabs, best);
  return 0;
}
