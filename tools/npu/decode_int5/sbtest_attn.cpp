#include <immintrin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

static void EncGroup(const float* v, uint8_t* out) {
  unsigned max_exp = 0;
  for (int i = 0; i < 8; ++i) {
    uint32_t u;
    std::memcpy(&u, &v[i], 4);
    max_exp = std::max(max_exp, (u >> 23) & 0xFFu);
  }
  uint32_t ib = max_exp >= 7 ? (260u - max_exp) << 23 : 0u;
  float inv;
  std::memcpy(&inv, &ib, 4);
  out[0] = (uint8_t)max_exp;
  for (int i = 0; i < 8; ++i) {
    float q = std::fmin(127.0f, std::fmax(-128.0f, std::nearbyint(v[i] * inv)));
    out[1 + i] = (uint8_t)(int8_t)q;
  }
}
static void DecGroup(const uint8_t* p, float* v) {
  float mult = std::ldexp(1.0f, (int)p[0] - 127) / 64.0f;
  for (int i = 0; i < 8; ++i) v[i] = (float)(int8_t)p[1 + i] * mult;
}
static float bf16r(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  u &= 0xFFFF0000u;
  float r;
  std::memcpy(&r, &u, 4);
  return r;
}

int main(int argc, char** argv) {
  const char* xclbin = argv[1];
  const char* insts = argv[2];
  const int L = atoi(argv[3]);      // total keys (pos + n)
  const int n = argc > 4 ? atoi(argv[4]) : 16;
  const int reps = argc > 5 ? atoi(argv[5]) : 10;
  const int pos = L - n, HD = 128, KVH = 4, QH = 16;
  const int qchunks = (L + 15) / 16, mchunks = (qchunks + 7) / 8;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0, 1);
  const float log2e = 1.4426950408889634f, sc = log2e / std::sqrt((float)HD);
  std::vector<float> Q((size_t)QH * 16 * HD, 0.f), K((size_t)KVH * L * HD), V((size_t)KVH * L * HD);
  for (int h = 0; h < QH; ++h) for (int t = 0; t < n; ++t) for (int d = 0; d < HD; ++d) Q[((size_t)h * 16 + t) * HD + d] = nd(rng) * sc * (getenv("QS") ? (float)atof(getenv("QS")) : 1.5f);
  for (auto& x : K) x = nd(rng) * 1.5f;
  for (auto& x : V) x = nd(rng);
  // quantize + build device formats; keep decoded copies for the reference
  std::vector<float> Qd = Q, Kd = K, Vd = V;
  std::vector<uint8_t> A((size_t)33 * 128 * 9, 0);
  std::vector<uint8_t> Bbuf((size_t)8 * mchunks * 3 * 4 * 1536, 0);
  std::vector<float> Mrow((size_t)4 * 64, 0.f);
  for (int r = 0; r < KVH; ++r) {
    float kmax = 0;
    for (int t = 0; t < L; ++t) { float s2 = 0; for (int d = 0; d < HD; ++d) s2 += K[((size_t)r * L + t) * HD + d] * K[((size_t)r * L + t) * HD + d]; kmax = std::max(kmax, std::sqrt(s2)); }
    for (int i = 0; i < 64; ++i) {
      int hq = r * 4 + i / 16, t = i % 16;
      float qn = 0; for (int d = 0; d < HD; ++d) qn += Q[((size_t)hq * 16 + t) * HD + d] * Q[((size_t)hq * 16 + t) * HD + d];
      Mrow[r * 64 + i] = std::sqrt(qn) * kmax;
    }
    // Q elements: [rb][ks] blocks of 8 groups (rib)
    uint8_t* qb = &A[(size_t)(1 + r * 8) * 1152];
    for (int rb = 0; rb < 8; ++rb) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
      int i = rb * 8 + rib, hq = r * 4 + i / 16, t = i % 16;
      float* src = &Q[((size_t)hq * 16 + t) * HD + ks * 8];
      uint8_t* o = qb + ((size_t)(rb * 16 + ks) * 8 + rib) * 9;
      EncGroup(src, o);
      DecGroup(o, &Qd[((size_t)hq * 16 + t) * HD + ks * 8]);
    }
  }
  for (int r = 0; r < KVH; ++r)
    for (int q = 0; q < qchunks; ++q) {
      int c = q % 8, m = q / 8;
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
      for (int j = 0; j < 3; ++j)
        std::memcpy(&Bbuf[((((size_t)q) * 3 + j) * 4 + r) * 1536], chunk + j * 1536, 1536); (void)c; (void)m;
    }
  int32_t hdr[2] = {pos, n};
  std::memcpy(A.data(), hdr, 8);
  std::memcpy(A.data() + 128, Mrow.data(), Mrow.size() * 4);
  const size_t c_bytes = (size_t)8 * 34 * 4 * 1024;
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(xclbin)};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  std::ifstream f(insts, std::ios::binary | std::ios::ate);
  size_t isz = f.tellg(); f.seekg(0);
  std::vector<char> iw(isz); f.read(iw.data(), isz);
  xrt::bo ib(dev, isz, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
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
  // merge
  std::vector<double> O((size_t)KVH * 64 * HD, 0.0), Ls((size_t)KVH * 64, 0.0);
  for (int c = 0; c < 8; ++c) for (int r = 0; r < KVH; ++r) {
    for (int i = 0; i < 34; ++i) {
      const float* e = (const float*)(C + (((size_t)c * 34 + i) * 4 + r) * 1024);
      if (i < 32) {
        for (int k = 0; k < 256; ++k) {
          int tile = i * 4 + k / 64, lane = k % 64, rb = tile / 16, db = tile % 16;
          int row = rb * 8 + lane / 8, dim = db * 8 + lane % 8;
          O[((size_t)r * 64 + row) * HD + dim] += e[k];
        }
      } else {
        for (int k = 0; k < 256; ++k) { int tile = (i - 32) * 4 + k / 64, lane = k % 64; Ls[(size_t)r * 64 + tile * 8 + lane / 8] += e[k]; }
      }
    }
  }
  double num = 0, den = 0; double maxabs = 0;
  for (int r = 0; r < KVH; ++r) for (int i = 0; i < 64; ++i) {
    int hq = r * 4 + i / 16, t = i % 16;
    if (t >= n) continue;
    int lim = pos + t;
    std::vector<double> s(lim + 1);
    double mx = -1e30;
    for (int tok = 0; tok <= lim; ++tok) { double d = 0; for (int x = 0; x < HD; ++x) d += (double)Qd[((size_t)hq * 16 + t) * HD + x] * Kd[((size_t)r * L + tok) * HD + x]; s[tok] = d; mx = std::max(mx, d); }
    double sum = 0; for (auto& v : s) { v = std::exp2(v - mx); sum += v; }
    for (int x = 0; x < HD; ++x) {
      double ref = 0; for (int tok = 0; tok <= lim; ++tok) ref += s[tok] * Vd[((size_t)r * L + tok) * HD + x];
      ref /= sum;
      double got = Ls[(size_t)r * 64 + i] > 0 ? O[((size_t)r * 64 + i) * HD + x] / Ls[(size_t)r * 64 + i] : 0.0;
      num += (got - ref) * (got - ref); den += ref * ref; maxabs = std::max(maxabs, std::fabs(got - ref));
    }
  }
  std::printf("attn L=%d n=%d: rel err %.5f max abs %.5f; best %.4f ms (kv %.1f MB)\n", L, n, std::sqrt(num / den), maxabs, best, 8.0 * mchunks * 3 * 6144 / 1e6);
  return 0;
}
