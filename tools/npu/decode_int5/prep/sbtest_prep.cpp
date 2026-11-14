// Attention prep probe: sbtest-prep <xclbin> <insts> <NQ> <NKV> [reps]; per core NQ Q blocks (4 heads x 16 tokens) and NKV K/V chunks.
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
static void Stripe(std::mt19937& rng, uint8_t* dst, float* xd) {
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> x(16 * 128);
  for (int t = 0; t < 16; ++t) for (int d = 0; d < 128; ++d) x[t * 128 + d] = nd(rng) * (d % 37 == 0 ? 4.f : 1.f) * (1.f + t % 3);
  for (int br = 0; br < 2; ++br) for (int sbc = 0; sbc < 8; ++sbc) for (int bc = 0; bc < 2; ++bc) for (int rib = 0; rib < 8; ++rib) {
    int t = br * 8 + rib, d0 = sbc * 16 + bc * 8;
    uint8_t* g = dst + (sbc * 4 + br * 2 + bc) * 72 + rib * 9;
    EncGroup(&x[t * 128 + d0], g);
    DecGroup(g, &xd[t * 128 + d0]);
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
  const int NQ = atoi(argv[3]), NKV = atoi(argv[4]), reps = argc > 5 ? atoi(argv[5]) : 10;
  const size_t EL = 13824, PIECE = 1536;
  const float eps = 1e-5f, theta = 1000000.f;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> wq(128), wk(128);
  for (auto& v : wq) v = (0.5f + std::fabs(nd(rng))) * 0.1275f;
  for (auto& v : wk) v = 0.5f + std::fabs(nd(rng));
  const size_t col_in = (size_t)(NQ + NKV) * 4 * EL, col_out = (size_t)(NQ * 6 + NKV * 3) * 4 * PIECE;
  std::vector<uint8_t> A(8 * col_in, 0), O(8 * col_out, 0);
  std::vector<float> qref((size_t)8 * 4 * NQ * 4 * 16 * 128), kref((size_t)8 * 4 * NKV * 16 * 128), vd((size_t)8 * 4 * NKV * 16 * 128);
  for (int c = 0; c < 8; ++c) for (int r = 0; r < 4; ++r) {
    for (int j = 0; j < NQ; ++j) {
      uint8_t* el = &A[(size_t)c * col_in + ((size_t)j * 4 + r) * EL];
      int pos0 = (c * 4 + r) * 977 + j * 16;
      for (int hq = 0; hq < 4; ++hq) {
        std::vector<float> xd(16 * 128);
        Stripe(rng, el + hq * 2304, xd.data());
        RefNR(xd.data(), wq, pos0, eps, theta, &qref[((((size_t)(c * 4 + r) * NQ + j) * 4) + hq) * 16 * 128]);
      }
      MakeTab(el + 9216, pos0, theta);
      uint16_t* wp = (uint16_t*)(el + 13312);
      for (int d = 0; d < 128; ++d) wp[d] = Bf16(wq[d]);
      *(uint16_t*)(el + 13568) = Bf16(eps);
    }
    for (int j = 0; j < NKV; ++j) {
      uint8_t* el = &A[(size_t)c * col_in + ((size_t)(NQ + j) * 4 + r) * EL];
      int pos0 = (c * 4 + r) * 523 + j * 16;
      size_t idx = (size_t)(c * 4 + r) * NKV + j;
      std::vector<float> xk(16 * 128);
      Stripe(rng, el, xk.data());
      RefNR(xk.data(), wk, pos0, eps, theta, &kref[idx * 16 * 128]);
      Stripe(rng, el + 2304, &vd[idx * 16 * 128]);
      MakeTab(el + 4608, pos0, theta);
      uint16_t* wp = (uint16_t*)(el + 8704);
      for (int d = 0; d < 128; ++d) wp[d] = Bf16(wk[d]);
      *(uint16_t*)(el + 8960) = Bf16(eps);
    }
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
  xrt::bo ba(dev, A.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bb(dev, 4096, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bc(dev, O.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memcpy(ba.map<void*>(), A.data(), A.size());
  std::memset(bb.map<void*>(), 0, 4096);
  std::memset(bc.map<void*>(), 0, O.size());
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
  double nq = 0, dq = 0, nk = 0, dk = 0, nv = 0, dv = 0;
  for (int c = 0; c < 8; ++c) for (int r = 0; r < 4; ++r) {
    const uint8_t* co = C + (size_t)c * col_out;
    for (int j = 0; j < NQ; ++j) {
      uint8_t qb[9216];
      for (int u = 0; u < 6; ++u) std::memcpy(qb + u * 1536, co + (((size_t)j * 6 + u) * 4 + r) * PIECE, 1536);
      for (int hq = 0; hq < 4; ++hq) for (int br = 0; br < 2; ++br) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
        float d8[8]; DecGroup(qb + ((hq * 2 + br) * 16 + ks) * 72 + rib * 9, d8);
        const float* rr = &qref[((((size_t)(c * 4 + r) * NQ + j) * 4) + hq) * 16 * 128 + (br * 8 + rib) * 128 + ks * 8];
        for (int i = 0; i < 8; ++i) { nq += (double)(d8[i] - rr[i]) * (d8[i] - rr[i]); dq += (double)rr[i] * rr[i]; }
      }
    }
    for (int j = 0; j < NKV; ++j) {
      uint8_t ch[4608];
      const size_t base = (size_t)NQ * 6 * 4 * PIECE;
      for (int e = 0; e < 3; ++e) std::memcpy(ch + e * 1536, co + base + (((size_t)j * 3 + e) * 4 + r) * PIECE, 1536);
      size_t idx = (size_t)(c * 4 + r) * NKV + j;
      for (int br = 0; br < 2; ++br) for (int ks = 0; ks < 16; ++ks) for (int rib = 0; rib < 8; ++rib) {
        float d8[8]; DecGroup(ch + (br * 16 + ks) * 72 + rib * 9, d8);
        const float* rr = &kref[idx * 16 * 128 + (br * 8 + rib) * 128 + ks * 8];
        for (int i = 0; i < 8; ++i) { nk += (double)(d8[i] - rr[i]) * (d8[i] - rr[i]); dk += (double)rr[i] * rr[i]; }
      }
      for (int br = 0; br < 2; ++br) for (int db = 0; db < 16; ++db) for (int g = 0; g < 8; ++g) {
        float d8[8]; DecGroup(ch + 2304 + (br * 16 + db) * 72 + g * 9, d8);
        for (int i8 = 0; i8 < 8; ++i8) { double rr = vd[idx * 16 * 128 + (br * 8 + i8) * 128 + db * 8 + g]; nv += (d8[i8] - rr) * (d8[i8] - rr); dv += rr * rr; }
      }
    }
  }
  std::printf("prep NQ=%d NKV=%d: rel err Q %.5f K %.5f V %.5f; best %.4f ms\n", NQ, NKV, std::sqrt(nq / dq), std::sqrt(nk / dk), std::sqrt(nv / dv), best);
  return 0;
}
