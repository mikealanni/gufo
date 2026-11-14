// QK-norm + RoPE stripe probe: sbtest-prep <xclbin> <insts> <NE> [reps]; 32*NE stripes of 16 tokens x 128 dims.
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

int main(int argc, char** argv) {
  const int NE = atoi(argv[3]), reps = argc > 4 ? atoi(argv[4]) : 10;
  const size_t ELEM = 6912, CELEM = 2304;
  const int NS = 32 * NE;
  const float eps = 1e-5f, theta = 1000000.f;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> w(128);
  for (auto& v : w) v = 0.5f + std::fabs(nd(rng));
  std::vector<uint8_t> A((size_t)NS * ELEM, 0), Cout((size_t)NS * CELEM, 0);
  std::vector<float> xd((size_t)NS * 16 * 128), ref((size_t)NS * 16 * 128);
  auto place = [&](int s) { int core = s % 32, j = s / 32; int col = core / 4, row = core % 4; return ((size_t)(col * NE + j) * 4 + row); };
  for (int s = 0; s < NS; ++s) {
    uint8_t* el = &A[place(s) * ELEM];
    std::vector<float> x(16 * 128);
    for (int t = 0; t < 16; ++t) for (int d = 0; d < 128; ++d) x[t * 128 + d] = nd(rng) * (d % 37 == 0 ? 4.f : 1.f) * (1.f + t % 3);
    for (int br = 0; br < 2; ++br) for (int sbc = 0; sbc < 8; ++sbc) for (int bc = 0; bc < 2; ++bc) for (int rib = 0; rib < 8; ++rib) {
      int t = br * 8 + rib, d0 = sbc * 16 + bc * 8;
      uint8_t* g = el + (sbc * 4 + br * 2 + bc) * 72 + rib * 9;
      EncGroup(&x[t * 128 + d0], g);
      DecGroup(g, &xd[((size_t)s * 16 + t) * 128 + d0]);
    }
    uint16_t* tab = (uint16_t*)(el + 2304);
    for (int br = 0; br < 2; ++br) for (int ib = 0; ib < 8; ++ib) for (int row = 0; row < 8; ++row) for (int col = 0; col < 8; ++col) {
      float pos = (float)(s * 16 + br * 8 + row);
      int i = ib * 8 + col;
      float ang = pos * std::pow(theta, -2.0f * i / 128.f);
      tab[((br * 8 + ib) * 2) * 64 + row * 8 + col] = Bf16(std::cos(ang));
      tab[((br * 8 + ib) * 2 + 1) * 64 + row * 8 + col] = Bf16(std::sin(ang));
    }
    uint16_t* wp = (uint16_t*)(el + 6400);
    for (int d = 0; d < 128; ++d) wp[d] = Bf16(w[d]);
    *(uint16_t*)(el + 6656) = Bf16(eps);
    for (int t = 0; t < 16; ++t) {
      const float* xr = &xd[((size_t)s * 16 + t) * 128];
      double ss = 0; for (int d = 0; d < 128; ++d) ss += (double)xr[d] * xr[d];
      float sc = 1.f / std::sqrt((float)(ss / 128) + eps);
      float pos = (float)(s * 16 + t);
      float y[128];
      for (int d = 0; d < 128; ++d) y[d] = xr[d] * sc * w[d];
      for (int i = 0; i < 64; ++i) {
        float ang = pos * std::pow(theta, -2.0f * i / 128.f), c = std::cos(ang), sn = std::sin(ang);
        ref[((size_t)s * 16 + t) * 128 + i] = y[i] * c - y[i + 64] * sn;
        ref[((size_t)s * 16 + t) * 128 + i + 64] = y[i + 64] * c + y[i] * sn;
      }
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
  xrt::bo bc(dev, Cout.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memcpy(ba.map<void*>(), A.data(), A.size());
  std::memset(bb.map<void*>(), 0, 4096);
  std::memset(bc.map<void*>(), 0, Cout.size());
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
  double num = 0, den = 0, maxabs = 0;
  for (int s = 0; s < NS; ++s) {
    const uint8_t* oe = C + place(s) * CELEM;
    for (int br = 0; br < 2; ++br) for (int sbc = 0; sbc < 8; ++sbc) for (int bc2 = 0; bc2 < 2; ++bc2) for (int rib = 0; rib < 8; ++rib) {
      int t = br * 8 + rib, d0 = sbc * 16 + bc2 * 8;
      float d8[8]; DecGroup(oe + (sbc * 4 + br * 2 + bc2) * 72 + rib * 9, d8);
      for (int i = 0; i < 8; ++i) { double rr = ref[((size_t)s * 16 + t) * 128 + d0 + i]; num += (d8[i] - rr) * (d8[i] - rr); den += rr * rr; maxabs = std::max(maxabs, std::fabs(d8[i] - rr)); }
    }
  }
  std::printf("prep NE=%d stripes=%d: rel err %.5f max abs %.4f; best %.4f ms\n", NE, NS, std::sqrt(num / den), maxabs, best);
  return 0;
}
