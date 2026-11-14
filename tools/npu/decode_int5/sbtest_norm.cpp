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
static uint16_t Bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
// A wire: tile (rowblk, kt), block ((sbr*8+sbc)*2+bis)*72, 8 rows x 9 B
static size_t Off(int r, int c, int K) {
  int rb = r / 128, rr = r % 128, kt = c / 64, cc = c % 64;
  int sbr = rr / 16, bis = (rr % 16) / 8, rib = rr % 8, sbc = cc / 8;
  return ((size_t)rb * (K / 64) + kt) * 9216 + ((sbr * 8 + sbc) * 2 + bis) * 72 + rib * 9;
}
int main(int argc, char** argv) {
  const int M = 1024, K = 2048;
  const float eps = 1e-5f;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> X((size_t)M * K), g(K), Xd((size_t)M * K);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < K; ++c) X[(size_t)r * K + c] = nd(rng) * (c % 97 == 0 ? 8.f : 1.f) * (1.f + (r % 5));
  for (auto& v : g) v = 0.5f + 2.5f * std::fabs(nd(rng)) * 0.5f;
  std::vector<uint8_t> A((size_t)M * K / 8 * 9, 0), B(9216 * 8, 0), C(A.size(), 0);
  for (int r = 0; r < M; ++r)
    for (int c8 = 0; c8 < K / 8; ++c8) {
      uint8_t* p = &A[Off(r, c8 * 8, K)];
      EncGroup(&X[(size_t)r * K + c8 * 8], p);
      DecGroup(p, &Xd[(size_t)r * K + c8 * 8]);
    }
  for (int i = 0; i < K; ++i) {
    uint16_t b = Bf16(g[i]);
    std::memcpy(&B[i * 2], &b, 2);
  }
  { uint16_t b = Bf16(eps); std::memcpy(&B[4096], &b, 2); }
  for (int q = 1; q < 8; ++q) std::memcpy(&B[q * 9216], &B[0], 9216);
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(argv[1])};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
  size_t isz = f.tellg();
  f.seekg(0);
  std::vector<char> iw(isz);
  f.read(iw.data(), isz);
  xrt::bo ib(dev, isz, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
  std::memcpy(ib.map<void*>(), iw.data(), isz);
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo ba(dev, A.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bb(dev, B.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bc(dev, C.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memcpy(ba.map<void*>(), A.data(), A.size());
  std::memcpy(bb.map<void*>(), B.data(), B.size());
  std::memset(bc.map<void*>(), 0, C.size());
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  for (int rep = 0; rep < 3; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = kern(3, ib, (unsigned)(isz / 4), ba, bb, bc);
    auto t_end = t0 + std::chrono::seconds(20);
    for (;;) {
      auto st = run.state();
      if (st == ERT_CMD_STATE_COMPLETED) break;
      if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED && st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) { std::printf("run ended in state %d\n", (int)st); return 3; }
      if (std::chrono::steady_clock::now() > t_end) { std::printf("timeout, state %d\n", (int)st); return 4; }
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("rep %d: %.3f ms\n", rep, ms);
  }
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(C.data(), bc.map<void*>(), C.size());
  double se = 0, sr = 0, sq = 0, mx = 0;
  size_t nz = 0;
  std::vector<float> row(K);
  for (int r = 0; r < M; ++r) {
    double ss = 0;
    for (int c = 0; c < K; ++c) ss += (double)Xd[(size_t)r * K + c] * Xd[(size_t)r * K + c];
    double sc = 1.0 / std::sqrt(ss / K + eps);
    for (int c8 = 0; c8 < K / 8; ++c8) {
      float d[8], e8[8];
      DecGroup(&C[Off(r, c8 * 8, K)], d);
      for (int i = 0; i < 8; ++i) e8[i] = (float)(Xd[(size_t)r * K + c8 * 8 + i] * sc * g[c8 * 8 + i]);
      uint8_t tmp[9];
      float q[8];
      EncGroup(e8, tmp);
      DecGroup(tmp, q);
      for (int i = 0; i < 8; ++i) {
        se += (double)(d[i] - e8[i]) * (d[i] - e8[i]);
        sq += (double)(q[i] - e8[i]) * (q[i] - e8[i]);
        sr += (double)e8[i] * e8[i];
        if (d[i] != 0) ++nz;
      }
    }
  }
  std::printf("rel rms err device %.5f  (bfp-quant floor %.5f)  nonzero %zu/%zu\n", std::sqrt(se / sr), std::sqrt(sq / sr), nz, (size_t)M * K);
  return 0;
}
