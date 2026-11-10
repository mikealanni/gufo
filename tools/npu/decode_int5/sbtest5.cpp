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

int main(int argc, char** argv) {
  const char* xclbin = argv[1];
  const char* insts = argv[2];
  const int K = atoi(argv[3]), N = atoi(argv[4]), reps = argc > 5 ? atoi(argv[5]) : 20;
  const int M = 16;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> X((size_t)M * K), W((size_t)N * K);
  for (auto& x : X) x = nd(rng);
  for (auto& w : W) w = nd(rng) * 0.05f;
  const int tk = K / 64;
  std::vector<uint8_t> A((size_t)M * K / 8 * 9), B, C((size_t)M * N / 8 * 9);
  std::vector<float> Xd((size_t)M * K), Wd((size_t)N * K);
  std::vector<float> scl((size_t)N * (K / 32));
  std::vector<int> qv((size_t)N * K);
  for (int t = 0; t < tk; ++t)
    for (int gi = 0; gi < 128; ++gi) {
      int rib = gi & 7, block = (gi >> 3) & 1, sc = gi >> 4;
      int row = block * 8 + rib;
      float v[8];
      for (int i = 0; i < 8; ++i) v[i] = X[(size_t)row * K + t * 64 + sc * 8 + i];
      uint8_t* o = &A[((size_t)t * 128 + gi) * 9];
      EncGroup(v, o);
      float d[8];
      DecGroup(o, d);
      for (int i = 0; i < 8; ++i) Xd[(size_t)row * K + t * 64 + sc * 8 + i] = d[i];
    }
  const int strips = N / 128;
  const int BQ = 1536;
  B.assign((size_t)strips * tk * 4 * BQ, 0);
  auto bf16r = [](float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    u &= 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
  };
  std::vector<float> Wex((size_t)N * K);
  for (int n = 0; n < N; ++n)
    for (int kg = 0; kg < K / 32; ++kg) {
      float amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(W[(size_t)n * K + kg * 32 + i]));
      float sc = bf16r(amax / 15.0f);
      if (sc == 0) sc = 1;
      for (int i = 0; i < 32; ++i) {
        float q = std::fmin(15.f, std::fmax(-15.f, std::nearbyint(W[(size_t)n * K + kg * 32 + i] / sc)));
        Wex[(size_t)n * K + kg * 32 + i] = q * sc;
        scl[(size_t)n * (K / 32) + kg] = sc;
        qv[(size_t)n * K + kg * 32 + i] = (int)q;
      }
    }
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < K / 8; ++g) {
      float d[8];
      uint8_t tmp[9];
      EncGroup(&Wex[(size_t)n * K + g * 8], tmp);
      DecGroup(tmp, d);
      for (int i = 0; i < 8; ++i) Wd[(size_t)n * K + g * 8 + i] = d[i];
    }
  for (int s2 = 0; s2 < strips; ++s2)
    for (int t = 0; t < tk; ++t)
      for (int r = 0; r < 4; ++r) {
        uint8_t* base = &B[(((size_t)s2 * tk + t) * 4 + r) * BQ];
        uint16_t* words = (uint16_t*)base;
        uint16_t* scl16 = (uint16_t*)(base + 1408);
        for (int p = 0; p < 32; ++p) {
          int sb = p >> 4, sr = (p >> 1) & 7, blk = p & 1;
          for (int lane = 0; lane < 64; ++lane) {
            int cib = lane >> 3, kk = lane & 7;
            int n = s2 * 128 + r * 32 + sb * 16 + blk * 8 + cib;
            int x = qv[(size_t)n * K + t * 64 + sr * 8 + kk] + 16;
            words[(p / 3) * 64 + lane] |= (uint16_t)(x << (5 * (p % 3)));
          }
        }
        for (int sb = 0; sb < 2; ++sb)
          for (int blk = 0; blk < 2; ++blk)
            for (int q = 0; q < 2; ++q)
              for (int cib = 0; cib < 8; ++cib) {
                int n = s2 * 128 + r * 32 + sb * 16 + blk * 8 + cib;
                float sc = scl[(size_t)n * (K / 32) + t * 2 + q];
                uint32_t u;
                std::memcpy(&u, &sc, 4);
                scl16[((sb * 2 + blk) * 2 + q) * 8 + cib] = (uint16_t)(u >> 16);
              }
      }
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(xclbin)};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  std::ifstream f(insts, std::ios::binary | std::ios::ate);
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
  double best = 1e9, tot = 0;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = kern(3, ib, (unsigned)(isz / 4), ba, bb, bc);
    run.wait();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, ms);
    tot += ms;
  }
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(C.data(), bc.map<void*>(), C.size());
  std::vector<float> Y((size_t)M * N, 0), R((size_t)M * N, 0);
  for (int s = 0; s < strips; ++s)
    for (int r = 0; r < 4; ++r)
      for (int G = 0; G < 64; ++G) {
        int sb = G >> 5, acc = (G >> 3) & 3, rib = G & 7;
        int br = acc >> 1, bcc = acc & 1;
        int row = br * 8 + rib, col = s * 128 + r * 32 + sb * 16 + bcc * 8;
        float d[8];
        DecGroup(&C[(((size_t)s * 256) + r * 64 + G) * 9], d);
        for (int i = 0; i < 8; ++i) Y[(size_t)row * N + col + i] = d[i];
      }
  double num = 0, den = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double a = 0;
      for (int k = 0; k < K; ++k) a += (double)Xd[(size_t)m * K + k] * Wd[(size_t)n * K + k];
      R[(size_t)m * N + n] = (float)a;
      num += (a - Y[(size_t)m * N + n]) * (a - Y[(size_t)m * N + n]);
      den += a * a;
    }
  double wbytes = (double)B.size();
  std::printf("K=%d N=%d rel_rms_err=%.5f best=%.4f ms avg=%.4f ms  weights %.1f MB = %.1f GB/s (best)\n", K, N, std::sqrt(num / den), best, tot / reps, wbytes / 1e6, wbytes / (best * 1e-3) / 1e9);
  return 0;
}
