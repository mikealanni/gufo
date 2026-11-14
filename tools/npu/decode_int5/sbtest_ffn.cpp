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
// Quantize W[N][K] to int5+bf16 scale, pack, and return the dequantized bfp16 re-encoded matrix.
static void QuantPack(const std::vector<float>& W, int N, int K, std::vector<uint8_t>& B, std::vector<float>& Wd) {
  const int strips = N / 128, tk = K / 64, BQ = 1536;
  B.assign((size_t)strips * tk * 4 * BQ, 0);
  Wd.assign((size_t)N * K, 0);
  std::vector<int> qv((size_t)N * K);
  std::vector<float> scl((size_t)N * (K / 32));
  std::vector<float> Wex((size_t)N * K);
  for (int n = 0; n < N; ++n)
    for (int kg = 0; kg < K / 32; ++kg) {
      float amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(W[(size_t)n * K + kg * 32 + i]));
      float sc = bf16r(amax / 15.0f);
      if (sc == 0) sc = 1;
      scl[(size_t)n * (K / 32) + kg] = sc;
      for (int i = 0; i < 32; ++i) {
        float q = std::fmin(15.f, std::fmax(-15.f, std::nearbyint(W[(size_t)n * K + kg * 32 + i] / sc)));
        qv[(size_t)n * K + kg * 32 + i] = (int)q;
        Wex[(size_t)n * K + kg * 32 + i] = q * sc;
      }
    }
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < K / 8; ++g) {
      uint8_t tmp[9];
      EncGroup(&Wex[(size_t)n * K + g * 8], tmp);
      DecGroup(tmp, &Wd[(size_t)n * K + g * 8]);
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
}

int main(int argc, char** argv) {
  const char* xclbin = argv[1];
  const char* insts = argv[2];
  const int reps = argc > 3 ? atoi(argv[3]) : 20;
  const int M = 16, K1 = 2048, I = 7680, N1 = 15360, N2 = 2048;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> X((size_t)M * K1), Wg((size_t)I * K1), Wu((size_t)I * K1), Wdn((size_t)N2 * I);
  for (auto& x : X) x = nd(rng);
  const float wstd = getenv("WSTD") ? atof(getenv("WSTD")) : 0.06f;
  for (auto& w : Wg) w = nd(rng) * wstd;
  for (auto& w : Wu) w = nd(rng) * 0.03f;
  for (auto& w : Wdn) w = nd(rng) * 0.03f;
  // interleave gate/up rows at 8-row granularity
  std::vector<float> Wgu((size_t)N1 * K1);
  for (int f = 0; f < I / 8; ++f)
    for (int i = 0; i < 8; ++i) {
      std::memcpy(&Wgu[(size_t)(f * 16 + i) * K1], &Wg[(size_t)(f * 8 + i) * K1], K1 * 4);
      std::memcpy(&Wgu[(size_t)(f * 16 + 8 + i) * K1], &Wu[(size_t)(f * 8 + i) * K1], K1 * 4);
    }
  std::vector<uint8_t> Bgu, Bdn;
  std::vector<float> Wgu_d, Wdn_d;
  QuantPack(Wgu, N1, K1, Bgu, Wgu_d);
  QuantPack(Wdn, N2, I, Bdn, Wdn_d);
  std::vector<uint8_t> A((size_t)M * K1 / 8 * 9), B, C((size_t)(120 * 256 + 16 * 256) * 9);
  B = Bgu;
  B.insert(B.end(), Bdn.begin(), Bdn.end());
  std::vector<float> Xd((size_t)M * K1);
  for (int t = 0; t < K1 / 64; ++t)
    for (int gi = 0; gi < 128; ++gi) {
      int rib = gi & 7, block = (gi >> 3) & 1, sc = gi >> 4;
      int row = block * 8 + rib;
      float v[8], d[8];
      for (int i = 0; i < 8; ++i) v[i] = X[(size_t)row * K1 + t * 64 + sc * 8 + i];
      uint8_t* o = &A[((size_t)t * 128 + gi) * 9];
      EncGroup(v, o);
      DecGroup(o, d);
      for (int i = 0; i < 8; ++i) Xd[(size_t)row * K1 + t * 64 + sc * 8 + i] = d[i];
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
    while (run.state() != ERT_CMD_STATE_COMPLETED) {}
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, ms);
    tot += ms;
  }
  {
    // queued submission: K runs in flight, one wait per batch; and reuse of one run object
    for (int K : {2, 4}) {
      std::vector<xrt::run> runs;
      for (int i = 0; i < K; ++i) {
        xrt::run r(kern);
        r.set_arg(0, 3u); r.set_arg(1, ib); r.set_arg(2, (unsigned)(isz / 4)); r.set_arg(3, ba); r.set_arg(4, bb); r.set_arg(5, bc);
        runs.push_back(std::move(r));
      }
      double bestq = 1e9;
      for (int rep = 0; rep < 10; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (auto& r : runs) r.start();
        for (auto& r : runs) r.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / K;
        bestq = std::min(bestq, ms);
      }
      std::printf("queued %d runs in flight: best %.4f ms per dispatch\n", K, bestq);
    }
    xrt::run r1(kern);
    r1.set_arg(0, 3u); r1.set_arg(1, ib); r1.set_arg(2, (unsigned)(isz / 4)); r1.set_arg(3, ba); r1.set_arg(4, bb); r1.set_arg(5, bc);
    double bestr = 1e9;
    for (int rep = 0; rep < 20; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      r1.start(); r1.wait();
      bestr = std::min(bestr, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("one reused run, start+wait: best %.4f ms\n", bestr);
  }
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(C.data(), bc.map<void*>(), C.size());
  // reference
  std::vector<double> act((size_t)M * I);
  for (int m = 0; m < M; ++m)
    for (int f2 = 0; f2 < I / 8; ++f2)
      for (int i = 0; i < 8; ++i) {
        double g = 0, u = 0;
        const float* xr = &Xd[(size_t)m * K1];
        const float* wg = &Wgu_d[(size_t)(f2 * 16 + i) * K1];
        const float* wu = &Wgu_d[(size_t)(f2 * 16 + 8 + i) * K1];
        for (int k = 0; k < K1; ++k) {
          g += (double)xr[k] * wg[k];
          u += (double)xr[k] * wu[k];
        }
        act[(size_t)m * I + f2 * 8 + i] = g / (1.0 + std::exp(-g)) * u;
      }
  std::vector<float> Xa((size_t)M * I);
  for (int m = 0; m < M; ++m)
    for (int g8 = 0; g8 < I / 8; ++g8) {
      float v[8];
      uint8_t tmp[9];
      for (int i = 0; i < 8; ++i) v[i] = _cvtsh_ss(_cvtss_sh((float)act[(size_t)m * I + g8 * 8 + i], _MM_FROUND_TO_NEAREST_INT));
      EncGroup(v, tmp);
      DecGroup(tmp, &Xa[(size_t)m * I + g8 * 8]);
    }
  std::vector<float> Y((size_t)M * N2, 0);
  for (int s = 0; s < N2 / 128; ++s)
    for (int r = 0; r < 4; ++r)
      for (int G = 0; G < 64; ++G) {
        int sb = G >> 5, acc = (G >> 3) & 3, rib = G & 7;
        int row = (acc >> 1) * 8 + rib, col = s * 128 + r * 32 + sb * 16 + (acc & 1) * 8;
        float d[8];
        DecGroup(&C[((size_t)(120 * 256) + s * 256 + r * 64 + G) * 9], d);
        for (int i = 0; i < 8; ++i) Y[(size_t)row * N2 + col + i] = d[i];
      }
  double num = 0, den = 0, num2 = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N2; ++n) {
      double a1 = 0, a2 = 0;
      for (int k = 0; k < I; ++k) {
        a1 += (double)Xa[(size_t)m * I + k] * Wdn_d[(size_t)n * I + k];
        a2 += act[(size_t)m * I + k] * Wdn_d[(size_t)n * I + k];
      }
      double y = Y[(size_t)m * N2 + n];
      num += (a1 - y) * (a1 - y);
      num2 += (a2 - y) * (a2 - y);
      den += a1 * a1;
    }
  std::printf("fused ffn: rel err vs encoded-act ref %.5f, vs exact act %.5f; best %.4f ms avg %.4f ms\n", std::sqrt(num / den), std::sqrt(num2 / den), best, tot / reps);
  return 0;
}
