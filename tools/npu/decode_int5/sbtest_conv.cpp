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
static void QuantPack(const std::vector<float>& W, int N, int K, std::vector<uint8_t>& B, std::vector<float>& Wd, int extra = 0) {
  const int strips = N / 128, tk = K / 64, BQ = 1536;
  B.assign((size_t)strips * (tk + extra) * 4 * BQ, 0);
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
        uint8_t* base = &B[(((size_t)s2 * (tk + extra) + t) * 4 + r) * BQ];
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


static int NewRow(int role, int chan) {  // role 0=B,1=x,2=C
  int j = chan >> 10, col = (chan >> 7) & 7, r = (chan >> 5) & 3, c = chan & 31;
  int g = 3 * j + role;
  return (g * 8 + col) * 128 + r * 32 + c;
}
int main(int argc, char** argv) {
  const char* xclbin = argv[1];
  const char* insts = argv[2];
  const int reps = argc > 3 ? atoi(argv[3]) : 5;
  const int M = 16, K = 2048, H = 2048, N1 = 6144, T = 7, KP = 5;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> Wi((size_t)N1 * K), Wo((size_t)H * K), tap((size_t)H * T);
  for (auto& w : Wi) w = nd(rng) * 0.03f;
  for (auto& w : Wo) w = nd(rng) * 0.03f;
  for (auto& w : tap) w = nd(rng) * 0.4f;
  std::vector<float> Wperm((size_t)N1 * K);
  const int rolerow[3] = {0, 4096, 2048};  // B, x, C row bases in [B|C|x]
  for (int role = 0; role < 3; ++role)
    for (int ch = 0; ch < H; ++ch)
      std::memcpy(&Wperm[(size_t)NewRow(role, ch) * K], &Wi[(size_t)(rolerow[role] + ch) * K], K * 4);
  std::vector<uint8_t> Bi, Bo;
  std::vector<float> Wi_d, Wo_d;
  QuantPack(Wperm, N1, K, Bi, Wi_d, 1);
  QuantPack(Wo, H, K, Bo, Wo_d);
  const int tk1 = K / 64 + 1, BQ4 = 4 * 1536;
  for (int g = 0; g < 18; ++g)
    for (int col = 0; col < 8; ++col)
      for (int r = 0; r < 4; ++r) {
        int s2 = g * 8 + col, j = g / 3;
        uint8_t* base = &Bi[(((size_t)s2 * tk1 + K / 64) * 4 + r) * 1536];
        int32_t layer = 0;
        std::memcpy(base, &layer, 4);
        uint16_t* tp = (uint16_t*)(base + 64);
        for (int t = 0; t < T; ++t)
          for (int cb = 0; cb < 4; ++cb)
            for (int i = 0; i < 8; ++i) {
              int chan = ((j * 8 + col) * 4 + r) * 32 + cb * 8 + i;
              float v = bf16r(tap[(size_t)chan * T + t]);
              uint32_t u;
              std::memcpy(&u, &v, 4);
              tp[(t * 4 + cb) * 8 + i] = (uint16_t)(u >> 16);
            }
      }
  (void)BQ4;
  std::vector<uint8_t> B = Bi;
  B.insert(B.end(), Bo.begin(), Bo.end());
  const size_t a_bytes = (size_t)M * (K + 64) / 8 * 9;
  std::vector<uint8_t> A(a_bytes, 0), C((size_t)(48 + 16) * 4096);
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
  std::memcpy(bb.map<void*>(), B.data(), B.size());
  std::memset(bc.map<void*>(), 0, C.size());
  bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<float> hist((size_t)(T - 1) * H, 0.f);  // reference history
  for (int step = 0; step < 3; ++step) {
    // step 0: reset (kprev=-1); step 1: kprev=KP after step 0's 16 tokens; step 2: kprev=16
    std::vector<float> X((size_t)M * K), Xd((size_t)M * K);
    for (auto& x : X) x = nd(rng);
    for (int t = 0; t < K / 64; ++t)
      for (int gi = 0; gi < 128; ++gi) {
        int rib = gi & 7, block = (gi >> 3) & 1, sc = gi >> 4;
        int row = block * 8 + rib;
        float v[8], d[8];
        for (int i = 0; i < 8; ++i) v[i] = X[(size_t)row * K + t * 64 + sc * 8 + i];
        uint8_t* o = &A[((size_t)t * 128 + gi) * 9];
        EncGroup(v, o);
        DecGroup(o, d);
        for (int i = 0; i < 8; ++i) Xd[(size_t)row * K + t * 64 + sc * 8 + i] = d[i];
      }
    int32_t kprev = step == 0 ? -1 : (step == 1 ? KP : 16);
    std::memcpy(ba.map<void*>(), A.data(), A.size());
    ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t0 = std::chrono::steady_clock::now();
    auto run = kern(3, ib, (unsigned)(isz / 4), ba, bb, bc);
    { auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(20); for (;;) { auto st = run.state(); if (st == ERT_CMD_STATE_COMPLETED) break; if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED && st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) { std::printf("run ended in state %d\n", (int)st); return 3; } if (std::chrono::steady_clock::now() > t_end) { std::printf("timeout waiting, state %d\n", (int)st); return 4; } } }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C.data(), bc.map<void*>(), C.size());
    // reference conv in double
    std::vector<double> bx((size_t)(T - 1 + M) * H), yr((size_t)M * H);
    std::vector<double> Cc((size_t)M * H);
    for (int t = 0; t < T - 1; ++t)
      for (int h = 0; h < H; ++h) bx[(size_t)t * H + h] = hist[(size_t)t * H + h];
    for (int m = 0; m < M; ++m)
      for (int ch = 0; ch < H; ++ch) {
        double b = 0, x = 0, c = 0;
        const float* xr = &Xd[(size_t)m * K];
        const float* wb = &Wi_d[(size_t)NewRow(0, ch) * K];
        const float* wx = &Wi_d[(size_t)NewRow(1, ch) * K];
        const float* wc = &Wi_d[(size_t)NewRow(2, ch) * K];
        for (int k = 0; k < K; ++k) {
          b += (double)xr[k] * wb[k];
          x += (double)xr[k] * wx[k];
          c += (double)xr[k] * wc[k];
        }
        bx[(size_t)(m + T - 1) * H + ch] = b * x;
        Cc[(size_t)m * H + ch] = c;
      }
    for (int m = 0; m < M; ++m)
      for (int ch = 0; ch < H; ++ch) {
        double s = 0;
        for (int j = 0; j < T; ++j) s += (double)tap[(size_t)ch * T + j] * bx[(size_t)(m + j) * H + ch];
        yr[(size_t)m * H + ch] = s * Cc[(size_t)m * H + ch];
      }
    // next reference history
    int acc = step == 0 ? KP : 16;
    for (int t = 0; t < T - 1; ++t)
      for (int h = 0; h < H; ++h) hist[(size_t)t * H + h] = (float)bx[(size_t)(acc + t) * H + h];
    for (int g : {1, 4})
      for (int col = 0; col < 8; ++col)
        for (int r = 0; r < 4; ++r) {
          int s2 = g * 8 + col, j = g / 3;
          const uint16_t* src = (const uint16_t*)&C[((size_t)(16 + j * 8 + col) * 4096 + r * 1024)];
          uint16_t* dst = (uint16_t*)((uint8_t*)bb.map<void*>() + (((size_t)s2 * (K / 64 + 1) + K / 64) * 4 + r) * 1536 + 512);
          for (int cb = 0; cb < 4; ++cb) {
            uint16_t nh[6][8];
            for (int i = 0; i < 6; ++i) {
              int idx = acc + i;
              const uint16_t* row = idx < 6 ? dst + (cb * 6 + idx) * 8 : src + cb * 128 + (idx - 6) * 8;
              std::memcpy(nh[i], row, 16);
            }
            std::memcpy(dst + cb * 48, nh, 96);
          }
        }
    bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // reference out_proj on encoded y
    std::vector<float> Ye((size_t)M * H);
    for (int m = 0; m < M; ++m)
      for (int g8 = 0; g8 < H / 8; ++g8) {
        float v[8];
        uint8_t tmp[9];
        for (int i = 0; i < 8; ++i) v[i] = _cvtsh_ss(_cvtss_sh((float)yr[(size_t)m * H + g8 * 8 + i], _MM_FROUND_TO_NEAREST_INT));
        EncGroup(v, tmp);
        DecGroup(tmp, &Ye[(size_t)m * H + g8 * 8]);
      }
    std::vector<float> Y((size_t)M * H, 0);
    for (int s = 0; s < H / 128; ++s)
      for (int r = 0; r < 4; ++r)
        for (int G = 0; G < 64; ++G) {
          int sb = G >> 5, a2 = (G >> 3) & 3, rib = G & 7;
          int row = (a2 >> 1) * 8 + rib, col = s * 128 + r * 32 + sb * 16 + (a2 & 1) * 8;
          float d[8];
          DecGroup(&C[((size_t)(48 + s) * 4096 + r * 1024 + G * 9)], d);
          for (int i = 0; i < 8; ++i) Y[(size_t)row * H + col + i] = d[i];
        }
    double num = 0, den = 0, dn2 = 0, nn2 = 0;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < H; ++n) {
        double a1 = 0;
        for (int k = 0; k < H; ++k) a1 += (double)Ye[(size_t)m * H + k] * Wo_d[(size_t)n * H + k];
        double y = Y[(size_t)m * H + n];
        num += (a1 - y) * (a1 - y);
        den += a1 * a1;
      }
    (void)dn2; (void)nn2;
    std::printf("step %d kprev %d: rel err %.5f  (%.4f ms)\n", step, kprev, std::sqrt(num / den), ms);
  }
  double best = 1e9;
  for (int r = 0; r < 30; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = kern(3, ib, (unsigned)(isz / 4), ba, bb, bc);
    while (run.state() != ERT_CMD_STATE_COMPLETED) {}
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("best %.4f ms\n", best);
  return 0;
}
