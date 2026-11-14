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
static size_t TileIdx(int mt, int nt) { return (size_t)((mt / 4) * 2 + nt / 8) * 32 + (nt % 8) * 4 + (mt % 4); }
static uint8_t* CBlk(uint8_t* base, int r, int c) {  // block start for row r (multiple of 8), col c (mult of 8); returns row r's 9B group
  int mt = r / 128, nt = c / 128, rr = r % 128, cc = c % 128;
  int sbr = rr / 16, br = (rr % 16) / 8, rib = rr % 8, sbc = cc / 16, bc = (cc % 16) / 8;
  return base + TileIdx(mt, nt) * 18432 + (((sbr * 8 + sbc) * 4) + br * 2 + bc) * 72 + rib * 9;
}
int main(int argc, char** argv) {
  const int M = 1024, K = 2048;
  const float eps = 1e-5f;
  const size_t RB = (size_t)M * K / 8 * 9;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> X((size_t)M * K), Xd((size_t)M * K), Br((size_t)M * K), Bd((size_t)M * K), g(K);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < K; ++c) X[(size_t)r * K + c] = nd(rng) * (c % 97 == 0 ? 8.f : 1.f) * (1.f + (r % 5));
  for (auto& v : Br) v = nd(rng) * 0.7f;
  for (auto& v : g) v = 0.5f + 1.25f * std::fabs(nd(rng));
  const int var = argc > 3 ? atoi(argv[3]) : 0;
  std::vector<uint8_t> A(3 * RB, 0), B(9216 * 16, 0), Z(RB, 0);
  uint8_t* BRp = A.data() + RB;
  uint8_t* Xp = A.data() + (var ? 0 : 2) * RB;
  for (int r = 0; r < M; ++r)
    for (int c8 = 0; c8 < K / 8; ++c8) {
      EncGroup(&X[(size_t)r * K + c8 * 8], CBlk(Xp, r, c8 * 8));
      DecGroup(CBlk(Xp, r, c8 * 8), &Xd[(size_t)r * K + c8 * 8]);
      EncGroup(&Br[(size_t)r * K + c8 * 8], CBlk(BRp, r, c8 * 8));
      DecGroup(CBlk(BRp, r, c8 * 8), &Bd[(size_t)r * K + c8 * 8]);
    }
  for (int i = 0; i < K; ++i) {
    uint16_t b = Bf16(g[i]);
    std::memcpy(&B[i * 2], &b, 2);
  }
  { uint16_t b = Bf16(eps); std::memcpy(&B[4096], &b, 2); }
  for (int q = 1; q < 16; ++q) std::memcpy(&B[q * 9216], &B[0], 9216);
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
  xrt::bo bc(dev, Z.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memcpy(ba.map<void*>(), A.data(), A.size());
  std::memcpy(bb.map<void*>(), B.data(), B.size());
  std::memset(bc.map<void*>(), 0, Z.size());
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
    std::printf("rep %d: %.3f ms\n", rep, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  ba.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(A.data(), ba.map<void*>(), A.size());
  std::memcpy(Z.data(), bc.map<void*>(), Z.size());
  uint8_t* XPp = A.data() + (var ? 2 : 0) * RB;
  double sx = 0, sxr = 0, se = 0, sr = 0, sq = 0;
  for (int r = 0; r < M; ++r) {
    std::vector<float> xp(K);
    double ss = 0;
    for (int c8 = 0; c8 < K / 8; ++c8) {
      float d[8];
      DecGroup(CBlk(XPp, r, c8 * 8), d);
      for (int i = 0; i < 8; ++i) {
        float ref = Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i];
        sx += (double)(d[i] - ref) * (d[i] - ref);
        sxr += (double)ref * ref;
        xp[c8 * 8 + i] = ref;
        ss += (double)ref * ref;
      }
    }
    double sc = 1.0 / std::sqrt(ss / K + eps);
    for (int c8 = 0; c8 < K / 8; ++c8) {
      float d[8], e8[8];
      DecGroup(&Z[Off(r, c8 * 8, K)], d);
      for (int i = 0; i < 8; ++i) e8[i] = (float)(xp[c8 * 8 + i] * sc * g[c8 * 8 + i]);
      uint8_t tmp[9];
      float q[8];
      EncGroup(e8, tmp);
      DecGroup(tmp, q);
      for (int i = 0; i < 8; ++i) {
        se += (double)(d[i] - e8[i]) * (d[i] - e8[i]);
        sq += (double)(q[i] - e8[i]) * (q[i] - e8[i]);
        sr += (double)e8[i] * e8[i];
      }
    }
  }
  {
    // per 64-col tile error of x'
    for (int kt = 0; kt < 32; kt += 1) {
      double a = 0, b = 0;
      for (int r = 0; r < M; ++r)
        for (int c8 = kt * 8; c8 < kt * 8 + 8; ++c8) {
          float d[8];
          DecGroup(CBlk(XPp, r, c8 * 8), d);
          for (int i = 0; i < 8; ++i) { float ref = Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i]; a += (d[i]-ref)*(d[i]-ref); b += ref*ref; }
        }
      std::printf("kt%d:%.2f ", kt, std::sqrt(a / b));
    }
    std::printf("\n");
    for (int rb = 0; rb < 8; ++rb) {
      double a = 0, b = 0;
      for (int r = rb * 128; r < rb * 128 + 128; ++r)
        for (int c8 = 0; c8 < 256; ++c8) {
          float d[8];
          DecGroup(CBlk(XPp, r, c8 * 8), d);
          for (int i = 0; i < 8; ++i) { float ref = Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i]; a += (d[i]-ref)*(d[i]-ref); b += ref*ref; }
        }
      std::printf("rb%d:%.2f ", rb, std::sqrt(a / b));
    }
    std::printf("\n");
  }
  {
    for (int kt = 0; kt < 32; ++kt) {
      double a = 0, b = 0;
      for (int r = 0; r < M; ++r) {
        double ss = 0;
        for (int c = 0; c < K; ++c) { float ref = Xd[(size_t)r * K + c] + Bd[(size_t)r * K + c]; ss += ref * ref; }
        double sc = 1.0 / std::sqrt(ss / K + eps);
        for (int c8 = kt * 8; c8 < kt * 8 + 8; ++c8) {
          float d[8];
          DecGroup(&Z[Off(r, c8 * 8, K)], d);
          for (int i = 0; i < 8; ++i) { float ref = (Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i]) * sc * g[c8 * 8 + i]; a += (d[i]-ref)*(d[i]-ref); b += ref*ref; }
        }
      }
      std::printf("z%d:%.3f ", kt, std::sqrt(a / b));
    }
    std::printf("\n");
  }
  {
    for (int r : {16, 17, 24, 31}) for (int c8 : {0, 1, 8, 9}) {
      float ref8[8]; for (int i = 0; i < 8; ++i) ref8[i] = Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i];
      uint8_t enc[9]; EncGroup(ref8, enc);
      long found = -1;
      for (size_t o = 0; o + 9 <= RB; o += 9) { int d = 0; for (int q = 0; q < 9; ++q) d += std::abs((int)enc[q] - (int)XPp[o + q]); if (d <= 2) { found = (long)o; break; } }
      uint8_t* exp = CBlk(XPp, r, c8 * 8);
      std::printf("r%d c8=%d expected@%ld found@%ld\n", r, c8, (long)(exp - XPp), found);
    }
  }
  {
    long bad = 0, tot = 0; long firstbad[3] = {-1,-1,-1};
    for (int r = 0; r < M; ++r) for (int c8 = 0; c8 < K / 8; ++c8) {
      float ref8[8]; for (int i = 0; i < 8; ++i) ref8[i] = Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i];
      uint8_t enc[9]; EncGroup(ref8, enc);
      uint8_t* p = CBlk(XPp, r, c8 * 8); int d = 0; for (int q = 0; q < 9; ++q) d += std::abs((int)enc[q] - (int)p[q]);
      ++tot; if (d > 2) { if (bad < 3) firstbad[bad] = (long)r * 10000 + c8; ++bad; }
    }
    {
      for (int rbk = 0; rbk < 8; ++rbk) { for (int kt = 0; kt < 32; ++kt) { int bd = 0;
        for (int r = rbk*128; r < rbk*128+128; ++r) for (int c8 = kt*8; c8 < kt*8+8; ++c8) {
          float ref8[8]; for (int i = 0; i < 8; ++i) ref8[i] = Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i];
          uint8_t enc[9]; EncGroup(ref8, enc); uint8_t* p = CBlk(XPp, r, c8 * 8); int d = 0; for (int q = 0; q < 9; ++q) d += std::abs((int)enc[q] - (int)p[q]); if (d > 2) ++bd; }
        std::printf("%c", bd == 0 ? '.' : (bd == 1024 ? '#' : 'o')); } std::printf("\n"); }
    }
    for (int r = 0; r < 40; ++r) { for (int c8 = 0; c8 < 8; ++c8) {
          float ref8[8]; for (int i = 0; i < 8; ++i) ref8[i] = Xd[(size_t)r * K + c8 * 8 + i] + Bd[(size_t)r * K + c8 * 8 + i];
          uint8_t enc[9]; EncGroup(ref8, enc); uint8_t* p = CBlk(XPp, r, c8 * 8); int d = 0; for (int q = 0; q < 9; ++q) d += std::abs((int)enc[q] - (int)p[q]); std::printf("%c", d > 2 ? 'X' : '.'); } std::printf(" "); if (r % 8 == 7) std::printf("\n"); }
    std::printf("group mismatches %ld / %ld first %ld %ld %ld\n", bad, tot, firstbad[0], firstbad[1], firstbad[2]);
  }
  std::printf("x' rel rms err %.5f ; z rel rms err %.5f (bfp floor %.5f)\n", std::sqrt(sx / sxr), std::sqrt(se / sr), std::sqrt(sq / sr));
  return 0;
}
