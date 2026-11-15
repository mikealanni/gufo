// kdh gdn-mode probe: sbtest-kdh-gdn <xclbin> <insts> <tokens>   (one dispatch = 16 heads, tokens multiple of 64)
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

static uint16_t Bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float BfF(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static float Rb(float f) { return BfF(Bf16(f)); }
static std::vector<char> ReadAll(const char* p) { std::ifstream f(p, std::ios::binary | std::ios::ate); size_t n = f.tellg(); f.seekg(0); std::vector<char> v(n); f.read(v.data(), n); return v; }

int main(int argc, char** argv) {
  const int T = atoi(argv[3]), groups = T / 64, reps = 5, H = 16;
  const size_t in_col = (size_t)(T / 4) * 9216, out_col = (size_t)groups * 4 * 18432;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0, 1);
  std::uniform_real_distribution<float> ud(0, 1);
  std::vector<uint8_t> IN(8 * in_col, 0);
  std::vector<std::vector<float>> q(H), k(H), v(H), dec(H), bet(H);
  for (int h = 0; h < H; ++h) {
    q[h].resize((size_t)T * 128); k[h].resize((size_t)T * 128); v[h].resize((size_t)T * 128); dec[h].resize(T); bet[h].resize(T);
    const int col = h / 2, hh = h % 2;
    for (int t = 0; t < T; ++t) {
      double qs = 0, ks = 0;
      for (int i = 0; i < 128; ++i) { q[h][t * 128 + i] = nd(rng); k[h][t * 128 + i] = nd(rng); v[h][t * 128 + i] = nd(rng); qs += q[h][t * 128 + i] * q[h][t * 128 + i]; ks += k[h][t * 128 + i] * k[h][t * 128 + i]; }
      for (int i = 0; i < 128; ++i) { q[h][t * 128 + i] = Rb((float)(q[h][t * 128 + i] / std::sqrt(qs) / std::sqrt(128.0))); k[h][t * 128 + i] = Rb((float)(k[h][t * 128 + i] / std::sqrt(ks))); v[h][t * 128 + i] = Rb(v[h][t * 128 + i]); }
      dec[h][t] = 0.9f + 0.0999f * ud(rng);
      bet[h][t] = ud(rng);
      uint8_t* rec = &IN[(size_t)col * in_col + (size_t)(t / 4) * 9216 + ((t % 4) * 2 + hh) * 1024];
      uint16_t* b = (uint16_t*)rec;
      for (int i = 0; i < 128; ++i) { b[i] = Bf16(q[h][t * 128 + i]); b[128 + i] = Bf16(k[h][t * 128 + i]); b[256 + i] = Bf16(v[h][t * 128 + i]); }
      float sc[2] = {dec[h][t], bet[h][t]};
      std::memcpy(rec + 768, sc, 8);
      int32_t reset = t == 0 ? 1 : 0;
      std::memcpy(rec + 776, &reset, 4);
    }
  }
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(argv[1])};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  auto iw = ReadAll(argv[2]);
  xrt::bo ib(dev, iw.size(), XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
  std::memcpy(ib.map<void*>(), iw.data(), iw.size());
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo bx(dev, 4, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  xrt::bo bi(dev, IN.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
  xrt::bo bo(dev, 8 * out_col, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
  std::memcpy(bi.map<void*>(), IN.data(), IN.size());
  std::memset(bo.map<void*>(), 0, 8 * out_col);
  bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = kern(3, ib, (unsigned)(iw.size() / 4), bx, bi, bo);
    run.wait();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, ms);
    std::printf("rep %d: %.3f ms\n", r, ms);
  }
  bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint8_t* O = bo.map<uint8_t*>();
  double num = 0, den = 0;
  for (int h = 0; h < H; ++h) {
    std::vector<float> S(128 * 128, 0.f);
    const int col = h / 2, hh = h % 2;
    for (int t = 0; t < T; ++t) {
      float kvr[128] = {0}, sqr[128] = {0}, kq = 0, d[128], dh[128], dl[128], oo[128];
      for (int i = 0; i < 128; ++i) {
        float kk = k[h][t * 128 + i], qq = q[h][t * 128 + i];
        kq += kk * qq;
        for (int j = 0; j < 128; ++j) { kvr[j] += S[i * 128 + j] * kk; sqr[j] += S[i * 128 + j] * qq; }
      }
      const float decay = dec[h][t], beta = bet[h][t];
      const float dhi = Rb(decay), dlo = Rb(decay - dhi);
      for (int j = 0; j < 128; ++j) {
        d[j] = (v[h][t * 128 + j] - decay * kvr[j]) * beta;
        dh[j] = Rb(d[j]); dl[j] = Rb(d[j] - dh[j]);
        oo[j] = decay * sqr[j] + kq * d[j];
      }
      for (int i = 0; i < 128; ++i) for (int j = 0; j < 128; ++j) S[i * 128 + j] = Rb(dhi * S[i * 128 + j] + dlo * S[i * 128 + j] + k[h][t * 128 + i] * dh[j] + k[h][t * 128 + i] * dl[j]);
      for (int half = 0; half < 2; ++half) {
        const int row = hh * 2 + half;
        const float* dev_o = (const float*)(O + (size_t)col * out_col + (size_t)(t / 64) * 4 * 18432 + (size_t)row * 18432 + (size_t)(t % 64) * 256);
        for (int j = 0; j < 64; ++j) { double df = dev_o[j] - oo[half * 64 + j]; num += df * df; den += (double)oo[half * 64 + j] * oo[half * 64 + j]; }
      }
    }
  }
  std::printf("tokens %d rel_err %.5f best %.3f ms (%.2f us/token for 16 heads)\n", T, std::sqrt(num / den), best, best * 1000 / T);
  return 0;
}
