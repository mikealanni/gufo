// HC down projection at decode: 320 rows x K = 10240, one token. The current
// kernel reads Q8_0 weights (3.5 MB); the candidate reads a Q4_1 copy
// (2.05 MB) with the same one-wave-per-row structure and dp4a against the
// Q8_1 input. Reports max error against a double reference and GB/s.
//
//   tools/bench/build.sh hc_down_q4_1_bench && /tmp/hc_down_q4_1_bench
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x)                                                        \
  do {                                                                  \
    hipError_t e_ = (x);                                                \
    if (e_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,            \
                   hipGetErrorString(e_));                              \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

constexpr int kRows = 320;
constexpr int kK = 10240;
constexpr int kBlocks = kK / 32;  // 320
constexpr int kCopies = 24;       // distinct weight copies, defeats the cache

struct Q8_0Block { half d; int8_t qs[32]; };          // 34 bytes
struct Q4_1Block { half d; half m; uint8_t qs[16]; };  // 20 bytes
struct Q8_1Block { half d; half s; int8_t qs[32]; };   // 36 bytes

__device__ __forceinline__ int Int2(const int8_t* p, int i) {  // 2-byte aligned
  const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
  return h[2 * i] | (static_cast<int>(h[2 * i + 1]) << 16);
}
__device__ __forceinline__ int Int4(const void* p, int i) {
  return reinterpret_cast<const int*>(p)[i];
}
__device__ __forceinline__ int Dp4a(int a, int b, int c) {
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ int Dp4aU(int a, int b, int c) {  // a unsigned bytes
  return __builtin_amdgcn_sudot4(false, a, true, b, c, false);
}
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
  return v;
}

// The shipped structure: 32 lanes per row, four lanes per 32-weight block, two
// iterations in flight.
__launch_bounds__(32) __global__ void DownQ8(const Q8_0Block* __restrict__ w,
                                             const Q8_1Block* __restrict__ x,
                                             float* __restrict__ out) {
  constexpr int prefetch = 2;
  const int lane = threadIdx.x;
  const int part = (lane & 3) * 2;
  float sum = 0.f;
  for (int first = lane / 4; first < kBlocks; first += 8 * prefetch) {
    int v[prefetch][2], u[prefetch][2];
    half dw[prefetch], dx[prefetch];
#pragma unroll
    for (int j = 0; j < prefetch; ++j) {
      const int kb = first + 8 * j;
      const Q8_0Block& b = w[blockIdx.x * kBlocks + kb];
      dw[j] = b.d;
#pragma unroll
      for (int i = 0; i < 2; ++i) v[j][i] = Int2(b.qs, part + i);
      const Q8_1Block& a = x[kb];
      dx[j] = a.d;
#pragma unroll
      for (int i = 0; i < 2; ++i) u[j][i] = Int4(a.qs, part + i);
    }
#pragma unroll
    for (int j = 0; j < prefetch; ++j) {
      int dot = Dp4a(v[j][0], u[j][0], 0);
      dot = Dp4a(v[j][1], u[j][1], dot);
      sum += __half2float(dw[j]) * __half2float(dx[j]) * static_cast<float>(dot);
    }
  }
  sum = WaveSum(sum);
  if (lane == 0) out[blockIdx.x] = sum;
}

// Q4_1: a block's 16 code bytes are four ints; lane `part` takes one int (four
// low nibbles = weights 4p.., four high nibbles = weights 16 + 4p..). The m * s
// term is added once per block, by the lane with part 0.
template <int kPrefetch>
__launch_bounds__(32) __global__ void DownQ4(const Q4_1Block* __restrict__ w,
                                             const Q8_1Block* __restrict__ x,
                                             float* __restrict__ out) {
  const int lane = threadIdx.x;
  const int part = lane & 3;
  float sum = 0.f;
  for (int first = lane / 4; first < kBlocks; first += 8 * kPrefetch) {
    int v[kPrefetch], ulo[kPrefetch], uhi[kPrefetch];
    float dw[kPrefetch], mw[kPrefetch], dx[kPrefetch], sx[kPrefetch];
#pragma unroll
    for (int j = 0; j < kPrefetch; ++j) {
      const int kb = first + 8 * j;
      const Q4_1Block& b = w[blockIdx.x * kBlocks + kb];
      dw[j] = __half2float(b.d);
      mw[j] = __half2float(b.m);
      v[j] = Int4(b.qs, part);
      const Q8_1Block& a = x[kb];
      dx[j] = __half2float(a.d);
      sx[j] = __half2float(a.s);
      ulo[j] = Int4(a.qs, part);
      uhi[j] = Int4(a.qs, part + 4);
    }
#pragma unroll
    for (int j = 0; j < kPrefetch; ++j) {
      int dot = Dp4aU(v[j] & 0x0F0F0F0F, ulo[j], 0);
      dot = Dp4aU((v[j] >> 4) & 0x0F0F0F0F, uhi[j], dot);
      sum += dw[j] * dx[j] * static_cast<float>(dot);
      if (part == 0) sum += mw[j] * sx[j];
    }
  }
  sum = WaveSum(sum);
  if (lane == 0) out[blockIdx.x] = sum;
}

int main() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  // Q8_0 weights and the Q4_1 copy derived from them exactly as the loader does.
  std::vector<Q8_0Block> w8(static_cast<size_t>(kRows) * kBlocks);
  std::vector<Q4_1Block> w4(w8.size());
  for (size_t b = 0; b < w8.size(); ++b) {
    const float scale = 0.002f + 0.004f * (rng() & 255) / 255.0f;
    w8[b].d = __float2half(scale);
    float x[32], lo = 1e30f, hi = -1e30f;
    for (int j = 0; j < 32; ++j) {
      w8[b].qs[j] = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
      x[j] = __half2float(w8[b].d) * w8[b].qs[j];
      lo = std::fmin(lo, x[j]);
      hi = std::fmax(hi, x[j]);
    }
    const float d = (hi - lo) / 15.0f, inv = d != 0 ? 1.0f / d : 0.0f;
    w4[b].d = __float2half(d);
    w4[b].m = __float2half(lo);
    for (int j = 0; j < 16; ++j) {
      const uint32_t q0 = static_cast<uint32_t>((x[j] - lo) * inv + 0.5f);
      const uint32_t q1 = static_cast<uint32_t>((x[j + 16] - lo) * inv + 0.5f);
      w4[b].qs[j] = static_cast<uint8_t>((q0 & 15) | ((q1 & 15) << 4));
    }
  }
  std::vector<Q8_1Block> xq(kBlocks);
  std::vector<float> xf(kK);
  for (int b = 0; b < kBlocks; ++b) {
    float amax = 0;
    for (int e = 0; e < 32; ++e) {
      xf[b * 32 + e] = ud(rng);
      amax = std::fmax(amax, std::fabs(xf[b * 32 + e]));
    }
    const float dq = amax / 127.0f;
    int qsum = 0;
    for (int e = 0; e < 32; ++e) {
      const int qi = dq != 0 ? static_cast<int>(std::lround(xf[b * 32 + e] / dq)) : 0;
      xq[b].qs[e] = static_cast<int8_t>(qi);
      qsum += qi;
    }
    xq[b].d = __float2half(dq);
    xq[b].s = __float2half(dq * qsum);
  }
  // Double reference: Q8_0 weights times the Q8_1 input (what the shipped path
  // computes) and the Q4_1 weights times the same input.
  std::vector<double> ref8(kRows), ref4(kRows);
  for (int r = 0; r < kRows; ++r) {
    double a8 = 0, a4 = 0;
    for (int b = 0; b < kBlocks; ++b) {
      const size_t i = static_cast<size_t>(r) * kBlocks + b;
      const double dx = __half2float(xq[b].d);
      double s8 = 0, s4 = 0;
      for (int j = 0; j < 32; ++j) {
        const int q4 = j < 16 ? (w4[i].qs[j] & 15) : (w4[i].qs[j - 16] >> 4);
        s8 += static_cast<double>(w8[i].qs[j]) * xq[b].qs[j];
        s4 += (__half2float(w4[i].d) * q4 + __half2float(w4[i].m)) * static_cast<double>(xq[b].qs[j]);
      }
      a8 += __half2float(w8[i].d) * dx * s8;
      a4 += dx * s4;
    }
    ref8[r] = a8;
    ref4[r] = a4;
  }

  Q8_0Block* d8;
  Q4_1Block* d4;
  Q8_1Block* dx;
  float* dout;
  CHECK(hipMalloc(&d8, w8.size() * sizeof(Q8_0Block) * kCopies));
  CHECK(hipMalloc(&d4, w4.size() * sizeof(Q4_1Block) * kCopies));
  CHECK(hipMalloc(&dx, kBlocks * sizeof(Q8_1Block)));
  CHECK(hipMalloc(&dout, kRows * sizeof(float)));
  for (int c = 0; c < kCopies; ++c) {
    CHECK(hipMemcpy(d8 + c * w8.size(), w8.data(), w8.size() * sizeof(Q8_0Block), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d4 + c * w4.size(), w4.data(), w4.size() * sizeof(Q4_1Block), hipMemcpyHostToDevice));
  }
  CHECK(hipMemcpy(dx, xq.data(), kBlocks * sizeof(Q8_1Block), hipMemcpyHostToDevice));
  std::vector<float> got(kRows);
  auto run = [&](int mode, int copy) {
    if (mode == 0) DownQ8<<<kRows, 32>>>(d8 + copy * w8.size(), dx, dout);
    if (mode == 1) DownQ4<1><<<kRows, 32>>>(d4 + copy * w4.size(), dx, dout);
    if (mode == 2) DownQ4<2><<<kRows, 32>>>(d4 + copy * w4.size(), dx, dout);
    if (mode == 3) DownQ4<4><<<kRows, 32>>>(d4 + copy * w4.size(), dx, dout);
    if (mode == 4) DownQ4<5><<<kRows, 32>>>(d4 + copy * w4.size(), dx, dout);
    if (mode == 5) DownQ4<8><<<kRows, 32>>>(d4 + copy * w4.size(), dx, dout);
  };
  const char* names[6] = {"Q8_0 (shipped structure)", "Q4_1, 1 block in flight",
                          "Q4_1, 2 in flight", "Q4_1, 4 in flight", "Q4_1, 5 in flight", "Q4_1, 8 in flight"};
  const double bytes[6] = {w8.size() * 34.0, w4.size() * 20.0, w4.size() * 20.0, w4.size() * 20.0, w4.size() * 20.0, w4.size() * 20.0};
  hipEvent_t t0, t1;
  CHECK(hipEventCreate(&t0));
  CHECK(hipEventCreate(&t1));
  for (int mode = 0; mode < 6; ++mode) {
    run(mode, 0);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(got.data(), dout, kRows * sizeof(float), hipMemcpyDeviceToHost));
    double err = 0, norm = 0;
    const auto& ref = mode == 0 ? ref8 : ref4;
    for (int r = 0; r < kRows; ++r) {
      err = std::fmax(err, std::fabs(got[r] - ref[r]));
      norm = std::fmax(norm, std::fabs(ref[r]));
    }
    double best = 1e9;
    for (int round = 0; round < 8; ++round) {
      for (int c = 0; c < kCopies; ++c) run(mode, c);
      CHECK(hipDeviceSynchronize());
      CHECK(hipEventRecord(t0));
      for (int it = 0; it < 20; ++it)
        for (int c = 0; c < kCopies; ++c) run(mode, c);
      CHECK(hipEventRecord(t1));
      CHECK(hipEventSynchronize(t1));
      float ms = 0;
      CHECK(hipEventElapsedTime(&ms, t0, t1));
      best = std::fmin(best, ms * 1000.0 / (20.0 * kCopies));
    }
    std::printf("%-28s %7.2f us  %6.1f GB/s  max err %.2e (relative %.2e)\n", names[mode],
                best, bytes[mode] / (best * 1e3), err, err / norm);
  }
  return 0;
}
