// Dense Q4_1 decode GEMV (the decode copy of ssm_in / ssm_out / attn_qkv /
// attn_out) at batch 1 on gfx1151: current dp4a + Q8_1 activations versus
// T-MAC style dequant -> F32 FMA with F32 activations, against a wide-load
// streaming floor over the same bytes.
//
//   tools/bench/build.sh q4_1_dense_decode_bench && /tmp/q4_1_dense_decode_bench
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

constexpr int kBlockBytes = 20;  // half d, half m, 16 nibble bytes
constexpr int kWavesPerBlock = 8;
constexpr int kRowsPerWave = 4;
constexpr int kCopies = 10;  // distinct weight copies cycled to defeat the cache

struct Shape {
  const char* name;
  int rows;
  int k;
};
// Per layer: 36 linear-attention layers read ssm_in + ssm_out, 12 attention
// layers read attn_qkv + attn_out.
constexpr Shape kShapes[4] = {{"ssm_in  16384 x 2560", 16384, 2560},
                              {"ssm_out  2560 x 6144", 2560, 6144},
                              {"attn_qkv 13312 x 2560", 13312, 2560},
                              {"attn_out 2560 x 6144", 2560, 6144}};

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
  return v;
}

__device__ __forceinline__ int Dp4aU8S8(uint32_t unsigned_q, uint32_t signed_a,
                                        int c) {
  // nibbles are 0..15 (unsigned), activations signed
  return __builtin_amdgcn_sudot4(false, static_cast<int>(unsigned_q), true,
                                 static_cast<int>(signed_a), c, false);
}

struct Q8Block {
  half d;
  half s;
  int8_t qs[32];
};

template <int kMode>
__launch_bounds__(32 * kWavesPerBlock) __global__
    void Gemv(const uint8_t* __restrict__ weights, const Q8Block* __restrict__ act_q,
              const float* __restrict__ act_f, const float* __restrict__ act_sum,
              float* __restrict__ out, int rows, int k) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * kWavesPerBlock + wave) * kRowsPerWave;
  if (row0 >= rows) return;
  const int blocks = k / 32;
  const int row_bytes = blocks * kBlockBytes;
  const uint8_t* base = weights + static_cast<size_t>(row0) * row_bytes;
  const int units = kRowsPerWave * blocks;
  float acc[kRowsPerWave] = {0.f, 0.f, 0.f, 0.f};
  for (int u = lane; u < units; u += 32) {
    const int rr = u / blocks;
    const int b = u - rr * blocks;
    const uint8_t* blk = base + static_cast<size_t>(rr) * row_bytes + b * kBlockBytes;
    const uint32_t head = *reinterpret_cast<const uint32_t*>(blk);
    const uint4 q = *reinterpret_cast<const uint4*>(blk + 4);
    const float d = __half2float(__ushort_as_half(head & 0xFFFFu));
    const float m = __half2float(__ushort_as_half(head >> 16));
    float c;
    if constexpr (kMode == 1) {
      const Q8Block& a = act_q[b];
      const uint32_t* ai = reinterpret_cast<const uint32_t*>(a.qs);
      const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
      int sumi = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sumi = Dp4aU8S8(qw[i] & 0x0F0F0F0FU, ai[i], sumi);
        sumi = Dp4aU8S8((qw[i] >> 4) & 0x0F0F0F0FU, ai[i + 4], sumi);
      }
      c = d * __half2float(a.d) * static_cast<float>(sumi) + m * __half2float(a.s);
    } else {
      const float4* xa = reinterpret_cast<const float4*>(act_f + b * 32);
      const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 xl = xa[i];      // elements 4i..4i+3  (low nibbles)
        const float4 xh = xa[i + 4];  // elements 16+4i..   (high nibbles)
        const uint32_t lo = qw[i] & 0x0F0F0F0FU;
        const uint32_t hi = (qw[i] >> 4) & 0x0F0F0F0FU;
        sum = fmaf(static_cast<float>(lo & 0xFFu), xl.x, sum);
        sum = fmaf(static_cast<float>((lo >> 8) & 0xFFu), xl.y, sum);
        sum = fmaf(static_cast<float>((lo >> 16) & 0xFFu), xl.z, sum);
        sum = fmaf(static_cast<float>(lo >> 24), xl.w, sum);
        sum = fmaf(static_cast<float>(hi & 0xFFu), xh.x, sum);
        sum = fmaf(static_cast<float>((hi >> 8) & 0xFFu), xh.y, sum);
        sum = fmaf(static_cast<float>((hi >> 16) & 0xFFu), xh.z, sum);
        sum = fmaf(static_cast<float>(hi >> 24), xh.w, sum);
      }
      c = d * sum + m * act_sum[b];
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWave; ++r) acc[r] += (rr == r) ? c : 0.f;
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWave; ++r) {
    const float s = WaveSum(acc[r]);
    if (lane == 0) out[row0 + r] = s;
  }
}

__launch_bounds__(256) __global__ void Stream(const uint4* __restrict__ p,
                                              size_t n, float* __restrict__ out) {
  uint32_t x = 0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * 256) {
    const uint4 v = p[i];
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x12345678u) out[0] = 1.f;
}

int main() {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  double total_bytes = 0, total_ms[3] = {0, 0, 0};
  const char* mode_names[3] = {"wide-load streaming floor", "dp4a + Q8_1 (current)",
                               "F32 dequant + FMA (T-MAC)"};
  for (const Shape& sh : kShapes) {
    const int blocks = sh.k / 32;
    const size_t mat_bytes = static_cast<size_t>(sh.rows) * blocks * kBlockBytes;
    std::vector<uint8_t> w(mat_bytes);
    for (size_t b = 0; b < mat_bytes / kBlockBytes; ++b) {
      uint8_t* p = w.data() + b * kBlockBytes;
      const half d = __float2half(0.002f + 0.004f * (rng() & 255) / 255.0f);
      const half m = __float2half(-0.03f + 0.02f * (rng() & 255) / 255.0f);
      std::memcpy(p, &d, 2);
      std::memcpy(p + 2, &m, 2);
      for (int j = 0; j < 16; ++j) p[4 + j] = static_cast<uint8_t>(rng());
    }
    std::vector<float> xf(sh.k), xs(blocks);
    std::vector<Q8Block> xq(blocks);
    for (int b = 0; b < blocks; ++b) {
      float amax = 0, sum = 0;
      for (int e = 0; e < 32; ++e) {
        xf[b * 32 + e] = __half2float(__float2half(ud(rng)));
        amax = std::fmax(amax, std::fabs(xf[b * 32 + e]));
        sum += xf[b * 32 + e];
      }
      xs[b] = sum;
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
    uint8_t* d_w;
    Q8Block* d_xq;
    float *d_xf, *d_xs, *d_out;
    CHECK(hipMalloc(&d_w, mat_bytes * kCopies));
    for (int c = 0; c < kCopies; ++c)
      CHECK(hipMemcpy(d_w + c * mat_bytes, w.data(), mat_bytes, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_xq, xq.size() * sizeof(Q8Block)));
    CHECK(hipMalloc(&d_xf, xf.size() * sizeof(float)));
    CHECK(hipMalloc(&d_xs, xs.size() * sizeof(float)));
    CHECK(hipMalloc(&d_out, sh.rows * sizeof(float)));
    CHECK(hipMemcpy(d_xq, xq.data(), xq.size() * sizeof(Q8Block), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_xf, xf.data(), xf.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_xs, xs.data(), xs.size() * sizeof(float), hipMemcpyHostToDevice));

    const dim3 grid((sh.rows + kWavesPerBlock * kRowsPerWave - 1) / (kWavesPerBlock * kRowsPerWave));
    const dim3 block(32 * kWavesPerBlock);
    const auto launch = [&](int mode, int copy) {
      const uint8_t* wp = d_w + static_cast<size_t>(copy) * mat_bytes;
      if (mode == 0)
        Stream<<<dim3(grid.x * 2), 256>>>(reinterpret_cast<const uint4*>(wp), mat_bytes / 16, d_out);
      if (mode == 1) Gemv<1><<<grid, block>>>(wp, d_xq, d_xf, d_xs, d_out, sh.rows, sh.k);
      if (mode == 2) Gemv<2><<<grid, block>>>(wp, d_xq, d_xf, d_xs, d_out, sh.rows, sh.k);
    };
    // accuracy against double
    std::vector<double> ref(sh.rows);
    for (int r = 0; r < sh.rows; r += std::max(1, sh.rows / 64)) {
      const uint8_t* row = w.data() + static_cast<size_t>(r) * blocks * kBlockBytes;
      double acc = 0;
      for (int b = 0; b < blocks; ++b) {
        const uint8_t* blk = row + b * kBlockBytes;
        half dh, mh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&mh, blk + 2, 2);
        double sq = 0, sx = 0;
        for (int j = 0; j < 16; ++j) {
          sq += (blk[4 + j] & 15) * static_cast<double>(xf[b * 32 + j]);
          sq += (blk[4 + j] >> 4) * static_cast<double>(xf[b * 32 + 16 + j]);
        }
        for (int e = 0; e < 32; ++e) sx += xf[b * 32 + e];
        acc += __half2float(dh) * sq + __half2float(mh) * sx;
      }
      ref[r] = acc;
    }
    std::vector<float> got(sh.rows);
    double err1 = 0, err2 = 0, norm = 0;
    for (int mode = 1; mode <= 2; ++mode) {
      CHECK(hipMemset(d_out, 0, sh.rows * sizeof(float)));
      launch(mode, 0);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(got.data(), d_out, sh.rows * sizeof(float), hipMemcpyDeviceToHost));
      for (int r = 0; r < sh.rows; r += std::max(1, sh.rows / 64)) {
        const double e = std::fabs(got[r] - ref[r]);
        (mode == 1 ? err1 : err2) = std::fmax(mode == 1 ? err1 : err2, e);
        norm = std::fmax(norm, std::fabs(ref[r]));
      }
    }
    double best[3] = {1e9, 1e9, 1e9};
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    for (int round = 0; round < 5; ++round) {
      for (int mode = 0; mode < 3; ++mode) {
        for (int c = 0; c < kCopies; ++c) launch(mode, c);
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(t0));
        for (int it = 0; it < 10; ++it)
          for (int c = 0; c < kCopies; ++c) launch(mode, c);
        CHECK(hipEventRecord(t1));
        CHECK(hipEventSynchronize(t1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, t0, t1));
        best[mode] = std::fmin(best[mode], ms / (10.0 * kCopies));
      }
    }
    std::printf("%-24s %6.1f MB  err dp4a %.2e  err fma %.2e (rel to %.2f)\n", sh.name,
                mat_bytes / 1e6, err1 / norm, err2 / norm, norm);
    for (int mode = 0; mode < 3; ++mode)
      std::printf("    %-28s %7.3f ms  %6.1f GB/s\n", mode_names[mode], best[mode],
                  mat_bytes / (best[mode] * 1e6));
    total_bytes += mat_bytes;
    for (int mode = 0; mode < 3; ++mode) total_ms[mode] += best[mode];
    hipFree(d_w);
    hipFree(d_xq);
    hipFree(d_xf);
    hipFree(d_xs);
    hipFree(d_out);
  }
  std::printf("\nall four matrices once: %.1f MB\n", total_bytes / 1e6);
  for (int mode = 0; mode < 3; ++mode)
    std::printf("  %-28s %7.3f ms  %6.1f GB/s\n", mode_names[mode], total_ms[mode],
                total_bytes / (total_ms[mode] * 1e6));
  return 0;
}
