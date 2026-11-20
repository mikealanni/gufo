// int4 x int4 WMMA prefill GEMM experiment for gfx1151.
//
// Weights: groups of 256 K values, 136 bytes each: two (fp16 scale, fp16 zero)
// pairs for the two 128-wide halves, then 128 nibble bytes (byte j holds K
// 2j in the low nibble and 2j+1 in the high nibble), stored already rotated
// along K by R = D2 H D1 / 16 (signed Hadamard over 256).
// Activations: x / awq_scale, rotated the same way, then quantized per 128
// block to signed int4 with a float scale and the integer sum of the codes.
// Output: out[n][m] = sum_k W[m][k] x[n][k] reconstructed from
//   sx * (sc * dot(q_w, q_x) + zp * sum(q_x)) per 128 block.
//
// Usage: iu4_gemm_bench <model.mq4> <weight tensor name> <awq tensor name> N
#include <fcntl.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define HIP_OK(x)                                            \
  do {                                                       \
    hipError_t e_ = (x);                                     \
    if (e_ != hipSuccess) {                                  \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(e_));                   \
      std::exit(1);                                          \
    }                                                        \
  } while (0)

typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

constexpr int kGroup = 128;
constexpr int kBlockBytes = 136;

// ---------------------------------------------------------------- producer
// One workgroup of 256 threads per (token, 256-wide K group).
__device__ __forceinline__ float WaveSum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o);
  return v;
}
__device__ __forceinline__ float WaveMax(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

constexpr int kCandidates = 8;
__constant__ float kClipDiv[kCandidates] = {7.0f, 6.5f, 6.0f, 5.5f,
                                            5.0f, 4.5f, 4.0f, 3.5f};

__global__ void __launch_bounds__(256)
    ProduceInt4(const float* __restrict__ x, const float* __restrict__ awq,
                const float* __restrict__ signs1,
                const float* __restrict__ signs2, uint8_t* __restrict__ qs,
                float* __restrict__ ds, int* __restrict__ ss,
                float* __restrict__ rotated_f32, int K, int use_search) {
  __shared__ float v[256];
  __shared__ float red[8][kCandidates + 1];
  const int groups = K / 256;
  const int g256 = blockIdx.x % groups;
  const int n = blockIdx.x / groups;
  const int t = threadIdx.x;
  const int kk = g256 * 256 + t;
  v[t] = x[(size_t)n * K + kk] / awq[kk] * signs1[t];
  __syncthreads();
  for (int stride = 1; stride < 256; stride <<= 1) {
    if (t < 128) {
      const int i = (t / stride) * stride * 2 + (t % stride);
      const float a = v[i], b = v[i + stride];
      v[i] = a + b;
      v[i + stride] = a - b;
    }
    __syncthreads();
  }
  float val = v[t] * 0.0625f * signs2[t];
  if (rotated_f32)
    rotated_f32[(size_t)n * K + kk] = val;
  const int wave = t >> 5, lane = t & 31;
  float amax = WaveMax(fabsf(val));
  if (lane == 0)
    red[wave][0] = amax;
  __syncthreads();
  // half h = t / 128 owns waves 4h..4h+3
  const int h = t >> 7;
  float half_amax = fmaxf(fmaxf(red[4 * h][0], red[4 * h + 1][0]),
                          fmaxf(red[4 * h + 2][0], red[4 * h + 3][0]));
  float best_d = half_amax / 7.0f;
  if (use_search && half_amax > 0.0f) {
    float err[kCandidates];
    for (int c = 0; c < kCandidates; ++c) {
      const float d = half_amax / kClipDiv[c];
      const float q = fminf(fmaxf(rintf(val / d), -8.0f), 7.0f);
      const float e = val - q * d;
      err[c] = WaveSum(e * e);
    }
    __syncthreads();
    if (lane == 0)
      for (int c = 0; c < kCandidates; ++c)
        red[wave][c] = err[c];
    __syncthreads();
    float best = 1e30f;
    for (int c = 0; c < kCandidates; ++c) {
      const float e = red[4 * h][c] + red[4 * h + 1][c] + red[4 * h + 2][c] +
                      red[4 * h + 3][c];
      if (e < best) {
        best = e;
        best_d = half_amax / kClipDiv[c];
      }
    }
  }
  const float inv = best_d > 0.0f ? 1.0f / best_d : 0.0f;
  const int q = (int)fminf(fmaxf(rintf(val * inv), -8.0f), 7.0f);
  // pack: even K low nibble, odd K high nibble
  const int qn = __shfl_xor(q, 1);
  if ((t & 1) == 0) {
    const uint8_t byte = (uint8_t)((q & 0xF) | ((qn & 0xF) << 4));
    qs[(size_t)n * (K / 2) + (size_t)g256 * 128 + (t >> 1)] = byte;
  }
  // integer sum of the half's codes
  float sq = WaveSum((float)q);
  __syncthreads();
  if (lane == 0)
    red[wave][0] = sq;
  __syncthreads();
  if ((t & 127) == 0) {
    const int sum = (int)(red[4 * h][0] + red[4 * h + 1][0] +
                          red[4 * h + 2][0] + red[4 * h + 3][0]);
    const size_t gi = (size_t)n * (K / kGroup) + g256 * 2 + h;
    ds[gi] = best_d;
    ss[gi] = sum;
  }
}

// ----------------------------------------------------- producer, wave per
// group One wave handles one 256-wide group: lane l owns elements 8l..8l+7.
// Hadamard strides 1,2,4 are in registers, strides 8..128 are lane-xor
// shuffles.
__device__ __forceinline__ float Reduce16Max(float v) {
  for (int o = 8; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o));
  return v;
}
__device__ __forceinline__ float Reduce16Sum(float v) {
  for (int o = 8; o > 0; o >>= 1)
    v += __shfl_xor(v, o);
  return v;
}

__global__ void __launch_bounds__(256)
    ProduceInt4Wave(const float* __restrict__ x, const float* __restrict__ awq,
                    const float* __restrict__ signs1,
                    const float* __restrict__ signs2, uint8_t* __restrict__ qs,
                    float* __restrict__ ds, int* __restrict__ ss, int K,
                    int total_groups, int use_search) {
  const int lane = threadIdx.x & 31;
  const int gid = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (gid >= total_groups)
    return;
  const int groups = K / 256;
  const int g256 = gid % groups;
  const int n = gid / groups;
  const size_t base = (size_t)n * K + (size_t)g256 * 256 + lane * 8;
  const float4 x0 = *reinterpret_cast<const float4*>(x + base);
  const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
  const float4 a0 =
      *reinterpret_cast<const float4*>(awq + g256 * 256 + lane * 8);
  const float4 a1 =
      *reinterpret_cast<const float4*>(awq + g256 * 256 + lane * 8 + 4);
  const float4 p0 = *reinterpret_cast<const float4*>(signs1 + lane * 8);
  const float4 p1 = *reinterpret_cast<const float4*>(signs1 + lane * 8 + 4);
  float v[8] = {x0.x / a0.x * p0.x, x0.y / a0.y * p0.y, x0.z / a0.z * p0.z,
                x0.w / a0.w * p0.w, x1.x / a1.x * p1.x, x1.y / a1.y * p1.y,
                x1.z / a1.z * p1.z, x1.w / a1.w * p1.w};
#pragma unroll
  for (int st = 1; st < 8; st <<= 1)
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if ((i & st) == 0) {
        const float a = v[i], b = v[i + st];
        v[i] = a + b;
        v[i + st] = a - b;
      }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const bool hi = (lane & m) != 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float o = __shfl_xor(v[i], m);
      v[i] = hi ? (o - v[i]) : (v[i] + o);
    }
  }
  const float4 q0 = *reinterpret_cast<const float4*>(signs2 + lane * 8);
  const float4 q1 = *reinterpret_cast<const float4*>(signs2 + lane * 8 + 4);
  const float sg[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i)
    v[i] *= 0.0625f * sg[i];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  amax = Reduce16Max(amax);  // lanes 0-15 = first half, 16-31 = second half
  float best_d = amax / 7.0f;
  if (use_search && amax > 0.0f) {
    float best = 1e30f;
#pragma unroll
    for (int c = 0; c < kCandidates; ++c) {
      const float d = amax / kClipDiv[c], inv = 1.0f / d;
      float e2 = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float q = fminf(fmaxf(rintf(v[i] * inv), -8.0f), 7.0f);
        const float e = v[i] - q * d;
        e2 += e * e;
      }
      e2 = Reduce16Sum(e2);
      if (e2 < best) {
        best = e2;
        best_d = d;
      }
    }
  }
  const float inv = best_d > 0.0f ? 1.0f / best_d : 0.0f;
  int q[8];
  int qsum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = (int)fminf(fmaxf(rintf(v[i] * inv), -8.0f), 7.0f);
    qsum += q[i];
  }
  uint32_t word = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    word |= (uint32_t)(q[i] & 0xF) << (4 * i);
  *reinterpret_cast<uint32_t*>(qs + (size_t)n * (K / 2) + (size_t)g256 * 128 +
                               lane * 4) = word;
  float sum = Reduce16Sum((float)qsum);
  if ((lane & 15) == 0) {
    const size_t gi = (size_t)n * (K / kGroup) + g256 * 2 + (lane >> 4);
    ds[gi] = best_d;
    ss[gi] = (int)sum;
  }
}

// -------------------------------------------------------------------- GEMM
// Block: 4 waves (2x2), tile 64 tokens x 64 features. Wave tile 32 x 32 via
// 2x2 WMMA tiles. A = activations (signed), B = weights (unsigned nibbles),
// D[token][feature].
__global__ void __launch_bounds__(128)
    Iu4Gemm(const uint8_t* __restrict__ w, const uint8_t* __restrict__ qs,
            const float* __restrict__ ds, const int* __restrict__ ss,
            float* __restrict__ out, int M, int N, int K) {
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = lane & 15, half = lane >> 4;
  const int m0 = blockIdx.x * 64 + (wave & 1) * 32;
  const int n0 = blockIdx.y * 64 + (wave >> 1) * 32;
  const int groups128 = K / kGroup;
  const size_t wrow_bytes = (size_t)(K / 256) * kBlockBytes;

  float acc[2][2][8];
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i)
        acc[a][b][i] = 0.0f;

  for (int g = 0; g < groups128; ++g) {
    const int g256 = g >> 1, hh = g & 1;
    i32x8 d[2][2];
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b)
        d[a][b] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    // per-lane weight header values for the feature column this lane serves
    float sc[2], zp[2];
    const uint8_t* wp[2];
    for (int b = 0; b < 2; ++b) {
      const int m = m0 + b * 16 + col;
      const uint8_t* blk =
          w + (size_t)m * wrow_bytes + (size_t)g256 * kBlockBytes;
      sc[b] = __half2float(*reinterpret_cast<const __half*>(blk + 4 * hh));
      zp[b] = __half2float(*reinterpret_cast<const __half*>(blk + 4 * hh + 2));
      wp[b] = blk + 8 + 64 * hh;
    }
    const uint8_t* xp[2];
    float sx[2];
    int sxs[2];
    for (int a = 0; a < 2; ++a) {
      const int n = n0 + a * 16 + col;
      xp[a] = qs + (size_t)n * (K / 2) + (size_t)g * 64;
      sx[a] = ds[(size_t)n * groups128 + g];
      sxs[a] = ss[(size_t)n * groups128 + g];
    }
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      i32x2 av[2], bv[2];
      for (int a = 0; a < 2; ++a) {
        const uint2 r = *reinterpret_cast<const uint2*>(xp[a] + 8 * kk);
        av[a] = (i32x2){(int)r.x, (int)r.y};
      }
      for (int b = 0; b < 2; ++b) {
        const uint2 r = *reinterpret_cast<const uint2*>(wp[b] + 8 * kk);
        bv[b] = (i32x2){(int)r.x, (int)r.y};
      }
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
          d[a][b] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, av[a], false, bv[b], d[a][b], false);
    }
    // epilogue: D[token row r][feature col]; row r = 2*i + half
    for (int a = 0; a < 2; ++a) {
      for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + half;
        const float sxr = __shfl(sx[a], r);
        const float ssr = (float)__shfl(sxs[a], r);
        for (int b = 0; b < 2; ++b)
          acc[a][b][i] += sxr * (sc[b] * (float)d[a][b][i] + zp[b] * ssr);
      }
    }
  }
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i) {
        const int n = n0 + a * 16 + 2 * i + half;
        const int m = m0 + b * 16 + col;
        out[(size_t)n * M + m] = acc[a][b][i];
      }
}

// ----------------------------------------------------------- GEMM, tiled v2
// Block: 8 waves (2 token x 4 feature), tile 128 tokens x 128 features, one
// 256-wide K block (two 128 groups) staged in LDS per step. Wave tile is
// 64 tokens x 32 features (4 x 2 WMMA tiles).
constexpr int kTileT = 128, kTileF = 128;
constexpr int kRowStride =
    136;  // LDS row stride in bytes (8-byte aligned, bank-skewed)

__global__ void __launch_bounds__(256)
    Iu4Gemm2(const uint8_t* __restrict__ w, const uint8_t* __restrict__ qs,
             const float* __restrict__ ds, const int* __restrict__ ss,
             float* __restrict__ out, int M, int N, int K) {
  __shared__ __attribute__((aligned(16))) uint8_t a_s[kTileT * kRowStride];
  __shared__ __attribute__((aligned(16))) uint8_t b_s[kTileF * kRowStride];
  __shared__ float sx_s[kTileT][2];
  __shared__ int ssum_s[kTileT][2];
  const int tid = threadIdx.x;
  const int wave = tid >> 5, lane = tid & 31;
  const int col = lane & 15, half = lane >> 4;
  const int wt = wave & 1, wf = wave >> 1;
  const int f0 = blockIdx.x * kTileF;
  const int t0 = blockIdx.y * kTileT;
  const int groups128 = K / kGroup;
  const int nblk = K / 256;
  const size_t wrow_bytes = (size_t)nblk * kBlockBytes;

  float acc[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i)
        acc[a][b][i] = 0.0f;

  for (int kb = 0; kb < nblk; ++kb) {
    __syncthreads();
#ifndef EXP_NO_STAGE
    // stage activations: 128 tokens x 128 bytes
    for (int idx = tid; idx < kTileT * 8; idx += 256) {
      const int row = idx >> 3, part = idx & 7;
      const uint4 v = *reinterpret_cast<const uint4*>(
          qs + (size_t)(t0 + row) * (K / 2) + (size_t)kb * 128 + part * 16);
      *reinterpret_cast<uint4*>(a_s + row * kRowStride + part * 16) = v;
    }
    // stage weights: 128 features x 136 bytes
    for (int idx = tid; idx < kTileF * 17; idx += 256) {
      const int row = idx / 17, part = idx % 17;
      const uint2 v =
          *reinterpret_cast<const uint2*>(w + (size_t)(f0 + row) * wrow_bytes +
                                          (size_t)kb * kBlockBytes + part * 8);
      *reinterpret_cast<uint2*>(b_s + row * kRowStride + part * 8) = v;
    }
    for (int idx = tid; idx < kTileT * 2; idx += 256) {
      const int row = idx >> 1, h = idx & 1;
      sx_s[row][h] = ds[(size_t)(t0 + row) * groups128 + kb * 2 + h];
      ssum_s[row][h] = ss[(size_t)(t0 + row) * groups128 + kb * 2 + h];
    }
#endif
    __syncthreads();
    for (int hh = 0; hh < 2; ++hh) {
      i32x8 d[4][2];
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 2; ++b)
          d[a][b] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        i32x2 av[4], bv[2];
        for (int a = 0; a < 4; ++a) {
          const uint2 r = *reinterpret_cast<const uint2*>(
              a_s + (wt * 64 + a * 16 + col) * kRowStride + 64 * hh + 8 * kk);
          av[a] = (i32x2){(int)r.x, (int)r.y};
        }
        for (int b = 0; b < 2; ++b) {
          const uint2 r = *reinterpret_cast<const uint2*>(
              b_s + (wf * 32 + b * 16 + col) * kRowStride + 8 + 64 * hh +
              8 * kk);
          bv[b] = (i32x2){(int)r.x, (int)r.y};
        }
        for (int a = 0; a < 4; ++a)
          for (int b = 0; b < 2; ++b)
            d[a][b] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, av[a], false, bv[b], d[a][b], false);
      }
      float sc[2], zp[2];
      for (int b = 0; b < 2; ++b) {
        const uint8_t* hdr =
            b_s + (wf * 32 + b * 16 + col) * kRowStride + 4 * hh;
        sc[b] = __half2float(*reinterpret_cast<const __half*>(hdr));
        zp[b] = __half2float(*reinterpret_cast<const __half*>(hdr + 2));
      }
#ifdef EXP_NO_EPI
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i)
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += (float)d[a][b][i];
      (void)sc;
      (void)zp;
#else
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i) {
          const int row = wt * 64 + a * 16 + 2 * i + half;
          const float sxr = sx_s[row][hh];
          const float ssr = (float)ssum_s[row][hh];
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += sxr * (sc[b] * (float)d[a][b][i] + zp[b] * ssr);
        }
#endif
    }
  }
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i) {
        const int n = t0 + wt * 64 + a * 16 + 2 * i + half;
        const int m = f0 + wf * 32 + b * 16 + col;
        out[(size_t)n * M + m] = acc[a][b][i];
      }
}

// ------------------------------------------------------ GEMM, tiled v3
// Weights are pre-XORed with 0x88 per byte so each nibble is the signed value
// q - 8 (symmetric grid, zero point -8 scale): no zero-point correction and
// both WMMA operands signed. The next K block is prefetched into registers
// while the current one is multiplied.
__global__ void __launch_bounds__(256)
    Iu4Gemm3(const uint8_t* __restrict__ w, const uint8_t* __restrict__ qs,
             const float* __restrict__ ds, float* __restrict__ out, int M,
             int N, int K) {
  constexpr int kRS =
      144;  // 16-byte aligned LDS rows for 128-bit fragment loads
  __shared__ __attribute__((aligned(16))) uint8_t a_s[kTileT * kRS];
  __shared__ __attribute__((aligned(16))) uint8_t b_s[kTileF * kRS];
  __shared__ float sx_s[kTileT][2];
  const int tid = threadIdx.x;
  const int wave = tid >> 5, lane = tid & 31;
  const int col = lane & 15, half = lane >> 4;
  const int wt = wave & 1, wf = wave >> 1;
  const int f0 = blockIdx.x * kTileF;
  const int t0 = blockIdx.y * kTileT;
  const int groups128 = K / kGroup;
  const int nblk = K / 256;
  const size_t wrow_bytes = (size_t)nblk * kBlockBytes;

  float acc[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i)
        acc[a][b][i] = 0.0f;

  uint4 ra[4];
  uint2 rb[9];
  float rs = 0.0f;
  const auto prefetch = [&](int kb) {
    for (int j = 0; j < 4; ++j) {
      const int idx = tid + j * 256, row = idx >> 3, part = idx & 7;
      ra[j] = *reinterpret_cast<const uint4*>(
          qs + (size_t)(t0 + row) * (K / 2) + (size_t)kb * 128 + part * 16);
    }
    for (int j = 0; j < 9; ++j) {
      const int idx = tid + j * 256;
      if (idx < kTileF * 17) {
        const int row = idx / 17, part = idx % 17;
        rb[j] = *reinterpret_cast<const uint2*>(
            w + (size_t)(f0 + row) * wrow_bytes + (size_t)kb * kBlockBytes +
            part * 8);
      }
    }
    {
      const int row = tid >> 1, h = tid & 1;
      rs = ds[(size_t)(t0 + row) * groups128 + kb * 2 + h];
    }
  };
  prefetch(0);
  for (int kb = 0; kb < nblk; ++kb) {
#ifndef EXP_NOSYNC
    __syncthreads();
#endif
#ifdef EXP_NOSTAGE
    if (kb == 0) {
#endif
      for (int j = 0; j < 4; ++j) {
        const int idx = tid + j * 256, row = idx >> 3, part = idx & 7;
        *reinterpret_cast<uint4*>(a_s + row * kRS + part * 16) = ra[j];
      }
      for (int j = 0; j < 9; ++j) {
        const int idx = tid + j * 256;
        if (idx < kTileF * 17) {
          const int row = idx / 17, part = idx % 17;
          *reinterpret_cast<uint2*>(b_s + row * kRS +
                                    (part == 0 ? 0 : part * 8 + 8)) = rb[j];
        }
      }
      sx_s[tid >> 1][tid & 1] = rs;
#ifdef EXP_NOSTAGE
    }
#endif
#ifndef EXP_NOSYNC
    __syncthreads();
#endif
#ifndef EXP_NOSTAGE
    if (kb + 1 < nblk)
      prefetch(kb + 1);
#endif
#pragma nounroll
    for (int hh = 0; hh < 2; ++hh) {
      i32x8 d[4][2];
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 2; ++b)
          d[a][b] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
      const uint8_t* a_base = a_s + (wt * 64 + col) * kRS + 64 * hh;
      const uint8_t* b_base = b_s + (wf * 32 + col) * kRS + 16 + 64 * hh;
      uint4 fa[2][4], fb[2][2];
      for (int a = 0; a < 4; ++a)
        fa[0][a] = *reinterpret_cast<const uint4*>(a_base + a * 16 * kRS);
      for (int b = 0; b < 2; ++b)
        fb[0][b] = *reinterpret_cast<const uint4*>(b_base + b * 16 * kRS);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int cur = j & 1, nxt = cur ^ 1;
        if (j + 1 < 4) {
          for (int a = 0; a < 4; ++a)
            fa[nxt][a] = *reinterpret_cast<const uint4*>(a_base + a * 16 * kRS +
                                                         16 * (j + 1));
          for (int b = 0; b < 2; ++b)
            fb[nxt][b] = *reinterpret_cast<const uint4*>(b_base + b * 16 * kRS +
                                                         16 * (j + 1));
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 2; ++b) {
              const int ax = h ? (int)fa[cur][a].z : (int)fa[cur][a].x;
              const int ay = h ? (int)fa[cur][a].w : (int)fa[cur][a].y;
              const int bx = h ? (int)fb[cur][b].z : (int)fb[cur][b].x;
              const int by = h ? (int)fb[cur][b].w : (int)fb[cur][b].y;
              d[a][b] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                  true, i32x2{ax, ay}, true, i32x2{bx, by}, d[a][b], false);
            }
        __builtin_amdgcn_sched_barrier(0);
      }
#ifdef EXP_NO_EPI
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i)
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += (float)d[a][b][i];
      continue;
#endif
      float sc[2];
      for (int b = 0; b < 2; ++b)
        sc[b] = __half2float(*reinterpret_cast<const __half*>(
            b_s + (wf * 32 + b * 16 + col) * kRS + 4 * hh));
#ifdef EXP_PT
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i)
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += sc[b] * (float)d[a][b][i];
#else
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i) {
          const float sxr = sx_s[wt * 64 + a * 16 + 2 * i + half][hh];
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += (sxr * sc[b]) * (float)d[a][b][i];
        }
#endif
    }
  }
#ifdef EXP_PT
  for (int a = 0; a < 4; ++a)
    for (int i = 0; i < 8; ++i) {
      const float sxr = sx_s[wt * 64 + a * 16 + 2 * i + half][0];
      for (int b = 0; b < 2; ++b)
        acc[a][b][i] *= sxr;
    }
#endif
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i) {
        const int n = t0 + wt * 64 + a * 16 + 2 * i + half;
        const int m = f0 + wf * 32 + b * 16 + col;
        out[(size_t)n * M + m] = acc[a][b][i];
      }
}
__global__ void __launch_bounds__(512)
    Iu4Gemm5(const uint8_t* __restrict__ w, const uint8_t* __restrict__ qs,
             const float* __restrict__ ds, float* __restrict__ out, int M,
             int N, int K) {
  constexpr int kRS =
      144;  // 16-byte aligned LDS rows for 128-bit fragment loads
  __shared__ __attribute__((aligned(16))) uint8_t a_s[256 * kRS];
  __shared__ __attribute__((aligned(16))) uint8_t b_s[kTileF * kRS];
  __shared__ float sx_s[256][2];
  const int tid = threadIdx.x;
  const int wave = tid >> 5, lane = tid & 31;
  const int col = lane & 15, half = lane >> 4;
  const int wt = wave & 3, wf = wave >> 2;
  const int f0 = blockIdx.x * kTileF;
  const int t0 = blockIdx.y * 256;
  const int groups128 = K / kGroup;
  const int nblk = K / 256;
  const size_t wrow_bytes = (size_t)nblk * kBlockBytes;

  float acc[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i)
        acc[a][b][i] = 0.0f;

  uint4 ra[4];
  uint2 rb[5];
  float rs = 0.0f;
  const auto prefetch = [&](int kb) {
    for (int j = 0; j < 4; ++j) {
      const int idx = tid + j * 512, row = idx >> 3, part = idx & 7;
      ra[j] = *reinterpret_cast<const uint4*>(
          qs + (size_t)(t0 + row) * (K / 2) + (size_t)kb * 128 + part * 16);
    }
    for (int j = 0; j < 5; ++j) {
      const int idx = tid + j * 512;
      if (idx < kTileF * 17) {
        const int row = idx / 17, part = idx % 17;
        rb[j] = *reinterpret_cast<const uint2*>(
            w + (size_t)(f0 + row) * wrow_bytes + (size_t)kb * kBlockBytes +
            part * 8);
      }
    }
    {
      const int row = tid >> 1, h = tid & 1;
      rs = ds[(size_t)(t0 + row) * groups128 + kb * 2 + h];
    }
  };
  prefetch(0);
  for (int kb = 0; kb < nblk; ++kb) {
#ifndef EXP_NOSYNC
    __syncthreads();
#endif
#ifdef EXP_NOSTAGE
    if (kb == 0) {
#endif
      for (int j = 0; j < 4; ++j) {
        const int idx = tid + j * 512, row = idx >> 3, part = idx & 7;
        *reinterpret_cast<uint4*>(a_s + row * kRS + part * 16) = ra[j];
      }
      for (int j = 0; j < 5; ++j) {
        const int idx = tid + j * 512;
        if (idx < kTileF * 17) {
          const int row = idx / 17, part = idx % 17;
          *reinterpret_cast<uint2*>(b_s + row * kRS +
                                    (part == 0 ? 0 : part * 8 + 8)) = rb[j];
        }
      }
      sx_s[tid >> 1][tid & 1] = rs;
#ifdef EXP_NOSTAGE
    }
#endif
#ifndef EXP_NOSYNC
    __syncthreads();
#endif
#ifndef EXP_NOSTAGE
    if (kb + 1 < nblk)
      prefetch(kb + 1);
#endif
#pragma nounroll
    for (int hh = 0; hh < 2; ++hh) {
      i32x8 d[4][2];
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 2; ++b)
          d[a][b] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
      const uint8_t* a_base = a_s + (wt * 64 + col) * kRS + 64 * hh;
      const uint8_t* b_base = b_s + (wf * 32 + col) * kRS + 16 + 64 * hh;
      uint4 fa[2][4], fb[2][2];
      for (int a = 0; a < 4; ++a)
        fa[0][a] = *reinterpret_cast<const uint4*>(a_base + a * 16 * kRS);
      for (int b = 0; b < 2; ++b)
        fb[0][b] = *reinterpret_cast<const uint4*>(b_base + b * 16 * kRS);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int cur = j & 1, nxt = cur ^ 1;
        if (j + 1 < 4) {
          for (int a = 0; a < 4; ++a)
            fa[nxt][a] = *reinterpret_cast<const uint4*>(a_base + a * 16 * kRS +
                                                         16 * (j + 1));
          for (int b = 0; b < 2; ++b)
            fb[nxt][b] = *reinterpret_cast<const uint4*>(b_base + b * 16 * kRS +
                                                         16 * (j + 1));
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 2; ++b) {
              const int ax = h ? (int)fa[cur][a].z : (int)fa[cur][a].x;
              const int ay = h ? (int)fa[cur][a].w : (int)fa[cur][a].y;
              const int bx = h ? (int)fb[cur][b].z : (int)fb[cur][b].x;
              const int by = h ? (int)fb[cur][b].w : (int)fb[cur][b].y;
              d[a][b] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                  true, i32x2{ax, ay}, true, i32x2{bx, by}, d[a][b], false);
            }
        __builtin_amdgcn_sched_barrier(0);
      }
#ifdef EXP_NO_EPI
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i)
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += (float)d[a][b][i];
      continue;
#endif
      float sc[2];
      for (int b = 0; b < 2; ++b)
        sc[b] = __half2float(*reinterpret_cast<const __half*>(
            b_s + (wf * 32 + b * 16 + col) * kRS + 4 * hh));
#ifdef EXP_PT
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i)
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += sc[b] * (float)d[a][b][i];
#else
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 8; ++i) {
          const float sxr = sx_s[wt * 64 + a * 16 + 2 * i + half][hh];
          for (int b = 0; b < 2; ++b)
            acc[a][b][i] += (sxr * sc[b]) * (float)d[a][b][i];
        }
#endif
    }
  }
#ifdef EXP_PT
  for (int a = 0; a < 4; ++a)
    for (int i = 0; i < 8; ++i) {
      const float sxr = sx_s[wt * 64 + a * 16 + 2 * i + half][0];
      for (int b = 0; b < 2; ++b)
        acc[a][b][i] *= sxr;
    }
#endif
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i) {
        const int n = t0 + wt * 64 + a * 16 + 2 * i + half;
        const int m = f0 + wf * 32 + b * 16 + col;
        out[(size_t)n * M + m] = acc[a][b][i];
      }
}
namespace v6 {
constexpr int kNW = 8;
constexpr int kAB = 128 * 64;
constexpr int kSlot = 2 * kAB;

template<typename T>
__device__ __forceinline__ const T* At(const void* p, uint32_t off) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(p) + off);
}

__device__ __forceinline__ int RowShare(int v, int k) {
#define V6_SHARE(k) \
  case k:           \
    return __builtin_amdgcn_update_dpp(0, v, 0x150 + k, 0xf, 0xf, false);
  switch (k) {
    V6_SHARE(0)
    V6_SHARE(1)
    V6_SHARE(2)
    V6_SHARE(3)
    V6_SHARE(4)
    V6_SHARE(5) V6_SHARE(6) V6_SHARE(7) V6_SHARE(8) V6_SHARE(9) V6_SHARE(10)
        V6_SHARE(11) V6_SHARE(12) V6_SHARE(13) V6_SHARE(14) V6_SHARE(15) default
        : __builtin_unreachable();
  }
#undef V6_SHARE
}
}  // namespace v6

template<bool PT>
__global__ void __launch_bounds__(256)
    Iu4Gemm6(const uint8_t* __restrict__ w, const uint8_t* __restrict__ qs,
             const float* __restrict__ ds, float* __restrict__ out, int M,
             int N, int K) {
  using namespace v6;
#ifdef EXP6_PAD
  __shared__ char pad[EXP6_PAD];
  if (threadIdx.x == 1023)
    pad[threadIdx.x] = 1;
#endif
  __shared__ __attribute__((aligned(16))) char lds[2 * kSlot];
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wt = wave & 3, wf = wave >> 2;
  const int tb = N >> 7;
  const int f0 = (blockIdx.x / tb) * 128, t0 = (blockIdx.x % tb) * 128;
  const int E = K >> 7;
  const int groups128 = K / kGroup;
  const uint32_t wrow = (uint32_t)(K >> 8) * kBlockBytes;
  const uint32_t qrow = (uint32_t)(K >> 1);
  const int sr = 16 * wave + lr;
  const char* gw = reinterpret_cast<const char*>(w) + (size_t)(f0 + sr) * wrow +
                   8u + 8u * hi;
  const char* gq =
      reinterpret_cast<const char*>(qs) + (size_t)(t0 + sr) * qrow + 8u * hi;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wt * 2) * 1024 + prow * 8;
  const uint32_t boff = kAB + (wf * 4) * 1024 + lr * 8;
  const char* gsc =
      reinterpret_cast<const char*>(w) + (size_t)(f0 + wf * 64 + lr) * wrow;
  const float* gd =
      ds +
      (size_t)(t0 + wt * 32 + 16 * (lr >> 3) + 8 * hi + (lr & 7)) * groups128;

  float sum[2][4][8];
  for (int a = 0; a < 2; ++a)
    for (int c = 0; c < 4; ++c)
      for (int j = 0; j < 8; ++j)
        sum[a][c][j] = 0.0f;

  i32x2 sa[4], sb[4];
  const auto stage_load = [&](int e) {
    const uint32_t eo =
        (uint32_t)(e >> 1) * kBlockBytes + (uint32_t)(e & 1) * 64;
    const uint32_t qo = (uint32_t)e * 64;
    for (int i = 0; i < 4; ++i) {
      sb[i] = *At<i32x2>(gw, eo + 16u * i);
      sa[i] = *At<i32x2>(gq, qo + 16u * i);
    }
  };
  const auto stage_store = [&](char* slot) {
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<i32x2*>(slot + lds_off + i * 256) = sa[i];
      *reinterpret_cast<i32x2*>(slot + kAB + lds_off + i * 256) = sb[i];
    }
  };
  stage_load(0);
  stage_store(lds);
  unsigned short scn[4];
  float wsn = 0.0f;
  const auto meta_load = [&](int e) {
    for (int c = 0; c < 4; ++c)
      scn[c] =
          *At<unsigned short>(gsc, (uint32_t)(e >> 1) * kBlockBytes +
                                       (uint32_t)(e & 1) * 4 + c * 16 * wrow);
    if (!PT)
      wsn = gd[e];
  };
  meta_load(0);
  __syncthreads();

  const auto epoch = [&](auto nxt_tag, int e) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    char* slot = lds + (e & 1) * kSlot;
    float sc[4];
    for (int c = 0; c < 4; ++c)
      sc[c] = __half2float(__ushort_as_half(scn[c]));
    const int wsf = __float_as_int(wsn);
    __builtin_amdgcn_sched_barrier(0);
#ifndef EXP6_NOLOAD
    if (nxt)
      stage_load(e + 1);
#endif
    if (nxt)
      meta_load(e + 1);
    __builtin_amdgcn_sched_barrier(0);
    i32x2 av = *reinterpret_cast<const i32x2*>(slot + aoff);
    i32x2 bv[4];
    for (int c = 0; c < 4; ++c)
      bv[c] = *reinterpret_cast<const i32x2*>(slot + boff + c * 1024);
    i32x8 C[4];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int i = a * 8 + s + 1;
        i32x2 an;
        i32x2 bn[4];
        if (i < 16) {
          const int na = i >> 3, ns = i & 7;
          an = *reinterpret_cast<const i32x2*>(slot + aoff + na * 1024 +
                                               ns * 128);
          for (int c = 0; c < 4; ++c)
            bn[c] = *reinterpret_cast<const i32x2*>(slot + boff + c * 1024 +
                                                    ns * 128);
        }
        for (int c = 0; c < 4; ++c) {
          const i32x8 z = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
          C[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, av, true, bv[c], s == 0 ? z : C[c], false);
        }
        if (i < 16) {
          av = an;
          for (int c = 0; c < 4; ++c)
            bv[c] = bn[c];
        }
        if (s != 7)
          __builtin_amdgcn_sched_barrier(0);
      }
      float scf[8];
      if (!PT) {
        for (int j = 0; j < 8; ++j) {
          scf[j] = __int_as_float(RowShare(wsf, 8 * a + j));
          asm volatile("" : "+v"(scf[j]));
        }
      }
      for (int c = 0; c < 4; ++c) {
        for (int j = 0; j < 8; ++j) {
#ifdef EXP6_NOFOLD
          sum[a][c][j] += (float)C[c][j];
#else
          if (PT)
            sum[a][c][j] = __builtin_fmaf(sc[c], (float)C[c][j], sum[a][c][j]);
          else
            sum[a][c][j] =
                __builtin_fmaf(scf[j] * sc[c], (float)C[c][j], sum[a][c][j]);
#endif
        }
        asm volatile(""
                     : "+v"(sum[a][c][0]), "+v"(sum[a][c][1]),
                       "+v"(sum[a][c][2]), "+v"(sum[a][c][3]),
                       "+v"(sum[a][c][4]), "+v"(sum[a][c][5]),
                       "+v"(sum[a][c][6]), "+v"(sum[a][c][7]));
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (nxt)
      stage_store(lds + ((e + 1) & 1) * kSlot);
    __syncthreads();
  };
  for (int e = 0; e + 1 < E; ++e)
    epoch(std::true_type{}, e);
  epoch(std::false_type{}, E - 1);

  for (int a = 0; a < 2; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wt * 32 + a * 16 + 8 * hi + j;
      float r = 1.0f;
      if (PT)
        r = ds[(size_t)t * groups128];
      for (int c = 0; c < 4; ++c)
        out[(size_t)t * M + f0 + wf * 64 + c * 16 + lr] = sum[a][c][j] * r;
    }
}

template<bool PT>
__global__ void __launch_bounds__(512)
    Iu4Gemm8(const uint8_t* __restrict__ w, const uint8_t* __restrict__ qs,
             const float* __restrict__ ds, float* __restrict__ out, int M,
             int N, int K) {
  using namespace v6;
  constexpr int kAB8 = 256 * 64, kSlot8 = kAB8 + 128 * 64;
  __shared__ __attribute__((aligned(16))) char lds[2 * kSlot8];
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wt = wave & 7, wf = wave >> 3;
  const int tb = N >> 8;
  const int f0 = (blockIdx.x / tb) * 128, t0 = (blockIdx.x % tb) * 256;
  const int E = K >> 7;
  const int groups128 = K / kGroup;
  const uint32_t wrow = (uint32_t)(K >> 8) * kBlockBytes;
  const uint32_t qrow = (uint32_t)(K >> 1);
  const int sr = 16 * wave + lr;
  const int swr = 16 * (wave >> 1) + lr, wh = wave & 1;
  const char* gw = reinterpret_cast<const char*>(w) +
                   (size_t)(f0 + swr) * wrow + 8u + 8u * hi + 32u * wh;
  const char* gq =
      reinterpret_cast<const char*>(qs) + (size_t)(t0 + sr) * qrow + 8u * hi;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wt * 2) * 1024 + prow * 8;
  const uint32_t boff = kAB8 + (wf * 4) * 1024 + lr * 8;
  const char* gsc =
      reinterpret_cast<const char*>(w) + (size_t)(f0 + wf * 64 + lr) * wrow;
  const float* gd =
      ds +
      (size_t)(t0 + wt * 32 + 16 * (lr >> 3) + 8 * hi + (lr & 7)) * groups128;

  float sum[2][4][8];
  for (int a = 0; a < 2; ++a)
    for (int c = 0; c < 4; ++c)
      for (int j = 0; j < 8; ++j)
        sum[a][c][j] = 0.0f;

  i32x2 sa[4], sb[2];
  const auto stage_load = [&](int e) {
    const uint32_t eo =
        (uint32_t)(e >> 1) * kBlockBytes + (uint32_t)(e & 1) * 64;
    const uint32_t qo = (uint32_t)e * 64;
    for (int i = 0; i < 2; ++i)
      sb[i] = *At<i32x2>(gw, eo + 16u * i);
    for (int i = 0; i < 4; ++i)
      sa[i] = *At<i32x2>(gq, qo + 16u * i);
  };
  const auto stage_store = [&](char* slot) {
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<i32x2*>(slot + lds_off + i * 256) = sa[i];
    for (int i = 0; i < 2; ++i)
      *reinterpret_cast<i32x2*>(slot + kAB8 + (wave >> 1) * 1024 +
                                (2 * wh + i) * 256 + hi * 128 + lr * 8) = sb[i];
  };
  stage_load(0);
  stage_store(lds);
  unsigned short scn[4];
  float wsn = 0.0f;
  const auto meta_load = [&](int e) {
    for (int c = 0; c < 4; ++c)
      scn[c] =
          *At<unsigned short>(gsc, (uint32_t)(e >> 1) * kBlockBytes +
                                       (uint32_t)(e & 1) * 4 + c * 16 * wrow);
    if (!PT)
      wsn = gd[e];
  };
  meta_load(0);
  __syncthreads();

  const auto epoch = [&](auto nxt_tag, int e) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    char* slot = lds + (e & 1) * kSlot8;
    float sc[4];
    for (int c = 0; c < 4; ++c)
      sc[c] = __half2float(__ushort_as_half(scn[c]));
    const int wsf = __float_as_int(wsn);
    __builtin_amdgcn_sched_barrier(0);
#ifndef EXP6_NOLOAD
    if (nxt)
      stage_load(e + 1);
#endif
    if (nxt)
      meta_load(e + 1);
    __builtin_amdgcn_sched_barrier(0);
    i32x2 av = *reinterpret_cast<const i32x2*>(slot + aoff);
    i32x2 bv[4];
    for (int c = 0; c < 4; ++c)
      bv[c] = *reinterpret_cast<const i32x2*>(slot + boff + c * 1024);
    i32x8 C[4];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int i = a * 8 + s + 1;
        i32x2 an;
        i32x2 bn[4];
        if (i < 16) {
          const int na = i >> 3, ns = i & 7;
          an = *reinterpret_cast<const i32x2*>(slot + aoff + na * 1024 +
                                               ns * 128);
          for (int c = 0; c < 4; ++c)
            bn[c] = *reinterpret_cast<const i32x2*>(slot + boff + c * 1024 +
                                                    ns * 128);
        }
        for (int c = 0; c < 4; ++c) {
          const i32x8 z = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
          C[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, av, true, bv[c], s == 0 ? z : C[c], false);
        }
        if (i < 16) {
          av = an;
          for (int c = 0; c < 4; ++c)
            bv[c] = bn[c];
        }
        if (s != 7)
          __builtin_amdgcn_sched_barrier(0);
      }
      float scf[8];
      if (!PT) {
        for (int j = 0; j < 8; ++j) {
          scf[j] = __int_as_float(RowShare(wsf, 8 * a + j));
          asm volatile("" : "+v"(scf[j]));
        }
      }
      for (int c = 0; c < 4; ++c) {
        for (int j = 0; j < 8; ++j) {
#ifdef EXP6_NOFOLD
          sum[a][c][j] += (float)C[c][j];
#else
          if (PT)
            sum[a][c][j] = __builtin_fmaf(sc[c], (float)C[c][j], sum[a][c][j]);
          else
            sum[a][c][j] =
                __builtin_fmaf(scf[j] * sc[c], (float)C[c][j], sum[a][c][j]);
#endif
        }
        asm volatile(""
                     : "+v"(sum[a][c][0]), "+v"(sum[a][c][1]),
                       "+v"(sum[a][c][2]), "+v"(sum[a][c][3]),
                       "+v"(sum[a][c][4]), "+v"(sum[a][c][5]),
                       "+v"(sum[a][c][6]), "+v"(sum[a][c][7]));
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (nxt)
      stage_store(lds + ((e + 1) & 1) * kSlot8);
    __syncthreads();
  };
  for (int e = 0; e + 1 < E; ++e)
    epoch(std::true_type{}, e);
  epoch(std::false_type{}, E - 1);

  for (int a = 0; a < 2; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wt * 32 + a * 16 + 8 * hi + j;
      float r = 1.0f;
      if (PT)
        r = ds[(size_t)t * groups128];
      for (int c = 0; c < 4; ++c)
        out[(size_t)t * M + f0 + wf * 64 + c * 16 + lr] = sum[a][c][j] * r;
    }
}

__global__ void Repack9(const uint8_t* __restrict__ w, uint8_t* __restrict__ pw,
                        unsigned short* __restrict__ psc, int M, int K) {
  const int row = blockIdx.x, e = blockIdx.y;
  const int nblk = K / 256;
  const uint8_t* src =
      w + (size_t)row * nblk * kBlockBytes + (size_t)(e >> 1) * kBlockBytes;
  const int t = threadIdx.x;
  if (t < 64)
    pw[((size_t)e * M + row) * 64 + t] = src[8 + 64 * (e & 1) + t];
  if (t == 64)
    psc[(size_t)e * M + row] =
        *reinterpret_cast<const unsigned short*>(src + 4 * (e & 1));
}

template<bool PT>
__global__ void __launch_bounds__(256)
    Iu4Gemm9(const uint8_t* __restrict__ pw,
             const unsigned short* __restrict__ psc,
             const uint8_t* __restrict__ qs, const float* __restrict__ ds,
             float* __restrict__ out, int M, int N, int K) {
  using namespace v6;
  __shared__ __attribute__((aligned(16))) char lds[2 * 8192];
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wf = wave & 3, wth = wave >> 2;
  const int tb = N >> 7;
  const int f0 = (blockIdx.x / tb) * 128, t0 = (blockIdx.x % tb) * 128;
  const int E = K >> 7;
  const int groups128 = K / kGroup;
  const uint32_t qrow = (uint32_t)(K >> 1);
  const int sr = 16 * wave + lr;
  const char* gq =
      reinterpret_cast<const char*>(qs) + (size_t)(t0 + sr) * qrow + 8u * hi;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wth * 4) * 1024 + prow * 8;
  const int frow = f0 + wf * 32 + lr;
  const float* gd =
      ds +
      (size_t)(t0 + wth * 64 + 16 * (lr >> 3) + 8 * hi + (lr & 7)) * groups128;

  float sum[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < 8; ++j)
        sum[a][b][j] = 0.0f;

  i32x2 sa[4];
  const auto stage_load = [&](int e) {
    for (int i = 0; i < 4; ++i)
      sa[i] = *At<i32x2>(gq, (uint32_t)e * 64 + 16u * i);
  };
  const auto stage_store = [&](char* slot) {
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<i32x2*>(slot + lds_off + i * 256) = sa[i];
  };
  using W = uint4[2][4];
  uint4 wA[2][4];
  const auto wload = [&](W& dst, int e) {
    for (int b = 0; b < 2; ++b) {
      const char* p = reinterpret_cast<const char*>(pw) +
                      ((size_t)e * M + frow + 16 * b) * 64;
      for (int q = 0; q < 4; ++q)
        dst[b][q] = *reinterpret_cast<const uint4*>(p + 16 * q);
    }
  };
  unsigned short scn[2];
  float wsn0 = 0.0f, wsn1 = 0.0f;
  const auto meta_load = [&](int e) {
    for (int b = 0; b < 2; ++b)
      scn[b] = psc[(size_t)e * M + frow + 16 * b];
    if (!PT) {
      wsn0 = gd[e];
      wsn1 = gd[e + 32 * groups128];
    }
  };
  stage_load(0);
  stage_store(lds);
  wload(wA, 0);
  meta_load(0);
  __syncthreads();

  const auto epoch = [&](W& wc, auto nxt_tag, int e, int par) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    char* slot = lds + par * 8192;
    float sc[2];
    for (int b = 0; b < 2; ++b)
      sc[b] = __half2float(__ushort_as_half(scn[b]));
    const int wsf0 = __float_as_int(wsn0), wsf1 = __float_as_int(wsn1);
    __builtin_amdgcn_sched_barrier(0);
    if (nxt) {
      stage_load(e + 1);
      meta_load(e + 1);
    }
    __builtin_amdgcn_sched_barrier(0);
    i32x8 C[2][4];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      i32x2 ar[4];
      for (int a = 0; a < 4; ++a)
        ar[a] =
            *reinterpret_cast<const i32x2*>(slot + aoff + a * 1024 + s * 128);
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const uint4 wq = wc[b][s >> 1];
        const i32x2 wv =
            (s & 1) ? i32x2{(int)wq.z, (int)wq.w} : i32x2{(int)wq.x, (int)wq.y};
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          const i32x8 z = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
          C[b][a] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, ar[a], true, wv, s == 0 ? z : C[b][a], false);
        }
      }
      if (nxt && (s & 1)) {
        for (int b = 0; b < 2; ++b) {
          const char* p = reinterpret_cast<const char*>(pw) +
                          ((size_t)(e + 1) * M + frow + 16 * b) * 64;
          wc[b][s >> 1] = *reinterpret_cast<const uint4*>(p + 16 * (s >> 1));
        }
      }
      if (s != 7)
        __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      float scf[8];
      if (!PT) {
        const int wsf = a < 2 ? wsf0 : wsf1;
        for (int j = 0; j < 8; ++j) {
          scf[j] = __int_as_float(RowShare(wsf, 8 * (a & 1) + j));
          asm volatile("" : "+v"(scf[j]));
        }
      }
      for (int b = 0; b < 2; ++b) {
        for (int j = 0; j < 8; ++j) {
          if (PT)
            sum[a][b][j] =
                __builtin_fmaf(sc[b], (float)C[b][a][j], sum[a][b][j]);
          else
            sum[a][b][j] =
                __builtin_fmaf(scf[j] * sc[b], (float)C[b][a][j], sum[a][b][j]);
        }
        asm volatile(""
                     : "+v"(sum[a][b][0]), "+v"(sum[a][b][1]),
                       "+v"(sum[a][b][2]), "+v"(sum[a][b][3]),
                       "+v"(sum[a][b][4]), "+v"(sum[a][b][5]),
                       "+v"(sum[a][b][6]), "+v"(sum[a][b][7]));
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (nxt)
      stage_store(lds + (par ^ 1) * 8192);
    __syncthreads();
  };
  int e = 0;
  for (; e + 2 < E; e += 2) {
    epoch(wA, std::true_type{}, e, 0);
    epoch(wA, std::true_type{}, e + 1, 1);
  }
  epoch(wA, std::true_type{}, e, 0);
  epoch(wA, std::false_type{}, e + 1, 1);

  for (int a = 0; a < 4; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wth * 64 + a * 16 + 8 * hi + j;
      const float r = PT ? ds[(size_t)t * groups128] : 1.0f;
      for (int b = 0; b < 2; ++b)
        out[(size_t)t * M + f0 + wf * 32 + b * 16 + lr] = sum[a][b][j] * r;
    }
}

template<bool PT, bool Y>
__device__ __forceinline__ void Gemm10Body(
    const uint8_t* __restrict__ pw, const unsigned short* __restrict__ psc,
    const uint8_t* __restrict__ qs, const float* __restrict__ ds,
    float* __restrict__ out, int M, int N, int K, char* lds) {
  using namespace v6;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wf = wave & 3, wth = wave >> 2;
  const int tb = N >> 7;
  const int f0 = (blockIdx.x / tb) * 128, t0 = (blockIdx.x % tb) * 128;
  const int E = K >> 7;
  const int groups128 = K / kGroup;
  const uint32_t qrow = (uint32_t)(K >> 1);
  const int sr = 16 * wave + lr;
  const char* gq =
      reinterpret_cast<const char*>(qs) + (size_t)(t0 + sr) * qrow + 8u * hi;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wth * 4) * 1024 + prow * 8;
  const int frow = f0 + wf * 32 + lr;
  const float* gd =
      ds +
      (size_t)(t0 + wth * 64 + 16 * (lr >> 3) + 8 * hi + (lr & 7)) * groups128;

  float sum[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < 8; ++j)
        sum[a][b][j] = 0.0f;

  float sxn = 0.0f;
  i32x2 sa[4];
  const auto stage_load = [&](int e) {
    for (int i = 0; i < 4; ++i)
      sa[i] = *At<i32x2>(gq, (uint32_t)e * 64 + 16u * i);
    if (!PT && tid < 128)
      sxn = ds[(size_t)(t0 + tid) * groups128 + e];
  };
  const auto stage_store = [&](char* slot, char* sreg) {
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<i32x2*>(slot + lds_off + i * 256) = sa[i];
    if (!PT && tid < 128)
      *reinterpret_cast<float*>(sreg + tid * 4) = sxn;
  };
  uint4 wc[2][4];
  for (int b = 0; b < 2; ++b) {
    const char* p = reinterpret_cast<const char*>(pw) +
                    ((size_t)0 * M + frow + 16 * b) * 64;
    for (int q = 0; q < 4; ++q)
      wc[b][q] = *reinterpret_cast<const uint4*>(p + 16 * q);
  }
  unsigned short scn[2];
  const auto meta_load = [&](int e) {
    for (int b = 0; b < 2; ++b)
      scn[b] = psc[(size_t)e * M + frow + 16 * b];
  };
  stage_load(0);
  stage_store(lds, lds + 16384);
  meta_load(0);
  __syncthreads();

  i32x8 Cp[2][2];
  for (int b = 0; b < 2; ++b)
    for (int a = 0; a < 2; ++a)
      Cp[b][a] = (i32x8){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                         0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
  float scp[2] = {0.0f, 0.0f};
  const char* sslp = lds + 16384;

  const auto fold = [&](int h, i32x8(&C)[2][2], const float (&sc)[2],
                        const char* ssl) {
#pragma unroll
    for (int a2 = 0; a2 < 2; ++a2) {
      const int a = 2 * h + a2;
      float scf[8];
      if (!PT) {
        const float4 s0 = *reinterpret_cast<const float4*>(
            ssl + (wth * 64 + 16 * a + 8 * hi) * 4);
        const float4 s1 = *reinterpret_cast<const float4*>(
            ssl + (wth * 64 + 16 * a + 8 * hi) * 4 + 16);
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        for (int j = 0; j < 8; ++j) {
          scf[j] = sv[j];

          asm volatile("" : "+v"(scf[j]));
        }
      }
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        for (int j = 0; j < 8; ++j) {
#ifdef EXP10_NOFOLD
          sum[a][b][j] += __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f);
          continue;
#endif
          if (PT)
            sum[a][b][j] = __builtin_fmaf(
                sc[b], __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f),
                sum[a][b][j]);
          else
            sum[a][b][j] = __builtin_fmaf(
                scf[j] * sc[b],
                __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f),
                sum[a][b][j]);
        }
        asm volatile(""
                     : "+v"(sum[a][b][0]), "+v"(sum[a][b][1]),
                       "+v"(sum[a][b][2]), "+v"(sum[a][b][3]),
                       "+v"(sum[a][b][4]), "+v"(sum[a][b][5]),
                       "+v"(sum[a][b][6]), "+v"(sum[a][b][7]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  };

  const auto wmma = [&](int h, i32x8(&C)[2][2], const char* slot, bool reload,
                        int e) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      i32x2 ar[2];
      for (int a2 = 0; a2 < 2; ++a2)
        ar[a2] = *reinterpret_cast<const i32x2*>(slot + aoff +
                                                 (2 * h + a2) * 1024 + s * 128);
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const uint4 wq = wc[b][s >> 1];
        const i32x2 wv =
            (s & 1) ? i32x2{(int)wq.z, (int)wq.w} : i32x2{(int)wq.x, (int)wq.y};
#pragma unroll
        for (int a2 = 0; a2 < 2; ++a2) {
          const i32x8 z =
              (i32x8){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                      0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
          C[b][a2] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, ar[a2], true, wv, s == 0 ? z : C[b][a2], false);
        }
      }
#if defined(EXP10_NOLOAD) || defined(EXP10_NOW)
      if (false) {
#else
      if (reload && (s & 1)) {
#endif
        for (int b = 0; b < 2; ++b) {
          const char* p = reinterpret_cast<const char*>(pw) +
                          ((size_t)(e + 1) * M + frow + 16 * b) * 64;
          wc[b][s >> 1] = *reinterpret_cast<const uint4*>(p + 16 * (s >> 1));
        }
      }
      if (s != 7)
        __builtin_amdgcn_sched_barrier(0);
    }
  };

  const auto epoch = [&](auto nxt_tag, int e, int par) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    const char* slot = lds + par * 8192;
    float sc[2];
    for (int b = 0; b < 2; ++b)
      sc[b] = __half2float(__ushort_as_half(scn[b]));

    __builtin_amdgcn_sched_barrier(0);
#ifndef EXP10_NOLOAD
    if (nxt) {
#ifndef EXP10_NOTOK
      stage_load(e + 1);
#endif
      meta_load(e + 1);
    }
#endif
    __builtin_amdgcn_sched_barrier(0);
    i32x8 C0[2][2], C1[2][2];
    if (Y)
      fold(1, Cp, scp, sslp);
    wmma(0, C0, slot, false, e);
    fold(0, C0, sc, lds + 16384 + (e & 3) * 512);
    wmma(1, C1, slot, nxt, e);
    if (Y) {
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a)
          Cp[b][a] = C1[b][a];
      scp[0] = sc[0];
      scp[1] = sc[1];
      sslp = lds + 16384 + (e & 3) * 512;
    } else {
      fold(1, C1, sc, lds + 16384 + (e & 3) * 512);
    }
    if (nxt)
      stage_store(lds + (par ^ 1) * 8192, lds + 16384 + ((e + 1) & 3) * 512);
    __syncthreads();
  };
  int e = 0;
  for (; e + 2 < E; e += 2) {
    epoch(std::true_type{}, e, 0);
    epoch(std::true_type{}, e + 1, 1);
  }
  epoch(std::true_type{}, e, 0);
  epoch(std::false_type{}, e + 1, 1);
  if (Y)
    fold(1, Cp, scp, sslp);

  for (int a = 0; a < 4; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wth * 64 + a * 16 + 8 * hi + j;
      const float r = PT ? ds[(size_t)t * groups128] : 1.0f;
      for (int b = 0; b < 2; ++b)
        out[(size_t)t * M + f0 + wf * 32 + b * 16 + lr] = sum[a][b][j] * r;
    }
}

template<bool PT>
__global__ void __launch_bounds__(256)
    Iu4Gemm10(const uint8_t* __restrict__ pw,
              const unsigned short* __restrict__ psc,
              const uint8_t* __restrict__ qs, const float* __restrict__ ds,
              float* __restrict__ out, int M, int N, int K) {
  __shared__ __attribute__((aligned(16))) char lds[16384 + 2048];
  if ((threadIdx.x >> 7) & 1)
    Gemm10Body<PT, true>(pw, psc, qs, ds, out, M, N, K, lds);
  else
    Gemm10Body<PT, false>(pw, psc, qs, ds, out, M, N, K, lds);
}

__global__ void TransposeScales(const float* __restrict__ ds,
                                float* __restrict__ dsT, int N, int G) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N)
    return;
  for (int e = 0; e < G; ++e)
    dsT[(size_t)e * N + t] = ds[(size_t)t * G + e];
}

template<bool PT, bool Y>
__device__ __forceinline__ void Gemm11Body(
    const uint8_t* __restrict__ pw, const unsigned short* __restrict__ psc,
    const uint8_t* __restrict__ qs, const float* __restrict__ dsT,
    const float* __restrict__ ds, float* __restrict__ out, int M, int N,
    int K) {
  using namespace v6;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wf = wave & 3, wth = wave >> 2;
  const int tb = N >> 7;
  const int f0 = (blockIdx.x / tb) * 128, t0 = (blockIdx.x % tb) * 128;
  const int E = K >> 7;
  const int groups128 = K / kGroup;
  const uint32_t qrow = (uint32_t)(K >> 1);
  const int frow = f0 + wf * 32 + lr;
  const char* tokp[4];
  for (int a = 0; a < 4; ++a)
    tokp[a] = reinterpret_cast<const char*>(qs) +
              (size_t)(t0 + wth * 64 + 16 * a + prow) * qrow;
  const float* sxp = dsT + (t0 + wth * 64 + 8 * hi);

  float sum[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < 8; ++j)
        sum[a][b][j] = 0.0f;

  uint4 tc[2][4], wc[2][4];
  const auto tload = [&](int h, int e) {
    for (int a2 = 0; a2 < 2; ++a2) {
      const char* p = tokp[2 * h + a2] + (size_t)e * 64;
      for (int q = 0; q < 4; ++q)
        tc[a2][q] = *reinterpret_cast<const uint4*>(p + 16 * q);
    }
  };
  const auto tload1 = [&](int h, int e, int q) {
    for (int a2 = 0; a2 < 2; ++a2)
      tc[a2][q] = *reinterpret_cast<const uint4*>(tokp[2 * h + a2] +
                                                  (size_t)e * 64 + 16 * q);
  };
  const auto wload1 = [&](int e, int q) {
    for (int b = 0; b < 2; ++b)
      wc[b][q] = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const char*>(pw) +
          ((size_t)e * M + frow + 16 * b) * 64 + 16 * q);
  };
  for (int q = 0; q < 4; ++q)
    wload1(0, q);
  tload(0, 0);
  unsigned short scn[2];
  const auto meta_load = [&](int e) {
    for (int b = 0; b < 2; ++b)
      scn[b] = psc[(size_t)e * M + frow + 16 * b];
  };
  meta_load(0);

  const i32x8 kz = (i32x8){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                           0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
  i32x8 Cp[2][2];
  for (int b = 0; b < 2; ++b)
    for (int a = 0; a < 2; ++a)
      Cp[b][a] = kz;
  float scp[2] = {0.0f, 0.0f};
  float svp[2][8];
  int lastE = 0;

  const auto sload = [&](int h, int e, float (&sv)[2][8]) {
    if (PT)
      return;
    for (int a2 = 0; a2 < 2; ++a2) {
      const float4 s0 = *reinterpret_cast<const float4*>(sxp + (size_t)e * N +
                                                         16 * (2 * h + a2));
      const float4 s1 = *reinterpret_cast<const float4*>(sxp + (size_t)e * N +
                                                         16 * (2 * h + a2) + 4);
      sv[a2][0] = s0.x;
      sv[a2][1] = s0.y;
      sv[a2][2] = s0.z;
      sv[a2][3] = s0.w;
      sv[a2][4] = s1.x;
      sv[a2][5] = s1.y;
      sv[a2][6] = s1.z;
      sv[a2][7] = s1.w;
    }
  };

  const auto fold = [&](int h, i32x8(&C)[2][2], const float (&sc)[2],
                        const float (&sv)[2][8]) {
#pragma unroll
    for (int a2 = 0; a2 < 2; ++a2) {
      const int a = 2 * h + a2;
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        for (int j = 0; j < 8; ++j) {
          const float cf = __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f);
          if (PT)
            sum[a][b][j] = __builtin_fmaf(sc[b], cf, sum[a][b][j]);
          else
            sum[a][b][j] = __builtin_fmaf(sv[a2][j] * sc[b], cf, sum[a][b][j]);
        }
        asm volatile(""
                     : "+v"(sum[a][b][0]), "+v"(sum[a][b][1]),
                       "+v"(sum[a][b][2]), "+v"(sum[a][b][3]),
                       "+v"(sum[a][b][4]), "+v"(sum[a][b][5]),
                       "+v"(sum[a][b][6]), "+v"(sum[a][b][7]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  };

  const auto wmma = [&](int h, i32x8(&C)[2][2], int e, bool nxt) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const uint4 wq = wc[b][s >> 1];
        const i32x2 wv =
            (s & 1) ? i32x2{(int)wq.z, (int)wq.w} : i32x2{(int)wq.x, (int)wq.y};
#pragma unroll
        for (int a2 = 0; a2 < 2; ++a2) {
          const uint4 tq = tc[a2][s >> 1];
          const i32x2 tv = (s & 1) ? i32x2{(int)tq.z, (int)tq.w}
                                   : i32x2{(int)tq.x, (int)tq.y};
          C[b][a2] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, tv, true, wv, s == 0 ? kz : C[b][a2], false);
        }
      }
      if (s & 1) {
        if (h == 0)
          tload1(1, e, s >> 1);
        else if (nxt)
          tload1(0, e + 1, s >> 1);
        if (h == 1 && nxt)
          wload1(e + 1, s >> 1);
      }
      if (s != 7)
        __builtin_amdgcn_sched_barrier(0);
    }
  };

  const auto epoch = [&](auto nxt_tag, int e) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    float sc[2];
    for (int b = 0; b < 2; ++b)
      sc[b] = __half2float(__ushort_as_half(scn[b]));
    if (nxt)
      meta_load(e + 1);
    float sv0[2][8], sv1[2][8];
    i32x8 C0[2][2], C1[2][2];
    sload(0, e, sv0);
    if (Y) {
      sload(1, e > 0 ? e - 1 : 0, svp);
      fold(1, Cp, scp, svp);
    }
    wmma(0, C0, e, nxt);
    fold(0, C0, sc, sv0);
    sload(1, e, sv1);
    wmma(1, C1, e, nxt);
    if (Y) {
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a)
          Cp[b][a] = C1[b][a];
      scp[0] = sc[0];
      scp[1] = sc[1];
      lastE = e;
    } else {
      fold(1, C1, sc, sv1);
    }
  };
  for (int e = 0; e + 1 < E; ++e)
    epoch(std::true_type{}, e);
  epoch(std::false_type{}, E - 1);
  if (Y) {
    sload(1, lastE, svp);
    fold(1, Cp, scp, svp);
  }

  for (int a = 0; a < 4; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wth * 64 + a * 16 + 8 * hi + j;
      const float r = PT ? ds[(size_t)t * groups128] : 1.0f;
      for (int b = 0; b < 2; ++b)
        out[(size_t)t * M + f0 + wf * 32 + b * 16 + lr] = sum[a][b][j] * r;
    }
}

template<bool PT>
__global__ void __launch_bounds__(256)
    Iu4Gemm11(const uint8_t* __restrict__ pw,
              const unsigned short* __restrict__ psc,
              const uint8_t* __restrict__ qs, const float* __restrict__ dsT,
              const float* __restrict__ ds, float* __restrict__ out, int M,
              int N, int K) {
  if ((threadIdx.x >> 7) & 1)
    Gemm11Body<PT, true>(pw, psc, qs, dsT, ds, out, M, N, K);
  else
    Gemm11Body<PT, false>(pw, psc, qs, dsT, ds, out, M, N, K);
}

__global__ void RelayoutTok(const uint8_t* __restrict__ qs,
                            uint8_t* __restrict__ qsT, int N, int K) {
  const int E = K >> 7;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = (size_t)N * E * 8;
  if (idx >= total)
    return;
  const int t = idx / (E * 8), r = idx % (E * 8), e = r / 8, o = r % 8;
  const int i = o >> 1, hi = o & 1;
  const size_t dst = ((size_t)(t >> 4) * E + e) * 1024 +
                     (hi * 16 + (t & 15)) * 32 + i * 8;
  *reinterpret_cast<uint2*>(qsT + dst) = *reinterpret_cast<const uint2*>(
      qs + (size_t)t * (K / 2) + (size_t)e * 64 + 16 * i + 8 * hi);
}

__global__ void RelayoutW(const uint8_t* __restrict__ pw,
                          uint8_t* __restrict__ pwT, int M, int K) {
  const int E = K >> 7;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = (size_t)E * M * 4;
  if (idx >= total)
    return;
  const int q = idx & 3;
  const size_t er = idx >> 2;
  const int row = er % M, e = er / M;
  const size_t dst =
      (((size_t)e * (M >> 4) + (row >> 4)) * 4 + q) * 256 + (row & 15) * 16;
  *reinterpret_cast<uint4*>(pwT + dst) =
      *reinterpret_cast<const uint4*>(pw + ((size_t)e * M + row) * 64 + 16 * q);
}

template<bool PT, bool Y>
__device__ __forceinline__ void Gemm12Body(
    const uint8_t* __restrict__ pw, const unsigned short* __restrict__ psc,
    const uint8_t* __restrict__ qs, const float* __restrict__ ds,
    const float* __restrict__ dsT, float* __restrict__ out, int M, int N, int K,
    char* lds) {
  using namespace v6;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wf = wave & 3, wth = wave >> 2;
  const int tb = N >> 7;
  const int f0 = (blockIdx.x / tb) * 128, t0 = (blockIdx.x % tb) * 128;
  const int E = K >> 7;
  const int groups128 = K / kGroup;
  const uint32_t qrow = (uint32_t)(K >> 1);
  const int sr = 16 * wave + lr;
  const char* gq = reinterpret_cast<const char*>(qs) +
                   (size_t)((t0 >> 4) + wave) * E * 1024 + lane * 32;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wth * 4) * 1024 + prow * 8;
  const int frow = f0 + wf * 32 + lr;
  const float* gd =
      ds +
      (size_t)(t0 + wth * 64 + 16 * (lr >> 3) + 8 * hi + (lr & 7)) * groups128;

  float sum[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < 8; ++j)
        sum[a][b][j] = 0.0f;

  float sxn = 0.0f;
  i32x2 sa[4];
  const auto stage_load = [&](int e) {
    for (int i = 0; i < 2; ++i) {
      const uint4 v = *reinterpret_cast<const uint4*>(gq + e * 1024 + 16 * i);
      sa[2 * i] = i32x2{(int)v.x, (int)v.y};
      sa[2 * i + 1] = i32x2{(int)v.z, (int)v.w};
    }
    if (!PT && tid < 128)
      sxn = dsT[(size_t)e * N + t0 + tid];
  };
  const auto stage_store = [&](char* slot, char* sreg) {
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<i32x2*>(slot + lds_off + i * 256) = sa[i];
    if (!PT && tid < 128)
      *reinterpret_cast<float*>(sreg + tid * 4) = sxn;
  };
  uint4 wc[2][4];
  for (int b = 0; b < 2; ++b) {
    for (int q = 0; q < 4; ++q)
      wc[b][q] = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const char*>(pw) +
          (((size_t)0 * (M >> 4) + (frow >> 4) + b) * 4 + q) * 256 + lr * 16);
  }
  unsigned short scn[2];
  const auto meta_load = [&](int e) {
    for (int b = 0; b < 2; ++b)
      scn[b] = psc[(size_t)e * M + frow + 16 * b];
  };
  stage_load(0);
  stage_store(lds, lds + 16384);
  meta_load(0);
  __syncthreads();

  i32x8 Cp[2][2];
  for (int b = 0; b < 2; ++b)
    for (int a = 0; a < 2; ++a)
      Cp[b][a] = (i32x8){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                         0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
  float scp[2] = {0.0f, 0.0f};
  const char* sslp = lds + 16384;

  const auto fold = [&](int h, i32x8(&C)[2][2], const float (&sc)[2],
                        const char* ssl) {
#pragma unroll
    for (int a2 = 0; a2 < 2; ++a2) {
      const int a = 2 * h + a2;
      float scf[8];
      if (!PT) {
        const float4 s0 = *reinterpret_cast<const float4*>(
            ssl + (wth * 64 + 16 * a + 8 * hi) * 4);
        const float4 s1 = *reinterpret_cast<const float4*>(
            ssl + (wth * 64 + 16 * a + 8 * hi) * 4 + 16);
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        for (int j = 0; j < 8; ++j) {
          scf[j] = sv[j];

          asm volatile("" : "+v"(scf[j]));
        }
      }
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        for (int j = 0; j < 8; ++j) {
#ifdef EXP10_NOFOLD
          sum[a][b][j] += __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f);
          continue;
#endif
          if (PT)
            sum[a][b][j] = __builtin_fmaf(
                sc[b], __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f),
                sum[a][b][j]);
          else
            sum[a][b][j] = __builtin_fmaf(
                scf[j] * sc[b],
                __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f),
                sum[a][b][j]);
        }
        asm volatile(""
                     : "+v"(sum[a][b][0]), "+v"(sum[a][b][1]),
                       "+v"(sum[a][b][2]), "+v"(sum[a][b][3]),
                       "+v"(sum[a][b][4]), "+v"(sum[a][b][5]),
                       "+v"(sum[a][b][6]), "+v"(sum[a][b][7]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  };

  const auto wmma = [&](int h, i32x8(&C)[2][2], const char* slot, bool reload,
                        int e) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      i32x2 ar[2];
      for (int a2 = 0; a2 < 2; ++a2)
        ar[a2] = *reinterpret_cast<const i32x2*>(slot + aoff +
                                                 (2 * h + a2) * 1024 + s * 128);
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const uint4 wq = wc[b][s >> 1];
        const i32x2 wv =
            (s & 1) ? i32x2{(int)wq.z, (int)wq.w} : i32x2{(int)wq.x, (int)wq.y};
#pragma unroll
        for (int a2 = 0; a2 < 2; ++a2) {
          const i32x8 z =
              (i32x8){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                      0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
          C[b][a2] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, ar[a2], true, wv, s == 0 ? z : C[b][a2], false);
        }
      }
#if defined(EXP10_NOLOAD) || defined(EXP10_NOW)
      if (false) {
#else
      if (reload && (s & 1)) {
#endif
        for (int b = 0; b < 2; ++b) {
          wc[b][s >> 1] = *reinterpret_cast<const uint4*>(
              reinterpret_cast<const char*>(pw) +
              (((size_t)(e + 1) * (M >> 4) + (frow >> 4) + b) * 4 + (s >> 1)) *
                  256 +
              lr * 16);
        }
      }
      if (s != 7)
        __builtin_amdgcn_sched_barrier(0);
    }
  };

  const auto epoch = [&](auto nxt_tag, int e, int par) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    const char* slot = lds + par * 8192;
    float sc[2];
    for (int b = 0; b < 2; ++b)
      sc[b] = __half2float(__ushort_as_half(scn[b]));

    __builtin_amdgcn_sched_barrier(0);
#ifndef EXP10_NOLOAD
    if (nxt) {
#ifndef EXP10_NOTOK
      stage_load(e + 1);
#endif
      meta_load(e + 1);
    }
#endif
    __builtin_amdgcn_sched_barrier(0);
    i32x8 C0[2][2], C1[2][2];
    if (Y)
      fold(1, Cp, scp, sslp);
    wmma(0, C0, slot, false, e);
    fold(0, C0, sc, lds + 16384 + (e & 3) * 512);
    wmma(1, C1, slot, nxt, e);
    if (Y) {
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a)
          Cp[b][a] = C1[b][a];
      scp[0] = sc[0];
      scp[1] = sc[1];
      sslp = lds + 16384 + (e & 3) * 512;
    } else {
      fold(1, C1, sc, lds + 16384 + (e & 3) * 512);
    }
    if (nxt)
      stage_store(lds + (par ^ 1) * 8192, lds + 16384 + ((e + 1) & 3) * 512);
#ifndef EXP12_NOSYNC
    __syncthreads();
#endif
  };
  int e = 0;
  for (; e + 2 < E; e += 2) {
    epoch(std::true_type{}, e, 0);
    epoch(std::true_type{}, e + 1, 1);
  }
  epoch(std::true_type{}, e, 0);
  epoch(std::false_type{}, e + 1, 1);
  if (Y)
    fold(1, Cp, scp, sslp);

  for (int a = 0; a < 4; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wth * 64 + a * 16 + 8 * hi + j;
      const float r = PT ? ds[(size_t)t * groups128] : 1.0f;
      for (int b = 0; b < 2; ++b)
        out[(size_t)t * M + f0 + wf * 32 + b * 16 + lr] = sum[a][b][j] * r;
    }
}

template<bool PT>
__global__ void __launch_bounds__(256)
    Iu4Gemm12(const uint8_t* __restrict__ pw,
              const unsigned short* __restrict__ psc,
              const uint8_t* __restrict__ qs, const float* __restrict__ ds,
              const float* __restrict__ dsT, float* __restrict__ out, int M,
              int N, int K) {
  __shared__ __attribute__((aligned(16))) char lds[16384 + 2048];
  if ((threadIdx.x >> 7) & 1)
    Gemm12Body<PT, true>(pw, psc, qs, ds, dsT, out, M, N, K, lds);
  else
    Gemm12Body<PT, false>(pw, psc, qs, ds, dsT, out, M, N, K, lds);
}

template<bool PT, bool Y>
__device__ __forceinline__ void Gemm13Body(
    const uint8_t* __restrict__ pw, const unsigned short* __restrict__ psc,
    const uint8_t* __restrict__ qs, const float* __restrict__ ds,
    const float* __restrict__ dsT, float* __restrict__ out, int M, int N, int K,
    char* lds) {
  using namespace v6;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wf = wave & 3, wth = wave >> 2;
  const int tb = N >> 7;
  const int f0 = (blockIdx.x / tb) * 128, t0 = (blockIdx.x % tb) * 128;
  const int E = K >> 7;
  const int groups128 = K / kGroup;
  const uint32_t qrow = (uint32_t)(K >> 1);
  const int sr = 16 * wave + lr;
  const char* gq = reinterpret_cast<const char*>(qs) +
                   (size_t)((t0 >> 4) + wave) * E * 1024 + lane * 32;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wth * 4) * 1024 + prow * 8;
  const int frow = f0 + wf * 32 + lr;
  const float* gd =
      ds +
      (size_t)(t0 + wth * 64 + 16 * (lr >> 3) + 8 * hi + (lr & 7)) * groups128;

  float sum[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < 8; ++j)
        sum[a][b][j] = 0.0f;

  float sxn[2] = {0.0f, 0.0f};
  i32x2 sa[2][4];
  const auto stage_load = [&](int e) {
    for (int k = 0; k < 2; ++k) {
      for (int i = 0; i < 2; ++i) {
        const uint4 v = *reinterpret_cast<const uint4*>(
            gq + (e + k) * 1024 + 16 * i);
        sa[k][2 * i] = i32x2{(int)v.x, (int)v.y};
        sa[k][2 * i + 1] = i32x2{(int)v.z, (int)v.w};
      }
      if (!PT && tid < 128)
        sxn[k] = dsT[(size_t)(e + k) * N + t0 + tid];
    }
  };
  const auto stage_store = [&](char* slot, int e) {
    for (int k = 0; k < 2; ++k) {
      for (int i = 0; i < 4; ++i)
        *reinterpret_cast<i32x2*>(slot + k * 8192 + lds_off + i * 256) =
            sa[k][i];
      if (!PT && tid < 128)
        *reinterpret_cast<float*>(lds + 32768 + ((e + k) & 7) * 512 +
                                  tid * 4) = sxn[k];
    }
  };
  uint4 wc[2][4];
  for (int b = 0; b < 2; ++b) {
    for (int q = 0; q < 4; ++q)
      wc[b][q] = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const char*>(pw) +
          (((size_t)0 * (M >> 4) + (frow >> 4) + b) * 4 + q) * 256 + lr * 16);
  }
  unsigned short scn[2];
  const auto meta_load = [&](int e) {
    for (int b = 0; b < 2; ++b)
      scn[b] = psc[(size_t)e * M + frow + 16 * b];
  };
  stage_load(0);
  stage_store(lds, 0);
  meta_load(0);
  __syncthreads();

  i32x8 Cp[2][2];
  for (int b = 0; b < 2; ++b)
    for (int a = 0; a < 2; ++a)
      Cp[b][a] = (i32x8){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                         0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
  float scp[2] = {0.0f, 0.0f};
  const char* sslp = lds + 32768;

  const auto fold = [&](int h, i32x8(&C)[2][2], const float (&sc)[2],
                        const char* ssl) {
#pragma unroll
    for (int a2 = 0; a2 < 2; ++a2) {
      const int a = 2 * h + a2;
      float scf[8];
      if (!PT) {
        const float4 s0 = *reinterpret_cast<const float4*>(
            ssl + (wth * 64 + 16 * a + 8 * hi) * 4);
        const float4 s1 = *reinterpret_cast<const float4*>(
            ssl + (wth * 64 + 16 * a + 8 * hi) * 4 + 16);
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        for (int j = 0; j < 8; ++j) {
          scf[j] = sv[j];

          asm volatile("" : "+v"(scf[j]));
        }
      }
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        for (int j = 0; j < 8; ++j) {
#ifdef EXP10_NOFOLD
          sum[a][b][j] += __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f);
          continue;
#endif
          if (PT)
            sum[a][b][j] = __builtin_fmaf(
                sc[b], __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f),
                sum[a][b][j]);
          else
            sum[a][b][j] = __builtin_fmaf(
                scf[j] * sc[b],
                __fsub_rn(__int_as_float(C[b][a2][j]), 12582912.0f),
                sum[a][b][j]);
        }
        asm volatile(""
                     : "+v"(sum[a][b][0]), "+v"(sum[a][b][1]),
                       "+v"(sum[a][b][2]), "+v"(sum[a][b][3]),
                       "+v"(sum[a][b][4]), "+v"(sum[a][b][5]),
                       "+v"(sum[a][b][6]), "+v"(sum[a][b][7]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  };

  const auto wmma = [&](int h, i32x8(&C)[2][2], const char* slot, bool reload,
                        int e) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      i32x2 ar[2];
      for (int a2 = 0; a2 < 2; ++a2)
        ar[a2] = *reinterpret_cast<const i32x2*>(slot + aoff +
                                                 (2 * h + a2) * 1024 + s * 128);
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const uint4 wq = wc[b][s >> 1];
        const i32x2 wv =
            (s & 1) ? i32x2{(int)wq.z, (int)wq.w} : i32x2{(int)wq.x, (int)wq.y};
#pragma unroll
        for (int a2 = 0; a2 < 2; ++a2) {
          const i32x8 z =
              (i32x8){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                      0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
          C[b][a2] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, ar[a2], true, wv, s == 0 ? z : C[b][a2], false);
        }
      }
#if defined(EXP10_NOLOAD) || defined(EXP10_NOW)
      if (false) {
#else
      if (reload && (s & 1)) {
#endif
        for (int b = 0; b < 2; ++b) {
          wc[b][s >> 1] = *reinterpret_cast<const uint4*>(
              reinterpret_cast<const char*>(pw) +
              (((size_t)(e + 1) * (M >> 4) + (frow >> 4) + b) * 4 + (s >> 1)) *
                  256 +
              lr * 16);
        }
      }
      if (s != 7)
        __builtin_amdgcn_sched_barrier(0);
    }
  };

  const auto sub = [&](auto nxt_tag, int e, const char* slot) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    float sc[2];
    for (int b = 0; b < 2; ++b)
      sc[b] = __half2float(__ushort_as_half(scn[b]));
    __builtin_amdgcn_sched_barrier(0);
    if (nxt)
      meta_load(e + 1);
    __builtin_amdgcn_sched_barrier(0);
    i32x8 C0[2][2], C1[2][2];
    const char* ssl = lds + 32768 + (e & 7) * 512;
    if (Y)
      fold(1, Cp, scp, sslp);
    wmma(0, C0, slot, false, e);
    fold(0, C0, sc, ssl);
    wmma(1, C1, slot, nxt, e);
    if (Y) {
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a)
          Cp[b][a] = C1[b][a];
      scp[0] = sc[0];
      scp[1] = sc[1];
      sslp = ssl;
    } else {
      fold(1, C1, sc, ssl);
    }
  };
  int e = 0;
  for (; e + 2 < E; e += 2) {
    const char* slot = lds + (e & 2) * 8192;
    stage_load(e + 2);
    sub(std::true_type{}, e, slot);
    sub(std::true_type{}, e + 1, slot + 8192);
    stage_store(lds + ((e + 2) & 2) * 8192, e + 2);
    __syncthreads();
  }
  {
    const char* slot = lds + (e & 2) * 8192;
    sub(std::true_type{}, e, slot);
    sub(std::false_type{}, e + 1, slot + 8192);
  }
  if (Y)
    fold(1, Cp, scp, sslp);

  for (int a = 0; a < 4; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wth * 64 + a * 16 + 8 * hi + j;
      const float r = PT ? ds[(size_t)t * groups128] : 1.0f;
      for (int b = 0; b < 2; ++b)
        out[(size_t)t * M + f0 + wf * 32 + b * 16 + lr] = sum[a][b][j] * r;
    }
}


template<bool PT>
__global__ void __launch_bounds__(256)
    Iu4Gemm13(const uint8_t* __restrict__ pw,
              const unsigned short* __restrict__ psc,
              const uint8_t* __restrict__ qs, const float* __restrict__ ds,
              const float* __restrict__ dsT, float* __restrict__ out, int M,
              int N, int K) {
  __shared__ __attribute__((aligned(16))) char lds[32768 + 4096];
  if ((threadIdx.x >> 7) & 1)
    Gemm13Body<PT, true>(pw, psc, qs, ds, dsT, out, M, N, K, lds);
  else
    Gemm13Body<PT, false>(pw, psc, qs, ds, dsT, out, M, N, K, lds);
}

__global__ void XorNibbles(uint8_t* w, size_t bytes) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= bytes)
    return;
  // payload bytes are at block offsets 8..135 of every 136-byte block
  if (i % 136 >= 8)
    w[i] ^= 0x88;
}

// -------------------------------------------------------------------- host
static std::vector<float> Signs(uint32_t seed) {
  std::vector<float> s(256);
  uint32_t st = seed;
  for (auto& x : s) {
    st = (st * 1103515245u + 12345u) & 0x7fffffffu;
    x = ((st >> 16) & 1) ? 1.0f : -1.0f;
  }
  return s;
}

struct Entry {
  std::string name;
  uint8_t qt;
  std::vector<uint32_t> dims;
  uint64_t off, len;
};

static bool ReadIndex(const uint8_t* base, size_t size,
                      std::vector<Entry>* out) {
  uint32_t count;
  uint64_t meta_off, data_off;
  std::memcpy(&count, base + 12, 4);
  std::memcpy(&meta_off, base + 16, 8);
  std::memcpy(&data_off, base + 24, 8);
  // metadata JSON is followed by the index; find the index by scanning for
  // the entry count word that precedes a plausible first name length.
  size_t pos = meta_off;
  int depth = 0;
  bool in_str = false;
  for (; pos < data_off; ++pos) {
    const char c = (char)base[pos];
    if (in_str) {
      if (c == '\\')
        ++pos;
      else if (c == '"')
        in_str = false;
    } else if (c == '"')
      in_str = true;
    else if (c == '{')
      ++depth;
    else if (c == '}' && --depth == 0) {
      ++pos;
      break;
    }
  }
  uint32_t n;
  std::memcpy(&n, base + pos, 4);
  pos += 4;
  uint64_t off = data_off;
  for (uint32_t i = 0; i < n; ++i) {
    Entry e;
    uint16_t nl;
    std::memcpy(&nl, base + pos, 2);
    pos += 2;
    e.name.assign((const char*)base + pos, nl);
    pos += nl;
    e.qt = base[pos++];
    const uint8_t nd = base[pos++];
    e.dims.resize(nd);
    for (int d = 0; d < nd; ++d) {
      std::memcpy(&e.dims[d], base + pos, 4);
      pos += 4;
    }
    pos += 4;  // group size
    std::memcpy(&e.len, base + pos, 8);
    pos += 8;
    e.off = off;
    off += e.len;
    out->push_back(std::move(e));
  }
  return off <= size;
}

static float Half(const uint8_t* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  return __half2float(*reinterpret_cast<__half*>(&h));
}

#ifndef EXP12
#define Iu4GemmX Iu4Gemm13
#else
#define Iu4GemmX Iu4Gemm12
#endif

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s model weight awq N\n", argv[0]);
    return 1;
  }
  const int N = std::atoi(argv[4]);
  int fd = open(argv[1], O_RDONLY);
  struct stat st;
  fstat(fd, &st);
  const uint8_t* base =
      (const uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  std::vector<Entry> index;
  if (!ReadIndex(base, st.st_size, &index)) {
    std::fprintf(stderr, "bad index\n");
    return 1;
  }
  const Entry *we = nullptr, *ae = nullptr;
  for (auto& e : index) {
    if (e.name == argv[2])
      we = &e;
    if (e.name == argv[3])
      ae = &e;
  }
  if (!we || !ae || we->qt != 44) {
    std::fprintf(stderr, "tensor not found / not MQ4\n");
    return 1;
  }
  const int M = we->dims[0], K = we->dims[1];
  std::printf("W %s [%d x %d] N=%d\n", argv[2], M, K, N);
  std::vector<float> awq(K);
  for (int i = 0; i < K; ++i)
    awq[i] = Half(base + ae->off + 2 * i);

  // synthetic activations: gaussian with a few heavy channels
  std::vector<float> x((size_t)N * K);
  uint64_t r = 88172645463325252ull;
  auto rnd = [&]() {
    r ^= r << 13;
    r ^= r >> 7;
    r ^= r << 17;
    return (float)((r >> 11) * (1.0 / 9007199254740992.0));
  };
  for (auto& v : x) {
    float a = rnd() + 1e-9f, b = rnd();
    v = std::sqrt(-2 * std::log(a)) * std::cos(6.2831853f * b) * awq[0];
  }
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < 8; ++c)
      x[(size_t)n * K + (c * 613) % K] *= 12.0f;

  uint8_t *dw, *dqs;
  float *dx, *dawq, *ds1, *ds2, *dds, *dout, *drot;
  int* dss;
  HIP_OK(hipMalloc(&dw, we->len));
  HIP_OK(hipMemcpy(dw, base + we->off, we->len, hipMemcpyHostToDevice));
  HIP_OK(hipMalloc(&dx, x.size() * 4));
  HIP_OK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMalloc(&dawq, K * 4));
  HIP_OK(hipMemcpy(dawq, awq.data(), K * 4, hipMemcpyHostToDevice));
  auto s1 = Signs(42), s2 = Signs(1042);
  HIP_OK(hipMalloc(&ds1, 1024));
  HIP_OK(hipMalloc(&ds2, 1024));
  HIP_OK(hipMemcpy(ds1, s1.data(), 1024, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(ds2, s2.data(), 1024, hipMemcpyHostToDevice));
  HIP_OK(hipMalloc(&dqs, (size_t)N * K / 2));
  HIP_OK(hipMalloc(&dds, (size_t)N * (K / kGroup) * 4));
  HIP_OK(hipMalloc(&dss, (size_t)N * (K / kGroup) * 4));
  HIP_OK(hipMalloc(&drot, x.size() * 4));
  HIP_OK(hipMalloc(&dout, (size_t)N * M * 4));

  const int use_search = argc > 5 ? std::atoi(argv[5]) : 1;
  ProduceInt4<<<N*(K / 256), 256>>>(dx, dawq, ds1, ds2, dqs, dds, dss, drot, K,
                                    use_search);
  HIP_OK(hipDeviceSynchronize());
  dim3 grid(M / 64, N / 64);
  Iu4Gemm<<<grid, 128>>>(dw, dqs, dds, dss, dout, M, N, K);
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  hipEventRecord(a);
  const int reps = 10;
  for (int i = 0; i < reps; ++i)
    Iu4Gemm<<<grid, 128>>>(dw, dqs, dds, dss, dout, M, N, K);
  hipEventRecord(b);
  hipEventSynchronize(b);
  float ms;
  hipEventElapsedTime(&ms, a, b);
  ms /= reps;
  std::printf(
      "gemm %.3f ms  %.1f TOPS  (%.0f tok/s-equivalent for this shape)\n", ms,
      2.0 * M * N * K / (ms * 1e-3) / 1e12, N / (ms * 1e-3));

  {
    float* dout2;
    HIP_OK(hipMalloc(&dout2, (size_t)N * M * 4));
    dim3 grid2(M / kTileF, N / kTileT);
    Iu4Gemm2<<<grid2, 256>>>(dw, dqs, dds, dss, dout2, M, N, K);
    HIP_OK(hipDeviceSynchronize());
    hipEventRecord(a);
    for (int i = 0; i < reps; ++i)
      Iu4Gemm2<<<grid2, 256>>>(dw, dqs, dds, dss, dout2, M, N, K);
    hipEventRecord(b);
    hipEventSynchronize(b);
    float ms2;
    hipEventElapsedTime(&ms2, a, b);
    ms2 /= reps;
    std::vector<float> o1((size_t)N * M), o2((size_t)N * M);
    HIP_OK(hipMemcpy(o1.data(), dout, o1.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(o2.data(), dout2, o2.size() * 4, hipMemcpyDeviceToHost));
    double md = 0, mx = 0;
    for (size_t i = 0; i < o1.size(); ++i) {
      md = std::max(md, (double)std::fabs(o1[i] - o2[i]));
      mx = std::max(mx, (double)std::fabs(o1[i]));
    }
    std::printf("v2   %.3f ms  %.1f TOPS  max |v1-v2| %.3e (max |out| %.3e)\n",
                ms2, 2.0 * M * N * K / (ms2 * 1e-3) / 1e12, md, mx);
    {
      uint8_t* dw3;
      HIP_OK(hipMalloc(&dw3, we->len));
      HIP_OK(hipMemcpy(dw3, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dw3, we->len);
      const int reps3 =
          std::getenv("REPS3") ? std::atoi(std::getenv("REPS3")) : reps;
      float* dout3;
      HIP_OK(hipMalloc(&dout3, (size_t)N * M * 4));
      Iu4Gemm3<<<grid2, 256>>>(dw3, dqs, dds, dout3, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < reps3; ++i)
        Iu4Gemm3<<<grid2, 256>>>(dw3, dqs, dds, dout3, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms3;
      hipEventElapsedTime(&ms3, a, b);
      ms3 /= reps3;
      std::vector<float> o3((size_t)N * M);
      HIP_OK(hipMemcpy(o3.data(), dout3, o3.size() * 4, hipMemcpyDeviceToHost));
      double md3 = 0;
      for (size_t i = 0; i < o3.size(); ++i)
        md3 = std::max(md3, (double)std::fabs(o3[i] - o2[i]));
      std::printf("v3   %.3f ms  %.1f TOPS  max |v2-v3| %.3e\n", ms3,
                  2.0 * M * N * K / (ms3 * 1e-3) / 1e12, md3);
    }
    {
      uint8_t* dw5;
      HIP_OK(hipMalloc(&dw5, we->len));
      HIP_OK(hipMemcpy(dw5, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dw5, we->len);
      float* dout5;
      HIP_OK(hipMalloc(&dout5, (size_t)N * M * 4));
      dim3 g5(M / 128, N / 256);
      Iu4Gemm5<<<g5, 512>>>(dw5, dqs, dds, dout5, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < 1000; ++i)
        Iu4Gemm5<<<g5, 512>>>(dw5, dqs, dds, dout5, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms5;
      hipEventElapsedTime(&ms5, a, b);
      ms5 /= 1000;
      std::vector<float> o5((size_t)N * M);
      HIP_OK(hipMemcpy(o5.data(), dout5, o5.size() * 4, hipMemcpyDeviceToHost));
      double md5 = 0;
      for (size_t i = 0; i < o5.size(); ++i)
        md5 = std::max(md5, (double)std::fabs(o5[i] - o2[i]));
      std::printf("v5   %.3f ms  %.1f TOPS  max |v2-v5| %.3e\n", ms5,
                  2.0 * M * N * K / (ms5 * 1e-3) / 1e12, md5);
    }
    {
      uint8_t* dw6;
      HIP_OK(hipMalloc(&dw6, we->len));
      HIP_OK(hipMemcpy(dw6, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dw6, we->len);
      float* dout6;
      HIP_OK(hipMalloc(&dout6, (size_t)N * M * 4));
      const int g6 = (M / 128) * (N / 128);
#ifdef EXP_PT
      constexpr bool kPT6 = true;
#else
      constexpr bool kPT6 = false;
#endif
      Iu4Gemm6<kPT6><<<g6, 256>>>(dw6, dqs, dds, dout6, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < 1000; ++i)
        Iu4Gemm6<kPT6><<<g6, 256>>>(dw6, dqs, dds, dout6, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms6;
      hipEventElapsedTime(&ms6, a, b);
      ms6 /= 1000;
      std::vector<float> o6((size_t)N * M);
      HIP_OK(hipMemcpy(o6.data(), dout6, o6.size() * 4, hipMemcpyDeviceToHost));
      double md6 = 0;
      for (size_t i = 0; i < o6.size(); ++i)
        md6 = std::max(md6, (double)std::fabs(o6[i] - o2[i]));
      std::printf("v6   %.3f ms  %.1f TOPS  max |v2-v6| %.3e\n", ms6,
                  2.0 * M * N * K / (ms6 * 1e-3) / 1e12, md6);
    }
    {
      uint8_t* dw9;
      HIP_OK(hipMalloc(&dw9, we->len));
      HIP_OK(hipMemcpy(dw9, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dw9, we->len);
      uint8_t* pw9;
      unsigned short* psc9;
      HIP_OK(hipMalloc(&pw9, (size_t)M * K / 2));
      HIP_OK(hipMalloc(&psc9, (size_t)M * (K / 128) * 2));
      Repack9<<<dim3(M, K / 128), 128>>>(dw9, pw9, psc9, M, K);
      float* dout9;
      HIP_OK(hipMalloc(&dout9, (size_t)N * M * 4));
      const int g9 = (M / 128) * (N / 128);
#ifdef EXP_PT
      constexpr bool kPT9 = true;
#else
      constexpr bool kPT9 = false;
#endif
      Iu4Gemm9<kPT9><<<g9, 256>>>(pw9, psc9, dqs, dds, dout9, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < 1000; ++i)
        Iu4Gemm9<kPT9><<<g9, 256>>>(pw9, psc9, dqs, dds, dout9, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms9;
      hipEventElapsedTime(&ms9, a, b);
      ms9 /= 1000;
      std::vector<float> o9((size_t)N * M);
      HIP_OK(hipMemcpy(o9.data(), dout9, o9.size() * 4, hipMemcpyDeviceToHost));
      double md9 = 0;
      for (size_t i = 0; i < o9.size(); ++i)
        md9 = std::max(md9, (double)std::fabs(o9[i] - o2[i]));
      std::printf("v9   %.3f ms  %.1f TOPS  max |v2-v9| %.3e\n", ms9,
                  2.0 * M * N * K / (ms9 * 1e-3) / 1e12, md9);
    }
    {
      uint8_t* dwa;
      HIP_OK(hipMalloc(&dwa, we->len));
      HIP_OK(hipMemcpy(dwa, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dwa, we->len);
      uint8_t* pwa;
      unsigned short* psca;
      HIP_OK(hipMalloc(&pwa, (size_t)M * K / 2));
      HIP_OK(hipMalloc(&psca, (size_t)M * (K / 128) * 2));
      Repack9<<<dim3(M, K / 128), 128>>>(dwa, pwa, psca, M, K);
      float* douta;
      HIP_OK(hipMalloc(&douta, (size_t)N * M * 4));
      const int ga = (M / 128) * (N / 128);
#ifdef EXP_PT
      constexpr bool kPTa = true;
#else
      constexpr bool kPTa = false;
#endif
      Iu4Gemm10<kPTa><<<ga, 256>>>(pwa, psca, dqs, dds, douta, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < 1000; ++i)
        Iu4Gemm10<kPTa><<<ga, 256>>>(pwa, psca, dqs, dds, douta, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float msa;
      hipEventElapsedTime(&msa, a, b);
      msa /= 1000;
      std::vector<float> oa((size_t)N * M);
      HIP_OK(hipMemcpy(oa.data(), douta, oa.size() * 4, hipMemcpyDeviceToHost));
      double mda = 0;
      size_t nbad = 0, ex = 0;
      for (size_t i = 0; i < oa.size(); ++i) {
        const double d = std::fabs(oa[i] - o2[i]);
        mda = std::max(mda, d);
        if (d > 1e-4) {
          if (nbad < 6)
            std::printf("bad n=%zu m=%zu got %g want %g\n", i / M, i % M, oa[i],
                        o2[i]);
          ++nbad;
        }
      }
      std::printf("bad count %zu of %zu\n", nbad, oa.size());
      (void)ex;
      std::printf("v10   %.3f ms  %.1f TOPS  max |v2-v10| %.3e\n", msa,
                  2.0 * M * N * K / (msa * 1e-3) / 1e12, mda);
    }
    {
      uint8_t* dwc;
      HIP_OK(hipMalloc(&dwc, we->len));
      HIP_OK(hipMemcpy(dwc, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dwc, we->len);
      uint8_t* pwc;
      unsigned short* pscc;
      HIP_OK(hipMalloc(&pwc, (size_t)M * K / 2));
      HIP_OK(hipMalloc(&pscc, (size_t)M * (K / 128) * 2));
      Repack9<<<dim3(M, K / 128), 128>>>(dwc, pwc, pscc, M, K);
      uint8_t *pwT, *qsT;
      HIP_OK(hipMalloc(&pwT, (size_t)M * K / 2));
      HIP_OK(hipMalloc(&qsT, (size_t)N * K / 2));
      RelayoutW<<<((size_t)(K / 128) * M * 4 + 255) / 256, 256>>>(pwc, pwT, M,
                                                                  K);
      RelayoutTok<<<((size_t)N * (K / 128) * 8 + 255) / 256, 256>>>(dqs, qsT, N,
                                                                    K);
      float* dsTc;
      HIP_OK(hipMalloc(&dsTc, (size_t)N * (K / 128) * 4));
      TransposeScales<<<(N + 255) / 256, 256>>>(dds, dsTc, N, K / 128);
      float* doutc;
      HIP_OK(hipMalloc(&doutc, (size_t)N * M * 4));
      const int gc = (M / 128) * (N / 128);
#ifdef EXP_PT
      constexpr bool kPTc = true;
#else
      constexpr bool kPTc = false;
#endif
      Iu4GemmX<kPTc><<<gc, 256>>>(pwT, pscc, qsT, dds, dsTc, doutc, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < 1000; ++i)
        Iu4GemmX<kPTc><<<gc, 256>>>(pwT, pscc, qsT, dds, dsTc, doutc, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float msc;
      hipEventElapsedTime(&msc, a, b);
      msc /= 1000;
      std::vector<float> oc((size_t)N * M);
      HIP_OK(hipMemcpy(oc.data(), doutc, oc.size() * 4, hipMemcpyDeviceToHost));
      double mdc = 0;
      size_t nbad = 0, ex = 0;
      for (size_t i = 0; i < oc.size(); ++i) {
        const double d = std::fabs(oc[i] - o2[i]);
        mdc = std::max(mdc, d);
        if (d > 1e-4) {
          if (nbad < 6)
            std::printf("bad n=%zu m=%zu got %g want %g\n", i / M, i % M, oc[i],
                        o2[i]);
          ++nbad;
        }
      }
      std::printf("bad count %zu of %zu\n", nbad, oc.size());
      (void)ex;
      std::printf("vX    %.3f ms  %.1f TOPS  max |v2-v12| %.3e\n", msc,
                  2.0 * M * N * K / (msc * 1e-3) / 1e12, mdc);
    }
    {
      uint8_t* dwb;
      HIP_OK(hipMalloc(&dwb, we->len));
      HIP_OK(hipMemcpy(dwb, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dwb, we->len);
      uint8_t* pwb;
      unsigned short* pscb;
      HIP_OK(hipMalloc(&pwb, (size_t)M * K / 2));
      HIP_OK(hipMalloc(&pscb, (size_t)M * (K / 128) * 2));
      Repack9<<<dim3(M, K / 128), 128>>>(dwb, pwb, pscb, M, K);
      float* dsTb;
      HIP_OK(hipMalloc(&dsTb, (size_t)N * (K / 128) * 4));
      TransposeScales<<<(N + 255) / 256, 256>>>(dds, dsTb, N, K / 128);
      float* doutb;
      HIP_OK(hipMalloc(&doutb, (size_t)N * M * 4));
      const int gb = (M / 128) * (N / 128);
#ifdef EXP_PT
      constexpr bool kPTb = true;
#else
      constexpr bool kPTb = false;
#endif
      Iu4Gemm11<kPTb><<<gb, 256>>>(pwb, pscb, dqs, dsTb, dds, doutb, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < 1000; ++i)
        Iu4Gemm11<kPTb><<<gb, 256>>>(pwb, pscb, dqs, dsTb, dds, doutb, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float msb;
      hipEventElapsedTime(&msb, a, b);
      msb /= 1000;
      std::vector<float> ob((size_t)N * M);
      HIP_OK(hipMemcpy(ob.data(), doutb, ob.size() * 4, hipMemcpyDeviceToHost));
      double mdb = 0;
      size_t nbad = 0, ex = 0;
      for (size_t i = 0; i < ob.size(); ++i) {
        const double d = std::fabs(ob[i] - o2[i]);
        mdb = std::max(mdb, d);
        if (d > 1e-4) {
          if (nbad < 6)
            std::printf("bad n=%zu m=%zu got %g want %g\n", i / M, i % M, ob[i],
                        o2[i]);
          ++nbad;
        }
      }
      std::printf("bad count %zu of %zu\n", nbad, ob.size());
      (void)ex;
      std::printf("v11   %.3f ms  %.1f TOPS  max |v2-v11| %.3e\n", msb,
                  2.0 * M * N * K / (msb * 1e-3) / 1e12, mdb);
    }
    {
      uint8_t* dw8;
      HIP_OK(hipMalloc(&dw8, we->len));
      HIP_OK(hipMemcpy(dw8, dw, we->len, hipMemcpyDeviceToDevice));
      XorNibbles<<<(we->len + 255) / 256, 256>>>(dw8, we->len);
      float* dout8;
      HIP_OK(hipMalloc(&dout8, (size_t)N * M * 4));
      const int g8 = (M / 128) * (N / 256);
#ifdef EXP_PT
      constexpr bool kPT8 = true;
#else
      constexpr bool kPT8 = false;
#endif
      Iu4Gemm8<kPT8><<<g8, 512>>>(dw8, dqs, dds, dout8, M, N, K);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < 1000; ++i)
        Iu4Gemm8<kPT8><<<g8, 512>>>(dw8, dqs, dds, dout8, M, N, K);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms8;
      hipEventElapsedTime(&ms8, a, b);
      ms8 /= 1000;
      std::vector<float> o8((size_t)N * M);
      HIP_OK(hipMemcpy(o8.data(), dout8, o8.size() * 4, hipMemcpyDeviceToHost));
      double md8 = 0;
      for (size_t i = 0; i < o8.size(); ++i)
        md8 = std::max(md8, (double)std::fabs(o8[i] - o2[i]));
      std::printf("v8   %.3f ms  %.1f TOPS  max |v2-v8| %.3e\n", ms8,
                  2.0 * M * N * K / (ms8 * 1e-3) / 1e12, md8);
    }
    hipEventRecord(a);
    for (int i = 0; i < reps; ++i)
      ProduceInt4<<<N*(K / 256), 256>>>(dx, dawq, ds1, ds2, dqs, dds, dss,
                                        nullptr, K, use_search);
    hipEventRecord(b);
    hipEventSynchronize(b);
    float msp;
    hipEventElapsedTime(&msp, a, b);
    std::printf("producer v1 %.3f ms for %d x %d\n", msp / reps, N, K);
    {
      uint8_t* qs2;
      float* ds2b;
      int* ss2;
      HIP_OK(hipMalloc(&qs2, (size_t)N * K / 2));
      HIP_OK(hipMalloc(&ds2b, (size_t)N * (K / kGroup) * 4));
      HIP_OK(hipMalloc(&ss2, (size_t)N * (K / kGroup) * 4));
      const int tg = N * (K / 256);
      ProduceInt4Wave<<<(tg + 7) / 8, 256>>>(dx, dawq, ds1, ds2, qs2, ds2b, ss2,
                                             K, tg, use_search);
      HIP_OK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int i = 0; i < reps; ++i)
        ProduceInt4Wave<<<(tg + 7) / 8, 256>>>(dx, dawq, ds1, ds2, qs2, ds2b,
                                               ss2, K, tg, use_search);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float msw;
      hipEventElapsedTime(&msw, a, b);
      std::vector<uint8_t> qa((size_t)N * K / 2), qb((size_t)N * K / 2);
      HIP_OK(hipMemcpy(qa.data(), dqs, qa.size(), hipMemcpyDeviceToHost));
      HIP_OK(hipMemcpy(qb.data(), qs2, qb.size(), hipMemcpyDeviceToHost));
      size_t diff = 0;
      for (size_t i = 0; i < qa.size(); ++i)
        diff += qa[i] != qb[i];
      std::printf(
          "producer wave %.3f ms; code bytes differing from v1: %.4f%%\n",
          msw / reps, 100.0 * diff / qa.size());
    }
  }
  // verification against a CPU model of the same arithmetic
  std::vector<float> out((size_t)N * M), rot((size_t)N * K),
      dsv((size_t)N * K / kGroup);
  std::vector<int> ssv((size_t)N * K / kGroup);
  std::vector<uint8_t> qsv((size_t)N * K / 2);
  HIP_OK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(rot.data(), drot, rot.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(dsv.data(), dds, dsv.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(ssv.data(), dss, ssv.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(qsv.data(), dqs, qsv.size(), hipMemcpyDeviceToHost));
  double max_rel_kernel = 0, sum_a4 = 0, sum_ref = 0, sum_pt = 0;
  std::vector<int> ptq(K);
  int checked = 0;
  for (int n = 0; n < N; n += std::max(1, N / 7)) {
    float ptmax = 0;
    for (int i = 0; i < K; ++i)
      ptmax = std::max(ptmax, std::fabs(rot[(size_t)n * K + i]));
    const float pts = ptmax / 7.0f;
    for (int i = 0; i < K; ++i)
      ptq[i] = std::max(
          -8, std::min(7, (int)std::lrintf(rot[(size_t)n * K + i] / pts)));
    for (int m = 0; m < M; m += std::max(1, M / 61)) {
      double ptd = 0;
      const uint8_t* wr = base + we->off + (size_t)m * (K / 256) * kBlockBytes;
      double kernel_ref = 0, a16 = 0;
      for (int g = 0; g < K / kGroup; ++g) {
        const uint8_t* blk = wr + (size_t)(g >> 1) * kBlockBytes;
        const float sc = Half(blk + 4 * (g & 1)),
                    zp = Half(blk + 4 * (g & 1) + 2);
        const uint8_t* pay = blk + 8 + 64 * (g & 1);
        long dot = 0;
        for (int j = 0; j < 64; ++j) {
          const int wlo = pay[j] & 15, whi = pay[j] >> 4;
          const uint8_t xb = qsv[(size_t)n * (K / 2) + (size_t)g * 64 + j];
          int xlo = xb & 15, xhi = xb >> 4;
          if (xlo > 7)
            xlo -= 16;
          if (xhi > 7)
            xhi -= 16;
          dot += wlo * xlo + whi * xhi;
          ptd += (double)(sc * wlo + zp) * ptq[g * kGroup + 2 * j] * pts +
                 (double)(sc * whi + zp) * ptq[g * kGroup + 2 * j + 1] * pts;
          a16 += (double)(sc * wlo + zp) *
                     rot[(size_t)n * K + g * kGroup + 2 * j] +
                 (double)(sc * whi + zp) *
                     rot[(size_t)n * K + g * kGroup + 2 * j + 1];
        }
        kernel_ref +=
            (double)dsv[(size_t)n * (K / kGroup) + g] *
            (sc * (double)dot + zp * (double)ssv[(size_t)n * (K / kGroup) + g]);
      }
      const double got = out[(size_t)n * M + m];
      max_rel_kernel =
          std::max(max_rel_kernel, std::fabs(got - kernel_ref) /
                                       (std::fabs(kernel_ref) + 1e-3));
      sum_a4 += (kernel_ref - a16) * (kernel_ref - a16);
      sum_pt += (ptd - a16) * (ptd - a16);
      sum_ref += a16 * a16;
      ++checked;
    }
  }
  std::printf("checked %d outputs: kernel vs cpu-model max rel err %.2e\n",
              checked, max_rel_kernel);
  std::printf(
      "int4-activation error vs fp32-activation (same weights): rel RMS %.4f\n",
      std::sqrt(sum_a4 / sum_ref));
  std::printf("per-token-scale int4 activation error: rel RMS %.4f\n",
              std::sqrt(sum_pt / sum_ref));
  return 0;
}
