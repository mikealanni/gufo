#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/ops/iu4.hpp"

namespace gufo::hip {
namespace {

__global__ void Iu4RepackKernel(const std::uint8_t* __restrict__ blocks,
                                std::uint8_t* __restrict__ tiles,
                                std::uint16_t* __restrict__ scales,
                                unsigned long long* __restrict__ bad,
                                std::uint32_t rows, std::uint32_t groups) {
  const std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = (std::size_t)groups * rows * 4;
  if (idx >= total)
    return;
  const std::uint32_t q = idx & 3;
  const std::size_t er = idx >> 2;
  const std::uint32_t row = er % rows;
  const std::uint32_t e = er / rows;
  const std::uint8_t* src =
      blocks + ((std::size_t)row * (groups >> 1) + (e >> 1)) * kIu4BlockBytes;
  const std::uint32_t half = e & 1;
  if (q == 0) {
    std::uint16_t sc, zp;
    __builtin_memcpy(&sc, src + 4 * half, 2);
    __builtin_memcpy(&zp, src + 4 * half + 2, 2);
    scales[(std::size_t)e * rows + row] = sc;
    const float fs = __half2float(__ushort_as_half(sc));
    const float fz = __half2float(__ushort_as_half(zp));
    if (fabsf(fz + 8.0f * fs) > 1e-3f * fabsf(8.0f * fs) + 1e-12f)
      atomicAdd(bad, 1ull);
  }
  uint4 v;
  __builtin_memcpy(&v, src + 8 + 64 * half + 16 * q, 16);
  v.x ^= 0x88888888u;
  v.y ^= 0x88888888u;
  v.z ^= 0x88888888u;
  v.w ^= 0x88888888u;
  const std::size_t dst =
      (((std::size_t)e * (rows >> 4) + (row >> 4)) * 4 + q) * 256 +
      (row & 15) * 16;
  __builtin_memcpy(tiles + dst, &v, 16);
}

__device__ __forceinline__ float Reduce16Max(float v) {
  for (int o = 8; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

// One workgroup per token row, one wave per 256-wide group of the row: lane l
// owns elements 8l..8l+7. Hadamard strides 1, 2, 4 are register butterflies,
// strides 8 to 128 are lane-xor butterflies.
__global__ void __launch_bounds__(1024) Iu4NormQuantizeKernel(
    const float* __restrict__ x, const float* __restrict__ norm_weight,
    const float* __restrict__ awq, const float* __restrict__ signs, float eps,
    std::uint32_t n, std::uint32_t padded_n, std::uint32_t k,
    std::uint8_t* __restrict__ tiles, float* __restrict__ scales) {
  __shared__ float partial[32];
  const std::uint32_t token = blockIdx.x;
  const std::uint32_t wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t groups256 = k >> 8, e_groups = k >> 7;
  const bool live = token < n;
  float v[8];
  float xin[8];
  float sumsq = 0.0f;
  if (live) {
    const std::size_t base = (std::size_t)token * k + wave * 256 + lane * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(x + base);
    const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
    xin[0] = x0.x;
    xin[1] = x0.y;
    xin[2] = x0.z;
    xin[3] = x0.w;
    xin[4] = x1.x;
    xin[5] = x1.y;
    xin[6] = x1.z;
    xin[7] = x1.w;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sumsq += xin[i] * xin[i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    sumsq += __shfl_xor(sumsq, o);
  if (lane == 0)
    partial[wave] = sumsq;
  __syncthreads();
  float total = 0.0f;
  for (std::uint32_t w = 0; w < groups256; ++w)
    total += partial[w];
  const float inv_rms = rsqrtf(total / (float)k + eps);
  if (live) {
    const std::size_t kk = (std::size_t)wave * 256 + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float w = xin[i] * inv_rms * norm_weight[kk + i];
      if (awq != nullptr)
        w /= awq[kk + i];
      v[i] = w * signs[lane * 8 + i];
    }
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
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] *= 0.0625f * signs[256 + lane * 8 + i];
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = 0.0f;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  amax = Reduce16Max(amax);
  const float scale = amax / 7.0f;
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  std::uint32_t word = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = (int)fminf(fmaxf(rintf(v[i] * inv), -8.0f), 7.0f);
    word |= (std::uint32_t)(q & 0xF) << (4 * i);
  }
  const std::uint32_t e = wave * 2 + (lane >> 4);
  const std::uint32_t l16 = lane & 15;
  const std::uint32_t chunk = l16 >> 1, j = l16 & 1;
  const std::size_t dst = ((std::size_t)(token >> 4) * e_groups + e) * 1024 +
                          ((chunk & 1) * 16 + (token & 15)) * 32 +
                          (chunk >> 1) * 8 + 4 * j;
  *reinterpret_cast<std::uint32_t*>(tiles + dst) = word;
  if (l16 == 0)
    scales[(std::size_t)e * padded_n + token] = scale;
}

}  // namespace

void FillIu4RotationSigns(float* out) {
  const std::uint32_t seeds[2] = {42u, 1042u};
  for (int s = 0; s < 2; ++s) {
    std::uint32_t state = seeds[s];
    for (int i = 0; i < 256; ++i) {
      state = (state * 1103515245u + 12345u) & 0x7fffffffu;
      out[256 * s + i] = ((state >> 16) & 1) ? 1.0f : -1.0f;
    }
  }
}

void LaunchIu4NormQuantize(const float* x, const float* norm_weight,
                           const float* awq, const float* signs, float eps,
                           std::size_t n, std::size_t padded_n, std::size_t k,
                           std::uint8_t* tiles, float* scales,
                           hipStream_t stream) {
  Iu4NormQuantizeKernel<<<static_cast<unsigned>(padded_n),
                          static_cast<unsigned>(32 * (k >> 8)), 0, stream>>>(
      x, norm_weight, awq, signs, eps, static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(padded_n), static_cast<std::uint32_t>(k),
      tiles, scales);
}

void LaunchIu4RepackWeights(const void* blocks, void* tiles,
                            std::uint16_t* scales,
                            unsigned long long* bad_zero_points,
                            std::size_t rows, std::size_t cols,
                            hipStream_t stream) {
  const std::uint32_t groups = static_cast<std::uint32_t>(cols >> 7);
  const std::size_t total = (std::size_t)groups * rows * 4;
  Iu4RepackKernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(blocks),
      static_cast<std::uint8_t*>(tiles), scales, bad_zero_points,
      static_cast<std::uint32_t>(rows), groups);
}

}  // namespace gufo::hip
