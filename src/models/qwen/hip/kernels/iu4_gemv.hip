// Single-token GEMV over the same tile-major int4 weights the prefill GEMM
// reads. The activation is the rotated fp32 vector (rotation and awq division
// folded in by the producers), so decode and prefill share one weight copy.
// One workgroup owns a 16-row tile; its eight waves split the 128-wide
// K groups and combine through LDS. A lane owns one row and two of the four
// 16-byte chunks of a group (q and q + 2), which keeps every wave read a
// contiguous 1 KB.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/ops/iu4.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kGemvWaves = 8;

template<int T>
__device__ __forceinline__ void ChunkDot(const uint4 w,
                                         const float* __restrict__ x,
                                         std::size_t token_stride,
                                         std::uint32_t tokens,
                                         float (&acc)[T]) {
  const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
  if constexpr (T == 1) {
    // Single token: no per-token index math, one serial accumulation chain.
    float a0 = acc[0];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4 a = *reinterpret_cast<const float4*>(x + 8 * j);
      const float4 b = *reinterpret_cast<const float4*>(x + 8 * j + 4);
      const std::uint32_t v = words[j];
      a0 += static_cast<float>(static_cast<int>(v << 28) >> 28) * a.x;
      a0 += static_cast<float>(static_cast<int>(v << 24) >> 28) * a.y;
      a0 += static_cast<float>(static_cast<int>(v << 20) >> 28) * a.z;
      a0 += static_cast<float>(static_cast<int>(v << 16) >> 28) * a.w;
      a0 += static_cast<float>(static_cast<int>(v << 12) >> 28) * b.x;
      a0 += static_cast<float>(static_cast<int>(v << 8) >> 28) * b.y;
      a0 += static_cast<float>(static_cast<int>(v << 4) >> 28) * b.z;
      a0 += static_cast<float>(static_cast<int>(v) >> 28) * b.w;
    }
    acc[0] = a0;
    (void)token_stride;
    (void)tokens;
    return;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const std::uint32_t v = words[j];
    float c[8];
    c[0] = static_cast<float>(static_cast<int>(v << 28) >> 28);
    c[1] = static_cast<float>(static_cast<int>(v << 24) >> 28);
    c[2] = static_cast<float>(static_cast<int>(v << 20) >> 28);
    c[3] = static_cast<float>(static_cast<int>(v << 16) >> 28);
    c[4] = static_cast<float>(static_cast<int>(v << 12) >> 28);
    c[5] = static_cast<float>(static_cast<int>(v << 8) >> 28);
    c[6] = static_cast<float>(static_cast<int>(v << 4) >> 28);
    c[7] = static_cast<float>(static_cast<int>(v) >> 28);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const std::uint32_t tt = t < static_cast<int>(tokens) ? t : tokens - 1;
      const float* xt = x + tt * token_stride + 8 * j;
      const float4 a = *reinterpret_cast<const float4*>(xt);
      const float4 b = *reinterpret_cast<const float4*>(xt + 4);
      acc[t] += c[0] * a.x + c[1] * a.y + c[2] * a.z + c[3] * a.w +
                c[4] * b.x + c[5] * b.y + c[6] * b.z + c[7] * b.w;
    }
  }
}

template<int T>
__global__ void __launch_bounds__(256) Iu4GemvKernel(
    const std::uint8_t* __restrict__ tiles,
    const std::uint16_t* __restrict__ scales, const float* __restrict__ x,
    float* __restrict__ out, bool accumulate, std::uint32_t padded_rows,
    std::uint32_t valid_rows, std::uint32_t groups, std::uint32_t k,
    std::uint32_t tokens) {
  __shared__ float partial[kGemvWaves][T][16];
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t r = lane & 15, qh = lane >> 4;
  const std::uint32_t row_tile = blockIdx.x;
  const std::uint32_t row = row_tile * 16 + r;
  const std::uint32_t tiles_per_group = padded_rows >> 4;
  float acc[T];
#pragma unroll
  for (int t = 0; t < T; ++t)
    acc[t] = 0.0f;
  for (std::uint32_t e = wave; e < groups; e += kGemvWaves) {
    const std::uint8_t* base =
        tiles + ((std::size_t)e * tiles_per_group + row_tile) * 1024 + r * 16;
    const uint4 w0 = *reinterpret_cast<const uint4*>(base + qh * 256);
    const uint4 w1 = *reinterpret_cast<const uint4*>(base + (qh + 2) * 256);
    const float sc = __half2float(
        __ushort_as_half(scales[(std::size_t)e * padded_rows + row]));
    const float* xe = x + (std::size_t)e * 128 + qh * 32;
    float s[T];
#pragma unroll
    for (int t = 0; t < T; ++t)
      s[t] = 0.0f;
    ChunkDot<T>(w0, xe, k, tokens, s);
    ChunkDot<T>(w1, xe + 64, k, tokens, s);
#pragma unroll
    for (int t = 0; t < T; ++t)
      acc[t] += sc * s[t];
  }
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float total = acc[t] + __shfl_xor(acc[t], 16);
    if (qh == 0)
      partial[wave][t][r] = total;
  }
  __syncthreads();
  if (threadIdx.x < 16 * T) {
    const std::uint32_t t = threadIdx.x >> 4, rr = threadIdx.x & 15;
    float total = 0.0f;
#pragma unroll
    for (std::uint32_t w = 0; w < kGemvWaves; ++w)
      total += partial[w][t][rr];
    const std::uint32_t orow = row_tile * 16 + rr;
    if (t < tokens && orow < valid_rows) {
      float* dst = out + (std::size_t)t * valid_rows + orow;
      *dst = accumulate ? *dst + total : total;
    }
  }
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v2h = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

// Activations converted for the matrix cores: fp16, with the eight values of
// every 8-wide word in nibble-pair order (0 4 1 5 2 6 3 7) so the weights'
// nibbles can be expanded without a shuffle. The contraction is a plain sum,
// so only the weights and activations must agree on the order.
__global__ void Iu4PrepareX16Kernel(const float* __restrict__ x,
                                    __half* __restrict__ x16,
                                    std::size_t words) {
  const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= words)
    return;
  const float4 a = *reinterpret_cast<const float4*>(x + 8 * i);
  const float4 b = *reinterpret_cast<const float4*>(x + 8 * i + 4);
  const __half2 p0 = __floats2half2_rn(a.x, b.x);
  const __half2 p1 = __floats2half2_rn(a.y, b.y);
  const __half2 p2 = __floats2half2_rn(a.z, b.z);
  const __half2 p3 = __floats2half2_rn(a.w, b.w);
  uint4 packed;
  packed.x = *reinterpret_cast<const std::uint32_t*>(&p0);
  packed.y = *reinterpret_cast<const std::uint32_t*>(&p1);
  packed.z = *reinterpret_cast<const std::uint32_t*>(&p2);
  packed.w = *reinterpret_cast<const std::uint32_t*>(&p3);
  *reinterpret_cast<uint4*>(x16 + 8 * i) = packed;
}

// One word of eight nibbles (stored as two's complement codes) as eight
// fp16 codes in the pair order above: n ^ 8 is the unsigned nibble, 0x6400 | n
// is 1024 + n, and subtracting 1032 leaves the exact signed code.
__device__ __forceinline__ void ExpandWord(std::uint32_t w, v2h (&out)[4]) {
  w ^= 0x88888888u;
  const v2h bias = {static_cast<_Float16>(1032.0f),
                    static_cast<_Float16>(1032.0f)};
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const std::uint32_t bits = ((w >> (4 * p)) & 0x000F000Fu) | 0x64006400u;
    v2h v;
    __builtin_memcpy(&v, &bits, 4);
    out[p] = v - bias;
  }
}

// Up to 16 tokens per pass on the matrix cores: A is the dequantized weights
// of a 16-row tile (a lane's row), B the fp16 activations (a lane's token).
// A workgroup owns four row tiles; two sets of four waves split the K groups.
// Per group the eight WMMAs accumulate from zero and the per-row group scale is
// applied to the accumulator.
__global__ void __launch_bounds__(256) Iu4GemvWmmaKernel(
    const std::uint8_t* __restrict__ tiles,
    const std::uint16_t* __restrict__ scales, const __half* __restrict__ x16,
    float* __restrict__ out, bool accumulate, std::uint32_t padded_rows,
    std::uint32_t valid_rows, std::uint32_t groups, std::uint32_t k,
    std::uint32_t tokens) {
  __shared__ float partial[4][32][8];
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t rt = wave & 3, ks = wave >> 2;
  const std::uint32_t row_tile = blockIdx.x * 4 + rt;
  const std::uint32_t sub = lane & 15, half_id = lane >> 4;
  const std::uint32_t tiles_per_group = padded_rows >> 4;
  const std::uint32_t token = sub < tokens ? sub : tokens - 1;
  v8f total = {};
  if (row_tile * 16 < padded_rows) {
    for (std::uint32_t e = ks; e < groups; e += 2) {
      const std::uint8_t* wbase =
          tiles + ((std::size_t)e * tiles_per_group + row_tile) * 1024 +
          sub * 16;
      const __half* xbase = x16 + (std::size_t)token * k + e * 128;
      uint4 wq[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        wq[q] = *reinterpret_cast<const uint4*>(wbase + q * 256);
      v8f acc = {};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const std::uint32_t words[4] = {wq[q].x, wq[q].y, wq[q].z, wq[q].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          v2h a0[4], a1[4];
          ExpandWord(words[2 * h], a0);
          ExpandWord(words[2 * h + 1], a1);
          union {
            v16h v;
            v2h p[8];
          } a;
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            a.p[p] = a0[p];
            a.p[4 + p] = a1[p];
          }
          union {
            v16h v;
            uint4 u[2];
          } b;
          const __half* xp = xbase + q * 32 + h * 16;
          b.u[0] = *reinterpret_cast<const uint4*>(xp);
          b.u[1] = *reinterpret_cast<const uint4*>(xp + 8);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.v, b.v, acc);
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float sc = __half2float(__ushort_as_half(
            scales[(std::size_t)e * padded_rows + row_tile * 16 + 2 * i +
                   half_id]));
        total[i] += sc * acc[i];
      }
    }
  }
  if (ks == 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      partial[rt][lane][i] = total[i];
  }
  __syncthreads();
  if (ks == 0 && sub < tokens) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float v = total[i] + partial[rt][lane][i];
      const std::uint32_t orow = row_tile * 16 + 2 * i + half_id;
      if (orow < valid_rows) {
        float* dst = out + (std::size_t)sub * valid_rows + orow;
        *dst = accumulate ? *dst + v : v;
      }
    }
  }
}

template<int T>
void Launch(const void* tiles, const std::uint16_t* weight_scales,
            const float* x, float* out, bool accumulate,
            std::size_t padded_rows, std::size_t valid_rows, std::size_t k,
            std::size_t tokens, hipStream_t stream) {
  Iu4GemvKernel<T><<<dim3(static_cast<unsigned>(padded_rows / 16)), 256, 0,
                     stream>>>(static_cast<const std::uint8_t*>(tiles),
                               weight_scales, x, out, accumulate,
                               static_cast<std::uint32_t>(padded_rows),
                               static_cast<std::uint32_t>(valid_rows),
                               static_cast<std::uint32_t>(k / 128),
                               static_cast<std::uint32_t>(k),
                               static_cast<std::uint32_t>(tokens));
}

}  // namespace

void LaunchIu4PrepareX16(const float* x, void* x16, std::size_t tokens,
                         std::size_t k, hipStream_t stream) {
  const std::size_t words = tokens * k / 8;
  Iu4PrepareX16Kernel<<<static_cast<unsigned>((words + 255) / 256), 256, 0,
                        stream>>>(x, static_cast<__half*>(x16), words);
}

void LaunchIu4Gemv(const void* tiles, const std::uint16_t* weight_scales,
                   const float* x, float* out, bool accumulate,
                   std::size_t padded_rows, std::size_t valid_rows,
                   std::size_t k, hipStream_t stream, std::size_t tokens,
                   void* x16, bool x16_ready) {
  // Chunks of up to 16 tokens; each chunk reads the weights once.
  for (std::size_t first = 0; first < tokens; first += 16) {
    const std::size_t n = tokens - first < 16 ? tokens - first : 16;
    const float* xc = x + first * k;
    float* oc = out + first * valid_rows;
    if (n > 4 && x16 != nullptr) {
      if (!(x16_ready && tokens <= 16)) {
        const std::size_t words = n * k / 8;
        Iu4PrepareX16Kernel<<<static_cast<unsigned>((words + 255) / 256), 256,
                              0, stream>>>(xc, static_cast<__half*>(x16),
                                           words);
      }
      Iu4GemvWmmaKernel<<<dim3(static_cast<unsigned>((padded_rows / 16 + 3) /
                                                     4)),
                          256, 0, stream>>>(
          static_cast<const std::uint8_t*>(tiles), weight_scales,
          static_cast<const __half*>(x16), oc, accumulate,
          static_cast<std::uint32_t>(padded_rows),
          static_cast<std::uint32_t>(valid_rows),
          static_cast<std::uint32_t>(k / 128), static_cast<std::uint32_t>(k),
          static_cast<std::uint32_t>(n));
      continue;
    }
    if (n == 1)
      Launch<1>(tiles, weight_scales, xc, oc, accumulate, padded_rows,
                valid_rows, k, n, stream);
    else if (n == 2)
      Launch<2>(tiles, weight_scales, xc, oc, accumulate, padded_rows,
                valid_rows, k, n, stream);
    else if (n <= 4)
      Launch<4>(tiles, weight_scales, xc, oc, accumulate, padded_rows,
                valid_rows, k, n, stream);
    else if (n <= 8)
      Launch<8>(tiles, weight_scales, xc, oc, accumulate, padded_rows,
                valid_rows, k, n, stream);
    else
      Launch<16>(tiles, weight_scales, xc, oc, accumulate, padded_rows,
                 valid_rows, k, n, stream);
  }
}

}  // namespace gufo::hip
