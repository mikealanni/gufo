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

__device__ __forceinline__ float ChunkDot(const uint4 w,
                                          const float* __restrict__ x) {
  float acc = 0.0f;
  const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float4 a = *reinterpret_cast<const float4*>(x + 8 * j);
    const float4 b = *reinterpret_cast<const float4*>(x + 8 * j + 4);
    const std::uint32_t v = words[j];
    acc += static_cast<float>(static_cast<int>(v << 28) >> 28) * a.x;
    acc += static_cast<float>(static_cast<int>(v << 24) >> 28) * a.y;
    acc += static_cast<float>(static_cast<int>(v << 20) >> 28) * a.z;
    acc += static_cast<float>(static_cast<int>(v << 16) >> 28) * a.w;
    acc += static_cast<float>(static_cast<int>(v << 12) >> 28) * b.x;
    acc += static_cast<float>(static_cast<int>(v << 8) >> 28) * b.y;
    acc += static_cast<float>(static_cast<int>(v << 4) >> 28) * b.z;
    acc += static_cast<float>(static_cast<int>(v) >> 28) * b.w;
  }
  return acc;
}

__global__ void __launch_bounds__(256) Iu4GemvKernel(
    const std::uint8_t* __restrict__ tiles,
    const std::uint16_t* __restrict__ scales, const float* __restrict__ x,
    float* __restrict__ out, bool accumulate, std::uint32_t padded_rows,
    std::uint32_t valid_rows, std::uint32_t groups) {
  __shared__ float partial[kGemvWaves][16];
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t r = lane & 15, qh = lane >> 4;
  const std::uint32_t row_tile = blockIdx.x;
  const std::uint32_t row = row_tile * 16 + r;
  const std::uint32_t tiles_per_group = padded_rows >> 4;
  float acc = 0.0f;
  for (std::uint32_t e = wave; e < groups; e += kGemvWaves) {
    const std::uint8_t* base =
        tiles + ((std::size_t)e * tiles_per_group + row_tile) * 1024 + r * 16;
    const uint4 w0 = *reinterpret_cast<const uint4*>(base + qh * 256);
    const uint4 w1 = *reinterpret_cast<const uint4*>(base + (qh + 2) * 256);
    const float sc = __half2float(
        __ushort_as_half(scales[(std::size_t)e * padded_rows + row]));
    const float* xe = x + (std::size_t)e * 128 + qh * 32;
    const float s = ChunkDot(w0, xe) + ChunkDot(w1, xe + 64);
    acc += sc * s;
  }
  acc += __shfl_xor(acc, 16);
  if (qh == 0)
    partial[wave][r] = acc;
  __syncthreads();
  if (threadIdx.x < 16) {
    float total = 0.0f;
#pragma unroll
    for (std::uint32_t w = 0; w < kGemvWaves; ++w)
      total += partial[w][threadIdx.x];
    const std::uint32_t orow = row_tile * 16 + threadIdx.x;
    if (orow < valid_rows)
      out[orow] = accumulate ? out[orow] + total : total;
  }
}

}  // namespace

void LaunchIu4Gemv(const void* tiles, const std::uint16_t* weight_scales,
                   const float* x, float* out, bool accumulate,
                   std::size_t padded_rows, std::size_t valid_rows,
                   std::size_t k, hipStream_t stream) {
  Iu4GemvKernel<<<dim3(static_cast<unsigned>(padded_rows / 16)), 256, 0,
                  stream>>>(static_cast<const std::uint8_t*>(tiles),
                            weight_scales, x, out, accumulate,
                            static_cast<std::uint32_t>(padded_rows),
                            static_cast<std::uint32_t>(valid_rows),
                            static_cast<std::uint32_t>(k / 128));
}

}  // namespace gufo::hip
