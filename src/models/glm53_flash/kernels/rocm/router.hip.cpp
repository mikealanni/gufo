#include "src/models/glm53_flash/kernels/rocm/router.hpp"

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kExperts = 152;
constexpr int kHidden = 4096;
constexpr int kWaveLanes = 32;
constexpr int kTokenWaves = 4;
constexpr int kTileCols = 6;
constexpr int kColGroups = (kExperts + kTileCols * 16 - 1) / (kTileCols * 16);

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__global__ void SplitF16Kernel(const float* __restrict__ in,
                               _Float16* __restrict__ hi,
                               _Float16* __restrict__ lo, long long n,
                               float scale) {
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const float f = in[i] * scale;
    const _Float16 h = static_cast<_Float16>(f);
    hi[i] = h;
    lo[i] = static_cast<_Float16>(f - static_cast<float>(h));
  }
}

// One wave: 16 tokens x (kTileCols * 16) experts.
__global__ __launch_bounds__(kTokenWaves* kWaveLanes) void RouterGemmKernel(
    const float* __restrict__ y, const _Float16* __restrict__ w_hi,
    const _Float16* __restrict__ w_lo, float* __restrict__ out, int tokens,
    int num_experts) {
  const int lane = threadIdx.x % kWaveLanes;
  const int wave = threadIdx.x / kWaveLanes;
  const int col = lane & 15;
  const int row_hi = lane >> 4;
  const int token0 = (blockIdx.x * kTokenWaves + wave) * 16;
  const int expert0 = blockIdx.y * kTileCols * 16;
  const int token = token0 + col;
  const float* y_row =
      y + static_cast<std::size_t>(min(token, tokens - 1)) * kHidden;
  v8f acc[kTileCols];
#pragma unroll
  for (int t = 0; t < kTileCols; ++t) {
    acc[t] = v8f{};
  }
  for (int step = 0; step < kHidden / 16; ++step) {
    v16h a;
    v16h a_lo;
#pragma unroll
    for (int e = 0; e < 16; e += 4) {
      const float4 f = *reinterpret_cast<const float4*>(y_row + step * 16 + e);
      const float fv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const _Float16 hi = static_cast<_Float16>(fv[u]);
        a[e + u] = hi;
        a_lo[e + u] = static_cast<_Float16>(fv[u] - static_cast<float>(hi));
      }
    }
#pragma unroll
    for (int t = 0; t < kTileCols; ++t) {
      const int _exp_idx = expert0 + t * 16 + col;
      if (_exp_idx >= num_experts)
        continue;
      const std::size_t base =
          static_cast<std::size_t>(_exp_idx) * kHidden + step * 16;
      const v16h bh = *reinterpret_cast<const v16h*>(w_hi + base);
      const v16h bl = *reinterpret_cast<const v16h*>(w_lo + base);
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bh, acc[t]);
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bl, acc[t]);
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, bh, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < kTileCols; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = token0 + 2 * i + row_hi;
      if (row < tokens) {
        if (expert0 + t * 16 + col < num_experts)
          out[static_cast<std::size_t>(row) * num_experts + expert0 + t * 16 +
              col] = acc[t][i];
      }
    }
  }
}

}  // namespace

int SplitF16(const float* in, std::uint16_t* hi, std::uint16_t* lo, long long n,
             hipStream_t stream, float scale) {
  SplitF16Kernel<<<2048, 256, 0, stream>>>(in, reinterpret_cast<_Float16*>(hi),
                                           reinterpret_cast<_Float16*>(lo), n,
                                           scale);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int RouterGemm(const float* y, const std::uint16_t* w_hi,
               const std::uint16_t* w_lo, float* out, int tokens,
               int num_experts, hipStream_t stream) {
  const int col_groups = (num_experts + kTileCols * 16 - 1) / (kTileCols * 16);
  const int blocks = (tokens + 16 * kTokenWaves - 1) / (16 * kTokenWaves);
  RouterGemmKernel<<<dim3(blocks, col_groups), kTokenWaves * kWaveLanes, 0,
                     stream>>>(y, reinterpret_cast<const _Float16*>(w_hi),
                               reinterpret_cast<const _Float16*>(w_lo), out,
                               tokens, num_experts);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
