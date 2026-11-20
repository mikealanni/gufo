#include "src/models/glm53_flash/kernels/rocm/mtp.hpp"

#include <hip/hip_runtime.h>

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kMtpHidden = 4096;

// One block per (row, half): half 0 normalizes the embedding row by enorm and
// half 1 the trunk/block hidden row by hnorm; each writes its 4096-float slot
// of the packed 8192-wide row.
__global__ void MtpPackKernel(const float* __restrict__ embd,
                              const float* __restrict__ h, int carry_row,
                              const float* __restrict__ carry,
                              const float* __restrict__ enorm_w,
                              const float* __restrict__ hnorm_w, float eps,
                              float* __restrict__ concat, int rows) {
  const int half = blockIdx.x & 1;
  const int row = blockIdx.x >> 1;
  if (row >= rows)
    return;
  // Embedding rows come from EmbedTokens and repeat across the four
  // hyper-connection streams, so `embd` rows are 4 * 4096 wide.
  const float* src = embd + static_cast<long long>(row) * (4 * kMtpHidden);
  const float* w = enorm_w;
  if (half == 1) {
    if (carry != nullptr && row == carry_row) {
      src = carry;
    } else {
      const long long h_index =
          carry != nullptr ? row - (row > carry_row) : row;
      src = h + h_index * kMtpHidden;
    }
    w = hnorm_w;
  }
  const int tid = threadIdx.x;
  // One float4 (4 hidden values) per thread: 1024 threads x 4 = 4096.
  const float4* in = reinterpret_cast<const float4*>(src);
  const float4 val = in[tid];
  float sum = (val.x * val.x + val.y * val.y) + (val.z * val.z + val.w * val.w);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, sum, off, 32);
  }
  __shared__ float warp_partials[32];
  __shared__ float block_scale;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0)
    warp_partials[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    float total = 0.f;
    for (int i = 0; i < (1024 + 31) / 32; ++i)
      total += warp_partials[i];
    block_scale = rsqrtf(total / kMtpHidden + eps);
  }
  __syncthreads();
  const float scale = block_scale;
  const float4 wv = reinterpret_cast<const float4*>(w)[tid];
  float4* out = reinterpret_cast<float4*>(
      concat + static_cast<long long>(row) * (2 * kMtpHidden) +
      half * kMtpHidden);
  float4 o;
  o.x = val.x * scale * wv.x;
  o.y = val.y * scale * wv.y;
  o.z = val.z * scale * wv.z;
  o.w = val.w * scale * wv.w;
  out[tid] = o;
}

__global__ void MtpVecAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out, long long count) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i * 4 < count) {
    const float4 va = reinterpret_cast<const float4*>(a)[i];
    const float4 vb = reinterpret_cast<const float4*>(b)[i];
    float4 o;
    o.x = va.x + vb.x;
    o.y = va.y + vb.y;
    o.z = va.z + vb.z;
    o.w = va.w + vb.w;
    reinterpret_cast<float4*>(out)[i] = o;
  }
}

}  // namespace

int MtpPack(const float* embd, const float* h, int carry_row,
            const float* carry, const float* enorm_w, const float* hnorm_w,
            float eps, float* concat, int rows, hipStream_t stream) {
  if (rows <= 0)
    return 0;
  MtpPackKernel<<<rows * 2, 1024, 0, stream>>>(
      embd, h, carry_row, carry, enorm_w, hnorm_w, eps, concat, rows);
  return static_cast<int>(hipGetLastError());
}

int MtpVecAdd(const float* a, const float* b, float* out, int n, int rows,
              hipStream_t stream) {
  const long long count = static_cast<long long>(rows) * n;
  const long long quads = count / 4;
  MtpVecAddKernel<<<static_cast<unsigned>((quads + 255) / 256), 256, 0,
                    stream>>>(a, b, out, count);
  return static_cast<int>(hipGetLastError());
}

}  // namespace gufo::models::glm53_flash::rocm
