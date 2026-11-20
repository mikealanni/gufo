#include "src/models/glm53_flash/kernels/rocm/moe.hpp"

#include <hip/hip_fp16.h>

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr float kClamp = 10.0F;
constexpr float kMinWeightSum = 6.103515625e-5F;
constexpr int kElemThreads = 256;
constexpr int kRouteThreads = 512;

__device__ float Sigmoid(float v) {
  return 1.0F / (1.0F + expf(-v));
}

__global__ void SwigluClampKernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ act, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float g = fminf(gate[i], kClamp);
  const float u = fminf(fmaxf(up[i], -kClamp), kClamp);
  act[i] = g * Sigmoid(g) * u;
}

__global__ __launch_bounds__(kRouteThreads) void MoeRouteKernel(
    const float* __restrict__ logits, const float* __restrict__ bias,
    float scale, int* __restrict__ ids, float* __restrict__ weights,
    int experts) {
  logits += blockIdx.x * experts;
  ids += blockIdx.x * kMoeUsed;
  weights += blockIdx.x * kMoeUsed;
  __shared__ float prob[kMaxMoeExperts];
  __shared__ float sel[kMaxMoeExperts];
  __shared__ bool taken[kMaxMoeExperts];
  __shared__ float best_value[kRouteThreads];
  __shared__ int best_index[kRouteThreads];
  const int t = threadIdx.x;
  for (int e = t; e < experts; e += kRouteThreads) {
    prob[e] = Sigmoid(logits[e]);
    sel[e] = prob[e] + bias[e];
    taken[e] = false;
  }
  __syncthreads();
  __shared__ int chosen[kMoeUsed];
  for (int round = 0; round < kMoeUsed; ++round) {
    float value = -INFINITY;
    int index = experts;
    if (t < experts && !taken[t]) {
      value = sel[t];
      index = t;
    }
    best_value[t] = value;
    best_index[t] = index;
    __syncthreads();
    for (int stride = kRouteThreads / 2; stride > 0; stride >>= 1) {
      if (t < stride) {
        const float ov = best_value[t + stride];
        const int oi = best_index[t + stride];
        if (ov > best_value[t] || (ov == best_value[t] && oi < best_index[t])) {
          best_value[t] = ov;
          best_index[t] = oi;
        }
      }
      __syncthreads();
    }
    if (t == 0) {
      chosen[round] = best_index[0];
      taken[best_index[0]] = true;
    }
    __syncthreads();
  }
  if (t == 0) {
    float sum = 0.0F;
    for (int i = 0; i < kMoeUsed; ++i) {
      sum += prob[chosen[i]];
    }
    sum = fmaxf(sum, kMinWeightSum);
    for (int i = 0; i < kMoeUsed; ++i) {
      ids[i] = chosen[i];
      weights[i] = prob[chosen[i]] / sum * scale;
    }
  }
}

__global__ void MoeCombineKernel(const float* __restrict__ parts,
                                 const float* __restrict__ weights,
                                 const float* __restrict__ shared,
                                 float* __restrict__ out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= kMoeHidden) {
    return;
  }
  const int token = blockIdx.y;
  parts += static_cast<std::size_t>(token) * kMoeUsed * kMoeHidden;
  weights += token * kMoeUsed;
  shared += static_cast<std::size_t>(token) * kMoeHidden;
  out += static_cast<std::size_t>(token) * kMoeHidden;
  float acc = 0.0F;
  for (int i = 0; i < kMoeUsed; ++i) {
    acc += weights[i] * parts[i * kMoeHidden + c];
  }
  out[c] = acc + shared[c];
}

__global__ void ToHalfKernel(const float* __restrict__ src,
                             __half* __restrict__ dst, long long n) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = __float2half(src[i]);
  }
}

__global__ void MoeCombineF16Kernel(const __half* __restrict__ parts,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ shared,
                                    float* __restrict__ out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= kMoeHidden) {
    return;
  }
  const int token = blockIdx.y;
  parts += static_cast<std::size_t>(token) * kMoeUsed * kMoeHidden;
  weights += token * kMoeUsed;
  shared += static_cast<std::size_t>(token) * kMoeHidden;
  out += static_cast<std::size_t>(token) * kMoeHidden;
  float acc = 0.0F;
  for (int i = 0; i < kMoeUsed; ++i) {
    acc += weights[i] * __half2float(parts[i * kMoeHidden + c]);
  }
  out[c] = acc + shared[c];
}

}  // namespace

int SwigluClamp(const float* gate, const float* up, float* act, int n,
                hipStream_t stream) {
  SwigluClampKernel<<<(n + kElemThreads - 1) / kElemThreads, kElemThreads, 0,
                      stream>>>(gate, up, act, n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MoeRoute(const float* logits, const float* bias, float scale, int* ids,
             float* weights, int experts, hipStream_t stream, int tokens) {
  MoeRouteKernel<<<tokens, kRouteThreads, 0, stream>>>(logits, bias, scale, ids,
                                                       weights, experts);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MoeCombine(const float* parts, const float* weights, const float* shared,
               float* out, hipStream_t stream, int tokens) {
  MoeCombineKernel<<<dim3((kMoeHidden + kElemThreads - 1) / kElemThreads,
                          tokens),
                     kElemThreads, 0, stream>>>(parts, weights, shared, out);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int ToHalf(const float* src, void* dst, long long n, hipStream_t stream) {
  ToHalfKernel<<<static_cast<unsigned>((n + kElemThreads - 1) / kElemThreads),
                 kElemThreads, 0, stream>>>(src, static_cast<__half*>(dst), n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MoeCombineF16(const void* parts, const float* weights, const float* shared,
                  float* out, hipStream_t stream, int tokens) {
  MoeCombineF16Kernel<<<dim3((kMoeHidden + kElemThreads - 1) / kElemThreads,
                             tokens),
                        kElemThreads, 0, stream>>>(
      static_cast<const __half*>(parts), weights, shared, out);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
