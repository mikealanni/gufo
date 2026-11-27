#include "src/models/raon1/kernels/rocm/raon.hpp"

#include <cmath>
#include <cstdlib>

#include "src/models/glm53_flash/kernels/rocm/xor_lane.hpp"

namespace gufo::models::raon1::rocm {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kHcThreads = 640;
constexpr int kRdaThreads = 128;
constexpr int kRdaWarps = kRdaThreads / 32;

__device__ float Sigmoid(float v) {
  return 1.0F / (1.0F + expf(-v));
}

__device__ float BlockSum(float v, float* scratch) {
  for (int s = warpSize / 2; s > 0; s >>= 1)
    v += glm53_flash::rocm::XorLane(v, s);
  if ((threadIdx.x % warpSize) == 0)
    scratch[threadIdx.x / warpSize] = v;
  __syncthreads();
  float total = 0.0F;
  for (int w = 0; w < kWaves; ++w)
    total += scratch[w];
  return total;
}

__device__ float BlockMax(float v, float* scratch) {
  for (int s = warpSize / 2; s > 0; s >>= 1)
    v = fmaxf(v, glm53_flash::rocm::XorLane(v, s));
  if ((threadIdx.x % warpSize) == 0)
    scratch[threadIdx.x / warpSize] = v;
  __syncthreads();
  float best = -INFINITY;
  for (int w = 0; w < kWaves; ++w)
    best = fmaxf(best, scratch[w]);
  return best;
}

__device__ float Helmert(int i, int j) {
  const int hm = kRaonStreams - 1;
  const float v =
      (i <= j ? 1.0F : (i == j + 1 ? -static_cast<float>(j + 1) : 0.0F)) /
      sqrtf(static_cast<float>((j + 1) * (j + 2)));
  return (j < hm) ? v : 0.0F;
}

__global__ void F32MatvecKernel(const float* __restrict__ w,
                                const float* __restrict__ x,
                                float* __restrict__ y, int m, int k) {
  __shared__ float reduce[kWaves];
  const int row = blockIdx.x;
  const float* wr = w + static_cast<std::size_t>(row) * k;
  float acc = 0.0F;
  for (int i = threadIdx.x; i < k; i += kThreads) {
    acc = fmaf(wr[i], x[i], acc);
  }
  const float total = BlockSum(acc, reduce);
  if (threadIdx.x == 0)
    y[row] = total;
}

__global__ void F16MatvecKernel(const __half* __restrict__ w,
                                const float* __restrict__ x,
                                float* __restrict__ y, int k,
                                std::uint64_t row0) {
  __shared__ float reduce[kWaves];
  const int row = blockIdx.x;
  const __half* wr = w + (row0 + static_cast<std::uint64_t>(row)) * k;
  float acc = 0.0F;
  for (int i = threadIdx.x; i < k; i += kThreads) {
    acc = fmaf(__half2float(wr[i]), x[i], acc);
  }
  const float total = BlockSum(acc, reduce);
  if (threadIdx.x == 0)
    y[row] = total;
}

__global__ void F16MatvecExpertsKernel(const __half* __restrict__ w,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int m, int k,
                                       const int* __restrict__ ids,
                                       bool x_per_expert) {
  __shared__ float reduce[kWaves];
  const int row = blockIdx.x;
  const std::uint64_t expert = static_cast<std::uint64_t>(ids[blockIdx.y]);
  const __half* wr = w + (expert * static_cast<std::uint64_t>(m) + row) * k;
  const float* xr =
      x + (x_per_expert ? static_cast<std::size_t>(blockIdx.y) * k : 0);
  float acc = 0.0F;
  for (int i = threadIdx.x; i < k; i += kThreads) {
    acc = fmaf(__half2float(wr[i]), xr[i], acc);
  }
  const float total = BlockSum(acc, reduce);
  if (threadIdx.x == 0)
    y[static_cast<std::size_t>(blockIdx.y) * m + row] = total;
}

__device__ float WarpMax(float v) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1)
    v = fmaxf(v, glm53_flash::rocm::XorLane(v, off));
  return v;
}

__device__ float WarpSum(float v) {
  for (int s = warpSize / 2; s > 0; s >>= 1)
    v += glm53_flash::rocm::XorLane(v, s);
  return v;
}

__global__ void HcSelectKernel(const float* __restrict__ x,
                               const float* __restrict__ fn,
                               const float* __restrict__ scale,
                               const float* __restrict__ base, float hc_eps,
                               float magnitude, float* __restrict__ y,
                               float* __restrict__ out_pre,
                               float* __restrict__ out_post,
                               float* __restrict__ comb) {
  constexpr int kDim = kRaonStreams * kRaonHidden;
  constexpr int kHcWarps = kHcThreads / 32;
  __shared__ float reduce[kHcWarps];
  __shared__ float mixes[kRaonMixes];
  __shared__ float pre[kRaonStreams];
  __shared__ float pst[kRaonStreams];
  const int tok = blockIdx.x;
  const float* xt = x + static_cast<std::size_t>(tok) * kDim;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sq = 0.0F;
  for (int i = threadIdx.x; i < kDim; i += kHcThreads) {
    sq = fmaf(xt[i], xt[i], sq);
  }
  const float wsq = WarpSum(sq);
  if (lane == 0)
    reduce[warp] = wsq;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (int w = 0; w < kHcWarps; ++w)
      total += reduce[w];
    reduce[0] = rsqrtf(total / static_cast<float>(kDim) + hc_eps);
  }
  __syncthreads();
  const float inv = reduce[0];

  if (warp < kRaonMixes) {
    const float* wr = fn + static_cast<std::size_t>(warp) * kDim;
    float acc = 0.0F;
    for (int i = lane; i < kDim; i += 32) {
      acc = fmaf(wr[i], xt[i] * inv, acc);
    }
    const float total = WarpSum(acc);
    if (lane == 0)
      mixes[warp] = total;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int s = 0; s < kRaonStreams; ++s) {
      pre[s] = Sigmoid(mixes[s] * scale[0] + base[s]);
      pst[s] = magnitude * Sigmoid(mixes[kRaonStreams + s] * scale[1] +
                                   base[kRaonStreams + s]);
      const std::size_t pb = static_cast<std::size_t>(tok) * kRaonStreams;
      out_pre[pb + s] = pre[s];
      out_post[pb + s] = pst[s];
    }
    const int hm = kRaonStreams - 1;
    float tz[kRaonStreams * kRaonStreams];
    for (int i = 0; i < hm; ++i) {
      for (int j = 0; j < hm; ++j) {
        const int k = 2 * kRaonStreams + i * hm + j;
        tz[i * hm + j] = mixes[k] * scale[2] + base[k];
      }
    }
    float cell[kRaonStreams * kRaonStreams];
    for (int i = 0; i < kRaonStreams; ++i) {
      for (int j = 0; j < kRaonStreams; ++j) {
        float acc = 0.0F;
        for (int m = 0; m < hm; ++m) {
          for (int n = 0; n < hm; ++n) {
            acc = fmaf(Helmert(i, m) * Helmert(j, n), tz[m * hm + n], acc);
          }
        }
        cell[i * kRaonStreams + j] = acc;
      }
    }
    float rho = 0.0F;
    for (int i = 0; i < kRaonStreams; ++i) {
      for (int j = 0; j < kRaonStreams; ++j) {
        const float b = (i == j ? 0.5F : 0.0F) + 0.5F / kRaonStreams;
        rho = fmaxf(rho, -cell[i * kRaonStreams + j] / b);
      }
    }
    rho = fmaxf(rho, 0.0F);
    const float gamma = Sigmoid(scale[3]);
    const float denom = 1.0F + rho;
    comb += static_cast<std::size_t>(tok) * kRaonStreams * kRaonStreams;
    for (int d = 0; d < kRaonStreams; ++d) {
      out_post[static_cast<std::size_t>(tok) * kRaonStreams + d] = pst[d];
      for (int s = 0; s < kRaonStreams; ++s) {
        const float b = (d == s ? 0.5F : 0.0F) + 0.5F / kRaonStreams;
        const float phi = b + cell[d * kRaonStreams + s] / denom;
        comb[d + kRaonStreams * s] =
            (1.0F - gamma) * (d == s ? 1.0F : 0.0F) + gamma * phi;
      }
    }
  }
  __syncthreads();

  float* yt = y + static_cast<std::size_t>(tok) * kRaonHidden;
  for (int i = threadIdx.x; i < kRaonHidden; i += kHcThreads) {
    float acc = 0.0F;
    for (int s = 0; s < kRaonStreams; ++s) {
      acc =
          fmaf(xt[static_cast<std::size_t>(s) * kRaonHidden + i], pre[s], acc);
    }
    yt[i] = acc;
  }
}

// dots[t][m] = fn[m] . x[t] for kRaonMixes mixes over `kTok` tokens per block.
constexpr int kHcDotTokens = 8;
__global__ __launch_bounds__(kRaonMixes *
                             32) void HcDotsKernel(const float* __restrict__ x,
                                                   const float* __restrict__ fn,
                                                   float* __restrict__ dots,
                                                   int tokens) {
  constexpr int kDim = kRaonStreams * kRaonHidden;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int t0 = blockIdx.x * kHcDotTokens;
  float acc[kHcDotTokens];
#pragma unroll
  for (int t = 0; t < kHcDotTokens; ++t)
    acc[t] = 0.0F;
  const float4* wr = reinterpret_cast<const float4*>(
      fn + static_cast<std::size_t>(warp) * kDim);
  for (int i = lane; i < kDim / 4; i += 32) {
    const float4 w4 = wr[i];
#pragma unroll
    for (int t = 0; t < kHcDotTokens; ++t) {
      const int tok = min(t0 + t, tokens - 1);
      const float4 x4 = reinterpret_cast<const float4*>(
          x + static_cast<std::size_t>(tok) * kDim)[i];
      acc[t] =
          fmaf(w4.x, x4.x,
               fmaf(w4.y, x4.y, fmaf(w4.z, x4.z, fmaf(w4.w, x4.w, acc[t]))));
    }
  }
#pragma unroll
  for (int t = 0; t < kHcDotTokens; ++t) {
    const float total = WarpSum(acc[t]);
    if (lane == 0 && t0 + t < tokens) {
      dots[static_cast<std::size_t>(t0 + t) * kRaonMixes + warp] = total;
    }
  }
}

constexpr int kHcChunk = 512;
constexpr int kHcChunks = kRaonStreams * kRaonHidden / kHcChunk;
// Partial mixes: dots[(t * kHcChunks + c) * 17 + m] over a 512-wide chunk.
__global__ __launch_bounds__(kRaonMixes * 32) void HcDotsPartKernel(
    const float* __restrict__ x, const float* __restrict__ fn,
    float* __restrict__ dots) {
  constexpr int kDim = kRaonStreams * kRaonHidden;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int c = blockIdx.x;
  const int t = blockIdx.y;
  const float4* xr = reinterpret_cast<const float4*>(
      x + static_cast<std::size_t>(t) * kDim + c * kHcChunk);
  const float4* wr = reinterpret_cast<const float4*>(
      fn + static_cast<std::size_t>(warp) * kDim + c * kHcChunk);
  float acc = 0.0F;
#pragma unroll
  for (int j = 0; j < kHcChunk / 128; ++j) {
    const float4 w4 = wr[lane + 32 * j];
    const float4 x4 = xr[lane + 32 * j];
    acc = fmaf(w4.x, x4.x,
               fmaf(w4.y, x4.y, fmaf(w4.z, x4.z, fmaf(w4.w, x4.w, acc))));
  }
  const float total = WarpSum(acc);
  if (lane == 0) {
    dots[(static_cast<std::size_t>(t) * kHcChunks + c) * kRaonMixes + warp] =
        total;
  }
}

__global__ void HcSelectPreKernel(const float* __restrict__ x,
                                  const float* __restrict__ dots,
                                  const float* __restrict__ scale,
                                  const float* __restrict__ base, float hc_eps,
                                  float magnitude, float* __restrict__ y,
                                  float* __restrict__ out_pre,
                                  float* __restrict__ out_post,
                                  float* __restrict__ comb, int chunks) {
  constexpr int kDim = kRaonStreams * kRaonHidden;
  constexpr int kHcWarps = kHcThreads / 32;
  __shared__ float reduce[kHcWarps];
  __shared__ float mixes[kRaonMixes];
  __shared__ float pre[kRaonStreams];
  __shared__ float pst[kRaonStreams];
  const int tok = blockIdx.x;
  const float* xt = x + static_cast<std::size_t>(tok) * kDim;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sq = 0.0F;
  for (int i = threadIdx.x; i < kDim; i += kHcThreads) {
    sq = fmaf(xt[i], xt[i], sq);
  }
  const float wsq = WarpSum(sq);
  if (lane == 0)
    reduce[warp] = wsq;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (int w = 0; w < kHcWarps; ++w)
      total += reduce[w];
    reduce[0] = rsqrtf(total / static_cast<float>(kDim) + hc_eps);
  }
  __syncthreads();
  const float inv = reduce[0];

  if (threadIdx.x < kRaonMixes) {
    float sum = 0.0F;
    for (int c = 0; c < chunks; ++c) {
      sum += dots[(static_cast<std::size_t>(tok) * chunks + c) * kRaonMixes +
                  threadIdx.x];
    }
    mixes[threadIdx.x] = sum * inv;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int s = 0; s < kRaonStreams; ++s) {
      pre[s] = Sigmoid(mixes[s] * scale[0] + base[s]);
      pst[s] = magnitude * Sigmoid(mixes[kRaonStreams + s] * scale[1] +
                                   base[kRaonStreams + s]);
      const std::size_t pb = static_cast<std::size_t>(tok) * kRaonStreams;
      out_pre[pb + s] = pre[s];
      out_post[pb + s] = pst[s];
    }
    const int hm = kRaonStreams - 1;
    float tz[kRaonStreams * kRaonStreams];
    for (int i = 0; i < hm; ++i) {
      for (int j = 0; j < hm; ++j) {
        const int k = 2 * kRaonStreams + i * hm + j;
        tz[i * hm + j] = mixes[k] * scale[2] + base[k];
      }
    }
    float cell[kRaonStreams * kRaonStreams];
    for (int i = 0; i < kRaonStreams; ++i) {
      for (int j = 0; j < kRaonStreams; ++j) {
        float acc = 0.0F;
        for (int m = 0; m < hm; ++m) {
          for (int n = 0; n < hm; ++n) {
            acc = fmaf(Helmert(i, m) * Helmert(j, n), tz[m * hm + n], acc);
          }
        }
        cell[i * kRaonStreams + j] = acc;
      }
    }
    float rho = 0.0F;
    for (int i = 0; i < kRaonStreams; ++i) {
      for (int j = 0; j < kRaonStreams; ++j) {
        const float b = (i == j ? 0.5F : 0.0F) + 0.5F / kRaonStreams;
        rho = fmaxf(rho, -cell[i * kRaonStreams + j] / b);
      }
    }
    rho = fmaxf(rho, 0.0F);
    const float gamma = Sigmoid(scale[3]);
    const float denom = 1.0F + rho;
    comb += static_cast<std::size_t>(tok) * kRaonStreams * kRaonStreams;
    for (int d = 0; d < kRaonStreams; ++d) {
      out_post[static_cast<std::size_t>(tok) * kRaonStreams + d] = pst[d];
      for (int s = 0; s < kRaonStreams; ++s) {
        const float b = (d == s ? 0.5F : 0.0F) + 0.5F / kRaonStreams;
        const float phi = b + cell[d * kRaonStreams + s] / denom;
        comb[d + kRaonStreams * s] =
            (1.0F - gamma) * (d == s ? 1.0F : 0.0F) + gamma * phi;
      }
    }
  }
  __syncthreads();

  float* yt = y + static_cast<std::size_t>(tok) * kRaonHidden;
  for (int i = threadIdx.x; i < kRaonHidden; i += kHcThreads) {
    float acc = 0.0F;
    for (int s = 0; s < kRaonStreams; ++s) {
      acc =
          fmaf(xt[static_cast<std::size_t>(s) * kRaonHidden + i], pre[s], acc);
    }
    yt[i] = acc;
  }
}

// One block per token; thread i owns float4 column i of all four streams.
__global__ __launch_bounds__(kRaonHidden / 4) void HcApplyKernel(
    const float* __restrict__ x, const float* __restrict__ block,
    const float* __restrict__ post, const float* __restrict__ comb,
    float* __restrict__ dst) {
  const int tok = blockIdx.x;
  const int i = threadIdx.x;
  const std::size_t xb =
      static_cast<std::size_t>(tok) * kRaonStreams * kRaonHidden;
  const float* cb =
      comb + static_cast<std::size_t>(tok) * kRaonStreams * kRaonStreams;
  const float* pb = post + static_cast<std::size_t>(tok) * kRaonStreams;
  const float4 b = reinterpret_cast<const float4*>(
      block + static_cast<std::size_t>(tok) * kRaonHidden)[i];
  float4 v[kRaonStreams];
  for (int s = 0; s < kRaonStreams; ++s) {
    v[s] = reinterpret_cast<const float4*>(
        x + xb + static_cast<std::size_t>(s) * kRaonHidden)[i];
  }
  for (int d = 0; d < kRaonStreams; ++d) {
    const float p = pb[d];
    float4 acc = make_float4(b.x * p, b.y * p, b.z * p, b.w * p);
    for (int s = 0; s < kRaonStreams; ++s) {
      const float c = cb[s * kRaonStreams + d];
      acc.x = fmaf(v[s].x, c, acc.x);
      acc.y = fmaf(v[s].y, c, acc.y);
      acc.z = fmaf(v[s].z, c, acc.z);
      acc.w = fmaf(v[s].w, c, acc.w);
    }
    reinterpret_cast<float4*>(
        dst + xb + static_cast<std::size_t>(d) * kRaonHidden)[i] = acc;
  }
}

__global__ void RmsNormRowsKernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float* __restrict__ y, int n, float eps,
                                  __half* __restrict__ y16) {
  __shared__ float reduce[kWaves];
  const int row = blockIdx.x;
  const float* in = x + static_cast<std::size_t>(row) * n;
  float* out = y + static_cast<std::size_t>(row) * n;
  float sq = 0.0F;
  for (int i = threadIdx.x; i < n; i += kThreads)
    sq = fmaf(in[i], in[i], sq);
  const float inv = rsqrtf(BlockSum(sq, reduce) / static_cast<float>(n) + eps);
  for (int i = threadIdx.x; i < n; i += kThreads) {
    const float v = in[i] * inv * (w != nullptr ? w[i] : 1.0F);
    out[i] = v;
    if (y16 != nullptr)
      y16[static_cast<std::size_t>(row) * n + i] = __float2half(v);
  }
}

__global__ void L2HeadsKernel(const float* __restrict__ x,
                              float* __restrict__ y, int hd, float eps) {
  __shared__ float reduce[kWaves];
  const int h = blockIdx.x;
  const float* in = x + static_cast<std::size_t>(h) * hd;
  float* out = y + static_cast<std::size_t>(h) * hd;
  float sq = 0.0F;
  for (int i = threadIdx.x; i < hd; i += kThreads)
    sq = fmaf(in[i], in[i], sq);
  const float inv = rsqrtf(BlockSum(sq, reduce) + eps);
  for (int i = threadIdx.x; i < hd; i += kThreads)
    out[i] = in[i] * inv;
}

__global__ void RdaConvKernel(
    const float* __restrict__ wq, const float* __restrict__ wk,
    const float* __restrict__ wv, const float* __restrict__ rawq,
    const float* __restrict__ rawk, const float* __restrict__ rawv,
    float* __restrict__ hist, float* __restrict__ outq,
    float* __restrict__ outk, float* __restrict__ outv, int channels,
    int taps) {
  const int w = blockIdx.x;
  const int ch = threadIdx.x + blockIdx.y * kThreads;
  if (ch >= channels)
    return;
  const float* weights = w == 0 ? wq : (w == 1 ? wk : wv);
  const float* raw = w == 0 ? rawq : (w == 1 ? rawk : rawv);
  float* out = w == 0 ? outq : (w == 1 ? outk : outv);
  const float* wr = weights + static_cast<std::size_t>(ch) * (taps + 1);
  float acc = wr[taps] * raw[ch];
  float* h = hist + static_cast<std::size_t>(w) * channels * taps +
             static_cast<std::size_t>(ch) * taps;
  for (int t = 0; t < taps; ++t)
    acc = fmaf(wr[t], h[t], acc);
  out[ch] = acc * Sigmoid(acc);
  for (int t = 0; t + 1 < taps; ++t)
    h[t] = h[t + 1];
  h[taps - 1] = raw[ch];
}

__global__ void RdaDecayKernel(const float* __restrict__ gproj,
                               const float* __restrict__ dt,
                               const float* __restrict__ sa,
                               float* __restrict__ decay, int hd,
                               float lower_bound) {
  const int h = blockIdx.x;
  const int i = threadIdx.x;
  if (i >= hd)
    return;
  const std::size_t k = static_cast<std::size_t>(h) * hd + i;
  const float x = gproj[k] + dt[k];
  decay[k] = expf(lower_bound * Sigmoid(-sa[h] * x));
}

__global__ void RdaRecurKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ decay,
    const float* __restrict__ beta, const float* __restrict__ ratio,
    const float* __restrict__ g2, const float* __restrict__ norm_w,
    float* __restrict__ state, float* __restrict__ out,
    float* __restrict__ pre_norm, int hd, float eps) {
  constexpr int kItemsPerLane = 4;
  __shared__ float su[kRaonHeadDim];
  __shared__ float reduce[kRdaWarps];
  __shared__ float inv_shared;
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float b = Sigmoid(beta[h]);
  const float r =
      expf(logf(Sigmoid(beta[h] + ratio[h])) - logf(Sigmoid(beta[h])));
  const float qscale = rsqrtf(static_cast<float>(hd));
  float* s = state + static_cast<std::size_t>(h) * hd * hd;
  const float* qh = q + static_cast<std::size_t>(h) * hd;
  const float* kh = k + static_cast<std::size_t>(h) * hd;
  const float* vh = v + static_cast<std::size_t>(h) * hd;
  const float* dh = decay + static_cast<std::size_t>(h) * hd;
  const float* gh = g2 + static_cast<std::size_t>(h) * hd;
  float* oh = out + static_cast<std::size_t>(h) * hd;

  for (int rr = 0; rr < hd / kRdaWarps; ++rr) {
    const int jv = warp + rr * kRdaWarps;
    float* row = s + static_cast<std::size_t>(jv) * hd;
    float acc = 0.0F;
    for (int base = lane * kItemsPerLane; base < hd; base += kRdaThreads) {
#pragma unroll
      for (int j = 0; j < kItemsPerLane; ++j) {
        const int i = base + j;
        const float value = row[i] * dh[i];
        row[i] = value;
        acc = fmaf(value, kh[i], acc);
      }
    }
    const float total = WarpSum(acc);
    if (lane == 0)
      su[jv] = total;
  }
  __syncthreads();

  for (int rr = 0; rr < hd / kRdaWarps; ++rr) {
    const int jv = warp + rr * kRdaWarps;
    float* row = s + static_cast<std::size_t>(jv) * hd;
    const float diff = (vh[jv] * r - su[jv]) * b;
    float acc = 0.0F;
    for (int base = lane * kItemsPerLane; base < hd; base += kRdaThreads) {
#pragma unroll
      for (int j = 0; j < kItemsPerLane; ++j) {
        const int i = base + j;
        const float value = fmaf(diff, kh[i], row[i]);
        row[i] = value;
        acc = fmaf(value * qh[i], qscale, acc);
      }
    }
    const float value = WarpSum(acc);
    if (lane == 0) {
      oh[jv] = value;
      if (pre_norm != nullptr) {
        pre_norm[static_cast<std::size_t>(h) * hd + jv] = value;
      }
    }
  }
  __syncthreads();

  const int norm_jv = warp * 32 + lane;
  const float sq = (norm_jv < hd) ? oh[norm_jv] * oh[norm_jv] : 0.0F;
  const float warp_sq = WarpSum(sq);
  if (lane == 0)
    reduce[warp] = warp_sq;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (int w = 0; w < kRdaWarps; ++w)
      total += reduce[w];
    inv_shared = rsqrtf(total / static_cast<float>(hd) + eps);
  }
  __syncthreads();
  if (norm_jv < hd) {
    oh[norm_jv] =
        oh[norm_jv] * inv_shared * norm_w[norm_jv] * Sigmoid(gh[norm_jv]);
  }
}

// Sums 16 per-lane values across the wave; lane L ends with the total of
// v[L & 15].
__device__ __forceinline__ float ReduceScatter16(float (&v)[16], int lane) {
#pragma unroll
  for (int i = 0; i < 16; ++i)
    v[i] += glm53_flash::rocm::XorLane(v[i], 16);
#pragma unroll
  for (int off = 8; off >= 1; off >>= 1) {
    const bool hi = (lane & off) != 0;
#pragma unroll
    for (int i = 0; i < off; ++i) {
      const float send = hi ? v[i] : v[i + off];
      const float keep = hi ? v[i + off] : v[i];
      v[i] = keep + glm53_flash::rocm::XorLane(send, off);
    }
  }
  return v[0];
}

// Fused RDA for `n` tokens: causal conv + SiLU, L2 norm, decay, delta-rule
// recurrence with the per-head state held in registers, and the gated output
// norm. One block per head, 128 threads (channel = thread, rows = 32 per warp).
__global__ __launch_bounds__(128) void RdaPrefillKernel(
    const float* __restrict__ wq, const float* __restrict__ wk,
    const float* __restrict__ wv, const float* __restrict__ rawq,
    const float* __restrict__ rawk, const float* __restrict__ rawv,
    float* __restrict__ hist, const float* __restrict__ gproj,
    const float* __restrict__ dt, const float* __restrict__ sa,
    const float* __restrict__ beta, const float* __restrict__ ratio,
    const float* __restrict__ g2, const float* __restrict__ norm_w,
    float* __restrict__ state, float* __restrict__ out, int n, int channels,
    int heads, float lower_bound, float eps) {
  constexpr int kHd = 128;
  constexpr int kTaps = 3;
  __shared__ __attribute__((aligned(16))) float sq[kHd];
  __shared__ __attribute__((aligned(16))) float sk[kHd];
  __shared__ __attribute__((aligned(16))) float sv[kHd];
  __shared__ __attribute__((aligned(16))) float sd[kHd];
  __shared__ float red[2][4];
  const int h = blockIdx.x;
  const int c = threadIdx.x;
  const int warp = c >> 5;
  const int lane = c & 31;
  const int gc = h * kHd + c;
  const float* wsrc[3] = {wq, wk, wv};
  const float* raws[3] = {rawq, rawk, rawv};
  float wc[3][kTaps + 1];
  float hs[3][kTaps];
  for (int w = 0; w < 3; ++w) {
    for (int t = 0; t <= kTaps; ++t) {
      wc[w][t] = wsrc[w][static_cast<std::size_t>(gc) * (kTaps + 1) + t];
    }
    for (int t = 0; t < kTaps; ++t) {
      hs[w][t] =
          hist[(static_cast<std::size_t>(w) * channels + gc) * kTaps + t];
    }
  }
  // Thread c owns state row c: 128 columns in registers.
  float st[kHd];
  {
    const float4* row = reinterpret_cast<const float4*>(
        state + (static_cast<std::size_t>(h) * kHd + c) * kHd);
#pragma unroll
    for (int j = 0; j < kHd / 4; ++j) {
      const float4 v = row[j];
      st[4 * j] = v.x;
      st[4 * j + 1] = v.y;
      st[4 * j + 2] = v.z;
      st[4 * j + 3] = v.w;
    }
  }
  const float dtc = dt[gc];
  const float sah = sa[h];
  const float nw = norm_w[c];
  const float qscale = rsqrtf(static_cast<float>(kHd));

  float cur_raw[3];
  float cur_g = 0.0F;
  float cur_gate = 0.0F;
  float cur_bt = 0.0F;
  float cur_rt = 0.0F;
  auto fetch = [&](int i) {
    for (int w = 0; w < 3; ++w) {
      cur_raw[w] = raws[w][static_cast<std::size_t>(i) * channels + gc];
    }
    cur_g = gproj[static_cast<std::size_t>(i) * channels + gc];
    cur_gate = g2[static_cast<std::size_t>(i) * channels + gc];
    cur_bt = beta[static_cast<std::size_t>(i) * heads + h];
    cur_rt = ratio[static_cast<std::size_t>(i) * heads + h];
  };
  if (n > 0)
    fetch(0);
  for (int i = 0; i < n; ++i) {
    float conv[3];
    for (int w = 0; w < 3; ++w) {
      float acc = wc[w][kTaps] * cur_raw[w];
      for (int t = 0; t < kTaps; ++t)
        acc = fmaf(wc[w][t], hs[w][t], acc);
      conv[w] = acc * Sigmoid(acc);
      for (int t = 0; t + 1 < kTaps; ++t)
        hs[w][t] = hs[w][t + 1];
      hs[w][kTaps - 1] = cur_raw[w];
    }
    const float gx = cur_g + dtc;
    const float gate_in = cur_gate;
    const float bt = cur_bt;
    const float rt = cur_rt;
    if (i + 1 < n)
      fetch(i + 1);
    const float qq = WarpSum(conv[0] * conv[0]);
    const float kk = WarpSum(conv[1] * conv[1]);
    if (lane == 0) {
      red[0][warp] = qq;
      red[1][warp] = kk;
    }
    __syncthreads();
    const float qn =
        rsqrtf(red[0][0] + red[0][1] + red[0][2] + red[0][3] + 1e-6F);
    const float kn =
        rsqrtf(red[1][0] + red[1][1] + red[1][2] + red[1][3] + 1e-6F);
    sq[c] = conv[0] * qn;
    sk[c] = conv[1] * kn;
    sv[c] = conv[2];
    sd[c] = expf(lower_bound * Sigmoid(-sah * gx));
    const float b = Sigmoid(bt);
    const float r = expf(logf(Sigmoid(bt + rt)) - logf(Sigmoid(bt)));
    __syncthreads();

    float a0 = 0.0F;
    float a1 = 0.0F;
    float a2 = 0.0F;
    float a3 = 0.0F;
#pragma unroll
    for (int j = 0; j < kHd / 4; ++j) {
      const float4 d4 = reinterpret_cast<const float4*>(sd)[j];
      const float4 k4 = reinterpret_cast<const float4*>(sk)[j];
      st[4 * j] *= d4.x;
      st[4 * j + 1] *= d4.y;
      st[4 * j + 2] *= d4.z;
      st[4 * j + 3] *= d4.w;
      a0 = fmaf(st[4 * j], k4.x, a0);
      a1 = fmaf(st[4 * j + 1], k4.y, a1);
      a2 = fmaf(st[4 * j + 2], k4.z, a2);
      a3 = fmaf(st[4 * j + 3], k4.w, a3);
      __builtin_amdgcn_sched_barrier(0);
    }
    const float diff = (sv[c] * r - ((a0 + a1) + (a2 + a3))) * b;
    float o0 = 0.0F;
    float o1 = 0.0F;
    float o2 = 0.0F;
    float o3 = 0.0F;
#pragma unroll
    for (int j = 0; j < kHd / 4; ++j) {
      const float4 k4 = reinterpret_cast<const float4*>(sk)[j];
      const float4 q4 = reinterpret_cast<const float4*>(sq)[j];
      st[4 * j] = fmaf(diff, k4.x, st[4 * j]);
      st[4 * j + 1] = fmaf(diff, k4.y, st[4 * j + 1]);
      st[4 * j + 2] = fmaf(diff, k4.z, st[4 * j + 2]);
      st[4 * j + 3] = fmaf(diff, k4.w, st[4 * j + 3]);
      o0 = fmaf(st[4 * j], q4.x, o0);
      o1 = fmaf(st[4 * j + 1], q4.y, o1);
      o2 = fmaf(st[4 * j + 2], q4.z, o2);
      o3 = fmaf(st[4 * j + 3], q4.w, o3);
      __builtin_amdgcn_sched_barrier(0);
    }
    const float ov = ((o0 + o1) + (o2 + o3)) * qscale;
    const float osq = WarpSum(ov * ov);
    if (lane == 0)
      red[0][warp] = osq;
    __syncthreads();
    const float inv = rsqrtf((red[0][0] + red[0][1] + red[0][2] + red[0][3]) /
                                 static_cast<float>(kHd) +
                             eps);
    out[static_cast<std::size_t>(i) * channels + gc] =
        ov * inv * nw * Sigmoid(gate_in);
  }

  {
    float4* row = reinterpret_cast<float4*>(
        state + (static_cast<std::size_t>(h) * kHd + c) * kHd);
#pragma unroll
    for (int j = 0; j < kHd / 4; ++j) {
      row[j] =
          make_float4(st[4 * j], st[4 * j + 1], st[4 * j + 2], st[4 * j + 3]);
    }
  }
  for (int w = 0; w < 3; ++w) {
    for (int t = 0; t < kTaps; ++t) {
      hist[(static_cast<std::size_t>(w) * channels + gc) * kTaps + t] =
          hs[w][t];
    }
  }
}

__global__ __launch_bounds__(256) void F32GemmKernel(
    const float* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int m, int n, int k) {
  constexpr int kTile = 64;
  constexpr int kK = 16;
  __shared__ float ws[kK][kTile + 4];
  __shared__ float xs[kK][kTile + 4];
  const int tx = threadIdx.x & 15;
  const int ty = threadIdx.x >> 4;
  const int row0 = blockIdx.x * kTile;
  const int tok0 = blockIdx.y * kTile;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      acc[i][j] = 0.0F;
  const int lr = threadIdx.x >> 2;       // 0..63 tile row/token
  const int lc = (threadIdx.x & 3) * 4;  // 0,4,8,12 k offset
  for (int k0 = 0; k0 < k; k0 += kK) {
    float4 wv = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    float4 xv = wv;
    if (row0 + lr < m && k0 + lc < k) {
      wv = *reinterpret_cast<const float4*>(
          w + static_cast<std::size_t>(row0 + lr) * k + k0 + lc);
    }
    if (tok0 + lr < n && k0 + lc < k) {
      xv = *reinterpret_cast<const float4*>(
          x + static_cast<std::size_t>(tok0 + lr) * k + k0 + lc);
    }
    ws[lc][lr] = wv.x;
    ws[lc + 1][lr] = wv.y;
    ws[lc + 2][lr] = wv.z;
    ws[lc + 3][lr] = wv.w;
    xs[lc][lr] = xv.x;
    xs[lc + 1][lr] = xv.y;
    xs[lc + 2][lr] = xv.z;
    xs[lc + 3][lr] = xv.w;
    __syncthreads();
#pragma unroll
    for (int c = 0; c < kK; ++c) {
      const float4 w4 = *reinterpret_cast<const float4*>(&ws[c][ty * 4]);
      const float4 x4 = *reinterpret_cast<const float4*>(&xs[c][tx * 4]);
      const float wr[4] = {w4.x, w4.y, w4.z, w4.w};
      const float xr[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[i][j] = fmaf(wr[i], xr[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int gr = row0 + ty * 4 + i;
      const int gt = tok0 + tx * 4 + j;
      if (gr < m && gt < n)
        y[static_cast<std::size_t>(gt) * m + gr] = acc[i][j];
    }
  }
}

struct CopyList {
  const float* src[7];
  float* dst[7];
  int count[7];
};

__global__ void CopyListKernel(CopyList list) {
  const int a = blockIdx.y;
  const int n = list.count[a];
  for (int i = threadIdx.x + blockIdx.x * kThreads; i < n;
       i += gridDim.x * kThreads) {
    list.dst[a][i] = list.src[a][i];
  }
}

__device__ void ScanFill(const std::uint32_t* counts, int experts, int rows,
                         std::int32_t* tiles, int cap, std::uint32_t* scratch) {
  const int e = threadIdx.x;
  const std::uint32_t c = e < experts ? counts[e] : 0U;
  const std::uint32_t padded = (c + 15U) / 16U * 16U;
  const std::uint32_t cnt = (padded + rows - 1) / rows;
  scratch[e] = cnt;
  __syncthreads();
  for (int off = 1; off < 512; off <<= 1) {
    const std::uint32_t v = e >= off ? scratch[e - off] : 0U;
    __syncthreads();
    scratch[e] += v;
    __syncthreads();
  }
  const std::uint32_t end = scratch[e];
  const std::uint32_t begin = end - cnt;
  for (std::uint32_t j = 0; j < cnt; ++j)
    tiles[begin + j] = static_cast<std::int32_t>(e | (j << 16));
  __syncthreads();
  const std::uint32_t total = scratch[511];
  for (int i = static_cast<int>(total) + e; i < cap; i += 512)
    tiles[i] = static_cast<std::int32_t>(0x7FFF0000);
  __syncthreads();
}

__global__ __launch_bounds__(512) void BuildTilesKernel(
    const std::uint32_t* __restrict__ counts, int experts,
    std::int32_t* __restrict__ t48, int cap48, std::int32_t* __restrict__ t64,
    int cap64, std::int32_t* __restrict__ t128, int cap128) {
  __shared__ std::uint32_t scratch[512];
  ScanFill(counts, experts, 48, t48, cap48, scratch);
  ScanFill(counts, experts, 64, t64, cap64, scratch);
  ScanFill(counts, experts, 128, t128, cap128, scratch);
}

constexpr int kPoolRows = 64;

// pool[b][0/1][d] = min/max of the latents of rows [64 b, 64 b + 64).
__global__ void PoolUpdateKernel(const float* __restrict__ lat,
                                 float* __restrict__ pool, int first_block,
                                 int total_rows) {
  const int b = first_block + blockIdx.x;
  const int d = threadIdx.x;
  const int r0 = b * kPoolRows;
  const int r1 = min(r0 + kPoolRows, total_rows);
  float mn = INFINITY;
  float mx = -INFINITY;
  for (int r = r0; r < r1; ++r) {
    const float v = lat[static_cast<std::size_t>(r) * kRaonRank + d];
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  pool[(static_cast<std::size_t>(b) * 2) * kRaonRank + d] = mn;
  pool[(static_cast<std::size_t>(b) * 2 + 1) * kRaonRank + d] = mx;
}

constexpr int kSparseGroups = 8;

// qm[g][h][:] = mean nope query of token group g (n / kSparseGroups tokens).
__global__ void QNopeMeanKernel(const float* __restrict__ q,
                                float* __restrict__ qm, int n, int nope,
                                int rope) {
  const int h = blockIdx.x;
  const int g = blockIdx.y;
  const int i = threadIdx.x;
  const int t0 = g * n / kSparseGroups;
  const int t1 = (g + 1) * n / kSparseGroups;
  float acc = 0.0F;
  for (int t = t0; t < t1; ++t) {
    acc +=
        q[(static_cast<std::size_t>(t) * kRaonHeads + h) * (nope + rope) + i];
  }
  qm[(static_cast<std::size_t>(g) * kRaonHeads + h) * nope + i] =
      acc / static_cast<float>(max(t1 - t0, 1));
}

// scores[b] = max over heads of the Quest bound sum_d max(q_d min_d, q_d
// max_d).
__global__ __launch_bounds__(256) void BlockScoreKernel(
    const float* __restrict__ qabs, const float* __restrict__ pool,
    float* __restrict__ scores, int first_block, int qcount) {
  __shared__ float best[8];
  const int b = first_block + blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float* mn = pool + (static_cast<std::size_t>(b) * 2) * kRaonRank;
  const float* mx = mn + kRaonRank;
  float top = -INFINITY;
  for (int h = warp; h < qcount; h += 8) {
    const float* q = qabs + static_cast<std::size_t>(h) * kRaonRank;
    float acc = 0.0F;
    for (int d = lane; d < kRaonRank; d += 32) {
      acc += fmaxf(q[d] * mn[d], q[d] * mx[d]);
    }
    top = fmaxf(top, WarpSum(acc));
  }
  if (lane == 0)
    best[warp] = top;
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = best[0];
    for (int w = 1; w < 8; ++w)
      v = fmaxf(v, best[w]);
    scores[b] = v;
  }
}

// Picks the k highest scores in [lo, hi) into sel (descending).
__global__ __launch_bounds__(1024) void SelectTopKernel(
    float* __restrict__ scores, int lo, int hi, int k, int* __restrict__ sel) {
  __shared__ float bv[1024];
  __shared__ int bi[1024];
  for (int it = 0; it < k; ++it) {
    float v = -INFINITY;
    int idx = -1;
    for (int i = lo + threadIdx.x; i < hi; i += 1024) {
      if (scores[i] > v) {
        v = scores[i];
        idx = i;
      }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s2 = 512; s2 > 0; s2 >>= 1) {
      if (threadIdx.x < s2 && bv[threadIdx.x + s2] > bv[threadIdx.x]) {
        bv[threadIdx.x] = bv[threadIdx.x + s2];
        bi[threadIdx.x] = bi[threadIdx.x + s2];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      sel[it] = bi[0];
      if (bi[0] >= 0)
        scores[bi[0]] = -INFINITY;
    }
    __syncthreads();
  }
}

// kpos = sink rows, k selected blocks, local rows [local_start, pos0), chunk
// rows.
__global__ void BuildKeyIndexKernel(const int* __restrict__ sel, int k,
                                    int sink_rows, int local_start, int pos0,
                                    int n, int* __restrict__ kpos) {
  const int sel_rows = k * kPoolRows;
  const int local_rows = pos0 - local_start;
  const int total = sink_rows + sel_rows + local_rows + n;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total;
       i += gridDim.x * blockDim.x) {
    int v;
    if (i < sink_rows) {
      v = i;
    } else if (i < sink_rows + sel_rows) {
      const int j = i - sink_rows;
      v = sel[j / kPoolRows] * kPoolRows + (j % kPoolRows);
    } else if (i < sink_rows + sel_rows + local_rows) {
      v = local_start + (i - sink_rows - sel_rows);
    } else {
      v = pos0 + (i - sink_rows - sel_rows - local_rows);
    }
    kpos[i] = v;
  }
}

__global__ void GatherKvFloatKernel(const float* __restrict__ lat,
                                    const float* __restrict__ extra,
                                    const int* __restrict__ kpos,
                                    float* __restrict__ lat_g,
                                    float* __restrict__ extra_g) {
  const int i = blockIdx.x;
  const std::size_t src = static_cast<std::size_t>(kpos[i]);
  for (int d = threadIdx.x; d < kRaonRank; d += blockDim.x) {
    lat_g[static_cast<std::size_t>(i) * kRaonRank + d] =
        lat[src * kRaonRank + d];
  }
  if (threadIdx.x < kRaonRope) {
    extra_g[static_cast<std::size_t>(i) * kRaonRope + threadIdx.x] =
        extra[src * kRaonRope + threadIdx.x];
  }
}

__global__ void GatherKvKernel(const float* __restrict__ lat,
                               const float* __restrict__ extra,
                               const int* __restrict__ kpos,
                               __half* __restrict__ lat16,
                               float* __restrict__ extra_g) {
  const int i = blockIdx.x;
  const std::size_t src = static_cast<std::size_t>(kpos[i]);
  for (int d = threadIdx.x; d < kRaonRank; d += blockDim.x) {
    lat16[static_cast<std::size_t>(i) * kRaonRank + d] =
        __float2half(lat[src * kRaonRank + d]);
  }
  if (threadIdx.x < kRaonRope) {
    extra_g[static_cast<std::size_t>(i) * kRaonRope + threadIdx.x] =
        extra[src * kRaonRope + threadIdx.x];
  }
}

__global__ void WaitFlagKernel(const int* flag, int epoch) {
  while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) <
         epoch) {
    __builtin_amdgcn_s_sleep(8);
  }
}

__global__ void TransposeKbKernel(const __half* __restrict__ kb,
                                  __half* __restrict__ kbt) {
  const std::size_t i =
      threadIdx.x + blockIdx.x * static_cast<std::size_t>(kThreads);
  if (i >= static_cast<std::size_t>(kRaonHeads) * 128 * 512)
    return;
  const std::size_t r = i % 512;
  const std::size_t j = (i / 512) % 128;
  const std::size_t h = i / (512 * 128);
  kbt[i] = kb[(h * 512 + r) * 128 + j];
}

constexpr int kF32SmallMax = 16;
// y[t][r] = w[r] . x[t] for up to 16 tokens; one block per row, so the weight
// row is read once for all tokens.
__global__ __launch_bounds__(256) void F32SmallGemmKernel(
    const float* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int m, int n, int k) {
  __shared__ float red[kF32SmallMax][8];
  const int r = blockIdx.x;
  const float* wr = w + static_cast<std::size_t>(r) * k;
  float acc[kF32SmallMax];
#pragma unroll
  for (int t = 0; t < kF32SmallMax; ++t)
    acc[t] = 0.0F;
  for (int i = threadIdx.x; i < k; i += 256) {
    const float wv = wr[i];
#pragma unroll
    for (int t = 0; t < kF32SmallMax; ++t) {
      if (t < n)
        acc[t] = fmaf(wv, x[static_cast<std::size_t>(t) * k + i], acc[t]);
    }
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int t = 0; t < kF32SmallMax; ++t) {
    const float v = WarpSum(acc[t]);
    if (lane == 0)
      red[t][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x < n) {
    float total = 0.0F;
    for (int wv = 0; wv < 8; ++wv)
      total += red[threadIdx.x][wv];
    y[static_cast<std::size_t>(threadIdx.x) * m + r] = total;
  }
}

__global__ void KvNormKernel(const float* __restrict__ kv,
                             const float* __restrict__ w,
                             float* __restrict__ lat, float* __restrict__ extra,
                             int stride, int rank, int rope, float eps) {
  __shared__ float reduce[kWaves];
  const int t = blockIdx.x;
  const float* in = kv + static_cast<std::size_t>(t) * stride;
  float sq = 0.0F;
  for (int i = threadIdx.x; i < rank; i += kThreads)
    sq = fmaf(in[i], in[i], sq);
  const float inv =
      rsqrtf(BlockSum(sq, reduce) / static_cast<float>(rank) + eps);
  for (int i = threadIdx.x; i < rank; i += kThreads) {
    lat[static_cast<std::size_t>(t) * rank + i] = in[i] * inv * w[i];
  }
  for (int i = threadIdx.x; i < rope; i += kThreads) {
    extra[static_cast<std::size_t>(t) * rope + i] = in[rank + i];
  }
}

__global__ void ReplicateStreamsKernel(const float* __restrict__ src,
                                       float* __restrict__ dst, int h) {
  const std::size_t t = blockIdx.y;
  const int i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= h)
    return;
  const float v = src[t * h + i];
  for (int s = 0; s < kRaonStreams; ++s) {
    dst[(t * kRaonStreams + s) * h + i] = v;
  }
}

constexpr int kMlaSplitThreads = 256;
constexpr int kMlaMaxSplits = 128;

constexpr int kMlaHeadsPerBlock = 8;

constexpr int kMlaTile = 8;
constexpr int kMlaRow = kRaonRank + kRaonRope;
constexpr int kMlaRowPad = kMlaRow + 4;

// Single-pass split-K absorbed MLA decode attention. Same contract as
// the former two-pass kernel, but each latent row is read from global memory
// once: key tiles are staged in LDS, scored for all heads of the block and
// folded into the running {max, sum, acc} before the next tile is loaded.
__global__ __launch_bounds__(kMlaSplitThreads) void MlaSplitTileKernel(
    const float* __restrict__ qabs, const float* __restrict__ qe,
    const float* __restrict__ lat, const float* __restrict__ extra,
    float* __restrict__ part, int heads, int rank, int rope, int pos0,
    int chunk, int splits, float scale) {
  constexpr int kHb = kMlaHeadsPerBlock;
  constexpr int kQ = kRaonRank / 32 + 2;
  constexpr int kLoadsLat = kMlaTile * kRaonRank / 4 / kMlaSplitThreads;
  __shared__ __attribute__((aligned(16))) float Ks[kMlaTile][kMlaRowPad];
  __shared__ float S[kHb][kMlaTile];
  __shared__ float W[kHb][kMlaTile];
  __shared__ float sm_scale[kHb];
  __shared__ float sm_m[kHb];
  __shared__ float sm_l[kHb];
  const int s = blockIdx.x;
  const int h0 = blockIdx.y * kHb;
  const int t = blockIdx.z;
  const int len = pos0 + t + 1;
  const int start = s * chunk;
  if (start >= len) {
    if (threadIdx.x < kHb) {
      float* out =
          part +
          ((static_cast<std::size_t>(t) * heads + h0 + threadIdx.x) * splits +
           s) *
              (rank + 2);
      out[rank] = -INFINITY;
      out[rank + 1] = 0.0F;
    }
    return;
  }
  const int end = min(len, start + chunk);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  float qv[kHb][kQ];
#pragma unroll
  for (int hh = 0; hh < kHb; ++hh) {
    const float* qa =
        qabs + (static_cast<std::size_t>(t) * heads + h0 + hh) * rank;
    const float* qx =
        qe + (static_cast<std::size_t>(t) * heads + h0 + hh) * rope;
#pragma unroll
    for (int j = 0; j < kRaonRank / 32; ++j)
      qv[hh][j] = qa[lane + 32 * j];
    qv[hh][kRaonRank / 32] = qx[lane];
    qv[hh][kRaonRank / 32 + 1] = qx[lane + 32];
  }
  float a0[kHb];
  float a1[kHb];
#pragma unroll
  for (int hh = 0; hh < kHb; ++hh) {
    a0[hh] = 0.0F;
    a1[hh] = 0.0F;
  }
  if (threadIdx.x < kHb) {
    sm_m[threadIdx.x] = -INFINITY;
    sm_l[threadIdx.x] = 0.0F;
  }

  float4 pre_lat[kLoadsLat];
  float4 pre_rope = make_float4(0.F, 0.F, 0.F, 0.F);
  auto fetch = [&](int base) {
#pragma unroll
    for (int i = 0; i < kLoadsLat; ++i) {
      const int idx = threadIdx.x + i * kMlaSplitThreads;
      const int k = idx / (kRaonRank / 4);
      const int c = idx % (kRaonRank / 4);
      const int p = base + k;
      pre_lat[i] = p < end
                       ? *reinterpret_cast<const float4*>(
                             lat + static_cast<std::size_t>(p) * rank + c * 4)
                       : make_float4(0.F, 0.F, 0.F, 0.F);
    }
    const int k = threadIdx.x / (kRaonRope / 4);
    const int c = threadIdx.x % (kRaonRope / 4);
    const int p = base + k;
    pre_rope = k < kMlaTile && p < end
                   ? *reinterpret_cast<const float4*>(
                         extra + static_cast<std::size_t>(p) * rope + c * 4)
                   : make_float4(0.F, 0.F, 0.F, 0.F);
  };
  auto stage = [&]() {
#pragma unroll
    for (int i = 0; i < kLoadsLat; ++i) {
      const int idx = threadIdx.x + i * kMlaSplitThreads;
      const int k = idx / (kRaonRank / 4);
      const int c = idx % (kRaonRank / 4);
      *reinterpret_cast<float4*>(&Ks[k][c * 4]) = pre_lat[i];
    }
    const int k = threadIdx.x / (kRaonRope / 4);
    const int c = threadIdx.x % (kRaonRope / 4);
    if (k < kMlaTile)
      *reinterpret_cast<float4*>(&Ks[k][kRaonRank + c * 4]) = pre_rope;
  };

  fetch(start);
  for (int base = start; base < end; base += kMlaTile) {
    stage();
    __syncthreads();
    if (base + kMlaTile < end)
      fetch(base + kMlaTile);
#pragma unroll
    for (int kk = 0; kk < kMlaTile / (kMlaSplitThreads / 32); ++kk) {
      const int key = warp * (kMlaTile / (kMlaSplitThreads / 32)) + kk;
      float kv[kQ];
#pragma unroll
      for (int j = 0; j < kRaonRank / 32; ++j)
        kv[j] = Ks[key][lane + 32 * j];
      kv[kRaonRank / 32] = Ks[key][kRaonRank + lane];
      kv[kRaonRank / 32 + 1] = Ks[key][kRaonRank + lane + 32];
#pragma unroll
      for (int hh = 0; hh < kHb; ++hh) {
        float dot = 0.0F;
#pragma unroll
        for (int j = 0; j < kQ; ++j)
          dot = fmaf(qv[hh][j], kv[j], dot);
        const float score = WarpSum(dot) * scale;
        if (lane == 0)
          S[hh][key] = score;
      }
    }
    __syncthreads();
    if (threadIdx.x < kHb) {
      const int hh = threadIdx.x;
      const int valid = min(kMlaTile, end - base);
      float tm = -INFINITY;
      for (int k = 0; k < valid; ++k)
        tm = fmaxf(tm, S[hh][k]);
      const float mold = sm_m[hh];
      const float mnew = fmaxf(mold, tm);
      const float sc = mold == -INFINITY ? 0.0F : expf(mold - mnew);
      float l = sm_l[hh] * sc;
      for (int k = 0; k < kMlaTile; ++k) {
        const float w = k < valid ? expf(S[hh][k] - mnew) : 0.0F;
        W[hh][k] = w;
        l += w;
      }
      sm_scale[hh] = sc;
      sm_m[hh] = mnew;
      sm_l[hh] = l;
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < kHb; ++hh) {
      a0[hh] *= sm_scale[hh];
      a1[hh] *= sm_scale[hh];
    }
#pragma unroll
    for (int k = 0; k < kMlaTile; ++k) {
      const float l0 = Ks[k][threadIdx.x];
      const float l1 = Ks[k][threadIdx.x + kMlaSplitThreads];
#pragma unroll
      for (int hh = 0; hh < kHb; ++hh) {
        const float w = W[hh][k];
        a0[hh] = fmaf(w, l0, a0[hh]);
        a1[hh] = fmaf(w, l1, a1[hh]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int hh = 0; hh < kHb; ++hh) {
    float* out =
        part + ((static_cast<std::size_t>(t) * heads + h0 + hh) * splits + s) *
                   (rank + 2);
    out[threadIdx.x] = a0[hh];
    out[threadIdx.x + kMlaSplitThreads] = a1[hh];
    if (threadIdx.x == 0) {
      out[rank] = sm_m[hh];
      out[rank + 1] = sm_l[hh];
    }
  }
}


using mw_v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using mw_v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kMwRows = 32;
constexpr int kMwKeys = 16;
constexpr int kMwDim = kRaonRank + kRaonRope;
constexpr int kMwKs = kMwDim + 8;
constexpr int kMwVs = 24;
constexpr int kMwPs = 24;

__global__ void MlaQHalfKernel(const float* __restrict__ qabs,
                               const float* __restrict__ qe,
                               _Float16* __restrict__ out, int rank,
                               int rope) {
  const std::size_t row = blockIdx.x;
  for (int i = threadIdx.x; i < rank + rope; i += blockDim.x) {
    const float v = i < rank ? qabs[row * rank + i] : qe[row * rope + i - rank];
    out[row * (rank + rope) + i] = static_cast<_Float16>(v);
  }
}

// Absorbed MLA attention on WMMA. A block owns 32 (token, head) rows and one
// key split; the latent keys are staged once per 16-key tile and shared by
// every row. Writes the same partials as MlaSplitTileKernel.
__global__ __launch_bounds__(256) void MlaSplitWmmaKernel(
    const _Float16* __restrict__ q16, const float* __restrict__ lat,
    const float* __restrict__ extra, float* __restrict__ part, int heads,
    int nrows, int pos0, int chunk, int splits, float scale) {
  constexpr int rank = kRaonRank;
  constexpr int rope = kRaonRope;
  __shared__ __attribute__((aligned(16))) _Float16 Ks[kMwKeys][kMwKs];
  __shared__ __attribute__((aligned(16))) _Float16 Vt[rank][kMwVs];
  __shared__ __attribute__((aligned(16))) float Sp[4][kMwRows][kMwKeys];
  __shared__ __attribute__((aligned(16))) _Float16 Ps[kMwRows][kMwPs];
  __shared__ float corr_s[kMwRows];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int col = lane & 15;
  const int hid = lane >> 4;
  const int s = blockIdx.x;
  const int row0 = blockIdx.y * kMwRows;
  const int rlast = min(row0 + kMwRows, nrows) - 1;
  const int len_max = pos0 + rlast / heads + 1;
  const int start = s * chunk;
  const int srow = tid >> 3;
  const int sub = tid & 7;
  const int grow = row0 + srow;
  float* base_out = part + (static_cast<std::size_t>(grow) * splits + s) *
                               (rank + 2);
  if (start >= len_max) {
    if (sub == 0 && grow < nrows) {
      base_out[rank] = -INFINITY;
      base_out[rank + 1] = 0.0F;
    }
    return;
  }
  const int end = min(len_max, start + chunk);
  const int len_row =
      grow < nrows ? pos0 + grow / heads + 1 : 0;
  const int limit = min(len_row, end);

  const int rt = wave & 1;
  const int qq = wave >> 1;
  mw_v16h qa[9];
#pragma unroll
  for (int j = 0; j < 9; ++j) {
    const int r = row0 + rt * 16 + col;
    qa[j] = r < nrows ? *reinterpret_cast<const mw_v16h*>(
                            q16 + static_cast<std::size_t>(r) * kMwDim +
                            (qq * 9 + j) * 16)
                      : mw_v16h{};
  }
  mw_v8f o[2][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
      o[a][b] = mw_v8f{};
  float m = -INFINITY;
  float lsum = 0.0F;

  float4 pre_lat[8];
  float4 pre_rope;
  auto fetch = [&](int base) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = tid + i * 256;
      const int k = idx >> 7;
      const int c = idx & 127;
      const int p = base + k;
      pre_lat[i] = p < end ? *reinterpret_cast<const float4*>(
                                 lat + static_cast<std::size_t>(p) * rank +
                                 c * 4)
                           : make_float4(0.F, 0.F, 0.F, 0.F);
    }
    const int k = tid >> 4;
    const int c = tid & 15;
    const int p = base + k;
    pre_rope = p < end ? *reinterpret_cast<const float4*>(
                             extra + static_cast<std::size_t>(p) * rope + c * 4)
                       : make_float4(0.F, 0.F, 0.F, 0.F);
  };
  auto stage = [&]() {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = tid + i * 256;
      const int k = idx >> 7;
      const int c = idx & 127;
      const _Float16 h0 = static_cast<_Float16>(pre_lat[i].x);
      const _Float16 h1 = static_cast<_Float16>(pre_lat[i].y);
      const _Float16 h2 = static_cast<_Float16>(pre_lat[i].z);
      const _Float16 h3 = static_cast<_Float16>(pre_lat[i].w);
      Ks[k][c * 4 + 0] = h0;
      Ks[k][c * 4 + 1] = h1;
      Ks[k][c * 4 + 2] = h2;
      Ks[k][c * 4 + 3] = h3;
      Vt[c * 4 + 0][k] = h0;
      Vt[c * 4 + 1][k] = h1;
      Vt[c * 4 + 2][k] = h2;
      Vt[c * 4 + 3][k] = h3;
    }
    const int k = tid >> 4;
    const int c = tid & 15;
    Ks[k][rank + c * 4 + 0] = static_cast<_Float16>(pre_rope.x);
    Ks[k][rank + c * 4 + 1] = static_cast<_Float16>(pre_rope.y);
    Ks[k][rank + c * 4 + 2] = static_cast<_Float16>(pre_rope.z);
    Ks[k][rank + c * 4 + 3] = static_cast<_Float16>(pre_rope.w);
  };

  fetch(start);
  for (int base = start; base < end; base += kMwKeys) {
    __syncthreads();
    stage();
    __syncthreads();
    if (base + kMwKeys < end)
      fetch(base + kMwKeys);
    {
      mw_v8f acc = mw_v8f{};
#pragma unroll
      for (int j = 0; j < 9; ++j) {
        const mw_v16h b = *reinterpret_cast<const mw_v16h*>(
            &Ks[col][(qq * 9 + j) * 16]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[j], b, acc);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        Sp[qq][rt * 16 + 2 * i + hid][col] = acc[i];
    }
    __syncthreads();
    {
      float v[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = sub * 2 + e;
        const float sc = (Sp[0][srow][key] + Sp[1][srow][key] +
                          Sp[2][srow][key] + Sp[3][srow][key]) *
                         scale;
        v[e] = base + key < limit ? sc : -INFINITY;
      }
      float mx = fmaxf(v[0], v[1]);
      mx = fmaxf(mx, __shfl_xor(mx, 1, 8));
      mx = fmaxf(mx, __shfl_xor(mx, 2, 8));
      mx = fmaxf(mx, __shfl_xor(mx, 4, 8));
      const float mn = fmaxf(m, mx);
      const float corr = mn == -INFINITY ? 1.0F : __expf(m - mn);
      float psum = 0.0F;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float p = v[e] == -INFINITY ? 0.0F : __expf(v[e] - mn);
        psum += p;
        Ps[srow][sub * 2 + e] = static_cast<_Float16>(p);
      }
      lsum = lsum * corr + psum;
      m = mn;
      if (sub == 0)
        corr_s[srow] = corr;
    }
    __syncthreads();
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const mw_v16h pa =
          *reinterpret_cast<const mw_v16h*>(&Ps[a * 16 + col][0]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float c = corr_s[a * 16 + 2 * i + hid];
#pragma unroll
        for (int b = 0; b < 4; ++b)
          o[a][b][i] *= c;
      }
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const mw_v16h vb = *reinterpret_cast<const mw_v16h*>(
            &Vt[wave * 64 + b * 16 + col][0]);
        o[a][b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, o[a][b]);
      }
    }
  }
  lsum += __shfl_xor(lsum, 1, 8);
  lsum += __shfl_xor(lsum, 2, 8);
  lsum += __shfl_xor(lsum, 4, 8);
  if (sub == 0 && grow < nrows) {
    base_out[rank] = m;
    base_out[rank + 1] = lsum;
  }
#pragma unroll
  for (int a = 0; a < 2; ++a) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = row0 + a * 16 + 2 * i + hid;
      if (r >= nrows)
        continue;
      float* out = part + (static_cast<std::size_t>(r) * splits + s) *
                              (rank + 2);
#pragma unroll
      for (int b = 0; b < 4; ++b)
        out[wave * 64 + b * 16 + col] = o[a][b][i];
    }
  }
}

__global__ __launch_bounds__(kMlaSplitThreads) void MlaCombineKernel(
    const float* __restrict__ part, float* __restrict__ ctx, int heads,
    int rank, int splits) {
  const int h = blockIdx.x;
  const int t = blockIdx.y;
  const float* base =
      part + (static_cast<std::size_t>(t) * heads + h) * splits * (rank + 2);
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s)
    m = fmaxf(m, base[s * (rank + 2) + rank]);
  float z = 0.0F;
  float a0 = 0.0F;
  float a1 = 0.0F;
  for (int s = 0; s < splits; ++s) {
    const float* ps = base + s * (rank + 2);
    const float ms = ps[rank];
    if (ms == -INFINITY)
      continue;
    const float f = expf(ms - m);
    z = fmaf(ps[rank + 1], f, z);
    a0 = fmaf(ps[threadIdx.x], f, a0);
    a1 = fmaf(ps[threadIdx.x + kMlaSplitThreads], f, a1);
  }
  float* out = ctx + (static_cast<std::size_t>(t) * heads + h) * rank;
  const float inv = 1.0F / z;
  out[threadIdx.x] = a0 * inv;
  out[threadIdx.x + kMlaSplitThreads] = a1 * inv;
}

__global__ void MlaAttendKernel(const float* __restrict__ qabs,
                                const float* __restrict__ qe,
                                const float* __restrict__ lat,
                                const float* __restrict__ extra,
                                float* __restrict__ ctx, int rank, int rope,
                                int n, float scale) {
  __shared__ float acc[kRaonRank];
  __shared__ float reduce[kWaves];
  const int h = blockIdx.x;
  const float* qa = qabs + static_cast<std::size_t>(h) * rank;
  const float* qx = qe + static_cast<std::size_t>(h) * rope;
  float* out = ctx + static_cast<std::size_t>(h) * rank;
  float mx = -INFINITY;
  float sum = 0.0F;
  for (int i = threadIdx.x; i < rank; i += kThreads)
    acc[i] = 0.0F;
  __syncthreads();
  for (int pos = 0; pos < n; ++pos) {
    const float* lp = lat + static_cast<std::size_t>(pos) * rank;
    const float* ep = extra + static_cast<std::size_t>(pos) * rope;
    float dot = 0.0F;
    for (int i = threadIdx.x; i < rank; i += kThreads)
      dot = fmaf(qa[i], lp[i], dot);
    for (int i = threadIdx.x; i < rope; i += kThreads)
      dot = fmaf(qx[i], ep[i], dot);
    float score = BlockSum(dot, reduce) * scale;
    const float nmx = fmaxf(mx, score);
    const float rescale = expf(mx - nmx);
    mx = nmx;
    const float w = expf(score - mx);
    sum = sum * rescale + w;
    for (int i = threadIdx.x; i < rank; i += kThreads) {
      acc[i] = acc[i] * rescale + fmaf(w, lp[i], 0.0F);
    }
    __syncthreads();
  }
  const float inv = 1.0F / sum;
  for (int i = threadIdx.x; i < rank; i += kThreads)
    out[i] = acc[i] * inv;
}

__global__ void MoeRouteKernel(const float* __restrict__ logits,
                               const float* __restrict__ bias, float scale,
                               int* __restrict__ ids,
                               float* __restrict__ weights, int experts,
                               int used, bool normalize) {
  __shared__ float prob[kRaonExperts];
  __shared__ float sel[kRaonExperts];
  __shared__ bool taken[kRaonExperts];
  __shared__ float best_v[kThreads];
  __shared__ int best_i[kThreads];
  __shared__ int chosen[kRaonUsed];
  const int t = threadIdx.x;
  logits += static_cast<std::size_t>(blockIdx.x) * experts;
  ids += static_cast<std::size_t>(blockIdx.x) * used;
  weights += static_cast<std::size_t>(blockIdx.x) * used;
  for (int e = t; e < experts; e += kThreads) {
    prob[e] = Sigmoid(logits[e]);
    sel[e] = prob[e] + bias[e];
    taken[e] = false;
  }
  __syncthreads();
  for (int r = 0; r < used; ++r) {
    float bv = -INFINITY;
    int bi = experts;
    if (t < experts && !taken[t]) {
      bv = sel[t];
      bi = t;
    }
    best_v[t] = bv;
    best_i[t] = bi;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
      if (t < s) {
        if (best_v[t + s] > best_v[t]) {
          best_v[t] = best_v[t + s];
          best_i[t] = best_i[t + s];
        }
      }
      __syncthreads();
    }
    if (t == 0)
      chosen[r] = best_i[0];
    if (t == best_i[0] && t < experts)
      taken[t] = true;
    __syncthreads();
  }
  __shared__ float wsum;
  if (t == 0) {
    float s = normalize ? 1e-20F : 0.0F;
    for (int r = 0; r < used; ++r) {
      if (!normalize)
        s = 1.0F;
      else
        s += prob[chosen[r]];
    }
    wsum = s;
    for (int r = 0; r < used; ++r) {
      ids[r] = chosen[r];
      weights[r] = prob[chosen[r]] / wsum * scale;
    }
  }
}

__global__ void SituActKernel(const float* __restrict__ g,
                              const float* __restrict__ u,
                              float* __restrict__ y, int n, float beta,
                              float lb) {
  const int i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= n)
    return;
  const float gg = tanhf(g[i] / beta) * beta * Sigmoid(g[i]);
  const float uu = tanhf(u[i] / lb) * lb;
  y[i] = gg * uu;
}

__global__ void SituActHalfKernel(const float* __restrict__ g,
                                  const float* __restrict__ u,
                                  __half* __restrict__ y, std::size_t n,
                                  float beta, float lb) {
  const std::size_t i =
      threadIdx.x + blockIdx.x * static_cast<std::size_t>(kThreads);
  if (i >= n)
    return;
  const float gg = tanhf(g[i] / beta) * beta * Sigmoid(g[i]);
  const float uu = tanhf(u[i] / lb) * lb;
  y[i] = __float2half(gg * uu);
}

__global__ void ToHalfKernel(const float* __restrict__ x,
                             __half* __restrict__ y, std::size_t n) {
  const std::size_t i =
      threadIdx.x + blockIdx.x * static_cast<std::size_t>(kThreads);
  if (i < n)
    y[i] = __float2half(x[i]);
}

__global__ void MoeCombineHalfKernel(const __half* __restrict__ parts,
                                     const float* __restrict__ weights,
                                     float* __restrict__ out, int n, int used) {
  const int t = blockIdx.y;
  const int i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= n)
    return;
  float acc = 0.0F;
  for (int r = 0; r < used; ++r) {
    acc = fmaf(
        weights[static_cast<std::size_t>(t) * used + r],
        __half2float(parts[(static_cast<std::size_t>(t) * used + r) * n + i]),
        acc);
  }
  out[static_cast<std::size_t>(t) * n + i] = acc;
}

__global__ void Q5ToQ8Kernel(const std::uint8_t* __restrict__ src,
                             std::uint8_t* __restrict__ dst,
                             std::size_t blocks) {
  const std::size_t b =
      threadIdx.x + blockIdx.x * static_cast<std::size_t>(kThreads);
  if (b >= blocks)
    return;
  const std::uint8_t* in = src + b * 22;
  std::uint8_t* out = dst + b * 34;
  out[0] = in[0];
  out[1] = in[1];
  std::uint32_t qh = in[2] | (in[3] << 8) | (in[4] << 16) |
                     (static_cast<std::uint32_t>(in[5]) << 24);
  for (int j = 0; j < 16; ++j) {
    const int lo = (in[6 + j] & 0xF) | (((qh >> j) & 1) << 4);
    const int hi = (in[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
    out[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(lo - 16));
    out[2 + j + 16] =
        static_cast<std::uint8_t>(static_cast<std::int8_t>(hi - 16));
  }
}

__global__ void MoeCombineKernel(const float* __restrict__ parts,
                                 const float* __restrict__ weights,
                                 float* __restrict__ out, int n, int used) {
  const int i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= n)
    return;
  float acc = 0.0F;
  for (int r = 0; r < used; ++r) {
    acc = fmaf(weights[r], parts[static_cast<std::size_t>(r) * n + i], acc);
  }
  out[i] = acc;
}

__global__ void SteerAddKernel(float* __restrict__ x,
                               const float* __restrict__ steer, float scale,
                               int n) {
  const int i = threadIdx.x + blockIdx.x * kThreads;
  if (i < n)
    x[i] = fmaf(scale, steer[i], x[i]);
}

__global__ void VectorMulKernel(const float* __restrict__ x,
                                const float* __restrict__ gate,
                                float* __restrict__ out, int n) {
  const int i = threadIdx.x + blockIdx.x * kThreads;
  if (i < n)
    out[i] = x[i] * Sigmoid(gate[i]);
}

__global__ void AddVectorsKernel(const float* __restrict__ x,
                                 const float* __restrict__ y,
                                 float* __restrict__ out, int n) {
  const int i = threadIdx.x + blockIdx.x * kThreads;
  if (i < n)
    out[i] = x[i] + y[i];
}

__global__ void MlaQeGatherKernel(const float* __restrict__ q,
                                  float* __restrict__ qe,
                                  float* __restrict__ qn, int nope, int rope) {
  const int h = blockIdx.x + blockIdx.y * gridDim.x;
  const int i = threadIdx.x;
  if (qn != nullptr && i < nope) {
    qn[static_cast<std::size_t>(h) * nope + i] =
        q[static_cast<std::size_t>(h) * (nope + rope) + i];
  }
  if (i < rope) {
    qe[static_cast<std::size_t>(h) * rope + i] =
        q[static_cast<std::size_t>(h) * (nope + rope) + nope + i];
  }
}

__global__ void HeadMeanNormKernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   float* __restrict__ y, int h, int streams,
                                   float eps) {
  __shared__ float reduce[kWaves];
  float sq = 0.0F;
  for (int i = threadIdx.x; i < h; i += kThreads) {
    float acc = 0.0F;
    for (int s = 0; s < streams; ++s) {
      acc += x[static_cast<std::size_t>(s) * h + i];
    }
    const float v = acc / static_cast<float>(streams);
    y[i] = v;
    sq = fmaf(v, v, sq);
  }
  const float inv = rsqrtf(BlockSum(sq, reduce) / static_cast<float>(h) + eps);
  for (int i = threadIdx.x; i < h; i += kThreads)
    y[i] *= inv * w[i];
}

__global__ void ArgmaxRowsKernel(const float* __restrict__ logits, int vocab,
                                 int* __restrict__ out) {
  __shared__ float bv[kThreads];
  __shared__ int bi[kThreads];
  const int t = threadIdx.x;
  logits += static_cast<std::size_t>(blockIdx.x) * vocab;
  float v = -INFINITY;
  int idx = 0;
  for (int i = t; i < vocab; i += kThreads) {
    if (logits[i] > v) {
      v = logits[i];
      idx = i;
    }
  }
  bv[t] = v;
  bi[t] = idx;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (t < s) {
      if (bv[t + s] > bv[t]) {
        bv[t] = bv[t + s];
        bi[t] = bi[t + s];
      }
    }
    __syncthreads();
  }
  if (t == 0)
    out[blockIdx.x] = bi[0];
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kWQ = 64;
constexpr int kWK = 32;
constexpr int kKS = 200;
constexpr int kVS = 40;
constexpr int kPS = 40;

__global__ __launch_bounds__(128) void MlaWmmaAttnKernel(
    const __half* __restrict__ q16, const __half* __restrict__ k16,
    const __half* __restrict__ v16, float* __restrict__ out, std::uint32_t n,
    std::uint32_t pos0, float scale, const int* __restrict__ kpos,
    std::uint32_t nk) {
  __shared__ __attribute__((aligned(16))) _Float16 Ks[kWK][kKS];
  __shared__ __attribute__((aligned(16))) _Float16 Vt[128][kVS];
  __shared__ __attribute__((aligned(16))) _Float16 Ps[4][16][kPS];
  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int h = blockIdx.y;
  const std::uint32_t q0 = blockIdx.x * kWQ;
  const std::uint32_t wq0 = q0 + wave * 16;

  v16h qa[12];
  {
    const std::uint32_t t = wq0 + col;
    for (int ks = 0; ks < 12; ++ks) {
      if (t < n) {
        qa[ks] = *reinterpret_cast<const v16h*>(
            q16 + (static_cast<std::size_t>(t) * kRaonHeads + h) * 192 +
            ks * 16);
      } else {
        qa[ks] = v16h{};
      }
    }
  }
  v8f o[8];
  for (int dt = 0; dt < 8; ++dt)
    o[dt] = v8f{};
  float m[8];
  float lsum[8];
  for (int i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    lsum[i] = 0.0F;
  }
  const std::uint32_t last_q = min(q0 + kWQ, n) - 1;
  const std::uint32_t key_end = kpos != nullptr ? nk : pos0 + last_q + 1;
  for (std::uint32_t k0 = 0; k0 < key_end; k0 += kWK) {
    __syncthreads();
    for (int idx = tid; idx < kWK * 24; idx += 128) {
      const int r = idx / 24;
      const int c8 = idx % 24;
      const std::uint32_t s = k0 + r;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (s < key_end)
        val = *reinterpret_cast<const uint4*>(
            k16 + (static_cast<std::size_t>(s) * kRaonHeads + h) * 192 +
            c8 * 8);
      *reinterpret_cast<uint4*>(&Ks[r][c8 * 8]) = val;
    }
    for (int idx = tid; idx < kWK * 16; idx += 128) {
      const int r = idx / 16;
      const int c8 = idx % 16;
      const std::uint32_t s = k0 + r;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (s < key_end)
        val = *reinterpret_cast<const uint4*>(
            v16 + (static_cast<std::size_t>(s) * kRaonHeads + h) * 128 +
            c8 * 8);
      const _Float16* hv = reinterpret_cast<const _Float16*>(&val);
      for (int j = 0; j < 8; ++j)
        Vt[c8 * 8 + j][r] = hv[j];
    }
    __syncthreads();
    v8f sc[2];
    for (int kt = 0; kt < 2; ++kt) {
      v8f acc = v8f{};
      for (int ks = 0; ks < 12; ++ks) {
        const v16h b =
            *reinterpret_cast<const v16h*>(&Ks[kt * 16 + col][ks * 16]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], b, acc);
      }
      sc[kt] = acc;
    }
    float corr[8];
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t t = wq0 + 2 * i + half_id;
      float v0 = sc[0][i] * scale;
      float v1 = sc[1][i] * scale;
      const std::uint32_t s0 = k0 + col;
      const std::uint32_t s1 = k0 + 16 + col;
      const bool ok_row = t < n;
      if (!(ok_row && s0 < key_end &&
            (kpos != nullptr ? static_cast<std::uint32_t>(kpos[s0]) : s0) <=
                pos0 + t))
        v0 = -INFINITY;
      if (!(ok_row && s1 < key_end &&
            (kpos != nullptr ? static_cast<std::uint32_t>(kpos[s1]) : s1) <=
                pos0 + t))
        v1 = -INFINITY;
      float mx = fmaxf(v0, v1);
      for (int off = 8; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor(mx, off, 16));
      const float mn = fmaxf(m[i], mx);
      corr[i] = mn == -INFINITY ? 1.0F : __expf(m[i] - mn);
      const float p0 = v0 == -INFINITY ? 0.0F : __expf(v0 - mn);
      const float p1 = v1 == -INFINITY ? 0.0F : __expf(v1 - mn);
      lsum[i] = lsum[i] * corr[i] + p0 + p1;
      m[i] = mn;
      Ps[wave][2 * i + half_id][col] = static_cast<_Float16>(p0);
      Ps[wave][2 * i + half_id][16 + col] = static_cast<_Float16>(p1);
    }
    for (int dt = 0; dt < 8; ++dt)
      for (int i = 0; i < 8; ++i)
        o[dt][i] *= corr[i];
    __syncthreads();
    v16h pa[2];
    for (int ks = 0; ks < 2; ++ks)
      pa[ks] = *reinterpret_cast<const v16h*>(&Ps[wave][col][ks * 16]);
    for (int dt = 0; dt < 8; ++dt) {
      for (int ks = 0; ks < 2; ++ks) {
        const v16h b =
            *reinterpret_cast<const v16h*>(&Vt[dt * 16 + col][ks * 16]);
        o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa[ks], b, o[dt]);
      }
    }
  }
  for (int i = 0; i < 8; ++i) {
    float l = lsum[i];
    for (int off = 8; off > 0; off >>= 1)
      l += __shfl_xor(l, off, 16);
    const std::uint32_t t = wq0 + 2 * i + half_id;
    if (t >= n)
      continue;
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    for (int dt = 0; dt < 8; ++dt)
      out[(static_cast<std::size_t>(t) * kRaonHeads + h) * 128 + dt * 16 +
          col] = o[dt][i] * inv;
  }
}

__global__ __launch_bounds__(256)
    __attribute__((amdgpu_waves_per_eu(6, 6))) void MlaWmmaAttnFastKernel(
        const __half* __restrict__ q16, const __half* __restrict__ k16,
        const __half* __restrict__ v16, float* __restrict__ out,
        std::uint32_t n, std::uint32_t pos0, float scale, std::uint32_t nk) {
  constexpr int QW = 8;
  constexpr int TK = 32;
  constexpr int kTh = QW * 32;
  constexpr int kQ = QW * 16;
  constexpr int NT = TK / 16;
  constexpr int VS = TK + 8;
  __shared__ __attribute__((aligned(16))) _Float16 Ks[TK][200];
  __shared__ __attribute__((aligned(16))) _Float16 Vt[128][VS];
  __shared__ __attribute__((aligned(16))) _Float16 Ps[QW][16][VS];
  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int h = blockIdx.y;
  const std::uint32_t q0 = blockIdx.x * kQ;
  const std::uint32_t wq0 = q0 + wave * 16;
  const float sl2 = scale * 1.4426950408889634F;
  v16h qa[12];
  {
    const std::uint32_t t = wq0 + col;
    for (int ks = 0; ks < 12; ++ks) {
      qa[ks] =
          t < n
              ? *reinterpret_cast<const v16h*>(
                    q16 + (static_cast<std::size_t>(t) * kRaonHeads + h) * 192 +
                    ks * 16)
              : v16h{};
    }
  }
  v8f o[8];
  for (int dt = 0; dt < 8; ++dt)
    o[dt] = v8f{};
  float m[8], lsum[8];
  for (int i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    lsum[i] = 0.0F;
  }
  const std::uint32_t last_q = min(q0 + kQ, n) - 1;
  const std::uint32_t key_end = pos0 + last_q + 1;
  const bool full_rows = q0 + kQ <= n;
  constexpr int kKL = TK * 24 / kTh;
  constexpr int kVL = 128 * (TK / 8) / kTh;
  uint4 pk[kKL], pv[kVL];
  auto fetch = [&](std::uint32_t k0) {
#pragma unroll
    for (int i = 0; i < kKL; ++i) {
      const int idx = tid + i * kTh;
      const int r = idx / 24, c8 = idx % 24;
      const std::uint32_t s = k0 + r;
      pk[i] = make_uint4(0, 0, 0, 0);
      if (s < key_end)
        pk[i] = *reinterpret_cast<const uint4*>(
            k16 + (static_cast<std::size_t>(s) * kRaonHeads + h) * 192 +
            c8 * 8);
    }
#pragma unroll
    for (int i = 0; i < kVL; ++i) {
      const int idx = tid + i * kTh;
      const int c8 = idx / TK, r = idx % TK;
      const std::uint32_t s = k0 + r;
      pv[i] = make_uint4(0, 0, 0, 0);
      if (s < key_end)
        pv[i] = *reinterpret_cast<const uint4*>(
            v16 + (static_cast<std::size_t>(s) * kRaonHeads + h) * 128 +
            c8 * 8);
    }
  };
  auto stage = [&]() {
#pragma unroll
    for (int i = 0; i < kKL; ++i) {
      const int idx = tid + i * kTh;
      const int r = idx / 24, c8 = idx % 24;
      *reinterpret_cast<uint4*>(&Ks[r][c8 * 8]) = pk[i];
    }
#pragma unroll
    for (int i = 0; i < kVL; ++i) {
      const int idx = tid + i * kTh;
      const int c8 = idx / TK, r = idx % TK;
      const _Float16* hv = reinterpret_cast<const _Float16*>(&pv[i]);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        Vt[c8 * 8 + j][r] = hv[j];
    }
  };
  for (std::uint32_t k0 = 0; k0 < key_end; k0 += TK) {
    __syncthreads();
    fetch(k0);
    stage();
    __syncthreads();
    v8f sc[NT];
#pragma unroll
    for (int kt = 0; kt < NT; ++kt) {
      v8f acc = v8f{};
#pragma unroll
      for (int ks = 0; ks < 12; ++ks) {
        const v16h b =
            *reinterpret_cast<const v16h*>(&Ks[kt * 16 + col][ks * 16]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], b, acc);
      }
      sc[kt] = acc;
    }
    const bool fast =
        full_rows && (k0 + TK <= key_end) && (k0 + TK - 1 <= pos0 + q0);
    float corr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t t = wq0 + 2 * i + half_id;
      float v[NT];
#pragma unroll
      for (int kt = 0; kt < NT; ++kt)
        v[kt] = sc[kt][i] * sl2;
      if (!fast) {
#pragma unroll
        for (int kt = 0; kt < NT; ++kt) {
          const std::uint32_t s0 = k0 + kt * 16 + col;
          if (!(t < n && s0 < key_end && s0 <= pos0 + t))
            v[kt] = -INFINITY;
        }
      }
      float mx = v[0];
#pragma unroll
      for (int kt = 1; kt < NT; ++kt)
        mx = fmaxf(mx, v[kt]);
      for (int off = 8; off > 0; off >>= 1)
        mx = fmaxf(mx, glm53_flash::rocm::XorLane(mx, off));
      const float mn = fmaxf(m[i], mx);
      corr[i] = mn == -INFINITY ? 1.0F : exp2f(m[i] - mn);
      float ps = 0.0F;
#pragma unroll
      for (int kt = 0; kt < NT; ++kt) {
        const float p = v[kt] == -INFINITY ? 0.0F : exp2f(v[kt] - mn);
        ps += p;
        Ps[wave][2 * i + half_id][kt * 16 + col] = static_cast<_Float16>(p);
      }
      lsum[i] = lsum[i] * corr[i] + ps;
      m[i] = mn;
    }
    bool need = false;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      need |= corr[i] != 1.0F;
    if (__builtin_amdgcn_ballot_w32(need) != 0) {
#pragma unroll
      for (int dt = 0; dt < 8; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          o[dt][i] *= corr[i];
    }
    __builtin_amdgcn_wave_barrier();
    v16h pa[NT];
#pragma unroll
    for (int ks = 0; ks < NT; ++ks)
      pa[ks] = *reinterpret_cast<const v16h*>(&Ps[wave][col][ks * 16]);
#pragma unroll
    for (int dt = 0; dt < 8; ++dt)
#pragma unroll
      for (int ks = 0; ks < NT; ++ks) {
        const v16h b =
            *reinterpret_cast<const v16h*>(&Vt[dt * 16 + col][ks * 16]);
        o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa[ks], b, o[dt]);
      }
  }
  for (int i = 0; i < 8; ++i) {
    float l = lsum[i];
    for (int off = 8; off > 0; off >>= 1)
      l += glm53_flash::rocm::XorLane(l, off);
    const std::uint32_t t = wq0 + 2 * i + half_id;
    if (t >= n)
      continue;
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    for (int dt = 0; dt < 8; ++dt)
      out[(static_cast<std::size_t>(t) * kRaonHeads + h) * 128 + dt * 16 +
          col] = o[dt][i] * inv;
  }
}

__global__ void BuildKvHalfKernel(const float* __restrict__ knope,
                                  const float* __restrict__ extra,
                                  __half* __restrict__ k16, std::size_t rows) {
  const std::size_t i =
      threadIdx.x + blockIdx.x * static_cast<std::size_t>(kThreads);
  const std::size_t total = rows * kRaonHeads * 192;
  if (i >= total)
    return;
  const std::size_t d = i % 192;
  const std::size_t h = (i / 192) % kRaonHeads;
  const std::size_t r = i / (192 * kRaonHeads);
  const float v = d < 128 ? knope[(r * kRaonHeads + h) * 128 + d]
                          : extra[r * 64 + (d - 128)];
  k16[i] = __float2half(v);
}

}  // namespace

int F32Matvec(const float* w, const float* x, float* y, int m, int k,
              hipStream_t stream) {
  F32MatvecKernel<<<m, kThreads, 0, stream>>>(w, x, y, m, k);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int F16Matvec(const __half* w, const float* x, float* y, int m, int k,
              hipStream_t stream, std::uint64_t row0) {
  F16MatvecKernel<<<m, kThreads, 0, stream>>>(w, x, y, k, row0);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int F16MatvecExperts(const __half* w, const float* x, float* y, int m, int k,
                     int used, const int* ids, bool x_per_expert,
                     hipStream_t stream) {
  F16MatvecExpertsKernel<<<dim3(m, used), kThreads, 0, stream>>>(
      w, x, y, m, k, ids, x_per_expert);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HcSelect(const float* x, const float* fn, const float* scale,
             const float* base, float hc_eps, float magnitude, float* y,
             float* pre, float* post, float* comb, hipStream_t stream,
             int tokens) {
  HcSelectKernel<<<tokens, kHcThreads, 0, stream>>>(
      x, fn, scale, base, hc_eps, magnitude, y, pre, post, comb);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HcDotsPart(const float* x, const float* fn, float* dots, int tokens,
               hipStream_t stream) {
  HcDotsPartKernel<<<dim3(kHcChunks, tokens), kRaonMixes * 32, 0, stream>>>(
      x, fn, dots);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HcDots(const float* x, const float* fn, float* dots, int tokens,
           hipStream_t stream) {
  HcDotsKernel<<<(tokens + kHcDotTokens - 1) / kHcDotTokens, kRaonMixes * 32, 0,
                 stream>>>(x, fn, dots, tokens);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HcSelectPre(const float* x, const float* dots, const float* scale,
                const float* base, float hc_eps, float magnitude, float* y,
                float* pre, float* post, float* comb, hipStream_t stream,
                int tokens, int chunks) {
  HcSelectPreKernel<<<tokens, kHcThreads, 0, stream>>>(
      x, dots, scale, base, hc_eps, magnitude, y, pre, post, comb, chunks);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HcApply(const float* x, const float* block, const float* post,
            const float* comb, float* dst, hipStream_t stream, int tokens) {
  HcApplyKernel<<<tokens, kRaonHidden / 4, 0, stream>>>(x, block, post, comb,
                                                        dst);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int RmsNormRows(const float* x, const float* w, float* y, int n, int rows,
                float eps, hipStream_t stream, __half* y16) {
  RmsNormRowsKernel<<<rows, kThreads, 0, stream>>>(x, w, y, n, eps, y16);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int L2Heads(const float* x, float* y, int heads, int hd, float eps,
            hipStream_t stream) {
  L2HeadsKernel<<<heads, kThreads, 0, stream>>>(x, y, hd, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int RdaConv(const float* wq, const float* wk, const float* wv,
            const float* rawq, const float* rawk, const float* rawv,
            float* hist, float* outq, float* outk, float* outv, int channels,
            int taps, hipStream_t stream) {
  RdaConvKernel<<<dim3(3, (channels + kThreads - 1) / kThreads), kThreads, 0,
                  stream>>>(wq, wk, wv, rawq, rawk, rawv, hist, outq, outk,
                            outv, channels, taps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int RdaDecay(const float* gproj, const float* dt, const float* sa, float* decay,
             int heads, int hd, float lower_bound, hipStream_t stream) {
  RdaDecayKernel<<<heads, hd, 0, stream>>>(gproj, dt, sa, decay, hd,
                                           lower_bound);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int RdaRecur(const float* q, const float* k, const float* v, const float* decay,
             const float* beta, const float* ratio, const float* g2,
             const float* norm_w, float* state, float* out, float* pre_norm,
             int heads, int hd, float eps, hipStream_t stream) {
  RdaRecurKernel<<<heads, kRdaThreads, 0, stream>>>(
      q, k, v, decay, beta, ratio, g2, norm_w, state, out, pre_norm, hd, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MlaAttendSplit(const float* qabs, const float* qe, const float* lat,
                   const float* extra, float* part, float* ctx, int heads,
                   int rank, int rope, int pos0, int tokens,
                   hipStream_t stream) {
  const int max_len = pos0 + tokens;
  int chunk = std::max(256, (max_len + kMlaMaxSplits - 1) / kMlaMaxSplits);
  chunk = (chunk + 63) / 64 * 64;
  const int splits = (max_len + chunk - 1) / chunk;
  MlaSplitTileKernel<<<dim3(splits, heads / kMlaHeadsPerBlock, tokens),
                       kMlaSplitThreads, 0, stream>>>(
      qabs, qe, lat, extra, part, heads, rank, rope, pos0, chunk, splits,
      1.0F / sqrtf(static_cast<float>(kRaonNoPe + kRaonRope)));
  MlaCombineKernel<<<dim3(heads, tokens), kMlaSplitThreads, 0, stream>>>(
      part, ctx, heads, rank, splits);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MlaAttendSplitWmma(const float* qabs, const float* qe, void* q16,
                       const float* lat, const float* extra, float* part,
                       float* ctx, int heads, int pos0, int tokens,
                       hipStream_t stream) {
  const int nrows = tokens * heads;
  const int max_len = pos0 + tokens;
  int chunk = std::max(256, (max_len + kMlaMaxSplits - 1) / kMlaMaxSplits);
  chunk = (chunk + 63) / 64 * 64;
  const int splits = (max_len + chunk - 1) / chunk;
  MlaQHalfKernel<<<nrows, 192, 0, stream>>>(qabs, qe,
                                            static_cast<_Float16*>(q16),
                                            kRaonRank, kRaonRope);
  MlaSplitWmmaKernel<<<dim3(splits, (nrows + kMwRows - 1) / kMwRows), 256, 0,
                       stream>>>(
      static_cast<const _Float16*>(q16), lat, extra, part, heads, nrows, pos0,
      chunk, splits, 1.0F / sqrtf(static_cast<float>(kRaonNoPe + kRaonRope)));
  MlaCombineKernel<<<dim3(heads, tokens), kMlaSplitThreads, 0, stream>>>(
      part, ctx, heads, kRaonRank, splits);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MlaAttend(const float* qabs, const float* qe, const float* lat,
              const float* extra, float* out, int heads, int rank, int rope,
              int vd, int n, hipStream_t stream) {
  MlaAttendKernel<<<heads, kThreads, 0, stream>>>(
      qabs, qe, lat, extra, out, rank, rope, n,
      1.0F / sqrtf(static_cast<float>(kRaonNoPe + kRaonRope)));
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MoeRoute(const float* logits, const float* bias, float scale, int* ids,
             float* weights, int experts, int used, bool normalize,
             hipStream_t stream, int tokens) {
  MoeRouteKernel<<<tokens, kThreads, 0, stream>>>(
      logits, bias, scale, ids, weights, experts, used, normalize);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int SituAct(const float* g, const float* u, float* out, int n, float beta,
            float linear_beta, hipStream_t stream) {
  SituActKernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      g, u, out, n, beta, linear_beta);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int SituActHalf(const float* g, const float* u, __half* out, std::size_t n,
                float beta, float linear_beta, hipStream_t stream) {
  SituActHalfKernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      g, u, out, n, beta, linear_beta);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int ToHalf(const float* x, __half* out, std::size_t n, hipStream_t stream) {
  ToHalfKernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(x, out,
                                                                       n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MoeCombineHalf(const __half* parts, const float* weights, float* out, int n,
                   int used, int tokens, hipStream_t stream) {
  MoeCombineHalfKernel<<<dim3((n + kThreads - 1) / kThreads, tokens), kThreads,
                         0, stream>>>(parts, weights, out, n, used);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int RdaPrefill(const float* wq, const float* wk, const float* wv,
               const float* rawq, const float* rawk, const float* rawv,
               float* hist, const float* gproj, const float* dt,
               const float* sa, const float* beta, const float* ratio,
               const float* g2, const float* norm_w, float* state, float* out,
               int n, int heads, float lower_bound, float eps,
               hipStream_t stream) {
  RdaPrefillKernel<<<heads, 128, 0, stream>>>(
      wq, wk, wv, rawq, rawk, rawv, hist, gproj, dt, sa, beta, ratio, g2,
      norm_w, state, out, n, heads * 128, heads, lower_bound, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int F32Gemm(const float* w, const float* x, float* y, int m, int n, int k,
            hipStream_t stream) {
  F32GemmKernel<<<dim3((m + 63) / 64, (n + 63) / 64), 256, 0, stream>>>(
      w, x, y, m, n, k);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int RecordOperands(const float* const src[7], float* const dst[7],
                   const int count[7], hipStream_t stream) {
  CopyList list;
  for (int i = 0; i < 7; ++i) {
    list.src[i] = src[i];
    list.dst[i] = dst[i];
    list.count[i] = count[i];
  }
  CopyListKernel<<<dim3(64, 7), kThreads, 0, stream>>>(list);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int BuildTiles(const std::uint32_t* counts, int experts, std::int32_t* t48,
               int cap48, std::int32_t* t64, int cap64, std::int32_t* t128,
               int cap128, hipStream_t stream) {
  BuildTilesKernel<<<1, 512, 0, stream>>>(counts, experts, t48, cap48, t64,
                                          cap64, t128, cap128);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int WaitFlag(const int* flag, int epoch, hipStream_t stream) {
  WaitFlagKernel<<<1, 1, 0, stream>>>(flag, epoch);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int TransposeKb(const __half* kb, __half* kbt, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(kRaonHeads) * 128 * 512;
  TransposeKbKernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      kb, kbt);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int F32SmallGemm(const float* w, const float* x, float* y, int m, int n, int k,
                 hipStream_t stream) {
  if (n > kF32SmallMax)
    return -1;
  F32SmallGemmKernel<<<m, 256, 0, stream>>>(w, x, y, m, n, k);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KvNorm(const float* kv, const float* w, float* lat, float* extra, int n,
           int stride, int rank, int rope, float eps, hipStream_t stream) {
  KvNormKernel<<<n, kThreads, 0, stream>>>(kv, w, lat, extra, stride, rank,
                                           rope, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int ReplicateStreams(const float* src, float* dst, int tokens, int h,
                     hipStream_t stream) {
  ReplicateStreamsKernel<<<dim3((h + kThreads - 1) / kThreads, tokens),
                           kThreads, 0, stream>>>(src, dst, h);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int PoolUpdate(const float* lat, float* pool, int first_block, int blocks,
               int total_rows, hipStream_t stream) {
  PoolUpdateKernel<<<blocks, kRaonRank, 0, stream>>>(lat, pool, first_block,
                                                     total_rows);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int SparseSelect(const float* mla_q, float* qm, const float* qabs,
                 const float* pool, float* scores, int* sel, int* kpos, int n,
                 int first_block, int score_end, int k, int sink_rows,
                 int local_start, int pos0, int qgroups, hipStream_t stream) {
  (void)qabs;
  (void)qm;
  (void)mla_q;
  BlockScoreKernel<<<score_end - first_block, 256, 0, stream>>>(
      qabs, pool, scores, first_block, qgroups * kRaonHeads);
  SelectTopKernel<<<1, 1024, 0, stream>>>(scores, first_block, score_end, k,
                                          sel);
  BuildKeyIndexKernel<<<64, 256, 0, stream>>>(sel, k, sink_rows, local_start,
                                              pos0, n, kpos);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int QNopeMean(const float* q, float* qm, int n, int heads, int nope, int rope,
              hipStream_t stream) {
  QNopeMeanKernel<<<dim3(heads, kSparseGroups), nope, 0, stream>>>(q, qm, n,
                                                                   nope, rope);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int GatherKvFloat(const float* lat, const float* extra, const int* kpos,
                  int cnt, float* lat_g, float* extra_g, hipStream_t stream) {
  GatherKvFloatKernel<<<cnt, 128, 0, stream>>>(lat, extra, kpos, lat_g,
                                               extra_g);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int GatherKv(const float* lat, const float* extra, const int* kpos, int cnt,
             __half* lat16, float* extra_g, hipStream_t stream) {
  GatherKvKernel<<<cnt, 128, 0, stream>>>(lat, extra, kpos, lat16, extra_g);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MlaWmmaAttend(const __half* q16, const __half* k16, const __half* v16,
                  float* out, int n, int pos0, hipStream_t stream,
                  const int* kpos, int nk) {
  const float scale = 1.0F / sqrtf(static_cast<float>(kRaonNoPe + kRaonRope));
  if (kpos == nullptr) {
    MlaWmmaAttnFastKernel<<<dim3((n + 127) / 128, kRaonHeads), 256, 0,
                            stream>>>(q16, k16, v16, out, n, pos0, scale, nk);
  } else {
    MlaWmmaAttnKernel<<<dim3((n + kWQ - 1) / kWQ, kRaonHeads), 128, 0,
                        stream>>>(q16, k16, v16, out, n, pos0, scale, kpos, nk);
  }
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int BuildKvHalf(const float* knope, const float* extra, __half* k16,
                std::size_t rows, hipStream_t stream) {
  const std::size_t total = rows * kRaonHeads * 192;
  BuildKvHalfKernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      knope, extra, k16, rows);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int Q5ToQ8(const void* src, void* dst, std::size_t blocks, hipStream_t stream) {
  Q5ToQ8Kernel<<<(blocks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(src), static_cast<std::uint8_t*>(dst),
      blocks);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MoeCombine(const float* parts, const float* weights, float* out, int n,
               int used, hipStream_t stream) {
  MoeCombineKernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      parts, weights, out, n, used);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int VectorMul(const float* x, const float* gate, float* out, int n,
              hipStream_t stream) {
  VectorMulKernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      x, gate, out, n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int AddVectors(const float* x, const float* y, float* out, int n,
               hipStream_t stream) {
  AddVectorsKernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      x, y, out, n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int MlaQeGather(const float* q, float* qe, float* qn, int heads, int nope,
                int rope, hipStream_t stream, int tokens) {
  MlaQeGatherKernel<<<dim3(heads, tokens), std::max(nope, rope), 0, stream>>>(
      q, qe, qn, nope, rope);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int SteerAdd(float* x, const float* steer, float scale, int n,
             hipStream_t stream) {
  SteerAddKernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      x, steer, scale, n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HeadMeanNorm(const float* x, const float* w, float* y, int h, int streams,
                 float eps, hipStream_t stream) {
  HeadMeanNormKernel<<<1, kThreads, 0, stream>>>(x, w, y, h, streams, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int ArgmaxRows(const float* logits, int rows, int vocab, int* out,
               hipStream_t stream) {
  ArgmaxRowsKernel<<<rows, kThreads, 0, stream>>>(logits, vocab, out);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::raon1::rocm
