#include "src/models/ling32/gpu/kernels.hip.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>

namespace gufo::models::ling32::gpu {
namespace {

constexpr int kElemThreads = 256;
constexpr int kScoreThreads = 256;
constexpr int kMaxKdaDim = 128;
constexpr int kMaxExperts = 512;
constexpr int kMaxGroups = 8;
constexpr int kMaxUsed = 8;
constexpr int kMaxGroupsUsed = 4;

__device__ float Sigmoid(float v) {
  return 1.0F / (1.0F + __expf(-v));
}

__device__ float Silu(float v) {
  return v * Sigmoid(v);
}

__device__ float BlockSum(float v, float* scratch, int tid, int threads) {
  scratch[tid] = v;
  __syncthreads();
  for (int step = threads / 2; step > 0; step >>= 1) {
    if (tid < step)
      scratch[tid] += scratch[tid + step];
    __syncthreads();
  }
  return scratch[0];
}

__device__ float BlockMax(float v, float* scratch, int tid, int threads) {
  scratch[tid] = v;
  __syncthreads();
  for (int step = threads / 2; step > 0; step >>= 1) {
    if (tid < step)
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + step]);
    __syncthreads();
  }
  return scratch[0];
}

__global__ void AddToKernel(float* __restrict__ x, const float* __restrict__ y,
                            std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i];
}

__global__ void F32GemvDoubleKernel(const float* __restrict__ w,
                                    const float* __restrict__ x,
                                    float* __restrict__ out,
                                    std::uint32_t cols) {
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  __shared__ double partial[kElemThreads];
  double acc = 0.0;
  for (int i = tid; i < static_cast<int>(cols); i += kElemThreads) {
    acc += static_cast<double>(w[static_cast<std::size_t>(row) * cols + i]) *
           static_cast<double>(x[i]);
  }
  partial[tid] = acc;
  __syncthreads();
  for (int step = kElemThreads / 2; step > 0; step >>= 1) {
    if (tid < step)
      partial[tid] += partial[tid + step];
    __syncthreads();
  }
  if (tid == 0)
    out[row] = static_cast<float>(partial[0]);
}

struct Q8Block {
  std::uint16_t d;
  std::int8_t qs[32];
};

#pragma pack(push, 1)
struct Q4_0Block {
  std::uint16_t d;
  std::uint8_t qs[16];
};
struct Q4KBlock {
  std::uint16_t d, dmin;
  std::uint8_t scales[12], qs[128];
};
struct Q5KBlock {
  std::uint16_t d, dmin;
  std::uint8_t scales[12], qh[32], qs[128];
};
struct Q6KBlock {
  std::uint8_t ql[128], qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};
#pragma pack(pop)

__device__ float F16Val(std::uint16_t h) {
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

__device__ void GetQ4ScaleMin(int idx, const std::uint8_t* s, std::uint8_t& sc,
                              std::uint8_t& mn) {
  if (idx < 4) {
    sc = s[idx] & 0x3FU;
    mn = s[idx + 4] & 0x3FU;
    return;
  }
  sc = static_cast<std::uint8_t>((s[idx + 4] & 0x0FU) |
                                 ((s[idx - 4] >> 6U) << 4U));
  mn = static_cast<std::uint8_t>((s[idx + 4] >> 4U) | ((s[idx] >> 6U) << 4U));
}

__device__ float Q4KVal(const Q4KBlock& b, int i) {
  const int group = i / 32;
  const int pair = group / 2;
  const bool high = (group & 1) != 0;
  const int lane = i % 32;
  const std::uint8_t packed = b.qs[pair * 32 + lane];
  const std::uint8_t q = high ? (packed >> 4U) : (packed & 0x0FU);
  std::uint8_t sc = 0, mn = 0;
  GetQ4ScaleMin(group, b.scales, sc, mn);
  return F16Val(b.d) * static_cast<float>(sc) * static_cast<float>(q) -
         F16Val(b.dmin) * static_cast<float>(mn);
}

__device__ float Q5KVal(const Q5KBlock& b, int i) {
  const int gg = i / 64;
  const int wv = i % 64;
  const int lane = wv % 32;
  const bool lo = wv < 32;
  const std::uint8_t qb = b.qs[gg * 32 + lane];
  const std::uint8_t q4 = lo ? (qb & 0x0FU) : (qb >> 4U);
  const int bit = 2 * gg + (lo ? 0 : 1);
  const int q = q4 + (((b.qh[lane] >> bit) & 1U) ? 16 : 0);
  const int sis = 2 * gg + (lo ? 0 : 1);
  std::uint8_t sc = 0, mn = 0;
  GetQ4ScaleMin(sis, b.scales, sc, mn);
  return F16Val(b.d) * static_cast<float>(sc) * static_cast<float>(q) -
         F16Val(b.dmin) * static_cast<float>(mn);
}

__device__ float Q6KVal(const Q6KBlock& b, int i) {
  const int half = i / 128;
  const int within = i % 128;
  const int seg = within / 32;
  const int lane = within % 32;
  const int qlb = half * 64;
  const std::uint8_t qh = b.qh[half * 32 + lane];
  std::uint8_t low = 0, high = 0;
  switch (seg) {
    case 0:
      low = b.ql[qlb + lane] & 0x0FU;
      high = qh & 0x03U;
      break;
    case 1:
      low = b.ql[qlb + 32 + lane] & 0x0FU;
      high = (qh >> 2U) & 0x03U;
      break;
    case 2:
      low = b.ql[qlb + lane] >> 4U;
      high = (qh >> 4U) & 0x03U;
      break;
    default:
      low = b.ql[qlb + 32 + lane] >> 4U;
      high = (qh >> 6U) & 0x03U;
      break;
  }
  const int si = half * 8 + (lane / 16) + seg * 2;
  const int quant = static_cast<int>((high << 4U) | low) - 32;
  return F16Val(b.d) * static_cast<float>(b.scales[si]) *
         static_cast<float>(quant);
}

__device__ float DequantElement(const void* src, std::uint32_t i, int type) {
  switch (type) {
    case 0:
      return static_cast<const float*>(src)[i];
    case 2: {
      const auto& b = static_cast<const Q4_0Block*>(src)[i / 32];
      const std::uint32_t within = i % 32;
      const std::uint8_t packed = b.qs[within % 16];
      const int q = within < 16 ? (packed & 0x0FU) : (packed >> 4U);
      return F16Val(b.d) * static_cast<float>(q - 8);
    }
    case 8: {
      const auto& b = static_cast<const Q8Block*>(src)[i / 32];
      return F16Val(b.d) * static_cast<float>(b.qs[i % 32]);
    }
    case 12:
      return Q4KVal(static_cast<const Q4KBlock*>(src)[i / 256], i % 256);
    case 13:
      return Q5KVal(static_cast<const Q5KBlock*>(src)[i / 256], i % 256);
    case 14:
      return Q6KVal(static_cast<const Q6KBlock*>(src)[i / 256], i % 256);
    default:
      return 0.0F;
  }
}

__global__ void DequantKernel(const void* __restrict__ src,
                              float* __restrict__ dst, std::uint32_t n,
                              int type) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    dst[i] = DequantElement(src, i, type);
  }
}

__global__ void DequantExpertsKernel(const void* __restrict__ base,
                                     const std::int32_t* __restrict__ ids,
                                     float* __restrict__ dst,
                                     std::uint32_t slots, std::uint32_t per,
                                     std::uint32_t expert_bytes, int type) {
  const std::uint32_t n = slots * per;
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < n;
       g += stride) {
    const std::uint32_t slot = g / per;
    const std::uint32_t i = g - slot * per;
    const std::int32_t e = ids[slot];
    const auto* src = static_cast<const std::uint8_t*>(base) +
                      static_cast<std::size_t>(e) * expert_bytes;
    dst[g] = DequantElement(src, i, type);
  }
}

__global__ void RopeHeadsKernel(const float* __restrict__ in,
                                float* __restrict__ out, std::uint32_t stride,
                                std::uint32_t offset, std::uint32_t dim,
                                std::uint32_t pos, float base,
                                std::uint32_t out_stride,
                                const std::int32_t* __restrict__ pos_ptr,
                                std::uint32_t out_pos_stride) {
  if (pos_ptr != nullptr)
    pos = static_cast<std::uint32_t>(*pos_ptr);
  const std::uint32_t half = dim / 2;
  const std::uint32_t i = threadIdx.x;
  if (i >= half)
    return;
  const float* row =
      in + static_cast<std::size_t>(blockIdx.x) * stride + offset;
  const float angle =
      static_cast<float>(pos) *
      powf(base, -2.0F * static_cast<float>(i) / static_cast<float>(dim));
  float s = 0.0F;
  float c = 0.0F;
  sincosf(angle, &s, &c);
  const float a = row[i];
  const float b = row[i + half];
  float* dst = out + static_cast<std::size_t>(blockIdx.x) * out_stride +
               static_cast<std::size_t>(pos) * out_pos_stride;
  dst[i] = a * c - b * s;
  dst[i + half] = a * s + b * c;
}

__global__ void CopyRowKernel(const float* __restrict__ src,
                              float* __restrict__ dst, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

__global__ void AppendRowKernel(const float* __restrict__ row,
                                float* __restrict__ cache, std::uint32_t pos,
                                std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim)
    cache[static_cast<std::size_t>(pos) * dim + i] = row[i];
}

__global__ void KdaConvKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ cq,
    const float* __restrict__ ck, const float* __restrict__ cv,
    float* __restrict__ history, float* __restrict__ oq, float* __restrict__ ok,
    float* __restrict__ ov, std::uint32_t channels, std::uint32_t taps) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= channels)
    return;
  const std::uint32_t hist_len = (taps - 1) * channels;
  const float* in[3] = {q, k, v};
  const float* cw[3] = {cq, ck, cv};
  float* out[3] = {oq, ok, ov};
  for (std::uint32_t bank = 0; bank < 3; ++bank) {
    const float* tap_w = cw[bank] + static_cast<std::size_t>(ch) * taps;
    float acc = tap_w[taps - 1] * in[bank][ch];
    for (std::uint32_t t = 0; t + 1 < taps; ++t) {
      acc += tap_w[t] * history[static_cast<std::size_t>(bank) * hist_len +
                                static_cast<std::size_t>(t) * channels + ch];
    }
    out[bank][ch] = Silu(acc);
    const float raw = in[bank][ch];
    for (std::uint32_t t = 0; t + 2 < taps; ++t) {
      history[static_cast<std::size_t>(bank) * hist_len +
              static_cast<std::size_t>(t) * channels + ch] =
          history[static_cast<std::size_t>(bank) * hist_len +
                  static_cast<std::size_t>(t + 1) * channels + ch];
    }
    history[static_cast<std::size_t>(bank) * hist_len +
            static_cast<std::size_t>(taps - 2) * channels + ch] = raw;
  }
}

__global__ __launch_bounds__(kMaxKdaDim) void KdaStepKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ fa,
    const float* __restrict__ ga, const float* __restrict__ beta,
    const float* __restrict__ dt, const float* __restrict__ sa,
    const float* __restrict__ norm_w, float* __restrict__ state,
    float* __restrict__ out, std::uint32_t dim, float lower_bound, float eps) {
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid >= static_cast<int>(dim))
    return;
  __shared__ float sq[kMaxKdaDim];
  __shared__ float sk[kMaxKdaDim];
  __shared__ float sd[kMaxKdaDim];
  __shared__ float su[kMaxKdaDim];
  __shared__ float so[kMaxKdaDim];
  __shared__ float scratch[kMaxKdaDim];

  const std::size_t hoff = static_cast<std::size_t>(h) * dim;
  const float qv = q[hoff + tid];
  const float kv = k[hoff + tid];
  sq[tid] = qv;
  sk[tid] = kv;
  const float fv = fa[hoff + tid] + dt[hoff + tid];
  const float gv = ga[hoff + tid];
  const float decay = __expf(lower_bound * Sigmoid(fv * sa[h]));
  sd[tid] = decay;
  const float sumq = BlockSum(qv * qv, scratch, tid, dim);
  const float sumk = BlockSum(kv * kv, scratch, tid, dim);
  const float qn = sq[tid] * rsqrtf(sumq + eps);
  const float kn = sk[tid] * rsqrtf(sumk + eps);
  sq[tid] = qn;
  sk[tid] = kn;
  __syncthreads();

  float* row = state + (hoff + static_cast<std::size_t>(tid)) * dim;
  double acc = 0.0;
  for (int i = 0; i < static_cast<int>(dim); ++i) {
    row[i] *= sd[i];
    acc += static_cast<double>(row[i]) * static_cast<double>(sk[i]);
  }
  su[tid] = static_cast<float>(acc);
  __syncthreads();

  const float b = Sigmoid(beta[h]);
  const float delta = (v[hoff + tid] - su[tid]) * b;
  double o = 0.0;
  for (int i = 0; i < static_cast<int>(dim); ++i) {
    row[i] += delta * sk[i];
    o += static_cast<double>(row[i]) * static_cast<double>(sq[i]);
  }
  so[tid] = static_cast<float>(o) * rsqrtf(static_cast<float>(dim));
  __syncthreads();

  const float local_sum = so[tid] * so[tid] / static_cast<float>(dim);
  const float sumsq = BlockSum(local_sum, scratch, tid, dim);
  const float scale = rsqrtf(sumsq + eps) * norm_w[tid] * Sigmoid(gv);
  out[hoff + tid] = so[tid] * scale;
}

__global__ void MlaScoresKernel(const float* __restrict__ q,
                                const float* __restrict__ cache,
                                float* __restrict__ scores, std::uint32_t rows,
                                std::uint32_t dim, float scale) {
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float scratch[kScoreThreads];
  const float* qh = q + static_cast<std::size_t>(h) * dim;
  for (std::uint32_t r = 0; r < rows; ++r) {
    float acc = 0.0F;
    for (std::uint32_t c = static_cast<std::uint32_t>(tid); c < dim;
         c += blockDim.x) {
      acc += qh[c] * cache[static_cast<std::size_t>(r) * dim + c];
    }
    const float sum = BlockSum(acc, scratch, tid, blockDim.x);
    if (tid == 0)
      scores[static_cast<std::size_t>(h) * rows + r] = sum * scale;
    __syncthreads();
  }
}

__global__ void MlaGatherKernel(const float* __restrict__ scores,
                                const float* __restrict__ cache,
                                float* __restrict__ ctx, std::uint32_t rows,
                                std::uint32_t dim, std::uint32_t value_dim) {
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float mx;
  __shared__ float denom;
  __shared__ float scratch[kElemThreads];
  const float* sh = scores + static_cast<std::size_t>(h) * rows;

  float local_max = -INFINITY;
  for (std::uint32_t r = static_cast<std::uint32_t>(tid); r < rows;
       r += blockDim.x) {
    local_max = fmaxf(local_max, sh[r]);
  }
  const float best = BlockMax(local_max, scratch, tid, blockDim.x);
  if (tid == 0)
    mx = best;
  __syncthreads();

  float local_sum = 0.0F;
  for (std::uint32_t r = static_cast<std::uint32_t>(tid); r < rows;
       r += blockDim.x) {
    local_sum += __expf(sh[r] - mx);
  }
  const float total = BlockSum(local_sum, scratch, tid, blockDim.x);
  if (tid == 0)
    denom = total;
  __syncthreads();

  for (std::uint32_t c = static_cast<std::uint32_t>(tid); c < value_dim;
       c += blockDim.x) {
    float acc = 0.0F;
    for (std::uint32_t r = 0; r < rows; ++r) {
      const float p = __expf(sh[r] - mx) / denom;
      acc += p * cache[static_cast<std::size_t>(r) * dim + c];
    }
    ctx[static_cast<std::size_t>(h) * value_dim + c] = acc;
  }
}

__global__ void MlaGateKernel(const float* __restrict__ value,
                              const float* __restrict__ gate,
                              float* __restrict__ out, std::uint32_t total,
                              std::uint32_t head_dim) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  out[i] = value[i] * Sigmoid(gate[i / head_dim]);
}

__global__ __launch_bounds__(kMaxExperts) void RouterLingKernel(
    const float* __restrict__ logits, const float* __restrict__ bias,
    std::int32_t* __restrict__ ids, float* __restrict__ weights,
    std::uint32_t experts, std::uint32_t groups, std::uint32_t groups_used,
    std::uint32_t used, float scale) {
  const int tid = threadIdx.x;
  logits += static_cast<std::size_t>(blockIdx.x) * experts;
  ids += static_cast<std::size_t>(blockIdx.x) * used;
  weights += static_cast<std::size_t>(blockIdx.x) * used;
  __shared__ float probs[kMaxExperts];
  __shared__ float sel[kMaxExperts];
  __shared__ float gscore[kMaxGroups];
  __shared__ int group_used[kMaxGroups];
  __shared__ int best_ids[kMaxUsed];
  __shared__ float total;

  if (tid < static_cast<int>(experts)) {
    probs[tid] = Sigmoid(logits[tid]);
    sel[tid] = probs[tid] + bias[tid];
  }
  __syncthreads();

  if (tid < static_cast<int>(groups)) {
    const std::uint32_t per = experts / groups;
    float first = -INFINITY;
    float second = -INFINITY;
    for (std::uint32_t i = 0; i < per; ++i) {
      const float v = sel[static_cast<std::size_t>(tid) * per + i];
      if (v > first) {
        second = first;
        first = v;
      } else if (v > second) {
        second = v;
      }
    }
    gscore[tid] = first + second;
    group_used[tid] = 0;
  }
  __syncthreads();

  if (tid == 0) {
    for (std::uint32_t k = 0; k < groups_used; ++k) {
      int best_g = -1;
      float best_v = -INFINITY;
      for (std::uint32_t g = 0; g < groups; ++g) {
        if (group_used[g] == 0 && gscore[g] > best_v) {
          best_v = gscore[g];
          best_g = static_cast<int>(g);
        }
      }
      group_used[best_g < 0 ? 0 : best_g] = 1;
    }
  }
  __syncthreads();

  if (tid < static_cast<int>(experts)) {
    const std::uint32_t per = experts / groups;
    if (group_used[tid / per] == 0)
      sel[tid] = -INFINITY;
  }
  __syncthreads();

  __shared__ float wave_v[kMaxExperts / 32];
  __shared__ int wave_i[kMaxExperts / 32];
  for (std::uint32_t k = 0; k < used; ++k) {
    float bv = tid < static_cast<int>(experts) ? sel[tid] : -INFINITY;
    int bi = tid;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off);
      const int oi = __shfl_xor(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if ((tid & 31) == 0) {
      wave_v[tid >> 5] = bv;
      wave_i[tid >> 5] = bi;
    }
    __syncthreads();
    float fv = wave_v[0];
    int fi = wave_i[0];
    for (int w = 1; w < kMaxExperts / 32; ++w) {
      if (wave_v[w] > fv || (wave_v[w] == fv && wave_i[w] < fi)) {
        fv = wave_v[w];
        fi = wave_i[w];
      }
    }
    if (tid == 0)
      best_ids[k] = fi;
    if (tid == fi)
      sel[tid] = -INFINITY;
    __syncthreads();
  }
  if (tid == 0) {
    float sum = 0.0F;
    for (std::uint32_t k = 0; k < used; ++k)
      sum += probs[best_ids[k]];
    total = fmaxf(sum, 6.103515625e-5F);
  }
  __syncthreads();

  if (tid < static_cast<int>(used)) {
    ids[tid] = best_ids[tid];
    weights[tid] = probs[best_ids[tid]] / total * scale;
  }
}

__global__ void SwigluClampKernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ out, std::uint32_t n,
                                  float limit) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float u = fminf(fmaxf(up[i], -limit), limit);
  out[i] = fminf(Silu(gate[i]), limit) * u;
}

__global__ void SwigluPlainKernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ out, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = Silu(gate[i]) * up[i];
}

__global__ void MoeCombineKernel(const float* __restrict__ parts,
                                 const float* __restrict__ weights,
                                 const float* __restrict__ shared,
                                 float* __restrict__ out, std::uint32_t used,
                                 std::uint32_t hidden) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= hidden)
    return;
  float acc = shared[c];
  for (std::uint32_t e = 0; e < used; ++e) {
    acc += weights[e] * parts[static_cast<std::size_t>(e) * hidden + c];
  }
  out[c] = acc;
}

constexpr int kWave = 32;
constexpr int kMlaHeads = 32;
constexpr std::uint32_t kMlaMaxSplits = 64;
constexpr int kMlaLatent = 512;
constexpr int kMlaKey = 576;
constexpr int kMlaHeadsPerWave = 4;
constexpr int kMlaLaneLatent = kMlaLatent / kWave;
constexpr int kMlaLaneRope = (kMlaKey - kMlaLatent) / kWave;

__device__ float WaveSum32(float v) {
  for (int off = kWave / 2; off > 0; off >>= 1)
    v += __shfl_xor(v, off);
  return v;
}

constexpr int kKdaRowsPerBlock = 8;
constexpr int kKdaRowsPerWave = 1;

__global__ __launch_bounds__(256) void KdaDecodeKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ fa,
    const float* __restrict__ beta, const float* __restrict__ dt,
    const float* __restrict__ sa, float* __restrict__ state,
    float* __restrict__ so, float lower_bound, float eps) {
  const int h = blockIdx.x;
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const std::size_t hoff = static_cast<std::size_t>(h) * 128;
  const float4 q4 = *reinterpret_cast<const float4*>(q + hoff + lane * 4);
  const float4 k4 = *reinterpret_cast<const float4*>(k + hoff + lane * 4);
  const float4 f4 = *reinterpret_cast<const float4*>(fa + hoff + lane * 4);
  const float4 d4 = *reinterpret_cast<const float4*>(dt + hoff + lane * 4);
  const float a = sa[h];
  float qv[4] = {q4.x, q4.y, q4.z, q4.w};
  float kv[4] = {k4.x, k4.y, k4.z, k4.w};
  const float fv[4] = {f4.x + d4.x, f4.y + d4.y, f4.z + d4.z, f4.w + d4.w};
  float dec[4];
  float qq = 0.0F;
  float kk = 0.0F;
  for (int e = 0; e < 4; ++e) {
    dec[e] = __expf(lower_bound * Sigmoid(fv[e] * a));
    qq += qv[e] * qv[e];
    kk += kv[e] * kv[e];
  }
  const float qn = rsqrtf(WaveSum32(qq) + eps);
  const float kn = rsqrtf(WaveSum32(kk) + eps);
  for (int e = 0; e < 4; ++e) {
    qv[e] *= qn;
    kv[e] *= kn;
  }
  const float b = Sigmoid(beta[h]);
  const float out_scale = rsqrtf(128.0F);
  for (int r = 0; r < kKdaRowsPerWave; ++r) {
    const int row = blockIdx.y * kKdaRowsPerBlock + wave * kKdaRowsPerWave + r;
    float4* sp = reinterpret_cast<float4*>(
                     state + (hoff + static_cast<std::size_t>(row)) * 128) +
                 lane;
    float4 sv = *sp;
    sv.x *= dec[0];
    sv.y *= dec[1];
    sv.z *= dec[2];
    sv.w *= dec[3];
    const float u =
        WaveSum32(sv.x * kv[0] + sv.y * kv[1] + sv.z * kv[2] + sv.w * kv[3]);
    const float delta = (v[hoff + row] - u) * b;
    sv.x += delta * kv[0];
    sv.y += delta * kv[1];
    sv.z += delta * kv[2];
    sv.w += delta * kv[3];
    const float o =
        WaveSum32(sv.x * qv[0] + sv.y * qv[1] + sv.z * qv[2] + sv.w * qv[3]);
    *sp = sv;
    if (lane == 0)
      so[hoff + row] = o * out_scale;
  }
}

__global__ __launch_bounds__(128) void KdaDecodeOutKernel(
    const float* __restrict__ so, const float* __restrict__ ga,
    const float* __restrict__ norm_w, float* __restrict__ out, float eps) {
  __shared__ float part[4];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 128;
  const int tid = threadIdx.x;
  const float val = so[base + tid];
  const float sum = WaveSum32(val * val / 128.0F);
  if ((tid & 31) == 0)
    part[tid >> 5] = sum;
  __syncthreads();
  const float total = part[0] + part[1] + part[2] + part[3];
  out[base + tid] =
      val * rsqrtf(total + eps) * norm_w[tid] * Sigmoid(ga[base + tid]);
}

// ---- batched prefill kernels ----

__global__ void NarrowRowsKernel(const float* __restrict__ in,
                                 __half* __restrict__ out, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = __float2half(in[i]);
}

__global__ void SwigluHalfKernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 __half* __restrict__ out, std::size_t n,
                                 float limit) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float g = Silu(gate[i]);
  float u = up[i];
  if (limit > 0.0F) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
  }
  out[i] = __float2half(g * u);
}

__global__ void MoeCombineBatchKernel(const __half* __restrict__ parts,
                                      const float* __restrict__ weights,
                                      const float* __restrict__ shared,
                                      float* __restrict__ out,
                                      std::uint32_t used,
                                      std::uint32_t hidden) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t t = blockIdx.y;
  if (c >= hidden)
    return;
  float acc = shared[t * hidden + c];
  for (std::uint32_t e = 0; e < used; ++e) {
    acc += weights[t * used + e] *
           __half2float(parts[(t * used + e) * hidden + c]);
  }
  out[t * hidden + c] = acc;
}

__global__ void EncodeQ8Kernel(const void* __restrict__ src,
                               std::uint8_t* __restrict__ dst,
                               std::size_t blocks, int type) {
  const std::size_t b =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= blocks)
    return;
  float x[32];
  float amax = 0.0F;
  for (int j = 0; j < 32; ++j) {
    x[j] = DequantElement(src, static_cast<std::uint32_t>(b * 32 + j), type);
    amax = fmaxf(amax, fabsf(x[j]));
  }
  const float d = amax / 127.0F;
  const float id = d > 0.0F ? 1.0F / d : 0.0F;
  std::uint8_t* out = dst + b * 34;
  const __half dh = __float2half(d);
  *reinterpret_cast<__half*>(out) = dh;
  for (int j = 0; j < 32; ++j)
    out[2 + j] =
        static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(x[j] * id)));
}

constexpr int kConvChunk = 64;

__global__ void KdaConvBatchKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ cq,
    const float* __restrict__ ck, const float* __restrict__ cv,
    const float* __restrict__ history, float* __restrict__ oq,
    float* __restrict__ ok, float* __restrict__ ov, std::uint32_t channels,
    std::uint32_t n) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= channels)
    return;
  const std::uint32_t t0 = blockIdx.y * kConvChunk;
  const std::uint32_t t1 = min(n, t0 + kConvChunk);
  const float* in[3] = {q, k, v};
  const float* cw[3] = {cq, ck, cv};
  float* out[3] = {oq, ok, ov};
  const std::size_t hist_len = 3 * static_cast<std::size_t>(channels);
  for (int bank = 0; bank < 3; ++bank) {
    const float* tap = cw[bank] + static_cast<std::size_t>(ch) * 4;
    const float* hist = history + bank * hist_len;
    float h[3];
    for (int j = 0; j < 3; ++j) {
      const int src = static_cast<int>(t0) - 3 + j;
      h[j] = src >= 0
                 ? in[bank][static_cast<std::size_t>(src) * channels + ch]
                 : hist[(src + 3) * static_cast<std::size_t>(channels) + ch];
    }
    for (std::uint32_t t = t0; t < t1; ++t) {
      const float raw = in[bank][static_cast<std::size_t>(t) * channels + ch];
      const float acc =
          tap[3] * raw + tap[0] * h[0] + tap[1] * h[1] + tap[2] * h[2];
      out[bank][static_cast<std::size_t>(t) * channels + ch] = Silu(acc);
      h[0] = h[1];
      h[1] = h[2];
      h[2] = raw;
    }
  }
}

__global__ void KdaConvHistKernel(const float* __restrict__ q,
                                  const float* __restrict__ k,
                                  const float* __restrict__ v,
                                  float* __restrict__ history,
                                  std::uint32_t channels, std::uint32_t n) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= channels)
    return;
  const float* in[3] = {q, k, v};
  const std::size_t hist_len = 3 * static_cast<std::size_t>(channels);
  for (int bank = 0; bank < 3; ++bank) {
    float* hist = history + bank * hist_len;
    float h[3];
    for (int j = 0; j < 3; ++j) {
      const int src = static_cast<int>(n) - 3 + j;
      h[j] = src >= 0
                 ? in[bank][static_cast<std::size_t>(src) * channels + ch]
                 : hist[(src + 3) * static_cast<std::size_t>(channels) + ch];
    }
    for (int j = 0; j < 3; ++j)
      hist[j * static_cast<std::size_t>(channels) + ch] = h[j];
  }
}

constexpr int kRecurRowsPerWave = 4;
constexpr int kRecurRowsPerBlock = 32;

__global__ __launch_bounds__(256) void KdaRecurBatchKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ fa,
    const float* __restrict__ beta, const float* __restrict__ dt,
    const float* __restrict__ sa, float* __restrict__ state,
    float* __restrict__ so, std::uint32_t n, std::uint32_t heads,
    float lower_bound, float eps, float* __restrict__ ckpt) {
  const int h = blockIdx.x;
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const std::size_t di = static_cast<std::size_t>(heads) * 128;
  const std::size_t hoff = static_cast<std::size_t>(h) * 128;
  const float a = sa[h];
  const float4 d4 = *reinterpret_cast<const float4*>(dt + hoff + lane * 4);
  float4 s[kRecurRowsPerWave];
  for (int r = 0; r < kRecurRowsPerWave; ++r) {
    const int row =
        blockIdx.y * kRecurRowsPerBlock + wave * kRecurRowsPerWave + r;
    s[r] = *(reinterpret_cast<const float4*>(
                 state + (hoff + static_cast<std::size_t>(row)) * 128) +
             lane);
  }
  const float out_scale = rsqrtf(128.0F);
  for (std::uint32_t t = 0; t < n; ++t) {
    const std::size_t base = t * di + hoff;
    const float4 q4 = *reinterpret_cast<const float4*>(q + base + lane * 4);
    const float4 k4 = *reinterpret_cast<const float4*>(k + base + lane * 4);
    const float4 f4 = *reinterpret_cast<const float4*>(fa + base + lane * 4);
    float qv[4] = {q4.x, q4.y, q4.z, q4.w};
    float kv[4] = {k4.x, k4.y, k4.z, k4.w};
    const float fv[4] = {f4.x + d4.x, f4.y + d4.y, f4.z + d4.z, f4.w + d4.w};
    float dec[4];
    float qq = 0.0F;
    float kk = 0.0F;
    for (int e = 0; e < 4; ++e) {
      dec[e] = __expf(lower_bound * Sigmoid(fv[e] * a));
      qq += qv[e] * qv[e];
      kk += kv[e] * kv[e];
    }
    const float qn = rsqrtf(WaveSum32(qq) + eps);
    const float kn = rsqrtf(WaveSum32(kk) + eps);
    for (int e = 0; e < 4; ++e) {
      qv[e] *= qn;
      kv[e] *= kn;
    }
    const float b = Sigmoid(beta[static_cast<std::size_t>(t) * heads + h]);
    for (int r = 0; r < kRecurRowsPerWave; ++r) {
      const int row =
          blockIdx.y * kRecurRowsPerBlock + wave * kRecurRowsPerWave + r;
      float4 sv = s[r];
      sv.x *= dec[0];
      sv.y *= dec[1];
      sv.z *= dec[2];
      sv.w *= dec[3];
      const float u =
          WaveSum32(sv.x * kv[0] + sv.y * kv[1] + sv.z * kv[2] + sv.w * kv[3]);
      const float delta = (v[base + row] - u) * b;
      sv.x += delta * kv[0];
      sv.y += delta * kv[1];
      sv.z += delta * kv[2];
      sv.w += delta * kv[3];
      const float o =
          WaveSum32(sv.x * qv[0] + sv.y * qv[1] + sv.z * qv[2] + sv.w * qv[3]);
      s[r] = sv;
      if (lane == 0)
        so[base + row] = o * out_scale;
    }
    if (ckpt != nullptr) {
      for (int r = 0; r < kRecurRowsPerWave; ++r) {
        const int row =
            blockIdx.y * kRecurRowsPerBlock + wave * kRecurRowsPerWave + r;
        *(reinterpret_cast<float4*>(
              ckpt + static_cast<std::size_t>(t) * heads * 128 * 128 +
              (hoff + static_cast<std::size_t>(row)) * 128) +
          lane) = s[r];
      }
    }
  }
  for (int r = 0; r < kRecurRowsPerWave; ++r) {
    const int row =
        blockIdx.y * kRecurRowsPerBlock + wave * kRecurRowsPerWave + r;
    *(reinterpret_cast<float4*>(state +
                                (hoff + static_cast<std::size_t>(row)) * 128) +
      lane) = s[r];
  }
}

__global__ __launch_bounds__(256) void MlaKvPrepKernel(
    const float* __restrict__ kvall, const float* __restrict__ norm_w,
    float* __restrict__ cache, std::uint32_t pos0, float eps, float base) {
  __shared__ float scratch[256];
  const std::uint32_t t = blockIdx.x;
  const int tid = threadIdx.x;
  const float* src = kvall + static_cast<std::size_t>(t) * kMlaKey;
  float* dst = cache + static_cast<std::size_t>(pos0 + t) * kMlaKey;
  const float a0 = src[tid];
  const float a1 = src[tid + 256];
  const float sum = BlockSum(a0 * a0 + a1 * a1, scratch, tid, 256);
  const float inv = rsqrtf(sum / kMlaLatent + eps);
  dst[tid] = a0 * inv * norm_w[tid];
  dst[tid + 256] = a1 * inv * norm_w[tid + 256];
  if (tid < 32) {
    const float angle = static_cast<float>(pos0 + t) *
                        powf(base, -2.0F * static_cast<float>(tid) / 64.0F);
    float sn = 0.0F;
    float cs = 0.0F;
    sincosf(angle, &sn, &cs);
    const float a = src[kMlaLatent + tid];
    const float b = src[kMlaLatent + tid + 32];
    dst[kMlaLatent + tid] = a * cs - b * sn;
    dst[kMlaLatent + tid + 32] = a * sn + b * cs;
  }
}

__global__ void RopeQBatchKernel(float* __restrict__ q, std::uint32_t heads,
                                 std::uint32_t stride, std::uint32_t offset,
                                 std::uint32_t pos0, float base) {
  const std::uint32_t t = blockIdx.x / heads;
  const std::uint32_t i = threadIdx.x;
  float* row = q + static_cast<std::size_t>(blockIdx.x) * stride + offset;
  const float angle = static_cast<float>(pos0 + t) *
                      powf(base, -2.0F * static_cast<float>(i) / 64.0F);
  float sn = 0.0F;
  float cs = 0.0F;
  sincosf(angle, &sn, &cs);
  const float a = row[i];
  const float b = row[i + 32];
  row[i] = a * cs - b * sn;
  row[i + 32] = a * sn + b * cs;
}

__global__ void LatentToHalfKernel(const float* __restrict__ cache,
                                   __half* __restrict__ out, std::size_t rows) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < rows * kMlaLatent) {
    const std::size_t r = i / kMlaLatent;
    out[i] = __float2half(cache[r * kMlaKey + i % kMlaLatent]);
  }
}

__global__ void BuildKvbHalfKernel(const void* __restrict__ kb,
                                   const void* __restrict__ vb,
                                   __half* __restrict__ wk,
                                   __half* __restrict__ wv) {
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 4096U * 512U)
    return;
  const std::uint32_t m = idx / 512U;
  const std::uint32_t r = idx % 512U;
  const std::uint32_t h = m / 128U;
  const std::uint32_t c = m % 128U;
  wk[idx] = __float2half(DequantElement(kb, (h * 512U + r) * 128U + c, 8));
  wv[idx] = __float2half(DequantElement(vb, idx, 8));
}

constexpr int kAtTQ = 32;
constexpr int kAtTK = 16;
constexpr int kAtQK = 192;
constexpr int kAtVD = 128;
constexpr int kAtQS = kAtQK + 4;

__global__ __launch_bounds__(256) void MlaAttnKernel(
    const float* __restrict__ q, const float* __restrict__ kn,
    const float* __restrict__ vv, const float* __restrict__ cache,
    float* __restrict__ out, std::uint32_t n, std::uint32_t pos0, float scale) {
  __shared__ __attribute__((aligned(16))) float Qs[kAtTQ][kAtQS];
  __shared__ __attribute__((aligned(16))) float Ks[kAtTK][kAtQS];
  __shared__ __attribute__((aligned(16))) float Vs[kAtTK][kAtVD];
  __shared__ float P[kAtTQ][kAtTK];
  const int tid = threadIdx.x;
  const int h = blockIdx.y;
  const std::uint32_t q0 = blockIdx.x * kAtTQ;
  const int qi = tid / 8;
  const int sub = tid % 8;
  for (int idx = tid; idx < kAtTQ * kAtQK / 4; idx += 256) {
    const int r = idx / (kAtQK / 4);
    const int d4 = idx % (kAtQK / 4);
    const std::uint32_t t = q0 + r;
    float4 v4 = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (t < n) {
      v4 = *reinterpret_cast<const float4*>(
          q + (static_cast<std::size_t>(t) * 32 + h) * kAtQK + d4 * 4);
      v4.x *= scale;
      v4.y *= scale;
      v4.z *= scale;
      v4.w *= scale;
    }
    *reinterpret_cast<float4*>(&Qs[r][d4 * 4]) = v4;
  }
  float m = -INFINITY;
  float l = 0.0F;
  float acc[16];
  for (int j = 0; j < 16; ++j)
    acc[j] = 0.0F;
  const std::uint32_t last_q = min(q0 + kAtTQ, n) - 1;
  const std::uint32_t key_end = pos0 + last_q + 1;
  const std::uint32_t my_t = q0 + qi;
  for (std::uint32_t k0 = 0; k0 < key_end; k0 += kAtTK) {
    __syncthreads();
    for (int idx = tid; idx < kAtTK * kAtQK / 4; idx += 256) {
      const int r = idx / (kAtQK / 4);
      const int d4 = idx % (kAtQK / 4);
      const std::uint32_t s = k0 + r;
      float4 val = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (s < key_end) {
        val = d4 < 32 ? *reinterpret_cast<const float4*>(
                            kn + (static_cast<std::size_t>(s) * 32 + h) * 128 +
                            d4 * 4)
                      : *reinterpret_cast<const float4*>(
                            cache + static_cast<std::size_t>(s) * kMlaKey +
                            kMlaLatent + (d4 - 32) * 4);
      }
      *reinterpret_cast<float4*>(&Ks[r][d4 * 4]) = val;
    }
    for (int idx = tid; idx < kAtTK * kAtVD / 4; idx += 256) {
      const int r = idx / (kAtVD / 4);
      const int d4 = idx % (kAtVD / 4);
      const std::uint32_t s = k0 + r;
      float4 val = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (s < key_end)
        val = *reinterpret_cast<const float4*>(
            vv + (static_cast<std::size_t>(s) * 32 + h) * 128 + d4 * 4);
      *reinterpret_cast<float4*>(&Vs[r][d4 * 4]) = val;
    }
    __syncthreads();
    float d0 = 0.0F;
    float d1 = 0.0F;
    const float4* qrow = reinterpret_cast<const float4*>(&Qs[qi][0]);
    const float4* k0row = reinterpret_cast<const float4*>(&Ks[sub * 2][0]);
    const float4* k1row = reinterpret_cast<const float4*>(&Ks[sub * 2 + 1][0]);
#pragma unroll 8
    for (int d = 0; d < kAtQK / 4; ++d) {
      const float4 qv = qrow[d];
      const float4 a = k0row[d];
      const float4 b = k1row[d];
      d0 += qv.x * a.x + qv.y * a.y + qv.z * a.z + qv.w * a.w;
      d1 += qv.x * b.x + qv.y * b.y + qv.z * b.z + qv.w * b.w;
    }
    const std::uint32_t s0 = k0 + sub * 2;
    const float sc0 =
        (s0 <= pos0 + my_t && my_t < n && s0 < key_end) ? d0 : -INFINITY;
    const float sc1 = (s0 + 1 <= pos0 + my_t && my_t < n && s0 + 1 < key_end)
                          ? d1
                          : -INFINITY;
    float mx = fmaxf(sc0, sc1);
    for (int off = 4; off > 0; off >>= 1)
      mx = fmaxf(mx, __shfl_xor(mx, off, 8));
    const float mn = fmaxf(m, mx);
    const float corr = mn == -INFINITY ? 1.0F : __expf(m - mn);
    const float p0 = sc0 == -INFINITY ? 0.0F : __expf(sc0 - mn);
    const float p1 = sc1 == -INFINITY ? 0.0F : __expf(sc1 - mn);
    float ps = p0 + p1;
    for (int off = 4; off > 0; off >>= 1)
      ps += __shfl_xor(ps, off, 8);
    l = l * corr + ps;
    m = mn;
    P[qi][sub * 2] = p0;
    P[qi][sub * 2 + 1] = p1;
    for (int j = 0; j < 16; ++j)
      acc[j] *= corr;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kAtTK; ++kk) {
      const float p = P[qi][kk];
      const float4* vrow = reinterpret_cast<const float4*>(&Vs[kk][sub * 16]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float4 vq = vrow[j];
        acc[j * 4 + 0] += p * vq.x;
        acc[j * 4 + 1] += p * vq.y;
        acc[j * 4 + 2] += p * vq.z;
        acc[j * 4 + 3] += p * vq.w;
      }
    }
  }
  if (my_t < n) {
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    float4* dst = reinterpret_cast<float4*>(
        out + (static_cast<std::size_t>(my_t) * 32 + h) * kAtVD + sub * 16);
    for (int j = 0; j < 4; ++j) {
      dst[j] = make_float4(acc[j * 4] * inv, acc[j * 4 + 1] * inv,
                           acc[j * 4 + 2] * inv, acc[j * 4 + 3] * inv);
    }
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

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__global__ void BuildKvHalfKernel(const float* __restrict__ kn,
                                  const float* __restrict__ vv,
                                  const float* __restrict__ cache,
                                  __half* __restrict__ k16,
                                  __half* __restrict__ v16, std::size_t rows) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // one thread per (row, head, d4) over K (48 float4 groups) and V (32)
  const std::size_t groups = rows * 32 * 80;
  if (i >= groups)
    return;
  const std::size_t g = i % 80;
  const std::size_t h = (i / 80) % 32;
  const std::size_t r = i / (80 * 32);
  if (g < 48) {
    const float4 v4 =
        g < 32
            ? *reinterpret_cast<const float4*>(kn + (r * 32 + h) * 128 + g * 4)
            : *reinterpret_cast<const float4*>(cache + r * kMlaKey +
                                               kMlaLatent + (g - 32) * 4);
    __half* dst = k16 + (r * 32 + h) * 192 + g * 4;
    dst[0] = __float2half(v4.x);
    dst[1] = __float2half(v4.y);
    dst[2] = __float2half(v4.z);
    dst[3] = __float2half(v4.w);
  } else {
    const std::size_t gv = g - 48;
    const float4 v4 =
        *reinterpret_cast<const float4*>(vv + (r * 32 + h) * 128 + gv * 4);
    __half* dst = v16 + (r * 32 + h) * 128 + gv * 4;
    dst[0] = __float2half(v4.x);
    dst[1] = __float2half(v4.y);
    dst[2] = __float2half(v4.z);
    dst[3] = __float2half(v4.w);
  }
}

constexpr int kWQ = 64;
constexpr int kWK = 32;
constexpr int kKS = 200;
constexpr int kVS = 40;
constexpr int kPS = 40;

__global__ __launch_bounds__(128) void MlaWmmaAttnKernel(
    const __half* __restrict__ q16, const __half* __restrict__ k16,
    const __half* __restrict__ v16, float* __restrict__ out, std::uint32_t n,
    std::uint32_t pos0, float scale) {
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
            q16 + (static_cast<std::size_t>(t) * 32 + h) * 192 + ks * 16);
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
  const std::uint32_t key_end = pos0 + last_q + 1;
  for (std::uint32_t k0 = 0; k0 < key_end; k0 += kWK) {
    __syncthreads();
    for (int idx = tid; idx < kWK * 24; idx += 128) {
      const int r = idx / 24;
      const int c8 = idx % 24;
      const std::uint32_t s = k0 + r;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (s < key_end)
        val = *reinterpret_cast<const uint4*>(
            k16 + (static_cast<std::size_t>(s) * 32 + h) * 192 + c8 * 8);
      *reinterpret_cast<uint4*>(&Ks[r][c8 * 8]) = val;
    }
    for (int idx = tid; idx < kWK * 16; idx += 128) {
      const int r = idx / 16;
      const int c8 = idx % 16;
      const std::uint32_t s = k0 + r;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (s < key_end)
        val = *reinterpret_cast<const uint4*>(
            v16 + (static_cast<std::size_t>(s) * 32 + h) * 128 + c8 * 8);
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
      if (!(ok_row && s0 <= pos0 + t && s0 < key_end))
        v0 = -INFINITY;
      if (!(ok_row && s1 <= pos0 + t && s1 < key_end))
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
      out[(static_cast<std::size_t>(t) * 32 + h) * 128 + dt * 16 + col] =
          o[dt][i] * inv;
  }
}

__global__ void KdaHistRestoreKernel(const float* __restrict__ old_hist,
                                     const float* __restrict__ raw,
                                     float* __restrict__ hist,
                                     std::uint32_t channels, std::uint32_t a) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= channels)
    return;
  const std::size_t hist_len = 3 * static_cast<std::size_t>(channels);
  for (int bank = 0; bank < 3; ++bank) {
    float v[3];
    for (int j = 0; j < 3; ++j) {
      const int idx = static_cast<int>(a) - 2 + j;
      v[j] =
          idx >= 0
              ? raw[(static_cast<std::size_t>(bank) * 8 + idx) * channels + ch]
              : old_hist[bank * hist_len +
                         (idx + 3) * static_cast<std::size_t>(channels) + ch];
    }
    for (int j = 0; j < 3; ++j)
      hist[bank * hist_len + j * static_cast<std::size_t>(channels) + ch] =
          v[j];
  }
}

__global__ void MoeCombineBatchF32Kernel(const float* __restrict__ parts,
                                         const float* __restrict__ weights,
                                         const float* __restrict__ shared,
                                         float* __restrict__ out,
                                         std::uint32_t used,
                                         std::uint32_t hidden) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t t = blockIdx.y;
  if (c >= hidden)
    return;
  float acc = shared[t * hidden + c];
  for (std::uint32_t e = 0; e < used; ++e)
    acc += weights[t * used + e] * parts[(t * used + e) * hidden + c];
  out[t * hidden + c] = acc;
}

constexpr int kPoolBlock = 64;
constexpr int kMaxPoolBlocks = 4096;
constexpr int kSelThreads = 256;
constexpr int kSelPer = kMaxPoolBlocks / kSelThreads;

__global__ __launch_bounds__(256) void PoolUpdateKernel(
    const float* __restrict__ cache, float* __restrict__ pool,
    std::uint32_t first_block) {
  const std::uint32_t b = first_block + blockIdx.x;
  for (int d = threadIdx.x; d < kMlaKey; d += 256) {
    float mn = INFINITY;
    float mx = -INFINITY;
    for (int r = 0; r < kPoolBlock; ++r) {
      const float v =
          cache[(static_cast<std::size_t>(b) * kPoolBlock + r) * kMlaKey + d];
      mn = fminf(mn, v);
      mx = fmaxf(mx, v);
    }
    pool[(static_cast<std::size_t>(b) * 2) * kMlaKey + d] = mn;
    pool[(static_cast<std::size_t>(b) * 2 + 1) * kMlaKey + d] = mx;
  }
}

__global__ __launch_bounds__(256) void ScoreBlocksKernel(
    const float* __restrict__ pool, const float* __restrict__ qvec,
    float* __restrict__ scores) {
  __shared__ float scratch[256];
  const std::uint32_t b = blockIdx.x;
  float acc = 0.0F;
  for (int d = threadIdx.x; d < kMlaKey; d += 256) {
    const float mn = pool[(static_cast<std::size_t>(b) * 2) * kMlaKey + d];
    const float mx = pool[(static_cast<std::size_t>(b) * 2 + 1) * kMlaKey + d];
    for (int h = 0; h < kMlaHeads; ++h) {
      const float q = qvec[h * kMlaKey + d];
      acc += fmaxf(q * mn, q * mx);
    }
  }
  const float total = BlockSum(acc, scratch, threadIdx.x, 256);
  if (threadIdx.x == 0)
    scores[b] = total;
}

__device__ __forceinline__ std::uint32_t OrderedKey(float v) {
  const std::uint32_t u = __float_as_uint(v);
  return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

__global__ __launch_bounds__(kSelThreads) void SelectTopKernel(
    float* __restrict__ scores, std::uint32_t nb, std::uint32_t budget,
    std::uint32_t recent, std::int32_t* __restrict__ sel) {
  __shared__ std::uint32_t wc[kSelThreads / 32];
  const int tid = threadIdx.x;
  for (std::uint32_t i = tid; i < nb; i += kSelThreads) {
    if (i == 0 || i + recent >= nb)
      scores[i] = INFINITY;
  }
  __syncthreads();
  std::uint32_t lo = 0;
  std::uint32_t hi = 0xFFFFFFFFU;
  // Largest key T with at least `budget` entries >= T.
  for (int iter = 0; iter < 32; ++iter) {
    const std::uint32_t mid = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(lo) + hi + 1ULL) >> 1U);
    std::uint32_t cnt = 0;
    for (std::uint32_t i = tid; i < nb; i += kSelThreads)
      cnt += OrderedKey(scores[i]) >= mid ? 1U : 0U;
    for (int off = 16; off > 0; off >>= 1)
      cnt += __shfl_xor(cnt, off);
    if ((tid & 31) == 0)
      wc[tid >> 5] = cnt;
    __syncthreads();
    std::uint32_t total = 0;
    for (int w = 0; w < kSelThreads / 32; ++w)
      total += wc[w];
    __syncthreads();
    if (total >= budget)
      lo = mid;
    else
      hi = mid - 1U;
  }
  // Index-ordered compaction: each thread owns 4 consecutive blocks.
  __shared__ std::uint32_t scan[kSelThreads];
  const std::uint32_t base = static_cast<std::uint32_t>(tid) * kSelPer;
  std::uint32_t gt = 0;
  std::uint32_t eq = 0;
  for (std::uint32_t j = 0; j < kSelPer; ++j) {
    const std::uint32_t i = base + j;
    if (i < nb) {
      const std::uint32_t key = OrderedKey(scores[i]);
      gt += key > lo ? 1U : 0U;
      eq += key == lo ? 1U : 0U;
    }
  }
  scan[tid] = gt;
  __syncthreads();
  for (int off = 1; off < kSelThreads; off <<= 1) {
    const std::uint32_t v = tid >= off ? scan[tid - off] : 0U;
    __syncthreads();
    scan[tid] += v;
    __syncthreads();
  }
  const std::uint32_t gt_before = scan[tid] - gt;
  const std::uint32_t greater = min(scan[kSelThreads - 1], budget);
  __syncthreads();
  scan[tid] = eq;
  __syncthreads();
  for (int off = 1; off < kSelThreads; off <<= 1) {
    const std::uint32_t v = tid >= off ? scan[tid - off] : 0U;
    __syncthreads();
    scan[tid] += v;
    __syncthreads();
  }
  const std::uint32_t eq_before = scan[tid] - eq;
  std::uint32_t gp = gt_before;
  std::uint32_t ep = eq_before;
  for (std::uint32_t j = 0; j < kSelPer; ++j) {
    const std::uint32_t i = base + j;
    if (i >= nb)
      continue;
    const std::uint32_t key = OrderedKey(scores[i]);
    if (key > lo) {
      if (gp < budget)
        sel[gp] = static_cast<std::int32_t>(i);
      ++gp;
    } else if (key == lo) {
      if (greater + ep < budget)
        sel[greater + ep] = static_cast<std::int32_t>(i);
      ++ep;
    }
  }
}

__global__ __launch_bounds__(144) void GatherLatKernel(
    const float* __restrict__ cache, const std::int32_t* __restrict__ sel,
    float* __restrict__ out, std::uint32_t budget, std::uint32_t tail_start) {
  const std::size_t r = blockIdx.x;
  const std::size_t src_row =
      r < static_cast<std::size_t>(budget) * kPoolBlock
          ? static_cast<std::size_t>(
                max(0, min(sel[r / kPoolBlock],
                           static_cast<int>(tail_start / kPoolBlock) - 1))) *
                    kPoolBlock +
                r % kPoolBlock
          : tail_start + (r - static_cast<std::size_t>(budget) * kPoolBlock);
  reinterpret_cast<float4*>(out + r * kMlaKey)[threadIdx.x] =
      reinterpret_cast<const float4*>(cache + src_row * kMlaKey)[threadIdx.x];
}

__global__ void ProbeRopeKernel(const float* __restrict__ qrow,
                                float* __restrict__ qvec) {
  const int h = blockIdx.x;
  const int i = threadIdx.x;
  qvec[h * kMlaKey + kMlaLatent + i] = qrow[h * 192 + 128 + i];
}

__global__ __launch_bounds__(256) void SgemmKernel(const float* __restrict__ w,
                                                   const float* __restrict__ x,
                                                   float* __restrict__ out,
                                                   int m, int n, int k) {
  __shared__ float Ws[16][33];
  __shared__ float Xs[16][33];
  const int tx = threadIdx.x % 16;
  const int ty = threadIdx.x / 16;
  const int m0 = blockIdx.x * 32;
  const int n0 = blockIdx.y * 32;
  float acc[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
  for (int k0 = 0; k0 < k; k0 += 16) {
    for (int e = threadIdx.x; e < 512; e += 256) {
      const int r = e / 16;
      const int c = e % 16;
      Ws[c][r] = (m0 + r < m && k0 + c < k)
                     ? w[static_cast<std::size_t>(m0 + r) * k + k0 + c]
                     : 0.0F;
      Xs[c][r] = (n0 + r < n && k0 + c < k)
                     ? x[static_cast<std::size_t>(n0 + r) * k + k0 + c]
                     : 0.0F;
    }
    __syncthreads();
    for (int kk = 0; kk < 16; ++kk) {
      const float a0 = Ws[kk][tx * 2];
      const float a1 = Ws[kk][tx * 2 + 1];
      const float b0 = Xs[kk][ty * 2];
      const float b1 = Xs[kk][ty * 2 + 1];
      acc[0][0] += a0 * b0;
      acc[1][0] += a1 * b0;
      acc[0][1] += a0 * b1;
      acc[1][1] += a1 * b1;
    }
    __syncthreads();
  }
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const int row = m0 + tx * 2 + i;
      const int tok = n0 + ty * 2 + j;
      if (row < m && tok < n)
        out[static_cast<std::size_t>(tok) * m + row] = acc[i][j];
    }
  }
}

// out[h][r] = sum_c W[h][r][c] * x[h * x_stride + c]; W is Q8_0 with
// `rows` rows of `cols` per head.
__global__ __launch_bounds__(256) void HeadGemvQ8Kernel(
    const std::uint8_t* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, std::uint32_t rows, std::uint32_t cols,
    std::uint32_t x_stride, std::uint32_t out_stride) {
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const std::uint32_t row = blockIdx.x * 8 + wave;
  const std::uint32_t h = blockIdx.y;
  if (row >= rows)
    return;
  const std::size_t row_bytes = (cols / 32) * 34;
  const std::uint8_t* src =
      w + (static_cast<std::size_t>(h) * rows + row) * row_bytes;
  const float* xh = x + static_cast<std::size_t>(h) * x_stride;
  float acc = 0.0F;
  for (std::uint32_t b = 0; b < cols / 32; ++b) {
    const std::uint8_t* blk = src + b * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const float q = static_cast<float>(static_cast<std::int8_t>(blk[2 + lane]));
    acc += d * q * xh[b * 32 + lane];
  }
  acc = WaveSum32(acc);
  if (lane == 0)
    out[static_cast<std::size_t>(h) * out_stride + row] = acc;
}

__global__ __launch_bounds__(256) void MlaDecodeSplitKernel(
    const float* __restrict__ qvec, const float* __restrict__ cache,
    float* __restrict__ part_m, float* __restrict__ part_l,
    float* __restrict__ part_acc, const std::int32_t* __restrict__ pos_ptr,
    std::uint32_t splits, float scale) {
  const std::uint32_t rows = static_cast<std::uint32_t>(*pos_ptr) + 1;
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const std::uint32_t split = blockIdx.x;
  const std::uint32_t chunk = (rows + splits - 1) / splits;
  const std::uint32_t begin = split * chunk;
  const std::uint32_t end = min(rows, begin + chunk);
  float q_lat[kMlaHeadsPerWave][kMlaLaneLatent];
  float q_rope[kMlaHeadsPerWave][kMlaLaneRope];
  float m[kMlaHeadsPerWave];
  float l[kMlaHeadsPerWave];
  float acc[kMlaHeadsPerWave][kMlaLaneLatent];
  for (int j = 0; j < kMlaHeadsPerWave; ++j) {
    const float* qh =
        qvec + static_cast<std::size_t>(wave * kMlaHeadsPerWave + j) * kMlaKey;
    for (int e = 0; e < kMlaLaneLatent; ++e) {
      q_lat[j][e] = qh[lane * kMlaLaneLatent + e] * scale;
      acc[j][e] = 0.0F;
    }
    for (int e = 0; e < kMlaLaneRope; ++e)
      q_rope[j][e] = qh[kMlaLatent + lane * kMlaLaneRope + e] * scale;
    m[j] = -INFINITY;
    l[j] = 0.0F;
  }
  for (std::uint32_t r = begin; r < end; ++r) {
    const float* row = cache + static_cast<std::size_t>(r) * kMlaKey;
    float lat[kMlaLaneLatent];
    float rp[kMlaLaneRope];
    const float4* lat4 =
        reinterpret_cast<const float4*>(row + lane * kMlaLaneLatent);
    for (int e = 0; e < kMlaLaneLatent / 4; ++e) {
      const float4 v = lat4[e];
      lat[e * 4 + 0] = v.x;
      lat[e * 4 + 1] = v.y;
      lat[e * 4 + 2] = v.z;
      lat[e * 4 + 3] = v.w;
    }
    for (int e = 0; e < kMlaLaneRope; ++e)
      rp[e] = row[kMlaLatent + lane * kMlaLaneRope + e];
    for (int j = 0; j < kMlaHeadsPerWave; ++j) {
      float dot = 0.0F;
      for (int e = 0; e < kMlaLaneLatent; ++e)
        dot += q_lat[j][e] * lat[e];
      for (int e = 0; e < kMlaLaneRope; ++e)
        dot += q_rope[j][e] * rp[e];
      dot = WaveSum32(dot);
      const float mn = fmaxf(m[j], dot);
      const float corr = __expf(m[j] - mn);
      const float p = __expf(dot - mn);
      l[j] = l[j] * corr + p;
      for (int e = 0; e < kMlaLaneLatent; ++e)
        acc[j][e] = acc[j][e] * corr + p * lat[e];
      m[j] = mn;
    }
  }
  for (int j = 0; j < kMlaHeadsPerWave; ++j) {
    const int h = wave * kMlaHeadsPerWave + j;
    if (lane == 0) {
      part_m[split * kMlaHeads + h] = m[j];
      part_l[split * kMlaHeads + h] = l[j];
    }
    float* dst =
        part_acc +
        (static_cast<std::size_t>(split) * kMlaHeads + h) * kMlaLatent +
        lane * kMlaLaneLatent;
    for (int e = 0; e < kMlaLaneLatent; ++e)
      dst[e] = acc[j][e];
  }
}

__global__ __launch_bounds__(256) void MlaDecodeCombineKernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, float* __restrict__ out,
    std::uint32_t splits) {
  const int h = blockIdx.x;
  float big = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s)
    big = fmaxf(big, part_m[s * kMlaHeads + h]);
  float denom = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s)
    denom +=
        part_l[s * kMlaHeads + h] * __expf(part_m[s * kMlaHeads + h] - big);
  for (int d = threadIdx.x; d < kMlaLatent; d += blockDim.x) {
    float a = 0.0F;
    for (std::uint32_t s = 0; s < splits; ++s) {
      a += part_acc[(static_cast<std::size_t>(s) * kMlaHeads + h) * kMlaLatent +
                    d] *
           __expf(part_m[s * kMlaHeads + h] - big);
    }
    out[static_cast<std::size_t>(h) * kMlaLatent + d] = a / denom;
  }
}

}  // namespace

void AddTo(float* x, const float* y, std::uint32_t n, hipStream_t stream) {
  hipLaunchKernelGGL(AddToKernel, dim3((n + kElemThreads - 1) / kElemThreads),
                     dim3(kElemThreads), 0, stream, x, y, n);
}

void F32GemvDouble(const float* w, const float* x, float* out,
                   std::uint32_t rows, std::uint32_t cols, hipStream_t stream) {
  F32GemvDoubleKernel<<<rows, kElemThreads, 0, stream>>>(w, x, out, cols);
}

void DequantRows(const void* src, float* dst, std::uint32_t elements, int type,
                 hipStream_t stream) {
  constexpr std::uint32_t threads = 256;
  constexpr std::uint32_t max_blocks = 4096;
  const std::uint32_t blocks =
      std::min((elements + threads - 1) / threads, max_blocks);
  DequantKernel<<<blocks, threads, 0, stream>>>(src, dst, elements, type);
}

void DequantExperts(const void* src, const std::int32_t* ids, float* dst,
                    std::uint32_t slots, std::uint32_t elements_per_slot,
                    std::uint32_t expert_bytes, int type, hipStream_t stream) {
  constexpr std::uint32_t threads = 256;
  constexpr std::uint32_t max_blocks = 4096;
  const std::uint32_t n = slots * elements_per_slot;
  const std::uint32_t blocks =
      std::min((n + threads - 1) / threads, max_blocks);
  DequantExpertsKernel<<<blocks, threads, 0, stream>>>(
      src, ids, dst, slots, elements_per_slot, expert_bytes, type);
}

void RopeHeads(const float* in, float* out, std::uint32_t heads,
               std::uint32_t stride, std::uint32_t offset, std::uint32_t dim,
               std::uint32_t pos, float base, hipStream_t stream,
               std::uint32_t out_stride, const std::int32_t* pos_ptr,
               std::uint32_t out_pos_stride) {
  hipLaunchKernelGGL(RopeHeadsKernel, dim3(heads), dim3(kMaxKdaDim), 0, stream,
                     in, out, stride, offset, dim, pos, base,
                     out_stride == 0 ? dim : out_stride, pos_ptr,
                     out_pos_stride);
}

void CopyRow(const float* src, float* dst, std::uint32_t n,
             hipStream_t stream) {
  hipLaunchKernelGGL(CopyRowKernel, dim3((n + kElemThreads - 1) / kElemThreads),
                     dim3(kElemThreads), 0, stream, src, dst, n);
}

void AppendRow(const float* row, float* cache, std::uint32_t pos,
               std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(AppendRowKernel,
                     dim3((dim + kElemThreads - 1) / kElemThreads),
                     dim3(kElemThreads), 0, stream, row, cache, pos, dim);
}

void KdaConv(const float* q, const float* k, const float* v, const float* cq,
             const float* ck, const float* cv, float* history, float* oq,
             float* ok, float* ov, std::uint32_t channels, std::uint32_t taps,
             hipStream_t stream) {
  hipLaunchKernelGGL(KdaConvKernel,
                     dim3((channels + kElemThreads - 1) / kElemThreads),
                     dim3(kElemThreads), 0, stream, q, k, v, cq, ck, cv,
                     history, oq, ok, ov, channels, taps);
}

void KdaStep(const float* q, const float* k, const float* v, const float* fa,
             const float* ga, const float* beta, const float* dt,
             const float* sa, const float* norm_w, float* state, float* out,
             std::uint32_t heads, std::uint32_t dim, float lower_bound,
             float eps, hipStream_t stream) {
  hipLaunchKernelGGL(KdaStepKernel, dim3(heads), dim3(dim), 0, stream, q, k, v,
                     fa, ga, beta, dt, sa, norm_w, state, out, dim, lower_bound,
                     eps);
}

void MlaScores(const float* q, const float* cache, float* scores,
               std::uint32_t heads, std::uint32_t rows, std::uint32_t dim,
               float scale, hipStream_t stream) {
  hipLaunchKernelGGL(MlaScoresKernel, dim3(heads), dim3(kScoreThreads), 0,
                     stream, q, cache, scores, rows, dim, scale);
}

void MlaGather(const float* scores, const float* cache, float* ctx,
               std::uint32_t heads, std::uint32_t rows, std::uint32_t dim,
               std::uint32_t value_dim, hipStream_t stream) {
  hipLaunchKernelGGL(MlaGatherKernel, dim3(heads), dim3(kElemThreads), 0,
                     stream, scores, cache, ctx, rows, dim, value_dim);
}

void MlaGate(const float* value, const float* gate, float* out,
             std::uint32_t heads, std::uint32_t head_dim, hipStream_t stream) {
  const std::uint32_t n = heads * head_dim;
  hipLaunchKernelGGL(MlaGateKernel, dim3((n + kElemThreads - 1) / kElemThreads),
                     dim3(kElemThreads), 0, stream, value, gate, out, n,
                     head_dim);
}

void RouterLing(const float* logits, const float* bias, std::int32_t* ids,
                float* weights, std::uint32_t experts, std::uint32_t groups,
                std::uint32_t groups_used, std::uint32_t used, float scale,
                hipStream_t stream, std::uint32_t tokens) {
  hipLaunchKernelGGL(RouterLingKernel, dim3(tokens), dim3(kMaxExperts), 0,
                     stream, logits, bias, ids, weights, experts, groups,
                     groups_used, used, scale);
}

void SwigluClamp(const float* gate, const float* up, float* out,
                 std::uint32_t n, float limit, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluClampKernel,
                     dim3((n + kElemThreads - 1) / kElemThreads),
                     dim3(kElemThreads), 0, stream, gate, up, out, n, limit);
}

void SwigluPlain(const float* gate, const float* up, float* out,
                 std::uint32_t n, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluPlainKernel,
                     dim3((n + kElemThreads - 1) / kElemThreads),
                     dim3(kElemThreads), 0, stream, gate, up, out, n);
}

void MoeCombine(const float* parts, const float* weights, const float* shared,
                float* out, std::uint32_t used, std::uint32_t hidden,
                hipStream_t stream) {
  hipLaunchKernelGGL(
      MoeCombineKernel, dim3((hidden + kElemThreads - 1) / kElemThreads),
      dim3(kElemThreads), 0, stream, parts, weights, shared, out, used, hidden);
}

void KdaDecode(const float* q, const float* k, const float* v, const float* fa,
               const float* ga, const float* beta, const float* dt,
               const float* sa, const float* norm_w, float* state, float* so,
               float* out, std::uint32_t heads, float lower_bound, float eps,
               hipStream_t stream) {
  hipLaunchKernelGGL(KdaDecodeKernel, dim3(heads, 128 / kKdaRowsPerBlock),
                     dim3(256), 0, stream, q, k, v, fa, beta, dt, sa, state, so,
                     lower_bound, eps);
  hipLaunchKernelGGL(KdaDecodeOutKernel, dim3(heads), dim3(128), 0, stream, so,
                     ga, norm_w, out, eps);
}

__global__ void SplitHiLoKernel(const float* __restrict__ in,
                                __half* __restrict__ out, std::size_t rows,
                                std::uint32_t k, int mode) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= rows * k)
    return;
  const std::size_t r = i / k;
  const std::uint32_t j = static_cast<std::uint32_t>(i % k);
  const float v = in[i];
  const __half hi = __float2half(v);
  const __half lo = __float2half(v - __half2float(hi));
  __half* o = out + r * 3U * k + j;
  if (mode == 0) {
    o[0] = hi;
    o[k] = lo;
    o[2U * k] = hi;
  } else {
    o[0] = hi;
    o[k] = hi;
    o[2U * k] = lo;
  }
}

void SplitHiLo(const float* in, __half* out, std::size_t rows, std::uint32_t k,
               bool weight, hipStream_t stream) {
  const std::size_t total = rows * k;
  hipLaunchKernelGGL(SplitHiLoKernel,
                     dim3(static_cast<unsigned>((total + 255) / 256)),
                     dim3(256), 0, stream, in, out, rows, k, weight ? 1 : 0);
}

void NarrowRows(const float* in, __half* out, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(NarrowRowsKernel,
                     dim3(static_cast<unsigned>((count + 255) / 256)),
                     dim3(256), 0, stream, in, out, count);
}

void SwigluHalf(const float* gate, const float* up, __half* out, std::size_t n,
                float limit, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluHalfKernel,
                     dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0,
                     stream, gate, up, out, n, limit);
}

void MoeCombineBatch(const __half* parts, const float* weights,
                     const float* shared, float* out, std::uint32_t tokens,
                     std::uint32_t used, std::uint32_t hidden,
                     hipStream_t stream) {
  hipLaunchKernelGGL(MoeCombineBatchKernel, dim3((hidden + 255) / 256, tokens),
                     dim3(256), 0, stream, parts, weights, shared, out, used,
                     hidden);
}

void EncodeQ8(const void* src, void* dst, std::size_t blocks, int type,
              hipStream_t stream) {
  hipLaunchKernelGGL(
      EncodeQ8Kernel, dim3(static_cast<unsigned>((blocks + 127) / 128)),
      dim3(128), 0, stream, src, static_cast<std::uint8_t*>(dst), blocks, type);
}

void KdaConvBatch(const float* q, const float* k, const float* v,
                  const float* cq, const float* ck, const float* cv,
                  float* history, float* oq, float* ok, float* ov,
                  std::uint32_t channels, std::uint32_t n, hipStream_t stream) {
  hipLaunchKernelGGL(
      KdaConvBatchKernel,
      dim3((channels + 255) / 256, (n + kConvChunk - 1) / kConvChunk),
      dim3(256), 0, stream, q, k, v, cq, ck, cv, history, oq, ok, ov, channels,
      n);
  hipLaunchKernelGGL(KdaConvHistKernel, dim3((channels + 255) / 256), dim3(256),
                     0, stream, q, k, v, history, channels, n);
}

void KdaRecurBatch(const float* q, const float* k, const float* v,
                   const float* fa, const float* ga, const float* beta,
                   const float* dt, const float* sa, const float* norm_w,
                   float* state, float* so, float* out, std::uint32_t n,
                   std::uint32_t heads, float lower_bound, float eps,
                   hipStream_t stream, float* ckpt) {
  hipLaunchKernelGGL(KdaRecurBatchKernel, dim3(heads, 128 / kRecurRowsPerBlock),
                     dim3(256), 0, stream, q, k, v, fa, beta, dt, sa, state, so,
                     n, heads, lower_bound, eps, ckpt);
  hipLaunchKernelGGL(KdaDecodeOutKernel, dim3(n * heads), dim3(128), 0, stream,
                     so, ga, norm_w, out, eps);
}

void MlaKvPrep(const float* kvall, const float* norm_w, float* cache,
               std::uint32_t pos0, std::uint32_t n, float eps, float base,
               hipStream_t stream) {
  hipLaunchKernelGGL(MlaKvPrepKernel, dim3(n), dim3(256), 0, stream, kvall,
                     norm_w, cache, pos0, eps, base);
}

void RopeQBatch(float* q, std::uint32_t n, std::uint32_t heads,
                std::uint32_t stride, std::uint32_t offset, std::uint32_t pos0,
                float base, hipStream_t stream) {
  hipLaunchKernelGGL(RopeQBatchKernel, dim3(n * heads), dim3(32), 0, stream, q,
                     heads, stride, offset, pos0, base);
}

void LatentToHalf(const float* cache, __half* out, std::size_t rows,
                  hipStream_t stream) {
  const std::size_t total = rows * kMlaLatent;
  hipLaunchKernelGGL(LatentToHalfKernel,
                     dim3(static_cast<unsigned>((total + 255) / 256)),
                     dim3(256), 0, stream, cache, out, rows);
}

void BuildKvbHalf(const void* kb, const void* vb, __half* wk, __half* wv,
                  hipStream_t stream) {
  hipLaunchKernelGGL(BuildKvbHalfKernel, dim3((4096U * 512U + 255) / 256),
                     dim3(256), 0, stream, kb, vb, wk, wv);
}

void MlaAttention(const float* q, const float* kn, const float* vv,
                  const float* cache, float* out, std::uint32_t n,
                  std::uint32_t pos0, float scale, hipStream_t stream) {
  hipLaunchKernelGGL(MlaAttnKernel, dim3((n + kAtTQ - 1) / kAtTQ, 32),
                     dim3(256), 0, stream, q, kn, vv, cache, out, n, pos0,
                     scale);
}

void BuildTiles(const std::uint32_t* counts, std::uint32_t experts,
                std::int32_t* t48, std::uint32_t cap48, std::int32_t* t64,
                std::uint32_t cap64, std::int32_t* t128, std::uint32_t cap128,
                hipStream_t stream) {
  hipLaunchKernelGGL(BuildTilesKernel, dim3(1), dim3(512), 0, stream, counts,
                     static_cast<int>(experts), t48, static_cast<int>(cap48),
                     t64, static_cast<int>(cap64), t128,
                     static_cast<int>(cap128));
}

void BuildKvHalf(const float* kn, const float* vv, const float* cache,
                 __half* k16, __half* v16, std::size_t rows,
                 hipStream_t stream) {
  const std::size_t groups = rows * 32 * 80;
  hipLaunchKernelGGL(BuildKvHalfKernel,
                     dim3(static_cast<unsigned>((groups + 255) / 256)),
                     dim3(256), 0, stream, kn, vv, cache, k16, v16, rows);
}

void MlaWmmaAttention(const __half* q16, const __half* k16, const __half* v16,
                      float* out, std::uint32_t n, std::uint32_t pos0,
                      float scale, hipStream_t stream) {
  hipLaunchKernelGGL(MlaWmmaAttnKernel, dim3((n + kWQ - 1) / kWQ, 32),
                     dim3(128), 0, stream, q16, k16, v16, out, n, pos0, scale);
}

void KdaHistRestore(const float* old_hist, const float* raw, float* hist,
                    std::uint32_t channels, std::uint32_t a,
                    hipStream_t stream) {
  hipLaunchKernelGGL(KdaHistRestoreKernel, dim3((channels + 255) / 256),
                     dim3(256), 0, stream, old_hist, raw, hist, channels, a);
}

void MoeCombineBatchF32(const float* parts, const float* weights,
                        const float* shared, float* out, std::uint32_t tokens,
                        std::uint32_t used, std::uint32_t hidden,
                        hipStream_t stream) {
  hipLaunchKernelGGL(MoeCombineBatchF32Kernel,
                     dim3((hidden + 255) / 256, tokens), dim3(256), 0, stream,
                     parts, weights, shared, out, used, hidden);
}

void PoolUpdate(const float* cache, float* pool, std::uint32_t first_block,
                std::uint32_t count, hipStream_t stream) {
  if (count == 0)
    return;
  hipLaunchKernelGGL(PoolUpdateKernel, dim3(count), dim3(256), 0, stream, cache,
                     pool, first_block);
}

void SparseSelect(const float* pool, const float* qvec, float* scores,
                  std::int32_t* sel, std::uint32_t nb, std::uint32_t budget,
                  std::uint32_t recent, hipStream_t stream) {
  hipLaunchKernelGGL(ScoreBlocksKernel, dim3(nb), dim3(256), 0, stream, pool,
                     qvec, scores);
  hipLaunchKernelGGL(SelectTopKernel, dim3(1), dim3(kSelThreads), 0, stream,
                     scores, nb, budget, recent, sel);
}

void GatherLatents(const float* cache, const std::int32_t* sel, float* out,
                   std::uint32_t budget, std::uint32_t tail_start,
                   std::uint32_t total_rows, hipStream_t stream) {
  hipLaunchKernelGGL(GatherLatKernel, dim3(total_rows), dim3(144), 0, stream,
                     cache, sel, out, budget, tail_start);
}

void ProbeRope(const float* qrow, float* qvec, hipStream_t stream) {
  hipLaunchKernelGGL(ProbeRopeKernel, dim3(kMlaHeads), dim3(64), 0, stream,
                     qrow, qvec);
}

__global__ void SetI32Kernel(std::int32_t* dst, std::int32_t v) {
  *dst = v;
}

void SetI32(std::int32_t* dst, std::int32_t v, hipStream_t stream) {
  hipLaunchKernelGGL(SetI32Kernel, dim3(1), dim3(1), 0, stream, dst, v);
}

void SgemmF32(const float* w, const float* x, float* out, std::uint32_t m,
              std::uint32_t n, std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL(SgemmKernel, dim3((m + 31) / 32, (n + 31) / 32), dim3(256),
                     0, stream, w, x, out, static_cast<int>(m),
                     static_cast<int>(n), static_cast<int>(k));
}

void HeadGemvQ8(const void* w, const float* x, float* out, std::uint32_t heads,
                std::uint32_t rows, std::uint32_t cols, std::uint32_t x_stride,
                std::uint32_t out_stride, hipStream_t stream) {
  hipLaunchKernelGGL(HeadGemvQ8Kernel, dim3((rows + 7) / 8, heads), dim3(256),
                     0, stream, static_cast<const std::uint8_t*>(w), x, out,
                     rows, cols, x_stride, out_stride);
}

void MlaDecode(const float* qvec, const float* cache, float* part_m,
               float* part_l, float* part_acc, float* out,
               const std::int32_t* pos_ptr, float scale, hipStream_t stream) {
  hipLaunchKernelGGL(MlaDecodeSplitKernel, dim3(kMlaMaxSplits), dim3(256), 0,
                     stream, qvec, cache, part_m, part_l, part_acc, pos_ptr,
                     kMlaMaxSplits, scale);
  hipLaunchKernelGGL(MlaDecodeCombineKernel, dim3(kMlaHeads), dim3(256), 0,
                     stream, part_m, part_l, part_acc, out, kMlaMaxSplits);
}

__global__ void StoreRowKernel(const float* __restrict__ src,
                               float* __restrict__ cache,
                               const std::int32_t* __restrict__ pos_ptr,
                               std::uint32_t n, std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    cache[static_cast<std::size_t>(*pos_ptr) * dim + i] = src[i];
}

void StoreRow(const float* src, float* cache, const std::int32_t* pos_ptr,
              std::uint32_t n, std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(
      StoreRowKernel, dim3((n + kElemThreads - 1) / kElemThreads),
      dim3(kElemThreads), 0, stream, src, cache, pos_ptr, n, dim);
}

}  // namespace gufo::models::ling32::gpu
