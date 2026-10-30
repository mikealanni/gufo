#include "src/models/laguna/gpu/kernels/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace gufo::models::laguna::gpu::rocm {
namespace {

constexpr int kBlock = 256;
constexpr int kAttnBlock = 128;
constexpr int kQ8Block = 32;
constexpr int kQ8Bytes = 34;
constexpr float kNegInf = -3.0e38F;
constexpr float kSwigluLimit = 1024.0F;
constexpr unsigned kMaxTopK = 16;

unsigned Blocks(std::size_t count, int block = kBlock) {
  return static_cast<unsigned>((count + block - 1) / block);
}

__device__ float Q8Value(const unsigned char* row, std::uint32_t i) {
  const unsigned char* b = row + (i / kQ8Block) * kQ8Bytes;
  const float d = __half2float(*reinterpret_cast<const __half*>(b));
  return d * static_cast<float>(static_cast<signed char>(b[2 + i % kQ8Block]));
}

__device__ float WaveSum(float v) {
  for (int off = warpSize / 2; off > 0; off >>= 1)
    v += __shfl_xor(v, off);
  return v;
}

__device__ float BlockSum(float v, float* scratch) {
  scratch[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      scratch[threadIdx.x] += scratch[threadIdx.x + s];
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

__global__ void EmbedQ8Kernel(const unsigned char* table,
                              const std::int32_t* tokens, float* out,
                              std::uint32_t hidden) {
  const unsigned char* row =
      table + static_cast<std::size_t>(tokens[blockIdx.x]) *
                  (hidden / kQ8Block) * kQ8Bytes;
  float* dst = out + static_cast<std::size_t>(blockIdx.x) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = Q8Value(row, i);
  }
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t dim, std::uint32_t groups,
                              float eps) {
  __shared__ float scratch[kBlock];
  const std::uint32_t width = dim / groups;
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim +
                           static_cast<std::size_t>(blockIdx.y) * width;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    sum += x[base + i] * x[base + i];
  }
  const float total = BlockSum(sum, scratch);
  const float inv = 1.0F / sqrtf(total / static_cast<float>(width) + eps);
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    out[base + i] = x[base + i] * inv * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

// Per-head RMSNorm (q/k norms, `groups` = head count). The grid-per-(row,
// group) kernel above launches one 256-thread block for a 128-wide row, so
// q_norm alone was ~147k blocks per layer and ~170 ms per 2048-token prefill.
// Here one block owns a full token row and each warp reduces one head with
// warp shuffles only (no block syncs), cutting the launch count by `groups`
// and the reduction to a single warp.
__global__ __launch_bounds__(256) void RmsNormGroupsKernel(
    const float* __restrict__ x, const float* __restrict__ gamma,
    float* __restrict__ out, std::uint32_t dim, std::uint32_t groups,
    float eps) {
  const std::uint32_t warp = threadIdx.x / warpSize;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t nwarps = blockDim.x / warpSize;
  const std::uint32_t width = dim / groups;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * dim;
  const float* xrow = x + row;
  float* orow = out + row;
  for (std::uint32_t g = warp; g < groups; g += nwarps) {
    const float* src = xrow + static_cast<std::size_t>(g) * width;
    float* dst = orow + static_cast<std::size_t>(g) * width;
    float sum = 0.0F;
    for (std::uint32_t i = lane; i < width; i += warpSize)
      sum += src[i] * src[i];
    sum = WaveSum(sum);
    const float inv =
        1.0F / sqrtf(sum / static_cast<float>(width) + eps);
    for (std::uint32_t i = lane; i < width; i += warpSize)
      dst[i] = src[i] * inv * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

__global__ void Q8RowsKernel(const unsigned char* w, const float* x, float* out,
                             std::uint32_t rows, std::uint32_t cols) {
  __shared__ float scratch[kBlock];
  const std::uint32_t r = blockIdx.x;
  const std::size_t t = blockIdx.y;
  const unsigned char* row =
      w + static_cast<std::size_t>(r) * (cols / kQ8Block) * kQ8Bytes;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    sum += Q8Value(row, i) * x[t * cols + i];
  }
  const float total = BlockSum(sum, scratch);
  if (threadIdx.x == 0)
    out[t * rows + r] = total;
}

__global__ void RopeTableKernel(float* x, std::uint32_t heads, std::uint32_t d,
                                std::uint32_t n_dims, const float* inv_freq,
                                float mscale, const std::uint32_t* start_pos) {
  const std::uint32_t half = n_dims / 2;
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * half)
    return;
  const std::uint32_t head = idx / half;
  const std::uint32_t pair = idx % half;
  const float pos = static_cast<float>(*start_pos + blockIdx.y);
  const float theta = pos * inv_freq[pair];
  float s = 0.0F;
  float c = 0.0F;
  sincosf(theta, &s, &c);
  c *= mscale;
  s *= mscale;
  float* p = x + (static_cast<std::size_t>(blockIdx.y) * heads + head) * d;
  const float x0 = p[pair];
  const float x1 = p[pair + half];
  p[pair] = x0 * c - x1 * s;
  p[pair + half] = x0 * s + x1 * c;
}

__global__ void StoreKvKernel(const float* k, const float* v, __half* k_cache,
                              __half* v_cache, std::uint32_t row_dim,
                              const std::uint32_t* start_pos,
                              std::uint32_t ring) {
  std::uint32_t row = *start_pos + blockIdx.y;
  if (ring != 0)
    row %= ring;
  const std::size_t dst = static_cast<std::size_t>(row) * row_dim;
  const std::size_t src = static_cast<std::size_t>(blockIdx.y) * row_dim;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < row_dim;
       i += gridDim.x * blockDim.x) {
    k_cache[dst + i] = __float2half(k[src + i]);
    v_cache[dst + i] = __float2half(v[src + i]);
  }
}

// One block per (query head, token); thread d owns output dimension d.
__global__ void AttentionKernel(const float* q, const __half* k_cache,
                                const __half* v_cache, const float* gate,
                                float* out, const std::uint32_t* start_pos,
                                std::uint32_t heads, std::uint32_t kv_heads,
                                std::uint32_t d, std::uint32_t window,
                                std::uint32_t ring) {
  __shared__ float q_s[kAttnBlock];
  __shared__ float p_s[kAttnBlock];
  __shared__ float red[kAttnBlock];
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t pos = *start_pos + blockIdx.y;
  const std::uint32_t kv_head = head / (heads / kv_heads);
  const std::size_t row = static_cast<std::size_t>(kv_heads) * d;
  const float scale = rsqrtf(static_cast<float>(d));
  const std::uint32_t first =
      (window > 0 && pos >= window) ? pos - window + 1 : 0;

  q_s[tid] = q[(static_cast<std::size_t>(blockIdx.y) * heads + head) * d + tid];
  float m = kNegInf;
  float l = 0.0F;
  float acc = 0.0F;
  __syncthreads();

  for (std::uint32_t tile = first; tile <= pos; tile += kAttnBlock) {
    const std::uint32_t key = tile + tid;
    float s = kNegInf;
    if (key <= pos) {
      const std::uint32_t kr_row = ring != 0 ? key % ring : key;
      const __half* kr = k_cache + kr_row * row + kv_head * d;
      float dot = 0.0F;
      for (std::uint32_t i = 0; i < d; i += 2) {
        const __half2 h = *reinterpret_cast<const __half2*>(kr + i);
        const float2 f = __half22float2(h);
        dot = fmaf(q_s[i], f.x, dot);
        dot = fmaf(q_s[i + 1], f.y, dot);
      }
      s = dot * scale;
    }
    red[tid] = s;
    __syncthreads();
    for (unsigned stride = kAttnBlock / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        red[tid] = fmaxf(red[tid], red[tid + stride]);
      __syncthreads();
    }
    const float m_new = fmaxf(m, red[0]);
    __syncthreads();
    const float p = key <= pos ? expf(s - m_new) : 0.0F;
    p_s[tid] = p;
    red[tid] = p;
    __syncthreads();
    for (unsigned stride = kAttnBlock / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        red[tid] += red[tid + stride];
      __syncthreads();
    }
    const float alpha = expf(m - m_new);
    l = l * alpha + red[0];
    acc *= alpha;
    const std::uint32_t count =
        min(static_cast<std::uint32_t>(kAttnBlock), pos + 1 - tile);
    for (std::uint32_t j = 0; j < count; ++j) {
      const std::uint32_t key_j = tile + j;
      const std::uint32_t vr_row = ring != 0 ? key_j % ring : key_j;
      acc += p_s[j] * __half2float(v_cache[vr_row * row + kv_head * d + tid]);
    }
    m = m_new;
    __syncthreads();
  }
  const float g = gate[static_cast<std::size_t>(blockIdx.y) * heads + head];
  const float softplus = g > 20.0F ? g : log1pf(expf(g));
  out[(static_cast<std::size_t>(blockIdx.y) * heads + head) * d + tid] =
      acc / l * softplus;
}

// One wave per (token, expert); 8 experts per block.
__global__ void RouterF32Kernel(const float* w, const float* x, float* out,
                                std::uint32_t n_experts, std::uint32_t k) {
  const std::uint32_t wave = threadIdx.x / warpSize;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t e = blockIdx.x * (kBlock / warpSize) + wave;
  if (e >= n_experts)
    return;
  const std::size_t t = blockIdx.y;
  float sum = 0.0F;
  for (std::uint32_t i = lane; i < k; i += warpSize) {
    sum += w[static_cast<std::size_t>(e) * k + i] * x[t * k + i];
  }
  sum = WaveSum(sum);
  if (lane == 0)
    out[t * n_experts + e] = sum;
}

// Tiled F32 router GEMV/GEMM: out[n][m] = x[n][k] * w[m][k]^T. The warp-per-
// (token, expert) kernel above re-reads the whole expert matrix for every
// token and the whole token row for every expert, leaving it instruction and
// memory bound. Here each block stages both operands through shared memory
// once per K tile and each thread accumulates a 4 token x 4 expert patch.
// 128-expert tiles keep the 3 MB expert matrix resident while the activation
// row is read only ceil(256/128) = 2 times; this is traffic-bound, not
// FLOP-bound (1.6 G-MAC total).
constexpr int kRouteTokTile = 64;
constexpr int kRouteExpTile = 128;
constexpr int kRouteKTile = 16;
constexpr int kRouteThreads = 512;

__global__ __launch_bounds__(kRouteThreads) void RouterGemmKernel(
    const float* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, std::uint32_t n_experts, std::uint32_t k,
    std::uint32_t n_tokens) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t tx = tid & 31U;
  const std::uint32_t ty = tid >> 5U;
  const std::uint32_t tok_block = blockIdx.y * kRouteTokTile;
  const std::uint32_t exp_block = blockIdx.x * kRouteExpTile;
  __shared__ float xs[kRouteKTile][kRouteTokTile + 4];
  __shared__ float ws[kRouteKTile][kRouteExpTile + 4];
  float acc[4][4] = {};
  const std::uint32_t tok0 = tok_block + ty * 4U;
  const std::uint32_t exp0 = exp_block + tx * 4U;
  for (std::uint32_t k0 = 0; k0 < k; k0 += kRouteKTile) {
    for (std::uint32_t i = tid; i < kRouteKTile * kRouteTokTile;
         i += kRouteThreads) {
      const std::uint32_t kk = i / kRouteTokTile;
      const std::uint32_t t = i % kRouteTokTile;
      const std::uint32_t gk = k0 + kk;
      const std::uint32_t gt = tok_block + t;
      xs[kk][t] =
          (gk < k && gt < n_tokens) ? x[std::size_t{gt} * k + gk] : 0.0F;
    }
    for (std::uint32_t i = tid; i < kRouteKTile * kRouteExpTile;
         i += kRouteThreads) {
      const std::uint32_t kk = i / kRouteExpTile;
      const std::uint32_t e = i % kRouteExpTile;
      const std::uint32_t gk = k0 + kk;
      const std::uint32_t ge = exp_block + e;
      ws[kk][e] =
          (gk < k && ge < n_experts) ? w[std::size_t{ge} * k + gk] : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kRouteKTile; ++kk) {
      const float* xr = &xs[kk][ty * 4];
      const float* wr = &ws[kk][tx * 4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const float xv = xr[t];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[t][j] += xv * wr[j];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (std::uint32_t t = 0; t < 4; ++t) {
    const std::uint32_t gt = tok0 + t;
    if (gt >= n_tokens)
      continue;
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const std::uint32_t ge = exp0 + j;
      if (ge < n_experts)
        out[std::size_t{gt} * n_experts + ge] = acc[t][j];
    }
  }
}

// Tiled Q8_0 gate projection: out[n][r] = sum_k dequant(w[r][k]) * x[n][k].
// Same tiling as RouterGemmKernel; the gate weight rows (72 or 48 per token
// in the old grid-per-(row, token) kernel) were re-read for every token.
__global__ __launch_bounds__(kRouteThreads) void GateGemmKernel(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, std::uint32_t rows, std::uint32_t cols,
    std::uint32_t n_tokens) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t tx = tid & 31U;
  const std::uint32_t ty = tid >> 5U;
  const std::uint32_t tok_block = blockIdx.y * kRouteTokTile;
  const std::uint32_t row_block = blockIdx.x * kRouteExpTile;
  __shared__ float xs[kRouteKTile][kRouteTokTile + 4];
  __shared__ float ws[kRouteKTile][kRouteExpTile + 4];
  float acc[4][4] = {};
  const std::uint32_t tok0 = tok_block + ty * 4U;
  const std::uint32_t row0 = row_block + tx * 4U;
  const std::uint32_t blocks_per_row = cols / kQ8Block;
  for (std::uint32_t k0 = 0; k0 < cols; k0 += kRouteKTile) {
    for (std::uint32_t i = tid; i < kRouteKTile * kRouteTokTile;
         i += kRouteThreads) {
      const std::uint32_t kk = i / kRouteTokTile;
      const std::uint32_t t = i % kRouteTokTile;
      const std::uint32_t gk = k0 + kk;
      const std::uint32_t gt = tok_block + t;
      xs[kk][t] =
          (gk < cols && gt < n_tokens) ? x[std::size_t{gt} * cols + gk] : 0.0F;
    }
    for (std::uint32_t i = tid; i < kRouteKTile * kRouteExpTile;
         i += kRouteThreads) {
      const std::uint32_t kk = i / kRouteExpTile;
      const std::uint32_t e = i % kRouteExpTile;
      const std::uint32_t gk = k0 + kk;
      const std::uint32_t gr = row_block + e;
      float value = 0.0F;
      if (gk < cols && gr < rows) {
        const unsigned char* row =
            w + std::size_t{gr} * blocks_per_row * kQ8Bytes;
        const unsigned char* b = row + (gk / kQ8Block) * kQ8Bytes;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        value = d * static_cast<float>(
                        static_cast<signed char>(b[2 + gk % kQ8Block]));
      }
      ws[kk][e] = value;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kRouteKTile; ++kk) {
      const float* xr = &xs[kk][ty * 4];
      const float* wr = &ws[kk][tx * 4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const float xv = xr[t];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[t][j] += xv * wr[j];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (std::uint32_t t = 0; t < 4; ++t) {
    const std::uint32_t gt = tok0 + t;
    if (gt >= n_tokens)
      continue;
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const std::uint32_t gr = row0 + j;
      if (gr < rows)
        out[std::size_t{gt} * rows + gr] = acc[t][j];
    }
  }
}

// One block of n_experts threads (<= 256) per token; k rounds of arg-max.
__global__ void SigmoidBiasTopKKernel(const float* logits, const float* bias,
                                      std::int32_t* ids, float* weights,
                                      std::uint32_t n_experts, std::uint32_t k,
                                      bool normalize, float scale) {
  __shared__ float best_val[kBlock / 32];
  __shared__ int best_idx[kBlock / 32];
  __shared__ float chosen_p[kMaxTopK];
  __shared__ int chosen_i[kMaxTopK];
  const std::size_t t = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid / warpSize;
  const std::uint32_t lane = tid % warpSize;
  float p = 0.0F;
  float sel = kNegInf;
  if (tid < n_experts) {
    p = 1.0F / (1.0F + expf(-logits[t * n_experts + tid]));
    sel = p + bias[tid];
  }
  for (std::uint32_t round = 0; round < k; ++round) {
    float v = sel;
    int idx = static_cast<int>(tid);
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(v, off);
      const int oi = __shfl_xor(idx, off);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0) {
      best_val[wave] = v;
      best_idx[wave] = idx;
    }
    __syncthreads();
    if (tid == 0) {
      float bv = best_val[0];
      int bi = best_idx[0];
      for (unsigned w = 1; w < blockDim.x / warpSize; ++w) {
        if (best_val[w] > bv || (best_val[w] == bv && best_idx[w] < bi)) {
          bv = best_val[w];
          bi = best_idx[w];
        }
      }
      chosen_i[round] = bi;
    }
    __syncthreads();
    if (static_cast<int>(tid) == chosen_i[round]) {
      chosen_p[round] = p;
      sel = kNegInf;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float sum = 0.0F;
    for (std::uint32_t i = 0; i < k; ++i)
      sum += chosen_p[i];
    const float denom = normalize ? fmaxf(sum, 6.103515625e-5F) : 1.0F;
    for (std::uint32_t i = 0; i < k; ++i) {
      ids[t * k + i] = chosen_i[i];
      weights[t * k + i] = chosen_p[i] / denom * scale;
    }
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float g = gate[i];
  gate[i] = g / (1.0F + expf(-g)) * up[i];
}

// One block of cols / 4 threads per row, one float4 each.
__global__ void SwigluScaledKernel(float* gate, const float* up, float* scale,
                                   std::uint32_t cols) {
  __shared__ float red[kBlock / 32];
  const std::size_t base =
      static_cast<std::size_t>(blockIdx.x) * cols + threadIdx.x * 4;
  const float4 g = *reinterpret_cast<const float4*>(gate + base);
  const float4 u = *reinterpret_cast<const float4*>(up + base);
  float v[4] = {
      g.x / (1.0F + expf(-g.x)) * u.x, g.y / (1.0F + expf(-g.y)) * u.y,
      g.z / (1.0F + expf(-g.z)) * u.z, g.w / (1.0F + expf(-g.w)) * u.w};
  float amax =
      fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off));
  }
  if (threadIdx.x % warpSize == 0)
    red[threadIdx.x / warpSize] = amax;
  __syncthreads();
  float row_max = 0.0F;
  for (unsigned w = 0; w < blockDim.x / warpSize; ++w) {
    row_max = fmaxf(row_max, red[w]);
  }
  const float s = row_max > kSwigluLimit
                      ? exp2f(ceilf(log2f(row_max / kSwigluLimit)))
                      : 1.0F;
  *reinterpret_cast<float4*>(gate + base) =
      make_float4(v[0] / s, v[1] / s, v[2] / s, v[3] / s);
  if (threadIdx.x == 0)
    scale[blockIdx.x] = s;
}

__global__ void MoeCombineKernel(float* x, const float* expert_out,
                                 const float* weights,
                                 const float* expert_scale, const float* shared,
                                 const float* shared_scale, std::uint32_t k,
                                 std::uint32_t dim) {
  const std::size_t t = blockIdx.y;
  const float sh_scale = shared_scale != nullptr ? shared_scale[t] : 1.0F;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim;
       i += gridDim.x * blockDim.x) {
    float acc = sh_scale * shared[t * dim + i];
    for (std::uint32_t s = 0; s < k; ++s) {
      const float es = expert_scale != nullptr ? expert_scale[t * k + s] : 1.0F;
      acc += weights[t * k + s] * es * expert_out[(t * k + s) * dim + i];
    }
    x[t * dim + i] += acc;
  }
}

__global__ void NarrowRowsKernel(const float* in, __half* out, std::uint32_t k,
                                 std::uint32_t ldx) {
  const std::size_t row = blockIdx.y;
  for (std::uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < k;
       i += gridDim.x * blockDim.x * 4) {
    const float4 v = *reinterpret_cast<const float4*>(in + row * k + i);
    __half2* dst = reinterpret_cast<__half2*>(out + row * ldx + i);
    dst[0] = __floats2half2_rn(v.x, v.y);
    dst[1] = __floats2half2_rn(v.z, v.w);
  }
}

__global__ void AddKernel(float* x, const float* y, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += y[i];
}

}  // namespace

void EmbedQ8(const void* table, const std::int32_t* tokens, float* out,
             std::uint32_t n_tokens, std::uint32_t hidden, hipStream_t stream) {
  EmbedQ8Kernel<<<n_tokens, kBlock, 0, stream>>>(
      static_cast<const unsigned char*>(table), tokens, out, hidden);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  if (groups > 1 && dim % groups == 0 && (dim / groups) <= 256) {
    RmsNormGroupsKernel<<<n_rows, 256, 0, stream>>>(x, gamma, out, dim, groups,
                                                    eps);
    return;
  }
  RmsNormKernel<<<dim3(n_rows, groups), kBlock, 0, stream>>>(x, gamma, out, dim,
                                                             groups, eps);
}

void Q8Rows(const void* w, const float* x, float* out, std::uint32_t n_tokens,
            std::uint32_t rows, std::uint32_t cols, hipStream_t stream) {
  if (n_tokens < 8) {
    Q8RowsKernel<<<dim3(rows, n_tokens), kBlock, 0, stream>>>(
        static_cast<const unsigned char*>(w), x, out, rows, cols);
    return;
  }
  GateGemmKernel<<<dim3(Blocks(rows, kRouteExpTile),
                        Blocks(n_tokens, kRouteTokTile)),
                   kRouteThreads, 0, stream>>>(
      static_cast<const unsigned char*>(w), x, out, rows, cols, n_tokens);
}

void RopeTable(float* x, std::uint32_t n_tokens, std::uint32_t heads,
               std::uint32_t head_dim, std::uint32_t n_dims,
               const float* inv_freq, float mscale,
               const std::uint32_t* start_pos, hipStream_t stream) {
  RopeTableKernel<<<dim3(Blocks(std::size_t{heads} * n_dims / 2), n_tokens),
                    kBlock, 0, stream>>>(x, heads, head_dim, n_dims, inv_freq,
                                         mscale, start_pos);
}

void StoreKv(const float* k, const float* v, __half* k_cache, __half* v_cache,
             std::uint32_t n_tokens, std::uint32_t row_dim,
             const std::uint32_t* start_pos, std::uint32_t ring,
             hipStream_t stream) {
  StoreKvKernel<<<dim3(Blocks(row_dim), n_tokens), kBlock, 0, stream>>>(
      k, v, k_cache, v_cache, row_dim, start_pos, ring);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* gate, float* out, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t head_dim,
               std::uint32_t window, std::uint32_t ring, hipStream_t stream) {
  AttentionKernel<<<dim3(heads, n_tokens), kAttnBlock, 0, stream>>>(
      q, k_cache, v_cache, gate, out, start_pos, heads, kv_heads, head_dim,
      window, ring);
}

void RouterF32(const float* w, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t n_experts, std::uint32_t k,
               hipStream_t stream) {
  // Tiny batches (single-token decode) are cheaper on the wave-per-expert
  // kernel: the tiled kernel would stage a 64x16 shared tile for one row.
  if (n_tokens < 8) {
    RouterF32Kernel<<<dim3(Blocks(n_experts, kBlock / 32), n_tokens), kBlock, 0,
                      stream>>>(w, x, out, n_experts, k);
    return;
  }
  RouterGemmKernel<<<dim3(Blocks(n_experts, kRouteExpTile),
                          Blocks(n_tokens, kRouteTokTile)),
                     kRouteThreads, 0, stream>>>(w, x, out, n_experts, k, n_tokens);
}

void SigmoidBiasTopK(const float* logits, const float* bias, std::int32_t* ids,
                     float* weights, std::uint32_t n_tokens,
                     std::uint32_t n_experts, std::uint32_t k, bool normalize,
                     float scale, hipStream_t stream) {
  SigmoidBiasTopKKernel<<<n_tokens, kBlock, 0, stream>>>(
      logits, bias, ids, weights, n_experts, k, normalize, scale);
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  SwigluKernel<<<Blocks(count), kBlock, 0, stream>>>(gate, up, count);
}

void SwigluScaled(float* gate, const float* up, float* scale,
                  std::uint32_t rows, std::uint32_t cols, hipStream_t stream) {
  SwigluScaledKernel<<<rows, cols / 4, 0, stream>>>(gate, up, scale, cols);
}

void MoeCombine(float* x, const float* expert_out, const float* weights,
                const float* expert_scale, const float* shared,
                const float* shared_scale, std::uint32_t n_tokens,
                std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  MoeCombineKernel<<<dim3(Blocks(dim), n_tokens), kBlock, 0, stream>>>(
      x, expert_out, weights, expert_scale, shared, shared_scale, k, dim);
}

void NarrowRows(const float* in, __half* out, std::uint32_t n_rows,
                std::uint32_t k, std::uint32_t ldx, hipStream_t stream) {
  NarrowRowsKernel<<<dim3(Blocks(k / 4), n_rows), kBlock, 0, stream>>>(in, out,
                                                                       k, ldx);
}

void AddInPlace(float* x, const float* y, std::size_t count,
                hipStream_t stream) {
  AddKernel<<<Blocks(count), kBlock, 0, stream>>>(x, y, count);
}

__global__ void ToHalfKernel(const float* src, __half* dst, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    dst[i] = __float2half(src[i]);
}

void ToHalf(const float* src, __half* dst, std::size_t count,
            hipStream_t stream) {
  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  ToHalfKernel<<<blocks, threads, 0, stream>>>(src, dst, count);
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  const uint4 lo = *reinterpret_cast<const uint4*>(p);
  const uint4 hi = *(reinterpret_cast<const uint4*>(p) + 1);
  v16h r;
  __builtin_memcpy(&r, &lo, 16);
  __builtin_memcpy(reinterpret_cast<char*>(&r) + 16, &hi, 16);
  return r;
}

__device__ __forceinline__ v16h LoadFragGlobal(const __half* p) {
  const uint4 lo = *reinterpret_cast<const uint4*>(p);
  const uint4 hi = *(reinterpret_cast<const uint4*>(p) + 1);
  v16h r;
  __builtin_memcpy(&r, &lo, 16);
  __builtin_memcpy(reinterpret_cast<char*>(&r) + 16, &hi, 16);
  return r;
}

constexpr int kWmmaKeys = 32;
constexpr int kNb = kWmmaKeys / 16;
constexpr int kWmmaHeadDim = 128;
constexpr int kWmmaGroup = 8;
constexpr int kWmmaColumns = 16;
constexpr int kWmmaWaves = 8;
constexpr int kWmmaThreads = kWmmaWaves * 32;
constexpr int kWmmaKStride = 136;
constexpr int kWmmaVtStride = kWmmaKeys + 8;
constexpr std::uint32_t kSparseBlock = 64;
constexpr std::uint32_t kSparseSink = 2;
constexpr std::uint32_t kSparseLocal = 32;
constexpr std::uint32_t kSparseList = 512;

// kHeadsPerKv is heads/kv_heads for this layer group (6 for Laguna's
// group-6 layers, 9 for group-9). The 16 WMMA output columns hold
// kTokensPerWave tokens x kHeadsPerKv heads, so a group with a higher
// ratio simply fits fewer tokens per wave (9 has to drop to one).
template <int kHeadsPerKv>
__global__ __launch_bounds__(kWmmaThreads) void AttentionWmmaKernelLaguna(
    const __half* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    float* __restrict__ out, std::uint32_t n_tokens,
    const std::uint32_t* __restrict__ start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t window,
    const __half* __restrict__ kmm, std::uint32_t nblk,
    std::uint32_t top_blocks) {
  // K tile and V^T tile; the block scores of the sparse prologue reuse them.
  constexpr int kKBytes = kWmmaKeys * kWmmaKStride * 2;
  constexpr int kVBytes = kWmmaHeadDim * kWmmaVtStride * 2;
  __shared__ __align__(16) char smem[kKBytes + kVBytes];
  auto* k_s = reinterpret_cast<__half*>(smem);
  auto* v_t = reinterpret_cast<__half*>(smem + kKBytes);
  __shared__ std::uint32_t blist[kSparseList];
  __shared__ std::uint32_t scan_tmp[kWmmaWaves + 2];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  // kWmmaColumns (16) output columns per wave are filled with
  // kTokensPerWave tokens x kHeadsPerKv heads. Group-6 fits 2 tokens per
  // wave; group-9 fits only 1 (9 * 2 = 18 > 16), so a block covers 8
  // tokens instead of 16.
  constexpr std::uint32_t kTokensPerWave = kWmmaColumns / kHeadsPerKv;
  constexpr std::uint32_t kBlockTokens = kWmmaWaves * kTokensPerWave;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t t0 = blockIdx.y * kBlockTokens;
  const std::uint32_t start = *start_pos;
  const std::size_t kv_row = static_cast<std::size_t>(kv_heads) * kWmmaHeadDim;
  constexpr float kLog2e = 1.4426950408889634F;
  // Scores and the running maximum are kept in log2 units.
  const float scale = 0.08838834764831845F * kLog2e;  // log2(e) / sqrt(128)

  // Column `sub` of this wave is (token offset, head offset) within the kv
  // group. `token` must stride by kTokensPerWave per wave, not 1, so every
  // token of the block is covered exactly once without gaps or overlap.
  // The per-kv_head Q/output stride is kHeadsPerKv, NOT the WMMA tile
  // width (kWmmaGroup=8); using the tile width silently read/wrote the
  // wrong global head slot for every kv_head>=1.
  const std::uint32_t token = t0 + wave * kTokensPerWave + sub / kHeadsPerKv;
  const std::uint32_t head = kv_head * kHeadsPerKv + sub % kHeadsPerKv;
  const bool live = token < n_tokens;
  const bool head_live = sub < kHeadsPerKv * kTokensPerWave;
  const std::uint32_t qpos = start + token;

  v16h q_frag[8];
  {
    const __half* qr =
        q + (static_cast<std::size_t>(live && head_live ? token : 0) * heads +
             (head_live ? head : 0)) *
                kWmmaHeadDim;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint4 lo = make_uint4(0, 0, 0, 0);
      uint4 hi = lo;
      if (live && head_live) {
        lo = *reinterpret_cast<const uint4*>(qr + ks * 16);
        hi = *reinterpret_cast<const uint4*>(qr + ks * 16 + 8);
      }
      __builtin_memcpy(&q_frag[ks], &lo, 16);
      __builtin_memcpy(reinterpret_cast<char*>(&q_frag[ks]) + 16, &hi, 16);
    }
  }
  // v_perm selectors: low half-word from this lane's keys and high from the
  // partner's (lower lane half), or the reverse (upper lane half), for the
  // low (X = 0) and high (X = 1) half-word of a packed pair.
  std::uint32_t perm_sel[2];
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const std::uint32_t own_pair = (2 * x) | ((2 * x + 1) << 8);
    const std::uint32_t partner_pair = (4 + 2 * x) | ((5 + 2 * x) << 8);
    perm_sel[x] = half_id == 0 ? (own_pair | (partner_pair << 16))
                               : (partner_pair | (own_pair << 16));
  }

  float run_max = -3.0e38F;
  // The sink counts once; the two lane halves each sum their own keys.
  float run_sum = 0.0F;
  v8f o_acc[8];
#pragma unroll
  for (int t = 0; t < 8; ++t)
    o_acc[t] = v8f{};

  const std::uint32_t last_token = min(t0 + kBlockTokens, n_tokens) - 1;
  const std::uint32_t pos_max = start + last_token;
  const std::uint32_t pos_min = start + t0;
  const std::uint32_t first_key =
      (window > 0 && pos_min >= window) ? pos_min - window + 1 : 0;

  // Key tiles to visit: a contiguous run, or (sparse) the sink blocks, the
  // best-scoring older blocks and the local window, each 64 keys.
  std::uint32_t n_tiles = (pos_max - first_key) / kWmmaKeys + 1;
  std::uint32_t list_count = 0;
  if (false) {
    const std::uint32_t local_last = pos_max / kSparseBlock;
    const std::uint32_t pmin_block = pos_min / kSparseBlock;
    const std::uint32_t local_first =
        pmin_block >= kSparseLocal ? pmin_block - kSparseLocal : 0;
    const std::uint32_t cand_first = kSparseSink;
    const std::uint32_t cand_end = local_first;
    const std::uint32_t cand_count =
        cand_end > cand_first ? cand_end - cand_first : 0;
    auto* scores = reinterpret_cast<std::uint32_t*>(smem);
    if (cand_count <= top_blocks) {
      for (std::uint32_t b = tid; b <= local_last; b += kWmmaThreads)
        blist[b] = b;
      list_count = local_last + 1;
    } else {
      for (std::uint32_t b = tid; b < cand_count; b += kWmmaThreads)
        scores[b] = 0;
      __syncthreads();
      // Quest bound per block and query: q+ . kmax + q- . kmin, as two WMMA
      // chains with the pooled rows as the A operand.
      const v16h zero16 = v16h{};
      for (std::uint32_t t = wave; t * 16 < cand_count; t += kWmmaWaves) {
        std::uint32_t row = cand_first + t * 16 + sub;
        if (row >= cand_end)
          row = cand_end - 1;
        const __half* base = kmm + (std::size_t{kv_head} * nblk + row) * 256;
        v8f acc = v8f{};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const v16h a_max = LoadFragGlobal(base + ks * 16);
          const v16h a_min = LoadFragGlobal(base + 128 + ks * 16);
          acc = Wmma(a_max, __builtin_elementwise_max(q_frag[ks], zero16), acc);
          acc = Wmma(a_min, __builtin_elementwise_min(q_frag[ks], zero16), acc);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const std::uint32_t blk = cand_first + t * 16 + 2 * i + half_id;
          if (live && blk < cand_end) {
            const std::uint32_t bits = __float_as_uint(acc[i]);
            const std::uint32_t key =
                bits ^ ((bits >> 31) != 0 ? 0xFFFFFFFFU : 0x80000000U);
            atomicMax(scores + (blk - cand_first), key);
          }
        }
      }
      __syncthreads();
      // Largest threshold with at least top_blocks scores at or above it.
      std::uint32_t thr = 0;
      for (int bit = 31; bit >= 0; --bit) {
        const std::uint32_t trial = thr | (1U << bit);
        std::uint32_t cnt = 0;
        for (std::uint32_t b = tid; b < cand_count; b += kWmmaThreads)
          cnt += scores[b] >= trial ? 1U : 0U;
        for (int off = 16; off > 0; off >>= 1)
          cnt += __shfl_xor(cnt, off);
        if (lane == 0)
          scan_tmp[wave] = cnt;
        __syncthreads();
        std::uint32_t total = 0;
        for (int w = 0; w < kWmmaWaves; ++w)
          total += scan_tmp[w];
        __syncthreads();
        if (total >= top_blocks)
          thr = trial;
      }
      // Ordered compaction: scores above the threshold, then ties by index.
      const std::uint32_t per = (cand_count + kWmmaThreads - 1) / kWmmaThreads;
      const std::uint32_t b0 = tid * per;
      const std::uint32_t b1 = min(b0 + per, cand_count);
      std::uint32_t gt = 0;
      std::uint32_t eq = 0;
      for (std::uint32_t b = b0; b < b1; ++b) {
        gt += scores[b] > thr ? 1U : 0U;
        eq += scores[b] == thr ? 1U : 0U;
      }
      auto block_scan = [&](std::uint32_t value, std::uint32_t* total) {
        std::uint32_t inc = value;
        for (int off = 1; off < 32; off <<= 1) {
          const std::uint32_t up = __shfl_up(inc, off);
          if (lane >= static_cast<std::uint32_t>(off))
            inc += up;
        }
        if (lane == 31)
          scan_tmp[wave] = inc;
        __syncthreads();
        std::uint32_t wave_off = 0;
        std::uint32_t sum = 0;
        for (int w = 0; w < kWmmaWaves; ++w) {
          if (w < static_cast<int>(wave))
            wave_off += scan_tmp[w];
          sum += scan_tmp[w];
        }
        __syncthreads();
        *total = sum;
        return wave_off + inc - value;
      };
      std::uint32_t total_gt = 0;
      std::uint32_t total_eq = 0;
      const std::uint32_t gt_off = block_scan(gt, &total_gt);
      const std::uint32_t eq_off = block_scan(eq, &total_eq);
      const std::uint32_t eq_take =
          top_blocks > total_gt ? top_blocks - total_gt : 0;
      const std::uint32_t my_eq =
          eq_off < eq_take ? min(eq, eq_take - eq_off) : 0;
      std::uint32_t sel_total = 0;
      const std::uint32_t sel_off = block_scan(gt + my_eq, &sel_total);
      // Sinks, selected blocks in ascending order, then the local window.
      for (std::uint32_t b = tid; b < kSparseSink; b += kWmmaThreads)
        blist[b] = b;
      std::uint32_t out_at = kSparseSink + sel_off;
      std::uint32_t eq_seen = 0;
      for (std::uint32_t b = b0; b < b1; ++b) {
        const std::uint32_t sc = scores[b];
        bool take = sc > thr;
        if (!take && sc == thr && eq_seen < my_eq) {
          take = true;
          ++eq_seen;
        }
        if (take)
          blist[out_at++] = cand_first + b;
      }
      const std::uint32_t local_base = kSparseSink + sel_total;
      for (std::uint32_t b = local_first + tid; b <= local_last;
           b += kWmmaThreads)
        blist[local_base + (b - local_first)] = b;
      list_count = local_base + (local_last - local_first + 1);
    }
    __syncthreads();
    n_tiles = list_count * (kSparseBlock / kWmmaKeys);
  }
  const auto tile_key = [&](std::uint32_t j) {
    if (false) {
      return blist[j / (kSparseBlock / kWmmaKeys)] * kSparseBlock +
             (j % (kSparseBlock / kWmmaKeys)) * kWmmaKeys;
    } else {
      return first_key + j * kWmmaKeys;
    }
  };

  constexpr int kLoads = kWmmaKeys * 16 / kWmmaThreads;
  uint4 kpre[kLoads];
  uint4 vpre[kLoads];
  // Lanes run along keys: the V^T stores of a wave then hit consecutive words.
  const auto fetch = [&](std::uint32_t kt) {
#pragma unroll
    for (int u = 0; u < kLoads; ++u) {
      const std::uint32_t idx = tid + u * kWmmaThreads;
      const std::uint32_t key = idx % kWmmaKeys;
      const std::uint32_t c = idx / kWmmaKeys;
      kpre[u] = make_uint4(0, 0, 0, 0);
      vpre[u] = make_uint4(0, 0, 0, 0);
      if (kt + key <= pos_max) {
        const std::size_t src =
            (kt + key) * kv_row + kv_head * kWmmaHeadDim + c * 8;
        kpre[u] = *reinterpret_cast<const uint4*>(k_cache + src);
        vpre[u] = *reinterpret_cast<const uint4*>(v_cache + src);
      }
    }
  };
  const auto stage = [&]() {
    const bool odd = (lane & 1U) != 0;
#pragma unroll
    for (int u = 0; u < kLoads; ++u) {
      const std::uint32_t idx = tid + u * kWmmaThreads;
      const std::uint32_t key = idx % kWmmaKeys;
      const std::uint32_t c = idx / kWmmaKeys;
      *reinterpret_cast<uint4*>(k_s + key * kWmmaKStride + c * 8) = kpre[u];
      const std::uint32_t words[4] = {vpre[u].x, vpre[u].y, vpre[u].z,
                                      vpre[u].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // Dims 2j and 2j + 1 of this key; the lane pair swaps halves so each
        // lane stores one 32-bit word holding two neighbouring keys.
        const std::uint32_t mine = words[j];
        const std::uint32_t other = __shfl_xor(mine, 1);
        const std::uint32_t lo_dim = odd ? (mine >> 16) : (mine & 0xffffU);
        const std::uint32_t lo_other = odd ? (other >> 16) : (other & 0xffffU);
        const std::uint32_t dim = c * 8 + 2 * j + (odd ? 1U : 0U);
        const std::uint32_t pair =
            odd ? (lo_other | (lo_dim << 16)) : (lo_dim | (lo_other << 16));
        *reinterpret_cast<std::uint32_t*>(v_t + dim * kWmmaVtStride +
                                          (key & ~1U)) = pair;
      }
    }
  };
  fetch(tile_key(0));
  for (std::uint32_t j = 0; j < n_tiles; ++j) {
    const std::uint32_t kt = tile_key(j);
    stage();
    __syncthreads();
    if (j + 1 < n_tiles)
      fetch(tile_key(j + 1));

    // S^T blocks: rows are keys, columns are this wave's queries.
    v8f s_acc[kNb];
#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
      s_acc[nb] = v8f{};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const v16h k_frag =
            LoadFrag(k_s + (nb * 16 + sub) * kWmmaKStride + ks * 16);
        s_acc[nb] = Wmma(k_frag, q_frag[ks], s_acc[nb]);
      }
    }

    float val[kNb][8];
    float tile_max = -3.0e38F;
#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint32_t key = kt + nb * 16 + 2 * i + half_id;
        const bool in_window =
            window == 0 || qpos < window || key + window > qpos;
        const bool valid = live && key <= qpos && in_window;
        val[nb][i] = valid ? s_acc[nb][i] * scale : -3.0e38F;
        tile_max = fmaxf(tile_max, val[nb][i]);
      }
    }
    tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16));
    const float m_new = fmaxf(run_max, tile_max);
    const float alpha = __builtin_amdgcn_exp2f(run_max - m_new);
    float psum = 0.0F;
    v16h p_frag[kNb];
#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
      std::uint32_t own[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float p0 = __builtin_amdgcn_exp2f(val[nb][2 * m] - m_new);
        const float p1 = __builtin_amdgcn_exp2f(val[nb][2 * m + 1] - m_new);
        psum += p0 + p1;
        const __half2 h2 = __floats2half2_rn(p0, p1);
        own[m] = *reinterpret_cast<const std::uint32_t*>(&h2);
      }
      std::uint32_t partner[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        partner[m] = static_cast<std::uint32_t>(
            __shfl_xor(static_cast<int>(own[m]), 16));
      }
      // Keys 2i come from the lower lane half, 2i + 1 from the upper one.
      std::uint32_t dw[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        dw[i] =
            __builtin_amdgcn_perm(partner[i / 2], own[i / 2], perm_sel[i & 1]);
      }
      __builtin_memcpy(&p_frag[nb], dw, 32);
    }
    run_sum = run_sum * alpha + psum;
    run_max = m_new;
    // The running maximum rarely moves once a few tiles are in: skip the
    // rescale when no lane of the wave changed it.
    if (__builtin_amdgcn_ballot_w32(alpha != 1.0F) != 0) {
#pragma unroll
      for (int t = 0; t < 8; ++t)
        o_acc[t] *= alpha;
    }

#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const v16h v_frag =
            LoadFrag(v_t + (t * 16 + sub) * kWmmaVtStride + nb * 16);
        o_acc[t] = Wmma(v_frag, p_frag[nb], o_acc[t]);
      }
    }
    __syncthreads();
  }

  if (!live)
    return;
  const float total = run_sum + __shfl_xor(run_sum, 16);
  // Scalar's AttentionKernel multiplies by softplus(gate[token][head]) at
  // the very end (log1pf(expf(g)), guarded for large g) -- `sinks` here is
  // that same per-token-per-head gate buffer (passed as `gate.get()` by the
  // caller), not an actual attention sink; this kernel used to ignore it
  // entirely, which was the real cause of every WMMA-vs-scalar mismatch.
  const float g =
      head_live ? sinks[static_cast<std::size_t>(token) * heads + head] : 0.0F;
  const float softplus = g > 20.0F ? g : log1pf(expf(g));
  const float inv = softplus / total;
  float* dst =
      out + (static_cast<std::size_t>(token) * heads + head) * kWmmaHeadDim;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (live && head_live)
        dst[t * 16 + 2 * i + half_id] = o_acc[t][i] * inv;
  }
}

bool AttentionWmma(const __half* q, const __half* k_cache,
                   const __half* v_cache, const float* gate, float* out,
                   std::uint32_t n_tokens, const std::uint32_t* start_pos,
                   std::uint32_t heads, std::uint32_t kv_heads,
                   std::uint32_t head_dim, std::uint32_t window,
                   std::uint32_t ring, hipStream_t stream) {
  if (head_dim != kWmmaHeadDim)
    return false;
  // The kernel's `head_live` mask already handles any ratio up to
  // kWmmaGroup (e.g. Laguna's 48/8=6 heads-per-kv); the old `!=` here
  // rejected every real call (6 != 8) and silently left `out` untouched,
  // so callers that ignore the return value got stale data from whatever
  // the previous layer wrote into the same scratch buffer. That was the
  // actual cause of the "WMMA multi-token mismatch" seen across several
  // sessions, not a stride/layout bug in the kernel itself.
  if (kv_heads == 0 || heads == 0 || heads % kv_heads != 0)
    return false;
  const std::uint32_t heads_per_kv = heads / kv_heads;
  if (heads_per_kv < 1 || heads_per_kv > kWmmaColumns)
    return false;
  const auto launch = [&](auto tag) {
    constexpr int kHeadsPerKv = decltype(tag)::value;
    constexpr int kTokensPerWave = kWmmaColumns / kHeadsPerKv;
    constexpr int kBlockTokens = kWmmaWaves * kTokensPerWave;
    AttentionWmmaKernelLaguna<kHeadsPerKv>
        <<<dim3(kv_heads, (n_tokens + kBlockTokens - 1) / kBlockTokens),
           kWmmaThreads, 0, stream>>>(q, k_cache, v_cache, gate, out, n_tokens,
                                      start_pos, heads, kv_heads, window,
                                      nullptr, 0, 0);
  };
  using std::integral_constant;
  switch (heads_per_kv) {
    case 6:
      launch(integral_constant<int, 6>{});
      break;
    case 9:
      launch(integral_constant<int, 9>{});
      break;
    default:
      return false;
  }
  return true;

__global__ void CaptureTapKernel(float* dst, const float* x, std::uint32_t rows,
                                 std::uint32_t hidden, std::size_t stride,
                                 std::size_t tap_offset) {
  const std::uint32_t r = blockIdx.y;
  const float* src = x + static_cast<std::size_t>(r) * hidden;
  float* out = dst + r * stride + tap_offset;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    out[i] = src[i];
}

void CaptureTap(float* dst, const float* x, std::uint32_t n_rows,
                std::uint32_t hidden, std::size_t stride,
                std::size_t tap_offset, hipStream_t stream) {
  if (n_rows == 0)
    return;
  CaptureTapKernel<<<dim3(1, n_rows), hidden, 0, stream>>>(
      dst, x, n_rows, hidden, stride, tap_offset);
}

}  // namespace gufo::models::laguna::gpu::rocm
