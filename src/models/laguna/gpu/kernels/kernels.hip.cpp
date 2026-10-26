#include "src/models/laguna/gpu/kernels/kernels.hpp"

#include <algorithm>
#include <cmath>

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
      for (std::uint32_t i = 0; i < d; ++i) {
        dot += q_s[i] * __half2float(kr[i]);
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
  RmsNormKernel<<<dim3(n_rows, groups), kBlock, 0, stream>>>(x, gamma, out, dim,
                                                             groups, eps);
}

void Q8Rows(const void* w, const float* x, float* out, std::uint32_t n_tokens,
            std::uint32_t rows, std::uint32_t cols, hipStream_t stream) {
  Q8RowsKernel<<<dim3(rows, n_tokens), kBlock, 0, stream>>>(
      static_cast<const unsigned char*>(w), x, out, rows, cols);
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
  RouterF32Kernel<<<dim3(Blocks(n_experts, kBlock / 32), n_tokens), kBlock, 0,
                    stream>>>(w, x, out, n_experts, k);
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

}  // namespace gufo::models::laguna::gpu::rocm
