#include "src/models/qwen38_flash_next/d1_npu.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/hip/ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

using core::GgmlType;

constexpr std::uint32_t M = 512;
constexpr std::uint32_t kThreads = 128;

std::size_t RowBytes(GgmlType type, std::size_t columns) {
  switch (type) {
    case GgmlType::kQ4_0:
      return columns / 32 * 18;
    case GgmlType::kQ6_K:
      return columns / 256 * 210;
    case GgmlType::kQ8_0:
      return columns / 32 * 34;
    default:
      return 0;
  }
}

// Same Q6_K / Q4_0 unit decoders as the jev judge (jev_npu.hip.cpp).
// Kind: 0 = Q4_0, 1 = Q6_K, 2 = Q8_0.
template<int Kind>
__device__ void D1DequantUnit(const std::uint8_t* b, float* o) {
  constexpr bool Q6 = Kind == 1;
  if constexpr (Kind == 2) {
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
#pragma unroll
    for (int j = 0; j < 32; ++j)
      o[j] = d * static_cast<float>(static_cast<std::int8_t>(b[2 + j]));
  } else if constexpr (Q6) {
    const float fd = __half2float(*reinterpret_cast<const __half*>(b + 208));
    const std::uint8_t* ql = b;
    const std::uint8_t* qh = b + 128;
    const std::uint8_t* sc = b + 192;
    int vals[256];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int qbase = half * 64;
#pragma unroll
      for (int lane = 0; lane < 32; ++lane) {
        const std::uint8_t hb = qh[half * 32 + lane];
        int* v = vals + half * 128 + lane;
        v[0] = static_cast<int>((hb & 0x03U) << 4) |
               static_cast<int>(ql[qbase + lane] & 0x0FU);
        v[32] = static_cast<int>(((hb >> 2) & 0x03U) << 4) |
                static_cast<int>(ql[qbase + 32 + lane] & 0x0FU);
        v[64] = static_cast<int>(((hb >> 4) & 0x03U) << 4) |
                static_cast<int>(ql[qbase + lane] >> 4);
        v[96] = static_cast<int>(((hb >> 6) & 0x03U) << 4) |
                static_cast<int>(ql[qbase + 32 + lane] >> 4);
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int e = 0; e < 128; ++e) {
        const int si = half * 8 + (e % 32) / 16 + (e / 32) * 2;
        o[half * 128 + e] =
            fd * static_cast<float>(static_cast<std::int8_t>(sc[si])) *
            static_cast<float>(vals[half * 128 + e] - 32);
      }
  } else {
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      o[j] = d * static_cast<float>(static_cast<int>(b[2 + j] & 0x0FU) - 8);
      o[j + 16] = d * static_cast<float>(static_cast<int>(b[2 + j] >> 4U) - 8);
    }
  }
}

/// Block-float group: one shared exponent byte, eight signed 8-bit mantissas
/// (value = q * 2^(e - 127) / 64). Same encoding as the shared NPU staging.
__device__ inline void D1EncodeGroup(const float* v, std::uint8_t* out) {
  unsigned max_exp = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    max_exp = max(max_exp, (__float_as_uint(v[i]) >> 23) & 0xFFu);
  const float inv = __uint_as_float(max_exp >= 7 ? (260u - max_exp) << 23 : 0u);
  out[0] = static_cast<std::uint8_t>(max_exp);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float q = fminf(127.0F, fmaxf(-128.0F, rintf(v[i] * inv)));
    out[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
  }
}

/// A tiles of TileM rows x 64 K for an NPU program built with m = TileM.
template<int TileM>
__global__ void __launch_bounds__(256)
    D1EncodeActKernel(const __half* __restrict__ x,
                      std::uint8_t* __restrict__ out, std::uint32_t k_slice,
                      std::size_t ldx, std::uint32_t valid_rows) {
  __shared__ __align__(16) __half tile[TileM * 64];
  __shared__ __align__(16) std::uint8_t staged[TileM * 64 / 8 * 9];
  const std::uint32_t tiles_k = k_slice / 64;
  const std::uint32_t tile_row = blockIdx.x / tiles_k;
  const std::uint32_t tile_col = blockIdx.x % tiles_k;
  const std::uint32_t t = threadIdx.x;
#pragma unroll
  for (int j = 0; j < TileM / 32; ++j) {
    const std::uint32_t row = (t >> 3) + 32 * j;
    const std::uint32_t vec = t & 7;
    const std::uint32_t col = tile_col * 64 + vec * 8;
    uint4 raw = make_uint4(0, 0, 0, 0);
    const std::size_t abs_row = std::size_t{tile_row} * TileM + row;
    if (abs_row < valid_rows)
      raw = *reinterpret_cast<const uint4*>(&x[abs_row * ldx + col]);
    *reinterpret_cast<uint4*>(&tile[row * 64 + vec * 8]) = raw;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < TileM / 32; ++j) {
    const std::uint32_t gi = t + 256 * j;
    const std::uint32_t rib = gi & 7;
    const std::uint32_t block = (gi >> 3) & 1;
    const std::uint32_t sc = (gi >> 4) & 7;
    const std::uint32_t sr = gi >> 7;
    const std::uint32_t row = sr * 16 + block * 8 + rib;
    const uint4 raw = *reinterpret_cast<const uint4*>(&tile[row * 64 + sc * 8]);
    const __half* h = reinterpret_cast<const __half*>(&raw);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = __half2float(h[i]);
    D1EncodeGroup(v, staged + gi * 9);
  }
  __syncthreads();
  const uint4* src = reinterpret_cast<const uint4*>(staged);
  uint4* dst = reinterpret_cast<uint4*>(
      out + static_cast<std::size_t>(blockIdx.x) * (TileM * 64 / 8 * 9));
  for (std::uint32_t i = t; i < TileM * 64 / 8 * 9 / 16; i += 256)
    dst[i] = src[i];
}

/// C tiles of (4 * TileM) rows x 128 columns -> row-major floats.
template<int TileM>
__global__ void D1DecodeOutKernel(const std::uint8_t* __restrict__ in,
                                  float* __restrict__ out, std::uint32_t tokens,
                                  std::uint32_t cols, std::size_t out_pitch,
                                  std::uint32_t valid_cols) {
  constexpr std::uint32_t kRows = 4 * TileM;
  const std::uint32_t padded = (tokens + kRows - 1) / kRows * kRows;
  const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (g >= std::size_t{padded} * cols / 8)
    return;
  constexpr std::uint32_t kGroupsPerTile = kRows * 128 / 8;
  const std::size_t tile = g / kGroupsPerTile;
  const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
  const std::uint32_t tiles_n = cols / 128;
  const std::uint32_t tile_row = static_cast<std::uint32_t>(tile / tiles_n);
  const std::uint32_t tile_col = static_cast<std::uint32_t>(tile % tiles_n);
  const std::uint32_t rib = gi & 7;
  const std::uint32_t bc = (gi >> 3) & 1;
  const std::uint32_t br = (gi >> 4) & 1;
  const std::uint32_t sc = (gi >> 5) & 7;
  const std::uint32_t sr = gi >> 8;
  const std::size_t row =
      std::size_t{tile_row} * kRows + (sr * 2 + br) * 8 + rib;
  const std::uint32_t col = tile_col * 128 + (sc * 2 + bc) * 8;
  if (row >= tokens || (valid_cols != 0 && col >= valid_cols))
    return;
  const std::uint8_t* p = in + g * 9;
  const float mult = ldexpf(1.0F, static_cast<int>(p[0]) - 127) / 64.0F;
  float* dst = out + row * out_pitch + col;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    dst[i] = static_cast<float>(static_cast<std::int8_t>(p[1 + i])) * mult;
}

template<int Kind>
__global__ void D1EmbedKernel(const std::uint8_t* __restrict__ table,
                              const std::int32_t* __restrict__ ids,
                              float* __restrict__ out, std::uint32_t n,
                              std::uint32_t hidden, std::uint64_t row_bytes,
                              std::uint32_t vocab, std::size_t units) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= units)
    return;
  constexpr std::uint32_t kUnit = Kind == 1 ? 256u : 32u;
  const std::uint32_t per_row = hidden / kUnit;
  const std::uint32_t row = static_cast<std::uint32_t>(g / per_row);
  const std::uint32_t unit = static_cast<std::uint32_t>(g % per_row);
  float* o = out + static_cast<std::size_t>(row) * hidden + unit * kUnit;
  if (row < n) {
    const std::int32_t id = ids[row];
    if (id >= 0 && static_cast<std::uint32_t>(id) < vocab) {
      D1DequantUnit<Kind>(table + static_cast<std::size_t>(id) * row_bytes +
                              unit * (Kind == 1   ? 210u
                                      : Kind == 2 ? 34u
                                                  : 18u),
                          o);
      return;
    }
  }
  for (std::uint32_t e = 0; e < kUnit; ++e)
    o[e] = 0.0F;
}

/// Logit of vocabulary entry `ids[blockIdx.x]` for slot row `blockIdx.y`.
template<int Kind>
__global__ void D1LogitKernel(const std::uint8_t* __restrict__ table,
                              const std::int32_t* __restrict__ ids,
                              const float* __restrict__ hidden_rows,
                              float* __restrict__ out, std::uint32_t hidden,
                              std::uint64_t row_bytes, std::uint32_t cols) {
  constexpr int kUnit = Kind == 1 ? 256 : 32;
  const std::uint32_t j = blockIdx.x, s = blockIdx.y;
  const std::uint8_t* row =
      table + static_cast<std::size_t>(ids[j]) * row_bytes;
  const float* vec = hidden_rows + static_cast<std::size_t>(s) * hidden;
  float partial = 0.0F;
  for (std::uint32_t unit = threadIdx.x; unit < hidden / kUnit;
       unit += blockDim.x) {
    float v[kUnit];
    D1DequantUnit<Kind>(row + unit * (Kind == 1   ? 210u
                                      : Kind == 2 ? 34u
                                                  : 18u),
                        v);
    const float* vv = vec + static_cast<std::size_t>(unit) * kUnit;
    for (int e = 0; e < kUnit; ++e)
      partial += v[e] * vv[e];
  }
  __shared__ float red[kThreads];
  red[threadIdx.x] = partial;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (unsigned i = 0; i < blockDim.x; ++i)
      sum += red[i];
    out[static_cast<std::size_t>(s) * cols + j] = sum;
  }
}

__global__ void D1AddKernel(float* x, const float* y, std::size_t cnt) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < cnt)
    x[i] += y[i];
}

__global__ void D1ToHalfKernel(const float* x, __half* y, std::size_t cnt) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < cnt)
    y[i] = __float2half(x[i]);
}

/// act = silu(gate) * up from the fused gate|up GEMM rows (pitch `pitch`).
__global__ void D1SwigluKernel(const float* __restrict__ gu,
                               float* __restrict__ act, std::uint32_t inter,
                               std::uint32_t pitch, std::size_t cnt) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cnt)
    return;
  const std::size_t row = i / inter;
  const std::uint32_t col = static_cast<std::uint32_t>(i % inter);
  const float g = gu[row * pitch + col];
  act[i] = g / (1.0F + __expf(-g)) * gu[row * pitch + inter + col];
}

/// Gated short convolution: P rows are [B | C | x] (3*hidden). out[t][c] =
/// C[t][c] * sum_j w[c][j] * (B*x)[t-(L-1)+j][c]. Before the token's sequence
/// start the history is zero, or `hist` ((taps-1) x hidden, oldest first)
/// when continuing a stream. With `bx_all` the combined (history + new) B*x
/// stream is recorded so a speculative pass can commit any accepted prefix.
__global__ void D1ConvKernel(const float* __restrict__ p,
                             const float* __restrict__ w,
                             const std::int32_t* __restrict__ seq_lo,
                             const float* __restrict__ hist,
                             float* __restrict__ bx_all,
                             float* __restrict__ out, std::uint32_t n,
                             std::uint32_t hidden, std::uint32_t taps) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (c >= hidden || t >= n)
    return;
  const std::size_t pitch = 3ull * hidden;
  const int lo = seq_lo[t];
  float sum = 0.0F;
  for (std::uint32_t j = 0; j < taps; ++j) {
    const int s =
        static_cast<int>(t) - static_cast<int>(taps - 1) + static_cast<int>(j);
    float bx;
    if (s >= lo && s >= 0)
      bx = p[s * pitch + c] * p[s * pitch + 2ull * hidden + c];
    else if (s < 0 && hist != nullptr)
      bx = hist[static_cast<std::size_t>(static_cast<int>(taps - 1) + s) *
                    hidden +
                c];
    else
      continue;
    sum += w[static_cast<std::size_t>(c) * taps + j] * bx;
  }
  out[static_cast<std::size_t>(t) * hidden + c] =
      sum * p[t * pitch + hidden + c];
  if (bx_all != nullptr) {
    bx_all[(static_cast<std::size_t>(taps - 1) + t) * hidden + c] =
        p[t * pitch + c] * p[t * pitch + 2ull * hidden + c];
    if (t < taps - 1 && hist != nullptr)
      bx_all[static_cast<std::size_t>(t) * hidden + c] =
          hist[static_cast<std::size_t>(t) * hidden + c];
  }
}

/// New history = the (taps-1) entries of the combined stream that end at the
/// last accepted token.
__global__ void D1ConvCommitKernel(const float* __restrict__ bx_all,
                                   float* __restrict__ hist,
                                   std::uint32_t accepted, std::uint32_t hidden,
                                   std::uint32_t taps) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= hidden)
    return;
  for (std::uint32_t i = 0; i + 1 < taps; ++i)
    hist[static_cast<std::size_t>(i) * hidden + c] =
        bx_all[static_cast<std::size_t>(accepted + i) * hidden + c];
}

__global__ void D1SplitQkvKernel(const float* qkv, float* q, float* k, float* v,
                                 std::uint32_t n, std::uint32_t q_dim,
                                 std::uint32_t kv_dim, std::uint32_t pitch) {
  const std::uint32_t t = blockIdx.x;
  if (t >= n)
    return;
  for (std::uint32_t i = threadIdx.x; i < q_dim + 2 * kv_dim; i += blockDim.x) {
    const float value = qkv[static_cast<std::size_t>(t) * pitch + i];
    if (i < q_dim)
      q[static_cast<std::size_t>(t) * q_dim + i] = value;
    else if (i < q_dim + kv_dim)
      k[static_cast<std::size_t>(t) * kv_dim + (i - q_dim)] = value;
    else
      v[static_cast<std::size_t>(t) * kv_dim + (i - q_dim - kv_dim)] = value;
  }
}

/// gate[t][h] = dot(wgate[h], z[t]); one block per token.
__global__ void D1GateKernel(const float* __restrict__ z,
                             const float* __restrict__ wgate,
                             float* __restrict__ gate, std::uint32_t hidden,
                             std::uint32_t heads) {
  const std::uint32_t t = blockIdx.x;
  __shared__ float red[kThreads];
  for (std::uint32_t h = 0; h < heads; ++h) {
    float partial = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
      partial += wgate[static_cast<std::size_t>(h) * hidden + i] *
                 z[static_cast<std::size_t>(t) * hidden + i];
    red[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride)
        red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      gate[static_cast<std::size_t>(t) * heads + h] = red[0];
    __syncthreads();
  }
}

/// Half-split (NEOX) rotary embedding at position = row; `factors` (global
/// layers) divide the angle per frequency index.
__global__ void D1RopeKernel(float* q, float* k,
                             const float* __restrict__ factors,
                             const std::int32_t* __restrict__ seq_lo,
                             const std::int32_t* __restrict__ pos_arr,
                             std::uint32_t pos_base, std::uint32_t n,
                             std::uint32_t heads, std::uint32_t kv_heads,
                             std::uint32_t hd, float theta) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t dd = threadIdx.x;
  if (row >= n || dd >= hd / 2)
    return;
  float* base;
  if (h < heads) {
    base = q + (static_cast<std::size_t>(row) * heads + h) * hd;
  } else {
    const std::uint32_t kh = h - heads;
    if (kh >= kv_heads)
      return;
    base = k + (static_cast<std::size_t>(row) * kv_heads + kh) * hd;
  }
  const float position =
      pos_arr != nullptr
          ? static_cast<float>(pos_arr[row])
          : static_cast<float>(row - static_cast<std::uint32_t>(seq_lo[row]) +
                               pos_base);
  float angle = position * powf(theta, -2.0F * static_cast<float>(dd) /
                                           static_cast<float>(hd));
  if (factors != nullptr)
    angle /= factors[dd];
  const float c = cosf(angle), sn = sinf(angle);
  const float a = base[dd], b = base[dd + hd / 2];
  base[dd] = a * c - b * sn;
  base[dd + hd / 2] = b * c + a * sn;
}

/// One block per (token, head), online softmax over key chunks. Stateless:
/// tokens see their own sequence [seq_lo, seq_hi) of the current buffers
/// (global layers bidirectional, restricted by `groups`; window > 0 causal).
/// Stateful: keys are absolute positions in the K/V cache, the query sits at
/// pos0 + t, and every layer is causal (window > 0 also limits the span).
__global__ void D1AttnKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate,
    const std::int32_t* __restrict__ groups,
    const std::int32_t* __restrict__ seq_lo,
    const std::int32_t* __restrict__ seq_hi, float* __restrict__ out,
    std::uint32_t n, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t hd, float scale, std::uint32_t window, std::uint32_t stateful,
    std::uint32_t pos0, const std::int32_t* __restrict__ pos_arr,
    std::size_t stream_stride) {
  const std::uint32_t t = blockIdx.x, h = blockIdx.y;
  if (stream_stride != 0) {
    k += static_cast<std::size_t>(t) * stream_stride;
    v += static_cast<std::size_t>(t) * stream_stride;
  }
  const std::uint32_t kh = h / (heads / kv_heads);
  const std::uint32_t kv_dim = kv_heads * hd;
  __shared__ float qs[128];
  __shared__ float sc[M];
  __shared__ float red[kThreads];
  qs[threadIdx.x] =
      q[(static_cast<std::size_t>(t) * heads + h) * hd + threadIdx.x];
  __syncthreads();
  std::uint32_t first, last;
  if (stateful) {
    const std::uint32_t qa =
        pos_arr != nullptr ? static_cast<std::uint32_t>(pos_arr[t]) : pos0 + t;
    last = qa;
    first = window > 0 && qa + 1 > window ? qa + 1 - window : 0;
  } else {
    first = static_cast<std::uint32_t>(seq_lo[t]);
    last = static_cast<std::uint32_t>(seq_hi[t]) - 1;
    if (window > 0) {
      last = t;
      if (t + 1 > window && t + 1 - window > first)
        first = t + 1 - window;
    }
  }
  float m = -INFINITY, l = 0.0F, acc = 0.0F;
  for (std::uint32_t s0 = first; s0 <= last; s0 += M) {
    const std::uint32_t chunk =
        min(static_cast<std::uint32_t>(M), last - s0 + 1);
    float local_max = -INFINITY;
    for (std::uint32_t i = threadIdx.x; i < chunk; i += blockDim.x) {
      const std::uint32_t s = s0 + i;
      bool valid = true;
      if (!stateful && window == 0) {
        const std::int32_t own = groups[t], key = groups[s];
        valid = own < 0 || key < 0 || own == key;
      }
      float score = -INFINITY;
      if (valid) {
        const float* kr = k + static_cast<std::size_t>(s) * kv_dim + kh * hd;
        float dot = 0.0F;
        for (std::uint32_t d = 0; d < hd; ++d)
          dot += qs[d] * kr[d];
        score = dot * scale;
      }
      sc[i] = score;
      local_max = fmaxf(local_max, score);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride)
        red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
      __syncthreads();
    }
    const float chunk_max = red[0];
    __syncthreads();
    const float new_m = fmaxf(m, chunk_max);
    const float alpha = m == -INFINITY ? 0.0F : __expf(m - new_m);
    float local_sum = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < chunk; i += blockDim.x) {
      const float e = sc[i] == -INFINITY ? 0.0F : __expf(sc[i] - new_m);
      sc[i] = e;
      local_sum += e;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride)
        red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    l = l * alpha + red[0];
    __syncthreads();
    acc *= alpha;
    for (std::uint32_t i = 0; i < chunk; ++i) {
      const float p = sc[i];
      if (p != 0.0F)
        acc += p * v[static_cast<std::size_t>(s0 + i) * kv_dim + kh * hd +
                     threadIdx.x];
    }
    m = new_m;
    __syncthreads();
  }
  const float g =
      1.0F / (1.0F + __expf(-gate[static_cast<std::size_t>(t) * heads + h]));
  out[(static_cast<std::size_t>(t) * heads + h) * hd + threadIdx.x] =
      acc / l * g;
}

/// logits[r][v] = dot(embedding row v, hidden_rows[r]) for every vocabulary
/// row v and up to kMaxRows hidden vectors. One block per vocabulary row.
constexpr int kMaxRows = 32;
template<int Kind>
__global__ void D1LogitAllKernel(const std::uint8_t* __restrict__ table,
                                 const float* __restrict__ hidden_rows,
                                 float* __restrict__ out, std::uint32_t hidden,
                                 std::uint64_t row_bytes, std::uint32_t vocab,
                                 std::uint32_t rows) {
  constexpr int kUnit = Kind == 1 ? 256 : 32;
  const std::uint32_t v = blockIdx.x;
  const std::uint8_t* row = table + static_cast<std::size_t>(v) * row_bytes;
  float partial[kMaxRows];
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r)
    partial[r] = 0.0F;
  for (std::uint32_t unit = threadIdx.x; unit < hidden / kUnit;
       unit += blockDim.x) {
    float w[kUnit];
    D1DequantUnit<Kind>(row + unit * (Kind == 1   ? 210u
                                      : Kind == 2 ? 34u
                                                  : 18u),
                        w);
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if (static_cast<std::uint32_t>(r) >= rows)
        break;
      const float* hv = hidden_rows + static_cast<std::size_t>(r) * hidden +
                        static_cast<std::size_t>(unit) * kUnit;
      float dot = 0.0F;
      for (int e = 0; e < kUnit; ++e)
        dot += w[e] * hv[e];
      partial[r] += dot;
    }
  }
  __shared__ float red[kMaxRows][64];
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r)
    red[r][threadIdx.x] = partial[r];
  __syncthreads();
  for (unsigned stride = 32; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
#pragma unroll
      for (int r = 0; r < kMaxRows; ++r)
        red[r][threadIdx.x] += red[r][threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x < kMaxRows && threadIdx.x < rows)
    out[static_cast<std::size_t>(threadIdx.x) * vocab + v] =
        red[threadIdx.x][0];
}

/// argmax over `vocab` logits per row; one block of 1024 threads per row.
__global__ void D1ArgmaxKernel(const float* __restrict__ logits,
                               std::uint32_t vocab,
                               std::int32_t* __restrict__ out) {
  const float* row = logits + static_cast<std::size_t>(blockIdx.x) * vocab;
  float best = -INFINITY;
  std::int32_t best_id = 0;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += blockDim.x)
    if (row[i] > best) {
      best = row[i];
      best_id = static_cast<std::int32_t>(i);
    }
  __shared__ float vals[1024];
  __shared__ std::int32_t ids[1024];
  vals[threadIdx.x] = best;
  ids[threadIdx.x] = best_id;
  __syncthreads();
  for (unsigned stride = 512; stride > 0; stride /= 2) {
    if (threadIdx.x < stride &&
        (vals[threadIdx.x + stride] > vals[threadIdx.x] ||
         (vals[threadIdx.x + stride] == vals[threadIdx.x] &&
          ids[threadIdx.x + stride] < ids[threadIdx.x]))) {
      vals[threadIdx.x] = vals[threadIdx.x + stride];
      ids[threadIdx.x] = ids[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[blockIdx.x] = ids[0];
}

/// z[j] = h + silu(W[j] h) for every drafter head j; one block per output.
__global__ void D1DraftHeadKernel(const __half* __restrict__ w,
                                  const float* __restrict__ h,
                                  float* __restrict__ z, std::uint32_t hidden) {
  const std::uint32_t j = blockIdx.y, o = blockIdx.x;
  const __half* row = w + (static_cast<std::size_t>(j) * hidden + o) * hidden;
  float partial = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    partial += __half2float(row[i]) * h[i];
  __shared__ float red[kThreads];
  red[threadIdx.x] = partial;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float v = red[0];
    z[static_cast<std::size_t>(j) * hidden + o] =
        h[o] + v / (1.0F + __expf(-v));
  }
}

/// Batched decode: row t is stream t with its own conv history
/// ((taps-1) x hidden, oldest first), updated in place.
__global__ void D1ConvBatchKernel(const float* __restrict__ p,
                                  const float* __restrict__ w,
                                  float* __restrict__ hist_all,
                                  float* __restrict__ out, std::uint32_t n,
                                  std::uint32_t hidden, std::uint32_t taps) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (c >= hidden || t >= n)
    return;
  const std::size_t pitch = 3ull * hidden;
  float* hist = hist_all + static_cast<std::size_t>(t) * (taps - 1) * hidden;
  const float bx = p[t * pitch + c] * p[t * pitch + 2ull * hidden + c];
  float sum = w[static_cast<std::size_t>(c) * taps + taps - 1] * bx;
  for (std::uint32_t j = 0; j + 1 < taps; ++j)
    sum += w[static_cast<std::size_t>(c) * taps + j] *
           hist[static_cast<std::size_t>(j) * hidden + c];
  out[static_cast<std::size_t>(t) * hidden + c] =
      sum * p[t * pitch + hidden + c];
  for (std::uint32_t j = 0; j + 2 < taps; ++j)
    hist[static_cast<std::size_t>(j) * hidden + c] =
        hist[static_cast<std::size_t>(j + 1) * hidden + c];
  hist[static_cast<std::size_t>(taps - 2) * hidden + c] = bx;
}

/// Writes row t's K and V at its stream's current position.
__global__ void D1StoreKvBatchKernel(const float* __restrict__ k,
                                     const float* __restrict__ v,
                                     float* __restrict__ ck,
                                     float* __restrict__ cv,
                                     const std::int32_t* __restrict__ pos,
                                     std::size_t stream_stride,
                                     std::uint32_t kv_dim, std::uint32_t n) {
  const std::uint32_t t = blockIdx.x;
  if (t >= n)
    return;
  const std::size_t base = static_cast<std::size_t>(t) * stream_stride +
                           static_cast<std::size_t>(pos[t]) * kv_dim;
  for (std::uint32_t i = threadIdx.x; i < kv_dim; i += blockDim.x) {
    ck[base + i] = k[static_cast<std::size_t>(t) * kv_dim + i];
    cv[base + i] = v[static_cast<std::size_t>(t) * kv_dim + i];
  }
}

template<typename T>
T* DeviceAlloc(std::size_t count) {
  void* p = nullptr;
  if (hipMalloc(&p, count * sizeof(T)) != hipSuccess)
    return nullptr;
  return static_cast<T*>(p);
}

template<typename T>
T* PinnedAlloc(std::size_t count) {
  void* p = nullptr;
  if (hipHostMalloc(&p, count * sizeof(T)) != hipSuccess)
    return nullptr;
  return static_cast<T*>(p);
}

}  // namespace

struct D1NpuModel::Impl {
  lfm3::Weights w;
  std::unique_ptr<rocm::NpuGemm> npu;
  hipStream_t stream{nullptr};
  int prog_in{-1}, prog_h{-1}, prog_qkv{-1}, prog_gu{-1}, prog_down{-1};
  // Per layer weight buffer ids (conv layers use in/out, attention qkv/out).
  std::vector<int> w_in, w_out, w_qkv, w_gu, w_down;
  int a_h{-1}, a_i{-1}, c_in{-1}, c_h{-1}, c_qkv{-1}, c_gu{-1};

  std::uint8_t* emb_table{nullptr};
  float *g_op{nullptr}, *g_ffn{nullptr}, *g_post{nullptr}, *g_post_ffn{nullptr},
      *g_qn{nullptr}, *g_kn{nullptr}, *g_out{nullptr}, *g_conv{nullptr},
      *g_gate{nullptr}, *g_factors{nullptr};
  float *x{nullptr}, *z{nullptr}, *z2{nullptr}, *branch{nullptr},
      *p_buf{nullptr}, *conv_out{nullptr}, *qkv_buf{nullptr}, *q{nullptr},
      *k{nullptr}, *v{nullptr}, *qn{nullptr}, *kn{nullptr}, *gate_buf{nullptr},
      *attn{nullptr}, *gu_buf{nullptr}, *act{nullptr}, *sel{nullptr},
      *sel_norm{nullptr}, *logit_buf{nullptr};
  __half* half_buf{nullptr};
  std::int32_t *ids{nullptr}, *group_buf{nullptr}, *vocab_ids{nullptr};
  std::int32_t *seq_lo_h{nullptr}, *seq_hi_h{nullptr};  // pinned, per token
  std::int32_t *seq_lo{nullptr}, *seq_hi{nullptr};      // device copies
  mutable std::int32_t vocab_capacity{0};

  // Generation state (EnableGeneration): K/V caches for the attention layers,
  // conv history and the combined B*x stream of the last pass per conv layer.
  bool stateful{false};
  std::uint32_t pos0{0}, pos{0}, max_ctx{0}, last_n{0};
  std::vector<float*> k_cache, v_cache, conv_hist, conv_bx;
  float* logit_all{nullptr};
  float* attn_scratch{nullptr};
  __half* draft_w{nullptr};
  float* draft_z{nullptr};
  std::uint32_t draft_k{0};
  // Batched decode: up to batch_streams independent streams, one row each.
  bool batch{false};
  std::uint32_t batch_streams{0}, batch_ctx{0};
  std::int32_t* pos_dev{nullptr};
  std::int32_t* pos_host{nullptr};
  std::vector<float*> bk_cache, bv_cache, bconv_hist;
  std::int32_t* argmax_dev{nullptr};
  std::int32_t* argmax_host{nullptr};
  bool Layers(std::uint32_t n);
  mutable double t_encode{0}, t_gemm{0}, t_half{0}, t_memset{0}, t_enckernel{0},
      t_final{0};
  std::uint32_t tile_m{128};  // NPU program tile rows per core row (128 or 64)
  std::uint32_t pass_rows{
      512};  // rows the program computes per pass (4 * tile_m)
  double step_layers_ms{0};
  bool prof_fine{false};
  bool sync_each{true};
  bool use_runsync{true};
  mutable int n_gemm{0};
  std::size_t hidden{0}, inter{0}, heads{0}, hd{0}, q_dim{0};
  std::uint32_t taps{0};

  static constexpr std::uint32_t kInPitch = 6144, kQkvPitch = 3072,
                                 kGuPitch = 15360;

  // Writes `rows` x cols of `src` (float) to a block-float A buffer.
  void EncodeA(const float* src, std::uint32_t n, std::uint32_t cols,
               int buf) const {
    const auto t0 = std::chrono::steady_clock::now();
    struct Done {
      const Impl* self;
      std::chrono::steady_clock::time_point t0;
      ~Done() {
        self->t_encode += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
      }
    } done{this, t0};
    const std::uint32_t enc_rows = (n + tile_m - 1) / tile_m * tile_m;
    const bool fine = prof_fine;
    auto lap = [&](double& acc, std::chrono::steady_clock::time_point& t) {
      if (!fine)
        return;
      hipStreamSynchronize(stream);
      const auto now = std::chrono::steady_clock::now();
      acc += std::chrono::duration<double, std::milli>(now - t).count();
      t = now;
    };
    auto t = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(D1ToHalfKernel,
                       dim3(static_cast<std::uint32_t>(
                           (static_cast<std::size_t>(n) * cols + 255) / 256)),
                       dim3(256), 0, stream, src, half_buf,
                       static_cast<std::size_t>(n) * cols);
    lap(t_half, t);
    hipMemsetAsync(
        half_buf + static_cast<std::size_t>(n) * cols, 0,
        static_cast<std::size_t>(enc_rows - n) * cols * sizeof(__half), stream);
    lap(t_memset, t);
    if (tile_m == 128)
      rocm::NpuEncodeActivations(half_buf, cols, enc_rows, cols, 0, cols,
                                 npu->Device(buf), stream);
    else
      hipLaunchKernelGGL(
          D1EncodeActKernel<64>, dim3(enc_rows / 64 * (cols / 64)), dim3(256),
          0, stream, half_buf, static_cast<std::uint8_t*>(npu->Device(buf)),
          cols, static_cast<std::size_t>(cols), enc_rows);
    lap(t_enckernel, t);
    // FinalizeWeights only copies in isolate mode (the GPU must be done then);
    // otherwise the stream already orders the encode before the NPU run.
    if (sync_each)
      hipStreamSynchronize(stream);
    npu->FinalizeWeights(buf);
    lap(t_final, t);
    if (std::getenv("GUFO_D1_TRACE")) {
      const auto* b = static_cast<const std::uint8_t*>(npu->Host(buf));
      const std::size_t bytes = std::size_t{enc_rows} * cols / 8 * 9;
      std::size_t nz = 0;
      for (std::size_t i = 0; i < bytes; ++i)
        nz += b[i] != 0;
      std::fprintf(
          stderr,
          "[d1 npu ] act buf %d rows %u cols %u bytes %zu nonzero %zu\n", buf,
          enc_rows, cols, bytes, nz);
    }
  }

  bool Gemm(int program, int a, int weight, int c) const {
    const auto t0 = std::chrono::steady_clock::now();
    struct Done {
      const Impl* self;
      std::chrono::steady_clock::time_point t0;
      ~Done() {
        self->t_gemm += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        ++self->n_gemm;
      }
    } done{this, t0};
    if (use_runsync) {
      if (!npu->RunSync(stream, a, weight, c, program))
        return false;
    } else {
      const int bs[1] = {weight};
      const int cs[1] = {c};
      if (!npu->Submit(stream, a, bs, cs, 1, program))
        return false;
      if (!npu->Wait(stream))
        return false;
    }
    if (sync_each)
      hipStreamSynchronize(stream);
    return npu->Healthy();
  }

  // Block-float C stream -> row-major floats (first valid_cols columns).
  void Decode(int c_buf, float* out, std::uint32_t n, std::uint32_t cols,
              std::size_t pitch, std::uint32_t valid_cols) const {
    if (tile_m == 128) {
      rocm::NpuDecodeOutput(npu->Staged(stream, c_buf), out, n, cols, pitch, 0,
                            false, stream, valid_cols);
      return;
    }
    const std::uint32_t rows = (n + 255u) / 256u * 256u;
    const std::size_t groups = std::size_t{rows} * cols / 8;
    hipLaunchKernelGGL(
        D1DecodeOutKernel<64>,
        dim3(static_cast<std::uint32_t>((groups + 255) / 256)), dim3(256), 0,
        stream, static_cast<const std::uint8_t*>(npu->Staged(stream, c_buf)),
        out, n, cols, pitch, valid_cols);
  }

  void Add(float* dst, const float* src, std::uint32_t n) const {
    hipLaunchKernelGGL(D1AddKernel,
                       dim3(static_cast<std::uint32_t>(
                           (static_cast<std::size_t>(n) * hidden + 255) / 256)),
                       dim3(256), 0, stream, dst, src,
                       static_cast<std::size_t>(n) * hidden);
  }
};

D1NpuModel::D1NpuModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
D1NpuModel::~D1NpuModel() = default;
const lfm3::Weights& D1NpuModel::weights() const {
  return impl_->w;
}

std::unique_ptr<D1NpuModel> D1NpuModel::Create(const core::GgufReader& reader,
                                               std::string* error) {
  auto fail = [&](const std::string& message) -> std::unique_ptr<D1NpuModel> {
    if (error)
      *error = "d1 npu: " + message;
    return nullptr;
  };
  std::string err;
  auto bound = lfm3::Weights::Bind(reader, &err);
  if (!bound)
    return fail(err);
  auto impl = std::make_unique<Impl>();
  impl->w = *bound;
  const auto& c = impl->w.config;
  if (c.hidden != 2048 || c.intermediate != 7680 || c.heads != 16 ||
      c.head_dim != 128 || c.conv_width != 7)
    return fail("only the 2048/7680/16x128 d1 geometry has NPU programs");
  impl->hidden = c.hidden;
  impl->inter = c.intermediate;
  impl->heads = c.heads;
  impl->hd = c.head_dim;
  impl->q_dim = c.heads * c.head_dim;
  impl->taps = c.conv_width;
  const std::size_t H = impl->hidden, I = impl->inter, Q = impl->q_dim;
  for (const auto& l : impl->w.layers) {
    for (const auto* t : {&l.gate, &l.up, &l.down, &l.in_proj, &l.out_proj,
                          &l.q, &l.k, &l.v, &l.attn_out}) {
      if (t->data && t->type != GgmlType::kQ8_0 && t->type != GgmlType::kQ4_K &&
          t->type != GgmlType::kQ5_K && t->type != GgmlType::kQ6_K)
        return fail(
            "projection matrices must be Q8_0, Q4_K, Q5_K or Q6_K (the NPU "
            "weight encoder does not read Q4_0)");
    }
  }
  const auto emb_type = impl->w.embedding.type;
  if (emb_type != GgmlType::kQ4_0 && emb_type != GgmlType::kQ6_K &&
      emb_type != GgmlType::kQ8_0)
    return fail("embedding must be Q4_0, Q6_K or Q8_0");

  const char* npu_dir = std::getenv("GUFO_NPU_DIR");
  const std::string dir = npu_dir ? npu_dir : "tools/npu";
  // GUFO_D1_TILE=64 selects the 256-row NPU program (m = 64 per core row).
  // Default: the 256-row program when its XCLBIN and streams are installed.
  impl->tile_m =
      std::ifstream(dir + "/runtime_mk/final_4096xKx1024_64x64x128.xclbin")
                  .good() &&
              std::ifstream(dir +
                            "/runtime_mk/insts_256x2048x15360_64x64x128.txt")
                  .good()
          ? 64u
          : 128u;
  if (const char* tile_env = std::getenv("GUFO_D1_TILE"))
    impl->tile_m = std::atoi(tile_env) == 64 ? 64u : 128u;
  impl->pass_rows = 4 * impl->tile_m;
  const std::string tile_tag = std::to_string(impl->tile_m) + "x64x128";
  const std::uint32_t pass_rows = impl->pass_rows;
  auto insts = [&](std::uint32_t k, std::uint32_t n) {
    return dir + "/runtime_mk/insts_" + std::to_string(pass_rows) + "x" +
           std::to_string(k) + "x" + std::to_string(n) + "_" + tile_tag +
           ".txt";
  };
  const std::pair<std::uint32_t, std::uint32_t> shapes[] = {
      {2048, 1024}, {2048, 2048},  {2048, 3072},
      {2048, 6144}, {2048, 15360}, {7680, 2048}};
  for (const auto& [k, n] : shapes)
    if (!std::ifstream(insts(k, n)).good())
      return fail("missing instruction file " + insts(k, n));

  rocm::NpuGemm::Config cfg;
  cfg.xclbin = dir + "/runtime_mk/final_4096xKx1024_" + tile_tag + ".xclbin";
  cfg.insts = insts(2048, 1024);
  cfg.tokens = pass_rows;
  cfg.k = 2048;
  cfg.n = 1024;
  impl->npu = rocm::NpuGemm::Create(cfg, &err);
  if (!impl->npu)
    return fail(err);
  impl->prog_h = impl->npu->AddProgram(insts(2048, 2048), &err);
  impl->prog_qkv = impl->npu->AddProgram(insts(2048, 3072), &err);
  impl->prog_in = impl->npu->AddProgram(insts(2048, 6144), &err);
  impl->prog_gu = impl->npu->AddProgram(insts(2048, 15360), &err);
  impl->prog_down = impl->npu->AddProgram(insts(7680, 2048), &err);
  if (impl->prog_h < 0 || impl->prog_qkv < 0 || impl->prog_in < 0 ||
      impl->prog_gu < 0 || impl->prog_down < 0)
    return fail("program load failed: " + err);

  hipStreamCreate(&impl->stream);
  auto a_bytes = [&](std::size_t cols) {
    return std::size_t{pass_rows} * cols / 8 * 9;
  };
  impl->a_h = impl->npu->NewBuffer(a_bytes(H), nullptr);
  impl->a_i = impl->npu->NewBuffer(a_bytes(I), nullptr);
  impl->c_in = impl->npu->NewBuffer(a_bytes(Impl::kInPitch), nullptr);
  impl->c_h = impl->npu->NewBuffer(a_bytes(H), nullptr);
  impl->c_qkv = impl->npu->NewBuffer(a_bytes(Impl::kQkvPitch), nullptr);
  impl->c_gu = impl->npu->NewBuffer(a_bytes(Impl::kGuPitch), nullptr);
  if (impl->a_h < 0 || impl->a_i < 0 || impl->c_in < 0 || impl->c_h < 0 ||
      impl->c_qkv < 0 || impl->c_gu < 0)
    return fail("staging buffer allocation failed");
  std::memset(impl->npu->Host(impl->a_h), 0, a_bytes(H));
  std::memset(impl->npu->Host(impl->a_i), 0, a_bytes(I));

  // Activation / scratch buffers (pinned, zero-copy for the GPU and host).
  auto& s = *impl;
  s.x = DeviceAlloc<float>(M * H);
  s.z = DeviceAlloc<float>(M * H);
  s.z2 = DeviceAlloc<float>(M * H);
  s.branch = DeviceAlloc<float>(M * H);
  s.p_buf = DeviceAlloc<float>(std::size_t{M} * Impl::kInPitch);
  s.conv_out = DeviceAlloc<float>(M * H);
  s.qkv_buf = DeviceAlloc<float>(std::size_t{M} * Impl::kQkvPitch);
  s.q = DeviceAlloc<float>(M * Q);
  s.k = DeviceAlloc<float>(M * 512);
  s.v = DeviceAlloc<float>(M * 512);
  s.qn = DeviceAlloc<float>(M * Q);
  s.kn = DeviceAlloc<float>(M * 512);
  s.gate_buf = DeviceAlloc<float>(M * s.heads);
  s.attn = DeviceAlloc<float>(M * Q);
  s.gu_buf = DeviceAlloc<float>(std::size_t{M} * Impl::kGuPitch);
  s.act = DeviceAlloc<float>(M * I);
  s.half_buf =
      DeviceAlloc<__half>(M * std::max<std::size_t>(I, std::max(H, Q)));
  s.use_runsync = std::getenv("GUFO_D1_NO_RUNSYNC") == nullptr;
  s.prof_fine = std::getenv("GUFO_D1_PROF2") != nullptr;
  s.sync_each = std::getenv("GUFO_NPU_ISOLATE") != nullptr || s.prof_fine ||
                std::getenv("GUFO_D1_SYNC") != nullptr;
  s.ids = PinnedAlloc<std::int32_t>(M);
  s.group_buf = PinnedAlloc<std::int32_t>(M);
  s.seq_lo_h = PinnedAlloc<std::int32_t>(M);
  s.seq_hi_h = PinnedAlloc<std::int32_t>(M);
  s.seq_lo = DeviceAlloc<std::int32_t>(M);
  s.seq_hi = DeviceAlloc<std::int32_t>(M);
  if (!s.x || !s.z || !s.z2 || !s.branch || !s.p_buf || !s.conv_out ||
      !s.qkv_buf || !s.q || !s.k || !s.v || !s.qn || !s.kn || !s.gate_buf ||
      !s.attn || !s.gu_buf || !s.act || !s.half_buf || !s.ids || !s.group_buf ||
      !s.seq_lo_h || !s.seq_hi_h || !s.seq_lo || !s.seq_hi)
    return fail("scratch allocation failed");

  // GGUF tensors are mmap'd file pages the GPU cannot fault on: copy what
  // the kernels read into pinned memory.
  const std::size_t layers = c.layers.size();
  // Tables are filled on the host in pinned staging arrays, then uploaded.
  std::vector<float> h_op(layers * H), h_ffn(layers * H), h_post(layers * H),
      h_post_ffn(layers * H), h_qn(layers * s.hd), h_kn(layers * s.hd),
      h_out(H), h_conv(layers * H * s.taps), h_gate(layers * s.heads * H),
      h_factors(s.hd / 2);
  s.g_op = DeviceAlloc<float>(layers * H);
  s.g_ffn = DeviceAlloc<float>(layers * H);
  s.g_post = DeviceAlloc<float>(layers * H);
  s.g_post_ffn = DeviceAlloc<float>(layers * H);
  s.g_qn = DeviceAlloc<float>(layers * s.hd);
  s.g_kn = DeviceAlloc<float>(layers * s.hd);
  s.g_out = DeviceAlloc<float>(H);
  s.g_conv = DeviceAlloc<float>(layers * H * s.taps);
  s.g_gate = DeviceAlloc<float>(layers * s.heads * H);
  s.g_factors = DeviceAlloc<float>(s.hd / 2);
  if (!s.g_op || !s.g_ffn || !s.g_post || !s.g_post_ffn || !s.g_qn || !s.g_kn ||
      !s.g_out || !s.g_conv || !s.g_gate || !s.g_factors)
    return fail("norm table allocation failed");
  auto copy_f32 = [](const lfm3::Tensor& t, float* dst, std::size_t count) {
    if (t.data)
      std::memcpy(dst, t.data, count * sizeof(float));
  };
  for (std::size_t l = 0; l < layers; ++l) {
    const auto& lw = s.w.layers[l];
    copy_f32(lw.norm, h_op.data() + l * H, H);
    copy_f32(lw.ffn_norm, h_ffn.data() + l * H, H);
    copy_f32(lw.post_norm, h_post.data() + l * H, H);
    copy_f32(lw.post_ffn_norm, h_post_ffn.data() + l * H, H);
    copy_f32(lw.q_norm, h_qn.data() + l * s.hd, s.hd);
    copy_f32(lw.k_norm, h_kn.data() + l * s.hd, s.hd);
    copy_f32(lw.conv, h_conv.data() + l * H * s.taps, H * s.taps);
    if (lw.attn_gate.data)
      for (std::uint32_t r = 0; r < s.heads; ++r)
        lfm3::DecodeRow(
            lw.attn_gate, r,
            std::span<float>(h_gate.data() + (l * s.heads + r) * H, H));
  }
  copy_f32(s.w.output_norm, h_out.data(), H);
  copy_f32(s.w.rope_freqs, h_factors.data(), s.hd / 2);
  hipMemcpy(s.g_op, h_op.data(), h_op.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_ffn, h_ffn.data(), h_ffn.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_post, h_post.data(), h_post.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_post_ffn, h_post_ffn.data(), h_post_ffn.size() * 4,
            hipMemcpyHostToDevice);
  hipMemcpy(s.g_qn, h_qn.data(), h_qn.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_kn, h_kn.data(), h_kn.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_out, h_out.data(), h_out.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_conv, h_conv.data(), h_conv.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_gate, h_gate.data(), h_gate.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_factors, h_factors.data(), h_factors.size() * 4,
            hipMemcpyHostToDevice);
  {
    const std::size_t bytes = s.w.embedding.bytes;
    s.emb_table = DeviceAlloc<std::uint8_t>(bytes);
    if (!s.emb_table || hipMemcpy(s.emb_table, s.w.embedding.data, bytes,
                                  hipMemcpyHostToDevice) != hipSuccess)
      return fail("embedding upload failed");
  }

  // Weights -> block-float NPU buffers, one fused buffer per projection.
  auto encode = [&](const std::vector<const lfm3::Tensor*>& srcs,
                    std::uint32_t k_total, std::uint32_t n_total,
                    std::vector<int>& out_ids) -> bool {
    const std::size_t tb = static_cast<std::size_t>(k_total) * n_total / 8 * 9;
    const int buf = s.npu->NewBuffer(tb, nullptr);
    if (buf < 0)
      return false;
    std::memset(s.npu->Host(buf), 0, tb);
    std::uint32_t col0 = 0;
    for (const auto* t : srcs) {
      const std::size_t bytes = t->bytes;
      void* pin = nullptr;
      if (hipHostMalloc(&pin, bytes) != hipSuccess)
        return false;
      std::memcpy(pin, t->data, bytes);
      rocm::NpuEncodeWeightsQuant(
          pin, static_cast<std::uint32_t>(t->type), k_total, 0, t->rows,
          k_total, 0, k_total,
          static_cast<std::uint8_t*>(s.npu->Device(buf)) +
              (static_cast<std::size_t>(col0) / 128) * k_total * 128 / 8 * 9,
          s.stream);
      hipStreamSynchronize(s.stream);
      hipHostFree(pin);
      col0 += t->rows;
    }
    s.npu->FinalizeWeights(buf);
    if (std::getenv("GUFO_D1_TRACE") && out_ids.empty()) {
      const auto* b = static_cast<const std::uint8_t*>(s.npu->Host(buf));
      std::size_t nz = 0;
      for (std::size_t i = 0; i < tb; ++i)
        nz += b[i] != 0;
      std::fprintf(stderr, "[d1 npu ] weight buf %d bytes %zu nonzero %zu\n",
                   buf, tb, nz);
    }
    out_ids.push_back(buf);
    return true;
  };
  s.w_in.assign(layers, -1);
  s.w_out.assign(layers, -1);
  s.w_qkv.assign(layers, -1);
  s.w_gu.assign(layers, -1);
  s.w_down.assign(layers, -1);
  for (std::size_t l = 0; l < layers; ++l) {
    const auto& lw = s.w.layers[l];
    std::vector<int> id;
    if (c.layers[l] == lfm3::LayerType::kConvolution) {
      if (!encode({&lw.in_proj}, 2048, 6144, id) ||
          !encode({&lw.out_proj}, 2048, 2048, id))
        return fail("conv weight encode failed");
      s.w_in[l] = id[0];
      s.w_out[l] = id[1];
    } else {
      if (!encode({&lw.q, &lw.k, &lw.v}, 2048, 3072, id) ||
          !encode({&lw.attn_out}, 2048, 2048, id))
        return fail("attention weight encode failed");
      s.w_qkv[l] = id[0];
      s.w_out[l] = id[1];
    }
    id.clear();
    if (!encode({&lw.gate, &lw.up}, 2048, 15360, id) ||
        !encode({&lw.down}, 7680, 2048, id))
      return fail("ffn weight encode failed");
    s.w_gu[l] = id[0];
    s.w_down[l] = id[1];
  }
  return std::unique_ptr<D1NpuModel>(new D1NpuModel(std::move(impl)));
}

bool D1NpuModel::Impl::Layers(std::uint32_t n) {
  Impl& s = *this;
  const auto& c = s.w.config;
  const std::uint32_t H = static_cast<std::uint32_t>(s.hidden);
  const std::uint32_t I = static_cast<std::uint32_t>(s.inter);
  const std::uint32_t heads = static_cast<std::uint32_t>(s.heads);
  const std::uint32_t hd = static_cast<std::uint32_t>(s.hd);
  const std::uint32_t kv_heads = 4;
  const std::uint32_t kv_dim = kv_heads * hd;
  const std::uint32_t q_dim = static_cast<std::uint32_t>(s.q_dim);
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  const bool trace = std::getenv("GUFO_D1_TRACE") != nullptr;
  const auto stat = [&](const char* name, std::uint32_t l, const float* p,
                        std::size_t count) {
    if (!trace || l > 1)
      return;
    std::vector<float> h(count);
    hipMemcpyAsync(h.data(), p, count * sizeof(float), hipMemcpyDeviceToHost,
                   s.stream);
    hipStreamSynchronize(s.stream);
    double mean = 0;
    std::size_t zeros = 0, bad = 0;
    for (std::size_t i = 0; i < count; ++i) {
      mean += std::abs(h[i]);
      zeros += h[i] == 0.0F;
      bad += !std::isfinite(h[i]);
    }
    std::fprintf(stderr,
                 "[d1 npu ] L%u %-8s mean|.| %.5f zeros %zu/%zu bad %zu first "
                 "%.4f %.4f\n",
                 l, name, mean / count, zeros, count, bad, h[0], h[1]);
  };
  for (std::uint32_t l = 0; l < c.layers.size(); ++l) {
    stat("x_in", l, s.x, static_cast<std::size_t>(n) * H);
    rocm::RmsNormRows(s.x, s.g_op + static_cast<std::size_t>(l) * H, s.z, n, H,
                      1, c.eps, s.stream);
    stat("z", l, s.z, static_cast<std::size_t>(n) * H);
    if (c.layers[l] == lfm3::LayerType::kConvolution) {
      s.EncodeA(s.z, n, H, s.a_h);
      if (!s.Gemm(s.prog_in, s.a_h, s.w_in[l], s.c_in))
        return false;
      s.Decode(s.c_in, s.p_buf, n, Impl::kInPitch, Impl::kInPitch, 0);
      stat("p", l, s.p_buf, static_cast<std::size_t>(n) * Impl::kInPitch);
      if (s.batch) {
        hipLaunchKernelGGL(D1ConvBatchKernel, dim3((H + 127) / 128, n),
                           dim3(128), 0, s.stream, s.p_buf,
                           s.g_conv + static_cast<std::size_t>(l) * H * s.taps,
                           s.bconv_hist[l], s.conv_out, n, H, s.taps);
      } else {
        if (s.stateful)
          hipMemcpyAsync(
              s.conv_bx[l], s.conv_hist[l],
              static_cast<std::size_t>(s.taps - 1) * H * sizeof(float),
              hipMemcpyDeviceToDevice, s.stream);
        hipLaunchKernelGGL(
            D1ConvKernel, dim3((H + 127) / 128, n), dim3(128), 0, s.stream,
            s.p_buf, s.g_conv + static_cast<std::size_t>(l) * H * s.taps,
            s.seq_lo, s.stateful ? s.conv_hist[l] : nullptr,
            s.stateful ? s.conv_bx[l] : nullptr, s.conv_out, n, H, s.taps);
      }
      s.EncodeA(s.conv_out, n, H, s.a_h);
      if (!s.Gemm(s.prog_h, s.a_h, s.w_out[l], s.c_h))
        return false;
      s.Decode(s.c_h, s.branch, n, H, H, H);
      stat("conv", l, s.conv_out, static_cast<std::size_t>(n) * H);
      stat("branch", l, s.branch, static_cast<std::size_t>(n) * H);
    } else {
      s.EncodeA(s.z, n, H, s.a_h);
      if (!s.Gemm(s.prog_qkv, s.a_h, s.w_qkv[l], s.c_qkv))
        return false;
      s.Decode(s.c_qkv, s.qkv_buf, n, Impl::kQkvPitch, Impl::kQkvPitch, 0);
      hipLaunchKernelGGL(D1SplitQkvKernel, dim3(n), dim3(128), 0, s.stream,
                         s.qkv_buf, s.q, s.k, s.v, n, q_dim, kv_dim,
                         Impl::kQkvPitch);
      hipLaunchKernelGGL(D1GateKernel, dim3(n), dim3(kThreads), 0, s.stream,
                         s.z,
                         s.g_gate + static_cast<std::size_t>(l) * heads * H,
                         s.gate_buf, H, heads);
      rocm::RmsNormRows(s.q, s.g_qn + static_cast<std::size_t>(l) * hd, s.qn,
                        n * heads, hd, 1, c.eps, s.stream);
      rocm::RmsNormRows(s.k, s.g_kn + static_cast<std::size_t>(l) * hd, s.kn,
                        n * kv_heads, hd, 1, c.eps, s.stream);
      const bool swa = c.layers[l] == lfm3::LayerType::kSlidingAttention;
      hipLaunchKernelGGL(
          D1RopeKernel, dim3(n, heads + kv_heads), dim3(hd / 2), 0, s.stream,
          s.qn, s.kn, swa ? static_cast<const float*>(nullptr) : s.g_factors,
          s.seq_lo, s.batch ? s.pos_dev : nullptr, s.stateful ? s.pos0 : 0u, n,
          heads, kv_heads, hd, swa ? c.theta_swa : c.theta);
      if (s.batch) {
        const std::size_t stride =
            static_cast<std::size_t>(s.batch_ctx) * kv_dim;
        hipLaunchKernelGGL(D1StoreKvBatchKernel, dim3(n), dim3(128), 0,
                           s.stream, s.kn, s.v, s.bk_cache[l], s.bv_cache[l],
                           s.pos_dev, stride, kv_dim, n);
        hipLaunchKernelGGL(D1AttnKernel, dim3(n, heads), dim3(hd), 0, s.stream,
                           s.qn, s.bk_cache[l], s.bv_cache[l], s.gate_buf,
                           s.group_buf, s.seq_lo, s.seq_hi, s.attn, n, heads,
                           kv_heads, hd, scale, swa ? c.window : 0u, 1u, 0u,
                           s.pos_dev, stride);
      } else if (s.stateful) {
        // Tiled / split-K attention over the cache plus this chunk, then the
        // chunk joins the cache (writes follow the read, as the kernels need).
        lfm3::gpu::Attention(
            s.qn, s.kn, s.v, s.gate_buf, s.k_cache[l], s.v_cache[l], s.attn,
            static_cast<int>(n), static_cast<int>(heads),
            static_cast<int>(kv_heads), static_cast<int>(hd),
            static_cast<int>(s.pos0), static_cast<int>(s.max_ctx),
            swa ? static_cast<int>(c.window) : 0, s.stream, s.attn_scratch);
        lfm3::gpu::StoreKV(s.kn, s.v, s.k_cache[l], s.v_cache[l],
                           static_cast<int>(n), static_cast<int>(kv_dim),
                           static_cast<int>(s.pos0),
                           static_cast<int>(s.max_ctx), s.stream);
      } else {
        hipLaunchKernelGGL(D1AttnKernel, dim3(n, heads), dim3(hd), 0, s.stream,
                           s.qn, s.kn, s.v, s.gate_buf, s.group_buf, s.seq_lo,
                           s.seq_hi, s.attn, n, heads, kv_heads, hd, scale,
                           swa ? c.window : 0u, 0u, 0u, nullptr, 0);
      }
      s.EncodeA(s.attn, n, q_dim, s.a_h);
      if (!s.Gemm(s.prog_h, s.a_h, s.w_out[l], s.c_h))
        return false;
      s.Decode(s.c_h, s.branch, n, H, H, H);
    }
    rocm::RmsNormRows(s.branch, s.g_post + static_cast<std::size_t>(l) * H,
                      s.z2, n, H, 1, c.eps, s.stream);
    stat("z2", l, s.z2, static_cast<std::size_t>(n) * H);
    s.Add(s.x, s.z2, n);

    rocm::RmsNormRows(s.x, s.g_ffn + static_cast<std::size_t>(l) * H, s.z, n, H,
                      1, c.eps, s.stream);
    s.EncodeA(s.z, n, H, s.a_h);
    if (!s.Gemm(s.prog_gu, s.a_h, s.w_gu[l], s.c_gu))
      return false;
    s.Decode(s.c_gu, s.gu_buf, n, Impl::kGuPitch, Impl::kGuPitch, 0);
    hipLaunchKernelGGL(D1SwigluKernel,
                       dim3(static_cast<std::uint32_t>(
                           (static_cast<std::size_t>(n) * I + 255) / 256)),
                       dim3(256), 0, s.stream, s.gu_buf, s.act, I,
                       Impl::kGuPitch, static_cast<std::size_t>(n) * I);
    s.EncodeA(s.act, n, I, s.a_i);
    if (!s.Gemm(s.prog_down, s.a_i, s.w_down[l], s.c_h))
      return false;
    s.Decode(s.c_h, s.branch, n, H, H, H);
    rocm::RmsNormRows(s.branch, s.g_post_ffn + static_cast<std::size_t>(l) * H,
                      s.z2, n, H, 1, c.eps, s.stream);
    s.Add(s.x, s.z2, n);
    if (trace) {
      std::vector<float> h(static_cast<std::size_t>(n) * H);
      hipMemcpyAsync(h.data(), s.x, h.size() * sizeof(float),
                     hipMemcpyDeviceToHost, s.stream);
      hipStreamSynchronize(s.stream);
      double mean = 0;
      for (float v : h)
        mean += std::abs(v);
      const std::size_t last = static_cast<std::size_t>(n - 1) * H;
      std::fprintf(
          stderr, "[d1 npu ] layer %u mean|x| %.5f last %.4f %.4f %.4f %.4f\n",
          l, mean / h.size(), h[last], h[last + 1], h[last + 2], h[last + 3]);
    }
  }

  return true;
}

bool D1NpuModel::Forward(std::span<const std::uint32_t> tokens,
                         std::span<const std::int32_t> groups,
                         std::span<const std::uint32_t> sequences,
                         std::span<const std::uint32_t> slots,
                         std::span<const std::uint32_t> vocabulary,
                         std::span<float> logits) const {
  auto& s = *impl_;
  const auto t_start = std::chrono::steady_clock::now();
  s.t_encode = s.t_gemm = s.t_half = s.t_memset = s.t_enckernel = s.t_final = 0;
  s.n_gemm = 0;
  const auto& c = s.w.config;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t H = static_cast<std::uint32_t>(s.hidden);
  const std::uint32_t I = static_cast<std::uint32_t>(s.inter);
  const std::uint32_t heads = static_cast<std::uint32_t>(s.heads);
  const std::uint32_t hd = static_cast<std::uint32_t>(s.hd);
  const std::uint32_t kv_heads = 4;
  const std::uint32_t kv_dim = kv_heads * hd;
  const std::uint32_t q_dim = static_cast<std::uint32_t>(s.q_dim);
  const std::uint32_t cols = static_cast<std::uint32_t>(vocabulary.size());
  if (n == 0 || n > s.pass_rows || slots.empty() || cols == 0 ||
      logits.size() != slots.size() * cols ||
      (!groups.empty() && groups.size() != n) || !s.npu->Healthy())
    return false;
  for (auto id : tokens)
    if (id >= c.vocab)
      return false;
  for (auto slot : slots)
    if (slot >= n)
      return false;
  for (auto id : vocabulary)
    if (id >= c.vocab)
      return false;
  for (std::uint32_t l = 0; l < c.layers.size(); ++l)
    if (c.layers[l] != lfm3::LayerType::kConvolution &&
        c.kv_heads[l] != kv_heads)
      return false;

  for (std::uint32_t i = 0; i < n; ++i)
    s.ids[i] = static_cast<std::int32_t>(tokens[i]);
  // Sequences: ascending start offsets (first 0); empty means one sequence.
  {
    std::uint32_t begin = 0;
    const std::size_t count = sequences.empty() ? 1 : sequences.size();
    if (!sequences.empty() && sequences[0] != 0)
      return false;
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint32_t lo = sequences.empty() ? 0 : sequences[i];
      const std::uint32_t hi =
          (!sequences.empty() && i + 1 < count) ? sequences[i + 1] : n;
      if (lo < begin || hi <= lo || hi > n)
        return false;
      for (std::uint32_t t = lo; t < hi; ++t) {
        s.seq_lo_h[t] = static_cast<std::int32_t>(lo);
        s.seq_hi_h[t] = static_cast<std::int32_t>(hi);
      }
      begin = hi;
    }
    if (begin != n)
      return false;
    hipMemcpyAsync(s.seq_lo, s.seq_lo_h, n * sizeof(std::int32_t),
                   hipMemcpyHostToDevice, s.stream);
    hipMemcpyAsync(s.seq_hi, s.seq_hi_h, n * sizeof(std::int32_t),
                   hipMemcpyHostToDevice, s.stream);
  }
  if (groups.empty())
    for (std::uint32_t t = 0; t < n; ++t)
      s.group_buf[t] = -1;
  else
    std::memcpy(s.group_buf, groups.data(), groups.size_bytes());
  const int kind = s.w.embedding.type == GgmlType::kQ6_K   ? 1
                   : s.w.embedding.type == GgmlType::kQ8_0 ? 2
                                                           : 0;
  {
    const std::size_t units =
        static_cast<std::size_t>(n) * (H / (kind == 1 ? 256u : 32u));
    const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
    const dim3 grid(static_cast<std::uint32_t>((units + 255) / 256));
    if (kind == 1)
      hipLaunchKernelGGL((D1EmbedKernel<1>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else if (kind == 2)
      hipLaunchKernelGGL((D1EmbedKernel<2>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else
      hipLaunchKernelGGL((D1EmbedKernel<0>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
  }
  if (!s.Layers(n))
    return false;

  // Answer slots: gather rows, final norm, then dot with the chosen
  // vocabulary entries (the embedding table is tied to the output head).
  const std::uint32_t slot_count = static_cast<std::uint32_t>(slots.size());
  if (!s.sel) {
    s.sel = DeviceAlloc<float>(std::size_t{128} * H);
    s.sel_norm = DeviceAlloc<float>(std::size_t{128} * H);
    if (!s.sel || !s.sel_norm)
      return false;
  }
  if (slot_count > 128)
    return false;
  for (std::uint32_t i = 0; i < slot_count; ++i)
    hipMemcpyAsync(s.sel + static_cast<std::size_t>(i) * H,
                   s.x + static_cast<std::size_t>(slots[i]) * H,
                   H * sizeof(float), hipMemcpyDeviceToDevice, s.stream);
  rocm::RmsNormRows(s.sel, s.g_out, s.sel_norm, slot_count, H, 1, c.eps,
                    s.stream);
  if (static_cast<std::int32_t>(cols) > s.vocab_capacity) {
    if (s.vocab_ids)
      hipHostFree(s.vocab_ids);
    if (s.logit_buf)
      hipFree(s.logit_buf);
    s.vocab_ids = PinnedAlloc<std::int32_t>(cols);
    s.logit_buf = DeviceAlloc<float>(std::size_t{128} * cols);
    if (!s.vocab_ids || !s.logit_buf)
      return false;
    s.vocab_capacity = static_cast<std::int32_t>(cols);
  }
  for (std::uint32_t j = 0; j < cols; ++j)
    s.vocab_ids[j] = static_cast<std::int32_t>(vocabulary[j]);
  const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
  if (kind == 1)
    hipLaunchKernelGGL((D1LogitKernel<1>), dim3(cols, slot_count),
                       dim3(kThreads), 0, s.stream, s.emb_table, s.vocab_ids,
                       s.sel_norm, s.logit_buf, H, row_bytes, cols);
  else if (kind == 2)
    hipLaunchKernelGGL((D1LogitKernel<2>), dim3(cols, slot_count),
                       dim3(kThreads), 0, s.stream, s.emb_table, s.vocab_ids,
                       s.sel_norm, s.logit_buf, H, row_bytes, cols);
  else
    hipLaunchKernelGGL((D1LogitKernel<0>), dim3(cols, slot_count),
                       dim3(kThreads), 0, s.stream, s.emb_table, s.vocab_ids,
                       s.sel_norm, s.logit_buf, H, row_bytes, cols);
  hipMemcpyAsync(logits.data(), s.logit_buf, logits.size_bytes(),
                 hipMemcpyDeviceToHost, s.stream);
  hipStreamSynchronize(s.stream);
  if (std::getenv("GUFO_D1_PROF")) {
    const double total = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_start)
                             .count();
    std::fprintf(stderr,
                 "[d1 prof] n=%u total %.1f ms: encode %.1f gemm %.1f (%d "
                 "calls) other %.1f\n",
                 n, total, s.t_encode, s.t_gemm, s.n_gemm,
                 total - s.t_encode - s.t_gemm);
    if (s.prof_fine)
      std::fprintf(stderr,
                   "[d1 prof] encode parts: tohalf %.1f memset %.1f kernel "
                   "%.1f finalize %.1f\n",
                   s.t_half, s.t_memset, s.t_enckernel, s.t_final);
  }
  return s.npu->Healthy();
}

namespace {
bool s_pos_would_overflow(const D1NpuModel& model, std::size_t more) {
  return model.Position() + more + 1 > model.MaxContext();
}
}  // namespace

bool D1NpuModel::EnableGeneration(std::uint32_t max_context,
                                  std::string* error) {
  auto& s = *impl_;
  if (s.max_ctx != 0)
    return true;
  const auto& c = s.w.config;
  const std::size_t layers = c.layers.size();
  const std::size_t H = s.hidden;
  const std::size_t kv_dim = 4 * s.hd;
  s.k_cache.assign(layers, nullptr);
  s.v_cache.assign(layers, nullptr);
  s.conv_hist.assign(layers, nullptr);
  s.conv_bx.assign(layers, nullptr);
  for (std::size_t l = 0; l < layers; ++l) {
    if (c.layers[l] == lfm3::LayerType::kConvolution) {
      s.conv_hist[l] = DeviceAlloc<float>((s.taps - 1) * H);
      s.conv_bx[l] = DeviceAlloc<float>((s.taps - 1 + M) * H);
      if (!s.conv_hist[l] || !s.conv_bx[l]) {
        if (error)
          *error = "d1 npu: conv state allocation failed";
        return false;
      }
    } else {
      s.k_cache[l] = DeviceAlloc<float>(std::size_t{max_context} * kv_dim);
      s.v_cache[l] = DeviceAlloc<float>(std::size_t{max_context} * kv_dim);
      if (!s.k_cache[l] || !s.v_cache[l]) {
        if (error)
          *error = "d1 npu: K/V cache allocation failed (context too large)";
        return false;
      }
    }
  }
  s.attn_scratch = static_cast<float*>(nullptr);
  {
    const std::size_t bytes = lfm3::gpu::AttentionScratchBytes(
        static_cast<int>(s.heads), static_cast<int>(s.hd),
        static_cast<int>(max_context), static_cast<int>(M));
    void* scratch = nullptr;
    if (hipMalloc(&scratch, bytes) != hipSuccess) {
      if (error)
        *error = "d1 npu: attention scratch allocation failed";
      return false;
    }
    s.attn_scratch = static_cast<float*>(scratch);
  }
  s.logit_all = DeviceAlloc<float>(std::size_t{kMaxRows} * c.vocab);
  s.argmax_dev = DeviceAlloc<std::int32_t>(128);
  s.argmax_host = PinnedAlloc<std::int32_t>(128);
  if (!s.logit_all || !s.argmax_dev || !s.argmax_host) {
    if (error)
      *error = "d1 npu: logit buffer allocation failed";
    return false;
  }
  s.max_ctx = max_context;
  ResetState();
  return true;
}

void D1NpuModel::ResetState() const {
  auto& s = *impl_;
  s.pos = 0;
  for (std::size_t l = 0; l < s.conv_hist.size(); ++l)
    if (s.conv_hist[l])
      hipMemsetAsync(s.conv_hist[l], 0, (s.taps - 1) * s.hidden * sizeof(float),
                     s.stream);
  hipStreamSynchronize(s.stream);
}

std::uint32_t D1NpuModel::Position() const {
  return impl_->pos;
}
std::uint32_t D1NpuModel::PassRows() const {
  return impl_->pass_rows;
}
std::uint32_t D1NpuModel::MaxContext() const {
  return impl_->max_ctx;
}

bool D1NpuModel::Step(std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> next_tokens) const {
  auto& s = *impl_;
  const auto& c = s.w.config;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t want = static_cast<std::uint32_t>(next_tokens.size());
  const std::uint32_t H = static_cast<std::uint32_t>(s.hidden);
  if (s.max_ctx == 0 || n == 0 || n > s.pass_rows || want == 0 || want > n ||
      want > kMaxRows || s.pos + n > s.max_ctx || !s.npu->Healthy())
    return false;
  for (auto id : tokens)
    if (id >= c.vocab)
      return false;
  for (std::uint32_t i = 0; i < n; ++i) {
    s.ids[i] = static_cast<std::int32_t>(tokens[i]);
    s.seq_lo_h[i] = 0;
    s.seq_hi_h[i] = static_cast<std::int32_t>(n);
    s.group_buf[i] = -1;
  }
  hipMemcpyAsync(s.seq_lo, s.seq_lo_h, n * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, s.stream);
  hipMemcpyAsync(s.seq_hi, s.seq_hi_h, n * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, s.stream);
  const int kind = s.w.embedding.type == GgmlType::kQ6_K   ? 1
                   : s.w.embedding.type == GgmlType::kQ8_0 ? 2
                                                           : 0;
  {
    const std::size_t units =
        static_cast<std::size_t>(n) * (H / (kind == 1 ? 256u : 32u));
    const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
    const dim3 grid(static_cast<std::uint32_t>((units + 255) / 256));
    if (kind == 1)
      hipLaunchKernelGGL((D1EmbedKernel<1>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else if (kind == 2)
      hipLaunchKernelGGL((D1EmbedKernel<2>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else
      hipLaunchKernelGGL((D1EmbedKernel<0>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
  }
  const auto step_t0 = std::chrono::steady_clock::now();
  s.stateful = true;
  s.pos0 = s.pos;
  const bool ok = s.Layers(n);
  s.stateful = false;
  if (std::getenv("GUFO_D1_PROF2")) {
    hipStreamSynchronize(s.stream);
    s.step_layers_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - step_t0)
                           .count();
  }
  if (!ok)
    return false;
  s.last_n = n;
  if (!s.sel) {
    s.sel = DeviceAlloc<float>(std::size_t{128} * H);
    s.sel_norm = DeviceAlloc<float>(std::size_t{128} * H);
    if (!s.sel || !s.sel_norm)
      return false;
  }
  hipMemcpyAsync(s.sel, s.x + static_cast<std::size_t>(n - want) * H,
                 static_cast<std::size_t>(want) * H * sizeof(float),
                 hipMemcpyDeviceToDevice, s.stream);
  rocm::RmsNormRows(s.sel, s.g_out, s.sel_norm, want, H, 1, c.eps, s.stream);
  const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
  if (kind == 1)
    hipLaunchKernelGGL((D1LogitAllKernel<1>), dim3(c.vocab), dim3(64), 0,
                       s.stream, s.emb_table, s.sel_norm, s.logit_all, H,
                       row_bytes, c.vocab, want);
  else if (kind == 2)
    hipLaunchKernelGGL((D1LogitAllKernel<2>), dim3(c.vocab), dim3(64), 0,
                       s.stream, s.emb_table, s.sel_norm, s.logit_all, H,
                       row_bytes, c.vocab, want);
  else
    hipLaunchKernelGGL((D1LogitAllKernel<0>), dim3(c.vocab), dim3(64), 0,
                       s.stream, s.emb_table, s.sel_norm, s.logit_all, H,
                       row_bytes, c.vocab, want);
  hipLaunchKernelGGL(D1ArgmaxKernel, dim3(want), dim3(1024), 0, s.stream,
                     s.logit_all, c.vocab, s.argmax_dev);
  hipMemcpyAsync(s.argmax_host, s.argmax_dev, want * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, s.stream);
  hipStreamSynchronize(s.stream);
  for (std::uint32_t i = 0; i < want; ++i)
    next_tokens[i] = static_cast<std::uint32_t>(s.argmax_host[i]);
  if (std::getenv("GUFO_D1_PROF2"))
    std::fprintf(stderr,
                 "[d1 step] n=%u layers %.1f ms, tail(logits+argmax) %.1f ms\n",
                 n, s.step_layers_ms,
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - step_t0)
                         .count() -
                     s.step_layers_ms);
  return s.npu->Healthy();
}

bool D1NpuModel::LastHidden(std::vector<float>* out, std::uint32_t rows) const {
  auto& s = *impl_;
  if (!s.sel_norm || rows == 0 || rows > 128)
    return false;
  out->resize(std::size_t{rows} * s.hidden);
  hipMemcpy(out->data(), s.sel_norm, out->size() * sizeof(float),
            hipMemcpyDeviceToHost);
  return true;
}

void D1NpuModel::Commit(std::uint32_t accepted) const {
  auto& s = *impl_;
  if (accepted > s.last_n)
    accepted = s.last_n;
  for (std::size_t l = 0; l < s.conv_hist.size(); ++l)
    if (s.conv_hist[l])
      hipLaunchKernelGGL(
          D1ConvCommitKernel,
          dim3(static_cast<std::uint32_t>((s.hidden + 127) / 128)), dim3(128),
          0, s.stream, s.conv_bx[l], s.conv_hist[l], accepted,
          static_cast<std::uint32_t>(s.hidden), s.taps);
  hipStreamSynchronize(s.stream);
  s.pos += accepted;
}

bool D1NpuModel::EnableBatch(std::uint32_t streams, std::uint32_t ctx,
                             std::string* error) {
  auto& s = *impl_;
  const auto& c = s.w.config;
  if (s.max_ctx == 0 || streams == 0 || streams > 128 || s.batch_streams != 0) {
    if (error)
      *error = "d1 npu: EnableGeneration first; batch is 1..128 streams, once";
    return false;
  }
  const std::size_t layers = c.layers.size();
  const std::size_t H = s.hidden;
  const std::size_t kv_dim = 4 * s.hd;
  s.bk_cache.assign(layers, nullptr);
  s.bv_cache.assign(layers, nullptr);
  s.bconv_hist.assign(layers, nullptr);
  for (std::size_t l = 0; l < layers; ++l) {
    if (c.layers[l] == lfm3::LayerType::kConvolution) {
      s.bconv_hist[l] =
          DeviceAlloc<float>(std::size_t{streams} * (s.taps - 1) * H);
      if (!s.bconv_hist[l])
        return false;
      hipMemset(s.bconv_hist[l], 0,
                std::size_t{streams} * (s.taps - 1) * H * 4);
    } else {
      s.bk_cache[l] = DeviceAlloc<float>(std::size_t{streams} * ctx * kv_dim);
      s.bv_cache[l] = DeviceAlloc<float>(std::size_t{streams} * ctx * kv_dim);
      if (!s.bk_cache[l] || !s.bv_cache[l]) {
        if (error)
          *error = "d1 npu: batch K/V allocation failed";
        return false;
      }
    }
  }
  s.pos_dev = DeviceAlloc<std::int32_t>(streams);
  s.pos_host = PinnedAlloc<std::int32_t>(streams);
  if (!s.pos_dev || !s.pos_host)
    return false;
  std::memset(s.pos_host, 0, streams * sizeof(std::int32_t));
  s.batch_streams = streams;
  s.batch_ctx = ctx;
  return true;
}

bool D1NpuModel::BatchLoad(std::uint32_t stream, std::uint32_t length) const {
  auto& s = *impl_;
  const auto& c = s.w.config;
  if (s.batch_streams == 0 || stream >= s.batch_streams ||
      length > s.batch_ctx || length > s.pos)
    return false;
  const std::size_t H = s.hidden, kv_dim = 4 * s.hd;
  hipStreamSynchronize(s.stream);
  for (std::size_t l = 0; l < c.layers.size(); ++l) {
    if (c.layers[l] == lfm3::LayerType::kConvolution) {
      hipMemcpyAsync(s.bconv_hist[l] + std::size_t{stream} * (s.taps - 1) * H,
                     s.conv_hist[l], (s.taps - 1) * H * sizeof(float),
                     hipMemcpyDeviceToDevice, s.stream);
    } else {
      hipMemcpyAsync(s.bk_cache[l] + std::size_t{stream} * s.batch_ctx * kv_dim,
                     s.k_cache[l], std::size_t{length} * kv_dim * 4,
                     hipMemcpyDeviceToDevice, s.stream);
      hipMemcpyAsync(s.bv_cache[l] + std::size_t{stream} * s.batch_ctx * kv_dim,
                     s.v_cache[l], std::size_t{length} * kv_dim * 4,
                     hipMemcpyDeviceToDevice, s.stream);
    }
  }
  hipStreamSynchronize(s.stream);
  s.pos_host[stream] = static_cast<std::int32_t>(length);
  return true;
}

bool D1NpuModel::BatchStep(std::span<const std::uint32_t> tokens,
                           std::span<std::uint32_t> next) const {
  auto& s = *impl_;
  const auto& c = s.w.config;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t H = static_cast<std::uint32_t>(s.hidden);
  if (s.batch_streams == 0 || n == 0 || n > s.batch_streams ||
      n > s.pass_rows || next.size() != n || !s.npu->Healthy())
    return false;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (tokens[i] >= c.vocab ||
        s.pos_host[i] + 1 >= static_cast<std::int32_t>(s.batch_ctx))
      return false;
    s.ids[i] = static_cast<std::int32_t>(tokens[i]);
    s.seq_lo_h[i] = 0;
    s.seq_hi_h[i] = 1;
    s.group_buf[i] = -1;
  }
  hipMemcpyAsync(s.pos_dev, s.pos_host, n * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, s.stream);
  hipMemcpyAsync(s.seq_lo, s.seq_lo_h, n * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, s.stream);
  hipMemcpyAsync(s.seq_hi, s.seq_hi_h, n * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, s.stream);
  const int kind = s.w.embedding.type == GgmlType::kQ6_K   ? 1
                   : s.w.embedding.type == GgmlType::kQ8_0 ? 2
                                                           : 0;
  {
    const std::size_t units =
        static_cast<std::size_t>(n) * (H / (kind == 1 ? 256u : 32u));
    const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
    const dim3 grid(static_cast<std::uint32_t>((units + 255) / 256));
    if (kind == 1)
      hipLaunchKernelGGL((D1EmbedKernel<1>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else if (kind == 2)
      hipLaunchKernelGGL((D1EmbedKernel<2>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else
      hipLaunchKernelGGL((D1EmbedKernel<0>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
  }
  s.stateful = true;
  s.batch = true;
  const bool ok = s.Layers(n);
  s.stateful = false;
  s.batch = false;
  if (!ok)
    return false;
  if (!s.sel) {
    s.sel = DeviceAlloc<float>(std::size_t{128} * H);
    s.sel_norm = DeviceAlloc<float>(std::size_t{128} * H);
    if (!s.sel || !s.sel_norm)
      return false;
  }
  rocm::RmsNormRows(s.x, s.g_out, s.sel_norm, n, H, 1, c.eps, s.stream);
  const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
  for (std::uint32_t r0 = 0; r0 < n; r0 += kMaxRows) {
    const std::uint32_t rows = std::min<std::uint32_t>(kMaxRows, n - r0);
    const float* hv = s.sel_norm + static_cast<std::size_t>(r0) * H;
    if (kind == 1)
      hipLaunchKernelGGL((D1LogitAllKernel<1>), dim3(c.vocab), dim3(64), 0,
                         s.stream, s.emb_table, hv, s.logit_all, H, row_bytes,
                         c.vocab, rows);
    else if (kind == 2)
      hipLaunchKernelGGL((D1LogitAllKernel<2>), dim3(c.vocab), dim3(64), 0,
                         s.stream, s.emb_table, hv, s.logit_all, H, row_bytes,
                         c.vocab, rows);
    else
      hipLaunchKernelGGL((D1LogitAllKernel<0>), dim3(c.vocab), dim3(64), 0,
                         s.stream, s.emb_table, hv, s.logit_all, H, row_bytes,
                         c.vocab, rows);
    hipLaunchKernelGGL(D1ArgmaxKernel, dim3(rows), dim3(1024), 0, s.stream,
                       s.logit_all, c.vocab, s.argmax_dev + r0);
  }
  hipMemcpyAsync(s.argmax_host, s.argmax_dev, n * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, s.stream);
  hipStreamSynchronize(s.stream);
  for (std::uint32_t i = 0; i < n; ++i) {
    next[i] = static_cast<std::uint32_t>(s.argmax_host[i]);
    s.pos_host[i] += 1;
  }
  return s.npu->Healthy();
}

bool D1NpuModel::CaptureHidden(std::span<const std::uint32_t> tokens,
                               std::vector<std::uint16_t>* hidden) const {
  auto& s = *impl_;
  const auto& c = s.w.config;
  const std::uint32_t H = static_cast<std::uint32_t>(s.hidden);
  if (s.max_ctx == 0 || hidden == nullptr || tokens.empty() ||
      tokens.size() > s.max_ctx)
    return false;
  ResetState();
  hidden->assign(tokens.size() * static_cast<std::size_t>(H), 0);
  __half* host_half = nullptr;
  if (hipHostMalloc(reinterpret_cast<void**>(&host_half),
                    std::size_t{s.pass_rows} * H * sizeof(__half)) !=
      hipSuccess)
    return false;
  for (std::size_t at = 0; at < tokens.size();) {
    const std::uint32_t n = static_cast<std::uint32_t>(
        std::min<std::size_t>(s.pass_rows, tokens.size() - at));
    for (std::uint32_t i = 0; i < n; ++i) {
      s.ids[i] = static_cast<std::int32_t>(tokens[at + i]);
      s.seq_lo_h[i] = 0;
      s.seq_hi_h[i] = static_cast<std::int32_t>(n);
      s.group_buf[i] = -1;
    }
    hipMemcpyAsync(s.seq_lo, s.seq_lo_h, n * sizeof(std::int32_t),
                   hipMemcpyHostToDevice, s.stream);
    hipMemcpyAsync(s.seq_hi, s.seq_hi_h, n * sizeof(std::int32_t),
                   hipMemcpyHostToDevice, s.stream);
    const int kind = s.w.embedding.type == GgmlType::kQ6_K   ? 1
                     : s.w.embedding.type == GgmlType::kQ8_0 ? 2
                                                             : 0;
    const std::size_t units =
        static_cast<std::size_t>(n) * (H / (kind == 1 ? 256u : 32u));
    const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
    const dim3 grid(static_cast<std::uint32_t>((units + 255) / 256));
    if (kind == 1)
      hipLaunchKernelGGL((D1EmbedKernel<1>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else if (kind == 2)
      hipLaunchKernelGGL((D1EmbedKernel<2>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    else
      hipLaunchKernelGGL((D1EmbedKernel<0>), grid, dim3(256), 0, s.stream,
                         s.emb_table, s.ids, s.x, n, H, row_bytes, c.vocab,
                         units);
    s.stateful = true;
    s.pos0 = s.pos;
    const bool ok = s.Layers(n);
    s.stateful = false;
    if (!ok) {
      hipHostFree(host_half);
      return false;
    }
    s.last_n = n;
    rocm::RmsNormRows(s.x, s.g_out, s.z, n, H, 1, c.eps, s.stream);
    hipLaunchKernelGGL(
        D1ToHalfKernel,
        dim3(static_cast<std::uint32_t>((std::size_t{n} * H + 255) / 256)),
        dim3(256), 0, s.stream, s.z, s.half_buf, std::size_t{n} * H);
    hipMemcpyAsync(host_half, s.half_buf, std::size_t{n} * H * sizeof(__half),
                   hipMemcpyDeviceToHost, s.stream);
    hipStreamSynchronize(s.stream);
    std::memcpy(hidden->data() + at * H, host_half,
                std::size_t{n} * H * sizeof(__half));
    Commit(n);
    at += n;
  }
  hipHostFree(host_half);
  return s.npu->Healthy();
}

bool D1NpuModel::LoadDrafter(const std::string& path, std::string* error) {
  auto& s = *impl_;
  std::ifstream file(path, std::ios::binary);
  std::uint32_t k = 0, hidden = 0;
  file.read(reinterpret_cast<char*>(&k), 4);
  file.read(reinterpret_cast<char*>(&hidden), 4);
  if (!file || k == 0 || k > kMaxRows || hidden != s.hidden) {
    if (error)
      *error = "d1 npu: bad drafter file";
    return false;
  }
  std::vector<__half> host(std::size_t{k} * hidden * hidden);
  file.read(reinterpret_cast<char*>(host.data()), host.size() * sizeof(__half));
  if (!file) {
    if (error)
      *error = "d1 npu: truncated drafter file";
    return false;
  }
  s.draft_w = DeviceAlloc<__half>(host.size());
  s.draft_z = DeviceAlloc<float>(std::size_t{k} * hidden);
  if (!s.draft_w || !s.draft_z ||
      hipMemcpy(s.draft_w, host.data(), host.size() * sizeof(__half),
                hipMemcpyHostToDevice) != hipSuccess) {
    if (error)
      *error = "d1 npu: drafter upload failed";
    return false;
  }
  s.draft_k = k;
  return true;
}

std::uint32_t D1NpuModel::DrafterHeads() const {
  return impl_->draft_k;
}

bool D1NpuModel::Draft(std::uint32_t row,
                       std::vector<std::uint32_t>* drafts) const {
  auto& s = *impl_;
  const auto& c = s.w.config;
  const std::uint32_t H = static_cast<std::uint32_t>(s.hidden);
  if (s.draft_k == 0 || drafts == nullptr || !s.sel_norm)
    return false;
  const int kind = s.w.embedding.type == GgmlType::kQ6_K   ? 1
                   : s.w.embedding.type == GgmlType::kQ8_0 ? 2
                                                           : 0;
  hipLaunchKernelGGL(
      D1DraftHeadKernel, dim3(H, s.draft_k), dim3(kThreads), 0, s.stream,
      s.draft_w, s.sel_norm + static_cast<std::size_t>(row) * H, s.draft_z, H);
  const std::uint64_t row_bytes = RowBytes(s.w.embedding.type, H);
  if (kind == 1)
    hipLaunchKernelGGL((D1LogitAllKernel<1>), dim3(c.vocab), dim3(64), 0,
                       s.stream, s.emb_table, s.draft_z, s.logit_all, H,
                       row_bytes, c.vocab, s.draft_k);
  else if (kind == 2)
    hipLaunchKernelGGL((D1LogitAllKernel<2>), dim3(c.vocab), dim3(64), 0,
                       s.stream, s.emb_table, s.draft_z, s.logit_all, H,
                       row_bytes, c.vocab, s.draft_k);
  else
    hipLaunchKernelGGL((D1LogitAllKernel<0>), dim3(c.vocab), dim3(64), 0,
                       s.stream, s.emb_table, s.draft_z, s.logit_all, H,
                       row_bytes, c.vocab, s.draft_k);
  hipLaunchKernelGGL(D1ArgmaxKernel, dim3(s.draft_k), dim3(1024), 0, s.stream,
                     s.logit_all, c.vocab, s.argmax_dev);
  hipMemcpyAsync(s.argmax_host, s.argmax_dev, s.draft_k * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, s.stream);
  hipStreamSynchronize(s.stream);
  drafts->assign(s.argmax_host, s.argmax_host + s.draft_k);
  return true;
}

bool D1NpuModel::Generate(std::span<const std::uint32_t> prompt,
                          const GenerateOptions& options,
                          std::vector<std::uint32_t>* output,
                          GenerateStats* stats) const {
  if (prompt.empty() || output == nullptr)
    return false;
  const auto start = std::chrono::steady_clock::now();
  ResetState();
  GenerateStats st;
  st.prompt_tokens = static_cast<std::uint32_t>(prompt.size());
  std::vector<std::uint32_t> history(prompt.begin(), prompt.end());
  std::uint32_t pending = 0;
  std::vector<std::uint32_t> head_draft;
  for (std::size_t at = 0; at < prompt.size();) {
    const std::size_t take =
        std::min<std::size_t>(PassRows(), prompt.size() - at);
    std::uint32_t next = 0;
    if (!Step(prompt.subspan(at, take), std::span(&next, 1)))
      return false;
    Commit(static_cast<std::uint32_t>(take));
    pending = next;
    head_draft.clear();
    if (options.use_heads && DrafterHeads() > 0)
      (void)Draft(0, &head_draft);
    at += take;
    ++st.passes;
  }
  const auto decode_start = std::chrono::steady_clock::now();
  st.prefill_ms =
      std::chrono::duration<double, std::milli>(decode_start - start).count();
  const auto is_stop = [&](std::uint32_t id) {
    return std::find(options.stop.begin(), options.stop.end(), id) !=
           options.stop.end();
  };
  while (output->size() < options.max_new) {
    // pending is the model's next token, not yet in the state.
    output->push_back(pending);
    history.push_back(pending);
    if (is_stop(pending) || output->size() >= options.max_new)
      break;
    // Prompt lookup: continue the most recent earlier occurrence of the tail.
    std::vector<std::uint32_t> draft;
    if (options.draft_max > 0 && history.size() > options.ngram) {
      const std::size_t tail = history.size() - options.ngram;
      for (std::size_t from = tail; from-- > 0;) {
        if (std::equal(history.begin() + from,
                       history.begin() + from + options.ngram,
                       history.begin() + tail)) {
          for (std::size_t k = from + options.ngram;
               k < tail && draft.size() < options.draft_max; ++k)
            draft.push_back(history[k]);
          break;
        }
      }
    }
    if (draft.size() < 2 && !head_draft.empty()) {
      // No useful prompt match: use the drafter heads' chain.
      draft.assign(
          head_draft.begin(),
          head_draft.begin() +
              std::min<std::size_t>(head_draft.size(), options.draft_max));
    }
    if (const char* force = std::getenv("GUFO_D1_FORCE_DRAFT"))
      while (draft.size() < static_cast<std::size_t>(std::atoi(force)))
        draft.push_back(
            1);  // junk drafts: verification width without acceptance
    if (s_pos_would_overflow(*this, 1 + draft.size()))
      break;
    std::vector<std::uint32_t> tokens;
    tokens.push_back(pending);
    tokens.insert(tokens.end(), draft.begin(), draft.end());
    std::vector<std::uint32_t> next(tokens.size());
    if (!Step(tokens, next))
      return false;
    ++st.passes;
    if (std::getenv("GUFO_D1_DRAFT_TRACE") && st.passes < 14) {
      std::fprintf(stderr, "[draft] pending %u | draft", pending);
      for (auto d : draft)
        std::fprintf(stderr, " %u", d);
      std::fprintf(stderr, " | next");
      for (auto t : next)
        std::fprintf(stderr, " %u", t);
      std::fprintf(stderr, "\n");
    }
    std::size_t good = 0;
    while (good < draft.size() && draft[good] == next[good])
      ++good;
    Commit(static_cast<std::uint32_t>(1 + good));
    st.drafted += static_cast<std::uint32_t>(draft.size());
    st.accepted += static_cast<std::uint32_t>(good);
    bool stopped = false;
    for (std::size_t i = 0; i < good && output->size() < options.max_new; ++i) {
      output->push_back(draft[i]);
      history.push_back(draft[i]);
      if (is_stop(draft[i])) {
        stopped = true;
        break;
      }
    }
    if (stopped)
      break;
    pending = next[good];
    head_draft.clear();
    if (options.use_heads && DrafterHeads() > 0)
      (void)Draft(static_cast<std::uint32_t>(good), &head_draft);
  }
  st.new_tokens = static_cast<std::uint32_t>(output->size());
  st.decode_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - decode_start)
                     .count();
  if (stats)
    *stats = st;
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
