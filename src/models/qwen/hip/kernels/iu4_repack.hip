#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

#include "src/models/qwen/hip/ops/iu4.hpp"

namespace gufo::hip {
namespace {

// Engine head e of a GDN projection holds artifact head (e % 16) * 3 + e / 16
// (the artifact keeps value heads grouped by key head, the engine tiled).
__device__ __forceinline__ std::uint32_t SourceRow(std::uint32_t row,
                                                   std::uint32_t perm_start,
                                                   std::uint32_t perm_chunk) {
  if (perm_chunk == 0 || row < perm_start)
    return row;
  const std::uint32_t rr = row - perm_start;
  const std::uint32_t c = rr / perm_chunk, w = rr % perm_chunk;
  return perm_start + ((c % 16) * 3 + c / 16) * perm_chunk + w;
}

__global__ void Iu4RepackKernel(const std::uint8_t* __restrict__ blocks,
                                std::uint8_t* __restrict__ tiles,
                                std::uint16_t* __restrict__ scales,
                                unsigned long long* __restrict__ bad,
                                std::uint32_t rows, std::uint32_t padded_rows,
                                std::uint32_t groups, std::uint32_t perm_start,
                                std::uint32_t perm_chunk) {
  const std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = (std::size_t)groups * rows * 4;
  if (idx >= total)
    return;
  const std::uint32_t q = idx & 3;
  const std::size_t er = idx >> 2;
  const std::uint32_t row = er % rows;
  const std::uint32_t e = er / rows;
  const std::uint32_t src_row = SourceRow(row, perm_start, perm_chunk);
  const std::uint8_t* src =
      blocks +
      ((std::size_t)src_row * (groups >> 1) + (e >> 1)) * kIu4BlockBytes;
  const std::uint32_t half = e & 1;
  if (q == 0) {
    std::uint16_t sc, zp;
    __builtin_memcpy(&sc, src + 4 * half, 2);
    __builtin_memcpy(&zp, src + 4 * half + 2, 2);
    scales[(std::size_t)e * padded_rows + row] = sc;
    const float fs = __half2float(__ushort_as_half(sc));
    const float fz = __half2float(__ushort_as_half(zp));
    if (fabsf(fz + 8.0f * fs) > 1e-3f * fabsf(8.0f * fs) + 1e-12f)
      atomicAdd(bad, 1ull);
  }
  uint4 v;
  __builtin_memcpy(&v, src + 8 + 64 * half + 16 * q, 16);
  v.x ^= 0x88888888u;
  v.y ^= 0x88888888u;
  v.z ^= 0x88888888u;
  v.w ^= 0x88888888u;
  const std::size_t dst =
      (((std::size_t)e * (padded_rows >> 4) + (row >> 4)) * 4 + q) * 256 +
      (row & 15) * 16;
  __builtin_memcpy(tiles + dst, &v, 16);
}

// Lane xor within a row of 16 as a DPP operand (strides 1 to 8); stride 16 and
// up cross rows and go through the generic shuffle.
template<int kMask>
__device__ __forceinline__ float XorLanes(float v) {
  if constexpr (kMask < 16)
    return __int_as_float(__builtin_amdgcn_mov_dpp(
        __float_as_int(v), 0x160 | kMask, 0xF, 0xF, false));
  else
    return __shfl_xor(v, kMask);
}

__device__ __forceinline__ float Reduce16Max(float v) {
  v = fmaxf(v, XorLanes<8>(v));
  v = fmaxf(v, XorLanes<4>(v));
  v = fmaxf(v, XorLanes<2>(v));
  return fmaxf(v, XorLanes<1>(v));
}

// Rotates the 8 values of this lane (already divided by awq, not yet multiplied
// by the pre-butterfly signs) and quantizes the 256-wide group held by the
// wave: strides 1, 2, 4 are register butterflies, strides 8 to 128 are
// lane-xor butterflies; each 128 half gets one scale.
__device__ __forceinline__ void RotateQuantizeStore(
    float (&v)[8], bool live, const float* __restrict__ signs,
    std::uint32_t lane, std::uint32_t group256, std::uint32_t token,
    std::uint32_t padded_n, std::uint32_t e_groups,
    std::uint8_t* __restrict__ tiles, float* __restrict__ scales) {
  if (live) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] *= signs[lane * 8 + i];
#pragma unroll
    for (int st = 1; st < 8; st <<= 1)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if ((i & st) == 0) {
          const float a = v[i], b = v[i + st];
          v[i] = a + b;
          v[i + st] = a - b;
        }
    const auto stage = [&](auto mask_tag) {
      constexpr int m = decltype(mask_tag)::value;
      const bool hi = (lane & m) != 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float o = XorLanes<m>(v[i]);
        v[i] = hi ? (o - v[i]) : (v[i] + o);
      }
    };
    stage(std::integral_constant<int, 1>{});
    stage(std::integral_constant<int, 2>{});
    stage(std::integral_constant<int, 4>{});
    stage(std::integral_constant<int, 8>{});
    stage(std::integral_constant<int, 16>{});
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] *= 0.0625f * signs[256 + lane * 8 + i];
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = 0.0f;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  amax = Reduce16Max(amax);
  const float scale = amax / 7.0f;
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  std::uint32_t word = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = (int)fminf(fmaxf(rintf(v[i] * inv), -8.0f), 7.0f);
    word |= (std::uint32_t)(q & 0xF) << (4 * i);
  }
  const std::uint32_t e = group256 * 2 + (lane >> 4);
  const std::uint32_t l16 = lane & 15;
  const std::uint32_t chunk = l16 >> 1, j = l16 & 1;
  const std::size_t dst = ((std::size_t)(token >> 4) * e_groups + e) * 1024 +
                          ((chunk & 1) * 16 + (token & 15)) * 32 +
                          (chunk >> 1) * 8 + 4 * j;
  *reinterpret_cast<std::uint32_t*>(tiles + dst) = word;
  if (l16 == 0)
    scales[(std::size_t)e * padded_n + token] = scale;
}

// One workgroup per token row, one wave per 256-wide group of the row: lane l
// owns elements 8l..8l+7.
__global__ void __launch_bounds__(1024) Iu4NormQuantizeKernel(
    const float* __restrict__ x, const float* __restrict__ norm_weight,
    const float* __restrict__ awq, const float* __restrict__ signs, float eps,
    std::uint32_t n, std::uint32_t padded_n, std::uint32_t k,
    std::uint8_t* __restrict__ tiles, float* __restrict__ scales,
    __half* __restrict__ half_out) {
  __shared__ float partial[32];
  const std::uint32_t token = blockIdx.x;
  const std::uint32_t wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t groups256 = k >> 8, e_groups = k >> 7;
  const bool live = token < n;
  float v[8];
  float sumsq = 0.0f;
  if (live) {
    const std::size_t base = (std::size_t)token * k + wave * 256 + lane * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(x + base);
    const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
    v[0] = x0.x;
    v[1] = x0.y;
    v[2] = x0.z;
    v[3] = x0.w;
    v[4] = x1.x;
    v[5] = x1.y;
    v[6] = x1.z;
    v[7] = x1.w;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sumsq += v[i] * v[i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    sumsq += __shfl_xor(sumsq, o);
  if (lane == 0)
    partial[wave] = sumsq;
  __syncthreads();
  float total = 0.0f;
  for (std::uint32_t w = 0; w < groups256; ++w)
    total += partial[w];
  const float inv_rms = rsqrtf(total / (float)k + eps);
  if (live) {
    const std::size_t kk = (std::size_t)wave * 256 + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float w = v[i] * inv_rms * norm_weight[kk + i];
      if (half_out != nullptr)
        half_out[(std::size_t)token * k + kk + i] = __float2half(w);
      if (awq != nullptr)
        w /= awq[kk + i];
      v[i] = w;
    }
  }
  RotateQuantizeStore(v, live, signs, lane, wave, token, padded_n, k >> 7,
                      tiles, scales);
}

// Value heads of the engine's GDN output are tiled (engine head e belongs to
// key head e % 16), the artifact groups them by key head: artifact head h reads
// engine head (h % 3) * 16 + h / 3.
__device__ __forceinline__ std::uint32_t EngineHead(std::uint32_t h) {
  return (h % 3) * 16 + h / 3;
}

// One wave per (token, 256-wide group): quantizes an fp32 activation row that
// is already normalized (attention output, GDN output, SwiGLU product).
__global__ void __launch_bounds__(256) Iu4QuantizeKernel(
    const float* __restrict__ x, const float* __restrict__ awq,
    const float* __restrict__ signs, std::uint32_t n, std::uint32_t padded_n,
    std::uint32_t k, std::uint32_t remap_heads,
    std::uint8_t* __restrict__ tiles, float* __restrict__ scales) {
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t gid = blockIdx.x * 8 + (threadIdx.x >> 5);
  const std::uint32_t groups256 = k >> 8;
  if (gid >= padded_n * groups256)
    return;
  const std::uint32_t token = gid / groups256, g = gid % groups256;
  const bool live = token < n;
  float v[8];
  if (live) {
    const std::uint32_t pos = g * 256 + lane * 8;  // artifact-order position
    std::uint32_t src = pos;
    if (remap_heads != 0)
      src = EngineHead(pos >> 7) * 128 + (pos & 127);
    const float4 x0 =
        *reinterpret_cast<const float4*>(x + (std::size_t)token * k + src);
    const float4 x1 =
        *reinterpret_cast<const float4*>(x + (std::size_t)token * k + src + 4);
    v[0] = x0.x;
    v[1] = x0.y;
    v[2] = x0.z;
    v[3] = x0.w;
    v[4] = x1.x;
    v[5] = x1.y;
    v[6] = x1.z;
    v[7] = x1.w;
    if (awq != nullptr) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        v[i] /= awq[pos + i];
    }
  }
  RotateQuantizeStore(v, live, signs, lane, g, token, padded_n, k >> 7, tiles,
                      scales);
}

// Converts the leading `columns` FP16 columns (from the NPU) of each row to
// FP32.
__global__ void Iu4HalfColumnsToF32Kernel(float* __restrict__ f32,
                                          const __half* __restrict__ f16,
                                          std::size_t rows, std::uint32_t ld,
                                          std::uint32_t columns) {
  const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * columns)
    return;
  const std::size_t r = i / columns, c = i % columns;
  f32[r * ld + c] = __half2float(f16[r * ld + c]);
}

// Converts the columns from `first` on of each row from FP32 to FP16.
__global__ void Iu4F32ColumnsToHalfKernel(const float* __restrict__ f32,
                                          __half* __restrict__ f16,
                                          std::size_t rows, std::uint32_t ld,
                                          std::uint32_t first) {
  const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t width = ld - first;
  if (i >= rows * width)
    return;
  const std::size_t r = i / width, c = first + i % width;
  f16[r * ld + c] = __float2half(f32[r * ld + c]);
}

__device__ __forceinline__ float Reduce16Sum(float v) {
  v += XorLanes<8>(v);
  v += XorLanes<4>(v);
  v += XorLanes<2>(v);
  return v + XorLanes<1>(v);
}

// SwiGLU product silu(gate) * up fused into the quantizer. Columns below
// npu_columns come from the NPU: as FP16 when the stream source is empty, else
// decoded straight from the block-float output streams.
__device__ __forceinline__ float SiluOf(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void __launch_bounds__(256) Iu4SwiGluQuantizeKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    const __half* __restrict__ npu_half, Iu4NpuSwigluSource source,
    std::uint32_t npu_columns, const float* __restrict__ awq,
    const float* __restrict__ signs, std::uint32_t n, std::uint32_t padded_n,
    std::uint32_t k, std::uint8_t* __restrict__ tiles,
    float* __restrict__ scales) {
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t gid = blockIdx.x * 8 + (threadIdx.x >> 5);
  const std::uint32_t groups256 = k >> 8;
  if (gid >= padded_n * groups256)
    return;
  const std::uint32_t token = gid / groups256, g = gid % groups256;
  const bool live = token < n;
  float v[8];
  if (live) {
    const std::uint32_t pos = g * 256 + lane * 8;
    const std::size_t base = (std::size_t)token * k + pos;
    if (pos < npu_columns && source.slices != 0) {
      // Stream layout: tile (512 token rows x 128 columns) of 8192 groups, a
      // group being 8 consecutive columns of one token (9 bytes).
      const std::uint32_t slice = pos >> 10, c = pos & 1023;
      const std::uint32_t tile_row = token >> 9, r_in = token & 511;
      const std::uint32_t tile_col = c >> 7, c_in = c & 127;
      const std::uint32_t gi = (r_in & 7) | (((c_in >> 3) & 1) << 3) |
                               (((r_in >> 3) & 1) << 4) | ((c_in >> 4) << 5) |
                               ((r_in >> 4) << 8);
      const std::size_t group =
          ((std::size_t)tile_row * 8 + tile_col) * 8192 + gi;
      const std::uint8_t* pg =
          static_cast<const std::uint8_t*>(source.gate[slice]) + group * 9;
      const std::uint8_t* pu =
          static_cast<const std::uint8_t*>(source.up[slice]) + group * 9;
      const float mg = ldexpf(1.0f, static_cast<int>(pg[0]) - 127) / 64.0f;
      const float mu = ldexpf(1.0f, static_cast<int>(pu[0]) - 127) / 64.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        v[i] = SiluOf(static_cast<float>(static_cast<std::int8_t>(pg[1 + i])) *
                      mg) *
               (static_cast<float>(static_cast<std::int8_t>(pu[1 + i])) * mu);
    } else if (pos < npu_columns) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        v[i] = __half2float(npu_half[base + i]);
    } else {
      const float4 g0 = *reinterpret_cast<const float4*>(gate + base);
      const float4 g1 = *reinterpret_cast<const float4*>(gate + base + 4);
      const float4 u0 = *reinterpret_cast<const float4*>(up + base);
      const float4 u1 = *reinterpret_cast<const float4*>(up + base + 4);
      const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      const float uv[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i)
        v[i] = SiluOf(gv[i]) * uv[i];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] /= awq[pos + i];
  }
  RotateQuantizeStore(v, live, signs, lane, g, token, padded_n, k >> 7, tiles,
                      scales);
}

// Gated DeltaNet output: per (token, head) RMSNorm over the 128 value dims with
// the head-shared ssm_norm weight, times silu(z), fused into the quantizer of
// the out projection. A wave handles one 256 group of the artifact-order row,
// two heads; artifact head h reads engine head (h % 3) * 16 + h / 3.
__global__ void __launch_bounds__(256) Iu4PostNormGateQuantizeKernel(
    const float* __restrict__ raw, const float* __restrict__ ssm_norm,
    const float* __restrict__ gate, const float* __restrict__ awq,
    const float* __restrict__ signs, std::uint32_t n, std::uint32_t padded_n,
    std::uint32_t inner, std::uint8_t* __restrict__ tiles,
    float* __restrict__ scales) {
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t gid = blockIdx.x * 8 + (threadIdx.x >> 5);
  const std::uint32_t groups256 = inner >> 8;
  if (gid >= padded_n * groups256)
    return;
  const std::uint32_t token = gid / groups256, g = gid % groups256;
  const bool live = token < n;
  float v[8];
  if (live) {
    const std::uint32_t pos = g * 256 + lane * 8;  // artifact order
    const std::uint32_t head = pos >> 7, within = pos & 127;
    const std::size_t src =
        (std::size_t)token * inner + EngineHead(head) * 128 + within;
    const float4 x0 = *reinterpret_cast<const float4*>(raw + src);
    const float4 x1 = *reinterpret_cast<const float4*>(raw + src + 4);
    const float4 z0 = *reinterpret_cast<const float4*>(gate + src);
    const float4 z1 = *reinterpret_cast<const float4*>(gate + src + 4);
    const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const float zv[8] = {z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w};
    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sq += xv[i] * xv[i];
    sq = Reduce16Sum(sq);  // the 16 lanes of this head
    const float inv_rms = rsqrtf(sq / 128.0f + 1e-6f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float silu = zv[i] / (1.0f + expf(-zv[i]));
      v[i] = xv[i] * inv_rms * ssm_norm[within + i] * silu / awq[pos + i];
    }
  }
  RotateQuantizeStore(v, live, signs, lane, g, token, padded_n, inner >> 7,
                      tiles, scales);
}

}  // namespace

void LaunchIu4HalfColumnsToF32(float* act_f32, const void* act_f16,
                               std::size_t rows, std::size_t ld,
                               std::size_t columns, hipStream_t stream) {
  const std::size_t total = rows * columns;
  Iu4HalfColumnsToF32Kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      act_f32, static_cast<const __half*>(act_f16), rows,
      static_cast<std::uint32_t>(ld), static_cast<std::uint32_t>(columns));
}

void LaunchIu4F32ColumnsToHalf(const float* act_f32, void* act_f16,
                               std::size_t rows, std::size_t ld,
                               std::size_t first, hipStream_t stream) {
  const std::size_t total = rows * (ld - first);
  Iu4F32ColumnsToHalfKernel<<<(total + 255) / 256, 256, 0, stream>>>(
      act_f32, static_cast<__half*>(act_f16), rows,
      static_cast<std::uint32_t>(ld), static_cast<std::uint32_t>(first));
}

void FillIu4RotationSigns(float* out) {
  const std::uint32_t seeds[2] = {42u, 1042u};
  for (int s = 0; s < 2; ++s) {
    std::uint32_t state = seeds[s];
    for (int i = 0; i < 256; ++i) {
      state = (state * 1103515245u + 12345u) & 0x7fffffffu;
      out[256 * s + i] = ((state >> 16) & 1) ? 1.0f : -1.0f;
    }
  }
}

void LaunchIu4NormQuantize(const float* x, const float* norm_weight,
                           const float* awq, const float* signs, float eps,
                           std::size_t n, std::size_t padded_n, std::size_t k,
                           std::uint8_t* tiles, float* scales,
                           hipStream_t stream, void* half_out) {
  Iu4NormQuantizeKernel<<<static_cast<unsigned>(padded_n),
                          static_cast<unsigned>(32 * (k >> 8)), 0, stream>>>(
      x, norm_weight, awq, signs, eps, static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(padded_n), static_cast<std::uint32_t>(k),
      tiles, scales, static_cast<__half*>(half_out));
}

void LaunchIu4Quantize(const float* x, const float* awq, const float* signs,
                       bool remap_gdn_heads, std::size_t n,
                       std::size_t padded_n, std::size_t k, std::uint8_t* tiles,
                       float* scales, hipStream_t stream) {
  const std::size_t waves = padded_n * (k >> 8);
  Iu4QuantizeKernel<<<static_cast<unsigned>((waves + 7) / 8), 256, 0, stream>>>(
      x, awq, signs, static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(padded_n), static_cast<std::uint32_t>(k),
      remap_gdn_heads ? 1u : 0u, tiles, scales);
}

void LaunchIu4SwiGluQuantize(
    const float* gate, const float* up, const void* npu_half,
    const Iu4NpuSwigluSource* npu_source, std::size_t npu_columns,
    const float* awq, const float* signs, std::size_t n, std::size_t padded_n,
    std::size_t k, std::uint8_t* tiles, float* scales, hipStream_t stream) {
  const std::size_t waves = padded_n * (k >> 8);
  Iu4NpuSwigluSource source;
  if (npu_source != nullptr)
    source = *npu_source;
  Iu4SwiGluQuantizeKernel<<<static_cast<unsigned>((waves + 7) / 8), 256, 0,
                            stream>>>(
      gate, up, static_cast<const __half*>(npu_half), source,
      static_cast<std::uint32_t>(npu_columns), awq, signs,
      static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(padded_n),
      static_cast<std::uint32_t>(k), tiles, scales);
}

void LaunchIu4PostNormGateQuantize(const float* raw, const float* ssm_norm,
                                   const float* gate, const float* awq,
                                   const float* signs, std::size_t n,
                                   std::size_t padded_n, std::size_t inner,
                                   std::uint8_t* tiles, float* scales,
                                   hipStream_t stream) {
  const std::size_t waves = padded_n * (inner >> 8);
  Iu4PostNormGateQuantizeKernel<<<static_cast<unsigned>((waves + 7) / 8), 256,
                                  0, stream>>>(
      raw, ssm_norm, gate, awq, signs, static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(padded_n), static_cast<std::uint32_t>(inner),
      tiles, scales);
}

void LaunchIu4RepackWeights(const void* blocks, void* tiles,
                            std::uint16_t* scales,
                            unsigned long long* bad_zero_points,
                            std::size_t rows, std::size_t padded_rows,
                            std::size_t cols, std::uint32_t perm_start,
                            std::uint32_t perm_chunk, hipStream_t stream) {
  const std::uint32_t groups = static_cast<std::uint32_t>(cols >> 7);
  const std::size_t total = (std::size_t)groups * rows * 4;
  Iu4RepackKernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(blocks),
      static_cast<std::uint8_t*>(tiles), scales, bad_zero_points,
      static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(padded_rows),
      groups, perm_start, perm_chunk);
}

}  // namespace gufo::hip
