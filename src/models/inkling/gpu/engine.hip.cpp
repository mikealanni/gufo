#include "src/models/inkling/gpu/engine.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/inkling/gpu/gptq.hpp"
#include "src/models/inkling/gpu/iu4_experts.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::inkling::gpu {
namespace {

using gufo::core::GgmlType;

constexpr int kThreads = 128;
constexpr int kTokenTile = 8;
constexpr int kMaxStates = 8;
constexpr int kConvTaps = 4;
constexpr int kHistory = 3;
constexpr int kMaxBatch = 4096;
constexpr std::size_t kBankMin = 3072;
constexpr int kWinLen =
    3 + 8;  // history + up to 8 verification inputs per channel
constexpr std::size_t kMaxVerifyRows = 8;
constexpr std::uint32_t kNpuSlice = 1024;
constexpr int kFlashHd = 128;
// Batches of at least this many tokens take the matrix-core path (experts
// sorted per expert so each weight row is read once); smaller ones, at most
// kMaxVerifyRows, take the per-token decode kernels. In between, the decode
// kernels read an expert once per token and the sorted path is faster.
constexpr int kWmmaMinTokens = 9;
// Rows the decode-path scratch is sized for.
constexpr int kScratchRows = 32;
constexpr float kMixEps = 1e-6F;
constexpr int kTypeF32 = 0;
constexpr int kTypeBf16 = 1;
constexpr int kTypeQ8 = 2;
constexpr int kTypeNvfp4 = 3;

#define HIP_OK(expr)                                             \
  do {                                                           \
    const hipError_t err_ = (expr);                              \
    if (err_ != hipSuccess) {                                    \
      std::fprintf(stderr, "inkling HIP error %s at %s:%d\n",    \
                   hipGetErrorString(err_), __FILE__, __LINE__); \
      std::abort();                                              \
    }                                                            \
  } while (0)

__constant__ float kE2m1[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                0, -1, -2, -3, -4, -6, -8, -12};

__device__ float Ue4m3(unsigned char x) {
  if (x == 0 || x == 0x7F)
    return 0.0F;
  const int e = (x >> 3) & 0xF;
  const int m = x & 7;
  const float raw = e == 0 ? ldexpf(static_cast<float>(m), -9)
                           : ldexpf(1.0F + static_cast<float>(m) / 8.0F, e - 7);
  return raw * 0.5F;
}

// E2M1 doubled value of a nibble: {0,1,2,3,4,6,8,12} with the sign in bit 3
__device__ __forceinline__ float E2m1(unsigned int n) {
  const int m = n & 7;
  const int v = m < 4 ? m : (2 + (m & 1)) << ((m >> 1) - 1);
  return (n & 8) != 0 ? -static_cast<float>(v) : static_cast<float>(v);
}

__device__ float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

template<int TY>
__device__ void Load32(const unsigned char* row, int chunk, float* w,
                       int cols) {
  if constexpr (TY == kTypeF32) {
    const float* p = reinterpret_cast<const float*>(row) + chunk * 32;
    for (int i = 0; i < 32; ++i)
      w[i] = p[i];
  } else if constexpr (TY == kTypeBf16) {
    const unsigned short* p =
        reinterpret_cast<const unsigned short*>(row) + chunk * 32;
    for (int i = 0; i < 32; ++i)
      w[i] = __uint_as_float(static_cast<unsigned>(p[i]) << 16);
  } else if constexpr (TY == kTypeQ8) {
    const float d =
        __half2float(*reinterpret_cast<const __half*>(row + cols + chunk * 2));
    const uint4* p = reinterpret_cast<const uint4*>(row + chunk * 32);
    const uint4 v[2] = {p[0], p[1]};
    for (int h = 0; h < 2; ++h) {
      const unsigned int u[4] = {v[h].x, v[h].y, v[h].z, v[h].w};
      for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 4; ++k) {
          w[h * 16 + i * 4 + k] =
              d * static_cast<float>(
                      static_cast<signed char>((u[i] >> (8 * k)) & 0xFF));
        }
      }
    }
  } else {
    for (int s2 = 0; s2 < 2; ++s2) {
      const int sb = chunk * 2 + s2;
      const float d = Ue4m3(row[(cols >> 1) + sb]);
      const uint2 u = *reinterpret_cast<const uint2*>(row + sb * 8);
      const unsigned int uu[2] = {u.x, u.y};
      for (int j = 0; j < 8; ++j) {
        const unsigned int byte = (uu[j >> 2] >> ((j & 3) * 8)) & 0xFF;
        w[s2 * 16 + j] = E2m1(byte & 15) * d;
        w[s2 * 16 + 8 + j] = E2m1(byte >> 4) * d;
      }
    }
  }
}

__device__ float BlockSum(float v, float* sh) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor(v, off);
  const int nw = (blockDim.x + 31) >> 5;
  if ((threadIdx.x & 31) == 0)
    sh[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = 0.0F;
  for (int w = 0; w < nw; ++w)
    r += sh[w];
  __syncthreads();
  return r;
}

__device__ __forceinline__ int Dot4(unsigned int a, unsigned int b, int c) {
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true,
                                 static_cast<int>(b), c, false);
}

// four E2M1 nibbles (one per byte, 0..15) -> four signed int8 values (doubled
// E2M1)
__device__ __forceinline__ unsigned int Nib4(unsigned int n) {
  const unsigned int mag = n & 0x07070707U;
  const unsigned int pos = __builtin_amdgcn_perm(0x0C080604U, 0x03020100U, mag);
  const unsigned int neg = __builtin_amdgcn_perm(0xF4F8FAFCU, 0xFDFEFF00U, mag);
  const unsigned int m = ((n >> 3) & 0x01010101U) * 0xFFU;
  return (neg & m) | (pos & ~m);
}

// x rows -> int8 per 32-block (scale = amax / 127)
__global__ void QuantRows(const float* x, signed char* xq, float* xs,
                          int cols) {
  const int row = blockIdx.x;
  const int unit = blockIdx.y;
  const int lane = threadIdx.x;
  const float v = x[static_cast<std::size_t>(row) * cols + unit * 32 + lane];
  float amax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, off));
  const float sc = amax / 127.0F;
  const float q = sc > 0.0F ? roundf(v / sc) : 0.0F;
  xq[static_cast<std::size_t>(row) * cols + unit * 32 + lane] =
      static_cast<signed char>(q);
  if (lane == 0)
    xs[static_cast<std::size_t>(row) * (cols / 32) + unit] = sc;
}

// Item i: Y[i][r] = scale[g] * W[g*rows + r] . x[xrow[i]] with int8
// activations; one wave per row.
template<int TY>
__global__ void GemmDp4(const unsigned char* w, std::size_t row_bytes, int cols,
                        int rows, const int* group, const int* xrow,
                        const float* scale, const signed char* xq,
                        const float* xs, float* y, const int* skip = nullptr,
                        bool nt = false) {
  if (skip != nullptr && *skip != 0)
    return;
  const int lane = threadIdx.x % warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (r >= rows)
    return;
  const int i = blockIdx.y;
  const int g = group != nullptr ? group[i] : 0;
  if (g < 0) {
    if (lane == 0)
      y[static_cast<std::size_t>(i) * rows + r] = 0.0F;
    return;
  }
  const int xr = xrow != nullptr ? xrow[i] : 0;
  const unsigned char* row =
      w + (static_cast<std::size_t>(g) * rows + r) * row_bytes;
  const signed char* xp = xq + static_cast<std::size_t>(xr) * cols;
  const float* xsp = xs + static_cast<std::size_t>(xr) * (cols / 32);
  float acc = 0.0F;
  for (int u = lane; u < cols / 32; u += warpSize) {
    const uint4* xv = reinterpret_cast<const uint4*>(xp + u * 32);
    const uint4 x0 = xv[0];
    const uint4 x1 = xv[1];
    if constexpr (TY == kTypeQ8) {
      const float d =
          __half2float(*reinterpret_cast<const __half*>(row + cols + u * 2));
      const uint4* wv = reinterpret_cast<const uint4*>(row + u * 32);
      const uint4 w0 = wv[0];
      const uint4 w1 = wv[1];
      int a = Dot4(w0.x, x0.x, 0);
      a = Dot4(w0.y, x0.y, a);
      a = Dot4(w0.z, x0.z, a);
      a = Dot4(w0.w, x0.w, a);
      a = Dot4(w1.x, x1.x, a);
      a = Dot4(w1.y, x1.y, a);
      a = Dot4(w1.z, x1.z, a);
      a = Dot4(w1.w, x1.w, a);
      acc += static_cast<float>(a) * d * xsp[u];
    } else {
      uint4 q;
      if (nt) {
        // expert weights are read once: buffer loads with DLC do not allocate
        // in the last-level cache, which keeps the prefetched dense weights
        // there (the resource is the wave-uniform row)
        typedef unsigned int U32x4 __attribute__((ext_vector_type(4)));
        const __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<unsigned char*>(row), static_cast<short>(0), 0x7FFFFFFF,
            0x31004000);
        const U32x4 t = __builtin_bit_cast(
            U32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, u * 16, 0, 4));
        q = make_uint4(t.x, t.y, t.z, t.w);
      } else {
        q = *reinterpret_cast<const uint4*>(row + u * 16);
      }
      const float d0 = Ue4m3(row[(cols >> 1) + u * 2]);
      const float d1 = Ue4m3(row[(cols >> 1) + u * 2 + 1]);
      const unsigned int lo = 0x0F0F0F0FU;
      int s0 = Dot4(Nib4(q.x & lo), x0.x, 0);
      s0 = Dot4(Nib4(q.y & lo), x0.y, s0);
      s0 = Dot4(Nib4((q.x >> 4) & lo), x0.z, s0);
      s0 = Dot4(Nib4((q.y >> 4) & lo), x0.w, s0);
      int s1 = Dot4(Nib4(q.z & lo), x1.x, 0);
      s1 = Dot4(Nib4(q.w & lo), x1.y, s1);
      s1 = Dot4(Nib4((q.z >> 4) & lo), x1.z, s1);
      s1 = Dot4(Nib4((q.w >> 4) & lo), x1.w, s1);
      acc +=
          (static_cast<float>(s0) * d0 + static_cast<float>(s1) * d1) * xsp[u];
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down(acc, off);
  if (lane == 0)
    y[static_cast<std::size_t>(i) * rows + r] =
        acc * (scale != nullptr ? scale[g] : 1.0F);
}

// Y[t][r] = W[r] . x[t] for NT tokens with int8 activations; the weight row is
// read once.
template<int NT>
__global__ void GemmDp4Q8N(const unsigned char* w, std::size_t row_bytes,
                           int cols, int rows, const signed char* xq,
                           const float* xs, float* y, std::size_t ldy) {
  const int lane = threadIdx.x % warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (r >= rows)
    return;
  const unsigned char* row = w + static_cast<std::size_t>(r) * row_bytes;
  float acc[NT] = {};
  for (int u = lane; u < cols / 32; u += warpSize) {
    const float d =
        __half2float(*reinterpret_cast<const __half*>(row + cols + u * 2));
    const uint4* wv = reinterpret_cast<const uint4*>(row + u * 32);
    const uint4 w0 = wv[0];
    const uint4 w1 = wv[1];
    for (int t = 0; t < NT; ++t) {
      const uint4* xv = reinterpret_cast<const uint4*>(
          xq + static_cast<std::size_t>(t) * cols + u * 32);
      const uint4 x0 = xv[0];
      const uint4 x1 = xv[1];
      int a = Dot4(w0.x, x0.x, 0);
      a = Dot4(w0.y, x0.y, a);
      a = Dot4(w0.z, x0.z, a);
      a = Dot4(w0.w, x0.w, a);
      a = Dot4(w1.x, x1.x, a);
      a = Dot4(w1.y, x1.y, a);
      a = Dot4(w1.z, x1.z, a);
      a = Dot4(w1.w, x1.w, a);
      acc[t] += static_cast<float>(a) * d *
                xs[static_cast<std::size_t>(t) * (cols / 32) + u];
    }
  }
  for (int t = 0; t < NT; ++t) {
    float a = acc[t];
    for (int off = warpSize / 2; off > 0; off >>= 1)
      a += __shfl_down(a, off);
    if (lane == 0)
      y[static_cast<std::size_t>(t) * ldy + r] = a;
  }
}

struct TouchList {
  const void* p[12];
  std::size_t bytes[12];
  int n;
};

// Reads every range of the list (a cache prefetch: nothing is written).
__global__ void __launch_bounds__(256)
    TouchKernel(TouchList l, unsigned int* sink) {
  unsigned int acc = 0;
  for (int k = 0; k < l.n; ++k) {
    const uint4* base = static_cast<const uint4*>(l.p[k]);
    const std::size_t cnt = l.bytes[k] / 16;
    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < cnt; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
      const uint4 v = base[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9E3779B1U)
    sink[0] = acc;
}

// Quantizes up to three distinct [rows][cols] inputs in one launch (blockIdx.z
// selects the input) into xq [3][rows * cols] and xs [3][rows * cols / 32].
__global__ void QuantRowsMulti(const float* x0, const float* x1,
                               const float* x2, signed char* xq, float* xs,
                               int cols, int rows_cap) {
  const int z = blockIdx.z;
  const float* x = z == 0 ? x0 : (z == 1 ? x1 : x2);
  const int row = blockIdx.x;
  const int unit = blockIdx.y;
  const int lane = threadIdx.x;
  const std::size_t off = static_cast<std::size_t>(z) * rows_cap;
  const float v = x[static_cast<std::size_t>(row) * cols + unit * 32 + lane];
  float amax = fabsf(v);
  for (int o = 16; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, o));
  const float sc = amax / 127.0F;
  const float q = sc > 0.0F ? roundf(v / sc) : 0.0F;
  xq[off * cols + static_cast<std::size_t>(row) * cols + unit * 32 + lane] =
      static_cast<signed char>(q);
  if (lane == 0)
    xs[off * (cols / 32) + static_cast<std::size_t>(row) * (cols / 32) + unit] =
        sc;
}

struct MultiSeg {
  const unsigned char* w;
  std::size_t row_bytes;
  float* y;
  std::size_t ldy;
  int rows;
  int xi;      // which quantized input
  int block0;  // first block of this matrix
};
struct MultiArgs {
  MultiSeg seg[6];
  int nseg;
  int cols;
  int rows_cap;
  const signed char* xq;
  const float* xs;
};

// Several Q8_0 matrices over (up to three) quantized inputs in one launch; the
// arithmetic of every row is that of GemmDp4Q8N<NT>.
template<int NT>
__global__ void GemmDp4Q8Multi(MultiArgs a) {
  int si = 0;
  while (si + 1 < a.nseg &&
         static_cast<int>(blockIdx.x) >= a.seg[si + 1].block0)
    ++si;
  const MultiSeg& sg = a.seg[si];
  const int lane = threadIdx.x % warpSize;
  const int r = (blockIdx.x - sg.block0) * (blockDim.x / warpSize) +
                threadIdx.x / warpSize;
  if (r >= sg.rows)
    return;
  const int cols = a.cols;
  const signed char* xq =
      a.xq + static_cast<std::size_t>(sg.xi) * a.rows_cap * cols;
  const float* xs =
      a.xs + static_cast<std::size_t>(sg.xi) * a.rows_cap * (cols / 32);
  const unsigned char* row = sg.w + static_cast<std::size_t>(r) * sg.row_bytes;
  float acc[NT] = {};
  for (int u = lane; u < cols / 32; u += warpSize) {
    const float d =
        __half2float(*reinterpret_cast<const __half*>(row + cols + u * 2));
    const uint4* wv = reinterpret_cast<const uint4*>(row + u * 32);
    const uint4 w0 = wv[0];
    const uint4 w1 = wv[1];
    for (int t = 0; t < NT; ++t) {
      const uint4* xv = reinterpret_cast<const uint4*>(
          xq + static_cast<std::size_t>(t) * cols + u * 32);
      const uint4 x0 = xv[0];
      const uint4 x1 = xv[1];
      int acci = Dot4(w0.x, x0.x, 0);
      acci = Dot4(w0.y, x0.y, acci);
      acci = Dot4(w0.z, x0.z, acci);
      acci = Dot4(w0.w, x0.w, acci);
      acci = Dot4(w1.x, x1.x, acci);
      acci = Dot4(w1.y, x1.y, acci);
      acci = Dot4(w1.z, x1.z, acci);
      acci = Dot4(w1.w, x1.w, acci);
      acc[t] += static_cast<float>(acci) * d *
                xs[static_cast<std::size_t>(t) * (cols / 32) + u];
    }
  }
  for (int t = 0; t < NT; ++t) {
    float v = acc[t];
    for (int off = warpSize / 2; off > 0; off >>= 1)
      v += __shfl_down(v, off);
    if (lane == 0)
      sg.y[static_cast<std::size_t>(t) * sg.ldy + r] = v;
  }
}

// Y[t][r] = W[r] . X[t]; one wave per output row, kTokenTile tokens per launch
// row.
template<int TY>
__global__ void GemmDense(const unsigned char* w, std::size_t row_bytes,
                          int cols, int rows, const float* x, std::size_t ldx,
                          float* y, std::size_t ldy, int n) {
  const int lane = threadIdx.x % warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (r >= rows)
    return;
  const int t0 = blockIdx.y * kTokenTile;
  const int tn = min(kTokenTile, n - t0);
  const unsigned char* row = w + static_cast<std::size_t>(r) * row_bytes;
  float acc[kTokenTile] = {};
  float wv[32];
  for (int c = lane; c < cols / 32; c += warpSize) {
    Load32<TY>(row, c, wv, cols);
    for (int t = 0; t < tn; ++t) {
      const float4* x4 =
          reinterpret_cast<const float4*>(x + (t0 + t) * ldx + c * 32);
      float a = 0.0F;
      for (int k = 0; k < 8; ++k) {
        const float4 xv = x4[k];
        a += wv[4 * k] * xv.x + wv[4 * k + 1] * xv.y + wv[4 * k + 2] * xv.z +
             wv[4 * k + 3] * xv.w;
      }
      acc[t] += a;
    }
  }
  for (int t = 0; t < tn; ++t) {
    float a = acc[t];
    for (int off = warpSize / 2; off > 0; off >>= 1)
      a += __shfl_down(a, off);
    if (lane == 0)
      y[(t0 + t) * ldy + r] = a;
  }
}

constexpr int kRoutedWaves = 8;

// Item i: Y[i][r] = scale[g] * W[g*rows + r] . X[xrow[i]]; one wave per output
// row.
template<int TY>
__global__ void GemmRouted(const unsigned char* w, std::size_t row_bytes,
                           int cols, int rows, const int* group,
                           const int* xrow, const float* scale, const float* x,
                           float* y) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + wave;
  if (r >= rows)
    return;
  const int i = blockIdx.y;
  const int g = group[i];
  const unsigned char* row =
      w + (static_cast<std::size_t>(g) * rows + r) * row_bytes;
  const float* xp = x + static_cast<std::size_t>(xrow[i]) * cols;
  float wv[32];
  float acc = 0.0F;
  for (int c = lane; c < cols / 32; c += warpSize) {
    Load32<TY>(row, c, wv, cols);
    const float4* x4 = reinterpret_cast<const float4*>(xp + c * 32);
    for (int k = 0; k < 8; ++k) {
      const float4 xv = x4[k];
      acc += wv[4 * k] * xv.x + wv[4 * k + 1] * xv.y + wv[4 * k + 2] * xv.z +
             wv[4 * k + 3] * xv.w;
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down(acc, off);
  if (lane == 0)
    y[static_cast<std::size_t>(i) * rows + r] =
        acc * (scale != nullptr ? scale[g] : 1.0F);
}

__global__ void EmbedQ8(const unsigned char* table, std::size_t row_bytes,
                        const int* tokens, float* out, int h) {
  const unsigned char* row =
      table + static_cast<std::size_t>(tokens[blockIdx.x]) * row_bytes;
  for (int c = threadIdx.x; c < h; c += blockDim.x) {
    const unsigned char* b = row + (c / 32) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    out[static_cast<std::size_t>(blockIdx.x) * h + c] =
        d *
        static_cast<float>(reinterpret_cast<const signed char*>(b + 2)[c % 32]);
  }
}

// out[row] = rms(in[row]) * w ; rows of `dim`, strided.
__global__ void Rms(const float* in, const float* w, float* out, int dim,
                    std::size_t ld_in, std::size_t ld_out, float eps,
                    __half* out_h) {
  __shared__ float red[256];
  const float* x = in + blockIdx.x * ld_in;
  float s = 0.0F;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    s += x[i] * x[i];
  s = BlockSum(s, red);
  const float inv = rsqrtf(s / static_cast<float>(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = x[i] * inv * (w != nullptr ? w[i] : 1.0F);
    if (out != nullptr)
      out[blockIdx.x * ld_out + i] = v;
    if (out_h != nullptr)
      out_h[blockIdx.x * ld_out + i] = __float2half(v);
  }
}

// Wide-row RMS norm: one pass over 128-bit loads, the row kept in registers
// (dim <= 3072 and 16-byte aligned rows), fp32 and/or fp16 outputs.
constexpr int kRmsVecMaxDim = 3072;
__global__ void __launch_bounds__(256)
    RmsVec(const float* in, const float* w, float* out, int dim,
           std::size_t ld_in, std::size_t ld_out, float eps, __half* out_h) {
  __shared__ float red[256];
  constexpr int kPer = kRmsVecMaxDim / 4 / 256;
  const float4* x4 = reinterpret_cast<const float4*>(in + blockIdx.x * ld_in);
  const int n4 = dim / 4;
  float4 v[kPer];
  float s = 0.0F;
  for (int k = 0; k < kPer; ++k) {
    const int i = threadIdx.x + k * 256;
    if (i < n4) {
      v[k] = x4[i];
      s +=
          v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
    }
  }
  s = BlockSum(s, red);
  const float inv = rsqrtf(s / static_cast<float>(dim) + eps);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4* o4 = reinterpret_cast<float4*>(out + blockIdx.x * ld_out);
  __half2* h2 = reinterpret_cast<__half2*>(out_h + blockIdx.x * ld_out);
  for (int k = 0; k < kPer; ++k) {
    const int i = threadIdx.x + k * 256;
    if (i >= n4)
      continue;
    const float4 wv = w != nullptr ? w4[i] : make_float4(1.F, 1.F, 1.F, 1.F);
    const float4 r = make_float4(v[k].x * inv * wv.x, v[k].y * inv * wv.y,
                                 v[k].z * inv * wv.z, v[k].w * inv * wv.w);
    if (out != nullptr)
      o4[i] = r;
    if (out_h != nullptr) {
      h2[2 * i] = __floats2half2_rn(r.x, r.y);
      h2[2 * i + 1] = __floats2half2_rn(r.z, r.w);
    }
  }
}

// per-row L2 clip after optional per-element scale indexed [(row % scale_rows)
// * dim + i]
__global__ void Clip(float* x, const float* scale, int scale_rows, int dim,
                     float threshold, const int* tau_pos = nullptr,
                     int rows_per_token = 1, float tau_floor = 1.0F,
                     float tau_alpha = 0.0F) {
  __shared__ float red[256];
  float* p = x + static_cast<std::size_t>(blockIdx.x) * dim;
  const int sr = scale != nullptr ? blockIdx.x % scale_rows : 0;
  float s = 0.0F;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    if (scale != nullptr)
      p[i] *= scale[sr * dim + i];
    s += p[i] * p[i];
  }
  s = BlockSum(s, red);
  float f = fminf(1.0F, threshold / (sqrtf(s) + 1e-6F));
  if (tau_pos != nullptr) {
    const int token = blockIdx.x / rows_per_token;
    f *= 1.0F +
         tau_alpha * logf(fmaxf(1.0F, static_cast<float>(*tau_pos + token + 1) /
                                          tau_floor));
  }
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    p[i] *= f;
}

struct ClipJob {
  float* x;
  const float* scale;
  int scale_rows, dim, blocks, rows_per_token;
  float threshold, tau_floor, tau_alpha;
  const int* tau_pos;
};
struct ClipJobs {
  ClipJob job[3];
};

// the three per-row clips of a decode attention visit in one launch; each
// block runs exactly the arithmetic of Clip
__global__ void ClipMulti(ClipJobs jobs) {
  __shared__ float red[256];
  int b = blockIdx.x;
  int j = 0;
  while (j < 2 && b >= jobs.job[j].blocks) {
    b -= jobs.job[j].blocks;
    ++j;
  }
  const ClipJob& J = jobs.job[j];
  float* p = J.x + static_cast<std::size_t>(b) * J.dim;
  const int sr = J.scale != nullptr ? b % J.scale_rows : 0;
  float s = 0.0F;
  for (int i = threadIdx.x; i < J.dim; i += blockDim.x) {
    if (J.scale != nullptr)
      p[i] *= J.scale[sr * J.dim + i];
    s += p[i] * p[i];
  }
  s = BlockSum(s, red);
  float f = fminf(1.0F, J.threshold / (sqrtf(s) + 1e-6F));
  if (J.tau_pos != nullptr) {
    const int token = b / J.rows_per_token;
    f *= 1.0F + J.tau_alpha *
                    logf(fmaxf(1.0F, static_cast<float>(*J.tau_pos + token + 1) /
                                         J.tau_floor));
  }
  for (int i = threadIdx.x; i < J.dim; i += blockDim.x)
    p[i] *= f;
}

// Warp-per-row variants of Clip and the per-head RMS norm for short rows
// (dim <= 128): eight rows per 256-thread block, one shuffle reduction per row.
__global__ void __launch_bounds__(256)
    ClipWave(float* x, const float* scale, int scale_rows, int dim,
             float threshold, const int* tau_pos, int rows_per_token,
             float tau_floor, float tau_alpha, int rows) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + wave;
  if (row >= rows)
    return;
  float* p = x + static_cast<std::size_t>(row) * dim;
  const int sr = scale != nullptr ? row % scale_rows : 0;
  float vals[4];
  float s = 0.0F;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = lane + 32 * k;
    vals[k] = 0.0F;
    if (i < dim) {
      float v = p[i];
      if (scale != nullptr)
        v *= scale[sr * dim + i];
      vals[k] = v;
      s += v * v;
    }
  }
  for (int off = 16; off > 0; off >>= 1)
    s += __shfl_xor(s, off);
  float f = fminf(1.0F, threshold / (sqrtf(s) + 1e-6F));
  if (tau_pos != nullptr) {
    const int token = row / rows_per_token;
    f *= 1.0F +
         tau_alpha * logf(fmaxf(1.0F, static_cast<float>(*tau_pos + token + 1) /
                                          tau_floor));
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = lane + 32 * k;
    if (i < dim)
      p[i] = vals[k] * f;
  }
}

__global__ void __launch_bounds__(256)
    RmsWave(const float* in, float* out, int dim, int rows, float eps) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + wave;
  if (row >= rows)
    return;
  const float* x = in + static_cast<std::size_t>(row) * dim;
  float vals[4];
  float s = 0.0F;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = lane + 32 * k;
    vals[k] = 0.0F;
    if (i < dim) {
      vals[k] = x[i];
      s += vals[k] * vals[k];
    }
  }
  for (int off = 16; off > 0; off >>= 1)
    s += __shfl_xor(s, off);
  const float inv = rsqrtf(s / static_cast<float>(dim) + eps);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = lane + 32 * k;
    if (i < dim)
      out[static_cast<std::size_t>(row) * dim + i] = vals[k] * inv;
  }
}

__global__ void TauScale(float* x, int per_token, const int* pos0,
                         float floor_v, float alpha) {
  const float tau =
      1.0F +
      alpha * logf(fmaxf(1.0F,
                         static_cast<float>(*pos0 + blockIdx.x + 1) / floor_v));
  for (int i = threadIdx.x; i < per_token; i += blockDim.x)
    x[static_cast<std::size_t>(blockIdx.x) * per_token + i] *= tau;
}

__device__ __forceinline__ void ConvOne(const float* cur, const float* w,
                                         const float* dyn, int heads,
                                         float* hist, float* out, int ch, int n,
                                         const float* add, float* win, int c) {
  const int per = dyn != nullptr ? ch / heads : ch;
  const float norm =
      rsqrtf(static_cast<float>((dyn != nullptr ? 2 : 1) * kConvTaps + 1));
  float wn[kConvTaps];
  for (int k = 0; k < kHistory; ++k)
    wn[k] = hist[c * kHistory + k];
  if (win != nullptr) {
    for (int k = 0; k < kHistory; ++k)
      win[c * kWinLen + k] = wn[k];
  }
  float wt[kConvTaps];
  for (int j = 0; j < kConvTaps; ++j)
    wt[j] = w[c * kConvTaps + j];
  for (int t = 0; t < n; ++t) {
    const float x = cur[static_cast<std::size_t>(t) * ch + c];
    wn[kHistory] = x;
    if (win != nullptr)
      win[c * kWinLen + kHistory + t] = x;
    float o = 0.0F;
    for (int j = 0; j < kConvTaps; ++j)
      o += wt[j] * wn[j];
    if (dyn != nullptr) {
      const float* d = dyn + static_cast<std::size_t>(t) * heads * kConvTaps +
                       (c / per) * kConvTaps;
      for (int lag = 0; lag < kConvTaps; ++lag)
        o += d[lag] * wn[kHistory - lag];
    }
    out[static_cast<std::size_t>(t) * ch + c] =
        (x + o) * norm +
        (add != nullptr ? add[static_cast<std::size_t>(t) * ch + c] : 0.0F);
    for (int k = 0; k < kHistory; ++k)
      wn[k] = wn[k + 1];
  }
  for (int k = 0; k < kHistory; ++k)
    hist[c * kHistory + k] = wn[k];
}

__global__ void Conv(const float* cur, const float* w, const float* dyn,
                     int heads, float* hist, float* out, int ch, int n,
                     const float* add, float* win) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ch)
    return;
  ConvOne(cur, w, dyn, heads, hist, out, ch, n, add, win, c);
}

// the key and value convolutions of a decode visit: one thread runs the key
// channel and then the value channel (the value output reuses the key input)
__global__ void Conv2(const float* cur_a, const float* w_a, const float* dyn_a,
                      float* hist_a, float* out_a, const float* cur_b,
                      const float* w_b, const float* dyn_b, float* hist_b,
                      float* out_b, int heads, int ch, int n) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ch)
    return;
  ConvOne(cur_a, w_a, dyn_a, heads, hist_a, out_a, ch, n, nullptr, nullptr, c);
  ConvOne(cur_b, w_b, dyn_b, heads, hist_b, out_b, ch, n, nullptr, nullptr, c);
}

__global__ void ConvT(const float* cur, const float* w, const float* dyn,
                      int heads, const float* hist, float* out, int ch,
                      const float* add) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ch)
    return;
  const int t = blockIdx.y;
  const int per = dyn != nullptr ? ch / heads : ch;
  float win[kConvTaps];
  for (int j = 0; j < kConvTaps; ++j) {
    const int src = t - (kHistory - j);
    win[j] = src >= 0 ? cur[static_cast<std::size_t>(src) * ch + c]
                      : hist[c * kHistory + (src + kHistory)];
  }
  float o = 0.0F;
  for (int j = 0; j < kConvTaps; ++j)
    o += w[c * kConvTaps + j] * win[j];
  if (dyn != nullptr) {
    const float* d = dyn + static_cast<std::size_t>(t) * heads * kConvTaps +
                     (c / per) * kConvTaps;
    for (int lag = 0; lag < kConvTaps; ++lag)
      o += d[lag] * win[kHistory - lag];
  }
  out[static_cast<std::size_t>(t) * ch + c] =
      (win[kHistory] + o) *
          rsqrtf(static_cast<float>((dyn != nullptr ? 2 : 1) * kConvTaps + 1)) +
      (add != nullptr ? add[static_cast<std::size_t>(t) * ch + c] : 0.0F);
}

__global__ void ConvHist(const float* cur, float* hist, int ch, int n) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ch)
    return;
  float nh[kHistory];
  for (int k = 0; k < kHistory; ++k) {
    const int src = n - kHistory + k;
    nh[k] = src >= 0 ? cur[static_cast<std::size_t>(src) * ch + c]
                     : hist[c * kHistory + src + kHistory];
  }
  for (int k = 0; k < kHistory; ++k)
    hist[c * kHistory + k] = nh[k];
}

constexpr int kTapCount = 4;
inline int g_tap_layers[kTapCount] = {
    4, 22, 40, 48};  // virtual layers whose outputs feed the drafter

// feature rows of the drafter: feat[pos][tap][h] as F16
__global__ void TapStore(const float* cur, __half* feat, const int* pos0,
                         int tap, int h) {
  const std::size_t dst =
      (static_cast<std::size_t>(*pos0 + blockIdx.x) * kTapCount + tap) * h;
  for (int i = threadIdx.x; i < h; i += blockDim.x)
    feat[dst + i] =
        __float2half(cur[static_cast<std::size_t>(blockIdx.x) * h + i]);
}

struct HistEntry {
  float* hist;
  int ch;
};

// after a verification batch: history = the window after `keep` of the batch's
// inputs
__global__ void FixHist(const HistEntry* entries, int keep) {
  const HistEntry e = entries[blockIdx.y];
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= e.ch)
    return;
  const float* win = e.hist + static_cast<std::size_t>(e.ch) * kHistory;
  float v[kHistory];
  for (int m = 0; m < kHistory; ++m)
    v[m] = win[c * kWinLen + keep + m];
  for (int m = 0; m < kHistory; ++m)
    e.hist[c * kHistory + m] = v[m];
}

struct StatePtrs {
  const float* p[kMaxStates];
  int count;
  bool half_blocks;  ///< the block states are FP16 rows (cur is always F32)
};

__device__ __forceinline__ float4 StateRow4(const void* row, bool half, int i) {
  if (!half)
    return reinterpret_cast<const float4*>(row)[i];
  const uint2 u = reinterpret_cast<const uint2*>(row)[i];
  const float2 a = __half22float2(__builtin_bit_cast(__half2, u.x));
  const float2 b = __half22float2(__builtin_bit_cast(__half2, u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float StateAt(const void* row, bool half, int i) {
  return half ? __half2float(reinterpret_cast<const __half*>(row)[i])
              : reinterpret_cast<const float*>(row)[i];
}

__device__ __forceinline__ const void* StateRowPtr(const StatePtrs& blocks,
                                                   const float* cur, int s,
                                                   std::size_t t, int h) {
  if (s == blocks.count)
    return cur + t * h;
  return blocks.half_blocks
             ? static_cast<const void*>(
                   reinterpret_cast<const __half*>(blocks.p[s]) + t * h)
             : static_cast<const void*>(blocks.p[s] + t * h);
}

// Norm and query scores of every state of one token in a single pass over the
// row (128-bit loads, all states accumulated together, one cross-wave
// reduction): inv[s] = 1 / rms(state s), score[q][s] = state . query[q] *
// inv[s] / sqrt(h). The block is 256 threads.
__device__ void MixScores(const void* const* st, int ns, bool half_blocks,
                          const float* query, int nq, int h, float* inv,
                          float (*score)[kMaxStates], float* sred) {
  constexpr int kAcc = 5 * kMaxStates;
  float acc[kAcc];
#pragma unroll
  for (int k = 0; k < kAcc; ++k)
    acc[k] = 0.0F;
  const int h4 = h / 4;
  for (int i = threadIdx.x; i < h4; i += blockDim.x) {
    float4 qv[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (q < nq)
        qv[q] = reinterpret_cast<const float4*>(query + q * h)[i];
    }
#pragma unroll
    for (int s = 0; s < kMaxStates; ++s) {
      if (s < ns) {
        const float4 v = StateRow4(st[s], half_blocks && s < ns - 1, i);
        acc[s * 5] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (q < nq)
            acc[s * 5 + 1 + q] +=
                v.x * qv[q].x + v.y * qv[q].y + v.z * qv[q].z + v.w * qv[q].w;
        }
      }
    }
  }
  const int wave = threadIdx.x >> 5;
#pragma unroll
  for (int k = 0; k < kAcc; ++k) {
    float v = acc[k];
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_xor(v, off);
    if ((threadIdx.x & 31) == 0)
      sred[wave * kAcc + k] = v;
  }
  __syncthreads();
  if (threadIdx.x < ns * 5) {
    float tot = 0.0F;
    for (int w = 0; w < 8; ++w)
      tot += sred[w * kAcc + threadIdx.x];
    sred[8 * kAcc + threadIdx.x] = tot;
  }
  __syncthreads();
  if (threadIdx.x < ns) {
    const int s = threadIdx.x;
    inv[s] = rsqrtf(sred[8 * kAcc + s * 5] / static_cast<float>(h) + kMixEps);
    for (int q = 0; q < nq; ++q)
      score[q][s] = sred[8 * kAcc + s * 5 + 1 + q] * inv[s] *
                    rsqrtf(static_cast<float>(h));
  }
  __syncthreads();
}

// states = rms(blocks..., cur); out[q][t] = softmax(states . query[q] /
// sqrt(h)) mixture of normalized states. One wave per state computes its norm
// and dots in parallel.
__global__ void Mix(StatePtrs blocks, const float* cur, const float* query,
                    int nq, int h, int n, float* out) {
  __shared__ float inv[kMaxStates];
  __shared__ float score[4][kMaxStates];
  __shared__ float wgt[4][kMaxStates];
  const int t = blockIdx.x;
  const int ns = blocks.count + 1;
  const int wave = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const std::size_t toff = static_cast<std::size_t>(t) * h;
  if (wave < ns) {
    const float4* x = reinterpret_cast<const float4*>(
        (wave < blocks.count ? blocks.p[wave] : cur) + toff);
    float acc[5] = {};
    for (int i = lane; i < h / 4; i += warpSize) {
      const float4 v = x[i];
      acc[0] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
      for (int q = 0; q < nq; ++q) {
        const float4 w = reinterpret_cast<const float4*>(query + q * h)[i];
        acc[1 + q] += v.x * w.x + v.y * w.y + v.z * w.z + v.w * w.w;
      }
    }
    for (int k = 0; k < 5; ++k) {
      for (int off = warpSize / 2; off > 0; off >>= 1)
        acc[k] += __shfl_xor(acc[k], off);
    }
    if (lane == 0) {
      inv[wave] = rsqrtf(acc[0] / static_cast<float>(h) + kMixEps);
      for (int q = 0; q < nq; ++q)
        score[q][wave] = acc[1 + q] * inv[wave] * rsqrtf(static_cast<float>(h));
    }
  }
  __syncthreads();
  if (threadIdx.x < nq) {
    const int q = threadIdx.x;
    float mx = -1e30F;
    for (int s = 0; s < ns; ++s)
      mx = fmaxf(mx, score[q][s]);
    float den = 0.0F;
    for (int s = 0; s < ns; ++s) {
      wgt[q][s] = expf(score[q][s] - mx);
      den += wgt[q][s];
    }
    for (int s = 0; s < ns; ++s)
      wgt[q][s] = wgt[q][s] / den * inv[s];
  }
  __syncthreads();
  for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < h;
       i += gridDim.y * blockDim.x) {
    float a[4] = {};
    for (int s = 0; s < ns; ++s) {
      const float v = (s < blocks.count ? blocks.p[s] : cur)[toff + i];
      for (int q = 0; q < nq; ++q)
        a[q] += wgt[q][s] * v;
    }
    for (int q = 0; q < nq; ++q)
      out[(static_cast<std::size_t>(q) * n + t) * h + i] = a[q];
  }
}

// states = rms(blocks..., cur); out[q][t] = softmax(states . query[q] /
// sqrt(h)) mixture of normalized states
__global__ void MixBig(StatePtrs blocks, const float* cur, const float* query,
                       int nq, int h, int n, float* out) {
  __shared__ float sred[9 * 5 * kMaxStates];
  __shared__ float inv[kMaxStates];
  __shared__ float score[4][kMaxStates];
  const int t = blockIdx.x;
  const int ns = blocks.count + 1;
  const void* st[kMaxStates];
  for (int s = 0; s < ns; ++s)
    st[s] = StateRowPtr(blocks, cur, s, t, h);
  MixScores(st, ns, blocks.half_blocks, query, nq, h, inv, score, sred);
  for (int q = 0; q < nq; ++q) {
    float mx = -1e30F;
    for (int s = 0; s < ns; ++s)
      mx = fmaxf(mx, score[q][s]);
    float e[kMaxStates];
    float den = 0.0F;
    for (int s = 0; s < ns; ++s) {
      e[s] = expf(score[q][s] - mx);
      den += e[s];
    }
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
      float a = 0.0F;
      for (int s = 0; s < ns; ++s)
        a += e[s] / den * StateAt(st[s], blocks.half_blocks && s < ns - 1, i) *
             inv[s];
      out[(static_cast<std::size_t>(q) * n + t) * h + i] = a;
    }
  }
}

struct NormPtrs {
  const float* w[3];
  float* out[3];
  __half* outh[3];
};

// MixBig followed by the per-query RMS norm with weights; writes float and half
// rows.
__global__ void MixRms(StatePtrs blocks, const float* cur, const float* query,
                       int nq, int h, int n, NormPtrs np, float eps) {
  extern __shared__ float mixbuf[];
  __shared__ float sred[9 * 5 * kMaxStates];
  __shared__ float inv[kMaxStates];
  __shared__ float score[4][kMaxStates];
  __shared__ float rsum[256];
  const int t = blockIdx.x;
  const int ns = blocks.count + 1;
  const void* st[kMaxStates];
  for (int s = 0; s < ns; ++s)
    st[s] = StateRowPtr(blocks, cur, s, t, h);
  MixScores(st, ns, blocks.half_blocks, query, nq, h, inv, score, sred);
  for (int q = 0; q < nq; ++q) {
    float mx = -1e30F;
    for (int s = 0; s < ns; ++s)
      mx = fmaxf(mx, score[q][s]);
    float e[kMaxStates];
    float den = 0.0F;
    for (int s = 0; s < ns; ++s) {
      e[s] = expf(score[q][s] - mx);
      den += e[s];
    }
    float ss = 0.0F;
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
      float a = 0.0F;
      for (int s = 0; s < ns; ++s)
        a += e[s] / den * StateAt(st[s], blocks.half_blocks && s < ns - 1, i) *
             inv[s];
      mixbuf[q * h + i] = a;
      ss += a * a;
    }
    const float r = rsqrtf(BlockSum(ss, rsum) / static_cast<float>(h) + eps);
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
      const float v = mixbuf[q * h + i] * r * np.w[q][i];
      const std::size_t at = static_cast<std::size_t>(t) * h + i;
      np.out[q][at] = v;
      if (np.outh[q] != nullptr)
        np.outh[q][at] = __float2half(v);
    }
  }
}

__global__ void Control(float* cur, const float* msg, const float* gain,
                        const float* forget, int h) {
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * h;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    const float c = cur[base + i];
    cur[base + i] =
        c - tanhf(forget[base + i]) * c + tanhf(gain[base + i]) * msg[base + i];
  }
}

constexpr int kControlRank = 64;
constexpr int kControlWaves = 8;

// low[row][r] = down[r] . [rms(cur[row]); rms(msg[row])] for 8 outputs per
// block
// BF16 weight values (the checkpoint's own precision) widen exactly
__device__ __forceinline__ float4 Bf16x4(uint2 u) {
  return make_float4(
      __uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000U),
      __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000U));
}

// up to three loop controllers of one visit batched along blockIdx.z
struct CtlBatch {
  float* cur[3];
  const float* msg[3];
  const void* down[3];
  const void* gain[3];
  const void* forget[3];
  float* low[3];
};

template<bool H>
__global__ void __launch_bounds__(kControlWaves * 32)
    ControlLow(CtlBatch b, int h, float eps) {
  __shared__ float red[2][kControlWaves];
  const float* cur = b.cur[blockIdx.z];
  const float* msg = b.msg[blockIdx.z];
  const void* down_v = b.down[blockIdx.z];
  float* low = b.low[blockIdx.z];
  const int row = blockIdx.x;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float* c = cur + static_cast<std::size_t>(row) * h;
  const float* m = msg + static_cast<std::size_t>(row) * h;
  const float4* c4 = reinterpret_cast<const float4*>(c);
  const float4* m4 = reinterpret_cast<const float4*>(m);
  float sc = 0.0F;
  float sm = 0.0F;
  for (int i = threadIdx.x; i < h / 4; i += blockDim.x) {
    const float4 cv = c4[i];
    const float4 mv = m4[i];
    sc += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
    sm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
  }
  for (int off = 16; off > 0; off >>= 1) {
    sc += __shfl_xor(sc, off);
    sm += __shfl_xor(sm, off);
  }
  if (lane == 0) {
    red[0][wave] = sc;
    red[1][wave] = sm;
  }
  __syncthreads();
  float tc = 0.0F;
  float tm = 0.0F;
  for (int w = 0; w < kControlWaves; ++w) {
    tc += red[0][w];
    tm += red[1][w];
  }
  const float ic = rsqrtf(tc / static_cast<float>(h) + eps);
  const float im = rsqrtf(tm / static_cast<float>(h) + eps);
  const int r = blockIdx.y * kControlWaves + wave;
  const float4* wa = nullptr;
  const float4* wb = nullptr;
  const uint2* ha = nullptr;
  const uint2* hb = nullptr;
  if constexpr (H) {
    const unsigned short* w = static_cast<const unsigned short*>(down_v) +
                              static_cast<std::size_t>(r) * 2 * h;
    ha = reinterpret_cast<const uint2*>(w);
    hb = reinterpret_cast<const uint2*>(w + h);
  } else {
    const float* w =
        static_cast<const float*>(down_v) + static_cast<std::size_t>(r) * 2 * h;
    wa = reinterpret_cast<const float4*>(w);
    wb = reinterpret_cast<const float4*>(w + h);
  }
  float acc = 0.0F;
  for (int i = lane; i < h / 4; i += 32) {
    float4 a, b;
    if constexpr (H) {
      a = Bf16x4(ha[i]);
      b = Bf16x4(hb[i]);
    } else {
      a = wa[i];
      b = wb[i];
    }
    const float4 cv = c4[i];
    const float4 mv = m4[i];
    acc += (a.x * cv.x + a.y * cv.y + a.z * cv.z + a.w * cv.w) * ic +
           (b.x * mv.x + b.y * mv.y + b.z * mv.z + b.w * mv.w) * im;
  }
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off);
  if (lane == 0)
    low[static_cast<std::size_t>(row) * kControlRank + r] = acc;
}

// cur += -tanh(forget) * cur + tanh(gain) * msg with gain/forget = W . low
template<bool H>
__global__ void ControlApply(CtlBatch b, int h) {
  __shared__ float lw[kControlRank];
  float* cur = b.cur[blockIdx.z];
  const float* msg = b.msg[blockIdx.z];
  const float* low = b.low[blockIdx.z];
  const void* gain_v = b.gain[blockIdx.z];
  const void* forget_v = b.forget[blockIdx.z];
  const int row = blockIdx.x;
  if (threadIdx.x < kControlRank)
    lw[threadIdx.x] =
        low[static_cast<std::size_t>(row) * kControlRank + threadIdx.x];
  __syncthreads();
  const int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= h)
    return;
  float g = 0.0F;
  float f = 0.0F;
  for (int k = 0; k < kControlRank / 4; ++k) {
    float4 gv, fv;
    if constexpr (H) {
      gv = Bf16x4(reinterpret_cast<const uint2*>(
          static_cast<const unsigned short*>(gain_v) +
          static_cast<std::size_t>(j) * kControlRank)[k]);
      fv = Bf16x4(reinterpret_cast<const uint2*>(
          static_cast<const unsigned short*>(forget_v) +
          static_cast<std::size_t>(j) * kControlRank)[k]);
    } else {
      gv = reinterpret_cast<const float4*>(static_cast<const float*>(gain_v) +
                                           static_cast<std::size_t>(j) *
                                               kControlRank)[k];
      fv = reinterpret_cast<const float4*>(static_cast<const float*>(forget_v) +
                                           static_cast<std::size_t>(j) *
                                               kControlRank)[k];
    }
    g += gv.x * lw[4 * k] + gv.y * lw[4 * k + 1] + gv.z * lw[4 * k + 2] +
         gv.w * lw[4 * k + 3];
    f += fv.x * lw[4 * k] + fv.y * lw[4 * k + 1] + fv.z * lw[4 * k + 2] +
         fv.w * lw[4 * k + 3];
  }
  const std::size_t idx = static_cast<std::size_t>(row) * h + j;
  const float c = cur[idx];
  cur[idx] = c - tanhf(f) * c + tanhf(g) * msg[idx];
}

// K rows are [row][kv head][128]; V is stored block-transposed per KV head:
// [head][64-key block][d][key in block], so a key tile is one contiguous 16 KB.
__global__ void StoreKv(const float* k, const float* v, __half* kc, __half* vc,
                        int dim, const int* pos0, int ring, int vblocks) {
  const int row = *pos0 + blockIdx.x;
  const int r = ring > 0 ? row % ring : row;
  const std::size_t dst = static_cast<std::size_t>(r) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    kc[dst + i] =
        __float2half(k[static_cast<std::size_t>(blockIdx.x) * dim + i]);
    const int g = i / kFlashHd;
    const int d = i % kFlashHd;
    vc[((static_cast<std::size_t>(g) * vblocks + (r >> 6)) * kFlashHd + d) *
           64 +
       (r & 63)] =
        __float2half(v[static_cast<std::size_t>(blockIdx.x) * dim + i]);
  }
}

float HalfToFloat(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

uint16_t FloatToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  u += 0x7FFFU + ((u >> 16U) & 1U);
  return static_cast<uint16_t>(u >> 16U);
}

template<typename T>
void WriteDump(const std::string& path, const std::vector<T>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "inkling: cannot write %s\n", path.c_str());
    std::abort();
  }
  std::fwrite(v.data(), sizeof(T), v.size(), f);
  std::fclose(f);
}

struct AttnArgs {
  int nh, nkv, hd, nrel, extent, window, ring, vblocks;
  const int* pos0;
  float scale;
  int pack{0};  // few-row decode: one block per kv group, rows x heads packed
};

// one block per (head, token); scores in dynamic shared memory
__global__ void Attention(AttnArgs a, const float* q, const float* r,
                          const float* sink, const __half* kc, const __half* vc,
                          const float* bias, float* out) {
  extern __shared__ float sc[];
  __shared__ float red[kThreads];
  __shared__ float rh[32];
  const int h = blockIdx.x;
  const int t = blockIdx.y;
  const int pos = *a.pos0 + t;
  const int g = h / (a.nh / a.nkv);
  const int first = a.window > 0 && pos + 1 > a.window ? pos + 1 - a.window : 0;
  const float* qh = q + (static_cast<std::size_t>(t) * a.nh + h) * a.hd;
  if (threadIdx.x < a.nrel)
    rh[threadIdx.x] =
        r[(static_cast<std::size_t>(t) * a.nh + h) * a.nrel + threadIdx.x];
  const float sk = sink[static_cast<std::size_t>(t) * a.nh + h];
  __syncthreads();
  const int kvdim = a.nkv * a.hd;
  float mx = 0.0F;
  for (int j = first + threadIdx.x; j <= pos; j += blockDim.x) {
    const __half* kj =
        kc + static_cast<std::size_t>(a.ring > 0 ? j % a.ring : j) * kvdim +
        g * a.hd;
    float d = 0.0F;
    for (int i = 0; i < a.hd; ++i)
      d += qh[i] * __half2float(kj[i]);
    const int dist = pos - j;
    float b = 0.0F;
    if (dist < a.extent) {
      for (int i = 0; i < a.nrel; ++i)
        b += bias[static_cast<std::size_t>(dist) * a.nrel + i] * rh[i];
    }
    const float s = d * a.scale + b - sk;
    sc[j - first] = s;
    mx = fmaxf(mx, s);
  }
  // block max
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  float den = 0.0F;
  for (int j = first + threadIdx.x; j <= pos; j += blockDim.x) {
    const float e = expf(sc[j - first] - mx);
    sc[j - first] = e;
    den += e;
  }
  den = BlockSum(den, red) + expf(-mx);
  for (int d = threadIdx.x; d < a.hd; d += blockDim.x) {
    float o = 0.0F;
    for (int j = first; j <= pos; ++j)
      o +=
          sc[j - first] *
          __half2float(
              vc[static_cast<std::size_t>(a.ring > 0 ? j % a.ring : j) * kvdim +
                 g * a.hd + d]);
    out[(static_cast<std::size_t>(t) * a.nh + h) * a.hd + d] = o / den;
  }
}

__global__ void SigmoidMul(float* x, const float* g, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] *= Sigmoid(g[i]);
}

__global__ void SigmoidMulH(float* x, const float* g, __half* xh,
                            std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i] * Sigmoid(g[i]);
    x[i] = v;
    xh[i] = __float2half(v);
  }
}

__global__ void AddInto(float* x, const float* y, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += y[i];
}

__global__ void Concat2(const float* a, const float* b, float* out, int h) {
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    out[static_cast<std::size_t>(blockIdx.x) * 2 * h + i] =
        a[static_cast<std::size_t>(blockIdx.x) * h + i];
    out[static_cast<std::size_t>(blockIdx.x) * 2 * h + h + i] =
        b[static_cast<std::size_t>(blockIdx.x) * h + i];
  }
}

// one block per token: sigmoid router, top-k on prob+bias, normalized scaled
// weights
__global__ void Route(const float* logits, const float* bias, int ne, int nu,
                      float wscale, int* ids, float* wts, float prune = 0.0F) {
  __shared__ float sel[256];
  __shared__ float prob[256];
  __shared__ float best_v[256];
  __shared__ int best_i[256];
  const int t = blockIdx.x;
  const int e = threadIdx.x;
  float p = 0.0F;
  if (e < ne) {
    p = Sigmoid(logits[static_cast<std::size_t>(t) * ne + e]);
    prob[e] = p;
    sel[e] = p + bias[e];
  } else {
    sel[e] = -1e30F;
  }
  __syncthreads();
  float sum = 0.0F;
  for (int k = 0; k < nu; ++k) {
    float bv = sel[e];
    int bi = e;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off);
      const int oi = __shfl_xor(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (e % warpSize == 0) {
      best_v[e / warpSize] = bv;
      best_i[e / warpSize] = bi;
    }
    __syncthreads();
    int win = best_i[0];
    float wv = best_v[0];
    for (int w = 1; w < blockDim.x / warpSize; ++w) {
      if (best_v[w] > wv || (best_v[w] == wv && best_i[w] < win)) {
        wv = best_v[w];
        win = best_i[w];
      }
    }
    if (e == 0) {
      ids[t * nu + k] = win;
      wts[t * nu + k] = prob[win];
      sel[win] = -1e30F;
    }
    sum += prob[win];
    __syncthreads();
  }
  if (e < nu) {
    const float w = wts[t * nu + e] / (sum + 1e-20F) * wscale;
    const bool drop = prune > 0.0F && w < prune * wscale;
    wts[t * nu + e] = drop ? 0.0F : w;
    if (drop)
      ids[t * nu + e] = -1;
  }
}

__global__ void Situ(const float* gu, const int* group, const float* scale,
                     float* act, int ff, float beta, float lb) {
  const int item = blockIdx.x;
  const float s =
      scale != nullptr && group[item] >= 0 ? scale[group[item]] : 1.0F;
  for (int j = threadIdx.x; j < ff; j += blockDim.x) {
    const float g = gu[static_cast<std::size_t>(item) * 2 * ff + j] * s;
    const float u = gu[static_cast<std::size_t>(item) * 2 * ff + ff + j] * s;
    act[static_cast<std::size_t>(item) * ff + j] =
        beta * tanhf(g / beta) * Sigmoid(g) * lb * tanhf(u / lb);
  }
}

__global__ void CountExperts(const int* ids, int items, unsigned int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < items && ids[i] >= 0)
    atomicAdd(&counts[ids[i]], 1U);
}

__global__ void ToHalf(const float* x, __half* y, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    y[i] = __float2half(x[i]);
}

__global__ void SituHalf(const float* gu, const int* group, const float* scale,
                         __half* act, int ff, float beta, float lb) {
  const int item = blockIdx.x;
  const float s = scale != nullptr ? scale[group[item]] : 1.0F;
  for (int j = threadIdx.x; j < ff; j += blockDim.x) {
    const float g = gu[static_cast<std::size_t>(item) * 2 * ff + j] * s;
    const float u = gu[static_cast<std::size_t>(item) * 2 * ff + ff + j] * s;
    act[static_cast<std::size_t>(item) * ff + j] =
        __float2half(beta * tanhf(g / beta) * Sigmoid(g) * lb * tanhf(u / lb));
  }
}

__global__ void SituHalfIn(const __half* gu, __half* act, int ff, float beta,
                           float lb) {
  const int item = blockIdx.x;
  for (int j = threadIdx.x; j < ff; j += blockDim.x) {
    const float g =
        __half2float(gu[static_cast<std::size_t>(item) * 2 * ff + j]);
    const float u =
        __half2float(gu[static_cast<std::size_t>(item) * 2 * ff + ff + j]);
    act[static_cast<std::size_t>(item) * ff + j] =
        __float2half(beta * tanhf(g / beta) * Sigmoid(g) * lb * tanhf(u / lb));
  }
}

// ---- int4 emulation (INKLING_IU4_EMU): rotated symmetric int4 numerics run
// through the existing f16 routed GEMM -------------------------------------
__device__ __forceinline__ float EmuUe4m3(unsigned b) {
  const unsigned e = (b >> 3) & 15U, m = b & 7U;
  if (b == 0x7F)
    return 0.0F;
  return e == 0 ? static_cast<float>(m) * (1.0F / 512.0F)
                : (1.0F + static_cast<float>(m) * 0.125F) *
                      exp2f(static_cast<float>(e) - 7.0F);
}

// y = R v with R = D2 H D1 / 16 over one 256 block held in LDS.
__device__ __forceinline__ void EmuRot256(float* v, const float* s1,
                                          const float* s2) {
  const int t = threadIdx.x;
  v[t] *= s1[t];
  for (int len = 1; len < 256; len <<= 1) {
    __syncthreads();
    if (t < 128) {
      const int i = (t / len) * 2 * len + (t % len);
      const float a = v[i], b = v[i + len];
      v[i] = a + b;
      v[i + len] = a - b;
    }
  }
  __syncthreads();
  v[t] *= s2[t] * (1.0F / 16.0F);
  __syncthreads();
}

// NVFP4 gate/up rows -> rotated symmetric int4 (per-128 scale) as Q8Rep rows.
__global__ void EmuWeights(const std::uint8_t* src, std::uint8_t* dst, int k,
                           std::size_t src_row_bytes, std::size_t dst_row_bytes,
                           const float* s1, const float* s2, bool quant) {
  __shared__ float v[256];
  __shared__ float red[2][8];
  const std::size_t row = blockIdx.x;
  const std::uint8_t* sr = src + row * src_row_bytes;
  std::uint8_t* dr = dst + row * dst_row_bytes;
  const int t = threadIdx.x;
  constexpr float kMags[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  for (int kb = 0; kb < k / 256; ++kb) {
    const int j = kb * 8 + t / 32;
    const int p = t % 32;
    const int byte = (p < 8)    ? p
                     : (p < 16) ? p - 8
                     : (p < 24) ? 8 + p - 16
                                : 8 + p - 24;
    const unsigned raw = sr[j * 16 + byte];
    const unsigned nb =
        (p < 8 || (p >= 16 && p < 24)) ? (raw & 15U) : (raw >> 4);
    const float mag = kMags[nb & 7U];
    const float sc = EmuUe4m3(sr[k / 2 + j * 2 + p / 16]);
    v[t] = ((nb & 8U) ? -mag : mag) * sc;
    __syncthreads();
    EmuRot256(v, s1, s2);
    float a = fabsf(v[t]);
    for (int o = 16; o > 0; o >>= 1)
      a = fmaxf(a, __shfl_xor(a, o));
    if ((t & 31) == 0)
      red[t / 128][(t / 32) % 4] = a;
    __syncthreads();
    const int g = t / 128;
    const float amax =
        fmaxf(fmaxf(red[g][0], red[g][1]), fmaxf(red[g][2], red[g][3]));
    const float s = __half2float(__float2half(amax / 7.0F));
    float q = s > 0.0F ? fminf(7.0F, fmaxf(-8.0F, rintf(v[t] / s))) : 0.0F;
    if (!quant)
      q = fminf(127.0F,
                fmaxf(-128.0F, rintf(v[t] * (127.0F / fmaxf(amax, 1e-30F)))));
    dr[kb * 256 + t] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
    if ((t & 31) == 0) {
      const __half hs = __float2half(quant ? s : amax / 127.0F);
      std::memcpy(dr + k + j * 2, &hs, 2);
    }
    __syncthreads();
  }
}

// Fake-quantize activations: rotate each 256 block, per-32 scale, round to
// int4, return the rotated dequantized values.
__global__ void EmuActs(const __half* x, __half* out, int h, const float* s1,
                        const float* s2, bool quant, bool group128) {
  __shared__ float v[256];
  const std::size_t tok = blockIdx.x;
  const int t = threadIdx.x;
  for (int kb = 0; kb < h / 256; ++kb) {
    v[t] = __half2float(x[tok * h + kb * 256 + t]);
    __syncthreads();
    EmuRot256(v, s1, s2);
    float a = fabsf(v[t]);
    for (int o = 16; o > 0; o >>= 1)
      a = fmaxf(a, __shfl_xor(a, o));
    if (group128) {
      // per-128 scale: combine the four 32-lane groups of each half
      __shared__ float gmax[8];
      if ((t & 31) == 0)
        gmax[t / 32] = a;
      __syncthreads();
      const int g = t / 128;
      a = fmaxf(fmaxf(gmax[g * 4], gmax[g * 4 + 1]),
                fmaxf(gmax[g * 4 + 2], gmax[g * 4 + 3]));
      __syncthreads();
    }
    const float s = __half2float(__float2half(a / 7.0F));
    const float q =
        s > 0.0F ? fminf(7.0F, fmaxf(-8.0F, rintf(v[t] / s))) : 0.0F;
    out[tok * h + kb * 256 + t] =
        quant ? __float2half(q * s) : __float2half(v[t]);
    __syncthreads();
  }
}

// Compact list of the (expert, token tile) pairs that have rows: the launch
// grid is bounded by rows / tile + experts instead of experts * tiles. Unused
// entries point far past any bucket, so those blocks exit at once.
__global__ void BuildTileList(const int* bounds, int ne, int bn, int* out,
                              int max_tiles) {
  __shared__ int start[256];
  const int t = threadIdx.x;
  for (int i = t; i < max_tiles; i += blockDim.x)
    out[i] = 0x7FFF0000;
  int nt = 0;
  if (t < ne)
    nt = (bounds[t + 1] - bounds[t] + bn - 1) / bn;
  start[t] = nt;
  __syncthreads();
  if (t == 0) {
    int run = 0;
    for (int e = 0; e < ne; ++e) {
      const int c = start[e];
      start[e] = run;
      run += c;
    }
  }
  __syncthreads();
  if (t < ne) {
    const int s0 = start[t];
    for (int j = 0; j < nt; ++j)
      if (s0 + j < max_tiles)
        out[s0 + j] = t | (j << 16);
  }
}

// Sums the split-K slices in a fixed order.
__global__ void SplitKReduce(const float* part, float* y, std::size_t count,
                             int slices) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float a = part[i];
  for (int z = 1; z < slices; ++z)
    a += part[static_cast<std::size_t>(z) * count + i];
  y[i] = a;
}

__global__ void SetDenseBounds(int* bounds, int n) {
  bounds[0] = 0;
  bounds[1] = n;
}

__global__ void EmuSitu(const float* gu, const int* ids, const float* es,
                        __half* act, int ff, float beta, float lb) {
  const int item = blockIdx.x;
  const float sc = es != nullptr ? es[ids[item]] : 1.0F;
  for (int j = threadIdx.x; j < ff; j += blockDim.x) {
    const float g = gu[static_cast<std::size_t>(item) * 2 * ff + j] * sc;
    const float u = gu[static_cast<std::size_t>(item) * 2 * ff + ff + j] * sc;
    act[static_cast<std::size_t>(item) * ff + j] =
        __float2half(beta * tanhf(g / beta) * Sigmoid(g) * lb * tanhf(u / lb));
  }
}

__global__ void CombineHalf(const __half* y, const float* wts, float* out,
                            int nu, int h) {
  const int h8 = h / 8;
  for (int i = threadIdx.x; i < h8; i += blockDim.x) {
    float a[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int k = 0; k < nu; ++k) {
      const float w = wts[blockIdx.x * nu + k];
      if (w == 0.0F)
        continue;
      const uint4 raw = reinterpret_cast<const uint4*>(
          y + (static_cast<std::size_t>(blockIdx.x) * nu + k) * h)[i];
      const __half2* pr = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(pr[j]);
        a[2 * j] += w * f.x;
        a[2 * j + 1] += w * f.y;
      }
    }
    float4* o = reinterpret_cast<float4*>(
                    out + static_cast<std::size_t>(blockIdx.x) * h) +
                2 * i;
    o[0] = make_float4(a[0], a[1], a[2], a[3]);
    o[1] = make_float4(a[4], a[5], a[6], a[7]);
  }
}

__global__ void Combine(const float* y, const float* wts, float* out, int nu,
                        int h, const float* down_scale, const int* ids) {
  for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < h;
       i += gridDim.y * blockDim.x) {
    float a = 0.0F;
    for (int k = 0; k < nu; ++k) {
      const int eid = ids != nullptr ? ids[blockIdx.x * nu + k] : 0;
      const float sc =
          down_scale != nullptr && eid >= 0 ? down_scale[eid] : 1.0F;
      a += wts[blockIdx.x * nu + k] * sc *
           y[(static_cast<std::size_t>(blockIdx.x) * nu + k) * h + i];
    }
    out[static_cast<std::size_t>(blockIdx.x) * h + i] = a;
  }
}

__global__ void Softcap(float* x, int count, float cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = cap * tanhf(x[i] / cap);
}

using V16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using V8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kLdsStride = 40;
constexpr int kWmmaThreads = 128;

struct WmmaArgs {
  const unsigned char* w;
  std::size_t row_bytes;
  int cols;
  int rows;
  const __half* x;
  int a_div;
  float* y;
  const float* scale;
  const int* tiles;
  const int* sorted;
  int n;
  int ksplit{
      1};  ///< split-K slices (grid z); partial sums go to y + z * n * rows
  int kchunk{0};  ///< K per slice
};

// 16 consecutive K elements [k0, k0 + 16) of one weight row as F16.
template<int TY>
__device__ void LoadB16(const unsigned char* row, int k0, __half* dst,
                        int cols) {
  if constexpr (TY == kTypeF32) {
    const float* p = reinterpret_cast<const float*>(row) + k0;
    for (int i = 0; i < 16; ++i)
      dst[i] = __float2half(p[i]);
  } else if constexpr (TY == kTypeBf16) {
    const unsigned short* p = reinterpret_cast<const unsigned short*>(row) + k0;
    for (int i = 0; i < 16; ++i)
      dst[i] = __float2half(__uint_as_float(static_cast<unsigned>(p[i]) << 16));
  } else if constexpr (TY == kTypeQ8) {
    const float d = __half2float(
        *reinterpret_cast<const __half*>(row + cols + (k0 >> 5) * 2));
    const uint4 v = *reinterpret_cast<const uint4*>(row + k0);
    const unsigned int u[4] = {v.x, v.y, v.z, v.w};
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 4; ++k) {
        dst[4 * i + k] =
            __float2half(d * static_cast<float>(static_cast<signed char>(
                                 (u[i] >> (8 * k)) & 0xFF)));
      }
    }
  } else {
    const int sb = k0 >> 4;
    const float d = Ue4m3(row[(cols >> 1) + sb]);
    const uint2 q = *reinterpret_cast<const uint2*>(row + sb * 8);
    const unsigned int u[2] = {q.x, q.y};
    for (int j = 0; j < 8; ++j) {
      const unsigned int byte = (u[j >> 2] >> ((j & 3) * 8)) & 0xFF;
      dst[j] = __float2half(E2m1(byte & 15) * d);
      dst[8 + j] = __float2half(E2m1(byte >> 4) * d);
    }
  }
}

// Y[item][n] = scale[e] * sum_k X[item / a_div][k] * W[e*rows + n][k]; 64x64
// tile per block, 4 waves of 32x32.
template<int TY>
__global__ void __launch_bounds__(kWmmaThreads) WmmaGemm(WmmaArgs a) {
  __shared__ __half As[kTileM][kLdsStride];
  __shared__ __half Bs[kTileN][kLdsStride];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int n0 = blockIdx.x * kTileN;
  int expert = 0;
  int start;
  int cnt;
  if (a.tiles != nullptr) {
    expert = a.tiles[blockIdx.y * 3];
    start = a.tiles[blockIdx.y * 3 + 1];
    cnt = a.tiles[blockIdx.y * 3 + 2];
  } else {
    start = blockIdx.y * kTileM;
    cnt = min(kTileM, a.n - start);
  }
  if (cnt <= 0)
    return;
  const int lr = tid >> 1;
  const int lh = tid & 1;
  const int item =
      lr < cnt ? (a.sorted != nullptr ? a.sorted[start + lr] : start + lr) : -1;
  const __half* xr =
      item >= 0 ? a.x + static_cast<std::size_t>(item / a.a_div) * a.cols
                : nullptr;
  const unsigned char* wr =
      a.w + (static_cast<std::size_t>(expert) * a.rows + n0 + lr) * a.row_bytes;
  V8f acc[2][2] = {};
  const int wm = wave >> 1;
  const int wn = wave & 1;
  const int kbeg = a.ksplit > 1 ? static_cast<int>(blockIdx.z) * a.kchunk : 0;
  const int kend = a.ksplit > 1 ? min(a.cols, kbeg + a.kchunk) : a.cols;
  float* const yout =
      a.ksplit > 1 ? a.y + static_cast<std::size_t>(blockIdx.z) * a.n * a.rows
                   : a.y;
  for (int k0 = kbeg; k0 < kend; k0 += 32) {
    uint4* ad = reinterpret_cast<uint4*>(&As[lr][lh * 16]);
    if (xr != nullptr) {
      const uint4* xs = reinterpret_cast<const uint4*>(xr + k0 + lh * 16);
      ad[0] = xs[0];
      ad[1] = xs[1];
    } else {
      ad[0] = make_uint4(0, 0, 0, 0);
      ad[1] = make_uint4(0, 0, 0, 0);
    }
    LoadB16<TY>(wr, k0 + lh * 16, &Bs[lr][lh * 16], a.cols);
    __syncthreads();
    for (int kk = 0; kk < 2; ++kk) {
      V16h af[2];
      V16h bf[2];
      for (int mi = 0; mi < 2; ++mi) {
        const __half* p = &As[wm * 32 + mi * 16 + (lane & 15)][kk * 16];
        for (int i = 0; i < 16; ++i)
          af[mi][i] = static_cast<_Float16>(__half2float(p[i]));
      }
      for (int ni = 0; ni < 2; ++ni) {
        const __half* p = &Bs[wn * 32 + ni * 16 + (lane & 15)][kk * 16];
        for (int i = 0; i < 16; ++i)
          bf[ni][i] = static_cast<_Float16>(__half2float(p[i]));
      }
      for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni)
          acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              af[mi], bf[ni], acc[mi][ni]);
      }
    }
    __syncthreads();
  }
  const float sc = a.scale != nullptr ? a.scale[expert] : 1.0F;
  for (int mi = 0; mi < 2; ++mi) {
    for (int ni = 0; ni < 2; ++ni) {
      for (int i = 0; i < 8; ++i) {
        const int m = wm * 32 + mi * 16 + 2 * i + (lane >> 4);
        const int nn = n0 + wn * 32 + ni * 16 + (lane & 15);
        if (m >= cnt)
          continue;
        const int out_item =
            a.sorted != nullptr ? a.sorted[start + m] : start + m;
        yout[static_cast<std::size_t>(out_item) * a.rows + nn] =
            acc[mi][ni][i] * sc;
      }
    }
  }
}

// Prefill loop control with the gain and forget GEMMs (K = kControlRank, F32
// weights) fused with the update cur += -tanh(forget) * cur + tanh(gain) * msg:
// same fp16 operands and fp32 accumulation as two WmmaGemm launches, without
// writing and re-reading the two fp32 tensors.
__global__ void __launch_bounds__(kWmmaThreads)
    ControlGemm(const __half* x, const float* gw, const float* fw, float* cur,
                const float* msg, int h, int n) {
  __shared__ __half As[kTileM][kLdsStride];
  __shared__ __half Bg[kTileN][kLdsStride];
  __shared__ __half Bf[kTileN][kLdsStride];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int n0 = blockIdx.x * kTileN;
  const int start = blockIdx.y * kTileM;
  const int cnt = min(kTileM, n - start);
  if (cnt <= 0)
    return;
  const int lr = tid >> 1;
  const int lh = tid & 1;
  const __half* xr =
      lr < cnt ? x + static_cast<std::size_t>(start + lr) * kControlRank
               : nullptr;
  const unsigned char* gr = reinterpret_cast<const unsigned char*>(
      gw + static_cast<std::size_t>(n0 + lr) * kControlRank);
  const unsigned char* fr = reinterpret_cast<const unsigned char*>(
      fw + static_cast<std::size_t>(n0 + lr) * kControlRank);
  V8f ag[2][2] = {};
  V8f af_[2][2] = {};
  const int wm = wave >> 1;
  const int wn = wave & 1;
  for (int k0 = 0; k0 < kControlRank; k0 += 32) {
    uint4* ad = reinterpret_cast<uint4*>(&As[lr][lh * 16]);
    if (xr != nullptr) {
      const uint4* xs = reinterpret_cast<const uint4*>(xr + k0 + lh * 16);
      ad[0] = xs[0];
      ad[1] = xs[1];
    } else {
      ad[0] = make_uint4(0, 0, 0, 0);
      ad[1] = make_uint4(0, 0, 0, 0);
    }
    LoadB16<kTypeF32>(gr, k0 + lh * 16, &Bg[lr][lh * 16], kControlRank);
    LoadB16<kTypeF32>(fr, k0 + lh * 16, &Bf[lr][lh * 16], kControlRank);
    __syncthreads();
    for (int kk = 0; kk < 2; ++kk) {
      V16h xa[2];
      V16h bg[2];
      V16h bf[2];
      for (int mi = 0; mi < 2; ++mi) {
        const __half* p = &As[wm * 32 + mi * 16 + (lane & 15)][kk * 16];
        for (int i = 0; i < 16; ++i)
          xa[mi][i] = static_cast<_Float16>(__half2float(p[i]));
      }
      for (int ni = 0; ni < 2; ++ni) {
        const __half* pg = &Bg[wn * 32 + ni * 16 + (lane & 15)][kk * 16];
        const __half* pf = &Bf[wn * 32 + ni * 16 + (lane & 15)][kk * 16];
        for (int i = 0; i < 16; ++i) {
          bg[ni][i] = static_cast<_Float16>(__half2float(pg[i]));
          bf[ni][i] = static_cast<_Float16>(__half2float(pf[i]));
        }
      }
      for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
          ag[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              xa[mi], bg[ni], ag[mi][ni]);
          af_[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              xa[mi], bf[ni], af_[mi][ni]);
        }
      }
    }
    __syncthreads();
  }
  for (int mi = 0; mi < 2; ++mi) {
    for (int ni = 0; ni < 2; ++ni) {
      for (int i = 0; i < 8; ++i) {
        const int m = wm * 32 + mi * 16 + 2 * i + (lane >> 4);
        const int nn = n0 + wn * 32 + ni * 16 + (lane & 15);
        if (m >= cnt)
          continue;
        const std::size_t idx = static_cast<std::size_t>(start + m) * h + nn;
        const float c = cur[idx];
        cur[idx] = c - tanhf(af_[mi][ni][i] * 1.0F) * c +
                   tanhf(ag[mi][ni][i] * 1.0F) * msg[idx];
      }
    }
  }
}

__global__ void SortItems(const int* ids, int items, int ne, int* sorted,
                          int* tiles, int max_tiles) {
  __shared__ int cnt[256];
  __shared__ int start[256];
  __shared__ int cursor[256];
  for (int e = threadIdx.x; e < ne; e += blockDim.x) {
    cnt[e] = 0;
    cursor[e] = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < items; i += blockDim.x)
    atomicAdd(&cnt[ids[i]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    int k = 0;
    for (int e = 0; e < ne; ++e) {
      start[e] = s;
      for (int off = 0; off < cnt[e]; off += kTileM) {
        tiles[3 * k] = e;
        tiles[3 * k + 1] = s + off;
        tiles[3 * k + 2] = min(kTileM, cnt[e] - off);
        ++k;
      }
      s += cnt[e];
    }
    for (; k < max_tiles; ++k)
      tiles[3 * k + 2] = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < items; i += blockDim.x) {
    const int e = ids[i];
    sorted[start[e] + atomicAdd(&cursor[e], 1)] = i;
  }
}

constexpr int kSparseSink = 2;
constexpr int kSparseLocal = 16;
constexpr int kSparseTop = 192;
constexpr bool kGroupBound = false;
constexpr int kListCap = kSparseSink + kSparseLocal + kSparseTop + 4;
constexpr int kBlockKeys = 64;
constexpr int kMaxSparseBlocks = 4096;
constexpr std::uint32_t kSparseAutoContext = 65536;

constexpr int kPackWaves = 4;
constexpr int kFlashWaves = 8;
constexpr int kFlashThreads = 32 * kFlashWaves;
constexpr int kFlashQ = 16 * kFlashWaves;
constexpr int kFlashK = 64;
constexpr int kKStride = 136;
constexpr int kVStride = 72;
constexpr int kPStride = 72;

constexpr int kZStride = 88;
constexpr int kSplitKeys = 2048;
constexpr int kFlashRel = 16;
constexpr float kNegInf = -1e30F;
constexpr float kLog2e = 1.4426950408889634F;
constexpr float kLn2 = 0.6931471805599453F;

// Flash attention with relative bias and a zero sink logit. One block: 64
// queries of one head; 4 waves x 16 query rows.
template<int WAVES>
__global__ void __launch_bounds__(WAVES * 32)
    FlashRel(AttnArgs a, const float* q, const float* r, const float* sink,
             const __half* kc, const __half* vc, const __half* bias, float* out,
             int n, int nsplit, int split_keys, float* pm, float* pl, float* po,
             const int* blist, const int* bcount, int list_cap,
             const float* gate = nullptr, __half* out_h = nullptr) {
  __shared__ __half Ks[kFlashK][kKStride];
  __shared__ __half Vt[kFlashHd][kVStride];
  // per wave: the bias rows Z (consumed by the score epilogue) and then the P
  // rows share one region
  __shared__ __half PZ[WAVES][16 * kZStride];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int gh = a.nh / a.nkv;
  const int pk = a.pack;
  const int h0 = blockIdx.y;
  // Full-attention work grows with the query block: launch the longest first
  // so no long block starts last.
  const int qblock = (pk == 0 && blist == nullptr && a.window == 0)
                         ? static_cast<int>(gridDim.x) - 1 - blockIdx.x
                         : blockIdx.x;
  const int q0 = pk > 0 ? 0 : qblock * (WAVES * 16);
  const int g = pk > 0 ? blockIdx.y : h0 / gh;
  const int nrows = pk > 0 ? n * gh : n;
  const auto row_tok = [&](int row) {
    return min(pk > 0 ? row / gh : q0 + row, n - 1);
  };
  const auto row_head = [&](int row) {
    return pk > 0 ? g * gh + row % gh : h0;
  };
  const int kvdim = a.nkv * a.hd;
  const int pos0 = *a.pos0;
  const int qmax_pos = pos0 + (pk > 0 ? n : min(q0 + (WAVES * 16), n)) - 1;
  const int qmin_pos = pos0 + q0;
  // Q fragments for this wave's 16 rows
  V16h qf[8];
  {
    const int row = min(wave * 16 + (lane & 15), nrows - 1);
    const int t = row_tok(row);
    const int h = row_head(row);
    const float* qp = q + (static_cast<std::size_t>(t) * a.nh + h) * kFlashHd;
    for (int c = 0; c < 8; ++c) {
      for (int i = 0; i < 16; ++i)
        qf[c][i] = static_cast<_Float16>(qp[c * 16 + i]);
    }
  }
  V16h rf;
  {
    const int row = min(wave * 16 + (lane & 15), nrows - 1);
    const int t = row_tok(row);
    const int h = row_head(row);
    const float* rp = r + (static_cast<std::size_t>(t) * a.nh + h) * kFlashRel;
    for (int i = 0; i < 16; ++i)
      rf[i] = static_cast<_Float16>(rp[i]);
  }
  const int qp0 = pos0 + (pk > 0 ? (wave * 16) / gh : q0 + wave * 16);
  float sinkv[8];
  int qpos[8];
  for (int i = 0; i < 8; ++i) {
    const int row = wave * 16 + 2 * i + (lane >> 4);
    const int t = row_tok(row);
    sinkv[i] =
        sink[static_cast<std::size_t>(t) * a.nh + row_head(row)] * kLog2e;
    qpos[i] = pos0 + (pk > 0 ? row / gh : q0 + row);
  }
  V8f o[8] = {};
  float m[8];
  float l[8];
  const int split = blockIdx.z;
  for (int i = 0; i < 8; ++i) {
    // the zero sink logit belongs to split 0 only (counted once after the
    // 16-lane sum)
    m[i] = split == 0 ? 0.0F : kNegInf;
    l[i] = split == 0 ? 1.0F / 16.0F : 0.0F;
  }
  const int kfirst_pos = a.window > 0 ? max(0, qmin_pos - a.window + 1) : 0;
  int kb_begin = (kfirst_pos / kFlashK) * kFlashK;
  int it_begin = 0;
  int it_end = 0;
  const int* list = nullptr;
  if (blist != nullptr) {
    const int tile_head = blockIdx.x * a.nh + (pk > 0 ? g : h0);
    list = blist + static_cast<std::size_t>(tile_head) * list_cap;
    const int count = bcount[tile_head];
    const int per = (list_cap + nsplit - 1) / nsplit;
    it_begin = split * per;
    it_end = min(count, (split + 1) * per);
  } else {
    int kb_end = qmax_pos;
    if (nsplit > 1 && a.window > 0) {
      const int base = kb_begin;
      kb_begin = base + split * split_keys;
      kb_end = min(kb_end, base + (split + 1) * split_keys - 1);
    } else if (nsplit > 1) {
      kb_begin = max(kb_begin, split * split_keys);
      kb_end = min(kb_end, (split + 1) * split_keys - 1);
    }
    it_end = kb_end >= kb_begin ? (kb_end - kb_begin) / kFlashK + 1 : 0;
  }
  for (int it = it_begin; it < it_end; ++it) {
    const int kb =
        list != nullptr ? list[it] * kFlashK : kb_begin + it * kFlashK;
    __syncthreads();
    {
      constexpr int kKPer = (WAVES * 32) / kFlashK;  // threads per key
      constexpr int kKU4 = kFlashHd / kKPer / 8;     // uint4 per thread
      const int key = tid / kKPer;
      const int part = tid % kKPer;
      const int j = kb + key;
      const bool ok = j <= qmax_pos;
      const std::size_t jr = a.ring > 0 ? j % a.ring : j;
      const uint4* ks = reinterpret_cast<const uint4*>(
          kc + jr * kvdim + g * kFlashHd + part * (kFlashHd / kKPer));
      uint4* kd = reinterpret_cast<uint4*>(&Ks[key][part * (kFlashHd / kKPer)]);
      for (int i = 0; i < kKU4; ++i)
        kd[i] = ok ? ks[i] : make_uint4(0, 0, 0, 0);
      // V^T tile: rows of 64 keys per head dimension (keys past the causal
      // limit get zero probability, the cache is zero-initialised)
      constexpr int kVPer = (WAVES * 32) / kFlashHd;  // threads per row
      constexpr int kVU4 = 8 / kVPer;
      const int vrow = tid / kVPer;
      const int vpart = tid % kVPer;
      const int vb = (a.ring > 0 ? (kb % a.ring) : kb) >> 6;
      const uint4* vsrc =
          reinterpret_cast<const uint4*>(
              vc + ((static_cast<std::size_t>(g) * a.vblocks + vb) * kFlashHd +
                    vrow) *
                       64) +
          vpart * kVU4;
      uint4* vdst = reinterpret_cast<uint4*>(&Vt[vrow][0]) + vpart * kVU4;
      for (int i = 0; i < kVU4; ++i)
        vdst[i] = vsrc[i];
    }
    {
      // relative-position bias of this wave's rows against the tile's distances
      const int dmin = qp0 - (kb + kFlashK - 1);
      for (int nt = 0; nt < 5; ++nt) {
        const int dist = dmin + nt * 16 + (lane & 15);
        const int idx = (dist < 0 || dist >= a.extent) ? a.extent : dist;
        V16h bt;
        std::memcpy(&bt, bias + static_cast<std::size_t>(idx) * kFlashRel,
                    sizeof(V16h));
        V8f z = {};
        z = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(rf, bt, z);
        for (int i = 0; i < 8; ++i)
          PZ[wave][(2 * i + (lane >> 4)) * kZStride + nt * 16 + (lane & 15)] =
              __float2half(z[i]);
      }
    }
    __syncthreads();
    V8f sc[4] = {};
    for (int c = 0; c < 8; ++c) {
      for (int ni = 0; ni < 4; ++ni) {
        V16h bf;
        std::memcpy(&bf, &Ks[ni * 16 + (lane & 15)][c * 16], sizeof(V16h));
        sc[ni] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], bf, sc[ni]);
      }
    }
    const float scale_l2 = a.scale * kLog2e;
    float pmax[8];
    for (int i = 0; i < 8; ++i)
      pmax[i] = kNegInf;
    // every score of the wave's 16 rows lies inside the causal window and
    // the bias extent
    const bool interior = pk == 0 && qp0 >= kb + kFlashK - 1 &&
                          qp0 + 15 - kb < a.extent &&
                          (a.window == 0 || qp0 + 15 - kb < a.window);
    for (int ni = 0; ni < 4; ++ni) {
      const int jpos = kb + ni * 16 + (lane & 15);
      for (int i = 0; i < 8; ++i) {
        const int dist = qpos[i] - jpos;
        const bool valid =
            interior || (dist >= 0 && (a.window == 0 || dist < a.window));
        float v = kNegInf;
        if (valid) {
          const int lr = 2 * i + (lane >> 4);
          const float b =
              (interior || dist < a.extent)
                  ? __half2float(PZ[wave][lr * kZStride + dist -
                                          (qp0 - (kb + kFlashK - 1))])
                  : 0.0F;
          v = sc[ni][i] * scale_l2 + b * kLog2e - sinkv[i];
        }
        sc[ni][i] = v;
        pmax[i] = fmaxf(pmax[i], v);
      }
    }
    for (int i = 0; i < 8; ++i) {
      for (int off = 8; off > 0; off >>= 1)
        pmax[i] = fmaxf(pmax[i], __shfl_xor(pmax[i], off));
      const float mn = fmaxf(m[i], pmax[i]);
      const float alpha = exp2f(m[i] - mn);
      m[i] = mn;
      l[i] *= alpha;
      for (int ni = 0; ni < 8; ++ni)
        o[ni][i] *= alpha;
    }
    for (int ni = 0; ni < 4; ++ni) {
      for (int i = 0; i < 8; ++i) {
        const float pv = (!interior && sc[ni][i] <= kNegInf)
                             ? 0.0F
                             : exp2f(sc[ni][i] - m[i]);
        l[i] += pv;
        PZ[wave][(2 * i + (lane >> 4)) * kPStride + ni * 16 + (lane & 15)] =
            __float2half(pv);
      }
    }
    // the P rows are wave-private: LDS accesses of one wave stay in order
    __builtin_amdgcn_wave_barrier();
    for (int c = 0; c < 4; ++c) {
      V16h pf;
      std::memcpy(&pf, &PZ[wave][(lane & 15) * kPStride + c * 16],
                  sizeof(V16h));
      for (int ni = 0; ni < 8; ++ni) {
        V16h vf;
        std::memcpy(&vf, &Vt[ni * 16 + (lane & 15)][c * 16], sizeof(V16h));
        o[ni] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pf, vf, o[ni]);
      }
    }
  }
  for (int i = 0; i < 8; ++i) {
    float ls = l[i];
    for (int off = 8; off > 0; off >>= 1)
      ls += __shfl_xor(ls, off);
    const int orow = wave * 16 + 2 * i + (lane >> 4);
    const int t = pk > 0 ? orow / gh : q0 + orow;
    if (pk > 0 ? orow >= nrows : t >= n)
      continue;
    const int h = row_head(orow);
    if (nsplit > 1) {
      const std::size_t part =
          (static_cast<std::size_t>(t) * a.nh + h) * nsplit + split;
      if ((lane & 15) == 0) {
        pm[part] = m[i] * kLn2;
        pl[part] = ls;
      }
      for (int ni = 0; ni < 8; ++ni)
        po[part * kFlashHd + ni * 16 + (lane & 15)] = o[ni][i];
      continue;
    }
    if (out_h != nullptr) {
      // the sigmoid gate of the attention output, half rows only
      for (int ni = 0; ni < 8; ++ni) {
        const std::size_t at =
            (static_cast<std::size_t>(t) * a.nh + h) * kFlashHd + ni * 16 +
            (lane & 15);
        out_h[at] = __float2half((o[ni][i] / ls) * Sigmoid(gate[at]));
      }
      continue;
    }
    for (int ni = 0; ni < 8; ++ni) {
      out[(static_cast<std::size_t>(t) * a.nh + h) * kFlashHd + ni * 16 +
          (lane & 15)] = o[ni][i] / ls;
    }
  }
}

constexpr int kDecThreads = 256;
constexpr int kDecKeys = 256;
constexpr int kDecMaxGh = 8;

// Single-token attention: one block per (key split, kv group); the group's
// query heads share every K/V load. Writes unnormalised partials for
// FlashCombine.
__global__ void __launch_bounds__(kDecThreads)
    DecodeAttn(AttnArgs a, const float* q, const float* r, const float* sink,
               const __half* kc, const __half* vc, const __half* bias,
               float* pm, float* pl, float* po, int nsplit) {
  __shared__ float qs[kDecMaxGh][kFlashHd];
  __shared__ float rs[kDecMaxGh][kFlashRel];
  __shared__ float sks[kDecMaxGh];
  __shared__ float sc[kDecMaxGh][kDecKeys];
  __shared__ float red[kDecThreads / 32][kDecMaxGh];
  __shared__ float ost[kDecMaxGh][kFlashHd];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int split = blockIdx.x;
  const int g = blockIdx.y;
  const int gh = a.nh / a.nkv;
  const int pos = *a.pos0;
  const int first = a.window > 0 ? max(0, pos - a.window + 1) : 0;
  const int lo = first + split * kDecKeys;
  const int hi = min(pos, lo + kDecKeys - 1);
  const int nk = hi >= lo ? hi - lo + 1 : 0;
  const int kvdim = a.nkv * kFlashHd;
  for (int i = tid; i < gh * kFlashHd; i += kDecThreads)
    qs[i / kFlashHd][i % kFlashHd] = __half2float(
        __float2half(q[static_cast<std::size_t>(g * gh) * kFlashHd + i]));
  for (int i = tid; i < gh * kFlashRel; i += kDecThreads)
    rs[i / kFlashRel][i % kFlashRel] = __half2float(
        __float2half(r[static_cast<std::size_t>(g * gh) * kFlashRel + i]));
  if (tid < gh)
    sks[tid] = sink[g * gh + tid];
  __syncthreads();
  float pmax[kDecMaxGh];
  for (int h = 0; h < kDecMaxGh; ++h)
    pmax[h] = kNegInf;
  const int l16 = tid & 15;
  const int kslot = tid >> 4;
  float qreg[kDecMaxGh][8];
  float rreg[kDecMaxGh];
  for (int h = 0; h < kDecMaxGh; ++h) {
    rreg[h] = h < gh ? rs[h][l16] : 0.0F;
    for (int e = 0; e < 8; ++e)
      qreg[h][e] = h < gh ? qs[h][l16 * 8 + e] : 0.0F;
  }
  for (int base = 0; base < nk; base += 16) {
    const int kk = base + kslot;
    const bool valid = kk < nk;
    const int j = lo + (valid ? kk : 0);
    const std::size_t jr = a.ring > 0 ? j % a.ring : j;
    const uint4 w =
        reinterpret_cast<const uint4*>(kc + jr * kvdim + g * kFlashHd)[l16];
    const __half2* hp = reinterpret_cast<const __half2*>(&w);
    float kf[8];
    for (int e = 0; e < 4; ++e) {
      const float2 f = __half22float2(hp[e]);
      kf[2 * e] = f.x;
      kf[2 * e + 1] = f.y;
    }
    const int dist = pos - j;
    const float bv =
        dist < a.extent
            ? __half2float(
                  bias[static_cast<std::size_t>(dist) * kFlashRel + l16])
            : 0.0F;
    float acc[kDecMaxGh];
    float z[kDecMaxGh];
    for (int h = 0; h < kDecMaxGh; ++h) {
      float t = 0.0F;
      for (int e = 0; e < 8; ++e)
        t += qreg[h][e] * kf[e];
      acc[h] = t;
      z[h] = rreg[h] * bv;
    }
    for (int off = 8; off > 0; off >>= 1) {
      for (int h = 0; h < kDecMaxGh; ++h) {
        acc[h] += __shfl_xor(acc[h], off);
        z[h] += __shfl_xor(z[h], off);
      }
    }
    for (int h = 0; h < kDecMaxGh; ++h) {
      if (h >= gh || !valid)
        continue;
      const float b = dist < a.extent ? __half2float(__float2half(z[h])) : 0.0F;
      const float v = acc[h] * a.scale + b - sks[h];
      if (l16 == h)
        sc[h][kk] = v;
      pmax[h] = fmaxf(pmax[h], v);
    }
  }
  for (int h = 0; h < kDecMaxGh; ++h) {
    float v = pmax[h];
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor(v, off));
    if (lane == 0)
      red[wave][h] = v;
  }
  __syncthreads();
  float m[kDecMaxGh];
  for (int h = 0; h < kDecMaxGh; ++h) {
    float v = split == 0 ? 0.0F : kNegInf;
    for (int w = 0; w < kDecThreads / 32; ++w)
      v = fmaxf(v, red[w][h]);
    m[h] = v;
  }
  __syncthreads();
  float lsum[kDecMaxGh] = {};
  for (int kk = tid; kk < nk; kk += kDecThreads) {
    for (int h = 0; h < kDecMaxGh; ++h) {
      if (h >= gh)
        continue;
      const float p = expf(sc[h][kk] - m[h]);
      lsum[h] += p;
      sc[h][kk] = __half2float(__float2half(p));
    }
  }
  for (int h = 0; h < kDecMaxGh; ++h) {
    float v = lsum[h];
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_xor(v, off);
    if (lane == 0)
      red[wave][h] = v;
  }
  __syncthreads();
  float l[kDecMaxGh];
  for (int h = 0; h < kDecMaxGh; ++h) {
    float v = split == 0 ? expf(-m[h]) : 0.0F;
    for (int w = 0; w < kDecThreads / 32; ++w)
      v += red[w][h];
    l[h] = v;
  }
  const int d = tid & (kFlashHd - 1);
  const int part = tid / kFlashHd;
  float o[kDecMaxGh] = {};
#pragma unroll 4
  for (int jb = ((lo >> 3) << 3) + part * 8; jb <= hi; jb += 16) {
    const std::size_t jr = a.ring > 0 ? jb % a.ring : jb;
    const uint4 w = *reinterpret_cast<const uint4*>(
        vc +
        ((static_cast<std::size_t>(g) * a.vblocks + (jr >> 6)) * kFlashHd + d) *
            64 +
        (jr & 63));
    const __half* hv = reinterpret_cast<const __half*>(&w);
    for (int e = 0; e < 8; ++e) {
      const int j = jb + e;
      if (j < lo || j > hi)
        continue;
      const float vv = __half2float(hv[e]);
      for (int h = 0; h < kDecMaxGh; ++h)
        if (h < gh)
          o[h] += sc[h][j - lo] * vv;
    }
  }
  if (part == 1)
    for (int h = 0; h < kDecMaxGh; ++h)
      ost[h][d] = o[h];
  __syncthreads();
  if (part == 0) {
    for (int h = 0; h < kDecMaxGh; ++h) {
      if (h >= gh)
        continue;
      const std::size_t idx =
          static_cast<std::size_t>(g * gh + h) * nsplit + split;
      po[idx * kFlashHd + d] = o[h] + ost[h][d];
      if (d == 0) {
        pm[idx] = m[h];
        pl[idx] = l[h];
      }
    }
  }
}

// pooled per-block key bounds: kmm[b][g][0|1][128] = min / max over the 64 keys
__global__ void PoolKeys(const __half* kc, __half* kmm, const int* pos0, int n,
                         int nkv) {
  const int pos = *pos0;
  const int b = pos / kBlockKeys + blockIdx.x;
  if (b * kBlockKeys > pos + n - 1)
    return;
  const int g = blockIdx.y;
  const int d = threadIdx.x;
  const int total = pos + n;
  float mn = 1e30F;
  float mx = -1e30F;
  for (int j = b * kBlockKeys; j < min((b + 1) * kBlockKeys, total); ++j) {
    const float v = __half2float(
        kc[(static_cast<std::size_t>(j) * nkv + g) * kFlashHd + d]);
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  kmm[((static_cast<std::size_t>(b) * nkv + g) * 2) * kFlashHd + d] =
      __float2half(mn);
  kmm[((static_cast<std::size_t>(b) * nkv + g) * 2 + 1) * kFlashHd + d] =
      __float2half(mx);
}

// one block per (query tile of 64, head): sink + local + top-k older key blocks
// by the Quest bound of the tile's mean query
__global__ void SelectBlocks(const float* q, const __half* kmm, const int* pos0,
                             int n, int nh, int nkv, int* blist, int* bcount,
                             int pack) {
  extern __shared__ float score[];
  __shared__ float qbar[kFlashHd];
  __shared__ float qlo[kFlashHd];
  __shared__ float qhi[kFlashHd];
  __shared__ int taken;
  __shared__ float lo_t;
  __shared__ float hi_t;
  __shared__ int cnt;
  const int tile = blockIdx.x;
  const int h = blockIdx.y;
  const int gh = nh / nkv;
  const int g = pack > 0 ? h : h / gh;
  const int pos = *pos0;
  const int rows = min(kFlashQ, n - tile * kFlashQ);
  const int tile_first = pos + tile * kFlashQ;
  const int tile_last = tile_first + rows - 1;
  const int first_blk = tile_first / kBlockKeys;
  const int last_blk = tile_last / kBlockKeys;
  const int local_start = max(first_blk - kSparseLocal, kSparseSink);
  const int ncand = max(local_start - kSparseSink, 0);
  int* out = blist + static_cast<std::size_t>(tile * nh + h) * kListCap;
  for (int d = threadIdx.x; d < kFlashHd; d += blockDim.x) {
    float a = 0.0F;
    float lo = 1e30F;
    float hi = -1e30F;
    const int heads = pack > 0 ? gh : 1;
    for (int t = 0; t < rows; ++t) {
      for (int hl = 0; hl < heads; ++hl) {
        const float v = q[(static_cast<std::size_t>(tile * kFlashQ + t) * nh +
                           (pack > 0 ? g * gh + hl : h)) *
                              kFlashHd +
                          d];
        a += v;
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
      }
    }
    qbar[d] = a / (rows * heads);
    qlo[d] = lo;
    qhi[d] = hi;
  }
  if (threadIdx.x == 0) {
    taken = 0;
  }
  __syncthreads();
  auto push = [&](int blk) {
    const int slot = atomicAdd(&taken, 1);
    if (slot < kListCap)
      out[slot] = blk;
  };
  for (int b = threadIdx.x; b < min(kSparseSink, last_blk + 1); b += blockDim.x)
    push(b);
  if (ncand <= kSparseTop) {
    for (int i = threadIdx.x; i < ncand; i += blockDim.x)
      push(kSparseSink + i);
  } else {
    for (int i = threadIdx.x; i < ncand; i += blockDim.x) {
      const __half* mm =
          kmm + ((static_cast<std::size_t>(kSparseSink + i) * nkv + g) * 2) *
                    kFlashHd;
      float sc = 0.0F;
      for (int d = 0; d < kFlashHd; ++d) {
        const float k0 = __half2float(mm[d]);
        const float k1 = __half2float(mm[kFlashHd + d]);
        if constexpr (kGroupBound) {
          sc += fmaxf(fmaxf(qlo[d] * k0, qlo[d] * k1),
                      fmaxf(qhi[d] * k0, qhi[d] * k1));
        } else {
          const float qd = qbar[d];
          sc += fmaxf(qd * k0, qd * k1);
        }
      }
      score[i] = sc;
    }
    __syncthreads();
    // bisection on the threshold value over [min, max] of the candidate scores
    if (threadIdx.x == 0) {
      float mn = 1e30F;
      float mx = -1e30F;
      for (int i = 0; i < ncand; ++i) {
        mn = fminf(mn, score[i]);
        mx = fmaxf(mx, score[i]);
      }
      lo_t = mn;
      hi_t = mx;
    }
    __syncthreads();
    for (int it = 0; it < 24; ++it) {
      if (threadIdx.x == 0)
        cnt = 0;
      __syncthreads();
      const float mid = 0.5F * (lo_t + hi_t);
      int local = 0;
      for (int i = threadIdx.x; i < ncand; i += blockDim.x)
        local += score[i] >= mid ? 1 : 0;
      atomicAdd(&cnt, local);
      __syncthreads();
      if (threadIdx.x == 0) {
        if (cnt > kSparseTop)
          lo_t = mid;
        else
          hi_t = mid;
      }
      __syncthreads();
    }
    // hi_t keeps at most kSparseTop candidates; take everything at or above it
    for (int i = threadIdx.x; i < ncand; i += blockDim.x) {
      if (score[i] >= hi_t)
        push(kSparseSink + i);
    }
  }
  for (int b = local_start + threadIdx.x; b <= last_blk; b += blockDim.x)
    push(b);
  __syncthreads();
  if (threadIdx.x == 0)
    bcount[tile * nh + h] = min(taken, kListCap);
}

__global__ void FlashCombine(const float* pm, const float* pl, const float* po,
                             float* out, int nh, int nsplit) {
  const std::size_t base =
      (static_cast<std::size_t>(blockIdx.x) * nh + blockIdx.y) * nsplit;
  float mx = kNegInf;
  for (int s = 0; s < nsplit; ++s)
    mx = fmaxf(mx, pm[base + s]);
  float den = 0.0F;
  for (int s = 0; s < nsplit; ++s)
    den += expf(pm[base + s] - mx) * pl[base + s];
  for (int d = threadIdx.x; d < kFlashHd; d += blockDim.x) {
    float acc = 0.0F;
    for (int s = 0; s < nsplit; ++s)
      acc += expf(pm[base + s] - mx) * po[(base + s) * kFlashHd + d];
    out[(static_cast<std::size_t>(blockIdx.x) * nh + blockIdx.y) * kFlashHd +
        d] = acc / den;
  }
}

// Q8_0 (34 B blocks) / NVFP4 (36 B blocks) rows -> [qs][scales] rows padded to
// 16 bytes
__device__ float Ue4m3Raw(int code) {
  const int e = (code >> 3) & 0xF;
  const int m = code & 7;
  return e == 0 ? ldexpf(static_cast<float>(m), -9)
                : ldexpf(1.0F + static_cast<float>(m) / 8.0F, e - 7);
}

// true E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}; returns the nibble
__device__ int Fp4Nearest(float v) {
  const float a = fabsf(v);
  int best = 0;
  float err = a;
  const float levels[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  for (int i = 1; i < 8; ++i) {
    const float d = fabsf(a - levels[i]);
    if (d < err) {
      err = d;
      best = i;
    }
  }
  return best | (v < 0.0F ? 8 : 0);
}

__device__ float Fp4Value(int nib) {
  const float levels[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float v = levels[nib & 7];
  return (nib & 8) != 0 ? -v : v;
}

// Q8 rows (repacked: [qs cols][fp16 scales cols/32]) -> NVFP4 rows (repacked:
// [nibbles cols/2][ue4m3 scales cols/16]); one 16-element group per thread
__global__ void Q8ToNvfp4(const unsigned char* src, std::size_t src_stride,
                          unsigned char* dst, std::size_t dst_stride, int cols,
                          const int* ids = nullptr) {
  const std::size_t srow = ids != nullptr ? ids[blockIdx.x] : blockIdx.x;
  const unsigned char* in = src + srow * src_stride;
  unsigned char* out = dst + static_cast<std::size_t>(blockIdx.x) * dst_stride;
  for (int gidx = threadIdx.x; gidx < cols / 16; gidx += blockDim.x) {
    float x[16];
    const float d = __half2float(
        *reinterpret_cast<const __half*>(in + cols + (gidx / 2) * 2));
    float amax = 0.0F;
    for (int i = 0; i < 16; ++i) {
      x[i] =
          d * static_cast<float>(static_cast<signed char>(in[gidx * 16 + i]));
      amax = fmaxf(amax, fabsf(x[i]));
    }
    int best_code = 0;
    float best_err = 3.0e38F;
    if (amax > 0.0F) {
      const float ideal = amax / 6.0F;
      int c0 = 1;
      float bd = 3.0e38F;
      for (int c = 1; c < 0x7F; ++c) {
        const float dd = fabsf(Ue4m3Raw(c) - ideal);
        if (dd < bd) {
          bd = dd;
          c0 = c;
        }
      }
      for (int c = max(1, c0 - 1); c <= min(0x7E, c0 + 1); ++c) {
        const float sc = Ue4m3Raw(c);
        float err = 0.0F;
        for (int i = 0; i < 16; ++i) {
          const float q = Fp4Value(Fp4Nearest(x[i] / sc)) * sc;
          err += (q - x[i]) * (q - x[i]);
        }
        if (err < best_err) {
          best_err = err;
          best_code = c;
        }
      }
    }
    const float sc = best_code != 0 ? Ue4m3Raw(best_code) : 1.0F;
    out[cols / 2 + gidx] = static_cast<unsigned char>(best_code);
    const int block = gidx / 4;
    const int unit = (gidx % 4) / 2;
    const int half = gidx % 2;
    unsigned char* nib = out + block * 32 + unit * 16 + half * 8;
    for (int i = 0; i < 8; ++i) {
      const int lo = Fp4Nearest(x[i] / sc);
      const int hi = Fp4Nearest(x[i + 8] / sc);
      nib[i] = static_cast<unsigned char>(lo | (hi << 4));
    }
  }
}

__global__ void RepackRows(const unsigned char* src, std::size_t src_stride,
                           unsigned char* dst, std::size_t dst_stride, int cols,
                           bool nvfp4) {
  const unsigned char* in =
      src + static_cast<std::size_t>(blockIdx.x) * src_stride;
  unsigned char* out = dst + static_cast<std::size_t>(blockIdx.x) * dst_stride;
  if (!nvfp4) {
    const int nb = cols / 32;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
      out[cols + b * 2] = in[b * 34];
      out[cols + b * 2 + 1] = in[b * 34 + 1];
      for (int i = 0; i < 32; ++i)
        out[b * 32 + i] = in[b * 34 + 2 + i];
    }
  } else {
    const int nb = cols / 64;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
      for (int i = 0; i < 4; ++i)
        out[cols / 2 + b * 4 + i] = in[b * 36 + i];
      for (int i = 0; i < 32; ++i)
        out[b * 32 + i] = in[b * 36 + 4 + i];
    }
  }
}

struct Tensor {
  void* dev{nullptr};
  int type{kTypeF32};
  std::uint64_t cols{0};
  std::uint64_t rows{0};
  std::size_t row_bytes{0};
};

}  // namespace

struct ModelState {
  Config config;
  std::uint32_t n_phys{0};
  std::unordered_map<std::string, Tensor> tensors;
  std::vector<void*> allocations;
  // per-core-layer message queries (attn, k, v, ffn) laid out [cc][4][h]
  float* mq_table{nullptr};
  std::size_t resident_bytes{0};
  bool has_draft{false};
  bool d2{false};          // a DFlash2 block drafter is loaded (tensors "d2.*")
  bool has_dspark{false};  // the DSpark drafter is loaded (tensors "d.*")
  int d2_layers{5}, d2_window{2048}, d2_mask{200056};
  float d2_theta{10000.0F}, d2_softcap{30.0F};
  float draft_fuse[4][4]{};     // softmax of every row of dspark_fuse
  std::vector<int> draft_rank;  // token ids by descending markov_w1 row norm
  // optional NPU offload of the q / gate / k / v projections (INKLING_NPU=1)
  std::unique_ptr<gufo::models::qwen38_flash_next::rocm::NpuGemm> npu;
  int npu_prog_o{-1},
      npu_prog_o_small{-1};  // K=3072 programs for the attention output
  int npu_prog_small{-1};    // M=2048 program for banks of at most 2048 rows
  std::uint32_t npu_tokens{
      2048};  // rows of the runtime-M program = rows of one bank
  std::unordered_map<std::string, std::vector<int>>
      npu_w;  // tensor -> 1024-row slices
  // Rotated int4 copies of the routed experts (INKLING_IU4), built at first use
  // and shared by every session of the model.
  std::mutex iu4_mutex;
  std::vector<std::pair<std::uint8_t*, std::uint16_t*>> iu4_w =
      std::vector<std::pair<std::uint8_t*, std::uint16_t*>>(64,
                                                            {nullptr, nullptr});
  std::vector<std::pair<std::uint8_t*, std::uint16_t*>> iu4_dw =
      std::vector<std::pair<std::uint8_t*, std::uint16_t*>>(64,
                                                            {nullptr, nullptr});
  float* iu4_signs{nullptr};
  void* ModelAlloc(std::size_t bytes) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, bytes));
    HIP_OK(hipMemset(p, 0, bytes));
    HIP_OK(hipDeviceSynchronize());  // the zeroing must land before any stream
                                     // writes
    allocations.push_back(p);
    return p;
  }
  ~ModelState() {
    npu.reset();
    for (void* p : allocations)
      (void)hipFree(p);
  }
  const Tensor& T(const std::string& name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
      std::fprintf(stderr, "inkling: missing tensor %s\n", name.c_str());
      std::abort();
    }
    return it->second;
  }
  bool Has(const std::string& name) const { return tensors.count(name) != 0; }
  std::string P(std::uint32_t il, const char* s) const {
    return "blk." + std::to_string(il) + "." + s;
  }
  std::uint32_t Phys(std::uint32_t slot) const {
    if (slot < n_phys)
      return slot;
    return config.loop_core_start +
           (slot - n_phys) % (config.loop_core_end - config.loop_core_start);
  }
};

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& path,
                                   std::string* error) {
  auto reader = gufo::core::GgufReader::OpenFile(path, error);
  if (!reader)
    return nullptr;
  const auto cfg = ParseConfig(*reader, error);
  if (!cfg)
    return nullptr;
  auto state = std::make_shared<ModelState>();
  state->config = *cfg;
  state->n_phys =
      cfg->num_layers -
      (cfg->loop_num_loops - 1) * (cfg->loop_core_end - cfg->loop_core_start);
  hipStream_t npu_stream = nullptr;
  if (const char* dir = std::getenv("GUFO_NPU_DIR");
      dir != nullptr && std::getenv("INKLING_NPU") != nullptr) {
    gufo::models::qwen38_flash_next::rocm::NpuGemm::Config nc;
    if (const char* mm = std::getenv("INKLING_NPU_M"))
      state->npu_tokens = static_cast<std::uint32_t>(std::atoi(mm));
    nc.xclbin =
        std::string(dir) + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
    nc.insts = std::string(dir) + "/runtime_mk/insts_" +
               std::to_string(state->npu_tokens) + "x" +
               std::to_string(cfg->hidden_size) + "x1024_128x64x128.txt";
    nc.tokens = state->npu_tokens;
    nc.k = cfg->hidden_size;
    nc.n = kNpuSlice;
    std::string npu_error;
    state->npu =
        gufo::models::qwen38_flash_next::rocm::NpuGemm::Create(nc, &npu_error);
    if (state->npu) {
      if (state->npu_tokens > 2048) {
        std::string add_error;
        state->npu_prog_small = state->npu->AddProgram(
            std::string(dir) + "/runtime_mk/insts_2048x" +
                std::to_string(cfg->hidden_size) + "x1024_128x64x128.txt",
            &add_error);
      }
      std::string out_error;
      state->npu_prog_o = state->npu->AddProgram(
          std::string(dir) + "/runtime_mk/insts_" +
              std::to_string(state->npu_tokens) + "x3072x1024_128x64x128.txt",
          &out_error);
      if (state->npu_tokens > 2048)
        state->npu_prog_o_small = state->npu->AddProgram(
            std::string(dir) +
                "/runtime_mk/insts_2048x3072x1024_128x64x128.txt",
            &out_error);
      HIP_OK(hipStreamCreate(&npu_stream));
    } else {
      std::fprintf(stderr, "inkling: NPU unavailable: %s\n", npu_error.c_str());
    }
  }
  gufo::hip::TensorUpload tensor_upload(*reader);
  struct LoadTimer {
    std::chrono::steady_clock::time_point start{
        std::chrono::steady_clock::now()};
    double upload_ms{0.0};
    bool on{std::getenv("INKLING_LOAD_TIMING") != nullptr};
    ~LoadTimer() {
      if (on)
        std::fprintf(stderr,
                     "inkling load: tensors %.0f ms, of which upload %.0f ms\n",
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count(),
                     upload_ms);
    }
  } load_timer;
  const auto upload = [&](const gufo::core::GgufTensorInfo& info,
                          void* device) -> hipError_t {
    std::string upload_error;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = tensor_upload.Upload(info, device, &upload_error);
    load_timer.upload_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    return ok ? hipSuccess : hipErrorUnknown;
  };
  const auto npu_name = [](std::string_view n) {
    for (const char* suffix :
         {".attn_q.weight", ".attn_attn_gate.weight", ".attn_gate.weight",
          ".attn_k.weight", ".attn_v.weight"}) {
      if (n.size() > std::strlen(suffix) &&
          n.substr(n.size() - std::strlen(suffix)) == suffix)
        return true;
    }
    return false;
  };
  for (const auto& info : reader->GetTensors()) {
    Tensor t;
    switch (info.type) {
      case GgmlType::kF32:
        t.type = kTypeF32;
        break;
      case GgmlType::kBF16:
        t.type = kTypeBf16;
        break;
      case GgmlType::kQ8_0:
        t.type = kTypeQ8;
        break;
      case GgmlType::kNVFP4:
        t.type = kTypeNvfp4;
        break;
      default:
        if (error != nullptr)
          *error = "unsupported tensor type in " + std::string(info.name);
        return nullptr;
    }
    t.cols = info.dimensions[0];
    t.rows = 1;
    for (std::size_t d = 1; d < info.dimensions.size(); ++d)
      t.rows *= info.dimensions[d];
    t.row_bytes = info.size_bytes / t.rows;
    const bool repack = (t.type == kTypeQ8 || t.type == kTypeNvfp4) &&
                        info.name != "token_embd.weight";
    if (repack) {
      void* tmp = nullptr;
      HIP_OK(hipMalloc(&tmp, info.size_bytes));
      HIP_OK(upload(info, tmp));
      if (state->npu && t.type == kTypeQ8 && t.cols == cfg->hidden_size &&
          npu_name(info.name)) {
        std::vector<int> ids;
        for (std::uint64_t r0 = 0; r0 < t.rows; r0 += kNpuSlice) {
          const int id = state->npu->NewBuffer(state->npu->BBytes(), error);
          gufo::models::qwen38_flash_next::rocm::NpuEncodeWeights(
              tmp, static_cast<std::uint32_t>(t.cols),
              static_cast<std::uint32_t>(r0), kNpuSlice,
              static_cast<std::uint32_t>(t.cols), 0,
              static_cast<std::uint32_t>(t.cols), state->npu->Device(id),
              npu_stream);
          ids.push_back(id);
        }
        HIP_OK(hipStreamSynchronize(npu_stream));
        for (const int id : ids)
          state->npu->FinalizeWeights(id);
        state->npu_w.emplace(std::string(info.name), std::move(ids));
      }
      const bool out_name =
          info.name.size() > 19 &&
          info.name.substr(info.name.size() - 19) == ".attn_output.weight";
      if (state->npu && state->npu_prog_o >= 0 && t.type == kTypeQ8 &&
          out_name && t.cols == 3072) {
        // rows are padded to a multiple of the slice with zeros
        const std::uint64_t rows_pad =
            (t.rows + kNpuSlice - 1) / kNpuSlice * kNpuSlice;
        void* padded = nullptr;
        HIP_OK(hipMalloc(&padded, rows_pad * t.row_bytes));
        HIP_OK(hipMemset(padded, 0, rows_pad * t.row_bytes));
        HIP_OK(
            hipMemcpy(padded, tmp, info.size_bytes, hipMemcpyDeviceToDevice));
        std::vector<int> ids;
        for (std::uint64_t r0 = 0; r0 < rows_pad; r0 += kNpuSlice) {
          const int id = state->npu->NewBuffer(state->npu->BBytes(3072), error);
          gufo::models::qwen38_flash_next::rocm::NpuEncodeWeights(
              padded, 3072, static_cast<std::uint32_t>(r0), kNpuSlice, 3072, 0,
              3072, state->npu->Device(id), npu_stream);
          ids.push_back(id);
        }
        HIP_OK(hipStreamSynchronize(npu_stream));
        for (const int id : ids)
          state->npu->FinalizeWeights(id);
        state->npu_w.emplace(std::string(info.name), std::move(ids));
        HIP_OK(hipFree(padded));
      }
      const std::size_t qs_bytes = t.type == kTypeQ8 ? t.cols : t.cols / 2;
      const std::size_t sc_bytes =
          t.type == kTypeQ8 ? (t.cols / 32) * 2 : t.cols / 16;
      const std::size_t stride = (qs_bytes + sc_bytes + 15) / 16 * 16;
      HIP_OK(hipMalloc(&t.dev, stride * t.rows));
      HIP_OK(hipMemset(t.dev, 0, stride * t.rows));
      RepackRows<<<static_cast<unsigned>(t.rows), 256>>>(
          static_cast<const unsigned char*>(tmp), t.row_bytes,
          static_cast<unsigned char*>(t.dev), stride, static_cast<int>(t.cols),
          t.type == kTypeNvfp4);
      HIP_OK(hipDeviceSynchronize());
      HIP_OK(hipFree(tmp));
      t.row_bytes = stride;
      const bool expert_name =
          info.name.size() > 12 &&
          info.name.substr(info.name.size() - 12) == "_exps.weight";
      static const bool expert_nvfp4 =
          std::getenv("INKLING_EXPERT_NVFP4") == nullptr ||
          std::getenv("INKLING_EXPERT_NVFP4")[0] != '0';
      if (expert_nvfp4 && expert_name && t.type == kTypeQ8 &&
          t.cols % 64 == 0) {
        const std::size_t nstride = (t.cols / 2 + t.cols / 16 + 15) / 16 * 16;
        void* nv = nullptr;
        HIP_OK(hipMalloc(&nv, nstride * t.rows));
        HIP_OK(hipMemset(nv, 0, nstride * t.rows));
        Q8ToNvfp4<<<static_cast<unsigned>(t.rows), 128>>>(
            static_cast<const unsigned char*>(t.dev), stride,
            static_cast<unsigned char*>(nv), nstride, static_cast<int>(t.cols));
        HIP_OK(hipDeviceSynchronize());
        HIP_OK(hipFree(t.dev));
        t.dev = nv;
        t.type = kTypeNvfp4;
        t.row_bytes = nstride;
        state->allocations.push_back(t.dev);
        state->resident_bytes += nstride * t.rows;
        state->tensors.emplace(std::string(info.name), t);
        continue;
      }
      state->allocations.push_back(t.dev);
      state->resident_bytes += stride * t.rows;
      state->tensors.emplace(std::string(info.name), t);
      continue;
    }
    HIP_OK(hipMalloc(&t.dev, info.size_bytes));
    HIP_OK(upload(info, t.dev));
    state->allocations.push_back(t.dev);
    state->resident_bytes += info.size_bytes;
    state->tensors.emplace(std::string(info.name), t);
    // The loop controllers are F32 in the GGUF but BF16 in the checkpoint:
    // when every value is exactly a BF16, decode reads a half-size copy.
    const std::string_view nm = info.name;
    const auto ends = [&](std::string_view suffix) {
      return nm.size() >= suffix.size() &&
             nm.substr(nm.size() - suffix.size()) == suffix;
    };
    if (t.type == kTypeF32 && info.data != nullptr &&
        nm.find(".loop_") != std::string_view::npos &&
        (ends("down.weight") || ends("gain.weight") || ends("forget.weight"))) {
      const std::size_t count = info.size_bytes / 4;
      const auto* w = static_cast<const std::uint32_t*>(info.data);
      bool exact = true;
      for (std::size_t i = 0; i < count && exact; ++i)
        exact = (w[i] & 0xFFFFU) == 0;
      if (exact) {
        std::vector<std::uint16_t> half(count);
        for (std::size_t i = 0; i < count; ++i)
          half[i] = static_cast<std::uint16_t>(w[i] >> 16);
        Tensor th = t;
        th.type = kTypeBf16;
        th.row_bytes = t.row_bytes / 2;
        th.dev = nullptr;
        HIP_OK(hipMalloc(&th.dev, count * 2));
        HIP_OK(
            hipMemcpy(th.dev, half.data(), count * 2, hipMemcpyHostToDevice));
        state->allocations.push_back(th.dev);
        state->resident_bytes += count * 2;
        state->tensors.emplace(std::string(info.name) + ".h", th);
      }
    }
  }
  {
    const Tensor& res = state->T("loop_res_query.weight");
    const Tensor& kv = state->T("loop_kv_res_query.weight");
    const std::size_t hb = res.cols * 4;
    const std::size_t ncc = res.rows / 2;
    HIP_OK(hipMalloc(&state->mq_table, ncc * 4 * hb));
    state->allocations.push_back(state->mq_table);
    char* dst = reinterpret_cast<char*>(state->mq_table);
    const char* r = static_cast<const char*>(res.dev);
    const char* k = static_cast<const char*>(kv.dev);
    for (std::size_t c = 0; c < ncc; ++c) {
      const std::size_t src[4][2] = {{0, 2 * c}, {1, 2 * c}, {1, 2 * c + 1},
                                     {0, 2 * c + 1}};
      for (int j = 0; j < 4; ++j)
        HIP_OK(hipMemcpy(dst + (c * 4 + j) * hb,
                         (src[j][0] == 0 ? r : k) + src[j][1] * hb, hb,
                         hipMemcpyDeviceToDevice));
    }
  }
  // F16 copy of every relative-bias table with one zero row appended
  for (std::uint32_t il = 0; il < state->n_phys; ++il) {
    const std::string name = state->P(il, "attn_rel_bias.weight");
    const Tensor& src = state->T(name);
    Tensor t;
    t.type = kTypeBf16;
    t.cols = src.cols;
    t.rows = src.rows + 1;
    t.row_bytes = src.cols * 2;
    const std::size_t bytes = t.rows * t.row_bytes;
    HIP_OK(hipMalloc(&t.dev, bytes));
    HIP_OK(hipMemset(t.dev, 0, bytes));
    const std::size_t count = src.rows * src.cols;
    ToHalf<<<(count + 255) / 256, 256>>>(static_cast<const float*>(src.dev),
                                         static_cast<__half*>(t.dev), count);
    HIP_OK(hipDeviceSynchronize());
    state->allocations.push_back(t.dev);
    state->resident_bytes += bytes;
    state->tensors.emplace(name + ".h", t);
  }
  auto model = std::shared_ptr<Model>(new Model());
  model->state_ = std::move(state);
  return model;
}

const Config& Model::config() const noexcept {
  return state_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return state_->resident_bytes;
}

// ---------------------------------------------------------------- DSpark
// drafter kernels
constexpr float kDraftRopeTheta = 1000000.0F;
constexpr int kDraftSlots = 8;

// per position: mix = sum_k w[k] * feat[pos][k]; out = rms(mix) * norm
__global__ void DraftFuseNorm(const __half* feat, int pos0, const float* w4,
                              const float* norm, float* out, int h, float eps) {
  __shared__ float red[256];
  const __half* f =
      feat + static_cast<std::size_t>(pos0 + blockIdx.x) * kTapCount * h;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    float m = 0.0F;
    for (int k = 0; k < kTapCount; ++k)
      m += w4[k] * __half2float(f[static_cast<std::size_t>(k) * h + i]);
    out[static_cast<std::size_t>(blockIdx.x) * h + i] = m;
    ss += m * m;
  }
  ss = BlockSum(ss, red);
  const float inv = rsqrtf(ss / static_cast<float>(h) + eps);
  for (int i = threadIdx.x; i < h; i += blockDim.x)
    out[static_cast<std::size_t>(blockIdx.x) * h + i] *= inv * norm[i];
}

// per (row, head): rms over 128 (optional weight) then NEOX rope at position
// pos_base + row
__global__ void DraftNormRope(float* x, const float* w, int heads, int pos_base,
                              float eps, bool rope,
                              float theta = kDraftRopeTheta) {
  __shared__ float red[128];
  float* p =
      x + (static_cast<std::size_t>(blockIdx.x) * heads + blockIdx.y) * 128;
  const int d = threadIdx.x;
  float v = p[d];
  float ss = BlockSum(v * v, red);
  const float inv = rsqrtf(ss / 128.0F + eps);
  v = v * inv * (w != nullptr ? w[d] : 1.0F);
  if (!rope) {
    p[d] = v;
    return;
  }
  __shared__ float sh[128];
  sh[d] = v;
  __syncthreads();
  const int i = d & 63;
  // angle in double with range reduction: small rope bases give large angles
  // at deep positions, where single-precision sin and cos lose their accuracy
  const double freq =
      exp(-static_cast<double>(i) * (log(static_cast<double>(theta)) / 64.0));
  const double ang_d = fmod(static_cast<double>(pos_base + blockIdx.x) * freq,
                            6.283185307179586);
  const float ang = static_cast<float>(ang_d);
  const float c = cosf(ang);
  const float sn = sinf(ang);
  p[d] = d < 64 ? sh[i] * c - sh[i + 64] * sn : sh[i + 64] * c + sh[i] * sn;
}

// context K/V (rows [n][1024]) -> f16 cache rows at positions pos0 + row (K
// already normed and roped)
__global__ void DraftStoreCtx(const float* k, const float* v, __half* kc,
                              __half* vc, int pos0, int cap) {
  const std::size_t dst =
      static_cast<std::size_t>((pos0 + blockIdx.x) % cap) * 1024;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    kc[dst + i] =
        __float2half(k[static_cast<std::size_t>(blockIdx.x) * 1024 + i]);
    vc[dst + i] =
        __float2half(v[static_cast<std::size_t>(blockIdx.x) * 1024 + i]);
  }
}

// block attention for one head and one key split: all 8 query slots share every
// key load. Keys are the cached context rows [lo, P) (ring of cap rows)
// followed by the 8 block keys (non-causal).
__global__ void DraftAttnPartial(const float* q, const float* kb,
                                 const float* vb, const __half* kc,
                                 const __half* vc, int P, int lo, int cap,
                                 int nsplit, float* pm, float* pl, float* po) {
  __shared__ float sm[4][kDraftSlots], sl[4][kDraftSlots];
  __shared__ float so[4][kDraftSlots][128];
  const int h = blockIdx.x;
  const int split = blockIdx.y;
  const int g = h / 3;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int slot = lane & 7;
  const int chunk = lane >> 3;
  const int nctx = P - lo;
  const int total = nctx + kDraftSlots;
  const int per = (total + nsplit - 1) / nsplit;
  const int st = split * per;
  const int en = min(total, st + per);
  constexpr float scale = 0.08838834764831843F;
  float q32[32];
  float o32[32];
  {
    const float* qp =
        q + (static_cast<std::size_t>(slot) * 24 + h) * 128 + chunk * 32;
    for (int t = 0; t < 32; ++t) {
      q32[t] = qp[t] * scale;
      o32[t] = 0.0F;
    }
  }
  float m = -3.0e38F;
  float l = 0.0F;
  for (int j = st + warp; j < en; j += 4) {
    float k32[32], v32[32];
    if (j < nctx) {
      const std::size_t base = static_cast<std::size_t>((lo + j) % cap) * 1024 +
                               g * 128 + chunk * 32;
      const uint4* kp = reinterpret_cast<const uint4*>(kc + base);
      const uint4* vp = reinterpret_cast<const uint4*>(vc + base);
      for (int u = 0; u < 4; ++u) {
        const uint4 kw = kp[u];
        const uint4 vw = vp[u];
        const __half2* kh = reinterpret_cast<const __half2*>(&kw);
        const __half2* vh = reinterpret_cast<const __half2*>(&vw);
        for (int e = 0; e < 4; ++e) {
          const float2 kf = __half22float2(kh[e]);
          const float2 vf = __half22float2(vh[e]);
          k32[u * 8 + e * 2] = kf.x;
          k32[u * 8 + e * 2 + 1] = kf.y;
          v32[u * 8 + e * 2] = vf.x;
          v32[u * 8 + e * 2 + 1] = vf.y;
        }
      }
    } else {
      const std::size_t base =
          (static_cast<std::size_t>(j - nctx) * 8 + g) * 128 + chunk * 32;
      for (int t = 0; t < 32; ++t) {
        k32[t] = kb[base + t];
        v32[t] = vb[base + t];
      }
    }
    float dot = 0.0F;
    for (int t = 0; t < 32; ++t)
      dot += q32[t] * k32[t];
    dot += __shfl_xor(dot, 8);
    dot += __shfl_xor(dot, 16);
    const float mn = fmaxf(m, dot);
    const float al = expf(m - mn);
    const float pr = expf(dot - mn);
    l = l * al + pr;
    for (int t = 0; t < 32; ++t)
      o32[t] = o32[t] * al + pr * v32[t];
    m = mn;
  }
  if (chunk == 0) {
    sm[warp][slot] = m;
    sl[warp][slot] = l;
  }
  for (int t = 0; t < 32; ++t)
    so[warp][slot][chunk * 32 + t] = o32[t];
  __syncthreads();
  for (int idx = threadIdx.x; idx < kDraftSlots * 128; idx += blockDim.x) {
    const int s = idx >> 7;
    const int d = idx & 127;
    float mx = -3.0e38F;
    for (int w = 0; w < 4; ++w)
      mx = fmaxf(mx, sm[w][s]);
    float den = 0.0F, acc = 0.0F;
    for (int w = 0; w < 4; ++w) {
      const float sc = sm[w][s] > -1.0e38F ? expf(sm[w][s] - mx) : 0.0F;
      den += sc * sl[w][s];
      acc += sc * so[w][s][d];
    }
    const std::size_t part =
        (static_cast<std::size_t>(s) * 24 + h) * nsplit + split;
    po[part * 128 + d] = acc;
    if (d == 0) {
      pm[part] = mx;
      pl[part] = den;
    }
  }
}

__global__ void DraftAttnCombine(const float* pm, const float* pl,
                                 const float* po, float* out, int nsplit) {
  const std::size_t base =
      (static_cast<std::size_t>(blockIdx.x) * 24 + blockIdx.y) * nsplit;
  float mx = -3.0e38F;
  for (int s = 0; s < nsplit; ++s)
    mx = fmaxf(mx, pm[base + s]);
  float den = 0.0F, acc = 0.0F;
  for (int s = 0; s < nsplit; ++s) {
    const float sc = pm[base + s] > -1.0e38F ? expf(pm[base + s] - mx) : 0.0F;
    den += sc * pl[base + s];
    acc += sc * po[(base + s) * 128 + threadIdx.x];
  }
  out[(static_cast<std::size_t>(blockIdx.x) * 24 + blockIdx.y) * 128 +
      threadIdx.x] = acc / den;
}

// x += y; h = rms(x) * w
__global__ void DraftAddNorm(float* x, const float* y, const float* w,
                             float* hout, int h, float eps) {
  __shared__ float red[256];
  float* xp = x + static_cast<std::size_t>(blockIdx.x) * h;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    const float v =
        xp[i] +
        (y != nullptr ? y[static_cast<std::size_t>(blockIdx.x) * h + i] : 0.0F);
    xp[i] = v;
    ss += v * v;
  }
  ss = BlockSum(ss, red);
  const float inv = rsqrtf(ss / static_cast<float>(h) + eps);
  if (hout != nullptr) {
    for (int i = threadIdx.x; i < h; i += blockDim.x)
      hout[static_cast<std::size_t>(blockIdx.x) * h + i] = xp[i] * inv * w[i];
  }
}

__global__ void DraftSwiglu(const float* gate, const float* up, float* act,
                            std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = gate[i];
    act[i] = g / (1.0F + expf(-g)) * up[i];
  }
}

// mask rows for slots 1..7 (slot 0 is the embedded anchor) and the first-layer
// norm of every slot
__global__ void DraftSlotsInit(const float* mask, const float* anchor_row,
                               float* x, int h) {
  const int slot = blockIdx.x;
  for (int i = threadIdx.x; i < h; i += blockDim.x)
    x[static_cast<std::size_t>(slot) * h + i] =
        slot == 0 ? anchor_row[i] : mask[i];
}

// joint input row = [state(256), w1[prev](256), cur(2304), carry(2304)]
__global__ void DraftJointIn(const float* state, const unsigned short* w1,
                             const int* prev_p, const float* cur,
                             const float* carry, float* out, int h, int r) {
  const int prev = *prev_p;
  for (int i = threadIdx.x; i < 2 * r + 2 * h; i += blockDim.x) {
    float v;
    if (i < r)
      v = state[i];
    else if (i < 2 * r)
      v = __uint_as_float(static_cast<unsigned>(
                              w1[static_cast<std::size_t>(prev) * r + (i - r)])
                          << 16);
    else if (i < 2 * r + h)
      v = cur[i - 2 * r];
    else
      v = carry[i - 2 * r - h];
    out[i] = v;
  }
}

// state = gate*state + (1-gate)*cand; latent = tanh(u2)
__global__ void DraftMarkovState(const float* joint, const float* bias,
                                 float* state, float* latent, int r) {
  const int i = threadIdx.x;
  if (i >= r)
    return;
  const float u0 = joint[i] + bias[i];
  const float u1 = joint[r + i] + bias[r + i];
  const float u2 = joint[2 * r + i] + bias[2 * r + i];
  const float gate = 1.0F / (1.0F + expf(-u0));
  const float cand = tanhf(u1);
  state[i] = gate * (state[i] - cand) + cand;
  latent[i] = tanhf(u2);
}

// y[r] = w[r] . x + a1[r] + a2[r]; w is bf16 [rows][cols]; one wave per row
__global__ void DraftBf16Gemv(const unsigned short* w, const float* x, float* y,
                              int cols, int rows, const float* a1,
                              const float* a2) {
  const int lane = threadIdx.x % warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (r >= rows)
    return;
  const unsigned short* wr = w + static_cast<std::size_t>(r) * cols;
  float acc = 0.0F;
  for (int c = lane; c < cols; c += warpSize)
    acc += __uint_as_float(static_cast<unsigned>(wr[c]) << 16) * x[c];
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down(acc, off);
  if (lane == 0)
    y[r] =
        acc + (a1 != nullptr ? a1[r] : 0.0F) + (a2 != nullptr ? a2[r] : 0.0F);
}

__global__ void HalfToFloat(const __half* in, float* out, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = __half2float(in[i]);
}

// ---------------------------------------------------------------- DFlash2
// drafter kernels (block of 8 rows, 32 heads, two-tap dynamic convolutions and
// a low-rank candidate path selector)
constexpr int kD2Taps = 2;

// grouped dynamic causal convolution over the rows of the block:
// out[t][c] = sum over taps of (base[which][tap][c] + dyn[t][which][tap][c /
// group]) * in[t - tap][c]; dyn is the kernel projection of the block rows,
// laid out [which][tap][group]
__global__ void D2Conv(const float* in, const float* proj, const float* base,
                       float* out, int h, int groups, int gsz, int which) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x; c < h; c += blockDim.x) {
    float acc = 0.0F;
    for (int tap = 0; tap < kD2Taps; ++tap) {
      if (t - tap < 0)
        break;
      const float x = in[static_cast<std::size_t>(t - tap) * h + c];
      const float d =
          proj[static_cast<std::size_t>(t) * (2 * kD2Taps * groups) +
               (which * kD2Taps + tap) * groups + c / gsz];
      acc += (base[(which * kD2Taps + tap) * h + c] + d) * x;
    }
    out[static_cast<std::size_t>(t) * h + c] = acc;
  }
}

// block attention for one head and one key split with the sliding window of
// the DFlash2 layers: context rows [lo, P) from the ring, then the 8 block keys
// (all visible). A context key j is hidden from slot s when
// P + s - (lo + j) >= window.
__global__ void D2AttnPartial(const float* q, const float* kb, const float* vb,
                              const __half* kc, const __half* vc, int P, int lo,
                              int cap, int nsplit, int nh, int gq, int window,
                              float* pm, float* pl, float* po) {
  __shared__ float sm[4][kDraftSlots], sl[4][kDraftSlots];
  __shared__ float so[4][kDraftSlots][128];
  const int h = blockIdx.x;
  const int split = blockIdx.y;
  const int g = h / gq;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int slot = lane & 7;
  const int chunk = lane >> 3;
  const int nctx = P - lo;
  const int total = nctx + kDraftSlots;
  const int per = (total + nsplit - 1) / nsplit;
  const int st = split * per;
  const int en = min(total, st + per);
  constexpr float scale = 0.08838834764831843F;
  float q32[32];
  float o32[32];
  {
    const float* qp =
        q + (static_cast<std::size_t>(slot) * nh + h) * 128 + chunk * 32;
    for (int t = 0; t < 32; ++t) {
      q32[t] = qp[t] * scale;
      o32[t] = 0.0F;
    }
  }
  float m = -3.0e38F;
  float l = 0.0F;
  for (int j = st + warp; j < en; j += 4) {
    float k32[32], v32[32];
    if (j < nctx) {
      const std::size_t base = static_cast<std::size_t>((lo + j) % cap) * 1024 +
                               g * 128 + chunk * 32;
      const uint4* kp = reinterpret_cast<const uint4*>(kc + base);
      const uint4* vp = reinterpret_cast<const uint4*>(vc + base);
      for (int u = 0; u < 4; ++u) {
        const uint4 kw = kp[u];
        const uint4 vw = vp[u];
        const __half2* kh = reinterpret_cast<const __half2*>(&kw);
        const __half2* vh = reinterpret_cast<const __half2*>(&vw);
        for (int e = 0; e < 4; ++e) {
          const float2 kf = __half22float2(kh[e]);
          const float2 vf = __half22float2(vh[e]);
          k32[u * 8 + e * 2] = kf.x;
          k32[u * 8 + e * 2 + 1] = kf.y;
          v32[u * 8 + e * 2] = vf.x;
          v32[u * 8 + e * 2 + 1] = vf.y;
        }
      }
    } else {
      const std::size_t base =
          (static_cast<std::size_t>(j - nctx) * 8 + g) * 128 + chunk * 32;
      for (int t = 0; t < 32; ++t) {
        k32[t] = kb[base + t];
        v32[t] = vb[base + t];
      }
    }
    float dot = 0.0F;
    for (int t = 0; t < 32; ++t)
      dot += q32[t] * k32[t];
    dot += __shfl_xor(dot, 8);
    dot += __shfl_xor(dot, 16);
    if (j < nctx && window > 0 && P + slot - (lo + j) >= window)
      dot = -3.0e38F;
    const float mn = fmaxf(m, dot);
    const float al = expf(m - mn);
    const float pr = expf(dot - mn);
    l = l * al + pr;
    for (int t = 0; t < 32; ++t)
      o32[t] = o32[t] * al + pr * v32[t];
    m = mn;
  }
  if (chunk == 0) {
    sm[warp][slot] = m;
    sl[warp][slot] = l;
  }
  for (int t = 0; t < 32; ++t)
    so[warp][slot][chunk * 32 + t] = o32[t];
  __syncthreads();
  for (int idx = threadIdx.x; idx < kDraftSlots * 128; idx += blockDim.x) {
    const int s = idx >> 7;
    const int d = idx & 127;
    float mx = -3.0e38F;
    for (int w = 0; w < 4; ++w)
      mx = fmaxf(mx, sm[w][s]);
    float den = 0.0F, acc = 0.0F;
    for (int w = 0; w < 4; ++w) {
      const float sc = sm[w][s] > -1.0e38F ? expf(sm[w][s] - mx) : 0.0F;
      den += sc * sl[w][s];
      acc += sc * so[w][s][d];
    }
    const std::size_t part =
        (static_cast<std::size_t>(s) * nh + h) * nsplit + split;
    po[part * 128 + d] = acc;
    if (d == 0) {
      pm[part] = mx;
      pl[part] = den;
    }
  }
}

__global__ void D2AttnCombine(const float* pm, const float* pl, const float* po,
                              float* out, int nsplit, int nh) {
  const std::size_t base =
      (static_cast<std::size_t>(blockIdx.x) * nh + blockIdx.y) * nsplit;
  float mx = -3.0e38F;
  for (int s = 0; s < nsplit; ++s)
    mx = fmaxf(mx, pm[base + s]);
  float den = 0.0F, acc = 0.0F;
  for (int s = 0; s < nsplit; ++s) {
    const float sc = pm[base + s] > -1.0e38F ? expf(pm[base + s] - mx) : 0.0F;
    den += sc * pl[base + s];
    acc += sc * po[(base + s) * 128 + threadIdx.x];
  }
  out[(static_cast<std::size_t>(blockIdx.x) * nh + blockIdx.y) * 128 +
      threadIdx.x] = acc / den;
}

// the 16 largest logits of each row (unordered), one block per row
__global__ void __launch_bounds__(256)
    D2TopK(const float* logits, int n, std::size_t ld, int* ids, float* vals) {
  constexpr int kK = 16;
  __shared__ float sv[256 * kK];
  __shared__ int si[256 * kK];
  __shared__ float rv[256];
  __shared__ int ri[256];
  const float* x = logits + static_cast<std::size_t>(blockIdx.x) * ld;
  float lv[kK];
  int li[kK];
  for (int j = 0; j < kK; ++j) {
    lv[j] = -3.0e38F;
    li[j] = 0;
  }
  float mn = -3.0e38F;
  int mi = 0;
  for (int i = threadIdx.x; i < n; i += 256) {
    const float v = x[i];
    if (v > mn) {
      lv[mi] = v;
      li[mi] = i;
      mn = lv[0];
      mi = 0;
      for (int j = 1; j < kK; ++j) {
        if (lv[j] < mn) {
          mn = lv[j];
          mi = j;
        }
      }
    }
  }
  for (int j = 0; j < kK; ++j) {
    sv[threadIdx.x * kK + j] = lv[j];
    si[threadIdx.x * kK + j] = li[j];
  }
  __syncthreads();
  for (int round = 0; round < kK; ++round) {
    float best = -3.4e38F;
    int bj = 0;
    for (int j = 0; j < kK; ++j) {
      const float v = sv[threadIdx.x * kK + j];
      if (v > best) {
        best = v;
        bj = j;
      }
    }
    rv[threadIdx.x] = best;
    ri[threadIdx.x] = threadIdx.x;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (threadIdx.x < s && rv[threadIdx.x + s] > rv[threadIdx.x]) {
        rv[threadIdx.x] = rv[threadIdx.x + s];
        ri[threadIdx.x] = ri[threadIdx.x + s];
      }
      __syncthreads();
    }
    const int win = ri[0];
    if (threadIdx.x == win) {
      ids[blockIdx.x * kK + round] = si[win * kK + bj];
      vals[blockIdx.x * kK + round] = rv[0];
      sv[win * kK + bj] = -3.4e38F;
    }
    __syncthreads();
  }
}

// greedy path over the 16 candidates of each of the 7 draft positions:
// score = unary + (predecessor[prev] * hp) . successor[cand], softcapped
__global__ void __launch_bounds__(256)
    D2Select(const int* cand, const float* unary, const float* hp,
             const unsigned short* pred, const unsigned short* succ,
             const int* anchor, int rank, int steps, float softcap, int* out,
             float* conf) {
  __shared__ float pv[256];
  __shared__ float score[16];
  __shared__ int prev_s;
  if (threadIdx.x == 0)
    prev_s = *anchor;
  __syncthreads();
  const int grp = threadIdx.x >> 4;
  const int lane = threadIdx.x & 15;
  for (int s = 0; s < steps; ++s) {
    const int prev = prev_s;
    for (int r = threadIdx.x; r < rank; r += 256)
      pv[r] =
          __uint_as_float(static_cast<unsigned>(
                              pred[static_cast<std::size_t>(prev) * rank + r])
                          << 16) *
          hp[s * rank + r];
    __syncthreads();
    const int c = cand[s * 16 + grp];
    float dot = 0.0F;
    for (int r = lane; r < rank; r += 16)
      dot += __uint_as_float(static_cast<unsigned>(
                                 succ[static_cast<std::size_t>(c) * rank + r])
                             << 16) *
             pv[r];
    for (int off = 8; off > 0; off >>= 1)
      dot += __shfl_xor(dot, off, 16);
    if (lane == 0) {
      float v = unary[s * 16 + grp] + dot;
      if (softcap > 0.0F)
        v = softcap * tanhf(v / softcap);
      score[grp] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int bi = 0;
      for (int k = 1; k < 16; ++k)
        if (score[k] > score[bi])
          bi = k;
      float tot = 0.0F;
      for (int k = 0; k < 16; ++k)
        tot += expf(score[k] - score[bi]);
      const int pick = bi;
      const float c = 1.0F / tot;
      const int tok = cand[s * 16 + pick];
      out[s] = tok;
      conf[s] = c;
      prev_s = tok;
    }
    __syncthreads();
  }
}

__global__ void GatherRows(const unsigned char* src, std::size_t row_bytes,
                           const int* ids, unsigned char* dst) {
  const uint4* in = reinterpret_cast<const uint4*>(
      src + static_cast<std::size_t>(ids[blockIdx.x]) * row_bytes);
  uint4* out = reinterpret_cast<uint4*>(
      dst + static_cast<std::size_t>(blockIdx.x) * row_bytes);
  for (std::size_t i = threadIdx.x; i < row_bytes / 16; i += blockDim.x)
    out[i] = in[i];
}

__global__ void ArgMaxRow(const float* x, int n, int* out, int* out2,
                          const int* map = nullptr, float* conf = nullptr,
                          int* stop = nullptr, float stop_conf = 0.0F) {
  x += static_cast<std::size_t>(blockIdx.x) * n;
  out += blockIdx.x;
  out2 += blockIdx.x;
  __shared__ float bv[256];
  __shared__ int bi[256];
  __shared__ float sv[256];
  float best = -3.0e38F;
  float sum = 0.0F;
  int idx = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = x[i];
    if (conf == nullptr) {
      if (v > best) {
        best = v;
        idx = i;
      }
      continue;
    }
    if (v > best) {
      sum = sum * expf(best - v) + 1.0F;
      best = v;
      idx = i;
    } else {
      sum += expf(v - best);
    }
  }
  bv[threadIdx.x] = best;
  bi[threadIdx.x] = idx;
  sv[threadIdx.x] = sum;
  __syncthreads();
  if (conf != nullptr) {
    float gm = -3.0e38F;
    for (int i = 0; i < 256; ++i)
      gm = fmaxf(gm, bv[i]);
    float tot = 0.0F;
    for (int i = threadIdx.x; i < 256; i += 256)
      tot += sv[i] * expf(bv[i] - gm);
    __syncthreads();
    sv[threadIdx.x] = tot;
    __syncthreads();
    if (threadIdx.x == 0) {
      float t = 0.0F;
      for (int i = 0; i < 256; ++i)
        t += sv[i];
      *conf = 1.0F / t;
      if (stop != nullptr && 1.0F / t < stop_conf)
        *stop = 1;
    }
    __syncthreads();
  }
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && (bv[threadIdx.x + s] > bv[threadIdx.x] ||
                            (bv[threadIdx.x + s] == bv[threadIdx.x] &&
                             bi[threadIdx.x + s] < bi[threadIdx.x]))) {
      bv[threadIdx.x] = bv[threadIdx.x + s];
      bi[threadIdx.x] = bi[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int id = map != nullptr ? map[bi[0]] : bi[0];
    *out = id;
    *out2 = id;
  }
}

// Everything one token bank needs between layers; a session swaps two of these
// so each bank has its own residual stream, blocks and scratch.
struct Scratch {
  int* d_tokens{nullptr};
  float *cur{nullptr}, *tmp{nullptr}, *inp_sa{nullptr}, *in[3]{}, *q{nullptr},
      *k{nullptr}, *v{nullptr}, *r{nullptr}, *dyn_k{nullptr}, *dyn_v{nullptr},
      *sink{nullptr}, *gate{nullptr}, *att{nullptr}, *o{nullptr},
      *mixed{nullptr}, *msgs{nullptr}, *ffn_inp{nullptr}, *x{nullptr},
      *cat{nullptr}, *low{nullptr}, *gain{nullptr}, *forget{nullptr},
      *router{nullptr}, *wts{nullptr}, *gu{nullptr}, *act{nullptr}, *y{nullptr},
      *moe{nullptr}, *prelude{nullptr}, *bank_out{nullptr}, *d_logits{nullptr};
  unsigned int* rcounts{nullptr};
  int *rbounds{nullptr}, *rcursors{nullptr}, *rrows_token{nullptr},
      *rrows_slot{nullptr};
  int *sorted{nullptr}, *tiles{nullptr};
  int *tl_a{nullptr}, *tl_b{nullptr};  ///< compact (expert, token tile) lists
  int* dense_bounds_buf{nullptr};      ///< {0, n} of the dense routed GEMM
  int dense_bounds_n{-1};
  float* ksbuf{nullptr};  ///< split-K partial sums
  __half *xh{nullptr}, *act16{nullptr}, *xh0{nullptr}, *xh1{nullptr},
      *xh2{nullptr}, *xhx{nullptr}, *atth{nullptr}, *cath{nullptr};
  signed char* xq{nullptr};
  float* xqs{nullptr};
  float *split_m{nullptr}, *split_l{nullptr}, *split_o{nullptr};
  __half *gu16{nullptr}, *y16{nullptr};
  int *blist{nullptr}, *bcount{nullptr};
  int *ids{nullptr}, *xrow{nullptr}, *d_pos{nullptr};
  float* block_pool[16]{};
  int pool_used{0};
  std::uint32_t pos_base{0};  // absolute position of this bank's first token
  int npu_a[3]{-1, -1, -1};
  int npu_c[8]{-1, -1, -1, -1, -1, -1, -1, -1};
  int npu_ao{-1};
};

struct Session::Impl : Scratch {
  std::shared_ptr<ModelState> m;
  std::uint32_t ctx{0};
  std::uint32_t position{0};
  struct Slot {
    __half* k{nullptr};
    __half* v{nullptr};
    int ring{0};
    int vblocks{0};
    __half* kmm{nullptr};
    float* hist_k{nullptr};
    float* hist_v{nullptr};
    float* hist_attn{nullptr};
    float* hist_ffn{nullptr};
  };
  std::vector<Slot> slots;
  bool banked{false};
  __half* feat{nullptr};  // tapped hidden states per position (drafter
                          // features), when enabled
  // ---- drafter state
  std::uint32_t dwarmed{0};
  bool dready{false};
  __half* dk[3]{};
  __half* dv[3]{};
  int dcap[3]{};
  float *dctx{}, *dck{}, *dcv{}, *dx{}, *dh{}, *dq{}, *dkb{}, *dvb{}, *datt{},
      *dgate{}, *dup{}, *dact{}, *dpm{}, *dpl{}, *dpo{}, *dcarry{}, *dstate{},
      *dlatent{}, *djin{}, *djout{}, *deff{}, *dlog{}, *dtmp{}, *dw{};
  int *dprev{}, *dtok{}, *dout{};
  // DFlash2 drafter state
  static constexpr int kD2Cap = 4096;
  static constexpr int kD2Layers = 5;
  __half* d2k[kD2Layers]{};
  __half* d2v[kD2Layers]{};
  float *d2cat{}, *d2th{}, *d2ctx{}, *d2ck{}, *d2cv{}, *d2emb{}, *d2q{},
      *d2att{}, *d2proj{}, *d2conv{}, *d2log{}, *d2hp{}, *d2uv{}, *d2gate{},
      *d2up{}, *d2act{}, *d2pm{}, *d2pl{}, *d2po{};
  int *d2tok{}, *d2ids{};
  float *d2x{}, *d2h{}, *d2tmp{}, *d2kb{}, *d2vb{}, *d2conf{};
  int* d2out{};
  std::uint32_t d2warmed{0};
  bool d2ready{false};
  bool prefer_d2{true};  // which drafter Draft uses when both are loaded
  static constexpr int kDraftWarmChunk = 256;
  static constexpr int kDraftSplit = 64;
  static constexpr int kDraftShortlistMax = 65536;
  unsigned char* dhead{nullptr};
  std::size_t dhead_stride{0};
  int* dmap{nullptr};
  float* dconf{nullptr};
  int* dstop{nullptr};
  int kDraftShortlist{std::getenv("INKLING_SHORTLIST") != nullptr
                          ? std::atoi(std::getenv("INKLING_SHORTLIST"))
                          : 32768};
  HistEntry* hist_entries{nullptr};
  int n_hist_entries{0};
  int last_rows{0};
  std::uint32_t pos_before_last{0};
  int n_logit_rows{1};
  bool all_rows{false};  // keep the logits of every row (verification batches)
  std::size_t bank_rows{2048};
  int npu_mask{std::getenv("INKLING_NPU_PARTS") != nullptr
                   ? std::atoi(std::getenv("INKLING_NPU_PARTS"))
                   : 15};  // 1=q 2=gate 4=k 8=v
  Scratch saved[2];        // bank scratch while another bank is active
  int active{0};
  void Activate(int bank) {
    if (two_stream)
      stream = bank == 0 ? stream0 : stream1;
    if (bank == active)
      return;
    saved[active] = static_cast<Scratch&>(*this);
    static_cast<Scratch&>(*this) = saved[bank];
    active = bank;
  }
  struct SnapRegion {
    void* dev;
    std::size_t bytes;
  };
  // Device ranges that make up a snapshot at `pos` rows: the order is the
  // layout of the snapshot bytes.
  std::vector<SnapRegion> SnapRegions(std::uint32_t pos, std::uint32_t dwarm,
                                      bool drafter) const {
    const Config& c = m->config;
    const std::size_t kv = static_cast<std::size_t>(c.num_kv) * c.head_dim;
    const std::size_t h = c.hidden_size;
    std::vector<SnapRegion> out;
    for (const auto& sl : slots) {
      const std::size_t rows_total = static_cast<std::size_t>(sl.vblocks) * 64;
      if (sl.ring > 0) {
        out.push_back({sl.k, rows_total * kv * 2});
        out.push_back({sl.v, rows_total * kv * 2});
      } else {
        const std::size_t rows = std::min<std::size_t>(pos, rows_total);
        out.push_back({sl.k, rows * kv * 2});
        const std::size_t nb = (rows + 63) / 64;
        for (int g = 0; g < c.num_kv; ++g) {
          out.push_back(
              {sl.v + static_cast<std::size_t>(g) * sl.vblocks * 128 * 64,
               nb * 128 * 64 * 2});
        }
        if (sl.kmm != nullptr) {
          const std::size_t blocks =
              std::min<std::size_t>(pos / kBlockKeys + 2, ctx / kBlockKeys + 2);
          out.push_back({sl.kmm, blocks * c.num_kv * 2 * kFlashHd * 2});
        }
      }
      out.push_back({sl.hist_k, kv * (kHistory + kWinLen) * 4});
      out.push_back({sl.hist_v, kv * (kHistory + kWinLen) * 4});
      out.push_back({sl.hist_attn, h * (kHistory + kWinLen) * 4});
      out.push_back({sl.hist_ffn, h * (kHistory + kWinLen) * 4});
    }
    if (feat != nullptr)
      out.push_back({feat, static_cast<std::size_t>(pos) * kTapCount * h * 2});
    if (drafter && dready && m->has_dspark) {
      for (int l = 0; l < 3; ++l) {
        const std::size_t rows =
            l == 1 ? std::min<std::size_t>(dwarm, dcap[l]) : dcap[l];
        out.push_back({dk[l], rows * 1024 * 2});
        out.push_back({dv[l], rows * 1024 * 2});
      }
    }
    return out;
  }
  std::vector<void*> allocs;
  std::vector<float> logits;  // host copy of rows fetched by Logits(row)
  std::vector<unsigned char> logits_have;
  int* d_argmax{nullptr};
  int argmax_host[8]{};
  int logits_rows{0};
  int logits_last{0};
  std::size_t allocated_bytes{0};
  std::unordered_map<long long, std::pair<int*, int>> tile_maps;
  const int* TileMap(int n, int bn, int* count) {
    const long long key = static_cast<long long>(n) * 1024 + bn;
    auto it = tile_maps.find(key);
    if (it == tile_maps.end()) {
      const int ne = static_cast<int>(m->config.expert_count);
      const int per = (n + bn - 1) / bn;
      std::vector<int> host;
      host.reserve(static_cast<std::size_t>(ne) * per);
      for (int e = 0; e < ne; ++e) {
        for (int j = 0; j < per; ++j)
          host.push_back(e | (j << 16));
      }
      int* dev = static_cast<int*>(AllocBytes(host.size() * sizeof(int)));
      allocated_bytes -= host.size() * sizeof(int);  // grows with prompt shapes
      HIP_OK(hipMemcpy(dev, host.data(), host.size() * sizeof(int),
                       hipMemcpyHostToDevice));
      it = tile_maps
               .emplace(key, std::make_pair(dev, static_cast<int>(host.size())))
               .first;
    }
    *count = it->second.second;
    return it->second.first;
  }
  bool sparse{false};
  bool no_dp4{std::getenv("INKLING_NO_DP4") != nullptr};
  int *xrow_token{nullptr}, *xrow_item{nullptr};
  hipStream_t stream{nullptr};
  // GPU-only token banks run on two streams so one bank's memory-bound kernels
  // overlap the other bank's matrix work; bank 1's attention waits for bank
  // 0's keys, values and conv histories through ev_bank.
  hipStream_t stream0{nullptr}, stream1{nullptr};
  hipEvent_t ev_bank{nullptr};
  bool two_stream{false};
  // NPU submits and waits are host callbacks; with two compute streams they
  // all go through one control stream (ordered by event) so a blocked wait can
  // never sit in front of the submit it needs.
  hipStream_t npu_ctl{nullptr};
  hipEvent_t npu_ev[64]{};
  unsigned npu_ev_i{0};
  void NpuSubmit(int a, const int* b, const int* c, int count, int prog) {
    auto& npu = *m->npu;
    if (npu_ctl == nullptr) {
      npu.Submit(stream, a, b, c, count, prog);
      return;
    }
    const hipEvent_t e = npu_ev[npu_ev_i++ % 64];
    HIP_OK(hipEventRecord(e, stream));
    HIP_OK(hipStreamWaitEvent(npu_ctl, e, 0));
    npu.Submit(npu_ctl, a, b, c, count, prog);
  }
  void NpuWaitFor(std::uint64_t ticket) {
    auto& npu = *m->npu;
    if (npu_ctl == nullptr) {
      npu.Wait(stream, ticket);
      return;
    }
    npu.Wait(npu_ctl, ticket);
    const hipEvent_t e = npu_ev[npu_ev_i++ % 64];
    HIP_OK(hipEventRecord(e, npu_ctl));
    HIP_OK(hipStreamWaitEvent(stream, e, 0));
  }
  hipGraph_t graph{nullptr};
  hipGraphExec_t graph_exec{nullptr};

  float* Alloc(std::size_t floats) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, floats * sizeof(float)));
    allocated_bytes += floats * sizeof(float);
    allocs.push_back(p);
    return static_cast<float*>(p);
  }
  // After a restore, zero the cache rows past `pos` in the full-attention
  // slots. Attention reads whole key/value tiles and relies on rows past the
  // position being zero (as in a fresh session); a session reused from a
  // longer conversation still holds that conversation's rows there.
  void ClearPast(std::uint32_t pos) {
    const Config& c = m->config;
    const std::size_t kv = static_cast<std::size_t>(c.num_kv) * c.head_dim;
    for (auto& sl : slots) {
      if (sl.ring > 0)
        continue;
      const std::size_t rows_total = static_cast<std::size_t>(sl.vblocks) * 64;
      const std::size_t rows = std::min<std::size_t>(pos, rows_total);
      if (rows < rows_total)
        HIP_OK(hipMemsetAsync(sl.k + rows * kv, 0, (rows_total - rows) * kv * 2,
                              stream));
      const std::size_t nb = (rows + 63) / 64;
      if (nb < static_cast<std::size_t>(sl.vblocks)) {
        for (int g = 0; g < c.num_kv; ++g)
          HIP_OK(hipMemsetAsync(
              sl.v + (static_cast<std::size_t>(g) * sl.vblocks + nb) * 128 * 64,
              0, (sl.vblocks - nb) * 128 * 64 * 2, stream));
      }
      if (sl.kmm != nullptr) {
        const std::size_t total = ctx / kBlockKeys + 2;
        const std::size_t blocks =
            std::min<std::size_t>(pos / kBlockKeys + 2, total);
        if (blocks < total)
          HIP_OK(hipMemsetAsync(sl.kmm + blocks * c.num_kv * 2 * kFlashHd, 0,
                                (total - blocks) * c.num_kv * 2 * kFlashHd * 2,
                                stream));
      }
    }
  }
  void* AllocBytes(std::size_t bytes) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, bytes));
    HIP_OK(hipMemset(p, 0, bytes));
    allocated_bytes += bytes;
    allocs.push_back(p);
    return p;
  }
  // Buffers created on first use are not part of the session's measured state:
  // the server compares it with a claim taken from a probe session that never
  // ran a forward pass.
  void* AllocLazy(std::size_t bytes) {
    void* p = AllocBytes(bytes);
    HIP_OK(hipDeviceSynchronize());  // the zeroing must land before any stream
                                     // writes
    allocated_bytes -= bytes;
    return p;
  }
  ~Impl() {
    if (pf_stream != nullptr)
      (void)hipStreamDestroy(pf_stream);
    if (pf_fork != nullptr)
      (void)hipEventDestroy(pf_fork);
    if (pf_join != nullptr)
      (void)hipEventDestroy(pf_join);
    if (graph_exec != nullptr)
      (void)hipGraphExecDestroy(graph_exec);
    if (graph != nullptr)
      (void)hipGraphDestroy(graph);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
    for (void* p : allocs)
      (void)hipFree(p);
  }

  const int* gemm_skip{nullptr};
  bool xq_reuse{false};
  const float* xq_src{nullptr};
  int xq_n{0};
  int xq_cols{0};
  // dense projections through the routed-GEMM kernel: one expert, identity rows
  bool dense_routed{std::getenv("INKLING_DENSE_ROUTED") == nullptr ||
                    std::getenv("INKLING_DENSE_ROUTED")[0] != '0'};
  int dense_bn{std::getenv("INKLING_DENSE_BN") != nullptr
                   ? std::atoi(std::getenv("INKLING_DENSE_BN"))
                   : 192};
  float* situ_params{nullptr};
  // int4 emulation state (INKLING_IU4_EMU)
  std::vector<void*> emu_w = std::vector<void*>(64, nullptr);
  // rotated int4 routed gate/up path (INKLING_IU4)
  float* iu4_signs{nullptr};  ///< the model's shared signs
  std::uint8_t* iu4_act{nullptr};
  float* iu4_act_scales{nullptr};
  std::uint8_t* iu4_tok{nullptr};
  float* iu4_tok_scales{nullptr};
  int iu4_tok_padded{0};
  int iu4_padded{0};
  // down projection (INKLING_IU4=2) activations
  std::uint8_t* iu4_act2{nullptr};
  float* iu4_act2_scales{nullptr};
  float* emu_signs{nullptr};
  __half* emu_x{nullptr};
  float* emu_gu{nullptr};
  int* dense_ident{nullptr};
  int* dense_tiles{nullptr};
  const int* DenseMaps(int n, int bn, int* ntiles) {
    constexpr int kMaxTiles = kMaxBatch / 128 + 2;
    if (dense_ident == nullptr) {
      std::vector<int> ident(kMaxBatch);
      std::vector<int> tiles(kMaxTiles);
      for (int i = 0; i < kMaxBatch; ++i)
        ident[i] = i;
      for (int t = 0; t < kMaxTiles; ++t)
        tiles[t] = t << 16;
      dense_ident = static_cast<int*>(AllocLazy(kMaxBatch * sizeof(int)));
      dense_tiles = static_cast<int*>(AllocLazy(kMaxTiles * sizeof(int)));
      HIP_OK(hipMemcpy(dense_ident, ident.data(), kMaxBatch * sizeof(int),
                       hipMemcpyHostToDevice));
      HIP_OK(hipMemcpy(dense_tiles, tiles.data(), kMaxTiles * sizeof(int),
                       hipMemcpyHostToDevice));
    }
    // one {0, n} buffer per bank, rewritten when n changes (a new buffer per
    // prompt length would grow the session's measured state forever)
    if (dense_bounds_n != n) {
      SetDenseBounds<<<1, 1, 0, stream>>>(dense_bounds_buf, n);
      dense_bounds_n = n;
    }
    *ntiles = (n + bn - 1) / bn;
    return dense_bounds_buf;
  }
  // Decode-sized batches (1 to 8 rows): the Q8_0 matrices of one layer visit
  // run in one launch after one quantize launch (instead of a quantize and a
  // GEMV launch per matrix). False when a matrix is not Q8_0 of K 2304.
  // Decode steps: the dense weights of a layer visit are read on a side
  // stream while the latency-bound mixes, controllers and attention run (the
  // DRAM is idle then), so the GEMVs that follow find them in the last-level
  // cache.
  hipStream_t pf_stream{nullptr};
  hipEvent_t pf_fork{nullptr}, pf_join{nullptr};
  unsigned int* pf_sink{nullptr};
  bool pf_open{false};
  bool pf_on{std::getenv("INKLING_DECODE_PREFETCH") == nullptr ||
             std::getenv("INKLING_DECODE_PREFETCH")[0] != '0'};
  bool expert_nt{std::getenv("INKLING_EXPERT_NT") == nullptr ||
                 std::getenv("INKLING_EXPERT_NT")[0] != '0'};
  static constexpr int kPrefetchBlocks = 8;
  std::vector<std::uint32_t> visit_slots;  // slots of a step's visits, in order
  int visit_i{0};
  void BuildVisits() {
    const Config& c = m->config;
    visit_slots.clear();
    visit_i = 0;
    for (std::uint32_t il = 0; il < c.loop_core_start; ++il)
      visit_slots.push_back(il);
    for (std::uint32_t pass = 0; pass < c.loop_num_loops; ++pass)
      for (std::uint32_t ip = c.loop_core_start; ip < c.loop_core_end; ++ip)
        visit_slots.push_back(
            pass == 0 ? ip
                      : m->n_phys +
                            (pass - 1) * (c.loop_core_end - c.loop_core_start) +
                            (ip - c.loop_core_start));
    for (std::uint32_t il = c.loop_core_end; il < m->n_phys; ++il)
      visit_slots.push_back(il);
  }
  void Prefetch(std::initializer_list<const Tensor*> ts, int n) {
    if (!pf_on || pf_stream == nullptr || n != 1)  // graph-captured step only
      return;
    TouchList l{};
    for (const Tensor* t : ts) {
      if (l.n < 12 && t->dev != nullptr) {
        l.p[l.n] = t->dev;
        l.bytes[l.n] = static_cast<std::size_t>(t->rows) * t->row_bytes;
        ++l.n;
      }
    }
    if (l.n == 0)
      return;
    HIP_OK(hipEventRecord(pf_fork, stream));
    HIP_OK(hipStreamWaitEvent(pf_stream, pf_fork, 0));
    TouchKernel<<<kPrefetchBlocks, 256, 0, pf_stream>>>(l, pf_sink);
    pf_open = true;
  }
  void PrefetchJoin() {
    if (!pf_open)
      return;
    HIP_OK(hipEventRecord(pf_join, pf_stream));
    HIP_OK(hipStreamWaitEvent(stream, pf_join, 0));
    pf_open = false;
  }
  struct MultiMat {
    const Tensor* w;
    const float* x;
    float* y;
    int ldy;
  };
  signed char* xqm{nullptr};
  float* xqsm{nullptr};
  bool DenseMulti(const MultiMat* mats, int count, int n) {
    static const bool enabled = std::getenv("INKLING_FUSE_PROJ") == nullptr ||
                                std::getenv("INKLING_FUSE_PROJ")[0] != '0';
    if (!enabled || no_dp4 || n < 1 || n > 8 || count > 6)
      return false;
    const int cols = 2304;
    for (int i = 0; i < count; ++i)
      if (mats[i].w->type != kTypeQ8 ||
          static_cast<int>(mats[i].w->cols) != cols)
        return false;
    if (xqm == nullptr)
      return false;
    const float* xin[3] = {nullptr, nullptr, nullptr};
    int nx = 0;
    MultiArgs a{};
    a.nseg = count;
    a.cols = cols;
    a.rows_cap = 8;
    a.xq = xqm;
    a.xs = xqsm;
    int block = 0;
    const int rpb = kThreads * 2 / 32;
    for (int i = 0; i < count; ++i) {
      int xi = -1;
      for (int j = 0; j < nx; ++j)
        if (xin[j] == mats[i].x)
          xi = j;
      if (xi < 0) {
        if (nx == 3)
          return false;
        xin[nx] = mats[i].x;
        xi = nx++;
      }
      a.seg[i] = {static_cast<const unsigned char*>(mats[i].w->dev),
                  mats[i].w->row_bytes,
                  mats[i].y,
                  static_cast<std::size_t>(mats[i].ldy),
                  static_cast<int>(mats[i].w->rows),
                  xi,
                  block};
      block += (static_cast<int>(mats[i].w->rows) + rpb - 1) / rpb;
    }
    QuantRowsMulti<<<dim3(n, cols / 32, nx), 32, 0, stream>>>(
        xin[0], nx > 1 ? xin[1] : xin[0], nx > 2 ? xin[2] : xin[0], xqm, xqsm,
        cols, 8);
    switch (n) {
#define INKLING_MULTI(NT)                                      \
  case NT:                                                     \
    GemmDp4Q8Multi<NT><<<block, kThreads * 2, 0, stream>>>(a); \
    break;
      INKLING_MULTI(1)
      INKLING_MULTI(2)
      INKLING_MULTI(3)
      INKLING_MULTI(4)
      INKLING_MULTI(5)
      INKLING_MULTI(6)
      INKLING_MULTI(7)
      INKLING_MULTI(8)
#undef INKLING_MULTI
      default:
        return false;
    }
    return true;
  }
  void Dense(const Tensor& w, const float* xin, std::size_t ldx, float* yout,
             std::size_t ldy, int n, int rows = -1,
             const __half* xhin = nullptr) {
    const int nrows = rows < 0 ? static_cast<int>(w.rows) : rows;
    if (n == 1 && w.type == kTypeQ8 && ldx == w.cols && !no_dp4) {
      if (!(xq_reuse && xin == xq_src && n == xq_n && w.cols == xq_cols)) {
        QuantRows<<<dim3(1, w.cols / 32), 32, 0, stream>>>(
            xin, xq, xqs, static_cast<int>(w.cols));
        xq_src = xin;
        xq_n = n;
        xq_cols = static_cast<int>(w.cols);
      }
      const int rpb8 = kThreads * 2 / 32;
      GemmDp4<kTypeQ8>
          <<<dim3((nrows + rpb8 - 1) / rpb8, 1), kThreads * 2, 0, stream>>>(
              static_cast<const unsigned char*>(w.dev), w.row_bytes,
              static_cast<int>(w.cols), nrows, nullptr, nullptr, nullptr, xq,
              xqs, yout, gemm_skip);
      return;
    }
    if (n >= 2 && n <= 8 && w.type == kTypeQ8 && ldx == w.cols && !no_dp4) {
      if (!(xq_reuse && xin == xq_src && n == xq_n && w.cols == xq_cols)) {
        QuantRows<<<dim3(n, w.cols / 32), 32, 0, stream>>>(
            xin, xq, xqs, static_cast<int>(w.cols));
        xq_src = xin;
        xq_n = n;
        xq_cols = static_cast<int>(w.cols);
      }
      const int rpb8 = kThreads * 2 / 32;
      const dim3 g8((nrows + rpb8 - 1) / rpb8);
      const auto* wp = static_cast<const unsigned char*>(w.dev);
#define INKLING_DP4N(NT)                                                       \
  case NT:                                                                     \
    GemmDp4Q8N<NT><<<g8, kThreads * 2, 0, stream>>>(                           \
        wp, w.row_bytes, static_cast<int>(w.cols), nrows, xq, xqs, yout, ldy); \
    return;
      switch (n) {
        INKLING_DP4N(2)
        INKLING_DP4N(3)
        INKLING_DP4N(4)
        INKLING_DP4N(5)
        INKLING_DP4N(6)
        INKLING_DP4N(7)
        INKLING_DP4N(8)
        default:
          break;
      }
#undef INKLING_DP4N
    }
    if (n >= kWmmaMinTokens && nrows % kTileN == 0 &&
        ldy == static_cast<std::size_t>(nrows) &&
        static_cast<std::uint64_t>(nrows) <= w.rows) {
      const std::size_t xcount = static_cast<std::size_t>(n) * w.cols;
      const __half* xsrc = xhin;
      if (xsrc == nullptr || ldx != w.cols) {
        ToHalf<<<(xcount + 255) / 256, 256, 0, stream>>>(xin, xh, xcount);
        xsrc = xh;
      }
      if (w.type == kTypeQ8 && ldx == w.cols && w.cols % 64 == 0 &&
          dense_routed) {
        int ntiles = 0;
        const int* bounds = DenseMaps(n, dense_bn, &ntiles);
        if (gufo::models::qwen38_flash_next::rocm::RoutedQ8RepGemm(
                w.dev, xsrc, dense_tiles, static_cast<std::uint32_t>(ntiles),
                static_cast<std::uint32_t>(dense_bn), bounds, dense_ident,
                dense_ident, yout, static_cast<std::size_t>(nrows), w.cols,
                stream))
          return;
      }
      WmmaArgs wa{static_cast<const unsigned char*>(w.dev),
                  w.row_bytes,
                  static_cast<int>(w.cols),
                  nrows,
                  xsrc,
                  1,
                  yout,
                  nullptr,
                  nullptr,
                  nullptr,
                  n};
      const dim3 wg(nrows / kTileN, (n + kTileM - 1) / kTileM);
      if (ldx == w.cols) {
        switch (w.type) {
          case kTypeF32: {
            static const bool split_on =
                std::getenv("INKLING_SPLITK") == nullptr ||
                std::getenv("INKLING_SPLITK")[0] != '0';
            constexpr int kSplit = 4;
            if (split_on && nrows <= 128 && w.cols >= 2048 &&
                w.cols % (32 * kSplit) == 0 && n >= kWmmaMinTokens &&
                ksbuf != nullptr) {
              wa.ksplit = kSplit;
              wa.kchunk = static_cast<int>(w.cols) / kSplit;
              wa.y = ksbuf;
              const dim3 wgs(wg.x, wg.y, kSplit);
              WmmaGemm<kTypeF32><<<wgs, kWmmaThreads, 0, stream>>>(wa);
              const std::size_t count = static_cast<std::size_t>(n) * nrows;
              SplitKReduce<<<(count + 255) / 256, 256, 0, stream>>>(
                  ksbuf, yout, count, kSplit);
              return;
            }
            WmmaGemm<kTypeF32><<<wg, kWmmaThreads, 0, stream>>>(wa);
            return;
          }
          case kTypeBf16:
            WmmaGemm<kTypeBf16><<<wg, kWmmaThreads, 0, stream>>>(wa);
            return;
          case kTypeQ8:
            WmmaGemm<kTypeQ8><<<wg, kWmmaThreads, 0, stream>>>(wa);
            return;
          default:
            break;
        }
      }
    }
    const int rpb = kThreads * 2 / 32;
    const dim3 grid((nrows + rpb - 1) / rpb, (n + kTokenTile - 1) / kTokenTile);
    const auto* wp = static_cast<const unsigned char*>(w.dev);
    switch (w.type) {
      case kTypeF32:
        GemmDense<kTypeF32><<<grid, kThreads * 2, 0, stream>>>(
            wp, w.row_bytes, w.cols, nrows, xin, ldx, yout, ldy, n);
        break;
      case kTypeBf16:
        GemmDense<kTypeBf16><<<grid, kThreads * 2, 0, stream>>>(
            wp, w.row_bytes, w.cols, nrows, xin, ldx, yout, ldy, n);
        break;
      case kTypeQ8:
        GemmDense<kTypeQ8><<<grid, kThreads * 2, 0, stream>>>(
            wp, w.row_bytes, w.cols, nrows, xin, ldx, yout, ldy, n);
        break;
      default:
        std::abort();
    }
  }
  void Routed(const Tensor& w, int rows, int items, const float* scale,
              const float* xin, float* yout) {
    {
      const int nx =
          (xrow == xrow_token) ? items / m->config.expert_used : items;
      QuantRows<<<dim3(nx, w.cols / 32), 32, 0, stream>>>(
          xin, xq, xqs, static_cast<int>(w.cols));
      const int rpb8 = kThreads * 2 / 32;
      const dim3 g8((rows + rpb8 - 1) / rpb8, items);
      if (w.type == kTypeNvfp4) {
        GemmDp4<kTypeNvfp4><<<g8, kThreads * 2, 0, stream>>>(
            static_cast<const unsigned char*>(w.dev), w.row_bytes,
            static_cast<int>(w.cols), rows, ids_flat, xrow, scale, xq, xqs,
            yout, nullptr, expert_nt && items <= m->config.expert_used);
      } else {
        GemmDp4<kTypeQ8><<<g8, kThreads * 2, 0, stream>>>(
            static_cast<const unsigned char*>(w.dev), w.row_bytes,
            static_cast<int>(w.cols), rows, ids_flat, xrow, scale, xq, xqs,
            yout);
      }
      return;
    }
    const int rpb = kThreads * 2 / 32;
    const dim3 grid((rows + rpb - 1) / rpb, items);
    const auto* wp = static_cast<const unsigned char*>(w.dev);
    if (w.type == kTypeNvfp4) {
      GemmRouted<kTypeNvfp4><<<grid, kThreads * 2, 0, stream>>>(
          wp, w.row_bytes, w.cols, rows, ids_flat, xrow, scale, xin, yout);
    } else {
      GemmRouted<kTypeQ8><<<grid, kThreads * 2, 0, stream>>>(
          wp, w.row_bytes, w.cols, rows, ids_flat, xrow, scale, xin, yout);
    }
  }
  int* ids_flat{nullptr};
  void RmsRows(const float* in_, const float* w, float* out, int rows, int dim,
               std::size_t ld_in, std::size_t ld_out, __half* out_h = nullptr) {
    if (dim >= 1024 && dim <= kRmsVecMaxDim && dim % 4 == 0 && ld_in % 4 == 0 &&
        ld_out % 4 == 0) {
      RmsVec<<<rows, 256, 0, stream>>>(in_, w, out, dim, ld_in, ld_out,
                                       m->config.rms_eps, out_h);
      return;
    }
    Rms<<<rows, 256, 0, stream>>>(in_, w, out, dim, ld_in, ld_out,
                                  m->config.rms_eps, out_h);
  }
  void Mixer(const std::vector<const float*>& blocks, const float* cur_,
             const float* query, int nq, int n, float* out) {
    StatePtrs sp{};
    sp.count = static_cast<int>(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
      sp.p[i] = blocks[i];
    sp.half_blocks = F16Blocks(n);
    if (n >= kWmmaMinTokens) {
      MixBig<<<n, 256, 0, stream>>>(
          sp, cur_, query, nq, static_cast<int>(m->config.hidden_size), n, out);
    } else {
      static const int mix_threads =
          std::getenv("INKLING_MIX_THREADS") != nullptr
              ? std::atoi(std::getenv("INKLING_MIX_THREADS"))
              : 576;
      static const int mix_split =
          std::getenv("INKLING_MIX_SPLIT") != nullptr
              ? std::atoi(std::getenv("INKLING_MIX_SPLIT"))
              : 4;
      Mix<<<dim3(n, mix_split), mix_threads, 0, stream>>>(
          sp, cur_, query, nq, static_cast<int>(m->config.hidden_size), n, out);
    }
  }
  void MixRmsLaunch(const std::vector<const float*>& blocks, const float* cur_,
                    const float* query, int nq, int n, const NormPtrs& np) {
    StatePtrs sp{};
    sp.count = static_cast<int>(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
      sp.p[i] = blocks[i];
    sp.half_blocks = F16Blocks(n);
    const int hh = static_cast<int>(m->config.hidden_size);
    MixRms<<<n, 256, nq * hh * sizeof(float), stream>>>(
        sp, cur_, query, nq, hh, n, np, m->config.rms_eps);
  }
  void ControlLaunch(const CtlBatch& cb, int groups, bool half_w, int n) {
    const int h = static_cast<int>(m->config.hidden_size);
    if (half_w) {
      ControlLow<true><<<dim3(n, kControlRank / kControlWaves, groups),
                         kControlWaves * 32, 0, stream>>>(cb, h,
                                                          m->config.rms_eps);
      ControlApply<true>
          <<<dim3(n, (h + 255) / 256, groups), 256, 0, stream>>>(cb, h);
    } else {
      ControlLow<false><<<dim3(n, kControlRank / kControlWaves, groups),
                          kControlWaves * 32, 0, stream>>>(cb, h,
                                                           m->config.rms_eps);
      ControlApply<false>
          <<<dim3(n, (h + 255) / 256, groups), 256, 0, stream>>>(cb, h);
    }
  }
  // the three attention-input controllers of a visit in one launch pair;
  // false when the small-batch half-weight path does not apply
  bool ControlStep3(float* const cur_[3], const float* msgs3, std::uint32_t phys,
                    int n) {
    static const char* const kGroup[3] = {"attn", "k", "v"};
    if (n >= kWmmaMinTokens || m->config.loop_gate_rank != kControlRank)
      return false;
    CtlBatch cb{};
    for (int j = 0; j < 3; ++j) {
      const std::string fb =
          "blk." + std::to_string(phys) + ".loop_" + kGroup[j] + "_";
      if (m->tensors.count(fb + "down.weight.h") == 0)
        return false;
      cb.cur[j] = cur_[j];
      cb.msg[j] = msgs3 + static_cast<std::size_t>(j) * n *
                              m->config.hidden_size;
      cb.down[j] = m->T(fb + "down.weight.h").dev;
      cb.gain[j] = m->T(fb + "gain.weight.h").dev;
      cb.forget[j] = m->T(fb + "forget.weight.h").dev;
      cb.low[j] = low + static_cast<std::size_t>(j) * kMaxVerifyRows *
                            kControlRank;
    }
    ControlLaunch(cb, 3, true, n);
    return true;
  }
  void ControlStep(float* cur_, const float* message, std::uint32_t phys,
                   const char* group, int n) {
    const int h = static_cast<int>(m->config.hidden_size);
    const bool big = n >= kWmmaMinTokens;
    if (!big && m->config.loop_gate_rank == kControlRank) {
      const std::string fb =
          "blk." + std::to_string(phys) + ".loop_" + group + "_";
      static const bool ctl_half = std::getenv("INKLING_CTL_HALF") == nullptr ||
                                   std::getenv("INKLING_CTL_HALF")[0] != '0';
      const bool half_w =
          ctl_half && m->tensors.count(fb + "down.weight.h") != 0;
      CtlBatch cb{};
      cb.cur[0] = cur_;
      cb.msg[0] = message;
      cb.down[0] = m->T(fb + "down.weight" + (half_w ? ".h" : "")).dev;
      cb.gain[0] = m->T(fb + "gain.weight" + (half_w ? ".h" : "")).dev;
      cb.forget[0] = m->T(fb + "forget.weight" + (half_w ? ".h" : "")).dev;
      cb.low[0] = low;
      ControlLaunch(cb, 1, half_w, n);
      return;
    }
    RmsRows(cur_, nullptr, big ? nullptr : cat, n, h, h, 2 * h,
            big ? cath : nullptr);
    RmsRows(message, nullptr, big ? nullptr : cat + h, n, h, h, 2 * h,
            big ? cath + h : nullptr);
    const std::string base =
        "blk." + std::to_string(phys) + ".loop_" + group + "_";
    Dense(m->T(base + "down.weight"), cat, 2 * h, low, m->config.loop_gate_rank,
          n, -1, big ? cath : nullptr);
    if (big && m->config.loop_gate_rank == kControlRank && h % kTileN == 0) {
      const std::size_t lowcount = static_cast<std::size_t>(n) * kControlRank;
      ToHalf<<<(lowcount + 255) / 256, 256, 0, stream>>>(low, xh, lowcount);
      ControlGemm<<<dim3(h / kTileN, (n + kTileM - 1) / kTileM), kWmmaThreads,
                    0, stream>>>(
          xh, static_cast<const float*>(m->T(base + "gain.weight").dev),
          static_cast<const float*>(m->T(base + "forget.weight").dev), cur_,
          message, h, n);
      return;
    }
    Dense(m->T(base + "gain.weight"), low, m->config.loop_gate_rank, gain, h,
          n);
    Dense(m->T(base + "forget.weight"), low, m->config.loop_gate_rank, forget,
          h, n);
    Control<<<n, 256, 0, stream>>>(cur_, message, gain, forget, h);
  }
  void Convolve(const float* in_, const Tensor& w, const float* dyn, int heads,
                float* hist, float* out, int ch, int n,
                const float* add = nullptr) {
    if (n >= kWmmaMinTokens) {
      ConvT<<<dim3((ch + 127) / 128, n), 128, 0, stream>>>(
          in_, static_cast<const float*>(w.dev), dyn, heads, hist, out, ch,
          add);
      ConvHist<<<(ch + 127) / 128, 128, 0, stream>>>(in_, hist, ch, n);
      return;
    }
    Conv<<<(ch + 127) / 128, 128, 0, stream>>>(
        in_, static_cast<const float*>(w.dev), dyn, heads, hist, out, ch, n,
        add,
        all_rows && n <= static_cast<int>(kMaxVerifyRows)
            ? hist + static_cast<std::size_t>(ch) * kHistory
            : nullptr);
  }
  float* Pool() { return block_pool[pool_used++]; }
  // Opt-in fast profile (INKLING_F16_BLOCKS=1): prefill block states live as
  // FP16 rows, the large reads of MixBig/MixRms (logits KL ~2.7e-3).
  static bool F16Blocks(int n) {
    static const bool on = std::getenv("INKLING_F16_BLOCKS") != nullptr &&
                           std::getenv("INKLING_F16_BLOCKS")[0] != '0';
    return on && n >= kWmmaMinTokens;
  }

  struct LayerCtx {
    std::uint32_t slot{0}, ip{0}, p{0};
    bool swa{false}, big{false};
    int n{0};
    std::size_t nh_{0};
    const float* msgs{nullptr};
    float* inp[3]{};
    __half* inh[3]{};
    const float* dk{nullptr};
    const float* dv{nullptr};
    float* vtmp{nullptr};
    std::uint64_t ticket{0};
    bool npu_on{false};
    bool gate_done{false};
    int vi{0};  // index of the visit in the decode step
    bool out_npu{false};
  };

  void BeginLayer(LayerCtx& L, std::uint32_t slot, std::uint32_t ip,
                  const float* msgs_, int n) {
    L.slot = slot;
    L.ip = ip;
    L.p = m->Phys(slot);
    L.swa = m->config.is_swa[slot];
    L.n = n;
    L.nh_ = static_cast<std::size_t>(n) * m->config.hidden_size;
    L.big = n >= kWmmaMinTokens;
    L.msgs = msgs_;
    L.dk = L.dv = nullptr;
  }

  // stage 1: attention input mix and norms -> L.inp / L.inh
  void LayerFront(LayerCtx& L, std::vector<const float*>& blocks) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = L.p;
    const bool swa = L.swa;
    Slot& sl = slots[L.slot];
    const int n = L.n;
    const std::size_t nh_ = L.nh_;
    const bool big = L.big;
    const float* msgs_ = L.msgs;
    (void)c;
    (void)h;
    (void)nh;
    (void)nkv;
    (void)hd;
    (void)nrel;
    (void)p;
    (void)swa;
    (void)sl;
    (void)n;
    (void)nh_;
    (void)big;
    (void)msgs_;
    float* inp[3] = {in[0], in[1], in[2]};
    const bool has_res = m->Has(m->P(p, "attn_res_query.weight"));
    __half* inh[3] = {big ? xh0 : nullptr, big ? xh1 : nullptr,
                      big ? xh2 : nullptr};
    const bool fused_attn = has_res && big && msgs_ == nullptr;
    if (fused_attn) {
      const int nq = swa ? 1 : 3;
      NormPtrs np{};
      if (swa) {
        np.w[0] =
            static_cast<const float*>(m->T(m->P(p, "attn_norm.weight")).dev);
        np.out[0] = mixed;
        np.outh[0] = xh0;
        inp[0] = inp[1] = inp[2] = mixed;
        inh[1] = inh[2] = xh0;
      } else {
        static const char* const kN[3] = {"attn_q_inp_norm.weight",
                                          "attn_k_inp_norm.weight",
                                          "attn_v_inp_norm.weight"};
        for (int j = 0; j < 3; ++j) {
          np.w[j] = static_cast<const float*>(m->T(m->P(p, kN[j])).dev);
          np.out[j] = mixed + j * nh_;
          np.outh[j] = inh[j];
          inp[j] = mixed + j * nh_;
        }
      }
      MixRmsLaunch(
          blocks, cur,
          static_cast<const float*>(m->T(m->P(p, "attn_res_query.weight")).dev),
          nq, n, np);
    } else {
      if (has_res) {
        const int nq = swa ? 1 : 3;
        Mixer(blocks, cur,
              static_cast<const float*>(
                  m->T(m->P(p, "attn_res_query.weight")).dev),
              nq, n, mixed);
        for (int j = 0; j < 3; ++j)
          inp[j] = mixed + (swa ? 0 : j) * nh_;
        if (swa && msgs_ != nullptr) {
          for (int j = 1; j < 3; ++j) {
            HIP_OK(hipMemcpyAsync(in[j], mixed, nh_ * 4,
                                  hipMemcpyDeviceToDevice, stream));
            inp[j] = in[j];
          }
        }
      } else {
        for (int j = 0; j < 3; ++j) {
          HIP_OK(hipMemcpyAsync(in[j], cur, nh_ * 4, hipMemcpyDeviceToDevice,
                                stream));
          inp[j] = in[j];
        }
      }
      if (msgs_ != nullptr) {
        static const char* const kGroup[3] = {"attn", "k", "v"};
        static const bool ctl3 = std::getenv("INKLING_CTL_BATCH") == nullptr ||
                                 std::getenv("INKLING_CTL_BATCH")[0] != '0';
        const bool distinct = inp[0] != inp[1] && inp[1] != inp[2];
        if (!(ctl3 && distinct && ControlStep3(inp, msgs_, p, n)))
          for (int j = 0; j < 3; ++j)
            ControlStep(inp[j], msgs_ + j * nh_, p, kGroup[j], n);
      }
      if (swa) {
        const float* w =
            static_cast<const float*>(m->T(m->P(p, "attn_norm.weight")).dev);
        if (msgs_ != nullptr) {
          for (int j = 0; j < 3; ++j)
            RmsRows(inp[j], w, inp[j], n, h, h, h, inh[j]);
        } else {
          RmsRows(inp[0], w, inp[0], n, h, h, h, inh[0]);
          inp[1] = inp[2] = inp[0];
          inh[1] = inh[2] = inh[0];
        }
      } else {
        static const char* const kN[3] = {"attn_q_inp_norm.weight",
                                          "attn_k_inp_norm.weight",
                                          "attn_v_inp_norm.weight"};
        for (int j = 0; j < 3; ++j)
          RmsRows(inp[j], static_cast<const float*>(m->T(m->P(p, kN[j])).dev),
                  inp[j], n, h, h, h, inh[j]);
      }
    }
    for (int j = 0; j < 3; ++j) {
      L.inp[j] = inp[j];
      L.inh[j] = inh[j];
    }
  }

  // Encode the layer inputs and queue the q/gate/k/v slices on the NPU (first
  // row chunk); NpuQkvEnd waits for them and decodes the results.
  void NpuQkvBegin(LayerCtx& L, int r0) {
    namespace rocm = gufo::models::qwen38_flash_next::rocm;
    auto& npu = *m->npu;
    const std::uint32_t h = m->config.hidden_size;
    const int n = L.n;
    const auto& wq = m->npu_w.at(m->P(L.p, "attn_q.weight"));
    const auto& wg = m->npu_w.at(m->P(L.p, "attn_gate.weight"));
    const auto& wk = m->npu_w.at(m->P(L.p, "attn_k.weight"));
    const auto& wv = m->npu_w.at(m->P(L.p, "attn_v.weight"));
    const bool k_shared = L.inp[1] == L.inp[0];
    const bool v_shared = L.inp[2] == L.inp[0];
    const int rows = std::min<int>(m->npu_tokens, n - r0);
    const std::uint32_t enc =
        static_cast<std::uint32_t>((rows + 127) / 128 * 128);
    rocm::NpuEncodeActivations(L.inh[0] + static_cast<std::size_t>(r0) * h, h,
                               enc, h, 0, h, npu.Device(npu_a[0]), stream);
    if (!k_shared)
      rocm::NpuEncodeActivations(L.inh[1] + static_cast<std::size_t>(r0) * h, h,
                                 enc, h, 0, h, npu.Device(npu_a[1]), stream);
    if (!v_shared)
      rocm::NpuEncodeActivations(L.inh[2] + static_cast<std::size_t>(r0) * h, h,
                                 enc, h, 0, h, npu.Device(npu_a[2]), stream);
    std::vector<int> b;
    std::vector<int> c;
    if ((npu_mask & 1) != 0) {
      for (std::size_t i = 0; i < wq.size(); ++i) {
        b.push_back(wq[i]);
        c.push_back(npu_c[i]);
      }
    }
    if ((npu_mask & 2) != 0) {
      for (std::size_t i = 0; i < wg.size(); ++i) {
        b.push_back(wg[i]);
        c.push_back(npu_c[3 + i]);
      }
    }
    if (k_shared && (npu_mask & 4) != 0) {
      b.push_back(wk[0]);
      c.push_back(npu_c[6]);
    }
    if (v_shared && (npu_mask & 8) != 0) {
      b.push_back(wv[0]);
      c.push_back(npu_c[7]);
    }
    const int prog =
        (rows <= 2048 && m->npu_prog_small >= 0) ? m->npu_prog_small : 0;
    if (!b.empty())
      NpuSubmit(npu_a[0], b.data(), c.data(), static_cast<int>(b.size()), prog);
    if (!k_shared && (npu_mask & 4) != 0)
      NpuSubmit(npu_a[1], &wk[0], &npu_c[6], 1, prog);
    if (!v_shared && (npu_mask & 8) != 0)
      NpuSubmit(npu_a[2], &wv[0], &npu_c[7], 1, prog);
    L.ticket = npu.Ticket();
  }

  void NpuQkvEnd(LayerCtx& L, int r0) {
    namespace rocm = gufo::models::qwen38_flash_next::rocm;
    auto& npu = *m->npu;
    const int rows = std::min<int>(m->npu_tokens, L.n - r0);
    const auto& wq = m->npu_w.at(m->P(L.p, "attn_q.weight"));
    const auto& wg = m->npu_w.at(m->P(L.p, "attn_gate.weight"));
    const int qd = static_cast<int>(wq.size()) * static_cast<int>(kNpuSlice);
    NpuWaitFor(L.ticket);
    if ((npu_mask & 1) != 0)
      for (std::size_t i = 0; i < wq.size(); ++i)
        rocm::NpuDecodeOutput(npu.Staged(stream, npu_c[i]),
                              q + static_cast<std::size_t>(r0) * qd, rows,
                              kNpuSlice, qd, i * kNpuSlice, false, stream);
    if ((npu_mask & 2) != 0)
      for (std::size_t i = 0; i < wg.size(); ++i)
        rocm::NpuDecodeOutput(npu.Staged(stream, npu_c[3 + i]),
                              gate + static_cast<std::size_t>(r0) * qd, rows,
                              kNpuSlice, qd, i * kNpuSlice, false, stream);
    if ((npu_mask & 4) != 0)
      rocm::NpuDecodeOutput(npu.Staged(stream, npu_c[6]),
                            tmp + static_cast<std::size_t>(r0) * kNpuSlice,
                            rows, kNpuSlice, kNpuSlice, 0, false, stream);
    if ((npu_mask & 8) != 0)
      rocm::NpuDecodeOutput(npu.Staged(stream, npu_c[7]),
                            v + static_cast<std::size_t>(r0) * kNpuSlice, rows,
                            kNpuSlice, kNpuSlice, 0, false, stream);
  }

  void DumpAttn(std::uint32_t p, int n, const float* q, const float* r,
                const float* sink, const Slot& sl) {
    static const char* const dir = std::getenv("INKLING_ATTN_DUMP_DIR");
    static const int layer =
        std::getenv("INKLING_ATTN_DUMP_LAYER") != nullptr
            ? std::atoi(std::getenv("INKLING_ATTN_DUMP_LAYER"))
            : -1;
    static const int pos = std::getenv("INKLING_ATTN_DUMP_POS") != nullptr
                               ? std::atoi(std::getenv("INKLING_ATTN_DUMP_POS"))
                               : -1;
    static bool dumped = false;
    constexpr int kDumpRows = 128;
    if (dir == nullptr || dumped || layer < 0 || pos < 0 ||
        p != static_cast<std::uint32_t>(layer) ||
        pos_base != static_cast<std::uint32_t>(pos) || n < kDumpRows)
      return;
    dumped = true;
    HIP_OK(hipStreamSynchronize(stream));
    const Config& c = m->config;
    const int nh = c.num_heads, nkv = c.num_kv, hd = c.head_dim;
    const int nrel = c.relative_length;
    const int cache_rows =
        sl.ring > 0 ? sl.ring
                    : std::min<int>(static_cast<int>(pos_base) + kDumpRows,
                                    sl.vblocks * 64);
    const int nb = (cache_rows + 63) / 64;
    std::vector<float> dump_q(static_cast<std::size_t>(kDumpRows) * nh * hd);
    std::vector<float> dump_r(static_cast<std::size_t>(kDumpRows) * nh * nrel);
    std::vector<float> dump_s(static_cast<std::size_t>(kDumpRows) * nh);
    std::vector<uint16_t> dump_k(static_cast<std::size_t>(cache_rows) * nkv *
                                 hd);
    std::vector<uint16_t> dump_vb(static_cast<std::size_t>(nkv) * nb * hd * 64);
    std::vector<uint16_t> dump_v(static_cast<std::size_t>(cache_rows) * nkv *
                                 hd);
    HIP_OK(
        hipMemcpy(dump_q.data(), q, dump_q.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(
        hipMemcpy(dump_r.data(), r, dump_r.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(dump_s.data(), sink, dump_s.size() * 4,
                     hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(dump_k.data(), sl.k, dump_k.size() * 2,
                     hipMemcpyDeviceToHost));
    for (int g = 0; g < nkv; ++g)
      HIP_OK(hipMemcpy(
          dump_vb.data() + static_cast<std::size_t>(g) * nb * hd * 64,
          sl.v + static_cast<std::size_t>(g) * sl.vblocks * hd * 64,
          static_cast<std::size_t>(nb) * hd * 64 * 2, hipMemcpyDeviceToHost));
    const Tensor& tb = m->T(m->P(p, "attn_rel_bias.weight.h"));
    const std::size_t bias_count =
        static_cast<std::size_t>(tb.rows) * tb.row_bytes / 2;
    std::vector<uint16_t> hb(bias_count);
    HIP_OK(hipMemcpy(hb.data(), tb.dev, bias_count * 2, hipMemcpyDeviceToHost));
    std::vector<float> bias(bias_count);
    for (std::size_t i = 0; i < bias_count; ++i)
      bias[i] = HalfToFloat(hb[i]);
    std::vector<uint16_t> vk(dump_k.size()), vv(dump_v.size());
    for (std::size_t i = 0; i < dump_k.size(); ++i)
      vk[i] = FloatToBf16(HalfToFloat(dump_k[i]));
    for (int j = 0; j < cache_rows; ++j) {
      const int slot = sl.ring > 0 ? j % sl.ring : j;
      for (int g = 0; g < nkv; ++g) {
        for (int d = 0; d < hd; ++d) {
          const std::size_t src =
              (static_cast<std::size_t>(g) * nb + (slot >> 6)) * hd * 64 +
              static_cast<std::size_t>(d) * 64 + (slot & 63);
          vv[(static_cast<std::size_t>(j) * nkv + g) * hd + d] =
              FloatToBf16(HalfToFloat(dump_vb[src]));
        }
      }
    }
    const std::string root(dir);
    WriteDump(root + "/Q.bin", dump_q);
    WriteDump(root + "/R.bin", dump_r);
    WriteDump(root + "/S.bin", dump_s);
    WriteDump(root + "/Bias.bin", bias);
    WriteDump(root + "/K.bin", vk);
    WriteDump(root + "/V.bin", vv);
    std::fprintf(stderr,
                 "inkling: dumped attn layer %u pos %u rows %d keys %d\n", p,
                 pos_base, kDumpRows, cache_rows);
  }

  // stage 2: q/k/v/gate/rel/sink projections (GPU or NPU)
  void LayerQkvBegin(LayerCtx& L) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = L.p;
    const bool swa = L.swa;
    Slot& sl = slots[L.slot];
    const int n = L.n;
    const std::size_t nh_ = L.nh_;
    const bool big = L.big;
    float* const* inp = L.inp;
    __half* const* inh = L.inh;
    const float* msgs_ = L.msgs;
    (void)c;
    (void)h;
    (void)nh;
    (void)nkv;
    (void)hd;
    (void)nrel;
    (void)p;
    (void)swa;
    (void)sl;
    (void)n;
    (void)nh_;
    (void)big;
    (void)inp;
    (void)inh;
    (void)msgs_;
    L.npu_on =
        npu_a[0] >= 0 && big && m->npu_w.count(m->P(p, "attn_q.weight")) != 0;
    if (L.npu_on)
      NpuQkvBegin(L, 0);
    xq_src = nullptr;
    xq_reuse = true;
    bool fused = false;
    L.gate_done = false;
    if (!L.npu_on && n <= static_cast<int>(kMaxVerifyRows)) {
      const MultiMat mats[5] = {
          {&m->T(m->P(p, "attn_q.weight")), inp[0], q, nh * hd},
          {&m->T(m->P(p, "attn_rel.weight")), inp[0], r, nrel * nh},
          {&m->T(m->P(p, "attn_k.weight")), inp[1], tmp, nkv * hd},
          {&m->T(m->P(p, "attn_v.weight")), inp[2], v, nkv * hd},
          {&m->T(m->P(p, "attn_gate.weight")), inp[0], gate, nh * hd}};
      fused = DenseMulti(mats, 5, n);
      L.gate_done = fused;
    }
    if (!fused && (!L.npu_on || (npu_mask & 1) == 0))
      Dense(m->T(m->P(p, "attn_q.weight")), inp[0], h, q, nh * hd, n, -1,
            inh[0]);
    if (!fused)
      Dense(m->T(m->P(p, "attn_rel.weight")), inp[0], h, r, nrel * nh, n, -1,
            inh[0]);
    if (!fused && (!L.npu_on || (npu_mask & 4) == 0))
      Dense(m->T(m->P(p, "attn_k.weight")), inp[1], h, tmp, nkv * hd, n, -1,
            inh[1]);
    if (!swa)
      Dense(m->T(m->P(p, "attn_k_conv_proj.weight")), inp[1], h, dyn_k,
            kConvTaps * nkv, n, -1, inh[1]);
    if (!fused && (!L.npu_on || (npu_mask & 8) == 0))
      Dense(m->T(m->P(p, "attn_v.weight")), inp[2], h, v, nkv * hd, n, -1,
            inh[2]);
    if (!swa) {
      Dense(m->T(m->P(p, "attn_v_conv_proj.weight")), inp[2], h, dyn_v,
            kConvTaps * nkv, n, -1, inh[2]);
      L.dk = dyn_k;
      L.dv = dyn_v;
    }
    Dense(m->T(m->P(p, "attn_sink_proj.weight")), inp[0], h, sink, nh, n, -1,
          inh[0]);
    xq_reuse = false;
    xq_src = nullptr;
  }

  void LayerQkvEnd(LayerCtx& L) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = L.p;
    const bool swa = L.swa;
    Slot& sl = slots[L.slot];
    const int n = L.n;
    const std::size_t nh_ = L.nh_;
    const bool big = L.big;
    float* const* inp = L.inp;
    __half* const* inh = L.inh;
    const float* msgs_ = L.msgs;
    (void)c;
    (void)h;
    (void)nh;
    (void)nkv;
    (void)hd;
    (void)nrel;
    (void)p;
    (void)swa;
    (void)sl;
    (void)n;
    (void)nh_;
    (void)big;
    (void)inp;
    (void)inh;
    (void)msgs_;
    if (L.npu_on) {
      NpuQkvEnd(L, 0);
      for (int r0 = static_cast<int>(m->npu_tokens); r0 < n;
           r0 += static_cast<int>(m->npu_tokens)) {
        NpuQkvBegin(L, r0);
        NpuQkvEnd(L, r0);
      }
    }
    if (!L.gate_done && (!L.npu_on || (npu_mask & 2) == 0))
      Dense(m->T(m->P(p, "attn_gate.weight")), inp[0], h, gate, nh * hd, n, -1,
            inh[0]);
  }

  // stage 3: convolutions, clipping, KV store, attention -> att / atth
  void LayerAttn(LayerCtx& L) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = L.p;
    const bool swa = L.swa;
    Slot& sl = slots[L.slot];
    const int n = L.n;
    const std::size_t nh_ = L.nh_;
    const bool big = L.big;
    float* const* inp = L.inp;
    __half* const* inh = L.inh;
    const float* msgs_ = L.msgs;
    (void)c;
    (void)h;
    (void)nh;
    (void)nkv;
    (void)hd;
    (void)nrel;
    (void)p;
    (void)swa;
    (void)sl;
    (void)n;
    (void)nh_;
    (void)big;
    (void)inp;
    (void)inh;
    (void)msgs_;
    float* vtmp = tmp;
    L.vtmp = vtmp;
    if (n < kWmmaMinTokens && !(all_rows && n <= static_cast<int>(kMaxVerifyRows))) {
      const int ch = nkv * hd;
      Conv2<<<(ch + 127) / 128, 128, 0, stream>>>(
          tmp, static_cast<const float*>(m->T(m->P(p, "attn_k_conv.weight")).dev),
          L.dk, sl.hist_k, k, v,
          static_cast<const float*>(m->T(m->P(p, "attn_v_conv.weight")).dev),
          L.dv, sl.hist_v, vtmp, nkv, ch, n);
    } else {
      Convolve(tmp, m->T(m->P(p, "attn_k_conv.weight")), L.dk, nkv, sl.hist_k,
               k, nkv * hd, n);
      Convolve(v, m->T(m->P(p, "attn_v_conv.weight")), L.dv, nkv, sl.hist_v,
               vtmp, nkv * hd, n);
    }
    const int* tau_p = swa ? nullptr : d_pos;
    const float tau_floor = static_cast<float>(c.log_scale_floor);
    if (big && hd <= 128 && nrel <= 128) {
      ClipWave<<<(n * nh + 7) / 8, 256, 0, stream>>>(
          q, static_cast<const float*>(m->T(m->P(p, "attn_q_clip.weight")).dev),
          nh, hd, c.qk_clip, tau_p, nh, tau_floor, c.log_scale_alpha, n * nh);
      ClipWave<<<(n * nkv + 7) / 8, 256, 0, stream>>>(
          k, nullptr, 1, hd, c.qk_clip, nullptr, 1, 1.0F, 0.0F, n * nkv);
      RmsWave<<<(n * nkv + 7) / 8, 256, 0, stream>>>(vtmp, vtmp, hd, n * nkv,
                                                     m->config.rms_eps);
      ClipWave<<<(n * nh + 7) / 8, 256, 0, stream>>>(
          r,
          static_cast<const float*>(m->T(m->P(p, "attn_rel_clip.weight")).dev),
          nh, nrel, c.relative_clip, tau_p, nh, tau_floor, c.log_scale_alpha,
          n * nh);
    } else {
      ClipJobs cj{};
      cj.job[0] = {q,
                   static_cast<const float*>(
                       m->T(m->P(p, "attn_q_clip.weight")).dev),
                   nh,
                   hd,
                   n * nh,
                   nh,
                   c.qk_clip,
                   tau_floor,
                   c.log_scale_alpha,
                   tau_p};
      cj.job[1] = {k,    nullptr, 1,    hd,   n * nkv,
                   1,    c.qk_clip, 1.0F, 0.0F, nullptr};
      cj.job[2] = {r,
                   static_cast<const float*>(
                       m->T(m->P(p, "attn_rel_clip.weight")).dev),
                   nh,
                   nrel,
                   n * nh,
                   nh,
                   c.relative_clip,
                   tau_floor,
                   c.log_scale_alpha,
                   tau_p};
      ClipMulti<<<cj.job[0].blocks + cj.job[1].blocks + cj.job[2].blocks, 128,
                  0, stream>>>(cj);
      RmsRows(vtmp, nullptr, vtmp, n * nkv, hd, hd, hd);
    }
    StoreKv<<<n, 256, 0, stream>>>(k, vtmp, sl.k, sl.v, nkv * hd, d_pos,
                                   sl.ring, sl.vblocks);
    DumpAttn(p, n, q, r, sink, sl);
    AttnArgs aa{nh,
                nkv,
                hd,
                nrel,
                static_cast<int>(c.relative_extent),
                swa ? static_cast<int>(c.sliding_window) : 0,
                sl.ring,
                sl.vblocks,
                d_pos,
                1.0F / std::sqrt(static_cast<float>(hd))};
    bool fused_gate = false;
    const int span =
        swa ? static_cast<int>(c.sliding_window) : static_cast<int>(ctx);
    if (hd == kFlashHd && nrel == kFlashRel) {
      const __half* bias_w = static_cast<const __half*>(
          m->T(m->P(p, "attn_rel_bias.weight.h")).dev);
      const bool sparse_now =
          sparse && !swa &&
          (n < kWmmaMinTokens ||
           pos_base + n >
               (kSparseSink + kSparseLocal + kSparseTop + 2) * kBlockKeys);
      if (sparse && !swa) {
        PoolKeys<<<dim3((n + kBlockKeys - 1) / kBlockKeys + 1, nkv), kFlashHd,
                   0, stream>>>(sl.k, sl.kmm, d_pos, n, nkv);
      }
      if (n <= static_cast<int>(kMaxVerifyRows) && !sparse_now &&
          !(n == 1 && swa) && n * (nh / nkv) <= 128) {
        AttnArgs pa = aa;
        pa.pack = nh / nkv;
        const int pk_keys = swa ? 256 : 512;
        const int pk_split =
            swa ? (span + 72 + pk_keys - 1) / pk_keys
                : std::max(1, static_cast<int>((ctx + pk_keys - 1) / pk_keys));
        FlashRel<kPackWaves>
            <<<dim3(1, nkv, pk_split), kPackWaves * 32, 0, stream>>>(
                pa, q, r, sink, sl.k, sl.v, bias_w, att, n, pk_split, pk_keys,
                split_m, split_l, split_o, nullptr, nullptr, 0);
        if (pk_split > 1)
          FlashCombine<<<dim3(n, nh), 128, 0, stream>>>(
              split_m, split_l, split_o, att, nh, pk_split);
      } else if (n == 1 && !sparse_now && nh / nkv <= kDecMaxGh) {
        const int nsd = (span + kDecKeys - 1) / kDecKeys;
        DecodeAttn<<<dim3(nsd, nkv), kDecThreads, 0, stream>>>(
            aa, q, r, sink, sl.k, sl.v, bias_w, split_m, split_l, split_o, nsd);
        FlashCombine<<<dim3(1, nh), 128, 0, stream>>>(split_m, split_l, split_o,
                                                      att, nh, nsd);
      } else if (sparse_now) {
        const int ntile = (n + kFlashQ - 1) / kFlashQ;
        const bool grouped = n <= static_cast<int>(kMaxVerifyRows) &&
                             std::getenv("INKLING_SPARSE_PER_HEAD") == nullptr;
        const int gh = nh / nkv;
        SelectBlocks<<<dim3(ntile, grouped ? nkv : nh), 256,
                       kMaxSparseBlocks * sizeof(float), stream>>>(
            q, sl.kmm, d_pos, n, nh, nkv, blist, bcount, grouped ? gh : 0);
        const int nsp = n < kWmmaMinTokens ? (grouped ? 64 : 32) : 1;
        if (grouped) {
          AttnArgs pa = aa;
          pa.pack = gh;
          FlashRel<kPackWaves>
              <<<dim3(1, nkv, nsp), kPackWaves * 32, 0, stream>>>(
                  pa, q, r, sink, sl.k, sl.v, bias_w, att, n, nsp, kSplitKeys,
                  split_m, split_l, split_o, blist, bcount, kListCap);
        } else if (n < kWmmaMinTokens) {
          FlashRel<kPackWaves>
              <<<dim3(1, nh, nsp), kPackWaves * 32, 0, stream>>>(
                  aa, q, r, sink, sl.k, sl.v, bias_w, att, n, nsp, kSplitKeys,
                  split_m, split_l, split_o, blist, bcount, kListCap);
        } else {
          FlashRel<8><<<dim3((n + kFlashQ - 1) / kFlashQ, nh, nsp),
                        kFlashThreads, 0, stream>>>(
              aa, q, r, sink, sl.k, sl.v, bias_w, att, n, nsp, kSplitKeys,
              split_m, split_l, split_o, blist, bcount, kListCap);
        }
        if (nsp > 1)
          FlashCombine<<<dim3(n, nh), 128, 0, stream>>>(split_m, split_l,
                                                        split_o, att, nh, nsp);
      } else {
        const int split_keys =
            swa ? 256 : (n <= 8 ? kSplitKeys / 4 : kSplitKeys);
        const bool split =
            n < kWmmaMinTokens &&
            (swa ? n <= 8 : ctx > static_cast<std::uint32_t>(split_keys));
        const int nsplit =
            !split ? 1
            : swa  ? (span + 72 + split_keys - 1) / split_keys
                   : static_cast<int>((ctx + split_keys - 1) / split_keys);
        fused_gate = big && nsplit == 1;
        FlashRel<8><<<dim3((n + kFlashQ - 1) / kFlashQ, nh, nsplit),
                      kFlashThreads, 0, stream>>>(
            aa, q, r, sink, sl.k, sl.v, bias_w, att, n, nsplit, split_keys,
            split_m, split_l, split_o, nullptr, nullptr, 0,
            fused_gate ? gate : nullptr, fused_gate ? atth : nullptr);
        if (nsplit > 1)
          FlashCombine<<<dim3(n, nh), 128, 0, stream>>>(
              split_m, split_l, split_o, att, nh, nsplit);
      }
    } else {
      Attention<<<dim3(nh, n), kThreads, (span + 1) * sizeof(float), stream>>>(
          aa, q, r, sink, sl.k, sl.v,
          static_cast<const float*>(m->T(m->P(p, "attn_rel_bias.weight")).dev),
          att);
    }
    if (big) {
      if (!fused_gate)
        SigmoidMulH<<<(n * nh * hd + 255) / 256, 256, 0, stream>>>(
            att, gate, atth, static_cast<std::size_t>(n) * nh * hd);
    } else {
      SigmoidMul<<<(n * nh * hd + 255) / 256, 256, 0, stream>>>(
          att, gate, static_cast<std::size_t>(n) * nh * hd);
    }
  }

  // stage 4: output projection, residual, FFN mix, MoE -> cur
  // attention output projection: queue on the NPU (INKLING_NPU_OUT=1) or
  // compute on the GPU
  void LayerOut1a(LayerCtx& L) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = L.p;
    const bool swa = L.swa;
    Slot& sl = slots[L.slot];
    const int n = L.n;
    const std::size_t nh_ = L.nh_;
    const bool big = L.big;
    float* const* inp = L.inp;
    __half* const* inh = L.inh;
    const float* msgs_ = L.msgs;
    (void)c;
    (void)h;
    (void)nh;
    (void)nkv;
    (void)hd;
    (void)nrel;
    (void)p;
    (void)swa;
    (void)sl;
    (void)n;
    (void)nh_;
    (void)big;
    (void)inp;
    (void)inh;
    (void)msgs_;
    static const bool out_on = [] {
      const char* e = std::getenv("INKLING_NPU_OUT");
      return e == nullptr || e[0] != '0';
    }();
    L.out_npu = out_on && npu_ao >= 0 && big &&
                m->npu_w.count(m->P(p, "attn_output.weight")) != 0;
    if (L.out_npu) {
      namespace rocm = gufo::models::qwen38_flash_next::rocm;
      auto& npu = *m->npu;
      const auto& wo = m->npu_w.at(m->P(p, "attn_output.weight"));
      const std::uint32_t enc =
          static_cast<std::uint32_t>((n + 127) / 128 * 128);
      rocm::NpuEncodeActivations(atth, nh * hd, enc, nh * hd, 0, nh * hd,
                                 npu.Device(npu_ao), stream);
      const int prog = (n <= 2048 && m->npu_prog_o_small >= 0)
                           ? m->npu_prog_o_small
                           : m->npu_prog_o;
      NpuSubmit(npu_ao, wo.data(), npu_c, static_cast<int>(wo.size()), prog);
      L.ticket = npu.Ticket();
    }
  }

  void LayerOut1b(LayerCtx& L, std::vector<const float*>& blocks) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = L.p;
    const bool swa = L.swa;
    Slot& sl = slots[L.slot];
    const int n = L.n;
    const std::size_t nh_ = L.nh_;
    const bool big = L.big;
    float* const* inp = L.inp;
    __half* const* inh = L.inh;
    const float* msgs_ = L.msgs;
    (void)c;
    (void)h;
    (void)nh;
    (void)nkv;
    (void)hd;
    (void)nrel;
    (void)p;
    (void)swa;
    (void)sl;
    (void)n;
    (void)nh_;
    (void)big;
    (void)inp;
    (void)inh;
    (void)msgs_;
    const std::uint32_t ip = L.ip;
    if (L.out_npu) {
      namespace rocm = gufo::models::qwen38_flash_next::rocm;
      auto& npu = *m->npu;
      NpuWaitFor(L.ticket);
      for (int i = 0; i < 3; ++i) {
        const std::uint32_t valid =
            (i == 2) ? static_cast<std::uint32_t>(h - 2 * 1024) : 0U;
        rocm::NpuDecodeOutput(npu.Staged(stream, npu_c[i]), o, n, kNpuSlice, h,
                              i * kNpuSlice, false, stream, valid);
      }
    } else {
      Dense(m->T(m->P(p, "attn_output.weight")), att, nh * hd, o, h, n, -1,
            big ? atth : nullptr);
    }
    const bool push = ip % c.attn_res_block_size == 0;
    // ffn_inp = conv(o) (+ the layer input unless it starts a new block)
    Convolve(o, m->T(m->P(p, "attn_post_conv.weight")), nullptr, 1,
             sl.hist_attn, ffn_inp, h, n, push ? nullptr : cur);
    if (push) {
      float* b = Pool();
      if (F16Blocks(n)) {
        ToHalf<<<(nh_ + 255) / 256, 256, 0, stream>>>(
            cur, reinterpret_cast<__half*>(b), nh_);
      } else {
        HIP_OK(
            hipMemcpyAsync(b, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
      }
      blocks.push_back(b);
    }
  }

  void LayerOut2(LayerCtx& L, std::vector<const float*>& blocks) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = L.p;
    const bool swa = L.swa;
    Slot& sl = slots[L.slot];
    const int n = L.n;
    const std::size_t nh_ = L.nh_;
    const bool big = L.big;
    float* const* inp = L.inp;
    __half* const* inh = L.inh;
    const float* msgs_ = L.msgs;
    (void)c;
    (void)h;
    (void)nh;
    (void)nkv;
    (void)hd;
    (void)nrel;
    (void)p;
    (void)swa;
    (void)sl;
    (void)n;
    (void)nh_;
    (void)big;
    (void)inp;
    (void)inh;
    (void)msgs_;
    const float* ffn_w =
        static_cast<const float*>(m->T(m->P(p, "ffn_norm.weight")).dev);
    const float* ffn_q =
        static_cast<const float*>(m->T(m->P(p, "ffn_res_query.weight")).dev);
    if (big && msgs_ == nullptr) {
      NormPtrs np{};
      np.w[0] = ffn_w;
      np.out[0] = x;
      np.outh[0] = xhx;
      MixRmsLaunch(blocks, ffn_inp, ffn_q, 1, n, np);
    } else {
      Mixer(blocks, ffn_inp, ffn_q, 1, n, x);
      if (msgs_ != nullptr)
        ControlStep(x, msgs_ + 3 * nh_, p, "mlp", n);
      RmsRows(x, ffn_w, x, n, h, h, h, big ? xhx : nullptr);
    }
    if (pf_on && L.vi + 1 < static_cast<int>(visit_slots.size()))
      Prefetch({&m->T(m->P(m->Phys(visit_slots[L.vi + 1]), "attn_q.weight"))},
               n);
    Moe(p, n, big ? xhx : nullptr);
    Convolve(moe, m->T(m->P(p, "ffn_post_conv.weight")), nullptr, 1,
             sl.hist_ffn, cur, h, n, ffn_inp);
  }

  // returns layer output in `cur`
  void Layer(std::uint32_t slot, std::uint32_t ip,
             std::vector<const float*>& blocks, const float* msgs_, int n) {
    LayerCtx L;
    BeginLayer(L, slot, ip, msgs_, n);
    L.vi = visit_i++;
    if (pf_on) {
      // q was read during the previous visit's expert input; the first visit
      // reads it now, the gate projection beside the attention inputs
      const std::uint32_t pp = L.p;
      Prefetch(
          {&m->T(m->P(pp, L.vi == 0 ? "attn_q.weight" : "attn_gate.weight"))},
          n);
    }
    LayerFront(L, blocks);
    LayerQkvBegin(L);
    LayerQkvEnd(L);
    if (pf_on)
      Prefetch({&m->T(m->P(L.p, "attn_output.weight")),
                &m->T(m->P(L.p, "ffn_gate_inp.weight"))},
               n);
    LayerAttn(L);
    LayerOut1a(L);
    LayerOut1b(L, blocks);
    LayerOut2(L, blocks);
  }

  // Reads the GPTQ artifact of layer p (INKLING_GPTQ_DIR): which 0 = gate/up,
  // 1 = down. Returns false when there is none.
  bool LoadGptq(std::uint32_t p, int which, void* tiles, std::size_t tb,
                void* scales, std::size_t sb) {
    const char* dir = std::getenv("INKLING_GPTQ_DIR");
    if (dir == nullptr)
      return false;
    static const int parts = std::getenv("INKLING_GPTQ_PARTS") != nullptr
                                 ? std::atoi(std::getenv("INKLING_GPTQ_PARTS"))
                                 : 3;
    if (((parts >> which) & 1) == 0)
      return false;
    const Config& c = m->config;
    const int h = c.hidden_size, ne = c.expert_count, ff = c.expert_ff;
    const int rows2 = 2 * ff;
    const int kd = (ff + 255) / 256 * 256;
    std::ifstream in(std::string(dir) + "/layer_" + std::to_string(p) + ".gptq",
                     std::ios::binary);
    if (!in)
      return false;
    std::uint64_t hdr[8];
    in.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    namespace iu4 = gufo::models::inkling::iu4;
    if (!in || hdr[0] != 0x31504b4e49515447ULL ||
        hdr[1] != static_cast<std::uint64_t>(ne) ||
        hdr[2] != static_cast<std::uint64_t>(rows2) ||
        hdr[3] != static_cast<std::uint64_t>(h) ||
        hdr[4] != static_cast<std::uint64_t>(kd)) {
      std::fprintf(
          stderr,
          "inkling: gptq artifact for layer %u does not match the model\n", p);
      return false;
    }
    const std::size_t tb_gu =
        static_cast<std::size_t>(ne) * iu4::WeightBytesPerExpert(rows2, h);
    const std::size_t sb_gu =
        static_cast<std::size_t>(ne) * iu4::ScaleBytesPerExpert(rows2, h);
    std::size_t off = sizeof hdr;
    if (which == 1)
      off += tb_gu + sb_gu;
    in.seekg(static_cast<std::streamoff>(off));
    const auto fill = [&](void* dev, std::size_t bytes) {
      constexpr std::size_t kChunk = std::size_t{64} << 20;
      std::vector<char> buf(std::min(bytes, kChunk));
      for (std::size_t o = 0; o < bytes; o += kChunk) {
        const std::size_t n = std::min(kChunk, bytes - o);
        in.read(buf.data(), static_cast<std::streamsize>(n));
        HIP_OK(hipMemcpy(static_cast<char*>(dev) + o, buf.data(), n,
                         hipMemcpyHostToDevice));
      }
    };
    fill(tiles, tb);
    fill(scales, sb);
    return static_cast<bool>(in);
  }

  struct GptqLayer {
    float* gu{nullptr};
    float* dn{nullptr};
    std::vector<std::uint64_t> cnt;
  };
  struct GptqState {
    rocblas_handle blas{nullptr};
    float* signs{nullptr};
    float *xr_gu{nullptr}, *xr_dn{nullptr};
    std::map<std::uint32_t, GptqLayer> layers;
  } gq;
  bool GptqWants(std::uint32_t p) const {
    static const std::uint64_t mask = [] {
      std::uint64_t v = 0;
      if (const char* e = std::getenv("INKLING_GPTQ_CAPTURE")) {
        const std::string str = e;
        std::size_t pos = 0;
        while (pos < str.size()) {
          const std::size_t end = str.find(',', pos);
          const int x = std::atoi(str.substr(pos, end - pos).c_str());
          if (x >= 0 && x < 64)
            v |= 1ULL << x;
          if (end == std::string::npos)
            break;
          pos = end + 1;
        }
      }
      return v;
    }();
    return ((mask >> p) & 1ULL) != 0;
  }
  void GptqCapture(std::uint32_t p, const __half* xmoe, const __half* act16) {
    namespace gp = gufo::models::inkling::gptq;
    const Config& c = m->config;
    const int h = c.hidden_size, ne = c.expert_count, nu = c.expert_used,
              ff = c.expert_ff;
    const int kd = (ff + 255) / 256 * 256;
    if (gq.blas == nullptr) {
      rocblas_create_handle(&gq.blas);
      rocblas_set_stream(gq.blas, stream);
      float sg[512];
      gufo::models::inkling::iu4::FillSigns(sg);
      HIP_OK(hipMalloc(&gq.signs, sizeof(sg)));
      HIP_OK(hipMemcpy(gq.signs, sg, sizeof(sg), hipMemcpyHostToDevice));
      const std::size_t rows_max =
          static_cast<std::size_t>(kMaxBatch) * nu + ne * 16;
      HIP_OK(hipMalloc(&gq.xr_gu, rows_max * h * 4));
      HIP_OK(hipMalloc(&gq.xr_dn, rows_max * kd * 4));
    }
    rocblas_set_stream(gq.blas, stream);
    GptqLayer& L = gq.layers[p];
    if (L.gu == nullptr) {
      HIP_OK(hipMalloc(&L.gu, static_cast<std::size_t>(ne) * h * h * 4));
      HIP_OK(hipMemset(L.gu, 0, static_cast<std::size_t>(ne) * h * h * 4));
      HIP_OK(hipMalloc(&L.dn, static_cast<std::size_t>(ne) * kd * kd * 4));
      HIP_OK(hipMemset(L.dn, 0, static_cast<std::size_t>(ne) * kd * kd * 4));
      L.cnt.assign(ne, 0);
    }
    HIP_OK(hipStreamSynchronize(stream));
    std::vector<int> rb(ne + 1);
    std::vector<unsigned int> rc(ne);
    HIP_OK(hipMemcpy(rb.data(), rbounds, rb.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(rc.data(), rcounts, rc.size() * 4, hipMemcpyDeviceToHost));
    const int total = rb[ne];
    gp::RotateRows(xmoe, rrows_token, rrows_slot, total, h, h, gq.xr_gu,
                   gq.signs, stream);
    gp::RotateRows(act16, rrows_slot, rrows_slot, total, ff, kd, gq.xr_dn,
                   gq.signs, stream);
    for (int e = 0; e < ne; ++e) {
      const int rows = rb[e + 1] - rb[e];
      if (rows <= 0)
        continue;
      gp::AccumulateHessian(
          gq.blas, gq.xr_gu + static_cast<std::size_t>(rb[e]) * h, rows, h,
          L.gu + static_cast<std::size_t>(e) * h * h);
      gp::AccumulateHessian(
          gq.blas, gq.xr_dn + static_cast<std::size_t>(rb[e]) * kd, rows, kd,
          L.dn + static_cast<std::size_t>(e) * kd * kd);
      L.cnt[e] += rc[e];
    }
    HIP_OK(hipStreamSynchronize(stream));
  }
  bool GptqFinish(const std::string& dir, std::string* error) {
    namespace gp = gufo::models::inkling::gptq;
    namespace iu4 = gufo::models::inkling::iu4;
    const Config& c = m->config;
    const int h = c.hidden_size, ne = c.expert_count, ff = c.expert_ff;
    const int rows2 = 2 * ff;
    const int kd = (ff + 255) / 256 * 256;
    if (gq.blas == nullptr) {
      if (error != nullptr)
        *error =
            "nothing captured (set INKLING_GPTQ_CAPTURE and run a prefill)";
      return false;
    }
    rocblas_set_stream(gq.blas, stream);
    gp::Scratch sgu, sdn;
    float *wgu_rot{nullptr}, *wdn_rot{nullptr};
    if (!gp::AllocScratch(sgu, rows2, h) || !gp::AllocScratch(sdn, h, kd)) {
      if (error != nullptr)
        *error = "gptq scratch allocation failed";
      return false;
    }
    HIP_OK(hipMalloc(&wgu_rot, static_cast<std::size_t>(rows2) * h * 4));
    HIP_OK(hipMalloc(&wdn_rot, static_cast<std::size_t>(h) * kd * 4));
    const std::size_t tb_gu = iu4::WeightBytesPerExpert(rows2, h);
    const std::size_t sb_gu = iu4::ScaleBytesPerExpert(rows2, h);
    const std::size_t tb_dn = iu4::WeightBytesPerExpert(h, kd);
    const std::size_t sb_dn = iu4::ScaleBytesPerExpert(h, kd);
    std::uint8_t *t_gu, *t_dn;
    std::uint16_t *s_gu, *s_dn;
    HIP_OK(hipMalloc(&t_gu, tb_gu * ne));
    HIP_OK(hipMalloc(&s_gu, sb_gu * ne));
    HIP_OK(hipMalloc(&t_dn, tb_dn * ne));
    HIP_OK(hipMalloc(&s_dn, sb_dn * ne));
    for (auto& [p, L] : gq.layers) {
      const Tensor& wgu = m->T(m->P(p, "ffn_gate_up_exps.weight"));
      const Tensor& wdn = m->T(m->P(p, "ffn_down_exps.weight"));
      int rtn_gu = 0, rtn_dn = 0;
      for (int e = 0; e < ne; ++e) {
        const float damp_gu =
            L.cnt[e] < static_cast<std::uint64_t>(h) ? 0.2F : 0.05F;
        const float damp_dn =
            L.cnt[e] < static_cast<std::uint64_t>(kd) ? 0.2F : 0.05F;
        gp::RotateWeights(
            static_cast<const std::uint8_t*>(wgu.dev) +
                static_cast<std::size_t>(e) * rows2 * wgu.row_bytes,
            wgu.row_bytes, wgu_rot, rows2, h, h, gq.signs, stream);
        if (L.cnt[e] < 64 ||
            !gp::Quantize(gq.blas, sgu, wgu_rot,
                          L.gu + static_cast<std::size_t>(e) * h * h, h,
                          damp_gu, stream)) {
          gp::QuantizeRtn(sgu, wgu_rot, stream);
          ++rtn_gu;
        }
        gp::Pack(sgu, t_gu + e * tb_gu, s_gu + e * (sb_gu / 2), stream);
        gp::RotateWeights(static_cast<const std::uint8_t*>(wdn.dev) +
                              static_cast<std::size_t>(e) * h * wdn.row_bytes,
                          wdn.row_bytes, wdn_rot, h, ff, kd, gq.signs, stream);
        if (L.cnt[e] < 64 ||
            !gp::Quantize(gq.blas, sdn, wdn_rot,
                          L.dn + static_cast<std::size_t>(e) * kd * kd, ff,
                          damp_dn, stream)) {
          gp::QuantizeRtn(sdn, wdn_rot, stream);
          ++rtn_dn;
        }
        gp::Pack(sdn, t_dn + e * tb_dn, s_dn + e * (sb_dn / 2), stream);
        HIP_OK(hipStreamSynchronize(stream));
      }
      std::ofstream out(dir + "/layer_" + std::to_string(p) + ".gptq",
                        std::ios::binary);
      const std::uint64_t hdr[8] = {0x31504b4e49515447ULL,
                                    static_cast<std::uint64_t>(ne),
                                    static_cast<std::uint64_t>(rows2),
                                    static_cast<std::uint64_t>(h),
                                    static_cast<std::uint64_t>(kd),
                                    0,
                                    0,
                                    0};
      out.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
      const auto dump = [&](const void* dev, std::size_t bytes) {
        std::vector<char> host(bytes);
        HIP_OK(hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost));
        out.write(host.data(), static_cast<std::streamsize>(bytes));
      };
      dump(t_gu, tb_gu * ne);
      dump(s_gu, sb_gu * ne);
      dump(t_dn, tb_dn * ne);
      dump(s_dn, sb_dn * ne);
      std::fprintf(
          stderr,
          "inkling: gptq layer %u written (rtn fallback gate/up %d, down %d)\n",
          p, rtn_gu, rtn_dn);
    }
    return true;
  }

  void Moe(std::uint32_t p, int n, const __half* xhx_in = nullptr) {
    const Config& c = m->config;
    const int h = c.hidden_size, ne = c.expert_count, nu = c.expert_used,
              ff = c.expert_ff;
    Dense(m->T(m->P(p, "ffn_gate_inp.weight")), x, h, router, ne, n, -1,
          xhx_in);
    static const float prune_tau =
        std::getenv("INKLING_PRUNE") != nullptr
            ? static_cast<float>(std::atof(std::getenv("INKLING_PRUNE")))
            : 0.0F;
    static const float prune_prefill_cfg =
        std::getenv("INKLING_PRUNE_PREFILL") != nullptr
            ? static_cast<float>(
                  std::atof(std::getenv("INKLING_PRUNE_PREFILL")))
            : 0.0F;
    const bool compact_path =
        n >= kWmmaMinTokens &&
        m->T(m->P(p, "ffn_gate_up_exps.weight")).type == kTypeNvfp4 &&
        std::getenv("INKLING_OWN_EXPERTS") == nullptr;
    const float prune_prefill = compact_path ? prune_prefill_cfg : 0.0F;
    Route<<<n, 256, 0, stream>>>(
        router,
        static_cast<const float*>(m->T(m->P(p, "exp_probs_b.bias")).dev), ne,
        nu, c.expert_weights_scale, ids, wts,
        n < kWmmaMinTokens ? prune_tau : prune_prefill);
    ids_flat = ids;
    const float* s_gu = m->Has(m->P(p, "ffn_gate_up_exps.scale"))
                            ? static_cast<const float*>(
                                  m->T(m->P(p, "ffn_gate_up_exps.scale")).dev)
                            : nullptr;
    const float* s_dn = m->Has(m->P(p, "ffn_down_exps.scale"))
                            ? static_cast<const float*>(
                                  m->T(m->P(p, "ffn_down_exps.scale")).dev)
                            : nullptr;
    const int items = n * nu;
    if (n >= kWmmaMinTokens &&
        m->T(m->P(p, "ffn_gate_up_exps.weight")).type == kTypeNvfp4 &&
        std::getenv("INKLING_OWN_EXPERTS") == nullptr) {
      namespace qfn = gufo::models::qwen38_flash_next::rocm;
      const Tensor& wgu = m->T(m->P(p, "ffn_gate_up_exps.weight"));
      const Tensor& wdn = m->T(m->P(p, "ffn_down_exps.weight"));
      HIP_OK(hipMemsetAsync(rcounts, 0, ne * sizeof(unsigned int), stream));
      CountExperts<<<(items + 255) / 256, 256, 0, stream>>>(ids, items,
                                                            rcounts);
      qfn::RoutedCompact(ids, rcounts, rbounds, rcursors, rrows_token,
                         rrows_slot, n, nu, ne, stream);
      const __half* xmoe = xhx_in;
      if (xmoe == nullptr) {
        ToHalf<<<(static_cast<std::size_t>(n) * h + 255) / 256, 256, 0,
                 stream>>>(x, xh, static_cast<std::size_t>(n) * h);
        xmoe = xh;
      }
      static const char* const dump_layer = std::getenv("INKLING_DUMP_MOE");
      static bool dumped = false;
      static bool dumped_out = false;
      if (dump_layer != nullptr && !dumped &&
          p == static_cast<std::uint32_t>(std::atoi(dump_layer)) && n >= 2048) {
        dumped = true;
        HIP_OK(hipStreamSynchronize(stream));
        std::vector<int> hid(static_cast<std::size_t>(items));
        HIP_OK(
            hipMemcpy(hid.data(), ids, hid.size() * 4, hipMemcpyDeviceToHost));
        std::vector<int> cnt(ne, 0);
        for (int v : hid)
          if (v >= 0)
            ++cnt[v];
        std::vector<int> order(ne);
        for (int i = 0; i < ne; ++i)
          order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return cnt[a] > cnt[b]; });
        const int pick = 4;
        std::vector<__half> hx(static_cast<std::size_t>(n) * h);
        HIP_OK(
            hipMemcpy(hx.data(), xmoe, hx.size() * 2, hipMemcpyDeviceToHost));
        FILE* fo = std::fopen("/tmp/inkling_moe_dump.bin", "wb");
        const std::int32_t hdr[6] = {n, h, nu, ne, ff, pick};
        std::fwrite(hdr, 4, 6, fo);
        std::fwrite(hid.data(), 4, hid.size(), fo);
        std::fwrite(hx.data(), 2, hx.size(), fo);
        const std::size_t gu_bytes =
            static_cast<std::size_t>(2 * ff) * wgu.row_bytes;
        const std::size_t dn_bytes =
            static_cast<std::size_t>(h) * wdn.row_bytes;
        std::vector<std::uint8_t> buf(std::max(gu_bytes, dn_bytes));
        for (int i = 0; i < pick; ++i) {
          const std::int32_t e = order[i];
          std::fwrite(&e, 4, 1, fo);
          HIP_OK(hipMemcpy(buf.data(),
                           static_cast<const std::uint8_t*>(wgu.dev) +
                               static_cast<std::size_t>(e) * gu_bytes,
                           gu_bytes, hipMemcpyDeviceToHost));
          std::fwrite(buf.data(), 1, gu_bytes, fo);
        }
        std::fclose(fo);
        std::fprintf(stderr, "inkling: dumped moe layer %u (row_bytes %zu)\n",
                     p, static_cast<std::size_t>(wgu.row_bytes));
      }
      static const int bn_env = std::getenv("INKLING_BN") != nullptr
                                    ? std::atoi(std::getenv("INKLING_BN"))
                                    : 0;
      const int bn =
          bn_env > 0 ? bn_env : (n >= 2048 ? 192 : (n >= 512 ? 128 : 48));
      int ntiles = 0;
      static const bool tile_list =
          std::getenv("INKLING_TILE_LIST") == nullptr ||
          std::getenv("INKLING_TILE_LIST")[0] != '0';
      const int* tmap =
          tile_list && ne <= 256 ? nullptr : TileMap(n, bn, &ntiles);
      if (tile_list && ne <= 256) {
        const int max_tiles = (items + bn - 1) / bn + ne;
        BuildTileList<<<1, 256, 0, stream>>>(rbounds, ne, bn, tl_a, max_tiles);
        tmap = tl_a;
        ntiles = max_tiles;
      }
      if (situ_params == nullptr) {
        const float host[2] = {c.situ_beta, c.situ_linear_beta};
        situ_params = static_cast<float*>(AllocLazy(sizeof(host)));
        HIP_OK(
            hipMemcpy(situ_params, host, sizeof(host), hipMemcpyHostToDevice));
      }
      static const bool fuse_situ =
          std::getenv("INKLING_SITU_FUSED") == nullptr ||
          std::getenv("INKLING_SITU_FUSED")[0] != '0';
      bool ok1 = false;
      static const bool iu4_emu = std::getenv("INKLING_IU4_EMU") != nullptr;
      static const int emu_mode =
          iu4_emu ? std::atoi(std::getenv("INKLING_IU4_EMU")) : 0;
      static const int iu4_level = std::getenv("INKLING_IU4") != nullptr
                                       ? std::atoi(std::getenv("INKLING_IU4"))
                                       : 0;
      // INKLING_IU4_SKIP=<physical layers, comma separated> keeps those layers
      // on the NVFP4 experts
      static const std::uint64_t iu4_skip = [] {
        std::uint64_t mask = 0;
        if (const char* e = std::getenv("INKLING_IU4_SKIP")) {
          const std::string str = e;
          std::size_t pos = 0;
          while (pos < str.size()) {
            const std::size_t end = str.find(',', pos);
            const int v = std::atoi(str.substr(pos, end - pos).c_str());
            if (v >= 0 && v < 64)
              mask |= 1ULL << v;
            if (end == std::string::npos)
              break;
            pos = end + 1;
          }
        }
        return mask;
      }();
      const bool iu4_on = iu4_level >= 1 && ((iu4_skip >> p) & 1ULL) == 0;
      bool iu4_down_done = false;
      if (iu4_on) {
        namespace iu4 = gufo::models::inkling::iu4;
        const int rows2 = 2 * ff;
        const int kd = (ff + 255) / 256 * 256;
        {
          // the repacked weights are shared by every session of the model
          std::lock_guard<std::mutex> lock(m->iu4_mutex);
          if (m->iu4_signs == nullptr) {
            float sg[512];
            iu4::FillSigns(sg);
            m->iu4_signs = static_cast<float*>(m->ModelAlloc(sizeof(sg)));
            HIP_OK(
                hipMemcpy(m->iu4_signs, sg, sizeof(sg), hipMemcpyHostToDevice));
          }
          if (m->iu4_w[p].first == nullptr) {
            auto* tiles = static_cast<std::uint8_t*>(
                m->ModelAlloc(static_cast<std::size_t>(ne) *
                              iu4::WeightBytesPerExpert(rows2, h)));
            auto* scales = static_cast<std::uint16_t*>(
                m->ModelAlloc(static_cast<std::size_t>(ne) *
                              iu4::ScaleBytesPerExpert(rows2, h)));
            const std::size_t tb = static_cast<std::size_t>(ne) *
                                   iu4::WeightBytesPerExpert(rows2, h);
            const std::size_t sb = static_cast<std::size_t>(ne) *
                                   iu4::ScaleBytesPerExpert(rows2, h);
            if (!LoadGptq(p, 0, tiles, tb, scales, sb))
              iu4::RepackExperts(static_cast<const std::uint8_t*>(wgu.dev),
                                 wgu.row_bytes, tiles, scales, ne, rows2, h, h,
                                 m->iu4_signs, stream);
            HIP_OK(hipStreamSynchronize(stream));
            m->iu4_w[p] = {tiles, scales};
          }
          if (iu4_level >= 2 && m->iu4_dw[p].first == nullptr) {
            auto* tiles = static_cast<std::uint8_t*>(
                m->ModelAlloc(static_cast<std::size_t>(ne) *
                              iu4::WeightBytesPerExpert(h, kd)));
            auto* scales = static_cast<std::uint16_t*>(
                m->ModelAlloc(static_cast<std::size_t>(ne) *
                              iu4::ScaleBytesPerExpert(h, kd)));
            const std::size_t tb =
                static_cast<std::size_t>(ne) * iu4::WeightBytesPerExpert(h, kd);
            const std::size_t sb =
                static_cast<std::size_t>(ne) * iu4::ScaleBytesPerExpert(h, kd);
            if (!LoadGptq(p, 1, tiles, tb, scales, sb))
              iu4::RepackExperts(static_cast<const std::uint8_t*>(wdn.dev),
                                 wdn.row_bytes, tiles, scales, ne, h, ff, kd,
                                 m->iu4_signs, stream);
            HIP_OK(hipStreamSynchronize(stream));
            m->iu4_dw[p] = {tiles, scales};
          }
        }
        iu4_signs = m->iu4_signs;
        if (iu4_act == nullptr) {
          const int max_rows = kMaxBatch * nu + ne * 16;
          iu4_padded = (max_rows + 128 + 127) / 128 * 128;
          iu4_act = static_cast<std::uint8_t*>(AllocLazy(
              static_cast<std::size_t>(iu4_padded) / 16 * (h / 128) * 1024));
          iu4_act_scales = static_cast<float*>(
              AllocLazy(static_cast<std::size_t>(h / 128) * iu4_padded * 4));
          iu4_tok_padded = (kMaxBatch + 127) / 128 * 128;
          iu4_tok = static_cast<std::uint8_t*>(
              AllocLazy(static_cast<std::size_t>(iu4_tok_padded) / 16 *
                        (h / 128) * 1024));
          iu4_tok_scales = static_cast<float*>(AllocLazy(
              static_cast<std::size_t>(h / 128) * iu4_tok_padded * 4));
          iu4_act2 = static_cast<std::uint8_t*>(AllocLazy(
              static_cast<std::size_t>(iu4_padded) / 16 * (kd / 128) * 1024));
          iu4_act2_scales = static_cast<float*>(
              AllocLazy(static_cast<std::size_t>(kd / 128) * iu4_padded * 4));
        }
        int nt128 = 0;
        const int* tmap128 = tile_list && ne <= 256
                                 ? nullptr
                                 : TileMap(n, iu4::kTokenTile, &nt128);
        if (tile_list && ne <= 256) {
          nt128 = (items + iu4::kTokenTile - 1) / iu4::kTokenTile + ne;
          BuildTileList<<<1, 256, 0, stream>>>(rbounds, ne, iu4::kTokenTile,
                                               tl_b, nt128);
          tmap128 = tl_b;
        }
        static const bool tok_once =
            std::getenv("INKLING_IU4_TOKONCE") == nullptr ||
            std::getenv("INKLING_IU4_TOKONCE")[0] != '0';
        if (tok_once)
          iu4::QuantizeRoutedActsOnce(
              xmoe, n, h, h, iu4_tok, iu4_tok_scales, iu4_tok_padded,
              rrows_token, rrows_slot, rbounds, ne, items + ne * 16, iu4_act,
              iu4_act_scales, iu4_padded, iu4_signs, stream);
        else
          iu4::QuantizeRoutedActs(
              xmoe, rrows_token, rrows_slot, rbounds, ne, items + ne * 16, h, h,
              iu4_act, iu4_act_scales, iu4_padded, iu4_signs, stream);
        iu4::GateUpSitu(m->iu4_w[p].first, m->iu4_w[p].second, iu4_act,
                        iu4_act_scales, iu4_padded, tmap128, nt128, rbounds,
                        rrows_slot, act16, ff, h, s_gu, situ_params, stream);
        ok1 = true;
        if (iu4_level >= 2) {
          // down projection on int4 too: K = ff is padded to 1280
          iu4::QuantizeRoutedActs(
              act16, rrows_slot, rrows_slot, rbounds, ne, items + ne * 16, ff,
              kd, iu4_act2, iu4_act2_scales, iu4_padded, iu4_signs, stream);
          iu4::DownGemm(m->iu4_dw[p].first, m->iu4_dw[p].second, iu4_act2,
                        iu4_act2_scales, iu4_padded, tmap128, nt128, rbounds,
                        rrows_slot, y16, h, kd, s_dn, stream);
          iu4_down_done = true;
        }
        static bool iu4_checked = false;
        if (!iu4_checked && std::getenv("INKLING_IU4_CHECK") != nullptr &&
            p == 10 && n >= 2048) {
          iu4_checked = true;
          HIP_OK(hipStreamSynchronize(stream));
          std::vector<__half> a(static_cast<std::size_t>(items) * ff),
              b(a.size());
          HIP_OK(
              hipMemcpy(a.data(), act16, a.size() * 2, hipMemcpyDeviceToHost));
          // reference: the Q8Rep emulation of the same numerics
          const int rows_total = ne * 2 * ff;
          const std::size_t dst_rb =
              (static_cast<std::size_t>(h) + h / 16 + 15) / 16 * 16;
          if (emu_signs == nullptr) {
            std::vector<float> sg(512);
            iu4::FillSigns(sg.data());
            emu_signs = static_cast<float*>(AllocBytes(512 * 4));
            HIP_OK(hipMemcpy(emu_signs, sg.data(), 512 * 4,
                             hipMemcpyHostToDevice));
            emu_x = static_cast<__half*>(
                AllocBytes(static_cast<std::size_t>(kMaxBatch) * h * 2));
            emu_gu = static_cast<float*>(AllocBytes(
                static_cast<std::size_t>(kMaxBatch) * nu * 2 * ff * 4));
          }
          void* ew = AllocBytes(static_cast<std::size_t>(rows_total) * dst_rb);
          EmuWeights<<<rows_total, 256, 0, stream>>>(
              static_cast<const std::uint8_t*>(wgu.dev),
              static_cast<std::uint8_t*>(ew), h, wgu.row_bytes, dst_rb,
              emu_signs, emu_signs + 256, true);
          EmuActs<<<n, 256, 0, stream>>>(xmoe, emu_x, h, emu_signs,
                                         emu_signs + 256, true, true);
          qfn::RoutedQ8RepGemm(ew, emu_x, tmap, ntiles, bn, rbounds,
                               rrows_token, rrows_slot, emu_gu, 2 * ff, h,
                               stream);
          EmuSitu<<<items, 256, 0, stream>>>(emu_gu, ids, s_gu, act16, ff,
                                             c.situ_beta, c.situ_linear_beta);
          HIP_OK(hipStreamSynchronize(stream));
          HIP_OK(
              hipMemcpy(b.data(), act16, b.size() * 2, hipMemcpyDeviceToHost));
          double num = 0, den = 0, maxd = 0;
          std::size_t bad = 0;
          for (std::size_t i = 0; i < a.size(); ++i) {
            const double x = __half2float(a[i]), y = __half2float(b[i]);
            num += (x - y) * (x - y);
            den += y * y;
            maxd = std::max(maxd, std::fabs(x - y));
            if (std::fabs(x - y) > 0.05 * std::fabs(y) + 1e-3)
              ++bad;
          }
          std::fprintf(stderr,
                       "inkling: iu4 check layer 10: rel rms diff %.4f, max "
                       "|d| %.4f, elements off by >5%%: %zu of %zu\n",
                       std::sqrt(num / den), maxd, bad, a.size());
        }
      } else if (iu4_emu) {
        const int rows_total = ne * 2 * ff;
        const std::size_t dst_rb =
            (static_cast<std::size_t>(h) + h / 16 + 15) / 16 * 16;
        if (emu_signs == nullptr) {
          std::vector<float> sg(512);
          unsigned st = 42;
          for (int i = 0; i < 256; ++i) {
            st = (st * 1103515245U + 12345U) & 0x7fffffffU;
            sg[i] = ((st >> 16) & 1U) ? -1.0F : 1.0F;
          }
          st = 1042;
          for (int i = 0; i < 256; ++i) {
            st = (st * 1103515245U + 12345U) & 0x7fffffffU;
            sg[256 + i] = ((st >> 16) & 1U) ? -1.0F : 1.0F;
          }
          emu_signs = static_cast<float*>(AllocBytes(512 * 4));
          HIP_OK(
              hipMemcpy(emu_signs, sg.data(), 512 * 4, hipMemcpyHostToDevice));
          emu_x = static_cast<__half*>(
              AllocBytes(static_cast<std::size_t>(kMaxBatch) * h * 2));
          emu_gu = static_cast<float*>(AllocBytes(
              static_cast<std::size_t>(kMaxBatch) * nu * 2 * ff * 4));
        }
        if (emu_w[p] == nullptr) {
          emu_w[p] = AllocBytes(static_cast<std::size_t>(rows_total) * dst_rb);
          EmuWeights<<<rows_total, 256, 0, stream>>>(
              static_cast<const std::uint8_t*>(wgu.dev),
              static_cast<std::uint8_t*>(emu_w[p]), h, wgu.row_bytes, dst_rb,
              emu_signs, emu_signs + 256, (emu_mode & 1) != 0);
        }
        EmuActs<<<n, 256, 0, stream>>>(xmoe, emu_x, h, emu_signs,
                                       emu_signs + 256, (emu_mode & 2) != 0,
                                       (emu_mode & 16) != 0);
        ok1 = qfn::RoutedQ8RepGemm(emu_w[p], emu_x, tmap, ntiles, bn, rbounds,
                                   rrows_token, rrows_slot, emu_gu, 2 * ff, h,
                                   stream);
        EmuSitu<<<items, 256, 0, stream>>>(emu_gu, ids, s_gu, act16, ff,
                                           c.situ_beta, c.situ_linear_beta);
      } else if (fuse_situ && bn != 64) {
        ok1 = qfn::RoutedNvfp4SituGemmF16(
            wgu.dev, xmoe, tmap, ntiles, bn, rbounds, rrows_token, rrows_slot,
            act16, ff, h, s_gu, situ_params, stream);
      } else {
        ok1 = qfn::RoutedNvfp4GemmF16(wgu.dev, xmoe, tmap, ntiles, bn, rbounds,
                                      rrows_token, rrows_slot, gu16, 2 * ff, h,
                                      s_gu, stream);
        if (dumped && dump_layer != nullptr && !dumped_out &&
            p == static_cast<std::uint32_t>(std::atoi(dump_layer))) {
          dumped_out = true;
          HIP_OK(hipStreamSynchronize(stream));
          std::vector<__half> hy(static_cast<std::size_t>(items) * 2 * ff);
          HIP_OK(
              hipMemcpy(hy.data(), gu16, hy.size() * 2, hipMemcpyDeviceToHost));
          std::vector<float> hs(ne, 1.0F);
          if (s_gu != nullptr)
            HIP_OK(hipMemcpy(hs.data(), s_gu, ne * 4, hipMemcpyDeviceToHost));
          FILE* fo = std::fopen("/tmp/inkling_moe_out.bin", "wb");
          std::fwrite(hs.data(), 4, hs.size(), fo);
          std::fwrite(hy.data(), 2, hy.size(), fo);
          std::fclose(fo);
        }
        SituHalfIn<<<items, 256, 0, stream>>>(gu16, act16, ff, c.situ_beta,
                                              c.situ_linear_beta);
      }
      const bool ok2 = iu4_down_done ||
                       qfn::RoutedNvfp4GemmF16(wdn.dev, act16, tmap, ntiles, bn,
                                               rbounds, rrows_slot, rrows_slot,
                                               y16, h, ff, s_dn, stream);
      if (GptqWants(p))
        GptqCapture(p, xmoe, act16);
      if (!ok1 || !ok2) {
        std::fprintf(stderr,
                     "inkling: shared routed gemm rejected the launch\n");
        std::abort();
      }
      CombineHalf<<<n, 256, 0, stream>>>(y16, wts, moe, nu, h);
      return;
    }
    if (n >= kWmmaMinTokens) {
      const int max_tiles = items / kTileM + ne;
      SortItems<<<1, 1024, 0, stream>>>(ids, items, ne, sorted, tiles,
                                        max_tiles);
      const Tensor& wgu = m->T(m->P(p, "ffn_gate_up_exps.weight"));
      const Tensor& wdn = m->T(m->P(p, "ffn_down_exps.weight"));
      ToHalf<<<(static_cast<std::size_t>(n) * h + 255) / 256, 256, 0, stream>>>(
          x, xh, static_cast<std::size_t>(n) * h);
      WmmaArgs a1{static_cast<const unsigned char*>(wgu.dev),
                  wgu.row_bytes,
                  static_cast<int>(wgu.cols),
                  2 * ff,
                  xh,
                  nu,
                  gu,
                  nullptr,
                  tiles,
                  sorted,
                  n};
      WmmaArgs a2{static_cast<const unsigned char*>(wdn.dev),
                  wdn.row_bytes,
                  static_cast<int>(wdn.cols),
                  h,
                  act16,
                  1,
                  y,
                  s_dn,
                  tiles,
                  sorted,
                  n};
      const dim3 g1(2 * ff / kTileN, max_tiles);
      const dim3 g2(h / kTileN, max_tiles);
      if (wgu.type == kTypeNvfp4) {
        WmmaGemm<kTypeNvfp4><<<g1, kWmmaThreads, 0, stream>>>(a1);
      } else {
        WmmaGemm<kTypeQ8><<<g1, kWmmaThreads, 0, stream>>>(a1);
      }
      SituHalf<<<items, 256, 0, stream>>>(gu, ids, s_gu, act16, ff, c.situ_beta,
                                          c.situ_linear_beta);
      if (wdn.type == kTypeNvfp4) {
        WmmaGemm<kTypeNvfp4><<<g2, kWmmaThreads, 0, stream>>>(a2);
      } else {
        WmmaGemm<kTypeQ8><<<g2, kWmmaThreads, 0, stream>>>(a2);
      }
      Combine<<<dim3(n, (h + 255) / 256), 256, 0, stream>>>(y, wts, moe, nu, h,
                                                          nullptr, nullptr);
      return;
    }
    xrow = xrow_token;
    Routed(m->T(m->P(p, "ffn_gate_up_exps.weight")), 2 * ff, items, nullptr, x,
           gu);
    Situ<<<items, 256, 0, stream>>>(gu, ids, s_gu, act, ff, c.situ_beta,
                                    c.situ_linear_beta);
    xrow = xrow_item;
    Routed(m->T(m->P(p, "ffn_down_exps.weight")), h, items, s_dn, act, y);
    Combine<<<dim3(n, (h + 255) / 256), 256, 0, stream>>>(y, wts, moe, nu, h,
                                                          nullptr, nullptr);
  }

  void Forward(std::span<const std::int32_t> tokens) {
    static const bool time_forward = std::getenv("INKLING_FWD_TIME") != nullptr;
    const auto fwd_t0 = std::chrono::steady_clock::now();
    Activate(0);
    const Config& c = m->config;
    const int n = static_cast<int>(tokens.size());
    if (banked && n >= static_cast<int>(kBankMin) &&
        n <= static_cast<int>(2 * bank_rows)) {
      BodyBanks(tokens);
    } else {
      HIP_OK(hipMemcpy(d_tokens, tokens.data(), n * 4, hipMemcpyHostToDevice));
      const int pos = static_cast<int>(position);
      pos_base = position;
      HIP_OK(hipMemcpy(d_pos, &pos, 4, hipMemcpyHostToDevice));
      if (n == 1) {
        if (graph_exec == nullptr) {
          HIP_OK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
          Body(1);
          HIP_OK(hipStreamEndCapture(stream, &graph));
          HIP_OK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        }
        HIP_OK(hipGraphLaunch(graph_exec, stream));
      } else {
        Body(n);
      }
    }
    const auto fwd_t1 = std::chrono::steady_clock::now();
    static const bool alloc_log = std::getenv("INKLING_ALLOC_LOG") != nullptr;
    if (alloc_log) {
      static std::size_t last = 0;
      if (allocated_bytes != last) {
        std::fprintf(stderr, "inkling: session allocated_bytes %zu (n=%d)\n",
                     allocated_bytes, n);
        last = allocated_bytes;
      }
    }
    const int rows = all_rows ? n : 1;
    ArgMaxRow<<<rows, 256, 0, stream>>>(
        d_logits, static_cast<int>(c.vocab_unpadded), d_argmax, d_argmax);
    HIP_OK(hipMemcpyAsync(argmax_host, d_argmax, rows * 4,
                          hipMemcpyDeviceToHost, stream));
    HIP_OK(hipStreamSynchronize(stream));
    if (time_forward) {
      const auto fwd_t2 = std::chrono::steady_clock::now();
      std::fprintf(
          stderr, "forward n=%d: launch %.1f ms, wait %.1f ms\n", n,
          std::chrono::duration<double, std::milli>(fwd_t1 - fwd_t0).count(),
          std::chrono::duration<double, std::milli>(fwd_t2 - fwd_t1).count());
    }
    logits_rows = rows;
    logits_last = rows - 1;
    logits_have.assign(static_cast<std::size_t>(rows), 0);
    n_logit_rows = rows;
    position += n;
  }

  const float* DW(const std::string& n) const {
    return static_cast<const float*>(m->T("d." + n).dev);
  }
  const float* DW2(const std::string& n) const {
    return static_cast<const float*>(m->T("d2." + n).dev);
  }
  void DraftEnsure() {
    if (dready)
      return;
    const int h = static_cast<int>(m->config.hidden_size);
    constexpr int kSlots = 8;
    const auto f = [&](std::size_t n) {
      return static_cast<float*>(AllocBytes(n * 4));
    };
    for (int l = 0; l < 3; ++l) {
      dcap[l] = l == 1 ? static_cast<int>(ctx) : 1024;
      dk[l] = static_cast<__half*>(
          AllocBytes(static_cast<std::size_t>(dcap[l]) * 1024 * 2));
      dv[l] = static_cast<__half*>(
          AllocBytes(static_cast<std::size_t>(dcap[l]) * 1024 * 2));
    }
    dctx = f(static_cast<std::size_t>(kDraftWarmChunk) * h);
    dck = f(static_cast<std::size_t>(kDraftWarmChunk) * 1024);
    dcv = f(static_cast<std::size_t>(kDraftWarmChunk) * 1024);
    dx = f(kSlots * h);
    dh = f(kSlots * h);
    dtmp = f(kSlots * h);
    dq = f(kSlots * 3072);
    dkb = f(kSlots * 1024);
    dvb = f(kSlots * 1024);
    datt = f(kSlots * 3072);
    dgate = f(kSlots * 9216);
    dup = f(kSlots * 9216);
    dact = f(kSlots * 9216);
    const std::size_t parts =
        static_cast<std::size_t>(kSlots) * 24 * kDraftSplit;
    dpm = f(parts);
    dpl = f(parts);
    dpo = f(parts * 128);
    dcarry = f(h);
    dstate = f(256);
    dlatent = f(256);
    djin = f(5120);
    djout = f(768);
    deff = f(h);
    dlog = f(kDraftShortlistMax);
    dw = f(16);
    HIP_OK(hipMemcpy(dw, m->draft_fuse, 16 * 4, hipMemcpyHostToDevice));
    dconf = f(8);
    dstop = static_cast<int*>(AllocBytes(4));
    {
      const Tensor& head = m->T("output.weight");
      const int n = std::min<int>(kDraftShortlist,
                                  static_cast<int>(m->draft_rank.size()));
      kDraftShortlist = n;
      dmap = static_cast<int*>(AllocBytes(static_cast<std::size_t>(n) * 4));
      const std::size_t hc = m->config.hidden_size;
      dhead_stride = (hc / 2 + hc / 16 + 15) / 16 * 16;
      dhead = static_cast<unsigned char*>(
          AllocBytes(static_cast<std::size_t>(n) * dhead_stride));
      HIP_OK(hipMemcpy(dmap, m->draft_rank.data(),
                       static_cast<std::size_t>(n) * 4, hipMemcpyHostToDevice));
      Q8ToNvfp4<<<n, 128, 0, stream>>>(
          static_cast<const unsigned char*>(head.dev), head.row_bytes, dhead,
          dhead_stride, static_cast<int>(hc), dmap);
      HIP_OK(hipStreamSynchronize(stream));
    }
    dprev = static_cast<int*>(AllocBytes(4));
    dtok = static_cast<int*>(AllocBytes(4));
    dout = static_cast<int*>(AllocBytes(8 * 4));
    dready = true;
  }
  void DraftAttention(int layer, int P) {
    const int lo = layer == 1 ? 0 : std::max(0, P - 1024);
    const int total = P - lo + 8;
    const int nsplit = std::min(kDraftSplit, std::max(1, (total + 255) / 256));
    DraftAttnPartial<<<dim3(24, nsplit), 128, 0, stream>>>(
        dq, dkb, dvb, dk[layer], dv[layer], P, lo, dcap[layer], nsplit, dpm,
        dpl, dpo);
    DraftAttnCombine<<<dim3(8, 24), 128, 0, stream>>>(dpm, dpl, dpo, datt,
                                                      nsplit);
  }
  void DraftWarm(int P) {
    const int h = static_cast<int>(m->config.hidden_size);
    const float eps = m->config.rms_eps;
    for (int p0 = static_cast<int>(dwarmed); p0 < P; p0 += kDraftWarmChunk) {
      const int n = std::min(kDraftWarmChunk, P - p0);
      for (int l = 0; l < 3; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        DraftFuseNorm<<<n, 256, 0, stream>>>(
            feat, p0, dw + 4 * l, DW(b + "dflare_norm.weight"), dctx, h, eps);
        Dense(m->T("d." + b + "attn_k_ctx.weight"), dctx, h, dck, 1024, n);
        Dense(m->T("d." + b + "attn_v_ctx.weight"), dctx, h, dcv, 1024, n);
        DraftNormRope<<<dim3(n, 8), 128, 0, stream>>>(
            dck, DW(b + "attn_k_ctx_norm.weight"), 8, p0, eps, true);
        DraftStoreCtx<<<n, 256, 0, stream>>>(dck, dcv, dk[l], dv[l], p0,
                                             dcap[l]);
      }
    }
    dwarmed = static_cast<std::uint32_t>(P);
  }
  bool Draft(std::int32_t anchor, std::int32_t* out, int count,
             float* conf_out = nullptr, float stop_conf = 0.0F) {
    if (m->d2 && (!m->has_dspark || prefer_d2))
      return Draft2(anchor, out, count, conf_out);
    const int h = static_cast<int>(m->config.hidden_size);
    const float eps = m->config.rms_eps;
    const int P = static_cast<int>(position);
    if (feat == nullptr || !m->has_draft || P == 0)
      return false;
    DraftEnsure();
    if (dwarmed > static_cast<std::uint32_t>(P))
      dwarmed = static_cast<std::uint32_t>(P);
    DraftWarm(P);
    DraftFuseNorm<<<1, 256, 0, stream>>>(
        feat, P - 1, dw + 12, DW("dspark_norm.weight"), dcarry, h, eps);
    HIP_OK(hipMemcpyAsync(dtok, &anchor, 4, hipMemcpyHostToDevice, stream));
    HIP_OK(hipMemcpyAsync(dprev, &anchor, 4, hipMemcpyHostToDevice, stream));
    const Tensor& emb = m->T("token_embd.weight");
    EmbedQ8<<<1, 256, 0, stream>>>(static_cast<const unsigned char*>(emb.dev),
                                   emb.row_bytes, dtok, dtmp, h);
    RmsRows(dtmp, static_cast<const float*>(m->T("token_embd_norm.weight").dev),
            dtmp, 1, h, h, h);
    DraftSlotsInit<<<8, 256, 0, stream>>>(DW("mask_embd.weight"), dtmp, dx, h);
    DraftAddNorm<<<8, 256, 0, stream>>>(
        dx, nullptr, DW("blk.0.attn_norm.weight"), dh, h, eps);
    for (int l = 0; l < 3; ++l) {
      const std::string b = "d.blk." + std::to_string(l) + ".";
      Dense(m->T(b + "attn_q.weight"), dh, h, dq, 3072, 8);
      Dense(m->T(b + "attn_k.weight"), dh, h, dkb, 1024, 8);
      Dense(m->T(b + "attn_v.weight"), dh, h, dvb, 1024, 8);
      DraftNormRope<<<dim3(8, 24), 128, 0, stream>>>(
          dq, DW("blk." + std::to_string(l) + ".attn_q_norm.weight"), 24, P,
          eps, true);
      DraftNormRope<<<dim3(8, 8), 128, 0, stream>>>(
          dkb, DW("blk." + std::to_string(l) + ".attn_k_norm.weight"), 8, P,
          eps, true);
      DraftAttention(l, P);
      Dense(m->T(b + "attn_output.weight"), datt, 3072, dtmp, h, 8);
      DraftAddNorm<<<8, 256, 0, stream>>>(
          dx, dtmp, DW("blk." + std::to_string(l) + ".ffn_norm.weight"), dh, h,
          eps);
      Dense(m->T(b + "ffn_gate.weight"), dh, h, dgate, 9216, 8);
      Dense(m->T(b + "ffn_up.weight"), dh, h, dup, 9216, 8);
      DraftSwiglu<<<(8 * 9216 + 255) / 256, 256, 0, stream>>>(dgate, dup, dact,
                                                              8 * 9216);
      Dense(m->T(b + "ffn_down.weight"), dact, 9216, dtmp, h, 8);
      const float* next =
          l == 2 ? DW("output_norm.weight")
                 : DW("blk." + std::to_string(l + 1) + ".attn_norm.weight");
      DraftAddNorm<<<8, 256, 0, stream>>>(dx, dtmp, next, dh, h, eps);
    }
    const unsigned short* joint_w =
        static_cast<const unsigned short*>(m->T("d.joint_proj.weight").dev);
    const unsigned short* w1 =
        static_cast<const unsigned short*>(m->T("d.markov_w1.weight").dev);
    const unsigned short* how = static_cast<const unsigned short*>(
        m->T("d.dspark_hidden_out.weight").dev);
    const Tensor& head = m->T("output.weight");
    HIP_OK(hipMemsetAsync(dstate, 0, 256 * 4, stream));
    HIP_OK(hipMemsetAsync(dstop, 0, 4, stream));
    const int steps = std::min(count, 7);
    for (int s = 1; s <= steps; ++s) {
      const float* cur = dh + static_cast<std::size_t>(s) * h;
      DraftJointIn<<<1, 256, 0, stream>>>(dstate, w1, dprev, cur, dcarry, djin,
                                          h, 256);
      DraftBf16Gemv<<<768 / 4, 128, 0, stream>>>(joint_w, djin, djout, 5120,
                                                 768, nullptr, nullptr);
      DraftMarkovState<<<1, 256, 0, stream>>>(djout, DW("joint_proj.bias"),
                                              dstate, dlatent, 256);
      DraftBf16Gemv<<<h / 4, 128, 0, stream>>>(how, dlatent, deff, 256, h, cur,
                                               DW("dspark_hidden_out.bias"));
      QuantRows<<<dim3(1, h / 32), 32, 0, stream>>>(deff, xq, xqs, h);
      {
        const int rpb8 = kThreads * 2 / 32;
        GemmDp4<kTypeNvfp4><<<dim3((kDraftShortlist + rpb8 - 1) / rpb8, 1),
                              kThreads * 2, 0, stream>>>(
            dhead, dhead_stride, h, kDraftShortlist, nullptr, nullptr, nullptr,
            xq, xqs, dlog, stop_conf > 0.0F ? dstop : nullptr);
      }
      ArgMaxRow<<<1, 256, 0, stream>>>(
          dlog, kDraftShortlist, dprev, dout + (s - 1), dmap, dconf + (s - 1),
          stop_conf > 0.0F ? dstop : nullptr, stop_conf);
    }
    HIP_OK(hipMemcpyAsync(out, dout, steps * 4, hipMemcpyDeviceToHost, stream));
    if (conf_out != nullptr)
      HIP_OK(hipMemcpyAsync(conf_out, dconf, steps * 4, hipMemcpyDeviceToHost,
                            stream));
    HIP_OK(hipStreamSynchronize(stream));
    return true;
  }
  // ---- DFlash2 drafter
  void DraftEnsure2() {
    if (d2ready)
      return;
    const int h = static_cast<int>(m->config.hidden_size);
    const std::size_t vocab = m->config.vocab_unpadded;
    const auto f = [&](std::size_t n) {
      return static_cast<float*>(AllocBytes(n * 4));
    };
    for (int l = 0; l < kD2Layers; ++l) {
      d2k[l] = static_cast<__half*>(
          AllocBytes(static_cast<std::size_t>(kD2Cap) * 1024 * 2));
      d2v[l] = static_cast<__half*>(
          AllocBytes(static_cast<std::size_t>(kD2Cap) * 1024 * 2));
    }
    constexpr std::size_t kChunk = kDraftWarmChunk;
    d2cat = f(kChunk * 4 * h);
    d2th = f(kChunk * h);
    d2ctx = f(kChunk * h);
    d2ck = f(kChunk * 1024);
    d2cv = f(kChunk * 1024);
    d2emb = f(2 * h);
    d2x = f(8 * h);
    d2h = f(8 * h);
    d2tmp = f(8 * h);
    d2q = f(8 * 4096);
    d2kb = f(8 * 1024);
    d2vb = f(8 * 1024);
    d2att = f(8 * 4096);
    d2proj = f(8 * 576);
    d2conv = f(8 * h);
    d2gate = f(8 * 6912);
    d2up = f(8 * 6912);
    d2act = f(8 * 6912);
    const std::size_t parts = static_cast<std::size_t>(8) * 32 * kDraftSplit;
    d2pm = f(parts);
    d2pl = f(parts);
    d2po = f(parts * 128);
    d2log = f(7 * vocab);
    d2hp = f(7 * 256);
    d2uv = f(7 * 16);
    d2conf = f(8);
    d2tok = static_cast<int*>(AllocBytes(2 * 4));
    d2ids = static_cast<int*>(AllocBytes(7 * 16 * 4));
    d2out = static_cast<int*>(AllocBytes(8 * 4));
    d2ready = true;
  }
  void DraftWarm2(int P) {
    const int h = static_cast<int>(m->config.hidden_size);
    const float eps = 1.0e-6F;
    const float theta = m->d2_theta;
    const int start = std::max<int>(static_cast<int>(d2warmed),
                                    std::max(0, P - m->d2_window - 8));
    for (int p0 = start; p0 < P; p0 += kDraftWarmChunk) {
      const int n = std::min(kDraftWarmChunk, P - p0);
      const std::size_t count = static_cast<std::size_t>(n) * 4 * h;
      HalfToFloat<<<(count + 255) / 256, 256, 0, stream>>>(
          feat + static_cast<std::size_t>(p0) * 4 * h, d2cat, count);
      Dense(m->T("d2.fc.weight"), d2cat, 4 * h, d2th, h, n);
      RmsRows(d2th, DW2("enc.output_norm.weight"), d2ctx, n, h, h, h);
      for (int l = 0; l < kD2Layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        Dense(m->T("d2." + b + "attn_k.weight"), d2ctx, h, d2ck, 1024, n);
        Dense(m->T("d2." + b + "attn_v.weight"), d2ctx, h, d2cv, 1024, n);
        DraftNormRope<<<dim3(n, 8), 128, 0, stream>>>(
            d2ck, DW2(b + "attn_k_norm.weight"), 8, p0, eps, true, theta);
        DraftStoreCtx<<<n, 256, 0, stream>>>(d2ck, d2cv, d2k[l], d2v[l], p0,
                                             kD2Cap);
      }
    }
    d2warmed = static_cast<std::uint32_t>(P);
  }
  void D2Conv_(const float* in, const float* proj, const std::string& base,
               float* out, int which) {
    D2Conv<<<8, 256, 0, stream>>>(in, proj, DW2(base), out,
                                  static_cast<int>(m->config.hidden_size), 144,
                                  16, which);
  }
  bool Draft2(std::int32_t anchor, std::int32_t* out, int count,
              float* conf_out) {
    const int h = static_cast<int>(m->config.hidden_size);
    const float eps = 1.0e-6F;
    const float theta = m->d2_theta;
    const int P = static_cast<int>(position);
    const int vocab = static_cast<int>(m->config.vocab_unpadded);
    if (feat == nullptr || !m->has_draft || P == 0)
      return false;
    DraftEnsure2();
    if (d2warmed > static_cast<std::uint32_t>(P))
      d2warmed = static_cast<std::uint32_t>(P);
    DraftWarm2(P);
    const int toks[2] = {anchor, m->d2_mask};
    HIP_OK(hipMemcpyAsync(d2tok, toks, 8, hipMemcpyHostToDevice, stream));
    const Tensor& emb = m->T("token_embd.weight");
    EmbedQ8<<<2, 256, 0, stream>>>(static_cast<const unsigned char*>(emb.dev),
                                   emb.row_bytes, d2tok, d2emb, h);
    static const bool emb_norm = std::getenv("INKLING_D2_EMBNORM") == nullptr ||
                                 std::getenv("INKLING_D2_EMBNORM")[0] != '0';
    if (emb_norm)
      RmsRows(d2emb,
              static_cast<const float*>(m->T("token_embd_norm.weight").dev),
              d2emb, 2, h, h, h);
    DraftSlotsInit<<<8, 256, 0, stream>>>(d2emb + h, d2emb, d2x, h);
    DraftAddNorm<<<8, 256, 0, stream>>>(
        d2x, nullptr, DW2("blk.0.attn_norm.weight"), d2h, h, eps);
    const int lo = std::max(0, P - m->d2_window + 1);
    const int total = P - lo + 8;
    const int nsplit = std::min(kDraftSplit, std::max(1, (total + 255) / 256));
    for (int l = 0; l < kD2Layers; ++l) {
      const std::string b = "blk." + std::to_string(l) + ".";
      const std::string d = "d2." + b;
      Dense(m->T(d + "attn_conv_proj.weight"), d2h, h, d2proj, 576, 8);
      D2Conv_(d2h, d2proj, b + "attn_conv_base", d2conv, 0);
      Dense(m->T(d + "attn_q.weight"), d2conv, h, d2q, 4096, 8);
      Dense(m->T(d + "attn_k.weight"), d2conv, h, d2kb, 1024, 8);
      Dense(m->T(d + "attn_v.weight"), d2conv, h, d2vb, 1024, 8);
      DraftNormRope<<<dim3(8, 32), 128, 0, stream>>>(
          d2q, DW2(b + "attn_q_norm.weight"), 32, P, eps, true, theta);
      DraftNormRope<<<dim3(8, 8), 128, 0, stream>>>(
          d2kb, DW2(b + "attn_k_norm.weight"), 8, P, eps, true, theta);
      D2AttnPartial<<<dim3(32, nsplit), 128, 0, stream>>>(
          d2q, d2kb, d2vb, d2k[l], d2v[l], P, lo, kD2Cap, nsplit, 32, 4,
          m->d2_window, d2pm, d2pl, d2po);
      D2AttnCombine<<<dim3(8, 32), 128, 0, stream>>>(d2pm, d2pl, d2po, d2att,
                                                     nsplit, 32);
      Dense(m->T(d + "attn_output.weight"), d2att, 4096, d2tmp, h, 8);
      D2Conv_(d2tmp, d2proj, b + "attn_conv_base", d2conv, 1);
      DraftAddNorm<<<8, 256, 0, stream>>>(
          d2x, d2conv, DW2(b + "ffn_norm.weight"), d2h, h, eps);
      Dense(m->T(d + "ffn_conv_proj.weight"), d2h, h, d2proj, 576, 8);
      D2Conv_(d2h, d2proj, b + "ffn_conv_base", d2conv, 0);
      Dense(m->T(d + "ffn_gate.weight"), d2conv, h, d2gate, 6912, 8);
      Dense(m->T(d + "ffn_up.weight"), d2conv, h, d2up, 6912, 8);
      DraftSwiglu<<<(8 * 6912 + 255) / 256, 256, 0, stream>>>(d2gate, d2up,
                                                              d2act, 8 * 6912);
      Dense(m->T(d + "ffn_down.weight"), d2act, 6912, d2tmp, h, 8);
      D2Conv_(d2tmp, d2proj, b + "ffn_conv_base", d2conv, 1);
      const float* next =
          l + 1 < kD2Layers
              ? DW2("blk." + std::to_string(l + 1) + ".attn_norm.weight")
              : DW2("output_norm.weight");
      DraftAddNorm<<<8, 256, 0, stream>>>(d2x, d2conv, next, d2h, h, eps);
    }
    const float* rows = d2h + h;  // draft positions 1..7
    Dense(m->T("output.weight"), rows, h, d2log, vocab, 7, vocab);
    D2TopK<<<7, 256, 0, stream>>>(d2log, vocab, vocab, d2ids, d2uv);
    Dense(m->T("d2.selector_hidden.weight"), rows, h, d2hp, 256, 7);
    D2Select<<<1, 256, 0, stream>>>(
        d2ids, d2uv, d2hp,
        static_cast<const unsigned short*>(
            m->T("d2.selector_predecessor.weight").dev),
        static_cast<const unsigned short*>(
            m->T("d2.selector_successor.weight").dev),
        d2tok, 256, 7, m->d2_softcap, d2out, d2conf);
    const int steps = std::min(count, 7);
    HIP_OK(
        hipMemcpyAsync(out, d2out, steps * 4, hipMemcpyDeviceToHost, stream));
    if (conf_out != nullptr)
      HIP_OK(hipMemcpyAsync(conf_out, d2conf, steps * 4, hipMemcpyDeviceToHost,
                            stream));
    HIP_OK(hipStreamSynchronize(stream));
    if (conf_out != nullptr) {
      // A verified row costs about as much as half an accepted token is worth:
      // keep a slot while the chance that the whole prefix is accepted holds
      // above the threshold (reported as 1 for kept slots, 0 after the first
      // slot that fails).
      static const float theta =
          std::getenv("INKLING_D2_THETA") != nullptr
              ? static_cast<float>(std::atof(std::getenv("INKLING_D2_THETA")))
              : 0.5F;
      float cum = 1.0F;
      for (int i = 0; i < steps; ++i) {
        cum *= conf_out[i];
        conf_out[i] = cum >= theta ? 1.0F : 0.0F;
      }
    }
    return true;
  }
  void Tap(int v, int n, const float* src = nullptr) {
    if (feat == nullptr)
      return;
    for (int k = 0; k < kTapCount; ++k) {
      if (g_tap_layers[k] == v)
        TapStore<<<n, 256, 0, stream>>>(
            src != nullptr ? src : cur, feat, d_pos, k,
            static_cast<int>(m->config.hidden_size));
    }
  }

  struct BankRun {
    int n{0};
    std::uint32_t pos{0};
    std::vector<const float*> blocks, prelude_blocks, bank;
    LayerCtx L;
  };
  struct StepDesc {
    std::uint32_t il, ip;
    int pass;
  };

  std::vector<StepDesc> BuildSteps() const {
    const Config& c = m->config;
    const std::uint32_t start = c.loop_core_start, end = c.loop_core_end;
    std::vector<StepDesc> steps;
    for (std::uint32_t il = 0; il < start; ++il)
      steps.push_back({il, il, -1});
    for (std::uint32_t pass = 0; pass < c.loop_num_loops; ++pass) {
      for (std::uint32_t ip = start; ip < end; ++ip) {
        const std::uint32_t il =
            pass == 0 ? ip
                      : m->n_phys + (pass - 1) * (end - start) + (ip - start);
        steps.push_back({il, ip, static_cast<int>(pass)});
      }
    }
    for (std::uint32_t il = end; il < m->n_phys; ++il)
      steps.push_back({il, il, -1});
    return steps;
  }

  void BankFront(int b, BankRun& R, const StepDesc& st) {
    Activate(b);
    const Config& c = m->config;
    const int h = c.hidden_size;
    const std::size_t nh_ = static_cast<std::size_t>(R.n) * h;
    const std::uint32_t start = c.loop_core_start;
    if (st.pass == 0 && st.ip == start) {
      HIP_OK(hipMemcpyAsync(prelude, cur, nh_ * 4, hipMemcpyDeviceToDevice,
                            stream));
      R.prelude_blocks = R.blocks;
    }
    if (st.pass > 0 && st.ip == start) {
      R.bank.assign(R.blocks.begin() + R.prelude_blocks.size() + 1,
                    R.blocks.end());
      HIP_OK(hipMemcpyAsync(bank_out, cur, nh_ * 4, hipMemcpyDeviceToDevice,
                            stream));
      HIP_OK(hipMemcpyAsync(cur, prelude, nh_ * 4, hipMemcpyDeviceToDevice,
                            stream));
      R.blocks = R.prelude_blocks;
    }
    const float* msgs_ = nullptr;
    if (st.pass > 0) {
      const std::uint32_t cc = st.ip - start;
      const float* mq = m->mq_table + static_cast<std::size_t>(cc) * 4 * h;
      Mixer(R.bank, bank_out, mq, 4, R.n, msgs);
      msgs_ = msgs;
    }
    BeginLayer(R.L, st.il, st.ip, msgs_, R.n);
    LayerFront(R.L, R.blocks);
  }

  // Two token banks, interleaved so one bank's NPU projections overlap the
  // other bank's GPU work: A.front A.qkv-begin | B.moe | A.qkv-end A.attn
  // A.out1 | B.front B.qkv-begin | A.moe | B.qkv-end B.attn B.out1 | ...
  void DebugSync(const char* what, int t) {
    static const bool on = std::getenv("INKLING_DEBUG_SYNC") != nullptr;
    if (!on)
      return;
    const hipError_t e = hipStreamSynchronize(stream);
    if (e != hipSuccess) {
      std::fprintf(stderr, "bank debug: failure after %s at step %d: %s\n",
                   what, t, hipGetErrorString(e));
      std::abort();
    }
  }

  void BodyBanks(std::span<const std::int32_t> tokens) {
    const Config& c = m->config;
    const int h = c.hidden_size;
    const int total = static_cast<int>(tokens.size());
    const int na = std::min<int>(static_cast<int>(bank_rows),
                                 (total / 2 + 127) / 128 * 128);
    const int nb = total - na;
    BankRun R[2];
    R[0].n = na;
    R[1].n = nb;
    R[0].pos = position;
    R[1].pos = position + na;
    for (int b = 0; b < 2; ++b) {
      Activate(b);
      pos_base = R[b].pos;
      pool_used = 0;
      const int pos = static_cast<int>(R[b].pos);
      HIP_OK(hipMemcpy(d_pos, &pos, 4, hipMemcpyHostToDevice));
      HIP_OK(hipMemcpy(d_tokens, tokens.data() + (b == 0 ? 0 : na), R[b].n * 4,
                       hipMemcpyHostToDevice));
      const Tensor& emb = m->T("token_embd.weight");
      EmbedQ8<<<R[b].n, 256, 0, stream>>>(
          static_cast<const unsigned char*>(emb.dev), emb.row_bytes, d_tokens,
          cur, h);
      RmsRows(cur,
              static_cast<const float*>(m->T("token_embd_norm.weight").dev),
              cur, R[b].n, h, h, h);
    }
    const std::vector<StepDesc> steps = BuildSteps();
    const int T = static_cast<int>(steps.size());
    for (int t = 0; t < T; ++t) {
      BankFront(0, R[0], steps[t]);
      LayerQkvBegin(R[0].L);
      if (t > 0) {
        Activate(1);
        LayerOut1b(R[1].L, R[1].blocks);
        LayerOut2(R[1].L, R[1].blocks);
        Tap(t - 1, R[1].n);
        DebugSync("B.out2", t);
      }
      Activate(0);
      LayerQkvEnd(R[0].L);
      LayerAttn(R[0].L);
      LayerOut1a(R[0].L);
      DebugSync("A.attn", t);
      if (two_stream)
        HIP_OK(hipEventRecord(ev_bank, stream0));
      BankFront(1, R[1], steps[t]);
      if (two_stream)
        HIP_OK(hipStreamWaitEvent(stream1, ev_bank, 0));
      LayerQkvBegin(R[1].L);
      Activate(0);
      LayerOut1b(R[0].L, R[0].blocks);
      LayerOut2(R[0].L, R[0].blocks);
      Tap(t, R[0].n);
      DebugSync("A.out2", t);
      Activate(1);
      LayerQkvEnd(R[1].L);
      LayerAttn(R[1].L);
      LayerOut1a(R[1].L);
      DebugSync("B.attn", t);
    }
    if (two_stream)
      HIP_OK(hipEventRecord(ev_bank, stream0));
    Activate(1);
    if (two_stream)
      HIP_OK(hipStreamWaitEvent(stream1, ev_bank, 0));
    LayerOut1b(R[1].L, R[1].blocks);
    LayerOut2(R[1].L, R[1].blocks);
    Tap(T - 1, R[1].n);
    const int n = R[1].n;
    Mixer(R[1].blocks, cur,
          static_cast<const float*>(m->T("output_res_query.weight").dev), 1, n,
          x);
    RmsRows(x, static_cast<const float*>(m->T("output_norm.weight").dev), x, n,
            h, h, h);
    Tap(48, n, x);
    const Tensor& out = m->T("output.weight");
    Dense(out, x + static_cast<std::size_t>(n - 1) * h, h, d_logits,
          c.vocab_unpadded, 1, static_cast<int>(c.vocab_unpadded));
    Softcap<<<(c.vocab_unpadded + 255) / 256, 256, 0, stream>>>(
        d_logits, c.vocab_unpadded, c.final_logit_softcapping);
  }

  void Body(int n) {
    const Config& c = m->config;
    const int h = c.hidden_size;
    const std::size_t nh_ = static_cast<std::size_t>(n) * h;
    const Tensor& emb = m->T("token_embd.weight");
    EmbedQ8<<<n, 256, 0, stream>>>(static_cast<const unsigned char*>(emb.dev),
                                   emb.row_bytes, d_tokens, cur, h);
    RmsRows(cur, static_cast<const float*>(m->T("token_embd_norm.weight").dev),
            cur, n, h, h, h);
    pool_used = 0;
    int v = 0;
    BuildVisits();
    std::vector<const float*> blocks;
    const std::uint32_t start = c.loop_core_start, end = c.loop_core_end;
    for (std::uint32_t il = 0; il < start; ++il) {
      Layer(il, il, blocks, nullptr, n);
      Tap(v++, n);
    }
    HIP_OK(
        hipMemcpyAsync(prelude, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
    const std::vector<const float*> prelude_blocks = blocks;
    for (std::uint32_t pass = 0; pass < c.loop_num_loops; ++pass) {
      std::vector<const float*> bank;
      if (pass > 0) {
        bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
        HIP_OK(hipMemcpyAsync(bank_out, cur, nh_ * 4, hipMemcpyDeviceToDevice,
                              stream));
        HIP_OK(hipMemcpyAsync(cur, prelude, nh_ * 4, hipMemcpyDeviceToDevice,
                              stream));
        blocks = prelude_blocks;
      }
      for (std::uint32_t ip = start; ip < end; ++ip) {
        const std::uint32_t il =
            pass == 0 ? ip
                      : m->n_phys + (pass - 1) * (end - start) + (ip - start);
        if (pass == 0) {
          Layer(il, ip, blocks, nullptr, n);
          Tap(v++, n);
          continue;
        }
        // per-layer message queries (attn, k, v, ffn) from the global tables;
        // see inkling-port-state.md
        const std::uint32_t cc = ip - start;
        const float* mq = m->mq_table + static_cast<std::size_t>(cc) * 4 * h;
        Mixer(bank, bank_out, mq, 4, n, msgs);
        Layer(il, ip, blocks, msgs, n);
        Tap(v++, n);
      }
    }
    for (std::uint32_t il = end; il < m->n_phys; ++il) {
      Layer(il, il, blocks, nullptr, n);
      Tap(v++, n);
    }
    PrefetchJoin();
    Mixer(blocks, cur,
          static_cast<const float*>(m->T("output_res_query.weight").dev), 1, n,
          x);
    RmsRows(x, static_cast<const float*>(m->T("output_norm.weight").dev), x, n,
            h, h, h);
    Tap(48, n, x);
    const Tensor& out = m->T("output.weight");
    if (all_rows && n <= static_cast<int>(kMaxVerifyRows)) {
      Dense(out, x, h, d_logits, c.vocab_unpadded, n,
            static_cast<int>(c.vocab_unpadded));
      Softcap<<<(static_cast<std::size_t>(n) * c.vocab_unpadded + 255) / 256,
                256, 0, stream>>>(d_logits, n * c.vocab_unpadded,
                                  c.final_logit_softcapping);
    } else {
      Dense(out, x + static_cast<std::size_t>(n - 1) * h, h, d_logits,
            c.vocab_unpadded, 1, static_cast<int>(c.vocab_unpadded));
      Softcap<<<(c.vocab_unpadded + 255) / 256, 256, 0, stream>>>(
          d_logits, c.vocab_unpadded, c.final_logit_softcapping);
    }
  }
};

std::size_t Session::AllocatedBytes() const noexcept {
  return impl_->allocated_bytes;
}

struct SnapshotHeader {
  char magic[4]{'I', 'N', 'K', 'S'};
  std::uint32_t version{SessionSnapshot::kPayloadVersion};
  std::uint32_t ctx{0};
  std::uint32_t position{0};
  std::uint32_t dwarmed{0};
  std::uint32_t drafter{0};
  std::uint32_t sparse{0};
  std::int32_t argmax{0};
  std::uint64_t vocab{0};
  std::uint64_t body_bytes{0};
};

// Snapshots live in GPU-allocated memory (the RAM is shared with the CPU):
// capture and restore are device-to-device copies at memory speed, while a
// device-to-host copy runs at only ~8 GB/s here. Freed buffers are recycled.
// device-to-device region copy (a kernel moves ~4x faster than the copy engine)
__global__ void CopyBytes(const unsigned char* src, unsigned char* dst,
                          std::size_t bytes) {
  const std::size_t n16 = bytes / 16;
  const uint4* s4 = reinterpret_cast<const uint4*>(src);
  uint4* d4 = reinterpret_cast<uint4*>(dst);
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       i < n16; i += stride)
    d4[i] = s4[i];
  for (std::size_t i = n16 * 16 + static_cast<std::size_t>(blockIdx.x) *
                                      blockDim.x + threadIdx.x;
       i < bytes; i += stride)
    dst[i] = src[i];
}

static void CopyDev(void* dst, const void* src, std::size_t bytes,
                    hipStream_t stream) {
  static const bool kernel = std::getenv("INKLING_SNAP_KERNEL") == nullptr ||
                             std::getenv("INKLING_SNAP_KERNEL")[0] != '0';
  const auto a = reinterpret_cast<std::uintptr_t>(src);
  const auto b = reinterpret_cast<std::uintptr_t>(dst);
  if (kernel && bytes >= (1U << 16) && a % 16 == 0 && b % 16 == 0) {
    CopyBytes<<<1024, 256, 0, stream>>>(
        static_cast<const unsigned char*>(src),
        static_cast<unsigned char*>(dst), bytes);
    return;
  }
  HIP_OK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream));
}

class SnapshotDevicePool {
public:
  static SnapshotDevicePool& Instance() {
    static SnapshotDevicePool pool;
    return pool;
  }
  void* Get(std::size_t size, std::size_t* capacity) {
    constexpr std::size_t kRound = std::size_t{256} << 20;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (std::size_t i = 0; i < free_.size(); ++i) {
        if (free_[i].second >= size) {
          void* p = free_[i].first;
          *capacity = free_[i].second;
          free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(i));
          return p;
        }
      }
    }
    // headroom: the next turn's snapshot is a little larger and must still fit
    const std::size_t want = size + size / 4;
    const std::size_t cap = (want + kRound - 1) / kRound * kRound;
    void* p = nullptr;
    if (hipMalloc(&p, cap) != hipSuccess) {
      (void)hipGetLastError();
      return nullptr;
    }
    *capacity = cap;
    return p;
  }
  void Put(void* p, std::size_t capacity) {
    constexpr std::size_t kMaxPooled = 3;
    std::lock_guard<std::mutex> lock(mu_);
    if (free_.size() >= kMaxPooled) {
      (void)hipFree(p);
      return;
    }
    free_.emplace_back(p, capacity);
  }

private:
  std::mutex mu_;
  std::vector<std::pair<void*, std::size_t>> free_;
};

struct SessionSnapshot::Data {
  ~Data() {
    if (dev != nullptr)
      SnapshotDevicePool::Instance().Put(dev, dev_capacity);
  }
  void* dev{nullptr};
  std::size_t dev_capacity{0};
  bool sparse{false};
  std::uint32_t position{0};
  std::uint32_t dwarmed{0};
  std::uint32_t ctx{0};
  bool drafter{false};
  int argmax{0};
  std::vector<float> logits;
  std::unique_ptr<unsigned char[]> bytes;
  std::size_t size{0};
};

SessionSnapshot::SessionSnapshot() = default;
SessionSnapshot::~SessionSnapshot() = default;
std::uint64_t SessionSnapshot::SizeBytes() const noexcept {
  return data_ == nullptr ? 0
                          : data_->size + data_->logits.size() * sizeof(float);
}
std::uint32_t SessionSnapshot::position() const noexcept {
  return data_ == nullptr ? 0 : data_->position;
}

void Session::PrepareDraft() {
  if (!impl_->m->has_draft || impl_->feat == nullptr)
    return;
  if (impl_->m->has_dspark)
    impl_->DraftEnsure();
  if (impl_->m->d2)
    impl_->DraftEnsure2();
}

void Session::SelectDrafter(bool dflash2) {
  impl_->prefer_d2 = dflash2;
}

bool Session::HasBothDrafters() const noexcept {
  return impl_->m->has_dspark && impl_->m->d2;
}

std::uint64_t Session::SnapshotBytes() const {
  auto& i = *impl_;
  std::uint64_t total = i.m->config.vocab_unpadded * sizeof(float);
  for (const auto& r : i.SnapRegions(i.position, i.dwarmed, i.dready))
    total += r.bytes;
  return total;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(std::string* error_msg) {
  auto& i = *impl_;
  (void)error_msg;
  const auto snap_t0 = std::chrono::steady_clock::now();
  auto snapshot = std::unique_ptr<SessionSnapshot>(new SessionSnapshot());
  snapshot->data_ = std::make_unique<SessionSnapshot::Data>();
  auto& d = *snapshot->data_;
  d.position = i.position;
  d.dwarmed = i.dwarmed;
  d.ctx = i.ctx;
  d.drafter = i.dready;
  d.sparse = i.sparse;
  const auto regions = i.SnapRegions(d.position, d.dwarmed, d.drafter);
  for (const auto& r : regions)
    d.size += r.bytes;
  static const bool on_device =
      std::getenv("INKLING_SNAP_DEVICE") == nullptr ||
      std::getenv("INKLING_SNAP_DEVICE")[0] != '0';
  if (on_device)
    d.dev = SnapshotDevicePool::Instance().Get(d.size, &d.dev_capacity);
  if (d.dev == nullptr)
    d.bytes.reset(new unsigned char[d.size]);
  std::size_t offset = 0;
  for (const auto& r : regions) {
    if (r.bytes > 0) {
      if (d.dev != nullptr)
        CopyDev(static_cast<unsigned char*>(d.dev) + offset, r.dev, r.bytes,
                i.stream);
      else
        HIP_OK(hipMemcpyAsync(d.bytes.get() + offset, r.dev, r.bytes,
                              hipMemcpyDeviceToHost, i.stream));
    }
    offset += r.bytes;
  }
  const std::size_t v = i.m->config.vocab_unpadded;
  d.logits.resize(v);
  HIP_OK(hipMemcpyAsync(
      d.logits.data(), i.d_logits + static_cast<std::size_t>(i.logits_last) * v,
      v * sizeof(float), hipMemcpyDeviceToHost, i.stream));
  d.argmax = i.argmax_host[i.logits_last];
  HIP_OK(hipStreamSynchronize(i.stream));
  if (std::getenv("INKLING_SNAP_LOG") != nullptr)
    std::fprintf(stderr, "inkling: snapshot pos %u size %.2f GB %s in %.1f ms\n",
                 d.position, static_cast<double>(d.size) / 1e9,
                 d.dev != nullptr ? "device" : "host",
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - snap_t0)
                     .count());
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  auto& i = *impl_;
  if (snapshot.data_ == nullptr || snapshot.data_->ctx != i.ctx ||
      snapshot.data_->sparse != i.sparse) {
    if (error_msg != nullptr)
      *error_msg = "snapshot does not match this session's context size";
    return false;
  }
  const auto& d = *snapshot.data_;
  if (d.drafter && !i.dready && i.m->has_dspark)
    i.DraftEnsure();
  const auto regions = i.SnapRegions(d.position, d.dwarmed, d.drafter);
  std::size_t offset = 0;
  for (const auto& r : regions) {
    if (r.bytes > 0) {
      if (d.dev != nullptr)
        CopyDev(r.dev, static_cast<const unsigned char*>(d.dev) + offset,
                r.bytes, i.stream);
      else
        HIP_OK(hipMemcpyAsync(r.dev, d.bytes.get() + offset, r.bytes,
                              hipMemcpyHostToDevice, i.stream));
    }
    offset += r.bytes;
  }
  if (offset != d.size) {
    if (error_msg != nullptr)
      *error_msg = "snapshot layout does not match this session";
    return false;
  }
  HIP_OK(hipMemcpyAsync(i.d_logits, d.logits.data(),
                        d.logits.size() * sizeof(float), hipMemcpyHostToDevice,
                        i.stream));
  i.ClearPast(d.position);
  HIP_OK(hipStreamSynchronize(i.stream));
  i.position = d.position;
  i.dwarmed = i.m->has_dspark ? d.dwarmed : 0;
  i.d2warmed = 0;
  i.argmax_host[0] = d.argmax;
  i.logits_rows = 1;
  i.logits_last = 0;
  i.logits_have.assign(1, 0);
  i.last_rows = 1;
  i.pos_before_last = d.position > 0 ? d.position - 1 : 0;
  return true;
}

namespace {
SnapshotHeader HeaderOf(const SessionSnapshot::Data& d) {
  SnapshotHeader header;
  header.ctx = d.ctx;
  header.position = d.position;
  header.dwarmed = d.dwarmed;
  header.drafter = d.drafter ? 1U : 0U;
  header.sparse = d.sparse ? 1U : 0U;
  header.argmax = d.argmax;
  header.vocab = d.logits.size();
  header.body_bytes = d.size;
  return header;
}
}  // namespace

std::uint64_t SessionSnapshot::PayloadBytes() const noexcept {
  return data_ == nullptr
             ? 0
             : sizeof(SnapshotHeader) + data_->logits.size() * sizeof(float) +
                   data_->size;
}

void SessionSnapshot::StreamPayload(
    const std::function<void(std::span<const std::uint8_t>)>& sink) const {
  if (data_ == nullptr)
    return;
  std::vector<std::uint8_t> head(sizeof(SnapshotHeader) +
                                 data_->logits.size() * sizeof(float));
  const SnapshotHeader header = HeaderOf(*data_);
  std::memcpy(head.data(), &header, sizeof(header));
  std::memcpy(head.data() + sizeof(header), data_->logits.data(),
              data_->logits.size() * sizeof(float));
  sink(head);
  if (data_->dev == nullptr) {
    sink(std::span<const std::uint8_t>(data_->bytes.get(), data_->size));
    return;
  }
  // device-resident: stage through pinned chunks on the persistence worker
  constexpr std::size_t kChunk = std::size_t{64} << 20;
  hipStream_t st = nullptr;
  HIP_OK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
  void* stage = nullptr;
  HIP_OK(hipHostMalloc(&stage, kChunk, hipHostMallocDefault));
  for (std::size_t o = 0; o < data_->size; o += kChunk) {
    const std::size_t n = std::min(kChunk, data_->size - o);
    HIP_OK(hipMemcpyAsync(stage,
                          static_cast<const unsigned char*>(data_->dev) + o, n,
                          hipMemcpyDeviceToHost, st));
    HIP_OK(hipStreamSynchronize(st));
    sink(std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(stage),
                                       n));
  }
  (void)hipHostFree(stage);
  (void)hipStreamDestroy(st);
}

bool SessionSnapshot::CopyPayloadTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (data_ == nullptr || destination.size() != PayloadBytes())
    return false;
  std::size_t offset = 0;
  StreamPayload([&](std::span<const std::uint8_t> part) {
    std::memcpy(destination.data() + offset, part.data(), part.size());
    offset += part.size();
  });
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  auto& i = *impl_;
  const auto fail = [&](const char* message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  if (payload.size() < sizeof(SnapshotHeader))
    return fail("snapshot payload is truncated");
  SnapshotHeader header;
  std::memcpy(&header, payload.data(), sizeof(header));
  if (std::memcmp(header.magic, "INKS", 4) != 0 ||
      header.version != SessionSnapshot::kPayloadVersion)
    return fail("snapshot payload version is not supported");
  const std::size_t vocab = i.m->config.vocab_unpadded;
  if (header.ctx != i.ctx || header.vocab != vocab ||
      (header.sparse != 0) != i.sparse)
    return fail("snapshot does not match this session's layout");
  if (payload.size() !=
      sizeof(header) + vocab * sizeof(float) + header.body_bytes)
    return fail("snapshot payload size does not match its header");
  const bool drafter = header.drafter != 0;
  if (drafter && !i.dready && i.m->has_dspark)
    i.DraftEnsure();
  const auto regions = i.SnapRegions(header.position, header.dwarmed, drafter);
  std::size_t total = 0;
  for (const auto& r : regions)
    total += r.bytes;
  if (total != header.body_bytes)
    return fail("snapshot layout does not match this session");
  const std::uint8_t* body =
      payload.data() + sizeof(header) + vocab * sizeof(float);
  std::size_t offset = 0;
  for (const auto& r : regions) {
    if (r.bytes > 0)
      HIP_OK(hipMemcpyAsync(r.dev, body + offset, r.bytes,
                            hipMemcpyHostToDevice, i.stream));
    offset += r.bytes;
  }
  HIP_OK(hipMemcpyAsync(i.d_logits, payload.data() + sizeof(header),
                        vocab * sizeof(float), hipMemcpyHostToDevice,
                        i.stream));
  i.ClearPast(header.position);
  HIP_OK(hipStreamSynchronize(i.stream));
  i.position = header.position;
  i.dwarmed = i.m->has_dspark ? header.dwarmed : 0;
  i.d2warmed = 0;
  i.argmax_host[0] = header.argmax;
  i.logits_rows = 1;
  i.logits_last = 0;
  i.logits_have.assign(1, 0);
  i.last_rows = 1;
  i.pos_before_last = header.position > 0 ? header.position - 1 : 0;
  return true;
}

Session::Session() = default;
Session::~Session() = default;

std::span<const float> Session::Logits() const noexcept {
  return Logits(static_cast<std::uint32_t>(impl_->logits_last));
}

std::span<const float> Session::Logits(std::uint32_t row) const noexcept {
  auto& i = *impl_;
  const std::size_t v = i.m->config.vocab_unpadded;
  if (i.logits.size() < static_cast<std::size_t>(i.logits_rows) * v)
    i.logits.resize(static_cast<std::size_t>(i.logits_rows) * v);
  if (i.logits_have[row] == 0) {
    HIP_OK(hipMemcpy(i.logits.data() + row * v, i.d_logits + row * v, v * 4,
                     hipMemcpyDeviceToHost));
    i.logits_have[row] = 1;
  }
  return std::span<const float>(i.logits).subspan(row * v, v);
}

std::int32_t Session::Argmax(std::uint32_t row) const noexcept {
  return impl_->argmax_host[row];
}

std::int32_t Session::Argmax() const noexcept {
  return impl_->argmax_host[impl_->logits_last];
}

bool Session::AppendAll(std::span<const std::int32_t> tokens,
                        std::string* error) {
  if (tokens.empty() || tokens.size() > kMaxVerifyRows ||
      impl_->position + tokens.size() > impl_->ctx) {
    if (error != nullptr)
      *error = "verification batch must be 1..8 tokens within the context";
    return false;
  }
  impl_->pos_before_last = impl_->position;
  impl_->last_rows = static_cast<int>(tokens.size());
  impl_->all_rows = true;
  impl_->Forward(tokens);
  impl_->all_rows = false;
  return true;
}

void Session::KeepRows(std::uint32_t keep) {
  auto& i = *impl_;
  if (static_cast<int>(keep) >= i.last_rows)
    return;
  const int max_ch = std::max<int>(
      static_cast<int>(i.m->config.hidden_size),
      static_cast<int>(i.m->config.num_kv * i.m->config.head_dim));
  FixHist<<<dim3((max_ch + 255) / 256, i.n_hist_entries), 256, 0, i.stream>>>(
      i.hist_entries, static_cast<int>(keep));
  HIP_OK(hipStreamSynchronize(i.stream));
  i.position = i.pos_before_last + keep;
  i.dwarmed = std::min(i.dwarmed, i.position);
  i.d2warmed = std::min(i.d2warmed, i.position);
  i.logits_last = static_cast<int>(keep) - 1;
  i.last_rows = static_cast<int>(keep);
}

bool Session::Draft(std::int32_t anchor, std::span<std::int32_t> out) {
  return impl_->Draft(anchor, out.data(), static_cast<int>(out.size()));
}

bool Session::Draft(std::int32_t anchor, std::span<std::int32_t> out,
                    std::span<float> confidence, float stop_confidence) {
  return impl_->Draft(anchor, out.data(), static_cast<int>(out.size()),
                      confidence.data(), stop_confidence);
}

void Session::Rewind(std::uint32_t position) noexcept {
  impl_->position = position;
  impl_->dwarmed = std::min(impl_->dwarmed, position);
  impl_->d2warmed = std::min(impl_->d2warmed, position);
}
std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

bool Session::GptqFinish(const std::string& dir, std::string* error) {
  return impl_->GptqFinish(dir, error);
}

void Session::Reset() noexcept {
  impl_->position = 0;
  impl_->dwarmed = 0;
  impl_->d2warmed = 0;
  const Config& c = impl_->m->config;
  const std::size_t kv = static_cast<std::size_t>(c.num_kv) * c.head_dim;
  for (auto& s : impl_->slots) {
    (void)hipMemset(s.hist_k, 0, kv * kHistory * 4);
    (void)hipMemset(s.hist_v, 0, kv * kHistory * 4);
    (void)hipMemset(s.hist_attn, 0,
                    static_cast<std::size_t>(c.hidden_size) * kHistory * 4);
    (void)hipMemset(s.hist_ffn, 0,
                    static_cast<std::size_t>(c.hidden_size) * kHistory * 4);
  }
}

bool Session::Append(std::span<const std::int32_t> tokens, std::string* error) {
  if (tokens.empty() || impl_->position + tokens.size() > impl_->ctx) {
    if (error != nullptr)
      *error = "context exceeded or empty append";
    return false;
  }
  const std::size_t chunk = impl_->banked ? 2 * impl_->bank_rows : kMaxBatch;
  for (std::size_t off = 0; off < tokens.size(); off += chunk) {
    impl_->Forward(
        tokens.subspan(off, std::min<std::size_t>(chunk, tokens.size() - off)));
  }
  return true;
}

bool Model::LoadDrafter(const std::string& path, std::string* error) {
  auto reader = gufo::core::GgufReader::OpenFile(path, error);
  if (!reader)
    return false;
  auto& st = *state_;
  const bool is_d2 = reader->FindTensor("selector_hidden.weight") != nullptr;
  const std::string prefix = is_d2 ? "d2." : "d.";
  for (const auto& info : reader->GetTensors()) {
    Tensor t;
    switch (info.type) {
      case GgmlType::kF32:
        t.type = kTypeF32;
        break;
      case GgmlType::kBF16:
        t.type = kTypeBf16;
        break;
      case GgmlType::kQ8_0:
        t.type = kTypeQ8;
        break;
      default:
        if (error != nullptr)
          *error =
              "unsupported drafter tensor type in " + std::string(info.name);
        return false;
    }
    t.cols = info.dimensions[0];
    t.rows = 1;
    for (std::size_t d = 1; d < info.dimensions.size(); ++d)
      t.rows *= info.dimensions[d];
    t.row_bytes = info.size_bytes / t.rows;
    const std::string name = prefix + std::string(info.name);
    if (t.type == kTypeQ8) {
      void* tmp = nullptr;
      HIP_OK(hipMalloc(&tmp, info.size_bytes));
      HIP_OK(hipMemcpy(tmp, info.data, info.size_bytes, hipMemcpyHostToDevice));
      const std::size_t stride = (t.cols + (t.cols / 32) * 2 + 15) / 16 * 16;
      HIP_OK(hipMalloc(&t.dev, stride * t.rows));
      HIP_OK(hipMemset(t.dev, 0, stride * t.rows));
      RepackRows<<<static_cast<unsigned>(t.rows), 256>>>(
          static_cast<const unsigned char*>(tmp), t.row_bytes,
          static_cast<unsigned char*>(t.dev), stride, static_cast<int>(t.cols),
          false);
      HIP_OK(hipDeviceSynchronize());
      HIP_OK(hipFree(tmp));
      t.row_bytes = stride;
      st.resident_bytes += stride * t.rows;
    } else {
      HIP_OK(hipMalloc(&t.dev, info.size_bytes));
      HIP_OK(
          hipMemcpy(t.dev, info.data, info.size_bytes, hipMemcpyHostToDevice));
      st.resident_bytes += info.size_bytes;
    }
    st.allocations.push_back(t.dev);
    if (info.name == "dspark_fuse.weight") {
      const float* f = static_cast<const float*>(info.data);
      for (int r = 0; r < 4; ++r) {
        float mx = f[r * 4];
        for (int k = 1; k < 4; ++k)
          mx = std::max(mx, f[r * 4 + k]);
        float sum = 0.0F;
        for (int k = 0; k < 4; ++k)
          sum += std::exp(f[r * 4 + k] - mx);
        for (int k = 0; k < 4; ++k)
          st.draft_fuse[r][k] = std::exp(f[r * 4 + k] - mx) / sum;
      }
    }
    if (info.name == "markov_w1.weight" && t.type == kTypeBf16) {
      const auto* raw = static_cast<const unsigned short*>(info.data);
      const std::size_t rows = t.rows;
      std::vector<std::pair<float, int>> norms(rows);
      for (std::size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (std::size_t c = 0; c < t.cols; ++c) {
          const std::uint32_t bits =
              static_cast<std::uint32_t>(raw[r * t.cols + c]) << 16;
          float v;
          std::memcpy(&v, &bits, 4);
          sum += static_cast<double>(v) * v;
        }
        norms[r] = {static_cast<float>(sum), static_cast<int>(r)};
      }
      const std::size_t keep = std::min<std::size_t>(65536, rows);
      std::partial_sort(norms.begin(), norms.begin() + keep, norms.end(),
                        [](const auto& a, const auto& b) {
                          return a.first > b.first ||
                                 (a.first == b.first && a.second < b.second);
                        });
      st.draft_rank.clear();
      for (std::size_t i = 0; i < keep; ++i)
        if (norms[i].second < static_cast<int>(st.config.vocab_unpadded))
          st.draft_rank.push_back(norms[i].second);
    }
    st.tensors.emplace(name, t);
  }
  if (!is_d2)
    st.has_dspark = true;
  if (is_d2) {
    st.d2 = true;
    if (const auto w =
            reader->GetMetadataUint32("dflash.attention.sliding_window"))
      st.d2_window = static_cast<int>(*w);
    if (const auto w = reader->GetMetadataUint32("dflash.mask_token_id"))
      st.d2_mask = static_cast<int>(*w);
    if (const auto w = reader->GetMetadataFloat32("dflash.rope.freq_base"))
      st.d2_theta = *w;
    if (const auto w =
            reader->GetMetadataFloat32("dflash.final_logit_softcapping"))
      st.d2_softcap = *w;
  }
  st.has_draft = true;
  return true;
}

bool Model::HasDrafter() const noexcept {
  return state_->has_draft;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error) {
  (void)error;
  const Config& c = state_->config;
  auto s = std::unique_ptr<Session>(new Session());
  s->impl_ = std::make_unique<Session::Impl>();
  auto& i = *s->impl_;
  i.m = state_;
  i.ctx = max_context;
  const std::size_t kv = static_cast<std::size_t>(c.num_kv) * c.head_dim;
  const std::size_t h = c.hidden_size;
  const std::size_t b = kMaxBatch;
  {
    const char* e = std::getenv("INKLING_SPARSE");
    const bool fits = max_context / kBlockKeys + 2 <= kMaxSparseBlocks + 2;
    const bool wanted =
        e != nullptr ? e[0] == '1' : max_context >= kSparseAutoContext;
    i.sparse = wanted && fits;
  }
  i.slots.resize(c.num_layers);
  for (auto& sl : i.slots) {
    const std::size_t slot_index = &sl - i.slots.data();
    const std::size_t ring_rows = c.sliding_window + kMaxBatch;
    const bool use_ring = c.is_swa[slot_index] && max_context > ring_rows;
    sl.ring = use_ring ? static_cast<int>(ring_rows) : 0;
    const std::size_t rows =
        (use_ring ? ring_rows : max_context + 63) / 64 * 64;
    sl.vblocks = static_cast<int>(rows / 64);
    sl.k = static_cast<__half*>(i.AllocBytes(rows * kv * 2));
    if (i.sparse && !c.is_swa[slot_index]) {
      sl.kmm = static_cast<__half*>(
          i.AllocBytes(static_cast<std::size_t>(max_context / kBlockKeys + 2) *
                       c.num_kv * 2 * kFlashHd * 2));
    }
    sl.v = static_cast<__half*>(i.AllocBytes(rows * kv * 2));
    sl.hist_k =
        static_cast<float*>(i.AllocBytes(kv * (kHistory + kWinLen) * 4));
    sl.hist_v =
        static_cast<float*>(i.AllocBytes(kv * (kHistory + kWinLen) * 4));
    sl.hist_attn =
        static_cast<float*>(i.AllocBytes(h * (kHistory + kWinLen) * 4));
    sl.hist_ffn =
        static_cast<float*>(i.AllocBytes(h * (kHistory + kWinLen) * 4));
  }
  const std::size_t qd = static_cast<std::size_t>(c.num_heads) * c.head_dim;
  auto alloc_scratch = [&](Scratch& sc, std::size_t b) {
    sc.d_tokens = static_cast<int*>(i.AllocBytes(b * 4));
    sc.cur = i.Alloc(b * h);
    sc.tmp = i.Alloc(b * std::max<std::size_t>(kv, h));
    sc.inp_sa = i.Alloc(b * h);
    for (auto& p : sc.in)
      p = i.Alloc(b * h);
    sc.q = i.Alloc(b * qd);
    sc.k = i.Alloc(b * kv);
    sc.v = i.Alloc(b * kv);
    sc.r = i.Alloc(b * c.relative_length * c.num_heads);
    sc.dyn_k = i.Alloc(b * kConvTaps * c.num_kv);
    sc.dyn_v = i.Alloc(b * kConvTaps * c.num_kv);
    sc.sink = i.Alloc(b * c.num_heads);
    sc.gate = i.Alloc(b * qd);
    sc.att = i.Alloc(b * qd);
    sc.o = i.Alloc(b * h);
    sc.mixed = i.Alloc(b * h * 3);
    sc.msgs = i.Alloc(b * h * 4);
    sc.ffn_inp = i.Alloc(b * h);
    sc.x = i.Alloc(b * h);
    sc.cat = i.Alloc(b * h * 2);
    sc.low = i.Alloc(std::max<std::size_t>(b * c.loop_gate_rank, h * 4));
    sc.gain = i.Alloc(b * h);
    sc.forget = i.Alloc(b * h);
    sc.router = i.Alloc(b * c.expert_count);
    sc.wts = i.Alloc(b * c.expert_used);
    sc.gu = i.Alloc(b * c.expert_used * 2 * c.expert_ff);
    sc.act = i.Alloc(b * c.expert_used * c.expert_ff);
    sc.y = i.Alloc(b * c.expert_used * h);
    sc.moe = i.Alloc(b * h);
    sc.prelude = i.Alloc(b * h);
    sc.bank_out = i.Alloc(b * h);
    sc.d_logits = i.Alloc(kMaxVerifyRows * c.vocab_padded);
    i.d_argmax = static_cast<int*>(i.AllocBytes(kMaxVerifyRows * 4));
    HIP_OK(hipStreamCreateWithFlags(&i.pf_stream, hipStreamNonBlocking));
    HIP_OK(hipEventCreateWithFlags(&i.pf_fork, hipEventDisableTiming));
    HIP_OK(hipEventCreateWithFlags(&i.pf_join, hipEventDisableTiming));
    i.pf_sink = static_cast<unsigned int*>(i.AllocBytes(64));
    i.xqm = static_cast<signed char*>(i.AllocBytes(std::size_t{3} * 8 * 2304));
    i.xqsm = static_cast<float*>(
        i.AllocBytes(std::size_t{3} * 8 * (2304 / 32) * sizeof(float)));
    sc.d_pos = static_cast<int*>(i.AllocBytes(4));
    if (state_->npu) {
      std::string npu_error;
      for (int j = 0; j < 3; ++j)
        i.npu_a[j] = state_->npu->NewBuffer(state_->npu->ABytes(), &npu_error);
      for (int j = 0; j < 8; ++j)
        i.npu_c[j] = state_->npu->NewBuffer(state_->npu->CBytes(), &npu_error);
    }
    sc.gu16 = static_cast<__half*>(
        i.AllocBytes(b * c.expert_used * 2 * c.expert_ff * 2));
    sc.y16 = static_cast<__half*>(i.AllocBytes(b * c.expert_used * h * 2));
    sc.xh0 = static_cast<__half*>(i.AllocBytes(b * h * 2));
    sc.xh1 = static_cast<__half*>(i.AllocBytes(b * h * 2));
    sc.xh2 = static_cast<__half*>(i.AllocBytes(b * h * 2));
    sc.xhx = static_cast<__half*>(i.AllocBytes(b * h * 2));
    sc.atth = static_cast<__half*>(i.AllocBytes(b * qd * 2));
    sc.cath = static_cast<__half*>(i.AllocBytes(b * h * 4));
    sc.rcounts = static_cast<unsigned int*>(
        i.AllocBytes(c.expert_count * sizeof(unsigned int)));
    sc.rbounds =
        static_cast<int*>(i.AllocBytes((c.expert_count + 1) * sizeof(int)));
    sc.rcursors = static_cast<int*>(i.AllocBytes(c.expert_count * sizeof(int)));
    sc.ksbuf = static_cast<float*>(
        i.AllocBytes(static_cast<std::size_t>(4) * b * 128 * sizeof(float)));
    sc.dense_bounds_buf = static_cast<int*>(i.AllocBytes(2 * sizeof(int)));
    sc.tl_a = static_cast<int*>(i.AllocBytes(1024 * sizeof(int)));
    sc.tl_b = static_cast<int*>(i.AllocBytes(1024 * sizeof(int)));
    {
      const std::size_t rows =
          gufo::models::qwen38_flash_next::rocm::RoutedCompactRows(
              b * c.expert_used, c.expert_count);
      sc.rrows_token = static_cast<int*>(i.AllocBytes(rows * sizeof(int)));
      sc.rrows_slot = static_cast<int*>(i.AllocBytes(rows * sizeof(int)));
    }
    sc.xq = static_cast<signed char*>(
        i.AllocBytes(static_cast<std::size_t>(kScratchRows) * c.expert_used *
                     std::max<std::size_t>(c.hidden_size, 3072) * 2));
    sc.xqs = i.Alloc(static_cast<std::size_t>(kScratchRows) * c.expert_used *
                     std::max<std::size_t>(c.hidden_size, 3072) / 16);
    sc.blist = static_cast<int*>(
        i.AllocBytes(static_cast<std::size_t>(kMaxBatch / kFlashQ) *
                     c.num_heads * kListCap * 4));
    sc.bcount = static_cast<int*>(i.AllocBytes(
        static_cast<std::size_t>(kMaxBatch / kFlashQ) * c.num_heads * 4));
    {
      const std::size_t splits = (max_context + kSplitKeys - 1) / kSplitKeys;
      sc.split_m = i.Alloc(kScratchRows * c.num_heads * splits);
      sc.split_l = i.Alloc(kScratchRows * c.num_heads * splits);
      sc.split_o = i.Alloc(kScratchRows * c.num_heads * splits * kFlashHd);
    }
    sc.xh = static_cast<__half*>(
        i.AllocBytes(b * std::max<std::size_t>(qd, 2 * h) * 2));
    sc.act16 =
        static_cast<__half*>(i.AllocBytes(b * c.expert_used * c.expert_ff * 2));
    sc.sorted = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
    sc.tiles = static_cast<int*>(
        i.AllocBytes((b * c.expert_used / kTileM + c.expert_count) * 3 * 4));
    sc.ids = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
    sc.xrow = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
    for (auto& p : sc.block_pool)
      p = i.Alloc(b * h);
    if (state_->npu) {
      std::string npu_error;
      for (int j = 0; j < 3; ++j)
        sc.npu_a[j] = state_->npu->NewBuffer(state_->npu->ABytes(), &npu_error);
      for (int j = 0; j < 8; ++j)
        sc.npu_c[j] = state_->npu->NewBuffer(state_->npu->CBytes(), &npu_error);
      sc.npu_ao = state_->npu->NewBuffer(state_->npu->ABytes(3072), &npu_error);
    }
  };
  alloc_scratch(static_cast<Scratch&>(i), b);
  {
    std::vector<int> tok(b * c.expert_used);
    std::vector<int> item(b * c.expert_used);
    for (std::size_t j = 0; j < tok.size(); ++j) {
      tok[j] = static_cast<int>(j / c.expert_used);
      item[j] = static_cast<int>(j);
    }
    i.xrow_token = static_cast<int*>(i.AllocBytes(tok.size() * 4));
    i.xrow_item = static_cast<int*>(i.AllocBytes(item.size() * 4));
    HIP_OK(hipMemcpy(i.xrow_token, tok.data(), tok.size() * 4,
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(i.xrow_item, item.data(), item.size() * 4,
                     hipMemcpyHostToDevice));
  }
  HIP_OK(hipStreamCreateWithFlags(&i.stream, hipStreamNonBlocking));
  if (const char* tp = std::getenv("INKLING_TAPS"); tp != nullptr) {
    std::sscanf(tp, "%d,%d,%d,%d", &g_tap_layers[0], &g_tap_layers[1],
                &g_tap_layers[2], &g_tap_layers[3]);
  }
  if (std::getenv("INKLING_DRAFT") != nullptr || state_->has_draft) {
    i.feat = static_cast<__half*>(i.AllocBytes(
        static_cast<std::size_t>(max_context) * kTapCount * h * 2));
  }
  {
    std::vector<HistEntry> entries;
    for (auto& sl : i.slots) {
      entries.push_back({sl.hist_k, static_cast<int>(kv)});
      entries.push_back({sl.hist_v, static_cast<int>(kv)});
      entries.push_back({sl.hist_attn, static_cast<int>(h)});
      entries.push_back({sl.hist_ffn, static_cast<int>(h)});
    }
    i.hist_entries = static_cast<HistEntry*>(
        i.AllocBytes(entries.size() * sizeof(HistEntry)));
    HIP_OK(hipMemcpy(i.hist_entries, entries.data(),
                     entries.size() * sizeof(HistEntry),
                     hipMemcpyHostToDevice));
    i.n_hist_entries = static_cast<int>(entries.size());
  }
  if (state_->npu || std::getenv("INKLING_BANKS") != nullptr) {
    i.banked = true;
    i.bank_rows = state_->npu ? state_->npu_tokens : 2048;
    alloc_scratch(i.saved[1], i.bank_rows);
    const char* streams_env = std::getenv("INKLING_BANK_STREAMS");
    const bool want_streams =
        streams_env != nullptr ? streams_env[0] != '0' : state_->npu != nullptr;
    if (want_streams) {
      i.stream0 = i.stream;
      HIP_OK(hipStreamCreateWithFlags(&i.stream1, hipStreamNonBlocking));
      HIP_OK(hipEventCreateWithFlags(&i.ev_bank, hipEventDisableTiming));
      i.two_stream = true;
      if (state_->npu) {
        HIP_OK(hipStreamCreateWithFlags(&i.npu_ctl, hipStreamNonBlocking));
        for (auto& e : i.npu_ev)
          HIP_OK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
      }
    }
  }
  return s;
}

}  // namespace gufo::models::inkling::gpu
