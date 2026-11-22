#include "src/models/inkling/gpu/engine.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
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
constexpr int kWmmaMinTokens = 32;
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
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  const float r = sh[0];
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
                        const float* xs, float* y) {
  const int lane = threadIdx.x % warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (r >= rows)
    return;
  const int i = blockIdx.y;
  const int g = group != nullptr ? group[i] : 0;
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
      const uint4 q = *reinterpret_cast<const uint4*>(row + u * 16);
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
    out[blockIdx.x * ld_out + i] = v;
    if (out_h != nullptr)
      out_h[blockIdx.x * ld_out + i] = __float2half(v);
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

__global__ void TauScale(float* x, int per_token, const int* pos0,
                         float floor_v, float alpha) {
  const float tau =
      1.0F +
      alpha * logf(fmaxf(1.0F,
                         static_cast<float>(*pos0 + blockIdx.x + 1) / floor_v));
  for (int i = threadIdx.x; i < per_token; i += blockDim.x)
    x[static_cast<std::size_t>(blockIdx.x) * per_token + i] *= tau;
}

__global__ void Conv(const float* cur, const float* w, const float* dyn,
                     int heads, float* hist, float* out, int ch, int n,
                     const float* add, float* win) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ch)
    return;
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
};

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
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
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
  __shared__ float red[256][5];
  __shared__ float inv[kMaxStates];
  __shared__ float score[4][kMaxStates];
  const int t = blockIdx.x;
  const int ns = blocks.count + 1;
  const float* st[kMaxStates];
  for (int s = 0; s < blocks.count; ++s)
    st[s] = blocks.p[s] + static_cast<std::size_t>(t) * h;
  st[blocks.count] = cur + static_cast<std::size_t>(t) * h;
  for (int s = 0; s < ns; ++s) {
    float acc[5] = {};
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
      const float v = st[s][i];
      acc[0] += v * v;
      for (int q = 0; q < nq; ++q)
        acc[1 + q] += v * query[q * h + i];
    }
    for (int k = 0; k < 5; ++k)
      red[threadIdx.x][k] = acc[k];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int k = 0; k < 5; ++k)
          red[threadIdx.x][k] += red[threadIdx.x + stride][k];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      inv[s] = rsqrtf(red[0][0] / static_cast<float>(h) + kMixEps);
      for (int q = 0; q < nq; ++q)
        score[q][s] = red[0][1 + q] * inv[s] * rsqrtf(static_cast<float>(h));
    }
    __syncthreads();
  }
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
        a += e[s] / den * st[s][i] * inv[s];
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
  __shared__ float red[256][5];
  __shared__ float inv[kMaxStates];
  __shared__ float score[4][kMaxStates];
  __shared__ float rsum[256];
  const int t = blockIdx.x;
  const int ns = blocks.count + 1;
  const float* st[kMaxStates];
  for (int s = 0; s < blocks.count; ++s)
    st[s] = blocks.p[s] + static_cast<std::size_t>(t) * h;
  st[blocks.count] = cur + static_cast<std::size_t>(t) * h;
  for (int s = 0; s < ns; ++s) {
    float acc[5] = {};
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
      const float v = st[s][i];
      acc[0] += v * v;
      for (int q = 0; q < nq; ++q)
        acc[1 + q] += v * query[q * h + i];
    }
    for (int k = 0; k < 5; ++k)
      red[threadIdx.x][k] = acc[k];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int k = 0; k < 5; ++k)
          red[threadIdx.x][k] += red[threadIdx.x + stride][k];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      inv[s] = rsqrtf(red[0][0] / static_cast<float>(h) + kMixEps);
      for (int q = 0; q < nq; ++q)
        score[q][s] = red[0][1 + q] * inv[s] * rsqrtf(static_cast<float>(h));
    }
    __syncthreads();
  }
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
        a += e[s] / den * st[s][i] * inv[s];
      mixbuf[q * h + i] = a;
      ss += a * a;
    }
    rsum[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride)
        rsum[threadIdx.x] += rsum[threadIdx.x + stride];
      __syncthreads();
    }
    const float r = rsqrtf(rsum[0] / static_cast<float>(h) + eps);
    __syncthreads();
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
__global__ void __launch_bounds__(kControlWaves * 32)
    ControlLow(const float* cur, const float* msg, const float* down, int h,
               float eps, float* low) {
  __shared__ float red[2][kControlWaves];
  const int row = blockIdx.x;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float* c = cur + static_cast<std::size_t>(row) * h;
  const float* m = msg + static_cast<std::size_t>(row) * h;
  float sc = 0.0F;
  float sm = 0.0F;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    sc += c[i] * c[i];
    sm += m[i] * m[i];
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
  const float* w = down + static_cast<std::size_t>(r) * 2 * h;
  float acc = 0.0F;
  for (int i = lane; i < h; i += 32)
    acc += w[i] * c[i] * ic + w[h + i] * m[i] * im;
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off);
  if (lane == 0)
    low[static_cast<std::size_t>(row) * kControlRank + r] = acc;
}

// cur += -tanh(forget) * cur + tanh(gain) * msg with gain/forget = W . low
__global__ void ControlApply(float* cur, const float* msg, const float* low,
                             const float* gain_w, const float* forget_w,
                             int h) {
  __shared__ float lw[kControlRank];
  const int row = blockIdx.x;
  if (threadIdx.x < kControlRank)
    lw[threadIdx.x] =
        low[static_cast<std::size_t>(row) * kControlRank + threadIdx.x];
  __syncthreads();
  const int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= h)
    return;
  const float4* g4 = reinterpret_cast<const float4*>(
      gain_w + static_cast<std::size_t>(j) * kControlRank);
  const float4* f4 = reinterpret_cast<const float4*>(
      forget_w + static_cast<std::size_t>(j) * kControlRank);
  float g = 0.0F;
  float f = 0.0F;
  for (int k = 0; k < kControlRank / 4; ++k) {
    const float4 gv = g4[k];
    const float4 fv = f4[k];
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
                      float wscale, int* ids, float* wts) {
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
    best_v[e] = sel[e];
    best_i[e] = e;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (e < s && (best_v[e + s] > best_v[e] || (best_v[e + s] == best_v[e] &&
                                                  best_i[e + s] < best_i[e]))) {
        best_v[e] = best_v[e + s];
        best_i[e] = best_i[e + s];
      }
      __syncthreads();
    }
    const int win = best_i[0];
    if (e == 0) {
      ids[t * nu + k] = win;
      wts[t * nu + k] = prob[win];
      sel[win] = -1e30F;
    }
    sum += prob[win];
    __syncthreads();
  }
  if (e < nu)
    wts[t * nu + e] = wts[t * nu + e] / (sum + 1e-20F) * wscale;
}

__global__ void Situ(const float* gu, const int* group, const float* scale,
                     float* act, int ff, float beta, float lb) {
  const int item = blockIdx.x;
  const float s = scale != nullptr ? scale[group[item]] : 1.0F;
  for (int j = threadIdx.x; j < ff; j += blockDim.x) {
    const float g = gu[static_cast<std::size_t>(item) * 2 * ff + j] * s;
    const float u = gu[static_cast<std::size_t>(item) * 2 * ff + ff + j] * s;
    act[static_cast<std::size_t>(item) * ff + j] =
        beta * tanhf(g / beta) * Sigmoid(g) * lb * tanhf(u / lb);
  }
}

__global__ void CountExperts(const int* ids, int items, unsigned int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < items)
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

__global__ void CombineHalf(const __half* y, const float* wts, float* out,
                            int nu, int h) {
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    float a = 0.0F;
    for (int k = 0; k < nu; ++k)
      a += wts[blockIdx.x * nu + k] *
           __half2float(
               y[(static_cast<std::size_t>(blockIdx.x) * nu + k) * h + i]);
    out[static_cast<std::size_t>(blockIdx.x) * h + i] = a;
  }
}

__global__ void Combine(const float* y, const float* wts, float* out, int nu,
                        int h, const float* down_scale, const int* ids) {
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    float a = 0.0F;
    for (int k = 0; k < nu; ++k) {
      const float sc =
          down_scale != nullptr ? down_scale[ids[blockIdx.x * nu + k]] : 1.0F;
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
  for (int k0 = 0; k0 < a.cols; k0 += 32) {
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
        a.y[static_cast<std::size_t>(out_item) * a.rows + nn] =
            acc[mi][ni][i] * sc;
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
constexpr int kMaxSparseBlocks = 2048;
constexpr std::uint32_t kSparseAutoContext = 65536;

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

// Flash attention with relative bias and a zero sink logit. One block: 64
// queries of one head; 4 waves x 16 query rows.
__global__ void __launch_bounds__(kFlashThreads)
    FlashRel(AttnArgs a, const float* q, const float* r, const float* sink,
             const __half* kc, const __half* vc, const __half* bias, float* out,
             int n, int nsplit, int split_keys, float* pm, float* pl, float* po,
             const int* blist, const int* bcount, int list_cap) {
  __shared__ __half Ks[kFlashK][kKStride];
  __shared__ __half Vt[kFlashHd][kVStride];
  // per wave: the bias rows Z (consumed by the score epilogue) and then the P
  // rows share one region
  __shared__ __half PZ[kFlashWaves][16 * kZStride];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int gh = a.nh / a.nkv;
  const int pk = a.pack;
  const int h0 = blockIdx.y;
  const int q0 = pk > 0 ? 0 : blockIdx.x * kFlashQ;
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
  const int qmax_pos = pos0 + (pk > 0 ? n : min(q0 + kFlashQ, n)) - 1;
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
    sinkv[i] = sink[static_cast<std::size_t>(t) * a.nh + row_head(row)];
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
    const int tile_head = blockIdx.x * a.nh + h0;
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
      constexpr int kKPer = kFlashThreads / kFlashK;  // threads per key
      constexpr int kKU4 = kFlashHd / kKPer / 8;      // uint4 per thread
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
      constexpr int kVPer = kFlashThreads / kFlashHd;  // threads per row
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
    float pmax[8];
    for (int i = 0; i < 8; ++i)
      pmax[i] = kNegInf;
    for (int ni = 0; ni < 4; ++ni) {
      const int jpos = kb + ni * 16 + (lane & 15);
      for (int i = 0; i < 8; ++i) {
        const int dist = qpos[i] - jpos;
        const bool valid = dist >= 0 && (a.window == 0 || dist < a.window);
        float v = kNegInf;
        if (valid) {
          const int lr = 2 * i + (lane >> 4);
          const float b =
              dist < a.extent
                  ? __half2float(PZ[wave][lr * kZStride + dist -
                                          (qp0 - (kb + kFlashK - 1))])
                  : 0.0F;
          v = sc[ni][i] * a.scale + b - sinkv[i];
        }
        sc[ni][i] = v;
        pmax[i] = fmaxf(pmax[i], v);
      }
    }
    for (int i = 0; i < 8; ++i) {
      for (int off = 8; off > 0; off >>= 1)
        pmax[i] = fmaxf(pmax[i], __shfl_xor(pmax[i], off));
      const float mn = fmaxf(m[i], pmax[i]);
      const float alpha = expf(m[i] - mn);
      m[i] = mn;
      l[i] *= alpha;
      for (int ni = 0; ni < 8; ++ni)
        o[ni][i] *= alpha;
    }
    for (int ni = 0; ni < 4; ++ni) {
      for (int i = 0; i < 8; ++i) {
        const float pv = sc[ni][i] <= kNegInf ? 0.0F : expf(sc[ni][i] - m[i]);
        l[i] += pv;
        PZ[wave][(2 * i + (lane >> 4)) * kPStride + ni * 16 + (lane & 15)] =
            __float2half(pv);
      }
    }
    __syncthreads();
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
        pm[part] = m[i];
        pl[part] = ls;
      }
      for (int ni = 0; ni < 8; ++ni)
        po[part * kFlashHd + ni * 16 + (lane & 15)] = o[ni][i];
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
                             int n, int nh, int nkv, int* blist, int* bcount) {
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
  const int g = h / (nh / nkv);
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
    for (int t = 0; t < rows; ++t) {
      const float v =
          q[(static_cast<std::size_t>(tile * kFlashQ + t) * nh + h) * kFlashHd +
            d];
      a += v;
      lo = fminf(lo, v);
      hi = fmaxf(hi, v);
    }
    qbar[d] = a / rows;
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
  std::size_t resident_bytes{0};
  bool has_draft{false};
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
      HIP_OK(hipMemcpy(tmp, info.data, info.size_bytes, hipMemcpyHostToDevice));
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
      state->allocations.push_back(t.dev);
      state->resident_bytes += stride * t.rows;
      state->tensors.emplace(std::string(info.name), t);
      continue;
    }
    HIP_OK(hipMalloc(&t.dev, info.size_bytes));
    HIP_OK(hipMemcpy(t.dev, info.data, info.size_bytes, hipMemcpyHostToDevice));
    state->allocations.push_back(t.dev);
    state->resident_bytes += info.size_bytes;
    state->tensors.emplace(std::string(info.name), t);
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
                              float eps, bool rope) {
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
  const float freq =
      expf(-static_cast<float>(i) * (logf(kDraftRopeTheta) / 64.0F));
  const float ang = static_cast<float>(pos_base + blockIdx.x) * freq;
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
  const int nctx = P - lo;
  const int total = nctx + kDraftSlots;
  const int chunk = (total + nsplit - 1) / nsplit;
  const int st = split * chunk;
  const int en = min(total, st + chunk);
  constexpr float scale = 0.08838834764831843F;
  float q4[kDraftSlots][4];
  float m[kDraftSlots], l[kDraftSlots], o4[kDraftSlots][4];
  for (int s = 0; s < kDraftSlots; ++s) {
    const float* qp =
        q + (static_cast<std::size_t>(s) * 24 + h) * 128 + lane * 4;
    for (int t = 0; t < 4; ++t) {
      q4[s][t] = qp[t] * scale;
      o4[s][t] = 0.0F;
    }
    m[s] = -3.0e38F;
    l[s] = 0.0F;
  }
  for (int j = st + warp; j < en; j += 4) {
    float k4[4], v4[4];
    if (j < nctx) {
      const std::size_t base =
          static_cast<std::size_t>((lo + j) % cap) * 1024 + g * 128 + lane * 4;
      for (int t = 0; t < 4; ++t) {
        k4[t] = __half2float(kc[base + t]);
        v4[t] = __half2float(vc[base + t]);
      }
    } else {
      const std::size_t base =
          (static_cast<std::size_t>(j - nctx) * 8 + g) * 128 + lane * 4;
      for (int t = 0; t < 4; ++t) {
        k4[t] = kb[base + t];
        v4[t] = vb[base + t];
      }
    }
    for (int s = 0; s < kDraftSlots; ++s) {
      float dot = q4[s][0] * k4[0] + q4[s][1] * k4[1] + q4[s][2] * k4[2] +
                  q4[s][3] * k4[3];
      for (int off = 16; off > 0; off >>= 1)
        dot += __shfl_xor(dot, off);
      const float mn = fmaxf(m[s], dot);
      const float a = expf(m[s] - mn);
      const float p = expf(dot - mn);
      l[s] = l[s] * a + p;
      for (int t = 0; t < 4; ++t)
        o4[s][t] = o4[s][t] * a + p * v4[t];
      m[s] = mn;
    }
  }
  for (int s = 0; s < kDraftSlots; ++s) {
    if (lane == 0) {
      sm[warp][s] = m[s];
      sl[warp][s] = l[s];
    }
    for (int t = 0; t < 4; ++t)
      so[warp][s][lane * 4 + t] = o4[s][t];
  }
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
                          const int* map = nullptr, float* conf = nullptr) {
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
  static constexpr int kDraftWarmChunk = 256;
  static constexpr int kDraftSplit = 64;
  static constexpr int kDraftShortlistMax = 65536;
  unsigned char* dhead{nullptr};
  int* dmap{nullptr};
  float* dconf{nullptr};
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
    if (bank == active)
      return;
    saved[active] = static_cast<Scratch&>(*this);
    static_cast<Scratch&>(*this) = saved[bank];
    active = bank;
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
  hipGraph_t graph{nullptr};
  hipGraphExec_t graph_exec{nullptr};

  float* Alloc(std::size_t floats) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, floats * sizeof(float)));
    allocated_bytes += floats * sizeof(float);
    allocs.push_back(p);
    return static_cast<float*>(p);
  }
  void* AllocBytes(std::size_t bytes) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, bytes));
    HIP_OK(hipMemset(p, 0, bytes));
    allocated_bytes += bytes;
    allocs.push_back(p);
    return p;
  }
  ~Impl() {
    if (graph_exec != nullptr)
      (void)hipGraphExecDestroy(graph_exec);
    if (graph != nullptr)
      (void)hipGraphDestroy(graph);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
    for (void* p : allocs)
      (void)hipFree(p);
  }

  void Dense(const Tensor& w, const float* xin, std::size_t ldx, float* yout,
             std::size_t ldy, int n, int rows = -1,
             const __half* xhin = nullptr) {
    const int nrows = rows < 0 ? static_cast<int>(w.rows) : rows;
    if (n == 1 && w.type == kTypeQ8 && ldx == w.cols && !no_dp4) {
      QuantRows<<<dim3(1, w.cols / 32), 32, 0, stream>>>(
          xin, xq, xqs, static_cast<int>(w.cols));
      const int rpb8 = kThreads * 2 / 32;
      GemmDp4<kTypeQ8>
          <<<dim3((nrows + rpb8 - 1) / rpb8, 1), kThreads * 2, 0, stream>>>(
              static_cast<const unsigned char*>(w.dev), w.row_bytes,
              static_cast<int>(w.cols), nrows, nullptr, nullptr, nullptr, xq,
              xqs, yout);
      return;
    }
    if (n >= 2 && n <= 8 && w.type == kTypeQ8 && ldx == w.cols && !no_dp4) {
      QuantRows<<<dim3(n, w.cols / 32), 32, 0, stream>>>(
          xin, xq, xqs, static_cast<int>(w.cols));
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
          case kTypeF32:
            WmmaGemm<kTypeF32><<<wg, kWmmaThreads, 0, stream>>>(wa);
            return;
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
            yout);
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
    Rms<<<rows, 256, 0, stream>>>(in_, w, out, dim, ld_in, ld_out,
                                  m->config.rms_eps, out_h);
  }
  void Mixer(const std::vector<const float*>& blocks, const float* cur_,
             const float* query, int nq, int n, float* out) {
    StatePtrs sp{};
    sp.count = static_cast<int>(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
      sp.p[i] = blocks[i];
    if (n >= kWmmaMinTokens) {
      MixBig<<<n, 256, 0, stream>>>(
          sp, cur_, query, nq, static_cast<int>(m->config.hidden_size), n, out);
    } else {
      Mix<<<n, 256, 0, stream>>>(
          sp, cur_, query, nq, static_cast<int>(m->config.hidden_size), n, out);
    }
  }
  void MixRmsLaunch(const std::vector<const float*>& blocks, const float* cur_,
                    const float* query, int nq, int n, const NormPtrs& np) {
    StatePtrs sp{};
    sp.count = static_cast<int>(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
      sp.p[i] = blocks[i];
    const int hh = static_cast<int>(m->config.hidden_size);
    MixRms<<<n, 256, nq * hh * sizeof(float), stream>>>(
        sp, cur_, query, nq, hh, n, np, m->config.rms_eps);
  }
  void ControlStep(float* cur_, const float* message, std::uint32_t phys,
                   const char* group, int n) {
    const int h = static_cast<int>(m->config.hidden_size);
    const bool big = n >= kWmmaMinTokens;
    if (!big && m->config.loop_gate_rank == kControlRank) {
      const std::string fb =
          "blk." + std::to_string(phys) + ".loop_" + group + "_";
      ControlLow<<<dim3(n, kControlRank / kControlWaves), kControlWaves * 32, 0,
                   stream>>>(
          cur_, message,
          static_cast<const float*>(m->T(fb + "down.weight").dev), h,
          m->config.rms_eps, low);
      ControlApply<<<dim3(n, (h + 255) / 256), 256, 0, stream>>>(
          cur_, message, low,
          static_cast<const float*>(m->T(fb + "gain.weight").dev),
          static_cast<const float*>(m->T(fb + "forget.weight").dev), h);
      return;
    }
    RmsRows(cur_, nullptr, cat, n, h, h, 2 * h, big ? cath : nullptr);
    RmsRows(message, nullptr, cat + h, n, h, h, 2 * h,
            big ? cath + h : nullptr);
    const std::string base =
        "blk." + std::to_string(phys) + ".loop_" + group + "_";
    Dense(m->T(base + "down.weight"), cat, 2 * h, low, m->config.loop_gate_rank,
          n, -1, big ? cath : nullptr);
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
      npu.Submit(stream, npu_a[0], b.data(), c.data(),
                 static_cast<int>(b.size()), prog);
    if (!k_shared && (npu_mask & 4) != 0)
      npu.Submit(stream, npu_a[1], &wk[0], &npu_c[6], 1, prog);
    if (!v_shared && (npu_mask & 8) != 0)
      npu.Submit(stream, npu_a[2], &wv[0], &npu_c[7], 1, prog);
    L.ticket = npu.Ticket();
  }

  void NpuQkvEnd(LayerCtx& L, int r0) {
    namespace rocm = gufo::models::qwen38_flash_next::rocm;
    auto& npu = *m->npu;
    const int rows = std::min<int>(m->npu_tokens, L.n - r0);
    const auto& wq = m->npu_w.at(m->P(L.p, "attn_q.weight"));
    const auto& wg = m->npu_w.at(m->P(L.p, "attn_gate.weight"));
    const int qd = static_cast<int>(wq.size()) * static_cast<int>(kNpuSlice);
    npu.Wait(stream, L.ticket);
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
    if (!L.npu_on || (npu_mask & 1) == 0)
      Dense(m->T(m->P(p, "attn_q.weight")), inp[0], h, q, nh * hd, n, -1,
            inh[0]);
    if (!L.npu_on || (npu_mask & 4) == 0)
      Dense(m->T(m->P(p, "attn_k.weight")), inp[1], h, tmp, nkv * hd, n, -1,
            inh[1]);
    if (!L.npu_on || (npu_mask & 8) == 0)
      Dense(m->T(m->P(p, "attn_v.weight")), inp[2], h, v, nkv * hd, n, -1,
            inh[2]);
    Dense(m->T(m->P(p, "attn_rel.weight")), inp[0], h, r, nrel * nh, n, -1,
          inh[0]);
    if (!swa) {
      Dense(m->T(m->P(p, "attn_k_conv_proj.weight")), inp[1], h, dyn_k,
            kConvTaps * nkv, n, -1, inh[1]);
      Dense(m->T(m->P(p, "attn_v_conv_proj.weight")), inp[2], h, dyn_v,
            kConvTaps * nkv, n, -1, inh[2]);
      L.dk = dyn_k;
      L.dv = dyn_v;
    }
    Dense(m->T(m->P(p, "attn_sink_proj.weight")), inp[0], h, sink, nh, n, -1,
          inh[0]);
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
    if (!L.npu_on || (npu_mask & 2) == 0)
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
    Convolve(tmp, m->T(m->P(p, "attn_k_conv.weight")), L.dk, nkv, sl.hist_k, k,
             nkv * hd, n);
    float* vtmp = tmp;
    L.vtmp = vtmp;
    Convolve(v, m->T(m->P(p, "attn_v_conv.weight")), L.dv, nkv, sl.hist_v, vtmp,
             nkv * hd, n);
    const int* tau_p = swa ? nullptr : d_pos;
    const float tau_floor = static_cast<float>(c.log_scale_floor);
    Clip<<<n * nh, 128, 0, stream>>>(
        q, static_cast<const float*>(m->T(m->P(p, "attn_q_clip.weight")).dev),
        nh, hd, c.qk_clip, tau_p, nh, tau_floor, c.log_scale_alpha);
    Clip<<<n * nkv, 128, 0, stream>>>(k, nullptr, 1, hd, c.qk_clip);
    RmsRows(vtmp, nullptr, vtmp, n * nkv, hd, hd, hd);
    Clip<<<n * nh, 128, 0, stream>>>(
        r, static_cast<const float*>(m->T(m->P(p, "attn_rel_clip.weight")).dev),
        nh, nrel, c.relative_clip, tau_p, nh, tau_floor, c.log_scale_alpha);
    StoreKv<<<n, 256, 0, stream>>>(k, vtmp, sl.k, sl.v, nkv * hd, d_pos,
                                   sl.ring, sl.vblocks);
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
        FlashRel<<<dim3(1, nkv, pk_split), kFlashThreads, 0, stream>>>(
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
        SelectBlocks<<<dim3(ntile, nh), 256, kMaxSparseBlocks * sizeof(float),
                       stream>>>(q, sl.kmm, d_pos, n, nh, nkv, blist, bcount);
        const int nsp = n < kWmmaMinTokens ? 32 : 1;
        FlashRel<<<dim3((n + kFlashQ - 1) / kFlashQ, nh, nsp), kFlashThreads, 0,
                   stream>>>(aa, q, r, sink, sl.k, sl.v, bias_w, att, n, nsp,
                             kSplitKeys, split_m, split_l, split_o, blist,
                             bcount, kListCap);
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
        FlashRel<<<dim3((n + kFlashQ - 1) / kFlashQ, nh, nsplit), kFlashThreads,
                   0, stream>>>(aa, q, r, sink, sl.k, sl.v, bias_w, att, n,
                                nsplit, split_keys, split_m, split_l, split_o,
                                nullptr, nullptr, 0);
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
      npu.Submit(stream, npu_ao, wo.data(), npu_c, static_cast<int>(wo.size()),
                 prog);
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
      npu.Wait(stream, L.ticket);
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
      HIP_OK(hipMemcpyAsync(b, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
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
    Moe(p, n, big ? xhx : nullptr);
    Convolve(moe, m->T(m->P(p, "ffn_post_conv.weight")), nullptr, 1,
             sl.hist_ffn, cur, h, n, ffn_inp);
  }

  // returns layer output in `cur`
  void Layer(std::uint32_t slot, std::uint32_t ip,
             std::vector<const float*>& blocks, const float* msgs_, int n) {
    LayerCtx L;
    BeginLayer(L, slot, ip, msgs_, n);
    LayerFront(L, blocks);
    LayerQkvBegin(L);
    LayerQkvEnd(L);
    LayerAttn(L);
    LayerOut1a(L);
    LayerOut1b(L, blocks);
    LayerOut2(L, blocks);
  }

  void Moe(std::uint32_t p, int n, const __half* xhx_in = nullptr) {
    const Config& c = m->config;
    const int h = c.hidden_size, ne = c.expert_count, nu = c.expert_used,
              ff = c.expert_ff;
    Dense(m->T(m->P(p, "ffn_gate_inp.weight")), x, h, router, ne, n, -1,
          xhx_in);
    Route<<<n, 256, 0, stream>>>(
        router,
        static_cast<const float*>(m->T(m->P(p, "exp_probs_b.bias")).dev), ne,
        nu, c.expert_weights_scale, ids, wts);
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
      const int bn = n >= 2048 ? 192 : (n >= 512 ? 128 : 48);
      int ntiles = 0;
      const int* tmap = TileMap(n, bn, &ntiles);
      const bool ok1 = qfn::RoutedNvfp4GemmF16(wgu.dev, xmoe, tmap, ntiles, bn,
                                               rbounds, rrows_token, rrows_slot,
                                               gu16, 2 * ff, h, s_gu, stream);
      SituHalfIn<<<items, 256, 0, stream>>>(gu16, act16, ff, c.situ_beta,
                                            c.situ_linear_beta);
      const bool ok2 = qfn::RoutedNvfp4GemmF16(wdn.dev, act16, tmap, ntiles, bn,
                                               rbounds, rrows_slot, rrows_slot,
                                               y16, h, ff, s_dn, stream);
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
      Combine<<<n, 256, 0, stream>>>(y, wts, moe, nu, h, nullptr, nullptr);
      return;
    }
    xrow = xrow_token;
    Routed(m->T(m->P(p, "ffn_gate_up_exps.weight")), 2 * ff, items, nullptr, x,
           gu);
    Situ<<<items, 256, 0, stream>>>(gu, ids, s_gu, act, ff, c.situ_beta,
                                    c.situ_linear_beta);
    xrow = xrow_item;
    Routed(m->T(m->P(p, "ffn_down_exps.weight")), h, items, s_dn, act, y);
    Combine<<<n, 256, 0, stream>>>(y, wts, moe, nu, h, nullptr, nullptr);
  }

  void Forward(std::span<const std::int32_t> tokens) {
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
    const int rows = all_rows ? n : 1;
    ArgMaxRow<<<rows, 256, 0, stream>>>(
        d_logits, static_cast<int>(c.vocab_unpadded), d_argmax, d_argmax);
    HIP_OK(hipMemcpyAsync(argmax_host, d_argmax, rows * 4,
                          hipMemcpyDeviceToHost, stream));
    HIP_OK(hipStreamSynchronize(stream));
    logits_rows = rows;
    logits_last = rows - 1;
    logits_have.assign(static_cast<std::size_t>(rows), 0);
    n_logit_rows = rows;
    position += n;
  }

  const float* DW(const std::string& n) const {
    return static_cast<const float*>(m->T("d." + n).dev);
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
    {
      const Tensor& head = m->T("output.weight");
      const int n = std::min<int>(kDraftShortlist,
                                  static_cast<int>(m->draft_rank.size()));
      kDraftShortlist = n;
      dmap = static_cast<int*>(AllocBytes(static_cast<std::size_t>(n) * 4));
      dhead = static_cast<unsigned char*>(
          AllocBytes(static_cast<std::size_t>(n) * head.row_bytes));
      HIP_OK(hipMemcpy(dmap, m->draft_rank.data(),
                       static_cast<std::size_t>(n) * 4, hipMemcpyHostToDevice));
      GatherRows<<<n, 256, 0, stream>>>(
          static_cast<const unsigned char*>(head.dev), head.row_bytes, dmap,
          dhead);
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
             float* conf_out = nullptr) {
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
      Tensor sub = head;
      sub.dev = dhead;
      sub.rows = static_cast<std::size_t>(kDraftShortlist);
      Dense(sub, deff, h, dlog, kDraftShortlist, 1, kDraftShortlist);
      ArgMaxRow<<<1, 256, 0, stream>>>(dlog, kDraftShortlist, dprev,
                                       dout + (s - 1), dmap, dconf + (s - 1));
    }
    HIP_OK(hipMemcpyAsync(out, dout, steps * 4, hipMemcpyDeviceToHost, stream));
    if (conf_out != nullptr)
      HIP_OK(hipMemcpyAsync(conf_out, dconf, steps * 4, hipMemcpyDeviceToHost,
                            stream));
    HIP_OK(hipStreamSynchronize(stream));
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
      const float* res =
          static_cast<const float*>(m->T("loop_res_query.weight").dev);
      const float* kv =
          static_cast<const float*>(m->T("loop_kv_res_query.weight").dev);
      float* mq = low;
      HIP_OK(hipMemcpyAsync(mq, res + 2 * cc * h, h * 4,
                            hipMemcpyDeviceToDevice, stream));
      HIP_OK(hipMemcpyAsync(mq + h, kv + 2 * cc * h, h * 4,
                            hipMemcpyDeviceToDevice, stream));
      HIP_OK(hipMemcpyAsync(mq + 2 * h, kv + (2 * cc + 1) * h, h * 4,
                            hipMemcpyDeviceToDevice, stream));
      HIP_OK(hipMemcpyAsync(mq + 3 * h, res + (2 * cc + 1) * h, h * 4,
                            hipMemcpyDeviceToDevice, stream));
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
      BankFront(1, R[1], steps[t]);
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
    Activate(1);
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
        const float* res =
            static_cast<const float*>(m->T("loop_res_query.weight").dev);
        const float* kv =
            static_cast<const float*>(m->T("loop_kv_res_query.weight").dev);
        float* mq = low;  // reuse as scratch: 4*h floats
        HIP_OK(hipMemcpyAsync(mq, res + 2 * cc * h, h * 4,
                              hipMemcpyDeviceToDevice, stream));
        HIP_OK(hipMemcpyAsync(mq + h, kv + 2 * cc * h, h * 4,
                              hipMemcpyDeviceToDevice, stream));
        HIP_OK(hipMemcpyAsync(mq + 2 * h, kv + (2 * cc + 1) * h, h * 4,
                              hipMemcpyDeviceToDevice, stream));
        HIP_OK(hipMemcpyAsync(mq + 3 * h, res + (2 * cc + 1) * h, h * 4,
                              hipMemcpyDeviceToDevice, stream));
        Mixer(bank, bank_out, mq, 4, n, msgs);
        Layer(il, ip, blocks, msgs, n);
        Tap(v++, n);
      }
    }
    for (std::uint32_t il = end; il < m->n_phys; ++il) {
      Layer(il, il, blocks, nullptr, n);
      Tap(v++, n);
    }
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
  i.logits_last = static_cast<int>(keep) - 1;
  i.last_rows = static_cast<int>(keep);
}

bool Session::Draft(std::int32_t anchor, std::span<std::int32_t> out) {
  return impl_->Draft(anchor, out.data(), static_cast<int>(out.size()));
}

bool Session::Draft(std::int32_t anchor, std::span<std::int32_t> out,
                    std::span<float> confidence) {
  return impl_->Draft(anchor, out.data(), static_cast<int>(out.size()),
                      confidence.data());
}

void Session::Rewind(std::uint32_t position) noexcept {
  impl_->position = position;
  impl_->dwarmed = std::min(impl_->dwarmed, position);
}
std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

void Session::Reset() noexcept {
  impl_->position = 0;
  impl_->dwarmed = 0;
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
    const std::string name = "d." + std::string(info.name);
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
    {
      const std::size_t rows =
          gufo::models::qwen38_flash_next::rocm::RoutedCompactRows(
              b * c.expert_used, c.expert_count);
      sc.rrows_token = static_cast<int*>(i.AllocBytes(rows * sizeof(int)));
      sc.rrows_slot = static_cast<int*>(i.AllocBytes(rows * sizeof(int)));
    }
    sc.xq = static_cast<signed char*>(
        i.AllocBytes(static_cast<std::size_t>(kWmmaMinTokens) * c.expert_used *
                     std::max<std::size_t>(c.hidden_size, 3072) * 2));
    sc.xqs = i.Alloc(static_cast<std::size_t>(kWmmaMinTokens) * c.expert_used *
                     std::max<std::size_t>(c.hidden_size, 3072) / 16);
    sc.blist = static_cast<int*>(
        i.AllocBytes(static_cast<std::size_t>(kMaxBatch / kFlashQ) *
                     c.num_heads * kListCap * 4));
    sc.bcount = static_cast<int*>(i.AllocBytes(
        static_cast<std::size_t>(kMaxBatch / kFlashQ) * c.num_heads * 4));
    {
      const std::size_t splits = (max_context + kSplitKeys - 1) / kSplitKeys;
      sc.split_m = i.Alloc(kWmmaMinTokens * c.num_heads * splits);
      sc.split_l = i.Alloc(kWmmaMinTokens * c.num_heads * splits);
      sc.split_o = i.Alloc(kWmmaMinTokens * c.num_heads * splits * kFlashHd);
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
  }
  return s;
}

}  // namespace gufo::models::inkling::gpu
