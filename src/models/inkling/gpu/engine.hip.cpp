#include "src/models/inkling/gpu/engine.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::inkling::gpu {
namespace {

using gufo::core::GgmlType;

constexpr int kThreads = 128;
constexpr int kTokenTile = 8;
constexpr int kMaxStates = 8;
constexpr int kConvTaps = 4;
constexpr int kHistory = 3;
constexpr int kMaxBatch = 2048;
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

__device__ float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

template<int TY>
__device__ void Load32(const unsigned char* row, int chunk, float* w) {
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
    const unsigned char* b = row + chunk * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const signed char* q = reinterpret_cast<const signed char*>(b + 2);
    for (int i = 0; i < 32; ++i)
      w[i] = d * static_cast<float>(q[i]);
  } else {
    const unsigned char* blk = row + (chunk >> 1) * 36;
    const int hf = chunk & 1;
    for (int s2 = 0; s2 < 2; ++s2) {
      const int s = hf * 2 + s2;
      const float d = Ue4m3(blk[s]);
      const unsigned int* qp =
          reinterpret_cast<const unsigned int*>(blk + 4 + s * 8);
      const unsigned int u[2] = {qp[0], qp[1]};
      for (int j = 0; j < 8; ++j) {
        const unsigned int byte = (u[j >> 2] >> ((j & 3) * 8)) & 0xFF;
        w[s2 * 16 + j] = kE2m1[byte & 15] * d;
        w[s2 * 16 + 8 + j] = kE2m1[byte >> 4] * d;
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
    Load32<TY>(row, c, wv);
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
    Load32<TY>(row, c, wv);
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
                    std::size_t ld_in, std::size_t ld_out, float eps) {
  __shared__ float red[256];
  const float* x = in + blockIdx.x * ld_in;
  float s = 0.0F;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    s += x[i] * x[i];
  s = BlockSum(s, red);
  const float inv = rsqrtf(s / static_cast<float>(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    out[blockIdx.x * ld_out + i] = x[i] * inv * (w != nullptr ? w[i] : 1.0F);
}

// per-row L2 clip after optional per-element scale indexed [(row % scale_rows)
// * dim + i]
__global__ void Clip(float* x, const float* scale, int scale_rows, int dim,
                     float threshold) {
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
  const float f = fminf(1.0F, threshold / (sqrtf(s) + 1e-6F));
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
                     int heads, float* hist, float* out, int ch, int n) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ch)
    return;
  const int per = dyn != nullptr ? ch / heads : ch;
  const float norm =
      rsqrtf(static_cast<float>((dyn != nullptr ? 2 : 1) * kConvTaps + 1));
  float win[kConvTaps];
  for (int k = 0; k < kHistory; ++k)
    win[k] = hist[c * kHistory + k];
  float wt[kConvTaps];
  for (int j = 0; j < kConvTaps; ++j)
    wt[j] = w[c * kConvTaps + j];
  for (int t = 0; t < n; ++t) {
    const float x = cur[static_cast<std::size_t>(t) * ch + c];
    win[kHistory] = x;
    float o = 0.0F;
    for (int j = 0; j < kConvTaps; ++j)
      o += wt[j] * win[j];
    if (dyn != nullptr) {
      const float* d = dyn + static_cast<std::size_t>(t) * heads * kConvTaps +
                       (c / per) * kConvTaps;
      for (int lag = 0; lag < kConvTaps; ++lag)
        o += d[lag] * win[kHistory - lag];
    }
    out[static_cast<std::size_t>(t) * ch + c] = (x + o) * norm;
    for (int k = 0; k < kHistory; ++k)
      win[k] = win[k + 1];
  }
  for (int k = 0; k < kHistory; ++k)
    hist[c * kHistory + k] = win[k];
}

__global__ void ConvT(const float* cur, const float* w, const float* dyn,
                      int heads, const float* hist, float* out, int ch) {
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
      rsqrtf(static_cast<float>((dyn != nullptr ? 2 : 1) * kConvTaps + 1));
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

struct StatePtrs {
  const float* p[kMaxStates];
  int count;
};

// states = rms(blocks..., cur); out[q][t] = softmax(states . query[q] /
// sqrt(h)) mixture of normalized states
__global__ void Mix(StatePtrs blocks, const float* cur, const float* query,
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

__global__ void Control(float* cur, const float* msg, const float* gain,
                        const float* forget, int h) {
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * h;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    const float c = cur[base + i];
    cur[base + i] =
        c - tanhf(forget[base + i]) * c + tanhf(gain[base + i]) * msg[base + i];
  }
}

__global__ void StoreKv(const float* k, const float* v, __half* kc, __half* vc,
                        int dim, const int* pos0, int ring) {
  const int row = *pos0 + blockIdx.x;
  const std::size_t dst =
      static_cast<std::size_t>(ring > 0 ? row % ring : row) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    kc[dst + i] =
        __float2half(k[static_cast<std::size_t>(blockIdx.x) * dim + i]);
    vc[dst + i] =
        __float2half(v[static_cast<std::size_t>(blockIdx.x) * dim + i]);
  }
}

struct AttnArgs {
  int nh, nkv, hd, nrel, extent, window, ring;
  const int* pos0;
  float scale;
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

__global__ void Combine(const float* y, const float* wts, float* out, int nu,
                        int h) {
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    float a = 0.0F;
    for (int k = 0; k < nu; ++k)
      a += wts[blockIdx.x * nu + k] *
           y[(static_cast<std::size_t>(blockIdx.x) * nu + k) * h + i];
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
__device__ void LoadB16(const unsigned char* row, int k0, __half* dst) {
  if constexpr (TY == kTypeF32) {
    const float* p = reinterpret_cast<const float*>(row) + k0;
    for (int i = 0; i < 16; ++i)
      dst[i] = __float2half(p[i]);
  } else if constexpr (TY == kTypeBf16) {
    const unsigned short* p = reinterpret_cast<const unsigned short*>(row) + k0;
    for (int i = 0; i < 16; ++i)
      dst[i] = __float2half(__uint_as_float(static_cast<unsigned>(p[i]) << 16));
  } else if constexpr (TY == kTypeQ8) {
    const unsigned char* b = row + (k0 >> 5) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const unsigned short* q =
        reinterpret_cast<const unsigned short*>(b + 2) + ((k0 & 31) >> 1);
    for (int i = 0; i < 8; ++i) {
      const unsigned short v = q[i];
      dst[2 * i] = __float2half(
          d * static_cast<float>(static_cast<signed char>(v & 0xFF)));
      dst[2 * i + 1] = __float2half(
          d * static_cast<float>(static_cast<signed char>(v >> 8)));
    }
  } else {
    const unsigned char* blk = row + (k0 >> 6) * 36;
    const int s = (k0 >> 4) & 3;
    const float d = Ue4m3(blk[s]);
    const unsigned int* qp =
        reinterpret_cast<const unsigned int*>(blk + 4 + s * 8);
    const unsigned int u[2] = {qp[0], qp[1]};
    for (int j = 0; j < 8; ++j) {
      const unsigned int byte = (u[j >> 2] >> ((j & 3) * 8)) & 0xFF;
      dst[j] = __float2half(kE2m1[byte & 15] * d);
      dst[8 + j] = __float2half(kE2m1[byte >> 4] * d);
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
    LoadB16<TY>(wr, k0 + lh * 16, &Bs[lr][lh * 16]);
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

constexpr int kFlashQ = 64;
constexpr int kFlashK = 64;
constexpr int kKStride = 136;
constexpr int kVStride = 72;
constexpr int kPStride = 72;
constexpr int kFlashHd = 128;
constexpr int kSplitKeys = 2048;
constexpr int kFlashRel = 16;
constexpr float kNegInf = -1e30F;

// Flash attention with relative bias and a zero sink logit. One block: 64
// queries of one head; 4 waves x 16 query rows.
__global__ void __launch_bounds__(128)
    FlashRel(AttnArgs a, const float* q, const float* r, const float* sink,
             const __half* kc, const __half* vc, const float* bias, float* out,
             int n, int nsplit, int split_keys, float* pm, float* pl, float* po,
             const int* blist, const int* bcount, int list_cap) {
  __shared__ __half Ks[kFlashK][kKStride];
  __shared__ __half Vt[kFlashHd][kVStride];
  __shared__ __half Ps[kFlashQ][kPStride];
  __shared__ float Rs[kFlashQ][kFlashRel];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * kFlashQ;
  const int g = h / (a.nh / a.nkv);
  const int kvdim = a.nkv * a.hd;
  const int pos0 = *a.pos0;
  const int qmax_pos = pos0 + min(q0 + kFlashQ, n) - 1;
  const int qmin_pos = pos0 + q0;
  for (int i = tid; i < kFlashQ * kFlashRel; i += 128) {
    const int row = i / kFlashRel;
    const int t = min(q0 + row, n - 1);
    Rs[row][i % kFlashRel] =
        r[(static_cast<std::size_t>(t) * a.nh + h) * kFlashRel + i % kFlashRel];
  }
  // Q fragments for this wave's 16 rows
  V16h qf[8];
  {
    const int t = min(q0 + wave * 16 + (lane & 15), n - 1);
    const float* qp = q + (static_cast<std::size_t>(t) * a.nh + h) * kFlashHd;
    for (int c = 0; c < 8; ++c) {
      for (int i = 0; i < 16; ++i)
        qf[c][i] = static_cast<_Float16>(qp[c * 16 + i]);
    }
  }
  float sinkv[8];
  int qpos[8];
  for (int i = 0; i < 8; ++i) {
    const int row = wave * 16 + 2 * i + (lane >> 4);
    const int t = min(q0 + row, n - 1);
    sinkv[i] = sink[static_cast<std::size_t>(t) * a.nh + h];
    qpos[i] = pos0 + q0 + row;
  }
  V8f o[8] = {};
  float m[8];
  float l[8];
  const int split = blockIdx.z;
  for (int i = 0; i < 8; ++i) {
    // the zero sink logit belongs to split 0 only (counted once after the 16-lane sum)
    m[i] = split == 0 ? 0.0F : kNegInf;
    l[i] = split == 0 ? 1.0F / 16.0F : 0.0F;
  }
  const int kfirst_pos = a.window > 0 ? max(0, qmin_pos - a.window + 1) : 0;
  int kb_begin = (kfirst_pos / kFlashK) * kFlashK;
  int it_begin = 0;
  int it_end = 0;
  const int* list = nullptr;
  if (blist != nullptr) {
    const int tile_head = blockIdx.x * a.nh + h;
    list = blist + static_cast<std::size_t>(tile_head) * list_cap;
    const int count = bcount[tile_head];
    const int per = (list_cap + nsplit - 1) / nsplit;
    it_begin = split * per;
    it_end = min(count, (split + 1) * per);
  } else {
    int kb_end = qmax_pos;
    if (nsplit > 1) {
      kb_begin = max(kb_begin, split * split_keys);
      kb_end = min(kb_end, (split + 1) * split_keys - 1);
    }
    it_end = kb_end >= kb_begin ? (kb_end - kb_begin) / kFlashK + 1 : 0;
  }
  for (int it = it_begin; it < it_end; ++it) {
    const int kb = list != nullptr ? list[it] * kFlashK : kb_begin + it * kFlashK;
    __syncthreads();
    {
      const int key = tid >> 1;
      const int half = tid & 1;
      const int j = kb + key;
      const bool ok = j <= qmax_pos;
      const std::size_t jr = a.ring > 0 ? j % a.ring : j;
      const uint4* ks = reinterpret_cast<const uint4*>(
          kc + jr * kvdim + g * kFlashHd + half * 64);
      const __half* vs = vc + jr * kvdim + g * kFlashHd + half * 64;
      uint4* kd = reinterpret_cast<uint4*>(&Ks[key][half * 64]);
      for (int i = 0; i < 8; ++i)
        kd[i] = ok ? ks[i] : make_uint4(0, 0, 0, 0);
      for (int i = 0; i < 64; ++i)
        Vt[half * 64 + i][key] = ok ? vs[i] : __float2half(0.0F);
    }
    __syncthreads();
    V8f sc[4] = {};
    for (int c = 0; c < 8; ++c) {
      for (int ni = 0; ni < 4; ++ni) {
        V16h bf;
        const __half* kp = &Ks[ni * 16 + (lane & 15)][c * 16];
        for (int i = 0; i < 16; ++i)
          bf[i] = static_cast<_Float16>(__half2float(kp[i]));
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
          float b = 0.0F;
          if (dist < a.extent) {
            const float* bp = bias + static_cast<std::size_t>(dist) * kFlashRel;
            const float* rp = Rs[wave * 16 + 2 * i + (lane >> 4)];
            for (int d = 0; d < kFlashRel; ++d)
              b += bp[d] * rp[d];
          }
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
        Ps[wave * 16 + 2 * i + (lane >> 4)][ni * 16 + (lane & 15)] =
            __float2half(pv);
      }
    }
    __syncthreads();
    for (int c = 0; c < 4; ++c) {
      V16h pf;
      const __half* pp = &Ps[wave * 16 + (lane & 15)][c * 16];
      for (int i = 0; i < 16; ++i)
        pf[i] = static_cast<_Float16>(__half2float(pp[i]));
      for (int ni = 0; ni < 8; ++ni) {
        V16h vf;
        const __half* vp = &Vt[ni * 16 + (lane & 15)][c * 16];
        for (int i = 0; i < 16; ++i)
          vf[i] = static_cast<_Float16>(__half2float(vp[i]));
        o[ni] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pf, vf, o[ni]);
      }
    }
  }
  for (int i = 0; i < 8; ++i) {
    float ls = l[i];
    for (int off = 8; off > 0; off >>= 1)
      ls += __shfl_xor(ls, off);
    const int t = q0 + wave * 16 + 2 * i + (lane >> 4);
    if (t >= n)
      continue;
    if (nsplit > 1) {
      const std::size_t part = (static_cast<std::size_t>(t) * a.nh + h) * nsplit + split;
      if ((lane & 15) == 0) {
        pm[part] = m[i];
        pl[part] = ls;
      }
      for (int ni = 0; ni < 8; ++ni) po[part * kFlashHd + ni * 16 + (lane & 15)] = o[ni][i];
      continue;
    }
    for (int ni = 0; ni < 8; ++ni) {
      out[(static_cast<std::size_t>(t) * a.nh + h) * kFlashHd + ni * 16 +
          (lane & 15)] = o[ni][i] / ls;
    }
  }
}


constexpr int kSparseSink = 2;
constexpr int kSparseLocal = 16;
constexpr int kSparseTop = 128;
constexpr int kListCap = kSparseSink + kSparseLocal + kSparseTop + 4;
constexpr int kBlockKeys = 64;
constexpr int kMaxSparseBlocks = 2048;

// pooled per-block key bounds: kmm[b][g][0|1][128] = min / max over the 64 keys
__global__ void PoolKeys(const __half* kc, __half* kmm, const int* pos0, int n, int nkv) {
  const int pos = *pos0;
  const int b = pos / kBlockKeys + blockIdx.x;
  if (b * kBlockKeys > pos + n - 1) return;
  const int g = blockIdx.y;
  const int d = threadIdx.x;
  const int total = pos + n;
  float mn = 1e30F;
  float mx = -1e30F;
  for (int j = b * kBlockKeys; j < min((b + 1) * kBlockKeys, total); ++j) {
    const float v = __half2float(kc[(static_cast<std::size_t>(j) * nkv + g) * kFlashHd + d]);
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  kmm[((static_cast<std::size_t>(b) * nkv + g) * 2) * kFlashHd + d] = __float2half(mn);
  kmm[((static_cast<std::size_t>(b) * nkv + g) * 2 + 1) * kFlashHd + d] = __float2half(mx);
}

// one block per (query tile of 64, head): sink + local + top-k older key blocks by the Quest bound of the tile's mean query
__global__ void SelectBlocks(const float* q, const __half* kmm, const int* pos0, int n, int nh, int nkv, int* blist,
                             int* bcount) {
  extern __shared__ float score[];
  __shared__ float qbar[kFlashHd];
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
    for (int t = 0; t < rows; ++t) a += q[(static_cast<std::size_t>(tile * kFlashQ + t) * nh + h) * kFlashHd + d];
    qbar[d] = a / rows;
  }
  if (threadIdx.x == 0) {
    taken = 0;
  }
  __syncthreads();
  auto push = [&](int blk) {
    const int slot = atomicAdd(&taken, 1);
    if (slot < kListCap) out[slot] = blk;
  };
  for (int b = threadIdx.x; b < min(kSparseSink, last_blk + 1); b += blockDim.x) push(b);
  if (ncand <= kSparseTop) {
    for (int i = threadIdx.x; i < ncand; i += blockDim.x) push(kSparseSink + i);
  } else {
    for (int i = threadIdx.x; i < ncand; i += blockDim.x) {
      const __half* mm = kmm + ((static_cast<std::size_t>(kSparseSink + i) * nkv + g) * 2) * kFlashHd;
      float sc = 0.0F;
      for (int d = 0; d < kFlashHd; ++d) {
        const float qd = qbar[d];
        sc += fmaxf(qd * __half2float(mm[d]), qd * __half2float(mm[kFlashHd + d]));
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
      if (threadIdx.x == 0) cnt = 0;
      __syncthreads();
      const float mid = 0.5F * (lo_t + hi_t);
      int local = 0;
      for (int i = threadIdx.x; i < ncand; i += blockDim.x) local += score[i] >= mid ? 1 : 0;
      atomicAdd(&cnt, local);
      __syncthreads();
      if (threadIdx.x == 0) {
        if (cnt > kSparseTop) lo_t = mid; else hi_t = mid;
      }
      __syncthreads();
    }
    // hi_t keeps at most kSparseTop candidates; take everything at or above it
    for (int i = threadIdx.x; i < ncand; i += blockDim.x) {
      if (score[i] >= hi_t) push(kSparseSink + i);
    }
  }
  for (int b = local_start + threadIdx.x; b <= last_blk; b += blockDim.x) push(b);
  __syncthreads();
  if (threadIdx.x == 0) bcount[tile * nh + h] = min(taken, kListCap);
}

__global__ void FlashCombine(const float* pm, const float* pl, const float* po, float* out, int nh, int nsplit) {
  const std::size_t base = (static_cast<std::size_t>(blockIdx.x) * nh + blockIdx.y) * nsplit;
  float mx = kNegInf;
  for (int s = 0; s < nsplit; ++s) mx = fmaxf(mx, pm[base + s]);
  float den = 0.0F;
  for (int s = 0; s < nsplit; ++s) den += expf(pm[base + s] - mx) * pl[base + s];
  for (int d = threadIdx.x; d < kFlashHd; d += blockDim.x) {
    float acc = 0.0F;
    for (int s = 0; s < nsplit; ++s) acc += expf(pm[base + s] - mx) * po[(base + s) * kFlashHd + d];
    out[(static_cast<std::size_t>(blockIdx.x) * nh + blockIdx.y) * kFlashHd + d] = acc / den;
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
  ~ModelState() {
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
    HIP_OK(hipMalloc(&t.dev, info.size_bytes));
    HIP_OK(hipMemcpy(t.dev, info.data, info.size_bytes, hipMemcpyHostToDevice));
    state->allocations.push_back(t.dev);
    state->resident_bytes += info.size_bytes;
    state->tensors.emplace(std::string(info.name), t);
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

struct Session::Impl {
  std::shared_ptr<ModelState> m;
  std::uint32_t ctx{0};
  std::uint32_t position{0};
  struct Slot {
    __half* k{nullptr};
    __half* v{nullptr};
    int ring{0};
    __half* kmm{nullptr};
    float* hist_k{nullptr};
    float* hist_v{nullptr};
    float* hist_attn{nullptr};
    float* hist_ffn{nullptr};
  };
  std::vector<Slot> slots;
  std::vector<void*> allocs;
  std::vector<float> logits;
  std::size_t allocated_bytes{0};
  // scratch
  int* d_tokens{nullptr};
  float *cur{nullptr}, *tmp{nullptr}, *inp_sa{nullptr}, *in[3]{}, *q{nullptr},
      *k{nullptr}, *v{nullptr}, *r{nullptr}, *dyn_k{nullptr}, *dyn_v{nullptr},
      *sink{nullptr}, *gate{nullptr}, *att{nullptr}, *o{nullptr},
      *mixed{nullptr}, *msgs{nullptr}, *ffn_inp{nullptr}, *x{nullptr},
      *cat{nullptr}, *low{nullptr}, *gain{nullptr}, *forget{nullptr},
      *router{nullptr}, *wts{nullptr}, *gu{nullptr}, *act{nullptr}, *y{nullptr},
      *moe{nullptr}, *prelude{nullptr}, *bank_out{nullptr}, *d_logits{nullptr};
  int *sorted{nullptr}, *tiles{nullptr};
  __half *xh{nullptr}, *act16{nullptr};
  float *split_m{nullptr}, *split_l{nullptr}, *split_o{nullptr};
  int *blist{nullptr}, *bcount{nullptr};
  bool sparse{false};
  int *ids{nullptr}, *xrow{nullptr}, *xrow_token{nullptr}, *xrow_item{nullptr},
      *d_pos{nullptr};
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  hipGraphExec_t graph_exec{nullptr};
  float* block_pool[16]{};
  int pool_used{0};

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
             std::size_t ldy, int n, int rows = -1) {
    const int nrows = rows < 0 ? static_cast<int>(w.rows) : rows;
    if (n >= kWmmaMinTokens && nrows % kTileN == 0 &&
        ldy == static_cast<std::size_t>(nrows) &&
        static_cast<std::uint64_t>(nrows) <= w.rows) {
      const std::size_t xcount = static_cast<std::size_t>(n) * w.cols;
      ToHalf<<<(xcount + 255) / 256, 256, 0, stream>>>(xin, xh, xcount);
      if (w.type == kTypeQ8 && ldx == w.cols && std::getenv("INKLING_SHARED_DENSE") != nullptr &&
          gufo::models::qwen38_flash_next::rocm::DenseF16Gemm(w.dev, xh, yout, n, nrows, w.cols, stream)) {
        return;
      }
      WmmaArgs wa{static_cast<const unsigned char*>(w.dev),
                  w.row_bytes,
                  static_cast<int>(w.cols),
                  nrows,
                  xh,
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
               std::size_t ld_in, std::size_t ld_out) {
    Rms<<<rows, 256, 0, stream>>>(in_, w, out, dim, ld_in, ld_out,
                                  m->config.rms_eps);
  }
  void Mixer(const std::vector<const float*>& blocks, const float* cur_,
             const float* query, int nq, int n, float* out) {
    StatePtrs sp{};
    sp.count = static_cast<int>(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
      sp.p[i] = blocks[i];
    Mix<<<n, 256, 0, stream>>>(sp, cur_, query, nq,
                               static_cast<int>(m->config.hidden_size), n, out);
  }
  void ControlStep(float* cur_, const float* message, std::uint32_t phys,
                   const char* group, int n) {
    const int h = static_cast<int>(m->config.hidden_size);
    RmsRows(cur_, nullptr, cat, n, h, h, 2 * h);
    RmsRows(message, nullptr, cat + h, n, h, h, 2 * h);
    const std::string base =
        "blk." + std::to_string(phys) + ".loop_" + group + "_";
    Dense(m->T(base + "down.weight"), cat, 2 * h, low, m->config.loop_gate_rank,
          n);
    Dense(m->T(base + "gain.weight"), low, m->config.loop_gate_rank, gain, h,
          n);
    Dense(m->T(base + "forget.weight"), low, m->config.loop_gate_rank, forget,
          h, n);
    Control<<<n, 256, 0, stream>>>(cur_, message, gain, forget, h);
  }
  void Convolve(const float* in_, const Tensor& w, const float* dyn, int heads,
                float* hist, float* out, int ch, int n) {
    if (n >= kWmmaMinTokens) {
      ConvT<<<dim3((ch + 127) / 128, n), 128, 0, stream>>>(
          in_, static_cast<const float*>(w.dev), dyn, heads, hist, out, ch);
      ConvHist<<<(ch + 127) / 128, 128, 0, stream>>>(in_, hist, ch, n);
      return;
    }
    Conv<<<(ch + 127) / 128, 128, 0, stream>>>(
        in_, static_cast<const float*>(w.dev), dyn, heads, hist, out, ch, n);
  }
  float* Pool() { return block_pool[pool_used++]; }

  // returns layer output in `cur`
  void Layer(std::uint32_t slot, std::uint32_t ip,
             std::vector<const float*>& blocks, const float* msgs_, int n) {
    const Config& c = m->config;
    const int h = c.hidden_size, nh = c.num_heads, nkv = c.num_kv,
              hd = c.head_dim, nrel = c.relative_length;
    const std::uint32_t p = m->Phys(slot);
    const bool swa = c.is_swa[slot];
    Slot& sl = slots[slot];
    const std::size_t nh_ = static_cast<std::size_t>(n) * h;
    HIP_OK(
        hipMemcpyAsync(inp_sa, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
    for (int j = 0; j < 3; ++j)
      HIP_OK(
          hipMemcpyAsync(in[j], cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
    if (m->Has(m->P(p, "attn_res_query.weight"))) {
      const int nq = swa ? 1 : 3;
      Mixer(
          blocks, cur,
          static_cast<const float*>(m->T(m->P(p, "attn_res_query.weight")).dev),
          nq, n, mixed);
      for (int j = 0; j < 3; ++j) {
        HIP_OK(hipMemcpyAsync(in[j], mixed + (swa ? 0 : j) * nh_, nh_ * 4,
                              hipMemcpyDeviceToDevice, stream));
      }
    }
    if (msgs_ != nullptr) {
      static const char* const kGroup[3] = {"attn", "k", "v"};
      for (int j = 0; j < 3; ++j)
        ControlStep(in[j], msgs_ + j * nh_, p, kGroup[j], n);
    }
    if (swa) {
      const float* w =
          static_cast<const float*>(m->T(m->P(p, "attn_norm.weight")).dev);
      if (msgs_ != nullptr) {
        for (int j = 0; j < 3; ++j)
          RmsRows(in[j], w, in[j], n, h, h, h);
      } else {
        RmsRows(in[0], w, in[0], n, h, h, h);
        HIP_OK(hipMemcpyAsync(in[1], in[0], nh_ * 4, hipMemcpyDeviceToDevice,
                              stream));
        HIP_OK(hipMemcpyAsync(in[2], in[0], nh_ * 4, hipMemcpyDeviceToDevice,
                              stream));
      }
    } else {
      static const char* const kN[3] = {"attn_q_inp_norm.weight",
                                        "attn_k_inp_norm.weight",
                                        "attn_v_inp_norm.weight"};
      for (int j = 0; j < 3; ++j)
        RmsRows(in[j], static_cast<const float*>(m->T(m->P(p, kN[j])).dev),
                in[j], n, h, h, h);
    }
    Dense(m->T(m->P(p, "attn_q.weight")), in[0], h, q, nh * hd, n);
    Dense(m->T(m->P(p, "attn_k.weight")), in[1], h, tmp, nkv * hd, n);
    Dense(m->T(m->P(p, "attn_v.weight")), in[2], h, v, nkv * hd, n);
    Dense(m->T(m->P(p, "attn_rel.weight")), in[0], h, r, nrel * nh, n);
    const float* dk = nullptr;
    const float* dv = nullptr;
    if (!swa) {
      Dense(m->T(m->P(p, "attn_k_conv_proj.weight")), in[1], h, dyn_k,
            kConvTaps * nkv, n);
      Dense(m->T(m->P(p, "attn_v_conv_proj.weight")), in[2], h, dyn_v,
            kConvTaps * nkv, n);
      dk = dyn_k;
      dv = dyn_v;
    }
    Convolve(tmp, m->T(m->P(p, "attn_k_conv.weight")), dk, nkv, sl.hist_k, k,
             nkv * hd, n);
    float* vtmp = tmp;
    Convolve(v, m->T(m->P(p, "attn_v_conv.weight")), dv, nkv, sl.hist_v, vtmp,
             nkv * hd, n);
    HIP_OK(hipMemcpyAsync(v, vtmp, static_cast<std::size_t>(n) * nkv * hd * 4,
                          hipMemcpyDeviceToDevice, stream));
    Clip<<<n * nh, 128, 0, stream>>>(
        q, static_cast<const float*>(m->T(m->P(p, "attn_q_clip.weight")).dev),
        nh, hd, c.qk_clip);
    Clip<<<n * nkv, 128, 0, stream>>>(k, nullptr, 1, hd, c.qk_clip);
    RmsRows(v, nullptr, v, n * nkv, hd, hd, hd);
    Clip<<<n * nh, 128, 0, stream>>>(
        r, static_cast<const float*>(m->T(m->P(p, "attn_rel_clip.weight")).dev),
        nh, nrel, c.relative_clip);
    if (!swa) {
      TauScale<<<n, 256, 0, stream>>>(q, nh * hd, d_pos,
                                      static_cast<float>(c.log_scale_floor),
                                      c.log_scale_alpha);
      TauScale<<<n, 256, 0, stream>>>(r, nh * nrel, d_pos,
                                      static_cast<float>(c.log_scale_floor),
                                      c.log_scale_alpha);
    }
    Dense(m->T(m->P(p, "attn_sink_proj.weight")), in[0], h, sink, nh, n);
    StoreKv<<<n, 256, 0, stream>>>(k, v, sl.k, sl.v, nkv * hd, d_pos, sl.ring);
    AttnArgs aa{nh,
                nkv,
                hd,
                nrel,
                static_cast<int>(c.relative_extent),
                swa ? static_cast<int>(c.sliding_window) : 0,
                sl.ring,
                d_pos,
                1.0F / std::sqrt(static_cast<float>(hd))};
    const int span =
        swa ? static_cast<int>(c.sliding_window) : static_cast<int>(ctx);
    if (hd == kFlashHd && nrel == kFlashRel) {
      const float* bias_w = static_cast<const float*>(m->T(m->P(p, "attn_rel_bias.weight")).dev);
      const bool sparse_now = sparse && !swa && (n < kWmmaMinTokens || position + n > (kSparseSink + kSparseLocal + kSparseTop + 2) * kBlockKeys);
      if (sparse && !swa) {
        PoolKeys<<<dim3((n + kBlockKeys - 1) / kBlockKeys + 1, nkv), kFlashHd, 0, stream>>>(sl.k, sl.kmm, d_pos, n, nkv);
      }
      if (sparse_now) {
        const int ntile = (n + kFlashQ - 1) / kFlashQ;
        SelectBlocks<<<dim3(ntile, nh), 256, kMaxSparseBlocks * sizeof(float), stream>>>(q, sl.kmm, d_pos, n, nh, nkv, blist, bcount);
        const int nsp = n < kWmmaMinTokens ? 8 : 1;
        FlashRel<<<dim3(ntile, nh, nsp), 128, 0, stream>>>(aa, q, r, sink, sl.k, sl.v, bias_w, att, n, nsp, kSplitKeys, split_m, split_l, split_o,
                                                         blist, bcount, kListCap);
        if (nsp > 1) FlashCombine<<<dim3(n, nh), 128, 0, stream>>>(split_m, split_l, split_o, att, nh, nsp);
      } else {
        const bool split = n < kWmmaMinTokens && !swa && ctx > kSplitKeys;
        const int nsplit = split ? static_cast<int>((ctx + kSplitKeys - 1) / kSplitKeys) : 1;
        FlashRel<<<dim3((n + kFlashQ - 1) / kFlashQ, nh, nsplit), 128, 0, stream>>>(
            aa, q, r, sink, sl.k, sl.v, bias_w, att, n, nsplit, kSplitKeys, split_m, split_l, split_o, nullptr, nullptr, 0);
        if (nsplit > 1) FlashCombine<<<dim3(n, nh), 128, 0, stream>>>(split_m, split_l, split_o, att, nh, nsplit);
      }
    } else {
      Attention<<<dim3(nh, n), kThreads, (span + 1) * sizeof(float), stream>>>(
          aa, q, r, sink, sl.k, sl.v,
          static_cast<const float*>(m->T(m->P(p, "attn_rel_bias.weight")).dev),
          att);
    }
    Dense(m->T(m->P(p, "attn_gate.weight")), in[0], h, gate, nh * hd, n);
    SigmoidMul<<<(n * nh * hd + 255) / 256, 256, 0, stream>>>(
        att, gate, static_cast<std::size_t>(n) * nh * hd);
    Dense(m->T(m->P(p, "attn_output.weight")), att, nh * hd, o, h, n);
    Convolve(o, m->T(m->P(p, "attn_post_conv.weight")), nullptr, 1,
             sl.hist_attn, cur, h, n);
    if (ip % c.attn_res_block_size == 0) {
      float* b = Pool();
      HIP_OK(
          hipMemcpyAsync(b, inp_sa, nh_ * 4, hipMemcpyDeviceToDevice, stream));
      blocks.push_back(b);
    } else {
      AddInto<<<(n * h + 255) / 256, 256, 0, stream>>>(cur, inp_sa, nh_);
    }
    HIP_OK(
        hipMemcpyAsync(ffn_inp, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
    Mixer(blocks, cur,
          static_cast<const float*>(m->T(m->P(p, "ffn_res_query.weight")).dev),
          1, n, x);
    if (msgs_ != nullptr)
      ControlStep(x, msgs_ + 3 * nh_, p, "mlp", n);
    RmsRows(x, static_cast<const float*>(m->T(m->P(p, "ffn_norm.weight")).dev),
            x, n, h, h, h);
    Moe(p, n);
    Convolve(moe, m->T(m->P(p, "ffn_post_conv.weight")), nullptr, 1,
             sl.hist_ffn, cur, h, n);
    AddInto<<<(n * h + 255) / 256, 256, 0, stream>>>(cur, ffn_inp, nh_);
  }

  void Moe(std::uint32_t p, int n) {
    const Config& c = m->config;
    const int h = c.hidden_size, ne = c.expert_count, nu = c.expert_used,
              ff = c.expert_ff;
    Dense(m->T(m->P(p, "ffn_gate_inp.weight")), x, h, router, ne, n);
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
      Combine<<<n, 256, 0, stream>>>(y, wts, moe, nu, h);
      return;
    }
    xrow = xrow_token;
    Routed(m->T(m->P(p, "ffn_gate_up_exps.weight")), 2 * ff, items, nullptr, x,
           gu);
    Situ<<<items, 256, 0, stream>>>(gu, ids, s_gu, act, ff, c.situ_beta,
                                    c.situ_linear_beta);
    xrow = xrow_item;
    Routed(m->T(m->P(p, "ffn_down_exps.weight")), h, items, s_dn, act, y);
    Combine<<<n, 256, 0, stream>>>(y, wts, moe, nu, h);
  }

  void Forward(std::span<const std::int32_t> tokens) {
    const Config& c = m->config;
    const int n = static_cast<int>(tokens.size());
    HIP_OK(hipMemcpy(d_tokens, tokens.data(), n * 4, hipMemcpyHostToDevice));
    const int pos = static_cast<int>(position);
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
    HIP_OK(hipStreamSynchronize(stream));
    logits.resize(c.vocab_unpadded);
    HIP_OK(hipMemcpy(logits.data(), d_logits, c.vocab_unpadded * 4,
                     hipMemcpyDeviceToHost));
    position += n;
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
    std::vector<const float*> blocks;
    const std::uint32_t start = c.loop_core_start, end = c.loop_core_end;
    for (std::uint32_t il = 0; il < start; ++il)
      Layer(il, il, blocks, nullptr, n);
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
      }
    }
    for (std::uint32_t il = end; il < m->n_phys; ++il)
      Layer(il, il, blocks, nullptr, n);
    Mixer(blocks, cur,
          static_cast<const float*>(m->T("output_res_query.weight").dev), 1, n,
          x);
    RmsRows(x, static_cast<const float*>(m->T("output_norm.weight").dev), x, n,
            h, h, h);
    const Tensor& out = m->T("output.weight");
    Dense(out, x + static_cast<std::size_t>(n - 1) * h, h, d_logits,
          c.vocab_unpadded, 1, static_cast<int>(c.vocab_unpadded));
    Softcap<<<(c.vocab_unpadded + 255) / 256, 256, 0, stream>>>(
        d_logits, c.vocab_unpadded, c.final_logit_softcapping);
  }
};

std::size_t Session::AllocatedBytes() const noexcept {
  return impl_->allocated_bytes;
}

Session::Session() = default;
Session::~Session() = default;

std::span<const float> Session::Logits() const noexcept {
  return impl_->logits;
}
std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

void Session::Reset() noexcept {
  impl_->position = 0;
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
  for (std::size_t off = 0; off < tokens.size(); off += kMaxBatch) {
    impl_->Forward(tokens.subspan(
        off, std::min<std::size_t>(kMaxBatch, tokens.size() - off)));
  }
  return true;
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
    i.sparse = e != nullptr && e[0] == '1' && max_context / kBlockKeys + 2 <= kMaxSparseBlocks + 2;
  }
  i.slots.resize(c.num_layers);
  for (auto& sl : i.slots) {
    const std::size_t slot_index = &sl - i.slots.data();
    const std::size_t ring_rows = c.sliding_window + kMaxBatch;
    const bool use_ring = c.is_swa[slot_index] && max_context > ring_rows;
    sl.ring = use_ring ? static_cast<int>(ring_rows) : 0;
    const std::size_t rows = use_ring ? ring_rows : max_context;
    sl.k = static_cast<__half*>(i.AllocBytes(rows * kv * 2));
    if (i.sparse && !c.is_swa[slot_index]) {
      sl.kmm = static_cast<__half*>(i.AllocBytes(static_cast<std::size_t>(max_context / kBlockKeys + 2) * c.num_kv * 2 * kFlashHd * 2));
    }
    sl.v = static_cast<__half*>(i.AllocBytes(rows * kv * 2));
    sl.hist_k = static_cast<float*>(i.AllocBytes(kv * kHistory * 4));
    sl.hist_v = static_cast<float*>(i.AllocBytes(kv * kHistory * 4));
    sl.hist_attn = static_cast<float*>(i.AllocBytes(h * kHistory * 4));
    sl.hist_ffn = static_cast<float*>(i.AllocBytes(h * kHistory * 4));
  }
  const std::size_t qd = static_cast<std::size_t>(c.num_heads) * c.head_dim;
  i.d_tokens = static_cast<int*>(i.AllocBytes(b * 4));
  i.cur = i.Alloc(b * h);
  i.tmp = i.Alloc(b * std::max<std::size_t>(kv, h));
  i.inp_sa = i.Alloc(b * h);
  for (auto& p : i.in)
    p = i.Alloc(b * h);
  i.q = i.Alloc(b * qd);
  i.k = i.Alloc(b * kv);
  i.v = i.Alloc(b * kv);
  i.r = i.Alloc(b * c.relative_length * c.num_heads);
  i.dyn_k = i.Alloc(b * kConvTaps * c.num_kv);
  i.dyn_v = i.Alloc(b * kConvTaps * c.num_kv);
  i.sink = i.Alloc(b * c.num_heads);
  i.gate = i.Alloc(b * qd);
  i.att = i.Alloc(b * qd);
  i.o = i.Alloc(b * h);
  i.mixed = i.Alloc(b * h * 3);
  i.msgs = i.Alloc(b * h * 4);
  i.ffn_inp = i.Alloc(b * h);
  i.x = i.Alloc(b * h);
  i.cat = i.Alloc(b * h * 2);
  i.low = i.Alloc(std::max<std::size_t>(b * c.loop_gate_rank, h * 4));
  i.gain = i.Alloc(b * h);
  i.forget = i.Alloc(b * h);
  i.router = i.Alloc(b * c.expert_count);
  i.wts = i.Alloc(b * c.expert_used);
  i.gu = i.Alloc(b * c.expert_used * 2 * c.expert_ff);
  i.act = i.Alloc(b * c.expert_used * c.expert_ff);
  i.y = i.Alloc(b * c.expert_used * h);
  i.moe = i.Alloc(b * h);
  i.prelude = i.Alloc(b * h);
  i.bank_out = i.Alloc(b * h);
  i.d_logits = i.Alloc(c.vocab_padded);
  i.d_pos = static_cast<int*>(i.AllocBytes(4));
  i.blist = static_cast<int*>(i.AllocBytes(static_cast<std::size_t>(kMaxBatch / kFlashQ) * c.num_heads * kListCap * 4));
  i.bcount = static_cast<int*>(i.AllocBytes(static_cast<std::size_t>(kMaxBatch / kFlashQ) * c.num_heads * 4));
  {
    const std::size_t splits = (max_context + kSplitKeys - 1) / kSplitKeys;
    i.split_m = i.Alloc(kWmmaMinTokens * c.num_heads * splits);
    i.split_l = i.Alloc(kWmmaMinTokens * c.num_heads * splits);
    i.split_o = i.Alloc(kWmmaMinTokens * c.num_heads * splits * kFlashHd);
  }
  i.xh = static_cast<__half*>(
      i.AllocBytes(b * std::max<std::size_t>(qd, 2 * h) * 2));
  i.act16 =
      static_cast<__half*>(i.AllocBytes(b * c.expert_used * c.expert_ff * 2));
  i.sorted = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
  i.tiles = static_cast<int*>(
      i.AllocBytes((b * c.expert_used / kTileM + c.expert_count) * 3 * 4));
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
  i.ids = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
  i.xrow = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
  for (auto& p : i.block_pool)
    p = i.Alloc(b * h);
  return s;
}

}  // namespace gufo::models::inkling::gpu
