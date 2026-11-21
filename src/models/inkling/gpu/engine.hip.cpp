#include "src/models/inkling/gpu/engine.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::inkling::gpu {
namespace {

using gufo::core::GgmlType;

constexpr int kThreads = 128;
constexpr int kTokenTile = 8;
constexpr int kMaxStates = 8;
constexpr int kConvTaps = 4;
constexpr int kHistory = 3;
constexpr int kMaxBatch = 512;
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
      const unsigned int* qp = reinterpret_cast<const unsigned int*>(blk + 4 + s * 8);
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

// Y[t][r] = W[r] . X[t]; one wave per output row, kTokenTile tokens per launch row.
template <int TY>
__global__ void GemmDense(const unsigned char* w, std::size_t row_bytes, int cols, int rows, const float* x, std::size_t ldx,
                          float* y, std::size_t ldy, int n) {
  const int lane = threadIdx.x % warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (r >= rows) return;
  const int t0 = blockIdx.y * kTokenTile;
  const int tn = min(kTokenTile, n - t0);
  const unsigned char* row = w + static_cast<std::size_t>(r) * row_bytes;
  float acc[kTokenTile] = {};
  float wv[32];
  for (int c = lane; c < cols / 32; c += warpSize) {
    Load32<TY>(row, c, wv);
    for (int t = 0; t < tn; ++t) {
      const float4* x4 = reinterpret_cast<const float4*>(x + (t0 + t) * ldx + c * 32);
      float a = 0.0F;
      for (int k = 0; k < 8; ++k) {
        const float4 xv = x4[k];
        a += wv[4 * k] * xv.x + wv[4 * k + 1] * xv.y + wv[4 * k + 2] * xv.z + wv[4 * k + 3] * xv.w;
      }
      acc[t] += a;
    }
  }
  for (int t = 0; t < tn; ++t) {
    float a = acc[t];
    for (int off = warpSize / 2; off > 0; off >>= 1) a += __shfl_down(a, off);
    if (lane == 0) y[(t0 + t) * ldy + r] = a;
  }
}

constexpr int kRoutedWaves = 8;

// Item i: Y[i][r] = scale[g] * W[g*rows + r] . X[xrow[i]]; one wave per output row.
template <int TY>
__global__ void GemmRouted(const unsigned char* w, std::size_t row_bytes, int cols, int rows, const int* group, const int* xrow,
                           const float* scale, const float* x, float* y) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int r = blockIdx.x * (blockDim.x / warpSize) + wave;
  if (r >= rows) return;
  const int i = blockIdx.y;
  const int g = group[i];
  const unsigned char* row = w + (static_cast<std::size_t>(g) * rows + r) * row_bytes;
  const float* xp = x + static_cast<std::size_t>(xrow[i]) * cols;
  float wv[32];
  float acc = 0.0F;
  for (int c = lane; c < cols / 32; c += warpSize) {
    Load32<TY>(row, c, wv);
    const float4* x4 = reinterpret_cast<const float4*>(xp + c * 32);
    for (int k = 0; k < 8; ++k) {
      const float4 xv = x4[k];
      acc += wv[4 * k] * xv.x + wv[4 * k + 1] * xv.y + wv[4 * k + 2] * xv.z + wv[4 * k + 3] * xv.w;
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) acc += __shfl_down(acc, off);
  if (lane == 0) y[static_cast<std::size_t>(i) * rows + r] = acc * (scale != nullptr ? scale[g] : 1.0F);
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

__global__ void TauScale(float* x, int per_token, const int* pos0, float floor_v,
                         float alpha) {
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

struct StatePtrs {
  const float* p[kMaxStates];
  int count;
};

// states = rms(blocks..., cur); out[q][t] = softmax(states . query[q] /
// sqrt(h)) mixture of normalized states
__global__ void Mix(StatePtrs blocks, const float* cur, const float* query,
                    int nq, int h, int n, float* out) {
  __shared__ float red[256];
  __shared__ float inv[kMaxStates];
  __shared__ float score[4][kMaxStates];
  const int t = blockIdx.x;
  const int ns = blocks.count + 1;
  const float* st[kMaxStates];
  for (int s = 0; s < blocks.count; ++s)
    st[s] = blocks.p[s] + static_cast<std::size_t>(t) * h;
  st[blocks.count] = cur + static_cast<std::size_t>(t) * h;
  for (int s = 0; s < ns; ++s) {
    float a = 0.0F;
    for (int i = threadIdx.x; i < h; i += blockDim.x)
      a += st[s][i] * st[s][i];
    a = BlockSum(a, red);
    if (threadIdx.x == 0)
      inv[s] = rsqrtf(a / static_cast<float>(h) + kMixEps);
    __syncthreads();
  }
  for (int q = 0; q < nq; ++q) {
    for (int s = 0; s < ns; ++s) {
      float a = 0.0F;
      for (int i = threadIdx.x; i < h; i += blockDim.x)
        a += st[s][i] * query[q * h + i];
      a = BlockSum(a, red);
      if (threadIdx.x == 0)
        score[q][s] = a * inv[s] * rsqrtf(static_cast<float>(h));
      __syncthreads();
    }
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
    __syncthreads();
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
                        int dim, const int* pos0) {
  const std::size_t dst = static_cast<std::size_t>(*pos0 + blockIdx.x) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    kc[dst + i] =
        __float2half(k[static_cast<std::size_t>(blockIdx.x) * dim + i]);
    vc[dst + i] =
        __float2half(v[static_cast<std::size_t>(blockIdx.x) * dim + i]);
  }
}

struct AttnArgs {
  int nh, nkv, hd, nrel, extent, window;
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
    const __half* kj = kc + static_cast<std::size_t>(j) * kvdim + g * a.hd;
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
      o += sc[j - first] *
           __half2float(vc[static_cast<std::size_t>(j) * kvdim + g * a.hd + d]);
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
    state->tensors.emplace(std::string(info.name), t);
  }
  auto model = std::shared_ptr<Model>(new Model());
  model->state_ = std::move(state);
  return model;
}

const Config& Model::config() const noexcept {
  return state_->config;
}

struct Session::Impl {
  std::shared_ptr<ModelState> m;
  std::uint32_t ctx{0};
  std::uint32_t position{0};
  struct Slot {
    __half* k{nullptr};
    __half* v{nullptr};
    float* hist_k{nullptr};
    float* hist_v{nullptr};
    float* hist_attn{nullptr};
    float* hist_ffn{nullptr};
  };
  std::vector<Slot> slots;
  std::vector<void*> allocs;
  std::vector<float> logits;
  // scratch
  int* d_tokens{nullptr};
  float *cur{nullptr}, *tmp{nullptr}, *inp_sa{nullptr}, *in[3]{}, *q{nullptr},
      *k{nullptr}, *v{nullptr}, *r{nullptr}, *dyn_k{nullptr}, *dyn_v{nullptr},
      *sink{nullptr}, *gate{nullptr}, *att{nullptr}, *o{nullptr},
      *mixed{nullptr}, *msgs{nullptr}, *ffn_inp{nullptr}, *x{nullptr},
      *cat{nullptr}, *low{nullptr}, *gain{nullptr}, *forget{nullptr},
      *router{nullptr}, *wts{nullptr}, *gu{nullptr}, *act{nullptr}, *y{nullptr},
      *moe{nullptr}, *prelude{nullptr}, *bank_out{nullptr}, *d_logits{nullptr};
  int *ids{nullptr}, *xrow{nullptr}, *xrow_token{nullptr}, *xrow_item{nullptr}, *d_pos{nullptr};
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  hipGraphExec_t graph_exec{nullptr};
  float* block_pool[16]{};
  int pool_used{0};

  float* Alloc(std::size_t floats) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, floats * sizeof(float)));
    allocs.push_back(p);
    return static_cast<float*>(p);
  }
  void* AllocBytes(std::size_t bytes) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, bytes));
    HIP_OK(hipMemset(p, 0, bytes));
    allocs.push_back(p);
    return p;
  }
  ~Impl() {
    if (graph_exec != nullptr) (void)hipGraphExecDestroy(graph_exec);
    if (graph != nullptr) (void)hipGraphDestroy(graph);
    if (stream != nullptr) (void)hipStreamDestroy(stream);
    for (void* p : allocs)
      (void)hipFree(p);
  }

  void Dense(const Tensor& w, const float* xin, std::size_t ldx, float* yout,
             std::size_t ldy, int n, int rows = -1) {
    const int nrows = rows < 0 ? static_cast<int>(w.rows) : rows;
    const int rpb = kThreads * 2 / 32;
    const dim3 grid((nrows + rpb - 1) / rpb, (n + kTokenTile - 1) / kTokenTile);
    const auto* wp = static_cast<const unsigned char*>(w.dev);
    switch (w.type) {
      case kTypeF32:
        GemmDense<kTypeF32><<<grid, kThreads * 2, 0, stream>>>(wp, w.row_bytes, w.cols, nrows,
                                                xin, ldx, yout, ldy, n);
        break;
      case kTypeBf16:
        GemmDense<kTypeBf16><<<grid, kThreads * 2, 0, stream>>>(wp, w.row_bytes, w.cols, nrows,
                                                 xin, ldx, yout, ldy, n);
        break;
      case kTypeQ8:
        GemmDense<kTypeQ8><<<grid, kThreads * 2, 0, stream>>>(wp, w.row_bytes, w.cols, nrows,
                                               xin, ldx, yout, ldy, n);
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
      GemmRouted<kTypeQ8><<<grid, kThreads * 2, 0, stream>>>(wp, w.row_bytes, w.cols, rows,
                                              ids_flat, xrow, scale, xin, yout);
    }
  }
  int* ids_flat{nullptr};
  void RmsRows(const float* in_, const float* w, float* out, int rows, int dim,
               std::size_t ld_in, std::size_t ld_out) {
    Rms<<<rows, 256, 0, stream>>>(in_, w, out, dim, ld_in, ld_out, m->config.rms_eps);
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
    Conv<<<(ch + 127) / 128, 128, 0, stream>>>(in_, static_cast<const float*>(w.dev), dyn,
                                    heads, hist, out, ch, n);
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
    HIP_OK(hipMemcpyAsync(inp_sa, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
    for (int j = 0; j < 3; ++j)
      HIP_OK(hipMemcpyAsync(in[j], cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
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
        HIP_OK(hipMemcpyAsync(in[1], in[0], nh_ * 4, hipMemcpyDeviceToDevice, stream));
        HIP_OK(hipMemcpyAsync(in[2], in[0], nh_ * 4, hipMemcpyDeviceToDevice, stream));
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
    StoreKv<<<n, 256, 0, stream>>>(k, v, sl.k, sl.v, nkv * hd, d_pos);
    AttnArgs aa{nh,
                nkv,
                hd,
                nrel,
                static_cast<int>(c.relative_extent),
                swa ? static_cast<int>(c.sliding_window) : 0,
                d_pos,
                1.0F / std::sqrt(static_cast<float>(hd))};
    const int span = swa ? static_cast<int>(c.sliding_window) : static_cast<int>(ctx);
    Attention<<<dim3(nh, n), kThreads, (span + 1) * sizeof(float), stream>>>(
        aa, q, r, sink, sl.k, sl.v,
        static_cast<const float*>(m->T(m->P(p, "attn_rel_bias.weight")).dev),
        att);
    Dense(m->T(m->P(p, "attn_gate.weight")), in[0], h, gate, nh * hd, n);
    SigmoidMul<<<(n * nh * hd + 255) / 256, 256, 0, stream>>>(
        att, gate, static_cast<std::size_t>(n) * nh * hd);
    Dense(m->T(m->P(p, "attn_output.weight")), att, nh * hd, o, h, n);
    Convolve(o, m->T(m->P(p, "attn_post_conv.weight")), nullptr, 1,
             sl.hist_attn, cur, h, n);
    if (ip % c.attn_res_block_size == 0) {
      float* b = Pool();
      HIP_OK(hipMemcpyAsync(b, inp_sa, nh_ * 4, hipMemcpyDeviceToDevice, stream));
      blocks.push_back(b);
    } else {
      AddInto<<<(n * h + 255) / 256, 256, 0, stream>>>(cur, inp_sa, nh_);
    }
    HIP_OK(hipMemcpyAsync(ffn_inp, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
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
    HIP_OK(hipMemcpy(logits.data(), d_logits, c.vocab_unpadded * 4, hipMemcpyDeviceToHost));
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
    HIP_OK(hipMemcpyAsync(prelude, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
    const std::vector<const float*> prelude_blocks = blocks;
    for (std::uint32_t pass = 0; pass < c.loop_num_loops; ++pass) {
      std::vector<const float*> bank;
      if (pass > 0) {
        bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
        HIP_OK(hipMemcpyAsync(bank_out, cur, nh_ * 4, hipMemcpyDeviceToDevice, stream));
        HIP_OK(hipMemcpyAsync(cur, prelude, nh_ * 4, hipMemcpyDeviceToDevice, stream));
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
    Softcap<<<(c.vocab_unpadded + 255) / 256, 256, 0, stream>>>(d_logits, c.vocab_unpadded,
                                                     c.final_logit_softcapping);
  }
};

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
  i.slots.resize(c.num_layers);
  for (auto& sl : i.slots) {
    sl.k = static_cast<__half*>(
        i.AllocBytes(static_cast<std::size_t>(max_context) * kv * 2));
    sl.v = static_cast<__half*>(
        i.AllocBytes(static_cast<std::size_t>(max_context) * kv * 2));
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
  {
    std::vector<int> tok(b * c.expert_used);
    std::vector<int> item(b * c.expert_used);
    for (std::size_t j = 0; j < tok.size(); ++j) {
      tok[j] = static_cast<int>(j / c.expert_used);
      item[j] = static_cast<int>(j);
    }
    i.xrow_token = static_cast<int*>(i.AllocBytes(tok.size() * 4));
    i.xrow_item = static_cast<int*>(i.AllocBytes(item.size() * 4));
    HIP_OK(hipMemcpy(i.xrow_token, tok.data(), tok.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(i.xrow_item, item.data(), item.size() * 4, hipMemcpyHostToDevice));
  }
  HIP_OK(hipStreamCreateWithFlags(&i.stream, hipStreamNonBlocking));
  i.ids = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
  i.xrow = static_cast<int*>(i.AllocBytes(b * c.expert_used * 4));
  for (auto& p : i.block_pool)
    p = i.Alloc(b * h);
  return s;
}

}  // namespace gufo::models::inkling::gpu
