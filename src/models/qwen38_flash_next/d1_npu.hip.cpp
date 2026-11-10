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
      D1DequantUnit<Kind>(
          table + static_cast<std::size_t>(id) * row_bytes +
              unit * (Kind == 1 ? 210u : Kind == 2 ? 34u : 18u),
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
    D1DequantUnit<Kind>(row + unit * (Kind == 1 ? 210u : Kind == 2 ? 34u : 18u), v);
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
/// C[t][c] * sum_j w[c][j] * (B*x)[t-(L-1)+j][c], zero history before the
/// token's sequence start.
__global__ void D1ConvKernel(const float* __restrict__ p,
                             const float* __restrict__ w,
                             const std::int32_t* __restrict__ seq_lo,
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
    const int s = static_cast<int>(t) - static_cast<int>(taps - 1) +
                  static_cast<int>(j);
    if (s < lo)
      continue;
    sum += w[static_cast<std::size_t>(c) * taps + j] * p[s * pitch + c] *
           p[s * pitch + 2ull * hidden + c];
  }
  out[static_cast<std::size_t>(t) * hidden + c] = sum * p[t * pitch + hidden + c];
}

__global__ void D1SplitQkvKernel(const float* qkv, float* q, float* k, float* v,
                                 std::uint32_t n, std::uint32_t q_dim,
                                 std::uint32_t kv_dim, std::uint32_t pitch) {
  const std::uint32_t t = blockIdx.x;
  if (t >= n)
    return;
  for (std::uint32_t i = threadIdx.x; i < q_dim + 2 * kv_dim;
       i += blockDim.x) {
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
__global__ void D1RopeKernel(float* q, float* k, const float* __restrict__ factors,
                             const std::int32_t* __restrict__ seq_lo,
                             std::uint32_t n, std::uint32_t heads,
                             std::uint32_t kv_heads, std::uint32_t hd,
                             float theta) {
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
  float angle = static_cast<float>(row - static_cast<std::uint32_t>(seq_lo[row])) *
                powf(theta, -2.0F * static_cast<float>(dd) /
                                static_cast<float>(hd));
  if (factors != nullptr)
    angle /= factors[dd];
  const float c = cosf(angle), sn = sinf(angle);
  const float a = base[dd], b = base[dd + hd / 2];
  base[dd] = a * c - b * sn;
  base[dd + hd / 2] = b * c + a * sn;
}

/// One block per (token, head). Tokens only see their own sequence
/// [seq_lo, seq_hi). window > 0: causal sliding layer. Otherwise a global
/// layer: every key of the sequence, restricted by `groups` (state tokens are
/// -1; a token sees keys that are state or in its own question).
__global__ void D1AttnKernel(const float* __restrict__ q,
                             const float* __restrict__ k,
                             const float* __restrict__ v,
                             const float* __restrict__ gate,
                             const std::int32_t* __restrict__ groups,
                             const std::int32_t* __restrict__ seq_lo,
                             const std::int32_t* __restrict__ seq_hi,
                             float* __restrict__ out, std::uint32_t n,
                             std::uint32_t heads, std::uint32_t kv_heads,
                             std::uint32_t hd, float scale,
                             std::uint32_t window) {
  const std::uint32_t t = blockIdx.x, h = blockIdx.y;
  const std::uint32_t kh = h / (heads / kv_heads);
  const std::uint32_t kv_dim = kv_heads * hd;
  __shared__ float qs[128];
  __shared__ float sc[M];
  __shared__ float red[kThreads];
  qs[threadIdx.x] = q[(static_cast<std::size_t>(t) * heads + h) * hd + threadIdx.x];
  __syncthreads();
  std::uint32_t first = static_cast<std::uint32_t>(seq_lo[t]);
  std::uint32_t last = static_cast<std::uint32_t>(seq_hi[t]) - 1;
  if (window > 0) {
    last = t;
    if (t + 1 > window && t + 1 - window > first)
      first = t + 1 - window;
  }
  float local_max = -INFINITY;
  for (std::uint32_t s = threadIdx.x; s < n; s += blockDim.x) {
    bool valid = s >= first && s <= last;
    if (valid && window == 0) {
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
    sc[s] = score;
    local_max = fmaxf(local_max, score);
  }
  red[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
    __syncthreads();
  }
  const float maximum = red[0];
  __syncthreads();
  float local_sum = 0.0F;
  for (std::uint32_t s = threadIdx.x; s < n; s += blockDim.x) {
    const float e = sc[s] == -INFINITY ? 0.0F : __expf(sc[s] - maximum);
    sc[s] = e;
    local_sum += e;
  }
  red[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv = 1.0F / red[0];
  const float g = 1.0F / (1.0F + __expf(-gate[static_cast<std::size_t>(t) * heads + h]));
  float acc = 0.0F;
  for (std::uint32_t s = first; s <= last; ++s) {
    const float p = sc[s];
    if (p != 0.0F)
      acc += p * v[static_cast<std::size_t>(s) * kv_dim + kh * hd + threadIdx.x];
  }
  out[(static_cast<std::size_t>(t) * heads + h) * hd + threadIdx.x] = acc * inv * g;
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
  float *x{nullptr}, *z{nullptr}, *z2{nullptr}, *branch{nullptr}, *p_buf{nullptr},
      *conv_out{nullptr}, *qkv_buf{nullptr}, *q{nullptr}, *k{nullptr},
      *v{nullptr}, *qn{nullptr}, *kn{nullptr}, *gate_buf{nullptr},
      *attn{nullptr}, *gu_buf{nullptr}, *act{nullptr}, *sel{nullptr},
      *sel_norm{nullptr}, *logit_buf{nullptr};
  __half* half_buf{nullptr};
  std::int32_t *ids{nullptr}, *group_buf{nullptr}, *vocab_ids{nullptr};
  std::int32_t *seq_lo_h{nullptr}, *seq_hi_h{nullptr};  // pinned, per token
  std::int32_t *seq_lo{nullptr}, *seq_hi{nullptr};      // device copies
  mutable std::int32_t vocab_capacity{0};

  mutable double t_encode{0}, t_gemm{0}, t_half{0}, t_memset{0}, t_enckernel{0}, t_final{0};
  bool prof_fine{false};
  bool sync_each{true};
  mutable int n_gemm{0};
  std::size_t hidden{0}, inter{0}, heads{0}, hd{0}, q_dim{0};
  std::uint32_t taps{0};

  static constexpr std::uint32_t kInPitch = 6144, kQkvPitch = 3072,
                                 kGuPitch = 15360;

  // Writes `rows` x cols of `src` (float) to a block-float A buffer.
  void EncodeA(const float* src, std::uint32_t n, std::uint32_t cols, int buf) const {
    const auto t0 = std::chrono::steady_clock::now();
    struct Done {
      const Impl* self;
      std::chrono::steady_clock::time_point t0;
      ~Done() {
        self->t_encode += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
      }
    } done{this, t0};
    const std::uint32_t enc_rows = (n + 127u) / 128u * 128u;
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
    hipMemsetAsync(half_buf + static_cast<std::size_t>(n) * cols, 0,
                   static_cast<std::size_t>(enc_rows - n) * cols * sizeof(__half),
                   stream);
    lap(t_memset, t);
    rocm::NpuEncodeActivations(half_buf, cols, enc_rows, cols, 0, cols,
                               npu->Device(buf), stream);
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
      std::fprintf(stderr, "[d1 npu ] act buf %d rows %u cols %u bytes %zu nonzero %zu\n", buf, enc_rows, cols, bytes, nz);
    }
  }

  bool Gemm(int program, int a, int weight, int c) const {
    const auto t0 = std::chrono::steady_clock::now();
    struct Done {
      const Impl* self;
      std::chrono::steady_clock::time_point t0;
      ~Done() {
        self->t_gemm += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
        ++self->n_gemm;
      }
    } done{this, t0};
    const int bs[1] = {weight};
    const int cs[1] = {c};
    if (!npu->Submit(stream, a, bs, cs, 1, program))
      return false;
    if (!npu->Wait(stream))
      return false;
    if (sync_each)
      hipStreamSynchronize(stream);
    return npu->Healthy();
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
const lfm3::Weights& D1NpuModel::weights() const { return impl_->w; }

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
    for (const auto* t :
         {&l.gate, &l.up, &l.down, &l.in_proj, &l.out_proj, &l.q, &l.k, &l.v,
          &l.attn_out}) {
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
  auto insts = [&](std::uint32_t k, std::uint32_t n) {
    return dir + "/runtime_mk/insts_" + std::to_string(M) + "x" +
           std::to_string(k) + "x" + std::to_string(n) + "_128x64x128.txt";
  };
  const std::pair<std::uint32_t, std::uint32_t> shapes[] = {
      {2048, 1024}, {2048, 2048}, {2048, 3072},
      {2048, 6144}, {2048, 15360}, {7680, 2048}};
  for (const auto& [k, n] : shapes)
    if (!std::ifstream(insts(k, n)).good())
      return fail("missing instruction file " + insts(k, n));

  rocm::NpuGemm::Config cfg;
  cfg.xclbin = dir + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
  cfg.insts = insts(2048, 1024);
  cfg.tokens = M;
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
  auto a_bytes = [](std::size_t cols) { return std::size_t{M} * cols / 8 * 9; };
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
  s.half_buf = DeviceAlloc<__half>(M * std::max<std::size_t>(I, std::max(H, Q)));
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
      !s.attn || !s.gu_buf || !s.act || !s.half_buf || !s.ids || !s.group_buf || !s.seq_lo_h || !s.seq_hi_h || !s.seq_lo || !s.seq_hi)
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
        lfm3::DecodeRow(lw.attn_gate, r,
                        std::span<float>(h_gate.data() + (l * s.heads + r) * H, H));
  }
  copy_f32(s.w.output_norm, h_out.data(), H);
  copy_f32(s.w.rope_freqs, h_factors.data(), s.hd / 2);
  hipMemcpy(s.g_op, h_op.data(), h_op.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_ffn, h_ffn.data(), h_ffn.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_post, h_post.data(), h_post.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_post_ffn, h_post_ffn.data(), h_post_ffn.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_qn, h_qn.data(), h_qn.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_kn, h_kn.data(), h_kn.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_out, h_out.data(), h_out.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_conv, h_conv.data(), h_conv.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_gate, h_gate.data(), h_gate.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(s.g_factors, h_factors.data(), h_factors.size() * 4, hipMemcpyHostToDevice);
  {
    const std::size_t bytes = s.w.embedding.bytes;
    s.emb_table = DeviceAlloc<std::uint8_t>(bytes);
    if (!s.emb_table ||
        hipMemcpy(s.emb_table, s.w.embedding.data, bytes,
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
      std::fprintf(stderr, "[d1 npu ] weight buf %d bytes %zu nonzero %zu\n", buf, tb, nz);
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
  if (n == 0 || n > M || slots.empty() || cols == 0 ||
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
    if (c.layers[l] != lfm3::LayerType::kConvolution && c.kv_heads[l] != kv_heads)
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
    std::fprintf(stderr, "[d1 npu ] L%u %-8s mean|.| %.5f zeros %zu/%zu bad %zu first %.4f %.4f\n",
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
      rocm::NpuDecodeOutput(s.npu->Staged(s.stream, s.c_in), s.p_buf, n,
                            Impl::kInPitch, Impl::kInPitch, 0, false, s.stream);
      stat("p", l, s.p_buf, static_cast<std::size_t>(n) * Impl::kInPitch);
      hipLaunchKernelGGL(D1ConvKernel, dim3((H + 127) / 128, n), dim3(128), 0,
                         s.stream, s.p_buf,
                         s.g_conv + static_cast<std::size_t>(l) * H * s.taps,
                         s.seq_lo, s.conv_out, n, H, s.taps);
      s.EncodeA(s.conv_out, n, H, s.a_h);
      if (!s.Gemm(s.prog_h, s.a_h, s.w_out[l], s.c_h))
        return false;
      rocm::NpuDecodeOutput(s.npu->Staged(s.stream, s.c_h), s.branch, n, H, H,
                            0, false, s.stream, H);
      stat("conv", l, s.conv_out, static_cast<std::size_t>(n) * H);
      stat("branch", l, s.branch, static_cast<std::size_t>(n) * H);
    } else {
      s.EncodeA(s.z, n, H, s.a_h);
      if (!s.Gemm(s.prog_qkv, s.a_h, s.w_qkv[l], s.c_qkv))
        return false;
      rocm::NpuDecodeOutput(s.npu->Staged(s.stream, s.c_qkv), s.qkv_buf, n,
                            Impl::kQkvPitch, Impl::kQkvPitch, 0, false,
                            s.stream);
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
      hipLaunchKernelGGL(D1RopeKernel, dim3(n, heads + kv_heads), dim3(hd / 2),
                         0, s.stream, s.qn, s.kn,
                         swa ? static_cast<const float*>(nullptr) : s.g_factors,
                         s.seq_lo, n, heads, kv_heads, hd,
                         swa ? c.theta_swa : c.theta);
      hipLaunchKernelGGL(D1AttnKernel, dim3(n, heads), dim3(hd), 0, s.stream,
                         s.qn, s.kn, s.v, s.gate_buf, s.group_buf, s.seq_lo,
                         s.seq_hi, s.attn, n, heads, kv_heads, hd, scale,
                         swa ? c.window : 0u);
      s.EncodeA(s.attn, n, q_dim, s.a_h);
      if (!s.Gemm(s.prog_h, s.a_h, s.w_out[l], s.c_h))
        return false;
      rocm::NpuDecodeOutput(s.npu->Staged(s.stream, s.c_h), s.branch, n, H, H,
                            0, false, s.stream, H);
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
    rocm::NpuDecodeOutput(s.npu->Staged(s.stream, s.c_gu), s.gu_buf, n,
                          Impl::kGuPitch, Impl::kGuPitch, 0, false, s.stream);
    hipLaunchKernelGGL(
        D1SwigluKernel,
        dim3(static_cast<std::uint32_t>(
            (static_cast<std::size_t>(n) * I + 255) / 256)),
        dim3(256), 0, s.stream, s.gu_buf, s.act, I, Impl::kGuPitch,
        static_cast<std::size_t>(n) * I);
    s.EncodeA(s.act, n, I, s.a_i);
    if (!s.Gemm(s.prog_down, s.a_i, s.w_down[l], s.c_h))
      return false;
    rocm::NpuDecodeOutput(s.npu->Staged(s.stream, s.c_h), s.branch, n, H, H, 0,
                          false, s.stream, H);
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
      std::fprintf(stderr, "[d1 npu ] layer %u mean|x| %.5f last %.4f %.4f %.4f %.4f\n",
                   l, mean / h.size(), h[last], h[last + 1], h[last + 2],
                   h[last + 3]);
    }
  }

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
                             std::chrono::steady_clock::now() - t_start).count();
    std::fprintf(stderr,
                 "[d1 prof] n=%u total %.1f ms: encode %.1f gemm %.1f (%d calls) other %.1f\n",
                 n, total, s.t_encode, s.t_gemm, s.n_gemm,
                 total - s.t_encode - s.t_gemm);
    if (s.prof_fine)
      std::fprintf(stderr,
                   "[d1 prof] encode parts: tohalf %.1f memset %.1f kernel %.1f finalize %.1f\n",
                   s.t_half, s.t_memset, s.t_enckernel, s.t_final);
  }
  return s.npu->Healthy();
}

}  // namespace gufo::models::qwen38_flash_next
