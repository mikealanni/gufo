#include "src/models/qwen38_flash_next/jev_npu.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

using core::GgmlType;
using tokenization::TokenId;

constexpr std::uint32_t M = 512;

// Batched judge calls (RunBatch): the NPU program and every staging buffer
// are already fixed at M rows regardless of how many are real tokens (see
// the single-question Run() path), so packing kJevBatchMax independent
// questions into kJevSeqCap-row slots instead of one question into M rows
// of mostly padding costs ~nothing extra on the NPU side. kJevSeqCap must
// stay a multiple of 128 (the activation-encode tile) and
// kJevBatchMax * kJevSeqCap must equal M.
constexpr std::uint32_t kJevSeqCap = 128;
constexpr std::uint32_t kJevBatchMax = M / kJevSeqCap;

struct BlockQ4_0 {
  std::uint16_t d;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18);

struct JevTensor {
  const void* data{nullptr};
  GgmlType type{GgmlType::kF32};
  std::size_t elements{0};
  std::size_t available_bytes{std::numeric_limits<std::size_t>::max()};
  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || elements == 0;
  }
};

struct JevLayer {
  JevTensor attn_norm, ffn_norm, wq, wk, wv, wo, q_norm, k_norm, ffn_gate,
      ffn_up, ffn_down;
};

struct JevConfig {
  std::uint32_t num_layers{0}, hidden{0}, intermediate{0}, heads{0},
      kv_heads{0}, head_dim{0}, vocab{0}, context{0};
  float rope_theta{1000000.0F};
  float rms_eps{1e-6F};
  // Per-head Q/K RMSNorm before RoPE is a Qwen3 architectural feature, not
  // a standard Llama one (MiniCPM5 has neither tensor). Set at load time
  // from whether attn_q_norm/attn_k_norm tensors actually exist; gates
  // both the load-time dequant and the runtime RmsNormRows(d_q/d_k, ...)
  // step in Run()/RunBatch() -- applying RMSNorm with a zero/garbage
  // gamma would silently corrupt output rather than just fail, so this
  // must be skipped, not defaulted.
  bool has_qk_norm{false};
};

JevTensor BindTensor(const core::GgufReader& reader, std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr)
    return {};
  const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (address >= region_address) {
      const std::size_t offset = address - region_address;
      if (offset < region.size) {
        available = region.size - offset;
        break;
      }
    }
  }
  JevTensor t;
  t.data = tensor->data;
  t.type = tensor->type;
  t.elements = tensor->ElementCount();
  t.available_bytes = available;
  return t;
}

std::size_t RowBytes(GgmlType type, std::size_t columns) {
  switch (type) {
    case GgmlType::kF32:
      return columns * sizeof(float);
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return columns * sizeof(std::uint16_t);
    default:
      break;
  }
  return gufo::quant::QuantizedRowBytes(type, columns);
}

void DequantizeRow(const JevTensor& tensor, std::size_t row, std::size_t cols,
                   float* dst) {
  const std::size_t rb = RowBytes(tensor.type, cols);
  const auto* src = static_cast<const std::uint8_t*>(tensor.data) + row * rb;
  gufo::quant::Dequantize(tensor.type, src, dst, cols);
}

void DequantizeTensor(const JevTensor& t, std::vector<float>& out) {
  out.assign(t.elements, 0.0f);
  const std::size_t rows =
      (t.elements + (RowBytes(t.type, 1) - 1)) /
      (RowBytes(t.type, 1));  // not exact for quantized; fall back per-row
  // Generic: dequantize row by row assuming last dim is columns.
  (void)rows;
}

// ---------------------------------------------------------------------------
// Judge GPU kernels. The judge computes nothing on the CPU: embeddings,
// norms, rope, attention, swiglu, residual adds and the gathered lm_head
// dot all run here against host-pinned (zero-copy) buffers.
// ---------------------------------------------------------------------------

template<bool Q6>
__device__ void JevDequantUnit(const std::uint8_t* b, float* o) {
  if constexpr (Q6) {
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
      o[2 * j] = d * static_cast<float>((b[2 + j] & 0x0FU) - 8);
      o[2 * j + 1] = d * static_cast<float>((b[2 + j] >> 4U) - 8);
    }
  }
}

template<bool Q6>
__global__ void JevEmbedKernel(const std::uint8_t* __restrict__ table,
                               const std::int32_t* __restrict__ ids,
                               float* __restrict__ out, std::uint32_t n,
                               std::uint32_t hidden, std::uint64_t row_bytes,
                               std::uint32_t vocab, std::size_t units) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= units)
    return;
  const std::uint32_t per_row = hidden / (Q6 ? 256u : 32u);
  const std::uint32_t row = static_cast<std::uint32_t>(g / per_row);
  const std::uint32_t unit = static_cast<std::uint32_t>(g % per_row);
  float* o =
      out + static_cast<std::size_t>(row) * hidden + unit * (Q6 ? 256u : 32u);
  if (row < n) {
    const std::int32_t id = ids[row];
    if (id >= 0 && static_cast<std::uint32_t>(id) < vocab) {
      JevDequantUnit<Q6>(
          table +
              static_cast<std::size_t>(static_cast<std::uint32_t>(id)) *
                  row_bytes +
              unit * (Q6 ? 210u : 18u),
          o);
      return;
    }
  }
  for (std::uint32_t e = 0; e < (Q6 ? 256u : 32u); ++e)
    o[e] = 0.0f;
}

template<bool Q6>
__global__ void JevGatherDotKernel(const std::uint8_t* __restrict__ table,
                                   const std::int32_t* __restrict__ ids,
                                   const float* __restrict__ vec,
                                   float* __restrict__ out,
                                   std::uint32_t hidden,
                                   std::uint64_t row_bytes) {
  const int label = blockIdx.x;
  const std::uint32_t per_row = hidden / (Q6 ? 256u : 32u);
  const std::uint8_t* row =
      table + static_cast<std::size_t>(static_cast<std::uint32_t>(ids[label])) *
                  row_bytes;
  float partial = 0.0f;
  for (std::uint32_t unit = threadIdx.x; unit < per_row; unit += blockDim.x) {
    constexpr int kUnit = Q6 ? 256 : 32;
    float v[kUnit];
    JevDequantUnit<Q6>(row + unit * (Q6 ? 210u : 18u), v);
    const float* vv = vec + static_cast<std::size_t>(unit) * kUnit;
    for (int e = 0; e < kUnit; ++e)
      partial += v[e] * vv[e];
  }
  __shared__ float red[64];
  red[threadIdx.x] = partial;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (unsigned i = 0; i < blockDim.x; ++i)
      sum += red[i];
    out[label] = sum;
  }
}

// `seq_cap` is the per-question slot stride for batched judge calls (each of
// up to kJevBatchMax independent questions occupies its own `seq_cap`-row
// slot in the same M-row buffer; see JevAttnKernel). Position resets to 0 at
// the start of every slot, since jev never continues a KV cache across
// calls -- there is no `startpos` offset to add.
__global__ void JevRopeKernel(float* q, float* k, const std::uint32_t* startpos,
                              std::uint32_t n, std::uint32_t q_dim,
                              std::uint32_t kv_dim, std::uint32_t heads,
                              std::uint32_t kv_heads, std::uint32_t hd,
                              float theta, std::uint32_t seq_cap) {
  const std::uint32_t row = blockIdx.x;
  if (row >= n)
    return;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t dd = threadIdx.x;  // 0..hd/2-1
  const float pos = static_cast<float>(*startpos + (row % seq_cap));
  const float freq = 1.0f / powf(theta, (2.0f * static_cast<float>(dd)) /
                                            static_cast<float>(hd));
  const float c = cosf(pos * freq);
  const float sn = sinf(pos * freq);
  float* base = nullptr;
  if (h < heads) {
    base = q + static_cast<std::size_t>(row) * q_dim +
           static_cast<std::size_t>(h) * hd;
  } else {
    const std::uint32_t kh = h - heads;
    if (kh >= kv_heads)
      return;
    base = k + static_cast<std::size_t>(row) * kv_dim +
           static_cast<std::size_t>(kh) * hd;
  }
  const float a = base[dd];
  const float b = base[dd + hd / 2];
  base[dd] = a * c - b * sn;
  base[dd + hd / 2] = a * sn + b * c;
}

__global__ void JevStoreKvKernel(const float* k, const float* v, __half* kc,
                                 __half* vc, std::uint32_t n,
                                 std::uint32_t kv_dim) {
  const std::uint32_t row = blockIdx.x;
  const std::size_t i = static_cast<std::size_t>(row) * kv_dim + threadIdx.x;
  if (row < n) {
    kc[i] = __float2half(k[i]);
    vc[i] = __float2half(v[i]);
  } else {
    kc[i] = __float2half(0.0f);
    vc[i] = __float2half(0.0f);
  }
}

// Batched variant: `n` is the total row count across all slots (B*seq_cap).
// Every row gets a real cache entry; JevAttnKernel never reads a row beyond
// its own question's valid length, so there is no padding-zero branch to
// get wrong here the way the single-question kernel above has.
__global__ void JevStoreKvBatchedKernel(const float* k, const float* v,
                                        __half* kc, __half* vc, std::uint32_t n,
                                        std::uint32_t kv_dim) {
  const std::uint32_t row = blockIdx.x;
  if (row >= n)
    return;
  const std::size_t i = static_cast<std::size_t>(row) * kv_dim + threadIdx.x;
  kc[i] = __float2half(k[i]);
  vc[i] = __float2half(v[i]);
}

/// Causal attention over one (row, head). One thread per head element: each
/// thread scores a strided set of keys (full head dot), reduces max/sum
/// block-wide, then accumulates its output element over the softmax row.
/// `hd` is the launch width (<= 128).
__global__ void JevAttnKernel(const float* __restrict__ q,
                              const __half* __restrict__ kc,
                              const __half* __restrict__ vc,
                              float* __restrict__ out, std::uint32_t n,
                              std::uint32_t q_dim, std::uint32_t kv_dim,
                              std::uint32_t heads, std::uint32_t kv_heads,
                              std::uint32_t hd, float scale) {
  const int row = blockIdx.x;
  if (row >= static_cast<int>(n))
    return;
  const int h = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int hd_i = static_cast<int>(hd);
  const int kvh = h * static_cast<int>(kv_heads) / static_cast<int>(heads);
  __shared__ float w[512];
  __shared__ float red[4];
  const float* qq = q + static_cast<std::size_t>(row) * q_dim +
                    static_cast<std::size_t>(h) * hd;
  const __half* kb = kc + static_cast<std::size_t>(kvh) * hd;
  float m = -1e30f;
  for (int p = tid; p <= row; p += hd_i) {
    const __half* kp = kb + static_cast<std::size_t>(p) * kv_dim;
    float d = 0.0f;
    for (std::uint32_t j = 0; j < hd; ++j)
      d += qq[j] * __half2float(kp[j]);
    w[p] = d * scale;
    m = fmaxf(m, w[p]);
  }
  for (int off = 16; off; off >>= 1)
    m = fmaxf(m, __shfl_down_sync(~0ull, m, static_cast<unsigned>(off)));
  if ((tid & 31) == 0)
    red[tid >> 5] = m;
  __syncthreads();
  const int warps = (hd_i + 31) / 32;
  // red[] holds each warp's FULL max (its lane 0 wrote it). Adopt the max
  // over ALL warps: keeping this thread's butterfly suffix-max and folding
  // in only red[1..] weighted each key by exp(s_p - suffixmax_p) -- fine on
  // near-uniform early layers, catastrophic on the peaked deep layers.
  m = red[0];
  for (int i = 1; i < warps; ++i)
    m = fmaxf(m, red[i]);
  float sum = 0.0f;
  for (int p = tid; p <= row; p += hd_i) {
    w[p] = __expf(w[p] - m);
    sum += w[p];
  }
  for (int off = 16; off; off >>= 1)
    sum += __shfl_down_sync(~0ull, sum, static_cast<unsigned>(off));
  __syncthreads();
  if ((tid & 31) == 0)
    red[tid >> 5] = sum;
  __syncthreads();
  sum = 0.0f;
  for (int i = 0; i < warps; ++i)
    sum += red[i];
  const float inv = 1.0f / sum;
  const __half* vb = vc + static_cast<std::size_t>(kvh) * hd + tid;
  float acc = 0.0f;
  for (int p = 0; p <= row; ++p)
    acc += w[p] * __half2float(vb[static_cast<std::size_t>(p) * kv_dim]);
  out[static_cast<std::size_t>(row) * q_dim + static_cast<std::size_t>(h) * hd +
      tid] = acc * inv;
}

// Batched variant: the KV cache holds up to kJevBatchMax independent
// questions, each in its own `seq_cap`-row slot (slot qid = row / seq_cap).
// Causal attention must stay inside that slot -- a question must never
// attend into another question's tokens, and must stop at its own real
// length (`lens[qid]`), not at the padding rows that fill out its slot.
// `local` is the query's row within its own slot; `kb`/`vb` are offset to
// the start of that slot's own KV range, so every index below is local.
__global__ void JevAttnBatchedKernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, float* __restrict__ out, std::uint32_t n,
    std::uint32_t q_dim, std::uint32_t kv_dim, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t hd, float scale,
    std::uint32_t seq_cap, const std::uint32_t* __restrict__ lens) {
  const int row = blockIdx.x;
  if (row >= static_cast<int>(n))
    return;
  const std::uint32_t qid = static_cast<std::uint32_t>(row) / seq_cap;
  const int local = row % static_cast<int>(seq_cap);
  const int valid_len = static_cast<int>(lens[qid]);
  if (local >= valid_len)
    return;  // padding row within this slot; its output is never read
  const int h = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int hd_i = static_cast<int>(hd);
  const int kvh = h * static_cast<int>(kv_heads) / static_cast<int>(heads);
  __shared__ float w[512];
  __shared__ float red[4];
  const float* qq = q + static_cast<std::size_t>(row) * q_dim +
                    static_cast<std::size_t>(h) * hd;
  const std::size_t slot_base =
      static_cast<std::size_t>(qid) * seq_cap * kv_dim;
  const __half* kb = kc + slot_base + static_cast<std::size_t>(kvh) * hd;
  float m = -1e30f;
  for (int p = tid; p <= local; p += hd_i) {
    const __half* kp = kb + static_cast<std::size_t>(p) * kv_dim;
    float d = 0.0f;
    for (std::uint32_t j = 0; j < hd; ++j)
      d += qq[j] * __half2float(kp[j]);
    w[p] = d * scale;
    m = fmaxf(m, w[p]);
  }
  for (int off = 16; off; off >>= 1)
    m = fmaxf(m, __shfl_down_sync(~0ull, m, static_cast<unsigned>(off)));
  if ((tid & 31) == 0)
    red[tid >> 5] = m;
  __syncthreads();
  const int warps = (hd_i + 31) / 32;
  // red[] holds each warp's FULL max (its lane 0 wrote it). Adopt the max
  // over ALL warps: keeping this thread's butterfly suffix-max and folding
  // in only red[1..] weighted each key by exp(s_p - suffixmax_p) -- fine on
  // near-uniform early layers, catastrophic on the peaked deep layers.
  m = red[0];
  for (int i = 1; i < warps; ++i)
    m = fmaxf(m, red[i]);
  float sum = 0.0f;
  for (int p = tid; p <= local; p += hd_i) {
    w[p] = __expf(w[p] - m);
    sum += w[p];
  }
  for (int off = 16; off; off >>= 1)
    sum += __shfl_down_sync(~0ull, sum, static_cast<unsigned>(off));
  __syncthreads();
  if ((tid & 31) == 0)
    red[tid >> 5] = sum;
  __syncthreads();
  sum = 0.0f;
  for (int i = 0; i < warps; ++i)
    sum += red[i];
  const float inv = 1.0f / sum;
  const __half* vb = vc + slot_base + static_cast<std::size_t>(kvh) * hd + tid;
  float acc = 0.0f;
  for (int p = 0; p <= local; ++p)
    acc += w[p] * __half2float(vb[static_cast<std::size_t>(p) * kv_dim]);
  out[static_cast<std::size_t>(row) * q_dim + static_cast<std::size_t>(h) * hd +
      tid] = acc * inv;
}

__global__ void JevSplitQkvKernel(const float* qkv, float* q, float* k,
                                  float* v, std::uint32_t n,
                                  std::uint32_t q_dim, std::uint32_t kv_dim,
                                  std::uint32_t n_qkv) {
  const std::uint32_t row = blockIdx.x;
  if (row >= n)
    return;
  const float* src = qkv + static_cast<std::size_t>(row) * n_qkv;
  float* dq = q + static_cast<std::size_t>(row) * q_dim;
  float* dk = k + static_cast<std::size_t>(row) * kv_dim;
  float* dv = v + static_cast<std::size_t>(row) * kv_dim;
  for (std::uint32_t d = threadIdx.x; d < q_dim; d += blockDim.x)
    dq[d] = src[d];
  for (std::uint32_t d = threadIdx.x; d < kv_dim; d += blockDim.x) {
    dk[d] = src[q_dim + d];
    dv[d] = src[q_dim + kv_dim + d];
  }
}

__global__ void JevSwigluGuKernel(const float* __restrict__ gu,
                                  float* __restrict__ act, std::uint32_t inter,
                                  std::uint32_t n_gu, std::size_t cnt) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cnt)
    return;
  const std::size_t row = i / inter;
  const std::uint32_t col = static_cast<std::uint32_t>(i % inter);
  const float g = gu[row * n_gu + col];
  act[i] = g / (1.0f + __expf(-g)) * gu[row * n_gu + inter + col];
}

__global__ void JevAddKernel(float* x, const float* y, std::size_t cnt) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < cnt)
    x[i] += y[i];
}

__global__ void JevToHalfKernel(const float* x, __half* y, std::size_t cnt) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < cnt)
    y[i] = __float2half(x[i]);
}

}  // namespace

struct JevNpuModel::Impl {
  JevConfig config;
  JevTensor token_embd, output_norm, output;
  std::vector<JevLayer> layers;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer;
  std::vector<TokenId> yes_ids, no_ids;
  std::string sys_prompt{
      "Judge whether the statement or question is true or correct. Answer "
      "with exactly one word: Yes or No."};
  // Qwen3's own chat template does not prepend a BOS token (add_bos_token
  // is false and the template itself has no `{{ bos_token }}`). Llama-
  // architecture models (MiniCPM5) do: their template literally starts
  // with `{{- bos_token }}` -- diffed character-for-character against
  // the real GGUF chat_template string. Empty for qwen3, "<s>" for llama.
  std::string bos_prefix;
  std::uint32_t emb_rows{0};

  std::unique_ptr<rocm::NpuGemm> npu;
  int prog_qkv{0}, prog_o{0}, prog_gu{0}, prog_down{0};
  std::vector<std::vector<int>> wqkv, wo, wgu, wdown;
  mutable int a_hidden{-1}, a_qdim{-1}, a_inter{-1}, c_qkv{-1}, c_o{-1},
      c_gu{-1}, c_down{-1};
  mutable float* h_qkv{nullptr};
  mutable float* h_gu{nullptr};

  float *d_x{nullptr}, *d_norm{nullptr}, *d_q{nullptr}, *d_k{nullptr},
      *d_v{nullptr};
  float *d_attn{nullptr}, *d_attn_out{nullptr}, *d_gate{nullptr},
      *d_up{nullptr};
  float *d_act{nullptr}, *d_ffn_out{nullptr};
  __half *d_normh{nullptr}, *d_ah{nullptr};  // half activation buffers
  std::vector<__half*> d_kcache, d_vcache;
  hipStream_t stream{nullptr};
  std::uint32_t* d_startpos{nullptr};
  float *d_qn{nullptr}, *d_kn{nullptr}, *d_last{nullptr}, *d_dot{nullptr};
  std::int32_t* d_ids{nullptr};
  std::int32_t* d_lmid{nullptr};
  // Batched judge calls (RunBatch): up to kJevBatchMax independent
  // questions share the M-row buffer, each in its own kJevSeqCap-row slot.
  // d_lens[qid] is that question's real (unpadded) token count.
  std::uint32_t* d_lens{nullptr};
  // qkv projection's real output width (q_dim + 2*kv_dim) doesn't always
  // have an exact-match instruction file on disk (e.g. MiniCPM5-2B's
  // n_qkv=2560 hits a real aiecc crash generating that shape). n_qkv_cap
  // is the smallest *existing* instruction-file N >= n_qkv -- the qkv
  // GEMM runs padded to this width, and only the first q_dim+2*kv_dim
  // real columns of each output row are ever read. Equals n_qkv itself
  // whenever an exact-match instruction file exists (every model this
  // engine has run so far besides MiniCPM5-2B).
  std::uint32_t n_qkv_cap{0};
  std::uint8_t* d_embtab{nullptr};
  std::uint8_t* d_outtab{nullptr};
  float *h_qnw{nullptr}, *h_knw{nullptr}, *h_attnorm{nullptr},
      *h_ffnorm{nullptr}, *h_fnorm{nullptr};

  bool EncodeWeights() {
    const std::uint32_t hidden = config.hidden;
    const std::uint32_t inter = config.intermediate;
    const std::uint32_t q_dim = config.heads * config.head_dim;
    const std::uint32_t kv_dim = config.kv_heads * config.head_dim;
    const std::uint32_t n_qkv = q_dim + 2 * kv_dim;
    const std::uint32_t n_qkv_cap =
        this->n_qkv_cap != 0 ? this->n_qkv_cap : n_qkv;
    const std::uint32_t n_gu = 2 * inter;

    auto encode_one = [&](const std::vector<JevTensor>& srcs,
                          const std::vector<std::uint32_t>& widths,
                          std::uint32_t k_total, std::uint32_t n_total_cap,
                          std::vector<int>& out_ids) -> bool {
      // Build fused buffer by encoding each 128-col chunk. The GEMM
      // program's instruction stream may be padded wider than the real
      // output width (n_qkv_cap: see its comment on the Impl struct) --
      // n_total_cap is that padded width. The weight buffer must be sized
      // to it, not just the real per-source widths, or the padded
      // program's DMA reads run past the end of this allocation. The
      // padding tail stays correctly zero from the memset below; only the
      // real per-source columns get written, unchanged.
      std::uint32_t n_total = 0;
      for (auto w : widths)
        n_total += w;
      n_total = std::max(n_total, n_total_cap);
      const std::size_t tb =
          static_cast<std::size_t>(k_total) * n_total / 8 * 9;
      const int buf = npu->NewBuffer(tb, nullptr);
      if (buf < 0)
        return false;
      std::memset(npu->Host(buf), 0, tb);
      std::uint32_t col0 = 0;
      for (std::size_t s = 0; s < srcs.size(); ++s) {
        const auto& t = srcs[s];
        const std::uint32_t w = widths[s];
        // The encode kernel reads weights on the GPU, but GGUF tensors are
        // mmap'd file pages: stage this tensor through pinned memory first.
        const std::size_t rows = t.elements / k_total;
        const std::size_t tensor_bytes = RowBytes(t.type, k_total) * rows;
        void* pin = nullptr;
        if (hipHostMalloc(&pin, tensor_bytes) != hipSuccess)
          return false;
        std::memcpy(pin, t.data, tensor_bytes);
        // One full-width encode pass per source. The kernel's tile stream is
        // column-tile-major (128 output rows per tile, K inner) and advances
        // through the tensor's rows by tile index, so `cols` must be the
        // source's full width and k_valid the full reduction length -- and
        // the destination is this source's span inside the fused per-layer
        // buffer (col0/128 whole tiles in).
        rocm::NpuEncodeWeightsQuant(
            pin, static_cast<std::uint32_t>(t.type), k_total, 0, w, k_total, 0,
            k_total,
            static_cast<std::uint8_t*>(npu->Device(buf)) +
                (static_cast<std::size_t>(col0) / 128) * k_total * 128 / 8 * 9,
            stream);
        hipStreamSynchronize(stream);
        hipHostFree(pin);
        col0 += w;
      }
      npu->FinalizeWeights(buf);
      out_ids.push_back(buf);
      return true;
    };

    wqkv.assign(config.num_layers, {});
    wo.assign(config.num_layers, {});
    wgu.assign(config.num_layers, {});
    wdown.assign(config.num_layers, {});
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
      const auto& layer = layers[l];
      std::vector<JevTensor> s;
      std::vector<std::uint32_t> ws;
      s = {layer.wq, layer.wk, layer.wv};
      ws = {q_dim, kv_dim, kv_dim};
      if (!encode_one(s, ws, hidden, n_qkv_cap, wqkv[l]))
        return false;
      s = {layer.wo};
      ws = {hidden};
      if (!encode_one(s, ws, q_dim, hidden, wo[l]))
        return false;
      s = {layer.ffn_gate, layer.ffn_up};
      ws = {inter, inter};
      if (!encode_one(s, ws, hidden, n_gu, wgu[l]))
        return false;
      s = {layer.ffn_down};
      ws = {hidden};
      if (!encode_one(s, ws, inter, hidden, wdown[l]))
        return false;
    }
    return true;
  }

  JevBinaryVerdict Run(std::string_view question,
                       std::string_view context) const {
    // Prompt
    std::string prompt;
    prompt.reserve(question.size() + context.size() + 512);
    prompt.append(bos_prefix);
    // Matches tok.apply_chat_template(..., enable_thinking=False) exactly
    // (diffed character-for-character against the real HF template output
    // -- every turn needs its own kImEnd; this was missing before and
    // produced near-constant garbage verdicts on Qwen3-1.7B specifically,
    // not an NPU bug).
    prompt.append(tokenization::kImStart);
    prompt.append("system\n");
    prompt.append(sys_prompt);
    prompt.append(tokenization::kImEnd);
    prompt.append("\n");
    prompt.append(tokenization::kImStart);
    prompt.append("user\n");
    if (!context.empty()) {
      prompt.append(context);
      prompt.append("\n\n");
    }
    prompt.append("Question: ");
    prompt.append(question);
    prompt.append("\nAnswer with Yes or No.");
    prompt.append(tokenization::kImEnd);
    prompt.append("\n");
    prompt.append(tokenization::kImStart);
    prompt.append("assistant\n");
    // Official Qwen3 enable_thinking=False framing: a pre-closed empty
    // think block. Without this, Qwen3 (a thinking model) saturates to
    // Yes on bare yes/no prompts regardless of framing (+9 logit gap,
    // reproduced on CPU bf16 reference) -- this is the fix, not a model
    // or NPU bug. See jev-gate-plan.md's 2026-11-05 status.
    prompt.append("<think>\n\n</think>\n\n");
    const auto tokens = tokenizer->Encode(prompt);
    if (tokens.empty())
      return {};
    const std::uint32_t n =
        std::min<std::uint32_t>(static_cast<std::uint32_t>(tokens.size()), M);
    // Activation-encode window: n rounded up to a 128-row tile.
    const std::uint32_t enc_rows = (n + 127u) / 128u * 128u;
    auto fp8_rows = [&](std::uint32_t cols) {
      return static_cast<std::size_t>(enc_rows) * cols / 8 * 9;
    };

    const std::uint32_t hidden = config.hidden;
    const std::uint32_t heads = config.heads;
    const std::uint32_t kv_heads = config.kv_heads;
    const std::uint32_t hd = config.head_dim;
    const std::uint32_t q_dim = heads * hd;
    const std::uint32_t kv_dim = kv_heads * hd;
    const std::uint32_t inter = config.intermediate;
    const std::uint32_t n_qkv = q_dim + 2 * kv_dim;
    const std::uint32_t n_qkv_cap =
        this->n_qkv_cap != 0 ? this->n_qkv_cap : n_qkv;
    const std::uint32_t n_gu = 2 * inter;

    // Embed with the GPU into d_x (M x hidden), zero-padded beyond n. No
    // KV reset is needed: JevStoreKvKernel rewrites every cache row.
    for (std::uint32_t i = 0; i < n; ++i)
      d_ids[i] = static_cast<std::int32_t>(tokens[i]);
    {
      const std::uint8_t* table = d_embtab;
      const std::uint64_t row_bytes = RowBytes(token_embd.type, hidden);
      if (token_embd.type == GgmlType::kQ6_K) {
        const std::size_t units = static_cast<std::size_t>(M) * (hidden / 256u);
        hipLaunchKernelGGL(
            (JevEmbedKernel<true>),
            dim3(static_cast<std::uint32_t>((units + 255) / 256)), dim3(256), 0,
            stream, table, d_ids, d_x, n, hidden, row_bytes, config.vocab,
            units);
      } else {
        const std::size_t units = static_cast<std::size_t>(M) * (hidden / 32u);
        hipLaunchKernelGGL(
            (JevEmbedKernel<false>),
            dim3(static_cast<std::uint32_t>((units + 255) / 256)), dim3(256), 0,
            stream, table, d_ids, d_x, n, hidden, row_bytes, config.vocab,
            units);
      }
    }
    // startpos device scalar = 0
    std::uint32_t zero = 0;
    memcpy(d_startpos, &zero, sizeof(zero));

    const std::size_t a_hidden_bytes =
        static_cast<std::size_t>(M) * hidden / 8 * 9;
    const std::size_t a_qdim_bytes =
        static_cast<std::size_t>(M) * q_dim / 8 * 9;
    const std::size_t a_inter_bytes =
        static_cast<std::size_t>(M) * inter / 8 * 9;
    const std::size_t c_qkv_bytes =
        static_cast<std::size_t>(M) * n_qkv_cap / 8 * 9;
    const std::size_t c_o_bytes = static_cast<std::size_t>(M) * hidden / 8 * 9;
    const std::size_t c_gu_bytes = static_cast<std::size_t>(M) * n_gu / 8 * 9;
    const std::size_t c_down_bytes =
        static_cast<std::size_t>(M) * hidden / 8 * 9;

    // NPU staging buffers are allocated once and reused across calls: XRT
    // buffer setup is milliseconds and must not repeat per decision.
    if (a_hidden < 0) {
      a_hidden = npu->NewBuffer(a_hidden_bytes, nullptr);
      a_qdim = npu->NewBuffer(a_qdim_bytes, nullptr);
      a_inter = npu->NewBuffer(a_inter_bytes, nullptr);
      c_qkv = npu->NewBuffer(c_qkv_bytes, nullptr);
      c_o = npu->NewBuffer(c_o_bytes, nullptr);
      c_gu = npu->NewBuffer(c_gu_bytes, nullptr);
      c_down = npu->NewBuffer(c_down_bytes, nullptr);
      if (a_hidden < 0 || a_qdim < 0 || a_inter < 0 || c_qkv < 0 || c_o < 0 ||
          c_gu < 0 || c_down < 0)
        return {};
      std::memset(npu->Host(a_hidden), 0, a_hidden_bytes);
      std::memset(npu->Host(a_qdim), 0, a_qdim_bytes);
      std::memset(npu->Host(a_inter), 0, a_inter_bytes);
    }
    float* qkv = nullptr;
    float* gu = nullptr;
    hipHostMalloc(&qkv, M * n_qkv_cap * sizeof(float));
    hipHostMalloc(&gu, M * n_gu * sizeof(float));
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
      // attn norm (gamma rows pre-dequantized at load)
      rocm::RmsNormRows(d_x, h_attnorm + static_cast<std::size_t>(l) * hidden,
                        d_norm, M, hidden, 1, config.rms_eps, stream);

      // encode d_norm (float) -> half -> NPU block-float
      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_norm, d_normh,
                         static_cast<std::size_t>(n) * hidden);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * hidden, 0,
          static_cast<std::size_t>(enc_rows - n) * hidden * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, hidden, enc_rows, hidden, 0, hidden,
                                 npu->Device(a_hidden), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_hidden);

      // qkv
      int bs[1] = {wqkv[l][0]};
      int cs[1] = {c_qkv};
      npu->Submit(stream, a_hidden, bs, cs, 1, prog_qkv);
      npu->Wait(stream);
      hipStreamSynchronize(stream);

      // decode C -> qkv (M x n_qkv_cap; only the first `n` rows' first
      // n_qkv real columns are read downstream -- decode n rows)
      rocm::NpuDecodeOutput(npu->Staged(stream, c_qkv), qkv, n, n_qkv_cap,
                            n_qkv_cap, 0, false, stream);
      hipStreamSynchronize(stream);

      // split q, k, v (GPU)
      // n_qkv_cap: qkv's row stride in the (possibly padded) decoded
      // buffer -- not the real n_qkv, which only matters for the q_dim/
      // kv_dim offsets within each row, already correct regardless.
      hipLaunchKernelGGL(JevSplitQkvKernel, dim3(n), dim3(128), 0, stream, qkv,
                         d_q, d_k, d_v, n, q_dim, kv_dim, n_qkv_cap);

      // q/k norm per head: each (token, head) is one row sharing one gamma.
      // Llama-architecture models (MiniCPM5) have no q_norm/k_norm tensors
      // -- skip the RMSNorm step and feed RoPE straight from d_q/d_k
      // (applying RMSNorm with a zero/garbage gamma would silently
      // corrupt output rather than just fail).
      if (config.has_qk_norm) {
        rocm::RmsNormRows(d_q, h_qnw + static_cast<std::size_t>(l) * hd, d_qn,
                          n * heads, hd, 1, config.rms_eps, stream);
        rocm::RmsNormRows(d_k, h_knw + static_cast<std::size_t>(l) * hd, d_kn,
                          n * kv_heads, hd, 1, config.rms_eps, stream);
      } else {
        hipMemcpyAsync(d_qn, d_q,
                       static_cast<std::size_t>(n) * q_dim * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
        hipMemcpyAsync(d_kn, d_k,
                       static_cast<std::size_t>(n) * kv_dim * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
      }

      // rope (NEOX) on the GPU; then store the roped K/V as half cache rows.
      hipLaunchKernelGGL(JevRopeKernel, dim3(n, heads + kv_heads), dim3(hd / 2),
                         0, stream, d_qn, d_kn, d_startpos, n, q_dim, kv_dim,
                         heads, kv_heads, hd, config.rope_theta, M);
      hipLaunchKernelGGL(JevStoreKvKernel, dim3(M), dim3(kv_dim), 0, stream,
                         d_kn, d_v, d_kcache[l], d_vcache[l], n, kv_dim);

      // causal attention (GPU)
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      hipLaunchKernelGGL(JevAttnKernel, dim3(n, heads), dim3(hd), 0, stream,
                         d_qn, d_kcache[l], d_vcache[l], d_attn, n, q_dim,
                         kv_dim, heads, kv_heads, hd, scale);
      hipStreamSynchronize(stream);
      // o GEMM
      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * q_dim + 255) / 256),
                         dim3(256), 0, stream, d_attn, d_normh,
                         static_cast<std::size_t>(n) * q_dim);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * q_dim, 0,
          static_cast<std::size_t>(enc_rows - n) * q_dim * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, q_dim, enc_rows, q_dim, 0, q_dim,
                                 npu->Device(a_qdim), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_qdim);
      int bs_o[1] = {wo[l][0]};
      int cs_o[1] = {c_o};
      npu->Submit(stream, a_qdim, bs_o, cs_o, 1, prog_o);
      npu->Wait(stream);
      hipStreamSynchronize(stream);
      rocm::NpuDecodeOutput(npu->Staged(stream, c_o), d_attn_out, n, hidden,
                            hidden, 0, false, stream);
      hipStreamSynchronize(stream);
      // residual x += attn_out (GPU)
      hipLaunchKernelGGL(JevAddKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_x, d_attn_out,
                         static_cast<std::size_t>(n) * hidden);

      // ffn norm
      rocm::RmsNormRows(d_x, h_ffnorm + static_cast<std::size_t>(l) * hidden,
                        d_norm, M, hidden, 1, config.rms_eps, stream);

      // encode d_norm -> a_hidden, submit gu
      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_norm, d_normh,
                         static_cast<std::size_t>(n) * hidden);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * hidden, 0,
          static_cast<std::size_t>(enc_rows - n) * hidden * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, hidden, enc_rows, hidden, 0, hidden,
                                 npu->Device(a_hidden), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_hidden);
      int bs_gu[1] = {wgu[l][0]};
      int cs_gu[1] = {c_gu};
      npu->Submit(stream, a_hidden, bs_gu, cs_gu, 1, prog_gu);
      npu->Wait(stream);
      hipStreamSynchronize(stream);

      // decode c_gu -> gate (first inter) + up (second inter)
      rocm::NpuDecodeOutput(npu->Staged(stream, c_gu), gu, n, n_gu, n_gu, 0,
                            false, stream);
      hipStreamSynchronize(stream);
      // swiglu straight from the fused gate|up GEMM output (GPU)
      hipLaunchKernelGGL(
          JevSwigluGuKernel,
          dim3(static_cast<std::uint32_t>(
              (static_cast<std::size_t>(n) * inter + 255) / 256)),
          dim3(256), 0, stream, gu, d_act, inter, n_gu,
          static_cast<std::size_t>(n) * inter);

      // down GEMM
      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * inter + 255) / 256),
                         dim3(256), 0, stream, d_act, d_normh,
                         static_cast<std::size_t>(n) * inter);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * inter, 0,
          static_cast<std::size_t>(enc_rows - n) * inter * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, inter, enc_rows, inter, 0, inter,
                                 npu->Device(a_inter), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_inter);
      int bs_down[1] = {wdown[l][0]};
      int cs_down[1] = {c_down};
      npu->Submit(stream, a_inter, bs_down, cs_down, 1, prog_down);
      npu->Wait(stream);
      hipStreamSynchronize(stream);
      rocm::NpuDecodeOutput(npu->Staged(stream, c_down), d_ffn_out, n, hidden,
                            hidden, 0, false, stream);
      hipStreamSynchronize(stream);
      // residual (GPU)
      hipLaunchKernelGGL(JevAddKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_x, d_ffn_out,
                         static_cast<std::size_t>(n) * hidden);
    }

    // Final norm on the last real token (GPU).
    rocm::RmsNormRows(d_x + static_cast<std::size_t>(n - 1) * hidden, h_fnorm,
                      d_last, 1, hidden, 1, config.rms_eps, stream);

    // lm_head: gather the yes/no rows and dot with the final hidden state.
    {
      const std::uint8_t* table = d_outtab;
      const std::uint64_t row_bytes = RowBytes(output.type, hidden);
      if (output.type == GgmlType::kQ6_K)
        hipLaunchKernelGGL((JevGatherDotKernel<true>), dim3(2), dim3(64), 0,
                           stream, table, d_lmid, d_last, d_dot, hidden,
                           row_bytes);
      else
        hipLaunchKernelGGL((JevGatherDotKernel<false>), dim3(2), dim3(64), 0,
                           stream, table, d_lmid, d_last, d_dot, hidden,
                           row_bytes);
    }
    hipStreamSynchronize(stream);
    const double yes = std::exp(static_cast<double>(d_dot[0]));
    const double no = std::exp(static_cast<double>(d_dot[1]));
    const double total = yes + no;
    hipHostFree(qkv);
    hipHostFree(gu);
    if (!(total > 0.0) || !std::isfinite(total))
      return {};
    return {static_cast<float>(yes / total), static_cast<float>(no / total)};
  }

  // Batched variant of Run(): answers up to kJevBatchMax independent
  // questions in one NPU pass instead of one question per pass. Each
  // question gets its own kJevSeqCap-row slot; the NPU program and every
  // staging buffer are already sized for the full M rows regardless of how
  // many are real tokens (see Run() above), so this uses capacity that
  // today's single-question path mostly pads and throws away. Returns one
  // verdict per input question, same order, {} entries dropped on failure
  // of the whole batch (all-or-nothing: one NPU pass, one failure mode).
  std::vector<JevBinaryVerdict> RunBatch(
      std::span<const std::pair<std::string_view, std::string_view>> items)
      const {
    const std::uint32_t count = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(items.size()), kJevBatchMax);
    if (count == 0)
      return {};

    const std::uint32_t hidden = config.hidden;
    const std::uint32_t heads = config.heads;
    const std::uint32_t kv_heads = config.kv_heads;
    const std::uint32_t hd = config.head_dim;
    const std::uint32_t q_dim = heads * hd;
    const std::uint32_t kv_dim = kv_heads * hd;
    const std::uint32_t inter = config.intermediate;
    const std::uint32_t n_qkv = q_dim + 2 * kv_dim;
    const std::uint32_t n_qkv_cap =
        this->n_qkv_cap != 0 ? this->n_qkv_cap : n_qkv;
    const std::uint32_t n_gu = 2 * inter;

    // n is always count*kJevSeqCap, never based on any one question's
    // length: every slot is kJevSeqCap rows regardless of how many of
    // those rows are real tokens for that question.
    const std::uint32_t n = count * kJevSeqCap;
    const std::uint32_t enc_rows = n;  // already a multiple of 128

    // Build one slot per question: real token ids, then -1 padding out to
    // kJevSeqCap. -1 padding for unused rows/slots is required, not just
    // tidy -- JevEmbedKernel zeros any row whose id is out of [0, vocab).
    std::vector<std::uint32_t> lens_host(count, 0);
    std::fill(d_ids, d_ids + M, -1);
    for (std::uint32_t qid = 0; qid < count; ++qid) {
      const auto [question, context] = items[qid];
      std::string prompt;
      prompt.reserve(question.size() + context.size() + 512);
      prompt.append(bos_prefix);
      // See Run()'s identical framing above -- kImEnd was missing after
      // every turn, a pre-existing bug this fix also corrects.
      prompt.append(tokenization::kImStart);
      prompt.append("system\n");
      prompt.append(sys_prompt);
      prompt.append(tokenization::kImEnd);
      prompt.append("\n");
      prompt.append(tokenization::kImStart);
      prompt.append("user\n");
      if (!context.empty()) {
        prompt.append(context);
        prompt.append("\n\n");
      }
      prompt.append("Question: ");
      prompt.append(question);
      prompt.append("\nAnswer with Yes or No.");
      prompt.append(tokenization::kImEnd);
      prompt.append("\n");
      prompt.append(tokenization::kImStart);
      prompt.append("assistant\n");
      // See Run()'s identical framing above: the no-think suffix is the
      // fix for Qwen3's bare-yes/no Yes-saturation, not an NPU bug.
      prompt.append("<think>\n\n</think>\n\n");
      const auto tokens = tokenizer->Encode(prompt);
      const std::uint32_t len = std::min<std::uint32_t>(
          static_cast<std::uint32_t>(tokens.size()), kJevSeqCap);
      lens_host[qid] = len;
      for (std::uint32_t i = 0; i < len; ++i)
        d_ids[qid * kJevSeqCap + i] = static_cast<std::int32_t>(tokens[i]);
    }
    if (std::any_of(lens_host.begin(), lens_host.end(),
                    [](std::uint32_t l) { return l == 0; }))
      return {};
    std::memcpy(d_lens, lens_host.data(), count * sizeof(std::uint32_t));

    {
      const std::uint8_t* table = d_embtab;
      const std::uint64_t row_bytes = RowBytes(token_embd.type, hidden);
      if (token_embd.type == GgmlType::kQ6_K) {
        const std::size_t units = static_cast<std::size_t>(M) * (hidden / 256u);
        hipLaunchKernelGGL(
            (JevEmbedKernel<true>),
            dim3(static_cast<std::uint32_t>((units + 255) / 256)), dim3(256), 0,
            stream, table, d_ids, d_x, n, hidden, row_bytes, config.vocab,
            units);
      } else {
        const std::size_t units = static_cast<std::size_t>(M) * (hidden / 32u);
        hipLaunchKernelGGL(
            (JevEmbedKernel<false>),
            dim3(static_cast<std::uint32_t>((units + 255) / 256)), dim3(256), 0,
            stream, table, d_ids, d_x, n, hidden, row_bytes, config.vocab,
            units);
      }
    }
    std::uint32_t zero = 0;
    memcpy(d_startpos, &zero, sizeof(zero));

    const std::size_t a_hidden_bytes =
        static_cast<std::size_t>(M) * hidden / 8 * 9;
    const std::size_t a_qdim_bytes =
        static_cast<std::size_t>(M) * q_dim / 8 * 9;
    const std::size_t a_inter_bytes =
        static_cast<std::size_t>(M) * inter / 8 * 9;
    const std::size_t c_qkv_bytes =
        static_cast<std::size_t>(M) * n_qkv_cap / 8 * 9;
    const std::size_t c_o_bytes = static_cast<std::size_t>(M) * hidden / 8 * 9;
    const std::size_t c_gu_bytes = static_cast<std::size_t>(M) * n_gu / 8 * 9;
    const std::size_t c_down_bytes =
        static_cast<std::size_t>(M) * hidden / 8 * 9;
    if (a_hidden < 0) {
      a_hidden = npu->NewBuffer(a_hidden_bytes, nullptr);
      a_qdim = npu->NewBuffer(a_qdim_bytes, nullptr);
      a_inter = npu->NewBuffer(a_inter_bytes, nullptr);
      c_qkv = npu->NewBuffer(c_qkv_bytes, nullptr);
      c_o = npu->NewBuffer(c_o_bytes, nullptr);
      c_gu = npu->NewBuffer(c_gu_bytes, nullptr);
      c_down = npu->NewBuffer(c_down_bytes, nullptr);
      if (a_hidden < 0 || a_qdim < 0 || a_inter < 0 || c_qkv < 0 || c_o < 0 ||
          c_gu < 0 || c_down < 0)
        return {};
      std::memset(npu->Host(a_hidden), 0, a_hidden_bytes);
      std::memset(npu->Host(a_qdim), 0, a_qdim_bytes);
      std::memset(npu->Host(a_inter), 0, a_inter_bytes);
    }
    float* qkv = nullptr;
    float* gu = nullptr;
    hipHostMalloc(&qkv, M * n_qkv_cap * sizeof(float));
    hipHostMalloc(&gu, M * n_gu * sizeof(float));
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
      rocm::RmsNormRows(d_x, h_attnorm + static_cast<std::size_t>(l) * hidden,
                        d_norm, M, hidden, 1, config.rms_eps, stream);

      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_norm, d_normh,
                         static_cast<std::size_t>(n) * hidden);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * hidden, 0,
          static_cast<std::size_t>(enc_rows - n) * hidden * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, hidden, enc_rows, hidden, 0, hidden,
                                 npu->Device(a_hidden), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_hidden);

      int bs[1] = {wqkv[l][0]};
      int cs[1] = {c_qkv};
      npu->Submit(stream, a_hidden, bs, cs, 1, prog_qkv);
      npu->Wait(stream);
      hipStreamSynchronize(stream);
      rocm::NpuDecodeOutput(npu->Staged(stream, c_qkv), qkv, M, n_qkv_cap,
                            n_qkv_cap, 0, false, stream);
      hipStreamSynchronize(stream);

      // n_qkv_cap: qkv's row stride in the (possibly padded) decoded
      // buffer -- not the real n_qkv, which only matters for the q_dim/
      // kv_dim offsets within each row, already correct regardless.
      hipLaunchKernelGGL(JevSplitQkvKernel, dim3(n), dim3(128), 0, stream, qkv,
                         d_q, d_k, d_v, n, q_dim, kv_dim, n_qkv_cap);
      // See Run()'s identical guard above: Llama-architecture models
      // (MiniCPM5) have no q_norm/k_norm tensors.
      if (config.has_qk_norm) {
        rocm::RmsNormRows(d_q, h_qnw + static_cast<std::size_t>(l) * hd, d_qn,
                          n * heads, hd, 1, config.rms_eps, stream);
        rocm::RmsNormRows(d_k, h_knw + static_cast<std::size_t>(l) * hd, d_kn,
                          n * kv_heads, hd, 1, config.rms_eps, stream);
      } else {
        hipMemcpyAsync(d_qn, d_q,
                       static_cast<std::size_t>(n) * q_dim * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
        hipMemcpyAsync(d_kn, d_k,
                       static_cast<std::size_t>(n) * kv_dim * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
      }
      hipLaunchKernelGGL(JevRopeKernel, dim3(n, heads + kv_heads), dim3(hd / 2),
                         0, stream, d_qn, d_kn, d_startpos, n, q_dim, kv_dim,
                         heads, kv_heads, hd, config.rope_theta, kJevSeqCap);
      hipLaunchKernelGGL(JevStoreKvBatchedKernel, dim3(n), dim3(kv_dim), 0,
                         stream, d_kn, d_v, d_kcache[l], d_vcache[l], n,
                         kv_dim);

      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      hipLaunchKernelGGL(JevAttnBatchedKernel, dim3(n, heads), dim3(hd), 0,
                         stream, d_qn, d_kcache[l], d_vcache[l], d_attn, n,
                         q_dim, kv_dim, heads, kv_heads, hd, scale, kJevSeqCap,
                         d_lens);
      hipStreamSynchronize(stream);

      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * q_dim + 255) / 256),
                         dim3(256), 0, stream, d_attn, d_normh,
                         static_cast<std::size_t>(n) * q_dim);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * q_dim, 0,
          static_cast<std::size_t>(enc_rows - n) * q_dim * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, q_dim, enc_rows, q_dim, 0, q_dim,
                                 npu->Device(a_qdim), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_qdim);
      int bs_o[1] = {wo[l][0]};
      int cs_o[1] = {c_o};
      npu->Submit(stream, a_qdim, bs_o, cs_o, 1, prog_o);
      npu->Wait(stream);
      hipStreamSynchronize(stream);
      rocm::NpuDecodeOutput(npu->Staged(stream, c_o), d_attn_out, M, hidden,
                            hidden, 0, false, stream);
      hipStreamSynchronize(stream);

      hipLaunchKernelGGL(JevAddKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_x, d_attn_out,
                         static_cast<std::size_t>(n) * hidden);

      rocm::RmsNormRows(d_x, h_ffnorm + static_cast<std::size_t>(l) * hidden,
                        d_norm, M, hidden, 1, config.rms_eps, stream);

      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_norm, d_normh,
                         static_cast<std::size_t>(n) * hidden);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * hidden, 0,
          static_cast<std::size_t>(enc_rows - n) * hidden * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, hidden, enc_rows, hidden, 0, hidden,
                                 npu->Device(a_hidden), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_hidden);
      int bs_gu[1] = {wgu[l][0]};
      int cs_gu[1] = {c_gu};
      npu->Submit(stream, a_hidden, bs_gu, cs_gu, 1, prog_gu);
      npu->Wait(stream);
      hipStreamSynchronize(stream);
      rocm::NpuDecodeOutput(npu->Staged(stream, c_gu), gu, M, n_gu, n_gu, 0,
                            false, stream);
      hipStreamSynchronize(stream);
      hipLaunchKernelGGL(
          JevSwigluGuKernel,
          dim3(static_cast<std::uint32_t>(
              (static_cast<std::size_t>(n) * inter + 255) / 256)),
          dim3(256), 0, stream, gu, d_act, inter, n_gu,
          static_cast<std::size_t>(n) * inter);

      hipLaunchKernelGGL(JevToHalfKernel, dim3((n * inter + 255) / 256),
                         dim3(256), 0, stream, d_act, d_normh,
                         static_cast<std::size_t>(n) * inter);
      hipMemsetAsync(
          d_normh + static_cast<std::size_t>(n) * inter, 0,
          static_cast<std::size_t>(enc_rows - n) * inter * sizeof(__half),
          stream);
      rocm::NpuEncodeActivations(d_normh, inter, enc_rows, inter, 0, inter,
                                 npu->Device(a_inter), stream);
      hipStreamSynchronize(stream);
      npu->FinalizeWeights(a_inter);
      int bs_down[1] = {wdown[l][0]};
      int cs_down[1] = {c_down};
      npu->Submit(stream, a_inter, bs_down, cs_down, 1, prog_down);
      npu->Wait(stream);
      hipStreamSynchronize(stream);
      rocm::NpuDecodeOutput(npu->Staged(stream, c_down), d_ffn_out, M, hidden,
                            hidden, 0, false, stream);
      hipStreamSynchronize(stream);

      hipLaunchKernelGGL(JevAddKernel, dim3((n * hidden + 255) / 256),
                         dim3(256), 0, stream, d_x, d_ffn_out,
                         static_cast<std::size_t>(n) * hidden);
    }

    std::vector<JevBinaryVerdict> out(count);
    for (std::uint32_t qid = 0; qid < count; ++qid) {
      const std::uint32_t last_row = qid * kJevSeqCap + lens_host[qid] - 1;
      rocm::RmsNormRows(d_x + static_cast<std::size_t>(last_row) * hidden,
                        h_fnorm, d_last, 1, hidden, 1, config.rms_eps, stream);
      const std::uint8_t* table = d_outtab;
      const std::uint64_t row_bytes = RowBytes(output.type, hidden);
      if (output.type == GgmlType::kQ6_K)
        hipLaunchKernelGGL((JevGatherDotKernel<true>), dim3(2), dim3(64), 0,
                           stream, table, d_lmid, d_last, d_dot, hidden,
                           row_bytes);
      else
        hipLaunchKernelGGL((JevGatherDotKernel<false>), dim3(2), dim3(64), 0,
                           stream, table, d_lmid, d_last, d_dot, hidden,
                           row_bytes);
      hipStreamSynchronize(stream);
      const double yes = std::exp(static_cast<double>(d_dot[0]));
      const double no = std::exp(static_cast<double>(d_dot[1]));
      const double total = yes + no;
      if (!(total > 0.0) || !std::isfinite(total))
        out[qid] = {};
      else
        out[qid] = {static_cast<float>(yes / total),
                    static_cast<float>(no / total)};
    }
    hipHostFree(qkv);
    hipHostFree(gu);
    return out;
  }
};

JevNpuModel::JevNpuModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
JevNpuModel::~JevNpuModel() = default;

std::unique_ptr<JevNpuModel> JevNpuModel::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  // "llama" covers MiniCPM5 (exported as plain Llama arch: RMSNorm + RoPE +
  // GQA + SwiGLU, the same family this loader already reads generically
  // via the architecture-derived metadata prefix below). It lacks Qwen3's
  // per-head Q/K RMSNorm tensors -- guarded below, not assumed.
  if (!architecture.has_value() ||
      (*architecture != "qwen3" && *architecture != "llama")) {
    if (error_msg)
      *error_msg = "jev requires a dense qwen3 or llama-architecture GGUF";
    return nullptr;
  }
  const std::string prefix = std::string(*architecture) + ".";
  auto impl = std::make_unique<Impl>();
  if (*architecture == "llama")
    impl->bos_prefix = "<s>";
  auto getu = [&](const std::string& k, std::uint32_t def) {
    auto v = reader.GetMetadataUint32(k);
    return v.has_value() ? *v : def;
  };
  impl->config.num_layers = getu(prefix + "block_count", 0);
  impl->config.hidden = getu(prefix + "embedding_length", 0);
  impl->config.intermediate = getu(prefix + "feed_forward_length", 0);
  impl->config.heads = getu(prefix + "attention.head_count", 0);
  impl->config.kv_heads = getu(prefix + "attention.head_count_kv", 0);
  impl->config.head_dim = getu(prefix + "attention.key_length", 0);
  if (impl->config.head_dim == 0 && impl->config.heads > 0)
    impl->config.head_dim = impl->config.hidden / impl->config.heads;
  impl->config.vocab = 0;  // set below after token_embd bound
  impl->config.context = getu(prefix + "context_length", 4096);
  {
    auto v = reader.GetMetadataFloat32(prefix + "rope.freq_base");
    if (v)
      impl->config.rope_theta = *v;
    auto e =
        reader.GetMetadataFloat32(prefix + "attention.layer_norm_rms_epsilon");
    if (e)
      impl->config.rms_eps = *e;
  }
  impl->token_embd = BindTensor(reader, "token_embd.weight");
  impl->output_norm = BindTensor(reader, "output_norm.weight");
  impl->output = BindTensor(reader, "output.weight");
  if (impl->output.empty())
    impl->output = impl->token_embd;
  if (impl->token_embd.elements % impl->config.hidden == 0)
    impl->config.vocab = static_cast<std::uint32_t>(impl->token_embd.elements /
                                                    impl->config.hidden);
  impl->layers.assign(impl->config.num_layers, {});
  for (std::uint32_t i = 0; i < impl->config.num_layers; ++i) {
    auto& L = impl->layers[i];
    const std::string b = "blk." + std::to_string(i) + ".";
    L.attn_norm = BindTensor(reader, b + "attn_norm.weight");
    L.ffn_norm = BindTensor(reader, b + "ffn_norm.weight");
    L.wq = BindTensor(reader, b + "attn_q.weight");
    L.wk = BindTensor(reader, b + "attn_k.weight");
    L.wv = BindTensor(reader, b + "attn_v.weight");
    L.wo = BindTensor(reader, b + "attn_output.weight");
    L.q_norm = BindTensor(reader, b + "attn_q_norm.weight");
    L.k_norm = BindTensor(reader, b + "attn_k_norm.weight");
    L.ffn_gate = BindTensor(reader, b + "ffn_gate.weight");
    L.ffn_up = BindTensor(reader, b + "ffn_up.weight");
    L.ffn_down = BindTensor(reader, b + "ffn_down.weight");
    if (i == 0)
      impl->config.has_qk_norm = !L.q_norm.empty() && !L.k_norm.empty();
  }
  impl->tokenizer = tokenization::QwenTokenizer::CreateFromGguf(reader);
  if (!impl->tokenizer) {
    if (error_msg)
      *error_msg = "jev: tokenizer load failed";
    return nullptr;
  }
  if (const char* env = std::getenv("GUFO_JEV_SYS"))
    impl->sys_prompt = env;
  {
    auto ye = impl->tokenizer->Encode("Yes");
    if (!ye.empty())
      impl->yes_ids.push_back(ye[0]);
    auto ne = impl->tokenizer->Encode("No");
    if (!ne.empty())
      impl->no_ids.push_back(ne[0]);
  }
  // NPU init. Program files are shape-baked (M x K x N); derive them from the
  // model so any Qwen3 size runs, and refuse to start when a shape is missing.
  const char* npu_dir = std::getenv("GUFO_NPU_DIR");
  std::string dir = npu_dir ? npu_dir : "tools/npu";
  std::string err;
  const std::uint32_t hidden = impl->config.hidden;
  const std::uint32_t q_dim = impl->config.heads * impl->config.head_dim;
  const std::uint32_t inter = impl->config.intermediate;
  const std::uint32_t n_qkv =
      q_dim + 2 * impl->config.kv_heads * impl->config.head_dim;
  const std::uint32_t n_gu = 2 * inter;
  auto insts = [&](std::uint32_t k, std::uint32_t n) {
    return dir + "/runtime_mk/insts_" + std::to_string(M) + "x" +
           std::to_string(k) + "x" + std::to_string(n) + "_128x64x128.txt";
  };
  // Smallest existing instruction-file N >= want, for shapes with no exact
  // match (see n_qkv_cap's comment on the Impl struct). Tries the exact
  // width first, then every width already generated for this engine.
  auto pad_to_available_n = [&](std::uint32_t k, std::uint32_t want) {
    if (std::ifstream(insts(k, want)).good())
      return want;
    for (std::uint32_t n :
         {1024u, 2048u, 3072u, 4096u, 6144u, 8192u, 10240u, 12288u}) {
      if (n >= want && std::ifstream(insts(k, n)).good())
        return n;
    }
    return want;  // no wider file either; AddProgram below reports the error
  };
  rocm::NpuGemm::Config cfg;
  cfg.xclbin = dir + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
  cfg.insts = insts(hidden, 1024);
  cfg.tokens = M;
  cfg.k = impl->config.hidden;
  cfg.n = 1024;
  impl->npu = rocm::NpuGemm::Create(cfg, &err);
  if (!impl->npu) {
    if (error_msg)
      *error_msg = "jev npu: " + err;
    return nullptr;
  }
  const std::uint32_t n_qkv_cap = pad_to_available_n(hidden, n_qkv);
  impl->n_qkv_cap = n_qkv_cap;
  impl->prog_qkv = impl->npu->AddProgram(insts(hidden, n_qkv_cap), &err);
  impl->prog_o = impl->npu->AddProgram(insts(q_dim, hidden), &err);
  impl->prog_gu = impl->npu->AddProgram(insts(hidden, n_gu), &err);
  impl->prog_down = impl->npu->AddProgram(insts(inter, hidden), &err);
  if (impl->prog_qkv < 0 || impl->prog_o < 0 || impl->prog_gu < 0 ||
      impl->prog_down < 0) {
    if (error_msg)
      *error_msg = "jev npu: program load failed: " + err;
    return nullptr;
  }
  if (impl->token_embd.type != GgmlType::kQ4_0 &&
      impl->token_embd.type != GgmlType::kQ6_K) {
    if (error_msg)
      *error_msg = "jev npu: embedding must be Q4_0 or Q6_K";
    return nullptr;
  }
  if (impl->output.type != GgmlType::kQ4_0 &&
      impl->output.type != GgmlType::kQ6_K) {
    if (error_msg)
      *error_msg = "jev npu: lm_head must be Q4_0 or Q6_K";
    return nullptr;
  }
  if (impl->config.head_dim == 0 || impl->config.head_dim > 128 ||
      impl->config.head_dim % 2 != 0) {
    if (error_msg)
      *error_msg = "jev npu: head_dim must be even and at most 128";
    return nullptr;
  }
  hipStreamCreate(&impl->stream);
  const std::uint32_t kv_dim = impl->config.kv_heads * impl->config.head_dim;
  hipHostMalloc(&impl->d_x, M * hidden * sizeof(float));
  hipHostMalloc(&impl->d_norm, M * hidden * sizeof(float));
  hipHostMalloc(&impl->d_q, M * q_dim * sizeof(float));
  hipHostMalloc(&impl->d_k, M * kv_dim * sizeof(float));
  hipHostMalloc(&impl->d_v, M * kv_dim * sizeof(float));
  hipHostMalloc(&impl->d_attn, M * q_dim * sizeof(float));
  hipHostMalloc(&impl->d_attn_out, M * hidden * sizeof(float));
  hipHostMalloc(&impl->d_gate, M * inter * sizeof(float));
  hipHostMalloc(&impl->d_up, M * inter * sizeof(float));
  hipHostMalloc(&impl->d_act, M * inter * sizeof(float));
  hipHostMalloc(&impl->d_ffn_out, M * hidden * sizeof(float));
  hipHostMalloc(&impl->d_normh,
                M * std::max({hidden, q_dim, inter}) * sizeof(__half));
  hipHostMalloc(&impl->d_ah,
                M * std::max({hidden, q_dim, kv_dim, inter}) * sizeof(__half));
  impl->d_kcache.assign(impl->config.num_layers, nullptr);
  impl->d_vcache.assign(impl->config.num_layers, nullptr);
  for (std::uint32_t l = 0; l < impl->config.num_layers; ++l) {
    hipHostMalloc(&impl->d_kcache[l], M * kv_dim * sizeof(__half));
    hipHostMalloc(&impl->d_vcache[l], M * kv_dim * sizeof(__half));
  }
  hipHostMalloc(&impl->d_startpos, sizeof(std::uint32_t));
  hipHostMalloc(&impl->d_lens, kJevBatchMax * sizeof(std::uint32_t));
  // The GGUF tensors are mmap'd file pages, which GPU page faults on: copy
  // the two quantized tables the judge kernels read into pinned memory.
  {
    const std::size_t emb_rows = impl->token_embd.elements / hidden;
    impl->emb_rows = static_cast<std::uint32_t>(emb_rows);
    const std::size_t emb_bytes =
        RowBytes(impl->token_embd.type, hidden) * emb_rows;
    if (hipHostMalloc(&impl->d_embtab, emb_bytes) != hipSuccess) {
      if (error_msg)
        *error_msg = "jev npu: embedding table copy failed";
      return nullptr;
    }
    memcpy(impl->d_embtab, impl->token_embd.data, emb_bytes);
    if (impl->output.data == impl->token_embd.data) {
      impl->d_outtab = impl->d_embtab;
    } else {
      const std::size_t out_rows = impl->output.elements / hidden;
      const std::size_t out_bytes =
          RowBytes(impl->output.type, hidden) * out_rows;
      if (hipHostMalloc(&impl->d_outtab, out_bytes) != hipSuccess) {
        if (error_msg)
          *error_msg = "jev npu: lm_head table copy failed";
        return nullptr;
      }
      memcpy(impl->d_outtab, impl->output.data, out_bytes);
    }
  }
  hipHostMalloc(&impl->d_qn, M * q_dim * sizeof(float));
  hipHostMalloc(&impl->d_kn, M * kv_dim * sizeof(float));
  hipHostMalloc(&impl->d_last, hidden * sizeof(float));
  hipHostMalloc(&impl->d_dot, 2 * sizeof(float));
  hipHostMalloc(&impl->d_ids, M * sizeof(std::int32_t));
  hipHostMalloc(&impl->d_lmid, 2 * sizeof(std::int32_t));
  const std::uint32_t hd = impl->config.head_dim;
  const std::uint32_t nlayer = impl->config.num_layers;
  hipHostMalloc(&impl->h_attnorm,
                static_cast<std::size_t>(nlayer) * hidden * sizeof(float));
  hipHostMalloc(&impl->h_ffnorm,
                static_cast<std::size_t>(nlayer) * hidden * sizeof(float));
  hipHostMalloc(&impl->h_qnw,
                static_cast<std::size_t>(nlayer) * hd * sizeof(float));
  hipHostMalloc(&impl->h_knw,
                static_cast<std::size_t>(nlayer) * hd * sizeof(float));
  hipHostMalloc(&impl->h_fnorm, hidden * sizeof(float));
  for (std::uint32_t l = 0; l < nlayer; ++l) {
    DequantizeRow(impl->layers[l].attn_norm, 0, hidden,
                  impl->h_attnorm + static_cast<std::size_t>(l) * hidden);
    DequantizeRow(impl->layers[l].ffn_norm, 0, hidden,
                  impl->h_ffnorm + static_cast<std::size_t>(l) * hidden);
    if (impl->config.has_qk_norm) {
      DequantizeRow(impl->layers[l].q_norm, 0, hd,
                    impl->h_qnw + static_cast<std::size_t>(l) * hd);
      DequantizeRow(impl->layers[l].k_norm, 0, hd,
                    impl->h_knw + static_cast<std::size_t>(l) * hd);
    }
  }
  DequantizeRow(impl->output_norm, 0, hidden, impl->h_fnorm);
  impl->d_lmid[0] =
      impl->yes_ids.empty() ? -1 : static_cast<std::int32_t>(impl->yes_ids[0]);
  impl->d_lmid[1] =
      impl->no_ids.empty() ? -1 : static_cast<std::int32_t>(impl->no_ids[0]);
  if (!impl->EncodeWeights()) {
    if (error_msg)
      *error_msg = "jev npu: weight encode failed";
    return nullptr;
  }
  return std::unique_ptr<JevNpuModel>(new JevNpuModel(std::move(impl)));
}

JevBinaryVerdict JevNpuModel::Answer(std::string_view question,
                                     std::string_view context) const {
  const auto t0 = std::chrono::steady_clock::now();
  auto v = impl_->Run(question, context);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  std::fprintf(stderr, "JEV_NPU judge %.1f ms yes=%.2f no=%.2f\n", ms, v.yes,
               v.no);
  std::fflush(stderr);
  return v;
}

std::vector<JevBinaryVerdict> JevNpuModel::AnswerBatch(
    std::span<const std::pair<std::string_view, std::string_view>> items)
    const {
  const auto t0 = std::chrono::steady_clock::now();
  auto v = impl_->RunBatch(items);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  std::fprintf(stderr, "JEV_NPU batch judge %.1f ms for %zu question(s)\n", ms,
               items.size());
  std::fflush(stderr);
  return v;
}

}  // namespace gufo::models::qwen38_flash_next
