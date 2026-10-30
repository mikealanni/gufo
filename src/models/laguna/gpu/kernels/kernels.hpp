#ifndef GUFO_MODELS_LAGUNA_GPU_KERNELS_KERNELS_HPP_
#define GUFO_MODELS_LAGUNA_GPU_KERNELS_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Laguna-private HIP launchers. Activations are row-major float32
/// [tokens][dim]; every launch is asynchronous on `stream`. Positions are read
/// from device memory so a captured decode graph replays at any depth.
namespace gufo::models::laguna::gpu::rocm {

/// out[t] = dequantized Q8_0 row tokens[t] of the embedding table.
void EmbedQ8(const void* table, const std::int32_t* tokens, float* out,
             std::uint32_t n_tokens, std::uint32_t hidden, hipStream_t stream);

/// RMSNorm of `groups` equal slices per row; gamma (dim / groups floats) is
/// shared by every slice and may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// out[t][r] = dot(Q8_0 row r of w, x[t]) for a small row count.
void Q8Rows(const void* w, const float* x, float* out, std::uint32_t n_tokens,
            std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

/// NEOX rotation of the first n_dims of every head at position
/// *start_pos + t, in place. inv_freq holds n_dims / 2 effective frequencies;
/// cos and sin are scaled by mscale.
void RopeTable(float* x, std::uint32_t n_tokens, std::uint32_t heads,
               std::uint32_t head_dim, std::uint32_t n_dims,
               const float* inv_freq, float mscale,
               const std::uint32_t* start_pos, hipStream_t stream);

/// Stores K and V rows as F16 at cache row (*start_pos + t) % ring (no
/// modulo when ring is 0).
void StoreKv(const float* k, const float* v, __half* k_cache, __half* v_cache,
             std::uint32_t n_tokens, std::uint32_t row_dim,
             const std::uint32_t* start_pos, std::uint32_t ring,
             hipStream_t stream);

void ToHalf(const float* src, __half* dst, std::size_t count,
            hipStream_t stream);

bool AttentionWmma(const __half* q, const __half* k_cache,
                   const __half* v_cache, const float* gate, float* out,
                   std::uint32_t n_tokens, const std::uint32_t* start_pos,
                   std::uint32_t heads, std::uint32_t kv_heads,
                   std::uint32_t head_dim, std::uint32_t window,
                   std::uint32_t ring, hipStream_t stream);

void ToHalf(const float* src, __half* dst, std::size_t count,
            hipStream_t stream);

bool AttentionWmma(const __half* q, const __half* k_cache,
                   const __half* v_cache, const float* gate, float* out,
                   std::uint32_t n_tokens, const std::uint32_t* start_pos,
                   std::uint32_t heads, std::uint32_t kv_heads,
                   std::uint32_t head_dim, std::uint32_t window,
                   std::uint32_t ring, hipStream_t stream);

/// Causal attention multiplied by softplus(gate[t][head]). `window` > 0
/// limits each query to its last `window` keys; `ring` > 0 wraps cache rows.
void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* gate, float* out, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t head_dim,
               std::uint32_t window, std::uint32_t ring, hipStream_t stream);

/// out[t][e] = dot(w[e], x[t]) for F32 router weights.
void RouterF32(const float* w, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t n_experts, std::uint32_t k,
               hipStream_t stream);

/// Sigmoid gating; experts are picked by sigmoid + bias, weighted by the
/// plain sigmoid, optionally renormalized, then scaled.
void SigmoidBiasTopK(const float* logits, const float* bias, std::int32_t* ids,
                     float* weights, std::uint32_t n_tokens,
                     std::uint32_t n_experts, std::uint32_t k, bool normalize,
                     float scale, hipStream_t stream);

/// gate[i] = silu(gate[i]) * up[i].
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);

/// Swiglu that divides a row by a power of two when its maximum exceeds a
/// limit keeping Q8_1 half sums finite; scale[r] receives the divisor.
void SwigluScaled(float* gate, const float* up, float* scale,
                  std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

/// x[t] += sum_s weights[t][s] * expert_scale[t * k + s] * expert_out[t * k +
/// s]
/// + shared_scale[t] * shared[t]; either scale may be null (meaning 1).
void MoeCombine(float* x, const float* expert_out, const float* weights,
                const float* expert_scale, const float* shared,
                const float* shared_scale, std::uint32_t n_tokens,
                std::uint32_t k, std::uint32_t dim, hipStream_t stream);

/// out[t][i] = half(in[t][i]).
void NarrowRows(const float* in, __half* out, std::uint32_t n_rows,
                std::uint32_t k, std::uint32_t ldx, hipStream_t stream);

/// x[i] += y[i].
void AddInPlace(float* x, const float* y, std::size_t count,
                hipStream_t stream);

/// dst[r][tap * hidden + i] = x[first_row + r][i]; the drafter reads rows at
/// `stride` with the tapped hidden states concatenated per row.
void CaptureTap(float* dst, const float* x, std::uint32_t n_rows,
                std::uint32_t hidden, std::size_t stride,
                std::size_t tap_offset, hipStream_t stream);

}  // namespace gufo::models::laguna::gpu::rocm

#endif  // GUFO_MODELS_LAGUNA_GPU_KERNELS_KERNELS_HPP_
