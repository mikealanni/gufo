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

/// WMMA causal attention multiplied by softplus(gate[t][head]); `window` > 0
/// limits each query to its last `window` keys and `ring` > 0 wraps cache
/// rows. Returns false when the shapes are unsupported so the caller can fall
/// back to the scalar `Attention`.
///
/// When `partials` is non-null the key range is split round-robin over
/// `splits` blocks (grid.z), each writing a partial online-softmax state
/// `(m, l, acc[head_dim])` into `partials[n_tokens][heads][splits][head_dim +
/// 2]`; a merge kernel combines them. This is the decode-depth lever: without
/// it one block sweeps the whole KV range per (head, token) and TG falls off
/// with context. `partials` must be nulled for prefill.
///
/// When `pooled_keys` is non-null, block-sparse attention (LAGUNA_SPARSE_ATTN)
/// replaces the full contiguous key sweep beyond the sink + local window with
/// the `top_blocks` highest-scoring 64-key blocks (Quest bound against
/// `pooled_keys`'s per-block min/max, laid out as UpdatePooledKeysLaguna
/// writes it: `pooled_keys[(kv_head * nblk + block) * 256]`, 128 half-words of
/// per-channel max then 128 of min). Exact (no blocks dropped) whenever the
/// candidate block count is at most `top_blocks`; `window` must be 0 (sparse
/// is for the full-attention layers only, not the sliding-window ones).
bool AttentionWmma(const __half* q, const __half* k_cache,
                   const __half* v_cache, const float* gate, float* out,
                   const std::uint32_t* start_pos, std::uint32_t heads,
                   std::uint32_t kv_heads, std::uint32_t head_dim,
                   std::uint32_t window, std::uint32_t ring,
                   std::uint32_t n_tokens, float* partials,
                   std::uint32_t splits, hipStream_t stream,
                   const __half* pooled_keys = nullptr, std::uint32_t nblk = 0,
                   std::uint32_t top_blocks = 0);

/// Updates the per-64-key-block channel min/max pooled keys used by
/// AttentionWmma's sparse path, for blocks newly covered by
/// [from_pos, end_pos) of `k_cache` (absolute positions; `k_cache` must be a
/// contiguous, non-ring layer cache - sparse applies only to Laguna's
/// full-attention layers, which never use the SWA ring buffer). Call once per
/// (layer, bank) right after StoreKv, before AttentionWmma.
void UpdatePooledKeysLaguna(const __half* k_cache, __half* pooled_keys,
                            std::uint32_t from_pos, std::uint32_t end_pos,
                            std::uint32_t kv_heads, std::uint32_t head_dim,
                            std::uint32_t nblk, hipStream_t stream);

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

/// NPU staging: encodes `tokens` (a multiple of 128) rows of `x` (pitch ldx,
/// k_slice columns) into block-float `hi` and residual `lo` buffers for the
/// shared-expert GEMM. See kernels.hip.cpp's 2026-10-31 NPU comment - this is
/// a straight copy of Flint's NpuEncodeBoth, not Laguna-specific math.
void NpuEncodeBoth(const float* x, std::size_t ldx, std::uint32_t tokens,
                   std::uint32_t k_slice, void* hi, void* lo,
                   hipStream_t stream);

/// silu(gate) * up from the NPU GEMM's hi+residual C streams (gate and up,
/// each split into its own hi/lo pair), written as F32 rows of valid_cols.
void NpuSwigluCombine(const void* gate_hi, const void* up_hi,
                      const void* gate_lo, const void* up_lo, float* out,
                      std::uint32_t tokens, std::uint32_t cols,
                      std::uint32_t valid_cols, hipStream_t stream);

/// Staged C streams of an NPU GEMM: the high pass and the low (residual)
/// pass of up to kNpuMaxSlices 1024-column slices.
inline constexpr std::uint32_t kNpuMaxSlices = 9;
struct NpuStreams {
  const std::uint8_t* hi[kNpuMaxSlices];
  const std::uint8_t* lo[kNpuMaxSlices];
};

/// QKV projections (no activation): dequantizes hi+lo and sums, writing each
/// of `slices` 1024-wide chunks at its own offset into a `pitch`-wide output
/// row - `pitch` is the full Q, K or V row width.
void NpuDecodeFused(const NpuStreams& streams, std::uint32_t slices, float* out,
                    std::uint32_t tokens, std::uint32_t cols, std::size_t pitch,
                    hipStream_t stream);

}  // namespace gufo::models::laguna::gpu::rocm

#endif  // GUFO_MODELS_LAGUNA_GPU_KERNELS_KERNELS_HPP_
