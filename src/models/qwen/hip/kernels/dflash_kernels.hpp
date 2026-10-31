#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip::kernels {

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream);

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream);

/// out[row][head][d] *= softplus(gate[row][head]) for every d in
/// [0, head_dim) - the same per-head output gate Laguna's own scalar
/// AttentionKernel applies, matching `laguna.attn_gate.weight`'s role.
/// `attn_out` is [num_rows][num_heads * head_dim], `gate` is
/// [num_rows][num_heads].
void LaunchDFlashAttnGateMul(float* attn_out, const float* gate,
                             std::uint32_t num_rows, std::uint32_t num_heads,
                             std::uint32_t head_dim, hipStream_t stream);

/// gate[row][m] = sum_k bf16_weight[m][k] * x[row][k]. A plain, unoptimized
/// reduction kernel - deliberately not routed through the shared batched
/// BF16 GEMM dispatch, which faults (hipErrorMemoryFault, confirmed via the
/// HIP driver's own fault report) for this projection's odd M (72, the
/// drafter's head count) that path isn't validated for. This GEMM is tiny
/// (num_rows <= block_size, M = num_heads) so a naive kernel costs nothing
/// measurable.
void LaunchDFlashAttnGateProjection(const void* bf16_weight, const float* x,
                                    float* out, std::uint32_t num_rows,
                                    std::uint32_t m, std::uint32_t k,
                                    hipStream_t stream);

/// RMSNorm each of `num_taps` consecutive `hidden_size`-wide segments of a
/// [num_rows][num_taps][hidden_size] buffer independently, with its own
/// gamma column (`weight[tap * hidden_size + i]`, matching the GGUF
/// `enc.aux_norm.weight` [hidden_size, num_taps] layout) - distinct from
/// LaunchBatchedPerHeadRMSNorm, which shares one gamma vector across heads.
void LaunchDFlashTapRMSNorm(const float* x, const float* weight, float* out,
                            std::uint32_t num_rows, std::uint32_t num_taps,
                            std::uint32_t hidden_size, float eps,
                            hipStream_t stream);

void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Thread-per-key reference route of the call above. Exposed so an equivalence
/// test can run both routes in one process; the production entry point picks
/// between them.
void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Wave-per-key route of the call above. Requires `head_dim` to be a multiple
/// of four so a K row can be read as `float4`.
void LaunchDFlashNonCausalAttentionWave(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Quantizes a BF16 weight matrix into a Q8_0 copy for the draft-only LM head.
/// `total_elements` must be a multiple of the Q8_0 block size.
void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream);

[[nodiscard]] std::size_t DFlashSelectorScratchElements(
    std::uint32_t vocab_size) noexcept;

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream);

struct DFlashSelectorSequence {
  std::uint32_t count;
  float temperature;
};

// Proposal rows are contiguous by sequence. Each token sequence includes its
// anchor before the proposal slots; all other buffers contain proposals only.
// Partial buffers need DFlashSelectorScratchElements(vocab_size) per proposal.
// Requested candidates include one-hot probabilities for greedy sequences.
void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream, const float* tail_logits = nullptr,
    std::uint32_t draft_head_rows = 0, std::uint32_t draft_tail_rows = 0,
    std::uint32_t draft_tail_base = 0);

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_
