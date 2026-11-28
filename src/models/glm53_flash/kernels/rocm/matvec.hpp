#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MATVEC_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MATVEC_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

// ggml type codes accepted by Matvec.
inline constexpr int kTypeF32 = 0;
inline constexpr int kTypeQ8_0 = 8;
inline constexpr int kTypeQ3_K = 11;
inline constexpr int kTypeQ4_K = 12;
inline constexpr int kTypeQ5_K = 13;
inline constexpr int kTypeQ6_K = 14;
inline constexpr int kTypeIQ2_XXS = 16;
inline constexpr int kTypeIQ2_S = 22;

// Device scratch the quantized-activation path needs for n columns of k.
std::size_t MatvecScratchBytes(int n, int k);

// out[col * m + row] = sum_k W[row, k] * x[col * k_dim + k] for n <= 8
// columns. W is a device tensor in its ggml block format with rows of k
// elements. Quantized types go through the vendored mmvq kernels; F32 uses a
// plain row-per-block kernel. scratch must hold MatvecScratchBytes(n, k).
// Returns 0 on success.
int Matvec(int ggml_type, const void* w, const float* x, float* out, int m,
           int n, int k, void* scratch, hipStream_t stream);

// Quantize n activation rows once (q holds MatvecScratchBytes(n, k)) and run
// several matrices against them with MatvecQuantized.
int QuantizeActivations(const float* x, int n, int k, void* q, hipStream_t stream);
int MatvecQuantized(int ggml_type, const void* w, const void* q, float* out,
                    int m, int n, int k, hipStream_t stream);

// Per-channel matvec: out[c * m + row] = sum_k W[c][row][k] * x[c * k + k'],
// for `channels` stacked weight slabs of m rows each (attn_k_b / attn_v_b
// heads). scratch must hold MatvecScratchBytes(channels, k). Returns 0 on
// success.
int MatvecBatched(int type_code, const void* w, const float* x, float* out,
                  int m, int k, int channels, void* scratch,
                  hipStream_t stream);

// MatvecBatched for up to eight tokens in one launch: x is [token][channel][k]
// and out is [token][channel][m], with the weight slab chosen by channel.
int MatvecBatchedTokens(int type_code, const void* w, const float* x,
                        float* out, int m, int k, int channels, int tokens,
                        void* scratch, hipStream_t stream);

// Routed-expert matvec: out[i * m + row] = sum_k W[ids[i]][row][k] * x_i[k]
// for i < n_used, with W stacked [n_experts][m][k] in its block format. x is
// one [k] row shared by all experts, or [n_used][k] when x_per_expert.
// scratch must hold MatvecScratchBytes(x_per_expert ? n_used : 1, k).
int MatvecExperts(int type_code, const void* w, const float* x, float* out,
                  int m, int k, int n_experts, int n_used, const int* ids,
                  bool x_per_expert, void* scratch, hipStream_t stream);
// MatvecExperts for up to eight tokens in one launch: ids is [token][n_used],
// x is [token][k] (or [token][n_used][k] when x_per_expert) and out is
// [token][n_used][m].
int MatvecExpertsTokens(int type_code, const void* w, const float* x,
                        float* out, int m, int k, int n_experts, int n_used,
                        int n_tokens, const int* ids, bool x_per_expert,
                        void* scratch, hipStream_t stream);

// Two same-shaped expert tensors (gate and up) applied to one activation: the
// activation is quantized once and both run in one launch. x_per_expert is
// false; out0 and out1 are laid out like MatvecExpertsTokens' out.
int MatvecExpertsPairTokens(int type_code, const void* w0, const void* w1,
                            const float* x, float* out0, float* out1, int m,
                            int k, int n_experts, int n_used, int n_tokens,
                            const int* ids, void* scratch, hipStream_t stream);

// Dense GEMM for any n: out[col * m + row], x is [n][k]. Up to 8 columns use
// the mmvq path, larger batches the mmq GEMM (K a multiple of 256; narrower K
// falls back to chunks of 8 columns). scratch must hold
// MatvecScratchBytes(min(n, 8), k). Returns 0 on success.
int Gemm(int type_code, const void* w, const float* x, float* out, int m, int n,
         int k, void* scratch, hipStream_t stream);

// Grouped routed-expert GEMM over `tokens` rows of x ([tokens][k]), ids
// [tokens][n_used]; out[(token * n_used + slot) * m + row]. IQ2_XXS / IQ2_S.
int ExpertsGemm(int type_code, const void* w, const float* x, float* out, int m,
                int k, int tokens, int n_experts, int n_used, const int* ids,
                hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MATVEC_HPP_
