#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_KDA_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_KDA_HPP_

#include <hip/hip_runtime.h>

namespace gufo::models::glm53_flash::rocm {

inline constexpr int kKdaHeads = 64;
inline constexpr int kKdaHeadDim = 128;
inline constexpr int kKdaInner = kKdaHeads * kKdaHeadDim;
inline constexpr int kKdaConvTaps = 4;
// Kimi Delta Attention pieces for `tokens` consecutive tokens (1 = decode).
// q, k, v, f, decay, o and g are [tokens][8192]; beta and braw [tokens][64].
//   hist    [3][3][8192] last three raw inputs per stream, oldest first
//   state   [64][128][128] delta state S[head][value row][key dim]
// Conv weights are the ggml [4,1,8192] tensors (conv_w[ch * 4 + tap]).

// q, k and v hold the raw projections and are replaced in place by
// SiLU(causal conv); hist carries across calls.
int KdaConv(float* q, float* k, float* v, float* hist, const float* conv_q,
            const float* conv_k, const float* conv_v, hipStream_t stream,
            int tokens = 1);

int KdaGates(const float* f, const float* dt, const float* ssm_a,
             const float* braw, float* decay, float* beta, hipStream_t stream,
             int tokens = 1);

// L2-normalizes q and k per head, applies the delta-rule update token by token
// and writes o = S q / sqrt(128).
int KdaRecur(const float* q, const float* k, const float* v, const float* decay,
             const float* beta, float* state, float* o, hipStream_t stream,
             int tokens = 1);

// In place per head: o = RMSNorm(o) * norm_w * sigmoid(g).
int KdaOut(float* o, const float* norm_w, const float* g, hipStream_t stream,
           int tokens = 1);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_KDA_HPP_
