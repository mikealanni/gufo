#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MOE_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MOE_HPP_

#include <hip/hip_runtime.h>

namespace gufo::models::glm53_flash::rocm {

// Largest expert count any GLM-5.3 artifact ships; the router sizes its
// per-expert shared arrays from this and takes the live count as an argument,
// so the 152-expert RCO export and the 288-expert artifacts share one kernel.
inline constexpr int kMaxMoeExperts = 288;
inline constexpr int kMoeUsed = 8;
inline constexpr int kMoeHidden = 4096;

// act[i] = min(gate, 10) * sigmoid(min(gate, 10)) * clamp(up, -10, 10).
int SwigluClamp(const float* gate, const float* up, float* act, int n,
                hipStream_t stream);

// Router: p = sigmoid(logits); the 8 experts with the largest p + bias are
// chosen (ties to the lower index); weights = p / sum(p of chosen) * scale.
// logits is [tokens][experts]; the router runs one block per token.
int MoeRoute(const float* logits, const float* bias, float scale, int* ids,
             float* weights, int experts, hipStream_t stream, int tokens = 1);

// out[c] = sum_i weights[i] * parts[i][c] + shared[c]. Batched calls advance
// every pointer by one token's worth per index (ids/weights 8,
// parts 8 * 4096, shared/out 4096).
int MoeCombine(const float* parts, const float* weights, const float* shared,
               float* out, hipStream_t stream, int tokens = 1);

// dst[i] = half(src[i]).
int ToHalf(const float* src, void* dst, long long n, hipStream_t stream);

// MoeCombine over F16 expert rows: parts is [tokens][8][4096] half.
int MoeCombineF16(const void* parts, const float* weights, const float* shared,
                  float* out, hipStream_t stream, int tokens);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MOE_HPP_
