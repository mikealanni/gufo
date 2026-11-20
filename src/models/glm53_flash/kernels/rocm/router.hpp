#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_ROUTER_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_ROUTER_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

// hi = F16(x * scale), lo = F16(x * scale - hi) for n values.
int SplitF16(const float* in, std::uint16_t* hi, std::uint16_t* lo, long long n,
             hipStream_t stream, float scale = 1.0F);

// out[t][e] = sum_k y[t][k] * w[e][k] over 288 experts and K = 4096, with the
// F32 router matrix given as its F16 hi / lo split and y split on the fly
// (three matrix products per tile, agreement with the F32 product ~1e-7).
int RouterGemm(const float* y, const std::uint16_t* w_hi,
               const std::uint16_t* w_lo, float* out, int tokens,
               int num_experts, hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_ROUTER_HPP_
