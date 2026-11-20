#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MTP_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MTP_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

// Packs MTP (NEXTN) block inputs: for each row r,
//   concat[r][0:4096]    = RMS(embd[r]) * enorm_w
//   concat[r][4096:8192] = RMS(h_r)     * hnorm_w
// with `h_r` the row of `h` at index (r - skip) after the carry row: when
// `carry` is not null, row `carry_row` of the output takes `carry` instead of
// `h`, and rows after it read `h` one row ahead (carry_row is 0 in practice).
int MtpPack(const float* embd, const float* h, int carry_row,
            const float* carry, const float* enorm_w, const float* hnorm_w,
            float eps, float* concat, int rows, hipStream_t stream);

// out[row][i] = a[row][i] + b[row][i] over rows * n elements.
int MtpVecAdd(const float* a, const float* b, float* out, int n, int rows,
              hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MTP_HPP_
