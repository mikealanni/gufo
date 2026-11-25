#ifndef GUFO_MODELS_INKLING_GPU_GPTQ_HPP_
#define GUFO_MODELS_INKLING_GPU_GPTQ_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstddef>
#include <cstdint>

// Offline GPTQ for the rotated symmetric int4 experts: the same rotation,
// per-128 scales, clip search and tile layout as iu4::RepackExperts, with the
// rounding error of each column compensated through the inverse Hessian of the
// expert's own inputs.
namespace gufo::models::inkling::gptq {

inline constexpr int kGroup = 128;

/// Rotated fp32 weights of `rows` NVFP4 rows (zero past k_src, padded to k).
void RotateWeights(const std::uint8_t* nvfp4, std::size_t src_row_bytes,
                   float* out, int rows, int k_src, int k, const float* signs,
                   hipStream_t stream);

/// Rotates compact rows of x (halves, k_src wide, token index from rows_in or
/// the row itself when null; rows with rows_out < 0 become zeros) into out
/// (floats, k wide).
void RotateRows(const __half* x, const std::int32_t* rows_in,
                const std::int32_t* rows_out, int n_rows, int k_src, int k,
                float* out, const float* signs, hipStream_t stream);

/// h (k x k, lower triangle, column major) += rows^T rows for m rows of x.
bool AccumulateHessian(rocblas_handle blas, const float* x, int m, int k,
                       float* h);

/// Scratch for one matrix solve (sized for rows x k).
struct Scratch {
  float* w{nullptr};          ///< working weights [rows][k]
  float* hinv{nullptr};       ///< [k][k]
  float* ublock{nullptr};     ///< [128][128]
  float* err{nullptr};        ///< [rows][128]
  float* scale{nullptr};      ///< [rows]
  std::int8_t* codes{nullptr};  ///< [rows][k]
  std::uint16_t* gscales{nullptr};  ///< [rows][k/128] half bits
  int* info{nullptr};
  int rows{0}, k{0};
};
bool AllocScratch(Scratch& s, int rows, int k);
void FreeScratch(Scratch& s);

/// GPTQ of w_rot (rows x k, row major) with the lower triangle of h (k x k,
/// column major, accumulated by AccumulateHessian). k_src real columns.
/// Writes int4 codes and per-group half scales into the scratch.
bool Quantize(rocblas_handle blas, Scratch& s, const float* w_rot, float* h,
              int k_src, float damp, hipStream_t stream);

/// Plain rounding (no Hessian), for tests and rare experts.
void QuantizeRtn(Scratch& s, const float* w_rot, hipStream_t stream);

/// Tile-major weights and scales in the iu4 layout for one expert.
void Pack(const Scratch& s, std::uint8_t* tiles, std::uint16_t* scales,
          hipStream_t stream);

}  // namespace gufo::models::inkling::gptq

#endif  // GUFO_MODELS_INKLING_GPU_GPTQ_HPP_
