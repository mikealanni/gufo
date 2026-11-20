#ifndef GUFO_MODELS_QWEN_HIP_OPS_IU4_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_IU4_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

// Bytes of one 256-wide K block of a rotated symmetric 4-bit projection: two
// (f16 scale, f16 zero point) pairs, then 128 payload bytes.
inline constexpr std::size_t kIu4BlockBytes = 136;

// Relayouts one projection [rows x cols] stored as 136-byte blocks into the
// tile-major form the int4 WMMA GEMM reads: for every 128-wide K group e and 16
// row tile, four 256-byte chunks of 16 rows x 16 payload bytes with the nibbles
// converted to signed codes. scales[e * rows + row] receives the f16 group
// scale. bad_zero_points counts groups whose zero point is not -8 scales (those
// groups cannot use the symmetric kernel); rows must be a multiple of 16 and
// cols a multiple of 256. For Gated DeltaNet projections the artifact groups
// value heads by key head while the engine tiles them: when perm_chunk is not
// zero, rows from perm_start on are taken in chunks of perm_chunk rows (one
// head) and engine chunk e reads artifact chunk (e % 16) * 3 + e / 16.
void LaunchIu4RepackWeights(const void* blocks, void* tiles,
                            std::uint16_t* scales,
                            unsigned long long* bad_zero_points,
                            std::size_t rows, std::size_t cols,
                            std::uint32_t perm_start, std::uint32_t perm_chunk,
                            hipStream_t stream);

// Rotation sign vectors the artifact was built with: 256 signs for the pre
// butterfly diagonal followed by 256 for the post butterfly diagonal.
void FillIu4RotationSigns(float* host_signs_512);

// Norm-fed activation producer. For every token row of x [n x k] (f32, k a
// multiple of 256 and at most 8192): RMSNorm with norm_weight, divide by awq
// (null for none), rotate by the signed Hadamard transform over 256 and
// quantize each 128-wide group to signed int4 with a float scale. Output is the
// tile-major layout the int4 GEMM reads: tokens in tiles of 16, for every 128
// group a 1024-byte block holding 16 lanes x 32 bytes, and scales stored
// [group][padded token]. Rows from n to padded_n are written as zero codes with
// scale zero so the GEMM can run on a multiple of 128 tokens.
void LaunchIu4NormQuantize(const float* x, const float* norm_weight,
                           const float* awq, const float* signs, float eps,
                           std::size_t n, std::size_t padded_n, std::size_t k,
                           std::uint8_t* tiles, float* scales,
                           hipStream_t stream);

// True when the int4 GEMM can run a projection of this shape: rows must be a
// multiple of 128 (one feature tile) and cols a multiple of 256.
bool Iu4GemmSupported(std::size_t rows, std::size_t cols);

// out[t][f] = (or +=) sum over K of the rotated int4 weights times the int4
// activations, in fp32, for t < n and f in [row_begin, rows). act_tiles and
// act_scales come from the activation producers with padded_n a multiple of
// 128; the weights come from LaunchIu4RepackWeights. out is token-major
// [n x rows]; the columns below row_begin are not touched (the NPU computes
// them). row_begin and rows must be multiples of 128.
void LaunchIu4Gemm(const void* tiles, const std::uint16_t* weight_scales,
                   const std::uint8_t* act_tiles, const float* act_scales,
                   float* out, bool accumulate, std::size_t rows,
                   std::size_t row_begin, std::size_t n, std::size_t padded_n,
                   std::size_t k, hipStream_t stream);

// Scalar fallback for projections that fail Iu4GemmSupported (rows at most 256
// and a divisor of 256): same inputs and outputs as LaunchIu4Gemm.
void LaunchIu4GemmSmall(const void* tiles, const std::uint16_t* weight_scales,
                        const std::uint8_t* act_tiles, const float* act_scales,
                        float* out, bool accumulate, std::size_t rows,
                        std::size_t n, std::size_t padded_n, std::size_t k,
                        hipStream_t stream);

// Swaps the two forms of a [rows x ld] activation whose first `columns`
// columns exist only as FP16 (from the NPU) and the rest only as FP32 (from the
// GPU): afterwards the FP32 matrix holds the converted NPU columns and the FP16
// matrix holds the converted GPU columns, so both are complete.
void LaunchIu4SwapHalfColumns(float* act_f32, void* act_f16, std::size_t rows,
                              std::size_t ld, std::size_t columns,
                              hipStream_t stream);

// Quantize-only activation producer for rows that are already normalized:
// divide by awq (null for none), rotate, quantize, same output layout as
// LaunchIu4NormQuantize. k is a multiple of 256, any size. With remap_gdn_heads
// the 128-wide chunks of the artifact-order row are read from the engine's
// tiled value-head order (the Gated DeltaNet output projection input).
void LaunchIu4Quantize(const float* x, const float* awq, const float* signs,
                       bool remap_gdn_heads, std::size_t n,
                       std::size_t padded_n, std::size_t k, std::uint8_t* tiles,
                       float* scales, hipStream_t stream);

}  // namespace gufo::hip
#endif
#endif
