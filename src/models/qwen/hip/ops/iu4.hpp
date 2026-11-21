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
                            std::size_t rows, std::size_t padded_rows,
                            std::size_t cols, std::uint32_t perm_start,
                            std::uint32_t perm_chunk, hipStream_t stream);

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
// scale zero so the GEMM can run on a multiple of 128 tokens. When half_out is
// set the normalized, pre-awq row is also written there as FP16 (row pitch k),
// the input the NPU reads.
void LaunchIu4NormQuantize(const float* x, const float* norm_weight,
                           const float* awq, const float* signs, float eps,
                           std::size_t n, std::size_t padded_n, std::size_t k,
                           std::uint8_t* tiles, float* scales,
                           hipStream_t stream, void* half_out = nullptr,
                           float* rot_out = nullptr);

// True when the int4 GEMM can run a projection of this shape: rows must be a
// multiple of 128 (one feature tile) and cols a multiple of 256.
bool Iu4GemmSupported(std::size_t rows, std::size_t cols);

// out[t][f] = (or +=) sum over K of the rotated int4 weights times the int4
// activations, in fp32, for t < n and f in [row_begin, rows). act_tiles and
// act_scales come from the activation producers with padded_n a multiple of
// 128; the weights come from LaunchIu4RepackWeights. out is token-major
// [n x rows]; the columns below row_begin are not touched (the NPU computes
// them). row_begin and rows must be multiples of 128. Projections with fewer
// rows are stored padded with zero rows to 128: pass the padded count as rows,
// the logical count as valid_rows and the real row pitch as out_ld (zero means
// rows).
void LaunchIu4Gemm(const void* tiles, const std::uint16_t* weight_scales,
                   const std::uint8_t* act_tiles, const float* act_scales,
                   float* out, bool accumulate, std::size_t rows,
                   std::size_t row_begin, std::size_t n, std::size_t padded_n,
                   std::size_t k, hipStream_t stream, std::size_t out_ld = 0,
                   std::size_t valid_rows = 0);

// Decode GEMV for `tokens` rows (x is [tokens][k], out [tokens][valid_rows],
// weights read once per 16 tokens; more than four tokens use the matrix
// cores when x16_scratch (16 x k halves) is given): out[f] (+)= sum over K of
// the rotated int4 weights times the rotated fp32 activation x (producers:
// rotate-only variants). Padded projections pass the padded row count and the
// logical valid_rows.
void LaunchIu4Gemv(const void* tiles, const std::uint16_t* weight_scales,
                   const float* x, float* out, bool accumulate,
                   std::size_t padded_rows, std::size_t valid_rows,
                   std::size_t k, hipStream_t stream, std::size_t tokens = 1,
                   void* x16_scratch = nullptr);

// NPU glue for the SwiGLU product when the NPU also takes down columns: the
// leading `columns` columns exist only as FP16 (from the NPU), the rest only as
// FP32 (from the GPU). The first call completes the FP32 matrix; the second
// builds the FP16 matrix the NPU reads for the down projection.
void LaunchIu4HalfColumnsToF32(float* act_f32, const void* act_f16,
                               std::size_t rows, std::size_t ld,
                               std::size_t columns, hipStream_t stream);
void LaunchIu4F32ColumnsToHalf(const float* act_f32, void* act_f16,
                               std::size_t rows, std::size_t ld,
                               std::size_t first, hipStream_t stream);

// Gate and up outputs of the NPU's leading FFN slices in its block-float stream
// layout (9 bytes per group of 8 columns: shared exponent, 8 int8 mantissas;
// 512 x 128 tiles of 2 x 2 super blocks of 8 x 8 blocks; 1024 columns per
// slice). The SwiGLU quantizer decodes these directly.
inline constexpr std::uint32_t kIu4NpuMaxSlices = 16;
struct Iu4NpuSwigluSource {
  const void* gate[kIu4NpuMaxSlices];
  const void* up[kIu4NpuMaxSlices];
  std::uint32_t slices{0};
};

// SwiGLU fused into the quantizer for the down projection input: the row is
// silu(gate) * up for columns from npu_columns on and the FP16 SwiGLU product
// of the NPU (npu_half, row pitch k) below that. Same output layout as the
// others.
void LaunchIu4SwiGluQuantize(const float* gate, const float* up,
                             const void* npu_half,
                             const Iu4NpuSwigluSource* npu_source,
                             std::size_t npu_columns, const float* awq,
                             const float* signs, std::size_t n,
                             std::size_t padded_n, std::size_t k,
                             std::uint8_t* tiles, float* scales,
                             hipStream_t stream, float* rot_out = nullptr);

// Gated DeltaNet epilogue fused into the quantizer of the out projection: per
// (token, head) RMSNorm of `raw` with `ssm_norm`, times silu(gate), through the
// artifact head order (see LaunchIu4Quantize), then awq divide, rotate,
// quantize.
void LaunchIu4PostNormGateQuantize(const float* raw, const float* ssm_norm,
                                   const float* gate, const float* awq,
                                   const float* signs, std::size_t n,
                                   std::size_t padded_n, std::size_t inner,
                                   std::uint8_t* tiles, float* scales,
                                   hipStream_t stream,
                                   float* rot_out = nullptr);

// Quantize-only activation producer for rows that are already normalized:
// divide by awq (null for none), rotate, quantize, same output layout as
// LaunchIu4NormQuantize. k is a multiple of 256, any size. With remap_gdn_heads
// the 128-wide chunks of the artifact-order row are read from the engine's
// tiled value-head order (the Gated DeltaNet output projection input).
void LaunchIu4Quantize(const float* x, const float* awq, const float* signs,
                       bool remap_gdn_heads, std::size_t n,
                       std::size_t padded_n, std::size_t k, std::uint8_t* tiles,
                       float* scales, hipStream_t stream,
                       float* rot_out = nullptr);

// Every producer takes an optional rot_out: when set, the rotated (and awq
// divided) fp32 row is written there instead of int4 codes, for the decode
// GEMV. Use n = padded_n = 1 and tiles/scales may be null.

}  // namespace gufo::hip
#endif
#endif
