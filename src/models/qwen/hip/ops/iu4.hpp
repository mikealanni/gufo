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
// cols a multiple of 256.
void LaunchIu4RepackWeights(const void* blocks, void* tiles,
                            std::uint16_t* scales,
                            unsigned long long* bad_zero_points,
                            std::size_t rows, std::size_t cols,
                            hipStream_t stream);

}  // namespace gufo::hip
#endif
#endif
