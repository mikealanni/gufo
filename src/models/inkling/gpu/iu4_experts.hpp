#ifndef GUFO_MODELS_INKLING_GPU_IU4_EXPERTS_HPP_
#define GUFO_MODELS_INKLING_GPU_IU4_EXPERTS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Rotated symmetric int4 x int4 WMMA path for Inkling's routed gate/up experts
// (opt-in, INKLING_IU4). Weights are repacked from the NVFP4 experts at load:
// a signed Hadamard rotation over 256, one scale per 128 K values, signed int4
// codes in the tile-major layout the GEMM streams straight into registers.
namespace gufo::models::inkling::iu4 {

inline constexpr int kFeatureTile = 64;  ///< gate features per CTA (64 up too)
inline constexpr int kTokenTile = 128;

/// 512 signs: 256 before the butterfly, 256 after.
void FillSigns(float* host_512);

/// Bytes of one expert's tile-major weights ([K/128][rows/16][4][256]) and its
/// scales ([K/128][rows] halves).
std::size_t WeightBytesPerExpert(std::size_t rows, std::size_t k);
std::size_t ScaleBytesPerExpert(std::size_t rows, std::size_t k);

/// nvfp4: [experts][rows][src_row_bytes] repacked NVFP4 rows. tiles/scales
/// receive the int4 form for every expert.
/// k_src real K values per row, zero padded to k (a multiple of 256).
void RepackExperts(const std::uint8_t* nvfp4, std::size_t src_row_bytes,
                   std::uint8_t* tiles, std::uint16_t* scales, int experts,
                   int rows, int k_src, int k, const float* signs,
                   hipStream_t stream);

/// Rotates and quantizes the activation row of every compact routed row
/// (rows_out < 0 are padding and become zero rows). tiles is tile-major
/// [row/16][K/128][1024 B]; scales [K/128][padded_rows] floats.
/// x rows are k_src halves; the quantized row is zero padded to k.
void QuantizeRoutedActs(const __half* x, const std::int32_t* rows_in,
                        const std::int32_t* rows_out,
                        const std::int32_t* pad_bounds, int experts,
                        int max_rows, int k_src, int k, std::uint8_t* tiles,
                        float* scales, int padded_rows, const float* signs,
                        hipStream_t stream);

/// y[slot][rows] = F16(expert_scale * down(act)) for every routed row; rows is
/// the output width (hidden size), k the padded K of tiles/act_tiles.
void DownGemm(const std::uint8_t* tiles, const std::uint16_t* scales,
              const std::uint8_t* act_tiles, const float* act_scales,
              int padded_rows, const std::int32_t* tile_map,
              std::uint32_t n_tiles, const std::int32_t* pad_bounds,
              const std::int32_t* rows_out, __half* y, int rows, int k,
              const float* expert_scale, hipStream_t stream);

/// act[slot][ff] = situ(gate, up) for every routed row, gate/up from the int4
/// GEMM with the expert scale applied. rows = 2 * ff rows per expert.
void GateUpSitu(const std::uint8_t* tiles, const std::uint16_t* scales,
                const std::uint8_t* act_tiles, const float* act_scales,
                int padded_rows, const std::int32_t* tile_map,
                std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                const std::int32_t* rows_out, __half* act, int ff, int k,
                const float* expert_scale, const float* situ,
                hipStream_t stream);

}  // namespace gufo::models::inkling::iu4

#endif  // GUFO_MODELS_INKLING_GPU_IU4_EXPERTS_HPP_
