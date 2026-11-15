#ifndef GUFO_NPU_RMSNORM_REF_HPP
#define GUFO_NPU_RMSNORM_REF_HPP

#include <cstdint>
#include <cmath>
#include <vector>

#include "src/npu/runtime/atb_layout.hpp"

namespace gufo::npu::rmsnorm {

inline void rmsnorm_rows(const float* x, const float* weight, float* out,
                         int rows, int hidden, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + std::size_t(r) * hidden;
    float sum = 0;
    for (int i = 0; i < hidden; ++i) sum += xr[i] * xr[i];
    const float scale = 1.0f / std::sqrt(sum / hidden + eps);
    float* o = out + std::size_t(r) * hidden;
    for (int i = 0; i < hidden; ++i) o[i] = xr[i] * scale * weight[i];
  }
}

inline std::vector<uint8_t> encode_a_wire(const float* z, int rows, int hidden) {
  const int tile_m = 128, tile_k = 64;
  const int l1_rows = (rows + tile_m - 1) / tile_m;
  const int l1_cols = hidden / tile_k;
  const int tile_bytes = tile_m * tile_k / 8 * 9;
  std::vector<uint8_t> stream(std::size_t{l1_rows} * l1_cols * tile_bytes);

  for (int l1_row = 0; l1_row < l1_rows; ++l1_row) {
    for (int l1_col = 0; l1_col < l1_cols; ++l1_col) {
      uint8_t* tile = stream.data() +
                      (std::size_t{l1_row} * l1_cols + l1_col) * tile_bytes;
      for (int sbr = 0; sbr < 8; ++sbr) {
        for (int sbc = 0; sbc < 8; ++sbc) {
          for (int bis = 0; bis < 2; ++bis) {
            uint8_t* blk = tile +
                           ((sbr * 8 + sbc) * 2 + bis) * 72;
            const int row0 = l1_row * tile_m + sbr * 16 + bis * 8;
            const int col0 = l1_col * tile_k + sbc * 8;
            for (int rib = 0; rib < 8; ++rib) {
              std::vector<float> block(8);
              for (int i = 0; i < 8; ++i) {
                int r = row0 + rib;
                int c = col0 + i;
                block[i] = (r < rows && c < hidden) ? z[r * hidden + c] : 0.0f;
              }
              auto encoded = atb::floatToBfp16(8, 8, block.data());
              std::memcpy(blk + rib * 9, encoded.data(), 9);
            }
          }
        }
      }
    }
  }
  return stream;
}

}  // namespace gufo::npu::rmsnorm

#endif  // GUFO_NPU_RMSNORM_REF_HPP
