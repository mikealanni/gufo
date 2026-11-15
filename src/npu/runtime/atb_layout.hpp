//===- atb_layout.hpp -------------------------------------------*- C++ -*-===//
// Asymmetric-tile-buffering (ATB) data layout + block-float helpers for the
// big-M prefill GEMM path (AGENT-7).
//
// Layouts (from the reference ATB designs):
//   A (activations): L1 tiles in row-major order over (M, K), each tile
//     2x1 vertically-stacked super-blocks of 8x8 row-major sub-blocks.
//   B (weights):     L1 tiles column-major over (K, N), each tile 1x2
//     horizontally-stacked super-blocks of 8x8 column-major sub-blocks.
//   C (output):      2x2 super-blocks of 8x8 blocks per L1 tile, tiles
//     row-major over (M, N).
// Both A and B are then encoded as v8bfp16ebs8 (9 bytes per 8 values: 1 shared
// exponent + 8x8-bit mantissas).
//
// Adapted from the reference implementations:
//   gemm_atb_layout.h (Apache-2.0, (c) 2026 Advanced Micro Devices, Inc.)
//   helper.h floatToBfp16/bfp16ebs8ToFloat (same license)
//===----------------------------------------------------------------------===//
#ifndef GUFO_NPU_ATB_LAYOUT_HPP
#define GUFO_NPU_ATB_LAYOUT_HPP

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gufo::npu::atb {

inline std::vector<uint8_t> floatToBfp16(int block, int size, float* array,
                                         int rounding = 0) {
  std::vector<uint8_t> res(size * 1.125);

  int mbits = 7;
  int start = 0, end, i, currentIndex = 1;
  unsigned int sign, exp, maxExp;
  unsigned int *p, mantissa;
  uint8_t valueInt8;

  while (true) {
    // decide on the block (starting and ending point)
    end = start + block;
    end = end > size ? size : end;

    // Find max exp
    maxExp = 0;
    for (i = start; i < end; i++) {
      p = (unsigned int*)(array + i);
      exp = *p >> 23;     // Get rid of mantissa
      exp &= 0x000000FF;  // Keep the last 8 bit exponent (remove sign)

      maxExp = maxExp < exp ? exp : maxExp;
    }

    // Round each number
    for (i = start; i < end; i++) {
      p = (unsigned int*)(array + i);

      sign = *p & 0x80000000;      // Sign
      exp = *p >> 23;              // Get rid of mantissa
      exp &= 0x000000FF;           // Keep the last 8 bit exponent (remove sign)
      mantissa = *p & 0x007FFFFF;  // 23-bit mantissa
      if (exp)
        mantissa |= 0x00800000;  // add the implicit for normal value

      if (exp >= 255)
        continue;  // Infinity or NaN remains

      // The rouding mode for the mantissa in AIE2p is always truncation
      // Each scalar value is stored in two's complement representation
      mantissa = sign ? ~mantissa + 1 : mantissa;
      // At least erase 23 - mbits + 1 (+1 is for making the implicit bit
      // explicit)
      valueInt8 = mantissa >> (23 - mbits + 1);

      // Note that shifting by more than 32 bits is undefined behavior in C++
      if (maxExp - exp >= 32) {
        valueInt8 = sign ? 0xff : 0x00;
      } else {
        // Perform an arithmetic right shift
        // Again, the rounding mode is truncation for AIE2p
        valueInt8 = static_cast<int8_t>(valueInt8) >> (maxExp - exp);
      }

      res[currentIndex] = valueInt8;
      currentIndex++;
    }
    res[currentIndex - 9] = (uint8_t)maxExp;
    currentIndex++;
    start = end;
    if (start >= size)
      break;
  }

  return res;
}

inline std::vector<float> bfp16ebs8ToFloat(int size, uint8_t* array,
                                           int verbose = 0) {
  std::vector<float> res(size / 1.125);

  int block = 8;
  int tempIndx = 0;
  for (int i = 0; i < size; i += block + 1) {
    uint8_t sharedExponent = (uint8_t)array[i];
    float multiplier;
    if (sharedExponent >= 127) {
      multiplier = 1.0 * (1 << (sharedExponent - 127));
    } else {
      multiplier = 1.0 / (1 << (127 - sharedExponent));
    }
    multiplier /= 64.0;
    if (verbose) {
      printf("shared_exponent = %d\n", sharedExponent);
      printf("multiplier = %f\n", multiplier);
    }
    for (int j = 1; j < block + 1; j++) {
      bool negative = array[i + j] & 0x80;
      if (negative) {
        // Two's complement for negative numbers
        uint8_t decoded = ~(array[i + j] - 1);
        res[tempIndx] = float(decoded) * multiplier;
      } else {
        res[tempIndx] = float(array[i + j] * multiplier);
      }
      res[tempIndx] = negative ? -res[tempIndx] : res[tempIndx];
      if (verbose) {
        printf("return_array[%d] = %f\n", tempIndx, res[tempIndx]);
      }
      tempIndx++;
    }
  }

  return res;
}

inline std::vector<float> layout_transpose_1x2_8x8block(
    const std::vector<float>& input, int rows, int cols) {
  assert(rows % 8 == 0 && "rows must be divisible by 8");
  assert(cols % 8 == 0 && "cols must be divisible by 8");
  assert((cols / 8) % 2 == 0 && "cols/8 must be divisible by 2 for 1x2 layout");
  std::vector<float> output(rows * cols);
  int block_rows = rows / 8;
  int block_cols = cols / 8;
  int output_idx = 0;
  // Iterate 1x2 super-blocks (two horizontally-stacked 8x8 blocks).
  for (int super_block_col = 0; super_block_col < block_cols;
       super_block_col += 2) {
    for (int super_block_row = 0; super_block_row < block_rows;
         super_block_row++) {
      for (int block_in_super = 0;
           block_in_super < std::min(2, block_cols - super_block_col);
           block_in_super++) {
        int current_block_row = super_block_row;
        int current_block_col = super_block_col + block_in_super;
        // Within each 8x8 block: column-major.
        for (int col_in_block = 0; col_in_block < 8; col_in_block++) {
          for (int row_in_block = 0; row_in_block < 8; row_in_block++) {
            int orig_row = current_block_row * 8 + row_in_block;
            int orig_col = current_block_col * 8 + col_in_block;
            int orig_idx = orig_row * cols + orig_col;
            output[output_idx++] = input[orig_idx];
          }
        }
      }
    }
  }
  return output;
}

inline std::vector<float> layout_transpose_L1_1x2_8x8block(
    const std::vector<float>& input, int rows, int cols, int L1_block_k,
    int L1_block_n) {
  assert(rows % L1_block_k == 0 && "rows must be divisible by L1_block_k");
  assert(cols % L1_block_n == 0 && "cols must be divisible by L1_block_n");
  assert(L1_block_k % 8 == 0 && "L1_block_k must be divisible by 8");
  assert(L1_block_n % 8 == 0 && "L1_block_n must be divisible by 8");
  assert((L1_block_n / 8) % 2 == 0 &&
         "L1_block_n/8 must be divisible by 2 for the 1x2 layout");

  std::vector<float> output(rows * cols);
  int L1_rows = rows / L1_block_k;
  int L1_cols = cols / L1_block_n;
  int output_idx = 0;

  // Outer order: column-major over L1 tiles.
  for (int L1_col = 0; L1_col < L1_cols; L1_col++) {
    for (int L1_row = 0; L1_row < L1_rows; L1_row++) {
      // Extract the current L1 tile.
      std::vector<float> tile(L1_block_k * L1_block_n);
      for (int i = 0; i < L1_block_k; i++) {
        for (int j = 0; j < L1_block_n; j++) {
          int orig_row = L1_row * L1_block_k + i;
          int orig_col = L1_col * L1_block_n + j;
          tile[i * L1_block_n + j] = input[orig_row * cols + orig_col];
        }
      }
      // Inner shuffle for this tile.
      std::vector<float> shuffled =
          layout_transpose_1x2_8x8block(tile, L1_block_k, L1_block_n);
      for (size_t k = 0; k < shuffled.size(); k++) {
        output[output_idx++] = shuffled[k];
      }
    }
  }
  return output;
}

inline std::vector<float> layout_A_2x1_8x8block(const std::vector<float>& input,
                                                int rows, int cols) {
  assert(rows % 8 == 0 && "rows must be divisible by 8");
  assert(cols % 8 == 0 && "cols must be divisible by 8");
  std::vector<float> output(rows * cols);
  int block_rows = rows / 8;
  int block_cols = cols / 8;
  assert(block_rows % 2 == 0 &&
         "block_rows must be divisible by 2 for 2x1 layout");
  int output_idx = 0;
  for (int super_block_row = 0; super_block_row < block_rows;
       super_block_row += 2) {
    for (int super_block_col = 0; super_block_col < block_cols;
         super_block_col++) {
      // 2x1 super-block: two 8x8 sub-blocks stacked vertically.
      for (int block_in_super = 0; block_in_super < 2; block_in_super++) {
        int cbr = super_block_row + block_in_super;
        int cbc = super_block_col;
        // 8x8 sub-block: row-major.
        for (int rib = 0; rib < 8; rib++) {
          for (int cib = 0; cib < 8; cib++) {
            int orig_row = cbr * 8 + rib;
            int orig_col = cbc * 8 + cib;
            output[output_idx++] = input[orig_row * cols + orig_col];
          }
        }
      }
    }
  }
  return output;
}

inline std::vector<float> layout_A_L1_2x1_8x8block(
    const std::vector<float>& input, int rows, int cols, int L1_block_m,
    int L1_block_k) {
  assert(rows % L1_block_m == 0 && "rows must be divisible by L1_block_m");
  assert(cols % L1_block_k == 0 && "cols must be divisible by L1_block_k");
  assert(L1_block_m % 8 == 0 && "L1_block_m must be divisible by 8");
  assert(L1_block_k % 8 == 0 && "L1_block_k must be divisible by 8");
  assert((L1_block_m / 8) % 2 == 0 &&
         "L1_block_m/8 must be divisible by 2 for the 2x1 layout");
  std::vector<float> output(rows * cols);
  int L1_rows = rows / L1_block_m;
  int L1_cols = cols / L1_block_k;
  int output_idx = 0;
  // Outer L1 traversal: row-major.
  for (int L1_row = 0; L1_row < L1_rows; L1_row++) {
    for (int L1_col = 0; L1_col < L1_cols; L1_col++) {
      std::vector<float> tile(L1_block_m * L1_block_k);
      for (int i = 0; i < L1_block_m; i++) {
        for (int j = 0; j < L1_block_k; j++) {
          int orig_row = L1_row * L1_block_m + i;
          int orig_col = L1_col * L1_block_k + j;
          tile[i * L1_block_k + j] = input[orig_row * cols + orig_col];
        }
      }
      std::vector<float> shuffled =
          layout_A_2x1_8x8block(tile, L1_block_m, L1_block_k);
      for (size_t i = 0; i < shuffled.size(); i++)
        output[output_idx++] = shuffled[i];
    }
  }
  return output;
}

inline std::vector<float> layout_inverse_C_2x2_8x8block(
    const std::vector<float>& input, int L1_block_m, int L1_block_n) {
  std::vector<float> output(L1_block_m * L1_block_n);
  int input_idx = 0;
  int blocks_per_row = L1_block_n / 8;
  int blocks_per_col = L1_block_m / 8;
  for (int super_block_row = 0; super_block_row < blocks_per_col;
       super_block_row += 2) {
    for (int super_block_col = 0; super_block_col < blocks_per_row;
         super_block_col += 2) {
      // Order within each 2x2 super-block: [0,0], [0,1], [1,0], [1,1].
      for (int block_row = 0; block_row < 2; block_row++) {
        for (int block_col = 0; block_col < 2; block_col++) {
          int cbr = super_block_row + block_row;
          int cbc = super_block_col + block_col;
          for (int rib = 0; rib < 8; rib++) {
            for (int cib = 0; cib < 8; cib++) {
              int out_row = cbr * 8 + rib;
              int out_col = cbc * 8 + cib;
              output[out_row * L1_block_n + out_col] = input[input_idx++];
            }
          }
        }
      }
    }
  }
  return output;
}

inline std::vector<float> layout_inverse_C_L1_2x2_8x8block(
    const std::vector<float>& input, int M, int N, int L1_block_m,
    int L1_block_n) {
  assert(M % L1_block_m == 0 && "M must be divisible by L1_block_m");
  assert(N % L1_block_n == 0 && "N must be divisible by L1_block_n");
  assert(L1_block_m % 16 == 0 &&
         "L1_block_m must be divisible by 16 for 2x2 8x8 blocks");
  assert(L1_block_n % 16 == 0 &&
         "L1_block_n must be divisible by 16 for 2x2 8x8 blocks");

  std::vector<float> output(M * N);
  int L1_rows = M / L1_block_m;
  int L1_cols = N / L1_block_n;
  int input_idx = 0;
  // L1-tile order: row-major (matches the output dispatch sequence in
  // the IRON design's runtime_sequence).
  for (int L1_row = 0; L1_row < L1_rows; L1_row++) {
    for (int L1_col = 0; L1_col < L1_cols; L1_col++) {
      int L1_block_size = L1_block_m * L1_block_n;
      std::vector<float> tile(L1_block_size);
      for (int i = 0; i < L1_block_size; i++)
        tile[i] = input[input_idx++];
      std::vector<float> unshuffled =
          layout_inverse_C_2x2_8x8block(tile, L1_block_m, L1_block_n);
      for (int i = 0; i < L1_block_m; i++) {
        for (int j = 0; j < L1_block_n; j++) {
          int out_row = L1_row * L1_block_m + i;
          int out_col = L1_col * L1_block_n + j;
          output[out_row * N + out_col] = unshuffled[i * L1_block_n + j];
        }
      }
    }
  }
  return output;
}

}  // namespace gufo::npu::atb

#endif  // GUFO_NPU_ATB_LAYOUT_HPP
