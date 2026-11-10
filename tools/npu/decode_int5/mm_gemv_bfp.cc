#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int kRows = 16;
constexpr int kCols = 32;
constexpr int kK = 64;

alignas(aie::vector_decl_align) bfloat16 c_stage[kRows * kCols];

extern "C" {

static int k_counter = 0;

void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                             bfp16ebs8 *__restrict pC_bfp16, int32_t kcnt) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  bfloat16 *pC = c_stage;
  for (int sb = 0; sb < kCols / 16; sb++) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB_stream(pB);
    pB_stream.seek(sb * (kK / 8) * 2);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream(pA);
    pA_stream.seek(0);
    aie::accum<accfloat, 64> acc0(aie::load_v<64>(pC));
    aie::accum<accfloat, 64> acc1(aie::load_v<64>(pC + 64));
    aie::accum<accfloat, 64> acc2(aie::load_v<64>(pC + 128));
    aie::accum<accfloat, 64> acc3(aie::load_v<64>(pC + 192));
    for (int ks = 0; ks < kK / 8; ks++) {
      aie::block_vector<bfp16ebs8, 64> a0 = pA_stream.pop();
      aie::block_vector<bfp16ebs8, 64> b0 = pB_stream.pop();
      aie::block_vector<bfp16ebs8, 64> a1 = pA_stream.pop();
      aie::block_vector<bfp16ebs8, 64> b1 = pB_stream.pop();
      acc0 = mac_8x8_8x8T(a0, b0, acc0);
      acc1 = mac_8x8_8x8T(a0, b1, acc1);
      acc2 = mac_8x8_8x8T(a1, b0, acc2);
      acc3 = mac_8x8_8x8T(a1, b1, acc3);
    }
    aie::store_v(pC, acc0.template to_vector<bfloat16>());
    aie::store_v(pC + 64, acc1.template to_vector<bfloat16>());
    aie::store_v(pC + 128, acc2.template to_vector<bfloat16>());
    aie::store_v(pC + 192, acc3.template to_vector<bfloat16>());
    pC += 256;
  }
  if (k_counter == kcnt - 1) {
    k_counter = 0;
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(pC_bfp16);
    out.seek(0);
    for (int i = 0; i < kRows * kCols / 64; i++) {
      aie::accum<accfloat, 64> a(aie::load_v<64>(c_stage + i * 64));
      out.push(a.template to_vector<bfp16ebs8>());
    }
  } else {
    k_counter += 1;
  }
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  const aie::vector<bfloat16, 64> zeros = aie::zeros<bfloat16, 64>();
  for (int i = 0; i < kRows * kCols / 64; i++)
    aie::store_v(c_stage + i * 64, zeros);
}
}
