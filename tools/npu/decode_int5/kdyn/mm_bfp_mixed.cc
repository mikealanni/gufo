//===- mm_bfp_mixed.cc ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define SWI31_TANH
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// The kernel's `c_bf16_2nd_half` BF16 staging buffer is flushed back to the
// BFP-encoded C output once per output tile, after exactly
// (K_Problemsize / k) * DIV matmul calls. The flush count is fixed at compile
// time, so this kernel only supports K = K_Problemsize at runtime. Both this
// example's Makefile default and the lit test pin K to this value; running
// at a different K via CLI override is not supported by this kernel.
constexpr int K_Problemsize = 2048;
constexpr int DIV = 4;
constexpr int M = 128 / 4;
constexpr int K = 64;
constexpr int N = 128;
constexpr int m = 128 / 4;
constexpr int k = 64;
constexpr int n = 128;
constexpr int r = 8;
constexpr int s = 8;
constexpr int t = 8;

alignas(aie::vector_decl_align) bfloat16 c_bf16_2nd_half[m * DIV * n / 2];

template <int Mz, int Nz>
void zero_vectorized_v64bfp16ebs8(bfp16ebs8 *__restrict cOut) {

  bfloat16 *c_ptr_bf16 = (bfloat16 *)cOut;
  const aie::vector<bfloat16, 64> zeros = aie::zeros<bfloat16, 64>();
  for (int i = 0; i < m * DIV * n / 64 / 2; i++) {
    aie::store_v(c_ptr_bf16 + i * 64, zeros);
  }

  bfloat16 *c_bf16_2nd_half_ptr = c_bf16_2nd_half;
  for (int i = 0; i < m * DIV * n / 64 / 2; i++) {
    aie::store_v(c_bf16_2nd_half_ptr + i * 64, zeros);
  }
}

__attribute__((aligned(64))) const uint8_t id_blk[72] = {
    127, 64, 0,  0,  0,  0,  0,  0,  0,  127, 0,  64, 0,  0,  0,  0,  0,  0,
    127, 0,  0,  64, 0,  0,  0,  0,  0,  127, 0,  0,  0,  64, 0,  0,  0,  0,
    127, 0,  0,  0,  0,  64, 0,  0,  0,  127, 0,  0,  0,  0,  0,  64, 0,  0,
    127, 0,  0,  0,  0,  0,  0,  64, 0,  127, 0,  0,  0,  0,  0,  0,  0,  64};

extern "C" {
__attribute__((always_inline)) static inline aie::accum<accfloat, 64> silu_mul(
    const aie::block_vector<bfp16ebs8, 64> &g,
    const aie::block_vector<bfp16ebs8, 64> &u,
    const aie::block_vector<bfp16ebs8, 64> &idb,
    const aie::accum<accfloat, 64> &z,
    const aie::vector<bfloat16, 64> &c_hi,
    const aie::vector<bfloat16, 64> &c_lo,
    const aie::vector<bfloat16, 64> &one) {
  aie::accum<accfloat, 64> ag = mac_8x8_8x8T(g, idb, z);
  aie::accum<accfloat, 64> au = mac_8x8_8x8T(u, idb, z);
  const aie::vector<bfloat16, 64> gb = ag.template to_vector<bfloat16>();
  const aie::vector<bfloat16, 64> ub = au.template to_vector<bfloat16>();
  aie::accum<accfloat, 64> ap = aie::mul(gb, c_hi);
  ap = aie::mac(ap, gb, c_lo);
#ifdef SWI31_TANH  // sigmoid(g) = 0.5 + 0.5*tanh(g/2): one native SFU op, no reciprocal
  const aie::vector<bfloat16, 64> half = aie::broadcast<bfloat16, 64>((bfloat16)0.5f);
  const aie::accum<accfloat, 64> ah = aie::mul(gb, half);
  const aie::vector<bfloat16, 64> th = aie::tanh<bfloat16>(ah.template to_vector<float>());
  aie::accum<accfloat, 64> as = aie::accum<accfloat, 64>(half);
  as = aie::mac(as, th, half);
  const aie::vector<bfloat16, 64> sig = as.template to_vector<bfloat16>();
  (void)c_hi; (void)c_lo; (void)one;
#elif defined(SWI31_NR)   // vector NR reciprocal: aie::inv(bf16) compiles to 64 SCALAR
  // divides per vec (IR-proven: scalar get_real/upd loop, no recip insn;
  // cost +9.3 ms). Trick: s = 2^-|arg| in (0,1], so d = 1+s in (1,2] where
  // y0=1/sqrt(2) has NR error e0=0.414 -> 3 steps -> 8.5e-4 (< bf16 quantum).
  // sig = 1/(1+2^arg) = (arg>=0) ? s/(1+s) : 1/(1+s)  [both use the SAME d].
  const aie::vector<float, 64> a = ap.template to_vector<float>();
  const aie::vector<float, 64> absa = aie::abs(a);
  const aie::vector<float, 64> neg = aie::sub(aie::zeros<accfloat, 64>(), absa)
                                         .template to_vector<float>();
  const aie::vector<bfloat16, 64> sb = aie::exp2<bfloat16>(neg);
  const aie::vector<float, 64> d(
      aie::accum<accfloat, 64>(aie::add(sb, one)).template to_vector<float>());
  aie::vector<float, 64> y = aie::broadcast<float, 64>(0.70710678f);
  const aie::vector<float, 64> two = aie::broadcast<float, 64>(2.0f);
  const aie::accum<accfloat, 64> acc_two = aie::accum<accfloat, 64>(two);
  y = aie::mul(y, aie::sub(acc_two, aie::mul(d, y).template to_vector<float>()).template to_vector<float>()).template to_vector<float>();
  y = aie::mul(y, aie::sub(acc_two, aie::mul(d, y).template to_vector<float>()).template to_vector<float>()).template to_vector<float>();
  y = aie::mul(y, aie::sub(acc_two, aie::mul(d, y).template to_vector<float>()).template to_vector<float>()).template to_vector<float>();
  const aie::vector<float, 64> sf(aie::accum<accfloat, 64>(sb).template to_vector<float>());
  const aie::vector<float, 64> num =
      aie::mul(sf, y).template to_vector<float>();
  const auto m = aie::gt(a, aie::broadcast<float, 64>(0.0f));
  // blend.hpp proven: select(v1, v2, m) returns m ? v2 : v1 (mask LAST, picks v2)
  const aie::vector<float, 64> sigf = aie::select(y, num, m);
  const aie::vector<bfloat16, 64> sig =
      aie::accum<accfloat, 64>(sigf).template to_vector<bfloat16>();
#else
  const aie::vector<float, 64> arg = ap.template to_vector<float>();
  const aie::vector<bfloat16, 64> t = aie::exp2<bfloat16>(arg);
  const aie::vector<bfloat16, 64> den = aie::add(t, one);
  const aie::vector<bfloat16, 64> sig = aie::inv(den);
#endif
  aie::accum<accfloat, 64> sa = aie::mul(gb, sig);
  const aie::vector<bfloat16, 64> silu = sa.template to_vector<bfloat16>();
  return aie::mul(silu, ub);
}



static int g_counter = 0;

static int k_counter = 0;
static int k_flush = (K_Problemsize / k) * DIV - 1;

void set_k(int kk) {
  k_flush = kk * DIV - 1;
  k_counter = 0;
  g_counter = 0;
}

void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                             bfp16ebs8 *__restrict pC_bfp16) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);

  bfloat16 *pC;
  if (g_counter == 0) {
    pC = (bfloat16 *)pC_bfp16;
  } else if (g_counter == 1) {
    pC = (bfloat16 *)pC_bfp16;
    pC += 1 * m * n;
  } else if (g_counter == 2) {
    pC = c_bf16_2nd_half;
  } else {
    pC = c_bf16_2nd_half + 1 * m * n;
  }

  if (g_counter == DIV - 1) {
    g_counter = 0;
  } else {
    g_counter = g_counter + 1;
  }
  int run_num = 1;
  for (int run = 0; run < run_num; run++) {
    for (int i = 0; i < m * n / (16 * 16); i = i + 4) {
      int block_row = i / (n / 16);
      int block_col = i % (n / 16);

      int A_stream_index = block_row * 2 * k / 8;
      int B_stream_index = block_col * (k / 8) * 2;

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB_stream(pB);
      pB_stream.seek(B_stream_index);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream(pA);
      pA_stream.seek(A_stream_index);

      aie::accum<accfloat, 64> chess_storage(dm1)
          acc0_data(aie::load_v<64>(pC));
      aie::accum<accfloat, 64> chess_storage(dm2)
          acc1_data(aie::load_v<64>(pC + 64));
      aie::accum<accfloat, 64> chess_storage(dm3)
          acc2_data(aie::load_v<64>(pC + 128));
      aie::accum<accfloat, 64> chess_storage(dm4)
          acc3_data(aie::load_v<64>(pC + 192));

      aie::block_vector<bfp16ebs8, 64> chess_storage(ex0) A0_data_bfp;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex1) A1_data_bfp;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex2) B0_data_bfp;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex3) B1_data_bfp;

      aie::block_vector<bfp16ebs8, 64> chess_storage(ex4) A0_data_bfp_pong;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex6) A1_data_bfp_pong;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex5) B0_data_bfp_pong;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex7) B1_data_bfp_pong;

      aie::block_vector<bfp16ebs8, 64> chess_storage(ex8) A0_data_bfp_2;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex9) A1_data_bfp_2;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex10) B0_data_bfp_2;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex11) B1_data_bfp_2;

      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream2(pA);
      pA_stream2.seek(A_stream_index);

      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();

      aie::store_v(pC, acc0_data.template to_vector<bfloat16>());
      acc0_data = aie::load_v<64>(pC + 256);
      aie::store_v(pC + 64, acc1_data.template to_vector<bfloat16>());
      acc1_data = aie::load_v<64>(pC + 320);
      aie::store_v(pC + 128, acc2_data.template to_vector<bfloat16>());
      acc2_data = aie::load_v<64>(pC + 384);
      aie::store_v(pC + 192, acc3_data.template to_vector<bfloat16>());
      acc3_data = aie::load_v<64>(pC + 448);

      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);
      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);
      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);
      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream3(pA);
      pA_stream3.seek(A_stream_index);

      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();

      aie::store_v(pC + 256, acc0_data.template to_vector<bfloat16>());
      acc0_data = aie::load_v<64>(pC + 512);
      aie::store_v(pC + 320, acc1_data.template to_vector<bfloat16>());
      acc1_data = aie::load_v<64>(pC + 576);
      aie::store_v(pC + 384, acc2_data.template to_vector<bfloat16>());
      acc2_data = aie::load_v<64>(pC + 640);
      aie::store_v(pC + 448, acc3_data.template to_vector<bfloat16>());
      acc3_data = aie::load_v<64>(pC + 704);

      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream4(pA);
      pA_stream4.seek(A_stream_index);

      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();

      aie::store_v(pC + 512, acc0_data.template to_vector<bfloat16>());
      acc0_data = aie::load_v<64>(pC + 768);
      aie::store_v(pC + 576, acc1_data.template to_vector<bfloat16>());
      acc1_data = aie::load_v<64>(pC + 832);
      aie::store_v(pC + 640, acc2_data.template to_vector<bfloat16>());
      acc2_data = aie::load_v<64>(pC + 896);
      aie::store_v(pC + 704, acc3_data.template to_vector<bfloat16>());
      acc3_data = aie::load_v<64>(pC + 960);

      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      aie::store_v(pC + 768, acc0_data.template to_vector<bfloat16>());
      aie::store_v(pC + 832, acc1_data.template to_vector<bfloat16>());
      aie::store_v(pC + 896, acc2_data.template to_vector<bfloat16>());
      aie::store_v(pC + 960, acc3_data.template to_vector<bfloat16>());

      pC += 256 * 4;
    }
  }

  if (k_counter == k_flush) { // // K / k  *  DIV
    k_counter = 0;
    // inplace convert the first half BF16 to pC_bfp16
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pCOut_stream_first(
        pC_bfp16);
    pCOut_stream_first.seek(0);

    bfloat16 *pC_bf16;
    pC_bf16 = (bfloat16 *)pC_bfp16;
    for (int i = 0; i < m * DIV * n / 2 / 64; i++) {
      aie::accum<accfloat, 64> acc_data(aie::load_v<64>(pC_bf16));
      pC_bf16 += 64;
      pCOut_stream_first.push(acc_data.template to_vector<bfp16ebs8>());
    }

    bfloat16 *c_bf16_2nd_half_ptr = c_bf16_2nd_half;
    for (int i = 0; i < m * DIV * n / 2 / 64; i++) {
      aie::accum<accfloat, 64> acc_data(aie::load_v<64>(c_bf16_2nd_half_ptr));
      c_bf16_2nd_half_ptr += 64;
      pCOut_stream_first.push(acc_data.template to_vector<bfp16ebs8>());
    }
  } else {
    k_counter += 1;
  }
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  zero_vectorized_v64bfp16ebs8<DIM_M, DIM_N>(cOut);
}

__attribute__((noinline)) void swi_half(const bfp16ebs8 *__restrict w, bfp16ebs8 *__restrict o, int off) {
  ::aie::set_rounding(::aie::rounding_mode::floor);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> in(w);
  in.seek(0);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)((uint8_t *)o + (size_t)off * 9));
  out.seek(0);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> idst((bfp16ebs8 *)id_blk);
  idst.seek(0);
  const aie::block_vector<bfp16ebs8, 64> idb = idst.pop();
  const aie::vector<bfloat16, 64> c_hi = aie::broadcast<bfloat16, 64>((bfloat16)-1.4453125f);
  const aie::vector<bfloat16, 64> c_lo = aie::broadcast<bfloat16, 64>((bfloat16)0.0026175f);
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  for (int s = 0; s < 64; ++s) {
    const aie::block_vector<bfp16ebs8, 64> g = in.pop();
    const aie::block_vector<bfp16ebs8, 64> u = in.pop();
    out.push(silu_mul(g, u, idb, z, c_hi, c_lo, one).template to_vector<bfp16ebs8>());
  }
}

// ---- conv mode: gating + 7-tap causal depthwise conv on in_proj wire tiles ----
// Each core owns 4 octets (32 channels) of every 128-channel unit. Scratch aliases c_bf16_2nd_half
// (free in this mode): comb[4 octets][24 rows][8 ch] bf16 (rows 0..7 = previous token block,
// 8..15 / 16..23 = current 16-token strip), then taps[4][7][64] (each tap pre-replicated over 8 rows).
constexpr int CV_COMB = 768;
constexpr int CV_TAPS = 1792;

__attribute__((noinline)) void conv_init(const bfloat16 *__restrict aux) {
  bfloat16 *taps = c_bf16_2nd_half + CV_COMB;
  for (int i = 0; i < CV_TAPS; i += 64) aie::store_v(taps + i, aie::load_v<64>(aux + i));
  const bfloat16 *h = aux + CV_TAPS;
  for (int o = 0; o < 4; ++o) aie::store_v(c_bf16_2nd_half + o * 192, aie::load_v<64>(h + o * 64));
}

__attribute__((noinline)) void conv_strip(const bfp16ebs8 *__restrict e, bfp16ebs8 *__restrict out, int uoff, int s,
                                          int rho) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *eb = (const uint8_t *)e + rho * 576;
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB((const bfp16ebs8 *)eb);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC((const bfp16ebs8 *)(eb + 2304));
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pX((const bfp16ebs8 *)(eb + 4608));
  pB.seek(0);
  pC.seek(0);
  pX.seek(0);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> idst((bfp16ebs8 *)id_blk);
  idst.seek(0);
  const aie::block_vector<bfp16ebs8, 64> idb = idst.pop();
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  const bfloat16 *taps = c_bf16_2nd_half + CV_COMB;
  uint8_t *ob0 = (uint8_t *)out + uoff + s * 576;
  for (int sbl = 0; sbl < 2; ++sbl) {
    for (int br = 0; br < 2; ++br) {
      for (int bc = 0; bc < 2; ++bc) {
        const int oo = sbl * 2 + bc;
        const aie::block_vector<bfp16ebs8, 64> gB = pB.pop();
        const aie::block_vector<bfp16ebs8, 64> gC = pC.pop();
        const aie::block_vector<bfp16ebs8, 64> gX = pX.pop();
        const aie::accum<accfloat, 64> aB = mac_8x8_8x8T(gB, idb, z);
        const aie::accum<accfloat, 64> aC = mac_8x8_8x8T(gC, idb, z);
        const aie::accum<accfloat, 64> aX = mac_8x8_8x8T(gX, idb, z);
        const aie::vector<bfloat16, 64> vB = aB.template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> vC = aC.template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> vX = aX.template to_vector<bfloat16>();
        bfloat16 *cb = c_bf16_2nd_half + oo * 192;
        aie::store_v(cb + (8 + br * 8) * 8, aie::mul(vB, vX).template to_vector<bfloat16>());
        const bfloat16 *tp = taps + oo * 448;
        const bfloat16 *win = cb + (8 * (1 + br) - 6) * 8;
        aie::accum<accfloat, 64> acc = aie::mul(aie::load_v<64>(tp), aie::load_unaligned_v<64>(win));
        for (int j = 1; j < 7; ++j) acc = aie::mac(acc, aie::load_v<64>(tp + j * 64), aie::load_unaligned_v<64>(win + j * 8));
        const aie::vector<bfloat16, 64> cv = acc.template to_vector<bfloat16>();
        const aie::accum<accfloat, 64> y = aie::mul(cv, vC);
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os((bfp16ebs8 *)(ob0 + (oo * 2 + br) * 72));
        os.seek(0);
        os.push(y.template to_vector<bfp16ebs8>());
      }
    }
  }
  for (int o = 0; o < 4; ++o) {
    bfloat16 *cb = c_bf16_2nd_half + o * 192;
    aie::store_v(cb, aie::load_v<64>(cb + 128));
  }
}
}
