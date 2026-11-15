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



__attribute__((always_inline)) static inline aie::accum<accfloat, 64> silu_mul_bf(
    const aie::vector<bfloat16, 64> &gb, const aie::vector<bfloat16, 64> &ub) {
  const aie::vector<bfloat16, 64> half = aie::broadcast<bfloat16, 64>((bfloat16)0.5f);
  const aie::accum<accfloat, 64> ah = aie::mul(gb, half);
  const aie::vector<bfloat16, 64> th = aie::tanh<bfloat16>(ah.template to_vector<float>());
  aie::accum<accfloat, 64> as = aie::accum<accfloat, 64>(half);
  as = aie::mac(as, th, half);
  const aie::vector<bfloat16, 64> sig = as.template to_vector<bfloat16>();
  aie::accum<accfloat, 64> sa = aie::mul(gb, sig);
  const aie::vector<bfloat16, 64> silu = sa.template to_vector<bfloat16>();
  return aie::mul(silu, ub);
}

static int g_swi = 0;
void set_swi(int f) { g_swi = f; }

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

  if (k_counter == k_flush && g_swi) {
    k_counter = 0;
    ::aie::set_rounding(::aie::rounding_mode::floor);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os(pC_bfp16);
    os.seek(0);
    const bfloat16 *src = (const bfloat16 *)pC_bfp16;
    for (int i = 0; i < 64; i++) {
      const aie::vector<bfloat16, 64> gv = aie::load_v<64>(src + i * 128);
      const aie::vector<bfloat16, 64> uv = aie::load_v<64>(src + i * 128 + 64);
      os.push(silu_mul_bf(gv, uv).template to_vector<bfp16ebs8>());
    }
    for (int i = 0; i < 64; i++) {
      const aie::vector<bfloat16, 64> gv = aie::load_v<64>(c_bf16_2nd_half + i * 128);
      const aie::vector<bfloat16, 64> uv = aie::load_v<64>(c_bf16_2nd_half + i * 128 + 64);
      os.push(silu_mul_bf(gv, uv).template to_vector<bfp16ebs8>());
    }
  } else if (k_counter == k_flush) { // // K / k  *  DIV
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

__attribute__((noinline)) void swi_jrel(const bfp16ebs8 *__restrict w, bfp16ebs8 *__restrict o, int jr) {
  if ((unsigned)jr < 4u) swi_half(w, o, (jr >> 1) * 1024 + (jr & 1) * 512);
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

extern "C" {
// ---- norm mode: x += rmsnorm(branch)*gp ; z = rmsnorm(x)*g, three passes over (branch, x) tile pairs ----
// Scratch aliases c_bf16_2nd_half (bf16 units): g[2048] gp[2048] acc0[256f] acc1[256f] sB[256] sZ[256] eps stash.
constexpr int NM_GP = 2048, NM_A0 = 4096, NM_A1 = 4608, NM_SB = 5120, NM_SZ = 5376, NM_EPS = 5632, NM_STASH = 5760;
extern "C" {
static inline void norm_init(const bfloat16 *__restrict aux) {
  for (int i = 0; i < 2048; i += 64) {
    aie::store_v(c_bf16_2nd_half + i, aie::load_v<64>(aux + i));
    aie::store_v(c_bf16_2nd_half + NM_GP + i, aie::load_v<64>(aux + 2112 + i));
  }
  float *acc = (float *)(c_bf16_2nd_half + NM_A0);
  const aie::vector<float, 16> z = aie::zeros<float, 16>();
  for (int i = 0; i < 512; i += 16) aie::store_v(acc + i, z);
  c_bf16_2nd_half[NM_EPS] = aux[2048];
}

static inline void norm_fin(const float *__restrict acc, bfloat16 *__restrict dst) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::block_vector<bfp16ebs8, 64> ob = aie::accum<accfloat, 64>(one).template to_vector<bfp16ebs8>();
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  const aie::vector<bfloat16, 64> inv_n = aie::broadcast<bfloat16, 64>((bfloat16)(1.0f / 2048.0f));
  const aie::vector<bfloat16, 64> eps = aie::broadcast<bfloat16, 64>(c_bf16_2nd_half[NM_EPS]);
  const aie::vector<bfloat16, 64> half = aie::broadcast<bfloat16, 64>((bfloat16)0.5f);
  const aie::vector<bfloat16, 64> c15 = aie::broadcast<bfloat16, 64>((bfloat16)1.5f);
  for (int q = 0; q < 4; ++q) {
    const aie::accum<accfloat, 64> a(aie::load_v<64>(acc + q * 64));
    const aie::accum<accfloat, 64> sacc = mac_8x8_8x8T(a.template to_vector<bfp16ebs8>(), ob, z);
    const aie::vector<bfloat16, 64> sv = sacc.template to_vector<bfloat16>();
    const aie::vector<bfloat16, 64> ms = aie::add(aie::mul(sv, inv_n).template to_vector<bfloat16>(), eps);
    const aie::vector<int16, 64> bi = ms.template cast_to<int16>();
    const aie::vector<int16, 64> yi = aie::sub(aie::broadcast<int16, 64>(0x5f37), aie::downshift(bi, 1));
    aie::vector<bfloat16, 64> y = yi.template cast_to<bfloat16>();
    const aie::vector<bfloat16, 64> hm = aie::mul(ms, half).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 64> t = aie::mul(aie::mul(y, y).template to_vector<bfloat16>(), hm).template to_vector<bfloat16>();
    y = aie::mul(y, aie::sub(c15, t)).template to_vector<bfloat16>();
    aie::store_v(dst + q * 64, y);
  }
}

static inline aie::vector<bfloat16, 64> norm_rep(const bfloat16 *p) {
  const aie::vector<bfloat16, 8> g8 = aie::load_v<8>(p);
  const aie::vector<bfloat16, 16> g16 = aie::concat(g8, g8);
  const aie::vector<bfloat16, 32> g32 = aie::concat(g16, g16);
  return aie::concat(g32, g32);
}

static inline void norm_addn(const bfloat16 *__restrict stash, const bfp16ebs8 *__restrict bs, bfp16ebs8 *__restrict out, int kt, int ph, int swap) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> idst((bfp16ebs8 *)id_blk);
  idst.seek(0);
  const aie::block_vector<bfp16ebs8, 64> idb = idst.pop();
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  float *acc0 = (float *)(c_bf16_2nd_half + NM_A0);
  float *acc1 = (float *)(c_bf16_2nd_half + NM_A1);
  const bfloat16 *sB = c_bf16_2nd_half + NM_SB;
  const bfloat16 *sZ = c_bf16_2nd_half + NM_SZ;
  const uint8_t *bb = swap ? (const uint8_t *)bs : (const uint8_t *)stash;
  const uint8_t *xb = swap ? (const uint8_t *)stash : (const uint8_t *)bs;
  uint8_t *ob = (uint8_t *)out + (kt & 7) * 2304;
  for (int sbr = 0; sbr < 2; ++sbr) {
    for (int sbc = 0; sbc < 8; ++sbc) {
      for (int bis = 0; bis < 2; ++bis) {
        const int ci = sbr * 16 + (sbc >> 1) * 4 + bis * 2 + (sbc & 1);
        const int q = sbr * 2 + bis;
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb((const bfp16ebs8 *)(bb + ci * 72));
        pb.seek(0);
        const aie::accum<accfloat, 64> ab = mac_8x8_8x8T(pb.pop(), idb, z);
        const aie::vector<bfloat16, 64> bv = ab.template to_vector<bfloat16>();
        if (ph == 0) {
          float *ap = acc0 + q * 64;
          aie::store_v(ap, aie::mac(aie::accum<accfloat, 64>(aie::load_v<64>(ap)), bv, bv).template to_vector<float>());
          continue;
        }
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> px((const bfp16ebs8 *)(xb + ci * 72));
        px.seek(0);
        const aie::vector<bfloat16, 64> pbv =
            aie::mul(norm_rep(c_bf16_2nd_half + NM_GP + kt * 64 + sbc * 8), aie::load_v<64>(sB + q * 64)).template to_vector<bfloat16>();
        const aie::accum<accfloat, 64> a1 = mac_8x8_8x8T(px.pop(), idb, z);
        const aie::accum<accfloat, 64> a2 = aie::mac(a1, bv, pbv);
        const aie::vector<bfloat16, 64> v = a2.template to_vector<bfloat16>();
        if (ph == 1) {
          float *ap = acc1 + q * 64;
          aie::store_v(ap, aie::mac(aie::accum<accfloat, 64>(aie::load_v<64>(ap)), v, v).template to_vector<float>());
          aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os(
              (bfp16ebs8 *)((uint8_t *)out + ((sbr * 8 + (kt & 7)) * 16 + (ci & 15)) * 72));
          os.seek(0);
          os.push(a2.template to_vector<bfp16ebs8>());
        } else {
          const aie::vector<bfloat16, 64> pz =
              aie::mul(norm_rep(c_bf16_2nd_half + kt * 64 + sbc * 8), aie::load_v<64>(sZ + q * 64)).template to_vector<bfloat16>();
          aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os((bfp16ebs8 *)(ob + (sbr * 16 + sbc * 2 + bis) * 72));
          os.seek(0);
          os.push(aie::mul(v, pz).template to_vector<bfp16ebs8>());
        }
      }
    }
  }
}

__attribute__((noinline)) void norm_step(const bfp16ebs8 *__restrict e, bfp16ebs8 *__restrict out, int idx, int rho, int swap) {
  if (idx < 16) {
    if (idx == 0) norm_init((const bfloat16 *)e);
    return;
  }
  const int j = idx - 16;
  const int jj = j & 63;
  bfloat16 *stash = c_bf16_2nd_half + NM_STASH;
  if (!(jj & 1)) {
    if (jj == 0 && j) norm_fin((const float *)(c_bf16_2nd_half + (j == 64 ? NM_A0 : NM_A1)), c_bf16_2nd_half + (j == 64 ? NM_SB : NM_SZ));
    const uint8_t *src = (const uint8_t *)e + rho * 2304;
    for (int i = 0; i < 2304; i += 64) aie::store_v((uint8_t *)stash + i, aie::load_v<64>(src + i));
  } else {
    norm_addn(stash, (const bfp16ebs8 *)((const uint8_t *)e + rho * 2304), out, jj >> 1, j >> 6, swap);
  }
}
}
}
