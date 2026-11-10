#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int kRows = 16;
constexpr int kCols = 32;
constexpr int kK = 64;

alignas(aie::vector_decl_align) bfloat16 c_stage[kRows * kCols];
alignas(aie::vector_decl_align) bfp16ebs8 b_scratch[kK * kCols / 8];

static inline aie::vector<bfloat16, 64> ExpandScales(const bfloat16 *p) {
  aie::vector<bfloat16, 8> s = aie::load_v<8>(p);
  auto z1 = aie::interleave_zip(s, s, 1);
  aie::vector<bfloat16, 16> s16 = aie::concat(z1.first, z1.second);
  auto z2 = aie::interleave_zip(s16, s16, 1);
  aie::vector<bfloat16, 32> s32 = aie::concat(z2.first, z2.second);
  auto z3 = aie::interleave_zip(s32, s32, 1);
  return aie::concat(z3.first, z3.second);
}

__attribute__((noinline)) static void Unpack(const uint16_t *__restrict pw,
                                             const bfloat16 *__restrict ps) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(b_scratch);
  out.seek(0);
  const aie::vector<uint16, 64> mask = aie::broadcast<uint16, 64>(31);
  const aie::vector<uint16, 64> magic = aie::broadcast<uint16, 64>(0x4300);
  const aie::vector<bfloat16, 64> m144 = aie::broadcast<bfloat16, 64>((bfloat16)-144.0f);
  for (int p = 0; p < 32; p++) {
    const int sb = p >> 4, sr = (p >> 1) & 7, blk = p & 1;
    aie::vector<uint16, 64> w = aie::load_v<64>(pw + (p / 3) * 64);
    aie::vector<uint16, 64> x = aie::bit_or(aie::bit_and(aie::downshift(w, 5 * (p % 3)), mask), magic);
    aie::vector<bfloat16, 64> s = ExpandScales(ps + (((sb * 2 + blk) * 2 + (sr >> 2)) * 8));
    aie::accum<accfloat, 64> a = aie::mul(x.cast_to<bfloat16>(), s);
    a = aie::mac(a, m144, s);
    out.push(a.template to_vector<bfp16ebs8>());
  }
}

static inline void MacTile(bfp16ebs8 *__restrict pA) {
  bfloat16 *pC = c_stage;
  for (int sb = 0; sb < kCols / 16; sb++) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB_stream(b_scratch);
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
}

__attribute__((noinline)) static void FlushSwiglu(bfp16ebs8 *__restrict pC_bfp16) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(pC_bfp16);
  out.seek(0);
  const aie::vector<bfloat16, 64> c_hi = aie::broadcast<bfloat16, 64>((bfloat16)-1.4453125f);
  const aie::vector<bfloat16, 64> c_lo = aie::broadcast<bfloat16, 64>((bfloat16)0.0026175f);
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  for (int sb = 0; sb < 2; sb++)
    for (int br = 0; br < 2; br++) {
      aie::vector<bfloat16, 64> gb = aie::load_v<64>(c_stage + sb * 256 + (br * 2) * 64);
      aie::vector<bfloat16, 64> ub = aie::load_v<64>(c_stage + sb * 256 + (br * 2 + 1) * 64);
      aie::accum<accfloat, 64> ap = aie::mul(gb, c_hi);
      ap = aie::mac(ap, gb, c_lo);
      aie::vector<float, 64> arg = ap.template to_vector<float>();
      aie::vector<bfloat16, 64> t = aie::exp2<bfloat16>(arg);
      aie::vector<bfloat16, 64> den = aie::add(t, one);
      aie::vector<bfloat16, 64> sig = aie::inv(den);
      aie::accum<accfloat, 64> sa = aie::mul(gb, sig);
      aie::vector<bfloat16, 64> silu = sa.template to_vector<bfloat16>();
      aie::accum<accfloat, 64> act = aie::mul(silu, ub);
      out.push(act.template to_vector<bfp16ebs8>());
    }
}

__attribute__((noinline)) static void FlushPlain(bfp16ebs8 *__restrict pC_bfp16) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(pC_bfp16);
  out.seek(0);
  for (int i = 0; i < kRows * kCols / 64; i++) {
    aie::accum<accfloat, 64> a(aie::load_v<64>(c_stage + i * 64));
    out.push(a.template to_vector<bfp16ebs8>());
  }
}

extern "C" {

static int kc_gu = 0;
static int kc_dn = 0;

void matmul_gu(bfp16ebs8 *__restrict pA, uint8_t *__restrict pBp,
               bfp16ebs8 *__restrict pC_bfp16, int32_t kcnt) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  Unpack((const uint16_t *)pBp, (const bfloat16 *)(pBp + 1408));
  MacTile(pA);
  if (kc_gu == kcnt - 1) {
    kc_gu = 0;
    FlushSwiglu(pC_bfp16);
  } else {
    kc_gu += 1;
  }
}

void matmul_dn(bfp16ebs8 *__restrict pA, uint8_t *__restrict pBp,
               bfp16ebs8 *__restrict pC_bfp16, int32_t kcnt) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  Unpack((const uint16_t *)pBp, (const bfloat16 *)(pBp + 1408));
  MacTile(pA);
  if (kc_dn == kcnt - 1) {
    kc_dn = 0;
    FlushPlain(pC_bfp16);
  } else {
    kc_dn += 1;
  }
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  const aie::vector<bfloat16, 64> zeros = aie::zeros<bfloat16, 64>();
  for (int i = 0; i < kRows * kCols / 64; i++)
    aie::store_v(c_stage + i * 64, zeros);
}
}
