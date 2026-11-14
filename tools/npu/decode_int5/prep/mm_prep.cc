// QK-norm + RoPE on one 16-token stripe of a 128-column head tile in the C wire layout.
// Element (6912 B, last 256 B: eps at 6656): 2304 B wire stripe [block (sbc*4+br*2+bc), 72 B each], 4096 B rope table, 256 B norm weight (at 6400), eps bf16 (at 6656), padding.
// Table: [br 2][iblock 8][cos,sin] bf16 vectors (lane = row*8+col): cos/sin of token (br*8+row), i = iblock*8+col.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" void __cxa_finalize(void *) {}

__attribute__((aligned(64))) const uint8_t id_blk[72] = {
    127, 64, 0,  0,  0,  0,  0,  0,  0,  127, 0,  64, 0,  0,  0,  0,  0,  0,
    127, 0,  0,  64, 0,  0,  0,  0,  0,  127, 0,  0,  0,  64, 0,  0,  0,  0,
    127, 0,  0,  0,  0,  64, 0,  0,  0,  127, 0,  0,  0,  0,  0,  64, 0,  0,
    127, 0,  0,  0,  0,  0,  0,  64, 0,  127, 0,  0,  0,  0,  0,  0,  0,  64};

constexpr int kElemBytes = 6912, kTabOff = 2304, kWOff = 6400, kEpsOff = 6656;

alignas(64) static bfloat16 dec[32 * 64];

static inline aie::vector<bfloat16, 64> rep8(const bfloat16 *p) {
  const aie::vector<bfloat16, 8> g8 = aie::load_v<8>(p);
  const aie::vector<bfloat16, 16> g16 = aie::concat(g8, g8);
  const aie::vector<bfloat16, 32> g32 = aie::concat(g16, g16);
  return aie::concat(g32, g32);
}

extern "C" {
__attribute__((noinline)) void prep_elem(const uint8_t *e, uint8_t *out) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> idst((bfp16ebs8 *)id_blk);
  idst.seek(0);
  const aie::block_vector<bfp16ebs8, 64> idb = idst.pop();
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> in((const bfp16ebs8 *)e);
  in.seek(0);
  for (int b = 0; b < 32; ++b) {
    const aie::block_vector<bfp16ebs8, 64> g = in.pop();
    const aie::accum<accfloat, 64> da = mac_8x8_8x8T(g, idb, z);
    aie::store_v(dec + b * 64, da.template to_vector<bfloat16>());
  }
  const bfloat16 *tab = (const bfloat16 *)(e + kTabOff);
  const bfloat16 *w = (const bfloat16 *)(e + kWOff);
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::block_vector<bfp16ebs8, 64> ob = aie::accum<accfloat, 64>(one).template to_vector<bfp16ebs8>();
  const aie::vector<bfloat16, 64> inv_n = aie::broadcast<bfloat16, 64>((bfloat16)(1.0f / 128.0f));
  const aie::vector<bfloat16, 64> eps = aie::broadcast<bfloat16, 64>(*(const bfloat16 *)(e + kEpsOff));
  const aie::vector<bfloat16, 64> half = aie::broadcast<bfloat16, 64>((bfloat16)0.5f);
  const aie::vector<bfloat16, 64> c15 = aie::broadcast<bfloat16, 64>((bfloat16)1.5f);
  for (int br = 0; br < 2; ++br) {
    aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
    for (int sbc = 0; sbc < 8; ++sbc)
      for (int bc = 0; bc < 2; ++bc) {
        const aie::vector<bfloat16, 64> v = aie::load_v<64>(dec + (sbc * 4 + br * 2 + bc) * 64);
        acc = aie::mac(acc, v, v);
      }
    const aie::accum<accfloat, 64> sacc = mac_8x8_8x8T(acc.template to_vector<bfp16ebs8>(), ob, z);
    const aie::vector<bfloat16, 64> sv = sacc.template to_vector<bfloat16>();
    const aie::vector<bfloat16, 64> ms = aie::add(aie::mul(sv, inv_n).template to_vector<bfloat16>(), eps);
    const aie::vector<int16, 64> bi = ms.template cast_to<int16>();
    const aie::vector<int16, 64> yi = aie::sub(aie::broadcast<int16, 64>(0x5f37), aie::downshift(bi, 1));
    aie::vector<bfloat16, 64> y = yi.template cast_to<bfloat16>();
    const aie::vector<bfloat16, 64> hm = aie::mul(ms, half).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 64> t = aie::mul(aie::mul(y, y).template to_vector<bfloat16>(), hm).template to_vector<bfloat16>();
    y = aie::mul(y, aie::sub(c15, t)).template to_vector<bfloat16>();
    for (int sbc = 0; sbc < 4; ++sbc)
      for (int bc = 0; bc < 2; ++bc) {
        const int ia = sbc * 4 + br * 2 + bc, ib = (sbc + 4) * 4 + br * 2 + bc;
        const int ib8 = sbc * 2 + bc;
        const aie::vector<bfloat16, 64> wa = rep8(w + sbc * 16 + bc * 8);
        const aie::vector<bfloat16, 64> wb = rep8(w + 64 + sbc * 16 + bc * 8);
        const aie::vector<bfloat16, 64> a = aie::mul(aie::mul(aie::load_v<64>(dec + ia * 64), y).template to_vector<bfloat16>(), wa).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> b = aie::mul(aie::mul(aie::load_v<64>(dec + ib * 64), y).template to_vector<bfloat16>(), wb).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> cs = aie::load_v<64>(tab + ((br * 8 + ib8) * 2) * 64);
        const aie::vector<bfloat16, 64> sn = aie::load_v<64>(tab + ((br * 8 + ib8) * 2 + 1) * 64);
        aie::accum<accfloat, 64> lo = aie::mul(a, cs);
        lo = aie::msc(lo, b, sn);
        aie::accum<accfloat, 64> hi = aie::mul(b, cs);
        hi = aie::mac(hi, a, sn);
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o0((bfp16ebs8 *)(out + ia * 72));
        o0.seek(0);
        o0.push(lo.template to_vector<bfp16ebs8>());
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o1((bfp16ebs8 *)(out + ib * 72));
        o1.seek(0);
        o1.push(hi.template to_vector<bfp16ebs8>());
      }
  }
}
}
