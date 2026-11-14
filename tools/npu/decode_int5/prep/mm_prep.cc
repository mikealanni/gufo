// Attention prep on the C wire layout of the QKV GEMM: QK-norm + RoPE (+ V block transpose), emitting the
// attention kernel's Q block (9216 B) and K/V chunk (4608 B) as 1536 B pieces.
// Q element (13824 B): 4 head stripes (2304 B each, blocks sbc*4+br*2+bc), rope table @9216 (4096), norm weight @13312, eps @13568.
// KV element (9216 B): K stripe, V stripe, rope table @4608, norm weight @8704, eps @8960.
// Table: [br 2][iblock 8][cos,sin] bf16 vectors (lane = row*8+col): cos/sin of token (br*8+row), i = iblock*8+col.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" void __cxa_finalize(void *) {}

__attribute__((aligned(64))) const uint8_t id_blk[72] = {
    127, 64, 0,  0,  0,  0,  0,  0,  0,  127, 0,  64, 0,  0,  0,  0,  0,  0,
    127, 0,  0,  64, 0,  0,  0,  0,  0,  127, 0,  0,  0,  64, 0,  0,  0,  0,
    127, 0,  0,  0,  0,  64, 0,  0,  0,  127, 0,  0,  0,  0,  0,  64, 0,  0,
    127, 0,  0,  0,  0,  0,  0,  64, 0,  127, 0,  0,  0,  0,  0,  0,  0,  64};

constexpr int kQTab = 9216, kQW = 13312, kQEps = 13568;
constexpr int kKTab = 4608, kKW = 8704, kKEps = 8960;

alignas(64) static bfloat16 dec[32 * 64];
alignas(64) static uint8_t scratch[9216];

static inline aie::vector<bfloat16, 64> rep8(const bfloat16 *p) {
  const aie::vector<bfloat16, 8> g8 = aie::load_v<8>(p);
  const aie::vector<bfloat16, 16> g16 = aie::concat(g8, g8);
  const aie::vector<bfloat16, 32> g32 = aie::concat(g16, g16);
  return aie::concat(g32, g32);
}

static inline void put(uint8_t *dst, const aie::accum<accfloat, 64> &v) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o((bfp16ebs8 *)dst);
  o.seek(0);
  o.push(v.template to_vector<bfp16ebs8>());
}

static inline void decode_stripe(const uint8_t *e) {
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
}

// dec holds one decoded stripe; out receives 2304 B in destination order (br*16 + ks).
static inline void norm_rope(const bfloat16 *tab, const bfloat16 *w, bfloat16 eps_b, uint8_t *out) {
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::block_vector<bfp16ebs8, 64> ob = aie::accum<accfloat, 64>(one).template to_vector<bfp16ebs8>();
  const aie::vector<bfloat16, 64> inv_n = aie::broadcast<bfloat16, 64>((bfloat16)(1.0f / 128.0f));
  const aie::vector<bfloat16, 64> eps = aie::broadcast<bfloat16, 64>(eps_b);
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
        put(out + (br * 16 + ib8) * 72, lo);
        put(out + (br * 16 + ib8 + 8) * 72, hi);
      }
  }
}

extern "C" {
__attribute__((noinline)) void prep_q(const uint8_t *e) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int hq = 0; hq < 4; ++hq) {
    decode_stripe(e + hq * 2304);
    norm_rope((const bfloat16 *)(e + kQTab), (const bfloat16 *)(e + kQW), *(const bfloat16 *)(e + kQEps), scratch + hq * 2304);
  }
}

__attribute__((noinline)) void prep_kv(const uint8_t *e) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  decode_stripe(e);
  norm_rope((const bfloat16 *)(e + kKTab), (const bfloat16 *)(e + kKW), *(const bfloat16 *)(e + kKEps), scratch);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> idst((bfp16ebs8 *)id_blk);
  idst.seek(0);
  const aie::block_vector<bfp16ebs8, 64> idb = idst.pop();
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> in((const bfp16ebs8 *)(e + 2304));
  in.seek(0);
  for (int sbc = 0; sbc < 8; ++sbc)
    for (int br = 0; br < 2; ++br)
      for (int bc = 0; bc < 2; ++bc) {
        const aie::block_vector<bfp16ebs8, 64> g = in.pop();  // block order sbc*4 + br*2 + bc
        const aie::accum<accfloat, 64> tv = mac_8x8_8x8T(idb, g, z);
        put(scratch + 2304 + (br * 16 + sbc * 2 + bc) * 72, tv);
      }
}

__attribute__((noinline)) void prep_emit(uint8_t *out, int idx) {
  const uint8_t *s = scratch + idx * 1536;
  for (int i = 0; i < 1536; i += 64) aie::store_v(out + i, aie::load_v<64>(s + i));
}
}
