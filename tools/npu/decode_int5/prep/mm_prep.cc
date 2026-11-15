// Attention prep on the C wire layout of the QKV GEMM: QK-norm + RoPE (+ V block transpose), emitting the
// attention kernel's Q block (9216 B) and K/V chunk (4608 B) as 1536 B pieces.
// Stripe objects (4608 B): two head stripes of a Q block, or the K and V stripe of a chunk.
// Table object (5120 B): rope table @0 (4096), q norm weight @4096, k norm weight @4352, eps (bf16) @4608.
// Table: [br 2][iblock 8][cos,sin] bf16 vectors (lane = row*8+col): cos/sin of token (br*8+row), i = iblock*8+col.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" void __cxa_finalize(void *) {}

__attribute__((aligned(64))) const uint8_t id_blk[72] = {
    127, 64, 0,  0,  0,  0,  0,  0,  0,  127, 0,  64, 0,  0,  0,  0,  0,  0,
    127, 0,  0,  64, 0,  0,  0,  0,  0,  127, 0,  0,  0,  64, 0,  0,  0,  0,
    127, 0,  0,  0,  0,  64, 0,  0,  0,  127, 0,  0,  0,  0,  0,  64, 0,  0,
    127, 0,  0,  0,  0,  0,  0,  64, 0,  127, 0,  0,  0,  0,  0,  0,  0,  64};


alignas(64) static bfloat16 dec[32 * 64];
alignas(64) static uint8_t scratch[9216];   // Q block, 4 head stripes
alignas(64) static uint8_t skv[4608];       // K/V chunk
alignas(64) static uint8_t stl[9216];       // two tail blocks (K bfp 1152 + V bf16 2048, rest zero)
alignas(64) static float qn[512];          // |q|^2 per hq: [hq][br*64 + row*8 + j]
alignas(64) static float vmax[8 * 64];
alignas(64) static uint8_t hdrbuf[1536];
alignas(64) static float mtmp[64];
alignas(64) static const int32_t rowv[64] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7};
alignas(64) static const int32_t colv[64] = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
alignas(64) static float knorm[128];

__attribute__((noinline)) static aie::vector<bfloat16, 64> rep8(const bfloat16 *p) {
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

// dec holds one decoded stripe; out receives 2304 B in destination order (br*16 + ks). Float arithmetic throughout:
// the products of bf16 values are exact in the accumulator and only the final bfp encode rounds.
alignas(64) static bfloat16 yhv[64];
alignas(64) static bfloat16 ylv[64];

static inline aie::vector<float, 64> fv(const aie::vector<bfloat16, 64> &v) {
  aie::accum<accfloat, 64> a;
  a.from_vector(v);
  return a.template to_vector<float>();
}

// Float product from bf16 pieces: hi*hi + hi*lo + lo*hi, exact products accumulated in the float accumulator.
static inline aie::vector<float, 64> fmulv(const aie::vector<float, 64> &a, const aie::vector<float, 64> &b) {
  const aie::vector<bfloat16, 64> ah = aie::accum<accfloat, 64>(a).template to_vector<bfloat16>();
  const aie::vector<bfloat16, 64> bh = aie::accum<accfloat, 64>(b).template to_vector<bfloat16>();
  const aie::vector<bfloat16, 64> al = aie::accum<accfloat, 64>(aie::sub(a, fv(ah))).template to_vector<bfloat16>();
  const aie::vector<bfloat16, 64> bl = aie::accum<accfloat, 64>(aie::sub(b, fv(bh))).template to_vector<bfloat16>();
  aie::accum<accfloat, 64> acc = aie::mul(ah, bh);
  acc = aie::mac(acc, ah, bl);
  acc = aie::mac(acc, al, bh);
  return acc.template to_vector<float>();
}

static inline void norm_rope(const bfloat16 *tab, const bfloat16 *w, bfloat16 eps_b, uint8_t *out, float *kn) {
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::block_vector<bfp16ebs8, 64> ob = aie::accum<accfloat, 64>(one).template to_vector<bfp16ebs8>();
  const aie::vector<float, 64> eps = fv(aie::broadcast<bfloat16, 64>(eps_b));
  const aie::vector<float, 64> inv_n = aie::broadcast<float, 64>(1.0f / 128.0f);
  const aie::vector<float, 64> c05 = aie::broadcast<float, 64>(0.5f);
  const aie::vector<float, 64> c15 = aie::broadcast<float, 64>(1.5f);
  for (int br = 0; br < 2; ++br) {
    aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
    for (int sbc = 0; sbc < 8; ++sbc)
      for (int bc = 0; bc < 2; ++bc) {
        const aie::vector<bfloat16, 64> v = aie::load_v<64>(dec + (sbc * 4 + br * 2 + bc) * 64);
        acc = aie::mac(acc, v, v);
      }
    aie::store_v(mtmp, acc.template to_vector<float>());
    float rs[8];
    for (int r = 0; r < 8; ++r) rs[r] = aie::reduce_add(aie::load_v<8>(mtmp + r * 8));
    const aie::vector<float, 64> sv = aie::concat(aie::broadcast<float, 8>(rs[0]), aie::broadcast<float, 8>(rs[1]), aie::broadcast<float, 8>(rs[2]), aie::broadcast<float, 8>(rs[3]), aie::broadcast<float, 8>(rs[4]), aie::broadcast<float, 8>(rs[5]), aie::broadcast<float, 8>(rs[6]), aie::broadcast<float, 8>(rs[7]));
    const aie::vector<float, 64> ms = aie::add(fmulv(sv, inv_n), eps);
    const aie::vector<int32, 64> bi = ms.template cast_to<int32>();
    aie::vector<float, 64> y = aie::sub(aie::broadcast<int32, 64>(0x5f3759df), aie::downshift(bi, 1)).template cast_to<float>();
    const aie::vector<float, 64> hm = fmulv(ms, c05);
    for (int it = 0; it < 3; ++it) y = fmulv(y, aie::sub(c15, fmulv(fmulv(y, y), hm)));
    const aie::vector<bfloat16, 64> yh = aie::accum<accfloat, 64>(y).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 64> yl = aie::accum<accfloat, 64>(aie::sub(y, fv(yh))).template to_vector<bfloat16>();
    aie::accum<accfloat, 64> ns = aie::zeros<accfloat, 64>();
    for (int sbc = 0; sbc < 4; ++sbc)
      for (int bc = 0; bc < 2; ++bc) {
        const int ia = sbc * 4 + br * 2 + bc, ib = (sbc + 4) * 4 + br * 2 + bc;
        const int ib8 = sbc * 2 + bc;
        const aie::vector<bfloat16, 64> wa = rep8(w + sbc * 16 + bc * 8);
        const aie::vector<bfloat16, 64> wb = rep8(w + 64 + sbc * 16 + bc * 8);
        const aie::vector<bfloat16, 64> xa = aie::mul(aie::load_v<64>(dec + ia * 64), wa).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> xb = aie::mul(aie::load_v<64>(dec + ib * 64), wb).template to_vector<bfloat16>();
        aie::accum<accfloat, 64> aa = aie::mul(xa, yh);
        aa = aie::mac(aa, xa, yl);
        aie::accum<accfloat, 64> ab = aie::mul(xb, yh);
        ab = aie::mac(ab, xb, yl);
        const aie::vector<bfloat16, 64> a = aa.template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> b = ab.template to_vector<bfloat16>();
        if (kn) {
          ns = aie::mac(ns, a, a);
          ns = aie::mac(ns, b, b);
        }
        const aie::vector<bfloat16, 64> cs = aie::load_v<64>(tab + ((br * 8 + ib8) * 2) * 64);
        const aie::vector<bfloat16, 64> sn = aie::load_v<64>(tab + ((br * 8 + ib8) * 2 + 1) * 64);
        aie::accum<accfloat, 64> lo = aie::mul(a, cs);
        lo = aie::msc(lo, b, sn);
        aie::accum<accfloat, 64> hi = aie::mul(b, cs);
        hi = aie::mac(hi, a, sn);
        put(out + (br * 16 + ib8) * 72, lo);
        put(out + (br * 16 + ib8 + 8) * 72, hi);
      }
    if (kn) {
      const aie::accum<accfloat, 64> rs2 = mac_8x8_8x8T(ns.template to_vector<bfp16ebs8>(), ob, z);
      aie::store_v(kn + br * 64, rs2.template to_vector<float>());
    }
  }
}

constexpr int kTW = 4096, kTWK = 4352, kTEps = 4608, kTHdr = 4864, kTKb = 4992;

extern "C" {
// Two head stripes (hq = 2*pair, 2*pair+1) of a Q block; t: table object (rope table, q/k norm weights, eps).
__attribute__((noinline)) void prep_q2(const uint8_t *s, const uint8_t *t, int pair) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < 2; ++i) {
    decode_stripe(s + i * 2304);
    norm_rope((const bfloat16 *)t, (const bfloat16 *)(t + kTW), *(const bfloat16 *)(t + kTEps), scratch + (pair * 2 + i) * 2304, qn + (pair * 2 + i) * 128);
  }
}

// K stripe then V stripe: K/V chunk (4608 B) into skv, two tail blocks into stl, |k|^2 into knorm.
__attribute__((noinline)) void prep_kv2(const uint8_t *s, const uint8_t *t) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  decode_stripe(s);
  norm_rope((const bfloat16 *)t, (const bfloat16 *)(t + kTWK), *(const bfloat16 *)(t + kTEps), skv, knorm);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> idst((bfp16ebs8 *)id_blk);
  idst.seek(0);
  const aie::block_vector<bfp16ebs8, 64> idb = idst.pop();
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> in((const bfp16ebs8 *)(s + 2304));
  in.seek(0);
  for (int sbc = 0; sbc < 8; ++sbc)
    for (int br = 0; br < 2; ++br)
      for (int bc = 0; bc < 2; ++bc) {
        const aie::block_vector<bfp16ebs8, 64> g = in.pop();
        const aie::accum<accfloat, 64> tv = mac_8x8_8x8T(idb, g, z);
        put(skv + 2304 + (br * 16 + sbc * 2 + bc) * 72, tv);
        const aie::accum<accfloat, 64> dv = mac_8x8_8x8T(g, idb, z);
        aie::store_v((bfloat16 *)(stl + br * 4608 + 1152 + (sbc * 2 + bc) * 128), dv.template to_vector<bfloat16>());
      }
  for (int br = 0; br < 2; ++br) {
    uint8_t *tl = stl + br * 4608;
    for (int i = 0; i < 1152; i += 64) aie::store_v(tl + i, aie::load_v<64>(skv + br * 1152 + i));
    const aie::vector<uint8_t, 64> zv = aie::zeros<uint8_t, 64>();
    for (int i = 3200; i < 4608; i += 64) aie::store_v(tl + i, zv);
  }
}

// Softmax bound probe over one K/V chunk object of the key history; hidx 0..2: chunks q-2..q, 3: chunk 0 (first 8 keys).
__attribute__((noinline)) void prep_kh(const uint8_t *s, const uint8_t *t, int hidx) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const int32_t pos = *(const int32_t *)(t + kTHdr);
  const aie::vector<int32, 64> rv = aie::load_v<64>(rowv), cv = aie::load_v<64>(colv);
  const aie::vector<float, 64> neg = aie::broadcast<float, 64>(-1e30f);
  const int nkb = hidx == 3 ? 1 : 2;
  for (int kb = 0; kb < nkb; ++kb) {
    const int tok0 = hidx == 3 ? 0 : pos - 32 + hidx * 16 + kb * 8;
    const aie::vector<int32, 64> tokkey = aie::add(aie::broadcast<int32, 64>(tok0), cv);
    for (int rb = 0; rb < 8; ++rb) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> qs((const bfp16ebs8 *)(scratch + (rb >> 1) * 2304 + (rb & 1) * 1152));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ks((const bfp16ebs8 *)(s + kb * 1152));
      qs.seek(0);
      ks.seek(0);
      aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
      for (int k = 0; k < 16; ++k) acc = mac_8x8_8x8T(qs.pop(), ks.pop(), acc);
      const aie::vector<float, 64> sc = acc.template to_vector<float>();
      const aie::vector<int32, 64> tokrow = aie::add(aie::broadcast<int32, 64>(pos + (rb & 1) * 8), rv);
      const aie::mask<64> near = aie::ge(tokkey, aie::sub(tokrow, aie::broadcast<int32, 64>(31)));
      const aie::mask<64> sink = aie::le(tokkey, aie::broadcast<int32, 64>(7));
      aie::vector<float, 64> v = aie::max(aie::select(neg, sc, near), aie::select(neg, sc, sink));
      v = aie::select(neg, v, aie::le(tokkey, tokrow));
      v = aie::select(neg, v, aie::ge(tokkey, aie::broadcast<int32, 64>(0)));
      aie::store_v(vmax + rb * 64, aie::max(aie::load_v<64>(vmax + rb * 64), v));
    }
  }
}

__attribute__((noinline)) void prep_mbegin() {
  const aie::vector<float, 64> neg = aie::broadcast<float, 64>(-1e30f);
  for (int rb = 0; rb < 8; ++rb) aie::store_v(vmax + rb * 64, neg);
}

alignas(64) static float mkb[64];

__attribute__((noinline)) static void mfin_rb(int rb, int32_t nrows) {
  const aie::vector<float, 64> kbv = aie::load_v<64>(mkb);
  const aie::vector<float, 64> c100 = aie::broadcast<float, 64>(100.0f);
  const aie::vector<float, 64> zero = aie::zeros<float, 64>();
  const aie::vector<int32, 64> rv = aie::load_v<64>(rowv);
  const aie::vector<float, 64> vm = aie::load_v<64>(vmax + rb * 64);
  float m8[8];
  for (int r = 0; r < 8; ++r) m8[r] = aie::reduce_max(vm.template extract<8>(r));
  const aie::vector<float, 64> mrow = aie::concat(aie::broadcast<float, 8>(m8[0]), aie::broadcast<float, 8>(m8[1]), aie::broadcast<float, 8>(m8[2]), aie::broadcast<float, 8>(m8[3]), aie::broadcast<float, 8>(m8[4]), aie::broadcast<float, 8>(m8[5]), aie::broadcast<float, 8>(m8[6]), aie::broadcast<float, 8>(m8[7]));
  const aie::vector<float, 64> q2 = aie::load_v<64>(qn + (rb >> 1) * 128 + (rb & 1) * 64);
  const aie::vector<float, 64> qb = aie::sqrt(q2);
  const aie::vector<float, 64> bound = aie::mul(qb, kbv).template to_vector<float>();
  aie::vector<float, 64> m = aie::max(mrow, aie::sub(bound, c100));
  m = aie::min(m, bound);
  const aie::vector<int32, 64> trow = aie::add(aie::broadcast<int32, 64>((rb & 1) * 8), rv);
  m = aie::select(zero, m, aie::lt(trow, aie::broadcast<int32, 64>(nrows)));
  aie::store_v(mtmp, m);
  for (int r = 0; r < 8; ++r) ((float *)(hdrbuf + 128))[rb * 8 + r] = mtmp[r * 8];
}

// Header piece: [128 B ints from the table object][M[64] floats]; row = kv head index for the key bound.
__attribute__((noinline)) void prep_mfin(const uint8_t *t, int row) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < 1536; i += 64) aie::store_v(hdrbuf + i, aie::zeros<uint8_t, 64>());
  for (int i = 0; i < 128; i += 64) aie::store_v(hdrbuf + i, aie::load_v<64>(t + kTHdr + i));
  const int32_t nrows = *(const int32_t *)(t + kTHdr + 4);
  aie::store_v(mkb, aie::broadcast<float, 64>(*(const float *)(t + kTKb + row * 4)));
  for (int rb = 0; rb < 8; ++rb) mfin_rb(rb, nrows);
}

// mode 0 (K/V dispatch) pieces 0..15: chunk(3), Q block(6), tails(6), |k|^2(1); mode 1 (Q dispatch) pieces 0..6: header, Q block(6).
__attribute__((noinline)) void prep_emit(uint8_t *out, int mode, int idx) {
  const uint8_t *s;
  if (mode == 0) {
    if (idx == 15) {
      for (int i = 0; i < 1536; i += 64) aie::store_v(out + i, aie::zeros<uint8_t, 64>());
      for (int i = 0; i < 128; i += 16) aie::store_v((float *)out + i, aie::load_v<16>(knorm + i));
      return;
    }
    s = idx < 3 ? skv + idx * 1536 : idx < 9 ? scratch + (idx - 3) * 1536 : stl + (idx - 9) * 1536;
  } else {
    s = idx == 0 ? hdrbuf : scratch + (idx - 1) * 1536;
  }
  for (int i = 0; i < 1536; i += 64) aie::store_v(out + i, aie::load_v<64>(s + i));
}
}
