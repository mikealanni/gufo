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
alignas(64) static float knorm[128];       // |k|^2 of the 16 tokens, lane row*8 of each 64-lane row sum

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
static inline void norm_rope(const bfloat16 *tab, const bfloat16 *w, bfloat16 eps_b, uint8_t *out, float *kn) {
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
    aie::accum<accfloat, 64> ns = aie::zeros<accfloat, 64>();
    for (int sbc = 0; sbc < 4; ++sbc)
      for (int bc = 0; bc < 2; ++bc) {
        const int ia = sbc * 4 + br * 2 + bc, ib = (sbc + 4) * 4 + br * 2 + bc;
        const int ib8 = sbc * 2 + bc;
        const aie::vector<bfloat16, 64> wa = rep8(w + sbc * 16 + bc * 8);
        const aie::vector<bfloat16, 64> wb = rep8(w + 64 + sbc * 16 + bc * 8);
        const aie::vector<bfloat16, 64> a = aie::mul(aie::mul(aie::load_v<64>(dec + ia * 64), y).template to_vector<bfloat16>(), wa).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> b = aie::mul(aie::mul(aie::load_v<64>(dec + ib * 64), y).template to_vector<bfloat16>(), wb).template to_vector<bfloat16>();
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
      const aie::accum<accfloat, 64> rs = mac_8x8_8x8T(ns.template to_vector<bfp16ebs8>(), ob, z);
      aie::store_v(kn + br * 64, rs.template to_vector<float>());
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

// Header piece: [128 B ints from the table object][M[64] floats]; row = kv head index for the key bound.
__attribute__((noinline)) void prep_mfin(const uint8_t *t, int row) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < 1536; i += 64) aie::store_v(hdrbuf + i, aie::zeros<uint8_t, 64>());
  for (int i = 0; i < 128; i += 64) aie::store_v(hdrbuf + i, aie::load_v<64>(t + kTHdr + i));
  const int32_t nrows = *(const int32_t *)(t + kTHdr + 4);
  const float kb = *(const float *)(t + kTKb + row * 4);
  const aie::vector<float, 64> kbv = aie::broadcast<float, 64>(kb);
  const aie::vector<float, 64> c100 = aie::broadcast<float, 64>(100.0f);
  const aie::vector<float, 64> zero = aie::zeros<float, 64>();
  const aie::vector<int32, 64> rv = aie::load_v<64>(rowv);
  for (int rb = 0; rb < 8; ++rb) {
    const aie::vector<float, 64> vm = aie::load_v<64>(vmax + rb * 64);
    float m8[8];
    for (int r = 0; r < 8; ++r) m8[r] = aie::reduce_max(vm.template extract<8>(r));
    aie::vector<float, 8> a0 = aie::load_v<8>(m8);
    aie::vector<float, 64> mrow = aie::concat(aie::broadcast<float, 8>(m8[0]), aie::broadcast<float, 8>(m8[1]), aie::broadcast<float, 8>(m8[2]), aie::broadcast<float, 8>(m8[3]), aie::broadcast<float, 8>(m8[4]), aie::broadcast<float, 8>(m8[5]), aie::broadcast<float, 8>(m8[6]), aie::broadcast<float, 8>(m8[7]));
    (void)a0;
    const aie::vector<float, 64> q2 = aie::load_v<64>(qn + (rb >> 1) * 128 + (rb & 1) * 64);
    const aie::vector<float, 64> qb = aie::sqrt(q2);
    const aie::accum<accfloat, 64> bacc = aie::mul(qb, kbv);
    const aie::vector<float, 64> bound = bacc.template to_vector<float>();
    aie::vector<float, 64> m = aie::max(mrow, aie::sub(bound, c100));
    m = aie::min(m, bound);
    const aie::vector<int32, 64> trow = aie::add(aie::broadcast<int32, 64>((rb & 1) * 8), rv);
    m = aie::select(zero, m, aie::lt(trow, aie::broadcast<int32, 64>(nrows)));
    aie::store_v(mtmp, m);
    for (int r = 0; r < 8; ++r) ((float *)(hdrbuf + 128))[rb * 8 + r] = mtmp[r * 8];
  }
}

// kind 0: K/V chunk piece, 1: Q block piece, 2: tail piece, 3: |k|^2 vector.
__attribute__((noinline)) void prep_emit(uint8_t *out, int kind, int idx) {
  if (kind == 4) {
    for (int i = 0; i < 1536; i += 64) aie::store_v(out + i, aie::load_v<64>(hdrbuf + i));
    return;
  }
  if (kind == 3) {
    for (int i = 0; i < 1536; i += 64) aie::store_v(out + i, aie::zeros<uint8_t, 64>());
    for (int i = 0; i < 128; i += 16) aie::store_v((float *)out + i, aie::load_v<16>(knorm + i));
    return;
  }
  const uint8_t *s = (kind == 0 ? skv : kind == 1 ? scratch : stl) + idx * 1536;
  for (int i = 0; i < 1536; i += 64) aie::store_v(out + i, aie::load_v<64>(s + i));
}
}
