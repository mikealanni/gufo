// Gated delta rule recurrence for one value head per core (state 128 x 128 bf16 held in L1).
// Token record (1024 B): q[128] bf16, k[128] bf16, v[128] bf16 (already conv/silu/l2norm/scale), decay f32 @768, beta f32 @772,
// reset i32 @776. Output record (512 B): o[128] f32.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" void __cxa_finalize(void *) {}

alignas(64) static bfloat16 S[128 * 128];

static inline aie::vector<float, 64> mulf(const aie::vector<float, 64> &a, const aie::vector<float, 64> &b) {
  return aie::mul(a, b).template to_vector<float>();
}

static constexpr int kRec = 1024;
static constexpr int kOutRec = 512;

__attribute__((noinline)) static void Token(const uint8_t *rec, uint8_t *out) {
  const bfloat16 *q = (const bfloat16 *)rec;
  const bfloat16 *k = q + 128;
  const bfloat16 *v = q + 256;
  const float decay = ((const float *)(rec + 768))[0];
  const float beta = ((const float *)(rec + 768))[1];
  const int reset = ((const int32_t *)(rec + 768))[2];
  if (reset) {
    const aie::vector<bfloat16, 64> z = aie::zeros<bfloat16, 64>();
    for (int i = 0; i < 256; ++i) aie::store_v(S + i * 64, z);
  }
  aie::accum<accfloat, 64> kv0 = aie::zeros<accfloat, 64>(), kv1 = aie::zeros<accfloat, 64>();
  aie::accum<accfloat, 64> sq0 = aie::zeros<accfloat, 64>(), sq1 = aie::zeros<accfloat, 64>();
  for (int i = 0; i < 128; ++i) {
    const aie::vector<bfloat16, 64> vk = aie::broadcast<bfloat16, 64>(k[i]);
    const aie::vector<bfloat16, 64> vq = aie::broadcast<bfloat16, 64>(q[i]);
    const aie::vector<bfloat16, 64> s0 = aie::load_v<64>(S + i * 128);
    const aie::vector<bfloat16, 64> s1 = aie::load_v<64>(S + i * 128 + 64);
    kv0 = aie::mac(kv0, s0, vk);
    kv1 = aie::mac(kv1, s1, vk);
    sq0 = aie::mac(sq0, s0, vq);
    sq1 = aie::mac(sq1, s1, vq);
  }
  float kq = 0.f;
  {
    aie::accum<accfloat, 64> a = aie::zeros<accfloat, 64>();
    a = aie::mac(a, aie::load_v<64>(k), aie::load_v<64>(q));
    a = aie::mac(a, aie::load_v<64>(k + 64), aie::load_v<64>(q + 64));
    kq = aie::reduce_add(a.to_vector<float>());
  }
  const aie::vector<float, 64> vdec = aie::broadcast<float, 64>(decay);
  const aie::vector<float, 64> vbeta = aie::broadcast<float, 64>(beta);
  const aie::vector<float, 64> vkq = aie::broadcast<float, 64>(kq);
  const aie::vector<float, 64> kvf0 = kv0.to_vector<float>(), kvf1 = kv1.to_vector<float>();
  const aie::vector<float, 64> sqf0 = sq0.to_vector<float>(), sqf1 = sq1.to_vector<float>();
  aie::accum<accfloat, 64> vv0, vv1;
  vv0.from_vector(aie::load_v<64>(v));
  vv1.from_vector(aie::load_v<64>(v + 64));
  const aie::vector<float, 64> d0 = mulf(aie::sub(vv0.to_vector<float>(), mulf(kvf0, vdec)), vbeta);
  const aie::vector<float, 64> d1 = mulf(aie::sub(vv1.to_vector<float>(), mulf(kvf1, vdec)), vbeta);
  const aie::vector<float, 64> o0 = aie::add(mulf(sqf0, vdec), mulf(vkq, d0));
  const aie::vector<float, 64> o1 = aie::add(mulf(sqf1, vdec), mulf(vkq, d1));
  aie::store_v((float *)out, o0);
  aie::store_v((float *)out + 64, o1);
  aie::accum<accfloat, 64> t0a, t1a, h0a, h1a, l0a, l1a;
  t0a.from_vector(d0);
  t1a.from_vector(d1);
  const aie::vector<bfloat16, 64> dh0 = t0a.to_vector<bfloat16>();
  const aie::vector<bfloat16, 64> dh1 = t1a.to_vector<bfloat16>();
  h0a.from_vector(dh0);
  h1a.from_vector(dh1);
  l0a.from_vector(aie::sub(d0, h0a.to_vector<float>()));
  l1a.from_vector(aie::sub(d1, h1a.to_vector<float>()));
  const aie::vector<bfloat16, 64> dl0 = l0a.to_vector<bfloat16>();
  const aie::vector<bfloat16, 64> dl1 = l1a.to_vector<bfloat16>();
  const bfloat16 dhi = (bfloat16)decay;
  const bfloat16 dlo = (bfloat16)(decay - (float)dhi);
  const aie::vector<bfloat16, 64> vdh = aie::broadcast<bfloat16, 64>(dhi);
  const aie::vector<bfloat16, 64> vdl = aie::broadcast<bfloat16, 64>(dlo);
  for (int i = 0; i < 128; ++i) {
    const aie::vector<bfloat16, 64> vk = aie::broadcast<bfloat16, 64>(k[i]);
    const aie::vector<bfloat16, 64> s0 = aie::load_v<64>(S + i * 128);
    const aie::vector<bfloat16, 64> s1 = aie::load_v<64>(S + i * 128 + 64);
    aie::accum<accfloat, 64> a0 = aie::mul(s0, vdh);
    aie::accum<accfloat, 64> a1 = aie::mul(s1, vdh);
    a0 = aie::mac(a0, s0, vdl);
    a1 = aie::mac(a1, s1, vdl);
    a0 = aie::mac(a0, vk, dh0);
    a1 = aie::mac(a1, vk, dh1);
    a0 = aie::mac(a0, vk, dl0);
    a1 = aie::mac(a1, vk, dl1);
    aie::store_v(S + i * 128, a0.to_vector<bfloat16>());
    aie::store_v(S + i * 128 + 64, a1.to_vector<bfloat16>());
  }
}

extern "C" void gdn_begin() { ::aie::set_rounding(::aie::rounding_mode::conv_even); }

extern "C" void gdn_chunk(const uint8_t *in, uint8_t *out) {
  for (int t = 0; t < 4; ++t) Token(in + t * kRec, out + t * kOutRec);
}
