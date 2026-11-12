// cfg31: PLAIN gu C wire (gate/up vecs paired by pack_swi B packing) ->
// on-core silu(g)*u -> dn-A tile permute. Replaces BOTH the gu swiglu-flush
// epilogue (+9.7 ms/layer, don't put it in the gu) and cfg30's copy.
// Wire slot s of tile T: blocks 16s+0..7 = gate vec, 16s+8..15 = up vec
// (PMAP-verified). Output vec number == slot s (n==s: the "permute" is a
// drop-half + halve), so this is a pure streaming pass: one stream on w,
// pop (g,u) pairs in slot order, push results in order. NO manual seeks:
// a second in.seek() after the first fill returns zeros (HW-proven probe).
// silu = cfg16 recipe (mac_8x8_8x8T identity-unpack + exp2/inv sigmoid).
// exp2/inv are ~16-lane SFU ops with long latency; TWO slots are interleaved
// per iteration so the dependency chains dual-issue (NOLIB bisect: SFU cost
// was +9.3 ms on 4 cores, raw walk+push floor 3.26 ms).
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <cstdint>

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
#ifdef SWI31_NR   // vector NR reciprocal: aie::inv(bf16) compiles to 64 SCALAR
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

__attribute__((noinline)) void swi_swiglu_permute(const bfp16ebs8 *__restrict w,
                                                  bfp16ebs8 *__restrict o) {
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> in(w);
  in.seek(0);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(o);
  out.seek(0);
#ifndef SWI31_NOIDB  // idst-race test: id_blk stream not created when defined
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> idst((bfp16ebs8 *)id_blk);
  idst.seek(0);
  const aie::block_vector<bfp16ebs8, 64> idb = idst.pop();
#endif
  const aie::vector<bfloat16, 64> c_hi =
      aie::broadcast<bfloat16, 64>((bfloat16)-1.4453125f);
  const aie::vector<bfloat16, 64> c_lo =
      aie::broadcast<bfloat16, 64>((bfloat16)0.0026175f);
  const aie::vector<bfloat16, 64> one =
      aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  for (int s = 0; s < 128; ++s) {
#ifdef SWI31_2SLOT  // two slots in flight: hide SFU latency
    const aie::block_vector<bfp16ebs8, 64> g0 = in.pop();
    const aie::block_vector<bfp16ebs8, 64> u0 = in.pop();
    const aie::block_vector<bfp16ebs8, 64> g1 = in.pop();
    const aie::block_vector<bfp16ebs8, 64> u1 = in.pop();
    const aie::accum<accfloat, 64> a0 = silu_mul(g0, u0, idb, z, c_hi, c_lo, one);
    const aie::accum<accfloat, 64> a1 = silu_mul(g1, u1, idb, z, c_hi, c_lo, one);
    out.push(a0.template to_vector<bfp16ebs8>());
    out.push(a1.template to_vector<bfp16ebs8>());
#else
    const aie::block_vector<bfp16ebs8, 64> g = in.pop();
    const aie::block_vector<bfp16ebs8, 64> u = in.pop();
#ifdef SWI31_RAW  // bring-up: pass gate vecs through, test walk+push floor
    out.push(g);
    continue;
#endif
#ifdef SWI31_NOIDB
    const aie::block_vector<bfp16ebs8, 64> idb = g;  // unused-value flow test
#endif
    out.push(silu_mul(g, u, idb, z, c_hi, c_lo, one).template to_vector<bfp16ebs8>());
#endif
  }
}
}
