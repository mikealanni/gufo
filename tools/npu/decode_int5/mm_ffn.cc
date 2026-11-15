#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// WP10: the AIE C runtime drags __cxa_finalize (atexit.o) + float32_lt (softfloat) into
// every core's program memory via the exit path, which never runs on the NPU. Stub it.
extern "C" void __cxa_finalize(void *) {}

constexpr int kRows = 16;
constexpr int kCols = 32;
constexpr int kK = 64;

// One arena holds everything live in L1: gemm scratch (m64: c_stage 4 KB + b_scratch +
// comb + bbuf = 9 KB from +8192) aliases the attention output accumulator (a_o), since
// gemm and attention dispatches never overlap. attn keeps 0..8192 for its row state,
// a_o lives at +8192..+40960, a_chunk at +40960.
alignas(128) uint8_t arena[42496];
// GEMM scratch sits at the very start of L1 data (same banks as the original 6 KB layout); it overlays the
// attention row state, which is re-initialised by every attention dispatch.
static bfloat16 *const c_stage = (bfloat16 *)(arena + 0);
static bfp16ebs8 *const b_scratch = (bfp16ebs8 *)(arena + 4096);
static int g_mrows = 1;  // A row-blocks per B tile (1 = decode, 4 = m64 prefill)
static int g_m = 0;      // current row-block within the current B tile
// Flushes write their result for all row-blocks into outbuf ([mb][out_unit bytes]); c_emit then hands
// one 1 KB C element per row-block to the C fifo (keeps the C stream at the decode size).
static uint8_t *const outbuf = arena + 27904;
static int out_unit = 64;

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
#ifdef INT4
  const aie::vector<uint16, 64> mask = aie::broadcast<uint16, 64>(15);
  const aie::vector<bfloat16, 64> m144 = aie::broadcast<bfloat16, 64>((bfloat16)-136.0f);  // 128 (exponent trick) + bias 8
#else
  const aie::vector<uint16, 64> mask = aie::broadcast<uint16, 64>(31);
  const aie::vector<bfloat16, 64> m144 = aie::broadcast<bfloat16, 64>((bfloat16)-144.0f);
#endif
  const aie::vector<uint16, 64> magic = aie::broadcast<uint16, 64>(0x4300);
  for (int p = 0; p < 32; p++) {
    const int sb = p >> 4, sr = (p >> 1) & 7, blk = p & 1;
#ifdef INT4
    aie::vector<uint16, 64> w = aie::load_v<64>(pw + (p / 4) * 64);  // four 4-bit values per 16-bit word
    aie::vector<uint16, 64> x = aie::bit_or(aie::bit_and(aie::downshift(w, 4 * (p % 4)), mask), magic);
#else
    aie::vector<uint16, 64> w = aie::load_v<64>(pw + (p / 3) * 64);
    aie::vector<uint16, 64> x = aie::bit_or(aie::bit_and(aie::downshift(w, 5 * (p % 3)), mask), magic);
#endif
    aie::vector<bfloat16, 64> s = ExpandScales(ps + (((sb * 2 + blk) * 2 + (sr >> 2)) * 8));
    aie::accum<accfloat, 64> a = aie::mul(x.cast_to<bfloat16>(), s);
    a = aie::mac(a, m144, s);
    out.push(a.template to_vector<bfp16ebs8>());
  }
}

static inline __attribute__((always_inline)) void MacTile(bfp16ebs8 *__restrict pA, bfloat16 *__restrict pC,
                                                                  int first) {
  for (int sb = 0; sb < kCols / 16; sb++) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB_stream(b_scratch);
    pB_stream.seek(sb * (kK / 8) * 2);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream(pA);
    pA_stream.seek(0);
    aie::vector<bfloat16, 64> v0 = aie::load_v<64>(pC), v1 = aie::load_v<64>(pC + 64),
                                      v2 = aie::load_v<64>(pC + 128), v3 = aie::load_v<64>(pC + 192);
    if (first) {  // first k-tile: start the C accumulator at zero (no zero_kernel call)
      const aie::vector<bfloat16, 64> z = aie::zeros<bfloat16, 64>();
      v0 = z; v1 = z; v2 = z; v3 = z;
    }
    aie::accum<accfloat, 64> acc0(v0);
    aie::accum<accfloat, 64> acc1(v1);
    aie::accum<accfloat, 64> acc2(v2);
    aie::accum<accfloat, 64> acc3(v3);
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
  for (int mb = 0; mb < g_mrows; mb++)
    for (int sb = 0; sb < 2; sb++)
      for (int br = 0; br < 2; br++) {
        const bfloat16 *base = c_stage + mb * 512;
        aie::vector<bfloat16, 64> gb = aie::load_v<64>(base + sb * 256 + (br * 2) * 64);
        aie::vector<bfloat16, 64> ub = aie::load_v<64>(base + sb * 256 + (br * 2 + 1) * 64);
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
  for (int i = 0; i < g_mrows * kRows * kCols / 64; i++) {
    aie::accum<accfloat, 64> a(aie::load_v<64>(c_stage + i * 64));
    out.push(a.template to_vector<bfp16ebs8>());
  }
}

// Per-strip argmax reduce: 16 rows x 32 cols -> (float max, uint32 col) per row.
__attribute__((noinline)) static void FlushArgmax(uint8_t *__restrict pC) {
  // WP10: compare IEEE-754 order-preserved uint keys, not floats: AIE has no scalar float
  // compare and the softfloat helper costs 186 B of the 16 KB program memory.
  float bv[kRows];
  uint32_t bc[kRows];
  uint32_t bk[kRows];
  for (int r = 0; r < kRows; ++r) { bv[r] = -3.4e38f; bc[r] = 0; bk[r] = 0; }
  for (int sb = 0; sb < 2; ++sb) {
    for (int acc = 0; acc < 4; ++acc) {
      const int rowbase = (acc >= 2) ? 8 : 0;
      const uint32_t colbase = (uint32_t)(sb * 16 + (acc & 1) * 8);
      const bfloat16 *p = c_stage + sb * 256 + acc * 64;
      for (int rr = 0; rr < 8; ++rr) {
        const int row = rowbase + rr;
        const bfloat16 *pr = p + rr * 8;
        for (int c = 0; c < 8; ++c) {
          const float v = (float)pr[c];
          const uint32_t fbits = (uint32_t)*(const uint16_t *)(pr + c) << 16;
          const uint32_t key = fbits & 0x80000000u ? ~fbits : (fbits | 0x80000000u);
          if (key > bk[row]) { bk[row] = key; bv[row] = v; bc[row] = colbase + (uint32_t)c; }
        }
      }
    }
  }
  for (int r = 0; r < kRows; ++r) {
    ((float *)pC)[r * 2] = bv[r];
    ((uint32_t *)pC)[r * 2 + 1] = bc[r];
  }
}


// Conv epilogue (WP10/AGENT-6): supports mrows row-blocks in one dispatch so a 64-row
// chunk needs no mid-chunk host history commit (avoids W4 weight-sync visibility race).
// comb = 4 cb * (8 + 16*mrows) rows * 8 cols at arena+14848; bbuf follows comb.
// mrows=1: comb ends 16384, bbuf at 16384 (1024B) - matches original layout.
static bfloat16 *const comb = (bfloat16 *)(arena + 14848);
static int strip_ctr = 0;

static inline aie::vector<bfloat16, 64> Rep8(const bfloat16 *p) {
  aie::vector<bfloat16, 8> v = aie::load_v<8>(p);
  aie::vector<bfloat16, 16> v2 = aie::concat(v, v);
  aie::vector<bfloat16, 32> v4 = aie::concat(v2, v2);
  return aie::concat(v4, v4);
}

__attribute__((noinline)) static void FlushConv(const uint8_t *__restrict ctl_b,
                                                uint8_t *__restrict pC) {
  const int s = strip_ctr;
  strip_ctr = (s == 5) ? 0 : s + 1;
  const int role = s % 3;
  const int mr = g_mrows;
  out_unit = role == 1 ? 1024 : (role == 2 ? 576 : 64);
  const int comb_rows = 8 + 16 * mr;  // 8 (hist slot) + 16 rows per row-block
  if (role == 0) {
    for (int mb = 0; mb < mr; mb++)
      for (int cb = 0; cb < 4; cb++)
        for (int rh = 0; rh < 2; rh++)
          aie::store_v(comb + 4 * comb_rows * 8 + ((mb * 4 + cb) * 16 + rh * 8) * 8,
                       aie::load_v<64>(c_stage + mb * 512 + (cb >> 1) * 256 + (rh * 2 + (cb & 1)) * 64));
    return;
  }
  if (role == 1) {
    const bfloat16 *hist = (const bfloat16 *)(ctl_b + 512);
    for (int cb = 0; cb < 4; cb++) {
      bfloat16 *cbp = comb + cb * comb_rows * 8;
      for (int r = 0; r < 6; r++)
        aie::store_v(cbp + (2 + r) * 8, aie::load_v<8>(hist + (cb * 6 + r) * 8));
      for (int mb = 0; mb < mr; mb++)
        for (int rh = 0; rh < 2; rh++) {
          aie::vector<bfloat16, 64> bv = aie::load_v<64>(comb + 4 * comb_rows * 8 + ((mb * 4 + cb) * 16 + rh * 8) * 8);
          aie::vector<bfloat16, 64> xv =
              aie::load_v<64>(c_stage + mb * 512 + (cb >> 1) * 256 + (rh * 2 + (cb & 1)) * 64);
          aie::accum<accfloat, 64> a = aie::mul(bv, xv);
          aie::vector<bfloat16, 64> r64 = a.template to_vector<bfloat16>();
          aie::store_v(cbp + (8 + (mb * 2 + rh) * 8) * 8, r64);
          aie::store_v((bfloat16 *)(pC + (mb * 4 + cb) * 256) + rh * 64, r64);
        }
    }
    return;
  }
  const bfloat16 *taps = (const bfloat16 *)(ctl_b + 64);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)pC);
  out.seek(0);
  for (int mb = 0; mb < mr; mb++)
    for (int cb = 0; cb < 4; cb++) {
      const bfloat16 *cbp = comb + cb * comb_rows * 8;
      for (int rh = 0; rh < 2; rh++) {
        aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
        for (int t = 0; t < 7; t++) {
          aie::vector<bfloat16, 64> wv = Rep8(taps + (t * 4 + cb) * 8);
          aie::vector<bfloat16, 64> xv = aie::load_unaligned_v<64>(cbp + ((mb * 2 + rh) * 8 + 2 + t) * 8);
          acc = aie::mac(acc, wv, xv);
        }
        aie::vector<bfloat16, 64> sv = acc.template to_vector<bfloat16>();
        aie::vector<bfloat16, 64> cv =
            aie::load_v<64>(c_stage + mb * 512 + (cb >> 1) * 256 + (rh * 2 + (cb & 1)) * 64);
        aie::accum<accfloat, 64> y = aie::mul(cv, sv);
        out.push(y.template to_vector<bfp16ebs8>());
      }
    }
}

#ifdef INT4
// The 4-bit decode xclbin carries no attention programs (its weight element is 1152 B, not the 1536 B attention chunks):
// stubs keep the unified core program linkable and free its program memory.
#else
#include "attn_kernel.h"
#endif

extern "C" {
#ifdef INT4
void attn_begin(uint8_t *, int32_t) {}
void attn_q(uint8_t *, int32_t, int32_t) {}
void attn_tail(uint8_t *, int32_t, int32_t, int32_t) {}
void attn_feed(uint8_t *, int32_t) {}
void attn_out(uint8_t *, int32_t) {}
#else
void attn_begin(uint8_t *hdr, int32_t row_id) { AttnBegin(hdr, row_id); }
void attn_q(uint8_t *pA, int32_t idx, int32_t row_id) { AttnQ(pA, idx, row_id); }
void attn_tail(uint8_t *pA, int32_t idx, int32_t row_id, int32_t col_id) { AttnTailFeed(pA, idx, row_id, col_id); }
void attn_feed(uint8_t *pB, int32_t col) { AttnFeed(pB, col); }
void attn_out(uint8_t *pC, int32_t idx) { AttnOut(pC, idx); }
#endif

static int kc_p1 = 0;
static int kc_dn = 0;

// Shared k-tile advance: unpack once per B tile, mac mrows A-blocks, advance the k counter.
// Returns true when the C tile (all k-tiles) is complete.
__attribute__((noinline)) static bool TileStep(bfp16ebs8 *__restrict pA, uint8_t *__restrict pBp,
                                              int32_t kcnt, int *kc) {
#ifdef INT4
  if (g_m == 0) Unpack((const uint16_t *)pBp, (const bfloat16 *)(pBp + 1024));
#else
  if (g_m == 0) Unpack((const uint16_t *)pBp, (const bfloat16 *)(pBp + 1408));
#endif
  MacTile(pA, c_stage + g_m * kRows * kCols, *kc == 0);
  if (++g_m < g_mrows) return false;
  g_m = 0;
  if (*kc == kcnt - 1) {
    *kc = 0;
    return true;
  }
  *kc += 1;
  return false;
}

void matmul_p1(bfp16ebs8 *__restrict pA, uint8_t *__restrict pBp, int32_t kcnt, int32_t mode,
               int32_t mrows) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  g_mrows = mrows;
  if (mode == 1) {
    if (TileStep(pA, pBp, kcnt, &kc_p1)) FlushConv(pBp, outbuf);
    return;
  }
  if (TileStep(pA, pBp, kcnt, &kc_p1)) {
    out_unit = 288;
    FlushSwiglu((bfp16ebs8 *)outbuf);
  }
}

void matmul_dn(bfp16ebs8 *__restrict pA, uint8_t *__restrict pBp, int32_t kcnt, int32_t mode,
               int32_t mrows) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  g_mrows = mrows;
  if (!TileStep(pA, pBp, kcnt, &kc_dn)) return;
  if (mode == 2) {
    out_unit = 128;
    FlushArgmax(outbuf);
  } else {
    out_unit = 576;
    FlushPlain((bfp16ebs8 *)outbuf);
  }
}

// Hand row-block mb of the last flush to the C fifo element (1 KB).
// outbuf + mb * out_unit is only 32 B aligned for some units (288 B swiglu pieces): use unaligned loads.
void c_emit(uint8_t *__restrict pC, int32_t mb) {
  const uint8_t *src = outbuf + mb * out_unit;
  for (int i = 0; i < out_unit; i += 64) aie::store_v(pC + i, aie::load_unaligned_v<64>(src + i));
}

}  // extern "C"
