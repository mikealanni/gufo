// Decode attention for one kv head group (4 q heads x 16 rows) over a slice of the context.
// Static softmax bound: p = exp2(s - M_row), M_row supplied by the host, so partials add.
//
// Structural constants (documented constraint, WP8): this kernel binds
//   * head_dim = 128  (16 dim-blocks of 8 per k or V^T, one O tile per row-block x dim-block),
//   * kv_heads = 4    (the 4 core rows = one kv head per row; kAttnKvh in the runtime),
//   * 4 q heads per kv head and 16 rows per pass (heads*rows = 64 rows = 8 row-blocks),
//   * 16-token chunks (3 stream elements: 2 x K + 1 x V^T) and 8 columns per chunk row.
// Any change to these requires matching changes in n32_core_placed.py (mode attn), the
// runtime Attn* layout, and the engine, which refuses other shapes (engine.cpp checks
// kvh == 4 && hd == 128 && heads == 16 before enabling the NPU attention path).
#ifndef GUFO_ATTN_KERNEL_H_
#define GUFO_ATTN_KERNEL_H_

// One shared copy loop keeps program memory (16 KB) in budget.
__attribute__((noinline)) static void CopyBytes(uint8_t *dst, const uint8_t *src, int n) {
  for (int i = 0; i < n; i += 64) aie::store_v(dst + i, aie::load_v<64>(src + i));
}

constexpr int kAttnRowBlocks = 8;   // 64 q rows = 8 blocks of 8
constexpr int kAttnDimBlocks = 16;  // 128 dims

alignas(128) bfp16ebs8 a_q[kAttnRowBlocks * 16 * 8];       // [rb][ks] blocks of 8 groups
alignas(128) uint8_t a_chunk[4608];                          // K (256 groups) then V^T (256 groups)
alignas(128) float a_o[kAttnRowBlocks * kAttnDimBlocks * 64];
static float *const a_m = (float *)(arena + 0);       // 64 rows
static int32_t *const a_hi = (int32_t *)(arena + 256);  // last token visible to the streamed (finalized-V) path
static int32_t *const a_pp = (int32_t *)(arena + 512);  // row position (last token visible to the tail path)
static int32_t *const a_lo = (int32_t *)(arena + 768);  // row window floor: first key valid for the row
static float *const a_l = (float *)(arena + 4096);
static int a_pos = 0, a_n = 0, a_feed = 0, a_chunk_idx = 0;
static int a_qhi = 0, a_win = 0;      // ring: absolute chunk of newest tokens; sliding window size (0 = full ctx)
static int a_tab[32];                 // ring: slot chunk -> qhi - absolute chunk (-1 = stale slot)

template <typename T>
static inline aie::vector<T, 64> RepEach8(const aie::vector<T, 8> &s) {
  auto z1 = aie::interleave_zip(s, s, 1);
  aie::vector<T, 16> s16 = aie::concat(z1.first, z1.second);
  auto z2 = aie::interleave_zip(s16, s16, 1);
  aie::vector<T, 32> s32 = aie::concat(z2.first, z2.second);
  auto z3 = aie::interleave_zip(s32, s32, 1);
  return aie::concat(z3.first, z3.second);
}

static inline aie::vector<int32, 64> LaneCol() {
  alignas(128) static const int32_t tbl[64] = {
      0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  return aie::load_v<64>(tbl);
}

// 2^a for a <= 0 with bf16 result: split a = n + f (n integer) so the bf16 exp2 argument stays small.
static inline aie::vector<bfloat16, 64> Exp2Wide(const aie::vector<float, 64> &a) {
  aie::vector<int32, 64> n = aie::to_fixed<int32>(a, 0);
  aie::vector<float, 64> f = aie::sub(a, aie::to_float<float>(n, 0));
  aie::vector<bfloat16, 64> p = aie::exp2<bfloat16>(f);
  aie::vector<int32, 64> lim = aie::broadcast<int32, 64>(-120);
  aie::vector<int32, 64> nc = aie::max(n, lim);
  aie::vector<int16, 64> n16 = aie::pack(nc);
  aie::vector<int16, 64> bits = aie::add(p.cast_to<int16>(), aie::upshift(n16, 7));
  aie::vector<bfloat16, 64> r = bits.cast_to<bfloat16>();
  return aie::select(r, aie::zeros<bfloat16, 64>(), aie::lt(n, lim));
}

// hdr: int32[0]=pos, [1]=n; float M rows for this kv head at byte offset 128 + row_id*256.
__attribute__((noinline)) static void AttnBegin(const uint8_t *hdr, int row_id) {
  // Rounding mode persists across dispatches: pin it so results never depend on the previous program.
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const int32_t *ih = (const int32_t *)hdr;
  a_pos = ih[0];
  a_n = ih[1];
  a_qhi = ih[2];
  a_win = ih[3];
  a_feed = 0;
  a_chunk_idx = 0;
  const int32_t *tab = ih + 4;  // ring slot->absolute map, 24 entries (unused when a_win==0)
  for (int s = 0; s < 24; s++) a_tab[s] = tab[s];
  const float *mrow = (const float *)(hdr + 128 + row_id * 256);
  for (int i = 0; i < 64; i++) {
    const int t = i & 15;
    a_m[i] = mrow[i];
    a_pp[i] = t < a_n ? a_pos + t : -1;
    a_hi[i] = t < a_n ? ((a_pos + t) & ~7) - 1 : -1;
    // window floor: first key this row may attend to = max(0, P - window + 1)
    a_lo[i] = (a_win && t < a_n) ? (a_pos + t - a_win + 1 > 0 ? a_pos + t - a_win + 1 : 0) : 0;
  }
  const aie::vector<float, 64> z = aie::zeros<float, 64>();
  for (int i = 0; i < kAttnRowBlocks * kAttnDimBlocks; i++) aie::store_v(a_o + i * 64, z);
  for (int i = 0; i < kAttnRowBlocks; i++) aie::store_v(a_l + i * 64, z);
}

__attribute__((noinline)) static void AttnQ(const uint8_t *pA, int idx, int row_id) {
  // element idx (0..31) of the Q stream; this core keeps elements row_id*8 .. row_id*8+7
  if (idx / 8 != row_id) return;
  uint8_t *dst = (uint8_t *)a_q + (idx % 8) * 1152;
  CopyBytes(dst, pA, 1152);
}

__attribute__((noinline)) static void AttnChunk(int tokbase) {
  const bfp16ebs8 *kv = (const bfp16ebs8 *)a_chunk;
  const aie::vector<int32, 64> col = LaneCol();
  const aie::vector<bfloat16, 64> ones = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::vector<bfloat16, 64> zero = aie::zeros<bfloat16, 64>();
  for (int rb = 0; rb < kAttnRowBlocks; rb++) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> qs(a_q);
    qs.seek(rb * 16);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> k0(kv);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> k1(kv);
    k0.seek(0);
    k1.seek(16);
    aie::accum<accfloat, 64> s0 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> s1 = aie::zeros<accfloat, 64>();
    for (int ks = 0; ks < 16; ks++) {
      aie::block_vector<bfp16ebs8, 64> a = qs.pop();
      aie::block_vector<bfp16ebs8, 64> b0 = k0.pop();
      aie::block_vector<bfp16ebs8, 64> b1 = k1.pop();
      s0 = mac_8x8_8x8T(a, b0, s0);
      s1 = mac_8x8_8x8T(a, b1, s1);
    }
    const aie::vector<float, 64> mv = RepEach8<float>(aie::load_v<8>(a_m + rb * 8));
    aie::vector<bfloat16, 64> p0 = Exp2Wide(aie::sub(s0.template to_vector<float>(), mv));
    aie::vector<bfloat16, 64> p1 = Exp2Wide(aie::sub(s1.template to_vector<float>(), mv));
    const aie::vector<int32, 64> rel =
        aie::sub(RepEach8<int32>(aie::load_v<8>(a_hi + rb * 8)), aie::broadcast<int32, 64>(tokbase));
    p0 = aie::select(zero, p0, aie::ge(rel, col));
    p1 = aie::select(zero, p1, aie::ge(rel, aie::add(col, aie::broadcast<int32, 64>(8))));
    // window floor: drop keys strictly before this row's window
    if (a_win) {
      const aie::vector<int32, 64> flo =
          aie::sub(RepEach8<int32>(aie::load_v<8>(a_lo + rb * 8)), aie::broadcast<int32, 64>(tokbase));
      p0 = aie::select(zero, p0, aie::ge(col, flo));
      p1 = aie::select(zero, p1, aie::ge(aie::add(col, aie::broadcast<int32, 64>(8)), flo));
    }
    aie::accum<accfloat, 64> la(aie::load_v<64>(a_l + rb * 64));
    la = aie::mac(la, p0, ones);
    la = aie::mac(la, p1, ones);
    aie::store_v(a_l + rb * 64, la.template to_vector<float>());
    aie::accum<accfloat, 64> pa0(p0), pa1(p1);
    aie::block_vector<bfp16ebs8, 64> pb0 = pa0.template to_vector<bfp16ebs8>();
    aie::block_vector<bfp16ebs8, 64> pb1 = pa1.template to_vector<bfp16ebs8>();
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> v0(kv);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> v1(kv);
    v0.seek(32);
    v1.seek(48);
    for (int db = 0; db < kAttnDimBlocks; db++) {
      aie::block_vector<bfp16ebs8, 64> vb0 = v0.pop();
      aie::block_vector<bfp16ebs8, 64> vb1 = v1.pop();
      float *op = a_o + (rb * kAttnDimBlocks + db) * 64;
      aie::accum<accfloat, 64> o(aie::load_v<64>(op));
      o = mac_8x8_8x8T(pb0, vb0, o);
      o = mac_8x8_8x8T(pb1, vb1, o);
      aie::store_v(op, o.template to_vector<float>());
    }
  }
}

// Feed one 1536 B element of the KV stream; every third element completes a 16-token chunk.
// chunk q = local * 8 + col covers tokens q*16 .. q*16+15.
__attribute__((noinline)) static void AttnFeed(const uint8_t *pB, int col) {
  uint8_t *dst = a_chunk + a_feed * 1536;
  CopyBytes(dst, pB, 1536);
  if (++a_feed < 3) return;
  a_feed = 0;
  const int s = a_chunk_idx * 8 + col;  // slot chunk this column just filled
  a_chunk_idx++;
  if (a_win) {
    // Ring cache: physical slot s holds the most recent chunk with q % 24 == s % 24.
    // a_tab maps slot -> qhi - absolute chunk; stale slots (d < 0) are skipped.
    const int d = a_tab[s];
    if (d >= 0) AttnChunk((a_qhi - d) * 16);
    return;
  }
  const int tokbase = s * 16;
  if (tokbase <= a_pos + a_n - 1) AttnChunk(tokbase);
}

// Tail path: the 8-token blocks that contain the pass rows use exact bf16 V so a row's result never depends on
// tokens appended after it. a_chunk holds [K block: 16 ks x 8 groups][V: 16 db x 8 tok x 8 dims bf16].
__attribute__((noinline)) static void AttnTail(int tb) {
  const int tokbase = (a_pos & ~7) + tb * 8;
  if (tokbase > a_pos + a_n - 1) return;
  const bfp16ebs8 *kt = (const bfp16ebs8 *)a_chunk;
  const bfloat16 *vt = (const bfloat16 *)(a_chunk + 1152);
  const aie::vector<int32, 64> col = LaneCol();
  const aie::vector<bfloat16, 64> ones = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::vector<bfloat16, 64> zero = aie::zeros<bfloat16, 64>();
  for (int rb = 0; rb < kAttnRowBlocks; rb++) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> qs(a_q);
    qs.seek(rb * 16);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ks(kt);
    ks.seek(0);
    aie::accum<accfloat, 64> s = aie::zeros<accfloat, 64>();
    for (int i = 0; i < 16; i++) {
      aie::block_vector<bfp16ebs8, 64> a = qs.pop();
      aie::block_vector<bfp16ebs8, 64> b = ks.pop();
      s = mac_8x8_8x8T(a, b, s);
    }
    const aie::vector<float, 64> mv = RepEach8<float>(aie::load_v<8>(a_m + rb * 8));
    aie::vector<bfloat16, 64> p = Exp2Wide(aie::sub(s.template to_vector<float>(), mv));
    const aie::vector<int32, 64> rel_hi =
        aie::sub(RepEach8<int32>(aie::load_v<8>(a_hi + rb * 8)), aie::broadcast<int32, 64>(tokbase));
    const aie::vector<int32, 64> rel_pp =
        aie::sub(RepEach8<int32>(aie::load_v<8>(a_pp + rb * 8)), aie::broadcast<int32, 64>(tokbase));
    p = aie::select(zero, p, aie::lt(rel_hi, col));
    p = aie::select(zero, p, aie::ge(rel_pp, col));
    if (a_win) {
      const aie::vector<int32, 64> flo =
          aie::sub(RepEach8<int32>(aie::load_v<8>(a_lo + rb * 8)), aie::broadcast<int32, 64>(tokbase));
      p = aie::select(zero, p, aie::ge(col, flo));
    }
    aie::accum<accfloat, 64> la(aie::load_v<64>(a_l + rb * 64));
    la = aie::mac(la, p, ones);
    aie::store_v(a_l + rb * 64, la.template to_vector<float>());
    for (int db = 0; db < kAttnDimBlocks; db++) {
      float *op = a_o + (rb * kAttnDimBlocks + db) * 64;
      aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat> mm;
      mm.mul(p, aie::load_v<64>(vt + db * 64));
      aie::vector<float, 64> d = mm.template to_vector<float>();
      aie::store_v(op, aie::add(aie::load_v<64>(op), d));
    }
  }
}

// A-stream element idx (0..35) of the tail: block tb = idx/12, kv head = (idx%12)/3, part = idx%3.
// Only column 0 computes the tail (its partial is added to the merged result).
__attribute__((noinline)) static void AttnTailFeed(const uint8_t *pA, int idx, int row_id, int col_id) {
  const int tb = idx / 12, r = (idx % 12) / 3, e = idx % 3;
  if (r != row_id || col_id != 0) return;
  uint8_t *dst = a_chunk + e * 1152;
  CopyBytes(dst, pA, 1152);
  if (e == 2) AttnTail(tb);
}

// Output element idx: 0..31 = a_o (32 KB), 32..33 = a_l (2 KB).
__attribute__((noinline)) static void AttnOut(uint8_t *pC, int idx) {
  const uint8_t *src = idx < 32 ? (const uint8_t *)a_o + idx * 1024 : (const uint8_t *)a_l + (idx - 32) * 1024;
  CopyBytes(pC, src, 1024);
}

#endif
