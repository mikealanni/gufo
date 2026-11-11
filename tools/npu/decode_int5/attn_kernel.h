// Decode attention for one kv head group (4 q heads x 16 rows) over a slice of the context.
// Static softmax bound: p = exp2(s - M_row), M_row supplied by the host, so partials add.
#ifndef GUFO_ATTN_KERNEL_H_
#define GUFO_ATTN_KERNEL_H_

constexpr int kAttnRowBlocks = 8;   // 64 q rows = 8 blocks of 8
constexpr int kAttnDimBlocks = 16;  // 128 dims

alignas(128) bfp16ebs8 a_q[kAttnRowBlocks * 16 * 8];       // [rb][ks] blocks of 8 groups
alignas(128) uint8_t a_chunk[4608];                          // K (256 groups) then V^T (256 groups)
alignas(128) float a_o[kAttnRowBlocks * kAttnDimBlocks * 64];
static float *const a_m = (float *)(arena + 0);
static int32_t *const a_hi = (int32_t *)(arena + 2048);
static float *const a_l = (float *)(arena + 4096);
static int a_pos = 0, a_n = 0, a_feed = 0, a_chunk_idx = 0;

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
  const int32_t *ih = (const int32_t *)hdr;
  a_pos = ih[0];
  a_n = ih[1];
  a_feed = 0;
  a_chunk_idx = 0;
  const float *mrow = (const float *)(hdr + 128 + row_id * 256);
  for (int rb = 0; rb < kAttnRowBlocks; rb++)
    for (int rib = 0; rib < 8; rib++) {
      const int t = (rb & 1) * 8 + rib;
      const int32_t hi = t < a_n ? a_pos + t : -1;
      for (int c = 0; c < 8; c++) {
        a_m[rb * 64 + rib * 8 + c] = mrow[rb * 8 + rib];
        a_hi[rb * 64 + rib * 8 + c] = hi;
      }
    }
  const aie::vector<float, 64> z = aie::zeros<float, 64>();
  for (int i = 0; i < kAttnRowBlocks * kAttnDimBlocks; i++) aie::store_v(a_o + i * 64, z);
  for (int i = 0; i < kAttnRowBlocks; i++) aie::store_v(a_l + i * 64, z);
}

__attribute__((noinline)) static void AttnQ(const uint8_t *pA, int idx, int row_id) {
  // element idx (0..31) of the Q stream; this core keeps elements row_id*8 .. row_id*8+7
  if (idx / 8 != row_id) return;
  uint8_t *dst = (uint8_t *)a_q + (idx % 8) * 1152;
  for (int i = 0; i < 1152; i += 64) aie::store_v(dst + i, aie::load_v<64>(pA + i));
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
    const aie::vector<float, 64> mv = aie::load_v<64>(a_m + rb * 64);
    aie::vector<bfloat16, 64> p0 = Exp2Wide(aie::sub(s0.template to_vector<float>(), mv));
    aie::vector<bfloat16, 64> p1 = Exp2Wide(aie::sub(s1.template to_vector<float>(), mv));
    const aie::vector<int32, 64> rel =
        aie::sub(aie::load_v<64>(a_hi + rb * 64), aie::broadcast<int32, 64>(tokbase));
    p0 = aie::select(zero, p0, aie::ge(rel, col));
    p1 = aie::select(zero, p1, aie::ge(rel, aie::add(col, aie::broadcast<int32, 64>(8))));
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
  for (int i = 0; i < 1536; i += 64) aie::store_v(dst + i, aie::load_v<64>(pB + i));
  if (++a_feed < 3) return;
  a_feed = 0;
  const int tokbase = (a_chunk_idx * 8 + col) * 16;
  a_chunk_idx++;
  if (tokbase <= a_pos + a_n - 1) AttnChunk(tokbase);
}

// Output element idx: 0..31 = a_o (32 KB), 32..33 = a_l (2 KB).
__attribute__((noinline)) static void AttnOut(uint8_t *pC, int idx) {
  const uint8_t *src = idx < 32 ? (const uint8_t *)a_o + idx * 1024 : (const uint8_t *)a_l + (idx - 32) * 1024;
  for (int i = 0; i < 1024; i += 64) aie::store_v(pC + i, aie::load_v<64>(src + i));
}

#endif
