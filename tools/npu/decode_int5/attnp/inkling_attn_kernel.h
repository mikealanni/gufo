// Inkling prefill attention kernel: one core owns 16 query tokens x 3 q heads (48 rows)
// of one kv head and streams the causal prefix in 16-token chunks. Bias scores (bias . r
// in log2 domain, minus sink) are supplied per chunk by the host through the A fifo.
// Static softmax bound: p = exp2(s - M_row), M_row supplied by the host, so partials add.
//
// Structural constants:
//   * head_dim = 128 (16 dim-blocks of 8),
//   * 3 q heads per kv head, 16 tokens per pass per column (48 rows = 6 row-blocks),
//   * 8 columns x 4 rows = 32 cores; 4 kv heads per pass (two passes for 8),
//   * 16-token chunks (3 stream elements: 2 x K + 1 x V^T) and 8 columns per chunk row,
//   * per chunk 2 extra A-fifo elements hold fp32 bias scores in accumulator lane order.
#ifndef GUFO_INKLING_ATTN_KERNEL_H_
#define GUFO_INKLING_ATTN_KERNEL_H_

__attribute__((noinline)) static void CopyBytes(uint8_t *dst, const uint8_t *src, int n) {
  for (int i = 0; i < n; i += 64) aie::store_v(dst + i, aie::load_v<64>(src + i));
}

constexpr int kAttnRowBlocks = 6;   // 48 q rows = 6 blocks of 8
constexpr int kAttnDimBlocks = 16;  // 128 dims
constexpr int kAttnHeads = 3;
constexpr int kAttnTokens = 16;

alignas(128) bfp16ebs8 a_q[8 * 16 * 8];  // [8 rb][ks] blocks of 8 groups; 6 used

static float *const a_m = (float *)(arena + 0);        // 48 rows: bound M_row
static float *const a_il = (float *)(arena + 256);     // 48 rows: exp2(-M_row) for sink entry
static int32_t *const a_pp = (int32_t *)(arena + 512); // 48 rows: last token visible
static int32_t *const a_hi = (int32_t *)(arena + 768); // 48 rows (causal bound)
static int32_t *const a_lo = (int32_t *)(arena + 1024);// 48 rows: window floor
static float *const a_l = (float *)(arena + 1280);     // 6*64 floats
static float *const a_bias = (float *)(arena + 2816);  // 6*2*64 floats = 3072 B
static float *const a_o = (float *)(arena + 5888);     // 6*16*64 floats = 24576 B
static uint8_t *const a_chunk = (uint8_t *)(arena + 30464); // 4608 B
static bfloat16 *const a_r = (bfloat16 *)(arena + 35072);     // 48 rows x 16 basis
static bfloat16 *const a_slice = (bfloat16 *)(arena + 36608); // 16 basis x 32 shifts
// total 37632 < 40448

static int a_pos = 0, a_n = 0, a_chunk_idx = 0;
static int a_qhi = 0, a_win = 0, a_key0 = 0;

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

static inline float Exp2Accurate(float x) {
  if (x < -100.0f) return 0.0f;
  int n = static_cast<int>(x);
  if (static_cast<float>(n) > x) --n;
  const float f = x - static_cast<float>(n);
  const float c1 = 0.6931471805599453f;
  const float c2 = 0.2402265069591007f;
  const float c3 = 0.05550410866482158f;
  const float c4 = 0.009618129107628477f;
  const float c5 = 0.0013333558146428443f;
  const float p = 1.0f + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * c5))));
  union { float f; uint32_t u; } s;
  s.u = static_cast<uint32_t>(n + 127) << 23;
  return p * s.f;
}

__attribute__((noinline)) static aie::vector<bfloat16, 64> Exp2Wide(const aie::vector<float, 64> &a) {
  const aie::vector<float, 64> x = aie::max(a, aie::broadcast<float, 64>(-100.0f));
  const aie::vector<int32, 64> n = aie::to_fixed<int32>(x, 0);
  const aie::vector<float, 64> f = aie::sub(x, aie::to_float(n, 0));
  const aie::vector<bfloat16, 64> fb = aie::accum<accfloat, 64>(f).to_vector<bfloat16>();
  aie::vector<bfloat16, 64> t = aie::broadcast<bfloat16, 64>((bfloat16)0.0013333558146428443f);
  const float cs[5] = {0.009618129107628477f, 0.05550410866482158f, 0.2402265069591007f,
                       0.6931471805599453f, 1.0f};
  for (int i = 0; i < 5; i++) {
    aie::accum<accfloat, 64> acc(aie::broadcast<float, 64>(cs[i]));
    acc = aie::mac(acc, t, fb);
    t = acc.template to_vector<bfloat16>();
  }
  const aie::vector<int32, 64> eb = aie::upshift(aie::add(n, aie::broadcast<int32, 64>(127)), 23);
  const aie::vector<bfloat16, 64> sb =
      aie::accum<accfloat, 64>(eb.cast_to<float>()).to_vector<bfloat16>();
  return aie::mul(t, sb).template to_vector<bfloat16>();
}

// hdr: int32[0]=pos, [1]=n, [2]=qhi, [3]=win, [4]=key0; float[32..79]=M rows;
//      float[80..127]=exp2(-M) rows.
__attribute__((noinline)) static void AttnBegin(const uint8_t *hdr) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const int32_t *ih = (const int32_t *)hdr;
  a_pos = ih[0];
  a_n = ih[1];
  a_qhi = ih[2];
  a_win = ih[3];
  a_key0 = ih[4];
  a_chunk_idx = 0;
  const float *mrow = (const float *)(hdr + 128);
  const float *ilrow = (const float *)(hdr + 128 + 192);
  const float *srow = (const float *)(hdr + 128 + 384);
  for (int i = 0; i < 48; i++) {
    const int tok = i / kAttnHeads;
    a_m[i] = mrow[i] + srow[i];
    a_il[i] = ilrow[i];
    a_pp[i] = tok < a_n ? a_pos + tok : -1;
    a_hi[i] = a_pp[i];
    a_lo[i] = (a_win && a_pp[i] >= 0)
                  ? (a_pos + tok - a_win + 1 > 0 ? a_pos + tok - a_win + 1 : 0)
                  : 0;
  }
  const aie::vector<float, 64> z = aie::zeros<float, 64>();
  for (int i = 0; i < kAttnRowBlocks * kAttnDimBlocks; i++) aie::store_v(a_o + i * 64, z);
  for (int rb = 0; rb < kAttnRowBlocks; rb++) {
    for (int k = 0; k < 64; k++) a_l[rb * 64 + k] = (k & 7) == 0 ? a_il[rb * 8 + k / 8] : 0.0f;
  }
}

__attribute__((noinline)) static void AttnQ(const uint8_t *pB, int idx) {
  uint8_t *dst = (uint8_t *)a_q + idx * 1536;
  CopyBytes(dst, pB, 1536);
}

__attribute__((noinline)) static void AttnR(const uint8_t *pB) { CopyBytes((uint8_t *)a_r, pB, 1536); }

// Bias scores of one 16-key chunk, from the basis rows (a_r) and the shifted table slice:
// slice[z][j] = log2e * Bias[dmax - j][z], so token tok sees keys 0..15 at j = 15 - tok + k.
__attribute__((noinline)) static void AttnBias(const uint8_t *pB, int idx) {
  CopyBytes((uint8_t *)a_slice, pB, 1024);
  for (int tok = 0; tok < kAttnTokens; tok++) {
    aie::accum<accfloat, 16> acc0 = aie::zeros<accfloat, 16>();
    aie::accum<accfloat, 16> acc1 = aie::zeros<accfloat, 16>();
    aie::accum<accfloat, 16> acc2 = aie::zeros<accfloat, 16>();
    for (int z = 0; z < 16; z++) {
      const aie::vector<bfloat16, 32> v = aie::load_v<32>(a_slice + z * 32);
      const aie::vector<bfloat16, 16> sh = aie::shuffle_down(v, 15 - tok).template extract<16>(0);
      acc0 = aie::mac(acc0, sh, a_r[(tok * 3 + 0) * 16 + z]);
      acc1 = aie::mac(acc1, sh, a_r[(tok * 3 + 1) * 16 + z]);
      acc2 = aie::mac(acc2, sh, a_r[(tok * 3 + 2) * 16 + z]);
    }
    const aie::vector<float, 16> o[3] = {acc0.template to_vector<float>(),
                                         acc1.template to_vector<float>(),
                                         acc2.template to_vector<float>()};
    for (int h = 0; h < 3; h++) {
      const int i = tok * 3 + h;
      float *dst = a_bias + (i / 8) * 128 + (i % 8) * 8;
      aie::store_v(dst, o[h].template extract<8>(0));
      aie::store_v(dst + 64, o[h].template extract<8>(1));
    }
  }
}

static const uint8_t *a_kvp = a_chunk;

__attribute__((always_inline)) static inline void ChunkP(
    int rb, int tokbase, const aie::accum<accfloat, 64> &s0,
    const aie::accum<accfloat, 64> &s1,
    aie::block_vector<bfp16ebs8, 64> &pb0,
    aie::block_vector<bfp16ebs8, 64> &pb1) {
  const aie::vector<int32, 64> col = LaneCol();
  const aie::vector<bfloat16, 64> ones = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  const aie::vector<bfloat16, 64> zero = aie::zeros<bfloat16, 64>();
  const aie::vector<float, 64> mv = RepEach8<float>(aie::load_v<8>(a_m + rb * 8));
  aie::vector<bfloat16, 64> p0 = Exp2Wide(aie::sub(s0.template to_vector<float>(), mv));
  aie::vector<bfloat16, 64> p1 = Exp2Wide(aie::sub(s1.template to_vector<float>(), mv));
  if (a_win || a_qhi >= 0) {
    const aie::vector<int32, 64> rel =
        aie::sub(RepEach8<int32>(aie::load_v<8>(a_hi + rb * 8)), aie::broadcast<int32, 64>(tokbase));
    p0 = aie::select(zero, p0, aie::ge(rel, col));
    p1 = aie::select(zero, p1, aie::ge(rel, aie::add(col, aie::broadcast<int32, 64>(8))));
  }
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
  pb0 = pa0.template to_vector<bfp16ebs8>();
  pb1 = pa1.template to_vector<bfp16ebs8>();
}

__attribute__((noinline)) static void AttnChunk(int tokbase) {
  const bfp16ebs8 *kv = (const bfp16ebs8 *)a_kvp;
  for (int rb = 0; rb < kAttnRowBlocks; rb += 2) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> qa(a_q);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> qb(a_q);
    qa.seek(rb * 16);
    qb.seek((rb + 1) * 16);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> k0(kv), k1(kv);
    k0.seek(0);
    k1.seek(16);
    aie::accum<accfloat, 64> s0a = aie::zeros<accfloat, 64>(), s1a = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> s0b = aie::zeros<accfloat, 64>(), s1b = aie::zeros<accfloat, 64>();
    for (int ks = 0; ks < 16; ks++) {
      aie::block_vector<bfp16ebs8, 64> b0 = k0.pop(), b1 = k1.pop();
      aie::block_vector<bfp16ebs8, 64> xa = qa.pop(), xb = qb.pop();
      s0a = mac_8x8_8x8T(xa, b0, s0a);
      s1a = mac_8x8_8x8T(xa, b1, s1a);
      s0b = mac_8x8_8x8T(xb, b0, s0b);
      s1b = mac_8x8_8x8T(xb, b1, s1b);
    }
    s0a = aie::add(s0a, aie::load_v<64>(a_bias + rb * 128));
    s1a = aie::add(s1a, aie::load_v<64>(a_bias + rb * 128 + 64));
    s0b = aie::add(s0b, aie::load_v<64>(a_bias + (rb + 1) * 128));
    s1b = aie::add(s1b, aie::load_v<64>(a_bias + (rb + 1) * 128 + 64));
    aie::block_vector<bfp16ebs8, 64> pa0, pa1, pc0, pc1;
    ChunkP(rb, tokbase, s0a, s1a, pa0, pa1);
    ChunkP(rb + 1, tokbase, s0b, s1b, pc0, pc1);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> v0(kv), v1(kv);
    v0.seek(32);
    v1.seek(48);
    for (int db = 0; db < kAttnDimBlocks; db++) {
      aie::block_vector<bfp16ebs8, 64> vb0 = v0.pop(), vb1 = v1.pop();
      float *opa = a_o + (rb * kAttnDimBlocks + db) * 64;
      float *opb = a_o + ((rb + 1) * kAttnDimBlocks + db) * 64;
      aie::accum<accfloat, 64> oa(aie::load_v<64>(opa));
      aie::accum<accfloat, 64> ob(aie::load_v<64>(opb));
      oa = mac_8x8_8x8T(pa0, vb0, oa);
      ob = mac_8x8_8x8T(pc0, vb0, ob);
      oa = mac_8x8_8x8T(pa1, vb1, oa);
      ob = mac_8x8_8x8T(pc1, vb1, ob);
      aie::store_v(opa, oa.template to_vector<float>());
      aie::store_v(opb, ob.template to_vector<float>());
    }
  }
}

__attribute__((noinline)) static void AttnFeedP2(const uint8_t *pB) {
  const int slot = a_chunk_idx++;
  const int tokbase = a_key0 + slot * 16;
  if (tokbase > a_pos + a_n - 1) return;
  a_kvp = pB;
  AttnChunk(tokbase);
  a_kvp = a_chunk;
}

__attribute__((noinline)) static void AttnOut(uint8_t *pC, int idx) {
  const uint8_t *src = idx < 24 ? (const uint8_t *)a_o + idx * 1024
                                : (const uint8_t *)a_l + (idx - 24) * 1024;
  CopyBytes(pC, src, 1024);
}

#endif
