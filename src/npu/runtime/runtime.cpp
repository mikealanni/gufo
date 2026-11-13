#include <immintrin.h>
#include "src/npu/runtime/runtime.hpp"
#include "src/npu/runtime/atb_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>
#include <stdexcept>

namespace gufo::npu::runtime {


namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
}  // namespace

void PackMatInt5(const float* W, std::uint32_t N, std::uint32_t K,
                 std::uint8_t* dst, std::uint32_t extra_tiles, const std::uint32_t* tile_perm) {
  const std::uint32_t strips = N / kStrip, tk = K / kTileK;
  std::vector<std::int8_t> q(std::size_t{N} * K);
  std::vector<std::uint16_t> sc(std::size_t{N} * (K / 32));
#pragma omp parallel for
  for (std::int64_t n = 0; n < N; ++n)
    for (std::uint32_t kg = 0; kg < K / 32; ++kg) {
      const float* row = W + std::size_t(n) * K + kg * 32;
      float amax = 0;
      for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(row[i]));
      std::uint16_t sb = Bf16Bits(amax / 15.0f);
      float s = Bf16Float(sb);
      if (s == 0) {
        sb = Bf16Bits(1.0f);
        s = 1.0f;
      }
      sc[std::size_t(n) * (K / 32) + kg] = sb;
      for (int i = 0; i < 32; ++i)
        q[std::size_t(n) * K + kg * 32 + i] = static_cast<std::int8_t>(
            std::fmin(15.f, std::fmax(-15.f, std::nearbyint(row[i] / s))));
    }
#pragma omp parallel for collapse(2)
  for (std::int64_t s2 = 0; s2 < strips; ++s2)
    for (std::int64_t t = 0; t < tk; ++t)
      for (int r = 0; r < 4; ++r) {
        std::uint8_t* base =
            dst + ((std::size_t(s2) * (tk + extra_tiles) + t) * 4 + r) * kQuarterBytes;
        auto* words = reinterpret_cast<std::uint16_t*>(base);
        auto* scl = reinterpret_cast<std::uint16_t*>(base + kWordBytes);
        for (int p = 0; p < 32; ++p) {
          const int sb = p >> 4, sr = (p >> 1) & 7, blk = p & 1;
          for (int lane = 0; lane < 64; ++lane) {
            const int cib = lane >> 3, kk = lane & 7;
            const std::size_t nidx = std::size_t(s2) * 128 + r * 32 + sb * 16 +
                                     blk * 8 + cib;
            const int x = q[nidx * K + (tile_perm ? tile_perm[t] : std::uint32_t(t)) * 64 + sr * 8 + kk] + 16;
            words[(p / 3) * 64 + lane] |=
                static_cast<std::uint16_t>(x << (5 * (p % 3)));
          }
        }
        for (int sb = 0; sb < 2; ++sb)
          for (int blk = 0; blk < 2; ++blk)
            for (int qq = 0; qq < 2; ++qq)
              for (int cib = 0; cib < 8; ++cib) {
                const std::size_t nidx = std::size_t(s2) * 128 + r * 32 +
                                         sb * 16 + blk * 8 + cib;
                scl[((sb * 2 + blk) * 2 + qq) * 8 + cib] =
                    sc[nidx * (K / 32) + (tile_perm ? tile_perm[t] : std::uint32_t(t)) * 2 + qq];
              }
      }
}

bool NpuRuntime::Init(const std::string& xclbin_path, std::string* error) {
  try {
    dev_ = xrt::device(0u);
    const xrt::xclbin xb{xclbin_path};
    dev_.register_xclbin(xb);
    ctx_ = xrt::hw_context(dev_, xb.get_uuid());
    kern_ = xrt::kernel(ctx_, xb.get_kernels().at(0).get_name());
    ready_ = true;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}

Program NpuRuntime::MakeProgram(std::uint32_t K, std::uint32_t N,
                                const std::string& insts_path,
                                std::size_t c_override, std::string* error,
                                std::uint32_t a_extra_k, std::uint32_t mrows,
                                int a_group, int c_group) {
  Program p;
  p.K = K;
  p.N = N;
  p.mrows = mrows;
  std::ifstream f(insts_path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("missing " + insts_path);
  const std::size_t sz = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<char> buf(sz);
  f.read(buf.data(), static_cast<std::streamsize>(sz));
  p.words = sz / 4;
  p.insts = xrt::bo(dev_, sz, XCL_BO_FLAGS_CACHEABLE, kern_.group_id(1));
  std::memcpy(p.insts.map<void*>(), buf.data(), sz);
  p.insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const std::size_t ab = std::size_t{16 * mrows} * (K + a_extra_k) / 8 * 9;
  const std::size_t cb = c_override ? c_override : std::size_t{N / kStrip} * mrows * kStripBytes;
  p.a = xrt::bo(dev_, ab, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(a_group));
  p.c = xrt::bo(dev_, cb, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(c_group));
  p.ah = p.a.map<std::uint8_t*>();
  p.ch = p.c.map<std::uint8_t*>();
  std::memset(p.ah, 0, ab);
  std::memset(p.ch, 0, cb);
  p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);  // flush dirty zero lines: evicted later they overwrite device writes
  p.c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  (void)error;
  return p;
}

WeightBuffer NpuRuntime::PackWeight(const float* W, std::uint32_t N,
                                    std::uint32_t K) {
  WeightBuffer wb;
  wb.bytes = std::size_t{N / kStrip} * (K / kTileK) * 4 * kQuarterBytes;
  wb.bo = xrt::bo(dev_, wb.bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  wb.host = wb.bo.map<std::uint8_t*>();
  std::memset(wb.host, 0, wb.bytes);
  PackMatInt5(W, N, K, wb.host);
  wb.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return wb;
}

void NpuRuntime::AttachTo(xrt::run& run, WeightBuffer& wb, Program& prog) {
  run.set_arg(0, 3u);
  run.set_arg(1, prog.insts);
  run.set_arg(2, static_cast<unsigned>(prog.words));
  if (prog.four_a) {  // cfg29 tap program: (A1, A2, B, C)
    run.set_arg(3, prog.a);
    run.set_arg(4, prog.a2);
    run.set_arg(5, wb.bo);
    run.set_arg(6, prog.c);
    return;
  }
  run.set_arg(3, prog.a);
  run.set_arg(4, wb.bo);
  run.set_arg(5, prog.c);
}

// Run objects cannot be re-arg'd after a completed dispatch on this XRT/xdna
// stack (set_arg silently keeps the old pointers -> stale-weight outputs, WP10).
// Create a fresh run per dispatch; destruction after wait() is safe (m16 path).
xrt::run& NpuRuntime::SlotRun(WeightBuffer& wb) {
  wb.run = xrt::run(kern_);
  return wb.run;
}

void NpuRuntime::Attach(WeightBuffer& wb, Program& prog) { AttachTo(wb.run, wb, prog); }

WeightBuffer& NpuRuntime::WeightSlot(WeightBuffer& wb, Program& prog) {
  if (!prog.wpin) {
    prog.wpin = &wb;  // first binding sticks; it is also the dispatch buffer.
  } else if (prog.wpin != &wb) {
    WeightBuffer& p0 = *prog.wpin;
    if (p0.bytes != wb.bytes)
      throw std::runtime_error("weight slot size mismatch for pinned program");
    std::memcpy(p0.host, wb.host, wb.bytes);
    p0.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);  // HOST_ONLY: no-op
  }
  WeightBuffer& w = *prog.wpin;
  AttachTo(SlotRun(w), w, prog);
  return w;
}

// m64 FFN (WP10 prefill): gu program (swiglu flush) -> swiglu C pieces byte-shuffle into
// the down program's A staging (their layout matches: piece (s,t r) == A tap (t s) r-th),
// then the plain 7680x2048 down gemm. No float round-trip, no requantization.
void NpuRuntime::Ffn64(WeightBuffer& wb_gu, Program& p_gu,
                       WeightBuffer& wb_dn, Program& p_dn,
                       const float* x, std::uint32_t n, float* y, Timings* t) {
  const std::uint32_t tk = p_gu.K / kTileK;  // 32 k-tiles; gu C strips = K2/k = 120
  auto t0 = Clock::now();
  WeightBuffer& w_gu = WeightSlot(wb_gu, p_gu);
  EncodeA(x, n, p_gu.K, p_gu);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  p_gu.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  w_gu.run.start();
  {
    const auto spin0 = Clock::now();
    for (;;) {
      const auto st = w_gu.run.state();
      if (st == ERT_CMD_STATE_COMPLETED) break;
      if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
          st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) break;
      if (Ms(spin0) > 200.0) break;
    }
    w_gu.run.wait();
    if (!p_gu.warmed) {  // WP10: a program's first C write is lost on this
      p_gu.warmed = true;  // XRT/xdna stack; redo the dispatch so C is valid.
      w_gu.run.start();
      w_gu.run.wait();
    }
  }
  if (t) t->wait += Ms(t0);
  p_gu.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  // gu C wire per (tap s, aie-row r) element: 4 swiglu row-blocks of 288 B at mb*288
  // (matches the fused m16 down-A DMA pattern [1,ktile,mrows,rows,288] x [0,SB,288,CB,1]).
  const std::uint8_t* cgu = p_gu.ch;
  for (std::uint32_t s = 0; s < 120; ++s)
    for (int r = 0; r < 4; ++r)
      for (std::uint32_t mb = 0; mb < p_dn.mrows; ++mb)
        std::memcpy(p_dn.ah + (std::size_t(s) * 4 + mb) * 1152 + std::size_t(r) * 288,
                    cgu + std::size_t(s) * p_gu.mrows * kStripBytes + std::size_t(mb) * 4 * kRowBytes +
                        std::size_t(r) * kRowBytes,
                    288);
  (void)tk;
  t0 = Clock::now();
  WeightBuffer& w_dn = WeightSlot(wb_dn, p_dn);
  p_dn.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  w_dn.run.start();
  {
    const auto spin0 = Clock::now();
    for (;;) {
      const auto st = w_dn.run.state();
      if (st == ERT_CMD_STATE_COMPLETED) break;
      if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
          st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) break;
      if (Ms(spin0) > 200.0) break;
    }
    w_dn.run.wait();
    if (!p_dn.warmed) {  // WP10: a program's first C write is lost on this
      p_dn.warmed = true;  // XRT/xdna stack; redo the dispatch so C is valid.
      w_dn.run.start();
      w_dn.run.wait();
    }
  }
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  p_dn.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  DecodeC(y, n, p_dn);
  if (t) t->dec += Ms(t0);
  if (t) t->calls += 2;
}

void NpuRuntime::EncodeA(const float* x, std::uint32_t n, std::uint32_t K,
                         Program& p) {
  const std::uint32_t tk = K / kTileK, mr = p.mrows;
#pragma omp parallel for
  for (std::int64_t t = 0; t < tk; ++t)
    for (std::uint32_t mb = 0; mb < mr; ++mb)
      for (int gi = 0; gi < 128; ++gi) {
        const int rib = gi & 7, block = (gi >> 3) & 1, sc = gi >> 4;
        const int row = int(mb * 16 + block * 8 + rib);
        std::uint8_t* o = p.ah + (std::size_t(t) * mr + mb) * 1152 + std::size_t(gi) * 9;
        if (row >= static_cast<int>(n)) {
          std::memset(o, 0, 9);
          continue;
        }
        const float* src = x + std::size_t(row) * K + t * 64 + sc * 8;
        float v[8];
        _mm256_storeu_ps(v, _mm256_cvtph_ps(_mm256_cvtps_ph(
                                _mm256_loadu_ps(src), _MM_FROUND_TO_NEAREST_INT)));
        EncGroup(v, o);
      }
}

void NpuRuntime::DecodeC(float* y, std::uint32_t n, Program& p,
                         const std::uint8_t* base, std::uint32_t N) {
  if (!base) {
    base = p.ch;
    N = p.N;
  }
  const std::uint32_t strips = N / kStrip;
#pragma omp parallel for
  for (std::int64_t s = 0; s < strips; ++s)
    for (int r = 0; r < 4; ++r)
      for (std::uint32_t mb = 0; mb < p.mrows; ++mb)
      for (int G = 0; G < 64; ++G) {
        const int sb = G >> 5, acc = (G >> 3) & 3, rib = G & 7;
        const int row = int(mb * 16 + (acc >> 1) * 8 + rib);
        if (row >= static_cast<int>(n))
          continue;
        const std::size_t col = std::size_t(s) * 128 + r * 32 + sb * 16 +
                                (acc & 1) * 8;
        DecGroup(base + std::size_t(s) * p.mrows * kStripBytes + std::size_t(mb) * 4 * kRowBytes +
                     std::size_t(r) * kRowBytes + std::size_t(G) * 9,
                 y + std::size_t(row) * N + col);
      }
}

void NpuRuntime::Gemm(Program& p, WeightBuffer& wb, const float* x,
                      std::uint32_t n, float* y, Timings* t) {
  WeightBuffer& w = WeightSlot(wb, p);
  auto t0 = Clock::now();
  EncodeA(x, n, p.K, p);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  w.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  {
    const auto spin0 = Clock::now();
    for (;;) {
      const auto st = w.run.state();
      if (st == ERT_CMD_STATE_COMPLETED)
        break;
      if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
          st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED)
        break;
    }
    w.run.wait();
    if (!p.warmed) {  // WP10: a program's first C write is lost on this
      p.warmed = true;  // XRT/xdna stack; redo the dispatch so C is valid.
      w.run.start();
      w.run.wait();
    }
  }
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  DecodeC(y, n, p);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

// --- AGENT-7 big-M prefill (ATB) path -----------------------------------
// The reference ATB helpers build a shuffled float copy, then encode it, then
// decode C back through another float copy. That costs ~1.5 s per 1024-token
// chunk on the host. These paths write/read the tile stream in place, fused
// with the bfp16 codec, so the host side is a pass over the data.
namespace {
constexpr std::uint32_t kAtbTileM = 128;  // L1 tile m
constexpr std::uint32_t kAtbTileN = 128;  // L1 tile n
constexpr std::uint32_t kAtbTileK = 64;   // L1 tile k
constexpr std::uint32_t kAtbTileBytes = kAtbTileM * kAtbTileK / 8 * 9;  // 9216
// One 8x8 sub-block: 8 bfp16ebs8 vectors of 9 bytes (one per row/column).
constexpr std::uint32_t kAtbBlkBytes = 8 * 9;  // 72

// v8bfp16ebs8 encoder for one 8-value block (AIE2p truncation semantics).
inline void EncodeBlock8(const float* x, std::uint8_t* out) {
  const __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
  const __m256i ex = _mm256_and_si256(_mm256_srli_epi32(u, 23), _mm256_set1_epi32(0xFF));
  __m256i m = _mm256_max_epu32(ex, _mm256_shuffle_epi32(ex, 0xB1));
  m = _mm256_max_epu32(m, _mm256_shuffle_epi32(m, 0x4E));
  m = _mm256_max_epu32(m, _mm256_permute2x128_si256(m, m, 1));
  out[0] = static_cast<std::uint8_t>(_mm256_cvtsi256_si32(m));
  __m256i mant = _mm256_and_si256(u, _mm256_set1_epi32(0x007FFFFF));
  const __m256i hasexp = _mm256_cmpgt_epi32(ex, _mm256_setzero_si256());
  mant = _mm256_or_si256(mant, _mm256_and_si256(hasexp, _mm256_set1_epi32(0x00800000)));
  const __m256i sgn = _mm256_srai_epi32(u, 31);
  mant = _mm256_sub_epi32(_mm256_xor_si256(mant, sgn), sgn);
  __m256i sv = _mm256_srai_epi32(_mm256_slli_epi32(_mm256_srli_epi32(mant, 17), 24), 24);
  const __m256i dsh = _mm256_sub_epi32(m, ex);
  sv = _mm256_blendv_epi8(_mm256_srav_epi32(sv, dsh), sgn, _mm256_cmpgt_epi32(dsh, _mm256_set1_epi32(31)));
  sv = _mm256_andnot_si256(_mm256_cmpeq_epi32(ex, _mm256_set1_epi32(255)), sv);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(out + 1), _mm256_cvtepi32_epi8(sv));
}

// Decode one 8-value block into dst[0..7].
inline float BlockScale(std::uint8_t e) {
  const float m = e >= 127 ? static_cast<float>(1u << (e - 127))
                           : 1.0f / static_cast<float>(1u << (127 - e));
  return m / 64.0f;
}
inline void DecodeBlock8(const std::uint8_t* in, float* dst) {
  const float mult = BlockScale(in[0]);
  const __m256i v = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + 1)));
  _mm256_storeu_ps(dst, _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(mult)));
}
}  // namespace

// Model-layout weights ([N rows][K cols], like PackWeight) -> ATB B stream.
// The kernel computes C = A @ B with B[k][n], so the stream is built as
// B[k][n] = W[n][k]. Tiles are the transposed 1x2 8x8 pattern over K x N.
WeightBuffer NpuRuntime::PackWeightAtb(const float* W, std::uint32_t N,
                                       std::uint32_t K, int b_group) {
  WeightBuffer wb;
  wb.bytes = std::size_t{K} * N / 8 * 9;
  wb.bo = xrt::bo(dev_, wb.bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(b_group));
  wb.host = wb.bo.map<std::uint8_t*>();
  const std::uint32_t l1_rows = K / kAtbTileK;
  const std::uint32_t l1_cols = N / kAtbTileN;
#pragma omp parallel for collapse(2)
  for (std::uint32_t l1_col = 0; l1_col < l1_cols; ++l1_col) {
    for (std::uint32_t l1_row = 0; l1_row < l1_rows; ++l1_row) {
      std::uint8_t* tile =
          wb.host + (std::size_t{l1_col} * l1_rows + l1_row) * kAtbTileBytes;
      for (std::uint32_t sbc_i = 0; sbc_i < 8; ++sbc_i) {
        for (std::uint32_t sbr = 0; sbr < 8; ++sbr) {
          for (std::uint32_t bis = 0; bis < 2; ++bis) {
            std::uint8_t* blk =
                tile + ((sbc_i * 8 + sbr) * 2 + bis) * kAtbBlkBytes;
            const std::uint32_t cbc = sbc_i * 2 + bis;
            const std::uint32_t row0 = (l1_row * kAtbTileK) + sbr * 8;  // k base
            const std::uint32_t col0 = (l1_col * kAtbTileN) + cbc * 8;  // n base
            for (std::uint32_t col_in = 0; col_in < 8; ++col_in) {
              float v[8];
              for (std::uint32_t row_in = 0; row_in < 8; ++row_in)
                v[row_in] = W[std::size_t(col0 + col_in) * K + (row0 + row_in)];
              EncodeBlock8(v, blk + col_in * 9);
            }
          }
        }
      }
    }
  }
  wb.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return wb;
}

WeightBuffer NpuRuntime::PackFfnGuAtb(const float* gate, const float* up,
                                      std::uint32_t na, std::uint32_t hidden,
                                      int b_group) {
  std::vector<float> buf(std::size_t{2} * na * hidden);
  for (std::uint32_t s = 0; s < 2 * na; ++s) {
    const std::uint32_t q = s >> 4, r = s & 15;
    const float* src = (r < 8 ? gate : up) + std::size_t{q * 8 + (r & 7)} * hidden;
    std::memcpy(buf.data() + std::size_t{s} * hidden, src, std::size_t{hidden} * 4);
  }
  return PackWeightAtb(buf.data(), 2 * na, hidden, b_group);
}

WeightBuffer NpuRuntime::PackFfnDnAtb(const float* Wd, std::uint32_t hidden,
                                      std::uint32_t INT, int b_group) {
  return PackWeightAtb(Wd, hidden, INT, b_group);
}

// M x K row-major activations -> ATB A stream (row-major 2x1 8x8 tiles).
void NpuRuntime::EncodeAAtbRaw(const float* x, std::uint32_t M, std::uint32_t K,
                               std::uint8_t* dst) const {
  const std::uint32_t l1_rows = M / kAtbTileM;
  const std::uint32_t l1_cols = K / kAtbTileK;
#pragma omp parallel for collapse(2)
  for (std::uint32_t l1_row = 0; l1_row < l1_rows; ++l1_row) {
    for (std::uint32_t l1_col = 0; l1_col < l1_cols; ++l1_col) {
      std::uint8_t* tile =
          dst + (std::size_t{l1_row} * l1_cols + l1_col) * kAtbTileBytes;
      for (std::uint32_t sbr = 0; sbr < 8; ++sbr) {
        for (std::uint32_t sbc = 0; sbc < 8; ++sbc) {
          for (std::uint32_t bis = 0; bis < 2; ++bis) {
            std::uint8_t* blk =
                tile + ((sbr * 8 + sbc) * 2 + bis) * kAtbBlkBytes;
            const std::uint32_t row0 = l1_row * kAtbTileM + sbr * 16 + bis * 8;
            const std::uint32_t col0 = l1_col * kAtbTileK + sbc * 8;
            for (std::uint32_t rib = 0; rib < 8; ++rib)
              EncodeBlock8(x + std::size_t(row0 + rib) * K + col0,
                           blk + rib * 9);
          }
        }
      }
    }
  }
}


void NpuRuntime::EncodeAAtb(const float* x, std::uint32_t M, std::uint32_t K,
                            Program& p) {
  atb_a_scratch_.resize(std::size_t{M} * K / 8 * 9);
  EncodeAAtbRaw(x, M, K, atb_a_scratch_.data());
  std::memcpy(p.ah, atb_a_scratch_.data(), atb_a_scratch_.size());
}

// ATB C stream -> M x N row-major floats.
// The kernel emits one 128x128 tile per AIE core. The 4x8 core array walks
// (m,n) tile passes of 4 m-tiles x 8 n-tiles, and the C outputs concatenate
// core-column-major within each pass (measured against a tagged-identity GEMM:
// tile (mt,nt) lands at stream slot
//   ((mt/4)*npass + nt/8)*32 + (nt%8)*4 + (mt%4),  npass = N/1024).
// Inside each tile the 8x8 blocks are 2x2-super-block ordered (row-pair
// outer, 8 super-columns), i.e. layout_inverse_C_2x2_8x8block with a
// 128x128 L1 block.
void NpuRuntime::DecodeCAtbRaw(const std::uint8_t* src, float* y, std::uint32_t M, std::uint32_t N) const {
  constexpr std::uint32_t kAtbTileC = kAtbTileM * kAtbTileN / 8 * 9;  // 18432
  const std::uint32_t mtiles = M / kAtbTileM;
  const std::uint32_t ntiles = N / kAtbTileN;
  const std::uint32_t npass = ntiles / 8;  // N / 1024
#pragma omp parallel for collapse(2)
  for (std::uint32_t mt = 0; mt < mtiles; ++mt) {
    for (std::uint32_t nt = 0; nt < ntiles; ++nt) {
      const std::size_t tile =
          (std::size_t{(mt / 4) * npass + nt / 8} * 32 + (nt % 8) * 4 +
           (mt % 4));
      const std::uint8_t* t = src + tile * kAtbTileC;
      for (std::uint32_t sbr = 0; sbr < 8; ++sbr) {
        for (std::uint32_t sbc = 0; sbc < 8; ++sbc) {
          for (std::uint32_t br = 0; br < 2; ++br) {
            for (std::uint32_t bc = 0; bc < 2; ++bc) {
              const std::uint8_t* blk =
                  t + (((sbr * 8 + sbc) * 4) + br * 2 + bc) * kAtbBlkBytes;
              const std::uint32_t row0 = mt * kAtbTileM + sbr * 16 + br * 8;
              const std::uint32_t col0 = nt * kAtbTileN + sbc * 16 + bc * 8;
              for (std::uint32_t rib = 0; rib < 8; ++rib) {
                float v[8];
                DecodeBlock8(blk + rib * 9, v);
                std::memcpy(y + std::size_t(row0 + rib) * N + col0, v,
                            sizeof(v));
              }
            }
          }
        }
      }
    }
  }
}


void NpuRuntime::DecodeCAtb(float* y, std::uint32_t M, std::uint32_t N,
                            Program& p) {
  atb_c_scratch_.resize(std::size_t{M} * N / 8 * 9);
  std::memcpy(atb_c_scratch_.data(), p.ch, atb_c_scratch_.size());
  DecodeCAtbRaw(atb_c_scratch_.data(), y, M, N);
}

void NpuRuntime::DecodeCRowAtb(const Program& p, std::uint32_t N, std::uint32_t row, float* out) const {
  constexpr std::uint32_t kAtbTileC = kAtbTileM * kAtbTileN / 8 * 9;
  const std::uint32_t mt = row / kAtbTileM, rr = row % kAtbTileM;
  const std::uint32_t sbr = rr / 16, br = (rr % 16) / 8, rib = rr % 8;
  const std::uint32_t ntiles = N / kAtbTileN, npass = ntiles / 8;
  for (std::uint32_t nt = 0; nt < ntiles; ++nt) {
    const std::size_t tile = std::size_t{(mt / 4) * npass + nt / 8} * 32 + (nt % 8) * 4 + (mt % 4);
    const std::uint8_t* t = p.ch + tile * kAtbTileC;
    for (std::uint32_t sbc = 0; sbc < 8; ++sbc)
      for (std::uint32_t bc = 0; bc < 2; ++bc) {
        const std::uint8_t* blk = t + (((sbr * 8 + sbc) * 4) + br * 2 + bc) * kAtbBlkBytes;
        DecodeBlock8(blk + rib * 9, out + std::size_t{nt} * kAtbTileN + sbc * 16 + bc * 8);
      }
  }
}

void NpuRuntime::DecodeAAtb(float* x, std::uint32_t M, std::uint32_t K, const Program& p) const {
  const std::uint32_t l1_rows = M / kAtbTileM, l1_cols = K / kAtbTileK;
  for (std::uint32_t l1_row = 0; l1_row < l1_rows; ++l1_row)
    for (std::uint32_t l1_col = 0; l1_col < l1_cols; ++l1_col) {
      const std::uint8_t* tile = p.ah + (std::size_t{l1_row} * l1_cols + l1_col) * kAtbTileBytes;
      for (std::uint32_t sbr = 0; sbr < 8; ++sbr)
        for (std::uint32_t sbc = 0; sbc < 8; ++sbc)
          for (std::uint32_t bis = 0; bis < 2; ++bis) {
            const std::uint8_t* blk = tile + ((sbr * 8 + sbc) * 2 + bis) * kAtbBlkBytes;
            const std::uint32_t row0 = l1_row * kAtbTileM + sbr * 16 + bis * 8;
            const std::uint32_t col0 = l1_col * kAtbTileK + sbc * 8;
            for (std::uint32_t rib = 0; rib < 8; ++rib) DecodeBlock8(blk + rib * 9, x + std::size_t(row0 + rib) * K + col0);
          }
    }
}

void NpuRuntime::GemmAtb(Program& p, WeightBuffer& wb, const float* x,
                         std::uint32_t M, float* y, Timings* t) {
  WeightBuffer& w = WeightSlot(wb, p);
  auto t0 = Clock::now();
  EncodeAAtb(x, M, p.K, p);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  w.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  // Plain driver wait: the decode path's state() spin costs ~3 ms per
  // dispatch here (each poll is a driver round-trip), which swamped the GEMM.
  w.run.wait();
  if (!p.warmed) {  // WP10: a program's first C write is lost on this stack.
    p.warmed = true;
    w.run.start();
    w.run.wait();
  }
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  DecodeCAtb(y, M, p.N, p);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

void NpuRuntime::Ffn(Program& p, WeightBuffer& wb, const float* x,
                     std::uint32_t n, float* y, Timings* t) {
  WeightBuffer& w = WeightSlot(wb, p);
  auto t0 = Clock::now();
  EncodeA(x, n, p.K, p);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  w.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  for (;;) {
    const auto st = w.run.state();
    if (st == ERT_CMD_STATE_COMPLETED)
      break;
    if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
        st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED)
      break;
  }
  w.run.wait();
  if (!p.warmed) {  // WP10: a program's first C write is lost on this
    p.warmed = true;  // XRT/xdna stack; redo the dispatch so C is valid.
    w.run.start();
    w.run.wait();
  }
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  DecodeC(y, n, p, p.ch + std::size_t{120} * kStripBytes, 2048);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

void NpuRuntime::Argmax(Program& p, WeightBuffer& wb, const float* x,
                        std::uint32_t n, std::uint32_t* out, Timings* t) {
  WeightBuffer& w = WeightSlot(wb, p);
  auto t0 = Clock::now();
  EncodeA(x, n, p.K, p);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  w.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  for (;;) {
    const auto st = w.run.state();
    if (st == ERT_CMD_STATE_COMPLETED)
      break;
    if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
        st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED)
      break;
  }
  w.run.wait();
  if (!p.warmed) {  // WP10: a program's first C write is lost on this
    p.warmed = true;  // XRT/xdna stack; redo the dispatch so C is valid.
    w.run.start();
    w.run.wait();
  }
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  const std::uint32_t strips = p.N / kStrip;
  struct Cand { float v; std::uint32_t id; };
  std::vector<Cand> best(n, {-INFINITY, 0});
  for (std::uint32_t si = 0; si < strips; ++si) {
    for (std::uint32_t cr = 0; cr < 4; ++cr) {
      const std::uint8_t* base = p.ch + std::size_t{si} * kStripBytes + std::size_t{cr} * kRowBytes;
      for (std::uint32_t r = 0; r < n; ++r) {
        float v;
        std::uint32_t lc;
        std::memcpy(&v, base + r * 8, 4);
        std::memcpy(&lc, base + r * 8 + 4, 4);
        const std::uint32_t id = si * kStrip + cr * 32 + lc;
        if (v > best[r].v || (v == best[r].v && id < best[r].id)) {
          best[r].v = v;
          best[r].id = id;
        }
      }
    }
  }
  for (std::uint32_t r = 0; r < n; ++r) out[r] = best[r].id;
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

namespace {
// Partial BO syncs that start or end inside a page are not reliable: flush whole pages.
void SyncToDevice(xrt::bo& bo, std::size_t offset, std::size_t size) {
  constexpr std::size_t kPage = 4096;
  const std::size_t lo = offset / kPage * kPage;
  const std::size_t hi = std::min<std::size_t>(bo.size(), (offset + size + kPage - 1) / kPage * kPage);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, hi - lo, lo);
}
constexpr std::uint32_t kConvHistOffset = 512;
constexpr std::uint32_t kConvTapOffset = 64;
constexpr std::uint32_t kConvMaxTaps = 7;
constexpr std::uint32_t kChanPerTriple = 1024;

inline std::size_t ConvRow(std::uint32_t role, std::uint32_t chan) {
  const std::uint32_t j = chan >> 10, col = (chan >> 7) & 7, r = (chan >> 5) & 3,
                      c = chan & 31;
  return (std::size_t{3 * j + role} * 8 + col) * 128 + r * 32 + c;
}
}  // namespace

ConvLayer NpuRuntime::PackConv(const float* w_in, const float* conv_w,
                               std::uint32_t taps, const float* w_out,
                               std::uint32_t hidden, Program& tmpl) {
  if (taps != kConvMaxTaps || hidden % kChanPerTriple != 0)
    throw std::runtime_error("conv layer shape unsupported by the conv program");
  ConvLayer cl;
  cl.hidden = hidden;
  cl.taps = taps;
  cl.prog = tmpl;
  const std::size_t cb = std::size_t{(3 * hidden + hidden) / kStrip} * tmpl.mrows * kStripBytes;
  cl.prog.c = xrt::bo(dev_, cb, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  cl.prog.ch = cl.prog.c.map<std::uint8_t*>();
  std::memset(cl.prog.ch, 0, cb);
  cl.prog.c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const std::uint32_t K = hidden, tk = K / kTileK;
  const std::size_t in_bytes = std::size_t{3 * hidden / kStrip} * (tk + 1) * 4 * kQuarterBytes;
  const std::size_t out_bytes = std::size_t{hidden / kStrip} * tk * 4 * kQuarterBytes;
  cl.w.bytes = in_bytes + out_bytes;
  cl.w.bo = xrt::bo(dev_, cl.w.bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  cl.w.host = cl.w.bo.map<std::uint8_t*>();
  std::memset(cl.w.host, 0, cl.w.bytes);
  {
    std::vector<float> perm(std::size_t{3 * hidden} * K);
    const std::uint32_t src_base[3] = {0, 2 * hidden, hidden};  // B, x, C row blocks of [B|C|x]
#pragma omp parallel for
    for (std::int64_t ch = 0; ch < hidden; ++ch)
      for (std::uint32_t role = 0; role < 3; ++role)
        std::memcpy(&perm[ConvRow(role, ch) * K],
                    w_in + (std::size_t{src_base[role]} + ch) * K, K * 4);
    PackMatInt5(perm.data(), 3 * hidden, K, cl.w.host, 1);
  }
  if (tmpl.mrows > 1) {
    // m64 conv: phase-2 A arrives pair-major (see the generator), so out_proj k-tiles are packed in the
    // order kt' = pair * slots + slot, which holds original tile 2 * slot + pair.
    const std::uint32_t slots = tk / 2;
    std::vector<std::uint32_t> perm(tk);
    for (std::uint32_t tp = 0; tp < tk; ++tp) perm[tp] = 2 * (tp % slots) + tp / slots;
    PackMatInt5(w_out, hidden, K, cl.w.host + in_bytes, 0, perm.data());
  } else {
    PackMatInt5(w_out, hidden, K, cl.w.host + in_bytes, 0);
  }
  const std::uint32_t groups = 3 * hidden / kStrip / 8;
  for (std::uint32_t g = 0; g < groups; ++g)
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t r = 0; r < 4; ++r) {
        const std::uint32_t s2 = g * 8 + col, j = g / 3;
        std::uint8_t* tile = cl.w.host + ((std::size_t{s2} * (tk + 1) + tk) * 4 + r) * kQuarterBytes;
        auto* tp = reinterpret_cast<std::uint16_t*>(tile + kConvTapOffset);
        for (std::uint32_t t = 0; t < taps; ++t)
          for (std::uint32_t cbk = 0; cbk < 4; ++cbk)
            for (std::uint32_t i = 0; i < 8; ++i) {
              const std::uint32_t chan = ((j * 8 + col) * 4 + r) * 32 + cbk * 8 + i;
              tp[(t * 4 + cbk) * 8 + i] = Bf16Bits(conv_w[std::size_t{chan} * taps + t]);
            }
      }
  cl.w.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  AttachTo(SlotRun(cl.w), cl.w, cl.prog);
  return cl;
}

void NpuRuntime::ConvReset(ConvLayer& cl) {
  const std::uint32_t tk = cl.hidden / kTileK, groups = 3 * cl.hidden / kStrip / 8;
  for (std::uint32_t g = 1; g < groups; g += 3)
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t r = 0; r < 4; ++r)
        std::memset(cl.w.host + ((std::size_t{g * 8 + col} * (tk + 1) + tk) * 4 + r) * kQuarterBytes +
                        kConvHistOffset, 0, 4 * 6 * 16);
  for (std::uint32_t g = 1; g < groups; g += 3)
    for (std::uint32_t col = 0; col < 8; ++col)
      SyncToDevice(cl.w.bo, (std::size_t{(g * 8 + col)} * (tk + 1) + tk) * 4 * kQuarterBytes,
                   4 * kQuarterBytes);
}

void NpuRuntime::ConvStep(ConvLayer& cl, const float* z, std::uint32_t n, float* y,
                          Timings* t) {
  auto t0 = Clock::now();
  EncodeA(z, n, cl.hidden, cl.prog);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  cl.prog.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  cl.w.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  for (;;) {
    const auto st = cl.w.run.state();
    if (st == ERT_CMD_STATE_COMPLETED)
      break;
    if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
        st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED)
      break;
  }
  cl.w.run.wait();
  if (!cl.prog.warmed) {  // WP10: a program's first C write is lost on this
    cl.prog.warmed = true;  // XRT/xdna stack; redo the dispatch so C is valid.
    cl.w.run.start();
    cl.w.run.wait();
  }
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  cl.prog.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  DecodeC(y, n, cl.prog, cl.prog.ch + std::size_t{3 * cl.hidden / kStrip} * cl.prog.mrows * kStripBytes, cl.hidden);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
  cl.last_n = n;
}

void NpuRuntime::ConvCommit(ConvLayer& cl, std::uint32_t k, bool settle) {
  k = std::min(k, cl.last_n);
  const std::uint32_t tk = cl.hidden / kTileK, groups = 3 * cl.hidden / kStrip / 8;
  const std::uint32_t hist = cl.taps - 1;
  for (std::uint32_t g = 1; g < groups; g += 3) {
    const std::uint32_t j = g / 3;
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t r = 0; r < 4; ++r) {
        const std::uint32_t s2 = g * 8 + col;
        auto* dst = reinterpret_cast<std::uint16_t*>(
            cl.w.host + ((std::size_t{s2} * (tk + 1) + tk) * 4 + r) * kQuarterBytes + kConvHistOffset);
        // role-1 bx output: per mb a 1024B region, per cbk 256B, per rh(8 rows) 128B, per row 16B.
        const std::uint8_t* sbase =
            cl.prog.ch + std::size_t{16 + j * 8 + col} * cl.prog.mrows * kStripBytes + std::size_t{r} * kRowBytes;
        for (std::uint32_t cbk = 0; cbk < 4; ++cbk) {
          std::uint16_t nh[6][8];
          for (std::uint32_t i = 0; i < hist; ++i) {
            const std::uint32_t idx = k + i;
            const std::uint16_t* row;
            if (idx < hist) {
              row = dst + (cbk * hist + idx) * 8;
            } else {
              const std::uint32_t ri = idx - hist;  // row index in this chunk's output
              const std::uint32_t mb = ri / 16;
              const std::uint32_t lr = ri % 16;
              row = reinterpret_cast<const std::uint16_t*>(
                  sbase + std::size_t{mb} * 4 * kRowBytes + std::size_t{cbk} * 256 +
                  std::size_t{lr / 8} * 128 + std::size_t{lr % 8} * 16);
            }
            std::memcpy(nh[i], row, 16);
          }
          std::memcpy(dst + cbk * hist * 8, nh, hist * 16);
        }
        // Push the new history out of the CPU caches ourselves: BO sync is not ordered with a fence
        // on this stack, and the device reads host memory directly.
        for (const char* q = reinterpret_cast<const char*>(dst); q < reinterpret_cast<const char*>(dst) + 4 * hist * 16 + 64; q += 64)
          _mm_clflushopt(const_cast<char*>(q));
        _mm_mfence();
      }
    static const int nsync = std::getenv("GUFO_SYNC_N") ? std::atoi(std::getenv("GUFO_SYNC_N")) : 1;
    for (int it = 0; it < nsync; ++it)
      for (std::uint32_t col = 0; col < 8; ++col)
        SyncToDevice(cl.w.bo, (std::size_t{(g * 8 + col)} * (tk + 1) + tk) * 4 * kQuarterBytes,
                     4 * kQuarterBytes);
  }
      // WP10: weight-page TO_DEVICE syncs on this XRT/xdna stack become visible to
  // the program's next dispatch only ~1 ms later, and no BO operation waits for
  // it (measured: 500 us late, 2 ms clean; full-BO sync and FROM_DEVICE
  // round-trips on the same pages do NOT help). When the caller dispatches this
  // layer immediately (m64 sub-steps), the wait must be explicit.
  if (settle) {
    static const int kSettleUs = [] {
      if (const char* e = std::getenv("GUFO_CONV_SETTLE_US")) return std::atoi(e);
      return 2000;
    }();
    if (kSettleUs > 0) usleep(static_cast<useconds_t>(kSettleUs));
  }
  cl.last_n = 0;  // C rows now reflected in history; a repeat commit is a no-op.
}

void NpuRuntime::ConvGetHistory(const ConvLayer& cl, std::uint16_t* rows6) {
  const std::uint32_t tk = cl.hidden / kTileK, groups = 3 * cl.hidden / kStrip / 8, hist = cl.taps - 1;
  for (std::uint32_t g = 1; g < groups; g += 3) {
    const std::uint32_t j = g / 3;
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t r = 0; r < 4; ++r) {
        const auto* src = reinterpret_cast<const std::uint16_t*>(
            cl.w.host + ((std::size_t{g * 8 + col} * (tk + 1) + tk) * 4 + r) * kQuarterBytes + kConvHistOffset);
        for (std::uint32_t cbk = 0; cbk < 4; ++cbk)
          for (std::uint32_t row = 0; row < hist; ++row)
            for (std::uint32_t i = 0; i < 8; ++i) {
              const std::uint32_t chan = ((j * 8 + col) * 4 + r) * 32 + cbk * 8 + i;
              rows6[std::size_t{row} * cl.hidden + chan] = src[(cbk * hist + row) * 8 + i];
            }
      }
  }
}

void NpuRuntime::ConvSetHistory(ConvLayer& cl, const std::uint16_t* rows6) {
  const std::uint32_t tk = cl.hidden / kTileK, groups = 3 * cl.hidden / kStrip / 8, hist = cl.taps - 1;
  for (std::uint32_t g = 1; g < groups; g += 3) {
    const std::uint32_t j = g / 3;
    for (std::uint32_t col = 0; col < 8; ++col) {
      for (std::uint32_t r = 0; r < 4; ++r) {
        auto* dst = reinterpret_cast<std::uint16_t*>(
            cl.w.host + ((std::size_t{g * 8 + col} * (tk + 1) + tk) * 4 + r) * kQuarterBytes + kConvHistOffset);
        for (std::uint32_t cbk = 0; cbk < 4; ++cbk)
          for (std::uint32_t row = 0; row < hist; ++row)
            for (std::uint32_t i = 0; i < 8; ++i) {
              const std::uint32_t chan = ((j * 8 + col) * 4 + r) * 32 + cbk * 8 + i;
              dst[(cbk * hist + row) * 8 + i] = rows6[std::size_t{row} * cl.hidden + chan];
            }
        for (const char* q = reinterpret_cast<const char*>(dst); q < reinterpret_cast<const char*>(dst) + 4 * hist * 16 + 64; q += 64)
          _mm_clflushopt(const_cast<char*>(q));
      }
      SyncToDevice(cl.w.bo, (std::size_t{(g * 8 + col)} * (tk + 1) + tk) * 4 * kQuarterBytes, 4 * kQuarterBytes);
    }
  }
  _mm_mfence();
}

void NpuRuntime::ConvCopyHistory(ConvLayer& dst, const ConvLayer& src) {
  const std::uint32_t tk = dst.hidden / kTileK, groups = 3 * dst.hidden / kStrip / 8;
  for (std::uint32_t g = 1; g < groups; g += 3)
    for (std::uint32_t col = 0; col < 8; ++col) {
      for (std::uint32_t r = 0; r < 4; ++r) {
        const std::size_t off = ((std::size_t{g * 8 + col} * (tk + 1) + tk) * 4 + r) * kQuarterBytes + kConvHistOffset;
        std::memcpy(dst.w.host + off, src.w.host + off, 4 * 6 * 16);
        for (std::size_t q = 0; q < 4 * 6 * 16 + 64; q += 64) _mm_clflushopt(dst.w.host + off + q);
      }
      SyncToDevice(dst.w.bo, (std::size_t{(g * 8 + col)} * (tk + 1) + tk) * 4 * kQuarterBytes, 4 * kQuarterBytes);
    }
  _mm_mfence();
}

void NpuRuntime::ConvRefresh(ConvLayer& cl) {
  const std::uint32_t tk = cl.hidden / kTileK, groups = 3 * cl.hidden / kStrip / 8;
  for (std::uint32_t g = 1; g < groups; g += 3)
    for (std::uint32_t col = 0; col < 8; ++col) {
      for (std::uint32_t r = 0; r < 4; ++r) {
        auto* dst = reinterpret_cast<volatile std::uint8_t*>(
            cl.w.host + ((std::size_t{g * 8 + col} * (tk + 1) + tk) * 4 + r) * kQuarterBytes + kConvHistOffset);
        for (std::uint32_t i = 0; i < 4 * 6 * 16; ++i) dst[i] = dst[i];
      }
      SyncToDevice(cl.w.bo, (std::size_t{(g * 8 + col)} * (tk + 1) + tk) * 4 * kQuarterBytes, 4 * kQuarterBytes);
    }
}

namespace {
constexpr std::uint32_t kAttnHd = 128, kAttnKvh = 4, kAttnChunkBytes = 3 * 4 * kQuarterBytes;
constexpr std::uint32_t kAttnQElems = 32, kAttnOutElems = 34, kAttnTailElems = 36;
constexpr std::size_t kAttnAElems = 1 + kAttnQElems + kAttnTailElems;
constexpr std::size_t kAttnPrefixCol = std::size_t{16} * 4 * 1536;

// Byte offset of piece `byte` (0..4607) of chunk q for kv head r inside the BO.
inline std::size_t AttnOff(std::uint32_t q, std::uint32_t r, std::uint32_t byte) {
  return std::size_t{q} * kAttnChunkBytes + std::size_t{byte / kQuarterBytes} * 4 * kQuarterBytes +
         std::size_t{r} * kQuarterBytes + byte % kQuarterBytes;
}
inline void AttnLoadChunk(const std::uint8_t* base, std::uint32_t q, std::uint32_t r, std::uint8_t* chunk) {
  for (std::uint32_t j = 0; j < 3; ++j)
    std::memcpy(chunk + j * kQuarterBytes, base + AttnOff(q, r, j * kQuarterBytes), kQuarterBytes);
}
inline void AttnStoreChunk(std::uint8_t* base, std::uint32_t q, std::uint32_t r, const std::uint8_t* chunk) {
  for (std::uint32_t j = 0; j < 3; ++j)
    std::memcpy(base + AttnOff(q, r, j * kQuarterBytes), chunk + j * kQuarterBytes, kQuarterBytes);
}
}  // namespace

AttnLayer NpuRuntime::MakeAttnLayer(std::uint32_t capacity_tokens, std::uint32_t ring_tokens,
                                    std::uint32_t window) {
  AttnLayer al;
  al.ring_tokens = ring_tokens;
  al.window = window;
  al.hist_tokens = ring_tokens ? ring_tokens : capacity_tokens;
  std::uint32_t mc = 1;
  if (ring_tokens) {
    al.capacity_chunks = ring_tokens / 16;  // ring: exactly ring_tokens/16 chunks (24 for 384)
  } else {
    const std::uint32_t qch = (capacity_tokens + 15) / 16;
    while (mc * 8 < qch) mc *= 2;
    al.capacity_chunks = mc * 8;
  }
  const std::size_t bytes = std::size_t{al.capacity_chunks} * kAttnChunkBytes;
  al.kv = xrt::bo(dev_, bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  al.host = al.kv.map<std::uint8_t*>();
  std::memset(al.host, 0, bytes);
  al.kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  al.kmax2.assign(kAttnKvh, 0.f);
  if (!ring_tokens) al.kpre.assign(std::size_t{al.capacity_chunks} * 16 * kAttnKvh, 0.f);
  if (!attn_a_) {
    attn_a_ = xrt::bo(dev_, kAttnAElems * 1152, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
    attn_c_ = xrt::bo(dev_, std::size_t{8} * kAttnOutElems * 4 * kRowBytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
    attn_ah_ = attn_a_.map<std::uint8_t*>();
    attn_ch_ = attn_c_.map<std::uint8_t*>();
    std::memset(attn_ah_, 0, kAttnAElems * 1152);
    attn_a_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    attn_c_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  return al;
}

void NpuRuntime::AttnReset(AttnLayer& al) {
  std::memset(al.host, 0, std::size_t{al.capacity_chunks} * kAttnChunkBytes);
  al.kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::fill(al.kmax2.begin(), al.kmax2.end(), 0.f);
  std::fill(al.kpre.begin(), al.kpre.end(), 0.f);
}

void NpuRuntime::AttnAppend(AttnLayer& al, std::uint32_t pos, std::uint32_t n, const float* k,
                            const std::uint16_t* v_hist, Timings* tm) {
  const auto t_start = Clock::now();
  const std::uint32_t kvd = kAttnKvh * kAttnHd;
  const std::uint32_t ring = al.ring_tokens;
  // Chunk list to touch: absolute chunks for the full layout, ring slots otherwise.
  std::uint32_t chunks[2], nch = 0;
  if (ring) {
    const std::uint32_t s0 = (pos % ring) / 16, s1 = ((pos + n - 1) % ring) / 16;
    if (s0 == s1) { chunks[0] = s0; nch = 1; } else { chunks[0] = s0; chunks[1] = s1; nch = 2; }
  } else {
    chunks[0] = pos / 16;
    nch = (pos + n - 1) / 16 - chunks[0] + 1;
  }
  // Token written at slot offset `tt` of chunk `q` by this pass, or kInvalid if that
  // slot belongs to another pass (full layout: q*16+tt is always the absolute token).
  const std::uint32_t kInvalid = 0xFFFFFFFFu;
  auto tok_at = [&](std::uint32_t q, std::uint32_t tt) -> std::uint32_t {
    if (!ring) {
      const std::uint32_t tok = q * 16 + tt;
      return (tok >= pos && tok < pos + n) ? tok : kInvalid;
    }
    const std::uint32_t d = (q * 16 + tt + ring - (pos % ring)) % ring;
    return d < n ? pos + d : kInvalid;
  };
#pragma omp parallel for
  for (std::int32_t r = 0; r < static_cast<std::int32_t>(kAttnKvh); ++r) {
    for (std::uint32_t t = 0; t < n; ++t) {
      float s2 = 0;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) s2 += k[std::size_t{t} * kvd + r * kAttnHd + d] * k[std::size_t{t} * kvd + r * kAttnHd + d];
      al.kmax2[r] = std::max(al.kmax2[r], s2);
      if (!al.kpre.empty()) al.kpre[std::size_t{pos + t} * kAttnKvh + r] = al.kmax2[r];
    }
    for (std::uint32_t ci = 0; ci < nch; ++ci) {
      const std::uint32_t q = chunks[ci];
      alignas(64) std::uint8_t chunk[4608];
      AttnLoadChunk(al.host, q, r, chunk);
      for (std::uint32_t tt = 0; tt < 16; ++tt) {
        const std::uint32_t tok = tok_at(q, tt);
        if (tok == kInvalid) continue;  // slot not written by this pass
        const std::uint32_t kb = tt / 8, i8 = tt % 8;
        for (std::uint32_t ks = 0; ks < 16; ++ks) {
          float v8[8];
          std::memcpy(v8, k + std::size_t{tok - pos} * kvd + r * kAttnHd + ks * 8, 32);
          EncGroup(v8, chunk + (std::size_t{kb} * 16 + ks) * 8 * 9 + i8 * 9);
        }
      }
      for (std::uint32_t kb = 0; kb < 2; ++kb) {
        // V groups are stored finalized, so a touched block is rewritten whole from
        // v_hist: it mirrors current cache contents in slot order (live values for
        // slots this pass does not write, zeros past total). Never clobber a live
        // neighbour of a partial block; never read past the cache.
        const std::uint32_t t0 = q * 16 + kb * 8;
        const std::uint32_t cap_tok = ring ? ring : al.hist_tokens;
        if (t0 >= cap_tok) continue;      // slots do not exist
        if (!ring && t0 + 8 <= pos) continue;  // fully-finished block (original skip)
        for (std::uint32_t db = 0; db < 16; ++db)
          for (std::uint32_t g = 0; g < 8; ++g) {
            float v8[8];
            for (std::uint32_t i8 = 0; i8 < 8; ++i8) {
              const std::uint32_t slot = t0 + i8;  // ring: slot == absolute mod ring
              v8[i8] = slot < cap_tok ? Bf16Float(v_hist[std::size_t{slot} * kvd + r * kAttnHd + db * 8 + g])
                                      : 0.f;
            }
            EncGroup(v8, chunk + 2304 + ((std::size_t{kb} * 16 + db) * 8 + g) * 9);
          }
      }
      AttnStoreChunk(al.host, q, r, chunk);
      if (ring) SyncToDevice(al.kv, std::size_t{q} * kAttnChunkBytes, kAttnChunkBytes);
    }
  }
  if (!ring) {
    const std::uint32_t q0 = pos / 16, q1 = (pos + n - 1) / 16;
    SyncToDevice(al.kv, std::size_t{q0} * kAttnChunkBytes, std::size_t{q1 - q0 + 1} * kAttnChunkBytes);
  }
  if (tm) tm->attn_append += Ms(t_start);
}

bool NpuRuntime::AttnRun(AttnLayer& al, std::uint32_t pos, std::uint32_t n, const float* q,
                         const std::uint16_t* k_hist, const std::uint16_t* v_hist,
                         const std::string& programs_dir, float* out, Timings* tm) {
  auto t_start = Clock::now();
  const std::uint32_t kvd = kAttnKvh * kAttnHd, qd = 16 * kAttnHd;
  const float qscale = 1.4426950408889634f / std::sqrt(static_cast<float>(kAttnHd));
  const std::uint32_t total = pos + n;
  const std::uint32_t ring = al.ring_tokens;
  std::uint32_t mc = 1;
  if (ring) {
    mc = al.capacity_chunks / 8;  // ring: fixed program (24 chunks = 3 per column)
  } else {
    while (mc * 8 * 16 < total) mc *= 2;
  }
  if (mc * 8 > al.capacity_chunks) return false;
  // header: pos, n, qhi, win, tab[24], then M per (kv head, row)
  auto* hdr = attn_ah_;
  std::memset(hdr, 0, 1152);
  std::int32_t ih[4] = {static_cast<std::int32_t>(pos), static_cast<std::int32_t>(n),
                       static_cast<std::int32_t>((total - 1) / 16),
                       static_cast<std::int32_t>(al.window)};
  std::memcpy(hdr, ih, 16);
  if (ring) {
    // tab[slot] = qhi - absolute chunk (latest occupant), -1 if that chunk is not needed
    std::int32_t* tab = reinterpret_cast<std::int32_t*>(hdr + 16);
    const std::int32_t qhi = ih[2];
    for (std::int32_t s = 0; s < 24; ++s) tab[s] = -1;
    const std::uint32_t lo0 = pos + 1 > al.window ? pos + 1 - al.window : 0;
    for (std::uint32_t q = lo0 / 16; q <= static_cast<std::uint32_t>(qhi); ++q)
      tab[q % 24] = qhi - static_cast<std::int32_t>(q);
  }
  auto hist_slot = [&](std::uint32_t tok) { return ring ? tok % ring : tok; };
  auto* mrow = reinterpret_cast<float*>(hdr + 128);
#pragma omp parallel for schedule(static, 1)
  for (std::int32_t rr = 0; rr < static_cast<std::int32_t>(kAttnKvh); ++rr) {
    const std::uint32_t r = static_cast<std::uint32_t>(rr);
    std::vector<float> qs(std::size_t{64} * kAttnHd);
    const float kbound = std::sqrt(al.kmax2[r]);
    // Q block for this kv head: rows i = hq*16 + t (hq = 0..3), zero for t >= n
    std::fill(qs.begin(), qs.end(), 0.f);
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t hq = i / 16, t = i % 16;
      if (t >= n) continue;
      const float* src = q + std::size_t{t} * qd + (r * 4 + hq) * kAttnHd;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) qs[std::size_t{i} * kAttnHd + d] = src[d] * qscale;
    }
    std::uint8_t* qb = attn_ah_ + std::size_t{1 + r * 8} * 1152;
    for (std::uint32_t rb = 0; rb < 8; ++rb)
      for (std::uint32_t ks = 0; ks < 16; ++ks)
        for (std::uint32_t rib = 0; rib < 8; ++rib) {
          const float* src = &qs[std::size_t{rb * 8 + rib} * kAttnHd + ks * 8];
          EncGroup(src, qb + (std::size_t{rb * 16 + ks} * 8 + rib) * 9);
        }
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t hq = i / 16, t = i % 16;
      float qn = 0;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) qn += qs[std::size_t{i} * kAttnHd + d] * qs[std::size_t{i} * kAttnHd + d];
      const float bound = std::sqrt(qn) * kbound;
      float m = -1e30f;
      if (t < n) {
        const std::uint32_t last = pos + t;  // keys 0..last valid for this row
        const std::uint32_t lo = last >= 32 ? last - 31 : 0;
        auto score = [&](std::uint32_t tok) {
          float s = 0;
          const std::uint16_t* kr = k_hist + std::size_t{hist_slot(tok)} * kvd + r * kAttnHd;
          for (std::uint32_t d = 0; d < kAttnHd; ++d) s += qs[std::size_t{i} * kAttnHd + d] * Bf16Float(kr[d]);
          return s;
        };
        for (std::uint32_t tok = lo; tok <= last; ++tok) m = std::max(m, score(tok));
        // extra early-token probing only applies to the absolute (full) cache
        for (std::uint32_t tok = 0; !ring && tok < std::min(8u, lo); ++tok) m = std::max(m, score(tok));
        m = std::min(std::max(m, bound - 100.f), bound);
      } else {
        m = 0.f;
      }
      mrow[r * 64 + i] = m;
      (void)hq;
    }
  }
  {
    // Tail: exact bf16 V (and the cached K groups) for the 8-token blocks the rows live in.
    const std::uint32_t t0 = pos & ~7u;
    for (std::uint32_t r = 0; r < kAttnKvh; ++r) {
      alignas(64) std::uint8_t chunk[4608];
      for (std::uint32_t tb = 0; tb < 3; ++tb) {
        std::uint8_t* dst = attn_ah_ + (std::size_t{1 + kAttnQElems} + (tb * 4 + r) * 3) * 1152;
        std::memset(dst, 0, 3 * 1152);
        const std::uint32_t tokbase = t0 + tb * 8;
        if (tokbase >= total) continue;
        AttnLoadChunk(al.host, hist_slot(tokbase) / 16, r, chunk);
        const std::uint32_t kb = (tokbase % 16) / 8;
        for (std::uint32_t ks = 0; ks < 16; ++ks)
          for (std::uint32_t tt = 0; tt < 8; ++tt)
            if (tokbase + tt < total)
              std::memcpy(dst + (ks * 8 + tt) * 9, chunk + (kb * 16 + ks) * 72 + tt * 9, 9);
        auto* vd = reinterpret_cast<std::uint16_t*>(dst + 1152);
        for (std::uint32_t db = 0; db < 16; ++db)
          for (std::uint32_t tt = 0; tt < 8; ++tt)
            if (tokbase + tt < total)
              std::memcpy(vd + (db * 8 + tt) * 8,
                          v_hist + std::size_t{hist_slot(tokbase + tt)} * kvd + r * kAttnHd + db * 8, 16);
      }
    }
  }
  attn_a_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto it = attn_insts_.find(mc);
  if (it == attn_insts_.end()) {
    const std::string path = programs_dir + "/insts_attn_" + std::to_string(mc) + ".txt";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    xrt::bo bo(dev_, sz, XCL_BO_FLAGS_CACHEABLE, kern_.group_id(1));
    std::memcpy(bo.map<void*>(), buf.data(), sz);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    it = attn_insts_.emplace(mc, std::make_pair(bo, sz / 4)).first;
  }
  auto rit = al.runs.find(mc);
  if (rit == al.runs.end()) {
    xrt::run run(kern_);
    run.set_arg(0, 3u);
    run.set_arg(1, it->second.first);
    run.set_arg(2, static_cast<unsigned>(it->second.second));
    run.set_arg(3, attn_a_);
    run.set_arg(4, al.kv);
    run.set_arg(5, attn_c_);
    rit = al.runs.emplace(mc, std::move(run)).first;
  }
  if (tm) { tm->attn_prep += Ms(t_start); t_start = Clock::now(); }
  rit->second.start();
  for (;;) {
    const auto st = rit->second.state();
    if (st == ERT_CMD_STATE_COMPLETED) break;
    if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED && st != ERT_CMD_STATE_RUNNING &&
        st != ERT_CMD_STATE_SUBMITTED)
      break;
  }
  rit->second.wait();
  attn_c_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (tm) { tm->attn_wait += Ms(t_start); t_start = Clock::now(); }
  // merge: partials of all columns share M, so they add
  bool ok = true;
  alignas(64) static thread_local std::vector<float> tot;
  for (std::uint32_t r = 0; r < kAttnKvh; ++r) {
    tot.assign(std::size_t{kAttnOutElems} * 256, 0.f);
    for (std::uint32_t c = 0; c < 8; ++c)
      for (std::uint32_t i = 0; i < kAttnOutElems; ++i) {
        const auto* e = reinterpret_cast<const float*>(attn_ch_ + ((std::size_t{c} * kAttnOutElems + i) * 4 + r) * kRowBytes);
        float* d = &tot[std::size_t{i} * 256];
        for (int x = 0; x < 256; x += 16)
          _mm512_storeu_ps(d + x, _mm512_add_ps(_mm512_loadu_ps(d + x), _mm512_loadu_ps(e + x)));
      }
    const float* o = tot.data();
    const float* lt = tot.data() + std::size_t{32} * 256;
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t hq = i / 16, t = i % 16;
      if (t >= n) continue;
      const std::uint32_t rb = i / 8, rib = i % 8;
      float l = 0;
      for (int cc = 0; cc < 8; ++cc) l += lt[rb * 64 + rib * 8 + cc];
      if (!(l > 1e-30f) || !(l < 1e30f)) { ok = false; continue; }
      const float inv = 1.0f / l;
      for (std::uint32_t db = 0; db < 16; ++db)
        for (int cc = 0; cc < 8; ++cc)
          out[std::size_t{t} * qd + (r * 4 + hq) * kAttnHd + db * 8 + cc] =
              o[(std::size_t{rb} * 16 + db) * 64 + rib * 8 + cc] * inv;
    }
  }
  (void)kAttnQElems;
  if (tm) { tm->attn_merge += Ms(t_start); ++tm->attn_calls; }
  return ok;
}

bool AttnPrefill::Init(const std::string& dir, std::string* error) {
  try {
    dir_ = dir;
    dev_ = xrt::device(0u);
    const xrt::xclbin xb{dir + "/final_attnp.xclbin"};
    dev_.register_xclbin(xb);
    ctx_ = xrt::hw_context(dev_, xb.get_uuid());
    kern_ = xrt::kernel(ctx_, xb.get_kernels().at(0).get_name());
    for (int s = 0; s < 2; ++s) {
      a_[s] = xrt::bo(dev_, kAttnPrefixCol * 8, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
      c_[s] = xrt::bo(dev_, std::size_t{8} * kAttnOutElems * 4 * kRowBytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
      ah_[s] = a_[s].map<std::uint8_t*>();
      ch_[s] = c_[s].map<std::uint8_t*>();
      std::memset(ah_[s], 0, kAttnPrefixCol * 8);
      a_[s].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      c_[s].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    ready_ = true;
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

bool AttnPrefill::Prep(int slot, AttnLayer& al, std::uint32_t pos0, std::uint32_t n, const float* q,
                       const std::uint16_t* k_hist, const std::uint16_t* v_hist, std::uint32_t* mc_out) {
  const std::uint32_t ring = al.ring_tokens;
  if (!ready_ || (!ring && al.kpre.empty()) || n == 0 || n > 128) return false;
  const std::uint32_t kvd = kAttnKvh * kAttnHd, qd = 16 * kAttnHd;
  const float qscale = 1.4426950408889634f / std::sqrt(static_cast<float>(kAttnHd));
  const std::uint32_t total = pos0 + n;
  static const std::uint32_t kMc[] = {8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
  std::uint32_t mc = 0;
  if (ring) {
    mc = al.capacity_chunks;  // ring: the whole ring is streamed, 24 chunks per column
    if (mc != 24) return false;
  } else {
    for (std::uint32_t v : kMc)
      if (v * 16 >= total) { mc = v; break; }
    if (mc == 0 || mc > al.capacity_chunks) return false;
  }
  auto hist_slot = [&](std::uint32_t tok) { return ring ? tok % ring : tok; };
  std::memset(ah_[slot], 0, kAttnPrefixCol * 8);
#pragma omp parallel for schedule(static, 1)
  for (std::int32_t cr = 0; cr < 32; ++cr) {
    const std::uint32_t c = static_cast<std::uint32_t>(cr) / 4, r = static_cast<std::uint32_t>(cr) % 4;
    const std::uint32_t pos = pos0 + 16 * c;
    const std::uint32_t nc = 16 * c >= n ? 0 : std::min(16u, n - 16 * c);
    std::uint8_t* core = ah_[slot] + std::size_t{c} * kAttnPrefixCol;
    auto unit = [&](std::uint32_t u) { return core + std::size_t{u} * 4 * 1536 + std::size_t{r} * 1536; };
    const std::uint32_t tot_c = pos + nc;
    std::int32_t ih[4] = {static_cast<std::int32_t>(pos), static_cast<std::int32_t>(nc),
                         static_cast<std::int32_t>(tot_c ? (tot_c - 1) / 16 : 0),
                         static_cast<std::int32_t>(al.window)};
    std::memcpy(unit(0), ih, 16);
    if (nc == 0) continue;
    if (ring) {
      auto* tab = reinterpret_cast<std::int32_t*>(unit(0) + 16);
      for (std::int32_t sl = 0; sl < 24; ++sl) tab[sl] = -1;
      const std::uint32_t lo0 = pos + 1 > al.window ? pos + 1 - al.window : 0;
      for (std::uint32_t qq = lo0 / 16; qq <= static_cast<std::uint32_t>(ih[2]); ++qq)
        tab[qq % 24] = ih[2] - static_cast<std::int32_t>(qq);
    }
    auto* mrow = reinterpret_cast<float*>(unit(0) + 128);
    std::vector<float> qs(std::size_t{64} * kAttnHd, 0.f);
    const float kbound = std::sqrt(ring ? al.kmax2[r] : al.kpre[std::size_t{tot_c - 1} * kAttnKvh + r]);
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t hq = i / 16, t = i % 16;
      if (t >= nc) continue;
      const float* src = q + std::size_t{16 * c + t} * qd + (r * 4 + hq) * kAttnHd;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) qs[std::size_t{i} * kAttnHd + d] = src[d] * qscale;
    }
    std::uint8_t qb[9216];
    for (std::uint32_t rb = 0; rb < 8; ++rb)
      for (std::uint32_t ks = 0; ks < 16; ++ks)
        for (std::uint32_t rib = 0; rib < 8; ++rib)
          EncGroup(&qs[std::size_t{rb * 8 + rib} * kAttnHd + ks * 8], qb + (std::size_t{rb * 16 + ks} * 8 + rib) * 9);
    for (std::uint32_t u = 0; u < 6; ++u) std::memcpy(unit(1 + u), qb + u * 1536, 1536);
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t t = i % 16;
      float qn = 0;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) qn += qs[std::size_t{i} * kAttnHd + d] * qs[std::size_t{i} * kAttnHd + d];
      const float bound = std::sqrt(qn) * kbound;
      float m = 0.f;
      if (t < nc) {
        m = -1e30f;
        const std::uint32_t last = pos + t;
        const std::uint32_t lo = last >= 32 ? last - 31 : 0;
        auto score = [&](std::uint32_t tok) {
          float sc = 0;
          const std::uint16_t* kr = k_hist + std::size_t{hist_slot(tok)} * kvd + r * kAttnHd;
          for (std::uint32_t d = 0; d < kAttnHd; ++d) sc += qs[std::size_t{i} * kAttnHd + d] * Bf16Float(kr[d]);
          return sc;
        };
        for (std::uint32_t tok = lo; tok <= last; ++tok) m = std::max(m, score(tok));
        for (std::uint32_t tok = 0; !ring && tok < std::min(8u, lo); ++tok) m = std::max(m, score(tok));
        m = std::min(std::max(m, bound - 100.f), bound);
      }
      mrow[i] = m;
    }
    const std::uint32_t t0 = pos & ~7u;
    alignas(64) std::uint8_t chunk[4608];
    for (std::uint32_t tb = 0; tb < 3; ++tb) {
      alignas(64) std::uint8_t tl[4608] = {0};
      const std::uint32_t tokbase = t0 + tb * 8;
      if (tokbase < tot_c) {
        AttnLoadChunk(al.host, hist_slot(tokbase) / 16, r, chunk);
        const std::uint32_t kb = (tokbase % 16) / 8;
        for (std::uint32_t ks = 0; ks < 16; ++ks)
          for (std::uint32_t tt = 0; tt < 8; ++tt)
            if (tokbase + tt < tot_c) std::memcpy(tl + (ks * 8 + tt) * 9, chunk + (kb * 16 + ks) * 72 + tt * 9, 9);
        auto* vd = reinterpret_cast<std::uint16_t*>(tl + 1152);
        for (std::uint32_t db = 0; db < 16; ++db)
          for (std::uint32_t tt = 0; tt < 8; ++tt)
            if (tokbase + tt < tot_c)
              std::memcpy(vd + (db * 8 + tt) * 8, v_hist + std::size_t{hist_slot(tokbase + tt)} * kvd + r * kAttnHd + db * 8, 16);
      }
      for (std::uint32_t e = 0; e < 3; ++e) std::memcpy(unit(7 + tb * 3 + e), tl + e * 1536, 1536);
    }
  }
  *mc_out = mc;
  a_[slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return true;
}

bool AttnPrefill::Launch(int slot, AttnLayer& al, std::uint32_t mc, xrt::run** run_out) {
  auto it = insts_.find(mc);
  if (it == insts_.end()) {
    const std::string path = dir_ + "/insts_attnp_" + std::to_string(mc) + ".txt";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    xrt::bo bo(dev_, sz, XCL_BO_FLAGS_CACHEABLE, kern_.group_id(1));
    std::memcpy(bo.map<void*>(), buf.data(), sz);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    it = insts_.emplace(mc, std::make_pair(bo, sz / 4)).first;
  }
  auto rit = runs_.find({&al, mc, slot});
  if (rit == runs_.end()) {
    xrt::run run(kern_);
    run.set_arg(0, 3u);
    run.set_arg(1, it->second.first);
    run.set_arg(2, static_cast<unsigned>(it->second.second));
    run.set_arg(3, a_[slot]);
    run.set_arg(4, al.kv);
    run.set_arg(5, c_[slot]);
    rit = runs_.emplace(std::make_tuple(&al, mc, slot), std::move(run)).first;
  }
  *run_out = &rit->second;
  return true;
}

bool AttnPrefill::Merge(int slot, std::uint32_t n, float* out) {
  const std::uint32_t qd = 16 * kAttnHd;
  int bad = 0;
#pragma omp parallel for collapse(2) reduction(+ : bad)
  for (std::int32_t c = 0; c < 8; ++c) {
    for (std::int32_t r = 0; r < static_cast<std::int32_t>(kAttnKvh); ++r) {
      const std::uint32_t nc = 16 * c >= n ? 0 : std::min(16u, n - 16 * c);
      if (nc == 0) continue;
      float tot[34 * 256];
      for (std::uint32_t i = 0; i < kAttnOutElems; ++i)
        std::memcpy(tot + std::size_t{i} * 256, ch_[slot] + ((std::size_t{c} * kAttnOutElems + i) * 4 + r) * kRowBytes, 1024);
      const float* o = tot;
      const float* lt = tot + std::size_t{32} * 256;
      for (std::uint32_t i = 0; i < 64; ++i) {
        const std::uint32_t hq = i / 16, t = i % 16;
        if (t >= nc) continue;
        const std::uint32_t rb = i / 8, rib = i % 8;
        float l = 0;
        for (int cc = 0; cc < 8; ++cc) l += lt[rb * 64 + rib * 8 + cc];
        if (!(l > 1e-30f) || !(l < 1e30f)) { ++bad; continue; }
        const float inv = 1.0f / l;
        for (std::uint32_t db = 0; db < 16; ++db)
          for (int cc = 0; cc < 8; ++cc)
            out[std::size_t{16 * c + t} * qd + (r * 4 + hq) * kAttnHd + db * 8 + cc] =
                o[(std::size_t{rb} * 16 + db) * 64 + rib * 8 + cc] * inv;
      }
    }
  }
  return bad == 0;
}


bool AttnPrefill::Run(AttnLayer& al, std::uint32_t pos0, std::uint32_t n, const float* q,
                      const std::uint16_t* k_hist, const std::uint16_t* v_hist, float* out,
                      NpuRuntime::Timings* tm) {
  auto t_start = Clock::now();
  std::uint32_t mc = 0;
  xrt::run* run = nullptr;
  if (!Prep(0, al, pos0, n, q, k_hist, v_hist, &mc) || !Launch(0, al, mc, &run)) return false;
  if (tm) { tm->attn_prep += Ms(t_start); t_start = Clock::now(); }
  run->start();
  run->wait();
  c_[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (tm) { tm->attn_wait += Ms(t_start); t_start = Clock::now(); }
  const bool ok = Merge(0, n, out);
  if (tm) { tm->attn_merge += Ms(t_start); ++tm->attn_calls; }
  return ok;
}

bool AttnPrefill::RunMany(AttnLayer& al, const Pass* ps, std::size_t np, const std::uint16_t* k_hist,
                          const std::uint16_t* v_hist, NpuRuntime::Timings* tm,
                          const std::function<void(std::size_t)>* pre) {
  if (np == 0) return true;
  auto t = Clock::now();
  std::vector<std::uint32_t> mcs(np);
  xrt::run* run = nullptr;
  if (pre) (*pre)(0);
  bool ok = Prep(0, al, ps[0].pos0, ps[0].n, ps[0].q, k_hist, v_hist, &mcs[0]) && Launch(0, al, mcs[0], &run);
  if (!ok) return false;
  run->start();
  if (tm) { tm->attn_prep += Ms(t); }
  for (std::size_t i = 0; i < np; ++i) {
    const int slot = static_cast<int>(i & 1);
    xrt::run* next = nullptr;
    bool nok = true;
    const bool ahead = i + 1 < np;
    if (ahead && !pre) {
      t = Clock::now();
      nok = Prep(slot ^ 1, al, ps[i + 1].pos0, ps[i + 1].n, ps[i + 1].q, k_hist, v_hist, &mcs[i + 1]) &&
            Launch(slot ^ 1, al, mcs[i + 1], &next);
      if (tm) tm->attn_prep += Ms(t);
    }
    t = Clock::now();
    run->wait();
    c_[slot].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (tm) tm->attn_wait += Ms(t);
    if (ahead && pre) {
      t = Clock::now();
      (*pre)(i + 1);
      nok = Prep(slot ^ 1, al, ps[i + 1].pos0, ps[i + 1].n, ps[i + 1].q, k_hist, v_hist, &mcs[i + 1]) &&
            Launch(slot ^ 1, al, mcs[i + 1], &next);
      if (tm) tm->attn_prep += Ms(t);
    }
    if (!nok) return false;
    if (next) next->start();
    t = Clock::now();
    ok = Merge(slot, ps[i].n, ps[i].out) && ok;
    if (tm) { tm->attn_merge += Ms(t); ++tm->attn_calls; }
    run = next;
  }
  return ok;
}

}  // namespace gufo::npu::runtime
