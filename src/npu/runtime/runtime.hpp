#ifndef GUFO_NPU_RUNTIME_HPP_
#define GUFO_NPU_RUNTIME_HPP_

#include <immintrin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "src/npu/runtime/weight_cache.hpp"

namespace gufo::npu::runtime {

/// Attention row maximum = the probed maximum (no lower clamp from the key-norm
/// bound). Rows whose true maximum exceeds the probe by more than the bf16
/// range overflow and are reported as unusable, then recomputed by the caller.
/// Needed for models with outlier keys.
extern bool g_attn_tight_m;

constexpr std::uint32_t kMaxRows = 16;
constexpr std::uint32_t kTileK = 64;
constexpr std::uint32_t kStrip = 128;
constexpr std::uint32_t kQuarterBytes = 1536;
constexpr std::uint32_t kWordBytes = 1408;
// Decode weight stream format: 5-bit values (three per 16-bit word, 1536 B per
// 2048 weights) or 4-bit values (four per word, 1152 B). The choice is made by
// the decode xclbin that is loaded (NpuRuntime::Init), and applies to the
// packers and byte counts below.
inline bool g_decode_int4 = false;
inline std::uint32_t QuarterBytesW() {
  return g_decode_int4 ? 1152 : 1536;
}
inline std::uint32_t WordBytesW() {
  return g_decode_int4 ? 1024 : 1408;
}
// C staging: one 1024 B slot per core row and strip (576 B used by bfp16
// tiles). C staging: one 1 KB element per core row and row-block. A strip of an
// m-row-block program is [row-block][core row][kRowBytes], i.e. mrows *
// kStripBytes bytes.
constexpr std::uint32_t kRowBytes = 1024;
constexpr std::uint32_t kStripBytes = 4 * kRowBytes;

inline float RoundHalf(float v) {
  return _cvtsh_ss(_cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT));
}

inline void EncGroup(const float* v, std::uint8_t* out) {
  const __m256 f = _mm256_loadu_ps(v);
  const __m256i bits = _mm256_castps_si256(f);
  const __m256i ex =
      _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
  __m128i m = _mm_max_epu32(_mm256_castsi256_si128(ex),
                            _mm256_extracti128_si256(ex, 1));
  m = _mm_max_epu32(m, _mm_shuffle_epi32(m, 0x4E));
  m = _mm_max_epu32(m, _mm_shuffle_epi32(m, 0xB1));
  const unsigned max_exp = static_cast<unsigned>(_mm_cvtsi128_si32(m));
  const std::uint32_t ib = max_exp >= 7 ? (260u - max_exp) << 23 : 0u;
  float inv;
  std::memcpy(&inv, &ib, 4);
  const __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(f, _mm256_set1_ps(inv)));
  const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(qi),
                                      _mm256_extracti128_si256(qi, 1));
  const __m128i p8 = _mm_packs_epi16(p16, p16);
  out[0] = static_cast<std::uint8_t>(max_exp);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(out + 1), p8);
}

inline void DecGroup(const std::uint8_t* p, float* v) {
  const int e = p[0];
  const std::uint32_t mb = e > 6 ? static_cast<std::uint32_t>(e - 6) << 23 : 0u;
  float mult;
  std::memcpy(&mult, &mb, 4);
  const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 1));
  const __m256i i32 = _mm256_cvtepi8_epi32(b8);
  _mm256_storeu_ps(
      v, _mm256_mul_ps(_mm256_cvtepi32_ps(i32), _mm256_set1_ps(mult)));
}

inline std::uint16_t Bf16Bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}
inline float Bf16Float(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Packer: int5 (3 values per 16-bit word, 11 words + 128B bf16 scales = 1536B
// per core per k-tile of 64), laid out for the split-B 16-row decode GEMM.
void PackMatInt5(const float* W, std::uint32_t N, std::uint32_t K,
                 std::uint8_t* dst, std::uint32_t extra_tiles = 0,
                 const std::uint32_t* tile_perm = nullptr);
void PackMatInt5Q(const std::int8_t* q, const std::uint16_t* sc,
                  std::uint32_t N, std::uint32_t K, std::uint8_t* dst,
                  std::uint32_t extra_tiles = 0,
                  const std::uint32_t* tile_perm = nullptr);

// Instruction-stream program (one runtime sequence) with its A/C staging BOs.
struct WeightBuffer;  // fwd: Program pins one via wpin (see
                      // NpuRuntime::WeightSlot)
struct Program {
  std::uint32_t K{}, N{};
  std::uint32_t mrows{
      1};  // A row-blocks per B tile (1 = 16 rows decode, 4 = m64 prefill)
  xrt::bo insts, a, c;
  xrt::bo a2;  // AGENT-8 cfg29: second A stream (device act-wire tap programs)
  bool four_a{false};  // program args are (A1, A2, B, C)
  std::size_t words{};
  std::uint8_t* ah{};
  std::uint8_t* ch{};
  WeightBuffer* wpin{};  // first-bound weight buffer (see WeightSlot)
  bool warmed{
      false};  // first dispatch is a throwaway (see WarmOnce in runtime.cpp)
};

// A packed weight buffer attached as a runnable kernel argument set.
struct WeightBuffer {
  xrt::run run;
  xrt::bo bo;
  std::uint8_t* host{};
  std::size_t bytes{};
};

// Short-conv layer fused as GEMM epilogue: in_proj (B|C|x co-located per core)
// + 7-tap temporal conv + out_proj in one dispatch. Conv history lives in the
// weight buffer's extra tile, so the device keeps no state across dispatches.
struct ConvLayer {
  WeightBuffer w;
  Program prog;
  std::uint32_t hidden{}, taps{};
  std::uint32_t last_n{0};
};

// Full-attention layer KV cache in the attention kernel layout, plus its
// per-bucket runs. Layout: [chunk q = token/16][3][4 kv heads][1536 B].
// Sliding-window layers use a ring: token t lives at slot t % ring_tokens, the
// kernel gets a slot->absolute map (tab) in the header and masks by absolute
// position.
struct AttnLayer {
  xrt::bo kv_full;  // 2 chunks of zero padding in front of the attention view,
                    // so key-history reads never underflow
  xrt::bo kv;
  std::uint8_t* host{};
  std::uint32_t capacity_chunks{};
  std::uint32_t ring_tokens{0};  // 0 = absolute (full-attention) layout
  std::uint32_t window{0};       // sliding window size (0 = full context)
  std::uint32_t hist_tokens{
      0};  // tokens held in v_hist/k_hist (cache bound for block writes)
  std::vector<float> kmax2;  // running max |k|^2 per kv head
  std::vector<float> kpre;   // prefix max |k|^2 per (token, kv head),
                             // full-attention layers only
  std::unordered_map<std::uint32_t, xrt::run> runs;
};

/// Device attention prep: QK-norm, RoPE, K/V chunk build, tail blocks and
/// softmax bounds from the QKV GEMM wire.
class AttnDevPrep {
public:
  static constexpr std::uint32_t kGroups =
      64;  // 16-token groups per dispatch (one 1024-row GEMM result)
  static constexpr std::size_t kTabBytes = 5120, kChunkBytes = 18432,
                               kPassBytes = 786432, kPassCol = 98304;
  static constexpr std::uint32_t kPasses =
      128;  // pass buffers held for one super-chunk (16 x 8)
  bool Init(const std::string& dir, xrt::device& dev, std::string* error);
  bool ready() const { return ready_; }
  /// Per-layer rope/norm constants. fac: per-frequency divisor (null for
  /// sliding-window layers).
  bool SetupLayer(std::size_t id, std::uint32_t max_tokens, const float* qn,
                  const float* kn, float eps, float theta, const float* fac);
  /// Runs both dispatches for rows [pos0, pos0 + n) of the QKV result in qkv_c
  /// (pos0 % 16 == 0, n <= 1024). pass0: index of the first of the 8 pass
  /// buffers this chunk fills. Updates al.kmax2 / al.kpre.
  bool Run(AttnLayer& al, std::size_t id, xrt::bo& qkv_c, std::uint32_t pos0,
           std::uint32_t n, std::uint32_t pass0, std::string* error);
  xrt::bo& PassBo(std::uint32_t i) { return pass_sub_[i]; }
  const std::uint8_t* PassHost(std::uint32_t i) {
    pass_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    return pass_.map<std::uint8_t*>() + std::size_t{i} * kPassBytes;
  }
  /// bf16 K and V of tokens [t0, t0 + n) decoded from the chunk layout (host
  /// history repair).
  static void ReadHist(const AttnLayer& al, std::uint32_t t0, std::uint32_t n,
                       std::uint16_t* k_out, std::uint16_t* v_out,
                       std::uint32_t kvd, std::uint32_t cap);
  double ms_kv{0}, ms_q{0}, ms_host{0};
  int calls{0};

private:
  struct LayerTab {
    xrt::bo bo;
    std::uint8_t* h{};
    std::uint32_t max_groups{0}, filled{0};
    std::vector<std::uint16_t> wq, wk;
    std::uint16_t eps{0};
    float theta{0};
    std::vector<float> fac;
  };
  bool ready_{false};
  xrt::device dev_;
  xrt::hw_context ctx_;
  xrt::kernel kern_;
  xrt::bo ib_kv_, ib_q_;
  std::size_t n_kv_{0}, n_q_{0};
  xrt::bo pass_, kn_;
  std::vector<xrt::bo> pass_sub_, pass_win_;
  std::vector<LayerTab> tabs_;
};

class NpuRuntime {
public:
  NpuRuntime() = default;
  /// Load device + xclbin + kernel. `xclbin_path` selects the built program.
  /// int4: the xclbin streams 4-bit weights (see g_decode_int4).
  bool Init(const std::string& xclbin_path, std::string* error,
            bool int4 = false);
  /// Programs of equal size share their A/C staging buffers (decision engines
  /// run layers strictly in order). `slot` keeps double-buffered pairs apart.
  void SetSharedScratch(bool on) { share_scratch_ = on; }
  void SetScratchSlot(int slot) { scratch_slot_ = slot; }
  /// A zeroed staging buffer; shared by every caller with the same size, group
  /// and slot while shared scratch is on.
  xrt::bo MakeScratchBo(std::size_t bytes, int group, int slot) {
    if (share_scratch_) {
      const auto key = std::make_tuple(bytes, std::size_t{0}, group, -1, slot);
      if (const auto it = scratch_pool_.find(key); it != scratch_pool_.end())
        return it->second.first;
      xrt::bo bo = MakeBo(bytes, group);
      std::memset(bo.map<void*>(), 0, bytes);
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      scratch_pool_[key] = {bo, bo};
      return bo;
    }
    xrt::bo bo = MakeBo(bytes, group);
    std::memset(bo.map<void*>(), 0, bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
  }

  xrt::device& device() { return dev_; }
  xrt::kernel& kernel() { return kern_; }

  /// Load an instruction stream file and create A/C BOs sized for (K,N).
  /// `c_override` allows custom C size (e.g. fused FFN combined buffer).
  /// `mrows` = A row-blocks per B tile (4 for m64 prefill streams).
  Program MakeProgram(std::uint32_t K, std::uint32_t N,
                      const std::string& insts_path, std::size_t c_override,
                      std::string* error, std::uint32_t a_extra_k = 0,
                      std::uint32_t mrows = 1, int a_group = 3,
                      int c_group = 3);

  /// Allocate and pack a weight matrix (N rows x K cols, row-major float).
  WeightBuffer PackWeight(const float* W, std::uint32_t N, std::uint32_t K);
  WeightBuffer PackWeightQ(const std::int8_t* q, const std::uint16_t* sc,
                           std::uint32_t N, std::uint32_t K);
  /// Cached variants: `fill` returns the N x K float matrix and runs only when
  /// the cache has no entry for `tag`.
  WeightBuffer PackWeightCached(WeightCache& cache, const std::string& tag,
                                std::uint32_t N, std::uint32_t K,
                                const std::function<const float*()>& fill);
  WeightBuffer PackWeightAtbCached(WeightCache& cache, const std::string& tag,
                                   std::uint32_t N, std::uint32_t K,
                                   int b_group,
                                   const std::function<const float*()>& fill);
  /// Zero-filled host-mapped weight buffer of `bytes` bytes (fused buffers a
  /// caller packs itself).
  WeightBuffer AllocWeightBo(std::size_t bytes, int group);
  /// Bind a weight buffer to a program (sets kernel args 0-5).
  void Attach(WeightBuffer& wb, Program& prog);
  /// WP10: this XRT/xdna stack keeps a program's FIRST weight buffer bound:
  /// later set_arg of another BO on the same program silently runs stale
  /// weights (bisected with sbtest-ffn64b: fresh-B bind + identical staged A =>
  /// deterministic garbage; in-place repacks of the pinned BO are always
  /// correct). WeightSlot pins wb on a program's first use and copies later
  /// weights into it, so every dispatch sees valid weights. Returns the buffer
  /// to dispatch with.
  WeightBuffer& WeightSlot(WeightBuffer& wb, Program& prog);
  /// Bind (insts, A, W, C) to an arbitrary run object (warm-up dispatches).
  void AttachTo(xrt::run& run, WeightBuffer& wb, Program& prog);
  /// Per-weight-buffer pooled run (slots are scarce on xdna; ~64 max).
  xrt::run& SlotRun(WeightBuffer& wb);

  void EncodeA(const float* x, std::uint32_t n, std::uint32_t K, Program& p);
  void DecodeC(float* y, std::uint32_t n, Program& p,
               const std::uint8_t* base = nullptr, std::uint32_t N = 0);

  struct Timings;  // defined below
  // --- AGENT-7/8 big-M prefill (ATB) path --------------------------------
  // Weights/outputs use the asymmetric-tile-buffering layout (see
  // atb_layout.hpp). The kernel computes C = A @ B with A = M x K and B = K x
  // N. Model weight tensors are [N rows][K cols] (same as PackWeight), so W is
  // the model layout and the B stream is built as B[k][n] = W[n][k]. C comes
  // back as ATB bfp16 and is decoded to M x N row-major float.
  /// Host-only pack of an N x K model-layout matrix into the ATB B stream (K *
  /// N / 8 * 9 bytes at dst).
  static void PackAtbHost(const float* W, std::uint32_t N, std::uint32_t K,
                          std::uint8_t* dst);
  WeightBuffer PackWeightAtb(const float* W, std::uint32_t N, std::uint32_t K,
                             int b_group = 3);

  /// AGENT-8: dispatch-only floor measurement: submit the program's run and
  /// wait for completion. Assumes A/B are already device-resident. Yields
  /// (dispatch + device compute) per call, excluding encode/sync/decode.
  void DispatchAtb(Program& p, WeightBuffer& wb) {
    WeightBuffer& w = WeightSlot(wb, p);
    w.run.start();
    w.run.wait();
  }

  /// AGENT-8 cfg29: re-point a program's A staging buffer at an external BO
  /// (the gu programs' C output) for the device-resident act-wire tap.
  void RebindA(Program& p, xrt::bo bo, xrt::bo bo2 = xrt::bo()) {
    p.a = bo;
    p.ah = static_cast<std::uint8_t*>(p.a.map<void*>());
    if (bo2) {
      p.a2 = bo2;
      p.four_a = true;
    }
  }

  /// AGENT-8: raw device buffer from this runtime's heap (tap-chain bring-up).
  xrt::bo MakeBo(std::size_t bytes, int group = 3) {
    return xrt::bo(dev_, bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(group));
  }
  /// Pack FFN gate+up interleaved weights (N = 2*na) for the ATB path.
  WeightBuffer PackFfnGuAtb(const float* gate, const float* up,
                            std::uint32_t na, std::uint32_t hidden,
                            int b_group = 3);
  /// Pack FFN down weights for the ATB path.
  WeightBuffer PackFfnDnAtb(const float* Wd, std::uint32_t hidden,
                            std::uint32_t INT, int b_group = 3);

private:
  bool share_scratch_{false};
  int scratch_slot_{0};
  std::map<std::tuple<std::size_t, std::size_t, int, int, int>,
           std::pair<xrt::bo, xrt::bo>>
      scratch_pool_;
  // Staging for the ATB A/C streams: the xrt BO mapping is write-combined, so
  // 9-byte block stores to it run at ~1 GB/s. Encode/decode in cached memory
  // and move the stream with one linear copy.
  std::vector<std::uint8_t> atb_a_scratch_, atb_c_scratch_;

public:
  /// valid_rows: only the 128-row tiles holding rows below it are
  /// encoded/decoded (the rest of the program's rows are padding).
  void EncodeAAtb(const float* x, std::uint32_t M, std::uint32_t K, Program& p,
                  std::uint32_t valid_rows = ~0u);
  void DecodeCAtb(float* y, std::uint32_t M, std::uint32_t N, Program& p,
                  std::uint32_t valid_rows = ~0u);
  /// Scratch-free variants (thread-safe): A stream into dst, C stream from src.
  void EncodeAAtbRaw(const float* x, std::uint32_t M, std::uint32_t K,
                     std::uint8_t* dst, std::uint32_t valid_rows = ~0u) const;
  void DecodeCAtbRaw(const std::uint8_t* src, float* y, std::uint32_t M,
                     std::uint32_t N, std::uint32_t valid_rows = ~0u) const;
  /// One output row (N floats) straight from the C wire.
  void DecodeCRowAtb(const Program& p, std::uint32_t N, std::uint32_t row,
                     float* out) const;
  /// Inverse of EncodeAAtb (reads p.ah).
  void DecodeAAtb(float* x, std::uint32_t M, std::uint32_t K,
                  const Program& p) const;
  /// C-wire encoder (inverse of DecodeCAtbRaw) for the device-resident residual
  /// stream.
  void EncodeCAtbRaw(const float* x, std::uint32_t M, std::uint32_t N,
                     std::uint8_t* dst) const;
  /// Program that carries only an instruction stream; the caller binds A/C.
  Program MakeBareProgram(const std::string& insts_path);
  /// Synchronous prefill GEMM: x (M x K) -> y (M x N), M = 16 * p.mrows.
  void GemmAtb(Program& p, WeightBuffer& wb, const float* x, std::uint32_t M,
               float* y, Timings* t = nullptr);

  // Synchronous GEMM: encode x (n x K), run, decode to y (n x N).
  // Updates timing accumulators.
  struct Timings {
    double enc{0}, syncA{0}, submit{0}, wait{0}, syncC{0}, dec{0};
    double attn_append{0}, attn_prep{0}, attn_wait{0}, attn_merge{0};
    int attn_calls{0};
    int calls{0};
  };
  void Gemm(Program& p, WeightBuffer& wb, const float* x, std::uint32_t n,
            float* y, Timings* t = nullptr);
  /// Split fused FFN: Start encodes and submits, Finish waits and decodes, so
  /// several programs can be in flight.
  void FfnStart(Program& p, WeightBuffer& wb, const float* x, std::uint32_t n,
                Timings* t = nullptr);
  void FfnFinish(Program& p, WeightBuffer& wb, float* y, std::uint32_t n,
                 Timings* t = nullptr, std::uint32_t gu_strips = 120,
                 std::uint32_t out_n = 2048);
  // Fused FFN: decode down-output from combined C buffer (offset after 120
  // strips).
  void Ffn(Program& p, WeightBuffer& wb, const float* x, std::uint32_t n,
           float* y, Timings* t = nullptr, std::uint32_t gu_strips = 120,
           std::uint32_t out_n = 2048);

  /// m64 prefill FFN: gu (swiglu flush) -> byte-shuffle into down's A -> plain
  /// down gemm.
  void Ffn64(WeightBuffer& wb_gu, Program& p_gu, WeightBuffer& wb_dn,
             Program& p_dn, const float* x, std::uint32_t n, float* y,
             Timings* t = nullptr, std::uint32_t taps = 120);

  /// Argmax GEMM over an int5 head: returns the greedy token id per row.
  /// C must have been produced by an argmax-mode program (per-strip reduced).
  void Argmax(Program& p, WeightBuffer& wb, const float* x, std::uint32_t n,
              std::uint32_t* out, Timings* t = nullptr);

  /// `w_in`: [3*hidden][hidden] rows ordered B | C | x (hidden each).
  /// `conv_w`: [hidden][taps] depthwise taps. `tmpl`: conv program template
  /// (shares instructions and the A BO; gets its own C BO).
  ConvLayer PackConv(const float* w_in, const float* conv_w, std::uint32_t taps,
                     const float* w_out, std::uint32_t hidden, Program& tmpl);
  void ConvStep(ConvLayer& cl, const float* z, std::uint32_t n, float* y,
                Timings* t = nullptr);
  /// Accept `k` of the `n` rows from the last ConvStep (history for next call).
  /// `settle` waits for the history write to become visible to the next
  /// dispatch (needed when one is issued immediately, i.e. inside an m64 step).
  void ConvCommit(ConvLayer& cl, std::uint32_t k, bool settle = false);
  void ConvReset(ConvLayer& cl);
  void ConvRefresh(ConvLayer& cl);
  /// History rows (the last taps-1 B*x rows, bf16, [6][hidden]) held in a conv
  /// layer's weight buffer.
  void ConvGetHistory(const ConvLayer& cl, std::uint16_t* rows6);
  void ConvSetHistory(ConvLayer& cl, const std::uint16_t* rows6);
  /// Copy the conv history of `src` (e.g. the 64-row program) into `dst` (the
  /// 16-row program).
  void ConvCopyHistory(ConvLayer& dst, const ConvLayer& src);

  // Attention on the NPU (4 kv heads x 4 q heads x 128 dims, up to 16 rows per
  // pass).
  AttnLayer MakeAttnLayer(std::uint32_t capacity_tokens,
                          std::uint32_t ring_tokens = 0,
                          std::uint32_t window = 0);
  void AttnReset(AttnLayer& al);
  /// Append rows [pos, pos+n) of k/v (n x 512, after norm+rope) to the device
  /// cache. `k_hist`/`v_hist` are the host bf16 caches ([slot][512], slot =
  /// token).
  void AttnAppend(AttnLayer& al, std::uint32_t pos, std::uint32_t n,
                  const float* k, const std::uint16_t* v_hist,
                  Timings* tm = nullptr);
  /// Same result for a 16-aligned pos and n a multiple of 16 on a full
  /// (non-ring) layer, parallel over chunks.
  void AttnAppendBulk(AttnLayer& al, std::uint32_t pos, std::uint32_t n,
                      const float* k, const std::uint16_t* v_hist,
                      Timings* tm = nullptr);
  /// q: n x 2048 (16 heads x 128, after norm+rope). Writes normalized attention
  /// output (n x 2048, before the output gate). Returns false if a row's
  /// softmax mass is outside the safe range (caller falls back to the CPU
  /// path).
  bool AttnRun(AttnLayer& al, std::uint32_t pos, std::uint32_t n,
               const float* q, const std::uint16_t* k_hist,
               const std::uint16_t* v_hist, const std::string& programs_dir,
               float* out, Timings* tm = nullptr);

private:
  xrt::device dev_;
  xrt::hw_context ctx_;
  xrt::kernel kern_;
  bool ready_{false};
  xrt::bo attn_a_, attn_c_;
  std::uint8_t* attn_ah_{};
  std::uint8_t* attn_ch_{};
  std::unordered_map<std::uint32_t, std::pair<xrt::bo, std::size_t>>
      attn_insts_;
};

/// Prefill attention: one pass covers up to 128 query rows (8 columns x 16
/// tokens); every core streams the causal prefix of its own rows, so nothing is
/// merged across columns.
class AttnPrefill {
public:
  /// packed: the v2 programs, where one core covers 16 << sg tokens (sg set by
  /// SetPack) and a pass up to 8 x that.
  bool Init(const std::string& dir, std::string* error, bool packed = false);
  bool ready() const { return ready_; }
  /// Query heads per kv head g (1, 2 or 4): g = 2 packs two token blocks into
  /// the rows the missing heads would pad.
  void SetPack(std::uint32_t g) { sg_ = packed_ && g == 2 ? 1u : 0u; }
  std::uint32_t PassTokens() const { return 128u << sg_; }
  /// Picks the exact-exp2 kernel (short contexts, no speed cost) or the
  /// hardware-exp2 kernel for later passes.
  void SetExact(bool exact) { exact_ = exact && have_exact_; }
  /// q: n x 2048 rows [pos0, pos0+n), n <= 128. Writes normalized attention (n
  /// x 2048, before the gate). Returns false when the pass cannot run on this
  /// program set or a row's softmax mass is unsafe.
  bool Run(AttnLayer& al, std::uint32_t pos0, std::uint32_t n, const float* q,
           const std::uint16_t* k_hist, const std::uint16_t* v_hist, float* out,
           NpuRuntime::Timings* tm);

  struct Pass {
    std::uint32_t pos0, n;
    const float* q;
    float* out;
  };
  /// Several independent passes of one layer (history already complete): host
  /// prep of pass i+1 and the merge of pass i overlap the device run of the
  /// neighbouring pass. Returns false if any pass cannot run. pre(i) (optional)
  /// runs before pass i is prepared; when it is set the prep of pass i+1 waits
  /// for pass i (ring layers append history per pass) and only the merge
  /// overlaps the device.
  struct DevPass {
    std::uint32_t pos0, n;
    float* out;
    std::uint32_t pass_idx;  // pass buffer index in AttnDevPrep
  };
  /// Passes whose Q, header, bounds, tails and K/V chunks were built on the
  /// device by AttnDevPrep.
  bool RunManyDev(AttnLayer& al, AttnDevPrep& dp, const DevPass* passes,
                  std::size_t np, NpuRuntime::Timings* tm);
  xrt::device& device() { return dev_; }
  /// When set, rows whose softmax mass is unusable do not fail a pass: they are
  /// listed in bad_rows (absolute token, q-head slot r*4+hq) and left for the
  /// caller to recompute.
  bool collect_bad{false};
  std::vector<std::pair<std::uint32_t, std::uint32_t>> bad_rows;
  /// Host reference of the A prefix of one pass (debug): the bytes Prep would
  /// send.
  bool DebugPrep(AttnLayer& al, std::uint32_t pos0, std::uint32_t n,
                 const float* q, const std::uint16_t* k_hist,
                 const std::uint16_t* v_hist, std::vector<std::uint8_t>* out);
  bool RunMany(AttnLayer& al, const Pass* passes, std::size_t np,
               const std::uint16_t* k_hist, const std::uint16_t* v_hist,
               NpuRuntime::Timings* tm,
               const std::function<void(std::size_t)>* pre = nullptr);

private:
  bool Prep(int slot, AttnLayer& al, std::uint32_t pos0, std::uint32_t n,
            const float* q, const std::uint16_t* k_hist,
            const std::uint16_t* v_hist, std::uint32_t* mc_out);
  bool Launch(int slot, AttnLayer& al, std::uint32_t mc, xrt::run** run_out);
  bool LaunchDev(int slot, AttnLayer& al, AttnDevPrep& dp,
                 std::uint32_t pass_idx, std::uint32_t mc, xrt::run** run_out);
  bool Merge(int slot, std::uint32_t n, float* out, std::uint32_t pos0 = 0);
  std::string dir_;
  xrt::device dev_;
  xrt::hw_context ctx_;
  xrt::kernel kern_;
  xrt::hw_context ctx_x_;
  xrt::kernel kern_x_;
  bool exact_{false}, have_exact_{false};
  bool ready_{false}, packed_{false};
  std::uint32_t sg_{0}, nt_{3};
  std::size_t pcol_{std::size_t{16} * 4 * 1536};
  std::string tag_;
  xrt::bo a_[2], c_[2];
  std::uint8_t* ah_[2]{};
  std::uint8_t* ch_[2]{};
  std::unordered_map<std::uint32_t, std::pair<xrt::bo, std::size_t>> insts_;
  std::map<std::tuple<const AttnLayer*, std::uint32_t, int>, xrt::run> runs_;
};

}  // namespace gufo::npu::runtime

#endif  // GUFO_NPU_RUNTIME_HPP_
