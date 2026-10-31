#include "src/models/laguna/gpu/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/gpu/device_model.hpp"
#include "src/models/laguna/gpu/kernels/kernels.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::laguna::gpu {
namespace {

constexpr std::uint32_t kVecBatch = 8;
// DFlash verify batches are n<=Session::kMaxVerifyRows (16) tokens, used=10
// -> rows up to 160. The underlying qfn_mmq_moe_vec/mul_mat_vec_moe_gated
// kernels chunk larger batches internally (mmvq_moe_max_batch groups) and
// assert tokens<=MMVQ_MAX_ROUTED_BATCH(64) - well above 16 - so this was a
// too-conservative proxy for "fits the vector path", tuned for n=1 decode
// only. Raised to let verify's small batches skip the wide/RoutedExperts
// path's host-sync overhead (see laguna-port-state.md); TEST EMPIRICALLY
// before trusting this further if expert_used or weight types change.
constexpr std::uint32_t kVecExpertRows = 160;
constexpr std::uint32_t kMaxTopK = 16;
constexpr std::uint32_t kMaxExperts = 256;
constexpr std::uint32_t kRoutePadRows = 16;
constexpr std::uint32_t kPairTileRows = 128;
constexpr std::uint32_t kPlainTileRows = 48;
// 2026-10-30: experimenting with a 128-row Q8_0 down-projection tile via the
// additive RoutedF16GemmQ8Wide (bypasses RoutedF16Gemm's tile_rows<=64 cap
// for Q8_0, which was tuned on Flash-Next's shapes, not Laguna's). Toggle
// here; see laguna-port-state.md for the A/B result.
constexpr bool kQ8WideDownTile = false;
constexpr std::uint32_t kQ8TileRows = kQ8WideDownTile ? 128 : 64;
constexpr std::uint32_t kRoutedTileMin = 16;
constexpr std::uint32_t kRoutedTileMax = 80;
constexpr std::uint32_t kRoutedTileStep = 16;
constexpr std::uint32_t kWarmTokens = 64;
// 2026-10-30: split-K decode attention (see kernels.hpp's AttentionSplit).
// Narrow decode/verify batches at depth otherwise sweep the whole KV range in
// one block per (head, token), so TG falls off linearly with context. 8 splits
// mirrors Flash-Next's kAttnSplits; only non-SWA layers need it (SWA is capped
// at sliding_window=512 and already flat). The depth threshold keeps shallow
// decode on the unchanged WMMA path, where the extra merge launch is pure cost.
constexpr std::uint32_t kSplitAttnSplits = 64;
constexpr std::uint32_t kSplitAttnMinDepth = 2048;
// Block-sparse attention (LAGUNA_SPARSE_ATTN): candidate older blocks beyond
// the sink + local window are kept exactly whenever there are at most this
// many of them; above that, only the top-scoring ones (Quest bound) survive.
// Same default as 27B/35B's GUFO_QWEN_SPARSE_TOPK; LAGUNA_SPARSE_TOPK
// overrides it.
constexpr std::uint32_t kSparseTopBlocksDefault = 256;
std::uint32_t SparseTopBlocks() {
  static const std::uint32_t value = [] {
    if (const char* v = std::getenv("LAGUNA_SPARSE_TOPK"))
      return static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    return kSparseTopBlocksDefault;
  }();
  return value;
}
// Routed-expert activations are stored divided by this (F16 range); the
// combine multiplies it back.
constexpr float kRoutedActScale = 64.0F;
// 2026-10-31: NPU shared-expert/QKV offload (prefill PP only). Fixed M=4096
// (no runtime-M sweep): engages only when n == npu->tokens() exactly
// (matching Flash-Next's own `n_tokens % npu->tokens() == 0` dispatch gate,
// specialized to equality since Laguna has no multi-group support) - a
// looser `n>=1024` admission measured as a real ~4% PP regression on
// non-aligned chunks. See laguna-port-state.md.

using core::GgmlType;

models::QwenTensorRef TensorRef(const DeviceTensor& t) {
  models::QwenTensorRef ref;
  ref.data = t.data;
  ref.type = t.type;
  ref.num_elements = static_cast<std::size_t>(t.cols) * t.rows;
  return ref;
}

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr)
    *error_msg = message;
  return false;
}

#define LAGUNA_HIP(call)                                                   \
  do {                                                                     \
    const hipError_t status_ = (call);                                     \
    if (status_ != hipSuccess) {                                           \
      return Fail(error_msg,                                               \
                  std::string(#call) + ": " + hipGetErrorString(status_)); \
    }                                                                      \
  } while (false)

template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  bool Allocate(std::size_t count) {
    Reset();
    if (hipMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) !=
        hipSuccess) {
      ptr_ = nullptr;
      return false;
    }
    count_ = count;
    return true;
  }
  void Reset() {
    if (ptr_ != nullptr)
      (void)hipFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }
  T* get() const noexcept { return ptr_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
  T* ptr_{nullptr};
  std::size_t count_{0};
};

int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1U));
  for (std::uint32_t cols = kRoutedTileMin; cols < kRoutedTileMax;
       cols += kRoutedTileStep) {
    if (cols >= 2 * mean)
      return static_cast<int>(cols);
  }
  return static_cast<int>(kRoutedTileMax);
}

bool DenseVecType(GgmlType type) {
  return type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
         type == GgmlType::kQ5_1 || type == GgmlType::kQ6_K;
}

std::uint32_t TileCapacityFor(std::size_t slots, std::uint32_t experts) {
  return static_cast<std::uint32_t>(
      (slots + std::size_t{experts} * (kRoutePadRows - 1)) / kRoutePadRows +
      experts + 1);
}

}  // namespace

struct ModelState {
  EngineOptions options;
  std::shared_ptr<core::GgufReader> reader;
  Config config;
  std::unique_ptr<DeviceModel> device;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
  std::vector<std::uint32_t> dflash_taps;
  std::uint32_t dflash_window{0};
  std::uint32_t drafter_block{0};
};

struct Session::Impl {
  std::shared_ptr<ModelState> model;
  hipStream_t stream{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t tile{0};
  std::uint32_t position{0};
  std::uint32_t logit_rows{1};
  float* h_logits{nullptr};
  std::int32_t* h_tokens{nullptr};
  std::uint32_t* h_pos{nullptr};
  std::int32_t* h_tiles{nullptr};
  __half* xh{nullptr};
  hipEvent_t ready{nullptr};
  std::uint32_t* h_counts{nullptr};

  std::vector<DeviceBuffer<__half>> k_cache, v_cache;
  std::vector<std::uint32_t> ring;
  // Block-sparse attention (LAGUNA_SPARSE_ATTN): per-64-key-block channel
  // min/max pooled keys, allocated only for the non-SWA (group-6,
  // il%4==0) layers - the SWA layers are already bounded by their ring
  // buffer and never use this. One shared buffer per layer (not banked:
  // the underlying k_cache isn't banked either, and EnqueueBanked runs
  // bank 0's full Attention() before bank 1's on the same stream, so a
  // single valid_end tracked on the host is safe - see Attention()).
  std::vector<DeviceBuffer<__half>> sparse_pooled;
  std::vector<std::uint32_t> sparse_valid_end;
  std::uint32_t sparse_max_blocks{0};
  std::unique_ptr<hip::QwenDFlashGpuExecutor> dflash;
  DeviceBuffer<float> tap_capture;
  std::uint32_t tap_pending{0}, tap_pending_pos{0}, tap_pending_first{0};
  bool inject_taps{true};
  DeviceBuffer<std::int32_t> tokens;
  DeviceBuffer<std::uint32_t> pos;
  // x/xn hold the residual stream and its normed copy; q/k/v/gate/attn/o are
  // transient per (layer, bank) scratch. Banked prefill (two independent
  // halves of one large chunk, NPU work of one half hidden behind the
  // other's GPU work - see EnqueueBanked) needs TWO persistent copies of
  // x/xn (each bank's residual stream must survive across all layers while
  // the other bank's work interleaves with it), but q/k/v/... are written
  // and fully consumed within one (layer, bank) step, so one shared copy
  // reused sequentially per bank is safe - HIP executes one stream's
  // kernels in launch order, so bank 0's reads always complete before bank
  // 1's writes to the same buffer even though both are "in flight" from the
  // host's point of view. `cx`/`cxn` (set by SetBank) are what every
  // per-layer helper (Attention/Moe/DenseFfn/RoutedExperts) actually reads
  // and writes; `x`/`xn` are bank 0's storage, `x2`/`xn2` bank 1's.
  DeviceBuffer<float> x, xn, x2, xn2, q, k, v, gate, attn, o, router;
  // Partial online-softmax states for the split-K decode attention path,
  // indexed [token][head][split][head_dim + 2] (acc then m, l).
  DeviceBuffer<float> attn_partials;
  DeviceBuffer<float> gate_e, up_e, sh_up, sh_out;
  // down_e/sh_gate/topk/expert_scale carry state from MoeRoute's return to
  // MoeFinish's start - EnqueueBanked calls MoeRoute for BOTH banks before
  // either bank's MoeFinish (that's the deliberate point of banking, to
  // hide one bank's NPU wait behind the other's RoutedExperts), so a
  // single shared buffer gets bank 1's MoeRoute overwriting bank 0's data
  // before MoeFinish(bank 0) ever reads it - confirmed as the actual cause
  // of banking's garbage-output bug (2026-10-31). Every OTHER per-layer
  // scratch buffer (q/k/v/gate/attn/o/router/ids/counts/routed-expert
  // tiles/gate_e/up_e/sh_up/sh_out/xh) is fully written-then-consumed
  // within a single MoeRoute or MoeFinish call for one bank, so reusing it
  // sequentially is still safe (HIP's in-order stream execution).
  DeviceBuffer<float> down_e[2], sh_gate[2], topk[2], expert_scale[2];
  DeviceBuffer<float> expert_const, dense_gate, dense_up, d_logits;
  DeviceBuffer<float> last;
  DeviceBuffer<std::int32_t> ids;
  DeviceBuffer<std::uint32_t> counts;
  DeviceBuffer<std::int32_t> route_bounds, route_cursors, rows_token, rows_slot;
  DeviceBuffer<std::int32_t> tiles_pair, tiles_down;
  DeviceBuffer<__half> act_half, qh;

  // Caches the MMQ-tile quantization of the last Dense() input so repeated
  // calls against the same activation (Q/K/V off attn_norm's output, or
  // gate/up off ffn_norm's) quantize once instead of once per projection.
  DeviceBuffer<char> quant_cache;
  const float* quant_cache_src{nullptr};
  std::uint32_t quant_cache_n{0}, quant_cache_k{0};
  bool Quantized(const float* in, std::uint32_t n, std::uint32_t k);
  // Same caching idea, for Dense()'s Q8_0 small-batch (n<=8) vec path -
  // Q/K/V off attn_norm's output (or gate/up off ffn_norm's) share one
  // input pointer, so quantize once per activation instead of once per
  // projection (same win Quantized() already gets on the wide path).
  DeviceBuffer<char> q8_vec_quant;
  const float* q8_vec_quant_src{nullptr};
  std::uint32_t q8_vec_quant_n{0}, q8_vec_quant_k{0};
  bool QuantizedVec(const float* in, std::uint32_t n, std::uint32_t k);
  // Caches QuantizeQ8Tiled's output for Dense()'s W8A8Gemm experiment
  // (2026-10-30): same dedup idea, for the int8-WMMA wide-batch path.
  DeviceBuffer<char> q8_tiled_quant;
  const float* q8_tiled_quant_src{nullptr};
  std::uint32_t q8_tiled_quant_n{0}, q8_tiled_quant_k{0};
  bool QuantizedTiled(const float* in, std::uint32_t n, std::uint32_t k);

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
    if (xh != nullptr)
      (void)hipFree(xh);
    if (ready != nullptr)
      (void)hipEventDestroy(ready);
    if (h_logits != nullptr)
      (void)hipHostFree(h_logits);
    if (h_tokens != nullptr)
      (void)hipHostFree(h_tokens);
    if (h_pos != nullptr)
      (void)hipHostFree(h_pos);
    if (h_tiles != nullptr)
      (void)hipHostFree(h_tiles);
    if (h_counts != nullptr)
      (void)hipHostFree(h_counts);
    for (auto& g : graphs) {
      if (g.exec != nullptr)
        (void)hipGraphExecDestroy(g.exec);
    }
  }

  bool Init(std::uint32_t max_context, std::string* error_msg);
  bool Dense(const DeviceTensor& w, const float* in, float* out,
             std::uint32_t n, std::string* error_msg);
  bool Experts(const DeviceTensor& w, const float* in, float* out,
               std::uint32_t rows, std::uint32_t used, std::uint32_t n_tokens,
               std::string* error_msg);
  bool Attention(std::uint32_t il, std::uint32_t n, std::string* error_msg);
  bool DenseFfn(const DeviceLayer& layer, std::uint32_t n,
                std::string* error_msg);
  bool Moe(const DeviceLayer& layer, std::uint32_t n, std::string* error_msg);
  bool MoeRoute(const DeviceLayer& layer, std::uint32_t n, int bank,
                bool* wide_out, bool* scaled_out, std::string* error_msg);
  bool MoeFinish(const DeviceLayer& layer, std::uint32_t n, int bank, bool wide,
                 bool scaled, std::string* error_msg);
  bool RoutedExperts(const DeviceLayer& layer, std::uint32_t n, int bank,
                     bool* scaled, std::string* error_msg);
  bool Forward(std::span<const std::int32_t> tokens, bool all_rows,
               std::uint32_t prefix_end, std::string* error_msg);
  bool Enqueue(std::uint32_t n, bool all_rows, std::uint32_t prefix_end,
               std::uint32_t* tap_rows_out, std::uint32_t* tap_first_out,
               std::string* error_msg);
  bool DflashInject(std::uint32_t rows, std::string* error_msg);

  // One captured HIP graph per decode/verify width (1..kMaxVerifyRows),
  // mirroring Flint's pattern: the whole per-step kernel sequence launches
  // once from host code, then replays as a single `hipGraphLaunch` on every
  // later call at that width - cuts per-kernel launch overhead, the
  // dominant cost at n=1..16. Capture only applies when `all_rows` and
  // `n<=kMaxVerifyRows` (the decode/verify path, not full prefill chunks),
  // and only once the DFlash tap-capture window has reached its steady
  // state for that width (tap_rows==n - true for plain decode always,
  // since dflash==nullptr there, and true for DFlash verify once position
  // has advanced past the window-fill edge case) - capturing while
  // tap_first/tap_rows are still settling would bake a stale, position-
  // dependent slice into the graph.
  struct StepGraph {
    hipGraphExec_t exec{nullptr};
    bool failed{false};
    std::uint32_t steps{0};
  };
  StepGraph graphs[Session::kMaxVerifyRows + 1];

  // cx/cxn/cbank select which bank's residual-stream buffers every
  // per-layer helper reads and writes; SetBank repoints them. Bank 0 is
  // the default/only bank outside EnqueueBanked.
  float* cx{nullptr};
  float* cxn{nullptr};
  int cbank{0};
  void SetBank(int bank) {
    cx = bank != 0 ? x2.get() : x.get();
    cxn = bank != 0 ? xn2.get() : xn.get();
    cbank = bank;
  }

  // NPU shared-expert offload state (prefill only), one slot per bank so
  // both banks can have a submission in flight at once (EnqueueBanked hides
  // bank 0's NPU wait behind bank 1's RoutedExperts and vice versa - see
  // its comment). `sh_a[b][0]/[1]` are the hi/residual A-operand encodes of
  // bank b's post-ffn_norm activation; `sh_c[b][0..3]` are hi-gate, hi-up,
  // lo-gate, lo-up C results. `sh_pending[b]` is set by ShexpSubmit and
  // consumed (Wait + combine) by the caller once other GPU work has had a
  // chance to overlap it. Outside banking, only slot 0 is ever used.
  bool npu_ready{false};
  int sh_a[2][2]{{-1, -1}, {-1, -1}};
  int sh_c[2][4]{{-1, -1, -1, -1}, {-1, -1, -1, -1}};
  std::uint64_t sh_ticket[2]{0, 0};
  bool sh_pending[2]{false, false};
  bool ShexpSubmit(const DeviceLayer& layer, std::uint32_t n, int bank,
                   std::string* error_msg);
  // QKV offload (same encode shared across Q/K/V - one activation).
  // kNpuMaxQkvSlices (11 = 9 Q + 1 K + 1 V, the widest, 72-head layer) is
  // the COMBINED cap across all three tensors in one submission -
  // `rocm::kNpuMaxSlices` (9) is only the per-tensor cap `NpuStreams` uses
  // for decode, a DIFFERENT, smaller limit. Checking the combined count
  // against the per-tensor 9 (as this used to) silently rejected every
  // 72-head layer (9+1+1=11 > 9), so QKV only ever ran on 48-head layers -
  // a real bug, found via the advisor, that invalidated every earlier QKV
  // measurement this session (it was running on far fewer layers than the
  // "QKV of 48 layers" staging message implied). One slot per bank, cross-
  // layer submitted (see EnqueueBanked): layer L's MoeFinish submits layer
  // L+1's QKV using bank b's just-updated residual stream, consumed at the
  // start of layer L+1's Attention(bank b) - hidden behind the OTHER
  // bank's remaining work for layer L, not a same-call submit-then-wait.
  static constexpr std::uint32_t kNpuMaxQkvSlices = 11;
  int q_a[2][2]{{-1, -1}, {-1, -1}};
  int q_c[2][2][kNpuMaxQkvSlices];
  std::uint32_t q_slices[2]{0, 0};
  std::uint64_t q_ticket[2]{0, 0};
  bool q_pending[2]{false, false};
  bool QkvSubmit(const DeviceLayer& layer, std::uint32_t n, int bank,
                 std::string* error_msg);
  bool QkvConsume(const DeviceLayer& layer, std::uint32_t heads, int bank,
                  std::string* error_msg);
  bool EnqueueBanked(std::span<const std::int32_t> input,
                     std::string* error_msg);
};

bool Session::Impl::Init(std::uint32_t max_context, std::string* error_msg) {
  const Config& c = model->config;
  if (c.expert_used > kMaxTopK || c.expert_count > kMaxExperts) {
    return Fail(error_msg, "Laguna: expert counts exceed the router limit");
  }
  capacity = max_context;
  tile = std::max<std::uint32_t>(1, model->options.prefill_tile);
  LAGUNA_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  LAGUNA_HIP(hipEventCreateWithFlags(&ready, hipEventDisableTiming));
  if (qfn_mmq_init(0) != 0)
    return Fail(error_msg, "mmq initialization failed");

  const std::size_t kv_row = static_cast<std::size_t>(c.num_kv) * c.head_dim;
  const std::uint32_t max_heads =
      *std::max_element(c.heads.begin(), c.heads.end());
  const std::size_t q_row = static_cast<std::size_t>(max_heads) * c.head_dim;
  const std::size_t slots = static_cast<std::size_t>(tile) * c.expert_used;
  k_cache.resize(c.num_layers);
  v_cache.resize(c.num_layers);
  ring.assign(c.num_layers, 0);
  bool ok = true;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    std::uint32_t rows = capacity;
    if (c.IsSwa(il)) {
      // Sized for 2*tile, not tile: EnqueueBanked writes a full tile's new
      // rows for bank 0 THEN another tile's worth for bank 1 within one
      // call, into the SAME ring buffer (banks are not KV-cache-separated,
      // only the residual stream is - see EnqueueBanked's comment). Sizing
      // this for one tile only (as it was before 2026-10-31's banking
      // work) let bank 1's writes wrap around and overwrite rows bank 1's
      // own sliding-window attention still needed from bank 0 - the actual
      // cause of banking's confirmed first-token-divergence bug. Always
      // sized for 2*tile (not just when banking is enabled) since it costs
      // little extra memory and removes the need to track whether banking
      // might run later.
      ring[il] = c.sliding_window + 2 * tile;
      rows = std::min(capacity, ring[il]);
      if (rows == capacity)
        ring[il] = 0;
    }
    ok = ok && k_cache[il].Allocate(kv_row * rows) &&
         v_cache[il].Allocate(kv_row * rows);
  }
  const bool sparse_attn = std::getenv("LAGUNA_SPARSE_ATTN") != nullptr;
  sparse_pooled.resize(c.num_layers);
  sparse_valid_end.assign(c.num_layers, 0);
  if (sparse_attn) {
    sparse_max_blocks = (capacity + 63) / 64;
    for (std::uint32_t il = 0; il < c.num_layers && ok; ++il) {
      if (!c.IsSwa(il)) {
        ok = sparse_pooled[il].Allocate(std::size_t{c.num_kv} *
                                        sparse_max_blocks * 256);
      }
    }
  }
  ok = ok && tokens.Allocate(tile) && pos.Allocate(4) &&
       x.Allocate(std::size_t{tile} * c.hidden_size) &&
       xn.Allocate(std::size_t{tile} * c.hidden_size) &&
       x2.Allocate(std::size_t{tile} * c.hidden_size) &&
       xn2.Allocate(std::size_t{tile} * c.hidden_size) &&
       q.Allocate(std::size_t{tile} * q_row) &&
       qh.Allocate(std::size_t{tile} * q_row) &&
       k.Allocate(std::size_t{tile} * kv_row) &&
       v.Allocate(std::size_t{tile} * kv_row) &&
       gate.Allocate(std::size_t{tile} * max_heads) &&
       attn.Allocate(std::size_t{tile} * q_row) &&
       attn_partials.Allocate(std::size_t{kMaxVerifyRows} * max_heads *
                              kSplitAttnSplits * (c.head_dim + 2)) &&
       o.Allocate(std::size_t{tile} * c.hidden_size) &&
       router.Allocate(std::size_t{tile} * c.expert_count) &&
       gate_e.Allocate(slots * c.expert_ff) &&
       up_e.Allocate(slots * c.expert_ff) &&
       down_e[0].Allocate(slots * c.hidden_size) &&
       down_e[1].Allocate(slots * c.hidden_size) && topk[0].Allocate(slots) &&
       topk[1].Allocate(slots) && ids.Allocate(slots) &&
       counts.Allocate(c.expert_count) &&
       route_bounds.Allocate(kMaxExperts + 1) &&
       route_cursors.Allocate(kMaxExperts) &&
       rows_token.Allocate(
           qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       rows_slot.Allocate(
           qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       tiles_pair.Allocate(TileCapacityFor(slots, c.expert_count)) &&
       tiles_down.Allocate(TileCapacityFor(slots, c.expert_count)) &&
       act_half.Allocate(slots * c.expert_ff) &&
       expert_scale[0].Allocate(slots) && expert_scale[1].Allocate(slots) &&
       expert_const.Allocate(slots) &&
       sh_gate[0].Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_gate[1].Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_up.Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_out.Allocate(std::size_t{tile} * c.hidden_size) &&
       dense_gate.Allocate(std::size_t{tile} * c.dense_ff) &&
       dense_up.Allocate(std::size_t{tile} * c.dense_ff) &&
       last.Allocate(std::size_t{kMaxVerifyRows} * c.hidden_size) &&
       d_logits.Allocate(std::size_t{kMaxVerifyRows} * c.vocab_size);
  ok = ok &&
       hipMalloc(reinterpret_cast<void**>(&xh),
                 std::size_t{tile} * c.hidden_size * sizeof(__half)) ==
           hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_logits),
                     std::size_t{kMaxVerifyRows} * c.vocab_size *
                         sizeof(float)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tokens),
                     std::size_t{tile} * sizeof(std::int32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_pos),
                     4 * sizeof(std::uint32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_counts),
                     kMaxExperts * sizeof(std::uint32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tiles),
                     2 * std::size_t{TileCapacityFor(slots, c.expert_count)} *
                         sizeof(std::int32_t)) == hipSuccess;
  if (!ok)
    return Fail(error_msg, "Laguna: session allocation failed");
  {
    const std::vector<float> efill(slots, kRoutedActScale);
    LAGUNA_HIP(hipMemcpy(expert_const.get(), efill.data(),
                         efill.size() * sizeof(float), hipMemcpyHostToDevice));
  }
  std::fill(h_logits, h_logits + std::size_t{kMaxVerifyRows} * c.vocab_size,
            0.0F);
  if (auto* npu = model->device->npu(); npu != nullptr) {
    std::string npu_error;
    const auto new_buf = [&](std::size_t bytes) {
      const int id = npu->NewBuffer(bytes, &npu_error);
      return id;
    };
    // One shexp slot set per bank (EnqueueBanked needs both banks' NPU
    // submissions in flight at once); outside banking only slot 0 is used.
    npu_ready = true;
    for (int b = 0; b < 2 && npu_ready; ++b) {
      const int a0 = new_buf(npu->ABytes());
      const int a1 = new_buf(npu->ABytes());
      const int c0 = new_buf(npu->CBytes());
      const int c1 = new_buf(npu->CBytes());
      const int c2 = new_buf(npu->CBytes());
      const int c3 = new_buf(npu->CBytes());
      npu_ready =
          a0 >= 0 && a1 >= 0 && c0 >= 0 && c1 >= 0 && c2 >= 0 && c3 >= 0;
      if (npu_ready) {
        sh_a[b][0] = a0;
        sh_a[b][1] = a1;
        sh_c[b][0] = c0;
        sh_c[b][1] = c1;
        sh_c[b][2] = c2;
        sh_c[b][3] = c3;
      }
    }
    if (!npu_ready) {
      std::fprintf(stderr, "NPU session buffers unavailable: %s\n",
                   npu_error.c_str());
    } else {
      bool qkv_ok = true;
      int qa[2][2];
      int qc[2][2][kNpuMaxQkvSlices];
      for (int b = 0; b < 2 && qkv_ok; ++b) {
        qa[b][0] = new_buf(npu->ABytes());
        qa[b][1] = new_buf(npu->ABytes());
        qkv_ok = qa[b][0] >= 0 && qa[b][1] >= 0;
        for (std::uint32_t s = 0; s < kNpuMaxQkvSlices && qkv_ok; ++s) {
          qc[b][0][s] = new_buf(npu->CBytes());
          qc[b][1][s] = new_buf(npu->CBytes());
          qkv_ok = qc[b][0][s] >= 0 && qc[b][1][s] >= 0;
        }
      }
      if (qkv_ok) {
        std::memcpy(q_a, qa, sizeof(qa));
        std::memcpy(q_c, qc, sizeof(qc));
      } else {
        std::fprintf(stderr, "NPU QKV buffers unavailable: %s\n",
                     npu_error.c_str());
      }
    }
  }
  SetBank(0);
  return true;
}

bool Session::Impl::Quantized(const float* in, std::uint32_t n,
                              std::uint32_t k) {
  if (quant_cache_src == in && quant_cache_n == n && quant_cache_k == k)
    return true;
  const std::size_t bytes =
      qfn_mmq_q8_1_mmq_bytes(static_cast<int>(n), static_cast<int>(k));
  if (quant_cache.bytes() < bytes && !quant_cache.Allocate(bytes))
    return false;
  if (qfn_mmq_quantize_q8_1_mmq(in, quant_cache.get(), static_cast<int>(n),
                                static_cast<int>(k), stream) != 0) {
    return false;
  }
  quant_cache_src = in;
  quant_cache_n = n;
  quant_cache_k = k;
  return true;
}

bool Session::Impl::QuantizedVec(const float* in, std::uint32_t n,
                                 std::uint32_t k) {
  if (q8_vec_quant_src == in && q8_vec_quant_n == n && q8_vec_quant_k == k)
    return true;
  const std::size_t bytes =
      qfn_mmq_q8_1_bytes(static_cast<int>(n), static_cast<int>(k));
  if (q8_vec_quant.bytes() < bytes && !q8_vec_quant.Allocate(bytes))
    return false;
  if (qfn_mmq_quantize_q8_1(in, q8_vec_quant.get(), static_cast<int>(n),
                            static_cast<int>(k), stream) != 0) {
    return false;
  }
  q8_vec_quant_src = in;
  q8_vec_quant_n = n;
  q8_vec_quant_k = k;
  return true;
}

bool Session::Impl::QuantizedTiled(const float* in, std::uint32_t n,
                                   std::uint32_t k) {
  if (q8_tiled_quant_src == in && q8_tiled_quant_n == n &&
      q8_tiled_quant_k == k)
    return true;
  const std::size_t bytes = qwen38_flash_next::rocm::Q8TiledBytes(n, k);
  if (q8_tiled_quant.bytes() < bytes && !q8_tiled_quant.Allocate(bytes))
    return false;
  qwen38_flash_next::rocm::QuantizeQ8Tiled(in, q8_tiled_quant.get(), n, k,
                                           stream);
  q8_tiled_quant_src = in;
  q8_tiled_quant_n = n;
  q8_tiled_quant_k = k;
  return true;
}

bool Session::Impl::ShexpSubmit(const DeviceLayer& layer, std::uint32_t n,
                                int bank, std::string* error_msg) {
  sh_pending[bank] = false;
  if (std::getenv("LAGUNA_NPU_NO_SHEXP") != nullptr)
    return true;
  auto* npu = model->device->npu();
  const Config& c = model->config;
  // Flash-Next's own NPU dispatch (executor.cpp's MoeExperts) requires
  // `n_tokens % npu->tokens() == 0` before engaging at all - Laguna has no
  // multi-group support (no runtime-M sweep), so the equivalent is exact
  // equality. Engaging for any smaller `n` (this file's n<=8 and n<4096
  // tail chunks were previously admitted via `n>=kNpuMinTokens`) submits a
  // fixed 4096-row program for real work far smaller than that, which
  // measured as a real ~4% PP regression at n=1024 - see
  // laguna-port-state.md. Requiring exact equality removes that case
  // entirely rather than tuning around it.
  if (!npu_ready || npu == nullptr || !npu->Healthy() ||
      layer.npu_shexp.size() != 2 || n != npu->tokens()) {
    return true;
  }
  // `cxn` holds this (layer, bank)'s post-ffn_norm activation, sized
  // tile*hidden_size - tile is Laguna's prefill chunk size, which the NPU
  // config was created to match exactly (see device_model.cpp's Upload()).
  const std::uint32_t enc_tokens = npu->tokens();
  rocm::NpuEncodeBoth(cxn, c.hidden_size, enc_tokens, c.hidden_size,
                      npu->Device(sh_a[bank][0]), npu->Device(sh_a[bank][1]),
                      stream);
  npu->Submit(stream, sh_a[bank][0], layer.npu_shexp.data(), sh_c[bank], 2, 0);
  npu->Submit(stream, sh_a[bank][1], layer.npu_shexp.data(), sh_c[bank] + 2, 2,
              0);
  sh_ticket[bank] = npu->Ticket();
  sh_pending[bank] = true;
  (void)error_msg;
  return true;
}

bool Session::Impl::QkvSubmit(const DeviceLayer& layer, std::uint32_t n,
                              int bank, std::string* error_msg) {
  q_pending[bank] = false;
  // Opt-in via LAGUNA_NPU_QKV=1: only called from EnqueueBanked's banked
  // path now (cross-layer: layer L's MoeFinish submits layer L+1's QKV,
  // consumed at the start of L+1's Attention(bank) - hidden behind the
  // OTHER bank's remaining layer-L work, not a same-call submit-then-
  // wait). See the kNpuMaxQkvSlices comment above for the slice-cap bug
  // this also fixes (was checked against the wrong, smaller limit,
  // silently skipping every 72-head layer).
  if (std::getenv("LAGUNA_NPU_QKV") == nullptr)
    return true;
  auto* npu = model->device->npu();
  const Config& c = model->config;
  if (!npu_ready || npu == nullptr || !npu->Healthy() || layer.npu_q.empty() ||
      layer.npu_k.empty() || layer.npu_v.empty() || n != npu->tokens()) {
    return true;
  }
  const std::uint32_t slices = static_cast<std::uint32_t>(
      layer.npu_q.size() + layer.npu_k.size() + layer.npu_v.size());
  if (slices > kNpuMaxQkvSlices)
    return true;
  const std::uint32_t enc_tokens = npu->tokens();
  rocm::NpuEncodeBoth(cxn, c.hidden_size, enc_tokens, c.hidden_size,
                      npu->Device(q_a[bank][0]), npu->Device(q_a[bank][1]),
                      stream);
  // One combined weight/id list (Q chunks, then K, then V) lets a single
  // Submit per hi/lo cover the whole QKV projection in one NPU round trip.
  std::vector<int> b_ids;
  b_ids.reserve(slices);
  b_ids.insert(b_ids.end(), layer.npu_q.begin(), layer.npu_q.end());
  b_ids.insert(b_ids.end(), layer.npu_k.begin(), layer.npu_k.end());
  b_ids.insert(b_ids.end(), layer.npu_v.begin(), layer.npu_v.end());
  npu->Submit(stream, q_a[bank][0], b_ids.data(), q_c[bank][0],
              static_cast<int>(slices), 0);
  npu->Submit(stream, q_a[bank][1], b_ids.data(), q_c[bank][1],
              static_cast<int>(slices), 0);
  q_slices[bank] = slices;
  q_ticket[bank] = npu->Ticket();
  q_pending[bank] = true;
  (void)error_msg;
  return true;
}

bool Session::Impl::QkvConsume(const DeviceLayer& layer, std::uint32_t heads,
                               int bank, std::string* error_msg) {
  q_pending[bank] = false;
  auto* npu = model->device->npu();
  npu->Wait(stream, q_ticket[bank]);
  const std::uint32_t n_q = static_cast<std::uint32_t>(layer.npu_q.size());
  const std::uint32_t n_k = static_cast<std::uint32_t>(layer.npu_k.size());
  const std::uint32_t n_v = static_cast<std::uint32_t>(layer.npu_v.size());
  const Config& c = model->config;
  const std::uint32_t kv_row = c.num_kv * c.head_dim;
  rocm::NpuStreams sq{}, sk{}, sv{};
  for (std::uint32_t i = 0; i < n_q; ++i) {
    sq.hi[i] =
        static_cast<const std::uint8_t*>(npu->Staged(stream, q_c[bank][0][i]));
    sq.lo[i] =
        static_cast<const std::uint8_t*>(npu->Staged(stream, q_c[bank][1][i]));
  }
  for (std::uint32_t i = 0; i < n_k; ++i) {
    sk.hi[i] = static_cast<const std::uint8_t*>(
        npu->Staged(stream, q_c[bank][0][n_q + i]));
    sk.lo[i] = static_cast<const std::uint8_t*>(
        npu->Staged(stream, q_c[bank][1][n_q + i]));
  }
  for (std::uint32_t i = 0; i < n_v; ++i) {
    sv.hi[i] = static_cast<const std::uint8_t*>(
        npu->Staged(stream, q_c[bank][0][n_q + n_k + i]));
    sv.lo[i] = static_cast<const std::uint8_t*>(
        npu->Staged(stream, q_c[bank][1][n_q + n_k + i]));
  }
  rocm::NpuDecodeFused(sq, n_q, q.get(), npu->tokens(), npu->n(),
                       heads * c.head_dim, stream);
  rocm::NpuDecodeFused(sk, n_k, k.get(), npu->tokens(), npu->n(), kv_row,
                       stream);
  rocm::NpuDecodeFused(sv, n_v, v.get(), npu->tokens(), npu->n(), kv_row,
                       stream);
  (void)error_msg;
  return true;
}

bool Session::Impl::Dense(const DeviceTensor& w, const float* in, float* out,
                          std::uint32_t n, std::string* error_msg) {
  if (w.type == GgmlType::kQ8_0) {
    // qfn_mmq_q8_0_dense_vec_preq avoids mul_mat_q's WMMA-tile mmq_x>=16
    // granularity floor (gfx1151 is WMMA-only, no MFMA - mmq_x=8 is never
    // usable there), which otherwise wastes 15/16 of a GEMM tile at n=1
    // (confirmed via profiling: 43% of all decode kernel time). Verified:
    // per-call output differs from the GEMM path by ~0.03-0.04 absolute
    // (on values ~15-90), consistent with ordinary Q8 rounding, not a
    // bug - and a full generation run at n=300 stayed coherent and
    // measurably faster (+6%) at the same prompt/seed. The two paths'
    // outputs do drift apart over many layers/tokens, same as every other
    // numeric-path change in this file (MoE routing's near-tie
    // sensitivity) - not specific to this change. See
    // laguna-port-state.md.
    if (n <= 8) {
      if (!QuantizedVec(in, n, w.cols))
        return Fail(error_msg, "Q8_0 vec activation quantization failed");
      if (qfn_mmq_q8_0_dense_vec_preq(w.data, nullptr, q8_vec_quant.get(), out,
                                      static_cast<int>(w.rows),
                                      static_cast<int>(n),
                                      static_cast<int>(w.cols), stream) != 0) {
        return Fail(error_msg, "Q8_0 dense vec projection failed");
      }
      return true;
    }
    // W8A8Gemm (int8 WMMA GEMM, reads the same block_q8_0 weight bytes
    // directly - no re-layout) replaces mul_mat_q for the wide (n>8) Q8_0
    // dense path: measured 1,222 -> 1,275-1,287 tok/s pp2048 (+4-5%).
    // Falls through to the MMQ path if W8A8Gemm declines (k not a multiple
    // of 32). See laguna-port-state.md.
    if (QuantizedTiled(in, n, w.cols) &&
        qwen38_flash_next::rocm::W8A8Gemm(w.data, q8_tiled_quant.get(), out, n,
                                          w.rows, w.cols, stream)) {
      return true;
    }
    if (!Quantized(in, n, w.cols))
      return Fail(error_msg, "Q8_0 activation quantization failed");
    if (qfn_mmq_q8_0_dense_gemm_preq(
            w.data, quant_cache.get(), out, static_cast<int>(w.rows),
            static_cast<int>(n), static_cast<int>(w.cols), stream) != 0) {
      return Fail(error_msg, "Q8_0 dense projection failed");
    }
    return true;
  }
  if (!DenseVecType(w.type))
    return Fail(error_msg, "unsupported dense weight type");
  for (std::uint32_t r0 = 0; r0 < n; r0 += kVecBatch) {
    const std::uint32_t rows = std::min(kVecBatch, n - r0);
    const float* src = in + static_cast<std::size_t>(r0) * w.cols;
    float* dst = out + static_cast<std::size_t>(r0) * w.rows;
    if (qfn_mmq_dense_vec(static_cast<int>(w.type), w.data, src, dst,
                          static_cast<int>(w.rows), static_cast<int>(w.cols),
                          static_cast<int>(rows), stream) != 0) {
      return Fail(error_msg, "quantized dense projection failed");
    }
  }
  return true;
}

bool Session::Impl::Experts(const DeviceTensor& w, const float* in, float* out,
                            std::uint32_t rows, std::uint32_t used,
                            std::uint32_t n_tokens, std::string* error_msg) {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(rows);
  const int U = static_cast<int>(used);
  if (rows * used > kVecExpertRows) {
    return Fail(error_msg, "Laguna: wide expert batch on the vector path");
  }
  (void)n_tokens;
  if (qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, in, ids.get(), out, M,
                      K, T, E, U, stream, nullptr, nullptr) != 0) {
    return Fail(error_msg, "routed expert projection failed");
  }
  return true;
}

bool Session::Impl::Attention(std::uint32_t il, std::uint32_t n,
                              std::string* error_msg) {
  const Config& c = model->config;
  const DeviceLayer& L = model->device->layers()[il];
  const std::uint32_t heads = c.Heads(il);
  const std::uint32_t kv_row = c.num_kv * c.head_dim;
  const bool swa = c.IsSwa(il);
  const RopeTables& rope = model->device->rope(swa);

  // Cross-layer QKV: EnqueueBanked submits layer il's QKV during layer
  // il-1's MoeFinish, so a pending submission here (q_pending[cbank]) is
  // consumed immediately with no further GPU work to overlap it with -
  // the overlap already happened, between the submit and here. Layer 0
  // has nothing to submit it during, so it always falls through to Dense.
  if (q_pending[cbank]) {
    if (!QkvConsume(L, heads, cbank, error_msg))
      return false;
  } else if (!Dense(L.attn_q, cxn, q.get(), n, error_msg) ||
             !Dense(L.attn_k, cxn, k.get(), n, error_msg) ||
             !Dense(L.attn_v, cxn, v.get(), n, error_msg)) {
    return false;
  }
  rocm::Q8Rows(L.attn_gate.data, cxn, gate.get(), n, L.attn_gate.rows,
               L.attn_gate.cols, stream);
  rocm::RmsNormRows(q.get(), static_cast<const float*>(L.attn_q_norm.data),
                    q.get(), n, heads * c.head_dim, heads, c.rms_eps, stream);
  rocm::RmsNormRows(k.get(), static_cast<const float*>(L.attn_k_norm.data),
                    k.get(), n, kv_row, c.num_kv, c.rms_eps, stream);
  rocm::RopeTable(q.get(), n, heads, c.head_dim, rope.n_dims, rope.inv_freq,
                  rope.mscale, pos.get() + cbank, stream);
  rocm::RopeTable(k.get(), n, c.num_kv, c.head_dim, rope.n_dims, rope.inv_freq,
                  rope.mscale, pos.get() + cbank, stream);
  rocm::StoreKv(k.get(), v.get(), k_cache[il].get(), v_cache[il].get(), n,
                kv_row, pos.get() + cbank, ring[il], stream);
  // Block-sparse attention's pooled-key maintenance is the CALLER's job
  // (Forward()'s live pre-pass, or EnqueueBanked's own per-bank call right
  // after this returns), never Attention()'s own - a decode/verify HIP
  // graph can capture this function, and UpdatePooledKeysLaguna's grid
  // size and from_pos/end_pos would then be baked into the recorded graph
  // as fixed launch arguments (not read from a device pointer at replay
  // time), so every later replay would re-pool only the ONE block range
  // that existed at capture time, forever - silently degrading sparse's
  // accuracy the longer a captured decode run goes (invisible on a short
  // needle-in-haystack generation, real on a long completion). See
  // Forward()'s UpdateSparsePooling-equivalent pre-pass and the call
  // EnqueueBanked makes right after this function returns.
  const __half* sparse_kmm = sparse_pooled[il].get();
  rocm::ToHalf(q.get(), qh.get(), std::size_t{n} * heads * c.head_dim, stream);
  // Four real bugs found and fixed in AttentionWmma/AttentionWmmaKernelLaguna
  // (shape guard; token stride missing *2; missing softplus(gate) multiply
  // the scalar reference applies but WMMA silently dropped; head address
  // using the tile width (kWmmaGroup=8) instead of the real per-kv_head
  // stride (heads/kv_heads=6) as the memory stride, which only worked by
  // coincidence for kv_head==0). Verified via `gufo bench
  // --validate-prefill` and a per-layer scalar-vs-WMMA trace
  // (LAGUNA_WMMA_DEBUG=1): every group-6 layer now matches scalar to
  // fp16-precision-level noise only. See laguna-port-state.md for the full
  // debugging trail, including why n=16 alone still fails top1_match (a
  // pre-existing near-tie numeric sensitivity, not a WMMA bug -- confirmed
  // by reproducing the same divergence with WMMA fully disabled).
  constexpr bool kWmmaAttentionEnabled = true;
  // Split-K only pays off where the single-block sweep is long: non-SWA
  // layers (SWA is capped by sliding_window) on the narrow decode/verify
  // batches (prefill's WMMA path is already fine and must stay untouched) past
  // kSplitAttnMinDepth. See kernels.hpp's AttentionSplit.
  const bool split_attn = !swa && n <= Session::kMaxVerifyRows &&
                          position + n >= kSplitAttnMinDepth;
  const bool use_wmma =
      kWmmaAttentionEnabled &&
      rocm::AttentionWmma(
          qh.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
          attn.get(), pos.get() + cbank, heads, c.num_kv, c.head_dim,
          swa ? c.sliding_window : 0, ring[il], n,
          split_attn ? attn_partials.get() : nullptr,
          split_attn ? kSplitAttnSplits : 1, stream, sparse_kmm,
          sparse_max_blocks, sparse_kmm != nullptr ? SparseTopBlocks() : 0);
  if (!use_wmma) {
    rocm::Attention(q.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
                    attn.get(), n, pos.get() + cbank, heads, c.num_kv,
                    c.head_dim, swa ? c.sliding_window : 0, ring[il], stream);
  }
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  if (split_attn && use_wmma && std::getenv("LAGUNA_SPLIT_DEBUG") != nullptr &&
      hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
      capture_status == hipStreamCaptureStatusNone) {
    // Differential trace: the split path is the same WMMA kernel with its key
    // tiles round-robined over grid.z and merged, so against the single-block
    // kernel it must match to fp32 rounding. Any larger difference is a real
    // split/merge bug (suspects: tile round-robin at the tail, ring modulo at
    // split boundaries, merge base/log2 mismatch).
    DeviceBuffer<float> ref_attn;
    if (!ref_attn.Allocate(std::size_t{n} * heads * c.head_dim))
      return Fail(error_msg, "Laguna: split debug allocation failed");
    const bool ref_ok = rocm::AttentionWmma(
        qh.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
        ref_attn.get(), pos.get() + cbank, heads, c.num_kv, c.head_dim,
        swa ? c.sliding_window : 0, ring[il], n, nullptr, 1, stream);
    if (!ref_ok)
      return Fail(error_msg, "Laguna: split debug reference failed");
    LAGUNA_HIP(hipStreamSynchronize(stream));
    std::vector<float> split_host(std::size_t{n} * heads * c.head_dim);
    std::vector<float> ref_host(split_host.size());
    LAGUNA_HIP(hipMemcpy(split_host.data(), attn.get(),
                         split_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    LAGUNA_HIP(hipMemcpy(ref_host.data(), ref_attn.get(),
                         ref_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    double max_abs = 0.0, max_ref = 0.0;
    for (std::size_t i = 0; i < split_host.size(); ++i) {
      max_abs = std::max(
          max_abs, static_cast<double>(std::fabs(split_host[i] - ref_host[i])));
      max_ref = std::max(max_ref, static_cast<double>(std::fabs(ref_host[i])));
    }
    std::fprintf(stderr,
                 "[laguna] split debug layer=%u n=%u depth=%u max_abs=%g "
                 "max_ref=%g rel=%g\n",
                 il, n, position, max_abs, max_ref,
                 max_ref > 0.0 ? max_abs / max_ref : 0.0);
  }
  if (use_wmma && std::getenv("LAGUNA_WMMA_DEBUG") != nullptr &&
      hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
      capture_status == hipStreamCaptureStatusNone) {
    DeviceBuffer<float> scalar_attn;
    scalar_attn.Allocate(std::size_t{n} * heads * c.head_dim);
    rocm::Attention(q.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
                    scalar_attn.get(), n, pos.get() + cbank, heads, c.num_kv,
                    c.head_dim, swa ? c.sliding_window : 0, ring[il], stream);
    LAGUNA_HIP(hipStreamSynchronize(stream));
    std::vector<float> wmma_host(std::size_t{n} * heads * c.head_dim);
    std::vector<float> scalar_host(wmma_host.size());
    LAGUNA_HIP(hipMemcpy(wmma_host.data(), attn.get(),
                         wmma_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    LAGUNA_HIP(hipMemcpy(scalar_host.data(), scalar_attn.get(),
                         scalar_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    for (std::uint32_t t = 0; t < n; ++t) {
      for (std::uint32_t h = 0; h < heads; ++h) {
        double max_abs = 0.0;
        double sum_abs_scalar = 0.0;
        const std::size_t base = (std::size_t{t} * heads + h) * c.head_dim;
        for (std::uint32_t d = 0; d < c.head_dim; ++d) {
          const double diff =
              std::fabs(wmma_host[base + d] - scalar_host[base + d]);
          max_abs = std::max(max_abs, diff);
          sum_abs_scalar += std::fabs(scalar_host[base + d]);
        }
        std::fprintf(stderr,
                     "[LAGUNA_WMMA_DEBUG] layer=%u token=%u head=%u "
                     "max_abs_diff=%.6f scalar_sum_abs=%.6f "
                     "wmma[0..3]=%.4f,%.4f,%.4f,%.4f "
                     "scalar[0..3]=%.4f,%.4f,%.4f,%.4f\n",
                     il, t, h, max_abs, sum_abs_scalar, wmma_host[base + 0],
                     wmma_host[base + 1], wmma_host[base + 2],
                     wmma_host[base + 3], scalar_host[base + 0],
                     scalar_host[base + 1], scalar_host[base + 2],
                     scalar_host[base + 3]);
      }
    }
  }
  if (!Dense(L.attn_output, attn.get(), o.get(), n, error_msg))
    return false;
  rocm::AddInPlace(cx, o.get(), std::size_t{n} * c.hidden_size, stream);
  return true;
}

bool Session::Impl::DenseFfn(const DeviceLayer& layer, std::uint32_t n,
                             std::string* error_msg) {
  const Config& c = model->config;
  if (!Dense(layer.ffn_gate, cxn, dense_gate.get(), n, error_msg) ||
      !Dense(layer.ffn_up, cxn, dense_up.get(), n, error_msg)) {
    return false;
  }
  rocm::Swiglu(dense_gate.get(), dense_up.get(), std::size_t{n} * c.dense_ff,
               stream);
  if (!Dense(layer.ffn_down, dense_gate.get(), o.get(), n, error_msg))
    return false;
  rocm::AddInPlace(cx, o.get(), std::size_t{n} * c.hidden_size, stream);
  return true;
}

bool Session::Impl::RoutedExperts(const DeviceLayer& layer, std::uint32_t n,
                                  int bank, bool* scaled,
                                  std::string* error_msg) {
  using qwen38_flash_next::rocm::WeightType;
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  const std::uint32_t experts = c.expert_count;
  const DeviceTensor& ge = layer.gate_exps;
  const DeviceTensor& ue = layer.up_exps;
  const DeviceTensor& de = layer.down_exps;
  // 2026-10-30: checked whether Laguna's gate/up experts are ever Q8_0 (the
  // fused RoutedGatedF16Gemm path only covers K-quant types) - confirmed via
  // instrumentation they are always Q4_K across every MoE layer in the
  // UD-Q4_K_M quant (de alone varies, Q5_K or Q8_0), so this model already
  // takes the fused gate/up path and there is nothing to add here. See
  // laguna-port-state.md for the full investigation.
  const bool gated = ge.type == ue.type &&
                     (ge.type == GgmlType::kQ4_K || ge.type == GgmlType::kQ5_K);
  *scaled = gated;
  const std::uint32_t pair_rows = gated ? kPairTileRows : kPlainTileRows;
  const std::uint32_t down_rows =
      de.type == GgmlType::kQ8_0 ? kQ8TileRows : kPlainTileRows;

  qwen38_flash_next::rocm::ExpertCounts(ids.get(), counts.get(), n, experts,
                                        used, stream);
  LAGUNA_HIP(hipMemcpyAsync(h_counts, counts.get(),
                            experts * sizeof(std::uint32_t),
                            hipMemcpyDeviceToHost, stream));
  LAGUNA_HIP(hipEventRecord(ready, stream));
  static const bool kSyncDebug = std::getenv("LAGUNA_SYNC_DEBUG") != nullptr;
  static double sync_total_us = 0.0;
  static std::uint32_t sync_calls = 0;
  if (kSyncDebug) {
    const auto t0 = std::chrono::steady_clock::now();
    LAGUNA_HIP(hipEventSynchronize(ready));
    const auto t1 = std::chrono::steady_clock::now();
    sync_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (++sync_calls % 47 == 0) {
      std::fprintf(stderr,
                   "[LAGUNA_SYNC_DEBUG] calls=%u cumulative=%.1f us "
                   "mean=%.2f us/call\n",
                   sync_calls, sync_total_us, sync_total_us / sync_calls);
    }
  } else {
    LAGUNA_HIP(hipEventSynchronize(ready));
  }
  std::uint32_t pair_tiles = 0;
  std::uint32_t down_tiles = 0;
  std::int32_t* host_pair = h_tiles;
  std::int32_t* host_down =
      h_tiles + TileCapacityFor(static_cast<std::size_t>(tile) * used, experts);
  for (std::uint32_t e = 0; e < experts; ++e) {
    const std::uint32_t padded =
        (h_counts[e] + kRoutePadRows - 1) / kRoutePadRows * kRoutePadRows;
    for (std::uint32_t j = 0; j * pair_rows < padded; ++j) {
      host_pair[pair_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    for (std::uint32_t j = 0; j * down_rows < padded; ++j) {
      host_down[down_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  if (pair_tiles == 0)
    return Fail(error_msg, "routed experts: no tiles");
  LAGUNA_HIP(hipMemcpyAsync(tiles_pair.get(), host_pair,
                            pair_tiles * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  LAGUNA_HIP(hipMemcpyAsync(tiles_down.get(), host_down,
                            down_tiles * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  qwen38_flash_next::rocm::RoutedCompact(
      ids.get(), counts.get(), route_bounds.get(), route_cursors.get(),
      rows_token.get(), rows_slot.get(), n, used, experts, stream);
  rocm::NarrowRows(cxn, xh, n, c.hidden_size, c.hidden_size, stream);

  using qwen38_flash_next::rocm::RoutedF16Gemm;
  using qwen38_flash_next::rocm::RoutedGatedF16Gemm;
  const auto gate_type = static_cast<WeightType>(ge.type);
  const auto down_type = static_cast<WeightType>(de.type);
  bool ok;
  if (gated) {
    ok = RoutedGatedF16Gemm(ge.data, ue.data, gate_type, xh, tiles_pair.get(),
                            pair_tiles, pair_rows, route_bounds.get(),
                            rows_token.get(), rows_slot.get(), act_half.get(),
                            ge.rows, ge.cols, stream, 1.0F / kRoutedActScale);
  } else {
    ok = RoutedF16Gemm(ge.data, gate_type, xh, tiles_pair.get(), pair_tiles,
                       pair_rows, route_bounds.get(), rows_token.get(),
                       rows_slot.get(), nullptr, gate_e.get(), nullptr, ge.rows,
                       ge.cols, stream) &&
         RoutedF16Gemm(ue.data, static_cast<WeightType>(ue.type), xh,
                       tiles_pair.get(), pair_tiles, pair_rows,
                       route_bounds.get(), rows_token.get(), rows_slot.get(),
                       gate_e.get(), nullptr, act_half.get(), ue.rows, ue.cols,
                       stream);
  }
  if (!ok)
    return Fail(error_msg, "routed gate/up projection failed");
  const bool down_wide = kQ8WideDownTile && down_type == WeightType::kQ8_0;
  const bool down_ok =
      down_wide
          ? qwen38_flash_next::rocm::RoutedF16GemmQ8Wide(
                de.data, act_half.get(), tiles_down.get(), down_tiles,
                route_bounds.get(), rows_slot.get(), rows_slot.get(), nullptr,
                down_e[bank].get(), nullptr, de.rows, de.cols, stream)
          : RoutedF16Gemm(de.data, down_type, act_half.get(), tiles_down.get(),
                          down_tiles, down_rows, route_bounds.get(),
                          rows_slot.get(), rows_slot.get(), nullptr,
                          down_e[bank].get(), nullptr, de.rows, de.cols,
                          stream);
  if (!down_ok) {
    return Fail(error_msg, "routed down projection failed");
  }
  return true;
}

bool Session::Impl::MoeRoute(const DeviceLayer& layer, std::uint32_t n,
                             int bank, bool* wide_out, bool* scaled_out,
                             std::string* error_msg) {
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  const bool wide = n * used > kVecExpertRows;
  *wide_out = wide;

  rocm::RouterF32(static_cast<const float*>(layer.router.data), cxn,
                  router.get(), n, c.expert_count, c.hidden_size, stream);
  rocm::SigmoidBiasTopK(router.get(),
                        static_cast<const float*>(layer.router_bias.data),
                        ids.get(), topk[bank].get(), n, c.expert_count, used,
                        c.expert_weights_norm, c.expert_weights_scale, stream);

  // Submit the shared-expert gate/up GEMM to the NPU before the (larger)
  // routed-expert GEMMs below, so its latency overlaps that GPU work
  // instead of blocking on it. When banked (EnqueueBanked), the actual
  // wait+combine is deferred to MoeFinish, called after the OTHER bank's
  // routed experts too - doubling the GPU work available to hide behind.
  if (!ShexpSubmit(layer, n, bank, error_msg))
    return false;
  if (!sh_pending[bank]) {
    if (!Dense(layer.gate_shexp, cxn, sh_gate[bank].get(), n, error_msg) ||
        !Dense(layer.up_shexp, cxn, sh_up.get(), n, error_msg)) {
      return false;
    }
    rocm::Swiglu(sh_gate[bank].get(), sh_up.get(), std::size_t{n} * c.shared_ff,
                 stream);
  }

  bool scaled = false;
  if (wide) {
    if (!RoutedExperts(layer, n, bank, &scaled, error_msg))
      return false;
  } else {
    const DeviceTensor& ge = layer.gate_exps;
    const DeviceTensor& ue = layer.up_exps;
    const bool paired =
        ge.type == ue.type && ge.rows == ue.rows && ge.cols == ue.cols;
    if (paired) {
      if (qfn_mmq_moe_vec(static_cast<int>(ge.type), ge.data, cxn, ids.get(),
                          gate_e.get(), static_cast<int>(ge.rows),
                          static_cast<int>(ge.cols), static_cast<int>(n),
                          static_cast<int>(ge.experts), static_cast<int>(used),
                          stream, ue.data, up_e.get()) != 0) {
        return Fail(error_msg, "paired expert projection failed");
      }
    } else if (!Experts(ge, cxn, gate_e.get(), n, used, n, error_msg) ||
               !Experts(ue, cxn, up_e.get(), n, used, n, error_msg)) {
      return false;
    }
    // Non-Q8_0 down projections read the half-precision sum of each Q8_1
    // block, which overflows for very large rows: keep rows inside range.
    scaled = layer.down_exps.type != GgmlType::kQ8_0;
    if (scaled) {
      rocm::SwigluScaled(gate_e.get(), up_e.get(), expert_scale[bank].get(),
                         n * used, c.expert_ff, stream);
    } else {
      rocm::Swiglu(gate_e.get(), up_e.get(),
                   static_cast<std::size_t>(n) * used * c.expert_ff, stream);
    }
    if (!Experts(layer.down_exps, gate_e.get(), down_e[bank].get(), n * used, 1,
                 n, error_msg)) {
      return false;
    }
  }
  *scaled_out = scaled;
  return true;
}

bool Session::Impl::MoeFinish(const DeviceLayer& layer, std::uint32_t n,
                              int bank, bool wide, bool scaled,
                              std::string* error_msg) {
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  if (sh_pending[bank]) {
    sh_pending[bank] = false;
    auto* npu = model->device->npu();
    npu->Wait(stream, sh_ticket[bank]);
    if (std::getenv("LAGUNA_NPU_SHEXP_DEBUG") != nullptr) {
      DeviceBuffer<float> ref_gate, ref_up;
      ref_gate.Allocate(std::size_t{n} * c.shared_ff);
      ref_up.Allocate(std::size_t{n} * c.shared_ff);
      Dense(layer.gate_shexp, cxn, ref_gate.get(), n, error_msg);
      Dense(layer.up_shexp, cxn, ref_up.get(), n, error_msg);
      rocm::Swiglu(ref_gate.get(), ref_up.get(), std::size_t{n} * c.shared_ff,
                   stream);
      rocm::NpuSwigluCombine(npu->Staged(stream, sh_c[bank][0]),
                             npu->Staged(stream, sh_c[bank][1]),
                             npu->Staged(stream, sh_c[bank][2]),
                             npu->Staged(stream, sh_c[bank][3]),
                             sh_gate[bank].get(), npu->tokens(), npu->n(),
                             c.shared_ff, stream);
      LAGUNA_HIP(hipStreamSynchronize(stream));
      std::vector<float> npu_host(std::size_t{n} * c.shared_ff);
      std::vector<float> ref_host(npu_host.size());
      LAGUNA_HIP(hipMemcpy(npu_host.data(), sh_gate[bank].get(),
                           npu_host.size() * sizeof(float),
                           hipMemcpyDeviceToHost));
      LAGUNA_HIP(hipMemcpy(ref_host.data(), ref_gate.get(),
                           ref_host.size() * sizeof(float),
                           hipMemcpyDeviceToHost));
      double max_abs = 0.0, max_ref = 0.0;
      for (std::size_t i = 0; i < npu_host.size(); ++i) {
        max_abs = std::max(
            max_abs, static_cast<double>(std::fabs(npu_host[i] - ref_host[i])));
        max_ref =
            std::max(max_ref, static_cast<double>(std::fabs(ref_host[i])));
      }
      std::fprintf(stderr,
                   "[laguna] shexp npu debug n=%u max_abs=%g max_ref=%g "
                   "rel=%g npu[0]=%g ref[0]=%g\n",
                   n, max_abs, max_ref, max_ref > 0.0 ? max_abs / max_ref : 0.0,
                   npu_host[0], ref_host[0]);
    } else {
      rocm::NpuSwigluCombine(npu->Staged(stream, sh_c[bank][0]),
                             npu->Staged(stream, sh_c[bank][1]),
                             npu->Staged(stream, sh_c[bank][2]),
                             npu->Staged(stream, sh_c[bank][3]),
                             sh_gate[bank].get(), npu->tokens(), npu->n(),
                             c.shared_ff, stream);
    }
  }
  if (!Dense(layer.down_shexp, sh_gate[bank].get(), sh_out.get(), n,
             error_msg)) {
    return false;
  }
  const float* expert_scale_ptr =
      wide ? expert_const.get() : expert_scale[bank].get();
  rocm::MoeCombine(cx, down_e[bank].get(), topk[bank].get(),
                   scaled ? expert_scale_ptr : nullptr, sh_out.get(), nullptr,
                   n, used, c.hidden_size, stream);
  return true;
}

bool Session::Impl::Moe(const DeviceLayer& layer, std::uint32_t n,
                        std::string* error_msg) {
  bool wide = false, scaled = false;
  return MoeRoute(layer, n, cbank, &wide, &scaled, error_msg) &&
         MoeFinish(layer, n, cbank, wide, scaled, error_msg);
}

bool Session::Impl::Enqueue(std::uint32_t n, bool all_rows,
                            std::uint32_t prefix_end,
                            std::uint32_t* tap_rows_out,
                            std::uint32_t* tap_first_out,
                            std::string* error_msg) {
  SetBank(0);
  const Config& c = model->config;
  const DeviceModel& dm = *model->device;
  LAGUNA_HIP(hipMemcpyAsync(tokens.get(), h_tokens, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  LAGUNA_HIP(hipMemcpyAsync(pos.get(), h_pos, 4 * sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream));
  rocm::EmbedQ8(dm.token_embd().data, tokens.get(), cx, n, c.hidden_size,
                stream);
  std::uint32_t tap_rows = 0, tap_first = 0;
  if (dflash != nullptr) {
    // Only the drafter's sliding window matters: capture the rows ending the
    // session prefix and skip forwards that finish far before it.
    //
    // `x` holds only this call's own `n` rows ([0, n) locally, [position,
    // end) in absolute terms) - `tap_first` must be an offset into that
    // local range, not an absolute position. Compute the absolute start of
    // the trailing dflash_window-sized slice first, then clip it into this
    // call's local range; for a full-prompt prefill (position == 0) this
    // reduces to the original absolute math, but for a 1-row decode step
    // (position > 0) it correctly captures just that row instead of
    // reading `end` rows out of a 1-row buffer.
    const std::uint32_t end = position + n;
    if (end + model->dflash_window >= prefix_end) {
      const std::uint32_t want_first =
          end > model->dflash_window ? end - model->dflash_window : 0;
      tap_first = want_first > position ? want_first - position : 0;
      tap_rows = n - tap_first;
      // tap_pos (Forward()'s local, mirroring this) is used post-call.
    }
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& L = dm.layers()[il];
    rocm::RmsNormRows(cx, static_cast<const float*>(L.attn_norm.data), cxn, n,
                      c.hidden_size, 1, c.rms_eps, stream);
    if (!Attention(il, n, error_msg))
      return false;
    rocm::RmsNormRows(cx, static_cast<const float*>(L.ffn_norm.data), cxn, n,
                      c.hidden_size, 1, c.rms_eps, stream);
    if (!(L.is_moe ? Moe(L, n, error_msg) : DenseFfn(L, n, error_msg)))
      return false;
    if (tap_rows != 0 &&
        std::find(model->dflash_taps.begin(), model->dflash_taps.end(),
                  il + 1) != model->dflash_taps.end()) {
      const auto tap =
          static_cast<std::size_t>(std::find(model->dflash_taps.begin(),
                                             model->dflash_taps.end(), il + 1) -
                                   model->dflash_taps.begin());
      rocm::CaptureTap(
          tap_capture.get(),
          cx + static_cast<std::size_t>(tap_first) * c.hidden_size, tap_rows,
          c.hidden_size,
          static_cast<std::size_t>(model->dflash_taps.size()) * c.hidden_size,
          tap * c.hidden_size, stream);
    }
  }
  const std::uint32_t head_rows = all_rows ? n : 1;
  const float* head_in =
      cx + (all_rows ? 0 : std::size_t{n - 1} * c.hidden_size);
  rocm::RmsNormRows(head_in, static_cast<const float*>(dm.output_norm().data),
                    last.get(), head_rows, c.hidden_size, 1, c.rms_eps, stream);
  if (!Dense(dm.output(), last.get(), d_logits.get(), head_rows, error_msg))
    return false;
  LAGUNA_HIP(
      hipMemcpyAsync(h_logits, d_logits.get(),
                     std::size_t{head_rows} * c.vocab_size * sizeof(float),
                     hipMemcpyDeviceToHost, stream));
  *tap_rows_out = tap_rows;
  *tap_first_out = tap_first;
  return true;
}

bool Session::Impl::Forward(std::span<const std::int32_t> input, bool all_rows,
                            std::uint32_t prefix_end, std::string* error_msg) {
  const auto n = static_cast<std::uint32_t>(input.size());
  std::copy(input.begin(), input.end(), h_tokens);
  h_pos[0] = position;
  logit_rows = all_rows ? n : 1;

  // Block-sparse attention's pooled-key maintenance, run live every call
  // (including ones a captured graph will replay) - see the long comment
  // on the equivalent, capture-skipped call inside Attention() for why
  // this can't just live there for the graphed decode path. k_cache is not
  // banked, so bank 0's view of it here is the whole, current cache
  // regardless of which bank API wrote it (EnqueueBanked handles its own
  // pooling directly, inside Attention(), since it never captures a graph).
  {
    const Config& c = model->config;
    const std::uint32_t end = position + n;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (sparse_pooled[il].get() == nullptr)
        continue;
      rocm::UpdatePooledKeysLaguna(k_cache[il].get(), sparse_pooled[il].get(),
                                   sparse_valid_end[il], end, c.num_kv,
                                   c.head_dim, sparse_max_blocks, stream);
      sparse_valid_end[il] = end;
    }
  }

  // Same tap-window math as Enqueue() (duplicated, not shared, since it must
  // run every call - including replayed ones, where Enqueue()'s body does
  // not re-execute - to decide graphability and drive DflashInject below).
  std::uint32_t tap_rows = 0, tap_first = 0, tap_pos = 0;
  if (dflash != nullptr) {
    const std::uint32_t end = position + n;
    if (end + model->dflash_window >= prefix_end) {
      const std::uint32_t want_first =
          end > model->dflash_window ? end - model->dflash_window : 0;
      tap_first = want_first > position ? want_first - position : 0;
      tap_rows = n - tap_first;
      tap_pos = position + tap_first;
    }
  }

  // Captured HIP graph per decode/verify width: the whole per-step kernel
  // sequence (Enqueue) launches once from host code, then replays as a
  // single hipGraphLaunch - cuts per-kernel launch overhead, the dominant
  // cost at n=1..16 on gfx1151. Only when tap_rows==n (tap_first==0, the
  // DFlash drafter's tap window has reached steady state - always true for
  // plain decode since dflash==nullptr keeps tap_rows at 0==n only when
  // n==0, which never happens, so this also correctly excludes plain decode
  // from ever reading a stale tap_rows==n check... ) - see the StepGraph
  // comment at its declaration for the full safety argument.
  const bool steady = dflash == nullptr || tap_rows == n;
  StepGraph& g = graphs[std::min<std::uint32_t>(n, Session::kMaxVerifyRows)];
  // Only capture once past the split-K depth threshold: the per-layer
  // Attention() dispatch is a host-side decision baked into the recorded
  // graph, so a graph captured at shallow depth would replay the non-split
  // kernel forever, exactly where split-K starts mattering.
  const bool graphable = all_rows && n <= Session::kMaxVerifyRows && steady &&
                         position + n >= kSplitAttnMinDepth;
  std::uint32_t enq_tap_rows = tap_rows, enq_tap_first = tap_first;
  if (graphable)
    ++g.steps;
  if (graphable && g.steps > 1 && g.exec == nullptr && !g.failed) {
    LAGUNA_HIP(hipStreamSynchronize(stream));
    LAGUNA_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    std::string capture_error;
    const bool enqueued = Enqueue(n, all_rows, prefix_end, &enq_tap_rows,
                                  &enq_tap_first, &capture_error);
    hipGraph_t graph = nullptr;
    const hipError_t ended = hipStreamEndCapture(stream, &graph);
    if (!enqueued || ended != hipSuccess ||
        hipGraphInstantiate(&g.exec, graph, nullptr, nullptr, 0) !=
            hipSuccess) {
      g.failed = true;
      g.exec = nullptr;
      (void)hipGetLastError();
      std::fprintf(stderr, "[laguna] decode graph unavailable: %s\n",
                   capture_error.c_str());
    }
    if (graph != nullptr)
      (void)hipGraphDestroy(graph);
  }
  if (graphable && g.exec != nullptr) {
    LAGUNA_HIP(hipGraphLaunch(g.exec, stream));
  } else if (!Enqueue(n, all_rows, prefix_end, &enq_tap_rows, &enq_tap_first,
                      error_msg)) {
    return false;
  }
  LAGUNA_HIP(hipStreamSynchronize(stream));
  position += n;
  if (tap_rows != 0) {
    tap_pending = tap_rows;
    tap_pending_pos = tap_pos;
    tap_pending_first = 0;
    if (inject_taps && !DflashInject(tap_pending, error_msg))
      return false;
  }
  return true;
}

bool Session::Impl::EnqueueBanked(std::span<const std::int32_t> input,
                                  std::string* error_msg) {
  // Banked prefill: splits one 2*tile chunk into two independent tile-sized
  // halves, processed per layer as bank 0 then bank 1, so each bank's NPU
  // shexp wait is hidden behind the OTHER bank's RoutedExperts GPU work (not
  // just its own) - see the StepGraph-adjacent comments on cx/cxn/SetBank
  // and ShexpSubmit/MoeRoute/MoeFinish. Scoped deliberately narrow for a
  // first cut: exactly two equal tile-sized halves (no odd remainder, no
  // DFlash tap capture, no graph capture - prefill chunks here are far
  // above kMaxVerifyRows anyway). See laguna-port-state.md.
  const Config& c = model->config;
  const DeviceModel& dm = *model->device;
  const std::uint32_t bn = tile;
  // h_pos[0]/h_pos[1] (distinct host slots, not reused) hold each bank's
  // absolute start position; written once up front, before any async copy
  // reads them, to avoid the exact bug this replaced: writing a PINNED
  // host source buffer again (overwriting bank 0's bytes with bank 1's)
  // before its hipMemcpyAsync had actually been executed by the DMA engine
  // - hipMemcpyAsync queues the copy and returns immediately, it does NOT
  // wait for the transfer, so reusing the same host buffer across banks
  // without a sync was a real race (this is what produced banking's
  // garbage output, not the buffer-aliasing or ring-buffer bugs fixed
  // above - those were real too, but this was the actual cause). Reading
  // straight from `input` (the caller's own stable span, never mutated
  // here) for tokens avoids needing a second host buffer at all.
  h_pos[0] = position;
  h_pos[1] = position + bn;
  for (int b = 0; b < 2; ++b) {
    SetBank(b);
    LAGUNA_HIP(hipMemcpyAsync(tokens.get(), input.data() + b * bn,
                              bn * sizeof(std::int32_t), hipMemcpyHostToDevice,
                              stream));
    LAGUNA_HIP(hipMemcpyAsync(pos.get() + b, h_pos + b, sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream));
    rocm::EmbedQ8(dm.token_embd().data, tokens.get(), cx, bn, c.hidden_size,
                  stream);
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& L = dm.layers()[il];
    for (int b = 0; b < 2; ++b) {
      SetBank(b);
      rocm::RmsNormRows(cx, static_cast<const float*>(L.attn_norm.data), cxn,
                        bn, c.hidden_size, 1, c.rms_eps, stream);
      if (!Attention(il, bn, error_msg))
        return false;
      // Live pooled-key maintenance for sparse attention - see the long
      // comment in Attention() for why this isn't done there. EnqueueBanked
      // never uses graph capture, so it's always safe to do it right here,
      // per bank, right after that bank's StoreKv (inside Attention()) has
      // actually written the new rows.
      if (sparse_pooled[il].get() != nullptr) {
        const std::uint32_t end = h_pos[b] + bn;
        rocm::UpdatePooledKeysLaguna(k_cache[il].get(), sparse_pooled[il].get(),
                                     sparse_valid_end[il], end, c.num_kv,
                                     c.head_dim, sparse_max_blocks, stream);
        sparse_valid_end[il] = end;
      }
      rocm::RmsNormRows(cx, static_cast<const float*>(L.ffn_norm.data), cxn, bn,
                        c.hidden_size, 1, c.rms_eps, stream);
    }
    if (!L.is_moe) {
      for (int b = 0; b < 2; ++b) {
        SetBank(b);
        if (!DenseFfn(L, bn, error_msg))
          return false;
      }
    } else {
      bool wide[2] = {false, false};
      bool scaled[2] = {false, false};
      for (int b = 0; b < 2; ++b) {
        SetBank(b);
        if (!MoeRoute(L, bn, b, &wide[b], &scaled[b], error_msg))
          return false;
      }
      for (int b = 0; b < 2; ++b) {
        SetBank(b);
        if (!MoeFinish(L, bn, b, wide[b], scaled[b], error_msg))
          return false;
      }
    }
    // Cross-layer QKV: submit layer il+1's QKV now, using bank b's just-
    // updated residual stream (the AddInPlace/MoeCombine residual add for
    // layer il already ran, above) - consumed at the start of il+1's
    // Attention(bank b). Hidden behind the OTHER bank's remaining work for
    // layer il (DenseFfn, or the MoeFinish calls above), not a same-call
    // submit-then-wait. Opt-in (LAGUNA_NPU_QKV=1); QkvSubmit itself no-ops
    // otherwise.
    if (il + 1 < c.num_layers) {
      const DeviceLayer& next = dm.layers()[il + 1];
      for (int b = 0; b < 2; ++b) {
        SetBank(b);
        rocm::RmsNormRows(cx, static_cast<const float*>(next.attn_norm.data),
                          cxn, bn, c.hidden_size, 1, c.rms_eps, stream);
        if (!QkvSubmit(next, bn, b, error_msg))
          return false;
      }
    }
  }
  // Append() never needs all-rows logits; only the final absolute token's
  // (bank 1's last row) matters, same convention as the non-banked path.
  SetBank(1);
  const float* head_in = cx + std::size_t{bn - 1} * c.hidden_size;
  rocm::RmsNormRows(head_in, static_cast<const float*>(dm.output_norm().data),
                    last.get(), 1, c.hidden_size, 1, c.rms_eps, stream);
  if (!Dense(dm.output(), last.get(), d_logits.get(), 1, error_msg))
    return false;
  LAGUNA_HIP(hipMemcpyAsync(h_logits, d_logits.get(),
                            std::size_t{c.vocab_size} * sizeof(float),
                            hipMemcpyDeviceToHost, stream));
  LAGUNA_HIP(hipStreamSynchronize(stream));
  position += 2 * bn;
  return true;
}

Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

bool Session::Append(std::span<const std::int32_t> tokens,
                     std::string* error_msg) {
  if (tokens.empty())
    return Fail(error_msg, "Laguna: no tokens to append");
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Laguna: context capacity exceeded");
  }
  const std::uint32_t prefix_end =
      impl_->position + static_cast<std::uint32_t>(tokens.size());
  // Banked prefill (two tile-sized halves, NPU work of one hidden behind
  // the other's GPU work) only when the NPU is actually in use and there's
  // no DFlash tap bookkeeping to interleave (EnqueueBanked doesn't support
  // it - see its comment). Scoped to exact 2*tile chunks; any remainder
  // falls through to the normal per-tile loop below.
  // 2026-10-31: was briefly shipped off by default after a real bug (a
  // host-side pinned-buffer race in the per-bank token/position upload -
  // the SAME host buffer was overwritten for bank 1 before bank 0's async
  // copy had actually run, plus the SWA ring buffer was sized for one
  // tile's worth of new rows instead of two) caused real-prose generation
  // to diverge from the first token. Both fixed and re-verified (bit-exact
  // on an isolated single-banked-call repro, normal-divergence-pattern on
  // a 45k-token real document, `--validate-prefill` unchanged) - on by
  // default again. See laguna-port-state.md for the fix and the
  // measured, modest (+2-3%, same order as the non-banked shexp-only
  // path) gain at this scale - QKV still doesn't benefit from banking as
  // built (its submit/consume stays within one Attention() call per bank,
  // not deferred across banks the way shexp's is), so LAGUNA_NPU_QKV
  // stays opt-in regardless of banking.
  const bool can_bank = std::getenv("LAGUNA_NPU_NO_BANK") == nullptr &&
                        impl_->dflash == nullptr &&
                        impl_->model->device->npu() != nullptr;
  std::size_t off = 0;
  if (can_bank) {
    for (; off + 2 * impl_->tile <= tokens.size(); off += 2 * impl_->tile) {
      if (!impl_->EnqueueBanked(tokens.subspan(off, 2 * impl_->tile),
                                error_msg)) {
        return false;
      }
    }
  }
  for (; off < tokens.size();) {
    const std::size_t n =
        std::min<std::size_t>(tokens.size() - off, impl_->tile);
    // all_rows=true when n==1 (plain single-token decode): identical to
    // "last row only" at n=1, but makes the step graphable (see Forward()'s
    // StepGraph comment) - this is what gives plain decode a flat per-step
    // graph replay instead of per-kernel launch overhead every step.
    const bool single = tokens.size() == 1;
    if (!impl_->Forward(tokens.subspan(off, n), single, prefix_end, error_msg))
      return false;
    off += n;
  }
  return true;
}

bool Session::AppendAll(std::span<const std::int32_t> tokens,
                        std::string* error_msg) {
  if (tokens.empty() || tokens.size() > kMaxVerifyRows) {
    return Fail(error_msg, "Laguna: unsupported verification width");
  }
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Laguna: context capacity exceeded");
  }
  impl_->inject_taps = false;
  const bool ok = impl_->Forward(
      tokens, true, impl_->position + static_cast<std::uint32_t>(tokens.size()),
      error_msg);
  impl_->inject_taps = true;
  return ok;
}

void Session::Rewind(std::uint32_t position) noexcept {
  if (position < impl_->position) {
    impl_->position = position;
    // A pooled block covering rows past the rewound position is stale (it
    // pooled KV that is about to be overwritten by a later Append) - drop
    // back to the block boundary at or below `position` so the next
    // UpdatePooledKeysLaguna call re-pools it from scratch instead of
    // trusting a min/max computed from since-discarded rows.
    for (auto& end : impl_->sparse_valid_end)
      end = std::min(end, (position / 64) * 64);
  }
}

std::span<const float> Session::Logits() const noexcept {
  const auto vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{impl_->logit_rows - 1} * vocab, vocab};
}

std::span<const float> Session::Logits(std::uint32_t row) const noexcept {
  const auto vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{row} * vocab, vocab};
}

std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

std::uint32_t Session::Capacity() const noexcept {
  return impl_->capacity;
}

void Session::Reset() noexcept {
  impl_->position = 0;
  std::fill(impl_->sparse_valid_end.begin(), impl_->sparse_valid_end.end(), 0);
  if (impl_->dflash != nullptr)
    impl_->dflash->Reset();
}

hip::QwenDFlashGpuExecutor* Session::DflashExecutor() const noexcept {
  return impl_->dflash.get();
}

bool Session::DflashInject(std::uint32_t rows, std::string* error_msg) {
  return impl_->DflashInject(rows, error_msg);
}

bool Session::Impl::DflashInject(std::uint32_t rows, std::string* error_msg) {
  if (dflash == nullptr)
    return true;
  if (rows > tap_pending) {
    tap_pending = 0;
    return Fail(error_msg, "Laguna: drafter capture out of range");
  }
  if (rows == 0)
    return true;
  if (!dflash->InjectTargetContextDevice(
          tap_capture.get() + static_cast<std::size_t>(tap_pending_first) *
                                  model->dflash_taps.size() *
                                  model->config.hidden_size,
          tap_pending_pos, rows)) {
    tap_pending = 0;
    return Fail(error_msg, "Laguna: drafter context injection failed");
  }
  tap_pending -= rows;
  tap_pending_pos += rows;
  tap_pending_first += rows;
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& b : impl_->k_cache)
    total += b.bytes();
  for (const auto& b : impl_->v_cache)
    total += b.bytes();
  return total;
}

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const EngineOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> model(new Model());
  auto impl = std::make_shared<ModelState>();
  impl->options = options;
  // Must match device_model.cpp's own read of the same env var (which picks
  // the NPU instruction file by M) - duplicated rather than threaded
  // through Config/EngineOptions, since DeviceModel::Upload only sees
  // Config, not EngineOptions.
  if (const char* t = std::getenv("LAGUNA_PREFILL_TILE")) {
    const int v = std::atoi(t);
    if (v > 0)
      impl->options.prefill_tile = static_cast<std::uint32_t>(v);
  }
  std::string err;
  impl->reader = core::GgufReader::OpenFile(model_path, &err);
  if (!impl->reader) {
    Fail(error_msg, "failed to open GGUF: " + err);
    return nullptr;
  }
  const auto config = ParseConfig(*impl->reader, &err);
  if (!config) {
    Fail(error_msg, "failed to parse Laguna config: " + err);
    return nullptr;
  }
  impl->config = *config;
  impl->device = DeviceModel::Upload(*impl->reader, impl->config, &err);
  if (!impl->device) {
    Fail(error_msg, "failed to upload Laguna weights: " + err);
    return nullptr;
  }
  if (!options.dflash_model_path.empty()) {
    auto dflash_reader =
        core::GgufReader::OpenFile(options.dflash_model_path, &err);
    if (!dflash_reader) {
      Fail(error_msg, "failed to open the drafter: " + err);
      return nullptr;
    }
    std::string dflash_err;
    impl->dflash_model = hip::QwenDFlashGpuModel::Create(
        std::move(dflash_reader), impl->config.hidden_size,
        impl->config.vocab_size, impl->config.num_layers,
        TensorRef(impl->device->token_embd()),
        TensorRef(impl->device->output()), &dflash_err);
    if (!impl->dflash_model) {
      Fail(error_msg, "failed to load the drafter: " + dflash_err);
      return nullptr;
    }
    const auto& dcfg = impl->dflash_model->GetDFlashConfig();
    impl->dflash_taps = dcfg.target_layer_ids;
    impl->drafter_block =
        std::min<std::uint32_t>(dcfg.block_size, Session::kMaxVerifyRows);
    impl->dflash_window =
        dcfg.sliding_window == 0
            ? options.max_context
            : std::min<std::uint32_t>(dcfg.sliding_window, options.max_context);
    if (!impl->dflash_taps.empty() &&
        impl->dflash_taps.back() >= impl->config.num_layers) {
      Fail(error_msg, "the drafter taps exceed the Laguna layer count");
      return nullptr;
    }
  }
  // Release the GGUF mmap now that all weights are on the device: keeping it
  // alive pins ~model-size host pages and trips the SVM/resident-memory
  // limit. Neither the DFlash drafter (its own reader, opened above) nor
  // anything else needs Laguna's own reader past this point.
  impl->reader.reset();
  model->state_ = std::move(impl);
  return model;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  std::unique_ptr<Session> session(new Session());
  session->impl_->model = state_;
  if (!session->impl_->Init(max_context, error_msg))
    return nullptr;
  if (HasDflash()) {
    session->impl_->dflash = hip::QwenDFlashGpuExecutor::Create(
        state_->dflash_model,
        std::min<std::uint32_t>(max_context, state_->options.max_context),
        error_msg);
    if (!session->impl_->dflash) {
      Fail(error_msg, "Laguna: drafter executor creation failed");
      return nullptr;
    }
    const std::size_t taps = state_->dflash_taps.size();
    if (taps != 0 &&
        !session->impl_->tap_capture.Allocate(
            std::size_t{state_->dflash_window + Session::kMaxVerifyRows} *
            taps * state_->config.hidden_size)) {
      Fail(error_msg, "Laguna: drafter capture allocation failed");
      return nullptr;
    }
  }
  const std::int32_t warm[kWarmTokens] = {};
  for (const std::uint32_t width : {1U, kWarmTokens}) {
    if (width > max_context)
      break;
    if (!session->impl_->Forward({warm, width}, false, width, error_msg)) {
      return nullptr;
    }
    session->impl_->position = 0;
    if (session->impl_->dflash != nullptr) {
      session->impl_->dflash->Reset();
      session->impl_->tap_pending = 0;
      session->impl_->tap_pending_pos = 0;
      session->impl_->tap_pending_first = 0;
    }
  }
  return session;
}

const Config& Model::config() const noexcept {
  return state_->config;
}

bool Model::HasDflash() const noexcept {
  return state_->dflash_model != nullptr;
}

std::uint32_t Model::DrafterBlockSize() const noexcept {
  return state_->drafter_block;
}

std::size_t Model::ResidentBytes() const noexcept {
  return state_->device->resident_bytes();
}

}  // namespace gufo::models::laguna::gpu
