#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/core/session_mode.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

class Executor;
class SnapshotAllocator;
struct ArgmaxCandidate;
struct SnapshotHeader;
struct SnapshotRegistry;
struct SnapshotLineage;
struct PrefillCheckpoint;

struct SnapshotBlock;
struct SnapshotChunk;

/// Mutable checkpoint state is copied immediately; committed rows borrow
/// append-only device storage until a rewind needs their shared device copy.
/// Byte readers assemble the payload under the same lock used by writers.
class SnapshotState {
public:
  void Materialize() const;
  [[nodiscard]] std::uint64_t DeviceBytes() const;

private:
  friend class Executor;
  friend class Session;
  friend struct SnapshotRegistry;
  struct Saved {
    std::uint64_t offset;
    std::uint64_t bytes;
    std::shared_ptr<const SnapshotChunk> chunk;
  };
  struct Region {
    void* device;
    std::uint8_t* host;
    std::uint64_t bytes;
    std::uint64_t live_bytes;
    std::uint64_t pending_bytes;
    std::uint64_t row_bytes;
    bool draft;
    bool pooled;
    std::vector<Saved> saved;
  };
  void Preserve(std::uint32_t position, std::uint32_t mtp_position,
                std::uint32_t blocks, std::uint32_t mtp_blocks,
                bool detach = true) const;
  std::shared_ptr<SnapshotRegistry> registry_;
  std::shared_ptr<const SnapshotLineage> lineage_;
  std::shared_ptr<void> storage_;
  mutable std::vector<Region> regions_;
  std::uint8_t* payload_ = nullptr;
  mutable std::shared_ptr<void> mutable_storage_;
  std::uint8_t* mutable_device_ = nullptr;
  std::uint64_t mutable_offset_ = 0;
  std::uint64_t mutable_bytes_ = 0;
  // The other small non-row regions (indexer ring, draft state, kept rows),
  // also frozen on the device until a byte reader needs them. Copying them
  // into untouched pageable payload pages cost several ms per capture.
  struct Extra {
    std::uint64_t offset;
    std::uint64_t bytes;
    std::uint8_t* device;
  };
  mutable std::vector<Extra> extras_;
  mutable std::shared_ptr<void> extra_storage_;
  std::uint64_t extra_bytes_ = 0;
};

/// Per-sequence state on the device: recurrent SSM state, KV and indexer
/// caches, PLE conv history, plus the host-side n-gram window. A
/// speculative forward additionally keeps per-token snapshots of every
/// recurrent buffer so the batch can be cut back to its accepted prefix.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Frontier of the target residual rows captured by the most recent
  /// forward, valid only while `rows` is nonzero.
  [[nodiscard]] std::uint32_t dflash_capture_position() const noexcept {
    return dflash_capture_pos_;
  }
  [[nodiscard]] std::uint32_t dflash_capture_rows() const noexcept {
    return dflash_capture_rows_;
  }
  /// Drops every token; the next Forward starts at position 0. Cache slot
  /// reassignment may retain bounded MTP scratch and its recorded graphs.
  void Reset(bool reuse_mtp = false);
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool CheckCancellation(std::string* error) const;
  [[nodiscard]] bool Cancelled() const noexcept { return cancelled_; }
  [[nodiscard]] std::uint64_t MutationEpoch() const noexcept {
    return mutation_epoch_;
  }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  [[nodiscard]] bool OwnsSnapshot(
      const SnapshotState& snapshot) const noexcept {
    return snapshots_ == snapshot.registry_;
  }
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt,
                       std::shared_ptr<qwen::vision::Encoder> encoder,
                       hipStream_t stream);
  void RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                           hipStream_t stream);
  [[nodiscard]] const qwen::vision::RopeLayout& VisionLayout() const noexcept {
    return vision_input_.layout();
  }

private:
  friend class Executor;
  friend struct PrefillCheckpoint;
  friend class gufo::models::qwen38_flash_next::ContinuationAdapter;
  Session() = default;

  struct LinearState {
    float* conv_state{nullptr};    ///< [kernel-1][channels]
    float* state{nullptr};         ///< [v_heads][d][d]
    RollbackRows conv_snapshots;   ///< [max_spec-1][kernel-1][channels]
    RollbackRows state_snapshots;  ///< [max_spec-1][v_heads][d][d]
  };
  struct AttentionState {
    const qwen::vision::DeviceRope* rope{nullptr};
    __half* k_cache{nullptr};  ///< [max_context][kv_heads*d]
    __half* v_cache{nullptr};  ///< [max_context][kv_heads*d]
    float* index_k{nullptr};   ///< [index_capacity_][indexer_dim] raw ring
    __half* block_k{nullptr};  ///< [max_context/ratio][indexer_dim]
  };
  /// Per-launch values the kernels read from device memory, so a captured
  /// graph replays at any position.
  struct Control {
    std::uint32_t position;      ///< first position of the trunk batch
    std::uint32_t blocks;        ///< indexer blocks pooled so far
    std::uint32_t mtp_position;  ///< first position of the draft batch
    std::int32_t hidden_row;     ///< kept trunk row the draft reads, or -1
    std::uint32_t mtp_blocks;    ///< complete predictor indexer blocks
  };
  struct MtpState {
    __half* k_cache{nullptr};
    __half* v_cache{nullptr};
    float* index_k{nullptr};
    __half* block_k{nullptr};
    std::uint32_t blocks{0};
    float* h{nullptr};  ///< [hc_dim] wide residual handed to the next draft
    float* target_hidden{nullptr};  ///< last max_speculative trunk rows
    std::uint32_t position{0};
    bool residual_valid{false};
  };

  mutable bool cancelled_{false};
  std::uint64_t mutation_epoch_{0};
  const Executor* owner_{nullptr};
  qwen::vision::DeviceInput vision_input_;
  std::uint32_t max_context_{0};
  bool mtp_enabled_{false};
  /// Speculative session with an MTP predictor loaded; the MTP block state is
  /// only meaningful when this is set.
  bool mtp_head_{false};
  std::uint32_t index_capacity_{0};  ///< power-of-two raw indexer ring rows
  std::uint32_t position_{0};
  std::uint32_t dflash_capture_pos_{0};
  std::uint32_t dflash_capture_rows_{0};
  ContinuationHooks* continuation_hooks_{nullptr};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  float* ple_history_{nullptr};  ///< [PleConvHistory()][hc_dim]
  RollbackRows ple_snapshots_;   ///< [max_spec-1][PleConvHistory()][hc_dim]
  NgramHistory ngram_;
  std::vector<NgramHistory> ngram_snapshots_;
  std::uint32_t spec_base_{0};    ///< position before the speculative batch
  std::uint32_t spec_tokens_{0};  ///< tokens of the pending speculative batch
  MtpState mtp_;
  Control* control_{nullptr};
  std::uint32_t blocks_{0};  ///< host shadow of Control::blocks
  /// Captured decode graphs by batch shape, and the shapes that ran eagerly
  /// once (the GEMM tier's arena must be grown before capture).
  std::unordered_map<std::uint64_t, hipGraphExec_t> graphs_;
  std::unordered_set<std::uint64_t> warmed_;
  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  std::size_t rollback_bytes_{0};
  std::function<bool()> is_cancelled_;
  std::vector<float*> rollback_allocations_;
  std::uint32_t rollback_depth_{0};
  void TrimRollback(std::uint32_t depth) noexcept;
  mutable std::shared_ptr<SnapshotRegistry> snapshots_;
  mutable std::shared_ptr<const SnapshotLineage> snapshot_lineage_;
  mutable std::uint32_t snapshot_position_{0};
  mutable std::uint32_t snapshot_mtp_position_{0};
  void PreserveSnapshots(std::uint32_t position,
                         std::uint32_t mtp_position) const;
};

/// A frozen mutable frontier with append-only caches owned by its live session.
/// It is consumed before another forward can reuse the checkpoint arena.
struct PrefillCheckpoint {
  ~PrefillCheckpoint();
  std::unique_ptr<Session> state;
  Session* source = nullptr;
  std::uint32_t tokens = 0;
  std::uint32_t hidden_rows = 0;
  std::vector<float> logits;
};

/// Runs the trunk graph on the GPU for one session at a time. Buffers are
/// sized once for `max_batch` tokens; longer prompts are fed in chunks.
class Executor {
public:
  struct Options {
    std::uint32_t max_batch{1};
    /// Rows of logits (and hidden states) a Forward call may return.
    std::uint32_t max_logit_rows{1};
    /// Longest speculative batch; bounds the recurrent snapshot storage.
    std::uint32_t max_speculative{1};
    /// Last trunk residual rows kept for a block-diffusion drafter. Zero
    /// disables the capture.
    std::uint32_t dflash_rows{0};
    /// Zero-based completed trunk layers whose residual mean is captured, in
    /// the drafter's feature order.
    std::vector<std::uint32_t> dflash_layers;
    /// YaRN frequency-scaling factor for the trunk's RoPE. 1 disables YaRN
    /// (plain theta, byte-identical to the pre-YaRN path); a caller that
    /// requests a context above the model's native length passes
    /// `requested_len / native_len`.
    float yarn_factor{1.0F};
  };

  ~Executor();
  // ForwardPair uses lazily allocated bank B buffers; they are freed by
  // the destructor below.
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, NgramTable* ngram, Options options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr) const;
  [[nodiscard]] bool EnsureRollback(Session& session, std::uint32_t depth,
                                    std::string* error_msg) const;
  [[nodiscard]] std::size_t SessionBytes(
      core::SessionMode mode, std::uint32_t max_context,
      std::uint32_t rollback_depth) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;
  /// Mean residual rows captured for the block-diffusion drafter, laid out as
  /// [row][dflash_layers in order][hidden]. Null when capture is disabled.
  [[nodiscard]] const float* dflash_capture() const noexcept {
    return dflash_capture_;
  }

  enum class ForwardMode { kDecode, kVerify, kPrefill };
  [[nodiscard]] std::unique_ptr<PrefillCheckpoint> MakePrefillCheckpoint(
      Session& session, std::span<const std::int32_t> tokens,
      std::uint32_t prefix, std::string* error_msg) const;

  /// Appends `tokens` (at most max_batch) to the session and returns the
  /// logits of the last `n_logits` tokens in `logits` (n_logits * vocab
  /// floats, host memory). A null `logits` keeps the rows on the GPU for
  /// verification. The final wide residual of those tokens stays on
  /// the device for MtpForward. kVerify permits Rollback (at most
  /// max_speculative rows). kPrefill uses consistent prompt arithmetic at
  /// every chunk width. A kPrefill call may name the `next` chunk of the same
  /// prompt: once its own n-gram rows have arrived it starts reading the
  /// next chunk's, so that read overlaps this chunk's layers. The rows are
  /// used only if the next batch has exactly those tokens.
  [[nodiscard]] bool ForwardPair(Session& session,
                                 std::span<const std::int32_t> tokens,
                                 std::uint32_t n_logits, float* logits,
                                 std::string* error_msg);
  [[nodiscard]] bool CanForwardPair(const Session& session, std::size_t n,
                                    std::uint32_t n_logits, std::uint32_t* nA,
                                    std::uint32_t* nB,
                                    std::string* error_msg = nullptr) const;
  [[nodiscard]] bool Forward(Session& session,
                             std::span<const std::int32_t> tokens,
                             std::uint32_t n_logits, float* logits,
                             ForwardMode mode, std::string* error_msg,
                             PrefillCheckpoint* checkpoint = nullptr,
                             std::span<const std::int32_t> next = {}) const;

  struct BatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    bool speculative;
  };
  /// Packs independent decode chains for shared projections. Every chain
  /// keeps its own recurrent/KV state and the decode arithmetic (<=8 rows).
  /// Logit rows remain in batch order until SelectBatchLogits is called.
  [[nodiscard]] bool ForwardBatch(std::span<const BatchItem> items,
                                  std::string* error_msg) const;
  [[nodiscard]] bool SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                       float* logits,
                                       std::string* error_msg) const;
  /// Rows a single session may verify at once when every proposal after the
  /// anchor is copied from its history (ForwardWide), or 0 without MTP.
  static constexpr std::uint32_t kWideRows = 64;
  [[nodiscard]] std::uint32_t max_copy_rows() const noexcept {
    return has_mtp() ? std::min({kWideRows, options_.max_batch, copy_rows_cap_})
                     : 0;
  }
  /// Caps max_copy_rows() before any session is created, so the wide
  /// rollback rows fit beside the configured sessions.
  void LimitCopyRows(std::uint32_t rows) noexcept { copy_rows_cap_ = rows; }
  /// Verifies one session's chain of up to max_copy_rows() tokens in one
  /// pass. The shared projections read their weights once for every row;
  /// recurrent state, attention and PLE run as consecutive segments of at
  /// most eight rows, so every row has the arithmetic of an ordinary
  /// verification. Logit rows are selected as after ForwardBatch, and
  /// Rollback() cuts the chain back to any kept prefix.
  [[nodiscard]] bool ForwardWide(Session& session,
                                 std::span<const std::int32_t> tokens,
                                 std::string* error_msg) const;
  /// Moves kept trunk hidden row `row` to row 0 (the draft block caught up
  /// to it, so the rows before it are no longer needed).
  [[nodiscard]] bool KeepHiddenRow(Session& session, std::uint32_t row,
                                   std::string* error_msg) const;
  [[nodiscard]] bool ReadVerificationRows(std::uint32_t row,
                                          std::span<float> logits,
                                          std::string* error_msg) const;

  /// Keeps the first `keep` (1..n) tokens of the last speculative batch and
  /// discards the rest. If `logits` is supplied, copies the kept frontier
  /// into it using the same synchronization as rollback.
  [[nodiscard]] bool Rollback(Session& session, std::uint32_t keep,
                              std::string* error_msg,
                              float* logits = nullptr) const;

  /// Runs the draft block over `tokens` (at most max_batch) at the session's
  /// MTP position. The hidden input of token i is the trunk residual of row
  /// `hidden_row + i` of the last Forward batch, or, with hidden_row < 0
  /// (single token), the draft block's own residual from the previous call.
  /// Only requested outputs are computed. Production requests a greedy
  /// token or compact candidate logits. Catch-up skips the output head
  /// when only the draft state is needed.
  struct MtpOutput {
    std::int32_t* token{nullptr};
    MtpCandidateLogits* candidates{nullptr};
    MtpTrace* trace{nullptr};  ///< final-row diagnostic; disables graph capture
    /// Prefill needs only persistent KV. A subsequent full forward with
    /// known trunk hidden rows is required before heads/recursive proposals.
    bool kv_only{false};
    /// Score every output row, not the drafting subset.
    bool full_vocab{false};
  };
  struct MtpHeadItem {
    Session* session;
    MtpOutput output;
  };
  struct MtpBatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    std::int32_t hidden_row;
  };
  /// Runs independent short predictor chains with shared projections and
  /// private attention caches, positions and carried hidden states.
  [[nodiscard]] bool MtpForwardBatch(std::span<const MtpBatchItem> items,
                                     std::string* error_msg) const;
  /// Projects each session's carried draft hidden state with shared weights.
  [[nodiscard]] bool MtpHeads(std::span<const MtpHeadItem> items,
                              std::string* error_msg) const;
  [[nodiscard]] bool MtpForward(Session& session,
                                std::span<const std::int32_t> tokens,
                                std::int32_t hidden_row, MtpOutput output,
                                std::string* error_msg,
                                const float* hidden_source = nullptr) const;
  /// Leading kept trunk rows, for independent predictor qualification.
  [[nodiscard]] bool CopyTrunkHidden(const Session& session,
                                     std::span<float> hidden,
                                     std::string* error_msg) const;

  /// Greedy verification keeps full logit rows on the GPU, including penalties.
  [[nodiscard]] bool GreedyMtpPredictions(
      std::span<ArgmaxCandidate> predictions,
      const sampling::SamplerState& sampler,
      std::span<const std::int32_t> drafts, std::string* error_msg) const;
  [[nodiscard]] static bool CanCompactMtpVerification(
      const sampling::SamplerState& sampler);
  /// Whether a sampled verification of `rows` rows can stay compact.
  [[nodiscard]] static bool CanCompactWideVerification(
      const sampling::SamplerState& sampler, std::size_t rows);
  /// Candidate spans remain valid until this executor next verifies a
  /// request. One transfer serves all rows; no full CPU rows are reconstructed.
  [[nodiscard]] bool SampledMtpCandidates(
      std::span<std::span<const sampling::LogitCandidate>> output,
      const sampling::SamplerState& sampler,
      std::span<const std::int32_t> drafts, std::string* error_msg) const;

  /// A session's complete context as one host byte payload: recurrent and
  /// PLE state, KV and indexer caches up to the position, and the draft
  /// block's caches plus the `hidden_rows` most recent kept trunk rows.
  /// The payload restores into any session of this executor whose context
  /// holds the position; the pending speculative batch must be empty.
  struct SnapshotInfo {
    std::uint32_t position{0};
    std::uint32_t hidden_rows{0};
  };
  [[nodiscard]] std::uint64_t SnapshotBytes(const Session& session,
                                            std::uint32_t hidden_rows) const;
  [[nodiscard]] std::uint64_t PrefillCheckpointBytes(
      const Session& session, std::uint32_t position) const;
  [[nodiscard]] bool SaveSnapshot(
      const Session& session, std::uint32_t hidden_rows,
      std::span<std::uint8_t> payload, std::string* error_msg,
      std::shared_ptr<SnapshotState>* deferred = nullptr,
      std::shared_ptr<void> storage = {}) const;
  /// Reuse at most the rollback rows needed by the restored operation;
  /// restoration never grows scratch. Callers derive this bound from the
  /// restored policy (or the concrete verifier width in a diagnostic).
  [[nodiscard]] bool RestoreSnapshot(
      Session& session, std::span<const std::uint8_t> payload,
      SnapshotInfo* info, std::string* error_msg, std::uint32_t next_drafts = 0,
      const SnapshotState* deferred = nullptr) const;

  /// Rewinds the draft block's own context.
  void MtpRewind(Session& session, std::uint32_t position) const {
    if (!session.mtp_head_)
      return;
    if (session.continuation_hooks_ && position < session.mtp_.position)
      session.continuation_hooks_->BeforeWrite(
          true, position, session.mtp_.position,
          std::min(session.mtp_.blocks, position / config().compress_ratio),
          session.mtp_.blocks);
    if (session.mtp_.position != position)
      session.mtp_.residual_valid = false;
    session.mtp_.position = position;
    session.mtp_.blocks =
        std::min(session.mtp_.blocks, position / config().compress_ratio);
  }
  [[nodiscard]] std::uint32_t MtpPosition(
      const Session& session) const noexcept {
    return session.mtp_.position;
  }

  [[nodiscard]] const Config& config() const noexcept {
    return model_->config();
  }
  [[nodiscard]] std::uint32_t max_batch() const noexcept {
    return options_.max_batch;
  }
  [[nodiscard]] std::uint32_t max_speculative() const noexcept {
    return options_.max_speculative;
  }
  [[nodiscard]] bool has_mtp() const noexcept { return model_->has_mtp(); }
  /// YaRN inverse-frequency device table and attention-score scale this
  /// trunk session resolved at Create() - null/1.0 when YaRN is off. Lets
  /// a DFlash drafter attached to this session keep its own RoPE aligned
  /// with the trunk's scaled positions past native context (this worktree
  /// only - see QwenDFlashGpuExecutor::SetYarnScaling).
  [[nodiscard]] const float* yarn_inv_freq() const noexcept {
    return d_yarn_inv_;
  }
  [[nodiscard]] float yarn_mscale() const noexcept { return yarn_mscale_; }

private:
  Executor() = default;

  /// Visits every device region of a snapshot in payload order with
  /// (device pointer or null when sizing, payload offset, bytes, name).
  /// Returns the payload size, or 0 once a visit failed.
  template<typename Visit>
  static std::uint64_t WalkSnapshot(const SnapshotHeader& h,
                                    const Session* session, Visit&& visit);

  /// An activation batch quantized once for the decode GEMVs; `data` is
  /// null when the batch is wide enough for the tiled path.
  struct Q8Input {
    const float* x;
    const void* data;
    std::uint32_t n;
    std::uint32_t k;
  };

  struct PreparedNpu {
    bool valid{false};
    bool linear{false};
    std::uint64_t ticket{0};
    std::uint32_t groups{0};
    std::uint32_t slices{0};
    const int* c_ids{nullptr};
  };

  bool Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                Q8Input* q, std::string* error_msg) const;
  /// Q8_1 rows of a small-batch input that its producer kernel already
  /// wrote (byte-identical to Quantize's), or null.
  const void* ReadyQ8(const float* x, std::uint32_t n_tokens,
                      std::uint32_t k) const;
  /// Forgets the producer-written Q8_1 rows (their sources are rewritten).
  void DropReadyQ8() const;
  bool Dense(const DeviceTensor& w, const Q8Input& q, float* out,
             std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const float* x, float* out,
             std::uint32_t n_tokens, std::string* error_msg) const;
  /// out = (up . x) * silu(gate . x); s_.shexp_gate is scratch.
  /// With `down` (the projection that consumes the result), a wide batch
  /// leaves only that projection's staged input form (F16 rows in
  /// s_.x_half or the tiled Q8 layout in s_.x_q8t, registered in the input
  /// cache); `out` then holds the gate projection, not the result.
  bool GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                  const float* x, float* out, std::uint32_t n_tokens,
                  const DeviceTensor* down, std::string* error_msg) const;
  /// Reuse or populate the F16 activation staging buffer.
  void PrepareHalfInput(const float* x, std::uint32_t rows,
                        std::uint32_t cols) const;
  /// Whether a dense Q4_K/Q5_K/Q5_1 matrix fits the F16 weight scratch.
  bool Expandable(const DeviceTensor& w) const;
  /// Expands w into s_.w_half (stream-ordered); nullptr if it cannot.
  const __half* ExpandHalf(const DeviceTensor& w) const;
  /// Whether a wide dense Q8_0 projection takes the F16 WMMA GEMM.
  bool DenseF16Route(const DeviceTensor& w, std::uint32_t n_tokens) const;
  void RoutedHints(const DeviceTensor& w, std::uint32_t n_tokens) const;
  /// Reads the routing of the current batch back and derives the tile
  /// hints for its expert GEMMs (tiled batches only).
  bool RouteHints(std::uint32_t n_tokens, std::string* error_msg) const;
  bool NanProbe(const char* tag, const void* data, std::size_t count,
                bool is_half, std::uint32_t n_tokens, std::uint32_t il,
                std::string* error_msg) const;
  void ProbeRouter(const char* tag, std::uint32_t n_tokens) const;
  bool NanProbeRows(const char* tag, const float* data, std::uint32_t rows,
                    std::uint32_t cols, std::uint32_t il,
                    std::string* error_msg) const;
  bool NanProbeRowsHalf(const char* tag, const void* data, std::size_t rows,
                        std::size_t cols, std::uint32_t il,
                        std::string* error_msg) const;
  /// Routed gate/up projections followed by SwiGLU.
  bool GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                    const float* x, const std::int32_t* ids, float* out,
                    std::uint32_t n_tokens, std::uint32_t n_used,
                    std::string* error_msg) const;
  bool Experts(const DeviceTensor& w, const float* x, const std::int32_t* ids,
               float* out, std::uint32_t n_rows, std::uint32_t n_used,
               std::uint32_t n_tokens, std::string* error_msg) const;
  /// `normed` says the previous Combine already produced m's grouped norm
  /// of res (F32 in s_.xn, or F16 plus tiled Q8 on the wide route).
  bool HcMix(const DeviceMixer& m, const float* res, bool normed, float* mixed,
             float* inject, std::uint32_t n_tokens, std::string* error_msg,
             bool half_only = false) const;
  /// Share mixer projections across decode requests without selecting the
  /// prefill arithmetic. Injection rows are compact across the whole batch.
  bool HcMixBatch(const DeviceMixer& m, const float* res, bool normed,
                  float* mixed, float* inject, std::uint32_t rows,
                  std::string* error) const;
  void CombineBatch(float* res, const float* gamma, std::uint32_t rows) const;
  /// Residual update by the block output plus the grouped norm for the next
  /// mixer (`gamma`); wide batches write it as F16 and tiled Q8.
  /// A non-null `next` (the mixer that reads the norm) also gets its inject
  /// logits formed by the combine when the F16 route's mixer would otherwise
  /// run the separate inject kernel; HcMix then skips that kernel.
  void Combine(void* res, const float* gamma, std::uint32_t n_tokens,
               bool res_half = false, const DeviceMixer* next = nullptr) const;
  [[nodiscard]] bool InjectFusable(const DeviceMixer& m,
                                   std::uint32_t n_tokens) const;
  /// Whether HcMix runs a decode batch of `m` as one HcMixQ8Fused launch.
  [[nodiscard]] bool FusedMixer(const DeviceMixer& m,
                                std::uint32_t n_tokens) const;
  /// Hashes the batch's n-gram rows and starts reading them from disk, so
  /// the read overlaps the layers before the PLE one.
  bool PleFetch(Session& s, std::span<const std::int32_t> tokens,
                bool speculative, std::string* error_msg) const;
  bool WaitPle(std::string* error_msg) const;
  /// Starts reading `next`'s rows into the prefetch buffer. Best effort: a
  /// prefetch that cannot start is skipped.
  void PrefetchPle(const Session& s, std::span<const std::int32_t> next) const;
  /// Waits for a prefetch in flight before another read; its rows stay
  /// claimable.
  void FinishPrefetch() const;
  bool Ple(const DeviceLayer& l, Session& s, std::uint32_t n_tokens, float* res,
           bool speculative, std::string* error_msg,
           bool embeddings_ready = false,
           PrefillCheckpoint* checkpoint = nullptr) const;
  bool LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                       const float* x, float* out, std::uint32_t n_tokens,
                       bool speculative, std::string* error_msg,
                       bool projections_ready = false,
                       bool project_output = true,
                       GdnCheckpoint checkpoint = {}) const;
  /// `pos`/`first_block` are device values; `start_pos` and `pool_grid`
  /// are their host-side counterparts for the eager-only decisions.
  bool Attention(const DeviceLayer& l, Session::AttentionState& s,
                 const float* x, float* out, std::uint32_t n_tokens,
                 const std::uint32_t* pos, const std::uint32_t* first_block,
                 std::uint32_t start_pos, std::uint32_t pool_grid,
                 std::uint32_t max_context, bool sparse, std::string* error_msg,
                 bool last_only = false, bool projections_ready = false,
                 bool project_output = true,
                 std::uint32_t checkpoint_tokens = 0) const;
  bool Moe(const DeviceLayer& l, const float* x, float* out,
           std::uint32_t n_tokens, std::string* error_msg,
           bool last_only = false) const;
  /// Runs routed experts after the router and shared expert are ready.
  bool MoeExperts(const DeviceLayer& l, const float* x, float* out,
                  std::uint32_t n_tokens, std::string* error_msg) const;
  bool MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t rows, std::string* error) const;
  /// Selects the greedy token or compact candidates from full MTP logits.
  bool MtpHead(const DeviceMixer& head, const float* res, bool token,
               bool candidates, bool full_vocab, std::string* error_msg) const;

  /// Enqueues one trunk batch (control and token upload through logits).
  bool ForwardBody(Session& session, std::uint32_t n, std::uint32_t n_logits,
                   bool download_logits, bool speculative, bool sparse,
                   std::uint32_t start_pos, std::uint32_t pool_grid,
                   std::uint32_t first_layer, std::uint32_t end_layer,
                   std::string* error_msg,
                   PrefillCheckpoint* checkpoint = nullptr) const;
  bool ForwardBodyRange(Session& session, std::uint32_t n,
                        std::uint32_t n_logits, bool download_logits,
                        bool speculative, bool sparse, std::uint32_t start_pos,
                        std::uint32_t pool_grid, std::uint32_t first_layer,
                        std::uint32_t end_layer, std::string* error_msg,
                        PrefillCheckpoint* checkpoint, bool initial_normed,
                        bool initial_res_half, bool convert_at_entry,
                        bool convert_at_exit, bool upload_control,
                        bool* out_normed, bool* out_res_is_half,
                        bool premixed = false,
                        const PreparedNpu* prepared = nullptr) const;
  bool MtpBody(Session& session, std::uint32_t n, std::uint32_t pos, bool token,
               bool candidates, std::string* error_msg, std::uint32_t pool_grid,
               const float* hidden_source, MtpTrace* trace, bool kv_only,
               bool full_vocab) const;
  /// Runs `body` eagerly, or as the session's captured graph for `key`
  /// when `graph` is set. A prefix may leave its work queued so the host
  /// can wait for disk reads while the GPU computes it.
  /// `label` names the pass in the GUFO_GRAPH_LOG=1 capture/launch log.
  bool Run(Session& session, std::uint64_t key, bool graph,
           const std::function<bool()>& body, std::string* error_msg,
           bool synchronize = true, const char* label = "trunk") const;

  const DeviceModel* model_{nullptr};
  NgramTable* ngram_{nullptr};
  Options options_;
  [[nodiscard]] std::shared_ptr<SnapshotAllocator> GetSnapshotAllocator() const;
  mutable std::once_flag snapshot_allocator_once_;
  mutable std::shared_ptr<SnapshotAllocator> snapshot_allocator_;
  mutable std::shared_ptr<void> checkpoint_storage_;
  hipStream_t stream_{nullptr};
  hipEvent_t counts_ready_{nullptr};
  hipblasHandle_t blas_{nullptr};
  std::unique_ptr<BlasLt> blaslt_;
  /// Device YaRN inverse-frequency table (`rotary_dim/2` floats) and its
  /// attention-score scale, uploaded once in Create. `d_yarn_inv_` is null when
  /// YaRN is disabled, so every RoPE kernel falls back to plain theta math
  /// (byte-identical to the pre-YaRN path).
  float* d_yarn_inv_{nullptr};
  float yarn_mscale_{1.0F};

  // Scratch, sized for max_batch tokens. Names follow reference.cpp.
  struct Scratch {
    std::int32_t* tokens;
    void* x_half;  ///< activations narrowed to the weight's 16-bit type
    std::size_t half_capacity{0};  ///< element capacity of x_half
    void* w_half;  ///< F16 copy of one dense K-quant/Q5_1 matrix (wide batches)
    std::size_t w_half_capacity{0};  ///< element capacity of w_half
    void* x_q8[2];  ///< Q8_1 activations of a decode batch, alternating
    /// Q8_1 rows of a decode batch written by the kernel that produces
    /// them: xn by HcCombine, lo by SiluScale, mixed by the mix epilogue.
    void* xn_q8_1;
    void* lo_q8_1;
    void* mixed_q8_1;
    void* gdn_q8_1;  ///< by the GDN epilogue (ssm_out's input)
    void* ctx_q8_1;  ///< by the attention output gate (attn_out's input)
    void* x_q8t;     ///< tiled Q8 activations of a wide batch (W8A8 route)
    float* res;
    __half*
        res_h;  ///< F16 residual of large wide-prefill chunks (see ForwardBody)
    float* xn;
    __half* xn_half;  ///< xn as F16 on the F16 mixer input route
    void* xn_q8t;     ///< xn as tiled Q8 for the W8A8 mixer down projection
    float* lo;
    float* hc_gate;
    float* mixed;
    float* inject;
    float* block_out;
    // linear attention
    float* qkv;
    float* z;
    float* qkvz;  ///< [t][qkv | z] from the stacked projection
    float* alpha_beta;
    float* conv_scratch;
    float* qn;
    float* kn;
    float* gdn_raw;
    float* gdn_out;
    // attention
    float* qg;  ///< [t][q|gate (; k ; v)]
    float* q;
    float* attn_gate;
    float* k;
    float* v;
    float* iq;
    float* ik;
    std::uint32_t* mask;
    float* scores;
    float* ctx;
    float* attn_partials;  ///< split-key partials of a narrow batch
    // ple
    float* ple_emb;
    float* ple_key;
    float* ple_value;
    float* ple_query;
    float* ple_gated;
    float* ple_norm;
    float* ple_conv;
    float* ple_history_scratch;
    // moe
    float* router;
    std::int32_t* ids;
    /// Expert 0 for the dense quantized head, which the MoE vector kernel
    /// reads as a one-expert weight.
    std::int32_t* dense_ids;
    std::uint32_t* expert_counts;
    // Routed WMMA route: 16-row padded bucket bounds, scatter cursors, the
    // compact row -> (token, slot) maps and the tiled Q8 gathered rows.
    std::int32_t* routed_bounds;
    std::int32_t* routed_cursors;
    std::int32_t* rows_token;
    std::int32_t* rows_slot;
    std::int32_t* routed_tiles;  ///< (expert | tile << 16) per launched tile
    float* weights;
    float* gate_e;
    float* up_e;
    float* down_e;
    float* shexp_gate;
    float* shexp_up;
    float* shexp_out;
    /// Shared-expert SwiGLU rows narrowed for the F16 down projection, so
    /// s_.x_half keeps the routed experts' token rows.
    __half* shexp_half;
    // head and hidden rows kept for the draft block
    float* logits;
    // mtp
    float* mtp_h;
    float* mtp_embd;
    float* mtp_eproj;
    float* mtp_res;
    ArgmaxCandidate* mtp_argmax;
    std::int32_t* mtp_token;
    std::uint32_t* mtp_ids;
    std::uint32_t* mtp_scratch_ids;
    float* mtp_scores;
  };
  mutable Scratch s_{};
  [[nodiscard]] Scratch RowScratch(const Scratch& base,
                                   std::uint32_t offset) const;
  void UseScratch(const Scratch& scratch) const;

  void SetScratchHost(const Scratch& scratch, Session::Control* control,
                      std::int32_t* tokens);
  [[nodiscard]] bool AllocateScratch(std::size_t T, Scratch* out,
                                     const Scratch* base,
                                     std::string* error_msg);
  [[nodiscard]] bool EnsureBankBScratch(std::string* error_msg);
  mutable Scratch bank_b_scratch_{};
  bool bank_b_allocated_{false};
  Session::Control* bank_b_control_host_{nullptr};
  std::int32_t* bank_b_tokens_host_{nullptr};
  /// Bank B's half of the routed tile/counts staging. HIP does not order
  /// copy-engine transfers against each other within one stream, so a
  /// `hipEventSynchronize` on the counts event does not prove bank A's tile
  /// map upload has drained; the two banks must not share the pinned source
  /// buffers (see FIX-ple-prefetch-race.md). Selected by `route_bank_`.
  std::uint32_t* bank_b_counts_host_{nullptr};
  std::int32_t* bank_b_tiles_host_{nullptr};
  hipEvent_t bank_b_counts_ready_{nullptr};
  /// 0 while A/default staging is live, 1 while bank B's is; switched only by
  /// `SetScratchHost`, which drains the stream first so no in-flight copy of
  /// the outgoing bank's pinned buffers can read the new contents.
  mutable std::uint32_t route_bank_{0};
  std::uint32_t* RouteCountsHost() const {
    return route_bank_ == 1 ? bank_b_counts_host_ : counts_host_;
  }
  std::int32_t* RouteTilesHost() const {
    return route_bank_ == 1 ? bank_b_tiles_host_ : tiles_host_;
  }
  hipEvent_t RouteCountsReady() const {
    return route_bank_ == 1 ? bank_b_counts_ready_ : counts_ready_;
  }
  bool DenseBatch(const DeviceTensor& w, const float* x, float* out,
                  std::uint32_t rows, std::string* error_msg) const;
  bool GatedDenseBatch(const DeviceTensor& up, const DeviceTensor& gate,
                       const float* x, float* out, std::uint32_t rows,
                       std::string* error) const;
  bool QuantizeBatch(const float* x, std::uint32_t rows, std::uint32_t cols,
                     std::string* error) const;
  bool AllocateBatch(std::string* error_msg) const;
  /// Shared transient storage, grown only for penalty-aware verification.
  mutable sampling::TokenPenalty* verification_penalties_{nullptr};
  mutable std::size_t verification_penalty_capacity_{0};
  static constexpr std::size_t kCompactPenaltyLimit = 4096;
  std::uint32_t* verification_ids_{nullptr};
  std::uint32_t* verification_scratch_ids_{nullptr};
  sampling::LogitCandidate* verification_candidates_{nullptr};
  sampling::LogitCandidate* verification_candidates_host_{nullptr};
  mutable std::vector<sampling::TokenPenalty> verification_penalty_host_;
  /// Compact sampled candidates of a wide verification's rows.
  mutable std::vector<std::vector<sampling::LogitCandidate>> wide_candidates_;
  /// Allocated only when concurrent decoding is first requested.
  mutable float* batch_logits_{nullptr};
  mutable Session::Control* batch_controls_{nullptr};
  mutable MtpCandidateLogits* batch_candidates_host_{nullptr};
  // Mapped descriptors, one slice per layer: GPU reads cannot race the host
  // preparing the next layer. ForwardBatch drains before reusing this table.
  mutable GdnBatchItem* batch_gdn_host_{nullptr};
  mutable GdnBatchItem* batch_gdn_{nullptr};
  mutable std::uint32_t batch_rows_{0};
  /// Rollback rows of a wide verification's later segments. Only one wide
  /// pass is pending at a time (Forward/Rollback of the session decoding
  /// alone), so the executor owns them. Segment s >= 1 keeps the state at
  /// its start (restored when every row before it is kept) and its own
  /// snapshot rows.
  struct WideSegment {
    std::vector<RollbackRows> conv;   ///< per layer, linear layers only
    std::vector<RollbackRows> state;  ///< per layer, linear layers only
    std::vector<float*> conv_start;   ///< per layer
    std::vector<float*> state_start;  ///< per layer
    RollbackRows ple;
    float* ple_start{nullptr};
  };
  mutable std::vector<WideSegment> wide_segments_;
  mutable std::vector<void*> wide_allocations_;
  mutable Session::Control* wide_controls_{nullptr};
  /// Rows of the pending wide verification, or 0.
  mutable std::uint32_t wide_rows_{0};
  std::uint32_t copy_rows_cap_{kWideRows};
  bool EnsureWideSegments(std::uint32_t segments, std::string* error) const;
  [[nodiscard]] std::size_t WideSegmentFloats() const;
  bool ForwardBatchImpl(std::span<const BatchItem> items, bool wide,
                        std::string* error) const;
  mutable const float* selected_logits_{nullptr};
  /// Last `dflash_rows` trunk residual means, sized once at Create.
  float* dflash_capture_{nullptr};
  std::uint32_t dflash_capture_rows_{0};
  bool dflash_captures_last_layer_{false};
  std::vector<std::uint32_t> dflash_taps_;
  [[nodiscard]] const float* VerificationLogits() const {
    return selected_logits_ != nullptr ? selected_logits_ : s_.logits;
  }
  std::uint32_t mask_words_{0};
  /// Bounded score scratch, shared by sequential selector chunks.
  std::size_t select_score_floats_{0};
  std::vector<void*> allocations_;
  /// Coherent pinned storage, read directly by the GPU through s_.ple_emb.
  /// Reused only after the preceding forward has drained its stream.
  float* host_emb_{nullptr};
  /// Second pinned n-gram buffer. Prefill reads the next chunk's rows into
  /// whichever buffer the current chunk is not using, so the table reads run
  /// while the GPU computes instead of after it.
  float* host_emb_ahead_{nullptr};
  mutable const float* ple_src_{nullptr};  ///< rows the next upload reads
  struct PleAhead {
    bool valid{false};
    const Session* owner{nullptr};
    std::vector<std::int32_t> tokens;
    NgramHistory before;  ///< session state the read assumed
    NgramHistory after;   ///< session state once these tokens are consumed
    float* buffer{nullptr};
  };
  mutable PleAhead ple_ahead_;
  mutable std::span<const std::int32_t> ple_lookahead_;
  mutable std::vector<std::uint32_t> host_rows_;
  mutable bool ple_pending_{false};
  /// The rows a prefill chunk reads are a pure function of its tokens and
  /// the n-gram history before them, so prefetched rows are claimed only by
  /// a batch with exactly those tokens and that history, whichever session
  /// it belongs to. A claim copies them into host_emb_, whose stable address
  /// is retained by captured graphs.
  struct PlePrefetch {
    bool pending{false};  ///< read in flight
    bool ready{false};    ///< rows complete in `rows`
    bool claimed{false};  ///< WaitPle collects them for the current batch
    NgramHistory before;
    NgramHistory after;
    std::vector<std::int32_t> tokens;
    std::vector<float> rows;
  };
  mutable PlePrefetch prefetch_;
  mutable std::span<const std::int32_t> prefetch_next_;
  // Pinned host staging the launched (or captured) work reads and writes.
  Session::Control* control_host_{nullptr};
  std::int32_t* tokens_host_{nullptr};
  std::uint32_t* counts_host_{nullptr};
  std::int32_t* tiles_host_{nullptr};         ///< routed tile map staging
  mutable std::uint32_t routed_max_rows_{0};  ///< 0 = no readback yet
  mutable std::uint32_t routed_n_tiles_{0};   ///< down projection tiles
  mutable std::uint32_t routed_64_tiles_{0};  ///< appended 64-token tiles
  mutable std::uint32_t routed_pair_offset_{0};
  mutable std::uint32_t routed_pair_tiles_{0};
  mutable std::uint32_t routed_pair_rows_{64};
  /// Per-expert tile maps of wide chunks, in routed_tiles: [0] big buckets,
  /// [1] the rest, for the gate/up (256 / 128 tokens) and down (192 / 128).
  mutable std::uint32_t routed_gate_offset_[2]{0, 0};
  mutable std::uint32_t routed_gate_tiles_[2]{0, 0};
  mutable std::uint32_t routed_down_offset_[2]{0, 0};
  mutable std::uint32_t routed_down_tiles_[2]{0, 0};
  mutable std::uint32_t routed_tile_rows_{48};  ///< token rows per tile
  mutable int routed_tile_cols_{0};
  float* logits_host_{nullptr};
  std::int32_t* mtp_token_host_{nullptr};
  /// Draft-head vocabulary subset (GUFO_MTP_VOCAB): the first rows of the
  /// output matrix plus a tail range holding the special tokens. Verification
  /// still scores the full vocabulary, so only drafting is approximate.
  std::uint32_t mtp_head_rows_{0};
  std::uint32_t mtp_tail_rows_{0};
  std::uint32_t mtp_tail_base_{0};
  [[nodiscard]] std::uint32_t MtpTokenId(
      std::uint32_t index, bool full_vocab = false) const noexcept {
    if (full_vocab)
      return index;
    return mtp_head_rows_ != 0 && index >= mtp_head_rows_
               ? index - mtp_head_rows_ + mtp_tail_base_
               : index;
  }
  MtpCandidateLogits* mtp_candidates_host_{nullptr};
  /// The model geometry allows the wide mixer route (see Combine).
  bool wide_mixer_{false};
  /// Set by Moe when its epilogue is left for the combine that follows.
  mutable bool moe_pending_{false};
  /// Set by GatedDense when s_.shexp_half holds the SwiGLU rows its F16
  /// down projection reads.
  mutable bool shexp_half_ready_{false};
  // What s_.x_q8t / s_.x_half currently hold (input pointer, rows, cols,
  // and the half type), so a projection over the same rows skips its
  // activation pass. Cleared whenever the source buffer is rewritten.
  mutable const float* q8t_src_{nullptr};
  /// NPU staging buffers (activation tiles and six output slices), -1 / empty
  /// when the NPU is not in use.
  /// One A buffer and kNpuSlicesMax C buffers per 4096-token group of a chunk.
  /// Banked by `route_bank_`: the two banks run their forward passes
  /// concurrently and Submit writes results into these host slots
  /// asynchronously, so a shared pool tears the other bank's outputs
  /// (observed as dequant overflow Infs; see FIX-ple-prefetch-race.md).
  std::array<std::vector<int>, 2> npu_a_{};
  std::array<std::vector<int>, 2> npu_a_lo_{};
  std::array<std::vector<int>, 2> npu_c_{};
  /// Device table of the C streams' GPU pointers, [group][kNpuSlicesMax]; the
  /// GDN epilogue reads z from them directly. Null when unavailable.
  std::array<const std::uint8_t**, 2> npu_c_dev_{};
  const std::vector<int>& NpuA() const { return npu_a_[route_bank_]; }
  const std::vector<int>& NpuALo() const { return npu_a_lo_[route_bank_]; }
  const std::vector<int>& NpuC() const { return npu_c_[route_bank_]; }
  const std::uint8_t** NpuCDev() const {
    return const_cast<const std::uint8_t**>(npu_c_dev_[route_bank_]);
  }
  std::array<std::vector<int>, 2> npu_whole_a_{};
  std::array<std::vector<int>, 2> npu_whole_c_{};
  std::uint32_t npu_whole_groups_{0};
  bool npu_whole_enabled_{false};
  struct WholeStreams {
    const std::uint8_t* const* table{nullptr};  ///< [group * 16 + slice]
    std::uint32_t groups{0};
    std::uint32_t group_tokens{0};
  };
  /// Set while a whole-projection linear layer runs: its qkv/z values are read
  /// straight from the NPU streams (fused convolution, z in the epilogue).
  /// Banked: the whole path's C buffers are per bank, so each bank needs its
  /// own view of the table while its layer runs.
  mutable std::array<WholeStreams, 2> whole_streams_{};
  WholeStreams& Whole() const { return whole_streams_[route_bank_]; }
  mutable void* gdn_chunk_scratch_{nullptr};
  mutable std::size_t gdn_chunk_scratch_bytes_{0};
  mutable std::uint32_t q8t_rows_{0};
  mutable std::size_t q8t_cols_{0};
  mutable const float* half_src_{nullptr};
  mutable std::uint32_t half_rows_{0};
  mutable std::size_t half_cols_{0};
  mutable bool half_bf16_{false};
  /// Set by a combine that wrote s_.xn_half / s_.xn_q8t instead of s_.xn.
  mutable bool xn_half_{false};
  mutable bool inject_fused_{
      false};  ///< the last combine formed the inject logits
  mutable const float* inject_fused_w_{
      nullptr};  ///< for the mixer with these weights
  /// Partial sums per inject logit the last HcMix left in s_.inject.
  mutable std::uint32_t inject_parts_{1};
  mutable unsigned q8_slot_{0};
  /// Producer-written Q8_1 rows (see ReadyQ8): source, rows and width.
  struct ReadyRows {
    const float* src{nullptr};
    const void* data{nullptr};
    std::uint32_t n{0};
    std::uint32_t k{0};
  };
  /// A decode combine left to the next mixer's one-launch HcMixQ8Fused
  /// (see Combine): HcCombineQ8_1's operands when Combine ran.
  struct PendingCombine {
    const DeviceMixer* mixer{nullptr};
    float* res{nullptr};
    const float* gamma{nullptr};
    const float* block_out{nullptr};
    const float* inject{nullptr};
    std::uint32_t inject_parts{1};
    float* xn{nullptr};
    void* xn_q8{nullptr};
    std::uint32_t n{0};
  };
  struct BankCache {
    const float* half_src_{nullptr};
    std::uint32_t half_rows_{0};
    std::uint32_t half_cols_{0};
    bool half_bf16_{false};

    const float* q8t_src_{nullptr};
    std::uint32_t q8t_rows_{0};
    std::uint32_t q8t_cols_{0};

    ReadyRows xn_q8_ready_{};
    ReadyRows mixed_q8_ready_{};
    /// See Combine: a decode combine handed to the next mixer's launch.
    PendingCombine pending_combine_{};

    bool shexp_half_ready_{false};
    bool moe_pending_{false};
    bool xn_half_{false};
    bool inject_fused_{false};
    const float* inject_fused_w_{nullptr};
    std::uint32_t inject_parts_{1};
    unsigned q8_slot_{0};

    bool normed{false};
    bool res_is_half{false};
  };
  void SaveCache(BankCache* cache) const;
  void LoadCache(const BankCache& cache) const;
  mutable ReadyRows xn_q8_ready_;
  mutable ReadyRows mixed_q8_ready_;
  mutable PendingCombine pending_combine_;
  /// HcMixQ8Fused's counters and intermediates (HcMixQ8FusedStateBytes).
  void* hc_fused_state_{nullptr};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
