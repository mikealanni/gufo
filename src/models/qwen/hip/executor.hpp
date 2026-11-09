#ifndef GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/models/qwen/hip/detail/prefill_task.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/moe_prefill.hpp"
#include "src/models/qwen/hip/npu_ffn.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/hip/prompt_hidden_buffer.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::hip {

class HipblasLtGemm;
class QwenGpuExecutor;

struct QwenGpuBatchItem {
  QwenGpuExecutor* executor{nullptr};
  tokenization::TokenId token_id{0};
  std::uint32_t position{0};
};

struct QwenGpuVerificationItem {
  QwenGpuExecutor* executor{nullptr};
  std::span<const tokenization::TokenId> tokens;
  std::uint32_t position{0};
  bool capture_logits{false};
};

struct QwenSampledVerificationResult {
  tokenization::TokenId token{0};
  bool accepted{false};
};

struct QwenGpuWeightRegion {
  // Original reader address for tensor offsets; host_copy owns GPU-visible
  // immutable bytes, backed by transparent huge pages when available.
  const void* host_data{nullptr};
  void* device_data{nullptr};
  void* host_copy{nullptr};
  std::size_t size{0};
};

struct QwenGpuMemoryUsage {
  std::size_t request_state_bytes{0};
  std::size_t temporary_scratch_bytes{0};

  [[nodiscard]] std::size_t TotalBytes() const noexcept {
    return request_state_bytes + temporary_scratch_bytes;
  }
};

/// Immutable GPU-visible Qwen model resources shared by executor sessions.
class QwenGpuModel {
public:
  QwenGpuModel(std::shared_ptr<const core::GgufReader> reader,
               models::QwenModelWeights weights,
               std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
               std::vector<QwenGpuWeightRegion> weight_regions,
               std::shared_ptr<models::qwen::vision::Encoder> vision = {});
  ~QwenGpuModel();

  QwenGpuModel(const QwenGpuModel&) = delete;
  QwenGpuModel& operator=(const QwenGpuModel&) = delete;
  QwenGpuModel(QwenGpuModel&&) = delete;
  QwenGpuModel& operator=(QwenGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenGpuModel> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr,
      std::shared_ptr<models::qwen::vision::Encoder> vision = {});

  [[nodiscard]] const std::shared_ptr<models::qwen::vision::Encoder>&
  VisionEncoder() const noexcept {
    return vision_;
  }
  [[nodiscard]] const models::QwenModelWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetWeightRegionCount() const noexcept {
    return weight_regions_.size();
  }
  [[nodiscard]] std::size_t GetResidentBytes() const noexcept;
  /// NPU offload of the FFN gate/up projections, or null when it is off.
  [[nodiscard]] QwenNpuFfn* NpuFfn() const;
  /// Grouped-expert prefill for qwen35moe, or null when unavailable.
  [[nodiscard]] QwenMoePrefill* MoePrefill(std::uint32_t max_tokens) const;
  /// bf16 [hi, hi, lo] weights of the router and shared-expert gate rows of
  /// `layer` (kRouterSplitRows rows of 3 * hidden), built on first use; null
  /// when the layer's gate tensors are not F32 or BF16.
  [[nodiscard]] const std::uint16_t* RouterSplitWeights(
      std::uint32_t layer) const;

private:
  // Keep the mapped GGUF storage alive until every registered region is
  // released.
  std::shared_ptr<const core::GgufReader> reader_;
  models::QwenModelWeights weights_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::vector<QwenGpuWeightRegion> weight_regions_;
  std::shared_ptr<models::qwen::vision::Encoder> vision_;
  mutable std::once_flag npu_once_;
  mutable std::unique_ptr<QwenNpuFfn> npu_ffn_;
  mutable std::once_flag moe_prefill_once_;
  mutable std::unique_ptr<QwenMoePrefill> moe_prefill_;
  mutable std::vector<std::uint16_t*> router_split_;
  mutable std::mutex router_split_mutex_;
};

class QwenGpuSnapshot final {
public:
  ~QwenGpuSnapshot();

  QwenGpuSnapshot(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot& operator=(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot(QwenGpuSnapshot&&) = delete;
  QwenGpuSnapshot& operator=(QwenGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_ + vision_layout_.images.size() *
                                sizeof(models::qwen::vision::ImageGrid);
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] QwenKvCacheStorage KvStorage() const noexcept {
    return kv_storage_;
  }
  [[nodiscard]] QwenRecurrentStateStorage RecurrentStateStorage()
      const noexcept {
    return recurrent_state_storage_;
  }
  [[nodiscard]] std::size_t CompactPayloadBytes() const;
  [[nodiscard]] std::size_t SerializeCompact(
      std::span<std::uint8_t> destination) const;
  [[nodiscard]] static models::qwen::vision::RopeLayout ReadRopeLayout(
      std::span<const std::uint8_t> payload);

private:
  QwenGpuSnapshot() = default;

  void* d_kv_f32_{nullptr};
  void* d_kv_f16_{nullptr};
  void* d_ssm_conv_{nullptr};
  void* d_ssm_deltanet_{nullptr};
  std::size_t kv_elements_per_plane_{0};
  std::size_t conv_elements_{0};
  std::size_t deltanet_elements_{0};
  std::size_t conv_elements_per_layer_{0};
  std::size_t deltanet_elements_per_layer_{0};
  std::uint32_t num_layers_{0};
  std::uint32_t attention_layers_{0};
  std::uint32_t full_attention_interval_{0};
  std::uint32_t kv_width_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::size_t payload_bytes_{0};
  QwenKvCacheStorage kv_storage_{QwenKvCacheStorage::kFp32};
  QwenRecurrentStateStorage recurrent_state_storage_{
      QwenRecurrentStateStorage::kFp32};

  friend class QwenGpuArena;
  friend class QwenGpuExecutor;
  models::qwen::vision::RopeLayout vision_layout_;
};

/// Shared decode-lifetime workspaces over stable arena allocations.
struct QwenDecodeScratch {
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> logits;
  std::span<hip_bfloat16> bf16;
  std::span<hip_bfloat16> weight_bf16;
  std::span<std::uint32_t> prompt_tokens;

  /// Aliases the first element of QwenSsmScratch::alpha. It is legal only
  /// after all layer execution has completed, during the sampling/output epoch.
  std::span<std::uint32_t> sampled_token;
};

struct QwenAttentionScratch {
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> output;
  std::span<float> split_k;
};

struct QwenSsmScratch {
  std::span<float> qkv;
  std::span<float> conv_out;
  std::span<float> gate;
  std::span<float> out;
  std::span<float> alpha;
  std::span<float> beta;
};

struct QwenFfnScratch {
  std::span<float> gate;
  std::span<float> up;
  std::span<float> activation;
  std::span<float> out;
};

/// Typed non-owning capability views over stable Qwen GPU arena allocations.
/// The nested aggregates carry element counts without owning memory or changing
/// any address used by kernels or graph capture.
struct QwenGpuScratchView {
  QwenDecodeScratch decode;
  QwenAttentionScratch attention;
  QwenSsmScratch ssm;
  QwenFfnScratch ffn;
};

static_assert(std::is_trivially_copyable_v<QwenDecodeScratch>);
static_assert(std::is_trivially_copyable_v<QwenAttentionScratch>);
static_assert(std::is_trivially_copyable_v<QwenSsmScratch>);
static_assert(std::is_trivially_copyable_v<QwenFfnScratch>);
static_assert(std::is_trivially_copyable_v<QwenGpuScratchView>);
static_assert(std::is_same_v<decltype(QwenDecodeScratch::sampled_token),
                             std::span<std::uint32_t>>);

template<typename T>
[[nodiscard]] constexpr T* OffsetIfPresent(T* pointer,
                                           std::size_t offset) noexcept {
  return pointer == nullptr ? nullptr : pointer + offset;
}

/// The per-chunk prefill scratch of an arena. A second set lets two chunks
/// advance layer by layer in one stream; persistent KV and recurrent state are
/// shared and never part of a bank.
struct QwenPrefillBank {
  float* hidden{nullptr};
  float* normed{nullptr};
  float* q{nullptr};
  float* k{nullptr};
  float* v{nullptr};
  float* attn_out{nullptr};
  float* ffn_gate{nullptr};
  float* ffn_up{nullptr};
  float* ffn_act{nullptr};
  float* ffn_out{nullptr};
  float* ssm_qkv{nullptr};
  float* conv_out{nullptr};
  float* ssm_gate{nullptr};
  float* ssm_out{nullptr};
  float* alpha_buf{nullptr};
  float* beta_buf{nullptr};
  float* ssm_kq_scales{nullptr};
  float* ssm_alpha_beta{nullptr};
  std::uint32_t* prompt_tokens{nullptr};
  void* scratch_bf16{nullptr};
  void* scratch_q8_act{nullptr};
};

/// Device scratch of the single-token MoE feed-forward (qwen35moe decode).
struct QwenMoeScratch {
  float* router_logits{nullptr};
  float* shared_logit{nullptr};
  std::int32_t* ids{nullptr};
  float* weights{nullptr};
  float* shared_weight{nullptr};
  float* act{nullptr};
  float* down_out{nullptr};
  float* shared_gate{nullptr};
  float* shared_up{nullptr};
  float* shared_act{nullptr};
  float* shared_out{nullptr};
};

/// Device scratch of the batched qwen35moe prefill feed-forward.
struct QwenMoePrefillScratch {
  __half* x_half{nullptr};
  float* router_logits{nullptr};
  float* shared_logit{nullptr};
  float* shared_gate{nullptr};
  float* shared_up{nullptr};
  float* shared_act{nullptr};
  float* shared_out{nullptr};
  std::uint16_t* router_x{nullptr};
  float* router_y{nullptr};
};

/// Preallocated, zero-allocation GPU execution arena on gfx1151.
class QwenGpuArena {
public:
  friend class QwenGpuExecutor;
  explicit QwenGpuArena(
      const core::ModelConfig& config, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());
  ~QwenGpuArena();

  QwenGpuArena(const QwenGpuArena&) = delete;
  QwenGpuArena& operator=(const QwenGpuArena&) = delete;
  QwenGpuArena(QwenGpuArena&&) noexcept;
  QwenGpuArena& operator=(QwenGpuArena&&) noexcept;

  void Reset();
  /// Pins the live recurrent state for rollback. Neither save nor restore
  /// copies anything: the state lives in one of two buffers, and the first
  /// pass after either call reads the pinned buffer and writes the other one.
  void SaveState(std::uint32_t valid_context);
  void RestoreState();
  /// The scratch currently installed in the d_* members, and a way to install
  /// another. The second bank is allocated on first use; null on failure.
  [[nodiscard]] QwenPrefillBank CurrentBank() const noexcept;
  void UseBank(const QwenPrefillBank& bank) noexcept;
  [[nodiscard]] const QwenPrefillBank* SecondBank();
  /// Names the buffers one recurrent-state pass reads and writes. Call exactly
  /// once per forward, replay or prefill pass, before its first launch; every
  /// launch of the pass then writes `*_out`, so the pinned buffer survives
  /// until the next SaveState.
  [[nodiscard]] QwenRecurrentStateIo BeginRecurrentWrite();
  /// Buffer indices of the most recent BeginRecurrentWrite, for graph keys.
  [[nodiscard]] std::uint64_t RecurrentIoIdentity() const noexcept;
  [[nodiscard]] bool BeginSsmReplayCapture();
  void DisableSsmReplayCapture();
  void MarkSsmReplayPosition(std::uint32_t position) noexcept;
  [[nodiscard]] bool CanReplaySsmPosition(
      std::uint32_t position) const noexcept;
  [[nodiscard]] SsmReplayCapture GetSsmReplayCapture() const noexcept;
  [[nodiscard]] bool IsSsmReplayCaptureActive() const noexcept {
    return replay_capture_active_;
  }
  [[nodiscard]] const float* GetReplayQkv(std::uint32_t layer,
                                          std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayAlpha(std::uint32_t layer,
                                            std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayBeta(std::uint32_t layer,
                                           std::uint32_t position) const;
  /// Grows the replay ring to hold `rows` rows of one pass (a power of two
  /// slots). Call before the pass captures; returns true when buffers moved.
  [[nodiscard]] bool EnsureSsmReplayCapacity(std::size_t rows);
  [[nodiscard]] static std::size_t SsmReplayLogBytes(
      const core::ModelConfig& config, std::size_t capacity);

  float* d_hidden{nullptr};
  float* d_normed{nullptr};
  float* d_q{nullptr};
  float* d_k{nullptr};
  float* d_v{nullptr};
  float* d_attn_out{nullptr};
  float* d_ffn_gate{nullptr};
  float* d_ffn_up{nullptr};
  float* d_ffn_act{nullptr};
  float* d_ffn_out{nullptr};
  float* d_ssm_qkv{nullptr};
  float* d_conv_out{nullptr};
  float* d_ssm_gate{nullptr};
  float* d_ssm_out{nullptr};
  float* d_alpha_buf{nullptr};
  float* d_beta_buf{nullptr};
  // opt-c170-deltanet-rowsplit prologue scratch: 3 floats per (token, key head)
  // for the k/q norms and the unnormalized k . q, and 2 floats per (token,
  // value head) for the decay and beta gates.
  float* d_ssm_kq_scales{nullptr};
  float* d_ssm_alpha_beta{nullptr};
  float* d_logits{nullptr};
  void* d_attention_kv_f16{nullptr};
  float* d_kv_cache{nullptr};
  /// Opt-in block-sparse attention (GUFO_QWEN_SPARSE_ATTN): per-layer key
  /// bounds, per-tile block lists and how many keys each layer's bounds cover.
  void* d_sparse_pooled{nullptr};
  std::int32_t* d_sparse_lists{nullptr};
  std::vector<std::uint32_t> sparse_pooled_end;
  SparseAttentionConfig sparse_config{};
  bool sparse_enabled{false};
  void InvalidateSparseAttention() noexcept {
    std::fill(sparse_pooled_end.begin(), sparse_pooled_end.end(), 0u);
  }
  // Mirrors of the live recurrent buffers (see BeginRecurrentWrite); reassigned
  // whenever the live buffer changes, so snapshots and resets read and write
  // the current state without knowing about the rollback pair.
  float* d_ssm_conv_state{nullptr};
  void* d_ssm_deltanet_state{nullptr};
  std::uint32_t* d_prompt_tokens{nullptr};
  float* d_target_layer_features{nullptr};
  QwenMoeScratch moe;
  QwenMoePrefillScratch moe_prefill;

  hipStream_t stream{nullptr};
  hipblasHandle_t hipblas_handle{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm;
  void* d_scratch_bf16{nullptr};
  float* d_split_k_attention{nullptr};
  hip_bfloat16* d_weights_bf16{nullptr};
  void* d_scratch_q8_act{nullptr};

  [[nodiscard]] std::uint32_t GetMaxBatch() const noexcept {
    return max_batch_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::size_t GetAttentionKvPlaneElements() const noexcept {
    // Qwen27B reaches 2^32 elements per plane at context 262144.
    return static_cast<std::size_t>(config_.FullAttentionLayerCount()) *
           config_.num_key_value_heads * max_context_ * config_.head_dim;
  }
  [[nodiscard]] QwenRecurrentStateStorage GetRecurrentStateStorage()
      const noexcept {
    return policy_.recurrent_state_storage;
  }
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const;
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      std::uint32_t valid_context) const;

  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);
  [[nodiscard]] QwenGpuScratchView GetScratchView(
      std::size_t batch_size = 1) noexcept;
  void SetTargetLayerCapture(std::span<const std::uint32_t> target_layer_ids);
  [[nodiscard]] std::span<const std::uint32_t> GetTargetLayerCapture()
      const noexcept {
    return target_layer_ids_;
  }
  [[nodiscard]] std::optional<std::size_t> GetTargetLayerCaptureIndex(
      std::uint32_t layer) const noexcept;

private:
  void FreeAll() noexcept;
  void AllocateRecurrentSnapshot();
  void SyncLivePointers() noexcept;
  [[nodiscard]] bool AllocateSsmReplayLog();

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::uint32_t max_batch_;
  QwenExecutionPolicy policy_;
  std::vector<std::uint32_t> target_layer_ids_;
  // Buffer 0 comes from the constructor; buffer 1 is allocated by the first
  // SaveState. `live_recurrent_` holds the current state, `saved_recurrent_`
  // the pinned one (meaningful while has_saved_state_), `source_recurrent_`
  // the buffer the most recent pass read.
  std::array<float*, 2> d_ssm_conv_buffers_{};
  std::array<void*, 2> d_ssm_deltanet_buffers_{};
  std::uint8_t live_recurrent_{0};
  std::uint8_t saved_recurrent_{0};
  std::uint8_t source_recurrent_{0};
  float* d_ssm_replay_qkv_{nullptr};
  float* d_ssm_replay_alpha_{nullptr};
  float* d_ssm_replay_beta_{nullptr};
  std::uint32_t* d_ssm_replay_enabled_{nullptr};
  std::size_t replay_capacity_{kSsmReplayCapacity};
  std::uint32_t saved_context_{0};
  std::uint32_t replay_last_position_{0};
  std::size_t replay_captured_positions_{0};
  bool has_saved_state_{false};
  bool replay_capture_active_{false};
  std::unique_ptr<QwenPrefillBank> second_bank_;
  float* d_moe_buf_{nullptr};
  void* d_moe_prefill_buf_{nullptr};
};

/// End-to-end GPU model executor running directly on the gfx1151 RDNA 3.5 CUs.
class QwenGpuExecutor {
public:
  explicit QwenGpuExecutor(
      std::shared_ptr<const QwenGpuModel> model,
      std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());
  ~QwenGpuExecutor();

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> Create(
      std::shared_ptr<const QwenGpuModel> model,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());

  /// Generates tokens auto-regressively on GPU with streaming callback.
  std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  /// Continues generation from an exact retained prompt prefix.
  ///
  /// The executor state must represent prompt_tokens[0:cached_prefix_tokens].
  /// A zero prefix is a cold prefill over the complete prompt.
  std::vector<tokenization::TokenId> GenerateFromPrefix(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::size_t cached_prefix_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }

  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }

  [[nodiscard]] const QwenGpuModel& GetModel() const noexcept {
    return *model_;
  }

  [[nodiscard]] const std::shared_ptr<const QwenGpuModel>& GetSharedModel()
      const noexcept {
    return model_;
  }

  [[nodiscard]] const QwenExecutionPolicy& GetExecutionPolicy() const noexcept {
    return policy_;
  }

  /// Runs one single token forward step on GPU, returning next token ID.
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true);

  /// Advances two to eight independent retained sessions in one
  /// layer-synchronous decode step. Stateless projections share weight reads;
  /// every row still addresses its own KV cache and recurrent state.
  [[nodiscard]] static std::vector<tokenization::TokenId> ForwardTokenBatch(
      std::span<const QwenGpuBatchItem> items);

  /// Runs batched prompt prefill on GPU, returning the first predicted token
  /// ID.
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos = 0, bool compute_logits = true);

  /// Verifies a chunk of speculative draft candidate tokens in a single
  /// parallel prefill forward pass on the GPU.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits = false);
  /// Shares exact projections across independent speculative chunks. Each
  /// sequence retains its own causal attention, recurrence and replay state.
  [[nodiscard]] static std::vector<std::vector<tokenization::TokenId>>
  ForwardVerificationBatch(std::span<const QwenGpuVerificationItem> items);
  /// Rows one session may verify in a single pass. A chain longer than eight
  /// rows (a prompt-lookup copy) shares every projection across its rows and
  /// runs attention in eight-row segments, so each row has the arithmetic of
  /// consecutive ordinary verifications; the replay ring grows to hold it.
  static constexpr std::size_t kMaxWideVerificationRows = 64;
  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos);
  [[nodiscard]] std::span<const float> GetVerificationHiddenStates()
      const noexcept {
    return h_verification_hidden_;
  }
  [[nodiscard]] std::span<const float> GetVerificationLogits() const noexcept {
    return h_verification_logits_;
  }
  [[nodiscard]] std::span<const float> CopyVerificationLogits(std::size_t row);

  /// Copies the logits produced by the most recent forward pass to host memory.
  [[nodiscard]] std::span<const float> CopyLastLogits();

  /// Samples the most recent device-resident logit row and transfers one token.
  [[nodiscard]] tokenization::TokenId SampleLastLogits(
      sampling::SamplerState& sampler);

  /// Restores a saved frontier row and samples with the same GPU operations
  /// as a fresh autoregressive request.
  [[nodiscard]] tokenization::TokenId SampleCachedLogits(
      std::span<const float> logits, sampling::SamplerState& sampler);

  /// Samples one row from the most recent verification batch without copying
  /// its vocabulary-sized logits to the host.
  [[nodiscard]] tokenization::TokenId SampleVerificationLogits(
      std::size_t row, sampling::SamplerState& sampler);

  /// Performs one exact sampled-speculation accept/residual decision on the
  /// GPU, returning only the decision and selected token.
  [[nodiscard]] QwenSampledVerificationResult VerifySampledToken(
      std::size_t row, tokenization::TokenId draft_token,
      std::span<const tokenization::TokenId> draft_candidate_ids,
      std::span<const float> draft_candidate_probabilities,
      double draft_token_probability, sampling::SamplerState& sampler);

  /// Pins the prompt-capture bound for `layer_count` captured layers up front,
  /// so no request pays for the pinned allocation. A ledger is charged for
  /// the pinned bytes while the executor holds them.
  void ReservePromptHiddenCapture(std::size_t layer_count,
                                  cache::ResourceLedger* ledger = nullptr);
  [[nodiscard]] std::size_t PromptHiddenChargedBytes() const noexcept {
    return h_prompt_hidden_.ChargedBytes();
  }

  /// Enables host capture of every final-layer prompt hidden state. Disabled
  /// by default so ordinary prefill does not incur device-to-host copies.
  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids = {});

  /// Returns the flattened [prompt_tokens, hidden_size] capture from the most
  /// recent ForwardPromptBatch call.
  [[nodiscard]] std::span<const float> GetPromptHiddenStates() const noexcept {
    return h_prompt_hidden_.View();
  }

  /// Copies the final-layer hidden state from the most recent forward pass.
  [[nodiscard]] std::span<const float> CopyLastHidden();

  [[nodiscard]] std::uint32_t GetMaxPromptBatch() const noexcept {
    return arena_.GetMaxBatch();
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return arena_.GetMaxContext();
  }
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const;
  void SetCancellationCheck(std::function<bool()> check) {
    cancellation_check_ = std::move(check);
    vision_input_.SetCancellationCheck(cancellation_check_);
  }
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      std::uint32_t valid_context) const;
  void ConfigureVision(
      std::shared_ptr<const models::qwen::vision::Prompt> prompt,
      std::shared_ptr<models::qwen::vision::Encoder> encoder);
  [[nodiscard]] models::qwen::vision::DeviceInput& VisionInput() noexcept {
    return vision_input_;
  }
  [[nodiscard]] const models::qwen::vision::RopeLayout& VisionLayout()
      const noexcept {
    return vision_input_.layout();
  }
  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);

  /// Resets GPU state. Any reset failure is retained and rethrown by the next
  /// fallible operation; invalidation/cleanup itself remains noexcept.
  void Reset() noexcept;
  void SaveState(std::uint32_t valid_context);
  void RestoreState();
  /// Ends replay capture after the last verification chunk.
  void FinishVerification();

private:
  void CheckPrefillCancellation() const;
  void PrefillLayerCheckpoint(std::uint32_t layer);
  std::function<bool()> cancellation_check_;
  std::array<hipEvent_t, 2> prefill_events_{};
  void CheckReset() const {
    if (reset_failure_)
      std::rethrow_exception(reset_failure_);
  }
  std::exception_ptr reset_failure_;
  void ReplaySsmState(std::uint32_t position, std::uint32_t count = 1);
  void EnsureVerificationLogits(std::size_t batch_size);
  [[nodiscard]] GpuSamplingParameters PrepareGpuSamplingParameters(
      const sampling::SamplerState& sampler);

  [[nodiscard]] tokenization::TokenId ForwardPromptChunk(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos, bool compute_logits);
  /// Two consecutive full chunks advanced layer by layer in one stream, the
  /// first on the spare scratch bank, so NPU stages overlap the other chunk's
  /// GPU work. Returns the second chunk's token when it computes logits.
  [[nodiscard]] tokenization::TokenId ForwardPromptChunkPair(
      std::span<const tokenization::TokenId> first_tokens,
      std::span<const tokenization::TokenId> second_tokens,
      std::uint32_t start_pos, bool compute_logits);
  /// Batched routed-expert feed-forward of a qwen35moe layer; false leaves the
  /// per-token path to run.
  [[nodiscard]] bool MoePrefillFfn(const models::QwenLayerWeights& layer,
                                   std::uint32_t layer_index,
                                   std::size_t batch_size);
  [[nodiscard]] detail::PrefillTask PrefillChunkTask(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos, bool compute_logits, int bank);
  [[nodiscard]] std::vector<tokenization::TokenId>
  ForwardDecodeEquivalentVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits);

  std::shared_ptr<const QwenGpuModel> model_;
  const models::QwenModelWeights& weights_;
  const tokenization::QwenTokenizer* tokenizer_;
  const QwenExecutionPolicy policy_;
  QwenGpuArena arena_;
  const detail::HipGraphCaptureKey graph_key_;
  detail::HipGraphDecodeExecutor graph_executor_;
  models::qwen::vision::DeviceInput vision_input_;
  std::vector<float> h_logits_;
  PromptHiddenBuffer h_prompt_hidden_;
  std::vector<float> h_verification_hidden_;
  std::vector<float> h_verification_logits_;
  std::vector<float> h_last_hidden_;
  float* d_verification_logits_{nullptr};
  std::size_t verification_logits_capacity_{0};
  std::size_t last_verification_rows_{0};
  std::size_t last_hidden_offset_{0};
  float* d_target_layer_features_{nullptr};
  GpuSamplingWorkspace sampling_workspace_;
  std::optional<tokenization::TokenId> next_token_;
  bool capture_prompt_hidden_{false};
  bool replaying_ssm_state_{false};
  bool moe_warmed_{false};
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_
