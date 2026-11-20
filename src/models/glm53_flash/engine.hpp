#ifndef GUFO_MODELS_GLM53_FLASH_ENGINE_HPP_
#define GUFO_MODELS_GLM53_FLASH_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/glm53_flash/kernels/rocm/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/lookup_draft.hpp"

namespace gufo::hip {
class QwenDFlashGpuExecutor;
class QwenDFlashGpuModel;
}  // namespace gufo::hip

namespace gufo::models::glm53_flash {

/// Longest chain one verification forward carries: the anchor plus drafts.
inline constexpr std::uint32_t kMaxDraftTokens = 7;

/// Trunk taps and block size the DFlash-2 drafter file asks for.
struct DrafterInfo {
  std::vector<std::uint32_t> target_layers;
  std::uint32_t block_size{0};
  std::uint32_t sliding_window{0};
};
[[nodiscard]] bool ReadDrafterInfo(const std::string& path, DrafterInfo* info,
                                   std::string* error);

struct ModelOptions {
  /// The GGUF's own context is a million tokens; sessions allocate for this.
  std::uint32_t max_context = 4096;
  /// Optional DFlash-2 drafter bound to this trunk's embedding and head.
  std::string dflash_model_path;
  /// Optional MTP (NEXTN) draft-block sidecar (`mtp-GLM-5.3-*.gguf`).
  std::string mtp_model_path;
  std::uint32_t max_draft_tokens = kMaxDraftTokens;
};

class Session;
class SessionSnapshot;

/// Gufo-owned API over the ROCm executor: one resident model, any number of
/// sessions with independent context state. Calls on different sessions must
/// not overlap: the executor owns one set of scratch buffers.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept { return eos_token_; }
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  /// Tokens one scheduled prefill step takes: a pair of NPU banks.
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  /// Draft tokens one decode step may verify beyond the sampled one.
  [[nodiscard]] std::uint32_t MaxDraftTokens() const noexcept {
    return options_.max_draft_tokens;
  }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] bool HasDflash() const noexcept {
    return dflash_model_ != nullptr;
  }
  [[nodiscard]] bool HasMtp() const noexcept {
    return executor_ && executor_->HasMtp();
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept {
    return resident_bytes_;
  }
  /// Worst-case private device state of one session, including the rollback
  /// copies and the drafter's context.
  [[nodiscard]] std::size_t SessionBytes(core::SessionMode mode,
                                         std::uint32_t context) const noexcept;
  [[nodiscard]] const core::GgufReader& reader() const noexcept {
    return *reader_;
  }

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<rocm::Executor> executor_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::vector<std::int32_t> stop_tokens_;
  std::int32_t eos_token_{0};
  std::size_t resident_bytes_{0};
  std::uint32_t dflash_window_{0};

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt, reusing the prefix the
/// session already holds; Evaluate appends one token; DecodeStep samples the
/// next token and, when drafts are available, verifies several in one pass.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: an extension feeds only the
  /// new tail, any divergence restarts the session (recurrent state cannot
  /// be rewound).
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);

  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  /// Samples the next token from the last logits and returns it with every
  /// draft token the target accepts. History drafts and DFlash proposals are
  /// verified exactly: greedy output equals ordinary decoding, sampled
  /// output keeps the target distribution and consumes one draw per token.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);

  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  /// Polled between layers and snapshot copies; true abandons the call and
  /// leaves the session needing a Sync.
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  /// A new request reusing cached context starts its own acceptance history.
  void ResetDraftPolicy();
  /// Measurement hook: a non-negative width makes every step draft exactly
  /// that many DFlash tokens (0 is an ordinary step); negative restores the
  /// controller.
  void ForceDraftWidth(int width) noexcept { forced_width_ = width; }

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
    /// Cycles by chain length minus one; widths[0] is ordinary decoding.
    std::uint64_t widths[kMaxDraftTokens + 1]{};
    /// Cycles drafted from the token history instead of the DFlash drafter.
    std::uint64_t history_cycles{0};
    std::uint64_t history_drafted{0};
    std::uint64_t history_accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 1;
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  /// Copies the whole context to host memory. The drafter's context is not
  /// part of it: drafting resumes once a forward primes the restored frontier.
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  friend class Model;
  Session(std::shared_ptr<Model> model,
          std::unique_ptr<rocm::Executor::Session> session);

  bool Feed(std::span<const std::int32_t> tokens, bool prefill,
            std::string* error_msg);
  bool DecodeImpl(std::size_t max_tokens, sampling::SamplerState& sampler,
                  DecodeResult* result, std::string* error_msg,
                  bool stop_at_eos);
  void SyncLookup();
  bool InitDflash(std::string* error_msg);
  [[nodiscard]] bool DflashReady() const noexcept;
  /// Hands the drafter the trunk rows captured by the last forward.
  bool DflashInject(std::uint32_t rows, std::string* error_msg);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Executor::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  std::vector<float> verify_logits_;
  qwen38_flash_next::LookupIndex lookup_;
  bool speculative_{false};
  float accepted_ema_{static_cast<float>(kMaxDraftTokens)};
  std::uint32_t plain_steps_{0};
  int forced_width_{-1};
  SpeculativeStats stats_;
  bool valid_{true};
  struct DflashState;
  std::unique_ptr<DflashState> dflash_;
};

/// Immutable host copy of a session checkpoint; the byte form restores in
/// memory and from disk.
class SessionSnapshot final {
public:
  ~SessionSnapshot();
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept { return size_; }
  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return {data_.get(), static_cast<std::size_t>(size_)};
  }

private:
  explicit SessionSnapshot(std::uint64_t size);

  std::unique_ptr<std::uint8_t[]> data_;
  std::uint64_t size_{0};

  friend class Session;
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_ENGINE_HPP_
