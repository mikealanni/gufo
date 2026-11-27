#ifndef GUFO_MODELS_RAON1_ENGINE_HPP_
#define GUFO_MODELS_RAON1_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/lookup_draft.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/raon1/kernels/rocm/executor.hpp"
#include "src/models/raon1/runtime/config.hpp"

namespace gufo::hip {
class QwenDFlashGpuExecutor;
class QwenDFlashGpuModel;
}  // namespace gufo::hip

namespace gufo::models::raon1 {

/// Longest chain one verification forward carries: the anchor plus drafts.
inline constexpr std::uint32_t kMaxDraftTokens = 7;

struct EngineOptions {
  std::uint32_t layers{0};
  std::uint32_t max_context{4096};
  /// Optional DFlash-2 drafter bound to this trunk's embedding and head.
  std::string dflash_model_path;
  std::uint32_t max_draft_tokens{kMaxDraftTokens};
};

class Model;

class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
    std::uint64_t history_drafted{0};
    std::uint64_t history_accepted{0};
  };
  using StopFn = std::function<bool(std::int32_t)>;
  /// Samples the next token from the last logits and returns it with every
  /// draft token the target accepts (history and DFlash-2 drafts, verified
  /// exactly).
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result, const StopFn& is_stop,
                                std::string* error = nullptr);
  [[nodiscard]] bool Speculative() const noexcept { return dflash_ != nullptr; }
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }
  [[nodiscard]] bool Evaluate(std::int32_t token, std::string* error = nullptr);
  [[nodiscard]] bool Sync(std::span<const std::int32_t> tokens,
                          std::string* error = nullptr);
  [[nodiscard]] bool SnapshotSupported() const noexcept;
  [[nodiscard]] std::size_t SnapshotBytes() const;
  [[nodiscard]] bool SaveSnapshot(std::span<std::uint8_t> out,
                                  std::string* error = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> in,
                                     std::string* error = nullptr);
  void SetPromptEnd(std::uint32_t end) noexcept { prompt_end_ = end; }
  [[nodiscard]] std::vector<float> Logits(std::string* error = nullptr) const;
  [[nodiscard]] std::int32_t Argmax(std::string* error = nullptr) const;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return tokens_;
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  void Reset() noexcept;

private:
  friend class Model;
  Session(std::shared_ptr<rocm::Executor> executor,
          std::unique_ptr<rocm::Executor::Session> impl);
  bool Feed(std::span<const std::int32_t> tokens, std::string* error);
  bool DflashInject(std::uint32_t rows, std::string* error);
  bool InjectPart(const float* features, std::uint32_t position,
                  std::uint32_t rows, std::uint32_t end, std::string* error);
  void SyncLookup();
  std::shared_ptr<Model> model_;
  std::shared_ptr<rocm::Executor> executor_;
  std::unique_ptr<rocm::Executor::Session> impl_;
  std::vector<std::int32_t> tokens_;
  std::uint32_t commit_row_{0};
  std::uint32_t prompt_end_{0};
  std::vector<float> verify_logits_;
  qwen38_flash_next::LookupIndex lookup_;
  float accepted_ema_{static_cast<float>(kMaxDraftTokens)};
  std::uint32_t plain_steps_{0};
  SpeculativeStats stats_;
  struct DflashState;
  std::unique_ptr<DflashState> dflash_;
};

class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& path, const EngineOptions& options,
      std::string* error = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error = nullptr, bool speculative = false);
  [[nodiscard]] bool HasDflash() const noexcept {
    return dflash_model_ != nullptr;
  }

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::uint32_t layers() const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] bool SnapshotSupported() const noexcept;

private:
  Model() = default;
  friend class Session;
  EngineOptions options_;
  std::shared_ptr<rocm::Executor> executor_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::size_t resident_bytes_{0};
};

}  // namespace gufo::models::raon1

#endif  // GUFO_MODELS_RAON1_ENGINE_HPP_
