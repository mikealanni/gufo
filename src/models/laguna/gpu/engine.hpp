#ifndef GUFO_MODELS_LAGUNA_GPU_ENGINE_HPP_
#define GUFO_MODELS_LAGUNA_GPU_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "src/models/laguna/runtime/config.hpp"

namespace gufo::hip {
class QwenDFlashGpuExecutor;
class QwenDFlashGpuModel;
}  // namespace gufo::hip

namespace gufo::models::laguna::gpu {

struct EngineOptions {
  /// Tokens one prefill pass processes.
  std::uint32_t prefill_tile{4096};
  /// Optional DFlash-2 drafter; its trunk taps bind to this model.
  std::string dflash_model_path;
  /// Upper bound for session contexts; bounds the drafter's context.
  std::uint32_t max_context{65536};
};

class Model;
struct ModelState;

/// One conversation: KV cache, position and the logits of the last token.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Appends `tokens` and leaves the logits of the last one in Logits().
  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::string* error_msg);

  /// Appends 1..kMaxVerifyRows tokens and keeps the logits of every row:
  /// row i is the distribution after tokens[i]. Used to verify drafts.
  [[nodiscard]] bool AppendAll(std::span<const std::int32_t> tokens,
                               std::string* error_msg);
  static constexpr std::uint32_t kMaxVerifyRows = 16;

  /// Drops cache rows past `position`; later appends overwrite them.
  void Rewind(std::uint32_t position) noexcept;

  /// DFlash-2 executor bound to this session, or null when no drafter was
  /// loaded. Multi-row AppendAll leaves the drafter un-injected; the caller
  /// commits the accepted prefix with DflashInject after verification.
  [[nodiscard]] hip::QwenDFlashGpuExecutor* DflashExecutor() const noexcept;
  /// Injects the first `rows` captured rows of the last forward into the
  /// drafter and returns true when its context covers `Position()`.
  bool DflashInject(std::uint32_t rows, std::string* error_msg);

  [[nodiscard]] std::span<const float> Logits() const noexcept;
  /// Row `row` of the last AppendAll (or Append of up to kMaxVerifyRows).
  [[nodiscard]] std::span<const float> Logits(std::uint32_t row) const noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t Capacity() const noexcept;

  /// Forgets everything: position 0, empty cache.
  void Reset() noexcept;

  /// Device bytes of the KV caches (scratch excluded).
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

private:
  friend class Model;
  Session();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The Laguna weights resident on the GPU, shared by sessions.
class Model final {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const EngineOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] bool HasDflash() const noexcept;
  /// Tokens the drafter proposes and one verification pass holds at most.
  [[nodiscard]] std::uint32_t DrafterBlockSize() const noexcept;

private:
  Model() = default;
  std::shared_ptr<ModelState> state_;
};

}  // namespace gufo::models::laguna::gpu

#endif  // GUFO_MODELS_LAGUNA_GPU_ENGINE_HPP_
