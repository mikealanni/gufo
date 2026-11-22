#ifndef GUFO_MODELS_INKLING_GPU_ENGINE_HPP_
#define GUFO_MODELS_INKLING_GPU_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "src/models/inkling/runtime/config.hpp"

namespace gufo::models::inkling::gpu {

class Model;
struct ModelState;

class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Appends `tokens` and leaves the logits of the last one in Logits().
  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::string* error_msg);
  /// Appends 1..kMaxVerifyRows tokens and keeps the logits of every row (row i
  /// is the distribution after tokens[i]).
  [[nodiscard]] bool AppendAll(std::span<const std::int32_t> tokens,
                               std::string* error_msg);
  static constexpr std::uint32_t kMaxVerifyRows = 8;
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::span<const float> Logits(std::uint32_t row) const noexcept;
  /// After AppendAll: keeps only the first `keep` rows (cache rows past them
  /// are overwritten by the next append and the convolution history is rewound
  /// to that point).
  void KeepRows(std::uint32_t keep);
  /// Drafts out.size() (at most 7) tokens after `anchor`, the next token to
  /// append; needs INKLING_DRAFT and Model::LoadDrafter.
  [[nodiscard]] bool Draft(std::int32_t anchor, std::span<std::int32_t> out);
  /// Drops cache rows past `position`; later appends overwrite them.
  void Rewind(std::uint32_t position) noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  void Reset() noexcept;
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

private:
  friend class Model;
  Session();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Model final {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, std::string* error_msg = nullptr);
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);
  [[nodiscard]] bool LoadDrafter(const std::string& path,
                                 std::string* error_msg = nullptr);
  [[nodiscard]] bool HasDrafter() const noexcept;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;

private:
  Model() = default;
  std::shared_ptr<ModelState> state_;
};

}  // namespace gufo::models::inkling::gpu

#endif  // GUFO_MODELS_INKLING_GPU_ENGINE_HPP_
