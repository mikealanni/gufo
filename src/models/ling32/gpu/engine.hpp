#ifndef GUFO_MODELS_LING32_GPU_ENGINE_HPP_
#define GUFO_MODELS_LING32_GPU_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/ling32/runtime/config.hpp"

namespace gufo::models::ling32::gpu {

class Model;
class Session;

class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::string* error_msg = nullptr);
  [[nodiscard]] bool MtpAppend(std::int32_t token,
                               std::string* error_msg = nullptr);
  void ResetMtp();
  void EnableMtp();
  // Speculative verification: runs tokens (at most 8) from the current
  // position without advancing it; logits receives tokens.size() rows.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens,
                            std::vector<float>* logits,
                            std::string* error_msg = nullptr);
  // Keeps tokens[0..keep] of the last Verify and rolls the rest back.
  [[nodiscard]] bool Commit(std::uint32_t keep,
                            std::string* error_msg = nullptr);
  // Seeds the MTP chain from the trunk's current hidden state.
  void BeginDraft();
  [[nodiscard]] const std::vector<float>& Logits() const noexcept;
  [[nodiscard]] const std::vector<float>& MtpLogits() const noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  void Reset();

private:
  friend class Model;
  Session();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, std::string* error_msg = nullptr);
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;

private:
  friend class Session;
  Model() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::ling32::gpu

#endif  // GUFO_MODELS_LING32_GPU_ENGINE_HPP_
