#ifndef GUFO_MODELS_INKLING_GPU_ENGINE_HPP_
#define GUFO_MODELS_INKLING_GPU_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "src/models/inkling/runtime/config.hpp"

namespace gufo::models::inkling::gpu {

class Model;
struct ModelState;
class Session;

/// Immutable copy of a session's context at one position: the full-attention
/// K/V rows, sliding-window rings, convolution histories, drafter features
/// and the last logits. Restores into any session of the same model and
/// context size.
class SessionSnapshot final {
public:
  ~SessionSnapshot();
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept;
  [[nodiscard]] std::uint32_t position() const noexcept;
  /// Versioned byte form for disk: header and logits, then the cache bytes.
  static constexpr std::uint32_t kPayloadVersion = 1;
  [[nodiscard]] std::uint64_t PayloadBytes() const noexcept;
  /// Hands the payload to `sink` in two spans without copying it.
  void StreamPayload(
      const std::function<void(std::span<const std::uint8_t>)>& sink) const;
  [[nodiscard]] bool CopyPayloadTo(
      std::span<std::uint8_t> destination) const noexcept;

  struct Data;  // engine-internal

private:
  SessionSnapshot();
  std::unique_ptr<Data> data_;
  friend class Session;
};

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
  /// Allocates the drafter's buffers now (otherwise on the first Draft).
  void PrepareDraft();
  /// Bytes a snapshot of the current position needs.
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  /// Restores a payload written by SessionSnapshot::StreamPayload.
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::span<const float> Logits(std::uint32_t row) const noexcept;
  /// Greedy token of a row, computed on the device (no logits download).
  [[nodiscard]] std::int32_t Argmax(std::uint32_t row) const noexcept;
  [[nodiscard]] std::int32_t Argmax() const noexcept;
  /// After AppendAll: keeps only the first `keep` rows (cache rows past them
  /// are overwritten by the next append and the convolution history is rewound
  /// to that point).
  void KeepRows(std::uint32_t keep);
  /// Drafts out.size() (at most 7) tokens after `anchor`, the next token to
  /// append; needs INKLING_DRAFT and Model::LoadDrafter.
  [[nodiscard]] bool Draft(std::int32_t anchor, std::span<std::int32_t> out);
  /// Also returns each drafted token's top-1 probability over the proposal
  /// vocabulary (out.size() entries).
  /// Slots after the first whose probability is below stop_confidence (> 0)
  /// are skipped on the device; their tokens and confidences are stale.
  [[nodiscard]] bool Draft(std::int32_t anchor, std::span<std::int32_t> out,
                           std::span<float> confidence,
                           float stop_confidence = 0.0F);
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
  /// Appends one token to each of 1..kMaxDecodeBatch sessions in one forward
  /// pass: the rows share every weight read, while each row keeps its own
  /// cache, convolution history, position and drafter features. All sessions
  /// must belong to this model and share one context size. Afterwards each
  /// session's Logits() / Argmax() hold its own next-token distribution.
  static constexpr std::uint32_t kMaxDecodeBatch = 8;
  [[nodiscard]] bool DecodeBatch(std::span<Session* const> sessions,
                                 std::span<const std::int32_t> tokens,
                                 std::string* error_msg = nullptr);
  /// One forward pass for 1..kMaxDecodeBatch sessions and 1..kMaxVerifyRows
  /// tokens each (rows[b] for sessions[b], concatenated in `tokens`). Each
  /// session's rows are appended at consecutive positions after its own
  /// position. Afterwards Logits(row) / Argmax(row) of each session hold its
  /// rows (row i after its i-th token) and its position advanced by rows[b].
  /// Used for batched speculative verification; call KeepRows(keep) per
  /// session to drop unaccepted rows.
  [[nodiscard]] bool StepBatch(std::span<Session* const> sessions,
                               std::span<const std::uint32_t> rows,
                               std::span<const std::int32_t> tokens,
                               std::string* error_msg = nullptr);
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;

private:
  Model() = default;
  std::shared_ptr<ModelState> state_;
};

}  // namespace gufo::models::inkling::gpu

#endif  // GUFO_MODELS_INKLING_GPU_ENGINE_HPP_
