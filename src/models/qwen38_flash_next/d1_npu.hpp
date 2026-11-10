#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_D1_NPU_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_D1_NPU_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/lfm3/weights.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

/// d1 decision backbone (LFM3 layout) with every projection on the XDNA2 NPU
/// and the rest on the GPU. One prefill over at most 512 prompt tokens; the
/// logits at the requested answer slots are returned for a chosen vocabulary
/// subset. Matrices must be Q4_0; the embedding Q6_K or Q4_0. The GGUF reader
/// must outlive the model (tensors borrow its mapping).
class D1NpuModel {
public:
  ~D1NpuModel();
  D1NpuModel(const D1NpuModel&) = delete;
  D1NpuModel& operator=(const D1NpuModel&) = delete;

  [[nodiscard]] static std::unique_ptr<D1NpuModel> Create(
      const core::GgufReader& reader, std::string* error = nullptr);

  [[nodiscard]] const lfm3::Weights& weights() const;

  /// `groups` is empty for a lone question (full bidirectional global
  /// attention) or one tag per token (-1 state, 1.. question ids).
  /// `sequences` lists ascending start offsets of independent sequences
  /// packed into the one pass (empty = a single sequence); tokens never see,
  /// convolve with or rotate relative to another sequence.
  /// `logits` receives slots.size() * vocabulary.size() floats.
  [[nodiscard]] bool Forward(std::span<const std::uint32_t> tokens,
                             std::span<const std::int32_t> groups,
                             std::span<const std::uint32_t> sequences,
                             std::span<const std::uint32_t> slots,
                             std::span<const std::uint32_t> vocabulary,
                             std::span<float> logits) const;

  /// Greedy generation on the NPU (causal LFM3 models). EnableGeneration
  /// allocates the K/V caches and conv state; Step runs one pass of at most
  /// 512 tokens continuing the stream and returns the greedy next token after
  /// each of its last next_tokens.size() rows; Commit keeps the first
  /// `accepted` tokens of that pass (the rest are rolled back).
  [[nodiscard]] bool EnableGeneration(std::uint32_t max_context,
                                      std::string* error = nullptr);
  void ResetState() const;
  [[nodiscard]] std::uint32_t Position() const;
  [[nodiscard]] std::uint32_t MaxContext() const;
  /// Rows one NPU pass computes (4 x the program tile: 512 or 256).
  [[nodiscard]] std::uint32_t PassRows() const;
  [[nodiscard]] bool Step(std::span<const std::uint32_t> tokens,
                          std::span<std::uint32_t> next_tokens) const;
  void Commit(std::uint32_t accepted) const;
  /// Debug: the final-norm hidden rows of the last Step (rows x hidden floats).
  [[nodiscard]] bool LastHidden(std::vector<float>* out, std::uint32_t rows) const;

  /// Teacher-forced pass over `tokens` (any length, chunked): the final-norm
  /// hidden state of every row as fp16 bits (n x hidden), for drafter
  /// training. Resets and leaves the stream state at the end of `tokens`.
  [[nodiscard]] bool CaptureHidden(std::span<const std::uint32_t> tokens,
                                   std::vector<std::uint16_t>* hidden) const;

  /// Batched greedy decode for data generation and concurrent serving: up to
  /// `streams` independent streams (one row each) share every NPU pass. A
  /// stream's state is loaded from the single-stream state (after Step +
  /// Commit of its prompt); BatchStep feeds one token per stream.
  [[nodiscard]] bool EnableBatch(std::uint32_t streams, std::uint32_t ctx,
                                 std::string* error = nullptr);
  [[nodiscard]] bool BatchLoad(std::uint32_t stream, std::uint32_t length) const;
  [[nodiscard]] bool BatchStep(std::span<const std::uint32_t> tokens,
                               std::span<std::uint32_t> next) const;

  /// Medusa-style drafter heads (K residual-SiLU blocks over the final-norm
  /// hidden, tied to the embedding head) written by the training script:
  /// u32 K, u32 hidden, then K x hidden x hidden fp16 weights [out][in].
  [[nodiscard]] bool LoadDrafter(const std::string& path,
                                 std::string* error = nullptr);
  [[nodiscard]] std::uint32_t DrafterHeads() const;
  /// Tokens that follow the model's own next token, from the final-norm hidden
  /// of row `row` among the rows returned by the last Step.
  [[nodiscard]] bool Draft(std::uint32_t row,
                           std::vector<std::uint32_t>* drafts) const;

  struct GenerateOptions {
    std::uint32_t max_new{128};
    std::vector<std::uint32_t> stop;  // token ids that end generation
    std::uint32_t draft_max{15};      // prompt-lookup draft length per pass
    std::uint32_t ngram{3};           // match length for prompt lookup
    bool use_heads{true};             // drafter heads when loaded
  };
  struct GenerateStats {
    std::uint32_t passes{0}, prompt_tokens{0}, new_tokens{0}, drafted{0},
        accepted{0};
    double prefill_ms{0}, decode_ms{0};
  };
  /// Prefill `prompt` then decode greedily with prompt-lookup speculation.
  [[nodiscard]] bool Generate(std::span<const std::uint32_t> prompt,
                              const GenerateOptions& options,
                              std::vector<std::uint32_t>* output,
                              GenerateStats* stats = nullptr) const;

private:
  struct Impl;
  explicit D1NpuModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_D1_NPU_HPP_
