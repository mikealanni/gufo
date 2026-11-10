#ifndef GUFO_MODELS_LFM3_NPU_ENGINE_HPP_
#define GUFO_MODELS_LFM3_NPU_ENGINE_HPP_
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::lfm3::npu {

struct Options {
  std::string gguf;
  std::string bf16;
  std::string programs;
  std::uint32_t context{8192};
};

/// LFM3 engine with every dense projection on the NPU (16-row int5 split-B
/// GEMV programs) and all other work on the host CPU. No GPU is touched.
class Engine {
 public:
  static constexpr std::uint32_t kMaxRows = 16;
  ~Engine();
  static std::unique_ptr<Engine> Create(const Options& options,
                                        std::string* error);
  void Reset();
  std::uint32_t Position() const;
  /// Runs up to kMaxRows tokens from the current position. `hidden` (optional)
  /// receives the final-norm rows, tokens.size() x hidden floats.
  bool Step(std::span<const std::uint32_t> tokens, std::vector<float>* hidden);
  /// Logits for one final-norm row against the tied Q8_0 embedding.
  void Logits(const float* hidden_row, std::vector<float>* logits);
  /// Index of the largest logit (greedy token), without storing all logits.
  std::uint32_t Argmax(const float* hidden_row);
  /// Greedy token per hidden row in one sweep over the embedding.
  void ArgmaxRows(const float* hidden, std::uint32_t rows, std::uint32_t* out);
  /// Accepts the first k rows of the last Step (conv history and position).
  void Commit(std::uint32_t k);
  bool LoadHeads(const std::string& path, std::string* error);
  std::uint32_t HeadCount() const;
  void LimitHeads(std::uint32_t k);
  /// Greedy continuation tokens (after the token the hidden row predicts).
  void Draft(const float* hidden_row, std::uint32_t* out);
  struct Impl;

 private:
  explicit Engine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::lfm3::npu
#endif
