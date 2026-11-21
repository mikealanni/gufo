#ifndef GUFO_MODELS_INKLING_REFERENCE_HPP_
#define GUFO_MODELS_INKLING_REFERENCE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/inkling/runtime/config.hpp"

namespace gufo::models::inkling {

// Correctness-first float32 forward for Inkling 2 on CPU. One token at a time;
// every format dequantized on demand. Not tuned; GPU/NPU paths replace it.
class ReferenceModel {
public:
  static std::unique_ptr<ReferenceModel> Open(const std::string& path,
                                              std::string* error);

  struct LayerState {
    // Per-layer KV cache: k/v held flat, token stride last.
    std::vector<float> k;
    std::vector<float> v;
    std::uint32_t n_tokens{0};
    // short-convolution histories (last n_shortconv_l_cache - 1 stream values)
    std::vector<float> conv_k;
    std::vector<float> conv_v;
    std::vector<float> conv_attn;
    std::vector<float> conv_ffn;
  };

  struct Session {
    std::vector<float> x;  // residual stream [hidden]
    std::vector<float> logits;
    std::uint32_t position{0};
    std::vector<LayerState> layers;
  };

  std::unique_ptr<Session> CreateSession() const;
  const std::vector<float>& Evaluate(Session& session, std::int32_t token) const;
  std::uint32_t n_layers() const { return config_.num_layers; }
  const Config& config() const { return config_; }

private:
  ReferenceModel() = default;

  struct Loaded {
    Loaded() = default;
    explicit Loaded(const gufo::core::GgufTensorInfo* t);
    static Loaded RowView(const gufo::core::GgufTensorInfo* t);

    void DequantizeRow(std::uint64_t row, float* out) const;
    void MatVec(std::span<const float> x, std::span<float> y) const;
    void MatVecRow(std::uint64_t row, std::span<const float> x, float* out) const;
    [[nodiscard]] std::uint64_t rows() const { return rows_; }
    [[nodiscard]] std::uint64_t cols() const { return cols_; }
    [[nodiscard]] gufo::core::GgmlType type() const { return type_; }
    [[nodiscard]] const std::vector<float>& data() const { return data_; }

  private:
    void Setup(const gufo::core::GgufTensorInfo* t);
    const gufo::core::GgufTensorInfo* t_{nullptr};
    gufo::core::GgmlType type_{gufo::core::GgmlType::kF16};
    std::uint64_t rows_{0};
    std::uint64_t cols_{0};
    std::vector<float> data_;
  };

  const Loaded& Load(const std::string& name) const;
  const Loaded& LoadRow(const std::string& name) const;
  const gufo::core::GgufTensorInfo* Find(const std::string& suffix) const;

  void Rms(std::span<float> x, const float* w) const;
  void RmsNorm(std::span<float> x) const;

  std::vector<float> Mix(std::uint32_t hidden, const std::vector<float>& cur,
                         const std::vector<float>& query,
                         const std::vector<std::vector<float>>& bank) const;

  std::vector<float> RunLayer(std::uint32_t slot, std::uint32_t ip,
                              std::vector<float> cur, bool has_messages,
                              const std::vector<std::vector<float>>& bank,
                              const std::vector<float>& bank_out, Session& session,
                              std::uint32_t pos,
                              std::vector<std::vector<float>>& blocks) const;

  std::unique_ptr<gufo::core::GgufReader> reader_;
  std::unordered_map<std::string, const gufo::core::GgufTensorInfo*> tensors_;
  Config config_;
  mutable std::unordered_map<std::string, Loaded> cache_;
};

}  // namespace gufo::models::inkling

#endif  // GUFO_MODELS_INKLING_REFERENCE_HPP_
