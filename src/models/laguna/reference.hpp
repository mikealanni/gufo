#ifndef GUFO_MODELS_LAGUNA_REFERENCE_HPP_
#define GUFO_MODELS_LAGUNA_REFERENCE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/runtime/config.hpp"

namespace gufo::models::laguna {

// Correctness-first float32 forward for Laguna on CPU. One token at a time;
// every format dequantized on demand. Not tuned; GPU/NPU paths replace it.
class ReferenceModel {
public:
  static std::unique_ptr<ReferenceModel> Open(const std::string& path,
                                              std::string* error);

  struct LayerState {
    // Per-layer KV cache: k/v held as head-major [n_tokens][n_kv][head_dim]
    // flattened to [n_kv * head_dim * n_tokens] with token stride last.
    std::vector<float> k;
    std::vector<float> v;
    std::uint32_t n_tokens{0};
  };

  struct Session {
    std::vector<float> x;  // residual stream [hidden]
    std::vector<float> logits;
    std::uint32_t position{0};
    std::vector<LayerState> layers;
  };

  std::unique_ptr<Session> CreateSession() const;

  const std::vector<float>& Evaluate(Session& session,
                                     std::int32_t token) const;

  std::uint32_t n_layers() const { return config_.num_layers; }
  const Config& config() const { return config_; }

private:
  ReferenceModel() = default;

  struct Loaded {
  public:
    Loaded() = default;
    explicit Loaded(const gufo::core::GgufTensorInfo* t);
    static Loaded RowView(const gufo::core::GgufTensorInfo* t);

    void DequantizeRow(std::uint64_t row, float* out) const;
    void MatVec(std::span<const float> x, std::span<float> y) const;
    void MatVecRows(std::uint64_t row0, std::uint64_t nrows,
                    std::span<const float> x, std::span<float> y) const;
    void MatVecView(std::uint64_t row0, std::uint64_t nrows,
                    std::span<const float> x, std::span<float> y) const;

    std::uint64_t rows() const { return rows_; }
    std::uint64_t cols() const { return cols_; }
    const std::vector<float>& data() const { return data_; }
    const float* w() const { return data_.data(); }

  private:
    void Setup(const gufo::core::GgufTensorInfo* t);

    const gufo::core::GgufTensorInfo* t_{nullptr};
    std::uint64_t rows_{0}, cols_{0};
    std::vector<float> data_;
  };

  const Loaded& Load(const std::string& name) const;
  const Loaded& LoadRow(const std::string& name) const;

  void Rms(std::span<float> x, const float* w) const;
  void Rope(std::uint32_t layer, std::span<float> q_or_k,
            std::uint32_t pos) const;

  void Attention(std::uint32_t layer, Session& session,
                 std::span<const float> xn, std::span<float> out) const;
  void DenseFfn(std::uint32_t layer, std::span<const float> in,
                std::span<float> out) const;
  void MoEFfn(std::uint32_t layer, Session& session, std::span<const float> in,
              std::span<float> out) const;

  std::unique_ptr<gufo::core::GgufReader> reader_;
  std::unordered_map<std::string, const gufo::core::GgufTensorInfo*> tensors_;
  Config config_;
  mutable std::unordered_map<std::string, Loaded> cache_;
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_REFERENCE_HPP_
