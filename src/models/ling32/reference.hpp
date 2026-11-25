#ifndef GUFO_MODELS_LING32_REFERENCE_HPP_
#define GUFO_MODELS_LING32_REFERENCE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling32/runtime/config.hpp"

namespace gufo::models::ling32 {

// Correctness-first float32 forward for Ling-3.2-Air (`bailingmoe3`) on CPU.
// One token at a time, every format dequantized on demand. Mirrors the pinned
// upstream llama.cpp graph; not tuned, GPU paths replace it.
class ReferenceModel {
public:
  static std::unique_ptr<ReferenceModel> Open(const std::string& path,
                                              std::string* error);

  struct LayerState {
    std::vector<float> conv;   // 3 * d_inner (KDA conv history, d_conv-1 taps)
    std::vector<float> state;  // heads * hd * hd (KDA delta state)
    std::vector<float>
        lat;                // n * kv_lora_rank (MLA absorbed keys, includes pe)
    std::vector<float> pe;  // n * rope_dim (MLA rope keys)
  };

  // Streaming state for one sequence. Evaluate() advances it one token.
  struct Session {
    std::vector<float> x;           // hidden_size residual
    std::vector<float> logits;      // vocab
    std::vector<float> mtp_logits;  // vocab
    std::vector<float> mtp_h;       // normalized hidden from last trunk token
    std::vector<float> mtp_prev_h;  // normalized hidden from previous token
    std::vector<float> mtp_pending_h;  // hidden paired with the last token
    std::int32_t mtp_prev_token{-1};
    bool mtp_has_prev{false};
    std::uint32_t position{0};
    std::vector<LayerState> layers;
  };

  std::unique_ptr<Session> CreateSession() const;

  // Runs one token through the trunk and returns fresh logits.
  const std::vector<float>& Evaluate(Session& session,
                                     std::int32_t token) const;

  const std::vector<float>& MtpEvaluate(Session& session, std::int32_t token,
                                        std::span<const float> h_prev,
                                        std::uint32_t position) const;

  std::uint32_t n_layers() const;

private:
  ReferenceModel() = default;

  // F32 tensor materialized on demand (or row-view only for huge tensors).
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

  void AttnBlock(std::uint32_t layer, Session& session,
                 std::span<const float> xn, std::span<float> block) const;
  void KdaBlock(std::uint32_t layer, Session& session,
                std::span<const float> xn, std::span<float> block) const;
  void MlaBlock(std::uint32_t layer, Session& session,
                std::span<const float> xn, std::span<float> block) const;
  const std::vector<float>& MtpForward(Session& session, std::int32_t token,
                                       std::span<const float> h_prev,
                                       std::uint32_t position,
                                       bool emit_logits) const;
  void FfnBlock(std::uint32_t layer, std::span<const float> fin,
                std::span<float> ffn_out) const;

  std::unique_ptr<gufo::core::GgufReader> reader_;
  std::unordered_map<std::string, const gufo::core::GgufTensorInfo*> tensors_;
  Config config_;
  mutable std::unordered_map<std::string, Loaded> cache_;
};

}  // namespace gufo::models::ling32

#endif  // GUFO_MODELS_LING32_REFERENCE_HPP_
