#ifndef GUFO_NPU_MODEL_SPEC_HPP_
#define GUFO_NPU_MODEL_SPEC_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gufo::npu {

enum class LayerKind { kAttention, kConvolution };
enum class RopeKind { kNone, kNormal, kScaled };
enum class FfnKind { kSwiGLU, kGeGLU, kPlain };
enum class BitWidth { kInt4, kInt5, kInt6, kBf16, kQ8 };

struct GemmOp {
  std::string name;           // logical name / program key
  std::string weight;         // tensor base name (without layer prefix) or full name
  std::uint32_t K{0};
  std::uint32_t N{0};
  std::string mode{"plain"};  // "plain" or "ffn" (phase2) - used for inst selection
  // For fused FFN: gate/up/down packed together; handled specially.
  bool fused_ffn{false};
};

struct AttentionOp {
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t window{0};  // 0 => global
  RopeKind rope{RopeKind::kNormal};
  float rope_theta{10000.0f};
  bool qk_norm{false};
  bool has_gate{true};  // LFM3-style per-head input gate
  std::string q_proj{"self_attn.q_proj.weight"};
  std::string k_proj{"self_attn.k_proj.weight"};
  std::string v_proj{"self_attn.v_proj.weight"};
  std::string o_proj{"self_attn.o_proj.weight"};
  std::string q_norm{"self_attn.q_norm.weight"};
  std::string k_norm{"self_attn.k_norm.weight"};
  std::string attn_gate{"self_attn.gate.weight"};
};

struct ConvOp {
  std::uint32_t width{0};  // taps
  std::string in_proj{"conv.in_proj.weight"};
  std::string out_proj{"conv.out_proj.weight"};
  std::string conv_weight{"conv.conv.weight"};
};

struct FfnOp {
  FfnKind kind{FfnKind::kSwiGLU};
  std::uint32_t intermediate{0};
  std::string gate{"feed_forward.w1.weight"};
  std::string up{"feed_forward.w3.weight"};
  std::string down{"feed_forward.w2.weight"};
};

struct LayerSpec {
  LayerKind kind{LayerKind::kAttention};
  std::string norm{"operator_norm.weight"};
  std::string post_norm;      // empty => residual add without post norm
  std::string ffn_norm{"ffn_norm.weight"};
  std::string post_ffn_norm;  // empty => residual add without post norm
  AttentionOp attention;
  ConvOp conv;
  FfnOp ffn;
};

struct EmbeddingSpec {
  std::string tensor{"token_embd.weight"};
  BitWidth bits{BitWidth::kQ8};
  bool tied_output{true};
};

struct ModelSpec {
  std::string name;
  std::uint32_t hidden{0};
  std::uint32_t vocab{0};
  std::uint32_t layers{0};
  std::uint32_t context{0};
  float norm_eps{1e-5f};
  std::vector<LayerSpec> layer_specs;  // per-layer (can be 1 pattern repeated) or explicit
  std::string output_norm{"output_norm.weight"};
  EmbeddingSpec embedding;
  std::uint32_t eos_id{0};
  std::vector<std::uint32_t> stop_ids;
  // rope scaling / freqs tensor name if any
  std::string rope_freqs;
  float rope_theta_swa{10000.0f};
  float rope_theta{10000.0f};

  /// Distinct GEMM (K,N,mode) programs needed by this model.
  struct ProgramNeed {
    std::uint32_t K, N;
    std::string mode;  // plain or ffn
    std::string inst_file;
  };
  std::vector<ProgramNeed> RequiredPrograms() const;

  /// Validate shape constraints for the current GEMM program (K%64==0, N%1024==0, rows<=16).
  /// Returns list of violation messages (empty => ok, possibly after padding).
  std::vector<std::string> ValidateConstraints() const;

  static std::optional<ModelSpec> LoadJsonFile(const std::string& path,
                                               std::string* error);
  /// Build the canonical LFM3-1.7B spec (matches current hardcoded behavior).
  static ModelSpec Lfm3_1_7B();
};

}  // namespace gufo::npu

#endif  // GUFO_NPU_MODEL_SPEC_HPP_
