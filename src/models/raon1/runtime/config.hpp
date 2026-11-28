#ifndef GUFO_MODELS_RAON1_RUNTIME_CONFIG_HPP_
#define GUFO_MODELS_RAON1_RUNTIME_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::raon1 {

/// Hyperparameters of the Raon-1 mini (`raon1`) trunk.
///
/// The artifact mixes two attention kinds: RDA layers (`attn_q/k/v` plus the
/// `ssm_*` delta-rule recurrence) and NoPE MLA layers with an absorbed latent
/// cache. `head_count_kv` is a per-layer flag array: a non-zero entry marks an
/// MLA layer. Every residual site runs the Raon mHC chart (four streams, a
/// Helmert 17-mixer).
struct Config {
  std::uint32_t num_layers{0};        ///< 25.
  std::uint32_t hidden_size{0};       ///< 2560.
  std::uint32_t num_heads{0};         ///< 40 (both attention kinds).
  std::uint32_t kv_lora_rank{0};      ///< 512, also the MLA key/value width.
  std::uint32_t q_lora_rank{0};       ///< 576.
  std::uint32_t qk_nope_head_dim{0};  ///< 128.
  std::uint32_t qk_rope_head_dim{0};  ///< 64, never rotated (NoPE).
  std::uint32_t v_head_dim{0};        ///< 128.
  std::uint32_t kda_head_dim{0};      ///< 128.
  std::uint32_t ssm_conv_kernel{0};   ///< 4.
  std::uint32_t expert_count{0};      ///< 128.
  std::uint32_t expert_used{0};       ///< 6.
  std::uint32_t expert_shared{0};     ///< 1.
  std::uint32_t expert_ff{0};         ///< 1088.
  std::uint32_t dense_ff{0};          ///< 12032.
  std::uint32_t leading_dense{0};     ///< 1.
  std::uint32_t hc_count{0};          ///< 4 residual streams.
  std::uint32_t vocab_size{0};        ///< 259840.
  std::uint32_t context_length{0};    ///< 262144.
  float rms_eps{0.0F};                ///< 1e-5.
  float latent_norm_eps{0.0F};        ///< 1e-6 (Q/KV latent projections).
  float hc_eps{0.0F};                 ///< 1e-5 (mHC selector RMS).
  float hc_magnitude{0.0F};           ///< 2.0, post gate scale.
  float situ_beta{0.0F};              ///< 4.0.
  float situ_linear_beta{0.0F};       ///< 25.0.
  float kda_gate_lower_bound{0.0F};   ///< -5.0.
  float expert_weights_scale{0.0F};   ///< 1.0.
  bool expert_weights_norm{false};
  std::vector<std::uint8_t> kv_layer;        ///< `head_count_kv` per layer.
  std::vector<std::uint32_t> effort_layers;  ///< 12, 14, 16, 18, 20.
  std::vector<float> effort_alphas;          ///< low..max.
  std::vector<std::string> effort_names;     ///< "low".."max".
  std::string effort_default;                ///< "high".

  [[nodiscard]] bool IsMla(std::uint32_t layer) const noexcept {
    return layer < kv_layer.size() && kv_layer[layer] != 0;
  }
  [[nodiscard]] bool IsRda(std::uint32_t layer) const noexcept {
    return layer < num_layers && !IsMla(layer);
  }
  [[nodiscard]] bool IsDense(std::uint32_t layer) const noexcept {
    return layer < leading_dense;
  }
  /// mHC mixer width: pre (hc) + post (hc) + Helmert chart ((hc - 1)^2).
  [[nodiscard]] std::uint32_t HcRank() const noexcept {
    return 2 * hc_count + (hc_count - 1) * (hc_count - 1);
  }
  [[nodiscard]] std::uint32_t HcDim() const noexcept {
    return hc_count * hidden_size;
  }
  /// RDA projection width (heads * head_dim).
  [[nodiscard]] std::uint32_t KdaDim() const noexcept {
    return num_heads * kda_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmChannels() const noexcept {
    return 3 * KdaDim();
  }
  [[nodiscard]] std::uint32_t MlaHeadKey() const noexcept {
    return qk_nope_head_dim + qk_rope_head_dim;
  }
  [[nodiscard]] std::uint32_t MlaQDim() const noexcept {
    return num_heads * MlaHeadKey();
  }
  [[nodiscard]] std::uint32_t MlaVDim() const noexcept {
    return num_heads * v_head_dim;
  }
  [[nodiscard]] std::uint32_t NumMlaLayers() const noexcept;
  [[nodiscard]] std::uint32_t NumRdaLayers() const noexcept;
};

/// Parses `raon1.*` metadata and pins the Raon-1 mini shape. Returns nullopt
/// and a message on any mismatch, so a foreign revision fails at load rather
/// than inside the kernels.
[[nodiscard]] std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                                std::string* error_msg);

}  // namespace gufo::models::raon1

#endif  // GUFO_MODELS_RAON1_RUNTIME_CONFIG_HPP_
