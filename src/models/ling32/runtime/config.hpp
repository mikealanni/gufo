#ifndef GUFO_MODELS_LING32_RUNTIME_CONFIG_HPP_
#define GUFO_MODELS_LING32_RUNTIME_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::ling32 {

/// Hyperparameters of the Ling-3.2-Air (`bailingmoe3`) trunk.
///
/// The artifact mixes two attention kinds: KDA layers (`attn_q/k/v` plus the
/// `ssm_*` gated-delta recurrence) and dense MLA layers with absorbed keys.
/// `head_count_kv` is a per-layer flag array: non-zero marks an MLA layer.
struct Config {
  std::uint32_t num_layers{43};  ///< 42 trunk + 1 MTP.
  std::uint32_t nextn_layers{1};
  std::uint32_t hidden_size{2560};
  std::uint32_t vocab_size{157184};
  std::uint32_t context_length{262144};
  std::uint32_t num_heads{32};
  std::uint32_t kda_head_dim{128};
  std::uint32_t ssm_conv_kernel{4};
  std::uint32_t key_length{576};  ///< kv_lora_rank + rope_dimension.
  std::uint32_t value_length{128};
  std::uint32_t key_length_mla{192};
  std::uint32_t value_length_mla{128};
  std::uint32_t kv_lora_rank{512};
  std::uint32_t q_lora_rank{0};  ///< Optional; absent in this artifact.
  std::uint32_t rope_dimension{64};
  std::uint32_t expert_count{512};
  std::uint32_t expert_used{8};
  std::uint32_t expert_groups{8};
  std::uint32_t expert_groups_used{4};
  std::uint32_t expert_shared{1};
  std::uint32_t expert_ff{768};
  std::uint32_t shared_ff{768};
  std::uint32_t dense_ff{6144};
  std::uint32_t leading_dense{2};
  std::uint32_t gating_func{2};  ///< SIGMOID.
  float rms_eps{1e-6F};
  float freq_base{6.0e6F};
  float gate_lower_bound{-5.0F};
  float expert_weights_scale{2.5F};
  bool kda_safe_gate{true};
  bool expert_weights_norm{true};

  std::vector<std::uint32_t> rope_sections;  ///< [8, 12, 12, 0].
  std::vector<std::uint8_t> kv_layer;        ///< 1 = MLA, 0 = KDA, per layer.
  std::vector<float> swiglu_clamp_exp;       ///< Clamp per MoE layer.
  std::vector<float> swiglu_clamp_shexp;     ///< Clamp per shared expert.

  [[nodiscard]] std::uint32_t TrunkLayers() const noexcept {
    return num_layers - nextn_layers;
  }
  [[nodiscard]] std::uint32_t KdaDim() const noexcept {
    return kda_head_dim * num_heads;
  }
  [[nodiscard]] std::uint32_t MlaQDim() const noexcept {
    return key_length_mla;
  }
  [[nodiscard]] std::uint32_t MlaVDim() const noexcept {
    return value_length_mla;
  }
  [[nodiscard]] std::uint32_t QkNope() const noexcept {
    return key_length_mla - rope_dimension;
  }
  [[nodiscard]] bool IsMla(std::uint32_t layer) const noexcept {
    return layer < kv_layer.size() && kv_layer[layer] != 0;
  }
  [[nodiscard]] bool IsKda(std::uint32_t layer) const noexcept {
    return layer < kv_layer.size() && kv_layer[layer] == 0;
  }
  [[nodiscard]] bool IsDense(std::uint32_t layer) const noexcept {
    return layer < leading_dense;
  }
  [[nodiscard]] bool HasMtp() const noexcept { return nextn_layers > 0; }
  [[nodiscard]] std::uint32_t MtpLayer() const noexcept {
    return TrunkLayers();
  }
  [[nodiscard]] std::uint32_t NumMlaLayers() const noexcept {
    std::uint32_t count = 0;
    for (std::uint32_t layer = 0; layer < TrunkLayers(); ++layer) {
      count += IsMla(layer) ? 1U : 0U;
    }
    return count;
  }
  [[nodiscard]] std::uint32_t NumKdaLayers() const noexcept {
    return TrunkLayers() - NumMlaLayers();
  }
};

[[nodiscard]] std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                                std::string* error_msg);

}  // namespace gufo::models::ling32

#endif  // GUFO_MODELS_LING32_RUNTIME_CONFIG_HPP_
