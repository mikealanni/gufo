#ifndef GUFO_MODELS_LAGUNA_RUNTIME_CONFIG_HPP_
#define GUFO_MODELS_LAGUNA_RUNTIME_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::laguna {

struct Config {
  std::uint32_t num_layers{0};
  std::uint32_t context_length{0};
  std::uint32_t hidden_size{0};
  std::uint32_t dense_ff{0};
  std::uint32_t num_kv{0};
  std::uint32_t head_dim{0};
  std::uint32_t vocab_size{0};
  std::uint32_t expert_count{0};
  std::uint32_t expert_used{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_ff{0};
  std::uint32_t leading_dense{0};
  std::uint32_t sliding_window{0};
  std::uint32_t swa_period{4};
  std::uint32_t rope_dims_full{0};
  std::uint32_t rope_dims_swa{0};
  std::uint32_t yarn_orig_ctx{0};
  float rms_eps{0.0F};
  float rope_base{0.0F};
  float rope_base_swa{0.0F};
  float yarn_factor{1.0F};
  float yarn_attn_factor{1.0F};
  float yarn_beta_fast{32.0F};
  float yarn_beta_slow{1.0F};
  float expert_weights_scale{1.0F};
  bool expert_weights_norm{true};
  std::uint32_t expert_gating_func{0};
  std::vector<std::uint32_t> heads;

  [[nodiscard]] std::uint32_t Heads(std::uint32_t layer) const {
    return heads[layer];
  }
  [[nodiscard]] bool IsSwa(std::uint32_t layer) const {
    return layer % swa_period != 0;
  }
  [[nodiscard]] std::uint32_t NumMoeLayers() const {
    return num_layers > leading_dense ? num_layers - leading_dense : 0;
  }
};

[[nodiscard]] std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                                std::string* error_msg);

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_RUNTIME_CONFIG_HPP_
