#ifndef GUFO_MODELS_INKLING_RUNTIME_CONFIG_HPP_
#define GUFO_MODELS_INKLING_RUNTIME_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::inkling {

struct Config {
  // Slot count of the artifact (loop expansion included): 24 physical layers
  // plus one slot per extra pass over the looped core.
  std::uint32_t num_layers{0};
  std::uint32_t context_length{0};
  std::uint32_t hidden_size{0};
  std::uint32_t dense_ff{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv{0};
  std::uint32_t head_dim{0};
  std::uint32_t expert_count{0};
  std::uint32_t expert_used{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_ff{0};
  std::uint32_t leading_dense{0};
  std::uint32_t expert_shared{0};
  std::uint32_t sliding_window{0};

  // Inkling 2 looped-transformer controls.
  std::uint32_t loop_num_loops{0};
  std::uint32_t loop_core_start{0};
  std::uint32_t loop_core_end{0};
  std::uint32_t loop_gate_rank{0};

  // Inkling 2 hybrid-attention controls.
  std::uint32_t attn_res_block_size{0};
  std::uint32_t shortconv_l_cache{0};
  std::uint32_t relative_length{0};
  std::uint32_t relative_extent{0};
  std::uint32_t log_scale_floor{0};
  float log_scale_alpha{0.0F};
  float qk_clip{0.0F};
  float relative_clip{0.0F};
  float final_logit_softcapping{0.0F};
  float situ_beta{0.0F};
  float situ_linear_beta{0.0F};
  float rms_eps{0.0F};
  float expert_weights_scale{1.0F};
  float expert_weights_norm{1.0F};
  std::uint32_t vocab_padded{0};
  std::uint32_t vocab_unpadded{0};
  std::uint32_t rope_dimension_count{0};
  bool has_relative{false};
  std::vector<bool> is_swa;

  [[nodiscard]] bool HasRope() const { return rope_dimension_count > 0; }
  [[nodiscard]] bool HasRelative() const { return has_relative; }
};

[[nodiscard]] std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                                std::string* error_msg);

}  // namespace gufo::models::inkling

#endif  // GUFO_MODELS_INKLING_RUNTIME_CONFIG_HPP_
