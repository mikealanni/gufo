#include "src/models/inkling/runtime/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>

namespace gufo::models::inkling {
namespace {

constexpr std::string_view kArchitecture = "inkling2";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key) {
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + std::string(key));
      return 0;
    }
    return *value;
  }

  float F32(std::string_view key) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + std::string(key));
      return 0.0F;
    }
    return *value;
  }

  bool Bool(std::string_view key, bool fallback) {
    const auto value = gguf.GetMetadataBool(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF bool " + std::string(key));
      return fallback;
    }
    return *value;
  }

  std::vector<bool> BoolArray(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<bool> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out.reserve(u->size());
      for (const auto value : *u) {
        out.push_back(value != 0);
      }
    } else if (const auto* b =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      out.reserve(b->size());
      for (const auto value : *b) {
        out.push_back(value != 0);
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

void Expect(Reader& r, std::string_view name, std::uint64_t value,
            std::uint64_t expected) {
  if (value != expected) {
    r.Fail(std::string("Inkling: ") + std::string(name) + " is " +
           std::to_string(value) + ", this build supports " +
           std::to_string(expected));
  }
}

}  // namespace

std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                  std::string* error_msg) {
  Reader r{reader, error_msg};
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture || *architecture != kArchitecture) {
    r.Fail("artifact is not inkling2");
    return std::nullopt;
  }

  Config c;
  const auto key = [](std::string_view name) {
    return std::string(kArchitecture) + "." + std::string(name);
  };
  c.num_layers = r.U32(key("block_count"));
  c.context_length = r.U32(key("context_length"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.dense_ff = r.U32(key("feed_forward_length"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  c.expert_count = r.U32(key("expert_count"));
  c.expert_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_ff = r.U32(key("expert_shared_feed_forward_length"));
  c.leading_dense = r.U32(key("leading_dense_block_count"));
  c.expert_shared = r.U32(key("expert_shared_count"));
  c.sliding_window = r.U32(key("attention.sliding_window"));
  c.loop_num_loops = r.U32(key("num_loops"));
  c.loop_core_start = r.U32(key("loop.core_start"));
  c.loop_core_end = r.U32(key("loop.core_end"));
  c.loop_gate_rank = r.U32(key("loop.gate_rank"));
  c.attn_res_block_size = r.U32(key("attn_res.block_size"));
  c.shortconv_l_cache = r.U32(key("shortconv.l_cache"));
  c.relative_length = r.U32(key("attention.relative_length"));
  c.relative_extent = r.U32(key("attention.relative_extent"));
  c.log_scale_floor = r.U32(key("attention.log_scale_floor"));
  c.log_scale_alpha = r.F32(key("attention.log_scale_alpha"));
  c.qk_clip = r.F32(key("attention.qk_clip"));
  c.relative_clip = r.F32(key("attention.relative_clip"));
  c.final_logit_softcapping = r.F32(key("final_logit_softcapping"));
  c.situ_beta = r.F32(key("activation.situ_beta"));
  c.situ_linear_beta = r.F32(key("activation.situ_linear_beta"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"));
  c.expert_weights_norm = r.Bool(key("expert_weights_norm"), true);
  c.expert_gating_func = r.U32(key("expert_gating_func"));
  c.rope_dimension_count = r.U32(key("rope.dimension_count"));
  c.has_relative = c.relative_length > 0;
  c.is_swa = r.BoolArray(key("attention.sliding_window_pattern"));

  Expect(r, "block_count", c.num_layers, 48);
  Expect(r, "embedding_length", c.hidden_size, 2304);
  Expect(r, "feed_forward_length", c.dense_ff, 1152);
  Expect(r, "attention.head_count", c.num_heads, 24);
  Expect(r, "attention.head_count_kv", c.num_kv, 8);
  Expect(r, "attention.key_length", c.head_dim, 128);
  Expect(r, "expert_count", c.expert_count, 128);
  Expect(r, "expert_used_count", c.expert_used, 8);
  Expect(r, "expert_feed_forward_length", c.expert_ff, 1152);
  Expect(r, "attention.sliding_window", c.sliding_window, 1024);
  Expect(r, "loop_num_loops", c.loop_num_loops, 3);
  Expect(r, "loop.core_start", c.loop_core_start, 6);
  Expect(r, "loop.core_end", c.loop_core_end, 18);
  Expect(r, "loop.gate_rank", c.loop_gate_rank, 64);
  Expect(r, "sliding_window_pattern entries", c.is_swa.size(), c.num_layers);

  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::inkling
