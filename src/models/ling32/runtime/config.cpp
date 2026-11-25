#include "src/models/ling32/runtime/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::ling32 {
namespace {

constexpr std::string_view kArchitecture = "bailingmoe3";

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

  bool Bool(std::string_view key) {
    const auto value = gguf.GetMetadataBool(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF bool " + std::string(key));
      return false;
    }
    return *value;
  }

  std::vector<float> F32Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<float> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* d = std::get_if<std::vector<double>>(&meta->value)) {
      out.reserve(d->size());
      for (const auto value : *d) {
        out.push_back(static_cast<float>(value));
      }
    } else if (const auto* u =
                   std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out.reserve(u->size());
      for (const auto value : *u) {
        out.push_back(static_cast<float>(value));
      }
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      out.reserve(s->size());
      for (const auto value : *s) {
        out.push_back(static_cast<float>(value));
      }
    } else {
      Fail("GGUF array must contain numbers: " + std::string(key));
    }
    return out;
  }

  std::vector<std::uint32_t> U32Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint32_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out.reserve(u->size());
      for (const auto value : *u) {
        out.push_back(static_cast<std::uint32_t>(value));
      }
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      out.reserve(s->size());
      for (const auto value : *s) {
        out.push_back(static_cast<std::uint32_t>(value));
      }
    } else if (const auto* d = std::get_if<std::vector<double>>(&meta->value)) {
      out.reserve(d->size());
      for (const auto value : *d) {
        out.push_back(static_cast<std::uint32_t>(value));
      }
    } else {
      Fail("GGUF array must contain numbers: " + std::string(key));
    }
    return out;
  }
};

void Expect(Reader& r, std::string_view name, std::uint64_t value,
            std::uint64_t expected) {
  if (value != expected) {
    r.Fail("Ling-3.2: " + std::string(name) + " is " + std::to_string(value) +
           ", this build supports " + std::to_string(expected));
  }
}

}  // namespace

std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                  std::string* error_msg) {
  Reader r{reader, error_msg};
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != kArchitecture) {
    r.Fail("artifact is not bailingmoe3");
    return std::nullopt;
  }

  Config c;
  const auto key = [](std::string_view name) {
    return std::string(kArchitecture) + "." + std::string(name);
  };
  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.key_length = r.U32(key("attention.key_length"));
  c.value_length = r.U32(key("attention.value_length"));
  c.key_length_mla = r.U32(key("attention.key_length_mla"));
  c.value_length_mla = r.U32(key("attention.value_length_mla"));
  c.kv_lora_rank = r.U32(key("attention.kv_lora_rank"));
  c.rope_dimension = r.U32(key("rope.dimension_count"));
  c.kda_head_dim = r.U32(key("kda.head_dim"));
  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.expert_count = r.U32(key("expert_count"));
  c.expert_used = r.U32(key("expert_used_count"));
  c.expert_groups = r.U32(key("expert_group_count"));
  c.expert_groups_used = r.U32(key("expert_group_used_count"));
  c.expert_shared = r.U32(key("expert_shared_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_ff = r.U32(key("expert_shared_feed_forward_length"));
  c.dense_ff = r.U32(key("feed_forward_length"));
  c.leading_dense = r.U32(key("leading_dense_block_count"));
  c.gating_func = r.U32(key("expert_gating_func"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"));
  c.freq_base = r.F32(key("rope.freq_base"));
  c.gate_lower_bound = r.F32(key("kda.gate_lower_bound"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"));
  c.expert_weights_norm = r.Bool(key("expert_weights_norm"));
  const auto safe = reader.GetMetadataBool(key("kda.safe_gate"));
  c.kda_safe_gate = safe.value_or(true);

  c.rope_sections = r.U32Array(key("rope.dimension_sections"));
  const auto kv = r.U32Array(key("attention.head_count_kv"));
  c.kv_layer.reserve(kv.size());
  for (const auto value : kv) {
    c.kv_layer.push_back(value != 0 ? 1U : 0U);
  }
  c.swiglu_clamp_exp = r.F32Array(key("swiglu_clamp_exp"));
  c.swiglu_clamp_shexp = r.F32Array(key("swiglu_clamp_shexp"));
  if (!r.ok) {
    return std::nullopt;
  }

  Expect(r, "block_count", c.num_layers, 43);
  Expect(r, "nextn_predict_layers", c.nextn_layers, 1);
  Expect(r, "embedding_length", c.hidden_size, 2560);
  Expect(r, "vocab_size", c.vocab_size, 157184);
  Expect(r, "attention.head_count", c.num_heads, 32);
  Expect(r, "attention.key_length", c.key_length, 576);
  Expect(r, "attention.value_length", c.value_length, 128);
  Expect(r, "attention.key_length_mla", c.key_length_mla, 192);
  Expect(r, "attention.value_length_mla", c.value_length_mla, 128);
  Expect(r, "attention.kv_lora_rank", c.kv_lora_rank, 512);
  Expect(r, "rope.dimension_count", c.rope_dimension, 64);
  Expect(r, "kda.head_dim", c.kda_head_dim, 128);
  Expect(r, "ssm.conv_kernel", c.ssm_conv_kernel, 4);
  Expect(r, "expert_count", c.expert_count, 512);
  Expect(r, "expert_used_count", c.expert_used, 8);
  Expect(r, "expert_group_count", c.expert_groups, 8);
  Expect(r, "expert_group_used_count", c.expert_groups_used, 4);
  Expect(r, "expert_shared_count", c.expert_shared, 1);
  Expect(r, "expert_feed_forward_length", c.expert_ff, 768);
  Expect(r, "expert_shared_feed_forward_length", c.shared_ff, 768);
  Expect(r, "feed_forward_length", c.dense_ff, 6144);
  Expect(r, "leading_dense_block_count", c.leading_dense, 2);
  Expect(r, "expert_gating_func", c.gating_func, 2);
  Expect(r, "kv_layer entries", c.kv_layer.size(), c.num_layers);
  Expect(r, "rope_sections entries", c.rope_sections.size(), 4);
  Expect(r, "swiglu_clamp_exp entries", c.swiglu_clamp_exp.size(),
         c.num_layers);
  Expect(r, "swiglu_clamp_shexp entries", c.swiglu_clamp_shexp.size(),
         c.num_layers);
  Expect(r, "trunk MLA layers", c.NumMlaLayers(), 7);
  Expect(r, "trunk KDA layers", c.NumKdaLayers(), 35);
  if (c.rope_sections != std::vector<std::uint32_t>({8, 12, 12, 0})) {
    r.Fail("Ling-3.2: unexpected rope.dimension_sections");
  }
  if (!c.kda_safe_gate || !(c.gate_lower_bound < 0.0F)) {
    r.Fail("Ling-3.2: unsupported KDA gate configuration");
  }
  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::ling32
