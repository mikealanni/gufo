#include "src/models/raon1/runtime/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::raon1 {
namespace {

constexpr std::string_view kArchitecture = "raon1";

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

  std::vector<std::uint64_t> U64Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto value : *s) {
        if (value < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(value));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }

  std::vector<float> F32Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<float> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* d = std::get_if<std::vector<double>>(&meta->value)) {
      for (const auto value : *d) {
        out.push_back(static_cast<float>(value));
      }
    } else if (const auto* u =
                   std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (const auto value : *u) {
        out.push_back(static_cast<float>(value));
      }
    } else {
      Fail("GGUF array must contain floats: " + std::string(key));
    }
    return out;
  }

  std::string String(std::string_view key, std::string fallback) {
    const auto value = gguf.GetMetadataString(key);
    if (!value.has_value()) {
      return fallback;
    }
    return std::string(*value);
  }
};

void Expect(Reader& r, std::string_view name, std::uint64_t value,
            std::uint64_t expected) {
  if (value != expected) {
    r.Fail("Raon-1 mini: " + std::string(name) + " is " +
           std::to_string(value) + ", this build supports " +
           std::to_string(expected));
  }
}

}  // namespace

std::uint32_t Config::NumMlaLayers() const noexcept {
  std::uint32_t count = 0;
  for (std::uint32_t layer = 0; layer < num_layers; ++layer) {
    count += IsMla(layer) ? 1U : 0U;
  }
  return count;
}

std::uint32_t Config::NumRdaLayers() const noexcept {
  return num_layers - NumMlaLayers();
}

std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                  std::string* error_msg) {
  Reader r{reader, error_msg};
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value()) {
    r.Fail("missing general.architecture");
    return std::nullopt;
  }
  if (*architecture != kArchitecture) {
    r.Fail("artifact is not " + std::string(kArchitecture));
    return std::nullopt;
  }

  Config c;
  const auto key = [](std::string_view name) {
    return std::string(kArchitecture) + "." + std::string(name);
  };
  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.kv_lora_rank = r.U32(key("attention.kv_lora_rank"));
  c.q_lora_rank = r.U32(key("attention.q_lora_rank"));
  c.v_head_dim = r.U32(key("attention.value_length_mla"));
  c.qk_rope_head_dim = r.U32(key("rope.dimension_count"));
  const auto mla_key_length = r.U32(key("attention.key_length_mla"));
  c.qk_nope_head_dim = mla_key_length > c.qk_rope_head_dim
                           ? mla_key_length - c.qk_rope_head_dim
                           : 0;
  c.kda_head_dim = r.U32(key("kda.head_dim"));
  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.expert_count = r.U32(key("expert_count"));
  c.expert_used = r.U32(key("expert_used_count"));
  c.expert_shared = r.U32(key("expert_shared_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.dense_ff = r.U32(key("feed_forward_length"));
  c.leading_dense = r.U32(key("leading_dense_block_count"));
  c.hc_count = r.U32(key("hyper_connection.count"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"));
  c.latent_norm_eps = r.F32(key("attention.latent_norm_rms_epsilon"));
  c.hc_eps = r.F32(key("hyper_connection.epsilon"));
  c.hc_magnitude = r.F32(key("hyper_connection.magnitude"));
  c.situ_beta = r.F32(key("activation.situ_beta"));
  c.situ_linear_beta = r.F32(key("activation.situ_linear_beta"));
  c.kda_gate_lower_bound = r.F32(key("kda.gate_lower_bound"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"));
  c.expert_weights_norm = r.Bool(key("expert_weights_norm"), true);

  const auto kv = r.U64Array(key("attention.head_count_kv"));
  c.kv_layer.reserve(kv.size());
  for (const auto value : kv) {
    c.kv_layer.push_back(value != 0 ? 1 : 0);
  }
  const auto effort = r.U64Array(key("reasoning_effort.layers"));
  for (const auto value : effort) {
    c.effort_layers.push_back(static_cast<std::uint32_t>(value));
  }
  c.effort_alphas = r.F32Array(key("reasoning_effort.alpha"));
  for (const auto name :
       reader.GetMetadataStringArray(key("reasoning_effort.names"))) {
    c.effort_names.emplace_back(name);
  }
  c.effort_default = r.String(key("reasoning_effort.default"), "high");
  if (!r.ok) {
    return std::nullopt;
  }

  // Pinned Raon-1 mini shape. A different revision has to bring its own
  // kernels and layouts rather than run on these.
  Expect(r, "block_count", c.num_layers, 25);
  Expect(r, "embedding_length", c.hidden_size, 2560);
  Expect(r, "attention.head_count", c.num_heads, 40);
  Expect(r, "attention.kv_lora_rank", c.kv_lora_rank, 512);
  Expect(r, "attention.q_lora_rank", c.q_lora_rank, 576);
  Expect(r, "attention.key_length_mla", mla_key_length, 192);
  Expect(r, "attention.value_length_mla", c.v_head_dim, 128);
  Expect(r, "rope.dimension_count", c.qk_rope_head_dim, 64);
  Expect(r, "attention.qk_nope_head_dim", c.qk_nope_head_dim, 128);
  Expect(r, "kda.head_dim", c.kda_head_dim, 128);
  Expect(r, "ssm.conv_kernel", c.ssm_conv_kernel, 4);
  Expect(r, "expert_count", c.expert_count, 128);
  Expect(r, "expert_used_count", c.expert_used, 6);
  Expect(r, "expert_shared_count", c.expert_shared, 1);
  Expect(r, "expert_feed_forward_length", c.expert_ff, 1088);
  Expect(r, "feed_forward_length", c.dense_ff, 12032);
  Expect(r, "leading_dense_block_count", c.leading_dense, 1);
  Expect(r, "hyper_connection.count", c.hc_count, 4);
  Expect(r, "vocab_size", c.vocab_size, 259840);
  Expect(r, "kv_layer entries", c.kv_layer.size(), c.num_layers);
  Expect(r, "trunk MLA layers", c.NumMlaLayers(), 7);
  Expect(r, "trunk RDA layers", c.NumRdaLayers(), 18);
  Expect(r, "reasoning_effort.layers", c.effort_layers.size(), 5);
  Expect(r, "reasoning_effort.alpha", c.effort_alphas.size(), 5);
  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::raon1
