#include "src/models/glm53_flash/runtime/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm53_flash {
namespace {

constexpr std::string_view kArchitecture = "glm5-next";
constexpr std::string_view kLegacyArchitecture = "glm5next";

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
};

void Expect(Reader& r, std::string_view name, std::uint64_t value,
            std::uint64_t expected) {
  if (value != expected) {
    r.Fail("GLM-5.3-Flash: " + std::string(name) + " is " +
           std::to_string(value) + ", this build supports " +
           std::to_string(expected));
  }
}

}  // namespace

std::uint32_t Config::NumMlaLayers() const noexcept {
  std::uint32_t count = 0;
  for (std::uint32_t layer = 0; layer < TrunkLayers(); ++layer) {
    count += IsMla(layer) ? 1U : 0U;
  }
  return count;
}

std::uint32_t Config::NumKdaLayers() const noexcept {
  return TrunkLayers() - NumMlaLayers();
}

std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                  std::string* error_msg) {
  Reader r{reader, error_msg};
  const auto architecture = reader.GetMetadataString("general.architecture");
  std::string_view prefix{kArchitecture};
  if (!architecture.has_value()) {
    r.Fail("missing general.architecture");
    return std::nullopt;
  }
  if (*architecture == kLegacyArchitecture) {
    prefix = kLegacyArchitecture;
  } else if (*architecture != kArchitecture) {
    r.Fail("artifact is not " + std::string(kArchitecture));
    return std::nullopt;
  }

  Config c;
  const auto key = [prefix](std::string_view name) {
    return std::string(prefix) + "." + std::string(name);
  };
  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.key_length = r.U32(key("attention.key_length"));
  c.value_length = r.U32(key("attention.value_length"));
  c.key_length_mla = r.U32(key("attention.key_length_mla"));
  c.value_length_mla = r.U32(key("attention.value_length_mla"));
  c.kv_lora_rank = r.U32(key("attention.kv_lora_rank"));
  c.q_lora_rank = r.U32(key("attention.q_lora_rank"));
  c.indexer_heads = r.U32(key("attention.indexer.head_count"));
  c.indexer_key_dim = r.U32(key("attention.indexer.key_length"));
  c.indexer_top_k = r.U32(key("attention.indexer.top_k"));
  c.indexer_kpool = r.U32(key("attention.indexer.kpool"));
  const auto share =
      reader.GetMetadataBool(key("attention.indexer.index_share_mtp"));
  c.index_share_mtp = share.value_or(false);
  c.expert_count = r.U32(key("expert_count"));
  c.expert_used = r.U32(key("expert_used_count"));
  c.expert_shared = r.U32(key("expert_shared_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_ff = r.U32(key("expert_shared_feed_forward_length"));
  if (!r.ok) {
    r.ok = true;
    c.shared_ff = c.expert_ff;
  }
  c.dense_ff = r.U32(key("feed_forward_length"));
  c.leading_dense = r.U32(key("leading_dense_block_count"));
  c.hc_count = r.U32(key("hyper_connection.count"));
  c.sinkhorn_iters = r.U32(key("hyper_connection.sinkhorn_iterations"));
  c.kda_head_dim = r.U32(key("kda.head_dim"));
  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"));
  if (!r.ok) {
    r.ok = true;
    c.nextn_layers = 0;
  }
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"));
  c.layer_norm_eps = r.F32(key("attention.layer_norm_epsilon"));
  c.hc_eps = r.F32(key("hyper_connection.epsilon"));
  c.kda_gate_lower_bound = r.F32(key("kda.gate_lower_bound"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"));
  c.expert_weights_norm = r.Bool(key("expert_weights_norm"), true);

  const auto kv = r.U64Array(key("attention.head_count_kv"));
  c.kv_layer.reserve(kv.size());
  for (const auto value : kv) {
    c.kv_layer.push_back(value != 0 ? 1 : 0);
  }
  c.swiglu_clamp_exp = r.F32Array(key("swiglu_clamp_exp"));
  c.swiglu_clamp_shexp = r.F32Array(key("swiglu_clamp_shexp"));
  if (!r.ok) {
    return std::nullopt;
  }

  // Pinned GLM-5.3-Flash shape. A different revision has to bring its own
  // kernels and layouts rather than run on these.
  if (c.num_layers != 45 && c.num_layers != 46) {
    r.Fail("GLM-5.3-Flash: block_count is " + std::to_string(c.num_layers) +
           ", this build supports 45 (pruned MTP) or 46");
  }
  Expect(r, "embedding_length", c.hidden_size, 4096);
  Expect(r, "attention.head_count", c.num_heads, 64);
  Expect(r, "attention.key_length", c.key_length, 512);
  Expect(r, "attention.value_length", c.value_length, 512);
  Expect(r, "attention.key_length_mla", c.key_length_mla, 256);
  Expect(r, "attention.value_length_mla", c.value_length_mla, 256);
  Expect(r, "attention.kv_lora_rank", c.kv_lora_rank, 512);
  Expect(r, "attention.q_lora_rank", c.q_lora_rank, 1536);
  Expect(r, "attention.indexer.head_count", c.indexer_heads, 32);
  Expect(r, "attention.indexer.key_length", c.indexer_key_dim, 128);
  Expect(r, "attention.indexer.top_k", c.indexer_top_k, 2048);
  Expect(r, "attention.indexer.kpool", c.indexer_kpool, 4);
  if (c.expert_count != 288 && c.expert_count != 152)
    r.Fail("GLM-5.3-Flash: expert_count is " + std::to_string(c.expert_count));
  Expect(r, "expert_used_count", c.expert_used, 8);
  Expect(r, "expert_shared_count", c.expert_shared, 1);
  Expect(r, "expert_feed_forward_length", c.expert_ff, 2048);
  Expect(r, "expert_shared_feed_forward_length", c.shared_ff, 2048);
  Expect(r, "feed_forward_length", c.dense_ff, 12288);
  Expect(r, "leading_dense_block_count", c.leading_dense, 3);
  Expect(r, "hyper_connection.count", c.hc_count, 4);
  Expect(r, "hyper_connection.sinkhorn_iterations", c.sinkhorn_iters, 20);
  Expect(r, "kda.head_dim", c.kda_head_dim, 128);
  Expect(r, "ssm.conv_kernel", c.ssm_conv_kernel, 4);
  Expect(r, "vocab_size", c.vocab_size, 154880);
  Expect(r, "block_count minus nextn", c.num_layers - c.nextn_layers, 45);
  if (c.nextn_layers > 1) {
    r.Fail("GLM-5.3-Flash: more than one nextn layer");
  }
  Expect(r, "kv_layer entries", c.kv_layer.size(), c.num_layers);
  Expect(r, "swiglu_clamp_exp entries", c.swiglu_clamp_exp.size(),
         c.num_layers);
  Expect(r, "swiglu_clamp_shexp entries", c.swiglu_clamp_shexp.size(),
         c.num_layers);
  Expect(r, "trunk MLA layers", c.NumMlaLayers(), 11);
  Expect(r, "trunk KDA layers", c.NumKdaLayers(), 34);
  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::glm53_flash
