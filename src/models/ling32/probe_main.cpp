#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling32/reference.hpp"
#include "src/models/ling32/runtime/config.hpp"
#include "src/models/ling32/runtime/manifest.hpp"

namespace {

bool RunReference(const std::string& path, std::string* error) {
  auto model = gufo::models::ling32::ReferenceModel::Open(path, error);
  if (!model) {
    return false;
  }
  auto session = model->CreateSession();
  constexpr int steps = 4;
  for (int t = 0; t < steps; ++t) {
    const std::vector<float>& logits = model->Evaluate(*session, t + 1);
    if (logits.empty()) {
      *error = "empty logits";
      return false;
    }
    const auto top = std::max_element(logits.begin(), logits.end());
    if (!std::isfinite(*top)) {
      *error = "non-finite logits at step " + std::to_string(t);
      return false;
    }
    const auto id = static_cast<std::int32_t>(top - logits.begin());
    std::printf("ref step=%d top=%d logit=%.6f\n", t, id, *top);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <model-00001-of-N.gguf>\n", argv[0]);
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto config = gufo::models::ling32::ParseConfig(*reader, &error);
  if (!config) {
    std::fprintf(stderr, "config rejected: %s\n", error.c_str());
    return 1;
  }
  const auto& c = *config;
  std::printf(
      "arch=bailingmoe3 layers=%u trunk=%u dense=%u mla=%u kda=%u "
      "hidden=%u heads=%u kda_dim=%u conv=%u\n"
      "kv_lora=%u qk=%u nope=%u rope=%u v=%u vocab=%u ctx=%u nextn=%u\n"
      "experts=%u used=%u groups=%u/%u shared=%u ff=%u/%u dense_ff=%u\n"
      "eps=%g base=%g gate_lb=%g scale=%g norm=%d tensors=%zu\n",
      c.num_layers, c.TrunkLayers(), c.leading_dense, c.NumMlaLayers(),
      c.NumKdaLayers(), c.hidden_size, c.num_heads, c.KdaDim(),
      c.ssm_conv_kernel, c.kv_lora_rank, c.key_length_mla, c.QkNope(),
      c.rope_dimension, c.MlaVDim(), c.vocab_size, c.context_length,
      c.nextn_layers, c.expert_count, c.expert_used, c.expert_groups,
      c.expert_groups_used, c.expert_shared, c.expert_ff, c.shared_ff,
      c.dense_ff, c.rms_eps, c.freq_base, c.gate_lower_bound,
      c.expert_weights_scale, c.expert_weights_norm ? 1 : 0,
      reader->GetTensors().size());
  auto manifest = gufo::models::ling32::BuildManifest(c, *reader, &error);
  if (!manifest) {
    std::fprintf(stderr, "manifest rejected: %s\n", error.c_str());
    return 1;
  }
  const auto gib = [](std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1 << 30);
  };
  std::printf(
      "manifest entries=%zu total=%.2f GiB float=%.1f MiB quant=%.2f GiB "
      "experts=%.2f GiB mtp=%.1f MiB\n",
      manifest->entries.size(), gib(manifest->total_bytes()),
      static_cast<double>(manifest->float_bytes) / (1 << 20),
      gib(manifest->quantized_bytes), gib(manifest->expert_bytes),
      static_cast<double>(manifest->mtp_bytes) / (1 << 20));
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    std::printf("blk.%02u %s%s%s clamp=%.0f/%.0f\n", l,
                c.IsMla(l) ? "mla" : "kda", c.IsDense(l) ? "+dense" : "+moe",
                c.HasMtp() && l == c.MtpLayer() ? " mtp" : "",
                c.swiglu_clamp_exp[l], c.swiglu_clamp_shexp[l]);
  }
  return 0;
}
