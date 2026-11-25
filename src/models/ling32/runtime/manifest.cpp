#include "src/models/ling32/runtime/manifest.hpp"

#include <initializer_list>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::ling32 {
namespace {

bool IsQuantized(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kF32:
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
    case core::GgmlType::kI32:
      return false;
    default:
      return true;
  }
}

struct Builder {
  const Config& config;
  const core::GgufReader& reader;
  std::unordered_map<std::string_view, const core::GgufTensorInfo*> by_name;
  Manifest manifest;
  std::string* error{nullptr};
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  void Add(Slot slot, std::uint32_t layer, std::string_view name,
           std::initializer_list<std::uint64_t> dims) {
    if (!ok) {
      return;
    }
    const auto it = by_name.find(name);
    if (it == by_name.end()) {
      Fail("missing tensor " + std::string(name));
      return;
    }
    const auto* tensor = it->second;
    if (tensor->dimensions.size() != dims.size()) {
      Fail(std::string(name) + " has " +
           std::to_string(tensor->dimensions.size()) +
           " dimensions, expected " + std::to_string(dims.size()));
      return;
    }
    std::size_t index = 0;
    for (const auto expected : dims) {
      if (tensor->dimensions[index] != expected) {
        Fail(std::string(name) + " dimension " + std::to_string(index) +
             " is " + std::to_string(tensor->dimensions[index]) +
             ", expected " + std::to_string(expected));
        return;
      }
      ++index;
    }
    manifest.entries.push_back(ManifestEntry{slot, layer, tensor});
    manifest.float_bytes += IsQuantized(tensor->type) ? 0 : tensor->size_bytes;
    manifest.quantized_bytes +=
        IsQuantized(tensor->type) ? tensor->size_bytes : 0;
    if (slot == Slot::kExpertGate || slot == Slot::kExpertUp ||
        slot == Slot::kExpertDown) {
      manifest.expert_bytes += tensor->size_bytes;
    }
    if (config.HasMtp() && layer == config.MtpLayer()) {
      manifest.mtp_bytes += tensor->size_bytes;
    }
  }
};

}  // namespace

bool IsFloatSlot(Slot slot) noexcept {
  switch (slot) {
    case Slot::kOutputNorm:
    case Slot::kAttnNorm:
    case Slot::kFfnNorm:
    case Slot::kKdaA:
    case Slot::kKdaBeta:
    case Slot::kKdaDt:
    case Slot::kKdaConvQ:
    case Slot::kKdaConvK:
    case Slot::kKdaConvV:
    case Slot::kKdaNorm:
    case Slot::kMlaKvANorm:
    case Slot::kRouter:
    case Slot::kRouterBias:
    case Slot::kNextnEnorm:
    case Slot::kNextnHnorm:
    case Slot::kNextnSharedHeadNorm:
      return true;
    default:
      return false;
  }
}

std::optional<Manifest> BuildManifest(const Config& config,
                                      const core::GgufReader& reader,
                                      std::string* error_msg) {
  Builder b{config, reader, {}, {}, error_msg, true};
  b.by_name.reserve(reader.GetTensors().size());
  for (const auto& tensor : reader.GetTensors()) {
    b.by_name.emplace(tensor.name, &tensor);
  }

  const auto h = config.hidden_size;
  b.Add(Slot::kTokenEmbd, 0, "token_embd.weight", {h, config.vocab_size});
  b.Add(Slot::kOutput, 0, "output.weight", {h, config.vocab_size});
  b.Add(Slot::kOutputNorm, 0, "output_norm.weight", {h});

  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto p = [l](std::string_view suffix) {
      return "blk." + std::to_string(l) + "." + std::string(suffix);
    };
    b.Add(Slot::kAttnNorm, l, p("attn_norm.weight"), {h});
    b.Add(Slot::kFfnNorm, l, p("ffn_norm.weight"), {h});

    if (config.IsKda(l)) {
      const auto dim = config.KdaDim();
      b.Add(Slot::kKdaQ, l, p("attn_q.weight"), {h, dim});
      b.Add(Slot::kKdaK, l, p("attn_k.weight"), {h, dim});
      b.Add(Slot::kKdaV, l, p("attn_v.weight"), {h, dim});
      b.Add(Slot::kKdaOutput, l, p("attn_output.weight"), {dim, h});
      b.Add(Slot::kKdaA, l, p("ssm_a"), {1, config.num_heads});
      b.Add(Slot::kKdaBeta, l, p("ssm_beta.weight"), {h, config.num_heads});
      b.Add(Slot::kKdaDt, l, p("ssm_dt.bias"), {dim});
      b.Add(Slot::kKdaConvQ, l, p("ssm_conv1d_q.weight"),
            {config.ssm_conv_kernel, 1, dim});
      b.Add(Slot::kKdaConvK, l, p("ssm_conv1d_k.weight"),
            {config.ssm_conv_kernel, 1, dim});
      b.Add(Slot::kKdaConvV, l, p("ssm_conv1d_v.weight"),
            {config.ssm_conv_kernel, 1, dim});
      b.Add(Slot::kKdaFA, l, p("ssm_f_a.weight"), {h, dim});
      b.Add(Slot::kKdaGA, l, p("ssm_g_a.weight"), {h, dim});
      b.Add(Slot::kKdaNorm, l, p("ssm_norm.weight"), {config.kda_head_dim});
    } else {
      const auto qdim = config.num_heads * config.MlaQDim();
      const auto vdim = config.num_heads * config.MlaVDim();
      b.Add(Slot::kMlaQ, l, p("attn_q.weight"), {h, qdim});
      b.Add(Slot::kMlaKvA, l, p("attn_kv_a_mqa.weight"),
            {h, config.kv_lora_rank + config.rope_dimension});
      b.Add(Slot::kMlaKvANorm, l, p("attn_kv_a_norm.weight"),
            {config.kv_lora_rank});
      b.Add(Slot::kMlaKB, l, p("attn_k_b.weight"),
            {config.QkNope(), config.kv_lora_rank, config.num_heads});
      b.Add(Slot::kMlaVB, l, p("attn_v_b.weight"),
            {config.kv_lora_rank, config.MlaVDim(), config.num_heads});
      b.Add(Slot::kMlaGate, l, p("attn_gate.weight"), {h, config.num_heads});
      b.Add(Slot::kMlaOutput, l, p("attn_output.weight"), {vdim, h});
    }

    if (config.IsDense(l)) {
      b.Add(Slot::kDenseGate, l, p("ffn_gate.weight"), {h, config.dense_ff});
      b.Add(Slot::kDenseUp, l, p("ffn_up.weight"), {h, config.dense_ff});
      b.Add(Slot::kDenseDown, l, p("ffn_down.weight"), {config.dense_ff, h});
    } else {
      b.Add(Slot::kRouter, l, p("ffn_gate_inp.weight"),
            {h, config.expert_count});
      b.Add(Slot::kRouterBias, l, p("exp_probs_b.bias"), {config.expert_count});
      b.Add(Slot::kExpertGate, l, p("ffn_gate_exps.weight"),
            {h, config.expert_ff, config.expert_count});
      b.Add(Slot::kExpertUp, l, p("ffn_up_exps.weight"),
            {h, config.expert_ff, config.expert_count});
      b.Add(Slot::kExpertDown, l, p("ffn_down_exps.weight"),
            {config.expert_ff, h, config.expert_count});
      b.Add(Slot::kSharedGate, l, p("ffn_gate_shexp.weight"),
            {h, config.shared_ff});
      b.Add(Slot::kSharedUp, l, p("ffn_up_shexp.weight"),
            {h, config.shared_ff});
      b.Add(Slot::kSharedDown, l, p("ffn_down_shexp.weight"),
            {config.shared_ff, h});
    }
  }

  if (config.HasMtp()) {
    const auto mtp = config.MtpLayer();
    const auto p = [mtp](std::string_view suffix) {
      return "blk." + std::to_string(mtp) + "." + std::string(suffix);
    };
    b.Add(Slot::kNextnEhProj, mtp, p("nextn.eh_proj.weight"), {2 * h, h});
    b.Add(Slot::kNextnEnorm, mtp, p("nextn.enorm.weight"), {h});
    b.Add(Slot::kNextnHnorm, mtp, p("nextn.hnorm.weight"), {h});
    b.Add(Slot::kNextnSharedHeadNorm, mtp, p("layer_output_norm.weight"), {h});
  }

  if (!b.ok) {
    return std::nullopt;
  }
  if (b.manifest.entries.size() != reader.GetTensors().size()) {
    b.Fail("tensor count is " + std::to_string(reader.GetTensors().size()) +
           ", manifest covers " + std::to_string(b.manifest.entries.size()));
    return std::nullopt;
  }
  return std::move(b.manifest);
}

}  // namespace gufo::models::ling32
