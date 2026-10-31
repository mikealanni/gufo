#include "src/models/qwen/dflash_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::speculative {
namespace {

models::QwenTensorRef TensorRef(const core::GgufReader& reader,
                                std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address < region_address) {
      continue;
    }
    const auto offset = tensor_address - region_address;
    if (offset < region.size) {
      available_bytes = region.size - offset;
      break;
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

float Bf16ToFloat(std::uint16_t h) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

std::uint16_t FloatToHalf(float f) noexcept {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

// Re-encodes a BF16 dense weight matrix to Q8_0 (block_q8_0: one F16 scale
// + 32 signed int8 values per 32-element block) so the drafter's forward
// pass can use Laguna's Q8_0 dense path (qfn_mmq_q8_0_dense_vec_preq for
// its small batches, same fix as the target model's decode path) instead
// of reading the full BF16 matrix every call - roughly half the bytes for
// the same row. Synchronous, single-threaded: the drafter is a handful of
// layers, a few hundred MB total, not Laguna's 68 GB main model (which has
// its own, parallel, async re-encoder in device_model.cpp - not reused
// here since it is a private method tightly coupled to that uploader).
// Buffers are kept alive via `storage` (shared_ptr, not a plain owned
// vector - QwenDFlashWeights is copied by value at least once while
// loading, which would reallocate and dangle any pointer into a plain
// vector member).
models::QwenTensorRef ReencodeBf16ToQ8_0(
    const models::QwenTensorRef& bf16_ref, std::uint32_t rows,
    std::uint32_t cols,
    std::vector<std::shared_ptr<std::vector<char>>>* storage) {
  constexpr std::uint32_t kBlock = 32;
  constexpr std::size_t kBytes = 34;
  if (bf16_ref.empty() || bf16_ref.type != core::GgmlType::kBF16 ||
      cols % kBlock != 0 || rows == 0) {
    return {};
  }
  const std::size_t row_bytes = (cols / kBlock) * kBytes;
  auto buf = std::make_shared<std::vector<char>>(
      row_bytes * static_cast<std::size_t>(rows));
  const auto* src = static_cast<const std::uint16_t*>(bf16_ref.data);
  std::vector<float> values(cols);
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::uint16_t* row_src = src + static_cast<std::size_t>(r) * cols;
    for (std::uint32_t i = 0; i < cols; ++i)
      values[i] = Bf16ToFloat(row_src[i]);
    char* dst = buf->data() + static_cast<std::size_t>(r) * row_bytes;
    for (std::uint32_t b = 0; b < cols / kBlock; ++b) {
      const float* v = values.data() + b * kBlock;
      float amax = 0.0F;
      for (std::uint32_t i = 0; i < kBlock; ++i)
        amax = std::max(amax, std::fabs(v[i]));
      const float scale = amax / 127.0F;
      const float inv = scale > 0.0F ? 1.0F / scale : 0.0F;
      const std::uint16_t d = FloatToHalf(scale);
      char* block = dst + static_cast<std::size_t>(b) * kBytes;
      std::memcpy(block, &d, sizeof(d));
      for (std::uint32_t i = 0; i < kBlock; ++i) {
        const int q = static_cast<int>(std::lround(v[i] * inv));
        block[2 + i] = static_cast<char>(std::clamp(q, -127, 127));
      }
    }
  }
  storage->push_back(buf);
  models::QwenTensorRef out;
  out.data = buf->data();
  out.type = core::GgmlType::kQ8_0;
  out.num_elements = static_cast<std::size_t>(rows) * cols;
  return out;
}

[[nodiscard]] bool IsSupportedMatrixType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ6_K;
}

[[nodiscard]] bool IsSupportedVectorType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
}

[[nodiscard]] bool BindRequiredTensor(const core::GgufReader& reader,
                                      std::string_view name,
                                      std::size_t expected_elements,
                                      bool matrix,
                                      models::QwenTensorRef& destination,
                                      std::string* error_msg) {
  destination = TensorRef(reader, name);
  if (destination.empty()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required tensor '" +
                            std::string(name) + "'");
    return false;
  }
  if (destination.num_elements != expected_elements) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) + "' has " +
                            std::to_string(destination.num_elements) +
                            " elements; expected " +
                            std::to_string(expected_elements));
    return false;
  }
  const bool supported = matrix ? IsSupportedMatrixType(destination.type)
                                : IsSupportedVectorType(destination.type);
  if (!supported) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' has unsupported type " +
                            std::string(core::ToString(destination.type)));
    return false;
  }
  if (!destination.FitsAvailableStorage()) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' exceeds its mapped GGUF storage");
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::uint32_t> RequiredUint32(
    const core::GgufReader& reader, std::string_view key,
    std::string* error_msg) {
  auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required metadata '" +
                            std::string(key) + "'");
  }
  return value;
}

[[nodiscard]] std::optional<std::vector<std::uint64_t>>
GetNonnegativeIntegerArray(const core::GgufReader& reader,
                           std::string_view key) {
  const auto* metadata = reader.FindMetadata(key);
  if (metadata == nullptr) {
    return std::nullopt;
  }
  if (const auto* values =
          std::get_if<std::vector<std::uint64_t>>(&metadata->value)) {
    return *values;
  }
  if (const auto* values =
          std::get_if<std::vector<std::int64_t>>(&metadata->value)) {
    std::vector<std::uint64_t> converted;
    converted.reserve(values->size());
    for (const std::int64_t value : *values) {
      if (value < 0) {
        return std::nullopt;
      }
      converted.push_back(static_cast<std::uint64_t>(value));
    }
    return converted;
  }
  return std::nullopt;
}

}  // namespace

std::optional<QwenDFlashWeights> QwenDFlashWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "dflash") {
    SetError(error_msg, "DFlash-2 GGUF requires general.architecture='dflash'");
    return std::nullopt;
  }

  core::ModelConfig config;
  config.architecture = "dflash";
  const std::string prefix = "dflash.";

  const auto hidden_size =
      RequiredUint32(reader, prefix + "embedding_length", error_msg);
  const auto intermediate_size =
      RequiredUint32(reader, prefix + "feed_forward_length", error_msg);
  const auto num_attention_heads =
      RequiredUint32(reader, prefix + "attention.head_count", error_msg);
  const auto num_key_value_heads =
      RequiredUint32(reader, prefix + "attention.head_count_kv", error_msg);
  const auto head_dim =
      RequiredUint32(reader, prefix + "attention.key_length", error_msg);
  auto rotary_dim = reader.GetMetadataUint32(prefix + "rope.dimension_count");
  if (!rotary_dim.has_value()) {
    const auto sections =
        GetNonnegativeIntegerArray(reader, prefix + "rope.dimension_sections");
    if (sections.has_value()) {
      const auto& values = *sections;
      if (values.size() == 4 && values[0] > 0 && values[1] == 0 &&
          values[2] == 0 && values[3] == 0 &&
          values[0] <= std::numeric_limits<std::uint32_t>::max() / 2U) {
        rotary_dim = static_cast<std::uint32_t>(values[0] * 2U);
      }
    }
  }
  if (!rotary_dim.has_value() && head_dim.has_value()) {
    // Without an explicit count the whole head is rotated, as in llama.cpp.
    rotary_dim = head_dim;
  }
  const auto context_length =
      RequiredUint32(reader, prefix + "context_length", error_msg);
  const auto num_layers =
      RequiredUint32(reader, prefix + "block_count", error_msg);
  const auto rope_theta = reader.GetMetadataFloat32(prefix + "rope.freq_base");
  if (!hidden_size || !intermediate_size || !num_attention_heads ||
      !num_key_value_heads || !head_dim || !rotary_dim || !context_length ||
      !num_layers || !rope_theta) {
    if (!rotary_dim) {
      SetError(error_msg,
               "DFlash-2 GGUF requires a valid rotary dimension or "
               "degenerate four-section M-RoPE layout");
    }
    if (!rope_theta) {
      SetError(
          error_msg,
          "DFlash-2 GGUF is missing required metadata 'dflash.rope.freq_base'");
    }
    return std::nullopt;
  }

  config.hidden_size = *hidden_size;
  config.intermediate_size = *intermediate_size;
  config.num_attention_heads = *num_attention_heads;
  config.num_key_value_heads = *num_key_value_heads;
  config.head_dim = *head_dim;
  config.rotary_dim = *rotary_dim;
  config.rope_theta = *rope_theta;
  config.context_length = *context_length;
  config.num_layers = *num_layers;
  config.full_attention_interval = 1;
  config.mtp_num_layers = 0;

  if (config.hidden_size == 0 || config.intermediate_size == 0 ||
      config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
      config.head_dim == 0 || config.num_layers == 0 ||
      config.num_attention_heads % config.num_key_value_heads != 0 ||
      config.rotary_dim == 0 || config.rotary_dim > config.head_dim ||
      (config.rotary_dim % 2U) != 0) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid transformer topology");
    return std::nullopt;
  }

  QwenDFlashWeights weights;
  weights.config = config;
  auto& df_cfg = weights.dflash_config;

  const auto block_size =
      RequiredUint32(reader, prefix + "block_size", error_msg);
  const auto conv_kernel_size =
      reader.GetMetadataUint32(prefix + "conv_kernel_size");
  const auto conv_group_size =
      reader.GetMetadataUint32(prefix + "conv_group_size");
  const auto selector_rank = reader.GetMetadataUint32(prefix + "selector_rank");
  const auto selector_top_k =
      reader.GetMetadataUint32(prefix + "selector_top_k");
  const auto sliding_window =
      RequiredUint32(reader, prefix + "attention.sliding_window", error_msg);
  auto mask_token_id = reader.GetMetadataUint32("tokenizer.ggml.mask_token_id");
  if (!mask_token_id.has_value()) {
    mask_token_id = reader.GetMetadataUint32(prefix + "mask_token_id");
  }
  const auto is_causal = reader.GetMetadataBool(prefix + "attention.causal");
  if (!block_size || !sliding_window || !mask_token_id ||
      !is_causal.has_value()) {
    if (!is_causal.has_value()) {
      SetError(error_msg,
               "DFlash GGUF is missing required metadata "
               "'dflash.attention.causal'");
    }
    return std::nullopt;
  }

  const bool is_v1 =
      !conv_kernel_size.has_value() && !selector_rank.has_value();
  df_cfg.is_v1 = is_v1;
  df_cfg.block_size = *block_size;
  df_cfg.conv_kernel_size = conv_kernel_size.value_or(0);
  df_cfg.conv_group_size = conv_group_size.value_or(0);
  df_cfg.selector_rank = selector_rank.value_or(0);
  df_cfg.selector_top_k = selector_top_k.value_or(0);
  df_cfg.sliding_window = *sliding_window;
  df_cfg.mask_token_id = *mask_token_id;
  df_cfg.num_layers = config.num_layers;

  if (is_v1) {
    if (df_cfg.block_size < 2 || df_cfg.sliding_window < df_cfg.block_size ||
        *is_causal) {
      SetError(error_msg, "DFlash-1 GGUF has an invalid diffusion topology");
      return std::nullopt;
    }
  } else {
    if (df_cfg.block_size < 2 || df_cfg.conv_kernel_size == 0 ||
        df_cfg.conv_group_size == 0 ||
        config.hidden_size % df_cfg.conv_group_size != 0 ||
        df_cfg.selector_rank == 0 || df_cfg.selector_top_k == 0 ||
        df_cfg.selector_top_k > 16U ||
        df_cfg.sliding_window < df_cfg.block_size || *is_causal) {
      SetError(error_msg, "DFlash-2 GGUF has an invalid diffusion topology");
      return std::nullopt;
    }
  }

  // GGUF target_layers are layer-input indices (the convention used by the
  // converter). Gufo taps layer outputs, so normalize N to zero-based N - 1.
  const auto encoded_target_layers =
      GetNonnegativeIntegerArray(reader, prefix + "target_layers");
  if (!encoded_target_layers.has_value()) {
    SetError(error_msg,
             "DFlash-2 GGUF is missing required integer array "
             "'dflash.target_layers'");
    return std::nullopt;
  }
  if (encoded_target_layers->empty()) {
    SetError(error_msg, "DFlash-2 target layer list must not be empty");
    return std::nullopt;
  }
  df_cfg.target_layer_ids.reserve(encoded_target_layers->size());
  for (const std::uint64_t encoded_layer : *encoded_target_layers) {
    if (encoded_layer == 0 ||
        !std::in_range<std::uint32_t>(encoded_layer - 1U)) {
      SetError(error_msg,
               "DFlash-2 target layer indices must use positive GGUF "
               "layer-input numbering");
      return std::nullopt;
    }
    const auto normalized = static_cast<std::uint32_t>(encoded_layer - 1U);
    if (!df_cfg.target_layer_ids.empty() &&
        normalized <= df_cfg.target_layer_ids.back()) {
      SetError(error_msg,
               "DFlash-2 target layer indices must be strictly increasing");
      return std::nullopt;
    }
    df_cfg.target_layer_ids.push_back(normalized);
  }

  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t groups = is_v1 ? 0 : hidden / df_cfg.conv_group_size;
  const std::size_t dynamic_coefficients =
      std::size_t{2} * df_cfg.conv_kernel_size * groups;

  // DFlash-2 uses the target model's embedding and output head. Private copies
  // are accepted but not required.
  if (reader.FindTensor("d2t") != nullptr) {
    SetError(error_msg,
             "DFlash-2 reduced-vocabulary d2t exports are not supported yet");
    return std::nullopt;
  }

  if (!is_v1) {
    const auto* predecessor_tensor =
        reader.FindTensor("selector_predecessor.weight");
    if (predecessor_tensor == nullptr || predecessor_tensor->data == nullptr ||
        predecessor_tensor->ElementCount() % df_cfg.selector_rank != 0) {
      SetError(error_msg,
               "DFlash-2 selector_predecessor.weight has invalid dimensions");
      return std::nullopt;
    }
    weights.selector_predecessor =
        TensorRef(reader, "selector_predecessor.weight");
    if (!IsSupportedMatrixType(weights.selector_predecessor.type) ||
        !weights.selector_predecessor.FitsAvailableStorage()) {
      SetError(error_msg,
               "DFlash-2 selector_predecessor.weight has unsupported storage");
      return std::nullopt;
    }
    const std::size_t inferred_vocab =
        weights.selector_predecessor.num_elements / df_cfg.selector_rank;
    if (inferred_vocab == 0 || !std::in_range<std::uint32_t>(inferred_vocab)) {
      SetError(error_msg, "DFlash-2 selector vocabulary size is invalid");
      return std::nullopt;
    }
    config.vocab_size = static_cast<std::uint32_t>(inferred_vocab);
    weights.config.vocab_size = config.vocab_size;

    if (!BindRequiredTensor(reader, "selector_successor.weight",
                            inferred_vocab * df_cfg.selector_rank, true,
                            weights.selector_successor, error_msg) ||
        !BindRequiredTensor(reader, "selector_hidden.weight",
                            hidden * df_cfg.selector_rank, true,
                            weights.selector_hidden, error_msg)) {
      return std::nullopt;
    }
  }

  const std::size_t encoder_input = df_cfg.target_layer_ids.size() * hidden;
  if (!BindRequiredTensor(reader, "fc.weight", hidden * encoder_input, true,
                          weights.fc_projection, error_msg) ||
      !BindRequiredTensor(reader, "enc.output_norm.weight", hidden, false,
                          weights.fc_norm, error_msg) ||
      !BindRequiredTensor(reader, "output_norm.weight", hidden, false,
                          weights.output_norm, error_msg)) {
    return std::nullopt;
  }
  // Same Q8_0 re-encode as the per-layer dense weights below -
  // RunInjectGemm (executor.cpp) already has a Q8_0 dispatch branch for
  // this projection, just never exercised while it was BF16. This one
  // tensor alone is 113 MB (hidden * encoder_input BF16 elements).
  {
    auto q8 = ReencodeBf16ToQ8_0(weights.fc_projection, hidden, encoder_input,
                                 &weights.owned_reencoded_storage);
    if (!q8.empty())
      weights.fc_projection = q8;
  }
  // Per-tap pre-norm, applied to each captured target hidden state before
  // concatenation + fc_projection. Optional - not every DFlash export has
  // it; when absent, InjectTargetContextRows skips the normalization.
  const auto aux_norm = TensorRef(reader, "enc.aux_norm.weight");
  if (!aux_norm.empty()) {
    if (aux_norm.num_elements != encoder_input) {
      SetError(error_msg, "DFlash-2 tensor 'enc.aux_norm.weight' has " +
                              std::to_string(aux_norm.num_elements) +
                              " elements; expected " +
                              std::to_string(encoder_input));
      return std::nullopt;
    }
    if (aux_norm.type != core::GgmlType::kF32) {
      // LaunchDFlashTapRMSNorm reads this directly as float32; reject any
      // other encoding rather than silently misinterpreting its bytes.
      SetError(error_msg,
               "DFlash-2 tensor 'enc.aux_norm.weight' must be "
               "F32, got " +
                   std::string(core::ToString(aux_norm.type)));
      return std::nullopt;
    }
    weights.aux_norm = aux_norm;
  }
  const auto fc_scale = TensorRef(reader, "fc.scale");
  if (!fc_scale.empty() &&
      (fc_scale.num_elements != 1 || !IsSupportedVectorType(fc_scale.type) ||
       !fc_scale.FitsAvailableStorage() || fc_scale.Get(0) != 1.0F)) {
    SetError(error_msg,
             "DFlash-2 fc.scale must be absent or a unit F32/BF16 scalar");
    return std::nullopt;
  }

  weights.layers.resize(df_cfg.num_layers);
  for (std::uint32_t i = 0; i < df_cfg.num_layers; ++i) {
    const std::string layer_prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    auto& transformer = layer.transformer;
    transformer.is_full_attention = true;

    if (!BindRequiredTensor(reader, layer_prefix + "attn_norm.weight", hidden,
                            false, transformer.attn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q.weight",
                            attention * hidden, true, transformer.attn_q,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k.weight", kv * hidden,
                            true, transformer.attn_k, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_v.weight", kv * hidden,
                            true, transformer.attn_v, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_output.weight",
                            hidden * attention, true, transformer.attn_output,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q_norm.weight",
                            config.head_dim, false, transformer.attn_q_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k_norm.weight",
                            config.head_dim, false, transformer.attn_k_norm,
                            error_msg) ||
        // Per-head output gate (softplus(gate) * attention_output), the
        // same mechanism Laguna's own scalar attention applies - see
        // AttentionKernel in laguna/gpu/kernels/kernels.hip.cpp. Only
        // Laguna's v1 (block-diffusion) drafter GGUF has this tensor -
        // v2 (gated-delta-net) drafters, e.g. 27B/35B/GLM's, do not, and
        // making it unconditionally required broke loading theirs (found
        // via this session's own regression check against ~/gufo).
        (is_v1 && !BindRequiredTensor(reader, layer_prefix + "attn_gate.weight",
                                      hidden * config.num_attention_heads, true,
                                      transformer.attn_gate, error_msg)) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_norm.weight", hidden,
                            false, transformer.ffn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_gate.weight",
                            intermediate * hidden, true, transformer.ffn_gate,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_up.weight",
                            intermediate * hidden, true, transformer.ffn_up,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_down.weight",
                            hidden * intermediate, true, transformer.ffn_down,
                            error_msg) ||
        (!is_v1 &&
         (!BindRequiredTensor(reader, layer_prefix + "attn_conv_base",
                              hidden * df_cfg.conv_kernel_size * 2U, false,
                              layer.attention_conv_base, error_msg) ||
          !BindRequiredTensor(reader, layer_prefix + "attn_conv_proj.weight",
                              dynamic_coefficients * hidden, true,
                              layer.attention_conv_projection, error_msg) ||
          !BindRequiredTensor(reader, layer_prefix + "ffn_conv_base",
                              hidden * df_cfg.conv_kernel_size * 2U, false,
                              layer.ffn_conv_base, error_msg) ||
          !BindRequiredTensor(reader, layer_prefix + "ffn_conv_proj.weight",
                              dynamic_coefficients * hidden, true,
                              layer.ffn_conv_projection, error_msg)))) {
      return std::nullopt;
    }
    // Re-encode the dense BF16 weights to Q8_0: the drafter's own forward
    // pass (RunBlockGemm in executor.cpp) already has a Q8_0 dispatch
    // branch (LaunchBatchedQuantGEMMFp32), it was just never exercised
    // because every drafter weight was BF16. Halves the bytes read per
    // call for the same math - see laguna-port-state.md for the measured
    // impact. (rows, cols) per tensor follows the same GGUF convention
    // the binding calls above already use (ne0 = cols = fastest/input
    // dim, ne1 = rows = output dim); attn_output's K is the attention
    // width, not hidden, since it projects attention-space back to
    // hidden-space.
    const auto requantize = [&](models::QwenTensorRef& ref, std::uint32_t rows,
                                std::uint32_t cols) {
      auto q8 =
          ReencodeBf16ToQ8_0(ref, rows, cols, &weights.owned_reencoded_storage);
      if (!q8.empty())
        ref = q8;
    };
    requantize(transformer.attn_q, attention, hidden);
    requantize(transformer.attn_k, kv, hidden);
    requantize(transformer.attn_v, kv, hidden);
    requantize(transformer.attn_output, hidden, attention);
    requantize(transformer.ffn_gate, intermediate, hidden);
    requantize(transformer.ffn_up, intermediate, hidden);
    requantize(transformer.ffn_down, hidden, intermediate);
    // DEBUG (2026-10-30): force attn_gate.weight's pages resident via a CPU
    // read before any GPU kernel touches it, to test whether its GPU fault
    // (confirmed, see laguna-port-state.md) is a demand-paging/mmap
    // residency issue rather than an addressing bug.
    if (std::getenv("LAGUNA_DFLASH_DEBUG") != nullptr) {
      const auto* bytes =
          static_cast<const volatile std::uint8_t*>(transformer.attn_gate.data);
      const std::size_t total_bytes =
          transformer.attn_gate.num_elements * 2;  // BF16
      std::uint64_t checksum = 0;
      for (std::size_t off = 0; off < total_bytes; off += 4096)
        checksum += bytes[off];
      checksum += bytes[total_bytes - 1];
      std::uintptr_t file_offset = 0;
      for (const auto& region : reader.GetMappedRegions()) {
        const auto tensor_addr =
            reinterpret_cast<std::uintptr_t>(transformer.attn_gate.data);
        const auto region_addr = reinterpret_cast<std::uintptr_t>(region.data);
        if (tensor_addr >= region_addr &&
            tensor_addr - region_addr < region.size) {
          file_offset = tensor_addr - region_addr;
          break;
        }
      }
      std::fprintf(stderr,
                   "[LAGUNA_DFLASH_DEBUG] prefaulted attn_gate layer=%u "
                   "bytes=%zu checksum=%llu computed_file_offset=%zu\n",
                   i, total_bytes, static_cast<unsigned long long>(checksum),
                   static_cast<std::size_t>(file_offset));
    }
  }

  return weights;
}

}  // namespace gufo::speculative
