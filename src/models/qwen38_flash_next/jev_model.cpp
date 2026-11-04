#include "src/models/qwen38_flash_next/jev_model.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

using core::GgmlType;
using tokenization::TokenId;

constexpr std::size_t kQ4Block = 32;
constexpr std::size_t kQ6Block = 256;

struct BlockQ4_0 {
  std::uint16_t d;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18);

struct BlockQ4_1 {
  std::uint16_t d;
  std::uint16_t m;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_1) == 20);

struct JevTensor {
  const void* data{nullptr};
  GgmlType type{GgmlType::kF32};
  std::size_t elements{0};
  std::size_t available_bytes{std::numeric_limits<std::size_t>::max()};

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || elements == 0;
  }
};

struct JevLayer {
  JevTensor attn_norm;
  JevTensor ffn_norm;
  JevTensor wq;
  JevTensor wk;
  JevTensor wv;
  JevTensor wo;
  JevTensor q_norm;
  JevTensor k_norm;
  JevTensor ffn_gate;
  JevTensor ffn_up;
  JevTensor ffn_down;
};

struct JevConfig {
  std::uint32_t num_layers{0};
  std::uint32_t hidden{0};
  std::uint32_t intermediate{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t vocab{0};
  std::uint32_t context{0};
  float rope_theta{1000000.0F};
  float rms_eps{1e-6F};
};

[[nodiscard]] std::size_t RowBytes(GgmlType type, std::size_t columns,
                                   std::string* error_msg) {
  switch (type) {
    case GgmlType::kF32:
      return columns * sizeof(float);
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return columns * sizeof(std::uint16_t);
    default:
      break;
  }
  const std::size_t bytes = gufo::quant::QuantizedRowBytes(type, columns);
  if (bytes == 0 && error_msg != nullptr) {
    *error_msg = "jev tensor has an unsupported row layout";
  }
  return bytes;
}

JevTensor BindTensor(const core::GgufReader& reader, std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (address >= region_address) {
      const std::size_t offset = address - region_address;
      if (offset < region.size) {
        available = region.size - offset;
        break;
      }
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .elements = tensor->ElementCount(),
          .available_bytes = available};
}

void DequantizeQ4_0(const void* src, float* dst, std::size_t count) {
  const auto* blocks = static_cast<const BlockQ4_0*>(src);
  for (std::size_t b = 0; b < count / kQ4Block; ++b) {
    const float d = gufo::quant::Fp16ToFloat(blocks[b].d);
    const auto* qs = blocks[b].qs;
    for (std::size_t j = 0; j < 16; ++j) {
      dst[b * kQ4Block + j] =
          d * static_cast<float>(static_cast<int>(qs[j] & 0x0FU) - 8);
      dst[b * kQ4Block + j + 16] =
          d * static_cast<float>(static_cast<int>(qs[j] >> 4U) - 8);
    }
  }
}

void DequantizeQ4_1(const void* src, float* dst, std::size_t count) {
  const auto* blocks = static_cast<const BlockQ4_1*>(src);
  for (std::size_t b = 0; b < count / kQ4Block; ++b) {
    const float d = gufo::quant::Fp16ToFloat(blocks[b].d);
    const float m = gufo::quant::Fp16ToFloat(blocks[b].m);
    const auto* qs = blocks[b].qs;
    for (std::size_t j = 0; j < 16; ++j) {
      dst[b * kQ4Block + j] = (d * static_cast<float>(qs[j] & 0x0FU)) + m;
      dst[b * kQ4Block + j + 16] = (d * static_cast<float>(qs[j] >> 4U)) + m;
    }
  }
}

void DequantizeRow(const JevTensor& tensor, std::size_t row, std::size_t cols,
                   float* out) {
  const std::size_t stride = RowBytes(tensor.type, cols, nullptr);
  const auto* src =
      static_cast<const std::uint8_t*>(tensor.data) + row * stride;
  switch (tensor.type) {
    case GgmlType::kQ4_0:
      DequantizeQ4_0(src, out, cols);
      return;
    case GgmlType::kQ4_1:
      DequantizeQ4_1(src, out, cols);
      return;
    default:
      gufo::quant::Dequantize(tensor.type, src, out, cols);
      return;
  }
}

float DotQ4_0(const void* row, std::span<const float> x, std::size_t k) {
  const auto* blocks = static_cast<const BlockQ4_0*>(row);
  float sum = 0.0F;
  for (std::size_t b = 0; b < k / kQ4Block; ++b) {
    const float d = gufo::quant::Fp16ToFloat(blocks[b].d);
    const auto* qs = blocks[b].qs;
    const float* v = x.data() + b * kQ4Block;
    for (std::size_t j = 0; j < 16; ++j) {
      sum += d * static_cast<float>(static_cast<int>(qs[j] & 0x0FU) - 8) * v[j];
      sum +=
          d * static_cast<float>(static_cast<int>(qs[j] >> 4U) - 8) * v[j + 16];
    }
  }
  return sum;
}

float DotQ4_1(const void* row, std::span<const float> x, std::size_t k) {
  const auto* blocks = static_cast<const BlockQ4_1*>(row);
  float sum = 0.0F;
  for (std::size_t b = 0; b < k / kQ4Block; ++b) {
    const float d = gufo::quant::Fp16ToFloat(blocks[b].d);
    const float m = gufo::quant::Fp16ToFloat(blocks[b].m);
    const auto* qs = blocks[b].qs;
    const float* v = x.data() + b * kQ4Block;
    for (std::size_t j = 0; j < 16; ++j) {
      sum += ((d * static_cast<float>(qs[j] & 0x0FU)) + m) * v[j];
      sum += ((d * static_cast<float>(qs[j] >> 4U)) + m) * v[j + 16];
    }
  }
  return sum;
}

[[nodiscard]] float DotRow(const JevTensor& tensor, std::size_t row,
                           std::span<const float> x) {
  const std::size_t k = x.size();
  const std::size_t stride = RowBytes(tensor.type, k, nullptr);
  const auto* src =
      static_cast<const std::uint8_t*>(tensor.data) + row * stride;
  switch (tensor.type) {
    case GgmlType::kQ4_0:
      return DotQ4_0(src, x, k);
    case GgmlType::kQ4_1:
      return DotQ4_1(src, x, k);
    default:
      return gufo::quant::Dot(tensor.type, src, x, k);
  }
}

void Gemv(const JevTensor& tensor, std::span<const float> x,
          std::span<float> y) {
#pragma omp parallel for schedule(static)
  for (std::size_t m = 0; m < y.size(); ++m) {
    y[m] = DotRow(tensor, m, x);
  }
}

void RmsNorm(std::span<const float> x, const JevTensor& weight, float eps,
             std::span<float> out) {
  double sum = 0.0;
  for (const float v : x) {
    sum += static_cast<double>(v) * static_cast<double>(v);
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(sum / static_cast<double>(x.size())) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * scale * static_cast<const float*>(weight.data)[i];
  }
}

void RmsNormHead(std::span<float> head, const JevTensor& weight, float eps) {
  double sum = 0.0;
  for (const float v : head) {
    sum += static_cast<double>(v) * static_cast<double>(v);
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(sum / static_cast<double>(head.size())) +
                eps);
  const auto* w = static_cast<const float*>(weight.data);
  for (std::size_t i = 0; i < head.size(); ++i) {
    head[i] = head[i] * scale * w[i];
  }
}

float Silu(float v) noexcept {
  return v / (1.0F + std::exp(-v));
}

[[nodiscard]] std::string LowerTrimmed(std::string_view text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  std::string out;
  out.reserve(end - begin);
  for (std::size_t i = begin; i < end; ++i) {
    const char c = text[i];
    out.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c);
  }
  return out;
}

}  // namespace

struct JevModel::Impl {
  JevConfig config;
  JevTensor token_embd;
  JevTensor output_norm;
  JevTensor output;
  std::vector<JevLayer> layers;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer;
  std::vector<TokenId> yes_ids;
  std::vector<TokenId> no_ids;

  mutable std::vector<std::vector<float>> keys;
  mutable std::vector<std::vector<float>> values;

  void ResetCache() const {
    if (keys.size() != config.num_layers) {
      keys.assign(config.num_layers, {});
      values.assign(config.num_layers, {});
    }
    for (auto& cache : keys) {
      cache.clear();
    }
    for (auto& cache : values) {
      cache.clear();
    }
  }

  // Runs one token and returns the final hidden vector before the LM head.
  void Forward(std::uint32_t token, std::uint32_t pos,
               std::span<float> hidden_out) const {
    const std::size_t hidden = config.hidden;
    const std::size_t heads = config.heads;
    const std::size_t kv_heads = config.kv_heads;
    const std::size_t head_dim = config.head_dim;
    const std::size_t q_dim = heads * head_dim;
    const std::size_t kv_dim = kv_heads * head_dim;

    std::vector<float> x(hidden);
    DequantizeRow(token_embd, token, hidden, x.data());

    std::vector<float> norm(hidden);
    std::vector<float> q(q_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    std::vector<float> attn(q_dim);
    std::vector<float> attn_out(hidden);
    std::vector<float> gate(config.intermediate);
    std::vector<float> up(config.intermediate);
    std::vector<float> act(config.intermediate);
    std::vector<float> ffn_out(hidden);
    std::vector<float> scores(pos + 1);

    const float attention_scale =
        1.0F / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t layer_idx = 0; layer_idx < config.num_layers;
         ++layer_idx) {
      const auto& layer = layers[layer_idx];
      RmsNorm(x, layer.attn_norm, config.rms_eps, norm);
      Gemv(layer.wq, norm, q);
      Gemv(layer.wk, norm, k);
      Gemv(layer.wv, norm, v);

      for (std::size_t h = 0; h < heads; ++h) {
        RmsNormHead(std::span<float>(q.data() + h * head_dim, head_dim),
                    layer.q_norm, config.rms_eps);
      }
      for (std::size_t h = 0; h < kv_heads; ++h) {
        RmsNormHead(std::span<float>(k.data() + h * head_dim, head_dim),
                    layer.k_norm, config.rms_eps);
      }

      models::ForwardRoPE(q, k, static_cast<std::uint32_t>(heads),
                          static_cast<std::uint32_t>(kv_heads),
                          static_cast<std::uint32_t>(head_dim),
                          static_cast<std::uint32_t>(head_dim), pos,
                          config.rope_theta);

      auto& layer_keys = keys[layer_idx];
      auto& layer_values = values[layer_idx];
      layer_keys.insert(layer_keys.end(), k.begin(), k.end());
      layer_values.insert(layer_values.end(), v.begin(), v.end());

      const std::size_t group = heads / kv_heads;
      for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t kv_h = h / group;
        const float* q_head = q.data() + h * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t p = 0; p <= pos; ++p) {
          const float* k_head =
              layer_keys.data() + p * kv_dim + kv_h * head_dim;
          float dot = 0.0F;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot += q_head[d] * k_head[d];
          }
          scores[p] = dot * attention_scale;
          max_score = std::max(max_score, scores[p]);
        }
        double sum_exp = 0.0;
        for (std::size_t p = 0; p <= pos; ++p) {
          scores[p] = std::exp(scores[p] - max_score);
          sum_exp += static_cast<double>(scores[p]);
        }
        float* out_head = attn.data() + h * head_dim;
        std::fill(out_head, out_head + head_dim, 0.0F);
        for (std::size_t p = 0; p <= pos; ++p) {
          const float weight = static_cast<float>(scores[p] / sum_exp);
          const float* v_head =
              layer_values.data() + p * kv_dim + kv_h * head_dim;
          for (std::size_t d = 0; d < head_dim; ++d) {
            out_head[d] += weight * v_head[d];
          }
        }
      }

      Gemv(layer.wo, attn, attn_out);
      for (std::size_t i = 0; i < hidden; ++i) {
        x[i] += attn_out[i];
      }

      RmsNorm(x, layer.ffn_norm, config.rms_eps, norm);
      Gemv(layer.ffn_gate, norm, gate);
      Gemv(layer.ffn_up, norm, up);
      for (std::size_t i = 0; i < config.intermediate; ++i) {
        act[i] = Silu(gate[i]) * up[i];
      }
      Gemv(layer.ffn_down, act, ffn_out);
      for (std::size_t i = 0; i < hidden; ++i) {
        x[i] += ffn_out[i];
      }
    }

    RmsNorm(x, output_norm, config.rms_eps, hidden_out);
  }

  [[nodiscard]] float LabelLogit(std::span<const float> hidden,
                                 TokenId token) const {
    return DotRow(output, token, hidden);
  }

  [[nodiscard]] JevBinaryVerdict Score(std::string_view question,
                                       std::string_view context) const {
    std::string prompt;
    prompt.reserve(question.size() + context.size() + 512);
    prompt.append(tokenization::kImStart);
    prompt.append(
        "system\nYou judge software reasoning. Answer each yes/no question "
        "with Yes or No.<|im_end|>\n");
    prompt.append(tokenization::kImStart);
    prompt.append("user\n");
    if (!context.empty()) {
      prompt.append(context);
      prompt.append("\n\n");
    }
    prompt.append("Question: ");
    prompt.append(question);
    prompt.append("\nAnswer with Yes or No.<|im_end|>\n");
    prompt.append(tokenization::kImStart);
    prompt.append("assistant\n");

    const auto tokens = tokenizer->Encode(prompt);
    if (tokens.empty()) {
      return {};
    }

    ResetCache();
    std::vector<float> hidden(config.hidden);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      Forward(tokens[i], static_cast<std::uint32_t>(i), hidden);
    }

    double yes = 0.0;
    for (const auto id : yes_ids) {
      yes += std::exp(static_cast<double>(LabelLogit(hidden, id)));
    }
    double no = 0.0;
    for (const auto id : no_ids) {
      no += std::exp(static_cast<double>(LabelLogit(hidden, id)));
    }
    const double total = yes + no;
    if (!(total > 0.0) || !std::isfinite(total)) {
      return {};
    }
    return {static_cast<float>(yes / total), static_cast<float>(no / total)};
  }
};

JevModel::JevModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
JevModel::~JevModel() = default;

std::unique_ptr<JevModel> JevModel::LoadFromGguf(const core::GgufReader& reader,
                                                 std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "qwen3") {
    if (error_msg != nullptr) {
      *error_msg = "jev requires a dense qwen3 GGUF";
    }
    return nullptr;
  }
  const std::string prefix = std::string(*architecture) + ".";

  auto config = std::make_unique<Impl>();
  auto& cfg = config->config;
  const auto layers = reader.GetMetadataUint32(prefix + "block_count");
  const auto hidden = reader.GetMetadataUint32(prefix + "embedding_length");
  const auto intermediate =
      reader.GetMetadataUint32(prefix + "feed_forward_length");
  const auto heads = reader.GetMetadataUint32(prefix + "attention.head_count");
  const auto kv_heads =
      reader.GetMetadataUint32(prefix + "attention.head_count_kv");
  if (!layers || !hidden || !intermediate || !heads || !kv_heads) {
    if (error_msg != nullptr) {
      *error_msg = "jev GGUF is missing dense Qwen3 architecture metadata";
    }
    return nullptr;
  }
  cfg.num_layers = *layers;
  cfg.hidden = *hidden;
  cfg.intermediate = *intermediate;
  cfg.heads = *heads;
  cfg.kv_heads = *kv_heads;
  cfg.head_dim = reader.GetMetadataUint32(prefix + "attention.key_length")
                     .value_or(cfg.hidden / cfg.heads);
  cfg.context =
      reader.GetMetadataUint32(prefix + "context_length").value_or(4096);
  cfg.rope_theta =
      reader.GetMetadataFloat32(prefix + "rope.freq_base").value_or(1000000.0F);
  cfg.rms_eps =
      reader.GetMetadataFloat32(prefix + "attention.layer_norm_rms_epsilon")
          .value_or(1e-6F);
  if (cfg.kv_heads == 0 || cfg.head_dim == 0 ||
      (cfg.heads % cfg.kv_heads) != 0) {
    if (error_msg != nullptr) {
      *error_msg = "jev GGUF has an invalid attention layout";
    }
    return nullptr;
  }

  config->token_embd = BindTensor(reader, "token_embd.weight");
  config->output_norm = BindTensor(reader, "output_norm.weight");
  config->output = BindTensor(reader, "output.weight");
  if (config->output.empty()) {
    config->output = config->token_embd;
  }
  if (config->token_embd.empty() || config->output_norm.empty() ||
      config->output.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "jev GGUF is missing embedding or output tensors";
    }
    return nullptr;
  }
  if (config->token_embd.elements % cfg.hidden != 0) {
    if (error_msg != nullptr) {
      *error_msg = "jev token embedding is not a whole number of rows";
    }
    return nullptr;
  }
  cfg.vocab =
      static_cast<std::uint32_t>(config->token_embd.elements / cfg.hidden);

  config->layers.resize(cfg.num_layers);
  for (std::uint32_t i = 0; i < cfg.num_layers; ++i) {
    const std::string layer_prefix = "blk." + std::to_string(i) + ".";
    auto& layer = config->layers[i];
    layer.attn_norm = BindTensor(reader, layer_prefix + "attn_norm.weight");
    layer.ffn_norm = BindTensor(reader, layer_prefix + "ffn_norm.weight");
    layer.wq = BindTensor(reader, layer_prefix + "attn_q.weight");
    layer.wk = BindTensor(reader, layer_prefix + "attn_k.weight");
    layer.wv = BindTensor(reader, layer_prefix + "attn_v.weight");
    layer.wo = BindTensor(reader, layer_prefix + "attn_output.weight");
    layer.q_norm = BindTensor(reader, layer_prefix + "attn_q_norm.weight");
    layer.k_norm = BindTensor(reader, layer_prefix + "attn_k_norm.weight");
    layer.ffn_gate = BindTensor(reader, layer_prefix + "ffn_gate.weight");
    layer.ffn_up = BindTensor(reader, layer_prefix + "ffn_up.weight");
    layer.ffn_down = BindTensor(reader, layer_prefix + "ffn_down.weight");
    if (layer.attn_norm.empty() || layer.ffn_norm.empty() || layer.wq.empty() ||
        layer.wk.empty() || layer.wv.empty() || layer.wo.empty() ||
        layer.q_norm.empty() || layer.k_norm.empty() ||
        layer.ffn_gate.empty() || layer.ffn_up.empty() ||
        layer.ffn_down.empty()) {
      if (error_msg != nullptr) {
        *error_msg = "jev GGUF is missing tensors in " + layer_prefix;
      }
      return nullptr;
    }
  }

  config->tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(reader, error_msg);
  if (!config->tokenizer) {
    return nullptr;
  }
  for (std::size_t id = 0; id < config->tokenizer->GetVocabSize(); ++id) {
    const std::string word =
        LowerTrimmed(config->tokenizer->DecodeToken(static_cast<TokenId>(id)));
    if (word == "yes") {
      config->yes_ids.push_back(static_cast<TokenId>(id));
    } else if (word == "no") {
      config->no_ids.push_back(static_cast<TokenId>(id));
    }
  }
  if (config->yes_ids.empty() || config->no_ids.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "jev vocabulary has no Yes/No answer tokens";
    }
    return nullptr;
  }

  return std::unique_ptr<JevModel>(new JevModel(std::move(config)));
}

JevBinaryVerdict JevModel::Answer(std::string_view question,
                                  std::string_view context) const {
  if (question.empty()) {
    return {};
  }
  return impl_->Score(question, context);
}

}  // namespace gufo::models::qwen38_flash_next
