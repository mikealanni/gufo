#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/moe_ffn.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"

namespace gufo::hip {
namespace {

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

}  // namespace

QwenMtpGpuExecutor::QwenMtpGpuExecutor(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context)
    : model_(std::move(model)),
      max_context_(max_context),
      h_last_hidden_(model_->GetConfig().hidden_size),
      h_logits_(model_->GetConfig().vocab_size) {
  try {
    Allocate();
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenMtpGpuExecutor::~QwenMtpGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenMtpGpuExecutor> QwenMtpGpuExecutor::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU executor context is invalid";
    }
    return nullptr;
  }
  try {
    return std::unique_ptr<QwenMtpGpuExecutor>(
        new QwenMtpGpuExecutor(std::move(model), max_context));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenMtpGpuExecutor::Allocate() {
  const auto& config = model_->GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  const auto stream_error = hipStreamCreate(&stream_);
  if (stream_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream creation failed: ") +
                             hipGetErrorString(stream_error));
  }
  AllocateBuffer(d_target_hidden_, hidden);
  AllocateBuffer(d_embedding_, hidden);
  AllocateBuffer(d_fusion_, 2 * hidden);
  AllocateBuffer(d_hidden_, hidden);
  AllocateBuffer(d_normed_, hidden);
  AllocateBuffer(d_qg_, 2 * attention);
  AllocateBuffer(d_q_, attention);
  AllocateBuffer(d_k_, kv);
  AllocateBuffer(d_v_, kv);
  AllocateBuffer(d_gate_, attention);
  AllocateBuffer(d_context_, attention);
  AllocateBuffer(d_attn_out_, hidden);
  AllocateBuffer(d_ffn_act_, std::max<std::size_t>(intermediate, 1));
  if (config.IsMoe()) {
    const std::size_t experts = config.num_experts;
    const std::size_t used = config.num_experts_used;
    const std::size_t ff = config.expert_ff;
    const std::size_t shared_ff = config.shared_expert_ff;
    const std::size_t total = experts + 1 + used + used + 1 + (used * ff) +
                              (used * hidden) + (3 * shared_ff) + hidden + 8;
    AllocateBuffer(d_moe_buffer_, total);
    float* cursor = d_moe_buffer_;
    const auto take = [&cursor](std::size_t count) {
      float* start = cursor;
      cursor += count;
      return start;
    };
    moe_.router_logits = take(experts);
    moe_.shared_logit = take(1);
    moe_.ids = reinterpret_cast<std::int32_t*>(take(used));
    moe_.weights = take(used);
    moe_.shared_weight = take(1);
    moe_.act = take(used * ff);
    moe_.down_out = take(used * hidden);
    moe_.shared_gate = take(shared_ff);
    moe_.shared_up = take(shared_ff);
    moe_.shared_act = take(shared_ff);
    moe_.shared_out = take(hidden);
  }
  AllocateBuffer(d_ffn_out_, hidden);
  AllocateBuffer(d_feedback_hidden_, hidden);
  AllocateBuffer(d_logits_, config.vocab_size);
  AllocateBuffer(d_kv_cache_, 2 * total_kv);
  d_kv_cache_f16_ =
      detail::AllocateDevice(2 * total_kv * sizeof(std::uint16_t));
  AllocateBuffer(d_split_k_scratch_,
                 detail::DecodeAttentionScratchElements(
                     config.num_attention_heads, config.head_dim));
  AllocateBuffer(d_out_token_, 1);
}

void QwenMtpGpuExecutor::Free() noexcept {
  const auto free_buffer = [](auto*& pointer) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
      pointer = nullptr;
    }
  };
  free_buffer(d_target_hidden_);
  free_buffer(d_embedding_);
  free_buffer(d_fusion_);
  free_buffer(d_hidden_);
  free_buffer(d_normed_);
  free_buffer(d_qg_);
  free_buffer(d_q_);
  free_buffer(d_k_);
  free_buffer(d_v_);
  free_buffer(d_gate_);
  free_buffer(d_context_);
  free_buffer(d_attn_out_);
  free_buffer(d_ffn_act_);
  free_buffer(d_moe_buffer_);
  free_buffer(d_ffn_out_);
  free_buffer(d_feedback_hidden_);
  free_buffer(d_logits_);
  free_buffer(d_kv_cache_);
  free_buffer(d_kv_cache_f16_);
  free_buffer(d_split_k_scratch_);
  free_buffer(d_out_token_);
  free_buffer(d_prime_tokens_);
  free_buffer(d_prime_buffer_);
  free_buffer(d_prime_bf16_);
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenMtpGpuExecutor::Reset() noexcept {
  // Attention only reads the valid prefix; new rows overwrite stale KV.
  next_position_ = 0;
}

void QwenMtpGpuExecutor::Synchronize() {
  const auto status = hipStreamSynchronize(stream_);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream synchronize failed: ") +
                             hipGetErrorString(status));
  }
}

void QwenMtpGpuExecutor::Rewind(std::uint32_t position) {
  if (position > next_position_) {
    throw std::out_of_range("MTP rewind position exceeds executed context");
  }
  next_position_ = position;
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardTargetHidden(
    tokenization::TokenId input_token, std::span<const float> target_hidden,
    std::uint32_t position, bool compute_logits) {
  if (target_hidden.size() != model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP target hidden width is invalid");
  }
  const auto copy_error = hipMemcpyAsync(d_target_hidden_, target_hidden.data(),
                                         target_hidden.size_bytes(),
                                         hipMemcpyHostToDevice, stream_);
  if (copy_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hidden copy failed: ") +
                             hipGetErrorString(copy_error));
  }
  return Run(input_token, d_target_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardFeedback(
    tokenization::TokenId input_token, std::uint32_t position,
    bool compute_logits) {
  return Run(input_token, d_feedback_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::Run(tokenization::TokenId input_token,
                                              const float* hidden_input,
                                              std::uint32_t position,
                                              bool compute_logits) {
  if (position != next_position_ || position >= max_context_) {
    throw std::out_of_range("MTP GPU position is not sequential");
  }
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  LaunchEmbeddingLookup(weights.token_embedding.data,
                        weights.token_embedding.type, input_token, d_embedding_,
                        hidden, stream_);
  if (vision_input_)
    vision_input_->Inject(d_embedding_, position + 1, 1, hidden, 1, stream_);
  LaunchRMSNorm(d_embedding_,
                static_cast<const float*>(weights.embedding_norm.data),
                d_fusion_, hidden, 1.0e-6F, stream_);
  LaunchRMSNorm(hidden_input,
                static_cast<const float*>(weights.hidden_norm.data),
                d_fusion_ + hidden, hidden, 1.0e-6F, stream_);
  LaunchGEMV(weights.fusion_projection.data, weights.fusion_projection.type,
             d_fusion_, d_hidden_, hidden, 2 * hidden, stream_,
             models::qwen::QwenGemmMode::kHipMtp);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.attn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedQKVProjections(
      layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
      layer.attn_k.type, layer.attn_v.data, layer.attn_v.type, d_normed_, d_qg_,
      d_k_, d_v_, 2 * attention, kv, hidden, stream_);
  LaunchUnpackQG(d_qg_, d_q_, d_gate_, config.num_attention_heads,
                 config.head_dim, stream_);
  LaunchPerHeadRMSNorm(d_q_, static_cast<const float*>(layer.attn_q_norm.data),
                       d_q_, config.num_attention_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchPerHeadRMSNorm(d_k_, static_cast<const float*>(layer.attn_k_norm.data),
                       d_k_, config.num_key_value_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchRoPE(d_q_, d_k_, config.num_attention_heads, config.num_key_value_heads,
             config.head_dim, config.rotary_dim, position, config.rope_theta,
             stream_, vision_input_ ? vision_input_->rope() : nullptr);
  LaunchAttention(
      d_q_, d_k_, d_v_, d_gate_, d_kv_cache_, d_kv_cache_ + total_kv,
      d_kv_cache_f16_, static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
      d_context_, 0, position, max_context_, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim, stream_, d_split_k_scratch_);
  LaunchGEMV(layer.attn_output.data, layer.attn_output.type, d_context_,
             d_attn_out_, hidden, attention, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  LaunchResidualAdd(d_hidden_, d_attn_out_, d_hidden_, hidden, stream_);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.ffn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  if (config.IsMoe()) {
    MoeFfnDecodeToken(layer, config, moe_, d_normed_, d_ffn_out_, stream_);
  } else {
    LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                          layer.ffn_up.data, layer.ffn_up.type, d_normed_,
                          d_ffn_act_, config.intermediate_size, hidden,
                          stream_);
    LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type, d_ffn_act_, d_ffn_out_,
               hidden, config.intermediate_size, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
  }
  LaunchResidualAdd(d_hidden_, d_ffn_out_, d_hidden_, hidden, stream_);
  LaunchRMSNorm(d_hidden_,
                static_cast<const float*>(weights.shared_head_norm.data),
                d_feedback_hidden_, hidden, 1.0e-6F, stream_);
  ++next_position_;

  if (!compute_logits) {
    return 0;
  }
  // The draft only proposes: scoring the most frequent token ids (the BPE
  // vocabulary is ordered by merge rank) reads a fraction of the output matrix.
  static const std::size_t kDraftVocab = [] {
    const char* v = std::getenv("GUFO_MTP_VOCAB");
    return v != nullptr ? static_cast<std::size_t>(std::atoll(v))
                        : std::size_t{65536};
  }();
  const std::size_t scored =
      (kDraftVocab == 0 || !config.IsMoe())
          ? static_cast<std::size_t>(config.vocab_size)
          : std::min<std::size_t>(kDraftVocab, config.vocab_size);
  LaunchGEMV(weights.output.data, weights.output.type, d_feedback_hidden_,
             d_logits_, scored, hidden, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  LaunchGPUArgmax(d_logits_, d_out_token_, scored, stream_);
  tokenization::TokenId result = 0;
  const auto copy_error = hipMemcpyAsync(&result, d_out_token_, sizeof(result),
                                         hipMemcpyDeviceToHost, stream_);
  if (copy_error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP GPU result synchronization failed");
  }
  return CheckedSampleToken(result, config.vocab_size);
}

void QwenMtpGpuExecutor::EnsurePrimeScratch() {
  if (d_prime_buffer_ != nullptr) {
    return;
  }
  const auto& config = model_->GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  prime_buffer_elements_ =
      kPrimeChunk * ((8 * hidden) + (3 * kv) + (2 * attention));
  AllocateBuffer(d_prime_buffer_, prime_buffer_elements_);
  AllocateBuffer(d_prime_tokens_, kPrimeChunk);
  d_prime_bf16_ =
      detail::AllocateDevice(kPrimeChunk * 2 * hidden * sizeof(std::uint16_t));
  HIP_CHECK(
      hipMemset(d_prime_buffer_, 0, prime_buffer_elements_ * sizeof(float)));
}

void QwenMtpGpuExecutor::PrimeBatch(
    std::span<const tokenization::TokenId> tokens,
    std::span<const float> hidden_rows, std::uint32_t start_position) {
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;
  if (start_position != next_position_ ||
      hidden_rows.size() != tokens.size() * hidden ||
      static_cast<std::size_t>(start_position) + tokens.size() > max_context_) {
    throw std::out_of_range("MTP batch priming range is invalid");
  }
  EnsurePrimeScratch();
  float* cursor = d_prime_buffer_;
  const auto take = [&cursor](std::size_t count) {
    float* start = cursor;
    cursor += count;
    return start;
  };
  float* emb = take(kPrimeChunk * hidden);
  float* hid = take(kPrimeChunk * hidden);
  float* en = take(kPrimeChunk * hidden);
  float* hn = take(kPrimeChunk * hidden);
  float* cat = take(kPrimeChunk * 2 * hidden);
  float* fused = take(kPrimeChunk * hidden);
  float* normed = take(kPrimeChunk * hidden);
  float* k = take(kPrimeChunk * kv);
  float* v = take(kPrimeChunk * kv);
  float* k_out = take(kPrimeChunk * kv);
  float* q = take(kPrimeChunk * attention);
  float* q_out = take(kPrimeChunk * attention);
  for (std::size_t offset = 0; offset < tokens.size(); offset += kPrimeChunk) {
    const std::size_t n = std::min(kPrimeChunk, tokens.size() - offset);
    const std::uint32_t position =
        start_position + static_cast<std::uint32_t>(offset);
    std::vector<std::uint32_t> host_tokens(tokens.begin() + offset,
                                           tokens.begin() + offset + n);
    HIP_CHECK(hipMemcpyAsync(d_prime_tokens_, host_tokens.data(),
                             n * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                             stream_));
    HIP_CHECK(hipMemcpyAsync(hid, hidden_rows.data() + (offset * hidden),
                             n * hidden * sizeof(float), hipMemcpyHostToDevice,
                             stream_));
    LaunchBatchedEmbeddingLookup(weights.token_embedding.data,
                                 weights.token_embedding.type, d_prime_tokens_,
                                 emb, n, hidden, stream_);
    if (vision_input_) {
      vision_input_->Inject(emb, position + 1, static_cast<std::uint32_t>(n),
                            hidden, 1, stream_);
    }
    LaunchBatchedRMSNorm(emb,
                         static_cast<const float*>(weights.embedding_norm.data),
                         en, nullptr, n, hidden, 1.0e-6F, stream_);
    LaunchBatchedRMSNorm(hid,
                         static_cast<const float*>(weights.hidden_norm.data),
                         hn, nullptr, n, hidden, 1.0e-6F, stream_);
    LaunchConcatRows(en, hn, cat, n, hidden, stream_);
    // Decode-equivalent FP32-activation GEMMs: the cache entries match what
    // the single-token path would have written.
    LaunchBatchedQuantGEMMFp32(weights.fusion_projection.type,
                               weights.fusion_projection.data, cat, fused, n,
                               hidden, 2 * hidden, stream_);
    LaunchBatchedRMSNorm(fused, static_cast<const float*>(layer.attn_norm.data),
                         normed, nullptr, n, hidden, 1.0e-6F, stream_);
    LaunchBatchedQuantGEMMFp32(layer.attn_k.type, layer.attn_k.data, normed, k,
                               n, kv, hidden, stream_);
    LaunchBatchedQuantGEMMFp32(layer.attn_v.type, layer.attn_v.data, normed, v,
                               n, kv, hidden, stream_);
    LaunchBatchedFusedQKNormRoPEKvWrite(
        q, k, v, static_cast<const float*>(layer.attn_q_norm.data),
        static_cast<const float*>(layer.attn_k_norm.data), q_out, k_out,
        d_kv_cache_, d_kv_cache_ + total_kv, d_kv_cache_f16_,
        static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv, 0, position, n,
        max_context_, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.rotary_dim, config.rope_theta, 1.0e-6F, stream_,
        vision_input_ ? vision_input_->rope() : nullptr);
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  next_position_ = start_position + static_cast<std::uint32_t>(tokens.size());
}

std::size_t QwenMtpGpuExecutor::StateBytes() const noexcept {
  const auto& config = model_->GetConfig();
  return sizeof(std::uint32_t) +
         static_cast<std::size_t>(next_position_) * config.num_key_value_heads *
             config.head_dim * 2 * (sizeof(float) + sizeof(std::uint16_t));
}

void QwenMtpGpuExecutor::SaveState(std::span<std::uint8_t> destination) const {
  if (destination.size() != StateBytes()) {
    throw std::invalid_argument("MTP state destination has the wrong size");
  }
  const auto& config = model_->GetConfig();
  const std::size_t head_values =
      static_cast<std::size_t>(next_position_) * config.head_dim;
  const std::size_t head_stride =
      static_cast<std::size_t>(max_context_) * config.head_dim;
  const std::size_t total_kv = config.num_key_value_heads * head_stride;
  std::uint8_t* out = destination.data();
  std::memcpy(out, &next_position_, sizeof(next_position_));
  out += sizeof(next_position_);
  HIP_CHECK(hipStreamSynchronize(stream_));
  for (std::size_t plane = 0; plane < 2; ++plane) {
    for (std::size_t head = 0; head < config.num_key_value_heads; ++head) {
      const std::size_t offset = (plane * total_kv) + (head * head_stride);
      HIP_CHECK(hipMemcpy(out, d_kv_cache_ + offset,
                          head_values * sizeof(float), hipMemcpyDeviceToHost));
      out += head_values * sizeof(float);
    }
  }
  for (std::size_t plane = 0; plane < 2; ++plane) {
    for (std::size_t head = 0; head < config.num_key_value_heads; ++head) {
      const std::size_t offset = (plane * total_kv) + (head * head_stride);
      HIP_CHECK(hipMemcpy(
          out, static_cast<const std::uint16_t*>(d_kv_cache_f16_) + offset,
          head_values * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
      out += head_values * sizeof(std::uint16_t);
    }
  }
}

void QwenMtpGpuExecutor::LoadState(std::span<const std::uint8_t> source) {
  if (source.size() < sizeof(std::uint32_t)) {
    throw std::invalid_argument("MTP state payload is truncated");
  }
  std::uint32_t position = 0;
  std::memcpy(&position, source.data(), sizeof(position));
  const auto& config = model_->GetConfig();
  const std::size_t expected =
      sizeof(std::uint32_t) + static_cast<std::size_t>(position) *
                                  config.num_key_value_heads * config.head_dim *
                                  2 * (sizeof(float) + sizeof(std::uint16_t));
  if (position > max_context_ || source.size() != expected) {
    throw std::invalid_argument("MTP state payload does not match the cache");
  }
  const std::size_t head_values =
      static_cast<std::size_t>(position) * config.head_dim;
  const std::size_t head_stride =
      static_cast<std::size_t>(max_context_) * config.head_dim;
  const std::size_t total_kv = config.num_key_value_heads * head_stride;
  const std::uint8_t* in = source.data() + sizeof(position);
  HIP_CHECK(hipStreamSynchronize(stream_));
  for (std::size_t plane = 0; plane < 2; ++plane) {
    for (std::size_t head = 0; head < config.num_key_value_heads; ++head) {
      const std::size_t offset = (plane * total_kv) + (head * head_stride);
      HIP_CHECK(hipMemcpy(d_kv_cache_ + offset, in, head_values * sizeof(float),
                          hipMemcpyHostToDevice));
      in += head_values * sizeof(float);
    }
  }
  for (std::size_t plane = 0; plane < 2; ++plane) {
    for (std::size_t head = 0; head < config.num_key_value_heads; ++head) {
      const std::size_t offset = (plane * total_kv) + (head * head_stride);
      HIP_CHECK(hipMemcpy(static_cast<std::uint16_t*>(d_kv_cache_f16_) + offset,
                          in, head_values * sizeof(std::uint16_t),
                          hipMemcpyHostToDevice));
      in += head_values * sizeof(std::uint16_t);
    }
  }
  next_position_ = position;
}

QwenGpuMemoryUsage QwenMtpGpuExecutor::EstimateMemoryUsage(
    const QwenMtpGpuModel& model, std::uint32_t max_context) {
  const auto& config = model.GetConfig();
  const std::size_t kv_values = 2 * static_cast<std::size_t>(max_context) *
                                config.num_key_value_heads * config.head_dim;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t scratch_floats =
      kPrimeChunk *
          ((8 * hidden) + (2 * config.num_key_value_heads * config.head_dim) +
           (2 * attention)) +
      config.vocab_size + (32 * hidden);
  return {.request_state_bytes =
              kv_values * (sizeof(float) + sizeof(std::uint16_t)),
          .temporary_scratch_bytes =
              (scratch_floats * sizeof(float)) +
              (kPrimeChunk * 2 * hidden * sizeof(std::uint16_t))};
}

QwenGpuMemoryUsage QwenMtpGpuExecutor::GetMemoryUsage() const noexcept {
  return EstimateMemoryUsage(*model_, max_context_);
}

std::span<const float> QwenMtpGpuExecutor::CopyLastHidden() {
  const auto error = hipMemcpyAsync(h_last_hidden_.data(), d_feedback_hidden_,
                                    h_last_hidden_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP hidden synchronization failed");
  }
  return h_last_hidden_;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastLogits() {
  const auto error = hipMemcpyAsync(h_logits_.data(), d_logits_,
                                    h_logits_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP logit synchronization failed");
  }
  return h_logits_;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
