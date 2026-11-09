#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
void QwenGpuExecutor::CheckPrefillCancellation() const {
  if (cancellation_check_ && cancellation_check_())
    throw std::runtime_error("Qwen prefill cancelled");
}

void QwenGpuExecutor::PrefillLayerCheckpoint(std::uint32_t layer) {
  CheckPrefillCancellation();
  // Keep one four-layer group ahead of the GPU. Without backpressure, AR
  // can enqueue the whole prompt before the client disconnects; checking
  // cancellation on the host then cannot stop seconds of abandoned work.
  if (!cancellation_check_ || layer == 0 || layer % 4 != 0)
    return;
  for (auto& event : prefill_events_) {
    if (!event)
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  }
  const auto slot = (layer / 4 - 1) % prefill_events_.size();
  HIP_CHECK(hipEventRecord(prefill_events_[slot], arena_.stream));
  if (layer < 8)
    return;
  hipError_t status;
  while ((status = hipEventQuery(prefill_events_[1 - slot])) ==
         hipErrorNotReady) {
    CheckPrefillCancellation();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  HIP_CHECK(status);
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  CheckPrefillCancellation();
  if (prompt_tokens.empty()) {
    return 0;
  }
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  if (capture_prompt_hidden_) {
    h_prompt_hidden_.Clear();
    const std::size_t captured_layers =
        std::max<std::size_t>(arena_.GetTargetLayerCapture().size(), 1U);
    const std::size_t row = captured_layers * weights_.config.hidden_size;
    // Pinned up to one prompt batch (the serve path's largest call); a longer
    // prompt from a non-serve caller captures into pageable memory instead.
    h_prompt_hidden_.Reserve(prompt_tokens.size() * row,
                             std::size_t{GetMaxPromptBatch()} * row);
  }

  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }

  // With the NPU offload, consecutive full chunks advance in pairs so the NPU
  // stages of one overlap the GPU work of the other.
  const std::size_t max_batch = arena_.GetMaxBatch();
  const char* lockstep_env = std::getenv("GUFO_QWEN_LOCKSTEP");
  QwenNpuFfn* const npu_ffn = model_->NpuFfn();
  const bool lockstep = npu_ffn != nullptr && npu_ffn->Tokens() == max_batch &&
                        !capture_prompt_hidden_ &&
                        vision_input_.rope() == nullptr &&
                        (lockstep_env == nullptr || lockstep_env[0] != '0');

  static const std::size_t kMinPairedTokens = [] {
    const char* v = std::getenv("GUFO_QWEN_PAIR_MIN");
    return v != nullptr ? static_cast<std::size_t>(std::atoll(v))
                        : std::size_t{5000};
  }();
  tokenization::TokenId next_token = 0;
  for (std::size_t offset = 0; offset < prompt_tokens.size();
       offset += arena_.GetMaxBatch()) {
    CheckPrefillCancellation();
    const std::size_t remaining = prompt_tokens.size() - offset;
    if (lockstep && remaining >= kMinPairedTokens) {
      // Two full chunks, or one shorter run split in halves: either way the
      // NPU stages of one chunk overlap the GPU work of the other.
      const std::size_t first =
          remaining >= 2 * max_batch ? max_batch : (remaining + 1) / 2;
      const std::size_t second = std::min(max_batch, remaining - first);
      const bool is_last = first + second == remaining;
      next_token =
          ForwardPromptChunkPair(prompt_tokens.subspan(offset, first),
                                 prompt_tokens.subspan(offset + first, second),
                                 start_pos + static_cast<std::uint32_t>(offset),
                                 compute_logits && is_last);
      offset += first + second - max_batch;
      continue;
    }
    const std::size_t chunk_size = std::min<std::size_t>(
        arena_.GetMaxBatch(), prompt_tokens.size() - offset);
    const bool is_last = offset + chunk_size == prompt_tokens.size();
    next_token =
        ForwardPromptChunk(prompt_tokens.subspan(offset, chunk_size),
                           start_pos + static_cast<std::uint32_t>(offset),
                           compute_logits && is_last);
  }
  return next_token;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
