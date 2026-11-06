#include "src/core/sampling.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <span>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"

namespace gufo::hip {

kernels::DFlashSampling QwenDFlashGpuExecutor::PrepareDraftSampling(
    const sampling::SamplerState* sampler, hipStream_t stream) {
  kernels::DFlashSampling result;
  if (sampler == nullptr)
    return result;
  const auto& config = sampler->config();
  // The selector already restricts q to a small lattice. Filtering that
  // lattice a second time lost useful coverage on short sampled completions.
  result.repeat_penalty = config.repeat_penalty;
  result.frequency_penalty = config.frequency_penalty;
  result.presence_penalty = config.presence_penalty;
  result.repeat_last_n =
      std::min<std::size_t>(config.repeat_last_n, max_context_);
  const auto penalties =
      config.frequency_penalty != 0.0F || config.presence_penalty != 0.0F
          ? sampler->penalties()
          : std::span<const sampling::TokenPenalty>{};
  const auto history = config.repeat_penalty != 1.0F
                           ? sampler->history()
                           : std::span<const tokenization::TokenId>{};
  const auto bytes = penalties.size_bytes() + history.size_bytes();
  const auto capacity = std::size_t{injection_capacity_} *
                        GetTargetFeaturesSize() * sizeof(float);
  // Context projection has finished; reuse its input scratch for this proposal.
  // An unusually large policy can always retain the original valid q.
  if (bytes > capacity)
    return {};
  if (!penalties.empty()) {
    HIP_CHECK(hipMemcpyAsync(d_target_features_, penalties.data(),
                             penalties.size_bytes(), hipMemcpyHostToDevice,
                             stream));
    result.penalties =
        reinterpret_cast<const sampling::TokenPenalty*>(d_target_features_);
    result.penalty_count = penalties.size();
  }
  if (!history.empty()) {
    auto* destination = reinterpret_cast<std::byte*>(d_target_features_) +
                        penalties.size_bytes();
    HIP_CHECK(hipMemcpyAsync(destination, history.data(), history.size_bytes(),
                             hipMemcpyHostToDevice, stream));
    result.history =
        reinterpret_cast<const tokenization::TokenId*>(destination);
    result.history_count = history.size();
  }
  return result;
}

}  // namespace gufo::hip
