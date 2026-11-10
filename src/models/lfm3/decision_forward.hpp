#ifndef GUFO_MODELS_LFM3_DECISION_FORWARD_HPP_
#define GUFO_MODELS_LFM3_DECISION_FORWARD_HPP_
#include <cstdint>
#include <span>
#include <vector>

#include "src/models/lfm3/weights.hpp"
namespace gufo::models::lfm3 {
// FP32 host oracle for the d1 decision pass: one prefill over the whole
// prompt, global attention bidirectional (optionally masked by question
// groups), sliding attention and convolutions causal. Returns one row of
// logits per slot over `vocabulary` (all entries when empty). With `causal`
// the global layers become causal, which must reproduce Reference::Step.
std::vector<float> DecisionForward(const Weights& weights,
                                   std::span<const std::uint32_t> tokens,
                                   std::span<const std::int32_t> groups,
                                   std::span<const std::uint32_t> slots,
                                   std::span<const std::uint32_t> vocabulary,
                                   bool causal = false,
                                   unsigned threads = 4,
                                   std::vector<float>* final_hidden = nullptr);
}  // namespace gufo::models::lfm3
#endif
