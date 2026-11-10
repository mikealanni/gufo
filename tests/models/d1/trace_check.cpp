#include <iostream>

#include "src/core/gguf_reader.hpp"
#include "src/models/d1/judge.hpp"
#include "src/models/lfm3/decision_forward.hpp"

// Usage: d1_trace_check MODEL.gguf QUESTION CONTEXT
// One yes/no decision on the FP32 host oracle (use GUFO_D1_TRACE=1).
int main(int argc, char** argv) {
  using namespace gufo;
  if (argc < 4) return 2;
  std::string error;
  auto reader = core::GgufReader::OpenFile(argv[1], &error);
  auto weights = reader ? models::lfm3::Weights::Bind(*reader, &error) : std::nullopt;
  if (!weights) { std::cerr << error << '\n'; return 1; }
  auto judge = models::d1::Judge::Create(
      *reader,
      [&](auto tokens, auto groups, auto sequences, auto slots, auto vocabulary, auto logits) {
        // Oracle: run every packed sequence on its own, then concatenate.
        const std::size_t seq_count = sequences.empty() ? 1 : sequences.size();
        std::size_t slot_at = 0;
        for (std::size_t q = 0; q < seq_count; ++q) {
          const std::size_t lo = sequences.empty() ? 0 : sequences[q];
          const std::size_t hi = (!sequences.empty() && q + 1 < seq_count) ? sequences[q + 1] : tokens.size();
          std::vector<std::uint32_t> own_slots;
          while (slot_at + own_slots.size() < slots.size() && slots[slot_at + own_slots.size()] < hi)
            own_slots.push_back(slots[slot_at + own_slots.size()] - lo);
          const auto out = models::lfm3::DecisionForward(
              *weights, tokens.subspan(lo, hi - lo),
              groups.empty() ? groups : groups.subspan(lo, hi - lo), own_slots, vocabulary);
          std::copy(out.begin(), out.end(), logits.begin() + slot_at * vocabulary.size());
          slot_at += own_slots.size();
        }
        return true;
      }, 512, &error);
  const std::pair<std::string_view, std::string_view> item{argv[2], argv[3]};
  const auto v = judge->Answer(std::span(&item, 1));
  std::cout << "yes=" << v[0].yes << " no=" << v[0].no << '\n';
}
