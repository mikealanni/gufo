#include <chrono>
#include <iostream>

#include "src/core/gguf_reader.hpp"
#include "src/models/d1/judge.hpp"
#include "src/models/lfm3/decision_forward.hpp"

// Usage: d1_judge_check MODEL.gguf
// Runs the yes/no judge on the FP32 host oracle: shared-context packing,
// distinct contexts and an oversized state.
int main(int argc, char** argv) {
  using namespace gufo;
  if (argc < 2) return 2;
  std::string error;
  auto reader = core::GgufReader::OpenFile(argv[1], &error);
  auto weights = reader ? models::lfm3::Weights::Bind(*reader, &error) : std::nullopt;
  if (!weights) {
    std::cerr << error << '\n';
    return 1;
  }
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
      },
      512, &error);
  if (!judge) {
    std::cerr << error << '\n';
    return 1;
  }
  const std::string train =
      "The agent is about to run: python train.py --epochs 50 on a 20GB model, "
      "using the GPU and most of RAM.";
  const std::string listing = "The agent is about to run: ls -la /tmp";
  const std::string big(6000, 'x');
  const std::vector<std::pair<std::string_view, std::string_view>> items = {
      {"Will this command use the GPU or most of the machine's RAM?", train},
      {"Is this a quick read-only shell command?", train},
      {"Will this command use the GPU or most of the machine's RAM?", listing},
      {"Is this a quick read-only shell command?", listing},
      {"Is the text a long run of the letter x?", big}};
  const auto start = std::chrono::steady_clock::now();
  const auto verdicts = judge->Answer(items);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
  if (verdicts.size() != items.size()) {
    std::cerr << "judge failed\n";
    return 1;
  }
  for (std::size_t i = 0; i < items.size(); ++i)
    std::cout << i << " yes=" << verdicts[i].yes << " no=" << verdicts[i].no
              << "  " << items[i].first.substr(0, 60) << '\n';
  std::cout << "ms " << ms << '\n';
  return 0;
}
