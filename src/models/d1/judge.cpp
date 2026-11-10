#include "src/models/d1/judge.hpp"

#include <algorithm>
#include <map>
#include <unordered_map>

#include "src/models/lfm3/weights.hpp"
#include "src/models/qwen/control_tokens.hpp"

namespace gufo::models::d1 {
namespace {
constexpr std::size_t kSetSize = 16;
constexpr int kMaxShrinkRounds = 8;
}  // namespace

struct Judge::Impl {
  std::unique_ptr<lfm3::Tokenizer> tokenizer;
  lfm3::Config config;
  float temperature{1.0F};
  ForwardFn forward;
  std::uint32_t max_tokens{512};
  mutable std::mutex mutex;
};

Judge::Judge() : impl_(std::make_unique<Impl>()) {}
Judge::~Judge() = default;

std::unique_ptr<Judge> Judge::Create(const core::GgufReader& reader,
                                     ForwardFn forward,
                                     std::uint32_t max_tokens,
                                     std::string* error) {
  std::string err;
  auto config = lfm3::Config::FromGguf(reader, &err);
  if (!config || !config->decision) {
    if (error)
      *error = "d1 judge: not a d1 decision GGUF" +
               (err.empty() ? std::string() : ": " + err);
    return nullptr;
  }
  auto tokenizer = lfm3::Tokenizer::Load(reader, &err);
  if (!tokenizer) {
    if (error)
      *error = "d1 judge: " + err;
    return nullptr;
  }
  std::unique_ptr<Judge> judge(new Judge);
  auto& m = *judge->impl_;
  m.config = *config;
  m.tokenizer = std::move(tokenizer);
  m.forward = std::move(forward);
  m.max_tokens = max_tokens;
  m.temperature = reader.GetMetadataFloat32("lfm3.decision.temperature.noul")
                      .value_or(1.0F);
  return judge;
}

namespace {
struct Sequence {
  std::vector<std::uint32_t> tokens, slots;
  std::vector<std::int32_t> groups;
  std::vector<std::size_t> members;  // item index per question
  std::vector<Question> questions;   // readouts remapped to vocabulary columns
};
}  // namespace

std::vector<JudgeVerdict> Judge::Answer(
    std::span<const std::pair<std::string_view, std::string_view>> items)
    const {
  auto& m = *impl_;
  if (items.empty())
    return {};
  std::lock_guard lock(m.mutex);
  std::vector<JudgeVerdict> verdicts(items.size());
  const auto plain = [&](std::string_view s) {
    return Plain(*m.tokenizer, s, m.config.vocab);
  };
  // The yes/no readout forms are identical for every noul question, so one
  // shared vocabulary serves all sequences of the call.
  std::vector<std::uint32_t> vocabulary;
  std::unordered_map<std::uint32_t, std::uint32_t> columns;
  const auto remap = [&](Question& q) {
    for (auto& forms : q.readout)
      for (auto& id : forms) {
        auto [at, added] = columns.emplace(id, vocabulary.size());
        if (added)
          vocabulary.push_back(id);
        id = at->second;
      }
  };

  // Same context -> same state -> one sequence of up to kSetSize questions.
  std::map<std::string, std::vector<std::size_t>> by_context;
  for (std::size_t i = 0; i < items.size(); ++i)
    by_context[std::string(items[i].second)].push_back(i);
  std::vector<Sequence> sequences;
  for (const auto& [context_text, members] : by_context) {
    for (std::size_t first = 0; first < members.size(); first += kSetSize) {
      const auto count = std::min(kSetSize, members.size() - first);
      std::string context = context_text;
      bool done = false;
      for (int round = 0; round <= kMaxShrinkRounds && !done; ++round) {
        Sequence seq;
        std::vector<std::string> suffixes;
        for (std::size_t j = 0; j < count; ++j) {
          auto spec = json::Value::object();
          spec["type"] = "noul";
          spec["instructions"] =
              std::string(items[members[first + j]].first);
          auto q = ParseQuestion(*m.tokenizer, "q" + std::to_string(j), spec);
          q.temperature = m.temperature;
          suffixes.push_back(Suffix(*m.tokenizer, q, plain(q.body)));
          seq.questions.push_back(std::move(q));
          seq.members.push_back(members[first + j]);
        }
        std::string prefix(m.tokenizer->Piece(m.tokenizer->Bos()));
        prefix += tokenization::kImStart;
        prefix += "user\n";
        std::string header;
        if (!context.empty()) {
          prefix += plain(context) + "\n\n\nQUESTION:\n";
          header = "QUESTION:\n";
        }
        auto prompt =
            Pack(*m.tokenizer, prefix, header, suffixes, seq.questions);
        if (prompt.tokens.size() > m.max_tokens) {
          // Keep the end of an oversized state; the questions stay whole.
          if (context.empty() || round == kMaxShrinkRounds)
            return {};
          const auto over = prompt.tokens.size() - m.max_tokens;
          const auto keep = context.size() > over * 4 + 64
                                ? context.size() - over * 4 - 32
                                : 0;
          context = context.substr(context.size() - keep);
          continue;
        }
        for (auto& q : seq.questions)
          remap(q);
        seq.tokens = std::move(prompt.tokens);
        seq.slots = std::move(prompt.slots);
        seq.groups = count > 1 ? std::move(prompt.groups)
                               : std::vector<std::int32_t>(seq.tokens.size(), -1);
        sequences.push_back(std::move(seq));
        done = true;
      }
    }
  }

  // First-fit in order: consecutive sequences share a pass while they fit.
  for (std::size_t first = 0; first < sequences.size();) {
    std::size_t last = first, total = 0;
    while (last < sequences.size() &&
           total + sequences[last].tokens.size() <= m.max_tokens)
      total += sequences[last++].tokens.size();
    if (last == first)
      return {};
    std::vector<std::uint32_t> tokens, starts, slots;
    std::vector<std::int32_t> groups;
    for (std::size_t i = first; i < last; ++i) {
      const auto& seq = sequences[i];
      starts.push_back(static_cast<std::uint32_t>(tokens.size()));
      for (auto slot : seq.slots)
        slots.push_back(static_cast<std::uint32_t>(tokens.size()) + slot);
      tokens.insert(tokens.end(), seq.tokens.begin(), seq.tokens.end());
      groups.insert(groups.end(), seq.groups.begin(), seq.groups.end());
    }
    // A lone question in a lone sequence keeps the unmasked specialisation.
    const bool lone = last - first == 1 && sequences[first].questions.size() == 1;
    std::vector<float> logits(slots.size() * vocabulary.size());
    if (!m.forward(tokens, lone ? std::span<const std::int32_t>() : groups,
                   starts, slots, vocabulary, logits))
      return {};
    std::size_t slot = 0;
    for (std::size_t i = first; i < last; ++i) {
      const auto& seq = sequences[i];
      for (std::size_t j = 0; j < seq.questions.size(); ++j, ++slot) {
        const auto scores = Probabilities(
            seq.questions[j].readout,
            std::span<const float>(logits).subspan(slot * vocabulary.size(),
                                                   vocabulary.size()),
            seq.questions[j].temperature);
        verdicts[seq.members[j]] = {static_cast<float>(scores[0]),
                                    static_cast<float>(scores[1])};
      }
    }
    first = last;
  }
  return verdicts;
}
}  // namespace gufo::models::d1
