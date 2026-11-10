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

std::vector<JudgeVerdict> Judge::Answer(
    std::span<const std::pair<std::string_view, std::string_view>> items)
    const {
  auto& m = *impl_;
  if (items.empty())
    return {};
  std::lock_guard lock(m.mutex);
  std::vector<JudgeVerdict> verdicts(items.size());
  // Same context -> same state -> one packed pass (up to kSetSize questions).
  std::map<std::string, std::vector<std::size_t>> by_context;
  for (std::size_t i = 0; i < items.size(); ++i)
    by_context[std::string(items[i].second)].push_back(i);
  const auto plain = [&](std::string_view s) {
    return Plain(*m.tokenizer, s, m.config.vocab);
  };
  for (const auto& [context_text, members] : by_context) {
    for (std::size_t first = 0; first < members.size(); first += kSetSize) {
      const auto count = std::min(kSetSize, members.size() - first);
      std::string context = context_text;
      bool done = false;
      for (int round = 0; round <= kMaxShrinkRounds && !done; ++round) {
        std::vector<Question> questions;
        std::vector<std::string> suffixes;
        for (std::size_t j = 0; j < count; ++j) {
          auto spec = json::Value::object();
          spec["type"] = "noul";
          spec["instructions"] =
              std::string(items[members[first + j]].first);
          auto q = ParseQuestion(*m.tokenizer, "q" + std::to_string(j), spec);
          q.temperature = m.temperature;
          suffixes.push_back(Suffix(*m.tokenizer, q, plain(q.body)));
          questions.push_back(std::move(q));
        }
        std::string prefix(m.tokenizer->Piece(m.tokenizer->Bos()));
        prefix += tokenization::kImStart;
        prefix += "user\n";
        std::string header;
        if (!context.empty()) {
          prefix += plain(context) + "\n\n\nQUESTION:\n";
          header = "QUESTION:\n";
        }
        std::vector<std::uint32_t> vocabulary;
        std::unordered_map<std::uint32_t, std::uint32_t> columns;
        for (auto& q : questions)
          for (auto& forms : q.readout)
            for (auto& id : forms) {
              auto [at, added] = columns.emplace(id, vocabulary.size());
              if (added)
                vocabulary.push_back(id);
              id = at->second;
            }
        const auto prompt =
            Pack(*m.tokenizer, prefix, header, suffixes, questions);
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
        const auto groups = count > 1
                                ? std::span<const std::int32_t>(prompt.groups)
                                : std::span<const std::int32_t>();
        std::vector<float> logits(prompt.slots.size() * vocabulary.size());
        if (!m.forward(prompt.tokens, groups, prompt.slots, vocabulary,
                       logits))
          return {};
        for (std::size_t j = 0; j < count; ++j) {
          const auto scores = Probabilities(
              questions[j].readout,
              std::span<const float>(logits).subspan(j * vocabulary.size(),
                                                     vocabulary.size()),
              questions[j].temperature);
          verdicts[members[first + j]] = {static_cast<float>(scores[0]),
                                          static_cast<float>(scores[1])};
        }
        done = true;
      }
    }
  }
  return verdicts;
}
}  // namespace gufo::models::d1
