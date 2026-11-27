#include "src/models/raon1/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/dflash.hpp"

namespace gufo::models::raon1 {

namespace {

namespace qfn = qwen38_flash_next;

constexpr std::uint32_t kHistoryMinMatch = qfn::LookupIndex::kGram;
constexpr std::uint32_t kHistoryMinDraft = 2;
constexpr float kPlainBelow = 0.45F;
constexpr std::uint32_t kProbeEvery = 6;

void AssignError(std::string* error, std::string_view message) {
  if (error != nullptr)
    *error = message;
}

models::QwenTensorRef Ref(const rocm::DeviceTensor& t) {
  models::QwenTensorRef ref;
  ref.data = t.data;
  ref.type = t.type == 12  ? core::GgmlType::kQ4_K
             : t.type == 8 ? core::GgmlType::kQ8_0
                           : core::GgmlType::kQ6_K;
  ref.num_elements = static_cast<std::size_t>(t.cols) * t.rows;
  return ref;
}

qfn::MtpProposal MakeProposal(std::span<const sampling::TokenId> ids,
                              std::span<const float> probabilities,
                              sampling::TokenId token) {
  constexpr std::uint32_t kUnits = 1U << 24;
  qfn::MtpProposal proposal;
  if (ids.empty() || ids.size() > qfn::kMtpCandidates ||
      ids.size() != probabilities.size()) {
    return proposal;
  }
  double total = 0.0;
  std::size_t largest = 0;
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    if (!std::isfinite(probabilities[i]) || probabilities[i] < 0.0F) {
      return proposal;
    }
    total += probabilities[i];
    if (probabilities[i] > probabilities[largest])
      largest = i;
  }
  if (!(total > 0.0))
    return proposal;
  std::array<std::uint32_t, qfn::kMtpCandidates> mass{};
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    mass[i] = static_cast<std::uint32_t>(
        std::floor(probabilities[i] / total * kUnits));
    sum += mass[i];
  }
  mass[largest] += kUnits - sum;
  proposal.size = ids.size();
  double selected = 0.0;
  bool found = false;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    proposal.ids[i] = ids[i];
    proposal.probabilities[i] = static_cast<float>(mass[i]) / kUnits;
    if (ids[i] == token) {
      selected += static_cast<double>(proposal.probabilities[i]);
      found = true;
    }
  }
  if (!found)
    return qfn::MtpProposal{};
  proposal.token = token;
  proposal.probability = static_cast<float>(selected);
  return proposal;
}

qfn::MtpProposal PointMass(std::int32_t token) {
  qfn::MtpProposal proposal;
  proposal.size = 1;
  proposal.ids[0] = static_cast<sampling::TokenId>(token);
  proposal.probabilities[0] = 1.0F;
  proposal.token = proposal.ids[0];
  proposal.probability = 1.0F;
  return proposal;
}

}  // namespace

struct Session::DflashState {
  std::unique_ptr<hip::QwenDFlashGpuExecutor> executor;
};

Session::Session(std::shared_ptr<rocm::Executor> executor,
                 std::unique_ptr<rocm::Executor::Session> impl)
    : executor_(std::move(executor)), impl_(std::move(impl)) {}

Session::~Session() = default;

bool Session::InjectPart(const float* features, std::uint32_t position,
                         std::uint32_t rows, std::uint32_t end,
                         std::string* error) {
  if (!dflash_ || rows == 0)
    return true;
  auto& drafter = *dflash_->executor;
  end = std::max(end, prompt_end_);
  constexpr std::uint32_t tail = 1024;
  const std::uint32_t window = std::min(drafter.ContextWindowRows(), tail);
  if (position + rows + window < end) {
    if (!drafter.SkipTargetContext(position, rows)) {
      AssignError(error, "drafter context skip failed");
      return false;
    }
    return true;
  }
  hipStreamSynchronize(executor_->stream());
  if (!drafter.InjectTargetContextDevice(features, position, rows)) {
    AssignError(error, "drafter context injection failed");
    return false;
  }
  return true;
}

bool Session::DflashInject(std::uint32_t rows, std::string* error) {
  if (!dflash_ || rows == 0)
    return true;
  const float* features = executor_->DflashCapture();
  if (features == nullptr || executor_->DflashRows() < rows)
    return true;
  hipStreamSynchronize(executor_->stream());
  if (!dflash_->executor->InjectTargetContextDevice(
          features, executor_->DflashPosition(), rows)) {
    AssignError(error, "drafter context injection failed");
    return false;
  }
  return true;
}

void Session::SyncLookup() {
  if (lookup_.size() > tokens_.size())
    lookup_ = qfn::LookupIndex{};
  for (std::size_t i = lookup_.size(); i < tokens_.size(); ++i) {
    lookup_.Append(tokens_[i]);
  }
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error) {
  constexpr std::size_t kBatch = rocm::Executor::kMaxBatch;
  std::size_t last_rows = 1;
  for (std::size_t i = 0; i < tokens.size();) {
    const std::size_t remaining = tokens.size() - i;
    const int banks = executor_->PipelineBanks(remaining);
    const std::size_t total = std::min(remaining, banks * kBatch);
    if (executor_->CanPair(*impl_, static_cast<int>(total), banks)) {
      if (!executor_->EvaluatePipelined(*impl_, tokens.subspan(i, total), banks,
                                        error)) {
        return false;
      }
      for (std::size_t part = 0; part < executor_->PipelineParts(); ++part) {
        if (!InjectPart(
                executor_->PipelineCapture(part),
                executor_->PipelinePosition(part),
                executor_->PipelineRows(part),
                static_cast<std::uint32_t>(tokens_.size() + tokens.size()),
                error)) {
          return false;
        }
      }
      last_rows = total / banks;
      i += total;
      continue;
    }
    const auto chunk = tokens.subspan(i, std::min(kBatch, remaining));
    if (!executor_->EvaluateBatch(*impl_, chunk, error))
      return false;
    if (dflash_ &&
        !InjectPart(executor_->DflashCapture(), executor_->DflashPosition(),
                    static_cast<std::uint32_t>(chunk.size()),
                    static_cast<std::uint32_t>(tokens_.size() + tokens.size()),
                    error)) {
      return false;
    }
    last_rows = chunk.size();
    i += chunk.size();
  }
  hipStreamSynchronize(executor_->stream());
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  commit_row_ = static_cast<std::uint32_t>(
      std::min<std::size_t>(last_rows, rocm::Executor::kHeadRows) - 1);
  return true;
}

namespace {
struct SnapshotHeader {
  std::uint32_t position;
  std::uint32_t commit_row;
  std::uint64_t tokens;
};
}  // namespace

bool Session::SnapshotSupported() const noexcept {
  return executor_->SnapshotSupported();
}

std::size_t Session::SnapshotBytes() const {
  return sizeof(SnapshotHeader) + tokens_.size() * sizeof(std::int32_t) +
         executor_->config().vocab_size * sizeof(float) +
         executor_->SnapshotBytes(*impl_);
}

bool Session::SaveSnapshot(std::span<std::uint8_t> out, std::string* error) {
  if (out.size() != SnapshotBytes()) {
    AssignError(error, "snapshot buffer size mismatch");
    return false;
  }
  const SnapshotHeader header{impl_->position, commit_row_, tokens_.size()};
  std::memcpy(out.data(), &header, sizeof(header));
  std::memcpy(out.data() + sizeof(header), tokens_.data(),
              tokens_.size() * sizeof(std::int32_t));
  const std::size_t vocab = executor_->config().vocab_size;
  const std::size_t logits_at =
      sizeof(header) + tokens_.size() * sizeof(std::int32_t);
  std::vector<float> logits(vocab);
  if (!executor_->ReadLogitRows(commit_row_, 1, logits.data())) {
    AssignError(error, "snapshot logits readback failed");
    return false;
  }
  std::memcpy(out.data() + logits_at, logits.data(), vocab * sizeof(float));
  return executor_->SaveSnapshot(
      *impl_, out.subspan(logits_at + vocab * sizeof(float)), error);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> in,
                              std::string* error) {
  SnapshotHeader header{};
  if (in.size() < sizeof(header)) {
    AssignError(error, "snapshot too small");
    return false;
  }
  std::memcpy(&header, in.data(), sizeof(header));
  const std::size_t vocab = executor_->config().vocab_size;
  const std::size_t logits_at =
      sizeof(header) + header.tokens * sizeof(std::int32_t);
  const std::size_t head = logits_at + vocab * sizeof(float);
  if (header.tokens != header.position || in.size() < head) {
    AssignError(error, "snapshot header invalid");
    return false;
  }
  if (!executor_->RestoreSnapshot(*impl_, header.position, in.subspan(head),
                                  error)) {
    return false;
  }
  tokens_.resize(header.tokens);
  std::memcpy(tokens_.data(), in.data() + sizeof(header),
              header.tokens * sizeof(std::int32_t));
  commit_row_ = header.commit_row;
  std::vector<float> logits(vocab);
  std::memcpy(logits.data(), in.data() + logits_at, vocab * sizeof(float));
  if (!executor_->WriteLogitRow(commit_row_, logits.data())) {
    AssignError(error, "snapshot logits restore failed");
    return false;
  }
  prompt_end_ = 0;
  lookup_ = qfn::LookupIndex{};
  plain_steps_ = 0;
  accepted_ema_ = static_cast<float>(kMaxDraftTokens);
  if (dflash_) {
    dflash_->executor->Reset();
    if (header.position > 0 &&
        !dflash_->executor->SkipTargetContext(0, header.position)) {
      AssignError(error, "drafter context reset failed");
      return false;
    }
  }
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error) {
  return Feed(std::span<const std::int32_t>(&token, 1), error);
}

bool Session::Sync(std::span<const std::int32_t> tokens, std::string* error) {
  return Feed(tokens, error);
}

std::vector<float> Session::Logits(std::string* error) const {
  std::vector<float> logits(executor_->config().vocab_size);
  if (!executor_->ReadLogitRows(commit_row_, 1, logits.data())) {
    AssignError(error, "logit readback failed");
    return {};
  }
  return logits;
}

std::int32_t Session::Argmax(std::string*) const {
  return executor_->GreedyRows(commit_row_ + 1)[commit_row_];
}

std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

void Session::Reset() noexcept {
  executor_->ResetSession(*impl_);
  tokens_.clear();
  lookup_ = qfn::LookupIndex{};
  commit_row_ = 0;
  prompt_end_ = 0;
  accepted_ema_ = static_cast<float>(kMaxDraftTokens);
  plain_steps_ = 0;
  if (dflash_)
    dflash_->executor->Reset();
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         const StopFn& is_stop, std::string* error) {
  auto& exec = *executor_;
  *result = {};
  if (max_tokens == 0 || tokens_.empty()) {
    AssignError(error, "decode needs a positive budget and a synced prompt");
    return false;
  }
  const std::size_t room = model_->options_.max_context - tokens_.size();
  const std::size_t cap = std::min<std::size_t>(
      {max_tokens, room, model_->options_.max_draft_tokens + 1});
  if (cap == 0) {
    result->stop = true;
    return true;
  }
  const auto t_entry = std::chrono::steady_clock::now();
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_greedy =
      !sampled && sampler.config().can_use_unmodified_argmax();
  std::int32_t anchor;
  if (gpu_greedy) {
    anchor = Argmax();
  } else {
    const auto logits = Logits(error);
    if (logits.empty())
      return false;
    anchor = static_cast<std::int32_t>(sampler.Sample(logits));
  }
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  static const bool timing = std::getenv("GUFO_RAON1_SPEC_TIME") != nullptr;
  const auto t_start = std::chrono::steady_clock::now();
  auto lap = [&]() {
    hipStreamSynchronize(executor_->stream());
    return std::chrono::steady_clock::now();
  };
  const auto base = static_cast<std::uint32_t>(tokens_.size());
  std::vector<std::int32_t> chain{anchor};
  std::vector<qfn::MtpProposal> proposals;
  bool history = false;
  bool drafted_by_model = false;

  if (cap > 1 && dflash_) {
    SyncLookup();
    lookup_.Append(anchor);
    const auto draft = lookup_.Draft(
        kHistoryMinMatch,
        std::min<std::size_t>(cap - 1, model_->options_.max_draft_tokens));
    if (draft.size() >= kHistoryMinDraft) {
      history = true;
      for (const auto token : draft) {
        chain.push_back(token);
        if (sampled)
          proposals.push_back(PointMass(token));
      }
    } else if (dflash_->executor->GetInjectedContextLength() == base &&
               (accepted_ema_ >= kPlainBelow ||
                ++plain_steps_ % kProbeEvery == 0)) {
      const auto wanted = std::clamp<std::uint32_t>(
          static_cast<std::uint32_t>(std::ceil(accepted_ema_)) + 1, 2,
          static_cast<std::uint32_t>(cap - 1));
      const auto selector =
          model_->dflash_model_->GetDFlashConfig().selector_top_k;
      std::uint64_t draft_rng = 0;
      std::vector<float> uniforms;
      std::vector<sampling::TokenId> candidate_ids;
      std::vector<float> candidate_probabilities;
      std::vector<tokenization::TokenId> drafted;
      if (sampled) {
        draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
        for (std::uint32_t row = 0; row < wanted; ++row) {
          uniforms.push_back(sampling::Uniform(&draft_rng));
        }
      }
      try {
        drafted = dflash_->executor->ForwardBlock(
            anchor, base, wanted, sampled ? sampler.config().temperature : 0.0F,
            uniforms, nullptr, sampled ? &candidate_ids : nullptr,
            sampled ? &candidate_probabilities : nullptr);
      } catch (const std::exception&) {
        drafted.clear();
      }
      bool complete = !drafted.empty() && (!sampled || selector != 0);
      std::vector<qfn::MtpProposal> block;
      if (complete && sampled) {
        complete = drafted.size() * selector == candidate_ids.size() &&
                   candidate_ids.size() == candidate_probabilities.size();
        for (std::size_t row = 0; complete && row < drafted.size(); ++row) {
          auto proposal = MakeProposal(
              std::span(candidate_ids).subspan(row * selector, selector),
              std::span(candidate_probabilities)
                  .subspan(row * selector, selector),
              static_cast<sampling::TokenId>(drafted[row]));
          complete =
              proposal.size != 0 &&
              proposal.token == static_cast<sampling::TokenId>(drafted[row]);
          if (complete)
            block.push_back(proposal);
        }
      }
      if (complete) {
        for (const auto token : drafted) {
          chain.push_back(static_cast<std::int32_t>(token));
        }
        proposals = std::move(block);
        drafted_by_model = true;
      }
    }
  }

  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = executor_->config().vocab_size;
  const bool speculative = k > 1;
  const auto t_drafted = timing ? lap() : t_start;
  if (speculative) {
    exec.SetRecording(true);
    if (!exec.SaveState(*impl_, error)) {
      exec.SetRecording(false);
      return false;
    }
  }
  const bool forwarded = exec.EvaluateBatch(*impl_, chain, error);
  exec.SetRecording(false);
  if (!forwarded)
    return false;
  const auto t_verified = timing ? lap() : t_start;

  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  if (speculative) {
    std::vector<std::int32_t> greedy;
    if (gpu_greedy) {
      greedy = exec.GreedyRows(k);
    } else {
      verify_logits_.resize(static_cast<std::size_t>(k) * vocab);
      if (!exec.ReadLogitRows(0, k, verify_logits_.data())) {
        AssignError(error, "verification readback failed");
        return false;
      }
    }
    while (keep < k) {
      if (gpu_greedy) {
        const std::int32_t prediction = greedy[keep - 1];
        if (is_stop(prediction)) {
          result->stop = true;
          break;
        }
        if (prediction != chain[keep])
          break;
        sampler.Accept(static_cast<sampling::TokenId>(prediction));
        ++keep;
        continue;
      }
      const auto row =
          std::span<const float>(verify_logits_)
              .subspan(static_cast<std::size_t>(keep - 1) * vocab, vocab);
      if (sampled) {
        const auto verified =
            qfn::VerifyMtpProposal(row, proposals[keep - 1], sampler);
        const auto token = static_cast<std::int32_t>(verified.token);
        if (is_stop(token)) {
          result->stop = true;
          break;
        }
        if (!verified.accepted) {
          correction = token;
          break;
        }
        sampler.Accept(static_cast<sampling::TokenId>(token));
        ++keep;
        continue;
      }
      const auto decision =
          qfn::VerifyDraft(row, chain[keep], sampler, is_stop);
      if (decision != qfn::DraftDecision::kAccept) {
        result->stop = decision == qfn::DraftDecision::kStop;
        break;
      }
      ++keep;
    }
    if (keep < k && !exec.Rollback(*impl_, keep, error))
      return false;
  }
  commit_row_ = speculative ? keep - 1 : 0;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);

  stats_.cycles += 1;
  if (speculative) {
    stats_.drafted += k - 1;
    stats_.accepted += keep - 1;
    if (history) {
      stats_.history_drafted += k - 1;
      stats_.history_accepted += keep - 1;
    }
    if (drafted_by_model) {
      accepted_ema_ =
          0.7F * accepted_ema_ + 0.3F * static_cast<float>(keep - 1);
    }
  }
  if (!DflashInject(keep, error))
    return false;
  if (timing) {
    const auto t_end = lap();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(
        stderr, "SPEC k=%u keep=%u draft=%.2f verify=%.2f accept+inject=%.2f\n",
        k, keep, ms(t_start, t_drafted), ms(t_drafted, t_verified),
        ms(t_verified, t_end));
    std::fprintf(stderr, "ANCHOR %.2f\n", ms(t_entry, t_start));
  }
  if (correction) {
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& path,
                                   const EngineOptions& options,
                                   std::string* error) {
  rocm::ExecutorOptions executor_options;
  executor_options.layers = options.layers;
  if (const char* layers = std::getenv("GUFO_RAON1_GPU_LAYERS")) {
    executor_options.layers = static_cast<std::uint32_t>(std::atoi(layers));
  }
  executor_options.max_context = options.max_context;
  if (!options.dflash_model_path.empty()) {
    auto reader = core::GgufReader::OpenFile(options.dflash_model_path, error);
    if (!reader)
      return nullptr;
    auto weights = speculative::QwenDFlashWeights::LoadFromGguf(*reader, error);
    if (!weights)
      return nullptr;
    const auto& df = weights->dflash_config;
    if (df.block_size < 2 || options.max_draft_tokens + 1 > df.block_size) {
      AssignError(error, "draft limit exceeds the drafter's block size");
      return nullptr;
    }
    executor_options.dflash_layers = df.target_layer_ids;
  }
  auto executor = rocm::Executor::Create(path, executor_options, error);
  if (!executor)
    return nullptr;
  auto model = std::shared_ptr<Model>(new Model());
  model->options_ = options;
  model->executor_ = std::move(executor);
  model->resident_bytes_ = model->executor_->ResidentBytes();
  if (!options.dflash_model_path.empty()) {
    const auto* embedding = model->executor_->TokenEmbedding();
    const auto* output = model->executor_->Output();
    if (embedding == nullptr || output == nullptr) {
      AssignError(error, "drafter needs the embedding and output tensors");
      return nullptr;
    }
    std::shared_ptr<const core::GgufReader> reader =
        core::GgufReader::OpenFile(options.dflash_model_path, error);
    if (!reader)
      return nullptr;
    const auto& config = model->executor_->config();
    model->dflash_model_ = hip::QwenDFlashGpuModel::Create(
        std::move(reader), config.hidden_size, config.vocab_size,
        config.num_layers, Ref(*embedding), Ref(*output), error);
    if (!model->dflash_model_)
      return nullptr;
  }
  return model;
}

bool Model::SnapshotSupported() const noexcept {
  return executor_->SnapshotSupported();
}

std::unique_ptr<Session> Model::CreateSession(std::string* error,
                                              bool speculative) {
  auto impl = executor_->CreateSession(error);
  if (!impl)
    return nullptr;
  auto session =
      std::unique_ptr<Session>(new Session(executor_, std::move(impl)));
  session->model_ = shared_from_this();
  if (speculative && dflash_model_ != nullptr) {
    auto drafter = hip::QwenDFlashGpuExecutor::Create(
        dflash_model_, options_.max_context, error);
    if (!drafter)
      return nullptr;
    session->dflash_ = std::make_unique<Session::DflashState>();
    session->dflash_->executor = std::move(drafter);
  }
  return session;
}

const Config& Model::config() const noexcept {
  return executor_->config();
}

std::uint32_t Model::layers() const noexcept {
  return executor_->layers();
}

std::uint32_t Model::VocabSize() const noexcept {
  return executor_->config().vocab_size;
}

std::uint32_t Model::MaxContext() const noexcept {
  return executor_->config().context_length;
}

std::size_t Model::ResidentBytes() const noexcept {
  return resident_bytes_;
}

std::string Model::ModelName() const {
  return "KRAON Raon-1 mini";
}

}  // namespace gufo::models::raon1
