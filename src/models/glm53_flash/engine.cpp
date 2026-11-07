#include "src/models/glm53_flash/engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::glm53_flash {

namespace {

namespace qfn = qwen38_flash_next;

// Two 4096-row banks of the NPU pair path; larger chunks were slower.
constexpr std::uint32_t kPrefillChunk = 4096;
// A history draft needs this many matching tokens; shorter matches are noise.
constexpr std::uint32_t kHistoryMinMatch = qfn::LookupIndex::kGram;
constexpr std::uint32_t kHistoryMinDraft = 2;
// Below this running acceptance the drafter costs more than it returns, so
// ordinary steps run and the drafter is probed on every sixth.
constexpr float kPlainBelow = 0.8F;
constexpr std::uint32_t kProbeEvery = 6;

constexpr std::array<char, 8> kSnapshotMagic{'G', 'L', 'M', 'S',
                                             'E', 'S', 'S', '1'};

struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t reserved;
  std::uint64_t executor_bytes;
};

std::uint64_t HostBytes(std::uint32_t tokens, std::uint32_t vocab) {
  return sizeof(SnapshotHeader) +
         static_cast<std::uint64_t>(tokens) * sizeof(std::int32_t) +
         static_cast<std::uint64_t>(vocab) * sizeof(float);
}

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

models::QwenTensorRef Ref(const rocm::DeviceTensor& t) {
  models::QwenTensorRef ref;
  ref.data = t.data;
  ref.type = core::GgmlType::kQ6_K;
  ref.num_elements = static_cast<std::size_t>(t.k) * t.rows;
  return ref;
}

/// Fits the drafter's top-k distribution into the compact proposal form used
/// by verification: integer masses sum to 2^24, so the F32 q sums to exactly
/// one and matches the distribution the block drew from.
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
    if (probabilities[i] > probabilities[largest]) {
      largest = i;
    }
  }
  if (!(total > 0.0)) {
    return proposal;
  }
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
  if (!found) {
    return qfn::MtpProposal{};
  }
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

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.max_context < 2) {
    AssignError(error_msg, "context must hold at least two tokens");
    return nullptr;
  }
  if (options.max_draft_tokens == 0 ||
      options.max_draft_tokens > kMaxDraftTokens) {
    AssignError(error_msg, "draft token limit must be between one and seven");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  const auto id = [&](const char* key) -> std::optional<std::int32_t> {
    const auto value = m->reader_->GetMetadataUint32(key);
    if (!value) {
      return std::nullopt;
    }
    return static_cast<std::int32_t>(*value);
  };
  const auto eos = id("tokenizer.ggml.eos_token_id");
  if (!eos) {
    AssignError(error_msg, "GGUF has no end-of-sequence token");
    return nullptr;
  }
  m->eos_token_ = *eos;
  m->stop_tokens_ = {*eos};
  for (const char* key :
       {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
    if (const auto token = id(key)) {
      m->stop_tokens_.push_back(*token);
    }
  }
  for (const auto& tensor : m->reader_->GetTensors()) {
    m->resident_bytes_ += static_cast<std::size_t>(tensor.size_bytes);
  }

  rocm::ExecutorOptions exec;
  exec.max_context = options.max_context;
  exec.prefill_chunk = kPrefillChunk;
  exec.logit_rows = options.max_draft_tokens + 1;
  if (!options.dflash_model_path.empty()) {
    DrafterInfo info;
    if (!ReadDrafterInfo(options.dflash_model_path, &info, error_msg)) {
      return nullptr;
    }
    if (info.block_size < 2 || options.max_draft_tokens + 1 > info.block_size) {
      AssignError(error_msg, "draft limit exceeds the drafter's block size");
      return nullptr;
    }
    exec.dflash_layers = info.target_layers;
    m->dflash_window_ = std::min(info.sliding_window, options.max_context);
    exec.dflash_rows = m->dflash_window_;
  }
  m->executor_ = rocm::Executor::Create(model_path, exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  if (!options.dflash_model_path.empty()) {
    const auto* embedding = m->executor_->TokenEmbedding();
    const auto* output = m->executor_->Output();
    if (embedding == nullptr || output == nullptr ||
        embedding->type != rocm::kTypeQ6_K || output->type != rocm::kTypeQ6_K) {
      AssignError(error_msg, "drafter needs Q6_K embedding and output tensors");
      return nullptr;
    }
    std::shared_ptr<const core::GgufReader> dflash_reader =
        core::GgufReader::OpenFile(options.dflash_model_path, error_msg);
    if (!dflash_reader) {
      return nullptr;
    }
    const auto& config = m->executor_->config();
    m->dflash_model_ = hip::QwenDFlashGpuModel::Create(
        std::move(dflash_reader), config.hidden_size, config.vocab_size,
        config.TrunkLayers(), Ref(*embedding), Ref(*output), error_msg);
    if (!m->dflash_model_) {
      return nullptr;
    }
  }
  return m;
}

bool ReadDrafterInfo(const std::string& path, DrafterInfo* info,
                     std::string* error) {
  auto reader = core::GgufReader::OpenFile(path, error);
  if (!reader) {
    return false;
  }
  auto weights = speculative::QwenDFlashWeights::LoadFromGguf(*reader, error);
  if (!weights) {
    return false;
  }
  info->target_layers = weights->dflash_config.target_layer_ids;
  info->block_size = weights->dflash_config.block_size;
  info->sliding_window = weights->dflash_config.sliding_window;
  return true;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(error_msg);
  if (!native) {
    return nullptr;
  }
  auto session = std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
  session->speculative_ = mode == core::SessionMode::kSpeculative;
  if (session->speculative_ && dflash_model_ != nullptr &&
      !session->InitDflash(error_msg)) {
    return nullptr;
  }
  return session;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  tokenization::TokenizerOptions options;
  options.parse_special_tokens = true;
  std::vector<std::int32_t> out;
  for (auto token : tokenizer_->Encode(text, options)) {
    out.push_back(static_cast<std::int32_t>(token));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
         stop_tokens_.end();
}

std::uint32_t Model::VocabSize() const noexcept {
  return executor_->config().vocab_size;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(2 * executor_->PrefillChunk(), options_.max_context);
}

std::string Model::ModelName() const {
  return std::string(
      reader_->GetMetadataString("general.name").value_or("GLM-5.3-Flash"));
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  const auto& c = executor_->config();
  const std::size_t kda_layers = c.NumKdaLayers();
  const std::size_t mla_layers = c.NumMlaLayers();
  constexpr std::size_t kKdaBytes =
      (3 * 3 * 8192 + 64 * 128 * 128) * sizeof(float);
  // Latent rows (F16), pooled indexer keys and the ring of unpooled rows.
  const std::size_t per_mla =
      static_cast<std::size_t>(context) * 512 * 2 +
      (static_cast<std::size_t>(context) / 4 + 1) * 128 * sizeof(float) +
      2 * static_cast<std::size_t>(kPrefillChunk + 4) * 128 * sizeof(float);
  // Rollback copies of the recurrent state plus the recorded KDA operands.
  const std::size_t rollback =
      kda_layers * (kKdaBytes + (options_.max_draft_tokens + 1) *
                                    (5 * 8192 + 64) * sizeof(float));
  std::size_t total = kda_layers * kKdaBytes + mla_layers * per_mla + rollback;
  if (mode == core::SessionMode::kSpeculative && dflash_model_ != nullptr) {
    // Five layers of K and V over the sliding window plus the injection
    // scratch; measured allocations are an order of magnitude below this.
    total += static_cast<std::size_t>(dflash_window_) * 5 * 2 * 1024 *
             sizeof(float) * 2;
  }
  return total;
}

struct Session::DflashState {
  std::unique_ptr<hip::QwenDFlashGpuExecutor> executor;
};

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Executor::Session> session)
    : model_(std::move(model)), session_(std::move(session)) {
  logits_.resize(model_->VocabSize());
}

Session::~Session() = default;

bool Session::InitDflash(std::string* error_msg) {
  auto executor = hip::QwenDFlashGpuExecutor::Create(
      model_->dflash_model_, model_->options_.max_context, error_msg);
  if (!executor) {
    return false;
  }
  dflash_ = std::make_unique<DflashState>();
  dflash_->executor = std::move(executor);
  return true;
}

bool Session::DflashReady() const noexcept {
  return dflash_ && dflash_->executor &&
         dflash_->executor->GetInjectedContextLength() ==
             static_cast<std::uint32_t>(tokens_.size());
}

bool Session::DflashInject(std::uint32_t rows, std::string* error_msg) {
  if (!dflash_ || rows == 0) {
    return true;
  }
  auto& exec = *model_->executor_;
  const float* features = exec.DflashCapture();
  if (features == nullptr || exec.DflashRows() < rows) {
    return true;
  }
  if (!dflash_->executor->InjectTargetContextDevice(
          features, exec.DflashPosition(), rows)) {
    AssignError(error_msg, "drafter context injection failed");
    return false;
  }
  return true;
}

std::uint32_t Session::Position() const noexcept {
  return session_->position;
}

std::uint32_t Session::ContextSize() const noexcept {
  return model_->options_.max_context;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return model_->SessionBytes(dflash_ ? core::SessionMode::kSpeculative
                                      : core::SessionMode::kAutoregressive,
                              model_->options_.max_context);
}

void Session::ResetDraftPolicy() {
  accepted_ema_ = static_cast<float>(model_->options_.max_draft_tokens);
  plain_steps_ = 0;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  model_->executor_->SetCancellationCheck(std::move(check));
}

void Session::Reset() {
  valid_ = false;
  model_->executor_->ResetSession(*session_);
  tokens_.clear();
  lookup_ = qfn::LookupIndex{};
  ResetDraftPolicy();
  if (dflash_) {
    dflash_->executor->Reset();
  }
  valid_ = true;
}

void Session::SyncLookup() {
  if (lookup_.size() > tokens_.size()) {
    lookup_ = qfn::LookupIndex{};
  }
  for (std::size_t i = lookup_.size(); i < tokens_.size(); ++i) {
    lookup_.Append(tokens_[i]);
  }
}

bool Session::Feed(std::span<const std::int32_t> tokens, bool prefill,
                   std::string* error_msg) {
  auto& exec = *model_->executor_;
  const std::size_t base = tokens_.size();
  const std::size_t end = base + tokens.size();
  const std::size_t window = model_->dflash_window_;
  bool ok;
  if (prefill && tokens.size() > 1) {
    // Only the drafter's attention window of the prompt matters, so forwards
    // that end before it skip the injection.
    const auto hook = [&](std::string* error) {
      const std::size_t forward_end =
          static_cast<std::size_t>(exec.DflashPosition()) + exec.DflashRows();
      if (!dflash_ || forward_end + window < end) {
        return true;
      }
      return DflashInject(exec.DflashRows(), error);
    };
    ok = exec.Prefill(*session_, tokens, error_msg, 0, hook);
  } else {
    ok = exec.Step(*session_, tokens, 1, error_msg);
    if (ok && dflash_) {
      ok = DflashInject(1, error_msg);
    }
  }
  if (!ok) {
    return false;
  }
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  if (!exec.ReadLogitRows(0, 1, logits_.data())) {
    AssignError(error_msg, "logit readback failed");
    return false;
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (!valid_) {
    Reset();
  }
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), true, error_msg);
  valid_ = ok;
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok =
      Feed(std::span<const std::int32_t>(&token, 1), false, error_msg);
  valid_ = ok;
  return ok;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  valid_ = false;
  const bool ok =
      DecodeImpl(max_tokens, sampler, result, error_msg, stop_at_eos);
  valid_ = ok;
  return ok;
}

bool Session::DecodeImpl(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  auto& exec = *model_->executor_;
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap = std::min<std::size_t>(
      {max_tokens, room, model_->options_.max_draft_tokens + 1});
  if (cap == 0) {
    result->stop = true;
    return true;
  }
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  const bool sampled = sampler.config().uses_random_sampling();
  const auto base = static_cast<std::uint32_t>(tokens_.size());
  std::vector<std::int32_t> chain{anchor};
  std::vector<qfn::MtpProposal> proposals;
  bool history = false;
  bool drafted_by_model = false;

  if (cap > 1 && speculative_) {
    // The history index includes the anchor, which every outcome commits.
    SyncLookup();
    lookup_.Append(anchor);
    const auto draft = lookup_.Draft(
        kHistoryMinMatch,
        std::min<std::size_t>(cap - 1, model_->options_.max_draft_tokens));
    const bool forced = forced_width_ >= 0;
    if (!forced && draft.size() >= kHistoryMinDraft) {
      history = true;
      for (const auto token : draft) {
        chain.push_back(token);
        if (sampled) {
          proposals.push_back(PointMass(token));
        }
      }
    } else if (DflashReady() && cap > 1 &&
               (forced ? forced_width_ > 0
                       : accepted_ema_ >= kPlainBelow ||
                             ++plain_steps_ % kProbeEvery == 0)) {
      const auto wanted =
          forced ? std::min<std::uint32_t>(
                       static_cast<std::uint32_t>(forced_width_),
                       static_cast<std::uint32_t>(cap - 1))
                 : std::clamp<std::uint32_t>(
                       static_cast<std::uint32_t>(std::ceil(accepted_ema_)) + 1,
                       2, static_cast<std::uint32_t>(cap - 1));
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
          if (complete) {
            block.push_back(proposal);
          }
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
  const auto vocab = model_->VocabSize();
  const bool speculative = k > 1;
  const bool gpu_greedy =
      !sampled && sampler.config().can_use_unmodified_argmax();
  if (speculative) {
    exec.SetRecording(true);
    const bool saved = exec.SaveState(*session_, error_msg);
    if (!saved) {
      exec.SetRecording(false);
      return false;
    }
  }
  const bool forwarded = exec.Step(*session_, chain, k, error_msg);
  exec.SetRecording(false);
  if (!forwarded) {
    return false;
  }

  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  if (speculative) {
    std::vector<std::int32_t> greedy;
    // A sampled session used to download a full vocabulary row per draft.
    // The candidate path returns each row's exact post-penalty top-k
    // instead; the CPU keeps its FP64 filtering and rejection sampler.
    const bool compact = !gpu_greedy && sampled &&
                         rocm::Executor::CanCompactVerification(sampler);
    std::vector<std::vector<sampling::LogitCandidate>> candidates;
    bool cpu_rows = !compact;
    const auto download_remaining = [&](std::uint32_t first) {
      verify_logits_.resize(static_cast<std::size_t>(k) * vocab);
      if (!exec.ReadLogitRows(first, k - first,
                              verify_logits_.data() + first * vocab)) {
        AssignError(error_msg, "verification readback failed");
        return false;
      }
      cpu_rows = true;
      return true;
    };
    if (gpu_greedy) {
      greedy = exec.ArgmaxRows(k);
    } else if (!compact) {
      if (!download_remaining(0))
        return false;
    } else if (!exec.SampledCandidates(&candidates, sampler,
                                       std::span(chain).subspan(1),
                                       error_msg)) {
      return false;
    }
    while (keep < k) {
      if (gpu_greedy) {
        const std::int32_t prediction = greedy[keep - 1];
        if (is_stop(prediction)) {
          result->stop = true;
          break;
        }
        if (prediction != chain[keep]) {
          break;
        }
        sampler.Accept(static_cast<sampling::TokenId>(prediction));
        ++keep;
        continue;
      }
      if (sampled) {
        // Accepting the anchor or a draft may activate a tool grammar. Keep
        // the complete-row path for constrained draws.
        if (compact && !cpu_rows && sampler.NeedsConstraintMask() &&
            !download_remaining(keep - 1))
          return false;
        const auto verified =
            compact && !cpu_rows
                ? qfn::VerifyMtpProposal(std::span(candidates[keep - 1]), vocab,
                                         proposals[keep - 1], sampler)
                : qfn::VerifyMtpProposal(
                      std::span<const float>(verify_logits_)
                          .subspan(static_cast<std::size_t>(keep - 1) * vocab,
                                   vocab),
                      proposals[keep - 1], sampler);
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
      const auto row =
          std::span<const float>(verify_logits_)
              .subspan(static_cast<std::size_t>(keep - 1) * vocab, vocab);
      const auto decision =
          qfn::VerifyDraft(row, chain[keep], sampler, is_stop);
      if (decision != qfn::DraftDecision::kAccept) {
        result->stop = decision == qfn::DraftDecision::kStop;
        break;
      }
      ++keep;
    }
    if (keep < k && !exec.Rollback(*session_, keep, error_msg)) {
      return false;
    }
  }
  // The next call samples from the last committed row.
  const std::uint32_t row = speculative ? keep - 1 : 0;
  if (!exec.ReadLogitRows(row, 1, logits_.data())) {
    AssignError(error_msg, "logit readback failed");
    return false;
  }
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);

  stats_.cycles += 1;
  stats_.widths[k - 1] += 1;
  if (speculative) {
    stats_.drafted += k - 1;
    stats_.accepted += keep - 1;
    if (history) {
      stats_.history_cycles += 1;
      stats_.history_drafted += k - 1;
      stats_.history_accepted += keep - 1;
    }
    if (drafted_by_model) {
      accepted_ema_ =
          0.7F * accepted_ema_ + 0.3F * static_cast<float>(keep - 1);
    }
  }
  if (!DflashInject(keep, error_msg)) {
    return false;
  }
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, preserving the actual
    // draw: resampling the target would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

std::uint64_t Session::SnapshotBytes() const {
  return HostBytes(static_cast<std::uint32_t>(tokens_.size()),
                   model_->VocabSize()) +
         model_->executor_->SnapshotBytes(*session_);
}

SessionSnapshot::SessionSnapshot(std::uint64_t size)
    : data_(new std::uint8_t[size]), size_(size) {}

SessionSnapshot::~SessionSnapshot() = default;

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  const auto token_count = static_cast<std::uint32_t>(tokens_.size());
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(*session_);
  std::unique_ptr<SessionSnapshot> snapshot(new SessionSnapshot(
      HostBytes(token_count, model_->VocabSize()) + executor_bytes));
  std::uint8_t* out = snapshot->data_.get();
  const SnapshotHeader header{
      .magic = kSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .reserved = 0,
      .executor_bytes = executor_bytes,
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  std::memcpy(out, tokens_.data(), tokens_.size() * sizeof(std::int32_t));
  out += tokens_.size() * sizeof(std::int32_t);
  std::memcpy(out, logits_.data(), logits_.size() * sizeof(float));
  out += logits_.size() * sizeof(float);
  if (!model_->executor_->SaveSnapshot(
          *session_,
          std::span<std::uint8_t>(out,
                                  static_cast<std::size_t>(executor_bytes)),
          error_msg)) {
    return nullptr;
  }
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshot(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  SnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.token_count > ContextSize() ||
      payload.size() != HostBytes(header.token_count, header.vocab_size) +
                            header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);
  valid_ = false;
  if (!model_->executor_->RestoreSnapshot(
          *session_, header.token_count,
          std::span<const std::uint8_t>(
              in, static_cast<std::size_t>(header.executor_bytes)),
          error_msg)) {
    Reset();
    return false;
  }
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  lookup_ = qfn::LookupIndex{};
  ResetDraftPolicy();
  stats_ = {};
  if (dflash_) {
    // The checkpoint does not carry drafter context; drafting resumes once a
    // forward primes the restored frontier again.
    dflash_->executor->Reset();
  }
  valid_ = true;
  return true;
}

}  // namespace gufo::models::glm53_flash
