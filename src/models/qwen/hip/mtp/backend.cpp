#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/mtp.hpp"

namespace gufo::hip {

QwenMtpGpuDraftBackend::QwenMtpGpuDraftBackend(
    std::unique_ptr<QwenMtpGpuExecutor> executor, QwenMtpGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      target_hidden_(executor_->GetHiddenSize()) {}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, QwenMtpGpuDraftConfig config,
    std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU draft configuration is invalid";
    }
    return nullptr;
  }
  auto executor = QwenMtpGpuExecutor::Create(std::move(model),
                                             config.max_context, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  executor->SetVisionInput(config.vision_input);
  return std::unique_ptr<QwenMtpGpuDraftBackend>(
      new QwenMtpGpuDraftBackend(std::move(executor), config));
}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::CreateFromGguf(
    std::string_view model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenMtpGpuDraftConfig config, std::string* error_msg) {
  if (model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  std::shared_ptr<const QwenMtpGpuModel> model;
  if (model_path == "embedded") {
    model = QwenMtpGpuModel::CreateEmbedded(std::move(target_model), error_msg);
  } else {
    auto reader_owner =
        core::GgufReader::OpenFile(std::string(model_path), error_msg);
    if (reader_owner == nullptr) {
      return nullptr;
    }
    std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
    model = QwenMtpGpuModel::Create(std::move(reader), std::move(target_model),
                                    error_msg);
  }
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenMtpGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty() || context.hidden_size == 0 ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    last_error_ = "MTP prompt hidden-state shape is invalid";
    return false;
  }

  try {
    if (context.prompt_tokens.size() > 1) {
      executor_->PrimeBatch(
          context.prompt_tokens.subspan(1),
          context.prompt_hidden_states.first(
              (context.prompt_tokens.size() - 1) * context.hidden_size),
          0);
    }
    // The prompt rows may be pinned and reused by the next target prefill.
    executor_->Synchronize();
    const std::size_t final_offset =
        (context.prompt_tokens.size() - 1) * context.hidden_size;
    const auto final_hidden =
        context.prompt_hidden_states.subspan(final_offset, context.hidden_size);
    std::ranges::copy(final_hidden, target_hidden_.begin());
    primed_ = true;
    return true;
  } catch (const std::exception& exception) {
    last_error_ = exception.what();
    // Earlier per-row uploads may still be reading the pinned prompt rows.
    try {
      executor_->Synchronize();
    } catch (...) {
    }
    Reset();
    return false;
  }
}

bool QwenMtpGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || proposal_active_ ||
      executor_->GetNextPosition() + 1 != position ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    return false;
  }
  if (context.prompt_tokens.empty()) {
    return true;
  }
  // Row r pairs token r with the hidden state of the token before it: the
  // carried frontier hidden for r == 0, then the supplied rows.
  std::vector<float> paired(context.prompt_tokens.size() *
                            target_hidden_.size());
  std::ranges::copy(target_hidden_, paired.begin());
  std::copy_n(
      context.prompt_hidden_states.begin(),
      (context.prompt_tokens.size() - 1) * target_hidden_.size(),
      paired.begin() + static_cast<std::ptrdiff_t>(target_hidden_.size()));
  executor_->PrimeBatch(context.prompt_tokens, paired, position - 1);
  std::ranges::copy(context.prompt_hidden_states.last(target_hidden_.size()),
                    target_hidden_.begin());
  return true;
}

speculative::DraftProposal QwenMtpGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("MTP GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("MTP GPU proposal feedback is pending");
  }
  if (executor_->GetNextPosition() + 1 != current_pos) {
    throw std::logic_error("MTP GPU draft position is inconsistent");
  }

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  // Verifying a row reads its own experts, so MoE targets pay for every
  // extra draft row; three is the sweet spot on prose and code.
  constexpr std::uint32_t kMoeDraftCap = 3;
  std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  if (executor_->IsMoe()) {
    count = std::min(count, kMoeDraftCap);
  }
  if (count == 0) {
    return proposal;
  }

  proposal_checkpoint_ = executor_->GetNextPosition();
  proposal_input_ = prompt_tokens.back();
  proposal_target_hidden_ = target_hidden_;
  step_hidden_ = target_hidden_;
  step_checkpoint_ = proposal_checkpoint_;
  step_valid_ = true;
  committed_target_hidden_.clear();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  auto token = executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                              proposal_checkpoint_, true);
  proposed_tokens_.push_back(token);
  for (std::uint32_t index = 1; index < count; ++index) {
    token =
        executor_->ForwardFeedback(token, proposal_checkpoint_ + index, true);
    proposed_tokens_.push_back(token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

void QwenMtpGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("MTP GPU proposal feedback is invalid");
  }
  if (committed_target_hidden_.size() !=
      (accepted.size() + 1) * target_hidden_.size()) {
    throw std::logic_error("MTP feedback requires each committed target row");
  }

  // Pair input t+1 with target hidden t, as during prompt priming. Verification
  // has already updated target_hidden_ to the final committed row; it cannot
  // be reused for the earlier anchor. Accepted rows also need target features,
  // rather than the tentative draft hidden states used while proposing.
  executor_->Rewind(proposal_checkpoint_);
  (void)executor_->ForwardTargetHidden(proposal_input_, proposal_target_hidden_,
                                       proposal_checkpoint_, false);
  for (std::size_t index = 0; index < accepted.size(); ++index) {
    (void)executor_->ForwardTargetHidden(
        accepted[index],
        std::span<const float>(committed_target_hidden_)
            .subspan(index * target_hidden_.size(), target_hidden_.size()),
        proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1, false);
  }
  proposal_active_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

void QwenMtpGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  if (proposal_active_) {
    committed_target_hidden_.insert(committed_target_hidden_.end(),
                                    hidden.begin(), hidden.end());
  }
  std::ranges::copy(hidden, target_hidden_.begin());
}

void QwenMtpGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_, 0.0F);
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  step_valid_ = false;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
}

void QwenMtpGpuDraftBackend::DiscardPendingTargetContext(
    std::uint32_t position) {
  if (proposal_active_ || !step_valid_ || step_checkpoint_ + 1 != position) {
    throw std::logic_error("MTP cancellation frontier is not retained");
  }
  executor_->Rewind(step_checkpoint_);
  target_hidden_ = step_hidden_;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

std::vector<std::uint8_t> QwenMtpGpuDraftBackend::SerializeState() const {
  const std::size_t state_bytes = executor_->StateBytes();
  std::vector<std::uint8_t> payload(
      state_bytes + (target_hidden_.size() * sizeof(float)) + 1);
  executor_->SaveState(std::span(payload).first(state_bytes));
  std::memcpy(payload.data() + state_bytes, target_hidden_.data(),
              target_hidden_.size() * sizeof(float));
  payload.back() = primed_ ? 1 : 0;
  return payload;
}

void QwenMtpGpuDraftBackend::LoadSerializedState(
    std::span<const std::uint8_t> payload) {
  const std::size_t tail = (target_hidden_.size() * sizeof(float)) + 1;
  if (payload.size() < tail + sizeof(std::uint32_t)) {
    throw std::invalid_argument("MTP snapshot payload is truncated");
  }
  const std::size_t state_bytes = payload.size() - tail;
  executor_->LoadState(payload.first(state_bytes));
  std::memcpy(target_hidden_.data(), payload.data() + state_bytes,
              target_hidden_.size() * sizeof(float));
  primed_ = payload.back() != 0;
  proposal_active_ = false;
  step_valid_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

namespace {
class QwenMtpDraftSnapshot final : public speculative::IDraftBackendSnapshot {
public:
  explicit QwenMtpDraftSnapshot(std::vector<std::uint8_t> payload)
      : payload_(std::move(payload)) {}
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return payload_.size();
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    return payload_.size();
  }
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    if (destination.size() != payload_.size()) {
      throw std::invalid_argument(
          "MTP snapshot destination has the wrong size");
    }
    std::memcpy(destination.data(), payload_.data(), payload_.size());
    return payload_.size();
  }
  [[nodiscard]] std::span<const std::uint8_t> Payload() const noexcept {
    return payload_;
  }

private:
  std::vector<std::uint8_t> payload_;
};
}  // namespace

std::size_t QwenMtpGpuDraftBackend::SnapshotPayloadBytes() const {
  return executor_->StateBytes() + (target_hidden_.size() * sizeof(float)) + 1;
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenMtpGpuDraftBackend::Snapshot() const {
  if (proposal_active_) {
    throw std::logic_error(
        "MTP snapshot requires a committed proposal boundary");
  }
  return std::make_unique<QwenMtpDraftSnapshot>(SerializeState());
}

void QwenMtpGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* mtp = dynamic_cast<const QwenMtpDraftSnapshot*>(&snapshot);
  if (mtp == nullptr) {
    throw std::invalid_argument("snapshot does not belong to an MTP draft");
  }
  LoadSerializedState(mtp->Payload());
}

void QwenMtpGpuDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  LoadSerializedState(payload);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
