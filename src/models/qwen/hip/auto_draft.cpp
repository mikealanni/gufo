#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/auto_draft.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace gufo::hip {
namespace {

constexpr double kRateSmoothing = 0.3;
constexpr double kSwitchMargin = 1.05;
constexpr std::uint32_t kFirstProbeStep = 12;
constexpr std::uint32_t kProbeSteps = 3;
constexpr std::uint32_t kReprobeIdleSteps = 128;
constexpr double kMaxStepSeconds = 2.0;
constexpr std::uint8_t kSnapshotVersion = 1;

class AutoSnapshot final : public speculative::IDraftBackendSnapshot {
public:
  AutoSnapshot(std::unique_ptr<speculative::IDraftBackendSnapshot> dflash,
               std::unique_ptr<speculative::IDraftBackendSnapshot> mtp)
      : dflash_(std::move(dflash)), mtp_(std::move(mtp)) {}
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return dflash_->PayloadBytes() + mtp_->PayloadBytes();
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    return 1 + (2 * sizeof(std::uint64_t)) + dflash_->PersistentPayloadBytes() +
           mtp_->PersistentPayloadBytes();
  }
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    const std::size_t dflash_bytes = dflash_->PersistentPayloadBytes();
    const std::size_t mtp_bytes = mtp_->PersistentPayloadBytes();
    if (destination.size() != PersistentPayloadBytes()) {
      throw std::invalid_argument("auto draft snapshot destination is wrong");
    }
    std::uint8_t* out = destination.data();
    *out++ = kSnapshotVersion;
    const std::uint64_t dflash_size = dflash_bytes;
    const std::uint64_t mtp_size = mtp_bytes;
    std::memcpy(out, &dflash_size, sizeof(dflash_size));
    out += sizeof(dflash_size);
    (void)dflash_->SerializePersistent(
        std::span<std::uint8_t>(out, dflash_bytes));
    out += dflash_bytes;
    std::memcpy(out, &mtp_size, sizeof(mtp_size));
    out += sizeof(mtp_size);
    (void)mtp_->SerializePersistent(std::span<std::uint8_t>(out, mtp_bytes));
    return destination.size();
  }
  [[nodiscard]] const speculative::IDraftBackendSnapshot& dflash() const {
    return *dflash_;
  }
  [[nodiscard]] const speculative::IDraftBackendSnapshot& mtp() const {
    return *mtp_;
  }

private:
  std::unique_ptr<speculative::IDraftBackendSnapshot> dflash_;
  std::unique_ptr<speculative::IDraftBackendSnapshot> mtp_;
};

}  // namespace

QwenAutoDraftBackend::QwenAutoDraftBackend(
    std::unique_ptr<QwenDFlashGpuDraftBackend> dflash,
    std::unique_ptr<QwenMtpGpuDraftBackend> mtp,
    std::vector<std::uint32_t> layer_ids,
    std::vector<std::size_t> dflash_columns, std::size_t mtp_column,
    std::size_t hidden_size)
    : dflash_(std::move(dflash)),
      mtp_(std::move(mtp)),
      layer_ids_(std::move(layer_ids)),
      dflash_columns_(std::move(dflash_columns)),
      mtp_column_(mtp_column),
      hidden_size_(hidden_size) {}

std::unique_ptr<QwenAutoDraftBackend> QwenAutoDraftBackend::Create(
    std::unique_ptr<QwenDFlashGpuDraftBackend> dflash,
    std::unique_ptr<QwenMtpGpuDraftBackend> mtp, std::uint32_t num_layers,
    std::string* error_msg) {
  if (dflash == nullptr || mtp == nullptr || num_layers == 0) {
    if (error_msg != nullptr) {
      *error_msg = "auto draft needs both a DFlash2 and an MTP backend";
    }
    return nullptr;
  }
  const auto dflash_layers = dflash->TargetHiddenLayerIds();
  std::vector<std::uint32_t> ids(dflash_layers.begin(), dflash_layers.end());
  std::vector<std::size_t> dflash_columns(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    dflash_columns[i] = i;
  }
  // MTP reads the output of the last block.
  const std::uint32_t final_layer = num_layers - 1;
  std::size_t mtp_column = ids.size();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] == final_layer) {
      mtp_column = i;
    }
  }
  if (mtp_column == ids.size()) {
    ids.push_back(final_layer);
  }
  const std::size_t hidden = mtp->GetHiddenSize();
  return std::unique_ptr<QwenAutoDraftBackend>(new QwenAutoDraftBackend(
      std::move(dflash), std::move(mtp), std::move(ids),
      std::move(dflash_columns), mtp_column, hidden));
}

void QwenAutoDraftBackend::Columns(std::span<const float> rows,
                                   std::size_t row_count,
                                   std::span<const std::size_t> columns,
                                   std::vector<float>& out) const {
  const std::size_t width = layer_ids_.size() * hidden_size_;
  out.resize(row_count * columns.size() * hidden_size_);
  float* destination = out.data();
  for (std::size_t row = 0; row < row_count; ++row) {
    for (const std::size_t column : columns) {
      std::memcpy(destination,
                  rows.data() + (row * width) + (column * hidden_size_),
                  hidden_size_ * sizeof(float));
      destination += hidden_size_;
    }
  }
}

bool QwenAutoDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  const std::size_t width = layer_ids_.size() * hidden_size_;
  if (context.hidden_size != width ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * width) {
    return false;
  }
  Reset();
  const std::size_t rows = context.prompt_tokens.size();
  const std::size_t mtp_columns[] = {mtp_column_};
  const bool skip_dflash = dflash_->CanSkipTargetContext(context, 0);
  if (!skip_dflash)
    Columns(context.prompt_hidden_states, rows, dflash_columns_,
            dflash_columns_scratch_);
  Columns(context.prompt_hidden_states, rows, mtp_columns,
          mtp_columns_scratch_);
  const auto& dflash_states = dflash_columns_scratch_;
  const auto& mtp_states = mtp_columns_scratch_;
  speculative::DraftTargetContext dflash_context = context;
  dflash_context.prompt_hidden_states = dflash_states;
  dflash_context.hidden_size = dflash_columns_.size() * hidden_size_;
  speculative::DraftTargetContext mtp_context = context;
  mtp_context.prompt_hidden_states = mtp_states;
  mtp_context.hidden_size = hidden_size_;
  return dflash_->PrimeTargetContext(dflash_context) &&
         mtp_->PrimeTargetContext(mtp_context);
}

bool QwenAutoDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  const std::size_t width = layer_ids_.size() * hidden_size_;
  if (context.hidden_size != width ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * width) {
    return false;
  }
  const std::size_t rows = context.prompt_tokens.size();
  const std::size_t mtp_columns[] = {mtp_column_};
  const bool skip_dflash = dflash_->CanSkipTargetContext(context, position);
  if (!skip_dflash)
    Columns(context.prompt_hidden_states, rows, dflash_columns_,
            dflash_columns_scratch_);
  Columns(context.prompt_hidden_states, rows, mtp_columns,
          mtp_columns_scratch_);
  const auto& dflash_states = dflash_columns_scratch_;
  const auto& mtp_states = mtp_columns_scratch_;
  speculative::DraftTargetContext dflash_context = context;
  dflash_context.prompt_hidden_states = dflash_states;
  dflash_context.hidden_size = dflash_columns_.size() * hidden_size_;
  speculative::DraftTargetContext mtp_context = context;
  mtp_context.prompt_hidden_states = mtp_states;
  mtp_context.hidden_size = hidden_size_;
  mtp_->MarkStep();
  return dflash_->AppendTargetContext(dflash_context, position) &&
         mtp_->AppendTargetContext(mtp_context, position);
}

QwenAutoDraftBackend::Which QwenAutoDraftBackend::Choose() {
  const auto df = static_cast<std::size_t>(Which::kDFlash);
  const auto mt = static_cast<std::size_t>(Which::kMtp);
  ++rates_[df].idle_steps;
  ++rates_[mt].idle_steps;
  if (exploring_ > 0) {
    --exploring_;
    return proposer_;
  }
  if (steps_in_request_ == kFirstProbeStep && rates_[mt].samples == 0) {
    exploring_ = kProbeSteps - 1;
    return Which::kMtp;
  }
  const Which best = rates_[mt].tokens_per_second >
                             rates_[df].tokens_per_second * kSwitchMargin
                         ? Which::kMtp
                         : Which::kDFlash;
  const Which other = best == Which::kDFlash ? Which::kMtp : Which::kDFlash;
  if (rates_[static_cast<std::size_t>(other)].idle_steps > kReprobeIdleSteps &&
      steps_in_request_ > kFirstProbeStep) {
    exploring_ = kProbeSteps - 1;
    return other;
  }
  if (steps_in_request_ < kFirstProbeStep) {
    return Which::kDFlash;
  }
  return best;
}

speculative::DraftProposal QwenAutoDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  const auto now = std::chrono::steady_clock::now();
  if (have_last_ && pending_committed_ > 0) {
    const double seconds =
        std::chrono::duration<double>(now - last_propose_).count();
    if (seconds > 0.0 && seconds < kMaxStepSeconds) {
      auto& rate = rates_[static_cast<std::size_t>(last_which_)];
      const double observed = static_cast<double>(pending_committed_) / seconds;
      rate.tokens_per_second =
          rate.samples == 0
              ? observed
              : (kRateSmoothing * observed) +
                    ((1.0 - kRateSmoothing) * rate.tokens_per_second);
      ++rate.samples;
      rate.idle_steps = 0;
    }
  }
  pending_committed_ = 0;
  proposer_ = Choose();
  ++steps_in_request_;
  last_propose_ = now;
  step_frontier_ = prompt_tokens.back();
  step_position_ = current_pos;
  mtp_rows_.clear();
  mtp_->MarkStep();
  auto proposal = proposer_ == Which::kDFlash
                      ? dflash_->Propose(prompt_tokens, current_pos, max_tokens)
                      : mtp_->Propose(prompt_tokens, current_pos, max_tokens);
  proposal_open_ = !proposal.tokens.empty();
  last_which_ = proposer_;
  have_last_ = proposal_open_;
  return proposal;
}

speculative::DraftProposal QwenAutoDraftBackend::ProposeSampled(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  proposer_ = Which::kDFlash;
  ++steps_in_request_;
  step_frontier_ = prompt_tokens.back();
  step_position_ = current_pos;
  mtp_rows_.clear();
  mtp_->MarkStep();
  auto proposal = dflash_->ProposeSampled(prompt_tokens, current_pos,
                                          max_tokens, temperature, rng_state);
  proposal_open_ = !proposal.tokens.empty();
  have_last_ = false;
  return proposal;
}

void QwenAutoDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  const std::size_t width = layer_ids_.size() * hidden_size_;
  if (hidden.size() != width) {
    throw std::invalid_argument("auto draft target hidden width is invalid");
  }
  std::vector<float> dflash_row;
  Columns(hidden, 1, dflash_columns_, dflash_row);
  dflash_->UpdateTargetHidden(dflash_row);
  const auto* mtp_row = hidden.data() + (mtp_column_ * hidden_size_);
  if (proposal_open_ && proposer_ == Which::kMtp) {
    mtp_->UpdateTargetHidden(std::span<const float>(mtp_row, hidden_size_));
  } else {
    mtp_rows_.insert(mtp_rows_.end(), mtp_row, mtp_row + hidden_size_);
  }
}

void QwenAutoDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  if (proposer_ == Which::kDFlash) {
    dflash_->AcceptFeedback(accepted, correction_token);
    // MTP follows the committed tokens: the frontier and each accepted draft.
    std::vector<tokenization::TokenId> consumed;
    consumed.reserve(accepted.size() + 1);
    consumed.push_back(step_frontier_);
    consumed.insert(consumed.end(), accepted.begin(), accepted.end());
    if (mtp_rows_.size() != consumed.size() * hidden_size_ ||
        !mtp_->AppendTargetContext({.prompt_tokens = consumed,
                                    .prompt_hidden_states = mtp_rows_,
                                    .hidden_size = hidden_size_,
                                    .first_token = correction_token},
                                   step_position_)) {
      throw std::runtime_error("MTP failed to follow the committed tokens");
    }
  } else {
    mtp_->AcceptFeedback(accepted, correction_token);
  }
  mtp_rows_.clear();
  proposal_open_ = false;
  RecordStep(proposer_, accepted.size() + 1);
}

void QwenAutoDraftBackend::RecordStep(Which which, std::size_t committed) {
  // The step's wall time is only known once the next one starts; remember the
  // yield now and let Propose close the interval.
  pending_committed_ = committed;
  (void)which;
}

void QwenAutoDraftBackend::DiscardPendingTargetContext(std::uint32_t position) {
  dflash_->DiscardPendingTargetContext(position);
  mtp_->DiscardPendingTargetContext(position);
  mtp_rows_.clear();
  proposal_open_ = false;
}

std::pair<std::size_t, std::size_t> QwenAutoDraftBackend::RequestMemoryBytes()
    const {
  const auto a = dflash_->RequestMemoryBytes();
  const auto b = mtp_->RequestMemoryBytes();
  return {a.first + b.first, a.second + b.second};
}

std::size_t QwenAutoDraftBackend::SnapshotPayloadBytes() const {
  return dflash_->SnapshotPayloadBytes() + mtp_->SnapshotPayloadBytes();
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenAutoDraftBackend::Snapshot() const {
  if (proposal_open_) {
    throw std::logic_error("auto draft snapshot needs a committed boundary");
  }
  return std::make_unique<AutoSnapshot>(dflash_->Snapshot(), mtp_->Snapshot());
}

void QwenAutoDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* auto_snapshot = dynamic_cast<const AutoSnapshot*>(&snapshot);
  if (auto_snapshot == nullptr) {
    throw std::invalid_argument("snapshot does not belong to an auto draft");
  }
  dflash_->RestoreSnapshot(auto_snapshot->dflash());
  mtp_->RestoreSnapshot(auto_snapshot->mtp());
  mtp_rows_.clear();
  proposal_open_ = false;
}

void QwenAutoDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  constexpr std::size_t kHeader = 1 + sizeof(std::uint64_t);
  if (payload.size() < kHeader || payload[0] != kSnapshotVersion) {
    throw std::invalid_argument("auto draft snapshot payload is invalid");
  }
  std::uint64_t dflash_size = 0;
  std::memcpy(&dflash_size, payload.data() + 1, sizeof(dflash_size));
  if (dflash_size > payload.size() - kHeader - sizeof(std::uint64_t)) {
    throw std::invalid_argument("auto draft snapshot payload is truncated");
  }
  const auto dflash_payload = payload.subspan(kHeader, dflash_size);
  const std::size_t mtp_header = kHeader + dflash_size;
  std::uint64_t mtp_size = 0;
  std::memcpy(&mtp_size, payload.data() + mtp_header, sizeof(mtp_size));
  if (mtp_size != payload.size() - mtp_header - sizeof(mtp_size)) {
    throw std::invalid_argument("auto draft snapshot payload is inconsistent");
  }
  dflash_->RestorePersistentSnapshot(dflash_payload);
  mtp_->RestorePersistentSnapshot(
      payload.subspan(mtp_header + sizeof(mtp_size), mtp_size));
  mtp_rows_.clear();
  proposal_open_ = false;
}

void QwenAutoDraftBackend::Reset() noexcept {
  dflash_->Reset();
  mtp_->Reset();
  mtp_rows_.clear();
  proposal_open_ = false;
  BeginRequest();
}

void QwenAutoDraftBackend::BeginRequest() noexcept {
  dflash_->BeginRequest();
  mtp_->BeginRequest();
  rates_[0] = {};
  rates_[1] = {};
  steps_in_request_ = 0;
  exploring_ = 0;
  have_last_ = false;
  pending_committed_ = 0;
  proposer_ = Which::kDFlash;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
