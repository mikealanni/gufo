// NPU offload of the Qwen3.8-27B FFN gate/up projections during prefill (see
// src/models/qwen/hip/npu_ffn.hpp). The qwen executor registers nothing about
// the NPU itself; RegisterQwenNpuFfn() hands it this factory.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/qwen/hip/npu_ffn.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kTokens = 4096;
constexpr std::uint32_t kMinRows = 1536;
constexpr std::uint32_t kHidden = 5120;
constexpr std::uint32_t kIntermediate = 17408;
constexpr std::uint32_t kMoeHidden = 2048;
constexpr std::uint32_t kMoeSsmInner = 4096;
constexpr std::uint32_t kMoeQkvSlices = 6;
constexpr std::uint32_t kMoeGateSlices = 2;
constexpr std::uint32_t kMoeSsmOutSlices = 1;
constexpr std::uint32_t kMoeAttnQSlices = 3;
constexpr std::uint32_t kMoeAttnOSlices = 1;
constexpr std::uint32_t kDenseAttnQSlices = 4;
constexpr std::uint32_t kDenseAttnOSlices = 2;
constexpr std::uint32_t kSliceColumns = 1024;
constexpr std::uint32_t kDefaultSlices = 7;
constexpr std::uint32_t kIntermediateK = kIntermediate;
constexpr std::uint32_t kDefaultDownSlices = 1;
constexpr std::uint32_t kDefaultSsmOutSlices = 2;
constexpr std::uint32_t kDefaultQkvSlices = 4;
constexpr std::uint32_t kDefaultGateSlices = 2;
constexpr std::uint32_t kMaxSlices = 16;

bool EncodableType(core::GgmlType type) {
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
         type == core::GgmlType::kQ3_K || type == core::GgmlType::kIQ4_NL ||
         type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
}

class Qwen27bNpuFfn final : public hip::QwenNpuFfn {
public:
  static std::unique_ptr<hip::QwenNpuFfn> Create(
      const QwenModelWeights& weights, std::string* error) {
    const auto& config = weights.config;
    const bool dense27b = config.hidden_size == kHidden &&
                          config.intermediate_size == kIntermediate;
    const bool moe35b = config.IsMoe() && config.hidden_size == kMoeHidden &&
                        config.ssm_inner_size == kMoeSsmInner;
    if (!dense27b && !moe35b) {
      return nullptr;
    }
    const std::uint32_t hidden = config.hidden_size;
    const std::uint32_t ssm_inner = config.ssm_inner_size;
    const char* dir = std::getenv("GUFO_NPU_DIR");
    if (dir == nullptr) {
      return nullptr;
    }
    std::uint32_t slices = dense27b ? kDefaultSlices : 0;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_FFN_SLICES")) {
      slices = static_cast<std::uint32_t>(
          std::clamp(std::atoi(v), 0, static_cast<int>(kMaxSlices)));
    }
    std::uint32_t qkv_slices = dense27b ? kDefaultQkvSlices : kMoeQkvSlices;
    std::uint32_t gate_slices = dense27b ? kDefaultGateSlices : kMoeGateSlices;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_QKV_SLICES")) {
      qkv_slices = static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 10));
    }
    if (const char* v = std::getenv("GUFO_QWEN_NPU_GATE_SLICES")) {
      gate_slices = static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 6));
    }
    std::uint32_t down_slices = dense27b ? kDefaultDownSlices : 0;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_DOWN_SLICES")) {
      down_slices = static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 5));
    }
    std::uint32_t ssm_out_slices =
        dense27b ? kDefaultSsmOutSlices : kMoeSsmOutSlices;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_SSM_OUT_SLICES")) {
      ssm_out_slices =
          static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 5));
    }
    const char* attn_env = std::getenv("GUFO_QWEN_NPU_ATTN");
    const bool attn_enabled = attn_env == nullptr || attn_env[0] != '0';
    std::uint32_t attn_q_slices =
        dense27b ? kDenseAttnQSlices : kMoeAttnQSlices;
    std::uint32_t attn_o_slices =
        dense27b ? kDenseAttnOSlices : kMoeAttnOSlices;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_ATTN_Q_SLICES")) {
      attn_q_slices =
          static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 8));
    }
    if (const char* v = std::getenv("GUFO_QWEN_NPU_ATTN_O_SLICES")) {
      attn_o_slices =
          static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 2));
    }
    if (slices == 0 && qkv_slices + gate_slices + down_slices + ssm_out_slices +
                               attn_q_slices + attn_o_slices ==
                           0) {
      return nullptr;
    }
    NpuGemm::Config cfg;
    cfg.xclbin =
        std::string(dir) + "/runtime_k/final_4096xKx1024_128x64x128.xclbin";
    cfg.insts = std::string(dir) + "/runtime_k/insts_4096x" +
                std::to_string(hidden) + "x1024_128x64x128.txt";
    cfg.tokens = kTokens;
    cfg.k = hidden;
    cfg.n = kSliceColumns;
    auto npu = NpuGemm::Create(cfg, error);
    if (npu == nullptr) {
      return nullptr;
    }
    auto ffn = std::unique_ptr<Qwen27bNpuFfn>(new Qwen27bNpuFfn());
    ffn->hidden_ = hidden;
    ffn->ssm_inner_ = ssm_inner;
    ffn->slices_ = slices;
    ffn->qkv_slices_ = qkv_slices;
    ffn->gate_slices_ = gate_slices;
    ffn->down_slices_ = down_slices;
    ffn->ssm_out_slices_ = ssm_out_slices;
    if (ssm_out_slices != 0 || attn_o_slices != 0) {
      ffn->ssm_out_program_ = npu->AddProgram(
          std::string(dir) + "/runtime_k/insts_4096x" +
              std::to_string(ssm_inner) + "x1024_128x64x128.txt",
          error);
      if (ffn->ssm_out_program_ < 0) {
        return nullptr;
      }
      for (auto& slot : ffn->slots_) {
        slot.a_ssm_out = npu->NewBuffer(npu->ABytes(ssm_inner), error);
        if (slot.a_ssm_out < 0) {
          return nullptr;
        }
      }
    }
    if (down_slices != 0) {
      ffn->down_program_ = npu->AddProgram(
          std::string(dir) + "/runtime_k/insts_4096x17408x1024_128x64x128.txt",
          error);
      if (ffn->down_program_ < 0) {
        return nullptr;
      }
      for (auto& slot : ffn->slots_) {
        slot.a_down = npu->NewBuffer(npu->ABytes(kIntermediateK), error);
        if (slot.a_down < 0) {
          return nullptr;
        }
      }
    }
    for (auto& slot : ffn->slots_) {
      slot.a = npu->NewBuffer(npu->ABytes(), error);
      if (slot.a < 0) {
        return nullptr;
      }
    }
    for (std::uint32_t i = 0;
         i < std::max({2 * slices, qkv_slices + gate_slices, down_slices,
                       ssm_out_slices, attn_q_slices, attn_o_slices});
         ++i) {
      const int id = npu->NewBuffer(npu->CBytes(), error);
      if (id < 0) {
        return nullptr;
      }
      ffn->slots_[0].c.push_back(id);
      const int id2 = npu->NewBuffer(npu->CBytes(), error);
      if (id2 < 0) {
        return nullptr;
      }
      ffn->slots_[1].c.push_back(id2);
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
      return nullptr;
    }
    std::size_t staged = 0;
    std::uint32_t layers_on_npu = 0;
    ffn->b_.resize(weights.layers.size());
    ffn->columns_.assign(weights.layers.size(), 0);
    ffn->lin_b_.resize(weights.layers.size());
    ffn->down_b_.resize(weights.layers.size());
    ffn->ssm_out_b_.resize(weights.layers.size());
    ffn->ssm_out_columns_.assign(weights.layers.size(), 0);
    ffn->down_columns_.assign(weights.layers.size(), 0);
    ffn->qkv_columns_.assign(weights.layers.size(), 0);
    ffn->gate_columns_.assign(weights.layers.size(), 0);
    const auto encode = [&](const QwenTensorRef& w, std::uint32_t count,
                            std::vector<int>& ids, std::uint32_t k) {
      for (std::uint32_t i = 0; i < count; ++i) {
        const int id = npu->NewBuffer(npu->BBytes(k), error);
        if (id < 0) {
          return false;
        }
        if (w.type == core::GgmlType::kQ8_0) {
          NpuEncodeWeights(w.data, k, i * kSliceColumns, kSliceColumns, k, 0, k,
                           npu->Device(id), stream);
        } else {
          NpuEncodeWeightsQuant(w.data, static_cast<std::uint32_t>(w.type), k,
                                i * kSliceColumns, kSliceColumns, k, 0, k,
                                npu->Device(id), stream);
        }
        ids.push_back(id);
        staged += npu->BBytes(k);
      }
      return true;
    };
    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
      const auto& layer = weights.layers[l];
      std::vector<int> ids;
      bool ok = true;
      if (slices != 0 && EncodableType(layer.ffn_gate.type) &&
          EncodableType(layer.ffn_up.type)) {
        ok = encode(layer.ffn_gate, slices, ids, hidden) &&
             encode(layer.ffn_up, slices, ids, hidden);
        if (ok) {
          ffn->columns_[l] = slices * kSliceColumns;
          ++layers_on_npu;
        }
      }
      std::vector<int> lin;
      if (ok && layer.is_full_attention && attn_enabled && attn_q_slices != 0 &&
          EncodableType(layer.attn_q.type) &&
          layer.attn_q.num_elements ==
              std::size_t{2} * config.AttentionSize() * hidden) {
        const std::uint32_t q_slices =
            std::min(attn_q_slices, 2 * config.AttentionSize() / kSliceColumns);
        ok = encode(layer.attn_q, q_slices, lin, hidden);
        if (ok) {
          ffn->qkv_columns_[l] = q_slices * kSliceColumns;
        }
      }
      if (ok && !layer.is_full_attention && qkv_slices + gate_slices != 0 &&
          EncodableType(layer.attn_qkv.type) &&
          EncodableType(layer.attn_gate.type) &&
          layer.attn_qkv.num_elements ==
              std::size_t{config.SsmQkvSize()} * hidden &&
          layer.attn_gate.num_elements == std::size_t{ssm_inner} * hidden) {
        ok = encode(layer.attn_qkv, qkv_slices, lin, hidden) &&
             encode(layer.attn_gate, gate_slices, lin, hidden);
        if (ok) {
          ffn->qkv_columns_[l] = qkv_slices * kSliceColumns;
          ffn->gate_columns_[l] = gate_slices * kSliceColumns;
        }
      }
      std::vector<int> ssm_out;
      if (ok && layer.is_full_attention && attn_enabled && attn_o_slices != 0 &&
          EncodableType(layer.attn_output.type) &&
          layer.attn_output.num_elements == std::size_t{hidden} * ssm_inner) {
        ok = encode(layer.attn_output, attn_o_slices, ssm_out, ssm_inner);
        if (ok) {
          ffn->ssm_out_columns_[l] = attn_o_slices * kSliceColumns;
        }
      }
      if (ok && !layer.is_full_attention && ssm_out_slices != 0 &&
          EncodableType(layer.ssm_out.type) &&
          layer.ssm_out.num_elements == std::size_t{hidden} * ssm_inner) {
        ok = encode(layer.ssm_out, ssm_out_slices, ssm_out, ssm_inner);
        if (ok) {
          ffn->ssm_out_columns_[l] = ssm_out_slices * kSliceColumns;
        }
      }
      std::vector<int> down;
      if (ok && down_slices != 0 && dense27b &&
          EncodableType(layer.ffn_down.type) &&
          layer.ffn_down.num_elements == std::size_t{hidden} * kIntermediateK) {
        ok = encode(layer.ffn_down, down_slices, down, kIntermediateK);
        if (ok) {
          ffn->down_columns_[l] = down_slices * kSliceColumns;
        }
      }
      if (!ok) {
        hipStreamDestroy(stream);
        return nullptr;
      }
      hipStreamSynchronize(stream);
      for (const int id : down) {
        npu->FinalizeWeights(id);
      }
      for (const int id : ssm_out) {
        npu->FinalizeWeights(id);
      }
      ffn->ssm_out_b_[l] = std::move(ssm_out);
      ffn->down_b_[l] = std::move(down);
      for (const int id : ids) {
        npu->FinalizeWeights(id);
      }
      for (const int id : lin) {
        npu->FinalizeWeights(id);
      }
      ffn->b_[l] = std::move(ids);
      ffn->lin_b_[l] = std::move(lin);
    }
    hipStreamDestroy(stream);
    std::fprintf(stderr,
                 "NPU: FFN gate/up first %u columns of %u layers, linear qkv "
                 "%u gate %u, down %u columns, %.2f GiB staged\n",
                 slices * kSliceColumns, layers_on_npu,
                 qkv_slices * kSliceColumns, gate_slices * kSliceColumns,
                 down_slices * kSliceColumns,
                 static_cast<double>(staged) / (1024.0 * 1024.0 * 1024.0));
    ffn->npu_ = std::move(npu);
    return ffn;
  }

  std::uint32_t Tokens() const noexcept override { return kTokens; }
  bool Accepts(std::uint32_t rows) const noexcept override {
    return rows >= kMinRows && rows <= kTokens;
  }
  void SetRows(std::uint32_t rows) noexcept override { rows_ = rows; }
  void SetPaired(bool paired) noexcept override { paired_ = paired; }
  bool Paired() const noexcept override { return paired_; }
  std::uint32_t Columns(std::uint32_t layer) const noexcept override {
    return layer < columns_.size() ? FfnUsed() * kSliceColumns : 0;
  }

  bool Begin(std::uint32_t layer, int bank, const void* normed,
             hipStream_t stream) override {
    Slot& slot = slots_[bank];
    if (!npu_->Healthy() || !Acquire(stream)) {
      return false;
    }
    NpuEncodeActivations(static_cast<const __half*>(normed), hidden_, rows_,
                         hidden_, 0, hidden_, npu_->Device(slot.a), stream);
    const auto& full = b_[layer];
    const std::uint32_t used = FfnUsed();
    std::vector<int> b(full.begin(), full.begin() + used);
    b.insert(b.end(), full.begin() + slices_, full.begin() + slices_ + used);
    if (!npu_->Submit(stream, slot.a, b.data(), slot.c.data(),
                      static_cast<int>(b.size()))) {
      Release(this);
      return false;
    }
    slot.ticket = npu_->Ticket();
    return true;
  }

  void Finish(std::uint32_t layer, int bank, void* out, std::size_t ld,
              hipStream_t stream) override {
    (void)layer;
    Slot& slot = slots_[bank];
    npu_->Wait(stream, slot.ticket);
    const std::uint32_t used = FfnUsed();
    for (std::uint32_t i = 0; i < used; ++i) {
      NpuSwiglu(npu_->Staged(stream, slot.c[i]),
                npu_->Staged(stream, slot.c[used + i]), nullptr, nullptr,
                static_cast<__half*>(out), rows_, kSliceColumns, kSliceColumns,
                stream, ld, i * kSliceColumns);
    }
    // The shared buffers are free once these kernels have read them.
    hipLaunchHostFunc(
        stream, [](void* self) { Release(static_cast<Qwen27bNpuFfn*>(self)); },
        this);
  }

  std::uint32_t SsmOutColumns(std::uint32_t layer) const noexcept override {
    return layer < ssm_out_columns_.size() ? Scaled(ssm_out_columns_[layer])
                                           : 0;
  }

  bool BeginSsmOut(std::uint32_t layer, int bank, const void* act,
                   hipStream_t stream) override {
    Slot& slot = slots_[bank];
    if (!npu_->Healthy() || !Acquire(stream)) {
      return false;
    }
    NpuEncodeActivations(static_cast<const __half*>(act), ssm_inner_, rows_,
                         ssm_inner_, 0, ssm_inner_,
                         npu_->Device(slot.a_ssm_out), stream);
    const auto& b = ssm_out_b_[layer];
    if (!npu_->Submit(
            stream, slot.a_ssm_out, b.data(), slot.c.data(),
            static_cast<int>(Used(static_cast<std::uint32_t>(b.size()))),
            ssm_out_program_)) {
      Release(this);
      return false;
    }
    slot.ticket = npu_->Ticket();
    return true;
  }

  void FinishSsmOut(std::uint32_t layer, int bank, float* residual,
                    std::size_t ld, hipStream_t stream) override {
    Slot& slot = slots_[bank];
    npu_->Wait(stream, slot.ticket);
    const std::uint32_t count = Used(ssm_out_columns_[layer] / kSliceColumns);
    for (std::uint32_t i = 0; i < count; ++i) {
      NpuDecodeOutput(npu_->Staged(stream, slot.c[i]), residual, rows_,
                      kSliceColumns, ld, i * kSliceColumns, true, stream);
    }
    hipLaunchHostFunc(
        stream, [](void* self) { Release(static_cast<Qwen27bNpuFfn*>(self)); },
        this);
  }

  std::uint32_t DownColumns(std::uint32_t layer) const noexcept override {
    return layer < down_columns_.size() ? Scaled(down_columns_[layer]) : 0;
  }

  bool BeginDown(std::uint32_t layer, int bank, const void* act,
                 hipStream_t stream) override {
    Slot& slot = slots_[bank];
    if (!npu_->Healthy() || !Acquire(stream)) {
      return false;
    }
    NpuEncodeActivations(static_cast<const __half*>(act), kIntermediateK, rows_,
                         kIntermediateK, 0, kIntermediateK,
                         npu_->Device(slot.a_down), stream);
    const auto& b = down_b_[layer];
    if (!npu_->Submit(
            stream, slot.a_down, b.data(), slot.c.data(),
            static_cast<int>(Used(static_cast<std::uint32_t>(b.size()))),
            down_program_)) {
      Release(this);
      return false;
    }
    slot.ticket = npu_->Ticket();
    return true;
  }

  void FinishDown(std::uint32_t layer, int bank, float* residual,
                  std::size_t ld, hipStream_t stream) override {
    Slot& slot = slots_[bank];
    npu_->Wait(stream, slot.ticket);
    const std::uint32_t count = Used(down_columns_[layer] / kSliceColumns);
    for (std::uint32_t i = 0; i < count; ++i) {
      NpuDecodeOutput(npu_->Staged(stream, slot.c[i]), residual, rows_,
                      kSliceColumns, ld, i * kSliceColumns, true, stream);
    }
    hipLaunchHostFunc(
        stream, [](void* self) { Release(static_cast<Qwen27bNpuFfn*>(self)); },
        this);
  }

  std::uint32_t QkvColumns(std::uint32_t layer) const noexcept override {
    return layer < qkv_columns_.size() ? Scaled(qkv_columns_[layer]) : 0;
  }
  std::uint32_t GateColumns(std::uint32_t layer) const noexcept override {
    return layer < gate_columns_.size() ? Scaled(gate_columns_[layer]) : 0;
  }

  bool BeginLinear(std::uint32_t layer, int bank, const void* normed,
                   hipStream_t stream) override {
    Slot& slot = slots_[bank];
    if (!npu_->Healthy() || !Acquire(stream)) {
      return false;
    }
    NpuEncodeActivations(static_cast<const __half*>(normed), hidden_, rows_,
                         hidden_, 0, hidden_, npu_->Device(slot.a), stream);
    const auto& full = lin_b_[layer];
    const std::uint32_t qkv_full = qkv_columns_[layer] / kSliceColumns;
    const std::uint32_t qkv_used = Used(qkv_full);
    const std::uint32_t gate_used = Used(gate_columns_[layer] / kSliceColumns);
    std::vector<int> b(full.begin(), full.begin() + qkv_used);
    b.insert(b.end(), full.begin() + qkv_full,
             full.begin() + qkv_full + gate_used);
    if (!npu_->Submit(stream, slot.a, b.data(), slot.c.data(),
                      static_cast<int>(b.size()))) {
      Release(this);
      return false;
    }
    slot.ticket = npu_->Ticket();
    return true;
  }

  void FinishLinear(std::uint32_t layer, int bank, float* qkv,
                    std::size_t qkv_ld, float* gate, std::size_t gate_ld,
                    hipStream_t stream) override {
    Slot& slot = slots_[bank];
    npu_->Wait(stream, slot.ticket);
    const std::uint32_t qkv_count = Used(qkv_columns_[layer] / kSliceColumns);
    const std::uint32_t gate_count = Used(gate_columns_[layer] / kSliceColumns);
    for (std::uint32_t i = 0; i < qkv_count; ++i) {
      NpuDecodeOutput(npu_->Staged(stream, slot.c[i]), qkv, rows_,
                      kSliceColumns, qkv_ld, i * kSliceColumns, false, stream);
    }
    for (std::uint32_t i = 0; i < gate_count; ++i) {
      NpuDecodeOutput(npu_->Staged(stream, slot.c[qkv_count + i]), gate, rows_,
                      kSliceColumns, gate_ld, i * kSliceColumns, false, stream);
    }
    hipLaunchHostFunc(
        stream, [](void* self) { Release(static_cast<Qwen27bNpuFfn*>(self)); },
        this);
  }

private:
  Qwen27bNpuFfn() = default;

  /// Slices the NPU computes for `rows` active rows: the program always runs
  /// kTokens rows, so a shorter chunk gets proportionally fewer slices to keep
  /// the NPU and the GPU finishing together.
  [[nodiscard]] std::uint32_t Used(std::uint32_t slices) const noexcept {
    if (slices == 0) {
      return 0;
    }
    return std::max<std::uint32_t>(1,
                                   (slices * rows_ + (kTokens / 2)) / kTokens);
  }
  /// A lone chunk cannot hide the NPU's wait behind a partner's GPU work, so
  /// it hands the FFN one slice less than a paired chunk does.
  [[nodiscard]] std::uint32_t FfnUsed() const noexcept {
    return Used(slices_ > 1 && !paired_ ? slices_ - 1 : slices_);
  }
  [[nodiscard]] std::uint32_t Scaled(std::uint32_t columns) const noexcept {
    return Used(columns / kSliceColumns) * kSliceColumns;
  }

  /// One stream owns the shared NPU buffers at a time. Its own layers follow
  /// each other in stream order, so they need no host-side wait; another
  /// stream is refused until the owner's last queued layer has finished.
  bool Acquire(hipStream_t stream) {
    hipStream_t expected = nullptr;
    if (!owner_.compare_exchange_strong(expected, stream) &&
        expected != stream) {
      return false;
    }
    pending_.fetch_add(1);
    return true;
  }
  static void Release(Qwen27bNpuFfn* self) {
    if (self->pending_.fetch_sub(1) == 1) {
      self->owner_.store(nullptr);
    }
  }

  std::unique_ptr<NpuGemm> npu_;
  std::uint32_t hidden_{0};
  std::uint32_t rows_{kTokens};
  bool paired_{false};
  std::uint32_t ssm_inner_{0};
  std::uint32_t slices_{0};
  std::uint32_t down_slices_{0};
  std::uint32_t ssm_out_slices_{0};
  int ssm_out_program_{0};
  std::vector<std::vector<int>> ssm_out_b_;
  std::vector<std::uint32_t> ssm_out_columns_;
  int down_program_{0};
  std::vector<std::vector<int>> down_b_;
  std::vector<std::uint32_t> down_columns_;
  std::uint32_t qkv_slices_{0};
  std::uint32_t gate_slices_{0};
  std::vector<std::vector<int>> lin_b_;
  std::vector<std::uint32_t> qkv_columns_;
  std::vector<std::uint32_t> gate_columns_;
  /// Operand buffers of one chunk in flight; the ticket is the NPU submission
  /// count after that chunk's latest stage, so a wait covers only its own runs.
  struct Slot {
    int a{-1};
    int a_down{-1};
    int a_ssm_out{-1};
    std::vector<int> c;
    std::uint64_t ticket{0};
  };
  std::array<Slot, 2> slots_;
  std::vector<std::vector<int>> b_;
  std::vector<std::uint32_t> columns_;
  std::atomic<hipStream_t> owner_{nullptr};
  std::atomic<int> pending_{0};
};

}  // namespace

void RegisterQwenNpuFfn() {
  hip::SetQwenNpuFfnFactory(&Qwen27bNpuFfn::Create);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
