#ifndef GUFO_MODELS_RAON1_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_RAON1_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
#include "src/models/raon1/runtime/config.hpp"

namespace gufo::models::raon1::rocm {
namespace qfn = gufo::models::qwen38_flash_next::rocm;

struct ExecutorOptions {
  /// Trunk layers whose residual (all four streams) feeds the drafter.
  std::vector<std::uint32_t> dflash_layers;
  std::uint32_t layers{0};
  std::uint32_t max_context{4096};
  bool head{true};
};

struct DeviceTensor {
  void* data{nullptr};
  int type{-1};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::size_t row_bytes{0};
  std::size_t bytes{0};
};

class Executor {
public:
  struct LayerState {
    float* conv{nullptr};
    float* state{nullptr};
    float* lat{nullptr};
    float* extra{nullptr};
    float* pool{nullptr};
  };
  struct Session {
    std::uint64_t id{0};
    std::uint32_t position{0};
    std::vector<LayerState> layers;
    /// Recurrent-state copies taken by SaveState (RDA layers only).
    std::vector<float*> snap_conv;
    std::vector<float*> snap_state;
    std::uint32_t snap_position{0};
    ~Session();
  };

  static std::unique_ptr<Executor> Create(const std::string& path,
                                          const ExecutorOptions& options,
                                          std::string* error);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  std::unique_ptr<Session> CreateSession(std::string* error);

  bool Evaluate(Session& session, std::int32_t token, std::string* error);
  static constexpr int kMaxBatch = 1024;
  static constexpr int kHeadRows = 8;
  static constexpr int kMaxVerify = 8;
  static constexpr int kMoeGemmMin = 16;
  bool EvaluateBatch(Session& session, std::span<const std::int32_t> tokens,
                     std::string* error);
  /// Two consecutive chunks of one prompt run as two pipelined banks (the
  /// second trails the first by a layer on its own stream and host thread);
  /// logits are those of the second chunk. Returns false without side effects
  /// when the pair does not fit (check CanPair first).
  [[nodiscard]] bool CanPair(const Session& session, int tokens,
                             int banks) const;
  [[nodiscard]] int PipelineBanks(std::size_t tokens) const;
  /// Drafter taps of the chunks of the last EvaluatePipelined, in order:
  /// {bank, first position, rows}.
  [[nodiscard]] std::size_t PipelineParts() const {
    return last_pipeline_.size();
  }
  [[nodiscard]] const float* PipelineCapture(std::size_t i) const {
    return banks_[last_pipeline_[i].bank]->capture;
  }
  [[nodiscard]] std::uint32_t PipelinePosition(std::size_t i) const {
    return last_pipeline_[i].position;
  }
  [[nodiscard]] std::uint32_t PipelineRows(std::size_t i) const {
    return last_pipeline_[i].rows;
  }
  bool EvaluatePipelined(Session& session, std::span<const std::int32_t> tokens,
                         int banks, std::string* error);
  /// Copies the recurrent state so a rejected draft can be replayed.
  bool SaveState(Session& session, std::string* error);
  /// Undoes the last forward and re-applies only its first `keep` tokens to the
  /// recurrent state; requires SetRecording(true) and SaveState before it.
  bool Rollback(Session& session, std::uint32_t keep, std::string* error);
  void SetRecording(bool on) { recording_ = on; }
  [[nodiscard]] bool SnapshotSupported() const noexcept { return true; }
  [[nodiscard]] std::size_t SnapshotBytes(const Session& session) const;
  bool SaveSnapshot(const Session& session, std::span<std::uint8_t> out,
                    std::string* error);
  bool RestoreSnapshot(Session& session, std::uint32_t position,
                       std::span<const std::uint8_t> in, std::string* error);
  /// Drafter taps of the last forward of bank `bank` (0 for plain batches).
  [[nodiscard]] const float* DflashCapture(int bank = 0) const {
    return banks_[bank]->capture;
  }
  [[nodiscard]] std::uint32_t DflashRows(int bank = 0) const {
    return banks_[bank]->cap_rows;
  }
  [[nodiscard]] std::uint32_t DflashPosition() const {
    return banks_[0]->cap_pos;
  }
  /// Greedy token of each of the last `rows` rows of the last forward.
  std::vector<std::int32_t> GreedyRows(std::uint32_t rows);
  bool ReadLogitRows(std::uint32_t first, std::uint32_t count, float* dst);
  bool WriteLogitRow(std::uint32_t row, const float* src);
  [[nodiscard]] const DeviceTensor* TokenEmbedding() const;
  [[nodiscard]] const DeviceTensor* Output() const;
  bool Logits(Session& session, float* dst, std::string* error);
  std::vector<float> Logits(Session& session);
  std::int32_t Argmax(Session& session, std::string* error);
  void ResetSession(Session& session);

  [[nodiscard]] const Config& config() const { return config_; }
  [[nodiscard]] std::uint32_t layers() const { return layers_; }
  [[nodiscard]] hipStream_t stream() const { return banks_[0]->stream; }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;

private:
  Executor() = default;

  bool Upload(const std::string& name, std::string* error);
  bool UploadImpl(const std::string& name, const core::GgufTensorInfo& t,
                  std::string* error);
  const DeviceTensor* Find(const std::string& name) const;
  bool Resolve(std::string* error);
  bool AllocScratch(std::string* error);
  // Quantize one activation once and let the following Matvec calls on it
  // (q/k/v/g, q_a/kv_a, shared gate/up) share the Q8_1 copy; PreqEnd clears it.
  void PreqBegin(const float* x, int n, int k);
  void PreqEnd() { preq_x_ = nullptr; }
  const float* preq_x_{nullptr};
  int preq_n_{0};
  int preq_k_{0};
  void* preq_buf_{nullptr};
  std::size_t preq_bytes_{0};
  bool Matvec(const DeviceTensor& w, const float* x, float* y, int m, int n,
              std::string* error);
  bool MatvecRows(const DeviceTensor& w, std::uint64_t row0, int m,
                  const float* x, float* y, std::string* error);
  bool F32Matvec(const DeviceTensor& w, const float* x, float* y, int m, int n,
                 std::string* error);

  std::unique_ptr<core::GgufReader> reader_;
  Config config_;
  ExecutorOptions options_;
  std::uint32_t layers_{0};
  std::unordered_map<std::string, DeviceTensor> tensors_;

  struct Scratch {
    float* x{nullptr};
    float* y{nullptr};
    float* pre{nullptr};
    float* post{nullptr};
    float* comb{nullptr};
    float* attn_in{nullptr};
    float* block{nullptr};
    float* ffn_in{nullptr};
    float* ffn_out{nullptr};
    float* head{nullptr};
    float* logits{nullptr};
    float* q{nullptr};
    float* k{nullptr};
    float* v{nullptr};
    float* fa{nullptr};
    float* gproj{nullptr};
    float* beta{nullptr};
    float* delta{nullptr};
    float* g2{nullptr};
    float* decay{nullptr};
    float* rda_out{nullptr};
    float* qr{nullptr};
    float* mla_q{nullptr};
    float* mla_qe{nullptr};
    float* kv{nullptr};
    float* qabs{nullptr};
    float* ctx{nullptr};
    float* gate{nullptr};
    float* moe_logits{nullptr};
    int* ids{nullptr};
    float* weights{nullptr};
    float* gf{nullptr};
    float* uf{nullptr};
    float* act{nullptr};
    float* dpart{nullptr};
    float* parts{nullptr};
    float* mla_part{nullptr};
    float* mla_qh{nullptr};
    float* quant_scratch{nullptr};
    int* argmax{nullptr};
    int* tok{nullptr};
    __half* a16{nullptr};
    __half* f16{nullptr};
    ~Scratch();
  };
  int last_rows_{1};
  struct Rec {
    float* q{nullptr};
    float* k{nullptr};
    float* v{nullptr};
    float* gproj{nullptr};
    float* g2{nullptr};
    float* beta{nullptr};
    float* delta{nullptr};
  };
  std::vector<Rec> rec_;
  bool recording_{false};
  bool tap_input_{false};
  /// Opt-in block-sparse long-context prefill attention (GUFO_RAON1_SPARSE=
  /// "k,window,min_depth"): sink + local window + top-k 64-row blocks by a
  /// Quest bound. Approximate; exact attention below min_depth.
  struct Sparse {
    bool on{false};
    int k{24};
    int window{1536};
    int min_depth{12288};
    int sink{256};
  } sparse_;
  struct Prefill {
    __half* x16{nullptr};
    __half* act16{nullptr};
    __half* down_e{nullptr};
    float* gate_e{nullptr};
    float* up_e{nullptr};
    std::uint32_t* counts{nullptr};
    std::int32_t* bounds{nullptr};
    std::int32_t* cursors{nullptr};
    std::int32_t* rows_token{nullptr};
    std::int32_t* rows_slot{nullptr};
    std::int32_t* tiles48{nullptr};
    std::int32_t* tiles64{nullptr};
    std::int32_t* tiles128{nullptr};
    float* situ{nullptr};
    std::size_t tile_cap{0};
    std::vector<std::uint32_t> host_counts;
    std::vector<std::int32_t> host_tiles;
    ~Prefill();
  };
  std::unordered_map<std::uint32_t, void*> down_q8_;
  static constexpr std::uint32_t kMlaBlock = 2048;
  struct MlaPrefill {
    __half* lat16{nullptr};
    float* knope{nullptr};
    float* vexp{nullptr};
    __half* k16{nullptr};
    __half* v16{nullptr};
    __half* q16{nullptr};
    float* scores{nullptr};
    int* sel{nullptr};
    int* kpos{nullptr};
    float* extra_g{nullptr};
    float* lat_g{nullptr};
    std::unordered_map<std::uint32_t, __half*> kbt;
    std::unordered_map<std::uint32_t, __half*> vbh;
  };
  struct FullKv {
    __half* k{nullptr};
    __half* v{nullptr};
    std::size_t len{0};
    std::uint64_t owner{0};
    std::uint32_t cap{0};
  };
  std::mutex full_mutex_;
  std::vector<FullKv> full_;
  std::uint32_t full_cap_{0};
  bool EnsureFull(std::uint32_t tokens);
  void InvalidateFull() noexcept {
    for (auto& f : full_)
      f.owner = 0;
  }
  static constexpr int kHalfGemmMin = 32;
  std::unordered_map<const void*, __half*> half_w_;
  bool BuildHalfWeights(std::string* error);
  bool AllocPrefill(bool first, std::string* error);
  bool AllocXh(std::string* error);
  std::size_t half_max_k_{0};
  bool PrefillMoe(std::uint32_t layer, int n, std::string* error);
  struct Bank {
    hipStream_t stream{nullptr};
    std::unique_ptr<Scratch> sc;
    std::unique_ptr<Prefill> pf;
    std::unique_ptr<MlaPrefill> mla;
    __half* xh{nullptr};
    void* mmq{nullptr};
    std::size_t mmq_bytes{0};
    std::uint32_t mla_cap{0};
    float* capture{nullptr};
    std::uint32_t cap_rows{0};
    std::uint32_t cap_pos{0};
    const float* hint_src[2]{nullptr, nullptr};
    __half* hint_buf[2]{nullptr, nullptr};
    int npu_a{-1};
    int npu_a2{-1};
    std::vector<int> npu_c;
    /// Ring of asynchronous NPU operations: a helper thread waits for the
    /// encode, submits, waits for the NPU and raises the flag a stalled stream
    /// is polling, so the bank's host thread keeps enqueuing meanwhile.
    struct NpuSlot {
      std::thread helper;
      hipEvent_t event{nullptr};
      int* flag{nullptr};
      int epoch{0};
    };
    static constexpr int kNpuSlots = 8;
    NpuSlot slots[kNpuSlots];
    int next_slot{0};
  };
  struct LayerGate;
  std::vector<std::unique_ptr<Bank>> banks_;
  static thread_local Bank* cur_bank_;
  int logit_bank_{0};
  bool AllocCapture(Bank& bank, std::string* error);
  struct PipelinePart {
    int bank;
    std::uint32_t position;
    std::uint32_t rows;
  };
  std::vector<PipelinePart> last_pipeline_;
  [[nodiscard]] Bank& B() const {
    return cur_bank_ != nullptr ? *cur_bank_ : *banks_[0];
  }
  [[nodiscard]] Scratch* LS() const { return banks_[logit_bank_]->sc.get(); }
  bool AddBank(std::string* error);
  /// XDNA2 offload (GUFO_NPU_DIR): the RDA layers' q/k/v/g projections run as
  /// 4 x 5 slices of 1024 outputs on the NPU for 1024-token chunks while the
  /// bank's stream is free for the other banks' GPU work.
  static constexpr int kNpuSlices = 5;
  static constexpr int kNpuProj = 4;
  std::unique_ptr<qfn::NpuGemm> npu_;
  std::mutex npu_mutex_;
  std::vector<std::vector<int>> npu_layers_;
  bool SetupNpu(std::string* error);
  bool AllocNpuBank(Bank& bank, std::string* error);
  struct NpuRun {
    int a;
    const int* w;
    int* c;
    int count;
  };
  struct NpuOp {
    int slot{-1};
    int epoch{0};
  };
  /// Starts the runs once the stream's earlier work (the encodes) finishes.
  NpuOp NpuStart(const std::vector<NpuRun>& runs);
  /// Stalls the stream until the operation's NPU runs are done.
  void NpuFinish(const NpuOp& op);
  bool NpuQkvgBegin(std::uint32_t layer, int n, NpuOp* op);
  void NpuQkvgEnd(const NpuOp& op, int n);
  /// A projection staged as `halves` K halves (2560 each) x `slices` output
  /// slices of 1024; weight ids are [half * slices + slice].
  struct NpuLin {
    int halves{0};
    int slices{0};
    int n_out{0};
    std::vector<int> w;
    [[nodiscard]] bool ok() const { return !w.empty(); }
  };
  struct NpuLayer {
    NpuLin out;   // attn_output: K = 5120 (RDA) or MlaVDim, N = hidden
    NpuLin gate;  // MLA attn_gate
    NpuLin shg;   // shared expert gate
    NpuLin shu;   // shared expert up
  };
  std::vector<NpuLayer> npu_lin_;
  bool NpuLinSetup(NpuLin& lin, const DeviceTensor& w, std::uint32_t k_total,
                   int n_out, std::string* error);
  /// Encodes the F16 activations in xh_ (pitch ldx) and starts the runs; the
  /// stream is synchronized first. Returns the ticket to pass to NpuLinEnd.
  NpuOp NpuLinBegin(const NpuLin& lin, int n, std::size_t ldx,
                    const __half* x16);
  /// Waits for the ticket and decodes the result into out (pitch `pitch`).
  void NpuLinEnd(const NpuLin& lin, const NpuOp& op, float* out,
                 std::size_t pitch, int n);
  [[nodiscard]] bool NpuOn(int n) const {
    return npu_ != nullptr && n == static_cast<int>(kMaxBatch) &&
           npu_->Healthy() && B().npu_a >= 0;
  }
  bool RunBatch(Session& session, std::span<const std::int32_t> tokens,
                std::string* error, std::uint32_t pos0, LayerGate* in,
                LayerGate* out, bool head);
};

}  // namespace gufo::models::raon1::rocm

#endif  // GUFO_MODELS_RAON1_KERNELS_ROCM_EXECUTOR_HPP_
