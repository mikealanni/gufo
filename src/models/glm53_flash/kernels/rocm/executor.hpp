#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/kernels/rocm/dsa.hpp"
#include "src/models/glm53_flash/runtime/config.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
class NpuGemm;
}

namespace gufo::models::glm53_flash::rocm {

struct ExecutorOptions {
  // Trunk layers to run (0 = all). A prefix is useful for layer-level checks.
  std::uint32_t layers{0};
  std::uint32_t max_context{4096};
  std::uint32_t top_pools{kDsaTopPools};
  // Largest token batch one forward pass handles (prefill chunk).
  std::uint32_t prefill_chunk{256};
  // Skip the output head (embedding to residual only).
  bool head{true};
  // Drafter feature taps: the input of each listed layer (the output of the
  // layer before it), stream-averaged, for the last `dflash_rows` rows of a
  // forward pass is kept in DflashCapture().
  std::vector<std::uint32_t> dflash_layers;
  std::uint32_t dflash_rows{0};
  // Largest number of trailing rows whose logits one forward can return.
  std::uint32_t logit_rows{1};
};

struct DeviceTensor {
  void* data{nullptr};
  // Q8_0 rows of the same weights for the W8A8 prefill GEMMs: `data` itself
  // for Q8_0 tensors, a re-encoded copy for Q6_K, null otherwise.
  void* q8{nullptr};
  bool q8_owned{false};
  // attn_k_b / attn_v_b only: all head slabs dequantized to F16 for the
  // strided-batched prefill GEMMs.
  void* f16{nullptr};
  int type{-1};
  std::uint32_t k{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  std::size_t slab_bytes{0};  // bytes of one expert / head slab
  [[nodiscard]] const float* f32() const {
    return static_cast<const float*>(data);
  }
};

// GPU decode engine for GLM-5.3-Flash: one token per step on one stream.
class Executor {
public:
  struct LayerState {
    float* conv{nullptr};   // KDA: [3][3][8192]
    float* state{nullptr};  // KDA: [64][128][128]
    DsaCache cache{};       // DSA
  };
  struct Session {
    std::uint64_t id{0};
    std::uint32_t position{0};
    std::vector<LayerState> layers;
    std::vector<float*> saved_conv;  // SaveState snapshots
    std::vector<float*> saved_state;
    std::uint32_t saved_position{0};
    // KDA operands of the last recorded forward, per layer, for Rollback.
    struct KdaLog {
      float *raw_q{}, *raw_k{}, *raw_v{}, *decay{}, *beta{};
    };
    std::vector<KdaLog> kda_log;
    ~Session();
  };

  static std::unique_ptr<Executor> Create(const std::string& path,
                                          const ExecutorOptions& options,
                                          std::string* error);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  std::unique_ptr<Session> CreateSession(std::string* error);

  // Runs one token through the trunk (and head unless disabled).
  bool Evaluate(Session& session, std::int32_t token, std::string* error);

  // Runs `tokens` (at most prefill_chunk) as one batch and leaves the logits of
  // the last `rows` of them (rows <= logit_rows); ArgmaxRows() reads the
  // greedy choice for each.
  bool Verify(Session& session, std::span<const std::int32_t> tokens,
              std::uint32_t rows, std::string* error);
  // Greedy token of each of the last `rows` verified rows.
  std::vector<std::int32_t> ArgmaxRows(std::uint32_t rows);
  // Saves / restores the recurrent state (KDA) so a rejected draft can be
  // replayed; the caller restores session.position itself.
  bool SaveState(Session& session, std::string* error);
  void RestoreState(Session& session);
  // After SaveState + a recorded Step: undo the forward and re-apply only its
  // first `keep` tokens to the recurrent state (conv history and delta state
  // rebuilt from the recorded operands), leaving session.position at
  // saved position + keep. Attention caches need no repair.
  bool Rollback(Session& session, std::uint32_t keep, std::string* error);
  // Records the KDA operands of the next Steps (set before a verification).
  void SetRecording(bool on) { recording_ = on; }

  // Drafter features of the last forward: [rows][taps * 4096] on the device,
  // for absolute positions DflashPosition() ...
  [[nodiscard]] const float* DflashCapture() const;
  [[nodiscard]] std::uint32_t DflashRows() const { return capture_rows_; }
  [[nodiscard]] std::uint32_t DflashPosition() const {
    return capture_position_;
  }
  // Device views of the embedding and output head for the drafter.
  [[nodiscard]] const DeviceTensor* TokenEmbedding() const;
  [[nodiscard]] const DeviceTensor* Output() const;
  [[nodiscard]] const Config& config() const { return config_; }

  // One batched forward of `tokens` (at most prefill_chunk) that keeps the
  // logits of the last `logit_rows` of them and the drafter taps.
  bool Step(Session& session, std::span<const std::int32_t> tokens,
            std::uint32_t logit_rows, std::string* error) {
    return Forward(session, tokens.data(), static_cast<int>(tokens.size()),
                   static_cast<int>(logit_rows), error);
  }

  // Runs a prompt in chunks of prefill_chunk tokens; logits (via Logits) are
  // those of the final token.
  // `chunk` (0 = prefill_chunk) may be smaller than the configured capacity.
  // `after_forward` runs once each forward (a bank pair counts as one) has
  // finished, while its drafter captures are still valid.
  using ForwardHook = std::function<bool(std::string*)>;
  bool Prefill(Session& session, std::span<const std::int32_t> tokens,
               std::string* error, std::size_t chunk = 0,
               const ForwardHook& after_forward = {});

  // `count` rows of the previous forward's logits starting at row `first`
  // (of the rows it kept), [count][vocab].
  bool ReadLogitRows(std::uint32_t first, std::uint32_t count,
                     float* destination);

  // Zeroes the recurrent state and rewinds the session to position 0.
  void ResetSession(Session& session);

  // Exact byte form of a session's state at its current position: the KDA
  // recurrent state and every DSA row below the position.
  [[nodiscard]] std::uint64_t SnapshotBytes(const Session& session) const;
  bool SaveSnapshot(const Session& session, std::span<std::uint8_t> out,
                    std::string* error);
  bool RestoreSnapshot(Session& session, std::uint32_t position,
                       std::span<const std::uint8_t> in, std::string* error);

  // Polled between layers; a true result abandons the forward, leaving the
  // session unusable until restored or reset.
  void SetCancellationCheck(std::function<bool()> check) {
    cancel_ = std::move(check);
  }

  // Host copies; valid after Evaluate returns.
  std::vector<float> Logits(const Session& session);
  std::vector<float> Residual(const Session& session);

  [[nodiscard]] std::uint32_t layers() const { return layers_; }
  [[nodiscard]] std::uint32_t PrefillChunk() const {
    return options_.prefill_chunk;
  }

  // Per-stage wall times when GUFO_GLM53_PROFILE is set (adds syncs).
  void PrintProfile();

private:
  Executor() = default;

  struct LayerWeights;
  struct Scratch;

  bool CopyToDevice(void* dst, const core::GgufTensorInfo& t,
                    std::string* error);
  bool Upload(const std::string& name, std::string* error);
  const DeviceTensor* Find(const std::string& name) const;
  bool Resolve(std::string* error);
  bool SetupNpu(std::string* error);
  bool SetupRouter(std::string* error);
  bool AllocScratch(std::string* error);

  bool Forward(Session& session, const std::int32_t* tokens, int count,
               int logit_rows, std::string* error);
  // A projection input: F32 rows and, for wide batches, the same rows
  // quantized once into the tiled Q8 layout the W8A8 GEMM reads.
  struct Act {
    const float* f{nullptr};
    const void* q{nullptr};
  };
  Act Prep(const float* x, int tokens, int k) const;
  void Proj(const DeviceTensor& w, const Act& x, float* out, int tokens) const;
  void KdaBlock(std::uint32_t layer, Session& s, int tokens) const;
  void DsaBlock(std::uint32_t layer, Session& s, int tokens) const;
  void HeadGemm(const DeviceTensor& w, void* out, bool half_out, int tokens,
                int k, int rows) const;
  void RoutedExpertsF16(const LayerWeights& w, int tokens) const;
  void FfnBlock(std::uint32_t layer, int tokens) const;
  void RunLayer(std::uint32_t layer, Session& session, int tokens, int& cur,
                bool attn_prepared, int ffn_rows);
  // Rows of the last layer's FFN that reach the head; every row elsewhere.
  int FfnRows(std::uint32_t layer, int tokens, int logit_rows) const;
  void FinishHead(float* residual, int tokens, int logit_rows);
  bool EmbedRows(const std::int32_t* tokens, int count, float* dst,
                 std::string* error);
  void NpuQkvQueue(int bank, int tokens);
  void NpuQkvFinish(int bank) const;
  void SwapBank();
  void NpuShexpQueue(std::uint32_t layer) const;
  void NpuShexpFinish() const;
  bool ForwardPair(Session& session, const std::int32_t* tokens, int count,
                   int logit_rows, std::string* error);
  void HcPreStep(const void* x, const DeviceTensor& fn, const float* scale,
                 const float* base, const float* norm_w, int tokens) const;
  void HcPostStep(const void* x, const float* block, void* next,
                  int tokens) const;
  void* ResidualAt(float* base, std::size_t rows_offset) const;

  std::unique_ptr<core::GgufReader> reader_;
  Config config_;
  ExecutorOptions options_;
  std::uint32_t layers_{0};
  std::unordered_map<std::string, DeviceTensor> tensors_;
  std::vector<LayerWeights> weights_;
  std::unique_ptr<Scratch> scratch_;
  struct BankFields {
    float* x[2]{};
    float *y{}, *post{}, *comb{}, *hc_mixes{};
  };
  BankFields bank_b_;
  std::unique_ptr<gufo::models::qwen38_flash_next::rocm::NpuGemm> npu_;
  std::vector<std::vector<int>> npu_qkv_;
  // Q8_0 re-encode buffer for projections whose resident Q8 copy the NPU
  // staging released; passes that do not run on the NPU rebuild it per use.
  void* q8_fallback_{nullptr};
  std::size_t q8_fallback_bytes_{0};
  int npu_a_[2][1]{{-1}, {-1}};
  std::vector<int> npu_c_[2];
  std::uint64_t npu_ticket_[2]{};
  int qkv_bank_{-1};
  std::unordered_map<const float*, std::pair<std::uint16_t*, std::uint16_t*>>
      hc_fn_split_;
  std::vector<std::uint16_t*> router_hi_;
  std::vector<std::uint16_t*> router_lo_;
  bool res_half_{false};
  int shexp_bank_{-1};
  std::vector<std::vector<int>> npu_sh_;
  std::vector<int> npu_sh_c_[2];
  const std::uint8_t** npu_sh_tab_[2]{};
  mutable std::uint64_t npu_sh_ticket_[2]{};
  const std::uint8_t** npu_tab_[2]{};
  std::uint32_t current_layer_{0};
  hipStream_t stream_{nullptr};
  void* staging_{nullptr};
  void* blas_{nullptr};  // hipblasHandle_t
  float* last_residual_{nullptr};
  std::vector<std::uint32_t> dflash_taps_;
  std::uint32_t capture_rows_{0};
  int last_logit_rows_{1};
  std::uint32_t capture_position_{0};
  DeviceTensor token_embd_dev_{};
  bool recording_{false};
  std::function<bool()> cancel_;
  [[nodiscard]] bool Cancelled() const { return cancel_ && cancel_(); }
  bool profile_{false};
  mutable std::size_t arena_used_{0};
  mutable double profile_seconds_[10]{};
  mutable std::chrono::steady_clock::time_point profile_mark_{};
  void Mark(int bucket) const;
  void DropGraph();
  hipGraphExec_t graph_exec_{nullptr};
  std::uint64_t graph_session_{0};
  std::uint64_t session_counter_{0};
  int graph_logit_rows_{0};
  bool graph_warm_{false};
  hipStream_t capture_stream_{nullptr};
};

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
