#ifndef GUFO_MODELS_RAON1_KERNELS_ROCM_RAON_HPP_
#define GUFO_MODELS_RAON1_KERNELS_ROCM_RAON_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::raon1::rocm {

constexpr int kRaonHeads = 40;
constexpr int kRaonHeadDim = 128;
constexpr int kRaonStreams = 4;
constexpr int kRaonMixes = 17;
constexpr int kRaonHidden = 2560;
constexpr int kRaonFfn = 1088;
constexpr int kRaonExperts = 128;
constexpr int kRaonUsed = 6;
constexpr int kRaonRank = 512;
constexpr int kRaonNoPe = 128;
constexpr int kRaonRope = 64;
constexpr int kRaonVDim = 128;

// y[i] = sum_j w[i][j] * x[j], with w stored as F32 rows of width k.
int F32Matvec(const float* w, const float* x, float* y, int m, int k,
              hipStream_t stream);

// y[row] = sum_j half(w[(row0 + row)][j]) * x[j].
int F16Matvec(const __half* w, const float* x, float* y, int m, int k,
              hipStream_t stream, std::uint64_t row0 = 0);

// y[token][row] = sum_j half(w[(ids[token] * m + row)][j]) * x[token][j] for
// one token (the Raon executor calls it once per token, n=1).
int F16MatvecExperts(const __half* w, const float* x, float* y, int m, int k,
                     int used, const int* ids, bool x_per_expert,
                     hipStream_t stream);

// Per-token mHC selector: RMS-normalize the four residuals, run fn matvec,
// apply sigmoid gates and the Helmert chart, then mix streams into `y`.
// `x` and `dst` are [tokens][4][H]; fn is [17][4H] F32 (dequantized BF16).
int HcSelect(const float* x, const float* fn, const float* scale,
             const float* base, float hc_eps, float magnitude, float* y,
             float* pre, float* post, float* comb, hipStream_t stream,
             int tokens = 1);

/// dots[t][17] = fn . x[t] (fn is [17][4H] F32).
int HcDots(const float* x, const float* fn, float* dots, int tokens,
           hipStream_t stream);
/// HcSelect with the 17 raw mixes (fn . x per token) precomputed in `dots`
/// [tokens][17]; used for wide batches where a GEMM computes them.
int HcSelectPre(const float* x, const float* dots, const float* scale,
                const float* base, float hc_eps, float magnitude, float* y,
                float* pre, float* post, float* comb, hipStream_t stream,
                int tokens, int chunks = 1);
/// Chunked partial mixes for few tokens: dots [tokens][kHcPartChunks][17],
/// consumed by HcSelectPre with chunks = kHcPartChunks.
inline constexpr int kHcPartChunks = 20;
int HcDotsPart(const float* x, const float* fn, float* dots, int tokens,
               hipStream_t stream);

// dst[dst_stream][i] = block[i]*post[dst] + sum_s x[s][i]*comb[dst+4*s].
int HcApply(const float* x, const float* block, const float* post,
            const float* comb, float* dst, hipStream_t stream, int tokens = 1);

// y[i] = x[i] * rsqrt(mean(x^2)+eps) * w[i] (w may be null).
int RmsNormRows(const float* x, const float* w, float* y, int n, int rows,
                float eps, hipStream_t stream, __half* y16 = nullptr);

// y[i] = x[i] * rsqrt(sum(x^2)+eps) over each head of `n` floats.
int L2Heads(const float* x, float* y, int heads, int hd, float eps,
            hipStream_t stream);

// RDA causal 1D convolution over the last `taps` raw inputs plus current.
// `hist` holds [3][channels][taps], `raw` is [3][channels] current values.
// Writes gated result to `out` [3][channels] and shifts the history.
int RdaConv(const float* wq, const float* wk, const float* wv,
            const float* rawq, const float* rawk, const float* rawv,
            float* hist, float* outq, float* outk, float* outv, int channels,
            int taps, hipStream_t stream);

// decay[i] = exp(lower_bound * sigmoid(-sa[h] * (gproj[h*hd+i] + dt[i]))).
int RdaDecay(const float* gproj, const float* dt, const float* sa, float* decay,
             int heads, int hd, float lower_bound, hipStream_t stream);

// Per-head delta-rule recurrence (one token). `state` is [heads][vd][hd],
// `q/k/v` are [channels], `decay` and `beta` per head, `g2` per channel.
// `ratio` is the per-head value write scale. Writes `out` [channels].
int RdaRecur(const float* q, const float* k, const float* v, const float* decay,
             const float* beta, const float* ratio, const float* g2,
             const float* norm_w, float* state, float* out, float* pre_norm,
             int heads, int hd, float eps, hipStream_t stream);

/// Split-K absorbed MLA attention for `tokens` consecutive queries starting at
/// position pos0: qabs [t][heads][rank], qe [t][heads][rope], ctx
/// [t][heads][rank]. `part` needs tokens * heads * kMlaPartSplits * (rank + 2)
/// floats.
inline constexpr int kMlaPartSplits = 128;
int MlaAttendSplit(const float* qabs, const float* qe, const float* lat,
                   const float* extra, float* part, float* ctx, int heads,
                   int rank, int rope, int pos0, int tokens,
                   hipStream_t stream);

// Absorbed NoPE MLA attention over cached latents [n][rank] and extra
// [n][rope]. qabs is [heads][rank], qe is [heads][rope], wk/wv unused here
// (already applied by the caller). Writes attention output [heads][vd].
int MlaAttend(const float* qabs, const float* qe, const float* lat,
              const float* extra, float* out, int heads, int rank, int rope,
              int vd, int n, hipStream_t stream);

// Router: sigmoid(logits) + bias, top-k (k <= 8). Writes sorted ids and
// normalized weights. `scale` multiplies the final weights.
int MoeRoute(const float* logits, const float* bias, float scale, int* ids,
             float* weights, int experts, int used, bool normalize,
             hipStream_t stream, int tokens = 1);

int SituActHalf(const float* g, const float* u, __half* out, std::size_t n,
                float beta, float linear_beta, hipStream_t stream);
int ToHalf(const float* x, __half* out, std::size_t n, hipStream_t stream);
/// out[t][i] = sum_r weights[t][r] * parts[t][r][i].
int MoeCombineHalf(const __half* parts, const float* weights, float* out, int n,
                   int used, int tokens, hipStream_t stream);
/// Fused RDA over `n` tokens (conv, L2 norm, decay, delta recurrence, gated
/// norm); raw q/k/v, gproj, g2 are [n][heads*128], beta/ratio [n][heads].
/// `hist` is [3][channels][3] (oldest first), `state` [heads][128][128].
int RdaPrefill(const float* wq, const float* wk, const float* wv,
               const float* rawq, const float* rawk, const float* rawv,
               float* hist, const float* gproj, const float* dt,
               const float* sa, const float* beta, const float* ratio,
               const float* g2, const float* norm_w, float* state, float* out,
               int n, int heads, float lower_bound, float eps,
               hipStream_t stream);
/// y[t][r] = sum_k w[r][k] * x[t][k] in F32 (tiled, for wide batches).
int F32Gemm(const float* w, const float* x, float* y, int m, int n, int k,
            hipStream_t stream);
/// Copies seven float arrays in one launch.
int RecordOperands(const float* const src[7], float* const dst[7],
                   const int count[7], hipStream_t stream);
/// Device-side routed tile maps (48- and 64-row tiles) from expert counts;
/// unused entries hold the empty-tile sentinel.
int BuildTiles(const std::uint32_t* counts, int experts, std::int32_t* t48,
               int cap48, std::int32_t* t64, int cap64, std::int32_t* t128,
               int cap128, hipStream_t stream);
/// Stalls the stream until *flag (pinned host memory) reaches `epoch`.
int WaitFlag(const int* flag, int epoch, hipStream_t stream);
/// F32GemmSmall: y[t][r] = w[r] . x[t] for n <= 16 tokens, one block per row.
int F32SmallGemm(const float* w, const float* x, float* y, int m, int n, int k,
                 hipStream_t stream);
/// kv rows [n][stride]: lat = rmsnorm(kv[:rank]) * w, extra =
/// kv[rank:rank+rope].
int KvNorm(const float* kv, const float* w, float* lat, float* extra, int n,
           int stride, int rank, int rope, float eps, hipStream_t stream);
/// dst[t][s][h] = src[t][h] for the four residual streams.
int ReplicateStreams(const float* src, float* dst, int tokens, int h,
                     hipStream_t stream);
/// Causal WMMA attention over per-head expanded K [s][40][192], V [s][40][128]
/// and Q [t][40][192] (all F16); out F32 [t][40][128]. Key s is visible to
/// query t when s <= pos0 + t.
int MlaWmmaAttend(const __half* q16, const __half* k16, const __half* v16,
                  float* out, int n, int pos0, hipStream_t stream,
                  const int* kpos = nullptr, int nk = 0);
/// Block-sparse long-context attention helpers (opt-in, approximate).
int PoolUpdate(const float* lat, float* pool, int first_block, int blocks,
               int total_rows, hipStream_t stream);
int QNopeMean(const float* q, float* qm, int n, int heads, int nope, int rope,
              hipStream_t stream);
int SparseSelect(const float* mla_q, float* qm, const float* qabs,
                 const float* pool, float* scores, int* sel, int* kpos, int n,
                 int first_block, int score_end, int k, int sink_rows,
                 int local_start, int pos0, int qgroups, hipStream_t stream);
int GatherKvFloat(const float* lat, const float* extra, const int* kpos,
                  int cnt, float* lat_g, float* extra_g, hipStream_t stream);
int GatherKv(const float* lat, const float* extra, const int* kpos, int cnt,
             __half* lat16, float* extra_g, hipStream_t stream);
int BuildKvHalf(const float* knope, const float* extra, __half* k16,
                std::size_t rows, hipStream_t stream);
/// kb [40*512][128] F16 -> kbt [40*128][512].
int TransposeKb(const __half* kb, __half* kbt, hipStream_t stream);
/// Lossless Q5_0 -> Q8_0 re-encode of `blocks` 32-element blocks.
int Q5ToQ8(const void* src, void* dst, std::size_t blocks, hipStream_t stream);

// SiTU activation: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb).
int SituAct(const float* g, const float* u, float* out, int n, float beta,
            float linear_beta, hipStream_t stream);

// Weighted sum of per-expert outputs: out[c] = sum_i w[i]*parts[i][c].
int MoeCombine(const float* parts, const float* weights, float* out, int n,
               int used, hipStream_t stream);

// out[i] = x[i] * gate[i].
int VectorMul(const float* x, const float* gate, float* out, int n,
              hipStream_t stream);

// out[i] = x[i] + y[i].
int AddVectors(const float* x, const float* y, float* out, int n,
               hipStream_t stream);

// Gather the RoPE half of each MLA query into a contiguous [heads][rope].
int MlaQeGather(const float* q, float* qe, float* qn, int heads, int nope,
                int rope, hipStream_t stream, int tokens = 1);

// out[i] = in[i] + scale * steer[i] (effort vector).
int SteerAdd(float* x, const float* steer, float scale, int n,
             hipStream_t stream);

// y[i] = mean over `streams` of x[s][i], then RMSNorm.
int HeadMeanNorm(const float* x, const float* w, float* y, int h, int streams,
                 float eps, hipStream_t stream);

// Greedy token of each logits row.
int ArgmaxRows(const float* logits, int rows, int vocab, int* out,
               hipStream_t stream);

}  // namespace gufo::models::raon1::rocm

#endif  // GUFO_MODELS_RAON1_KERNELS_ROCM_RAON_HPP_
