#ifndef GUFO_MODELS_LING32_GPU_KERNELS_HPP_
#define GUFO_MODELS_LING32_GPU_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::ling32::gpu {

void AddTo(float* x, const float* y, std::uint32_t n, hipStream_t stream);

void F32GemvDouble(const float* w, const float* x, float* out,
                   std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

void DequantRows(const void* src, float* dst, std::uint32_t elements, int type,
                 hipStream_t stream);

void DequantExperts(const void* src, const std::int32_t* ids, float* dst,
                    std::uint32_t slots, std::uint32_t elements_per_slot,
                    std::uint32_t expert_bytes, int type, hipStream_t stream);

void RopeHeads(const float* in, float* out, std::uint32_t heads,
               std::uint32_t stride, std::uint32_t offset, std::uint32_t dim,
               std::uint32_t pos, float base, hipStream_t stream,
               std::uint32_t out_stride = 0,
               const std::int32_t* pos_ptr = nullptr,
               std::uint32_t out_pos_stride = 0);

void CopyRow(const float* src, float* dst, std::uint32_t n, hipStream_t stream);
void AppendRow(const float* row, float* cache, std::uint32_t pos,
               std::uint32_t dim, hipStream_t stream);

void KdaConv(const float* q, const float* k, const float* v, const float* cq,
             const float* ck, const float* cv, float* history, float* oq,
             float* ok, float* ov, std::uint32_t channels, std::uint32_t taps,
             hipStream_t stream);

void KdaStep(const float* q, const float* k, const float* v, const float* fa,
             const float* ga, const float* beta, const float* dt,
             const float* sa, const float* norm_w, float* state, float* out,
             std::uint32_t heads, std::uint32_t dim, float lower_bound,
             float eps, hipStream_t stream);

void MlaScores(const float* q, const float* cache, float* scores,
               std::uint32_t heads, std::uint32_t rows, std::uint32_t dim,
               float scale, hipStream_t stream);

void MlaGather(const float* scores, const float* cache, float* ctx,
               std::uint32_t heads, std::uint32_t rows, std::uint32_t dim,
               std::uint32_t value_dim, hipStream_t stream);

void MlaGate(const float* value, const float* gate, float* out,
             std::uint32_t heads, std::uint32_t head_dim, hipStream_t stream);

void KdaDecode(const float* q, const float* k, const float* v, const float* fa,
               const float* ga, const float* beta, const float* dt,
               const float* sa, const float* norm_w, float* state, float* so,
               float* out, std::uint32_t heads, float lower_bound, float eps,
               hipStream_t stream);

/// F32 [rows][k] -> F16 [rows][3k] so one F16 GEMM with K'=3k reproduces the
/// F32 product: activations become [hi | lo | hi], weights [hi | hi | lo].
void SplitHiLo(const float* in, __half* out, std::size_t rows, std::uint32_t k,
               bool weight, hipStream_t stream);
void NarrowRows(const float* in, __half* out, std::size_t count,
                hipStream_t stream);
void SwigluHalf(const float* gate, const float* up, __half* out, std::size_t n,
                float limit, hipStream_t stream);
void MoeCombineBatch(const __half* parts, const float* weights,
                     const float* shared, float* out, std::uint32_t tokens,
                     std::uint32_t used, std::uint32_t hidden,
                     hipStream_t stream);
void EncodeQ8(const void* src, void* dst, std::size_t blocks, int type,
              hipStream_t stream);
void KdaConvBatch(const float* q, const float* k, const float* v,
                  const float* cq, const float* ck, const float* cv,
                  float* history, float* oq, float* ok, float* ov,
                  std::uint32_t channels, std::uint32_t n, hipStream_t stream);
void KdaRecurBatch(const float* q, const float* k, const float* v,
                   const float* fa, const float* ga, const float* beta,
                   const float* dt, const float* sa, const float* norm_w,
                   float* state, float* so, float* out, std::uint32_t n,
                   std::uint32_t heads, float lower_bound, float eps,
                   hipStream_t stream, float* ckpt = nullptr);
void KdaHistRestore(const float* old_hist, const float* raw, float* hist,
                    std::uint32_t channels, std::uint32_t a,
                    hipStream_t stream);
void MlaKvPrep(const float* kvall, const float* norm_w, float* cache,
               std::uint32_t pos0, std::uint32_t n, float eps, float base,
               hipStream_t stream);
void RopeQBatch(float* q, std::uint32_t n, std::uint32_t heads,
                std::uint32_t stride, std::uint32_t offset, std::uint32_t pos0,
                float base, hipStream_t stream);
void LatentToHalf(const float* cache, __half* out, std::size_t rows,
                  hipStream_t stream);
void BuildKvbHalf(const void* kb, const void* vb, __half* wk, __half* wv,
                  hipStream_t stream);
void MlaAttention(const float* q, const float* kn, const float* vv,
                  const float* cache, float* out, std::uint32_t n,
                  std::uint32_t pos0, float scale, hipStream_t stream);

void BuildTiles(const std::uint32_t* counts, std::uint32_t experts,
                std::int32_t* t48, std::uint32_t cap48, std::int32_t* t64,
                std::uint32_t cap64, std::int32_t* t128, std::uint32_t cap128,
                hipStream_t stream);

void BuildKvHalf(const float* kn, const float* vv, const float* cache,
                 __half* k16, __half* v16, std::size_t rows,
                 hipStream_t stream);
void MlaWmmaAttention(const __half* q16, const __half* k16, const __half* v16,
                      float* out, std::uint32_t n, std::uint32_t pos0,
                      float scale, hipStream_t stream);

void MoeCombineBatchF32(const float* parts, const float* weights,
                        const float* shared, float* out, std::uint32_t tokens,
                        std::uint32_t used, std::uint32_t hidden,
                        hipStream_t stream);

constexpr std::uint32_t kPoolBlockRows = 64;
constexpr std::uint32_t kSparseBlocks = 96;
constexpr std::uint32_t kSparseRecent = 4;
void PoolUpdate(const float* cache, float* pool, std::uint32_t first_block,
                std::uint32_t count, hipStream_t stream);
void SparseSelect(const float* pool, const float* qvec, float* scores,
                  std::int32_t* sel, std::uint32_t nb, std::uint32_t budget,
                  std::uint32_t recent, hipStream_t stream);
void GatherLatents(const float* cache, const std::int32_t* sel, float* out,
                   std::uint32_t budget, std::uint32_t tail_start,
                   std::uint32_t total_rows, hipStream_t stream);
void ProbeRope(const float* qrow, float* qvec, hipStream_t stream);

void SetI32(std::int32_t* dst, std::int32_t v, hipStream_t stream);

void SgemmF32(const float* w, const float* x, float* out, std::uint32_t m,
              std::uint32_t n, std::uint32_t k, hipStream_t stream);

void HeadGemvQ8(const void* w, const float* x, float* out, std::uint32_t heads,
                std::uint32_t rows, std::uint32_t cols, std::uint32_t x_stride,
                std::uint32_t out_stride, hipStream_t stream);

constexpr std::uint32_t kMlaMaxSplitCount = 128;
void MlaDecode(const float* qvec, const float* cache, float* part_m,
               float* part_l, float* part_acc, float* out,
               const std::int32_t* pos_ptr, float scale, hipStream_t stream);
void StoreRow(const float* src, float* cache, const std::int32_t* pos_ptr,
              std::uint32_t n, std::uint32_t dim, hipStream_t stream);

void RouterLing(const float* logits, const float* bias, std::int32_t* ids,
                float* weights, std::uint32_t experts, std::uint32_t groups,
                std::uint32_t groups_used, std::uint32_t used, float scale,
                hipStream_t stream, std::uint32_t tokens = 1);

void SwigluClamp(const float* gate, const float* up, float* out,
                 std::uint32_t n, float limit, hipStream_t stream);
void SwigluPlain(const float* gate, const float* up, float* out,
                 std::uint32_t n, hipStream_t stream);

void MoeCombine(const float* parts, const float* weights, const float* shared,
                float* out, std::uint32_t used, std::uint32_t hidden,
                hipStream_t stream);

}  // namespace gufo::models::ling32::gpu

#endif  // GUFO_MODELS_LING32_GPU_KERNELS_HPP_
