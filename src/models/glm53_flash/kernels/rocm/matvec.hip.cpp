#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"

#include "ds4_mmq.h"
#include "mmvq.hip.hpp"
#include "quantize.hip.hpp"

// The vendored mmq tier asks the DS4 backend for producer-folded activations.
// GLM never registers any; the weak definition yields to the real one when the
// DS4 backend is linked into the same binary.
extern "C" __attribute__((weak)) int ds4_hip_q8_fold_take_q81(
    const void*, std::uint64_t, const void** q81) {
  *q81 = nullptr;
  return 0;
}

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kF32Threads = 256;

__global__ void MatvecF32Kernel(const float* __restrict__ w,
                                const float* __restrict__ x,
                                float* __restrict__ out, int m, int n, int k) {
  const int row = blockIdx.x;
  const int col = blockIdx.y;
  const float* wr = w + static_cast<std::size_t>(row) * k;
  const float* xc = x + static_cast<std::size_t>(col) * k;
  float acc = 0.0F;
  for (int i = threadIdx.x; i < k; i += kF32Threads) {
    acc += wr[i] * xc[i];
  }
  __shared__ float partial[kF32Threads];
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kF32Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[static_cast<std::size_t>(col) * m + row] = partial[0];
  }
}

// Routed-expert matvec for a handful of tokens. One lane owns a whole
// 256-weight block of a row (16-byte loads, no re-read of the activation per
// row): the quantized activation row of the slot's token sits in LDS,
// interleaved so the lanes of a row read consecutive words, and each wave
// covers two or four rows. Dot products are the same dp4a sums as the mmvq
// tier.
constexpr int kExpertThreads = 256;
constexpr int kExpertWaves = kExpertThreads / 32;

using Unaligned16 = uint4 __attribute__((aligned(2)));
using Unaligned8 = uint2 __attribute__((aligned(2)));

// Negates byte i of g where bit i of nib is set. The grid magnitudes are
// below 128 and never zero, so the per-byte two's complement cannot carry.
__device__ __forceinline__ int ApplySignNibble(uint32_t g, uint32_t nib) {
  const uint32_t spread = __umul24(nib, 0x00204081U) & 0x01010101U;
  const uint32_t mask = __builtin_amdgcn_perm(0U, 0U, 0x0C0C0C0CU + spread);
  return static_cast<int>((g ^ mask) + spread);
}

template<int KB>
struct ExpertRows {
  static constexpr int kPerWave = 32 / KB;
  static constexpr int kPerBlock = kExpertWaves * kPerWave;
};

template<int KB>
__device__ __forceinline__ void StageActivation(
    const block_q8_1* __restrict__ y, int* __restrict__ yq,
    float* __restrict__ yd) {
  constexpr int kBlocks = KB * 8;
  for (int idx = threadIdx.x; idx < kBlocks * 8; idx += kExpertThreads) {
    const int q = idx >> 3;
    const int wd = idx & 7;
    const int b = q >> 3;
    const int ib = q & 7;
    yq[(ib * 8 + wd) * KB + b] = reinterpret_cast<const int*>(y[q].qs)[wd];
    if (wd == 0) {
      yd[ib * KB + b] = __low2float(y[q].ds);
    }
  }
}

template<int KB>
__device__ __forceinline__ float Iq2xxsBlockDot(const uint4 (&q)[4], float d,
                                                const int* yq, const float* yd,
                                                int b) {
  const int words[16] = {static_cast<int>(q[0].x), static_cast<int>(q[0].y),
                         static_cast<int>(q[0].z), static_cast<int>(q[0].w),
                         static_cast<int>(q[1].x), static_cast<int>(q[1].y),
                         static_cast<int>(q[1].z), static_cast<int>(q[1].w),
                         static_cast<int>(q[2].x), static_cast<int>(q[2].y),
                         static_cast<int>(q[2].z), static_cast<int>(q[2].w),
                         static_cast<int>(q[3].x), static_cast<int>(q[3].y),
                         static_cast<int>(q[3].z), static_cast<int>(q[3].w)};
  float acc = 0.0F;
#pragma unroll
  for (int ib = 0; ib < 8; ++ib) {
    const uint32_t idx_word = words[2 * ib];
    const uint32_t aux32 = words[2 * ib + 1];
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
      const uint2 grid_pos = reinterpret_cast<const uint2*>(
          iq2xxs_grid)[(idx_word >> (4 * k0)) & 0xFFU];
      const uint32_t s7 = (aux32 >> (7 * k0 / 2)) & 127U;
      const uint32_t s8 = s7 | ((__popc(s7) & 1U) << 7);
      const int grid0 = ApplySignNibble(grid_pos.x, s8 & 15U);
      sumi = ds4_ggml_hip_dp4a(grid0, yq[(ib * 8 + k0) * KB + b], sumi);
      const int grid1 = ApplySignNibble(grid_pos.y, s8 >> 4);
      sumi = ds4_ggml_hip_dp4a(grid1, yq[(ib * 8 + k0 + 1) * KB + b], sumi);
    }
    const int ls = static_cast<int>(aux32 >> 27 | 1U);
    sumi = sumi * ls / 8;
    acc += d * yd[ib * KB + b] * static_cast<float>(sumi);
  }
  return acc;
}

template<int KB>
__device__ __forceinline__ float Iq2sBlockDot(const uint4 (&q)[4],
                                              const uint2 (&tail)[2], float d,
                                              const int* yq, const float* yd,
                                              int b) {
  const uint32_t qs_words[16] = {q[0].x, q[0].y, q[0].z, q[0].w, q[1].x, q[1].y,
                                 q[1].z, q[1].w, q[2].x, q[2].y, q[2].z, q[2].w,
                                 q[3].x, q[3].y, q[3].z, q[3].w};
  const uint32_t qh_words[2] = {tail[0].x, tail[0].y};
  const uint32_t scale_words[2] = {tail[1].x, tail[1].y};
  float acc = 0.0F;
#pragma unroll
  for (int ib = 0; ib < 8; ++ib) {
    const uint32_t idx_word = qs_words[ib];
    const uint32_t sign_word = qs_words[8 + ib];
    const int qh = (qh_words[ib >> 2] >> (8 * (ib & 3))) & 0xFF;
    const int scales = (scale_words[ib >> 2] >> (8 * (ib & 3))) & 0xFF;
    const int ls0 = scales & 0x0F;
    const int ls1 = scales >> 4;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
      const int* grid_pos = reinterpret_cast<const int*>(
          iq2s_grid +
          (((idx_word >> (4 * l0)) & 0xFFU) | ((qh << (8 - l0)) & 0x300)));
      const uint32_t sgn = (sign_word >> (4 * l0)) & 0xFFU;
      const int grid_l = ApplySignNibble(grid_pos[0], sgn & 15U);
      const int grid_h = ApplySignNibble(grid_pos[1], sgn >> 4);
      const int u0 = yq[(ib * 8 + l0) * KB + b];
      const int u1 = yq[(ib * 8 + l0 + 1) * KB + b];
      if (l0 < 4) {
        sumi0 = ds4_ggml_hip_dp4a(grid_l, u0, sumi0);
        sumi0 = ds4_ggml_hip_dp4a(grid_h, u1, sumi0);
      } else {
        sumi1 = ds4_ggml_hip_dp4a(grid_l, u0, sumi1);
        sumi1 = ds4_ggml_hip_dp4a(grid_h, u1, sumi1);
      }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    acc += d * yd[ib * KB + b] * static_cast<float>(sumi);
  }
  return acc;
}

template<bool kS, int KB>
__global__ __launch_bounds__(kExpertThreads) void ExpertsDotKernel(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
    size_t expert_bytes, size_t row_bytes, const int32_t* __restrict__ ids,
    const block_q8_1* __restrict__ y, int n_used, int y_rows_per_token,
    float* __restrict__ out0, float* __restrict__ out1, int slots, int m) {
  constexpr int kBlockBytes = kS ? 82 : 66;
  using Rows = ExpertRows<KB>;
  __shared__ int yq[KB * 64];
  __shared__ float yd[KB * 8];
  const bool second = static_cast<int>(blockIdx.y) >= slots;
  const int slot = second ? static_cast<int>(blockIdx.y) - slots
                          : static_cast<int>(blockIdx.y);
  const uint8_t* __restrict__ w = second ? w1 : w0;
  float* __restrict__ out = second ? out1 : out0;
  const int token = slot / n_used;
  const int yrow = y_rows_per_token == 1 ? slot : token;
  const int expert = ids[slot];
  StageActivation<KB>(y + static_cast<size_t>(yrow) * KB * 8, yq, yd);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int b = lane % KB;
  const int sub_row = lane / KB;
  const int row =
      blockIdx.x * Rows::kPerBlock + wave * Rows::kPerWave + sub_row;
  const uint8_t* p =
      w + static_cast<size_t>(expert < 0 ? 0 : expert) * expert_bytes +
      static_cast<size_t>(row < m ? row : m - 1) * row_bytes +
      static_cast<size_t>(b) * kBlockBytes;
  const float d =
      __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(p)));
  uint4 q[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = *reinterpret_cast<const Unaligned16*>(p + 2 + 16 * i);
  }
  uint2 tail[2] = {};
  if constexpr (kS) {
    tail[0] = *reinterpret_cast<const Unaligned8*>(p + 66);
    tail[1] = *reinterpret_cast<const Unaligned8*>(p + 74);
  }
  float v = kS ? Iq2sBlockDot<KB>(q, tail, d, yq, yd, b)
               : Iq2xxsBlockDot<KB>(q, d, yq, yd, b);
#pragma unroll
  for (int off = KB / 2; off > 0; off >>= 1) {
    v += __shfl_xor(v, off, 32);
  }
  if (b == 0 && row < m) {
    out[static_cast<size_t>(slot) * m + row] = expert < 0 ? 0.0F : v;
  }
}

template<bool kS, int KB>
int LaunchExpertsDot(const void* w0, const void* w1, int m, int n_experts,
                     int n_used, int slots, int groups, const int* ids,
                     const void* y, int y_rows_per_token, float* out0,
                     float* out1, hipStream_t stream) {
  constexpr size_t kBlockBytes = kS ? 82 : 66;
  const size_t row_bytes = KB * kBlockBytes;
  const size_t expert_bytes = static_cast<size_t>(m) * row_bytes;
  (void)n_experts;
  const dim3 grid(
      (m + ExpertRows<KB>::kPerBlock - 1) / ExpertRows<KB>::kPerBlock,
      slots * groups);
  ExpertsDotKernel<kS, KB><<<grid, kExpertThreads, 0, stream>>>(
      static_cast<const uint8_t*>(w0), static_cast<const uint8_t*>(w1),
      expert_bytes, row_bytes, ids, static_cast<const block_q8_1*>(y), n_used,
      y_rows_per_token, out0, out1, slots, m);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

}  // namespace

std::size_t MatvecScratchBytes(int n, int k) {
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  return static_cast<std::size_t>(n) * padded * sizeof(block_q8_1) / QK8_1;
}

int Matvec(int type_code, const void* w, const float* x, float* out, int m,
           int n, int k, void* scratch, hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || m <= 0 || n <= 0 ||
      k <= 0) {
    return -1;
  }
  if (type_code == kTypeF32) {
    MatvecF32Kernel<<<dim3(m, n), kF32Threads, 0, stream>>>(
        static_cast<const float*>(w), x, out, m, n, k);
    return hipGetLastError() == hipSuccess ? 0 : -2;
  }
  if (scratch == nullptr || n > MMVQ_MAX_BATCH_SIZE) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(
      x, nullptr, scratch, type, k, k, static_cast<std::int64_t>(k) * n,
      static_cast<std::int64_t>(k) * n, padded, n, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int stride_col_y = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, scratch, nullptr, fusion, out, k, m, n,
                            stride_row_x, stride_col_y, m, 1, 1, 1, 0,
                            n * stride_col_y, 0, 1, 1, 0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int QuantizeActivations(const float* x, int n, int k, void* q,
                        hipStream_t stream) {
  if (x == nullptr || q == nullptr || n <= 0 || k <= 0 || n > MMVQ_MAX_BATCH_SIZE)
    return -1;
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(
      x, nullptr, q, GGML_TYPE_Q6_K, k, k, static_cast<std::int64_t>(k) * n,
      static_cast<std::int64_t>(k) * n, padded, n, 1, 1, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int MatvecQuantized(int type_code, const void* w, const void* q, float* out,
                    int m, int n, int k, hipStream_t stream) {
  if (w == nullptr || q == nullptr || out == nullptr || m <= 0 || n <= 0 ||
      k <= 0 || type_code == kTypeF32 || n > MMVQ_MAX_BATCH_SIZE)
    return -1;
  const auto type = static_cast<ggml_type>(type_code);
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int stride_col_y = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, q, nullptr, fusion, out, k, m, n,
                            stride_row_x, stride_col_y, m, 1, 1, 1, 0,
                            n * stride_col_y, 0, 1, 1, 0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int MatvecBatched(int type_code, const void* w, const float* x, float* out,
                  int m, int k, int channels, void* scratch,
                  hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || scratch == nullptr ||
      type_code == kTypeF32 || m <= 0 || k <= 0 || channels <= 0) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(
      x, nullptr, scratch, type, k, k, static_cast<std::int64_t>(k) * channels,
      static_cast<std::int64_t>(k) * channels, padded, channels, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int stride_col_y = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, scratch, nullptr, fusion, out, k, m, 1,
                            stride_row_x, stride_col_y, m, channels, channels,
                            channels, m * stride_row_x, stride_col_y, m, 1, 1,
                            0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int MatvecBatchedTokens(int type_code, const void* w, const float* x,
                        float* out, int m, int k, int channels, int tokens,
                        void* scratch, hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || scratch == nullptr ||
      type_code == kTypeF32 || m <= 0 || k <= 0 || channels <= 0 ||
      tokens <= 0 || tokens > MMVQ_MAX_BATCH_SIZE) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  const std::int64_t rows = static_cast<std::int64_t>(channels) * tokens;
  ds4_quantize_row_q8_1_hip(
      x, nullptr, scratch, type, k, k, static_cast<std::int64_t>(k) * rows,
      static_cast<std::int64_t>(k) * rows, padded, rows, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int row_blocks = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(
      w, type, scratch, nullptr, fusion, out, k, m, tokens, stride_row_x,
      channels * row_blocks, channels * m, channels, channels, channels,
      m * stride_row_x, row_blocks, m, 1, 1, 0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int MatvecExpertsTokens(int type_code, const void* w, const float* x,
                        float* out, int m, int k, int n_experts, int n_used,
                        int n_tokens, const int* ids, bool x_per_expert,
                        void* scratch, hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || scratch == nullptr ||
      ids == nullptr || type_code == kTypeF32 || m <= 0 || k <= 0 ||
      n_used <= 0 || n_experts <= 0 || n_tokens <= 0 || n_tokens > 8) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const int rows_y = x_per_expert ? n_tokens * n_used : n_tokens;
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(
      x, nullptr, scratch, type, k, k, static_cast<std::int64_t>(k) * rows_y,
      static_cast<std::int64_t>(k) * rows_y, padded, rows_y, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int row_blocks = static_cast<int>(padded / QK8_1);
  const int channels_y = x_per_expert ? n_used : 1;
  const int stride_col_y = x_per_expert ? n_used * row_blocks : row_blocks;
  const bool xxs = type_code == kTypeIQ2_XXS;
  if ((xxs || type_code == kTypeIQ2_S) && (k == 4096 || k == 2048)) {
    const int slots = n_tokens * n_used;
    const int per_token = x_per_expert ? 1 : 0;
    if (xxs) {
      return k == 4096
                 ? LaunchExpertsDot<false, 16>(w, w, m, n_experts, n_used,
                                               slots, 1, ids, scratch,
                                               per_token, out, out, stream)
                 : LaunchExpertsDot<false, 8>(w, w, m, n_experts, n_used, slots,
                                              1, ids, scratch, per_token, out,
                                              out, stream);
    }
    return k == 4096 ? LaunchExpertsDot<true, 16>(w, w, m, n_experts, n_used,
                                                  slots, 1, ids, scratch,
                                                  per_token, out, out, stream)
                     : LaunchExpertsDot<true, 8>(w, w, m, n_experts, n_used,
                                                 slots, 1, ids, scratch,
                                                 per_token, out, out, stream);
  }
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, scratch, ids, fusion, out, k, m, n_tokens,
                            stride_row_x, stride_col_y, n_used * m, n_experts,
                            channels_y, n_used, m * stride_row_x, row_blocks, m,
                            1, 1, 0, 0, 0, n_used, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int MatvecExpertsPairTokens(int type_code, const void* w0, const void* w1,
                            const float* x, float* out0, float* out1, int m,
                            int k, int n_experts, int n_used, int n_tokens,
                            const int* ids, void* scratch, hipStream_t stream) {
  const bool xxs = type_code == kTypeIQ2_XXS;
  const bool fast =
      (xxs || type_code == kTypeIQ2_S) && (k == 4096 || k == 2048);
  if (!fast || w0 == nullptr || w1 == nullptr || x == nullptr ||
      out0 == nullptr || out1 == nullptr || scratch == nullptr ||
      ids == nullptr || m <= 0 || n_used <= 0 || n_experts <= 0 ||
      n_tokens <= 0 || n_tokens > 8) {
    const int rc =
        MatvecExpertsTokens(type_code, w0, x, out0, m, k, n_experts, n_used,
                            n_tokens, ids, false, scratch, stream);
    return rc != 0 ? rc
                   : MatvecExpertsTokens(type_code, w1, x, out1, m, k,
                                         n_experts, n_used, n_tokens, ids,
                                         false, scratch, stream);
  }
  const auto type = static_cast<ggml_type>(type_code);
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(
      x, nullptr, scratch, type, k, k, static_cast<std::int64_t>(k) * n_tokens,
      static_cast<std::int64_t>(k) * n_tokens, padded, n_tokens, 1, 1, stream);
  const int slots = n_tokens * n_used;
  if (xxs) {
    return k == 4096 ? LaunchExpertsDot<false, 16>(w0, w1, m, n_experts, n_used,
                                                   slots, 2, ids, scratch, 0,
                                                   out0, out1, stream)
                     : LaunchExpertsDot<false, 8>(w0, w1, m, n_experts, n_used,
                                                  slots, 2, ids, scratch, 0,
                                                  out0, out1, stream);
  }
  return k == 4096
             ? LaunchExpertsDot<true, 16>(w0, w1, m, n_experts, n_used, slots,
                                          2, ids, scratch, 0, out0, out1,
                                          stream)
             : LaunchExpertsDot<true, 8>(w0, w1, m, n_experts, n_used, slots, 2,
                                         ids, scratch, 0, out0, out1, stream);
}

int MatvecExperts(int type_code, const void* w, const float* x, float* out,
                  int m, int k, int n_experts, int n_used, const int* ids,
                  bool x_per_expert, void* scratch, hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || scratch == nullptr ||
      ids == nullptr || type_code == kTypeF32 || m <= 0 || k <= 0 ||
      n_used <= 0 || n_experts <= 0) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const int channels_y = x_per_expert ? n_used : 1;
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(x, nullptr, scratch, type, k, k,
                            static_cast<std::int64_t>(k) * channels_y,
                            static_cast<std::int64_t>(k) * channels_y, padded,
                            channels_y, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int stride_col_y = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, scratch, ids, fusion, out, k, m, 1,
                            stride_row_x, stride_col_y, m, n_experts,
                            channels_y, n_used, m * stride_row_x, stride_col_y,
                            m, 1, 1, 0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int Gemm(int type_code, const void* w, const float* x, float* out, int m, int n,
         int k, void* scratch, hipStream_t stream) {
  if (type_code == kTypeF32 || n <= MMVQ_MAX_BATCH_SIZE) {
    return Matvec(type_code, w, x, out, m, n, k, scratch, stream);
  }
  if ((k % 256) != 0 || (type_code != kTypeQ6_K && type_code != kTypeQ8_0 &&
                         type_code != kTypeQ4_K)) {
    for (int col = 0; col < n; col += MMVQ_MAX_BATCH_SIZE) {
      const int cols =
          n - col < MMVQ_MAX_BATCH_SIZE ? n - col : MMVQ_MAX_BATCH_SIZE;
      const int rc = Matvec(type_code, w, x + static_cast<std::size_t>(col) * k,
                            out + static_cast<std::size_t>(col) * m, m, cols, k,
                            scratch, stream);
      if (rc != 0) {
        return rc;
      }
    }
    return 0;
  }
  switch (type_code) {
    case kTypeQ6_K:
      return ds4_mmq_q6_K_dense(w, x, out, m, n, k, stream);
    case kTypeQ8_0:
      return ds4_mmq_q8_0_dense(w, x, out, m, n, k, stream);
    case kTypeQ4_K:
      return ds4_mmq_q4_K_dense(w, x, out, m, n, k, stream);
    default:
      return -1;
  }
}

int ExpertsGemm(int type_code, const void* w, const float* x, float* out, int m,
                int k, int tokens, int n_experts, int n_used, const int* ids,
                hipStream_t stream) {
  switch (type_code) {
    case kTypeIQ2_XXS:
      return ds4_mmq_iq2_xxs_moe(w, x, ids, out, m, k, tokens, n_experts,
                                 n_used, stream);
    case kTypeIQ2_S:
      return ds4_mmq_iq2_s_moe(w, x, ids, out, m, k, tokens, n_experts, n_used,
                               stream);
    default:
      return -1;
  }
}

}  // namespace gufo::models::glm53_flash::rocm
