#include "src/models/inkling/gpu/gptq.hpp"

#include <hip/hip_fp16.h>
#include <rocsolver/rocsolver.h>

#include <cstdio>
#include <cstring>

namespace gufo::models::inkling::gptq {
namespace {

constexpr int kRowsPerBlock = 64;

__device__ __forceinline__ float Ue4m3(unsigned b) {
  const unsigned e = (b >> 3) & 15U, m = b & 7U;
  if (b == 0x7F)
    return 0.0f;
  return e == 0 ? static_cast<float>(m) * (1.0f / 512.0f)
                : (1.0f + static_cast<float>(m) * 0.125f) *
                      exp2f(static_cast<float>(e) - 7.0f);
}

__device__ __forceinline__ void Rot256(float* v, const float* s1,
                                       const float* s2) {
  const int t = threadIdx.x;
  v[t] *= s1[t];
  for (int len = 1; len < 256; len <<= 1) {
    __syncthreads();
    if (t < 128) {
      const int i = (t / len) * 2 * len + (t % len);
      const float a = v[i], b = v[i + len];
      v[i] = a + b;
      v[i + len] = a - b;
    }
  }
  __syncthreads();
  v[t] *= s2[t] * (1.0f / 16.0f);
  __syncthreads();
}

__global__ void __launch_bounds__(256)
    RotWeightsKernel(const std::uint8_t* __restrict__ src,
                     std::size_t src_row_bytes, float* __restrict__ out,
                     int k_src, int k, const float* __restrict__ signs) {
  __shared__ float v[256];
  __shared__ float s1[256], s2[256];
  const int t = threadIdx.x;
  s1[t] = signs[t];
  s2[t] = signs[256 + t];
  const std::size_t row = blockIdx.x;
  const std::uint8_t* sr = src + row * src_row_bytes;
  constexpr float kMags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int kb = 0; kb < k / 256; ++kb) {
    const int j = kb * 8 + (t >> 5);
    const int p = t & 31;
    const int byte = (p < 8) ? p : (p < 16) ? p - 8 : (p < 24) ? 8 + p - 16
                                                              : 8 + p - 24;
    float val = 0.0f;
    if (kb * 256 + t < k_src) {
      const unsigned raw = sr[j * 16 + byte];
      const unsigned nb =
          (p < 8 || (p >= 16 && p < 24)) ? (raw & 15U) : (raw >> 4);
      const float mag = kMags[nb & 7U];
      const float sc = Ue4m3(sr[k_src / 2 + j * 2 + (p >> 4)]);
      val = ((nb & 8U) ? -mag : mag) * sc;
    }
    v[t] = val;
    __syncthreads();
    Rot256(v, s1, s2);
    out[row * k + kb * 256 + t] = v[t];
    __syncthreads();
  }
}

__global__ void __launch_bounds__(256)
    RotRowsKernel(const __half* __restrict__ x,
                  const std::int32_t* __restrict__ rows_in,
                  const std::int32_t* __restrict__ rows_out, int k_src, int k,
                  float* __restrict__ out, const float* __restrict__ signs) {
  __shared__ float v[256];
  __shared__ float s1[256], s2[256];
  const int t = threadIdx.x;
  s1[t] = signs[t];
  s2[t] = signs[256 + t];
  const std::size_t c = blockIdx.x;
  const bool live = rows_out == nullptr || rows_out[c] >= 0;
  const std::size_t token =
      rows_in != nullptr ? static_cast<std::size_t>(rows_in[c]) : c;
  for (int kb = 0; kb < k / 256; ++kb) {
    const int col = kb * 256 + t;
    v[t] = live && col < k_src ? __half2float(x[token * k_src + col]) : 0.0f;
    __syncthreads();
    Rot256(v, s1, s2);
    out[c * k + col] = v[t];
    __syncthreads();
  }
}

__global__ void DampKernel(float* h, int k, int k_src, float damp) {
  __shared__ float red[1024];
  float s = 0.0f;
  for (int i = threadIdx.x; i < k_src; i += blockDim.x)
    s += h[static_cast<std::size_t>(i) * k + i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o)
      red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  const float add = damp * red[0] / static_cast<float>(k_src) + 1e-20f;
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    h[static_cast<std::size_t>(i) * k + i] += add;
}

__global__ void MirrorKernel(const float* __restrict__ lower,
                             float* __restrict__ full, int k) {
  const std::size_t i = blockIdx.x;
  for (int j = threadIdx.x; j < k; j += blockDim.x) {
    const std::size_t a = static_cast<std::size_t>(j) * k + i;  // (i, j)
    const std::size_t b = i * k + j;                            // (j, i)
    full[a] = i >= static_cast<std::size_t>(j) ? lower[a] : lower[b];
  }
}

__device__ float BestScale(const float* v) {
  float amax = 0.0f;
  for (int i = 0; i < kGroup; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  const float s0 = amax / 7.0f;
  float best_s = s0;
  float best_err = 3.0e38f;
  for (int ci = 0; ci < 8; ++ci) {
    const float sc = s0 * (1.0f - 0.04f * ci);
    float err = 0.0f;
    for (int i = 0; i < kGroup; ++i) {
      const float q =
          sc > 0.0f ? fminf(7.0f, fmaxf(-8.0f, rintf(v[i] / sc))) : 0.0f;
      const float e = v[i] - q * sc;
      err += e * e;
    }
    if (err < best_err) {
      best_err = err;
      best_s = sc;
    }
  }
  return __half2float(__float2half(best_s));
}

__global__ void ScaleKernel(const float* __restrict__ w, float* __restrict__ scale,
                            std::uint16_t* __restrict__ gscales, int rows, int k,
                            int b0, int g) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const float s = BestScale(w + static_cast<std::size_t>(row) * k + b0);
  scale[row] = s;
  gscales[static_cast<std::size_t>(row) * (k / kGroup) + g] =
      __half_as_ushort(__float2half(s));
}

__global__ void TransposeBlockKernel(const float* __restrict__ u,
                                     float* __restrict__ ub, int k, int b0) {
  for (int i = threadIdx.x; i < kGroup * kGroup; i += blockDim.x) {
    const int j = i / kGroup, c = i % kGroup;
    ub[i] = u[static_cast<std::size_t>(b0 + c) * k + b0 + j];  // U(b0+j, b0+c)
  }
}

__global__ void __launch_bounds__(kRowsPerBlock)
    BlockKernel(float* __restrict__ w, const float* __restrict__ ub,
                const float* __restrict__ scale, std::int8_t* __restrict__ codes,
                float* __restrict__ err, int rows, int k, int b0) {
  __shared__ float lw[kRowsPerBlock][kGroup + 1];
  const int t = threadIdx.x;
  const int row = blockIdx.x * kRowsPerBlock + t;
  const bool live = row < rows;
  if (live)
    for (int c = 0; c < kGroup; ++c)
      lw[t][c] = w[static_cast<std::size_t>(row) * k + b0 + c];
  const float s = live ? scale[row] : 0.0f;
  for (int j = 0; j < kGroup; ++j) {
    const float wv = lw[t][j];
    const float q = s > 0.0f ? fminf(7.0f, fmaxf(-8.0f, rintf(wv / s))) : 0.0f;
    const float e = (wv - q * s) / ub[j * kGroup + j];
    if (live) {
      codes[static_cast<std::size_t>(row) * k + b0 + j] =
          static_cast<std::int8_t>(q);
      err[static_cast<std::size_t>(row) * kGroup + j] = e;
    }
    for (int c = j + 1; c < kGroup; ++c)
      lw[t][c] -= e * ub[j * kGroup + c];
  }
}

__global__ void RtnKernel(const float* __restrict__ w,
                          std::int8_t* __restrict__ codes,
                          std::uint16_t* __restrict__ gscales, int rows, int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  for (int g = 0; g < k / kGroup; ++g) {
    const float* v = w + static_cast<std::size_t>(row) * k + g * kGroup;
    const float s = BestScale(v);
    gscales[static_cast<std::size_t>(row) * (k / kGroup) + g] =
        __half_as_ushort(__float2half(s));
    for (int i = 0; i < kGroup; ++i) {
      const float q = s > 0.0f ? fminf(7.0f, fmaxf(-8.0f, rintf(v[i] / s))) : 0.0f;
      codes[static_cast<std::size_t>(row) * k + g * kGroup + i] =
          static_cast<std::int8_t>(q);
    }
  }
}

__global__ void PackKernel(const std::int8_t* __restrict__ codes,
                           const std::uint16_t* __restrict__ gscales,
                           std::uint8_t* __restrict__ tiles,
                           std::uint16_t* __restrict__ scales, int rows, int k) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  const int groups = k / kGroup;
  if (idx >= static_cast<std::size_t>(groups) * rows * 4)
    return;
  const int q = static_cast<int>(idx & 3);
  const std::size_t r = idx >> 2;
  const int row = static_cast<int>(r % rows);
  const int ee = static_cast<int>(r / rows);
  const std::int8_t* c = codes + static_cast<std::size_t>(row) * k + ee * kGroup;
  std::uint8_t bytes[16];
  for (int b = 0; b < 16; ++b) {
    const int c0 = c[2 * (16 * q + b)] & 0xF;
    const int c1 = c[2 * (16 * q + b) + 1] & 0xF;
    bytes[b] = static_cast<std::uint8_t>(c0 | (c1 << 4));
  }
  uint4 w;
  memcpy(&w, bytes, 16);
  const std::size_t dst =
      ((static_cast<std::size_t>(ee) * (rows >> 4) + (row >> 4)) * 4 + q) *
          256 +
      (row & 15) * 16;
  *reinterpret_cast<uint4*>(tiles + dst) = w;
  if (q == 0)
    scales[static_cast<std::size_t>(ee) * rows + row] =
        gscales[static_cast<std::size_t>(row) * groups + ee];
}

}  // namespace

void RotateWeights(const std::uint8_t* nvfp4, std::size_t src_row_bytes,
                   float* out, int rows, int k_src, int k, const float* signs,
                   hipStream_t stream) {
  RotWeightsKernel<<<rows, 256, 0, stream>>>(nvfp4, src_row_bytes, out, k_src,
                                             k, signs);
}

void RotateRows(const __half* x, const std::int32_t* rows_in,
                const std::int32_t* rows_out, int n_rows, int k_src, int k,
                float* out, const float* signs, hipStream_t stream) {
  RotRowsKernel<<<n_rows, 256, 0, stream>>>(x, rows_in, rows_out, k_src, k, out,
                                            signs);
}

bool AccumulateHessian(rocblas_handle blas, const float* x, int m, int k,
                       float* h) {
  if (m <= 0)
    return true;
  const float one = 1.0f;
  return rocblas_ssyrk(blas, rocblas_fill_lower, rocblas_operation_none, k, m,
                       &one, x, k, &one, h, k) == rocblas_status_success;
}

bool AllocScratch(Scratch& s, int rows, int k) {
  s.rows = rows;
  s.k = k;
  const std::size_t rk = static_cast<std::size_t>(rows) * k;
  return hipMalloc(&s.w, rk * 4) == hipSuccess &&
         hipMalloc(&s.hinv, static_cast<std::size_t>(k) * k * 4) == hipSuccess &&
         hipMalloc(&s.ublock, kGroup * kGroup * 4) == hipSuccess &&
         hipMalloc(&s.err, static_cast<std::size_t>(rows) * kGroup * 4) ==
             hipSuccess &&
         hipMalloc(&s.scale, static_cast<std::size_t>(rows) * 4) == hipSuccess &&
         hipMalloc(&s.codes, rk) == hipSuccess &&
         hipMalloc(&s.gscales,
                   static_cast<std::size_t>(rows) * (k / kGroup) * 2) ==
             hipSuccess &&
         hipMalloc(&s.info, 4) == hipSuccess;
}

void FreeScratch(Scratch& s) {
  hipFree(s.w);
  hipFree(s.hinv);
  hipFree(s.ublock);
  hipFree(s.err);
  hipFree(s.scale);
  hipFree(s.codes);
  hipFree(s.gscales);
  hipFree(s.info);
  s = Scratch{};
}

bool Quantize(rocblas_handle blas, Scratch& s, const float* w_rot, float* h,
              int k_src, float damp, hipStream_t stream) {
  const int rows = s.rows, k = s.k;
  rocblas_set_stream(blas, stream);
  (void)hipMemcpyAsync(s.w, w_rot, static_cast<std::size_t>(rows) * k * 4,
                       hipMemcpyDeviceToDevice, stream);
  DampKernel<<<1, 1024, 0, stream>>>(h, k, k_src, damp);
  const auto check = [&](const char* what) {
    int info = 0;
    (void)hipMemcpyAsync(&info, s.info, 4, hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
    if (info != 0)
      std::fprintf(stderr, "gptq: %s failed (info %d)\n", what, info);
    return info == 0;
  };
  if (rocsolver_spotrf(blas, rocblas_fill_lower, k, h, k, s.info) !=
          rocblas_status_success ||
      !check("cholesky"))
    return false;
  if (rocsolver_spotri(blas, rocblas_fill_lower, k, h, k, s.info) !=
          rocblas_status_success ||
      !check("inverse"))
    return false;
  MirrorKernel<<<k, 256, 0, stream>>>(h, s.hinv, k);
  if (rocsolver_spotrf(blas, rocblas_fill_upper, k, s.hinv, k, s.info) !=
          rocblas_status_success ||
      !check("inverse cholesky"))
    return false;
  const int groups = k / kGroup;
  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  const float minus_one = -1.0f, one = 1.0f;
  for (int g = 0; g < groups; ++g) {
    const int b0 = g * kGroup;
    ScaleKernel<<<blocks, kRowsPerBlock, 0, stream>>>(s.w, s.scale, s.gscales,
                                                      rows, k, b0, g);
    TransposeBlockKernel<<<1, 1024, 0, stream>>>(s.hinv, s.ublock, k, b0);
    BlockKernel<<<blocks, kRowsPerBlock, 0, stream>>>(s.w, s.ublock, s.scale,
                                                      s.codes, s.err, rows, k, b0);
    const int b1 = b0 + kGroup;
    if (b1 < k &&
        rocblas_sgemm(blas, rocblas_operation_transpose,
                      rocblas_operation_none, k - b1, rows, kGroup, &minus_one,
                      s.hinv + static_cast<std::size_t>(b1) * k + b0, k, s.err,
                      kGroup, &one, s.w + b1, k) != rocblas_status_success)
      return false;
  }
  return true;
}

void QuantizeRtn(Scratch& s, const float* w_rot, hipStream_t stream) {
  RtnKernel<<<(s.rows + 63) / 64, 64, 0, stream>>>(w_rot, s.codes, s.gscales,
                                                    s.rows, s.k);
}

void Pack(const Scratch& s, std::uint8_t* tiles, std::uint16_t* scales,
          hipStream_t stream) {
  const std::size_t total =
      static_cast<std::size_t>(s.k / kGroup) * s.rows * 4;
  PackKernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      s.codes, s.gscales, tiles, scales, s.rows, s.k);
}

}  // namespace gufo::models::inkling::gptq
