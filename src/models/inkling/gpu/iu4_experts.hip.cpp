// Rotated symmetric int4 x int4 WMMA path for Inkling's routed gate/up
// experts. The GEMM body follows the exact per-128 kernel of the dense int4
// port: a wave holds 16 gate features and the matching 16 up features, so the
// SiTU activation runs on the accumulators in registers.
#include "src/models/inkling/gpu/iu4_experts.hpp"

#include <cstring>
#include <type_traits>

namespace gufo::models::inkling::iu4 {
namespace {

typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

constexpr int kLdsBytes = 32768 + 4096;
// Integer accumulators start at 1.5 * 2^23 so a signed int32 dot product
// leaves float(1.5 * 2^23 + c); one subtraction converts it exactly.
constexpr int kMagic = 0x4B400000;
constexpr float kMagicFloat = 12582912.0f;

__device__ __forceinline__ i32x8 MagicInit() {
  return (i32x8){kMagic, kMagic, kMagic, kMagic,
                 kMagic, kMagic, kMagic, kMagic};
}

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// tanh(x) = sign(x) * (1 - e) / (1 + e), e = exp(-2|x|): a few instructions
// instead of the libm routine, whose register footprint slowed the whole GEMM.
__device__ __forceinline__ float FastTanh(float x) {
  const float e = __expf(-2.0f * fabsf(x));
  const float t = (1.0f - e) * __frcp_rn(1.0f + e);
  return copysignf(t, x);
}

__device__ __forceinline__ float Ue4m3(unsigned b) {
  const unsigned e = (b >> 3) & 15U, m = b & 7U;
  if (b == 0x7F)
    return 0.0f;
  return e == 0 ? static_cast<float>(m) * (1.0f / 512.0f)
                : (1.0f + static_cast<float>(m) * 0.125f) *
                      exp2f(static_cast<float>(e) - 7.0f);
}

// y = R v with R = D2 H D1 / 16 over one 256 block held in LDS (256 threads).
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

// Quantizes the rotated 256 block in v (LDS): one scale per 128 half, signed
// codes into qc (LDS, as ints); returns this half's scale in `scale_out`.
__device__ __forceinline__ void QuantHalves(const float* v, int* qc,
                                            float* red, float& scale_out,
                                            bool round_scale_half) {
  const int t = threadIdx.x;
  float a = fabsf(v[t]);
  for (int o = 16; o > 0; o >>= 1)
    a = fmaxf(a, __shfl_xor(a, o));
  if ((t & 31) == 0)
    red[t >> 5] = a;
  __syncthreads();
  const int g = t >> 7;
  const float amax = fmaxf(fmaxf(red[g * 4], red[g * 4 + 1]),
                           fmaxf(red[g * 4 + 2], red[g * 4 + 3]));
  float s = amax / 7.0f;
  if (round_scale_half)
    s = __half2float(__float2half(s));
  const float q = s > 0.0f ? fminf(7.0f, fmaxf(-8.0f, rintf(v[t] / s))) : 0.0f;
  qc[t] = static_cast<int>(q);
  scale_out = s;
  __syncthreads();
}

__global__ void __launch_bounds__(256)
    RepackKernel(const std::uint8_t* __restrict__ src,
                 std::size_t src_row_bytes, std::uint8_t* __restrict__ tiles,
                 std::uint16_t* __restrict__ scales, int rows, int k,
                 const float* __restrict__ signs) {
  __shared__ float v[256];
  __shared__ int qc[256];
  __shared__ float red[8];
  __shared__ float s1[256], s2[256];
  const int t = threadIdx.x;
  s1[t] = signs[t];
  s2[t] = signs[256 + t];
  const std::size_t gr = blockIdx.x;  // expert * rows + row
  const int expert = static_cast<int>(gr / rows);
  const int row = static_cast<int>(gr % rows);
  const std::uint8_t* sr = src + gr * src_row_bytes;
  const int groups = k >> 7;
  std::uint8_t* tiles_e =
      tiles + static_cast<std::size_t>(expert) * groups * rows * 64;
  std::uint16_t* scales_e =
      scales + static_cast<std::size_t>(expert) * groups * rows;
  constexpr float kMags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int kb = 0; kb < k / 256; ++kb) {
    const int j = kb * 8 + (t >> 5);
    const int p = t & 31;
    const int byte = (p < 8) ? p : (p < 16) ? p - 8 : (p < 24) ? 8 + p - 16
                                                              : 8 + p - 24;
    const unsigned raw = sr[j * 16 + byte];
    const unsigned nb = (p < 8 || (p >= 16 && p < 24)) ? (raw & 15U) : (raw >> 4);
    const float mag = kMags[nb & 7U];
    const float sc = Ue4m3(sr[k / 2 + j * 2 + (p >> 4)]);
    v[t] = ((nb & 8U) ? -mag : mag) * sc;
    __syncthreads();
    Rot256(v, s1, s2);
    float s;
    QuantHalves(v, qc, red, s, true);
    const int g = t >> 7;
    const int e = kb * 2 + g;
    if ((t & 127) == 0)
      scales_e[static_cast<std::size_t>(e) * rows + row] =
          __half_as_ushort(__float2half(s));
    if (t < 8) {
      const int gg = t >> 2, q = t & 3;
      const int ee = kb * 2 + gg;
      std::uint8_t bytes[16];
#pragma unroll
      for (int b = 0; b < 16; ++b) {
        const int c0 = qc[gg * 128 + 2 * (16 * q + b)] & 0xF;
        const int c1 = qc[gg * 128 + 2 * (16 * q + b) + 1] & 0xF;
        bytes[b] = static_cast<std::uint8_t>(c0 | (c1 << 4));
      }
      uint4 w;
      std::memcpy(&w, bytes, 16);
      const std::size_t dst =
          ((static_cast<std::size_t>(ee) * (rows >> 4) + (row >> 4)) * 4 + q) *
              256 +
          (row & 15) * 16;
      *reinterpret_cast<uint4*>(tiles_e + dst) = w;
    }
    __syncthreads();
  }
}

// Lane xor within a row of 16 as a DPP operand (strides 1 to 8); stride 16
// crosses rows and goes through the generic shuffle.
template<int kMask>
__device__ __forceinline__ float XorLanes(float v) {
  if constexpr (kMask < 16)
    return __int_as_float(__builtin_amdgcn_mov_dpp(
        __float_as_int(v), 0x160 | kMask, 0xF, 0xF, false));
  else
    return __shfl_xor(v, kMask);
}

__device__ __forceinline__ float Reduce16Max(float v) {
  v = fmaxf(v, XorLanes<8>(v));
  v = fmaxf(v, XorLanes<4>(v));
  v = fmaxf(v, XorLanes<2>(v));
  return fmaxf(v, XorLanes<1>(v));
}

// One workgroup per compact routed row, one wave per 256-wide K block: lane l
// owns elements 8l..8l+7. Register/DPP Walsh-Hadamard, one scale per 128.
__global__ void __launch_bounds__(1024)
    ActKernel(const __half* __restrict__ x,
              const std::int32_t* __restrict__ rows_in,
              const std::int32_t* __restrict__ rows_out,
              const std::int32_t* __restrict__ pad_bounds, int experts, int k,
              std::uint8_t* __restrict__ tiles, float* __restrict__ scales,
              int padded_rows, const float* __restrict__ signs) {
  const int c = blockIdx.x;
  if (c >= pad_bounds[experts])
    return;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int groups = k >> 7;
  const bool live = rows_out[c] >= 0;
  float v[8];
  if (live) {
    const std::size_t token = static_cast<std::size_t>(rows_in[c]);
    const uint4 raw = *reinterpret_cast<const uint4*>(
        x + token * k + wave * 256 + lane * 8);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(h2[i]);
      v[2 * i] = f.x;
      v[2 * i + 1] = f.y;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] *= signs[lane * 8 + i];
#pragma unroll
    for (int st = 1; st < 8; st <<= 1)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if ((i & st) == 0) {
          const float a = v[i], b = v[i + st];
          v[i] = a + b;
          v[i + st] = a - b;
        }
    const auto stage = [&](auto mask_tag) {
      constexpr int m = decltype(mask_tag)::value;
      const bool hi = (lane & m) != 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float o = XorLanes<m>(v[i]);
        v[i] = hi ? (o - v[i]) : (v[i] + o);
      }
    };
    stage(std::integral_constant<int, 1>{});
    stage(std::integral_constant<int, 2>{});
    stage(std::integral_constant<int, 4>{});
    stage(std::integral_constant<int, 8>{});
    stage(std::integral_constant<int, 16>{});
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] *= 0.0625f * signs[256 + lane * 8 + i];
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = 0.0f;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  amax = Reduce16Max(amax);
  const float scale = amax / 7.0f;
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  std::uint32_t word = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = static_cast<int>(fminf(fmaxf(rintf(v[i] * inv), -8.0f), 7.0f));
    word |= static_cast<std::uint32_t>(q & 0xF) << (4 * i);
  }
  const int e = wave * 2 + (lane >> 4);
  const int l16 = lane & 15;
  const int chunk = l16 >> 1, j = l16 & 1;
  const std::size_t dst =
      (static_cast<std::size_t>(c >> 4) * groups + e) * 1024 +
      ((chunk & 1) * 16 + (c & 15)) * 32 + (chunk >> 1) * 8 + 4 * j;
  *reinterpret_cast<std::uint32_t*>(tiles + dst) = word;
  if (l16 == 0)
    scales[static_cast<std::size_t>(e) * padded_rows + c] = scale;
}

template<bool kDefer>
__device__ __forceinline__ void GateUpBody(
    const std::uint8_t* __restrict__ pw, const std::uint16_t* __restrict__ psc,
    const std::uint8_t* __restrict__ qs, const float* __restrict__ dsT,
    int Npad, int K, int ff, int f0, int c0, int c_end,
    const std::int32_t* __restrict__ rows_out, __half* __restrict__ act,
    float es, float beta, float lb, char* lds) {
  const int M = 2 * ff;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wf = wave & 3, wth = wave >> 2;
  const int E = K >> 7;
  const char* gq = reinterpret_cast<const char*>(qs) +
                   static_cast<size_t>((c0 >> 4) + wave) * E * 1024 + lane * 32;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wth * 4) * 1024 + prow * 8;
  // b = 0: gate features, b = 1: the matching up features
  const int fr0 = f0 + wf * 16;
  const int tb[2] = {fr0 >> 4, (ff + fr0) >> 4};
  const int mrow[2] = {fr0 + lr, ff + fr0 + lr};

  float sum[4][2][8];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < 8; ++j)
        sum[a][b][j] = 0.0f;

  float sxn[2] = {0.0f, 0.0f};
  i32x2 sa[2][4];
  const auto stage_load = [&](int e) {
    for (int k = 0; k < 2; ++k) {
      for (int i = 0; i < 2; ++i) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(gq + (e + k) * 1024 + 16 * i);
        sa[k][2 * i] = i32x2{(int)v.x, (int)v.y};
        sa[k][2 * i + 1] = i32x2{(int)v.z, (int)v.w};
      }
      if (tid < 128)
        sxn[k] = dsT[static_cast<size_t>(e + k) * Npad + c0 + tid];
    }
  };
  const auto stage_store = [&](char* slot, int e) {
    for (int k = 0; k < 2; ++k) {
      for (int i = 0; i < 4; ++i)
        *reinterpret_cast<i32x2*>(slot + k * 8192 + lds_off + i * 256) =
            sa[k][i];
      if (tid < 128)
        *reinterpret_cast<float*>(lds + 32768 + ((e + k) & 7) * 512 + tid * 4) =
            sxn[k];
    }
  };
  uint4 wc[2][4];
  uint4 wn[2][4];
  for (int b = 0; b < 2; ++b)
    for (int q = 0; q < 4; ++q)
      wc[b][q] = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const char*>(pw) +
          (((size_t)0 * (M >> 4) + tb[b]) * 4 + q) * 256 + lr * 16);
  unsigned short scn[2];
  const auto meta_load = [&](int e) {
    for (int b = 0; b < 2; ++b)
      scn[b] = psc[static_cast<size_t>(e) * M + mrow[b]];
  };
  stage_load(0);
  stage_store(lds, 0);
  meta_load(0);
  __syncthreads();

  i32x8 Cp[2][2];
  for (int b = 0; b < 2; ++b)
    for (int a = 0; a < 2; ++a)
      Cp[b][a] = MagicInit();
  float scp[2] = {0.0f, 0.0f};
  const char* sslp = lds + 32768;

  const auto fold = [&](int h, i32x8(&C)[2][2], const float (&sc)[2],
                        const char* ssl) {
#pragma unroll
    for (int a2 = 0; a2 < 2; ++a2) {
      const int a = 2 * h + a2;
      float scf[8];
      const float4 s0 = *reinterpret_cast<const float4*>(
          ssl + (wth * 64 + 16 * a + 8 * hi) * 4);
      const float4 s1 = *reinterpret_cast<const float4*>(
          ssl + (wth * 64 + 16 * a + 8 * hi) * 4 + 16);
      const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
      for (int j = 0; j < 8; ++j) {
        scf[j] = sv[j];
        asm volatile("" : "+v"(scf[j]));
      }
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        for (int j = 0; j < 8; ++j)
          sum[a][b][j] = __builtin_fmaf(
              scf[j] * sc[b],
              __fsub_rn(__int_as_float(C[b][a2][j]), kMagicFloat),
              sum[a][b][j]);
        asm volatile(""
                     : "+v"(sum[a][b][0]), "+v"(sum[a][b][1]),
                       "+v"(sum[a][b][2]), "+v"(sum[a][b][3]),
                       "+v"(sum[a][b][4]), "+v"(sum[a][b][5]),
                       "+v"(sum[a][b][6]), "+v"(sum[a][b][7]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  };

  const auto wmma = [&](int h, i32x8(&C)[2][2], const char* slot,
                        uint4(&wsrc)[2][4], uint4(&wdst)[2][4], bool reload,
                        int e) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      i32x2 ar[2];
      for (int a2 = 0; a2 < 2; ++a2)
        ar[a2] = *reinterpret_cast<const i32x2*>(slot + aoff +
                                                 (2 * h + a2) * 1024 + s * 128);
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const uint4 wq = wsrc[b][s >> 1];
        const i32x2 wv =
            (s & 1) ? i32x2{(int)wq.z, (int)wq.w} : i32x2{(int)wq.x, (int)wq.y};
#pragma unroll
        for (int a2 = 0; a2 < 2; ++a2)
          C[b][a2] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, ar[a2], true, wv, s == 0 ? MagicInit() : C[b][a2], false);
      }
      if (reload && (s & 1)) {
        for (int b = 0; b < 2; ++b)
          wdst[b][s >> 1] = *reinterpret_cast<const uint4*>(
              reinterpret_cast<const char*>(pw) +
              (((size_t)(e + 1) * (M >> 4) + tb[b]) * 4 + (s >> 1)) * 256 +
              lr * 16);
      }
      if (s != 7)
        __builtin_amdgcn_sched_barrier(0);
    }
  };

  const auto sub = [&](auto nxt_tag, int e, const char* slot,
                       uint4(&wsrc)[2][4], uint4(&wdst)[2][4]) {
    constexpr bool nxt = decltype(nxt_tag)::value;
    float sc[2];
    for (int b = 0; b < 2; ++b)
      sc[b] = __half2float(__ushort_as_half(scn[b]));
    __builtin_amdgcn_sched_barrier(0);
    if (nxt)
      meta_load(e + 1);
    __builtin_amdgcn_sched_barrier(0);
    i32x8 C0[2][2], C1[2][2];
    const char* ssl = lds + 32768 + (e & 7) * 512;
    if (kDefer)
      fold(1, Cp, scp, sslp);
    wmma(0, C0, slot, wsrc, wdst, nxt, e);
    fold(0, C0, sc, ssl);
    wmma(1, C1, slot, wsrc, wdst, false, e);
    if (kDefer) {
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a)
          Cp[b][a] = C1[b][a];
      scp[0] = sc[0];
      scp[1] = sc[1];
      sslp = ssl;
    } else {
      fold(1, C1, sc, ssl);
    }
  };
  int e = 0;
  for (; e + 2 < E; e += 2) {
    const char* slot = lds + (e & 2) * 8192;
    stage_load(e + 2);
    sub(std::true_type{}, e, slot, wc, wn);
    sub(std::true_type{}, e + 1, slot + 8192, wn, wc);
    stage_store(lds + ((e + 2) & 2) * 8192, e + 2);
    __syncthreads();
  }
  {
    const char* slot = lds + (e & 2) * 8192;
    sub(std::true_type{}, e, slot, wc, wn);
    sub(std::false_type{}, e + 1, slot + 8192, wn, wc);
  }
  if (kDefer)
    fold(1, Cp, scp, sslp);

  // Epilogue: gate and up go to LDS as F16 (the K loop's buffers are dead
  // now), then every thread applies SiTU on coalesced rows. Keeping the
  // transcendentals out of the accumulator-live region stops the register
  // allocator from spilling the K loop.
  __syncthreads();
  __half* eg = reinterpret_cast<__half*>(lds);
  const int fi = wf * 16 + lr;
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int tl = wth * 64 + a * 16 + 8 * hi + j;
#pragma unroll
      for (int b = 0; b < 2; ++b)
        eg[(tl * 2 + b) * 64 + fi] = __float2half(sum[a][b][j] * es);
    }
  __syncthreads();
  for (int idx = tid; idx < kTokenTile * 64; idx += 256) {
    const int tl = idx >> 6, f2 = idx & 63;
    const int c = c0 + tl;
    if (c >= c_end)
      continue;
    const std::int32_t dst = rows_out[c];
    if (dst < 0)
      continue;
    const float g = __half2float(eg[(tl * 2) * 64 + f2]);
    const float u = __half2float(eg[(tl * 2 + 1) * 64 + f2]);
    act[static_cast<std::size_t>(dst) * ff + f0 + f2] = __float2half(
        beta * FastTanh(g / beta) * SigmoidF(g) * lb * FastTanh(u / lb));
  }
}

__global__ void __launch_bounds__(256)
    GateUpKernel(const std::uint8_t* __restrict__ pw,
                 const std::uint16_t* __restrict__ psc,
                 const std::uint8_t* __restrict__ qs,
                 const float* __restrict__ dsT, int Npad, int K, int ff,
                 const std::int32_t* __restrict__ tiles,
                 const std::int32_t* __restrict__ pad_bounds,
                 const std::int32_t* __restrict__ rows_out,
                 __half* __restrict__ act, const float* __restrict__ expert_scale,
                 const float* __restrict__ situ) {
  __shared__ __attribute__((aligned(16))) char lds[kLdsBytes];
  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * kTokenTile;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  if (bucket_rows - t_local <= 0)
    return;
  const int M = 2 * ff;
  const int E = K >> 7;
  const std::uint8_t* pw_e =
      pw + static_cast<std::size_t>(expert) * E * M * 64;
  const std::uint16_t* psc_e = psc + static_cast<std::size_t>(expert) * E * M;
  const int f0 = static_cast<int>(blockIdx.x) * kFeatureTile;
  const float es = expert_scale != nullptr ? expert_scale[expert] : 1.0f;
  const int c0 = bucket_begin + t_local;
  const int c_end = bucket_begin + bucket_rows;
  if ((threadIdx.x >> 7) & 1)
    GateUpBody<true>(pw_e, psc_e, qs, dsT, Npad, K, ff, f0, c0, c_end, rows_out,
                     act, es, situ[0], situ[1], lds);
  else
    GateUpBody<false>(pw_e, psc_e, qs, dsT, Npad, K, ff, f0, c0, c_end,
                      rows_out, act, es, situ[0], situ[1], lds);
}

}  // namespace

void FillSigns(float* host_512) {
  unsigned st = 42;
  for (int i = 0; i < 256; ++i) {
    st = (st * 1103515245U + 12345U) & 0x7fffffffU;
    host_512[i] = ((st >> 16) & 1U) ? -1.0f : 1.0f;
  }
  st = 1042;
  for (int i = 0; i < 256; ++i) {
    st = (st * 1103515245U + 12345U) & 0x7fffffffU;
    host_512[256 + i] = ((st >> 16) & 1U) ? -1.0f : 1.0f;
  }
}

std::size_t WeightBytesPerExpert(std::size_t rows, std::size_t k) {
  return (k / 128) * rows * 64;
}
std::size_t ScaleBytesPerExpert(std::size_t rows, std::size_t k) {
  return (k / 128) * rows * 2;
}

void RepackExperts(const std::uint8_t* nvfp4, std::size_t src_row_bytes,
                   std::uint8_t* tiles, std::uint16_t* scales, int experts,
                   int rows, int k, const float* signs, hipStream_t stream) {
  RepackKernel<<<static_cast<unsigned>(experts) * rows, 256, 0, stream>>>(
      nvfp4, src_row_bytes, tiles, scales, rows, k, signs);
}

void QuantizeRoutedActs(const __half* x, const std::int32_t* rows_in,
                        const std::int32_t* rows_out,
                        const std::int32_t* pad_bounds, int experts,
                        int max_rows, int k, std::uint8_t* tiles,
                        float* scales, int padded_rows, const float* signs,
                        hipStream_t stream) {
  ActKernel<<<max_rows, (k / 256) * 32, 0, stream>>>(x, rows_in, rows_out, pad_bounds,
                                          experts, k, tiles, scales,
                                          padded_rows, signs);
}

void GateUpSitu(const std::uint8_t* tiles, const std::uint16_t* scales,
                const std::uint8_t* act_tiles, const float* act_scales,
                int padded_rows, const std::int32_t* tile_map,
                std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                const std::int32_t* rows_out, __half* act, int ff, int k,
                const float* expert_scale, const float* situ,
                hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>(ff / kFeatureTile), n_tiles);
  GateUpKernel<<<grid, 256, 0, stream>>>(tiles, scales, act_tiles, act_scales,
                                         padded_rows, k, ff, tile_map,
                                         pad_bounds, rows_out, act,
                                         expert_scale, situ);
}

}  // namespace gufo::models::inkling::iu4
