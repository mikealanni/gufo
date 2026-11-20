// int4 x int4 WMMA GEMM for rotated symmetric 4-bit weights.
//
// out[t][f] (+)= sum over 128-wide K groups e of
//   sx[e][t] * sw[e][f] * dot(q_act[t][e], q_w[f][e])
// where q_* are signed int4 codes and the dot product is the integer
// v_wmma_i32_16x16x16_iu4 accumulation. CTA tile 128 features x 128 tokens,
// eight waves, each wave 32 features x 64 tokens; weights stream from global
// memory straight into registers (double buffered across the two 128 groups
// of one barrier step), activation tiles through LDS, and half the waves defer
// one fold so the scale epilogue of one wave overlaps the matrix work of its
// neighbours. K is processed 256 at a time per barrier.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

#include "src/models/qwen/hip/ops/iu4.hpp"

namespace gufo::hip {
namespace {

typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

constexpr int kFeatureTile = 128;
constexpr int kTokenTile = 128;
constexpr int kLdsBytes = 32768 + 4096;
// Integer accumulators start at the exponent of 2^23 + 2^22 so that adding
// the signed int32 dot product leaves float(1.5 * 2^23 + c) and a single
// subtraction of 12582912.0f converts it exactly (|c| <= 8192 per group).
constexpr int kMagic = 0x4B400000;
constexpr float kMagicFloat = 12582912.0f;

__device__ __forceinline__ i32x8 MagicInit() {
  return (i32x8){kMagic, kMagic, kMagic, kMagic,
                 kMagic, kMagic, kMagic, kMagic};
}

template<bool kAdd, bool kDefer>
__device__ __forceinline__ void Iu4GemmBody(
    const std::uint8_t* __restrict__ pw, const std::uint16_t* __restrict__ psc,
    const std::uint8_t* __restrict__ qs, const float* __restrict__ dsT,
    float* __restrict__ out, int M, int N, int Npad, int K, int row_begin,
    int group, char* lds) {
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int lr = lane & 15, hi = lane >> 4;
  const int prow = 8 * (lr & 1) + (lr >> 1);
  const int wf = wave & 3, wth = wave >> 2;
  const int tb = Npad >> 7;
  // `group` feature tiles per CTA group keeps neighbouring CTAs on the same
  // weight tiles (L2 reuse).
  const int gsz = group * tb, gid = blockIdx.x / gsz, wi = blockIdx.x % gsz;
  const int f0 = row_begin + (gid * group + wi % group) * kFeatureTile;
  const int t0 = (wi / group) * kTokenTile;
  const int E = K >> 7;
  const char* gq = reinterpret_cast<const char*>(qs) +
                   (size_t)((t0 >> 4) + wave) * E * 1024 + lane * 32;
  const uint32_t lds_off = wave * 1024 + hi * 128 + lr * 8;
  const uint32_t aoff = (wth * 4) * 1024 + prow * 8;
  const int frow = f0 + wf * 32 + lr;

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
        sxn[k] = dsT[(size_t)(e + k) * Npad + t0 + tid];
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
          (((size_t)0 * (M >> 4) + (frow >> 4) + b) * 4 + q) * 256 + lr * 16);
  unsigned short scn[2];
  const auto meta_load = [&](int e) {
    for (int b = 0; b < 2; ++b)
      scn[b] = psc[(size_t)e * M + frow + 16 * b];
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
              (((size_t)(e + 1) * (M >> 4) + (frow >> 4) + b) * 4 + (s >> 1)) *
                  256 +
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

  for (int a = 0; a < 4; ++a)
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + wth * 64 + a * 16 + 8 * hi + j;
      if (t >= N)
        continue;
      for (int b = 0; b < 2; ++b) {
        float* dst = out + (size_t)t * M + f0 + wf * 32 + b * 16 + lr;
        if (kAdd)
          *dst += sum[a][b][j];
        else
          *dst = sum[a][b][j];
      }
    }
}

template<bool kAdd>
__global__ void __launch_bounds__(256)
    Iu4GemmKernel(const std::uint8_t* __restrict__ pw,
                  const std::uint16_t* __restrict__ psc,
                  const std::uint8_t* __restrict__ qs,
                  const float* __restrict__ dsT, float* __restrict__ out, int M,
                  int N, int Npad, int K, int row_begin, int group) {
  __shared__ __attribute__((aligned(16))) char lds[kLdsBytes];
  if ((threadIdx.x >> 7) & 1)
    Iu4GemmBody<kAdd, true>(pw, psc, qs, dsT, out, M, N, Npad, K, row_begin,
                            group, lds);
  else
    Iu4GemmBody<kAdd, false>(pw, psc, qs, dsT, out, M, N, Npad, K, row_begin,
                             group, lds);
}

// Scalar fallback for projections too small for the WMMA tile (the 48 row GDN
// a/b gates): one thread per output element, integer dot products group by
// group.
template<bool kAdd>
__global__ void Iu4GemmSmallKernel(const std::uint8_t* __restrict__ pw,
                                   const std::uint16_t* __restrict__ psc,
                                   const std::uint8_t* __restrict__ qs,
                                   const float* __restrict__ dsT,
                                   float* __restrict__ out, int M, int N,
                                   int Npad, int K) {
  const int f = threadIdx.x % M;
  const int t = blockIdx.x * (blockDim.x / M) + threadIdx.x / M;
  if (t >= N)
    return;
  const int E = K >> 7;
  float acc = 0.0f;
  for (int e = 0; e < E; ++e) {
    int dot = 0;
    for (int b = 0; b < 64; ++b) {
      const int q = b >> 4, within = b & 15;
      const std::uint8_t wbyte =
          pw[(((size_t)e * (M >> 4) + (f >> 4)) * 4 + q) * 256 + (f & 15) * 16 +
             within];
      const int chunk = b >> 3, w8 = b & 7;
      const std::uint8_t abyte =
          qs[((size_t)(t >> 4) * E + e) * 1024 +
             ((chunk & 1) * 16 + (t & 15)) * 32 + (chunk >> 1) * 8 + w8];
      const int w0 = (int)(wbyte & 15), w1 = (int)(wbyte >> 4);
      const int a0 = (int)(abyte & 15), a1 = (int)(abyte >> 4);
      dot += ((w0 ^ 8) - 8) * ((a0 ^ 8) - 8) + ((w1 ^ 8) - 8) * ((a1 ^ 8) - 8);
    }
    acc += dsT[(size_t)e * Npad + t] *
           __half2float(__ushort_as_half(psc[(size_t)e * M + f])) * (float)dot;
  }
  float* dst = out + (size_t)t * M + f;
  if (kAdd)
    *dst += acc;
  else
    *dst = acc;
}

}  // namespace

bool Iu4GemmSupported(std::size_t rows, std::size_t cols) {
  return rows % kFeatureTile == 0 && cols % 256 == 0;
}

void LaunchIu4Gemm(const void* tiles, const std::uint16_t* weight_scales,
                   const std::uint8_t* act_tiles, const float* act_scales,
                   float* out, bool accumulate, std::size_t rows,
                   std::size_t row_begin, std::size_t n, std::size_t padded_n,
                   std::size_t k, hipStream_t stream) {
  const std::size_t tiles_n = (rows - row_begin) / kFeatureTile;
  const int group = tiles_n % 4 == 0 ? 4 : tiles_n % 2 == 0 ? 2 : 1;
  const unsigned blocks =
      static_cast<unsigned>(tiles_n * (padded_n / kTokenTile));
  const auto* pw = static_cast<const std::uint8_t*>(tiles);
  if (accumulate)
    Iu4GemmKernel<true><<<blocks, 256, 0, stream>>>(
        pw, weight_scales, act_tiles, act_scales, out, (int)rows, (int)n,
        (int)padded_n, (int)k, (int)row_begin, group);
  else
    Iu4GemmKernel<false><<<blocks, 256, 0, stream>>>(
        pw, weight_scales, act_tiles, act_scales, out, (int)rows, (int)n,
        (int)padded_n, (int)k, (int)row_begin, group);
}

void LaunchIu4GemmSmall(const void* tiles, const std::uint16_t* weight_scales,
                        const std::uint8_t* act_tiles, const float* act_scales,
                        float* out, bool accumulate, std::size_t rows,
                        std::size_t n, std::size_t padded_n, std::size_t k,
                        hipStream_t stream) {
  const int per_block = 256 / static_cast<int>(rows);
  const unsigned blocks =
      static_cast<unsigned>((n + per_block - 1) / per_block);
  const unsigned threads = static_cast<unsigned>(per_block * rows);
  const auto* pw = static_cast<const std::uint8_t*>(tiles);
  if (accumulate)
    Iu4GemmSmallKernel<true><<<blocks, threads, 0, stream>>>(
        pw, weight_scales, act_tiles, act_scales, out, (int)rows, (int)n,
        (int)padded_n, (int)k);
  else
    Iu4GemmSmallKernel<false><<<blocks, threads, 0, stream>>>(
        pw, weight_scales, act_tiles, act_scales, out, (int)rows, (int)n,
        (int)padded_n, (int)k);
}

}  // namespace gufo::hip
