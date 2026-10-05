// IQ3_S gated expert decode (gate and up, 640 x 2560 per expert, 10 routed
// experts of 512) on gfx1151: bytes per call and GB/s for kernel variants.
//
//   tools/bench/build.sh iq3s_moe_gated_bench && /tmp/iq3s_moe_gated_bench
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x)                                                        \
  do {                                                                  \
    hipError_t e_ = (x);                                                \
    if (e_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,            \
                   hipGetErrorString(e_));                              \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

constexpr int kExperts = 512;
constexpr int kUsed = 10;
constexpr int kRows = 640;
constexpr int kK = 2560;
constexpr int kBlockBytes = 110;
constexpr int kBlocksPerRow = kK / 256;
constexpr int kRowBytes = kBlocksPerRow * kBlockBytes;  // 1100
constexpr int kRowDwords = kRowBytes / 4;               // 275
constexpr int kSubBlocks = kK / 32;                     // 80

__device__ __forceinline__ float Dot2(half2 a, half2 b, float c) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
  return c;
}

__device__ __forceinline__ void Decode(const uint8_t* __restrict__ b, int ib32,
                                       const uint32_t* __restrict__ grid,
                                       half2 w[16], float& scale) {
  const uint8_t* qs = b + 2 + ib32 * 8;
  const uint8_t* signs = b + 74 + ib32 * 4;
  const uint32_t qh = b[66 + ib32];
  const uint32_t sc = (b[106 + ib32 / 2] >> (4 * (ib32 % 2))) & 0x0FU;
  const half d = __ushort_as_half(
      static_cast<unsigned short>(b[0] | (static_cast<unsigned>(b[1]) << 8)));
  scale = __half2float(d) * static_cast<float>(1 + 2 * sc);
  const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t index = qs[g] | ((qh << (8 - g)) & 256U);
    const uint32_t magnitudes = grid[index];
    const uint32_t bits = (signs[g / 2] >> (4 * (g % 2))) & 0xFU;
    const uint32_t neg = ((bits * 0x00204081U) & 0x01010101U);
    const uint32_t codes = ((magnitudes ^ (neg * 0xFFU)) + neg) ^ 0x80808080U;
    const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
    w[2 * g] = __hadd2(__builtin_bit_cast(half2, p0), magic);
    w[2 * g + 1] = __hadd2(__builtin_bit_cast(half2, p1), magic);
  }
}

template<bool kFakeGrid>
__device__ __forceinline__ void DecodeT(const uint8_t* __restrict__ b, int ib32,
                                        const uint32_t* __restrict__ grid,
                                        half2 w[16], float& scale) {
  const uint8_t* qs = b + 2 + ib32 * 8;
  const uint8_t* signs = b + 74 + ib32 * 4;
  const uint32_t qh = b[66 + ib32];
  const uint32_t sc = (b[106 + ib32 / 2] >> (4 * (ib32 % 2))) & 0x0FU;
  const half d = __ushort_as_half(
      static_cast<unsigned short>(b[0] | (static_cast<unsigned>(b[1]) << 8)));
  scale = __half2float(d) * static_cast<float>(1 + 2 * sc);
  const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t index = qs[g] | ((qh << (8 - g)) & 256U);
    const uint32_t magnitudes =
        kFakeGrid ? ((index * 0x01030507U) & 0x0F0F0F0FU) | 0x01010101U
                  : grid[index];
    const uint32_t bits = (signs[g / 2] >> (4 * (g % 2))) & 0xFU;
    const uint32_t neg = ((bits * 0x00204081U) & 0x01010101U);
    const uint32_t codes = ((magnitudes ^ (neg * 0xFFU)) + neg) ^ 0x80808080U;
    const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
    w[2 * g] = __hadd2(__builtin_bit_cast(half2, p0), magic);
    w[2 * g + 1] = __hadd2(__builtin_bit_cast(half2, p1), magic);
  }
}

// Decode with qs and signs read as 16-bit pairs (the fields are 2-byte
// aligned in a 110-byte block).
__device__ __forceinline__ void DecodeW(const uint8_t* __restrict__ b, int ib32,
                                        const uint32_t* __restrict__ grid,
                                        half2 w[16], float& scale) {
  const auto* h = reinterpret_cast<const uint16_t*>(b);
  const uint32_t q0 = h[1 + ib32 * 4] | (static_cast<uint32_t>(h[2 + ib32 * 4]) << 16);
  const uint32_t q1 = h[3 + ib32 * 4] | (static_cast<uint32_t>(h[4 + ib32 * 4]) << 16);
  const uint32_t sg = h[37 + ib32 * 2] | (static_cast<uint32_t>(h[38 + ib32 * 2]) << 16);
  const uint32_t qhw = h[33 + ib32 / 2];
  const uint32_t qh = (qhw >> (8 * (ib32 & 1))) & 0xFFU;
  const uint32_t scw = h[53 + ib32 / 4];
  const uint32_t sc = (scw >> (8 * ((ib32 / 2) & 1) + 4 * (ib32 & 1))) & 0x0FU;
  scale = __half2float(__ushort_as_half(h[0])) * static_cast<float>(1 + 2 * sc);
  const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t qsb = ((g < 4 ? q0 : q1) >> (8 * (g % 4))) & 0xFFU;
    const uint32_t index = qsb | ((qh << (8 - g)) & 256U);
    const uint32_t magnitudes = grid[index];
    const uint32_t bits = (sg >> (4 * g)) & 0xFU;
    const uint32_t neg = ((bits * 0x00204081U) & 0x01010101U);
    const uint32_t codes = ((magnitudes ^ (neg * 0xFFU)) + neg) ^ 0x80808080U;
    const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
    w[2 * g] = __hadd2(__builtin_bit_cast(half2, p0), magic);
    w[2 * g + 1] = __hadd2(__builtin_bit_cast(half2, p1), magic);
  }
}

// Ceiling: stream the routed gate and up rows as coalesced dwords.
__global__ void ReadOnly(const uint32_t* __restrict__ gate,
                         const uint32_t* __restrict__ up,
                         const int* __restrict__ ids, float* __restrict__ out) {
  const int expert = ids[blockIdx.y];
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const uint32_t* g = gate + (static_cast<size_t>(expert) * kRows + row) * kRowDwords;
  const uint32_t* u = up + (static_cast<size_t>(expert) * kRows + row) * kRowDwords;
  uint32_t acc = 0;
  for (int i = lane; i < kRowDwords; i += 32)
    acc += g[i] ^ u[i];
  if (acc == 0x12345678U)
    out[row] = 1.0f;
}

// Variant 1: one wave per row, each lane decodes sub-blocks lane, lane + 32,
// ... straight from global memory; grid table in LDS.
template<int kRowsPerWave, bool kGridLds>
__global__ __launch_bounds__(256) void V1(const uint8_t* __restrict__ gate,
                                          const uint8_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid_g,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  __shared__ uint32_t grid_l[512];
  __shared__ half2 x_l[kK / 2];
  const int tid = threadIdx.x;
  for (int i = tid; i < 512; i += 256)
    grid_l[i] = grid_g[i];
  for (int i = tid; i < kK / 2; i += 256)
    x_l[i] = reinterpret_cast<const half2*>(x)[i];
  __syncthreads();
  const uint32_t* grid = kGridLds ? grid_l : grid_g;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int rr = 0; rr < kRowsPerWave; ++rr) {
    const int row = (blockIdx.x * 8 + wave) * kRowsPerWave + rr;
    const uint8_t* g = gate + (static_cast<size_t>(expert) * kRows + row) * kRowBytes;
    const uint8_t* u = up + (static_cast<size_t>(expert) * kRows + row) * kRowBytes;
    float ag = 0.0f;
    float au = 0.0f;
    for (int sb = lane; sb < kSubBlocks; sb += 32) {
      half2 wg[16];
      half2 wu[16];
      float dg = 0.0f;
      float du = 0.0f;
      Decode(g + (sb / 8) * kBlockBytes, sb % 8, grid, wg, dg);
      Decode(u + (sb / 8) * kBlockBytes, sb % 8, grid, wu, du);
      const half2* xs = x_l + sb * 16;
      float sg = 0.0f;
      float su = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        sg = Dot2(wg[i], xs[i], sg);
        su = Dot2(wu[i], xs[i], su);
      }
      ag = __fmaf_rn(dg, sg, ag);
      au = __fmaf_rn(du, su, au);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ag += __shfl_xor(ag, o, 32);
      au += __shfl_xor(au, o, 32);
    }
    if (lane == 0)
      out[static_cast<size_t>(blockIdx.y) * kRows + row] =
          (ag / (1.0f + __expf(-ag))) * au;
  }
}

// Variant 4: no LDS at all (grid and x read through the caches); each wave
// handles kRowsPerWave rows so launch and ramp costs amortize.
template<int kRowsPerWave>
__global__ __launch_bounds__(256) void V4(const uint8_t* __restrict__ gate,
                                          const uint8_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  const int tid = threadIdx.x;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const half2* xg = reinterpret_cast<const half2*>(x);
  for (int rr = 0; rr < kRowsPerWave; ++rr) {
    const int row = (blockIdx.x * 8 + wave) * kRowsPerWave + rr;
    const uint8_t* g = gate + (static_cast<size_t>(expert) * kRows + row) * kRowBytes;
    const uint8_t* u = up + (static_cast<size_t>(expert) * kRows + row) * kRowBytes;
    float ag = 0.0f;
    float au = 0.0f;
    for (int sb = lane; sb < kSubBlocks; sb += 32) {
      half2 wg[16];
      half2 wu[16];
      float dg = 0.0f;
      float du = 0.0f;
      Decode(g + (sb / 8) * kBlockBytes, sb % 8, grid, wg, dg);
      Decode(u + (sb / 8) * kBlockBytes, sb % 8, grid, wu, du);
      const half2* xs = xg + sb * 16;
      float sg = 0.0f;
      float su = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        sg = Dot2(wg[i], xs[i], sg);
        su = Dot2(wu[i], xs[i], su);
      }
      ag = __fmaf_rn(dg, sg, ag);
      au = __fmaf_rn(du, su, au);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ag += __shfl_xor(ag, o, 32);
      au += __shfl_xor(au, o, 32);
    }
    if (lane == 0)
      out[static_cast<size_t>(blockIdx.y) * kRows + row] =
          (ag / (1.0f + __expf(-ag))) * au;
  }
}

// Bytes [o, o + 4) of a dword-aligned row, branch-free.
__device__ __forceinline__ uint32_t LoadU32NoBranch(
    const uint32_t* __restrict__ row, int o) {
  const uint32_t lo = row[o >> 2];
  const uint32_t hi = row[(o >> 2) + 1];
  return __builtin_amdgcn_alignbyte(hi, lo, o & 3);
}

// Variant 5: no LDS; kLanes lanes per row (80 sub-blocks divide evenly for
// 16), 32 / kLanes rows per wave; kDword selects branch-free dword loads.
template<int kLanes, bool kDword>
__global__ __launch_bounds__(256) void V5(const uint8_t* __restrict__ gate,
                                          const uint8_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  constexpr int kRowsPerWave = 32 / kLanes;
  const int tid = threadIdx.x;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int sub = lane % kLanes;
  const int row = (blockIdx.x * 8 + (tid >> 5)) * kRowsPerWave + lane / kLanes;
  const half2* xg = reinterpret_cast<const half2*>(x);
  const size_t off = (static_cast<size_t>(expert) * kRows + row) * kRowBytes;
  const uint8_t* mats[2] = {gate + off, up + off};
  float acc[2] = {0.0f, 0.0f};
  const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
  for (int sb = sub; sb < kSubBlocks; sb += kLanes) {
    const int blk = (sb / 8) * kBlockBytes;
    const int ib = sb % 8;
    const half2* xs = xg + sb * 16;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      uint32_t q0, q1, sg, qhw, scw, dw;
      if constexpr (kDword) {
        const auto* r = reinterpret_cast<const uint32_t*>(mats[m]);
        q0 = LoadU32NoBranch(r, blk + 2 + ib * 8);
        q1 = LoadU32NoBranch(r, blk + 6 + ib * 8);
        sg = LoadU32NoBranch(r, blk + 74 + ib * 4);
        qhw = LoadU32NoBranch(r, blk + 66 + (ib & ~3));
        scw = LoadU32NoBranch(r, blk + 106);
        dw = LoadU32NoBranch(r, blk);
      } else {
        const uint8_t* b = mats[m] + blk;
        __builtin_memcpy(&q0, b + 2 + ib * 8, 4);
        __builtin_memcpy(&q1, b + 6 + ib * 8, 4);
        __builtin_memcpy(&sg, b + 74 + ib * 4, 4);
        __builtin_memcpy(&qhw, b + 66 + (ib & ~3), 4);
        __builtin_memcpy(&scw, b + 106, 4);
        dw = b[0] | (static_cast<uint32_t>(b[1]) << 8);
      }
      const uint32_t qh = (qhw >> (8 * (ib & 3))) & 0xFFU;
      const uint32_t sc = (scw >> (8 * (ib / 2) + 4 * (ib % 2))) & 0x0FU;
      const float scale =
          __half2float(__ushort_as_half(static_cast<unsigned short>(dw & 0xFFFFU))) *
          static_cast<float>(1 + 2 * sc);
      float s = 0.0f;
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const uint32_t qsb = ((g < 4 ? q0 : q1) >> (8 * (g % 4))) & 0xFFU;
        const uint32_t index = qsb | ((qh << (8 - g)) & 256U);
        const uint32_t magnitudes = grid[index];
        const uint32_t bits = (sg >> (4 * g)) & 0xFU;
        const uint32_t neg = ((bits * 0x00204081U) & 0x01010101U);
        const uint32_t codes = ((magnitudes ^ (neg * 0xFFU)) + neg) ^ 0x80808080U;
        const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
        const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
        s = Dot2(__hadd2(__builtin_bit_cast(half2, p0), magic), xs[2 * g], s);
        s = Dot2(__hadd2(__builtin_bit_cast(half2, p1), magic), xs[2 * g + 1], s);
      }
      acc[m] = __fmaf_rn(scale, s, acc[m]);
    }
  }
#pragma unroll
  for (int o = kLanes / 2; o > 0; o >>= 1) {
    acc[0] += __shfl_xor(acc[0], o, kLanes);
    acc[1] += __shfl_xor(acc[1], o, kLanes);
  }
  if (sub == 0)
    out[static_cast<size_t>(blockIdx.y) * kRows + row] =
        (acc[0] / (1.0f + __expf(-acc[0]))) * acc[1];
}

// Variant 6: each wave streams its gate and up rows with coalesced dword
// loads into a per-wave LDS buffer, then decodes from LDS. Grid and x come
// through the caches; there is no block-level prologue or barrier.
template<int kRowsPerWave>
__global__ __launch_bounds__(256) void V6(const uint32_t* __restrict__ gate,
                                          const uint32_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  constexpr int kFetch = (kRowDwords + 31) / 32;
  __shared__ uint32_t buf[8][2][kRowDwords + 1];
  const int tid = threadIdx.x;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const half2* xg = reinterpret_cast<const half2*>(x);
  const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
  for (int rr = 0; rr < kRowsPerWave; ++rr) {
    const int row = (blockIdx.x * 8 + wave) * kRowsPerWave + rr;
    const size_t off = (static_cast<size_t>(expert) * kRows + row) * kRowDwords;
#pragma unroll
    for (int i = 0; i < kFetch; ++i) {
      const int idx = lane + 32 * i;
      if (idx < kRowDwords) {
        buf[wave][0][idx] = gate[off + idx];
        buf[wave][1][idx] = up[off + idx];
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    float acc[2] = {0.0f, 0.0f};
    for (int sb = lane; sb < kSubBlocks; sb += 32) {
      const half2* xs = xg + sb * 16;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const auto* b =
            reinterpret_cast<const uint8_t*>(buf[wave][m]) + (sb / 8) * kBlockBytes;
        half2 w[16];
        float scale = 0.0f;
        Decode(b, sb % 8, grid, w, scale);
        float sdot = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          sdot = Dot2(w[i], xs[i], sdot);
        acc[m] = __fmaf_rn(scale, sdot, acc[m]);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      acc[0] += __shfl_xor(acc[0], o, 32);
      acc[1] += __shfl_xor(acc[1], o, 32);
    }
    if (lane == 0)
      out[static_cast<size_t>(blockIdx.y) * kRows + row] =
          (acc[0] / (1.0f + __expf(-acc[0]))) * acc[1];
    __builtin_amdgcn_wave_barrier();
  }
}

// Ablation of V6: kFakeGrid replaces the grid gather with arithmetic,
// kFakeX replaces the x loads with a constant.
template<bool kFakeGrid, bool kFakeX>
__global__ __launch_bounds__(256) void V7(const uint32_t* __restrict__ gate,
                                          const uint32_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  constexpr int kFetch = (kRowDwords + 31) / 32;
  __shared__ uint32_t buf[8][2][kRowDwords + 1];
  const int tid = threadIdx.x;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int row = blockIdx.x * 8 + wave;
  const half2* xg = reinterpret_cast<const half2*>(x);
  const size_t off = (static_cast<size_t>(expert) * kRows + row) * kRowDwords;
#pragma unroll
  for (int i = 0; i < kFetch; ++i) {
    const int idx = lane + 32 * i;
    if (idx < kRowDwords) {
      buf[wave][0][idx] = gate[off + idx];
      buf[wave][1][idx] = up[off + idx];
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  float acc[2] = {0.0f, 0.0f};
  const half2 xc = __floats2half2_rn(0.25f, -0.5f);
  for (int sb = lane; sb < kSubBlocks; sb += 32) {
    const half2* xs = xg + sb * 16;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const auto* b =
          reinterpret_cast<const uint8_t*>(buf[wave][m]) + (sb / 8) * kBlockBytes;
      half2 w[16];
      float scale = 0.0f;
      DecodeT<kFakeGrid>(b, sb % 8, grid, w, scale);
      float sdot = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        sdot = Dot2(w[i], kFakeX ? xc : xs[i], sdot);
      acc[m] = __fmaf_rn(scale, sdot, acc[m]);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc[0] += __shfl_xor(acc[0], o, 32);
    acc[1] += __shfl_xor(acc[1], o, 32);
  }
  if (lane == 0)
    out[static_cast<size_t>(blockIdx.y) * kRows + row] =
        (acc[0] / (1.0f + __expf(-acc[0]))) * acc[1];
}

// Variant 8: V6 with the next rows' loads in flight (registers) while the
// current rows decode from LDS.
template<int kRowsPerWave>
__global__ __launch_bounds__(256) void V8(const uint32_t* __restrict__ gate,
                                          const uint32_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  constexpr int kFetch = (kRowDwords + 31) / 32;
  __shared__ uint32_t buf[8][2][kRowDwords + 1];
  const int tid = threadIdx.x;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const half2* xg = reinterpret_cast<const half2*>(x);
  const int first = (blockIdx.x * 8 + wave) * kRowsPerWave;
  uint32_t pg[kFetch];
  uint32_t pu[kFetch];
  const auto fetch = [&](int row) {
    const size_t off = (static_cast<size_t>(expert) * kRows + row) * kRowDwords;
#pragma unroll
    for (int i = 0; i < kFetch; ++i) {
      const int idx = lane + 32 * i;
      pg[i] = idx < kRowDwords ? gate[off + idx] : 0U;
      pu[i] = idx < kRowDwords ? up[off + idx] : 0U;
    }
  };
  fetch(first);
  for (int rr = 0; rr < kRowsPerWave; ++rr) {
    const int row = first + rr;
#pragma unroll
    for (int i = 0; i < kFetch; ++i) {
      const int idx = lane + 32 * i;
      if (idx < kRowDwords) {
        buf[wave][0][idx] = pg[i];
        buf[wave][1][idx] = pu[i];
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    if (rr + 1 < kRowsPerWave)
      fetch(row + 1);
    float acc[2] = {0.0f, 0.0f};
    for (int sb = lane; sb < kSubBlocks; sb += 32) {
      const half2* xs = xg + sb * 16;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const auto* b =
            reinterpret_cast<const uint8_t*>(buf[wave][m]) + (sb / 8) * kBlockBytes;
        half2 w[16];
        float scale = 0.0f;
        Decode(b, sb % 8, grid, w, scale);
        float sdot = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          sdot = Dot2(w[i], xs[i], sdot);
        acc[m] = __fmaf_rn(scale, sdot, acc[m]);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      acc[0] += __shfl_xor(acc[0], o, 32);
      acc[1] += __shfl_xor(acc[1], o, 32);
    }
    if (lane == 0)
      out[static_cast<size_t>(blockIdx.y) * kRows + row] =
          (acc[0] / (1.0f + __expf(-acc[0]))) * acc[1];
    __builtin_amdgcn_wave_barrier();
  }
}

// Variant 9: V8 with 16-bit field reads and 128-bit x loads.
template<int kRowsPerWave>
__global__ __launch_bounds__(256) void V9(const uint32_t* __restrict__ gate,
                                          const uint32_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  constexpr int kFetch = (kRowDwords + 31) / 32;
  __shared__ uint32_t buf[8][2][kRowDwords + 1];
  const int tid = threadIdx.x;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const half2* xg = reinterpret_cast<const half2*>(x);
  const int first = (blockIdx.x * 8 + wave) * kRowsPerWave;
  uint32_t pg[kFetch];
  uint32_t pu[kFetch];
  const auto fetch = [&](int row) {
    const size_t off = (static_cast<size_t>(expert) * kRows + row) * kRowDwords;
#pragma unroll
    for (int i = 0; i < kFetch; ++i) {
      const int idx = lane + 32 * i;
      pg[i] = idx < kRowDwords ? gate[off + idx] : 0U;
      pu[i] = idx < kRowDwords ? up[off + idx] : 0U;
    }
  };
  fetch(first);
  for (int rr = 0; rr < kRowsPerWave; ++rr) {
    const int row = first + rr;
#pragma unroll
    for (int i = 0; i < kFetch; ++i) {
      const int idx = lane + 32 * i;
      if (idx < kRowDwords) {
        buf[wave][0][idx] = pg[i];
        buf[wave][1][idx] = pu[i];
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    if (rr + 1 < kRowsPerWave)
      fetch(row + 1);
    float acc[2] = {0.0f, 0.0f};
    for (int sb = lane; sb < kSubBlocks; sb += 32) {
      uint4 xv[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        xv[q] = reinterpret_cast<const uint4*>(xg + sb * 16)[q];
      const half2* xs = reinterpret_cast<const half2*>(xv);
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const auto* b =
            reinterpret_cast<const uint8_t*>(buf[wave][m]) + (sb / 8) * kBlockBytes;
        half2 w[16];
        float scale = 0.0f;
        DecodeW(b, sb % 8, grid, w, scale);
        float sdot = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          sdot = Dot2(w[i], xs[i], sdot);
        acc[m] = __fmaf_rn(scale, sdot, acc[m]);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      acc[0] += __shfl_xor(acc[0], o, 32);
      acc[1] += __shfl_xor(acc[1], o, 32);
    }
    if (lane == 0)
      out[static_cast<size_t>(blockIdx.y) * kRows + row] =
          (acc[0] / (1.0f + __expf(-acc[0]))) * acc[1];
    __builtin_amdgcn_wave_barrier();
  }
}

// Variant 10: V9 with the grid codebook in LDS (one 2 KB fill per block).
template<int kRowsPerWave>
__global__ __launch_bounds__(256) void V10(const uint32_t* __restrict__ gate,
                                          const uint32_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  constexpr int kFetch = (kRowDwords + 31) / 32;
  __shared__ uint32_t buf[8][2][kRowDwords + 1];
  __shared__ uint32_t grid_l[512];
  const int tid = threadIdx.x;
  for (int i = tid; i < 512; i += 256)
    grid_l[i] = grid[i];
  __syncthreads();
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const half2* xg = reinterpret_cast<const half2*>(x);
  const int first = (blockIdx.x * 8 + wave) * kRowsPerWave;
  uint32_t pg[kFetch];
  uint32_t pu[kFetch];
  const auto fetch = [&](int row) {
    const size_t off = (static_cast<size_t>(expert) * kRows + row) * kRowDwords;
#pragma unroll
    for (int i = 0; i < kFetch; ++i) {
      const int idx = lane + 32 * i;
      pg[i] = idx < kRowDwords ? gate[off + idx] : 0U;
      pu[i] = idx < kRowDwords ? up[off + idx] : 0U;
    }
  };
  fetch(first);
  for (int rr = 0; rr < kRowsPerWave; ++rr) {
    const int row = first + rr;
#pragma unroll
    for (int i = 0; i < kFetch; ++i) {
      const int idx = lane + 32 * i;
      if (idx < kRowDwords) {
        buf[wave][0][idx] = pg[i];
        buf[wave][1][idx] = pu[i];
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    if (rr + 1 < kRowsPerWave)
      fetch(row + 1);
    float acc[2] = {0.0f, 0.0f};
    for (int sb = lane; sb < kSubBlocks; sb += 32) {
      uint4 xv[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        xv[q] = reinterpret_cast<const uint4*>(xg + sb * 16)[q];
      const half2* xs = reinterpret_cast<const half2*>(xv);
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const auto* b =
            reinterpret_cast<const uint8_t*>(buf[wave][m]) + (sb / 8) * kBlockBytes;
        half2 w[16];
        float scale = 0.0f;
        DecodeW(b, sb % 8, grid_l, w, scale);
        float sdot = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          sdot = Dot2(w[i], xs[i], sdot);
        acc[m] = __fmaf_rn(scale, sdot, acc[m]);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      acc[0] += __shfl_xor(acc[0], o, 32);
      acc[1] += __shfl_xor(acc[1], o, 32);
    }
    if (lane == 0)
      out[static_cast<size_t>(blockIdx.y) * kRows + row] =
          (acc[0] / (1.0f + __expf(-acc[0]))) * acc[1];
    __builtin_amdgcn_wave_barrier();
  }
}

// Unaligned bytes [o, o + 4) of a dword-aligned row, o even.
__device__ __forceinline__ uint32_t LoadU32(const uint32_t* __restrict__ row,
                                            int o) {
  const uint32_t lo = row[o >> 2];
  if ((o & 3) == 0)
    return lo;
  const uint32_t hi = row[(o >> 2) + 1];
  return __builtin_amdgcn_alignbyte(hi, lo, o & 3);
}

// Variant 3: v1 with aligned dword loads (alignbyte for the 2-byte offsets).
template<bool kGridLds>
__global__ __launch_bounds__(256) void V3(const uint32_t* __restrict__ gate,
                                          const uint32_t* __restrict__ up,
                                          const uint32_t* __restrict__ grid_g,
                                          const half* __restrict__ x,
                                          const int* __restrict__ ids,
                                          float* __restrict__ out) {
  __shared__ uint32_t grid_l[512];
  __shared__ half2 x_l[kK / 2];
  const int tid = threadIdx.x;
  for (int i = tid; i < 512; i += 256)
    grid_l[i] = grid_g[i];
  for (int i = tid; i < kK / 2; i += 256)
    x_l[i] = reinterpret_cast<const half2*>(x)[i];
  __syncthreads();
  const uint32_t* grid = kGridLds ? grid_l : grid_g;
  const int expert = ids[blockIdx.y];
  const int lane = tid & 31;
  const int row = blockIdx.x * 8 + (tid >> 5);
  const uint32_t* rows[2] = {
      gate + (static_cast<size_t>(expert) * kRows + row) * kRowDwords,
      up + (static_cast<size_t>(expert) * kRows + row) * kRowDwords};
  float acc[2] = {0.0f, 0.0f};
  const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
  for (int sb = lane; sb < kSubBlocks; sb += 32) {
    const int blk = (sb / 8) * kBlockBytes;
    const int ib = sb % 8;
    const half2* xs = x_l + sb * 16;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const uint32_t* r = rows[m];
      const uint32_t q0 = LoadU32(r, blk + 2 + ib * 8);
      const uint32_t q1 = LoadU32(r, blk + 6 + ib * 8);
      const uint32_t sg = LoadU32(r, blk + 74 + ib * 4);
      const uint32_t qhw = LoadU32(r, blk + 66 + (ib & ~3));
      const uint32_t scw = LoadU32(r, blk + 106);
      const uint32_t dw = LoadU32(r, blk);
      const uint32_t qh = (qhw >> (8 * (ib & 3))) & 0xFFU;
      const uint32_t sc = (scw >> (8 * (ib / 2) + 4 * (ib % 2))) & 0x0FU;
      const float scale =
          __half2float(__ushort_as_half(static_cast<unsigned short>(dw & 0xFFFFU))) *
          static_cast<float>(1 + 2 * sc);
      float s = 0.0f;
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const uint32_t qsb = ((g < 4 ? q0 : q1) >> (8 * (g % 4))) & 0xFFU;
        const uint32_t index = qsb | ((qh << (8 - g)) & 256U);
        const uint32_t magnitudes = grid[index];
        const uint32_t bits = (sg >> (4 * g)) & 0xFU;
        const uint32_t neg = ((bits * 0x00204081U) & 0x01010101U);
        const uint32_t codes = ((magnitudes ^ (neg * 0xFFU)) + neg) ^ 0x80808080U;
        const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
        const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
        s = Dot2(__hadd2(__builtin_bit_cast(half2, p0), magic), xs[2 * g], s);
        s = Dot2(__hadd2(__builtin_bit_cast(half2, p1), magic), xs[2 * g + 1], s);
      }
      acc[m] = __fmaf_rn(scale, s, acc[m]);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc[0] += __shfl_xor(acc[0], o, 32);
    acc[1] += __shfl_xor(acc[1], o, 32);
  }
  if (lane == 0)
    out[static_cast<size_t>(blockIdx.y) * kRows + row] =
        (acc[0] / (1.0f + __expf(-acc[0]))) * acc[1];
}

template<typename Launch>
double TimeUs(Launch launch, int reps) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  for (int i = 0; i < 5; ++i)
    launch();
  CHECK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i)
    launch();
  CHECK(hipEventRecord(b));
  CHECK(hipEventSynchronize(b));
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, a, b));
  return 1000.0 * ms / reps;
}

int main() {
  const size_t matrix = static_cast<size_t>(kExperts) * kRows * kRowBytes;
  std::vector<uint8_t> host(matrix);
  std::mt19937 rng(7);
  for (auto& v : host)
    v = static_cast<uint8_t>(rng());
  // Plausible d: small positive halves.
  for (size_t blk = 0; blk < matrix / kBlockBytes; ++blk) {
    const __half d = __float2half(0.001f + 0.0001f * (blk % 7));
    std::memcpy(&host[blk * kBlockBytes], &d, 2);
  }
  std::vector<uint32_t> grid(512);
  for (auto& g : grid) {
    uint32_t w = 0;
    for (int j = 0; j < 4; ++j)
      w |= static_cast<uint32_t>(1 + 2 * (rng() % 8)) << (8 * j);
    g = w;
  }
  std::vector<half> xh(kK);
  for (auto& v : xh)
    v = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f);
  // Many routed sets, rotated per call, so each call reads cold rows
  // (51 sets x 10 experts x 1.4 MB far exceeds the 32 MB last-level cache).
  constexpr int kSets = 51;
  std::vector<int> ids(kUsed * kSets);
  for (int i = 0; i < kUsed * kSets; ++i)
    ids[i] = (i * 7 + 3) % kExperts;

  uint8_t *gate, *up;
  uint32_t* grid_d;
  half* x;
  int* ids_d;
  float* out;
  CHECK(hipMalloc(&gate, matrix));
  CHECK(hipMalloc(&up, matrix));
  CHECK(hipMalloc(&grid_d, 512 * 4));
  CHECK(hipMalloc(&x, kK * 2));
  CHECK(hipMalloc(&ids_d, ids.size() * 4));
  CHECK(hipMalloc(&out, kUsed * kRows * 4));
  CHECK(hipMemcpy(gate, host.data(), matrix, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(up, host.data(), matrix, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(grid_d, grid.data(), 512 * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(x, xh.data(), kK * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(ids_d, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));

  const double bytes = 2.0 * kUsed * kRows * kRowBytes;
  const auto report = [&](const char* name, double us) {
    std::printf("%-28s %8.2f us  %7.1f GB/s  %5.1f %% of 225.7\n", name, us,
                bytes / us / 1e3, 100.0 * bytes / us / 1e3 / 225.7);
  };
  const int reps = 200;
  int call = 0;
  const auto set = [&] { return ids_d + (call++ % kSets) * kUsed; };
  report("read-only ceiling", TimeUs([&] {
    ReadOnly<<<dim3(kRows / 8, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), set(), out);
  }, reps));
  report("v1 rpw=1 grid lds", TimeUs([&] {
    V1<1, true><<<dim3(kRows / 8, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v1 rpw=1 grid global", TimeUs([&] {
    V1<1, false><<<dim3(kRows / 8, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v3 dword grid lds", TimeUs([&] {
    V3<true><<<dim3(kRows / 8, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v3 dword grid global", TimeUs([&] {
    V3<false><<<dim3(kRows / 8, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v4 no-lds rpw=1", TimeUs([&] {
    V4<1><<<dim3(kRows / 8, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v4 no-lds rpw=4", TimeUs([&] {
    V4<4><<<dim3(kRows / 32, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v5 16 lanes bytes", TimeUs([&] {
    V5<16, false><<<dim3(kRows / 16, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v5 16 lanes dwords", TimeUs([&] {
    V5<16, true><<<dim3(kRows / 16, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v5 8 lanes dwords", TimeUs([&] {
    V5<8, true><<<dim3(kRows / 32, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v5 32 lanes dwords", TimeUs([&] {
    V5<32, true><<<dim3(kRows / 8, kUsed), 256>>>(gate, up, grid_d, x, set(), out);
  }, reps));
  report("v6 lds rows rpw=1", TimeUs([&] {
    V6<1><<<dim3(kRows / 8, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v6 lds rows rpw=2", TimeUs([&] {
    V6<2><<<dim3(kRows / 16, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  const auto run7 = [&](const char* name, auto kernel) {
    report(name, TimeUs([&] {
      kernel<<<dim3(kRows / 8, kUsed), 256>>>(
          reinterpret_cast<const uint32_t*>(gate),
          reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
    }, reps));
  };
  run7("v7 real", V7<false, false>);
  run7("v7 fake grid", V7<true, false>);
  run7("v7 fake x", V7<false, true>);
  run7("v7 fake grid + x", V7<true, true>);
  report("v8 prefetch rpw=2", TimeUs([&] {
    V8<2><<<dim3(kRows / 16, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v8 prefetch rpw=4", TimeUs([&] {
    V8<4><<<dim3(kRows / 32, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v8 prefetch rpw=8", TimeUs([&] {
    V8<8><<<dim3(kRows / 64, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v9 rpw=8", TimeUs([&] {
    V9<8><<<dim3(kRows / 64, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v9 rpw=4", TimeUs([&] {
    V9<4><<<dim3(kRows / 32, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v10 lds grid rpw=8", TimeUs([&] {
    V10<8><<<dim3(kRows / 64, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v10 lds grid rpw=16", TimeUs([&] {
    V10<16><<<dim3(kRows / 128, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  report("v9 rpw=16", TimeUs([&] {
    V9<16><<<dim3(kRows / 128, kUsed), 256>>>(
        reinterpret_cast<const uint32_t*>(gate),
        reinterpret_cast<const uint32_t*>(up), grid_d, x, set(), out);
  }, reps));
  return 0;
}
