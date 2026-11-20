// IQ4_NL routed-expert down projection at decode (2560 rows x K = 640, 10 of
// 512 experts, 48 layers per iteration) on gfx1151. Three kernels share one
// memory access pattern, so the GB/s differences isolate the arithmetic:
//
//   mode 0  read-only floor: streams the same bytes, almost no ALU work
//   mode 1  current style: Q8_1 activations, LUT to int8, v_dot4 (dp4a)
//   mode 2  T-MAC style: LUT to exact F16 weights, F16 activations, v_dot2
//   mode 3  wide-load floor: the same bytes as contiguous 16-byte loads
//   mode 4  T-MAC proper: LUT bytes to F32, F32 activations, plain FMA
//
// If mode 1 already sits at the mode 0 floor, an F16/FMA decode cannot help.
//
//   tools/bench/build.sh iq4nl_down_decode_bench && /tmp/iq4nl_down_decode_bench
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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
constexpr int kLayers = 48;
constexpr int kRows = 2560;
constexpr int kK = 640;
constexpr int kBlocks = kK / 32;           // 20
constexpr int kBlockBytes = 18;            // half d + 16 nibble bytes
constexpr int kRowBytes = kBlocks * kBlockBytes;  // 360
constexpr int kRowsPerWave = 4;
constexpr int kUnitsPerRow = kBlocks * 2;  // 8 bytes (16 weights) per unit
constexpr int kUnitsPerWave = kRowsPerWave * kUnitsPerRow;  // 160 = 5 per lane
constexpr int kWavesPerBlock = 8;

constexpr int8_t kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                1,    13,   25,  38,  53,  69,  89,  113};

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
  return v;
}

__device__ __forceinline__ float Dot2(half2 a, half2 b, float c) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
  return c;
}

// Four IQ4_NL codes (one nibble per byte) to kvalues_iq4nl + 128, as bytes.
__device__ __forceinline__ uint32_t Iq4NlBytes(uint32_t nibbles) {
  constexpr uint32_t kT0 = 0x3F2D1801U;
  constexpr uint32_t kT1 = 0x766A5D4FU;
  constexpr uint32_t kT2 = 0xA6998D81U;
  constexpr uint32_t kT3 = 0xF1D9C5B5U;
  const uint32_t sel = nibbles & 0x07070707U;
  const uint32_t lo = __builtin_amdgcn_perm(kT1, kT0, sel);
  const uint32_t hi = __builtin_amdgcn_perm(kT3, kT2, sel);
  const uint32_t mask = ((nibbles >> 3U) & 0x01010101U) * 0xFFU;
  return (hi & mask) | (lo & ~mask);
}

__device__ __forceinline__ int Dp4a(uint32_t a, uint32_t b, int c) {
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true,
                                 static_cast<int>(b), c, false);
}

struct Q8Block {
  half d;
  half s;
  int8_t qs[32];
};
static_assert(sizeof(Q8Block) == 36, "block_q8_1");

template <int kMode>
__launch_bounds__(32 * kWavesPerBlock) __global__
    void Down(const uint8_t* __restrict__ weights,  // [expert][row][360]
              const int* __restrict__ experts,      // this layer's 10 experts
              const Q8Block* __restrict__ act_q,    // [slot][20]
              const half* __restrict__ act_h,       // [slot][640]
              const float* __restrict__ act_f,      // [slot][640]
              float* __restrict__ out) {            // [slot][2560]
  const int slot = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * kWavesPerBlock + wave) * kRowsPerWave;
  const int expert = experts[slot];
  const uint8_t* base =
      weights + (static_cast<size_t>(expert) * kRows + row0) * kRowBytes;
  float acc[kRowsPerWave] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < kUnitsPerWave / 32; ++i) {
    const int u = lane + 32 * i;
    const int rr = u / kUnitsPerRow;
    const int unit = u % kUnitsPerRow;
    const int b = unit >> 1;
    const int h = unit & 1;
    const uint8_t* blk = base + static_cast<size_t>(rr) * kRowBytes +
                         b * kBlockBytes;
    const uint16_t* hp = reinterpret_cast<const uint16_t*>(blk);
    const float d = __half2float(__ushort_as_half(hp[0]));
    // 8 nibble bytes at +2 + 8h, two-byte aligned: read as four halves.
    const uint32_t w0 = hp[1 + 4 * h] | (static_cast<uint32_t>(hp[2 + 4 * h]) << 16);
    const uint32_t w1 = hp[3 + 4 * h] | (static_cast<uint32_t>(hp[4 + 4 * h]) << 16);
    float c = 0.f;
    if constexpr (kMode == 0) {
      c = __uint_as_float((w0 ^ (w1 << 1)) & 0x3F7FFFFFu) * d;
    } else if constexpr (kMode == 1) {
      const Q8Block& a = act_q[slot * kBlocks + b];
      const uint32_t* ai = reinterpret_cast<const uint32_t*>(a.qs);
      const uint32_t lo0 = Iq4NlBytes(w0 & 0x0F0F0F0FU) ^ 0x80808080U;
      const uint32_t hi0 = Iq4NlBytes((w0 >> 4) & 0x0F0F0F0FU) ^ 0x80808080U;
      const uint32_t lo1 = Iq4NlBytes(w1 & 0x0F0F0F0FU) ^ 0x80808080U;
      const uint32_t hi1 = Iq4NlBytes((w1 >> 4) & 0x0F0F0F0FU) ^ 0x80808080U;
      int sumi = Dp4a(lo0, ai[2 * h], 0);
      sumi = Dp4a(lo1, ai[2 * h + 1], sumi);
      sumi = Dp4a(hi0, ai[4 + 2 * h], sumi);
      sumi = Dp4a(hi1, ai[4 + 2 * h + 1], sumi);
      c = d * __half2float(a.d) * static_cast<float>(sumi);
    } else if constexpr (kMode == 4) {
      const float4* xl = reinterpret_cast<const float4*>(act_f + slot * kK + b * 32 + 8 * h);
      const float4* xh = reinterpret_cast<const float4*>(act_f + slot * kK + b * 32 + 16 + 8 * h);
      const float4 xa[4] = {xl[0], xl[1], xh[0], xh[1]};
      const uint32_t words[4] = {w0 & 0x0F0F0F0FU, w1 & 0x0F0F0F0FU,
                                 (w0 >> 4) & 0x0F0F0F0FU, (w1 >> 4) & 0x0F0F0F0FU};
      float sum = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t bytes = Iq4NlBytes(words[q]);
        const float4 x = xa[q];
        sum = fmaf(static_cast<float>((bytes >> 0) & 0xFFu) - 128.0f, x.x, sum);
        sum = fmaf(static_cast<float>((bytes >> 8) & 0xFFu) - 128.0f, x.y, sum);
        sum = fmaf(static_cast<float>((bytes >> 16) & 0xFFu) - 128.0f, x.z, sum);
        sum = fmaf(static_cast<float>((bytes >> 24) & 0xFFu) - 128.0f, x.w, sum);
      }
      c = d * sum;
    } else {
      const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
      const uint4 xl = *reinterpret_cast<const uint4*>(
          act_h + slot * kK + b * 32 + 8 * h);
      const uint4 xh = *reinterpret_cast<const uint4*>(
          act_h + slot * kK + b * 32 + 16 + 8 * h);
      const uint32_t xlw[4] = {xl.x, xl.y, xl.z, xl.w};
      const uint32_t xhw[4] = {xh.x, xh.y, xh.z, xh.w};
      float sum = 0.f;
      const uint32_t words[4] = {w0 & 0x0F0F0F0FU, w1 & 0x0F0F0F0FU,
                                 (w0 >> 4) & 0x0F0F0F0FU,
                                 (w1 >> 4) & 0x0F0F0F0FU};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t bytes = Iq4NlBytes(words[q]);
        const uint32_t p0 = __builtin_amdgcn_perm(bytes, 0x64646464U, 0x01050004U);
        const uint32_t p1 = __builtin_amdgcn_perm(bytes, 0x64646464U, 0x03070206U);
        const half2 wa = __hadd2(__builtin_bit_cast(half2, p0), magic);
        const half2 wb = __hadd2(__builtin_bit_cast(half2, p1), magic);
        // q 0,1: low nibbles (xl); q 2,3: high nibbles (xh); odd q: second word.
        const uint32_t* xs = q < 2 ? xlw : xhw;
        const int xo = (q & 1) * 2;
        sum = Dot2(wa, __builtin_bit_cast(half2, xs[xo]), sum);
        sum = Dot2(wb, __builtin_bit_cast(half2, xs[xo + 1]), sum);
      }
      c = d * sum;
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWave; ++r) acc[r] += (rr == r) ? c : 0.f;
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWave; ++r) {
    const float s = WaveSum(acc[r]);
    if (lane == 0) out[static_cast<size_t>(slot) * kRows + row0 + r] = s;
  }
}

// Streams the 32 rows of this block (11,520 contiguous bytes) as 16-byte loads.
__launch_bounds__(32 * kWavesPerBlock) __global__
    void Stream(const uint8_t* __restrict__ weights, const int* __restrict__ experts,
                float* __restrict__ out) {
  const int slot = blockIdx.y;
  const size_t bytes = static_cast<size_t>(kWavesPerBlock) * kRowsPerWave * kRowBytes;
  const uint4* p = reinterpret_cast<const uint4*>(
      weights + (static_cast<size_t>(experts[slot]) * kRows +
                 static_cast<size_t>(blockIdx.x) * kWavesPerBlock * kRowsPerWave) * kRowBytes);
  uint32_t x = 0;
  for (size_t i = threadIdx.x; i < bytes / 16; i += 32 * kWavesPerBlock) {
    const uint4 v = p[i];
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x12345678u) out[0] = 1.f;
}

int main() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  const size_t weight_bytes =
      static_cast<size_t>(kExperts) * kRows * kRowBytes;  // ~472 MB
  std::vector<uint8_t> w(weight_bytes);
  for (size_t b = 0; b < weight_bytes / kBlockBytes; ++b) {
    uint8_t* p = w.data() + b * kBlockBytes;
    const half d = __float2half(0.004f + 0.004f * (rng() & 255) / 255.0f);
    std::memcpy(p, &d, 2);
    for (int j = 0; j < 16; ++j) p[2 + j] = static_cast<uint8_t>(rng());
  }
  std::vector<half> xh(kUsed * kK);
  std::vector<Q8Block> xq(kUsed * kBlocks);
  for (int s = 0; s < kUsed; ++s) {
    for (int b = 0; b < kBlocks; ++b) {
      float amax = 0;
      float v[32];
      for (int e = 0; e < 32; ++e) {
        xh[s * kK + b * 32 + e] = __float2half(ud(rng));
        v[e] = __half2float(xh[s * kK + b * 32 + e]);
        amax = std::fmax(amax, std::fabs(v[e]));
      }
      const float dq = amax / 127.0f;
      Q8Block& q = xq[s * kBlocks + b];
      int sum = 0;
      for (int e = 0; e < 32; ++e) {
        const int qi = dq != 0 ? static_cast<int>(std::lround(v[e] / dq)) : 0;
        q.qs[e] = static_cast<int8_t>(qi);
        sum += qi;
      }
      q.d = __float2half(dq);
      q.s = __float2half(dq * sum);
    }
  }
  std::vector<int> experts(kLayers * kUsed);
  for (int l = 0; l < kLayers; ++l) {
    std::vector<int> pick(kExperts);
    for (int e = 0; e < kExperts; ++e) pick[e] = e;
    std::shuffle(pick.begin(), pick.end(), rng);
    for (int s = 0; s < kUsed; ++s) experts[l * kUsed + s] = pick[s];
  }

  uint8_t* d_w;
  half* d_xh;
  float* d_xf;
  Q8Block* d_xq;
  int* d_ex;
  float* d_out;
  CHECK(hipMalloc(&d_w, weight_bytes));
  CHECK(hipMalloc(&d_xh, xh.size() * sizeof(half)));
  CHECK(hipMalloc(&d_xf, xh.size() * sizeof(float)));
  CHECK(hipMalloc(&d_xq, xq.size() * sizeof(Q8Block)));
  CHECK(hipMalloc(&d_ex, experts.size() * sizeof(int)));
  CHECK(hipMalloc(&d_out, kUsed * kRows * sizeof(float)));
  CHECK(hipMemcpy(d_w, w.data(), weight_bytes, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_xh, xh.data(), xh.size() * sizeof(half), hipMemcpyHostToDevice));
  {
    std::vector<float> xf(xh.size());
    for (size_t i = 0; i < xh.size(); ++i) xf[i] = __half2float(xh[i]);
    CHECK(hipMemcpy(d_xf, xf.data(), xf.size() * sizeof(float), hipMemcpyHostToDevice));
  }
  CHECK(hipMemcpy(d_xq, xq.data(), xq.size() * sizeof(Q8Block), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_ex, experts.data(), experts.size() * sizeof(int), hipMemcpyHostToDevice));

  const dim3 grid(kRows / (kWavesPerBlock * kRowsPerWave), kUsed);
  const dim3 block(32 * kWavesPerBlock);
  const auto launch = [&](int mode, int layer) {
    const int* ex = d_ex + layer * kUsed;
    if (mode == 0) Down<0><<<grid, block>>>(d_w, ex, d_xq, d_xh, d_xf, d_out);
    if (mode == 1) Down<1><<<grid, block>>>(d_w, ex, d_xq, d_xh, d_xf, d_out);
    if (mode == 2) Down<2><<<grid, block>>>(d_w, ex, d_xq, d_xh, d_xf, d_out);
    if (mode == 3) Stream<<<grid, block>>>(d_w, ex, d_out);
    if (mode == 4) Down<4><<<grid, block>>>(d_w, ex, d_xq, d_xh, d_xf, d_out);
  };

  // Accuracy against a double reference of layer 0 (exact F16 activations).
  std::vector<double> ref(kUsed * kRows);
  for (int s = 0; s < kUsed; ++s) {
    const int e = experts[s];
    for (int r = 0; r < kRows; ++r) {
      const uint8_t* row = w.data() + (static_cast<size_t>(e) * kRows + r) * kRowBytes;
      double acc = 0;
      for (int b = 0; b < kBlocks; ++b) {
        const uint8_t* blk = row + b * kBlockBytes;
        half dh;
        std::memcpy(&dh, blk, 2);
        double sb = 0;
        for (int j = 0; j < 16; ++j) {
          sb += kValues[blk[2 + j] & 15] * static_cast<double>(__half2float(xh[s * kK + b * 32 + j]));
          sb += kValues[blk[2 + j] >> 4] * static_cast<double>(__half2float(xh[s * kK + b * 32 + 16 + j]));
        }
        acc += static_cast<double>(__half2float(dh)) * sb;
      }
      ref[s * kRows + r] = acc;
    }
  }
  double norm = 0;
  for (double v : ref) norm = std::fmax(norm, std::fabs(v));
  std::vector<float> got(kUsed * kRows);
  for (int mode : {1, 2, 4}) {
    CHECK(hipMemset(d_out, 0, got.size() * sizeof(float)));
    launch(mode, 0);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(got.data(), d_out, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    double err = 0;
    for (size_t i = 0; i < got.size(); ++i) err = std::fmax(err, std::fabs(got[i] - ref[i]));
    std::printf("mode %d max |error| %.5f (reference max %.3f, relative %.2e)\n",
                mode, err, norm, err / norm);
  }

  const double bytes_per_iter =
      static_cast<double>(kLayers) * kUsed * kRows * kRowBytes;
  const char* names[5] = {"read-only floor", "dp4a + Q8_1 (current style)",
                          "F16 LUT + v_dot2 (T-MAC style)",
                          "wide-load streaming floor",
                          "F32 LUT + FMA (T-MAC proper)"};
  double best[5] = {1e9, 1e9, 1e9, 1e9, 1e9};
  hipEvent_t t0, t1;
  CHECK(hipEventCreate(&t0));
  CHECK(hipEventCreate(&t1));
  for (int round = 0; round < 5; ++round) {
    for (int mode = 0; mode < 5; ++mode) {
      for (int w_ = 0; w_ < 2; ++w_)
        for (int l = 0; l < kLayers; ++l) launch(mode, l);
      CHECK(hipDeviceSynchronize());
      CHECK(hipEventRecord(t0));
      for (int it = 0; it < 20; ++it)
        for (int l = 0; l < kLayers; ++l) launch(mode, l);
      CHECK(hipEventRecord(t1));
      CHECK(hipEventSynchronize(t1));
      float ms = 0;
      CHECK(hipEventElapsedTime(&ms, t0, t1));
      best[mode] = std::fmin(best[mode], ms / 20.0);
    }
  }
  std::printf("\nper token: %d layers x %d experts, %.1f MB of weights\n",
              kLayers, kUsed, bytes_per_iter / 1e6);
  for (int mode = 0; mode < 5; ++mode)
    std::printf("  %-32s %7.3f ms  %6.1f GB/s\n", names[mode], best[mode],
                bytes_per_iter / (best[mode] * 1e6));
  return 0;
}
