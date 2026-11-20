// Dense-GEMV compression-tradeoff microbench for gfx1151.
// Answers: for the dense decode GEMV (the ~40%-of-traffic, near-BW-peak path),
// does storing weights in FEWER bits reduce kernel time, or does the extra
// dequant ALU put the kernel under the DRAM line? Reports bytes/weight vs
// achieved GB/s vs a read-only streaming ceiling at each size.
//
// Build (no model, tiny memory, runs fine standalone):
//   hipcc -O3 --offload-arch=gfx1151 -std=c++17 \
//     tools/bench/dense_gemv_bpw_bench.hip -o /tmp/dense_bpw_bench
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                       \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                 \
                   hipGetErrorString(e_));                                   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// Logical dense shape: rows = 2560 (hidden), K = 2560 (one GEMV, one token).
constexpr int kRows = 2560;
constexpr int kK = 2560;
constexpr int kRowSets = 16;   // rotate so each call reads a cold row window

__device__ __forceinline__ int Dp4a(int a, int b, int c) {
  int r;
  asm volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
}

// Activation block Q8_1: half ds (d) + int8 qs[32]; one per 32 weights.
struct BlockQ8_1 {
  half ds[2];
  int8_t qs[32];
};

// ---- weight formats (byte size per 32-weight block is what hits DRAM) ----
struct BlockQ8_0 {  // 34 B / 32 = 8.5 bpw, dequant-free int8 (dp4a)
  half d;
  int8_t qs[32];
};
struct BlockQ5_1 {  // 24 B / 32 = 6 bpw, current dense decode format
  uint8_t qs[16];   // low nibbles
  uint8_t qh[4];    // high bits
  half2 dmin;
};
struct BlockQ4_1 {  // 20 B / 32 = 5 bpw
  uint8_t qs[16];
  half2 ds;  // d, min
};
struct BlockQ4_0 {  // 18 B / 32 = 4.5 bpw
  half d;
  uint8_t qs[16];
};

__device__ __forceinline__ uint32_t Int4(const void* p, int i) {
  return reinterpret_cast<const uint32_t*>(p)[i];
}

// Each wave owns one row; lanes stride the K/32 blocks. Activation in LDS.
template <typename WT, int kBpwX8>
__global__ __launch_bounds__(256) void Gemv(
    const WT* __restrict__ w, const BlockQ8_1* __restrict__ x,
    const int* __restrict__ rowset, float* __restrict__ out) {
  __shared__ BlockQ8_1 xl[kK / 32];
  const int tid = threadIdx.x;
  for (int i = tid; i < kK / 32; i += 256) xl[i] = x[i];
  __syncthreads();
  const int row = blockIdx.x * (256 / 32) + (tid >> 5);
  const int lane = tid & 31;
  const int off = rowset[blockIdx.y];  // cold window base (in rows)
  const WT* wr = w + (static_cast<size_t>(off) + row) * (kK / 32);
  int sum = 0;
  const int nblocks = kK / 32;
  for (int kb = lane; kb < nblocks; kb += 32) {
    const WT b = wr[kb];
    const BlockQ8_1 xa = xl[kb];
    // dequant weight nibbles/bytes -> int8 lanes, dp4a against activation.
    // The unpack ALU grows as bpw drops; that is the cost being measured.
    if constexpr (std::is_same_v<WT, BlockQ8_0>) {
      const uint32_t w0 = Int4(b.qs, 0), w1 = Int4(b.qs, 1);
      const uint32_t w2 = Int4(b.qs, 2), w3 = Int4(b.qs, 3);
      sum = Dp4a(w0, Int4(xa.qs, 0), sum);
      sum = Dp4a(w1, Int4(xa.qs, 1), sum);
      sum = Dp4a(w2, Int4(xa.qs, 2), sum);
      sum = Dp4a(w3, Int4(xa.qs, 3), sum);
    } else if constexpr (std::is_same_v<WT, BlockQ5_1>) {
      const uint32_t q = Int4(b.qs, 0);
      const uint32_t h = Int4(b.qh, 0);
      const uint32_t xl0 = Int4(xa.qs, 0), xl1 = Int4(xa.qs, 1);
      uint32_t lo = q & 0x0F0F0F0F;
      uint32_t hi = (q >> 4) & 0x0F0F0F0F;
      lo |= ((h & 0x00000001) ? 0x10101010 : 0) & 0x00000011;  // representative
      sum = Dp4a(lo, xl0, sum);
      sum = Dp4a(hi, xl1, sum);
    } else if constexpr (std::is_same_v<WT, BlockQ4_1> ||
                         std::is_same_v<WT, BlockQ4_0>) {
      const uint32_t q = Int4(b.qs, 0);
      uint32_t lo = q & 0x0F0F0F0F;
      uint32_t hi = (q >> 4) & 0x0F0F0F0F;
      sum = Dp4a(lo, Int4(xa.qs, 0), sum);
      sum = Dp4a(hi, Int4(xa.qs, 1), sum);
    }
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
  if (lane == 0) out[static_cast<size_t>(blockIdx.y) * kRows + row] = sum;
}

// Read-only ceiling: stream the same bytes, no dot, no dequant.
template <int kBpwX8>
__global__ __launch_bounds__(256) void ReadOnly(const uint8_t* __restrict__ w,
                                                const int* __restrict__ rowset,
                                                float* __restrict__ out) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const uint8_t* wr = w + (static_cast<size_t>(rowset[blockIdx.y]) + row) *
                              (kK * kBpwX8 / 8);
  uint32_t acc = 0;
  const int dw = kK * kBpwX8 / 8 / 4;
  const uint32_t* p = reinterpret_cast<const uint32_t*>(wr);
  for (int i = lane; i < dw; i += 32) acc += p[i];
  if (acc == 0x1234u) out[row] = 1.f;
}

template <typename F>
double TimeUs(F f, int reps) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  for (int i = 0; i < 5; ++i) f();
  CHECK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i) f();
  CHECK(hipEventRecord(b));
  CHECK(hipEventSynchronize(b));
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, a, b));
  return 1000.0 * ms / reps;
}

template <typename WT, int kBytesPerBlk>
void run(const char* name, std::mt19937& rng) {
  const int blk_per_row = kK / 32;
  const size_t rows_total = static_cast<size_t>(kRowSets) * kRows;
  const size_t wbytes = rows_total * blk_per_row * sizeof(WT);
  WT* wd;
  BlockQ8_1* xd;
  int* rsd;
  float* outd;
  CHECK(hipMalloc(&wd, wbytes));
  CHECK(hipMalloc(&xd, (kK / 32) * sizeof(BlockQ8_1)));
  CHECK(hipMalloc(&rsd, kRowSets * sizeof(int)));
  CHECK(hipMalloc(&outd, kRowSets * kRows * sizeof(float)));
  std::vector<uint8_t> host(wbytes);
  for (auto& v : host) v = static_cast<uint8_t>(rng());
  CHECK(hipMemcpy(wd, host.data(), wbytes, hipMemcpyHostToDevice));
  std::vector<int> rs(kRowSets);
  for (int i = 0; i < kRowSets; ++i) rs[i] = i * kRows;
  CHECK(hipMemcpy(rsd, rs.data(), rs.size() * 4, hipMemcpyHostToDevice));
  int call = 0;
  const auto set = [&] { return rsd + (call++ % kRowSets); };
  const double us_g = TimeUs(
      [&] {
        Gemv<WT, kBytesPerBlk * 8 / 32>
            <<<kRows / 8, 256>>>(wd, reinterpret_cast<BlockQ8_1*>(xd), set(),
                                 outd);
      },
      200);
  const double us_ro = TimeUs(
      [&] {
        ReadOnly<kBytesPerBlk * 8 / 32><<<kRows / 8, 256>>>(
            reinterpret_cast<uint8_t*>(wd), set(), outd);
      },
      200);
  const double bytes = static_cast<double>(kRows) * kK * kBytesPerBlk / 32.0;
  std::printf(
      "%-10s %5.1f bpw  gemv %7.2f us %6.1f GB/s   ROceil %7.2f us %6.1f "
      "GB/s  (%.1f%% of RO)\n",
      name, kBytesPerBlk * 8.0 / 32.0, us_g, bytes / us_g / 1e3, us_ro,
      bytes / us_ro / 1e3, 100.0 * us_ro / us_g);
  hipFree(wd);
  hipFree(xd);
  hipFree(rsd);
  hipFree(outd);
}

int main() {
  std::mt19937 rng(7);
  std::printf(
      "dense GEMV shape rows=%d K=%d, rotate %d cold windows. GB/s of weight "
      "traffic.\n",
      kRows, kK, kRowSets);
  run<BlockQ8_0, 34>("Q8_0", rng);
  run<BlockQ5_1, 24>("Q5_1*", rng);  // * dequant approximated
  run<BlockQ4_1, 20>("Q4_1", rng);
  run<BlockQ4_0, 18>("Q4_0", rng);
  std::printf(
      "\nIf GB/s stays ~flat as bpw drops, fewer bits cut time ~proportionally "
      "(BW-bound).\nIf GB/s falls with bpw, dequant ALU is the new wall -> "
      "compression stops paying.\n");
  return 0;
}
