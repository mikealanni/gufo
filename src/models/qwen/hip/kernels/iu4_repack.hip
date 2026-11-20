#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/ops/iu4.hpp"

namespace gufo::hip {
namespace {

__global__ void Iu4RepackKernel(const std::uint8_t* __restrict__ blocks,
                                std::uint8_t* __restrict__ tiles,
                                std::uint16_t* __restrict__ scales,
                                unsigned long long* __restrict__ bad,
                                std::uint32_t rows, std::uint32_t groups) {
  const std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = (std::size_t)groups * rows * 4;
  if (idx >= total)
    return;
  const std::uint32_t q = idx & 3;
  const std::size_t er = idx >> 2;
  const std::uint32_t row = er % rows;
  const std::uint32_t e = er / rows;
  const std::uint8_t* src =
      blocks + ((std::size_t)row * (groups >> 1) + (e >> 1)) * kIu4BlockBytes;
  const std::uint32_t half = e & 1;
  if (q == 0) {
    std::uint16_t sc, zp;
    __builtin_memcpy(&sc, src + 4 * half, 2);
    __builtin_memcpy(&zp, src + 4 * half + 2, 2);
    scales[(std::size_t)e * rows + row] = sc;
    const float fs = __half2float(__ushort_as_half(sc));
    const float fz = __half2float(__ushort_as_half(zp));
    if (fabsf(fz + 8.0f * fs) > 1e-3f * fabsf(8.0f * fs) + 1e-12f)
      atomicAdd(bad, 1ull);
  }
  uint4 v;
  __builtin_memcpy(&v, src + 8 + 64 * half + 16 * q, 16);
  v.x ^= 0x88888888u;
  v.y ^= 0x88888888u;
  v.z ^= 0x88888888u;
  v.w ^= 0x88888888u;
  const std::size_t dst =
      (((std::size_t)e * (rows >> 4) + (row >> 4)) * 4 + q) * 256 +
      (row & 15) * 16;
  __builtin_memcpy(tiles + dst, &v, 16);
}

}  // namespace

void LaunchIu4RepackWeights(const void* blocks, void* tiles,
                            std::uint16_t* scales,
                            unsigned long long* bad_zero_points,
                            std::size_t rows, std::size_t cols,
                            hipStream_t stream) {
  const std::uint32_t groups = static_cast<std::uint32_t>(cols >> 7);
  const std::size_t total = (std::size_t)groups * rows * 4;
  Iu4RepackKernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(blocks),
      static_cast<std::uint8_t*>(tiles), scales, bad_zero_points,
      static_cast<std::uint32_t>(rows), groups);
}

}  // namespace gufo::hip
