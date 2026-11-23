#include <hip/hip_runtime.h>
#include <cstdio>
#include <random>
#include <vector>
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
__global__ void __launch_bounds__(256) k(const _Float16* d, float* out, int iters) {
  v16h a, b;
  const _Float16* p = d + (threadIdx.x & 31) * 16;
  for (int i = 0; i < 16; ++i) { a[i] = p[i]; b[i] = p[i + 512]; }
  v8f acc[12];
  for (int j = 0; j < 12; ++j) for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 12; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[j]);
    // perturb operands so the data toggles like real data
    a[it & 15] += (_Float16)0.001f;
  }
  float s = 0; for (int j = 0; j < 12; ++j) for (int i = 0; i < 8; ++i) s += acc[j][i];
  if (s == 123.456f) out[0] = s;
}
int main(int argc, char** argv) {
  int waves_per_block = 8;
  std::vector<_Float16> h(2048); std::mt19937 r(1); std::uniform_real_distribution<float> u(-1, 1);
  for (auto& x : h) x = (_Float16)u(r);
  _Float16* d; float* o; hipMalloc(&d, 4096); hipMalloc(&o, 4); hipMemcpy(d, h.data(), 4096, hipMemcpyHostToDevice);
  for (int blocks : {80, 160, 320, 640}) {
    int iters = 20000;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernelGGL(k, dim3(blocks), dim3(256), 0, 0, d, o, 100);
    hipDeviceSynchronize();
    hipEventRecord(a);
    hipLaunchKernelGGL(k, dim3(blocks), dim3(256), 0, 0, d, o, iters);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    double flops = 2.0 * 16 * 16 * 16 * 12.0 * iters * blocks * waves_per_block;
    printf("blocks %d: %.2f ms  %.1f TFLOPS\n", blocks, ms, flops / ms / 1e9);
  }
}
