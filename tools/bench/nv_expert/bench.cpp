#ifdef PAIRV
#define PAIRTAG "pair_"
#else
#define PAIRTAG ""
#endif
#ifndef BMV
#define BMV 128
#endif
#ifndef BNV
#define BNV 192
#endif
#include "kernel_extract.inc"
#include <random>
#include <string>
using namespace nvx;
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

struct Problem {
  int ne, m, k, tokens;
  std::vector<int> count, bounds;   // padded bounds
  std::int32_t *d_tiles{}, *d_bounds{}, *d_rows_in{}, *d_rows_out{};
  void* d_w{}; __half* d_x{}; __half* d_out{}; float* d_scale{};
  int n_tiles{0}; int bn{192};
  size_t row_bytes{};
};

static void build(Problem& p, int ne, int m, int k, int tokens, int topk, int bn, unsigned seed) {
  p.ne = ne; p.m = m; p.k = k; p.tokens = tokens; p.bn = bn;
  p.row_bytes = ((k / 2 + k / 16 + 15) / 16) * 16;
  std::mt19937 rng(seed);
  // each token picks topk distinct experts
  std::vector<std::vector<int>> per(ne);
  std::vector<int> ids(ne);
  for (int t = 0; t < tokens; ++t) {
    for (int i = 0; i < ne; ++i) ids[i] = i;
    for (int s = 0; s < topk; ++s) { int j = s + rng() % (ne - s); std::swap(ids[s], ids[j]); per[ids[s]].push_back(t); }
  }
  p.bounds.assign(ne + 1, 0);
  std::vector<int> rows_in, rows_out;
  int pos = 0;
  std::vector<int> tiles;
  for (int e = 0; e < ne; ++e) {
    int c = (int)per[e].size();
    int padded = (c + 15) / 16 * 16;
    p.bounds[e] = pos;
    for (int i = 0; i < padded; ++i) {
      rows_in.push_back(i < c ? per[e][i] : 0);
      rows_out.push_back(i < c ? pos + i : -1);
    }
    pos += padded;
    int nt = (padded + bn - 1) / bn;
    for (int t = 0; t < nt; ++t) tiles.push_back(e | (t << 16));
  }
  p.bounds[ne] = pos;
  p.n_tiles = (int)tiles.size();
  size_t wbytes = (size_t)ne * m * p.row_bytes;
  std::vector<uint8_t> w(wbytes);
  for (auto& b : w) b = (uint8_t)rng();
  // keep UE4M3 scale bytes in a sane range: scales live after the k/2 code bytes of every row
  for (size_t r = 0; r < (size_t)ne * m; ++r)
    for (int j = 0; j < k / 16; ++j) w[r * p.row_bytes + k / 2 + j] = 0x30 + (rng() % 16);
  std::vector<__half> x((size_t)tokens * k);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (auto& v : x) v = __float2half(u(rng));
  CK(hipMalloc(&p.d_w, wbytes)); CK(hipMemcpy(p.d_w, w.data(), wbytes, hipMemcpyHostToDevice));
  CK(hipMalloc(&p.d_x, x.size() * 2)); CK(hipMemcpy(p.d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
  CK(hipMalloc(&p.d_tiles, tiles.size() * 4)); CK(hipMemcpy(p.d_tiles, tiles.data(), tiles.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&p.d_bounds, p.bounds.size() * 4)); CK(hipMemcpy(p.d_bounds, p.bounds.data(), p.bounds.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&p.d_rows_in, rows_in.size() * 4)); CK(hipMemcpy(p.d_rows_in, rows_in.data(), rows_in.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&p.d_rows_out, rows_out.size() * 4)); CK(hipMemcpy(p.d_rows_out, rows_out.data(), rows_out.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&p.d_out, (size_t)pos * m * 2));
  printf("experts %d m %d k %d tokens %d topk %d slots %d padded %d tiles %d\n", ne, m, k, tokens, topk, tokens * topk, pos, p.n_tiles);
}

template <int BN>
void launch_base(const Problem& p, hipStream_t s) {
#ifdef PAIRV
  const int mm = p.m / 2;
  dim3 grid((mm + 63) / 64, p.n_tiles);
  hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kNVFP4, 128, BN, 2, true>), grid, dim3(kThreads), 0, s,
                     p.d_w, p.d_x, p.d_tiles, p.d_bounds, p.d_rows_in, p.d_rows_out, (const float*)nullptr,
                     (float*)nullptr, p.d_out, (size_t)mm, (size_t)p.k,
                     (const void*)((const char*)p.d_w + (size_t)mm * p.row_bytes),
                     (const std::uint32_t*)nullptr, 1.0f, (const float*)nullptr, (size_t)p.m);
#else
  dim3 grid((p.m + BMV - 1) / BMV, p.n_tiles);
  hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kNVFP4, BMV, BN, 2>), grid, dim3(kThreads), 0, s,
                     p.d_w, p.d_x, p.d_tiles, p.d_bounds, p.d_rows_in, p.d_rows_out, (const float*)nullptr,
                     (float*)nullptr, p.d_out, (size_t)p.m, (size_t)p.k, (const void*)nullptr,
                     (const std::uint32_t*)nullptr, 1.0f, (const float*)nullptr);
#endif
}

int main(int argc, char** argv) {
  int k = argc > 1 ? atoi(argv[1]) : 2304;
  int m = argc > 2 ? atoi(argv[2]) : 2304;
  Problem p; build(p, 128, m, k, 4096, 8, BNV, 1);
  hipStream_t s; CK(hipStreamCreate(&s));
  auto run = [&](int reps) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int i = 0; i < 3; ++i) launch_base<BNV>(p, s);
    CK(hipStreamSynchronize(s));
    hipEventRecord(a, s);
    for (int i = 0; i < reps; ++i) launch_base<BNV>(p, s);
    hipEventRecord(b, s); CK(hipEventSynchronize(b));
    float ms; hipEventElapsedTime(&ms, a, b); return ms / reps;
  };
  float ms = run(20);
  {
    std::vector<__half> out((size_t)p.bounds[p.ne] * m);
    CK(hipMemcpy(out.data(), p.d_out, out.size() * 2, hipMemcpyDeviceToHost));
    const char* mode = argc > 3 ? argv[3] : "";
    std::string fn = std::string(PAIRTAG "ref_") + std::to_string(k) + "_" + std::to_string(m) + ".bin";
    if (std::string(mode) == "save") {
      FILE* f = fopen(fn.c_str(), "wb"); fwrite(out.data(), 2, out.size(), f); fclose(f);
    } else if (std::string(mode) == "cmp") {
      std::vector<__half> ref(out.size());
      FILE* f = fopen(fn.c_str(), "rb"); if (!f || fread(ref.data(), 2, ref.size(), f) != ref.size()) { printf("no ref\n"); return 1; } fclose(f);
      // compare only valid rows: padding rows are not written; compare where ref nonzero or out nonzero
      double maxd = 0; size_t bad = 0;
      for (size_t i = 0; i < out.size(); ++i) { double d = fabs((double)__half2float(out[i]) - (double)__half2float(ref[i])); if (d > maxd) maxd = d; if (d > 1e-3) ++bad; }
      printf("compare vs baseline: max diff %.6f, elements > 1e-3: %zu\n", maxd, bad);
    }
  }
  double flops = 2.0 * p.tokens * 8 * (double)m * k;
  printf("baseline: %.3f ms  %.1f TFLOPS (useful)\n", ms, flops / ms / 1e9);
  return 0;
}
