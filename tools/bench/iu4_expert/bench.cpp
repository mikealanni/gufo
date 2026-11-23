// Standalone harness for the int4 routed gate/up kernel (fixed random routing).
#ifndef KERNEL_FILE
#define KERNEL_FILE "../../../src/models/inkling/gpu/iu4_experts.hip.cpp"
#endif
#include KERNEL_FILE
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace gufo::models::inkling::iu4;
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

int main(int argc, char** argv) {
  const int ne = 128, ff = 1152, K = 2304, tokens = 4096, topk = 8;
  const int rows2 = 2 * ff;
  const size_t row_bytes = ((K / 2 + K / 16 + 15) / 16) * 16;
  std::mt19937 rng(1);
  std::vector<std::vector<int>> per(ne);
  std::vector<int> ids(ne);
  for (int t = 0; t < tokens; ++t) {
    for (int i = 0; i < ne; ++i) ids[i] = i;
    for (int s = 0; s < topk; ++s) { int j = s + rng() % (ne - s); std::swap(ids[s], ids[j]); per[ids[s]].push_back(t * topk + s); }
  }
  std::vector<int> bounds(ne + 1, 0), rows_in, rows_out;
  int pos = 0;
  for (int e = 0; e < ne; ++e) {
    int c = (int)per[e].size(), padded = (c + 15) / 16 * 16;
    bounds[e] = pos;
    for (int i = 0; i < padded; ++i) { rows_in.push_back(i < c ? per[e][i] / topk : 0); rows_out.push_back(i < c ? per[e][i] : -1); }
    pos += padded;
  }
  bounds[ne] = pos;
  const int max_tiles = (tokens * topk + 127) / 128 + ne;
  std::vector<int> tl(max_tiles, 0x7FFF0000);
  { int w = 0; for (int e = 0; e < ne; ++e) { int nt = (bounds[e + 1] - bounds[e] + 127) / 128; for (int j = 0; j < nt; ++j) tl[w++] = e | (j << 16); } }
  size_t wbytes = (size_t)ne * rows2 * row_bytes;
  std::vector<uint8_t> w(wbytes);
  for (auto& b : w) b = (uint8_t)rng();
  for (size_t r = 0; r < (size_t)ne * rows2; ++r) for (int j = 0; j < K / 16; ++j) w[r * row_bytes + K / 2 + j] = 0x30 + (rng() % 16);
  std::vector<__half> x((size_t)tokens * K);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (auto& v : x) v = __float2half(u(rng));
  uint8_t *d_nv, *d_tiles, *d_act; uint16_t* d_sc; float *d_as, *d_signs, *d_es, *d_situ; __half *d_x, *d_out;
  int *d_bounds, *d_in, *d_outr, *d_tl;
  const int padded_rows = (pos + 128 + 127) / 128 * 128;
  CK(hipMalloc(&d_nv, wbytes)); CK(hipMemcpy(d_nv, w.data(), wbytes, hipMemcpyHostToDevice));
  CK(hipMalloc(&d_tiles, (size_t)ne * WeightBytesPerExpert(rows2, K)));
  CK(hipMalloc(&d_sc, (size_t)ne * ScaleBytesPerExpert(rows2, K)));
  CK(hipMalloc(&d_act, (size_t)padded_rows / 16 * (K / 128) * 1024)); CK(hipMemset(d_act, 0, (size_t)padded_rows / 16 * (K / 128) * 1024));
  CK(hipMalloc(&d_as, (size_t)(K / 128) * padded_rows * 4)); CK(hipMemset(d_as, 0, (size_t)(K / 128) * padded_rows * 4));
  CK(hipMalloc(&d_x, x.size() * 2)); CK(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
  CK(hipMalloc(&d_out, (size_t)tokens * topk * ff * 2));
  CK(hipMalloc(&d_bounds, bounds.size() * 4)); CK(hipMemcpy(d_bounds, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&d_in, rows_in.size() * 4)); CK(hipMemcpy(d_in, rows_in.data(), rows_in.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&d_outr, rows_out.size() * 4)); CK(hipMemcpy(d_outr, rows_out.data(), rows_out.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&d_tl, tl.size() * 4)); CK(hipMemcpy(d_tl, tl.data(), tl.size() * 4, hipMemcpyHostToDevice));
  float signs[512]; FillSigns(signs);
  CK(hipMalloc(&d_signs, sizeof(signs))); CK(hipMemcpy(d_signs, signs, sizeof(signs), hipMemcpyHostToDevice));
  std::vector<float> es(ne, 0.01f); CK(hipMalloc(&d_es, ne * 4)); CK(hipMemcpy(d_es, es.data(), ne * 4, hipMemcpyHostToDevice));
  float situ[2] = {7.0f, 7.0f}; CK(hipMalloc(&d_situ, 8)); CK(hipMemcpy(d_situ, situ, 8, hipMemcpyHostToDevice));
  hipStream_t s; CK(hipStreamCreate(&s));
  RepackExperts(d_nv, row_bytes, d_tiles, d_sc, ne, rows2, K, d_signs, s);
  QuantizeRoutedActs(d_x, d_in, d_outr, d_bounds, ne, pos, K, d_act, d_as, padded_rows, d_signs, s);
  CK(hipStreamSynchronize(s));
  auto run = [&]() { GateUpSitu(d_tiles, d_sc, d_act, d_as, padded_rows, d_tl, max_tiles, d_bounds, d_outr, d_out, ff, K, d_es, d_situ, s); };
  for (int i = 0; i < 3; ++i) run();
  CK(hipStreamSynchronize(s));
  const char* mode = argc > 1 ? argv[1] : "";
  std::vector<__half> out((size_t)tokens * topk * ff);
  CK(hipMemcpy(out.data(), d_out, out.size() * 2, hipMemcpyDeviceToHost));
  std::string fn = "ref_gateup.bin";
  if (std::string(mode) == "save") { FILE* f = fopen(fn.c_str(), "wb"); fwrite(out.data(), 2, out.size(), f); fclose(f); }
  else if (std::string(mode) == "cmp") {
    std::vector<__half> ref(out.size()); FILE* f = fopen(fn.c_str(), "rb"); if (!f || fread(ref.data(), 2, ref.size(), f) != ref.size()) { printf("no ref\n"); return 1; } fclose(f);
    double maxd = 0; size_t bad = 0; for (size_t i = 0; i < out.size(); ++i) { double d = fabs((double)__half2float(out[i]) - (double)__half2float(ref[i])); if (d > maxd) maxd = d; if (d > 1e-4) ++bad; }
    printf("compare vs baseline: max diff %.6f, elements > 1e-4: %zu\n", maxd, bad);
  }
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  float best = 1e9;
  for (int rep = 0; rep < 5; ++rep) {
    CK(hipEventRecord(a, s));
    for (int i = 0; i < 20; ++i) run();
    CK(hipEventRecord(b, s)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); best = std::min(best, ms / 20);
  }
  double flops = 2.0 * (double)tokens * topk * rows2 * K;
  printf("gate/up int4: %.3f ms (min of 5x20)  %.1f TOPS (useful)  padded rows %d\n", best, flops / best / 1e9, pos);
  return 0;
}
