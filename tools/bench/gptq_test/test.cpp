// Standalone check of the GPU GPTQ solver against the production repack and a dumped expert.
//   test rtn <wgu.bin>                       plain rounding must equal iu4::RepackExperts
//   test gptq <dump_dir> <expert> <train> <damp> <out_prefix>
#include "src/models/inkling/gpu/gptq.hip.cpp"
#include "src/models/inkling/gpu/iu4_experts.hip.cpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); std::exit(1);} } while (0)
using namespace gufo::models::inkling;

constexpr int kRows = 2304, kK = 2304, kRowBytes = 1296;

static std::vector<char> ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { std::printf("cannot open %s\n", p.c_str()); std::exit(2); }
  std::vector<char> v(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(v.data(), static_cast<std::streamsize>(v.size()));
  return v;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  float hs[512];
  iu4::FillSigns(hs);
  float* signs;
  CK(hipMalloc(&signs, sizeof hs));
  CK(hipMemcpy(signs, hs, sizeof hs, hipMemcpyHostToDevice));
  rocblas_handle blas;
  rocblas_create_handle(&blas);
  gptq::Scratch sc;
  if (!gptq::AllocScratch(sc, kRows, kK)) { std::printf("alloc failed\n"); return 1; }
  float* wrot;
  CK(hipMalloc(&wrot, (size_t)kRows * kK * 4));
  std::uint8_t* src;
  const size_t gu_bytes = (size_t)kRows * kRowBytes;
  CK(hipMalloc(&src, gu_bytes));
  const size_t tb = iu4::WeightBytesPerExpert(kRows, kK), sb = iu4::ScaleBytesPerExpert(kRows, kK);
  std::uint8_t *t_ref, *t_new;
  std::uint16_t *s_ref, *s_new;
  CK(hipMalloc(&t_ref, tb)); CK(hipMalloc(&t_new, tb));
  CK(hipMalloc(&s_ref, sb)); CK(hipMalloc(&s_new, sb));

  if (mode == "rtn") {
    auto w = ReadAll(argv[2]);
    CK(hipMemcpy(src, w.data(), gu_bytes, hipMemcpyHostToDevice));
    iu4::RepackExperts(src, kRowBytes, t_ref, s_ref, 1, kRows, kK, kK, signs, 0);
    gptq::RotateWeights(src, kRowBytes, wrot, kRows, kK, kK, signs, 0);
    gptq::QuantizeRtn(sc, wrot, 0);
    gptq::Pack(sc, t_new, s_new, 0);
    CK(hipDeviceSynchronize());
    std::vector<std::uint8_t> a(tb), b(tb);
    std::vector<std::uint16_t> c(sb / 2), d(sb / 2);
    CK(hipMemcpy(a.data(), t_ref, tb, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), t_new, tb, hipMemcpyDeviceToHost));
    CK(hipMemcpy(c.data(), s_ref, sb, hipMemcpyDeviceToHost));
    CK(hipMemcpy(d.data(), s_new, sb, hipMemcpyDeviceToHost));
    size_t bad_t = 0, bad_s = 0;
    for (size_t i = 0; i < tb; ++i) bad_t += a[i] != b[i];
    for (size_t i = 0; i < sb / 2; ++i) bad_s += c[i] != d[i];
    std::printf("rtn vs repack: tile bytes differing %zu of %zu, scales differing %zu of %zu\n", bad_t, tb, bad_s, sb / 2);
    return 0;
  }

  const std::string dir = argv[2];
  const int expert = std::atoi(argv[3]);
  const double train_frac = std::atof(argv[4]);
  const float damp = static_cast<float>(std::atof(argv[5]));
  const std::string outp = argv[6];
  auto w = ReadAll(dir + "/wgu_" + std::to_string(expert) + ".bin");
  CK(hipMemcpy(src, w.data(), gu_bytes, hipMemcpyHostToDevice));
  gptq::RotateWeights(src, kRowBytes, wrot, kRows, kK, kK, signs, 0);
  std::vector<std::uint16_t> xrows;
  for (int c = 0; c < 8; ++c) {
    auto ids = ReadAll(dir + "/ids_" + std::to_string(c) + ".bin");
    auto x = ReadAll(dir + "/x_" + std::to_string(c) + ".bin");
    const int* idp = reinterpret_cast<const int*>(ids.data());
    const size_t n = ids.size() / 32;
    for (size_t t = 0; t < n; ++t) {
      bool hit = false;
      for (int s = 0; s < 8; ++s) hit |= idp[t * 8 + s] == expert;
      if (!hit) continue;
      const std::uint16_t* xr = reinterpret_cast<const std::uint16_t*>(x.data()) + t * kK;
      xrows.insert(xrows.end(), xr, xr + kK);
    }
  }
  const int m = static_cast<int>(xrows.size() / kK);
  const int ntr = static_cast<int>(m * train_frac);
  std::printf("expert %d: %d tokens, train %d\n", expert, m, ntr);
  __half* dx;
  float* dxr;
  CK(hipMalloc(&dx, xrows.size() * 2));
  CK(hipMalloc(&dxr, (size_t)m * kK * 4));
  CK(hipMemcpy(dx, xrows.data(), xrows.size() * 2, hipMemcpyHostToDevice));
  gptq::RotateRows(dx, nullptr, nullptr, m, kK, kK, dxr, signs, 0);
  float* h;
  CK(hipMalloc(&h, (size_t)kK * kK * 4));
  CK(hipMemset(h, 0, (size_t)kK * kK * 4));
  if (!gptq::AccumulateHessian(blas, dxr, ntr, kK, h)) { std::printf("syrk failed\n"); return 1; }
  CK(hipDeviceSynchronize());
  if (!gptq::Quantize(blas, sc, wrot, h, kK, damp, 0)) { std::printf("quantize failed\n"); return 1; }
  CK(hipDeviceSynchronize());
  std::vector<std::int8_t> codes((size_t)kRows * kK);
  std::vector<std::uint16_t> gs((size_t)kRows * (kK / 128));
  CK(hipMemcpy(codes.data(), sc.codes, codes.size(), hipMemcpyDeviceToHost));
  CK(hipMemcpy(gs.data(), sc.gscales, gs.size() * 2, hipMemcpyDeviceToHost));
  std::ofstream(outp + ".codes", std::ios::binary).write(reinterpret_cast<char*>(codes.data()), codes.size());
  std::ofstream(outp + ".scales", std::ios::binary).write(reinterpret_cast<char*>(gs.data()), gs.size() * 2);
  std::printf("wrote %s.codes/.scales\n", outp.c_str());
  return 0;
}
