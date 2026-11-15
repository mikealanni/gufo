// WP10 probe 2: Ffn64 reuse where BOTH weight buffers are REUSED (same BOs),
// data repacked in place. If call#2 is correct here, the stale-state bug is in
// per-call BO binding, and the engine fix is shared weight staging buffers.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/npu/runtime/runtime.hpp"

using namespace gufo::npu::runtime;

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tools/npu/decode_int5/runtime";
  std::string err;
  NpuRuntime rt;
  if (!rt.Init(dir + "/final_16x64x128_ffn.xclbin", &err)) {
    std::fprintf(stderr, "init: %s\n", err.c_str());
    return 1;
  }
  auto p_ffn = rt.MakeProgram(2048, 2048, dir + "/insts_ffn.txt",
                              std::size_t{136} * kStripBytes, &err);
  auto p_gu64 = rt.MakeProgram(2048, 15360, dir + "/insts_ffngu64.txt", 0, &err, 0, 4);
  auto p_dn64 = rt.MakeProgram(7680, 2048, dir + "/insts_plain_7680x2048_m64.txt", 0, &err, 0, 4);
  if (!err.empty()) { std::fprintf(stderr, "progs: %s\n", err.c_str()); return 1; }

  std::mt19937 gen(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> G(std::size_t(15360) * 2048), D(std::size_t(2048) * 7680);
  for (auto& v : G) v = nd(gen) * 0.02f;
  for (auto& v : D) v = nd(gen) * 0.02f;

  const std::size_t gu_bytes = std::size_t{120} * 32 * 4 * kQuarterBytes;
  const std::size_t dn_bytes = std::size_t{16} * 120 * 4 * kQuarterBytes;
  WeightBuffer m;  // fused m16 layout: gu then down
  m.bytes = gu_bytes + dn_bytes;
  m.bo = xrt::bo(rt.device(), m.bytes, XRT_BO_FLAGS_HOST_ONLY, rt.kernel().group_id(3));
  m.host = m.bo.map<std::uint8_t*>();
  std::memset(m.host, 0, m.bytes);
  PackMatInt5(G.data(), 15360, 2048, m.host);
  PackMatInt5(D.data(), 2048, 7680, m.host + gu_bytes);
  m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // dn64 weight in a SHARED buffer reused across calls (single WeightBuffer).
  auto wb_dn = rt.PackWeight(D.data(), 2048, 7680);

  const std::uint32_t n = 64;
  std::vector<float> X(std::size_t(n) * 2048);
  for (auto& v : X) v = nd(gen);
  std::vector<float> y16(std::size_t(n) * 2048), y64(std::size_t(n) * 2048);
  auto run_pair = [&](const char* tag) {
    rt.Ffn(p_ffn, m, X.data(), 16, y16.data());
    rt.Ffn64(m, p_gu64, wb_dn, p_dn64, X.data(), n, y64.data());
    double s0 = 0, sden = 0;
    for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) {
      const double a = y16[i], d = y16[i] - y64[i];
      s0 += d * d;
      sden += a * a;
    }
    std::printf("%s rel %.6f\n", tag, std::sqrt(s0 / (sden + 1e-30)));
  };
  run_pair("call1 (BOs first use)");
  if (getenv("GUFO_INTER")) {
    auto p16i = rt.MakeProgram(2048, 2048, getenv("GUFO_INTER")[1] == 49 ? dir + "/insts_plain_2048x2048.txt" : dir + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
    std::vector<float> wi(std::size_t(2048) * 2048), yi(std::size_t(16) * 2048);
    for (auto& v : wi) v = nd(gen) * 0.02f;
    auto wbi = rt.PackWeight(wi.data(), 2048, 2048);
    rt.Gemm(p16i, wbi, X.data(), 16, yi.data());
    rt.Gemm(p16i, wbi, X.data(), 16, yi.data());
  }
  const int nf = getenv("GUFO_FILL") ? atoi(getenv("GUFO_FILL")) : 0;
  std::vector<xrt::bo> fillers;
  for (int i = 0; i < nf; ++i)
    fillers.emplace_back(rt.device(), 65536, XRT_BO_FLAGS_HOST_ONLY, rt.kernel().group_id(3));
  run_pair("call2 same BO same data");
  for (auto& v : D) v = nd(gen) * 0.02f;
  PackMatInt5(D.data(), 2048, 7680, m.host + gu_bytes);
  m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::memcpy(wb_dn.host, wb_dn.host, 1);  // keep alive
  PackMatInt5(D.data(), 2048, 7680, wb_dn.host);  // repack SAME dn BO
  wb_dn.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run_pair("call3 same BOs, dn data changed");
  {  // call6: SECOND gu/dn program pair, SHARED weight BOs, different (fake layer) data.
    auto pg2 = rt.MakeProgram(2048, 15360, dir + "/insts_ffngu64.txt", 0, &err, 0, 4);
    auto pd2 = rt.MakeProgram(7680, 2048, dir + "/insts_plain_7680x2048_m64.txt", 0, &err, 0, 4);
    // "layer2" weights: repack shared m.host/dn host, then Ffn64 on pair2.
    for (auto& v : G) v = nd(gen) * 0.02f;
    for (auto& v : D) v = nd(gen) * 0.02f;
    PackMatInt5(G.data(), 15360, 2048, m.host);
    PackMatInt5(D.data(), 2048, 7680, m.host + gu_bytes);
    m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    PackMatInt5(D.data(), 2048, 7680, wb_dn.host);
    wb_dn.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    rt.Ffn(p_ffn, m, X.data(), 16, y16.data());
    rt.Ffn64(m, pg2, wb_dn, pd2, X.data(), n, y64.data());
    double s0 = 0, sden = 0;
    for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) {
      const double a = y16[i], d = y16[i] - y64[i];
      s0 += d * d;
      sden += a * a;
    }
    std::printf("call6 pair2 shared-BO (0 => shared-weight design works): rel %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
    return 0;
  }
  if (getenv("GUFO_STOP")) return 0;
  {  // call5: Ffn64 with a FRESH dn weight BO (minimal process, cleanest repro)
    for (auto& v : D) v = nd(gen) * 0.02f;
    PackMatInt5(D.data(), 2048, 7680, m.host + gu_bytes);
    m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto wb_dn2 = rt.PackWeight(D.data(), 2048, 7680);
    std::vector<std::uint8_t> gub(std::size_t(120) * 16384);
    {  // pre-call5 snapshot: capture gu C from call3's Ffn64 by re-running? Use post-diff vs p_dn64.ah
      // simpler: compare p_dn64.ah (staged A == swiglu of guC) before/after call5's shuffle.
    }
    std::memcpy(gub.data(), p_dn64.ah, 552960);  // staged A from call3 (valid guC)
    std::vector<float> ysave = y64;  // call3 output (dn2 weights)
    rt.Ffn(p_ffn, m, X.data(), 16, y16.data());
    rt.Ffn64(m, p_gu64, wb_dn2, p_dn64, X.data(), n, y64.data());
    {
      int dA = 0;
      for (int i = 0; i < 552960; ++i) dA += gub[i] != p_dn64.ah[i];
      std::printf("call5: staged A diff vs call3 (expect 0, gu unchanged): %d\n", dA);
      double s0 = 0, sden = 0;
      for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) {
        const double a = ysave[i], d = ysave[i] - y64[i];
        s0 += d * d;
        sden += a * a;
      }
      std::printf("call5 y64 vs call3 y64 (0 => STALE dn weights): rel %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
    }
    double s0 = 0, sden = 0;
    for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) {
      const double a = y16[i], d = y16[i] - y64[i];
      s0 += d * d;
      sden += a * a;
    }
    std::printf("call5 FRESH dn BO Ffn64: rel %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
    rt.Ffn64(m, p_gu64, wb_dn2, p_dn64, X.data(), n, y64.data());
    s0 = 0;
    for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) {
      const double a = y16[i], d = y16[i] - y64[i];
      s0 += d * d;
    }
    std::printf("call5b same again: rel %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
  }
  if (getenv("GUFO_STOP")) return 0;
  if (nf) {  // rebind test with LARGE weight BOs: does big-B rebinding break?
    {
      std::vector<float> Wg(std::size_t(15360) * 2048);
      for (auto& v : Wg) v = nd(gen) * 0.02f;
      auto wb_g = rt.PackWeight(Wg.data(), 15360, 2048);
      std::vector<float> yg(std::size_t(n) * 2048, 0.f);  // p64r gemm K=2048 N=2048: use plain
      auto p64z = rt.MakeProgram(2048, 2048, dir + "/insts_plain_2048x2048.txt", 0, &err);
      rt.Gemm(p64z, wb_g, X.data(), n, yg.data());  // garbage dims OK; just exercise bind
    }
    {  // 11.8MB dn-size rebind between dispatches on p_dn64-like plain program
      std::vector<float> W1(std::size_t(7680) * 2048), W2(std::size_t(7680) * 2048);
      for (auto& v : W1) v = nd(gen) * 0.02f;
      for (auto& v : W2) v = nd(gen) * 0.02f;
      auto w1 = rt.PackWeight(W1.data(), 2048, 7680);
      auto w2 = rt.PackWeight(W2.data(), 2048, 7680);
      std::vector<float> xd(std::size_t(n) * 7680);
      for (auto& v : xd) v = nd(gen);
      std::vector<float> ya(std::size_t(n) * 2048, 0.f), yb(std::size_t(n) * 2048, 0.f), yc(std::size_t(n) * 2048, 0.f);
      rt.Gemm(p_dn64, w1, xd.data(), n, ya.data());  // warm bind w1
      rt.Gemm(p_dn64, w2, xd.data(), n, yb.data());  // rebind to w2
      rt.Gemm(p_dn64, w1, xd.data(), n, yc.data());  // back to w1
      double s0 = 0, sden = 0;
      for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) {
        const double a = ya[i], d = ya[i] - yc[i];
        s0 += d * d;
        sden += a * a;
      }
      std::printf("BIGREBIND w1->w2->w1 (ya vs yc, expect 0): rel %.6f\n", std::sqrt(s0 / (sden + 1e-30)));
      s0 = 0; sden = 0;
      for (std::size_t i = 0; i < std::size_t(16) * 2048; ++i) {
        const double a = yb[i];
        s0 += 0;
        sden += a * a;
      }
      std::printf("BIGREBIND yb energy %.6f\n", std::sqrt(sden / (16.0 * 2048)));
    }
    return 0;
  }
  for (auto& v : G) v = nd(gen) * 0.02f;
  PackMatInt5(G.data(), 15360, 2048, m.host);
  m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run_pair("call4 same BOs, gu data changed");
  return 0;
}
