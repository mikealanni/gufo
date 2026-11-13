#pragma once
// Device-resident FFN for 1024-row prefill chunks: ATB gate/up GEMM pair -> on-core
// swiglu reshape -> ATB down GEMM. Only the FFN input is encoded and the output decoded on the host.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "runtime.hpp"

namespace gufo::npu::runtime {

class FfnAtbChain {
 public:
  static constexpr std::uint32_t kM = 1024, kHid = 2048, kInt = 7680, kNa = 4096, kNb = 3584;

  bool Init(const std::string& dir, std::string* err) {
    dir_ = dir;
    if (std::getenv("GUFO_ATB_KDYN")) {
      dnp_ = &gu_;
      swp_ = &gu_;
      kd_ = true;
      ip_ = "insts_kd_";
      const char* xo = std::getenv("GUFO_ATB_KD_XCLBIN");
      return gu_.Init(dir + "/" + (xo ? xo : "final_kd.xclbin"), err);
    }
    return gu_.Init(dir + "/final_atb_k2048.xclbin", err) &&
           dnp_->Init(dir + "/final_atb_k7680.xclbin", err) &&
           swp_->Init(dir + "/final_swresh_k2048.xclbin", err);
  }

  bool AddLayer(const float* gate, const float* up, const float* down, std::string* err) {
    auto l = std::make_unique<Layer>();
    const std::uint32_t mrows = kM / 16;
    const auto pack_swi = [&](std::uint32_t c0, std::uint32_t n) {
      std::vector<float> ws(std::size_t{2} * n * kHid);
      for (std::uint32_t s = 0; s < 2 * n; ++s) {
        const std::uint32_t q = s >> 4, r = s & 15;
        const float* src = (r < 8 ? gate : up) + std::size_t{c0 + q * 8 + (r & 7)} * kHid;
        std::memcpy(ws.data() + std::size_t{s} * kHid, src, std::size_t{kHid} * 4);
      }
      return ws;
    };
    const std::vector<float> wsa = pack_swi(0, kNa), wsb = pack_swi(kNa, kNb);
    l->wqa = gu_.PackWeightAtb(wsa.data(), 2 * kNa, kHid, 4);
    l->wqb = gu_.PackWeightAtb(wsb.data(), 2 * kNb, kHid, 4);
    l->wd = dnp_->PackWeightAtb(down, kHid, kInt, 4);
    l->qa = gu_.MakeProgram(kHid, 2 * kNa, dir_ + "/" + ip_ + "1024x2048x8192.txt",
                            std::size_t{kM} * 2 * kNa / 8 * 9, err, 0, mrows, 3, 5);
    l->qb = gu_.MakeProgram(kHid, 2 * kNb, dir_ + "/" + ip_ + "1024x2048x7168.txt",
                            std::size_t{kM} * 2 * kNb / 8 * 9, err, 0, mrows, 3, 5);
    l->sw = swp_->MakeProgram(kInt, kHid, dir_ + (kd_ ? "/insts_kd_resh_1024x7680x2048.txt" : "/insts_swresh_1024x7680x2048.txt"),
                            std::size_t{kM} * kInt / 8 * 9, err, 0, mrows, 3, 5);
    l->sw.a = l->qa.c;
    l->fake.bo = l->qb.c;
    l->fake.bytes = std::size_t{kM} * 2 * kNb / 8 * 9;
    swp_->WeightSlot(l->fake, l->sw);
    l->dn = dnp_->MakeProgram(kInt, kHid, dir_ + "/" + ip_ + "1024x7680x2048.txt",
                            std::size_t{kM} * kHid / 8 * 9, err, 0, mrows, 3, 5);
    dnp_->RebindA(l->dn, l->sw.c);
    dnp_->WeightSlot(l->wd, l->dn);
    layers_.push_back(std::move(l));
    return err == nullptr || err->empty();
  }

  bool AddAttn(std::size_t layer, const float* qkv, const float* o, std::string* err) {
    auto a = std::make_unique<Attn>();
    if (attn_.size() <= layer) attn_.resize(layer + 1);
    const std::uint32_t mrows = kM / 16;
    a->wq = gu_.PackWeightAtb(qkv, 3072, kHid, 4);
    a->wo = gu_.PackWeightAtb(o, kHid, kHid, 4);
    a->pq = gu_.MakeProgram(kHid, 3072, dir_ + "/" + ip_ + "1024x2048x3072.txt",
                            std::size_t{kM} * 3072 / 8 * 9, err, 0, mrows, 3, 5);
    a->po = gu_.MakeProgram(kHid, kHid, dir_ + "/" + ip_ + "1024x2048x2048.txt",
                            std::size_t{kM} * kHid / 8 * 9, err, 0, mrows, 3, 5);
    attn_[layer] = std::move(a);
    return err == nullptr || err->empty();
  }

  /// Run every program once so the throwaway first dispatch ("first C write is lost") happens at load time.
  void Warm() {
    for (auto& lp : layers_) {
      Layer& l = *lp;
      Dispatch(gu_, l.wqa, l.qa);
      Dispatch(gu_, l.wqb, l.qb);
      Dispatch(*swp_, l.fake, l.sw);
      Dispatch(*dnp_, l.wd, l.dn);
    }
    for (auto& ap : attn_) {
      if (!ap) continue;
      Dispatch(gu_, ap->wq, ap->pq);
      Dispatch(gu_, ap->wo, ap->po);
    }
  }

  bool HasAttn(std::size_t layer) const { return layer < attn_.size() && attn_[layer] != nullptr; }

  /// Both run on the K=2048 context shared with the gate/up GEMM. n <= 1024 rows.
  void RunQkv(std::size_t layer, const float* z, std::uint32_t n, float* y) {
    Gemm1024(*attn_[layer], true, z, n, kHid, 3072, y);
  }
  void RunO(std::size_t layer, const float* a, std::uint32_t n, float* y) {
    Gemm1024(*attn_[layer], false, a, n, kHid, kHid, y);
  }

  /// z: n x 2048 floats (n <= 1024). y: n x 2048 floats.
  void Run(std::size_t layer, const float* z, std::uint32_t n, float* y) {
    Layer& l = *layers_[layer];
    const float* src = z;
    if (n < kM) {
      pad_.assign(std::size_t{kM} * kHid, 0.f);
      std::memcpy(pad_.data(), z, std::size_t{n} * kHid * 4);
      src = pad_.data();
    }
    const auto t0 = std::chrono::steady_clock::now();
    gu_.EncodeAAtb(src, kM, kHid, l.qa);
    std::memcpy(l.qb.ah, l.qa.ah, std::size_t{kM} * kHid / 8 * 9);
    l.qa.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    l.qb.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto t1 = std::chrono::steady_clock::now();
    if (dbg_) std::fprintf(stderr, "[atb] %s\n", "wqa");
    Dispatch(gu_, l.wqa, l.qa);
    const auto d1 = std::chrono::steady_clock::now();
    if (dbg_) std::fprintf(stderr, "[atb] %s\n", "wqb");
    Dispatch(gu_, l.wqb, l.qb);
    const auto d2 = std::chrono::steady_clock::now();
    if (dbg_) std::fprintf(stderr, "[atb] %s\n", "fake");
    Dispatch(*swp_, l.fake, l.sw);
    const auto d3 = std::chrono::steady_clock::now();
    if (dbg_) std::fprintf(stderr, "[atb] %s\n", "wd");
    Dispatch(*dnp_, l.wd, l.dn);
    const auto d4 = std::chrono::steady_clock::now();
    ph_ms[0] += std::chrono::duration<double, std::milli>(d1 - t1).count();
    ph_ms[1] += std::chrono::duration<double, std::milli>(d2 - d1).count();
    ph_ms[2] += std::chrono::duration<double, std::milli>(d3 - d2).count();
    ph_ms[3] += std::chrono::duration<double, std::milli>(d4 - d3).count();
    l.dn.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto t2 = std::chrono::steady_clock::now();
    if (n == kM) {
      dnp_->DecodeCAtb(y, kM, kHid, l.dn);
    } else {
      out_.resize(std::size_t{kM} * kHid);
      dnp_->DecodeCAtb(out_.data(), kM, kHid, l.dn);
      std::memcpy(y, out_.data(), std::size_t{n} * kHid * 4);
    }
    const auto t3 = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    enc_ms += ms(t0, t1);
    dev_ms += ms(t1, t2);
    dec_ms += ms(t2, t3);
    if (const char* dp = std::getenv("GUFO_ATB_DUMP"); dp && layer < 3 && calls < 3) {
      char fn[600];
      std::snprintf(fn, sizeof fn, "%s_%zu_%zu.f32", dp, calls, layer);
      if (FILE* f = std::fopen(fn, "wb")) { std::fwrite(y, 4, std::size_t{n} * kHid, f); std::fclose(f); }
    }
    ++calls;
  }

  double enc_ms{0}, dev_ms{0}, dec_ms{0}, ph_ms[4]{0, 0, 0, 0};
  std::size_t calls{0};

  std::size_t Layers() const { return layers_.size(); }

 private:
  struct Layer {
    Program qa, qb, sw, dn;
    WeightBuffer wqa, wqb, wd, fake;
  };

  struct Attn {
    Program pq, po;
    WeightBuffer wq, wo;
  };

  void Gemm1024(Attn& a, bool qkv, const float* z, std::uint32_t n, std::uint32_t k, std::uint32_t nout,
                float* y) {
    Program& p = qkv ? a.pq : a.po;
    WeightBuffer& w = qkv ? a.wq : a.wo;
    const float* src = z;
    if (n < kM) {
      pad_.assign(std::size_t{kM} * k, 0.f);
      std::memcpy(pad_.data(), z, std::size_t{n} * k * 4);
      src = pad_.data();
    }
    gu_.EncodeAAtb(src, kM, k, p);
    p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    Dispatch(gu_, w, p);
    p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (n == kM) {
      gu_.DecodeCAtb(y, kM, nout, p);
    } else {
      out_.resize(std::size_t{kM} * nout);
      gu_.DecodeCAtb(out_.data(), kM, nout, p);
      std::memcpy(y, out_.data(), std::size_t{n} * nout * 4);
    }
  }

  static void Dispatch(NpuRuntime& r, WeightBuffer& w, Program& p) {
    WeightBuffer& ww = r.WeightSlot(w, p);
    ww.run.start();
    ww.run.wait();
    if (!p.warmed) {  // first C write is lost on this stack
      p.warmed = true;
      ww.run.start();
      ww.run.wait();
    }
  }

  std::string dir_;
  NpuRuntime gu_, dn_, sw_;
  NpuRuntime* dnp_{&dn_};
  NpuRuntime* swp_{&sw_};
  bool kd_{false};
  bool dbg_{std::getenv("GUFO_ATB_DBG") != nullptr};
  std::string ip_{"insts_atb_"};
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<std::unique_ptr<Attn>> attn_;
  std::vector<float> pad_, out_;
};

}  // namespace gufo::npu::runtime
