#pragma once
#include <mutex>
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
    if (kd_ && !std::getenv("GUFO_NO_SWIFUSE") && swifuse_ok_) {
      l->fused = true;
      const std::size_t abytes = std::size_t{kM} * kInt / 8 * 9;
      l->swc = gu_.MakeBo(2 * abytes, 3);
      std::memset(l->swc.map<void*>(), 0, 2 * abytes);
      l->swc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      l->qa = gu_.MakeProgram(kHid, 2 * kNa, dir_ + "/" + ip_ + "swi_a_1024x2048x8192.txt", 64, err, 0, mrows, 3, 5);
      l->qb = gu_.MakeProgram(kHid, 2 * kNb, dir_ + "/" + ip_ + "swi_b_1024x2048x7168.txt", 64, err, 0, mrows, 3, 5);
      l->qa.c = l->swc;
      l->qa.ch = l->swc.map<std::uint8_t*>();
      l->qb.c = l->swc;
      l->qb.ch = l->qa.ch;
      l->dn = dnp_->MakeProgram(kInt, kHid, dir_ + "/" + ip_ + "dn2_1024x7680x2048.txt", std::size_t{kM} * kHid / 8 * 9, err, 0,
                                mrows, 3, 5);
      dnp_->RebindA(l->dn, l->swc);
      dnp_->WeightSlot(l->wd, l->dn);
      layers_.push_back(std::move(l));
      return err == nullptr || err->empty();
    }
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
      if (!l.fused) Dispatch(*swp_, l.fake, l.sw);
      Dispatch(*dnp_, l.wd, l.dn);
    }
    for (auto& cp : conv_) {
      if (!cp) continue;
      Dispatch(gu_, cp->wi, cp->pi);
      Dispatch(gu_, cp->wo, cp->po);
      if (cp->dev) Dispatch(gu_, cp->wx, cp->pc);
    }
    for (auto& ap : attn_) {
      if (!ap) continue;
      Dispatch(gu_, ap->wq, ap->pq);
      Dispatch(gu_, ap->wo, ap->po);
    }
  }

  bool AddConv(std::size_t layer, const float* w_in, const float* w_out, std::string* err) {
    auto c = std::make_unique<Conv>();
    if (conv_.size() <= layer) conv_.resize(layer + 1);
    const std::uint32_t mrows = kM / 16;
    c->wi = gu_.PackWeightAtb(w_in, 3 * kHid, kHid, 4);
    c->wo = gu_.PackWeightAtb(w_out, kHid, kHid, 4);
    c->pi = gu_.MakeProgram(kHid, 3 * kHid, dir_ + "/" + ip_ + "1024x2048x6144.txt",
                            std::size_t{448} * 18432, err, 0, mrows, 3, 5);
    c->po = gu_.MakeProgram(kHid, kHid, dir_ + "/" + ip_ + "1024x2048x2048.txt",
                            std::size_t{kM} * kHid / 8 * 9, err, 0, mrows, 3, 5);
    try {
      c->pc = gu_.MakeProgram(kHid, kHid, dir_ + "/" + ip_ + "conv_1024.txt", 1, err, 0, mrows, 3, 5);
      c->pc.a = c->pi.c;
      c->pc.ah = c->pi.ch;
      c->pc.c = c->po.a;
      c->pc.ch = c->po.ah;
      c->wx.bytes = 64 * 9216;
      c->wx.bo = gu_.MakeBo(c->wx.bytes, 3);
      c->wx.host = c->wx.bo.map<std::uint8_t*>();
      std::memset(c->wx.host, 0, c->wx.bytes);
      c->wx.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      gu_.WeightSlot(c->wx, c->pc);
      c->dev = kd_;
    } catch (const std::exception&) {
      c->dev = false;
    }
    conv_[layer] = std::move(c);
    return err == nullptr || err->empty();
  }

  bool HasConvDev(std::size_t layer) const { return HasConv(layer) && conv_[layer]->dev; }

  /// in_proj without the host decode: result stays on the C wire (program.ch).
  void RunConvInRaw(std::size_t layer, const float* z, std::uint32_t n) {
    Conv& c = *conv_[layer];
    const float* src = z;
    if (n < kM) {
      pad_.assign(std::size_t{kM} * kHid, 0.f);
      std::memcpy(pad_.data(), z, std::size_t{n} * kHid * 4);
      src = pad_.data();
    }
    gu_.EncodeAAtb(src, kM, kHid, c.pi);
    c.pi.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    Dispatch(gu_, c.wi, c.pi);
    c.pi.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  // ---- split API: encode/decode run outside the caller's device lock, Exec* inside it ----
  struct Job {
    std::vector<std::uint8_t> a, c;
    bool staged{false};
  };
  void EncodeJob(const float* z, std::uint32_t n, std::uint32_t k, Job& j) {
    std::vector<float> pad;
    const float* src = z;
    if (n < kM) {
      pad.assign(std::size_t{kM} * k, 0.f);
      std::memcpy(pad.data(), z, std::size_t{n} * k * 4);
      src = pad.data();
    }
    j.a.resize(std::size_t{kM} * k / 8 * 9);
    gu_.EncodeAAtbRaw(src, kM, k, j.a.data());
  }
  void DecodeJob(const Job& j, std::uint32_t n, std::uint32_t nout, float* y) {
    if (n == kM) {
      gu_.DecodeCAtbRaw(j.c.data(), y, kM, nout);
    } else {
      std::vector<float> tmp(std::size_t{kM} * nout);
      gu_.DecodeCAtbRaw(j.c.data(), tmp.data(), kM, nout);
      std::memcpy(y, tmp.data(), std::size_t{n} * nout * 4);
    }
  }
  /// A program's A/C staging BOs are single-owner: Stage takes the owner lock and copies A in, Finish copies C out and
  /// releases it, so both copies run outside the device lock while the device serves other layers.
  struct Port {
    std::mutex m;
  };
  void Stage(Port& pt, Program& p, const std::vector<Program*>& ins, Job& j) {
    pt.m.lock();
    for (Program* q : ins) {
      std::memcpy(q->ah, j.a.data(), j.a.size());
      q->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    (void)p;
    j.staged = true;
  }
  void Finish(Port& pt, Program& p, std::size_t cbytes, Job& j) {
    j.c.assign(p.ch, p.ch + cbytes);
    j.staged = false;
    pt.m.unlock();
  }
  void StageQkv(std::size_t l, Job& j) { Stage(*attn_[l]->port_q, attn_[l]->pq, {&attn_[l]->pq}, j); }
  void FinishQkv(std::size_t l, Job& j) { Finish(*attn_[l]->port_q, attn_[l]->pq, std::size_t{kM} * 3072 / 8 * 9, j); }
  void StageO(std::size_t l, Job& j) { Stage(*attn_[l]->port_o, attn_[l]->po, {&attn_[l]->po}, j); }
  void FinishO(std::size_t l, Job& j) { Finish(*attn_[l]->port_o, attn_[l]->po, std::size_t{kM} * kHid / 8 * 9, j); }
  void StageFfn(std::size_t l, Job& j) { Stage(*layers_[l]->port, layers_[l]->dn, {&layers_[l]->qa, &layers_[l]->qb}, j); }
  void FinishFfn(std::size_t l, Job& j) { Finish(*layers_[l]->port, layers_[l]->dn, std::size_t{kM} * kHid / 8 * 9, j); }


  // ---- device-resident residual stream: x, branch and the normalized A wire never leave the device ----
  static constexpr std::size_t kRegionBytes = std::size_t{kM} * kHid / 8 * 9;
  bool DevRes() const { return devres_; }
  bool SetupDevRes(const std::vector<const float*>& g_a, const std::vector<const float*>& g_f, const float* g_final,
                   const std::vector<const float*>& p_a, const std::vector<const float*>& p_f, const float* p_final, float eps,
                   std::string* err) {
    try {
      const std::size_t nl = layers_.size();
      if (g_a.size() != nl || g_f.size() != nl) throw std::runtime_error("norm weight count");
      resc_ = gu_.MakeBo(3 * kRegionBytes, 3);
      resa_ = gu_.MakeBo(kRegionBytes, 3);
      resc_h_ = resc_.map<std::uint8_t*>();
      resa_h_ = resa_.map<std::uint8_t*>();
      std::memset(resc_h_, 0, 3 * kRegionBytes);
      std::memset(resa_h_, 0, kRegionBytes);
      resc_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      resa_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      br_ = xrt::bo(resc_, kRegionBytes, kRegionBytes);
      const auto bind_c = [&](Program& p) {
        p.c = br_;
        p.ch = resc_h_ + kRegionBytes;
      };
      for (auto& lp : layers_) {
        gu_.RebindA(lp->qa, resa_);
        gu_.RebindA(lp->qb, resa_);
        bind_c(lp->dn);
      }
      for (auto& ap : attn_) {
        if (!ap) continue;
        gu_.RebindA(ap->pq, resa_);
        bind_c(ap->po);
      }
      for (auto& cp : conv_) {
        if (!cp) continue;
        gu_.RebindA(cp->pi, resa_);
        bind_c(cp->po);
      }
      const auto bf = [](float f) {
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        return static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
      };
      const auto mk = [&](const float* g, const float* gp, int variant) {
        auto np = std::make_unique<NormP>();
        np->p = gu_.MakeBareProgram(dir_ + "/insts_kd_norm" + std::to_string(variant) + "_1024.txt");
        np->p.a = resc_;
        np->p.c = resa_;
        np->w.bytes = std::size_t{16} * 9216;
        np->w.bo = gu_.MakeBo(np->w.bytes, 3);
        np->w.host = np->w.bo.map<std::uint8_t*>();
        std::memset(np->w.host, 0, np->w.bytes);
        auto* h = reinterpret_cast<std::uint16_t*>(np->w.host);
        for (std::size_t i = 0; i < kHid; ++i) h[i] = bf(g[i]);
        h[kHid] = bf(eps);
        for (std::size_t i = 0; i < kHid; ++i) h[2112 + i] = bf(gp[i]);
        for (std::size_t q = 1; q < 16; ++q) std::memcpy(np->w.host + q * 9216, np->w.host, 9216);
        np->w.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return np;
      };
      norms_.clear();
      for (std::size_t l = 0; l < nl; ++l) {
        norms_.push_back(mk(g_a[l], p_a[l], 0));
        norms_.push_back(mk(g_f[l], p_f[l], 1));
      }
      norms_.push_back(mk(g_final, p_final, 0));
      devres_ = true;
    } catch (const std::exception& e) {
      if (err) *err = e.what();
      return false;
    }
    return true;
  }
  /// Chunk start: x (n rows) in C wire into region X0, branch region zeroed.
  void DevResInit(const float* x, std::uint32_t n) {
    std::vector<float> pad;
    const float* src = x;
    if (n < kM) {
      pad.assign(std::size_t{kM} * kHid, 0.f);
      std::memcpy(pad.data(), x, std::size_t{n} * kHid * 4);
      src = pad.data();
    }
    gu_.EncodeCAtbRaw(src, kM, kHid, resc_h_ + 2 * kRegionBytes);
    std::memset(resc_h_ + kRegionBytes, 0, kRegionBytes);
    resc_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  /// site 0: before the attention/conv block, 1: before the FFN, l == layers: final norm. Caller holds the device lock.
  void ExecNorm(std::size_t l, int site) {
    NormP& np = *norms_[l == layers_.size() ? 2 * l : 2 * l + site];
    Dispatch(gu_, np.w, np.p);
  }
  void DevResFinal(std::uint32_t n, float* z) {
    resa_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    Program view;
    view.ah = resa_h_;
    std::vector<float> tmp(std::size_t{kM} * kHid);
    gu_.DecodeAAtb(tmp.data(), kM, kHid, view);
    std::memcpy(z, tmp.data(), std::size_t{n} * kHid * 4);
  }
  void QkvOut(std::size_t l, Job& j) {
    Program& p = attn_[l]->pq;
    p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    j.c.assign(p.ch, p.ch + std::size_t{kM} * 3072 / 8 * 9);
  }
  xrt::bo& QkvBo(std::size_t l) { return attn_[l]->pq.c; }
  void ReleaseQkv(std::size_t l, Job& j) {
    j.staged = false;
    attn_[l]->port_q->m.unlock();
  }
  void ReleaseO(std::size_t l, Job& j) {
    j.staged = false;
    attn_[l]->port_o->m.unlock();
  }
  void ReleaseFfn(std::size_t l, Job& j) {
    j.staged = false;
    layers_[l]->port->m.unlock();
  }

  void ExecGemm(Program& p, WeightBuffer& w, Job& j, std::uint32_t nout) {
    if (!j.staged) {
      std::memcpy(p.ah, j.a.data(), j.a.size());
      p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    Dispatch(gu_, w, p);
    p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (!j.staged) j.c.assign(p.ch, p.ch + std::size_t{kM} * nout / 8 * 9);
  }
  void ExecQkv(std::size_t l, Job& j) { ExecGemm(attn_[l]->pq, attn_[l]->wq, j, 3072); }
  void ExecO(std::size_t l, Job& j) { ExecGemm(attn_[l]->po, attn_[l]->wo, j, kHid); }
  void ExecFfn(std::size_t l, Job& j) {
    Layer& ly = *layers_[l];
    if (!j.staged) {
      std::memcpy(ly.qa.ah, j.a.data(), j.a.size());
      std::memcpy(ly.qb.ah, j.a.data(), j.a.size());
      ly.qa.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      ly.qb.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    Dispatch(gu_, ly.wqa, ly.qa);
    Dispatch(gu_, ly.wqb, ly.qb);
    if (!ly.fused) Dispatch(*swp_, ly.fake, ly.sw);
    Dispatch(*dnp_, ly.wd, ly.dn);
    ly.dn.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (!j.staged) j.c.assign(ly.dn.ch, ly.dn.ch + std::size_t{kM} * kHid / 8 * 9);
  }
  /// Conv in/out run inside the caller's per-layer turn: A is copied in before the device lock is taken and C is copied
  /// out after it is released.
  void StageConvIn(std::size_t l, Job& j) {
    Conv& c = *conv_[l];
    std::memcpy(c.pi.ah, j.a.data(), j.a.size());
    c.pi.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    j.staged = true;
  }
  void ExecConvIn(std::size_t l, Job& j) {
    Conv& c = *conv_[l];
    if (!j.staged) {
      std::memcpy(c.pi.ah, j.a.data(), j.a.size());
      c.pi.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    Dispatch(gu_, c.wi, c.pi);
    c.pi.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }
  void ExecConvOut(std::size_t l, Job& j, bool defer_copy = false) {
    Conv& c = *conv_[l];
    Dispatch(gu_, c.wo, c.po);
    c.po.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (!defer_copy) FinishConvOut(l, j);
  }
  void FinishConvOut(std::size_t l, Job& j) {
    Conv& c = *conv_[l];
    j.c.assign(c.po.ch, c.po.ch + std::size_t{kM} * kHid / 8 * 9);
  }

  /// One in_proj wire row (6144 floats, model order B|C|x).
  void ConvWireRow(std::size_t layer, std::uint32_t row, float* out) {
    gu_.DecodeCRowAtb(conv_[layer]->pi, 3 * kHid, row, out);
  }

  /// Gating and 7-tap conv on the device: in_proj wire -> out_proj A wire.
  /// taps: kHid x 7 floats; hist: 6 x kHid bf16 bit patterns (the history entering token 0).
  void RunConvDev(std::size_t layer, const float* taps, const std::uint16_t* hist) {
    Conv& c = *conv_[layer];
    const auto bf = [](float f) {
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      return static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    };
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t e = 0; e < 2; ++e)
        for (std::uint32_t rho = 0; rho < 4; ++rho) {
          auto* base = reinterpret_cast<std::uint16_t*>(c.wx.host + std::size_t{(col * 2 + e) * 4 + rho} * 9216);
          std::uint16_t* tp = base;
          std::uint16_t* hp = base + 1792;
          const std::uint32_t nt = col + 8 * e;
          for (std::uint32_t oo = 0; oo < 4; ++oo) {
            const std::uint32_t o = rho * 4 + oo;
            for (std::uint32_t e8 = 0; e8 < 8; ++e8) {
              const std::uint32_t ch = nt * 128 + o * 8 + e8;
              for (std::uint32_t j = 0; j < 7; ++j) {
                const std::uint16_t v = bf(taps[std::size_t{ch} * 7 + j]);
                for (std::uint32_t r = 0; r < 8; ++r) tp[(oo * 7 + j) * 64 + r * 8 + e8] = v;
              }
              for (std::uint32_t r = 0; r < 8; ++r) hp[oo * 64 + r * 8 + e8] = r < 2 ? 0 : hist[std::size_t{r - 2} * kHid + ch];
            }
          }
        }
    c.wx.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto d0 = std::chrono::steady_clock::now();
    Dispatch(gu_, c.wx, c.pc);
    cvdev_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
  }
  double cvdev_ms{0};

  /// out_proj on an A wire the device already filled.
  void RunConvOutRaw(std::size_t layer, std::uint32_t n, float* y) {
    Conv& c = *conv_[layer];
    Dispatch(gu_, c.wo, c.po);
    c.po.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (n == kM) {
      gu_.DecodeCAtb(y, kM, kHid, c.po);
    } else {
      out_.resize(std::size_t{kM} * kHid);
      gu_.DecodeCAtb(out_.data(), kM, kHid, c.po);
      std::memcpy(y, out_.data(), std::size_t{n} * kHid * 4);
    }
  }

  /// Debug: decode the A wire the conv kernel wrote.
  void ConvDevA(std::size_t layer, float* x) {
    Conv& c = *conv_[layer];
    c.po.a.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    gu_.DecodeAAtb(x, kM, kHid, c.po);
  }

  bool HasConv(std::size_t layer) const { return layer < conv_.size() && conv_[layer] != nullptr; }

  /// in_proj: z (n x 2048) -> y (n x 6144, model order [B|C|x]).
  void RunConvIn(std::size_t layer, const float* z, std::uint32_t n, float* y) {
    cv_ = 0;
    Gemm1024(conv_[layer]->pi, conv_[layer]->wi, z, n, kHid, 3 * kHid, y);
  }
  void RunConvOut(std::size_t layer, const float* a, std::uint32_t n, float* y) {
    cv_ = 1;
    Gemm1024(conv_[layer]->po, conv_[layer]->wo, a, n, kHid, kHid, y);
  }

  bool HasAttn(std::size_t layer) const { return layer < attn_.size() && attn_[layer] != nullptr; }

  /// Both run on the K=2048 context shared with the gate/up GEMM. n <= 1024 rows.
  void RunQkv(std::size_t layer, const float* z, std::uint32_t n, float* y) {
    cv_ = -1;
    Gemm1024(attn_[layer]->pq, attn_[layer]->wq, z, n, kHid, 3072, y);
  }
  void RunO(std::size_t layer, const float* a, std::uint32_t n, float* y) {
    cv_ = -1;
    Gemm1024(attn_[layer]->po, attn_[layer]->wo, a, n, kHid, kHid, y);
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
    if (!l.fused) Dispatch(*swp_, l.fake, l.sw);
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
    xrt::bo swc;
    bool fused{false};
    std::unique_ptr<Port> port = std::make_unique<Port>();
  };

  struct Attn {
    Program pq, po;
    WeightBuffer wq, wo;
    std::unique_ptr<Port> port_q = std::make_unique<Port>(), port_o = std::make_unique<Port>();
  };

  struct NormP {
    Program p;
    WeightBuffer w;
  };
  std::vector<std::unique_ptr<NormP>> norms_;
  xrt::bo resc_, resa_, br_;
  std::uint8_t *resc_h_{}, *resa_h_{};
  bool devres_{false};

  struct Conv {
    Program pi, po, pc;
    WeightBuffer wi, wo, wx;
    bool dev{false};
  };

  void Gemm1024(Program& p, WeightBuffer& w, const float* z, std::uint32_t n, std::uint32_t k,
                std::uint32_t nout, float* y) {
    const float* src = z;
    if (n < kM) {
      pad_.assign(std::size_t{kM} * k, 0.f);
      std::memcpy(pad_.data(), z, std::size_t{n} * k * 4);
      src = pad_.data();
    }
    const auto q0 = std::chrono::steady_clock::now();
    gu_.EncodeAAtb(src, kM, k, p);
    p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto q1 = std::chrono::steady_clock::now();
    Dispatch(gu_, w, p);
    p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto q2 = std::chrono::steady_clock::now();
    last_[0] = std::chrono::duration<double, std::milli>(q1 - q0).count();
    last_[1] = std::chrono::duration<double, std::milli>(q2 - q1).count();
    if (n == kM) {
      gu_.DecodeCAtb(y, kM, nout, p);
    } else {
      out_.resize(std::size_t{kM} * nout);
      gu_.DecodeCAtb(out_.data(), kM, nout, p);
      std::memcpy(y, out_.data(), std::size_t{n} * nout * 4);
    }
    last_[2] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - q2).count();
    if (cv_ >= 0)
      for (int i = 0; i < 3; ++i) acc_[cv_][i] += last_[i];
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
  bool swifuse_ok_{true};
  bool dbg_{std::getenv("GUFO_ATB_DBG") != nullptr};
  std::string ip_{"insts_atb_"};
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<std::unique_ptr<Attn>> attn_;
  std::vector<std::unique_ptr<Conv>> conv_;
  std::vector<float> pad_, out_;
 public:
  double last_[3]{}, acc_[2][3]{};
  int cv_{-1};
};

}  // namespace gufo::npu::runtime
