#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "src/npu/runtime/runtime.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using namespace gufo::npu::runtime;

constexpr std::uint32_t kRows = 16;
constexpr std::uint32_t kRows64 = 64;
constexpr std::uint32_t kPrefillRows = kRows64;
constexpr std::uint32_t kStripRows = 128;
constexpr std::uint32_t kPassCols = 1024;
constexpr std::uint32_t kAttnKv = 4 * 128;
constexpr std::uint32_t kExactCtx = 4096;
constexpr std::uint32_t PadN(std::uint32_t n) { return (n + kPassCols - 1) / kPassCols * kPassCols; }

double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

struct Config {
  std::uint32_t hidden{1536}, inter{4608}, layers{24}, heads{16}, kv_heads{2}, head_dim{128}, vocab{130560};
  float eps{1e-6f}, theta{5e6f};
};

float Bf(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Safetensors {
  void* map{nullptr};
  std::size_t size{0};
  std::string header;
  std::size_t data0{0};
  ~Safetensors() {
    if (map) munmap(map, size);
  }
  bool Open(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    fstat(fd, &st);
    size = static_cast<std::size_t>(st.st_size);
    map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
      map = nullptr;
      return false;
    }
    std::uint64_t n;
    std::memcpy(&n, map, 8);
    header.assign(static_cast<const char*>(map) + 8, n);
    data0 = 8 + n;
    return true;
  }
  const std::uint16_t* Find(const std::string& name, std::size_t* count) const {
    const std::string key = "\"" + name + "\"";
    const auto at = header.find(key);
    if (at == std::string::npos) return nullptr;
    const auto off = header.find("\"data_offsets\"", at);
    const auto lb = header.find('[', off);
    std::size_t a = 0, b = 0;
    std::sscanf(header.c_str() + lb, "[%zu,%zu]", &a, &b);
    *count = (b - a) / 2;
    return reinterpret_cast<const std::uint16_t*>(static_cast<const char*>(map) + data0 + a);
  }
  void Load(const std::string& name, float* dst, std::size_t expect) const {
    std::size_t cnt = 0;
    const auto* p = Find(name, &cnt);
    if (!p || cnt != expect) {
      std::fprintf(stderr, "tensor %s missing or wrong size (%zu vs %zu)\n", name.c_str(), cnt, expect);
      std::exit(1);
    }
#pragma omp parallel for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(cnt); ++i) dst[i] = Bf(p[i]);
  }
};

void RmsNorm(const float* x, const float* w, float* out, std::uint32_t d, float eps) {
  float sum = 0;
  for (std::uint32_t i = 0; i < d; ++i) sum += x[i] * x[i];
  const float scale = 1.0f / std::sqrt(sum / d + eps);
  for (std::uint32_t i = 0; i < d; ++i) out[i] = x[i] * scale * w[i];
}

std::vector<char> ReadNpy(const std::string& path, std::size_t* bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::uint16_t hl;
  std::memcpy(&hl, b.data() + 8, 2);
  const std::size_t off = 10 + hl;
  *bytes = b.size() - off;
  return std::vector<char>(b.begin() + off, b.end());
}

struct QMat {
  std::vector<std::int8_t> q;
  std::vector<std::uint16_t> s;
  std::uint32_t N{0}, K{0};
  void Resize(std::uint32_t n, std::uint32_t k) {
    N = n;
    K = k;
    q.assign(std::size_t{n} * k, 0);
    s.assign(std::size_t{n} * (k / 32), 0x3F80);
  }
  void Load(const std::string& dir, const std::string& name, std::uint32_t row0, std::uint32_t rows) {
    std::size_t nb = 0;
    auto qb = ReadNpy(dir + "/" + name + ".q.npy", &nb);
    if (nb != std::size_t{rows} * K) {
      std::fprintf(stderr, "%s q size mismatch\n", name.c_str());
      std::exit(1);
    }
    std::memcpy(&q[std::size_t{row0} * K], qb.data(), nb);
    auto sb = ReadNpy(dir + "/" + name + ".s.npy", &nb);
    if (nb != std::size_t{rows} * (K / 32) * 2) {
      std::fprintf(stderr, "%s s size mismatch\n", name.c_str());
      std::exit(1);
    }
    std::memcpy(&s[std::size_t{row0} * (K / 32)], sb.data(), nb);
  }
  void CopyRows(const QMat& src, std::uint32_t srow, std::uint32_t drow, std::uint32_t rows) {
    std::memcpy(&q[std::size_t{drow} * K], &src.q[std::size_t{srow} * K], std::size_t{rows} * K);
    std::memcpy(&s[std::size_t{drow} * (K / 32)], &src.s[std::size_t{srow} * (K / 32)], std::size_t{rows} * (K / 32) * 2);
  }
};

struct Layer {
  std::vector<float> attn_norm, ffn_norm;
  WeightBuffer qkv, out, ffn, ffn_dn;
  Program p_qkv, p_out, p_ffn, p_qkv64, p_out64, p_gu64, p_dn64;
  std::vector<float> k, v;
  std::vector<float> wqkv, wo, wgate, wup, wdown;
  AttnLayer al;
  std::uint32_t napp{0};
  WeightBuffer kw_qkv, kw_o, kw_a, kw_b, kw_d;
  Program kp_qkv, kp_o, kq_a, kq_b, kq_d;
  xrt::bo swc;
  std::vector<std::uint16_t> kh, vh;
};

void MatVec(const float* W, std::uint32_t N, std::uint32_t K, const float* x, std::uint32_t n, float* y) {
#pragma omp parallel for schedule(static)
  for (std::int64_t o = 0; o < N; ++o) {
    const float* w = W + std::size_t(o) * K;
    for (std::uint32_t r = 0; r < n; ++r) {
      const float* xr = x + std::size_t{r} * K;
      float s = 0;
      for (std::uint32_t i = 0; i < K; ++i) s += w[i] * xr[i];
      y[std::size_t{r} * N + o] = s;
    }
  }
}

struct Engine {
  Config c;
  NpuRuntime rt;
  NpuRuntime::Timings tim{};
  std::vector<Layer> layers;
  std::vector<float> norm_w;
  Program p_argmax;
  WeightBuffer w_argmax;
  bool npu_argmax{false};
  std::vector<std::uint16_t> embed, head;
  std::uint32_t pos{0}, ctx{0};
  std::uint32_t gu_strips{0};
  bool m64{std::getenv("MC5_NO_M64") == nullptr};
  NpuRuntime rk;
  bool kd{false};
  std::uint32_t kd_min{std::getenv("MC5_KD_MIN") ? static_cast<std::uint32_t>(std::atoi(std::getenv("MC5_KD_MIN"))) : 256u};
  std::vector<float> kx, ky;
  bool npu_attn{std::getenv("MC5_HOST_ATTN") == nullptr};
  std::string programs_dir;
  std::vector<float> ogain;
  std::uint32_t host_ctx{std::getenv("MC5_HOST_CTX") ? static_cast<std::uint32_t>(std::atoi(std::getenv("MC5_HOST_CTX"))) : 256u};
  std::unique_ptr<AttnPrefill> apre;
  const char* qdir{std::getenv("MC5_QDIR")};
  double t_npu{0}, t_cpu{0};
  bool cpu_ref{std::getenv("MC5_CPU") != nullptr || std::getenv("MC5_CPU_OP") != nullptr};
  std::string cpu_ops{std::getenv("MC5_CPU") ? "qkv,out,ffn" : (std::getenv("MC5_CPU_OP") ? std::getenv("MC5_CPU_OP") : "")};
  bool Cpu(const char* op) const { return cpu_ops.find(op) != std::string::npos; }

  bool Init(const std::string& model_dir, const std::string& programs, std::uint32_t context) {
    ctx = context;
    programs_dir = programs;
    if (const char* gp = std::getenv("MC5_OGAIN")) {
      std::ifstream gf(gp);
      float gv;
      while (gf >> gv) ogain.push_back(gv);
    }
    if (npu_attn && !std::getenv("MC5_NO_ATTNP")) {
      auto ap = std::make_unique<AttnPrefill>();
      std::string aerr;
      if (ap->Init(programs, &aerr)) apre = std::move(ap);
      else std::fprintf(stderr, "prefill attention unavailable: %s\n", aerr.c_str());
    }
    Safetensors st;
    if (!st.Open(model_dir + "/model-00000-of-00001.safetensors")) {
      std::fprintf(stderr, "cannot open safetensors\n");
      return false;
    }
    std::string err;
    if (!rt.Init(programs + "/final_16x64x128_ffn.xclbin", &err)) {
      std::fprintf(stderr, "npu init: %s\n", err.c_str());
      return false;
    }
    const std::uint32_t H = c.hidden, I = c.inter, qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    const std::uint32_t qkv_n = qd + 2 * kvd, qkv_p = PadN(qkv_n), h_p = PadN(H);
    gu_strips = 2 * I / kStripRows;
    const auto t0 = Clock::now();
    std::size_t cnt = 0;
    const auto* e = st.Find("model.embed_tokens.weight", &cnt);
    embed.assign(e, e + cnt);
    const auto* hd = st.Find("lm_head.weight", &cnt);
    head.assign(hd, hd + cnt);
    norm_w.resize(H);
    st.Load("model.norm.weight", norm_w.data(), H);
    if (std::getenv("MC5_KD") != nullptr) {
      std::string kerr;
      kd = rk.Init(programs + "/final_kd.xclbin", &kerr);
      if (!kd) std::fprintf(stderr, "kd unavailable: %s\n", kerr.c_str());
    }
    layers.resize(c.layers);
    std::vector<float> buf;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
      Layer& L = layers[l];
      const std::string pre = "model.layers." + std::to_string(l) + ".";
      L.attn_norm.resize(H);
      L.ffn_norm.resize(H);
      st.Load(pre + "input_layernorm.weight", L.attn_norm.data(), H);
      st.Load(pre + "post_attention_layernorm.weight", L.ffn_norm.data(), H);
      L.p_qkv = rt.MakeProgram(H, qkv_p, programs + "/insts_plain_1536x3072.txt", 0, &err);
      L.p_out = rt.MakeProgram(qd, h_p, programs + "/insts_plain_2048x2048.txt", 0, &err);
      if (m64) {
        L.p_qkv64 = rt.MakeProgram(H, qkv_p, programs + "/insts_plain_1536x3072_m64.txt", 0, &err, 0, 4);
        L.p_out64 = rt.MakeProgram(qd, h_p, programs + "/insts_plain_2048x2048_m64.txt", 0, &err, 0, 4);
        L.p_gu64 = rt.MakeProgram(H, 2 * I, programs + "/insts_ffngu64_1536.txt", 0, &err, 0, 4);
        L.p_dn64 = rt.MakeProgram(I, h_p, programs + "/insts_plain_4608x2048_m64.txt", 0, &err, 0, 4);
      }
      L.p_ffn = rt.MakeProgram(H, H, programs + "/insts_ffn_1536.txt",
                               std::size_t{gu_strips + h_p / kStripRows} * kStripBytes, &err);
      buf.assign(std::size_t{qkv_p} * H, 0.f);
      st.Load(pre + "self_attn.q_proj.weight", buf.data(), std::size_t{qd} * H);
      st.Load(pre + "self_attn.k_proj.weight", buf.data() + std::size_t{qd} * H, std::size_t{kvd} * H);
      st.Load(pre + "self_attn.v_proj.weight", buf.data() + std::size_t{qd + kvd} * H, std::size_t{kvd} * H);
      if (cpu_ref) L.wqkv = buf;
      if (qdir) {
        QMat m;
        m.Resize(qkv_p, H);
        m.Load(qdir, std::to_string(l) + ".q", 0, qd);
        m.Load(qdir, std::to_string(l) + ".k", qd, kvd);
        m.Load(qdir, std::to_string(l) + ".v", qd + kvd, kvd);
        L.qkv = rt.PackWeightQ(m.q.data(), m.s.data(), qkv_p, H);
      } else {
        L.qkv = rt.PackWeight(buf.data(), qkv_p, H);
      }
      buf.assign(std::size_t{h_p} * qd, 0.f);
      st.Load(pre + "self_attn.o_proj.weight", buf.data(), std::size_t{H} * qd);
      if (cpu_ref) L.wo.assign(buf.begin(), buf.begin() + std::size_t{H} * qd);
      if (qdir) {
        QMat m;
        m.Resize(h_p, qd);
        m.Load(qdir, std::to_string(l) + ".o", 0, H);
        if (ogain.size() == c.layers) {
          const float inv = 1.0f / ogain[l];
          for (auto& sc : m.s) sc = Bf16Bits(Bf(sc) * inv);
        }
        L.out = rt.PackWeightQ(m.q.data(), m.s.data(), h_p, qd);
      } else {
        L.out = rt.PackWeight(buf.data(), h_p, qd);
      }
      std::vector<float> gate(std::size_t{I} * H), up(std::size_t{I} * H);
      st.Load(pre + "mlp.gate_proj.weight", gate.data(), gate.size());
      st.Load(pre + "mlp.up_proj.weight", up.data(), up.size());
      if (cpu_ref) { L.wgate = gate; L.wup = up; }
      buf.resize(std::size_t{2 * I} * H);
      for (std::size_t f = 0; f < I / 8; ++f)
        for (std::size_t i = 0; i < 8; ++i) {
          std::memcpy(&buf[(f * 16 + i) * H], &gate[(f * 8 + i) * H], H * 4);
          std::memcpy(&buf[(f * 16 + 8 + i) * H], &up[(f * 8 + i) * H], H * 4);
        }
      const std::size_t gu_bytes = std::size_t{gu_strips} * (H / kTileK) * 4 * kQuarterBytes;
      const std::size_t dn_bytes = std::size_t{h_p / kStripRows} * (I / kTileK) * 4 * kQuarterBytes;
      WeightBuffer m;
      m.bytes = gu_bytes + dn_bytes;
      m.bo = xrt::bo(rt.device(), m.bytes, XRT_BO_FLAGS_HOST_ONLY, rt.kernel().group_id(3));
      m.host = m.bo.map<std::uint8_t*>();
      std::memset(m.host, 0, m.bytes);
      if (qdir) {
        QMat g, u, gu;
        g.Resize(I, H);
        u.Resize(I, H);
        gu.Resize(2 * I, H);
        g.Load(qdir, std::to_string(l) + ".gate", 0, I);
        u.Load(qdir, std::to_string(l) + ".up", 0, I);
        for (std::size_t f = 0; f < I / 8; ++f) {
          gu.CopyRows(g, f * 8, f * 16, 8);
          gu.CopyRows(u, f * 8, f * 16 + 8, 8);
        }
        PackMatInt5Q(gu.q.data(), gu.s.data(), 2 * I, H, m.host);
      } else {
        PackMatInt5(buf.data(), 2 * I, H, m.host);
      }
      buf.assign(std::size_t{h_p} * I, 0.f);
      st.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{H} * I);
      if (cpu_ref) L.wdown.assign(buf.begin(), buf.begin() + std::size_t{H} * I);
      QMat dn;
      if (qdir) {
        dn.Resize(h_p, I);
        dn.Load(qdir, std::to_string(l) + ".down", 0, H);
        PackMatInt5Q(dn.q.data(), dn.s.data(), h_p, I, m.host + gu_bytes);
      } else {
        PackMatInt5(buf.data(), h_p, I, m.host + gu_bytes);
      }
      if (m64) L.ffn_dn = qdir ? rt.PackWeightQ(dn.q.data(), dn.s.data(), h_p, I) : rt.PackWeight(buf.data(), h_p, I);
      m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      L.ffn = std::move(m);
      if (npu_attn) {
        L.al = rt.MakeAttnLayer(ctx);
        L.kh.assign(std::size_t{ctx} * kAttnKv, 0);
        L.vh.assign(std::size_t{ctx} * kAttnKv, 0);
      }
      if (kd) PackKd(L, st, pre, programs, &err);
      L.k.assign(std::size_t{ctx} * kvd, 0.f);
      L.v.assign(std::size_t{ctx} * kvd, 0.f);
    }
    if (!std::getenv("MC5_CPU_ARGMAX")) {
      const std::uint32_t vp = PadN(c.vocab);
      p_argmax = rt.MakeProgram(H, vp, programs + "/insts_argmax_1536x131072.txt", 0, &err);
      buf.assign(std::size_t{vp} * H, 0.f);
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(std::size_t{c.vocab} * H); ++i) buf[i] = Bf(head[i]);
      w_argmax = rt.PackWeight(buf.data(), vp, H);
      npu_argmax = true;
    }
    std::fprintf(stderr, "mc5: packed %u layers in %.1f s\n", c.layers, Ms(t0) / 1000.0);
    return true;
  }

  std::uint32_t Chunk(std::uint32_t remaining) const {
    if (kd && remaining >= kd_min) return std::min(remaining, 1024u);
    return std::min(remaining, kPrefillRows);
  }

  void Reset() {
    pos = 0;
    if (npu_attn)
      for (auto& L : layers) {
        rt.AttnReset(L.al);
        L.napp = 0;
        std::fill(L.kh.begin(), L.kh.end(), std::uint16_t{0});
        std::fill(L.vh.begin(), L.vh.end(), std::uint16_t{0});
      }
  }

  void EnsureAppended(Layer& L, std::uint32_t upto) {
    const std::uint32_t hd = c.head_dim, kvd = c.kv_heads * hd, rep = kAttnKv / kvd;
    std::vector<float> kd(std::size_t{16} * kAttnKv);
    while (L.napp < upto) {
      const std::uint32_t m = std::min(16u, upto - L.napp);
      for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t h = 0; h < 4; ++h) std::memcpy(&kd[std::size_t{r} * kAttnKv + h * hd], &L.k[std::size_t{L.napp + r} * kvd + (h / rep) * hd], hd * 4);
      rt.AttnAppend(L.al, L.napp, m, kd.data(), L.vh.data(), &tim);
      L.napp += m;
    }
  }

  void Rope(float* x, std::uint32_t nh, std::uint32_t p) const {
    const std::uint32_t hd = c.head_dim, half = hd / 2;
    for (std::uint32_t i = 0; i < half; ++i) {
      const double f = std::pow(static_cast<double>(c.theta), -2.0 * i / hd);
      const double a = static_cast<double>(p) * f;
      const float cs = static_cast<float>(std::cos(a)), sn = static_cast<float>(std::sin(a));
      for (std::uint32_t h = 0; h < nh; ++h) {
        float* v = x + std::size_t{h} * hd;
        const float a0 = v[i], a1 = v[i + half];
        v[i] = a0 * cs - a1 * sn;
        v[i + half] = a1 * cs + a0 * sn;
      }
    }
  }

  void Attend(Layer& L, const float* q, std::uint32_t p, float* out) const {
    const std::uint32_t hd = c.head_dim, kvd = c.kv_heads * hd, grp = c.heads / c.kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> sc(p + 1);
    for (std::uint32_t h = 0; h < c.heads; ++h) {
      const float* qh = q + std::size_t{h} * hd;
      const std::uint32_t kh = h / grp;
      float mx = -INFINITY;
      for (std::uint32_t t = 0; t <= p; ++t) {
        const float* kk = &L.k[std::size_t{t} * kvd + kh * hd];
        float s = 0;
        for (std::uint32_t i = 0; i < hd; ++i) s += qh[i] * kk[i];
        s *= scale;
        sc[t] = s;
        mx = std::max(mx, s);
      }
      float den = 0;
      for (std::uint32_t t = 0; t <= p; ++t) {
        sc[t] = std::exp(sc[t] - mx);
        den += sc[t];
      }
      float* o = out + std::size_t{h} * hd;
      std::fill(o, o + hd, 0.f);
      for (std::uint32_t t = 0; t <= p; ++t) {
        const float w = sc[t] / den;
        const float* vv = &L.v[std::size_t{t} * kvd + kh * hd];
        for (std::uint32_t i = 0; i < hd; ++i) o[i] += w * vv[i];
      }
    }
  }

  void PackKd(Layer& L, const Safetensors& st, const std::string& pre, const std::string& programs, std::string* err) {
    const std::uint32_t H = c.hidden, I = c.inter, qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    const std::uint32_t qkv_n = qd + 2 * kvd, qkv_p = PadN(qkv_n), h_p = PadN(H);
    const std::uint32_t half_a = 2048, half_b = I - half_a;
    std::vector<float> buf(std::size_t{qkv_p} * H, 0.f);
    st.Load(pre + "self_attn.q_proj.weight", buf.data(), std::size_t{qd} * H);
    st.Load(pre + "self_attn.k_proj.weight", buf.data() + std::size_t{qd} * H, std::size_t{kvd} * H);
    st.Load(pre + "self_attn.v_proj.weight", buf.data() + std::size_t{qd + kvd} * H, std::size_t{kvd} * H);
    L.kp_qkv = rk.MakeProgram(H, qkv_p, programs + "/insts_kd_1024x1536x3072.txt", std::size_t{1024} * qkv_p / 8 * 9, err, 0, 64, 3, 5);
    L.kw_qkv = rk.PackWeightAtb(buf.data(), qkv_p, H, 4);
    buf.assign(std::size_t{h_p} * qd, 0.f);
    st.Load(pre + "self_attn.o_proj.weight", buf.data(), std::size_t{H} * qd);
    L.kp_o = rk.MakeProgram(qd, h_p, programs + "/insts_kd_1024x2048x2048.txt", std::size_t{1024} * h_p / 8 * 9, err, 0, 64, 3, 5);
    L.kw_o = rk.PackWeightAtb(buf.data(), h_p, qd, 4);
    std::vector<float> gate(std::size_t{I} * H), up(std::size_t{I} * H), ws;
    st.Load(pre + "mlp.gate_proj.weight", gate.data(), gate.size());
    st.Load(pre + "mlp.up_proj.weight", up.data(), up.size());
    auto pack = [&](std::uint32_t c0, std::uint32_t n) {
      ws.resize(std::size_t{2} * n * H);
      for (std::uint32_t r = 0; r < 2 * n; ++r) {
        const std::uint32_t q = r >> 4, rr = r & 15;
        const float* src = (rr < 8 ? gate.data() : up.data()) + std::size_t{c0 + q * 8 + (rr & 7)} * H;
        std::memcpy(ws.data() + std::size_t{r} * H, src, H * 4);
      }
      return rk.PackWeightAtb(ws.data(), 2 * n, H, 4);
    };
    L.kw_a = pack(0, half_a);
    L.kw_b = pack(half_a, half_b);
    const std::size_t abytes = std::size_t{1024} * I / 8 * 9;
    L.swc = rk.MakeBo(2 * abytes, 3);
    std::memset(L.swc.map<void*>(), 0, 2 * abytes);
    L.swc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    L.kq_a = rk.MakeProgram(H, 2 * half_a, programs + "/insts_kd_swi_a_1024x1536x4096.txt", 64, err, 0, 64, 3, 5);
    L.kq_b = rk.MakeProgram(H, 2 * half_b, programs + "/insts_kd_swi_b_1024x1536x5120.txt", 64, err, 0, 64, 3, 5);
    L.kq_a.c = L.swc;
    L.kq_a.ch = L.swc.map<std::uint8_t*>();
    L.kq_b.c = L.swc;
    L.kq_b.ch = L.kq_a.ch;
    L.kq_d = rk.MakeProgram(I, h_p, programs + "/insts_kd_dn2_1024x4608x2048.txt", std::size_t{1024} * h_p / 8 * 9, err, 0, 64, 3, 5);
    rk.RebindA(L.kq_d, L.swc);
    buf.assign(std::size_t{h_p} * I, 0.f);
    st.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{H} * I);
    L.kw_d = rk.PackWeightAtb(buf.data(), h_p, I, 4);
    rk.WeightSlot(L.kw_d, L.kq_d);
  }

  void KGemm(Program& p, WeightBuffer& wb, const float* z, std::uint32_t n, std::uint32_t K, float* out) {
    if (kx.size() < std::size_t{1024} * 4608) kx.assign(std::size_t{1024} * 4608, 0.f);
    if (ky.size() < std::size_t{1024} * 3072) ky.assign(std::size_t{1024} * 3072, 0.f);
    if (n == 1024) {
      rk.GemmAtb(p, wb, z, 1024, out, &tim);
      return;
    }
    std::memcpy(kx.data(), z, std::size_t{n} * K * 4);
    std::memset(kx.data() + std::size_t{n} * K, 0, std::size_t{1024 - n} * K * 4);
    rk.GemmAtb(p, wb, kx.data(), 1024, ky.data(), &tim);
    std::memcpy(out, ky.data(), std::size_t{n} * p.N * 4);
  }

  void KDispatch(WeightBuffer& wb, Program& p) {
    WeightBuffer& ww = rk.WeightSlot(wb, p);
    ww.run.start();
    ww.run.wait();
    if (!p.warmed) {
      p.warmed = true;
      ww.run.start();
      ww.run.wait();
    }
  }

  void KMlp(Layer& L, const float* z, std::uint32_t n, float* out_h) {
    const std::uint32_t H = c.hidden, h_p = PadN(H);
    if (kx.size() < std::size_t{1024} * 4608) kx.assign(std::size_t{1024} * 4608, 0.f);
    if (ky.size() < std::size_t{1024} * 3072) ky.assign(std::size_t{1024} * 3072, 0.f);
    const float* src = z;
    if (n < 1024) {
      std::memcpy(kx.data(), z, std::size_t{n} * H * 4);
      std::memset(kx.data() + std::size_t{n} * H, 0, std::size_t{1024 - n} * H * 4);
      src = kx.data();
    }
    rk.EncodeAAtbRaw(src, 1024, H, L.kq_a.ah);
    std::memcpy(L.kq_b.ah, L.kq_a.ah, std::size_t{1024} * H / 8 * 9);
    L.kq_a.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    L.kq_b.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    KDispatch(L.kw_a, L.kq_a);
    KDispatch(L.kw_b, L.kq_b);
    KDispatch(L.kw_d, L.kq_d);
    L.kq_d.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    rk.DecodeCAtbRaw(L.kq_d.ch, ky.data(), 1024, h_p);
    for (std::uint32_t r = 0; r < n; ++r) std::memcpy(out_h + std::size_t{r} * H, &ky[std::size_t{r} * h_p], H * 4);
  }

  void Step(const std::uint32_t* tok, std::uint32_t n, std::vector<float>* hidden) {
    const std::uint32_t H = c.hidden, qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    const std::uint32_t qkv_n = qd + 2 * kvd, qkv_p = PadN(qkv_n), h_p = PadN(H);
    std::vector<float> x(std::size_t{n} * H), z(std::size_t{n} * H), qkv(std::size_t{n} * qkv_p),
        att(std::size_t{n} * qd), br(std::size_t{n} * h_p), brh(std::size_t{n} * H);
    for (std::uint32_t r = 0; r < n; ++r)
      for (std::uint32_t i = 0; i < H; ++i) x[std::size_t{r} * H + i] = Bf(embed[std::size_t{tok[r]} * H + i]);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
      Layer& L = layers[l];
      for (std::uint32_t r = 0; r < n; ++r) RmsNorm(&x[std::size_t{r} * H], L.attn_norm.data(), &z[std::size_t{r} * H], H, c.eps);
      auto t0 = Clock::now();
      const bool u64 = m64 && n > kRows;
      const bool ukd = kd && n >= kd_min && n <= 1024;
      if (ukd) KGemm(L.kp_qkv, L.kw_qkv, z.data(), n, H, qkv.data());
      else if (Cpu("qkv")) MatVec(L.wqkv.data(), qkv_n, H, z.data(), n, qkv.data());
      else rt.Gemm(u64 ? L.p_qkv64 : L.p_qkv, L.qkv, z.data(), n, qkv.data(), &tim);
      t_npu += Ms(t0);
      t0 = Clock::now();
      if (npu_attn) {
        std::vector<float> qa(std::size_t{n} * qd), kd(std::size_t{n} * kAttnKv);
        const std::uint32_t hd = c.head_dim, rep = kAttnKv / kvd;
        for (std::uint32_t r = 0; r < n; ++r) {
          float* row = &qkv[std::size_t{r} * (Cpu("qkv") ? qkv_n : qkv_p)];
          Rope(row, c.heads, pos + r);
          Rope(row + qd, c.kv_heads, pos + r);
          std::memcpy(&qa[std::size_t{r} * qd], row, qd * 4);
          for (std::uint32_t h = 0; h < 4; ++h) {
            const std::uint32_t src = h / rep;
            std::memcpy(&kd[std::size_t{r} * kAttnKv + h * hd], row + qd + src * hd, hd * 4);
            for (std::uint32_t i = 0; i < hd; ++i) {
              L.kh[std::size_t{pos + r} * kAttnKv + h * hd + i] = Bf16Bits(row[qd + src * hd + i]);
              L.vh[std::size_t{pos + r} * kAttnKv + h * hd + i] = Bf16Bits(row[qd + kvd + src * hd + i]);
            }
          }
        }
        if (pos + n <= host_ctx) {
          for (std::uint32_t r = 0; r < n; ++r) {
            const float* row = &qkv[std::size_t{r} * (Cpu("qkv") ? qkv_n : qkv_p)];
            std::memcpy(&L.k[std::size_t{pos + r} * kvd], row + qd, kvd * 4);
            std::memcpy(&L.v[std::size_t{pos + r} * kvd], row + qd + kvd, kvd * 4);
          }
#pragma omp parallel for
          for (std::int64_t r = 0; r < n; ++r) Attend(L, &qa[std::size_t(r) * qd], pos + static_cast<std::uint32_t>(r), &att[std::size_t(r) * qd]);
        } else if (apre && n > kRows) {
          EnsureAppended(L, pos);
          L.napp = pos + n;
          apre->SetExact(pos + n <= kExactCtx);
          for (std::uint32_t o = 0; o < n; o += 16)
            rt.AttnAppend(L.al, pos + o, std::min(16u, n - o), &kd[std::size_t{o} * kAttnKv], L.vh.data(), &tim);
          for (std::uint32_t o = 0; o < n; o += 128) {
            const std::uint32_t n2 = std::min(128u, n - o);
            if (!apre->Run(L.al, pos + o, n2, &qa[std::size_t{o} * qd], L.kh.data(), L.vh.data(), &att[std::size_t{o} * qd], &tim)) {
              std::fprintf(stderr, "prefill attention failed (layer %u pos %u)\n", l, pos + o);
              std::exit(3);
            }
          }
        } else {
        EnsureAppended(L, pos);
        L.napp = pos + n;
        for (std::uint32_t o = 0, m = 0; o < n; o += m) {
          m = std::min({16u, n - o, 16u - (pos + o) % 16u});
          rt.AttnAppend(L.al, pos + o, m, &kd[std::size_t{o} * kAttnKv], L.vh.data(), &tim);
          if (!rt.AttnRun(L.al, pos + o, m, &qa[std::size_t{o} * qd], L.kh.data(), L.vh.data(), programs_dir, &att[std::size_t{o} * qd], &tim)) {
            std::fprintf(stderr, "npu attention declined (layer %u pos %u)\n", l, pos + o);
            std::exit(3);
          }
        }
        }
        if (std::getenv("MC5_ATTN_CHECK")) {
          std::vector<float> ref(std::size_t{n} * qd);
          for (std::uint32_t r = 0; r < n; ++r) {
            const float* row = &qkv[std::size_t{r} * (Cpu("qkv") ? qkv_n : qkv_p)];
            std::memcpy(&L.k[std::size_t{pos + r} * kvd], row + qd, kvd * 4);
            std::memcpy(&L.v[std::size_t{pos + r} * kvd], row + qd + kvd, kvd * 4);
            Attend(L, qa.data() + std::size_t{r} * qd, pos + r, &ref[std::size_t{r} * qd]);
          }
          double num = 0, den = 0, sg = 0, dot = 0;
          for (std::size_t i = 0; i < ref.size(); ++i) {
            const double d = att[i] - ref[i];
            num += d * d;
            den += static_cast<double>(ref[i]) * ref[i];
            sg += d;
            dot += static_cast<double>(att[i]) * ref[i];
          }
          std::fprintf(stderr, "attn check layer %u pos %u n %u rel_err %.4f mean_err %.2e gain %.4f\n", l, pos, n, std::sqrt(num / den), sg / ref.size(), dot / den);
        }
      } else {
      for (std::uint32_t r = 0; r < n; ++r) {
        float* row = &qkv[std::size_t{r} * (Cpu("qkv") ? qkv_n : qkv_p)];
        Rope(row, c.heads, pos + r);
        Rope(row + qd, c.kv_heads, pos + r);
        std::memcpy(&L.k[std::size_t{pos + r} * kvd], row + qd, kvd * 4);
        std::memcpy(&L.v[std::size_t{pos + r} * kvd], row + qd + kvd, kvd * 4);
        if (std::getenv("MC5_HOST_BF16KV"))
          for (std::uint32_t i = 0; i < kvd; ++i) {
            L.k[std::size_t{pos + r} * kvd + i] = Bf(Bf16Bits(L.k[std::size_t{pos + r} * kvd + i]));
            L.v[std::size_t{pos + r} * kvd + i] = Bf(Bf16Bits(L.v[std::size_t{pos + r} * kvd + i]));
          }
        Attend(L, row, pos + r, &att[std::size_t{r} * qd]);
      }
      }
      t_cpu += Ms(t0);
      t0 = Clock::now();
      if (ukd) {
        KGemm(L.kp_o, L.kw_o, att.data(), n, qd, br.data());
        for (std::uint32_t r = 0; r < n; ++r) std::memcpy(&brh[std::size_t{r} * H], &br[std::size_t{r} * h_p], H * 4);
      } else if (Cpu("out")) MatVec(L.wo.data(), H, qd, att.data(), n, brh.data());
      else {
        rt.Gemm(u64 ? L.p_out64 : L.p_out, L.out, att.data(), n, br.data(), &tim);
        for (std::uint32_t r = 0; r < n; ++r) std::memcpy(&brh[std::size_t{r} * H], &br[std::size_t{r} * h_p], H * 4);
      }
      t_npu += Ms(t0);
      for (std::size_t i = 0; i < std::size_t{n} * H; ++i) x[i] += brh[i];
      for (std::uint32_t r = 0; r < n; ++r) RmsNorm(&x[std::size_t{r} * H], L.ffn_norm.data(), &z[std::size_t{r} * H], H, c.eps);
      t0 = Clock::now();
      if (ukd) {
        KMlp(L, z.data(), n, brh.data());
      } else if (Cpu("ffn")) {
        std::vector<float> g(std::size_t{n} * c.inter), u(std::size_t{n} * c.inter);
        MatVec(L.wgate.data(), c.inter, H, z.data(), n, g.data());
        MatVec(L.wup.data(), c.inter, H, z.data(), n, u.data());
        for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        MatVec(L.wdown.data(), H, c.inter, g.data(), n, brh.data());
      } else if (u64) {
        std::vector<float> yo(std::size_t{n} * h_p);
        rt.Ffn64(L.ffn, L.p_gu64, L.ffn_dn, L.p_dn64, z.data(), n, yo.data(), &tim, c.inter / kTileK);
        for (std::uint32_t r = 0; r < n; ++r) std::memcpy(&brh[std::size_t{r} * H], &yo[std::size_t{r} * h_p], H * 4);
      } else {
        rt.Ffn(L.p_ffn, L.ffn, z.data(), n, brh.data(), &tim, gu_strips, H);
      }
      t_npu += Ms(t0);
      for (std::size_t i = 0; i < std::size_t{n} * H; ++i) x[i] += brh[i];
    }
    pos += n;
    if (hidden) {
      hidden->resize(std::size_t{n} * H);
      for (std::uint32_t r = 0; r < n; ++r) RmsNorm(&x[std::size_t{r} * H], norm_w.data(), &(*hidden)[std::size_t{r} * H], H, c.eps);
    }
  }

  std::uint32_t Argmax(const float* h) {
    std::uint32_t id = 0;
    if (npu_argmax) {
      rt.Argmax(p_argmax, w_argmax, h, 1, &id, &tim);
      if (id < c.vocab) return id;
    }
    std::vector<float> lg(c.vocab);
    Logits(h, lg.data());
    return static_cast<std::uint32_t>(std::max_element(lg.begin(), lg.end()) - lg.begin());
  }

  float LogitAt(const float* h, std::uint32_t id) const {
    const std::uint16_t* w = &head[std::size_t{id} * c.hidden];
    float s = 0;
    for (std::uint32_t i = 0; i < c.hidden; ++i) s += Bf(w[i]) * h[i];
    return s;
  }

  void Logits(const float* h, float* out) const {
    const std::uint32_t H = c.hidden;
#pragma omp parallel for schedule(static)
    for (std::int64_t v = 0; v < c.vocab; ++v) {
      const std::uint16_t* w = &head[std::size_t(v) * H];
      float s = 0;
      for (std::uint32_t i = 0; i < H; ++i) s += Bf(w[i]) * h[i];
      out[v] = s;
    }
  }
};

std::vector<std::int32_t> ReadTokens(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::vector<std::int32_t> t(b.size() / 4);
  std::memcpy(t.data(), b.data(), t.size() * 4);
  return t;
}

}  // namespace

int GemmTest(const char* prog_dir, const char* inst, std::uint32_t K, std::uint32_t N) {
  NpuRuntime rt;
  std::string err;
  if (!rt.Init(std::string(prog_dir) + "/final_16x64x128_ffn.xclbin", &err)) return 1;
  std::vector<float> W(std::size_t{N} * K), x(std::size_t{kRows} * K), y(std::size_t{kRows} * N), ref(y.size());
  std::srand(1);
  for (auto& v : W) v = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 0.1f;
  for (auto& v : x) v = std::rand() / static_cast<float>(RAND_MAX) - 0.5f;
  auto prog = rt.MakeProgram(K, N, std::string(prog_dir) + "/" + inst, 0, &err);
  auto wb = rt.PackWeight(W.data(), N, K);
  rt.Gemm(prog, wb, x.data(), kRows, y.data(), nullptr);
  MatVec(W.data(), N, K, x.data(), kRows, ref.data());
  double num = 0, den = 0;
  std::uint32_t worst = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double d = y[i] - ref[i];
    num += d * d;
    den += static_cast<double>(ref[i]) * ref[i];
  }
  std::printf("%s K=%u N=%u rel_err %.4f  y[0..3]=%.3f %.3f %.3f %.3f ref=%.3f %.3f %.3f %.3f\n", inst, K, N, std::sqrt(num / den), y[0], y[1], y[2], y[3], ref[0], ref[1], ref[2], ref[3]);
  (void)worst;
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 6 && std::string(argv[1]) == "gemmtest") return GemmTest(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]));
  if (argc < 5) {
    std::fprintf(stderr, "usage: gufo-mc5 nll|top MODEL_DIR PROGRAMS TOKENS.i32 [max_tokens]\n");
    return 2;
  }
  const std::string cmd = argv[1];
  const auto toks = ReadTokens(argv[4]);
  if (cmd == "judge") {
    Engine je;
    if (!je.Init(argv[2], argv[3], 512 + kRows64)) return 1;
    std::size_t at = 3;
    const std::uint32_t yes = static_cast<std::uint32_t>(toks[0]), no = static_cast<std::uint32_t>(toks[1]), cases = static_cast<std::uint32_t>(toks[2]);
    std::vector<float> lg(je.c.vocab), hd;
    const auto t0 = Clock::now();
    std::uint32_t total = 0;
    for (std::uint32_t k = 0; k < cases; ++k) {
      const std::uint32_t len = static_cast<std::uint32_t>(toks[at++]);
      je.Reset();
      for (std::uint32_t o = 0, m = 0; o < len; o += m) {
        m = je.Chunk(len - o);
        std::vector<std::uint32_t> in(m);
        for (std::uint32_t i = 0; i < m; ++i) in[i] = static_cast<std::uint32_t>(toks[at + o + i]);
        je.Step(in.data(), m, &hd);
        if (o + m == len) {
          const float* last = &hd[std::size_t{m - 1} * je.c.hidden];
          lg[yes] = je.LogitAt(last, yes);
          lg[no] = je.LogitAt(last, no);
        }
      }
      at += len;
      total += len;
      std::printf("%+8.3f\n", lg[yes] - lg[no]);
    }
    std::fprintf(stderr, "npu total %.0f ms (enc %.0f syncA %.0f submit %.0f wait %.0f syncC %.0f dec %.0f), host attn+rope %.0f ms\n", je.t_npu, je.tim.enc, je.tim.syncA, je.tim.submit, je.tim.wait, je.tim.syncC, je.tim.dec, je.t_cpu);
    std::fprintf(stderr, "judge: %u cases %u tokens %.1f ms (%.1f ms/case)\n", cases, total, Ms(t0), Ms(t0) / cases);
    return 0;
  }
  std::uint32_t n_tok = static_cast<std::uint32_t>(toks.size());
  if (argc > 5) n_tok = std::min<std::uint32_t>(n_tok, static_cast<std::uint32_t>(std::atoi(argv[5])));
  Engine e;
  if (!e.Init(argv[2], argv[3], n_tok + kRows64)) return 1;
  std::vector<float> logits(e.c.vocab), hid;
  if (cmd == "nll") {
    double total = 0;
    std::uint32_t count = 0;
    const auto t0 = Clock::now();
    for (std::uint32_t o = 0, m = 0; o < n_tok; o += m) {
      m = e.Chunk(n_tok - o);
      std::vector<std::uint32_t> in(m);
      for (std::uint32_t i = 0; i < m; ++i) in[i] = static_cast<std::uint32_t>(toks[o + i]);
      e.Step(in.data(), m, &hid);
      for (std::uint32_t i = 0; i < m && o + i + 1 < n_tok; ++i) {
        e.Logits(&hid[std::size_t{i} * e.c.hidden], logits.data());
        float mx = -INFINITY;
        for (float v : logits) mx = std::max(mx, v);
        double den = 0;
        for (float v : logits) den += std::exp(static_cast<double>(v - mx));
        total += std::log(den) + mx - logits[static_cast<std::uint32_t>(toks[o + i + 1])];
        ++count;
      }
    }
    std::printf("tokens %u nll %.10f time %.1f s (npu %.0f ms, host attn %.0f ms)\n", count, total / count, Ms(t0) / 1000.0, e.t_npu, e.t_cpu);
    return 0;
  }
  if (cmd == "generate") {
    const std::uint32_t gen = argc > 5 ? static_cast<std::uint32_t>(std::atoi(argv[5])) : 64;
    Engine g;
    if (!g.Init(argv[2], argv[3], static_cast<std::uint32_t>(toks.size()) + gen + kRows64)) return 1;
    std::vector<float> hd;
    const auto t0 = Clock::now();
    const std::uint32_t len = static_cast<std::uint32_t>(toks.size());
    for (std::uint32_t o = 0, m = 0; o < len; o += m) {
      m = g.Chunk(len - o);
      std::vector<std::uint32_t> in(m);
      for (std::uint32_t i = 0; i < m; ++i) in[i] = static_cast<std::uint32_t>(toks[o + i]);
      g.Step(in.data(), m, &hd);
    }
    std::uint32_t next = g.Argmax(&hd[hd.size() - g.c.hidden]);
    const double pf = Ms(t0);
    const auto t1 = Clock::now();
    std::vector<std::uint32_t> out{next};
    for (std::uint32_t i = 1; i < gen; ++i) {
      g.Step(&next, 1, &hd);
      next = g.Argmax(hd.data());
      out.push_back(next);
    }
    const double dc = Ms(t1);
    for (auto t : out) std::printf("%u ", t);
    std::printf("\n");
    std::fprintf(stderr, "prefill %u tok %.1f ms (%.0f tok/s); decode %u tok %.1f ms (%.2f tok/s, %.1f ms/tok)\n", len, pf, len / pf * 1000, gen - 1, dc, (gen - 1) / dc * 1000, dc / (gen - 1));
    return 0;
  }
  if (cmd == "top") {
    for (std::uint32_t o = 0, m = 0; o < n_tok; o += m) {
      m = e.Chunk(n_tok - o);
      std::vector<std::uint32_t> in(m);
      for (std::uint32_t i = 0; i < m; ++i) in[i] = static_cast<std::uint32_t>(toks[o + i]);
      e.Step(in.data(), m, &hid);
      if (o + m == n_tok) e.Logits(&hid[std::size_t{m - 1} * e.c.hidden], logits.data());
    }
    std::vector<std::uint32_t> idx(e.c.vocab);
    for (std::uint32_t i = 0; i < e.c.vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](auto a, auto b) { return logits[a] > logits[b]; });
    for (int i = 0; i < 8; ++i) std::printf("%u %.4f\n", idx[i], logits[idx[i]]);
    return 0;
  }
  return 2;
}
