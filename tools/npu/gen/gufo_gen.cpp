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

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"
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
  bool rope_neox{false}, qk_norm{false}, tied{false};
};

float Bf(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Safetensors {
  std::unique_ptr<gufo::core::GgufReader> r;
  std::string arch;
  bool Open(const std::string& path) {
    std::string e;
    r = gufo::core::GgufReader::OpenFile(path, &e);
    if (!r) {
      std::fprintf(stderr, "gguf: %s\n", e.c_str());
      return false;
    }
    const auto a = r->GetMetadataString("general.architecture");
    arch = a ? std::string(*a) : "llama";
    return true;
  }
  static std::string Map(const std::string& n) {
    if (n == "model.embed_tokens.weight") return "token_embd.weight";
    if (n == "lm_head.weight") return "output.weight";
    if (n == "model.norm.weight") return "output_norm.weight";
    unsigned l = 0;
    char rest[128];
    if (std::sscanf(n.c_str(), "model.layers.%u.%127s", &l, rest) == 2) {
      static const std::pair<const char*, const char*> kMap[] = {
          {"input_layernorm.weight", "attn_norm.weight"}, {"post_attention_layernorm.weight", "ffn_norm.weight"},
          {"self_attn.q_proj.weight", "attn_q.weight"}, {"self_attn.k_proj.weight", "attn_k.weight"},
          {"self_attn.v_proj.weight", "attn_v.weight"}, {"self_attn.o_proj.weight", "attn_output.weight"},
          {"self_attn.q_norm.weight", "attn_q_norm.weight"}, {"self_attn.k_norm.weight", "attn_k_norm.weight"},
          {"mlp.gate_proj.weight", "ffn_gate.weight"}, {"mlp.up_proj.weight", "ffn_up.weight"},
          {"mlp.down_proj.weight", "ffn_down.weight"}};
      for (const auto& m : kMap)
        if (std::string(rest) == m.first) return "blk." + std::to_string(l) + "." + m.second;
    }
    return n;
  }
  bool Has(const std::string& n) const { return r->FindTensor(Map(n)) != nullptr; }
  void Dims(const std::string& n, std::uint32_t* cols, std::uint32_t* rows) const {
    const auto* t = r->FindTensor(Map(n));
    *cols = static_cast<std::uint32_t>(t->dimensions[0]);
    *rows = t->dimensions.size() > 1 ? static_cast<std::uint32_t>(t->dimensions[1]) : 1;
  }
  void Load(const std::string& name, float* dst, std::size_t expect) const {
    const auto* t = r->FindTensor(Map(name));
    if (!t) {
      std::fprintf(stderr, "tensor %s missing\n", name.c_str());
      std::exit(1);
    }
    std::uint32_t cols, rows;
    Dims(name, &cols, &rows);
    if (std::size_t{cols} * rows != expect) {
      std::fprintf(stderr, "tensor %s size %zu expected %zu\n", name.c_str(), std::size_t{cols} * rows, expect);
      std::exit(1);
    }
    const gufo::models::lfm3::Tensor tn{t->data, t->type, cols, rows, t->size_bytes};
#pragma omp parallel for
    for (std::int64_t row = 0; row < rows; ++row)
      gufo::models::lfm3::DecodeRow(tn, static_cast<std::uint32_t>(row), std::span<float>(dst + std::size_t(row) * cols, cols));
  }
  void LoadBf16(const std::string& name, std::vector<std::uint16_t>* out, std::size_t expect) const;
};

void Safetensors::LoadBf16(const std::string& name, std::vector<std::uint16_t>* out, std::size_t expect) const {
  const auto* t = r->FindTensor(Map(name));
  if (!t) {
    std::fprintf(stderr, "tensor %s missing\n", name.c_str());
    std::exit(1);
  }
  std::uint32_t cols, rows;
  Dims(name, &cols, &rows);
  if (std::size_t{cols} * rows != expect) {
    std::fprintf(stderr, "tensor %s size mismatch\n", name.c_str());
    std::exit(1);
  }
  out->resize(expect);
  const gufo::models::lfm3::Tensor tn{t->data, t->type, cols, rows, t->size_bytes};
#pragma omp parallel
  {
    std::vector<float> tmp(cols);
#pragma omp for
    for (std::int64_t row = 0; row < rows; ++row) {
      gufo::models::lfm3::DecodeRow(tn, static_cast<std::uint32_t>(row), std::span<float>(tmp.data(), cols));
      for (std::uint32_t i = 0; i < cols; ++i) (*out)[std::size_t(row) * cols + i] = Bf16Bits(tmp[i]);
    }
  }
}

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
  std::vector<float> qn, kn;
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
  bool m64{std::getenv("GEN_NO_M64") == nullptr};
  NpuRuntime rk;
  bool kd{false};
  std::uint32_t kd_min{std::getenv("GEN_KD_MIN") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GEN_KD_MIN"))) : 256u};
  std::vector<float> kx, ky;
  std::size_t kxn{0}, kyn{0};
  bool npu_attn{std::getenv("GEN_HOST_ATTN") == nullptr};
  std::string programs_dir;
  std::vector<float> ogain;
  std::uint32_t host_ctx{std::getenv("GEN_HOST_CTX") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GEN_HOST_CTX"))) : 256u};
  std::unique_ptr<AttnPrefill> apre;
  const char* qdir{std::getenv("GEN_QDIR")};
  double t_npu{0}, t_cpu{0};
  bool cpu_ref{std::getenv("GEN_CPU") != nullptr || std::getenv("GEN_CPU_OP") != nullptr};
  std::string cpu_ops{std::getenv("GEN_CPU") ? "qkv,out,ffn" : (std::getenv("GEN_CPU_OP") ? std::getenv("GEN_CPU_OP") : "")};
  bool Cpu(const char* op) const { return cpu_ops.find(op) != std::string::npos; }

  struct Names {
    std::string qkv16, out16, qkv64, out64, dn64, gu64, ffn16, argmax, kqkv, ko, swa, swb, dn2;
  } nm;
  std::uint32_t half_a{0};

  void EnsureProgs(const std::string& programs, std::uint32_t H, std::uint32_t qd, std::uint32_t qkv_p, std::uint32_t h_p, std::uint32_t I, std::uint32_t vp) {
    const std::string repo = programs + "/..";
    const std::string env = "source " + repo + "/../gen/env.sh; ";
    const std::string gi = "bash " + repo + "/gen_insts.sh ";
    const std::string gk = "bash " + repo + "/kdyn/gen_kd.sh ";
    const auto need = [&](const std::string& file, const std::string& cmd) {
      std::ifstream f(programs + "/" + file);
      if (f.good()) return;
      std::fprintf(stderr, "gen: generating %s\n", file.c_str());
      const std::string full = "bash -c '" + env + cmd + "' >/dev/null";
      if (std::system(full.c_str()) != 0) {
        std::fprintf(stderr, "gen: failed to generate %s (toolchain env: tools/npu/gen/env.sh)\n", file.c_str());
        std::exit(1);
      }
    };
    const auto n = [](std::uint32_t v) { return std::to_string(v); };
    need(nm.qkv16, gi + "plain " + n(H) + " " + n(qkv_p));
    need(nm.out16, gi + "plain " + n(qd) + " " + n(h_p));
    need(nm.ffn16, gi + "ffn " + n(H) + " " + n(2 * I) + " " + n(I) + " " + n(h_p) + " 1 " + nm.ffn16);
    need(nm.argmax, gi + "argmax " + n(H) + " " + n(vp) + " " + nm.argmax);
    if (m64) {
      need(nm.qkv64, gi + "plain " + n(H) + " " + n(qkv_p) + " 4");
      need(nm.out64, gi + "plain " + n(qd) + " " + n(h_p) + " 4");
      need(nm.gu64, gi + "ffn " + n(H) + " " + n(2 * I) + " 0 0 4 " + nm.gu64);
      need(nm.dn64, gi + "plain " + n(I) + " " + n(h_p) + " 4");
    }
    if (std::getenv("GEN_KD") != nullptr) {
      const std::string base = "-M 1024 ";
      need(nm.kqkv + ".txt", gk + nm.kqkv + " " + n(I) + " " + base + "-K " + n(H) + " -N " + n(qkv_p));
      need(nm.ko + ".txt", gk + nm.ko + " " + n(I) + " " + base + "-K " + n(qd) + " -N " + n(h_p));
      need(nm.swa + ".txt", gk + nm.swa + " " + n(I) + " " + base + "-K " + n(H) + " -N " + n(2 * half_a) + " --mode swigu --noff 0");
      need(nm.swb + ".txt", gk + nm.swb + " " + n(I) + " " + base + "-K " + n(H) + " -N " + n(2 * (I - half_a)) + " --mode swigu --noff " + n(half_a / 64));
      need(nm.dn2 + ".txt", gk + nm.dn2 + " " + n(I) + " " + base + "-K " + n(I) + " -N " + n(h_p) + " --a2 1");
    }
  }

  bool Init(const std::string& model_dir, const std::string& programs, std::uint32_t context) {
    ctx = context;
    programs_dir = programs;
    if (const char* gp = std::getenv("GEN_OGAIN")) {
      std::ifstream gf(gp);
      float gv;
      while (gf >> gv) ogain.push_back(gv);
    }
    if (npu_attn && !std::getenv("GEN_NO_ATTNP")) {
      auto ap = std::make_unique<AttnPrefill>();
      std::string aerr;
      if (ap->Init(programs, &aerr)) apre = std::move(ap);
      else std::fprintf(stderr, "prefill attention unavailable: %s\n", aerr.c_str());
    }
    Safetensors st;
    if (!st.Open(model_dir)) return false;
    {
      const auto& r = *st.r;
      const std::string a = st.arch + ".";
      auto u32 = [&](const char* k, std::uint32_t d) { const auto v = r.GetMetadataUint32(a + k); return v ? *v : d; };
      c.layers = u32("block_count", 0);
      c.hidden = u32("embedding_length", 0);
      c.inter = u32("feed_forward_length", 0);
      c.heads = u32("attention.head_count", 0);
      c.kv_heads = u32("attention.head_count_kv", c.heads);
      c.head_dim = u32("attention.key_length", c.hidden / std::max(1u, c.heads));
      const auto th = r.GetMetadataFloat32(a + "rope.freq_base");
      c.theta = th ? *th : 10000.f;
      const auto ep = r.GetMetadataFloat32(a + "attention.layer_norm_rms_epsilon");
      c.eps = ep ? *ep : 1e-6f;
      std::uint32_t cols, rows;
      st.Dims("model.embed_tokens.weight", &cols, &rows);
      c.vocab = rows;
      c.rope_neox = st.arch != "llama" && st.arch != "minicpm" && st.arch != "mistral";
      c.qk_norm = st.Has("model.layers.0.self_attn.q_norm.weight");
      c.tied = !st.Has("lm_head.weight");
      if (c.layers == 0 || c.hidden == 0 || c.inter == 0 || c.heads == 0) {
        std::fprintf(stderr, "unsupported GGUF metadata for architecture %s\n", st.arch.c_str());
        return false;
      }
      if (u32("rope.dimension_count", c.head_dim) != c.head_dim) {
        std::fprintf(stderr, "partial rope is not supported by the generic engine\n");
        return false;
      }
      if (st.Has("model.layers.0.self_attn.q_proj.bias") || r.FindTensor("blk.0.attn_q.bias")) {
        std::fprintf(stderr, "attention bias is not supported by the generic engine\n");
        return false;
      }
      if (c.hidden % 64 || (c.heads * c.head_dim) % 64 || c.inter % 512) {
        std::fprintf(stderr, "shapes must satisfy hidden%%64, q_dim%%64, ffn%%512 (hidden %u q %u ffn %u)\n", c.hidden, c.heads * c.head_dim, c.inter);
        return false;
      }
      if (npu_attn && !(c.head_dim == 128 && c.heads == 16 && (c.kv_heads == 1 || c.kv_heads == 2 || c.kv_heads == 4))) {
        std::fprintf(stderr, "gen: attention shape %ux%u/%u runs on the host (NPU kernel is 16 heads, 128 dims, 1/2/4 kv heads)\n", c.heads, c.head_dim, c.kv_heads);
        npu_attn = false;
        apre.reset();
      }
      std::fprintf(stderr, "gen: %s layers %u hidden %u ffn %u heads %u/%u hd %u vocab %u rope %s qknorm %d tied %d\n", st.arch.c_str(), c.layers, c.hidden, c.inter, c.heads, c.kv_heads, c.head_dim, c.vocab, c.rope_neox ? "neox" : "interleaved", c.qk_norm, c.tied);
    }
    std::string err;
    if (!rt.Init(programs + "/final_16x64x128_ffn.xclbin", &err)) {
      std::fprintf(stderr, "npu init: %s\n", err.c_str());
      return false;
    }
    const std::uint32_t H = c.hidden, I = c.inter, qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    const std::uint32_t qkv_n = qd + 2 * kvd, qkv_p = PadN(qkv_n), h_p = PadN(H);
    gu_strips = 2 * I / kStripRows;
    kxn = std::max({H, qd, I});
    kyn = std::max(qkv_p, h_p);
    const std::uint32_t vp_n = PadN(c.vocab);
    const auto t0 = Clock::now();
    const std::string mk = std::to_string(H) + "x" + std::to_string(I);
    const auto plain = [&](std::uint32_t K, std::uint32_t N, bool m64n) { return "insts_plain_" + std::to_string(K) + "x" + std::to_string(N) + (m64n ? "_m64.txt" : ".txt"); };
    nm.qkv16 = plain(H, qkv_p, false);
    nm.out16 = plain(qd, h_p, false);
    nm.qkv64 = plain(H, qkv_p, true);
    nm.out64 = plain(qd, h_p, true);
    nm.dn64 = plain(I, h_p, true);
    nm.gu64 = "insts_ffngu64_" + mk + ".txt";
    nm.ffn16 = "insts_ffn_" + mk + ".txt";
    nm.argmax = "insts_argmax_" + std::to_string(H) + "x" + std::to_string(vp_n) + ".txt";
    half_a = (I / 2) / 512 * 512;
    nm.kqkv = "insts_kd_1024x" + std::to_string(H) + "x" + std::to_string(qkv_p);
    nm.ko = "insts_kd_1024x" + std::to_string(qd) + "x" + std::to_string(h_p);
    nm.swa = "insts_kd_swi_a_" + mk + "_" + std::to_string(half_a);
    nm.swb = "insts_kd_swi_b_" + mk + "_" + std::to_string(half_a);
    nm.dn2 = "insts_kd_dn2_1024x" + std::to_string(I) + "x" + std::to_string(h_p);
    EnsureProgs(programs, H, qd, qkv_p, h_p, I, vp_n);
    st.LoadBf16("model.embed_tokens.weight", &embed, std::size_t{c.vocab} * H);
    if (!c.tied) st.LoadBf16("lm_head.weight", &head, std::size_t{c.vocab} * H);
    else head = embed;
    norm_w.resize(H);
    st.Load("model.norm.weight", norm_w.data(), H);
    if (std::getenv("GEN_KD") != nullptr) {
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
      if (c.qk_norm) {
        L.qn.resize(c.head_dim);
        L.kn.resize(c.head_dim);
        st.Load(pre + "self_attn.q_norm.weight", L.qn.data(), c.head_dim);
        st.Load(pre + "self_attn.k_norm.weight", L.kn.data(), c.head_dim);
      }
      L.p_qkv = rt.MakeProgram(H, qkv_p, programs + "/" + nm.qkv16, 0, &err);
      L.p_out = rt.MakeProgram(qd, h_p, programs + "/" + nm.out16, 0, &err);
      if (m64) {
        L.p_qkv64 = rt.MakeProgram(H, qkv_p, programs + "/" + nm.qkv64, 0, &err, 0, 4);
        L.p_out64 = rt.MakeProgram(qd, h_p, programs + "/" + nm.out64, 0, &err, 0, 4);
        L.p_gu64 = rt.MakeProgram(H, 2 * I, programs + "/" + nm.gu64, 0, &err, 0, 4);
        L.p_dn64 = rt.MakeProgram(I, h_p, programs + "/" + nm.dn64, 0, &err, 0, 4);
      }
      L.p_ffn = rt.MakeProgram(H, H, programs + "/" + nm.ffn16,
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
    if (!std::getenv("GEN_CPU_ARGMAX")) {
      const std::uint32_t vp = PadN(c.vocab);
      p_argmax = rt.MakeProgram(H, vp, programs + "/" + nm.argmax, 0, &err);
      buf.assign(std::size_t{vp} * H, 0.f);
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(std::size_t{c.vocab} * H); ++i) buf[i] = Bf(head[i]);
      w_argmax = rt.PackWeight(buf.data(), vp, H);
      npu_argmax = true;
    }
    std::fprintf(stderr, "gen: packed %u layers in %.1f s\n", c.layers, Ms(t0) / 1000.0);
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
        const std::uint32_t i0 = c.rope_neox ? i : 2 * i, i1 = c.rope_neox ? i + half : 2 * i + 1;
        const float a0 = v[i0], a1 = v[i1];
        v[i0] = a0 * cs - a1 * sn;
        v[i1] = a1 * cs + a0 * sn;
      }
    }
  }

  void QkNorm(const Layer& L, float* row, std::uint32_t qd) const {
    if (!c.qk_norm) return;
    const std::uint32_t hd = c.head_dim;
    for (std::uint32_t h = 0; h < c.heads; ++h) RmsNorm(row + h * hd, L.qn.data(), row + h * hd, hd, c.eps);
    for (std::uint32_t h = 0; h < c.kv_heads; ++h) RmsNorm(row + qd + h * hd, L.kn.data(), row + qd + h * hd, hd, c.eps);
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
    const std::uint32_t half_b = I - half_a;
    std::vector<float> buf(std::size_t{qkv_p} * H, 0.f);
    st.Load(pre + "self_attn.q_proj.weight", buf.data(), std::size_t{qd} * H);
    st.Load(pre + "self_attn.k_proj.weight", buf.data() + std::size_t{qd} * H, std::size_t{kvd} * H);
    st.Load(pre + "self_attn.v_proj.weight", buf.data() + std::size_t{qd + kvd} * H, std::size_t{kvd} * H);
    L.kp_qkv = rk.MakeProgram(H, qkv_p, programs + "/" + nm.kqkv + ".txt", std::size_t{1024} * qkv_p / 8 * 9, err, 0, 64, 3, 5);
    L.kw_qkv = rk.PackWeightAtb(buf.data(), qkv_p, H, 4);
    buf.assign(std::size_t{h_p} * qd, 0.f);
    st.Load(pre + "self_attn.o_proj.weight", buf.data(), std::size_t{H} * qd);
    L.kp_o = rk.MakeProgram(qd, h_p, programs + "/" + nm.ko + ".txt", std::size_t{1024} * h_p / 8 * 9, err, 0, 64, 3, 5);
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
    L.kq_a = rk.MakeProgram(H, 2 * half_a, programs + "/" + nm.swa + ".txt", 64, err, 0, 64, 3, 5);
    L.kq_b = rk.MakeProgram(H, 2 * half_b, programs + "/" + nm.swb + ".txt", 64, err, 0, 64, 3, 5);
    L.kq_a.c = L.swc;
    L.kq_a.ch = L.swc.map<std::uint8_t*>();
    L.kq_b.c = L.swc;
    L.kq_b.ch = L.kq_a.ch;
    L.kq_d = rk.MakeProgram(I, h_p, programs + "/" + nm.dn2 + ".txt", std::size_t{1024} * h_p / 8 * 9, err, 0, 64, 3, 5);
    rk.RebindA(L.kq_d, L.swc);
    buf.assign(std::size_t{h_p} * I, 0.f);
    st.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{H} * I);
    L.kw_d = rk.PackWeightAtb(buf.data(), h_p, I, 4);
    rk.WeightSlot(L.kw_d, L.kq_d);
  }

  void KGemm(Program& p, WeightBuffer& wb, const float* z, std::uint32_t n, std::uint32_t K, float* out) {
    if (kx.size() < std::size_t{1024} * kxn) kx.assign(std::size_t{1024} * kxn, 0.f);
    if (ky.size() < std::size_t{1024} * kyn) ky.assign(std::size_t{1024} * kyn, 0.f);
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
    if (kx.size() < std::size_t{1024} * kxn) kx.assign(std::size_t{1024} * kxn, 0.f);
    if (ky.size() < std::size_t{1024} * kyn) ky.assign(std::size_t{1024} * kyn, 0.f);
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
          QkNorm(L, row, qd);
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
        if (std::getenv("GEN_ATTN_CHECK")) {
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
        QkNorm(L, row, qd);
        Rope(row, c.heads, pos + r);
        Rope(row + qd, c.kv_heads, pos + r);
        std::memcpy(&L.k[std::size_t{pos + r} * kvd], row + qd, kvd * 4);
        std::memcpy(&L.v[std::size_t{pos + r} * kvd], row + qd + kvd, kvd * 4);
        if (std::getenv("GEN_HOST_BF16KV"))
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

int Chat(const char* model, const char* programs, const std::string& prompt, std::uint32_t max_new, bool bench_synth, std::uint32_t bench_prompt) {
  std::string e;
  auto reader = gufo::core::GgufReader::OpenFile(model, &e);
  if (!reader) {
    std::fprintf(stderr, "gguf: %s\n", e.c_str());
    return 1;
  }
  auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &e);
  if (!tok) {
    std::fprintf(stderr, "tokenizer: %s\n", e.c_str());
    return 1;
  }
  std::vector<std::uint32_t> ids;
  if (bench_synth) {
    for (std::uint32_t i = 0; i < bench_prompt; ++i) ids.push_back(1000 + (i * 7919) % 5000);
  } else {
    std::string text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (std::getenv("GEN_THINK") == nullptr && tok->Encode("<think>").size() == 1) text += "<think>\n\n</think>\n\n";
    const auto tmpl = reader->GetMetadataString("tokenizer.chat_template");
    if (tmpl && (tmpl->rfind("{{- bos_token", 0) == 0 || tmpl->rfind("{{ bos_token", 0) == 0)) {
      if (auto b = reader->GetMetadataUint32("tokenizer.ggml.bos_token_id")) ids.push_back(*b);
    }
    for (auto t : tok->Encode(text)) ids.push_back(t);
  }
  std::vector<std::uint32_t> stop;
  if (auto eo = reader->GetMetadataUint32("tokenizer.ggml.eos_token_id")) stop.push_back(*eo);
  for (const char* sp : {"<|im_end|>", "<|endoftext|>"}) {
    const auto v = tok->Encode(sp);
    if (v.size() == 1) stop.push_back(v[0]);
  }
  Engine g;
  if (!g.Init(model, programs, static_cast<std::uint32_t>(ids.size()) + max_new + 1024)) return 1;
  std::vector<float> hd;
  const auto t0 = Clock::now();
  for (std::uint32_t o = 0, m = 0; o < ids.size(); o += m) {
    m = g.Chunk(static_cast<std::uint32_t>(ids.size()) - o);
    g.Step(&ids[o], m, &hd);
  }
  std::uint32_t next = g.Argmax(&hd[hd.size() - g.c.hidden]);
  const double pf = Ms(t0);
  const auto t1 = Clock::now();
  std::uint32_t produced = 0;
  std::string out;
  for (std::uint32_t i = 0; i < max_new; ++i) {
    if (!bench_synth && std::find(stop.begin(), stop.end(), next) != stop.end()) break;
    ++produced;
    if (!bench_synth) {
      const std::string piece = tok->DecodeTokenCopy(next);
      std::fputs(piece.c_str(), stdout);
      std::fflush(stdout);
    }
    g.Step(&next, 1, &hd);
    next = g.Argmax(hd.data());
  }
  const double dc = Ms(t1);
  std::fprintf(stderr, "\nprefill %zu tok %.1f ms (%.0f tok/s); decode %u tok %.1f ms (%.2f tok/s)\n", ids.size(), pf, ids.size() / pf * 1000, produced, dc, produced / dc * 1000);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 5 && std::string(argv[1]) == "chat") return Chat(argv[2], argv[3], argv[4], argc > 5 ? std::atoi(argv[5]) : 256, false, 0);
  if (argc >= 5 && std::string(argv[1]) == "bench") return Chat(argv[2], argv[3], "", static_cast<std::uint32_t>(std::atoi(argv[5 < argc ? 5 : 4])), true, static_cast<std::uint32_t>(std::atoi(argv[4])));
  if (argc == 6 && std::string(argv[1]) == "gemmtest") return GemmTest(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]));
  if (argc < 5) {
    std::fprintf(stderr, "usage: gufo-gen chat|bench|nll|top|generate|judge MODEL.gguf PROGRAMS ...\n");
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
