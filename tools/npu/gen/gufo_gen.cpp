#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
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

float Fp16(std::uint16_t h) { return gufo::quant::Fp16ToFloat(h); }

bool DecodeRowAny(const gufo::models::lfm3::Tensor& tn, std::uint32_t row, std::span<float> out) {
  using gufo::core::GgmlType;
  const std::uint8_t* d = static_cast<const std::uint8_t*>(tn.data) + std::size_t{row} * (tn.bytes / tn.rows);
  const std::uint32_t cols = tn.cols;
  switch (tn.type) {
    case GgmlType::kQ4_1:
      for (std::uint32_t b = 0; b < cols / 32; ++b) {
        const std::uint8_t* blk = d + std::size_t{b} * 20;
        std::uint16_t dh, mh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&mh, blk + 2, 2);
        const float dd = Fp16(dh), mm = Fp16(mh);
        for (int i = 0; i < 16; ++i) {
          out[b * 32 + i] = dd * (blk[4 + i] & 0xF) + mm;
          out[b * 32 + 16 + i] = dd * (blk[4 + i] >> 4) + mm;
        }
      }
      return true;
    case GgmlType::kQ5_0:
      for (std::uint32_t b = 0; b < cols / 32; ++b) {
        const std::uint8_t* blk = d + std::size_t{b} * 22;
        std::uint16_t dh;
        std::uint32_t qh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&qh, blk + 2, 4);
        const float dd = Fp16(dh);
        for (int i = 0; i < 16; ++i) {
          const int x0 = ((blk[6 + i] & 0xF) | (((qh >> i) & 1) << 4)) - 16;
          const int x1 = ((blk[6 + i] >> 4) | (((qh >> (i + 16)) & 1) << 4)) - 16;
          out[b * 32 + i] = dd * x0;
          out[b * 32 + 16 + i] = dd * x1;
        }
      }
      return true;
    case GgmlType::kQ5_1:
      for (std::uint32_t b = 0; b < cols / 32; ++b) {
        const std::uint8_t* blk = d + std::size_t{b} * 24;
        std::uint16_t dh, mh;
        std::uint32_t qh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&mh, blk + 2, 2);
        std::memcpy(&qh, blk + 4, 4);
        const float dd = Fp16(dh), mm = Fp16(mh);
        for (int i = 0; i < 16; ++i) {
          const int x0 = (blk[8 + i] & 0xF) | (((qh >> i) & 1) << 4);
          const int x1 = (blk[8 + i] >> 4) | (((qh >> (i + 16)) & 1) << 4);
          out[b * 32 + i] = dd * x0 + mm;
          out[b * 32 + 16 + i] = dd * x1 + mm;
        }
      }
      return true;
    case GgmlType::kF32: case GgmlType::kF16: case GgmlType::kBF16: case GgmlType::kQ8_0: case GgmlType::kQ4_0:
    case GgmlType::kQ4_K: case GgmlType::kQ5_K: case GgmlType::kQ6_K:
      gufo::models::lfm3::DecodeRow(tn, row, out);
      return true;
    default:
      return false;
  }
}

bool TypeOk(gufo::core::GgmlType t) {
  using gufo::core::GgmlType;
  switch (t) {
    case GgmlType::kQ4_1: case GgmlType::kQ5_0: case GgmlType::kQ5_1: case GgmlType::kF32: case GgmlType::kF16: case GgmlType::kBF16:
    case GgmlType::kQ8_0: case GgmlType::kQ4_0: case GgmlType::kQ4_K: case GgmlType::kQ5_K: case GgmlType::kQ6_K:
      return true;
    default:
      return false;
  }
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
  bool HasG(const std::string& g) const { return r->FindTensor(g) != nullptr; }
  void DimsG(const std::string& g, std::uint32_t* cols, std::uint32_t* rows) const {
    const auto* t = r->FindTensor(g);
    *cols = static_cast<std::uint32_t>(t->dimensions[0]);
    *rows = t->dimensions.size() > 1 ? static_cast<std::uint32_t>(t->dimensions[1]) : 1;
  }
  void Load(const std::string& name, float* dst, std::size_t expect) const { LoadG(Map(name), dst, expect); }
  void LoadG(const std::string& gname, float* dst, std::size_t expect) const {
    const std::string name = gname;
    const auto* t = r->FindTensor(gname);
    if (!t) {
      std::fprintf(stderr, "tensor %s missing\n", name.c_str());
      std::exit(1);
    }
    std::uint32_t cols, rows;
    DimsG(gname, &cols, &rows);
    if (std::size_t{cols} * rows != expect) {
      std::fprintf(stderr, "tensor %s size %zu expected %zu\n", name.c_str(), std::size_t{cols} * rows, expect);
      std::exit(1);
    }
    if (!TypeOk(t->type)) {
      std::fprintf(stderr, "tensor %s has unsupported type %s\n", name.c_str(), std::string(gufo::core::ToString(t->type)).c_str());
      std::exit(1);
    }
    const gufo::models::lfm3::Tensor tn{t->data, t->type, cols, rows, t->size_bytes};
#pragma omp parallel for
    for (std::int64_t row = 0; row < rows; ++row)
      DecodeRowAny(tn, static_cast<std::uint32_t>(row), std::span<float>(dst + std::size_t(row) * cols, cols));
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
  if (!TypeOk(t->type)) {
    std::fprintf(stderr, "tensor %s has unsupported type %s\n", name.c_str(), std::string(gufo::core::ToString(t->type)).c_str());
    std::exit(1);
  }
  const gufo::models::lfm3::Tensor tn{t->data, t->type, cols, rows, t->size_bytes};
#pragma omp parallel
  {
    std::vector<float> tmp(cols);
#pragma omp for
    for (std::int64_t row = 0; row < rows; ++row) {
      DecodeRowAny(tn, static_cast<std::uint32_t>(row), std::span<float>(tmp.data(), cols));
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
  Program kp_qkv2, kp_o2, kq_a2, kq_b2, kq_d2;
  xrt::bo swc2;
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
  std::vector<float> kx, ky, rcos, rsin, s_qa, s_kd;
  std::vector<float> kxs[2], ymlp[2];
  std::size_t kxn{0}, kyn{0};
  std::uint32_t super_rows{std::getenv("GEN_SUPER") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GEN_SUPER"))) : 4096u};
  bool npu_attn{std::getenv("GEN_HOST_ATTN") == nullptr};
  std::string programs_dir;
  std::vector<float> ogain;
  std::uint32_t host_ctx{std::getenv("GEN_HOST_CTX") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GEN_HOST_CTX"))) : 256u};
  std::unique_ptr<AttnPrefill> apre;
  const char* qdir{std::getenv("GEN_QDIR")};
  double t_npu{0}, t_cpu{0}, ph[8]{0, 0, 0, 0, 0, 0, 0, 0};
  bool cpu_ref{std::getenv("GEN_CPU") != nullptr || std::getenv("GEN_CPU_OP") != nullptr};
  std::string cpu_ops{std::getenv("GEN_CPU") ? "qkv,out,ffn" : (std::getenv("GEN_CPU_OP") ? std::getenv("GEN_CPU_OP") : "")};
  bool Cpu(const char* op) const { return cpu_ops.find(op) != std::string::npos; }

  struct Names {
    std::string qkv16, out16, qkv64, out64, dn64, gu64, ffn16, argmax, kqkv, ko, swa, swb, dn2;
  } nm;
  std::uint32_t half_a{0};
  int dec_threads{std::getenv("GEN_DEC_THREADS") ? std::atoi(std::getenv("GEN_DEC_THREADS")) : 2};
  int pre_threads{std::getenv("GEN_PRE_THREADS") ? std::atoi(std::getenv("GEN_PRE_THREADS")) : 8};

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
    ctx = context;
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
    BuildRope();
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
    if (kd && remaining >= kd_min) return std::min(remaining, super_rows);
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

  void BuildRope() {
    const std::uint32_t hd = c.head_dim, half = hd / 2;
    rcos.resize(std::size_t{ctx + 64} * half);
    rsin.resize(std::size_t{ctx + 64} * half);
#pragma omp parallel for
    for (std::int64_t p = 0; p < ctx + 64; ++p)
      for (std::uint32_t i = 0; i < half; ++i) {
        const double a = static_cast<double>(p) * std::pow(static_cast<double>(c.theta), -2.0 * i / hd);
        rcos[std::size_t(p) * half + i] = static_cast<float>(std::cos(a));
        rsin[std::size_t(p) * half + i] = static_cast<float>(std::sin(a));
      }
  }

  void Rope(float* x, std::uint32_t nh, std::uint32_t p) const {
    const std::uint32_t hd = c.head_dim, half = hd / 2;
    for (std::uint32_t i = 0; i < half; ++i) {
      const float cs = rcos[std::size_t{p} * half + i], sn = rsin[std::size_t{p} * half + i];
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

  void PackKdMlp(Layer& L, const Safetensors& st, const std::string& pre, const std::string& programs, std::string* err) {
    const std::uint32_t H = c.hidden, I = c.inter, h_p = PadN(H);
    const std::uint32_t half_b = I - half_a;
    std::vector<float> buf;
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
    L.swc2 = rk.MakeBo(2 * abytes, 3);
    std::memset(L.swc2.map<void*>(), 0, 2 * abytes);
    L.swc2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    L.kq_a2 = rk.MakeProgram(H, 2 * half_a, programs + "/" + nm.swa + ".txt", 64, err, 0, 64, 3, 5);
    L.kq_b2 = rk.MakeProgram(H, 2 * half_b, programs + "/" + nm.swb + ".txt", 64, err, 0, 64, 3, 5);
    L.kq_a2.c = L.swc2;
    L.kq_a2.ch = L.swc2.map<std::uint8_t*>();
    L.kq_b2.c = L.swc2;
    L.kq_b2.ch = L.kq_a2.ch;
    L.kq_d2 = rk.MakeProgram(I, h_p, programs + "/" + nm.dn2 + ".txt", std::size_t{1024} * h_p / 8 * 9, err, 0, 64, 3, 5);
    rk.RebindA(L.kq_d2, L.swc2);
    rk.WeightSlot(L.kw_d, L.kq_d2);
    for (int set = 0; set < 2; ++set) {
      KDispatch(L.kw_a, set ? L.kq_a2 : L.kq_a);
      KDispatch(L.kw_b, set ? L.kq_b2 : L.kq_b);
      KDispatch(L.kw_d, set ? L.kq_d2 : L.kq_d);
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
    PackKdMlp(L, st, pre, programs, err);
    L.kp_qkv2 = rk.MakeProgram(H, qkv_p, programs + "/" + nm.kqkv + ".txt", std::size_t{1024} * qkv_p / 8 * 9, err, 0, 64, 3, 5);
    L.kp_o2 = rk.MakeProgram(qd, h_p, programs + "/" + nm.ko + ".txt", std::size_t{1024} * h_p / 8 * 9, err, 0, 64, 3, 5);
    KDispatch(L.kw_qkv, L.kp_qkv);
    KDispatch(L.kw_qkv, L.kp_qkv2);
    KDispatch(L.kw_o, L.kp_o);
    KDispatch(L.kw_o, L.kp_o2);
  }

  // Chunk pipeline: while the device runs chunk i the host encodes chunk i+1 (second program, own A buffer) and decodes chunk i-1.
  void KGemm(Program& p0, Program& p1, WeightBuffer& wb, const float* z, std::uint32_t n_all, std::uint32_t K, float* out) {
    if (kxs[0].size() < std::size_t{1024} * kxn) {
      kxs[0].assign(std::size_t{1024} * kxn, 0.f);
      kxs[1].assign(std::size_t{1024} * kxn, 0.f);
    }
    if (ky.size() < std::size_t{1024} * kyn) ky.assign(std::size_t{1024} * kyn, 0.f);
    const std::uint32_t nch = (n_all + 1023) / 1024;
    Program* pr[2] = {&p0, &p1};
    auto encode = [&](std::uint32_t i) {
      const std::uint32_t o = i * 1024, n = std::min(1024u, n_all - o);
      const float* src = z + std::size_t{o} * K;
      if (n < 1024) {
        std::memcpy(kxs[i & 1].data(), src, std::size_t{n} * K * 4);
        std::memset(kxs[i & 1].data() + std::size_t{n} * K, 0, std::size_t{1024 - n} * K * 4);
        src = kxs[i & 1].data();
      }
      rk.EncodeAAtb(src, 1024, K, *pr[i & 1]);
      pr[i & 1]->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    };
    auto decode = [&](std::uint32_t i) {
      const std::uint32_t o = i * 1024, n = std::min(1024u, n_all - o);
      Program& p = *pr[i & 1];
      p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      if (n == 1024) {
        rk.DecodeCAtb(out + std::size_t{o} * p.N, 1024, p.N, p);
      } else {
        rk.DecodeCAtb(ky.data(), 1024, p.N, p);
        std::memcpy(out + std::size_t{o} * p.N, ky.data(), std::size_t{n} * p.N * 4);
      }
    };
    encode(0);
    std::future<void> dec;
    for (std::uint32_t i = 0; i < nch; ++i) {
      std::future<void> enc;
      if (i + 1 < nch) enc = std::async(std::launch::async, [&, i] { encode(i + 1); });
      KDispatch(wb, *pr[i & 1]);
      if (enc.valid()) enc.get();
      if (dec.valid()) dec.get();
      dec = std::async(std::launch::async, [&, i] { decode(i); });
    }
    if (dec.valid()) dec.get();
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

  void KMlp(Layer& L, const float* z_all, std::uint32_t n_all, float* out_all) {
    const std::uint32_t H = c.hidden, h_p = PadN(H);
    if (kxs[0].size() < std::size_t{1024} * kxn) {
      kxs[0].assign(std::size_t{1024} * kxn, 0.f);
      kxs[1].assign(std::size_t{1024} * kxn, 0.f);
    }
    if (ky.size() < std::size_t{1024} * kyn) ky.assign(std::size_t{1024} * kyn, 0.f);
    const std::uint32_t nch = (n_all + 1023) / 1024;
    Program* qa[2] = {&L.kq_a, &L.kq_a2};
    Program* qb[2] = {&L.kq_b, &L.kq_b2};
    Program* qd[2] = {&L.kq_d, &L.kq_d2};
    auto encode = [&](std::uint32_t i) {
      const std::uint32_t o = i * 1024, n = std::min(1024u, n_all - o);
      const float* src = z_all + std::size_t{o} * H;
      if (n < 1024) {
        std::memcpy(kxs[i & 1].data(), src, std::size_t{n} * H * 4);
        std::memset(kxs[i & 1].data() + std::size_t{n} * H, 0, std::size_t{1024 - n} * H * 4);
        src = kxs[i & 1].data();
      }
      rk.EncodeAAtbRaw(src, 1024, H, qa[i & 1]->ah);
      std::memcpy(qb[i & 1]->ah, qa[i & 1]->ah, std::size_t{1024} * H / 8 * 9);
      qa[i & 1]->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      qb[i & 1]->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    };
    auto decode = [&](std::uint32_t i) {
      const std::uint32_t o = i * 1024, n = std::min(1024u, n_all - o);
      qd[i & 1]->c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::vector<float>& yb = ymlp[i & 1];
      if (yb.size() < std::size_t{1024} * h_p) yb.resize(std::size_t{1024} * h_p);
      rk.DecodeCAtbRaw(qd[i & 1]->ch, yb.data(), 1024, h_p);
      for (std::uint32_t r = 0; r < n; ++r) std::memcpy(out_all + std::size_t{o + r} * H, &yb[std::size_t{r} * h_p], H * 4);
    };
    encode(0);
    std::future<void> dec;
    for (std::uint32_t i = 0; i < nch; ++i) {
      std::future<void> enc;
      if (i + 1 < nch) enc = std::async(std::launch::async, [&, i] { encode(i + 1); });
      KDispatch(L.kw_a, *qa[i & 1]);
      KDispatch(L.kw_b, *qb[i & 1]);
      KDispatch(L.kw_d, *qd[i & 1]);
      if (enc.valid()) enc.get();
      if (dec.valid()) dec.get();
      dec = std::async(std::launch::async, [&, i] { decode(i); });
    }
    if (dec.valid()) dec.get();
  }

  void Step(const std::uint32_t* tok, std::uint32_t n, std::vector<float>* hidden) {
    omp_set_num_threads(n <= 2 ? dec_threads : pre_threads);
    const std::uint32_t H = c.hidden, qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    const std::uint32_t qkv_n = qd + 2 * kvd, qkv_p = PadN(qkv_n), h_p = PadN(H);
    std::vector<float> x(std::size_t{n} * H), z(std::size_t{n} * H), qkv(std::size_t{n} * qkv_p),
        att(std::size_t{n} * qd), br(std::size_t{n} * h_p), brh(std::size_t{n} * H);
    for (std::uint32_t r = 0; r < n; ++r)
      for (std::uint32_t i = 0; i < H; ++i) x[std::size_t{r} * H + i] = Bf(embed[std::size_t{tok[r]} * H + i]);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
      Layer& L = layers[l];
      auto tp = Clock::now();
#pragma omp parallel for
      for (std::int64_t r = 0; r < n; ++r) RmsNorm(&x[std::size_t(r) * H], L.attn_norm.data(), &z[std::size_t(r) * H], H, c.eps);
      ph[0] += Ms(tp);
      auto t0 = Clock::now();
      const bool u64 = m64 && n > kRows;
      const bool ukd = kd && n >= kd_min && n <= super_rows;
      if (ukd) KGemm(L.kp_qkv, L.kp_qkv2, L.kw_qkv, z.data(), n, H, qkv.data());
      else if (Cpu("qkv")) MatVec(L.wqkv.data(), qkv_n, H, z.data(), n, qkv.data());
      else rt.Gemm(u64 ? L.p_qkv64 : L.p_qkv, L.qkv, z.data(), n, qkv.data(), &tim);
      t_npu += Ms(t0);
      t0 = Clock::now();
      if (npu_attn) {
        s_qa.resize(std::size_t{n} * qd);
        s_kd.resize(std::size_t{n} * kAttnKv);
        std::vector<float>& qa = s_qa;
        std::vector<float>& kd = s_kd;
        const std::uint32_t hd = c.head_dim, rep = kAttnKv / kvd;
        auto tprep = Clock::now();
#pragma omp parallel for
        for (std::int64_t r = 0; r < n; ++r) {
          float* row = &qkv[std::size_t(r) * (Cpu("qkv") ? qkv_n : qkv_p)];
          QkNorm(L, row, qd);
          Rope(row, c.heads, pos + static_cast<std::uint32_t>(r));
          Rope(row + qd, c.kv_heads, pos + static_cast<std::uint32_t>(r));
          std::memcpy(&qa[std::size_t(r) * qd], row, qd * 4);
          for (std::uint32_t h = 0; h < 4; ++h) {
            const std::uint32_t src = h / rep;
            std::memcpy(&kd[std::size_t(r) * kAttnKv + h * hd], row + qd + src * hd, hd * 4);
            for (std::uint32_t i = 0; i < hd; ++i) {
              L.kh[std::size_t(pos + r) * kAttnKv + h * hd + i] = Bf16Bits(row[qd + src * hd + i]);
              L.vh[std::size_t(pos + r) * kAttnKv + h * hd + i] = Bf16Bits(row[qd + kvd + src * hd + i]);
            }
          }
        }
        ph[3] += Ms(tprep);
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
          const auto tap = Clock::now();
          if (pos % 16 == 0 && n % 16 == 0) {
            rt.AttnAppendBulk(L.al, pos, n, kd.data(), L.vh.data(), &tim);
          } else {
            for (std::uint32_t o = 0; o < n; o += 16)
              rt.AttnAppend(L.al, pos + o, std::min(16u, n - o), &kd[std::size_t{o} * kAttnKv], L.vh.data(), &tim);
          }
          ph[5] += Ms(tap);
          std::vector<AttnPrefill::Pass> passes;
          for (std::uint32_t o = 0; o < n; o += 128)
            passes.push_back({pos + o, std::min(128u, n - o), &qa[std::size_t{o} * qd], &att[std::size_t{o} * qd]});
          const auto tr = Clock::now();
          if (!apre->RunMany(L.al, passes.data(), passes.size(), L.kh.data(), L.vh.data(), &tim)) {
            std::fprintf(stderr, "prefill attention failed (layer %u pos %u)\n", l, pos);
            std::exit(3);
          }
          ph[4] += Ms(tr);
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
      ph[2] += Ms(t0);
      t0 = Clock::now();
      if (ukd) {
        KGemm(L.kp_o, L.kp_o2, L.kw_o, att.data(), n, qd, br.data());
        for (std::uint32_t r = 0; r < n; ++r) std::memcpy(&brh[std::size_t{r} * H], &br[std::size_t{r} * h_p], H * 4);
      } else if (Cpu("out")) MatVec(L.wo.data(), H, qd, att.data(), n, brh.data());
      else {
        rt.Gemm(u64 ? L.p_out64 : L.p_out, L.out, att.data(), n, br.data(), &tim);
        for (std::uint32_t r = 0; r < n; ++r) std::memcpy(&brh[std::size_t{r} * H], &br[std::size_t{r} * h_p], H * 4);
      }
      t_npu += Ms(t0);
      tp = Clock::now();
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(std::size_t{n} * H); ++i) x[i] += brh[i];
#pragma omp parallel for
      for (std::int64_t r = 0; r < n; ++r) RmsNorm(&x[std::size_t(r) * H], L.ffn_norm.data(), &z[std::size_t(r) * H], H, c.eps);
      ph[0] += Ms(tp);
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
      ph[1] += Ms(t0);
      tp = Clock::now();
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(std::size_t{n} * H); ++i) x[i] += brh[i];
      ph[0] += Ms(tp);
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


// ---------------------------------------------------------------------------------------------------------------------
// Hybrid models (qwen35 architecture: gated delta-net layers + non-causal gated attention, one forward pass per sequence).
// Same engine core as the dense path (pipelined 1024-row GEMM and fused MLP chunks, scratch reuse); only the layer kinds and
// the layer-major loop over a batch of sequences differ.
// ---------------------------------------------------------------------------------------------------------------------
constexpr std::uint32_t kKH = 16, kVH = 32, kKD = 128, kVD = 128, kConvK = 4, kRotDim = 64;
constexpr std::uint32_t kKeyDim = kKH * kKD, kValDim = kVH * kVD, kConvDim = 2 * kKeyDim + kValDim;
constexpr std::uint32_t kGdnInN = kConvDim + kValDim;

float SiluF(float x) { return x / (1.0f + std::exp(-x)); }

struct GdnDev {
  xrt::device dev;
  xrt::kernel kern;
  std::vector<std::pair<std::uint32_t, xrt::bo>> insts;
  std::vector<std::uint32_t> inst_words;
  xrt::bo bin[2], bx, bout[2];
  std::uint8_t* in_h[2]{nullptr, nullptr};
  const std::uint8_t* out_h[2]{nullptr, nullptr};
  static constexpr std::uint32_t kMaxTok = 2048;
  bool Init(NpuRuntime& rk, const std::string& programs) {
    try {
      dev = rk.device();
      kern = rk.kernel();
      for (std::uint32_t t : {64u, 128u, 256u, 512u, 1024u, 2048u}) {
        std::ifstream f(programs + "/insts_kdh_gdn_" + std::to_string(t) + ".txt", std::ios::binary | std::ios::ate);
        if (!f) return false;
        const std::size_t n = static_cast<std::size_t>(f.tellg());
        f.seekg(0);
        std::vector<char> v(n);
        f.read(v.data(), static_cast<std::streamsize>(n));
        xrt::bo b(dev, n, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
        std::memcpy(b.map<void*>(), v.data(), n);
        b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        insts.emplace_back(t, b);
        inst_words.push_back(static_cast<std::uint32_t>(n / 4));
      }
      bx = xrt::bo(dev, 4, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
      for (int i = 0; i < 2; ++i) {
        bin[i] = xrt::bo(dev, std::size_t{8} * (kMaxTok / 4 + 1) * 9216, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
        bout[i] = xrt::bo(dev, std::size_t{8} * (kMaxTok / 64) * 73728, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
        in_h[i] = bin[i].map<std::uint8_t*>();
        out_h[i] = bout[i].map<std::uint8_t*>();
        std::memset(in_h[i], 0, std::size_t{8} * (kMaxTok / 4 + 1) * 9216);
        std::memset(bout[i].map<void*>(), 0, std::size_t{8} * (kMaxTok / 64) * 73728);
      }
      return true;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "gdn device: %s\n", e.what());
      return false;
    }
  }
  static std::uint32_t Bucket(std::uint32_t tokens) {
    for (std::uint32_t t : {64u, 128u, 256u, 512u, 1024u, 2048u})
      if (tokens <= t) return t;
    return 0;
  }
  void Run(int set, std::uint32_t bucket) {
    std::size_t idx = 0;
    for (; idx < insts.size(); ++idx)
      if (insts[idx].first == bucket) break;
    bin[set].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kern(3, insts[idx].second, inst_words[idx], bx, bin[set], bout[set]);
    run.wait();
    bout[set].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }
};

struct HybridEngine : Engine {
  struct HL {
    bool linear{false};
    std::vector<float> conv, convT, a_neg, dt_bias, gnorm, w_a, w_b;
  };
  std::vector<HL> hl;
  GdnDev gd;
  bool gdn_dev{std::getenv("GEN_GDN_HOST") == nullptr};
  std::vector<float> readout;
  std::uint32_t n_codes{0};
  std::vector<float> s_proj, s_gated, s_outp, s_att, s_qkv;
  double mx[8]{0, 0, 0, 0, 0, 0, 0, 0};

  static std::uint32_t HfHead(std::uint32_t j) { return j < 16 ? 2 * j : 2 * (j - 16) + 1; }

  // GGUF stores the 32 value heads grouped even-first; rebuild the reference (repeat-interleave) order.
  static void PermBlocks(std::vector<float>& data, std::size_t first_row, std::uint32_t block_rows, std::size_t cols) {
    std::vector<float> tmp(std::size_t{kVH} * block_rows * cols);
    std::memcpy(tmp.data(), &data[first_row * cols], tmp.size() * 4);
    for (std::uint32_t j = 0; j < kVH; ++j)
      std::memcpy(&data[(first_row + std::size_t{HfHead(j)} * block_rows) * cols], &tmp[std::size_t{j} * block_rows * cols], std::size_t{block_rows} * cols * 4);
  }
  static void PermCols(std::vector<float>& data, std::size_t rows, std::size_t cols, std::uint32_t block_cols) {
    std::vector<float> tmp(cols);
    for (std::size_t r = 0; r < rows; ++r) {
      float* row = &data[r * cols];
      std::memcpy(tmp.data(), row, cols * 4);
      for (std::uint32_t j = 0; j < kVH; ++j) std::memcpy(row + std::size_t{HfHead(j)} * block_cols, &tmp[std::size_t{j} * block_cols], std::size_t{block_cols} * 4);
    }
  }

  // MLP weights from the GGUF names (ffn_gate/up/down), same packing as the dense path.
  void PackKdMlpG(Layer& L, const Safetensors& st, const std::string& b, const std::string& programs, std::string* err) {
    const std::uint32_t H = c.hidden, I = c.inter, h_p = PadN(H);
    const std::uint32_t half_b = I - half_a;
    std::vector<float> gate(std::size_t{I} * H), up(std::size_t{I} * H), ws, buf;
    st.LoadG(b + "ffn_gate.weight", gate.data(), gate.size());
    st.LoadG(b + "ffn_up.weight", up.data(), up.size());
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
    for (int set = 0; set < 2; ++set) {
      xrt::bo& sw = set ? L.swc2 : L.swc;
      sw = rk.MakeBo(2 * abytes, 3);
      std::memset(sw.map<void*>(), 0, 2 * abytes);
      sw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      Program& pa = set ? L.kq_a2 : L.kq_a;
      Program& pb = set ? L.kq_b2 : L.kq_b;
      Program& pd = set ? L.kq_d2 : L.kq_d;
      pa = rk.MakeProgram(H, 2 * half_a, programs + "/" + nm.swa + ".txt", 64, err, 0, 64, 3, 5);
      pb = rk.MakeProgram(H, 2 * half_b, programs + "/" + nm.swb + ".txt", 64, err, 0, 64, 3, 5);
      pa.c = sw;
      pa.ch = sw.map<std::uint8_t*>();
      pb.c = sw;
      pb.ch = pa.ch;
      pd = rk.MakeProgram(I, h_p, programs + "/" + nm.dn2 + ".txt", std::size_t{1024} * h_p / 8 * 9, err, 0, 64, 3, 5);
      rk.RebindA(pd, sw);
      if (set == 0) {
        buf.assign(std::size_t{h_p} * I, 0.f);
        st.LoadG(b + "ffn_down.weight", buf.data(), std::size_t{H} * I);
        L.kw_d = rk.PackWeightAtb(buf.data(), h_p, I, 4);
      }
      rk.WeightSlot(L.kw_d, pd);
    }
    for (int set = 0; set < 2; ++set) {
      KDispatch(L.kw_a, set ? L.kq_a2 : L.kq_a);
      KDispatch(L.kw_b, set ? L.kq_b2 : L.kq_b);
      KDispatch(L.kw_d, set ? L.kq_d2 : L.kq_d);
    }
  }

  bool InitH(const std::string& model, const std::string& programs, std::uint32_t context, const std::vector<std::uint32_t>& code_ids) {
    ctx = context;
    programs_dir = programs;
    Safetensors st;
    if (!st.Open(model)) return false;
    const auto& r = *st.r;
    c.layers = *r.GetMetadataUint32("qwen35.block_count");
    c.hidden = *r.GetMetadataUint32("qwen35.embedding_length");
    c.inter = *r.GetMetadataUint32("qwen35.feed_forward_length");
    c.heads = *r.GetMetadataUint32("qwen35.attention.head_count");
    c.kv_heads = *r.GetMetadataUint32("qwen35.attention.head_count_kv");
    c.head_dim = *r.GetMetadataUint32("qwen35.attention.key_length");
    c.theta = *r.GetMetadataFloat32("qwen35.rope.freq_base");
    c.eps = *r.GetMetadataFloat32("qwen35.attention.layer_norm_rms_epsilon");
    const std::uint32_t interval = *r.GetMetadataUint32("qwen35.full_attention_interval");
    if (c.head_dim != 256 || c.heads != 16 || c.kv_heads != 4 || *r.GetMetadataUint32("qwen35.ssm.time_step_rank") != kVH ||
        *r.GetMetadataUint32("qwen35.ssm.group_count") != kKH || *r.GetMetadataUint32("qwen35.ssm.state_size") != kKD ||
        *r.GetMetadataUint32("qwen35.rope.dimension_count") != kRotDim || c.hidden % 64 || c.inter % 512) {
      std::fprintf(stderr, "hybrid: unsupported qwen35 shape\n");
      return false;
    }
    std::string err;
    if (!rk.Init(programs + "/final_kdh.xclbin", &err)) {
      std::fprintf(stderr, "npu init: %s\n", err.c_str());
      return false;
    }
    kd = true;
    const std::uint32_t H = c.hidden, I = c.inter, qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    const std::uint32_t attn_n = 2 * qd + 2 * kvd, h_p = PadN(H);
    half_a = (I / 2) / 512 * 512;
    const std::string mk = std::to_string(H) + "x" + std::to_string(I);
    nm.kqkv = "insts_kdh_1024x" + std::to_string(H) + "x" + std::to_string(attn_n);
    nm.ko = "insts_kdh_1024x" + std::to_string(kValDim) + "x" + std::to_string(h_p);
    nm.swa = "insts_kdh_swi_a_" + mk + "_" + std::to_string(half_a);
    nm.swb = "insts_kdh_swi_b_" + mk + "_" + std::to_string(half_a);
    nm.dn2 = "insts_kdh_dn2_1024x" + std::to_string(I) + "x" + std::to_string(h_p);
    const std::string gdn_in = "insts_kdh_1024x" + std::to_string(H) + "x" + std::to_string(kGdnInN);
    {
      const std::string env = "source " + programs + "/../../gen/env.sh; ";
      const std::string gk = "bash " + programs + "/../kdh/gen_kdh.sh ";
      const auto need = [&](const std::string& file, const std::string& cmd) {
        std::ifstream f(programs + "/" + file);
        if (f.good()) return;
        std::fprintf(stderr, "gen: generating %s\n", file.c_str());
        if (std::system(("bash -c '" + env + cmd + "' >/dev/null").c_str()) != 0) {
          std::fprintf(stderr, "gen: failed to generate %s\n", file.c_str());
          std::exit(1);
        }
      };
      const auto n = [](std::uint32_t v) { return std::to_string(v); };
      const std::string base = "-M 1024 ";
      need(gdn_in + ".txt", gk + gdn_in + " " + n(I) + " " + base + "-K " + n(H) + " -N " + n(kGdnInN));
      need(nm.kqkv + ".txt", gk + nm.kqkv + " " + n(I) + " " + base + "-K " + n(H) + " -N " + n(attn_n));
      need(nm.ko + ".txt", gk + nm.ko + " " + n(I) + " " + base + "-K " + n(kValDim) + " -N " + n(h_p));
      need(nm.swa + ".txt", gk + nm.swa + " " + n(I) + " " + base + "-K " + n(H) + " -N " + n(2 * half_a) + " --mode swigu --noff 0");
      need(nm.swb + ".txt", gk + nm.swb + " " + n(I) + " " + base + "-K " + n(H) + " -N " + n(2 * (I - half_a)) + " --mode swigu --noff " + n(half_a / 64));
      need(nm.dn2 + ".txt", gk + nm.dn2 + " " + n(I) + " " + base + "-K " + n(I) + " -N " + n(h_p) + " --a2 1");
      for (std::uint32_t t : {64u, 128u, 256u, 512u, 1024u, 2048u})
        need("insts_kdh_gdn_" + n(t) + ".txt", gk + "insts_kdh_gdn_" + n(t) + " 7680 -M " + n(t) + " -K 64 -N 128 --mode gdn");
    }
    if (gdn_dev && !gd.Init(rk, programs)) gdn_dev = false;
    {
      std::uint32_t cc, rr;
      st.DimsG("token_embd.weight", &cc, &rr);
      c.vocab = rr;
      const auto* t = r.FindTensor("token_embd.weight");
      embed.resize(std::size_t{cc} * rr);
      const gufo::models::lfm3::Tensor tn{t->data, t->type, cc, rr, t->size_bytes};
#pragma omp parallel
      {
        std::vector<float> tmp(cc);
#pragma omp for
        for (std::int64_t row = 0; row < rr; ++row) {
          DecodeRowAny(tn, static_cast<std::uint32_t>(row), std::span<float>(tmp.data(), cc));
          for (std::uint32_t i = 0; i < cc; ++i) embed[std::size_t(row) * cc + i] = Bf16Bits(tmp[i]);
        }
      }
    }
    norm_w.resize(H);
    st.LoadG("output_norm.weight", norm_w.data(), H);
    {
      std::uint32_t cc, rr;
      st.DimsG("output.weight", &cc, &rr);
      const auto* t = r.FindTensor("output.weight");
      const gufo::models::lfm3::Tensor tn{t->data, t->type, cc, rr, t->size_bytes};
      n_codes = static_cast<std::uint32_t>(code_ids.size());
      readout.resize(std::size_t{n_codes} * H);
      for (std::uint32_t k = 0; k < n_codes; ++k) DecodeRowAny(tn, code_ids[k], std::span<float>(&readout[std::size_t{k} * H], H));
    }
    kxn = std::max({H, qd, I, kValDim});
    kyn = std::max<std::size_t>(kGdnInN, h_p);
    layers.resize(c.layers);
    hl.resize(c.layers);
    std::vector<float> buf;
    const auto t0 = Clock::now();
    for (std::uint32_t l = 0; l < c.layers; ++l) {
      Layer& L = layers[l];
      HL& h = hl[l];
      const std::string b = "blk." + std::to_string(l) + ".";
      h.linear = (l % interval) != interval - 1;
      L.attn_norm.resize(H);
      L.ffn_norm.resize(H);
      st.LoadG(b + "attn_norm.weight", L.attn_norm.data(), H);
      st.LoadG(b + "post_attention_norm.weight", L.ffn_norm.data(), H);
      if (h.linear) {
        h.conv.resize(std::size_t{kConvDim} * kConvK);
        st.LoadG(b + "ssm_conv1d.weight", h.conv.data(), h.conv.size());
        PermBlocks(h.conv, 2 * kKeyDim, kVD, kConvK);
        h.a_neg.resize(kVH);
        h.dt_bias.resize(kVH);
        {
          std::vector<float> ta(kVH), td(kVH);
          st.LoadG(b + "ssm_a", ta.data(), kVH);
          st.LoadG(b + "ssm_dt.bias", td.data(), kVH);
          for (std::uint32_t j = 0; j < kVH; ++j) {
            h.a_neg[HfHead(j)] = ta[j];
            h.dt_bias[HfHead(j)] = td[j];
          }
        }
        h.gnorm.resize(kVD);
        st.LoadG(b + "ssm_norm.weight", h.gnorm.data(), kVD);
        h.w_a.resize(std::size_t{kVH} * H);
        h.w_b.resize(std::size_t{kVH} * H);
        if (st.HasG(b + "ssm_beta_alpha.weight")) {
          std::vector<float> ba(std::size_t{2 * kVH} * H);
          st.LoadG(b + "ssm_beta_alpha.weight", ba.data(), ba.size());
          std::memcpy(h.w_b.data(), ba.data(), std::size_t{kVH} * H * 4);
          std::memcpy(h.w_a.data(), ba.data() + std::size_t{kVH} * H, std::size_t{kVH} * H * 4);
        } else {
          st.LoadG(b + "ssm_alpha.weight", h.w_a.data(), h.w_a.size());
          st.LoadG(b + "ssm_beta.weight", h.w_b.data(), h.w_b.size());
        }
        PermBlocks(h.w_a, 0, 1, H);
        PermBlocks(h.w_b, 0, 1, H);
        buf.resize(std::size_t{kGdnInN} * H);
        if (st.HasG(b + "attn_qkv_gate.weight")) {
          st.LoadG(b + "attn_qkv_gate.weight", buf.data(), buf.size());
        } else {
          st.LoadG(b + "attn_qkv.weight", buf.data(), std::size_t{kConvDim} * H);
          st.LoadG(b + "attn_gate.weight", buf.data() + std::size_t{kConvDim} * H, std::size_t{kValDim} * H);
        }
        {
          std::vector<float> vb(buf.begin() + std::size_t{2 * kKeyDim} * H, buf.begin() + std::size_t{kConvDim} * H);
          PermBlocks(vb, 0, kVD, H);
          std::memcpy(&buf[std::size_t{2 * kKeyDim} * H], vb.data(), vb.size() * 4);
          std::vector<float> zb(buf.begin() + std::size_t{kConvDim} * H, buf.end());
          PermBlocks(zb, 0, kVD, H);
          std::memcpy(&buf[std::size_t{kConvDim} * H], zb.data(), zb.size() * 4);
        }
        L.kp_qkv = rk.MakeProgram(H, kGdnInN, programs + "/" + gdn_in + ".txt", std::size_t{1024} * kGdnInN / 8 * 9, &err, 0, 64, 3, 5);
        L.kp_qkv2 = rk.MakeProgram(H, kGdnInN, programs + "/" + gdn_in + ".txt", std::size_t{1024} * kGdnInN / 8 * 9, &err, 0, 64, 3, 5);
        L.kw_qkv = rk.PackWeightAtb(buf.data(), kGdnInN, H, 4);
        buf.assign(std::size_t{h_p} * kValDim, 0.f);
        {
          std::vector<float> ow(std::size_t{H} * kValDim);
          st.LoadG(b + "ssm_out.weight", ow.data(), ow.size());
          PermCols(ow, H, kValDim, kVD);
          std::memcpy(buf.data(), ow.data(), ow.size() * 4);
        }
      } else {
        L.qn.resize(c.head_dim);
        L.kn.resize(c.head_dim);
        st.LoadG(b + "attn_q_norm.weight", L.qn.data(), c.head_dim);
        st.LoadG(b + "attn_k_norm.weight", L.kn.data(), c.head_dim);
        buf.resize(std::size_t{attn_n} * H);
        st.LoadG(b + "attn_q.weight", buf.data(), std::size_t{2 * qd} * H);
        st.LoadG(b + "attn_k.weight", buf.data() + std::size_t{2 * qd} * H, std::size_t{kvd} * H);
        st.LoadG(b + "attn_v.weight", buf.data() + std::size_t{2 * qd + kvd} * H, std::size_t{kvd} * H);
        L.kp_qkv = rk.MakeProgram(H, attn_n, programs + "/" + nm.kqkv + ".txt", std::size_t{1024} * attn_n / 8 * 9, &err, 0, 64, 3, 5);
        L.kp_qkv2 = rk.MakeProgram(H, attn_n, programs + "/" + nm.kqkv + ".txt", std::size_t{1024} * attn_n / 8 * 9, &err, 0, 64, 3, 5);
        L.kw_qkv = rk.PackWeightAtb(buf.data(), attn_n, H, 4);
        buf.assign(std::size_t{h_p} * qd, 0.f);
        st.LoadG(b + "attn_output.weight", buf.data(), std::size_t{H} * qd);
      }
      L.kp_o = rk.MakeProgram(kValDim, h_p, programs + "/" + nm.ko + ".txt", std::size_t{1024} * h_p / 8 * 9, &err, 0, 64, 3, 5);
      L.kp_o2 = rk.MakeProgram(kValDim, h_p, programs + "/" + nm.ko + ".txt", std::size_t{1024} * h_p / 8 * 9, &err, 0, 64, 3, 5);
      L.kw_o = rk.PackWeightAtb(buf.data(), h_p, kValDim, 4);
      KDispatch(L.kw_qkv, L.kp_qkv);
      KDispatch(L.kw_qkv, L.kp_qkv2);
      KDispatch(L.kw_o, L.kp_o);
      KDispatch(L.kw_o, L.kp_o2);
      PackKdMlpG(L, st, b, programs, &err);
      if (l % 8 == 0) std::fprintf(stderr, "hybrid: packed layer %u (%.1f s)\n", l, Ms(t0) / 1000.0);
    }
    return true;
  }

  // ---- delta-net layer ----
  void GdnHdrDev(const HL& h, std::uint32_t ne) {
    for (std::uint32_t set = 0; set < 2; ++set)
      for (std::uint32_t col = 0; col < 8; ++col) {
        auto* w = reinterpret_cast<std::uint16_t*>(gd.in_h[set] + std::size_t{col} * (ne / 4 + 1) * 9216);
        const std::uint32_t kh = 8 * set + col;
        for (std::uint32_t j = 0; j < kConvK; ++j)
          for (std::uint32_t i = 0; i < 128; ++i) {
            w[j * 128 + i] = Bf16Bits(h.conv[std::size_t{kh * 128 + i} * kConvK + j]);
            w[512 + j * 128 + i] = Bf16Bits(h.conv[std::size_t{kKeyDim + kh * 128 + i} * kConvK + j]);
            for (std::uint32_t hs = 0; hs < 2; ++hs)
              w[1024 + hs * 512 + j * 128 + i] = Bf16Bits(h.conv[std::size_t{2 * kKeyDim + (16 * set + 2 * col + hs) * 128 + i} * kConvK + j]);
          }
      }
  }

  void Gates(const HL& h, const float* z, std::uint32_t T, std::vector<float>* ab) {
    const std::uint32_t H = c.hidden;
    ab->resize(std::size_t{T} * 2 * kVH);
#pragma omp parallel for collapse(2)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::uint32_t j = 0; j < 2 * kVH; ++j) {
        const float* wrow = (j < kVH ? h.w_a.data() : h.w_b.data()) + std::size_t{j % kVH} * H;
        const float* zr = z + std::size_t(t) * H;
        float s = 0;
        for (std::uint32_t i = 0; i < H; ++i) s += wrow[i] * zr[i];
        (*ab)[std::size_t(t) * 2 * kVH + j] = s;
      }
  }

  void GdnPrepDev(const HL& h, const float* proj, const float* z, std::uint32_t T, std::uint32_t tok_off, std::uint32_t ne) {
    std::vector<float> ab;
    Gates(h, z, T, &ab);
#pragma omp parallel for collapse(2)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::uint32_t v = 0; v < kVH; ++v) {
        const std::uint32_t kh = v / (kVH / kKH);
        const float* row = proj + std::size_t(t) * kGdnInN;
        const std::uint32_t g = tok_off + static_cast<std::uint32_t>(t), set = v / 16, hh = v % 16, col = hh / 2, hs = hh % 2;
        std::uint8_t* rec = gd.in_h[set] + (std::size_t{col} * (ne / 4 + 1) + 1 + g / 4) * 9216 + ((g % 4) * 2 + hs) * 1024;
        auto* b = reinterpret_cast<std::uint16_t*>(rec);
        for (std::uint32_t i = 0; i < kKD; ++i) {
          b[i] = Bf16Bits(row[kh * kKD + i]);
          b[128 + i] = Bf16Bits(row[kKeyDim + kh * kKD + i]);
          b[256 + i] = Bf16Bits(row[2 * kKeyDim + v * kVD + i]);
        }
        const float av = ab[std::size_t(t) * 2 * kVH + v], bv = ab[std::size_t(t) * 2 * kVH + kVH + v];
        const float x = av + h.dt_bias[v];
        const float sp = x > 20.f ? x : std::log1p(std::exp(x));
        const float sc[2] = {std::exp(h.a_neg[v] * sp), 1.0f / (1.0f + std::exp(-bv))};
        std::memcpy(rec + 768, sc, 8);
        const std::int32_t reset = t == 0 ? 1 : 0;
        std::memcpy(rec + 776, &reset, 4);
      }
  }

  void GdnPadDev(std::uint32_t from, std::uint32_t to, std::uint32_t ne) {
    for (std::uint32_t g = from; g < to; ++g)
      for (std::uint32_t v = 0; v < kVH; ++v) {
        const std::uint32_t set = v / 16, hh = v % 16, col = hh / 2, hs = hh % 2;
        std::uint8_t* rec = gd.in_h[set] + (std::size_t{col} * (ne / 4 + 1) + 1 + g / 4) * 9216 + ((g % 4) * 2 + hs) * 1024;
        std::memset(rec, 0, 1024);
        const float sc[2] = {1.0f, 0.0f};
        std::memcpy(rec + 768, sc, 8);
      }
  }

  void GdnPostDev(const HL& h, const float* proj, std::uint32_t T, std::uint32_t tok_off, std::uint32_t ne, float* gated) {
#pragma omp parallel for collapse(2)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::uint32_t v = 0; v < kVH; ++v) {
        const std::uint32_t g = tok_off + static_cast<std::uint32_t>(t), set = v / 16, hh = v % 16, col = hh / 2, hs = hh % 2;
        float o[kVD];
        for (std::uint32_t half = 0; half < 2; ++half) {
          const std::uint32_t row = hs * 2 + half;
          const std::uint8_t* src = gd.out_h[set] + (std::size_t{col} * (ne / 64) + g / 64) * 73728 + std::size_t{row} * 18432 + std::size_t{g % 64} * 256;
          std::memcpy(o + half * 64, src, 256);
        }
        float ss = 0;
        for (std::uint32_t j = 0; j < kVD; ++j) ss += o[j] * o[j];
        const float rs = 1.0f / std::sqrt(ss / kVD + c.eps);
        const float* zz = &proj[std::size_t(t) * kGdnInN + kConvDim + v * kVD];
        float* gt = &gated[std::size_t(t) * kValDim + v * kVD];
        for (std::uint32_t j = 0; j < kVD; ++j) gt[j] = h.gnorm[j] * (o[j] * rs) * SiluF(zz[j]);
      }
  }

  // Host reference of the whole delta-net stage (conv, norms, recurrence, gated norm), fp32 state.
  void GdnHost(const HL& h, const float* proj, const float* z, std::uint32_t T, float* gated) {
    std::vector<float> ab;
    Gates(h, z, T, &ab);
    std::vector<float> qkv(std::size_t{T} * kConvDim);
#pragma omp parallel for
    for (std::int64_t t = 0; t < T; ++t) {
      float* o = &qkv[std::size_t(t) * kConvDim];
      for (std::uint32_t ch = 0; ch < kConvDim; ++ch) {
        float s = 0;
        for (std::uint32_t j = 0; j < kConvK; ++j) {
          const std::int64_t src = t - (kConvK - 1) + j;
          if (src >= 0) s += h.conv[std::size_t{ch} * kConvK + j] * proj[std::size_t(src) * kGdnInN + ch];
        }
        o[ch] = SiluF(s);
      }
    }
#pragma omp parallel for
    for (std::int64_t v = 0; v < kVH; ++v) {
      const std::uint32_t kh = static_cast<std::uint32_t>(v) / (kVH / kKH);
      std::vector<float> S(std::size_t{kKD} * kVD, 0.f), q(kKD), k(kKD), kv(kVD), o(kVD);
      for (std::uint32_t t = 0; t < T; ++t) {
        const float* row = &qkv[std::size_t{t} * kConvDim];
        std::memcpy(q.data(), row + kh * kKD, kKD * 4);
        std::memcpy(k.data(), row + kKeyDim + kh * kKD, kKD * 4);
        const float* vv = row + 2 * kKeyDim + v * kVD;
        float qn = 0, kn = 0;
        for (std::uint32_t i = 0; i < kKD; ++i) {
          qn += q[i] * q[i];
          kn += k[i] * k[i];
        }
        const float qs = 1.0f / std::sqrt(qn + 1e-6f) / std::sqrt(static_cast<float>(kKD)), ks = 1.0f / std::sqrt(kn + 1e-6f);
        for (std::uint32_t i = 0; i < kKD; ++i) {
          q[i] *= qs;
          k[i] *= ks;
        }
        const float x = ab[std::size_t{t} * 2 * kVH + v] + h.dt_bias[v];
        const float sp = x > 20.f ? x : std::log1p(std::exp(x));
        const float decay = std::exp(h.a_neg[v] * sp);
        const float beta = 1.0f / (1.0f + std::exp(-ab[std::size_t{t} * 2 * kVH + kVH + v]));
        std::fill(kv.begin(), kv.end(), 0.f);
        for (std::uint32_t i = 0; i < kKD; ++i) {
          float* Si = &S[std::size_t{i} * kVD];
          for (std::uint32_t j = 0; j < kVD; ++j) {
            Si[j] *= decay;
            kv[j] += Si[j] * k[i];
          }
        }
        for (std::uint32_t j = 0; j < kVD; ++j) kv[j] = (vv[j] - kv[j]) * beta;
        std::fill(o.begin(), o.end(), 0.f);
        for (std::uint32_t i = 0; i < kKD; ++i) {
          float* Si = &S[std::size_t{i} * kVD];
          for (std::uint32_t j = 0; j < kVD; ++j) {
            Si[j] += k[i] * kv[j];
            o[j] += Si[j] * q[i];
          }
        }
        float ss = 0;
        for (std::uint32_t j = 0; j < kVD; ++j) ss += o[j] * o[j];
        const float rs = 1.0f / std::sqrt(ss / kVD + c.eps);
        const float* zz = &proj[std::size_t{t} * kGdnInN + kConvDim + v * kVD];
        float* gt = &gated[std::size_t{t} * kValDim + v * kVD];
        for (std::uint32_t j = 0; j < kVD; ++j) gt[j] = h.gnorm[j] * (o[j] * rs) * SiluF(zz[j]);
      }
    }
  }

  using Seg = std::pair<std::uint32_t, std::uint32_t>;

  void GdnLayer(Layer& L, const HL& h, const float* z, std::uint32_t Tt, const std::vector<Seg>& segs, float* out_h) {
    const std::uint32_t H = c.hidden, h_p = PadN(H);
    s_proj.resize(std::size_t{Tt} * kGdnInN);
    s_gated.resize(std::size_t{Tt} * kValDim);
    KGemm(L.kp_qkv, L.kp_qkv2, L.kw_qkv, z, Tt, H, s_proj.data());
    const std::uint32_t ne = GdnDev::Bucket(Tt);
    if (gdn_dev && ne) {
      GdnHdrDev(h, ne);
      for (const auto& sg : segs) GdnPrepDev(h, s_proj.data() + std::size_t{sg.first} * kGdnInN, z + std::size_t{sg.first} * H, sg.second, sg.first, ne);
      GdnPadDev(Tt, ne, ne);
      gd.Run(0, ne);
      gd.Run(1, ne);
      for (const auto& sg : segs) GdnPostDev(h, s_proj.data() + std::size_t{sg.first} * kGdnInN, sg.second, sg.first, ne, s_gated.data() + std::size_t{sg.first} * kValDim);
    } else {
      for (const auto& sg : segs) GdnHost(h, s_proj.data() + std::size_t{sg.first} * kGdnInN, z + std::size_t{sg.first} * H, sg.second, s_gated.data() + std::size_t{sg.first} * kValDim);
    }
    s_outp.resize(std::size_t{Tt} * h_p);
    KGemm(L.kp_o, L.kp_o2, L.kw_o, s_gated.data(), Tt, kValDim, s_outp.data());
    for (std::uint32_t t = 0; t < Tt; ++t) std::memcpy(out_h + std::size_t{t} * H, &s_outp[std::size_t{t} * h_p], std::size_t{H} * 4);
  }

  // ---- gated, non-causal full attention (head_dim 256, partial rope) on the host ----
  void AttnHost(const Layer& L, const float* proj, std::uint32_t T, float* att) {
    const std::uint32_t qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim, hd = c.head_dim, attn_n = 2 * qd + 2 * kvd;
    std::vector<float> q(std::size_t{T} * qd), gate(std::size_t{T} * qd), k(std::size_t{T} * kvd), v(std::size_t{T} * kvd);
    std::vector<float> cs(std::size_t{T} * (kRotDim / 2)), sn(std::size_t{T} * (kRotDim / 2));
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::uint32_t i = 0; i < kRotDim / 2; ++i) {
        const double ang = static_cast<double>(t) * std::pow(static_cast<double>(c.theta), -2.0 * i / kRotDim);
        cs[std::size_t{t} * (kRotDim / 2) + i] = static_cast<float>(std::cos(ang));
        sn[std::size_t{t} * (kRotDim / 2) + i] = static_cast<float>(std::sin(ang));
      }
    auto norm_rope = [&](float* x, const float* nw, std::uint32_t t) {
      float ss = 0;
      for (std::uint32_t i = 0; i < hd; ++i) ss += x[i] * x[i];
      const float rs = 1.0f / std::sqrt(ss / hd + c.eps);
      for (std::uint32_t i = 0; i < hd; ++i) x[i] *= rs * nw[i];
      for (std::uint32_t i = 0; i < kRotDim / 2; ++i) {
        const float cc = cs[std::size_t{t} * (kRotDim / 2) + i], ss2 = sn[std::size_t{t} * (kRotDim / 2) + i];
        const float a = x[i], b = x[i + kRotDim / 2];
        x[i] = a * cc - b * ss2;
        x[i + kRotDim / 2] = b * cc + a * ss2;
      }
    };
#pragma omp parallel for
    for (std::int64_t t = 0; t < T; ++t) {
      const float* row = &proj[std::size_t(t) * attn_n];
      for (std::uint32_t h = 0; h < c.heads; ++h) {
        float* qh = &q[std::size_t(t) * qd + h * hd];
        std::memcpy(qh, row + h * 2 * hd, hd * 4);
        std::memcpy(&gate[std::size_t(t) * qd + h * hd], row + h * 2 * hd + hd, hd * 4);
        norm_rope(qh, L.qn.data(), static_cast<std::uint32_t>(t));
      }
      for (std::uint32_t h = 0; h < c.kv_heads; ++h) {
        float* kh = &k[std::size_t(t) * kvd + h * hd];
        std::memcpy(kh, row + 2 * qd + h * hd, hd * 4);
        norm_rope(kh, L.kn.data(), static_cast<std::uint32_t>(t));
        std::memcpy(&v[std::size_t(t) * kvd + h * hd], row + 2 * qd + kvd + h * hd, hd * 4);
      }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const std::uint32_t grp = c.heads / c.kv_heads;
#pragma omp parallel
    {
      std::vector<float> sc(T);
#pragma omp for collapse(2) schedule(dynamic, 4)
      for (std::int64_t t = 0; t < T; ++t)
        for (std::uint32_t h = 0; h < c.heads; ++h) {
          const float* qh = &q[std::size_t(t) * qd + h * hd];
          const std::uint32_t kh = h / grp;
          float mx2 = -INFINITY;
          for (std::uint32_t u = 0; u < T; ++u) {
            const float* kk = &k[std::size_t{u} * kvd + kh * hd];
            float s = 0;
            for (std::uint32_t i = 0; i < hd; ++i) s += qh[i] * kk[i];
            s *= scale;
            sc[u] = s;
            mx2 = std::max(mx2, s);
          }
          float den = 0;
          for (std::uint32_t u = 0; u < T; ++u) {
            sc[u] = std::exp(sc[u] - mx2);
            den += sc[u];
          }
          float* o = &att[std::size_t(t) * qd + h * hd];
          std::fill(o, o + hd, 0.f);
          for (std::uint32_t u = 0; u < T; ++u) {
            const float wgt = sc[u] / den;
            const float* vv = &v[std::size_t{u} * kvd + kh * hd];
            for (std::uint32_t i = 0; i < hd; ++i) o[i] += wgt * vv[i];
          }
          const float* gg = &gate[std::size_t(t) * qd + h * hd];
          for (std::uint32_t i = 0; i < hd; ++i) o[i] *= 1.0f / (1.0f + std::exp(-gg[i]));
        }
    }
  }

  void AttnLayer(Layer& L, const float* z, std::uint32_t Tt, const std::vector<Seg>& segs, float* out_h) {
    const std::uint32_t H = c.hidden, h_p = PadN(H), qd = c.heads * c.head_dim, attn_n = 2 * qd + 2 * c.kv_heads * c.head_dim;
    s_proj.resize(std::size_t{Tt} * attn_n);
    s_att.resize(std::size_t{Tt} * qd);
    KGemm(L.kp_qkv, L.kp_qkv2, L.kw_qkv, z, Tt, H, s_proj.data());
    for (const auto& sg : segs) AttnHost(L, s_proj.data() + std::size_t{sg.first} * attn_n, sg.second, s_att.data() + std::size_t{sg.first} * qd);
    s_outp.resize(std::size_t{Tt} * h_p);
    KGemm(L.kp_o, L.kp_o2, L.kw_o, s_att.data(), Tt, qd, s_outp.data());
    for (std::uint32_t t = 0; t < Tt; ++t) std::memcpy(out_h + std::size_t{t} * H, &s_outp[std::size_t{t} * h_p], std::size_t{H} * 4);
  }

  struct Seq {
    const std::int32_t* tok;
    std::uint32_t len, marker;
  };

  // Layer-major forward over a batch of sequences (rows of all sequences share the GEMM chunks). Returns the final-norm hidden
  // state at each sequence's marker.
  void ForwardBatch(const std::vector<Seq>& seqs, std::vector<float>* hidden) {
    const std::uint32_t H = c.hidden;
    std::vector<Seg> segs;
    std::uint32_t T = 0;
    for (const auto& q : seqs) {
      segs.push_back({T, q.len});
      T += q.len;
    }
    std::vector<float> x(std::size_t{T} * H), z(std::size_t{T} * H), br(std::size_t{T} * H);
    for (std::size_t i = 0; i < seqs.size(); ++i)
#pragma omp parallel for
      for (std::int64_t t = 0; t < seqs[i].len; ++t)
        for (std::uint32_t k = 0; k < H; ++k) x[std::size_t(segs[i].first + t) * H + k] = Bf(embed[std::size_t(seqs[i].tok[t]) * H + k]);
    auto norm_rows = [&](const float* w) {
#pragma omp parallel for
      for (std::int64_t r = 0; r < T; ++r) RmsNorm(&x[std::size_t(r) * H], w, &z[std::size_t(r) * H], H, c.eps);
    };
    for (std::uint32_t l = 0; l < c.layers; ++l) {
      Layer& L = layers[l];
      norm_rows(L.attn_norm.data());
      if (hl[l].linear) GdnLayer(L, hl[l], z.data(), T, segs, br.data());
      else AttnLayer(L, z.data(), T, segs, br.data());
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(x.size()); ++i) x[i] += br[i];
      norm_rows(L.ffn_norm.data());
      KMlp(L, z.data(), T, br.data());
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(x.size()); ++i) x[i] += br[i];
    }
    hidden->resize(seqs.size() * H);
    for (std::size_t i = 0; i < seqs.size(); ++i)
      RmsNorm(&x[std::size_t{segs[i].first + seqs[i].marker} * H], norm_w.data(), &(*hidden)[i * H], H, c.eps);
  }

  void Logits(const float* hid, std::uint32_t count, float* out) const {
    const std::uint32_t H = c.hidden;
    for (std::uint32_t k = 0; k < count && k < n_codes; ++k) {
      float s = 0;
      for (std::uint32_t i = 0; i < H; ++i) s += readout[std::size_t{k} * H + i] * hid[i];
      out[k] = s;
    }
  }
};

// Batched decision forward: `decide-eval MODEL.gguf PROGRAMS EVAL.bin CODE_IDS.txt` (EVAL.bin: count, then len marker n_options tokens...).
int DecideEval(const char* model, const char* programs, const char* path, const char* ids_path) {
  std::vector<std::uint32_t> ids;
  {
    std::ifstream f(ids_path);
    std::uint32_t v;
    while (f >> v) ids.push_back(v);
  }
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::vector<std::int32_t> d(b.size() / 4);
  std::memcpy(d.data(), b.data(), d.size() * 4);
  HybridEngine e;
  if (!e.InitH(model, programs, 4096, ids)) return 1;
  struct Item { const std::int32_t* tok; std::uint32_t len, marker, cnt; };
  std::vector<Item> items;
  std::size_t at = 0;
  const std::int32_t cases = d[at++];
  for (std::int32_t k = 0; k < cases; ++k) {
    Item it{nullptr, static_cast<std::uint32_t>(d[at]), static_cast<std::uint32_t>(d[at + 1]), static_cast<std::uint32_t>(d[at + 2])};
    at += 3;
    it.tok = &d[at];
    at += it.len;
    items.push_back(it);
  }
  const std::uint32_t group = std::getenv("GEN_GROUP") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GEN_GROUP"))) : 12;
  const int reps = std::getenv("GEN_REPEAT") ? std::atoi(std::getenv("GEN_REPEAT")) : 1;
  std::uint64_t total = 0;
  Clock::time_point t0;
  for (int rep = 0; rep < reps; ++rep) {
    if (rep == reps - 1) {
      t0 = Clock::now();
      total = 0;
    }
    for (std::size_t g = 0; g < items.size(); g += group) {
      std::vector<HybridEngine::Seq> sq;
      for (std::size_t i = g; i < std::min(items.size(), g + group); ++i) {
        sq.push_back({items[i].tok, items[i].len, items[i].marker});
        total += items[i].len;
      }
      std::vector<float> hid;
      e.ForwardBatch(sq, &hid);
      if (rep == reps - 1)
        for (std::size_t i = 0; i < sq.size(); ++i) {
          std::vector<float> lg(items[g + i].cnt);
          e.Logits(&hid[i * e.c.hidden], items[g + i].cnt, lg.data());
          for (float v : lg) std::printf("%.5f ", v);
          std::printf("\n");
        }
    }
  }
  std::fprintf(stderr, "decide: %zu cases %llu tokens %.1f ms (%.1f ms/case, %.0f tok/s) group %u\n", items.size(), static_cast<unsigned long long>(total), Ms(t0), Ms(t0) / items.size(), total / Ms(t0) * 1000, group);
  return 0;
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
  std::fprintf(stderr, "\nmlp host enc+dec %.0f ms | mlp device %.0f ms\n", g.ph[6], g.ph[7]);
  std::fprintf(stderr, "\nattention parts: rope+dup+convert %.0f ms | AttnAppend %.0f ms | prefill-attention run %.0f ms\n", g.ph[3], g.ph[5], g.ph[4]);
  std::fprintf(stderr, "\nhost norm+resid %.0f ms | ffn %.0f ms | attention+rope %.0f ms | gemm enc %.0f dec %.0f sync+wait %.0f ms\n", g.ph[0], g.ph[1], g.ph[2], g.tim.enc, g.tim.dec, g.tim.wait + g.tim.syncA + g.tim.syncC);
  std::fprintf(stderr, "\nprefill %zu tok %.1f ms (%.0f tok/s); decode %u tok %.1f ms (%.2f tok/s)\n", ids.size(), pf, ids.size() / pf * 1000, produced, dc, produced / dc * 1000);
  return 0;
}

int Interactive(const char* model, const char* programs) {
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
  const std::uint32_t ctx = std::getenv("GEN_CTX") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GEN_CTX"))) : 8192;
  const std::uint32_t max_new = std::getenv("GEN_MAXNEW") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GEN_MAXNEW"))) : 512;
  std::vector<std::uint32_t> stop;
  if (auto eo = reader->GetMetadataUint32("tokenizer.ggml.eos_token_id")) stop.push_back(*eo);
  for (const char* sp : {"<|im_end|>", "<|endoftext|>"}) {
    const auto v = tok->Encode(sp);
    if (v.size() == 1) stop.push_back(v[0]);
  }
  const bool nothink = std::getenv("GEN_THINK") == nullptr && tok->Encode("<think>").size() == 1;
  bool bos = false;
  if (const auto tmpl = reader->GetMetadataString("tokenizer.chat_template"))
    bos = tmpl->rfind("{{- bos_token", 0) == 0 || tmpl->rfind("{{ bos_token", 0) == 0;
  Engine g;
  if (!g.Init(model, programs, ctx)) return 1;
  std::vector<float> hd;
  auto feed = [&](const std::vector<std::uint32_t>& ids) {
    for (std::uint32_t o = 0, m = 0; o < ids.size(); o += m) {
      m = g.Chunk(static_cast<std::uint32_t>(ids.size()) - o);
      g.Step(&ids[o], m, &hd);
    }
  };
  std::fprintf(stderr, "ready. type a message and press Enter; empty line or Ctrl-D quits. /reset starts a new conversation. (the NPU stays locked while this runs)\n");
  bool first = true;
  std::string line;
  for (;;) {
    std::fprintf(stderr, "\n> ");
    if (!std::getline(std::cin, line) || line.empty()) break;
    if (line == "/reset") {
      g.Reset();
      first = true;
      std::fprintf(stderr, "(conversation reset)\n");
      continue;
    }
    std::vector<std::uint32_t> ids;
    if (first && bos)
      if (auto b = reader->GetMetadataUint32("tokenizer.ggml.bos_token_id")) ids.push_back(*b);
    std::string text = "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n";
    if (nothink) text += "<think>\n\n</think>\n\n";
    for (auto t : tok->Encode(text)) ids.push_back(t);
    first = false;
    if (g.pos + ids.size() + max_new + 8 > ctx) {
      std::fprintf(stderr, "(context full, resetting conversation)\n");
      g.Reset();
      ids.clear();
      if (bos)
        if (auto b = reader->GetMetadataUint32("tokenizer.ggml.bos_token_id")) ids.push_back(*b);
      for (auto t : tok->Encode(text)) ids.push_back(t);
    }
    const auto t0 = Clock::now();
    feed(ids);
    std::uint32_t next = g.Argmax(&hd[hd.size() - g.c.hidden]);
    const double pf = Ms(t0);
    const auto t1 = Clock::now();
    std::uint32_t produced = 0;
    for (std::uint32_t i = 0; i < max_new && g.pos + 2 < ctx; ++i) {
      if (std::find(stop.begin(), stop.end(), next) != stop.end()) break;
      ++produced;
      std::fputs(tok->DecodeTokenCopy(next).c_str(), stdout);
      std::fflush(stdout);
      g.Step(&next, 1, &hd);
      next = g.Argmax(hd.data());
    }
    const double dc = Ms(t1);
    std::printf("\n");
    feed(tok->Encode("<|im_end|>\n"));
    std::fprintf(stderr, "[prefill %zu tok %.0f ms (%.0f tok/s) | decode %u tok %.2f tok/s | context %u]\n", ids.size(), pf, ids.size() / pf * 1000, produced, produced / dc * 1000, g.pos);
  }
  return 0;
}

std::string UnescapeJev(std::string_view text) {
  std::string out;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '\\' || i + 1 == text.size()) {
      out.push_back(text[i]);
      continue;
    }
    const char next = text[++i];
    out.push_back(next == 'n' ? '\n' : next == 't' ? '\t' : next);
  }
  return out;
}

// jev judge protocol: prints READY, then one verdict line per request line "question<TAB>context" (backslash escapes).
int JudgeStdio(const char* model, const char* programs) {
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
  const auto yes_ids = tok->Encode("Yes"), no_ids = tok->Encode("No");
  if (yes_ids.empty() || no_ids.empty()) return 1;
  std::string sys = "Judge whether the statement or question is true or correct. Answer with exactly one word: Yes or No.";
  if (const char* env = std::getenv("GUFO_JEV_SYS")) sys = env;
  std::vector<std::uint32_t> bos;
  if (const auto tmpl = reader->GetMetadataString("tokenizer.chat_template"))
    if (tmpl->rfind("{{- bos_token", 0) == 0 || tmpl->rfind("{{ bos_token", 0) == 0)
      if (auto b = reader->GetMetadataUint32("tokenizer.ggml.bos_token_id")) bos.push_back(*b);
  const std::uint32_t ctx = 4096;
  Engine g;
  if (!g.Init(model, programs, ctx)) return 1;
  std::vector<std::uint32_t> prefix = bos;
  for (auto t : tok->Encode("<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n")) prefix.push_back(t);
  std::vector<float> hd;
  auto feed = [&](const std::uint32_t* ids, std::uint32_t n) {
    for (std::uint32_t o = 0, m = 0; o < n; o += m) {
      m = g.Chunk(n - o);
      g.Step(ids + o, m, &hd);
    }
  };
  const std::uint32_t p0 = static_cast<std::uint32_t>(prefix.size());
  g.Reset();
  feed(prefix.data(), p0);
  std::printf("READY\n");
  std::fflush(stdout);
  std::string line;
  while (std::getline(std::cin, line)) {
    const auto tab = line.find('\t');
    const std::string question = UnescapeJev(std::string_view(line).substr(0, tab));
    const std::string context = tab == std::string::npos ? std::string() : UnescapeJev(std::string_view(line).substr(tab + 1));
    const auto t0 = Clock::now();
    std::string body;
    if (!context.empty()) body += context + "\n\n";
    body += "Question: " + question + "\nAnswer with Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    auto ids = tok->Encode(body);
    if (ids.size() + p0 + 8 > ctx) ids.erase(ids.begin(), ids.end() - (ctx - p0 - 8));
    g.pos = p0;
    for (auto& L : g.layers) L.napp = std::min(L.napp, p0);
    feed(ids.data(), static_cast<std::uint32_t>(ids.size()));
    const float* last = &hd[hd.size() - g.c.hidden];
    const float ly = g.LogitAt(last, yes_ids[0]), ln = g.LogitAt(last, no_ids[0]);
    const double m = std::max(ly, ln);
    const double ey = std::exp(ly - m), en = std::exp(ln - m);
    std::printf("%s %.3f %.3f %.1f\n", ly >= ln ? "Yes" : "No", ey / (ey + en), en / (ey + en), Ms(t0));
    std::fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 6 && std::string(argv[1]) == "decide-eval") return DecideEval(argv[2], argv[3], argv[4], argv[5]);
  if (argc >= 4 && std::string(argv[1]) == "judge-stdio") return JudgeStdio(argv[2], argv[3]);
  if (argc >= 4 && std::string(argv[1]) == "interactive") return Interactive(argv[2], argv[3]);
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
