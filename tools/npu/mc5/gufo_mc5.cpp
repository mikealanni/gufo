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

struct Layer {
  std::vector<float> attn_norm, ffn_norm;
  WeightBuffer qkv, out, ffn, ffn_dn;
  Program p_qkv, p_out, p_ffn, p_qkv64, p_out64, p_gu64, p_dn64;
  std::vector<float> k, v;
  std::vector<float> wqkv, wo, wgate, wup, wdown;
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
  double t_npu{0}, t_cpu{0};
  bool cpu_ref{std::getenv("MC5_CPU") != nullptr || std::getenv("MC5_CPU_OP") != nullptr};
  std::string cpu_ops{std::getenv("MC5_CPU") ? "qkv,out,ffn" : (std::getenv("MC5_CPU_OP") ? std::getenv("MC5_CPU_OP") : "")};
  bool Cpu(const char* op) const { return cpu_ops.find(op) != std::string::npos; }

  bool Init(const std::string& model_dir, const std::string& programs, std::uint32_t context) {
    ctx = context;
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
      L.qkv = rt.PackWeight(buf.data(), qkv_p, H);
      buf.assign(std::size_t{h_p} * qd, 0.f);
      st.Load(pre + "self_attn.o_proj.weight", buf.data(), std::size_t{H} * qd);
      if (cpu_ref) L.wo.assign(buf.begin(), buf.begin() + std::size_t{H} * qd);
      L.out = rt.PackWeight(buf.data(), h_p, qd);
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
      PackMatInt5(buf.data(), 2 * I, H, m.host);
      buf.assign(std::size_t{h_p} * I, 0.f);
      st.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{H} * I);
      if (cpu_ref) L.wdown.assign(buf.begin(), buf.begin() + std::size_t{H} * I);
      PackMatInt5(buf.data(), h_p, I, m.host + gu_bytes);
      if (m64) L.ffn_dn = rt.PackWeight(buf.data(), h_p, I);
      m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      L.ffn = std::move(m);
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

  void Reset() { pos = 0; }

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
      if (Cpu("qkv")) MatVec(L.wqkv.data(), qkv_n, H, z.data(), n, qkv.data());
      else rt.Gemm(u64 ? L.p_qkv64 : L.p_qkv, L.qkv, z.data(), n, qkv.data(), &tim);
      t_npu += Ms(t0);
      t0 = Clock::now();
      for (std::uint32_t r = 0; r < n; ++r) {
        float* row = &qkv[std::size_t{r} * (Cpu("qkv") ? qkv_n : qkv_p)];
        Rope(row, c.heads, pos + r);
        Rope(row + qd, c.kv_heads, pos + r);
        std::memcpy(&L.k[std::size_t{pos + r} * kvd], row + qd, kvd * 4);
        std::memcpy(&L.v[std::size_t{pos + r} * kvd], row + qd + kvd, kvd * 4);
        Attend(L, row, pos + r, &att[std::size_t{r} * qd]);
      }
      t_cpu += Ms(t0);
      t0 = Clock::now();
      if (Cpu("out")) MatVec(L.wo.data(), H, qd, att.data(), n, brh.data());
      else {
        rt.Gemm(u64 ? L.p_out64 : L.p_out, L.out, att.data(), n, br.data(), &tim);
        for (std::uint32_t r = 0; r < n; ++r) std::memcpy(&brh[std::size_t{r} * H], &br[std::size_t{r} * h_p], H * 4);
      }
      t_npu += Ms(t0);
      for (std::size_t i = 0; i < std::size_t{n} * H; ++i) x[i] += brh[i];
      for (std::uint32_t r = 0; r < n; ++r) RmsNorm(&x[std::size_t{r} * H], L.ffn_norm.data(), &z[std::size_t{r} * H], H, c.eps);
      t0 = Clock::now();
      if (Cpu("ffn")) {
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
      for (std::uint32_t o = 0; o < len; o += kPrefillRows) {
        const std::uint32_t m = std::min(kPrefillRows, len - o);
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
    for (std::uint32_t o = 0; o < n_tok; o += kPrefillRows) {
      const std::uint32_t m = std::min(kPrefillRows, n_tok - o);
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
    for (std::uint32_t o = 0; o < len; o += kPrefillRows) {
      const std::uint32_t m = std::min(kPrefillRows, len - o);
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
    for (std::uint32_t o = 0; o < n_tok; o += kPrefillRows) {
      const std::uint32_t m = std::min(kPrefillRows, n_tok - o);
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
