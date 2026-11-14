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

double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

constexpr std::uint32_t kH = 2560, kI = 9216, kLayers = 32, kHeads = 16, kKvHeads = 4, kHeadDim = 256;
constexpr std::uint32_t kKH = 16, kVH = 32, kKD = 128, kVD = 128, kConvK = 4, kRot = 64;
constexpr std::uint32_t kKeyDim = kKH * kKD, kValDim = kVH * kVD, kConvDim = 2 * kKeyDim + kValDim;
constexpr std::uint32_t kQDim = kHeads * kHeadDim, kKvDim = kKvHeads * kHeadDim;
constexpr std::uint32_t kRows = 64;
constexpr std::uint32_t kPassCols = 1024;
constexpr std::uint32_t PadN(std::uint32_t n) { return (n + kPassCols - 1) / kPassCols * kPassCols; }
constexpr float kEps = 1e-6f, kTheta = 1e7f;
constexpr std::uint32_t kOutN = PadN(kH);
constexpr std::uint32_t kGdnInN = kConvDim + kValDim;
constexpr std::uint32_t kAttnInN = 2 * kQDim + 2 * kKvDim;
constexpr std::uint32_t kReadoutRows = 255;

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
    const std::string key = "\"" + name + "\":";
    const auto at = header.find(key);
    if (at == std::string::npos) return nullptr;
    const auto off = header.find("\"data_offsets\"", at);
    const auto lb = header.find('[', off);
    std::size_t a = 0, b = 0;
    std::sscanf(header.c_str() + lb, "[%zu,%zu]", &a, &b);
    *count = (b - a) / 2;
    return reinterpret_cast<const std::uint16_t*>(static_cast<const char*>(map) + data0 + a);
  }
};

struct Weights {
  Safetensors s[2];
  const std::uint16_t* Raw(const std::string& name, std::size_t expect) const {
    for (const auto& f : s) {
      std::size_t cnt = 0;
      const auto* p = f.Find(name, &cnt);
      if (p) {
        if (cnt != expect) {
          std::fprintf(stderr, "tensor %s size %zu expected %zu\n", name.c_str(), cnt, expect);
          std::exit(1);
        }
        return p;
      }
    }
    std::fprintf(stderr, "tensor %s missing\n", name.c_str());
    std::exit(1);
  }
  void Load(const std::string& name, float* dst, std::size_t expect) const {
    const auto* p = Raw(name, expect);
#pragma omp parallel for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(expect); ++i) dst[i] = Bf(p[i]);
  }
  std::vector<float> Vec(const std::string& name, std::size_t n, float add = 0.f) const {
    std::vector<float> v(n);
    Load(name, v.data(), n);
    for (auto& x : v) x += add;
    return v;
  }
};

struct QMat {
  std::vector<std::int8_t> q;
  std::vector<std::uint16_t> s;
};

void RmsNormRows(const float* x, const float* w, float* out, std::uint32_t rows, std::uint32_t d) {
#pragma omp parallel for
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + std::size_t(r) * d;
    float sum = 0;
    for (std::uint32_t i = 0; i < d; ++i) sum += xr[i] * xr[i];
    const float scale = 1.0f / std::sqrt(sum / d + kEps);
    float* o = out + std::size_t(r) * d;
    for (std::uint32_t i = 0; i < d; ++i) o[i] = xr[i] * scale * w[i];
  }
}

float Silu(float x) { return x / (1.0f + std::exp(-x)); }

struct Layer {
  bool linear{false};
  std::vector<float> ln1, ln2;
  // gated delta net
  std::vector<float> conv, a_log, dt_bias, gnorm, w_a, w_b;
  // attention
  std::vector<float> qn, kn;
  WeightBuffer w_in, w_out, w_gu, w_dn;
  Program p_in, p_out, p_gu, p_dn;
};

struct Engine {
  NpuRuntime rt;
  NpuRuntime::Timings tim{};
  std::vector<Layer> layers;
  std::vector<float> norm_w, readout;
  std::vector<std::uint16_t> embed_raw;
  const std::uint16_t* embed{nullptr};
  Weights w;
  double t_npu{0}, t_host{0};
  const char* qdir{std::getenv("Q35_QDIR")};

  bool Init(const std::string& model_dir, const std::string& programs) {
    if (!w.s[0].Open(model_dir + "/model-00001-of-00002.safetensors") || !w.s[1].Open(model_dir + "/model-00002-of-00002.safetensors")) {
      std::fprintf(stderr, "cannot open safetensors\n");
      return false;
    }
    Safetensors ro;
    if (!ro.Open(model_dir + "/readout.safetensors")) return false;
    std::size_t cnt = 0;
    const auto* rp = ro.Find("weight", &cnt);
    readout.resize(cnt);
    for (std::size_t i = 0; i < cnt; ++i) readout[i] = Bf(rp[i]);
    std::string err;
    if (!rt.Init(programs + "/final_16x64x128_ffn.xclbin", &err)) {
      std::fprintf(stderr, "npu init: %s\n", err.c_str());
      return false;
    }
    embed = w.Raw("language_model.embed_tokens.weight", 0 + std::size_t{248320} * kH);
    norm_w = w.Vec("language_model.norm.weight", kH, 1.0f);
    layers.resize(kLayers);
    const auto t0 = Clock::now();
    std::vector<float> buf;
    for (std::uint32_t l = 0; l < kLayers; ++l) {
      Layer& L = layers[l];
      const std::string pre = "language_model.layers." + std::to_string(l) + ".";
      L.linear = (l % 4) != 3;
      L.ln1 = w.Vec(pre + "input_layernorm.weight", kH, 1.0f);
      L.ln2 = w.Vec(pre + "post_attention_layernorm.weight", kH, 1.0f);
      if (L.linear) {
        const std::string a = pre + "linear_attn.";
        L.conv = w.Vec(a + "conv1d.weight", std::size_t{kConvDim} * kConvK);
        L.a_log = w.Vec(a + "A_log", kVH);
        L.dt_bias = w.Vec(a + "dt_bias", kVH);
        L.gnorm = w.Vec(a + "norm.weight", kVD);
        L.w_a = w.Vec(a + "in_proj_a.weight", std::size_t{kVH} * kH);
        L.w_b = w.Vec(a + "in_proj_b.weight", std::size_t{kVH} * kH);
        L.p_in = rt.MakeProgram(kH, kGdnInN, programs + "/insts_plain_2560x12288_m64.txt", 0, &err, 0, 4);
        buf.resize(std::size_t{kGdnInN} * kH);
        w.Load(a + "in_proj_qkv.weight", buf.data(), std::size_t{kConvDim} * kH);
        w.Load(a + "in_proj_z.weight", buf.data() + std::size_t{kConvDim} * kH, std::size_t{kValDim} * kH);
        L.w_in = rt.PackWeight(buf.data(), kGdnInN, kH);
        L.p_out = rt.MakeProgram(kValDim, kOutN, programs + "/insts_plain_4096x3072_m64.txt", 0, &err, 0, 4);
        buf.assign(std::size_t{kOutN} * kValDim, 0.f);
        w.Load(a + "out_proj.weight", buf.data(), std::size_t{kH} * kValDim);
        L.w_out = rt.PackWeight(buf.data(), kOutN, kValDim);
      } else {
        const std::string a = pre + "self_attn.";
        L.qn = w.Vec(a + "q_norm.weight", kHeadDim, 1.0f);
        L.kn = w.Vec(a + "k_norm.weight", kHeadDim, 1.0f);
        L.p_in = rt.MakeProgram(kH, kAttnInN, programs + "/insts_plain_2560x10240_m64.txt", 0, &err, 0, 4);
        buf.resize(std::size_t{kAttnInN} * kH);
        w.Load(a + "q_proj.weight", buf.data(), std::size_t{2 * kQDim} * kH);
        w.Load(a + "k_proj.weight", buf.data() + std::size_t{2 * kQDim} * kH, std::size_t{kKvDim} * kH);
        w.Load(a + "v_proj.weight", buf.data() + std::size_t{2 * kQDim + kKvDim} * kH, std::size_t{kKvDim} * kH);
        L.w_in = rt.PackWeight(buf.data(), kAttnInN, kH);
        L.p_out = rt.MakeProgram(kQDim, kOutN, programs + "/insts_plain_4096x3072_m64.txt", 0, &err, 0, 4);
        buf.assign(std::size_t{kOutN} * kQDim, 0.f);
        w.Load(a + "o_proj.weight", buf.data(), std::size_t{kH} * kQDim);
        L.w_out = rt.PackWeight(buf.data(), kOutN, kQDim);
      }
      L.p_gu = rt.MakeProgram(kH, 2 * kI, programs + "/insts_ffngu64_2560.txt", 0, &err, 0, 4);
      L.p_dn = rt.MakeProgram(kI, kOutN, programs + "/insts_plain_9216x3072_m64.txt", 0, &err, 0, 4);
      std::vector<float> gate(std::size_t{kI} * kH), up(std::size_t{kI} * kH);
      w.Load(pre + "mlp.gate_proj.weight", gate.data(), gate.size());
      w.Load(pre + "mlp.up_proj.weight", up.data(), up.size());
      buf.resize(std::size_t{2 * kI} * kH);
      for (std::size_t f = 0; f < kI / 8; ++f)
        for (std::size_t i = 0; i < 8; ++i) {
          std::memcpy(&buf[(f * 16 + i) * kH], &gate[(f * 8 + i) * kH], kH * 4);
          std::memcpy(&buf[(f * 16 + 8 + i) * kH], &up[(f * 8 + i) * kH], kH * 4);
        }
      L.w_gu = rt.PackWeight(buf.data(), 2 * kI, kH);
      buf.assign(std::size_t{kOutN} * kI, 0.f);
      w.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{kH} * kI);
      L.w_dn = rt.PackWeight(buf.data(), kOutN, kI);
      if (l % 8 == 0) std::fprintf(stderr, "q35: packed layer %u (%.1f s)\n", l, Ms(t0) / 1000.0);
    }
    std::fprintf(stderr, "q35: packed %u layers in %.1f s\n", kLayers, Ms(t0) / 1000.0);
    return true;
  }

  void Gemm(Program& p, WeightBuffer& wb, const float* z, std::uint32_t T, std::uint32_t K, float* out) {
    const auto t0 = Clock::now();
    for (std::uint32_t o = 0; o < T; o += kRows) {
      const std::uint32_t n = std::min(kRows, T - o);
      rt.Gemm(p, wb, z + std::size_t{o} * K, n, out + std::size_t{o} * p.N, &tim);
    }
    t_npu += Ms(t0);
  }

  void Gdn(Layer& L, const float* z, std::uint32_t T, float* out_h) {
    std::vector<float> proj(std::size_t{T} * kGdnInN);
    Gemm(L.p_in, L.w_in, z, T, kH, proj.data());
    const auto t0 = Clock::now();
    std::vector<float> ab(std::size_t{T} * 2 * kVH);
#pragma omp parallel for collapse(2)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::uint32_t h = 0; h < 2 * kVH; ++h) {
        const float* wrow = (h < kVH ? L.w_a.data() : L.w_b.data()) + std::size_t{h % kVH} * kH;
        const float* zr = z + std::size_t(t) * kH;
        float s = 0;
        for (std::uint32_t i = 0; i < kH; ++i) s += wrow[i] * zr[i];
        ab[std::size_t(t) * 2 * kVH + h] = s;
      }
    std::vector<float> qkv(std::size_t{T} * kConvDim);
#pragma omp parallel for
    for (std::int64_t c = 0; c < kConvDim; ++c) {
      const float* wc = &L.conv[std::size_t(c) * kConvK];
      for (std::uint32_t t = 0; t < T; ++t) {
        float s = 0;
        for (std::uint32_t j = 0; j < kConvK; ++j) {
          const std::int64_t src = static_cast<std::int64_t>(t) - (kConvK - 1) + j;
          if (src >= 0) s += wc[j] * proj[std::size_t(src) * kGdnInN + c];
        }
        qkv[std::size_t{t} * kConvDim + c] = Silu(s);
      }
    }
    std::vector<float> gated(std::size_t{T} * kValDim);
#pragma omp parallel for
    for (std::int64_t h = 0; h < kVH; ++h) {
      const std::uint32_t kh = static_cast<std::uint32_t>(h) / (kVH / kKH);
      std::vector<float> S(std::size_t{kKD} * kVD, 0.f), q(kKD), k(kKD), kv(kVD), o(kVD);
      const float a_exp = std::exp(L.a_log[h]);
      for (std::uint32_t t = 0; t < T; ++t) {
        const float* row = &qkv[std::size_t{t} * kConvDim];
        std::memcpy(q.data(), row + kh * kKD, kKD * 4);
        std::memcpy(k.data(), row + kKeyDim + kh * kKD, kKD * 4);
        const float* v = row + 2 * kKeyDim + h * kVD;
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
        const float av = ab[std::size_t{t} * 2 * kVH + h], bv = ab[std::size_t{t} * 2 * kVH + kVH + h];
        const float x = av + L.dt_bias[h];
        const float sp = x > 20.f ? x : std::log1p(std::exp(x));
        const float decay = std::exp(-a_exp * sp);
        const float beta = 1.0f / (1.0f + std::exp(-bv));
        std::fill(kv.begin(), kv.end(), 0.f);
        for (std::uint32_t i = 0; i < kKD; ++i) {
          float* Si = &S[std::size_t{i} * kVD];
          for (std::uint32_t j = 0; j < kVD; ++j) {
            Si[j] *= decay;
            kv[j] += Si[j] * k[i];
          }
        }
        for (std::uint32_t j = 0; j < kVD; ++j) kv[j] = (v[j] - kv[j]) * beta;
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
        const float rs = 1.0f / std::sqrt(ss / kVD + kEps);
        const float* zz = &proj[std::size_t{t} * kGdnInN + kConvDim + h * kVD];
        float* g = &gated[std::size_t{t} * kValDim + h * kVD];
        for (std::uint32_t j = 0; j < kVD; ++j) g[j] = L.gnorm[j] * (o[j] * rs) * Silu(zz[j]);
      }
    }
    t_host += Ms(t0);
    std::vector<float> outp(std::size_t{T} * kOutN);
    Gemm(L.p_out, L.w_out, gated.data(), T, kValDim, outp.data());
    for (std::uint32_t t = 0; t < T; ++t) std::memcpy(out_h + std::size_t{t} * kH, &outp[std::size_t{t} * kOutN], kH * 4);
  }

  void Attn(Layer& L, const float* z, std::uint32_t T, float* out_h) {
    std::vector<float> proj(std::size_t{T} * kAttnInN);
    Gemm(L.p_in, L.w_in, z, T, kH, proj.data());
    const auto t0 = Clock::now();
    std::vector<float> q(std::size_t{T} * kQDim), gate(std::size_t{T} * kQDim), k(std::size_t{T} * kKvDim), v(std::size_t{T} * kKvDim);
    std::vector<float> cs(std::size_t{T} * kRot / 2), sn(std::size_t{T} * kRot / 2);
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::uint32_t i = 0; i < kRot / 2; ++i) {
        const double ang = static_cast<double>(t) * std::pow(static_cast<double>(kTheta), -2.0 * i / kRot);
        cs[std::size_t{t} * (kRot / 2) + i] = static_cast<float>(std::cos(ang));
        sn[std::size_t{t} * (kRot / 2) + i] = static_cast<float>(std::sin(ang));
      }
    auto norm_rope = [&](float* x, const float* nw, std::uint32_t t) {
      float ss = 0;
      for (std::uint32_t i = 0; i < kHeadDim; ++i) ss += x[i] * x[i];
      const float rs = 1.0f / std::sqrt(ss / kHeadDim + kEps);
      for (std::uint32_t i = 0; i < kHeadDim; ++i) x[i] *= rs * nw[i];
      for (std::uint32_t i = 0; i < kRot / 2; ++i) {
        const float c = cs[std::size_t{t} * (kRot / 2) + i], s = sn[std::size_t{t} * (kRot / 2) + i];
        const float a = x[i], b = x[i + kRot / 2];
        x[i] = a * c - b * s;
        x[i + kRot / 2] = b * c + a * s;
      }
    };
#pragma omp parallel for
    for (std::int64_t t = 0; t < T; ++t) {
      const float* row = &proj[std::size_t(t) * kAttnInN];
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        float* qh = &q[std::size_t(t) * kQDim + h * kHeadDim];
        std::memcpy(qh, row + h * 2 * kHeadDim, kHeadDim * 4);
        std::memcpy(&gate[std::size_t(t) * kQDim + h * kHeadDim], row + h * 2 * kHeadDim + kHeadDim, kHeadDim * 4);
        norm_rope(qh, L.qn.data(), static_cast<std::uint32_t>(t));
      }
      for (std::uint32_t h = 0; h < kKvHeads; ++h) {
        float* kh = &k[std::size_t(t) * kKvDim + h * kHeadDim];
        std::memcpy(kh, row + 2 * kQDim + h * kHeadDim, kHeadDim * 4);
        norm_rope(kh, L.kn.data(), static_cast<std::uint32_t>(t));
        std::memcpy(&v[std::size_t(t) * kKvDim + h * kHeadDim], row + 2 * kQDim + kKvDim + h * kHeadDim, kHeadDim * 4);
      }
    }
    std::vector<float> att(std::size_t{T} * kQDim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const std::uint32_t grp = kHeads / kKvHeads;
#pragma omp parallel
    {
      std::vector<float> sc(T);
#pragma omp for collapse(2) schedule(dynamic, 4)
      for (std::int64_t t = 0; t < T; ++t)
        for (std::uint32_t h = 0; h < kHeads; ++h) {
          const float* qh = &q[std::size_t(t) * kQDim + h * kHeadDim];
          const std::uint32_t kh = h / grp;
          float mx = -INFINITY;
          for (std::uint32_t u = 0; u < T; ++u) {
            const float* kk = &k[std::size_t{u} * kKvDim + kh * kHeadDim];
            float s = 0;
            for (std::uint32_t i = 0; i < kHeadDim; ++i) s += qh[i] * kk[i];
            s *= scale;
            sc[u] = s;
            mx = std::max(mx, s);
          }
          float den = 0;
          for (std::uint32_t u = 0; u < T; ++u) {
            sc[u] = std::exp(sc[u] - mx);
            den += sc[u];
          }
          float* o = &att[std::size_t(t) * kQDim + h * kHeadDim];
          std::fill(o, o + kHeadDim, 0.f);
          for (std::uint32_t u = 0; u < T; ++u) {
            const float wgt = sc[u] / den;
            const float* vv = &v[std::size_t{u} * kKvDim + kh * kHeadDim];
            for (std::uint32_t i = 0; i < kHeadDim; ++i) o[i] += wgt * vv[i];
          }
          const float* g = &gate[std::size_t(t) * kQDim + h * kHeadDim];
          for (std::uint32_t i = 0; i < kHeadDim; ++i) o[i] *= 1.0f / (1.0f + std::exp(-g[i]));
        }
    }
    t_host += Ms(t0);
    std::vector<float> outp(std::size_t{T} * kOutN);
    Gemm(L.p_out, L.w_out, att.data(), T, kQDim, outp.data());
    for (std::uint32_t t = 0; t < T; ++t) std::memcpy(out_h + std::size_t{t} * kH, &outp[std::size_t{t} * kOutN], kH * 4);
  }

  void Mlp(Layer& L, const float* z, std::uint32_t T, float* out_h) {
    std::vector<float> yo(std::size_t{kRows} * kOutN);
    const auto t0 = Clock::now();
    for (std::uint32_t o = 0; o < T; o += kRows) {
      const std::uint32_t n = std::min(kRows, T - o);
      rt.Ffn64(L.w_gu, L.p_gu, L.w_dn, L.p_dn, z + std::size_t{o} * kH, n, yo.data(), &tim, kI / kTileK);
      for (std::uint32_t r = 0; r < n; ++r) std::memcpy(out_h + std::size_t{o + r} * kH, &yo[std::size_t{r} * kOutN], kH * 4);
    }
    t_npu += Ms(t0);
  }

  void Forward(const std::int32_t* tok, std::uint32_t T, const std::vector<std::uint32_t>& markers, std::vector<float>* hidden) {
    std::vector<float> x(std::size_t{T} * kH), z(std::size_t{T} * kH), br(std::size_t{T} * kH);
#pragma omp parallel for
    for (std::int64_t t = 0; t < T; ++t)
      for (std::uint32_t i = 0; i < kH; ++i) x[std::size_t(t) * kH + i] = Bf(embed[std::size_t(tok[t]) * kH + i]);
    for (std::uint32_t l = 0; l < kLayers; ++l) {
      Layer& L = layers[l];
      RmsNormRows(x.data(), L.ln1.data(), z.data(), T, kH);
      if (L.linear) Gdn(L, z.data(), T, br.data());
      else Attn(L, z.data(), T, br.data());
      for (std::size_t i = 0; i < x.size(); ++i) x[i] += br[i];
      RmsNormRows(x.data(), L.ln2.data(), z.data(), T, kH);
      Mlp(L, z.data(), T, br.data());
      for (std::size_t i = 0; i < x.size(); ++i) x[i] += br[i];
      if (std::getenv("Q35_TRACE")) {
        double s = 0;
        for (std::uint32_t i = 0; i < kH; ++i) s += std::fabs(x[std::size_t{markers.back()} * kH + i]);
        std::fprintf(stderr, "layer %u marker |x| mean %.6f\n", l, s / kH);
      }
    }
    hidden->resize(markers.size() * kH);
    for (std::size_t m = 0; m < markers.size(); ++m) RmsNormRows(&x[std::size_t{markers[m]} * kH], norm_w.data(), &(*hidden)[m * kH], 1, kH);
  }
};

}  // namespace

int EvalBatch(const char* model, const char* programs, const char* path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::vector<std::int32_t> d(b.size() / 4);
  std::memcpy(d.data(), b.data(), d.size() * 4);
  Engine e;
  if (!e.Init(model, programs)) return 1;
  std::size_t at = 0;
  const std::int32_t cases = d[at++];
  const auto t0 = Clock::now();
  std::uint64_t total = 0;
  for (std::int32_t c = 0; c < cases; ++c) {
    const std::uint32_t len = static_cast<std::uint32_t>(d[at++]), marker = static_cast<std::uint32_t>(d[at++]), cnt = static_cast<std::uint32_t>(d[at++]);
    std::vector<float> hid;
    e.Forward(&d[at], len, {marker}, &hid);
    at += len;
    total += len;
    for (std::uint32_t k = 0; k < cnt; ++k) {
      float s = 0;
      for (std::uint32_t i = 0; i < kH; ++i) s += e.readout[std::size_t{k} * kH + i] * hid[i];
      std::printf("%.5f ", s);
    }
    std::printf("\n");
  }
  std::fprintf(stderr, "eval: %d cases %llu tokens %.1f ms (%.1f ms/case, npu %.0f host %.0f)\n", cases, static_cast<unsigned long long>(total), Ms(t0), Ms(t0) / cases, e.t_npu, e.t_host);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "eval") return EvalBatch(argv[2], argv[3], argv[4]);
  if (argc < 6) {
    std::fprintf(stderr, "usage: gufo-q35 logits MODEL_DIR PROGRAMS TOKENS.i32 MARKER[,MARKER...] [COUNT]\n");
    return 2;
  }
  std::ifstream f(argv[4], std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::vector<std::int32_t> toks(b.size() / 4);
  std::memcpy(toks.data(), b.data(), toks.size() * 4);
  std::vector<std::uint32_t> markers;
  for (char* p = argv[5]; *p;) {
    markers.push_back(static_cast<std::uint32_t>(std::strtoul(p, &p, 10)));
    if (*p == ',') ++p;
  }
  const std::uint32_t count = argc > 6 ? static_cast<std::uint32_t>(std::atoi(argv[6])) : 8;
  Engine e;
  if (!e.Init(argv[2], argv[3])) return 1;
  std::vector<float> hid;
  const auto t0 = Clock::now();
  e.Forward(toks.data(), static_cast<std::uint32_t>(toks.size()), markers, &hid);
  std::fprintf(stderr, "forward %zu tokens %.1f ms (npu %.0f ms, host %.0f ms)\n", toks.size(), Ms(t0), e.t_npu, e.t_host);
  std::fprintf(stderr, "npu: enc %.0f syncA %.0f submit %.0f wait %.0f syncC %.0f dec %.0f ms\n", e.tim.enc, e.tim.syncA, e.tim.submit, e.tim.wait, e.tim.syncC, e.tim.dec);
  for (std::size_t m = 0; m < markers.size(); ++m) {
    for (std::uint32_t c = 0; c < count; ++c) {
      float s = 0;
      for (std::uint32_t i = 0; i < kH; ++i) s += e.readout[std::size_t{c} * kH + i] * hid[m * kH + i];
      std::printf("%.5f ", s);
    }
    std::printf("\n");
  }
  return 0;
}
