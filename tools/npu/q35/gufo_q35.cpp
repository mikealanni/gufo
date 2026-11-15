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

struct GdnDev {
  xrt::device dev;
  xrt::hw_context ctx;
  xrt::kernel kern;
  std::vector<std::pair<std::uint32_t, xrt::bo>> insts;
  std::vector<std::uint32_t> inst_words;
  xrt::bo bin, bx, bout;
  std::uint8_t* in_h{nullptr};
  const float* out_h{nullptr};
  static constexpr std::uint32_t kMaxNe = 512;
  static constexpr std::uint32_t kMaxTok = 4 * kMaxNe;
  bool Init(const std::string& programs) {
    try {
      dev = xrt::device(0u);
      const xrt::xclbin xb{programs + "/final_gdn.xclbin"};
      dev.register_xclbin(xb);
      ctx = xrt::hw_context(dev, xb.get_uuid());
      kern = xrt::kernel(ctx, xb.get_kernels().at(0).get_name());
      for (std::uint32_t ne : {16u, 32u, 64u, 128u, 256u, 512u}) {
        std::ifstream f(programs + "/insts_gdn_" + std::to_string(ne) + ".txt", std::ios::binary | std::ios::ate);
        if (!f) return false;
        const std::size_t n = static_cast<std::size_t>(f.tellg());
        f.seekg(0);
        std::vector<char> v(n);
        f.read(v.data(), static_cast<std::streamsize>(n));
        xrt::bo b(dev, n, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
        std::memcpy(b.map<void*>(), v.data(), n);
        b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        insts.emplace_back(ne, b);
        inst_words.push_back(static_cast<std::uint32_t>(n / 4));
      }
      bin = xrt::bo(dev, std::size_t{8} * kMaxNe * 4 * 4096, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
      bx = xrt::bo(dev, 4, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
      bout = xrt::bo(dev, std::size_t{8} * kMaxNe * 4 * 2048, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
      in_h = bin.map<std::uint8_t*>();
      out_h = bout.map<float*>();
      std::memset(in_h, 0, std::size_t{8} * kMaxNe * 4 * 4096);
      std::memset(bout.map<void*>(), 0, std::size_t{8} * kMaxNe * 4 * 2048);
      return true;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "gdn device: %s\n", e.what());
      return false;
    }
  }
  static std::uint32_t Bucket(std::uint32_t tokens) {
    for (std::uint32_t ne : {16u, 32u, 64u, 128u, 256u, 512u})
      if (tokens <= 4 * ne) return ne;
    return 0;
  }
  void Run(std::uint32_t ne) {
    std::size_t idx = 0;
    for (; idx < insts.size(); ++idx)
      if (insts[idx].first == ne) break;
    bin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kern(3, insts[idx].second, inst_words[idx], bin, bx, bout);
    run.wait();
    bout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }
};

struct Layer {
  bool linear{false};
  std::vector<float> ln1, ln2;
  // gated delta net
  std::vector<float> conv, convT, a_log, dt_bias, gnorm, w_a, w_b;
  // attention
  std::vector<float> qn, kn;
  WeightBuffer w_in, w_out, w_gu, w_dn;
  Program p_in, p_out, p_gu, p_dn;
  WeightBuffer kw_in, kw_out, kw_g, kw_u, kw_d;
  Program kp_in, kp_out, kp_g, kp_u, kp_d;
  WeightBuffer kw_a, kw_b;
  Program kq_a, kq_b, kq_d;
  xrt::bo swc;
  bool fused{false};
};

struct Engine {
  NpuRuntime rt, rk;
  GdnDev gd;
  bool gdn_dev{std::getenv("Q35_GDN_DEV") != nullptr};
  NpuRuntime::Timings tim{};
  bool kd{std::getenv("Q35_KD") != nullptr};
  std::vector<Layer> layers;
  std::vector<float> norm_w, readout;
  std::vector<std::uint16_t> embed_raw;
  const std::uint16_t* embed{nullptr};
  Weights w;
  double t_npu{0}, t_host{0}, ph[4]{0, 0, 0, 0}, mx[8]{0, 0, 0, 0, 0, 0, 0, 0};
  const char* qdir{std::getenv("Q35_QDIR")};
  bool bf16_state{std::getenv("Q35_BF16_STATE") != nullptr};
  bool gdn_emu{std::getenv("Q35_GDN_EMU") != nullptr};

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
    if (!(kd ? rk.Init(programs + "/final_kd.xclbin", &err) : rt.Init(programs + "/final_16x64x128_ffn.xclbin", &err))) {
      std::fprintf(stderr, "npu init: %s\n", err.c_str());
      return false;
    }
    embed = w.Raw("language_model.embed_tokens.weight", 0 + std::size_t{248320} * kH);
    norm_w = w.Vec("language_model.norm.weight", kH, 1.0f);
    if (gdn_dev && !gd.Init(programs)) gdn_dev = false;
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
        if (kd) {
          PackKdIn(L, a, programs, true, buf, &err);
          PackKdOut(L, a + "out_proj.weight", kValDim, programs, buf, &err);
        } else {
        L.p_in = rt.MakeProgram(kH, kGdnInN, programs + "/insts_plain_2560x12288_m64.txt", 0, &err, 0, 4);
        buf.resize(std::size_t{kGdnInN} * kH);
        w.Load(a + "in_proj_qkv.weight", buf.data(), std::size_t{kConvDim} * kH);
        w.Load(a + "in_proj_z.weight", buf.data() + std::size_t{kConvDim} * kH, std::size_t{kValDim} * kH);
        if (qdir) {
          QMat m;
          m.Resize(kGdnInN, kH);
          m.Load(qdir, std::to_string(l) + ".qkv", 0, kConvDim);
          m.Load(qdir, std::to_string(l) + ".z", kConvDim, kValDim);
          L.w_in = rt.PackWeightQ(m.q.data(), m.s.data(), kGdnInN, kH);
        } else {
          L.w_in = rt.PackWeight(buf.data(), kGdnInN, kH);
        }
        L.p_out = rt.MakeProgram(kValDim, kOutN, programs + "/insts_plain_4096x3072_m64.txt", 0, &err, 0, 4);
        buf.assign(std::size_t{kOutN} * kValDim, 0.f);
        w.Load(a + "out_proj.weight", buf.data(), std::size_t{kH} * kValDim);
        if (qdir) {
          QMat m;
          m.Resize(kOutN, kValDim);
          m.Load(qdir, std::to_string(l) + ".out", 0, kH);
          L.w_out = rt.PackWeightQ(m.q.data(), m.s.data(), kOutN, kValDim);
        } else {
          L.w_out = rt.PackWeight(buf.data(), kOutN, kValDim);
        }
        }
      } else {
        const std::string a = pre + "self_attn.";
        L.qn = w.Vec(a + "q_norm.weight", kHeadDim, 1.0f);
        L.kn = w.Vec(a + "k_norm.weight", kHeadDim, 1.0f);
        if (kd) {
          PackKdIn(L, a, programs, false, buf, &err);
          PackKdOut(L, a + "o_proj.weight", kQDim, programs, buf, &err);
        } else {
        L.p_in = rt.MakeProgram(kH, kAttnInN, programs + "/insts_plain_2560x10240_m64.txt", 0, &err, 0, 4);
        buf.resize(std::size_t{kAttnInN} * kH);
        w.Load(a + "q_proj.weight", buf.data(), std::size_t{2 * kQDim} * kH);
        w.Load(a + "k_proj.weight", buf.data() + std::size_t{2 * kQDim} * kH, std::size_t{kKvDim} * kH);
        w.Load(a + "v_proj.weight", buf.data() + std::size_t{2 * kQDim + kKvDim} * kH, std::size_t{kKvDim} * kH);
        if (qdir) {
          QMat m;
          m.Resize(kAttnInN, kH);
          m.Load(qdir, std::to_string(l) + ".q", 0, 2 * kQDim);
          m.Load(qdir, std::to_string(l) + ".k", 2 * kQDim, kKvDim);
          m.Load(qdir, std::to_string(l) + ".v", 2 * kQDim + kKvDim, kKvDim);
          L.w_in = rt.PackWeightQ(m.q.data(), m.s.data(), kAttnInN, kH);
        } else {
          L.w_in = rt.PackWeight(buf.data(), kAttnInN, kH);
        }
        L.p_out = rt.MakeProgram(kQDim, kOutN, programs + "/insts_plain_4096x3072_m64.txt", 0, &err, 0, 4);
        buf.assign(std::size_t{kOutN} * kQDim, 0.f);
        w.Load(a + "o_proj.weight", buf.data(), std::size_t{kH} * kQDim);
        if (qdir) {
          QMat m;
          m.Resize(kOutN, kQDim);
          m.Load(qdir, std::to_string(l) + ".o", 0, kH);
          L.w_out = rt.PackWeightQ(m.q.data(), m.s.data(), kOutN, kQDim);
        } else {
          L.w_out = rt.PackWeight(buf.data(), kOutN, kQDim);
        }
        }
      }
      if (kd) {
        PackKdMlp(L, pre, programs, buf, &err);
      } else {
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
      if (qdir) {
        QMat g, u, gu;
        g.Resize(kI, kH);
        u.Resize(kI, kH);
        gu.Resize(2 * kI, kH);
        g.Load(qdir, std::to_string(l) + ".gate", 0, kI);
        u.Load(qdir, std::to_string(l) + ".up", 0, kI);
        for (std::size_t f = 0; f < kI / 8; ++f) {
          gu.CopyRows(g, f * 8, f * 16, 8);
          gu.CopyRows(u, f * 8, f * 16 + 8, 8);
        }
        L.w_gu = rt.PackWeightQ(gu.q.data(), gu.s.data(), 2 * kI, kH);
      } else {
        L.w_gu = rt.PackWeight(buf.data(), 2 * kI, kH);
      }
      buf.assign(std::size_t{kOutN} * kI, 0.f);
      w.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{kH} * kI);
      if (qdir) {
        QMat m;
        m.Resize(kOutN, kI);
        m.Load(qdir, std::to_string(l) + ".down", 0, kH);
        L.w_dn = rt.PackWeightQ(m.q.data(), m.s.data(), kOutN, kI);
      } else {
        L.w_dn = rt.PackWeight(buf.data(), kOutN, kI);
      }
      }
      if (l % 8 == 0) std::fprintf(stderr, "q35: packed layer %u (%.1f s)\n", l, Ms(t0) / 1000.0);
    }
    std::fprintf(stderr, "q35: packed %u layers in %.1f s\n", kLayers, Ms(t0) / 1000.0);
    return true;
  }

  void PackKdIn(Layer& L, const std::string& a, const std::string& programs, bool gdn, std::vector<float>& buf, std::string* err) {
    if (gdn) {
      L.kp_in = rk.MakeProgram(kH, kGdnInN, programs + "/insts_kd_1024x2560x12288.txt", std::size_t{1024} * kGdnInN / 8 * 9, err, 0, 64, 3, 5);
      buf.resize(std::size_t{kGdnInN} * kH);
      w.Load(a + "in_proj_qkv.weight", buf.data(), std::size_t{kConvDim} * kH);
      w.Load(a + "in_proj_z.weight", buf.data() + std::size_t{kConvDim} * kH, std::size_t{kValDim} * kH);
      L.kw_in = rk.PackWeightAtb(buf.data(), kGdnInN, kH, 4);
    } else {
      L.kp_in = rk.MakeProgram(kH, kAttnInN, programs + "/insts_kd_1024x2560x10240.txt", std::size_t{1024} * kAttnInN / 8 * 9, err, 0, 64, 3, 5);
      buf.resize(std::size_t{kAttnInN} * kH);
      w.Load(a + "q_proj.weight", buf.data(), std::size_t{2 * kQDim} * kH);
      w.Load(a + "k_proj.weight", buf.data() + std::size_t{2 * kQDim} * kH, std::size_t{kKvDim} * kH);
      w.Load(a + "v_proj.weight", buf.data() + std::size_t{2 * kQDim + kKvDim} * kH, std::size_t{kKvDim} * kH);
      L.kw_in = rk.PackWeightAtb(buf.data(), kAttnInN, kH, 4);
    }
  }

  void PackKdOut(Layer& L, const std::string& name, std::uint32_t K, const std::string& programs, std::vector<float>& buf, std::string* err) {
    L.kp_out = rk.MakeProgram(K, kOutN, programs + "/insts_kd_1024x4096x3072.txt", std::size_t{1024} * kOutN / 8 * 9, err, 0, 64, 3, 5);
    buf.assign(std::size_t{kOutN} * K, 0.f);
    w.Load(name, buf.data(), std::size_t{kH} * K);
    L.kw_out = rk.PackWeightAtb(buf.data(), kOutN, K, 4);
  }

  void PackKdMlp(Layer& L, const std::string& pre, const std::string& programs, std::vector<float>& buf, std::string* err) {
    if (!std::getenv("Q35_NOFUSE")) {
      constexpr std::uint32_t kHalf = kI / 2;
      std::vector<float> gate(std::size_t{kI} * kH), up(std::size_t{kI} * kH), ws(std::size_t{2} * kHalf * kH);
      w.Load(pre + "mlp.gate_proj.weight", gate.data(), gate.size());
      w.Load(pre + "mlp.up_proj.weight", up.data(), up.size());
      auto pack = [&](std::uint32_t c0) {
        for (std::uint32_t r = 0; r < 2 * kHalf; ++r) {
          const std::uint32_t q = r >> 4, rr = r & 15;
          const float* src = (rr < 8 ? gate.data() : up.data()) + std::size_t{c0 + q * 8 + (rr & 7)} * kH;
          std::memcpy(ws.data() + std::size_t{r} * kH, src, kH * 4);
        }
        return rk.PackWeightAtb(ws.data(), 2 * kHalf, kH, 4);
      };
      L.kw_a = pack(0);
      L.kw_b = pack(kHalf);
      const std::size_t abytes = std::size_t{1024} * kI / 8 * 9;
      L.swc = rk.MakeBo(2 * abytes, 3);
      std::memset(L.swc.map<void*>(), 0, 2 * abytes);
      L.swc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      L.kq_a = rk.MakeProgram(kH, 2 * kHalf, programs + "/insts_kd_swi_a_1024x2560x9216.txt", 64, err, 0, 64, 3, 5);
      L.kq_b = rk.MakeProgram(kH, 2 * kHalf, programs + "/insts_kd_swi_b_1024x2560x9216.txt", 64, err, 0, 64, 3, 5);
      L.kq_a.c = L.swc;
      L.kq_a.ch = L.swc.map<std::uint8_t*>();
      L.kq_b.c = L.swc;
      L.kq_b.ch = L.kq_a.ch;
      L.kq_d = rk.MakeProgram(kI, kOutN, programs + "/insts_kd_dn2_1024x9216x3072.txt", std::size_t{1024} * kOutN / 8 * 9, err, 0, 64, 3, 5);
      rk.RebindA(L.kq_d, L.swc);
      buf.assign(std::size_t{kOutN} * kI, 0.f);
      w.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{kH} * kI);
      L.kw_d = rk.PackWeightAtb(buf.data(), kOutN, kI, 4);
      rk.WeightSlot(L.kw_d, L.kq_d);
      L.fused = true;
      return;
    }
    L.kp_g = rk.MakeProgram(kH, kI, programs + "/insts_kd_1024x2560x9216.txt", std::size_t{1024} * kI / 8 * 9, err, 0, 64, 3, 5);
    L.kp_u = rk.MakeProgram(kH, kI, programs + "/insts_kd_1024x2560x9216.txt", std::size_t{1024} * kI / 8 * 9, err, 0, 64, 3, 5);
    L.kp_d = rk.MakeProgram(kI, kOutN, programs + "/insts_kd_1024x9216x3072.txt", std::size_t{1024} * kOutN / 8 * 9, err, 0, 64, 3, 5);
    buf.resize(std::size_t{kI} * kH);
    w.Load(pre + "mlp.gate_proj.weight", buf.data(), buf.size());
    L.kw_g = rk.PackWeightAtb(buf.data(), kI, kH, 4);
    w.Load(pre + "mlp.up_proj.weight", buf.data(), buf.size());
    L.kw_u = rk.PackWeightAtb(buf.data(), kI, kH, 4);
    buf.assign(std::size_t{kOutN} * kI, 0.f);
    w.Load(pre + "mlp.down_proj.weight", buf.data(), std::size_t{kH} * kI);
    L.kw_d = rk.PackWeightAtb(buf.data(), kOutN, kI, 4);
  }

  std::vector<float> kx, ky, s_proj, s_gated, s_outp, s_att, s_qkv, s_ab;

  void KGemm(Program& p, WeightBuffer& wb, const float* z, std::uint32_t T, std::uint32_t K, float* out) {
    const auto t0 = Clock::now();
    if (kx.size() < std::size_t{1024} * 9216) kx.assign(std::size_t{1024} * 9216, 0.f);
    if (ky.size() < std::size_t{1024} * kGdnInN) ky.assign(std::size_t{1024} * kGdnInN, 0.f);
    for (std::uint32_t o = 0; o < T; o += 1024) {
      const std::uint32_t n = std::min(1024u, T - o);
      if (n == 1024) {
        rk.GemmAtb(p, wb, z + std::size_t{o} * K, 1024, out + std::size_t{o} * p.N, &tim);
        continue;
      }
      std::memcpy(kx.data(), z + std::size_t{o} * K, std::size_t{n} * K * 4);
      std::memset(kx.data() + std::size_t{n} * K, 0, std::size_t{1024 - n} * K * 4);
      rk.GemmAtb(p, wb, kx.data(), 1024, ky.data(), &tim);
      std::memcpy(out + std::size_t{o} * p.N, ky.data(), std::size_t{n} * p.N * 4);
    }
    t_npu += Ms(t0);
  }

  void Gemm(Program& p, WeightBuffer& wb, const float* z, std::uint32_t T, std::uint32_t K, float* out) {
    const auto t0 = Clock::now();
    for (std::uint32_t o = 0; o < T; o += kRows) {
      const std::uint32_t n = std::min(kRows, T - o);
      rt.Gemm(p, wb, z + std::size_t{o} * K, n, out + std::size_t{o} * p.N, &tim);
    }
    t_npu += Ms(t0);
  }

  void Gdn(Layer& L, const float* z, std::uint32_t Tt, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& segs, float* out_h) {
    auto ta = Clock::now();
    s_proj.resize(std::size_t{Tt} * kGdnInN);
    s_gated.resize(std::size_t{Tt} * kValDim);
    std::vector<float>& proj = s_proj;
    std::vector<float>& gated = s_gated;
    mx[4] += Ms(ta); ta = Clock::now();
    if (kd) KGemm(L.kp_in, L.kw_in, z, Tt, kH, proj.data());
    else Gemm(L.p_in, L.w_in, z, Tt, kH, proj.data());
    mx[0] += Ms(ta);
    const auto t0 = Clock::now();
    const std::uint32_t ne = GdnDev::Bucket(Tt);
    if (gdn_dev && ne) {
      for (const auto& sg : segs) GdnPrepDev(L, proj.data() + std::size_t{sg.first} * kGdnInN, z + std::size_t{sg.first} * kH, sg.second, sg.first, ne);
      GdnPadDev(Tt, 4 * ne, ne);
      const auto td = Clock::now();
      gd.Run(ne);
      mx[7] += Ms(td);
      for (const auto& sg : segs) GdnPostDev(L, proj.data() + std::size_t{sg.first} * kGdnInN, sg.second, sg.first, ne, gated.data() + std::size_t{sg.first} * kValDim);
    } else {
      for (const auto& sg : segs) GdnHost(L, proj.data() + std::size_t{sg.first} * kGdnInN, z + std::size_t{sg.first} * kH, sg.second, gated.data() + std::size_t{sg.first} * kValDim);
    }
    t_host += Ms(t0);
    mx[1] += Ms(t0);
    ta = Clock::now();
    s_outp.resize(std::size_t{Tt} * kOutN);
    std::vector<float>& outp = s_outp;
    if (kd) KGemm(L.kp_out, L.kw_out, gated.data(), Tt, kValDim, outp.data());
    else Gemm(L.p_out, L.w_out, gated.data(), Tt, kValDim, outp.data());
    mx[2] += Ms(ta);
    for (std::uint32_t t = 0; t < Tt; ++t) std::memcpy(out_h + std::size_t{t} * kH, &outp[std::size_t{t} * kOutN], kH * 4);
  }

  void GdnPrepDev(Layer& L, const float* proj, const float* z, std::uint32_t T, std::uint32_t tok_off, std::uint32_t ne) {
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
    if (L.convT.empty()) {
      L.convT.resize(std::size_t{kConvK} * kConvDim);
      for (std::uint32_t c = 0; c < kConvDim; ++c)
        for (std::uint32_t j = 0; j < kConvK; ++j) L.convT[std::size_t{j} * kConvDim + c] = L.conv[std::size_t{c} * kConvK + j];
    }
    s_qkv.resize(std::size_t{T} * kConvDim);
    float* qkv = s_qkv.data();
#pragma omp parallel for
    for (std::int64_t t = 0; t < T; ++t) {
      float* o = &qkv[std::size_t(t) * kConvDim];
      std::fill(o, o + kConvDim, 0.f);
      for (std::uint32_t j = 0; j < kConvK; ++j) {
        const std::int64_t src = t - (kConvK - 1) + j;
        if (src < 0) continue;
        const float* in = proj + std::size_t(src) * kGdnInN;
        const float* wj = &L.convT[std::size_t{j} * kConvDim];
#pragma omp simd
        for (std::uint32_t c = 0; c < kConvDim; ++c) o[c] += wj[c] * in[c];
      }
      for (std::uint32_t c = 0; c < kConvDim; ++c) o[c] = Silu(o[c]);
    }
#pragma omp parallel for collapse(2)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::uint32_t h = 0; h < kVH; ++h) {
        const std::uint32_t kh = h / (kVH / kKH);
        const float* row = &qkv[std::size_t(t) * kConvDim];
        float qn = 0, kn = 0;
        for (std::uint32_t i = 0; i < kKD; ++i) {
          qn += row[kh * kKD + i] * row[kh * kKD + i];
          kn += row[kKeyDim + kh * kKD + i] * row[kKeyDim + kh * kKD + i];
        }
        const float qs = 1.0f / std::sqrt(qn + 1e-6f) / std::sqrt(static_cast<float>(kKD)), ks = 1.0f / std::sqrt(kn + 1e-6f);
        const std::uint32_t g = tok_off + static_cast<std::uint32_t>(t), col = h / 4, rw = h % 4, e = g / 4, tt = g % 4;
        std::uint8_t* rec = gd.in_h + ((((std::size_t{col} * ne + e) * 4 + rw) * 4) + tt) * 1024;
        auto* b = reinterpret_cast<std::uint16_t*>(rec);
        for (std::uint32_t i = 0; i < kKD; ++i) {
          b[i] = Bf16Bits(row[kh * kKD + i] * qs);
          b[128 + i] = Bf16Bits(row[kKeyDim + kh * kKD + i] * ks);
          b[256 + i] = Bf16Bits(row[2 * kKeyDim + h * kVD + i]);
        }
        const float av = ab[std::size_t{static_cast<std::uint32_t>(t)} * 2 * kVH + h], bv = ab[std::size_t{static_cast<std::uint32_t>(t)} * 2 * kVH + kVH + h];
        const float x = av + L.dt_bias[h];
        const float sp = x > 20.f ? x : std::log1p(std::exp(x));
        const float sc[2] = {std::exp(-std::exp(L.a_log[h]) * sp), 1.0f / (1.0f + std::exp(-bv))};
        std::memcpy(rec + 768, sc, 8);
        const std::int32_t reset = t == 0 ? 1 : 0;
        std::memcpy(rec + 776, &reset, 4);
      }
  }

  void GdnPadDev(std::uint32_t from, std::uint32_t to, std::uint32_t ne) {
    for (std::uint32_t g = from; g < to; ++g)
      for (std::uint32_t h = 0; h < kVH; ++h) {
        const std::uint32_t col = h / 4, rw = h % 4, e = g / 4, tt = g % 4;
        std::uint8_t* rec = gd.in_h + ((((std::size_t{col} * ne + e) * 4 + rw) * 4) + tt) * 1024;
        std::memset(rec, 0, 1024);
        const float sc[2] = {1.0f, 0.0f};
        std::memcpy(rec + 768, sc, 8);
      }
  }

  void GdnPostDev(Layer& L, const float* proj, std::uint32_t T, std::uint32_t tok_off, std::uint32_t ne, float* gated) {
#pragma omp parallel for collapse(2)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::uint32_t h = 0; h < kVH; ++h) {
        const std::uint32_t g = tok_off + static_cast<std::uint32_t>(t), col = h / 4, rw = h % 4, e = g / 4, tt = g % 4;
        const float* o = gd.out_h + (((std::size_t{col} * ne + e) * 4 + rw) * 4 + tt) * 128;
        float ss = 0;
        for (std::uint32_t j = 0; j < kVD; ++j) ss += o[j] * o[j];
        const float rs = 1.0f / std::sqrt(ss / kVD + kEps);
        const float* zz = &proj[std::size_t(t) * kGdnInN + kConvDim + h * kVD];
        float* gt = &gated[std::size_t(t) * kValDim + h * kVD];
        for (std::uint32_t j = 0; j < kVD; ++j) gt[j] = L.gnorm[j] * (o[j] * rs) * Silu(zz[j]);
      }
  }

  void GdnHost(Layer& L, const float* proj, const float* z, std::uint32_t T, float* gated) {
    auto tq = Clock::now();
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
    mx[5] += Ms(tq); tq = Clock::now();
    mx[5] += Ms(tq); tq = Clock::now();
    std::vector<float> qkv(std::size_t{T} * kConvDim);
    if (L.convT.empty()) {
      L.convT.resize(std::size_t{kConvK} * kConvDim);
      for (std::uint32_t c = 0; c < kConvDim; ++c)
        for (std::uint32_t j = 0; j < kConvK; ++j) L.convT[std::size_t{j} * kConvDim + c] = L.conv[std::size_t{c} * kConvK + j];
    }
#pragma omp parallel for
    for (std::int64_t t = 0; t < T; ++t) {
      float* o = &qkv[std::size_t(t) * kConvDim];
      std::fill(o, o + kConvDim, 0.f);
      for (std::uint32_t j = 0; j < kConvK; ++j) {
        const std::int64_t src = t - (kConvK - 1) + j;
        if (src < 0) continue;
        const float* in = proj + std::size_t(src) * kGdnInN;
        const float* wj = &L.convT[std::size_t{j} * kConvDim];
#pragma omp simd
        for (std::uint32_t c = 0; c < kConvDim; ++c) o[c] += wj[c] * in[c];
      }
      for (std::uint32_t c = 0; c < kConvDim; ++c) o[c] = Silu(o[c]);
    }
    mx[6] += Ms(tq); tq = Clock::now();
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
        if (gdn_emu) {
          std::vector<float> qb(kKD), kb(kKD), vb(kVD), kvr(kVD, 0.f), sqr(kVD, 0.f), dh(kVD), dl(kVD);
          for (std::uint32_t i = 0; i < kKD; ++i) {
            qb[i] = Bf(Bf16Bits(q[i]));
            kb[i] = Bf(Bf16Bits(k[i]));
          }
          for (std::uint32_t j = 0; j < kVD; ++j) vb[j] = Bf(Bf16Bits(v[j]));
          float kq = 0;
          for (std::uint32_t i = 0; i < kKD; ++i) {
            kq += kb[i] * qb[i];
            const float* Si = &S[std::size_t{i} * kVD];
            for (std::uint32_t j = 0; j < kVD; ++j) {
              kvr[j] += Si[j] * kb[i];
              sqr[j] += Si[j] * qb[i];
            }
          }
          const float d_hi = Bf(Bf16Bits(decay)), d_lo = Bf(Bf16Bits(decay - d_hi));
          for (std::uint32_t j = 0; j < kVD; ++j) {
            const float delta = (vb[j] - decay * kvr[j]) * beta;
            dh[j] = Bf(Bf16Bits(delta));
            dl[j] = Bf(Bf16Bits(delta - dh[j]));
            o[j] = decay * sqr[j] + kq * delta;
          }
          for (std::uint32_t i = 0; i < kKD; ++i) {
            float* Si = &S[std::size_t{i} * kVD];
            for (std::uint32_t j = 0; j < kVD; ++j) Si[j] = Bf(Bf16Bits(d_hi * Si[j] + d_lo * Si[j] + kb[i] * dh[j] + kb[i] * dl[j]));
          }
        } else {
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
            if (bf16_state) Si[j] = Bf(Bf16Bits(Si[j]));
            o[j] += Si[j] * q[i];
          }
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
  }

  void Attn(Layer& L, const float* z, std::uint32_t Tt, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& segs, float* out_h) {
    s_proj.resize(std::size_t{Tt} * kAttnInN);
    s_att.resize(std::size_t{Tt} * kQDim);
    std::vector<float>& proj = s_proj;
    std::vector<float>& att = s_att;
    auto ta = Clock::now();
    if (kd) KGemm(L.kp_in, L.kw_in, z, Tt, kH, proj.data());
    else Gemm(L.p_in, L.w_in, z, Tt, kH, proj.data());
    mx[3] += Ms(ta);
    const auto t0 = Clock::now();
    for (const auto& sg : segs) AttnHost(L, proj.data() + std::size_t{sg.first} * kAttnInN, sg.second, att.data() + std::size_t{sg.first} * kQDim);
    t_host += Ms(t0);
    s_outp.resize(std::size_t{Tt} * kOutN);
    std::vector<float>& outp = s_outp;
    if (kd) KGemm(L.kp_out, L.kw_out, att.data(), Tt, kQDim, outp.data());
    else Gemm(L.p_out, L.w_out, att.data(), Tt, kQDim, outp.data());
    for (std::uint32_t t = 0; t < Tt; ++t) std::memcpy(out_h + std::size_t{t} * kH, &outp[std::size_t{t} * kOutN], kH * 4);
  }

  void AttnHost(Layer& L, const float* proj, std::uint32_t T, float* att) {
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
  }

  void Dispatch(WeightBuffer& wb, Program& p) {
    WeightBuffer& ww = rk.WeightSlot(wb, p);
    ww.run.start();
    ww.run.wait();
    if (!p.warmed) {
      p.warmed = true;
      ww.run.start();
      ww.run.wait();
    }
  }

  void MlpFused(Layer& L, const float* z, std::uint32_t T, float* out_h) {
    if (kx.size() < std::size_t{1024} * 9216) kx.assign(std::size_t{1024} * 9216, 0.f);
    if (ky.size() < std::size_t{1024} * kGdnInN) ky.assign(std::size_t{1024} * kGdnInN, 0.f);
    const auto t0 = Clock::now();
    for (std::uint32_t o = 0; o < T; o += 1024) {
      const std::uint32_t n = std::min(1024u, T - o);
      const float* src = z + std::size_t{o} * kH;
      if (n < 1024) {
        std::memcpy(kx.data(), src, std::size_t{n} * kH * 4);
        std::memset(kx.data() + std::size_t{n} * kH, 0, std::size_t{1024 - n} * kH * 4);
        src = kx.data();
      }
      auto t1 = Clock::now();
      rk.EncodeAAtbRaw(src, 1024, kH, L.kq_a.ah);
      std::memcpy(L.kq_b.ah, L.kq_a.ah, std::size_t{1024} * kH / 8 * 9);
      L.kq_a.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      L.kq_b.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      tim.enc += Ms(t1);
      t1 = Clock::now();
      Dispatch(L.kw_a, L.kq_a);
      Dispatch(L.kw_b, L.kq_b);
      Dispatch(L.kw_d, L.kq_d);
      L.kq_d.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      tim.wait += Ms(t1);
      t1 = Clock::now();
      rk.DecodeCAtbRaw(L.kq_d.ch, ky.data(), 1024, kOutN);
      tim.dec += Ms(t1);
      for (std::uint32_t r = 0; r < n; ++r) std::memcpy(out_h + std::size_t{o + r} * kH, &ky[std::size_t{r} * kOutN], kH * 4);
    }
    t_npu += Ms(t0);
  }

  void Mlp(Layer& L, const float* z, std::uint32_t T, float* out_h) {
    if (kd && L.fused) {
      MlpFused(L, z, T, out_h);
      return;
    }
    if (kd) {
      std::vector<float> g(std::size_t{T} * kI), u(std::size_t{T} * kI), o(std::size_t{T} * kOutN);
      KGemm(L.kp_g, L.kw_g, z, T, kH, g.data());
      KGemm(L.kp_u, L.kw_u, z, T, kH, u.data());
      const auto t1 = Clock::now();
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(g.size()); ++i) g[i] = Silu(g[i]) * u[i];
      t_host += Ms(t1);
      KGemm(L.kp_d, L.kw_d, g.data(), T, kI, o.data());
      for (std::uint32_t t = 0; t < T; ++t) std::memcpy(out_h + std::size_t{t} * kH, &o[std::size_t{t} * kOutN], kH * 4);
      return;
    }
    std::vector<float> yo(std::size_t{kRows} * kOutN);
    const auto t0 = Clock::now();
    for (std::uint32_t o = 0; o < T; o += kRows) {
      const std::uint32_t n = std::min(kRows, T - o);
      rt.Ffn64(L.w_gu, L.p_gu, L.w_dn, L.p_dn, z + std::size_t{o} * kH, n, yo.data(), &tim, kI / kTileK);
      for (std::uint32_t r = 0; r < n; ++r) std::memcpy(out_h + std::size_t{o + r} * kH, &yo[std::size_t{r} * kOutN], kH * 4);
    }
    t_npu += Ms(t0);
  }

  struct Seq {
    const std::int32_t* tok;
    std::uint32_t len, marker;
  };

  void ForwardBatch(const std::vector<Seq>& seqs, std::vector<float>* hidden) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> segs;
    std::uint32_t T = 0;
    for (const auto& q : seqs) {
      segs.push_back({T, q.len});
      T += q.len;
    }
    std::vector<float> x(std::size_t{T} * kH), z(std::size_t{T} * kH), br(std::size_t{T} * kH);
    for (std::size_t i = 0; i < seqs.size(); ++i)
#pragma omp parallel for
      for (std::int64_t t = 0; t < seqs[i].len; ++t)
        for (std::uint32_t c = 0; c < kH; ++c) x[std::size_t(segs[i].first + t) * kH + c] = Bf(embed[std::size_t(seqs[i].tok[t]) * kH + c]);
    for (std::uint32_t l = 0; l < kLayers; ++l) {
      Layer& L = layers[l];
      auto tp = Clock::now();
      RmsNormRows(x.data(), L.ln1.data(), z.data(), T, kH);
      ph[0] += Ms(tp); tp = Clock::now();
      if (L.linear) Gdn(L, z.data(), T, segs, br.data());
      else Attn(L, z.data(), T, segs, br.data());
      ph[1] += Ms(tp); tp = Clock::now();
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(x.size()); ++i) x[i] += br[i];
      RmsNormRows(x.data(), L.ln2.data(), z.data(), T, kH);
      ph[2] += Ms(tp); tp = Clock::now();
      Mlp(L, z.data(), T, br.data());
      ph[3] += Ms(tp); tp = Clock::now();
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(x.size()); ++i) x[i] += br[i];
      ph[2] += Ms(tp);
    }
    hidden->resize(seqs.size() * kH);
    for (std::size_t i = 0; i < seqs.size(); ++i)
      RmsNormRows(&x[std::size_t{segs[i].first + seqs[i].marker} * kH], norm_w.data(), &(*hidden)[i * kH], 1, kH);
  }

  void Forward(const std::int32_t* tok, std::uint32_t T, const std::vector<std::uint32_t>& markers, std::vector<float>* hidden) {
    hidden->clear();
    for (auto m : markers) {
      std::vector<float> h;
      ForwardBatch({{tok, T, m}}, &h);
      hidden->insert(hidden->end(), h.begin(), h.end());
    }
  }
};

}  // namespace

int KdTest(const char* programs, std::uint32_t K, std::uint32_t N) {
  NpuRuntime rt;
  std::string err;
  if (!rt.Init(std::string(programs) + "/final_kd.xclbin", &err)) {
    std::fprintf(stderr, "init %s\n", err.c_str());
    return 1;
  }
  const std::uint32_t M = 1024;
  std::vector<float> W(std::size_t{N} * K), x(std::size_t{M} * K), y(std::size_t{M} * N), ref(y.size());
  std::srand(3);
  for (auto& v : W) v = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 0.1f;
  for (auto& v : x) v = std::rand() / static_cast<float>(RAND_MAX) - 0.5f;
  Program p = rt.MakeProgram(K, N, std::string(programs) + "/insts_kd_1024x" + std::to_string(K) + "x" + std::to_string(N) + ".txt", std::size_t{M} * N / 8 * 9, &err, 0, M / 16, 3, 5);
  WeightBuffer wb = rt.PackWeightAtb(W.data(), N, K, 4);
  NpuRuntime::Timings tm{};
  for (int it = 0; it < 3; ++it) {
    const auto t0 = Clock::now();
    rt.GemmAtb(p, wb, x.data(), M, y.data(), &tm);
    std::fprintf(stderr, "run %d: %.1f ms (enc %.1f syncA %.1f wait %.1f dec %.1f)\n", it, Ms(t0), tm.enc, tm.syncA, tm.wait, tm.dec);
    tm = {};
  }
#pragma omp parallel for
  for (std::int64_t r = 0; r < M; ++r)
    for (std::uint32_t n = 0; n < N; ++n) {
      float s = 0;
      for (std::uint32_t k = 0; k < K; ++k) s += x[std::size_t(r) * K + k] * W[std::size_t{n} * K + k];
      ref[std::size_t(r) * N + n] = s;
    }
  double num = 0, den = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    num += (y[i] - ref[i]) * (y[i] - ref[i]);
    den += ref[i] * ref[i];
  }
  std::printf("kd %ux%u rel_err %.5f\n", K, N, std::sqrt(num / den));
  return 0;
}

int EvalBatch(const char* model, const char* programs, const char* path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::vector<std::int32_t> d(b.size() / 4);
  std::memcpy(d.data(), b.data(), d.size() * 4);
  Engine e;
  if (!e.Init(model, programs)) return 1;
  std::size_t at = 0;
  const std::int32_t cases = d[at++];
  const std::uint32_t group = std::getenv("Q35_GROUP") ? static_cast<std::uint32_t>(std::atoi(std::getenv("Q35_GROUP"))) : 1;
  struct Item { const std::int32_t* tok; std::uint32_t len, marker, cnt; };
  std::vector<Item> items;
  for (std::int32_t c = 0; c < cases; ++c) {
    Item it{nullptr, static_cast<std::uint32_t>(d[at]), static_cast<std::uint32_t>(d[at + 1]), static_cast<std::uint32_t>(d[at + 2])};
    at += 3;
    it.tok = &d[at];
    at += it.len;
    items.push_back(it);
  }
  const int reps = std::getenv("Q35_REPEAT") ? std::atoi(std::getenv("Q35_REPEAT")) : 1;
  auto t0 = Clock::now();
  std::uint64_t total = 0;
  for (int rep = 0; rep < reps; ++rep) {
  if (rep == reps - 1) {
    t0 = Clock::now();
    total = 0;
    e.tim = {};
    e.t_npu = e.t_host = 0;
    for (auto& v : e.ph) v = 0;
    for (auto& v : e.mx) v = 0;
  }
  for (std::size_t g = 0; g < items.size(); g += group) {
    std::vector<Engine::Seq> sq;
    for (std::size_t i = g; i < std::min(items.size(), g + group); ++i) {
      sq.push_back({items[i].tok, items[i].len, items[i].marker});
      total += items[i].len;
    }
    std::vector<float> hid;
    e.ForwardBatch(sq, &hid);
    for (std::size_t i = 0; i < sq.size(); ++i) {
      for (std::uint32_t k = 0; k < items[g + i].cnt; ++k) {
        float s = 0;
        for (std::uint32_t c = 0; c < kH; ++c) s += e.readout[std::size_t{k} * kH + c] * hid[i * kH + c];
        std::printf("%.5f ", s);
      }
      std::printf("\n");
    }
  }
  }
  std::fprintf(stderr, "npu: enc %.0f syncA %.0f submit %.0f wait %.0f syncC %.0f dec %.0f ms\n", e.tim.enc, e.tim.syncA, e.tim.submit, e.tim.wait, e.tim.syncC, e.tim.dec);
  std::fprintf(stderr, "mixer: gdn in %.0f host %.0f out %.0f alloc %.0f | attn in %.0f | ab %.0f conv %.0f gdn-dispatch %.0f\n", e.mx[0], e.mx[1], e.mx[2], e.mx[4], e.mx[3], e.mx[5], e.mx[6], e.mx[7]);
  std::fprintf(stderr, "phases: norm1 %.0f mixer %.0f resid+norm2 %.0f mlp %.0f ms\n", e.ph[0], e.ph[1], e.ph[2], e.ph[3]);
  std::fprintf(stderr, "eval: %d cases %llu tokens %.1f ms (%.1f ms/case, npu %.0f host %.0f) group %u\n", cases, static_cast<unsigned long long>(total), Ms(t0), Ms(t0) / cases, e.t_npu, e.t_host, group);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "kdtest") return KdTest(argv[2], std::atoi(argv[3]), std::atoi(argv[4]));
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
