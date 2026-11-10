#include "src/models/lfm3/npu/engine.hpp"

#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/weights.hpp"
#include "src/npu/model_spec.hpp"
#include "src/npu/runtime/runtime.hpp"

namespace gufo::models::lfm3::npu {
namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

constexpr std::uint32_t kRows = 16;
using gufo::npu::runtime::kQuarterBytes;

inline std::uint16_t Bf16Bits(float f) { return gufo::npu::runtime::Bf16Bits(f); }
inline float Bf16Float(std::uint16_t b) { return gufo::npu::runtime::Bf16Float(b); }

struct Safetensors {
  void* map{nullptr};
  std::size_t size{0};
  std::string header;
  std::size_t data0{0};
  ~Safetensors() { if (map) munmap(map, size); }
  bool Open(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st; fstat(fd, &st);
    size = static_cast<std::size_t>(st.st_size);
    map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { map = nullptr; return false; }
    std::uint64_t n; std::memcpy(&n, map, 8);
    header.assign(static_cast<const char*>(map) + 8, n);
    data0 = 8 + n;
    return true;
  }
  const std::uint16_t* Find(const std::string& name, std::size_t* count) const {
    const std::string key = "\"" + name + "\"";
    auto at = header.find(key);
    if (at == std::string::npos) return nullptr;
    auto off = header.find("\"data_offsets\"", at);
    auto lb = header.find('[', off);
    std::size_t a = 0, b = 0;
    std::sscanf(header.c_str() + lb, "[%zu,%zu]", &a, &b);
    *count = (b - a) / 2;
    return reinterpret_cast<const std::uint16_t*>(
        static_cast<const char*>(map) + data0 + a);
  }
};

void RmsNorm(const float* x, const float* w, float* out, std::uint32_t n,
             std::uint32_t d, float eps, std::size_t stride_in = 0) {
  if (!stride_in) stride_in = d;
#pragma omp parallel for
  for (std::int64_t r = 0; r < n; ++r) {
    const float* xr = x + r * stride_in;
    float sum = 0;
    for (std::uint32_t i = 0; i < d; ++i) sum += xr[i] * xr[i];
    const float scale = 1.0f / std::sqrt(sum / d + eps);
    float* o = out + std::size_t(r) * d;
    for (std::uint32_t i = 0; i < d; ++i) o[i] = xr[i] * scale * w[i];
  }
}

int LogitThreads() {
  static const int n = std::getenv("GUFO_LOGIT_THREADS")
                           ? std::atoi(std::getenv("GUFO_LOGIT_THREADS"))
                           : 16;
  return n;
}
struct QRow { std::vector<std::int8_t> q; std::vector<float> d; };
QRow QuantizeRow(const float* x, std::uint32_t n) {
  QRow r; r.q.resize(n); r.d.resize(n / 32);
  for (std::uint32_t b = 0; b < n / 32; ++b) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
    const float d = amax / 127.0f;
    const float inv = d ? 1.0f / d : 0.0f;
    r.d[b] = d;
    for (int i = 0; i < 32; ++i)
      r.q[b * 32 + i] = static_cast<std::int8_t>(std::nearbyint(x[b * 32 + i] * inv));
  }
  return r;
}
inline float DotRowQ8(const std::uint8_t* row, const QRow& x, std::uint32_t nb) {
  __m512 acc = _mm512_setzero_ps();
  for (std::uint32_t b = 0; b < nb; ++b) {
    const std::uint8_t* blk = row + std::size_t{b} * 34;
    std::uint16_t hd; std::memcpy(&hd, blk, 2);
    const float dw = _cvtsh_ss(hd);
    const __m512i w16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2)));
    const __m512i x16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.q.data() + b * 32)));
    const __m512i prod = _mm512_madd_epi16(w16, x16);
    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(prod), _mm512_set1_ps(dw * x.d[b]), acc);
  }
  return _mm512_reduce_add_ps(acc);
}

void Sweep(const std::uint8_t* base, std::uint32_t V, std::uint32_t H, const float* hidden,
           std::uint32_t rows, std::uint32_t* out) {
  const std::uint32_t nb = H / 32, np = nb / 2;
  const std::size_t rb = std::size_t{nb} * 34;
  std::vector<QRow> xs(rows);
  for (std::uint32_t r = 0; r < rows; ++r) xs[r] = QuantizeRow(hidden + std::size_t{r} * H, H);
  std::vector<std::int8_t> xq(std::size_t{rows} * H);
  std::vector<std::int32_t> corr(std::size_t{rows} * np * 16);
  std::vector<float> xsc(std::size_t{rows} * np * 16);
  for (std::uint32_t r = 0; r < rows; ++r)
    for (std::uint32_t i = 0; i < H; ++i) xq[std::size_t{r} * H + i] = xs[r].q[i];
  for (std::uint32_t r = 0; r < rows; ++r)
    for (std::uint32_t p = 0; p < np; ++p)
      for (int l = 0; l < 16; ++l) {
        int sum = 0;
        for (int j = 0; j < 4; ++j) sum += xs[r].q[p * 64 + l * 4 + j];
        corr[(std::size_t{r} * np + p) * 16 + l] = -128 * sum;
        xsc[(std::size_t{r} * np + p) * 16 + l] = xs[r].d[p * 2 + l / 8];
      }
  std::vector<float> best_v(rows, -INFINITY);
  std::vector<std::uint32_t> best(rows, 0);
  const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
#pragma omp parallel num_threads(LogitThreads())
  {
    std::vector<float> lv(rows, -INFINITY);
    std::vector<std::uint32_t> lb(rows, 0);
#pragma omp for schedule(static) nowait
    for (std::int64_t v = 0; v < V; ++v) {
      const std::uint8_t* row = base + std::size_t(v) * rb;
      __m512 acc[16];
      for (std::uint32_t r = 0; r < rows; ++r) acc[r] = _mm512_setzero_ps();
      for (std::uint32_t p = 0; p < np; ++p) {
        const std::uint8_t* b0 = row + std::size_t{p} * 68;
        std::uint16_t h0, h1; std::memcpy(&h0, b0, 2); std::memcpy(&h1, b0 + 34, 2);
        const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + 2));
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + 36));
        const __m512i wu = _mm512_xor_si512(_mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), flip);
        const __m512 dw = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(_cvtsh_ss(h0))),
                                             _mm256_set1_ps(_cvtsh_ss(h1)), 1);
        for (std::uint32_t r = 0; r < rows; ++r) {
          const std::size_t ri = std::size_t{r} * np + p;
          __m512i d = _mm512_loadu_si512(corr.data() + ri * 16);
          d = _mm512_dpbusd_epi32(d, wu, _mm512_loadu_si512(xq.data() + std::size_t{r} * H + p * 64));
          acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d),
                                   _mm512_mul_ps(dw, _mm512_loadu_ps(xsc.data() + ri * 16)), acc[r]);
        }
      }
      for (std::uint32_t r = 0; r < rows; ++r) {
        const float d = _mm512_reduce_add_ps(acc[r]);
        if (d > lv[r]) { lv[r] = d; lb[r] = static_cast<std::uint32_t>(v); }
      }
    }
#pragma omp critical
    for (std::uint32_t r = 0; r < rows; ++r)
      if (lv[r] > best_v[r] || (lv[r] == best_v[r] && lb[r] < best[r])) { best_v[r] = lv[r]; best[r] = lb[r]; }
  }
  for (std::uint32_t r = 0; r < rows; ++r) out[r] = best[r];
}

}  // namespace

struct Engine::Impl {
  gufo::npu::ModelSpec spec;
  std::unique_ptr<core::GgufReader> reader;
  std::optional<Weights> w;
  Safetensors st;
  bool have_st{false};
  Options opt;
  gufo::npu::runtime::NpuRuntime rt;
  gufo::npu::runtime::Program p_in, p_h, p_qkv, p_ffn;
  std::vector<gufo::npu::runtime::WeightBuffer> m_in, m_out, m_qkv, m_ffn;
  gufo::npu::runtime::Timings tim;
  std::uint32_t H{}, I{}, heads{}, hd{}, kvh{}, qd{}, kvd{}, taps{}, window{};
  std::uint32_t pos{0};
  std::vector<std::vector<float>> gate_w;
  std::vector<std::vector<float>> k_cache, v_cache, conv_hist;
  std::vector<std::uint32_t> capacity;
  std::vector<std::vector<float>> comb_store;
  std::uint32_t pending{0};
  std::uint32_t head_k{0};
  std::vector<std::uint16_t> head_w;
  std::vector<std::uint32_t> sub_ids;
  std::vector<std::uint8_t> sub_table;
  double t_cpu{0};

  void LoadRows(const std::string& st_name, const Tensor& t, float* dst,
                std::uint32_t rows, std::uint32_t K) {
    std::size_t count = 0;
    const std::uint16_t* src = have_st ? st.Find(st_name, &count) : nullptr;
    if (src) {
      if (count != std::size_t{rows} * K)
        throw std::runtime_error("safetensors shape mismatch " + st_name);
#pragma omp parallel for
      for (std::int64_t r = 0; r < rows; ++r)
        for (std::uint32_t k = 0; k < K; ++k)
          dst[std::size_t(r) * K + k] = Bf16Float(src[std::size_t(r) * K + k]);
      return;
    }
#pragma omp parallel for
    for (std::int64_t r = 0; r < rows; ++r)
      DecodeRow(t, static_cast<std::uint32_t>(r),
                std::span<float>(dst + std::size_t(r) * K, K));
  }
};

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Engine::~Engine() {
  if (!impl_) return;
  if (std::getenv("GUFO_NPU_STATS")) {
    const auto& t = impl_->tim;
    std::fprintf(stderr,
                 "npu engine: %d gemm calls: encode %.1f ms, run %.1f ms, decode %.1f ms, cpu %.1f ms\n",
                 t.calls, t.enc, t.syncA + t.submit + t.wait + t.syncC, t.dec, impl_->t_cpu);
    std::fprintf(stderr, "  syncA %.1f ms, submit %.1f ms, wait %.1f ms, syncC %.1f ms\n",
                 t.syncA, t.submit, t.wait, t.syncC);
  }
}

std::unique_ptr<Engine> Engine::Create(const Options& options, std::string* error) {
  try {
    auto im = std::make_unique<Impl>();
    Impl& s = *im;
    s.opt = options;
    if (!options.spec.empty()) {
      auto sp = gufo::npu::ModelSpec::LoadJsonFile(options.spec, error);
      if (!sp) return nullptr;
      s.spec = *sp;
    } else {
      s.spec = gufo::npu::ModelSpec::Lfm3_1_7B();
    }
    s.reader = core::GgufReader::OpenFile(options.gguf, error);
    if (!s.reader) return nullptr;
    s.w = Weights::Bind(*s.reader, error);
    if (!s.w) return nullptr;
    const auto& c = s.w->config;
    s.H = c.hidden; s.I = c.intermediate; s.heads = c.heads; s.hd = c.head_dim;
    s.kvh = 4; s.qd = s.heads * s.hd; s.kvd = s.kvh * s.hd; s.taps = c.conv_width; s.window = c.window;
    // Sync spec from actual model config (spec provides defaults/overrides, config is ground truth for this model)
    s.spec.hidden = s.H; s.spec.layers = static_cast<std::uint32_t>(c.layers.size());
    s.spec.norm_eps = c.eps; s.spec.rope_theta = c.theta; s.spec.rope_theta_swa = c.theta_swa;
    s.spec.eos_id = 7;
    if (auto e = s.reader->GetMetadataUint32("tokenizer.ggml.eos_token_id")) s.spec.eos_id = *e;
    if (!options.bf16.empty()) s.have_st = s.st.Open(options.bf16);

    const std::string xclbin = options.programs + "/final_16x64x128_ffn.xclbin";
    if (!s.rt.Init(xclbin, error)) return nullptr;

    // Programs needed (driven by spec/constraints; LFM3 uses these 4)
    s.p_in = s.rt.MakeProgram(2048, 6144, options.programs + "/insts_plain_2048x6144.txt", 0, error);
    s.p_h = s.rt.MakeProgram(2048, 2048, options.programs + "/insts_plain_2048x2048.txt", 0, error);
    s.p_qkv = s.rt.MakeProgram(2048, 3072, options.programs + "/insts_plain_2048x3072.txt", 0, error);
    s.p_ffn = s.rt.MakeProgram(2048, 2048, options.programs + "/insts_ffn.txt",
                               (std::size_t{120} * 256 + 16 * 256) * 9, error);

    const auto t0 = Clock::now();
    const std::size_t L = c.layers.size();
    s.m_in.resize(L); s.m_out.resize(L); s.m_qkv.resize(L); s.m_ffn.resize(L);
    s.gate_w.resize(L); s.k_cache.resize(L); s.v_cache.resize(L);
    s.conv_hist.resize(L); s.capacity.assign(L, 0); s.comb_store.resize(L);
    std::vector<float> buf;
    for (std::size_t l = 0; l < L; ++l) {
      const auto& lw = s.w->layers[l];
      const std::string pre = "model.layers." + std::to_string(l) + ".";
      if (c.layers[l] == LayerType::kConvolution) {
        buf.resize(std::size_t{6144} * 2048);
        s.LoadRows(pre + "conv.in_proj.weight", lw.in_proj, buf.data(), 6144, 2048);
        s.m_in[l] = s.rt.PackWeight(buf.data(), 6144, 2048); s.rt.Attach(s.m_in[l], s.p_in);
        s.LoadRows(pre + "conv.out_proj.weight", lw.out_proj, buf.data(), 2048, 2048);
        s.m_out[l] = s.rt.PackWeight(buf.data(), 2048, 2048); s.rt.Attach(s.m_out[l], s.p_h);
        s.conv_hist[l].assign(std::size_t{s.taps - 1} * s.H, 0.0f);
      } else {
        buf.resize(std::size_t{3072} * 2048);
        s.LoadRows(pre + "self_attn.q_proj.weight", lw.q, buf.data(), 2048, 2048);
        s.LoadRows(pre + "self_attn.k_proj.weight", lw.k, buf.data() + std::size_t{2048} * 2048, 512, 2048);
        s.LoadRows(pre + "self_attn.v_proj.weight", lw.v, buf.data() + std::size_t{2560} * 2048, 512, 2048);
        s.m_qkv[l] = s.rt.PackWeight(buf.data(), 3072, 2048); s.rt.Attach(s.m_qkv[l], s.p_qkv);
        s.LoadRows(pre + "self_attn.o_proj.weight", lw.attn_out, buf.data(), 2048, 2048);
        s.m_out[l] = s.rt.PackWeight(buf.data(), 2048, 2048); s.rt.Attach(s.m_out[l], s.p_h);
        s.gate_w[l].resize(std::size_t{s.heads} * s.H);
        for (std::uint32_t h = 0; h < s.heads; ++h)
          DecodeRow(lw.attn_gate, h, std::span<float>(s.gate_w[l].data() + std::size_t{h} * s.H, s.H));
        s.capacity[l] = c.layers[l] == LayerType::kSlidingAttention
                            ? std::min(options.context, s.window + kRows)
                            : options.context;
        s.k_cache[l].assign(std::size_t{s.capacity[l]} * s.kvd, 0.0f);
        s.v_cache[l].assign(std::size_t{s.capacity[l]} * s.kvd, 0.0f);
      }
      {
        std::vector<float> gate(std::size_t{7680} * 2048), up(std::size_t{7680} * 2048);
        s.LoadRows(pre + "feed_forward.w1.weight", lw.gate, gate.data(), 7680, 2048);
        s.LoadRows(pre + "feed_forward.w3.weight", lw.up, up.data(), 7680, 2048);
        buf.resize(std::size_t{15360} * 2048);
        for (std::size_t f = 0; f < 7680 / 8; ++f)
          for (std::size_t i = 0; i < 8; ++i) {
            std::memcpy(&buf[(f * 16 + i) * 2048], &gate[(f * 8 + i) * 2048], 2048 * 4);
            std::memcpy(&buf[(f * 16 + 8 + i) * 2048], &up[(f * 8 + i) * 2048], 2048 * 4);
          }
        const std::size_t gu_bytes = std::size_t{120} * 32 * 4 * kQuarterBytes;
        const std::size_t dn_bytes = std::size_t{16} * 120 * 4 * kQuarterBytes;
        gufo::npu::runtime::WeightBuffer m;
        m.bytes = gu_bytes + dn_bytes;
        m.bo = xrt::bo(s.rt.device(), m.bytes, XRT_BO_FLAGS_HOST_ONLY, s.rt.kernel().group_id(3));
        m.host = m.bo.map<std::uint8_t*>();
        std::memset(m.host, 0, m.bytes);
        gufo::npu::runtime::PackMatInt5(buf.data(), 15360, 2048, m.host);
        buf.resize(std::size_t{2048} * 7680);
        s.LoadRows(pre + "feed_forward.w2.weight", lw.down, buf.data(), 2048, 7680);
        gufo::npu::runtime::PackMatInt5(buf.data(), 2048, 7680, m.host + gu_bytes);
        m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.m_ffn[l] = std::move(m);
        s.rt.Attach(s.m_ffn[l], s.p_ffn);
      }
    }
    std::fprintf(stderr, "npu engine: packed %zu layers in %.1f s (spec: %s)\n", L,
                 Ms(t0) / 1000.0, s.spec.name.c_str());
    return std::unique_ptr<Engine>(new Engine(std::move(im)));
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
}

void Engine::Reset() {
  Impl& s = *impl_; s.pos = 0;
  for (auto& v : s.conv_hist) std::fill(v.begin(), v.end(), 0.0f);
  for (auto& v : s.k_cache) std::fill(v.begin(), v.end(), 0.0f);
  for (auto& v : s.v_cache) std::fill(v.begin(), v.end(), 0.0f);
}
std::uint32_t Engine::Position() const { return impl_->pos; }

bool Engine::Step(std::span<const std::uint32_t> tokens, std::vector<float>* hidden) {
  Impl& s = *impl_;
  const auto& c = s.w->config;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > kMaxRows || s.pos + n > s.opt.context) return false;
  const std::uint32_t H = s.H, I = s.I, hd = s.hd, heads = s.heads, kvh = s.kvh, kvd = s.kvd, qd = s.qd;
  std::vector<float> x(std::size_t{n} * H), z(x.size()), z2(x.size()), branch(x.size()),
      p(std::size_t{n} * 6144), conv_out(x.size()), qkv(std::size_t{n} * 3072), attn(std::size_t{n} * qd);
  for (std::uint32_t i = 0; i < n; ++i)
    DecodeRow(s.w->embedding, tokens[i], std::span<float>(&x[std::size_t{i} * H], H));
  auto cpu0 = Clock::now();
  for (std::size_t l = 0; l < c.layers.size(); ++l) {
    const auto& lw = s.w->layers[l];
    RmsNorm(x.data(), static_cast<const float*>(lw.norm.data), z.data(), n, H, c.eps);
    s.t_cpu += Ms(cpu0);
    if (c.layers[l] == LayerType::kConvolution) {
      s.rt.Gemm(s.p_in, s.m_in[l], z.data(), n, p.data(), &s.tim);
      cpu0 = Clock::now();
      const auto* cw = static_cast<const float*>(lw.conv.data);
      const std::uint32_t T = s.taps;
      std::vector<float> comb(std::size_t{T - 1 + n} * H);
      std::memcpy(comb.data(), s.conv_hist[l].data(), std::size_t{T - 1} * H * 4);
#pragma omp parallel for
      for (std::int64_t t = 0; t < n; ++t)
        for (std::uint32_t ch = 0; ch < H; ++ch)
          comb[(std::size_t(t) + T - 1) * H + ch] =
              p[std::size_t(t) * 6144 + ch] * p[std::size_t(t) * 6144 + 2 * H + ch];
#pragma omp parallel for
      for (std::int64_t t = 0; t < n; ++t)
        for (std::uint32_t ch = 0; ch < H; ++ch) {
          float sum = 0;
          for (std::uint32_t j = 0; j < T; ++j)
            sum += cw[std::size_t{ch} * T + j] * comb[(std::size_t(t) + j) * H + ch];
          conv_out[std::size_t(t) * H + ch] = sum * p[std::size_t(t) * 6144 + H + ch];
        }
      s.comb_store[l] = std::move(comb);
      s.t_cpu += Ms(cpu0);
      s.rt.Gemm(s.p_h, s.m_out[l], conv_out.data(), n, branch.data(), &s.tim);
    } else {
      s.rt.Gemm(s.p_qkv, s.m_qkv[l], z.data(), n, qkv.data(), &s.tim);
      cpu0 = Clock::now();
      const bool swa = c.layers[l] == LayerType::kSlidingAttention;
      const float theta = swa ? c.theta_swa : c.theta;
      const auto* fac = static_cast<const float*>(s.w->rope_freqs.data);
      const auto* qn = static_cast<const float*>(lw.q_norm.data);
      const auto* kn = static_cast<const float*>(lw.k_norm.data);
      std::vector<float> q(std::size_t{n} * qd), k(std::size_t{n} * kvd), v(std::size_t{n} * kvd), gate(std::size_t{n} * heads);
      for (std::uint32_t t = 0; t < n; ++t) {
        std::memcpy(&q[std::size_t{t} * qd], &qkv[std::size_t{t} * 3072], qd * 4);
        std::memcpy(&k[std::size_t{t} * kvd], &qkv[std::size_t{t} * 3072 + qd], kvd * 4);
        std::memcpy(&v[std::size_t{t} * kvd], &qkv[std::size_t{t} * 3072 + qd + kvd], kvd * 4);
      }
#pragma omp parallel for collapse(2)
      for (std::int64_t t = 0; t < n; ++t)
        for (std::uint32_t h = 0; h < heads; ++h) {
          float sum = 0;
          const float* zr = &z[std::size_t(t) * H];
          const float* gw = &s.gate_w[l][std::size_t{h} * H];
          for (std::uint32_t i = 0; i < H; ++i) sum += gw[i] * zr[i];
          gate[std::size_t(t) * heads + h] = sum;
        }
      for (std::uint32_t t = 0; t < n; ++t) {
        float cs[64], sn[64];
        {
          const float position = static_cast<float>(s.pos + t);
          for (std::uint32_t i = 0; i < hd / 2; ++i) {
            float angle = position * std::pow(theta, -2.0f * static_cast<float>(i) / hd);
            if (!swa) angle /= fac[i];
            cs[i] = std::cos(angle); sn[i] = std::sin(angle);
          }
        }
        for (std::uint32_t h = 0; h < heads + kvh; ++h) {
          float* base = h < heads ? &q[std::size_t{t} * qd + h * hd] : &k[std::size_t{t} * kvd + (h - heads) * hd];
          const float* nw = h < heads ? qn : kn;
          float sum = 0;
          for (std::uint32_t i = 0; i < hd; ++i) sum += base[i] * base[i];
          const float scale = 1.0f / std::sqrt(sum / hd + c.eps);
          for (std::uint32_t i = 0; i < hd; ++i) base[i] = base[i] * scale * nw[i];
          for (std::uint32_t i = 0; i < hd / 2; ++i) {
            const float a = base[i], b = base[i + hd / 2];
            base[i] = a * cs[i] - b * sn[i];
            base[i + hd / 2] = b * cs[i] + a * sn[i];
          }
        }
        const std::uint32_t P = s.pos + t;
        const std::uint32_t slot = P % s.capacity[l];
        std::memcpy(&s.k_cache[l][std::size_t{slot} * kvd], &k[std::size_t{t} * kvd], kvd * 4);
        std::memcpy(&s.v_cache[l][std::size_t{slot} * kvd], &v[std::size_t{t} * kvd], kvd * 4);
      }
      const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
      for (std::int64_t t = 0; t < n; ++t)
        for (std::int64_t h = 0; h < heads; ++h) {
          const std::uint32_t P = s.pos + static_cast<std::uint32_t>(t);
          const std::uint32_t first = swa && P + 1 > s.window ? P + 1 - s.window : 0;
          const std::uint32_t kh = h / (heads / kvh);
          std::vector<float> sc(P + 1 - first);
          float mx = -1e30f;
          const float* qh = &q[std::size_t(t) * qd + h * hd];
          for (std::uint32_t u = first; u <= P; ++u) {
            const float* kr = &s.k_cache[l][std::size_t{u % s.capacity[l]} * kvd + kh * hd];
            float dot = 0;
            for (std::uint32_t d = 0; d < hd; ++d) dot += qh[d] * kr[d];
            sc[u - first] = dot * inv_sqrt; mx = std::max(mx, sc[u - first]);
          }
          float den = 0;
          for (auto& e : sc) { e = std::exp(e - mx); den += e; }
          const float gsig = 1.0f / (1.0f + std::exp(-gate[std::size_t(t) * heads + h]));
          float* o = &attn[std::size_t(t) * qd + h * hd];
          for (std::uint32_t d = 0; d < hd; ++d) o[d] = 0;
          for (std::uint32_t u = first; u <= P; ++u) {
            const float* vr = &s.v_cache[l][std::size_t{u % s.capacity[l]} * kvd + kh * hd];
            const float wgt = sc[u - first] / den;
            for (std::uint32_t d = 0; d < hd; ++d) o[d] += wgt * vr[d];
          }
          for (std::uint32_t d = 0; d < hd; ++d) o[d] *= gsig;
        }
      s.t_cpu += Ms(cpu0);
      s.rt.Gemm(s.p_h, s.m_out[l], attn.data(), n, branch.data(), &s.tim);
    }
    cpu0 = Clock::now();
    if (lw.post_norm.data) {
      RmsNorm(branch.data(), static_cast<const float*>(lw.post_norm.data), z2.data(), n, H, c.eps);
      for (std::size_t i = 0; i < x.size(); ++i) x[i] += z2[i];
    } else {
      for (std::size_t i = 0; i < x.size(); ++i) x[i] += branch[i];
    }
    RmsNorm(x.data(), static_cast<const float*>(lw.ffn_norm.data), z.data(), n, H, c.eps);
    s.t_cpu += Ms(cpu0);
    s.rt.Ffn(s.p_ffn, s.m_ffn[l], z.data(), n, branch.data(), &s.tim);
    cpu0 = Clock::now();
    if (lw.post_ffn_norm.data) {
      RmsNorm(branch.data(), static_cast<const float*>(lw.post_ffn_norm.data), z2.data(), n, H, c.eps);
      for (std::size_t i = 0; i < x.size(); ++i) x[i] += z2[i];
    } else {
      for (std::size_t i = 0; i < x.size(); ++i) x[i] += branch[i];
    }
  }
  if (hidden) {
    hidden->resize(std::size_t{n} * H);
    RmsNorm(x.data(), static_cast<const float*>(s.w->output_norm.data), hidden->data(), n, H, c.eps);
  }
  s.pending = n;
  s.t_cpu += Ms(cpu0);
  return true;
}

void Engine::Commit(std::uint32_t k) {
  Impl& s = *impl_;
  const auto& c = s.w->config;
  k = std::min(k, s.pending);
  for (std::size_t l = 0; l < c.layers.size(); ++l)
    if (c.layers[l] == LayerType::kConvolution)
      std::memcpy(s.conv_hist[l].data(), s.comb_store[l].data() + std::size_t{k} * s.H,
                  std::size_t{s.taps - 1} * s.H * 4);
  s.pos += k; s.pending = 0;
}

void Engine::Logits(const float* hidden_row, std::vector<float>* logits) {
  Impl& s = *impl_;
  const auto& e = s.w->embedding;
  const std::uint32_t V = e.rows, nb = s.H / 32;
  const std::size_t rb = std::size_t{nb} * 34;
  logits->resize(V);
  const QRow x = QuantizeRow(hidden_row, s.H);
  const auto* base = static_cast<const std::uint8_t*>(e.data);
#pragma omp parallel for num_threads(16) schedule(static)
  for (std::int64_t v = 0; v < V; ++v)
    (*logits)[v] = DotRowQ8(base + std::size_t(v) * rb, x, nb);
}

std::uint32_t Engine::Argmax(const float* hidden_row) {
  Impl& s = *impl_;
  const auto& e = s.w->embedding;
  const std::uint32_t V = e.rows, nb = s.H / 32;
  const std::size_t rb = std::size_t{nb} * 34;
  const QRow x = QuantizeRow(hidden_row, s.H);
  const auto* base = static_cast<const std::uint8_t*>(e.data);
  std::uint32_t best = 0; float best_v = -INFINITY;
#pragma omp parallel num_threads(16)
  {
    std::uint32_t lb = 0; float lv = -INFINITY;
#pragma omp for schedule(static) nowait
    for (std::int64_t v = 0; v < V; ++v) {
      const float d = DotRowQ8(base + std::size_t(v) * rb, x, nb);
      if (d > lv) { lv = d; lb = static_cast<std::uint32_t>(v); }
    }
#pragma omp critical
    if (lv > best_v || (lv == best_v && lb < best)) { best_v = lv; best = lb; }
  }
  return best;
}

void Engine::ArgmaxRows(const float* hidden, std::uint32_t rows, std::uint32_t* out) {
  Impl& s = *impl_;
  Sweep(static_cast<const std::uint8_t*>(s.w->embedding.data), s.w->embedding.rows, s.H, hidden, rows, out);
}

bool Engine::LoadHeads(const std::string& path, std::string* error) {
  Impl& s = *impl_;
  std::ifstream f(path, std::ios::binary);
  std::uint32_t k = 0, hd = 0;
  f.read(reinterpret_cast<char*>(&k), 4);
  f.read(reinterpret_cast<char*>(&hd), 4);
  if (!f || hd != s.H || k == 0 || k > kMaxRows) {
    if (error) *error = "bad draft head file " + path;
    return false;
  }
  s.head_w.resize(std::size_t{k} * s.H * s.H);
  f.read(reinterpret_cast<char*>(s.head_w.data()), s.head_w.size() * 2);
  if (!f) { if (error) *error = "short draft head file " + path; return false; }
  s.head_k = k;
  std::ifstream vf(path + ".vocab", std::ios::binary | std::ios::ate);
  if (vf) {
    const std::size_t n = static_cast<std::size_t>(vf.tellg()) / 4;
    s.sub_ids.resize(n);
    vf.seekg(0);
    vf.read(reinterpret_cast<char*>(s.sub_ids.data()), n * 4);
    const std::size_t rb = std::size_t{s.H / 32} * 34;
    s.sub_table.resize(n * rb);
    const auto* base = static_cast<const std::uint8_t*>(s.w->embedding.data);
    for (std::size_t i = 0; i < n; ++i)
      std::memcpy(s.sub_table.data() + i * rb, base + std::size_t{s.sub_ids[i]} * rb, rb);
  }
  return true;
}

std::uint32_t Engine::HeadCount() const { return impl_->head_k; }
void Engine::LimitHeads(std::uint32_t k) { impl_->head_k = std::min(impl_->head_k, k); }

void Engine::Draft(const float* hidden_row, std::uint32_t* out) {
  Impl& s = *impl_;
  const std::uint32_t K = s.head_k, H = s.H;
  std::vector<float> xs(std::size_t{K} * H);
#pragma omp parallel for num_threads(16) collapse(2) schedule(static)
  for (std::int64_t k = 0; k < K; ++k)
    for (std::int64_t o = 0; o < H; ++o) {
      const std::uint16_t* w = s.head_w.data() + (std::size_t(k) * H + o) * H;
      __m512 acc = _mm512_setzero_ps();
      for (std::uint32_t i = 0; i < H; i += 16) {
        const __m512 wf = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i)));
        acc = _mm512_fmadd_ps(wf, _mm512_loadu_ps(hidden_row + i), acc);
      }
      const float y = _mm512_reduce_add_ps(acc);
      xs[std::size_t(k) * H + o] = hidden_row[o] + y / (1.0f + std::exp(-y));
    }
  if (s.sub_ids.empty()) { ArgmaxRows(xs.data(), K, out); return; }
  Sweep(s.sub_table.data(), static_cast<std::uint32_t>(s.sub_ids.size()), H, xs.data(), K, out);
  for (std::uint32_t k = 0; k < K; ++k) out[k] = s.sub_ids[out[k]];
}

}  // namespace gufo::models::lfm3::npu
