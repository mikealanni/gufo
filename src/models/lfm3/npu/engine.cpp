#include "src/models/lfm3/npu/engine.hpp"

#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/weights.hpp"
#include "src/npu/model_spec.hpp"
#include "src/npu/runtime/ffn_atb.hpp"
#include "src/npu/runtime/runtime.hpp"

namespace gufo::models::lfm3::npu {
namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

constexpr std::uint32_t kRows = 16;
using gufo::npu::runtime::kQuarterBytes;

inline std::uint16_t Bf16Bits(float f) {
  return gufo::npu::runtime::Bf16Bits(f);
}
inline float Bf16Float(std::uint16_t b) {
  return gufo::npu::runtime::Bf16Float(b);
}

struct Safetensors {
  void* map{nullptr};
  std::size_t size{0};
  std::string header;
  std::size_t data0{0};
  ~Safetensors() {
    if (map)
      munmap(map, size);
  }
  bool Open(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
      return false;
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
    auto at = header.find(key);
    if (at == std::string::npos)
      return nullptr;
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
  if (!stride_in)
    stride_in = d;
#pragma omp parallel for
  for (std::int64_t r = 0; r < n; ++r) {
    const float* xr = x + r * stride_in;
    float sum = 0;
#pragma omp simd reduction(+ : sum)
    for (std::uint32_t i = 0; i < d; ++i)
      sum += xr[i] * xr[i];
    const float scale = 1.0f / std::sqrt(sum / d + eps);
    float* o = out + std::size_t(r) * d;
#pragma omp simd
    for (std::uint32_t i = 0; i < d; ++i)
      o[i] = xr[i] * scale * w[i];
  }
}

int LogitThreads() {
  static const int n = std::getenv("GUFO_LOGIT_THREADS")
                           ? std::atoi(std::getenv("GUFO_LOGIT_THREADS"))
                           : 16;
  return n;
}
struct QRow {
  std::vector<std::int8_t> q;
  std::vector<float> d;
};
QRow QuantizeRow(const float* x, std::uint32_t n) {
  QRow r;
  r.q.resize(n);
  r.d.resize(n / 32);
  for (std::uint32_t b = 0; b < n / 32; ++b) {
    float amax = 0;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(x[b * 32 + i]));
    const float d = amax / 127.0f;
    const float inv = d ? 1.0f / d : 0.0f;
    r.d[b] = d;
    for (int i = 0; i < 32; ++i)
      r.q[b * 32 + i] =
          static_cast<std::int8_t>(std::nearbyint(x[b * 32 + i] * inv));
  }
  return r;
}
inline float DotRowQ8(const std::uint8_t* row, const QRow& x,
                      std::uint32_t nb) {
  __m512 acc = _mm512_setzero_ps();
  for (std::uint32_t b = 0; b < nb; ++b) {
    const std::uint8_t* blk = row + std::size_t{b} * 34;
    std::uint16_t hd;
    std::memcpy(&hd, blk, 2);
    const float dw = _cvtsh_ss(hd);
    const __m512i w16 = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2)));
    const __m512i x16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x.q.data() + b * 32)));
    const __m512i prod = _mm512_madd_epi16(w16, x16);
    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(prod), _mm512_set1_ps(dw * x.d[b]),
                          acc);
  }
  return _mm512_reduce_add_ps(acc);
}

void Sweep(const std::uint8_t* base, std::uint32_t V, std::uint32_t H,
           const float* hidden, std::uint32_t rows, std::uint32_t* out) {
  const std::uint32_t nb = H / 32, np = nb / 2;
  const std::size_t rb = std::size_t{nb} * 34;
  std::vector<QRow> xs(rows);
  for (std::uint32_t r = 0; r < rows; ++r)
    xs[r] = QuantizeRow(hidden + std::size_t{r} * H, H);
  std::vector<std::int8_t> xq(std::size_t{rows} * H);
  std::vector<std::int32_t> corr(std::size_t{rows} * np * 16);
  std::vector<float> xsc(std::size_t{rows} * np * 16);
  for (std::uint32_t r = 0; r < rows; ++r)
    for (std::uint32_t i = 0; i < H; ++i)
      xq[std::size_t{r} * H + i] = xs[r].q[i];
  for (std::uint32_t r = 0; r < rows; ++r)
    for (std::uint32_t p = 0; p < np; ++p)
      for (int l = 0; l < 16; ++l) {
        int sum = 0;
        for (int j = 0; j < 4; ++j)
          sum += xs[r].q[p * 64 + l * 4 + j];
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
      for (std::uint32_t r = 0; r < rows; ++r)
        acc[r] = _mm512_setzero_ps();
      for (std::uint32_t p = 0; p < np; ++p) {
        const std::uint8_t* b0 = row + std::size_t{p} * 68;
        std::uint16_t h0, h1;
        std::memcpy(&h0, b0, 2);
        std::memcpy(&h1, b0 + 34, 2);
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + 2));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + 36));
        const __m512i wu = _mm512_xor_si512(
            _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), flip);
        const __m512 dw = _mm512_insertf32x8(
            _mm512_castps256_ps512(_mm256_set1_ps(_cvtsh_ss(h0))),
            _mm256_set1_ps(_cvtsh_ss(h1)), 1);
        for (std::uint32_t r = 0; r < rows; ++r) {
          const std::size_t ri = std::size_t{r} * np + p;
          __m512i d = _mm512_loadu_si512(corr.data() + ri * 16);
          d = _mm512_dpbusd_epi32(
              d, wu,
              _mm512_loadu_si512(xq.data() + std::size_t{r} * H + p * 64));
          acc[r] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(d),
              _mm512_mul_ps(dw, _mm512_loadu_ps(xsc.data() + ri * 16)), acc[r]);
        }
      }
      for (std::uint32_t r = 0; r < rows; ++r) {
        const float d = _mm512_reduce_add_ps(acc[r]);
        if (d > lv[r]) {
          lv[r] = d;
          lb[r] = static_cast<std::uint32_t>(v);
        }
      }
    }
#pragma omp critical
    for (std::uint32_t r = 0; r < rows; ++r)
      if (lv[r] > best_v[r] || (lv[r] == best_v[r] && lb[r] < best[r])) {
        best_v[r] = lv[r];
        best[r] = lb[r];
      }
  }
  for (std::uint32_t r = 0; r < rows; ++r)
    out[r] = best[r];
}

}  // namespace

struct Engine::Impl {
  gufo::npu::ModelSpec spec;
  std::unique_ptr<core::GgufReader> reader;
  std::optional<Weights> w;
  Safetensors st;
  bool have_st{false};
  Options opt;
  bool devres_ok{false};
  bool dec_on{false};
  bool decision{false};  // decision-only process: only the 512-row chain
                         // weights are packed
  const std::int32_t* dec_groups{nullptr};
  std::unique_ptr<gufo::npu::runtime::AttnDevPrep> dprep;
  std::vector<bool> dprep_layer;
  std::vector<std::vector<xrt::bo>>
      dp_qkv;  // [layer][chunk]: QKV GEMM results kept for the batched prep
  gufo::npu::runtime::NpuRuntime rt;
  gufo::npu::runtime::Program p_conv;
  gufo::npu::runtime::Program p_conv64;
  // WP10: every program binds its weights once (first-bind-sticky stack), so
  // programs are per-layer and weights bind at creation; no rebind happens.
  std::vector<gufo::npu::runtime::Program> pH, pQkv, pFfn, pH64, pQkv64, pGu64,
      pDn64;
  std::vector<gufo::npu::runtime::WeightBuffer>
      m_ffn_dn;      // down proj packed for p_dn64
  bool m64 = false;  // m64 prefill programs available and in use for n > 16
  std::unique_ptr<gufo::npu::runtime::FfnAtbChain>
      atb;  // GUFO_ATB=1: 1024-row device-resident FFN
  std::unique_ptr<gufo::npu::runtime::AttnPrefill>
      apre;  // 128-row prefill attention (full-attention layers)
  std::vector<gufo::npu::runtime::WeightBuffer> m_out, m_qkv, m_ffn;
  std::vector<gufo::npu::runtime::ConvLayer> m_conv;
  std::vector<gufo::npu::runtime::ConvLayer> m_conv64;
  std::vector<gufo::npu::runtime::AttnLayer> m_attn;
  bool npu_attn{false};
  bool int4{false};
  std::string programs_dir;
  gufo::npu::runtime::NpuRuntime::Timings tim;
  std::uint32_t H{}, I{}, heads{}, hd{}, kvh{}, qd{}, kvd{}, taps{}, window{};
  std::uint32_t pos{0};
  std::vector<std::vector<float>> gate_w;
  // WP0: KV cache stored as bf16 to halve memory traffic (read as fp32 via
  // AVX-512 upconvert in attention).
  std::vector<std::vector<std::uint16_t>> k_cache, v_cache;
  std::vector<std::uint32_t> capacity;
  std::uint32_t pending{0};
  std::uint32_t head_k{0};
  std::vector<std::uint16_t> head_w;
  std::vector<std::uint32_t> sub_ids;
  std::vector<std::uint8_t> sub_table;
  // WP4: int5 head on NPU (argmax GEMM + per-strip reduce)
  bool npu_argmax{false};
  gufo::npu::runtime::Program p_argmax;
  gufo::npu::runtime::WeightBuffer m_argmax;
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
  if (!impl_)
    return;
  if (std::getenv("GUFO_NPU_STATS")) {
    const auto& t = impl_->tim;
    std::fprintf(stderr,
                 "npu engine: %d gemm calls: encode %.1f ms, run %.1f ms, "
                 "decode %.1f ms, cpu %.1f ms\n",
                 t.calls, t.enc, t.syncA + t.submit + t.wait + t.syncC, t.dec,
                 impl_->t_cpu);
    std::fprintf(
        stderr,
        "  syncA %.1f ms, submit %.1f ms, wait %.1f ms, syncC %.1f ms\n",
        t.syncA, t.submit, t.wait, t.syncC);
    std::fprintf(stderr,
                 "  attn calls %d: append %.1f ms, prep %.1f ms, wait %.1f ms, "
                 "merge %.1f ms\n",
                 t.attn_calls, t.attn_append, t.attn_prep, t.attn_wait,
                 t.attn_merge);
  }
}

std::unique_ptr<Engine> Engine::Create(const Options& options_in,
                                       std::string* error) {
  Options options = options_in;
  const bool int4 = std::getenv("GUFO_LFM3_INT4") != nullptr;
  const std::string pdir = int4 ? options.programs + "/q4" : options.programs;
  try {
    auto im = std::make_unique<Impl>();
    Impl& s = *im;
    s.opt = options;
    if (!options.spec.empty()) {
      auto sp = gufo::npu::ModelSpec::LoadJsonFile(options.spec, error);
      if (!sp)
        return nullptr;
      s.spec = *sp;
    } else {
      s.spec = gufo::npu::ModelSpec::Lfm3_1_7B();
    }
    s.reader = core::GgufReader::OpenFile(options.gguf, error);
    if (!s.reader)
      return nullptr;
    s.w = Weights::Bind(*s.reader, error);
    if (!s.w)
      return nullptr;
    const auto& c = s.w->config;
    s.H = c.hidden;
    s.I = c.intermediate;
    s.heads = c.heads;
    s.hd = c.head_dim;
    s.kvh = 4;
    s.qd = s.heads * s.hd;
    s.kvd = s.kvh * s.hd;
    s.taps = c.conv_width;
    s.window = c.window;
    // Sync spec from actual model config (spec provides defaults/overrides,
    // config is ground truth for this model)
    s.spec.hidden = s.H;
    s.spec.layers = static_cast<std::uint32_t>(c.layers.size());
    s.spec.norm_eps = c.eps;
    s.spec.rope_theta = c.theta;
    s.spec.rope_theta_swa = c.theta_swa;
    s.spec.eos_id = 7;
    if (auto e = s.reader->GetMetadataUint32("tokenizer.ggml.eos_token_id"))
      s.spec.eos_id = *e;
    if (!options.bf16.empty())
      s.have_st = s.st.Open(options.bf16);

    const std::string xclbin = pdir + "/final_16x64x128_ffn.xclbin";
    if (!s.rt.Init(xclbin, error, int4))
      return nullptr;

    // Programs needed (driven by spec/constraints; LFM3 uses these 4)
    s.p_conv = s.rt.MakeProgram(
        2048, 2048, pdir + "/insts_conv.txt",
        std::size_t{64} * gufo::npu::runtime::kStripBytes, error, 64);
    // WP10 m64 prefill programs: rows batch 4x16 -> weight streams amortized
    // 4x.
    const auto have_m64 = [pdir](const char* f) {
      std::ifstream t(pdir + "/" + f);
      return t.good();
    };
    const bool have_conv64 = have_m64("insts_conv_m64.txt");
    if (have_conv64)
      s.p_conv64 = s.rt.MakeProgram(
          2048, 2048, pdir + "/insts_conv_m64.txt",
          std::size_t{64} * 4 * gufo::npu::runtime::kStripBytes, error, 64, 4);
    // 64-row prefill programs (WP10). GUFO_NO_M64=1 falls back to 16-row
    // passes.
    s.m64 = std::getenv("GUFO_NO_M64") == nullptr &&
            have_m64("insts_plain_2048x2048_m64.txt") &&
            have_m64("insts_plain_2048x3072_m64.txt") &&
            have_m64("insts_ffngu64.txt") &&
            have_m64("insts_plain_7680x2048_m64.txt");

    if (std::getenv("GUFO_ATB") != nullptr && s.m64) {
      s.atb = std::make_unique<gufo::npu::runtime::FfnAtbChain>();
      s.atb->SetRows(options.rows);
      std::string aerr;
      if (!s.atb->Init(options.programs, &aerr)) {
        std::fprintf(stderr, "GUFO_ATB init failed: %s\n", aerr.c_str());
        s.atb.reset();
      } else if (!std::getenv("GUFO_NO_ATTNP") && !options.decision) {
        auto ap = std::make_unique<gufo::npu::runtime::AttnPrefill>();
        std::string perr;
        if (ap->Init(options.programs, &perr))
          s.apre = std::move(ap);
        else
          std::fprintf(stderr, "prefill attention unavailable: %s\n",
                       perr.c_str());
      }
    }
    const auto t0 = Clock::now();
    const std::size_t L = c.layers.size();
    s.m_conv.resize(L);
    s.m_conv64.resize(L);
    s.m_attn.resize(L);
    s.m_out.resize(L);
    s.programs_dir = options.programs;
    s.int4 = int4;
    s.decision = options.decision;
    s.npu_attn = !options.decision && (!int4 || std::getenv("GUFO_ATB")) &&
                 !std::getenv("GUFO_CPU_ATTN") && options.context >= 1;
    s.m_qkv.resize(L);
    s.m_ffn.resize(L);
    s.m_ffn_dn.resize(L);
    s.gate_w.resize(L);
    s.k_cache.resize(L);
    s.v_cache.resize(L);
    s.capacity.assign(L, 0);
    std::vector<float> buf;
    s.pH.resize(L);
    s.pQkv.resize(L);
    s.pFfn.resize(L);
    if (s.m64) {
      s.pH64.resize(L);
      s.pQkv64.resize(L);
      s.pGu64.resize(L);
      s.pDn64.resize(L);
    }
    for (std::size_t l = 0; l < L; ++l) {
      const auto& lw = s.w->layers[l];
      const std::string pre = "model.layers." + std::to_string(l) + ".";
      s.pH[l] = s.rt.MakeProgram(2048, 2048,
                                 pdir + "/insts_plain_2048x2048.txt", 0, error);
      s.pQkv[l] = s.rt.MakeProgram(
          2048, 3072, pdir + "/insts_plain_2048x3072.txt", 0, error);
      s.pFfn[l] = s.rt.MakeProgram(
          2048, 2048, pdir + "/insts_ffn.txt",
          std::size_t{136} * gufo::npu::runtime::kStripBytes, error);
      if (s.m64) {
        s.pH64[l] = s.rt.MakeProgram(2048, 2048,
                                     pdir + "/insts_plain_2048x2048_m64.txt", 0,
                                     error, 0, 4);
        s.pQkv64[l] = s.rt.MakeProgram(2048, 3072,
                                       pdir + "/insts_plain_2048x3072_m64.txt",
                                       0, error, 0, 4);
      }
      if (c.layers[l] == LayerType::kConvolution) {
        buf.resize(std::size_t{6144} * 2048);
        s.LoadRows(pre + "conv.in_proj.weight", lw.in_proj, buf.data(), 6144,
                   2048);
        std::vector<float> wout(std::size_t{2048} * 2048);
        s.LoadRows(pre + "conv.out_proj.weight", lw.out_proj, wout.data(), 2048,
                   2048);
        if (!options.decision)
          s.m_conv[l] =
              s.rt.PackConv(buf.data(), static_cast<const float*>(lw.conv.data),
                            s.taps, wout.data(), 2048, s.p_conv);
        if (have_conv64 && !options.decision)
          s.m_conv64[l] =
              s.rt.PackConv(buf.data(), static_cast<const float*>(lw.conv.data),
                            s.taps, wout.data(), 2048, s.p_conv64);
        if (s.atb) {
          std::string cerr;
          if (!s.atb->AddConv(l, buf.data(), wout.data(), &cerr)) {
            std::fprintf(stderr, "GUFO_ATB conv init failed: %s\n",
                         cerr.c_str());
            s.atb.reset();
          }
        }
      } else {
        buf.resize(std::size_t{3072} * 2048);
        s.LoadRows(pre + "self_attn.q_proj.weight", lw.q, buf.data(), 2048,
                   2048);
        s.LoadRows(pre + "self_attn.k_proj.weight", lw.k,
                   buf.data() + std::size_t{2048} * 2048, 512, 2048);
        s.LoadRows(pre + "self_attn.v_proj.weight", lw.v,
                   buf.data() + std::size_t{2560} * 2048, 512, 2048);
        if (!options.decision)
          s.m_qkv[l] = s.rt.PackWeight(buf.data(), 3072, 2048);
        s.LoadRows(pre + "self_attn.o_proj.weight", lw.attn_out, buf.data(),
                   2048, 2048);
        if (!options.decision)
          s.m_out[l] = s.rt.PackWeight(buf.data(), 2048, 2048);
        if (s.atb) {
          std::string aerr;
          std::vector<float> wqkv(std::size_t{3072} * 2048);
          s.LoadRows(pre + "self_attn.q_proj.weight", lw.q, wqkv.data(), 2048,
                     2048);
          s.LoadRows(pre + "self_attn.k_proj.weight", lw.k,
                     wqkv.data() + std::size_t{2048} * 2048, 512, 2048);
          s.LoadRows(pre + "self_attn.v_proj.weight", lw.v,
                     wqkv.data() + std::size_t{2560} * 2048, 512, 2048);
          if (!s.atb->AddAttn(l, wqkv.data(), buf.data(), &aerr)) {
            std::fprintf(stderr, "GUFO_ATB attn layer %zu: %s\n", l,
                         aerr.c_str());
            s.atb.reset();
          }
        }
        s.gate_w[l].resize(std::size_t{s.heads} * s.H);
        for (std::uint32_t h = 0; h < s.heads; ++h)
          DecodeRow(
              lw.attn_gate, h,
              std::span<float>(s.gate_w[l].data() + std::size_t{h} * s.H, s.H));
        const bool l_swa = c.layers[l] == LayerType::kSlidingAttention;
        // Sliding layers run on the NPU with a 1.5x-window ring cache (384
        // tokens for window 256).
        const std::uint32_t l_ring = l_swa ? s.window + s.window / 2 : 0;
        s.capacity[l] = l_swa ? std::max({s.window + kRows, l_ring,
                                          options.decision ? options.rows : 0u})
                              : options.context;
        if (s.npu_attn && s.kvh == 4 && s.hd == 128 && s.heads == 16 &&
            (c.layers[l] == LayerType::kFullAttention ||
             (l_ring && l_ring % 16 == 0 && options.context >= l_ring)))
          s.m_attn[l] = l_swa ? s.rt.MakeAttnLayer(l_ring, l_ring, s.window)
                              : s.rt.MakeAttnLayer(options.context);
        s.k_cache[l].assign(std::size_t{s.capacity[l]} * s.kvd, 0);
        s.v_cache[l].assign(std::size_t{s.capacity[l]} * s.kvd, 0);
      }
      {
        std::vector<float> gate(std::size_t{7680} * 2048),
            up(std::size_t{7680} * 2048);
        s.LoadRows(pre + "feed_forward.w1.weight", lw.gate, gate.data(), 7680,
                   2048);
        s.LoadRows(pre + "feed_forward.w3.weight", lw.up, up.data(), 7680,
                   2048);
        buf.resize(std::size_t{15360} * 2048);
        for (std::size_t f = 0; f < 7680 / 8; ++f)
          for (std::size_t i = 0; i < 8; ++i) {
            std::memcpy(&buf[(f * 16 + i) * 2048], &gate[(f * 8 + i) * 2048],
                        2048 * 4);
            std::memcpy(&buf[(f * 16 + 8 + i) * 2048], &up[(f * 8 + i) * 2048],
                        2048 * 4);
          }
        const std::size_t gu_bytes =
            std::size_t{120} * 32 * 4 * gufo::npu::runtime::QuarterBytesW();
        const std::size_t dn_bytes =
            std::size_t{16} * 120 * 4 * gufo::npu::runtime::QuarterBytesW();
        gufo::npu::runtime::WeightBuffer m;
        if (!options.decision) {
          m.bytes = gu_bytes + dn_bytes;
          m.bo = xrt::bo(s.rt.device(), m.bytes, XRT_BO_FLAGS_HOST_ONLY,
                         s.rt.kernel().group_id(3));
          m.host = m.bo.map<std::uint8_t*>();
          std::memset(m.host, 0, m.bytes);
          gufo::npu::runtime::PackMatInt5(buf.data(), 15360, 2048, m.host);
        }
        buf.resize(std::size_t{2048} * 7680);
        s.LoadRows(pre + "feed_forward.w2.weight", lw.down, buf.data(), 2048,
                   7680);
        if (!options.decision) {
          gufo::npu::runtime::PackMatInt5(buf.data(), 2048, 7680,
                                          m.host + gu_bytes);
          if (s.m64)
            s.m_ffn_dn[l] = s.rt.PackWeight(buf.data(), 2048, 7680);
        }
        if (s.atb) {
          std::string aerr;
          if (!s.atb->AddLayer(gate.data(), up.data(), buf.data(), &aerr)) {
            std::fprintf(stderr, "GUFO_ATB layer %zu: %s\n", l, aerr.c_str());
            s.atb.reset();
          }
        }
        if (!options.decision)
          m.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (s.m64 && !options.decision) {
          s.pGu64[l] = s.rt.MakeProgram(
              2048, 15360, pdir + "/insts_ffngu64.txt", 0, error, 0, 4);
          s.pDn64[l] = s.rt.MakeProgram(7680, 2048,
                                        pdir + "/insts_plain_7680x2048_m64.txt",
                                        0, error, 0, 4);
        }
        s.m_ffn[l] = std::move(m);
        (void)0;
      }
    }
    if (s.atb && std::getenv("GUFO_DEVRES")) {
      bool ok = s.apre != nullptr && !s.w->layers.empty();
      std::vector<const float*> ga, gf, pa, pf;
      for (std::size_t l = 0; l < L && ok; ++l) {
        const auto& lw = s.w->layers[l];
        ok = ok && lw.norm.data && lw.ffn_norm.data && lw.post_norm.data &&
             lw.post_ffn_norm.data && s.atb->Layers() == L;
        if (c.layers[l] == LayerType::kConvolution)
          ok =
              ok && s.atb->HasConvDev(l) && s.taps == 7 && s.m_conv64[l].hidden;
        else
          ok = ok && s.atb->HasAttn(l) && s.m_attn[l].host;
        ga.push_back(static_cast<const float*>(lw.norm.data));
        gf.push_back(static_cast<const float*>(lw.ffn_norm.data));
        pf.push_back(static_cast<const float*>(lw.post_norm.data));
        pa.push_back(static_cast<const float*>(
            s.w->layers[l ? l - 1 : 0].post_ffn_norm.data));
      }
      std::string derr;
      if (ok &&
          s.atb->SetupDevRes(
              ga, gf, static_cast<const float*>(s.w->output_norm.data), pa, pf,
              static_cast<const float*>(s.w->layers[L - 1].post_ffn_norm.data),
              c.eps, &derr))
        s.devres_ok = true;
      else
        std::fprintf(stderr, "GUFO_DEVRES unavailable: %s\n", derr.c_str());
    }
    if (s.atb && s.apre && !s.devres_ok &&
        std::getenv("GUFO_DEVPREP") != nullptr) {
      auto dp = std::make_unique<gufo::npu::runtime::AttnDevPrep>();
      std::string perr;
      if (dp->Init(options.programs, s.apre->device(), &perr)) {
        s.dprep_layer.assign(L, false);
        s.dp_qkv.resize(L);
        const auto* fac = static_cast<const float*>(s.w->rope_freqs.data);
        bool all = true;
        for (std::size_t l = 0; l < L; ++l) {
          if (c.layers[l] != LayerType::kFullAttention || !s.m_attn[l].host ||
              s.m_attn[l].ring_tokens)
            continue;
          if (const char* only = std::getenv("GUFO_DEVPREP_ONLY"))
            if (static_cast<std::size_t>(std::atoi(only)) != l)
              continue;
          const auto& lw = s.w->layers[l];
          if (dp->SetupLayer(l, s.capacity[l],
                             static_cast<const float*>(lw.q_norm.data),
                             static_cast<const float*>(lw.k_norm.data), c.eps,
                             c.theta, fac))
            s.dprep_layer[l] = true;
          else
            all = false;
        }
        if (all)
          s.dprep = std::move(dp);
        else
          std::fprintf(stderr, "GUFO_DEVPREP unavailable: layer setup\n");
      } else {
        std::fprintf(stderr, "GUFO_DEVPREP unavailable: %s\n", perr.c_str());
      }
    }
    if (s.atb)
      s.atb->Warm();
    // WP4: pack tied embedding as int5 head for NPU argmax, if program exists.
    s.npu_argmax = false;
    if (std::getenv("GUFO_CPU_ARGMAX") == nullptr && !options.decision) {
      const std::string ainst = pdir + "/insts_argmax_2048x128000.txt";
      std::ifstream af(ainst, std::ios::binary);
      if (af.good()) {
        af.close();
        try {
          const std::uint32_t V = s.w->embedding.rows;
          s.p_argmax = s.rt.MakeProgram(s.H, V, ainst, 0, error);
          std::vector<float> emb(std::size_t{V} * s.H);
          const std::uint32_t nb = s.H / 32;
          const std::size_t rb = std::size_t{nb} * 34;
          const auto* base =
              static_cast<const std::uint8_t*>(s.w->embedding.data);
#pragma omp parallel for schedule(static)
          for (std::int64_t v = 0; v < V; ++v) {
            const std::uint8_t* row = base + std::size_t(v) * rb;
            float* out = emb.data() + std::size_t(v) * s.H;
            for (std::uint32_t b = 0; b < nb; ++b) {
              std::uint16_t hd;
              std::memcpy(&hd, row + std::size_t{b} * 34, 2);
              const float d = _cvtsh_ss(hd);
              for (int i = 0; i < 32; ++i)
                out[b * 32 + i] = static_cast<float>(static_cast<int8_t>(
                                      row[b * 34 + 2 + i])) *
                                  d;
            }
          }
          s.m_argmax = s.rt.PackWeight(emb.data(), V, s.H);
          ;
          s.npu_argmax = true;
          if (std::getenv("GUFO_ARGMAX_SELFTEST")) {
            for (std::uint32_t j : {0u, 1u, 7u, 100u, 1024u, 2047u}) {
              std::vector<float> x(s.H, 0.f);
              x[j] = 4.0f;
              std::uint32_t got = 0;
              s.rt.Argmax(s.p_argmax, s.m_argmax, x.data(), 1, &got, &s.tim);
              std::uint32_t want = 0;
              float wv = -1e30f;
              for (std::uint32_t v = 0; v < V; ++v) {
                const float e = emb[std::size_t{v} * s.H + j];
                if (e > wv) {
                  wv = e;
                  want = v;
                }
              }
              std::fprintf(stderr,
                           "[argmax selftest] onehot dim %u: npu id %u (emb "
                           "%.4f)  expected id %u (emb %.4f)\n",
                           j, got,
                           got < V ? emb[std::size_t{got} * s.H + j] : 0.f,
                           want, wv);
              const std::uint8_t* base = s.p_argmax.ch;
              float v0;
              std::uint32_t lc0;
              std::memcpy(&v0, base, 4);
              std::memcpy(&lc0, base + 4, 4);
              std::fprintf(stderr,
                           "[argmax selftest]   strip0 cr0 row0 raw (%g, %u); "
                           "expected logit scale 4*emb\n",
                           v0, lc0);
            }
          }
          std::fprintf(stderr,
                       "npu engine: packed int5 argmax head (%u x %u)\n", V,
                       s.H);
        } catch (...) {
          s.npu_argmax = false;
        }
      }
    }
    std::fprintf(stderr, "npu engine: packed %zu layers in %.1f s (spec: %s)\n",
                 L, Ms(t0) / 1000.0, s.spec.name.c_str());
    auto eng = std::unique_ptr<Engine>(new Engine(std::move(im)));
    return eng;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}

static std::uint32_t SuperChunks() {
  static const std::uint32_t v =
      std::getenv("GUFO_SUPER")
          ? static_cast<std::uint32_t>(std::atoi(std::getenv("GUFO_SUPER")))
          : 16;
  return v < 1 ? 1 : v;
}

void Engine::Reset() {
  Impl& s = *impl_;
  s.pos = 0;
  for (auto& cl : s.m_conv)
    if (cl.hidden)
      s.rt.ConvReset(cl);
  for (auto& cl : s.m_conv64)
    if (cl.hidden)
      s.rt.ConvReset(cl);
  for (auto& al : s.m_attn)
    if (al.host)
      s.rt.AttnReset(al);
  for (auto& v : s.k_cache)
    std::fill(v.begin(), v.end(), std::uint16_t{0});
  for (auto& v : s.v_cache)
    std::fill(v.begin(), v.end(), std::uint16_t{0});
}
std::uint32_t Engine::Position() const {
  return impl_->pos;
}
std::uint32_t Engine::PrefillRows() const {
  return impl_->atb ? impl_->atb->M() * SuperChunks()
                    : (impl_->m64 ? kMaxRows : 16);
}

static double g_lock_ms;
struct TlEv {
  const char* n;
  std::size_t l;
  double a, b;
};
static std::vector<TlEv> g_tl;
static std::mutex g_tl_mu;
static const bool g_tl_on = std::getenv("GUFO_TL") != nullptr;
static Clock::time_point g_tl0 = Clock::now();
static void TlAdd(const char* n, std::size_t l, Clock::time_point t) {
  if (!g_tl_on)
    return;
  const auto e = Clock::now();
  std::lock_guard<std::mutex> g(g_tl_mu);
  g_tl.push_back(
      {n, l, std::chrono::duration<double, std::milli>(t - g_tl0).count(),
       std::chrono::duration<double, std::milli>(e - g_tl0).count()});
}
struct TlScope {
  const char* n;
  std::size_t l;
  Clock::time_point t;
  TlScope(const char* n_, std::size_t l_) : n(n_), l(l_), t(Clock::now()) {}
  ~TlScope() {
    if (!g_tl_on)
      return;
    const auto e = Clock::now();
    std::lock_guard<std::mutex> g(g_tl_mu);
    g_tl.push_back(
        {n, l, std::chrono::duration<double, std::milli>(t - g_tl0).count(),
         std::chrono::duration<double, std::milli>(e - g_tl0).count()});
  }
};
static double g_busy[4];
static double g_dr_norm, g_dr_zdec, g_dr_init, g_dr_fin, g_dr_qkvdec,
    g_dr_devtot;
static double g_w_conv, g_w_a, g_w_b, g_w_c, g_f_enc, g_f_exe, g_f_dec;
static double g_t_post, g_t_fin, g_t_cin, g_t_cmid, g_t_cout, g_t_norm,
    g_t_prep, g_t_attn_npu, g_t_attn_cpu, g_t_resid, g_t_conv, g_t_qkv,
    g_t_oproj;
bool Engine::Step(std::span<const std::uint32_t> tokens,
                  std::vector<float>* hidden) {
  Impl& s = *impl_;
  const auto& c = s.w->config;
  const std::uint32_t nTot = static_cast<std::uint32_t>(tokens.size());
  // m64 prefill (WP10): 17..64 rows run on the m64 programs; EncodeA zero-pads
  // and DecodeC skips rows >= n, so n need not be a multiple of 16.
  if (nTot == 0 || s.pos + nTot > s.opt.context)
    return false;
  const std::uint32_t max_rows = s.atb ? s.atb->M() * SuperChunks() : 64;
  if (nTot > max_rows || (nTot > 16 && !s.m64))
    return false;
  const std::uint32_t pos0 = s.pos;
  constexpr std::uint32_t kExactAttnCtx = 4096;
  if (s.apre)
    s.apre->SetExact(s.pos + nTot <= kExactAttnCtx ||
                     std::getenv("GUFO_ATTN_EXACT"));
  const std::uint32_t H = s.H, I = s.I, hd = s.hd, heads = s.heads, kvh = s.kvh,
                      kvd = s.kvd, qd = s.qd;
  const bool devres = !s.dec_on && s.devres_ok && nTot > 64;
  std::uint32_t kChunk = s.atb ? s.atb->M() : 1024;
  const std::uint32_t nChunks = (nTot + kChunk - 1) / kChunk;
  if (devres)
    kChunk = (nTot + nChunks - 1) / nChunks;
  if (devres && kChunk <= 64)
    return false;
  struct ChunkState {
    std::vector<float> X, Z, Z2, Br, QKV, ATT, QALL, GATE, KALL, VALL, CIN, CY,
        BXB;
    std::uint32_t n{0}, pos0{0}, row0{0};
    bool attn_done{false};
    bool dev_prep{false};
    bool dev_prepped{false};
  };
  std::vector<ChunkState> css(nChunks);
  std::mutex npu_mu, tick_mu;
  std::condition_variable tick_cv;
  std::vector<std::uint32_t> conv_turn(c.layers.size(), 0),
      app_turn(c.layers.size(), 0);
  for (std::uint32_t ci = 0; ci < nChunks; ++ci) {
    ChunkState& cs = css[ci];
    cs.row0 = ci * kChunk;
    cs.n = std::min(kChunk, nTot - cs.row0);
    cs.pos0 = pos0 + cs.row0;
    cs.X.resize(std::size_t{cs.n} * H);
    cs.Z.resize(cs.X.size());
    cs.Z2.resize(cs.X.size());
    cs.Br.resize(cs.X.size());
    for (std::uint32_t i = 0; i < cs.n; ++i)
      DecodeRow(s.w->embedding, tokens[cs.row0 + i],
                std::span<float>(&cs.X[std::size_t{i} * H], H));
  }
  const bool dp_ok = !s.dec_on && s.dprep && css.back().n > 64;
  const auto dev_prep_chunks = [&](std::size_t l,
                                   std::vector<ChunkState*> chunks) -> bool {
    std::lock_guard<std::mutex> g_(npu_mu);
    for (ChunkState* cs : chunks) {
      if (cs->dev_prepped)
        continue;
      std::string derr;
      const std::uint32_t ci = cs->row0 / kChunk;
      if (!s.dprep->Run(s.m_attn[l], l, s.dp_qkv[l][ci], cs->pos0, cs->n,
                        ci * 8, &derr)) {
        std::fprintf(stderr, "devprep L%zu chunk %u: %s\n", l, ci,
                     derr.c_str());
        return false;
      }
      cs->dev_prepped = true;
    }
    return true;
  };
  auto cpu0 = Clock::now();
  const auto layer_run = [&](ChunkState& cs, int seg, std::size_t l) {
    const std::uint32_t nTot = cs.n;
    const std::uint32_t pos0 = cs.pos0;
    std::uint32_t lpos = pos0;
    const std::uint32_t ci_ = cs.row0 / kChunk;
    auto& X = cs.X;
    auto& Z = cs.Z;
    auto& Z2 = cs.Z2;
    auto& Br = cs.Br;
    auto& QKV = cs.QKV;
    auto& ATT = cs.ATT;
    auto& QALL = cs.QALL;
    auto& GATE = cs.GATE;
    auto& KALL = cs.KALL;
    auto& VALL = cs.VALL;
    auto& CIN = cs.CIN;
    auto& CY = cs.CY;
    auto& BXB = cs.BXB;
    auto cpu0 = Clock::now();
    (void)cpu0;
    const auto& lw = s.w->layers[l];
    const bool big_attn = s.atb && nTot > 64 &&
                          c.layers[l] != LayerType::kConvolution &&
                          s.atb->HasAttn(l);
    static const bool no_bigconv = std::getenv("GUFO_NO_BIGCONV") != nullptr;
    const bool big_conv = s.atb && !no_bigconv && nTot > 64 &&
                          c.layers[l] == LayerType::kConvolution &&
                          s.atb->HasConv(l) &&
                          (s.m_conv64[l].hidden || s.decision);
    const bool defer_any =
        big_attn && ((s.apre && s.m_attn[l].host) || s.dec_on);
    const bool defer_sw = defer_any && s.m_attn[l].ring_tokens;
    const bool defer = defer_any && !defer_sw && !s.dec_on;
    if (defer_any) {
      QALL.resize(std::size_t{nTot} * qd);
      GATE.resize(std::size_t{nTot} * heads);
    }
    if (defer_any)
      KALL.resize(std::size_t{nTot} * kvd);
    if (defer_sw)
      VALL.resize(std::size_t{nTot} * kvd);
    if (seg == 0 || seg == 1) {
      if (big_attn && devres) {
        QKV.resize(std::size_t{nTot} * 3072);
        ATT.resize(std::size_t{nTot} * qd);
        gufo::npu::runtime::FfnAtbChain::Job jq;
        jq.staged = true;
        {
          std::lock_guard<std::mutex> g_(npu_mu);
          const auto tl0 = Clock::now();
          {
            const auto tn = Clock::now();
            s.atb->ExecNorm(l, 0);
            g_dr_norm += Ms(tn);
          }
          s.atb->ExecQkv(l, jq);
          g_lock_ms += Ms(tl0);
        }
        {
          const auto tq = Clock::now();
          s.atb->QkvOut(l, jq);
          s.atb->DecodeJob(jq, nTot, 3072, QKV.data());
          g_dr_qkvdec += Ms(tq);
        }
        {
          const auto tz = Clock::now();
          s.atb->DevResFinal(nTot, Z.data());
          g_dr_zdec += Ms(tz);
        }
      } else if (big_attn) {
        {
          TlScope tl_("norm", l);
          RmsNorm(X.data(), static_cast<const float*>(lw.norm.data), Z.data(),
                  nTot, H, c.eps);
        }
        QKV.resize(std::size_t{nTot} * 3072);
        ATT.resize(std::size_t{nTot} * qd);
        const auto tq0 = Clock::now();
        {
          gufo::npu::runtime::FfnAtbChain::Job jq;
          {
            TlScope tl_("q.enc", l);
            s.atb->EncodeJob(Z.data(), nTot, 2048, jq);
            s.atb->StageQkv(l, jq);
          }
          cs.dev_prep = false;
          cs.dev_prepped = false;
          {
            TlScope tl_("q.exec", l);
            std::lock_guard<std::mutex> g_(npu_mu);
            const auto tl0 = Clock::now();
            if (dp_ok && defer && s.dprep_layer[l] && pos0 % 16 == 0) {
              auto& cv = s.dp_qkv[l];
              if (cv.size() <= ci_)
                cv.resize(ci_ + 1);
              if (!cv[ci_])
                cv[ci_] = s.atb->NewQkvC();
              s.atb->ExecQkvTo(l, jq, cv[ci_]);
              cs.dev_prep = true;
            } else {
              s.atb->ExecQkv(l, jq);
            }
            g_lock_ms += Ms(tl0);
          }
          if (cs.dev_prep)
            s.atb->ReleaseQkv(l, jq);
          else {
            TlScope tl_("q.dec", l);
            s.atb->FinishQkv(l, jq);
            s.atb->DecodeJob(jq, nTot, 3072, QKV.data());
          }
        }
        g_t_qkv += Ms(tq0);
      }
    }
    const auto finish = [&](std::uint32_t o, std::uint32_t n,
                            std::vector<float>& x, std::vector<float>& z,
                            std::vector<float>& z2, const float* branch) {
      TlScope tl_fin_("finish", l);
      const auto tfin0 = Clock::now();
      if (lw.post_norm.data) {
        RmsNorm(branch, static_cast<const float*>(lw.post_norm.data), z2.data(),
                n, H, c.eps);
        for (std::size_t i = 0; i < x.size(); ++i)
          x[i] += z2[i];
      } else {
        for (std::size_t i = 0; i < x.size(); ++i)
          x[i] += branch[i];
      }
      RmsNorm(x.data(), static_cast<const float*>(lw.ffn_norm.data), z.data(),
              n, H, c.eps);
      std::copy(x.begin(), x.end(), X.begin() + std::size_t{o} * H);
      std::copy(z.begin(), z.end(), Z.begin() + std::size_t{o} * H);
      g_t_fin += Ms(tfin0);
    };
    const auto finish_all = [&](const float* branch) {
      const auto tfin0 = Clock::now();
      if (lw.post_norm.data) {
        RmsNorm(branch, static_cast<const float*>(lw.post_norm.data), Z2.data(),
                nTot, H, c.eps);
#pragma omp parallel for
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(X.size()); ++i)
          X[i] += Z2[i];
      } else {
#pragma omp parallel for
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(X.size()); ++i)
          X[i] += branch[i];
      }
      RmsNorm(X.data(), static_cast<const float*>(lw.ffn_norm.data), Z.data(),
              nTot, H, c.eps);
      g_t_fin += Ms(tfin0);
    };
    if (seg == 0 || seg == 1) {
      if (big_conv) {
        const auto tcv0 = Clock::now();
        if (!devres)
          RmsNorm(X.data(), static_cast<const float*>(lw.norm.data), Z.data(),
                  nTot, H, c.eps);
        CIN.resize(std::size_t{nTot} * 6144);
        CY.resize(std::size_t{nTot} * H);
        static const int conv_dev =
            std::getenv("GUFO_CONV_DEV")
                ? std::atoi(std::getenv("GUFO_CONV_DEV"))
                : 1;
        const bool dev_conv = conv_dev && s.atb->HasConvDev(l) && s.taps == 7;
        gufo::npu::runtime::FfnAtbChain::Job jc, jco;
        if (dev_conv && conv_dev == 1 && !devres)
          s.atb->EncodeJob(Z.data(), nTot, 2048, jc);
        if (devres)
          jc.staged = true;
        std::unique_lock<std::mutex> cvlk;
        {
          std::unique_lock<std::mutex> tl(tick_mu);
          tick_cv.wait(tl, [&] { return conv_turn[l] == ci_; });
        }
        if (dev_conv && conv_dev == 1 && !devres)
          s.atb->StageConvIn(l, jc);
        cvlk = std::unique_lock<std::mutex>(npu_mu);
        if (devres) {
          s.atb->ExecNorm(l, 0);
        }
        const auto tcl0 = Clock::now();
        const auto conv_done = [&](const auto& mid) {
          g_lock_ms += Ms(tcl0);
          cvlk.unlock();
          mid();
          {
            std::lock_guard<std::mutex> tl(tick_mu);
            ++conv_turn[l];
          }
          tick_cv.notify_all();
        };
        const auto tcA = Clock::now();
        constexpr std::uint32_t kHist = 6;
        std::vector<float> wrow(6144);
        const auto bx_rows = [&](std::uint32_t first, std::uint16_t* dst) {
          for (std::uint32_t r = 0; r < kHist; ++r) {
            s.atb->ConvWireRow(l, first + r, wrow.data());
            for (std::uint32_t ch = 0; ch < H; ++ch)
              dst[std::size_t{r} * H + ch] =
                  Bf16Bits(wrow[ch] * wrow[2 * H + ch]);
          }
        };
        std::vector<float> devA;
        if (dev_conv && conv_dev == 1) {
          s.atb->ExecConvIn(l, jc);
          g_t_cin += Ms(tcA);
          const auto tcB1 = Clock::now();
          std::vector<std::uint16_t> h0(std::size_t{kHist} * H),
              nh(std::size_t{kHist} * H);
          s.rt.ConvGetHistory(s.m_conv64[l], h0.data());
          bx_rows(nTot - kHist, nh.data());
          s.atb->RunConvDev(l, static_cast<const float*>(lw.conv.data),
                            h0.data());
          s.rt.ConvSetHistory(s.m_conv64[l], nh.data());
          if (s.m_conv[l].hidden)
            s.rt.ConvCopyHistory(s.m_conv[l], s.m_conv64[l]);
          g_t_cmid += Ms(tcB1);
          const auto tcC1 = Clock::now();
          s.atb->ExecConvOut(l, jco, true);
          g_t_cout += Ms(tcC1);
          g_t_conv += Ms(tcv0);
          TlAdd("conv", l, tcv0);
          if (devres) {
            s.atb->ExecNorm(l, 1);
            conv_done([] {});
          } else {
            conv_done([&] { s.atb->FinishConvOut(l, jco); });
            s.atb->DecodeJob(jco, nTot, 2048, Br.data());
            finish_all(Br.data());
          }
        } else {
          s.atb->RunConvIn(l, Z.data(), nTot, CIN.data());
          g_t_cin += Ms(tcA);
          const auto tcB = Clock::now();
          if (dev_conv) {
            std::vector<std::uint16_t> h0(std::size_t{kHist} * H);
            s.rt.ConvGetHistory(s.m_conv64[l], h0.data());
            s.atb->RunConvDev(l, static_cast<const float*>(lw.conv.data),
                              h0.data());
            devA.resize(std::size_t{s.atb ? s.atb->M() : 1024} * H);
            s.atb->ConvDevA(l, devA.data());
          }
          // Temporary host middle: Bx = bf16(B*x), 7-tap causal conv over
          // [history | Bx], y = C * conv.
          std::vector<std::uint16_t> hist(std::size_t{kHist} * H);
          s.rt.ConvGetHistory(s.m_conv64[l], hist.data());
          const auto* taps = static_cast<const float*>(lw.conv.data);
          const std::uint32_t nt = s.taps;
          BXB.resize(std::size_t{nTot + kHist} * H);
          auto& bx = BXB;
          for (std::uint32_t r = 0; r < kHist; ++r)
            for (std::uint32_t ch = 0; ch < H; ++ch)
              bx[std::size_t{r} * H + ch] =
                  Bf16Float(hist[std::size_t{r} * H + ch]);
#pragma omp parallel for
          for (std::int64_t t = 0; t < nTot; ++t) {
            const float* row = &CIN[std::size_t(t) * 6144];
            float* dst = &bx[std::size_t(t + kHist) * H];
            for (std::uint32_t ch = 0; ch < H; ++ch)
              dst[ch] = Bf16Float(Bf16Bits(row[ch] * row[2 * H + ch]));
          }
#pragma omp parallel for
          for (std::int64_t t = 0; t < nTot; ++t) {
            const float* row = &CIN[std::size_t(t) * 6144];
            for (std::uint32_t ch = 0; ch < H; ++ch) {
              float acc = 0;
              for (std::uint32_t j = 0; j < nt; ++j)
                acc += Bf16Float(Bf16Bits(taps[std::size_t{ch} * nt + j])) *
                       bx[std::size_t(t + j) * H + ch];
              CY[std::size_t(t) * H + ch] =
                  row[H + ch] * Bf16Float(Bf16Bits(acc));
            }
          }
          std::vector<std::uint16_t> nh(std::size_t{kHist} * H);
          for (std::uint32_t r = 0; r < kHist; ++r)
            for (std::uint32_t ch = 0; ch < H; ++ch)
              nh[std::size_t{r} * H + ch] =
                  Bf16Bits(bx[std::size_t{nTot + r} * H + ch]);
          s.rt.ConvSetHistory(s.m_conv64[l], nh.data());
          if (s.m_conv[l].hidden)
            s.rt.ConvCopyHistory(s.m_conv[l], s.m_conv64[l]);
          g_t_cmid += Ms(tcB);
          if (!devA.empty()) {
            double num = 0, den = 0, mx = 0;
            for (std::size_t i = 0; i < std::size_t{nTot} * H; ++i) {
              const double d = devA[i] - CY[i];
              num += d * d;
              den += double(CY[i]) * CY[i];
              mx = std::max(mx, std::abs(d));
            }
            static int shown = 0;
            if (shown++ < 12)
              std::fprintf(
                  stderr,
                  "[convdev] L%zu relL2 %.5f maxabs %.4g (ref rms %.4g)\n", l,
                  std::sqrt(num / den), mx,
                  std::sqrt(den / (double(nTot) * H)));
          }
          const auto tcC = Clock::now();
          s.atb->RunConvOut(l, CY.data(), nTot, Br.data());
          g_t_cout += Ms(tcC);
          g_t_conv += Ms(tcv0);
          TlAdd("conv", l, tcv0);
          conv_done([] {});
          finish_all(Br.data());
        }
      }
      const bool ordered_append = (seg == 0 || seg == 1) && defer && !big_conv;
      const std::uint32_t blk = defer_any ? 512u : 64u;
      for (std::uint32_t o = 0; !big_conv && o < nTot; o += blk) {
        const std::uint32_t n = std::min<std::uint32_t>(blk, nTot - o);
        const bool use64 = s.m64 && (n > 16 || nTot > 64);
        lpos = pos0 + o;
        const bool lite = big_attn && defer_any;
        std::vector<float> x, z, z2, branch, qkv, attn;
        if (!lite) {
          x.assign(X.begin() + std::size_t{o} * H,
                   X.begin() + std::size_t{o + n} * H);
          z.resize(x.size());
          z2.resize(x.size());
          branch.resize(x.size());
          qkv.resize(std::size_t{n} * 3072);
          attn.resize(std::size_t{n} * qd);
        }
        const float* zsrc = lite ? Z.data() + std::size_t{o} * H : z.data();
        const float* qkvsrc =
            lite ? QKV.data() + std::size_t{o} * 3072 : qkv.data();
        {
          const auto tn0 = Clock::now();
          if (lite) {
          } else if (big_attn)
            std::copy(Z.begin() + std::size_t{o} * H,
                      Z.begin() + std::size_t{o + n} * H, z.begin());
          else
            RmsNorm(x.data(), static_cast<const float*>(lw.norm.data), z.data(),
                    n, H, c.eps);
          g_t_norm += Ms(tn0);
        }
        s.t_cpu += Ms(cpu0);
        if (c.layers[l] == LayerType::kConvolution) {
          if (use64 && !s.m_conv64.empty() && s.m_conv64[l].hidden) {
            if (std::getenv("GUFO_REFRESH"))
              s.rt.ConvRefresh(s.m_conv64[l]);
            const auto tcv0 = Clock::now();
            s.rt.ConvStep(s.m_conv64[l], z.data(), n, branch.data(), &s.tim);
            g_t_conv += Ms(tcv0);
            if (const char* dd = std::getenv("GUFO_DUMP_DIR")) {
              if (lpos < 192 && l < 4) {
                char fn[512];
                std::snprintf(fn, sizeof fn, "%s/p%u_C%zu.f32", dd, lpos, l);
                if (FILE* f = std::fopen(fn, "wb")) {
                  std::fwrite(branch.data(), 4, branch.size(), f);
                  std::fclose(f);
                }
              }
            }
            if (std::getenv("GUFO_CONV_REPEAT") && l < 6) {
              std::vector<float> b2(branch.size());
              s.rt.ConvStep(s.m_conv64[l], z.data(), n, b2.data(), nullptr);
              int bad = 0;
              std::string rows;
              for (std::uint32_t r = 0; r < n; ++r) {
                bool d = false;
                for (std::uint32_t cc = 0; cc < H; ++cc)
                  if (b2[std::size_t{r} * H + cc] !=
                      branch[std::size_t{r} * H + cc]) {
                    d = true;
                    break;
                  }
                if (d) {
                  ++bad;
                  rows += std::to_string(r) + ",";
                }
              }
              if (bad)
                std::fprintf(stderr,
                             "[repeat] L%zu pos %u: %d rows differ: %s\n", l,
                             lpos, bad, rows.c_str());
            }
            if (std::getenv("GUFO_NANCHK")) {
              int nn = 0, ninf = 0;
              float mx = 0;
              int badrows = 0;
              std::string badlist;
              for (uint32_t r = 0; r < n; ++r) {
                bool br = false;
                for (uint32_t c = 0; c < H; ++c) {
                  float v = branch[size_t(r) * H + c];
                  if (v != v) {
                    nn++;
                    br = true;
                  } else if (v > 1e30 || v < -1e30) {
                    ninf++;
                    br = true;
                  }
                  if (v == v && std::abs(v) > std::abs(mx))
                    mx = v;
                }
                if (br) {
                  badrows++;
                  if (badlist.size() < 120)
                    badlist += std::to_string(r) + ",";
                }
              }
              std::fprintf(stderr,
                           "[conv64] L%zu nnan=%d ninf=%d badrows=%d/%u "
                           "mx=%.4g bad=[%s]\n",
                           l, nn, ninf, badrows, n, mx, badlist.c_str());
            }
            s.rt.ConvCommit(s.m_conv64[l], n, /*settle=*/false);
            if (s.m_conv[l].hidden)
              s.rt.ConvCopyHistory(s.m_conv[l], s.m_conv64[l]);
          } else {
            TlScope tl_conv_("conv", l);
            for (std::uint32_t o = 0; o < n; o += 16) {
              const std::uint32_t m = std::min(16u, n - o);
              s.rt.ConvStep(s.m_conv[l], z.data() + std::size_t{o} * H, m,
                            branch.data() + std::size_t{o} * H, &s.tim);
              if (n > 16 || nTot > 64)
                s.rt.ConvCommit(s.m_conv[l], m, /*settle=*/true);
            }
          }
          if (const char* dd = std::getenv("GUFO_DUMP_DIR")) {
            if (lpos < 192 && l < 4) {
              char fn[512];
              std::snprintf(fn, sizeof fn, "%s/p%u_B%zu.f32", dd, lpos, l);
              if (FILE* f = std::fopen(fn, "wb")) {
                std::fwrite(branch.data(), 4, branch.size(), f);
                std::fclose(f);
              }
            }
          }
          cpu0 = Clock::now();
        } else {
          {
            const auto tq0 = Clock::now();
            if (lite) {
            } else if (big_attn)
              std::copy(QKV.begin() + std::size_t{o} * 3072,
                        QKV.begin() + std::size_t{o + n} * 3072, qkv.begin());
            else {
              TlScope tl_("qkv", l);
              s.rt.Gemm(use64 ? s.pQkv64[l] : s.pQkv[l], s.m_qkv[l], z.data(),
                        n, qkv.data(), &s.tim);
            }
            g_t_qkv += Ms(tq0);
          }
          cpu0 = Clock::now();
          const bool swa = c.layers[l] == LayerType::kSlidingAttention;
          const float theta = swa ? c.theta_swa : c.theta;
          const auto* fac = static_cast<const float*>(s.w->rope_freqs.data);
          const auto* qn = static_cast<const float*>(lw.q_norm.data);
          const auto* kn = static_cast<const float*>(lw.k_norm.data);
          const auto tp0 = Clock::now();
          std::vector<float> q(std::size_t{n} * qd), k(std::size_t{n} * kvd),
              v(std::size_t{n} * kvd), gate(std::size_t{n} * heads);
          const bool dev_l = cs.dev_prep;
          if (!dev_l)
            for (std::uint32_t t = 0; t < n; ++t) {
              std::memcpy(&q[std::size_t{t} * qd],
                          &qkvsrc[std::size_t{t} * 3072], qd * 4);
              std::memcpy(&k[std::size_t{t} * kvd],
                          &qkvsrc[std::size_t{t} * 3072 + qd], kvd * 4);
              std::memcpy(&v[std::size_t{t} * kvd],
                          &qkvsrc[std::size_t{t} * 3072 + qd + kvd], kvd * 4);
            }
#pragma omp parallel for collapse(2)
          for (std::int64_t t = 0; t < n; ++t)
            for (std::uint32_t h = 0; h < heads; ++h) {
              float sum = 0;
              const float* zr = &zsrc[std::size_t(t) * H];
              const float* gw = &s.gate_w[l][std::size_t{h} * H];
              for (std::uint32_t i = 0; i < H; ++i)
                sum += gw[i] * zr[i];
              gate[std::size_t(t) * heads + h] = sum;
            }
          if (!dev_l)
#pragma omp parallel for
            for (std::int64_t t = 0; t < n; ++t) {
              float cs[64], sn[64];
              {
                const float position = static_cast<float>(lpos + t);
                for (std::uint32_t i = 0; i < hd / 2; ++i) {
                  float angle =
                      position *
                      std::pow(theta, -2.0f * static_cast<float>(i) / hd);
                  if (!swa)
                    angle /= fac[i];
                  cs[i] = std::cos(angle);
                  sn[i] = std::sin(angle);
                }
              }
              for (std::uint32_t h = 0; h < heads + kvh; ++h) {
                float* base = h < heads
                                  ? &q[std::size_t{t} * qd + h * hd]
                                  : &k[std::size_t{t} * kvd + (h - heads) * hd];
                const float* nw = h < heads ? qn : kn;
                float sum = 0;
                for (std::uint32_t i = 0; i < hd; ++i)
                  sum += base[i] * base[i];
                const float scale = 1.0f / std::sqrt(sum / hd + c.eps);
                for (std::uint32_t i = 0; i < hd; ++i)
                  base[i] = base[i] * scale * nw[i];
                for (std::uint32_t i = 0; i < hd / 2; ++i) {
                  const float a = base[i], b = base[i + hd / 2];
                  base[i] = a * cs[i] - b * sn[i];
                  base[i + hd / 2] = b * cs[i] + a * sn[i];
                }
              }
              const std::uint32_t P = lpos + t;
              const std::uint32_t slot = P % s.capacity[l];
              if (defer_sw)
                continue;
              // WP0: store KV as bf16 (halves memory traffic).
              for (std::uint32_t i = 0; i < kvd; ++i) {
                s.k_cache[l][std::size_t{slot} * kvd + i] =
                    Bf16Bits(k[std::size_t{t} * kvd + i]);
                s.v_cache[l][std::size_t{slot} * kvd + i] =
                    Bf16Bits(v[std::size_t{t} * kvd + i]);
              }
            }
          g_t_prep += Ms(tp0);
          TlAdd("prep", l, tp0);
          const auto ta0 = Clock::now();
          bool npu_done = false;
          if (defer_any) {
            npu_done = true;
            if (!dev_l) {
              std::copy(k.begin(), k.end(),
                        KALL.begin() + std::size_t{o} * kvd);
              if (defer_sw)
                std::copy(v.begin(), v.end(),
                          VALL.begin() + std::size_t{o} * kvd);
              std::copy(q.begin(), q.end(), QALL.begin() + std::size_t{o} * qd);
            }
            std::copy(gate.begin(), gate.end(),
                      GATE.begin() + std::size_t{o} * heads);
          } else if (s.m_attn[l].host && !s.int4 &&
                     !(s.dec_on && c.layers[l] == LayerType::kFullAttention)) {
            // Attn kernel is m16: one append+run pass per 16-row block.
            npu_done = true;
            TlScope tl_attn_("attn", l);
            // A pass never crosses a 16-token chunk boundary (multi-row passes
            // that do come back wrong).
            for (std::uint32_t o = 0, m = 0; o < n; o += m) {
              m = std::min({16u, n - o, 16u - (lpos + o) % 16u});
              s.rt.AttnAppend(s.m_attn[l], lpos + o, m,
                              k.data() + std::size_t{o} * kvd,
                              s.v_cache[l].data(), &s.tim);
              npu_done =
                  s.rt.AttnRun(
                      s.m_attn[l], lpos + o, m, q.data() + std::size_t{o} * qd,
                      s.k_cache[l].data(), s.v_cache[l].data(), s.programs_dir,
                      attn.data() + std::size_t{o} * qd, &s.tim) &&
                  npu_done;
            }
            if (npu_done) {
              for (std::uint32_t t = 0; t < n; ++t)
                for (std::uint32_t h = 0; h < heads; ++h) {
                  const float g =
                      1.0f /
                      (1.0f + std::exp(-gate[std::size_t(t) * heads + h]));
                  float* o = &attn[std::size_t(t) * qd + h * hd];
                  for (std::uint32_t d = 0; d < hd; ++d)
                    o[d] *= g;
                }
            } else {
              static bool warned = false;
              if (!warned) {
                std::fprintf(stderr, "npu attention fallback (layer %zu)\n", l);
                warned = true;
              }
            }
          }
          g_t_attn_npu += Ms(ta0);
          const auto tc0 = Clock::now();
          static const bool check = std::getenv("GUFO_ATTN_CHECK") != nullptr;
          std::vector<float> attn_npu;
          if (check && npu_done) {
            attn_npu = attn;
            npu_done = false;
          }
          if (!npu_done) {
            const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
            const std::uint32_t cap = s.capacity[l];
            // WP0: bf16 KV + SIMD dot/accum, no per-(token,head) heap
            // allocation. Split-K over context for long contexts: chunked
            // online-softmax partials + reduction.
            struct AttnPartial {
              float m{-1e30f};
              float s{0.0f};
              alignas(64) float o[128]{};
            };
            constexpr std::uint32_t kSplitChunk = 1024;
            constexpr std::uint32_t kSplitMinLen = 4096;
            const std::uint32_t pairs = n * heads;
            // Decide per-(t,h) whether to split.
            std::vector<std::uint32_t> firsts(pairs), lens(pairs),
                chunk_counts(pairs), chunk_off(pairs);
            std::uint32_t total_chunks = 0;
            for (std::uint32_t t = 0; t < n; ++t) {
              const std::uint32_t P = lpos + t;
              const std::uint32_t first =
                  swa && P + 1 > s.window ? P + 1 - s.window : 0;
              const bool dec_g = s.dec_on && !swa;
              const std::uint32_t len = dec_g ? nTot : P + 1 - first;
              for (std::uint32_t h = 0; h < heads; ++h) {
                const std::uint32_t idx = t * heads + h;
                firsts[idx] = first;
                lens[idx] = len;
                const std::uint32_t cc =
                    (len >= kSplitMinLen)
                        ? (len + kSplitChunk - 1) / kSplitChunk
                        : 1;
                chunk_counts[idx] = cc;
                chunk_off[idx] = total_chunks;
                total_chunks += cc;
              }
            }
            std::vector<AttnPartial> partials(total_chunks);
            std::vector<std::uint32_t> task_owner(total_chunks);
            for (std::uint32_t idx = 0, tsk = 0; idx < pairs; ++idx)
              for (std::uint32_t c = 0; c < chunk_counts[idx]; ++c)
                task_owner[tsk++] = idx;
#pragma omp parallel for schedule(dynamic, 1)
            for (std::int64_t task = 0;
                 task < static_cast<std::int64_t>(total_chunks); ++task) {
              const std::uint32_t idx =
                  task_owner[static_cast<std::size_t>(task)];
              const std::uint32_t t = idx / heads, h = idx % heads;
              const std::uint32_t local =
                  static_cast<std::uint32_t>(task) - chunk_off[idx];
              const std::uint32_t first = firsts[idx], len = lens[idx];
              const std::uint32_t ccount = chunk_counts[idx];
              const std::uint32_t chunk_len = (len + ccount - 1) / ccount;
              const std::uint32_t c0 = local * chunk_len;
              const std::uint32_t c1 = std::min(len, c0 + chunk_len);
              const std::uint32_t kh = h / (heads / kvh);
              const float* qh = &q[std::size_t(t) * qd + h * hd];
              AttnPartial& ptl = partials[static_cast<std::size_t>(task)];
              if (ccount == 1) {
                // Single-chunk fast path: one dot pass with score buffer (no
                // recompute).
                std::vector<float> sc(len);
                float mx = -1e30f;
                for (std::uint32_t i = 0; i < len; ++i) {
                  const std::uint32_t u = first + i;
                  if (s.dec_on && !swa && s.dec_groups &&
                      s.dec_groups[t] >= 0 && s.dec_groups[u] >= 0 &&
                      s.dec_groups[u] != s.dec_groups[t]) {
                    sc[i] = -1e30f;
                    continue;
                  }
                  const std::uint16_t* kr =
                      &s.k_cache[l][std::size_t{u % cap} * kvd + kh * hd];
                  __m512 acc = _mm512_setzero_ps();
                  for (std::uint32_t d = 0; d < hd; d += 16) {
                    const __m512 qv = _mm512_loadu_ps(qh + d);
                    const __m256i kb = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(kr + d));
                    const __m512 kv = _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(kb), 16));
                    acc = _mm512_fmadd_ps(qv, kv, acc);
                  }
                  const float dot = _mm512_reduce_add_ps(acc) * inv_sqrt;
                  sc[i] = dot;
                  if (dot > mx)
                    mx = dot;
                }
                float sum = 0.0f;
                for (std::uint32_t i = 0; i < len; ++i) {
                  sc[i] = std::exp(sc[i] - mx);
                  sum += sc[i];
                }
                alignas(64) float acco[128] = {0};
                for (std::uint32_t i = 0; i < len; ++i) {
                  const float w = sc[i];
                  const std::uint32_t u = first + i;
                  const __m512 wv = _mm512_set1_ps(w);
                  const std::uint16_t* vr =
                      &s.v_cache[l][std::size_t{u % cap} * kvd + kh * hd];
                  for (std::uint32_t d = 0; d < hd; d += 16) {
                    __m512 ov = _mm512_loadu_ps(acco + d);
                    const __m256i vb = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(vr + d));
                    const __m512 vv = _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(vb), 16));
                    ov = _mm512_fmadd_ps(wv, vv, ov);
                    _mm512_storeu_ps(acco + d, ov);
                  }
                }
                ptl.m = mx;
                ptl.s = sum;
                for (std::uint32_t d = 0; d < hd; ++d)
                  ptl.o[d] = acco[d];
              } else {
                // Split-K chunk: two-pass (recompute dots) to avoid large score
                // storage.
                float mx = -1e30f;
                for (std::uint32_t i = c0; i < c1; ++i) {
                  const std::uint32_t u = first + i;
                  const std::uint16_t* kr =
                      &s.k_cache[l][std::size_t{u % cap} * kvd + kh * hd];
                  __m512 acc = _mm512_setzero_ps();
                  for (std::uint32_t d = 0; d < hd; d += 16) {
                    const __m512 qv = _mm512_loadu_ps(qh + d);
                    const __m256i kb = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(kr + d));
                    const __m512 kv = _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(kb), 16));
                    acc = _mm512_fmadd_ps(qv, kv, acc);
                  }
                  const float dot = _mm512_reduce_add_ps(acc) * inv_sqrt;
                  if (dot > mx)
                    mx = dot;
                }
                ptl.m = mx;
                float sum = 0.0f;
                alignas(64) float acco[128] = {0};
                for (std::uint32_t i = c0; i < c1; ++i) {
                  const std::uint32_t u = first + i;
                  const std::uint16_t* kr =
                      &s.k_cache[l][std::size_t{u % cap} * kvd + kh * hd];
                  __m512 acc = _mm512_setzero_ps();
                  for (std::uint32_t d = 0; d < hd; d += 16) {
                    const __m512 qv = _mm512_loadu_ps(qh + d);
                    const __m256i kb = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(kr + d));
                    const __m512 kv = _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(kb), 16));
                    acc = _mm512_fmadd_ps(qv, kv, acc);
                  }
                  const float w =
                      std::exp(_mm512_reduce_add_ps(acc) * inv_sqrt - mx);
                  sum += w;
                  const __m512 wv = _mm512_set1_ps(w);
                  const std::uint16_t* vr =
                      &s.v_cache[l][std::size_t{u % cap} * kvd + kh * hd];
                  for (std::uint32_t d = 0; d < hd; d += 16) {
                    __m512 ov = _mm512_loadu_ps(acco + d);
                    const __m256i vb = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(vr + d));
                    const __m512 vv = _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(vb), 16));
                    ov = _mm512_fmadd_ps(wv, vv, ov);
                    _mm512_storeu_ps(acco + d, ov);
                  }
                }
                ptl.s = sum;
                for (std::uint32_t d = 0; d < hd; ++d)
                  ptl.o[d] = acco[d];
              }
            }
            // Reduce partials per (t,h) and apply gate.
#pragma omp parallel for collapse(2) schedule(static)
            for (std::int64_t t = 0; t < n; ++t)
              for (std::int64_t h = 0; h < heads; ++h) {
                const std::uint32_t idx =
                    static_cast<std::uint32_t>(t) * heads +
                    static_cast<std::uint32_t>(h);
                const std::uint32_t base = chunk_off[idx],
                                    ccount = chunk_counts[idx];
                float m = -1e30f;
                for (std::uint32_t c = 0; c < ccount; ++c)
                  m = std::max(m, partials[base + c].m);
                float s = 0.0f;
                alignas(64) float acco[128] = {0};
                for (std::uint32_t c = 0; c < ccount; ++c) {
                  const auto& pt = partials[base + c];
                  const float scale =
                      (pt.s == 0.0f) ? 0.0f : std::exp(pt.m - m);
                  s += pt.s * scale;
                  const __m512 sv = _mm512_set1_ps(scale);
                  for (std::uint32_t d = 0; d < hd; d += 16) {
                    __m512 ov = _mm512_loadu_ps(acco + d);
                    ov = _mm512_fmadd_ps(sv, _mm512_loadu_ps(pt.o + d), ov);
                    _mm512_storeu_ps(acco + d, ov);
                  }
                }
                const float inv_den = s ? 1.0f / s : 0.0f;
                const float gsig =
                    1.0f / (1.0f + std::exp(-gate[std::size_t(t) * heads + h]));
                float* o = &attn[std::size_t(t) * qd + h * hd];
                const __m512 gv = _mm512_set1_ps(gsig * inv_den);
                for (std::uint32_t d = 0; d < hd; d += 16)
                  _mm512_storeu_ps(
                      o + d, _mm512_mul_ps(_mm512_loadu_ps(acco + d), gv));
              }
          }
          g_t_attn_cpu += Ms(tc0);
          if (!attn_npu.empty()) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < std::size_t{n} * qd; ++i) {
              const double d = attn_npu[i] - attn[i];
              num += d * d;
              den += double(attn[i]) * attn[i];
            }
            std::fprintf(stderr,
                         "attn check layer %zu pos %u n %u: rel err %.4f\n", l,
                         lpos, n, std::sqrt(num / den));
          }
          s.t_cpu += Ms(cpu0);
          {
            const auto to0 = Clock::now();
            if (big_attn)
              std::copy(attn.begin(), attn.end(),
                        ATT.begin() + std::size_t{o} * qd);
            else {
              TlScope tl_("oproj", l);
              s.rt.Gemm(use64 ? s.pH64[l] : s.pH[l], s.m_out[l], attn.data(), n,
                        branch.data(), &s.tim);
            }
            g_t_oproj += Ms(to0);
          }
        }
        if (big_attn) {
          cpu0 = Clock::now();
          continue;
        }
        cpu0 = Clock::now();
        finish(o, n, x, z, z2, branch.data());
        s.t_cpu += Ms(cpu0);
      }
      lpos = pos0;
      if (ordered_append) {
        {
          std::unique_lock<std::mutex> tl(tick_mu);
          tick_cv.wait(tl, [&] { return app_turn[l] == ci_; });
        }
        if (!cs.dev_prep)
          for (std::uint32_t o2 = 0; o2 < nTot; o2 += 16)
            s.rt.AttnAppend(s.m_attn[l], pos0 + o2, std::min(16u, nTot - o2),
                            KALL.data() + std::size_t{o2} * kvd,
                            s.v_cache[l].data(), &s.tim);
        {
          std::lock_guard<std::mutex> tl(tick_mu);
          ++app_turn[l];
        }
        tick_cv.notify_all();
      }
    }
    if (seg == 0 || seg == 2) {
      if (defer_any) {
        const auto ta1 = Clock::now();
        bool many_ok = cs.attn_done;
        cs.attn_done = false;
        if (s.dec_on) {
          // d1 decision pass on the host: global layers see every token
          // (bidirectional, masked by question group); sliding layers stay
          // causal inside their window. No attention xclbin is needed.
          const bool swa_layer = c.layers[l] == LayerType::kSlidingAttention;
          const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
          const std::uint32_t cap = s.capacity[l];
#pragma omp parallel for schedule(dynamic, 1)
          for (std::int64_t item = 0; item < std::int64_t{nTot} * heads;
               ++item) {
            const std::uint32_t t = static_cast<std::uint32_t>(item / heads),
                                h = static_cast<std::uint32_t>(item % heads);
            const std::uint32_t kh = h / (heads / kvh);
            const float* qh = &QALL[std::size_t{t} * qd + h * hd];
            const std::int32_t own = s.dec_groups ? s.dec_groups[t] : -1;
            std::vector<float> sc(nTot, -1e30f);
            float mx = -1e30f;
            const std::uint32_t u_first =
                swa_layer && t + 1 > s.window ? t + 1 - s.window : 0;
            const std::uint32_t u_last = swa_layer ? t : nTot - 1;
            for (std::uint32_t u = u_first; u <= u_last; ++u) {
              if (!swa_layer && s.dec_groups &&
                  (((own == INT32_MIN) != (s.dec_groups[u] == INT32_MIN)) ||
                   (own >= 0 && s.dec_groups[u] >= 0 &&
                    s.dec_groups[u] != own))) {
                sc[u] = -1e30f;
                continue;
              }
              const std::uint16_t* kr =
                  &s.k_cache[l][std::size_t{u % cap} * kvd + kh * hd];
              __m512 acc = _mm512_setzero_ps();
              for (std::uint32_t d = 0; d < hd; d += 16) {
                const __m256i kb = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(kr + d));
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(qh + d),
                                      _mm512_castsi512_ps(_mm512_slli_epi32(
                                          _mm512_cvtepu16_epi32(kb), 16)),
                                      acc);
              }
              sc[u] = _mm512_reduce_add_ps(acc) * inv_sqrt;
              mx = std::max(mx, sc[u]);
            }
            float sum = 0.0f;
            for (std::uint32_t u = u_first; u <= u_last; ++u) {
              sc[u] = sc[u] <= -1e29f ? 0.0f : std::exp(sc[u] - mx);
              sum += sc[u];
            }
            alignas(64) float acco[128] = {0};
            for (std::uint32_t u = u_first; u <= u_last; ++u) {
              if (sc[u] == 0.0f)
                continue;
              const __m512 wv = _mm512_set1_ps(sc[u] / sum);
              const std::uint16_t* vr =
                  &s.v_cache[l][std::size_t{u % cap} * kvd + kh * hd];
              for (std::uint32_t d = 0; d < hd; d += 16) {
                const __m256i vb = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(vr + d));
                _mm512_storeu_ps(
                    acco + d,
                    _mm512_fmadd_ps(wv,
                                    _mm512_castsi512_ps(_mm512_slli_epi32(
                                        _mm512_cvtepu16_epi32(vb), 16)),
                                    _mm512_loadu_ps(acco + d)));
              }
            }
            std::memcpy(&ATT[std::size_t{t} * qd + h * hd], acco,
                        hd * sizeof(float));
          }
          many_ok = true;
        }
        if (!many_ok && cs.dev_prep && s.dprep && !dev_prep_chunks(l, {&cs}))
          cs.dev_prep = false;
        if (!many_ok && cs.dev_prep && s.dprep) {
          std::vector<gufo::npu::runtime::AttnPrefill::DevPass> dps;
          for (std::uint32_t o = 0; o < nTot; o += 128)
            dps.push_back({pos0 + o, std::min(128u, nTot - o),
                           ATT.data() + std::size_t{o} * qd,
                           ci_ * 8 + o / 128});
          many_ok = s.apre->RunManyDev(s.m_attn[l], *s.dprep, dps.data(),
                                       dps.size(), &s.tim);
          if (!many_ok)
            std::fprintf(stderr, "devprep attention L%zu failed (%zu passes)\n",
                         l, dps.size());
          cs.dev_prep = false;
        }
        if (const char* dd = std::getenv("GUFO_ATT_DUMP")) {
          static bool done = false;
          if (many_ok && l == 2 && !done) {
            done = true;
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s.att", dd);
            if (FILE* f = std::fopen(fn, "wb")) {
              std::fwrite(ATT.data(), 4, std::size_t{std::min(nTot, 256u)} * qd,
                          f);
              std::fclose(f);
            }
          }
        }
        if (!many_ok) {
          std::vector<gufo::npu::runtime::AttnPrefill::Pass> passes;
          for (std::uint32_t o = 0; o < nTot; o += 128)
            passes.push_back({pos0 + o, std::min(128u, nTot - o),
                              QALL.data() + std::size_t{o} * qd,
                              ATT.data() + std::size_t{o} * qd});
          std::function<void(std::size_t)> app = [&](std::size_t pi) {
            const std::uint32_t o = static_cast<std::uint32_t>(pi) * 128,
                                n2 = std::min(128u, nTot - o);
#pragma omp parallel for
            for (std::int32_t t = 0; t < static_cast<std::int32_t>(n2); ++t) {
              const std::size_t slot = (pos0 + o + t) % s.capacity[l];
              for (std::uint32_t i = 0; i < kvd; ++i) {
                s.k_cache[l][slot * kvd + i] =
                    Bf16Bits(KALL[std::size_t{o + t} * kvd + i]);
                s.v_cache[l][slot * kvd + i] =
                    Bf16Bits(VALL[std::size_t{o + t} * kvd + i]);
              }
            }
#pragma omp parallel for schedule(static, 1)
            for (std::int32_t o2 = 0; o2 < static_cast<std::int32_t>(n2);
                 o2 += 16)
              s.rt.AttnAppend(s.m_attn[l], pos0 + o + o2,
                              std::min(16u, n2 - o2),
                              KALL.data() + std::size_t{o + o2} * kvd,
                              s.v_cache[l].data(), nullptr);
          };
          many_ok = s.apre->RunMany(s.m_attn[l], passes.data(), passes.size(),
                                    s.k_cache[l].data(), s.v_cache[l].data(),
                                    &s.tim, defer_sw ? &app : nullptr);
          if (const char* dd = std::getenv("GUFO_ATT_DUMP")) {
            static bool done2 = false;
            if (many_ok && l == 2 && !done2) {
              done2 = true;
              char fn[512];
              std::snprintf(fn, sizeof fn, "%s.att", dd);
              if (FILE* f = std::fopen(fn, "wb")) {
                std::fwrite(ATT.data(), 4,
                            std::size_t{std::min(nTot, 256u)} * qd, f);
                std::fclose(f);
              }
            }
          }
        }
        for (std::uint32_t o = 0; !many_ok && o < nTot; o += 128) {
          const std::uint32_t n2 = std::min(128u, nTot - o);
          if (defer_sw) {
            for (std::uint32_t t = 0; t < n2; ++t) {
              const std::size_t slot = (pos0 + o + t) % s.capacity[l];
              for (std::uint32_t i = 0; i < kvd; ++i) {
                s.k_cache[l][slot * kvd + i] =
                    Bf16Bits(KALL[std::size_t{o + t} * kvd + i]);
                s.v_cache[l][slot * kvd + i] =
                    Bf16Bits(VALL[std::size_t{o + t} * kvd + i]);
              }
            }
            for (std::uint32_t o2 = 0; o2 < n2; o2 += 16)
              s.rt.AttnAppend(s.m_attn[l], pos0 + o + o2,
                              std::min(16u, n2 - o2),
                              KALL.data() + std::size_t{o + o2} * kvd,
                              s.v_cache[l].data(), &s.tim);
          }
          bool ok = s.apre->Run(s.m_attn[l], pos0 + o, n2,
                                QALL.data() + std::size_t{o} * qd,
                                s.k_cache[l].data(), s.v_cache[l].data(),
                                ATT.data() + std::size_t{o} * qd, &s.tim);
          for (std::uint32_t o2 = 0; !ok && o2 < n2; o2 += 16)
            ok = s.rt.AttnRun(
                s.m_attn[l], pos0 + o + o2, std::min(16u, n2 - o2),
                QALL.data() + std::size_t{o + o2} * qd, s.k_cache[l].data(),
                s.v_cache[l].data(), s.programs_dir,
                ATT.data() + std::size_t{o + o2} * qd, &s.tim);
          if (!ok)
            std::fprintf(stderr,
                         "prefill attention failed (layer %zu pos %u)\n", l,
                         pos0 + o);
        }
#pragma omp parallel for
        for (std::int64_t t = 0; t < nTot; ++t)
          for (std::uint32_t h = 0; h < heads; ++h) {
            const float g =
                1.0f / (1.0f + std::exp(-GATE[std::size_t(t) * heads + h]));
            float* o = &ATT[std::size_t(t) * qd + h * hd];
            for (std::uint32_t d = 0; d < hd; ++d)
              o[d] *= g;
          }
        g_t_attn_npu += Ms(ta1);
        TlAdd("attn", l, ta1);
      }
    }
    if (seg == 0 || seg == 3) {
      if (big_attn) {
        const auto to0 = Clock::now();
        {
          gufo::npu::runtime::FfnAtbChain::Job jo;
          s.atb->EncodeJob(ATT.data(), nTot, 2048, jo);
          s.atb->StageO(l, jo);
          {
            std::lock_guard<std::mutex> g_(npu_mu);
            const auto tl0 = Clock::now();
            s.atb->ExecO(l, jo);
            if (devres)
              s.atb->ExecNorm(l, 1);
            g_lock_ms += Ms(tl0);
          }
          if (devres) {
            s.atb->ReleaseO(l, jo);
          } else {
            s.atb->FinishO(l, jo);
            s.atb->DecodeJob(jo, nTot, 2048, Br.data());
          }
        }
        g_t_oproj += Ms(to0);
        TlAdd("oproj", l, to0);
        if (!devres) {
          TlScope tl_("finish", l);
          finish_all(Br.data());
        }
      }
      if (devres) {
        gufo::npu::runtime::FfnAtbChain::Job jf;
        jf.staged = true;
        {
          std::lock_guard<std::mutex> g_(npu_mu);
          const auto tl0 = Clock::now();
          s.atb->ExecFfn(l, jf);
          g_lock_ms += Ms(tl0);
        }
      } else if (s.atb && nTot > 64) {
        gufo::npu::runtime::FfnAtbChain::Job jf;
        const auto tf0 = Clock::now();
        s.atb->EncodeJob(Z.data(), nTot, 2048, jf);
        const auto tf1 = Clock::now();
        s.atb->StageFfn(l, jf);
        {
          std::lock_guard<std::mutex> g_(npu_mu);
          const auto tl0 = Clock::now();
          s.atb->ExecFfn(l, jf);
          g_lock_ms += Ms(tl0);
        }
        s.atb->FinishFfn(l, jf);
        const auto tf2 = Clock::now();
        s.atb->DecodeJob(jf, nTot, 2048, Br.data());
        TlAdd("ffn.enc", l, tf0);
        g_f_enc += std::chrono::duration<double, std::milli>(tf1 - tf0).count();
        g_f_exe += std::chrono::duration<double, std::milli>(tf2 - tf1).count();
        g_f_dec += Ms(tf2);
      } else {
        std::lock_guard<std::mutex> g_(npu_mu);
        if (s.m64 && nTot > 16)
          s.rt.Ffn64(s.m_ffn[l], s.pGu64[l], s.m_ffn_dn[l], s.pDn64[l],
                     Z.data(), nTot, Br.data(), &s.tim);
        else {
          TlScope tl_("ffn", l);
          s.rt.Ffn(s.pFfn[l], s.m_ffn[l], Z.data(), nTot, Br.data(), &s.tim);
        }
      }
      cpu0 = Clock::now();
      const auto tpost0 = Clock::now();
      if (devres) {
      } else if (lw.post_ffn_norm.data) {
        RmsNorm(Br.data(), static_cast<const float*>(lw.post_ffn_norm.data),
                Z2.data(), nTot, H, c.eps);
#pragma omp parallel for
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(X.size()); ++i)
          X[i] += Z2[i];
      } else {
#pragma omp parallel for
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(X.size()); ++i)
          X[i] += Br[i];
      }
      g_t_post += Ms(tpost0);
      TlAdd("post", l, tpost0);
      if (std::getenv("GUFO_NANCHK")) {
        bool bad = false;
        for (size_t i = 0; i < X.size() && !bad; ++i)
          if (!(X[i] == X[i]))
            bad = true;
        if (bad) {
          std::fprintf(stderr, "[nan] after layer %zu pos=%u\n", l, lpos);
        }
      }
      if (const char* dd = std::getenv("GUFO_DUMP_DIR")) {
        if (lpos < 192) {
          char fn[512];
          std::snprintf(fn, sizeof fn, "%s/p%u_L%zu.f32", dd, lpos, l);
          if (FILE* f = std::fopen(fn, "wb")) {
            std::fwrite(X.data(), 4, X.size(), f);
            std::fclose(f);
          }
        }
      }
    }
  };
  const std::size_t NL = c.layers.size();
  static const bool par_env =
      !(std::getenv("GUFO_PAR") && std::atoi(std::getenv("GUFO_PAR")) == 0);
  bool par_ok = par_env && nChunks > 1 && s.atb;
  for (auto& cs : css)
    par_ok = par_ok && cs.n > 64;
  const auto busy_now = [&] { return g_lock_ms + s.tim.attn_wait; };
  struct Stage {
    std::size_t l;
    int seg;
  };
  std::vector<Stage> stages;
  for (std::size_t l = 0; l < NL; ++l) {
    if (c.layers[l] == LayerType::kConvolution || nChunks == 1)
      stages.push_back({l, 0});
    else {
      stages.push_back({l, 1});
      stages.push_back({l, 2});
      stages.push_back({l, 3});
    }
  }
  const auto run_group = [&](const Stage* st, std::size_t ns) {
    const auto tw = Clock::now();
    const double b0 = busy_now();
    if (!par_ok) {
      for (std::size_t k = 0; k < ns; ++k)
        for (auto& cs : css)
          layer_run(cs, st[k].seg, st[k].l);
    } else {
      const int saved_threads = omp_get_max_threads();
      static const int nw =
          std::getenv("GUFO_PAR_WORKERS")
              ? std::max(2, std::atoi(std::getenv("GUFO_PAR_WORKERS")))
              : 4;
      std::atomic<std::uint32_t> next{0};
      const auto worker = [&] {
        omp_set_num_threads(saved_threads / nw > 0 ? saved_threads / nw : 1);
        for (;;) {
          const std::uint32_t i = next.fetch_add(1);
          if (i >= nChunks)
            break;
          for (std::size_t k = 0; k < ns; ++k)
            layer_run(css[i], st[k].seg, st[k].l);
        }
      };
      std::vector<std::thread> ts;
      for (int w = 1; w < nw; ++w)
        ts.emplace_back(worker);
      worker();
      for (auto& t : ts)
        t.join();
      omp_set_num_threads(saved_threads);
    }
    g_w_conv += Ms(tw);
    g_busy[0] += busy_now() - b0;
  };
  if (devres) {
    if (hidden)
      hidden->resize(std::size_t{nTot} * H);
    for (auto& cs : css) {
      {
        const auto ti = Clock::now();
        s.atb->DevResInit(cs.X.data(), cs.n);
        g_dr_init += Ms(ti);
      }
      for (std::size_t l = 0; l < NL; ++l)
        layer_run(cs, 0, l);
      {
        std::lock_guard<std::mutex> g_(npu_mu);
        s.atb->ExecNorm(NL, 0);
      }
      {
        const auto tf = Clock::now();
        if (hidden)
          s.atb->DevResFinal(cs.n, hidden->data() + std::size_t{cs.row0} * H);
        g_dr_fin += Ms(tf);
      }
    }
  }
  for (std::size_t si = 0; !devres && si < stages.size();) {
    if (stages[si].seg != 2) {
      std::size_t sj = si;
      while (sj < stages.size() && stages[sj].seg != 2 && (par_ok || sj == si))
        ++sj;
      run_group(&stages[si], sj - si);
      si = sj;
      continue;
    }
    const std::size_t l = stages[si].l;
    const auto tw = Clock::now();
    const double b0 = busy_now();
    if (s.apre && s.m_attn[l].host && s.atb->HasAttn(l)) {
      bool eligible = true;
      for (auto& cs : css)
        eligible = eligible && cs.n > 64;
      if (eligible && dp_ok && s.dprep_layer[l]) {
        bool alld = true;
        for (auto& cs : css)
          alld = alld && cs.dev_prep;
        if (alld) {
          std::vector<ChunkState*> cp;
          for (auto& cs : css)
            cp.push_back(&cs);
          alld = dev_prep_chunks(l, cp);
        }
        if (alld) {
          std::vector<gufo::npu::runtime::AttnPrefill::DevPass> dps;
          for (std::uint32_t ci = 0; ci < css.size(); ++ci)
            for (std::uint32_t o = 0; o < css[ci].n; o += 128)
              dps.push_back({css[ci].pos0 + o, std::min(128u, css[ci].n - o),
                             css[ci].ATT.data() + std::size_t{o} * qd,
                             ci * 8 + o / 128});
          const auto tra_ = Clock::now();
          struct RmEnd {
            std::size_t l;
            Clock::time_point t;
            ~RmEnd() { TlAdd("attnrun", l, t); }
          } rm_{l, tra_};
          if (s.apre->RunManyDev(s.m_attn[l], *s.dprep, dps.data(), dps.size(),
                                 &s.tim))
            for (auto& cs : css)
              cs.attn_done = true;
          else
            std::fprintf(stderr, "devprep attention L%zu failed (%zu passes)\n",
                         l, dps.size());
          eligible = false;
        }
      }
      if (eligible) {
        std::vector<gufo::npu::runtime::AttnPrefill::Pass> passes;
        std::vector<std::pair<ChunkState*, std::uint32_t>> owner;
        for (auto& cs : css)
          for (std::uint32_t o = 0; o < cs.n; o += 128) {
            passes.push_back({cs.pos0 + o, std::min(128u, cs.n - o),
                              cs.QALL.data() + std::size_t{o} * qd,
                              cs.ATT.data() + std::size_t{o} * qd});
            owner.push_back({&cs, o});
          }
        const bool ring = s.m_attn[l].ring_tokens != 0;
        std::function<void(std::size_t)> app = [&](std::size_t pi) {
          ChunkState& cs = *owner[pi].first;
          const std::uint32_t o = owner[pi].second,
                              n2 = std::min(128u, cs.n - o);
#pragma omp parallel for
          for (std::int32_t t = 0; t < static_cast<std::int32_t>(n2); ++t) {
            const std::size_t slot = (cs.pos0 + o + t) % s.capacity[l];
            for (std::uint32_t i = 0; i < kvd; ++i) {
              s.k_cache[l][slot * kvd + i] =
                  Bf16Bits(cs.KALL[std::size_t{o + t} * kvd + i]);
              s.v_cache[l][slot * kvd + i] =
                  Bf16Bits(cs.VALL[std::size_t{o + t} * kvd + i]);
            }
          }
#pragma omp parallel for schedule(static, 1)
          for (std::int32_t o2 = 0; o2 < static_cast<std::int32_t>(n2);
               o2 += 16)
            s.rt.AttnAppend(s.m_attn[l], cs.pos0 + o + o2,
                            std::min(16u, n2 - o2),
                            cs.KALL.data() + std::size_t{o + o2} * kvd,
                            s.v_cache[l].data(), nullptr);
        };
        const auto tra_ = Clock::now();
        struct RmEnd {
          std::size_t l;
          Clock::time_point t;
          ~RmEnd() { TlAdd("attnrun", l, t); }
        } rm_{l, tra_};
        if (s.apre->RunMany(s.m_attn[l], passes.data(), passes.size(),
                            s.k_cache[l].data(), s.v_cache[l].data(), &s.tim,
                            ring ? &app : nullptr))
          for (auto& cs : css)
            cs.attn_done = true;
      }
    }
    for (auto& cs : css)
      layer_run(cs, 2, l);
    g_w_b += Ms(tw);
    g_busy[2] += busy_now() - b0;
    ++si;
  }
  if (dp_ok) {
    const std::uint32_t end = pos0 + nTot, lo = end > 48 ? end - 48 : 0;
    for (std::size_t l = 0; l < c.layers.size(); ++l) {
      if (!s.dprep_layer[l])
        continue;
      const std::uint32_t cap = s.capacity[l];
      gufo::npu::runtime::AttnDevPrep::ReadHist(s.m_attn[l], lo, end - lo,
                                                s.k_cache[l].data(),
                                                s.v_cache[l].data(), kvd, cap);
      if (pos0 == 0)
        gufo::npu::runtime::AttnDevPrep::ReadHist(
            s.m_attn[l], 0, std::min(8u, end), s.k_cache[l].data(),
            s.v_cache[l].data(), kvd, cap);
    }
  }
  s.pos = pos0;
  if (hidden && !devres) {
    hidden->resize(std::size_t{nTot} * H);
    for (auto& cs : css)
      RmsNorm(cs.X.data(), static_cast<const float*>(s.w->output_norm.data),
              hidden->data() + std::size_t{cs.row0} * H, cs.n, H, c.eps);
  }
  if (devres && std::getenv("GUFO_ATB_STATS"))
    std::fprintf(stderr,
                 "[devres] norm-under-qkv %.0f zdec %.0f qkvdec %.0f init %.0f "
                 "final %.0f lock %.0f ms\n",
                 g_dr_norm, g_dr_zdec, g_dr_qkvdec, g_dr_init, g_dr_fin,
                 g_lock_ms);
  s.pending = nTot;
  s.t_cpu += Ms(cpu0);
  if (g_tl_on) {
    std::sort(g_tl.begin(), g_tl.end(),
              [](const TlEv& a, const TlEv& b) { return a.a < b.a; });
    for (const auto& e : g_tl)
      if (e.l < 6)
        std::fprintf(stderr, "[tl] L%zu %-8s %7.2f -> %7.2f (%.2f)\n", e.l, e.n,
                     e.a, e.b, e.b - e.a);
    g_tl.clear();
  }
  if (std::getenv("GUFO_ATB_STATS")) {
    std::fprintf(stderr,
                 "[wall] conv-layers %.0f attn A %.0f B %.0f C %.0f ms\n",
                 g_w_conv, g_w_a, g_w_b, g_w_c);
    std::fprintf(stderr,
                 "[busy] device-held: conv %.0f/%.0f  A %.0f/%.0f  B %.0f/%.0f "
                 " C %.0f/%.0f ms (held/wall)\n",
                 g_busy[0], g_w_conv, g_busy[1], g_w_a, g_busy[2], g_w_b,
                 g_busy[3], g_w_c);
    std::fprintf(stderr,
                 "[lock] exec-under-lock %.0f ms; attn run wait %.0f ms; "
                 "conv-dev %.0f ms\n",
                 g_lock_ms, s.tim.attn_wait, s.atb ? s.atb->cvdev_ms : 0.0);
    std::fprintf(stderr, "[ffn] enc %.0f exec %.0f dec %.0f ms\n", g_f_enc,
                 g_f_exe, g_f_dec);
    std::fprintf(stderr,
                 "[conv] in %.0f mid %.0f out %.0f post %.0f fin %.0f ms\n",
                 g_t_cin, g_t_cmid, g_t_cout, g_t_post, g_t_fin);
    if (s.atb)
      std::fprintf(stderr, "[conv] device conv dispatch %.0f ms\n",
                   s.atb->cvdev_ms);
    if (s.atb)
      std::fprintf(stderr,
                   "[conv] split in: enc %.0f dev %.0f dec %.0f | out: enc "
                   "%.0f dev %.0f dec %.0f ms (all conv layers)\n",
                   s.atb->acc_[0][0], s.atb->acc_[0][1], s.atb->acc_[0][2],
                   s.atb->acc_[1][0], s.atb->acc_[1][1], s.atb->acc_[1][2]);
    std::fprintf(stderr,
                 "[cpu] norm %.0f prep %.0f attn_npu %.0f attn_cpu %.0f conv "
                 "%.0f qkv %.0f oproj %.0f ms\n",
                 g_t_norm, g_t_prep, g_t_attn_npu, g_t_attn_cpu, g_t_conv,
                 g_t_qkv, g_t_oproj);
  }
  if (std::getenv("GUFO_ATB_STATS"))
    std::fprintf(stderr,
                 "[step] n %u cpu %.1f ms  npu enc %.1f wait %.1f dec %.1f "
                 "(cumulative)\n",
                 nTot, s.t_cpu, s.tim.enc, s.tim.wait, s.tim.dec);
  if (s.atb && std::getenv("GUFO_ATB_STATS") && s.atb->calls)
    std::fprintf(stderr,
                 "[atb] calls %zu enc %.1f dev %.1f dec %.1f ms | gu_a %.0f "
                 "gu_b %.0f sw %.0f dn %.0f\n",
                 s.atb->calls, s.atb->enc_ms, s.atb->dev_ms, s.atb->dec_ms,
                 s.atb->ph_ms[0], s.atb->ph_ms[1], s.atb->ph_ms[2],
                 s.atb->ph_ms[3]);
  return true;
}

void Engine::Commit(std::uint32_t k) {
  Impl& s = *impl_;
  const auto& c = s.w->config;
  k = std::min(k, s.pending);
  const bool use64conv = s.m64 && k > 16 && !s.m_conv64.empty();
  for (std::size_t l = 0; l < c.layers.size(); ++l)
    if (c.layers[l] == LayerType::kConvolution) {
      if (use64conv && s.m_conv64[l].hidden) {
        // already committed right after its dispatch in Step (a late rewrite
        // right before the next chunk's first dispatch is exactly what races
        // with the device)
        if (std::getenv("GUFO_RECOMMIT"))
          s.rt.ConvCommit(s.m_conv64[l], k);
      } else
        s.rt.ConvCommit(s.m_conv[l], k);
    }
  s.pos += k;
  s.pending = 0;
}

bool Engine::Decide(std::span<const std::uint32_t> tokens,
                    std::span<const std::int32_t> groups,
                    std::span<const std::uint32_t> sequences,
                    std::span<const std::uint32_t> slots,
                    std::span<const std::uint32_t> vocabulary,
                    std::span<float> logits) {
  Impl& s = *impl_;
  if (!s.atb)
    return false;
  const std::uint32_t H = s.H, n = static_cast<std::uint32_t>(tokens.size());
  const std::size_t cols =
      vocabulary.empty() ? s.w->embedding.rows : vocabulary.size();
  if (logits.size() < slots.size() * cols)
    return false;
  std::vector<std::uint32_t> starts(sequences.begin(), sequences.end());
  if (starts.empty())
    starts.push_back(0);
  starts.push_back(n);
  std::vector<float> hidden;
  bool ok = true;
  for (std::size_t q = 0; ok && q + 1 < starts.size(); ++q) {
    const std::uint32_t a = starts[q], b = starts[q + 1];
    Reset();
    s.dec_on = true;
    // A decision-only engine has no 64-row path: short sequences get masked
    // padding tokens at the end (causal layers never look forward, and global
    // attention blocks them) so every request takes the 512-row chain.
    constexpr std::uint32_t kMinRows = 65;
    std::vector<std::uint32_t> padded_tokens;
    std::vector<std::int32_t> padded_groups;
    std::span<const std::uint32_t> seq_tokens = tokens.subspan(a, b - a);
    if (s.decision && b - a < kMinRows) {
      padded_tokens.assign(seq_tokens.begin(), seq_tokens.end());
      padded_tokens.resize(kMinRows,
                           seq_tokens.empty() ? 0u : seq_tokens.back());
      padded_groups.assign(kMinRows, INT32_MIN);
      for (std::uint32_t i = 0; i < b - a; ++i)
        padded_groups[i] = groups.empty() ? 0 : groups[a + i];
      seq_tokens = padded_tokens;
      s.dec_groups = padded_groups.data();
    } else {
      s.dec_groups = groups.empty() ? nullptr : groups.data() + a;
    }
    ok = Step(seq_tokens, &hidden);
    s.dec_on = false;
    s.dec_groups = nullptr;
    if (!ok)
      break;
    for (std::size_t j = 0; j < slots.size(); ++j) {
      if (slots[j] < a || slots[j] >= b)
        continue;
      const float* hrow = &hidden[std::size_t{slots[j] - a} * H];
      std::vector<float> erow(H);
      for (std::size_t v = 0; v < cols; ++v) {
        DecodeRow(
            s.w->embedding,
            vocabulary.empty() ? static_cast<std::uint32_t>(v) : vocabulary[v],
            erow);
        float dot = 0.0f;
        for (std::uint32_t k = 0; k < H; ++k)
          dot += erow[k] * hrow[k];
        logits[j * cols + v] = dot;
      }
    }
  }
  Reset();
  return ok;
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
  if (s.npu_argmax) {
    std::uint32_t out = 0;
    s.rt.Argmax(s.p_argmax, s.m_argmax, hidden_row, 1, &out, &s.tim);
    if (std::getenv("GUFO_ARGMAX_CHECK")) {
      std::vector<float> lg;
      Logits(hidden_row, &lg);
      std::uint32_t best = 0;
      for (std::uint32_t v = 1; v < lg.size(); ++v)
        if (lg[v] > lg[best])
          best = v;
      std::fprintf(stderr,
                   "[argmax] npu %u (cpu logit %.3f) cpu %u (logit %.3f)\n",
                   out, lg[out], best, lg[best]);
      {
        const auto& wb = s.m_argmax;
        std::size_t wnz = 0, wtot = wb.bytes;
        for (std::size_t i = 0; i < wtot; i += 97)
          wnz += wb.host[i] != 0;
        std::fprintf(stderr,
                     "[argmax] weight BO bytes %zu, sampled nonzero %zu of "
                     "%zu; first %02x %02x %02x %02x %02x %02x %02x %02x\n",
                     wtot, wnz, wtot / 97, wb.host[0], wb.host[1], wb.host[2],
                     wb.host[3], wb.host[4], wb.host[5], wb.host[6],
                     wb.host[7]);
      }
      {
        std::size_t nz = 0,
                    total = std::size_t{1000} * gufo::npu::runtime::kStripBytes;
        for (std::size_t i = 0; i < total; ++i)
          nz += s.p_argmax.ch[i] != 0;
        std::uint32_t w4[8];
        std::memcpy(w4, s.p_argmax.ch, 32);
        std::fprintf(stderr,
                     "[argmax] C nonzero bytes %zu of %zu; first words %08x "
                     "%08x %08x %08x %08x %08x %08x %08x\n",
                     nz, total, w4[0], w4[1], w4[2], w4[3], w4[4], w4[5], w4[6],
                     w4[7]);
        std::size_t anz = 0;
        for (std::size_t i = 0; i < s.p_argmax.a.size(); ++i)
          anz += s.p_argmax.ah[i] != 0;
        std::fprintf(stderr, "[argmax] A nonzero bytes %zu of %zu\n", anz,
                     static_cast<std::size_t>(s.p_argmax.a.size()));
      }
      for (std::uint32_t si : {0u, 4u, 4u + 1u, 500u}) {
        for (std::uint32_t cr = 0; cr < 4; ++cr) {
          const std::uint8_t* base =
              s.p_argmax.ch +
              std::size_t{si} * gufo::npu::runtime::kStripBytes +
              std::size_t{cr} * gufo::npu::runtime::kRowBytes;
          float v;
          std::uint32_t lc;
          std::memcpy(&v, base, 4);
          std::memcpy(&lc, base + 4, 4);
          float cm = -1e30f;
          std::uint32_t ci = 0;
          for (std::uint32_t k = 0; k < 32; ++k) {
            const std::uint32_t id = si * 128 + cr * 32 + k;
            if (lg[id] > cm) {
              cm = lg[id];
              ci = k;
            }
          }
          std::fprintf(
              stderr,
              "[argmax] strip %u cr %u: npu (%.3f, %u)  cpu (%.3f, %u)\n", si,
              cr, v, lc, cm, ci);
        }
      }
    }
    return out;
  }
  const auto& e = s.w->embedding;
  const std::uint32_t V = e.rows, nb = s.H / 32;
  const std::size_t rb = std::size_t{nb} * 34;
  const QRow x = QuantizeRow(hidden_row, s.H);
  const auto* base = static_cast<const std::uint8_t*>(e.data);
  std::uint32_t best = 0;
  float best_v = -INFINITY;
#pragma omp parallel num_threads(16)
  {
    std::uint32_t lb = 0;
    float lv = -INFINITY;
#pragma omp for schedule(static) nowait
    for (std::int64_t v = 0; v < V; ++v) {
      const float d = DotRowQ8(base + std::size_t(v) * rb, x, nb);
      if (d > lv) {
        lv = d;
        lb = static_cast<std::uint32_t>(v);
      }
    }
#pragma omp critical
    if (lv > best_v || (lv == best_v && lb < best)) {
      best_v = lv;
      best = lb;
    }
  }
  return best;
}

void Engine::ArgmaxRows(const float* hidden, std::uint32_t rows,
                        std::uint32_t* out) {
  Impl& s = *impl_;
  if (s.npu_argmax) {
    // Run in chunks of 16 (argmax kernel is m16); each call does one NPU
    // dispatch.
    for (std::uint32_t off = 0; off < rows; off += 16) {
      const std::uint32_t n = std::min(16u, rows - off);
      s.rt.Argmax(s.p_argmax, s.m_argmax, hidden + std::size_t{off} * s.H, n,
                  out + off, &s.tim);
    }
    return;
  }
  // Sweep keeps one 16-row accumulator block.
  for (std::uint32_t off = 0; off < rows; off += 16)
    Sweep(static_cast<const std::uint8_t*>(s.w->embedding.data),
          s.w->embedding.rows, s.H, hidden + std::size_t{off} * s.H,
          std::min(16u, rows - off), out + off);
}

bool Engine::LoadHeads(const std::string& path, std::string* error) {
  Impl& s = *impl_;
  std::ifstream f(path, std::ios::binary);
  std::uint32_t k = 0, hd = 0;
  f.read(reinterpret_cast<char*>(&k), 4);
  f.read(reinterpret_cast<char*>(&hd), 4);
  if (!f || hd != s.H || k == 0 || k > kMaxRows) {
    if (error)
      *error = "bad draft head file " + path;
    return false;
  }
  s.head_w.resize(std::size_t{k} * s.H * s.H);
  f.read(reinterpret_cast<char*>(s.head_w.data()), s.head_w.size() * 2);
  if (!f) {
    if (error)
      *error = "short draft head file " + path;
    return false;
  }
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
      std::memcpy(s.sub_table.data() + i * rb,
                  base + std::size_t{s.sub_ids[i]} * rb, rb);
  }
  return true;
}

std::uint32_t Engine::HeadCount() const {
  return impl_->head_k;
}
void Engine::LimitHeads(std::uint32_t k) {
  impl_->head_k = std::min(impl_->head_k, k);
}

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
        const __m512 wf = _mm512_cvtph_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i)));
        acc = _mm512_fmadd_ps(wf, _mm512_loadu_ps(hidden_row + i), acc);
      }
      const float y = _mm512_reduce_add_ps(acc);
      xs[std::size_t(k) * H + o] = hidden_row[o] + y / (1.0f + std::exp(-y));
    }
  if (s.sub_ids.empty()) {
    ArgmaxRows(xs.data(), K, out);
    return;
  }
  // Draft uses the reduced 16384-token vocabulary; use CPU sweep over
  // sub_table.
  Sweep(s.sub_table.data(), static_cast<std::uint32_t>(s.sub_ids.size()), H,
        xs.data(), K, out);
  for (std::uint32_t k = 0; k < K; ++k)
    out[k] = s.sub_ids[out[k]];
}

}  // namespace gufo::models::lfm3::npu
