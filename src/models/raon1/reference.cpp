#include "src/models/raon1/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_map>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::raon1 {
namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;

float Half(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
  const std::uint32_t exp = (h >> 10U) & 0x1FU;
  const std::uint32_t man = h & 0x3FFU;
  std::uint32_t bits = sign;
  if (exp == 0) {
    if (man != 0) {
      // Subnormal half.
      std::uint32_t e = 127U - 15U + 1U;
      std::uint32_t m = man;
      while ((m & 0x400U) == 0) {
        m <<= 1U;
        --e;
      }
      bits |= (e << 23U) | ((m & 0x3FFU) << 13U);
    }
  } else if (exp == 0x1FU) {
    bits |= 0x7F800000U | (man << 13U);
  } else {
    bits |= ((exp + 127U - 15U) << 23U) | (man << 13U);
  }
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}

float Bf16(std::uint16_t h) {
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

std::uint16_t FloatToHalfBits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

struct Q8Block {
  float d{0.0F};
  std::uint16_t dh{0};
  std::int8_t qs[32]{};
};

void QuantizeQ8(const float* x, std::size_t n, std::vector<Q8Block>& out) {
  out.resize(n / 32);
  for (std::size_t b = 0; b < out.size(); ++b) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(x[b * 32 + j]));
    }
    Q8Block& q = out[b];
    q.d = amax / 127.0F;
    q.dh = FloatToHalfBits(q.d);
    const float id = q.d != 0.0F ? 1.0F / q.d : 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      q.qs[j] = static_cast<std::int8_t>(
          std::max(-127.0F, std::min(127.0F, std::round(x[b * 32 + j] * id))));
    }
  }
}

float DotQ5Q8(const void* src, const std::vector<Q8Block>& q, std::size_t k) {
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  float acc = 0.0F;
  for (std::size_t b = 0; b < k / 32; ++b) {
    const auto* blk = bytes + b * 22;
    std::uint16_t dh = 0;
    std::memcpy(&dh, blk, 2);
    std::uint32_t qh = 0;
    std::memcpy(&qh, blk + 2, 4);
    const float d5 = Half(dh);
    const float d8 = Half(q[b].dh);
    const float scale = d5 * d8;
    const std::uint8_t* qs = blk + 6;
    float block = 0.0F;
    for (std::size_t j = 0; j < 16; ++j) {
      const int x0 = static_cast<int>(qs[j] & 0x0FU) |
                     static_cast<int>(((qh >> j) & 1U) << 4U);
      const int x1 = static_cast<int>(qs[j] >> 4U) |
                     static_cast<int>(((qh >> (j + 16)) & 1U) << 4U);
      block += static_cast<float>((x0 - 16) * q[b].qs[j] +
                                  (x1 - 16) * q[b].qs[j + 16]);
    }
    acc += scale * block;
  }
  return acc;
}

// Q5_0: 32-element blocks, F16 scale + 4-byte high bits + 16 packed nibbles.
void DequantizeQ5_0(const void* src, float* dst, std::size_t k) {
  constexpr std::size_t kBlock = 32;
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / kBlock; ++b) {
    const auto* blk = bytes + b * 22;
    std::uint16_t dh = 0;
    std::memcpy(&dh, blk, 2);
    const float d = Half(dh);
    std::uint32_t qh = 0;
    std::memcpy(&qh, blk + 2, 4);
    const std::uint8_t* qs = blk + 6;
    for (std::size_t j = 0; j < 16; ++j) {
      const std::uint32_t x0 = (qs[j] & 0x0FU) | (((qh >> j) & 1U) << 4U);
      const std::uint32_t x1 = (qs[j] >> 4U) | (((qh >> (j + 16)) & 1U) << 4U);
      dst[b * kBlock + 2 * j] = d * (static_cast<float>(x0) - 16.0F);
      dst[b * kBlock + 2 * j + 1] = d * (static_cast<float>(x1) - 16.0F);
    }
  }
}

float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

void Rms(std::span<float> x, const float* w, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(
      1.0 / std::sqrt(sum / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * inv * (w != nullptr ? w[i] : 1.0F);
  }
}

void L2(std::span<float> x, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(1.0 / std::sqrt(sum + eps));
  for (auto& v : x)
    v *= inv;
}

// SiTU-GLU: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), all in fp32.
void Situ(std::span<const float> gate, std::span<const float> up, float beta,
          float lb, std::span<float> act) {
  for (std::size_t i = 0; i < act.size(); ++i) {
    const float g = std::tanh(gate[i] / beta) * beta * Sigmoid(gate[i]);
    const float u = std::tanh(up[i] / lb) * lb;
    act[i] = g * u;
  }
}

std::string P(std::uint32_t layer, const char* s) {
  return "blk." + std::to_string(layer) + "." + s;
}

bool EnvFlag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void DumpVector(const char* tag, std::uint32_t layer,
                std::span<const float> v) {
  const char* on = std::getenv("GUFO_RAON1_DUMP");
  if (on == nullptr || on[0] == '\0' || on[0] == '0')
    return;
  std::fprintf(stderr, "DUMP %s blk%u:", tag, layer);
  for (int i = 0; i < 4 && i < (int)v.size(); ++i)
    std::fprintf(stderr, " %.6f", v[i]);
  double sq = 0.0;
  for (const float x : v)
    sq += static_cast<double>(x) * x;
  std::fprintf(stderr, " rms=%.6g",
               std::sqrt(sq / std::max<std::size_t>(1, v.size())));
  std::fprintf(stderr, "\n");
  if (std::getenv("GUFO_RAON1_DUMP_FULL") != nullptr) {
    static std::unordered_map<std::string, int> counts;
    const std::string key = std::string(tag) + "_" + std::to_string(layer);
    const int index = counts[key]++;
    const std::string path = "/tmp/opencode/ref_" + std::string(tag) + "_b" +
                             std::to_string(layer) + "_" +
                             std::to_string(index) + ".bin";
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
      std::fwrite(v.data(), sizeof(float), v.size(), f);
      std::fclose(f);
    }
  }
}

}  // namespace

ReferenceModel::Loaded::Loaded(const GgufTensorInfo* t) {
  Setup(t);
  data_.resize(static_cast<std::size_t>(rows_) * cols_);
  for (std::uint64_t r = 0; r < rows_; ++r) {
    DequantizeRow(r, data_.data() + r * cols_);
  }
}

ReferenceModel::Loaded ReferenceModel::Loaded::RowView(
    const GgufTensorInfo* t) {
  Loaded f;
  f.Setup(t);
  return f;
}

void ReferenceModel::Loaded::Setup(const GgufTensorInfo* t) {
  t_ = t;
  rows_ = 1;
  for (std::size_t d = 1; d < t->dimensions.size(); ++d) {
    rows_ *= t->dimensions[d];
  }
  cols_ = t->dimensions[0];
}

void ReferenceModel::Loaded::DequantizeRow(std::uint64_t row,
                                           float* out) const {
  const std::size_t rb = static_cast<std::size_t>(
      t_->size_bytes / std::max<std::uint64_t>(rows_, 1));
  const char* src = static_cast<const char*>(t_->data) + row * rb;
  switch (t_->type) {
    case GgmlType::kF32:
      std::memcpy(out, src, cols_ * sizeof(float));
      break;
    case GgmlType::kF16: {
      const auto* p = reinterpret_cast<const std::uint16_t*>(src);
      for (std::uint64_t i = 0; i < cols_; ++i)
        out[i] = Half(p[i]);
      break;
    }
    case GgmlType::kBF16: {
      const auto* p = reinterpret_cast<const std::uint16_t*>(src);
      for (std::uint64_t i = 0; i < cols_; ++i)
        out[i] = Bf16(p[i]);
      break;
    }
    case GgmlType::kQ5_0:
      DequantizeQ5_0(src, out, cols_);
      break;
    case GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, out, cols_);
      break;
    case GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, out, cols_);
      break;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, out, cols_);
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out, cols_);
      break;
    default:
      std::abort();
  }
}

void ReferenceModel::Loaded::MatVec(std::span<const float> x,
                                    std::span<float> y) const {
  for (std::uint64_t r = 0; r < rows_; ++r) {
    const float* w = data_.data() + r * cols_;
    double acc = 0.0;
    for (std::uint64_t i = 0; i < cols_; ++i) {
      acc += static_cast<double>(w[i]) * x[i];
    }
    y[r] = static_cast<float>(acc);
  }
}

void ReferenceModel::Loaded::MatVecRows(std::uint64_t row0, std::uint64_t nrows,
                                        std::span<const float> x,
                                        std::span<float> y) const {
  for (std::uint64_t r = 0; r < nrows; ++r) {
    const float* w = data_.data() + (row0 + r) * cols_;
    double acc = 0.0;
    for (std::uint64_t i = 0; i < cols_; ++i) {
      acc += static_cast<double>(w[i]) * x[i];
    }
    y[r] = static_cast<float>(acc);
  }
}

void ReferenceModel::Loaded::MatVecView(std::uint64_t row0, std::uint64_t nrows,
                                        std::span<const float> x,
                                        std::span<float> y) const {
  if (t_->type == GgmlType::kQ5_0) {
    std::vector<Q8Block> q;
    QuantizeQ8(x.data(), cols_, q);
    const std::size_t rb = static_cast<std::size_t>(
        t_->size_bytes / std::max<std::uint64_t>(rows_, 1));
    const auto* base = static_cast<const char*>(t_->data);
    for (std::uint64_t r = 0; r < nrows; ++r) {
      y[r] = DotQ5Q8(base + (row0 + r) * rb, q, cols_);
    }
    return;
  }
  std::vector<float> rowbuf(cols_);
  for (std::uint64_t r = 0; r < nrows; ++r) {
    DequantizeRow(row0 + r, rowbuf.data());
    double acc = 0.0;
    for (std::uint64_t i = 0; i < cols_; ++i) {
      acc += static_cast<double>(rowbuf[i]) * x[i];
    }
    y[r] = static_cast<float>(acc);
  }
}

std::unique_ptr<ReferenceModel> ReferenceModel::Open(const std::string& path,
                                                     std::string* error) {
  auto model = std::unique_ptr<ReferenceModel>(new ReferenceModel());
  model->reader_ = gufo::core::GgufReader::OpenFile(path, error);
  if (!model->reader_) {
    return nullptr;
  }
  auto config = ParseConfig(*model->reader_, error);
  if (!config) {
    return nullptr;
  }
  model->config_ = *config;
  for (const auto& t : model->reader_->GetTensors()) {
    model->tensors_.emplace(std::string(t.name), &t);
  }
  // Default effort = the artifact's default name; override with the alpha
  // itself through GUFO_RAON1_EFFORT.
  if (const char* alpha = std::getenv("GUFO_RAON1_EFFORT")) {
    model->effort_alpha_ = static_cast<float>(std::atof(alpha));
  } else {
    for (std::size_t i = 0; i < model->config_.effort_names.size() &&
                            i < model->config_.effort_alphas.size();
         ++i) {
      if (model->config_.effort_names[i] == model->config_.effort_default) {
        model->effort_alpha_ = model->config_.effort_alphas[i];
      }
    }
  }
  if (EnvFlag("GUFO_RAON1_NO_EFFORT")) {
    model->effort_alpha_ = 0.0F;
  }
  return model;
}

const ReferenceModel::Loaded& ReferenceModel::Load(
    const std::string& name) const {
  auto it = cache_.find(name);
  if (it != cache_.end()) {
    return it->second;
  }
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    std::fprintf(stderr, "missing tensor %s\n", name.c_str());
    std::abort();
  }
  return cache_.emplace(name, Loaded(found->second)).first->second;
}

const ReferenceModel::Loaded& ReferenceModel::LoadRow(
    const std::string& name) const {
  auto it = cache_.find(name);
  if (it != cache_.end()) {
    return it->second;
  }
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    std::fprintf(stderr, "missing tensor %s\n", name.c_str());
    std::abort();
  }
  return cache_.emplace(name, Loaded::RowView(found->second)).first->second;
}

std::uint32_t ReferenceModel::n_layers() const {
  if (const char* cap = std::getenv("GUFO_RAON1_REF_LAYERS")) {
    return static_cast<std::uint32_t>(std::atoi(cap));
  }
  return config_.num_layers;
}

std::unique_ptr<ReferenceModel::Session> ReferenceModel::CreateSession() const {
  auto s = std::make_unique<Session>();
  s->x.assign(config_.HcDim(), 0.0F);
  s->layers.resize(n_layers());
  for (std::uint32_t layer = 0; layer < n_layers(); ++layer) {
    auto& l = s->layers[layer];
    if (config_.IsRda(layer)) {
      l.conv.assign(
          std::size_t{3} * (config_.ssm_conv_kernel - 1) * config_.KdaDim(),
          0.0F);
      l.state.assign(std::size_t{config_.num_heads} * config_.kda_head_dim *
                         config_.kda_head_dim,
                     0.0F);
    }
  }
  return s;
}

void ReferenceModel::HcMix(std::span<const float> x, const std::string& prefix,
                           std::vector<float>& pre, std::vector<float>& post,
                           std::vector<float>& comb) const {
  const std::uint32_t hc = config_.hc_count;
  const std::uint32_t hm = hc - 1;
  std::vector<float> xn(x.begin(), x.end());
  Rms(xn, nullptr, config_.hc_eps);
  std::vector<float> mixes(config_.HcRank(), 0.0F);
  Load(prefix + "_fn.weight").MatVec(xn, mixes);
  const float* scale = Load(prefix + "_scale.weight").w();
  const float* base = Load(prefix + "_base.weight").w();
  pre.assign(hc, 0.0F);
  post.assign(hc, 0.0F);
  comb.assign(std::size_t{hc} * hc, 0.0F);
  for (std::uint32_t s = 0; s < hc; ++s) {
    pre[s] = Sigmoid(mixes[s] * scale[0] + base[s]);
    post[s] =
        config_.hc_magnitude * Sigmoid(mixes[hc + s] * scale[1] + base[hc + s]);
  }
  // t_z (hm x hm) -> e = H t_z H^T, chart B = (I + 11^T/hc)/2.
  std::vector<float> tz(std::size_t{hm} * hm, 0.0F);
  for (std::uint32_t i = 0; i < hm; ++i) {
    for (std::uint32_t k = 0; k < hm; ++k) {
      tz[i * hm + k] =
          mixes[2 * hc + i * hm + k] * scale[2] + base[2 * hc + i * hm + k];
    }
  }
  // Helmert basis H[hc][hm].
  std::vector<float> H(std::size_t{hc} * hm, 0.0F);
  for (std::uint32_t i = 0; i < hc; ++i) {
    for (std::uint32_t j = 0; j < hm; ++j) {
      const float v =
          (i <= j ? 1.0F : (i == j + 1 ? -static_cast<float>(j + 1) : 0.0F)) /
          std::sqrt(static_cast<float>((j + 1) * (j + 2)));
      H[i * hm + j] = v;
    }
  }
  std::vector<float> e(std::size_t{hc} * hc, 0.0F);
  for (std::uint32_t i = 0; i < hc; ++i) {
    for (std::uint32_t j = 0; j < hc; ++j) {
      double acc = 0.0;
      for (std::uint32_t m = 0; m < hm; ++m) {
        for (std::uint32_t k = 0; k < hm; ++k) {
          acc += static_cast<double>(H[i * hm + m]) * tz[m * hm + k] *
                 H[j * hm + k];
        }
      }
      e[i * hc + j] = static_cast<float>(acc);
    }
  }
  DumpVector("hc_pre", std::stoi(prefix.substr(4, prefix.find('_', 4) - 4)),
             pre);
  DumpVector("hc_post", std::stoi(prefix.substr(4, prefix.find('_', 4) - 4)),
             post);
  const float inv_hc = 1.0F / static_cast<float>(hc);
  float rho = 0.0F;
  for (std::uint32_t i = 0; i < hc; ++i) {
    for (std::uint32_t j = 0; j < hc; ++j) {
      const float b = (i == j ? 0.5F : 0.0F) + 0.5F * inv_hc;
      rho = std::max(rho, -e[i * hc + j] / b);
    }
  }
  rho = std::max(rho, 0.0F);
  const float gamma = Sigmoid(scale[3]);
  const float denom = 1.0F + rho;
  const bool llama_order = EnvFlag("GUFO_RAON1_HC_LLAMA");
  for (std::uint32_t dst = 0; dst < hc; ++dst) {
    for (std::uint32_t src = 0; src < hc; ++src) {
      const float b = (dst == src ? 0.5F : 0.0F) + 0.5F * inv_hc;
      const float phi =
          b + (llama_order ? e[src * hc + dst] : e[dst * hc + src]) / denom;
      comb[dst + hc * src] =
          (1.0F - gamma) * (dst == src ? 1.0F : 0.0F) + gamma * phi;
    }
  }
  DumpVector("hc_comb", std::stoi(prefix.substr(4, prefix.find('_', 4) - 4)),
             comb);
}

void ReferenceModel::AttnBlock(std::uint32_t layer, Session& session, bool mla,
                               std::span<const float> xn,
                               std::span<float> block) const {
  auto& st = session.layers[layer];
  if (std::getenv("GUFO_RAON1_LAYER_DBG"))
    std::fprintf(stderr, "LAYER %u mla=%d\n", layer, (int)mla);
  const std::uint32_t hd = config_.kda_head_dim;
  const std::uint32_t inner = config_.KdaDim();
  if (!mla) {
    // RDA: conv, L2-normalised q/k, delta rule with separate write/erase gate.
    std::vector<float> q(inner), k(inner), v(inner);
    Load(P(layer, "attn_q.weight")).MatVec(xn, q);
    Load(P(layer, "attn_k.weight")).MatVec(xn, k);
    Load(P(layer, "attn_v.weight")).MatVec(xn, v);
    std::vector<float*> streams(3);
    streams[0] = q.data();
    streams[1] = k.data();
    streams[2] = v.data();
    const char* conv_names[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight",
                                 "ssm_conv1d_v.weight"};
    const std::uint32_t taps = config_.ssm_conv_kernel - 1;
    for (std::uint32_t w = 0; w < 3; ++w) {
      const std::size_t base = static_cast<std::size_t>(w) * taps * inner;
      const std::vector<float>& cw = Load(P(layer, conv_names[w])).data();
      std::vector<float> raw(streams[w], streams[w] + inner);
      for (std::uint32_t ch = 0; ch < inner; ++ch) {
        const float* tap_w = cw.data() + static_cast<std::size_t>(ch) * 4;
        float acc = tap_w[3] * raw[ch];
        for (std::uint32_t tap = 0; tap < taps; ++tap) {
          acc += tap_w[tap] *
                 st.conv[base + static_cast<std::size_t>(ch) * taps + tap];
        }
        streams[w][ch] = acc * Sigmoid(acc);
      }
      // Shift the per-channel raw window: drop oldest, append current.
      for (std::uint32_t ch = 0; ch < inner; ++ch) {
        const std::size_t off = base + static_cast<std::size_t>(ch) * taps;
        for (std::uint32_t tap = 0; tap + 1 < taps; ++tap) {
          st.conv[off + tap] = st.conv[off + tap + 1];
        }
        st.conv[off + taps - 1] = raw[ch];
      }
    }
    std::vector<float> fa(128);
    Load(P(layer, "ssm_f_a.weight")).MatVec(xn, fa);
    std::vector<float> gproj(inner);
    Load(P(layer, "ssm_f_b.weight")).MatVec(fa, gproj);
    const float* dt = Load(P(layer, "ssm_dt.bias")).w();
    const float* sa = Load(P(layer, "ssm_a")).w();
    std::vector<float> beta_raw(config_.num_heads);
    Load(P(layer, "ssm_beta.weight")).MatVec(xn, beta_raw);
    std::vector<float> delta(config_.num_heads);
    Load(P(layer, "ssm_rda.weight")).MatVec(xn, delta);
    const float* norm_w = Load(P(layer, "ssm_norm.weight")).w();
    std::vector<float> g2(inner);
    Load(P(layer, "ssm_g.weight")).MatVec(xn, g2);
    DumpVector("rda_q_conv", layer, q);
    DumpVector("rda_k_conv", layer, k);
    DumpVector("rda_v_conv", layer, v);
    std::vector<float> out(inner);
    const float qscale = 1.0F / std::sqrt(static_cast<float>(hd));
    for (std::uint32_t h = 0; h < config_.num_heads; ++h) {
      std::span<float> qh(q.data() + h * hd, hd);
      std::span<float> kh(k.data() + h * hd, hd);
      L2(qh, 1e-6F);
      L2(kh, 1e-6F);
      const float braw = beta_raw[h];
      const float b = Sigmoid(braw);
      const float ratio = std::exp(std::log(Sigmoid(braw + delta[h])) -
                                   std::log(Sigmoid(braw)));
      float* s_mat = st.state.data() + static_cast<std::size_t>(h) * hd * hd;
      std::vector<float> decay(hd);
      for (std::uint32_t i = 0; i < hd; ++i) {
        const float x = gproj[h * hd + i] + dt[h * hd + i];
        const float g = config_.kda_gate_lower_bound * Sigmoid(-sa[h] * x);
        decay[i] = std::exp(g);
      }
      DumpVector("rda_g1", layer, decay);
      std::vector<float> sb(hd);
      for (std::uint32_t i = 0; i < hd; ++i)
        sb[i] = b;
      DumpVector("rda_beta", layer, sb);
      for (std::uint32_t jv = 0; jv < hd; ++jv)
        v[h * hd + jv] *= ratio;
      DumpVector("rda_v_scaled", layer, v);
      std::vector<float> u(hd);
      for (std::uint32_t jv = 0; jv < hd; ++jv) {
        float* srow = s_mat + static_cast<std::size_t>(jv) * hd;
        double acc = 0.0;
        for (std::uint32_t i = 0; i < hd; ++i) {
          srow[i] *= decay[i];
          acc += static_cast<double>(srow[i]) * kh[i];
        }
        u[jv] = static_cast<float>(acc);
      }
      float* o = out.data() + h * hd;
      for (std::uint32_t jv = 0; jv < hd; ++jv) {
        const float vv = v[h * hd + jv];
        const float diff = (vv - u[jv]) * b;
        float* srow = s_mat + static_cast<std::size_t>(jv) * hd;
        double acc = 0.0;
        for (std::uint32_t i = 0; i < hd; ++i) {
          srow[i] += diff * kh[i];
          acc += static_cast<double>(srow[i]) * qh[i] * qscale;
        }
        o[jv] = static_cast<float>(acc);
      }
      DumpVector("rda_scan_out", layer, out);
      std::span<float> os(o, hd);
      Rms(os, norm_w, config_.rms_eps);
      DumpVector("rda_normed", layer, out);
      DumpVector("rda_g2", layer,
                 std::span<const float>(g2.data() + h * hd, hd));
      for (std::uint32_t jv = 0; jv < hd; ++jv) {
        o[jv] *= Sigmoid(g2[h * hd + jv]);
      }
      DumpVector("rda_final", layer, std::span<const float>(o, hd));
    }
    Load(P(layer, "attn_output.weight")).MatVec(out, block);
    return;
  }

  // NoPE MLA with the absorbed latent value.
  const std::uint32_t rank = config_.kv_lora_rank;
  const std::uint32_t nope = config_.qk_nope_head_dim;
  const std::uint32_t rope = config_.qk_rope_head_dim;
  const std::uint32_t heads = config_.num_heads;
  const std::uint32_t vd = config_.v_head_dim;
  std::vector<float> qr(config_.q_lora_rank);
  Load(P(layer, "attn_q_a.weight")).MatVec(xn, qr);
  Rms(qr, Load(P(layer, "attn_q_a_norm.weight")).w(), config_.latent_norm_eps);
  DumpVector("mla_q_norm", layer, qr);
  std::vector<float> q(heads * (nope + rope));
  Load(P(layer, "attn_q_b.weight")).MatVec(qr, q);
  std::vector<float> kv(rank + rope);
  Load(P(layer, "attn_kv_a_mqa.weight")).MatVec(xn, kv);
  std::vector<float> latent(kv.begin(), kv.begin() + rank);
  Rms(latent, Load(P(layer, "attn_kv_a_norm.weight")).w(),
      config_.latent_norm_eps);
  DumpVector("mla_kv_norm", layer, latent);
  st.lat.insert(st.lat.end(), latent.begin(), latent.end());
  st.extra.insert(st.extra.end(), kv.begin() + rank, kv.end());

  const std::uint32_t n = static_cast<std::uint32_t>(st.lat.size()) / rank;
  const float scale = 1.0F / std::sqrt(static_cast<float>(nope + rope));
  std::vector<float> out(heads * vd, 0.0F);
  const Loaded& wk = Load(P(layer, "attn_k_b.weight"));
  const Loaded& wv = Load(P(layer, "attn_v_b.weight"));
  std::vector<float> qabs(rank), ctx(rank), sc(n);
  for (std::uint32_t h = 0; h < heads; ++h) {
    const float* qh = q.data() + h * (nope + rope);
    wk.MatVecRows(static_cast<std::uint64_t>(h) * rank, rank,
                  std::span(qh, nope), qabs);
    const float* qe = qh + nope;
    float mx = -1e30F;
    for (std::uint32_t pos = 0; pos < n; ++pos) {
      const float* lp = st.lat.data() + static_cast<std::size_t>(pos) * rank;
      const float* ep = st.extra.data() + static_cast<std::size_t>(pos) * rope;
      double acc = 0.0;
      for (std::uint32_t c = 0; c < rank; ++c) {
        acc += static_cast<double>(qabs[c]) * lp[c];
      }
      for (std::uint32_t c = 0; c < rope; ++c) {
        acc += static_cast<double>(qe[c]) * ep[c];
      }
      sc[pos] = static_cast<float>(acc) * scale;
      mx = std::max(mx, sc[pos]);
    }
    double denom = 0.0;
    for (auto& s : sc) {
      s = std::exp(s - mx);
      denom += s;
    }
    std::fill(ctx.begin(), ctx.end(), 0.0F);
    for (std::uint32_t pos = 0; pos < n; ++pos) {
      const float a = static_cast<float>(sc[pos] / denom);
      const float* lp = st.lat.data() + static_cast<std::size_t>(pos) * rank;
      for (std::uint32_t c = 0; c < rank; ++c) {
        ctx[c] += a * lp[c];
      }
    }
    wv.MatVecRows(static_cast<std::uint64_t>(h) * vd, vd, ctx,
                  std::span(out.data() + static_cast<std::size_t>(h) * vd, vd));
  }
  std::vector<float> gate(config_.MlaVDim());
  Load(P(layer, "attn_gate.weight")).MatVec(xn, gate);
  for (std::uint32_t i = 0; i < out.size(); ++i) {
    out[i] *= Sigmoid(gate[i]);
  }
  DumpVector("mla_gated", layer, out);
  Load(P(layer, "attn_output.weight")).MatVec(out, block);
}

void ReferenceModel::Dump(std::uint32_t layer, const char* tag,
                          std::span<const float> v) const {
  DumpVector(tag, layer, v);
}

void ReferenceModel::FfnBlock(std::uint32_t layer, std::span<const float> fin,
                              std::span<float> ffn_out) const {
  const std::uint32_t H = config_.hidden_size;
  if (config_.IsDense(layer)) {
    std::vector<float> gate(config_.dense_ff), up(config_.dense_ff),
        act(config_.dense_ff);
    Load(P(layer, "ffn_gate.weight")).MatVec(fin, gate);
    Load(P(layer, "ffn_up.weight")).MatVec(fin, up);
    Situ(gate, up, config_.situ_beta, config_.situ_linear_beta, act);
    Load(P(layer, "ffn_down.weight")).MatVec(act, ffn_out);
    return;
  }
  std::fill(ffn_out.begin(), ffn_out.end(), 0.0F);
  std::vector<float> logits(config_.expert_count);
  Load(P(layer, "ffn_gate_inp.weight")).MatVec(fin, logits);
  const float* eb = Load(P(layer, "exp_probs_b.bias")).w();
  DumpVector("ffn_moe_logits", layer, logits);
  std::vector<float> probs(config_.expert_count), sel(config_.expert_count);
  for (std::uint32_t e = 0; e < config_.expert_count; ++e) {
    probs[e] = Sigmoid(logits[e]);
    sel[e] = probs[e] + eb[e];
  }
  DumpVector("ffn_moe_probs", layer, probs);
  DumpVector("ffn_moe_probs_biased", layer, sel);
  std::vector<std::uint32_t> idx(config_.expert_count);
  std::iota(idx.begin(), idx.end(), 0U);
  std::stable_sort(
      idx.begin(), idx.end(),
      [&](std::uint32_t a, std::uint32_t b) { return sel[a] > sel[b]; });
  float wsum = 0.0F;
  for (std::uint32_t i = 0; i < config_.expert_used; ++i) {
    wsum += probs[idx[i]];
  }
  if (config_.expert_weights_norm) {
    wsum += 1e-20F;
  } else {
    wsum = 1.0F;
  }
  std::vector<float> dumpw(config_.expert_used), dumpi(config_.expert_used);
  for (std::uint32_t i = 0; i < config_.expert_used; ++i) {
    dumpw[i] = probs[idx[i]] / wsum;
    dumpi[i] = static_cast<float>(idx[i]);
  }
  DumpVector("ffn_moe_topk", layer, dumpi);
  DumpVector("ffn_moe_weights", layer, dumpw);
  const std::uint32_t ff = config_.expert_ff;
  const Loaded& wg = LoadRow(P(layer, "ffn_gate_exps.weight"));
  const Loaded& wu = LoadRow(P(layer, "ffn_up_exps.weight"));
  const Loaded& wd = LoadRow(P(layer, "ffn_down_exps.weight"));
  std::vector<float> gf(ff), uf(ff), act(ff), dpart(H);
  for (std::uint32_t i = 0; i < config_.expert_used; ++i) {
    const std::uint32_t e = idx[i];
    const float wgt = probs[e] / wsum * config_.expert_weights_scale;
    wg.MatVecView(static_cast<std::uint64_t>(e) * ff, ff, fin, gf);
    wu.MatVecView(static_cast<std::uint64_t>(e) * ff, ff, fin, uf);
    Situ(gf, uf, config_.situ_beta, config_.situ_linear_beta, act);
    if (i == 0) {
      DumpVector("ffn_moe_gate", layer, gf);
      DumpVector("ffn_moe_up", layer, uf);
      DumpVector("ffn_moe_situ", layer, act);
    }
    wd.MatVecView(static_cast<std::uint64_t>(e) * H, H, act, dpart);
    if (i == 0)
      DumpVector("ffn_moe_down", layer, dpart);
    for (std::uint32_t c = 0; c < H; ++c) {
      ffn_out[c] += wgt * dpart[c];
    }
  }
  DumpVector("ffn_moe", layer, ffn_out);
  std::vector<float> sg(ff), su(ff);
  Load(P(layer, "ffn_gate_shexp.weight")).MatVec(fin, sg);
  Load(P(layer, "ffn_up_shexp.weight")).MatVec(fin, su);
  Situ(sg, su, config_.situ_beta, config_.situ_linear_beta, act);
  Load(P(layer, "ffn_down_shexp.weight")).MatVec(act, dpart);
  DumpVector("ffn_shared", layer, dpart);
  for (std::uint32_t c = 0; c < H; ++c) {
    ffn_out[c] += dpart[c];
  }
  if (effort_alpha_ != 0.0F) {
    const auto it = tensors_.find(P(layer, "ffn_effort_steer.weight"));
    if (it != tensors_.end()) {
      const float* cvec = Load(P(layer, "ffn_effort_steer.weight")).w();
      if (!EnvFlag("GUFO_RAON1_EFFORT_LLAMA")) {
        // HF: applied to the MoE branch before the mHC write-back.
        for (std::uint32_t c = 0; c < H; ++c) {
          ffn_out[c] = static_cast<float>(static_cast<double>(ffn_out[c]) +
                                          effort_alpha_ * cvec[c]);
        }
      }
    }
  }
}

const std::vector<float>& ReferenceModel::Evaluate(Session& session,
                                                   std::int32_t token) const {
  const std::uint32_t H = config_.hidden_size;
  const std::uint32_t hc = config_.hc_count;
  const GgufTensorInfo& embd = *tensors_.at("token_embd.weight");
  const Loaded row_view = Loaded::RowView(&embd);
  std::vector<float> row(H);
  row_view.DequantizeRow(static_cast<std::uint64_t>(token), row.data());
  for (std::uint32_t s = 0; s < hc; ++s) {
    std::copy_n(row.data(), H, session.x.data() + s * H);
  }

  const bool effort_llama = EnvFlag("GUFO_RAON1_EFFORT_LLAMA");
  for (std::uint32_t layer = 0; layer < n_layers(); ++layer) {
    const bool mla = config_.IsMla(layer);

    std::vector<float> pre, post, comb;
    HcMix(session.x, P(layer, "hc_attn"), pre, post, comb);
    std::vector<float> attn_in(H);
    for (std::uint32_t i = 0; i < H; ++i) {
      float acc = 0.0F;
      for (std::uint32_t s = 0; s < hc; ++s) {
        acc += session.x[s * H + i] * pre[s];
      }
      attn_in[i] = acc;
    }
    Rms(attn_in, Load(P(layer, "attn_norm.weight")).w(), config_.rms_eps);
    DumpVector("attn_norm", layer, attn_in);
    std::vector<float> block(H);
    AttnBlock(layer, session, mla, attn_in, block);
    DumpVector("attn_out", layer, block);
    for (std::uint32_t s = 0; s < hc; ++s) {
      DumpVector(("x_attn_pre_s" + std::to_string(s)).c_str(), layer,
                 std::span<const float>(session.x).subspan(s * H, H));
    }
    std::vector<float> next(session.x.size(), 0.0F);
    for (std::uint32_t d = 0; d < hc; ++d) {
      for (std::uint32_t i = 0; i < H; ++i) {
        float acc = block[i] * post[d];
        for (std::uint32_t s = 0; s < hc; ++s) {
          acc += session.x[s * H + i] * comb[d + hc * s];
        }
        next[d * H + i] = acc;
      }
    }
    session.x.swap(next);
    for (std::uint32_t s = 0; s < hc; ++s) {
      DumpVector(("x_attn_post_s" + std::to_string(s)).c_str(), layer,
                 std::span<const float>(session.x).subspan(s * H, H));
    }

    HcMix(session.x, P(layer, "hc_ffn"), pre, post, comb);
    std::vector<float> ffn_in(H);
    for (std::uint32_t i = 0; i < H; ++i) {
      float acc = 0.0F;
      for (std::uint32_t s = 0; s < hc; ++s) {
        acc += session.x[s * H + i] * pre[s];
      }
      ffn_in[i] = acc;
    }
    Rms(ffn_in, Load(P(layer, "ffn_norm.weight")).w(), config_.rms_eps);
    DumpVector("ffn_norm", layer, ffn_in);
    std::vector<float> ffn_out(H, 0.0F);
    FfnBlock(layer, ffn_in, ffn_out);
    DumpVector("ffn_out", layer, ffn_out);
    for (std::uint32_t s = 0; s < hc; ++s) {
      DumpVector(("x_ffn_pre_s" + std::to_string(s)).c_str(), layer,
                 std::span<const float>(session.x).subspan(s * H, H));
    }
    next.assign(session.x.size(), 0.0F);
    for (std::uint32_t d = 0; d < hc; ++d) {
      for (std::uint32_t i = 0; i < H; ++i) {
        float acc = ffn_out[i] * post[d];
        for (std::uint32_t s = 0; s < hc; ++s) {
          acc += session.x[s * H + i] * comb[d + hc * s];
        }
        next[d * H + i] = acc;
      }
    }
    session.x.swap(next);
    for (std::uint32_t s = 0; s < hc; ++s) {
      DumpVector(("x_ffn_post_s" + std::to_string(s)).c_str(), layer,
                 std::span<const float>(session.x).subspan(s * H, H));
    }
    if (effort_llama) {
      const auto it = tensors_.find(P(layer, "ffn_effort_steer.weight"));
      if (it != tensors_.end()) {
        const float* cvec = Load(P(layer, "ffn_effort_steer.weight")).w();
        for (std::uint32_t s = 0; s < hc; ++s) {
          for (std::uint32_t i = 0; i < H; ++i) {
            session.x[s * H + i] += effort_alpha_ * cvec[i];
          }
        }
      }
    }
    ++session.position;
  }

  std::vector<float> head(H);
  for (std::uint32_t i = 0; i < H; ++i) {
    float acc = 0.0F;
    for (std::uint32_t s = 0; s < hc; ++s) {
      acc += session.x[s * H + i];
    }
    head[i] = acc / static_cast<float>(hc);
  }
  Rms(head, Load("output_norm.weight").w(), config_.rms_eps);
  DumpVector("result_norm", 999, head);
  const Loaded& head_w = Load("output.weight");
  session.logits.resize(static_cast<std::size_t>(head_w.rows()));
  head_w.MatVec(head, session.logits);
  return session.logits;
}

}  // namespace gufo::models::raon1
