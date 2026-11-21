#include "src/models/inkling/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::inkling {
namespace {

using gufo::core::GgmlType;
using Vec = std::vector<float>;

constexpr std::uint32_t kNvfp4Block = 64;
constexpr std::uint32_t kNvfp4Bytes = 36;
constexpr float kMixEps = 1e-6F;
constexpr float kClipEps = 1e-6F;
constexpr float kRouterNormEps = 1e-20F;

float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

float Bf16(std::uint16_t h) {
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

float E2M1(std::uint8_t n) {
  static const float v[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                              0, -1, -2, -3, -4, -6, -8, -12};
  return v[n & 0xF];
}

float Ue4m3(std::uint8_t x) {
  if (x == 0 || x == 0x7F)
    return 0.0F;
  const int e = (x >> 3) & 0xF;
  const int m = x & 7;
  const float raw =
      e == 0 ? std::ldexp(static_cast<float>(m), -9)
             : std::ldexp(1.0F + static_cast<float>(m) / 8.0F, e - 7);
  return raw * 0.5F;
}

void DeqNvfp4(const std::uint8_t* src, float* out, std::uint64_t cols) {
  for (std::uint64_t b = 0; b < cols / kNvfp4Block; ++b) {
    const std::uint8_t* blk = src + b * kNvfp4Bytes;
    const std::uint8_t* qs = blk + 4;
    float* o = out + b * kNvfp4Block;
    for (int s = 0; s < 4; ++s) {
      const float d = Ue4m3(blk[s]);
      for (int j = 0; j < 8; ++j) {
        const std::uint8_t q = qs[s * 8 + j];
        o[s * 16 + j] = E2M1(q & 0xF) * d;
        o[s * 16 + 8 + j] = E2M1(q >> 4) * d;
      }
    }
  }
}

void Rms(Vec& x, const float* w, float eps) {
  double s = 0.0;
  for (const float v : x)
    s += static_cast<double>(v) * v;
  const float inv = static_cast<float>(
      1.0 / std::sqrt(s / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = x[i] * inv * (w != nullptr ? w[i] : 1.0F);
}

void ClipRows(Vec& x, std::size_t row, const float* scale, float threshold) {
  for (std::size_t o = 0; o + row <= x.size(); o += row) {
    double s = 0.0;
    for (std::size_t i = 0; i < row; ++i) {
      if (scale != nullptr)
        x[o + i] *= scale[o + i];
      s += static_cast<double>(x[o + i]) * x[o + i];
    }
    const float f = std::min(
        1.0F, threshold / (static_cast<float>(std::sqrt(s)) + kClipEps));
    for (std::size_t i = 0; i < row; ++i)
      x[o + i] *= f;
  }
}

}  // namespace

void ReferenceModel::Mat::Row(std::uint64_t row, float* out) const {
  const auto* src = static_cast<const std::uint8_t*>(t->data) + row * row_bytes;
  switch (t->type) {
    case GgmlType::kF32:
      std::memcpy(out, src, cols * sizeof(float));
      break;
    case GgmlType::kBF16:
      for (std::uint64_t i = 0; i < cols; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + i * 2, 2);
        out[i] = Bf16(h);
      }
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out, cols);
      break;
    case GgmlType::kNVFP4:
      DeqNvfp4(src, out, cols);
      break;
    default:
      std::fprintf(stderr, "unsupported type %d\n", static_cast<int>(t->type));
      std::abort();
  }
}

void ReferenceModel::Mat::MatVec(const float* x, float* y, std::uint64_t row0,
                                 std::uint64_t n) const {
  Vec w(cols);
  for (std::uint64_t r = 0; r < n; ++r) {
    Row(row0 + r, w.data());
    double a = 0.0;
    for (std::uint64_t i = 0; i < cols; ++i)
      a += static_cast<double>(w[i]) * x[i];
    y[r] = static_cast<float>(a);
  }
}

ReferenceModel::Mat ReferenceModel::M(const std::string& name) const {
  const auto* t = reader_->FindTensor(name);
  if (t == nullptr) {
    std::fprintf(stderr, "missing tensor %s\n", name.c_str());
    std::abort();
  }
  Mat m;
  m.t = t;
  m.cols = t->dimensions[0];
  m.rows = 1;
  for (std::size_t d = 1; d < t->dimensions.size(); ++d)
    m.rows *= t->dimensions[d];
  switch (t->type) {
    case GgmlType::kF32:
      m.row_bytes = m.cols * 4;
      break;
    case GgmlType::kBF16:
      m.row_bytes = m.cols * 2;
      break;
    case GgmlType::kQ8_0:
      m.row_bytes = m.cols / 32 * 34;
      break;
    default:
      m.row_bytes = m.cols / kNvfp4Block * kNvfp4Bytes;
      break;
  }
  return m;
}

Vec ReferenceModel::F(const std::string& name) const {
  const Mat m = M(name);
  Vec v(m.rows * m.cols);
  for (std::uint64_t r = 0; r < m.rows; ++r)
    m.Row(r, v.data() + r * m.cols);
  return v;
}

std::string ReferenceModel::P(std::uint32_t il, const char* s) const {
  return "blk." + std::to_string(il) + "." + s;
}

std::uint32_t ReferenceModel::Phys(std::uint32_t slot) const {
  if (slot < n_phys_)
    return slot;
  const std::uint32_t span = config_.loop_core_end - config_.loop_core_start;
  return config_.loop_core_start + (slot - n_phys_) % span;
}

bool ReferenceModel::IsSwa(std::uint32_t slot) const {
  return config_.is_swa[slot];
}

std::unique_ptr<ReferenceModel> ReferenceModel::Open(const std::string& path,
                                                     std::string* error) {
  auto reader = gufo::core::GgufReader::OpenFile(path, error);
  if (!reader)
    return nullptr;
  auto cfg = ParseConfig(*reader, error);
  if (!cfg)
    return nullptr;
  std::unique_ptr<ReferenceModel> m(new ReferenceModel());
  m->reader_ = std::move(reader);
  m->config_ = *cfg;
  m->n_phys_ =
      cfg->num_layers -
      (cfg->loop_num_loops - 1) * (cfg->loop_core_end - cfg->loop_core_start);
  return m;
}

std::unique_ptr<ReferenceModel::Session> ReferenceModel::CreateSession() const {
  auto s = std::make_unique<Session>();
  s->slots.resize(config_.num_layers);
  const std::size_t kv =
      static_cast<std::size_t>(config_.num_kv) * config_.head_dim;
  for (auto& sl : s->slots) {
    sl.hist_k.assign(kv * kHistory, 0.0F);
    sl.hist_v.assign(kv * kHistory, 0.0F);
    sl.hist_attn.assign(
        static_cast<std::size_t>(config_.hidden_size) * kHistory, 0.0F);
    sl.hist_ffn.assign(static_cast<std::size_t>(config_.hidden_size) * kHistory,
                       0.0F);
  }
  return s;
}

Vec ReferenceModel::Mix(const std::vector<Vec>& blocks, const Vec& cur,
                        const float* query, std::uint32_t nq) const {
  const std::uint32_t h = config_.hidden_size;
  std::vector<Vec> st;
  for (const auto& b : blocks) {
    st.push_back(b);
    Rms(st.back(), nullptr, kMixEps);
  }
  st.push_back(cur);
  Rms(st.back(), nullptr, kMixEps);
  const float scale = 1.0F / std::sqrt(static_cast<float>(h));
  Vec out(static_cast<std::size_t>(nq) * h, 0.0F);
  for (std::uint32_t q = 0; q < nq; ++q) {
    Vec sc(st.size());
    float mx = -1e30F;
    for (std::size_t i = 0; i < st.size(); ++i) {
      double d = 0.0;
      for (std::uint32_t j = 0; j < h; ++j)
        d += static_cast<double>(st[i][j]) * query[q * h + j];
      sc[i] = static_cast<float>(d) * scale;
      mx = std::max(mx, sc[i]);
    }
    double den = 0.0;
    for (auto& v : sc) {
      v = std::exp(v - mx);
      den += v;
    }
    for (std::size_t i = 0; i < st.size(); ++i) {
      const float w = sc[i] / static_cast<float>(den);
      for (std::uint32_t j = 0; j < h; ++j)
        out[q * h + j] += w * st[i][j];
    }
  }
  return out;
}

Vec ReferenceModel::Control(const Vec& cur, const Vec& message,
                            std::uint32_t phys, const char* group) const {
  const std::uint32_t h = config_.hidden_size;
  const std::uint32_t rank = config_.loop_gate_rank;
  Vec gate(2 * h);
  Vec a = cur;
  Vec b = message;
  Rms(a, nullptr, kMixEps);
  Rms(b, nullptr, kMixEps);
  std::copy(a.begin(), a.end(), gate.begin());
  std::copy(b.begin(), b.end(), gate.begin() + h);
  const std::string base =
      "blk." + std::to_string(phys) + ".loop_" + group + "_";
  Vec low(rank);
  M(base + "down.weight").MatVec(gate.data(), low.data(), 0, rank);
  Vec gain(h);
  Vec forget(h);
  M(base + "gain.weight").MatVec(low.data(), gain.data(), 0, h);
  M(base + "forget.weight").MatVec(low.data(), forget.data(), 0, h);
  Vec out(h);
  for (std::uint32_t i = 0; i < h; ++i) {
    out[i] = cur[i] - std::tanh(forget[i]) * cur[i] +
             std::tanh(gain[i]) * message[i];
  }
  return out;
}

Vec ReferenceModel::Convolve(const Vec& cur, const Vec& w, const Vec* dyn,
                             Vec& hist) const {
  const std::size_t ch = cur.size();
  const std::size_t heads = dyn != nullptr ? dyn->size() / kConvTaps : 1;
  const std::size_t per = ch / heads;
  Vec out(ch);
  const float norm =
      1.0F /
      std::sqrt(static_cast<float>((dyn != nullptr ? 2 : 1) * kConvTaps + 1));
  for (std::size_t c = 0; c < ch; ++c) {
    float win[kConvTaps];
    for (std::uint32_t k = 0; k < kHistory; ++k)
      win[k] = hist[c * kHistory + k];
    win[kHistory] = cur[c];
    float o = 0.0F;
    for (std::uint32_t j = 0; j < kConvTaps; ++j)
      o += w[c * kConvTaps + j] * win[j];
    if (dyn != nullptr) {
      for (std::uint32_t lag = 0; lag < kConvTaps; ++lag)
        o += (*dyn)[(c / per) * kConvTaps + lag] * win[kHistory - lag];
    }
    out[c] = (cur[c] + o) * norm;
    for (std::uint32_t k = 0; k + 1 < kHistory; ++k)
      hist[c * kHistory + k] = hist[c * kHistory + k + 1];
    hist[c * kHistory + kHistory - 1] = cur[c];
  }
  return out;
}

Vec ReferenceModel::Moe(std::uint32_t phys, const Vec& x) const {
  const std::uint32_t h = config_.hidden_size;
  const std::uint32_t ne = config_.expert_count;
  const std::uint32_t nu = config_.expert_used;
  const std::uint32_t ff = config_.expert_ff;
  Vec logits(ne);
  M(P(phys, "ffn_gate_inp.weight")).MatVec(x.data(), logits.data(), 0, ne);
  const Vec bias = F(P(phys, "exp_probs_b.bias"));
  Vec prob(ne);
  std::vector<std::uint32_t> idx(ne);
  for (std::uint32_t e = 0; e < ne; ++e) {
    prob[e] = Sigmoid(logits[e]);
    idx[e] = e;
  }
  std::partial_sort(idx.begin(), idx.begin() + nu, idx.end(),
                    [&](std::uint32_t a, std::uint32_t b) {
                      const float sa = prob[a] + bias[a];
                      const float sb = prob[b] + bias[b];
                      return sa != sb ? sa > sb : a < b;
                    });
  float sum = kRouterNormEps;
  for (std::uint32_t i = 0; i < nu; ++i)
    sum += prob[idx[i]];
  const Vec s_gu = reader_->HasTensor(P(phys, "ffn_gate_up_exps.scale"))
                       ? F(P(phys, "ffn_gate_up_exps.scale"))
                       : Vec(ne, 1.0F);
  const Vec s_dn = reader_->HasTensor(P(phys, "ffn_down_exps.scale"))
                       ? F(P(phys, "ffn_down_exps.scale"))
                       : Vec(ne, 1.0F);
  const Mat gu = M(P(phys, "ffn_gate_up_exps.weight"));
  const Mat dn = M(P(phys, "ffn_down_exps.weight"));
  const std::uint64_t gu_rows = 2ULL * ff;
  const float beta = config_.situ_beta;
  const float lb = config_.situ_linear_beta;
  Vec out(h, 0.0F);
  for (std::uint32_t i = 0; i < nu; ++i) {
    const std::uint32_t e = idx[i];
    const float w = prob[e] / sum * config_.expert_weights_scale;
    Vec g(gu_rows);
    gu.MatVec(x.data(), g.data(), e * gu_rows, gu_rows);
    Vec act(ff);
    for (std::uint32_t j = 0; j < ff; ++j) {
      const float gate = g[j] * s_gu[e];
      const float up = g[ff + j] * s_gu[e];
      act[j] = beta * std::tanh(gate / beta) * Sigmoid(gate) * lb *
               std::tanh(up / lb);
    }
    Vec y(h);
    dn.MatVec(act.data(), y.data(), static_cast<std::uint64_t>(e) * h, h);
    for (std::uint32_t j = 0; j < h; ++j)
      out[j] += w * y[j] * s_dn[e];
  }
  return out;
}

Vec ReferenceModel::Layer(std::uint32_t slot, std::uint32_t ip, Vec cur,
                          std::vector<Vec>& blocks, const std::vector<Vec>* msg,
                          Session& s, std::uint32_t pos) const {
  const std::uint32_t h = config_.hidden_size;
  const std::uint32_t p = Phys(slot);
  const bool swa = IsSwa(slot);
  const std::uint32_t nh = config_.num_heads;
  const std::uint32_t nkv = config_.num_kv;
  const std::uint32_t hd = config_.head_dim;
  const std::uint32_t nrel = config_.relative_length;
  const std::uint32_t extent = config_.relative_extent;
  const float eps = config_.rms_eps;
  SlotState& st = s.slots[slot];

  const Vec inp_sa = cur;
  Vec in[3] = {cur, cur, cur};
  if (reader_->HasTensor(P(p, "attn_res_query.weight"))) {
    const Vec q = F(P(p, "attn_res_query.weight"));
    const Vec mixed = Mix(blocks, cur, q.data(), swa ? 1 : 3);
    for (std::uint32_t j = 0; j < 3; ++j) {
      const std::uint32_t k = swa ? 0 : j;
      in[j].assign(mixed.begin() + k * h, mixed.begin() + (k + 1) * h);
    }
  }
  if (msg != nullptr) {
    static const char* const kGroup[3] = {"attn", "k", "v"};
    for (std::uint32_t j = 0; j < 3; ++j)
      in[j] = Control(in[j], (*msg)[j], p, kGroup[j]);
  }
  if (swa) {
    const Vec w = F(P(p, "attn_norm.weight"));
    if (msg != nullptr) {
      for (auto& v : in)
        Rms(v, w.data(), eps);
    } else {
      Rms(in[0], w.data(), eps);
      in[1] = in[2] = in[0];
    }
  } else {
    static const char* const kN[3] = {"attn_q_inp_norm.weight",
                                      "attn_k_inp_norm.weight",
                                      "attn_v_inp_norm.weight"};
    for (std::uint32_t j = 0; j < 3; ++j)
      Rms(in[j], F(P(p, kN[j])).data(), eps);
  }

  Vec q(static_cast<std::size_t>(nh) * hd);
  Vec k(static_cast<std::size_t>(nkv) * hd);
  Vec v(static_cast<std::size_t>(nkv) * hd);
  Vec r(static_cast<std::size_t>(nrel) * nh);
  M(P(p, "attn_q.weight")).MatVec(in[0].data(), q.data(), 0, q.size());
  M(P(p, "attn_k.weight")).MatVec(in[1].data(), k.data(), 0, k.size());
  M(P(p, "attn_v.weight")).MatVec(in[2].data(), v.data(), 0, v.size());
  M(P(p, "attn_rel.weight")).MatVec(in[0].data(), r.data(), 0, r.size());
  Vec dk;
  Vec dv;
  if (!swa) {
    dk.resize(static_cast<std::size_t>(kConvTaps) * nkv);
    dv.resize(dk.size());
    M(P(p, "attn_k_conv_proj.weight"))
        .MatVec(in[1].data(), dk.data(), 0, dk.size());
    M(P(p, "attn_v_conv_proj.weight"))
        .MatVec(in[2].data(), dv.data(), 0, dv.size());
  }
  k = Convolve(k, F(P(p, "attn_k_conv.weight")), swa ? nullptr : &dk,
               st.hist_k);
  v = Convolve(v, F(P(p, "attn_v_conv.weight")), swa ? nullptr : &dv,
               st.hist_v);
  ClipRows(q, hd, F(P(p, "attn_q_clip.weight")).data(), config_.qk_clip);
  ClipRows(k, hd, nullptr, config_.qk_clip);
  for (std::uint32_t g = 0; g < nkv; ++g) {
    Vec t(v.begin() + g * hd, v.begin() + (g + 1) * hd);
    Rms(t, nullptr, eps);
    std::copy(t.begin(), t.end(), v.begin() + g * hd);
  }
  ClipRows(r, nrel, F(P(p, "attn_rel_clip.weight")).data(),
           config_.relative_clip);
  if (!swa) {
    const float tau =
        1.0F + config_.log_scale_alpha *
                   std::log(std::max(
                       1.0F, static_cast<float>(pos + 1) /
                                 static_cast<float>(config_.log_scale_floor)));
    for (auto& x : q)
      x *= tau;
    for (auto& x : r)
      x *= tau;
  }
  Vec sink(nh);
  M(P(p, "attn_sink_proj.weight")).MatVec(in[0].data(), sink.data(), 0, nh);

  st.k.insert(st.k.end(), k.begin(), k.end());
  st.v.insert(st.v.end(), v.begin(), v.end());
  const std::uint32_t n_tok = pos + 1;
  const std::uint32_t first = swa && n_tok > config_.sliding_window
                                  ? n_tok - config_.sliding_window
                                  : 0;
  const Vec bias = F(P(p, "attn_rel_bias.weight"));
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  Vec att(static_cast<std::size_t>(nh) * hd, 0.0F);
  for (std::uint32_t hh = 0; hh < nh; ++hh) {
    const std::uint32_t g = hh / (nh / nkv);
    const float* qh = q.data() + hh * hd;
    const float* rh = r.data() + hh * nrel;
    Vec sc(n_tok - first + 1);
    float mx = 0.0F;
    for (std::uint32_t j = first; j < n_tok; ++j) {
      const float* kj =
          st.k.data() + (static_cast<std::size_t>(j) * nkv + g) * hd;
      double d = 0.0;
      for (std::uint32_t i = 0; i < hd; ++i)
        d += static_cast<double>(qh[i]) * kj[i];
      const std::uint32_t dist = pos - j;
      float b = 0.0F;
      if (dist < extent) {
        for (std::uint32_t i = 0; i < nrel; ++i)
          b += bias[static_cast<std::size_t>(dist) * nrel + i] * rh[i];
      }
      sc[j - first] = static_cast<float>(d) * scale + b - sink[hh];
      mx = std::max(mx, sc[j - first]);
    }
    sc.back() = 0.0F;
    double den = 0.0;
    for (auto& x : sc) {
      x = std::exp(x - mx);
      den += x;
    }
    for (std::uint32_t j = first; j < n_tok; ++j) {
      const float w = sc[j - first] / static_cast<float>(den);
      const float* vj =
          st.v.data() + (static_cast<std::size_t>(j) * nkv + g) * hd;
      for (std::uint32_t i = 0; i < hd; ++i)
        att[hh * hd + i] += w * vj[i];
    }
  }
  Vec gate(att.size());
  M(P(p, "attn_gate.weight")).MatVec(in[0].data(), gate.data(), 0, gate.size());
  for (std::size_t i = 0; i < att.size(); ++i)
    att[i] *= Sigmoid(gate[i]);
  Vec o(h);
  M(P(p, "attn_output.weight")).MatVec(att.data(), o.data(), 0, h);
  cur = Convolve(o, F(P(p, "attn_post_conv.weight")), nullptr, st.hist_attn);

  if (ip % config_.attn_res_block_size == 0) {
    blocks.push_back(inp_sa);
  } else {
    for (std::uint32_t i = 0; i < h; ++i)
      cur[i] += inp_sa[i];
  }
  const Vec ffn_inp = cur;
  const Vec fq = F(P(p, "ffn_res_query.weight"));
  Vec x = Mix(blocks, cur, fq.data(), 1);
  if (msg != nullptr)
    x = Control(x, (*msg)[3], p, "mlp");
  Rms(x, F(P(p, "ffn_norm.weight")).data(), eps);
  Vec f = Moe(p, x);
  f = Convolve(f, F(P(p, "ffn_post_conv.weight")), nullptr, st.hist_ffn);
  for (std::uint32_t i = 0; i < h; ++i)
    f[i] += ffn_inp[i];
  return f;
}

const Vec& ReferenceModel::Evaluate(Session& s, std::int32_t token) const {
  const std::uint32_t h = config_.hidden_size;
  const std::uint32_t pos = s.position;
  Vec cur(h);
  M("token_embd.weight").Row(static_cast<std::uint64_t>(token), cur.data());
  Rms(cur, F("token_embd_norm.weight").data(), config_.rms_eps);

  const std::uint32_t start = config_.loop_core_start;
  const std::uint32_t end = config_.loop_core_end;
  std::vector<Vec> blocks;
  for (std::uint32_t il = 0; il < start; ++il)
    cur = Layer(il, il, cur, blocks, nullptr, s, pos);

  const Vec prelude = cur;
  const std::vector<Vec> prelude_blocks = blocks;
  for (std::uint32_t pass = 0; pass < config_.loop_num_loops; ++pass) {
    std::vector<Vec> bank;
    Vec bank_out;
    if (pass > 0) {
      bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
      bank_out = cur;
      cur = prelude;
      blocks = prelude_blocks;
    }
    for (std::uint32_t ip = start; ip < end; ++ip) {
      const std::uint32_t il =
          pass == 0 ? ip : n_phys_ + (pass - 1) * (end - start) + (ip - start);
      if (pass == 0) {
        cur = Layer(il, ip, cur, blocks, nullptr, s, pos);
        continue;
      }
      // message queries: attn/ffn from loop_res_query, k/v from
      // loop_kv_res_query (column pairs per core layer)
      const std::uint32_t c = ip - start;
      const Vec res = F("loop_res_query.weight");
      const Vec kv = F("loop_kv_res_query.weight");
      Vec mq(4 * h);
      std::copy_n(res.begin() + (2 * c) * h, h, mq.begin());
      std::copy_n(kv.begin() + (2 * c) * h, h, mq.begin() + h);
      std::copy_n(kv.begin() + (2 * c + 1) * h, h, mq.begin() + 2 * h);
      std::copy_n(res.begin() + (2 * c + 1) * h, h, mq.begin() + 3 * h);
      const Vec m = Mix(bank, bank_out, mq.data(), 4);
      std::vector<Vec> msgs(4);
      for (std::uint32_t j = 0; j < 4; ++j)
        msgs[j].assign(m.begin() + j * h, m.begin() + (j + 1) * h);
      cur = Layer(il, ip, cur, blocks, &msgs, s, pos);
    }
  }
  for (std::uint32_t il = end; il < n_phys_; ++il)
    cur = Layer(il, il, cur, blocks, nullptr, s, pos);

  const Vec oq = F("output_res_query.weight");
  Vec x = Mix(blocks, cur, oq.data(), 1);
  Rms(x, F("output_norm.weight").data(), config_.rms_eps);
  const Mat out = M("output.weight");
  s.logits.resize(config_.vocab_unpadded);
  out.MatVec(x.data(), s.logits.data(), 0, config_.vocab_unpadded);
  const float cap = config_.final_logit_softcapping;
  for (auto& l : s.logits)
    l = cap * std::tanh(l / cap);
  ++s.position;
  return s.logits;
}

}  // namespace gufo::models::inkling
