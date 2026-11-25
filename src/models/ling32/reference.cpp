#include "src/models/ling32/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::ling32 {
namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;

constexpr float kWeightNormFloor = 6.103515625e-5F;
constexpr float kClampEps = 1e-6F;

float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

float Silu(float x) {
  return x * Sigmoid(x);
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

// build_gdn_l2_norm: rms_norm(x, eps/n) * 1/sqrt(n) == x/sqrt(sum(x^2)+eps).
void L2(std::span<float> x, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(1.0 / std::sqrt(sum + eps));
  for (auto& v : x)
    v *= inv;
}

// GGML NEOX/MROPE: pair dim i with i + n/2, angle = pos * base^(-2i/n).
void RopeHalf(std::span<float> x, std::uint32_t pos, float base) {
  const std::size_t half = x.size() / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const float angle =
        static_cast<float>(pos) *
        std::pow(static_cast<double>(base),
                 -2.0 * static_cast<double>(i) / static_cast<double>(x.size()));
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float a = x[i];
    const float b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = a * s + b * c;
  }
}

void Swiglu(std::span<const float> gate, std::span<const float> up,
            std::span<float> act, float limit) {
  if (limit > kClampEps) {
    for (std::size_t i = 0; i < act.size(); ++i) {
      const float u = std::clamp(up[i], -limit, limit);
      act[i] = std::min(Silu(gate[i]), limit) * u;
    }
  } else {
    for (std::size_t i = 0; i < act.size(); ++i) {
      act[i] = Silu(gate[i]) * up[i];
    }
  }
}

std::string P(std::uint32_t layer, const char* s) {
  return "blk." + std::to_string(layer) + "." + s;
}

template<typename T>
std::vector<T> TopIndices(std::span<const float> scores, std::size_t k) {
  std::vector<T> idx(scores.size());
  std::iota(idx.begin(), idx.end(), static_cast<T>(0));
  const std::size_t n = std::min(k, idx.size());
  std::partial_sort(idx.begin(), idx.begin() + n, idx.end(), [&](T a, T b) {
    if (scores[a] != scores[b])
      return scores[a] > scores[b];
    return a < b;
  });
  idx.resize(n);
  return idx;
}

}  // namespace

// F32 tensor materialized on demand (or row-view only for huge tensors).
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
    case GgmlType::kQ4_0:
      gufo::quant::DequantizeQ4_0(src, out, cols_);
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
    std::abort();
  }
  return cache_.emplace(name, Loaded::RowView(found->second)).first->second;
}

std::uint32_t ReferenceModel::n_layers() const {
  return config_.TrunkLayers();
}

std::unique_ptr<ReferenceModel::Session> ReferenceModel::CreateSession() const {
  auto s = std::make_unique<Session>();
  s->x.assign(config_.hidden_size, 0.0F);
  s->mtp_prev_h.assign(config_.hidden_size, 0.0F);
  s->mtp_pending_h.assign(config_.hidden_size, 0.0F);
  s->layers.resize(config_.num_layers);
  for (std::uint32_t l = 0; l < config_.num_layers; ++l) {
    if (config_.IsKda(l)) {
      s->layers[l].conv.assign(
          3 * (config_.ssm_conv_kernel - 1) * config_.KdaDim(), 0.0F);
      s->layers[l].state.assign(
          config_.num_heads * config_.kda_head_dim * config_.kda_head_dim,
          0.0F);
    } else {
      s->layers[l].lat.clear();
    }
  }
  return s;
}

void ReferenceModel::KdaBlock(std::uint32_t layer, Session& session,
                              std::span<const float> xn,
                              std::span<float> block) const {
  const std::uint32_t nh = config_.num_heads;
  const std::uint32_t dh = config_.kda_head_dim;
  const std::uint32_t di = config_.KdaDim();
  const std::uint32_t kc = config_.ssm_conv_kernel;
  const float eps = config_.rms_eps;
  auto& st = session.layers[layer];

  const char* conv_names[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight",
                               "ssm_conv1d_v.weight"};
  const char* proj_names[3] = {"attn_q.weight", "attn_k.weight",
                               "attn_v.weight"};
  std::vector<std::vector<float>> qkv(3, std::vector<float>(di));
  for (std::uint32_t w = 0; w < 3; ++w) {
    Load(P(layer, proj_names[w])).MatVec(xn, qkv[w]);
    const std::vector<float>& cw = Load(P(layer, conv_names[w])).data();
    const std::size_t base = static_cast<std::size_t>(w) * (kc - 1) * di;
    std::vector<float> raw(di);
    for (std::uint32_t ch = 0; ch < di; ++ch) {
      const float* tap = cw.data() + static_cast<std::size_t>(ch) * kc;
      float acc = tap[kc - 1] * qkv[w][ch];
      for (std::uint32_t t = 0; t + 1 < kc; ++t) {
        acc += tap[t] * st.conv[base + static_cast<std::size_t>(t) * di + ch];
      }
      raw[ch] = qkv[w][ch];
      qkv[w][ch] = Silu(acc);
    }
    for (std::uint32_t t = 0; t + 2 < kc; ++t) {
      std::copy_n(st.conv.data() + base + static_cast<std::size_t>(t + 1) * di,
                  di, st.conv.data() + base + static_cast<std::size_t>(t) * di);
    }
    std::copy_n(raw.data(), di,
                st.conv.data() + base + static_cast<std::size_t>(kc - 2) * di);
  }

  std::vector<float> fa(di), ga(di), braw(nh);
  Load(P(layer, "ssm_f_a.weight")).MatVec(xn, fa);
  Load(P(layer, "ssm_g_a.weight")).MatVec(xn, ga);
  Load(P(layer, "ssm_beta.weight")).MatVec(xn, braw);
  const float* dt = Load(P(layer, "ssm_dt.bias")).w();
  const float* sa = Load(P(layer, "ssm_a")).w();
  const float* norm_w = Load(P(layer, "ssm_norm.weight")).w();

  const float out_scale = 1.0F / std::sqrt(static_cast<float>(dh));
  std::vector<float> attn_out(di);
  std::vector<float> decay(dh), u(dh), qh(dh), kh(dh);
  for (std::uint32_t h = 0; h < nh; ++h) {
    const std::size_t hoff = static_cast<std::size_t>(h) * dh;
    std::copy_n(qkv[0].data() + hoff, dh, qh.data());
    std::copy_n(qkv[1].data() + hoff, dh, kh.data());
    L2(qh, eps);
    L2(kh, eps);
    const float beta = Sigmoid(braw[h]);
    for (std::uint32_t i = 0; i < dh; ++i) {
      decay[i] = std::exp(config_.gate_lower_bound *
                          Sigmoid((fa[hoff + i] + dt[hoff + i]) * sa[h]));
    }
    float* s = st.state.data() + static_cast<std::size_t>(h) * dh * dh;
    for (std::uint32_t j = 0; j < dh; ++j) {
      float* row = s + static_cast<std::size_t>(j) * dh;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < dh; ++i) {
        row[i] *= decay[i];
        acc += static_cast<double>(row[i]) * kh[i];
      }
      u[j] = static_cast<float>(acc);
    }
    std::span<float> os(attn_out.data() + hoff, dh);
    for (std::uint32_t j = 0; j < dh; ++j) {
      const float delta = (qkv[2][hoff + j] - u[j]) * beta;
      float* row = s + static_cast<std::size_t>(j) * dh;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < dh; ++i) {
        row[i] += delta * kh[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      os[j] = static_cast<float>(acc) * out_scale;
    }
    Rms(os, norm_w, eps);
    for (std::uint32_t j = 0; j < dh; ++j) {
      os[j] *= Sigmoid(ga[hoff + j]);
    }
  }
  Load(P(layer, "attn_output.weight")).MatVec(attn_out, block);
}

void ReferenceModel::MlaBlock(std::uint32_t layer, Session& session,
                              std::span<const float> xn,
                              std::span<float> block) const {
  const std::uint32_t nh = config_.num_heads;
  const std::uint32_t qk = config_.key_length_mla;
  const std::uint32_t rope = config_.rope_dimension;
  const std::uint32_t nope = config_.QkNope();
  const std::uint32_t kv = config_.kv_lora_rank;
  const std::uint32_t vd = config_.MlaVDim();
  const std::uint32_t key = kv + rope;
  const float eps = config_.rms_eps;
  auto& st = session.layers[layer];

  std::vector<float> qall(nh * qk), kvall(kv + rope);
  Load(P(layer, "attn_q.weight")).MatVec(xn, qall);
  Load(P(layer, "attn_kv_a_mqa.weight")).MatVec(xn, kvall);
  Rms(std::span<float>(kvall.data(), kv),
      Load(P(layer, "attn_kv_a_norm.weight")).w(), eps);

  std::vector<float> kpe(kvall.begin() + kv, kvall.end());
  RopeHalf(kpe, session.position, config_.freq_base);
  for (std::uint32_t h = 0; h < nh; ++h) {
    std::span<float> qpe(qall.data() + static_cast<std::size_t>(h) * qk + nope,
                         rope);
    RopeHalf(qpe, session.position, config_.freq_base);
  }

  std::vector<float> key_row(key);
  std::copy_n(kvall.data(), kv, key_row.data());
  std::copy(kpe.begin(), kpe.end(), key_row.data() + kv);
  st.lat.insert(st.lat.end(), key_row.begin(), key_row.end());

  const Loaded& wk = Load(P(layer, "attn_k_b.weight"));
  const Loaded& wv = Load(P(layer, "attn_v_b.weight"));
  std::vector<float> qabs(kv), ctx(kv), outhead(vd);
  std::vector<float> q(key), attn(static_cast<std::size_t>(nh) * vd);
  std::vector<float> gate(nh);
  Load(P(layer, "attn_gate.weight")).MatVec(xn, gate);
  for (std::uint32_t h = 0; h < nh; ++h) {
    wk.MatVecRows(static_cast<std::uint64_t>(h) * kv, kv,
                  std::span<const float>(
                      qall.data() + static_cast<std::size_t>(h) * qk, nope),
                  qabs);
    std::copy(qabs.begin(), qabs.end(), q.data());
    std::copy_n(qall.data() + static_cast<std::size_t>(h) * qk + nope, rope,
                q.data() + kv);

    const std::uint32_t n = static_cast<std::uint32_t>(st.lat.size()) / key;
    std::vector<float> score(n);
    float mx = -1.0e30F;
    const float scale = 1.0F / std::sqrt(static_cast<float>(qk));
    for (std::uint32_t t = 0; t < n; ++t) {
      const float* kr = st.lat.data() + static_cast<std::size_t>(t) * key;
      double acc = 0.0;
      for (std::uint32_t c = 0; c < key; ++c) {
        acc += static_cast<double>(q[c]) * kr[c];
      }
      score[t] = static_cast<float>(acc) * scale;
      mx = std::max(mx, score[t]);
    }
    double denom = 0.0;
    for (auto& v : score) {
      v = std::exp(v - mx);
      denom += v;
    }
    std::fill(ctx.begin(), ctx.end(), 0.0F);
    for (std::uint32_t t = 0; t < n; ++t) {
      const float p = static_cast<float>(score[t] / denom);
      const float* vr = st.lat.data() + static_cast<std::size_t>(t) * key;
      for (std::uint32_t c = 0; c < kv; ++c) {
        ctx[c] += p * vr[c];
      }
    }
    wv.MatVecRows(static_cast<std::uint64_t>(h) * vd, vd, ctx, outhead);
    const float g = Sigmoid(gate[h]);
    std::transform(outhead.begin(), outhead.end(),
                   attn.begin() + static_cast<std::size_t>(h) * vd,
                   [g](float v) { return v * g; });
  }
  Load(P(layer, "attn_output.weight")).MatVec(attn, block);
}

void ReferenceModel::FfnBlock(std::uint32_t layer, std::span<const float> fin,
                              std::span<float> ffn_out) const {
  if (config_.IsDense(layer)) {
    const std::uint32_t ff = config_.dense_ff;
    std::vector<float> gate(ff), up(ff), act(ff);
    Load(P(layer, "ffn_gate.weight")).MatVec(fin, gate);
    Load(P(layer, "ffn_up.weight")).MatVec(fin, up);
    Swiglu(gate, up, act, 0.0F);
    Load(P(layer, "ffn_down.weight")).MatVec(act, ffn_out);
    return;
  }

  const std::uint32_t ne = config_.expert_count;
  const std::uint32_t used = config_.expert_used;
  const std::uint32_t groups = config_.expert_groups;
  const std::uint32_t groups_used = config_.expert_groups_used;
  const std::uint32_t eff = config_.expert_ff;
  const std::uint32_t h = config_.hidden_size;
  std::vector<float> logits(ne), probs(ne), sel(ne);
  Load(P(layer, "ffn_gate_inp.weight")).MatVec(fin, logits);
  const float* eb = Load(P(layer, "exp_probs_b.bias")).w();
  for (std::uint32_t e = 0; e < ne; ++e) {
    probs[e] = Sigmoid(logits[e]);
    sel[e] = probs[e] + eb[e];
  }

  const std::uint32_t per = ne / groups;
  std::vector<float> gscore(groups, 0.0F);
  for (std::uint32_t g = 0; g < groups; ++g) {
    std::vector<float> sorted(per);
    for (std::uint32_t i = 0; i < per; ++i) {
      sorted[i] = sel[g * per + i];
    }
    std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(),
                      std::greater<float>());
    gscore[g] = sorted[0] + sorted[1];
  }
  const auto topg = TopIndices<std::uint32_t>(gscore, groups_used);
  std::vector<bool> allowed(ne, false);
  for (const auto g : topg) {
    for (std::uint32_t i = 0; i < per; ++i) {
      allowed[g * per + i] = true;
    }
  }
  std::vector<float> masked(ne);
  for (std::uint32_t e = 0; e < ne; ++e) {
    masked[e] = allowed[e] ? sel[e] : -std::numeric_limits<float>::infinity();
  }
  const auto top = TopIndices<std::uint32_t>(masked, used);

  float wsum = 0.0F;
  for (const auto e : top) {
    wsum += probs[e];
  }
  wsum = std::max(wsum, kWeightNormFloor);
  const float clamp_exp = config_.swiglu_clamp_exp[layer];
  const float clamp_shared = config_.swiglu_clamp_shexp[layer];
  const Loaded& wg = LoadRow(P(layer, "ffn_gate_exps.weight"));
  const Loaded& wu = LoadRow(P(layer, "ffn_up_exps.weight"));
  const Loaded& wd = LoadRow(P(layer, "ffn_down_exps.weight"));
  std::vector<float> gf(eff), uf(eff), act(eff), part(h);
  std::fill(ffn_out.begin(), ffn_out.end(), 0.0F);
  for (const auto e : top) {
    const float weight = probs[e] / wsum * config_.expert_weights_scale;
    wg.MatVecView(static_cast<std::uint64_t>(e) * eff, eff, fin, gf);
    wu.MatVecView(static_cast<std::uint64_t>(e) * eff, eff, fin, uf);
    Swiglu(gf, uf, act, clamp_exp);
    wd.MatVecView(static_cast<std::uint64_t>(e) * h, h, act, part);
    for (std::uint32_t c = 0; c < h; ++c) {
      ffn_out[c] += weight * part[c];
    }
  }

  std::vector<float> sg(eff), su(eff);
  Load(P(layer, "ffn_gate_shexp.weight")).MatVec(fin, sg);
  Load(P(layer, "ffn_up_shexp.weight")).MatVec(fin, su);
  Swiglu(sg, su, act, clamp_shared);
  Load(P(layer, "ffn_down_shexp.weight")).MatVec(act, part);
  for (std::uint32_t c = 0; c < h; ++c) {
    ffn_out[c] += part[c];
  }
}

void ReferenceModel::AttnBlock(std::uint32_t layer, Session& session,
                               std::span<const float> xn,
                               std::span<float> block) const {
  if (config_.IsKda(layer)) {
    KdaBlock(layer, session, xn, block);
  } else {
    MlaBlock(layer, session, xn, block);
  }
}

const std::vector<float>& ReferenceModel::Evaluate(Session& session,
                                                   std::int32_t token) const {
  if (session.mtp_has_prev) {
    MtpForward(session, session.mtp_prev_token, session.mtp_pending_h,
               session.position - 1, false);
  }

  const GgufTensorInfo& embd = *tensors_.at("token_embd.weight");
  const Loaded row_view = Loaded::RowView(&embd);
  row_view.DequantizeRow(static_cast<std::uint64_t>(token), session.x.data());

  const float eps = config_.rms_eps;
  std::vector<float> xn(config_.hidden_size), block(config_.hidden_size);
  std::vector<float> fin(config_.hidden_size), ffn(config_.hidden_size);
  for (std::uint32_t layer = 0; layer < n_layers(); ++layer) {
    std::copy(session.x.begin(), session.x.end(), xn.begin());
    Rms(xn, Load(P(layer, "attn_norm.weight")).w(), eps);
    AttnBlock(layer, session, xn, block);
    for (std::uint32_t i = 0; i < config_.hidden_size; ++i) {
      session.x[i] += block[i];
    }

    std::copy(session.x.begin(), session.x.end(), fin.begin());
    Rms(fin, Load(P(layer, "ffn_norm.weight")).w(), eps);
    FfnBlock(layer, fin, ffn);
    for (std::uint32_t i = 0; i < config_.hidden_size; ++i) {
      session.x[i] += ffn[i];
    }
  }

  Rms(session.x, Load("output_norm.weight").w(), eps);
  session.mtp_h.assign(session.x.begin(), session.x.end());
  session.mtp_pending_h.assign(session.mtp_prev_h.begin(),
                               session.mtp_prev_h.end());
  session.mtp_prev_h.assign(session.mtp_h.begin(), session.mtp_h.end());
  session.mtp_prev_token = token;
  session.mtp_has_prev = true;
  const Loaded& head_w = Load("output.weight");
  session.logits.resize(static_cast<std::size_t>(head_w.rows()));
  head_w.MatVec(session.x, session.logits);
  ++session.position;
  return session.logits;
}

const std::vector<float>& ReferenceModel::MtpForward(
    Session& session, std::int32_t token, std::span<const float> h_prev,
    std::uint32_t position, bool emit_logits) const {
  const std::uint32_t layer = config_.MtpLayer();
  const std::uint32_t h = config_.hidden_size;
  const float eps = config_.rms_eps;

  std::vector<float> embd(h), hn(h), eh(2U * h), cur(h), block(h);
  std::vector<float> fin(h), ffn(h);
  Loaded::RowView(&*tensors_.at("token_embd.weight"))
      .DequantizeRow(static_cast<std::uint64_t>(token), embd.data());
  Rms(embd, Load(P(layer, "nextn.enorm.weight")).w(), eps);
  std::copy(h_prev.begin(), h_prev.end(), hn.begin());
  Rms(hn, Load(P(layer, "nextn.hnorm.weight")).w(), eps);
  std::copy(embd.begin(), embd.end(), eh.begin());
  std::copy(hn.begin(), hn.end(), eh.begin() + h);
  Load(P(layer, "nextn.eh_proj.weight")).MatVec(eh, cur);

  std::copy(cur.begin(), cur.end(), fin.begin());
  Rms(fin, Load(P(layer, "attn_norm.weight")).w(), eps);
  const std::uint32_t saved = session.position;
  session.position = position;
  MlaBlock(layer, session, fin, block);
  session.position = saved;
  for (std::uint32_t i = 0; i < h; ++i)
    cur[i] += block[i];

  std::copy(cur.begin(), cur.end(), fin.begin());
  Rms(fin, Load(P(layer, "ffn_norm.weight")).w(), eps);
  FfnBlock(layer, fin, ffn);
  for (std::uint32_t i = 0; i < h; ++i)
    cur[i] += ffn[i];
  Rms(cur, Load(P(layer, "layer_output_norm.weight")).w(), eps);
  if (!emit_logits)
    return session.mtp_logits;
  session.mtp_h.assign(cur.begin(), cur.end());

  const Loaded& head_w = Load("output.weight");
  session.mtp_logits.resize(static_cast<std::size_t>(head_w.rows()));
  head_w.MatVec(cur, session.mtp_logits);
  return session.mtp_logits;
}

const std::vector<float>& ReferenceModel::MtpEvaluate(
    Session& session, std::int32_t token, std::span<const float> h_prev,
    std::uint32_t position) const {
  return MtpForward(session, token, h_prev, position, true);
}

}  // namespace gufo::models::ling32
