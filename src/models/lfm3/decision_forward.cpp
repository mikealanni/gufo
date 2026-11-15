#include "src/models/lfm3/decision_forward.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
namespace gufo::models::lfm3 {
namespace {
using Matrix = std::vector<float>;

template<typename Fn>
void ParallelFor(std::size_t count, unsigned threads, Fn fn) {
  std::atomic<std::size_t> next{0};
  auto work = [&] {
    for (std::size_t i; (i = next.fetch_add(1)) < count;)
      fn(i);
  };
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < std::max(1U, threads); ++t)
    pool.emplace_back(work);
  work();
  for (auto& t : pool)
    t.join();
}
// Y[n][rows] = X[n][cols] * W^T; each weight row is decoded once.
Matrix Project(const Tensor& w, const Matrix& x, std::size_t n,
               unsigned threads) {
  Matrix y(n * w.rows);
  ParallelFor(w.rows, threads, [&](std::size_t r) {
    std::vector<float> row(w.cols);
    DecodeRow(w, static_cast<std::uint32_t>(r), row);
    for (std::size_t t = 0; t < n; ++t) {
      const float* in = x.data() + t * w.cols;
      float sum = 0;
      for (std::uint32_t k = 0; k < w.cols; ++k)
        sum += row[k] * in[k];
      y[t * w.rows + r] = sum;
    }
  });
  return y;
}
void Norm(float* x, std::size_t count, const Tensor& w, float eps) {
  float sum = 0;
  for (std::size_t i = 0; i < count; ++i)
    sum += x[i] * x[i];
  const float scale = 1 / std::sqrt(sum / count + eps);
  const auto* weight = static_cast<const float*>(w.data);
  for (std::size_t i = 0; i < count; ++i)
    x[i] = x[i] * scale * weight[i];
}
void NormRows(Matrix& x, std::size_t n, std::size_t width, const Tensor& w,
              float eps) {
  for (std::size_t t = 0; t < n; ++t)
    Norm(x.data() + t * width, width, w, eps);
}
void Rope(float* x, std::size_t heads, const Config& c, std::uint32_t position,
          bool swa, const Tensor& factors) {
  const auto D = c.head_dim;
  const float theta = swa ? c.theta_swa : c.theta;
  for (std::size_t h = 0; h < heads; ++h) {
    float* p = x + h * D;
    for (std::uint32_t i = 0; i < D / 2; ++i) {
      float angle = position * std::pow(theta, -2.0F * i / D);
      if (!swa)
        angle /= static_cast<const float*>(factors.data)[i];
      const float a = p[i], b = p[i + D / 2];
      p[i] = a * std::cos(angle) - b * std::sin(angle);
      p[i + D / 2] = b * std::cos(angle) + a * std::sin(angle);
    }
  }
}
}  // namespace

std::vector<float> DecisionForward(const Weights& w,
                                   std::span<const std::uint32_t> tokens,
                                   std::span<const std::int32_t> groups,
                                   std::span<const std::uint32_t> slots,
                                   std::span<const std::uint32_t> vocabulary,
                                   bool causal, unsigned threads,
                                   std::vector<float>* final_hidden) {
  const auto& c = w.config;
  const std::size_t n = tokens.size(), H = c.hidden;
  if (!n || slots.empty() || (!groups.empty() && groups.size() != n))
    throw std::invalid_argument("invalid decision forward arguments");
  for (auto id : tokens)
    if (id >= c.vocab)
      throw std::invalid_argument("token out of range");
  for (auto s : slots)
    if (s >= n)
      throw std::invalid_argument("slot out of range");
  Matrix x(n * H);
  for (std::size_t t = 0; t < n; ++t)
    DecodeRow(w.embedding, tokens[t], std::span(x).subspan(t * H, H));
  for (std::size_t il = 0; il < c.layers.size(); ++il) {
    const auto& l = w.layers[il];
    Matrix z = x;
    NormRows(z, n, H, l.norm, c.eps);
    Matrix out;
    if (c.layers[il] == LayerType::kConvolution) {
      const Matrix p = Project(l.in_proj, z, n, threads);
      const std::size_t P = 3 * H, L = c.conv_width;
      const auto* weight = static_cast<const float*>(l.conv.data);
      Matrix conv(n * H);
      for (std::size_t t = 0; t < n; ++t)
        for (std::size_t ch = 0; ch < H; ++ch) {
          float sum = 0;
          for (std::size_t j = 0; j < L; ++j) {
            const std::ptrdiff_t s =
                static_cast<std::ptrdiff_t>(t) - (L - 1) + j;
            if (s < 0)
              continue;
            sum += weight[ch * L + j] * p[s * P + ch] * p[s * P + 2 * H + ch];
          }
          conv[t * H + ch] = sum * p[t * P + H + ch];
        }
      out = Project(l.out_proj, conv, n, threads);
    } else {
      const std::size_t kh_count = c.kv_heads[il], D = c.head_dim;
      Matrix q = Project(l.q, z, n, threads), k = Project(l.k, z, n, threads),
             v = Project(l.v, z, n, threads),
             gate = Project(l.attn_gate, z, n, threads);
      const bool swa = c.layers[il] == LayerType::kSlidingAttention;
      for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t h = 0; h < c.heads; ++h)
          Norm(q.data() + t * c.heads * D + h * D, D, l.q_norm, c.eps);
        for (std::size_t h = 0; h < kh_count; ++h)
          Norm(k.data() + t * kh_count * D + h * D, D, l.k_norm, c.eps);
        Rope(q.data() + t * c.heads * D, c.heads, c, t, swa, w.rope_freqs);
        Rope(k.data() + t * kh_count * D, kh_count, c, t, swa, w.rope_freqs);
      }
      Matrix a(n * c.heads * D);
      const std::size_t per_group = c.heads / kh_count;
      ParallelFor(n * c.heads, threads, [&](std::size_t item) {
        const std::size_t t = item / c.heads, h = item % c.heads;
        const std::size_t kh = h / per_group;
        const std::size_t first =
            swa && t + 1 > c.window ? t + 1 - c.window : 0;
        const std::size_t last = (swa || causal) ? t : n - 1;
        std::vector<float> scores;
        std::vector<std::size_t> keys;
        float max = -std::numeric_limits<float>::infinity();
        const float* qv = q.data() + t * c.heads * D + h * D;
        for (std::size_t s = first; s <= last; ++s) {
          if (!swa && !causal && !groups.empty()) {
            const auto own = groups[t], key = groups[s];
            if (!(own < 0 || key < 0 || own == key))
              continue;
          }
          const float* kv = k.data() + s * kh_count * D + kh * D;
          float score = 0;
          for (std::size_t d = 0; d < D; ++d)
            score += qv[d] * kv[d];
          score /= std::sqrt(float(D));
          scores.push_back(score);
          keys.push_back(s);
          max = std::max(max, score);
        }
        float den = 0;
        for (auto& s : scores) {
          s = std::exp(s - max);
          den += s;
        }
        const float g = 1 / (1 + std::exp(-gate[t * c.heads + h]));
        float* av = a.data() + t * c.heads * D + h * D;
        for (std::size_t d = 0; d < D; ++d) {
          float sum = 0;
          for (std::size_t i = 0; i < keys.size(); ++i)
            sum += (scores[i] / den) * v[keys[i] * kh_count * D + kh * D + d];
          av[d] = sum * g;
        }
      });
      out = Project(l.attn_out, a, n, threads);
    }
    if (l.post_norm.data)
      NormRows(out, n, H, l.post_norm, c.eps);
    for (std::size_t i = 0; i < n * H; ++i)
      x[i] += out[i];
    z = x;
    NormRows(z, n, H, l.ffn_norm, c.eps);
    Matrix gate = Project(l.gate, z, n, threads),
           up = Project(l.up, z, n, threads);
    for (std::size_t i = 0; i < gate.size(); ++i)
      gate[i] = (gate[i] / (1 + std::exp(-gate[i]))) * up[i];
    out = Project(l.down, gate, n, threads);
    if (l.post_ffn_norm.data)
      NormRows(out, n, H, l.post_ffn_norm, c.eps);
    for (std::size_t i = 0; i < n * H; ++i)
      x[i] += out[i];
    if (std::getenv("GUFO_D1_TRACE")) {
      double mean = 0;
      for (float v : x)
        mean += std::abs(v);
      std::fprintf(
          stderr, "[d1 host] layer %zu mean|x| %.5f last %.4f %.4f %.4f %.4f\n",
          il, mean / x.size(), x[(n - 1) * H], x[(n - 1) * H + 1],
          x[(n - 1) * H + 2], x[(n - 1) * H + 3]);
    }
  }
  Matrix hidden(slots.size() * H);
  for (std::size_t s = 0; s < slots.size(); ++s) {
    std::copy_n(x.begin() + slots[s] * H, H, hidden.begin() + s * H);
    Norm(hidden.data() + s * H, H, w.output_norm, c.eps);
  }
  if (final_hidden != nullptr)
    *final_hidden = hidden;
  const std::size_t cols = vocabulary.empty() ? c.vocab : vocabulary.size();
  std::vector<float> logits(slots.size() * cols);
  ParallelFor(cols, threads, [&](std::size_t j) {
    std::vector<float> row(H);
    DecodeRow(w.output, vocabulary.empty() ? j : vocabulary[j], row);
    for (std::size_t s = 0; s < slots.size(); ++s) {
      float sum = 0;
      for (std::size_t k = 0; k < H; ++k)
        sum += row[k] * hidden[s * H + k];
      logits[s * cols + j] = sum;
    }
  });
  return logits;
}
}  // namespace gufo::models::lfm3
