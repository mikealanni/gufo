#include "src/models/inkling/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::inkling {
namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;

float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

void RmsInplace(std::span<float> x, const float* w, float eps) {
  double sum = 0.0;
  for (const auto v : x) sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * inv * (w != nullptr ? w[i] : 1.0F);
  }
}

void RmsNorm(std::span<float> x, float eps) { RmsInplace(x, nullptr, eps); }

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t u32 = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &u32, sizeof(float));
  return f;
}

float E2M1Val(std::uint8_t nib) {
  static const float vals[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return vals[nib & 0xF];
}

float Ue4M3Val(std::uint8_t x) {
  if (x == 0 || x == 0x7F) {
    return 0.0F;
  }
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  float raw = (exp == 0) ? std::ldexp(static_cast<float>(man), -9) : std::ldexp(1.0F + static_cast<float>(man) / 8.0F, exp - 7);
  return raw * 0.5F;
}

void DeqNvfp4Row(const GgufTensorInfo* t, std::uint64_t row, std::uint64_t cols, float* out) {
  const std::uint64_t rowbytes = t->size_bytes / std::max<std::uint64_t>(rowspan_of(t), 1ULL);
  const char* src = static_cast<const char*>(t->data) + static_cast<std::size_t>(row) * static_cast<std::size_t>(rowbytes);
  const int nblocks = static_cast<int>(cols / 64);
  for (int b = 0; b < nblocks; ++b) {
    const std::uint8_t* blk = reinterpret_cast<const std::uint8_t*>(src) + static_cast<std::size_t>(b) * 36;
    const float d = Ue4M3Val(blk[0]);
    const float d1 = Ue4M3Val(blk[1]);
    const float d2 = Ue4M3Val(blk[2]);
    const float d3 = Ue4M3Val(blk[3]);
    const std::uint8_t* qs = blk + 4;
    for (int s = 0; s < 4; ++s) {
      const float dscale = (s == 0) ? d : (s == 1) ? d1 : (s == 2) ? d2 : d3;
      for (int j = 0; j < 8; ++j) {
        const int e = s * 16 + j;
        const std::uint8_t q = qs[(std::size_t)(s * 8) + j];
        out[e] = E2M1Val(q & 0xF) * dscale;
        const int e2 = s * 16 + 8 + j;
        if (e2 < static_cast<int>(cols)) {
          const std::uint8_t q2 = qs[(std::size_t)(s * 8) + j];
          out[e2] = E2M1Val(q2 >> 4) * dscale;
        }
      }
    }
  }
}

std::string P(std::uint32_t layer, const char* s) {
  return "blk." + std::to_string(layer) + "." + s;
}

}  // namespace

ReferenceModel::Loaded::Loaded(const GgufTensorInfo* t) { Setup(t); }

void ReferenceModel::Loaded::Setup(const GgufTensorInfo* t) {
  t_ = t;
  rows_ = 1;
  for (std::size_t d = 1; d < t->dimensions.size(); ++d) {
    rows_ *= t->dimensions[d];
  }
  cols_ = t->dimensions[0];
}

void ReferenceModel::Loaded::RowView::RowView(const GgufTensorInfo* t) { Setup(t); }

void ReferenceModel::Loaded::DequantizeRow(std::uint64_t row, float* out) const {
  const std::uint64_t rb = static_cast<std::size_t>(t_->size_bytes / std::max<std::uint64_t>(rows_, 1));
  const char* src = static_cast<const char*>(t_->data) + row * rb;
  switch (t_->type) {
    case GgmlType::kF32:
      std::memcpy(out, src, cols_ * sizeof(float));
      break;
    case GgmlType::kBF16: {
      const auto* p = reinterpret_cast<const std::uint16_t*>(src);
      for (std::uint64_t i = 0; i < cols_; ++i) out[i] = Bf16ToFloat(p[i]);
      break;
    }
    case GgmlType::kQ4_K: gufo::quant::DequantizeQ4_K(src, out, cols_); break;
    case GgmlType::kQ5_K: gufo::quant::DequantizeQ5_K(src, out, cols_); break;
    case GgmlType::kQ6_K: gufo::quant::DequantizeQ6_K(src, out, cols_); break;
    case GgmlType::kQ8_K: gufo::quant::DequantizeQ8_K(src, out, cols_); break;
    case GgmlType::kQ8_0: gufo::quant::DequantizeQ8_0(src, out, cols_); break;
    case GgmlType::kIQ4_XS: gufo::quant::DequantizeIQ4_XS(src, out, cols_); break;
    case GgmlType::kIQ3_S: gufo::quant::DequantizeIQ3_S(src, out, cols_); break;
    default:
      break;
  }
}

void ReferenceModel::Loaded::MatVec(std::span<const float> x, std::span<float> y) const {
  for (std::uint64_t r = 0; r < rows_; ++r) {
    const float* w = data_.data() + r * cols_;
    double acc = 0.0;
    for (std::uint64_t i = 0; i < cols_; ++i) acc += static_cast<double>(w[i]) * x[i];
    y[r] = static_cast<float>(acc);
  }
}

void ReferenceModel::Loaded::MatVecRow(std::uint64_t row, std::span<const float> x, float* out) const {
  const float* w = data_.data() + row * cols_;
  double acc = 0.0;
  for (std::uint64_t i = 0; i < cols_; ++i) acc += static_cast<double>(w[i]) * x[i];
  out[0] = static_cast<float>(acc);
}

const ReferenceModel::Loaded& ReferenceModel::Load(const std::string& name) const {
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

const ReferenceModel::Loaded& ReferenceModel::LoadRow(const std::string& name) const {
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

const GgufTensorInfo* ReferenceModel::Find(const std::string& suffix) const {
  for (const auto& entry : tensors_) {
    if (std::string(entry.first).find(suffix) != std::string::npos) {
      return entry.second;
    }
  }
  return nullptr;
}

void ReferenceModel::Rms(std::span<float> x, const float* w) const { RmsInplace(x, w, config_.rms_eps); }
void ReferenceModel::RmsNorm(std::span<float> x) const { RmsInplace(x, nullptr, config_.rms_eps); }

std::vector<float> ReferenceModel::Mix(std::uint32_t hidden, const std::vector<float>& cur,
                                      const std::vector<float>& query, const std::vector<std::vector<float>>& bank) const {
  std::vector<std::vector<float>> states;
  for (const auto& b : bank) {
    std::vector<float> n = b;
    RmsNorm(n, 1e-6F);
    states.push_back(n);
  }
  std::vector<float> n = cur;
  RmsNorm(n, 1e-6F);
  states.push_back(n);
  const float scale = 1.0F / std::sqrt(static_cast<float>(hidden));
  std::vector<float> scores(states.size(), -1.0e30F);
  float maxs = -1.0e30F;
  for (std::size_t i = 0; i < states.size(); ++i) {
    double dot = 0.0;
    for (std::uint32_t j = 0; j < hidden; ++j) dot += static_cast<double>(states[i][j]) * query[j];
    scores[i] = static_cast<float>(dot) * scale;
    maxs = std::max(maxs, scores[i]);
  }
  std::vector<float> w(states.size());
  double denom = 0.0;
  for (std::size_t i = 0; i < states.size(); ++i) {
    w[i] = std::exp(scores[i] - maxs);
    denom += w[i];
  }
  const float inv = 1.0F / static_cast<float>(denom);
  std::vector<float> out(hidden, 0.0F);
  for (std::size_t i = 0; i < states.size(); ++i) {
    const float wi = w[i] * inv;
    for (std::uint32_t j = 0; j < hidden; ++j) out[j] += wi * states[i][j];
  }
  return out;
}

}  // namespace gufo::models::inkling
