#include "src/models/qwen38_flash_next/rope_yarn.hpp"

#include <algorithm>
#include <cmath>

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

RopeScaling ComputeYarn(float rope_theta, std::uint32_t rotary_dim,
                        float factor, std::uint32_t orig_ctx, float beta_fast,
                        float beta_slow) {
  RopeScaling out;
  if (factor <= 1.0F || rotary_dim < 2 || orig_ctx == 0) {
    return out;  // enabled == false, empty table
  }
  const std::uint32_t half = rotary_dim / 2;
  const float n_dims = static_cast<float>(rotary_dim);
  const float freq_scale = 1.0F / factor;

  // Correction range bounds: wavelengths that fall outside
  // [beta_slow*2pi, beta_fast*2pi] are fully interpolated or extrapolated.
  const auto corr_dim = [&](float beta) {
    return static_cast<float>(
        static_cast<double>(n_dims) *
        std::log(static_cast<double>(orig_ctx) /
                 (static_cast<double>(beta) * 2.0 * kPi)) /
        (2.0 * std::log(static_cast<double>(rope_theta))));
  };
  const float low = std::max(0.0F, std::floor(corr_dim(beta_fast)));
  const float high = std::min(n_dims - 1.0F, std::ceil(corr_dim(beta_slow)));

  out.inv_freq.resize(half);
  for (std::uint32_t i = 0; i < half; ++i) {
    const float extrap =
        std::pow(rope_theta, -static_cast<float>(2 * i) / n_dims);
    const float interp = freq_scale * extrap;
    const float y =
        (static_cast<float>(i) - low) / std::max(0.001F, high - low);
    const float ramp = 1.0F - std::min(1.0F, std::max(0.0F, y));
    // Blend extrapolated (ramp) and interpolated (1-ramp); factor->1 gives the
    // plain frequency, factor->inf gives the fully interpolated one.
    out.inv_freq[i] = interp * (1.0F - ramp) + extrap * ramp;
  }
  out.mscale = 1.0F + 0.1F * std::log(factor);
  out.enabled = true;
  return out;
}

}  // namespace gufo::models::qwen38_flash_next
