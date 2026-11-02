#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROPE_YARN_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROPE_YARN_HPP_

#include <cstdint>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// YaRN RoPE frequency scaling for the Flash-Next text path.
///
/// Flash-Next is native to 262,144 tokens and Qwen's own card recommends
/// YaRN with `factor = 4.0` to reach ~1,048,576. YaRN replaces the plain
/// per-pair inverse frequency `theta^(-2i/rotary_dim)` with a blend of the
/// interpolated (`/factor`) and extrapolated frequencies, governed by the
/// `beta_fast`/`beta_slow` wavelength correction range, and applies a scalar
/// `mscale` to the rotation's cos/sin (the attention-score correction).
///
/// This mirrors the engine's own reference in `models/laguna`
/// (`device_model.cpp` `InvFreq` and `reference.cpp` `Rope`) so the two
/// models agree on the math. The blend is purely per-dimension: it is
/// orthogonal to the position (text uses `RopePosition(...) == physical`),
/// so it composes with the interleaved mRoPE path without change.
struct RopeScaling {
  /// Blended inverse frequency for each rotated pair, length `rotary_dim/2`.
  std::vector<float> inv_freq;
  /// Scalar folded into cos/sin of every rotation (attention-score scale).
  float mscale{1.0F};
  /// False when `factor <= 1`: callers then fall back to plain theta math so
  /// the output is byte-identical to the pre-YaRN path.
  bool enabled{false};
};

/// Computes the YaRN inverse-frequency table and `mscale` for one RoPE base
/// over `rotary_dim` (NEOX partial rotation over the first `rotary_dim` of
/// each head). `factor <= 1` returns `enabled == false` with an empty table.
///
/// The attention-score scale uses the derivation the Flash-Next runtime
/// shares with Laguna: `mscale = 1 + 0.1 * ln(factor)` (the GGUF
/// `rope.scaling.attn_factor` key is intentionally not used because it
/// differs between quantizations).
[[nodiscard]] RopeScaling ComputeYarn(float rope_theta,
                                      std::uint32_t rotary_dim, float factor,
                                      std::uint32_t orig_ctx, float beta_fast,
                                      float beta_slow);

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROPE_YARN_HPP_
