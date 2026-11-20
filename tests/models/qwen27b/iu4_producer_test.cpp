// Checks the norm-fed int4 activation producer against a double precision host
// model: RMSNorm, AWQ divide, signed Hadamard rotation over 256 and per-128
// symmetric int4 quantization, written in the tile-major layout the GEMM reads.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/qwen/hip/iu4_weights.hpp"
#include "src/models/qwen/hip/ops/iu4.hpp"

namespace {

constexpr int kSkip = 77;
constexpr std::size_t kTokens = 300, kPadded = 384, kK = 5120;
int failures = 0;

#define CHECK(cond, ...)               \
  do {                                 \
    if (!(cond)) {                     \
      std::printf("FAIL %s: ", #cond); \
      std::printf(__VA_ARGS__);        \
      std::printf("\n");               \
      ++failures;                      \
    }                                  \
  } while (0)

void Hadamard256(double* v) {
  for (int stride = 1; stride < 256; stride <<= 1)
    for (int i = 0; i < 256; ++i)
      if ((i & stride) == 0) {
        const double a = v[i], b = v[i + stride];
        v[i] = a + b;
        v[i + stride] = a - b;
      }
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_IU4_ARTIFACT");
  if (path == nullptr) {
    std::printf("GUFO_IU4_ARTIFACT not set\n");
    return kSkip;
  }
  gufo::hip::Iu4Weights weights;
  std::string error;
  if (!weights.Load(path, 4, &error)) {
    std::printf("load failed: %s\n", error.c_str());
    return 1;
  }
  const auto& proj = weights.Find(3, gufo::hip::Iu4Role::kFfnGate);
  std::vector<float> awq(kK), signs(512);
  CHECK(hipMemcpy(awq.data(), proj.awq, kK * 4, hipMemcpyDeviceToHost) ==
            hipSuccess,
        "awq copy");
  CHECK(hipMemcpy(signs.data(), weights.signs(), 512 * 4,
                  hipMemcpyDeviceToHost) == hipSuccess,
        "signs copy");

  std::uint64_t state = 88172645463325252ull;
  const auto uniform = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
  };
  std::vector<float> x(kTokens * kK), norm(kK);
  for (auto& v : x)
    v = static_cast<float>(std::sqrt(-2 * std::log(uniform() + 1e-12)) *
                           std::cos(6.283185307179586 * uniform()));
  for (std::size_t t = 0; t < kTokens; ++t)
    for (int c = 0; c < 8; ++c)
      x[t * kK + (c * 613) % kK] *= 30.0f;
  for (auto& w : norm)
    w = static_cast<float>(0.5 + uniform());

  const std::size_t e_groups = kK / 128;
  float *d_x, *d_norm, *d_scales;
  std::uint8_t* d_tiles;
  const std::size_t tile_bytes = (kPadded / 16) * e_groups * 1024;
  CHECK(hipMalloc(&d_x, x.size() * 4) == hipSuccess, "alloc");
  CHECK(hipMalloc(&d_norm, kK * 4) == hipSuccess, "alloc");
  CHECK(hipMalloc(&d_scales, e_groups * kPadded * 4) == hipSuccess, "alloc");
  CHECK(hipMalloc(&d_tiles, tile_bytes) == hipSuccess, "alloc");
  hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(d_norm, norm.data(), kK * 4, hipMemcpyHostToDevice);
  hipMemset(d_tiles, 0xAB, tile_bytes);
  hipMemset(d_scales, 0xFF, e_groups * kPadded * 4);
  gufo::hip::LaunchIu4NormQuantize(d_x, d_norm, proj.awq, weights.signs(),
                                   1e-6f, kTokens, kPadded, kK, d_tiles,
                                   d_scales, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess, "producer failed");
  std::vector<std::uint8_t> tiles(tile_bytes);
  std::vector<float> scales(e_groups * kPadded);
  hipMemcpy(tiles.data(), d_tiles, tile_bytes, hipMemcpyDeviceToHost);
  hipMemcpy(scales.data(), d_scales, scales.size() * 4, hipMemcpyDeviceToHost);

  std::size_t compared = 0, off_by_one = 0, bigger = 0;
  double worst_scale = 0.0;
  std::vector<double> v(kK);
  for (std::size_t t = 0; t < kPadded; ++t) {
    if (t < kTokens) {
      double ss = 0.0;
      for (std::size_t k = 0; k < kK; ++k)
        ss += static_cast<double>(x[t * kK + k]) * x[t * kK + k];
      const double inv = 1.0 / std::sqrt(ss / kK + 1e-6);
      for (std::size_t b = 0; b < kK / 256; ++b) {
        double blk[256];
        for (int i = 0; i < 256; ++i) {
          const std::size_t k = b * 256 + i;
          blk[i] = x[t * kK + k] * inv * norm[k] / awq[k] * signs[i];
        }
        Hadamard256(blk);
        for (int i = 0; i < 256; ++i)
          v[b * 256 + i] = blk[i] * 0.0625 * signs[256 + i];
      }
    } else {
      std::fill(v.begin(), v.end(), 0.0);
    }
    for (std::size_t e = 0; e < e_groups; ++e) {
      double amax = 0.0;
      for (int i = 0; i < 128; ++i)
        amax = std::max(amax, std::fabs(v[e * 128 + i]));
      const double scale = amax / 7.0;
      const float got_scale = scales[e * kPadded + t];
      if (scale > 0.0)
        worst_scale =
            std::max(worst_scale, std::fabs(got_scale - scale) / scale);
      else
        CHECK(got_scale == 0.0f, "pad scale %zu %zu: %g", t, e, got_scale);
      for (int i = 0; i < 128; ++i) {
        // K value i of the group sits in payload byte i/2, nibble i%2; the
        // GEMM layout keeps 8-byte chunks: chunk c -> (c&1)*16 lane offset and
        // (c>>1)*8 byte offset inside a 32-byte lane record.
        const int byte = i / 2, chunk = byte / 8, within = byte % 8;
        const std::size_t off = ((t / 16) * e_groups + e) * 1024 +
                                ((chunk & 1) * 16 + (t % 16)) * 32 +
                                (chunk >> 1) * 8 + within;
        const int code4 = (tiles[off] >> (4 * (i & 1))) & 15;
        const int code = code4 >= 8 ? code4 - 16 : code4;
        const double want_real = scale > 0.0 ? v[e * 128 + i] / scale : 0.0;
        const int want = static_cast<int>(
            std::min(7.0, std::max(-8.0, std::nearbyint(want_real))));
        ++compared;
        if (code != want) {
          if (std::abs(code - want) == 1 &&
              std::fabs(std::fabs(want_real - std::floor(want_real) - 0.5)) <
                  1e-3)
            ++off_by_one;
          else
            ++bigger;
        }
      }
    }
  }
  std::printf(
      "compared %zu codes: %zu rounding-boundary flips, %zu real "
      "mismatches, worst scale error %.2e\n",
      compared, off_by_one, bigger, worst_scale);
  CHECK(bigger == 0, "codes differ beyond rounding boundaries");
  CHECK(worst_scale < 1e-5, "scale error %.3e", worst_scale);
  CHECK(off_by_one < compared / 10000, "too many boundary flips");
  if (failures != 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
