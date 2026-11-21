// Checks the norm-fed int4 activation producer against a double precision host
// model: RMSNorm, AWQ divide, signed Hadamard rotation over 256 and per-128
// symmetric int4 quantization, written in the tile-major layout the GEMM reads.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Compares device tiles/scales with the double precision model whose
// pre-rotation inputs (normalized and awq-divided) are given per token.
void CompareToHost(const char* label, const std::vector<std::uint8_t>& tiles,
                   const std::vector<float>& scales,
                   const std::vector<double>& pre,
                   const std::vector<float>& signs, std::size_t n,
                   std::size_t padded, std::size_t K) {
  const std::size_t e_groups = K / 128;
  std::size_t compared = 0, off_by_one = 0, bigger = 0;
  double worst_scale = 0.0;
  std::vector<double> v(K);
  for (std::size_t t = 0; t < padded; ++t) {
    if (t < n) {
      for (std::size_t b = 0; b < K / 256; ++b) {
        double blk[256];
        for (int i = 0; i < 256; ++i)
          blk[i] = pre[t * K + b * 256 + i] * signs[i];
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
      const float got_scale = scales[e * padded + t];
      if (scale > 0.0)
        worst_scale =
            std::max(worst_scale, std::fabs(got_scale - scale) / scale);
      else
        CHECK(got_scale == 0.0f, "pad scale %zu %zu: %g", t, e, got_scale);
      for (int i = 0; i < 128; ++i) {
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
      "%s: compared %zu codes: %zu rounding-boundary flips, %zu real "
      "mismatches, worst scale error %.2e\n",
      label, compared, off_by_one, bigger, worst_scale);
  CHECK(bigger == 0, "%s: codes differ beyond rounding boundaries", label);
  CHECK(worst_scale < 1e-5, "%s: scale error %.3e", label, worst_scale);
  CHECK(off_by_one < compared / 10000, "%s: too many boundary flips", label);
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

  {
    std::vector<double> pre(kPadded * kK, 0.0);
    for (std::size_t t = 0; t < kTokens; ++t) {
      double ss = 0.0;
      for (std::size_t k = 0; k < kK; ++k)
        ss += static_cast<double>(x[t * kK + k]) * x[t * kK + k];
      const double inv = 1.0 / std::sqrt(ss / kK + 1e-6);
      for (std::size_t k = 0; k < kK; ++k)
        pre[t * kK + k] = x[t * kK + k] * inv * norm[k] / awq[k];
    }
    CompareToHost("norm-fed", tiles, scales, pre, signs, kTokens, kPadded, kK);
  }

  // Quantize-only producer: attention output and GDN output (K = 6144, the GDN
  // one with the value-head remap) and the SwiGLU product (K = 17408).
  struct Case {
    const char* label;
    gufo::hip::Iu4Role role;
    std::uint32_t layer;
    bool remap;
  };
  const Case cases[] = {
      {"attn-out", gufo::hip::Iu4Role::kAttnO, 3, false},
      {"gdn-out remap", gufo::hip::Iu4Role::kSsmOut, 0, true},
      {"swiglu", gufo::hip::Iu4Role::kFfnDown, 0, false},
  };
  for (const Case& c : cases) {
    const auto& p = weights.Find(c.layer, c.role);
    const std::size_t K = p.cols, ge = K / 128;
    std::vector<float> a(K), xs(kTokens * K);
    CHECK(
        hipMemcpy(a.data(), p.awq, K * 4, hipMemcpyDeviceToHost) == hipSuccess,
        "awq copy");
    for (auto& v : xs)
      v = static_cast<float>(std::sqrt(-2 * std::log(uniform() + 1e-12)) *
                             std::cos(6.283185307179586 * uniform()));
    for (std::size_t t = 0; t < kTokens; ++t)
      for (int ch = 0; ch < 8; ++ch)
        xs[t * K + (ch * 613) % K] *= 30.0f;
    float *dx, *dscale;
    std::uint8_t* dtile;
    const std::size_t tb = (kPadded / 16) * ge * 1024;
    hipMalloc(&dx, xs.size() * 4);
    hipMalloc(&dscale, ge * kPadded * 4);
    hipMalloc(&dtile, tb);
    hipMemcpy(dx, xs.data(), xs.size() * 4, hipMemcpyHostToDevice);
    hipMemset(dtile, 0xAB, tb);
    hipMemset(dscale, 0xFF, ge * kPadded * 4);
    gufo::hip::LaunchIu4Quantize(dx, p.awq, weights.signs(), c.remap, kTokens,
                                 kPadded, K, dtile, dscale, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "%s failed", c.label);
    std::vector<std::uint8_t> ht(tb);
    std::vector<float> hs(ge * kPadded);
    hipMemcpy(ht.data(), dtile, tb, hipMemcpyDeviceToHost);
    hipMemcpy(hs.data(), dscale, hs.size() * 4, hipMemcpyDeviceToHost);
    std::vector<double> pre(kPadded * K, 0.0);
    for (std::size_t t = 0; t < kTokens; ++t)
      for (std::size_t pos = 0; pos < K; ++pos) {
        std::size_t src = pos;
        if (c.remap) {
          const std::size_t h = pos / 128;
          src = ((h % 3) * 16 + h / 3) * 128 + pos % 128;
        }
        pre[t * K + pos] = xs[t * K + src] / a[pos];
      }
    CompareToHost(c.label, ht, hs, pre, signs, kTokens, kPadded, K);
    hipFree(dx);
    hipFree(dscale);
    hipFree(dtile);
  }
  // SwiGLU fused into the quantizer, with the leading columns from an FP16
  // buffer (the NPU case) and without.
  for (const std::size_t npu_cols : {std::size_t{0}, std::size_t{4096}}) {
    const auto& p = weights.Find(0, gufo::hip::Iu4Role::kFfnDown);
    const std::size_t K = p.cols, ge = K / 128;
    std::vector<float> a(K), gate(kTokens * K), up(kTokens * K);
    hipMemcpy(a.data(), p.awq, K * 4, hipMemcpyDeviceToHost);
    for (auto& v : gate)
      v = static_cast<float>((uniform() - 0.5) * 6.0);
    for (auto& v : up)
      v = static_cast<float>(uniform() * 2.0 - 1.0);
    std::vector<unsigned short> half_bits(kTokens * K, 0);
    std::vector<double> pre(kPadded * K, 0.0);
    const auto to_half = [](float f) {
      // round to nearest even through the device-independent bit trick
      std::uint32_t x;
      std::memcpy(&x, &f, 4);
      const std::uint32_t sign = (x >> 16) & 0x8000u;
      int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
      std::uint32_t man = x & 0x7fffffu;
      if (exp <= 0)
        return static_cast<unsigned short>(sign);
      if (exp >= 31)
        return static_cast<unsigned short>(sign | 0x7bffu);
      std::uint32_t h = (static_cast<std::uint32_t>(exp) << 10) | (man >> 13);
      if ((man & 0x1fffu) > 0x1000u || ((man & 0x1fffu) == 0x1000u && (h & 1)))
        ++h;
      return static_cast<unsigned short>(sign | h);
    };
    const auto from_half = [](unsigned short h) {
      const int exp = (h >> 10) & 31, man = h & 1023;
      const float v =
          exp == 0 ? std::ldexp(static_cast<float>(man), -24)
                   : std::ldexp(static_cast<float>(man | 1024), exp - 25);
      return (h & 0x8000) ? -v : v;
    };
    for (std::size_t t = 0; t < kTokens; ++t)
      for (std::size_t k = 0; k < K; ++k) {
        const double g = gate[t * K + k], u = up[t * K + k];
        const double act = g / (1.0 + std::exp(-g)) * u;
        double val = act;
        if (k < npu_cols) {
          half_bits[t * K + k] = to_half(static_cast<float>(act));
          val = from_half(half_bits[t * K + k]);
        }
        pre[t * K + k] = val / a[k];
      }
    float *dg, *du, *dscale;
    unsigned short* dh;
    std::uint8_t* dtile;
    const std::size_t tb = (kPadded / 16) * ge * 1024;
    hipMalloc(&dg, gate.size() * 4);
    hipMalloc(&du, up.size() * 4);
    hipMalloc(&dh, half_bits.size() * 2);
    hipMalloc(&dscale, ge * kPadded * 4);
    hipMalloc(&dtile, tb);
    hipMemcpy(dg, gate.data(), gate.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(du, up.data(), up.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dh, half_bits.data(), half_bits.size() * 2,
              hipMemcpyHostToDevice);
    hipMemset(dtile, 0xAB, tb);
    hipMemset(dscale, 0xFF, ge * kPadded * 4);
    gufo::hip::LaunchIu4SwiGluQuantize(dg, du, dh, npu_cols, p.awq,
                                       weights.signs(), kTokens, kPadded, K,
                                       dtile, dscale, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "swiglu producer failed");
    std::vector<std::uint8_t> ht(tb);
    std::vector<float> hs(ge * kPadded);
    hipMemcpy(ht.data(), dtile, tb, hipMemcpyDeviceToHost);
    hipMemcpy(hs.data(), dscale, hs.size() * 4, hipMemcpyDeviceToHost);
    CompareToHost(npu_cols ? "swiglu fused (npu cols)" : "swiglu fused", ht, hs,
                  pre, signs, kTokens, kPadded, K);
    hipFree(dg);
    hipFree(du);
    hipFree(dh);
    hipFree(dscale);
    hipFree(dtile);
  }

  // Gated DeltaNet epilogue fused into the out projection quantizer.
  {
    const auto& p = weights.Find(0, gufo::hip::Iu4Role::kSsmOut);
    const std::size_t K = p.cols, ge = K / 128;
    std::vector<float> a(K), raw(kTokens * K), z(kTokens * K), nw(128);
    hipMemcpy(a.data(), p.awq, K * 4, hipMemcpyDeviceToHost);
    for (auto& v : raw)
      v = static_cast<float>(std::sqrt(-2 * std::log(uniform() + 1e-12)) *
                             std::cos(6.283185307179586 * uniform()));
    for (auto& v : z)
      v = static_cast<float>((uniform() - 0.5) * 6.0);
    for (auto& w : nw)
      w = static_cast<float>(0.5 + uniform());
    std::vector<double> pre(kPadded * K, 0.0);
    for (std::size_t t = 0; t < kTokens; ++t)
      for (std::size_t head = 0; head < K / 128; ++head) {
        double ss = 0.0;
        for (int i = 0; i < 128; ++i)
          ss += static_cast<double>(raw[t * K + head * 128 + i]) *
                raw[t * K + head * 128 + i];
        const double inv = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
        // artifact head h reads engine head (h % 3) * 16 + h / 3
        const std::size_t eng = (head % 3) * 16 + head / 3;
        for (int i = 0; i < 128; ++i) {
          const double zz = z[t * K + eng * 128 + i];
          const double xx = raw[t * K + eng * 128 + i];
          pre[t * K + head * 128 + i] = xx * inv * nw[i] *
                                        (zz / (1.0 + std::exp(-zz))) /
                                        a[head * 128 + i];
        }
      }
    // The kernel normalizes each engine head, so recompute with the engine
    // head's own statistics: inv must come from the head that is read.
    for (std::size_t t = 0; t < kTokens; ++t)
      for (std::size_t head = 0; head < K / 128; ++head) {
        const std::size_t eng = (head % 3) * 16 + head / 3;
        double ss = 0.0;
        for (int i = 0; i < 128; ++i)
          ss += static_cast<double>(raw[t * K + eng * 128 + i]) *
                raw[t * K + eng * 128 + i];
        const double inv = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
        for (int i = 0; i < 128; ++i) {
          const double zz = z[t * K + eng * 128 + i];
          pre[t * K + head * 128 + i] = raw[t * K + eng * 128 + i] * inv *
                                        nw[i] * (zz / (1.0 + std::exp(-zz))) /
                                        a[head * 128 + i];
        }
      }
    float *dr, *dz, *dn, *dscale;
    std::uint8_t* dtile;
    const std::size_t tb = (kPadded / 16) * ge * 1024;
    hipMalloc(&dr, raw.size() * 4);
    hipMalloc(&dz, z.size() * 4);
    hipMalloc(&dn, 128 * 4);
    hipMalloc(&dscale, ge * kPadded * 4);
    hipMalloc(&dtile, tb);
    hipMemcpy(dr, raw.data(), raw.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dz, z.data(), z.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dn, nw.data(), 128 * 4, hipMemcpyHostToDevice);
    hipMemset(dtile, 0xAB, tb);
    hipMemset(dscale, 0xFF, ge * kPadded * 4);
    gufo::hip::LaunchIu4PostNormGateQuantize(dr, dn, dz, p.awq, weights.signs(),
                                             kTokens, kPadded, K, dtile, dscale,
                                             nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "post-norm producer failed");
    std::vector<std::uint8_t> ht(tb);
    std::vector<float> hs(ge * kPadded);
    hipMemcpy(ht.data(), dtile, tb, hipMemcpyDeviceToHost);
    hipMemcpy(hs.data(), dscale, hs.size() * 4, hipMemcpyDeviceToHost);
    CompareToHost("post-norm gate fused", ht, hs, pre, signs, kTokens, kPadded,
                  K);
  }
  if (failures != 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
