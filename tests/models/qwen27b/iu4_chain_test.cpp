// End to end check of the int4 prefill chain on one real projection: norm-fed
// producer, int4 GEMM and the repacked weights. The output must match the
// unquantized-activation product with the artifact's own weights up to the
// int4 activation error (about 0.115 relative RMS); a layout or scale mismatch
// anywhere in the chain shows up as an error near or above 1.
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31,
                      man = h & 1023;
  float v = exp == 0 ? std::ldexp(static_cast<float>(man), -24)
                     : std::ldexp(static_cast<float>(man | 1024),
                                  static_cast<int>(exp) - 25);
  return sign ? -v : v;
}

std::uint64_t FindOffset(const std::uint8_t* base, const std::string& wanted) {
  std::uint64_t meta_off, data_off;
  std::memcpy(&meta_off, base + 16, 8);
  std::memcpy(&data_off, base + 24, 8);
  std::size_t pos = meta_off;
  int depth = 0;
  bool in_str = false;
  for (; pos < data_off; ++pos) {
    const char c = static_cast<char>(base[pos]);
    if (in_str) {
      if (c == '\\')
        ++pos;
      else if (c == '"')
        in_str = false;
    } else if (c == '"') {
      in_str = true;
    } else if (c == '{') {
      ++depth;
    } else if (c == '}' && --depth == 0) {
      ++pos;
      break;
    }
  }
  std::uint32_t n;
  std::memcpy(&n, base + pos, 4);
  pos += 4;
  std::uint64_t offset = data_off;
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint16_t nl;
    std::memcpy(&nl, base + pos, 2);
    pos += 2;
    const std::string name(reinterpret_cast<const char*>(base + pos), nl);
    pos += nl + 1;
    const int nd = base[pos++];
    pos += 4 * nd + 4;
    std::uint64_t len;
    std::memcpy(&len, base + pos, 8);
    pos += 8;
    if (name == wanted)
      return offset;
    offset += len;
  }
  return 0;
}

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
  const int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return kSkip;
  struct stat st{};
  fstat(fd, &st);
  void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  const auto* base = static_cast<const std::uint8_t*>(map);

  gufo::hip::Iu4Weights weights;
  std::string error;
  if (!weights.Load(path, 4, &error)) {
    std::printf("load failed: %s\n", error.c_str());
    return 1;
  }
  const auto& proj = weights.Find(3, gufo::hip::Iu4Role::kFfnGate);
  const std::size_t M = proj.rows, K = proj.cols, n = 300, padded = 384;
  const std::size_t E = K / 128;
  std::vector<float> awq(K), signs(512);
  hipMemcpy(awq.data(), proj.awq, K * 4, hipMemcpyDeviceToHost);
  hipMemcpy(signs.data(), weights.signs(), 512 * 4, hipMemcpyDeviceToHost);

  std::uint64_t state = 0x1234567887654321ull;
  const auto uniform = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
  };
  std::vector<float> x(n * K), norm(K);
  for (auto& v : x)
    v = static_cast<float>(std::sqrt(-2 * std::log(uniform() + 1e-12)) *
                           std::cos(6.283185307179586 * uniform()));
  for (std::size_t t = 0; t < n; ++t)
    for (int c = 0; c < 8; ++c)
      x[t * K + (c * 613) % K] *= 30.0f;
  for (auto& w : norm)
    w = static_cast<float>(0.5 + uniform());

  float *d_x, *d_norm, *d_scales, *d_out;
  std::uint8_t* d_tiles;
  hipMalloc(&d_x, x.size() * 4);
  hipMalloc(&d_norm, K * 4);
  hipMalloc(&d_scales, E * padded * 4);
  hipMalloc(&d_tiles, (padded / 16) * E * 1024);
  hipMalloc(&d_out, padded * M * 4);
  hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(d_norm, norm.data(), K * 4, hipMemcpyHostToDevice);
  gufo::hip::LaunchIu4NormQuantize(d_x, d_norm, proj.awq, weights.signs(),
                                   1e-6f, n, padded, K, d_tiles, d_scales,
                                   nullptr);
  gufo::hip::LaunchIu4Gemm(proj.tiles, proj.scales, d_tiles, d_scales, d_out,
                           false, M, n, padded, K, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess, "chain failed");
  std::vector<float> got(n * M);
  hipMemcpy(got.data(), d_out, got.size() * 4, hipMemcpyDeviceToHost);

  // Host reference: unquantized rotated activations times the dequantized
  // artifact weights (scale * nibble + zero point), in double precision.
  const std::uint64_t woff =
      FindOffset(base, "model.language_model.layers.3.mlp.gate_proj.weight");
  std::vector<std::vector<double>> rotated(n);
  const std::size_t kTokensChecked = 12;
  for (std::size_t t = 0; t < kTokensChecked; ++t) {
    double ss = 0.0;
    for (std::size_t k = 0; k < K; ++k)
      ss += static_cast<double>(x[t * K + k]) * x[t * K + k];
    const double inv = 1.0 / std::sqrt(ss / K + 1e-6);
    rotated[t].resize(K);
    for (std::size_t b = 0; b < K / 256; ++b) {
      double blk[256];
      for (int i = 0; i < 256; ++i) {
        const std::size_t k = b * 256 + i;
        blk[i] = x[t * K + k] * inv * norm[k] / awq[k] * signs[i];
      }
      Hadamard256(blk);
      for (int i = 0; i < 256; ++i)
        rotated[t][b * 256 + i] = blk[i] * 0.0625 * signs[256 + i];
    }
  }
  double err2 = 0.0, ref2 = 0.0;
  std::size_t pairs = 0;
  for (std::size_t t = 0; t < kTokensChecked; ++t)
    for (int s = 0; s < 250; ++s) {
      const std::size_t f = static_cast<std::size_t>(uniform() * M) % M;
      double ref = 0.0;
      for (std::size_t e = 0; e < E; ++e) {
        const std::uint8_t* block = base + woff + (f * (K / 256) + e / 2) * 136;
        const double sc = HalfToFloat(
            *reinterpret_cast<const std::uint16_t*>(block + 4 * (e & 1)));
        const double zp = HalfToFloat(
            *reinterpret_cast<const std::uint16_t*>(block + 4 * (e & 1) + 2));
        for (int i = 0; i < 128; ++i) {
          const std::uint8_t byte = block[8 + 64 * (e & 1) + i / 2];
          const int nib = (byte >> (4 * (i & 1))) & 15;
          ref += (sc * nib + zp) * rotated[t][e * 128 + i];
        }
      }
      const double d = got[t * M + f] - ref;
      err2 += d * d;
      ref2 += ref * ref;
      ++pairs;
    }
  const double rel = std::sqrt(err2 / ref2);
  std::printf(
      "chain relative RMS error vs fp32 activations: %.4f over %zu "
      "outputs\n",
      rel, pairs);
  CHECK(rel > 0.06 && rel < 0.16, "relative error %.4f outside [0.06, 0.16]",
        rel);
  if (failures != 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
