// Checks the int4 GEMM against a double precision host model computed from the
// artifact's own blocks and the producer's output, covers the token tail
// (n not a multiple of 128), the accumulate variant, and prints throughput.
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
  float v;
  if (exp == 0)
    v = std::ldexp(static_cast<float>(man), -24);
  else
    v = std::ldexp(static_cast<float>(man | 1024), static_cast<int>(exp) - 25);
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
  using gufo::hip::Iu4Role;
  struct Case {
    const char* name;
    std::uint32_t layer;
    Iu4Role role;
    std::size_t n;
  };
  const Case cases[] = {
      {"model.language_model.layers.3.mlp.gate_proj.weight", 3,
       Iu4Role::kFfnGate, 300},
      {"model.language_model.layers.0.mlp.down_proj.weight", 0,
       Iu4Role::kFfnDown, 129},
      {"model.language_model.layers.3.self_attn.q_proj.weight", 3,
       Iu4Role::kAttnQ, 777},
      {"model.language_model.layers.0.linear_attn.out_proj.weight", 0,
       Iu4Role::kSsmOut, 128},
  };
  for (const Case& c : cases) {
    const auto& proj = weights.Find(c.layer, c.role);
    CHECK(proj.valid(), "%s missing", c.name);
    if (!proj.valid())
      continue;
    CHECK(gufo::hip::Iu4GemmSupported(proj.rows, proj.cols), "%s unsupported",
          c.name);
    const std::size_t M = proj.rows, K = proj.cols, n = c.n;
    const std::size_t padded = (n + 127) / 128 * 128, E = K / 128;
    // Synthetic activations, already as producer-style codes and scales: the
    // GEMM is checked on its own inputs so producer and GEMM errors do not mix.
    std::uint64_t state = 0x2545F4914F6CDD1Dull ^ (M * 31 + n);
    const auto next = [&]() {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      return state;
    };
    std::vector<std::uint8_t> tiles((padded / 16) * E * 1024, 0);
    std::vector<float> scales(E * padded, 0.0f);
    std::vector<int> codes(n * K);
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t e = 0; e < E; ++e) {
        scales[e * padded + t] = 0.01f + (next() % 1000) * 1e-5f;
        for (int i = 0; i < 128; ++i) {
          const int code = static_cast<int>(next() % 16) - 8;
          codes[t * K + e * 128 + i] = code;
          const int byte = i / 2, chunk = byte / 8, within = byte % 8;
          const std::size_t off = ((t / 16) * E + e) * 1024 +
                                  ((chunk & 1) * 16 + (t % 16)) * 32 +
                                  (chunk >> 1) * 8 + within;
          tiles[off] |= static_cast<std::uint8_t>((code & 15) << (4 * (i & 1)));
        }
      }
    std::uint8_t* d_tiles;
    float *d_scales, *d_out;
    hipMalloc(&d_tiles, tiles.size());
    hipMalloc(&d_scales, scales.size() * 4);
    hipMalloc(&d_out, padded * M * 4);
    hipMemcpy(d_tiles, tiles.data(), tiles.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_scales, scales.data(), scales.size() * 4,
              hipMemcpyHostToDevice);
    const float sentinel = 12345.0f;
    std::vector<float> init(padded * M, sentinel);
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t f = 0; f < M; ++f)
        init[t * M + f] = 1.0f;
    hipMemcpy(d_out, init.data(), init.size() * 4, hipMemcpyHostToDevice);
    gufo::hip::LaunchIu4Gemm(proj.tiles, proj.scales, d_tiles, d_scales, d_out,
                             true, M, n, padded, K, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "%s kernel failed", c.name);
    std::vector<float> got(padded * M);
    hipMemcpy(got.data(), d_out, got.size() * 4, hipMemcpyDeviceToHost);

    const std::uint64_t woff = FindOffset(base, c.name);
    std::size_t checked = 0, tail_bad = 0;
    double worst = 0.0, rms = 0.0;
    std::vector<double> refs;
    for (int s = 0; s < 3000; ++s) {
      const std::size_t t = (s < 600 && n > 1)
                                ? n - 1 - (s % std::min<size_t>(n, 40))
                                : next() % n;
      const std::size_t f = next() % M;
      double ref = 1.0;  // accumulate variant starts from 1
      for (std::size_t e = 0; e < E; ++e) {
        const std::uint8_t* block = base + woff + (f * (K / 256) + e / 2) * 136;
        const double sw = HalfToFloat(
            *reinterpret_cast<const std::uint16_t*>(block + 4 * (e & 1)));
        long dot = 0;
        for (int i = 0; i < 128; ++i) {
          const std::uint8_t byte = block[8 + 64 * (e & 1) + i / 2];
          const int nib = (byte >> (4 * (i & 1))) & 15;
          dot += static_cast<long>(nib - 8) * codes[t * K + e * 128 + i];
        }
        ref += scales[e * padded + t] * sw * static_cast<double>(dot);
      }
      const double err = std::fabs(got[t * M + f] - ref);
      worst = std::max(worst, err);
      rms += ref * ref;
      ++checked;
    }
    rms = std::sqrt(rms / checked);
    for (std::size_t t = n; t < padded; ++t)
      for (std::size_t f = 0; f < M; f += 997)
        if (got[t * M + f] != sentinel)
          ++tail_bad;
    std::printf(
        "%s n=%zu: worst abs err %.3e (rms output %.3f), tail "
        "clobbered %zu\n",
        c.name, n, worst, rms, tail_bad);
    CHECK(worst < 2e-5 * rms + 1e-5, "error too large");
    CHECK(tail_bad == 0, "wrote past n");

    if (c.n == 300) {
      // Throughput at 2048 tokens (not gating).
      const std::size_t N2 = 2048, E2 = K / 128;
      std::uint8_t* d_t2;
      float *d_s2, *d_o2;
      hipMalloc(&d_t2, (N2 / 16) * E2 * 1024);
      hipMemset(d_t2, 0x35, (N2 / 16) * E2 * 1024);
      hipMalloc(&d_s2, E2 * N2 * 4);
      std::vector<float> ones(E2 * N2, 0.01f);
      hipMemcpy(d_s2, ones.data(), ones.size() * 4, hipMemcpyHostToDevice);
      hipMalloc(&d_o2, N2 * M * 4);
      hipEvent_t a, b;
      hipEventCreate(&a);
      hipEventCreate(&b);
      for (int w = 0; w < 5; ++w)
        gufo::hip::LaunchIu4Gemm(proj.tiles, proj.scales, d_t2, d_s2, d_o2,
                                 false, M, N2, N2, K, nullptr);
      hipEventRecord(a);
      for (int r = 0; r < 100; ++r)
        gufo::hip::LaunchIu4Gemm(proj.tiles, proj.scales, d_t2, d_s2, d_o2,
                                 false, M, N2, N2, K, nullptr);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms;
      hipEventElapsedTime(&ms, a, b);
      ms /= 100;
      std::printf("  throughput at n=2048: %.3f ms, %.1f TOPS\n", ms,
                  2.0 * M * N2 * K / (ms * 1e-3) / 1e12);
      hipFree(d_t2);
      hipFree(d_s2);
      hipFree(d_o2);
    }
    hipFree(d_tiles);
    hipFree(d_scales);
    hipFree(d_out);
  }
  if (failures != 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
