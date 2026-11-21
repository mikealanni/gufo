// Checks the single-token int4 GEMV against a double precision host model
// computed from the artifact's own blocks, and prints bandwidth.
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
  };
  const Case cases[] = {
      {"model.language_model.layers.3.mlp.gate_proj.weight", 3,
       Iu4Role::kFfnGate},
      {"model.language_model.layers.0.mlp.down_proj.weight", 0,
       Iu4Role::kFfnDown},
      {"model.language_model.layers.3.self_attn.q_proj.weight", 3,
       Iu4Role::kAttnQ},
      {"model.language_model.layers.0.linear_attn.out_proj.weight", 0,
       Iu4Role::kSsmOut},
  };
  std::uint64_t state = 0x9E3779B97F4A7C15ull;
  const auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (const Case& c : cases) {
    const auto& proj = weights.Find(c.layer, c.role);
    CHECK(proj.valid(), "%s missing", c.name);
    if (!proj.valid())
      continue;
    const std::size_t M = proj.rows, K = proj.cols, E = K / 128;
    const std::uint64_t woff = FindOffset(base, c.name);
    for (const std::size_t T : {std::size_t{1}, std::size_t{3}, std::size_t{8},
                                std::size_t{16}, std::size_t{21}}) {
      std::vector<float> x(T * K);
      for (auto& v : x)
        v = (static_cast<int>(next() % 2001) - 1000) * 1e-3f;
      float *d_x, *d_out;
      void* d_x16;
      (void)hipMalloc(&d_x16, 16 * K * 2);
      (void)hipMalloc(&d_x, T * K * 4);
      (void)hipMalloc(&d_out, T * M * 4);
      (void)hipMemcpy(d_x, x.data(), T * K * 4, hipMemcpyHostToDevice);
      std::vector<float> init(T * M, 1.0f);
      (void)hipMemcpy(d_out, init.data(), init.size() * 4,
                      hipMemcpyHostToDevice);
      gufo::hip::LaunchIu4Gemv(proj.tiles, proj.scales, d_x, d_out, true,
                               proj.padded_rows, M, K, nullptr, T, d_x16);
      CHECK(hipDeviceSynchronize() == hipSuccess, "%s kernel failed", c.name);
      std::vector<float> got(T * M);
      (void)hipMemcpy(got.data(), d_out, got.size() * 4,
                      hipMemcpyDeviceToHost);
      double worst = 0.0, rms = 0.0;
      int checked = 0;
      for (int s = 0; s < 600; ++s) {
        const std::size_t f = next() % M, t = (s < 100) ? T - 1 : next() % T;
        double ref = 1.0;
        for (std::size_t e = 0; e < E; ++e) {
          const std::uint8_t* block =
              base + woff + (f * (K / 256) + e / 2) * 136;
          const double sw = HalfToFloat(
              *reinterpret_cast<const std::uint16_t*>(block + 4 * (e & 1)));
          double dot = 0;
          for (int i = 0; i < 128; ++i) {
            const std::uint8_t byte = block[8 + 64 * (e & 1) + i / 2];
            const int nib = (byte >> (4 * (i & 1))) & 15;
            dot += (nib - 8) * static_cast<double>(x[t * K + e * 128 + i]);
          }
          ref += sw * dot;
        }
        worst = std::max(worst, std::fabs(got[t * M + f] - ref));
        rms += ref * ref;
        ++checked;
      }
      rms = std::sqrt(rms / checked);
      CHECK(worst < (T > 4 ? 1.5e-3 : 1e-4) * rms + 1e-4, "error too large");
      hipEvent_t a, b;
      (void)hipEventCreate(&a);
      (void)hipEventCreate(&b);
      for (int w = 0; w < 5; ++w)
        gufo::hip::LaunchIu4Gemv(proj.tiles, proj.scales, d_x, d_out, false,
                                 proj.padded_rows, M, K, nullptr, T, d_x16);
      (void)hipEventRecord(a);
      for (int r = 0; r < 100; ++r)
        gufo::hip::LaunchIu4Gemv(proj.tiles, proj.scales, d_x, d_out, false,
                                 proj.padded_rows, M, K, nullptr, T, d_x16);
      (void)hipEventRecord(b);
      (void)hipEventSynchronize(b);
      float ms;
      (void)hipEventElapsedTime(&ms, a, b);
      ms /= 100;
      const double bytes = static_cast<double>(M) * K * 0.5 + M * E * 2.0;
      std::printf("%-12.12s %s T=%2zu: err %.2e (rms %.2f), %.3f ms, %.0f GB/s\n",
                  c.name + 24, "", T, worst, rms, ms,
                  bytes / (ms * 1e-3) / 1e9);
      (void)hipFree(d_x);
      (void)hipFree(d_x16);
      (void)hipFree(d_out);
    }
  }
  if (failures != 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
