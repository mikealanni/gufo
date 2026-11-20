// Checks the device-resident rotated 4-bit projections against the artifact:
// payload bytes and group scales must be preserved bit for bit (only the byte
// order changes) and the symmetric signed reading of every sampled weight must
// equal the artifact's scale-and-zero-point reading.
#include "src/models/qwen/hip/iu4_weights.hpp"

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/ops/iu4.hpp"

namespace {

constexpr std::uint32_t kLayers =
    4;  // three GDN layers and one attention layer
constexpr int kSkip = 77;

struct Entry {
  std::uint64_t offset;
  std::uint64_t length;
  std::uint32_t rows;
  std::uint32_t cols;
};

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31,
                      man = h & 1023;
  float v;
  if (exp == 0)
    v = std::ldexp(static_cast<float>(man), -24);
  else if (exp == 31)
    v = man == 0 ? INFINITY : NAN;
  else
    v = std::ldexp(static_cast<float>(man | 1024), static_cast<int>(exp) - 25);
  return sign ? -v : v;
}

float Half(const std::uint8_t* p) {
  std::uint16_t h;
  std::memcpy(&h, p, 2);
  return HalfToFloat(h);
}

// Independent reader of the container index (same layout the loader parses).
bool FindEntry(const std::uint8_t* base, std::size_t size,
               const std::string& wanted, Entry* out) {
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
    std::uint32_t dims[4] = {};
    for (int d = 0; d < nd; ++d) {
      std::memcpy(&dims[d], base + pos, 4);
      pos += 4;
    }
    pos += 4;
    std::uint64_t len;
    std::memcpy(&len, base + pos, 8);
    pos += 8;
    if (name == wanted) {
      *out = {offset, len, dims[0], nd > 1 ? dims[1] : 0};
      return true;
    }
    offset += len;
  }
  (void)size;
  return false;
}

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

void CheckProjection(const std::uint8_t* base, std::size_t size,
                     const gufo::hip::Iu4Projection& p,
                     const std::string& name) {
  Entry e{};
  CHECK(FindEntry(base, size, name, &e), "%s not in artifact", name.c_str());
  CHECK(p.valid() && p.rows == e.rows && p.cols == e.cols, "%s shape",
        name.c_str());
  if (failures != 0)
    return;
  const std::size_t rows = p.rows, cols = p.cols, groups = cols / 128;
  std::vector<std::uint8_t> tiles(rows * cols / 2);
  std::vector<std::uint16_t> scales(rows * groups);
  CHECK(hipMemcpy(tiles.data(), p.tiles, tiles.size(), hipMemcpyDeviceToHost) ==
            hipSuccess,
        "copy tiles");
  CHECK(hipMemcpy(scales.data(), p.scales, scales.size() * 2,
                  hipMemcpyDeviceToHost) == hipSuccess,
        "copy scales");
  std::uint64_t state = 0x9E3779B97F4A7C15ull ^ rows;
  const auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  std::size_t checked = 0;
  for (int s = 0; s < 4000; ++s) {
    const std::size_t row = next() % rows, g = next() % groups;
    const std::size_t byte = next() % 64;  // payload byte inside the 128 group
    const std::uint8_t* block =
        base + e.offset + (row * (cols / 256) + g / 2) * 136;
    const std::uint16_t sc_bits =
        *reinterpret_cast<const std::uint16_t*>(block + 4 * (g & 1));
    CHECK(scales[g * rows + row] == sc_bits, "scale %zu %zu", row, g);
    const std::uint8_t src = block[8 + 64 * (g & 1) + byte];
    const std::size_t q = byte / 16, within = byte % 16;
    const std::size_t dst = (((g * (rows / 16) + row / 16) * 4 + q) * 256) +
                            (row % 16) * 16 + within;
    const std::uint8_t got = tiles[dst];
    CHECK(got == (src ^ 0x88), "payload %zu %zu %zu: %02x vs %02x", row, g,
          byte, got, src);
    const float scale = Half(block + 4 * (g & 1));
    const float zero = Half(block + 4 * (g & 1) + 2);
    for (int nib = 0; nib < 2; ++nib) {
      const int code = nib ? (src >> 4) : (src & 15);
      const int signed_code = ((got >> (4 * nib)) & 15) ^ 8;
      const int sym = (signed_code & 8) ? signed_code - 16 : signed_code;
      (void)sym;
      const float artifact = scale * code + zero;
      const int sgn = ((got >> (4 * nib)) & 15);
      const int signed_value = sgn >= 8 ? sgn - 16 : sgn;
      CHECK(std::fabs(artifact - scale * signed_value) <=
                1e-3f * std::fabs(8.0f * scale) + 1e-12f,
            "value %zu %zu: %g vs %g", row, g, artifact, scale * signed_value);
    }
    ++checked;
  }
  std::printf("%s: %zu samples ok (%zu x %zu)\n", name.c_str(), checked, rows,
              cols);
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_IU4_ARTIFACT");
  if (path == nullptr) {
    std::printf("GUFO_IU4_ARTIFACT not set\n");
    return kSkip;
  }
  const int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    std::printf("cannot open %s\n", path);
    return kSkip;
  }
  struct stat st{};
  fstat(fd, &st);
  void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  const auto* base = static_cast<const std::uint8_t*>(map);

  gufo::hip::Iu4Weights weights;
  std::string error;
  const char* all = std::getenv("GUFO_IU4_ALL_LAYERS");
  const auto start = std::chrono::steady_clock::now();
  if (!weights.Load(path, all != nullptr ? 64 : kLayers, &error)) {
    std::printf("load failed: %s\n", error.c_str());
    return 1;
  }
  std::printf(
      "loaded %u layers, %.2f GiB on device in %.1f s\n", weights.layers(),
      weights.device_bytes() / 1073741824.0,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  using gufo::hip::Iu4Role;
  const std::string pre = "model.language_model.layers.";
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    const std::string lp = pre + std::to_string(l) + ".";
    CheckProjection(base, st.st_size, weights.Find(l, Iu4Role::kFfnGate),
                    lp + "mlp.gate_proj.weight");
    CheckProjection(base, st.st_size, weights.Find(l, Iu4Role::kFfnDown),
                    lp + "mlp.down_proj.weight");
    if (l == 3) {
      CheckProjection(base, st.st_size, weights.Find(l, Iu4Role::kAttnQ),
                      lp + "self_attn.q_proj.weight");
      CheckProjection(base, st.st_size, weights.Find(l, Iu4Role::kAttnO),
                      lp + "self_attn.o_proj.weight");
      CHECK(!weights.Find(l, Iu4Role::kSsmQkv).valid(), "attn layer has ssm");
    } else {
      CheckProjection(base, st.st_size, weights.Find(l, Iu4Role::kSsmQkv),
                      lp + "linear_attn.in_proj_qkv.weight");
      CheckProjection(base, st.st_size, weights.Find(l, Iu4Role::kSsmOut),
                      lp + "linear_attn.out_proj.weight");
      CHECK(!weights.Find(l, Iu4Role::kAttnQ).valid(), "ssm layer has attn");
    }
  }
  CheckProjection(base, st.st_size, weights.LmHead(), "lm_head.weight");
  CHECK(weights.Find(0, Iu4Role::kFfnGate).awq != nullptr, "awq missing");
  if (failures != 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
