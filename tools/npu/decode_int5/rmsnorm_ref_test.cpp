#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "tools/npu/decode_int5/rmsnorm_ref.hpp"

static inline uint16_t bf16r(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  u &= 0xFFFF0000u;
  return static_cast<uint16_t>(u >> 16);
}

int main() {
  const int rows = 64;
  const int hidden = 2048;
  const float eps = 1e-6f;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

  std::vector<float> x(std::size_t{rows} * hidden);
  std::vector<float> w(hidden);
  for (int i = 0; i < rows * hidden; ++i) x[i] = uni(rng);
  for (int i = 0; i < hidden; ++i) w[i] = uni(rng);

  std::vector<float> out(rows * hidden);
  gufo::npu::rmsnorm::rmsnorm_rows(x.data(), w.data(), out.data(), rows, hidden, eps);

  // bf16-rounded variant
  std::vector<float> out_bf16(rows * hidden);
  for (int i = 0; i < rows * hidden; ++i) {
    uint16_t b = bf16r(out[i]);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    std::memcpy(&out_bf16[i], &u, 4);
  }

  double max_err = 0;
  for (int i = 0; i < rows * hidden; ++i) {
    double d = std::fabs(double(out[i]) - double(out_bf16[i]));
    max_err = d > max_err ? d : max_err;
  }
  std::printf("rmsnorm max abs error (float vs bf16-rounded): %.6e\n", max_err);

  // Dump input
  {
    FILE* f = std::fopen("rmsnorm_in.f32", "wb");
    std::fwrite(x.data(), 4, x.size(), f);
    std::fclose(f);
  }

  // Encode A wire and dump
  auto a_wire = gufo::npu::rmsnorm::encode_a_wire(x.data(), rows, hidden);
  {
    FILE* f = std::fopen("rmsnorm_expected_a_wire.u8", "wb");
    std::fwrite(a_wire.data(), 1, a_wire.size(), f);
    std::fclose(f);
  }

  std::printf("A wire size: %zu bytes (%d rows x %d hidden)\n",
              a_wire.size(), rows, hidden);
  std::printf("DONE\n");
  return 0;
}
