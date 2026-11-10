#ifndef GUFO_NPU_RUNTIME_HPP_
#define GUFO_NPU_RUNTIME_HPP_

#include <immintrin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gufo::npu::runtime {

constexpr std::uint32_t kMaxRows = 16;
constexpr std::uint32_t kTileK = 64;
constexpr std::uint32_t kStrip = 128;
constexpr std::uint32_t kQuarterBytes = 1536;
constexpr std::uint32_t kWordBytes = 1408;

inline float RoundHalf(float v) {
  return _cvtsh_ss(_cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT));
}

inline void EncGroup(const float* v, std::uint8_t* out) {
  const __m256 f = _mm256_loadu_ps(v);
  const __m256i bits = _mm256_castps_si256(f);
  const __m256i ex = _mm256_and_si256(_mm256_srli_epi32(bits, 23),
                                      _mm256_set1_epi32(0xFF));
  __m128i m = _mm_max_epu32(_mm256_castsi256_si128(ex),
                            _mm256_extracti128_si256(ex, 1));
  m = _mm_max_epu32(m, _mm_shuffle_epi32(m, 0x4E));
  m = _mm_max_epu32(m, _mm_shuffle_epi32(m, 0xB1));
  const unsigned max_exp = static_cast<unsigned>(_mm_cvtsi128_si32(m));
  const std::uint32_t ib = max_exp >= 7 ? (260u - max_exp) << 23 : 0u;
  float inv;
  std::memcpy(&inv, &ib, 4);
  const __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(f, _mm256_set1_ps(inv)));
  const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(qi),
                                      _mm256_extracti128_si256(qi, 1));
  const __m128i p8 = _mm_packs_epi16(p16, p16);
  out[0] = static_cast<std::uint8_t>(max_exp);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(out + 1), p8);
}

inline void DecGroup(const std::uint8_t* p, float* v) {
  const int e = p[0];
  const std::uint32_t mb = e > 6 ? static_cast<std::uint32_t>(e - 6) << 23 : 0u;
  float mult;
  std::memcpy(&mult, &mb, 4);
  const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 1));
  const __m256i i32 = _mm256_cvtepi8_epi32(b8);
  _mm256_storeu_ps(v, _mm256_mul_ps(_mm256_cvtepi32_ps(i32),
                                    _mm256_set1_ps(mult)));
}

inline std::uint16_t Bf16Bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}
inline float Bf16Float(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Packer: int5 (3 values per 16-bit word, 11 words + 128B bf16 scales = 1536B
// per core per k-tile of 64), laid out for the split-B 16-row decode GEMM.
void PackMatInt5(const float* W, std::uint32_t N, std::uint32_t K,
                 std::uint8_t* dst);

// Instruction-stream program (one runtime sequence) with its A/C staging BOs.
struct Program {
  std::uint32_t K{}, N{};
  xrt::bo insts, a, c;
  std::size_t words{};
  std::uint8_t* ah{};
  std::uint8_t* ch{};
};

// A packed weight buffer attached as a runnable kernel argument set.
struct WeightBuffer {
  xrt::run run;
  xrt::bo bo;
  std::uint8_t* host{};
  std::size_t bytes{};
};

class NpuRuntime {
 public:
  NpuRuntime() = default;
  /// Load device + xclbin + kernel. `xclbin_path` selects the built program.
  bool Init(const std::string& xclbin_path, std::string* error);

  xrt::device& device() { return dev_; }
  xrt::kernel& kernel() { return kern_; }

  /// Load an instruction stream file and create A/C BOs sized for (K,N).
  /// `c_override` allows custom C size (e.g. fused FFN combined buffer).
  Program MakeProgram(std::uint32_t K, std::uint32_t N,
                      const std::string& insts_path, std::size_t c_override,
                      std::string* error);

  /// Allocate and pack a weight matrix (N rows x K cols, row-major float).
  WeightBuffer PackWeight(const float* W, std::uint32_t N, std::uint32_t K);
  /// Bind a weight buffer to a program (sets kernel args 0-5).
  void Attach(WeightBuffer& wb, Program& prog);

  void EncodeA(const float* x, std::uint32_t n, std::uint32_t K, Program& p);
  void DecodeC(float* y, std::uint32_t n, Program& p,
               const std::uint8_t* base = nullptr, std::uint32_t N = 0);

  // Synchronous GEMM: encode x (n x K), run, decode to y (n x N).
  // Updates timing accumulators.
  struct Timings {
    double enc{0}, syncA{0}, submit{0}, wait{0}, syncC{0}, dec{0};
    int calls{0};
  };
  void Gemm(Program& p, WeightBuffer& wb, const float* x, std::uint32_t n,
            float* y, Timings* t = nullptr);
  // Fused FFN: decode down-output from combined C buffer (offset after 120 strips).
  void Ffn(Program& p, WeightBuffer& wb, const float* x, std::uint32_t n,
           float* y, Timings* t = nullptr);

 private:
  xrt::device dev_;
  xrt::hw_context ctx_;
  xrt::kernel kern_;
  bool ready_{false};
};

}  // namespace gufo::npu::runtime

#endif  // GUFO_NPU_RUNTIME_HPP_
