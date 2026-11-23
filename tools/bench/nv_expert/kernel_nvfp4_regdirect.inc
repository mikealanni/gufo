// Standalone harness for the NVFP4 routed expert GEMM (extracted from the production kernel).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace nvx {
constexpr unsigned kThreads = 256;
enum class WeightType : std::uint32_t { kF32=0,kF16=1,kQ5_1=7,kQ8_0=8,kQ3_K=11,kQ4_K=12,kQ5_K=13,kQ6_K=14,kIQ2_XXS=16,kIQ4_NL=20,kIQ3_S=21,kIQ2_S=22,kIQ4_XS=23,kBF16=30,kNVFP4=40,kQ8Rep=41 };
__device__ __forceinline__ float SigmoidF(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float SiluF(float x) { return x * SigmoidF(x); }
struct Q8_0Block { std::uint16_t d; std::int8_t qs[32]; };
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

struct Q4_1Block {
  __half d;
  __half m;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q4_1Block) == 20, "block_q4_1 must be 20 bytes");

/// Byte `i` of the 16-byte block header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ3_K      ? (k / 256) * 110
         : kType == WeightType::kQ4_K    ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K    ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1    ? (k / 32) * sizeof(Q5_1Block)
         : kType == WeightType::kIQ4_NL  ? (k / 32) * 18
         : kType == WeightType::kIQ3_S   ? (k / 256) * 110
         : kType == WeightType::kIQ2_XXS ? (k / 256) * 66
         : kType == WeightType::kIQ2_S   ? (k / 256) * 82
         : kType == WeightType::kIQ4_XS  ? (k / 256) * 136
         : kType == WeightType::kNVFP4   ? ((k / 2 + k / 16 + 15) / 16) * 16
         : kType == WeightType::kQ8Rep   ? ((k + k / 16 + 15) / 16) * 16
                                         : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

const std::uint32_t* Iq3sGridDevice();
const std::uint32_t* Iq2xxsGridDevice();
const std::uint32_t* Iq2sGridDevice();

/// Four IQ4_NL codes (one nibble per byte) to their codebook values carried
/// as signed bytes plus 128, for the 1152-magic half conversion.
__device__ __forceinline__ std::uint32_t Iq4NlBytes(std::uint32_t nibbles) {
  // kvalues_iq4nl + 128, four per word, low byte first.
  constexpr std::uint32_t kT0 = 0x3F2D1801U;  // 1, 24, 45, 63
  constexpr std::uint32_t kT1 = 0x766A5D4FU;  // 79, 93, 106, 118
  constexpr std::uint32_t kT2 = 0xA6998D81U;  // 129, 141, 153, 166
  constexpr std::uint32_t kT3 = 0xF1D9C5B5U;  // 181, 197, 217, 241
  const std::uint32_t sel = nibbles & 0x07070707U;
  const std::uint32_t lo = __builtin_amdgcn_perm(kT1, kT0, sel);
  const std::uint32_t hi = __builtin_amdgcn_perm(kT3, kT2, sel);
  const std::uint32_t mask = ((nibbles >> 3U) & 0x01010101U) * 0xFFU;
  return (hi & mask) | (lo & ~mask);
}

/// IQ3_S: the four grid magnitudes of group g of a staged 32-element
/// sub-block, signed and carried as bytes plus 128 for the 1152-magic
/// conversion. The sub-block multiplier 1 + 2 scale is in the staged scale.
__device__ __forceinline__ std::uint32_t Iq3sBytes(const uint4& chunk, int g,
                                                   const std::uint32_t* grid) {
  const std::uint32_t words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
  const unsigned index = ((words[g / 4] >> (8 * (g % 4))) & 0xFFU) |
                         (((chunk.w & 0xFFU) << (8 - g)) & 256U);
  const std::uint32_t magnitudes = grid[index];
  const std::uint32_t signs = (chunk.z >> (8 * (g / 2) + 4 * (g % 2))) & 0xFU;
  // 0xFF in each negated byte; magnitudes are odd and at least 1, so the
  // per-byte two's complement never carries into the next byte.
  const std::uint32_t neg =
      SpreadHighBits(signs) >> 4U;  // 0x01 per negated byte
  const std::uint32_t mask = neg * 0xFFU;
  return ((magnitudes ^ mask) + neg) ^ 0x80808080U;
}

/// Four signed grid magnitudes (carried as bytes plus 128): `magnitudes` with
/// byte j negated when bit j of `signs` is set. The IQ2 magnitudes (8, 25, 43)
/// are never zero, so the per-byte two's complement does not carry.
__device__ __forceinline__ std::uint32_t ApplySigns4(std::uint32_t magnitudes,
                                                     std::uint32_t signs) {
  const std::uint32_t neg = SpreadHighBits(signs) >> 4U;
  const std::uint32_t mask = neg * 0xFFU;
  return ((magnitudes ^ mask) + neg) ^ 0x80808080U;
}

/// IQ2_XXS group g (0-7) of a staged 32-element sub-block: chunk.x holds the
/// four grid indices, chunk.y the four 7-bit sign indices (the eighth sign
/// bit is the parity of the seven) and the scale in its top nibble.
__device__ __forceinline__ std::uint32_t Iq2xxsBytes(
    const uint4& chunk, int g, const std::uint32_t* grid) {
  const int l = g >> 1;
  const unsigned index = (chunk.x >> (8 * l)) & 0xFFU;
  const std::uint32_t s7 = (chunk.y >> (7 * l)) & 127U;
  const std::uint32_t sign_byte = s7 | ((__popc(s7) & 1U) << 7U);
  return ApplySigns4(grid[2 * index + (g & 1)],
                     (sign_byte >> (4 * (g & 1))) & 0xFU);
}

/// IQ2_S group g of a staged sub-block: chunk.x holds the four low grid index
/// bytes, chunk.y the four sign bytes and chunk.z the high-bit byte.
__device__ __forceinline__ std::uint32_t Iq2sBytes(const uint4& chunk, int g,
                                                   const std::uint32_t* grid) {
  const int l = g >> 1;
  const unsigned index = ((chunk.x >> (8 * l)) & 0xFFU) |
                         (((chunk.z & 0xFFU) << (8 - 2 * l)) & 0x300U);
  const std::uint32_t sign_byte = (chunk.y >> (8 * l)) & 0xFFU;
  return ApplySigns4(grid[2 * index + (g & 1)],
                     (sign_byte >> (4 * (g & 1))) & 0xFU);
}

// A 16-byte load from a 2-byte aligned address. Going through memcpy into an
// array element keeps the destination in scratch memory, which forces the
// prefetch to wait for its global load and then bounce through scratch.
typedef uint4 UnalignedUint4 __attribute__((aligned(2)));
__device__ __forceinline__ uint4 LoadUint4Aligned2(const std::uint8_t* p) {
  return *reinterpret_cast<const UnalignedUint4*>(p);
}

/// NVFP4 (E2M1) nibbles as signed bytes plus 128 (doubled values 0, 1, 2, 3,
/// 4, 6, 8, 12), four per word, for the 1152 magic conversion.
__device__ __forceinline__ std::uint32_t Nv4Bytes(std::uint32_t nibbles) {
  constexpr std::uint32_t kP0 = 0x83828180U;  // 128 + 0, 1, 2, 3
  constexpr std::uint32_t kP1 = 0x8C888684U;  // 128 + 4, 6, 8, 12
  constexpr std::uint32_t kN0 = 0x7D7E7F80U;  // 128 - 0, 1, 2, 3
  constexpr std::uint32_t kN1 = 0x74787A7CU;  // 128 - 4, 6, 8, 12
  const std::uint32_t sel = nibbles & 0x07070707U;
  const std::uint32_t pos = __builtin_amdgcn_perm(kP1, kP0, sel);
  const std::uint32_t neg = __builtin_amdgcn_perm(kN1, kN0, sel);
  const std::uint32_t mask = ((nibbles >> 3U) & 0x01010101U) * 0xFFU;
  return (neg & mask) | (pos & ~mask);
}

/// UE4M3 scale byte of an NVFP4 sub-block (includes the 0.5 of the doubled
/// E2M1 values).
__device__ __forceinline__ float Nv4Scale(std::uint32_t x) {
  const std::uint32_t e = (x >> 3) & 0xFU;
  const std::uint32_t m = x & 7U;
  const float normal = __uint_as_float(((e + 119U) << 23) | (m << 20));
  const float sub = static_cast<float>(m) * 0.0009765625F;
  const float v = e == 0 ? sub : normal;
  return x == 0x7F ? 0.0F : v;
}

template<WeightType kType, int BM, int BN, int BK, bool kPair = false,
         bool kClampSwiglu = false, bool kSitu = false>
__launch_bounds__(256) __global__ void RoutedF16GEMMKernel(
    const void* __restrict__ w, const __half* __restrict__ x,
    const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ pad_bounds,
    const std::int32_t* __restrict__ rows_in,
    const std::int32_t* __restrict__ rows_out,
    const float* __restrict__ swiglu_gate, float* __restrict__ out,
    __half* __restrict__ out_half, std::size_t m, std::size_t k,
    const void* __restrict__ w_up,
    const std::uint32_t* __restrict__ iq3s_grid = nullptr,
    float out_scale = 1.0F, const float* __restrict__ expert_scale = nullptr,
    std::size_t expert_rows = 0) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 16);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ3 = kType == WeightType::kQ3_K;
  constexpr bool kQ8R = kType == WeightType::kQ8Rep;
  constexpr bool kQ8 = kType == WeightType::kQ8_0 || kQ8R;
  constexpr bool kNL = kType == WeightType::kIQ4_NL;
  constexpr bool kXs = kType == WeightType::kIQ4_XS;
  constexpr bool kI3 = kType == WeightType::kIQ3_S;
  constexpr bool kI2X = kType == WeightType::kIQ2_XXS;
  constexpr bool kI2S = kType == WeightType::kIQ2_S;
  constexpr bool kNV = kType == WeightType::kNVFP4;
  constexpr bool kGrid = kI3 || kI2X || kI2S;
  // GLM-5.3 experts clamp the gate at 10 and the up projection to +-10.
  // IQ2_XXS always carried it; other types ask for it via kClampSwiglu.
  constexpr bool kSwigluClamp = kI2X || kClampSwiglu;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = (kQ8 || kQ3) ? 2 * BK : BK;

  // LDS plan (bytes): the code plane holds BM rows x kChunks 16-byte chunks
  // with the chunks of nearby rows permuted so a fragment read (one row per
  // lane) covers all bank groups; the activation plane is
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  // The IQ3_S codebook is gathered eight times per K block and lane; staged
  // in LDS it costs no global round trip.
  constexpr int kGridWords = kI2S ? 2048 : 512;
  __shared__ std::uint32_t s_grid[kGrid ? kGridWords : 1];
  if constexpr (kGrid) {
    for (int i = threadIdx.x; i < kGridWords; i += 256) {
      s_grid[i] = iq3s_grid[i];
    }
  }
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  if (bucket_rows - t_local <= 0)
    return;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  // Rows between two experts' weights; a fused gate/up tensor holds 2 m.
  const std::size_t expert_stride = expert_rows != 0 ? expert_rows : m;
  const auto* w_expert =
      static_cast<const std::uint8_t*>(w) +
      static_cast<std::size_t>(expert) * expert_stride * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // Weight fetch: unit u of a thread is (row = tid / 2 + 128 u, chunk c =
  // tid % 2). Q4_K: the two 16-byte halves of one 32-byte nibble group (two
  // K blocks, low and high nibbles); Q5_1: one 24-byte K block each.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((tid >> 1) >= kRows) {
        weights = static_cast<const std::uint8_t*>(w_up) +
                  static_cast<std::size_t>(expert) * expert_stride * row_bytes;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  // Paired gate/up tiles reuse the full quantized block over four stages.
  // Keep its remaining codes in registers alongside the cached header.
  uint4 code_cache[kWaveRowTiles][4];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_i3_qh[kWaveRowTiles];  ///< IQ3_S: word with the high-bit byte
  std::uint32_t f_i3_sc[kWaveRowTiles];  ///< IQ3_S: word with the scale byte
  int f_i3_ib32[kWaveRowTiles];       ///< IQ3_S: sub-block in its super-block
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q5_1: d | m; Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  // Q3_K: the superblock's packed scales and d (.w low half), fetched once
  // per superblock, plus the current group's 16 qs bytes and 16 hmask bytes
  // (one 16-element half each, selected by the fetch chunk).
  uint4 f_q3_hdr[kWaveRowTiles];  ///< scales[0..11] as three words, then d
  uint4 f_q3_q[kWaveRowTiles];    ///< qs bytes 16 f_c .. 16 f_c + 15
  uint4 f_q3_h[kWaveRowTiles];    ///< hmask bytes 16 f_c .. 16 f_c + 15
  constexpr int kActFetch = BN <= 64 ? 2 : BN <= 128 ? 4 : (BN * 8 + 255) / 256;
  uint4 a_data[kActFetch];
  uint4 w_code[2], cur_code[2];
  std::uint32_t w_sc = 0, cur_sc = 0;
  const int d_local = kPair ? (wave_id & 3) * 16 + sub_lane : wave_id * 16 + sub_lane;
  const int d_row = r_block + d_local;
  const bool d_live = d_row < m_i;
  const std::uint8_t* d_base = w_expert;
  if constexpr (kPair) {
    if (wave_id >= 4)
      d_base = static_cast<const std::uint8_t*>(w_up) +
               static_cast<std::size_t>(expert) * expert_stride * row_bytes;
  }
  const std::uint8_t* d_ptr = d_base + static_cast<std::size_t>(d_live ? d_row : m_i - 1) * row_bytes;

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  bool a_ok[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    // Dead rows read row 0 of x (always valid) and are zeroed after the
    // load: straight-line loads keep the outstanding-load count static, so
    // the wait lands at the first use instead of right after the fetch.
    a_ok[i] = src >= 0;
    a_src[i] =
        src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8) : x;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      if constexpr (kQ5) {
        const int kb = kb0 + f_c;
        const auto* words = reinterpret_cast<const uint2*>(f_ptr[u]) + (kb * 3);
        const uint2 w0 = words[0];
        const uint2 w1 = words[1];
        const uint2 w2 = words[2];
        f_codes[u] = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high[u] = w0.y;
        f_dm[u] = w0.x;
      } else if constexpr (kNL) {
        // block_iq4_nl is 18 bytes: d, then the 16 code bytes.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 18);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        f_codes[u] = LoadUint4Aligned2(blk + 2);
      } else if constexpr (kXs) {
        // block_iq4_xs: d, scales_h, scales_l[4], then 8 sub-blocks of 16
        // code bytes. The 6-bit scale is formed in commit_stage.
        const int kb = kb0 + f_c;
        const int ib32 = kb % 8;
        const std::uint8_t* blk = f_ptr[u] + (kb / 8) * 136;
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        f_i3_qh[u] = *reinterpret_cast<const std::uint16_t*>(blk + 2);
        f_i3_sc[u] = *reinterpret_cast<const std::uint32_t*>(blk + 4);
        f_i3_ib32[u] = ib32;
        f_codes[u] = LoadUint4Aligned2(blk + 8 + ib32 * 16);
      } else if constexpr (kI3) {
        // One 32-element sub-block per thread: its 8 grid indices, 4 sign
        // bytes, high index bits and scale nibble packed into one chunk.
        const int kb = kb0 + f_c;
        const int ib32 = kb % 8;
        const std::uint8_t* blk = f_ptr[u] + (kb / 8) * 110;
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        std::uint32_t q[2];
        std::uint32_t sg = 0;
        __builtin_memcpy(q, blk + 2 + ib32 * 8, 8);
        __builtin_memcpy(&sg, blk + 74 + ib32 * 4, 4);
        // Raw 16-bit words holding the high-bit byte and the scale byte; the
        // bytes are picked out in commit_stage, so nothing consumes these
        // loads before the compute.
        f_i3_qh[u] =
            *reinterpret_cast<const std::uint16_t*>(blk + 66 + (ib32 & ~1));
        f_i3_sc[u] = *reinterpret_cast<const std::uint16_t*>(blk + 106 +
                                                             ((ib32 / 2) & ~1));
        f_i3_ib32[u] = ib32;
        f_codes[u] = make_uint4(q[0], q[1], sg, 0u);
      } else if constexpr (kI2X) {
        // block_iq2_xxs is 66 bytes: d, then eight 8-byte sub-blocks of four
        // grid indices and four sign indices plus the scale nibble.
        const int kb = kb0 + f_c;
        const int ib32 = kb % 8;
        const std::uint8_t* blk = f_ptr[u] + (kb / 8) * 66;
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        std::uint32_t q[2];
        __builtin_memcpy(q, blk + 2 + ib32 * 8, 8);
        f_codes[u] = make_uint4(q[0], q[1], 0u, 0u);
      } else if constexpr (kI2S) {
        // block_iq2_s is 82 bytes: d, 32 grid index bytes, 32 sign bytes, 8
        // high-bit bytes, 8 scale bytes (a nibble per 16 elements).
        const int kb = kb0 + f_c;
        const int ib32 = kb % 8;
        const std::uint8_t* blk = f_ptr[u] + (kb / 8) * 82;
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        std::uint32_t idx = 0;
        std::uint32_t sgn = 0;
        __builtin_memcpy(&idx, blk + 2 + ib32 * 4, 4);
        __builtin_memcpy(&sgn, blk + 34 + ib32 * 4, 4);
        const std::uint32_t qh = blk[66 + ib32];
        const std::uint32_t sc = blk[74 + ib32];
        f_codes[u] = make_uint4(idx, sgn, qh | (sc << 8U), 0u);
      } else if constexpr (kNV) {
        // Repacked row: 16 nibble bytes per 32-element block, then one UE4M3
        // scale byte per 16 elements.
        w_code[0] = *reinterpret_cast<const uint4*>(d_ptr + (kb0 * 16));
        w_code[1] = *reinterpret_cast<const uint4*>(d_ptr + (kb0 * 16) + 16);
        w_sc = *reinterpret_cast<const std::uint32_t*>(d_ptr + (k / 2) + (kb0 * 2));
      } else if constexpr (kQ8R) {
        // Repacked row: 32 code bytes per K block, then one fp16 scale per
        // block; every load is 16-byte aligned.
        const int j = kb0 + f_c;
        f_codes[u] = *reinterpret_cast<const uint4*>(f_ptr[u] + (j * 32));
        f_codes_hi[u] =
            *reinterpret_cast<const uint4*>(f_ptr[u] + (j * 32) + 16);
        f_dm[u] =
            *reinterpret_cast<const std::uint16_t*>(f_ptr[u] + k + (j * 2));
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        f_codes[u] = LoadUint4Aligned2(blk + 2);
        f_codes_hi[u] = LoadUint4Aligned2(blk + 18);
      } else if constexpr (kQ3) {
        // block_q3_K is 110 bytes: hmask[0..31], qs[32..95], scales[96..107],
        // d[108..109]. The thread's 16 elements are group bytes 16 f_c into
        // the half's qs row and the hmask (shared by every group).
        const std::uint8_t* blk = f_ptr[u] + ((kb0 / 8) * 110);
        if ((kb0 & 7) == 0) {
          uint4 hdr;
          __builtin_memcpy(&hdr.x, blk + 96, 4);
          __builtin_memcpy(&hdr.y, blk + 100, 4);
          __builtin_memcpy(&hdr.z, blk + 104, 4);
          __builtin_memcpy(&hdr.w, blk + 108, 2);
          f_q3_hdr[u] = hdr;
        }
        const int half = ((kb0 & 7) >> 2);
        const std::uint8_t* qsb = blk + 32 + half * 32 + 16 * f_c;
        f_q3_q[u] = LoadUint4Aligned2(qsb);
        f_q3_h[u] = LoadUint4Aligned2(blk + 16 * f_c);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kPair) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
              code_cache[u][group] = blk[kCodeChunk + group * 2 + f_c];
          }
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        if constexpr (kPair) {
          const int group = sb32 / 2;
          // Select register values explicitly: a runtime array index or C++
          // conditional can become an address select and spill the cache.
          // Three cndmask instructions per word also avoid four ANDs and
          // three ORs. These are wave32 predicates; group is uniform.
          const std::uint32_t low = 0u - (group & 1u);
          const std::uint32_t high = 0u - ((group >> 1u) & 1u);
          const auto select = [](std::uint32_t a, std::uint32_t b,
                                 std::uint32_t mask) {
            std::uint32_t value;
            asm("v_cndmask_b32_e64 %0, %1, %2, %3"
                : "=v"(value)
                : "v"(a), "v"(b), "s"(mask));
            return value;
          };
          const uint4& c0 = code_cache[u][0];
          const uint4& c1 = code_cache[u][1];
          const uint4& c2 = code_cache[u][2];
          const uint4& c3 = code_cache[u][3];
          f_codes[u] = make_uint4(
              select(select(c0.x, c1.x, low), select(c2.x, c3.x, low), high),
              select(select(c0.y, c1.y, low), select(c2.y, c3.y, low), high),
              select(select(c0.z, c1.z, low), select(c2.z, c3.z, low), high),
              select(select(c0.w, c1.w, low), select(c2.w, c3.w, low), high));
        } else {
          f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        }
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));
    }
  };

  const auto commit_stage = [&](int kb_commit) {
    cur_code[0] = w_code[0]; cur_code[1] = w_code[1]; cur_sc = w_sc;
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kI3) {
        // Fold the sub-block's scale nibble into the staged chunk here, once
        // the fetch has had the whole compute stage to land.
        const int ib32 = f_i3_ib32[u];
        const std::uint32_t qh = (f_i3_qh[u] >> (8 * (ib32 & 1))) & 0xFFU;
        const std::uint32_t scale_byte =
            (f_i3_sc[u] >> (8 * ((ib32 >> 1) & 1))) & 0xFFU;
        const std::uint32_t sc =
            ib32 % 2 == 0 ? (scale_byte & 0x0FU) : (scale_byte >> 4U);
        f_codes[u].w = qh | (sc << 8U);
      }
      if constexpr (kQ3) {
        // Codes are final per-element bytes here: 0x7C + q2 + 4*hmask_bit
        // (range 124..127), so the signed-magic conversion recovers q - 4.
        const uint4& qs = f_q3_q[u];
        const uint4& hm = f_q3_h[u];
        const std::uint32_t qw[4] = {qs.x, qs.y, qs.z, qs.w};
        const std::uint32_t hw[4] = {hm.x, hm.y, hm.z, hm.w};
        const int g = kb_commit & 7;
        const int half = g >> 2;
        const unsigned sp = static_cast<unsigned>(g & 3);
        const uint4 hdr = f_q3_hdr[u];
        const float d = __half2float(
            __ushort_as_half(static_cast<unsigned short>(hdr.w & 0xFFFFu)));
        const auto byte_at = [hdr](int i) -> unsigned {
          const std::uint32_t wd = i < 4 ? hdr.x : (i < 8 ? hdr.y : hdr.z);
          return (wd >> (8 * (i & 3))) & 0xFFu;
        };
        const auto scv = [byte_at](int i) -> int {
          const unsigned low =
              i < 8 ? (byte_at(i) & 0xFu) : ((byte_at(i - 8) >> 4) & 0xFu);
          const unsigned hi = (byte_at(8 + (i & 3)) >> (2 * (i >> 2))) & 3u;
          return static_cast<int>(low | (hi << 4)) - 32;
        };
#pragma unroll
        for (int kbl = 0; kbl < BK; ++kbl) {
          const unsigned shift = 2u * (sp + kbl);
          const unsigned bit = static_cast<unsigned>((half << 2) + sp) + kbl;
          std::uint32_t cw[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            cw[i] = 0x7C7C7C7Cu + ((qw[i] >> shift) & 0x03030303u) +
                    (((hw[i] >> bit) & 0x01010101u) << 2);
          }
          s_codes[swizzle(row, (2 * kbl) + f_c)] =
              make_uint4(cw[0], cw[1], cw[2], cw[3]);
          const int i0 = (half * 8) + 2 * static_cast<int>(sp + kbl);
          const float s0 = f_live[u] ? d * static_cast<float>(scv(i0)) : 0.F;
          const float s1 =
              f_live[u] ? d * static_cast<float>(scv(i0 + 1)) : 0.F;
          s_scale[(kbl * BM) + row] =
              __builtin_bit_cast(std::uint32_t, __floats2half2_rn(s0, s1));
        }
      } else if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else if constexpr (!kNV) {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else if constexpr (kNL) {
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else if constexpr (kNV) {
        const float lo = f_live[u] ? Nv4Scale(f_dm[u] & 0xFFU) : 0.0F;
        const float hi = f_live[u] ? Nv4Scale((f_dm[u] >> 8U) & 0xFFU) : 0.0F;
        scale_bias =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(lo, hi));
      } else if constexpr (kXs) {
        const int ib32 = f_i3_ib32[u];
        const std::uint32_t ls =
            ((f_i3_sc[u] >> (8 * (ib32 / 2) + 4 * (ib32 & 1))) & 0x0FU) |
            (((f_i3_qh[u] >> (2 * ib32)) & 3U) << 4U);
        const float d = __half2float(
            __ushort_as_half(static_cast<unsigned short>(f_dm[u])));
        scale_bias =
            f_live[u]
                ? __builtin_bit_cast(
                      std::uint32_t,
                      __floats2half2_rn(
                          d * static_cast<float>(static_cast<int>(ls) - 32),
                          0.0F))
                : 0U;
      } else if constexpr (kI2X) {
        const float d = __half2float(
            __ushort_as_half(static_cast<unsigned short>(f_dm[u])));
        const float mult =
            (0.5F + static_cast<float>(f_codes[u].y >> 28U)) * 0.25F;
        scale_bias = f_live[u]
                         ? __builtin_bit_cast(std::uint32_t,
                                              __floats2half2_rn(d * mult, 0.0F))
                         : 0U;
      } else if constexpr (kI2S) {
        const float d = __half2float(
            __ushort_as_half(static_cast<unsigned short>(f_dm[u])));
        const std::uint32_t sc = (f_codes[u].z >> 8U) & 0xFFU;
        const float lo = d * (0.5F + static_cast<float>(sc & 0x0FU)) * 0.25F;
        const float hi = d * (0.5F + static_cast<float>(sc >> 4U)) * 0.25F;
        scale_bias = f_live[u] ? __builtin_bit_cast(std::uint32_t,
                                                    __floats2half2_rn(lo, hi))
                               : 0U;
      } else if constexpr (kI3) {
        const float d = __half2float(
            __ushort_as_half(static_cast<unsigned short>(f_dm[u])));
        const float mult =
            static_cast<float>(1 + 2 * ((f_codes[u].w >> 8U) & 0x0FU));
        scale_bias = f_live[u]
                         ? __builtin_bit_cast(std::uint32_t,
                                              __floats2half2_rn(d * mult, 0.0F))
                         : 0U;
      } else if constexpr (kQ5) {
        s_high[(f_c * BM) + row] = f_high[u];
        const __half2 dm = __builtin_bit_cast(__half2, f_dm[u]);
        const float d = f_live[u] ? __low2float(dm) : 0.0F;
        const float mn = f_live[u] ? __high2float(dm) : 0.0F;
        scale_bias =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
      } else {
        const int sb32 = f_sb32[u];
        std::uint32_t sc = 0;
        std::uint32_t mn = 0;
        if (sb32 < 4) {
          sc = HeaderByte(f_header[u], 4 + sb32) & 0x3FU;
          mn = HeaderByte(f_header[u], 8 + sb32) & 0x3FU;
        } else {
          sc = (HeaderByte(f_header[u], 8 + sb32) & 0x0FU) |
               ((HeaderByte(f_header[u], sb32) >> 6U) << 4U);
          mn = (HeaderByte(f_header[u], 8 + sb32) >> 4U) |
               ((HeaderByte(f_header[u], 4 + sb32) >> 6U) << 4U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      if constexpr (!kQ3 && !kNV) {
        s_scale[(f_c * BM) + row] = scale_bias;
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_ok[i] ? a_data[i] : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  constexpr bool kTwoScale = kI2S || kQ3 || kNV;
  constexpr bool kSigned =
      kQ8 || kNL || kXs || kI3 || kI2X || kI2S || kQ3 || kNV;
  const __half2 magic = __floats2half2_rn(kSigned ? -1152.0F : -1024.0F,
                                          kSigned ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8 && !kQ3) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = cur_code[c];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        const std::uint32_t dmw = (cur_sc >> (16 * kb)) & 0xFFFFU;
        const float s_lo = d_live ? Nv4Scale(dmw & 0xFFU) : 0.0F;
        const float s_hi = d_live ? Nv4Scale(dmw >> 8U) : 0.0F;
        const __half2 sb = __floats2half2_rn(s_lo, s_hi);
        const __half2 scale2 = __low2half2(sb);
        // IQ2_S and Q3_K carry one scale per 16 elements: (low, high).
        const __half2 scale_hi2 = kTwoScale ? __high2half2(sb) : scale2;
        const __half2 bias2 =
            kTwoScale ? __floats2half2_rn(0.0F, 0.0F) : __high2half2(sb);
        std::uint32_t nib[8];
        __half2 h[16];
        if constexpr (kQ3) {
          // Q3_K: commit_stage already produced final one-byte codes, so the
          // block's 32 elements are just its two staged 16-byte chunks.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = words[i];
          }
        } else if constexpr (kI3) {
          const uint4 r = raw[u][kb];
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            nib[g] = Iq3sBytes(r, g, s_grid);
          }
        } else if constexpr (kI2X) {
          const uint4 r = raw[u][kb];
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            nib[g] = Iq2xxsBytes(r, g, s_grid);
          }
        } else if constexpr (kI2S) {
          const uint4 r = raw[u][kb];
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            nib[g] = Iq2sBytes(r, g, s_grid);
          }
        } else if constexpr (kNV) {
          // Elements 0-7 / 8-15 are the low / high nibbles of bytes 0-7
          // (scale 0); 16-23 / 24-31 those of bytes 8-15 (scale 1).
          const uint4 r = raw[u][kb];
          nib[0] = Nv4Bytes(r.x & 0x0F0F0F0FU);
          nib[1] = Nv4Bytes(r.y & 0x0F0F0F0FU);
          nib[2] = Nv4Bytes((r.x >> 4U) & 0x0F0F0F0FU);
          nib[3] = Nv4Bytes((r.y >> 4U) & 0x0F0F0F0FU);
          nib[4] = Nv4Bytes(r.z & 0x0F0F0F0FU);
          nib[5] = Nv4Bytes(r.w & 0x0F0F0F0FU);
          nib[6] = Nv4Bytes((r.z >> 4U) & 0x0F0F0F0FU);
          nib[7] = Nv4Bytes((r.w >> 4U) & 0x0F0F0F0FU);
        } else if constexpr (kNL || kXs) {
          const uint4 r = raw[u][kb];
          const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = Iq4NlBytes(words[i] & 0x0F0F0F0FU);
            nib[4 + i] = Iq4NlBytes((words[i] >> 4U) & 0x0F0F0F0FU);
          }
        } else if constexpr (kQ8) {
          // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
          // flipping the sign bit carries q + 128, which the 1152 magic
          // takes back out.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = words[i] ^ 0x80808080U;
          }
        } else if constexpr (kQ5) {
          // Q5_1: K block kb's 16 bytes are chunk kb; elements 0-15 take
          // the low nibbles, 16-31 the high, plus bit j of the high-bit
          // word.
          const uint4 r = raw[u][kb];
          const std::uint32_t high = s_high[(kb * BM) + row];
          const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = (words[i] & 0x0F0F0F0FU) |
                     SpreadHighBits((high >> (4 * i)) & 0xFU);
            nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                         SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
          }
        } else {
          // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
          // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
          // of chunk 1; Q5_K adds bit j of the staged high-bit word.
          const unsigned shift = 4U * static_cast<unsigned>(kb);
          const std::uint32_t words[8] = {raw[u][0].x, raw[u][0].y, raw[u][0].z,
                                          raw[u][0].w, raw[u][1].x, raw[u][1].y,
                                          raw[u][1].z, raw[u][1].w};
          const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
            if constexpr (kQ5K) {
              nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(nib[i], magic,
                        (kTwoScale && i >= 4) ? scale_hi2 : scale2, bias2,
                        h[2 * i], h[2 * i + 1]);
        }
        __builtin_memcpy(&a_lo[u], &h[0], 32);
        __builtin_memcpy(&a_hi[u], &h[8], 32);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kQ8 || kQ3 || kNL || kXs || kNV ||
                                 kGrid || kKQuant) &&
                                BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kQ8 || kQ3 || kNL || kXs || kNV ||
                                  kGrid || kKQuant) &&
                                 BN >= 48)) &&
                      kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage(kb0);
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Four waves compute gate rows and four compute the matching up rows.
    // Pair them in the existing LDS allocation, keeping the K accumulation
    // order and avoiding the gate's F32 write/read between projections.
    // Two padding floats keep the accumulator scatter off repeated banks.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes);
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
      }
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = t_local + j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < bucket_rows && r_block + r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
            // Preserve the separate projection epilogue's F32 evaluation
            // order before narrowing. Fast-math can otherwise regroup the
            // products and change an F16 rounding tie.
            __half values[2];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              float gate_v = base[idx + v];
              float up_v = base[idx + v + 4 * plane];
              if constexpr (kSitu) {
                // Inkling's SiTU activation; swiglu_gate carries {beta, lb}
                const float es =
                    expert_scale != nullptr ? expert_scale[expert] : 1.0F;
                const float g = gate_v * es;
                const float u = up_v * es;
                const float beta = swiglu_gate[0];
                const float lb = swiglu_gate[1];
                values[v] = __float2half(beta * tanhf(g / beta) * SigmoidF(g) *
                                         lb * tanhf(u / lb));
                continue;
              }
              if constexpr (kSwigluClamp) {
                gate_v = fminf(gate_v, 10.0F);
                up_v = fminf(fmaxf(up_v, -10.0F), 10.0F);
              }
              float product = up_v * gate_v;
              asm volatile("" : "+v"(product));
              float value = product * SigmoidF(gate_v);
              asm volatile("" : "+v"(value));
              values[v] = __float2half(value * out_scale);
            }
            const auto offset = static_cast<std::size_t>(dst) * m + r_block + r;
            if (m % 2 == 0 && r_block + r + 1 < m_i) {
              *reinterpret_cast<__half2*>(out_half + offset) =
                  __halves2half2(values[0], values[1]);
            } else {
              out_half[offset] = values[0];
              if (r_block + r + 1 < m_i)
                out_half[offset + 1] = values[1];
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      constexpr int kRounds = 16 * BM / (256 * 2);
      float* scratch = reinterpret_cast<float*>(lds);
      // Destination rows for every (token tile, round), loaded together so
      // their latency overlaps instead of repeating behind each barrier.
      std::int32_t dsts[kTokTiles][kRounds];
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int round = 0; round < kRounds; ++round) {
          const unsigned tr = ((round * 256 + tid) * 2) / BM;
          const unsigned t = t_local + j * 16 + tr;
          dsts[j][round] =
              t < unsigned(bucket_rows) ? rows_out[bucket_begin + t] : -1;
        }
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < kRounds; ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned r = r_block + row;
          if (r < m) {
            const int dst = dsts[j][round];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              if (expert_scale != nullptr) {
                const float es = expert_scale[expert];
                v.x *= es;
                v.y *= es;
              }
              const size_t o = size_t(dst) * m + r;
              if (swiglu_gate != nullptr) {
                v.x *= SiluF(swiglu_gate[o]);
                if (r + 1 < m)
                  v.y *= SiluF(swiglu_gate[o + 1]);
              }
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (expert_scale != nullptr)
              v *= expert_scale[expert];
            if (out_half != nullptr) {
              out_half[o] = __float2half(
                  swiglu_gate != nullptr ? v * SiluF(swiglu_gate[o]) : v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}
}  // namespace nvx
