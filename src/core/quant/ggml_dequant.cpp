#include "src/core/quant/ggml_dequant.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace gufo::quant {

// Standard half-precision float to single-precision float conversion
float Fp16ToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      float zero = 0.0f;
      std::memcpy(&zero, &sign, sizeof(float));
      return zero;
    }
    const float f =
        (static_cast<float>(mant) / 1024.0f) * std::ldexp(1.0f, -14);
    return (h & 0x8000) ? -f : f;
  }
  if (exp == 31) {
    const std::uint32_t val = sign | 0x7F800000U | (mant << 13);
    float f = 0.0f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
  }

  const std::uint32_t val = sign | ((exp + 112) << 23) | (mant << 13);
  float f = 0.0f;
  std::memcpy(&f, &val, sizeof(float));
  return f;
}

// Block layouts (block_q4_K, block_q5_K, block_q6_K, block_q3_K,
// block_q8_K, block_q8_0) now live in the header; they are the canonical
// layout source. Definitions removed here to avoid ODR redefinition.

namespace {

void GetQ4ScaleMin(std::size_t index, const std::uint8_t* packed,
                   std::uint8_t& scale, std::uint8_t& minimum) noexcept {
  if (index < 4) {
    scale = packed[index] & 0x3FU;
    minimum = packed[index + 4] & 0x3FU;
    return;
  }
  scale = static_cast<std::uint8_t>((packed[index + 4] & 0x0FU) |
                                    ((packed[index - 4] >> 6U) << 4U));
  minimum = static_cast<std::uint8_t>((packed[index + 4] >> 4U) |
                                      ((packed[index] >> 6U) << 4U));
}

std::array<std::int8_t, 16> UnpackQ3Scales(
    const std::uint8_t* packed) noexcept {
  std::array<std::int8_t, 16> scales{};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    const auto low =
        index < 8 ? packed[index] & 0x0FU : (packed[index - 8] >> 4U) & 0x0FU;
    const auto high = (packed[8 + (index % 4)] >> (2U * (index / 4))) & 0x03U;
    scales[index] =
        static_cast<std::int8_t>(static_cast<int>(low | (high << 4U)) - 32);
  }
  return scales;
}

float Q4Value(const block_q4_K& block, std::size_t index) noexcept {
  const std::size_t group = index / 32;
  const std::size_t pair = group / 2;
  const bool high_nibble = (group & 1U) != 0;
  const std::size_t lane = index % 32;
  const std::uint8_t packed = block.qs[(pair * 32) + lane];
  const std::uint8_t quant = high_nibble ? packed >> 4U : packed & 0x0FU;
  std::uint8_t scale = 0;
  std::uint8_t minimum = 0;
  GetQ4ScaleMin(group, block.scales, scale, minimum);
  return Fp16ToFloat(block.d) * static_cast<float>(scale) *
             static_cast<float>(quant) -
         Fp16ToFloat(block.dmin) * static_cast<float>(minimum);
}

float Q5Value(const block_q5_K& block, std::size_t index) noexcept {
  const std::size_t gg = index / 64;  // 0..3
  const std::size_t wv = index % 64;  // 0..63
  const std::size_t lane = wv % 32;   // 0..31
  const bool lohalf = (wv < 32);
  const std::uint8_t qb = block.qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const std::uint8_t qhb = block.qh[lane];
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((qhb >> bit) & 1U) ? 16U : 0U));       // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);  // 0..7
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQ4ScaleMin(sis, block.scales, sc, m);
  return Fp16ToFloat(block.d) * static_cast<float>(sc) *
             static_cast<float>(quant) -
         Fp16ToFloat(block.dmin) * static_cast<float>(m);
}

float Q6Value(const block_q6_K& block, std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within_half = index % 128;
  const std::size_t segment = within_half / 32;
  const std::size_t lane = within_half % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh = block.qh[(half * 32) + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = qh & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh >> 6U) & 0x03U;
      break;
  }

  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const auto quant =
      static_cast<std::int8_t>(static_cast<int>((high << 4U) | low) - 32);
  return Fp16ToFloat(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

float Q3Value(const block_q3_K& block,
              const std::array<std::int8_t, 16>& scales,
              std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within_half = index % 128;
  const std::size_t scale_pair = within_half / 32;
  const std::size_t lane = within_half % 32;
  const std::size_t quant_index = (half * 32) + lane;
  const std::uint8_t shift = static_cast<std::uint8_t>(2U * scale_pair);
  const std::uint8_t high_mask =
      static_cast<std::uint8_t>(1U << ((half * 4) + scale_pair));
  const auto low =
      static_cast<std::int8_t>((block.qs[quant_index] >> shift) & 0x03U);
  const auto quant = static_cast<std::int8_t>(
      static_cast<int>(low) - ((block.hmask[lane] & high_mask) != 0 ? 0 : 4));
  const std::size_t scale_index = (half * 8) + (scale_pair * 2) + (lane / 16);
  return Fp16ToFloat(block.d) * static_cast<float>(scales[scale_index]) *
         static_cast<float>(quant);
}

}  // namespace

std::size_t QuantizedRowBytes(core::GgmlType type,
                              std::size_t elements) noexcept {
  const std::size_t block_qk = QuantizedBlockElements(type);
  std::size_t block_bytes = 0;
  switch (type) {
    // GGML storage layouts also used by the DeepSeek runtime. Storage support
    // here does not imply that every model has a compute kernel for the type.
    case core::GgmlType::kQ4_0:
      block_bytes = 18;
      break;
    case core::GgmlType::kQ4_1:
      block_bytes = 20;
      break;
    case core::GgmlType::kQ5_0:
      block_bytes = 22;
      break;
    case core::GgmlType::kQ5_1:
      block_bytes = 24;
      break;
    case core::GgmlType::kQ8_1:
    case core::GgmlType::kNVFP4:
      block_bytes = 36;
      break;
    case core::GgmlType::kQ2_K:
      block_bytes = 84;
      break;
    case core::GgmlType::kIQ2_XXS:
      block_bytes = 66;
      break;
    case core::GgmlType::kIQ2_XS:
      block_bytes = 74;
      break;
    case core::GgmlType::kIQ3_XXS:
      block_bytes = 98;
      break;
    case core::GgmlType::kIQ2_S:
      block_bytes = 82;
      break;
    case core::GgmlType::kQ3_K:
      block_bytes = sizeof(block_q3_K);
      break;
    case core::GgmlType::kQ4_K:
      block_bytes = sizeof(block_q4_K);
      break;
    case core::GgmlType::kQ5_K:
      block_bytes = sizeof(block_q5_K);
      break;
    case core::GgmlType::kQ6_K:
      block_bytes = sizeof(block_q6_K);
      break;
    case core::GgmlType::kQ8_K:
      block_bytes = sizeof(block_q8_K);
      break;
    case core::GgmlType::kQ8_0:
      block_bytes = sizeof(block_q8_0);
      break;
    case core::GgmlType::kIQ4_NL:
      block_bytes = sizeof(block_iq4_nl);
      break;
    case core::GgmlType::kIQ4_XS:
      block_bytes = sizeof(block_iq4_xs);
      break;
    case core::GgmlType::kIQ3_S:
      block_bytes = sizeof(block_iq3_s);
      break;
    default:
      return 0;
  }
  if ((elements % block_qk) != 0) {
    return 0;
  }
  const std::size_t blocks = elements / block_qk;
  if (blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
    return 0;
  }
  return blocks * block_bytes;
}

std::size_t EncodedSizeBytes(core::GgmlType type,
                             std::size_t elements) noexcept {
  switch (type) {
    case core::GgmlType::kI32:
      if (elements >
          std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
        return 0;
      }
      return elements * sizeof(std::int32_t);
    case core::GgmlType::kF32:
      if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return 0;
      }
      return elements * sizeof(float);
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      if (elements >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        return 0;
      }
      return elements * sizeof(std::uint16_t);
    default:
      return QuantizedRowBytes(type, elements);
  }
}

void DequantizeQ4_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q4Value(blocks[b], i);
    }
  }
}

void DequantizeQ5_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q5_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q5Value(blocks[b], i);
    }
  }
}

void DequantizeQ6_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q6_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q6Value(blocks[b], i);
    }
  }
}

void DequantizeQ3_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q3_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    const auto scales = UnpackQ3Scales(blocks[b].scales);
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q3Value(blocks[b], scales, i);
    }
  }
}

void DequantizeQ8_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q8_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = blocks[b].d * static_cast<float>(blocks[b].qs[i]);
    }
  }
}

float DotProductQ4_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q4Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ5_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q5_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q5Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ6_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q6_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q6Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ3_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q3_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const auto scales = UnpackQ3Scales(blocks[b].scales);
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q3Value(blocks[b], scales, i) * v[i];
    }
  }
  return sum;
}

float DotProductQ8_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q8_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += blocks[b].d * static_cast<float>(blocks[b].qs[i]) * v[i];
    }
  }
  return sum;
}

void DequantizeQ8_0(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q8_0*>(src);
  const std::size_t nb = k / 32;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    for (std::size_t i = 0; i < 32; ++i) {
      dst[(b * 32) + i] = d * static_cast<float>(blocks[b].qs[i]);
    }
  }
}

float DotProductQ8_0(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q8_0*>(row_data);
  const std::size_t nb = k / 32;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    const float* v = vec.data() + b * 32;

    for (std::size_t i = 0; i < 32; ++i) {
      sum += d * static_cast<float>(blocks[b].qs[i]) * v[i];
    }
  }
  return sum;
}

// ---------------------------------------------------------------------------
// IQ4_NL / IQ4_XS / IQ3_S (opt-q4kxl): the Unsloth UD-Q4_K_XL Qwen3.8-27B shard
// mixes these three with Q3_K/Q4_K/Q5_K/Q6_K/Q8_0, so every one of them needs a
// CPU oracle here before the HIP path can be validated against it. All three
// mirror ggml-quants.c (dequantize_row_iq4_nl / _iq4_xs / _iq3_s) element for
// element.
// ---------------------------------------------------------------------------

namespace {

// 512-entry IQ3_S grid, verbatim from ggml-common.h. Each entry packs four
// uint8 magnitudes for one group of four elements.
constexpr std::uint32_t kIq3sGrid[512] = {
    0x01010101U, 0x01010103U, 0x01010105U, 0x0101010bU, 0x0101010fU,
    0x01010301U, 0x01010303U, 0x01010305U, 0x01010309U, 0x0101030dU,
    0x01010501U, 0x01010503U, 0x0101050bU, 0x01010707U, 0x01010901U,
    0x01010905U, 0x0101090bU, 0x0101090fU, 0x01010b03U, 0x01010b07U,
    0x01010d01U, 0x01010d05U, 0x01010f03U, 0x01010f09U, 0x01010f0fU,
    0x01030101U, 0x01030103U, 0x01030105U, 0x01030109U, 0x01030301U,
    0x01030303U, 0x0103030bU, 0x01030501U, 0x01030507U, 0x0103050fU,
    0x01030703U, 0x0103070bU, 0x01030909U, 0x01030d03U, 0x01030d0bU,
    0x01030f05U, 0x01050101U, 0x01050103U, 0x0105010bU, 0x0105010fU,
    0x01050301U, 0x01050307U, 0x0105030dU, 0x01050503U, 0x0105050bU,
    0x01050701U, 0x01050709U, 0x01050905U, 0x0105090bU, 0x0105090fU,
    0x01050b03U, 0x01050b07U, 0x01050f01U, 0x01050f07U, 0x01070107U,
    0x01070303U, 0x0107030bU, 0x01070501U, 0x01070505U, 0x01070703U,
    0x01070707U, 0x0107070dU, 0x01070909U, 0x01070b01U, 0x01070b05U,
    0x01070d0fU, 0x01070f03U, 0x01070f0bU, 0x01090101U, 0x01090307U,
    0x0109030fU, 0x01090503U, 0x01090509U, 0x01090705U, 0x01090901U,
    0x01090907U, 0x01090b03U, 0x01090f01U, 0x010b0105U, 0x010b0109U,
    0x010b0501U, 0x010b0505U, 0x010b050dU, 0x010b0707U, 0x010b0903U,
    0x010b090bU, 0x010b090fU, 0x010b0d0dU, 0x010b0f07U, 0x010d010dU,
    0x010d0303U, 0x010d0307U, 0x010d0703U, 0x010d0b05U, 0x010d0f03U,
    0x010f0101U, 0x010f0105U, 0x010f0109U, 0x010f0501U, 0x010f0505U,
    0x010f050dU, 0x010f0707U, 0x010f0b01U, 0x010f0b09U, 0x03010101U,
    0x03010103U, 0x03010105U, 0x03010109U, 0x03010301U, 0x03010303U,
    0x03010307U, 0x0301030bU, 0x0301030fU, 0x03010501U, 0x03010505U,
    0x03010703U, 0x03010709U, 0x0301070dU, 0x03010b09U, 0x03010b0dU,
    0x03010d03U, 0x03010f05U, 0x03030101U, 0x03030103U, 0x03030107U,
    0x0303010dU, 0x03030301U, 0x03030309U, 0x03030503U, 0x03030701U,
    0x03030707U, 0x03030903U, 0x03030b01U, 0x03030b05U, 0x03030f01U,
    0x03030f0dU, 0x03050101U, 0x03050305U, 0x0305030bU, 0x0305030fU,
    0x03050501U, 0x03050509U, 0x03050705U, 0x03050901U, 0x03050907U,
    0x03050b0bU, 0x03050d01U, 0x03050f05U, 0x03070103U, 0x03070109U,
    0x0307010fU, 0x03070301U, 0x03070307U, 0x03070503U, 0x0307050fU,
    0x03070701U, 0x03070709U, 0x03070903U, 0x03070d05U, 0x03070f01U,
    0x03090107U, 0x0309010bU, 0x03090305U, 0x03090309U, 0x03090703U,
    0x03090707U, 0x03090905U, 0x0309090dU, 0x03090b01U, 0x03090b09U,
    0x030b0103U, 0x030b0301U, 0x030b0307U, 0x030b0503U, 0x030b0701U,
    0x030b0705U, 0x030b0b03U, 0x030d0501U, 0x030d0509U, 0x030d050fU,
    0x030d0909U, 0x030d090dU, 0x030f0103U, 0x030f0107U, 0x030f0301U,
    0x030f0305U, 0x030f0503U, 0x030f070bU, 0x030f0903U, 0x030f0d05U,
    0x030f0f01U, 0x05010101U, 0x05010103U, 0x05010107U, 0x0501010bU,
    0x0501010fU, 0x05010301U, 0x05010305U, 0x05010309U, 0x0501030dU,
    0x05010503U, 0x05010507U, 0x0501050fU, 0x05010701U, 0x05010705U,
    0x05010903U, 0x05010907U, 0x0501090bU, 0x05010b01U, 0x05010b05U,
    0x05010d0fU, 0x05010f01U, 0x05010f07U, 0x05010f0bU, 0x05030101U,
    0x05030105U, 0x05030301U, 0x05030307U, 0x0503030fU, 0x05030505U,
    0x0503050bU, 0x05030703U, 0x05030709U, 0x05030905U, 0x05030b03U,
    0x05050103U, 0x05050109U, 0x0505010fU, 0x05050503U, 0x05050507U,
    0x05050701U, 0x0505070fU, 0x05050903U, 0x05050b07U, 0x05050b0fU,
    0x05050f03U, 0x05050f09U, 0x05070101U, 0x05070105U, 0x0507010bU,
    0x05070303U, 0x05070505U, 0x05070509U, 0x05070703U, 0x05070707U,
    0x05070905U, 0x05070b01U, 0x05070d0dU, 0x05090103U, 0x0509010fU,
    0x05090501U, 0x05090507U, 0x05090705U, 0x0509070bU, 0x05090903U,
    0x05090f05U, 0x05090f0bU, 0x050b0109U, 0x050b0303U, 0x050b0505U,
    0x050b070fU, 0x050b0901U, 0x050b0b07U, 0x050b0f01U, 0x050d0101U,
    0x050d0105U, 0x050d010fU, 0x050d0503U, 0x050d0b0bU, 0x050d0d03U,
    0x050f010bU, 0x050f0303U, 0x050f050dU, 0x050f0701U, 0x050f0907U,
    0x050f0b01U, 0x07010105U, 0x07010303U, 0x07010307U, 0x0701030bU,
    0x0701030fU, 0x07010505U, 0x07010703U, 0x07010707U, 0x0701070bU,
    0x07010905U, 0x07010909U, 0x0701090fU, 0x07010b03U, 0x07010d07U,
    0x07010f03U, 0x07030103U, 0x07030107U, 0x0703010bU, 0x07030309U,
    0x07030503U, 0x07030507U, 0x07030901U, 0x07030d01U, 0x07030f05U,
    0x07030f0dU, 0x07050101U, 0x07050305U, 0x07050501U, 0x07050705U,
    0x07050709U, 0x07050b01U, 0x07070103U, 0x07070301U, 0x07070309U,
    0x07070503U, 0x07070507U, 0x0707050fU, 0x07070701U, 0x07070903U,
    0x07070907U, 0x0707090fU, 0x07070b0bU, 0x07070f07U, 0x07090107U,
    0x07090303U, 0x0709030dU, 0x07090505U, 0x07090703U, 0x07090b05U,
    0x07090d01U, 0x07090d09U, 0x070b0103U, 0x070b0301U, 0x070b0305U,
    0x070b050bU, 0x070b0705U, 0x070b0909U, 0x070b0b0dU, 0x070b0f07U,
    0x070d030dU, 0x070d0903U, 0x070f0103U, 0x070f0107U, 0x070f0501U,
    0x070f0505U, 0x070f070bU, 0x09010101U, 0x09010109U, 0x09010305U,
    0x09010501U, 0x09010509U, 0x0901050fU, 0x09010705U, 0x09010903U,
    0x09010b01U, 0x09010f01U, 0x09030105U, 0x0903010fU, 0x09030303U,
    0x09030307U, 0x09030505U, 0x09030701U, 0x0903070bU, 0x09030907U,
    0x09030b03U, 0x09030b0bU, 0x09050103U, 0x09050107U, 0x09050301U,
    0x0905030bU, 0x09050503U, 0x09050707U, 0x09050901U, 0x09050b0fU,
    0x09050d05U, 0x09050f01U, 0x09070109U, 0x09070303U, 0x09070307U,
    0x09070501U, 0x09070505U, 0x09070703U, 0x0907070bU, 0x09090101U,
    0x09090105U, 0x09090509U, 0x0909070fU, 0x09090901U, 0x09090f03U,
    0x090b010bU, 0x090b010fU, 0x090b0503U, 0x090b0d05U, 0x090d0307U,
    0x090d0709U, 0x090d0d01U, 0x090f0301U, 0x090f030bU, 0x090f0701U,
    0x090f0907U, 0x090f0b03U, 0x0b010105U, 0x0b010301U, 0x0b010309U,
    0x0b010505U, 0x0b010901U, 0x0b010909U, 0x0b01090fU, 0x0b010b05U,
    0x0b010d0dU, 0x0b010f09U, 0x0b030103U, 0x0b030107U, 0x0b03010bU,
    0x0b030305U, 0x0b030503U, 0x0b030705U, 0x0b030f05U, 0x0b050101U,
    0x0b050303U, 0x0b050507U, 0x0b050701U, 0x0b05070dU, 0x0b050b07U,
    0x0b070105U, 0x0b07010fU, 0x0b070301U, 0x0b07050fU, 0x0b070909U,
    0x0b070b03U, 0x0b070d0bU, 0x0b070f07U, 0x0b090103U, 0x0b090109U,
    0x0b090501U, 0x0b090705U, 0x0b09090dU, 0x0b0b0305U, 0x0b0b050dU,
    0x0b0b0b03U, 0x0b0b0b07U, 0x0b0d0905U, 0x0b0f0105U, 0x0b0f0109U,
    0x0b0f0505U, 0x0d010303U, 0x0d010307U, 0x0d01030bU, 0x0d010703U,
    0x0d010707U, 0x0d010d01U, 0x0d030101U, 0x0d030501U, 0x0d03050fU,
    0x0d030d09U, 0x0d050305U, 0x0d050709U, 0x0d050905U, 0x0d050b0bU,
    0x0d050d05U, 0x0d050f01U, 0x0d070101U, 0x0d070309U, 0x0d070503U,
    0x0d070901U, 0x0d09050bU, 0x0d090907U, 0x0d090d05U, 0x0d0b0101U,
    0x0d0b0107U, 0x0d0b0709U, 0x0d0b0d01U, 0x0d0d010bU, 0x0d0d0901U,
    0x0d0f0303U, 0x0d0f0307U, 0x0f010101U, 0x0f010109U, 0x0f01010fU,
    0x0f010501U, 0x0f010505U, 0x0f01070dU, 0x0f010901U, 0x0f010b09U,
    0x0f010d05U, 0x0f030105U, 0x0f030303U, 0x0f030509U, 0x0f030907U,
    0x0f03090bU, 0x0f050103U, 0x0f050109U, 0x0f050301U, 0x0f05030dU,
    0x0f050503U, 0x0f050701U, 0x0f050b03U, 0x0f070105U, 0x0f070705U,
    0x0f07070bU, 0x0f070b07U, 0x0f090103U, 0x0f09010bU, 0x0f090307U,
    0x0f090501U, 0x0f090b01U, 0x0f0b0505U, 0x0f0b0905U, 0x0f0d0105U,
    0x0f0d0703U, 0x0f0f0101U,
};

constexpr std::uint8_t kSignMaskIq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

float Iq4NlValue(const block_iq4_nl& block, std::size_t index) noexcept {
  const std::size_t lane = index % 16;
  const bool high_nibble = index >= 16;
  const std::uint8_t packed = block.qs[lane];
  const std::uint8_t code = high_nibble ? (packed >> 4U) : (packed & 0x0FU);
  return Fp16ToFloat(block.d) * static_cast<float>(kValuesIq4Nl[code]);
}

float Iq4XsValue(const block_iq4_xs& block, std::size_t index) noexcept {
  const std::size_t ib = index / 32;  // 0..7 sub-block
  const std::size_t within = index % 32;
  const std::size_t lane = within % 16;
  const bool high_nibble = within >= 16;
  const std::uint8_t packed = block.qs[(ib * 16) + lane];
  const std::uint8_t code = high_nibble ? (packed >> 4U) : (packed & 0x0FU);
  const int ls =
      static_cast<int>((block.scales_l[ib / 2] >> (4U * (ib % 2))) & 0x0FU) |
      static_cast<int>(((block.scales_h >> (2U * ib)) & 0x03U) << 4U);
  return Fp16ToFloat(block.d) * static_cast<float>(ls - 32) *
         static_cast<float>(kValuesIq4Nl[code]);
}

float Iq3sValue(const block_iq3_s& block, std::size_t index) noexcept {
  const std::size_t ib32 = index / 32;    // 0..7 sub-block
  const std::size_t within = index % 32;  // 0..31
  const std::size_t l = within / 8;       // 0..3 group of eight
  const std::size_t j = within % 8;       // 0..7 element in the group
  // Each group of eight is two grid lookups of four values each.
  const std::size_t half = j / 4;  // 0 -> grid1, 1 -> grid2
  const std::size_t jj = j % 4;

  const std::uint8_t qh_byte = block.qh[ib32];
  const std::size_t qs_index = (ib32 * 8) + (2 * l) + half;
  const int shift = static_cast<int>(8 - (2 * l) - half);
  const std::uint32_t grid_index =
      static_cast<std::uint32_t>(block.qs[qs_index]) |
      ((static_cast<std::uint32_t>(qh_byte) << shift) & 256U);
  const auto* grid =
      reinterpret_cast<const std::uint8_t*>(&kIq3sGrid[grid_index]);

  const std::uint8_t sign_byte = block.signs[(ib32 * 4) + l];
  const bool negate = (sign_byte & kSignMaskIq2xs[j]) != 0;

  const std::uint8_t scale_byte = block.scales[ib32 / 2];
  const int scale_nibble = static_cast<int>(
      (ib32 % 2 == 0) ? (scale_byte & 0x0FU)
                      : static_cast<unsigned>(scale_byte >> 4U));
  const float db =
      Fp16ToFloat(block.d) * static_cast<float>(1 + (2 * scale_nibble));

  const float magnitude = static_cast<float>(grid[jj]);
  return negate ? -db * magnitude : db * magnitude;
}

}  // namespace

const std::uint32_t* Iq3sGrid() noexcept {
  return kIq3sGrid;
}

void DequantizeIQ4_NL(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_nl*>(src);
  const std::size_t nb = k / 32;
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 32; ++i) {
      dst[(b * 32) + i] = Iq4NlValue(blocks[b], i);
    }
  }
}

void DequantizeIQ4_XS(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_xs*>(src);
  const std::size_t nb = k / 256;
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Iq4XsValue(blocks[b], i);
    }
  }
}

void DequantizeIQ3_S(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_iq3_s*>(src);
  const std::size_t nb = k / 256;
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Iq3sValue(blocks[b], i);
    }
  }
}

float DotProductIQ4_NL(const void* row_data, std::span<const float> vec,
                       std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_nl*>(row_data);
  const std::size_t nb = k / 32;
  float sum = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + (b * 32);
    for (std::size_t i = 0; i < 32; ++i) {
      sum += Iq4NlValue(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductIQ4_XS(const void* row_data, std::span<const float> vec,
                       std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_xs*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + (b * 256);
    for (std::size_t i = 0; i < 256; ++i) {
      sum += Iq4XsValue(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductIQ3_S(const void* row_data, std::span<const float> vec,
                      std::size_t k) {
  const auto* blocks = static_cast<const block_iq3_s*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + (b * 256);
    for (std::size_t i = 0; i < 256; ++i) {
      sum += Iq3sValue(blocks[b], i) * v[i];
    }
  }
  return sum;
}

// ---------------------------------------------------------------------------
// IQ2_XXS / IQ2_S (GLM-5.3-Flash AJ MoE experts): mirrors ggml-quants.c
// dequantize_row_iq2_xxs / _iq2_s. Grids carry eight packed magnitudes per
// entry; tables verbatim from ggml-common.h.
// ---------------------------------------------------------------------------

namespace {

constexpr std::uint64_t kIq2xxsGrid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL,
    0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL,
    0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL,
    0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL,
    0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL,
    0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL,
    0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL,
    0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL,
    0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL,
    0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL,
    0x0808190819082b08ULL, 0x08081908192b0808ULL, 0x080819082b080819ULL,
    0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL,
    0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL,
    0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL,
    0x0808192b08190808ULL, 0x0808192b19080808ULL, 0x0808192b2b081908ULL,
    0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL,
    0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL,
    0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL,
    0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL,
    0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL,
    0x081908082b081908ULL, 0x081908082b190808ULL, 0x081908082b191919ULL,
    0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL,
    0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL,
    0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL,
    0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL,
    0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL,
    0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL,
    0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL,
    0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL,
    0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL,
    0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL,
    0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL,
    0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL,
    0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL,
    0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL,
    0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL,
    0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL,
    0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL,
    0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL,
    0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL,
    0x190819082b192b19ULL, 0x190819190819082bULL, 0x19081919082b1908ULL,
    0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL,
    0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL,
    0x1919080808080808ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL,
    0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL,
    0x191908192b2b1908ULL, 0x1919082b2b190819ULL, 0x191919082b190808ULL,
    0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL,
    0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL,
    0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL,
    0x192b080819080808ULL, 0x192b0808192b2b08ULL, 0x192b081908080808ULL,
    0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL,
    0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL,
    0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL,
    0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL,
    0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL,
    0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL,
    0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL,
    0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL,
    0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL,
    0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL,
    0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL,
    0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL,
    0x2b2b2b1908081908ULL,
};

constexpr std::uint64_t kIq2sGrid[1024] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL,
    0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL,
    0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL,
    0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL,
    0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL,
    0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL,
    0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL,
    0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x08080808192b192bULL,
    0x08080808192b2b19ULL, 0x080808082b080808ULL, 0x080808082b08082bULL,
    0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL,
    0x080808082b191908ULL, 0x080808082b2b0808ULL, 0x080808082b2b1919ULL,
    0x080808082b2b2b2bULL, 0x0808081908080819ULL, 0x0808081908081908ULL,
    0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL,
    0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL,
    0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL,
    0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL,
    0x0808081919190819ULL, 0x0808081919191908ULL, 0x080808191919192bULL,
    0x0808081919192b19ULL, 0x08080819192b0808ULL, 0x08080819192b1919ULL,
    0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL,
    0x080808192b190808ULL, 0x080808192b19082bULL, 0x080808192b191919ULL,
    0x080808192b2b0819ULL, 0x080808192b2b1908ULL, 0x0808082b08080808ULL,
    0x0808082b0808082bULL, 0x0808082b08081919ULL, 0x0808082b08082b08ULL,
    0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL,
    0x0808082b082b2b2bULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL,
    0x0808082b1908192bULL, 0x0808082b19082b19ULL, 0x0808082b19190808ULL,
    0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b081919ULL,
    0x0808082b2b082b2bULL, 0x0808082b2b191908ULL, 0x0808082b2b2b082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL,
    0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL,
    0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x08081908082b192bULL, 0x08081908082b2b19ULL,
    0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL,
    0x0808190819082b08ULL, 0x0808190819082b2bULL, 0x0808190819190819ULL,
    0x0808190819191908ULL, 0x080819081919192bULL, 0x0808190819192b19ULL,
    0x08081908192b0808ULL, 0x08081908192b082bULL, 0x08081908192b1919ULL,
    0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b08192bULL,
    0x080819082b082b19ULL, 0x080819082b190808ULL, 0x080819082b191919ULL,
    0x080819082b192b08ULL, 0x080819082b2b0819ULL, 0x080819082b2b1908ULL,
    0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL,
    0x0808191908082b08ULL, 0x0808191908082b2bULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x080819190819192bULL, 0x0808191908192b19ULL,
    0x08081919082b0808ULL, 0x08081919082b1919ULL, 0x08081919082b2b08ULL,
    0x0808191919080819ULL, 0x0808191919081908ULL, 0x080819191908192bULL,
    0x0808191919082b19ULL, 0x0808191919190808ULL, 0x080819191919082bULL,
    0x0808191919191919ULL, 0x0808191919192b08ULL, 0x08081919192b0819ULL,
    0x08081919192b1908ULL, 0x080819192b080808ULL, 0x080819192b08082bULL,
    0x080819192b081919ULL, 0x080819192b082b08ULL, 0x080819192b190819ULL,
    0x080819192b191908ULL, 0x080819192b2b0808ULL, 0x0808192b08080819ULL,
    0x0808192b08081908ULL, 0x0808192b0808192bULL, 0x0808192b08082b19ULL,
    0x0808192b08190808ULL, 0x0808192b08191919ULL, 0x0808192b19080808ULL,
    0x0808192b19081919ULL, 0x0808192b19082b08ULL, 0x0808192b19190819ULL,
    0x0808192b19191908ULL, 0x0808192b192b0808ULL, 0x0808192b2b080819ULL,
    0x0808192b2b081908ULL, 0x0808192b2b190808ULL, 0x08082b0808080808ULL,
    0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL,
    0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b080819192bULL,
    0x08082b0808192b19ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL,
    0x08082b08082b2b2bULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL,
    0x08082b081908192bULL, 0x08082b0819082b19ULL, 0x08082b0819190808ULL,
    0x08082b081919082bULL, 0x08082b0819191919ULL, 0x08082b0819192b08ULL,
    0x08082b08192b0819ULL, 0x08082b08192b1908ULL, 0x08082b082b080808ULL,
    0x08082b082b081919ULL, 0x08082b082b191908ULL, 0x08082b082b2b2b2bULL,
    0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL,
    0x08082b190819082bULL, 0x08082b1908191919ULL, 0x08082b1908192b08ULL,
    0x08082b19082b0819ULL, 0x08082b1919080808ULL, 0x08082b1919081919ULL,
    0x08082b1919082b08ULL, 0x08082b1919190819ULL, 0x08082b1919191908ULL,
    0x08082b19192b0808ULL, 0x08082b192b080819ULL, 0x08082b192b190808ULL,
    0x08082b2b08080808ULL, 0x08082b2b08190819ULL, 0x08082b2b08191908ULL,
    0x08082b2b082b082bULL, 0x08082b2b082b2b08ULL, 0x08082b2b082b2b2bULL,
    0x08082b2b19190808ULL, 0x08082b2b2b192b19ULL, 0x0819080808080819ULL,
    0x0819080808081908ULL, 0x081908080808192bULL, 0x0819080808082b19ULL,
    0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL,
    0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL,
    0x08190808082b192bULL, 0x0819080819080808ULL, 0x081908081908082bULL,
    0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL,
    0x0819080819191908ULL, 0x081908081919192bULL, 0x0819080819192b19ULL,
    0x08190808192b0808ULL, 0x08190808192b082bULL, 0x08190808192b1919ULL,
    0x08190808192b2b08ULL, 0x081908082b080819ULL, 0x081908082b081908ULL,
    0x081908082b08192bULL, 0x081908082b190808ULL, 0x081908082b191919ULL,
    0x081908082b192b08ULL, 0x081908082b2b0819ULL, 0x081908082b2b1908ULL,
    0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL,
    0x0819081908082b08ULL, 0x0819081908082b2bULL, 0x0819081908190819ULL,
    0x0819081908191908ULL, 0x081908190819192bULL, 0x0819081908192b19ULL,
    0x08190819082b0808ULL, 0x08190819082b082bULL, 0x08190819082b1919ULL,
    0x08190819082b2b08ULL, 0x0819081919080819ULL, 0x0819081919081908ULL,
    0x081908191908192bULL, 0x0819081919082b19ULL, 0x0819081919190808ULL,
    0x081908191919082bULL, 0x0819081919191919ULL, 0x0819081919192b08ULL,
    0x08190819192b0819ULL, 0x08190819192b1908ULL, 0x081908192b080808ULL,
    0x081908192b08082bULL, 0x081908192b081919ULL, 0x081908192b082b08ULL,
    0x081908192b190819ULL, 0x081908192b191908ULL, 0x0819082b08080819ULL,
    0x0819082b08081908ULL, 0x0819082b08082b19ULL, 0x0819082b08190808ULL,
    0x0819082b08191919ULL, 0x0819082b082b0819ULL, 0x0819082b082b1908ULL,
    0x0819082b19080808ULL, 0x0819082b19081919ULL, 0x0819082b19190819ULL,
    0x0819082b19191908ULL, 0x0819082b2b080819ULL, 0x0819082b2b081908ULL,
    0x0819082b2b190808ULL, 0x0819190808080808ULL, 0x081919080808082bULL,
    0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL,
    0x0819190808191908ULL, 0x081919080819192bULL, 0x0819190808192b19ULL,
    0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x08191908082b2b08ULL,
    0x0819190819080819ULL, 0x0819190819081908ULL, 0x081919081908192bULL,
    0x0819190819082b19ULL, 0x0819190819190808ULL, 0x081919081919082bULL,
    0x0819190819191919ULL, 0x0819190819192b08ULL, 0x08191908192b0819ULL,
    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x081919082b08082bULL,
    0x081919082b081919ULL, 0x081919082b082b08ULL, 0x081919082b190819ULL,
    0x081919082b191908ULL, 0x081919082b2b0808ULL, 0x0819191908080819ULL,
    0x0819191908081908ULL, 0x081919190808192bULL, 0x0819191908082b19ULL,
    0x0819191908190808ULL, 0x081919190819082bULL, 0x0819191908191919ULL,
    0x0819191908192b08ULL, 0x08191919082b0819ULL, 0x08191919082b1908ULL,
    0x0819191919080808ULL, 0x081919191908082bULL, 0x0819191919081919ULL,
    0x0819191919082b08ULL, 0x0819191919190819ULL, 0x0819191919191908ULL,
    0x08191919192b0808ULL, 0x081919192b080819ULL, 0x081919192b081908ULL,
    0x081919192b190808ULL, 0x0819192b08080808ULL, 0x0819192b08081919ULL,
    0x0819192b08082b08ULL, 0x0819192b08190819ULL, 0x0819192b08191908ULL,
    0x0819192b082b0808ULL, 0x0819192b19080819ULL, 0x0819192b19081908ULL,
    0x0819192b19190808ULL, 0x0819192b2b080808ULL, 0x0819192b2b2b2b2bULL,
    0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b080808192bULL,
    0x08192b0808082b19ULL, 0x08192b0808190808ULL, 0x08192b0808191919ULL,
    0x08192b0808192b08ULL, 0x08192b08082b0819ULL, 0x08192b0819080808ULL,
    0x08192b081908082bULL, 0x08192b0819081919ULL, 0x08192b0819082b08ULL,
    0x08192b0819190819ULL, 0x08192b0819191908ULL, 0x08192b08192b0808ULL,
    0x08192b082b080819ULL, 0x08192b082b081908ULL, 0x08192b1908080808ULL,
    0x08192b190808082bULL, 0x08192b1908081919ULL, 0x08192b1908082b08ULL,
    0x08192b1908190819ULL, 0x08192b1908191908ULL, 0x08192b19082b0808ULL,
    0x08192b1919080819ULL, 0x08192b1919081908ULL, 0x08192b1919190808ULL,
    0x08192b19192b2b19ULL, 0x08192b192b2b082bULL, 0x08192b2b08081908ULL,
    0x08192b2b08190808ULL, 0x08192b2b19080808ULL, 0x08192b2b1919192bULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL,
    0x082b080808082b08ULL, 0x082b080808190819ULL, 0x082b080808191908ULL,
    0x082b08080819192bULL, 0x082b080808192b19ULL, 0x082b0808082b0808ULL,
    0x082b0808082b1919ULL, 0x082b0808082b2b2bULL, 0x082b080819080819ULL,
    0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08081919082bULL,
    0x082b080819191919ULL, 0x082b0808192b1908ULL, 0x082b08082b080808ULL,
    0x082b08082b082b2bULL, 0x082b08082b191908ULL, 0x082b08082b2b2b2bULL,
    0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL,
    0x082b08190819082bULL, 0x082b081908191919ULL, 0x082b0819082b0819ULL,
    0x082b081919080808ULL, 0x082b08191908082bULL, 0x082b081919081919ULL,
    0x082b081919190819ULL, 0x082b081919191908ULL, 0x082b0819192b0808ULL,
    0x082b08192b080819ULL, 0x082b08192b081908ULL, 0x082b08192b190808ULL,
    0x082b082b08080808ULL, 0x082b082b08082b2bULL, 0x082b082b082b082bULL,
    0x082b082b082b2b08ULL, 0x082b082b082b2b2bULL, 0x082b082b19081908ULL,
    0x082b082b19190808ULL, 0x082b082b2b082b08ULL, 0x082b082b2b082b2bULL,
    0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL,
    0x082b19080808192bULL, 0x082b190808082b19ULL, 0x082b190808190808ULL,
    0x082b190808191919ULL, 0x082b190808192b08ULL, 0x082b1908082b0819ULL,
    0x082b1908082b1908ULL, 0x082b190819080808ULL, 0x082b19081908082bULL,
    0x082b190819081919ULL, 0x082b190819082b08ULL, 0x082b190819190819ULL,
    0x082b190819191908ULL, 0x082b1908192b0808ULL, 0x082b19082b080819ULL,
    0x082b19082b081908ULL, 0x082b19082b190808ULL, 0x082b191908080808ULL,
    0x082b191908081919ULL, 0x082b191908082b08ULL, 0x082b191908190819ULL,
    0x082b191908191908ULL, 0x082b1919082b0808ULL, 0x082b191919080819ULL,
    0x082b191919081908ULL, 0x082b191919190808ULL, 0x082b1919192b192bULL,
    0x082b19192b080808ULL, 0x082b192b08080819ULL, 0x082b192b08081908ULL,
    0x082b192b08190808ULL, 0x082b192b19080808ULL, 0x082b192b19192b19ULL,
    0x082b2b0808080808ULL, 0x082b2b0808081919ULL, 0x082b2b0808190819ULL,
    0x082b2b0808191908ULL, 0x082b2b0819080819ULL, 0x082b2b0819081908ULL,
    0x082b2b0819190808ULL, 0x082b2b082b082b2bULL, 0x082b2b082b2b2b2bULL,
    0x082b2b1908080819ULL, 0x082b2b1908081908ULL, 0x082b2b1908190808ULL,
    0x082b2b192b191919ULL, 0x082b2b2b08082b2bULL, 0x082b2b2b082b082bULL,
    0x082b2b2b192b1908ULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x190808080808192bULL,
    0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL,
    0x1908080808191919ULL, 0x1908080808192b08ULL, 0x1908080808192b2bULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x19080808082b192bULL,
    0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL,
    0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL,
    0x1908080819191908ULL, 0x190808081919192bULL, 0x1908080819192b19ULL,
    0x19080808192b0808ULL, 0x19080808192b082bULL, 0x19080808192b1919ULL,
    0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL,
    0x190808082b191919ULL, 0x190808082b192b08ULL, 0x190808082b2b0819ULL,
    0x190808082b2b1908ULL, 0x1908081908080808ULL, 0x190808190808082bULL,
    0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL,
    0x1908081908191908ULL, 0x190808190819192bULL, 0x1908081908192b19ULL,
    0x19080819082b0808ULL, 0x19080819082b082bULL, 0x19080819082b1919ULL,
    0x1908081919080819ULL, 0x1908081919081908ULL, 0x190808191908192bULL,
    0x1908081919082b19ULL, 0x1908081919190808ULL, 0x190808191919082bULL,
    0x1908081919191919ULL, 0x1908081919192b08ULL, 0x19080819192b0819ULL,
    0x19080819192b1908ULL, 0x190808192b080808ULL, 0x190808192b08082bULL,
    0x190808192b081919ULL, 0x190808192b082b08ULL, 0x190808192b190819ULL,
    0x190808192b191908ULL, 0x190808192b2b0808ULL, 0x1908082b08080819ULL,
    0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL,
    0x1908082b08191919ULL, 0x1908082b08192b08ULL, 0x1908082b082b1908ULL,
    0x1908082b19080808ULL, 0x1908082b19081919ULL, 0x1908082b19082b08ULL,
    0x1908082b19190819ULL, 0x1908082b19191908ULL, 0x1908082b192b0808ULL,
    0x1908082b2b080819ULL, 0x1908082b2b081908ULL, 0x1908190808080808ULL,
    0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL,
    0x1908190808082b2bULL, 0x1908190808190819ULL, 0x1908190808191908ULL,
    0x190819080819192bULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL,
    0x19081908082b082bULL, 0x19081908082b1919ULL, 0x19081908082b2b08ULL,
    0x1908190819080819ULL, 0x1908190819081908ULL, 0x190819081908192bULL,
    0x1908190819082b19ULL, 0x1908190819190808ULL, 0x190819081919082bULL,
    0x1908190819191919ULL, 0x1908190819192b08ULL, 0x19081908192b0819ULL,
    0x19081908192b1908ULL, 0x190819082b080808ULL, 0x190819082b08082bULL,
    0x190819082b081919ULL, 0x190819082b082b08ULL, 0x190819082b190819ULL,
    0x190819082b191908ULL, 0x190819082b2b0808ULL, 0x1908191908080819ULL,
    0x1908191908081908ULL, 0x190819190808192bULL, 0x1908191908082b19ULL,
    0x1908191908190808ULL, 0x190819190819082bULL, 0x1908191908191919ULL,
    0x1908191908192b08ULL, 0x19081919082b0819ULL, 0x19081919082b1908ULL,
    0x1908191919080808ULL, 0x190819191908082bULL, 0x1908191919081919ULL,
    0x1908191919082b08ULL, 0x1908191919190819ULL, 0x1908191919191908ULL,
    0x19081919192b0808ULL, 0x19081919192b2b2bULL, 0x190819192b080819ULL,
    0x190819192b081908ULL, 0x190819192b190808ULL, 0x1908192b08080808ULL,
    0x1908192b0808082bULL, 0x1908192b08081919ULL, 0x1908192b08082b08ULL,
    0x1908192b08190819ULL, 0x1908192b08191908ULL, 0x1908192b082b0808ULL,
    0x1908192b19080819ULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL,
    0x1908192b2b080808ULL, 0x1908192b2b2b1919ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808082b19ULL, 0x19082b0808190808ULL,
    0x19082b080819082bULL, 0x19082b0808191919ULL, 0x19082b0808192b08ULL,
    0x19082b08082b0819ULL, 0x19082b08082b1908ULL, 0x19082b0819080808ULL,
    0x19082b081908082bULL, 0x19082b0819081919ULL, 0x19082b0819082b08ULL,
    0x19082b0819190819ULL, 0x19082b0819191908ULL, 0x19082b08192b0808ULL,
    0x19082b082b081908ULL, 0x19082b082b190808ULL, 0x19082b1908080808ULL,
    0x19082b190808082bULL, 0x19082b1908081919ULL, 0x19082b1908082b08ULL,
    0x19082b1908190819ULL, 0x19082b1908191908ULL, 0x19082b19082b0808ULL,
    0x19082b1919080819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL,
    0x19082b192b080808ULL, 0x19082b192b19192bULL, 0x19082b2b08080819ULL,
    0x19082b2b08081908ULL, 0x19082b2b08190808ULL, 0x19082b2b19080808ULL,
    0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL,
    0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL,
    0x191908080819192bULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL,
    0x19190808082b082bULL, 0x19190808082b1919ULL, 0x19190808082b2b08ULL,
    0x1919080819080819ULL, 0x1919080819081908ULL, 0x191908081908192bULL,
    0x1919080819082b19ULL, 0x1919080819190808ULL, 0x191908081919082bULL,
    0x1919080819191919ULL, 0x1919080819192b08ULL, 0x19190808192b0819ULL,
    0x19190808192b1908ULL, 0x191908082b080808ULL, 0x191908082b08082bULL,
    0x191908082b081919ULL, 0x191908082b082b08ULL, 0x191908082b190819ULL,
    0x191908082b191908ULL, 0x1919081908080819ULL, 0x1919081908081908ULL,
    0x191908190808192bULL, 0x1919081908082b19ULL, 0x1919081908190808ULL,
    0x191908190819082bULL, 0x1919081908191919ULL, 0x1919081908192b08ULL,
    0x19190819082b0819ULL, 0x19190819082b1908ULL, 0x1919081919080808ULL,
    0x191908191908082bULL, 0x1919081919081919ULL, 0x1919081919082b08ULL,
    0x1919081919190819ULL, 0x1919081919191908ULL, 0x19190819192b0808ULL,
    0x191908192b080819ULL, 0x191908192b081908ULL, 0x191908192b190808ULL,
    0x1919082b08080808ULL, 0x1919082b08081919ULL, 0x1919082b08082b08ULL,
    0x1919082b08190819ULL, 0x1919082b08191908ULL, 0x1919082b082b0808ULL,
    0x1919082b19080819ULL, 0x1919082b19081908ULL, 0x1919082b19190808ULL,
    0x1919082b192b2b19ULL, 0x1919082b2b080808ULL, 0x1919190808080819ULL,
    0x1919190808081908ULL, 0x191919080808192bULL, 0x1919190808082b19ULL,
    0x1919190808190808ULL, 0x191919080819082bULL, 0x1919190808191919ULL,
    0x1919190808192b08ULL, 0x19191908082b0819ULL, 0x19191908082b1908ULL,
    0x1919190819080808ULL, 0x191919081908082bULL, 0x1919190819081919ULL,
    0x1919190819082b08ULL, 0x1919190819190819ULL, 0x1919190819191908ULL,
    0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b081908ULL,
    0x191919082b190808ULL, 0x1919191908080808ULL, 0x191919190808082bULL,
    0x1919191908081919ULL, 0x1919191908082b08ULL, 0x1919191908190819ULL,
    0x1919191908191908ULL, 0x19191919082b0808ULL, 0x1919191919080819ULL,
    0x1919191919081908ULL, 0x1919191919190808ULL, 0x191919192b080808ULL,
    0x1919192b08080819ULL, 0x1919192b08081908ULL, 0x1919192b08190808ULL,
    0x1919192b082b192bULL, 0x1919192b19080808ULL, 0x19192b0808080808ULL,
    0x19192b080808082bULL, 0x19192b0808081919ULL, 0x19192b0808082b08ULL,
    0x19192b0808190819ULL, 0x19192b0808191908ULL, 0x19192b08082b0808ULL,
    0x19192b0819080819ULL, 0x19192b0819081908ULL, 0x19192b0819190808ULL,
    0x19192b0819192b2bULL, 0x19192b082b080808ULL, 0x19192b1908080819ULL,
    0x19192b1908081908ULL, 0x19192b1908190808ULL, 0x19192b1919080808ULL,
    0x19192b2b08080808ULL, 0x19192b2b08192b19ULL, 0x19192b2b2b081919ULL,
    0x19192b2b2b2b2b08ULL, 0x192b080808080819ULL, 0x192b080808081908ULL,
    0x192b08080808192bULL, 0x192b080808190808ULL, 0x192b08080819082bULL,
    0x192b080808191919ULL, 0x192b080808192b08ULL, 0x192b0808082b0819ULL,
    0x192b0808082b1908ULL, 0x192b080819080808ULL, 0x192b080819081919ULL,
    0x192b080819082b08ULL, 0x192b080819190819ULL, 0x192b080819191908ULL,
    0x192b0808192b0808ULL, 0x192b08082b081908ULL, 0x192b08082b190808ULL,
    0x192b081908080808ULL, 0x192b08190808082bULL, 0x192b081908081919ULL,
    0x192b081908082b08ULL, 0x192b081908190819ULL, 0x192b081908191908ULL,
    0x192b0819082b0808ULL, 0x192b081919080819ULL, 0x192b081919081908ULL,
    0x192b081919190808ULL, 0x192b08192b080808ULL, 0x192b08192b192b19ULL,
    0x192b082b08081908ULL, 0x192b082b08190808ULL, 0x192b082b19080808ULL,
    0x192b082b1919192bULL, 0x192b082b2b2b0819ULL, 0x192b190808080808ULL,
    0x192b190808081919ULL, 0x192b190808082b08ULL, 0x192b190808190819ULL,
    0x192b190808191908ULL, 0x192b1908082b0808ULL, 0x192b190819080819ULL,
    0x192b190819081908ULL, 0x192b190819190808ULL, 0x192b19082b080808ULL,
    0x192b191908080819ULL, 0x192b191908081908ULL, 0x192b191908190808ULL,
    0x192b191919080808ULL, 0x192b191919082b2bULL, 0x192b1919192b2b08ULL,
    0x192b19192b19082bULL, 0x192b192b08080808ULL, 0x192b192b2b191908ULL,
    0x192b2b0808080819ULL, 0x192b2b0808081908ULL, 0x192b2b0808190808ULL,
    0x192b2b08192b1919ULL, 0x192b2b082b192b08ULL, 0x192b2b1908080808ULL,
    0x192b2b19082b2b2bULL, 0x192b2b2b1908082bULL, 0x192b2b2b2b2b0819ULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL,
    0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL,
    0x2b08080808192b19ULL, 0x2b080808082b0808ULL, 0x2b080808082b1919ULL,
    0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL,
    0x2b0808081919082bULL, 0x2b08080819191919ULL, 0x2b08080819192b08ULL,
    0x2b080808192b0819ULL, 0x2b0808082b080808ULL, 0x2b0808082b081919ULL,
    0x2b0808082b190819ULL, 0x2b0808082b191908ULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b08081908082b19ULL, 0x2b08081908190808ULL,
    0x2b0808190819082bULL, 0x2b08081908191919ULL, 0x2b08081908192b08ULL,
    0x2b080819082b0819ULL, 0x2b080819082b1908ULL, 0x2b08081919080808ULL,
    0x2b0808191908082bULL, 0x2b08081919081919ULL, 0x2b08081919082b08ULL,
    0x2b08081919190819ULL, 0x2b08081919191908ULL, 0x2b0808192b080819ULL,
    0x2b0808192b081908ULL, 0x2b0808192b190808ULL, 0x2b0808192b2b2b19ULL,
    0x2b08082b08080808ULL, 0x2b08082b08081919ULL, 0x2b08082b08082b2bULL,
    0x2b08082b08190819ULL, 0x2b08082b08191908ULL, 0x2b08082b19080819ULL,
    0x2b08082b19081908ULL, 0x2b08082b19190808ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b0819080808192bULL, 0x2b08190808082b19ULL,
    0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL,
    0x2b08190808192b08ULL, 0x2b081908082b0819ULL, 0x2b08190819080808ULL,
    0x2b0819081908082bULL, 0x2b08190819081919ULL, 0x2b08190819082b08ULL,
    0x2b08190819190819ULL, 0x2b08190819191908ULL, 0x2b081908192b0808ULL,
    0x2b0819082b080819ULL, 0x2b0819082b081908ULL, 0x2b0819082b190808ULL,
    0x2b08191908080808ULL, 0x2b0819190808082bULL, 0x2b08191908081919ULL,
    0x2b08191908082b08ULL, 0x2b08191908190819ULL, 0x2b08191908191908ULL,
    0x2b081919082b0808ULL, 0x2b08191919080819ULL, 0x2b08191919081908ULL,
    0x2b08191919190808ULL, 0x2b0819192b080808ULL, 0x2b0819192b082b2bULL,
    0x2b08192b08080819ULL, 0x2b08192b08081908ULL, 0x2b08192b08190808ULL,
    0x2b08192b082b2b19ULL, 0x2b08192b19080808ULL, 0x2b082b0808080808ULL,
    0x2b082b0808081919ULL, 0x2b082b0808190819ULL, 0x2b082b0808191908ULL,
    0x2b082b0819080819ULL, 0x2b082b0819081908ULL, 0x2b082b0819190808ULL,
    0x2b082b082b2b082bULL, 0x2b082b1908080819ULL, 0x2b082b1908081908ULL,
    0x2b082b1919080808ULL, 0x2b082b19192b1919ULL, 0x2b082b2b082b082bULL,
    0x2b082b2b19192b08ULL, 0x2b082b2b19192b2bULL, 0x2b082b2b2b08082bULL,
    0x2b082b2b2b2b082bULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL,
    0x2b19080808082b19ULL, 0x2b19080808190808ULL, 0x2b1908080819082bULL,
    0x2b19080808191919ULL, 0x2b19080808192b08ULL, 0x2b190808082b1908ULL,
    0x2b19080819080808ULL, 0x2b1908081908082bULL, 0x2b19080819081919ULL,
    0x2b19080819082b08ULL, 0x2b19080819190819ULL, 0x2b19080819191908ULL,
    0x2b190808192b0808ULL, 0x2b1908082b080819ULL, 0x2b1908082b081908ULL,
    0x2b1908082b190808ULL, 0x2b19081908080808ULL, 0x2b19081908081919ULL,
    0x2b19081908190819ULL, 0x2b19081908191908ULL, 0x2b19081919080819ULL,
    0x2b19081919081908ULL, 0x2b19081919190808ULL, 0x2b19081919192b2bULL,
    0x2b19082b08080819ULL, 0x2b19082b08081908ULL, 0x2b19082b08190808ULL,
    0x2b19082b19080808ULL, 0x2b19082b2b2b192bULL, 0x2b19190808080808ULL,
    0x2b1919080808082bULL, 0x2b19190808081919ULL, 0x2b19190808082b08ULL,
    0x2b19190808190819ULL, 0x2b19190808191908ULL, 0x2b191908082b0808ULL,
    0x2b19190819080819ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL,
    0x2b1919082b080808ULL, 0x2b1919082b19192bULL, 0x2b19191908080819ULL,
    0x2b19191908081908ULL, 0x2b19191908190808ULL, 0x2b19191919080808ULL,
    0x2b1919192b192b08ULL, 0x2b1919192b2b0819ULL, 0x2b19192b08080808ULL,
    0x2b19192b1908192bULL, 0x2b19192b192b1908ULL, 0x2b192b0808080819ULL,
    0x2b192b0808081908ULL, 0x2b192b0808190808ULL, 0x2b192b08082b192bULL,
    0x2b192b0819080808ULL, 0x2b192b082b2b2b19ULL, 0x2b192b1908080808ULL,
    0x2b192b1919082b19ULL, 0x2b192b191919082bULL, 0x2b192b2b2b190808ULL,
    0x2b2b080808080808ULL, 0x2b2b080808081919ULL, 0x2b2b080808082b2bULL,
    0x2b2b080808191908ULL, 0x2b2b0808082b082bULL, 0x2b2b0808082b2b2bULL,
    0x2b2b080819080819ULL, 0x2b2b080819081908ULL, 0x2b2b080819190808ULL,
    0x2b2b08082b2b082bULL, 0x2b2b08082b2b2b2bULL, 0x2b2b081919080808ULL,
    0x2b2b0819192b1919ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b2bULL,
    0x2b2b082b082b082bULL, 0x2b2b082b082b2b08ULL, 0x2b2b082b082b2b2bULL,
    0x2b2b082b2b08082bULL, 0x2b2b082b2b082b08ULL, 0x2b2b082b2b082b2bULL,
    0x2b2b082b2b2b2b08ULL, 0x2b2b190808080819ULL, 0x2b2b190808081908ULL,
    0x2b2b190808190808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b082b19ULL,
    0x2b2b19082b2b1908ULL, 0x2b2b191908080808ULL, 0x2b2b191908192b19ULL,
    0x2b2b192b19190819ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b2b08ULL,
    0x2b2b2b082b2b082bULL, 0x2b2b2b1919191908ULL, 0x2b2b2b192b08192bULL,
    0x2b2b2b2b08082b08ULL, 0x2b2b2b2b08082b2bULL, 0x2b2b2b2b082b0808ULL,
    0x2b2b2b2b082b082bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b2b082b08ULL,
    0x2b2b2b2b2b2b2b2bULL,
};

constexpr std::uint8_t kSignsIq2xs[128] = {
    0U,   129U, 130U, 3U,   132U, 5U,   6U,   135U, 136U, 9U,   10U,  139U,
    12U,  141U, 142U, 15U,  144U, 17U,  18U,  147U, 20U,  149U, 150U, 23U,
    24U,  153U, 154U, 27U,  156U, 29U,  30U,  159U, 160U, 33U,  34U,  163U,
    36U,  165U, 166U, 39U,  40U,  169U, 170U, 43U,  172U, 45U,  46U,  175U,
    48U,  177U, 178U, 51U,  180U, 53U,  54U,  183U, 184U, 57U,  58U,  187U,
    60U,  189U, 190U, 63U,  192U, 65U,  66U,  195U, 68U,  197U, 198U, 71U,
    72U,  201U, 202U, 75U,  204U, 77U,  78U,  207U, 80U,  209U, 210U, 83U,
    212U, 85U,  86U,  215U, 216U, 89U,  90U,  219U, 92U,  221U, 222U, 95U,
    96U,  225U, 226U, 99U,  228U, 101U, 102U, 231U, 232U, 105U, 106U, 235U,
    108U, 237U, 238U, 111U, 240U, 113U, 114U, 243U, 116U, 245U, 246U, 119U,
    120U, 249U, 250U, 123U, 252U, 125U, 126U, 255U,
};

constexpr std::uint8_t kMaskIq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

}  // namespace

const std::uint32_t* Iq2xxsGridWords() noexcept {
  return reinterpret_cast<const std::uint32_t*>(kIq2xxsGrid);
}

const std::uint32_t* Iq2sGridWords() noexcept {
  return reinterpret_cast<const std::uint32_t*>(kIq2sGrid);
}

void DequantizeIQ2_XXS(const void* src, float* dst, std::size_t k) {
  struct Blk {
    std::uint16_t d;
    std::uint16_t qs[32];
  };
  const auto* x = static_cast<const Blk*>(src);
  const std::size_t nb = k / 256;
  std::uint32_t aux32[2];
  const std::uint8_t* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
  float* y = dst;
  for (std::size_t i = 0; i < nb; ++i) {
    const float d = Fp16ToFloat(x[i].d);
    for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
      std::memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(std::uint32_t));
      const float db = d * (0.5F + static_cast<float>(aux32[1] >> 28)) * 0.25F;
      for (std::size_t l = 0; l < 4; ++l) {
        const auto* grid =
            reinterpret_cast<const std::uint8_t*>(kIq2xxsGrid + aux8[l]);
        const std::uint8_t signs = kSignsIq2xs[(aux32[1] >> 7 * l) & 127];
        for (std::size_t j = 0; j < 8; ++j) {
          y[j] = db * grid[j] * ((signs & kMaskIq2xs[j]) != 0 ? -1.0F : 1.0F);
        }
        y += 8;
      }
    }
  }
}

void DequantizeIQ2_S(const void* src, float* dst, std::size_t k) {
  struct Blk {
    std::uint16_t d;
    std::uint8_t qs[64];
    std::uint8_t qh[8];
    std::uint8_t scales[8];
  };
  const auto* x = static_cast<const Blk*>(src);
  const std::size_t nb = k / 256;
  float* y = dst;
  for (std::size_t i = 0; i < nb; ++i) {
    const float d = Fp16ToFloat(x[i].d);
    const std::uint8_t* qs = x[i].qs;
    const std::uint8_t* qh = x[i].qh;
    const std::uint8_t* signs = qs + 32;
    for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
      const float db0 =
          d * (0.5F + static_cast<float>(x[i].scales[ib32] & 0xF)) * 0.25F;
      const float db1 =
          d * (0.5F + static_cast<float>(x[i].scales[ib32] >> 4)) * 0.25F;
      for (std::size_t l = 0; l < 4; ++l) {
        const float dl = (l < 2) ? db0 : db1;
        const auto* grid = reinterpret_cast<const std::uint8_t*>(
            kIq2sGrid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
        for (std::size_t j = 0; j < 8; ++j) {
          y[j] =
              dl * grid[j] * ((signs[l] & kMaskIq2xs[j]) != 0 ? -1.0F : 1.0F);
        }
        y += 8;
      }
      qs += 4;
      signs += 4;
    }
  }
}

}  // namespace gufo::quant
