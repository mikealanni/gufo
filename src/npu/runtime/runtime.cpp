#include "src/npu/runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace gufo::npu::runtime {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
}  // namespace

void PackMatInt5(const float* W, std::uint32_t N, std::uint32_t K,
                 std::uint8_t* dst) {
  const std::uint32_t strips = N / kStrip, tk = K / kTileK;
  std::vector<std::int8_t> q(std::size_t{N} * K);
  std::vector<std::uint16_t> sc(std::size_t{N} * (K / 32));
#pragma omp parallel for
  for (std::int64_t n = 0; n < N; ++n)
    for (std::uint32_t kg = 0; kg < K / 32; ++kg) {
      const float* row = W + std::size_t(n) * K + kg * 32;
      float amax = 0;
      for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(row[i]));
      std::uint16_t sb = Bf16Bits(amax / 15.0f);
      float s = Bf16Float(sb);
      if (s == 0) {
        sb = Bf16Bits(1.0f);
        s = 1.0f;
      }
      sc[std::size_t(n) * (K / 32) + kg] = sb;
      for (int i = 0; i < 32; ++i)
        q[std::size_t(n) * K + kg * 32 + i] = static_cast<std::int8_t>(
            std::fmin(15.f, std::fmax(-15.f, std::nearbyint(row[i] / s))));
    }
#pragma omp parallel for collapse(2)
  for (std::int64_t s2 = 0; s2 < strips; ++s2)
    for (std::int64_t t = 0; t < tk; ++t)
      for (int r = 0; r < 4; ++r) {
        std::uint8_t* base =
            dst + ((std::size_t(s2) * tk + t) * 4 + r) * kQuarterBytes;
        auto* words = reinterpret_cast<std::uint16_t*>(base);
        auto* scl = reinterpret_cast<std::uint16_t*>(base + kWordBytes);
        for (int p = 0; p < 32; ++p) {
          const int sb = p >> 4, sr = (p >> 1) & 7, blk = p & 1;
          for (int lane = 0; lane < 64; ++lane) {
            const int cib = lane >> 3, kk = lane & 7;
            const std::size_t nidx = std::size_t(s2) * 128 + r * 32 + sb * 16 +
                                     blk * 8 + cib;
            const int x = q[nidx * K + t * 64 + sr * 8 + kk] + 16;
            words[(p / 3) * 64 + lane] |=
                static_cast<std::uint16_t>(x << (5 * (p % 3)));
          }
        }
        for (int sb = 0; sb < 2; ++sb)
          for (int blk = 0; blk < 2; ++blk)
            for (int qq = 0; qq < 2; ++qq)
              for (int cib = 0; cib < 8; ++cib) {
                const std::size_t nidx = std::size_t(s2) * 128 + r * 32 +
                                         sb * 16 + blk * 8 + cib;
                scl[((sb * 2 + blk) * 2 + qq) * 8 + cib] =
                    sc[nidx * (K / 32) + t * 2 + qq];
              }
      }
}

bool NpuRuntime::Init(const std::string& xclbin_path, std::string* error) {
  try {
    dev_ = xrt::device(0u);
    const xrt::xclbin xb{xclbin_path};
    dev_.register_xclbin(xb);
    ctx_ = xrt::hw_context(dev_, xb.get_uuid());
    kern_ = xrt::kernel(ctx_, xb.get_kernels().at(0).get_name());
    ready_ = true;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}

Program NpuRuntime::MakeProgram(std::uint32_t K, std::uint32_t N,
                                const std::string& insts_path,
                                std::size_t c_override, std::string* error) {
  Program p;
  p.K = K;
  p.N = N;
  std::ifstream f(insts_path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("missing " + insts_path);
  const std::size_t sz = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<char> buf(sz);
  f.read(buf.data(), static_cast<std::streamsize>(sz));
  p.words = sz / 4;
  p.insts = xrt::bo(dev_, sz, XCL_BO_FLAGS_CACHEABLE, kern_.group_id(1));
  std::memcpy(p.insts.map<void*>(), buf.data(), sz);
  p.insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const std::size_t ab = std::size_t{kMaxRows} * K / 8 * 9;
  const std::size_t cb =
      c_override ? c_override : std::size_t{kMaxRows} * N / 8 * 9;
  p.a = xrt::bo(dev_, ab, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  p.c = xrt::bo(dev_, cb, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  p.ah = p.a.map<std::uint8_t*>();
  p.ch = p.c.map<std::uint8_t*>();
  std::memset(p.ah, 0, ab);
  std::memset(p.ch, 0, cb);
  (void)error;
  return p;
}

WeightBuffer NpuRuntime::PackWeight(const float* W, std::uint32_t N,
                                    std::uint32_t K) {
  WeightBuffer wb;
  wb.bytes = std::size_t{N / kStrip} * (K / kTileK) * 4 * kQuarterBytes;
  wb.bo = xrt::bo(dev_, wb.bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  wb.host = wb.bo.map<std::uint8_t*>();
  std::memset(wb.host, 0, wb.bytes);
  PackMatInt5(W, N, K, wb.host);
  wb.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return wb;
}

void NpuRuntime::Attach(WeightBuffer& wb, Program& prog) {
  wb.run = xrt::run(kern_);
  wb.run.set_arg(0, 3u);
  wb.run.set_arg(1, prog.insts);
  wb.run.set_arg(2, static_cast<unsigned>(prog.words));
  wb.run.set_arg(3, prog.a);
  wb.run.set_arg(4, wb.bo);
  wb.run.set_arg(5, prog.c);
}

void NpuRuntime::EncodeA(const float* x, std::uint32_t n, std::uint32_t K,
                         Program& p) {
  const std::uint32_t tk = K / kTileK;
#pragma omp parallel for
  for (std::int64_t t = 0; t < tk; ++t)
    for (int gi = 0; gi < 128; ++gi) {
      const int rib = gi & 7, block = (gi >> 3) & 1, sc = gi >> 4;
      const int row = block * 8 + rib;
      std::uint8_t* o = p.ah + (std::size_t(t) * 128 + gi) * 9;
      if (row >= static_cast<int>(n)) {
        std::memset(o, 0, 9);
        continue;
      }
      const float* src = x + std::size_t(row) * K + t * 64 + sc * 8;
      float v[8];
      _mm256_storeu_ps(v, _mm256_cvtph_ps(_mm256_cvtps_ph(
                              _mm256_loadu_ps(src), _MM_FROUND_TO_NEAREST_INT)));
      EncGroup(v, o);
    }
}

void NpuRuntime::DecodeC(float* y, std::uint32_t n, Program& p,
                         const std::uint8_t* base, std::uint32_t N) {
  if (!base) {
    base = p.ch;
    N = p.N;
  }
  const std::uint32_t strips = N / kStrip;
#pragma omp parallel for
  for (std::int64_t s = 0; s < strips; ++s)
    for (int r = 0; r < 4; ++r)
      for (int G = 0; G < 64; ++G) {
        const int sb = G >> 5, acc = (G >> 3) & 3, rib = G & 7;
        const int row = (acc >> 1) * 8 + rib;
        if (row >= static_cast<int>(n))
          continue;
        const std::size_t col = std::size_t(s) * 128 + r * 32 + sb * 16 +
                                (acc & 1) * 8;
        DecGroup(base + ((std::size_t(s) * 256) + r * 64 + G) * 9,
                 y + std::size_t(row) * N + col);
      }
}

void NpuRuntime::Gemm(Program& p, WeightBuffer& wb, const float* x,
                      std::uint32_t n, float* y, Timings* t) {
  auto t0 = Clock::now();
  EncodeA(x, n, p.K, p);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  wb.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  {
    const auto spin0 = Clock::now();
    for (;;) {
      const auto st = wb.run.state();
      if (st == ERT_CMD_STATE_COMPLETED)
        break;
      if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
          st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED)
        break;
      if (Ms(spin0) > 200.0)
        break;
    }
    wb.run.wait();
  }
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  DecodeC(y, n, p);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

void NpuRuntime::Ffn(Program& p, WeightBuffer& wb, const float* x,
                     std::uint32_t n, float* y, Timings* t) {
  auto t0 = Clock::now();
  EncodeA(x, n, p.K, p);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  p.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  wb.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  for (;;) {
    const auto st = wb.run.state();
    if (st == ERT_CMD_STATE_COMPLETED)
      break;
    if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
        st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED)
      break;
  }
  wb.run.wait();
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  p.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  // Fused FFN: down output at offset 120 strips * 256 groups * 9 bytes, width 2048.
  DecodeC(y, n, p, p.ch + std::size_t{120} * 256 * 9, 2048);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

}  // namespace gufo::npu::runtime
