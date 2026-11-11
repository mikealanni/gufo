#include "src/npu/runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <omp.h>
#include <stdexcept>

namespace gufo::npu::runtime {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
}  // namespace

void PackMatInt5(const float* W, std::uint32_t N, std::uint32_t K,
                 std::uint8_t* dst, std::uint32_t extra_tiles) {
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
            dst + ((std::size_t(s2) * (tk + extra_tiles) + t) * 4 + r) * kQuarterBytes;
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
                                std::size_t c_override, std::string* error,
                                std::uint32_t a_extra_k) {
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
  const std::size_t ab = std::size_t{kMaxRows} * (K + a_extra_k) / 8 * 9;
  const std::size_t cb = c_override ? c_override : std::size_t{N / kStrip} * kStripBytes;
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
        DecGroup(base + std::size_t(s) * kStripBytes + std::size_t(r) * kRowBytes +
                     std::size_t(G) * 9,
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
  DecodeC(y, n, p, p.ch + std::size_t{120} * kStripBytes, 2048);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

void NpuRuntime::Argmax(Program& p, WeightBuffer& wb, const float* x,
                        std::uint32_t n, std::uint32_t* out, Timings* t) {
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
  const std::uint32_t strips = p.N / kStrip;
  struct Cand { float v; std::uint32_t id; };
  std::vector<Cand> best(n, {-INFINITY, 0});
  for (std::uint32_t si = 0; si < strips; ++si) {
    for (std::uint32_t cr = 0; cr < 4; ++cr) {
      const std::uint8_t* base = p.ch + std::size_t{si} * kStripBytes + std::size_t{cr} * kRowBytes;
      for (std::uint32_t r = 0; r < n; ++r) {
        float v;
        std::uint32_t lc;
        std::memcpy(&v, base + r * 8, 4);
        std::memcpy(&lc, base + r * 8 + 4, 4);
        const std::uint32_t id = si * kStrip + cr * 32 + lc;
        if (v > best[r].v || (v == best[r].v && id < best[r].id)) {
          best[r].v = v;
          best[r].id = id;
        }
      }
    }
  }
  for (std::uint32_t r = 0; r < n; ++r) out[r] = best[r].id;
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
}

namespace {
// Partial BO syncs that start or end inside a page are not reliable: flush whole pages.
void SyncToDevice(xrt::bo& bo, std::size_t offset, std::size_t size) {
  constexpr std::size_t kPage = 4096;
  const std::size_t lo = offset / kPage * kPage;
  const std::size_t hi = std::min<std::size_t>(bo.size(), (offset + size + kPage - 1) / kPage * kPage);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, hi - lo, lo);
}
constexpr std::uint32_t kConvHistOffset = 512;
constexpr std::uint32_t kConvTapOffset = 64;
constexpr std::uint32_t kConvMaxTaps = 7;
constexpr std::uint32_t kChanPerTriple = 1024;

inline std::size_t ConvRow(std::uint32_t role, std::uint32_t chan) {
  const std::uint32_t j = chan >> 10, col = (chan >> 7) & 7, r = (chan >> 5) & 3,
                      c = chan & 31;
  return (std::size_t{3 * j + role} * 8 + col) * 128 + r * 32 + c;
}
}  // namespace

ConvLayer NpuRuntime::PackConv(const float* w_in, const float* conv_w,
                               std::uint32_t taps, const float* w_out,
                               std::uint32_t hidden, Program& tmpl) {
  if (taps != kConvMaxTaps || hidden % kChanPerTriple != 0)
    throw std::runtime_error("conv layer shape unsupported by the conv program");
  ConvLayer cl;
  cl.hidden = hidden;
  cl.taps = taps;
  cl.prog = tmpl;
  const std::size_t cb = std::size_t{(3 * hidden + hidden) / kStrip} * kStripBytes;
  cl.prog.c = xrt::bo(dev_, cb, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  cl.prog.ch = cl.prog.c.map<std::uint8_t*>();
  std::memset(cl.prog.ch, 0, cb);
  const std::uint32_t K = hidden, tk = K / kTileK;
  const std::size_t in_bytes = std::size_t{3 * hidden / kStrip} * (tk + 1) * 4 * kQuarterBytes;
  const std::size_t out_bytes = std::size_t{hidden / kStrip} * tk * 4 * kQuarterBytes;
  cl.w.bytes = in_bytes + out_bytes;
  cl.w.bo = xrt::bo(dev_, cl.w.bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  cl.w.host = cl.w.bo.map<std::uint8_t*>();
  std::memset(cl.w.host, 0, cl.w.bytes);
  {
    std::vector<float> perm(std::size_t{3 * hidden} * K);
    const std::uint32_t src_base[3] = {0, 2 * hidden, hidden};  // B, x, C row blocks of [B|C|x]
#pragma omp parallel for
    for (std::int64_t ch = 0; ch < hidden; ++ch)
      for (std::uint32_t role = 0; role < 3; ++role)
        std::memcpy(&perm[ConvRow(role, ch) * K],
                    w_in + (std::size_t{src_base[role]} + ch) * K, K * 4);
    PackMatInt5(perm.data(), 3 * hidden, K, cl.w.host, 1);
  }
  PackMatInt5(w_out, hidden, K, cl.w.host + in_bytes, 0);
  const std::uint32_t groups = 3 * hidden / kStrip / 8;
  for (std::uint32_t g = 0; g < groups; ++g)
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t r = 0; r < 4; ++r) {
        const std::uint32_t s2 = g * 8 + col, j = g / 3;
        std::uint8_t* tile = cl.w.host + ((std::size_t{s2} * (tk + 1) + tk) * 4 + r) * kQuarterBytes;
        auto* tp = reinterpret_cast<std::uint16_t*>(tile + kConvTapOffset);
        for (std::uint32_t t = 0; t < taps; ++t)
          for (std::uint32_t cbk = 0; cbk < 4; ++cbk)
            for (std::uint32_t i = 0; i < 8; ++i) {
              const std::uint32_t chan = ((j * 8 + col) * 4 + r) * 32 + cbk * 8 + i;
              tp[(t * 4 + cbk) * 8 + i] = Bf16Bits(conv_w[std::size_t{chan} * taps + t]);
            }
      }
  cl.w.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  Attach(cl.w, cl.prog);
  return cl;
}

void NpuRuntime::ConvReset(ConvLayer& cl) {
  const std::uint32_t tk = cl.hidden / kTileK, groups = 3 * cl.hidden / kStrip / 8;
  for (std::uint32_t g = 1; g < groups; g += 3)
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t r = 0; r < 4; ++r)
        std::memset(cl.w.host + ((std::size_t{g * 8 + col} * (tk + 1) + tk) * 4 + r) * kQuarterBytes +
                        kConvHistOffset, 0, 4 * 6 * 16);
}

void NpuRuntime::ConvStep(ConvLayer& cl, const float* z, std::uint32_t n, float* y,
                          Timings* t) {
  auto t0 = Clock::now();
  EncodeA(z, n, cl.hidden, cl.prog);
  if (t) t->enc += Ms(t0);
  t0 = Clock::now();
  cl.prog.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (t) t->syncA += Ms(t0);
  t0 = Clock::now();
  cl.w.run.start();
  if (t) t->submit += Ms(t0);
  t0 = Clock::now();
  for (;;) {
    const auto st = cl.w.run.state();
    if (st == ERT_CMD_STATE_COMPLETED)
      break;
    if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
        st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED)
      break;
  }
  cl.w.run.wait();
  if (t) t->wait += Ms(t0);
  t0 = Clock::now();
  cl.prog.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (t) t->syncC += Ms(t0);
  t0 = Clock::now();
  DecodeC(y, n, cl.prog, cl.prog.ch + std::size_t{3 * cl.hidden / kStrip} * kStripBytes, cl.hidden);
  if (t) t->dec += Ms(t0);
  if (t) ++t->calls;
  cl.last_n = n;
}

void NpuRuntime::ConvCommit(ConvLayer& cl, std::uint32_t k) {
  const std::uint32_t tk = cl.hidden / kTileK, groups = 3 * cl.hidden / kStrip / 8;
  const std::uint32_t hist = cl.taps - 1;
  for (std::uint32_t g = 1; g < groups; g += 3) {
    const std::uint32_t j = g / 3;
    for (std::uint32_t col = 0; col < 8; ++col)
      for (std::uint32_t r = 0; r < 4; ++r) {
        const std::uint32_t s2 = g * 8 + col;
        auto* dst = reinterpret_cast<std::uint16_t*>(
            cl.w.host + ((std::size_t{s2} * (tk + 1) + tk) * 4 + r) * kQuarterBytes + kConvHistOffset);
        const auto* src = reinterpret_cast<const std::uint16_t*>(
            cl.prog.ch + std::size_t{16 + j * 8 + col} * kStripBytes + std::size_t{r} * kRowBytes);
        for (std::uint32_t cbk = 0; cbk < 4; ++cbk) {
          std::uint16_t nh[6][8];
          for (std::uint32_t i = 0; i < hist; ++i) {
            const std::uint32_t idx = k + i;
            const std::uint16_t* row = idx < hist ? dst + (cbk * hist + idx) * 8
                                                  : src + cbk * 128 + (idx - hist) * 8;
            std::memcpy(nh[i], row, 16);
          }
          std::memcpy(dst + cbk * hist * 8, nh, hist * 16);
        }
      }
    for (std::uint32_t col = 0; col < 8; ++col)
      SyncToDevice(cl.w.bo, (std::size_t{(g * 8 + col)} * (tk + 1) + tk) * 4 * kQuarterBytes,
                   4 * kQuarterBytes);
  }
}

namespace {
constexpr std::uint32_t kAttnHd = 128, kAttnKvh = 4, kAttnChunkBytes = 3 * 4 * kQuarterBytes;
constexpr std::uint32_t kAttnQElems = 32, kAttnOutElems = 34, kAttnTailElems = 36;
constexpr std::size_t kAttnAElems = 1 + kAttnQElems + kAttnTailElems;

// Byte offset of piece `byte` (0..4607) of chunk q for kv head r inside the BO.
inline std::size_t AttnOff(std::uint32_t q, std::uint32_t r, std::uint32_t byte) {
  return std::size_t{q} * kAttnChunkBytes + std::size_t{byte / kQuarterBytes} * 4 * kQuarterBytes +
         std::size_t{r} * kQuarterBytes + byte % kQuarterBytes;
}
inline void AttnLoadChunk(const std::uint8_t* base, std::uint32_t q, std::uint32_t r, std::uint8_t* chunk) {
  for (std::uint32_t j = 0; j < 3; ++j)
    std::memcpy(chunk + j * kQuarterBytes, base + AttnOff(q, r, j * kQuarterBytes), kQuarterBytes);
}
inline void AttnStoreChunk(std::uint8_t* base, std::uint32_t q, std::uint32_t r, const std::uint8_t* chunk) {
  for (std::uint32_t j = 0; j < 3; ++j)
    std::memcpy(base + AttnOff(q, r, j * kQuarterBytes), chunk + j * kQuarterBytes, kQuarterBytes);
}
}  // namespace

AttnLayer NpuRuntime::MakeAttnLayer(std::uint32_t capacity_tokens) {
  AttnLayer al;
  std::uint32_t mc = 1;
  const std::uint32_t qch = (capacity_tokens + 15) / 16;
  while (mc * 8 < qch) mc *= 2;
  al.capacity_chunks = mc * 8;
  const std::size_t bytes = std::size_t{al.capacity_chunks} * kAttnChunkBytes;
  al.kv = xrt::bo(dev_, bytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
  al.host = al.kv.map<std::uint8_t*>();
  std::memset(al.host, 0, bytes);
  al.kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  al.kmax2.assign(kAttnKvh, 0.f);
  if (!attn_a_) {
    attn_a_ = xrt::bo(dev_, kAttnAElems * 1152, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
    attn_c_ = xrt::bo(dev_, std::size_t{8} * kAttnOutElems * 4 * kRowBytes, XRT_BO_FLAGS_HOST_ONLY, kern_.group_id(3));
    attn_ah_ = attn_a_.map<std::uint8_t*>();
    attn_ch_ = attn_c_.map<std::uint8_t*>();
    std::memset(attn_ah_, 0, kAttnAElems * 1152);
  }
  return al;
}

void NpuRuntime::AttnReset(AttnLayer& al) {
  std::memset(al.host, 0, std::size_t{al.capacity_chunks} * kAttnChunkBytes);
  al.kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::fill(al.kmax2.begin(), al.kmax2.end(), 0.f);
}

void NpuRuntime::AttnAppend(AttnLayer& al, std::uint32_t pos, std::uint32_t n, const float* k,
                            const std::uint16_t* v_hist, Timings* tm) {
  const auto t_start = Clock::now();
  const std::uint32_t kvd = kAttnKvh * kAttnHd;
  const std::uint32_t q0 = pos / 16, q1 = (pos + n - 1) / 16;
  for (std::uint32_t r = 0; r < kAttnKvh; ++r) {
    for (std::uint32_t t = 0; t < n; ++t) {
      float s2 = 0;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) s2 += k[std::size_t{t} * kvd + r * kAttnHd + d] * k[std::size_t{t} * kvd + r * kAttnHd + d];
      al.kmax2[r] = std::max(al.kmax2[r], s2);
    }
    for (std::uint32_t q = q0; q <= q1; ++q) {
      alignas(64) std::uint8_t chunk[4608];
      AttnLoadChunk(al.host, q, r, chunk);
      for (std::uint32_t tt = 0; tt < 16; ++tt) {
        const std::uint32_t tok = q * 16 + tt;
        if (tok < pos || tok >= pos + n) continue;
        const std::uint32_t kb = tt / 8, i8 = tt % 8;
        for (std::uint32_t ks = 0; ks < 16; ++ks) {
          float v8[8];
          std::memcpy(v8, k + std::size_t{tok - pos} * kvd + r * kAttnHd + ks * 8, 32);
          EncGroup(v8, chunk + (std::size_t{kb} * 16 + ks) * 8 * 9 + i8 * 9);
        }
      }
      for (std::uint32_t kb = 0; kb < 2; ++kb) {
        const std::uint32_t t0 = q * 16 + kb * 8;
        if (t0 + 8 <= pos || t0 >= pos + n) continue;  // block untouched
        for (std::uint32_t db = 0; db < 16; ++db)
          for (std::uint32_t g = 0; g < 8; ++g) {
            float v8[8];
            for (std::uint32_t i8 = 0; i8 < 8; ++i8) {
              const std::uint32_t tok = t0 + i8;
              v8[i8] = tok < pos + n ? Bf16Float(v_hist[std::size_t{tok} * kvd + r * kAttnHd + db * 8 + g]) : 0.f;
            }
            EncGroup(v8, chunk + 2304 + ((std::size_t{kb} * 16 + db) * 8 + g) * 9);
          }
      }
      AttnStoreChunk(al.host, q, r, chunk);
    }
  }
  SyncToDevice(al.kv, std::size_t{q0} * kAttnChunkBytes, std::size_t{q1 - q0 + 1} * kAttnChunkBytes);
  if (tm) tm->attn_append += Ms(t_start);
}

bool NpuRuntime::AttnRun(AttnLayer& al, std::uint32_t pos, std::uint32_t n, const float* q,
                         const std::uint16_t* k_hist, const std::uint16_t* v_hist,
                         const std::string& programs_dir, float* out, Timings* tm) {
  auto t_start = Clock::now();
  const std::uint32_t kvd = kAttnKvh * kAttnHd, qd = 16 * kAttnHd;
  const float qscale = 1.4426950408889634f / std::sqrt(static_cast<float>(kAttnHd));
  const std::uint32_t total = pos + n;
  std::uint32_t mc = 1;
  while (mc * 8 * 16 < total) mc *= 2;
  if (mc * 8 > al.capacity_chunks) return false;
  // header: pos, n, then M per (kv head, row)
  auto* hdr = attn_ah_;
  std::memset(hdr, 0, 1152);
  const std::int32_t ih[2] = {static_cast<std::int32_t>(pos), static_cast<std::int32_t>(n)};
  std::memcpy(hdr, ih, 8);
  auto* mrow = reinterpret_cast<float*>(hdr + 128);
#pragma omp parallel for schedule(static, 1)
  for (std::int32_t rr = 0; rr < static_cast<std::int32_t>(kAttnKvh); ++rr) {
    const std::uint32_t r = static_cast<std::uint32_t>(rr);
    std::vector<float> qs(std::size_t{64} * kAttnHd);
    const float kbound = std::sqrt(al.kmax2[r]);
    // Q block for this kv head: rows i = hq*16 + t (hq = 0..3), zero for t >= n
    std::fill(qs.begin(), qs.end(), 0.f);
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t hq = i / 16, t = i % 16;
      if (t >= n) continue;
      const float* src = q + std::size_t{t} * qd + (r * 4 + hq) * kAttnHd;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) qs[std::size_t{i} * kAttnHd + d] = src[d] * qscale;
    }
    std::uint8_t* qb = attn_ah_ + std::size_t{1 + r * 8} * 1152;
    for (std::uint32_t rb = 0; rb < 8; ++rb)
      for (std::uint32_t ks = 0; ks < 16; ++ks)
        for (std::uint32_t rib = 0; rib < 8; ++rib) {
          const float* src = &qs[std::size_t{rb * 8 + rib} * kAttnHd + ks * 8];
          EncGroup(src, qb + (std::size_t{rb * 16 + ks} * 8 + rib) * 9);
        }
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t hq = i / 16, t = i % 16;
      float qn = 0;
      for (std::uint32_t d = 0; d < kAttnHd; ++d) qn += qs[std::size_t{i} * kAttnHd + d] * qs[std::size_t{i} * kAttnHd + d];
      const float bound = std::sqrt(qn) * kbound;
      float m = -1e30f;
      if (t < n) {
        const std::uint32_t last = pos + t;  // keys 0..last valid for this row
        const std::uint32_t lo = last >= 32 ? last - 31 : 0;
        auto score = [&](std::uint32_t tok) {
          float s = 0;
          const std::uint16_t* kr = k_hist + std::size_t{tok} * kvd + r * kAttnHd;
          for (std::uint32_t d = 0; d < kAttnHd; ++d) s += qs[std::size_t{i} * kAttnHd + d] * Bf16Float(kr[d]);
          return s;
        };
        for (std::uint32_t tok = lo; tok <= last; ++tok) m = std::max(m, score(tok));
        for (std::uint32_t tok = 0; tok < std::min(8u, lo); ++tok) m = std::max(m, score(tok));
        m = std::min(std::max(m, bound - 100.f), bound);
      } else {
        m = 0.f;
      }
      mrow[r * 64 + i] = m;
      (void)hq;
    }
  }
  {
    // Tail: exact bf16 V (and the cached K groups) for the 8-token blocks the rows live in.
    const std::uint32_t t0 = pos & ~7u;
    for (std::uint32_t r = 0; r < kAttnKvh; ++r) {
      alignas(64) std::uint8_t chunk[4608];
      for (std::uint32_t tb = 0; tb < 3; ++tb) {
        std::uint8_t* dst = attn_ah_ + (std::size_t{1 + kAttnQElems} + (tb * 4 + r) * 3) * 1152;
        std::memset(dst, 0, 3 * 1152);
        const std::uint32_t tokbase = t0 + tb * 8;
        if (tokbase >= total) continue;
        AttnLoadChunk(al.host, tokbase / 16, r, chunk);
        const std::uint32_t kb = (tokbase % 16) / 8;
        for (std::uint32_t ks = 0; ks < 16; ++ks)
          for (std::uint32_t tt = 0; tt < 8; ++tt)
            if (tokbase + tt < total)
              std::memcpy(dst + (ks * 8 + tt) * 9, chunk + (kb * 16 + ks) * 72 + tt * 9, 9);
        auto* vd = reinterpret_cast<std::uint16_t*>(dst + 1152);
        for (std::uint32_t db = 0; db < 16; ++db)
          for (std::uint32_t tt = 0; tt < 8; ++tt)
            if (tokbase + tt < total)
              std::memcpy(vd + (db * 8 + tt) * 8,
                          v_hist + std::size_t{tokbase + tt} * kvd + r * kAttnHd + db * 8, 16);
      }
    }
  }
  attn_a_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto it = attn_insts_.find(mc);
  if (it == attn_insts_.end()) {
    const std::string path = programs_dir + "/insts_attn_" + std::to_string(mc) + ".txt";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    xrt::bo bo(dev_, sz, XCL_BO_FLAGS_CACHEABLE, kern_.group_id(1));
    std::memcpy(bo.map<void*>(), buf.data(), sz);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    it = attn_insts_.emplace(mc, std::make_pair(bo, sz / 4)).first;
  }
  auto rit = al.runs.find(mc);
  if (rit == al.runs.end()) {
    xrt::run run(kern_);
    run.set_arg(0, 3u);
    run.set_arg(1, it->second.first);
    run.set_arg(2, static_cast<unsigned>(it->second.second));
    run.set_arg(3, attn_a_);
    run.set_arg(4, al.kv);
    run.set_arg(5, attn_c_);
    rit = al.runs.emplace(mc, std::move(run)).first;
  }
  if (tm) { tm->attn_prep += Ms(t_start); t_start = Clock::now(); }
  rit->second.start();
  for (;;) {
    const auto st = rit->second.state();
    if (st == ERT_CMD_STATE_COMPLETED) break;
    if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED && st != ERT_CMD_STATE_RUNNING &&
        st != ERT_CMD_STATE_SUBMITTED)
      break;
  }
  rit->second.wait();
  attn_c_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (tm) { tm->attn_wait += Ms(t_start); t_start = Clock::now(); }
  // merge: partials of all columns share M, so they add
  bool ok = true;
  alignas(64) static thread_local std::vector<float> tot;
  for (std::uint32_t r = 0; r < kAttnKvh; ++r) {
    tot.assign(std::size_t{kAttnOutElems} * 256, 0.f);
    for (std::uint32_t c = 0; c < 8; ++c)
      for (std::uint32_t i = 0; i < kAttnOutElems; ++i) {
        const auto* e = reinterpret_cast<const float*>(attn_ch_ + ((std::size_t{c} * kAttnOutElems + i) * 4 + r) * kRowBytes);
        float* d = &tot[std::size_t{i} * 256];
        for (int x = 0; x < 256; x += 16)
          _mm512_storeu_ps(d + x, _mm512_add_ps(_mm512_loadu_ps(d + x), _mm512_loadu_ps(e + x)));
      }
    const float* o = tot.data();
    const float* lt = tot.data() + std::size_t{32} * 256;
    for (std::uint32_t i = 0; i < 64; ++i) {
      const std::uint32_t hq = i / 16, t = i % 16;
      if (t >= n) continue;
      const std::uint32_t rb = i / 8, rib = i % 8;
      float l = 0;
      for (int cc = 0; cc < 8; ++cc) l += lt[rb * 64 + rib * 8 + cc];
      if (!(l > 1e-30f) || !(l < 1e30f)) { ok = false; continue; }
      const float inv = 1.0f / l;
      for (std::uint32_t db = 0; db < 16; ++db)
        for (int cc = 0; cc < 8; ++cc)
          out[std::size_t{t} * qd + (r * 4 + hq) * kAttnHd + db * 8 + cc] =
              o[(std::size_t{rb} * 16 + db) * 64 + rib * 8 + cc] * inv;
    }
  }
  (void)kAttnQElems;
  if (tm) { tm->attn_merge += Ms(t_start); ++tm->attn_calls; }
  return ok;
}

}  // namespace gufo::npu::runtime
