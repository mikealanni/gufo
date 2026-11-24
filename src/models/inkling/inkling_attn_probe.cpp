// Inkling NPU attention probe: window, full or ring visit with relative bias
// and sink.
//   gufo-inkling-attn <xclbin> <insts> <MC> <pos0> [reps=3] [mode=window]
//                     [tau_alpha=0.1] [tau_floor=1024]
// Compares NPU output against the fp64 CPU reference from reference.cpp and
// reports the static-bound fallback rate.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int HD = 128, KVH = 8, QH = 24, HPK = 3, NREL = 16, EXTENT = 1024;
constexpr int NR = 128, WINDOW = 1024;
constexpr float SCALE = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float LOG2E = 1.4426950408889634f;
constexpr int COLS = 8, ROWS = 4, BQ = 1536, CB = 1024;
constexpr int Q_ELEMS = 6, OUT_ELEMS = 26;
constexpr size_t QUARTER = BQ;
constexpr size_t CHUNK = 3 * ROWS * QUARTER;

uint16_t Bf16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

void EncGroup(const float* v, uint8_t* out) {
  unsigned max_exp = 0;
  for (int i = 0; i < 8; ++i) {
    uint32_t u;
    std::memcpy(&u, &v[i], 4);
    max_exp = std::max(max_exp, (u >> 23) & 0xFFu);
  }
  uint32_t ib = max_exp >= 7 ? (260u - max_exp) << 23 : 0u;
  float inv;
  std::memcpy(&inv, &ib, 4);
  out[0] = static_cast<uint8_t>(max_exp);
  for (int i = 0; i < 8; ++i) {
    float q = std::fmin(127.0f, std::fmax(-128.0f, std::nearbyint(v[i] * inv)));
    out[1 + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
  }
}

void EncGroupF16(const float* v, uint16_t* out) {
  for (int i = 0; i < 8; ++i) {
    float x = v[i];
    uint32_t u;
    std::memcpy(&u, &x, 4);
    out[i] = static_cast<uint16_t>(u >> 16);
  }
}

void DecGroup(const uint8_t* in, float* out) {
  const float mult = std::ldexp(1.0f, static_cast<int>(in[0]) - 127) / 64.0f;
  for (int i = 0; i < 8; ++i)
    out[i] = static_cast<float>(static_cast<int8_t>(in[1 + i])) * mult;
}

float Bf16Float(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16U;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

template<typename T>
std::vector<T> LoadBin(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::printf("cannot open %s\n", path.c_str());
    std::exit(2);
  }
  const std::streamoff size = f.tellg();
  if (size != static_cast<std::streamoff>(count * sizeof(T))) {
    std::printf("%s: expected %zu bytes, got %lld\n", path.c_str(),
                count * sizeof(T), static_cast<long long>(size));
    std::exit(2);
  }
  f.seekg(0);
  std::vector<T> v(count);
  f.read(reinterpret_cast<char*>(v.data()), size);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::printf(
        "usage: gufo-inkling-attn <xclbin> <insts> <MC> <pos0> [reps=3] "
        "[mode=window|full|ring] [tau_alpha=0.1] [tau_floor=1024]\n");
    return 2;
  }
  const char* xclbin = argv[1];
  const char* insts = argv[2];
  const int mc = std::atoi(argv[3]);
  const int pos0 = std::atoi(argv[4]);
  const int reps = argc > 5 ? std::atoi(argv[5]) : 3;
  const char* mode = argc > 6 ? argv[6] : "window";
  const float tau_alpha =
      argc > 7 ? static_cast<float>(std::atof(argv[7])) : 0.1f;
  const float tau_floor =
      argc > 8 ? static_cast<float>(std::atof(argv[8])) : 1024.0f;
  const bool full = std::strcmp(mode, "full") == 0;
  const bool ring = std::strcmp(mode, "ring") == 0;
  if (!full && !ring && std::strcmp(mode, "window") != 0) {
    std::printf("unknown mode %s\n", mode);
    return 2;
  }
  const int L = pos0 + NR;
  const int key0 = full ? 0 : std::max(0, pos0 - WINDOW + 1) & ~15;
  const int keys_end = full ? L - 1 : std::min(L - 1, key0 + mc * 16 - 1);
  if (keys_end < L - 1) {
    std::printf("mc*16=%d too small for key0=%d L=%d\n", mc * 16, key0, L);
    return 2;
  }

  const char* dump_env = std::getenv("INKLING_ATTN_DUMP_DIR");
  const std::string dump_dir = dump_env ? dump_env : "";
  const int kcache = ring ? WINDOW : L;
  std::vector<float> Q, R, S, Bias;
  std::vector<uint16_t> K, V;
  if (dump_dir.empty()) {
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Q.resize(static_cast<size_t>(NR) * QH * HD);
    R.resize(static_cast<size_t>(NR) * QH * NREL);
    S.resize(static_cast<size_t>(NR) * QH);
    Bias.resize(static_cast<size_t>(EXTENT + 1) * NREL);
    K.resize(static_cast<size_t>(kcache) * KVH * HD);
    V.resize(static_cast<size_t>(kcache) * KVH * HD);
    for (auto& x : Q)
      x = nd(rng);
    for (auto& x : R)
      x = nd(rng) * 0.5f;
    for (auto& x : S)
      x = nd(rng) * 0.3f;
    for (auto& x : Bias)
      x = nd(rng) * 0.2f;
    for (auto& x : K) {
      float f = nd(rng);
      uint32_t u;
      std::memcpy(&u, &f, 4);
      x = static_cast<uint16_t>(u >> 16);
    }
    for (auto& x : V) {
      float f = nd(rng);
      uint32_t u;
      std::memcpy(&u, &f, 4);
      x = static_cast<uint16_t>(u >> 16);
    }
  } else {
    Q = LoadBin<float>(dump_dir + "/Q.bin", static_cast<size_t>(NR) * QH * HD);
    R = LoadBin<float>(dump_dir + "/R.bin",
                       static_cast<size_t>(NR) * QH * NREL);
    S = LoadBin<float>(dump_dir + "/S.bin", static_cast<size_t>(NR) * QH);
    Bias = LoadBin<float>(dump_dir + "/Bias.bin",
                          static_cast<size_t>(EXTENT + 1) * NREL);
    K = LoadBin<uint16_t>(dump_dir + "/K.bin",
                          static_cast<size_t>(kcache) * KVH * HD);
    V = LoadBin<uint16_t>(dump_dir + "/V.bin",
                          static_cast<size_t>(kcache) * KVH * HD);
  }

  auto tau = [&](int t) -> float {
    if (!full)
      return 1.0f;
    const float p1 = static_cast<float>(pos0 + t + 1);
    return 1.0f + tau_alpha * std::log(std::max(1.0f, p1 / tau_floor));
  };
  if (full && dump_dir.empty()) {
    for (int t = 0; t < NR; ++t) {
      const float ta = tau(t);
      for (int h = 0; h < QH; ++h) {
        for (int d = 0; d < HD; ++d)
          Q[((size_t)t * QH + h) * HD + d] *= ta;
        for (int i = 0; i < NREL; ++i)
          R[((size_t)t * QH + h) * NREL + i] *= ta;
      }
    }
  }

  std::vector<float> Qd(Q.size());
  for (int t = 0; t < NR; ++t)
    for (int h = 0; h < QH; ++h)
      for (int ks = 0; ks < HD / 8; ++ks) {
        float src[8], dec[8];
        uint8_t enc[9];
        for (int d = 0; d < 8; ++d)
          src[d] = Q[((size_t)t * QH + h) * HD + ks * 8 + d] * (LOG2E * SCALE);
        EncGroup(src, enc);
        DecGroup(enc, dec);
        for (int d = 0; d < 8; ++d)
          Qd[((size_t)t * QH + h) * HD + ks * 8 + d] = dec[d] / (LOG2E * SCALE);
      }

  std::vector<float> Kd(K.size()), Vd(V.size());
  for (int j = 0; j < kcache; ++j)
    for (int g = 0; g < KVH; ++g)
      for (int ks = 0; ks < HD / 8; ++ks) {
        float src[8], dec[8];
        uint8_t enc[9];
        for (int d = 0; d < 8; ++d)
          src[d] = Bf16Float(K[((size_t)j * KVH + g) * HD + ks * 8 + d]);
        EncGroup(src, enc);
        DecGroup(enc, dec);
        for (int d = 0; d < 8; ++d)
          Kd[((size_t)j * KVH + g) * HD + ks * 8 + d] = dec[d];
      }
  for (int tb = 0; tb < (kcache + 7) / 8; ++tb)
    for (int g = 0; g < KVH; ++g)
      for (int d = 0; d < HD; ++d) {
        float src[8], dec[8];
        uint8_t enc[9];
        for (int tt = 0; tt < 8; ++tt) {
          const int tok = tb * 8 + tt;
          src[tt] = tok < kcache
                        ? Bf16Float(V[((size_t)tok * KVH + g) * HD + d])
                        : 0.0f;
        }
        EncGroup(src, enc);
        DecGroup(enc, dec);
        for (int tt = 0; tt < 8; ++tt) {
          const int tok = tb * 8 + tt;
          if (tok < kcache)
            Vd[((size_t)tok * KVH + g) * HD + d] = dec[tt];
        }
      }

  auto kid = [&](int j) -> int { return ring ? j % WINDOW : j; };
  auto qf = [&](int t, int h, int d) -> float {
    return Qd[((size_t)t * QH + h) * HD + d];
  };
  auto rf = [&](int t, int h, int i) -> float {
    return R[((size_t)t * QH + h) * NREL + i];
  };
  auto sf = [&](int t, int h) -> float { return S[(size_t)t * QH + h]; };
  auto kf = [&](int j, int g, int d) -> float {
    return Kd[((size_t)kid(j) * KVH + g) * HD + d];
  };
  auto vf = [&](int j, int g, int d) -> float {
    return Vd[((size_t)kid(j) * KVH + g) * HD + d];
  };
  auto kraw = [&](int j, int g, int d) -> float {
    return Bf16Float(K[((size_t)kid(j) * KVH + g) * HD + d]);
  };
  auto vraw = [&](int j, int g, int d) -> float {
    return Bf16Float(V[((size_t)kid(j) * KVH + g) * HD + d]);
  };
  auto biasf = [&](int dist, int i) -> float {
    return dist >= 0 && dist < EXTENT ? Bias[(size_t)dist * NREL + i] : 0.0f;
  };

  // CPU reference (fp64 accumulation).
  std::vector<float> Ref(static_cast<size_t>(NR) * QH * HD, 0.0f);
  double kmax = 0.0, biasmax = 0.0;
  for (int j = 0; j < kcache; ++j)
    for (int g = 0; g < KVH; ++g)
      for (int d = 0; d < HD; ++d)
        kmax = std::max(kmax, static_cast<double>(std::fabs(kf(j, g, d))));
  for (float x : Bias)
    biasmax = std::max(biasmax, static_cast<double>(std::fabs(x)));
  long long bad_rows = 0, total_rows = 0;
  double max_bound_gap = 0.0;
  for (int t = 0; t < NR; ++t) {
    const int p = pos0 + t;
    const int first = full ? 0 : std::max(0, p - WINDOW + 1);
    for (int h = 0; h < QH; ++h) {
      const int g = h / HPK;
      std::vector<double> sc;
      double mx = 0.0;
      for (int j = first; j <= p; ++j) {
        double dot = 0.0;
        for (int d = 0; d < HD; ++d)
          dot += static_cast<double>(qf(t, h, d)) * kf(j, g, d);
        double b = 0.0;
        const int dist = p - j;
        if (dist < EXTENT)
          for (int i = 0; i < NREL; ++i)
            b += static_cast<double>(biasf(dist, i)) * rf(t, h, i);
        const double s = dot * SCALE + b - sf(t, h);
        sc.push_back(s);
        mx = std::max(mx, s);
      }
      double qn = 0.0, rn = 0.0;
      for (int d = 0; d < HD; ++d)
        qn += static_cast<double>(qf(t, h, d)) * qf(t, h, d);
      for (int i = 0; i < NREL; ++i)
        rn += std::fabs(static_cast<double>(rf(t, h, i)));
      qn = std::sqrt(qn);
      const double bound =
          std::log2(1.0 + qn * kmax * SCALE + rn * biasmax +
                    std::max(0.0, -static_cast<double>(sf(t, h))));
      const double gap = bound - mx * LOG2E;
      ++total_rows;
      if (gap < 0.0) {
        ++bad_rows;
        max_bound_gap = std::max(max_bound_gap, -gap);
      }

      double den = std::exp(-mx);  // zero sink entry
      for (double& x : sc) {
        x = std::exp(x - mx);
        den += x;
      }
      for (size_t z = 0; z < sc.size(); ++z) {
        const int j = first + static_cast<int>(z);
        const double w = sc[z] / den;
        for (int d = 0; d < HD; ++d)
          Ref[((size_t)t * QH + h) * HD + d] +=
              static_cast<float>(w * vf(j, g, d));
      }
    }
  }

  // Build A and B buffers per pass (4 kv heads each).
  const size_t prefix_units = static_cast<size_t>(8) + mc;
  const size_t PREFIX_COL = prefix_units * ROWS * QUARTER;
  const size_t c_bytes = static_cast<size_t>(8) * OUT_ELEMS * 4 * CB;
  size_t b_bytes = static_cast<size_t>(mc) * CHUNK;

  auto encode_chunk = [&](int g, int q, uint8_t* chunk) {
    std::memset(chunk, 0, 3 * QUARTER);
    for (int kb = 0; kb < 2; ++kb) {
      for (int ks = 0; ks < 16; ++ks) {
        for (int tt = 0; tt < 8; ++tt) {
          const int tok = key0 + q * 16 + kb * 8 + tt;
          if (tok >= L)
            continue;
          float v[8];
          for (int i = 0; i < 8; ++i)
            v[i] = kraw(tok, g, ks * 8 + i);
          EncGroup(v, chunk + ((kb * 16 + ks) * 8 + tt) * 9);
        }
      }
      for (int db = 0; db < 16; ++db) {
        for (int gg = 0; gg < 8; ++gg) {
          float v[8];
          for (int tt = 0; tt < 8; ++tt) {
            const int tok = key0 + q * 16 + kb * 8 + tt;
            v[tt] = tok < L ? vraw(tok, g, db * 8 + gg) : 0.0f;
          }
          EncGroup(v, chunk + 2304 + ((kb * 16 + db) * 8 + gg) * 9);
        }
      }
    }
  };

  int worst_pass = 0;
  double total_num = 0, total_den = 0, total_max = 0;
  double best = 1e18;

  xrt::device dev(0u);
  xrt::xclbin xb{std::string(xclbin)};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());

  std::ifstream f(insts, std::ios::binary | std::ios::ate);
  if (!f) {
    std::printf("cannot open %s\n", insts);
    return 2;
  }
  size_t isz = f.tellg();
  f.seekg(0);
  std::vector<char> iw(isz);
  f.read(iw.data(), isz);
  xrt::bo ib(dev, isz, XRT_BO_FLAGS_CACHEABLE, kern.group_id(1));
  std::memcpy(ib.map<void*>(), iw.data(), isz);
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (int pass = 0; pass < 2; ++pass) {
    std::vector<uint8_t> A(8 * PREFIX_COL, 0), B(b_bytes, 0);
    for (int g = 0; g < 4; ++g) {
      const int kv = pass * 4 + g;
      const int qbase = kv * HPK;
      for (int q = 0; q < mc; ++q) {
        uint8_t chunk[3 * QUARTER];
        encode_chunk(kv, q, chunk);
        for (int j = 0; j < 3; ++j)
          std::memcpy(&B[((size_t)q * 3 + j) * ROWS * QUARTER + g * QUARTER],
                      chunk + j * QUARTER, QUARTER);
      }
    }
    for (int c = 0; c < COLS; ++c) {
      const int pos = pos0 + c * 16;
      for (int g = 0; g < 4; ++g) {
        const int kv = pass * 4 + g;
        const int qbase = kv * HPK;
        uint8_t* core = &A[(size_t)c * PREFIX_COL + g * QUARTER];
        auto unit = [&](int u) {
          return core + static_cast<size_t>(u) * ROWS * QUARTER;
        };
        int32_t h[5] = {pos, 16, (L - 1) / 16, full ? 0 : WINDOW, key0};
        std::memcpy(unit(0), h, sizeof h);
        float* mrow = reinterpret_cast<float*>(unit(0) + 128);
        float* ilrow = reinterpret_cast<float*>(unit(0) + 128 + 192);
        float* srow = reinterpret_cast<float*>(unit(0) + 128 + 384);
        std::vector<uint8_t> qb(6 * QUARTER, 0);
        for (int rb = 0; rb < 6; ++rb) {
          for (int ks = 0; ks < 16; ++ks) {
            for (int rib = 0; rib < 8; ++rib) {
              const int i = rb * 8 + rib;
              const int head = qbase + i % HPK;
              const int t = c * 16 + i / HPK;
              float src[8];
              for (int d = 0; d < 8; ++d)
                src[d] = Q[((size_t)t * QH + head) * HD + ks * 8 + d] *
                         (LOG2E * SCALE);
              EncGroup(src, &qb[(rb * 16 + ks) * 8 * 9 + rib * 9]);
            }
          }
        }
        for (int u = 0; u < Q_ELEMS; ++u)
          std::memcpy(unit(1 + u), &qb[u * QUARTER], QUARTER);
        for (int i = 0; i < 48; ++i) {
          const int head = qbase + i % HPK;
          const int t = c * 16 + i / HPK;
          const int abs_t = pos0 + t;
          const int first = full ? 0 : std::max(0, abs_t - WINDOW + 1);
          double mx = 0.0;  // zero sink logit participates in the bound
          for (int j = first; j <= abs_t; ++j) {
            double dot = 0;
            for (int d = 0; d < HD; ++d)
              dot += static_cast<double>(qf(t, head, d)) * kf(j, kv, d);
            double b = 0;
            const int dist = abs_t - j;
            if (dist < EXTENT)
              for (int z = 0; z < NREL; ++z)
                b += static_cast<double>(biasf(dist, z)) * rf(t, head, z);
            mx = std::max(mx, dot * SCALE + b - sf(t, head));
          }
          mrow[i] = static_cast<float>(mx * LOG2E);
          ilrow[i] = static_cast<float>(std::exp2(-mx * LOG2E));
          srow[i] = static_cast<float>(sf(t, head) * LOG2E);
        }
        {
          std::vector<uint16_t> rb16(48 * 16);
          for (int i = 0; i < 48; ++i)
            for (int z = 0; z < NREL; ++z)
              rb16[i * 16 + z] = Bf16(rf(c * 16 + i / HPK, qbase + i % HPK, z));
          std::memcpy(unit(7), rb16.data(), rb16.size() * 2);
        }
        for (int q = 0; q < mc; ++q) {
          std::vector<uint16_t> sl(16 * 32, 0);
          const int d0 = pos0 + c * 16 - (key0 + q * 16);
          for (int z = 0; z < NREL; ++z)
            for (int j = 0; j < 31; ++j) {
              const int dist = d0 + 15 - j;
              if (dist >= 0 && dist < EXTENT)
                sl[z * 32 + j] = Bf16(Bias[(size_t)dist * NREL + z] * LOG2E);
            }
          std::memcpy(unit(8 + q), sl.data(), sl.size() * 2);
        }
      }
    }

    xrt::bo ba(dev, A.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    xrt::bo bb(dev, B.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    xrt::bo bc(dev, c_bytes, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    std::memcpy(ba.map<void*>(), A.data(), A.size());
    std::memcpy(bb.map<void*>(), B.data(), B.size());
    std::memset(bc.map<void*>(), 0, c_bytes);
    ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    for (int r = 0; r < reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      auto run = kern(3, ib, static_cast<unsigned>(isz / 4), ba, bb, bc);
      {
        auto tend = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
          auto st = run.state();
          if (st == ERT_CMD_STATE_COMPLETED)
            break;
          if (st != ERT_CMD_STATE_NEW && st != ERT_CMD_STATE_QUEUED &&
              st != ERT_CMD_STATE_RUNNING && st != ERT_CMD_STATE_SUBMITTED) {
            std::printf("run ended in state %d\n", static_cast<int>(st));
            return 3;
          }
          if (std::chrono::steady_clock::now() > tend) {
            std::printf("timeout state %d\n", static_cast<int>(st));
            return 4;
          }
        }
      }
      best = std::min(best, std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count());
    }
    bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint8_t* C = bc.map<uint8_t*>();

    double num = 0, den = 0, mx = 0;
    const size_t cpc = static_cast<size_t>(OUT_ELEMS) * ROWS * CB;
    for (int c = 0; c < COLS; ++c) {
      for (int g = 0; g < 4; ++g) {
        std::vector<double> O(static_cast<size_t>(48) * HD, 0.0), Ls(48, 0.0);
        for (int i = 0; i < OUT_ELEMS; ++i) {
          const float* e = reinterpret_cast<const float*>(
              C + c * cpc + static_cast<size_t>(i) * ROWS * CB +
              static_cast<size_t>(g) * CB);
          if (i < 24) {
            for (int k = 0; k < 256; ++k) {
              const int tile = i * 4 + k / 64, lane = k % 64, rb = tile / 16,
                        db = tile % 16;
              O[(static_cast<size_t>(rb) * 8 + lane / 8) * HD + db * 8 +
                lane % 8] += e[k];
            }
          } else {
            for (int k = 0; k < 256; ++k) {
              const int tile = (i - 24) * 4 + k / 64, lane = k % 64;
              if (tile < 6)
                Ls[tile * 8 + lane / 8] += e[k];
            }
          }
        }
        for (int i = 0; i < 48; ++i) {
          const int head = (pass * 4 + g) * HPK + i % HPK;
          const int t = c * 16 + i / HPK;
          for (int d = 0; d < HD; ++d) {
            const double got = Ls[i] > 0 ? O[(size_t)i * HD + d] / Ls[i] : 0.0;
            const double ref = Ref[((size_t)t * QH + head) * HD + d];
            const double diff = got - ref;
            num += diff * diff;
            den += ref * ref;
            mx = std::max(mx, std::fabs(diff));
          }
        }
      }
    }
    std::printf("pass %d: rel %.5f max %.5f\n", pass, std::sqrt(num / den), mx);
    total_num += num;
    total_den += den;
    total_max = std::max(total_max, mx);
  }

  std::printf(
      "static-bound fallback %lld/%lld (%.3f%%) max gap %.3f log2\n", bad_rows,
      total_rows,
      100.0 * static_cast<double>(bad_rows) / static_cast<double>(total_rows),
      max_bound_gap);
  std::printf(
      "inkling attn mode=%s data=%s pos0=%d L=%d key0=%d mc=%d: rel %.5f "
      "maxabs %.5f best %.4f ms\n",
      mode, dump_dir.empty() ? "random" : dump_dir.c_str(), pos0, L, key0, mc,
      std::sqrt(total_num / total_den), total_max, best);
  return 0;
}
