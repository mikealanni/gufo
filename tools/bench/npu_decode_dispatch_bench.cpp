// NPU dispatch latency and weight-streaming bandwidth for a decode-sized GEMM
// (M = 512 rows, K = 2560, N = 1024: one 2.95 MB block-float weight slice per
// run), without a model or a GPU. It answers whether the NPU could take part in
// decode: the per-run latency of one slice and the weight bytes per second it
// sustains with several runs in flight.
//
// The binary name must start with "gufo" so comfy-monitor holds the GPU fabric
// clock while it has the NPU open; it also refuses to run a job until the
// fabric clock is actually held.
//
//   g++ -O2 -std=c++17 -I<xrt include dir> tools/bench/npu_decode_dispatch_bench.cpp \
//       -o /tmp/gufo_npu_decode_bench -L/opt/xilinx/xrt/lib -lxrt_coreutil
//   LD_LIBRARY_PATH=/opt/xilinx/xrt/lib /tmp/gufo_npu_decode_bench <xclbin> <insts> [M]
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr unsigned kOpcode = 3;
constexpr std::size_t kBytesPer8 = 9;
constexpr std::size_t kK = 2560;
constexpr std::size_t kN = 1024;
constexpr int kWeightSlices = 64;  // 189 MB: beyond any cache
using Clock = std::chrono::steady_clock;

double Us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

std::string Read(const char* path) {
  std::ifstream f(path);
  std::string s;
  std::getline(f, s);
  return s;
}

bool FabricClockHeld() {
  const std::string base = "/sys/class/drm/card0/device/";
  if (Read((base + "power_dpm_force_performance_level").c_str()) != "manual") {
    return false;
  }
  std::ifstream f(base + "pp_dpm_fclk");
  std::string line, last_line, starred;
  while (std::getline(f, line)) {
    last_line = line;
    if (line.find('*') != std::string::npos) starred = line;
  }
  return !starred.empty() && starred == last_line;
}

double Percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <xclbin> <insts> [M=512]\n", argv[0]);
    return 2;
  }
  const std::size_t m = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 512;
  const std::size_t a_bytes = m * kK / 8 * kBytesPer8;
  const std::size_t b_bytes = kK * kN / 8 * kBytesPer8;
  const std::size_t c_bytes = m * kN / 8 * kBytesPer8;

  xrt::device device(0u);
  const xrt::xclbin xclbin{std::string(argv[1])};
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  xrt::kernel kernel(context, xclbin.get_kernels().at(0).get_name());
  std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
  const auto insts_size = static_cast<std::size_t>(file.tellg());
  file.seekg(0);
  std::vector<char> words(insts_size);
  file.read(words.data(), static_cast<std::streamsize>(insts_size));
  xrt::bo instructions(device, insts_size, XCL_BO_FLAGS_CACHEABLE,
                       kernel.group_id(1));
  std::memcpy(instructions.map<void*>(), words.data(), insts_size);
  instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const std::size_t instruction_words = insts_size / sizeof(int);

  std::printf("NPU open; waiting for the fabric clock to be held...\n");
  bool held = false;
  for (int i = 0; i < 300 && !(held = FabricClockHeld()); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (!held) {
    std::fprintf(stderr,
                 "fabric clock not held after 30 s: refusing to run (start "
                 "bench mode or check comfy-monitor)\n");
    return 3;
  }
  std::this_thread::sleep_for(std::chrono::seconds(4));  // monitor settle
  std::printf("fabric clock held. M=%zu: A %.2f MB, B %.2f MB x %d slices, C %.2f MB\n",
              m, a_bytes / 1e6, b_bytes / 1e6, kWeightSlices, c_bytes / 1e6);

  const auto group = kernel.group_id(3);
  std::mt19937 rng(1);
  auto make = [&](std::size_t bytes, bool fill) {
    xrt::bo bo(device, bytes, XRT_BO_FLAGS_HOST_ONLY, group);
    auto* p = bo.map<std::uint8_t*>();
    if (fill) {
      for (std::size_t i = 0; i < bytes; ++i) {
        // 9-byte groups: an exponent near 127, then eight mantissas.
        p[i] = (i % 9 == 0) ? static_cast<std::uint8_t>(120 + (rng() & 7))
                            : static_cast<std::uint8_t>(rng());
      }
    } else {
      std::memset(p, 0, bytes);
    }
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
  };
  xrt::bo a = make(a_bytes, true);
  std::vector<xrt::bo> b, c;
  for (int i = 0; i < kWeightSlices; ++i) b.push_back(make(b_bytes, true));
  for (int i = 0; i < 16; ++i) c.push_back(make(c_bytes, false));
  const auto timeout = std::chrono::milliseconds(2000);

  // 1. Latency of a single run, weights cold in rotation, nothing else queued.
  std::vector<double> submit_us, total_us;
  for (int i = 0; i < 300; ++i) {
    const auto t0 = Clock::now();
    auto run = kernel(kOpcode, instructions, instruction_words, a,
                      b[i % kWeightSlices], c[0]);
    const auto t1 = Clock::now();
    if (run.wait(timeout) != ERT_CMD_STATE_COMPLETED) {
      std::fprintf(stderr, "run timed out\n");
      return 4;
    }
    const auto t2 = Clock::now();
    if (i >= 20) {
      submit_us.push_back(Us(t0, t1));
      total_us.push_back(Us(t0, t2));
    }
  }
  std::printf("\nsingle run (A=%.2f MB in, B=%.2f MB weights, C=%.2f MB out):\n",
              a_bytes / 1e6, b_bytes / 1e6, c_bytes / 1e6);
  std::printf("  submit call  median %6.0f us\n", Percentile(submit_us, 0.5));
  std::printf("  submit+wait  min %6.0f  median %6.0f  p95 %6.0f us\n",
              Percentile(total_us, 0.0), Percentile(total_us, 0.5),
              Percentile(total_us, 0.95));

  // 2. Pipelined: depth runs in flight, a different weight slice per run.
  std::printf("\npipelined (distinct weight slice per run):\n");
  for (int depth : {1, 2, 4, 8}) {
    constexpr int kRuns = 400;
    std::deque<xrt::run> inflight;
    const auto t0 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
      if (static_cast<int>(inflight.size()) == depth) {
        if (inflight.front().wait(timeout) != ERT_CMD_STATE_COMPLETED) {
          std::fprintf(stderr, "run timed out\n");
          return 4;
        }
        inflight.pop_front();
      }
      inflight.push_back(kernel(kOpcode, instructions, instruction_words, a,
                                b[i % kWeightSlices], c[i % 16]));
    }
    for (auto& r : inflight) {
      if (r.wait(timeout) != ERT_CMD_STATE_COMPLETED) {
        std::fprintf(stderr, "run timed out\n");
        return 4;
      }
    }
    const double us = Us(t0, Clock::now());
    std::printf("  depth %d: %6.1f us/run  %6.2f GB/s of weights  (%.2f GB/s counting A and C)\n",
                depth, us / kRuns, b_bytes * kRuns / (us * 1e3),
                (a_bytes + b_bytes + c_bytes) * double(kRuns) / (us * 1e3));
  }
  return 0;
}
