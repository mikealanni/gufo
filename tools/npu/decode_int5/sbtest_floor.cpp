// Dispatch floor probe: sbtest-floor <xclbin> <insts> [reps=200] [a_bytes b_bytes c_bytes]
// Times the submit call (run.start) and the wait for completion separately, spin vs blocking wait.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using Clock = std::chrono::steady_clock;
static double Us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

int main(int argc, char** argv) {
  const int reps = argc > 3 ? atoi(argv[3]) : 200;
  const size_t ab = argc > 4 ? strtoul(argv[4], 0, 0) : 1 << 20, bb = argc > 5 ? strtoul(argv[5], 0, 0) : 8 << 20,
               cb = argc > 6 ? strtoul(argv[6], 0, 0) : 1 << 20;
  xrt::device dev(0u);
  xrt::xclbin xb{std::string(argv[1])};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, xb.get_kernels().at(0).get_name());
  std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
  const size_t isz = f.tellg();
  f.seekg(0);
  std::vector<char> iw(isz);
  f.read(iw.data(), isz);
  xrt::bo ib(dev, isz, XRT_BO_FLAGS_CACHEABLE, kern.group_id(1));
  std::memcpy(ib.map<void*>(), iw.data(), isz);
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo ba(dev, ab, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3)), bbo(dev, bb, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3)),
      bc(dev, cb, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memset(ba.map<void*>(), 0, ab);
  std::memset(bbo.map<void*>(), 0, bb);
  std::memset(bc.map<void*>(), 0, cb);
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::run run(kern);
  run.set_arg(0, 3u); run.set_arg(1, ib); run.set_arg(2, (unsigned)(isz / 4)); run.set_arg(3, ba); run.set_arg(4, bbo); run.set_arg(5, bc);
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<double> sub, comp, tot;
    for (int r = 0; r < reps + 10; ++r) {
      const auto t0 = Clock::now();
      run.start();
      const auto t1 = Clock::now();
      if (mode == 0) { while (run.state() != ERT_CMD_STATE_COMPLETED) {} } else run.wait();
      const auto t2 = Clock::now();
      if (r >= 10) { sub.push_back(Us(t0, t1)); comp.push_back(Us(t1, t2)); tot.push_back(Us(t0, t2)); }
    }
    std::sort(sub.begin(), sub.end()); std::sort(comp.begin(), comp.end()); std::sort(tot.begin(), tot.end());
    auto med = [](std::vector<double>& v) { return v[v.size() / 2]; };
    std::printf("%s: insts %zu B | start() %.1f us (min %.1f) | start->done %.1f us (min %.1f) | total median %.1f us min %.1f\n",
                mode == 0 ? "spin " : "wait ", isz, med(sub), sub.front(), med(comp), comp.front(), med(tot), tot.front());
  }
  return 0;
}
