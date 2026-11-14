// Trace probe: sbtest-trace <xclbin> <insts> <c_bytes_before_trace> <trace_bytes> <out_file> [a_bytes b_bytes]
// Runs the program a few times (warm) and writes the trace words found after c_bytes_before_trace in the C buffer.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
  const size_t cb0 = strtoul(argv[3], 0, 0), tb = strtoul(argv[4], 0, 0);
  const size_t ab = argc > 7 ? strtoul(argv[7], 0, 0) : 1 << 20, bb = argc > 8 ? strtoul(argv[8], 0, 0) : 64 << 20;
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
      bc(dev, cb0 + tb, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  std::memset(ba.map<void*>(), 0, ab);
  std::memset(bbo.map<void*>(), 0, bb);
  std::memset(bc.map<void*>(), 0, cb0 + tb);
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  for (int r = 0; r < 3; ++r) {
    std::memset(bc.map<uint8_t*>() + cb0, 0, tb);
    bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kern(3, ib, (unsigned)(isz / 4), ba, bbo, bc);
    run.wait();
  }
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint32_t* w = reinterpret_cast<const uint32_t*>(bc.map<uint8_t*>() + cb0);
  std::ofstream o(argv[5]);
  size_t n = 0;
  for (size_t i = 0; i < tb / 4; ++i)
    if (w[i]) { char b[16]; std::snprintf(b, sizeof b, "%08x\n", w[i]); o << b; ++n; }
  std::printf("trace words written: %zu\n", n);
  return 0;
}
