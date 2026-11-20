// Cost of handing work from a GPU graph to a CPU-side worker (the NPU feeder)
// and waiting for its answer, per layer, on gfx1151. Two ways to do it inside
// a captured graph of 48 "layers":
//
//   doorbell : a one-thread kernel stores a sequence number to pinned memory,
//              a CPU thread polling it answers, a one-thread kernel spins on
//              the answer (bounded). No host nodes in the graph.
//   hostnode : hipLaunchHostFunc nodes for the submit and for the wait.
//
// Each layer also runs `filler` ms of GPU work between submit and wait, as the
// routed experts would, so the overlap and the added latency show up together.
//
//   tools/bench/build.sh gpu_cpu_signal_bench && /tmp/gpu_cpu_signal_bench
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

#define CHECK(x)                                                        \
  do {                                                                  \
    hipError_t e_ = (x);                                                \
    if (e_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,            \
                   hipGetErrorString(e_));                              \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

constexpr int kLayers = 48;
using Clock = std::chrono::steady_clock;

__global__ void Doorbell(uint32_t* bell, uint32_t* counter) {
  // One increment per replay layer: the counter lives on the host side of the
  // graph, so each layer rings with a fresh value.
  const uint32_t next = __atomic_load_n(counter, __ATOMIC_RELAXED) + 1;
  __atomic_store_n(counter, next, __ATOMIC_RELAXED);
  __threadfence_system();
  __atomic_store_n(bell, next, __ATOMIC_RELEASE);
}

__global__ void Await(const uint32_t* done, const uint32_t* counter,
                      uint32_t* timeouts) {
  const uint32_t want = __atomic_load_n(counter, __ATOMIC_RELAXED);
  for (int spins = 0; spins < 20000000; ++spins) {
    if (__atomic_load_n(done, __ATOMIC_ACQUIRE) >= want) return;
    __builtin_amdgcn_s_sleep(1);
  }
  atomicAdd(timeouts, 1u);
}

// Stand-in for the routed experts: a kernel that keeps the GPU busy.
__global__ void Filler(float* x, int iters) {
  float v = x[threadIdx.x + blockIdx.x * blockDim.x];
  for (int i = 0; i < iters; ++i) v = v * 1.0000001f + 0.5f;
  x[threadIdx.x + blockIdx.x * blockDim.x] = v;
}

struct HostLink {
  std::atomic<uint32_t> submitted{0}, completed{0};
  std::mutex m;
  std::condition_variable cv_work, cv_done;
  double npu_us{0};
};
HostLink g_link;

static void HostSubmit(void*) {
  std::lock_guard<std::mutex> l(g_link.m);
  g_link.submitted.fetch_add(1);
  g_link.cv_work.notify_one();
}
static void HostWait(void*) {
  std::unique_lock<std::mutex> l(g_link.m);
  const uint32_t want = g_link.submitted.load();
  g_link.cv_done.wait(l, [&] { return g_link.completed.load() >= want; });
}

int main(int argc, char** argv) {
  const double npu_us = argc > 1 ? std::atof(argv[1]) : 220.0;  // NPU run time
  const int filler_iters = argc > 2 ? std::atoi(argv[2]) : 20000;

  uint32_t *bell, *done, *counter, *timeouts;
  CHECK(hipHostMalloc(&bell, 4, hipHostMallocCoherent | hipHostMallocMapped));
  CHECK(hipHostMalloc(&done, 4, hipHostMallocCoherent | hipHostMallocMapped));
  CHECK(hipMalloc(&counter, 4));
  CHECK(hipMalloc(&timeouts, 4));
  float* x;
  CHECK(hipMalloc(&x, 256 * 4 * 256));
  CHECK(hipMemset(x, 0, 256 * 4 * 256));
  *bell = 0;
  *done = 0;
  CHECK(hipMemset(counter, 0, 4));
  CHECK(hipMemset(timeouts, 0, 4));
  uint32_t *bell_d, *done_d;
  CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&bell_d), bell, 0));
  CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&done_d), done, 0));

  // CPU feeder: answers each bell after `npu_us` (the NPU run), spin-polling.
  std::atomic<bool> stop{false};
  std::thread feeder([&] {
    uint32_t seen = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      const uint32_t b = __atomic_load_n(bell, __ATOMIC_ACQUIRE);
      if (b != seen) {
        seen = b;
        const auto t0 = Clock::now();
        while (std::chrono::duration<double, std::micro>(Clock::now() - t0).count() < npu_us) {
        }
        __atomic_store_n(done, b, __ATOMIC_RELEASE);
      }
    }
  });
  // Host-node worker: same role for the hostnode variant.
  std::thread worker([&] {
    uint32_t seen = 0;
    while (!stop.load()) {
      std::unique_lock<std::mutex> l(g_link.m);
      if (!g_link.cv_work.wait_for(l, std::chrono::milliseconds(1), [&] {
            return g_link.submitted.load() != seen;
          })) {
        continue;
      }
      seen = g_link.submitted.load();
      l.unlock();
      const auto t0 = Clock::now();
      while (std::chrono::duration<double, std::micro>(Clock::now() - t0).count() < npu_us) {
      }
      l.lock();
      g_link.completed.store(seen);
      g_link.cv_done.notify_all();
    }
  });

  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  auto build = [&](int mode, hipGraphExec_t* exec) {
    hipGraph_t g;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int l = 0; l < kLayers; ++l) {
      if (mode == 1) Doorbell<<<1, 1, 0, s>>>(bell_d, counter);
      if (mode == 2) CHECK(hipLaunchHostFunc(s, HostSubmit, nullptr));
      if (mode != 3) {  // mode 3: no NPU at all, filler only
      }
      Filler<<<256, 256, 0, s>>>(x, filler_iters);
      if (mode == 1) Await<<<1, 1, 0, s>>>(done_d, counter, timeouts);
      if (mode == 2) CHECK(hipLaunchHostFunc(s, HostWait, nullptr));
    }
    CHECK(hipStreamEndCapture(s, &g));
    CHECK(hipGraphInstantiate(exec, g, nullptr, nullptr, 0));
    CHECK(hipGraphDestroy(g));
  };
  const char* names[3] = {"filler only (no NPU hand-off)", "doorbell kernels", "host callback nodes"};
  hipGraphExec_t exec[3];
  build(3, &exec[0]);
  build(1, &exec[1]);
  build(2, &exec[2]);

  std::printf("%d layers, filler per layer ~%d iterations, NPU run %.0f us\n\n", kLayers,
              filler_iters, npu_us);
  double base_ms = 0;
  for (int m = 0; m < 3; ++m) {
    // reset the doorbell protocol before each variant
    CHECK(hipStreamSynchronize(s));
    CHECK(hipMemset(counter, 0, 4));
    *bell = 0;
    *done = 0;
    g_link.submitted = 0;
    g_link.completed = 0;
    for (int w = 0; w < 5; ++w) {
      CHECK(hipGraphLaunch(exec[m], s));
      CHECK(hipStreamSynchronize(s));
    }
    double best = 1e9;
    for (int r = 0; r < 20; ++r) {
      const auto t0 = Clock::now();
      CHECK(hipGraphLaunch(exec[m], s));
      CHECK(hipStreamSynchronize(s));
      best = std::fmin(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    if (m == 0) base_ms = best;
    uint32_t to = 0;
    CHECK(hipMemcpy(&to, timeouts, 4, hipMemcpyDeviceToHost));
    std::printf("  %-32s %8.3f ms per token-graph  (+%.0f us per layer vs filler only)%s\n",
                names[m], best, (best - base_ms) * 1000.0 / kLayers, to ? "  [TIMEOUTS]" : "");
  }
  stop = true;
  feeder.join();
  worker.join();
  return 0;
}
