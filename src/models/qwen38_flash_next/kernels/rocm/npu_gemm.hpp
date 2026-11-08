#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace gufo::models::qwen38_flash_next::rocm {

/// Block-float GEMM on the XDNA2 NPU for one fixed shape (tokens x k x n).
/// Operands live in pinned host buffers that the GPU writes and reads directly
/// and the NPU maps (hipHostMalloc memory imported into XRT), so staging needs
/// no copies. One hardware context serves every call: switching between
/// kernels costs milliseconds on this part.
///
/// GUFO_NPU_ISOLATE=1 keeps the two devices on separate pages instead: the NPU
/// runs on private XRT buffers and the CPU copies between them and the
/// GPU-visible ones around each run.
///
/// Work is ordered on a HIP stream: Submit enqueues the NPU runs from a host
/// callback (they proceed while the stream's later GPU kernels run) and Wait
/// blocks the stream, from a second callback, until they finish. At most four
/// runs are in flight at once, the depth the driver is specified for.
class NpuGemm {
public:
  struct Config {
    std::string xclbin;
    std::string insts;
    std::uint32_t tokens{4096};
    std::uint32_t k{2560};
    std::uint32_t n{1024};
  };

  static std::unique_ptr<NpuGemm> Create(const Config& config,
                                         std::string* error);
  ~NpuGemm();
  NpuGemm(const NpuGemm&) = delete;
  NpuGemm& operator=(const NpuGemm&) = delete;

  /// Bytes of one staged operand: A (tokens x k), B (k x n) or C (tokens x n).
  std::size_t ABytes() const noexcept;
  std::size_t BBytes() const noexcept;
  std::size_t CBytes() const noexcept;

  /// Allocates a pinned buffer of `bytes` visible to the GPU; returns its id,
  /// or -1. Host(id) is the CPU view and Device(id) the GPU pointer.
  int NewBuffer(std::size_t bytes, std::string* error);
  void* Host(int id) const;
  void* Device(int id) const;

  /// Call once after a weight buffer has been written (GPU finished): in
  /// isolate mode this copies it to the NPU's private buffer.
  void FinalizeWeights(int id);

  /// Enqueues `count` runs reading A buffer a_id and B buffers b_ids[i],
  /// writing C buffers c_ids[i]. The runs execute in order on the NPU.
  bool Submit(hipStream_t stream, int a_id, const int* b_ids, const int* c_ids,
              int count, int program = 0);
  /// Loads another instruction stream for the same kernel (a runtime-K program
  /// serving a different reduction length) and returns its id for Submit, or
  /// -1. The A and B operand sizes for reduction length k are ABytes(k) and
  /// BBytes(k).
  int AddProgram(const std::string& insts, std::string* error);
  std::size_t ABytes(std::uint32_t k) const noexcept;
  std::size_t BBytes(std::uint32_t k) const noexcept;
  /// Blocks the stream until every submitted run has finished and its C buffer
  /// is readable by the GPU.
  bool Wait(hipStream_t stream);
  /// Blocks the stream until the first `ticket` submissions have finished.
  bool Wait(hipStream_t stream, std::uint64_t ticket);
  /// Number of submissions made so far; pass to Wait to wait for them only.
  std::uint64_t Ticket() const noexcept;

  /// The GPU's view of C buffer `id`, to read once Wait has returned. A single
  /// pass over the pinned pages is cheaper than copying them first.
  void* Staged(hipStream_t stream, int id);

  /// False once a run timed out or failed: results since then are invalid.
  bool Healthy() const noexcept;
  /// Marks the NPU failed so later work stays on the GPU.
  void MarkFailed() noexcept;

  std::uint32_t tokens() const noexcept;
  std::uint32_t k() const noexcept;
  std::uint32_t n() const noexcept;

private:
  NpuGemm();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Hands the qwen executor the factory for the Qwen3.8-27B FFN NPU offload.
void RegisterQwenNpuFfn();
/// Hands the qwen executor the grouped-expert prefill for qwen35moe.
void RegisterQwenMoePrefill();

}  // namespace gufo::models::qwen38_flash_next::rocm
