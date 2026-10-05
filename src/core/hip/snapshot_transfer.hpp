#ifndef GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_
#define GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace gufo::hip {

/// Copies a frozen session on a separate stream. Callers have completed that
/// session's model operation before capture, and keep its state alive and
/// unchanged until all copies finish. Other sessions may continue executing.
/// Device-to-host copies land in a pinned staging buffer and are then copied
/// on the CPU, so the runtime never pins or stages a multi-gigabyte pageable
/// destination itself.
class SnapshotTransfer {
public:
  /// Optional bounds/tombstone check installed by a model that owns a device
  /// allocation registry. Null leaves copies unchecked, as before.
  using RangeCheck = void (*)(const char* site, const char* role,
                              const void* pointer, std::size_t bytes);
  /// Registers a pinned buffer with the same registry, so it is bounds-checked
  /// and reclaims any tombstone left by a recycled device address.
  using AllocRecord = void (*)(const char* site, const void* base,
                               std::size_t bytes);
  /// Drops a registered pinned buffer when it is freed.
  using AllocForget = void (*)(const void* base, const char* site);
  static std::mutex& DeviceMutex() noexcept {
    static std::mutex mutex;
    return mutex;
  }
  static void SetRangeCheck(RangeCheck check) noexcept { check_ = check; }
  static void SetAllocRecord(AllocRecord record) noexcept { record_ = record; }
  static void SetAllocForget(AllocForget forget) noexcept { forget_ = forget; }

  /// Streams and pinned staging buffers are pooled for the life of the
  /// process: after warm-up a snapshot creates and frees no device resources,
  /// which would otherwise race with graph replay on the decode thread and
  /// cost a pinned allocation per snapshot.
  SnapshotTransfer() : resources_(Acquire()) {}
  ~SnapshotTransfer() {
    (void)hipStreamSynchronize(resources_->stream);
    const std::lock_guard lock(PoolMutex());
    Pool().push_back(resources_);
  }
  SnapshotTransfer(const SnapshotTransfer&) = delete;
  SnapshotTransfer& operator=(const SnapshotTransfer&) = delete;

  void Copy(void* destination, const void* source, std::size_t bytes,
            hipMemcpyKind kind = hipMemcpyDeviceToHost) {
    if (kind != hipMemcpyDeviceToHost) {
      Guard("snapshot.Copy", "dst", destination, bytes);
      Guard("snapshot.Copy", "src", source, bytes);
      Check(hipMemcpyAsync(destination, source, bytes, kind, resources_->stream));
      Check(hipStreamSynchronize(resources_->stream));
      return;
    }
    EnsureStage();
    Guard("snapshot.Copy", "dst", resources_->stage, kStageBytes);
    auto* out = static_cast<std::uint8_t*>(destination);
    const auto* in = static_cast<const std::uint8_t*>(source);
    for (std::size_t done = 0; done < bytes;) {
      const std::size_t chunk = std::min(kStageBytes, bytes - done);
      Guard("snapshot.Copy", "src", in + done, chunk);
      Check(hipMemcpyAsync(resources_->stage, in + done, chunk, hipMemcpyDeviceToHost,
                           resources_->stream));
      Check(hipStreamSynchronize(resources_->stream));
      std::memcpy(out + done, resources_->stage, chunk);
      done += chunk;
    }
  }

  void Copy2D(void* destination, std::size_t destination_pitch,
              const void* source, std::size_t source_pitch, std::size_t width,
              std::size_t height, hipMemcpyKind kind = hipMemcpyDeviceToHost) {
    if (kind != hipMemcpyDeviceToHost) {
      Guard("snapshot.Copy2D", "dst", destination,
            Extent(destination_pitch, width, height));
      Guard("snapshot.Copy2D", "src", source,
            Extent(source_pitch, width, height));
      Check(hipMemcpy2DAsync(destination, destination_pitch, source,
                             source_pitch, width, height, kind, resources_->stream));
      Check(hipStreamSynchronize(resources_->stream));
      return;
    }
    EnsureStage();
    Guard("snapshot.Copy2D", "dst", resources_->stage, kStageBytes);
    auto* out = static_cast<std::uint8_t*>(destination);
    const auto* in = static_cast<const std::uint8_t*>(source);
    for (std::size_t row = 0; row < height; ++row) {
      auto* out_row = out + row * destination_pitch;
      const auto* in_row = in + row * source_pitch;
      for (std::size_t done = 0; done < width;) {
        const std::size_t chunk = std::min(kStageBytes, width - done);
        Guard("snapshot.Copy2D", "src", in_row + done, chunk);
        Check(hipMemcpyAsync(resources_->stage, in_row + done, chunk,
                             hipMemcpyDeviceToHost, resources_->stream));
        Check(hipStreamSynchronize(resources_->stream));
        std::memcpy(out_row + done, resources_->stage, chunk);
        done += chunk;
      }
    }
  }

private:
  static std::size_t Extent(std::size_t pitch, std::size_t width,
                            std::size_t height) noexcept {
    return height == 0 ? 0 : (height - 1) * pitch + width;
  }
  static void Guard(const char* site, const char* role, const void* pointer,
                    std::size_t bytes) {
    if (check_ != nullptr)
      check_(site, role, pointer, bytes);
  }
  inline static RangeCheck check_ = nullptr;
  inline static AllocRecord record_ = nullptr;
  inline static AllocForget forget_ = nullptr;
  struct Resources {
    hipStream_t stream{nullptr};
    void* stage{nullptr};  ///< pinned, allocated on the first host-bound copy
  };
  static std::mutex& PoolMutex() noexcept {
    static std::mutex mutex;
    return mutex;
  }
  static std::vector<Resources*>& Pool() {
    static std::vector<Resources*> pool;
    return pool;
  }
  static Resources* Acquire() {
    {
      const std::lock_guard lock(PoolMutex());
      auto& pool = Pool();
      if (!pool.empty()) {
        Resources* resources = pool.back();
        pool.pop_back();
        return resources;
      }
    }
    // Never freed: the pool only grows to the number of concurrent snapshots.
    auto* resources = new Resources;
    const std::lock_guard lock(DeviceMutex());
    const hipError_t status =
        hipStreamCreateWithFlags(&resources->stream, hipStreamNonBlocking);
    if (status != hipSuccess) {
      delete resources;
      throw std::runtime_error(hipGetErrorString(status));
    }
    return resources;
  }
  static void Check(hipError_t status) {
    if (status != hipSuccess)
      throw std::runtime_error(hipGetErrorString(status));
  }
  void EnsureStage() {
    if (resources_->stage != nullptr)
      return;
    const std::lock_guard lock(DeviceMutex());
    Check(hipHostMalloc(&resources_->stage, kStageBytes));
    if (record_ != nullptr)
      record_("snapshot.stage", resources_->stage, kStageBytes);
  }
  static constexpr std::size_t kStageBytes = std::size_t{32} << 20;
  Resources* resources_;
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_
