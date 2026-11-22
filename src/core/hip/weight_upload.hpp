#ifndef GUFO_CORE_HIP_WEIGHT_UPLOAD_HPP_
#define GUFO_CORE_HIP_WEIGHT_UPLOAD_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::hip {

/// Bounded disk-to-device pipeline. Copy queues a tensor without waiting for
/// its upload; destinations must remain alive until Finish or destruction.
/// The mapped GGUF payloads are never touched.
class WeightUpload {
public:
  static std::unique_ptr<WeightUpload> Create(
      std::span<const core::GgufMappedRegion> shards, std::string* error);
  ~WeightUpload();
  WeightUpload(const WeightUpload&) = delete;
  WeightUpload& operator=(const WeightUpload&) = delete;

  bool Copy(std::uint32_t shard, std::uint64_t offset, std::size_t size,
            void* device, std::string* error);
  /// Waits for every queued copy, including on failure.
  bool Finish(std::string* error);

private:
  struct State;
  WeightUpload();
  std::unique_ptr<State> state_;
};

/// Default way for a model loader to put a GGUF tensor on the device: the
/// staged WeightUpload pipeline, with a plain mapped-memory copy as fallback.
/// Loaders call Upload for every tensor instead of hipMemcpy from the mapping.
class TensorUpload {
public:
  explicit TensorUpload(const core::GgufReader& reader);
  ~TensorUpload();
  TensorUpload(const TensorUpload&) = delete;
  TensorUpload& operator=(const TensorUpload&) = delete;

  /// Copies the tensor's bytes to device + dst_offset and waits for them.
  bool Upload(const core::GgufTensorInfo& tensor, void* device,
              std::size_t dst_offset, std::string* error);
  bool Upload(const core::GgufTensorInfo& tensor, void* device,
              std::string* error) {
    return Upload(tensor, device, 0, error);
  }

private:
  std::vector<core::GgufMappedRegion> shards_;
  std::unique_ptr<WeightUpload> stager_;
};

}  // namespace gufo::hip

#endif
