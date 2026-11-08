// Null-object build of NpuGemm for configurations compiled without the XDNA2
// NPU (GUFO_ENABLE_NPU=OFF). The engine (executor.cpp) and the device model
// reference NpuGemm's members unconditionally, so this translation unit stands
// in for kernels/rocm/npu_gemm.cpp and supplies the symbols that a non-NPU link
// otherwise reports as undefined. It never touches XRT or the NPU, so it builds
// with only the HIP headers.
//
// Only one of npu_gemm.cpp and this file is ever compiled: CMake selects the
// real one when GUFO_ENABLE_NPU is on (defining GUFO_NPU and linking XRT) and
// this stub otherwise. NpuGemm::Create is dead code in a non-NPU build because
// device_model.cpp constructs it only under GUFO_NPU; every other member here
// returns the inert value the engine already handles, and the engine's
// `npu() != nullptr` guards keep the NPU path out of the prefill and decode
// loops, so this adds nothing to their runtime.

#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

// The header forward-declares Impl for the unique_ptr member; the real
// definition lives in npu_gemm.cpp, so this file owns it instead.
struct NpuGemm::Impl {};

NpuGemm::NpuGemm() = default;
NpuGemm::~NpuGemm() = default;

std::unique_ptr<NpuGemm> NpuGemm::Create(const Config& /*config*/,
                                         std::string* error) {
  if (error != nullptr) {
    *error = "NPU support was not compiled into this build";
  }
  return nullptr;
}

std::size_t NpuGemm::ABytes() const noexcept {
  return 0;
}
std::size_t NpuGemm::BBytes() const noexcept {
  return 0;
}
std::size_t NpuGemm::CBytes() const noexcept {
  return 0;
}

std::uint32_t NpuGemm::tokens() const noexcept {
  return 0;
}
std::uint32_t NpuGemm::k() const noexcept {
  return 0;
}
std::uint32_t NpuGemm::n() const noexcept {
  return 0;
}

int NpuGemm::NewBuffer(std::size_t /*bytes*/, std::string* error) {
  if (error != nullptr) {
    *error = "NPU support was not compiled into this build";
  }
  return -1;
}

void* NpuGemm::Host(int /*id*/) const {
  return nullptr;
}
void* NpuGemm::Device(int /*id*/) const {
  return nullptr;
}
void NpuGemm::FinalizeWeights(int /*id*/) {}

bool NpuGemm::Submit(hipStream_t /*stream*/, int /*a_id*/, const int* /*b_ids*/,
                     const int* /*c_ids*/, int /*count*/, int /*program*/) {
  return false;
}

int NpuGemm::AddProgram(const std::string& /*insts*/, std::string* /*error*/) {
  return -1;
}
std::size_t NpuGemm::ABytes(std::uint32_t /*k*/) const noexcept {
  return 0;
}
std::size_t NpuGemm::BBytes(std::uint32_t /*k*/) const noexcept {
  return 0;
}

void* NpuGemm::Staged(hipStream_t /*stream*/, int /*id*/) {
  return nullptr;
}

std::uint64_t NpuGemm::Ticket() const noexcept {
  return 0;
}

bool NpuGemm::Wait(hipStream_t /*stream*/, std::uint64_t /*ticket*/) {
  return false;
}

bool NpuGemm::Wait(hipStream_t /*stream*/) {
  return false;
}

bool NpuGemm::Healthy() const noexcept {
  return false;
}

void NpuGemm::MarkFailed() noexcept {}

void RegisterQwenNpuFfn() {}

}  // namespace gufo::models::qwen38_flash_next::rocm
