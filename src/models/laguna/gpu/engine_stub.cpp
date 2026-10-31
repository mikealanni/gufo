#include "src/models/laguna/gpu/engine.hpp"

namespace gufo::models::laguna::gpu {

struct ModelState {};
struct Session::Impl {};

Session::Session() = default;
Session::~Session() = default;
bool Session::Append(std::span<const std::int32_t>, std::string* error_msg) {
  if (error_msg != nullptr)
    *error_msg = "Laguna GPU path not compiled in";
  return false;
}
bool Session::AppendAll(std::span<const std::int32_t>, std::string* error_msg) {
  if (error_msg != nullptr)
    *error_msg = "Laguna GPU path not compiled in";
  return false;
}
void Session::Rewind(std::uint32_t) noexcept {}
std::span<const float> Session::Logits() const noexcept {
  return {};
}
std::span<const float> Session::Logits(std::uint32_t) const noexcept {
  return {};
}
std::uint32_t Session::Position() const noexcept {
  return 0;
}
std::uint32_t Session::Capacity() const noexcept {
  return 0;
}
void Session::Reset() noexcept {}
std::size_t Session::AllocatedBytes() const noexcept {
  return 0;
}

Model::~Model() = default;
std::shared_ptr<Model> Model::Load(const std::string&, const EngineOptions&,
                                   std::string* error_msg) {
  if (error_msg != nullptr)
    *error_msg = "Laguna GPU path not compiled in";
  return nullptr;
}
std::unique_ptr<Session> Model::CreateSession(std::uint32_t, std::string*) {
  return nullptr;
}
const Config& Model::config() const noexcept {
  static const Config kEmpty;
  return kEmpty;
}
std::size_t Model::ResidentBytes() const noexcept {
  return 0;
}

}  // namespace gufo::models::laguna::gpu
