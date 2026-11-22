#ifndef GUFO_MODELS_INKLING_REFERENCE_HPP_
#define GUFO_MODELS_INKLING_REFERENCE_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/inkling/runtime/config.hpp"

namespace gufo::models::inkling {

class ReferenceModel {
public:
  static std::unique_ptr<ReferenceModel> Open(const std::string& path,
                                              std::string* error);

  static constexpr std::uint32_t kConvTaps = 4;
  static constexpr std::uint32_t kHistory = kConvTaps - 1;

  struct SlotState {
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> hist_k;
    std::vector<float> hist_v;
    std::vector<float> hist_attn;
    std::vector<float> hist_ffn;
  };

  struct Session {
    std::vector<SlotState> slots;
    std::vector<float> logits;
    std::uint32_t position{0};
  };

  std::unique_ptr<Session> CreateSession() const;
  const std::vector<float>& Evaluate(Session& session,
                                     std::int32_t token) const;
  const Config& config() const { return config_; }

private:
  ReferenceModel() = default;

  struct Mat {
    const gufo::core::GgufTensorInfo* t{nullptr};
    std::uint64_t rows{0};
    std::uint64_t cols{0};
    std::uint64_t row_bytes{0};
    void Row(std::uint64_t row, float* out) const;
    void MatVec(const float* x, float* y, std::uint64_t row0,
                std::uint64_t n) const;
  };

  using Vec = std::vector<float>;

  Mat M(const std::string& name) const;
  Vec F(const std::string& name) const;
  std::string P(std::uint32_t il, const char* s) const;
  std::uint32_t Phys(std::uint32_t slot) const;
  bool IsSwa(std::uint32_t slot) const;

  Vec Mix(const std::vector<Vec>& blocks, const Vec& cur, const float* query,
          std::uint32_t nq) const;
  Vec Control(const Vec& cur, const Vec& message, std::uint32_t phys,
              const char* group) const;
  Vec Convolve(const Vec& cur, const Vec& w, const Vec* dyn, Vec& hist) const;
  Vec Layer(std::uint32_t slot, std::uint32_t ip, Vec cur,
            std::vector<Vec>& blocks, const std::vector<Vec>* messages,
            Session& s, std::uint32_t pos) const;
  Vec Moe(std::uint32_t phys, const Vec& x) const;

  std::unique_ptr<gufo::core::GgufReader> reader_;
  Config config_;
  std::uint32_t n_phys_{0};
};

}  // namespace gufo::models::inkling

#endif  // GUFO_MODELS_INKLING_REFERENCE_HPP_
