// On-disk cache of NPU-packed weight buffers (one file per buffer), keyed by the source model file and a format version.
#ifndef GUFO_NPU_RUNTIME_WEIGHT_CACHE_HPP_
#define GUFO_NPU_RUNTIME_WEIGHT_CACHE_HPP_

#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace gufo::npu::runtime {

/// Bump when any pack layout (int5, ATB bfp16, fused FFN) changes.
inline constexpr std::uint32_t kWeightCacheVersion = 1;

class WeightCache {
 public:
  /// Enabled unless GUFO_WEIGHT_CACHE=0 (or GEN_NO_WEIGHT_CACHE is set). GUFO_WEIGHT_CACHE_DIR overrides ~/.cache/gufo/npu-weights.
  void Open(const std::string& model_path) {
    dir_.clear();
    if (std::getenv("GEN_NO_WEIGHT_CACHE")) return;
    if (const char* off = std::getenv("GUFO_WEIGHT_CACHE"); off && std::string(off) == "0") return;
    struct stat st {};
    if (::stat(model_path.c_str(), &st) != 0) return;
    std::string root;
    if (const char* d = std::getenv("GUFO_WEIGHT_CACHE_DIR")) root = d;
    else if (const char* h = std::getenv("HOME")) root = std::string(h) + "/.cache/gufo/npu-weights";
    if (root.empty()) return;
    std::uint64_t h = 1469598103934665603ULL;
    const auto mix = [&h](const std::string& s) {
      for (unsigned char c : s) h = (h ^ c) * 1099511628211ULL;
    };
    mix(std::filesystem::absolute(model_path).string());
    mix(std::to_string(st.st_size));
    mix(std::to_string(st.st_mtime));
    mix(std::to_string(kWeightCacheVersion));
    char key[32];
    std::snprintf(key, sizeof key, "%016llx", static_cast<unsigned long long>(h));
    std::error_code ec;
    std::filesystem::create_directories(root + "/" + key, ec);
    if (!ec) dir_ = root + "/" + key;
  }
  [[nodiscard]] bool enabled() const { return !dir_.empty(); }

  /// Restores `bytes` bytes for `tag` into dst; false when the entry is missing or has another size.
  bool Load(const std::string& tag, std::uint8_t* dst, std::size_t bytes) const {
    if (dir_.empty()) return false;
    std::ifstream f(dir_ + "/" + tag + ".bin", std::ios::binary | std::ios::ate);
    if (!f || static_cast<std::size_t>(f.tellg()) != bytes) return false;
    f.seekg(0);
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(f);
  }

  /// Atomic write (temp file, then rename) so a killed run never leaves a truncated entry.
  void Store(const std::string& tag, const std::uint8_t* src, std::size_t bytes) const {
    if (dir_.empty()) return;
    const std::string path = dir_ + "/" + tag + ".bin", tmp = path + ".tmp" + std::to_string(::getpid());
    {
      std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(bytes));
      if (!f) {
        std::remove(tmp.c_str());
        return;
      }
    }
    std::rename(tmp.c_str(), path.c_str());
  }

 private:
  std::string dir_;
};

}  // namespace gufo::npu::runtime

#endif  // GUFO_NPU_RUNTIME_WEIGHT_CACHE_HPP_
