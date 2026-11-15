// On-disk cache of NPU-packed weight buffers (one file per buffer), keyed by
// the source model file and a format version.
#ifndef GUFO_NPU_RUNTIME_WEIGHT_CACHE_HPP_
#define GUFO_NPU_RUNTIME_WEIGHT_CACHE_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gufo::npu::runtime {

/// Bump when any pack layout (int5, ATB bfp16, fused FFN) changes.
inline constexpr std::uint32_t kWeightCacheVersion = 1;

class WeightCache {
public:
  /// Enabled unless GUFO_WEIGHT_CACHE=0 (or GEN_NO_WEIGHT_CACHE is set).
  /// GUFO_WEIGHT_CACHE_DIR overrides ~/.cache/gufo/npu-weights.
  void Open(const std::string& model_path) {
    dir_.clear();
    if (std::getenv("GEN_NO_WEIGHT_CACHE"))
      return;
    if (const char* off = std::getenv("GUFO_WEIGHT_CACHE");
        off && std::string(off) == "0")
      return;
    struct stat st{};
    if (::stat(model_path.c_str(), &st) != 0)
      return;
    std::string root;
    if (const char* d = std::getenv("GUFO_WEIGHT_CACHE_DIR"))
      root = d;
    else if (const char* h = std::getenv("HOME"))
      root = std::string(h) + "/.cache/gufo/npu-weights";
    if (root.empty())
      return;
    std::uint64_t h = 1469598103934665603ULL;
    const auto mix = [&h](const std::string& s) {
      for (unsigned char c : s)
        h = (h ^ c) * 1099511628211ULL;
    };
    mix(std::filesystem::absolute(model_path).string());
    mix(std::to_string(st.st_size));
    mix(std::to_string(st.st_mtime));
    mix(std::to_string(kWeightCacheVersion));
    char key[32];
    std::snprintf(key, sizeof key, "%016llx",
                  static_cast<unsigned long long>(h));
    std::error_code ec;
    std::filesystem::create_directories(root + "/" + key, ec);
    if (ec)
      return;
    dir_ = root + "/" + key;
    ::utimensat(AT_FDCWD, dir_.c_str(), nullptr, 0);
    Evict(root, key);
  }
  /// Least-recently-opened model directories are removed until the cache fits
  /// GUFO_WEIGHT_CACHE_GB (default 120, 0 = unlimited); the current model is
  /// kept.
  static void Evict(const std::string& root, const std::string& keep) {
    std::uint64_t cap_gb = 120;
    if (const char* c = std::getenv("GUFO_WEIGHT_CACHE_GB"))
      cap_gb = std::strtoull(c, nullptr, 10);
    if (cap_gb == 0)
      return;
    namespace fs = std::filesystem;
    struct Entry {
      fs::path path;
      fs::file_time_type used;
      std::uint64_t bytes;
    };
    std::vector<Entry> entries;
    std::uint64_t total = 0;
    std::error_code ec;
    for (const auto& d : fs::directory_iterator(root, ec)) {
      if (!d.is_directory(ec))
        continue;
      std::uint64_t bytes = 0;
      for (const auto& f : fs::directory_iterator(d.path(), ec))
        if (f.is_regular_file(ec))
          bytes += f.file_size(ec);
      total += bytes;
      if (d.path().filename() != keep)
        entries.push_back({d.path(), d.last_write_time(ec), bytes});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.used < b.used; });
    const std::uint64_t cap = cap_gb << 30;
    for (const Entry& e : entries) {
      if (total <= cap)
        break;
      fs::remove_all(e.path, ec);
      if (!ec)
        total -= e.bytes;
    }
  }
  [[nodiscard]] bool enabled() const { return !dir_.empty(); }

  /// Restores `bytes` bytes for `tag` into dst; false when the entry is missing
  /// or has another size.
  bool Load(const std::string& tag, std::uint8_t* dst,
            std::size_t bytes) const {
    if (dir_.empty())
      return false;
    std::ifstream f(dir_ + "/" + tag + ".bin",
                    std::ios::binary | std::ios::ate);
    if (!f || static_cast<std::size_t>(f.tellg()) != bytes)
      return false;
    f.seekg(0);
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(f);
  }

  /// Read-only mapping of the entry for `tag` when it exists with exactly
  /// `bytes` bytes; nullptr otherwise. Release with Unmap.
  const std::uint8_t* Map(const std::string& tag, std::size_t bytes) const {
    if (dir_.empty())
      return nullptr;
    const int fd = ::open((dir_ + "/" + tag + ".bin").c_str(), O_RDONLY);
    if (fd < 0)
      return nullptr;
    struct stat st{};
    void* p = MAP_FAILED;
    if (::fstat(fd, &st) == 0 && static_cast<std::size_t>(st.st_size) == bytes)
      p = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    return p == MAP_FAILED ? nullptr : static_cast<const std::uint8_t*>(p);
  }
  static void Unmap(const std::uint8_t* p, std::size_t bytes) {
    if (p)
      ::munmap(const_cast<std::uint8_t*>(p), bytes);
  }

  /// Atomic write (temp file, then rename) so a killed run never leaves a
  /// truncated entry.
  void Store(const std::string& tag, const std::uint8_t* src,
             std::size_t bytes) const {
    if (dir_.empty())
      return;
    const std::string path = dir_ + "/" + tag + ".bin",
                      tmp = path + ".tmp" + std::to_string(::getpid());
    {
      std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(src),
              static_cast<std::streamsize>(bytes));
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
