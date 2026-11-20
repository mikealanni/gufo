#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/iu4_weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <string_view>
#include <unordered_map>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/iu4.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint8_t kQuantMq4 = 44;
constexpr std::size_t kHeaderBytes = 32;

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31,
                      man = h & 1023;
  float v;
  if (exp == 0)
    v = std::ldexp(static_cast<float>(man), -24);
  else if (exp == 31)
    v = man == 0 ? INFINITY : NAN;
  else
    v = std::ldexp(static_cast<float>(man | 1024), static_cast<int>(exp) - 25);
  return sign ? -v : v;
}

struct Entry {
  std::uint8_t quant{0};
  std::vector<std::uint32_t> dims;
  std::uint64_t offset{0};
  std::uint64_t length{0};
};

struct Mapping {
  const std::uint8_t* base{nullptr};
  std::size_t size{0};
  ~Mapping() {
    if (base != nullptr)
      munmap(const_cast<std::uint8_t*>(base), size);
  }
};

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

bool ReadIndex(const Mapping& map, std::unordered_map<std::string, Entry>* out,
               std::string* error) {
  const std::uint8_t* base = map.base;
  if (map.size < kHeaderBytes)
    return Fail(error, "artifact header truncated");
  std::uint32_t count;
  std::uint64_t meta_off, data_off;
  std::memcpy(&count, base + 12, 4);
  std::memcpy(&meta_off, base + 16, 8);
  std::memcpy(&data_off, base + 24, 8);
  if (meta_off >= data_off || data_off > map.size)
    return Fail(error, "artifact offsets out of range");
  std::size_t pos = meta_off;
  int depth = 0;
  bool in_string = false;
  bool closed = false;
  for (; pos < data_off; ++pos) {
    const char c = static_cast<char>(base[pos]);
    if (in_string) {
      if (c == '\\')
        ++pos;
      else if (c == '"')
        in_string = false;
    } else if (c == '"') {
      in_string = true;
    } else if (c == '{') {
      ++depth;
    } else if (c == '}' && --depth == 0) {
      ++pos;
      closed = true;
      break;
    }
  }
  if (!closed || pos + 4 > data_off)
    return Fail(error, "artifact metadata is malformed");
  std::uint32_t n;
  std::memcpy(&n, base + pos, 4);
  pos += 4;
  std::uint64_t offset = data_off;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (pos + 2 > data_off)
      return Fail(error, "artifact index truncated");
    std::uint16_t name_len;
    std::memcpy(&name_len, base + pos, 2);
    pos += 2;
    if (pos + name_len + 2 > data_off)
      return Fail(error, "artifact index truncated");
    std::string name(reinterpret_cast<const char*>(base + pos), name_len);
    pos += name_len;
    Entry entry;
    entry.quant = base[pos++];
    const std::uint8_t ndim = base[pos++];
    if (pos + 4ull * ndim + 12 > data_off)
      return Fail(error, "artifact index truncated");
    entry.dims.resize(ndim);
    for (int d = 0; d < ndim; ++d) {
      std::memcpy(&entry.dims[d], base + pos, 4);
      pos += 4;
    }
    pos += 4;  // group size
    std::memcpy(&entry.length, base + pos, 8);
    pos += 8;
    entry.offset = offset;
    offset += entry.length;
    out->emplace(std::move(name), std::move(entry));
  }
  if (offset > map.size)
    return Fail(error, "artifact data section truncated");
  (void)count;
  return true;
}

struct RoleName {
  Iu4Role role;
  const char* path;  // below "model.language_model.layers.N."
};

constexpr RoleName kRoles[] = {
    {Iu4Role::kAttnQ, "self_attn.q_proj"},
    {Iu4Role::kAttnK, "self_attn.k_proj"},
    {Iu4Role::kAttnV, "self_attn.v_proj"},
    {Iu4Role::kAttnO, "self_attn.o_proj"},
    {Iu4Role::kSsmQkv, "linear_attn.in_proj_qkv"},
    {Iu4Role::kSsmZ, "linear_attn.in_proj_z"},
    {Iu4Role::kSsmA, "linear_attn.in_proj_a"},
    {Iu4Role::kSsmB, "linear_attn.in_proj_b"},
    {Iu4Role::kSsmOut, "linear_attn.out_proj"},
    {Iu4Role::kFfnGate, "mlp.gate_proj"},
    {Iu4Role::kFfnUp, "mlp.up_proj"},
    {Iu4Role::kFfnDown, "mlp.down_proj"},
};

constexpr std::size_t kRoleCount = static_cast<std::size_t>(Iu4Role::kCount);
const Iu4Projection kEmptyProjection{};

}  // namespace

Iu4Weights::~Iu4Weights() {
  Release();
}

void Iu4Weights::Release() noexcept {
  const auto release = [](Iu4Projection& p) {
    if (p.tiles != nullptr)
      (void)hipFree(p.tiles);
    if (p.scales != nullptr)
      (void)hipFree(p.scales);
    if (p.awq != nullptr)
      (void)hipFree(p.awq);
    p = {};
  };
  for (auto& p : projections_)
    release(p);
  projections_.clear();
  release(lm_head_);
  layers_ = 0;
  device_bytes_ = 0;
}

const Iu4Projection& Iu4Weights::Find(std::uint32_t layer,
                                      Iu4Role role) const noexcept {
  const std::size_t index = layer * kRoleCount + static_cast<std::size_t>(role);
  return index < projections_.size() ? projections_[index] : kEmptyProjection;
}

bool Iu4Weights::Load(const std::string& path, std::uint32_t layers,
                      std::string* error) {
  Release();
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return Fail(error, "cannot open " + path);
  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(kHeaderBytes)) {
    close(fd);
    return Fail(error, "cannot stat " + path);
  }
  void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED)
    return Fail(error, "cannot map " + path);
  Mapping map{static_cast<const std::uint8_t*>(mapped),
              static_cast<std::size_t>(st.st_size)};
  std::unordered_map<std::string, Entry> index;
  if (!ReadIndex(map, &index, error))
    return false;

  unsigned long long* d_bad = nullptr;
  if (hipMalloc(&d_bad, sizeof(*d_bad)) != hipSuccess)
    return Fail(error, "device allocation failed");
  std::unique_ptr<unsigned long long, decltype(&hipFree)> bad_guard(d_bad,
                                                                    &hipFree);
  if (hipMemset(d_bad, 0, sizeof(*d_bad)) != hipSuccess)
    return Fail(error, "device memset failed");

  hipStream_t stream = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess)
    return Fail(error, "stream creation failed");
  struct StreamGuard {
    hipStream_t s;
    ~StreamGuard() { (void)hipStreamDestroy(s); }
  } stream_guard{stream};

  const auto upload = [&](const std::string& name, const std::string& awq_name,
                          Iu4Projection* out) -> bool {
    const auto it = index.find(name);
    if (it == index.end())
      return Fail(error, "missing tensor " + name);
    const Entry& e = it->second;
    if (e.quant != kQuantMq4 || e.dims.size() != 2)
      return Fail(error, name + " is not a 4-bit projection");
    const std::size_t rows = e.dims[0], cols = e.dims[1];
    if (rows % 16 != 0 || cols % 256 != 0 ||
        e.length != rows * (cols / 256) * kIu4BlockBytes)
      return Fail(error, name + " has an unsupported shape");

    void* d_blocks = nullptr;
    if (hipMalloc(&d_blocks, e.length) != hipSuccess)
      return Fail(error, "device allocation failed for " + name);
    std::unique_ptr<void, decltype(&hipFree)> blocks_guard(d_blocks, &hipFree);
    if (hipMemcpyAsync(d_blocks, map.base + e.offset, e.length,
                       hipMemcpyHostToDevice, stream) != hipSuccess)
      return Fail(error, "upload failed for " + name);

    Iu4Projection p;
    p.rows = static_cast<std::uint32_t>(rows);
    p.cols = static_cast<std::uint32_t>(cols);
    const std::size_t tile_bytes = rows * cols / 2;
    const std::size_t scale_bytes = rows * (cols / 128) * sizeof(std::uint16_t);
    if (hipMalloc(&p.tiles, tile_bytes) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&p.scales), scale_bytes) !=
            hipSuccess) {
      if (p.tiles != nullptr)
        (void)hipFree(p.tiles);
      return Fail(error, "device allocation failed for " + name);
    }
    device_bytes_ += tile_bytes + scale_bytes;
    LaunchIu4RepackWeights(d_blocks, p.tiles, p.scales, d_bad, rows, cols,
                           stream);
    if (!awq_name.empty()) {
      const auto ait = index.find(awq_name);
      if (ait == index.end() || ait->second.length != cols * 2 ||
          ait->second.quant != 1) {
        (void)hipFree(p.tiles);
        (void)hipFree(p.scales);
        return Fail(error, "missing or malformed " + awq_name);
      }
      std::vector<float> host(cols);
      const std::uint8_t* src = map.base + ait->second.offset;
      for (std::size_t i = 0; i < cols; ++i) {
        std::uint16_t h;
        std::memcpy(&h, src + 2 * i, 2);
        host[i] = HalfToFloat(h);
      }
      if (hipMalloc(reinterpret_cast<void**>(&p.awq), cols * 4) != hipSuccess ||
          hipMemcpyAsync(p.awq, host.data(), cols * 4, hipMemcpyHostToDevice,
                         stream) != hipSuccess) {
        (void)hipFree(p.tiles);
        (void)hipFree(p.scales);
        return Fail(error, "awq upload failed for " + awq_name);
      }
      if (hipStreamSynchronize(stream) != hipSuccess)
        return Fail(error, "device error while loading " + name);
      device_bytes_ += cols * 4;
    } else if (hipStreamSynchronize(stream) != hipSuccess) {
      return Fail(error, "device error while loading " + name);
    }
    *out = p;
    return true;
  };

  projections_.assign(static_cast<std::size_t>(layers) * kRoleCount, {});
  layers_ = layers;
  for (std::uint32_t l = 0; l < layers; ++l) {
    const std::string prefix =
        "model.language_model.layers." + std::to_string(l) + ".";
    for (const auto& r : kRoles) {
      const std::string name = prefix + r.path + ".weight";
      if (index.find(name) == index.end())
        continue;  // the layer type does not have this projection
      if (!upload(name, prefix + r.path + ".awq_scale.weight",
                  &projections_[l * kRoleCount +
                                static_cast<std::size_t>(r.role)])) {
        Release();
        return false;
      }
    }
  }
  if (!upload("lm_head.weight", "", &lm_head_)) {
    Release();
    return false;
  }
  unsigned long long bad = 0;
  if (hipMemcpy(&bad, d_bad, sizeof(bad), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    Release();
    return Fail(error, "device error reading zero point count");
  }
  if (bad != 0) {
    Release();
    return Fail(error, std::to_string(bad) +
                           " weight groups have a zero point other than -8 "
                           "scales; the symmetric kernel cannot read them");
  }
  return true;
}

}  // namespace gufo::hip
#endif
