#include <immintrin.h>
#include <iterator>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/npu/engine.hpp"
#include "src/models/lfm3/tokenizer.hpp"

using gufo::models::lfm3::npu::Engine;

static std::vector<std::uint8_t> Slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  std::vector<std::uint8_t> b(static_cast<std::size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(b.data()), b.size());
  return b;
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "";
  gufo::models::lfm3::npu::Options opt;
  std::string in, out;
  std::size_t maxtok = 512, maxnew = 128;
  std::string prompt;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    std::string v = argv[i + 1];
    if (k == "--gguf") opt.gguf = v;
    else if (k == "--bf16") opt.bf16 = v;
    else if (k == "--programs") opt.programs = v;
    else if (k == "--tokens") in = v;
    else if (k == "--out") out = v;
    else if (k == "--maxtok") maxtok = std::stoul(v);
    else if (k == "--max-new") maxnew = std::stoul(v);
    else if (k == "--prompt") prompt = v;
    else if (k == "--context") opt.context = std::stoul(v);
  }
  if (!std::getenv("OMP_NUM_THREADS"))
    omp_set_num_threads(8);
  std::string err;
  auto engine = Engine::Create(opt, &err);
  if (!engine) {
    std::fprintf(stderr, "gufo-npu: %s\n", err.c_str());
    return 1;
  }
  if (cmd == "capture") {
    const std::string prefix = in.substr(0, in.size() - 7);
    auto tok = Slurp(in);
    auto docs = Slurp(prefix + ".docs");
    const std::size_t n_tok = tok.size() / 4, n_doc = docs.size() / 8;
    const auto* t32 = reinterpret_cast<const std::uint32_t*>(tok.data());
    const auto* starts = reinterpret_cast<const std::uint64_t*>(docs.data());
    std::ofstream to(out + ".tokens", std::ios::binary), ho(out + ".hidden", std::ios::binary),
        dofs(out + ".docs", std::ios::binary);
    std::uint64_t total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t d = 0; d < n_doc; ++d) {
      const std::size_t lo = starts[d], hi = d + 1 < n_doc ? starts[d + 1] : n_tok;
      std::size_t len = std::min(hi - lo, maxtok);
      if (len < 8) continue;
      engine->Reset();
      std::vector<std::uint16_t> half(len * 2048);
      for (std::size_t at = 0; at < len;) {
        std::size_t n = std::min<std::size_t>(Engine::kMaxRows, len - at);
        std::vector<float> h;
        if (!engine->Step(std::span<const std::uint32_t>(t32 + lo + at, n), &h)) {
          std::fprintf(stderr, "step failed\n");
          return 1;
        }
        for (std::size_t i = 0; i < n * 2048; ++i) {
          __m128i hv = _mm_cvtps_ph(_mm_set_ss(h[i]), _MM_FROUND_TO_NEAREST_INT);
          half[at * 2048 + i] = static_cast<std::uint16_t>(_mm_extract_epi16(hv, 0));
        }
        at += n;
      }
      dofs.write(reinterpret_cast<const char*>(&total), 8);
      to.write(reinterpret_cast<const char*>(t32 + lo), len * 4);
      ho.write(reinterpret_cast<const char*>(half.data()), half.size() * 2);
      total += len;
      if (d % 10 == 0)
        std::fprintf(stderr, "capture: doc %zu/%zu, %llu tokens, %.0f tok/s\n", d, n_doc,
                     (unsigned long long)total,
                     total / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("captured %llu tokens\n", (unsigned long long)total);
    return 0;
  }
  if (cmd == "generate") {
    auto reader = gufo::core::GgufReader::OpenFile(opt.gguf, &err);
    auto tokenizer = gufo::models::lfm3::Tokenizer::Load(*reader, &err);
    if (!tokenizer) {
      std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
      return 1;
    }
    std::string text = prompt;
    if (!text.empty() && text[0] == '@') {
      std::ifstream f(text.substr(1));
      text.assign(std::istreambuf_iterator<char>(f), {});
    }
    std::vector<std::uint32_t> ids;
    if (std::getenv("GUFO_RAW_PROMPT")) {
      ids = tokenizer->Encode(text);
    } else {
      const std::vector<std::pair<std::string, std::string>> messages = {{"user", text}};
      ids = tokenizer->Encode(gufo::models::lfm3::Tokenizer::Render(messages, false));
    }
    std::uint32_t eos = 7;
    if (auto e = reader->GetMetadataUint32("tokenizer.ggml.eos_token_id"))
      eos = *e;
    std::vector<std::uint32_t> out;
    std::vector<float> h;
    engine->Reset();
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t at = 0; at < ids.size();) {
      const std::size_t n = std::min<std::size_t>(Engine::kMaxRows, ids.size() - at);
      if (!engine->Step(std::span<const std::uint32_t>(ids.data() + at, n), &h))
        return 1;
      at += n;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::uint32_t next = engine->Argmax(h.data() + (h.size() - 2048));
    for (std::size_t i = 0; i < maxnew; ++i) {
      out.push_back(next);
      if (next == eos || tokenizer->IsStop(next))
        break;
      if (!engine->Step(std::span<const std::uint32_t>(&next, 1), &h))
        return 1;
      next = engine->Argmax(h.data());
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("%s\n", tokenizer->Decode(out).c_str());
    const double pre = std::chrono::duration<double>(t1 - t0).count();
    const double dec = std::chrono::duration<double>(t2 - t1).count();
    std::fprintf(stderr, "prompt %zu tokens in %.3f s (%.0f tok/s); %zu new tokens in %.3f s (%.2f tok/s)\n",
                 ids.size(), pre, ids.size() / pre, out.size(), dec, out.size() / dec);
    std::printf("ids:");
    for (auto t : out) std::printf(" %u", t);
    std::printf("\n");
    return 0;
  }
  std::fprintf(stderr, "usage: gufo-npu capture|generate ...\n");
  return 2;
}
