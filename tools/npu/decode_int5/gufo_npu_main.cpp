#include <immintrin.h>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/npu/engine.hpp"
#include "src/models/lfm3/tokenizer.hpp"

using gufo::models::lfm3::npu::Engine;

static double CpuSeconds() {
  rusage u;
  getrusage(RUSAGE_SELF, &u);
  return u.ru_utime.tv_sec + u.ru_stime.tv_sec + 1e-6 * (u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}

static std::vector<std::uint8_t> Slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  std::vector<std::uint8_t> b(static_cast<std::size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(b.data()), b.size());
  return b;
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "";
  // Prefill waits long on the NPU, so its host threads sleep; decode runs many tiny parallel regions per token and
  // needs the default short spin (passive waits cost about 6 ms per token).
  if (cmd == "capture" && !std::getenv("OMP_WAIT_POLICY") && !std::getenv("GUFO_OMP_ACTIVE")) {
    setenv("OMP_WAIT_POLICY", "passive", 1);
    execv("/proc/self/exe", argv);
  }
  // Decode threads mostly wait on the NPU: a short spin keeps per-token latency and sleeps through the long waits.
  if ((cmd == "generate" || cmd == "chat") && !std::getenv("OMP_WAIT_POLICY") && !std::getenv("GOMP_SPINCOUNT")) {
    setenv("GOMP_SPINCOUNT", "1000", 1);
    execv("/proc/self/exe", argv);
  }
  gufo::models::lfm3::npu::Options opt;
  std::string in, out;
  std::size_t maxtok = 512, maxnew = 128;
  std::string prompt, heads_path;
  std::size_t head_k = 2;
  const int dec_threads = std::getenv("GUFO_DECODE_THREADS") ? std::atoi(std::getenv("GUFO_DECODE_THREADS")) : 2;
  const int pre_threads = std::getenv("OMP_NUM_THREADS") ? std::atoi(std::getenv("OMP_NUM_THREADS")) : 12;
  // Passes above 16 rows run the m64 programs, which corrupt decode state; speculation stays on the m16 programs.
  const std::size_t max_fed = std::getenv("GUFO_MAXFED") ? std::stoul(std::getenv("GUFO_MAXFED")) : 16;
  bool spec = std::getenv("GUFO_NO_SPEC") == nullptr;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    std::string v = argv[i + 1];
    if (k == "--gguf") opt.gguf = v;
    else if (k == "--bf16") opt.bf16 = v;
    else if (k == "--programs") opt.programs = v;
    else if (k == "--spec") opt.spec = v;
    else if (k == "--tokens") in = v;
    else if (k == "--out") out = v;
    else if (k == "--maxtok") maxtok = std::stoul(v);
    else if (k == "--max-new") maxnew = std::stoul(v);
    else if (k == "--prompt") prompt = v;
    else if (k == "--heads") heads_path = v;
    else if (k == "--head-k") head_k = std::stoul(v);
    else if (k == "--context") opt.context = std::stoul(v);
  }
  if (!std::getenv("OMP_NUM_THREADS"))
    omp_set_num_threads(12);
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
        const std::size_t chunk = engine->PrefillRows();
        std::size_t n = std::min<std::size_t>(chunk, len - at);
        std::vector<float> h;
        if (!engine->Step(std::span<const std::uint32_t>(t32 + lo + at, n), &h)) {
          std::fprintf(stderr, "step failed\n");
          return 1;
        }
        for (std::size_t i = 0; i < n * 2048; ++i) {
          __m128i hv = _mm_cvtps_ph(_mm_set_ss(h[i]), _MM_FROUND_TO_NEAREST_INT);
          half[at * 2048 + i] = static_cast<std::uint16_t>(_mm_extract_epi16(hv, 0));
        }
        engine->Commit(n);
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
    if (!heads_path.empty() && !engine->LoadHeads(heads_path, &err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    engine->LimitHeads(static_cast<std::uint32_t>(head_k));
    std::vector<std::uint32_t> head_draft;
    if (const char* rc = std::getenv("GUFO_ROWCHECK")) {
      const std::size_t R = std::max<std::size_t>(2, std::stoul(rc));
      std::vector<std::uint32_t> tk;
      for (std::size_t i = 0; i < R; ++i) tk.push_back(static_cast<std::uint32_t>(1000 + 37 * i));
      auto prefill = [&](std::vector<float>* hh) {
        engine->Reset();
        for (std::size_t at = 0; at < ids.size();) {
          const std::size_t n = std::min<std::size_t>(engine->PrefillRows(), ids.size() - at);
          if (!engine->Step(std::span<const std::uint32_t>(ids.data() + at, n), hh)) std::exit(1);
          engine->Commit(n);
          at += n;
        }
        const std::size_t pad = std::getenv("GUFO_ROWCHECK_PAD") ? std::stoul(std::getenv("GUFO_ROWCHECK_PAD")) : 0;
        for (std::size_t i = 0; i < pad; ++i) {
          const std::uint32_t f = 500 + static_cast<std::uint32_t>(i);
          if (!engine->Step(std::span<const std::uint32_t>(&f, 1), hh)) std::exit(1);
          engine->Commit(1);
        }
      };
      std::vector<float> hm, hs, hb;
      prefill(&hm);
      engine->Step(std::span<const std::uint32_t>(tk.data(), tk.size()), &hm);
      engine->Commit(static_cast<std::uint32_t>(R));
      prefill(&hs);
      for (std::size_t i = 0; i < R; ++i) {
        engine->Step(std::span<const std::uint32_t>(tk.data() + i, 1), &hb);
        engine->Commit(1);
        hs.assign(hb.begin(), hb.end());
        double num = 0, den = 0;
        for (std::size_t c = 0; c < 2048; ++c) { const double d = hm[i * 2048 + c] - hs[c]; num += d * d; den += double(hs[c]) * hs[c]; }
        std::fprintf(stderr, "[rowcheck] row %zu of %zu: multi-row vs single-row rel err %.5f\n", i, R, std::sqrt(num / den));
      }
      {
        // rollback: run R rows, keep all but the last, then feed the last token again as a single row
        std::vector<float> hlast = hs, hr, hr2;
        prefill(&hr);
        engine->Step(std::span<const std::uint32_t>(tk.data(), tk.size()), &hr);
        engine->Commit(static_cast<std::uint32_t>(R - 1));
        engine->Step(std::span<const std::uint32_t>(tk.data() + R - 1, 1), &hr2);
        engine->Commit(1);
        double num = 0, den = 0;
        for (std::size_t c = 0; c < 2048; ++c) { const double d = hr2[c] - hlast[c]; num += d * d; den += double(hlast[c]) * hlast[c]; }
        std::fprintf(stderr, "[rowcheck] rollback (commit %zu of %zu, redo last): rel err %.5f\n", R - 1, R, std::sqrt(num / den));
      }
      return 0;
    }
    omp_set_num_threads(pre_threads);
    engine->Reset();
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t at = 0; at < ids.size();) {
      // Without m64 the engine accepts 16-row steps only (see Engine::Step).
      static const std::size_t chunk = engine->PrefillRows();
      const std::size_t n = std::min<std::size_t>(chunk, ids.size() - at);
      if (!engine->Step(std::span<const std::uint32_t>(ids.data() + at, n), &h))
        return 1;
      engine->Commit(n);
      at += n;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double cpu1 = CpuSeconds();
    omp_set_num_threads(dec_threads);
    std::uint32_t next = engine->Argmax(h.data() + (h.size() - 2048));
    std::vector<std::uint32_t> hist = ids;
    std::size_t passes = 0, drafted = 0, accepted = 0;
    double t_step = 0, t_arg = 0, t_draft = 0;
    auto lap = [](std::chrono::steady_clock::time_point a) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); };
    bool done = false;
    while (!done && out.size() < maxnew) {
      out.push_back(next);
      hist.push_back(next);
      if (next == eos || tokenizer->IsStop(next))
        break;
      std::vector<std::uint32_t> fed = {next};
      if (spec) {
        const std::size_t hn = hist.size();
        for (std::size_t m = std::min<std::size_t>(5, hn - 1); m >= 2 && fed.size() == 1; --m) {
          for (std::size_t p = hn - 1; p-- > m;) {
            bool same = true;
            for (std::size_t q = 0; q < m && same; ++q)
              same = hist[p - q] == hist[hn - 1 - q];
            if (same) {
              for (std::size_t q = p + 1; q < hn - 1 + 1 && fed.size() < max_fed && q < hn; ++q)
                fed.push_back(hist[q]);
              break;
            }
          }
        }
      }
      if (spec && fed.size() == 1)
        for (auto t : head_draft)
          if (fed.size() < max_fed)
            fed.push_back(t);
      auto ts = std::chrono::steady_clock::now();
      if (!engine->Step(std::span<const std::uint32_t>(fed.data(), fed.size()), &h))
        return 1;
      t_step += lap(ts);
      ++passes;
      drafted += fed.size() - 1;
      std::vector<std::uint32_t> pred(fed.size());
      std::size_t have = 0, j = 0;
      for (;;) {
        const std::size_t want = std::min<std::size_t>(fed.size(), have == 0 ? 4 : fed.size());
        if (want > have) {
          auto ta = std::chrono::steady_clock::now();
          engine->ArgmaxRows(h.data() + have * 2048, want - have, pred.data() + have);
          t_arg += lap(ta);
          have = want;
        }
        while (j + 1 < fed.size() && j < have && pred[j] == fed[j + 1])
          ++j;
        if (j >= have && have < fed.size())
          continue;
        break;
      }
      while (j + 1 < fed.size() && j >= have) {
        break;
      }
      for (std::size_t q = 1; q <= j; ++q) {
        out.push_back(fed[q]);
        hist.push_back(fed[q]);
        if (fed[q] == eos || tokenizer->IsStop(fed[q]))
          done = true;
      }
      accepted += j;
      engine->Commit(static_cast<std::uint32_t>(j + 1));
      if (j >= have)
        pred[j] = 0;
      next = pred[j];
      head_draft.clear();
      if (spec && engine->HeadCount() && !done) {
        head_draft.resize(engine->HeadCount());
        auto td = std::chrono::steady_clock::now();
        engine->Draft(h.data() + j * 2048, head_draft.data());
        t_draft += lap(td);
      }
    }
    if (out.size() > maxnew)
      out.resize(maxnew);
    std::fprintf(stderr, "per pass: step %.1f ms, verify-argmax %.1f ms, draft %.1f ms\n", t_step / std::max<std::size_t>(passes, 1), t_arg / std::max<std::size_t>(passes, 1), t_draft / std::max<std::size_t>(passes, 1));
    if (spec)
      std::fprintf(stderr, "speculation: %zu passes, %zu drafted, %zu accepted, %.2f tokens/pass\n", passes,
                   drafted, accepted, passes ? double(out.size()) / passes : 0.0);
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("%s\n", tokenizer->Decode(out).c_str());
    const double pre = std::chrono::duration<double>(t1 - t0).count();
    const double dec = std::chrono::duration<double>(t2 - t1).count();
    std::fprintf(stderr, "prompt %zu tokens in %.3f s (%.0f tok/s); %zu new tokens in %.3f s (%.2f tok/s)\n",
                 ids.size(), pre, ids.size() / pre, out.size(), dec, out.size() / dec);
    std::fprintf(stderr, "decode cpu: %.2f s over %.2f s = %.1f cores busy\n", CpuSeconds() - cpu1, dec, (CpuSeconds() - cpu1) / dec);
    std::printf("ids:");
    for (auto t : out) std::printf(" %u", t);
    std::printf("\n");
    return 0;
  }
  if (cmd == "chat") {
    auto reader = gufo::core::GgufReader::OpenFile(opt.gguf, &err);
    auto tokenizer = gufo::models::lfm3::Tokenizer::Load(*reader, &err);
    if (!tokenizer) {
      std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
      return 1;
    }
    std::uint32_t eos = 7;
    if (auto e = reader->GetMetadataUint32("tokenizer.ggml.eos_token_id"))
      eos = *e;
    std::vector<std::pair<std::string, std::string>> msgs;
    std::string line;
    if (!heads_path.empty() && !engine->LoadHeads(heads_path, &err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    engine->LimitHeads(static_cast<std::uint32_t>(head_k));
    std::fprintf(stderr, "gufo-npu chat (context %zu, draft heads %u, prompt lookup %s). Empty line or Ctrl-D quits.\n",
                 static_cast<std::size_t>(opt.context), engine->HeadCount(), spec ? "on" : "off");
    for (;;) {
      std::fprintf(stderr, "\nyou> ");
      if (!std::getline(std::cin, line) || line.empty())
        break;
      msgs.push_back({"user", line});
      std::vector<std::uint32_t> ids = tokenizer->Encode(gufo::models::lfm3::Tokenizer::Render(msgs, false));
      if (ids.size() + maxnew + 1 > opt.context) {
        std::fprintf(stderr, "[context full: %zu tokens]\n", ids.size());
        break;
      }
      omp_set_num_threads(pre_threads);
      engine->Reset();
      std::vector<float> h;
      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t at = 0; at < ids.size();) {
        const std::size_t chunk = engine->PrefillRows();
        const std::size_t n = std::min<std::size_t>(chunk, ids.size() - at);
        if (!engine->Step(std::span<const std::uint32_t>(ids.data() + at, n), &h))
          return 1;
        engine->Commit(n);
        at += n;
      }
      const auto t1 = std::chrono::steady_clock::now();
      omp_set_num_threads(dec_threads);
      std::uint32_t next = engine->Argmax(h.data() + (h.size() - 2048));
      std::vector<std::uint32_t> out, hist = ids, head_draft;
      std::printf("npu> ");
      std::fflush(stdout);
      const auto emit = [&](std::uint32_t t) {
        out.push_back(t);
        hist.push_back(t);
        std::printf("%s", tokenizer->Decode(std::span<const std::uint32_t>(&t, 1)).c_str());
        std::fflush(stdout);
      };
      bool done = false;
      std::size_t passes = 0;
      while (!done && out.size() < maxnew) {
        if (next == eos || tokenizer->IsStop(next))
          break;
        emit(next);
        std::vector<std::uint32_t> fed = {next};
        if (spec) {
          const std::size_t hn = hist.size();
          for (std::size_t m = std::min<std::size_t>(5, hn - 1); m >= 2 && fed.size() == 1; --m) {
            for (std::size_t p = hn - 1; p-- > m;) {
              bool same = true;
              for (std::size_t q = 0; q < m && same; ++q)
                same = hist[p - q] == hist[hn - 1 - q];
              if (same) {
                for (std::size_t q = p + 1; q < hn && fed.size() < max_fed; ++q)
                  fed.push_back(hist[q]);
                break;
              }
            }
          }
        }
        if (spec && fed.size() == 1)
          for (auto t : head_draft)
            if (fed.size() < max_fed)
              fed.push_back(t);
        if (!engine->Step(std::span<const std::uint32_t>(fed.data(), fed.size()), &h))
          return 1;
        ++passes;
        std::vector<std::uint32_t> pred(fed.size());
        std::size_t have = 0, j = 0;
        for (;;) {
          const std::size_t want = std::min<std::size_t>(fed.size(), have == 0 ? 4 : fed.size());
          if (want > have) {
            engine->ArgmaxRows(h.data() + have * 2048, want - have, pred.data() + have);
            have = want;
          }
          while (j + 1 < fed.size() && j < have && pred[j] == fed[j + 1])
            ++j;
          if (j >= have && have < fed.size())
            continue;
          break;
        }
        for (std::size_t q = 1; q <= j && !done; ++q) {
          if (fed[q] == eos || tokenizer->IsStop(fed[q])) {
            done = true;
            break;
          }
          emit(fed[q]);
        }
        engine->Commit(static_cast<std::uint32_t>(j + 1));
        if (j >= have)
          pred[j] = 0;
        next = pred[j];
        head_draft.clear();
        if (spec && engine->HeadCount() && !done) {
          head_draft.resize(engine->HeadCount());
          engine->Draft(h.data() + j * 2048, head_draft.data());
        }
      }
      const auto t2 = std::chrono::steady_clock::now();
      std::printf("\n");
      msgs.push_back({"assistant", tokenizer->Decode(out)});
      const double pre = std::chrono::duration<double>(t1 - t0).count();
      const double dec = std::chrono::duration<double>(t2 - t1).count();
      std::fprintf(stderr, "[prompt %zu tok in %.2f s = %.0f tok/s; %zu new tok in %.2f s = %.1f tok/s]\n", ids.size(), pre,
                   ids.size() / pre, out.size(), dec, out.size() / std::max(dec, 1e-9));
    }
    return 0;
  }
  std::fprintf(stderr, "usage: gufo-npu capture|generate|chat ...\n");
  return 2;
}
