#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/inkling/gpu/engine.hpp"
#include "src/models/qwen/tokenizer.hpp"

int main(int argc, char** argv) {
  constexpr int kTop = 5;
  const std::uint32_t kContext =
      std::getenv("INKLING_CTX") != nullptr
          ? static_cast<std::uint32_t>(std::atoi(std::getenv("INKLING_CTX")))
          : 8192U;
  if (argc >= 3 && std::string(argv[2]) == "--text") {
    const int kGenerate = std::getenv("INKLING_GEN") != nullptr
                              ? std::atoi(std::getenv("INKLING_GEN"))
                              : 64;
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    if (!tok) {
      std::fprintf(stderr, "tokenizer failed: %s\n", err.c_str());
      return 1;
    }
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session = model->CreateSession(kContext, &err);
    std::string prompt_text = argc > 3 ? argv[3] : "The capital of France is";
    if (const char* pf = std::getenv("INKLING_PROMPT_FILE")) {
      std::ifstream pin(pf);
      std::stringstream pss;
      pss << pin.rdbuf();
      prompt_text = pss.str();
    }
    const auto ids = tok->Encode(prompt_text, {});
    std::vector<std::int32_t> in(ids.begin(), ids.end());
    std::printf("prompt tokens: %zu\n", in.size());
    (void)session->Append(in, &err);
    for (int i = 0; i < kGenerate; ++i) {
      const auto logits = session->Logits();
      const auto best = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      std::printf("%s", tok->DecodeTokenCopy(
                               static_cast<gufo::tokenization::TokenId>(best))
                            .c_str());
      std::fflush(stdout);
      (void)session->Append(std::span(&best, 1), &err);
    }
    std::printf("\n");
    return 0;
  }
  if (argc >= 6 && std::string(argv[2]) == "--dwindows") {
    // --dwindows <text file> <count> <len> <out.bin>: logits of the last 64
    // tokens of each window through the few-row (decode/verify) path
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int count = std::atoi(argv[4]);
    const int len = std::atoi(argv[5]);
    std::ofstream out(argv[6], std::ios::binary);
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session =
        model->CreateSession(static_cast<std::uint32_t>(len) + 16, &err);
    for (int w = 0; w < count; ++w) {
      session->Reset();
      std::vector<std::int32_t> win;
      for (int i = 0; i < len; ++i)
        win.push_back(static_cast<std::int32_t>(
            ids[(static_cast<std::size_t>(w) * len + i) % ids.size()]));
      (void)session->Append(
          std::span<const std::int32_t>(win.data(), win.size() - 64), &err);
      for (int b = 0; b < 8; ++b) {
        (void)session->AppendAll(std::span<const std::int32_t>(
                                     win.data() + win.size() - 64 + b * 8, 8),
                                 &err);
        for (std::uint32_t r = 0; r < 8; ++r) {
          const auto l = session->Logits(r);
          out.write(reinterpret_cast<const char*>(l.data()),
                    static_cast<std::streamsize>(l.size() * sizeof(float)));
        }
      }
    }
    return 0;
  }
  if (argc >= 6 && std::string(argv[2]) == "--windows") {
    // --windows <text file> <count> <len> <out.bin>: final-token logits of
    // count prose windows
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int count = std::atoi(argv[4]);
    const int len = std::atoi(argv[5]);
    std::ofstream out(argv[6], std::ios::binary);
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session =
        model->CreateSession(static_cast<std::uint32_t>(len) + 16, &err);
    for (int w = 0; w < count; ++w) {
      session->Reset();
      std::vector<std::int32_t> win;
      for (int i = 0; i < len; ++i)
        win.push_back(static_cast<std::int32_t>(
            ids[(static_cast<std::size_t>(w) * len + i) % ids.size()]));
      if (std::getenv("INKLING_PRINT_IDS") != nullptr) {
        for (const auto id : win)
          std::printf("%d ", id);
        std::printf("\n");
      }
      if (std::getenv("INKLING_WINDOW_SERIAL") != nullptr) {
        // one token per step: the decode kernels (fp32 loop control)
        for (const auto token : win) {
          if (!session->Append(std::span(&token, 1), &err))
            return 1;
        }
      } else if (!session->Append(win, &err)) {
        return 1;
      }
      const auto l = session->Logits();
      out.write(reinterpret_cast<const char*>(l.data()),
                static_cast<std::streamsize>(l.size() * sizeof(float)));
    }
    return 0;
  }
  if (argc >= 7 && std::string(argv[2]) == "--gptq") {
    // --gptq <text file> <count> <len> <out dir>: calibration prefill of count
    // windows (INKLING_GPTQ_CAPTURE lists the layers), then the GPTQ solve
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int count = std::atoi(argv[4]);
    const int len = std::atoi(argv[5]);
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session =
        model->CreateSession(static_cast<std::uint32_t>(len) + 16, &err);
    for (int w = 0; w < count; ++w) {
      session->Reset();
      std::vector<std::int32_t> win;
      for (int i = 0; i < len; ++i)
        win.push_back(static_cast<std::int32_t>(
            ids[(static_cast<std::size_t>(w) * len + i) % ids.size()]));
      if (!session->Append(win, &err)) {
        std::fprintf(stderr, "prefill failed: %s\n", err.c_str());
        return 1;
      }
      std::fprintf(stderr, "gptq calibration window %d of %d\n", w + 1, count);
    }
    if (!session->GptqFinish(argv[6], &err)) {
      std::fprintf(stderr, "gptq failed: %s\n", err.c_str());
      return 1;
    }
    return 0;
  }
  if (argc >= 5 && std::string(argv[2]) == "--draft") {
    // --draft <text file> <drafter gguf> [width]: greedy speculative loop,
    // acceptance stats
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int width = argc > 5 ? std::atoi(argv[5]) : 7;
    const int gen = std::getenv("INKLING_GEN") != nullptr
                        ? std::atoi(std::getenv("INKLING_GEN"))
                        : 128;
    int kPrefix = std::getenv("INKLING_PREFIX") != nullptr
                      ? std::atoi(std::getenv("INKLING_PREFIX"))
                      : 600;
    if (kPrefix <= 0 || kPrefix > static_cast<int>(ids.size()))
      kPrefix = static_cast<int>(ids.size());
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    if (!model->LoadDrafter(argv[4], &err)) {
      std::fprintf(stderr, "drafter load failed: %s\n", err.c_str());
      return 1;
    }
    auto session =
        model->CreateSession(std::max<std::uint32_t>(4096, kContext), &err);
    std::vector<std::int32_t> prefix;
    for (int i = 0; i < kPrefix; ++i)
      prefix.push_back(static_cast<std::int32_t>(ids[i]));
    (void)session->Append(prefix, &err);
    const auto argmax = [](std::span<const float> l) {
      int a = 0;
      for (std::size_t j = 1; j < l.size(); ++j)
        if (l[j] > l[a])
          a = static_cast<int>(j);
      return a;
    };
    int anchor = argmax(session->Logits());
    std::vector<int> hist(width + 2, 0);
    int produced = 0, steps = 0;
    double draft_ms = 0.0, verify_ms = 0.0, host_ms = 0.0;
    std::vector<int> stream;
    const auto wall0 = std::chrono::steady_clock::now();
    while (produced < gen) {
      std::vector<std::int32_t> d(width);
      std::vector<float> cf(width);
      auto t0 = std::chrono::steady_clock::now();
      if (!session->Draft(anchor, d, cf)) {
        std::fprintf(stderr, "draft failed\n");
        return 1;
      }
      auto t1 = std::chrono::steady_clock::now();
      if (const char* tau_env = std::getenv("INKLING_TAU")) {
        const float tau = static_cast<float>(std::atof(tau_env));
        int keep = 0;
        while (keep < width && cf[keep] >= tau)
          ++keep;
        d.resize(static_cast<std::size_t>(keep));
      }
      std::vector<std::int32_t> batch{anchor};
      batch.insert(batch.end(), d.begin(), d.end());
      (void)session->AppendAll(batch, &err);
      auto t2 = std::chrono::steady_clock::now();
      if (steps == 0)
        std::printf("first draft %.1f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
      draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      verify_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
      int acc = 0;
      if (std::getenv("INKLING_CONF") != nullptr) {
        int a2 = 0;
        while (a2 < width &&
               argmax(session->Logits(static_cast<std::uint32_t>(a2))) == d[a2])
          ++a2;
        for (int i = 0; i < width; ++i)
          std::printf("CONF %d %.3f %d\n", i, cf[i],
                      i < a2 ? 1 : (i == a2 ? 0 : -1));
      }
      const auto ta = std::chrono::steady_clock::now();
      while (acc < static_cast<int>(d.size()) &&
             session->Argmax(static_cast<std::uint32_t>(acc)) == d[acc])
        ++acc;
      const int next = session->Argmax(static_cast<std::uint32_t>(acc));
      host_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - ta)
                     .count();
      session->KeepRows(static_cast<std::uint32_t>(acc + 1));
      stream.push_back(anchor);
      for (int i = 0; i < acc; ++i)
        stream.push_back(d[i]);
      produced += acc + 1;
      ++hist[acc];
      ++steps;
      anchor = next;
    }
    std::printf(
        "steps %d, tokens %d, mean accepted %.2f, draft %.2f ms, verify %.2f "
        "ms\n",
        steps, produced, static_cast<double>(produced - steps) / steps,
        draft_ms / steps, verify_ms / steps);
    std::printf("host argmax %.2f ms/step, wall %.1f tok/s\n", host_ms / steps,
                produced / std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - wall0)
                               .count());
    std::printf("accepted histogram:");
    for (int i = 0; i <= width; ++i)
      std::printf(" %d", hist[i]);
    std::printf("\nsample: ");
    for (int i = 0; i < 40 && i < static_cast<int>(stream.size()); ++i)
      std::printf("%d ", stream[i]);
    std::printf("\n");
    if (std::getenv("INKLING_SHOW") != nullptr) {
      for (const int t : stream)
        std::printf("%s", tok->DecodeTokenCopy(
                                 static_cast<gufo::tokenization::TokenId>(t))
                              .c_str());
      std::printf("\n");
    }
    return 0;
  }
  if (argc >= 4 && std::string(argv[2]) == "--snap") {
    // --snap <text file> [drafter gguf]: a snapshot restored into a fresh
    // session and into the advanced original continues token for token
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int prefix_len = std::getenv("INKLING_PREFIX") != nullptr
                               ? std::atoi(std::getenv("INKLING_PREFIX"))
                               : 1500;
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    if (argc >= 5 && !model->LoadDrafter(argv[4], &err)) {
      std::fprintf(stderr, "drafter load failed: %s\n", err.c_str());
      return 1;
    }
    const std::uint32_t ctx = std::max<std::uint32_t>(4096, kContext);
    auto a = model->CreateSession(ctx, &err);
    std::vector<std::int32_t> prefix;
    for (int i = 0; i < prefix_len; ++i)
      prefix.push_back(static_cast<std::int32_t>(ids[i]));
    (void)a->Append(prefix, &err);
    const auto argmax_of = [](gufo::models::inkling::gpu::Session& s) {
      return s.Argmax();
    };
    const auto t0 = std::chrono::steady_clock::now();
    auto snapshot = a->SaveSnapshot(&err);
    const double save_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
    std::printf("snapshot %.1f MB saved in %.0f ms (declared %.1f MB)\n",
                static_cast<double>(snapshot->SizeBytes()) / 1e6, save_ms,
                static_cast<double>(a->SnapshotBytes()) / 1e6);
    const auto generate = [&](gufo::models::inkling::gpu::Session& s) {
      std::vector<std::int32_t> out;
      for (int i = 0; i < 48; ++i) {
        const std::int32_t t = argmax_of(s);
        out.push_back(t);
        (void)s.Append(std::span(&t, 1), &err);
      }
      return out;
    };
    const auto first = generate(*a);
    auto b = model->CreateSession(ctx, &err);
    const auto t1 = std::chrono::steady_clock::now();
    const bool ok_b = b->RestoreSnapshot(*snapshot, &err);
    const double restore_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t1)
                                  .count();
    const auto second = generate(*b);
    const bool ok_a = a->RestoreSnapshot(*snapshot, &err);
    const auto third = generate(*a);
    std::printf("restore into fresh session: %s in %.0f ms, tokens %s\n",
                ok_b ? "ok" : "FAILED", restore_ms,
                first == second ? "IDENTICAL" : "DIFFERENT");
    std::printf("restore into advanced session: %s, tokens %s\n",
                ok_a ? "ok" : "FAILED",
                first == third ? "IDENTICAL" : "DIFFERENT");
    return first == second && first == third && ok_a && ok_b ? 0 : 1;
  }
  if (argc >= 4 && std::string(argv[2]) == "--vtime") {
    // --vtime <text file> <k>: time of a k-row verify and a single step at
    // INKLING_PREFIX tokens of context
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int k = argc > 4 ? std::atoi(argv[4]) : 4;
    const int prefix_len = std::getenv("INKLING_PREFIX") != nullptr
                               ? std::atoi(std::getenv("INKLING_PREFIX"))
                               : 300;
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session =
        model->CreateSession(std::max<std::uint32_t>(2048, kContext), &err);
    std::vector<std::int32_t> prefix;
    for (int i = 0; i < prefix_len; ++i)
      prefix.push_back(static_cast<std::int32_t>(ids[i]));
    std::vector<std::int32_t> next;
    for (int i = 0; i < k; ++i)
      next.push_back(static_cast<std::int32_t>(ids[prefix_len + i]));
    (void)session->Append(prefix, &err);
    constexpr int kReps = 20;
    for (int pass = 0; pass < 2; ++pass) {
      auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReps; ++r) {
        (void)session->AppendAll(next, &err);
        session->Rewind(static_cast<std::uint32_t>(prefix_len));
      }
      const double ms_batch = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count() /
                              kReps;
      t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReps; ++r) {
        (void)session->Append(std::span(&next[0], 1), &err);
        session->Rewind(static_cast<std::uint32_t>(prefix_len));
      }
      const double ms_one = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count() /
                            kReps;
      if (pass == 1)
        std::printf("context %d: verify %d rows %.2f ms, single step %.2f ms\n",
                    prefix_len, k, ms_batch, ms_one);
    }
    return 0;
  }
  if (argc >= 4 && std::string(argv[2]) == "--verify") {
    // --verify <text file> <k>: AppendAll(k rows) must match k sequential
    // single-token steps
    std::string err;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
    auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    std::ifstream in(argv[3]);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->Encode(ss.str(), {});
    const int k = argc > 4 ? std::atoi(argv[4]) : 2;
    const int kPrefix = std::getenv("INKLING_PREFIX") != nullptr
                            ? std::atoi(std::getenv("INKLING_PREFIX"))
                            : 300;
    auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &err);
    auto session =
        model->CreateSession(std::max<std::uint32_t>(2048, kContext), &err);
    std::vector<std::int32_t> prefix;
    for (int i = 0; i < kPrefix; ++i)
      prefix.push_back(static_cast<std::int32_t>(ids[i]));
    std::vector<std::int32_t> next;
    for (int i = 0; i < k; ++i)
      next.push_back(static_cast<std::int32_t>(ids[kPrefix + i]));
    (void)session->Append(prefix, &err);
    std::vector<std::vector<float>> seq;
    for (int i = 0; i < k; ++i) {
      (void)session->Append(std::span(&next[i], 1), &err);
      const auto l = session->Logits();
      seq.emplace_back(l.begin(), l.end());
    }
    session->Reset();
    (void)session->Append(prefix, &err);
    if (!session->AppendAll(next, &err)) {
      std::fprintf(stderr, "AppendAll failed: %s\n", err.c_str());
      return 1;
    }
    for (int i = 0; i < k; ++i) {
      const auto l = session->Logits(static_cast<std::uint32_t>(i));
      double maxd = 0.0;
      int a0 = 0, a1 = 0;
      for (std::size_t j = 0; j < l.size(); ++j) {
        maxd = std::max(maxd, std::fabs(static_cast<double>(l[j]) - seq[i][j]));
        if (l[j] > l[a0])
          a0 = static_cast<int>(j);
        if (seq[i][j] > seq[i][a1])
          a1 = static_cast<int>(j);
      }
      std::printf("row %d: max |dlogit| %.4f, argmax batch %d seq %d\n", i,
                  maxd, a0, a1);
    }
    {
      constexpr int kReps = 30;
      session->Reset();
      (void)session->Append(prefix, &err);
      auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReps; ++r) {
        (void)session->AppendAll(next, &err);
        session->Rewind(kPrefix);
      }
      const double ms_batch = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count() /
                              kReps;
      t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReps; ++r) {
        (void)session->Append(std::span(&next[0], 1), &err);
        session->Rewind(kPrefix);
      }
      const double ms_one = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count() /
                            kReps;
      std::printf("verify %d rows: %.2f ms; single token step: %.2f ms\n", k,
                  ms_batch, ms_one);
    }
    // KeepRows: batch of k, keep j, then one more token must match j sequential
    // steps + that token
    const std::int32_t extra = static_cast<std::int32_t>(ids[kPrefix + k]);
    for (int j = 1; j < k; ++j) {
      session->Reset();
      (void)session->Append(prefix, &err);
      for (int i = 0; i < j; ++i)
        (void)session->Append(std::span(&next[i], 1), &err);
      (void)session->Append(std::span(&extra, 1), &err);
      const auto ref_l = session->Logits();
      const std::vector<float> ref(ref_l.begin(), ref_l.end());
      session->Reset();
      (void)session->Append(prefix, &err);
      (void)session->AppendAll(next, &err);
      session->KeepRows(static_cast<std::uint32_t>(j));
      (void)session->Append(std::span(&extra, 1), &err);
      const auto got = session->Logits();
      double maxd = 0.0;
      for (std::size_t q = 0; q < got.size(); ++q)
        maxd = std::max(maxd, std::fabs(static_cast<double>(got[q]) - ref[q]));
      std::printf(
          "keep %d of %d then +1 token: max |dlogit| %.4f (position %u)\n", j,
          k, maxd, session->Position());
    }
    return 0;
  }
  if (argc < 3)
    return 2;
  std::string error;
  auto model = gufo::models::inkling::gpu::Model::Load(argv[1], &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(kContext, &error);
  const bool batch = std::getenv("INKLING_BATCH") != nullptr;
  std::vector<std::int32_t> toks;
  for (int i = 2; i < argc; ++i)
    toks.push_back(std::atoi(argv[i]));
  const std::size_t step = batch ? toks.size() : 1;
  for (std::size_t i = 0; i < toks.size(); i += step) {
    if (!session->Append(std::span(toks).subspan(i, step), &error)) {
      std::fprintf(stderr, "append failed: %s\n", error.c_str());
      return 1;
    }
    const auto logits = session->Logits();
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kTop, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("pos %zu:", i + step - 1);
    for (int k = 0; k < kTop; ++k)
      std::printf(" %d(%.4f)", idx[k], logits[idx[k]]);
    std::printf("\n");
  }
  if (const char* env = std::getenv("INKLING_BENCH")) {
    constexpr int kDecode = 32;
    constexpr int kBenchToken = 1000;
    const int n = std::atoi(env);
    session->Reset();
    std::vector<std::int32_t> prompt;
    if (const char* file = std::getenv("INKLING_BENCH_FILE")) {
      std::string err;
      auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
      auto tok =
          gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
      std::ifstream in(file);
      std::stringstream ss;
      ss << in.rdbuf();
      const auto ids = tok->Encode(ss.str(), {});
      for (int i = 0; i < n; ++i)
        prompt.push_back(static_cast<std::int32_t>(
            ids[static_cast<std::size_t>(i) % ids.size()]));
    } else {
      prompt.assign(n, kBenchToken);
    }
    const char* repeat_env = std::getenv("INKLING_BENCH_REPEAT");
    for (int rep = 1; repeat_env != nullptr && rep < std::atoi(repeat_env);
         ++rep) {
      session->Reset();
      const auto r0 = std::chrono::steady_clock::now();
      if (!session->Append(prompt, &error)) {
        std::fprintf(stderr, "bench prefill failed: %s\n", error.c_str());
        return 1;
      }
      const auto r1 = std::chrono::steady_clock::now();
      std::printf("warm pp %d: %.1f tok/s\n", n,
                  n / std::chrono::duration<double>(r1 - r0).count());
    }
    session->Reset();
    const auto t0 = std::chrono::steady_clock::now();
    if (!session->Append(prompt, &error)) {
      std::fprintf(stderr, "bench prefill failed: %s\n", error.c_str());
      return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < kDecode; ++i) {
      const auto logits = session->Logits();
      const auto best = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      if (!session->Append(std::span(&best, 1), &error)) {
        std::fprintf(stderr, "bench decode failed: %s\n", error.c_str());
        return 1;
      }
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("pp %d: %.1f tok/s, tg: %.2f tok/s\n", n,
                n / std::chrono::duration<double>(t1 - t0).count(),
                kDecode / std::chrono::duration<double>(t2 - t1).count());
  }
  return 0;
}
