#include "src/cli/judge/judge.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/jev_inline.hpp"

#if defined(GUFO_NPU)
#include "src/models/qwen38_flash_next/jev_npu.hpp"
#endif
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen38_flash_next/jev_model.hpp"
#endif

namespace gufo::cli {

namespace {

// Known-answer calibration battery: expected answer, question.
struct Case {
  char expected;  // 'Y' or 'N'
  const char* question;
};

constexpr Case kTrivia[] = {
    {'Y', "Is 2 greater than 1?"},
    {'N', "Is 1 greater than 2?"},
    {'N', "The sky is green. Is that true?"},
    {'Y', "Do trees photosynthesize?"},
    {'Y', "Is water wet?"},
    {'N', "Is the moon made of cheese?"},
    {'Y', "Does 2+2 equal 4?"},
    {'Y', "Is Paris in France?"},
    {'N', "Is the Earth flat?"},
    {'Y', "Do humans have exactly two legs?"},
};

constexpr Case kCode[] = {
    {'Y', "Is Python dynamically typed?"},
    {'N', "Is C++ garbage collected by default?"},
    {'Y',
     "Can a race condition occur in multithreaded code without "
     "synchronization?"},
    {'N', "Does git commit push changes to the remote repository?"},
    {'Y', "Should a mutex be released after use?"},
    {'N', "Is HTTP stateful by default?"},
    {'Y', "Does a memory leak get worse over time?"},
    {'Y', "Can the Rust compiler produce a binary for riscv?"},
    {'Y', "Does adding an index usually speed up SELECT queries?"},
    {'N', "Is recursion free of stack overhead?"},
    {'Y', "Does TCP guarantee ordered delivery?"},
    {'N', "Can a null pointer dereference succeed in C?"},
};

void print_battery(const auto& model, const Case* cases, std::size_t count) {
  std::size_t ok = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto verdict = model.Answer(cases[i].question, "");
    const bool yes = verdict.yes >= verdict.no;
    const char got = yes ? 'Y' : 'N';
    ok += got == cases[i].expected;
    std::printf("%s exp=%c got=%c yes=%.2f no=%.2f  %s\n",
                got == cases[i].expected ? "OK  " : "MISS", cases[i].expected,
                got, verdict.yes, verdict.no, cases[i].question);
  }
  std::printf("SCORE %zu/%zu\n", ok, count);
}

#if defined(GUFO_NPU)
void print_battery_batched(const models::qwen38_flash_next::JevNpuModel& model,
                           const Case* cases, std::size_t count) {
  std::size_t ok = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t base = 0; base < count; base += 4) {
    const std::size_t take = std::min<std::size_t>(4, count - base);
    std::vector<std::pair<std::string_view, std::string_view>> items;
    for (std::size_t i = 0; i < take; ++i)
      items.emplace_back(cases[base + i].question, std::string_view{});
    const auto verdicts = model.AnswerBatch(items);
    for (std::size_t i = 0; i < take; ++i) {
      const auto& v = i < verdicts.size()
                          ? verdicts[i]
                          : models::qwen38_flash_next::JevBinaryVerdict{};
      const bool yes = v.yes >= v.no;
      const char got = yes ? 'Y' : 'N';
      ok += got == cases[base + i].expected;
      std::printf("%s exp=%c got=%c yes=%.2f no=%.2f  %s\n",
                  got == cases[base + i].expected ? "OK  " : "MISS",
                  cases[base + i].expected, got, v.yes, v.no,
                  cases[base + i].question);
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  std::printf("SCORE %zu/%zu  total=%.1f ms  avg=%.1f ms/question\n", ok, count,
              ms, ms / static_cast<double>(count));
}
#endif

constexpr std::size_t kReplayPiece = 8;
constexpr std::size_t kReplayWindow = 2048;
constexpr std::size_t kReplayContext = 400;
constexpr std::size_t kReplayNext = 200;
constexpr float kReplayInlineCutoff = 0.6F;
constexpr float kReplayStrictCutoff = 0.9F;

std::string OneLine(std::string text) {
  for (char& c : text)
    if (c == '\n' || c == '\r')
      c = ' ';
  return text;
}

// Streams each trace through the inline scanner exactly as serve does and
// scores every detected question with the same context handling.
void ReplayTraces(const auto& model, const std::string& path) {
  namespace fs = std::filesystem;
  std::vector<fs::path> files;
  if (fs::is_directory(path)) {
    for (const auto& e : fs::directory_iterator(path))
      if (e.is_regular_file())
        files.push_back(e.path());
    std::sort(files.begin(), files.end());
  } else {
    files.emplace_back(path);
  }
  std::size_t total = 0, inline_ok = 0, strict_ok = 0, yes = 0, repeats = 0;
  for (const auto& file : files) {
    std::ifstream in(file);
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    models::qwen38_flash_next::JevInlineScanner scanner;
    std::string ctx;
    std::string last_q;
    std::size_t found = 0;
    for (std::size_t pos = 0; pos < text.size(); pos += kReplayPiece) {
      const std::string piece = text.substr(pos, kReplayPiece);
      ctx += piece;
      if (ctx.size() > kReplayWindow)
        ctx.erase(0, ctx.size() - kReplayWindow);
      const auto q = scanner.Push(piece);
      if (!q)
        continue;
      if (*q == last_q) {
        ++repeats;
        std::printf("STOP repeat  %s\n", q->c_str());
        break;
      }
      std::string c = ctx;
      if (c.size() > kReplayContext)
        c = c.substr(c.size() - kReplayContext);
      const auto v = model.Answer(*q, c);
      const float conf = std::max(v.yes, v.no);
      ++found;
      ++total;
      inline_ok += conf >= kReplayInlineCutoff;
      strict_ok += conf >= kReplayStrictCutoff;
      yes += v.yes >= v.no;
      last_q = *q;
      std::printf("%s yes=%.2f no=%.2f c60=%d c90=%d  Q: %s\n  next: %s\n",
                  file.filename().string().c_str(), v.yes, v.no,
                  conf >= kReplayInlineCutoff, conf >= kReplayStrictCutoff,
                  q->c_str(),
                  OneLine(text.substr(pos + kReplayPiece, kReplayNext))
                      .c_str());
    }
    std::printf("TRACE %s detections=%zu\n", file.filename().string().c_str(),
                found);
  }
  const auto pct = [&](std::size_t n) {
    return total ? 100.0 * static_cast<double>(n) / static_cast<double>(total)
                 : 0.0;
  };
  std::printf(
      "REPLAY traces=%zu detections=%zu clear0.6=%.1f%% clear0.9=%.1f%% "
      "yes=%.1f%% no=%.1f%% repeat_stops=%zu\n",
      files.size(), total, pct(inline_ok), pct(strict_ok), pct(yes),
      total ? 100.0 - pct(yes) : 0.0, repeats);
}

}  // namespace

int RunJudge(std::span<const char* const> args) {
  std::string model_path;
  std::string question;
  std::string context;
  std::string replay;
  int repeat = 5;
  bool battery = false;
  bool use_gpu = false;
  bool code = false;
  bool batched = false;

  ArgParser parser("gufo judge",
                   "Run the resident NPU judge on one question or the "
                   "calibration battery.");
  parser.AddOption("", "--model", "PATH", "Small GGUF judge model", "Model",
                   &model_path);
  parser.AddOption("-q", "--question", "TEXT", "Question to score", "Question",
                   &question);
  parser.AddOption("", "--context", "TEXT", "Optional preceding context",
                   "Question", &context);
  parser.AddOption("", "--repeat", "N", "Timed repetitions (default 5)",
                   "Timing", &repeat);
  parser.AddFlag("", "--battery", "Run the known-answer trivia battery",
                 "Battery", &battery);
  parser.AddFlag("", "--code", "Run the known-answer code battery", "Battery",
                 &code);
  parser.AddOption("", "--replay", "PATH",
                   "Replay thinking traces (file or dir) through the inline "
                   "scanner and judge",
                   "Battery", &replay);
  parser.AddFlag("", "--gpu", "Use the GPU judge instead of the NPU", "Model",
                 &use_gpu);
  parser.AddFlag(
      "", "--batched",
      "Run --battery/--code through AnswerBatch (4 questions/NPU pass)",
      "Battery", &batched);
  if (!parser.Parse(args))
    return 2;

  if (model_path.empty()) {
    std::fprintf(stderr, "judge: --model is required\n");
    return 2;
  }
  if (!battery && !code && question.empty() && replay.empty()) {
    std::fprintf(stderr, "judge: --question, --battery or --code required\n");
    return 2;
  }

  std::string err;
  auto reader = core::GgufReader::OpenFile(model_path, &err);
  if (!reader) {
    std::fprintf(stderr, "judge: %s\n", err.c_str());
    return 1;
  }

#if defined(GUFO_NPU)
  if (!use_gpu) {
    auto model =
        models::qwen38_flash_next::JevNpuModel::LoadFromGguf(*reader, &err);
    if (!model) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    if (!question.empty()) {
      for (int i = 0; i < repeat; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto verdict = model->Answer(question, context);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        std::printf("%s yes=%.3f no=%.3f  %.1f ms\n",
                    verdict.yes >= verdict.no ? "Yes" : "No", verdict.yes,
                    verdict.no, ms);
      }
    }
    if (!replay.empty())
      ReplayTraces(*model, replay);
    if (battery)
      batched ? print_battery_batched(*model, kTrivia, std::size(kTrivia))
              : print_battery(*model, kTrivia, std::size(kTrivia));
    if (code)
      batched ? print_battery_batched(*model, kCode, std::size(kCode))
              : print_battery(*model, kCode, std::size(kCode));
    return 0;
  }
#endif
#if defined(ENGINE_ENABLE_HIP)
  if (use_gpu) {
    auto model =
        models::qwen38_flash_next::JevModel::LoadFromGguf(*reader, &err);
    if (!model) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    if (!question.empty()) {
      for (int i = 0; i < repeat; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto verdict = model->Answer(question, context);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        std::printf("%s yes=%.3f no=%.3f  %.1f ms\n",
                    verdict.yes >= verdict.no ? "Yes" : "No", verdict.yes,
                    verdict.no, ms);
      }
    }
    if (battery)
      print_battery(*model, kTrivia, std::size(kTrivia));
    if (code)
      print_battery(*model, kCode, std::size(kCode));
    return 0;
  }
#endif
  std::fprintf(stderr, "judge: no judge backend available in this build\n");
  return 1;
}

}  // namespace gufo::cli
