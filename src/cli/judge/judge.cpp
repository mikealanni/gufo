#include "src/cli/judge/judge.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/gguf_reader.hpp"

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

}  // namespace

int RunJudge(std::span<const char* const> args) {
  std::string model_path;
  std::string question;
  std::string context;
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
  if (!battery && !code && question.empty()) {
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
