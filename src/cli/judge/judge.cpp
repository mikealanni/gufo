#include "src/cli/judge/judge.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/judge/goal_replay.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/jev_finish.hpp"

#if defined(GUFO_NPU)
#include <hip/hip_runtime.h>

#include "src/models/lfm3/tokenizer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
#include "src/models/qwen38_flash_next/d1_npu.hpp"
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

#if defined(GUFO_NPU)
std::string Unescape(std::string_view text) {
  std::string out;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '\\' || i + 1 == text.size()) {
      out.push_back(text[i]);
      continue;
    }
    const char next = text[++i];
    out.push_back(next == 'n' ? '\n' : next == 't' ? '\t' : next);
  }
  return out;
}

int ServeStdio(const models::qwen38_flash_next::JevNpuModel& model) {
  std::printf("READY\n");
  std::fflush(stdout);
  std::string line;
  while (std::getline(std::cin, line)) {
    const auto tab = line.find('\t');
    const std::string question =
        Unescape(std::string_view(line).substr(0, tab));
    const std::string context =
        tab == std::string::npos
            ? std::string()
            : Unescape(std::string_view(line).substr(tab + 1));
    const auto t0 = std::chrono::steady_clock::now();
    const auto verdict = model.Answer(question, context);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::printf("%s %.3f %.3f %.1f\n", verdict.yes >= verdict.no ? "Yes" : "No",
                verdict.yes, verdict.no, ms);
    std::fflush(stdout);
  }
  return 0;
}
#endif

// One labeled turn of the finish-detector corpus: a header of `key: value`
// lines, then `@@task` and `@@response` sections.
struct FinishDoc {
  std::string name, label, source, finish, task, response;
  bool tool{false};
  bool synthetic{false};
};

bool LoadFinishDoc(const std::filesystem::path& path, FinishDoc* doc) {
  std::ifstream in(path);
  if (!in)
    return false;
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const auto task_at = text.find("\n@@task\n");
  const auto response_at = text.find("\n@@response\n");
  if (task_at == std::string::npos || response_at == std::string::npos)
    return false;
  std::istringstream header(text.substr(0, task_at));
  for (std::string line; std::getline(header, line);) {
    const auto colon = line.find(": ");
    if (colon == std::string::npos)
      continue;
    const std::string key = line.substr(0, colon);
    const std::string value = line.substr(colon + 2);
    if (key == "label")
      doc->label = value;
    else if (key == "source")
      doc->source = value;
    else if (key == "finish")
      doc->finish = value;
    else if (key == "tool")
      doc->tool = value == "1";
    else if (key == "synthetic")
      doc->synthetic = value == "1";
  }
  constexpr std::size_t kTaskSkip = 8;
  constexpr std::size_t kResponseSkip = 12;
  doc->task =
      text.substr(task_at + kTaskSkip, response_at - task_at - kTaskSkip);
  doc->response = text.substr(response_at + kResponseSkip);
  doc->name = path.filename().string();
  return true;
}

std::string OneLineTail(const std::string& text) {
  constexpr std::size_t kTailChars = 140;
  std::string tail =
      text.size() > kTailChars ? text.substr(text.size() - kTailChars) : text;
  for (auto& c : tail)
    c = c == '\n' ? ' ' : c;
  return tail;
}

// Scores the rule-based finish detector against a labeled corpus directory.
int RunFinishReplay(const std::string& dir, bool verbose) {
  namespace flash = models::qwen38_flash_next;
  struct Tally {
    int fires{0}, tp{0}, fp{0};
  };
  std::vector<FinishDoc> docs;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    FinishDoc doc;
    if (LoadFinishDoc(entry.path(), &doc))
      docs.push_back(std::move(doc));
  }
  std::map<std::string, Tally> rules;
  std::map<std::string, std::array<int, 4>> sources;  // tp fp fn tn
  for (const auto& doc : docs) {
    const bool positive = doc.label == "unfinished";
    const auto hits = flash::EvaluateFinishRules(doc.task, doc.response,
                                                 doc.tool, doc.finish);
    const bool fired =
        !hits.empty() && hits.front().score >= flash::kJevFinishThreshold;
    for (const auto& hit : hits) {
      if (hit.score < flash::kJevFinishThreshold)
        continue;
      auto& tally = rules[hit.rule];
      ++tally.fires;
      ++(positive ? tally.tp : tally.fp);
    }
    auto& cell = sources[doc.source + (doc.synthetic ? "(syn)" : "")];
    ++cell[fired ? (positive ? 0 : 1) : (positive ? 2 : 3)];
    if (verbose && fired != positive && !doc.synthetic) {
      std::printf("%s %s [%s] ...%s\n", fired ? "FP" : "FN", doc.name.c_str(),
                  fired ? hits.front().rule.c_str() : doc.finish.c_str(),
                  OneLineTail(doc.response).c_str());
    }
  }
  std::printf("docs=%zu threshold=%.2f\n", docs.size(),
              flash::kJevFinishThreshold);
  for (const auto& [rule, tally] : rules) {
    std::printf(
        "rule %-10s TP=%d FP=%d precision=%.3f\n", rule.c_str(), tally.tp,
        tally.fp,
        tally.fires ? static_cast<double>(tally.tp) / tally.fires : 0.0);
  }
  int tp = 0, fp = 0, fn = 0, tn = 0;
  for (const auto& [source, c] : sources) {
    std::printf("%-16s TP=%d FP=%d FN=%d TN=%d\n", source.c_str(), c[0], c[1],
                c[2], c[3]);
    tp += c[0];
    fp += c[1];
    fn += c[2];
    tn += c[3];
  }
  std::printf("ALL TP=%d FP=%d FN=%d TN=%d precision=%.3f recall=%.3f\n", tp,
              fp, fn, tn, tp + fp ? static_cast<double>(tp) / (tp + fp) : 0.0,
              tp + fn ? static_cast<double>(tp) / (tp + fn) : 0.0);
  return 0;
}

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
  bool pack_test = false;
  bool npu_scale_bench = false;
  std::string generate_prompt;
  int max_new = 64;
  int draft_len = 15;
  bool finish_verbose = false;
  std::string finish_replay;
  std::string goal_replay;
  bool goal_verbose = false;
  bool goal_bare = false;
  bool stdio = false;

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
  parser.AddOption("", "--generate", "TEXT",
                   "Greedy chat reply on the NPU (LFM3 causal models)",
                   "Generate", &generate_prompt);
  parser.AddOption("", "--max-new", "N", "Tokens to generate (default 64)",
                   "Generate", &max_new);
  parser.AddOption("", "--draft", "N",
                   "Prompt-lookup draft tokens per pass, 0 = off (default 15)",
                   "Generate", &draft_len);
  parser.AddFlag("", "--npu-scale-bench",
                 "Time the K=2048,N=1024 NPU GEMM at M=512..4096 (compute vs overhead)",
                 "Battery", &npu_scale_bench);
  parser.AddFlag("", "--pack-test",
                 "Compare four different-context decisions: one by one vs one packed pass",
                 "Battery", &pack_test);
  parser.AddFlag(
      "", "--batched",
      "Run --battery/--code through AnswerBatch (4 questions/NPU pass)",
      "Battery", &batched);
  parser.AddOption("", "--finish-replay", "DIR",
                   "Score the rule-based finish detector on a labeled corpus",
                   "Finish", &finish_replay);
  parser.AddOption("", "--goal-replay", "DIR",
                   "Score the goal tracker on exported opencode sessions",
                   "Goal", &goal_replay);
  parser.AddFlag("", "--goal-verbose", "List goal-replay false continues",
                 "Goal", &goal_verbose);
  parser.AddFlag("", "--goal-bare",
                 "Goal replay: also replay a bare trigger phrase as the "
                 "latest message",
                 "Goal", &goal_bare);
  parser.AddFlag("", "--stdio",
                 "Serve questions from stdin: one 'question<TAB>context' line "
                 "each (\\n, \\t, \\\\ escaped), one verdict line back",
                 "Serve", &stdio);
  parser.AddFlag("", "--finish-verbose", "List finish-replay misses", "Finish",
                 &finish_verbose);
  if (!parser.Parse(args))
    return 2;

  if (!goal_replay.empty())
    return RunGoalReplay(goal_replay, goal_verbose, goal_bare);
  if (!finish_replay.empty())
    return RunFinishReplay(finish_replay, finish_verbose);

  if (model_path.empty()) {
    std::fprintf(stderr, "judge: --model is required\n");
    return 2;
  }
  if (npu_scale_bench) {
#if defined(GUFO_NPU)
    const char* dir_env = std::getenv("GUFO_NPU_DIR");
    const std::string dir = dir_env ? dir_env : "tools/npu";
    hipStream_t stream;
    (void)hipStreamCreate(&stream);
    for (std::uint32_t m : {512u, 1024u, 2048u, 4096u}) {
      models::qwen38_flash_next::rocm::NpuGemm::Config cfg;
      cfg.xclbin = dir + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin";
      cfg.insts = dir + "/runtime_mk/insts_" + std::to_string(m) +
                  "x2048x1024_128x64x128.txt";
      cfg.tokens = m;
      cfg.k = 2048;
      cfg.n = 1024;
      std::string e;
      auto gemm = models::qwen38_flash_next::rocm::NpuGemm::Create(cfg, &e);
      if (!gemm) {
        std::printf("M=%u: %s\n", m, e.c_str());
        continue;
      }
      const int a = gemm->NewBuffer(gemm->ABytes(), nullptr);
      const int b = gemm->NewBuffer(gemm->BBytes(), nullptr);
      const int c = gemm->NewBuffer(gemm->CBytes(), nullptr);
      std::memset(gemm->Host(a), 0, gemm->ABytes());
      std::memset(gemm->Host(b), 0, gemm->BBytes());
      gemm->FinalizeWeights(a);
      gemm->FinalizeWeights(b);
      const int bs[1] = {b};
      const int cs[1] = {c};
      for (int i = 0; i < 5; ++i) {
        gemm->Submit(stream, a, bs, cs, 1, 0);
        gemm->Wait(stream);
        (void)hipStreamSynchronize(stream);
      }
      const int reps = 40;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) {
        gemm->Submit(stream, a, bs, cs, 1, 0);
        gemm->Wait(stream);
        (void)hipStreamSynchronize(stream);
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count() /
                        reps;
      const double flop = 2.0 * m * 2048.0 * 1024.0;
      std::printf("M=%u K=2048 N=1024: %.3f ms/call  (%.2f TFLOP/s, weights %.1f MB)\n",
                  m, ms, flop / ms / 1e9, gemm->BBytes() / 1e6);
    }
    return 0;
#endif
  }
  if (!stdio && !battery && !code && !pack_test && question.empty() &&
      generate_prompt.empty()) {
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
  if (!use_gpu && !generate_prompt.empty()) {
    auto model = models::qwen38_flash_next::D1NpuModel::Create(*reader, &err);
    auto tokenizer = models::lfm3::Tokenizer::Load(*reader, &err);
    if (!model || !tokenizer || !model->EnableGeneration(std::getenv("GUFO_D1_CTX") ? static_cast<std::uint32_t>(std::atoi(std::getenv("GUFO_D1_CTX"))) : 8192u, &err)) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    if (!generate_prompt.empty() && generate_prompt[0] == '@') {
      std::ifstream file(generate_prompt.substr(1));
      std::stringstream buffer;
      buffer << file.rdbuf();
      generate_prompt = buffer.str();
    }
    const std::vector<std::pair<std::string, std::string>> messages = {
        {"user", generate_prompt}};
    const auto ids = tokenizer->Encode(
        models::lfm3::Tokenizer::Render(messages, false));
    models::qwen38_flash_next::D1NpuModel::GenerateOptions options;
    options.max_new = static_cast<std::uint32_t>(max_new);
    options.draft_max = static_cast<std::uint32_t>(std::max(0, draft_len));
    if (auto eos = reader->GetMetadataUint32("tokenizer.ggml.eos_token_id"))
      options.stop.push_back(*eos);
    for (int i = 0; i < repeat; ++i) {
      std::vector<std::uint32_t> out;
      models::qwen38_flash_next::D1NpuModel::GenerateStats stats;
      if (!model->Generate(ids, options, &out, &stats)) {
        std::fprintf(stderr, "judge: generation failed\n");
        return 1;
      }
      if (i == 0) {
        std::printf("%s\n", tokenizer->Decode(out).c_str());
        std::printf("ids:");
        for (auto id : out)
          std::printf(" %u", id);
        std::printf("\n");
      }
      std::printf(
          "pp %u tok %.1f ms (%.0f tok/s) | tg %u tok %.1f ms (%.1f tok/s) | "
          "passes %u drafted %u accepted %u\n",
          stats.prompt_tokens, stats.prefill_ms,
          stats.prompt_tokens * 1000.0 / stats.prefill_ms, stats.new_tokens,
          stats.decode_ms, stats.new_tokens * 1000.0 / stats.decode_ms,
          stats.passes, stats.drafted, stats.accepted);
    }
    return 0;
  }
  if (!use_gpu) {
    auto model =
        models::qwen38_flash_next::JevNpuModel::LoadFromGguf(*reader, &err);
    if (!model) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    if (stdio)
      return ServeStdio(*model);
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
    if (pack_test) {
      const std::vector<std::pair<std::string, std::string>> owned = {
          {"Will this command use the GPU or most of the machine's RAM?",
           "The agent is about to run: python train.py --epochs 50 on a 20GB model."},
          {"Is this a quick read-only shell command?",
           "The agent is about to run: ls -la /tmp"},
          {"Will this command use the GPU or most of the machine's RAM?",
           "The agent is about to run: cargo build --release for a large workspace."},
          {"Is this a quick read-only shell command?",
           "The agent is about to run: git status"}};
      std::vector<std::pair<std::string_view, std::string_view>> items;
      for (const auto& [q, c] : owned)
        items.emplace_back(q, c);
      for (int round = 0; round < 3; ++round) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<models::qwen38_flash_next::JevBinaryVerdict> single;
        for (const auto& item : items)
          single.push_back(model->Answer(item.first, item.second));
        const auto t1 = std::chrono::steady_clock::now();
        const auto packed = model->AnswerBatch(items);
        const auto t2 = std::chrono::steady_clock::now();
        std::printf("round %d: singles %.1f ms, packed %.1f ms\n", round,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count());
        if (round == 2)
          for (std::size_t i = 0; i < items.size(); ++i)
            std::printf("  %zu single yes=%.4f packed yes=%.4f\n", i,
                        single[i].yes,
                        i < packed.size() ? packed[i].yes : -1.0F);
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
