#include "src/cli/judge/judge.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
      std::printf(
          "%s yes=%.2f no=%.2f c60=%d c90=%d  Q: %s\n  next: %s\n",
          file.filename().string().c_str(), v.yes, v.no,
          conf >= kReplayInlineCutoff, conf >= kReplayStrictCutoff, q->c_str(),
          OneLine(text.substr(pos + kReplayPiece, kReplayNext)).c_str());
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
  std::string replay;
  int repeat = 5;
  bool battery = false;
  bool use_gpu = false;
  bool code = false;
  bool batched = false;
  bool finish_verbose = false;
  std::string finish_replay;
  std::string goal_replay;
  bool goal_verbose = false;
  bool goal_bare = false;

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
