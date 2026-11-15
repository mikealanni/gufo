#include "src/cli/judge/judge.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "src/models/qwen38_flash_next/d1_npu.hpp"
#include "src/models/qwen38_flash_next/jev_npu.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
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
  bool row_test = false;
  std::string capture_in, capture_out, drafter_path, gen_data_in, gen_data_out;
  int gen_streams = 96, gen_len = 192;
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
  parser.AddOption("", "--capture", "FILE",
                   "Teacher-forced hidden states of documents (split by a line "
                   "=====DOC=====)",
                   "Generate", &capture_in);
  parser.AddOption("", "--gen-data", "FILE",
                   "Greedy self-generated continuations of prompts (docs split "
                   "by =====DOC=====)",
                   "Generate", &gen_data_in);
  parser.AddOption("", "--gen-out", "PREFIX", "Output prefix for --gen-data",
                   "Generate", &gen_data_out);
  parser.AddOption("", "--streams", "N", "Parallel streams for --gen-data",
                   "Generate", &gen_streams);
  parser.AddOption("", "--gen-len", "N", "Tokens per stream for --gen-data",
                   "Generate", &gen_len);
  parser.AddOption("", "--drafter", "FILE", "Drafter heads file for --generate",
                   "Generate", &drafter_path);
  parser.AddOption("", "--capture-out", "PREFIX",
                   "Output prefix: PREFIX.tokens (u32), PREFIX.hidden (fp16), "
                   "PREFIX.docs (u64 starts)",
                   "Generate", &capture_out);
  parser.AddFlag("", "--row-test",
                 "Compare one multi-row Step against sequential one-row Steps",
                 "Generate", &row_test);
  parser.AddFlag(
      "", "--npu-scale-bench",
      "Time the K=2048,N=1024 NPU GEMM at M=512..4096 (compute vs overhead)",
      "Battery", &npu_scale_bench);
  parser.AddFlag(
      "", "--pack-test",
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
    struct Shape {
      std::uint32_t tile, k, n;
    };
    for (const Shape sh : {Shape{128, 2048, 1024}, Shape{64, 2048, 1024},
                           Shape{128, 2048, 6144}, Shape{64, 2048, 6144},
                           Shape{128, 2048, 15360}, Shape{64, 2048, 15360},
                           Shape{128, 7680, 2048}, Shape{64, 7680, 2048}}) {
      const std::uint32_t m = 4 * sh.tile;
      const std::string tag = std::to_string(sh.tile) + "x64x128";
      auto insts = [&](std::uint32_t k, std::uint32_t n) {
        return dir + "/runtime_mk/insts_" + std::to_string(m) + "x" +
               std::to_string(k) + "x" + std::to_string(n) + "_" + tag + ".txt";
      };
      models::qwen38_flash_next::rocm::NpuGemm::Config cfg;
      cfg.xclbin = dir + "/runtime_mk/final_4096xKx1024_" + tag + ".xclbin";
      cfg.insts = insts(2048, 1024);
      cfg.tokens = m;
      cfg.k = 2048;
      cfg.n = 1024;
      std::string e;
      auto gemm = models::qwen38_flash_next::rocm::NpuGemm::Create(cfg, &e);
      if (!gemm) {
        std::printf("tile %u: %s\n", sh.tile, e.c_str());
        continue;
      }
      const int prog = gemm->AddProgram(insts(sh.k, sh.n), &e);
      if (prog < 0) {
        std::printf("tile %u K=%u N=%u: %s\n", sh.tile, sh.k, sh.n, e.c_str());
        continue;
      }
      const int a = gemm->NewBuffer(std::size_t{m} * sh.k / 8 * 9, nullptr);
      const int b = gemm->NewBuffer(std::size_t{sh.k} * sh.n / 8 * 9, nullptr);
      const int c = gemm->NewBuffer(std::size_t{m} * sh.n / 8 * 9, nullptr);
      std::memset(gemm->Host(a), 0, std::size_t{m} * sh.k / 8 * 9);
      std::memset(gemm->Host(b), 0, std::size_t{sh.k} * sh.n / 8 * 9);
      gemm->FinalizeWeights(a);
      gemm->FinalizeWeights(b);
      const int bs[1] = {b};
      const int cs[1] = {c};
      for (int i = 0; i < 4; ++i) {
        gemm->Submit(stream, a, bs, cs, 1, prog);
        gemm->Wait(stream);
        (void)hipStreamSynchronize(stream);
      }
      const int reps = 30;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) {
        gemm->Submit(stream, a, bs, cs, 1, prog);
        gemm->Wait(stream);
        (void)hipStreamSynchronize(stream);
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count() /
                        reps;
      const double mb = sh.k * sh.n / 8.0 * 9 / 1e6;
      std::printf(
          "tile %3u M=%u K=%u N=%u: %.3f ms  (%.1f TFLOP/s, weights %.1f MB = "
          "%.1f GB/s)\n",
          sh.tile, m, sh.k, sh.n, ms, 2.0 * m * sh.k * sh.n / ms / 1e9, mb,
          mb / ms);
    }
    return 0;
#endif
  }
  if (!stdio && !battery && !code && !pack_test && question.empty() &&
      generate_prompt.empty() && capture_in.empty() && gen_data_in.empty() &&
      !row_test) {
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
  if (!use_gpu && row_test) {
    auto model = models::qwen38_flash_next::D1NpuModel::Create(*reader, &err);
    auto tokenizer = models::lfm3::Tokenizer::Load(*reader, &err);
    if (!model || !tokenizer || !model->EnableGeneration(4096, &err)) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    const std::vector<std::pair<std::string, std::string>> messages = {
        {"user", "Explain in a few sentences why the sky is blue."}};
    const auto ids =
        tokenizer->Encode(models::lfm3::Tokenizer::Render(messages, false));
    const std::uint32_t rows =
        std::getenv("GUFO_ROWS")
            ? static_cast<std::uint32_t>(std::atoi(std::getenv("GUFO_ROWS")))
            : 8u;
    std::vector<std::uint32_t> seq;
    std::vector<std::vector<float>> hid_seq(rows);
    std::vector<std::uint32_t> argmax_seq(rows);
    model->ResetState();
    std::uint32_t nextv = 0;
    (void)model->Step(ids, std::span(&nextv, 1));
    model->Commit(static_cast<std::uint32_t>(ids.size()));
    std::uint32_t cur = nextv;
    for (std::uint32_t i = 0; i < rows; ++i) {
      seq.push_back(cur);
      (void)model->Step(std::span<const std::uint32_t>(&seq.back(), 1),
                        std::span(&argmax_seq[i], 1));
      (void)model->LastHidden(&hid_seq[i], 1);
      model->Commit(1);
      cur = argmax_seq[i];
    }
    model->ResetState();
    (void)model->Step(ids, std::span(&nextv, 1));
    model->Commit(static_cast<std::uint32_t>(ids.size()));
    std::vector<std::uint32_t> argmax_multi(rows);
    (void)model->Step(seq, argmax_multi);
    std::vector<float> hid_multi;
    (void)model->LastHidden(&hid_multi, rows);
    {
      std::ofstream ids_out("/tmp/rowtest_ids.txt");
      for (auto id : ids)
        ids_out << id << ' ';
      ids_out << '\n';
      for (auto id : seq)
        ids_out << id << ' ';
      ids_out << '\n';
      std::ofstream a("/tmp/rowtest_seq.f32", std::ios::binary),
          b("/tmp/rowtest_multi.f32", std::ios::binary);
      for (std::uint32_t i = 0; i < rows; ++i)
        a.write(reinterpret_cast<const char*>(hid_seq[i].data()),
                hid_seq[i].size() * 4);
      b.write(reinterpret_cast<const char*>(hid_multi.data()),
              hid_multi.size() * 4);
    }
    if (!drafter_path.empty() && model->LoadDrafter(drafter_path, &err)) {
      for (std::uint32_t i = 0; i < rows; ++i) {
        std::vector<std::uint32_t> d;
        (void)model->Draft(i, &d);
        std::printf("draft row %u:", i);
        for (auto t : d)
          std::printf(" %u", t);
        std::printf("\n");
      }
    }
    for (std::uint32_t i = 0; i < rows; ++i) {
      double worst = 0, scale = 0;
      for (std::size_t j = 0; j < hid_seq[i].size(); ++j) {
        worst = std::max(
            worst, double(std::abs(hid_seq[i][j] -
                                   hid_multi[i * hid_seq[i].size() + j])));
        scale = std::max(scale, double(std::abs(hid_seq[i][j])));
      }
      std::printf(
          "row %u: argmax seq %u multi %u  hidden max|diff| %.5f (max|h| "
          "%.3f)\n",
          i, argmax_seq[i], argmax_multi[i], worst, scale);
    }
    return 0;
  }
  if (!use_gpu && !gen_data_in.empty()) {
    auto model = models::qwen38_flash_next::D1NpuModel::Create(*reader, &err);
    auto tokenizer = models::lfm3::Tokenizer::Load(*reader, &err);
    const std::uint32_t streams = static_cast<std::uint32_t>(gen_streams);
    const std::uint32_t max_prompt = 384;
    const std::uint32_t ctx =
        max_prompt + static_cast<std::uint32_t>(gen_len) + 8;
    if (!model || !tokenizer || !model->EnableGeneration(ctx, &err) ||
        !model->EnableBatch(streams, ctx, &err)) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    std::ifstream file(gen_data_in);
    std::string line, doc;
    std::vector<std::string> docs;
    while (std::getline(file, line)) {
      if (line == "=====DOC=====") {
        if (!doc.empty())
          docs.push_back(doc);
        doc.clear();
      } else {
        doc += line;
        doc += '\n';
      }
    }
    for (auto& d : docs)
      if (!d.empty() && d.back() == '\n')
        d.pop_back();
    std::uint32_t im_end = 0;
    {
      const auto e = tokenizer->Encode("<|im_end|>");
      if (!e.empty())
        im_end = e[0];
    }
    std::ofstream tok_out(gen_data_out + ".tokens", std::ios::binary);
    std::ofstream doc_out(gen_data_out + ".docs", std::ios::binary);
    std::uint64_t total = 0, generated = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t base = 0; base < docs.size(); base += streams) {
      const std::size_t count =
          std::min<std::size_t>(streams, docs.size() - base);
      std::vector<std::vector<std::uint32_t>> seqs(count);
      std::vector<std::uint32_t> cur(count), nxt(count);
      std::vector<char> done(count, 0);
      for (std::size_t i = 0; i < count; ++i) {
        auto ids = tokenizer->Encode(docs[base + i]);
        if (ids.size() > max_prompt)
          ids.erase(ids.begin() + 1,
                    ids.end() - (max_prompt - 1));  // keep BOS + tail
        std::uint32_t first = 0;
        model->ResetState();
        for (std::size_t at = 0; at < ids.size();) {
          const std::size_t take =
              std::min<std::size_t>(model->PassRows(), ids.size() - at);
          if (!model->Step(
                  std::span<const std::uint32_t>(ids).subspan(at, take),
                  std::span(&first, 1))) {
            std::fprintf(stderr, "judge: prefill failed\n");
            return 1;
          }
          model->Commit(static_cast<std::uint32_t>(take));
          at += take;
        }
        if (!model->BatchLoad(static_cast<std::uint32_t>(i),
                              static_cast<std::uint32_t>(ids.size()))) {
          std::fprintf(stderr, "judge: batch load failed\n");
          return 1;
        }
        seqs[i] = std::move(ids);
        cur[i] = first;
      }
      std::vector<std::size_t> prompt_len(count);
      for (std::size_t i = 0; i < count; ++i) {
        prompt_len[i] = seqs[i].size();
        seqs[i].push_back(cur[i]);
        done[i] = cur[i] == im_end;
      }
      for (int step = 1; step < gen_len; ++step) {
        if (!model->BatchStep(std::span<const std::uint32_t>(cur.data(), count),
                              std::span<std::uint32_t>(nxt.data(), count))) {
          std::fprintf(stderr, "judge: batch step failed\n");
          return 1;
        }
        for (std::size_t i = 0; i < count; ++i) {
          cur[i] = nxt[i];
          if (!done[i]) {
            seqs[i].push_back(nxt[i]);
            done[i] = nxt[i] == im_end;
          }
        }
        bool all_done = true;
        for (char d : done)
          all_done = all_done && d;
        if (all_done)
          break;
      }
      for (std::size_t i = 0; i < count; ++i) {
        doc_out.write(reinterpret_cast<const char*>(&total), sizeof(total));
        tok_out.write(reinterpret_cast<const char*>(seqs[i].data()),
                      seqs[i].size() * 4);
        total += seqs[i].size();
        generated += seqs[i].size() - prompt_len[i];
        // reset this stream slot's batch position for the next group
      }
      std::fprintf(
          stderr, "gen: %zu/%zu docs, %llu generated tokens, %.0f tok/s\n",
          base + count, docs.size(), static_cast<unsigned long long>(generated),
          generated / std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count());
    }
    std::printf("generated %llu tokens (%llu total) from %zu docs\n",
                static_cast<unsigned long long>(generated),
                static_cast<unsigned long long>(total), docs.size());
    return 0;
  }
  if (!use_gpu && !capture_in.empty()) {
    auto model = models::qwen38_flash_next::D1NpuModel::Create(*reader, &err);
    auto tokenizer = models::lfm3::Tokenizer::Load(*reader, &err);
    if (!model || !tokenizer || !model->EnableGeneration(8192, &err)) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    std::ifstream file(capture_in);
    std::string line, doc;
    std::vector<std::string> docs;
    std::vector<std::vector<std::uint32_t>> token_docs;
    if (capture_in.size() > 7 &&
        capture_in.substr(capture_in.size() - 7) == ".tokens") {
      // Pre-tokenized documents: PREFIX.tokens (u32) split by PREFIX.docs (u64
      // starts).
      const std::string prefix = capture_in.substr(0, capture_in.size() - 7);
      std::ifstream tokens_file(capture_in, std::ios::binary | std::ios::ate);
      const std::size_t token_count =
          static_cast<std::size_t>(tokens_file.tellg()) / 4;
      std::vector<std::uint32_t> all(token_count);
      tokens_file.seekg(0);
      tokens_file.read(reinterpret_cast<char*>(all.data()), token_count * 4);
      std::ifstream starts_file(prefix + ".docs",
                                std::ios::binary | std::ios::ate);
      const std::size_t doc_count =
          static_cast<std::size_t>(starts_file.tellg()) / 8;
      std::vector<std::uint64_t> starts(doc_count);
      starts_file.seekg(0);
      starts_file.read(reinterpret_cast<char*>(starts.data()), doc_count * 8);
      for (std::size_t d = 0; d < doc_count; ++d) {
        const std::size_t lo = starts[d];
        const std::size_t hi = d + 1 < doc_count ? starts[d + 1] : token_count;
        token_docs.emplace_back(all.begin() + lo, all.begin() + hi);
      }
    }
    while (std::getline(file, line)) {
      if (line == "=====DOC=====") {
        if (!doc.empty())
          docs.push_back(doc);
        doc.clear();
      } else {
        doc += line;
        doc += '\n';
      }
    }
    if (!doc.empty())
      docs.push_back(doc);
    std::ofstream tok_out(capture_out + ".tokens", std::ios::binary);
    std::ofstream hid_out(capture_out + ".hidden", std::ios::binary);
    std::ofstream doc_out(capture_out + ".docs", std::ios::binary);
    std::uint64_t total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t doc_total =
        token_docs.empty() ? docs.size() : token_docs.size();
    for (std::size_t d = 0; d < doc_total; ++d) {
      auto ids =
          token_docs.empty() ? tokenizer->Encode(docs[d]) : token_docs[d];
      if (ids.size() < 8)
        continue;
      const std::size_t cap = std::getenv("GUFO_CAPTURE_MAXTOK")
                                  ? static_cast<std::size_t>(std::atoi(
                                        std::getenv("GUFO_CAPTURE_MAXTOK")))
                                  : 8192;
      if (ids.size() > cap)
        ids.resize(cap);
      std::vector<std::uint16_t> hidden;
      if (!model->CaptureHidden(ids, &hidden)) {
        std::fprintf(stderr, "judge: capture failed at doc %zu\n", d);
        return 1;
      }
      doc_out.write(reinterpret_cast<const char*>(&total), sizeof(total));
      tok_out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * 4);
      hid_out.write(reinterpret_cast<const char*>(hidden.data()),
                    hidden.size() * 2);
      total += ids.size();
      if (d % 50 == 0)
        std::fprintf(stderr, "capture: doc %zu/%zu, %llu tokens, %.0f tok/s\n",
                     d, doc_total, static_cast<unsigned long long>(total),
                     total / std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count());
    }
    std::printf("captured %llu tokens from %zu docs\n",
                static_cast<unsigned long long>(total), doc_total);
    return 0;
  }
  if (!use_gpu && !generate_prompt.empty()) {
    auto model = models::qwen38_flash_next::D1NpuModel::Create(*reader, &err);
    auto tokenizer = models::lfm3::Tokenizer::Load(*reader, &err);
    if (!model || !tokenizer ||
        !model->EnableGeneration(std::getenv("GUFO_D1_CTX")
                                     ? static_cast<std::uint32_t>(std::atoi(
                                           std::getenv("GUFO_D1_CTX")))
                                     : 8192u,
                                 &err)) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    if (!generate_prompt.empty() && generate_prompt[0] == '@') {
      std::ifstream file(generate_prompt.substr(1));
      std::stringstream buffer;
      buffer << file.rdbuf();
      generate_prompt = buffer.str();
    }
    if (!drafter_path.empty() && !model->LoadDrafter(drafter_path, &err)) {
      std::fprintf(stderr, "judge: %s\n", err.c_str());
      return 1;
    }
    const std::vector<std::pair<std::string, std::string>> messages = {
        {"user", generate_prompt}};
    const auto ids = std::getenv("GUFO_RAW_PROMPT")
                         ? tokenizer->Encode(generate_prompt)
                         : tokenizer->Encode(models::lfm3::Tokenizer::Render(
                               messages, false));
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
           "The agent is about to run: python train.py --epochs 50 on a 20GB "
           "model."},
          {"Is this a quick read-only shell command?",
           "The agent is about to run: ls -la /tmp"},
          {"Will this command use the GPU or most of the machine's RAM?",
           "The agent is about to run: cargo build --release for a large "
           "workspace."},
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
