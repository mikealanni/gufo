#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling32/gpu/engine.hpp"
#include "src/models/ling32/reference.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

std::vector<std::int32_t> ParseTokens(const std::string& text) {
  std::vector<std::int32_t> out;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t comma = text.find(',', at);
    const std::string part = text.substr(
        at, comma == std::string::npos ? std::string::npos : comma - at);
    if (!part.empty())
      out.push_back(static_cast<std::int32_t>(std::atoi(part.c_str())));
    if (comma == std::string::npos)
      break;
    at = comma + 1;
  }
  return out;
}

std::vector<std::int32_t> TopTokens(const std::vector<float>& logits,
                                    int count) {
  std::vector<std::pair<float, std::int32_t>> rank;
  rank.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    rank.emplace_back(logits[i], static_cast<std::int32_t>(i));
  }
  std::partial_sort(rank.begin(), rank.begin() + count, rank.end(),
                    [](const auto& a, const auto& b) {
                      if (a.first != b.first)
                        return a.first > b.first;
                      return a.second < b.second;
                    });
  std::vector<std::int32_t> out;
  for (int i = 0; i < count; ++i)
    out.push_back(rank[i].second);
  return out;
}

std::int32_t ArgMax(const std::vector<float>& logits) {
  return static_cast<std::int32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

float TopProbability(const std::vector<float>& logits) {
  const float top = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (const float value : logits)
    sum += std::exp(static_cast<double>(value - top));
  return static_cast<float>(1.0 / sum);
}

void DumpLogits(const std::string& path, const std::vector<float>& logits) {
  if (path.empty())
    return;
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t count = logits.size();
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  out.write(reinterpret_cast<const char*>(logits.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
}

int RunCpu(const std::string& path, const std::vector<std::int32_t>& tokens,
           int generate, bool run_mtp, int mtp_steps,
           std::int32_t mtp_token_override, const std::string& logits_path,
           std::string* error) {
  auto model = gufo::models::ling32::ReferenceModel::Open(path, error);
  if (!model)
    return 1;
  auto session = model->CreateSession();
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto& logits = model->Evaluate(*session, tokens[i]);
    const auto top = TopTokens(logits, 5);
    std::printf("step=%zu token=%d top=%d:%0.6f", i, tokens[i], top[0],
                logits[top[0]]);
    for (std::size_t k = 1; k < top.size(); ++k)
      std::printf(" %d:%0.6f", top[k], logits[top[k]]);
    std::printf("\n");
  }
  DumpLogits(logits_path, session->logits);
  if (run_mtp && !tokens.empty()) {
    std::int32_t mtp_token =
        mtp_token_override >= 0 ? mtp_token_override : tokens.back();
    std::vector<float> h = session->mtp_pending_h;
    std::uint32_t pos = session->position == 0 ? 0 : session->position - 1;
    for (int step = 0; step < mtp_steps; ++step) {
      const auto& mtp = model->MtpEvaluate(*session, mtp_token, h, pos);
      const auto mtop = TopTokens(mtp, 5);
      std::printf("mtp step=%d token=%d top=%d:%0.6f", step, mtp_token, mtop[0],
                  mtp[mtop[0]]);
      for (std::size_t k = 1; k < mtop.size(); ++k)
        std::printf(" %d:%0.6f", mtop[k], mtp[mtop[k]]);
      std::printf("\n");
      mtp_token = mtop[0];
      h = session->mtp_h;
      ++pos;
    }
  }
  std::vector<std::uint32_t> generated;
  for (int i = 0; i < generate; ++i) {
    const auto& logits = session->logits;
    const auto top = TopTokens(logits, 1);
    generated.push_back(static_cast<std::uint32_t>(top[0]));
    std::printf("gen=%d token=%d logit=%0.6f\n", i, top[0], logits[top[0]]);
    model->Evaluate(*session, top[0]);
  }
  auto reader = gufo::core::GgufReader::OpenFile(path, error);
  if (!reader)
    return 1;
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, error);
  if (!tokenizer)
    return 1;
  std::printf("TEXT:%s\n", tokenizer->Decode(generated).c_str());
  return 0;
}

bool RunSpeculative(gufo::models::ling32::gpu::Session& session,
                    std::int32_t last_token, int generate, int width,
                    float threshold, std::vector<std::uint32_t>* generated,
                    std::string* error) {
  int emitted = 0;
  int accepted = 0;
  int proposed = 0;
  int mtp_calls = 0;
  std::vector<int> accepted_at(static_cast<std::size_t>(width));
  std::vector<int> proposed_at(static_cast<std::size_t>(width));
  while (emitted < generate) {
    std::vector<float> target = session.Logits();
    std::vector<std::int32_t> drafts;
    drafts.reserve(static_cast<std::size_t>(width));
    session.ResetMtp();
    for (int j = 0; j < width && emitted + j < generate; ++j) {
      const std::int32_t seed = j == 0 ? last_token : drafts.back();
      if (!session.MtpAppend(seed, error))
        return false;
      ++mtp_calls;
      const auto& mtp = session.MtpLogits();
      const std::int32_t draft = ArgMax(mtp);
      if (TopProbability(mtp) < threshold)
        break;
      drafts.push_back(draft);
    }
    if (drafts.empty())
      drafts.push_back(ArgMax(target));
    for (std::size_t j = 0; j < drafts.size() && emitted < generate; ++j) {
      const std::int32_t expected = ArgMax(target);
      const bool accept = expected == drafts[j];
      const std::int32_t next = accept ? drafts[j] : expected;
      if (accept)
        ++accepted;
      ++proposed;
      ++proposed_at[j];
      if (accept)
        ++accepted_at[j];
      ++emitted;
      generated->push_back(static_cast<std::uint32_t>(next));
      std::printf("gen=%d token=%d draft=%d accept=%d logit=%0.6f\n",
                  emitted - 1, next, drafts[j], accept ? 1 : 0, target[next]);
      if (!session.Append(std::span<const std::int32_t>(&next, 1), error))
        return false;
      last_token = next;
      target = session.Logits();
      if (!accept)
        break;
    }
  }
  std::fprintf(stderr,
               "draft_accept=%.4f (%d/%d) draft_width=%d threshold=%.3f "
               "mtp_calls=%d per_slot=",
               proposed == 0 ? 0.0 : static_cast<double>(accepted) / proposed,
               accepted, proposed, width, threshold, mtp_calls);
  for (int j = 0; j < width; ++j) {
    std::fprintf(stderr, "%d/%d ", accepted_at[static_cast<std::size_t>(j)],
                 proposed_at[static_cast<std::size_t>(j)]);
  }
  std::fprintf(stderr, "\n");
  return true;
}

bool RunSpecVerify(gufo::models::ling32::gpu::Session& session, int generate,
                   int width, std::vector<std::uint32_t>* generated,
                   std::string* error) {
  int rounds = 0;
  int accepted_total = 0;
  while (static_cast<int>(generated->size()) < generate) {
    const std::int32_t t = ArgMax(session.Logits());
    std::vector<std::int32_t> toks{t};
    session.BeginDraft();
    std::int32_t seed = t;
    for (int j = 0; j < width; ++j) {
      if (!session.MtpAppend(seed, error))
        return false;
      seed = ArgMax(session.MtpLogits());
      toks.push_back(seed);
    }
    std::vector<float> rows;
    if (!session.Verify(toks, &rows, error))
      return false;
    const std::size_t vocab = session.Logits().size();
    std::uint32_t a = 0;
    while (a < static_cast<std::uint32_t>(width)) {
      const std::vector<float> row(rows.begin() + a * vocab,
                                   rows.begin() + (a + 1) * vocab);
      if (ArgMax(row) != toks[a + 1])
        break;
      ++a;
    }
    for (std::uint32_t j = 0; j <= a; ++j)
      generated->push_back(static_cast<std::uint32_t>(toks[j]));
    accepted_total += static_cast<int>(a);
    ++rounds;
    if (!session.Commit(a, error))
      return false;
  }
  std::fprintf(stderr, "spec rounds=%d accepted_drafts=%d tokens/round=%.2f\n",
               rounds, accepted_total,
               static_cast<double>(generated->size()) / rounds);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> [--tokens a,b,c] [--text prompt] "
                 "[--gen N] [--draft N] [--draft-threshold P] [--ctx N] "
                 "[--batch] [--dump-logits PATH] [--mtp] [--mtp-steps N] "
                 "[--mtp-token ID]\n",
                 argv[0]);
    return 2;
  }
  std::string error;
  std::vector<std::int32_t> tokens;
  std::string prompt = "The capital of France is";
  bool text_mode = false;
  bool cpu_mode = false;
  bool batch_mode = false;
  std::string logits_path;
  int generate = 16;
  int draft_width = 0;
  float draft_threshold = 0.0F;
  std::uint32_t context = 4096;
  bool run_mtp = false;
  int mtp_steps = 1;
  std::int32_t mtp_token_override = -1;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--cpu") {
      cpu_mode = true;
    } else if (arg == "--batch") {
      batch_mode = true;
    } else if (arg == "--dump-logits" && i + 1 < argc) {
      logits_path = argv[++i];
    } else if (arg == "--tokens" && i + 1 < argc) {
      tokens = ParseTokens(argv[++i]);
    } else if (arg == "--text" && i + 1 < argc) {
      prompt = argv[++i];
      text_mode = true;
    } else if (arg == "--prompt-file" && i + 1 < argc) {
      std::ifstream in(argv[++i]);
      std::stringstream ss;
      ss << in.rdbuf();
      prompt = ss.str();
      text_mode = true;
    } else if (arg == "--gen" && i + 1 < argc) {
      generate = std::atoi(argv[++i]);
    } else if (arg == "--draft" && i + 1 < argc) {
      draft_width = std::atoi(argv[++i]);
    } else if (arg == "--draft-threshold" && i + 1 < argc) {
      draft_threshold = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--ctx" && i + 1 < argc) {
      context = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    } else if (arg == "--mtp") {
      run_mtp = true;
    } else if (arg == "--mtp-steps" && i + 1 < argc) {
      mtp_steps = std::atoi(argv[++i]);
    } else if (arg == "--mtp-token" && i + 1 < argc) {
      mtp_token_override = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  if (text_mode) {
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader) {
      std::fprintf(stderr, "open failed: %s\n", error.c_str());
      return 1;
    }
    auto tokenizer =
        gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
    if (!tokenizer) {
      std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
      return 1;
    }
    const auto encoded = tokenizer->Encode(prompt, {});
    tokens.assign(encoded.begin(), encoded.end());
    std::fprintf(stderr, "prompt_tokens=%zu\n", tokens.size());
  }
  if (tokens.empty()) {
    std::fprintf(stderr, "no tokens to run\n");
    return 2;
  }
  if (cpu_mode) {
    return RunCpu(argv[1], tokens, text_mode ? generate : 0, run_mtp, mtp_steps,
                  mtp_token_override, logits_path, &error);
  }

  const auto start = std::chrono::steady_clock::now();
  auto model = gufo::models::ling32::gpu::Model::Load(argv[1], &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }
  const auto loaded = std::chrono::steady_clock::now();
  std::fprintf(
      stderr, "load_ms=%.1f resident_mib=%.1f\n",
      std::chrono::duration<double, std::milli>(loaded - start).count(),
      static_cast<double>(model->ResidentBytes()) / (1 << 20));

  if (run_mtp || draft_width > 1)
    session->EnableMtp();
  if (batch_mode) {
    const auto prefill_start = std::chrono::steady_clock::now();
    if (!session->Append(tokens, &error)) {
      std::fprintf(stderr, "forward failed: %s\n", error.c_str());
      return 1;
    }
    const double prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      prefill_start)
            .count();
    std::fprintf(stderr, "prefill_tokens=%zu prefill_ms=%.1f pp_tok_s=%.2f\n",
                 tokens.size(), prefill_seconds * 1000.0,
                 static_cast<double>(tokens.size()) / prefill_seconds);
    const auto top = TopTokens(session->Logits(), 5);
    std::printf("step=%zu token=%d pos=%u top=%d:%0.6f", tokens.size() - 1,
                tokens.back(), session->Position(), top[0],
                session->Logits()[top[0]]);
    for (std::size_t k = 1; k < top.size(); ++k) {
      std::printf(" %d:%0.6f", top[k], session->Logits()[top[k]]);
    }
    std::printf("\n");
  } else {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (!session->Append(std::span<const std::int32_t>(&tokens[i], 1),
                           &error)) {
        std::fprintf(stderr, "forward failed: %s\n", error.c_str());
        return 1;
      }
      const auto top = TopTokens(session->Logits(), 5);
      std::printf("step=%zu token=%d pos=%u top=%d:%0.6f", i, tokens[i],
                  session->Position(), top[0], session->Logits()[top[0]]);
      for (std::size_t k = 1; k < top.size(); ++k) {
        std::printf(" %d:%0.6f", top[k], session->Logits()[top[k]]);
      }
      std::printf("\n");
    }
  }

  DumpLogits(logits_path, session->Logits());

  if (run_mtp) {
    std::int32_t mtp_token =
        mtp_token_override >= 0 ? mtp_token_override : tokens.back();
    session->ResetMtp();
    for (int step = 0; step < mtp_steps; ++step) {
      if (!session->MtpAppend(mtp_token, &error)) {
        std::fprintf(stderr, "MTP failed: %s\n", error.c_str());
        return 1;
      }
      const auto& mtp = session->MtpLogits();
      const auto mtop = TopTokens(mtp, 5);
      std::printf("mtp step=%d token=%d top=%d:%0.6f", step, mtp_token, mtop[0],
                  mtp[mtop[0]]);
      for (std::size_t k = 1; k < mtop.size(); ++k)
        std::printf(" %d:%0.6f", mtop[k], mtp[mtop[k]]);
      std::printf("\n");
      mtp_token = mtop[0];
    }
  }

  if (!text_mode || generate <= 0)
    return 0;

  std::vector<std::uint32_t> generated;
  const auto decode_start = std::chrono::steady_clock::now();
  if (draft_width > 1 && std::getenv("LING_SPEC_VERIFY") != nullptr) {
    if (!RunSpecVerify(*session, generate, draft_width - 1, &generated,
                       &error)) {
      std::fprintf(stderr, "speculative verify failed: %s\n", error.c_str());
      return 1;
    }
  } else if (draft_width > 1) {
    if (!RunSpeculative(*session, tokens.back(), generate, draft_width,
                        draft_threshold, &generated, &error)) {
      std::fprintf(stderr, "speculative decode failed: %s\n", error.c_str());
      return 1;
    }
  } else {
    for (int i = 0; i < generate; ++i) {
      const std::int32_t next = ArgMax(session->Logits());
      generated.push_back(static_cast<std::uint32_t>(next));
      std::printf("gen=%d token=%d logit=%0.6f\n", i, next,
                  session->Logits()[next]);
      if (!session->Append(std::span<const std::int32_t>(&next, 1), &error)) {
        std::fprintf(stderr, "decode failed: %s\n", error.c_str());
        return 1;
      }
    }
  }
  const auto decode_end = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start)
          .count();
  std::fprintf(stderr, "decode_ms=%.1f tok_s=%.2f\n", ms,
               generate * 1000.0 / ms);

  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "reopen failed: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("TEXT:%s\n", tokenizer->Decode(generated).c_str());
  return 0;
}
