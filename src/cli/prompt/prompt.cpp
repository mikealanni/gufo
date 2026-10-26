#include "src/cli/prompt/prompt.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/flint/chat_template.hpp"
#include "src/models/flint/reference.hpp"
#include "src/models/glm53_flash/chat_template.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/laguna/chat_template.hpp"
#include "src/models/laguna/reference.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/lookup_draft.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/flint/gpu/engine.hpp"
#include "src/models/glm53_flash/engine.hpp"
#include "src/models/laguna/gpu/engine.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#endif

namespace gufo::cli {

// Hash emitted IDs in a portable byte order, after the timed generation.
// Decoded text alone can hide different token sequences.
static void PrintTokenTrace(std::span<const tokenization::TokenId> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(token >> shift));
    }
  }
  std::cerr << "[TokenTrace]: count=" << tokens.size()
            << " sha256=" << crypto::Sha256Hex(bytes) << '\n';
}

static void RegisterImageOptions(ArgParser& parser, PromptOptions& opt) {
  parser.AddFlag("", "--add-vision-id",
                 "Prefix image inputs with Picture N: in the chat template",
                 "Prompt", &opt.add_vision_id);
  parser.AddOption("", "--mmproj", "PATH",
                   "Qwen BF16 vision sidecar (auto-discovered beside model)",
                   "Model", &opt.vision_model_path);
  parser.AddCustomOption(
      "", "--image", "PATH",
      "PNG/JPEG attached before text in the first user turn; repeat for "
      "multiple images",
      "Prompt",
      [&opt](std::string_view, std::string_view value, std::string* error) {
        if (value.empty()) {
          if (error)
            *error = "--image requires a file path";
          return false;
        }
        opt.image_paths.emplace_back(value);
        return true;
      });
}

static void AttachImages(const PromptOptions& opt,
                         tokenization::ChatMessage& message) {
  for (const auto& path : opt.image_paths) {
    message.images.push_back(
        {0, std::make_shared<const std::vector<std::uint8_t>>(
                core::ReadImageFile(path))});
  }
}

static void RegisterTextOptions(ArgParser& parser, PromptOptions& opt,
                                bool& speculative_explicit,
                                std::string_view command) {
  // Model
  parser.AddOption("-m", "--model", "PATH", "Path to GGUF model file", "Model",
                   &opt.model_path);
  RegisterImageOptions(parser, opt);

  // Prompt & Formatting
  if (command == "prompt") {
    parser.AddOption("-p", "--prompt", "TEXT", "Direct input prompt text",
                     "Prompt", &opt.prompt_text);
    parser.AddOption("-f", "--file", "PATH",
                     "File path to read input prompt text from", "Prompt",
                     &opt.prompt_file);
  }
  parser.AddOption("", "--system", "PROMPT",
                   "Optional system role instructions prepended to the prompt",
                   "Prompt", &opt.system_prompt);
  if (command == "prompt") {
    parser.AddInverseFlag("", "--raw",
                          "Disable chat template framing and pass raw tokens",
                          "Prompt", &opt.use_chat_template);
    parser.AddInverseFlag(
        "", "--no-display-prompt",
        "Suppress echoing the prompt before generated response", "Prompt",
        &opt.display_prompt);
  }
  // Sampling
  parser.AddOption("-n", "--max-tokens", "N",
                   "Maximum number of new tokens to generate (default: 128)",
                   "Sampling", &opt.max_tokens);
  RegisterSamplingOptions(parser, &opt.sampling, "Sampling", true, true);

  // Reasoning
  parser.AddOption("", "--think", "MODE",
                   "Reasoning mode: on, off, or auto (default: model)",
                   "Reasoning", &opt.reasoning_mode);
  parser.AddOption("", "--reasoning-effort", "LEVEL",
                   "Effort: auto, minimal, low, medium, high, xhigh, or max",
                   "Reasoning", &opt.reasoning_effort);
  parser.AddOption("", "--preserve-thinking", "MODE",
                   "Replay prior reasoning: on, off, or auto", "Reasoning",
                   &opt.preserve_thinking);

  // Speculative & Hardware
  const auto parse_speculative_backend =
      [&opt, &speculative_explicit](std::string_view, std::string_view value,
                                    std::string* error) -> bool {
    speculative_explicit = true;
    if (value == "off") {
      opt.speculative_backend.clear();
      opt.speculative_off = true;
    } else if (value == "dspark" || value == "dflash2" || value == "mtp") {
      opt.speculative_backend = value;
    } else {
      if (error != nullptr)
        *error = "Unknown speculative backend: " + std::string(value);
      return false;
    }
    return true;
  };
  parser.AddCustomOption(
      "", "--speculative", "MODE",
      "Draft backend: dspark (DeepSeek V4 Flash), dflash2, mtp, or off",
      "Speculative", parse_speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to Qwen DFlash2 GGUF file", "Speculative",
                   &opt.dflash_model_path);
  parser.AddOption(
      "", "--draft-policy", "POLICY",
      "DFlash2 block length: fixed or adaptive (default: adaptive)",
      "Speculative", &opt.draft_policy);
  parser.AddOption("", "--dspark-model", "PATH",
                   "Path to the DeepSeek V4 Flash DSpark support GGUF file",
                   "Speculative", &opt.dspark_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to quantized Qwen MTP draft head GGUF file",
                   "Speculative", &opt.mtp_model_path);
  parser.AddCustomOption(
      "-d", "--draft-tokens", "N",
      "Maximum speculative draft tokens evaluated per step (default: 7)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error = "Invalid integer for draft-tokens: " + std::string(value);
          }
          return false;
        }
        opt.draft_tokens = count;
        return true;
      });

  parser.AddCustomOption(
      "", "--min-draft-tokens", "N", "Adaptive draft floor (default: 1)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error =
                "Invalid integer for min-draft-tokens: " + std::string(value);
          }
          return false;
        }
        opt.min_draft_tokens = count;
        return true;
      });

  parser.AddFlag("", "--cpu",
                 "Force CPU OpenMP execution fallback instead of GPU ROCm",
                 "Hardware", &opt.force_cpu);
  parser.AddFlag("-v", "--verbose",
                 "Print detailed timing, latency breakdown, and tok/s metrics",
                 "General", &opt.verbose);

  if (command == "prompt") {
    parser.JoinPositionals(&opt.prompt_text);
  }
}

static void PrintTextHelp(std::string_view program_name,
                          std::string_view command) {
  PromptOptions opt;
  bool speculative_explicit = false;
  ArgParser parser(std::string(program_name) + " " + std::string(command),
                   command == "chat" ? "Interactive conversation."
                                     : "Execute one prompt request and exit.");
  RegisterTextOptions(parser, opt, speculative_explicit, command);
  parser.PrintHelp();
}

void PrintPromptHelp(std::string_view program_name) {
  PrintTextHelp(program_name, "prompt");
}

void PrintChatHelp(std::string_view program_name) {
  PrintTextHelp(program_name, "chat");
}

namespace {

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << load_seconds << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

bool IsDeepSeekV4Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "deepseek4";
}

bool IsLaguna(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "laguna";
}

int RunLagunaPromptCPU(const PromptOptions& opt, const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start);
#ifdef ENGINE_ENABLE_HIP
int RunLagunaPromptGPU(const PromptOptions& opt, const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start);
#endif

bool IsFlint(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "flint";
}

// Forward declarations.
int RunFlintPromptCPU(const PromptOptions& opt, const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start);
int RunFlintPromptGPU(const PromptOptions& opt, const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start);
bool IsGlm53Flash(const core::GgufReader& reader) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  return architecture == "glm5-next" || architecture == "glm5next";
}

int RunGlm53Prompt(const PromptOptions& opt, const core::GgufReader& reader,
                   std::chrono::steady_clock::time_point load_start) {
  std::string err;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(reader, &err);
  if (!tokenizer) {
    std::cerr << "GLM-5.3 tokenizer failed: " << err << '\n';
    return 1;
  }
  auto model = models::glm53_flash::ReferenceModel::Open(opt.model_path, &err);
  if (!model) {
    std::cerr << "GLM-5.3 load failed: " << err << '\n';
    PrintModelLoadTime(load_start, false);
    return 1;
  }
  PrintModelLoadTime(load_start, true);

  const std::string rendered =
      opt.use_chat_template
          ? models::glm53_flash::RenderPrompt(
                opt.system_prompt, opt.prompt_text, opt.reasoning_effort)
          : opt.prompt_text;
  tokenization::TokenizerOptions encode_options;
  encode_options.parse_special_tokens = true;
  const auto prompt = tokenizer->Encode(rendered, encode_options);
  if (prompt.empty()) {
    std::cerr << "GLM-5.3 prompt produced no tokens\n";
    return 1;
  }

  std::vector<tokenization::TokenId> stop_tokens;
  for (const char* key :
       {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id"}) {
    if (const auto id = reader.GetMetadataUint32(key)) {
      stop_tokens.push_back(*id);
    }
  }
  const auto is_stop = [&](tokenization::TokenId token) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
           stop_tokens.end();
  };

  auto session = model->CreateSession();
  sampling::SamplerState sampler(opt.sampling, prompt);
  const auto start = std::chrono::steady_clock::now();
  const std::vector<float>* logits = nullptr;
  for (const auto token : prompt) {
    logits = &model->Evaluate(*session, static_cast<std::int32_t>(token));
  }
  const auto prefill_end = std::chrono::steady_clock::now();

  std::size_t generated = 0;
  for (; generated < opt.max_tokens; ++generated) {
    const auto token = sampler.Sample(*logits);
    if (is_stop(token)) {
      break;
    }
    sampler.Accept(token);
    std::cout << tokenizer->DecodeToken(token) << std::flush;
    if (generated + 1 < opt.max_tokens) {
      logits = &model->Evaluate(*session, static_cast<std::int32_t>(token));
    }
  }
  std::cout << '\n';
  if (opt.verbose) {
    const auto seconds = [](auto a, auto b) {
      return std::chrono::duration<double>(b - a).count();
    };
    const auto end = std::chrono::steady_clock::now();
    std::cerr << "[GLM-5.3 CPU reference] prompt=" << prompt.size()
              << " tok in " << seconds(start, prefill_end)
              << " s, generated=" << generated << " tok in "
              << seconds(prefill_end, end) << " s\n";
  }
  return 0;
}

std::optional<ReasoningEffort> ParseReasoningEffort(std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

ReasoningOptions PromptReasoningOptions(const PromptOptions& options) {
  ReasoningOptions reasoning;
  if (options.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (options.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (options.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

std::vector<tokenization::TokenId> FlintStopTokens(
    const core::GgufReader& reader) {
  std::vector<tokenization::TokenId> stop_tokens;
  if (const auto eos =
          reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    stop_tokens.push_back(*eos);
  }
  for (const char* key :
       {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
    if (const auto id = reader.GetMetadataUint32(key)) {
      stop_tokens.push_back(*id);
    }
  }
  if (const auto* meta = reader.FindMetadata("tokenizer.ggml.eog_token_ids")) {
    if (const auto* ids =
            std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto id : *ids) {
        stop_tokens.push_back(static_cast<tokenization::TokenId>(id));
      }
    } else if (const auto* ids =
                   std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (const auto id : *ids) {
        stop_tokens.push_back(static_cast<tokenization::TokenId>(id));
      }
    }
  }
  return stop_tokens;
}

int RunFlintPrompt(const PromptOptions& opt, const core::GgufReader& reader,
                   std::chrono::steady_clock::time_point load_start) {
#ifdef ENGINE_ENABLE_HIP
  if (opt.force_cpu) {
    return RunFlintPromptCPU(opt, reader, load_start);
  }
  return RunFlintPromptGPU(opt, reader, load_start);
#else
  return RunFlintPromptCPU(opt, reader, load_start);
#endif
}

// ── Flint CPU reference ─────────────────────────────────────────────

int RunFlintPromptCPU(const PromptOptions& opt, const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start) {
  std::string err;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(reader, &err);
  if (!tokenizer) {
    std::cerr << "Flint tokenizer failed: " << err << '\n';
    return 1;
  }
  auto model = models::flint::ReferenceModel::Open(opt.model_path, &err);
  if (!model) {
    std::cerr << "Flint load failed: " << err << '\n';
    PrintModelLoadTime(load_start, false);
    return 1;
  }
  PrintModelLoadTime(load_start, true);

  const auto reasoning = PromptReasoningOptions(opt);
  models::flint::ChatTemplateOptions chat_options;
  chat_options.enable_thinking = reasoning.enabled.value_or(false);
  const std::string rendered =
      opt.use_chat_template
          ? models::flint::RenderPrompt(opt.system_prompt, opt.prompt_text,
                                        chat_options)
          : opt.prompt_text;
  tokenization::TokenizerOptions encode_options;
  encode_options.parse_special_tokens = true;
  const auto prompt = tokenizer->Encode(rendered, encode_options);
  if (prompt.empty()) {
    std::cerr << "Flint prompt produced no tokens\n";
    return 1;
  }

  const auto stop_tokens = FlintStopTokens(reader);
  const auto is_stop = [&](tokenization::TokenId token) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
           stop_tokens.end();
  };

  auto session = model->CreateSession();
  sampling::SamplerState sampler(opt.sampling, prompt);
  const auto start = std::chrono::steady_clock::now();
  const std::vector<float>* logits = nullptr;
  for (const auto token : prompt) {
    logits = &model->Evaluate(*session, static_cast<std::int32_t>(token));
  }
  const auto prefill_end = std::chrono::steady_clock::now();

  std::size_t generated = 0;
  for (; generated < opt.max_tokens; ++generated) {
    const auto token = sampler.Sample(*logits);
    if (is_stop(token)) {
      break;
    }
    sampler.Accept(token);
    std::cout << tokenizer->DecodeToken(token) << std::flush;
    if (generated + 1 < opt.max_tokens) {
      logits = &model->Evaluate(*session, static_cast<std::int32_t>(token));
    }
  }
  std::cout << '\n';
  if (opt.verbose) {
    const auto seconds = [](auto a, auto b) {
      return std::chrono::duration<double>(b - a).count();
    };
    const auto end = std::chrono::steady_clock::now();
    std::cerr << "[Flint CPU reference] prompt=" << prompt.size() << " tok in "
              << seconds(start, prefill_end) << " s, generated=" << generated
              << " tok in " << seconds(prefill_end, end) << " s\n";
  }
  return 0;
}

// ── Laguna CPU reference ─────────────────────────────────────────────

int RunLagunaPromptCPU(const PromptOptions& opt, const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start) {
  std::string err;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(reader, &err);
  if (!tokenizer) {
    std::cerr << "Laguna tokenizer failed: " << err << '\n';
    return 1;
  }
  auto model = models::laguna::ReferenceModel::Open(opt.model_path, &err);
  if (!model) {
    std::cerr << "Laguna load failed: " << err << '\n';
    PrintModelLoadTime(load_start, false);
    return 1;
  }
  PrintModelLoadTime(load_start, true);

  const auto reasoning = PromptReasoningOptions(opt);
  models::laguna::ChatTemplateOptions chat_options;
  chat_options.enable_thinking = reasoning.enabled.value_or(false);
  const std::string rendered =
      opt.use_chat_template
          ? models::laguna::RenderPrompt(opt.system_prompt, opt.prompt_text,
                                         chat_options)
          : opt.prompt_text;
  tokenization::TokenizerOptions encode_options;
  encode_options.parse_special_tokens = true;
  const auto prompt = tokenizer->Encode(rendered, encode_options);
  if (prompt.empty()) {
    std::cerr << "Laguna prompt produced no tokens\n";
    return 1;
  }

  const auto stop_tokens = FlintStopTokens(reader);
  const auto is_stop = [&](tokenization::TokenId token) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
           stop_tokens.end();
  };

  auto session = model->CreateSession();
  sampling::SamplerState sampler(opt.sampling, prompt);
  const auto start = std::chrono::steady_clock::now();
  const std::vector<float>* logits = nullptr;
  for (const auto token : prompt) {
    logits = &model->Evaluate(*session, static_cast<std::int32_t>(token));
  }
  const auto prefill_end = std::chrono::steady_clock::now();

  std::size_t generated = 0;
  for (; generated < opt.max_tokens; ++generated) {
    const auto token = sampler.Sample(*logits);
    if (is_stop(token)) {
      break;
    }
    sampler.Accept(token);
    std::cout << tokenizer->DecodeToken(token) << std::flush;
    if (generated + 1 < opt.max_tokens) {
      logits = &model->Evaluate(*session, static_cast<std::int32_t>(token));
    }
  }
  std::cout << '\n';
  if (opt.verbose) {
    const auto seconds = [](auto a, auto b) {
      return std::chrono::duration<double>(b - a).count();
    };
    const auto end = std::chrono::steady_clock::now();
    std::cerr << "[Laguna CPU reference] prompt=" << prompt.size() << " tok in "
              << seconds(start, prefill_end) << " s, generated=" << generated
              << " tok in " << seconds(prefill_end, end) << " s\n";
  }
  return 0;
}

#ifdef ENGINE_ENABLE_HIP

int RunLagunaPromptGPU(const PromptOptions& opt, const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start) {
  std::string err;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(reader, &err);
  if (!tokenizer) {
    std::cerr << "Laguna tokenizer failed: " << err << '\n';
    return 1;
  }
  auto model = models::laguna::gpu::Model::Load(opt.model_path, {}, &err);
  if (!model) {
    std::cerr << "Laguna GPU load failed: " << err << '\n';
    PrintModelLoadTime(load_start, false);
    return 1;
  }
  PrintModelLoadTime(load_start, true);

  const auto reasoning = PromptReasoningOptions(opt);
  models::laguna::ChatTemplateOptions chat_options;
  chat_options.enable_thinking = reasoning.enabled.value_or(false);
  const std::string rendered =
      opt.use_chat_template
          ? models::laguna::RenderPrompt(opt.system_prompt, opt.prompt_text,
                                         chat_options)
          : opt.prompt_text;
  tokenization::TokenizerOptions encode_options;
  encode_options.parse_special_tokens = true;
  const auto prompt = tokenizer->Encode(rendered, encode_options);
  if (prompt.empty()) {
    std::cerr << "Laguna prompt produced no tokens\n";
    return 1;
  }

  const auto session = model->CreateSession(
      static_cast<std::uint32_t>(prompt.size() + opt.max_tokens + 1 +
                                 models::laguna::gpu::Session::kMaxVerifyRows),
      &err);
  if (!session) {
    std::cerr << "Laguna GPU session creation failed: " << err << '\n';
    return 1;
  }
  const auto stop_tokens = FlintStopTokens(reader);
  const auto is_stop = [&](tokenization::TokenId token) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
           stop_tokens.end();
  };
  sampling::SamplerState sampler(opt.sampling, prompt);
  std::vector<std::int32_t> input(prompt.begin(), prompt.end());
  const auto start = std::chrono::steady_clock::now();
  if (!session->Append(input, &err)) {
    std::cerr << "Laguna GPU prefill failed: " << err << '\n';
    return 1;
  }
  const auto prefill_end = std::chrono::steady_clock::now();

  std::size_t generated = 0;
  auto token = sampler.Sample(session->Logits());
  while (generated < opt.max_tokens && !is_stop(token)) {
    sampler.Accept(token);
    std::cout << tokenizer->DecodeToken(token) << std::flush;
    ++generated;
    if (generated >= opt.max_tokens)
      break;
    const std::int32_t feed[1] = {static_cast<std::int32_t>(token)};
    if (!session->Append(feed, &err)) {
      std::cerr << "\nLaguna GPU decode failed: " << err << '\n';
      return 1;
    }
    token = sampler.Sample(session->Logits());
  }
  std::cout << '\n';
  if (opt.verbose) {
    const auto seconds = [](auto a, auto b) {
      return std::chrono::duration<double>(b - a).count();
    };
    const auto end = std::chrono::steady_clock::now();
    std::cerr << "[Laguna GPU] prompt=" << prompt.size() << " tok in "
              << seconds(start, prefill_end) << " s ("
              << prompt.size() / seconds(start, prefill_end)
              << " tok/s), generated=" << generated << " tok in "
              << seconds(prefill_end, end) << " s ("
              << generated / std::max(seconds(prefill_end, end), 1e-9)
              << " tok/s)\n";
  }
  return 0;
}

#endif  // ENGINE_ENABLE_HIP

#ifdef ENGINE_ENABLE_HIP

int RunFlintPromptGPU(const PromptOptions& opt, const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start) {
  std::string err;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(reader, &err);
  if (!tokenizer) {
    std::cerr << "Flint tokenizer failed: " << err << '\n';
    return 1;
  }
  auto model = models::flint::gpu::Model::Load(opt.model_path, {}, &err);
  if (!model) {
    std::cerr << "Flint GPU load failed: " << err << '\n';
    PrintModelLoadTime(load_start, false);
    return 1;
  }
  PrintModelLoadTime(load_start, true);

  const auto reasoning = PromptReasoningOptions(opt);
  models::flint::ChatTemplateOptions chat_options;
  chat_options.enable_thinking = reasoning.enabled.value_or(false);
  const std::string rendered =
      opt.use_chat_template
          ? models::flint::RenderPrompt(opt.system_prompt, opt.prompt_text,
                                        chat_options)
          : opt.prompt_text;
  tokenization::TokenizerOptions encode_options;
  encode_options.parse_special_tokens = true;
  const auto prompt = tokenizer->Encode(rendered, encode_options);
  if (prompt.empty()) {
    std::cerr << "Flint prompt produced no tokens\n";
    return 1;
  }

  const auto session = model->CreateSession(
      static_cast<std::uint32_t>(prompt.size() + opt.max_tokens + 1 +
                                 models::flint::gpu::Session::kMaxVerifyRows),
      &err);
  if (!session) {
    std::cerr << "Flint GPU session creation failed: " << err << '\n';
    return 1;
  }
  const auto stop_tokens = FlintStopTokens(reader);
  const auto is_stop = [&](tokenization::TokenId token) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
           stop_tokens.end();
  };

  sampling::SamplerState sampler(opt.sampling, prompt);
  std::vector<std::int32_t> input(prompt.begin(), prompt.end());
  const auto start = std::chrono::steady_clock::now();
  if (!session->Append(input, &err)) {
    std::cerr << "Flint GPU prefill failed: " << err << '\n';
    return 1;
  }
  const auto prefill_end = std::chrono::steady_clock::now();

  // Lossless speculation: tokens that followed an earlier copy of the current
  // tail are verified in one pass; a wrong guess costs only the wider pass.
  constexpr std::uint32_t kLookupMinMatch = 6;
  constexpr std::uint32_t kConfidentMatch = 12;
  constexpr double kAcceptSmoothing = 0.3;
  // Measured cost of a verify pass of 1..5 rows relative to one row.
  constexpr double kVerifyCost[] = {1.0, 1.14, 1.28, 1.40, 1.58};
  double accept_rate[2] = {0.6, 0.85};
  const bool speculate = !opt.speculative_off;
  const std::uint32_t max_draft =
      std::min<std::uint32_t>(std::max<std::uint32_t>(opt.draft_tokens, 1),
                              models::flint::gpu::Session::kMaxVerifyRows - 1);
  models::qwen38_flash_next::LookupIndex lookup;
  for (const auto token : prompt)
    lookup.Append(static_cast<std::int32_t>(token));

  std::size_t generated = 0;
  std::size_t drafted = 0;
  std::size_t accepted_drafts = 0;
  std::chrono::duration<double> forward_time{};
  auto token = sampler.Sample(session->Logits());
  bool stopped = false;
  while (!stopped && generated < opt.max_tokens) {
    if (is_stop(token))
      break;
    sampler.Accept(token);
    std::cout << tokenizer->DecodeToken(token) << std::flush;
    lookup.Append(static_cast<std::int32_t>(token));
    ++generated;
    if (generated >= opt.max_tokens)
      break;

    std::vector<std::int32_t> feed{static_cast<std::int32_t>(token)};
    std::size_t bucket = 0;
    if (speculate) {
      std::uint32_t matched = 0;
      const auto draft = lookup.Draft(kLookupMinMatch, max_draft, &matched);
      bucket = matched >= kConfidentMatch ? 1 : 0;
      // Longest draft prefix with the best expected tokens per unit cost.
      double best = 1.0;
      std::size_t take = 0;
      double reach = 1.0;
      double expected = 1.0;
      for (std::size_t k = 1; k <= draft.size(); ++k) {
        reach *= accept_rate[bucket];
        expected += reach;
        const double value = expected / kVerifyCost[k];
        if (value > best) {
          best = value;
          take = k;
        }
      }
      feed.insert(feed.end(), draft.begin(), draft.begin() + take);
    }
    const std::uint32_t base = session->Position();
    const auto step = std::chrono::steady_clock::now();
    const bool ok = feed.size() == 1 ? session->Append(feed, &err)
                                     : session->AppendAll(feed, &err);
    forward_time += std::chrono::steady_clock::now() - step;
    if (!ok) {
      std::cerr << "\nFlint GPU decode failed: " << err << '\n';
      return 1;
    }
    drafted += feed.size() - 1;

    // Row i holds the distribution after feed[i]; draft token feed[i + 1] is
    // tested against it, and the first miss yields the replacement token.
    std::uint32_t kept = 0;
    for (std::uint32_t row = 0;; ++row) {
      const auto logits = session->Logits(row);
      if (row + 1 >= feed.size()) {
        token = sampler.Sample(logits);
        break;
      }
      const auto guess = static_cast<tokenization::TokenId>(feed[row + 1]);
      tokenization::TokenId next;
      bool hit;
      if (opt.sampling.uses_random_sampling()) {
        const double p = sampler.Distribution(logits).probability(guess);
        hit = sampler.Uniform() < p;
        if (!hit) {
          const float one = 1.0F;
          next = sampler.SampleResidual(logits, {&guess, 1}, {&one, 1});
        }
      } else {
        next = sampler.Sample(logits);
        hit = next == guess;
      }
      if (!hit) {
        token = next;
        break;
      }
      ++kept;
      ++accepted_drafts;
      if (is_stop(guess)) {
        stopped = true;
        break;
      }
      sampler.Accept(guess);
      std::cout << tokenizer->DecodeToken(guess) << std::flush;
      lookup.Append(feed[row + 1]);
      if (++generated >= opt.max_tokens) {
        stopped = true;
        break;
      }
    }
    session->Rewind(base + 1 + kept);
    if (feed.size() > 1) {
      const double seen = static_cast<double>(kept) / (feed.size() - 1);
      accept_rate[bucket] = (1.0 - kAcceptSmoothing) * accept_rate[bucket] +
                            kAcceptSmoothing * seen;
    }
  }
  std::cout << '\n';
  if (opt.verbose) {
    const auto seconds = [](auto a, auto b) {
      return std::chrono::duration<double>(b - a).count();
    };
    const auto end = std::chrono::steady_clock::now();
    std::cerr << "[Flint GPU] prompt=" << prompt.size() << " tok in "
              << seconds(start, prefill_end) << " s, generated=" << generated
              << " tok in " << seconds(prefill_end, end) << " s ("
              << forward_time.count() << " s in forward, drafted " << drafted
              << ", accepted " << accepted_drafts << ")\n";
  }
  return 0;
}

#endif

void ResolvePromptSampling(const core::GgufReader& reader, PromptOptions* opt) {
  sampling::TextModelPreset preset = sampling::TextModelPreset::kUnspecified;
  const auto artifact_architecture =
      reader.GetMetadataString("general.architecture");
  if (artifact_architecture == "deepseek4") {
    preset = sampling::TextModelPreset::kDeepSeekV4Flash;
  } else if (artifact_architecture == "flint") {
    preset = sampling::TextModelPreset::kFlint;
  } else if (artifact_architecture == "qwen4exp") {
    preset = sampling::TextModelPreset::kQwen38;
  } else if (const auto config = reader.ExtractModelConfig()) {
    preset = sampling::TextPreset(*config);
  }
  opt->sampling = sampling::ResolveTextSampling(
      preset, PromptReasoningOptions(*opt).enabled, opt->sampling,
      opt->sampling_supplied);
}

#if defined(ENGINE_ENABLE_HIP)
constexpr std::uint32_t kDefaultContext = 4096;

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<models::glm53_flash::Model> LoadGlmModel(
    const PromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  std::string error;
  const bool dflash = opt.speculative_backend == "dflash2";
  if (!opt.speculative_backend.empty() && !dflash) {
    std::cerr << "GLM-5.3 supports only --speculative dflash2\n";
    return nullptr;
  }
  if (dflash && (opt.dflash_model_path.empty() || opt.min_draft_tokens != 1 ||
                 opt.draft_tokens == 0 ||
                 opt.draft_tokens > models::glm53_flash::kMaxDraftTokens)) {
    std::cerr << "GLM-5.3 DFlash2 requires --dflash-model, one to seven draft "
                 "tokens and --min-draft-tokens 1\n";
    return nullptr;
  }
  if (opt.use_chat_template &&
      !models::glm53_flash::ValidateGgufTemplate(reader, &error)) {
    std::cerr << "Unsupported GLM-5.3 chat template: " << error << '\n';
    return nullptr;
  }
  auto model = models::glm53_flash::Model::Load(
      opt.model_path,
      {.max_context = kDefaultContext,
       .dflash_model_path = dflash ? opt.dflash_model_path : "",
       .max_draft_tokens = opt.draft_tokens},
      &error);
  PrintModelLoadTime(load_start, model != nullptr);
  if (!model)
    std::cerr << "GLM-5.3 load failed: " << error << '\n';
  return model;
}

int GenerateGlmResponse(const PromptOptions& opt,
                        const models::glm53_flash::Model& model,
                        models::glm53_flash::Session& session,
                        std::span<const std::int32_t> prompt,
                        std::string* reply = nullptr) {
  if (prompt.empty() || prompt.size() >= session.ContextSize() ||
      opt.max_tokens > session.ContextSize() - prompt.size()) {
    std::cerr << "GLM-5.3 prompt and output exceed the " << kDefaultContext
              << "-token CLI context\n";
    return 1;
  }
  std::string error;
  if (!session.Sync(prompt, &error)) {
    std::cerr << "GLM-5.3 prefill failed: " << error << '\n';
    return 1;
  }
  session.ResetDraftPolicy();
  if (opt.verbose) {
    std::cout << "[Engine]: GLM-5.3 ROCm (gfx1151)\n"
              << "Model: " << model.ModelName() << '\n'
              << "Prompt tokens: " << prompt.size() << '\n'
              << "Max tokens: " << opt.max_tokens << '\n'
              << "--- Generation Output ---\n";
  }
  const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState sampler(opt.sampling, history);
  std::vector<tokenization::TokenId> generated;
  const auto start = std::chrono::steady_clock::now();
  const auto before = session.Statistics();
  while (generated.size() < opt.max_tokens) {
    models::glm53_flash::Session::DecodeResult decoded;
    if (!session.DecodeStep(opt.max_tokens - generated.size(), sampler,
                            &decoded, &error)) {
      std::cerr << "GLM-5.3 decode failed: " << error << '\n';
      return 1;
    }
    for (const auto token : decoded.tokens) {
      const auto piece = model.TokenText(token);
      std::cout << piece << std::flush;
      if (reply)
        reply->append(piece);
      generated.push_back(static_cast<tokenization::TokenId>(token));
    }
    if (decoded.stop)
      break;
  }
  std::cout << '\n';
  if (opt.verbose) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    const auto after = session.Statistics();
    const auto drafted = after.drafted - before.drafted;
    if (opt.speculative_backend == "dflash2") {
      std::cerr << "[Speculative]: acceptance="
                << (drafted ? static_cast<double>(after.accepted -
                                                  before.accepted) /
                                  static_cast<double>(drafted)
                            : 0.0)
                << " drafted=" << drafted
                << " accepted=" << after.accepted - before.accepted
                << " verification_steps=" << after.cycles - before.cycles
                << " history_steps="
                << after.history_cycles - before.history_cycles << '\n';
    }
    std::cout << "Generated " << generated.size() << " tokens on ROCm in "
              << seconds << "s (" << generated.size() / seconds << " tok/s)\n";
    PrintTokenTrace(generated);
  }
  return 0;
}

int RunGlmPrompt(const PromptOptions& opt, const core::GgufReader& reader,
                 std::chrono::steady_clock::time_point load_start) {
  auto model = LoadGlmModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      kDefaultContext, &error);
  if (!session) {
    std::cerr << "GLM-5.3 session failed: " << error << '\n';
    return 1;
  }
  std::string text = opt.prompt_text;
  if (opt.use_chat_template) {
    std::vector<tokenization::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    text = models::glm53_flash::RenderChat(
        messages, {},
        models::glm53_flash::RenderOptionsFor(PromptReasoningOptions(opt)));
  }
  const auto prompt = model->Tokenize(text);
  return GenerateGlmResponse(opt, *model, *session, prompt);
}

int RunGlmChat(const PromptOptions& opt, const core::GgufReader& reader,
               std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "GLM-5.3 interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadGlmModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      kDefaultContext, &error);
  if (!session) {
    std::cerr << "GLM-5.3 session failed: " << error << '\n';
    return 1;
  }
  const auto render_options =
      models::glm53_flash::RenderOptionsFor(PromptReasoningOptions(opt));
  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }
  std::cout << "=== Gufo Interactive Chat (GLM-5.3 Flash) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
      break;
    if (input.empty())
      continue;
    history.push_back({tokenization::ChatRole::kUser, input, "", ""});
    const auto tokens = model->Tokenize(
        models::glm53_flash::RenderChat(history, {}, render_options));
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (GenerateGlmResponse(opt, *model, *session, tokens, &reply) != 0)
      return 1;
    // The reply carries its reasoning block; the template splits it again.
    history.push_back(
        {tokenization::ChatRole::kAssistant, std::move(reply), "", ""});
    std::cout << '\n';
  }
  return 0;
}
#endif

std::shared_ptr<models::deepseek_v4_flash::Model> LoadDeepSeekModel(
    const PromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  if (opt.force_cpu) {
    std::cerr << "DeepSeek V4 Flash is supported only by the ROCm backend\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  std::string template_error;
  if (!models::deepseek_v4_flash::ValidateGgufTemplate(reader,
                                                       &template_error)) {
    std::cerr << "Unsupported DeepSeek chat template: " << template_error
              << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  const bool dspark_requested = opt.speculative_backend == "dspark";
  if (!opt.speculative_backend.empty() && !dspark_requested) {
    std::cerr << "DeepSeek V4 Flash supports only --speculative dspark\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (opt.speculative_backend == "dspark" && opt.dspark_model_path.empty()) {
    std::cerr << "--speculative dspark requires --dspark-model\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (dspark_requested && opt.min_draft_tokens != 1) {
    std::cerr << "DSpark uses model-owned adaptive drafting; "
                 "--min-draft-tokens is unsupported\n";
    return nullptr;
  }

  std::string error;
  auto model = models::deepseek_v4_flash::Model::Load(
      opt.model_path,
      models::deepseek_v4_flash::ModelOptions{
          .max_context = kDefaultContext,
          .dspark_model_path = dspark_requested ? opt.dspark_model_path : "",
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating DeepSeek V4 Flash model: " << error << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  PrintModelLoadTime(load_start);

  return model;
}

int GenerateDeepSeekResponse(
    const PromptOptions& opt,
    const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
    models::deepseek_v4_flash::Session& session,
    std::span<const int> prompt_tokens, std::string* reply = nullptr) {
  std::string error;
  const bool dspark_requested = model->HasDspark();
  const auto emit = [&](int token) {
    const auto piece = model->DecodeToken(token);
    if (reply != nullptr)
      reply->append(piece);
    std::cout << piece << std::flush;
  };
  if (prompt_tokens.empty()) {
    std::cerr << "DeepSeek V4 Flash prompt produced no tokens\n";
    return 1;
  }
  if (prompt_tokens.size() >= kDefaultContext ||
      opt.max_tokens >= kDefaultContext - prompt_tokens.size()) {
    std::cerr << "DeepSeek V4 Flash prompt and output exceed the 4096-token "
                 "CLI context\n";
    return 1;
  }

  if (!session.Sync(prompt_tokens, &error)) {
    std::cerr << "DeepSeek V4 Flash prefill failed: " << error << '\n';
    return 1;
  }
  // Chat can extend an existing prefix. A new turn keeps its support KV but
  // must discard the previous turn's deferred draw and controller history.
  session.BeginRequest();

  if (opt.verbose) {
    std::cout << "[Engine]: DeepSeek V4 Flash ROCm (gfx1151)\n"
              << "Model: " << model->ModelName() << '\n'
              << "Prompt tokens: " << prompt_tokens.size() << '\n'
              << "Max tokens: " << opt.max_tokens << '\n'
              << "--- Generation Output ---\n";
  }

  std::vector<sampling::TokenId> sampling_history;
  sampling_history.reserve(prompt_tokens.size());
  for (const int token : prompt_tokens) {
    if (token < 0) {
      std::cerr << "DeepSeek V4 Flash produced an invalid prompt token\n";
      return 1;
    }
    sampling_history.push_back(static_cast<sampling::TokenId>(token));
  }
  sampling::SamplerState sampler(opt.sampling, sampling_history);

  const bool use_dspark = session.DsparkEnabled();

  const auto generation_start = std::chrono::steady_clock::now();
  std::size_t generated = 0;
  std::vector<tokenization::TokenId> generated_ids;
  if (use_dspark) {
    std::vector<int> emitted;
    bool stop = false;
    while (generated < opt.max_tokens && !stop) {
      std::optional<models::deepseek_v4_flash::DsparkSamplerBridge> bridge;
      if (!opt.sampling.can_use_unmodified_argmax())
        bridge.emplace(sampler);
      if (!session.DsparkStep(opt.max_tokens - generated, opt.draft_tokens,
                              &emitted, &error,
                              bridge ? bridge->hook() : nullptr, true)) {
        std::cerr << "\nDeepSeek V4 Flash speculative decode failed: " << error
                  << '\n';
        return 1;
      }
      if (emitted.empty())
        break;
      if (bridge)
        sampler.SetRngState(bridge->rng_state());
      for (const int token : emitted) {
        if (model->IsStopToken(token) || generated >= opt.max_tokens) {
          stop = true;
          break;
        }
        sampler.Accept(static_cast<sampling::TokenId>(token));
        emit(token);
        if (opt.verbose)
          generated_ids.push_back(token);
        ++generated;
      }
    }
  } else {
    for (; generated < opt.max_tokens; ++generated) {
      const auto logits = session.CopyLogits(&error);
      if (logits.empty()) {
        std::cerr << "\nDeepSeek V4 Flash token selection failed: " << error
                  << '\n';
        return 1;
      }
      const int token = static_cast<int>(sampler.Sample(logits));
      if (model->IsStopToken(token)) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      emit(token);
      if (opt.verbose)
        generated_ids.push_back(token);
      // A reusable chat session must own the complete emitted prefix, like
      // the server and DSpark. Leaving its last token to the next bulk
      // prefill changes that call's boundary and numerical trajectory.
      if ((reply != nullptr || generated + 1 < opt.max_tokens) &&
          !session.Evaluate(token, &error)) {
        std::cerr << "\nDeepSeek V4 Flash decode failed: " << error << '\n';
        return 1;
      }
    }
  }
  std::cout << '\n';

  if (use_dspark) {
    const auto stats = session.DsparkStatistics();
    const double support_acceptance =
        stats.support_drafted != 0
            ? static_cast<double>(stats.support_accepted) /
                  static_cast<double>(stats.support_drafted)
            : 0.0;
    const double positional_acceptance =
        stats.support_drafted != 0
            ? static_cast<double>(stats.positional_accepted) /
                  static_cast<double>(stats.support_drafted)
            : 0.0;
    const double full_block_rate =
        stats.steps != 0 ? static_cast<double>(stats.full_blocks) /
                               static_cast<double>(stats.steps)
                         : 0.0;
    std::cerr << "[Speculative]: acceptance=" << support_acceptance
              << " drafted=" << stats.support_drafted
              << " accepted=" << stats.support_accepted
              << " verification_steps=" << stats.steps
              << " skipped=" << stats.skipped
              << " positional_acceptance=" << positional_acceptance
              << " positional_accepted=" << stats.positional_accepted
              << " full_block_rate=" << full_block_rate
              << " full_blocks=" << stats.full_blocks
              << " anchors=" << stats.anchors
              << " verifier_rows=" << stats.verifier_rows << '\n';
  }

  if (opt.verbose && generated > 0) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      generation_start)
            .count();
    std::cout << "Generated " << generated << " tokens on ROCm in " << seconds
              << "s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
    PrintTokenTrace(generated_ids);
  }
  return 0;
}
int RunDeepSeekPrompt(const PromptOptions& opt, const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start) {
  auto model = LoadDeepSeekModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      kDefaultContext, &error);
  if (!session) {
    std::cerr << "DeepSeek session creation failed: " << error << '\n';
    return 1;
  }
  std::vector<int> prompt_tokens;
  if (opt.use_chat_template) {
    const auto reasoning = PromptReasoningOptions(opt);
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back({.role = "system",
                          .content = opt.system_prompt,
                          .reasoning_content = {},
                          .tool_calls = {}});
    }
    messages.push_back({.role = "user",
                        .content = opt.prompt_text,
                        .reasoning_content = {},
                        .tool_calls = {}});
    prompt_tokens = model->EncodeChat(
        messages,
        models::deepseek_v4_flash::ResolveDeepSeekChatOptions(reasoning));
  } else {
    prompt_tokens = model->Tokenize(opt.prompt_text);
  }
  return GenerateDeepSeekResponse(opt, model, *session, prompt_tokens);
}

int RunDeepSeekChat(const PromptOptions& opt, const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "DeepSeek interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadDeepSeekModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      kDefaultContext, &error);
  if (!session) {
    std::cerr << "DeepSeek session creation failed: " << error << '\n';
    return 1;
  }
  std::vector<models::deepseek_v4_flash::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back({.role = "system",
                       .content = opt.system_prompt,
                       .reasoning_content = {},
                       .tool_calls = {}});
  }
  const auto reasoning = PromptReasoningOptions(opt);
  const auto chat_options =
      models::deepseek_v4_flash::ResolveDeepSeekChatOptions(reasoning);
  std::cout << "=== Gufo Interactive Chat (DeepSeek V4 Flash) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
      break;
    if (input.empty())
      continue;
    history.push_back({.role = "user",
                       .content = input,
                       .reasoning_content = {},
                       .tool_calls = {}});
    const auto tokens = model->EncodeChat(history, chat_options);
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (GenerateDeepSeekResponse(opt, model, *session, tokens, &reply) != 0)
      return 1;
    models::deepseek_v4_flash::ChatMessage response{};
    response.role = "assistant";
    if (chat_options.enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = reply.find(end);
      response.reasoning_content = reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        response.content = reply.substr(boundary + end.size());
      }
    } else {
      response.content = std::move(reply);
    }
    history.push_back(std::move(response));
    std::cout << '\n';
  }
  return 0;
}

std::shared_ptr<models::qwen::vision::Encoder> LoadQwenVision(
    const PromptOptions& opt, const core::GgufReader& reader) {
  if (opt.image_paths.empty() && opt.vision_model_path.empty())
    return {};
  if (reader.GetMetadataUint64("qwen35.embedding_length") != 5120) {
    throw std::invalid_argument(
        "image input supports Qwen3.8-27B and Flash-Next");
  }
  auto encoder = models::qwen::vision::Encoder::Open(
      opt.model_path, opt.vision_model_path, 5120);
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  return encoder;
}

std::shared_ptr<const models::qwen::vision::Prompt> PrepareVision(
    const PromptOptions& opt, const tokenization::QwenTokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder) {
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  const auto reasoning = PromptReasoningOptions(opt);
  return std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, messages, {},
          tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
          encoder->identity(), kDefaultContext));
}

std::shared_ptr<models::qwen38_flash_next::Model> LoadFlashNextModel(
    const PromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  std::string error;
  const bool supported_backend = opt.speculative_backend.empty() ||
                                 opt.speculative_backend == "mtp" ||
                                 opt.speculative_backend == "dflash2";
  if (opt.force_cpu || !supported_backend ||
      ((opt.speculative_backend == "mtp" ||
        opt.speculative_backend == "dflash2") &&
       opt.min_draft_tokens != 1)) {
    std::cerr << "Flash-Next requires ROCm and supports MTP or DFlash2 "
                 "with --min-draft-tokens 1\n";
    return nullptr;
  }
  if (opt.use_chat_template &&
      !tokenization::QwenChatTemplate::ValidateGgufTemplate(reader, &error)) {
    std::cerr << "Unsupported Flash-Next chat template: " << error << '\n';
    return nullptr;
  }
  auto model = models::qwen38_flash_next::Model::Load(
      opt.model_path,
      {.max_context = kDefaultContext,
       .mtp_model_path =
           opt.speculative_backend == "mtp" ? opt.mtp_model_path : "",
       .dflash_model_path =
           opt.speculative_backend == "dflash2" ? opt.dflash_model_path : "",
       .max_draft_tokens = opt.draft_tokens,
       .vision_model_path = opt.vision_model_path},
      &error);
  PrintModelLoadTime(load_start, model != nullptr);
  if (!model)
    std::cerr << "Flash-Next load failed: " << error << '\n';
  return model;
}

int GenerateFlashNextResponse(const PromptOptions& opt,
                              const models::qwen38_flash_next::Model& model,
                              models::qwen38_flash_next::Session& session,
                              std::span<const tokenization::TokenId> prompt,
                              std::string* reply = nullptr) {
  if (prompt.empty() || prompt.size() >= session.ContextSize() ||
      opt.max_tokens > session.ContextSize() - prompt.size()) {
    std::cerr
        << "Flash-Next prompt and output exceed the 4096-token CLI context\n";
    return 1;
  }
  const std::vector<std::int32_t> input(prompt.begin(), prompt.end());
  std::string error;
  if (!session.Sync(input, &error)) {
    std::cerr << "Flash-Next prefill failed: " << error << '\n';
    return 1;
  }
  sampling::SamplerState sampler(opt.sampling, prompt);
  std::vector<tokenization::TokenId> generated;
  const auto start = std::chrono::steady_clock::now();
  while (generated.size() < opt.max_tokens) {
    models::qwen38_flash_next::Session::DecodeResult decoded;
    if (!session.DecodeStep(opt.max_tokens - generated.size(), sampler,
                            &decoded, &error)) {
      std::cerr << "Flash-Next decode failed: " << error << '\n';
      return 1;
    }
    for (const auto token : decoded.tokens) {
      const auto piece = model.TokenText(token);
      std::cout << piece << std::flush;
      if (reply)
        reply->append(piece);
      generated.push_back(static_cast<tokenization::TokenId>(token));
    }
    if (decoded.stop)
      break;
  }
  std::cout << '\n';
  if (opt.verbose) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::cerr << "Generated " << generated.size() << " tokens ("
              << generated.size() / seconds << " tok/s)\n";
    PrintTokenTrace(generated);
  }
  return 0;
}

std::unique_ptr<speculative::SpeculativeVerifier> CreateQwenVerifier(
    const PromptOptions& opt, hip::QwenGpuExecutor& executor) {
  std::string err;
  std::unique_ptr<speculative::IDraftBackend> draft_backend;
  if (opt.speculative_backend == "dflash2") {
    std::string dflash_path = opt.dflash_model_path;
    if (dflash_path.empty()) {
      throw std::invalid_argument("DFlash2 requires --dflash-model");
    }
    hip::QwenDFlashGpuDraftConfig cfg{
        .max_context = executor.GetMaxContext(),
        .max_draft_tokens = static_cast<std::uint32_t>(opt.draft_tokens),
        .policy = speculative::ParseDFlashDraftPolicy(opt.draft_policy),
    };
    draft_backend = hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
        dflash_path, executor.GetSharedModel(), cfg, &err);
    if (draft_backend == nullptr) {
      throw std::runtime_error("Failed to initialize DFlash backend: " + err);
    }
  } else if (opt.speculative_backend == "mtp") {
    std::string mtp_path = opt.mtp_model_path;
    if (mtp_path.empty()) {
      throw std::invalid_argument("MTP requires --mtp-model");
    }
    hip::QwenMtpGpuDraftConfig cfg{
        .max_context = executor.GetMaxContext(),
        .max_draft_tokens = static_cast<std::uint32_t>(opt.draft_tokens),
        .vision_input = &executor.VisionInput(),
    };
    draft_backend = hip::QwenMtpGpuDraftBackend::CreateFromGguf(
        mtp_path, executor.GetSharedModel(), cfg, &err);
    if (draft_backend == nullptr) {
      throw std::runtime_error("Failed to initialize MTP backend: " + err);
    }
  } else {
    throw std::invalid_argument("Unknown speculative backend: " +
                                opt.speculative_backend);
  }

  speculative::SpeculativeOptions s_opts;
  s_opts.max_draft_tokens = opt.draft_tokens;
  s_opts.min_draft_tokens = opt.min_draft_tokens;
  s_opts.initial_draft_tokens = opt.draft_tokens;
  const bool block_diffusion_draft = opt.speculative_backend == "dflash2";
  s_opts.enable_adaptive_draft_length = !block_diffusion_draft;
  s_opts.use_batched_verification = true;
  return std::make_unique<speculative::SpeculativeVerifier>(
      executor, std::move(draft_backend), s_opts);
}

void GenerateQwenGpuResponse(
    const PromptOptions& opt, hip::QwenGpuExecutor& executor,
    speculative::SpeculativeVerifier* verifier,
    std::span<const tokenization::TokenId> prompt_tokens,
    std::string* reply = nullptr) {
  models::GenerationOptions generation;
  generation.max_new_tokens = opt.max_tokens;
  generation.sampling = opt.sampling;
  std::size_t generated_count = 0;
  std::vector<tokenization::TokenId> generated_ids;
  if (opt.verbose)
    generated_ids.reserve(opt.max_tokens);
  const auto on_token = [&](tokenization::TokenId token,
                            std::string_view piece) {
    std::cout << piece << std::flush;
    if (reply != nullptr)
      reply->append(piece);
    ++generated_count;
    if (opt.verbose)
      generated_ids.push_back(token);
    return true;
  };
  const auto start = std::chrono::steady_clock::now();
  if (verifier != nullptr) {
    (void)verifier->Generate(prompt_tokens, generation, on_token);
  } else {
    (void)executor.Generate(prompt_tokens, generation, on_token);
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::cout << '\n';
  if (opt.verbose) {
    if (verifier != nullptr) {
      const auto& stats = verifier->GetStats();
      std::cerr << "[Speculative]: acceptance=" << stats.AcceptanceRate()
                << " drafted=" << stats.total_draft_tokens
                << " accepted=" << stats.total_accepted_tokens
                << " verification_steps=" << stats.total_verification_steps
                << '\n';
    }
    const double rate = seconds > 0 ? generated_count / seconds : 0;
    std::cout << "Generated " << generated_count << " tokens on GPU in "
              << seconds << "s (" << rate << " tok/s)\n";
    PrintTokenTrace(generated_ids);
  }
}

#endif

}  // namespace

static std::optional<PromptOptions> ParseTextOptions(
    std::span<const char* const> args, std::string* error_msg,
    std::string_view command) {
  PromptOptions opt;
  gufo::cli::ArgParser parser("gufo prompt",
                              "Execute one prompt request and exit.");
  bool speculative_explicit = false;
  RegisterTextOptions(parser, opt, speculative_explicit, command);

  if (!parser.Parse(args, error_msg)) {
    return std::nullopt;
  }
  if (parser.IsHelpRequested()) {
    return std::nullopt;
  }
  opt.sampling_supplied = SamplingOptionsSupplied(parser);
  try {
    opt.sampling.Validate();
  } catch (const std::invalid_argument& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return std::nullopt;
  }
  if ((!opt.image_paths.empty() || !opt.vision_model_path.empty()) &&
      (opt.force_cpu || !opt.use_chat_template)) {
    if (error_msg)
      *error_msg = "image input requires ROCm and chat framing";
    return std::nullopt;
  }
  if (!speculative_explicit && !opt.dspark_model_path.empty()) {
    opt.speculative_backend = "dspark";
  }
  if (opt.force_cpu && !opt.speculative_backend.empty()) {
    if (error_msg != nullptr) {
      *error_msg =
          "speculative decoding requires the ROCm backend; remove --cpu";
    }
    return std::nullopt;
  }
  const auto& backend = opt.speculative_backend;
  if (backend == "dflash2" && opt.dflash_model_path.empty()) {
    if (error_msg != nullptr)
      *error_msg = "DFlash2 requires --dflash-model";
    return std::nullopt;
  }
  if (backend == "mtp" && opt.mtp_model_path.empty()) {
    if (error_msg != nullptr)
      *error_msg = "MTP requires --mtp-model";
    return std::nullopt;
  }

  if (opt.draft_tokens == 0 || opt.min_draft_tokens == 0 ||
      opt.min_draft_tokens > opt.draft_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "min-draft-tokens cannot exceed draft-tokens";
    }
    return std::nullopt;
  }
  if (!opt.draft_policy.empty() &&
      ((opt.draft_policy != "fixed" && opt.draft_policy != "adaptive") ||
       opt.speculative_backend != "dflash2")) {
    if (error_msg != nullptr)
      *error_msg = "--draft-policy requires DFlash2 and fixed or adaptive";
    return std::nullopt;
  }
  if (opt.min_draft_tokens != 1 && backend == "dflash2") {
    if (error_msg != nullptr)
      *error_msg =
          "DFlash2 requires --min-draft-tokens 1; bound blocks with "
          "--draft-tokens";
    return std::nullopt;
  }
  if (opt.reasoning_mode != "on" && opt.reasoning_mode != "off" &&
      opt.reasoning_mode != "auto") {
    if (error_msg != nullptr) {
      *error_msg = "--think must be on, off, or auto";
    }
    return std::nullopt;
  }
  if (opt.reasoning_effort != "auto" &&
      !ParseReasoningEffort(opt.reasoning_effort).has_value()) {
    if (error_msg != nullptr) {
      *error_msg =
          "--reasoning-effort must be auto, minimal, low, medium, high, "
          "xhigh, or max";
    }
    return std::nullopt;
  }
  if (opt.reasoning_mode == "off" && opt.reasoning_effort != "auto") {
    if (error_msg != nullptr) {
      *error_msg = "--reasoning-effort cannot be set while --think is off";
    }
    return std::nullopt;
  }
  if (opt.preserve_thinking != "on" && opt.preserve_thinking != "off" &&
      opt.preserve_thinking != "auto") {
    if (error_msg != nullptr) {
      *error_msg = "--preserve-thinking must be on, off, or auto";
    }
    return std::nullopt;
  }
  return opt;
}

std::optional<PromptOptions> ParsePromptOptions(
    std::span<const char* const> args, std::string* error_msg) {
  return ParseTextOptions(args, error_msg, "prompt");
}

int RunPrompt(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParsePromptOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintPromptHelp("gufo");
      return 2;
    }
    PrintPromptHelp("gufo");
    return 0;
  }

  auto opt = *opt_res;

  if (!opt.prompt_file.empty()) {
    std::ifstream file(opt.prompt_file);
    if (!file.is_open()) {
      std::cerr << "Error: could not open prompt file '" << opt.prompt_file
                << "'\n";
      return 1;
    }
    std::string file_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    if (!opt.prompt_text.empty()) {
      opt.prompt_text = file_content + "\n" + opt.prompt_text;
    } else {
      opt.prompt_text = std::move(file_content);
    }
  }

  if (opt.prompt_text.empty() && opt.model_path.empty()) {
    PrintPromptHelp("gufo");
    return 0;
  }

  if (opt.model_path.empty()) {
    std::cout
        << "gufo prompt: prompt received: \"" << opt.prompt_text << "\"\n"
        << "(Specify --model <PATH.gguf> to execute local model generation)\n";
    return 0;
  }

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  auto reader_owner = gufo::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader_owner) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));
  ResolvePromptSampling(*reader, &opt);

#if defined(ENGINE_ENABLE_HIP)
  if (IsDeepSeekV4Flash(*reader)) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "DeepSeek does not support image input\n";
      return 2;
    }
    return RunDeepSeekPrompt(opt, *reader, model_load_start);
  }
#else
  if (IsDeepSeekV4Flash(*reader)) {
    std::cerr << "DeepSeek V4 Flash requires ENGINE_ENABLE_HIP=ON\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
#endif

  if (IsLaguna(*reader)) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "Laguna does not support image input\n";
      return 2;
    }
#ifdef ENGINE_ENABLE_HIP
    if (!opt.force_cpu) {
      return RunLagunaPromptGPU(opt, *reader, model_load_start);
    }
#endif
    return RunLagunaPromptCPU(opt, *reader, model_load_start);
  }
  if (IsFlint(*reader)) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "Flint does not support image input\n";
      return 2;
    }
    return RunFlintPrompt(opt, *reader, model_load_start);
  }
  if (IsGlm53Flash(*reader)) {
#if defined(ENGINE_ENABLE_HIP)
    if (!opt.force_cpu) {
      if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
        std::cerr << "GLM-5.3 does not support image input\n";
        return 2;
      }
      return RunGlmPrompt(opt, *reader, model_load_start);
    }
#endif
    return RunGlm53Prompt(opt, *reader, model_load_start);
  }

  std::string rendered_prompt = opt.prompt_text;
  std::vector<tokenization::ContentSpan> content_spans;
  std::vector<tokenization::ChatMessage> messages;
  if (opt.use_chat_template) {
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    const auto rendered = tokenization::QwenChatTemplate::Render(
        messages, {},
        tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
        nullptr, nullptr, nullptr, &content_spans);
    if (rendered.has_value()) {
      rendered_prompt = *rendered;
    } else {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }
  }

  const auto encode_prompt = [&](const tokenization::QwenTokenizer& tokenizer) {
    return tokenization::QwenChatTemplate::EncodeRendered(
        tokenizer, rendered_prompt, 0, rendered_prompt.size(), content_spans,
        {});
  };

#if defined(ENGINE_ENABLE_HIP)
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {
    auto model = LoadFlashNextModel(opt, *reader, model_load_start);
    if (!model)
      return 1;
    auto session =
        model->CreateSession(opt.speculative_backend.empty()
                                 ? gufo::core::SessionMode::kAutoregressive
                                 : gufo::core::SessionMode::kSpeculative,
                             kDefaultContext, &err);
    if (!session) {
      std::cerr << "Flash-Next session failed: " << err << '\n';
      return 1;
    }
    try {
      if (!opt.image_paths.empty()) {
        AttachImages(opt, messages.back());
        auto vision = PrepareVision(opt, model->tokenizer(), messages,
                                    model->VisionEncoder());
        session->ConfigureVision(vision);
        return GenerateFlashNextResponse(opt, *model, *session, vision->tokens);
      }
      const auto ids = encode_prompt(model->tokenizer());
      const std::vector<tokenization::TokenId> prompt(ids.begin(), ids.end());
      return GenerateFlashNextResponse(opt, *model, *session, prompt);
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
  int dev_count = 0;
  if (!opt.force_cpu && hipGetDeviceCount(&dev_count) == hipSuccess &&
      dev_count > 0) {
    auto gpu_exec = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &err);
    if (gpu_exec) {
      PrintModelLoadTime(model_load_start);
      auto prompt_tokens = encode_prompt(gpu_exec->GetTokenizer());
      try {
        auto encoder = LoadQwenVision(opt, *reader);
        if (!opt.image_paths.empty()) {
          AttachImages(opt, messages.back());
          auto vision =
              PrepareVision(opt, gpu_exec->GetTokenizer(), messages, encoder);
          prompt_tokens = vision->tokens;
          gpu_exec->ConfigureVision(std::move(vision), std::move(encoder));
        }
      } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
      }

      if (opt.verbose) {
        const auto& config = gpu_exec->GetConfig();
        std::cout << "[Engine]: AMD Strix Halo gfx1151 GPU Executor\n"
                  << "Model: " << config.architecture << " ("
                  << config.num_layers
                  << " layers, hidden=" << config.hidden_size
                  << ", heads=" << config.num_attention_heads << ")\n"
                  << "Prompt tokens (" << prompt_tokens.size() << "): ";
        for (const auto t : prompt_tokens) {
          std::cout << "[" << t << ": '"
                    << gpu_exec->GetTokenizer().DecodeToken(t) << "'] ";
        }
        std::cout << "\nMax tokens: " << opt.max_tokens << "\n"
                  << "--- Generation Output ---\n";
      }

      std::unique_ptr<speculative::SpeculativeVerifier> verifier;
      try {
        if (!opt.speculative_backend.empty()) {
          verifier = CreateQwenVerifier(opt, *gpu_exec);
        }
        GenerateQwenGpuResponse(opt, *gpu_exec, verifier.get(), prompt_tokens);
      } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
      }
      return 0;
    }
    std::cerr << "Error creating Qwen GPU executor: " << err << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
#endif

  if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
    std::cerr << "Image input requires ROCm\n";
    return 1;
  }
  if (!opt.speculative_backend.empty()) {
    std::cerr << "Qwen speculative decoding requires the ROCm backend\n";
    return 1;
  }
  auto generator = models::QwenGenerator::CreateFromGguf(*reader, &err);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto prompt_tokens = encode_prompt(generator->GetTokenizer());

  if (opt.verbose) {
    const auto& config = generator->GetConfig();
    std::cout << "[Engine]: CPU (OpenMP Multi-Threaded)\n"
              << "Model: " << config.architecture << " (" << config.num_layers
              << " layers, hidden=" << config.hidden_size
              << ", heads=" << config.num_attention_heads << ")\n"
              << "Prompt tokens: " << prompt_tokens.size() << "\n"
              << "Max tokens: " << opt.max_tokens << "\n"
              << "Temperature: " << opt.sampling.temperature << "\n"
              << "--- Generation Output ---\n";
  }

  models::GenerationOptions gen_opts;
  gen_opts.max_new_tokens = opt.max_tokens;
  gen_opts.sampling = opt.sampling;

  const auto start_time = std::chrono::steady_clock::now();
  std::size_t generated_count = 0;

  const auto tokens = generator->Generate(
      prompt_tokens, gen_opts,
      [&](tokenization::TokenId, std::string_view piece) -> bool {
        std::cout << piece << std::flush;
        ++generated_count;
        return true;
      });

  std::cout << "\n";

  if (opt.verbose && generated_count > 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    const double sec = static_cast<double>(elapsed.count()) / 1000.0;
    const double tok_per_sec =
        (sec > 0.0) ? (static_cast<double>(generated_count) / sec) : 0.0;
    std::cout << "Generated " << generated_count << " tokens on CPU in " << sec
              << "s (" << tok_per_sec << " tok/s)\n";
  }

  return 0;
}

int RunChat(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParseTextOptions(args, &parse_err, "chat");
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintChatHelp("gufo");
      return 2;
    }
    PrintChatHelp("gufo");
    return 0;
  }

  auto opt = *opt_res;
  if (opt.model_path.empty()) {
    std::cout << "gufo chat: interactive conversation mode\n"
              << "(Specify --model <PATH.gguf> to load model weights)\n";
    return 0;
  }

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  auto reader_owner = gufo::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader_owner) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));
  ResolvePromptSampling(*reader, &opt);

#if defined(ENGINE_ENABLE_HIP)
  if (IsGlm53Flash(*reader) && !opt.force_cpu) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "GLM-5.3 does not support image input\n";
      return 2;
    }
    return RunGlmChat(opt, *reader, model_load_start);
  }
  if (IsDeepSeekV4Flash(*reader)) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "DeepSeek does not support image input\n";
      return 2;
    }
    return RunDeepSeekChat(opt, *reader, model_load_start);
  }
#else
  if (IsDeepSeekV4Flash(*reader)) {
    std::cerr << "DeepSeek V4 Flash requires ENGINE_ENABLE_HIP=ON\n";
    return 1;
  }
#endif

  const tokenization::QwenTokenizer* tokenizer = nullptr;
  std::string architecture;
  std::unique_ptr<models::QwenGenerator> generator;
#if defined(ENGINE_ENABLE_HIP)
  std::unique_ptr<hip::QwenGpuExecutor> gpu_executor;
  std::shared_ptr<models::qwen::vision::Encoder> vision_encoder;
  std::unique_ptr<speculative::SpeculativeVerifier> verifier;
  std::shared_ptr<models::qwen38_flash_next::Model> flash_model;
  std::unique_ptr<models::qwen38_flash_next::Session> flash_session;
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {
    flash_model = LoadFlashNextModel(opt, *reader, model_load_start);
    if (!flash_model)
      return 1;
    flash_session = flash_model->CreateSession(
        opt.speculative_backend.empty()
            ? gufo::core::SessionMode::kAutoregressive
            : gufo::core::SessionMode::kSpeculative,
        kDefaultContext, &err);
    if (!flash_session) {
      std::cerr << "Flash-Next session failed: " << err << '\n';
      return 1;
    }
    tokenizer = &flash_model->tokenizer();
    architecture = "qwen4exp";
    vision_encoder = flash_model->VisionEncoder();
  }
  int device_count = 0;
  if (tokenizer == nullptr && !opt.force_cpu &&
      hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0) {
    gpu_executor = hip::QwenGpuExecutor::CreateFromGguf(reader, &err);
    if (!gpu_executor) {
      std::cerr << "Error creating Qwen GPU executor: " << err << '\n';
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    try {
      vision_encoder = LoadQwenVision(opt, *reader);
      if (!opt.speculative_backend.empty()) {
        verifier = CreateQwenVerifier(opt, *gpu_executor);
      }
    } catch (const std::exception& exception) {
      std::cerr << exception.what() << '\n';
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    tokenizer = &gpu_executor->GetTokenizer();
    architecture = gpu_executor->GetConfig().architecture;
    if (opt.verbose)
      std::cout << "[Engine]: AMD Strix Halo gfx1151 GPU Executor\n";
  }
#endif
  if (tokenizer == nullptr) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "Image input requires ROCm\n";
      return 1;
    }
    if (!opt.speculative_backend.empty()) {
      std::cerr << "Qwen speculative decoding requires the ROCm backend\n";
      return 1;
    }
    generator = models::QwenGenerator::CreateFromGguf(*reader, &err);
    if (!generator) {
      std::cerr << "Error creating Qwen generator: " << err << '\n';
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    tokenizer = &generator->GetTokenizer();
    architecture = generator->GetConfig().architecture;
    if (opt.verbose)
      std::cout << "[Engine]: CPU (OpenMP Multi-Threaded)\n";
  }
  PrintModelLoadTime(model_load_start);

  std::cout << "=== Gufo Interactive Chat (" << architecture << ") ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";

  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }

  bool first_turn = true;
  std::string user_input;
  while (true) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, user_input)) {
      break;
    }
    if (user_input == "exit" || user_input == "quit") {
      break;
    }
    if (user_input.empty()) {
      continue;
    }

    history.push_back({tokenization::ChatRole::kUser, user_input, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    std::vector<tokenization::ContentSpan> content_spans;
    const auto rendered_prompt = tokenization::QwenChatTemplate::Render(
        history, {},
        tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
        nullptr, nullptr, nullptr, &content_spans);
    if (!rendered_prompt.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }

    auto prompt_tokens = tokenization::QwenChatTemplate::EncodeRendered(
        *tokenizer, *rendered_prompt, 0, rendered_prompt->size(), content_spans,
        {});

    std::cout << "<<< Assistant: ";
    std::string assistant_reply;

    try {
#if defined(ENGINE_ENABLE_HIP)
      if (first_turn)
        AttachImages(opt, history.back());
      first_turn = false;
      if (!opt.image_paths.empty()) {
        auto vision = PrepareVision(opt, *tokenizer, history, vision_encoder);
        prompt_tokens = vision->tokens;
        if (flash_session)
          flash_session->ConfigureVision(vision);
        if (gpu_executor)
          gpu_executor->ConfigureVision(vision, vision_encoder);
      }
      if (flash_model) {
        if (GenerateFlashNextResponse(opt, *flash_model, *flash_session,
                                      prompt_tokens, &assistant_reply) != 0)
          return 1;
      } else if (gpu_executor != nullptr) {
        GenerateQwenGpuResponse(opt, *gpu_executor, verifier.get(),
                                prompt_tokens, &assistant_reply);
      } else
#endif
      {
        models::GenerationOptions generation;
        generation.max_new_tokens = opt.max_tokens;
        generation.sampling = opt.sampling;
        (void)generator->Generate(
            prompt_tokens, generation,
            [&](tokenization::TokenId, std::string_view piece) {
              std::cout << piece << std::flush;
              assistant_reply += piece;
              return true;
            });
        std::cout << '\n';
      }
    } catch (const std::exception& exception) {
      std::cerr << "Generation failed: " << exception.what() << '\n';
      return 1;
    }

    std::cout << '\n';
    tokenization::ChatMessage reply{tokenization::ChatRole::kAssistant, ""};
    if (tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id)
            .enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = assistant_reply.find(end);
      reply.thought = assistant_reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        reply.content = assistant_reply.substr(boundary + end.size());
      }
    } else {
      reply.content = std::move(assistant_reply);
    }
    history.push_back(std::move(reply));
  }

  return 0;
}

}  // namespace gufo::cli
