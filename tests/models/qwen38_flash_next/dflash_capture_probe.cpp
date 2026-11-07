// Streams DFlash2 tap-layer features for text documents to disk.
//
// dflash_capture_probe --model FIRST_SHARD.gguf [--prompt-file doc.txt
//                      --out PREFIX | --corpus DIR --out DIR --stem STEM]
//                      [--layers 5,15,25,35,45] [--batch N] [--context N]
//                      [--raw]
//
// The forward pass is rendered through the Qwen chat template unless --raw
// is given. Each document row is stored exactly once, bf16, as the
// [taps][hidden] residual-mean feature vector that the drafter consumes;
// companion files hold the token ids and a metadata manifest.
//
// With --corpus, every regular file under DIR is captured in sorted order to
// OUT/STEM-%04d.* (indices match the per-file dflash-capture.sh naming). The
// model upload happens once and the session is reset between documents, and a
// document whose manifest already exists is skipped while one that fails is
// reported and skipped, so a corpus pass is resumable and one bad document
// never ends the run.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {

void Fp32ToBf16(const float* source, std::size_t count, std::uint16_t* sink) {
  for (std::size_t index = 0; index < count; ++index) {
    std::uint32_t bits;
    std::memcpy(&bits, source + index, sizeof(bits));
    bits = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    sink[index] = static_cast<std::uint16_t>(bits >> 16U);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt_path;
  std::string out_prefix;
  std::string corpus_dir;
  std::string stem;
  std::vector<std::uint32_t> layers{5, 15, 25, 35, 45};
  std::uint32_t batch = 512;
  std::uint32_t context = 4096;
  bool raw = false;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string arg = argv[argument];
    auto next = [&]() -> std::string {
      return argument + 1 < argc ? argv[++argument] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--prompt-file") {
      prompt_path = next();
    } else if (arg == "--out") {
      out_prefix = next();
    } else if (arg == "--layers") {
      layers.clear();
      const std::string_view value(next());
      std::size_t start = 0;
      while (start < value.size()) {
        const auto comma = value.find(',', start);
        const auto piece = value.substr(start, comma == std::string_view::npos
                                                   ? std::string_view::npos
                                                   : comma - start);
        std::uint32_t parsed = 0;
        const auto* begin = piece.data();
        const auto* end = begin + piece.size();
        if (const auto result = std::from_chars(begin, end, parsed);
            result.ec != std::errc{} || result.ptr != end || parsed >= 256) {
          std::fprintf(stderr, "--layers needs comma-separated layer ids\n");
          return 2;
        }
        layers.push_back(parsed);
        start = comma == std::string_view::npos ? value.size() : comma + 1;
      }
    } else if (arg == "--batch") {
      const auto value = next();
      const auto* begin = value.data();
      const auto* end = begin + value.size();
      if (const auto result = std::from_chars(begin, end, batch);
          result.ec != std::errc{} || result.ptr != end || batch == 0) {
        std::fprintf(stderr, "--batch requires a positive integer\n");
        return 2;
      }
    } else if (arg == "--context") {
      const auto value = next();
      const auto* begin = value.data();
      const auto* end = begin + value.size();
      if (const auto result = std::from_chars(begin, end, context);
          result.ec != std::errc{} || result.ptr != end || context == 0) {
        std::fprintf(stderr, "--context requires a positive integer\n");
        return 2;
      }
    } else if (arg == "--raw") {
      raw = true;
    } else if (arg == "--corpus") {
      corpus_dir = next();
    } else if (arg == "--stem") {
      stem = next();
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty() || out_prefix.empty() ||
      (prompt_path.empty() == corpus_dir.empty()) ||
      (!corpus_dir.empty() && stem.empty())) {
    std::fprintf(stderr,
                 "--model and --out plus exactly one of --prompt-file or "
                 "(--corpus with --stem) are required\n");
    return 2;
  }
  std::sort(layers.begin(), layers.end());
  if (layers.empty()) {
    std::fprintf(stderr, "--layers needs at least one tap layer\n");
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  const auto& config = weights->config;
  for (const auto layer : layers) {
    if (layer >= config.num_layers) {
      std::fprintf(stderr, "layer %u exceeds the %u model layers\n", layer,
                   config.num_layers);
      return 2;
    }
  }

  std::unique_ptr<q::NgramTable> ngram;
  if (config.ple_layer >= 0) {
    const auto& table = weights->ple_table;
    ngram = q::NgramTable::Open(
        reader->GetMappedRegions()[table.shard].file_descriptor,
        table.file_offset, table.rows, config.ple_head_dim, table.type, &error);
    if (!ngram) {
      std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
      return 1;
    }
  }
  auto device =
      q::rocm::DeviceModel::Upload(*weights, *reader, nullptr, nullptr, &error);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  q::rocm::Executor::Options options;
  options.max_batch = batch;
  options.max_logit_rows = 1;
  options.dflash_rows = batch;
  options.dflash_layers = layers;
  auto executor =
      q::rocm::Executor::Create(*device, ngram.get(), options, &error);
  if (!executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }
  auto session = executor->CreateSession(
      gufo::core::SessionMode::kAutoregressive, context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  const std::size_t tap_width = layers.size() * config.hidden_size;
  std::vector<float> features(static_cast<std::size_t>(batch) * tap_width);
  std::vector<std::uint16_t> half(static_cast<std::size_t>(batch) * tap_width);

  struct Document {
    std::string text;
    std::string prefix;
  };
  std::vector<Document> documents;
  if (corpus_dir.empty()) {
    std::ifstream input(prompt_path, std::ios::binary);
    if (!input) {
      std::fprintf(stderr, "cannot open %s\n", prompt_path.c_str());
      return 2;
    }
    documents.push_back({std::string((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>()),
                         out_prefix});
    if (documents.back().text.empty()) {
      std::fprintf(stderr, "cannot read %s\n", prompt_path.c_str());
      return 2;
    }
  } else {
    std::filesystem::create_directories(out_prefix);
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(corpus_dir)) {
      if (entry.is_regular_file()) {
        files.push_back(entry.path());
      }
    }
    std::sort(files.begin(), files.end());
    for (std::size_t index = 0; index < files.size(); ++index) {
      char number[16];
      std::snprintf(number, sizeof(number), "%04zu", index + 1);
      const std::string prefix =
          (std::filesystem::path(out_prefix) / (stem + "_" + number)).string();
      if (std::filesystem::exists(prefix + ".meta.json")) {
        std::printf("skip done %s\n", files[index].c_str());
        continue;
      }
      std::ifstream input(files[index], std::ios::binary);
      if (!input) {
        std::fprintf(stderr, "WARN: cannot open %s\n", files[index].c_str());
        continue;
      }
      documents.push_back({std::string((std::istreambuf_iterator<char>(input)),
                                       std::istreambuf_iterator<char>()),
                           prefix});
    }
  }

  std::size_t failed = 0;
  for (std::size_t number = 0; number < documents.size(); ++number) {
    const std::string& document = documents[number].text;
    const std::string& prefix = documents[number].prefix;
    // Documents share one session; rewind the cache instead of paying for a
    // fresh session (and model-side allocations) per document.
    session->Reset();
    std::string rendered = document;
    if (!raw) {
      const std::vector<gufo::tokenization::ChatMessage> messages{
          {gufo::tokenization::ChatRole::kSystem,
           "You are a helpful, respectful, and honest assistant.", "", ""},
          {gufo::tokenization::ChatRole::kUser, document, "", ""},
      };
      const auto chat = gufo::tokenization::QwenChatTemplate::Render(
          messages, {.add_generation_prompt = true, .enable_thinking = false});
      if (!chat) {
        std::fprintf(stderr, "WARN: chat template failed for %s\n",
                     prefix.c_str());
        ++failed;
        continue;
      }
      rendered = *chat;
    }
    std::vector<std::int32_t> tokens;
    for (auto id : tokenizer->Encode(rendered)) {
      tokens.push_back(static_cast<std::int32_t>(id));
    }
    if (tokens.empty() || tokens.size() > context) {
      std::fprintf(stderr, "WARN: %s has %zu tokens, limit %u\n",
                   prefix.c_str(), tokens.size(), context);
      ++failed;
      continue;
    }
    if (tokens.size() < 9) {
      std::fprintf(stderr, "WARN: %s needs at least 9 tokens\n",
                   prefix.c_str());
      ++failed;
      continue;
    }

    std::ofstream tokens_file(prefix + ".tokens.i32", std::ios::binary);
    std::ofstream features_file(prefix + ".features.f16", std::ios::binary);
    if (!tokens_file || !features_file) {
      std::fprintf(stderr, "WARN: cannot open output files under %s\n",
                   prefix.c_str());
      ++failed;
      continue;
    }

    bool ok = true;
    std::size_t captured = 0;
    for (std::size_t offset = 0; offset < tokens.size(); offset += batch) {
      const std::size_t count =
          std::min<std::size_t>(batch, tokens.size() - offset);
      if (!executor->Forward(
              *session,
              std::span<const std::int32_t>(tokens.data() + offset, count), 0,
              nullptr, q::rocm::Executor::ForwardMode::kPrefill, &error,
              nullptr)) {
        std::fprintf(stderr, "WARN: prefill failed for %s: %s\n",
                     prefix.c_str(), error.c_str());
        ok = false;
        break;
      }
      if (session->dflash_capture_rows() < count) {
        std::fprintf(stderr, "WARN: capture ring fell below the chunk width\n");
        ok = false;
        break;
      }
      const float* device_features = executor->dflash_capture();
      if (device_features == nullptr) {
        std::fprintf(stderr, "WARN: capture disabled\n");
        ok = false;
        break;
      }
      if (hipMemcpy(features.data(), device_features,
                    count * tap_width * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        std::fprintf(stderr, "WARN: feature copy failed\n");
        ok = false;
        break;
      }
      Fp32ToBf16(features.data(), count * tap_width, half.data());
      tokens_file.write(
          reinterpret_cast<const char*>(tokens.data() + offset),
          static_cast<std::streamsize>(count * sizeof(std::int32_t)));
      features_file.write(reinterpret_cast<const char*>(half.data()),
                          static_cast<std::streamsize>(count * tap_width *
                                                       sizeof(std::uint16_t)));
      captured += count;
    }
    tokens_file.close();
    features_file.close();
    if (!ok) {
      std::filesystem::remove(prefix + ".tokens.i32");
      std::filesystem::remove(prefix + ".features.f16");
      ++failed;
      continue;
    }
    if (!tokens_file || !features_file) {
      std::fprintf(stderr, "WARN: cannot write output files under %s\n",
                   prefix.c_str());
      ++failed;
      continue;
    }
    std::ofstream manifest(prefix + ".meta.json");
    if (!manifest) {
      std::fprintf(stderr, "WARN: cannot open %s.meta.json\n", prefix.c_str());
      ++failed;
      continue;
    }
    manifest << "{\"rows\": " << captured << ", \"tokens\": " << tokens.size()
             << ", \"taps\": [";
    for (std::size_t index = 0; index < layers.size(); ++index) {
      if (index != 0) {
        manifest << ", ";
      }
      manifest << layers[index];
    }
    manifest << "], \"hidden\": " << config.hidden_size
             << ", \"chat\": " << (raw ? "false" : "true") << "}\n";
    manifest.close();
    if (!manifest) {
      std::fprintf(stderr, "WARN: cannot write manifest for %s\n",
                   prefix.c_str());
      std::filesystem::remove(prefix + ".meta.json");
      ++failed;
      continue;
    }
    std::printf("%zu feature rows, %zu tokens -> %s\n", captured, tokens.size(),
                prefix.c_str());
  }
  if (failed != 0) {
    std::fprintf(stderr, "captured %zu of %zu documents\n",
                 documents.size() - failed, documents.size());
    return 1;
  }
  return 0;
}
