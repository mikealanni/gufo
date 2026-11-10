#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/decision_forward.hpp"

// Usage: lfm3_rowcheck MODEL.gguf  (reads /tmp/rowtest_*.{txt,f32} from the NPU row test)
int main(int argc, char** argv) {
  using namespace gufo::models::lfm3;
  if (argc < 2) return 2;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  auto weights = reader ? Weights::Bind(*reader, &error) : std::nullopt;
  if (!weights) { std::cerr << error << '\n'; return 1; }
  std::ifstream ids_in("/tmp/rowtest_ids.txt");
  std::string l1, l2; std::getline(ids_in, l1); std::getline(ids_in, l2);
  std::vector<std::uint32_t> tokens, seq;
  { std::istringstream a(l1); for (std::uint32_t v; a >> v;) tokens.push_back(v); }
  { std::istringstream b(l2); for (std::uint32_t v; b >> v;) seq.push_back(v); }
  const std::size_t prompt = tokens.size();
  tokens.insert(tokens.end(), seq.begin(), seq.end());
  std::vector<std::uint32_t> slots;
  for (std::size_t i = 0; i < seq.size(); ++i) slots.push_back(prompt + i);
  std::vector<float> host;
  const std::uint32_t one = 0;
  DecisionForward(*weights, tokens, {}, slots, std::span(&one, 1), true, 4, &host);
  const std::size_t H = weights->config.hidden;
  auto load = [&](const char* path) {
    std::ifstream f(path, std::ios::binary); std::vector<float> v(seq.size() * H);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * 4); return v; };
  const auto sq = load("/tmp/rowtest_seq.f32"), mu = load("/tmp/rowtest_multi.f32");
  for (std::size_t r = 0; r < seq.size(); ++r) {
    double ds = 0, dm = 0, sc = 0;
    for (std::size_t j = 0; j < H; ++j) {
      ds = std::max(ds, double(std::abs(sq[r * H + j] - host[r * H + j])));
      dm = std::max(dm, double(std::abs(mu[r * H + j] - host[r * H + j])));
      sc = std::max(sc, double(std::abs(host[r * H + j])));
    }
    std::cout << "row " << r << ": |seq-host| " << ds << "  |multi-host| " << dm << "  (max|h| " << sc << ")\n";
  }
}
