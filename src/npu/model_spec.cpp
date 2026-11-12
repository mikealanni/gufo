#include "src/npu/model_spec.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "src/core/json.hpp"

namespace gufo::npu {
namespace {

std::string GetStr(const gufo::json::Value& o, const std::string& k,
                   const std::string& def = "") {
  return o.member_str(k, def);
}
std::uint32_t GetU32(const gufo::json::Value& o, const std::string& k,
                     std::uint32_t def = 0) {
  return static_cast<std::uint32_t>(o.member_size(k, def));
}
double GetF(const gufo::json::Value& o, const std::string& k, double def = 0.0) {
  return o.member_double(k, def);
}
RopeKind ParseRope(const std::string& s) {
  if (s == "scaled" || s == "swa_scaled")
    return RopeKind::kScaled;
  if (s == "none")
    return RopeKind::kNone;
  return RopeKind::kNormal;
}
FfnKind ParseFfn(const std::string& s) {
  if (s == "geglu")
    return FfnKind::kGeGLU;
  if (s == "plain")
    return FfnKind::kPlain;
  return FfnKind::kSwiGLU;
}
LayerKind ParseLayerKind(const std::string& s) {
  if (s == "conv" || s == "convolution" || s == "shortconv")
    return LayerKind::kConvolution;
  return LayerKind::kAttention;
}

}  // namespace

ModelSpec ModelSpec::Lfm3_1_7B() {
  ModelSpec s;
  s.name = "LFM3-1.7B";
  s.hidden = 2048;
  s.vocab = 0;  // from GGUF
  s.layers = 24;
  s.context = 262144;
  s.norm_eps = 1e-5f;
  s.output_norm = "output_norm.weight";
  s.embedding.tensor = "token_embd.weight";
  s.embedding.bits = BitWidth::kQ8;
  s.embedding.tied_output = true;
  s.rope_freqs = "rope_freqs.weight";
  s.rope_theta = 1000000.0f;
  s.rope_theta_swa = 10000.0f;
  s.eos_id = 7;

  LayerSpec attn;
  attn.kind = LayerKind::kAttention;
  attn.norm = "operator_norm.weight";
  attn.post_norm = "ffn_norm.weight";  // LFM3 naming: post-attn norm in our usage
  attn.ffn_norm = "ffn_norm.weight";
  attn.post_ffn_norm = "post_ffn_norm.weight";
  attn.attention.heads = 16;
  attn.attention.kv_heads = 4;
  attn.attention.head_dim = 128;
  attn.attention.window = 0;
  attn.attention.rope = RopeKind::kNormal;
  attn.attention.rope_theta = 1000000.0f;
  attn.attention.has_gate = true;
  attn.ffn.kind = FfnKind::kSwiGLU;
  attn.ffn.intermediate = 7680;

  LayerSpec conv = attn;
  conv.kind = LayerKind::kConvolution;
  conv.conv.width = 4;
  conv.attention.window = 512;

  // LFM3-1.7B pattern is model-defined; caller can override layer_specs.
  // Default to all-attention except we leave empty and let loader/engine use config.
  s.layer_specs.clear();
  return s;
}

std::vector<ModelSpec::ProgramNeed> ModelSpec::RequiredPrograms() const {
  std::vector<ProgramNeed> needs;
  auto add = [&](std::uint32_t K, std::uint32_t N, const std::string& mode) {
    for (auto& n : needs)
      if (n.K == K && n.N == N && n.mode == mode)
        return;
    ProgramNeed pn{K, N, mode, ""};
    if (mode == "ffn") {
      pn.inst_file = "insts_ffn.txt";
    } else {
      pn.inst_file = "insts_plain_" + std::to_string(K) + "x" + std::to_string(N) + ".txt";
    }
    needs.push_back(pn);
  };
  // Derive from spec. Conservative: include all GEMM shapes referenced.
  // Attention: qkv (H, heads*hd + 2*kv_heads*hd), out (heads*hd, H)
  // Conv: in_proj (H, 3*H), out_proj (H,H)
  // FFN: fused gate_up+down handled by ffn program; separate projections if not fused.
  std::uint32_t H = hidden;
  for (const auto& ls : layer_specs) {
    if (ls.kind == LayerKind::kConvolution) {
      add(H, 3 * H, "plain");
      add(H, H, "plain");
    } else {
      const auto& a = ls.attention;
      std::uint32_t qd = a.heads * a.head_dim;
      std::uint32_t kvd = a.kv_heads * a.head_dim;
      add(H, qd + 2 * kvd, "plain");
      add(H, H, "plain");
    }
    // fused FFN
    add(H, H, "ffn");
  }
  if (needs.empty()) {
    // fallback for LFM3-style when layer_specs not enumerated explicitly
    add(H, 3 * H, "plain");
    add(H, H, "plain");
    add(H, H, "ffn");
  }
  return needs;
}

std::vector<std::string> ModelSpec::ValidateConstraints() const {
  std::vector<std::string> issues;
  auto check = [&](std::uint32_t K, std::uint32_t N, const char* ctx) {
    if (K % 64 != 0)
      issues.push_back(std::string(ctx) + ": K must be multiple of 64");
    if (N % 1024 != 0)
      issues.push_back(std::string(ctx) + ": N must be multiple of 1024 (pad/mask)");
  };
  for (const auto& p : RequiredPrograms()) {
    if (p.mode == "ffn") continue; // ffn program has fixed shapes
    check(p.K, p.N, "program");
  }
  return issues;
}

std::optional<ModelSpec> ModelSpec::LoadJsonFile(const std::string& path,
                                                 std::string* error) {
  std::ifstream f(path);
  if (!f) {
    if (error) *error = "cannot open spec: " + path;
    return std::nullopt;
  }
  std::string text((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  try {
    auto j = gufo::json::parse(text);
    ModelSpec s = Lfm3_1_7B();
    s.name = GetStr(j, "name", s.name);
    s.hidden = GetU32(j, "hidden", s.hidden);
    s.vocab = GetU32(j, "vocab", s.vocab);
    s.layers = GetU32(j, "layers", s.layers);
    s.context = GetU32(j, "context", s.context);
    s.norm_eps = static_cast<float>(GetF(j, "norm_eps", s.norm_eps));
    s.output_norm = GetStr(j, "output_norm", s.output_norm);
    s.rope_freqs = GetStr(j, "rope_freqs", s.rope_freqs);
    s.rope_theta = static_cast<float>(GetF(j, "rope_theta", s.rope_theta));
    s.rope_theta_swa = static_cast<float>(GetF(j, "rope_theta_swa", s.rope_theta_swa));
    s.eos_id = GetU32(j, "eos_id", s.eos_id);
    if (auto* emb = j.find("embedding")) {
      s.embedding.tensor = GetStr(*emb, "tensor", s.embedding.tensor);
      s.embedding.tied_output = emb->find("tied_output") ? emb->find("tied_output")->as_bool(true) : true;
    }
    if (auto* larr = j.find("layer_specs"); larr && larr->is_array()) {
      s.layer_specs.clear();
      for (const auto& lv : larr->items()) {
        LayerSpec ls;
        ls.kind = ParseLayerKind(GetStr(lv, "kind", "attention"));
        ls.norm = GetStr(lv, "norm", "operator_norm.weight");
        ls.post_norm = GetStr(lv, "post_norm", "");
        ls.ffn_norm = GetStr(lv, "ffn_norm", "ffn_norm.weight");
        ls.post_ffn_norm = GetStr(lv, "post_ffn_norm", "");
        if (auto* a = lv.find("attention")) {
          ls.attention.heads = GetU32(*a, "heads", 0);
          ls.attention.kv_heads = GetU32(*a, "kv_heads", 0);
          ls.attention.head_dim = GetU32(*a, "head_dim", 0);
          ls.attention.window = GetU32(*a, "window", 0);
          ls.attention.rope = ParseRope(GetStr(*a, "rope", "normal"));
          ls.attention.rope_theta = static_cast<float>(GetF(*a, "rope_theta", s.rope_theta));
          ls.attention.qk_norm = a->find("qk_norm") ? a->find("qk_norm")->as_bool(false) : false;
          ls.attention.has_gate = a->find("has_gate") ? a->find("has_gate")->as_bool(true) : true;
          ls.attention.q_proj = GetStr(*a, "q_proj", ls.attention.q_proj);
          ls.attention.k_proj = GetStr(*a, "k_proj", ls.attention.k_proj);
          ls.attention.v_proj = GetStr(*a, "v_proj", ls.attention.v_proj);
          ls.attention.o_proj = GetStr(*a, "o_proj", ls.attention.o_proj);
          ls.attention.q_norm = GetStr(*a, "q_norm", ls.attention.q_norm);
          ls.attention.k_norm = GetStr(*a, "k_norm", ls.attention.k_norm);
          ls.attention.attn_gate = GetStr(*a, "attn_gate", ls.attention.attn_gate);
        }
        if (auto* c = lv.find("conv")) {
          ls.conv.width = GetU32(*c, "width", 4);
          ls.conv.in_proj = GetStr(*c, "in_proj", ls.conv.in_proj);
          ls.conv.out_proj = GetStr(*c, "out_proj", ls.conv.out_proj);
          ls.conv.conv_weight = GetStr(*c, "conv_weight", ls.conv.conv_weight);
        }
        if (auto* ff = lv.find("ffn")) {
          ls.ffn.kind = ParseFfn(GetStr(*ff, "kind", "swiglu"));
          ls.ffn.intermediate = GetU32(*ff, "intermediate", 0);
          ls.ffn.gate = GetStr(*ff, "gate", ls.ffn.gate);
          ls.ffn.up = GetStr(*ff, "up", ls.ffn.up);
          ls.ffn.down = GetStr(*ff, "down", ls.ffn.down);
        }
        s.layer_specs.push_back(ls);
      }
    }
    if (s.layers == 0) s.layers = static_cast<std::uint32_t>(s.layer_specs.size());
    return s;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return std::nullopt;
  }
}

}  // namespace gufo::npu
