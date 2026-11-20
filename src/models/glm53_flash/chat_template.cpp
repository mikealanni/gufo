#include "src/models/glm53_flash/chat_template.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <optional>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::glm53_flash {

namespace {

constexpr std::string_view kObservationTag = "<|observation|>";

// A JSON text as Python's json.dumps writes it; text that does not parse
// is kept as the client wrote it.
std::string PythonJson(std::string_view text) {
  try {
    return json::parse(text).tojson();
  } catch (const std::exception&) {
    return std::string(text);
  }
}

std::string_view Strip(std::string_view text) {
  const auto space = [](char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
  };
  while (!text.empty() && space(text.front()))
    text.remove_prefix(1);
  while (!text.empty() && space(text.back()))
    text.remove_suffix(1);
  return text;
}

// The template's tool_to_json: the function's fields in order, minus the two
// client-side switches, each value written the way Python's json.dumps does.
std::string ToolJson(const tokenization::ChatTool& tool) {
  std::string function_json;
  if (!tool.definition_json.empty()) {
    try {
      const auto definition = json::parse(tool.definition_json);
      const json::Value* function = definition.find("function");
      const json::Value& source = function != nullptr ? *function : definition;
      if (source.is_object()) {
        std::string out = "{";
        bool first = true;
        for (const auto& [key, value] : source.members()) {
          if (key == "defer_loading" || key == "strict")
            continue;
          if (!first)
            out += ", ";
          first = false;
          out += '"' + key + "\": " + value.tojson();
        }
        return out + "}";
      }
    } catch (const std::exception&) {
    }
  }
  std::string out = "{\"name\": " + json::Value(tool.name).tojson();
  if (!tool.description.empty()) {
    out += ", \"description\": " + json::Value(tool.description).tojson();
  }
  out += ", \"parameters\": " +
         PythonJson(tool.parameters_json.empty() ? "{}" : tool.parameters_json);
  return out + "}";
}

void AppendTools(std::string& out,
                 std::span<const tokenization::ChatTool> tools) {
  out += kSystemTag;
  out +=
      "\n# Tools\n\nYou may call one or more functions to assist with the user "
      "query.\n\nYou are provided with function signatures within "
      "<tools></tools> XML tags:\n<tools>\n";
  for (const auto& tool : tools) {
    out += '\n';
    out += ToolJson(tool);
    out += "\n\n";
  }
  out +=
      "\n</tools>\n\nFor each function call, output the function name and "
      "arguments within the following XML format:\n"
      "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>"
      "{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>"
      "{arg-value-2}</arg_value>...</tool_call>";
}

void AppendAssistant(std::string& out, const tokenization::ChatMessage& m) {
  out += kAssistantTag;
  std::string_view content = m.content;
  std::string_view reasoning;
  bool has_reasoning = false;
  if (!m.thought.empty()) {
    reasoning = m.thought;
    has_reasoning = true;
  } else if (const auto close = content.find("</think>");
             close != std::string_view::npos) {
    // content.split('</think>')[0].split('<think>')[-1], and the last piece.
    auto head = content.substr(0, close);
    if (const auto open = head.rfind("<think>"); open != std::string_view::npos)
      head.remove_prefix(open + 7);
    reasoning = head;
    has_reasoning = true;
    content = content.substr(content.rfind("</think>") + 8);
  }
  if (has_reasoning) {
    out += "<think>";
    out += reasoning;
    out += "</think>";
  } else {
    out += "<think></think>";
  }
  out += Strip(content);
  if (!m.tool_calls.empty()) {
    out += '\n';
    for (const auto& call : m.tool_calls) {
      out += "<tool_call>";
      out += call.name;
      for (const auto& argument : call.arguments) {
        out += "<arg_key>";
        out += argument.name;
        out += "</arg_key><arg_value>";
        out += argument.is_string ? argument.value : PythonJson(argument.value);
        out += "</arg_value>";
      }
      out += "</tool_call>";
    }
    out += '\n';
  }
}

void AppendToolBlock(std::string& out,
                     std::span<const tokenization::ChatMessage> messages,
                     std::size_t begin, std::size_t end) {
  out += kObservationTag;
  const tokenization::ChatMessage* assistant =
      begin > 0 &&
              messages[begin - 1].role == tokenization::ChatRole::kAssistant
          ? &messages[begin - 1]
          : nullptr;
  bool can_sort = assistant != nullptr && !assistant->tool_calls.empty();
  if (can_sort) {
    for (std::size_t i = begin; i < end && can_sort; ++i) {
      const auto& id = messages[i].tool_call_id;
      const auto same = [&](const auto& other) {
        return other.tool_call_id == id;
      };
      const auto known = [&](const auto& call) { return call.id == id; };
      can_sort = !id.empty() &&
                 std::count_if(messages.begin() + begin, messages.begin() + end,
                               same) == 1 &&
                 std::any_of(assistant->tool_calls.begin(),
                             assistant->tool_calls.end(), known);
    }
    const auto& calls = assistant->tool_calls;
    for (std::size_t i = 0; i < calls.size() && can_sort; ++i) {
      can_sort = !calls[i].id.empty();
      for (std::size_t j = i + 1; j < calls.size() && can_sort; ++j)
        can_sort = calls[j].id != calls[i].id;
    }
  }
  const auto response = [&](const tokenization::ChatMessage& m) {
    out += "<tool_response>";
    out += m.content;
    out += "</tool_response>";
  };
  if (can_sort) {
    for (const auto& call : assistant->tool_calls) {
      for (std::size_t i = begin; i < end; ++i) {
        if (messages[i].tool_call_id == call.id)
          response(messages[i]);
      }
    }
    return;
  }
  for (std::size_t i = begin; i < end; ++i)
    response(messages[i]);
}

}  // namespace

ChatRenderOptions RenderOptionsFor(const ReasoningOptions& reasoning) {
  ChatRenderOptions options;
  options.thinking = reasoning.enabled.value_or(true);
  if (reasoning.effort) {
    switch (*reasoning.effort) {
      case ReasoningEffort::kMinimal:
      case ReasoningEffort::kLow:
        options.reasoning_effort = "low";
        break;
      case ReasoningEffort::kMedium:
      case ReasoningEffort::kHigh:
        options.reasoning_effort = "high";
        break;
      case ReasoningEffort::kXHigh:
      case ReasoningEffort::kMax:
        options.reasoning_effort = "max";
        break;
    }
  }
  return options;
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  // SHA-256 of the accepted tokenizer.chat_template renderings. Both artifacts
  // frame prompts identically for this renderer: the RCO export only differs in
  // Jinja spelling (`.0.` instead of `[0]`, `~` instead of `+`) and in extra
  // `can_sort` early breaks in its tool-sorting loop, none of which change the
  // rendered text.
  constexpr std::string_view kTemplateSha256[] = {
      "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724",
      "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
  };
  const auto text = reader.GetMetadataString("tokenizer.chat_template");
  if (!text) {
    if (error_msg != nullptr)
      *error_msg = "GGUF has no chat template";
    return false;
  }
  const auto hash = crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(text->data()), text->size()});
  bool matched = false;
  for (const auto accepted : kTemplateSha256) {
    matched = matched || hash == accepted;
  }
  if (!matched) {
    if (error_msg != nullptr)
      *error_msg = "template sha256 " + hash + " is not the compiled one";
    return false;
  }
  return true;
}

std::string RenderChat(std::span<const tokenization::ChatMessage> messages,
                       std::span<const tokenization::ChatTool> tools,
                       const ChatRenderOptions& options) {
  std::string out = "[gMASK]<sop>";
  out += kSystemTag;
  out += "Reasoning Effort: ";
  out += options.reasoning_effort == "low"    ? "Low"
         : options.reasoning_effort == "high" ? "High"
                                              : "Max";
  if (!tools.empty()) {
    AppendTools(out, tools);
  }
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const auto& m = messages[i];
    switch (m.role) {
      case tokenization::ChatRole::kUser:
        out += kUserTag;
        out += m.content;
        break;
      case tokenization::ChatRole::kAssistant:
        AppendAssistant(out, m);
        break;
      case tokenization::ChatRole::kTool: {
        if (i > 0 && messages[i - 1].role == tokenization::ChatRole::kTool)
          break;
        std::size_t end = i;
        while (end < messages.size() &&
               messages[end].role == tokenization::ChatRole::kTool)
          ++end;
        AppendToolBlock(out, messages, i, end);
        break;
      }
      case tokenization::ChatRole::kSystem:
      case tokenization::ChatRole::kDeveloper:
        out += kSystemTag;
        out += m.content;
        out += m.framing_suffix;
        break;
    }
  }
  if (options.add_generation_prompt) {
    out += kAssistantTag;
    out += options.thinking ? "<think>" : "<think></think>";
  }
  return out;
}

std::string RenderPrompt(std::string_view system, std::string_view user,
                         std::string_view reasoning_effort) {
  std::string out = "[gMASK]<sop>";
  out += kSystemTag;
  out += "Reasoning Effort: ";
  if (reasoning_effort == "low") {
    out += "Low";
  } else if (reasoning_effort == "high") {
    out += "High";
  } else {
    out += "Max";
  }
  if (!system.empty()) {
    out += kSystemTag;
    out += system;
  }
  out += kUserTag;
  out += user;
  out += kAssistantTag;
  out += kThinkOpen;
  return out;
}

}  // namespace gufo::models::glm53_flash
