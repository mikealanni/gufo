#include "src/models/inkling/chat_template.hpp"

#include <algorithm>
#include <string_view>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::models::inkling {
namespace {

constexpr std::string_view kEnd = "<|end_message|>";
constexpr std::string_view kText = "<|content_text|>";
constexpr std::string_view kThinking = "<|content_thinking|>";
constexpr std::string_view kModelEnd = "<|content_model_end_sampling|>";
constexpr std::string_view kModelRole = "<|message_model|>";
constexpr std::string_view kInvoke = "<|content_invoke_tool_json|>";
constexpr std::string_view kThinkingEffort = "0.9";
constexpr std::string_view kNoEffort = "0";

std::string_view RoleToken(std::string_view role) {
  if (role == "assistant")
    return kModelRole;
  if (role == "user")
    return "<|message_user|>";
  if (role == "tool")
    return "<|message_tool|>";
  return "<|message_system|>";
}

json::Value SortedKeys(const json::Value& value) {
  if (value.is_object()) {
    auto object = json::Value::object();
    std::vector<const json::Value::Member*> members;
    for (const auto& member : value.members())
      members.push_back(&member);
    std::sort(members.begin(), members.end(),
              [](const auto* a, const auto* b) { return a->first < b->first; });
    for (const auto* member : members)
      object[member->first] = SortedKeys(member->second);
    return object;
  }
  if (value.is_array()) {
    auto array = json::Value::array();
    for (const auto& item : value.items())
      array.push_back(SortedKeys(item));
    return array;
  }
  return value;
}

std::string ToolDeclarations(const std::vector<ChatTool>& tools) {
  auto specs = json::Value::array();
  for (const auto& tool : tools) {
    auto spec = json::Value::object();
    spec["description"] = tool.description;
    spec["name"] = tool.name;
    try {
      spec["parameters"] = json::parse(tool.parameters_json);
    } catch (const std::exception&) {
      spec["parameters"] = json::Value::object();
    }
    spec["type"] = "function";
    specs.push_back(SortedKeys(spec));
  }
  return specs.dump();
}

std::string InvokeJson(const ChatToolCall& call) {
  json::Value args = json::Value::object();
  try {
    args = json::parse(call.arguments_json);
  } catch (const std::exception&) {
  }
  std::string out =
      "{\"name\":" + json::Value(call.name).dump() + ",\"args\":{";
  if (args.is_object()) {
    std::vector<const json::Value::Member*> members;
    for (const auto& member : args.members())
      members.push_back(&member);
    std::sort(members.begin(), members.end(),
              [](const auto* a, const auto* b) { return a->first < b->first; });
    bool first = true;
    for (const auto* member : members) {
      if (!first)
        out += ",";
      first = false;
      out += json::Value(member->first).dump() + ":" + member->second.dump();
    }
  }
  return out + "}}";
}

}  // namespace

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const ChatTemplateOptions& options) {
  std::string out;
  const auto effort = [&] {
    out += "<|message_system|>";
    out += kText;
    out += "Thinking effort level: ";
    out += options.enable_thinking ? kThinkingEffort : kNoEffort;
    out += kEnd;
  };
  if (!options.tools.empty()) {
    out += "<|message_system|>tool_declare<|content_xml|>";
    out += ToolDeclarations(options.tools);
    out += kEnd;
  }
  const auto tool_name = [&](const ChatMessage& message) -> std::string {
    if (!message.name.empty())
      return message.name;
    if (message.tool_call_id.empty())
      return {};
    for (const auto& previous : messages) {
      for (const auto& call : previous.tool_calls) {
        if (call.id == message.tool_call_id)
          return call.name;
      }
    }
    return {};
  };
  bool effort_emitted = false;
  for (const auto& message : messages) {
    if (!effort_emitted && message.role != "system" &&
        message.role != "developer") {
      effort();
      effort_emitted = true;
    }
    std::string role(RoleToken(message.role));
    const bool assistant = message.role == "assistant";
    if (message.role == "tool")
      role += tool_name(message);
    const bool calls = assistant && !message.tool_calls.empty();
    if (assistant && !message.reasoning_content.empty()) {
      out += role;
      out += kThinking;
      out += message.reasoning_content;
      out += kEnd;
    }
    if (!message.content.empty() ||
        !(assistant && (calls || !message.reasoning_content.empty()))) {
      out += role;
      out += kText;
      out += message.content;
      out += kEnd;
    }
    if (calls) {
      for (const auto& call : message.tool_calls) {
        out += kModelRole;
        out += call.name;
        out += kInvoke;
        out += InvokeJson(call);
        out += kEnd;
      }
    }
    if (assistant)
      out += kModelEnd;
  }
  if (!effort_emitted)
    effort();
  out += kModelRole;
  if (!options.enable_thinking)
    out += kText;
  return out;
}

}  // namespace gufo::models::inkling
