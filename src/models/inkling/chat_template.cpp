#include "src/models/inkling/chat_template.hpp"

#include <string_view>

namespace gufo::models::inkling {
namespace {

constexpr std::string_view kEnd = "<|end_message|>";
constexpr std::string_view kText = "<|content_text|>";
constexpr std::string_view kThinking = "<|content_thinking|>";
constexpr std::string_view kModelEnd = "<|content_model_end_sampling|>";
constexpr std::string_view kModelRole = "<|message_model|>";
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
  bool effort_emitted = false;
  for (const auto& message : messages) {
    if (!effort_emitted && message.role != "system" &&
        message.role != "developer") {
      effort();
      effort_emitted = true;
    }
    const auto role = RoleToken(message.role);
    const bool assistant = message.role == "assistant";
    if (assistant && !message.reasoning_content.empty()) {
      out += role;
      out += kThinking;
      out += message.reasoning_content;
      out += kEnd;
    }
    if (!(assistant && message.content.empty() &&
          !message.reasoning_content.empty())) {
      out += role;
      out += kText;
      out += message.content;
      out += kEnd;
    }
    if (assistant)
      out += kModelEnd;
  }
  if (!effort_emitted)
    effort();
  out += kModelRole;
  return out;
}

}  // namespace gufo::models::inkling
