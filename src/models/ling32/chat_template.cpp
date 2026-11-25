#include "src/models/ling32/chat_template.hpp"

#include <utility>

namespace gufo::models::ling32 {
namespace {

constexpr std::string_view kRoleOpen = "<role>";
constexpr std::string_view kRoleClose = "</role>";
constexpr std::string_view kRoleEnd = "<|role_end|>";
constexpr std::string_view kSystemRole = "SYSTEM";
constexpr std::string_view kHumanRole = "HUMAN";
constexpr std::string_view kAssistantRole = "ASSISTANT";
constexpr std::string_view kObservationRole = "OBSERVATION";

void AppendTurn(std::string* prompt, std::string_view role,
                const std::string& content, const std::string& thought,
                bool enable_thinking) {
  prompt->append(kRoleOpen);
  prompt->append(role);
  prompt->append(kRoleClose);
  if (role == kAssistantRole) {
    if (!thought.empty()) {
      prompt->append("\n<think>\n");
      prompt->append(thought);
      prompt->append("</think>\n");
    } else if (enable_thinking) {
      prompt->append("\n<think>\n");
    } else {
      prompt->append("\n<think></think>\n");
    }
  }
  prompt->append(content);
  prompt->append(kRoleEnd);
}

}  // namespace

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const ChatTemplateOptions& options) {
  std::string prompt;
  for (const auto& message : messages) {
    if (message.role == "user") {
      AppendTurn(&prompt, kHumanRole, message.content, {},
                 options.enable_thinking);
    } else if (message.role == "system" || message.role == "developer") {
      AppendTurn(&prompt, kSystemRole, message.content, {},
                 options.enable_thinking);
    } else if (message.role == "assistant") {
      AppendTurn(&prompt, kAssistantRole, message.content, message.thought,
                 options.enable_thinking);
    } else if (message.role == "tool") {
      AppendTurn(&prompt, kObservationRole, message.content, {},
                 options.enable_thinking);
    } else {
      AppendTurn(&prompt, message.role, message.content, {},
                 options.enable_thinking);
    }
  }
  prompt.append(kRoleOpen);
  prompt.append(kAssistantRole);
  prompt.append(kRoleClose);
  return prompt;
}

}  // namespace gufo::models::ling32
