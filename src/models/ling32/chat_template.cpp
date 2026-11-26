#include "src/models/ling32/chat_template.hpp"

#include <cstddef>

namespace gufo::models::ling32 {
namespace {

constexpr std::string_view kRoleOpen = "<role>";
constexpr std::string_view kRoleClose = "</role>";
constexpr std::string_view kRoleEnd = "<|role_end|>";
constexpr std::string_view kThinkingOn = "detailed thinking on";
constexpr std::string_view kThinkingOff = "detailed thinking off";

void AppendRole(std::string* prompt, std::string_view role) {
  prompt->append(kRoleOpen);
  prompt->append(role);
  prompt->append(kRoleClose);
}

std::string StripNewlines(const std::string& text, bool leading,
                          bool trailing) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  if (leading) {
    while (begin < end && text[begin] == '\n')
      ++begin;
  }
  if (trailing) {
    while (end > begin && text[end - 1] == '\n')
      --end;
  }
  return text.substr(begin, end - begin);
}

}  // namespace

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const ChatTemplateOptions& options) {
  std::string prompt;
  const std::string_view thinking_instruction =
      options.enable_thinking ? kThinkingOn : kThinkingOff;

  // The official template always opens with one SYSTEM turn: the first system
  // message, then the thinking instruction unless that message carries one.
  AppendRole(&prompt, "SYSTEM");
  std::size_t first = 0;
  bool has_instruction = false;
  if (!messages.empty() && messages.front().role == "system") {
    const std::string& system = messages.front().content;
    has_instruction = system.find(kThinkingOn) != std::string::npos ||
                      system.find(kThinkingOff) != std::string::npos;
    prompt.append(system);
    if (!has_instruction && !system.empty() && system.back() != '\n')
      prompt.push_back('\n');
    first = 1;
  }
  if (!has_instruction)
    prompt.append(thinking_instruction);
  prompt.append(kRoleEnd);

  for (std::size_t index = first; index < messages.size(); ++index) {
    const ChatMessage& message = messages[index];
    if (message.role == "user") {
      AppendRole(&prompt, "HUMAN");
      prompt.append(message.content);
      prompt.append(kRoleEnd);
    } else if (message.role == "system") {
      AppendRole(&prompt, "SYSTEM");
      prompt.append(message.content);
      prompt.append(kRoleEnd);
    } else if (message.role == "assistant") {
      AppendRole(&prompt, "ASSISTANT");
      const std::string reasoning = StripNewlines(message.thought, true, true);
      if (!reasoning.empty()) {
        prompt.append("\n<think>");
        prompt.append(reasoning);
        prompt.append("</think>");
      } else {
        prompt.append("\n<think></think>");
      }
      prompt.append(StripNewlines(message.content, true, false));
      prompt.append(kRoleEnd);
    } else if (message.role == "tool") {
      if (index == first || messages[index - 1].role != "tool")
        AppendRole(&prompt, "OBSERVATION");
      prompt.append("\n<tool_response>\n");
      prompt.append(message.content);
      prompt.append("\n</tool_response>");
      if (index + 1 == messages.size() || messages[index + 1].role != "tool")
        prompt.append(kRoleEnd);
    } else {
      AppendRole(&prompt, message.role);
      prompt.append(message.content);
      prompt.append(kRoleEnd);
    }
  }

  AppendRole(&prompt, "ASSISTANT");
  prompt.append(options.enable_thinking ? "\n<think>" : "\n<think></think>");
  return prompt;
}

}  // namespace gufo::models::ling32
