#include "src/models/laguna/chat_template.hpp"

namespace gufo::models::laguna {
namespace {

std::string_view TrimRight(std::string_view s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' ||
                        s.back() == '\t' || s.back() == '\r')) {
    s.remove_suffix(1);
  }
  return s;
}

bool IsBlank(std::string_view s) {
  return TrimRight(s).empty() ||
         s.find_first_not_of(" \n\t\r") == std::string_view::npos;
}

}  // namespace

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const ChatTemplateOptions& options) {
  std::string out(kBosText);
  std::size_t first = 0;
  std::string_view system = kDefaultSystem;
  if (!messages.empty() && messages[0].role == "system") {
    system = messages[0].content;
    first = 1;
  }
  const bool has_sys = !IsBlank(system);
  if (has_sys || options.enable_thinking) {
    out += "<system>";
    if (has_sys) {
      out += TrimRight(system);
    }
    out += "</system>\n";
  }
  for (std::size_t i = first; i < messages.size(); ++i) {
    const auto& m = messages[i];
    if (m.role == "user") {
      out += "<user>" + m.content + "</user>\n";
    } else if (m.role == "assistant") {
      out += kAssistantOpen;
      if (options.enable_thinking) {
        out += kThinkOpen;
        out += m.reasoning_content;
        out += kThinkClose;
      } else {
        out += kThinkClose;
      }
      out += m.content;
      out += kAssistantClose;
      out += "\n";
    } else if (m.role == "tool") {
      out += "<tool_response>" + m.content + "</tool_response>\n";
    } else if (m.role == "system") {
      out += "<system>" + m.content + "</system>\n";
    }
  }
  out += kAssistantOpen;
  out += options.enable_thinking ? kThinkOpen : kThinkClose;
  return out;
}

std::string RenderPrompt(std::string_view system, std::string_view user,
                         const ChatTemplateOptions& options) {
  std::vector<ChatMessage> messages;
  if (!system.empty()) {
    messages.push_back({"system", std::string(system), ""});
  }
  messages.push_back({"user", std::string(user), ""});
  return RenderPrompt(messages, options);
}

}  // namespace gufo::models::laguna
