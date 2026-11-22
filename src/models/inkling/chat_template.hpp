#ifndef GUFO_MODELS_INKLING_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_INKLING_CHAT_TEMPLATE_HPP_

#include <string>
#include <vector>

namespace gufo::models::inkling {

struct ChatToolCall {
  std::string id;
  std::string name;
  std::string arguments_json{"{}"};  // a JSON object
};

struct ChatMessage {
  std::string role;  // system | developer | user | assistant | tool
  std::string content;
  std::string reasoning_content;
  std::string name;          // tool result: the tool's name
  std::string tool_call_id;  // tool result: resolved to the call's name
  std::vector<ChatToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
};

struct ChatTemplateOptions {
  bool enable_thinking{true};
  std::vector<ChatTool> tools;
};

// Mirrors the GGUF Jinja template (tool declarations, reasoning, tool calls and
// tool results), then opens the model turn.
[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const ChatTemplateOptions& options = {});

}  // namespace gufo::models::inkling

#endif  // GUFO_MODELS_INKLING_CHAT_TEMPLATE_HPP_
