#ifndef GUFO_MODELS_LING32_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LING32_CHAT_TEMPLATE_HPP_

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::models::ling32 {

struct ChatToolCall {
  std::string name;
  /// Argument name and value, the value exactly as it is written between the
  /// parameter tags.
  std::vector<std::pair<std::string, std::string>> arguments;
};

struct ChatMessage {
  std::string role;
  std::string content;
  std::string thought;
  std::vector<ChatToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json;
  /// The complete `{"type": "function", ...}` definition when the request
  /// carried one.
  std::string definition_json;
};

struct ChatTemplateOptions {
  bool enable_thinking{true};
  std::vector<ChatTool> tools;
  bool require_tool_call{false};
};

/// The model's own conversation layout (thinking instruction in the system
/// turn, `<think>` generation prompt, observation turns). Tools and tool calls
/// use the XML dialect the server parses and constrains:
/// `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...`.
[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const ChatTemplateOptions& options = {});

}  // namespace gufo::models::ling32

#endif  // GUFO_MODELS_LING32_CHAT_TEMPLATE_HPP_
