#ifndef GUFO_MODELS_INKLING_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_INKLING_CHAT_TEMPLATE_HPP_

#include <string>
#include <vector>

namespace gufo::models::inkling {

struct ChatMessage {
  std::string role;  // system | developer | user | assistant | tool
  std::string content;
  std::string reasoning_content;
};

struct ChatTemplateOptions {
  bool enable_thinking{true};
};

// Mirrors the GGUF Jinja template without tools, then opens the model turn.
[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const ChatTemplateOptions& options = {});

}  // namespace gufo::models::inkling

#endif  // GUFO_MODELS_INKLING_CHAT_TEMPLATE_HPP_
