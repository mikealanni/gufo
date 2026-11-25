#ifndef GUFO_MODELS_LING32_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LING32_CHAT_TEMPLATE_HPP_

#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::ling32 {

struct ChatMessage {
  std::string role;
  std::string content;
  std::string thought;
};

struct ChatTemplateOptions {
  bool enable_thinking{true};
};

[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const ChatTemplateOptions& options = {});

}  // namespace gufo::models::ling32

#endif  // GUFO_MODELS_LING32_CHAT_TEMPLATE_HPP_
