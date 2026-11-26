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

void AppendJsonString(std::string* output, std::string_view value) {
  output->push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output->append("\\\"");
        break;
      case '\\':
        output->append("\\\\");
        break;
      case '\n':
        output->append("\\n");
        break;
      case '\r':
        output->append("\\r");
        break;
      case '\t':
        output->append("\\t");
        break;
      default:
        output->push_back(static_cast<char>(character));
        break;
    }
  }
  output->push_back('"');
}

void AppendToolsPrompt(std::string* output, const std::vector<ChatTool>& tools,
                       bool require_tool_call) {
  output->append(
      "# Tools\n\nYou have access to the following functions:\n\n<tools>");
  for (const ChatTool& tool : tools) {
    output->push_back('\n');
    if (!tool.definition_json.empty()) {
      output->append(tool.definition_json);
      continue;
    }
    output->append("{\"type\": \"function\", \"function\": {\"name\": ");
    AppendJsonString(output, tool.name);
    output->append(", \"description\": ");
    AppendJsonString(output, tool.description);
    output->append(", \"parameters\": ");
    output->append(tool.parameters_json.empty() ? "{}" : tool.parameters_json);
    output->append("}}");
  }
  output->append(
      "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
      "following format with NO suffix:\n\n<tool_call>\n"
      "<function=example_function_name>\n"
      "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
      "<parameter=example_parameter_2>\nThis is the value for the second "
      "parameter\nthat can span\nmultiple lines\n</parameter>\n"
      "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls "
      "MUST follow the specified format: an inner <function=...></function> "
      "block must be nested within <tool_call></tool_call> XML tags\n- "
      "Required parameters MUST be specified\n- You may provide optional "
      "reasoning for your function call in natural language BEFORE the "
      "function call, but NOT after\n- If there is no function call available, "
      "answer the question like normal with your current knowledge and do not "
      "tell the user about function calls\n</IMPORTANT>");
  if (require_tool_call) {
    output->append(
        "\n\nYou must call at least one available function. Do not answer "
        "the user directly.");
  }
  output->push_back('\n');
}

void AppendToolCalls(std::string* output,
                     const std::vector<ChatToolCall>& calls) {
  bool first = true;
  for (const ChatToolCall& call : calls) {
    if (!first)
      output->push_back('\n');
    first = false;
    output->append("<tool_call>\n<function=");
    output->append(call.name);
    output->append(">\n");
    for (const auto& [name, value] : call.arguments) {
      output->append("<parameter=");
      output->append(name);
      output->append(">\n");
      output->append(value);
      output->append("\n</parameter>\n");
    }
    output->append("</function>\n</tool_call>");
  }
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
    if ((!options.tools.empty() || !has_instruction) && !system.empty() &&
        system.back() != '\n')
      prompt.push_back('\n');
    first = 1;
  }
  if (!options.tools.empty())
    AppendToolsPrompt(&prompt, options.tools, options.require_tool_call);
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
      const std::string content = StripNewlines(message.content, true, false);
      prompt.append(content);
      if (!message.tool_calls.empty()) {
        if (!content.empty())
          prompt.push_back('\n');
        AppendToolCalls(&prompt, message.tool_calls);
      }
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
