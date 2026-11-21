#ifndef GUFO_SERVER_RESPONSES_LOGPROBS_HPP_
#define GUFO_SERVER_RESPONSES_LOGPROBS_HPP_

#include "src/cli/serve/logprobs.hpp"

namespace gufo::server {

inline std::optional<SamplingRequestError> ParseResponsesLogprobs(
    const json::Value& body, sampling::SamplingConfig* config) {
  config->logprobs.reset();
  bool enabled = false;
  if (const auto* include = body.find("include");
      include && !include->is_null()) {
    if (!include->is_array())
      return SamplingRequestError{"'include' must be an array of strings",
                                  "invalid_include"};
    for (const auto& value : include->items()) {
      if (!value.is_string())
        return SamplingRequestError{"'include' must be an array of strings",
                                    "invalid_include"};
      enabled |= value.str() == "message.output_text.logprobs";
    }
  }
  std::uint32_t count = 0;
  if (auto error = detail::ReadSamplingInteger(body, "top_logprobs", 0U, 20U,
                                               &count, true))
    return error;
  if (enabled)
    config->logprobs = count;
  return std::nullopt;
}

// Streaming text events omit the bytes field carried by output_text parts.
inline json::Value ResponsesStreamLogprobs(const json::Value& tokens) {
  auto result = json::Value::array();
  for (const auto& token : tokens.items()) {
    auto entry = json::Value::object();
    entry["token"] = *token.find("token");
    entry["logprob"] = *token.find("logprob");
    entry["top_logprobs"] = json::Value::array();
    for (const auto& top : token.find("top_logprobs")->items()) {
      auto alternative = json::Value::object();
      alternative["token"] = *top.find("token");
      alternative["logprob"] = *top.find("logprob");
      entry["top_logprobs"].push_back(std::move(alternative));
    }
    result.push_back(std::move(entry));
  }
  return result;
}

}  // namespace gufo::server
#endif  // GUFO_SERVER_RESPONSES_LOGPROBS_HPP_
