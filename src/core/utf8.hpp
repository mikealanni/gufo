#ifndef GUFO_CORE_UTF8_HPP_
#define GUFO_CORE_UTF8_HPP_

#include <cstddef>
#include <string>
#include <string_view>

namespace gufo::core {

// Token pieces can split a Unicode scalar or contain malformed byte fallback
// output. JSON events must contain complete UTF-8; incomplete valid prefixes
// wait for the next piece, while malformed subsequences become U+FFFD.
class Utf8Decoder {
public:
  std::string Push(std::string_view bytes, bool final = false) {
    return Push(bytes, final, [](std::size_t, std::size_t) {});
  }

  // Report each decoded scalar's consumed input bytes and produced UTF-8
  // bytes. Consumers can preserve token provenance across byte fallback and
  // malformed sequences without implementing a second UTF-8 decoder.
  template<typename Observe>
  std::string Push(std::string_view bytes, bool final, Observe observe) {
    pending_.append(bytes);
    std::string output;
    output.reserve(pending_.size());
    std::size_t offset = 0;
    while (offset < pending_.size()) {
      const auto lead = static_cast<unsigned char>(pending_[offset]);
      if (lead < 0x80) {
        output.push_back(pending_[offset++]);
        observe(1, 1);
        continue;
      }
      const unsigned length = lead >= 0xC2 && lead <= 0xDF   ? 2
                              : lead >= 0xE0 && lead <= 0xEF ? 3
                              : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                             : 0;
      if (length == 0) {
        output.append("\xEF\xBF\xBD");
        ++offset;
        observe(1, 3);
        continue;
      }
      unsigned prefix = 1;
      while (prefix < length && offset + prefix < pending_.size()) {
        const auto byte = static_cast<unsigned char>(pending_[offset + prefix]);
        if (byte < 0x80 || byte > 0xBF ||
            (prefix == 1 &&
             ((lead == 0xE0 && byte < 0xA0) || (lead == 0xED && byte > 0x9F) ||
              (lead == 0xF0 && byte < 0x90) || (lead == 0xF4 && byte > 0x8F))))
          break;
        ++prefix;
      }
      if (prefix == length) {
        output.append(pending_, offset, length);
        offset += length;
        observe(length, length);
      } else if (offset + prefix == pending_.size() && !final) {
        break;
      } else {
        output.append("\xEF\xBF\xBD");
        offset += prefix;
        observe(prefix, 3);
      }
    }
    pending_.erase(0, offset);
    return output;
  }

private:
  std::string pending_;
};

}  // namespace gufo::core

#endif  // GUFO_CORE_UTF8_HPP_
