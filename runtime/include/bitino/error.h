// Error handling: load/plan-time failures throw bitino::Error with an
// actionable message. Kernel hot paths never throw.
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace bitino {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

namespace detail {
inline void append(std::ostringstream&) {}
template <typename T, typename... Rest>
void append(std::ostringstream& os, const T& v, const Rest&... rest) {
  os << v;
  append(os, rest...);
}
}  // namespace detail

template <typename... Args>
[[noreturn]] void fail(const Args&... args) {
  std::ostringstream os;
  detail::append(os, args...);
  throw Error(os.str());
}

#define BF_CHECK(cond, ...)                                                    \
  do {                                                                         \
    if (!(cond)) ::bitino::fail("check failed: " #cond " — ", __VA_ARGS__); \
  } while (0)

}  // namespace bitino
