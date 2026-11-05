// Small utilities: SHA-256, monotonic timer, file IO.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bitino {

std::string sha256_hex(const void* data, size_t len);

std::vector<uint8_t> read_file(const std::string& path);  // throws on failure
void write_file(const std::string& path, const std::string& contents);

inline double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

}  // namespace bitino
