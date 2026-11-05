#include <cstring>
#include <exception>

#include "bitino/cpu_features.h"
#include "testing.h"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  std::printf("CPU: %s\n", bitino::cpu_info().describe().c_str());
  int passed = 0, failed = 0;
  for (const auto& t : bftest::registry()) {
    if (filter && !std::strstr(t.name, filter)) continue;
    try {
      t.fn();
      ++passed;
      std::printf("[ PASS ] %s\n", t.name);
    } catch (const bftest::Failure& f) {
      ++failed;
      std::printf("[ FAIL ] %s\n    %s\n", t.name, f.msg.c_str());
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[ FAIL ] %s\n    exception: %s\n", t.name, e.what());
    }
  }
  std::printf("%d passed, %d failed\n", passed, failed);
  return failed ? 1 : 0;
}
