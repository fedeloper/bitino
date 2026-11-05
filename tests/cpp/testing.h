// Minimal test framework (no external dependency).
#pragma once

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

extern char** environ;

namespace bftest {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Failure {
  std::string msg;
};

// Re-executes this test binary running only the tests whose name contains
// `filter`, with extra environment variables (e.g. BITINO_DISABLE_FEATURES,
// which cpu_info() reads once per process); returns the child's exit status.
// Under user-mode emulation (qemu-aarch64) a child ARM binary cannot be
// exec'd directly: BITINO_TEST_LAUNCHER="<qemu>|<arg>|..." ('|'-separated,
// paths may contain spaces) prefixes the child command.
inline int run_self(const std::string& filter, const std::vector<std::string>& extra_env) {
  char exe[4096];
  ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (len <= 0) throw Failure{"readlink /proc/self/exe failed"};
  exe[len] = 0;
  std::vector<std::string> env_store;
  for (char** e = environ; *e; ++e) {
    std::string s = *e;
    bool overridden = false;
    for (const auto& x : extra_env) overridden |= s.compare(0, x.find('=') + 1, x, 0, x.find('=') + 1) == 0;
    if (!overridden) env_store.push_back(s);
  }
  for (const auto& x : extra_env) env_store.push_back(x);
  std::vector<char*> envp;
  for (auto& s : env_store) envp.push_back(&s[0]);
  envp.push_back(nullptr);
  std::vector<std::string> launch;
  if (const char* l = std::getenv("BITINO_TEST_LAUNCHER")) {
    std::string cur;
    for (const char* c = l;; ++c) {
      if (*c == '|' || *c == 0) {
        if (!cur.empty()) launch.push_back(cur);
        cur.clear();
        if (*c == 0) break;
      } else {
        cur += *c;
      }
    }
  }
  launch.push_back(exe);
  launch.push_back(filter);
  std::vector<char*> argv;
  for (auto& a : launch) argv.push_back(&a[0]);
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
  pid_t pid;
  int rc = posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) throw Failure{"posix_spawn failed: " + std::to_string(rc)};
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

}  // namespace bftest

#define BF_TEST(name)                                              \
  static void name();                                              \
  static ::bftest::Registrar name##_registrar(#name, name);        \
  static void name()

#define EXPECT_TRUE(cond)                                                              \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::ostringstream os_;                                                          \
      os_ << __FILE__ << ":" << __LINE__ << ": expected " #cond;                       \
      throw ::bftest::Failure{os_.str()};                                              \
    }                                                                                  \
  } while (0)

#define EXPECT_EQ(a, b)                                                                \
  do {                                                                                 \
    auto va_ = (a);                                                                    \
    auto vb_ = (b);                                                                    \
    if (!(va_ == vb_)) {                                                               \
      std::ostringstream os_;                                                          \
      os_ << __FILE__ << ":" << __LINE__ << ": " #a " == " #b " failed: " << va_      \
          << " vs " << vb_;                                                            \
      throw ::bftest::Failure{os_.str()};                                              \
    }                                                                                  \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                         \
  do {                                                                                 \
    double va_ = (a), vb_ = (b);                                                       \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                            \
      std::ostringstream os_;                                                          \
      os_ << __FILE__ << ":" << __LINE__ << ": |" #a " - " #b "| <= " << (tol)          \
          << " failed: " << va_ << " vs " << vb_;                                      \
      throw ::bftest::Failure{os_.str()};                                              \
    }                                                                                  \
  } while (0)

#define EXPECT_THROWS(stmt)                                                            \
  do {                                                                                 \
    bool threw_ = false;                                                               \
    try {                                                                              \
      stmt;                                                                            \
    } catch (const std::exception&) {                                                  \
      threw_ = true;                                                                   \
    }                                                                                  \
    if (!threw_) throw ::bftest::Failure{std::string(__FILE__) + ": expected throw: " #stmt}; \
  } while (0)
