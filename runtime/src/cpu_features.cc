#include "bitino/cpu_features.h"

#include <cstdlib>
#include <string>

#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace bitino {
namespace {

CpuInfo detect() {
  CpuInfo info;
#if defined(__aarch64__)
  info.arch = "aarch64";
  info.features |= kFeatNeon;
#if defined(__linux__)
  unsigned long hw = getauxval(AT_HWCAP);
  unsigned long hw2 = getauxval(AT_HWCAP2);
#ifdef HWCAP_ASIMDDP
  if (hw & HWCAP_ASIMDDP) info.features |= kFeatDotProd;
#endif
#ifdef HWCAP_ASIMDHP
  if (hw & HWCAP_ASIMDHP) info.features |= kFeatFp16;
#endif
#ifdef HWCAP_SVE
  if (hw & HWCAP_SVE) info.features |= kFeatSve;
#endif
#ifdef HWCAP2_I8MM
  if (hw2 & HWCAP2_I8MM) info.features |= kFeatI8mm;
#endif
#ifdef HWCAP2_BF16
  if (hw2 & HWCAP2_BF16) info.features |= kFeatBf16;
#endif
  (void)hw2;
#endif
#elif defined(__x86_64__)
  info.arch = "x86_64";
  __builtin_cpu_init();
  if (__builtin_cpu_supports("popcnt")) info.features |= kFeatPopcnt;
  if (__builtin_cpu_supports("avx2")) info.features |= kFeatAvx2;
  if (__builtin_cpu_supports("fma")) info.features |= kFeatFma;
  if (__builtin_cpu_supports("bmi2")) info.features |= kFeatBmi2;
  if (__builtin_cpu_supports("avxvnni")) info.features |= kFeatAvxVnni;
#else
  info.arch = "unknown";
#endif

  if (const char* dis = std::getenv("BITINO_DISABLE_FEATURES")) {
    std::string s = std::string(",") + dis + ",";
    auto off = [&](const char* name, uint32_t f) {
      if (s.find(std::string(",") + name + ",") != std::string::npos) info.features &= ~f;
    };
    off("neon", kFeatNeon);
    off("dotprod", kFeatDotProd);
    off("i8mm", kFeatI8mm);
    off("fp16", kFeatFp16);
    off("bf16", kFeatBf16);
    off("sve", kFeatSve);
    off("popcnt", kFeatPopcnt);
    off("avx2", kFeatAvx2);
    off("fma", kFeatFma);
    off("bmi2", kFeatBmi2);
    off("avxvnni", kFeatAvxVnni);
  }
  return info;
}

}  // namespace

std::string CpuInfo::describe() const {
  std::string s = arch + ":";
  auto add = [&](uint32_t f, const char* n) {
    if (features & f) s += std::string(" ") + n;
  };
  add(kFeatNeon, "neon");
  add(kFeatDotProd, "dotprod");
  add(kFeatI8mm, "i8mm");
  add(kFeatFp16, "fp16");
  add(kFeatBf16, "bf16");
  add(kFeatSve, "sve");
  add(kFeatPopcnt, "popcnt");
  add(kFeatAvx2, "avx2");
  add(kFeatFma, "fma");
  add(kFeatBmi2, "bmi2");
  add(kFeatAvxVnni, "avxvnni");
  return s;
}

const CpuInfo& cpu_info() {
  static const CpuInfo info = detect();
  return info;
}

}  // namespace bitino
