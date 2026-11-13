#include "bitino/cpu_features.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#if (defined(__aarch64__) || defined(__arm__)) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace bitino {
namespace {

#if (defined(__aarch64__) || defined(__arm__)) && defined(__linux__)
// Prime core part number from sysfs (see CpuInfo::prime_part).
uint32_t detect_prime_part() {
  uint32_t best_part = 0;
  long best_freq = -1;
  for (int c = 0; c < 64; ++c) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
    unsigned long long midr = 0;
    long freq = 0;
    FILE* f = std::fopen((base + "/regs/identification/midr_el1").c_str(), "r");
    if (!f) continue;
    const bool ok = std::fscanf(f, "%llx", &midr) == 1;
    std::fclose(f);
    if (!ok || ((midr >> 24) & 0xff) != 0x41) continue;  // Arm implementer only
    if ((f = std::fopen((base + "/cpufreq/cpuinfo_max_freq").c_str(), "r"))) {
      if (std::fscanf(f, "%ld", &freq) != 1) freq = 0;
      std::fclose(f);
    }
    const uint32_t part = static_cast<uint32_t>((midr >> 4) & 0xfff);
    if (freq > best_freq || (freq == best_freq && part > best_part)) {
      best_freq = freq;
      best_part = part;
    }
  }
  return best_part;
}
#endif

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
#elif defined(__APPLE__)
  // Apple silicon: dotprod on every M-series / A13+ core, i8mm from M2 / A15 (hw.optional.arm.FEAT_*).
  auto sysctl_flag = [](const char* name) {
    int v = 0;
    size_t len = sizeof(v);
    return sysctlbyname(name, &v, &len, nullptr, 0) == 0 && v != 0;
  };
  if (sysctl_flag("hw.optional.arm.FEAT_DotProd")) info.features |= kFeatDotProd;
  if (sysctl_flag("hw.optional.arm.FEAT_I8MM")) info.features |= kFeatI8mm;
  if (sysctl_flag("hw.optional.arm.FEAT_FP16")) info.features |= kFeatFp16;
  if (sysctl_flag("hw.optional.arm.FEAT_BF16")) info.features |= kFeatBf16;
#endif
#elif defined(__arm__)
  // 32-bit ARM (armeabi-v7a; also AArch32 on ARMv8 cores): the NEON kernels' AArch32 builds (kernels/neon_a32.h)
  // need Advanced SIMD, which the kernel reports in AT_HWCAP.
  info.arch = "arm";
#if defined(__linux__) && defined(HWCAP_NEON)
  const unsigned long hw = getauxval(AT_HWCAP);
  if (hw & HWCAP_NEON) info.features |= kFeatNeon;
#ifdef HWCAP_VFPv4
  if (hw & HWCAP_VFPv4) info.features |= kFeatFma;  // VFMA (float_opt fma_ on 32-bit ARM)
#endif
#elif defined(__ARM_NEON)
  info.features |= kFeatNeon;
#endif
#if defined(__ARM_FEATURE_FMA)
  info.features |= kFeatFma;
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

#if (defined(__aarch64__) || defined(__arm__)) && defined(__linux__)
  info.prime_part = detect_prime_part();
#endif
  if (const char* part = std::getenv("BITINO_CPU_PART")) info.prime_part = static_cast<uint32_t>(std::strtoul(part, nullptr, 16));
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
