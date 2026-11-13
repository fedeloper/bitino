// Runtime CPU capability detection. Optimized kernels declare the features
// they need; the kernel selector never picks a kernel whose features are absent.
#pragma once

#include <cstdint>
#include <string>

namespace bitino {

enum CpuFeature : uint32_t {
  kFeatNeon = 1u << 0,     // Advanced SIMD (baseline on arm64-v8a; AT_HWCAP NEON on 32-bit ARM)
  kFeatDotProd = 1u << 1,  // SDOT/UDOT (asimddp)
  kFeatI8mm = 1u << 2,     // SMMLA/USMMLA (i8mm)
  kFeatFp16 = 1u << 3,     // FP16 arithmetic (asimdhp)
  kFeatBf16 = 1u << 4,
  kFeatSve = 1u << 5,
  kFeatPopcnt = 1u << 6,   // x86 POPCNT (host builds)
  kFeatAvx2 = 1u << 7,     // x86 AVX2 (host builds)
  kFeatFma = 1u << 8,      // x86 FMA3; 32-bit ARM: VFPv4 (VFMA)
  kFeatBmi2 = 1u << 9,     // x86 BMI2
  kFeatAvxVnni = 1u << 10, // x86 AVX-VNNI (VEX-encoded VPDPBUSD, 256-bit; Alder Lake and later)
};

// MIDR part numbers (implementer Arm) used by kernel tuning choices
constexpr uint32_t kPartCortexA75 = 0xD0A;

struct CpuInfo {
  uint32_t features = 0;
  // Part number (MIDR_EL1[15:4], implementer Arm only) of the prime core: the CPU with the highest cpuinfo_max_freq,
  // ties to the highest part number (Linux / Android sysfs; 0 when unknown). Tuning only (instruction schedules
  // with the same results); BITINO_CPU_PART=<hex> overrides it.
  uint32_t prime_part = 0;
  std::string arch;  // "aarch64", "x86_64", ...
  bool has(uint32_t f) const { return (features & f) == f; }
  std::string describe() const;
};

// Detected once; the result is immutable. Can be masked for testing via the
// environment variable BITINO_DISABLE_FEATURES=dotprod,i8mm,...
const CpuInfo& cpu_info();

}  // namespace bitino
