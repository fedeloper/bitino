// Named kernel-tier sets ("stages") shared by bitino_bench, the C API and the Python package.
#pragma once

#include <map>
#include <set>
#include <string>

namespace bitino {

// Benchmark stages map to kernel tier sets (see docs/benchmarking.md).
inline const std::map<std::string, std::set<std::string>>& stage_tiers() {
  static const std::map<std::string, std::set<std::string>> m = {
      {"reference", {"reference"}},
      {"packed_naive", {"reference", "packed_naive"}},
      {"binary_scalar", {"reference", "binary_scalar"}},
      // "neon" / "int8_neon" are the platform-SIMD stages: on x86-64 the same stage selects the
      // AVX2 tiers (binary_avx2, int8_avx2), so stage names stay comparable across targets.
      {"neon", {"reference", "binary_scalar", "neon", "binary_avx2"}},
      {"int8_reference", {"reference", "binary_scalar", "neon", "binary_avx2", "int8_reference"}},
      {"int8_neon", {"reference", "binary_scalar", "neon", "binary_avx2", "int8_reference", "int8_neon", "int8_avx2"}},
      {"optimized", {"reference", "binary_scalar", "neon", "binary_avx2", "float_opt"}},
      {"optimized_int8", {"reference", "binary_scalar", "neon", "binary_avx2", "float_opt", "int8_reference", "int8_neon", "int8_avx2"}},
      // Full-integer models (model_int8_full.bitino, docs/int8.md): I8* operators. The reference
      // stage runs the portable oracle kernels only; int8_full adds the NEON kernels (AArch64) and the
      // portable wrappers of the binary kernels (int8_full_opt: x86-64 hosts, fast evaluation runs).
      {"int8_full_reference", {"reference", "int8_full_reference"}},
      {"int8_full", {"reference", "binary_scalar", "neon", "binary_avx2", "float_opt", "int8_full_reference", "int8_full_opt",
                     "int8_full_neon"}},
      {"all", {}},
  };
  return m;
}

}  // namespace bitino
