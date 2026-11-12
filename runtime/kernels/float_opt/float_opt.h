// float_opt tier: optimized FP32 kernels (NEON on AArch64, portable C++
// elsewhere, see float_simd.h; plus AVX2+FMA kernels on x86-64, see
// float_opt_avx2.h). Each translation unit registers its kernels.
#pragma once

#include "bitino/cpu_features.h"
#include "bitino/op.h"

namespace bitino {

// The NEON build only uses baseline AArch64 ASIMD (32-bit ARM: NEON, see
// float_simd.h); the portable build needs nothing.
#if defined(__aarch64__)
constexpr uint32_t kFloatOptFeatures = kFeatNeon;
#elif defined(__ARM_NEON) && defined(__arm__)
constexpr uint32_t kFloatOptFeatures = kFeatNeon;
#else
constexpr uint32_t kFloatOptFeatures = 0;
#endif

#if defined(__x86_64__)
// AVX2 + FMA kernels (float_opt_avx2.h), same tier, registered next to the
// portable ones with higher priority: the selector uses them when the CPU
// reports both features (BITINO_DISABLE_FEATURES=avx2 or fma masks them).
constexpr uint32_t kFloatOptAvx2Features = kFeatAvx2 | kFeatFma;
constexpr int kFloatOptAvx2Priority = 50;  // portable: 40, depthwise 45 (AVX2 depthwise: 55)
#endif

void register_float_opt_gemm_kernels(OpRegistry& reg);         // conv_gemm.cc: Conv2D (any groups), Dense
void register_float_opt_depthwise_kernels(OpRegistry& reg);    // depthwise.cc: Conv2D (depthwise)
void register_float_opt_elementwise_kernels(OpRegistry& reg);  // elementwise.cc: Add, ChannelAffine, Activation, pools
void register_float_opt_transformer_kernels(OpRegistry& reg);  // transformer_opt.cc: transformer / BHViT ops, Concat, Gather

namespace fopt {
struct GemmBlockArgs;
}
// One GEMM block on the NEON / portable backend (panel width 8; conv_gemm.cc),
// for kernels that pack B at run time (MatMul).
void float_opt_gemm_block_simd4(const fopt::GemmBlockArgs& g);

// Test hook (transformer_opt.cc): GELU on the 4-lane backend, either the kernel's
// path (`fast`: on AArch64 the compacting / branch-skipping gelu_neon) or the plain
// per-vector vgelu (leaf_unary<Simd4>). The two must be bit-identical.
void float_opt_gelu_simd4(const float* x, float* y, int64_t n, bool fast);

}  // namespace bitino
