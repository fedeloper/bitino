// Shared output stage of the int8 GEMM micro-kernels. Everything here has
// internal linkage (static) so each micro-kernel TU keeps its own copy,
// compiled with that TU's -march flags (see int8_gemm.h).
//
// Bit-exact with int8_epilogue() (int8_quant.h):
//   vcvtq_f32_s32  == static_cast<float>(int32)   (round to nearest)
//   vmulq / vaddq  == separate fp32 roundings (no FMA)
//   activations reproduce apply_activation() including -0.0f and NaN:
//     relu  : v > 0 ? v : 0
//     relu6 : std::min(std::max(v, 0), 6) == (6 < m ? 6 : m), m = (v < 0 ? 0 : v)
#pragma once

#include <arm_neon.h>

#include <cstdint>

#include "kernels/int8_neon/int8_gemm.h"

namespace bitino {
namespace int8_neon {

static inline float32x4_t int8_act_f32x4(float32x4_t v, int act) {
  const float32x4_t zero = vdupq_n_f32(0.f);
  if (act == 1) return vbslq_f32(vcgtq_f32(v, zero), v, zero);
  if (act == 2) {
    const float32x4_t six = vdupq_n_f32(6.f);
    float32x4_t m = vbslq_f32(vcltq_f32(v, zero), zero, v);
    return vbslq_f32(vcltq_f32(six, m), six, m);
  }
  return v;
}

// Stores output row p, channels [o0, o0+8) from two int32x4 accumulators.
static inline void int8_store_row8(const GemmArgs& g, int64_t p, int64_t o0, int32x4_t lo, int32x4_t hi) {
  if (p >= g.rows) return;
  float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(lo), vld1q_f32(g.scale + o0));
  float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(hi), vld1q_f32(g.scale + o0 + 4));
  f0 = int8_act_f32x4(vaddq_f32(f0, vld1q_f32(g.bias + o0)), g.act);
  f1 = int8_act_f32x4(vaddq_f32(f1, vld1q_f32(g.bias + o0 + 4)), g.act);
  float* dst = g.y + p * g.cols + o0;
  if (o0 + 8 <= g.cols) {
    vst1q_f32(dst, f0);
    vst1q_f32(dst + 4, f1);
  } else {
    float tmp[8];
    vst1q_f32(tmp, f0);
    vst1q_f32(tmp + 4, f1);
    for (int64_t j = 0; j < g.cols - o0; ++j) dst[j] = tmp[j];
  }
}

}  // namespace int8_neon
}  // namespace bitino
