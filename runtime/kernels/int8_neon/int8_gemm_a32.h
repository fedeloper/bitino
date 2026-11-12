// int8 GEMM micro-kernel for 32-bit ARM (armeabi-v7a / AArch32 on ARMv8 cores, BF_NEON_A32), shared by the
// int8_full GEMMs (int8_full_neon.cc: I8Conv2D / I8Dense / I8Conv2DN / I8MatMul) and the dynamic-range int8_neon
// kernels (int8_neon_a32.cc: Conv2D / Dense of stage optimized_int8). AArch32 has no dot-product instructions; the
// products are VMULL.S8 (8 int8 x int8 -> int16), VMLAL.S8 adds the next 8 (two products per int16 lane), and
// VPADAL.S16 accumulates pairs of int16 lanes into int32 lanes. Measured on the Redmi 9A's Cortex-A53 (2.0 GHz,
// L1-resident operands): 5.0-5.6 GMAC/s for the 4x2 tile below, vs 3.7-4.1 for the same tile in intrinsics (the
// compiler spills accumulators) and 5.5 / 5.3 for VMLAL.S16-by-lane 6x8 / 4x8 tiles on int16 operands
// (the development log (not published), "AArch32").
//
// Exactness: a lane of the int16 product pair holds a0*w0 + a1*w1. It cannot overflow when every weight is in
// [-127, 127] (|a| <= 128: |a0*w0 + a1*w1| <= 2 * 128 * 127 = 32512); callers check that (weights_fit_mull) and
// use another kernel otherwise. The int32 accumulation is exact while every partial sum fits int32, i.e.
// K * 128 * 127 < 2^31 (callers' K bounds); the order of the integer additions does not matter.
//
// Layouts (K zero-padded to kp, a multiple of 16):
//   A quad   [kp/16][4 rows][16 bytes]   (pack_quad)
//   W pair   [kp/16][2 cols][16 bytes]   (pack_pair)
#pragma once

#include "kernels/neon_a32.h"

#if BF_NEON_A32
#include <arm_neon.h>

#include <cstdint>
#include <cstring>

namespace bitino {
namespace a32gemm {

// y[r * ldy + c] = sum_k a[r][k] * w[c][k] for the 4 rows r of the quad and the 2 columns c of the pair;
// n16 = kp / 16 >= 1, ldy in int32 elements.
static inline void tile4x2(const int8_t* a, const int8_t* w, int32_t n16, int32_t* y, int32_t ldy) {
  const int32_t ldyb = ldy * 4;
  asm volatile(
      "vmov.i32 q8, #0\n\t"
      "vmov.i32 q9, #0\n\t"
      "vmov.i32 q10, #0\n\t"
      "vmov.i32 q11, #0\n\t"
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "vmov.i32 q14, #0\n\t"
      "vmov.i32 q15, #0\n\t"
      "1:\n\t"
      "vld1.8 {d0-d3}, [%[w]]!\n\t"  // columns 0 (q0), 1 (q1)
      "vld1.8 {d4-d7}, [%[a]]!\n\t"  // rows 0 (q2), 1 (q3)
      "vmull.s8 q4, d4, d0\n\t"
      "vmull.s8 q5, d4, d2\n\t"
      "vmull.s8 q6, d6, d0\n\t"
      "vmull.s8 q7, d6, d2\n\t"
      "vmlal.s8 q4, d5, d1\n\t"
      "vmlal.s8 q5, d5, d3\n\t"
      "vmlal.s8 q6, d7, d1\n\t"
      "vmlal.s8 q7, d7, d3\n\t"
      "vld1.8 {d4-d7}, [%[a]]!\n\t"  // rows 2, 3
      "vpadal.s16 q8, q4\n\t"
      "vpadal.s16 q9, q5\n\t"
      "vpadal.s16 q10, q6\n\t"
      "vpadal.s16 q11, q7\n\t"
      "vmull.s8 q4, d4, d0\n\t"
      "vmull.s8 q5, d4, d2\n\t"
      "vmull.s8 q6, d6, d0\n\t"
      "vmull.s8 q7, d6, d2\n\t"
      "vmlal.s8 q4, d5, d1\n\t"
      "vmlal.s8 q5, d5, d3\n\t"
      "vmlal.s8 q6, d7, d1\n\t"
      "vmlal.s8 q7, d7, d3\n\t"
      "subs %[n], %[n], #1\n\t"
      "vpadal.s16 q12, q4\n\t"
      "vpadal.s16 q13, q5\n\t"
      "vpadal.s16 q14, q6\n\t"
      "vpadal.s16 q15, q7\n\t"
      "bne 1b\n\t"
      // row r: [sum(acc r,c0), sum(acc r,c1)]
      "vpadd.i32 d0, d16, d17\n\t"
      "vpadd.i32 d1, d18, d19\n\t"
      "vpadd.i32 d0, d0, d1\n\t"
      "vpadd.i32 d2, d20, d21\n\t"
      "vpadd.i32 d3, d22, d23\n\t"
      "vpadd.i32 d2, d2, d3\n\t"
      "vpadd.i32 d4, d24, d25\n\t"
      "vpadd.i32 d5, d26, d27\n\t"
      "vpadd.i32 d4, d4, d5\n\t"
      "vpadd.i32 d6, d28, d29\n\t"
      "vpadd.i32 d7, d30, d31\n\t"
      "vpadd.i32 d6, d6, d7\n\t"
      "vst1.32 {d0}, [%[y]], %[ldyb]\n\t"
      "vst1.32 {d2}, [%[y]], %[ldyb]\n\t"
      "vst1.32 {d4}, [%[y]], %[ldyb]\n\t"
      "vst1.32 {d6}, [%[y]]\n\t"
      : [a] "+r"(a), [w] "+r"(w), [n] "+r"(n16), [y] "+r"(y)
      : [ldyb] "r"(ldyb)
      : "memory", "cc", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
        "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31");
}

// The same sums for R < 4 rows of a quad (the rows >= R are not read or written); intrinsics (tails only).
template <int R>
static inline void tile_rx2(const int8_t* a, const int8_t* w, int32_t n16, int32_t* y, int32_t ldy) {
  int32x4_t c[R][2];
  for (int r = 0; r < R; ++r) c[r][0] = c[r][1] = vdupq_n_s32(0);
  for (int32_t s = 0; s < n16; ++s, a += 64, w += 32) {
    const int8x16_t b0 = vld1q_s8(w), b1 = vld1q_s8(w + 16);
    for (int r = 0; r < R; ++r) {
      const int8x16_t x = vld1q_s8(a + 16 * r);
      c[r][0] = vpadalq_s16(c[r][0], vmlal_s8(vmull_s8(vget_low_s8(x), vget_low_s8(b0)), vget_high_s8(x), vget_high_s8(b0)));
      c[r][1] = vpadalq_s16(c[r][1], vmlal_s8(vmull_s8(vget_low_s8(x), vget_low_s8(b1)), vget_high_s8(x), vget_high_s8(b1)));
    }
  }
  for (int r = 0; r < R; ++r) {
    const int32x2_t s0 = vpadd_s32(vget_low_s32(c[r][0]), vget_high_s32(c[r][0]));
    const int32x2_t s1 = vpadd_s32(vget_low_s32(c[r][1]), vget_high_s32(c[r][1]));
    vst1_s32(y + r * ldy, vpadd_s32(s0, s1));
  }
}

// rows (1..4) of a quad x one column pair
static inline void tile_x2(int rows, const int8_t* a, const int8_t* w, int32_t n16, int32_t* y, int32_t ldy) {
  switch (rows) {
    case 1: return tile_rx2<1>(a, w, n16, y, ldy);
    case 2: return tile_rx2<2>(a, w, n16, y, ldy);
    case 3: return tile_rx2<3>(a, w, n16, y, ldy);
    default: return tile4x2(a, w, n16, y, ldy);
  }
}

// Packs rows r[0..rows) (kp bytes each, kp % 16 == 0) into the quad layout; rows >= `rows` are left untouched
// (tile_x2 does not read them).
static inline void pack_quad(const int8_t* const* r, int rows, int64_t kp, int8_t* dst) {
  for (int64_t k = 0; k < kp; k += 16, dst += 64)
    for (int i = 0; i < rows; ++i) vst1q_s8(dst + 16 * i, vld1q_s8(r[i] + k));
}

// Packs columns c0, c1 (K bytes each; c1 may be null = zero column) into the pair layout, zero-padded to kp.
static inline void pack_pair(const int8_t* c0, const int8_t* c1, int64_t k, int64_t kp, int8_t* dst) {
  std::memset(dst, 0, static_cast<size_t>(2 * kp));
  for (int64_t i = 0; i < k; ++i) {
    dst[(i / 16) * 32 + i % 16] = c0[i];
    if (c1) dst[(i / 16) * 32 + 16 + i % 16] = c1[i];
  }
}

// Exact-size byte copy for short im2col runs (a few to a few hundred bytes): fixed-size moves, the last one
// overlapping, inline (a memcpy call per 21-byte stem run cost ~1 ms per ReActNet-18 stem on the A53).
static inline __attribute__((always_inline)) void copy_bytes(int8_t* d, const int8_t* s, int64_t n) {
  if (n >= 16) {
    for (int64_t i = 0; i + 16 < n; i += 16) vst1q_s8(d + i, vld1q_s8(s + i));
    vst1q_s8(d + n - 16, vld1q_s8(s + n - 16));
  } else if (n >= 8) {
    vst1_s8(d, vld1_s8(s));
    vst1_s8(d + n - 8, vld1_s8(s + n - 8));
  } else if (n >= 4) {
    std::memcpy(d, s, 4);
    std::memcpy(d + n - 4, s + n - 4, 4);
  } else {
    for (int64_t i = 0; i < n; ++i) d[i] = s[i];
  }
}

// True if no weight is -128 (the int16 pair sums of tile4x2 are exact).
static inline bool weights_fit_mull(const int8_t* w, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    if (w[i] == -128) return false;
  return true;
}

}  // namespace a32gemm
}  // namespace bitino
#endif
