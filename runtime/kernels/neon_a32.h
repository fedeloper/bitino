// AArch32 (armeabi-v7a) NEON support for the NEON kernels written for AArch64.
//
// On a 32-bit ARM build with NEON (__ARM_NEON without __aarch64__) this header defines BF_NEON_A32 = 1 and
// provides the AArch64-only intrinsics those kernels use (ADDP on q registers, ADDV, UZP1/ZIP1, TBL on 64-byte
// tables, compares against zero, FCVTNS, ...) as inline functions built from ARMv7 NEON, with the AArch64
// results bit for bit. On every other target it only defines BF_NEON_A32 = 0, so including it is free.
//
// Floating point: AArch32 Advanced SIMD always runs with the "standard FPSCR value": round to nearest, default
// NaN and flush-to-zero (denormal inputs and results become +-0), unlike AArch64 NEON and the scalar VFP code,
// which are IEEE. The kernels' integer work (popcounts, bit packing compares against zero, which are done here
// on the bit patterns) is unaffected; float arithmetic on denormal values is the one place where an AArch32
// NEON kernel can differ from the reference (see docs/numerical-correctness.md, "AArch32").
#pragma once

#if defined(__ARM_NEON) && !defined(__aarch64__)
#define BF_NEON_A32 1
#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#define BF_A32_INLINE static inline __attribute__((always_inline))

// ---- 64-byte loads: clang lowers the AArch32 vld1q_*_x4 builtins to VLD1 with a 256-bit alignment hint, which
// faults (SIGBUS) on pointers that are only 4- or 16-byte aligned. Four plain loads instead.
BF_A32_INLINE float32x4x4_t bf_a32_vld1q_f32_x4(const float* p) {
  float32x4x4_t r;
  r.val[0] = vld1q_f32(p);
  r.val[1] = vld1q_f32(p + 4);
  r.val[2] = vld1q_f32(p + 8);
  r.val[3] = vld1q_f32(p + 12);
  return r;
}
BF_A32_INLINE uint8x16x4_t bf_a32_vld1q_u8_x4(const uint8_t* p) {
  uint8x16x4_t r;
  r.val[0] = vld1q_u8(p);
  r.val[1] = vld1q_u8(p + 16);
  r.val[2] = vld1q_u8(p + 32);
  r.val[3] = vld1q_u8(p + 48);
  return r;
}
#undef vld1q_f32_x4
#undef vld1q_u8_x4
#define vld1q_f32_x4(p) bf_a32_vld1q_f32_x4(p)
#define vld1q_u8_x4(p) bf_a32_vld1q_u8_x4(p)

// ---- pairwise adds of two q registers (ADDP): [a0+a1, a2+a3, ..., b0+b1, ...]
BF_A32_INLINE uint8x16_t vpaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vcombine_u8(vpadd_u8(vget_low_u8(a), vget_high_u8(a)), vpadd_u8(vget_low_u8(b), vget_high_u8(b)));
}
BF_A32_INLINE uint16x8_t vpaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vcombine_u16(vpadd_u16(vget_low_u16(a), vget_high_u16(a)), vpadd_u16(vget_low_u16(b), vget_high_u16(b)));
}
BF_A32_INLINE uint32x4_t vpaddq_u32(uint32x4_t a, uint32x4_t b) {
  return vcombine_u32(vpadd_u32(vget_low_u32(a), vget_high_u32(a)), vpadd_u32(vget_low_u32(b), vget_high_u32(b)));
}
BF_A32_INLINE int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
  return vcombine_s16(vpadd_s16(vget_low_s16(a), vget_high_s16(a)), vpadd_s16(vget_low_s16(b), vget_high_s16(b)));
}
BF_A32_INLINE int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  return vcombine_s32(vpadd_s32(vget_low_s32(a), vget_high_s32(a)), vpadd_s32(vget_low_s32(b), vget_high_s32(b)));
}

// ---- horizontal reductions (ADDV, UADDLV, FADDP, FMAXP)
BF_A32_INLINE uint32_t vaddvq_u32(uint32x4_t v) {
  const uint32x2_t s = vpadd_u32(vget_low_u32(v), vget_high_u32(v));
  return vget_lane_u32(vpadd_u32(s, s), 0);
}
BF_A32_INLINE int32_t vaddvq_s32(int32x4_t v) {
  const int32x2_t s = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
  return vget_lane_s32(vpadd_s32(s, s), 0);
}
BF_A32_INLINE uint16_t vaddlvq_u8(uint8x16_t v) {
  const uint64x2_t s = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(v)));
  return static_cast<uint16_t>(vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1));
}
BF_A32_INLINE uint32_t vaddlvq_u16(uint16x8_t v) {
  const uint64x2_t s = vpaddlq_u32(vpaddlq_u16(v));
  return static_cast<uint32_t>(vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1));
}
BF_A32_INLINE uint32_t vmaxvq_u32(uint32x4_t v) {
  const uint32x2_t m = vpmax_u32(vget_low_u32(v), vget_high_u32(v));
  return vget_lane_u32(vpmax_u32(m, m), 0);
}
// (v0 + v1) + (v2 + v3), as FADDP twice
BF_A32_INLINE float vaddvq_f32(float32x4_t v) {
  const float32x2_t s = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
  return vget_lane_f32(vpadd_f32(s, s), 0);
}
BF_A32_INLINE float vmaxvq_f32(float32x4_t v) {
  const float32x2_t m = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
  return vget_lane_f32(vpmax_f32(m, m), 0);
}

// ---- widening of the high half
BF_A32_INLINE int16x8_t vmovl_high_s8(int8x16_t v) { return vmovl_s8(vget_high_s8(v)); }
BF_A32_INLINE uint16x8_t vmovl_high_u8(uint8x16_t v) { return vmovl_u8(vget_high_u8(v)); }
BF_A32_INLINE int32x4_t vmovl_high_s16(int16x8_t v) { return vmovl_s16(vget_high_s16(v)); }
BF_A32_INLINE uint32x4_t vmovl_high_u16(uint16x8_t v) { return vmovl_u16(vget_high_u16(v)); }
#define vshrn_high_n_u32(r, a, n) vcombine_u16((r), vshrn_n_u32((a), (n)))

// ---- permutes (UZP1/UZP2, ZIP1/ZIP2)
BF_A32_INLINE uint8x16_t vuzp1q_u8(uint8x16_t a, uint8x16_t b) { return vuzpq_u8(a, b).val[0]; }
BF_A32_INLINE uint8x16_t vuzp2q_u8(uint8x16_t a, uint8x16_t b) { return vuzpq_u8(a, b).val[1]; }
BF_A32_INLINE uint16x8_t vuzp1q_u16(uint16x8_t a, uint16x8_t b) { return vuzpq_u16(a, b).val[0]; }
BF_A32_INLINE uint16x8_t vuzp2q_u16(uint16x8_t a, uint16x8_t b) { return vuzpq_u16(a, b).val[1]; }
BF_A32_INLINE uint8x16_t vzip1q_u8(uint8x16_t a, uint8x16_t b) { return vzipq_u8(a, b).val[0]; }
BF_A32_INLINE uint8x16_t vzip2q_u8(uint8x16_t a, uint8x16_t b) { return vzipq_u8(a, b).val[1]; }
BF_A32_INLINE uint16x8_t vzip1q_u16(uint16x8_t a, uint16x8_t b) { return vzipq_u16(a, b).val[0]; }
BF_A32_INLINE uint16x8_t vzip2q_u16(uint16x8_t a, uint16x8_t b) { return vzipq_u16(a, b).val[1]; }
BF_A32_INLINE float32x4_t vzip1q_f32(float32x4_t a, float32x4_t b) { return vzipq_f32(a, b).val[0]; }
BF_A32_INLINE float32x4_t vzip2q_f32(float32x4_t a, float32x4_t b) { return vzipq_f32(a, b).val[1]; }
BF_A32_INLINE uint64x2_t vzip1q_u64(uint64x2_t a, uint64x2_t b) { return vcombine_u64(vget_low_u64(a), vget_low_u64(b)); }
BF_A32_INLINE uint64x2_t vzip2q_u64(uint64x2_t a, uint64x2_t b) { return vcombine_u64(vget_high_u64(a), vget_high_u64(b)); }

// ---- table lookups on 16 / 32 / 64-byte tables (TBL / TBX: out-of-range indices give 0 / keep the lane)
BF_A32_INLINE uint8x16_t vqtbl1q_u8(uint8x16_t t, uint8x16_t idx) {
  uint8x8x2_t tt;
  tt.val[0] = vget_low_u8(t);
  tt.val[1] = vget_high_u8(t);
  return vcombine_u8(vtbl2_u8(tt, vget_low_u8(idx)), vtbl2_u8(tt, vget_high_u8(idx)));
}
BF_A32_INLINE uint8x8x4_t bf_a32_tbl32(uint8x16_t a, uint8x16_t b) {
  uint8x8x4_t t;
  t.val[0] = vget_low_u8(a);
  t.val[1] = vget_high_u8(a);
  t.val[2] = vget_low_u8(b);
  t.val[3] = vget_high_u8(b);
  return t;
}
BF_A32_INLINE uint8x16_t vqtbl2q_u8(uint8x16x2_t t, uint8x16_t idx) {
  const uint8x8x4_t tt = bf_a32_tbl32(t.val[0], t.val[1]);
  return vcombine_u8(vtbl4_u8(tt, vget_low_u8(idx)), vtbl4_u8(tt, vget_high_u8(idx)));
}
BF_A32_INLINE uint8x16_t vqtbx4q_u8(uint8x16_t r, uint8x16x4_t t, uint8x16_t idx) {
  const uint8x8x4_t lo = bf_a32_tbl32(t.val[0], t.val[1]), hi = bf_a32_tbl32(t.val[2], t.val[3]);
  const uint8x16_t idx2 = vsubq_u8(idx, vdupq_n_u8(32));  // indices 0..31 wrap to >= 224: out of range
  uint8x8_t r0 = vtbx4_u8(vget_low_u8(r), lo, vget_low_u8(idx)), r1 = vtbx4_u8(vget_high_u8(r), lo, vget_high_u8(idx));
  r0 = vtbx4_u8(r0, hi, vget_low_u8(idx2));
  r1 = vtbx4_u8(r1, hi, vget_high_u8(idx2));
  return vcombine_u8(r0, r1);
}
BF_A32_INLINE uint8x16_t vqtbl4q_u8(uint8x16x4_t t, uint8x16_t idx) { return vqtbx4q_u8(vdupq_n_u8(0), t, idx); }

// ---- compares against zero
BF_A32_INLINE uint8x16_t vceqzq_s8(int8x16_t v) { return vceqq_s8(v, vdupq_n_s8(0)); }
BF_A32_INLINE uint8x16_t vcltzq_s8(int8x16_t v) { return vcltq_s8(v, vdupq_n_s8(0)); }
// f32 compares against 0 with the IEEE results of FCMGE / FCMGT / FCMLT #0.0 (denormals compare by sign, NaN is
// false), computed on the bit patterns u: x >= 0 <=> u <= 0x7f800000 or u == 0x80000000 (-0); x > 0 <=> u in
// [1, 0x7f800000]; x < 0 <=> u in [0x80000001, 0xff800000]. A float compare would flush denormals to zero.
BF_A32_INLINE uint32x4_t vcgezq_f32(float32x4_t v) {
  const uint32x4_t u = vreinterpretq_u32_f32(v);
  return vorrq_u32(vcleq_u32(u, vdupq_n_u32(0x7f800000u)), vceqq_u32(u, vdupq_n_u32(0x80000000u)));
}
BF_A32_INLINE uint32x4_t vcgtzq_f32(float32x4_t v) {
  return vcltq_u32(vsubq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(1u)), vdupq_n_u32(0x7f800000u));
}
BF_A32_INLINE uint32x4_t vcltzq_f32(float32x4_t v) {
  return vcltq_u32(vsubq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x80000001u)), vdupq_n_u32(0x7f800000u));
}

// True if any of the n floats is a denormal (nonzero with a zero exponent field): the AArch32 NEON float compares
// of a row against thresholds are exact unless a value or a threshold is one (callers then use scalar code).
static inline bool bf_a32_any_denormal(const float* x, int64_t n) {
  uint32x4_t any = vdupq_n_u32(0);
  int64_t i = 0;
  // (u << 1) - 1 < 0x00ffffff  <=>  (u & 0x7fffffff) in [1, 0x007fffff]
  const uint32x4_t lim = vdupq_n_u32(0x00ffffffu), one = vdupq_n_u32(1u);
  for (; i + 4 <= n; i += 4) {
    const uint32x4_t u = vshlq_n_u32(vreinterpretq_u32_f32(vld1q_f32(x + i)), 1);
    any = vorrq_u32(any, vcltq_u32(vsubq_u32(u, one), lim));
  }
  bool r = vmaxvq_u32(any) != 0;
  for (; i < n; ++i) r |= std::fpclassify(x[i]) == FP_SUBNORMAL;
  return r;
}

// Whether a float compare of a row x against thresholds t can differ from IEEE on AArch32 NEON (which flushes
// denormal operands to +-0): only if some t is a denormal, or some t is +-0 and the x it meets is a denormal
// (a flushed denormal x keeps its order against a normal t). Thresholds are checked first: usually normal.
static inline bool bf_a32_thr_compare_inexact(const float* x, const float* t, int64_t n) {
  uint32x4_t den = vdupq_n_u32(0), zero = den;
  int64_t i = 0;
  const uint32x4_t lim = vdupq_n_u32(0x00ffffffu), one = vdupq_n_u32(1u), z = vdupq_n_u32(0u);
  for (; i + 4 <= n; i += 4) {
    const uint32x4_t u = vshlq_n_u32(vreinterpretq_u32_f32(vld1q_f32(t + i)), 1);
    den = vorrq_u32(den, vcltq_u32(vsubq_u32(u, one), lim));
    zero = vorrq_u32(zero, vceqq_u32(u, z));
  }
  bool any_den = vmaxvq_u32(den) != 0, any_zero = vmaxvq_u32(zero) != 0;
  for (; i < n; ++i) {
    any_den |= std::fpclassify(t[i]) == FP_SUBNORMAL;
    any_zero |= std::fpclassify(t[i]) == FP_ZERO;
  }
  return any_den || (any_zero && bf_a32_any_denormal(x, n));
}

// ---- float conversions and rounding
// nearbyint (ties to even) with the sign of zero of FRINTN: |v| < 2^23 rounds by adding and subtracting
// copysign(2^23, v) (the vector unit rounds to nearest even), larger values, infinities and NaN are integral.
BF_A32_INLINE float32x4_t bf_a32_rndn(float32x4_t v) {
#if defined(__ARM_FEATURE_DIRECTED_ROUNDING)
  return vrndnq_f32(v);
#else
  const uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x80000000u));
  const float32x4_t c = vreinterpretq_f32_u32(vorrq_u32(sign, vreinterpretq_u32_f32(vdupq_n_f32(8388608.f))));
  const float32x4_t r = vsubq_f32(vaddq_f32(v, c), c);
  const uint32x4_t small = vcltq_f32(vabsq_f32(v), vdupq_n_f32(8388608.f));
  return vbslq_f32(small, vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(r), sign)), v);
#endif
}
#if !defined(__ARM_FEATURE_DIRECTED_ROUNDING)
BF_A32_INLINE float32x4_t vrndnq_f32(float32x4_t v) { return bf_a32_rndn(v); }
// FCVTNS: round to nearest even, saturate, NaN -> 0 (VCVT.S32.F32 saturates and maps NaN to 0; after rounding
// it truncates nothing).
BF_A32_INLINE int32x4_t vcvtnq_s32_f32(float32x4_t v) { return vcvtq_s32_f32(bf_a32_rndn(v)); }
#endif

// ---- division (no vector divide on AArch32). Divisors that are the same power of two 2^e (|e| <= 126) in every
// lane, such as a 2x2 average pool's 4, multiply by the exact reciprocal 2^-e (the same result, except that a
// denormal quotient is flushed); otherwise four correctly rounded VFP divisions.
BF_A32_INLINE float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) {
  const uint32x4_t u = vreinterpretq_u32_f32(b);
  const uint32x2_t lo = vget_low_u32(u), hi = vget_high_u32(u);
  const uint32_t b0 = vget_lane_u32(lo, 0);
  const uint32_t e = (b0 >> 23) & 0xffu;
  if ((b0 & 0x007fffffu) == 0 && e >= 1 && e <= 253 && vget_lane_u32(lo, 1) == b0 && vget_lane_u32(hi, 0) == b0 &&
      vget_lane_u32(hi, 1) == b0)
    return vmulq_f32(a, vreinterpretq_f32_u32(vdupq_n_u32((b0 & 0x80000000u) | ((254u - e) << 23))));
  float32x4_t r = a;
  r = vsetq_lane_f32(vgetq_lane_f32(a, 0) / vgetq_lane_f32(b, 0), r, 0);
  r = vsetq_lane_f32(vgetq_lane_f32(a, 1) / vgetq_lane_f32(b, 1), r, 1);
  r = vsetq_lane_f32(vgetq_lane_f32(a, 2) / vgetq_lane_f32(b, 2), r, 2);
  r = vsetq_lane_f32(vgetq_lane_f32(a, 3) / vgetq_lane_f32(b, 3), r, 3);
  return r;
}

// ---- 32-bit integer division (int64 division is a slow library call on AArch32): UDIV when the CPU has the
// Thumb / ARM integer divide (HWCAP_IDIVT / IDIVA: every ARMv8 core, Cortex-A7/A15/A17), else the compiler's
// 32-bit helper.
#if defined(__linux__)
#if defined(__thumb__)
inline const bool kBfA32HwDiv = (getauxval(AT_HWCAP) & HWCAP_IDIVT) != 0;
#else
inline const bool kBfA32HwDiv = (getauxval(AT_HWCAP) & HWCAP_IDIVA) != 0;
#endif
#else
inline const bool kBfA32HwDiv = false;
#endif
BF_A32_INLINE uint32_t bf_a32_udiv(uint32_t a, uint32_t b) {
#if defined(__ARM_FEATURE_IDIV)
  return a / b;
#else
  if (kBfA32HwDiv) {
    uint32_t q;
    asm(".arch_extension idiv\n udiv %0, %1, %2" : "=r"(q) : "r"(a), "r"(b));
    return q;
  }
  return a / b;
#endif
}

// ---- integer intrinsics of the int8_full NEON kernels (kernels/int8_full/int8_full_neon.cc; 2026-11-12): the
// AArch64 "high half" forms are the AArch32 operation on the high halves (same lanes, same saturation), the
// reductions are pairwise-add / max trees (exact integer sums), TRN1 / TRN2 are the two results of VTRN.
BF_A32_INLINE int32x4_t vmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) { return vmlal_s16(a, vget_high_s16(b), vget_high_s16(c)); }
BF_A32_INLINE int32x4_t vmull_high_s16(int16x8_t a, int16x8_t b) { return vmull_s16(vget_high_s16(a), vget_high_s16(b)); }
BF_A32_INLINE int32x4_t vmlal_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) { return vmlal_n_s16(a, vget_high_s16(b), c); }
BF_A32_INLINE int16x8_t vqmovn_high_s32(int16x4_t r, int32x4_t a) { return vcombine_s16(r, vqmovn_s32(a)); }
BF_A32_INLINE int8x16_t vqmovn_high_s16(int8x8_t r, int16x8_t a) { return vcombine_s8(r, vqmovn_s16(a)); }
BF_A32_INLINE int16x8_t vsubl_high_s8(int8x16_t a, int8x16_t b) { return vsubl_s8(vget_high_s8(a), vget_high_s8(b)); }
BF_A32_INLINE int16x8_t vaddl_high_s8(int8x16_t a, int8x16_t b) { return vaddl_s8(vget_high_s8(a), vget_high_s8(b)); }
BF_A32_INLINE int16x8_t vaddw_high_s8(int16x8_t a, int8x16_t b) { return vaddw_s8(a, vget_high_s8(b)); }
BF_A32_INLINE int32x4_t vaddw_high_s16(int32x4_t a, int16x8_t b) { return vaddw_s16(a, vget_high_s16(b)); }
#define vshll_high_n_s16(a, n) vshll_n_s16(vget_high_s16(a), (n))
#define vshrn_high_n_s32(r, a, n) vcombine_s16((r), vshrn_n_s32((a), (n)))
// sum of 8 bytes modulo 256 (ADDV.8B)
BF_A32_INLINE uint8_t vaddv_u8(uint8x8_t v) {
  const uint64x1_t s = vpaddl_u32(vpaddl_u16(vpaddl_u8(v)));
  return static_cast<uint8_t>(vget_lane_u64(s, 0));
}
BF_A32_INLINE int32_t vaddlvq_s16(int16x8_t v) {
  const int64x2_t s = vpaddlq_s32(vpaddlq_s16(v));
  return static_cast<int32_t>(vgetq_lane_s64(s, 0) + vgetq_lane_s64(s, 1));
}
BF_A32_INLINE int8_t vmaxvq_s8(int8x16_t v) {
  int8x8_t m = vpmax_s8(vget_low_s8(v), vget_high_s8(v));
  m = vpmax_s8(m, m);
  m = vpmax_s8(m, m);
  m = vpmax_s8(m, m);
  return vget_lane_s8(m, 0);
}
BF_A32_INLINE uint16x8_t vcltzq_s16(int16x8_t v) { return vcltq_s16(v, vdupq_n_s16(0)); }
BF_A32_INLINE uint32x4_t vcltzq_s32(int32x4_t v) { return vcltq_s32(v, vdupq_n_s32(0)); }
BF_A32_INLINE uint8x16_t vtrn1q_u8(uint8x16_t a, uint8x16_t b) { return vtrnq_u8(a, b).val[0]; }
BF_A32_INLINE uint8x16_t vtrn2q_u8(uint8x16_t a, uint8x16_t b) { return vtrnq_u8(a, b).val[1]; }
BF_A32_INLINE uint16x8_t vtrn1q_u16(uint16x8_t a, uint16x8_t b) { return vtrnq_u16(a, b).val[0]; }
BF_A32_INLINE uint16x8_t vtrn2q_u16(uint16x8_t a, uint16x8_t b) { return vtrnq_u16(a, b).val[1]; }
BF_A32_INLINE uint32x4_t vtrn1q_u32(uint32x4_t a, uint32x4_t b) { return vtrnq_u32(a, b).val[0]; }
BF_A32_INLINE uint32x4_t vtrn2q_u32(uint32x4_t a, uint32x4_t b) { return vtrnq_u32(a, b).val[1]; }
BF_A32_INLINE uint64x2_t vtrn1q_u64(uint64x2_t a, uint64x2_t b) { return vcombine_u64(vget_low_u64(a), vget_low_u64(b)); }
BF_A32_INLINE uint64x2_t vtrn2q_u64(uint64x2_t a, uint64x2_t b) { return vcombine_u64(vget_high_u64(a), vget_high_u64(b)); }

#undef BF_A32_INLINE
#else
#define BF_NEON_A32 0
#endif
