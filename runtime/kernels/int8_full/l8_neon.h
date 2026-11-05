// NEON encode / decode of the 8-bit log2 code "l8" (q8::l8_encode / q8::l8_decode, int8_ops.l8_*), 16 values at
// a time. Bit-identical to the scalar definitions (tests/cpp/test_int8_full.cc: every code, every exponent offset,
// 2^24 fp32 bit patterns and the rounding boundaries).
//
// encode: SHRN #16 of the fp32 bits -> u16 [sign | Ew | t7]; UZP1 / UZP2 split the low bytes ((Ew & 1) << 7 | t7)
// and the high bytes (sign << 7 | Ew >> 1); Ew = SHL #1 + SRI #7; #{i : t7 >= B_i} = base[t7 >> 3] + (t7 >= thr[t7 >> 3])
// (every 8-wide bucket of t7 holds at most one boundary B_i); j = min(uqsub(8 min(uqsub(Ew, 110), 17) + kf, 9), 127)
// equals clamp(8 (Ew - 127) + kf + 127, 0, 127); code = sign ? -j : j.
// decode (exponent offset e): k = min(|c|, 127) + 1; the four bytes of ldexp(D[k & 7], e + (k >> 3) - 16) come from
// 16 / 32-entry TBL tables indexed by k & 7, k & 15, k >> 3 (index 0xFF for c == 0 gives 0), the sign bit is c's.
#pragma once

#if defined(__aarch64__)
#include <arm_neon.h>

#include <cstdint>

#include "kernels/int8_full/q8_math.h"

namespace bitino {
namespace l8neon {

struct EncTables {
  uint8x16_t base, thr;
  EncTables() {
    uint8_t b[16], t[16];
    for (int k = 0; k < 16; ++k) {
      b[k] = 0;
      t[k] = 255;
      for (int x : q8::kL8B) {
        if (x < 8 * k) ++b[k];
        if (x >= 8 * k && x < 8 * k + 8) t[k] = static_cast<uint8_t>(x);
      }
    }
    base = vld1q_u8(b);
    thr = vld1q_u8(t);
  }
};

// 16 fp32 values (w0: elements 0..3, ..., w3: 12..15) -> 16 l8 codes (exponent 0).
inline __attribute__((always_inline)) int8x16_t encode16(float32x4_t w0, float32x4_t w1, float32x4_t w2, float32x4_t w3,
                                                         const EncTables& T) {
  const uint16x8_t h0 = vshrn_high_n_u32(vshrn_n_u32(vreinterpretq_u32_f32(w0), 16), vreinterpretq_u32_f32(w1), 16);
  const uint16x8_t h1 = vshrn_high_n_u32(vshrn_n_u32(vreinterpretq_u32_f32(w2), 16), vreinterpretq_u32_f32(w3), 16);
  const uint8x16_t lo = vuzp1q_u8(vreinterpretq_u8_u16(h0), vreinterpretq_u8_u16(h1));
  const uint8x16_t hi = vuzp2q_u8(vreinterpretq_u8_u16(h0), vreinterpretq_u8_u16(h1));
  const uint8x16_t ew = vsriq_n_u8(vshlq_n_u8(hi, 1), lo, 7);
  const uint8x16_t t = vandq_u8(lo, vdupq_n_u8(0x7F));
  const uint8x16_t bk = vshrq_n_u8(t, 3);
  const uint8x16_t kf = vsubq_u8(vqtbl1q_u8(T.base, bk), vcgeq_u8(t, vqtbl1q_u8(T.thr, bk)));
  const uint8x16_t e = vminq_u8(vqsubq_u8(ew, vdupq_n_u8(110)), vdupq_n_u8(17));
  const uint8x16_t j = vminq_u8(vqsubq_u8(vaddq_u8(vshlq_n_u8(e, 3), kf), vdupq_n_u8(9)), vdupq_n_u8(127));
  const int8x16_t m = vreinterpretq_s8_u8(vcltzq_s8(vreinterpretq_s8_u8(hi)));
  return vsubq_s8(veorq_s8(vreinterpretq_s8_u8(j), m), m);
}

struct DecTables {
  uint8x16_t t0, t1, t2;
  uint8x16x2_t t3;
  explicit DecTables(int e = 0) {
    uint8_t b0[16] = {}, b1[16] = {}, b2[16], b3[32] = {};
    for (int r = 0; r < 8; ++r) {
      b0[r] = static_cast<uint8_t>(q8::kL8DBits[r]);
      b1[r] = static_cast<uint8_t>(q8::kL8DBits[r] >> 8);
    }
    for (int i = 0; i < 16; ++i) {
      const int ef = 127 + e + (i >> 3) - 16;  // parity of the biased exponent for k & 15 = i
      b2[i] = static_cast<uint8_t>(((ef & 1) << 7) | ((q8::kL8DBits[i & 7] >> 16) & 0x7F));
    }
    for (int q = 0; q <= 16; ++q) b3[q] = static_cast<uint8_t>((127 + e + q - 16) >> 1);
    t0 = vld1q_u8(b0);
    t1 = vld1q_u8(b1);
    t2 = vld1q_u8(b2);
    t3.val[0] = vld1q_u8(b3);
    t3.val[1] = vld1q_u8(b3 + 16);
  }
};

// 16 l8 codes -> 16 fp32 values (d[0]: elements 0..3, ...), exponent offset of the tables.
inline __attribute__((always_inline)) void decode16(int8x16_t c, const DecTables& T, float32x4_t d[4]) {
  const uint8x16_t k = vaddq_u8(vreinterpretq_u8_s8(vqabsq_s8(c)), vdupq_n_u8(1));
  const uint8x16_t z = vceqzq_s8(c);
  const uint8x16_t b0 = vqtbl1q_u8(T.t0, vorrq_u8(vandq_u8(k, vdupq_n_u8(7)), z));
  const uint8x16_t b1 = vqtbl1q_u8(T.t1, vorrq_u8(vandq_u8(k, vdupq_n_u8(7)), z));
  const uint8x16_t b2 = vqtbl1q_u8(T.t2, vorrq_u8(vandq_u8(k, vdupq_n_u8(15)), z));
  const uint8x16_t b3 = vorrq_u8(vqtbl2q_u8(T.t3, vorrq_u8(vshrq_n_u8(k, 3), z)),
                                 vandq_u8(vreinterpretq_u8_s8(c), vdupq_n_u8(0x80)));
  const uint8x16_t l01 = vzip1q_u8(b0, b1), h01 = vzip2q_u8(b0, b1), l23 = vzip1q_u8(b2, b3), h23 = vzip2q_u8(b2, b3);
  d[0] = vreinterpretq_f32_u16(vzip1q_u16(vreinterpretq_u16_u8(l01), vreinterpretq_u16_u8(l23)));
  d[1] = vreinterpretq_f32_u16(vzip2q_u16(vreinterpretq_u16_u8(l01), vreinterpretq_u16_u8(l23)));
  d[2] = vreinterpretq_f32_u16(vzip1q_u16(vreinterpretq_u16_u8(h01), vreinterpretq_u16_u8(h23)));
  d[3] = vreinterpretq_f32_u16(vzip2q_u16(vreinterpretq_u16_u8(h01), vreinterpretq_u16_u8(h23)));
}

// Linear residual: fp32(c - z) * s for 16 codes.
inline __attribute__((always_inline)) void decode16_linear(int8x16_t c, int32x4_t z, float32x4_t s, float32x4_t d[4]) {
  const int16x8_t l = vmovl_s8(vget_low_s8(c)), h = vmovl_high_s8(c);
  d[0] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(l)), z)), s);
  d[1] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_high_s16(l), z)), s);
  d[2] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(h)), z)), s);
  d[3] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_high_s16(h), z)), s);
}

}  // namespace l8neon
}  // namespace bitino
#endif
