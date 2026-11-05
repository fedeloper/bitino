// NEON helpers shared by the optimized binary kernels (AArch64 only).
//
// f32 -> bit packing: compare 4 floats per instruction, narrow the 32-bit
// lane masks to bytes, AND with the bit weights 1,2,..,128 and fold with
// pairwise adds. Produces exactly the bits of pack_row_f32() (bit = 1 iff
// !(x >= t) resp. !(x > t), so NaN maps to 1 like the scalar code).
#pragma once

#if defined(__aarch64__)
#include <arm_neon.h>

#include "kernels/bitpack.h"

namespace bitino {

// Returns the "is +1" byte masks (0xFF / 0x00) for 16 floats.
template <bool kGt, bool kThr>
inline uint8x16_t positive_mask16(const float* x, const float* thr) {
  uint32x4_t m[4];
  const float32x4x4_t v = vld1q_f32_x4(x);  // one 64-byte load
  float32x4x4_t t;
  if (kThr) t = vld1q_f32_x4(thr);
  for (int k = 0; k < 4; ++k) {
    if (kThr) {
      m[k] = kGt ? vcgtq_f32(v.val[k], t.val[k]) : vcgeq_f32(v.val[k], t.val[k]);
    } else {
      m[k] = kGt ? vcgtzq_f32(v.val[k]) : vcgezq_f32(v.val[k]);
    }
  }
  uint16x8_t h0 = vuzp1q_u16(vreinterpretq_u16_u32(m[0]), vreinterpretq_u16_u32(m[1]));
  uint16x8_t h1 = vuzp1q_u16(vreinterpretq_u16_u32(m[2]), vreinterpretq_u16_u32(m[3]));
  return vuzp1q_u8(vreinterpretq_u8_u16(h0), vreinterpretq_u8_u16(h1));
}

// Packs `words` full 32-float groups (no tail) into `out`. `thr` (if non-null)
// is indexed like `x`.
template <bool kGt, bool kThr>
inline void pack_words_f32_neon_t(const float* x, int64_t words, uint32_t* out, const float* thr) {
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t wts = vld1q_u8(kW);
  int64_t w = 0;
  // Two words per iteration share the final folds (with thresholds that would
  // be 32 live vectors after the 64-byte loads: one word at a time).
  for (; !kThr && w + 2 <= words; w += 2) {
    const float* xp = x + 32 * w;
    const float* tp = kThr ? thr + 32 * w : nullptr;
    uint8x16_t b0 = vbicq_u8(wts, positive_mask16<kGt, kThr>(xp, tp));
    uint8x16_t b1 = vbicq_u8(wts, positive_mask16<kGt, kThr>(xp + 16, kThr ? tp + 16 : nullptr));
    uint8x16_t b2 = vbicq_u8(wts, positive_mask16<kGt, kThr>(xp + 32, kThr ? tp + 32 : nullptr));
    uint8x16_t b3 = vbicq_u8(wts, positive_mask16<kGt, kThr>(xp + 48, kThr ? tp + 48 : nullptr));
    uint8x16_t r = vpaddq_u8(vpaddq_u8(b0, b1), vpaddq_u8(b2, b3));
    r = vpaddq_u8(r, r);
    vst1_u32(out + w, vreinterpret_u32_u8(vget_low_u8(r)));
  }
  for (; w < words; ++w) {
    const float* xp = x + 32 * w;
    const float* tp = kThr ? thr + 32 * w : nullptr;
    uint8x16_t b0 = vbicq_u8(wts, positive_mask16<kGt, kThr>(xp, tp));
    uint8x16_t b1 = vbicq_u8(wts, positive_mask16<kGt, kThr>(xp + 16, kThr ? tp + 16 : nullptr));
    uint8x16_t r = vpaddq_u8(b0, b1);
    r = vpaddq_u8(r, r);
    r = vpaddq_u8(r, r);
    out[w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 0);
  }
}

inline void pack_words_f32_neon(const float* x, int64_t words, uint32_t* out, BinarizeMode mode, const float* thr) {
  if (mode == BinarizeMode::kGt) {
    if (thr) pack_words_f32_neon_t<true, true>(x, words, out, thr);
    else pack_words_f32_neon_t<true, false>(x, words, out, nullptr);
  } else {
    if (thr) pack_words_f32_neon_t<false, true>(x, words, out, thr);
    else pack_words_f32_neon_t<false, false>(x, words, out, nullptr);
  }
}

// Drop-in replacement for pack_row_f32(): NEON for full words, scalar tail.
inline void pack_row_f32_neon(const float* x, int64_t c, uint32_t* out, BinarizeMode mode, const float* thr) {
  int64_t full = c / 32;
  pack_words_f32_neon(x, full, out, mode, thr);
  if (c > full * 32) pack_row_f32(x + full * 32, c - full * 32, out + full, mode, thr ? thr + full * 32 : nullptr);
}

// sign_zero bit-planes (ternary BDense, BMatMul): nz and neg words of c floats.
// Packs 16 lane masks (0 / ~0 per float, as 4 uint32x4) into 16 bits.
inline uint8x16_t narrow_masks(uint32x4_t m0, uint32x4_t m1, uint32x4_t m2, uint32x4_t m3) {
  uint16x8_t h0 = vuzp1q_u16(vreinterpretq_u16_u32(m0), vreinterpretq_u16_u32(m1));
  uint16x8_t h1 = vuzp1q_u16(vreinterpretq_u16_u32(m2), vreinterpretq_u16_u32(m3));
  return vuzp1q_u8(vreinterpretq_u8_u16(h0), vreinterpretq_u8_u16(h1));
}

// nz = (x > t) | (x < t) (NaN -> 0), neg = (x < t): exactly binarize(kSignZero, x, t).
// One nz and one neg word per 32 floats: 64-byte loads, the two planes' bit
// weights folded together (the last pairwise add yields [nz word, neg word]).
template <bool kThr>
inline void pack_ternary_words_neon(const float* x, int64_t words, uint32_t* nz, uint32_t* neg, const float* thr) {
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t wts = vld1q_u8(kW);
  for (int64_t w = 0; w < words; ++w) {
    uint8x16_t bn[2], bs[2];
    for (int h = 0; h < 2; ++h) {
      const int64_t i = 32 * w + 16 * h;
      const float32x4x4_t v = vld1q_f32_x4(x + i);
      float32x4x4_t t;
      if (kThr) t = vld1q_f32_x4(thr + i);
      uint32x4_t lt[4], nm[4];
      for (int k = 0; k < 4; ++k) {
        lt[k] = kThr ? vcltq_f32(v.val[k], t.val[k]) : vcltzq_f32(v.val[k]);
        nm[k] = vorrq_u32(kThr ? vcgtq_f32(v.val[k], t.val[k]) : vcgtzq_f32(v.val[k]), lt[k]);
      }
      bn[h] = vandq_u8(wts, narrow_masks(nm[0], nm[1], nm[2], nm[3]));
      bs[h] = vandq_u8(wts, narrow_masks(lt[0], lt[1], lt[2], lt[3]));
    }
    uint8x16_t r = vpaddq_u8(vpaddq_u8(bn[0], bn[1]), vpaddq_u8(bs[0], bs[1]));  // [nz 8 bytes | neg 8 bytes]
    r = vpaddq_u8(r, r);  // u32 lanes: nz word, neg word
    vst1q_lane_u32(nz + w, vreinterpretq_u32_u8(r), 0);
    vst1q_lane_u32(neg + w, vreinterpretq_u32_u8(r), 1);
  }
}

inline void pack_ternary_row_neon(const float* x, int64_t c, uint32_t* nz, uint32_t* neg, const float* thr) {
  const int64_t full = c / 32;
  if (thr) pack_ternary_words_neon<true>(x, full, nz, neg, thr);
  else pack_ternary_words_neon<false>(x, full, nz, neg, nullptr);
  if (c == full * 32) return;
  uint32_t bn = 0, bs = 0;
  for (int64_t b = 0; b < c - full * 32; ++b) {
    const float v = x[full * 32 + b], t = thr ? thr[full * 32 + b] : 0.f;
    bn |= uint32_t(v > t || v < t) << b;
    bs |= uint32_t(v < t) << b;
  }
  nz[full] = bn;
  neg[full] = bs;
}

}  // namespace bitino

#endif  // __aarch64__
