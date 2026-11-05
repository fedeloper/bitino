// SMMLA (i8mm) int32 GEMM tiles for i8conv2d.neon_sdot / i8dense.neon_sdot (int8_full), compiled with
// -march=armv8.2-a+dotprod+i8mm and called only when cpu_info() reports kFeatI8mm. Same int32 sums as the SDOT
// tiles (exact integer accumulation; |sum| < 2^31 by gemm_sdot_supported).
//
// Operands (8 x 8 tiles, K chunk 8 bytes):
//   A [kc8][4 row pairs][2 rows][8 k]            64 bytes per chunk (pack_tile_rows_i8mm)
//   W [ob][kc8][4 column pairs][2 columns][8 k]  64 bytes per chunk
// vmmlaq_s32(acc[i][j], a_i, b_j) accumulates the 2x2 block [r0.c0, r0.c1, r1.c0, r1.c1] (r = 2i + {0, 1},
// c = 2j + {0, 1}): 32 MACs per instruction (SDOT 16), 16 accumulators + 8 operands. The rows are re-paired with
// 64-bit ZIPs: zip1(acc[i][0], acc[i][1]) = row 2i, columns 0-3; zip2 = row 2i + 1.
// This TU includes nothing with external linkage besides arm_neon.h / cstdint (no inline function can be merged
// across -march boundaries).
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>

namespace bitino {
namespace int8_full_i8mm {

namespace {
inline int32x4_t zip1_64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}
inline int32x4_t zip2_64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}
}  // namespace

// y[r * ldy + 8 ob + j] = sum_k a[r][k] w[8 ob + j][k] for rows r < nr of the tile (NP = ceil(nr / 2) row pairs
// computed: short tiles, e.g. Dense with one row, skip the padding pairs), ob < nob.
template <int NP, bool kOdd>  // kOdd: nr = 2 NP - 1 (the last pair's second row is padding)
void tiles_np(const int8_t* __restrict a, const int8_t* __restrict w, int64_t kc8, int64_t nob, int32_t* __restrict y, int64_t ldy) {
  for (int64_t ob = 0; ob < nob; ++ob, w += kc8 * 64) {
    int32x4_t c[NP][4];
#pragma clang loop unroll(full)
    for (int i = 0; i < NP; ++i)
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j) c[i][j] = vdupq_n_s32(0);
    const int8_t* ap = a;
    const int8_t* wp = w;
    for (int64_t s = 0; s < kc8; ++s, ap += 64, wp += 64) {
      const int8x16_t b0 = vld1q_s8(wp), b1 = vld1q_s8(wp + 16), b2 = vld1q_s8(wp + 32), b3 = vld1q_s8(wp + 48);
#pragma clang loop unroll(full)
      for (int i = 0; i < NP; ++i) {
        const int8x16_t ai = vld1q_s8(ap + 16 * i);
        c[i][0] = vmmlaq_s32(c[i][0], ai, b0);
        c[i][1] = vmmlaq_s32(c[i][1], ai, b1);
        c[i][2] = vmmlaq_s32(c[i][2], ai, b2);
        c[i][3] = vmmlaq_s32(c[i][3], ai, b3);
      }
    }
    int32_t* yo = y + ob * 8;
#pragma clang loop unroll(full)
    for (int i = 0; i < NP; ++i) {
      vst1q_s32(yo + (2 * i) * ldy, zip1_64(c[i][0], c[i][1]));
      vst1q_s32(yo + (2 * i) * ldy + 4, zip1_64(c[i][2], c[i][3]));
      if (!kOdd || i + 1 < NP) {
        vst1q_s32(yo + (2 * i + 1) * ldy, zip2_64(c[i][0], c[i][1]));
        vst1q_s32(yo + (2 * i + 1) * ldy + 4, zip2_64(c[i][2], c[i][3]));
      }
    }
  }
}

void tiles_i32(const int8_t* a, const int8_t* w, int64_t kc8, int64_t nob, int32_t* y, int64_t ldy, int nr) {
  switch (nr) {
    case 1: return tiles_np<1, true>(a, w, kc8, nob, y, ldy);
    case 2: return tiles_np<1, false>(a, w, kc8, nob, y, ldy);
    case 3: return tiles_np<2, true>(a, w, kc8, nob, y, ldy);
    case 4: return tiles_np<2, false>(a, w, kc8, nob, y, ldy);
    case 5: return tiles_np<3, true>(a, w, kc8, nob, y, ldy);
    case 6: return tiles_np<3, false>(a, w, kc8, nob, y, ldy);
    case 7: return tiles_np<4, true>(a, w, kc8, nob, y, ldy);
    default: return tiles_np<4, false>(a, w, kc8, nob, y, ldy);
  }
}

// 8 rows of kp bytes (kp a multiple of 16) -> A [kp / 8][4 pairs][2 rows][8 bytes].
void pack_tile_rows(const int8_t* const rp[8], int64_t kp, int8_t* __restrict dst) {
  for (int64_t k = 0; k < kp; k += 16, dst += 128) {
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      const int64x2_t r0 = vreinterpretq_s64_s8(vld1q_s8(rp[2 * i] + k)), r1 = vreinterpretq_s64_s8(vld1q_s8(rp[2 * i + 1] + k));
      vst1q_s64(reinterpret_cast<int64_t*>(dst + 16 * i), vzip1q_s64(r0, r1));
      vst1q_s64(reinterpret_cast<int64_t*>(dst + 64 + 16 * i), vzip2q_s64(r0, r1));
    }
  }
}

}  // namespace int8_full_i8mm
}  // namespace bitino
#endif
