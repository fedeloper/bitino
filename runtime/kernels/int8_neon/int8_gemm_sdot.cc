// SDOT int8 GEMM micro-kernel (requires dotprod; compiled with +dotprod).
//
// Tile: 8 rows (pixels) x 8 columns (channels), K chunk C = 4 bytes.
// Per chunk: x0 = rows 0-3, x1 = rows 4-7, w0 = cols 0-3, w1 = cols 4-7, each
// lane holding 4 consecutive k. vdotq_laneq_s32(acc, w, x, l) adds to every
// column lane j the dot of w[j][0..3] with x[row l][0..3], so
//   acc[r][h] lane j = sum_k A[r][k] * W[4h + j][k]
// i.e. each accumulator is 4 consecutive output channels of one pixel, the
// NHWC output order. 16 accumulators + 4 operands = 20 vector registers.
//
// The kernel is instantiated for R = 1..8 valid rows so that short tiles
// (Dense with few rows, the last pixel tile) skip the padding rows. With
// R <= 2 two accumulator sets take alternate chunks to hide SDOT latency.
#include "kernels/int8_neon/int8_gemm.h"

#if defined(__aarch64__)
#include "kernels/int8_neon/int8_gemm_epilogue.h"

namespace bitino {
namespace int8_neon {
namespace {

template <int R>
struct SdotAcc {
  int32x4_t c[8][2];
  SdotAcc() {
#pragma clang loop unroll(full)
    for (int r = 0; r < 8; ++r) c[r][0] = c[r][1] = vdupq_n_s32(0);
  }
  // One K chunk (32 bytes of A and W).
  inline void step(const int8_t* a, const int8_t* w) {
    const int8x16_t x0 = vld1q_s8(a);
    const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16);
#define BF_SDOT_ROW(r, x, l)                              \
  if (R > r) {                                            \
    c[r][0] = vdotq_laneq_s32(c[r][0], w0, x, l);         \
    c[r][1] = vdotq_laneq_s32(c[r][1], w1, x, l);         \
  }
    BF_SDOT_ROW(0, x0, 0)
    BF_SDOT_ROW(1, x0, 1)
    BF_SDOT_ROW(2, x0, 2)
    BF_SDOT_ROW(3, x0, 3)
    if (R > 4) {
      const int8x16_t x1 = vld1q_s8(a + 16);
      BF_SDOT_ROW(4, x1, 0)
      BF_SDOT_ROW(5, x1, 1)
      BF_SDOT_ROW(6, x1, 2)
      BF_SDOT_ROW(7, x1, 3)
    }
#undef BF_SDOT_ROW
  }
};

template <int R>
void tiles_r(const GemmArgs& g, int64_t pb, int64_t ob0, int64_t ob1) {
  const int64_t kc = g.kc;
  const int8_t* a = g.a + pb * kc * 32;
  for (int64_t ob = ob0; ob < ob1; ++ob) {
    const int8_t* w = g.w + ob * kc * 32;
    SdotAcc<R> s0;
    int64_t c = 0;
    if (R <= 2) {
      SdotAcc<R> s1;
      for (; c + 2 <= kc; c += 2) {
        s0.step(a + 32 * c, w + 32 * c);
        s1.step(a + 32 * c + 32, w + 32 * c + 32);
      }
#pragma clang loop unroll(full)
      for (int r = 0; r < R; ++r) {
        s0.c[r][0] = vaddq_s32(s0.c[r][0], s1.c[r][0]);
        s0.c[r][1] = vaddq_s32(s0.c[r][1], s1.c[r][1]);
      }
    }
    for (; c < kc; ++c) s0.step(a + 32 * c, w + 32 * c);
    const int64_t p0 = pb * kTile, o0 = ob * kTile;
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) int8_store_row8(g, p0 + r, o0, s0.c[r][0], s0.c[r][1]);
  }
}

}  // namespace

void gemm_tiles_sdot(const GemmArgs& g, int64_t pb, int64_t ob0, int64_t ob1) {
  const int64_t rows = g.rows - pb * kTile;
  switch (rows < kTile ? rows : kTile) {
    case 1: tiles_r<1>(g, pb, ob0, ob1); break;
    case 2: tiles_r<2>(g, pb, ob0, ob1); break;
    case 3: tiles_r<3>(g, pb, ob0, ob1); break;
    case 4: tiles_r<4>(g, pb, ob0, ob1); break;
    case 5: tiles_r<5>(g, pb, ob0, ob1); break;
    case 6: tiles_r<6>(g, pb, ob0, ob1); break;
    case 7: tiles_r<7>(g, pb, ob0, ob1); break;
    default: tiles_r<8>(g, pb, ob0, ob1); break;
  }
}

}  // namespace int8_neon
}  // namespace bitino
#endif
