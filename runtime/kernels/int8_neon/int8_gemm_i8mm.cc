// SMMLA int8 GEMM micro-kernel (requires i8mm; compiled with +dotprod+i8mm,
// selected only when cpu_info() reports kFeatI8mm).
//
// Tile: 8 rows x 8 columns, K chunk C = 8 bytes. Per chunk the packed tile
// holds a_i = rows (2i, 2i+1) x 8 k and b_j = cols (2j, 2j+1) x 8 k, one
// 16-byte register each. vmmlaq_s32(acc, a_i, b_j) accumulates the 2x2 block
//   acc[i][j] = [r0.c0, r0.c1, r1.c0, r1.c1]   (r = 2i + {0,1}, c = 2j + {0,1})
// 32 MACs per instruction (SDOT: 16). 16 accumulators + 8 operands = 24
// vector registers. The epilogue re-pairs the 64-bit halves into rows:
//   zip1_64(acc[i][0], acc[i][1]) = row 2i, cols 0-3; zip2 = row 2i+1.
//
// Instantiated for NP = 1..4 valid row pairs (short tiles skip padding rows);
// with NP == 1 two accumulator sets take alternate chunks to hide latency.
#include "kernels/int8_neon/int8_gemm.h"

#if defined(__aarch64__)
#include "kernels/int8_neon/int8_gemm_epilogue.h"

namespace bitino {
namespace int8_neon {
namespace {

inline int32x4_t zip1_64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}
inline int32x4_t zip2_64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

template <int NP>
struct MmlaAcc {
  int32x4_t c[4][4];
  MmlaAcc() {
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i)
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j) c[i][j] = vdupq_n_s32(0);
  }
  // One K chunk (64 bytes of A and W).
  inline void step(const int8_t* a, const int8_t* w) {
    const int8x16_t b0 = vld1q_s8(w), b1 = vld1q_s8(w + 16), b2 = vld1q_s8(w + 32), b3 = vld1q_s8(w + 48);
#pragma clang loop unroll(full)
    for (int i = 0; i < NP; ++i) {
      const int8x16_t ai = vld1q_s8(a + 16 * i);
      c[i][0] = vmmlaq_s32(c[i][0], ai, b0);
      c[i][1] = vmmlaq_s32(c[i][1], ai, b1);
      c[i][2] = vmmlaq_s32(c[i][2], ai, b2);
      c[i][3] = vmmlaq_s32(c[i][3], ai, b3);
    }
  }
};

template <int NP>
void tiles_np(const GemmArgs& g, int64_t pb, int64_t ob0, int64_t ob1) {
  const int64_t kc = g.kc;
  const int8_t* a = g.a + pb * kc * 64;
  for (int64_t ob = ob0; ob < ob1; ++ob) {
    const int8_t* w = g.w + ob * kc * 64;
    MmlaAcc<NP> s0;
    int64_t c = 0;
    if (NP == 1) {
      MmlaAcc<NP> s1;
      for (; c + 2 <= kc; c += 2) {
        s0.step(a + 64 * c, w + 64 * c);
        s1.step(a + 64 * c + 64, w + 64 * c + 64);
      }
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j) s0.c[0][j] = vaddq_s32(s0.c[0][j], s1.c[0][j]);
    }
    for (; c < kc; ++c) s0.step(a + 64 * c, w + 64 * c);
    const int64_t p0 = pb * kTile, o0 = ob * kTile;
#pragma clang loop unroll(full)
    for (int i = 0; i < NP; ++i) {
      int8_store_row8(g, p0 + 2 * i, o0, zip1_64(s0.c[i][0], s0.c[i][1]), zip1_64(s0.c[i][2], s0.c[i][3]));
      int8_store_row8(g, p0 + 2 * i + 1, o0, zip2_64(s0.c[i][0], s0.c[i][1]), zip2_64(s0.c[i][2], s0.c[i][3]));
    }
  }
}

}  // namespace

void gemm_tiles_i8mm(const GemmArgs& g, int64_t pb, int64_t ob0, int64_t ob1) {
  const int64_t rows = g.rows - pb * kTile;
  switch (rows < kTile ? (rows + 1) / 2 : kTile / 2) {
    case 1: tiles_np<1>(g, pb, ob0, ob1); break;
    case 2: tiles_np<2>(g, pb, ob0, ob1); break;
    case 3: tiles_np<3>(g, pb, ob0, ob1); break;
    default: tiles_np<4>(g, pb, ob0, ob1); break;
  }
}

}  // namespace int8_neon
}  // namespace bitino
#endif
