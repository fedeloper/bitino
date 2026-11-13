// Optimized AArch64 NEON BConv2D / BDense ("bconv2d.neon_opt", "bdense.neon",
// "bdense.ternary_neon") and QBConv2D / QBDense ("qbconv2d.neon",
// "qbdense.neon": DoReFa bit-planes on the same micro-kernel, see the end).
// See the development log (not published) for the measurements behind each choice.
//
// BConv2D pipeline per run():
//  1. f32 input -> bits with NEON compares (bitpack_neon.h); skipped for bit
//     input. Small inputs are packed on the calling thread (one pool dispatch
//     less per run).
//  2. Work items = (pixel tile of mt_ pixels) x (output-channel chunk). Each of
//     n_tasks_ tasks owns a contiguous item range and a private patch buffer
//     (allocated in prepare), so im2col rows are built per tile (and group,
//     for grouped convolution) and stay in L1 instead of materializing the
//     full patch matrix. Output channels are split only when there are too
//     few pixel tiles (e.g. 7x7 layers).
//  3. Micro-kernel (inline asm): 4 tile rows x 4 output channels, 16 bytes per
//     step, acc_u8 += vcnt(a XOR w). A byte lane gains at most 8 per step, so
//     up to 31 steps are accumulated in uint8 before folding into uint32
//     totals. Weights are interleaved [oc block][step][4 oc] for one linear
//     stream. A tile row is one pixel (rows padded to 128 bits), or in pair
//     mode two pixels sharing each register half-and-half (rows padded to 64
//     bits only), which removes the padding waste of e.g. 3x3 x 32/64-channel
//     layers.
//  4. Vector epilogue: dot = K - 2*popcount (+ border padding correction),
//     float scale/offset with separate mul and add (no FMA, like the
//     reference), activation, f32 store or NEON bit packing of whole words.
#include <algorithm>
#include <cstring>

#include "bitino/cpu_features.h"
#include "kernels/bconv_plan.h"
#include "kernels/float_opt/eltwise_program.h"
#include "kernels/neon_a32.h"
#include "ops/op_defs.h"

#if defined(__aarch64__) || BF_NEON_A32
#include <arm_neon.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <limits>
#include <type_traits>

#include "kernels/binary_neon/bitpack_neon.h"
#include "kernels/int8_full/int8_full.h"
#include "kernels/int8_full/l8_neon.h"
#endif

namespace bitino {

#if defined(__aarch64__) || BF_NEON_A32
namespace {

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

constexpr int64_t kU8Steps = 31;        // 31 * 8 = 248 <= 255
constexpr int64_t kMaxTaps = 64;        // border tap masks are uint64
constexpr int64_t kMaxPairWords = 128;  // pair mode: row staged in a stack buffer

// Folds 4 uint8 accumulators (each lane <= 248) into one uint32x4 holding the
// 4 horizontal sums.
inline uint32x4_t reduce4(uint8x16_t c0, uint8x16_t c1, uint8x16_t c2, uint8x16_t c3) {
  uint16x8_t t01 = vpaddq_u16(vpaddlq_u8(c0), vpaddlq_u8(c1));
  uint16x8_t t23 = vpaddq_u16(vpaddlq_u8(c2), vpaddlq_u8(c3));
  return vpaddlq_u16(vpaddq_u16(t01, t23));
}

// Folds the 4 uint8 accumulators of one row after a run of n <= kU8Steps steps
// (each lane <= 8n) into one uint32x4 holding the 4 horizontal sums. Short
// runs pairwise-add in uint8 first while the sums still fit (8n * 2 <= 255 for
// n <= 15, 8n * 4 <= 255 for n <= 7): 5 / 6 instructions per row instead of 8.
inline uint32x4_t reduce4_n(uint8x16_t c0, uint8x16_t c1, uint8x16_t c2, uint8x16_t c3, int64_t n) {
  if (n <= 7) return vpaddlq_u16(vpaddlq_u8(vpaddq_u8(vpaddq_u8(c0, c1), vpaddq_u8(c2, c3))));
  if (n <= 15) return vpaddlq_u16(vpaddq_u16(vpaddlq_u8(vpaddq_u8(c0, c1)), vpaddlq_u8(vpaddq_u8(c2, c3))));
  return reduce4(c0, c1, c2, c3);
}

// The 4x4 tile micro-kernel: c[i][j] = sum over n steps of vcnt(a_i XOR w_j),
// a_i = 16 bytes of tile row i, w_j = 16 bytes of channel j (weights
// interleaved [step][4][16 bytes], one 64-byte load per step). Inline asm
// because clang's scheduler hoists all XORs ahead of the CNTs and spills.
// The first step sets the accumulators (no zeroing pass, 2 instructions per
// pair); the others run two steps per iteration so that each row loads both
// with one LDP: 4 LDP + 2 LD1 + 96 EOR/CNT/ADD + 2 loop = 52 instructions per
// step (one step per iteration: 55). Accumulators live in v0-v15 (v16-v31
// are clobbered: rows v16-v23, weights v24-v27, temporaries v28-v31).
// kAnd: vcnt(a_i AND w_j) instead (QBConv2D / QBDense bit-planes).
#if defined(__aarch64__)
#define BF_ROW4(OP, X, C0, C1, C2, C3)                                                                \
  OP " v28.16b, " X ".16b, v24.16b\n " OP " v29.16b, " X ".16b, v25.16b\n"                              \
  OP " v30.16b, " X ".16b, v26.16b\n " OP " v31.16b, " X ".16b, v27.16b\n"                              \
  "cnt v28.16b, v28.16b\n cnt v29.16b, v29.16b\n cnt v30.16b, v30.16b\n cnt v31.16b, v31.16b\n"          \
  "add %[" C0 "].16b, %[" C0 "].16b, v28.16b\n add %[" C1 "].16b, %[" C1 "].16b, v29.16b\n"              \
  "add %[" C2 "].16b, %[" C2 "].16b, v30.16b\n add %[" C3 "].16b, %[" C3 "].16b, v31.16b\n"
#define BF_SET4(OP, X, C0, C1, C2, C3)                                                                    \
  OP " %[" C0 "].16b, " X ".16b, v24.16b\n " OP " %[" C1 "].16b, " X ".16b, v25.16b\n"                   \
  OP " %[" C2 "].16b, " X ".16b, v26.16b\n " OP " %[" C3 "].16b, " X ".16b, v27.16b\n"                   \
  "cnt %[" C0 "].16b, %[" C0 "].16b\n cnt %[" C1 "].16b, %[" C1 "].16b\n"                                \
  "cnt %[" C2 "].16b, %[" C2 "].16b\n cnt %[" C3 "].16b, %[" C3 "].16b\n"
#define BF_ACC_OPERANDS                                                                                   \
  [c00] "+w"(c00), [c01] "+w"(c01), [c02] "+w"(c02), [c03] "+w"(c03), [c10] "+w"(c10), [c11] "+w"(c11),  \
  [c12] "+w"(c12), [c13] "+w"(c13), [c20] "+w"(c20), [c21] "+w"(c21), [c22] "+w"(c22), [c23] "+w"(c23),  \
  [c30] "+w"(c30), [c31] "+w"(c31), [c32] "+w"(c32), [c33] "+w"(c33)
#define BF_CLOBBERS                                                                                       \
  "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", \
  "v29", "v30", "v31"
#endif

struct Acc4x4 {
  uint8x16_t c[4][4];
};

#if defined(__aarch64__)
#define BF_RUN_4X4(OP)                                                                                    \
  asm volatile(                                                                                           \
      "ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[b]], #64\n"                                           \
      "ldr q16, [%[a0]], #16\n ldr q17, [%[a1]], #16\n ldr q18, [%[a2]], #16\n ldr q19, [%[a3]], #16\n"    \
      BF_SET4(OP, "v16", "c00", "c01", "c02", "c03") BF_SET4(OP, "v17", "c10", "c11", "c12", "c13")         \
      BF_SET4(OP, "v18", "c20", "c21", "c22", "c23") BF_SET4(OP, "v19", "c30", "c31", "c32", "c33")         \
      : [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3), [b] "+r"(b),                          \
        [c00] "=w"(c00), [c01] "=w"(c01), [c02] "=w"(c02), [c03] "=w"(c03), [c10] "=w"(c10), [c11] "=w"(c11), \
        [c12] "=w"(c12), [c13] "=w"(c13), [c20] "=w"(c20), [c21] "=w"(c21), [c22] "=w"(c22), [c23] "=w"(c23), \
        [c30] "=w"(c30), [c31] "=w"(c31), [c32] "=w"(c32), [c33] "=w"(c33)                                \
      :                                                                                                   \
      : BF_CLOBBERS);                                                                                     \
  if (pairs > 0) {                                                                                        \
    asm volatile(                                                                                         \
        "1:\n"                                                                                            \
        "ldp q16, q17, [%[a0]], #32\n ldp q18, q19, [%[a1]], #32\n"                                       \
        "ldp q20, q21, [%[a2]], #32\n ldp q22, q23, [%[a3]], #32\n"                                       \
        "ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[b]], #64\n"                                         \
        BF_ROW4(OP, "v16", "c00", "c01", "c02", "c03") BF_ROW4(OP, "v18", "c10", "c11", "c12", "c13")       \
        BF_ROW4(OP, "v20", "c20", "c21", "c22", "c23") BF_ROW4(OP, "v22", "c30", "c31", "c32", "c33")       \
        "ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[b]], #64\n"                                         \
        BF_ROW4(OP, "v17", "c00", "c01", "c02", "c03") BF_ROW4(OP, "v19", "c10", "c11", "c12", "c13")       \
        BF_ROW4(OP, "v21", "c20", "c21", "c22", "c23") BF_ROW4(OP, "v23", "c30", "c31", "c32", "c33")       \
        "subs %[n], %[n], #1\n"                                                                           \
        "b.ne 1b\n"                                                                                       \
        : [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3), [b] "+r"(b), [n] "+r"(pairs),       \
          BF_ACC_OPERANDS                                                                                 \
        :                                                                                                 \
        : BF_CLOBBERS);                                                                                   \
  }                                                                                                       \
  if ((n - 1) & 1) {                                                                                      \
    asm volatile(                                                                                         \
        "ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[b]], #64\n"                                         \
        "ldr q16, [%[a0]], #16\n ldr q18, [%[a1]], #16\n ldr q20, [%[a2]], #16\n ldr q22, [%[a3]], #16\n"  \
        BF_ROW4(OP, "v16", "c00", "c01", "c02", "c03") BF_ROW4(OP, "v18", "c10", "c11", "c12", "c13")       \
        BF_ROW4(OP, "v20", "c20", "c21", "c22", "c23") BF_ROW4(OP, "v22", "c30", "c31", "c32", "c33")       \
        : [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3), [b] "+r"(b), BF_ACC_OPERANDS        \
        :                                                                                                 \
        : BF_CLOBBERS);                                                                                   \
  }

// One run of n >= 1 steps (n <= kU8Steps); advances the pointers by n steps.
template <bool kAnd = false>
inline __attribute__((always_inline)) void run_4x4(const uint8_t*& a0, const uint8_t*& a1, const uint8_t*& a2, const uint8_t*& a3,
                                                   const uint8_t*& b, int64_t n, Acc4x4& acc) {
  uint8x16_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
  int64_t pairs = (n - 1) >> 1;
  if constexpr (kAnd) {
    BF_RUN_4X4("and")
  } else {
    BF_RUN_4X4("eor")
  }
  acc = Acc4x4{{{c00, c01, c02, c03}, {c10, c11, c12, c13}, {c20, c21, c22, c23}, {c30, c31, c32, c33}}};
}
#undef BF_RUN_4X4
#undef BF_ROW4
#undef BF_SET4
#undef BF_ACC_OPERANDS
#undef BF_CLOBBERS
#else
// AArch32: 16 q registers cannot hold the 16 accumulators of the 4x4 tile, so a 4x4 run is two runs of a 2x4 tile
// (rows 0-1, then rows 2-3, each over the same weights; the weight block of a run is 64 n bytes, L1 resident).
// Inline asm (compiled intrinsics spill and shuffle accumulators): accumulators q0-q3 (row 0, channels 0-3) and
// q4-q7 (row 1), rows q8 / q9, two weight registers q10 / q11 reloaded per channel pair, temporaries q12-q15;
// the first step sets the accumulators. 3 + 1 loads, 24 EOR/CNT/ADD and the loop per step and row pair. The
// 8 accumulators are stored to out[0..7] = row 0 channels 0-3, row 1 channels 0-3 (Acc4x4 row order).
#define BF_A32_SET2(OP)                                                                                   \
  OP " q0, q8, q10\n " OP " q4, q9, q10\n " OP " q1, q8, q11\n " OP " q5, q9, q11\n"                     \
  "vld1.8 {d20-d23}, [%[b]]!\n"                                                                          \
  "vcnt.8 q0, q0\n vcnt.8 q4, q4\n vcnt.8 q1, q1\n vcnt.8 q5, q5\n"                                      \
  OP " q2, q8, q10\n " OP " q6, q9, q10\n " OP " q3, q8, q11\n " OP " q7, q9, q11\n"                     \
  "vcnt.8 q2, q2\n vcnt.8 q6, q6\n vcnt.8 q3, q3\n vcnt.8 q7, q7\n"
#define BF_A32_ROW2(OP)                                                                                   \
  OP " q12, q8, q10\n " OP " q13, q9, q10\n " OP " q14, q8, q11\n " OP " q15, q9, q11\n"                 \
  "vld1.8 {d20-d23}, [%[b]]!\n"                                                                          \
  "vcnt.8 q12, q12\n vcnt.8 q13, q13\n vcnt.8 q14, q14\n vcnt.8 q15, q15\n"                              \
  "vadd.i8 q0, q0, q12\n vadd.i8 q4, q4, q13\n vadd.i8 q1, q1, q14\n vadd.i8 q5, q5, q15\n"              \
  OP " q12, q8, q10\n " OP " q13, q9, q10\n " OP " q14, q8, q11\n " OP " q15, q9, q11\n"                 \
  "vcnt.8 q12, q12\n vcnt.8 q13, q13\n vcnt.8 q14, q14\n vcnt.8 q15, q15\n"                              \
  "vadd.i8 q2, q2, q12\n vadd.i8 q6, q6, q13\n vadd.i8 q3, q3, q14\n vadd.i8 q7, q7, q15\n"
#define BF_A32_LOAD2 "vld1.8 {d16-d17}, [%[a0]]!\n vld1.8 {d18-d19}, [%[a1]]!\n vld1.8 {d20-d23}, [%[b]]!\n"
#define BF_A32_STORE8 "vst1.8 {d0-d3}, [%[o]]!\n vst1.8 {d4-d7}, [%[o]]!\n vst1.8 {d8-d11}, [%[o]]!\n vst1.8 {d12-d15}, [%[o]]!\n"
#define BF_A32_CLOBBERS                                                                                   \
  "cc", "memory", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", \
  "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31"
#define BF_A32_RUN2(OP)                                                                                   \
  asm volatile(BF_A32_LOAD2 BF_A32_SET2(OP)                                                               \
               "subs %[n], %[n], #1\n"                                                                   \
               "beq 2f\n"                                                                                \
               "1:\n" BF_A32_LOAD2 BF_A32_ROW2(OP)                                                        \
               "subs %[n], %[n], #1\n"                                                                   \
               "bne 1b\n"                                                                                \
               "2:\n" BF_A32_STORE8                                                                       \
               : [a0] "+r"(a0), [a1] "+r"(a1), [b] "+r"(b), [n] "+r"(n), [o] "+r"(o)                      \
               :                                                                                          \
               : BF_A32_CLOBBERS)

template <bool kAnd>
inline __attribute__((always_inline)) void run_2x4_a32(const uint8_t* a0, const uint8_t* a1, const uint8_t* b, int n, uint8x16_t* out) {
  uint8_t* o = reinterpret_cast<uint8_t*>(out);
  if constexpr (kAnd) {
    BF_A32_RUN2("vand");
  } else {
    BF_A32_RUN2("veor");
  }
}
#undef BF_A32_SET2
#undef BF_A32_ROW2
#undef BF_A32_LOAD2
#undef BF_A32_RUN2

// One run of n >= 1 steps (n <= kU8Steps); advances the pointers by n steps.
template <bool kAnd = false>
inline __attribute__((always_inline)) void run_4x4(const uint8_t*& a0, const uint8_t*& a1, const uint8_t*& a2, const uint8_t*& a3,
                                                   const uint8_t*& b, int64_t n, Acc4x4& acc) {
  run_2x4_a32<kAnd>(a0, a1, b, static_cast<int>(n), &acc.c[0][0]);
  run_2x4_a32<kAnd>(a2, a3, b, static_cast<int>(n), &acc.c[2][0]);
  a0 += 16 * n;
  a1 += 16 * n;
  a2 += 16 * n;
  a3 += 16 * n;
  b += 64 * n;
}
#endif

// popcount(a_i XOR w_j) (kAnd: a_i AND w_j) for 4 rows a_i and one
// interleaved 4-row weight block b ([step][4][16 bytes]); out[i] lane j.
template <bool kAnd = false>
inline __attribute__((always_inline)) void popcount_4x4(const uint8_t* a0, const uint8_t* a1, const uint8_t* a2, const uint8_t* a3,
                                                        const uint8_t* b, int64_t steps, uint32x4_t out[4]) {
  if (steps <= kU8Steps) {  // one uint8 run: no uint32 accumulation pass
    Acc4x4 acc;
    run_4x4<kAnd>(a0, a1, a2, a3, b, steps, acc);
    for (int i = 0; i < 4; ++i) out[i] = reduce4_n(acc.c[i][0], acc.c[i][1], acc.c[i][2], acc.c[i][3], steps);
    return;
  }
  uint32x4_t t[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
  for (int64_t s = 0; s < steps; s += kU8Steps) {
    Acc4x4 acc;
    const int64_t n = std::min(kU8Steps, steps - s);
    run_4x4<kAnd>(a0, a1, a2, a3, b, n, acc);
    for (int i = 0; i < 4; ++i) t[i] = vaddq_u32(t[i], reduce4_n(acc.c[i][0], acc.c[i][1], acc.c[i][2], acc.c[i][3], n));
  }
  for (int i = 0; i < 4; ++i) out[i] = t[i];
}

// Pair mode: each 16-byte A register holds 64 bits of an even pixel (bytes
// 0-7) and 64 bits of the odd pixel (bytes 8-15); weights hold the same 64
// bits twice. Folds the 4 accumulators of one pixel pair after a run of n
// steps into [c0 even, c0 odd, c1 even, c1 odd, ..., c3 odd] (uint16; a pixel
// row has at most kMaxPairWords * 32 = 4096 bits, so sums over several runs
// also fit). Short runs pairwise-add in uint8 first as reduce4_n.
inline uint16x8_t reduce4_pair_n(uint8x16_t c0, uint8x16_t c1, uint8x16_t c2, uint8x16_t c3, int64_t n) {
  if (n <= 7) return vpaddlq_u8(vpaddq_u8(vpaddq_u8(c0, c1), vpaddq_u8(c2, c3)));
  if (n <= 15) return vpaddq_u16(vpaddlq_u8(vpaddq_u8(c0, c1)), vpaddlq_u8(vpaddq_u8(c2, c3)));
  const uint16x8_t t01 = vpaddq_u16(vpaddlq_u8(c0), vpaddlq_u8(c1));  // [c0 lo,lo,hi,hi | c1 lo,lo,hi,hi]
  const uint16x8_t t23 = vpaddq_u16(vpaddlq_u8(c2), vpaddlq_u8(c3));
  return vpaddq_u16(t01, t23);
}

// Per-pixel uint32 sums of the 8 pixels of a pair tile from the folded rows
// t[k] (pixels 2k, 2k+1): even lanes are the even pixel.
inline void pair_rows_out(const uint16x8_t t[4], uint32x4_t out[8]) {
  for (int k = 0; k < 4; k += 2) {
    const uint16x8_t e = vuzp1q_u16(t[k], t[k + 1]), o = vuzp2q_u16(t[k], t[k + 1]);  // [pixel 2k | pixel 2k + 2]
    out[2 * k] = vmovl_u16(vget_low_u16(e));
    out[2 * k + 1] = vmovl_u16(vget_low_u16(o));
    out[2 * k + 2] = vmovl_high_u16(e);
    out[2 * k + 3] = vmovl_high_u16(o);
  }
}

// 8 pixels (4 interleaved pair rows) x 4 output channels in pair mode.
inline __attribute__((always_inline)) void popcount_8x4(const uint8_t* a0, const uint8_t* a1, const uint8_t* a2, const uint8_t* a3,
                                                        const uint8_t* b, int64_t steps, uint32x4_t out[8]) {
  uint16x8_t t[4];
  if (steps <= kU8Steps) {
    Acc4x4 acc;
    run_4x4(a0, a1, a2, a3, b, steps, acc);
    for (int k = 0; k < 4; ++k) t[k] = reduce4_pair_n(acc.c[k][0], acc.c[k][1], acc.c[k][2], acc.c[k][3], steps);
  } else {
    for (int k = 0; k < 4; ++k) t[k] = vdupq_n_u16(0);
    for (int64_t s = 0; s < steps; s += kU8Steps) {
      Acc4x4 acc;
      const int64_t n = std::min(kU8Steps, steps - s);
      run_4x4(a0, a1, a2, a3, b, n, acc);
      for (int k = 0; k < 4; ++k) t[k] = vaddq_u16(t[k], reduce4_pair_n(acc.c[k][0], acc.c[k][1], acc.c[k][2], acc.c[k][3], n));
    }
  }
  pair_rows_out(t, out);
}

// BITINO_BCONV_RES=v1 selects the previous single-phase I8BConv2DRes epilogue (compute_res; tests and A/B).
inline bool opts_res2_off() {
  const char* e = std::getenv("BITINO_BCONV_RES");
  return e && std::string(e) == "v1";
}

// BITINO_BCONV_DOT2=0 keeps the single-phase epilogue (compute_tile_a) for plain int8 / fp32 output (tests and A/B).
inline bool opts_dot2_off() {
  const char* e = std::getenv("BITINO_BCONV_DOT2");
  return e && std::string(e) == "0";
}

// BITINO_BCONV_T48=0 keeps the 4x4 tile in the fp32 two-phase BConv2D (tests and A/B; see popcount_4x8_u16).
inline bool opts_t48_off() {
  const char* e = std::getenv("BITINO_BCONV_T48");
  return e && std::string(e) == "0";
}

// ---- uint16 popcount tiles (I8BConv2DRes two-phase epilogue, compute_res2). The sums of a tile row fit uint16
// (K <= 32767 bits). Per tile row the 4 accumulators fold into [c0 c0' c1 c1' c2 c2' c3 c3'] (two partial sums
// per channel), two rows then into [row r: c0..c3 | row r+1: c0..c3] with one more ADDP.
inline uint16x8_t fold4_u16_n(uint8x16_t c0, uint8x16_t c1, uint8x16_t c2, uint8x16_t c3, int64_t n) {
  if (n <= 7) return vpaddlq_u8(vpaddq_u8(vpaddq_u8(c0, c1), vpaddq_u8(c2, c3)));  // [c0 x2 | c1 x2 | ...] after paddl
  if (n <= 15) return vpaddq_u16(vpaddlq_u8(vpaddq_u8(c0, c1)), vpaddlq_u8(vpaddq_u8(c2, c3)));
  return vpaddq_u16(vpaddq_u16(vpaddlq_u8(c0), vpaddlq_u8(c1)), vpaddq_u16(vpaddlq_u8(c2), vpaddlq_u8(c3)));
}
// 4 rows x 4 channels; out[0] = rows 0, 1, out[1] = rows 2, 3 (4 channels each).
inline __attribute__((always_inline)) void popcount_4x4_u16(const uint8_t* a0, const uint8_t* a1, const uint8_t* a2, const uint8_t* a3,
                                                            const uint8_t* b, int64_t steps, uint16x8_t out[2]) {
  uint16x8_t t[4];
  if (steps <= kU8Steps) {
    Acc4x4 acc;
    run_4x4(a0, a1, a2, a3, b, steps, acc);
    for (int i = 0; i < 4; ++i) t[i] = fold4_u16_n(acc.c[i][0], acc.c[i][1], acc.c[i][2], acc.c[i][3], steps);
  } else {
    for (int i = 0; i < 4; ++i) t[i] = vdupq_n_u16(0);
    for (int64_t s = 0; s < steps; s += kU8Steps) {
      Acc4x4 acc;
      const int64_t n = std::min(kU8Steps, steps - s);
      run_4x4(a0, a1, a2, a3, b, n, acc);
      for (int i = 0; i < 4; ++i) t[i] = vaddq_u16(t[i], fold4_u16_n(acc.c[i][0], acc.c[i][1], acc.c[i][2], acc.c[i][3], n));
    }
  }
  out[0] = vpaddq_u16(t[0], t[1]);
  out[1] = vpaddq_u16(t[2], t[3]);
}
// Pair mode: 8 pixels (4 pair rows) x 4 channels; out[k] = pixels (2k, 2k+1) for k = 0..3 as [pixel | pixel + 1]...
// precisely out[0] = [p0 | p2], out[1] = [p1 | p3], out[2] = [p4 | p6], out[3] = [p5 | p7].
inline __attribute__((always_inline)) void popcount_8x4_u16(const uint8_t* a0, const uint8_t* a1, const uint8_t* a2, const uint8_t* a3,
                                                            const uint8_t* b, int64_t steps, uint16x8_t out[4]) {
  uint16x8_t t[4];
  if (steps <= kU8Steps) {
    Acc4x4 acc;
    run_4x4(a0, a1, a2, a3, b, steps, acc);
    for (int k = 0; k < 4; ++k) t[k] = reduce4_pair_n(acc.c[k][0], acc.c[k][1], acc.c[k][2], acc.c[k][3], steps);
  } else {
    for (int k = 0; k < 4; ++k) t[k] = vdupq_n_u16(0);
    for (int64_t s = 0; s < steps; s += kU8Steps) {
      Acc4x4 acc;
      const int64_t n = std::min(kU8Steps, steps - s);
      run_4x4(a0, a1, a2, a3, b, n, acc);
      for (int k = 0; k < 4; ++k) t[k] = vaddq_u16(t[k], reduce4_pair_n(acc.c[k][0], acc.c[k][1], acc.c[k][2], acc.c[k][3], n));
    }
  }
  // t[k] = [c0 even, c0 odd, c1 even, ...] of pair row k (pixels 2k, 2k+1)
  out[0] = vuzp1q_u16(t[0], t[1]);
  out[1] = vuzp2q_u16(t[0], t[1]);
  out[2] = vuzp1q_u16(t[2], t[3]);
  out[3] = vuzp2q_u16(t[2], t[3]);
}

#if defined(__aarch64__)
// Pair mode, 8 pixels (4 pair rows) x 4 channels with uint16 accumulators (fp32 two-phase BConv2D): per step 16 EOR +
// 16 CNT, per pair row two levels of ADDP (bytes [c0 e x2 | c0 o x2 | c1 e x2 | ... | c3 o x2]) and one UADALP into
// t[k] = [c0 even, c0 odd, c1 even, ..., c3 odd] (reduce4_pair_n's format): the instruction count of run_4x4 without
// the uint8 runs to fold, scheduled as popcount_4x8_u16 (all EORs, all CNTs, then the ADDP levels; the next step's
// rows are loaded after the CNTs, its weights after the first ADDPs; the last step loads nothing). Temporaries
// v0-v15, rows v16-v19, weights v20-v23; the accumulators get registers above v23.
#define BF_P84_STEP(LA0, LA1, LA2, LA3, LW)                                                                 \
  "eor v0.16b, v16.16b, v20.16b\n eor v1.16b, v16.16b, v21.16b\n eor v2.16b, v16.16b, v22.16b\n"          \
  "eor v3.16b, v16.16b, v23.16b\n eor v4.16b, v17.16b, v20.16b\n eor v5.16b, v17.16b, v21.16b\n"          \
  "eor v6.16b, v17.16b, v22.16b\n eor v7.16b, v17.16b, v23.16b\n eor v8.16b, v18.16b, v20.16b\n"          \
  "eor v9.16b, v18.16b, v21.16b\n eor v10.16b, v18.16b, v22.16b\n eor v11.16b, v18.16b, v23.16b\n"        \
  "eor v12.16b, v19.16b, v20.16b\n eor v13.16b, v19.16b, v21.16b\n eor v14.16b, v19.16b, v22.16b\n"       \
  "eor v15.16b, v19.16b, v23.16b\n"                                                                        \
  "cnt v0.16b, v0.16b\n cnt v1.16b, v1.16b\n cnt v2.16b, v2.16b\n cnt v3.16b, v3.16b\n" LA0                 \
  "cnt v4.16b, v4.16b\n cnt v5.16b, v5.16b\n cnt v6.16b, v6.16b\n cnt v7.16b, v7.16b\n" LA1                 \
  "cnt v8.16b, v8.16b\n cnt v9.16b, v9.16b\n cnt v10.16b, v10.16b\n cnt v11.16b, v11.16b\n" LA2             \
  "cnt v12.16b, v12.16b\n cnt v13.16b, v13.16b\n cnt v14.16b, v14.16b\n cnt v15.16b, v15.16b\n" LA3         \
  "addp v0.16b, v0.16b, v1.16b\n addp v2.16b, v2.16b, v3.16b\n addp v4.16b, v4.16b, v5.16b\n"             \
  "addp v6.16b, v6.16b, v7.16b\n" LW                                                                       \
  "addp v8.16b, v8.16b, v9.16b\n addp v10.16b, v10.16b, v11.16b\n addp v12.16b, v12.16b, v13.16b\n"       \
  "addp v14.16b, v14.16b, v15.16b\n"                                                                       \
  "addp v0.16b, v0.16b, v2.16b\n addp v4.16b, v4.16b, v6.16b\n addp v8.16b, v8.16b, v10.16b\n"            \
  "addp v12.16b, v12.16b, v14.16b\n"                                                                       \
  "uadalp %[t0].8h, v0.16b\n uadalp %[t1].8h, v4.16b\n uadalp %[t2].8h, v8.16b\n uadalp %[t3].8h, v12.16b\n"

inline __attribute__((always_inline)) void popcount_8x4_pair_u16(const uint8_t* a0, const uint8_t* a1, const uint8_t* a2,
                                                                 const uint8_t* a3, const uint8_t* b, int64_t steps, uint16x8_t t[4]) {
  uint16x8_t t0, t1, t2, t3;
  int64_t n = steps - 1;
  asm volatile(
      "movi %[t0].8h, #0\n movi %[t1].8h, #0\n movi %[t2].8h, #0\n movi %[t3].8h, #0\n"
      "ldr q16, [%[a0]], #16\n ldr q17, [%[a1]], #16\n ldr q18, [%[a2]], #16\n ldr q19, [%[a3]], #16\n"
      "ldp q20, q21, [%[b]]\n ldp q22, q23, [%[b], #32]\n add %[b], %[b], #64\n"
      "cbz %[n], 2f\n"
      "1:\n" BF_P84_STEP("ldr q16, [%[a0]], #16\n", "ldr q17, [%[a1]], #16\n", "ldr q18, [%[a2]], #16\n", "ldr q19, [%[a3]], #16\n",
                         "ldp q20, q21, [%[b]]\n ldp q22, q23, [%[b], #32]\n add %[b], %[b], #64\n")
      "subs %[n], %[n], #1\n"
      "b.ne 1b\n"
      "2:\n" BF_P84_STEP("", "", "", "", "")
      : [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3), [b] "+r"(b), [n] "+r"(n), [t0] "=&w"(t0), [t1] "=&w"(t1),
        [t2] "=&w"(t2), [t3] "=&w"(t3)
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
  t[0] = t0;
  t[1] = t1;
  t[2] = t2;
  t[3] = t3;
}
#undef BF_P84_STEP
#endif

// Single-row variant (BDense with one input row; kAnd: QBDense / QBConv2D rows).
template <bool kAnd = false>
inline uint32x4_t popcount_1x4(const uint8_t* a, const uint8_t* b, int64_t steps) {
  uint32x4_t t = vdupq_n_u32(0);
  for (int64_t s = 0; s < steps;) {
    const int64_t e = std::min(steps, s + kU8Steps);
    uint8x16_t c0 = vdupq_n_u8(0), c1 = c0, c2 = c0, c3 = c0;
    for (; s < e; ++s) {
      uint8x16_t x = vld1q_u8(a + 16 * s);
      uint8x16x4_t w = vld1q_u8_x4(b + 64 * s);
      c0 = vaddq_u8(c0, vcntq_u8(kAnd ? vandq_u8(x, w.val[0]) : veorq_u8(x, w.val[0])));
      c1 = vaddq_u8(c1, vcntq_u8(kAnd ? vandq_u8(x, w.val[1]) : veorq_u8(x, w.val[1])));
      c2 = vaddq_u8(c2, vcntq_u8(kAnd ? vandq_u8(x, w.val[2]) : veorq_u8(x, w.val[2])));
      c3 = vaddq_u8(c3, vcntq_u8(kAnd ? vandq_u8(x, w.val[3]) : veorq_u8(x, w.val[3])));
    }
    t = vaddq_u32(t, reduce4(c0, c1, c2, c3));
  }
  return t;
}

// Activation with exactly the scalar semantics of apply_activation()
// (including NaN and signed zero), so results stay bit-identical.
inline float32x4_t activate(Activation act, float32x4_t v) {
  const float32x4_t zero = vdupq_n_f32(0.f);
  switch (act) {
    case Activation::kNone: return v;
    case Activation::kRelu: return vbslq_f32(vcgtq_f32(v, zero), v, zero);  // v > 0 ? v : 0
    case Activation::kRelu6: {
      const float32x4_t six = vdupq_n_f32(6.f);
      float32x4_t m = vbslq_f32(vcltq_f32(v, zero), zero, v);  // std::max(v, 0)
      return vbslq_f32(vcltq_f32(six, m), six, m);              // std::min(m, 6)
    }
  }
  return v;
}

// Interleaves weight rows [rows][kwp words] (zero-padded to a multiple of 4
// rows) into [rows/4][kwp/4 steps][4 rows][4 words].
void interleave_rows4(const uint32_t* src, int64_t rows, int64_t row_words, int64_t src_stride, int64_t kwp, uint32_t* dst) {
  int64_t blocks = (rows + 3) / 4, steps = kwp / 4;
  for (int64_t b = 0; b < blocks; ++b)
    for (int64_t s = 0; s < steps; ++s)
      for (int j = 0; j < 4; ++j)
        for (int q = 0; q < 4; ++q) {
          int64_t r = b * 4 + j, k = s * 4 + q;
          dst[((b * steps + s) * 4 + j) * 4 + q] = (r < rows && k < row_words) ? src[r * src_stride + k] : 0u;
        }
}

// Pair-mode weights: [rows/4][kw2/2 half-steps][4 rows][w0 w1 w0 w1] (each
// 64-bit chunk duplicated for the even/odd pixel halves of the A register).
void interleave_rows4_dup(const uint32_t* src, int64_t rows, int64_t row_words, int64_t kw2, uint32_t* dst) {
  int64_t blocks = (rows + 3) / 4, steps = kw2 / 2;
  for (int64_t b = 0; b < blocks; ++b)
    for (int64_t s = 0; s < steps; ++s)
      for (int j = 0; j < 4; ++j)
        for (int q = 0; q < 4; ++q) {
          int64_t r = b * 4 + j, k = s * 2 + (q & 1);
          dst[((b * steps + s) * 4 + j) * 4 + q] = (r < rows && k < row_words) ? src[r * row_words + k] : 0u;
        }
}

// Packs a whole f32 tensor [pix][c] into bits [pix][bit_words(c)]. Packing
// runs at several floats per cycle, so inputs up to kPackSerialFloats are
// packed on the calling thread: waking the pool would cost more than it saves.
constexpr int64_t kPackSerialFloats = 128 * 1024;

void pack_input(const KernelContext& ctx, const float* x, int64_t pix, int64_t c, uint32_t* dst, BinarizeMode mode,
                const float* thr) {
  const int64_t words = bit_words(c);
  if (c % 32 == 0 && !thr) {
    // Contiguous: pack the tensor as one flat run of words.
    const int64_t total = pix * words;
    ctx.pool->parallel_for((total + 63) / 64, kPackSerialFloats / (64 * 32), [&](int64_t b0, int64_t b1) {
      int64_t w0 = b0 * 64, w1 = std::min(total, b1 * 64);
      pack_words_f32_neon(x + w0 * 32, w1 - w0, dst + w0, mode, nullptr);
    });
    return;
  }
  ctx.pool->parallel_for(pix, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(c, 1)), [&](int64_t p0, int64_t p1) {
    for (int64_t p = p0; p < p1; ++p) pack_row_f32_neon(x + p * c, c, dst + p * words, mode, thr);
  });
}

// ---------------------------------------------- int8_full I/O of the binary kernels
// The int8_full binary operators (kernels/int8_full/int8_full_neon.cc) run these kernels with int8 tensors:
//  * int8 output (out(0) of dtype i8, attributes i8_qmin / i8_qmax): every float output value v the kernel
//    computes (unchanged ops) is stored as clamp(rne(v), qmin, qmax) with FCVTNS and saturating narrows,
//    the conversion the int8_full wrapper applied to the float output (equal to q8::round_f32 for finite v).
//  * int8 input (bdense.ternary_neon only, in(0) of dtype i8, in(4) int32 [tge, tgt] scalar or
//    [tge[K], tgt[K]] per channel): the nz / neg planes of the integer sign_zero binarization
//    (neg = q < tge, nz = neg | q >= tgt) are packed from the codes.
struct I8Out {
  bool on = false;
  int lo = -128, hi = 127;
  void parse(const KernelContext& ctx) {
    on = ctx.out(0).dtype() == DType::kI8;
    lo = static_cast<int>(ctx.attrs().get_int("i8_qmin", -128));
    hi = static_cast<int>(ctx.attrs().get_int("i8_qmax", 127));
  }
};
// 4 values -> 4 int8 (low 32 bits)
inline uint32_t i8_pack4(float32x4_t v, int16x4_t lo, int16x4_t hi) {
  const int16x4_t h = vmin_s16(vmax_s16(vqmovn_s32(vcvtnq_s32_f32(v)), lo), hi);
  return vget_lane_u32(vreinterpret_u32_s8(vmovn_s16(vcombine_s16(h, h))), 0);
}
inline void i8_store4(int8_t* y, float32x4_t v, int16x4_t lo, int16x4_t hi, int nv) {
  const uint32_t b = i8_pack4(v, lo, hi);
  if (nv == 4) {
    std::memcpy(y, &b, 4);
  } else {
    for (int j = 0; j < nv; ++j) y[j] = static_cast<int8_t>(b >> (8 * j));
  }
}
// rows r0..r3 of 4 channels each (stride ys bytes): two saturating narrows per pair of rows
inline void i8_store4x4(int8_t* y, int64_t ys, float32x4_t v0, float32x4_t v1, float32x4_t v2, float32x4_t v3, int16x8_t lo, int16x8_t hi) {
  int16x8_t a = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(v0)), vqmovn_s32(vcvtnq_s32_f32(v1)));
  int16x8_t b = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(v2)), vqmovn_s32(vcvtnq_s32_f32(v3)));
  a = vminq_s16(vmaxq_s16(a, lo), hi);
  b = vminq_s16(vmaxq_s16(b, lo), hi);
  const uint32x4_t w = vreinterpretq_u32_s8(vcombine_s8(vmovn_s16(a), vmovn_s16(b)));
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(y), w, 0);
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(y + ys), w, 1);
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(y + 2 * ys), w, 2);
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(y + 3 * ys), w, 3);
}

// int8_full I8BConv2DRes (attribute i8_res; int8_ops._bconv_res, int8_full::ResEpilogue): per output value
//   u = v + R(res);  w = u >= 0 ? u : u * slope[o];  w += out_bias[o];  y = l8_encode(w);  bit = !(s >= bit_thr[o]) (ge), s = u (bit_src "sum") or w ("out")
// with v the kernel's unchanged fp32 value. in(5) residual codes (same layout as the output), in(6) slope, in(7)
// out_bias (optional), in(8) bit_thr (optional, then out(1) is the bit tensor). Groups == 1 only.
struct I8Res {
  bool on = false, l8 = true, bits = false, gt = false, has_bias = false, bit_out = false;  // bit_out: bit_src "out" (bits of w)
  bool has_lut = false;            // in(9) out_lut: the l8 codes are remapped by apply_out_lut after the tile
  alignas(16) uint8_t lut[256];    // out_lut[code + 128]
  std::vector<float> slope, bias, bthr;  // by channel, padded to oc_pad + 4
  float dec[256];
  int32_t rz = 0;
  float rs = 1.f;
  l8neon::DecTables dt;
  l8neon::EncTables et;
  // pre stage (int8_full::ResEpilogue::pre): v >= 0 ? v : v * pslope; v * pmul + padd (fp32 ops, separate); v * G[n, o].
  // Absent steps are identities (slope 1, mul 1, add +0, G 1), exact in fp32 up to the sign of a zero, which neither
  // the l8 code nor the bit compares see. Two-phase epilogue only (compute_res2).
  bool pre = false, gate = false;
  std::vector<float> pslope, pmul, padd;  // by channel, padded to oc_pad + 4
  std::vector<float> gval;                // G per sample [n][oc_pad + 4] (bind_gate, each run)
  float g_scale = 1.f;
  int32_t g_zero = 0;
  int64_t oc_row = 0;
  void bind_gate(const KernelContext& ctx, int64_t n, int64_t oc) {
    if (!gate) return;
    const int8_t* q = ctx.in(13).data<int8_t>();
    gval.assign(static_cast<size_t>(n * oc_row), 1.f);
    for (int64_t b = 0; b < n; ++b)
      for (int64_t o = 0; o < oc; ++o) gval[b * oc_row + o] = static_cast<float>(static_cast<int64_t>(q[b * oc + o]) - g_zero) * g_scale;
  }
  void parse(const KernelContext& ctx, int64_t oc, int64_t oc_pad) {
    on = ctx.attrs().get_int("i8_res", 0) != 0;
    if (!on) return;
    int8_full::ResEpilogue ep(ctx);
    pre = ep.pre_on();
    if (pre) {
      gate = ep.gate;
      g_scale = ep.g_scale;
      g_zero = ep.g_zero;
      oc_row = oc_pad + 4;
      pslope.assign(static_cast<size_t>(oc_pad + 4), 1.f);
      pmul.assign(static_cast<size_t>(oc_pad + 4), 1.f);
      padd.assign(static_cast<size_t>(oc_pad + 4), -0.f);  // -0: the exact additive identity
      if (ep.pre_slope) std::copy_n(ep.pre_slope, oc, pslope.data());
      if (ep.pre_mul) std::copy_n(ep.pre_mul, oc, pmul.data());
      if (ep.pre_add) std::copy_n(ep.pre_add, oc, padd.data());
    }
    l8 = ep.r_l8;
    rz = ep.r_zero;
    rs = ep.r_scale;
    std::copy_n(ep.dec, 256, dec);
    dt = l8neon::DecTables(ep.r_exp);
    has_bias = ep.bias != nullptr;
    has_lut = ep.out_lut != nullptr;
    if (has_lut) std::memcpy(lut, ep.out_lut, 256);
    bits = ctx.has_in(8);
    bit_out = ep.bit_out;
    gt = ctx.attrs().get_string("bit_binarize", "ge") == "gt";
    slope.assign(static_cast<size_t>(oc_pad + 4), 0.f);
    bias.assign(static_cast<size_t>(oc_pad + 4), 0.f);
    bthr.assign(static_cast<size_t>(oc_pad + 4), 0.f);
    std::copy_n(ep.slope, oc, slope.data());
    if (ep.bias) std::copy_n(ep.bias, oc, bias.data());
    if (bits) std::copy_n(ctx.in(8).data<float>(), oc, bthr.data());
    // u > t <=> u >= nextafter(t, +inf) for finite t (also t = -0 / +0): one compare form in the vector code
    if (bits && gt)
      for (int64_t i = 0; i < oc; ++i) bthr[i] = std::nextafter(bthr[i], std::numeric_limits<float>::infinity());
#if BF_NEON_A32
    // AArch32 NEON compares flush a denormal threshold to 0 (nextafter(0, +inf) for gt, or a denormal bit_thr). The
    // int8_full wrapper takes this kernel there only when every value the bits compare is a multiple of 2^-126
    // (int8_full_neon.cc, a32_res_exact), so a threshold t with 0 < |t| < 2^-126 is replaced by the grid value that gives
    // the same compares: s >= t <=> s >= 2^-126 for t > 0, s >= 0 for t < 0 (s = -0 included, as in IEEE).
    if (bits)
      for (int64_t i = 0; i < oc; ++i)
        if (std::fpclassify(bthr[i]) == FP_SUBNORMAL) bthr[i] = bthr[i] > 0.f ? std::numeric_limits<float>::min() : 0.f;
#endif
  }
  // the pre stage of one value (scalar; tails), sample b
  float pre1(float v, int64_t b, int64_t o) const {
    v = v >= 0.f ? v : v * pslope[o];
    v = v * pmul[o];
    v = v + padd[o];
    return gate ? v * gval[b * oc_row + o] : v;
  }
  // scalar form of one value (tails)
  int8_t finish(float v, int64_t o, int8_t r, bool* bit) const {
    const float u = v + dec[r + 128];
    float w = u >= 0.f ? u : u * slope[o];
    if (has_bias) w = w + bias[o];
    *bit = !((bit_out ? w : u) >= bthr[o]);
    return q8::l8_encode(w);
  }
};

// int8 sign_zero row -> nz / neg planes (words of 32 codes; int16 compares, so T = 128 needs no care); kPc: per-channel
// thresholds (a compile-time flag since 2026-11-13: the runtime test per 8 codes cost ~2x)
template <bool kPc>
void pack_ternary_row_i8_t(const int8_t* q, int64_t k, const int32_t* tge, const int32_t* tgt, uint32_t* nz, uint32_t* neg) {
  constexpr bool pc = kPc;
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t bw = vld1q_u8(kW);
  const int16x8_t sge = vdupq_n_s16(static_cast<int16_t>(tge[0])), sgt = vdupq_n_s16(static_cast<int16_t>(tgt[0]));
  auto thr8 = [&](const int32_t* t, int64_t i, int16x8_t s) {  // 8 per-channel thresholds as int16
    return pc ? vcombine_s16(vmovn_s32(vld1q_s32(t + i)), vmovn_s32(vld1q_s32(t + i + 4))) : s;
  };
  auto masks16 = [&](int64_t i, uint8x16_t* z, uint8x16_t* n) {
    const int8x16_t v = vld1q_s8(q + i);
    const int16x8_t l = vmovl_s8(vget_low_s8(v)), h = vmovl_high_s8(v);
    const uint8x16_t ng = vcombine_u8(vmovn_u16(vcltq_s16(l, thr8(tge, i, sge))), vmovn_u16(vcltq_s16(h, thr8(tge, i + 8, sge))));
    const uint8x16_t ps = vcombine_u8(vmovn_u16(vcgeq_s16(l, thr8(tgt, i, sgt))), vmovn_u16(vcgeq_s16(h, thr8(tgt, i + 8, sgt))));
    *z = vandq_u8(vorrq_u8(ng, ps), bw);
    *n = vandq_u8(ng, bw);
  };
  int64_t w = 0;
  for (; 32 * w + 32 <= k; ++w) {
    uint8x16_t z0, n0, z1, n1;
    masks16(32 * w, &z0, &n0);
    masks16(32 * w + 16, &z1, &n1);
    uint8x16_t r = vpaddq_u8(vpaddq_u8(z0, z1), vpaddq_u8(n0, n1));
    r = vpaddq_u8(r, r);  // u32 lanes: nz word, neg word
    nz[w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 0);
    neg[w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 1);
  }
  if (32 * w < k) {
    uint32_t bn = 0, bs = 0;
    for (int64_t b = 0; 32 * w + b < k; ++b) {
      const int64_t i = 32 * w + b;
      const int qv = q[i];
      const bool ng = qv < tge[pc ? i : 0], ps = qv >= tgt[pc ? i : 0];
      bn |= uint32_t(ng || ps) << b;
      bs |= uint32_t(ng) << b;
    }
    nz[w] = bn;
    neg[w] = bs;
  }
}
// One pair of thresholds, both in int8 range (the usual case): the compares in int8 lanes (16 codes per CMGT / CMGE).
void pack_ternary_row_i8_s8(const int8_t* q, int64_t k, int8_t tge, int8_t tgt, uint32_t* nz, uint32_t* neg) {
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t bw = vld1q_u8(kW);
  const int8x16_t vge = vdupq_n_s8(tge), vgt = vdupq_n_s8(tgt);
  int64_t w = 0;
  for (; 32 * w + 32 <= k; ++w) {
    const int8x16_t v0 = vld1q_s8(q + 32 * w), v1 = vld1q_s8(q + 32 * w + 16);
    const uint8x16_t n0 = vcltq_s8(v0, vge), n1 = vcltq_s8(v1, vge);
    const uint8x16_t z0 = vandq_u8(vorrq_u8(n0, vcgeq_s8(v0, vgt)), bw), z1 = vandq_u8(vorrq_u8(n1, vcgeq_s8(v1, vgt)), bw);
    uint8x16_t r = vpaddq_u8(vpaddq_u8(z0, z1), vpaddq_u8(vandq_u8(n0, bw), vandq_u8(n1, bw)));
    r = vpaddq_u8(r, r);  // u32 lanes: nz word, neg word
    nz[w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 0);
    neg[w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 1);
  }
  if (32 * w < k) {
    uint32_t bn = 0, bs = 0;
    for (int64_t b = 0; 32 * w + b < k; ++b) {
      const int qv = q[32 * w + b];
      const bool ng = qv < tge, ps = qv >= tgt;
      bn |= uint32_t(ng || ps) << b;
      bs |= uint32_t(ng) << b;
    }
    nz[w] = bn;
    neg[w] = bs;
  }
}
void pack_ternary_row_i8(const int8_t* q, int64_t k, const int32_t* tge, const int32_t* tgt, bool pc, uint32_t* nz, uint32_t* neg) {
  if (pc) pack_ternary_row_i8_t<true>(q, k, tge, tgt, nz, neg);
  else if (tge[0] <= 127 && tgt[0] <= 127) pack_ternary_row_i8_s8(q, k, static_cast<int8_t>(tge[0]), static_cast<int8_t>(tgt[0]), nz, neg);
  else pack_ternary_row_i8_t<false>(q, k, tge, tgt, nz, neg);
}

// Phase 1 of the two-phase epilogues (BConv2DNeonOpt compute_res2 / compute_dot2, BDenseNeon tile2): int16
// dot = K - 2 popcount of the tile rows [0, np) (pixels or Dense rows; pair rows in pair mode) and the channels
// [u, u + nu) (nu <= 16; weight blocks of 4 interleaved as [block][step][4][16 bytes]) into db[row][16].
// kLce (fp32 BConv2D, AArch64): pair rows on popcount_8x4_pair_u16 (same sums).
template <bool kLce>
inline __attribute__((always_inline)) void popcount_8x4_u16_sel(const uint8_t* a0, const uint8_t* a1, const uint8_t* a2,
                                                                const uint8_t* a3, const uint8_t* b, int64_t steps, uint16x8_t out[4]) {
#if defined(__aarch64__)
  if constexpr (kLce) {
    uint16x8_t t[4];
    popcount_8x4_pair_u16(a0, a1, a2, a3, b, steps, t);
    out[0] = vuzp1q_u16(t[0], t[1]);
    out[1] = vuzp2q_u16(t[0], t[1]);
    out[2] = vuzp1q_u16(t[2], t[3]);
    out[3] = vuzp2q_u16(t[2], t[3]);
    return;
  }
#endif
  popcount_8x4_u16(a0, a1, a2, a3, b, steps, out);
}

template <bool kPair, bool kLce = false>
inline __attribute__((always_inline)) void dots_unit_rows(const uint8_t* wts, const uint8_t* base, int64_t steps_, int32_t k_bits,
                                                          int64_t np, int64_t u, int64_t nu, int16_t* db) {
  constexpr int kTp = kPair ? 8 : 4;
  const int64_t rb = steps_ * 16;
  const int64_t last_row = kPair ? (np - 1) / 2 : np - 1;
  const int16x8_t kb = vdupq_n_s16(static_cast<int16_t>(k_bits));
  for (int64_t jb = 0; jb < nu; jb += 4) {
    const uint8_t* b = wts + ((u + jb) / 4) * steps_ * 64;
    int16_t* dc = db + jb;
    const uint8_t* a = base;
    int64_t i = 0;
    for (; i + kTp <= np; i += kTp, a += 4 * rb) {
      if constexpr (kPair) {
        uint16x8_t t[4];
        popcount_8x4_u16_sel<kLce>(a, a + rb, a + 2 * rb, a + 3 * rb, b, steps_, t);
        int16_t* d = dc + i * 16;
        for (int k = 0; k < 4; ++k) {  // t[k]: pixels (4 (k / 2) + k % 2, + 2)
          const int16x8_t v = vmlsq_n_s16(kb, vreinterpretq_s16_u16(t[k]), 2);
          const int64_t r0 = 4 * (k / 2) + (k % 2);
          vst1_s16(d + r0 * 16, vget_low_s16(v));
          vst1_s16(d + (r0 + 2) * 16, vget_high_s16(v));
        }
      } else {
        uint16x8_t t[2];
        popcount_4x4_u16(a, a + rb, a + 2 * rb, a + 3 * rb, b, steps_, t);
        int16_t* d = dc + i * 16;
        for (int k = 0; k < 2; ++k) {
          const int16x8_t v = vmlsq_n_s16(kb, vreinterpretq_s16_u16(t[k]), 2);
          vst1_s16(d + (2 * k) * 16, vget_low_s16(v));
          vst1_s16(d + (2 * k + 1) * 16, vget_high_s16(v));
        }
      }
    }
    if constexpr (kPair) {
      for (; i < np; i += kTp) {  // last partial tile: pair rows clamped, valid pixels stored
        const int64_t r0 = i / 2;
        const uint8_t* ar[4];
        for (int r = 0; r < 4; ++r) ar[r] = base + std::min(r0 + r, last_row) * rb;
        uint16x8_t t[4];
        popcount_8x4_u16_sel<kLce>(ar[0], ar[1], ar[2], ar[3], b, steps_, t);
        alignas(16) int16_t tmp[8][4];
        for (int k = 0; k < 4; ++k) {
          const int16x8_t v = vmlsq_n_s16(kb, vreinterpretq_s16_u16(t[k]), 2);
          const int r = 4 * (k / 2) + (k % 2);
          vst1_s16(tmp[r], vget_low_s16(v));
          vst1_s16(tmp[r + 2], vget_high_s16(v));
        }
        for (int64_t x = 0; x < std::min<int64_t>(kTp, np - i); ++x) std::memcpy(dc + (i + x) * 16, tmp[x], 8);
      }
    } else {
      for (; i < np; ++i) {  // 1-3 rows left: one row at a time
        const uint32x4_t pc = popcount_1x4(base + i * rb, b, steps_);
        const int16x4_t v = vmls_n_s16(vget_low_s16(kb), vreinterpret_s16_u16(vmovn_u32(pc)), 2);
        vst1_s16(dc + i * 16, v);
      }
    }
  }
}

#if defined(__aarch64__)
// ---- 4 rows x 8 channels tile with uint16 accumulators (fp32 two-phase BConv2D, rows padded to 128 bits). Per
// 128-bit step: 32 EOR + 32 CNT, the 32 byte vectors of a row summed by three levels of ADDP (row r: channels 0-7 as
// [w0 x2 | w1 x2 | ... | w7 x2] bytes) and one UADALP per row into uint16 [ch 0..7]: 96 vector instructions per
// 32 (row, channel) pairs, like the 4x4 tile's EOR / CNT / ADD, but 12 instead of 16 register loads, no uint8 runs
// to fold (the uint16 lanes hold sums up to K <= 32767 bits) and a schedule that keeps the Cortex-A75's two NEON
// pipes busier (A75 2.47 vs 2.67 cycles per (row, channel, 128 bits), X3 0.766 vs 0.750: ledger float_opt.md
// round 3). Weights interleaved by 8 channels: [block][step][8][16 bytes] (interleave_rows8). Software pipelined:
// the next step's rows are loaded once the EORs of this one are issued, its weights after their last use; the last
// step loads nothing (no read past the rows or the weight block). Temporaries v0-v15, rows v16-v19, weights
// v20-v27 (reused as temporaries in the second half); the accumulators get v28-v31.
#define BF_T48_HALF1                                                                                       \
  "eor v0.16b, v16.16b, v20.16b\n eor v1.16b, v17.16b, v20.16b\n eor v2.16b, v18.16b, v20.16b\n"        \
  "eor v3.16b, v19.16b, v20.16b\n eor v4.16b, v16.16b, v21.16b\n eor v5.16b, v17.16b, v21.16b\n"        \
  "eor v6.16b, v18.16b, v21.16b\n eor v7.16b, v19.16b, v21.16b\n eor v8.16b, v16.16b, v22.16b\n"        \
  "eor v9.16b, v17.16b, v22.16b\n eor v10.16b, v18.16b, v22.16b\n eor v11.16b, v19.16b, v22.16b\n"      \
  "eor v12.16b, v16.16b, v23.16b\n eor v13.16b, v17.16b, v23.16b\n eor v14.16b, v18.16b, v23.16b\n"     \
  "eor v15.16b, v19.16b, v23.16b\n"                                                                      \
  "cnt v0.16b, v0.16b\n cnt v1.16b, v1.16b\n cnt v2.16b, v2.16b\n cnt v3.16b, v3.16b\n"                  \
  "cnt v4.16b, v4.16b\n cnt v5.16b, v5.16b\n cnt v6.16b, v6.16b\n cnt v7.16b, v7.16b\n"                  \
  "cnt v8.16b, v8.16b\n cnt v9.16b, v9.16b\n cnt v10.16b, v10.16b\n cnt v11.16b, v11.16b\n"              \
  "cnt v12.16b, v12.16b\n cnt v13.16b, v13.16b\n cnt v14.16b, v14.16b\n cnt v15.16b, v15.16b\n"          \
  "addp v0.16b, v0.16b, v4.16b\n addp v1.16b, v1.16b, v5.16b\n addp v2.16b, v2.16b, v6.16b\n"           \
  "addp v3.16b, v3.16b, v7.16b\n addp v8.16b, v8.16b, v12.16b\n addp v9.16b, v9.16b, v13.16b\n"         \
  "addp v10.16b, v10.16b, v14.16b\n addp v11.16b, v11.16b, v15.16b\n"                                    \
  "eor v4.16b, v16.16b, v24.16b\n eor v5.16b, v17.16b, v24.16b\n eor v6.16b, v18.16b, v24.16b\n"        \
  "eor v7.16b, v19.16b, v24.16b\n eor v12.16b, v16.16b, v25.16b\n eor v13.16b, v17.16b, v25.16b\n"      \
  "eor v14.16b, v18.16b, v25.16b\n eor v15.16b, v19.16b, v25.16b\n eor v20.16b, v16.16b, v26.16b\n"     \
  "eor v21.16b, v17.16b, v26.16b\n eor v22.16b, v18.16b, v26.16b\n eor v23.16b, v19.16b, v26.16b\n"     \
  "eor v24.16b, v16.16b, v27.16b\n eor v25.16b, v17.16b, v27.16b\n eor v26.16b, v18.16b, v27.16b\n"     \
  "eor v27.16b, v19.16b, v27.16b\n"
#define BF_T48_HALF2(LA0, LA1, LA2, LA3, LW0, LW1, LW2, LW3)                                               \
  "cnt v4.16b, v4.16b\n cnt v5.16b, v5.16b\n cnt v6.16b, v6.16b\n cnt v7.16b, v7.16b\n" LA0               \
  "cnt v12.16b, v12.16b\n cnt v13.16b, v13.16b\n cnt v14.16b, v14.16b\n cnt v15.16b, v15.16b\n" LA1       \
  "cnt v20.16b, v20.16b\n cnt v21.16b, v21.16b\n cnt v22.16b, v22.16b\n cnt v23.16b, v23.16b\n" LA2       \
  "cnt v24.16b, v24.16b\n cnt v25.16b, v25.16b\n cnt v26.16b, v26.16b\n cnt v27.16b, v27.16b\n" LA3       \
  "addp v4.16b, v4.16b, v12.16b\n addp v5.16b, v5.16b, v13.16b\n addp v6.16b, v6.16b, v14.16b\n"        \
  "addp v7.16b, v7.16b, v15.16b\n addp v12.16b, v20.16b, v24.16b\n addp v13.16b, v21.16b, v25.16b\n"    \
  "addp v14.16b, v22.16b, v26.16b\n addp v15.16b, v23.16b, v27.16b\n" LW0                                \
  "addp v0.16b, v0.16b, v8.16b\n addp v1.16b, v1.16b, v9.16b\n" LW1                                      \
  "addp v2.16b, v2.16b, v10.16b\n addp v3.16b, v3.16b, v11.16b\n" LW2                                    \
  "addp v4.16b, v4.16b, v12.16b\n addp v5.16b, v5.16b, v13.16b\n" LW3                                    \
  "addp v6.16b, v6.16b, v14.16b\n addp v7.16b, v7.16b, v15.16b\n"                                        \
  "addp v0.16b, v0.16b, v4.16b\n addp v1.16b, v1.16b, v5.16b\n addp v2.16b, v2.16b, v6.16b\n"           \
  "addp v3.16b, v3.16b, v7.16b\n"                                                                        \
  "uadalp %[c0].8h, v0.16b\n uadalp %[c1].8h, v1.16b\n uadalp %[c2].8h, v2.16b\n uadalp %[c3].8h, v3.16b\n"

// popcount(a_r XOR w_j) summed over `steps` (>= 1) 128-bit steps for rows a0..a3 and the 8 interleaved channels at
// b: out[r] = channels 0-7 of row r (uint16).
inline __attribute__((always_inline)) void popcount_4x8_u16(const uint8_t* a0, const uint8_t* a1, const uint8_t* a2, const uint8_t* a3,
                                                            const uint8_t* b, int64_t steps, uint16x8_t out[4]) {
  uint16x8_t c0, c1, c2, c3;
  int64_t n = steps - 1;
  asm volatile(
      "movi %[c0].8h, #0\n movi %[c1].8h, #0\n movi %[c2].8h, #0\n movi %[c3].8h, #0\n"
      "ldr q16, [%[a0]], #16\n ldr q17, [%[a1]], #16\n ldr q18, [%[a2]], #16\n ldr q19, [%[a3]], #16\n"
      "ldp q20, q21, [%[b]]\n ldp q22, q23, [%[b], #32]\n ldp q24, q25, [%[b], #64]\n ldp q26, q27, [%[b], #96]\n"
      "add %[b], %[b], #128\n"
      "cbz %[n], 2f\n"
      "1:\n" BF_T48_HALF1 BF_T48_HALF2("ldr q16, [%[a0]], #16\n", "ldr q17, [%[a1]], #16\n", "ldr q18, [%[a2]], #16\n",
                                       "ldr q19, [%[a3]], #16\n", "ldp q20, q21, [%[b]]\n", "ldp q22, q23, [%[b], #32]\n",
                                       "ldp q24, q25, [%[b], #64]\n", "ldp q26, q27, [%[b], #96]\n add %[b], %[b], #128\n")
      "subs %[n], %[n], #1\n"
      "b.ne 1b\n"
      "2:\n" BF_T48_HALF1 BF_T48_HALF2("", "", "", "", "", "", "", "")
      : [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3), [b] "+r"(b), [n] "+r"(n), [c0] "=&w"(c0), [c1] "=&w"(c1),
        [c2] "=&w"(c2), [c3] "=&w"(c3)
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}
#undef BF_T48_HALF1
#undef BF_T48_HALF2

// One row x 8 channels of the same layout (the 1-3 rows after the last 4-row tile): channels 0-7 (uint16).
inline uint16x8_t popcount_1x8_u16(const uint8_t* a, const uint8_t* b, int64_t steps) {
  uint16x8_t acc = vdupq_n_u16(0);
  for (int64_t s = 0; s < steps; ++s, a += 16, b += 128) {
    const uint8x16_t x = vld1q_u8(a);
    const uint8x16x4_t w0 = vld1q_u8_x4(b), w1 = vld1q_u8_x4(b + 64);
    const uint8x16_t p0 = vpaddq_u8(vcntq_u8(veorq_u8(x, w0.val[0])), vcntq_u8(veorq_u8(x, w0.val[1])));
    const uint8x16_t p1 = vpaddq_u8(vcntq_u8(veorq_u8(x, w0.val[2])), vcntq_u8(veorq_u8(x, w0.val[3])));
    const uint8x16_t p2 = vpaddq_u8(vcntq_u8(veorq_u8(x, w1.val[0])), vcntq_u8(veorq_u8(x, w1.val[1])));
    const uint8x16_t p3 = vpaddq_u8(vcntq_u8(veorq_u8(x, w1.val[2])), vcntq_u8(veorq_u8(x, w1.val[3])));
    acc = vpadalq_u8(acc, vpaddq_u8(vpaddq_u8(p0, p1), vpaddq_u8(p2, p3)));
  }
  return acc;
}

// Interleaves weight rows [rows][row_words] (zero-padded to a multiple of 8 rows and to kwp words) into
// [rows/8][kwp/4 steps][8 rows][4 words] (popcount_4x8_u16).
void interleave_rows8(const uint32_t* src, int64_t rows, int64_t row_words, int64_t kwp, uint32_t* dst) {
  const int64_t blocks = (rows + 7) / 8, steps = kwp / 4;
  for (int64_t b = 0; b < blocks; ++b)
    for (int64_t s = 0; s < steps; ++s)
      for (int j = 0; j < 8; ++j)
        for (int q = 0; q < 4; ++q) {
          const int64_t r = b * 8 + j, k = s * 4 + q;
          dst[((b * steps + s) * 8 + j) * 4 + q] = (r < rows && k < row_words) ? src[r * row_words + k] : 0u;
        }
}

// Phase 1 of compute_dot2 on the 4x8 tile (rows padded to 128 bits, nu a multiple of 8): int16 dot = K - 2 popcount
// of the tile rows [0, np) and channels [u, u + nu) into db[row][16], as dots_unit_rows<false>. The 1-3 rows after
// the last 4-row tile run one at a time (popcount_1x8_u16).
inline void dots_unit_rows8(const uint8_t* wts8, const uint8_t* base, int64_t steps, int32_t k_bits, int64_t np, int64_t u, int64_t nu,
                            int16_t* db) {
  const int64_t rb = steps * 16;
  const int16x8_t kb = vdupq_n_s16(static_cast<int16_t>(k_bits));
  for (int64_t jb = 0; jb < nu; jb += 8) {
    const uint8_t* b = wts8 + ((u + jb) / 8) * steps * 128;
    int16_t* dc = db + jb;
    int64_t i = 0;
    for (const uint8_t* a = base; i + 4 <= np; i += 4, a += 4 * rb) {
      uint16x8_t t[4];
      popcount_4x8_u16(a, a + rb, a + 2 * rb, a + 3 * rb, b, steps, t);
      for (int r = 0; r < 4; ++r) vst1q_s16(dc + (i + r) * 16, vmlsq_n_s16(kb, vreinterpretq_s16_u16(t[r]), 2));
    }
    for (; i < np; ++i)
      vst1q_s16(dc + i * 16, vmlsq_n_s16(kb, vreinterpretq_s16_u16(popcount_1x8_u16(base + i * rb, b, steps)), 2));
  }
}
#endif

// Phase 2 of the plain two-phase epilogue (BConv2DNeonOpt compute_dot2, BDenseNeon tile2): rows [0, np) of db, 16
// channels with the constants mul[0..16), add[0..16): v = act(fp32(dot) * mul + add), stored as int8 (FCVTNS,
// saturating narrow, clamp to [lo, hi], i8_store4x4's operations) to y8 + r * ys or as fp32 to yf + r * ys.
template <bool kI8, Activation A>
inline void dot2_store16(const int16_t* db, int64_t np, const float* mulp, const float* addp, int lo, int hi, int8_t* y8, float* yf,
                         int64_t ys) {
  float32x4_t mul[4], add[4];
  for (int k = 0; k < 4; ++k) {
    mul[k] = vld1q_f32(mulp + 4 * k);
    add[k] = vld1q_f32(addp + 4 * k);
  }
  const int16x8_t lo8 = vdupq_n_s16(static_cast<int16_t>(lo)), hi8 = vdupq_n_s16(static_cast<int16_t>(hi));
  for (int64_t r = 0; r < np; ++r) {
    const int16x8_t d0 = vld1q_s16(db + r * 16), d1 = vld1q_s16(db + r * 16 + 8);
    const int32x4_t di[4] = {vmovl_s16(vget_low_s16(d0)), vmovl_high_s16(d0), vmovl_s16(vget_low_s16(d1)), vmovl_high_s16(d1)};
    float32x4_t v[4];
    for (int k = 0; k < 4; ++k) v[k] = activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(di[k]), mul[k]), add[k]));
    if constexpr (kI8) {
      int16x8_t a = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(v[0])), vqmovn_s32(vcvtnq_s32_f32(v[1])));
      int16x8_t b = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(v[2])), vqmovn_s32(vcvtnq_s32_f32(v[3])));
      a = vminq_s16(vmaxq_s16(a, lo8), hi8);
      b = vminq_s16(vmaxq_s16(b, lo8), hi8);
      vst1q_s8(y8 + r * ys, vcombine_s8(vmovn_s16(a), vmovn_s16(b)));
    } else {
      for (int k = 0; k < 4; ++k) vst1q_f32(yf + r * ys + 4 * k, v[k]);
    }
  }
}

// ------------------------------------------------------------------ BConv2D
// Grouped convolution: patch rows are built per (pixel tile, group) with the
// shared BConvGroupLayout (bconv_plan.h); output channels are padded to a
// multiple of 4 per group ("padded channel index" gi*ocg_pad_ + j), so every
// 4-channel block belongs to one group. groups == 1 is the special case
// ocg_ = oc.
class BConv2DNeonOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Json& a = ctx.attrs();
    g_ = conv_geometry(a, ctx.in(0).shape(), ctx.in(1).shape());
    L_.init(g_);
    act_ = parse_activation(a);
    mode_ = parse_binarize(a);
    pad_value_ = static_cast<int>(a.get_number("pad_value", 0.0));
    bout_.parse(ctx);
    bit_out_ = bout_.active;
    bit_in_ = ctx.in(0).dtype() == DType::kBit;
    taps_ = L_.taps;
    k_words_ = L_.k_words;
    // Pair mode when padding rows to 128 bits would waste 2-3 words per row
    // (e.g. ic = 32 or 64 with a 3x3 kernel): rows are then padded to 64 bits.
    pair_ = (k_words_ % 4 == 1 || k_words_ % 4 == 2) && k_words_ <= kMaxPairWords;
    kwp_ = pair_ ? (k_words_ + 1) / 2 * 2 : (k_words_ + 3) / 4 * 4;
    steps_ = pair_ ? kwp_ / 2 : kwp_ / 4;  // 16-byte register steps per tile row
    k_bits_ = static_cast<int32_t>(L_.k_bits);
    out_words_ = bit_words(g_.oc);
    ocg_ = L_.ocg;
    ocg_pad_ = (ocg_ + 3) / 4 * 4;
    oc_pad_ = g_.groups * ocg_pad_;
    pixels_ = g_.n * g_.oh * g_.ow;

    // Weights: dense im2col rows by padded channel index (zero rows in the
    // padding), interleaved by 4 channels.
    const uint32_t* w = ctx.in(1).data<uint32_t>();
    std::vector<uint32_t> dense(static_cast<size_t>(g_.oc * k_words_));
    L_.pack_weight_rows(w, g_.oc, dense.data(), k_words_);
    std::vector<uint32_t> padded(static_cast<size_t>(oc_pad_ * k_words_), 0u);
    for (int64_t gi = 0; gi < g_.groups; ++gi)
      std::copy_n(dense.data() + gi * ocg_ * k_words_, ocg_ * k_words_, padded.data() + gi * ocg_pad_ * k_words_);
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(oc_pad_ * steps_ * 16));
    if (pair_) interleave_rows4_dup(padded.data(), oc_pad_, k_words_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    else interleave_rows4(padded.data(), oc_pad_, k_words_, k_words_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    // Per-tap channel sums, transposed to [tap][oc_pad] for vector loads.
    tap_sum_.assign(static_cast<size_t>(taps_ * oc_pad_), 0);
    for (int64_t o = 0; o < g_.oc; ++o)
      for (int64_t t = 0; t < taps_; ++t) tap_sum_[t * oc_pad_ + padded_index(o)] = L_.tap_sum(w, o, t);
    mul_.assign(static_cast<size_t>(oc_pad_), 1.f);
    add_.assign(static_cast<size_t>(oc_pad_), -0.f);  // -0 is the exact additive identity

    if (!bit_in_ || L_.regroup) xbits_ = aligned_alloc_shared(static_cast<size_t>(g_.n * g_.ih * g_.iw * L_.src_words * 4));

    // Pixel tile: about 16 KB of patch rows, a multiple of the micro-tile.
    const int64_t tp = pair_ ? 8 : 4;
    mt_ = std::max<int64_t>(tp, std::min<int64_t>(64, (16384 / (kwp_ * 4)) / tp * tp));
    n_ptiles_ = (pixels_ + mt_ - 1) / mt_;
    const int64_t threads = ctx.pool->num_threads();
    plan_chunks(threads);
    items_ = n_ptiles_ * static_cast<int64_t>(chunks_.size());
    n_tasks_ = threads > 1 ? std::min<int64_t>(items_, ctx.pool->chunks_per_thread() * threads) : 1;
    slot_words_ = mt_ * kwp_;
    patches_ = aligned_alloc_shared(static_cast<size_t>(n_tasks_ * slot_words_ * 4));
    masks_.assign(static_cast<size_t>(n_tasks_ * 2 * mt_), 0);
    epi_.prepare_epilogue(ctx, g_.oh * g_.ow);
    prepare_bit_output();
    i8_out_.parse(ctx);
    BF_CHECK(!i8_out_.on || (!bit_out_ && !epi_.active()), "bconv2d.neon_opt: int8 output without bit output / epilogue only");
    res_.parse(ctx, g_.oc, oc_pad_);
    BF_CHECK(!res_.on || (i8_out_.on && g_.groups == 1), "bconv2d.neon_opt: residual epilogue needs int8 output, groups 1");
    // Residual epilogue in two phases (compute_res2): int16 dot products of a 16-channel unit, then the epilogue per
    // pixel over 16 contiguous channels. |dot| <= K must fit int16.
    res2_ = res_.on && k_bits_ <= 32767 && (res_.pre || !opts_res2_off());
    BF_CHECK(!res_.pre || res2_, "bconv2d.neon_opt: residual pre stage needs the two-phase epilogue (K <= 32767)");
    // Plain int8 / fp32 output in two phases as well (compute_dot2; BITINO_BCONV_DOT2=0 keeps compute_tile_a).
    dot2_ = !res_.on && g_.groups == 1 && k_bits_ <= 32767 && (i8_out_.on || !direct_bits_) && !opts_dot2_off();
    // fp32 output with a chain epilogue: the epilogue runs on each 16-channel unit in registers (EltwiseProgram::store_unit16)
    unit16_ = dot2_ && !i8_out_.on && epi_.active() && g_.oc % 16 == 0 && epi_.unit16();
    if (res2_ || dot2_) dots_ = aligned_alloc_shared(static_cast<size_t>(n_tasks_ * mt_ * 16 * 2));
#if defined(__aarch64__)
    // fp32 two-phase output, rows of whole 128-bit steps: the 4x8 tile (dots_unit_rows8), weights by 8 channels
    // (same size: oc_pad_ == oc). BITINO_BCONV_T48=0 keeps the 4x4 tile (tests and A/B).
    // (int8_full, 2026-11-13: int8 output and the I8BConv2DRes two-phase epilogue as well; the same int16 dots)
    t48_ = (dot2_ || res2_) && !pair_ && g_.groups == 1 && g_.oc % 8 == 0 && !opts_t48_off();
    // Pair mode: popcount_8x4_pair_u16 on the same weights, on a Cortex-A75 prime core only (A75 -5..-8 %, X3 +2..4 %
    // per layer: the X3's four NEON pipes run run_4x4's interleaved form faster; ledger float_opt.md round 3).
    // BITINO_BCONV_T48=0 also keeps run_4x4 there.
    p84_ = (dot2_ || res2_) && pair_ && cpu_info().prime_part == kPartCortexA75 && !opts_t48_off();
    if (t48_) {
      wbuf_ = aligned_alloc_shared(static_cast<size_t>(oc_pad_ * steps_ * 16));
      interleave_rows8(padded.data(), oc_pad_, k_words_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    }
#endif
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(oc_pad_ * steps_ * 16 + tap_sum_.size() * 4); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    if (res_.gate) res_.bind_gate(ctx, g_.n, g_.oc);
    const uint32_t* xb;
    if (bit_in_ && !L_.regroup) {
      xb = ctx.in(0).data<uint32_t>();
    } else {
      auto* dst = static_cast<uint32_t*>(xbits_.get());
      const float* thr = opt_f32(ctx, 4);
      const int64_t pix = g_.n * g_.ih * g_.iw;
      if (!L_.regroup) {
        pack_input(ctx, ctx.in(0).data<float>(), pix, g_.ic, dst, mode_, thr);
      } else {
        const BConvGroupLayout& lay = L_;
        const int64_t ic = g_.ic, sw = L_.src_words;
        const BinarizeMode mode = mode_;
        const uint32_t* xin = bit_in_ ? ctx.in(0).data<uint32_t>() : nullptr;
        const float* x = bit_in_ ? nullptr : ctx.in(0).data<float>();
        ctx.pool->parallel_for(pix, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(ic, 1)), [&](int64_t p0, int64_t p1) {
          for (int64_t p = p0; p < p1; ++p) {
            if (xin) {
              regroup_bit_row(xin + p * bit_words(ic), lay.groups, lay.kic, dst + p * sw);
            } else {
              for (int64_t gi = 0; gi < lay.groups; ++gi)
                pack_row_f32_neon(x + p * ic + gi * lay.kic, lay.kic, dst + p * sw + gi * lay.ut, mode, thr ? thr + gi * lay.kic : nullptr);
            }
          }
        });
      }
      xb = dst;
    }
    const float* m = opt_f32(ctx, 2);
    const float* d = opt_f32(ctx, 3);
    for (int64_t gi = 0; gi < g_.groups; ++gi) {
      if (m) std::copy_n(m + gi * ocg_, ocg_, mul_.data() + gi * ocg_pad_);
      if (d) std::copy_n(d + gi * ocg_, ocg_, add_.data() + gi * ocg_pad_);
    }
    if (direct_bits_) load_output_threshold(ctx);
    if (epi_.active()) epi_.begin(ctx);
    ctx.pool->parallel_for(n_tasks_, 1, [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) run_task(ctx, xb, t);
    });
    // Fused epilogue (and bit output after it): per pixel tile inside the tasks with one channel chunk, else after them.
    if (epi_.active() && chunks_.size() > 1 && (!unit16_ || bit_out_)) {
      float* y = float_out(ctx);
      ctx.pool->parallel_for(n_ptiles_, 1, [&](int64_t b0, int64_t b1) {
        const int64_t q0 = b0 * mt_, q1 = std::min(pixels_, b1 * mt_);
        if (!unit16_) epi_.apply(y, q0, q1);
        if (bit_out_) pack_float_rows(ctx, q0, q1);
      });
    }
  }

 private:
  // ---- bit output (kernels/bit_output.h). Without an epilogue the tile
  // epilogue binarizes each 4-channel block against the output threshold
  // (direct_bits_); with one, the float rows go to yf_, the epilogue runs on
  // them and pack_float_rows() binarizes the finished rows.
  void prepare_bit_output() {
    direct_bits_ = bit_out_ && !epi_.active();
    bit_rmw_ = direct_bits_ && g_.groups > 1 && ocg_ % 4 != 0;
    if (direct_bits_) {
      thr_.assign(static_cast<size_t>(oc_pad_), 0.f);
      lim_.assign(static_cast<size_t>(oc_pad_), INT32_MIN);  // padded channels: bit 0
      neg_.assign(static_cast<size_t>(oc_pad_), 0);
    }
    if (bit_out_ && !direct_bits_) yf_ = aligned_alloc_shared(static_cast<size_t>(pixels_ * g_.oc * 4));
  }
  // Per run (direct bit output): the output threshold by padded channel index, and the
  // integer thresholds on the dot product when they represent the float epilogue exactly.
  void load_output_threshold(const KernelContext& ctx) {
    const float* t = bout_.threshold(ctx);
    if (t)
      for (int64_t gi = 0; gi < g_.groups; ++gi) std::copy_n(t + gi * ocg_, ocg_, thr_.data() + gi * ocg_pad_);
    int_ok_ = ithr_.update(k_bits_, g_.oc, opt_f32(ctx, 2), opt_f32(ctx, 3), act_, bout_.mode, t);
    if (int_ok_ && ithr_.generation() != ithr_gen_)
      for (int64_t gi = 0; gi < g_.groups; ++gi) {
        std::copy_n(ithr_.lim() + gi * ocg_, ocg_, lim_.data() + gi * ocg_pad_);
        std::copy_n(ithr_.neg() + gi * ocg_, ocg_, neg_.data() + gi * ocg_pad_);
      }
    ithr_gen_ = ithr_.generation();
  }
  // Where the float output goes: the output tensor, or yf_ (bit output after an epilogue).
  float* float_out(const KernelContext& ctx) const {
    return bit_out_ ? static_cast<float*>(yf_.get()) : ctx.out(0).data<float>();
  }
  void pack_float_rows(const KernelContext& ctx, int64_t q0, int64_t q1) const {
    const float* y = static_cast<const float*>(yf_.get());
    const float* thr = bout_.threshold(ctx);
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    for (int64_t q = q0; q < q1; ++q) pack_row_f32_neon(y + q * g_.oc, g_.oc, yb + q * out_words_, bout_.mode, thr);
  }

  int64_t padded_index(int64_t o) const { return (o / ocg_) * ocg_pad_ + o % ocg_; }

  // Output-channel chunks [c0, c1), split only when pixel tiles alone cannot
  // feed ~4 tasks per thread (e.g. 7x7 layers). Boundaries are 4-channel
  // blocks of one group, and multiples of 32 channels for bit output (and for
  // groups == 1, as before groups support) so that no two tasks write the
  // same output word.
  void plan_chunks(int64_t threads) {
    chunks_.clear();
    int64_t want = 1;
    if (threads > 1 && n_ptiles_ < 4 * threads) want = std::min((4 * threads + n_ptiles_ - 1) / n_ptiles_, (g_.oc + 31) / 32);
    if (want <= 1) {
      chunks_.push_back({0, g_.oc});
      return;
    }
    const int64_t target = (g_.oc + want - 1) / want;
    const bool word_aligned = bit_out_ || g_.groups == 1;
    int64_t c0 = 0;
    for (int64_t c = 1; c < g_.oc; ++c) {
      const bool boundary = (c % ocg_) % 4 == 0 && (!word_aligned || c % 32 == 0);
      if (boundary && c - c0 >= target) {
        chunks_.push_back({c0, c});
        c0 = c;
      }
    }
    chunks_.push_back({c0, g_.oc});
  }

  // Builds the patch rows of pixels [p0, p1) for group gi into `rows` and
  // their border masks. Normal mode: one row of kwp_ words per pixel; the
  // bytes beyond the layout's k_units units are never written and keep their
  // initial zeros. Pair mode: pixels 2k and 2k+1 share an interleaved row of
  // 2*kwp_ words where every 16 bytes hold 64 bits of each.
  // tile_any (mask + mt_): per micro-tile of 4 (pair mode 8) pixels, the OR of
  // their masks, left 0 when the border needs no correction (pad_value 1), so
  // compute_tile_a tests the border once per tile.
  void build_patches(const uint32_t* xb, int64_t gi, int64_t p0, int64_t p1, uint32_t* rows, uint64_t* mask) const {
    int64_t ox = p0 % g_.ow, oy = (p0 / g_.ow) % g_.oh, n = p0 / (g_.ow * g_.oh);
    const auto* src = reinterpret_cast<const uint8_t*>(xb);
    uint64_t* tile_any = mask + mt_;
    const int tile_shift = pair_ ? 3 : 2;
    const bool corr = pad_value_ != 1;
    std::fill_n(tile_any, ((p1 - p0) >> tile_shift) + 1, uint64_t(0));
    alignas(16) uint32_t tmp[2][kMaxPairWords];  // pair mode: even / odd pixel of the pair
    const int64_t chunks = kwp_ / 2;               // pair mode: 64-bit chunks per pixel
    if (pair_) {
      // Padding of the last 64-bit chunk (odd k_words_) and the upper half of
      // the last word (16-bit units, odd unit count) are never written.
      for (auto& t : tmp) t[kwp_ - 1] = t[k_words_ - 1] = 0;
    }
    // Pair mode, 3x3 kernel, 64-bit taps (groups 1, ic 33..64), stride 1 or 2, both pixels of the pair interior and
    // in one output row: chunk h of the pair row is (tap h of the even pixel, tap h of the odd one), and per kernel
    // row these are 8-byte input pixels: stride 1 (in[0..3]): (0,1) (1,2) (2,3); stride 2 (in[0..4]): (0,2) (1,3) (2,4).
    const bool pair3 = pair_ && L_.ut * L_.unit_bytes == 8 && L_.src_pixel_bytes() == 8 && g_.groups == 1 && g_.kh == 3 &&
                       g_.kw == 3 && g_.dh == 1 && g_.dw == 1 && (g_.sw == 1 || g_.sw == 2) && kwp_ == 18;
    for (int64_t p = p0; p < p1; ++p) {
      const int64_t r = p - p0;
      if (pair3 && !(r & 1) && p + 1 < p1 && ox + 1 < g_.ow) {
        const int64_t iy0 = oy * g_.sh - g_.pt, ix0 = ox * g_.sw - g_.pl;
        if (iy0 >= 0 && iy0 + 2 < g_.ih && ix0 >= 0 && ix0 + g_.sw + 2 < g_.iw) {
          const uint8_t* s = src + ((n * g_.ih + iy0) * g_.iw + ix0) * 8;
          uint8_t* d = reinterpret_cast<uint8_t*>(rows + (r >> 1) * 2 * kwp_);
          const int64_t rsb = g_.iw * 8;
          for (int ky = 0; ky < 3; ++ky, s += rsb, d += 48) {
            if (g_.sw == 1) {
              const uint8x16_t a = vld1q_u8(s), b = vld1q_u8(s + 16);
              vst1q_u8(d, a);
              vst1q_u8(d + 16, vextq_u8(a, b, 8));
              vst1q_u8(d + 32, b);
            } else {
              const uint64x2_t a = vreinterpretq_u64_u8(vld1q_u8(s)), b = vreinterpretq_u64_u8(vld1q_u8(s + 16));
              vst1q_u64(reinterpret_cast<uint64_t*>(d), vzip1q_u64(a, b));
              vst1q_u64(reinterpret_cast<uint64_t*>(d + 16), vzip2q_u64(a, b));
              vst1q_u8(d + 32, vcombine_u8(vget_low_u8(vreinterpretq_u8_u64(b)), vld1_u8(s + 32)));
            }
          }
          mask[r] = mask[r + 1] = 0;
          ++p;
          ox += 2;
          if (ox == g_.ow) {
            ox = 0;
            if (++oy == g_.oh) {
              oy = 0;
              ++n;
            }
          }
          continue;
        }
      }
      if (!pair_) {
        mask[r] = build_patch_row(g_, L_, src, gi, n, oy, ox, reinterpret_cast<uint8_t*>(rows + r * kwp_));
      } else {
        mask[r] = build_patch_row(g_, L_, src, gi, n, oy, ox, reinterpret_cast<uint8_t*>(tmp[r & 1]));
        uint32_t* pr = rows + (r >> 1) * 2 * kwp_;
        if (r & 1) {  // interleave the pair: 16 bytes = 64 bits of the even pixel, then of the odd one
          int64_t h = 0;
          for (; h + 2 <= chunks; h += 2) {
            const uint64x2_t e = vreinterpretq_u64_u32(vld1q_u32(tmp[0] + 2 * h));
            const uint64x2_t od = vreinterpretq_u64_u32(vld1q_u32(tmp[1] + 2 * h));
            vst1q_u64(reinterpret_cast<uint64_t*>(pr + 4 * h), vzip1q_u64(e, od));
            vst1q_u64(reinterpret_cast<uint64_t*>(pr + 4 * h + 4), vzip2q_u64(e, od));
          }
          if (h < chunks) vst1q_u32(pr + 4 * h, vcombine_u32(vld1_u32(tmp[0] + 2 * h), vld1_u32(tmp[1] + 2 * h)));
        } else if (p + 1 == p1) {  // unpaired last pixel: its halves only (the odd outputs are dropped)
          for (int64_t h = 0; h < chunks; ++h) vst1_u32(pr + 4 * h, vld1_u32(tmp[0] + 2 * h));
        }
      }
      if (corr) tile_any[r >> tile_shift] |= mask[r];
      if (++ox == g_.ow) {
        ox = 0;
        if (++oy == g_.oh) {
          oy = 0;
          ++n;
        }
      }
    }
  }

  // 64-byte aligned: the compiled tile loops inlined here (grouped / int8 / bit-output paths) are sensitive to their
  // placement (larqface-binaryfacenet's depthwise layers +2-3 % on the X3 when this function moved by 56 bytes mod 64)
  __attribute__((aligned(64))) void run_task(const KernelContext& ctx, const uint32_t* xb, int64_t task) {
    const int64_t nch = static_cast<int64_t>(chunks_.size());
    const int64_t i0 = task * items_ / n_tasks_, i1 = (task + 1) * items_ / n_tasks_;
    uint32_t* rows = static_cast<uint32_t*>(patches_.get()) + task * slot_words_;
    uint64_t* mask = masks_.data() + task * 2 * mt_;  // pixel masks, then tile_any (build_patches)
    uint32_t* yb = bit_out_ ? ctx.out(0).data<uint32_t>() : nullptr;
    int64_t cur_pt = -1, cur_g = -1;
    for (int64_t it = i0; it < i1; ++it) {
      const int64_t pt = it / nch;
      const Chunk ch = chunks_[it % nch];
      const int64_t p0 = pt * mt_, p1 = std::min(pixels_, p0 + mt_);
      if (bit_rmw_) {
        const int64_t w0 = ch.c0 >> 5, w1 = (ch.c1 + 31) >> 5;
        for (int64_t q = p0; q < p1; ++q) std::fill(yb + q * out_words_ + w0, yb + q * out_words_ + w1, 0u);
      }
      for (int64_t gi = ch.c0 / ocg_; gi * ocg_ < ch.c1; ++gi) {
        if (pt != cur_pt || gi != cur_g) {
          build_patches(xb, gi, p0, p1, rows, mask);
          cur_pt = pt;
          cur_g = gi;
        }
        const int64_t j0 = std::max<int64_t>(0, ch.c0 - gi * ocg_);
        const int64_t j1 = std::min(ocg_, ch.c1 - gi * ocg_);
        if (res2_) {
          int16_t* db = static_cast<int16_t*>(dots_.get()) + task * mt_ * 16;
          if (pair_) compute_res2<true>(ctx, rows, mask, p0, p1, j0, j1, db);
          else compute_res2<false>(ctx, rows, mask, p0, p1, j0, j1, db);
          if (res_.has_lut) apply_out_lut(ctx, p0, p1, j0, j1);
        } else if (res_.on) {
          compute_res(ctx, rows, mask, p0, p1, gi, j0, j1);
          if (res_.has_lut) apply_out_lut(ctx, p0, p1, j0, j1);
        } else if (dot2_) {
          int16_t* db = static_cast<int16_t*>(dots_.get()) + task * mt_ * 16;
          if (i8_out_.on) {
            if (pair_) compute_dot2_t<true, kOutI8>(ctx, rows, mask, p0, p1, j0, j1, db);
            else compute_dot2_t<false, kOutI8>(ctx, rows, mask, p0, p1, j0, j1, db);
          } else {
            if (pair_) compute_dot2_t<true, kOutF32>(ctx, rows, mask, p0, p1, j0, j1, db);
            else compute_dot2_t<false, kOutF32>(ctx, rows, mask, p0, p1, j0, j1, db);
          }
        } else if (i8_out_.on) {
          if (pair_) compute_tile_t<true, kOutI8>(ctx, rows, mask, p0, p1, gi, j0, j1);
          else compute_tile_t<false, kOutI8>(ctx, rows, mask, p0, p1, gi, j0, j1);
        } else if (!direct_bits_) {
          if (pair_) compute_tile_t<true, kOutF32>(ctx, rows, mask, p0, p1, gi, j0, j1);
          else compute_tile_t<false, kOutF32>(ctx, rows, mask, p0, p1, gi, j0, j1);
        } else if (int_ok_) {
          if (pair_) compute_tile_t<true, kOutBitsInt>(ctx, rows, mask, p0, p1, gi, j0, j1);
          else compute_tile_t<false, kOutBitsInt>(ctx, rows, mask, p0, p1, gi, j0, j1);
        } else {
          if (pair_) compute_tile_t<true, kOutBitsFloat>(ctx, rows, mask, p0, p1, gi, j0, j1);
          else compute_tile_t<false, kOutBitsFloat>(ctx, rows, mask, p0, p1, gi, j0, j1);
        }
      }
      if (nch == 1 && epi_.active()) {  // rows still in cache
        if (!unit16_) epi_.apply(float_out(ctx), p0, p1);
        if (bit_out_) pack_float_rows(ctx, p0, p1);
      }
    }
  }

  // I8BConv2DRes tiles, out of line so that the other output modes' code in run_task is unaffected.
  __attribute__((noinline)) void compute_res(const KernelContext& ctx, const uint32_t* rows, const uint64_t* mask, int64_t p0, int64_t p1,
                                             int64_t gi, int64_t j0, int64_t j1) const {
    if (pair_) compute_tile_a<true, kOutI8Res, Activation::kNone>(ctx, rows, mask, p0, p1, gi, j0, j1);
    else compute_tile_a<false, kOutI8Res, Activation::kNone>(ctx, rows, mask, p0, p1, gi, j0, j1);
  }

  // out_lut (an I8Lut absorbed by Session): codes of pixels [p0, p1), channels [j0, j1) -> lut[code + 128] in place,
  // 16 at a time with TBL / TBX over the four 64-byte quarters (the I8Lut kernel's lookup), scalar tail.
  __attribute__((noinline)) void apply_out_lut(const KernelContext& ctx, int64_t p0, int64_t p1, int64_t j0, int64_t j1) const {
    const uint8_t* tab = res_.lut;
    const uint8x16x4_t t0 = vld1q_u8_x4(tab), t1 = vld1q_u8_x4(tab + 64), t2 = vld1q_u8_x4(tab + 128), t3 = vld1q_u8_x4(tab + 192);
    const uint8x16_t k64 = vdupq_n_u8(64), k80 = vdupq_n_u8(0x80), k128 = vdupq_n_u8(128), k192 = vdupq_n_u8(192);
    const int64_t oc = g_.oc;
    uint8_t* y = reinterpret_cast<uint8_t*>(ctx.out(0).data<int8_t>());
    for (int64_t q = p0; q < p1; ++q) {
      uint8_t* row = y + q * oc;
      int64_t j = j0;
      for (; j + 16 <= j1; j += 16) {
        const uint8x16_t idx = veorq_u8(vld1q_u8(row + j), k80);
        uint8x16_t r = vqtbl4q_u8(t0, idx);
        r = vqtbx4q_u8(r, t1, vsubq_u8(idx, k64));
        r = vqtbx4q_u8(r, t2, vsubq_u8(idx, k128));
        r = vqtbx4q_u8(r, t3, vsubq_u8(idx, k192));
        vst1q_u8(row + j, r);
      }
      for (; j < j1; ++j) row[j] = tab[static_cast<uint8_t>(row[j] ^ 0x80)];
    }
  }

  // ---- I8BConv2DRes, two phases per 16-channel unit [u, u + 16) of channels [j0, j1) (groups == 1):
  //  1. popcount tiles of the unit's 4 channel blocks -> int16 dot = K - 2 popcount, db[pixel][16];
  //  2. per pixel, the epilogue of compute_tile_a's kOutI8Res on 16 contiguous channels: one 16-byte residual
  //     load, one 16-byte l8 store and the unit's 16 bits as one 16-bit store (no read-modify-write of bit words),
  //     with the unit's constants loaded once per tile instead of once per 4x4 micro-tile.
  // Same values per element as compute_res: fp32(dot (+ border correction)) * mul + add, then I8Res. A last unit of
  // fewer than 16 channels runs the scalar epilogue (I8Res::finish). Bits of a word beyond oc are written 0.
  template <bool kPair>
  __attribute__((noinline)) void compute_res2(const KernelContext& ctx, const uint32_t* rows, const uint64_t* mask, int64_t p0,
                                              int64_t p1, int64_t j0, int64_t j1, int16_t* db) const {
    const int64_t np = p1 - p0;
    for (int64_t u = j0; u < j1; u += 16) {
      const int64_t nu = std::min<int64_t>(16, j1 - u);
      dots_unit<kPair>(rows, np, u, nu, db);
      if (nu == 16) {
        res2_epilogue_dispatch(ctx, mask, p0, np, u, db);
      } else {
        res2_epilogue_tail(ctx, mask, p0, np, u, nu, db);
      }
    }
  }

  template <bool kPair>
  inline __attribute__((always_inline)) void dots_unit(const uint32_t* rows, int64_t np, int64_t u, int64_t nu, int16_t* db) const {
#if defined(__aarch64__)
    if (!kPair && t48_)
      return dots_unit_rows8(static_cast<const uint8_t*>(wbuf_.get()), reinterpret_cast<const uint8_t*>(rows), steps_, k_bits_, np, u, nu, db);
    if (kPair && p84_)
      return dots_unit_rows<true, true>(static_cast<const uint8_t*>(wbuf_.get()), reinterpret_cast<const uint8_t*>(rows), steps_, k_bits_, np,
                                        u, nu, db);
#endif
    dots_unit_rows<kPair>(static_cast<const uint8_t*>(wbuf_.get()), reinterpret_cast<const uint8_t*>(rows), steps_, k_bits_, np, u, nu, db);
  }

  void res2_epilogue_dispatch(const KernelContext& ctx, const uint64_t* mask, int64_t p0, int64_t np, int64_t u, int16_t* db) const {
#define BF_RES2(L, B, K) res2_epilogue<L, B, K>(ctx, mask, p0, np, u, db)
#define BF_RES2_K(L, B)                                 \
  do {                                                  \
    if (!res_.bits) BF_RES2(L, B, kBitsNone);           \
    else if (res_.bit_out) BF_RES2(L, B, kBitsOut);     \
    else BF_RES2(L, B, kBitsSum);                       \
  } while (0)
    if (res_.pre) {
      res2_epilogue_pre_dispatch(ctx, mask, p0, np, u, db);
      return;
    }
    if (res_.l8) {
      if (res_.has_bias) BF_RES2_K(true, true);
      else BF_RES2_K(true, false);
    } else {
      if (res_.has_bias) BF_RES2_K(false, true);
      else BF_RES2_K(false, false);
    }
#undef BF_RES2_K
#undef BF_RES2
  }

  // the pre stage variants, out of line (the existing instantiations keep their code)
  __attribute__((noinline)) void res2_epilogue_pre_dispatch(const KernelContext& ctx, const uint64_t* mask, int64_t p0, int64_t np,
                                                            int64_t u, int16_t* db) const {
#define BF_RES2(L, B, K) res2_epilogue<L, B, K, true>(ctx, mask, p0, np, u, db)
#define BF_RES2_K(L, B)                                 \
  do {                                                  \
    if (!res_.bits) BF_RES2(L, B, kBitsNone);           \
    else if (res_.bit_out) BF_RES2(L, B, kBitsOut);     \
    else BF_RES2(L, B, kBitsSum);                       \
  } while (0)
    if (res_.l8) {
      if (res_.has_bias) BF_RES2_K(true, true);
      else BF_RES2_K(true, false);
    } else {
      if (res_.has_bias) BF_RES2_K(false, true);
      else BF_RES2_K(false, false);
    }
#undef BF_RES2_K
#undef BF_RES2
  }

  // int32 border correction of the dot products of pixel mask m (pad_value != 1), channels o..o+15
  inline void res2_border(uint64_t m, int64_t o, int32x4_t d[4]) const {
    const int32_t ps = pad_value_ - 1;
    int32x4_t c[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
    for (; m; m &= m - 1) {
      const int32_t* ts = tap_sum_.data() + __builtin_ctzll(m) * oc_pad_ + o;
      for (int k = 0; k < 4; ++k) c[k] = vaddq_s32(c[k], vld1q_s32(ts + 4 * k));
    }
    for (int k = 0; k < 4; ++k) d[k] = vmlaq_n_s32(d[k], c[k], ps);
  }

  // pass 0: border pixels (pad_value != 1): the corrected dot products of a whole 16-channel unit into db (|dot| <= K:
  // int16 still) (tile_any = mask + mt_: OR of the masks of each micro-tile of 4 / 8 pixels, see build_patches)
  void dots_border(const uint64_t* mask, int64_t np, int64_t u, int16_t* db) const {
    const int tsh = pair_ ? 3 : 2;
    if (pad_value_ != 1)
      for (int64_t t = 0; t <= (np - 1) >> tsh; ++t) {
        if (!mask[mt_ + t]) continue;
        for (int64_t r = t << tsh; r < std::min<int64_t>(np, (t + 1) << tsh); ++r) {
          if (!mask[r]) continue;
          int32x4_t di[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
          res2_border(mask[r], u, di);
          int16_t* d = db + r * 16;
          vst1q_s16(d, vaddq_s16(vld1q_s16(d), vcombine_s16(vmovn_s32(di[0]), vmovn_s32(di[1]))));
          vst1q_s16(d + 8, vaddq_s16(vld1q_s16(d + 8), vcombine_s16(vmovn_s32(di[2]), vmovn_s32(di[3]))));
        }
      }
  }

  // ---- BConv2D with int8 (kOutI8) or fp32 (kOutF32) output, groups == 1: the two phases of compute_res2 with the
  // plain epilogue of compute_tile_a: per pixel and 16-channel unit, v = act(fp32(dot (+ border correction)) * mul +
  // add) from the unit's constants in registers, one 16-byte int8 store (FCVTNS, saturating narrow, clamp) or four
  // fp32 stores. Same operations per element as compute_tile_a; a last unit of fewer than 16 channels runs per 4-channel
  // block (border correction in int32 as compute_tile_a's general loop).
  template <bool kPair, int kOut, Activation A>
  __attribute__((noinline)) void compute_dot2(const KernelContext& ctx, const uint32_t* rows, const uint64_t* mask, int64_t p0,
                                              int64_t p1, int64_t j0, int64_t j1, int16_t* db) const {
    const int64_t np = p1 - p0;
    for (int64_t u = j0; u < j1; u += 16) {
      const int64_t nu = std::min<int64_t>(16, j1 - u);
      dots_unit<kPair>(rows, np, u, nu, db);
      if (nu == 16) {
        dots_border(mask, np, u, db);
        dot2_epilogue<kOut, A>(ctx, p0, np, u, db);
      } else {
        dot2_tail<kOut, A>(ctx, mask, p0, np, u, nu, db);
      }
    }
  }
  template <int kOut, Activation A>
  void dot2_epilogue(const KernelContext& ctx, int64_t p0, int64_t np, int64_t u, const int16_t* db) const {
    const int64_t oc = g_.oc;
    if (kOut == kOutF32 && unit16_)
      return epi_.store_unit16(db, np, mul_.data() + u, add_.data() + u, static_cast<int>(A), float_out(ctx), oc, p0, u);
    dot2_store16<kOut == kOutI8, A>(db, np, mul_.data() + u, add_.data() + u, i8_out_.lo, i8_out_.hi,
                                     kOut == kOutI8 ? ctx.out(0).data<int8_t>() + p0 * oc + u : nullptr,
                                     kOut == kOutF32 ? float_out(ctx) + p0 * oc + u : nullptr, oc);
  }
  template <int kOut, Activation A>
  void dot2_tail(const KernelContext& ctx, const uint64_t* mask, int64_t p0, int64_t np, int64_t u, int64_t nu, const int16_t* db) const {
    const int64_t oc = g_.oc;
    const int32_t pad_scale = pad_value_ - 1;
    const int16x4_t lo4 = vdup_n_s16(static_cast<int16_t>(i8_out_.lo)), hi4 = vdup_n_s16(static_cast<int16_t>(i8_out_.hi));
    int8_t* y8 = kOut == kOutI8 ? ctx.out(0).data<int8_t>() : nullptr;
    float* yf = kOut == kOutF32 ? float_out(ctx) : nullptr;
    for (int64_t jb = 0; jb < nu; jb += 4) {
      const int64_t o = u + jb;
      const int nvo = static_cast<int>(std::min<int64_t>(4, nu - jb));
      const float32x4_t mul = vld1q_f32(mul_.data() + o), add = vld1q_f32(add_.data() + o);
      for (int64_t r = 0; r < np; ++r) {
        int32x4_t dot = vmovl_s16(vld1_s16(db + r * 16 + jb));
        uint64_t m = mask[r];
        if (m && pad_scale != 0) {
          int32x4_t corr = vdupq_n_s32(0);
          for (; m; m &= m - 1) corr = vaddq_s32(corr, vld1q_s32(tap_sum_.data() + __builtin_ctzll(m) * oc_pad_ + o));
          dot = vmlaq_n_s32(dot, corr, pad_scale);
        }
        const float32x4_t v = activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot), mul), add));
        const int64_t q = p0 + r;
        if constexpr (kOut == kOutI8) {
          i8_store4(y8 + q * oc + o, v, lo4, hi4, nvo);
        } else {
          float* yp = yf + q * oc + o;
          if (nvo == 4) {
            vst1q_f32(yp, v);
          } else {
            float tmp[4];
            vst1q_f32(tmp, v);
            for (int jj = 0; jj < nvo; ++jj) yp[jj] = tmp[jj];
          }
        }
      }
    }
  }
  template <bool kPair, int kOut>
  void compute_dot2_t(const KernelContext& ctx, const uint32_t* rows, const uint64_t* mask, int64_t p0, int64_t p1, int64_t j0,
                      int64_t j1, int16_t* db) const {
    switch (act_) {
      case Activation::kNone: return compute_dot2<kPair, kOut, Activation::kNone>(ctx, rows, mask, p0, p1, j0, j1, db);
      case Activation::kRelu: return compute_dot2<kPair, kOut, Activation::kRelu>(ctx, rows, mask, p0, p1, j0, j1, db);
      case Activation::kRelu6: return compute_dot2<kPair, kOut, Activation::kRelu6>(ctx, rows, mask, p0, p1, j0, j1, db);
    }
  }

  // kBits: kBitsNone, kBitsSum (bits of u) or kBitsOut (bit_src "out": bits of w, after the bias)
  template <bool kL8, bool kBias, int kBits, bool kPre = false>
  void res2_epilogue(const KernelContext& ctx, const uint64_t* mask, int64_t p0, int64_t np, int64_t u, int16_t* db) const {
    const int64_t oc = g_.oc;
    int8_t* y = ctx.out(0).data<int8_t>() + p0 * oc + u;
    const int8_t* rp = ctx.in(5).data<int8_t>() + p0 * oc + u;
    auto* yb = kBits != kBitsNone ? reinterpret_cast<uint8_t*>(ctx.out(1).data<uint32_t>() + p0 * out_words_) + u / 8 : nullptr;
    const int64_t ybs = out_words_ * 4;
    const bool whole_word = (u & 31) == 0 && u + 16 >= oc;  // the word's upper half holds no channel: write it as 0
    float32x4_t mul[4], add[4], slope[4], bias[4], bthr[4];
    for (int k = 0; k < 4; ++k) {
      mul[k] = vld1q_f32(mul_.data() + u + 4 * k);
      add[k] = vld1q_f32(add_.data() + u + 4 * k);
      slope[k] = vld1q_f32(res_.slope.data() + u + 4 * k);
      if (kBias) bias[k] = vld1q_f32(res_.bias.data() + u + 4 * k);
      if (kBits) bthr[k] = vld1q_f32(res_.bthr.data() + u + 4 * k);
    }
    const l8neon::DecTables dt = res_.dt;
    const l8neon::EncTables et = res_.et;
    const int32x4_t rz = vdupq_n_s32(res_.rz);
    const float32x4_t rs = vdupq_n_f32(res_.rs);
    const float32x4_t zero = vdupq_n_f32(0.f);
    static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t bw = vld1q_u8(kW);
    dots_border(mask, np, u, db);
    // pass 1: u = fp32(dot) * mul + add + R(res) for every pixel of the tile (fewer live constants per pass)
    alignas(16) float ub[64 * 16];
    if constexpr (kPre) {
      // pre stage (I8Res::pre1): v = v >= 0 ? v : v * pslope;  v = v * pmul;  v = v + padd;  v = v * G[sample]
      float32x4_t ps[4], pm[4], pa[4];
      for (int k = 0; k < 4; ++k) {
        ps[k] = vld1q_f32(res_.pslope.data() + u + 4 * k);
        pm[k] = vld1q_f32(res_.pmul.data() + u + 4 * k);
        pa[k] = vld1q_f32(res_.padd.data() + u + 4 * k);
      }
      const int64_t ppi = g_.oh * g_.ow;
      for (int64_t r = 0; r < np; ++r, rp += oc) {
        const int16x8_t d0 = vld1q_s16(db + r * 16), d1 = vld1q_s16(db + r * 16 + 8);
        const int32x4_t di[4] = {vmovl_s16(vget_low_s16(d0)), vmovl_high_s16(d0), vmovl_s16(vget_low_s16(d1)), vmovl_high_s16(d1)};
        float32x4_t rr[4];
        if constexpr (kL8) l8neon::decode16(vld1q_s8(rp), dt, rr);
        else l8neon::decode16_linear(vld1q_s8(rp), rz, rs, rr);
        const float* gv = res_.gate ? res_.gval.data() + ((p0 + r) / ppi) * res_.oc_row + u : nullptr;
        for (int k = 0; k < 4; ++k) {
          float32x4_t v = vaddq_f32(vmulq_f32(vcvtq_f32_s32(di[k]), mul[k]), add[k]);
          v = vbslq_f32(vcgeq_f32(v, zero), v, vmulq_f32(v, ps[k]));
          v = vaddq_f32(vmulq_f32(v, pm[k]), pa[k]);
          if (gv) v = vmulq_f32(v, vld1q_f32(gv + 4 * k));
          vst1q_f32(ub + r * 16 + 4 * k, vaddq_f32(v, rr[k]));
        }
      }
    } else {
      for (int64_t r = 0; r < np; ++r, rp += oc) {
        const int16x8_t d0 = vld1q_s16(db + r * 16), d1 = vld1q_s16(db + r * 16 + 8);
        const int32x4_t di[4] = {vmovl_s16(vget_low_s16(d0)), vmovl_high_s16(d0), vmovl_s16(vget_low_s16(d1)), vmovl_high_s16(d1)};
        float32x4_t rr[4];
        if constexpr (kL8) l8neon::decode16(vld1q_s8(rp), dt, rr);
        else l8neon::decode16_linear(vld1q_s8(rp), rz, rs, rr);
        for (int k = 0; k < 4; ++k)
          vst1q_f32(ub + r * 16 + 4 * k, vaddq_f32(vaddq_f32(vmulq_f32(vcvtq_f32_s32(di[k]), mul[k]), add[k]), rr[k]));
      }
    }
    // pass 2: PReLU, bias, l8 encode, bits
    for (int64_t r = 0; r < np; ++r, y += oc) {
      float32x4_t w[4];
      uint32x4_t ge[4];
      for (int k = 0; k < 4; ++k) {
        const float32x4_t uu = vld1q_f32(ub + r * 16 + 4 * k);
        w[k] = vbslq_f32(vcgeq_f32(uu, zero), uu, vmulq_f32(uu, slope[k]));
        if constexpr (kBias) w[k] = vaddq_f32(w[k], bias[k]);
        if constexpr (kBits == kBitsSum) ge[k] = vcgeq_f32(uu, bthr[k]);
        if constexpr (kBits == kBitsOut) ge[k] = vcgeq_f32(w[k], bthr[k]);
      }
      vst1q_s8(y, l8neon::encode16(w[0], w[1], w[2], w[3], et));
      if constexpr (kBits != kBitsNone) {
        // bit = 1 iff !(s >= thr), s = u or w: 16 mask bytes, BIC with the bit weights, three ADDPs -> 2 bytes
        const uint16x8_t m01 = vuzp1q_u16(vreinterpretq_u16_u32(ge[0]), vreinterpretq_u16_u32(ge[1]));
        const uint16x8_t m23 = vuzp1q_u16(vreinterpretq_u16_u32(ge[2]), vreinterpretq_u16_u32(ge[3]));
        const uint8x16_t mb = vuzp1q_u8(vreinterpretq_u8_u16(m01), vreinterpretq_u8_u16(m23));
        uint8x16_t sb = vbicq_u8(bw, mb);
        sb = vpaddq_u8(sb, sb);
        sb = vpaddq_u8(sb, sb);
        sb = vpaddq_u8(sb, sb);
        vst1q_lane_u16(reinterpret_cast<uint16_t*>(yb + r * ybs), vreinterpretq_u16_u8(sb), 0);
      }
    }
    if (kBits != kBitsNone && whole_word)  // the word's upper half holds no channel: 0
      for (int64_t r = 0; r < np; ++r) std::memset(yb + r * ybs + 2, 0, 2);
  }

  // Last unit of nu < 16 channels: scalar epilogue; its bits (and 0 for the rest of the word) written whole.
  void res2_epilogue_tail(const KernelContext& ctx, const uint64_t* mask, int64_t p0, int64_t np, int64_t u, int64_t nu,
                          const int16_t* db) const {
    const int64_t oc = g_.oc;
    const int32_t ps = pad_value_ - 1;
    uint32_t* ybw = res_.bits ? ctx.out(1).data<uint32_t>() : nullptr;
    int8_t* y8 = ctx.out(0).data<int8_t>();
    const int8_t* r8 = ctx.in(5).data<int8_t>();
    for (int64_t r = 0; r < np; ++r) {
      const int64_t q = p0 + r;
      uint32_t bits = 0;
      for (int64_t jj = 0; jj < nu; ++jj) {
        const int64_t o = u + jj;
        int32_t dot = db[r * 16 + jj];
        if (pad_value_ != 1)
          for (uint64_t m = mask[r]; m; m &= m - 1) dot += ps * tap_sum_[__builtin_ctzll(m) * oc_pad_ + o];
        float v = static_cast<float>(dot) * mul_[o] + add_[o];
        if (res_.pre) v = res_.pre1(v, q / (g_.oh * g_.ow), o);
        bool bit;
        y8[q * oc + o] = res_.finish(v, o, r8[q * oc + o], &bit);
        bits |= static_cast<uint32_t>(bit) << jj;
      }
      if (ybw) {
        uint8_t* p = reinterpret_cast<uint8_t*>(ybw + q * out_words_) + u / 8;
        if ((u & 31) == 0) {
          std::memcpy(p, &bits, 4);
        } else {
          const uint16_t h = static_cast<uint16_t>(bits);
          std::memcpy(p, &h, 2);
        }
      }
    }
  }

  // Channels [j0, j1) of group gi (j0 a multiple of 4) for pixels [p0, p1).
  // kPair: 8-pixel tiles over interleaved pixel-pair rows (see build_patches).
  // kOut: kOutF32 stores floats (the output, or yf_ before an epilogue); bit
  // output without epilogue stores the binarization of each 4-channel block:
  // kOutBitsInt compares the integer dot product with the per-channel integer
  // thresholds (IntThresholds: no float epilogue at all), kOutBitsFloat (only
  // when those do not exist) computes the float value and compares it.
  // The activation is a template parameter (compute_tile_a) so that the
  // per-pixel epilogue carries no dispatch.
  static constexpr int kOutF32 = 0, kOutBitsFloat = 1, kOutBitsInt = 2, kOutI8 = 3;  // kOutI8: int8_full output
  static constexpr int kOutI8Res = 4;  // int8_full I8BConv2DRes: residual + PReLU epilogue, l8 output, bit output

  // I8BConv2DRes full micro-tiles of one 4-channel block (channels o..o+3, groups == 1): per 4 rows x 4 channels the
  // residual codes are gathered (4 lane loads), decoded, added, PReLU'd (+ bias), encoded to l8 (4 lane stores) and the
  // 4 bits per row OR-ed into the bit output words. Flags are template parameters so that the loop carries no
  // dispatch and the tables stay in registers. kBits: no bit output, bits of u (bit_src "sum") or of w ("out").
  static constexpr int kBitsNone = 0, kBitsSum = 1, kBitsOut = 2;
  template <bool kPair, bool kL8, bool kBias, int kBits, class DotOf>
  void res_tiles(const uint8_t* a, const uint8_t* b, int64_t row_bytes, const uint64_t* mask, const uint64_t* tile_any, int64_t n_full,
                 int8_t* yp, const int8_t* rp, uint32_t* wp, int64_t o, float32x4_t mul, float32x4_t add, const DotOf& dot_of) const {
    constexpr int kTp = kPair ? 8 : 4;
    const int64_t ys = g_.oc, ws = out_words_;
    const float32x4_t slope = vld1q_f32(res_.slope.data() + o), bias = vld1q_f32(res_.bias.data() + o);
    const float32x4_t bthr = vld1q_f32(res_.bthr.data() + o), zero = vdupq_n_f32(0.f);
    const int32x4_t sh = vdupq_n_s32(static_cast<int32_t>(o & 31));
    const bool first = (o & 31) == 0;
    const l8neon::DecTables dt = res_.dt;
    const l8neon::EncTables et = res_.et;
    const int32x4_t rz = vdupq_n_s32(res_.rz);
    const float32x4_t rs = vdupq_n_f32(res_.rs);
    static const uint32_t kBitW[4] = {1, 2, 4, 8};
    const uint32x4_t bitw = vld1q_u32(kBitW);
    for (int64_t t = 0; t < n_full; ++t, a += 4 * row_bytes, yp += kTp * ys, rp += kTp * ys) {
      uint32x4_t pc[kTp];
      if constexpr (kPair) popcount_8x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
      else popcount_4x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
      const bool any = tile_any[t] != 0;
      for (int r4 = 0; r4 < kTp; r4 += 4) {
        int8_t* y = yp + r4 * ys;
        const int8_t* rr = rp + r4 * ys;
        int32x4_t rc = vdupq_n_s32(0);
        rc = vld1q_lane_s32(reinterpret_cast<const int32_t*>(rr), rc, 0);
        rc = vld1q_lane_s32(reinterpret_cast<const int32_t*>(rr + ys), rc, 1);
        rc = vld1q_lane_s32(reinterpret_cast<const int32_t*>(rr + 2 * ys), rc, 2);
        rc = vld1q_lane_s32(reinterpret_cast<const int32_t*>(rr + 3 * ys), rc, 3);
        float32x4_t r[4];
        if constexpr (kL8) l8neon::decode16(vreinterpretq_s8_s32(rc), dt, r);
        else l8neon::decode16_linear(vreinterpretq_s8_s32(rc), rz, rs, r);
        float32x4_t w[4];
        uint32x4_t m[4];
        for (int k = 0; k < 4; ++k) {
          const int rr4 = r4 + k;
          const float32x4_t v = vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot_of(pc[rr4], any ? mask[t * kTp + rr4] : 0)), mul), add);
          const float32x4_t u = vaddq_f32(v, r[k]);
          w[k] = vbslq_f32(vcgeq_f32(u, zero), u, vmulq_f32(u, slope));
          if constexpr (kBias) w[k] = vaddq_f32(w[k], bias);
          if constexpr (kBits == kBitsSum) m[k] = vbicq_u32(bitw, vcgeq_f32(u, bthr));
          if constexpr (kBits == kBitsOut) m[k] = vbicq_u32(bitw, vcgeq_f32(w[k], bthr));
        }
        const uint32x4_t yc = vreinterpretq_u32_s8(l8neon::encode16(w[0], w[1], w[2], w[3], et));
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(y), yc, 0);
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(y + ys), yc, 1);
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(y + 2 * ys), yc, 2);
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(y + 3 * ys), yc, 3);
        if constexpr (kBits != kBitsNone) {
          const uint32x4_t bv = vshlq_u32(vpaddq_u32(vpaddq_u32(m[0], m[1]), vpaddq_u32(m[2], m[3])), sh);
          uint32_t* q = wp + (t * kTp + r4) * ws;
          if (first) {
            q[0] = vgetq_lane_u32(bv, 0);
            q[ws] = vgetq_lane_u32(bv, 1);
            q[2 * ws] = vgetq_lane_u32(bv, 2);
            q[3 * ws] = vgetq_lane_u32(bv, 3);
          } else {
            q[0] |= vgetq_lane_u32(bv, 0);
            q[ws] |= vgetq_lane_u32(bv, 1);
            q[2 * ws] |= vgetq_lane_u32(bv, 2);
            q[3 * ws] |= vgetq_lane_u32(bv, 3);
          }
        }
      }
    }
  }
  template <bool kPair, class DotOf>
  void res_tiles_dispatch(const uint8_t* a, const uint8_t* b, int64_t row_bytes, const uint64_t* mask, const uint64_t* tile_any,
                          int64_t n_full, int8_t* yp, const int8_t* rp, uint32_t* wp, int64_t o, float32x4_t mul, float32x4_t add,
                          const DotOf& dot_of) const {
#define BF_RES_TILES(L, B, K) res_tiles<kPair, L, B, K>(a, b, row_bytes, mask, tile_any, n_full, yp, rp, wp, o, mul, add, dot_of)
#define BF_RES_TILES_K(L, B)                     \
  do {                                           \
    if (!res_.bits) BF_RES_TILES(L, B, kBitsNone); \
    else if (res_.bit_out) BF_RES_TILES(L, B, kBitsOut); \
    else BF_RES_TILES(L, B, kBitsSum);           \
  } while (0)
    if (res_.l8) {
      if (res_.has_bias) BF_RES_TILES_K(true, true);
      else BF_RES_TILES_K(true, false);
    } else {
      if (res_.has_bias) BF_RES_TILES_K(false, true);
      else BF_RES_TILES_K(false, false);
    }
#undef BF_RES_TILES_K
#undef BF_RES_TILES
  }

  template <bool kPair, int kOut>
  void compute_tile_t(const KernelContext& ctx, const uint32_t* rows, const uint64_t* mask, int64_t p0, int64_t p1, int64_t gi,
                      int64_t j0, int64_t j1) const {
    if constexpr (kOut == kOutBitsInt) {  // no float epilogue
      compute_tile_a<kPair, kOut, Activation::kNone>(ctx, rows, mask, p0, p1, gi, j0, j1);
    } else {
      switch (act_) {
        case Activation::kNone: return compute_tile_a<kPair, kOut, Activation::kNone>(ctx, rows, mask, p0, p1, gi, j0, j1);
        case Activation::kRelu: return compute_tile_a<kPair, kOut, Activation::kRelu>(ctx, rows, mask, p0, p1, gi, j0, j1);
        case Activation::kRelu6: return compute_tile_a<kPair, kOut, Activation::kRelu6>(ctx, rows, mask, p0, p1, gi, j0, j1);
      }
    }
  }
  // Full micro-tiles (all kTp pixels and 4 channels valid) take a straight-line
  // path: consecutive rows without clamping, the border correction skipped
  // per tile (tile_any, see build_patches), whole-vector stores; bit output
  // binarizes 4 rows at once. The general loop below handles the last partial
  // tile, channel tails and the rarer output modes.
  template <bool kPair, int kOut, Activation A>
  void compute_tile_a(const KernelContext& ctx, const uint32_t* rows, const uint64_t* mask, int64_t p0, int64_t p1, int64_t gi,
                      int64_t j0, int64_t j1) const {
    constexpr int kTp = kPair ? 8 : 4;
    const auto* wts = static_cast<const uint8_t*>(wbuf_.get());
    const int64_t np = p1 - p0;
    const int64_t row_bytes = steps_ * 16;  // one tile row: a pixel, or a pixel pair in pair mode
    const int64_t last_row = kPair ? (np - 1) / 2 : np - 1;
    const auto* base = reinterpret_cast<const uint8_t*>(rows);
    const int32x4_t kb = vdupq_n_s32(k_bits_);
    const int32_t pad_scale = pad_value_ - 1;
    float* yf = kOut == kOutF32 ? float_out(ctx) : nullptr;
    int8_t* y8 = kOut == kOutI8 || kOut == kOutI8Res ? ctx.out(0).data<int8_t>() : nullptr;
    const int8_t* r8 = kOut == kOutI8Res ? ctx.in(5).data<int8_t>() : nullptr;
    uint32_t* yrb = kOut == kOutI8Res && res_.bits ? ctx.out(1).data<uint32_t>() : nullptr;
    uint32_t* yb = kOut == kOutBitsFloat || kOut == kOutBitsInt ? ctx.out(0).data<uint32_t>() : nullptr;
    const int16x4_t lo4 = vdup_n_s16(static_cast<int16_t>(i8_out_.lo)), hi4 = vdup_n_s16(static_cast<int16_t>(i8_out_.hi));
    const int16x8_t lo8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.lo)), hi8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.hi));
    static const uint32_t kBitW[4] = {1, 2, 4, 8};
    const uint32x4_t bitw = vld1q_u32(kBitW);
    const uint64_t* tile_any = mask + mt_;  // per micro-tile: OR of the pixel masks (build_patches)
    for (int64_t j = j0; j < j1; j += 4) {
      const int64_t pc_idx = gi * ocg_pad_ + j;  // padded channel index of the block
      const int64_t o = gi * ocg_ + j;           // first output channel of the block
      const uint8_t* b = wts + (pc_idx / 4) * steps_ * 64;
      const float32x4_t mul = vld1q_f32(mul_.data() + pc_idx), add = vld1q_f32(add_.data() + pc_idx);
      const int nvo = static_cast<int>(std::min<int64_t>(4, ocg_ - j));
      int64_t i_gen = 0;  // first pixel of the general loop
      if (kOut != kOutBitsFloat && nvo == 4 && !(kOut == kOutBitsInt && bit_rmw_)) {
        const int64_t n_full = np / kTp;
        auto dot_of = [&](uint32x4_t pc, uint64_t m) {
          int32x4_t dot = vsubq_s32(kb, vshlq_n_s32(vreinterpretq_s32_u32(pc), 1));
          if (m) {
            int32x4_t corr = vdupq_n_s32(0);
            for (; m; m &= m - 1) corr = vaddq_s32(corr, vld1q_s32(tap_sum_.data() + __builtin_ctzll(m) * oc_pad_ + pc_idx));
            dot = vmlaq_n_s32(dot, corr, pad_scale);
          }
          return dot;
        };
        const uint8_t* a = base;
        if constexpr (kOut == kOutI8Res) {
          res_tiles_dispatch<kPair>(a, b, row_bytes, mask, tile_any, n_full, y8 + p0 * g_.oc + o, r8 + p0 * g_.oc + o,
                                    yrb ? yrb + p0 * out_words_ + (o >> 5) : nullptr, o, mul, add, dot_of);
        } else if constexpr (kOut == kOutI8) {
          int8_t* yp = y8 + p0 * g_.oc + o;
          const int64_t ys = g_.oc;
          for (int64_t t = 0; t < n_full; ++t, a += 4 * row_bytes, yp += kTp * ys) {
            uint32x4_t pc[kTp];
            if constexpr (kPair) popcount_8x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
            else popcount_4x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
            const bool any = tile_any[t] != 0;
            float32x4_t v[kTp];
            for (int r = 0; r < kTp; ++r)
              v[r] = activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot_of(pc[r], any ? mask[t * kTp + r] : 0)), mul), add));
            for (int r = 0; r < kTp; r += 4) i8_store4x4(yp + r * ys, ys, v[r], v[r + 1], v[r + 2], v[r + 3], lo8, hi8);
          }
        } else if constexpr (kOut == kOutF32) {
          float* yp = yf + p0 * g_.oc + o;
          const int64_t ys = g_.oc;
          for (int64_t t = 0; t < n_full; ++t, a += 4 * row_bytes, yp += kTp * ys) {
            uint32x4_t pc[kTp];
            if constexpr (kPair) popcount_8x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
            else popcount_4x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
            if (tile_any[t] == 0) {
              for (int r = 0; r < kTp; ++r)
                vst1q_f32(yp + r * ys, activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot_of(pc[r], 0)), mul), add)));
            } else {
              for (int r = 0; r < kTp; ++r)
                vst1q_f32(yp + r * ys, activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot_of(pc[r], mask[t * kTp + r])), mul), add)));
            }
          }
        } else {
          // bit = neg ? dot > lim : dot < lim, i.e. (dot ^ neg) < (lim ^ neg) (~x reverses the order).
          const int32x4_t negm = vld1q_s32(neg_.data() + pc_idx);
          const int32x4_t limx = veorq_s32(vld1q_s32(lim_.data() + pc_idx), negm);
          const int32x4_t sh = vdupq_n_s32(static_cast<int32_t>(o & 31));
          uint32_t* wp = yb + p0 * out_words_ + (o >> 5);
          const int64_t ws = out_words_;
          const bool first = (o & 31) == 0;  // first block of the word (blocks run in channel order)
          for (int64_t t = 0; t < n_full; ++t, a += 4 * row_bytes) {
            uint32x4_t pc[kTp];
            if constexpr (kPair) popcount_8x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
            else popcount_4x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps_, pc);
            const bool any = tile_any[t] != 0;
            for (int r4 = 0; r4 < kTp; r4 += 4) {
              uint32x4_t m[4];
              for (int r = 0; r < 4; ++r) {
                const int32x4_t dot = dot_of(pc[r4 + r], any ? mask[t * kTp + r4 + r] : 0);
                m[r] = vandq_u32(bitw, vcltq_s32(veorq_s32(dot, negm), limx));
              }
              // lane r = the 4 bits of row r4 + r, shifted to the block's position in its word
              const uint32x4_t bits = vshlq_u32(vpaddq_u32(vpaddq_u32(m[0], m[1]), vpaddq_u32(m[2], m[3])), sh);
              uint32_t lanes[4];
              vst1q_u32(lanes, bits);
              for (int r = 0; r < 4; ++r, wp += ws) {
                if (first) *wp = lanes[r];
                else *wp |= lanes[r];
              }
            }
          }
        }
        i_gen = n_full * kTp;
      }
      for (int64_t i = i_gen; i < np; i += kTp) {
        const int nv = static_cast<int>(std::min<int64_t>(kTp, np - i));
        const int64_t r0 = kPair ? i / 2 : i;
        const uint8_t* a[4];
        for (int r = 0; r < 4; ++r) a[r] = base + std::min(r0 + r, last_row) * row_bytes;
        uint32x4_t pc[kTp];
        if constexpr (kPair) popcount_8x4(a[0], a[1], a[2], a[3], b, steps_, pc);
        else popcount_4x4(a[0], a[1], a[2], a[3], b, steps_, pc);
        for (int r = 0; r < nv; ++r) {
          int32x4_t dot = vsubq_s32(kb, vshlq_n_s32(vreinterpretq_s32_u32(pc[r]), 1));
          uint64_t m = mask[i + r];
          if (m && pad_scale != 0) {
            int32x4_t corr = vdupq_n_s32(0);
            for (; m; m &= m - 1) corr = vaddq_s32(corr, vld1q_s32(tap_sum_.data() + __builtin_ctzll(m) * oc_pad_ + pc_idx));
            dot = vmlaq_n_s32(dot, corr, pad_scale);
          }
          const int64_t q = p0 + i + r;
          if constexpr (kOut == kOutI8Res) {
            float vv[4];
            vst1q_f32(vv, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot), mul), add));
            uint32_t bits = 0;
            for (int jj = 0; jj < nvo; ++jj) {
              bool bit;
              y8[q * g_.oc + o + jj] = res_.finish(vv[jj], o + jj, r8[q * g_.oc + o + jj], &bit);
              bits |= static_cast<uint32_t>(bit) << jj;
            }
            if (yrb) {
              uint32_t* wq = yrb + q * out_words_ + (o >> 5);
              if ((o & 31) == 0) *wq = bits;
              else *wq |= bits << (o & 31);
            }
          } else if constexpr (kOut == kOutI8) {
            i8_store4(y8 + q * g_.oc + o, activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot), mul), add)), lo4, hi4, nvo);
          } else if constexpr (kOut == kOutF32) {
            float32x4_t v = vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot), mul), add);
            v = activate(A, v);
            float* yp = yf + q * g_.oc + o;
            if (nvo == 4) {
              vst1q_f32(yp, v);
            } else {
              float tmp[4];
              vst1q_f32(tmp, v);
              for (int jj = 0; jj < nvo; ++jj) yp[jj] = tmp[jj];
            }
          } else {
            uint32x4_t one;  // lanes whose bit is 1 (-1)
            if constexpr (kOut == kOutBitsInt) {
              const int32x4_t lim = vld1q_s32(lim_.data() + pc_idx);
              one = vbslq_u32(vreinterpretq_u32_s32(vld1q_s32(neg_.data() + pc_idx)), vcgtq_s32(dot, lim), vcltq_s32(dot, lim));
            } else {
              // bit = 1 iff !(v >= t) (ge) / !(v > t) (gt)
              float32x4_t v = vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot), mul), add);
              v = activate(A, v);
              const float32x4_t t = vld1q_f32(thr_.data() + pc_idx);
              one = vmvnq_u32(bout_.gt() ? vcgtq_f32(v, t) : vcgeq_f32(v, t));
            }
            uint32_t bits = vaddvq_u32(vandq_u32(bitw, one));
            bits &= (1u << nvo) - 1;  // padded channels (jj >= nvo) stay 0
            uint32_t* wp = yb + q * out_words_ + (o >> 5);
            if (bit_rmw_) {
              // Words zeroed per chunk in run_task; a block may straddle two words.
              const int64_t sh = o & 31;
              wp[0] |= bits << sh;
              if (sh + nvo > 32) wp[1] |= bits >> (32 - sh);
            } else if ((o & 31) == 0) {
              *wp = bits;  // first block of the word (blocks run in channel order)
            } else {
              *wp |= bits << (o & 31);
            }
          }
        }
      }
    }
  }

  struct Chunk {
    int64_t c0, c1;
  };

  Conv2DGeom g_;
  BConvGroupLayout L_;
  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  int pad_value_ = 0;
  bool bit_out_ = false, bit_in_ = false, pair_ = false, bit_rmw_ = false, direct_bits_ = false;
  int64_t taps_ = 0, k_words_ = 0, kwp_ = 0, steps_ = 0, out_words_ = 0, ocg_ = 0, ocg_pad_ = 0, oc_pad_ = 0, pixels_ = 0;
  int32_t k_bits_ = 0;
  int64_t mt_ = 4, n_ptiles_ = 0, items_ = 0, n_tasks_ = 1, slot_words_ = 0;
  std::vector<Chunk> chunks_;
  std::shared_ptr<void> wbuf_, xbits_, patches_;
  std::shared_ptr<void> dots_;  // compute_res2: per task, mt_ x 16 int16 dot products
  bool res2_ = false, dot2_ = false, unit16_ = false, t48_ = false, p84_ = false;
  std::shared_ptr<void> yf_;  // [pixels][oc] float output before binarization (bit output + epilogue)
  std::vector<int32_t> tap_sum_;
  std::vector<uint64_t> masks_;
  std::vector<float> mul_, add_;
  std::vector<float> thr_;  // output threshold by padded channel index (direct bit output)
  std::vector<int32_t> lim_, neg_;  // integer thresholds by padded channel index (direct bit output)
  IntThresholds ithr_;
  uint64_t ithr_gen_ = ~uint64_t(0);
  bool int_ok_ = false;
  BitOutput bout_;
  EltwiseProgram epi_;
  I8Out i8_out_;
  I8Res res_;
};

bool bconv_neon_opt_supported(const KernelContext& ctx) {
  if (!bconv_packed_grouped_supported(ctx)) return false;
  const Shape& w = ctx.in(1).shape();
  return w[1] * w[2] <= kMaxTaps;
}

// ------------------------------------------------------------------- BDense
// Rows (input vectors) x output blocks of 4. With enough rows the work is cut
// into row tiles of rt_ rows (all outputs of a tile computed by one task, the
// tile's input rows packed by that task) so that a fused epilogue
// (EltwiseProgram) runs on the tile while it is in cache. K of 1-2 words mod 4
// (e.g. K = 64) uses pair mode as BConv2D: two rows share each 16-byte
// register, halving the padding work. Few rows (GEMV-like): parallel over
// output blocks, epilogue afterwards.
class BDenseNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    mode_ = parse_binarize(ctx.attrs());
    k_ = ctx.in(1).dim(1);
    on_ = ctx.in(1).dim(0);
    words_ = bit_words(k_);
    rows_ = ctx.in(0).size() / k_;
    tiled_ = rows_ >= 8;
    pair_ = tiled_ && (words_ % 4 == 1 || words_ % 4 == 2) && words_ <= kMaxPairWords;
    kwp_ = pair_ ? (words_ + 1) / 2 * 2 : (words_ + 3) / 4 * 4;
    steps_ = pair_ ? kwp_ / 2 : kwp_ / 4;
    on_pad_ = (on_ + 3) / 4 * 4;
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * steps_ * 16));
    if (pair_) interleave_rows4_dup(ctx.in(1).data<uint32_t>(), on_, words_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    else interleave_rows4(ctx.in(1).data<uint32_t>(), on_, words_, words_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    // Normal: rows of kwp_ words. Pair: pair rows of steps_ x 16 bytes. Padding stays 0.
    xbuf_ = aligned_alloc_shared(static_cast<size_t>((pair_ ? (rows_ + 1) / 2 * 2 : rows_) * kwp_ * 4));
    mul_.assign(static_cast<size_t>(on_pad_), 1.f);
    add_.assign(static_cast<size_t>(on_pad_), -0.f);
    // Row tile: ~32 KB of f32 output, a multiple of 8 rows.
    rt_ = std::min<int64_t>(256, std::max<int64_t>(8, 8192 / std::max<int64_t>(on_, 1) / 8 * 8));
    // Several threads: at least ~4 row tiles per thread when the rows allow it (e.g. 197 tokens x 192 outputs gave 5
    // tiles of 40 rows: at 4 threads on 1 X2 + 3 A710 the A710 tiles were the critical path). Row tiles only
    // partition the output; every value is computed the same way.
    if (const int64_t th = ctx.pool->num_threads(); th > 1)
      rt_ = std::min(rt_, std::max<int64_t>(8, (rows_ + 4 * th - 1) / (4 * th) + 7) / 8 * 8);
    n_rt_ = (rows_ + rt_ - 1) / rt_;
    const Tensor& y = ctx.out(0);
    epi_.prepare_epilogue(ctx, y.dim(0) > 0 ? rows_ / y.dim(0) : rows_);
    bout_.parse(ctx);
    if (bout_.active) yf_ = aligned_alloc_shared(static_cast<size_t>(rows_ * on_ * 4));
    i8_out_.parse(ctx);
    BF_CHECK(!i8_out_.on || (!bout_.active && !epi_.active()), "bdense.neon: int8 output without bit output / epilogue only");
    // tile2 (two phases); BITINO_BCONV_DOT2=0 keeps tile_t
    two_ = tiled_ && rt_ <= 256 && k_ <= 32767 && on_ >= 16 && !opts_dot2_off();
    // fp32 output with a chain epilogue: run on each 16-channel unit in registers (EltwiseProgram::store_unit16)
    unit16_ = two_ && !i8_out_.on && epi_.active() && on_ % 16 == 0 && epi_.unit16();
#if defined(__aarch64__)
    // fp32 two-phase rows of whole 128-bit steps, whole 16-channel units and no bit output: every output goes through
    // tile2, so the weights are interleaved by 8 channels only (the 4x8 tile, dots_unit_rows8; as BConv2DNeonOpt).
    t48_ = two_ && !pair_ && !bout_.active && on_ % 16 == 0 && !opts_t48_off();  // int8 output too (int8_full, 2026-11-13)
    p84_ = two_ && pair_ && cpu_info().prime_part == kPartCortexA75 && !opts_t48_off();  // as BConv2DNeonOpt
    if (t48_) {
      wbuf_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * steps_ * 16));
      interleave_rows8(ctx.in(1).data<uint32_t>(), on_, words_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    }
#endif
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(on_pad_ * steps_ * 16); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, on_, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, on_, add_.data());
    if (bout_.active && !epi_.active() &&
        ithr_.update(k_, on_, opt_f32(ctx, 2), opt_f32(ctx, 3), act_, bout_.mode, bout_.threshold(ctx))) {
      run_bits_int(ctx);
      return;
    }
    if (i8_out_.on) return run_t(ctx, ctx.out(0).data<int8_t>());
    if (epi_.active()) epi_.begin(ctx);
    // Bit output: the float rows (and the epilogue) go to yf_, then each finished row tile is binarized.
    float* y = bout_.active ? static_cast<float*>(yf_.get()) : ctx.out(0).data<float>();
    run_t(ctx, y);
  }

 private:
  template <typename T>
  void run_t(const KernelContext& ctx, T* y) {
    if (tiled_) {
      // >= ~8K popcount steps per pool chunk (a dispatch costs ~25 us on the SM8450).
      const int64_t tile_steps = std::max<int64_t>(1, rt_ * (on_pad_ / 4) * steps_ / (pair_ ? 2 : 1));
      const int64_t grain = std::max<int64_t>(1, 8192 / tile_steps);
      ctx.pool->parallel_for(n_rt_, grain, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
          const int64_t r0 = t * rt_, r1 = std::min(rows_, r0 + rt_);
          pack_rows(ctx, r0, r1);
          if (two_) {
            if (pair_) tile2_rows<true>(r0, r1, y);
            else tile2_rows<false>(r0, r1, y);
          } else if (pair_) {
            tile_rows<true>(r0, r1, 0, on_pad_ / 4, y);
          } else {
            tile_rows<false>(r0, r1, 0, on_pad_ / 4, y);
          }
          if constexpr (std::is_same<T, float>::value) {
            if (epi_.active() && !unit16_) epi_.apply(y, r0, r1);  // rows still in cache
            if (bout_.active) pack_output_rows_neon(ctx, y, r0, r1);
          }
        }
      });
      return;
    }
    pack_rows(ctx, 0, rows_);
    const int64_t blocks = on_pad_ / 4;
    // >= 8K popcount steps (~10-20 us) per chunk: a pool dispatch costs ~25 us
    // on the SM8450, so smaller GEMVs run on the calling thread.
    const int64_t grain = std::max<int64_t>(1, 8192 / std::max<int64_t>(1, steps_ * rows_));
    ctx.pool->parallel_for(blocks, grain, [&](int64_t b0, int64_t b1) { tile_rows<false>(0, rows_, b0, b1, y); });
    if constexpr (std::is_same<T, float>::value) {
      if (epi_.active()) epi_.apply(y, 0, rows_);
      if (bout_.active) pack_output_rows_neon(ctx, y, 0, rows_);
    }
  }

  // Bit output (kernels/bit_output.h): binarizes finished float rows [r0, r1) of yf_.
  void pack_output_rows_neon(const KernelContext& ctx, const float* y, int64_t r0, int64_t r1) const {
    const float* thr = bout_.threshold(ctx);
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    const int64_t ow = bit_words(on_);
    for (int64_t r = r0; r < r1; ++r) pack_row_f32_neon(y + r * on_, on_, yb + r * ow, bout_.mode, thr);
  }

  // Bit output without epilogue: the bit of each output is an integer comparison
  // of its dot product (IntThresholds), stored straight into the bit rows. Words
  // are assembled in channel order by the task owning all blocks of a row tile
  // (tiled), or of 8 blocks = one word (GEMV-like, parallel over words).
  void run_bits_int(const KernelContext& ctx) {
    if (ithr_.generation() != ithr_gen_) {
      lim_.assign(static_cast<size_t>(on_pad_), INT32_MIN);  // padded channels: bit 0
      neg_.assign(static_cast<size_t>(on_pad_), 0);
      std::copy_n(ithr_.lim(), on_, lim_.data());
      std::copy_n(ithr_.neg(), on_, neg_.data());
      ithr_gen_ = ithr_.generation();
    }
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    const int64_t blocks = on_pad_ / 4;
    if (tiled_) {
      const int64_t tile_steps = std::max<int64_t>(1, rt_ * blocks * steps_ / (pair_ ? 2 : 1));
      const int64_t grain = std::max<int64_t>(1, 8192 / tile_steps);
      ctx.pool->parallel_for(n_rt_, grain, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
          const int64_t r0 = t * rt_, r1 = std::min(rows_, r0 + rt_);
          pack_rows(ctx, r0, r1);
          if (pair_) tile_bits_t<true>(r0, r1, 0, blocks, yb);
          else tile_bits_t<false>(r0, r1, 0, blocks, yb);
        }
      });
      return;
    }
    pack_rows(ctx, 0, rows_);
    const int64_t words = (blocks + 7) / 8;
    const int64_t grain = std::max<int64_t>(1, 8192 / std::max<int64_t>(1, 8 * steps_ * rows_));
    ctx.pool->parallel_for(words, grain, [&](int64_t w0, int64_t w1) { tile_bits_t<false>(0, rows_, 8 * w0, std::min(blocks, 8 * w1), yb); });
  }

  // Bits of blocks [b0, b1) (b0 a multiple of 8, i.e. word-aligned) for rows [r0, r1).
  template <bool kPair>
  void tile_bits_t(int64_t r0, int64_t r1, int64_t b0, int64_t b1, uint32_t* yb) const {
    constexpr int kTp = kPair ? 8 : 4;
    const auto* wts = static_cast<const uint8_t*>(wbuf_.get());
    const auto* xa = static_cast<const uint8_t*>(xbuf_.get());
    const int64_t steps = steps_, ow = bit_words(on_);
    const int64_t row_bytes = steps * 16;
    const int64_t last = kPair ? (r1 - 1) / 2 : r1 - 1;
    const int32x4_t kb = vdupq_n_s32(static_cast<int32_t>(k_));
    static const uint32_t kBitW[4] = {1, 2, 4, 8};
    const uint32x4_t bitw = vld1q_u32(kBitW);
    for (int64_t blk = b0; blk < b1; ++blk) {
      const int64_t o = blk * 4;
      const uint8_t* b = wts + blk * steps * 64;
      const int32x4_t lim = vld1q_s32(lim_.data() + o);
      const uint32x4_t neg = vreinterpretq_u32_s32(vld1q_s32(neg_.data() + o));
      const uint32_t valid = (1u << std::min<int64_t>(4, on_ - o)) - 1;
      auto store = [&](int64_t r, uint32x4_t pc) {
        const int32x4_t dot = vsubq_s32(kb, vshlq_n_s32(vreinterpretq_s32_u32(pc), 1));
        const uint32x4_t one = vbslq_u32(neg, vcgtq_s32(dot, lim), vcltq_s32(dot, lim));
        const uint32_t bits = vaddvq_u32(vandq_u32(bitw, one)) & valid;
        uint32_t* wp = yb + r * ow + (o >> 5);
        if ((o & 31) == 0) *wp = bits;  // first block of the word (blocks run in channel order)
        else *wp |= bits << (o & 31);
      };
      if constexpr (kPair) {
        for (int64_t r = r0; r < r1; r += kTp) {  // pair rows past the end are clamped, their outputs dropped
          const uint8_t* a[4];
          for (int i = 0; i < 4; ++i) a[i] = xa + std::min(r / 2 + i, last) * row_bytes;
          uint32x4_t pc[kTp];
          popcount_8x4(a[0], a[1], a[2], a[3], b, steps, pc);
          const int nv = static_cast<int>(std::min<int64_t>(kTp, r1 - r));
          for (int i = 0; i < nv; ++i) store(r + i, pc[i]);
        }
      } else {
        int64_t r = r0;
        for (; r + kTp <= r1; r += kTp) {
          const uint8_t* a = xa + r * row_bytes;
          uint32x4_t pc[kTp];
          popcount_4x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps, pc);
          for (int i = 0; i < kTp; ++i) store(r + i, pc[i]);
        }
        for (; r < r1; ++r) store(r, popcount_1x4(xa + r * row_bytes, b, steps));
      }
    }
  }

  // Binarizes / copies input rows [r0, r1) into xbuf_ (layout of the mode).
  void pack_rows(const KernelContext& ctx, int64_t r0, int64_t r1) const {
    auto* xb = static_cast<uint32_t*>(xbuf_.get());
    const bool f32 = ctx.in(0).dtype() == DType::kF32;
    const float* x = f32 ? ctx.in(0).data<float>() : nullptr;
    const uint32_t* xw = f32 ? nullptr : ctx.in(0).data<uint32_t>();
    const float* thr = opt_f32(ctx, 4);
    if (!pair_) {
      for (int64_t r = r0; r < r1; ++r) {
        if (f32) pack_row_f32_neon(x + r * k_, k_, xb + r * kwp_, mode_, thr);
        else std::copy_n(xw + r * words_, words_, xb + r * kwp_);
      }
      return;
    }
    uint32_t tmp[kMaxPairWords];
    tmp[kwp_ - 1] = 0;  // odd words_: padding half of the last 64-bit chunk
    for (int64_t r = r0; r < r1; ++r) {
      const uint32_t* src = tmp;
      if (f32) pack_row_f32_neon(x + r * k_, k_, tmp, mode_, thr);
      else if (words_ == kwp_) src = xw + r * words_;  // bit input, no padding word: interleaved straight from the row
      else std::copy_n(xw + r * words_, words_, tmp);
      uint32_t* pr = xb + (r >> 1) * 2 * kwp_ + (r & 1) * 2;
      for (int64_t h = 0; h < steps_; ++h) vst1_u32(pr + 4 * h, vld1_u32(src + 2 * h));
    }
  }

  // Tiled rows in two phases (as BConv2DNeonOpt::compute_dot2): per 16-channel unit the int16 dots of the row tile,
  // then the epilogue per row with the unit's constants in registers; channels beyond the last whole unit go
  // through tile_rows. Same operations per element as tile_t.
  template <bool kPair, typename T>
  void tile2_rows(int64_t r0, int64_t r1, T* y) const {
    switch (act_) {
      case Activation::kNone: return tile2<kPair, Activation::kNone>(r0, r1, y);
      case Activation::kRelu: return tile2<kPair, Activation::kRelu>(r0, r1, y);
      case Activation::kRelu6: return tile2<kPair, Activation::kRelu6>(r0, r1, y);
    }
  }
  template <bool kPair, Activation A, typename T>
  void tile2(int64_t r0, int64_t r1, T* y) const {
    constexpr bool kI8 = std::is_same<T, int8_t>::value;
    alignas(16) int16_t db[256 * 16];  // rt_ <= 256 rows
    const auto* wts = static_cast<const uint8_t*>(wbuf_.get());
    const auto* base = static_cast<const uint8_t*>(xbuf_.get()) + (kPair ? r0 / 2 : r0) * steps_ * 16;
    const int64_t np = r1 - r0, on16 = on_ / 16 * 16;
    for (int64_t u = 0; u < on16; u += 16) {
#if defined(__aarch64__)
      if (!kPair && t48_) dots_unit_rows8(wts, base, steps_, static_cast<int32_t>(k_), np, u, 16, db);
      else if (kPair && p84_) dots_unit_rows<true, true>(wts, base, steps_, static_cast<int32_t>(k_), np, u, 16, db);
      else
#endif
        dots_unit_rows<kPair>(wts, base, steps_, static_cast<int32_t>(k_), np, u, 16, db);
      int8_t* y8 = nullptr;
      float* yf = nullptr;
      if constexpr (kI8) {
        y8 = y + r0 * on_ + u;
      } else if (unit16_) {
        epi_.store_unit16(db, np, mul_.data() + u, add_.data() + u, static_cast<int>(A), y, on_, r0, u);
        continue;
      } else {
        yf = y + r0 * on_ + u;
      }
      dot2_store16<kI8, A>(db, np, mul_.data() + u, add_.data() + u, i8_out_.lo, i8_out_.hi, y8, yf, on_);
    }
    if (on16 < on_pad_) tile_rows<kPair>(r0, r1, on16 / 4, on_pad_ / 4, y);
  }

  // Outputs of blocks [b0, b1) for rows [r0, r1) (r0 a multiple of 8 in pair mode).
  template <bool kPair, typename T>
  void tile_rows(int64_t r0, int64_t r1, int64_t b0, int64_t b1, T* y) const {
    const bool sh = steps_ <= kU8Steps;
    switch (act_) {
      case Activation::kNone: return sh ? tile_t<kPair, true, Activation::kNone>(r0, r1, b0, b1, y)
                                        : tile_t<kPair, false, Activation::kNone>(r0, r1, b0, b1, y);
      case Activation::kRelu: return sh ? tile_t<kPair, true, Activation::kRelu>(r0, r1, b0, b1, y)
                                        : tile_t<kPair, false, Activation::kRelu>(r0, r1, b0, b1, y);
      case Activation::kRelu6: return sh ? tile_t<kPair, true, Activation::kRelu6>(r0, r1, b0, b1, y)
                                         : tile_t<kPair, false, Activation::kRelu6>(r0, r1, b0, b1, y);
    }
  }

  // kShort: steps_ <= kU8Steps (one uint8 accumulation run per tile).
  template <bool kPair, bool kShort, Activation A, typename T>
  void tile_t(int64_t r0, int64_t r1, int64_t b0, int64_t b1, T* y) const {
    constexpr bool kI8 = std::is_same<T, int8_t>::value;
    const int16x4_t lo4 = vdup_n_s16(static_cast<int16_t>(i8_out_.lo)), hi4 = vdup_n_s16(static_cast<int16_t>(i8_out_.hi));
    const int16x8_t lo8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.lo)), hi8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.hi));
    constexpr int kTp = kPair ? 8 : 4;
    const auto* wts = static_cast<const uint8_t*>(wbuf_.get());
    const auto* xa = static_cast<const uint8_t*>(xbuf_.get());
    const int64_t steps = steps_, on = on_;
    const int64_t row_bytes = steps * 16;  // a row, or a row pair in pair mode
    const int64_t last = kPair ? (r1 - 1) / 2 : r1 - 1;
    const int32x4_t kb = vdupq_n_s32(static_cast<int32_t>(k_));
    for (int64_t blk = b0; blk < b1; ++blk) {
      const int64_t o = blk * 4;
      const uint8_t* b = wts + blk * steps * 64;
      const float32x4_t mul = vld1q_f32(mul_.data() + o), add = vld1q_f32(add_.data() + o);
      const int nvo = static_cast<int>(std::min<int64_t>(4, on - o));
      auto value = [&](uint32x4_t pc) {  // dot = K - 2 * popcount, then the reference's float ops
        const int32x4_t dot = vsubq_s32(kb, vshlq_n_s32(vreinterpretq_s32_u32(pc), 1));
        return activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot), mul), add));
      };
      auto store = [&](T* yp, float32x4_t v) {
        if constexpr (kI8) {
          i8_store4(yp, v, lo4, hi4, nvo);
        } else if (nvo == 4) {
          vst1q_f32(yp, v);
        } else {
          float tmp[4];
          vst1q_f32(tmp, v);
          for (int j = 0; j < nvo; ++j) yp[j] = tmp[j];
        }
      };
      if constexpr (kPair) {
        for (int64_t r = r0; r < r1; r += kTp) {  // pair rows past the end are clamped, their outputs dropped
          const uint8_t* a[4];
          for (int i = 0; i < 4; ++i) a[i] = xa + std::min(r / 2 + i, last) * row_bytes;
          uint32x4_t pc[kTp];
          if constexpr (kShort) {
            const uint8_t* bb = b;  // run_4x4 advances its pointers
            Acc4x4 acc;
            run_4x4(a[0], a[1], a[2], a[3], bb, steps, acc);
            uint16x8_t t[4];
            for (int k = 0; k < 4; ++k) t[k] = reduce4_pair_n(acc.c[k][0], acc.c[k][1], acc.c[k][2], acc.c[k][3], steps);
            pair_rows_out(t, pc);
          } else {
            popcount_8x4(a[0], a[1], a[2], a[3], b, steps, pc);
          }
          T* yp = y + r * on + o;
          if (r + kTp <= r1 && nvo == 4) {
            if constexpr (kI8) {
              for (int i = 0; i < kTp; i += 4)
                i8_store4x4(yp + i * on, on, value(pc[i]), value(pc[i + 1]), value(pc[i + 2]), value(pc[i + 3]), lo8, hi8);
            } else {
              for (int i = 0; i < kTp; ++i) vst1q_f32(yp + i * on, value(pc[i]));
            }
          } else {
            const int nv = static_cast<int>(std::min<int64_t>(kTp, r1 - r));
            for (int i = 0; i < nv; ++i) store(yp + i * on, value(pc[i]));
          }
        }
      } else {
        int64_t r = r0;
        for (; r + kTp <= r1; r += kTp) {
          uint32x4_t pc[kTp];
          const uint8_t* a = xa + r * row_bytes;
          if constexpr (kShort) {
            const uint8_t *a0 = a, *a1 = a + row_bytes, *a2 = a + 2 * row_bytes, *a3 = a + 3 * row_bytes, *bb = b;
            Acc4x4 acc;
            run_4x4(a0, a1, a2, a3, bb, steps, acc);
            for (int i = 0; i < 4; ++i) pc[i] = reduce4_n(acc.c[i][0], acc.c[i][1], acc.c[i][2], acc.c[i][3], steps);
          } else {
            popcount_4x4(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, steps, pc);
          }
          T* yp = y + r * on + o;
          if (nvo == 4) {
            if constexpr (kI8) {
              i8_store4x4(yp, on, value(pc[0]), value(pc[1]), value(pc[2]), value(pc[3]), lo8, hi8);
            } else {
              for (int i = 0; i < kTp; ++i) vst1q_f32(yp + i * on, value(pc[i]));
            }
          } else {
            for (int i = 0; i < kTp; ++i) store(yp + i * on, value(pc[i]));
          }
        }
        for (; r < r1; ++r) store(y + r * on + o, value(popcount_1x4(xa + r * row_bytes, b, steps)));
      }
    }
  }

  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  bool tiled_ = false, pair_ = false, two_ = false, unit16_ = false, t48_ = false, p84_ = false;
  int64_t k_ = 0, on_ = 0, words_ = 0, kwp_ = 0, steps_ = 0, rows_ = 0, on_pad_ = 0, rt_ = 8, n_rt_ = 0;
  std::shared_ptr<void> wbuf_, xbuf_;
  std::shared_ptr<void> yf_;  // [rows][on] float output before binarization (bit output)
  std::vector<float> mul_, add_;
  BitOutput bout_;
  IntThresholds ithr_;
  uint64_t ithr_gen_ = ~uint64_t(0);
  std::vector<int32_t> lim_, neg_;  // integer thresholds [on_pad] (bit output without epilogue)
  EltwiseProgram epi_;
  I8Out i8_out_;
};

// sign_zero inputs go to bdense.ternary_neon. Fused epilogues are supported.
bool bdense_neon_supported(const KernelContext& ctx) { return ctx.attrs().get_string("input_binarize", "ge") != "sign_zero"; }

// ------------------------------------------------------------ ternary BDense
// sign_zero BDense ("bdense.ternary_neon"): x in {-1, 0, +1}, w = ±1 or ternary
// (w_zero), as nz / neg bit-planes (ternary_scalar/ternary_kernels.cc):
//   x·w = popcount(m) - 2·popcount(m & (neg_x ^ neg_w)),  m = nz_x & nz_w.
// Weight planes interleaved by 4 channels like bdense.neon; activation
// planes are rows padded to 128 bits (zero padding bits in both planes).

// pack_ternary_row_neon(): kernels/binary_neon/bitpack_neon.h (shared with bmatmul.neon).

// NR rows x 4 channels. kWz: per-channel popcount(m) as well (zero weights).
// pd[i] / pm[i]: lane j = channel j of the block.
template <int NR, bool kWz>
inline void ternary_rows4(const uint8_t* const* xn, const uint8_t* const* xs, const uint8_t* ws, const uint8_t* wn, int64_t steps,
                          uint32x4_t* pd, uint32x4_t* pm) {
  for (int i = 0; i < NR; ++i) pd[i] = pm[i] = vdupq_n_u32(0);
  for (int64_t s0 = 0; s0 < steps; s0 += kU8Steps) {
    const int64_t s1 = std::min(steps, s0 + kU8Steps);
    uint8x16_t ad[NR][4], am[NR][4];
    for (int i = 0; i < NR; ++i)
      for (int j = 0; j < 4; ++j) ad[i][j] = am[i][j] = vdupq_n_u8(0);
    for (int64_t s = s0; s < s1; ++s) {
      const uint8x16x4_t sw = vld1q_u8_x4(ws + 64 * s);
      uint8x16x4_t nw;
      if (kWz) nw = vld1q_u8_x4(wn + 64 * s);
      for (int i = 0; i < NR; ++i) {
        const uint8x16_t n = vld1q_u8(xn[i] + 16 * s), g = vld1q_u8(xs[i] + 16 * s);
        for (int j = 0; j < 4; ++j) {
          const uint8x16_t m = kWz ? vandq_u8(n, nw.val[j]) : n;
          ad[i][j] = vaddq_u8(ad[i][j], vcntq_u8(vandq_u8(m, veorq_u8(g, sw.val[j]))));
          if (kWz) am[i][j] = vaddq_u8(am[i][j], vcntq_u8(m));
        }
      }
    }
    for (int i = 0; i < NR; ++i) {
      pd[i] = vaddq_u32(pd[i], reduce4(ad[i][0], ad[i][1], ad[i][2], ad[i][3]));
      if (kWz) pm[i] = vaddq_u32(pm[i], reduce4(am[i][0], am[i][1], am[i][2], am[i][3]));
    }
  }
}

// 4 rows x 4 channels without zero weights: c[i][j] = Σ vcnt(nz_i AND
// (neg_i XOR wneg_j)) over n <= kU8Steps steps (inline asm like run_4x4: the
// first step sets the accumulators; 1 LD1 + 8 LDR + 64 EOR/AND/CNT/ADD + 2
// loop = 75 instructions per step). Accumulators v0-v15, nz rows v16-v19,
// neg rows v20-v23, weights v24-v27, temporaries v28-v31.
#if defined(__aarch64__)
#define BF_TSET4(N, G, C0, C1, C2, C3)                                                                    \
  "eor %[" C0 "].16b, " G ".16b, v24.16b\n eor %[" C1 "].16b, " G ".16b, v25.16b\n"                      \
  "eor %[" C2 "].16b, " G ".16b, v26.16b\n eor %[" C3 "].16b, " G ".16b, v27.16b\n"                      \
  "and %[" C0 "].16b, %[" C0 "].16b, " N ".16b\n and %[" C1 "].16b, %[" C1 "].16b, " N ".16b\n"          \
  "and %[" C2 "].16b, %[" C2 "].16b, " N ".16b\n and %[" C3 "].16b, %[" C3 "].16b, " N ".16b\n"          \
  "cnt %[" C0 "].16b, %[" C0 "].16b\n cnt %[" C1 "].16b, %[" C1 "].16b\n"                                \
  "cnt %[" C2 "].16b, %[" C2 "].16b\n cnt %[" C3 "].16b, %[" C3 "].16b\n"
#define BF_TROW4(N, G, C0, C1, C2, C3)                                                                    \
  "eor v28.16b, " G ".16b, v24.16b\n eor v29.16b, " G ".16b, v25.16b\n"                                 \
  "eor v30.16b, " G ".16b, v26.16b\n eor v31.16b, " G ".16b, v27.16b\n"                                 \
  "and v28.16b, v28.16b, " N ".16b\n and v29.16b, v29.16b, " N ".16b\n"                                 \
  "and v30.16b, v30.16b, " N ".16b\n and v31.16b, v31.16b, " N ".16b\n"                                 \
  "cnt v28.16b, v28.16b\n cnt v29.16b, v29.16b\n cnt v30.16b, v30.16b\n cnt v31.16b, v31.16b\n"          \
  "add %[" C0 "].16b, %[" C0 "].16b, v28.16b\n add %[" C1 "].16b, %[" C1 "].16b, v29.16b\n"              \
  "add %[" C2 "].16b, %[" C2 "].16b, v30.16b\n add %[" C3 "].16b, %[" C3 "].16b, v31.16b\n"
#define BF_TLOAD                                                                                          \
  "ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[w]], #64\n"                                               \
  "ldr q16, [%[n0]], #16\n ldr q17, [%[n1]], #16\n ldr q18, [%[n2]], #16\n ldr q19, [%[n3]], #16\n"        \
  "ldr q20, [%[g0]], #16\n ldr q21, [%[g1]], #16\n ldr q22, [%[g2]], #16\n ldr q23, [%[g3]], #16\n"
#define BF_TPTRS                                                                                          \
  [n0] "+r"(n0), [n1] "+r"(n1), [n2] "+r"(n2), [n3] "+r"(n3), [g0] "+r"(g0), [g1] "+r"(g1), [g2] "+r"(g2), \
  [g3] "+r"(g3), [w] "+r"(w)
inline __attribute__((always_inline)) void ternary_run_4x4(const uint8_t*& n0, const uint8_t*& n1, const uint8_t*& n2,
                                                           const uint8_t*& n3, const uint8_t*& g0, const uint8_t*& g1,
                                                           const uint8_t*& g2, const uint8_t*& g3, const uint8_t*& w, int64_t n,
                                                           Acc4x4& acc) {
  uint8x16_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
  asm volatile(BF_TLOAD
               BF_TSET4("v16", "v20", "c00", "c01", "c02", "c03") BF_TSET4("v17", "v21", "c10", "c11", "c12", "c13")
               BF_TSET4("v18", "v22", "c20", "c21", "c22", "c23") BF_TSET4("v19", "v23", "c30", "c31", "c32", "c33")
               : BF_TPTRS,
                 [c00] "=w"(c00), [c01] "=w"(c01), [c02] "=w"(c02), [c03] "=w"(c03), [c10] "=w"(c10), [c11] "=w"(c11),
                 [c12] "=w"(c12), [c13] "=w"(c13), [c20] "=w"(c20), [c21] "=w"(c21), [c22] "=w"(c22), [c23] "=w"(c23),
                 [c30] "=w"(c30), [c31] "=w"(c31), [c32] "=w"(c32), [c33] "=w"(c33)
               :
               : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                 "v29", "v30", "v31");
  int64_t rest = n - 1;
  if (rest > 0) {
    asm volatile("1:\n" BF_TLOAD
                 BF_TROW4("v16", "v20", "c00", "c01", "c02", "c03") BF_TROW4("v17", "v21", "c10", "c11", "c12", "c13")
                 BF_TROW4("v18", "v22", "c20", "c21", "c22", "c23") BF_TROW4("v19", "v23", "c30", "c31", "c32", "c33")
                 "subs %[n], %[n], #1\n"
                 "b.ne 1b\n"
                 : BF_TPTRS, [n] "+r"(rest),
                   [c00] "+w"(c00), [c01] "+w"(c01), [c02] "+w"(c02), [c03] "+w"(c03), [c10] "+w"(c10), [c11] "+w"(c11),
                   [c12] "+w"(c12), [c13] "+w"(c13), [c20] "+w"(c20), [c21] "+w"(c21), [c22] "+w"(c22), [c23] "+w"(c23),
                   [c30] "+w"(c30), [c31] "+w"(c31), [c32] "+w"(c32), [c33] "+w"(c33)
                 :
                 : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                   "v29", "v30", "v31");
  }
  acc = Acc4x4{{{c00, c01, c02, c03}, {c10, c11, c12, c13}, {c20, c21, c22, c23}, {c30, c31, c32, c33}}};
}
#undef BF_TSET4
#undef BF_TROW4
#undef BF_TLOAD
#undef BF_TPTRS
#else
// AArch32: two 2x4 runs as run_4x4. Accumulators q0-q3 / q4-q7, nz rows q8 / q9, neg rows q10 / q11, weights
// alternating in q12 / q15 (loaded one channel ahead), temporaries q13 / q14.
#define BF_A32_TSET(W, C0, C1)                                                                            \
  "veor " C0 ", q10, " W "\n veor " C1 ", q11, " W "\n vand " C0 ", " C0 ", q8\n vand " C1 ", " C1 ", q9\n"  \
  "vcnt.8 " C0 ", " C0 "\n vcnt.8 " C1 ", " C1 "\n"
#define BF_A32_TROW(W, C0, C1)                                                                            \
  "veor q13, q10, " W "\n veor q14, q11, " W "\n vand q13, q13, q8\n vand q14, q14, q9\n"                 \
  "vcnt.8 q13, q13\n vcnt.8 q14, q14\n vadd.i8 " C0 ", " C0 ", q13\n vadd.i8 " C1 ", " C1 ", q14\n"
#define BF_A32_TLOAD                                                                                      \
  "vld1.8 {d16-d17}, [%[n0]]!\n vld1.8 {d18-d19}, [%[n1]]!\n vld1.8 {d20-d21}, [%[g0]]!\n"                \
  "vld1.8 {d22-d23}, [%[g1]]!\n vld1.8 {d24-d25}, [%[w]]!\n vld1.8 {d30-d31}, [%[w]]!\n"
#define BF_A32_TSTEP(OP)                                                                                  \
  BF_A32_TLOAD OP("q12", "q0", "q4") "vld1.8 {d24-d25}, [%[w]]!\n" OP("q15", "q1", "q5")                   \
  "vld1.8 {d30-d31}, [%[w]]!\n" OP("q12", "q2", "q6") OP("q15", "q3", "q7")

inline __attribute__((always_inline)) void ternary_run_2x4_a32(const uint8_t* n0, const uint8_t* n1, const uint8_t* g0,
                                                               const uint8_t* g1, const uint8_t* w, int n, uint8x16_t* out) {
  uint8_t* o = reinterpret_cast<uint8_t*>(out);
  asm volatile(BF_A32_TSTEP(BF_A32_TSET)
               "subs %[n], %[n], #1\n"
               "beq 2f\n"
               "1:\n" BF_A32_TSTEP(BF_A32_TROW)
               "subs %[n], %[n], #1\n"
               "bne 1b\n"
               "2:\n" BF_A32_STORE8
               : [n0] "+r"(n0), [n1] "+r"(n1), [g0] "+r"(g0), [g1] "+r"(g1), [w] "+r"(w), [n] "+r"(n), [o] "+r"(o)
               :
               : BF_A32_CLOBBERS);
}
#undef BF_A32_TSET
#undef BF_A32_TROW
#undef BF_A32_TLOAD
#undef BF_A32_TSTEP
#undef BF_A32_STORE8
#undef BF_A32_CLOBBERS

inline __attribute__((always_inline)) void ternary_run_4x4(const uint8_t*& n0, const uint8_t*& n1, const uint8_t*& n2,
                                                           const uint8_t*& n3, const uint8_t*& g0, const uint8_t*& g1,
                                                           const uint8_t*& g2, const uint8_t*& g3, const uint8_t*& w, int64_t n,
                                                           Acc4x4& acc) {
  ternary_run_2x4_a32(n0, n1, g0, g1, w, static_cast<int>(n), &acc.c[0][0]);
  ternary_run_2x4_a32(n2, n3, g2, g3, w, static_cast<int>(n), &acc.c[2][0]);
  n0 += 16 * n;
  n1 += 16 * n;
  n2 += 16 * n;
  n3 += 16 * n;
  g0 += 16 * n;
  g1 += 16 * n;
  g2 += 16 * n;
  g3 += 16 * n;
  w += 64 * n;
}
#endif

// pd[i] lane j = popcount(nz_i AND (neg_i XOR wneg_j)) over `steps` steps.
inline __attribute__((always_inline)) void ternary_popcount_4x4(const uint8_t* const* xn, const uint8_t* const* xs, const uint8_t* ws,
                                                                int64_t steps, uint32x4_t* pd) {
  const uint8_t *n0 = xn[0], *n1 = xn[1], *n2 = xn[2], *n3 = xn[3], *g0 = xs[0], *g1 = xs[1], *g2 = xs[2], *g3 = xs[3], *w = ws;
  if (steps <= kU8Steps) {  // one uint8 run
    Acc4x4 acc;
    ternary_run_4x4(n0, n1, n2, n3, g0, g1, g2, g3, w, steps, acc);
    for (int i = 0; i < 4; ++i) pd[i] = reduce4_n(acc.c[i][0], acc.c[i][1], acc.c[i][2], acc.c[i][3], steps);
    return;
  }
  for (int i = 0; i < 4; ++i) pd[i] = vdupq_n_u32(0);
  for (int64_t s = 0; s < steps; s += kU8Steps) {
    Acc4x4 acc;
    const int64_t n = std::min(kU8Steps, steps - s);
    ternary_run_4x4(n0, n1, n2, n3, g0, g1, g2, g3, w, n, acc);
    for (int i = 0; i < 4; ++i) pd[i] = vaddq_u32(pd[i], reduce4_n(acc.c[i][0], acc.c[i][1], acc.c[i][2], acc.c[i][3], n));
  }
}

// Popcount of a row of `words` words (multiple of 4): popcount(nz_x).
inline int32_t row_popcount(const uint32_t* row, int64_t words) {
  const auto* p = reinterpret_cast<const uint8_t*>(row);
  int32_t s = 0;
  for (int64_t q = 0; q < words;) {
    uint8x16_t acc = vdupq_n_u8(0);
    for (const int64_t e = std::min(words, q + 4 * kU8Steps); q < e; q += 4) acc = vaddq_u8(acc, vcntq_u8(vld1q_u8(p + 4 * q)));
    s += static_cast<int32_t>(vaddlvq_u8(acc));
  }
  return s;
}

class BDenseTernaryNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    const Tensor& w = ctx.in(1);
    on_ = w.dim(0);
    k_ = w.dim(1);
    words_ = bit_words(k_);
    kwp_ = (words_ + 3) / 4 * 4;
    rows_ = ctx.in(0).size() / k_;
    on_pad_ = (on_ + 3) / 4 * 4;
    has_wz_ = ctx.has_in(5);
    i8_in_ = ctx.in(0).dtype() == DType::kI8;  // int8_full: thresholds in(4) = int32 [tge, tgt] (1 or k each)
    i8_out_.parse(ctx);
#if defined(__aarch64__)
    // fp32 in / out, weights without zeros, 8-channel blocks: the weight signs interleaved by 8 channels (the 4x8
    // tile's layout; same size). A 4-row tile whose rows have no zero input (popcount(nz) == K: x·w = K -
    // 2 popcount(neg_x ^ neg_w), the binary dot product) runs popcount_4x8_u16; rows with zeros run one at a time
    // (ternary_row8). BITINO_BCONV_T48=0 keeps the 4x4 tiles.
    // (int8_full: int8 codes in and / or int8 out as well, 2026-11-13: the same integer dots and fp32 values)
    fast_ = !has_wz_ && on_ % 8 == 0 && k_ <= 65535 && !opts_t48_off();
#endif
    wneg_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * kwp_ * 4));
#if defined(__aarch64__)
    if (fast_) interleave_rows8(w.data<uint32_t>(), on_, words_, kwp_, static_cast<uint32_t*>(wneg_.get()));
    else
#endif
      interleave_rows4(w.data<uint32_t>(), on_, words_, words_, kwp_, static_cast<uint32_t*>(wneg_.get()));
    if (has_wz_) {
      // nz = NOT zero-mask on the k valid bits; padding bits stay 0.
      std::vector<uint32_t> nzw(static_cast<size_t>(on_ * words_));
      const uint32_t* wz = ctx.in(5).data<uint32_t>();
      for (int64_t o = 0; o < on_; ++o)
        for (int64_t q = 0; q < words_; ++q) {
          const int64_t valid = k_ - q * 32;
          nzw[o * words_ + q] = ~wz[o * words_ + q] & (valid >= 32 ? ~0u : ((1u << valid) - 1));
        }
      wnz_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * kwp_ * 4));
      interleave_rows4(nzw.data(), on_, words_, words_, kwp_, static_cast<uint32_t*>(wnz_.get()));
    }
    xnz_ = aligned_alloc_shared(static_cast<size_t>(rows_ * kwp_ * 4));  // padding words stay 0
    xneg_ = aligned_alloc_shared(static_cast<size_t>(rows_ * kwp_ * 4));
    pm_row_.assign(static_cast<size_t>(rows_), 0);
    mul_.assign(static_cast<size_t>(on_pad_), 1.f);
    add_.assign(static_cast<size_t>(on_pad_), -0.f);
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(on_pad_ * kwp_ * 4 * (has_wz_ ? 2 : 1)); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    auto* xn = static_cast<uint32_t*>(xnz_.get());
    auto* xs = static_cast<uint32_t*>(xneg_.get());
    const int64_t k = k_, kwp = kwp_;
    if (i8_in_) {
      const int8_t* q = ctx.in(0).data<int8_t>();
      const int32_t* t = ctx.in(4).data<int32_t>();
      const int64_t nt = ctx.in(4).size() / 2;
      ctx.pool->parallel_for(rows_, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(k, 1)), [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          pack_ternary_row_i8(q + r * k, k, t, t + nt, nt > 1, xn + r * kwp, xs + r * kwp);
          pm_row_[r] = row_popcount(xn + r * kwp, kwp);
        }
      });
    } else {
      const float* x = ctx.in(0).data<float>();
      const float* thr = opt_f32(ctx, 4);
      ctx.pool->parallel_for(rows_, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(k, 1)), [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          pack_ternary_row_neon(x + r * k, k, xn + r * kwp, xs + r * kwp, thr);
          pm_row_[r] = row_popcount(xn + r * kwp, kwp);  // popcount(nz_x): x·w for w without zero entries uses it
        }
      });
    }
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, on_, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, on_, add_.data());
    const int64_t blocks = on_pad_ / 4, steps = kwp_ / 4;
    const auto* ws = static_cast<const uint8_t*>(wneg_.get());
    const auto* wn = static_cast<const uint8_t*>(wnz_.get());
    void* y = ctx.out(0).raw();
#if defined(__aarch64__)
    if (fast_) {
      const int64_t blocks8 = on_ / 8;
      const int64_t grain8 = std::max<int64_t>(1, 8192 / std::max<int64_t>(1, 2 * steps * rows_));
      ctx.pool->parallel_for(blocks8, grain8, [&](int64_t b0, int64_t b1) {
        for (int64_t blk = b0; blk < b1; ++blk) {
          if (i8_out_.on) {
            switch (act_) {
              case Activation::kNone: block8<Activation::kNone>(blk, ws, steps, static_cast<int8_t*>(y)); break;
              case Activation::kRelu: block8<Activation::kRelu>(blk, ws, steps, static_cast<int8_t*>(y)); break;
              case Activation::kRelu6: block8<Activation::kRelu6>(blk, ws, steps, static_cast<int8_t*>(y)); break;
            }
            continue;
          }
          switch (act_) {
            case Activation::kNone: block8<Activation::kNone>(blk, ws, steps, static_cast<float*>(y)); break;
            case Activation::kRelu: block8<Activation::kRelu>(blk, ws, steps, static_cast<float*>(y)); break;
            case Activation::kRelu6: block8<Activation::kRelu6>(blk, ws, steps, static_cast<float*>(y)); break;
          }
        }
      });
      return;
    }
#endif
    const int64_t grain = std::max<int64_t>(1, 8192 / std::max<int64_t>(1, steps * rows_ * (has_wz_ ? 2 : 1)));
    ctx.pool->parallel_for(blocks, grain, [&](int64_t b0, int64_t b1) {
      if (i8_out_.on) {
        switch (act_) {
          case Activation::kNone: return blocks_a<Activation::kNone, int8_t>(b0, b1, ws, wn, steps, static_cast<int8_t*>(y));
          case Activation::kRelu: return blocks_a<Activation::kRelu, int8_t>(b0, b1, ws, wn, steps, static_cast<int8_t*>(y));
          case Activation::kRelu6: return blocks_a<Activation::kRelu6, int8_t>(b0, b1, ws, wn, steps, static_cast<int8_t*>(y));
        }
      }
      switch (act_) {
        case Activation::kNone: return blocks_a<Activation::kNone, float>(b0, b1, ws, wn, steps, static_cast<float*>(y));
        case Activation::kRelu: return blocks_a<Activation::kRelu, float>(b0, b1, ws, wn, steps, static_cast<float*>(y));
        case Activation::kRelu6: return blocks_a<Activation::kRelu6, float>(b0, b1, ws, wn, steps, static_cast<float*>(y));
      }
    });
  }

 private:
  template <Activation A, typename T>
  void blocks_a(int64_t b0, int64_t b1, const uint8_t* ws, const uint8_t* wn, int64_t steps, T* y) const {
    for (int64_t blk = b0; blk < b1; ++blk) {
      if (has_wz_) block<true, A>(blk, ws, wn, steps, y);
      else block<false, A>(blk, ws, wn, steps, y);
    }
  }

  // Without zero weights 4-row tiles run the asm micro-kernel (ternary_run_4x4).
  template <bool kWz, Activation A, typename T>
  void block(int64_t blk, const uint8_t* ws, const uint8_t* wn, int64_t steps, T* y) const {
    constexpr int kRows = kWz ? 2 : 4;  // 16 or 32 byte accumulators
    const int64_t o = blk * 4;
    const uint8_t* bs = ws + blk * steps * 64;
    const uint8_t* bn = kWz ? wn + blk * steps * 64 : nullptr;
    const float32x4_t mul = vld1q_f32(mul_.data() + o), add = vld1q_f32(add_.data() + o);
    const int nvo = static_cast<int>(std::min<int64_t>(4, on_ - o));
    const auto* xa = static_cast<const uint8_t*>(xnz_.get());
    const auto* xb = static_cast<const uint8_t*>(xneg_.get());
    const int32_t* pmr = pm_row_.data();
    const int64_t on = on_, rows = rows_, rb = kwp_ * 4;  // locals: the output stores cannot alias them
    const int16x4_t lo4 = vdup_n_s16(static_cast<int16_t>(i8_out_.lo)), hi4 = vdup_n_s16(static_cast<int16_t>(i8_out_.hi));
    auto value = [&](int64_t r, uint32x4_t pd, uint32x4_t pm) {
      const int32x4_t pmv = kWz ? vreinterpretq_s32_u32(pm) : vdupq_n_s32(pmr[r]);
      const int32x4_t dot = vsubq_s32(pmv, vshlq_n_s32(vreinterpretq_s32_u32(pd), 1));
      return activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(dot), mul), add));
    };
    const int16x8_t lo8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.lo)), hi8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.hi));
    auto emit = [&](int64_t r, uint32x4_t pd, uint32x4_t pm) {
      const float32x4_t v = value(r, pd, pm);
      T* yp = y + r * on + o;
      if constexpr (std::is_same<T, int8_t>::value) {
        i8_store4(yp, v, lo4, hi4, nvo);
      } else if (nvo == 4) {
        vst1q_f32(yp, v);
      } else {
        float tmp[4];
        vst1q_f32(tmp, v);
        for (int j = 0; j < nvo; ++j) yp[j] = tmp[j];
      }
    };
    int64_t r = 0;
    for (; r + kRows <= rows; r += kRows) {
      const uint8_t *pn[kRows], *ps[kRows];
      for (int i = 0; i < kRows; ++i) {
        pn[i] = xa + (r + i) * rb;
        ps[i] = xb + (r + i) * rb;
      }
      uint32x4_t pd[kRows], pm[kRows];
      if constexpr (kWz) ternary_rows4<kRows, kWz>(pn, ps, bs, bn, steps, pd, pm);
      else ternary_popcount_4x4(pn, ps, bs, steps, pd);
      if constexpr (std::is_same<T, int8_t>::value && kRows == 4) {
        if (nvo == 4) {
          i8_store4x4(y + r * on + o, on, value(r, pd[0], pd[0]), value(r + 1, pd[1], pd[1]), value(r + 2, pd[2], pd[2]),
                      value(r + 3, pd[3], pd[3]), lo8, hi8);
          continue;
        }
      }
      for (int i = 0; i < kRows; ++i) emit(r + i, pd[i], kWz ? pm[i] : pd[i]);
    }
    for (; r < rows; ++r) {
      const uint8_t* pn[1] = {xa + r * rb};
      const uint8_t* ps[1] = {xb + r * rb};
      uint32x4_t pd[1], pm[1];
      ternary_rows4<1, kWz>(pn, ps, bs, bn, steps, pd, pm);
      emit(r, pd[0], pm[0]);
    }
  }

#if defined(__aarch64__)
  // popcount(nz_x & (neg_x ^ neg_w)) of one row and 8 channels (fast_ layout): uint16 [ch 0..7]
  static uint16x8_t ternary_row8(const uint8_t* n, const uint8_t* g, const uint8_t* b, int64_t steps) {
    uint16x8_t acc = vdupq_n_u16(0);
    for (int64_t s = 0; s < steps; ++s, n += 16, g += 16, b += 128) {
      const uint8x16_t nz = vld1q_u8(n), ng = vld1q_u8(g);
      const uint8x16x4_t w0 = vld1q_u8_x4(b), w1 = vld1q_u8_x4(b + 64);
      auto pc = [&](uint8x16_t w) { return vcntq_u8(vandq_u8(nz, veorq_u8(ng, w))); };
      const uint8x16_t p0 = vpaddq_u8(pc(w0.val[0]), pc(w0.val[1])), p1 = vpaddq_u8(pc(w0.val[2]), pc(w0.val[3]));
      const uint8x16_t p2 = vpaddq_u8(pc(w1.val[0]), pc(w1.val[1])), p3 = vpaddq_u8(pc(w1.val[2]), pc(w1.val[3]));
      acc = vpadalq_u8(acc, vpaddq_u8(vpaddq_u8(p0, p1), vpaddq_u8(p2, p3)));
    }
    return acc;
  }
  // The same for 4 rows (tiles with zero inputs), each step in two halves of 4 channels.
  static void ternary_rows4x8(const uint8_t* const* n, const uint8_t* const* g, const uint8_t* b, int64_t steps, uint16x8_t out[4]) {
    uint16x8_t acc[4] = {vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0)};
    for (int64_t s = 0; s < steps; ++s, b += 128) {
      uint8x16_t nz[4], ng[4], h[2][4];
      for (int r = 0; r < 4; ++r) {
        nz[r] = vld1q_u8(n[r] + 16 * s);
        ng[r] = vld1q_u8(g[r] + 16 * s);
      }
      for (int hh = 0; hh < 2; ++hh) {
        const uint8x16x4_t w = vld1q_u8_x4(b + 64 * hh);
        for (int r = 0; r < 4; ++r) {
          auto pc = [&](uint8x16_t wj) { return vcntq_u8(vandq_u8(nz[r], veorq_u8(ng[r], wj))); };
          h[hh][r] = vpaddq_u8(vpaddq_u8(pc(w.val[0]), pc(w.val[1])), vpaddq_u8(pc(w.val[2]), pc(w.val[3])));
        }
      }
      for (int r = 0; r < 4; ++r) acc[r] = vpadalq_u8(acc[r], vpaddq_u8(h[0][r], h[1][r]));
    }
    for (int r = 0; r < 4; ++r) out[r] = acc[r];
  }
  // Output channels [8 blk, 8 blk + 8) of every row (fast_): v = act(fp32(pm - 2 pd) * mul + add) as block(); int8
  // output (int8_full): clamp(rne(v)) by i8_pack4's operations.
  template <Activation A, typename T>
  void block8(int64_t blk, const uint8_t* ws, int64_t steps, T* y) const {
    const int64_t o = blk * 8;
    const uint8_t* b = ws + blk * steps * 128;
    const float32x4_t mul0 = vld1q_f32(mul_.data() + o), mul1 = vld1q_f32(mul_.data() + o + 4);
    const float32x4_t add0 = vld1q_f32(add_.data() + o), add1 = vld1q_f32(add_.data() + o + 4);
    const auto* xa = static_cast<const uint8_t*>(xnz_.get());
    const auto* xb = static_cast<const uint8_t*>(xneg_.get());
    const int32_t* pmr = pm_row_.data();
    const int64_t on = on_, rows = rows_, rb = kwp_ * 4, kfull = k_;
    const int16x8_t lo8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.lo)), hi8 = vdupq_n_s16(static_cast<int16_t>(i8_out_.hi));
    auto emit = [&](int64_t r, uint16x8_t pd) {
      const int32x4_t pm = vdupq_n_s32(pmr[r]);
      const int32x4_t d0 = vsubq_s32(pm, vshlq_n_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(pd))), 1));
      const int32x4_t d1 = vsubq_s32(pm, vshlq_n_s32(vreinterpretq_s32_u32(vmovl_high_u16(pd)), 1));
      T* yp = y + r * on + o;
      const float32x4_t v0 = activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(d0), mul0), add0));
      const float32x4_t v1 = activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(d1), mul1), add1));
      if constexpr (std::is_same<T, int8_t>::value) {
        int16x8_t h = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(v0)), vqmovn_s32(vcvtnq_s32_f32(v1)));
        h = vminq_s16(vmaxq_s16(h, lo8), hi8);
        vst1_s8(yp, vmovn_s16(h));
      } else {
        vst1q_f32(yp, v0);
        vst1q_f32(yp + 4, v1);
      }
    };
    auto row = [&](int64_t r) {
      emit(r, pmr[r] == kfull ? popcount_1x8_u16(xb + r * rb, b, steps) : ternary_row8(xa + r * rb, xb + r * rb, b, steps));
    };
    int64_t r = 0;
    for (; r + 4 <= rows; r += 4) {
      if (pmr[r] == kfull && pmr[r + 1] == kfull && pmr[r + 2] == kfull && pmr[r + 3] == kfull) {
        const uint8_t* a = xb + r * rb;
        uint16x8_t t[4];
        popcount_4x8_u16(a, a + rb, a + 2 * rb, a + 3 * rb, b, steps, t);
        for (int i = 0; i < 4; ++i) emit(r + i, t[i]);
      } else {
        const uint8_t* n4[4] = {xa + r * rb, xa + (r + 1) * rb, xa + (r + 2) * rb, xa + (r + 3) * rb};
        const uint8_t* g4[4] = {xb + r * rb, xb + (r + 1) * rb, xb + (r + 2) * rb, xb + (r + 3) * rb};
        uint16x8_t t[4];
        ternary_rows4x8(n4, g4, b, steps, t);
        for (int i = 0; i < 4; ++i) emit(r + i, t[i]);
      }
    }
    for (; r < rows; ++r) row(r);
  }
#endif

  Activation act_ = Activation::kNone;
  int64_t on_ = 0, k_ = 0, words_ = 0, kwp_ = 0, rows_ = 0, on_pad_ = 0;
  bool has_wz_ = false, i8_in_ = false, fast_ = false;
  I8Out i8_out_;
  std::shared_ptr<void> wneg_, wnz_, xnz_, xneg_;
  std::vector<int32_t> pm_row_;
  std::vector<float> mul_, add_;
};

bool bdense_ternary_neon_supported(const KernelContext& ctx) {
  return ctx.in(0).dtype() == DType::kF32 && ctx.attrs().get_string("input_binarize", "ge") == "sign_zero" && !has_epilogue(ctx) &&
         ctx.attrs().get_string("output", "f32") == "f32";
}


// ------------------------------------------------- QBConv2D / QBDense (NEON)
// DoReFa k-bit activations x binary weights (docs/operators_larq.md), exact
// integers: with level = Σ_b 2^b·bit_b and w = ±1 (bit 1 = -1),
//   Σ level·w = X − 2·D,  X = Σ_b 2^b·popcount(bit_b),  D = Σ_b 2^b·popcount(bit_b AND neg_w).
// The activations become k bit-planes; each (pixel or row, plane) is one row
// of the BConv2D micro-kernel run with AND instead of XOR. Zero padding of the
// quantized input is zero bits in every plane: no border correction (QBConv2D
// requires pad_value 0). Epilogue as the reference: float(acc) * post_mul +
// post_add (defaults 1 and -0 are exact identities), then the activation.

// Bit-planes of the DoReFa levels of `c` floats: level = nearbyint(clip(x, 0,
// 1) * n) with one float32 multiply, exactly dorefa_level() of the reference
// (std::max / std::min semantics with NaN; FCVTNS rounds half to even like
// nearbyint and maps NaN to 0 like the reference's float -> int conversion).
// Plane b of word w goes to out[b * plane_stride + w].
inline void pack_levels_row_neon(const float* x, int64_t c, float n, int planes, uint32_t* out, int64_t plane_stride) {
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t wts = vld1q_u8(kW);
  const float32x4_t zero = vdupq_n_f32(0.f), one = vdupq_n_f32(1.f), nv = vdupq_n_f32(n);
  auto word = [&](const float* xp, int64_t w) {
    int32x4_t l[8];
    for (int k = 0; k < 8; ++k) {
      const float32x4_t v = vld1q_f32(xp + 4 * k);
      const float32x4_t m = vbslq_f32(vcltq_f32(v, zero), zero, v);  // std::max(v, 0)
      const float32x4_t cl = vbslq_f32(vcltq_f32(one, m), one, m);   // std::min(m, 1)
      l[k] = vcvtnq_s32_f32(vmulq_f32(cl, nv));
    }
    for (int b = 0; b < planes; ++b) {
      const int32x4_t bit = vdupq_n_s32(1 << b);
      uint32x4_t m[8];
      for (int k = 0; k < 8; ++k) m[k] = vtstq_s32(l[k], bit);
      const uint8x16_t b0 = vandq_u8(wts, narrow_masks(m[0], m[1], m[2], m[3]));
      const uint8x16_t b1 = vandq_u8(wts, narrow_masks(m[4], m[5], m[6], m[7]));
      uint8x16_t r = vpaddq_u8(b0, b1);
      r = vpaddq_u8(r, r);
      r = vpaddq_u8(r, r);
      out[b * plane_stride + w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 0);
    }
  };
  const int64_t full = c / 32;
  for (int64_t w = 0; w < full; ++w) word(x + 32 * w, w);
  if (c == full * 32) return;
  alignas(16) float tmp[32] = {};  // tail: zero-padded copy (level(0) = 0)
  std::copy_n(x + 32 * full, c - 32 * full, tmp);
  word(tmp, full);
}

// Sum over rows of popcount over `words` words, rows `stride` words apart,
// weighted 2^b for row b (the X term of `planes` plane rows).
inline int32_t weighted_popcount(const uint32_t* row, int64_t words, int64_t stride, int planes) {
  int32_t x = 0;
  for (int b = 0; b < planes; ++b) {
    const auto* p = reinterpret_cast<const uint8_t*>(row + b * stride);
    int32_t s = 0;
    int64_t q = 0;
    for (; q + 4 <= words;) {  // uint8 lanes: at most 31 x 8 per lane before folding
      uint8x16_t acc = vdupq_n_u8(0);
      for (int64_t e = std::min(words, q + 4 * kU8Steps); q + 4 <= e; q += 4) acc = vaddq_u8(acc, vcntq_u8(vld1q_u8(p + 4 * q)));
      s += static_cast<int32_t>(vaddlvq_u8(acc));
    }
    for (; q < words; ++q) s += popcount32(row[b * stride + q]);
    x += s << b;
  }
  return x;
}

// Shared tile computation: rows [0, nrows) of `rows` (row r = item r / planes,
// plane r % planes; kwp words each) against output blocks [b0, b1).
// `xsum[i]` is the X term of item i; outputs y[(q0 + i) * y_stride + o].
struct QTile {
  const uint8_t* wts;  // interleave_rows4 of the neg_w rows (kwp words)
  int64_t steps, kwp, oc;
  int planes;
  const float* mul;
  const float* add;
  Activation act;
};

template <Activation A>
void qtile_rows(const QTile& t, const uint32_t* rows, const int32_t* xsum, int64_t items, int64_t b0, int64_t b1, float* y,
                int64_t y_stride, uint32x4_t* pcr) {
  const int64_t nrows = items * t.planes, row_bytes = t.kwp * 4;
  const auto* base = reinterpret_cast<const uint8_t*>(rows);
  for (int64_t blk = b0; blk < b1; ++blk) {
    const int64_t o = blk * 4;
    const uint8_t* b = t.wts + blk * t.steps * 64;
    const float32x4_t mul = vld1q_f32(t.mul + o), add = vld1q_f32(t.add + o);
    const int nvo = static_cast<int>(std::min<int64_t>(4, t.oc - o));
    int64_t r = 0;
    for (; r + 4 <= nrows; r += 4) {
      const uint8_t* a = base + r * row_bytes;
      popcount_4x4<true>(a, a + row_bytes, a + 2 * row_bytes, a + 3 * row_bytes, b, t.steps, pcr + r);
    }
    for (; r < nrows; ++r) pcr[r] = popcount_1x4<true>(base + r * row_bytes, b, t.steps);
    for (int64_t i = 0; i < items; ++i) {
      uint32x4_t d = pcr[i * t.planes];
      for (int pb = 1; pb < t.planes; ++pb) d = vaddq_u32(d, vshlq_u32(pcr[i * t.planes + pb], vdupq_n_s32(pb)));
      const int32x4_t acc = vsubq_s32(vdupq_n_s32(xsum[i]), vshlq_n_s32(vreinterpretq_s32_u32(d), 1));
      const float32x4_t v = activate(A, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc), mul), add));
      float* yp = y + i * y_stride + o;
      if (nvo == 4) {
        vst1q_f32(yp, v);
      } else {
        float tmp[4];
        vst1q_f32(tmp, v);
        for (int j = 0; j < nvo; ++j) yp[j] = tmp[j];
      }
    }
  }
}

inline void qtile(const QTile& t, const uint32_t* rows, const int32_t* xsum, int64_t items, int64_t b0, int64_t b1, float* y,
                  int64_t y_stride, uint32x4_t* pcr) {
  switch (t.act) {
    case Activation::kNone: return qtile_rows<Activation::kNone>(t, rows, xsum, items, b0, b1, y, y_stride, pcr);
    case Activation::kRelu: return qtile_rows<Activation::kRelu>(t, rows, xsum, items, b0, b1, y, y_stride, pcr);
    case Activation::kRelu6: return qtile_rows<Activation::kRelu6>(t, rows, xsum, items, b0, b1, y, y_stride, pcr);
  }
}

constexpr int kMaxQNeonPlanes = 8;
constexpr int64_t kQItems = 64;  // items (pixels / rows) per tile

float qneon_levels(const KernelContext& ctx) { return static_cast<float>((1 << ctx.attrs().get_int("act_bits", 0)) - 1); }

class QBConv2DNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Json& a = ctx.attrs();
    g_ = conv_geometry(a, ctx.in(0).shape(), ctx.in(1).shape());
    L_.init(g_);  // groups == 1 (supports()): 32-bit units, or 16-bit units when ic % 32 == 16
    act_ = parse_activation(a);
    planes_ = static_cast<int>(a.get_int("act_bits", 0));
    n_ = qneon_levels(ctx);
    kwp_ = (L_.k_words + 3) / 4 * 4;
    oc_pad_ = (g_.oc + 3) / 4 * 4;
    pixels_ = g_.n * g_.oh * g_.ow;
    std::vector<uint32_t> dense(static_cast<size_t>(g_.oc * kwp_));
    L_.pack_weight_rows(ctx.in(1).data<uint32_t>(), g_.oc, dense.data(), kwp_);
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(oc_pad_ * kwp_ * 4));
    interleave_rows4(dense.data(), g_.oc, kwp_, kwp_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    mul_.assign(static_cast<size_t>(oc_pad_), 1.f);
    add_.assign(static_cast<size_t>(oc_pad_), -0.f);
    in_pix_ = g_.n * g_.ih * g_.iw;
    xplanes_ = aligned_alloc_shared(static_cast<size_t>(planes_ * in_pix_ * L_.src_words * 4));
    // Pixel tile: ~16 KB of plane rows (whole micro-tiles of 4 rows when planes divides 4).
    mt_ = std::max<int64_t>(4, std::min<int64_t>(kQItems, 16384 / (planes_ * kwp_ * 4) / 4 * 4));
    n_ptiles_ = (pixels_ + mt_ - 1) / mt_;
    const int64_t threads = ctx.pool->num_threads();
    n_tasks_ = threads > 1 ? std::min<int64_t>(n_ptiles_, 4 * threads) : 1;
    slot_words_ = mt_ * planes_ * kwp_;
    patches_ = aligned_alloc_shared(static_cast<size_t>(n_tasks_ * slot_words_ * 4));  // row tails stay 0
    xsum_.assign(static_cast<size_t>(n_tasks_ * mt_), 0);
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(oc_pad_ * kwp_ * 4); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    auto* xp = static_cast<uint32_t*>(xplanes_.get());
    const int64_t ic = g_.ic, sw = L_.src_words, ps = in_pix_ * sw;
    const float n = n_;
    const int planes = planes_;
    ctx.pool->parallel_for(in_pix_, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(ic, 1)), [&](int64_t p0, int64_t p1) {
      for (int64_t p = p0; p < p1; ++p) pack_levels_row_neon(x + p * ic, ic, n, planes, xp + p * sw, ps);
    });
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, g_.oc, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, g_.oc, add_.data());
    QTile t{static_cast<const uint8_t*>(wbuf_.get()), kwp_ / 4, kwp_, g_.oc, planes_, mul_.data(), add_.data(), act_};
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(n_tasks_, 1, [&](int64_t t0, int64_t t1) {
      for (int64_t task = t0; task < t1; ++task) run_task(t, xp, y, task);
    });
  }

 private:
  void run_task(const QTile& t, const uint32_t* xp, float* y, int64_t task) {
    uint32_t* rows = static_cast<uint32_t*>(patches_.get()) + task * slot_words_;
    int32_t* xs = xsum_.data() + task * mt_;
    const int64_t plane_words = in_pix_ * L_.src_words;
    uint32x4_t pcr[kQItems * kMaxQNeonPlanes];
    for (int64_t pt = task * n_ptiles_ / n_tasks_; pt < (task + 1) * n_ptiles_ / n_tasks_; ++pt) {
      const int64_t p0 = pt * mt_, p1 = std::min(pixels_, p0 + mt_);
      int64_t ox = p0 % g_.ow, oy = (p0 / g_.ow) % g_.oh, n = p0 / (g_.ow * g_.oh);
      for (int64_t p = p0; p < p1; ++p) {
        uint32_t* r = rows + (p - p0) * planes_ * kwp_;
        for (int b = 0; b < planes_; ++b)
          build_patch_row(g_, L_, reinterpret_cast<const uint8_t*>(xp + b * plane_words), 0, n, oy, ox, reinterpret_cast<uint8_t*>(r + b * kwp_));
        xs[p - p0] = weighted_popcount(r, kwp_, kwp_, planes_);
        if (++ox == g_.ow) {
          ox = 0;
          if (++oy == g_.oh) {
            oy = 0;
            ++n;
          }
        }
      }
      qtile(t, rows, xs, p1 - p0, 0, oc_pad_ / 4, y + p0 * g_.oc, g_.oc, pcr);
    }
  }

  Conv2DGeom g_;
  BConvGroupLayout L_;
  Activation act_ = Activation::kNone;
  int planes_ = 0;
  float n_ = 1.f;
  int64_t kwp_ = 0, oc_pad_ = 0, pixels_ = 0, in_pix_ = 0, mt_ = 4, n_ptiles_ = 0, n_tasks_ = 1, slot_words_ = 0;
  std::shared_ptr<void> wbuf_, xplanes_, patches_;
  std::vector<float> mul_, add_;
  std::vector<int32_t> xsum_;
};

bool qbconv_neon_supported(const KernelContext& ctx) {
  const Json& a = ctx.attrs();
  const int64_t k = a.get_int("act_bits", 0);
  return a.get_int("groups", 1) == 1 && k >= 1 && k <= kMaxQNeonPlanes;
}

// QBDense: rows of k levels; row tiles of kQItems rows; few rows (GEMV-like):
// parallel over output blocks.
class QBDenseNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    planes_ = static_cast<int>(ctx.attrs().get_int("act_bits", 0));
    n_ = qneon_levels(ctx);
    k_ = ctx.in(1).dim(1);
    on_ = ctx.in(1).dim(0);
    words_ = bit_words(k_);
    kwp_ = (words_ + 3) / 4 * 4;
    rows_ = ctx.in(0).size() / k_;
    on_pad_ = (on_ + 3) / 4 * 4;
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * kwp_ * 4));
    interleave_rows4(ctx.in(1).data<uint32_t>(), on_, words_, words_, kwp_, static_cast<uint32_t*>(wbuf_.get()));
    xbuf_ = aligned_alloc_shared(static_cast<size_t>(rows_ * planes_ * kwp_ * 4));  // padding words stay 0
    xsum_.assign(static_cast<size_t>(rows_), 0);
    mul_.assign(static_cast<size_t>(on_pad_), 1.f);
    add_.assign(static_cast<size_t>(on_pad_), -0.f);
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(on_pad_ * kwp_ * 4); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    auto* xb = static_cast<uint32_t*>(xbuf_.get());
    const int64_t k = k_, rw = planes_ * kwp_, kwp = kwp_;
    const float n = n_;
    const int planes = planes_;
    ctx.pool->parallel_for(rows_, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(k, 1)), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        pack_levels_row_neon(x + r * k, k, n, planes, xb + r * rw, kwp);
        xsum_[r] = weighted_popcount(xb + r * rw, kwp, kwp, planes);
      }
    });
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, on_, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, on_, add_.data());
    QTile t{static_cast<const uint8_t*>(wbuf_.get()), kwp_ / 4, kwp_, on_, planes_, mul_.data(), add_.data(), act_};
    float* y = ctx.out(0).data<float>();
    const int64_t blocks = on_pad_ / 4, n_rt = (rows_ + kQItems - 1) / kQItems;
    // >= ~8K popcount steps per pool chunk (a dispatch costs ~25 us on the SM8450).
    const int64_t item_steps = std::max<int64_t>(1, std::min(rows_, kQItems) * planes_ * (kwp_ / 4));
    const int64_t grain = std::max<int64_t>(1, 8192 / item_steps);
    ctx.pool->parallel_for(n_rt * blocks, grain, [&](int64_t i0, int64_t i1) {
      uint32x4_t pcr[kQItems * kMaxQNeonPlanes];
      for (int64_t it = i0; it < i1;) {  // runs of blocks of one row tile
        const int64_t rt = it / blocks, b0 = it % blocks, b1 = std::min(blocks, b0 + (i1 - it));
        const int64_t r0 = rt * kQItems, nr = std::min(kQItems, rows_ - r0);
        qtile(t, xb + r0 * rw, xsum_.data() + r0, nr, b0, b1, y + r0 * on_, on_, pcr);
        it += b1 - b0;
      }
    });
  }

 private:
  Activation act_ = Activation::kNone;
  int planes_ = 0;
  float n_ = 1.f;
  int64_t k_ = 0, on_ = 0, words_ = 0, kwp_ = 0, rows_ = 0, on_pad_ = 0;
  std::shared_ptr<void> wbuf_, xbuf_;
  std::vector<int32_t> xsum_;
  std::vector<float> mul_, add_;
};

bool qbdense_neon_supported(const KernelContext& ctx) {
  const int64_t k = ctx.attrs().get_int("act_bits", 0);
  return k >= 1 && k <= kMaxQNeonPlanes;
}

}  // namespace

// Factories for the int8_full NEON kernels (kernels/int8_full/int8_full_neon.cc), which run these
// kernels on the packed binarization of an int8 input and requantize their fp32 output.
std::unique_ptr<Kernel> make_bconv2d_neon_opt_kernel() { return make_kernel<BConv2DNeonOpt>(); }
std::unique_ptr<Kernel> make_bdense_neon_kernel() { return make_kernel<BDenseNeon>(); }
std::unique_ptr<Kernel> make_bdense_ternary_neon_kernel() { return make_kernel<BDenseTernaryNeon>(); }

void register_binary_neon_opt_kernels(OpRegistry& reg) {
  reg.add_kernel({"BConv2D", "bconv2d.neon_opt", "neon", 40, kFeatNeon, bconv_neon_opt_supported, make_kernel<BConv2DNeonOpt>});
  reg.add_kernel({"BDense", "bdense.neon", "neon", 30, kFeatNeon, bdense_neon_supported, make_kernel<BDenseNeon>});
  reg.add_kernel({"BDense", "bdense.ternary_neon", "neon", 35, kFeatNeon, bdense_ternary_neon_supported,
                  make_kernel<BDenseTernaryNeon>});
  reg.add_kernel({"QBConv2D", "qbconv2d.neon", "neon", 30, kFeatNeon, qbconv_neon_supported, make_kernel<QBConv2DNeon>});
  reg.add_kernel({"QBDense", "qbdense.neon", "neon", 30, kFeatNeon, qbdense_neon_supported, make_kernel<QBDenseNeon>});
}

#else

void register_binary_neon_opt_kernels(OpRegistry&) {}

#endif  // __aarch64__ || BF_NEON_A32

}  // namespace bitino
