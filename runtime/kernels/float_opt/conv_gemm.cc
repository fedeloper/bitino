// float_opt Conv2D (any groups) and Dense as cache-blocked packed GEMMs:
//   C[M][N] = act(A[M][K] · B[K][N] + bias)
// Conv2D: M = n*oh*ow output pixels, N = oc / groups, K = kh*kw*kic (kic =
// ic / groups), one GEMM per group. Row m of A is the im2col patch of pixel m
// over the group's input channels in (ky, kx, c) order, which is also the
// order of an OHWI filter row, so filter row o of the group is column o of
// its B and the NHWC output of the group is C, stored with row stride oc at
// column offset group * N. Dense: M = rows, N = O, A = the input itself.
//
// B is packed once in prepare() into column panels of nr: panel p holds
// B[k][nr*p..nr*p+nr-1] for k = 0..K-1 contiguously ([K][nr]), zero padded
// past N; groups are packed one after the other. (Weights and bias are
// therefore read at prepare time and must be constants, as for the packed
// binary kernels.) A is not packed: the micro-kernels read rows of A in place
// (row stride lda). So 1x1 stride-1 unpadded convolutions and Dense read the
// input tensor directly; other convolutions build im2col rows for one block
// of mb pixels at a time into per-thread scratch that stays cache resident.
//
// Backends (the micro-kernels; everything else in this file is shared):
//   Simd4  NEON / portable, nr = 8: kMR = 8 rows, 4 k values per row at a
//          time multiplied by lane (FMLA by element). Register budget (NEON):
//          16 accumulators + 8 A + 2 B = 26 of 32 q registers. Per k: 16
//          FMLA, 2 B loads + 2 (amortized) A loads. Rows past the last 8-row
//          tile: a masked 8-row tile (>= 4 rows) or GEMV rows.
//   Avx2   x86-64 AVX2+FMA (float_opt_avx2.cc), nr = 16 (6x16 tiles) or 8
//          (12x8 tiles, for narrow N such as 8 channels per group).
//
// Blocking: K is split into blocks of at most kKcMax so one A block
// (mb x KC) and one B panel slice (KC x nr) stay in L1/L2. Partial sums live
// in C between K blocks; bias is added by the first and the activation
// applied by the last. Work items are (group, block of mb pixels, chunk of N
// panels); the N split is only used when there are too few (group, pixel
// block) items to feed the pool.
#include <cstring>
#include <type_traits>
#include <vector>

#include "kernels/float_opt/float_opt.h"
#include "kernels/float_opt/float_opt_args.h"
#include "kernels/float_opt/float_simd.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

#if defined(__x86_64__)
#include "kernels/float_opt/float_opt_avx2.h"
#endif

namespace bitino {
namespace {

using namespace fopt;

constexpr int kMR = 8;
constexpr int kNR = 8;
constexpr int64_t kKcMax = 256;  // max K block
// GEMMs below kInlineMacs (~15-20 µs on one big core; e.g. the 512x1000
// classifier GEMV) run on the calling thread (see parallel_grain()). The N
// split never makes work items smaller than kMinTaskMacs, since each item
// redoes its block's im2col.
constexpr int64_t kInlineMacs = int64_t(1) << 19;
constexpr int64_t kMinTaskMacs = int64_t(1) << 19;
constexpr int64_t kMaxGroupTaps = 64;  // kh*kw limit of the all-groups im2col
constexpr int64_t kGroupBlockFloats = 8192;  // 32 KB: all-groups im2col block (L1)

// ------------------------------------------------- Simd4 micro-kernels
template <int L, int MR>
inline void rank1_lane(f32x4 (&acc)[MR][2], const f32x4 (&av)[MR], const float* b) {
  f32x4 b0 = load(b), b1 = load(b + 4);
  for (int i = 0; i < MR; ++i) {
    acc[i][0] = madd_lane<L>(acc[i][0], b0, av[i]);
    acc[i][1] = madd_lane<L>(acc[i][1], b1, av[i]);
  }
}

// C[MR x 8] (+)= A[MR rows][kc] · Bp[kc][8]. Rows >= mvalid re-read the last
// valid row and are not stored; columns >= nvalid are computed on B's zero
// padding and not stored. bias != nullptr: first K block (start from bias),
// else accumulate onto C. last: apply the activation.
template <int MR>
inline void micro_kernel(const float* a, int64_t lda, int mvalid, int64_t kc, const float* bp, float* c, int64_t ldc,
                         int nvalid, const float* bias, bool last, Activation act) {
  const float* ar[MR];
  for (int i = 0; i < MR; ++i) ar[i] = a + std::min(i, mvalid - 1) * lda;
  const bool full = mvalid == MR && nvalid == kNR;
  float tile[MR * kNR];
  f32x4 acc[MR][2];
  if (bias) {
    f32x4 b0 = load(bias), b1 = load(bias + 4);
    for (int i = 0; i < MR; ++i) {
      acc[i][0] = b0;
      acc[i][1] = b1;
    }
  } else if (full) {
    for (int i = 0; i < MR; ++i) {
      acc[i][0] = load(c + i * ldc);
      acc[i][1] = load(c + i * ldc + 4);
    }
  } else {
    std::memset(tile, 0, sizeof tile);
    for (int i = 0; i < mvalid; ++i)
      for (int j = 0; j < nvalid; ++j) tile[i * kNR + j] = c[i * ldc + j];
    for (int i = 0; i < MR; ++i) {
      acc[i][0] = load(tile + i * kNR);
      acc[i][1] = load(tile + i * kNR + 4);
    }
  }
  int64_t k = 0;
#if BF_FLOAT_OPT_NEON
  if (kc >= 4) {
    // The portable loop below in inline asm (the same FMLA by element, in the
    // same order per output): A rows v16-v23 (4 k each), B for 4 k in v24-v31
    // (two 64-byte loads), accumulators v0-v15. 8 LDR + 2 LD1 + 64 FMLA + 3
    // loop = 77 instructions per 4 k (compiled intrinsics: 80-83).
    static_assert(MR == 8, "asm tile is 8 rows");
    float32x4_t c00 = acc[0][0], c01 = acc[0][1], c10 = acc[1][0], c11 = acc[1][1], c20 = acc[2][0], c21 = acc[2][1];
    float32x4_t c30 = acc[3][0], c31 = acc[3][1], c40 = acc[4][0], c41 = acc[4][1], c50 = acc[5][0], c51 = acc[5][1];
    float32x4_t c60 = acc[6][0], c61 = acc[6][1], c70 = acc[7][0], c71 = acc[7][1];
    const float* ap = ar[0];
    const float* b = bp;
    int64_t n4 = kc >> 2;
    const int64_t o1 = (ar[1] - ar[0]) * 4, o2 = (ar[2] - ar[0]) * 4, o3 = (ar[3] - ar[0]) * 4, o4 = (ar[4] - ar[0]) * 4;
    const int64_t o5 = (ar[5] - ar[0]) * 4, o6 = (ar[6] - ar[0]) * 4, o7 = (ar[7] - ar[0]) * 4;
#define BF_FM(L, BL, BH)                                                                                   \
  "fmla %[c00].4s, " BL ".4s, v16.s[" L "]\n fmla %[c01].4s, " BH ".4s, v16.s[" L "]\n"                   \
  "fmla %[c10].4s, " BL ".4s, v17.s[" L "]\n fmla %[c11].4s, " BH ".4s, v17.s[" L "]\n"                   \
  "fmla %[c20].4s, " BL ".4s, v18.s[" L "]\n fmla %[c21].4s, " BH ".4s, v18.s[" L "]\n"                   \
  "fmla %[c30].4s, " BL ".4s, v19.s[" L "]\n fmla %[c31].4s, " BH ".4s, v19.s[" L "]\n"                   \
  "fmla %[c40].4s, " BL ".4s, v20.s[" L "]\n fmla %[c41].4s, " BH ".4s, v20.s[" L "]\n"                   \
  "fmla %[c50].4s, " BL ".4s, v21.s[" L "]\n fmla %[c51].4s, " BH ".4s, v21.s[" L "]\n"                   \
  "fmla %[c60].4s, " BL ".4s, v22.s[" L "]\n fmla %[c61].4s, " BH ".4s, v22.s[" L "]\n"                   \
  "fmla %[c70].4s, " BL ".4s, v23.s[" L "]\n fmla %[c71].4s, " BH ".4s, v23.s[" L "]\n"
    asm volatile(
        "1:\n"
        "ldr q16, [%[ap]]\n ldr q17, [%[ap], %[o1]]\n ldr q18, [%[ap], %[o2]]\n ldr q19, [%[ap], %[o3]]\n"
        "ldr q20, [%[ap], %[o4]]\n ldr q21, [%[ap], %[o5]]\n ldr q22, [%[ap], %[o6]]\n ldr q23, [%[ap], %[o7]]\n"
        "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[b]], #64\n"
        "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[b]], #64\n"
        "add %[ap], %[ap], #16\n"
        BF_FM("0", "v24", "v25") BF_FM("1", "v26", "v27") BF_FM("2", "v28", "v29") BF_FM("3", "v30", "v31")
        "subs %[n], %[n], #1\n"
        "b.ne 1b\n"
        : [ap] "+r"(ap), [b] "+r"(b), [n] "+r"(n4), [c00] "+w"(c00), [c01] "+w"(c01), [c10] "+w"(c10), [c11] "+w"(c11),
          [c20] "+w"(c20), [c21] "+w"(c21), [c30] "+w"(c30), [c31] "+w"(c31), [c40] "+w"(c40), [c41] "+w"(c41),
          [c50] "+w"(c50), [c51] "+w"(c51), [c60] "+w"(c60), [c61] "+w"(c61), [c70] "+w"(c70), [c71] "+w"(c71)
        : [o1] "r"(o1), [o2] "r"(o2), [o3] "r"(o3), [o4] "r"(o4), [o5] "r"(o5), [o6] "r"(o6), [o7] "r"(o7)
        : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
#undef BF_FM
    acc[0][0] = c00, acc[0][1] = c01, acc[1][0] = c10, acc[1][1] = c11, acc[2][0] = c20, acc[2][1] = c21;
    acc[3][0] = c30, acc[3][1] = c31, acc[4][0] = c40, acc[4][1] = c41, acc[5][0] = c50, acc[5][1] = c51;
    acc[6][0] = c60, acc[6][1] = c61, acc[7][0] = c70, acc[7][1] = c71;
    k = kc & ~int64_t(3);
  }
#elif BF_FLOAT_OPT_NEON32
  {
    // AArch32 (16 q registers): the same multiplies by lane (VMLA: rounded product, then add, as madd()), in the
    // same order per output, as two passes of 4 rows x 8 columns over the K block; the B slice (<= 8 KB) stays
    // in L1. Inline asm (compiled intrinsics hoist the B loads and spill): A rows q0-q3 (a by-scalar operand must
    // be in d0-d15), B for two k in q4-q7, accumulators q8-q15 loaded from / stored to acc[h..h+3]. On the A53
    // NEON loads do not dual-issue with arithmetic: per 4 k 24 VLDR + 32 VMLA, B reloaded two groups after its
    // last use (measured 68 cycles, VLD1 with writeback 72). The last kc % 4 k: one k per iteration, A as lanes.
    static_assert(MR == 8, "two 4-row passes");
#define BF_FM32(BL, BH, L0, L1, L2, L3)                                                                  \
  "vmla.f32 q8, " BL ", " L0 "\n vmla.f32 q9, " BH ", " L0 "\n vmla.f32 q10, " BL ", " L1 "\n"             \
  "vmla.f32 q11, " BH ", " L1 "\n vmla.f32 q12, " BL ", " L2 "\n vmla.f32 q13, " BH ", " L2 "\n"          \
  "vmla.f32 q14, " BL ", " L3 "\n vmla.f32 q15, " BH ", " L3 "\n"
#define BF_VL4(D0, D1, D2, D3, O0, O1, O2, O3)                                                           \
  "vldr " D0 ", [%[b], #" O0 "]\n vldr " D1 ", [%[b], #" O1 "]\n vldr " D2 ", [%[b], #" O2 "]\n"           \
  "vldr " D3 ", [%[b], #" O3 "]\n"
    for (int h = 0; h < MR; h += 4) {
      const float *a0 = ar[h], *a1 = ar[h + 1], *a2 = ar[h + 2], *a3 = ar[h + 3];
      const float* b = bp;
      float* cp = reinterpret_cast<float*>(&acc[h][0]);
      int n4 = static_cast<int>(kc >> 2), n1 = static_cast<int>(kc & 3);
      asm volatile(
          "vld1.32 {d16-d19}, [%[c]]!\n vld1.32 {d20-d23}, [%[c]]!\n"
          "vld1.32 {d24-d27}, [%[c]]!\n vld1.32 {d28-d31}, [%[c]]\n"
          "cmp %[n], #0\n"
          "beq 2f\n"
          "1:\n"
          "vldr d0, [%[a0]]\n vldr d1, [%[a0], #8]\n vldr d2, [%[a1]]\n vldr d3, [%[a1], #8]\n"
          "vldr d4, [%[a2]]\n vldr d5, [%[a2], #8]\n vldr d6, [%[a3]]\n vldr d7, [%[a3], #8]\n"
          BF_VL4("d8", "d9", "d10", "d11", "0", "8", "16", "24")
          BF_VL4("d12", "d13", "d14", "d15", "32", "40", "48", "56")
          "add %[a0], %[a0], #16\n add %[a1], %[a1], #16\n add %[a2], %[a2], #16\n add %[a3], %[a3], #16\n"
          BF_FM32("q4", "q5", "d0[0]", "d2[0]", "d4[0]", "d6[0]")
          BF_FM32("q6", "q7", "d0[1]", "d2[1]", "d4[1]", "d6[1]")
          BF_VL4("d8", "d9", "d10", "d11", "64", "72", "80", "88")
          BF_VL4("d12", "d13", "d14", "d15", "96", "104", "112", "120")
          BF_FM32("q4", "q5", "d1[0]", "d3[0]", "d5[0]", "d7[0]")
          "add %[b], %[b], #128\n"
          BF_FM32("q6", "q7", "d1[1]", "d3[1]", "d5[1]", "d7[1]")
          "subs %[n], %[n], #1\n"
          "bne 1b\n"
          "2:\n"
          "cmp %[r], #0\n"
          "beq 4f\n"
          "3:\n"
          "vld1.32 {d0[0]}, [%[a0]]!\n vld1.32 {d0[1]}, [%[a1]]!\n vld1.32 {d1[0]}, [%[a2]]!\n vld1.32 {d1[1]}, [%[a3]]!\n"
          BF_VL4("d8", "d9", "d10", "d11", "0", "8", "16", "24")
          "add %[b], %[b], #32\n"
          BF_FM32("q4", "q5", "d0[0]", "d0[1]", "d1[0]", "d1[1]")
          "subs %[r], %[r], #1\n"
          "bne 3b\n"
          "4:\n"
          "sub %[c], %[c], #96\n"
          "vst1.32 {d16-d19}, [%[c]]!\n vst1.32 {d20-d23}, [%[c]]!\n"
          "vst1.32 {d24-d27}, [%[c]]!\n vst1.32 {d28-d31}, [%[c]]\n"
          : [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3), [b] "+r"(b), [n] "+r"(n4), [r] "+r"(n1), [c] "+r"(cp)
          :
          : "cc", "memory", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14",
            "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
            "d31");
    }
#undef BF_FM32
#undef BF_VL4
    k = kc;
  }
#endif
  for (; k + 4 <= kc; k += 4) {
    f32x4 av[MR];
    for (int i = 0; i < MR; ++i) av[i] = load(ar[i] + k);
    const float* b = bp + k * kNR;
    rank1_lane<0>(acc, av, b);
    rank1_lane<1>(acc, av, b + kNR);
    rank1_lane<2>(acc, av, b + 2 * kNR);
    rank1_lane<3>(acc, av, b + 3 * kNR);
  }
  for (; k < kc; ++k) {
    f32x4 b0 = load(bp + k * kNR), b1 = load(bp + k * kNR + 4);
    for (int i = 0; i < MR; ++i) {
      f32x4 ai = dup(ar[i][k]);
      acc[i][0] = madd(acc[i][0], b0, ai);
      acc[i][1] = madd(acc[i][1], b1, ai);
    }
  }
  if (last)
    for (int i = 0; i < MR; ++i) {
      acc[i][0] = activate(act, acc[i][0]);
      acc[i][1] = activate(act, acc[i][1]);
    }
  if (full) {
    for (int i = 0; i < MR; ++i) {
      store(c + i * ldc, acc[i][0]);
      store(c + i * ldc + 4, acc[i][1]);
    }
  } else {
    for (int i = 0; i < MR; ++i) {
      store(tile + i * kNR, acc[i][0]);
      store(tile + i * kNR + 4, acc[i][1]);
    }
    for (int i = 0; i < mvalid; ++i)
      for (int j = 0; j < nvalid; ++j) c[i * ldc + j] = tile[i * kNR + j];
  }
}

// Single row (GEMV): c[0..8) (+)= a[0..kc) · Bp[kc][8]. Four independent
// accumulator pairs (one per k mod 4) break the FMA dependency chain.
inline void gemv_kernel(const float* a, int64_t kc, const float* bp, float* c, int nvalid, const float* bias, bool last,
                        Activation act) {
  f32x4 s[4][2];
  for (auto& r : s) r[0] = r[1] = dup(0.f);
  int64_t k = 0;
  for (; k + 4 <= kc; k += 4) {
    f32x4 av = load(a + k);
    const float* b = bp + k * kNR;
    s[0][0] = madd_lane<0>(s[0][0], load(b), av);
    s[0][1] = madd_lane<0>(s[0][1], load(b + 4), av);
    s[1][0] = madd_lane<1>(s[1][0], load(b + 8), av);
    s[1][1] = madd_lane<1>(s[1][1], load(b + 12), av);
    s[2][0] = madd_lane<2>(s[2][0], load(b + 16), av);
    s[2][1] = madd_lane<2>(s[2][1], load(b + 20), av);
    s[3][0] = madd_lane<3>(s[3][0], load(b + 24), av);
    s[3][1] = madd_lane<3>(s[3][1], load(b + 28), av);
  }
  for (; k < kc; ++k) {
    f32x4 ak = dup(a[k]);
    s[0][0] = madd(s[0][0], load(bp + k * kNR), ak);
    s[0][1] = madd(s[0][1], load(bp + k * kNR + 4), ak);
  }
  float tile[kNR];
  if (bias) {
    std::memcpy(tile, bias, sizeof tile);
  } else {
    std::memset(tile, 0, sizeof tile);
    std::memcpy(tile, c, static_cast<size_t>(nvalid) * sizeof(float));
  }
  for (int j = 0; j < 2; ++j) {
    f32x4 v = add(load(tile + 4 * j), add(add(s[0][j], s[1][j]), add(s[2][j], s[3][j])));
    store(tile + 4 * j, last ? activate(act, v) : v);
  }
  std::memcpy(c, tile, static_cast<size_t>(nvalid) * sizeof(float));
}

// ---- 4-column panels (N <= 4: e.g. the RGB output convolution of image restoration / super-resolution models,
// where an 8-column tile computes 5 of 8 columns on padding). Same per-element arithmetic as micro_kernel /
// gemv_kernel, so the results do not depend on the panel width: tile rows start from the bias (or C) and take one
// madd per k in ascending order; GEMV rows keep gemv_kernel's four partial sums (one per k mod 4).
// Rows per tile: 8 (AArch64: 8 accumulators + 8 A + 4 B registers), 4 on 32-bit ARM (the by-scalar operand of
// VMLA must sit in q0-q7).
constexpr int kMR4 = BF_FLOAT_OPT_NEON32 ? 4 : 8;

template <int MR>
inline void micro_kernel_n4(const float* a, int64_t lda, int mvalid, int64_t kc, const float* bp, float* c, int64_t ldc, int nvalid,
                            const float* bias, bool last, Activation act) {
  const float* ar[MR];
  for (int i = 0; i < MR; ++i) ar[i] = a + std::min(i, mvalid - 1) * lda;
  f32x4 acc[MR];
  if (bias) {
    const f32x4 b0 = load(bias);
    for (int i = 0; i < MR; ++i) acc[i] = b0;
  } else {
    for (int i = 0; i < MR; ++i) {
      float t[4] = {0.f, 0.f, 0.f, 0.f};
      if (i < mvalid)
        for (int j = 0; j < nvalid; ++j) t[j] = c[i * ldc + j];
      acc[i] = load(t);
    }
  }
  int64_t k = 0;
  for (; k + 4 <= kc; k += 4) {
    const float* b = bp + k * 4;
    const f32x4 b0 = load(b), b1 = load(b + 4), b2 = load(b + 8), b3 = load(b + 12);
    for (int i = 0; i < MR; ++i) {
      const f32x4 av = load(ar[i] + k);
      acc[i] = madd_lane<0>(acc[i], b0, av);
      acc[i] = madd_lane<1>(acc[i], b1, av);
      acc[i] = madd_lane<2>(acc[i], b2, av);
      acc[i] = madd_lane<3>(acc[i], b3, av);
    }
  }
  for (; k < kc; ++k) {
    const f32x4 b0 = load(bp + k * 4);
    for (int i = 0; i < MR; ++i) acc[i] = madd(acc[i], b0, dup(ar[i][k]));
  }
  for (int i = 0; i < mvalid; ++i) {
    const f32x4 v = last ? activate(act, acc[i]) : acc[i];
    if (nvalid == 4) {
      store(c + i * ldc, v);
    } else {
      float t[4];
      store(t, v);
      for (int j = 0; j < nvalid; ++j) c[i * ldc + j] = t[j];
    }
  }
}

// gemv_kernel on a 4-column panel (its columns 0-3).
inline void gemv_kernel_n4(const float* a, int64_t kc, const float* bp, float* c, int nvalid, const float* bias, bool last,
                           Activation act) {
  f32x4 s[4];
  for (auto& r : s) r = dup(0.f);
  int64_t k = 0;
  for (; k + 4 <= kc; k += 4) {
    const f32x4 av = load(a + k);
    const float* b = bp + k * 4;
    s[0] = madd_lane<0>(s[0], load(b), av);
    s[1] = madd_lane<1>(s[1], load(b + 4), av);
    s[2] = madd_lane<2>(s[2], load(b + 8), av);
    s[3] = madd_lane<3>(s[3], load(b + 12), av);
  }
  for (; k < kc; ++k) s[0] = madd(s[0], load(bp + k * 4), dup(a[k]));
  float tile[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias) std::memcpy(tile, bias, sizeof tile);
  else std::memcpy(tile, c, static_cast<size_t>(nvalid) * sizeof(float));
  const f32x4 v = add(load(tile), add(add(s[0], s[1]), add(s[2], s[3])));
  store(tile, last ? activate(act, v) : v);
  std::memcpy(c, tile, static_cast<size_t>(nvalid) * sizeof(float));
}

void gemm_block_simd4_n4(const GemmBlockArgs& g) {
  const bool last = g.last != 0;
  const Activation act = static_cast<Activation>(g.act);
  const int64_t rows = g.rows, lda = g.lda, ldc = g.ldc;
  // rows of whole 8-row tiles or of a masked tile (>= 4 rows) in gemm_block_simd4; the rest are GEMV rows
  const int64_t rem = rows % kMR, tile_rows = rem >= 4 ? rows : rows - rem;
  for (int64_t p = g.p0; p < g.p1; ++p) {
    const float* bp = g.bp + p * g.panel_stride;
    const float* bias_p = g.bias ? g.bias + p * 4 : nullptr;
    const int nvalid = static_cast<int>(std::min<int64_t>(4, g.ncols - p * 4));
    float* cp = g.c + p * 4;
    for (int64_t m = 0; m < tile_rows; m += kMR4)
      micro_kernel_n4<kMR4>(g.a + m * lda, lda, static_cast<int>(std::min<int64_t>(kMR4, tile_rows - m)), g.kc, bp, cp + m * ldc, ldc,
                            nvalid, bias_p, last, act);
    for (int64_t m = tile_rows; m < rows; ++m) gemv_kernel_n4(g.a + m * lda, g.kc, bp, cp + m * ldc, nvalid, bias_p, last, act);
  }
}

// One GEMM block on the Simd4 backend (GemmBlockArgs: float_opt_args.h).
void gemm_block_simd4(const GemmBlockArgs& g) {
  if (g.nr == 4) return gemm_block_simd4_n4(g);
  const bool last = g.last != 0;
  const Activation act = static_cast<Activation>(g.act);
  for (int64_t p = g.p0; p < g.p1; ++p) {
    const float* bp = g.bp + p * g.panel_stride;
    const float* bias_p = g.bias ? g.bias + p * kNR : nullptr;
    const int nvalid = static_cast<int>(std::min<int64_t>(kNR, g.ncols - p * kNR));
    float* cp = g.c + p * kNR;
    const int64_t rows = g.rows, lda = g.lda, ldc = g.ldc;
    int64_t m = 0;
    for (; m + kMR <= rows; m += kMR)
      micro_kernel<kMR>(g.a + m * lda, lda, kMR, g.kc, bp, cp + m * ldc, ldc, nvalid, bias_p, last, act);
    const int64_t rem = rows - m;
    if (rem >= 4) {
      micro_kernel<kMR>(g.a + m * lda, lda, static_cast<int>(rem), g.kc, bp, cp + m * ldc, ldc, nvalid, bias_p, last, act);
    } else {
      for (; m < rows; ++m) gemv_kernel(g.a + m * lda, g.kc, bp, cp + m * ldc, nvalid, bias_p, last, act);
    }
  }
}

// ------------------------------------------------------------- backends
struct Simd4Gemm {
  // pixels per work item (multiple of kMR): 32, so that an im2col block of a 7x7x3 stem (147 floats a row) stays in
  // L1 while it is read once per panel (A53 32 KB; on the Cortex-A75 the stem gained 3 % over 64, the X3 is
  // neutral; ledger float_opt.md 2026-11-12 round 2). Row tiles / GEMV rows depend only on M % 8: unchanged.
  static constexpr int64_t kMB = 32;
  static constexpr int64_t kRowStep = 8;  // row tile
  // 4-column panels for N <= 4 (not on AArch32: unmeasured there, the 8-column asm tile kept)
  static int64_t nr_for(int64_t n) { return n <= 4 && !BF_FLOAT_OPT_NEON32 ? 4 : kNR; }
  static void block(const GemmBlockArgs& g) { gemm_block_simd4(g); }
};

#if defined(__x86_64__)
struct Avx2Gemm {
  static constexpr int64_t kMB = 96;      // multiple of the 6- and 12-row tiles
  static constexpr int64_t kRowStep = 12;
  // 16-wide panels unless that wastes more than 1/8 of the computed columns
  // and 8-wide panels waste less (e.g. N = 8, 24, 40).
  static int64_t nr_for(int64_t n) {
    const int64_t pad16 = round_up(n, 16) - n, pad8 = round_up(n, 8) - n;
    return pad8 < pad16 && pad16 * 8 > round_up(n, 16) ? 8 : 16;
  }
  static void block(const GemmBlockArgs& g) { fopt_avx2::gemm_block(g); }
};
#endif

// --------------------------------------------------------------- GEMM plan
template <class B>
struct GemmPlan {
  int64_t G = 1, M = 0, N = 0, K = 0, ldc = 0, nr = kNR;
  int64_t KC = 0, mb = 0, panels = 0, mblocks = 0, chunk_panels = 0, nchunks = 0, grain = 1;
  bool all_groups = false;
  Activation act = Activation::kNone;
  std::shared_ptr<void> bpack;  // [G][panels][K][nr]
  std::shared_ptr<void> bias;   // [G][panels * nr]

  // w: [G*n][k] row-major (OHWI filter or Dense weights); group g owns rows
  // [g*n, (g+1)*n). C rows have stride ldc_ (>= G*n).
  void pack(const float* w, const float* bias_src, int64_t groups, int64_t n, int64_t k, int64_t ldc_) {
    G = groups;
    N = n;
    K = k;
    ldc = ldc_;
    nr = B::nr_for(N);
    panels = (N + nr - 1) / nr;
    bpack = aligned_alloc_shared(static_cast<size_t>(G * panels * K * nr) * sizeof(float));
    bias = aligned_alloc_shared(static_cast<size_t>(G * panels * nr) * sizeof(float));
    float* bp = static_cast<float*>(bpack.get());
    float* bb = static_cast<float*>(bias.get());
    // aligned_alloc_shared zero-fills: the padding columns stay 0.
    for (int64_t g = 0; g < G; ++g)
      for (int64_t o = 0; o < N; ++o) {
        const int64_t row = g * N + o;
        float* dst = bp + (g * panels + o / nr) * K * nr + o % nr;
        for (int64_t j = 0; j < K; ++j) dst[j * nr] = w[row * K + j];
        bb[g * panels * nr + o] = bias_src ? bias_src[row] : 0.f;
      }
  }

  // all_groups: one work item covers every group of its pixel block (see
  // run_all_groups); requires a single K block.
  void schedule(int64_t m, int threads, bool all_groups_ = false) {
    M = m;
    // Small M (GEMV-like): A is tiny, so there is nothing to gain from K
    // blocking. Otherwise split K into near-equal blocks of <= kKcMax.
    if (M < kMR || K <= kKcMax) {
      KC = K;
    } else {
      int64_t nkb = (K + kKcMax - 1) / kKcMax;
      KC = round_up((K + nkb - 1) / nkb, 4);
    }
    all_groups = all_groups_ && G > 1 && KC == K;
    // All-groups blocks hold G * mb * K floats of im2col rows: keep them
    // within kGroupBlockFloats so building and consuming them stays in L1.
    mb = B::kMB;
    if (all_groups) mb = std::max(B::kRowStep, std::min(B::kMB, kGroupBlockFloats / (G * K) / B::kRowStep * B::kRowStep));
    mblocks = (M + mb - 1) / mb;
    // Split N only when there are too few (group, pixel block) items for ~4
    // work items per thread, and never below kMinTaskMacs per item.
    const int64_t gblocks = all_groups ? mblocks : G * mblocks;
    const int64_t block_macs = std::min(mb, M) * N * K * (all_groups ? G : 1);
    nchunks = 1;
    if (threads > 1 && gblocks < 4 * threads) {
      nchunks = std::min<int64_t>(panels, (4 * threads + gblocks - 1) / gblocks);
      nchunks = std::max<int64_t>(1, std::min(nchunks, block_macs / kMinTaskMacs));
    }
    chunk_panels = (panels + nchunks - 1) / nchunks;
    nchunks = (panels + chunk_panels - 1) / chunk_panels;
    grain = parallel_grain(gblocks * nchunks, M * G * N * K, kInlineMacs);
  }

  size_t bytes() const { return static_cast<size_t>(G * (panels * K + panels) * nr) * sizeof(float); }

  // get_a(g, m0, rows, k0, kc, scratch, &lda) returns A_g[m0.., k0..] (scratch
  // is this thread's buffer of mb * KC floats).
  template <typename GetA>
  void run(ThreadPool* pool, ScratchSlots& scratch, float* c, const GetA& get_a) const {
    const float* bp_all = static_cast<const float*>(bpack.get());
    const float* bias_all = static_cast<const float*>(bias.get());
    // Items: pixel block outermost, then group, then panel chunk, so
    // consecutive items read the same input pixels (all groups' channels).
    pool->parallel_for(mblocks * G * nchunks, grain, [&](int64_t t0, int64_t t1) {
      ScratchLease lease(scratch);
      float* buf = lease.data();
      int64_t built = -1, built_lda = 0;  // (group, A block) already in `buf` (single K block only), its row stride
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t blk = t / (G * nchunks), g = (t / nchunks) % G, ch = t % nchunks;
        const int64_t m0 = blk * mb, rows = std::min(mb, M - m0);
        const int64_t gm = g * mblocks + blk;
        GemmBlockArgs args;
        args.rows = rows;
        args.panel_stride = K * nr;
        args.p0 = ch * chunk_panels;
        args.p1 = std::min(panels, args.p0 + chunk_panels);
        args.nr = nr;
        args.c = c + m0 * ldc + g * N;
        args.ldc = ldc;
        args.ncols = N;
        args.act = static_cast<int32_t>(act);
        for (int64_t k0 = 0; k0 < K; k0 += KC) {
          const int64_t kc = std::min(KC, K - k0);
          int64_t lda = 0;
          const float* a;
          if (kc == K && built == gm) {
            a = buf;
            lda = built_lda;
          } else {
            a = get_a(g, m0, rows, k0, kc, buf, &lda);
            if (kc == K && a == buf) {
              built = gm;
              built_lda = lda;
            }
          }
          args.a = a;
          args.lda = lda;
          args.kc = kc;
          args.bp = bp_all + g * panels * K * nr + k0 * nr;
          args.bias = k0 == 0 ? bias_all + g * panels * nr : nullptr;
          args.last = k0 + kc >= K;
          B::block(args);
        }
      }
    });
  }

  // Grouped layers with a single K block: each work item (pixel block, panel
  // chunk) builds the A blocks of all groups at once, build(m0, rows, buf)
  // writing group g's row r to buf + (r * G + g) * K (buf holds G * mb * K
  // floats), then runs every group's GEMM on it (A_g = buf + g * K, row
  // stride G * K). One pass over the input pixels serves all groups.
  template <typename Build>
  void run_all_groups(ThreadPool* pool, ScratchSlots& scratch, float* c, const Build& build) const {
    const float* bp_all = static_cast<const float*>(bpack.get());
    const float* bias_all = static_cast<const float*>(bias.get());
    pool->parallel_for(mblocks * nchunks, grain, [&](int64_t t0, int64_t t1) {
      ScratchLease lease(scratch);
      float* buf = lease.data();
      int64_t built = -1;
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t blk = t / nchunks, ch = t % nchunks;
        const int64_t m0 = blk * mb, rows = std::min(mb, M - m0);
        if (built != blk) {
          build(m0, rows, buf);
          built = blk;
        }
        GemmBlockArgs args;
        args.lda = G * K;
        args.rows = rows;
        args.kc = K;
        args.panel_stride = K * nr;
        args.p0 = ch * chunk_panels;
        args.p1 = std::min(panels, args.p0 + chunk_panels);
        args.nr = nr;
        args.ldc = ldc;
        args.ncols = N;
        args.last = 1;
        args.act = static_cast<int32_t>(act);
        for (int64_t g = 0; g < G; ++g) {
          args.a = buf + g * K;
          args.bp = bp_all + g * panels * K * nr;
          args.bias = bias_all + g * panels * nr;
          args.c = c + m0 * ldc + g * N;
          B::block(args);
        }
      }
    });
  }
};

// ------------------------------------------------------------------- Conv2D
// Copies n floats. Short runs (im2col runs are often 3..21 floats) use 4-float
// vectors whose last one overlaps the previous instead of a scalar tail; the
// library memcpy is faster from ~5 vectors on (ledger: float_opt step 2).
inline void copy_floats(float* d, const float* s, int64_t n) {
  if (n < 4) {
    for (int64_t j = 0; j < n; ++j) d[j] = s[j];
  } else if (n <= 16) {
    for (int64_t j = 0; j + 4 < n; j += 4) store(d + j, load(s + j));
    store(d + n - 4, load(s + n - 4));
  } else {
    std::memcpy(d, s, static_cast<size_t>(n) * sizeof(float));
  }
}

// ------------------------------------------------- narrow direct convolution
// Convolutions with at most 4 output channels per group, stride 1, dilation 1 and a kh x kw > 1 kernel or groups > 1:
// the RGB output layer of image restoration / super-resolution models, depthwise layers with a channel multiplier
// (BSRC: 64 -> 128 channels, groups 64) and the grouped 1x1 that folds them back. The GEMM spends most of its time
// building im2col rows (each value used for <= 4 multiply-adds) or, for tiny groups, in per-tile overhead. Instead
// the input is transposed once per run into padded channel rows xt[n][iy][c][ix] (pad_value around the image, as
// im2col fills padding taps), and each vector lane is an output pixel: per group, 4 * kNarrowPV consecutive pixels
// of an output row times its outputs, for every k = (ky, kx, c) of the group in ascending order
// acc = madd(acc, x, w[k]) from the bias, the activation at the end. That is the GEMM's chain for every output
// (8-row tiles and masked tiles; used only when the GEMM plan has no GEMV rows, whose four partial sums differ), so
// the results are bit-identical to the GEMM path.
constexpr int kNarrowPV = BF_FLOAT_OPT_NEON32 ? 2 : 4;  // pixel vectors per tile (AArch32: 16 q registers)

// dst[c * ws + i] = x[i * ic + c] for i < iw, c < ic (one input row into its channel rows).
inline void transpose_row(const float* x, int64_t iw, int64_t ic, float* dst, int64_t ws) {
  int64_t i = 0;
#if BF_FLOAT_OPT_NEON || BF_FLOAT_OPT_NEON32
  if (ic % 4 == 0)
    for (; i + 4 <= iw; i += 4)
      for (int64_t c = 0; c < ic; c += 4) {
        const float32x4x2_t t0 = vtrnq_f32(vld1q_f32(x + i * ic + c), vld1q_f32(x + (i + 1) * ic + c));
        const float32x4x2_t t1 = vtrnq_f32(vld1q_f32(x + (i + 2) * ic + c), vld1q_f32(x + (i + 3) * ic + c));
        float* d = dst + c * ws + i;
        vst1q_f32(d, vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])));
        vst1q_f32(d + ws, vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])));
        vst1q_f32(d + 2 * ws, vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])));
        vst1q_f32(d + 3 * ws, vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])));
      }
#endif
  for (; i < iw; ++i)
    for (int64_t c = 0; c < ic; ++c) dst[c * ws + i] = x[i * ic + c];
}

template <int NO>
inline __attribute__((always_inline)) void narrow_madd(f32x4 (&acc)[NO], f32x4 x, f32x4 w) {
  acc[0] = madd_lane<0>(acc[0], x, w);
  if constexpr (NO > 1) acc[1] = madd_lane<1>(acc[1], x, w);
  if constexpr (NO > 2) acc[2] = madd_lane<2>(acc[2], x, w);
  if constexpr (NO > 3) acc[3] = madd_lane<3>(acc[3], x, w);
}

// One tile of 4 * kNarrowPV output pixels of one group: xr = the channel row of the group's first input channel at the
// tile's first input pixel (padded coordinates; kernel row ky is cs floats further), kic channel rows ws floats
// apart, wn = the group's packed weights [kh * kw * kic][4], bias[4]; t[o * TP + p] = output o of pixel p.
template <int NO>
inline void narrow_conv_tile(const float* xr, int64_t cs, int64_t kic, int64_t ws, int64_t kh, int64_t kw, const float* wn,
                             const float* bias, Activation act, float* t) {
  constexpr int PV = kNarrowPV, TP = 4 * PV;
  f32x4 acc[PV][NO];
  for (int v = 0; v < PV; ++v)
    for (int o = 0; o < NO; ++o) acc[v][o] = dup(bias[o]);
  const float* wk = wn;
  for (int64_t ky = 0; ky < kh; ++ky) {
    const float* row = xr + ky * cs;
    for (int64_t kx = 0; kx < kw; ++kx)
      for (int64_t c = 0; c < kic; ++c, wk += 4) {
        const f32x4 w = load(wk);
        const float* q = row + c * ws + kx;
        for (int v = 0; v < PV; ++v) narrow_madd<NO>(acc[v], load(q + 4 * v), w);
      }
  }
  for (int v = 0; v < PV; ++v)
    for (int o = 0; o < NO; ++o) store(t + o * TP + 4 * v, activate(act, acc[v][o]));
}

// t[c * tp + p] (c < oc, p < np) -> y[p * oc + c]: 4x4 transposes for whole blocks, scalar otherwise.
inline void narrow_store(const float* t, int64_t tp, int64_t oc, int64_t np, float* y) {
  int64_t p = 0;
#if BF_FLOAT_OPT_NEON || BF_FLOAT_OPT_NEON32
  if (oc % 4 == 0)
    for (; p + 4 <= np; p += 4)
      for (int64_t c = 0; c < oc; c += 4) {
        const float32x4x2_t a = vtrnq_f32(vld1q_f32(t + c * tp + p), vld1q_f32(t + (c + 1) * tp + p));
        const float32x4x2_t b = vtrnq_f32(vld1q_f32(t + (c + 2) * tp + p), vld1q_f32(t + (c + 3) * tp + p));
        float* d = y + p * oc + c;
        vst1q_f32(d, vcombine_f32(vget_low_f32(a.val[0]), vget_low_f32(b.val[0])));
        vst1q_f32(d + oc, vcombine_f32(vget_low_f32(a.val[1]), vget_low_f32(b.val[1])));
        vst1q_f32(d + 2 * oc, vcombine_f32(vget_high_f32(a.val[0]), vget_high_f32(b.val[0])));
        vst1q_f32(d + 3 * oc, vcombine_f32(vget_high_f32(a.val[1]), vget_high_f32(b.val[1])));
      }
#endif
  for (; p < np; ++p)
    for (int64_t c = 0; c < oc; ++c) y[p * oc + c] = t[c * tp + p];
}

template <class B>
class Conv2DGemm : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    pad_value_ = static_cast<float>(ctx.attrs().get_number("pad_value", 0.0));
    plan_.act = parse_activation(ctx.attrs());
    direct_ = g_.kh == 1 && g_.kw == 1 && g_.sh == 1 && g_.sw == 1 && g_.pt == 0 && g_.pb == 0 && g_.pl == 0 && g_.pr == 0;
    const int64_t k = g_.kh * g_.kw * g_.kic;
    plan_.pack(ctx.in(1).data<float>(), opt_f32(ctx, 2), g_.groups, g_.oc / g_.groups, k, g_.oc);
    plan_.schedule(g_.n * g_.oh * g_.ow, ctx.pool->num_threads(), !direct_ && g_.kh * g_.kw <= kMaxGroupTaps);
    if (plan_.all_groups) {
      padrow_ = aligned_alloc_shared(static_cast<size_t>(g_.ic) * sizeof(float));
      std::fill_n(static_cast<float*>(padrow_.get()), g_.ic, pad_value_);
    }
    scratch_.init(ctx.pool->num_threads(),
                  direct_ ? 1 : static_cast<size_t>(plan_.mb * round_up(plan_.KC, 4) * (plan_.all_groups ? g_.groups : 1)));
    prepare_narrow(ctx);
  }
  size_t prepacked_bytes() const override { return plan_.bytes(); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    if (narrow_) {
      run_narrow(ctx, x, y);
    } else if (direct_) {
      const int64_t ic = g_.ic, kic = g_.kic;
      plan_.run(ctx.pool, scratch_, y, [&](int64_t grp, int64_t m0, int64_t, int64_t k0, int64_t, float*, int64_t* lda) {
        *lda = ic;
        return x + m0 * ic + grp * kic + k0;
      });
    } else if (plan_.all_groups) {
      plan_.run_all_groups(ctx.pool, scratch_, y, [&](int64_t m0, int64_t rows, float* buf) { im2col_groups(x, m0, rows, buf); });
    } else {
      plan_.run(ctx.pool, scratch_, y, [&](int64_t grp, int64_t m0, int64_t rows, int64_t k0, int64_t kc, float* buf, int64_t* lda) {
        // rows 16-byte aligned (K = 147 of a 7x7 RGB stem: 7 % faster on the Cortex-A75, 3 % on the A53)
        const int64_t ld = round_up(kc, 4);
        im2col(x, grp * g_.kic, m0, rows, k0, kc, buf, ld);
        *lda = ld;
        return static_cast<const float*>(buf);
      });
    }
  }

 private:
  // Patch rows of pixels [m0, m0+rows), columns [k0, k0+kc) of the group whose
  // input channels start at coff -> dst[rows][kc] with row stride ld.
  void im2col(const float* x, int64_t coff, int64_t m0, int64_t rows, int64_t k0, int64_t kc, float* dst, int64_t ld) const {
    const Conv2DGeom& g = g_;
    const int64_t kic = g.kic;
    // With dw == 1 and one group, the in-bounds taps of a kernel row are
    // contiguous in memory (merged into one run below).
    const bool merge = g.dw == 1 && kic == g.ic;
    // All divisions are hoisted out of the row loop: (n, oy, ox) advance
    // incrementally and the starting tap is the same for every row.
    const int64_t ohw = g.oh * g.ow, t0 = k0 / kic, c00 = k0 % kic, ky00 = t0 / g.kw, kx00 = t0 % g.kw;
    int64_t n = m0 / ohw, oy = (m0 % ohw) / g.ow, ox = m0 % g.ow;
    for (int64_t r = 0; r < rows; ++r) {
      const float* xn = x + n * g.ih * g.iw * g.ic + coff;
      const int64_t iy0 = oy * g.sh - g.pt, ix0 = ox * g.sw - g.pl;
      if (++ox == g.ow) {
        ox = 0;
        if (++oy == g.oh) {
          oy = 0;
          ++n;
        }
      }
      float* d = dst + r * ld;
      int64_t left = kc, c0 = c00, ky = ky00, kx = kx00;
      while (left > 0) {
        const int64_t iy = iy0 + ky * g.dh, ix = ix0 + kx * g.dw;
        if (iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw) {
          // Merged run: taps kx..min(kw, iw-ix0)-1 of this kernel row in one
          // copy (e.g. 7 taps x 3 channels instead of 7 copies of 3 floats).
          const int64_t run_taps = merge ? std::min(g.kw, g.iw - ix0) - kx : 1;
          const int64_t len = std::min(run_taps * kic - c0, left);
          const float* s = xn + (iy * g.iw + ix) * g.ic + c0;
          copy_floats(d, s, len);
          d += len;
          left -= len;
          // Either the whole run was copied, or `left` hit 0 and the loop ends.
          kx += run_taps;
          c0 = 0;
        } else {
          const int64_t len = std::min(kic - c0, left);
          std::fill_n(d, len, pad_value_);
          d += len;
          left -= len;
          c0 = 0;
          ++kx;
        }
        if (kx == g.kw) {
          kx = 0;
          ++ky;
        }
      }
    }
  }

  // All groups of pixels [m0, m0+rows): group g's patch row r (K = kh*kw*kic
  // floats in (ky, kx, c) order) goes to dst + (r * groups + g) * K, written
  // sequentially from the pixel's tap source pointers (padding taps read
  // padrow_, ic copies of pad_value). Requires kh*kw <= kMaxGroupTaps.
  void im2col_groups(const float* x, int64_t m0, int64_t rows, float* dst) const {
    const Conv2DGeom& g = g_;
    const int64_t kic = g.kic, taps = g.kh * g.kw;
    const float* padrow = static_cast<const float*>(padrow_.get());
    const float* src[kMaxGroupTaps];
    const int64_t ohw = g.oh * g.ow;
    int64_t n = m0 / ohw, oy = (m0 % ohw) / g.ow, ox = m0 % g.ow;
    float* d = dst;
    for (int64_t r = 0; r < rows; ++r) {
      const float* xn = x + n * g.ih * g.iw * g.ic;
      const int64_t iy0 = oy * g.sh - g.pt, ix0 = ox * g.sw - g.pl;
      if (++ox == g.ow) {
        ox = 0;
        if (++oy == g.oh) {
          oy = 0;
          ++n;
        }
      }
      for (int64_t ky = 0, t = 0; ky < g.kh; ++ky) {
        const int64_t iy = iy0 + ky * g.dh;
        for (int64_t kx = 0; kx < g.kw; ++kx, ++t) {
          const int64_t ix = ix0 + kx * g.dw;
          src[t] = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw ? xn + (iy * g.iw + ix) * g.ic : padrow;
        }
      }
      // Fixed-size copies for the common small kic (one or two vector moves).
      for (int64_t off = 0; off < g.ic; off += kic) {
        if (kic == 4) {
          for (int64_t t = 0; t < taps; ++t, d += 4) std::memcpy(d, src[t] + off, 4 * sizeof(float));
        } else if (kic == 8) {
          for (int64_t t = 0; t < taps; ++t, d += 8) std::memcpy(d, src[t] + off, 8 * sizeof(float));
        } else {
          for (int64_t t = 0; t < taps; ++t, d += kic) copy_floats(d, src[t] + off, kic);
        }
      }
    }
  }

  // Narrow direct convolution (see narrow_conv_tile): Simd4 backend, <= 4 outputs per group, stride and dilation 1,
  // kh * kw > 1 or groups > 1, and no GEMV rows in the GEMM plan (pixel blocks of plan_.mb rows: the last block's
  // rows % 8 is 0 or >= 4).
  void prepare_narrow(const KernelContext& ctx) {
    const Conv2DGeom& g = g_;
    const int64_t ocg = g.oc / g.groups, m = g.n * g.oh * g.ow, r = m % plan_.mb % kMR;
    narrow_ = std::is_same<B, Simd4Gemm>::value && ocg <= 4 && g.sh == 1 && g.sw == 1 && g.dh == 1 && g.dw == 1 &&
              (g.kh * g.kw > 1 || g.groups > 1) && (r == 0 || r >= 4);
    // Cortex-A53 (AArch32): the per-run transposition of the input costs more than the im2col it saves for small
    // kernels (128x128 64->3 3x3: GEMM 38.5 ms, narrow 48.3 ms; 9x9: 347 -> 307 ms)
    if (BF_FLOAT_OPT_NEON32 && g.groups == 1 && g.kh * g.kw < 25) narrow_ = false;
    if (!narrow_) return;
    const int64_t k = g.kh * g.kw * g.kic;
    hp_ = g.ih + g.pt + g.pb;
    ws_ = round_up(std::max(g.iw + g.pl + g.pr, round_up(g.ow, 4 * kNarrowPV) + g.kw - 1), 4);
    wn_.assign(static_cast<size_t>(g.groups * k * 4), 0.f);
    nbias_.assign(static_cast<size_t>(g.groups * 4), 0.f);
    const float* w = ctx.in(1).data<float>();
    const float* b = opt_f32(ctx, 2);
    for (int64_t gi = 0; gi < g.groups; ++gi)
      for (int64_t j = 0; j < ocg; ++j) {
        const int64_t o = gi * ocg + j;
        for (int64_t kk = 0; kk < k; ++kk) wn_[(gi * k + kk) * 4 + j] = w[o * k + kk];
        nbias_[gi * 4 + j] = b ? b[o] : 0.f;
      }
    // padded channel rows; the padding (and the columns past the image read by the last tile of a row) keep pad_value
    const size_t floats = static_cast<size_t>(g.n * hp_ * g.ic * ws_);
    xt_ = aligned_alloc_shared(floats * sizeof(float));
    std::fill_n(static_cast<float*>(xt_.get()), floats, pad_value_);
    nscratch_.init(ctx.pool->num_threads(), static_cast<size_t>(g.oc * 4 * kNarrowPV));
  }
  void run_narrow(const KernelContext& ctx, const float* x, float* y) {
    const Conv2DGeom& g = g_;
    float* xt = static_cast<float*>(xt_.get());
    const int64_t ic = g.ic, ws = ws_, hp = hp_;
    const int64_t in_rows = g.n * g.ih, out_rows = g.n * g.oh;
    ctx.pool->parallel_for(in_rows, parallel_grain(in_rows, in_rows * g.iw * ic * 8, kInlineMacs), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t n = r / g.ih, iy = r % g.ih;
        transpose_row(x + r * g.iw * ic, g.iw, ic, xt + ((n * hp + iy + g.pt) * ic) * ws + g.pl, ws);
      }
    });
    const int64_t kic = g.kic, ocg = g.oc / g.groups, kg = g.kh * g.kw * kic, row_macs = g.ow * kg * g.oc;
    constexpr int64_t TP = 4 * kNarrowPV;
    const Activation act = plan_.act;
    ctx.pool->parallel_for(out_rows, parallel_grain(out_rows, out_rows * row_macs, kInlineMacs), [&](int64_t r0, int64_t r1) {
      ScratchLease lease(nscratch_);
      float* t = lease.data();  // [oc][TP]: one tile, all groups
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t n = r / g.oh, oy = r % g.oh;
        float* yr = y + r * g.ow * g.oc;
        for (int64_t ox0 = 0; ox0 < g.ow; ox0 += TP) {
          for (int64_t gi = 0; gi < g.groups; ++gi) {
            const float* xr = xt + ((n * hp + oy) * ic + gi * kic) * ws + ox0;
            const float* wg = wn_.data() + gi * kg * 4;
            const float* bg = nbias_.data() + gi * 4;
            float* tg = t + gi * ocg * TP;
            switch (ocg) {
              case 1: narrow_conv_tile<1>(xr, ic * ws, kic, ws, g.kh, g.kw, wg, bg, act, tg); break;
              case 2: narrow_conv_tile<2>(xr, ic * ws, kic, ws, g.kh, g.kw, wg, bg, act, tg); break;
              case 3: narrow_conv_tile<3>(xr, ic * ws, kic, ws, g.kh, g.kw, wg, bg, act, tg); break;
              default: narrow_conv_tile<4>(xr, ic * ws, kic, ws, g.kh, g.kw, wg, bg, act, tg); break;
            }
          }
          narrow_store(t, TP, g.oc, std::min<int64_t>(TP, g.ow - ox0), yr + ox0 * g.oc);
        }
      }
    });
  }

  Conv2DGeom g_;
  float pad_value_ = 0;
  bool direct_ = false, narrow_ = false;
  int64_t hp_ = 0, ws_ = 0;
  std::vector<float> wn_, nbias_;  // narrow: weights [groups][K / groups][4], bias [groups][4]
  std::shared_ptr<void> xt_;       // narrow: padded channel rows [n][hp][ic][ws]
  ScratchSlots nscratch_;          // narrow: per thread, one output tile [oc][4 * kNarrowPV]
  std::shared_ptr<void> padrow_;  // [ic] pad_value (grouped im2col)
  GemmPlan<B> plan_;
  ScratchSlots scratch_;
};

bool conv_gemm_supported(const KernelContext& ctx) {
  if (ctx.in(0).dtype() != DType::kF32 || ctx.in(1).dtype() != DType::kF32) return false;
  return ctx.attrs().get_int("groups", 1) == 1;
}

// Grouped convolution (groups > 1): one GEMM per group. Depthwise layers
// (kic == 1, channel multiplier 1) are also accepted, but the depthwise
// kernel has the higher priority and takes them.
bool conv_grouped_supported(const KernelContext& ctx) {
  if (ctx.in(0).dtype() != DType::kF32 || ctx.in(1).dtype() != DType::kF32) return false;
  return ctx.attrs().get_int("groups", 1) > 1;
}

// -------------------------------------------------------------------- Dense
template <class B>
class DenseGemm : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Tensor& w = ctx.in(1);
    plan_.act = parse_activation(ctx.attrs());
    plan_.pack(w.data<float>(), opt_f32(ctx, 2), 1, w.dim(0), w.dim(1), w.dim(0));
    plan_.schedule(ctx.in(0).size() / w.dim(1), ctx.pool->num_threads());
    scratch_.init(ctx.pool->num_threads(), 1);
  }
  size_t prepacked_bytes() const override { return plan_.bytes(); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    const int64_t k = plan_.K;
    plan_.run(ctx.pool, scratch_, ctx.out(0).data<float>(), [&](int64_t, int64_t m0, int64_t, int64_t k0, int64_t, float*, int64_t* lda) {
      *lda = k;
      return x + m0 * k + k0;
    });
  }

 private:
  GemmPlan<B> plan_;
  ScratchSlots scratch_;
};

bool dense_supported(const KernelContext& ctx) {
  return ctx.in(0).dtype() == DType::kF32 && ctx.in(1).dtype() == DType::kF32 && ctx.in(1).dim(1) > 0;
}

}  // namespace

void float_opt_gemm_block_simd4(const fopt::GemmBlockArgs& g) { gemm_block_simd4(g); }

void register_float_opt_gemm_kernels(OpRegistry& reg) {
  reg.add_kernel({"Conv2D", "conv2d.float_opt", "float_opt", 40, kFloatOptFeatures, conv_gemm_supported, make_kernel<Conv2DGemm<Simd4Gemm>>});
  reg.add_kernel({"Conv2D", "conv2d_grouped.float_opt", "float_opt", 40, kFloatOptFeatures, conv_grouped_supported,
                  make_kernel<Conv2DGemm<Simd4Gemm>>});
  reg.add_kernel({"Dense", "dense.float_opt", "float_opt", 40, kFloatOptFeatures, dense_supported, make_kernel<DenseGemm<Simd4Gemm>>});
#if defined(__x86_64__)
  reg.add_kernel({"Conv2D", "conv2d.float_opt_avx2", "float_opt", kFloatOptAvx2Priority, kFloatOptAvx2Features, conv_gemm_supported,
                  make_kernel<Conv2DGemm<Avx2Gemm>>});
  reg.add_kernel({"Conv2D", "conv2d_grouped.float_opt_avx2", "float_opt", kFloatOptAvx2Priority, kFloatOptAvx2Features,
                  conv_grouped_supported, make_kernel<Conv2DGemm<Avx2Gemm>>});
  reg.add_kernel({"Dense", "dense.float_opt_avx2", "float_opt", kFloatOptAvx2Priority, kFloatOptAvx2Features, dense_supported,
                  make_kernel<DenseGemm<Avx2Gemm>>});
#endif
}

}  // namespace bitino
