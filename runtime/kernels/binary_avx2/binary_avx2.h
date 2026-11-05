// Internal interface of the "binary_avx2" tier (x86-64 with AVX2 + POPCNT).
//
// Split across translation units by ISA level (like int8_neon/int8_gemm.h):
//   binary_avx2_kernels.cc  baseline x86-64: Kernel classes, weight packing,
//                           im2col (bconv_plan.h), threading, registration
//   binary_avx2_simd.cc     -mavx2 -mpopcnt: micro-kernel,
//                           epilogue, f32 -> bit packing
// The SIMD TU includes only this header and system headers: inline functions
// with external linkage compiled with -mavx2 could be merged by the linker
// with baseline copies and leak AVX2 instructions into code that runs on any
// x86-64. It is only called after cpu_info() reported avx2 and popcnt.
//
// Popcount strategy (measured, the development log (not published)): nibble
// lookup with VPSHUFB, uint8 accumulation for up to 31 steps, VPSADBW fold.
// A 256-bit register holds 64 bits of 4 output channels ("lanes"); the 64-bit
// chunk of the activation row is broadcast, so patch rows only need padding
// to 64 bits:
//   weights  [O_pad/4 quads][steps][4 channels][uint64]   (32 bytes per step)
//   rows     [R][2*steps uint32]                           (steps = 64-bit chunks)
// Micro-tile: 2 quads (8 output channels) x up to 4 rows.
#pragma once

#include <cstdint>

namespace bitino {
namespace avx2 {

constexpr int kOcBlock = 8;  // output channels per micro-tile (2 lane quads)
constexpr int kRowBlock = 4;

struct TileArgs {
  const uint8_t* wts = nullptr;      // interleaved weight quads (padded channel index, see above)
  int64_t steps = 0;                 // 64-bit chunks per row
  int32_t k_bits = 0;                // real bits per row
  const int32_t* tap_sum = nullptr;  // [taps][oc_pad] S[o][tap] (padded index); used iff pad_scale != 0
  int64_t oc_pad = 0;
  int32_t pad_scale = 0;             // pad_value - 1
  const float* mul = nullptr;        // [oc_pad], 1 in padding / when absent
  const float* add = nullptr;        // [oc_pad], -0 in padding / when absent
  int act = 0;                       // 0 none, 1 relu, 2 relu6
  float* yf = nullptr;               // f32 output [rows][y_stride] or nullptr
  uint32_t* yb = nullptr;            // bit output [rows][y_stride words] or nullptr
  int64_t y_stride = 0;              // floats (f32) or words (bit) per output row
  int64_t oc_total = 0;              // real output channels (bit padding)
  bool bit_rmw = false;              // bit output at channel offsets that are not multiples of 8:
                                     // OR into words the caller zeroed
  const float* out_thr = nullptr;    // bit output: threshold by padded channel index [oc_pad] (nullptr = 0)
  bool out_gt = false;               // bit output: bit = !(v > t) instead of !(v >= t)
  const int32_t* out_lim = nullptr;  // bit output: integer thresholds [oc_pad] (IntThresholds, kernels/bit_output.h);
  const int32_t* out_neg = nullptr;  //   bit = neg ? dot > lim : dot < lim, no float epilogue (nullptr: float compare)
};

// Output rows [0, np) of a tile whose patch rows start at `rows` (row r at
// rows + r * 2 * steps words) and map to output row index q0 + r; `mask[r]`
// is the out-of-bounds tap mask of row r (nullptr: none). Computes padded
// channels pb + j for j in [j0, j1) (j0 multiple of 8, j1 <= padded group
// width) and stores the first `ocg - j` valid ones at output channel ob + j.
void bconv_tile(const TileArgs& a, const uint32_t* rows, const uint64_t* mask, int64_t q0, int64_t np, int64_t pb, int64_t ob,
                int64_t j0, int64_t j1, int64_t ocg);

// f32 -> bits with the exact semantics of pack_row_f32() (bit = 1 iff
// !(x >= t) for ge, !(x > t) for gt; NaN -> 1). `thr` (nullable) is indexed
// like x. pack_words: `words` full groups of 32 floats.
void pack_words(const float* x, int64_t words, uint32_t* out, bool gt, const float* thr);
void pack_row(const float* x, int64_t c, uint32_t* out, bool gt, const float* thr);

// ---- Ternary BDense (bdense.ternary_avx2): x in {-1, 0, +1} (sign_zero),
// w = ±1 with an optional zero mask, as two bit-planes (see
// ternary_scalar/ternary_kernels.cc): nz (value != 0) and neg (value < 0).
//   x·w = popcount(m) - 2·popcount(m & (neg_x XOR neg_w)),  m = nz_x & nz_w.
// Weight planes use the lane-quad layout above; activation planes are rows of
// 2*steps words (padding bits 0 in both planes).
struct TernaryArgs {
  const uint8_t* wneg = nullptr;  // interleaved weight sign quads
  const uint8_t* wnz = nullptr;   // interleaved weight non-zero quads, nullptr = no zero weights
  int64_t steps = 0;
  const float* mul = nullptr;     // [oc_pad]
  const float* add = nullptr;     // [oc_pad]
  int act = 0;
  float* y = nullptr;             // [rows][y_stride]
  int64_t y_stride = 0;
};

// Rows [0, np) (planes at xnz / xneg + r * 2 * steps words, output row q0 + r),
// channels [j0, j1) (multiples of 8; the first `oc - j` valid ones are stored).
void ternary_tile(const TernaryArgs& a, const uint32_t* xnz, const uint32_t* xneg, int64_t q0, int64_t np, int64_t j0, int64_t j1,
                  int64_t oc);

// sign_zero bit-planes of `c` floats (threshold `thr`, nullable): nz bit = (x != t)
// ordered (NaN -> 0), neg bit = (x < t); exactly binarize(kSignZero, x, t).
void pack_ternary_row(const float* x, int64_t c, uint32_t* nz, uint32_t* neg, const float* thr);

// ---- k-bit activations x binary weights (qbconv2d.avx2 / qbdense.avx2,
// DoReFa levels, docs/operators_larq.md), k = planes <= kMaxQPlanes. With
// level = Σ_b 2^b·bit_b and w = ±1 (bit 1 = -1):
//   Σ level·w = X - 2·D,  X = Σ_b 2^b·popcount(bit_b),  D = Σ_b 2^b·popcount(bit_b & neg_w).
// Activation rows hold the planes back to back: row r, plane b at
// rows + (r * planes + b) * 2 * steps words. Weights: lane quads as above.
constexpr int kMaxQPlanes = 4;

struct QTileArgs {
  const uint8_t* wts = nullptr;
  int64_t steps = 0;
  int planes = 0;
  const float* mul = nullptr;  // [oc_pad]
  const float* add = nullptr;  // [oc_pad]
  int act = 0;
  float* y = nullptr;          // [rows][y_stride]
  int64_t y_stride = 0;
};

void qbinary_tile(const QTileArgs& a, const uint32_t* rows, int64_t q0, int64_t np, int64_t j0, int64_t j1, int64_t oc);

// DoReFa level bit-planes of `c` floats: level = nearbyint(clip(x, 0, 1) * n)
// with one float32 multiply (exactly dorefa_level() in larq_reference.cc);
// plane b (at out + b * plane_stride words) holds bit b of each level.
void pack_levels_row(const float* x, int64_t c, float n, int planes, uint32_t* out, int64_t plane_stride);

// ---- BMatMul (bmatmul.avx2): both operands as {nz, neg} bit-planes (BMode
// encoding of kernels/binary_scalar/bmatmul_bitplane.cc), rows of w64 64-bit
// words per plane, padding bits 0.
// Packs `k` floats with BMode `mode` (0 ge, 1 gt, 2 sign_zero, 3 gt0) into
// nz[0..2*w64) and neg[0..2*w64) 32-bit words.
void bmm_pack_row(int mode, const float* x, int64_t k, int64_t w64, uint32_t* nz, uint32_t* neg);
// y[j] = float(sum_k a.b) * scale for j in [0, n): `a` = row planes {nz[w64], neg[w64]},
// `b` = n column rows of {nz[w64], neg[w64]}.
void bmm_dot_row(const uint64_t* a, const uint64_t* b, int64_t n, int64_t w64, float scale, float* y);
// All m rows of `a` ({nz[w64], neg[w64]} per row) against b in column layout:
// plane p (0 nz, 1 neg) of word w for columns j at bc[(w * 2 + p) * npad + j]
// (npad multiple of 8, zero planes past n). y[r][j] = float(sum) * scale for
// j < n. VPSHUFB popcounts over 4 columns per vector, 8 columns per step.
void bmm_dot_rows_cols(const uint64_t* a, int64_t m, const uint64_t* bc, int64_t n, int64_t npad, int64_t w64, float scale, float* y);

}  // namespace avx2
}  // namespace bitino
