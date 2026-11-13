// Plain-data argument blocks of the float_opt leaf routines (the arithmetic
// inner loops). The kernel classes (packing, scheduling, threading) fill them
// in and call the leaf of their backend: the 4-lane NEON/portable code
// (float_simd.h, compiled with the baseline flags) or the AVX2+FMA code in
// float_opt_avx2.cc (compiled with -mavx2 -mfma, see float_opt_avx2.h).
//
// These structs cross the boundary between translation units compiled with
// different ISA flags, so they are aggregates without default member
// initializers or member functions: nothing here generates code.
#pragma once

#include <cstdint>

namespace bitino {
namespace fopt {

// Activation codes used by the leaves (same values as the Activation enum of
// ops_common.h: 0 none, 1 relu, 2 relu6).
enum : int32_t { kActNone = 0, kActRelu = 1, kActRelu6 = 2 };

// One GEMM block: C[rows][panels p0..p1) (+)= A[rows][kc] * Bp for one K block.
//   A row r starts at a + r * lda.
//   Panel p of the packed B (layout [K][nr], zero padded past the last column)
//   for this K block starts at bp + p * panel_stride.
//   C row r, column j of panel p is c[r * ldc + p * nr + j]; columns at or past
//   ncols (counted from column 0 of panel 0) are never written.
//   bias != nullptr: first K block, start from bias[p * nr + j]; else
//   accumulate onto C. last != 0: apply act before storing.
struct GemmBlockArgs {
  const float* a;
  int64_t lda, rows, kc;
  const float* bp;
  int64_t panel_stride, p0, p1, nr;
  float* c;
  int64_t ldc, ncols;
  const float* bias;
  int32_t last, act;
};

// Depthwise convolution (channel multiplier 1), NHWC. w is the repacked
// filter [kh*kw][cp] followed by the bias [cp] (cp = c rounded up to the
// backend's vector width, zero padded). Output rows r = n * oh + oy.
// [ox_lo, ox_hi) is the horizontal interior (every kx tap in bounds).
struct DwArgs {
  const float* x;
  float* y;
  const float* w;
  int64_t ih, iw, c, cp, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl;
  int64_t ox_lo, ox_hi;
  float pad_value;
  int32_t act;
};

// MaxPool2D / AvgPool2D over output rows r = n * oh + oy.
struct PoolArgs {
  const float* x;
  float* y;
  int64_t ih, iw, c, oh, ow, kh, kw, sh, sw, pt, pl;
  int32_t is_max, include_pad;
  // AvgPool2DPrelu (average pools only): y = avg(prelu(x, slope)), and y2 (if not null) = avg(x)
  const float* slope = nullptr;
  float* y2 = nullptr;
};

// Activation op types handled by the leaves.
enum : int32_t { kActOpRelu = 0, kActOpClamp = 1, kActOpLeaky = 2 };

// ------------------------------------------------- compiled fused programs
// A FusedElementwise stack program (ops/fused_elementwise.h) compiled by
// fused_elementwise.cc into register micro-ops, evaluated by leaf_fe_rows
// (float_opt_leaf.h) on a chunk of a row at a time: the value on top of the
// stack lives in registers ("top"), a binary op's second operand in "opnd",
// deeper stack entries are spilled to a small local array. Every micro-op
// performs exactly the float operation of the instruction it comes from,
// element by element in program order, so the result is bit-identical to one
// pass per instruction.
// Micro-ops reading a source come in three addressing variants, in this
// order: + kFeAtRow, + kFeAtTile, + kFeAtGather (see FeMicro::src).
enum : int32_t { kFeAtRow = 0, kFeAtTile = 1, kFeAtGather = 2 };
enum : int32_t {
  kFePush = 0,        // top = source                         (3 variants)
  kFeOpnd = 3,        // opnd = source                        (3 variants)
  kFeAddSrc = 6,      // top = top + source                   (3 variants)
  kFeMulSrc = 9,      // top = top * source                   (3 variants)
  kFeRAddSrc = 12,    // top = source + top                   (3 variants)
  kFeRMulSrc = 15,    // top = source * top                   (3 variants)
  kFeSpill = 18,      // spill top (before a push onto a non-empty stack)
  kFePop,             // opnd = top, top = last spilled value (first operand)
  kFeAdd,             // top = act(top + opnd), act = FeMicro::act
  kFeRAdd,            // top = act(opnd + top)
  kFeMul,             // top = top * opnd
  kFeAffineSB,        // top = top * p[ch] + q[ch]
  kFeAffineS,         // top = top * p[ch] + 0
  kFeAffineB,         // top = top + q[ch]
  kFeAffineN,         // top = top + 0
  kFeRelu,            // top > 0 ? top : 0
  kFeClamp,           // min(max(top, lo), hi)
  kFeLeaky,           // top >= 0 ? top : top * alpha
  kFePrelu,           // top >= 0 ? top : top * p[ch]
  kFeSign,            // top >= 0 ? 1 : -1
  kFeSign0,           // top > 0 ? 1 : (top < 0 ? -1 : 0)
  kFeGate,            // top = top * factors of the row's batch (row pointer)
  kFeRPrelu,          // affine(bias q) -> prelu(slope p) -> affine(bias r) (ReActNet RPReLU), one dispatch
  kFeNumOps
};

// Source of a micro-op for row r, channel ch (addressing variant in brackets):
enum : int32_t {
  kFeSrcRow = 0,     // [row]    p[r * c + ch] (p = nullptr: the evaluated buffer y itself)
  kFeSrcTile,        // [tile]   p[r * w + ch % w]
  kFeSrcGather,      // [gather] ids[r * (c / w) + ch / w] = id: id in [0, v) ? p[id * w + ch % w] : 0
  kFeSrcTileRow,     // [row]    as kFeSrcTile / kFeSrcGather when w does not fit the chunk width:
  kFeSrcGatherRow,   // [row]      the row is first copied to the per-thread row buffer `slot`
  kFeSrcGatherWhole, // [row]    gather of whole rows (w == c): the row pointer is p + id * c, or q (a row of c zeros)
};

struct FeMicro {
  int32_t op, src, act;
  int32_t slot;  // row buffer index (kFeSrcTileRow / kFeSrcGatherRow)
  int64_t w, v;  // tile / gather piece width, gather table rows
  const float* p;
  const float* q;
  const float* r;
  const int32_t* ids;
  float lo, hi, alpha;
};

// rows_per_batch: batch of row r = r / rows_per_batch (kFeGate: p + batch * c).
// row_ops: indices of the micro-ops that need a per-row pointer (sources, gate).
// flat (with c a multiple of the chunk): rows [r0, r1) are evaluated as one run of chunks (per batch: a gate's
// factors are those of the batch).
// scratch (per thread, pointer aligned): 2 * n_ops pointers, then n_slots * c floats of row buffers.
struct FeEvalArgs {
  const FeMicro* ops;
  int32_t n_ops;
  const int32_t* row_ops;
  int32_t n_row_ops, n_slots;
  int32_t chunk;  // floats per register chunk: the backend's full width, or half of it
  int32_t flat;   // every source is a plain row source (or a gate): rows of one batch may run as one run of chunks
  int64_t c, rows_per_batch;
};

}  // namespace fopt
}  // namespace bitino
