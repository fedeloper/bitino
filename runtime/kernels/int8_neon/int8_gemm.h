// Internal interface of the "int8_neon" tier (AArch64 only).
//
// Split across translation units by ISA level:
//   int8_neon_kernels.cc  baseline ISA: Kernel classes, quantization, im2col,
//                         weight packing, threading, registration
//   int8_gemm_sdot.cc     -march=...+dotprod        SDOT micro-kernel
//   int8_gemm_i8mm.cc     -march=...+dotprod+i8mm   SMMLA micro-kernel
// The micro-kernel TUs include only this header and int8_gemm_epilogue.h:
// inline functions with external linkage compiled under different -march
// flags could be merged by the linker and leak extension instructions into
// code that runs on CPUs without them.
//
// GEMM: y[p][o] = epilogue(sum_k A[p][k] * W[o][k]) over P rows (output
// pixels / Dense rows) and O columns (output channels). Both operands are
// packed in tiles of 8 rows, K split into chunks of C bytes (C = 4 for SDOT,
// 8 for SMMLA) and zero-padded to Kp = kc*C:
//   A  [ceil(P/8)][kc][8 rows][C]      W  [ceil(O/8)][kc][8 cols][C]
// so one chunk of a tile is a contiguous 32 (SDOT) or 64 (SMMLA) byte block,
// exactly what one step of the micro-kernel loads.
#pragma once

#include <cstdint>

namespace bitino {

class ThreadPool;

namespace int8_neon {

constexpr int kTile = 8;  // rows (pixels) and columns (channels) per tile

struct GemmArgs {
  const int8_t* a = nullptr;      // packed activations (see above)
  const int8_t* w = nullptr;      // packed weights
  const float* scale = nullptr;   // [ceil(O/8)*8] scale_x * scale_w[o], 0 in padding
  const float* bias = nullptr;    // [ceil(O/8)*8] bias[o], +0 when absent / padding
  float* y = nullptr;             // [P][O] row-major fp32 output
  int64_t rows = 0, cols = 0;     // P, O
  int64_t kc = 0;                 // K chunks
  int act = 0;                    // 0 none, 1 relu, 2 relu6
};

// Computes the output tiles (pb, ob) for ob in [ob0, ob1).
void gemm_tiles_sdot(const GemmArgs& args, int64_t pb, int64_t ob0, int64_t ob1);
void gemm_tiles_i8mm(const GemmArgs& args, int64_t pb, int64_t ob0, int64_t ob1);

// Vectorized, multithreaded per-tensor quantization (int8_quant.h contract):
// returns scale_x and writes q[0..n). `partial` holds `parts` floats of
// scratch for the max-abs reduction (parts = quant_parts(n)).
int64_t quant_parts(int64_t n);
float quantize_tensor(const float* x, int64_t n, int8_t* q, float* partial, int64_t parts, ThreadPool& pool);

}  // namespace int8_neon
}  // namespace bitino
