// Internal interface of the "int8_avx2" tier (x86-64 only): dynamic-range
// INT8 Conv2D (groups == 1) and Dense, bit-exact with "int8_reference"
// (contract: kernels/int8_quant.h, docs/int8.md).
//
// Split across translation units by ISA level, as int8_neon:
//   int8_avx2_kernels.cc   baseline x86-64: Kernel classes, weight
//                          quantization/packing, im2col, threading,
//                          registration
//   int8_gemm_avx2.cc      -mavx2: activation max-abs + quantization, and the
//                          VPMADDWD micro-kernel
//   int8_gemm_avxvnni.cc   -mavx2 -mavxvnni: the VPDPBUSD micro-kernel
// The ISA translation units include only this header and <immintrin.h>:
// inline functions with external linkage compiled under different flags could
// be merged by the linker and leak AVX instructions into code that runs on
// CPUs without them.
//
// GEMM: y[p][o] = epilogue(sum_k qx[p][k] * qw[o][k]) over P rows (output
// pixels / Dense rows) and O columns (output channels), K split into chunks
// of 4 bytes:
//   VNNI  4 k per chunk, A as u8 = qx + 128, W as s8. VPDPBUSD adds
//         sum_t A[4c+t] * W[4c+t] into each int32 lane without saturation
//         (wrapping), so acc = sum_k (qx+128)*qw = sum_k qx*qw + 128*sum_k qw
//         modulo 2^32; subtracting corr[o] = 128*sum_k qw[o][k] (also modulo
//         2^32) leaves sum_k qx*qw, which |.| <= K*16129 < 2^31 makes exact.
//   MADD  2 k per chunk, A and W as int16 (sign-extended int8). VPMADDWD
//         gives a0*w0 + a1*w1 per lane exactly (|.| <= 2*16129); the int32
//         accumulation is exact for K <= kInt8MaxK. (VPMADDUBSW is not used:
//         its int16 pair sums saturate.)
// Layouts (bytes):
//   A  row-major, row r at a + r*lda, chunk c at offset 4c; K padding is 0
//   W  [ceil(O/16)][kc][16 columns][4 bytes], zero padded (columns and K)
// Micro-tile: 6 rows x 16 columns (12 int32 accumulators, 2 W loads and 6
// 4-byte broadcasts per chunk); short tiles use extra accumulator sets over k.
#pragma once

#include <cstdint>

namespace bitino {
namespace int8_avx2 {

constexpr int kMR = 6;   // rows per micro-tile
constexpr int kNR = 16;  // columns per W panel

struct GemmArgs {
  const uint8_t* a;     // A rows (see above)
  int64_t lda;          // bytes between A rows
  const int8_t* w;      // packed W panels
  const int32_t* corr;  // [panels*16] VNNI correction (128*sum_k qw), nullptr for MADD
  const float* scale;   // [panels*16] scale_x * scale_w[o], 0 in padding
  const float* bias;    // [panels*16] bias[o], +0 when absent / padding
  float* y;             // [P][cols] row-major fp32 output
  int64_t cols, kc;     // O, K chunks
  int32_t act;          // 0 none, 1 relu, 2 relu6
};

// Output rows [r0, r0+rows) x panels [p0, p1).
void gemm_vnni(const GemmArgs& g, int64_t r0, int64_t rows, int64_t p0, int64_t p1);
void gemm_madd(const GemmArgs& g, int64_t r0, int64_t rows, int64_t p0, int64_t p1);

// Activation quantization (int8_quant.h): max |x| over [0, n), and
//   q = clamp(rne(x / scale), -127, 127)
// with an IEEE division and the MXCSR rounding mode (round to nearest even,
// as std::nearbyint), written as u8 = q + 128 (VNNI) or int16 (MADD).
float maxabs(const float* x, int64_t n);
void quantize_u8(const float* x, int64_t n, float scale, uint8_t* q);
void quantize_s16(const float* x, int64_t n, float scale, int16_t* q);

// W panels at run time (INT8 MatMul) from a K x N matrix of int16 values in
// [-127, 127] (row stride ldq): VNNI (vnni = true) bytes, or MADD int16 pairs,
// in the layout above (K padding and columns past N zero); VNNI also writes
// corr[j] = 128 * sum_k q[k][j] modulo 2^32 for j < N.
void pack_w_kn(const int16_t* q, int64_t k, int64_t n, int64_t ldq, int64_t kc, bool vnni, uint8_t* w, int32_t* corr);

}  // namespace int8_avx2
}  // namespace bitino
