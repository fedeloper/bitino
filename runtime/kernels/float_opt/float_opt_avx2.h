// AVX2 + FMA backend of the float_opt tier (x86-64 only).
//
// float_opt_avx2.cc is the only translation unit compiled with -mavx2 -mfma
// (per-file flags in CMakeLists.txt). It contains just the arithmetic leaf
// routines declared here; the kernel classes (weight packing, scheduling,
// threading, registration) stay in the baseline translation units
// (conv_gemm.cc, depthwise.cc, elementwise.cc) and call these functions only
// from kernels registered with required_features = kFeatAvx2 | kFeatFma, so
// the default x86-64 binary still runs on CPUs without AVX2 (the selector then
// keeps the portable float_opt kernels).
//
// float_opt_avx2.cc includes nothing that defines an inline function with
// external linkage (see float_opt_leaf.h): such a function, emitted with AVX
// instructions in this TU, could be chosen by the linker for baseline code.
#pragma once

#include <cstdint>

#include "kernels/float_opt/float_opt_args.h"

namespace bitino {
namespace fopt_avx2 {

constexpr int kLanes = 8;  // floats per vector (depthwise channel padding, GAP blocks)

// GEMM panel widths: 16 (6x16 micro-kernel: 12 accumulators, 2 B loads and 6
// broadcasts per k) or 8 (12x8: 12 accumulators, 1 B load and 12
// broadcasts), for problems whose column count would leave a 16-wide panel
// mostly empty (e.g. 8 output channels per group). Row tiles of 6 / 12.
void gemm_block(const fopt::GemmBlockArgs& g);  // g.nr must be 8 or 16

void depthwise_rows(const fopt::DwArgs& d, int64_t r0, int64_t r1);
void add(const float* a, const float* b, float* y, int64_t i0, int64_t i1, int32_t act);
void affine_rows(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t r0, int64_t r1);
void activation(int32_t type, const float* x, float* y, int64_t i0, int64_t i1, float lo, float hi, float alpha);
void prelu_rows(const float* x, const float* slope, float* y, int64_t c, int64_t r0, int64_t r1);
void pool_rows(const fopt::PoolArgs& g, int64_t r0, int64_t r1);
void global_avg_pool(const float* x, float* y, int64_t hw, int64_t c, int64_t t0, int64_t t1);
void mul(const float* a, const float* b, float* y, int64_t i0, int64_t i1);
void mul_rows(const float* x, const float* g, float* y, int64_t c, int64_t r0, int64_t r1);
void sign(const float* x, float* y, int64_t i0, int64_t i1, int32_t zero);
// Compiled FusedElementwise program on rows [r0, r1) (fopt::leaf_fe_rows), chunks of kFeChunk floats.
constexpr int kFeChunk = 32;
void fe_rows(const fopt::FeEvalArgs& e, float* y, int64_t r0, int64_t r1, void* scratch);

// Transformer / BHViT operators (transformer_leaf.h; numerics documented there).
void unary(int32_t op, const float* x, float* y, int64_t i0, int64_t i1);  // gelu, sigmoid, tanh, hardswish
void softmax_rows(const float* x, float* y, int64_t c, int64_t r0, int64_t r1);
void fake_quant(const float* x, const float* s, const float* z, float* y, int64_t len, int64_t inner, int64_t r0, int64_t r1, float pre,
                float lo, float hi);
void add_run(const float* a, const float* b, float* y, int64_t n);
void add_scalar_run(const float* a, float b, float* y, int64_t n);
void scale_run(float* y, float s, int64_t n);

// Double-precision helpers of the bit-exact LayerNorm / ReduceMean kernels:
// kRowsF64 rows at a time (row j at x + j * stride), every per-row sum in
// index order with one double rounding per operation, as the reference.
constexpr int kRowsF64 = 4;
void rowsum_f64(const float* x, int64_t stride, int64_t len, double* out);  // out[j] = sum_i (double)x_j[i]
void rowsqdev_f64(const float* x, int64_t stride, int64_t len, const double* mean, double* out);  // sum_i (x_j[i]-mean_j)^2
// y[i] = float(((double(x[i]) - mean) / sd) * g[i] + b[i]), g / b optional.
void layernorm_row(const float* x, const float* g, const float* b, double mean, double sd, float* y, int64_t len);
// y[i] = float((sum_k (double)x[k * inner + i]) / cnt) for i in [i0, i1), k in order.
void colmean_f64(const float* x, int64_t len, int64_t inner, double cnt, float* y, int64_t i0, int64_t i1);

}  // namespace fopt_avx2
}  // namespace bitino
