// INT8 dynamic-range quantization contract for the float Conv2D and Dense ops,
// shared by the "int8_reference" and "int8_neon" tiers (docs/int8.md).
//
// These tiers APPROXIMATE the float ops; the selector only uses them when the
// session policy lists them explicitly (see KernelPolicy in session.h).
//
// prepare() -- weights, per output channel o, symmetric:
//   W[o][k]      k runs over kh*kw*kic (Conv2D, OHWI) or K (Dense, [O,K])
//   m_o          = max_k |W[o][k]|
//   scale_w[o]   = m_o / 127            (fp32 division; 1 if that is 0, i.e.
//                                        m_o == 0 or m_o < ~127 * 2^-150)
//   qw[o][k]     = clamp(rne(W[o][k] / scale_w[o]), -127, 127)
//
// run() -- activations, per tensor, symmetric:
//   m            = max |x| over EVERY element of the input tensor (all batch
//                  entries / rows)
//   scale_x      = m / 127              (1 if that is 0, as for scale_w)
//   qx           = clamp(rne(x / scale_x), -127, 127)
//   qpad         = clamp(rne(pad_value / scale_x), -127, 127)
//                  (Conv2D spatial padding taps; pad_value does not take part
//                  in m, so it saturates when |pad_value| > m)
//   acc[p][o]    = sum_k qx * qw        exactly, in int32
//   y[p][o]      = act( float(acc) * (scale_x * scale_w[o]) + bias[o] )
//                  evaluated in fp32 in exactly that order, one rounding per
//                  operation, no FMA. An absent bias is +0.0f and is still
//                  added. act is apply_activation() (ops_common.h).
//
// rne = round to nearest, ties to even: std::nearbyint under the default
// FE_TONEAREST mode, identical to AArch64 FCVTNS (vcvtnq_s32_f32). Every
// x / scale is a true IEEE fp32 division, never a multiply by a reciprocal:
// x * (1/s) differs from x / s in the last ulp for some x, which can move a
// value across a .5 rounding boundary. With these rules every int8 kernel is
// bit-exact with every other one, whatever its blocking or threading.
//
// int32 exactness: |qx|,|qw| <= 127, so |acc| (and every partial sum) is at
// most K * 16129; K <= kInt8MaxK = floor((2^31 - 1) / 16129) = 133144 keeps it
// below 2^31. Kernels reject larger K in supports() (the selector falls back).
// Inputs must be finite; NaN/Inf in x or W is outside the contract.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "bitino/op.h"
#include "ops/ops_common.h"

namespace bitino {

constexpr int64_t kInt8MaxK = 133144;
static_assert(kInt8MaxK * 127 * 127 <= INT32_MAX && (kInt8MaxK + 1) * 127 * 127 > INT32_MAX, "int8 K bound");

// A subnormal max-abs can round m / 127 to 0; dividing by a zero scale would
// give 0/0 = NaN, and NaN -> int8 is undefined. Scale 1 quantizes such
// tensors to all zeros instead.
inline float int8_scale(float maxabs) {
  const float s = maxabs / 127.f;
  return s > 0.f ? s : 1.f;
}

// Per-node opt-out: a Conv2D/Dense node with attribute "allow_int8": 0 is never
// executed by an int8 kernel (used to keep accuracy-sensitive layers in float).
inline bool int8_allowed(const KernelContext& ctx) { return ctx.attrs().get_int("allow_int8", 1) != 0; }

inline int8_t int8_quantize(float v, float scale) {
  float r = std::nearbyint(v / scale);
  r = std::min(std::max(r, -127.f), 127.f);
  return static_cast<int8_t>(r);
}

// Per-row symmetric quantization of a [rows][k] fp32 matrix (weights).
inline void int8_quantize_rows(const float* w, int64_t rows, int64_t k, int8_t* q, float* scale) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* wr = w + r * k;
    float m = 0.f;
    for (int64_t j = 0; j < k; ++j) m = std::max(m, std::fabs(wr[j]));
    scale[r] = int8_scale(m);
    for (int64_t j = 0; j < k; ++j) q[r * k + j] = int8_quantize(wr[j], scale[r]);
  }
}

inline float int8_epilogue(int32_t acc, float scale_x, float scale_w, float bias, Activation act) {
  float s = scale_x * scale_w;
  float v = static_cast<float>(acc) * s;
  v = v + bias;
  return apply_activation(act, v);
}

}  // namespace bitino
