// float_opt leaf routines shared by every backend: depthwise convolution,
// elementwise ops and pools, written once against a vector policy V and
// instantiated with
//   Simd4 (float_simd.h: NEON on AArch64, plain C++ elsewhere; baseline flags)
//   Simd8 (float_opt_avx2.cc: AVX2 + FMA, compiled with -mavx2 -mfma).
//
// Policy V: a vector type V::T of V::W floats and static functions
//   load store dup add mul div madd(acc,a,b)=acc+a*b relu clamp(v,lo,hi)
//   select_ge0(v,neg) max_num(m,x) sign(v) (v >= 0 ? 1 : -1) sign0(v) (torch.sign)
// with the scalar semantics documented at each use below.
//
// Linkage rule (multi-ISA build): everything in this header has internal
// linkage (anonymous namespace) and calls no inline function with external
// linkage (no std::min/max, no apply_activation): an external inline function
// emitted by the AVX2 translation unit could be picked by the linker for the
// baseline code and leak AVX instructions onto CPUs without them. The scalar
// helpers below reproduce the reference semantics exactly instead.
#pragma once

#include <cstdint>
#include <utility>

#include "kernels/float_opt/float_opt_args.h"

namespace bitino {
namespace fopt {
namespace {

// std::max / std::min, reproduced (same comparisons, same NaN behaviour).
inline float leaf_max(float a, float b) { return (a < b) ? b : a; }
inline float leaf_min(float a, float b) { return (b < a) ? b : a; }

// apply_activation() of ops_common.h, reproduced exactly.
inline float leaf_act(int32_t act, float v) {
  if (act == kActRelu) return v > 0.f ? v : 0.f;
  if (act == kActRelu6) return leaf_min(leaf_max(v, 0.f), 6.f);
  return v;
}

inline int64_t leaf_max_i(int64_t a, int64_t b) { return a < b ? b : a; }
inline int64_t leaf_min_i(int64_t a, int64_t b) { return b < a ? b : a; }

// Fused activation on a vector, lane by lane equal to leaf_act (relu maps NaN
// to 0, relu6 keeps NaN).
template <class V>
inline typename V::T leaf_activate(int32_t act, typename V::T v) {
  if (act == kActRelu) return V::relu(v);
  if (act == kActRelu6) return V::clamp(v, V::dup(0.f), V::dup(6.f));
  return v;
}

// ---------------------------------------------------------------- depthwise
// Direct convolution over W-channel vectors. Each output row is split into
// left border, interior (every tap in bounds) and right border pixels. 3x3
// interiors keep the 9 tap weights of a channel block in registers across
// the row (blocks of 2W, then W channels); other pixels go channel blocks of
// 4W (4 accumulators), then W, then a scalar tail; on border pixels
// out-of-bounds taps contribute pad_value * w (skipped when 0).
template <class V>
struct Depthwise {
  using T = typename V::T;
  static constexpr int W = V::W;
  const DwArgs& d;

  void scalar(const float* xn, int64_t iy0, int64_t ix0, int64_t ch, float* yp) const {
    const float* w = d.w;
    float acc = w[d.kh * d.kw * d.cp + ch];
    for (int64_t ky = 0; ky < d.kh; ++ky) {
      const int64_t iy = iy0 + ky * d.dh;
      for (int64_t kx = 0; kx < d.kw; ++kx) {
        const int64_t ix = ix0 + kx * d.dw;
        const bool in = iy >= 0 && iy < d.ih && ix >= 0 && ix < d.iw;
        acc += (in ? xn[(iy * d.iw + ix) * d.c + ch] : d.pad_value) * w[(ky * d.kw + kx) * d.cp + ch];
      }
    }
    yp[ch] = leaf_act(d.act, acc);
  }

  template <int NV, bool kInside>
  void block(const float* xn, int64_t iy0, int64_t ix0, int64_t ch, float* yp) const {
    const float* w = d.w;
    const int64_t taps = d.kh * d.kw;
    T acc[NV];
    for (int v = 0; v < NV; ++v) acc[v] = V::load(w + taps * d.cp + ch + W * v);
    const T pv = V::dup(d.pad_value);
    for (int64_t ky = 0; ky < d.kh; ++ky) {
      const int64_t iy = iy0 + ky * d.dh;
      const bool row_ok = kInside || (iy >= 0 && iy < d.ih);
      for (int64_t kx = 0; kx < d.kw; ++kx) {
        const int64_t ix = ix0 + kx * d.dw;
        const float* wp = w + (ky * d.kw + kx) * d.cp + ch;
        if (kInside || (row_ok && ix >= 0 && ix < d.iw)) {
          const float* xp = xn + (iy * d.iw + ix) * d.c + ch;
          for (int v = 0; v < NV; ++v) acc[v] = V::madd(acc[v], V::load(xp + W * v), V::load(wp + W * v));
        } else if (d.pad_value != 0.f) {
          for (int v = 0; v < NV; ++v) acc[v] = V::madd(acc[v], pv, V::load(wp + W * v));
        }
      }
    }
    for (int v = 0; v < NV; ++v) V::store(yp + ch + W * v, leaf_activate<V>(d.act, acc[v]));
  }

  template <bool kInside>
  void pixel(const float* xn, int64_t iy0, int64_t ix0, float* yp) const {
    int64_t ch = 0;
    for (; ch + 4 * W <= d.c; ch += 4 * W) block<4, kInside>(xn, iy0, ix0, ch, yp);
    for (; ch + W <= d.c; ch += W) block<1, kInside>(xn, iy0, ix0, ch, yp);
    for (; ch < d.c; ++ch) scalar(xn, iy0, ix0, ch, yp);
  }

  template <int NV>
  void row_3x3(const float* xn, int64_t iy0, int64_t ch, float* yr) const {
    const float* w = d.w;
    T wv[9][NV], bv[NV];
    for (int t = 0; t < 9; ++t)
      for (int v = 0; v < NV; ++v) wv[t][v] = V::load(w + t * d.cp + ch + W * v);
    for (int v = 0; v < NV; ++v) bv[v] = V::load(w + 9 * d.cp + ch + W * v);
    const int64_t row = d.dh * d.iw * d.c, tap = d.dw * d.c, step = d.sw * d.c;
    const float* xp = xn + (iy0 * d.iw + d.ox_lo * d.sw - d.pl) * d.c + ch;
    float* yp = yr + d.ox_lo * d.c + ch;
    for (int64_t ox = d.ox_lo; ox < d.ox_hi; ++ox, xp += step, yp += d.c) {
      T acc[NV];
      for (int v = 0; v < NV; ++v) acc[v] = bv[v];
      for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) {
          const float* p = xp + ky * row + kx * tap;
          for (int v = 0; v < NV; ++v) acc[v] = V::madd(acc[v], V::load(p + W * v), wv[ky * 3 + kx][v]);
        }
      for (int v = 0; v < NV; ++v) V::store(yp + W * v, leaf_activate<V>(d.act, acc[v]));
    }
  }

  void interior_3x3(const float* xn, int64_t iy0, float* yr) const {
    int64_t ch = 0;
    for (; ch + 2 * W <= d.c; ch += 2 * W) row_3x3<2>(xn, iy0, ch, yr);
    for (; ch + W <= d.c; ch += W) row_3x3<1>(xn, iy0, ch, yr);
    for (; ch < d.c; ++ch)
      for (int64_t ox = d.ox_lo; ox < d.ox_hi; ++ox) scalar(xn, iy0, ox * d.sw - d.pl, ch, yr + ox * d.c);
  }

  void rows(int64_t r0, int64_t r1) const {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t n = r / d.oh, oy = r % d.oh;
      const int64_t iy0 = oy * d.sh - d.pt;
      const bool y_inside = iy0 >= 0 && iy0 + d.dh * (d.kh - 1) < d.ih;
      const float* xn = d.x + n * d.ih * d.iw * d.c;
      float* yr = d.y + r * d.ow * d.c;
      if (!y_inside) {
        for (int64_t ox = 0; ox < d.ow; ++ox) pixel<false>(xn, iy0, ox * d.sw - d.pl, yr + ox * d.c);
        continue;
      }
      for (int64_t ox = 0; ox < d.ox_lo; ++ox) pixel<false>(xn, iy0, ox * d.sw - d.pl, yr + ox * d.c);
      if (d.kh == 3 && d.kw == 3) {
        interior_3x3(xn, iy0, yr);
      } else {
        for (int64_t ox = d.ox_lo; ox < d.ox_hi; ++ox) pixel<true>(xn, iy0, ox * d.sw - d.pl, yr + ox * d.c);
      }
      for (int64_t ox = d.ox_hi; ox < d.ow; ++ox) pixel<false>(xn, iy0, ox * d.sw - d.pl, yr + ox * d.c);
    }
  }
};

template <class V>
void leaf_depthwise_rows(const DwArgs& d, int64_t r0, int64_t r1) {
  Depthwise<V>{d}.rows(r0, r1);
}

// -------------------------------------------------------------- elementwise
// Add, ChannelAffine and Activation use exactly the reference's float
// operations (no FMA), so every backend is bit-identical to the reference.

// y[i] = act(a[i] + b[i]) for i in [i0, i1).
template <class V>
void leaf_add(const float* a, const float* b, float* y, int64_t i0, int64_t i1, int32_t act) {
  constexpr int W = V::W;
  int64_t i = i0;
  for (; i + 2 * W <= i1; i += 2 * W) {
    V::store(y + i, leaf_activate<V>(act, V::add(V::load(a + i), V::load(b + i))));
    V::store(y + i + W, leaf_activate<V>(act, V::add(V::load(a + i + W), V::load(b + i + W))));
  }
  for (; i + W <= i1; i += W) V::store(y + i, leaf_activate<V>(act, V::add(V::load(a + i), V::load(b + i))));
  for (; i < i1; ++i) y[i] = leaf_act(act, a[i] + b[i]);
}

// Rows [r0, r1) of c channels: y = x * s + b. A missing scale/bias behaves as
// 1 / 0 and is computed the same way as the reference (x * 1 + b, x * s + 0;
// x * 1 == x for every non-signaling input, so that multiply is skipped, but
// the + 0 is kept: it turns -0 into +0 as in the reference).
template <class V, bool kS, bool kB>
void leaf_affine_rows_t(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t r0, int64_t r1) {
  using T = typename V::T;
  constexpr int W = V::W;
  const T zero = V::dup(0.f);
  auto one = [&](const float* xp, int64_t i) -> T {
    T v = V::load(xp + i);
    if (kS) v = V::mul(v, V::load(s + i));
    return V::add(v, kB ? V::load(b + i) : zero);
  };
  for (int64_t r = r0; r < r1; ++r) {
    const float* xp = x + r * c;
    float* yp = y + r * c;
    int64_t i = 0;
    for (; i + 2 * W <= c; i += 2 * W) {
      const T v0 = one(xp, i), v1 = one(xp, i + W);
      V::store(yp + i, v0);
      V::store(yp + i + W, v1);
    }
    for (; i + W <= c; i += W) V::store(yp + i, one(xp, i));
    for (; i < c; ++i) yp[i] = (kS ? xp[i] * s[i] : xp[i]) + (kB ? b[i] : 0.f);
  }
}

template <class V>
void leaf_affine_rows(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t r0, int64_t r1) {
  if (s && b) leaf_affine_rows_t<V, true, true>(x, s, b, y, c, r0, r1);
  else if (s) leaf_affine_rows_t<V, true, false>(x, s, b, y, c, r0, r1);
  else if (b) leaf_affine_rows_t<V, false, true>(x, s, b, y, c, r0, r1);
  else leaf_affine_rows_t<V, false, false>(x, s, b, y, c, r0, r1);
}

// Elementwise activation over [i0, i1):
//   kActOpRelu  : x > 0 ? x : 0
//   kActOpClamp : std::min(std::max(x, lo), hi)   (relu6, hardtanh)
//   kActOpLeaky : x >= 0 ? x : x * alpha
template <class V>
void leaf_activation(int32_t type, const float* x, float* y, int64_t i0, int64_t i1, float lo, float hi, float alpha) {
  constexpr int W = V::W;
  const typename V::T lov = V::dup(lo), hiv = V::dup(hi), av = V::dup(alpha);
  int64_t i = i0;
  switch (type) {
    case kActOpRelu:
      for (; i + W <= i1; i += W) V::store(y + i, V::relu(V::load(x + i)));
      for (; i < i1; ++i) y[i] = x[i] > 0.f ? x[i] : 0.f;
      break;
    case kActOpClamp:
      for (; i + W <= i1; i += W) V::store(y + i, V::clamp(V::load(x + i), lov, hiv));
      for (; i < i1; ++i) y[i] = leaf_min(leaf_max(x[i], lo), hi);
      break;
    case kActOpLeaky:
      for (; i + W <= i1; i += W) {
        typename V::T v = V::load(x + i);
        V::store(y + i, V::select_ge0(v, V::mul(v, av)));
      }
      for (; i < i1; ++i) y[i] = x[i] >= 0.f ? x[i] : x[i] * alpha;
      break;
  }
}

// PReLU over rows [r0, r1) of c channels: x >= 0 ? x : x * slope[ch].
template <class V>
void leaf_prelu_rows(const float* x, const float* slope, float* y, int64_t c, int64_t r0, int64_t r1) {
  constexpr int W = V::W;
  for (int64_t r = r0; r < r1; ++r) {
    const float* xp = x + r * c;
    float* yp = y + r * c;
    int64_t i = 0;
    for (; i + 2 * W <= c; i += 2 * W) {
      const typename V::T v0 = V::load(xp + i), v1 = V::load(xp + i + W);
      const typename V::T r0v = V::select_ge0(v0, V::mul(v0, V::load(slope + i)));
      const typename V::T r1v = V::select_ge0(v1, V::mul(v1, V::load(slope + i + W)));
      V::store(yp + i, r0v);
      V::store(yp + i + W, r1v);
    }
    for (; i + W <= c; i += W) {
      typename V::T v = V::load(xp + i);
      V::store(yp + i, V::select_ge0(v, V::mul(v, V::load(slope + i))));
    }
    for (; i < c; ++i) yp[i] = xp[i] >= 0.f ? xp[i] : xp[i] * slope[i];
  }
}

// y[i] = a[i] * b[i] for i in [i0, i1) (Mul).
template <class V>
void leaf_mul(const float* a, const float* b, float* y, int64_t i0, int64_t i1) {
  constexpr int W = V::W;
  int64_t i = i0;
  for (; i + W <= i1; i += W) V::store(y + i, V::mul(V::load(a + i), V::load(b + i)));
  for (; i < i1; ++i) y[i] = a[i] * b[i];
}

// Rows [r0, r1) of c channels: y = x * g[ch] (ChannelGate's product, no
// additive term, so -0 stays -0 unlike leaf_affine_rows with a null bias).
template <class V>
void leaf_mul_rows(const float* x, const float* g, float* y, int64_t c, int64_t r0, int64_t r1) {
  constexpr int W = V::W;
  for (int64_t r = r0; r < r1; ++r) {
    const float* xp = x + r * c;
    float* yp = y + r * c;
    int64_t i = 0;
    for (; i + W <= c; i += W) V::store(yp + i, V::mul(V::load(xp + i), V::load(g + i)));
    for (; i < c; ++i) yp[i] = xp[i] * g[i];
  }
}

// Activation sign (zero = 0): x >= 0 ? 1 : -1 (NaN -> -1);
// sign_zero (zero = 1): x > 0 ? 1 : (x < 0 ? -1 : 0) (NaN -> 0, never -0).
template <class V>
void leaf_sign(const float* x, float* y, int64_t i0, int64_t i1, int32_t zero) {
  constexpr int W = V::W;
  int64_t i = i0;
  if (zero) {
    for (; i + W <= i1; i += W) V::store(y + i, V::sign0(V::load(x + i)));
    for (; i < i1; ++i) y[i] = x[i] > 0.f ? 1.f : (x[i] < 0.f ? -1.f : 0.f);
  } else {
    for (; i + W <= i1; i += W) V::store(y + i, V::sign(V::load(x + i)));
    for (; i < i1; ++i) y[i] = x[i] >= 0.f ? 1.f : -1.f;
  }
}

// ------------------------------------------------- compiled fused programs
// leaf_fe_rows evaluates a compiled FusedElementwise program (FeEvalArgs,
// float_opt_args.h) on rows [r0, r1) of y. Each row is cut into chunks of NV
// vectors (then single vectors, then single floats); the whole program runs
// on one chunk with the stack top and the binary operand in registers. The
// per-op float operations are those of the one-pass-per-op leaves above
// (leaf_add, leaf_affine_rows_t, leaf_activation, leaf_prelu_rows, leaf_mul,
// leaf_mul_rows, leaf_sign), vector and scalar alike, so the result is
// bit-identical to them whatever the chunking.

// One-float policy for the row tails: the scalar semantics of the leaves'
// tail loops.
struct FeScalar {
  using T = float;
  static constexpr int W = 1;
  static T load(const float* p) { return *p; }
  static void store(float* p, T v) { *p = v; }
  static T dup(float s) { return s; }
  static T add(T a, T b) { return a + b; }
  static T mul(T a, T b) { return a * b; }
  static T relu(T v) { return v > 0.f ? v : 0.f; }
  static T clamp(T v, T lo, T hi) { return leaf_min(leaf_max(v, lo), hi); }
  static T select_ge0(T v, T neg) { return v >= 0.f ? v : neg; }
  static T sign(T v) { return v >= 0.f ? 1.f : -1.f; }
  static T sign0(T v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); }
};

#define BF_FE_UNROLL _Pragma("GCC unroll 8")

// GCC cannot see that every program starts with a push (top / opnd are always
// written before being read) and would warn on each inlined use.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

template <class F, int... I>
inline __attribute__((always_inline)) void each_v(F f, std::integer_sequence<int, I...>) {
  (f(I), ...);
}

template <class V, int NV>
struct FeChunk {
  using T = typename V::T;
  static constexpr int W = V::W;

  static inline __attribute__((always_inline)) void load(const float* s, T* d) {
    each_v([&](int v) __attribute__((always_inline)) { d[v] = V::load(s + v * W); }, std::make_integer_sequence<int, NV>{});
  }
  // NV vectors of a source at channel ch of the current row, addressing A
  // (row pointer rp, gather ids row rid); row sources are read at rp + so.
  template <int A>
  static inline __attribute__((always_inline)) void src(const FeMicro& m, const float* rp, const int32_t* rid, int64_t ch, int64_t so,
                                                        T* d) {
    if (A == kFeAtRow) {
      load(rp + so, d);
    } else if (A == kFeAtTile) {
      load(rp + ch % m.w, d);
    } else {
      const int64_t q = ch / m.w;
      const int32_t id = rid[q];
      if (id < 0 || id >= m.v) {
        each_v([&](int v) __attribute__((always_inline)) { d[v] = V::dup(0.f); }, std::make_integer_sequence<int, NV>{});
      } else {
        load(m.p + id * m.w + (ch - q * m.w), d);
      }
    }
  }

  // ch: channel of the chunk (per-channel parameters, tile / gather loads);
  // so: offset of the chunk from the row pointers rp and yr (row sources, output).
  static inline __attribute__((always_inline)) void run(const FeEvalArgs& e, const float* const* rp, const int32_t* const* rid,
                                                         int64_t ch, int64_t so, float* yr) {
    T top[NV], opnd[NV], spill[16][NV];
    int sp = 0;
    const FeMicro* ops = e.ops;
    const int32_t n = e.n_ops;
    for (int32_t k = 0; k < n; ++k) {
      const FeMicro& m = ops[k];
      switch (m.op) {
#define BF_FE_SRC_CASES(BASE, A)                                                 \
  case BASE + A: src<A>(m, rp[k], rid[k], ch, so, top); break;                        \
  case kFeOpnd + A: src<A>(m, rp[k], rid[k], ch, so, opnd); break;                    \
  case kFeAddSrc + A:                                                            \
    src<A>(m, rp[k], rid[k], ch, so, opnd);                                          \
    BF_FE_UNROLL for (int v = 0; v < NV; ++v) top[v] = V::add(top[v], opnd[v]);  \
    break;                                                                       \
  case kFeMulSrc + A:                                                            \
    src<A>(m, rp[k], rid[k], ch, so, opnd);                                          \
    BF_FE_UNROLL for (int v = 0; v < NV; ++v) top[v] = V::mul(top[v], opnd[v]);  \
    break;                                                                       \
  case kFeRAddSrc + A:                                                           \
    src<A>(m, rp[k], rid[k], ch, so, opnd);                                          \
    BF_FE_UNROLL for (int v = 0; v < NV; ++v) top[v] = V::add(opnd[v], top[v]);  \
    break;                                                                       \
  case kFeRMulSrc + A:                                                           \
    src<A>(m, rp[k], rid[k], ch, so, opnd);                                          \
    BF_FE_UNROLL for (int v = 0; v < NV; ++v) top[v] = V::mul(opnd[v], top[v]);  \
    break;
        BF_FE_SRC_CASES(kFePush, kFeAtRow)
        BF_FE_SRC_CASES(kFePush, kFeAtTile)
        BF_FE_SRC_CASES(kFePush, kFeAtGather)
#undef BF_FE_SRC_CASES
        case kFeSpill:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) spill[sp][v] = top[v];
          ++sp;
          break;
        case kFePop:
          --sp;
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) {
            opnd[v] = top[v];
            top[v] = spill[sp][v];
          }
          break;
        case kFeAdd:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = leaf_activate<V>(m.act, V::add(top[v], opnd[v]));
          break;
        case kFeRAdd:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = leaf_activate<V>(m.act, V::add(opnd[v], top[v]));
          break;
        case kFeMul:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::mul(top[v], opnd[v]);
          break;
        case kFeAffineSB:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::add(V::mul(top[v], V::load(m.p + ch + v * W)), V::load(m.q + ch + v * W));
          break;
        case kFeAffineS:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::add(V::mul(top[v], V::load(m.p + ch + v * W)), V::dup(0.f));
          break;
        case kFeAffineB:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::add(top[v], V::load(m.q + ch + v * W));
          break;
        case kFeAffineN:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::add(top[v], V::dup(0.f));
          break;
        case kFeRelu:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::relu(top[v]);
          break;
        case kFeClamp:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::clamp(top[v], V::dup(m.lo), V::dup(m.hi));
          break;
        case kFeLeaky:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::select_ge0(top[v], V::mul(top[v], V::dup(m.alpha)));
          break;
        case kFePrelu:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::select_ge0(top[v], V::mul(top[v], V::load(m.p + ch + v * W)));
          break;
        case kFeSign:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::sign(top[v]);
          break;
        case kFeSign0:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::sign0(top[v]);
          break;
        case kFeGate:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) top[v] = V::mul(top[v], V::load(rp[k] + ch + v * W));
          break;
        case kFeRPrelu:
          BF_FE_UNROLL
          for (int v = 0; v < NV; ++v) {
            const T t = V::add(top[v], V::load(m.q + ch + v * W));
            top[v] = V::add(V::select_ge0(t, V::mul(t, V::load(m.p + ch + v * W))), V::load(m.r + ch + v * W));
          }
          break;
        default:
          __builtin_unreachable();  // opcodes are validated when the program is compiled
      }
    }
    BF_FE_UNROLL
    for (int v = 0; v < NV; ++v) V::store(yr + so + v * W, top[v]);
  }

  // `count` consecutive chunks: chunk i at offset so + i * NV * W from the row
  // pointers, channel ch advancing alike and wrapping to 0 at c (the flat run
  // over several rows). The program ends with a kFeNumOps sentinel op
  // (fused_elementwise.cc), so each micro-op jumps straight to the next one's
  // handler (direct threading: 3-4 instructions per micro-op instead of a
  // loop, a bounds check and a jump table). Each handler performs exactly the
  // float operations of run()'s case for that opcode.
  static void run_chunks(const FeEvalArgs& e, const float* const* rp, const int32_t* const* rid, int64_t ch, int64_t so, int64_t count,
                         int64_t c, float* yr) {
    static void* const kTbl[kFeNumOps + 1] = {
        &&push_row,  &&push_tile,  &&push_gather,  &&opnd_row,  &&opnd_tile,  &&opnd_gather,  &&adds_row,  &&adds_tile,
        &&adds_gather, &&muls_row, &&muls_tile, &&muls_gather, &&radds_row, &&radds_tile, &&radds_gather, &&rmuls_row,
        &&rmuls_tile, &&rmuls_gather, &&spill_op, &&pop_op, &&add_op, &&radd_op, &&mul_op, &&affine_sb, &&affine_s,
        &&affine_b, &&affine_n, &&relu_op, &&clamp_op, &&leaky_op, &&prelu_op, &&sign_op, &&sign0_op, &&gate_op,
        &&rprelu_op, &&end_op};
    static_assert(kFePush == 0 && kFeOpnd == 3 && kFeAddSrc == 6 && kFeMulSrc == 9 && kFeRAddSrc == 12 && kFeRMulSrc == 15 &&
                      kFeSpill == 18 && kFePop == 19 && kFeAdd == 20 && kFeRAdd == 21 && kFeMul == 22 && kFeAffineSB == 23 &&
                      kFeAffineS == 24 && kFeAffineB == 25 && kFeAffineN == 26 && kFeRelu == 27 && kFeClamp == 28 && kFeLeaky == 29 &&
                      kFePrelu == 30 && kFeSign == 31 && kFeSign0 == 32 && kFeGate == 33 && kFeRPrelu == 34 && kFeNumOps == 35,
                  "kTbl follows the opcode order");
    constexpr int64_t CW = NV * W;
    // Straight-line per-vector statements (no loops: loops in blocks entered
    // by an indirect branch are not unrolled, and top / opnd would leave the
    // registers).
#define BF_FE_V(...) each_v([&](int v) __attribute__((always_inline)) { __VA_ARGS__; }, std::make_integer_sequence<int, NV>{})
    T top[NV], opnd[NV], spill[16][NV];
    int sp = 0;
    const FeMicro* const ops = e.ops;
    const FeMicro* m = ops;
    int64_t k = 0;  // index of *m (row pointers rp[k], gather ids rid[k])
#define BF_FE_NEXT \
  ++k;             \
  goto* kTbl[(++m)->op]
#define BF_FE_K k
#define BF_FE_SRC_HANDLERS(NAME, A)                                                                       \
  push_##NAME : src<A>(*m, rp[BF_FE_K], rid[BF_FE_K], ch, so, top);                                       \
  BF_FE_NEXT;                                                                                             \
  opnd_##NAME : src<A>(*m, rp[BF_FE_K], rid[BF_FE_K], ch, so, opnd);                                      \
  BF_FE_NEXT;                                                                                             \
  adds_##NAME : src<A>(*m, rp[BF_FE_K], rid[BF_FE_K], ch, so, opnd);                                      \
  BF_FE_V(top[v] = V::add(top[v], opnd[v]));                             \
  BF_FE_NEXT;                                                                                             \
  muls_##NAME : src<A>(*m, rp[BF_FE_K], rid[BF_FE_K], ch, so, opnd);                                      \
  BF_FE_V(top[v] = V::mul(top[v], opnd[v]));                             \
  BF_FE_NEXT;                                                                                             \
  radds_##NAME : src<A>(*m, rp[BF_FE_K], rid[BF_FE_K], ch, so, opnd);                                     \
  BF_FE_V(top[v] = V::add(opnd[v], top[v]));                             \
  BF_FE_NEXT;                                                                                             \
  rmuls_##NAME : src<A>(*m, rp[BF_FE_K], rid[BF_FE_K], ch, so, opnd);                                     \
  BF_FE_V(top[v] = V::mul(opnd[v], top[v]));                             \
  BF_FE_NEXT;
    goto* kTbl[m->op];
    BF_FE_SRC_HANDLERS(row, kFeAtRow)
    BF_FE_SRC_HANDLERS(tile, kFeAtTile)
    BF_FE_SRC_HANDLERS(gather, kFeAtGather)
  spill_op:
        BF_FE_V(spill[sp][v] = top[v]);
    ++sp;
    BF_FE_NEXT;
  pop_op:
    --sp;
    BF_FE_V(opnd[v] = top[v]; top[v] = spill[sp][v]);
    BF_FE_NEXT;
  add_op:
        BF_FE_V(top[v] = leaf_activate<V>(m->act, V::add(top[v], opnd[v])));
    BF_FE_NEXT;
  radd_op:
        BF_FE_V(top[v] = leaf_activate<V>(m->act, V::add(opnd[v], top[v])));
    BF_FE_NEXT;
  mul_op:
        BF_FE_V(top[v] = V::mul(top[v], opnd[v]));
    BF_FE_NEXT;
  affine_sb:
        BF_FE_V(top[v] = V::add(V::mul(top[v], V::load(m->p + ch + v * W)), V::load(m->q + ch + v * W)));
    BF_FE_NEXT;
  affine_s:
        BF_FE_V(top[v] = V::add(V::mul(top[v], V::load(m->p + ch + v * W)), V::dup(0.f)));
    BF_FE_NEXT;
  affine_b:
        BF_FE_V(top[v] = V::add(top[v], V::load(m->q + ch + v * W)));
    BF_FE_NEXT;
  affine_n:
        BF_FE_V(top[v] = V::add(top[v], V::dup(0.f)));
    BF_FE_NEXT;
  relu_op:
        BF_FE_V(top[v] = V::relu(top[v]));
    BF_FE_NEXT;
  clamp_op:
        BF_FE_V(top[v] = V::clamp(top[v], V::dup(m->lo), V::dup(m->hi)));
    BF_FE_NEXT;
  leaky_op:
        BF_FE_V(top[v] = V::select_ge0(top[v], V::mul(top[v], V::dup(m->alpha))));
    BF_FE_NEXT;
  prelu_op:
        BF_FE_V(top[v] = V::select_ge0(top[v], V::mul(top[v], V::load(m->p + ch + v * W))));
    BF_FE_NEXT;
  sign_op:
        BF_FE_V(top[v] = V::sign(top[v]));
    BF_FE_NEXT;
  sign0_op:
        BF_FE_V(top[v] = V::sign0(top[v]));
    BF_FE_NEXT;
  gate_op:
        BF_FE_V(top[v] = V::mul(top[v], V::load(rp[BF_FE_K] + ch + v * W)));
    BF_FE_NEXT;
  rprelu_op:
    BF_FE_V(const T t = V::add(top[v], V::load(m->q + ch + v * W));
            top[v] = V::add(V::select_ge0(t, V::mul(t, V::load(m->p + ch + v * W))), V::load(m->r + ch + v * W)));
    BF_FE_NEXT;
  end_op:
        BF_FE_V(V::store(yr + so + v * W, top[v]));
    if (--count == 0) return;
    so += CW;
    ch += CW;
    if (ch == c) ch = 0;
    sp = 0;
    m = ops;
    k = 0;
    goto* kTbl[m->op];
#undef BF_FE_SRC_HANDLERS
#undef BF_FE_V
#undef BF_FE_NEXT
#undef BF_FE_K
  }
};

// Rows [r0, r1) of y (in place for kFeSrcRow sources with p == nullptr).
// scratch: per-thread area laid out as documented at FeEvalArgs.
template <class V, int NV>
void leaf_fe_rows(const FeEvalArgs& e, float* y, int64_t r0, int64_t r1, void* scratch) {
  constexpr int W = V::W, CW = NV * W;
  const int64_t c = e.c;
  const float** rp = static_cast<const float**>(scratch);
  const int32_t** rid = reinterpret_cast<const int32_t**>(rp + e.n_ops);
  float* rowbuf = reinterpret_cast<float*>(rid + e.n_ops);
  if (e.flat && c % CW == 0) {
    // Only plain row sources: rows [r0, r1) are one run of chunks (no per-row setup).
    for (int32_t j = 0; j < e.n_row_ops; ++j) {
      const FeMicro& m = e.ops[e.row_ops[j]];
      rp[e.row_ops[j]] = (m.p ? m.p : y) + r0 * c;
    }
    float* yb = y + r0 * c;
    const int64_t n = (r1 - r0) * c;
    if (n > 0) FeChunk<V, NV>::run_chunks(e, rp, rid, 0, 0, n / CW, c, yb);
    return;
  }
  for (int64_t r = r0; r < r1; ++r) {
    for (int32_t j = 0; j < e.n_row_ops; ++j) {
      const int32_t k = e.row_ops[j];
      const FeMicro& m = e.ops[k];
      if (m.op == kFeGate) {
        rp[k] = m.p + (r / e.rows_per_batch) * c;
        continue;
      }
      switch (m.src) {
        case kFeSrcRow: rp[k] = (m.p ? m.p : y) + r * c; break;
        case kFeSrcTile: rp[k] = m.p + r * m.w; break;
        case kFeSrcGather: rid[k] = m.ids + r * (c / m.w); break;
        case kFeSrcTileRow: {
          float* d = rowbuf + m.slot * c;
          const float* x = m.p + r * m.w;
          for (int64_t o = 0; o < c; o += m.w) __builtin_memcpy(d + o, x, static_cast<size_t>(m.w) * sizeof(float));
          rp[k] = d;
          break;
        }
        case kFeSrcGatherRow: {
          float* d = rowbuf + m.slot * c;
          const int32_t* ids = m.ids + r * (c / m.w);
          for (int64_t q = 0; q * m.w < c; ++q) {
            const int64_t id = ids[q];
            if (id < 0 || id >= m.v) __builtin_memset(d + q * m.w, 0, static_cast<size_t>(m.w) * sizeof(float));
            else __builtin_memcpy(d + q * m.w, m.p + id * m.w, static_cast<size_t>(m.w) * sizeof(float));
          }
          rp[k] = d;
          break;
        }
        default: break;
      }
    }
    float* yr = y + r * c;
    int64_t ch = c / CW * CW;
    if (ch > 0) FeChunk<V, NV>::run_chunks(e, rp, rid, 0, 0, c / CW, c, yr);
    for (; ch + W <= c; ch += W) FeChunk<V, 1>::run(e, rp, rid, ch, ch, yr);
    for (; ch < c; ++ch) FeChunk<FeScalar, 1>::run(e, rp, rid, ch, ch, yr);
  }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// -------------------------------------------------------------------- pools
// MaxPool2D uses the reference's max (std::max semantics: a NaN input is
// ignored), so it is bit-identical. AvgPool2D sums in float (the reference
// sums in double) and divides by the tap count.
template <class V>
struct Pool {
  using T = typename V::T;
  static constexpr int W = V::W;
  const PoolArgs& g;

  static void tap_range(int64_t start, int64_t k, int64_t size, int64_t* lo, int64_t* hi) {
    *lo = leaf_max_i(0, -start);
    *hi = leaf_min_i(k, size - start);
    if (*hi < *lo) *hi = *lo;
  }

  void max_pixel(const float* xn, int64_t off, int64_t ky0, int64_t ky1, int64_t kx0, int64_t kx1, float* yp) const {
    const int64_t c = g.c, row = g.iw * c;
    const float ninf = -__builtin_inff();
    int64_t ch = 0;
    for (; ch + 4 * W <= c; ch += 4 * W) {
      T m0 = V::dup(ninf), m1 = m0, m2 = m0, m3 = m0;
      for (int64_t ky = ky0; ky < ky1; ++ky)
        for (int64_t kx = kx0; kx < kx1; ++kx) {
          const float* p = xn + (off + ky * row + kx * c + ch);
          m0 = V::max_num(m0, V::load(p));
          m1 = V::max_num(m1, V::load(p + W));
          m2 = V::max_num(m2, V::load(p + 2 * W));
          m3 = V::max_num(m3, V::load(p + 3 * W));
        }
      V::store(yp + ch, m0);
      V::store(yp + ch + W, m1);
      V::store(yp + ch + 2 * W, m2);
      V::store(yp + ch + 3 * W, m3);
    }
    for (; ch + W <= c; ch += W) {
      T m = V::dup(ninf);
      for (int64_t ky = ky0; ky < ky1; ++ky)
        for (int64_t kx = kx0; kx < kx1; ++kx) m = V::max_num(m, V::load(xn + (off + ky * row + kx * c + ch)));
      V::store(yp + ch, m);
    }
    for (; ch < c; ++ch) {
      float m = ninf;
      for (int64_t ky = ky0; ky < ky1; ++ky)
        for (int64_t kx = kx0; kx < kx1; ++kx) m = leaf_max(m, xn[off + ky * row + kx * c + ch]);
      yp[ch] = m;
    }
  }

  void avg_pixel(const float* xn, int64_t off, int64_t ky0, int64_t ky1, int64_t kx0, int64_t kx1, float cnt, float* yp) const {
    const int64_t c = g.c, row = g.iw * c;
    const T cv = V::dup(cnt);
    int64_t ch = 0;
    for (; ch + 4 * W <= c; ch += 4 * W) {
      T s0 = V::dup(0.f), s1 = s0, s2 = s0, s3 = s0;
      for (int64_t ky = ky0; ky < ky1; ++ky)
        for (int64_t kx = kx0; kx < kx1; ++kx) {
          const float* p = xn + (off + ky * row + kx * c + ch);
          s0 = V::add(s0, V::load(p));
          s1 = V::add(s1, V::load(p + W));
          s2 = V::add(s2, V::load(p + 2 * W));
          s3 = V::add(s3, V::load(p + 3 * W));
        }
      V::store(yp + ch, V::div(s0, cv));
      V::store(yp + ch + W, V::div(s1, cv));
      V::store(yp + ch + 2 * W, V::div(s2, cv));
      V::store(yp + ch + 3 * W, V::div(s3, cv));
    }
    for (; ch + W <= c; ch += W) {
      T s = V::dup(0.f);
      for (int64_t ky = ky0; ky < ky1; ++ky)
        for (int64_t kx = kx0; kx < kx1; ++kx) s = V::add(s, V::load(xn + (off + ky * row + kx * c + ch)));
      V::store(yp + ch, V::div(s, cv));
    }
    for (; ch < c; ++ch) {
      float s = 0.f;
      for (int64_t ky = ky0; ky < ky1; ++ky)
        for (int64_t kx = kx0; kx < kx1; ++kx) s += xn[off + ky * row + kx * c + ch];
      yp[ch] = s / cnt;
    }
  }

  void rows(int64_t r0, int64_t r1) const {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t n = r / g.oh, oy = r % g.oh, iy0 = oy * g.sh - g.pt;
      int64_t ky0, ky1;
      tap_range(iy0, g.kh, g.ih, &ky0, &ky1);
      for (int64_t ox = 0; ox < g.ow; ++ox) {
        const int64_t ix0 = ox * g.sw - g.pl;
        int64_t kx0, kx1;
        tap_range(ix0, g.kw, g.iw, &kx0, &kx1);
        // Offset of the window origin from image n; may be negative, so it
        // is only ever added to in-bounds tap offsets.
        const float* xn = g.x + n * g.ih * g.iw * g.c;
        const int64_t off = (iy0 * g.iw + ix0) * g.c;
        float* yp = g.y + (r * g.ow + ox) * g.c;
        if (g.is_max) {
          max_pixel(xn, off, ky0, ky1, kx0, kx1, yp);
        } else {
          const int64_t cnt = g.include_pad ? g.kh * g.kw : (ky1 - ky0) * (kx1 - kx0);
          avg_pixel(xn, off, ky0, ky1, kx0, kx1, static_cast<float>(leaf_max_i(cnt, 1)), yp);
        }
      }
    }
  }
};

template <class V>
void leaf_pool_rows(const PoolArgs& g, int64_t r0, int64_t r1) {
  Pool<V>{g}.rows(r0, r1);
}

// GlobalAvgPool work items t in [t0, t1): (batch b, block of 4W channels);
// each sums its channels over all hw pixels (4 vector accumulators) and
// divides by hw.
template <class V>
void leaf_global_avg_pool(const float* x, float* y, int64_t hw, int64_t c, int64_t t0, int64_t t1) {
  using T = typename V::T;
  constexpr int W = V::W, B = 4 * W;
  const int64_t cblocks = (c + B - 1) / B;
  const float count = static_cast<float>(hw);
  const T hv = V::dup(count);
  for (int64_t t = t0; t < t1; ++t) {
    const int64_t b = t / cblocks, c0 = (t % cblocks) * B, c1 = leaf_min_i(c, c0 + B);
    const float* xb = x + b * hw * c;
    float* yb = y + b * c;
    int64_t ch = c0;
    if (c1 - c0 == B) {
      T s0 = V::dup(0.f), s1 = s0, s2 = s0, s3 = s0;
      for (int64_t p = 0; p < hw; ++p) {
        const float* q = xb + p * c + ch;
        s0 = V::add(s0, V::load(q));
        s1 = V::add(s1, V::load(q + W));
        s2 = V::add(s2, V::load(q + 2 * W));
        s3 = V::add(s3, V::load(q + 3 * W));
      }
      V::store(yb + ch, V::div(s0, hv));
      V::store(yb + ch + W, V::div(s1, hv));
      V::store(yb + ch + 2 * W, V::div(s2, hv));
      V::store(yb + ch + 3 * W, V::div(s3, hv));
      continue;
    }
    for (; ch + W <= c1; ch += W) {
      T s = V::dup(0.f);
      for (int64_t p = 0; p < hw; ++p) s = V::add(s, V::load(xb + p * c + ch));
      V::store(yb + ch, V::div(s, hv));
    }
    for (; ch < c1; ++ch) {
      float s = 0.f;
      for (int64_t p = 0; p < hw; ++p) s += xb[p * c + ch];
      yb[ch] = s / count;
    }
  }
}

}  // namespace
}  // namespace fopt
}  // namespace bitino
