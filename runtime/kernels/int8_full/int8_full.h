// Shared pieces of the full-integer (int8_full) kernels: parameter parsing and
// the scalar definitions every implementation must reproduce bit for bit.
// Contracts: docs/int8.md ("Full-integer execution: int8_full"); oracle
// python/bitino/int8_ops.py.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "bitino/op.h"
#include "kernels/int8_full/q8_math.h"
#include "ops/transformer_common.h"

namespace bitino {
namespace int8_full {

// I8Quantize of one value: clamp(rne(fp32(x / s)) + z, -128, 127), IEEE fp32 division.
inline int8_t quantize1(float x, float s, float z) {
  const float v = std::nearbyint(x / s) + z;
  return static_cast<int8_t>(std::min(std::max(v, -128.f), 127.f));
}

// I8LayerNorm per-row factor r = fp32(xs / sqrt(xs*xs*vn + (eps*D)*D)), vn = D*S2 - S1^2,
// evaluated in IEEE double one operation at a time.
inline float layernorm_row_factor(int64_t d, int64_t s1, int64_t s2, double xs, double eps) {
  const int64_t vn = d * s2 - s1 * s1;
  const double c2 = (eps * static_cast<double>(d)) * static_cast<double>(d);
  const double a = xs * xs;
  const double b = a * static_cast<double>(vn);
  return static_cast<float>(xs / std::sqrt(b + c2));
}

// Epilogue of I8Conv2D / I8Dense: i8 -> clamp(y_zero + requant(sat32(acc + bias), mult[o], shift[o]));
// f32 -> fp32(fp32(float(acc) * scale[o]) + bias_f[o]).
// l8 -> l8_encode(fp32(fp32(float(acc) * scale[o]) + bias_f[o])) (scale / bias in units of 2^y_exp).
struct GemmEpilogue {
  bool f32 = false, l8 = false;
  const int32_t* bias_q = nullptr;
  const int32_t* mult = nullptr;
  const int32_t* shift = nullptr;
  const float* bias_f = nullptr;
  const float* scale = nullptr;
  int64_t zy = 0;
  int lo = -128, hi = 127;
  explicit GemmEpilogue(const KernelContext& ctx) {
    f32 = ctx.attrs().get_string("output", "i8") == "f32";
    l8 = ctx.attrs().get_string("output", "i8") == "l8";
    if (f32 || l8) {
      bias_f = ctx.has_in(2) ? ctx.in(2).data<float>() : nullptr;
      scale = ctx.in(3).data<float>();
    } else {
      bias_q = ctx.has_in(2) ? ctx.in(2).data<int32_t>() : nullptr;
      mult = ctx.in(3).data<int32_t>();
      shift = ctx.in(4).data<int32_t>();
      zy = ctx.attrs().get_int("y_zero", 0);
      lo = static_cast<int>(ctx.attrs().get_int("qmin", -128));
      hi = static_cast<int>(ctx.attrs().get_int("qmax", 127));
    }
  }
  // Output j of I8Conv2DN (internal twin node, src/graph_rewrite_i8.h): bias in(2), mult in(3 + 2j), shift in(4 + 2j),
  // y_zeros[j] / qmins[j] / qmaxs[j]; int8 output.
  GemmEpilogue(const KernelContext& ctx, int j) {
    bias_q = ctx.has_in(2) ? ctx.in(2).data<int32_t>() : nullptr;
    mult = ctx.in(3 + 2 * j).data<int32_t>();
    shift = ctx.in(4 + 2 * j).data<int32_t>();
    zy = ctx.attrs().get_ints("y_zeros", {})[static_cast<size_t>(j)];
    lo = static_cast<int>(ctx.attrs().get_ints("qmins", {})[static_cast<size_t>(j)]);
    hi = static_cast<int>(ctx.attrs().get_ints("qmaxs", {})[static_cast<size_t>(j)]);
  }
  int8_t q(int64_t acc, int64_t o) const {
    const int64_t a = acc + (bias_q ? bias_q[o] : 0);
    return q8::clamp8(zy + q8::requant(q8::sat32(a), mult[o], shift[o]), lo, hi);
  }
  float f(int64_t acc, int64_t o) const {
    const float v = static_cast<float>(acc) * scale[o];
    return bias_f ? v + bias_f[o] : v;
  }
  void store(const KernelContext& ctx, int64_t at, int64_t o, int64_t acc) const {
    if (f32) ctx.out(0).data<float>()[at] = f(acc, o);
    else if (l8) ctx.out(0).data<int8_t>()[at] = q8::l8_encode(f(acc, o));
    else ctx.out(0).data<int8_t>()[at] = q(acc, o);
  }
};

// Dequantization of operand p (attributes {p}_code: "linear" (default; {p}_scale, {p}_zero) or "l8" ({p}_exp)).
inline q8::Deq deq_of(const Json& a, const std::string& p) {
  q8::Deq d;
  d.l8 = a.get_string(p + "_code", "linear") == "l8";
  d.s = static_cast<float>(a.get_number(p + "_scale", 1.0));
  d.z = static_cast<int32_t>(a.get_int(p + "_zero", 0));
  d.e = static_cast<int>(a.get_int(p + "_exp", 0));
  return d;
}

// Epilogue of I8BConv2DRes (int8_ops._bconv_res): from the binary layer's fp32 value v = fp32(acc)*post_mul + post_add,
//   u = v + R(res);  w = u >= 0 ? u : u * slope[o];  w += out_bias[o] (optional);  y = l8_encode(w)
// R: r_code "l8" -> l8_decode(res, r_exp); "linear" -> fp32(res - r_zero) * r_scale.
// Bit output (bit_thr given): bit = 1 (-1) iff !(s >= bit_thr[o]) (ge) / !(s > bit_thr[o]) (gt), s = u (bit_src "sum",
// default) or s = w (bit_src "out"). Optional pre stage on v before the add (inputs 10-13, pre() below).
struct ResEpilogue {
  bool r_l8 = true;
  bool bit_out = false;  // bit_src "out": the bits binarize w instead of u
  int r_exp = 0;
  float r_scale = 1.f;
  int32_t r_zero = 0;
  const float* slope = nullptr;
  const float* bias = nullptr;
  const int8_t* out_lut = nullptr;  // in(9): output 0 = out_lut[code + 128] (an I8Lut absorbed by Session)
  // output "i8" (int8_full v2): the linear code clamp(rne(w) + y_zero, qmin, qmax) instead of l8_encode(w)
  bool y_lin = false;
  int32_t y_zero = 0, qmin = -128, qmax = 127;
  float dec[256];  // R of each residual code (index res + 128)
  explicit ResEpilogue(const KernelContext& ctx) {
    const Json& a = ctx.attrs();
    bit_out = a.get_string("bit_src", "sum") == "out";
    r_l8 = a.get_string("r_code", "l8") == "l8";
    r_exp = static_cast<int>(a.get_int("r_exp", 0));
    r_scale = static_cast<float>(a.get_number("r_scale", 1.0));
    r_zero = static_cast<int32_t>(a.get_int("r_zero", 0));
    slope = ctx.in(6).data<float>();
    bias = ctx.has_in(7) ? ctx.in(7).data<float>() : nullptr;
    out_lut = ctx.has_in(9) ? ctx.in(9).data<int8_t>() : nullptr;
    y_lin = a.get_string("output", "l8") == "i8";
    y_zero = static_cast<int32_t>(a.get_int("y_zero", 0));
    qmin = static_cast<int32_t>(a.get_int("qmin", -128));
    qmax = static_cast<int32_t>(a.get_int("qmax", 127));
    for (int q = -128; q <= 127; ++q)
      dec[q + 128] = r_l8 ? q8::l8_decode(q, r_exp) : static_cast<float>(static_cast<int64_t>(q) - r_zero) * r_scale;
    pre_slope = ctx.has_in(10) ? ctx.in(10).data<float>() : nullptr;
    pre_mul = ctx.has_in(11) ? ctx.in(11).data<float>() : nullptr;
    pre_add = ctx.has_in(12) ? ctx.in(12).data<float>() : nullptr;
    gate = ctx.has_in(13);
    gate_add = ctx.has_in(14) ? ctx.in(14).data<float>() : nullptr;
    g_scale = static_cast<float>(a.get_number("g_scale", 1.0));
    g_zero = static_cast<int32_t>(a.get_int("g_zero", 0));
  }
  // Pre stage (2026-11-05; int8_ops.bconv_res_pre), between v and the residual add, each step only if its input is given:
  //   v = v >= 0 ? v : v * pre_slope[o];  v = v * pre_mul[o];  v = v + pre_add[o];  v = v * G[n, o]
  // G = fp32(gate - g_zero) * g_scale of the int8 tensor in(13) [N, 1, 1, O] (per-sample squeeze-excitation multiplier).
  const float* pre_slope = nullptr;
  const float* pre_mul = nullptr;
  const float* pre_add = nullptr;
  bool gate = false;
  const float* gate_add = nullptr;  // in(14): v = v + gate_add[o] after the gate (2026-11-14)
  float g_scale = 1.f;
  int32_t g_zero = 0;
  std::vector<float> gval;  // G of the current run, [N * O] (bind_gate)
  bool pre_on() const { return pre_slope || pre_mul || pre_add || gate || gate_add; }
  void bind_gate(const KernelContext& ctx) {
    if (!gate) return;
    const Tensor& t = ctx.in(13);
    const int8_t* q = t.data<int8_t>();
    gval.resize(static_cast<size_t>(t.size()));
    for (int64_t i = 0; i < t.size(); ++i) gval[i] = static_cast<float>(static_cast<int64_t>(q[i]) - g_zero) * g_scale;
  }
  float pre(float v, int64_t n, int64_t o, int64_t oc) const {
    if (pre_slope) v = v >= 0.f ? v : v * pre_slope[o];
    if (pre_mul) v = v * pre_mul[o];
    if (pre_add) v = v + pre_add[o];
    if (gate) v = v * gval[n * oc + o];
    if (gate_add) v = v + gate_add[o];
    return v;
  }
  int8_t finish(float v, int64_t o, int8_t res) const {
    float s;
    return finish(v, o, res, &s);
  }
  // also returns the value the bit output binarizes (u or w)
  int8_t finish(float v, int64_t o, int8_t res, float* bit_value) const {
    const float u = v + dec[res + 128];
    float w = u >= 0.f ? u : u * slope[o];
    if (bias) w = w + bias[o];
    *bit_value = bit_out ? w : u;
    const int8_t c = y_lin ? lin_encode(w) : q8::l8_encode(w);
    return out_lut ? out_lut[c + 128] : c;
  }
  // int8_ops.lin_encode: rne(w) (saturating conversion; |w| beyond 2^20 gives the same code) + y_zero, clamped
  int8_t lin_encode(float w) const {
    const float r = std::nearbyint(std::fmin(std::fmax(w, -1048576.f), 1048576.f));
    const int64_t y = static_cast<int64_t>(r) + y_zero;
    return static_cast<int8_t>(y < qmin ? qmin : (y > qmax ? qmax : y));
  }
};

// I8BConv2DResI (int8_ops._bconv_res_i; int8_full v2): the integer epilogue of one output value from the exact binary
// dot product and the residual code. Every value is an int32 by the quantizer's bound (checked in the constructor).
struct ResIEpilogue {
  const int32_t *m1 = nullptr, *b = nullptr, *m2 = nullptr, *bthr = nullptr, *slope = nullptr, *ob = nullptr;
  const int32_t *pb = nullptr, *plo = nullptr, *phi = nullptr;  // pre stage: clamp(dot * m1 + pb, plo, phi)
  const int32_t* pslope = nullptr;  // in(15): PReLU pre stage (sigma = +1 for (0, MAX), -1 for (MIN, 0)), shift pk
  int pk = 0;
  int shift = 0, k = 0;
  int32_t y_zero = 0, qmin = -128, qmax = 127;
  bool bit_out = false;  // bit_src "out": the bits compare w instead of acc
  int64_t r_tile = 1;    // the residual has O / r_tile channels (channel o reads o mod (O / r_tile))
  ResIEpilogue(const KernelContext& ctx, int64_t K) {
    const Json& a = ctx.attrs();
    m1 = ctx.in(6).data<int32_t>();
    b = ctx.in(7).data<int32_t>();
    m2 = ctx.has_in(5) ? ctx.in(8).data<int32_t>() : nullptr;  // no residual: q * m2 = 0
    bthr = ctx.has_in(9) ? ctx.in(9).data<int32_t>() : nullptr;
    slope = ctx.has_in(10) ? ctx.in(10).data<int32_t>() : nullptr;
    ob = ctx.has_in(11) ? ctx.in(11).data<int32_t>() : nullptr;
    if (ctx.has_in(12)) {
      pb = ctx.in(12).data<int32_t>();
      plo = ctx.in(13).data<int32_t>();
      phi = ctx.in(14).data<int32_t>();
      if (ctx.has_in(15)) {
        pslope = ctx.in(15).data<int32_t>();
        const int64_t oc = ctx.in(6).size();
        for (int64_t c = 0; c < oc; ++c)
          BF_CHECK((plo[c] == 0 && phi[c] == INT32_MAX) || (plo[c] == INT32_MIN && phi[c] == 0),
                   "I8BConv2DResI: a PReLU pre stage needs (pre_lo, pre_hi) = (0, INT32_MAX) or (INT32_MIN, 0)");
      }
    }
    pk = static_cast<int>(a.get_int("pre_slope_shift", 0));
    shift = static_cast<int>(a.get_int("shift", 0));
    k = static_cast<int>(a.get_int("slope_shift", 0));
    y_zero = static_cast<int32_t>(a.get_int("y_zero", 0));
    qmin = static_cast<int32_t>(a.get_int("qmin", -128));
    qmax = static_cast<int32_t>(a.get_int("qmax", 127));
    bit_out = a.get_string("bit_src", "sum") == "out";
    r_tile = a.get_int("r_tile", 1);
    BF_CHECK(within_bound(ctx, K), "I8BConv2DResI: constants exceed the int32 / int16 bounds");
  }
  // int8_ops.bconv_res_i_bound: |acc|, |w| < 2^31 (int32 lanes compute the int64 values) and |m1|, |m2| <= 32767 (int16
  // multipliers of the vector epilogue)
  static bool within_bound(const KernelContext& ctx, int64_t K) {
    const int64_t oc = ctx.in(6).size();
    auto mx = [&](size_t i) {
      int64_t v = 0;
      if (ctx.has_in(i) && (i != 8 || ctx.has_in(5)))
        for (int64_t c = 0; c < oc; ++c) v = std::max<int64_t>(v, std::llabs(static_cast<long long>(ctx.in(i).data<int32_t>()[c])));
      return v;
    };
    const int k = static_cast<int>(ctx.attrs().get_int("slope_shift", 0));
    const int pk = static_cast<int>(ctx.attrs().get_int("pre_slope_shift", 0));
    const int64_t pre0 = K * mx(6) + mx(12);
    const int64_t pre = ctx.has_in(15) ? std::max(pre0, (((pre0 * mx(15)) >> 31) + 1) << pk) : pre0;
    if (pre >= (int64_t(1) << 31)) return false;
    const int64_t acc = pre + 128 * mx(8) + mx(7);
    int64_t w = acc;
    if (ctx.has_in(10)) w = std::max(acc, (((acc * mx(10)) >> 31) + 1) << k);
    w += mx(11);
    return std::max(acc, w) < (int64_t(1) << 31) && mx(6) <= 32767 && mx(8) <= 32767;
  }
  // y of one value; *bit = true iff the bit is -1 (!(s >= bit_thr))
  int8_t finish(int64_t dot, int64_t o, int8_t res, bool* bit) const {
    int64_t pre = dot * m1[o];
    if (pslope) {
      const int64_t sg = plo[o] == 0 ? 1 : -1;
      int64_t t = sg * (pre + pb[o]);
      if (t < 0) t = q8::mulhi(t, pslope[o]) * (int64_t(1) << pk);
      pre = sg * t;
    } else if (pb) {
      pre = std::min<int64_t>(std::max<int64_t>(pre + pb[o], plo[o]), phi[o]);
    }
    const int64_t acc = pre + (m2 ? static_cast<int64_t>(res) * m2[o] : 0) + b[o];
    int64_t w = acc;
    if (slope && acc < 0) w = q8::mulhi(acc, slope[o]) * (int64_t(1) << k);
    if (ob) w += ob[o];
    if (bthr) *bit = !((bit_out ? w : acc) >= bthr[o]);
    return q8::clamp8(q8::rshr(w, shift) + y_zero, qmin, qmax);
  }
};

// I8GatedAdd (int8_ops._gated_add; int8_full v2) constants and the bound check: 256 max|kx| + 65280 max|kp| + max|b| < 2^31
// (also covers the folded form q_x kx + q_p gk + b' with gk = (g - z_g) kp, b' = b - z_x kx - z_p gk the kernels use).
// g is [N, 1, 1, C] (squeeze-excitation gate) or [N, H, W, C] (full: star / GLU gating, 2026-11-13); optional in(7) g_lut
// i8 [1, 256]: the gate code is g_lut[g + 128] (a fused sigmoid I8Lut).
struct GatedAddParams {
  const int32_t *kx = nullptr, *kp = nullptr, *b = nullptr, *thr = nullptr;
  const int8_t* lut = nullptr;
  bool full = false;
  int64_t c = 0, nb = 0;
  int32_t zx = 0, zp = 0, zg = 0, zy = 0, qmin = -128, qmax = 127;
  int shift = 0;
  explicit GatedAddParams(const KernelContext& ctx) {
    const Json& a = ctx.attrs();
    c = ctx.in(0).dim(3);
    kx = ctx.in(3).data<int32_t>();
    kp = ctx.in(4).data<int32_t>();
    b = ctx.in(5).data<int32_t>();
    if (ctx.has_in(6)) {
      thr = ctx.in(6).data<int32_t>();
      nb = ctx.in(6).dim(0);
    }
    if (ctx.has_in(7)) lut = ctx.in(7).data<int8_t>();
    full = ctx.in(2).size() == ctx.in(0).size() && ctx.in(2).size() != ctx.in(0).dim(0) * c;
    zx = static_cast<int32_t>(a.get_int("x_zero", 0));
    zp = static_cast<int32_t>(a.get_int("p_zero", 0));
    zg = static_cast<int32_t>(a.get_int("g_zero", 0));
    zy = static_cast<int32_t>(a.get_int("y_zero", 0));
    qmin = static_cast<int32_t>(a.get_int("qmin", -128));
    qmax = static_cast<int32_t>(a.get_int("qmax", 127));
    shift = static_cast<int>(a.get_int("shift", 0));
    int64_t mx = 0, mp = 0, mb = 0;
    for (int64_t i = 0; i < c; ++i) {
      mx = std::max<int64_t>(mx, std::llabs(kx[i]));
      mp = std::max<int64_t>(mp, std::llabs(kp[i]));
      mb = std::max<int64_t>(mb, std::llabs(b[i]));
    }
    bound = 256 * mx + 65280 * mp + mb;
    BF_CHECK(bound < (int64_t(1) << 31), "I8GatedAdd: constants exceed the int32 bound");
  }
  int64_t bound = 0;
  int8_t gv(int8_t g) const { return lut ? lut[static_cast<int>(g) + 128] : g; }
  int64_t acc(int8_t x, int8_t p, int8_t g, int64_t ch) const {
    return (int64_t(x) - zx) * kx[ch] + (int64_t(p) - zp) * (int64_t(gv(g)) - zg) * kp[ch] + b[ch];
  }
  int8_t y(int64_t acc) const { return q8::clamp8(q8::rshr(acc, shift) + zy, qmin, qmax); }
};

// ---- No denormal can occur (AArch32 NEON flush-to-zero; 2026-11-13)
// AArch32 Advanced SIMD flushes denormal operands and results to +-0; on normal values (and zeros, infinities, NaNs)
// it rounds as IEEE fp32. A chain of fp32 operations therefore gives the IEEE result when every operand and every
// result is 0 or normal (|x| >= 2^-126). FtzBound tracks, per value, a lower bound L of its nonzero magnitudes and
// a power of two G it is a multiple of (x == 0 or |x| >= L; x in G Z), through:
//   * an integer:                          L = G = 1;
//   * x * c (c a constant, over channels): |x c| >= L min|c| and RN is monotone, so a nonzero result is at least
//     L' = 2^floor(log2(L min|c|)); every float of magnitude >= L' is a multiple of G' = L' 2^-23;
//   * x * y (two values):                  the same with L_x L_y;
//   * x + y / x + c:                       the exact sum is a multiple of G' = min(G_x, G_y) (G_c = ulp(c)), so a
//     nonzero sum is >= G'; RN keeps it a multiple of G' (exact below 2^24 G', a coarser ulp above): L' = G';
//   * PReLU (x >= 0 ? x : x * s):          the minimum of both branches.
// ok is false as soon as a bound L, or a constant operand, falls below 2^-126 (a value could then be denormal).
struct FtzBound {
  double L = 1.0, G = 1.0;
  bool ok = true;
  static double p2floor(double x) {
    int e = 0;
    std::frexp(x, &e);  // x = m 2^e, m in [0.5, 1)
    return std::ldexp(1.0, e - 1);
  }
  static constexpr double kMinNormal = 1.1754943508222875e-38;  // 2^-126
  static bool normal_or_zero(float c) { return c == 0.f || std::fabs(static_cast<double>(c)) >= kMinNormal || !std::isfinite(c); }
  // a nonzero product is >= L, a nonzero sum >= its G (= L): each must be normal
  void check_l() { ok = ok && L >= kMinNormal; }
  // x * c[i] for every i (c may hold zeros: those products are +-0)
  void mul_const(const float* c, int64_t n) {
    double m = INFINITY;
    for (int64_t i = 0; i < n; ++i) {
      ok = ok && normal_or_zero(c[i]);
      if (c[i] != 0.f && std::isfinite(c[i])) m = std::min(m, std::fabs(static_cast<double>(c[i])));
    }
    if (m == INFINITY) return;  // all zero (or non-finite): no new nonzero magnitudes
    L = p2floor(L * m);
    G = L * std::ldexp(1.0, -23);
    check_l();
  }
  void mul(const FtzBound& y) {
    ok = ok && y.ok;
    L = p2floor(L * y.L);
    G = L * std::ldexp(1.0, -23);
    check_l();
  }
  // x + c[i] for every i
  void add_const(const float* c, int64_t n) {
    double g = G;
    for (int64_t i = 0; i < n; ++i) {
      ok = ok && normal_or_zero(c[i]);
      if (c[i] != 0.f && std::isfinite(c[i])) g = std::min(g, p2floor(std::fabs(static_cast<double>(c[i]))) * std::ldexp(1.0, -23));
    }
    G = g;
    L = std::min(L, g);
    check_l();
  }
  void add(const FtzBound& y) {
    ok = ok && y.ok;
    G = std::min(G, y.G);
    L = std::min({L, y.L, G});
    check_l();
  }
  void prelu(const float* s, int64_t n) {
    FtzBound neg = *this;
    neg.mul_const(s, n);
    L = std::min(L, neg.L);
    G = std::min(G, neg.G);
    ok = ok && neg.ok;
  }
};

// True when every fp32 value of the I8BConv2DRes epilogue (int8_full::ResEpilogue and binary_value) is 0 or normal
// for every input, so a flush-to-zero vector implementation (AArch32 NEON) computes the IEEE values: v = fp32(acc) *
// post_mul + post_add (acc an integer), the pre stage, R(res), u, PReLU, bias; the bit thresholds (after the gt
// adjustment to the next float, as the NEON kernel compares) normal or zero.
inline bool res_epilogue_normal(const KernelContext& ctx) {
  const Json& a = ctx.attrs();
  auto cvec = [&](size_t i, const float** p, int64_t* n) {
    if (!ctx.has_in(i)) return false;
    *p = ctx.in(i).data<float>();
    *n = ctx.in(i).size();
    return true;
  };
  const float* c = nullptr;
  int64_t n = 0;
  FtzBound v;  // fp32(acc): an integer
  if (cvec(2, &c, &n)) v.mul_const(c, n);
  if (cvec(3, &c, &n)) v.add_const(c, n);
  if (cvec(10, &c, &n)) v.prelu(c, n);
  if (cvec(11, &c, &n)) v.mul_const(c, n);
  if (cvec(12, &c, &n)) v.add_const(c, n);
  if (ctx.has_in(13)) {  // G = fp32(q - g_zero) * g_scale
    FtzBound g;
    const float gs = static_cast<float>(a.get_number("g_scale", 1.0));
    g.mul_const(&gs, 1);
    v.mul(g);
  }
  if (cvec(14, &c, &n)) v.add_const(c, n);  // gate_add
  FtzBound r;  // R(res)
  if (a.get_string("r_code", "l8") == "l8") {
    const int e = static_cast<int>(a.get_int("r_exp", 0));
    // nonzero codes decode to +-ldexp(D, e + (k >> 3) - 16), D in [1, 2), k >= 2: >= 2^(e - 16), multiples of 2^(e - 39)
    r.L = std::ldexp(1.0, e - 16);
    r.G = std::ldexp(1.0, e - 39);
    r.ok = e - 16 >= -126 && e - 1 <= 127;
  } else {
    const float rs = static_cast<float>(a.get_number("r_scale", 1.0));
    r.mul_const(&rs, 1);
  }
  FtzBound u = v;
  u.add(r);
  FtzBound w = u;
  if (cvec(6, &c, &n)) w.prelu(c, n);
  if (cvec(7, &c, &n)) w.add_const(c, n);
  bool ok = v.ok && u.ok && w.ok;
  if (ok && cvec(8, &c, &n)) {
    const bool gt = a.get_string("bit_binarize", "ge") == "gt";
    for (int64_t i = 0; i < n && ok; ++i) {
      const float t = gt ? std::nextafter(c[i], std::numeric_limits<float>::infinity()) : c[i];
      ok = FtzBound::normal_or_zero(t);
    }
  }
  return ok;
}

// True when the fp32 epilogue of I8Conv2D / I8Dense (GemmEpilogue::f: fp32(acc) * scale[o] + bias[o], acc an integer)
// can produce no denormal value, so a flush-to-zero vector epilogue gives the IEEE values (and l8 codes).
inline bool gemm_f32_epilogue_normal(const KernelContext& ctx) {
  FtzBound v;
  v.mul_const(ctx.in(3).data<float>(), ctx.in(3).size());
  if (ctx.has_in(2)) v.add_const(ctx.in(2).data<float>(), ctx.in(2).size());
  return v.ok;
}

// Integer binarization thresholds of an int8 operand (per channel or scalar).
struct BinarizeI8 {
  q8::QBin mode = q8::QBin::kGe;
  std::vector<int16_t> tge, tgt;  // per channel (size C) or one entry
  void prepare(const KernelContext& ctx, int64_t c) {
    const std::string m = ctx.attrs().get_string("input_binarize", "ge");
    mode = m == "gt" ? q8::QBin::kGt : (m == "sign_zero" ? q8::QBin::kSignZero : q8::QBin::kGe);
    const q8::Deq d = deq_of(ctx.attrs(), "x");
    const float* thr = ctx.has_in(4) ? ctx.in(4).data<float>() : nullptr;
    const int64_t n = thr ? c : 1;
    tge.resize(static_cast<size_t>(n));
    tgt.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const float t = thr ? thr[i] : 0.f;
      tge[i] = static_cast<int16_t>(q8::qthr_ge(t, d));
      tgt[i] = static_cast<int16_t>(q8::qthr_gt(t, d));
    }
  }
  bool per_channel() const { return tge.size() > 1; }
  // Binarized values {-1,0,+1} of a row of c codes.
  void row(const int8_t* q, int64_t c, int8_t* out) const {
    for (int64_t i = 0; i < c; ++i) {
      const size_t j = per_channel() ? static_cast<size_t>(i) : 0;
      out[i] = static_cast<int8_t>(q8::qbin(mode, q[i], tge[j], tgt[j]));
    }
  }
};

// Epilogue of I8BDense / I8BConv2D: v = fp32(fp32(float(acc) * post_mul[o]) + post_add[o]);
// i8 -> clamp(rne(v), qmin, qmax); f32 -> v.
inline float binary_value(const KernelContext& ctx, int64_t o, int64_t acc) {
  const float v = static_cast<float>(acc) * ctx.in(2).data<float>()[o];
  return v + ctx.in(3).data<float>()[o];
}

inline void binary_store(const KernelContext& ctx, int64_t at, int64_t o, int64_t acc) {
  const float v = binary_value(ctx, o, acc);
  if (ctx.out(0).dtype() == DType::kF32) {
    ctx.out(0).data<float>()[at] = v;
  } else {
    ctx.out(0).data<int8_t>()[at] =
        q8::round_f32(v, static_cast<int>(ctx.attrs().get_int("qmin", -128)), static_cast<int>(ctx.attrs().get_int("qmax", 127)));
  }
}

// I8BMatMul parameters: operand binarization and the output combination
//   acc = c_ab*AB + c_rs*rowsum(A) + c_cs*colsum(B) + c0,  v = fp32(fp32(float(acc) * mul[j]) + add[j]).
struct BMatMulParams {
  q8::QBin ma = q8::QBin::kSignZero, mb = q8::QBin::kSignZero;
  int tge_a = 0, tgt_a = 0, tge_b = 0, tgt_b = 0;
  int64_t c_ab = 1, c_rs = 0, c_cs = 0, c0 = 0;
  const float* mul = nullptr;
  const float* add = nullptr;
  bool mul_vec = false, add_vec = false, f32 = false;
  int lo = -128, hi = 127;

  static q8::QBin qbin_of(const std::string& m) {
    switch (parse_bmode(m)) {
      case BMode::kGe: return q8::QBin::kGe;
      case BMode::kGt: return q8::QBin::kGt;
      case BMode::kSignZero: return q8::QBin::kSignZero;
      case BMode::kGt0: return q8::QBin::kGt0;
    }
    return q8::QBin::kSignZero;
  }
  explicit BMatMulParams(const KernelContext& ctx) {
    const Json& a = ctx.attrs();
    ma = qbin_of(a.get_string("a_binarize", "sign_zero"));
    mb = qbin_of(a.get_string("b_binarize", "sign_zero"));
    const float sa = static_cast<float>(a.get_number("a_scale", 1.0)), sb = static_cast<float>(a.get_number("b_scale", 1.0));
    const int32_t za = static_cast<int32_t>(a.get_int("a_zero", 0)), zb = static_cast<int32_t>(a.get_int("b_zero", 0));
    const float ta = static_cast<float>(a.get_number("a_threshold", 0.0)), tb = static_cast<float>(a.get_number("b_threshold", 0.0));
    tge_a = q8::qthr_ge(ta, sa, za);
    tgt_a = q8::qthr_gt(ta, sa, za);
    tge_b = q8::qthr_ge(tb, sb, zb);
    tgt_b = q8::qthr_gt(tb, sb, zb);
    c_ab = a.get_int("c_ab", 1);
    c_rs = a.get_int("c_rs", 0);
    c_cs = a.get_int("c_cs", 0);
    c0 = a.get_int("c0", 0);
    mul = ctx.in(2).data<float>();
    mul_vec = ctx.in(2).size() > 1;
    add = ctx.has_in(3) ? ctx.in(3).data<float>() : nullptr;
    add_vec = ctx.has_in(3) && ctx.in(3).size() > 1;
    f32 = a.get_string("output", "i8") == "f32";
    lo = static_cast<int>(a.get_int("qmin", -128));
    hi = static_cast<int>(a.get_int("qmax", 127));
  }
  int bin_a(int q) const { return q8::qbin(ma, q, tge_a, tgt_a); }
  int bin_b(int q) const { return q8::qbin(mb, q, tge_b, tgt_b); }
  int64_t combine(int64_t ab, int64_t rs, int64_t cs) const { return c_ab * ab + c_rs * rs + c_cs * cs + c0; }
  float value(int64_t acc, int64_t j) const {
    const float v = static_cast<float>(acc) * mul[mul_vec ? j : 0];
    return add ? v + add[add_vec ? j : 0] : v;
  }
  void store(const KernelContext& ctx, int64_t at, int64_t j, int64_t acc) const {
    const float v = value(acc, j);
    if (f32) ctx.out(0).data<float>()[at] = v;
    else ctx.out(0).data<int8_t>()[at] = q8::round_f32(v, lo, hi);
  }
};

}  // namespace int8_full
}  // namespace bitino
