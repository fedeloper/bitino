// Shared pieces of the full-integer (int8_full) kernels: parameter parsing and
// the scalar definitions every implementation must reproduce bit for bit.
// Contracts: docs/int8.md ("Full-integer execution: int8_full"); oracle
// python/bitino/int8_ops.py.
#pragma once

#include <cmath>
#include <cstdint>
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
    for (int q = -128; q <= 127; ++q)
      dec[q + 128] = r_l8 ? q8::l8_decode(q, r_exp) : static_cast<float>(static_cast<int64_t>(q) - r_zero) * r_scale;
    pre_slope = ctx.has_in(10) ? ctx.in(10).data<float>() : nullptr;
    pre_mul = ctx.has_in(11) ? ctx.in(11).data<float>() : nullptr;
    pre_add = ctx.has_in(12) ? ctx.in(12).data<float>() : nullptr;
    gate = ctx.has_in(13);
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
  float g_scale = 1.f;
  int32_t g_zero = 0;
  std::vector<float> gval;  // G of the current run, [N * O] (bind_gate)
  bool pre_on() const { return pre_slope || pre_mul || pre_add || gate; }
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
    const int8_t c = q8::l8_encode(w);
    return out_lut ? out_lut[c + 128] : c;
  }
};

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
