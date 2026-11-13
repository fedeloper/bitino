// Operator semantics of the full-integer I8* operators (stage int8_full):
// attribute validation and shape inference. Contracts: docs/int8.md
// ("Full-integer execution: int8_full"); numpy oracle python/bitino/int8_ops.py.
// Quantization parameters are node attributes (x_zero, y_zero, ...) or constant
// inputs (requantization multipliers, tables); activations are int8.
#include <algorithm>

#include "ops/op_defs.h"
#include "ops/ops_common.h"
#include "ops/transformer_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

void set_out(Outs& outs, DType dt, Shape s) {
  BF_CHECK(!outs.empty(), "missing output");
  outs[0]->dtype = dt;
  outs[0]->shape = std::move(s);
}

const TensorInfo& req(const Ins& ins, size_t i, const char* what) {
  BF_CHECK(i < ins.size() && ins[i], "missing required input ", i, " (", what, ")");
  return *ins[i];
}

bool has(const Ins& ins, size_t i) { return i < ins.size() && ins[i]; }

const TensorInfo& req_i8(const Ins& ins, size_t i, const char* what) {
  const auto& t = req(ins, i, what);
  BF_CHECK(t.dtype == DType::kI8, what, " must be i8, got ", dtype_name(t.dtype));
  return t;
}

void check_vec(const Ins& ins, size_t i, DType dt, int64_t n, const char* what) {
  BF_CHECK(has(ins, i), "missing ", what);
  BF_CHECK(ins[i]->dtype == dt && num_elements(ins[i]->shape) == n, what, " must be ", dtype_name(dt), "[", n, "], got ",
           dtype_name(ins[i]->dtype), shape_str(ins[i]->shape));
}

void check_q(const Node& n) {
  int64_t lo = n.attrs.get_int("qmin", -128), hi = n.attrs.get_int("qmax", 127);
  BF_CHECK(-128 <= lo && lo <= hi && hi <= 127, "qmin/qmax must satisfy -128 <= qmin <= qmax <= 127");
}

DType out_dtype(const Node& n) {
  std::string o = n.attrs.get_string("output", "i8");
  BF_CHECK(o == "i8" || o == "f32", "output must be i8 or f32, got '", o, "'");
  return o == "i8" ? DType::kI8 : DType::kF32;
}

// I8Conv2D / I8Dense may also write an l8 code ("output": "l8", int8 tensor).
DType gemm_out_dtype(const Node& n) {
  if (n.attrs.get_string("output", "i8") == "l8") return DType::kI8;
  return out_dtype(n);
}

// x of the binary operators: int8 codes (linear: x_scale > 0, x_zero; or x_code "l8" with x_exp), or a bit tensor.
void check_binary_x(const Node& n, const TensorInfo& x) {
  if (x.dtype == DType::kBit) return;
  BF_CHECK(x.dtype == DType::kI8, n.op, ": x must be i8 or bit");
  const std::string code = n.attrs.get_string("x_code", "linear");
  BF_CHECK(code == "linear" || code == "l8", n.op, ": x_code must be linear or l8");
  if (code == "l8") {
    BF_CHECK(n.attrs.contains("x_exp"), n.op, " x_code l8 needs x_exp");
  } else {
    BF_CHECK(n.attrs.contains("x_scale") && n.attrs.get_number("x_scale", 0) > 0 && n.attrs.contains("x_zero"),
             n.op, " needs x_scale > 0 and x_zero");
  }
}

// Requantizing GEMM epilogue inputs (I8Conv2D / I8Dense): output i8 ->
// bias i32[O]? , mult i32[O], shift i32[O]; output f32 -> bias f32[O]?, scale f32[O].
void check_gemm_epilogue(const Node& n, const Ins& ins, int64_t o) {
  if (n.attrs.get_string("output", "i8") == "l8") {
    if (has(ins, 2)) check_vec(ins, 2, DType::kF32, o, "bias");
    check_vec(ins, 3, DType::kF32, o, "scale");
  } else if (out_dtype(n) == DType::kI8) {
    if (has(ins, 2)) check_vec(ins, 2, DType::kI32, o, "bias");
    check_vec(ins, 3, DType::kI32, o, "mult");
    check_vec(ins, 4, DType::kI32, o, "shift");
    check_q(n);
  } else {
    if (has(ins, 2)) check_vec(ins, 2, DType::kF32, o, "bias");
    check_vec(ins, 3, DType::kF32, o, "scale");
  }
}

void infer_quantize(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "x");
  BF_CHECK(x.dtype == DType::kF32, "I8Quantize expects f32");
  BF_CHECK(n.attrs.get_number("scale", 0) > 0, "I8Quantize needs scale > 0");
  set_out(outs, DType::kI8, x.shape);
}

void infer_dequantize(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "q");
  BF_CHECK(n.attrs.get_number("scale", 0) > 0, "I8Dequantize needs scale > 0");
  set_out(outs, DType::kF32, x.shape);
}

void infer_lut(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  const auto& t = req(ins, 1, "table");
  BF_CHECK(t.shape.size() == 2 && t.shape[1] == 256, "I8Lut table must be [T, 256]");
  int64_t ax = n.attrs.get_int("axis", -1);
  if (ax < 0) ax += static_cast<int64_t>(x.shape.size());
  BF_CHECK(!x.shape.empty() && ax >= 0 && ax < static_cast<int64_t>(x.shape.size()), "I8Lut axis out of range");
  BF_CHECK(t.shape[0] == 1 || t.shape[0] == x.shape[ax], "I8Lut table rows must be 1 or the channel count along axis");
  BF_CHECK(t.dtype == DType::kI8 || t.dtype == DType::kF32, "I8Lut table must be i8 or f32");
  set_out(outs, t.dtype, x.shape);
}

bool broadcastable(const Shape& a, const Shape& b) {
  if (b.size() > a.size()) return false;
  for (size_t i = 0; i < b.size(); ++i) {
    int64_t da = a[a.size() - 1 - i], db = b[b.size() - 1 - i];
    if (db != da && db != 1) return false;
  }
  return true;
}

void infer_add_mul(const Node& n, const Ins& ins, Outs& outs) {
  const auto& a = req_i8(ins, 0, "a");
  const auto& b = req_i8(ins, 1, "b");
  BF_CHECK(broadcastable(a.shape, b.shape), n.op, ": b ", shape_str(b.shape), " does not broadcast to a ", shape_str(a.shape));
  check_q(n);
  set_out(outs, DType::kI8, a.shape);
}

// Internal twin operators (created by the int8 plan-time rewrites, src/graph_rewrite_i8.h; never in a model file):
// I8LutN(x, T_0, ..., T_{k-1}) -> y_0..y_{k-1}: y_j = I8Lut(x, T_j) with int8 tables of 1 row or one row per channel of
// the last axis. I8AddN(a, b) -> y_0..y_{k-1}: y_j = I8Add(a, b) with the shared input attributes and the j-th entries
// of out_mults / out_shifts / y_zeros / qmins / qmaxs.
void infer_lutn(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  BF_CHECK(!x.shape.empty() && ins.size() >= 3 && outs.size() == ins.size() - 1, "I8LutN: x, k >= 2 tables, k outputs");
  for (size_t j = 1; j < ins.size(); ++j) {
    const auto& t = req(ins, j, "table");
    BF_CHECK(t.dtype == DType::kI8 && t.shape.size() == 2 && t.shape[1] == 256 && (t.shape[0] == 1 || t.shape[0] == x.shape.back()),
             "I8LutN tables must be i8 [1 | C, 256]");
    outs[j - 1]->dtype = DType::kI8;
    outs[j - 1]->shape = x.shape;
  }
}

// I8Conv2DN(x, w, bias?, mult_0, shift_0, ..., mult_{k-1}, shift_{k-1}) -> y_0..y_{k-1}: k I8Conv2D (int8 output, no bit
// output) with the same x, w, bias and geometry, differing in mult / shift and y_zeros[j] / qmins[j] / qmaxs[j].
void infer_conv2dn(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  const auto& w = req_i8(ins, 1, "w");
  auto g = conv_geometry(n.attrs, x.shape, w.shape);
  const size_t k = outs.size();
  BF_CHECK(k >= 2 && ins.size() == 3 + 2 * k, "I8Conv2DN: x, w, bias?, then mult / shift per output");
  if (has(ins, 2)) check_vec(ins, 2, DType::kI32, g.oc, "bias");
  for (const char* key : {"y_zeros", "qmins", "qmaxs"}) BF_CHECK(n.attrs.get_ints(key, {}).size() == k, "I8Conv2DN: ", key, " per output");
  const auto lo = n.attrs.get_ints("qmins", {}), hi = n.attrs.get_ints("qmaxs", {});
  for (size_t j = 0; j < k; ++j) {
    check_vec(ins, 3 + 2 * j, DType::kI32, g.oc, "mult");
    check_vec(ins, 4 + 2 * j, DType::kI32, g.oc, "shift");
    BF_CHECK(-128 <= lo[j] && lo[j] <= hi[j] && hi[j] <= 127, "I8Conv2DN: qmin/qmax out of range");
    outs[j]->dtype = DType::kI8;
    outs[j]->shape = {g.n, g.oh, g.ow, g.oc};
  }
}

// I8PackBits(x, bounds) -> bits (internal, src/graph_rewrite_i8.h fold_lut_thresholds): bit c of a row is 1 (= -1) iff
// bounds[0][c] <= x[c] <= bounds[1][c] (an empty interval lo > hi gives 0 = +1); bounds i8 [2, C], C = last axis of x.
void infer_packbits(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  const auto& b = req(ins, 1, "bounds");
  BF_CHECK(!x.shape.empty() && b.dtype == DType::kI8 && b.shape == Shape({2, x.shape.back()}), n.op, ": bounds must be i8 [2, C]");
  set_out(outs, DType::kBit, x.shape);
}

void infer_addn(const Node& n, const Ins& ins, Outs& outs) {
  const auto& a = req_i8(ins, 0, "a");
  const auto& b = req_i8(ins, 1, "b");
  BF_CHECK(broadcastable(a.shape, b.shape), n.op, ": b ", shape_str(b.shape), " does not broadcast to a ", shape_str(a.shape));
  const size_t k = outs.size();
  BF_CHECK(k >= 2, "I8AddN needs >= 2 outputs");
  for (const char* key : {"out_mults", "out_shifts", "y_zeros", "qmins", "qmaxs"})
    BF_CHECK(n.attrs.get_ints(key, {}).size() == k, "I8AddN: ", key, " needs one entry per output");
  const auto lo = n.attrs.get_ints("qmins", {}), hi = n.attrs.get_ints("qmaxs", {});
  for (size_t j = 0; j < k; ++j) {
    BF_CHECK(-128 <= lo[j] && lo[j] <= hi[j] && hi[j] <= 127, "I8AddN: qmin/qmax out of range");
    outs[j]->dtype = DType::kI8;
    outs[j]->shape = a.shape;
  }
}

// I8AffineSum(x_0..x_{N-1}, K i32[N, C'], B i32[C']), C' in {1, C}: same-shape int8 inputs.
void infer_affine_sum(const Node& n, const Ins& ins, Outs& outs) {
  BF_CHECK(ins.size() >= 3, "I8AffineSum needs at least one input plus K and B");
  const size_t nin = ins.size() - 2;
  const auto& x0 = req_i8(ins, 0, "x0");
  BF_CHECK(!x0.shape.empty(), "I8AffineSum needs rank >= 1");
  const int64_t c = x0.shape.back();
  for (size_t i = 1; i < nin; ++i) BF_CHECK(req_i8(ins, i, "x").shape == x0.shape, "I8AffineSum inputs must have equal shapes");
  const auto& k = req(ins, nin, "K");
  const auto& b = req(ins, nin + 1, "B");
  BF_CHECK(k.dtype == DType::kI32 && k.shape.size() == 2 && k.shape[0] == static_cast<int64_t>(nin) && (k.shape[1] == 1 || k.shape[1] == c),
           "I8AffineSum K must be i32[N, 1|C]");
  BF_CHECK(b.dtype == DType::kI32 && num_elements(b.shape) == k.shape[1], "I8AffineSum B must be i32[1|C] like K");
  BF_CHECK(n.attrs.get_ints("zeros", {}).size() == nin, "I8AffineSum needs one zero point per input");
  check_q(n);
  set_out(outs, DType::kI8, x0.shape);
}

// I8Conv2D: x, w, bias?, mult | scale, shift?, bit_thr i32[O]? ; outputs y, bits (bit, iff bit_thr is given):
// bit -1 iff !(acc >= bit_thr[o]), acc = sum (x - x_zero) * w before the bias (int8_ops, I8Conv2D).
void infer_conv(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  const auto& w = req_i8(ins, 1, "w");
  auto g = conv_geometry(n.attrs, x.shape, w.shape);
  check_gemm_epilogue(n, ins, g.oc);
  set_out(outs, gemm_out_dtype(n), {g.n, g.oh, g.ow, g.oc});
  if (has(ins, 5)) {
    check_vec(ins, 5, DType::kI32, g.oc, "bit_thr");
    BF_CHECK(outs.size() == 2 && outs[1], "I8Conv2D with bit_thr needs 2 outputs");
    outs[1]->dtype = DType::kBit;
    outs[1]->shape = {g.n, g.oh, g.ow, g.oc};
  } else {
    BF_CHECK(outs.size() == 1, "I8Conv2D without bit_thr has 1 output");
  }
}

void infer_dense(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  const auto& w = req_i8(ins, 1, "w");
  BF_CHECK(w.shape.size() == 2 && !x.shape.empty() && x.shape.back() == w.shape[1], "I8Dense: x ", shape_str(x.shape), " w ",
           shape_str(w.shape));
  BF_CHECK(w.shape[1] <= 133144, "I8Dense: K too large for exact int32 accumulation");
  check_gemm_epilogue(n, ins, w.shape[0]);
  Shape s = x.shape;
  s.back() = w.shape[0];
  set_out(outs, gemm_out_dtype(n), s);
}

void infer_matmul(const Node& n, const Ins& ins, Outs& outs) {
  const auto& a = req_i8(ins, 0, "a");
  const auto& b = req_i8(ins, 1, "b");
  auto g = matmul_geometry(n.attrs, a.shape, b.shape);
  BF_CHECK(g.k <= 33025, "I8MatMul: K too large for exact int32 accumulation of zero-point-corrected products");
  if (out_dtype(n) == DType::kI8) {
    BF_CHECK(n.attrs.contains("mult") && n.attrs.contains("shift"), "I8MatMul needs mult/shift");
    check_q(n);
  } else {
    BF_CHECK(n.attrs.contains("scale"), "I8MatMul f32 output needs scale");
  }
  set_out(outs, out_dtype(n), g.out);
}

void infer_layernorm(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  BF_CHECK(!x.shape.empty(), "I8LayerNorm needs rank >= 1");
  int64_t d = x.shape.back();
  BF_CHECK(d <= 16384, "I8LayerNorm: D too large");
  check_vec(ins, 1, DType::kF32, d, "G");
  check_vec(ins, 2, DType::kF32, d, "B");
  BF_CHECK(n.attrs.get_number("x_scale", 0) > 0 && n.attrs.get_number("eps", 1e-5) >= 0, "I8LayerNorm needs x_scale > 0, eps >= 0");
  check_q(n);
  set_out(outs, DType::kI8, x.shape);
}

// I8LayerNormN(x, G_0, B_0, ..., G_{k-1}, B_{k-1}) -> y_0..y_{k-1} (internal twins, src/graph_rewrite_i8.cc): k I8LayerNorm
// of x with the same x_zero / x_scale / eps, output j with G_j, B_j, qmins[j], qmaxs[j].
void infer_layernormn(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  BF_CHECK(!x.shape.empty() && x.shape.back() <= 16384, "I8LayerNormN: bad x");
  const size_t k = outs.size();
  BF_CHECK(k >= 2 && k <= 4 && ins.size() == 1 + 2 * k, "I8LayerNormN: x, then G / B per output (2-4 outputs)");
  BF_CHECK(n.attrs.get_ints("qmins", {}).size() == k && n.attrs.get_ints("qmaxs", {}).size() == k, "I8LayerNormN: qmins / qmaxs per output");
  const auto lo = n.attrs.get_ints("qmins", {}), hi = n.attrs.get_ints("qmaxs", {});
  for (size_t j = 0; j < k; ++j) {
    check_vec(ins, 1 + 2 * j, DType::kF32, x.shape.back(), "G");
    check_vec(ins, 2 + 2 * j, DType::kF32, x.shape.back(), "B");
    BF_CHECK(-128 <= lo[j] && lo[j] <= hi[j] && hi[j] <= 127, "I8LayerNormN: qmin/qmax out of range");
    outs[j]->dtype = DType::kI8;
    outs[j]->shape = x.shape;
  }
}

void infer_softmax(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  check_vec(ins, 1, DType::kI32, 256, "exp table");
  BF_CHECK(!x.shape.empty() && x.shape.back() <= 65536, "I8Softmax: bad row length");
  DType o = out_dtype(n);
  if (o == DType::kI8) BF_CHECK(n.attrs.get_number("inv_scale", 0) > 0, "I8Softmax needs inv_scale > 0");
  check_q(n);
  set_out(outs, o, x.shape);
}

void infer_concat(const Node& n, const Ins& ins, Outs& outs) {
  BF_CHECK(!ins.empty(), "I8Concat needs inputs");
  const auto& x0 = req_i8(ins, 0, "x0");
  int r = static_cast<int>(x0.shape.size());
  int64_t ax = n.attrs.get_int("axis", -1);
  if (ax < 0) ax += r;
  BF_CHECK(ax >= 0 && ax < r, "I8Concat axis out of range");
  auto zs = n.attrs.get_ints("zeros", {}), ms = n.attrs.get_ints("mults", {}), ss = n.attrs.get_ints("shifts", {});
  BF_CHECK(zs.size() == ins.size() && ms.size() == ins.size() && ss.size() == ins.size(), "I8Concat needs zeros/mults/shifts per input");
  Shape s = x0.shape;
  s[ax] = 0;
  for (size_t i = 0; i < ins.size(); ++i) {
    const auto& x = req_i8(ins, i, "x");
    BF_CHECK(static_cast<int>(x.shape.size()) == r, "I8Concat rank mismatch");
    for (int d = 0; d < r; ++d)
      if (d != ax) BF_CHECK(x.shape[d] == x0.shape[d], "I8Concat shape mismatch");
    s[ax] += x.shape[ax];
  }
  set_out(outs, DType::kI8, s);
}

void infer_transpose(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  auto p = n.attrs.get_ints("perm", {});
  BF_CHECK(p.size() == x.shape.size() && p.size() <= 8, "I8Transpose perm rank");
  Shape s(p.size());
  std::vector<bool> seen(p.size(), false);
  for (size_t i = 0; i < p.size(); ++i) {
    BF_CHECK(p[i] >= 0 && p[i] < static_cast<int64_t>(p.size()) && !seen[p[i]], "I8Transpose perm invalid");
    seen[p[i]] = true;
    s[i] = x.shape[p[i]];
  }
  set_out(outs, DType::kI8, s);
}

void infer_gather(const Node&, const Ins& ins, Outs& outs) {
  const auto& t = req(ins, 0, "table");
  const auto& idx = req(ins, 1, "indices");
  BF_CHECK(t.shape.size() == 2 && (t.dtype == DType::kI8 || t.dtype == DType::kBit), "I8Gather table must be i8|bit [V, D]");
  BF_CHECK(idx.dtype == DType::kI32, "I8Gather indices must be i32");
  if (t.dtype == DType::kBit) {
    BF_CHECK(has(ins, 2) && ins[2]->dtype == DType::kI8 && ins[2]->shape == Shape({t.shape[0], 2}), "I8Gather bit table needs qpm i8[V,2]");
    if (has(ins, 3)) BF_CHECK(ins[3]->dtype == DType::kBit && ins[3]->shape == t.shape, "I8Gather zero_mask shape");
  }
  Shape s = idx.shape;
  s.push_back(t.shape[1]);
  set_out(outs, DType::kI8, s);
}

// I8MaxPool2D also pools bit tensors: +1 iff some element of the window is +1 (the maximum of +-1 values).
void infer_pool(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "x");
  BF_CHECK(x.dtype == DType::kI8 || (x.dtype == DType::kBit && n.op == "I8MaxPool2D"), n.op, ": x must be i8",
           n.op == "I8MaxPool2D" ? " or bit" : "");
  auto g = pool_geometry(n.attrs, x.shape);
  if (x.dtype == DType::kBit)  // no empty windows
    BF_CHECK(g.pt < g.kh && g.pb < g.kh && g.pl < g.kw && g.pr < g.kw, n.op, ": bit pooling needs padding < kernel");
  set_out(outs, x.dtype, {g.n, g.oh, g.ow, g.c});
}

void infer_gap(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  BF_CHECK(x.shape.size() == 4, "I8GlobalAvgPool expects NHWC");
  bool keep = n.attrs.get_int("keepdims", 0) != 0;
  set_out(outs, DType::kI8, keep ? Shape{x.shape[0], 1, 1, x.shape[3]} : Shape{x.shape[0], x.shape[3]});
}

void infer_reduce_mean(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  int r = static_cast<int>(x.shape.size());
  int64_t ax = n.attrs.get_int("axis", -1);
  if (ax < 0) ax += r;
  BF_CHECK(ax >= 0 && ax < r, "I8ReduceMean axis");
  Shape s;
  for (int d = 0; d < r; ++d) {
    if (d != ax) s.push_back(x.shape[d]);
    else if (n.attrs.get_int("keepdims", 0)) s.push_back(1);
  }
  set_out(outs, DType::kI8, s);
}

void infer_pad(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8(ins, 0, "x");
  auto p = n.attrs.get_ints("pads", {});
  BF_CHECK(p.size() == 2 * x.shape.size(), "I8Pad needs 2 values per dim");
  Shape s = x.shape;
  for (size_t d = 0; d < s.size(); ++d) s[d] += p[2 * d] + p[2 * d + 1];
  set_out(outs, DType::kI8, s);
}

void check_binary_epilogue(const Node& n, const Ins& ins, int64_t o) {
  check_vec(ins, 2, DType::kF32, o, "post_mul");
  check_vec(ins, 3, DType::kF32, o, "post_add");
  check_binary_x(n, *ins[0]);
  check_q(n);
}

void infer_bdense(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "x");  // int8 codes, or a bit tensor (Session rewrites only, e.g. I8PackBits)
  check_binary_x(n, x);
  const auto& w = req(ins, 1, "w");
  BF_CHECK(w.dtype == DType::kBit && w.shape.size() == 2 && x.shape.back() == w.shape[1], "I8BDense: w must be bit [O, K]");
  BinarizeMode m = parse_binarize(n.attrs);
  check_binary_epilogue(n, ins, w.shape[0]);
  if (has(ins, 4)) check_vec(ins, 4, DType::kF32, w.shape[1], "threshold");
  if (has(ins, 5)) {
    BF_CHECK(m == BinarizeMode::kSignZero, "I8BDense w_zero requires sign_zero");
    BF_CHECK(ins[5]->dtype == DType::kBit && ins[5]->shape == w.shape, "I8BDense w_zero shape");
  }
  Shape s = x.shape;
  s.back() = w.shape[0];
  set_out(outs, out_dtype(n), s);
}

void infer_bconv(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "x");
  check_binary_x(n, x);
  const auto& w = req(ins, 1, "w");
  BF_CHECK(w.dtype == DType::kBit, "I8BConv2D: w must be bit");
  auto g = conv_geometry(n.attrs, x.shape, w.shape);
  parse_binarize(n.attrs);
  int64_t pv = n.attrs.get_int("pad_value", 0);
  BF_CHECK(pv >= -1 && pv <= 1, "I8BConv2D pad_value must be -1, 0 or 1");
  check_binary_epilogue(n, ins, g.oc);
  if (has(ins, 4)) check_vec(ins, 4, DType::kF32, g.ic, "threshold");
  set_out(outs, out_dtype(n), {g.n, g.oh, g.ow, g.oc});
}

// I8BConv2DRes (int8_ops._bconv_res): inputs x, w, post_mul, post_add, threshold?, residual i8 [N,OH,OW,O],
// slope f32[O], out_bias f32[O]?, bit_thr f32[O]?, out_lut i8[1, 256]? (Session rewrite only); outputs y (i8 l8 code, or
// out_lut of it), bits (bit, iff bit_thr is given).
void infer_bconv_res(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "x");
  check_binary_x(n, x);
  const auto& w = req(ins, 1, "w");
  BF_CHECK(w.dtype == DType::kBit, "I8BConv2DRes: w must be bit");
  auto g = conv_geometry(n.attrs, x.shape, w.shape);
  parse_binarize(n.attrs);
  int64_t pv = n.attrs.get_int("pad_value", 0);
  BF_CHECK(pv >= -1 && pv <= 1, "I8BConv2DRes pad_value must be -1, 0 or 1");
  BF_CHECK(n.attrs.get_string("output", "l8") == "l8", "I8BConv2DRes output must be l8");
  check_vec(ins, 2, DType::kF32, g.oc, "post_mul");
  check_vec(ins, 3, DType::kF32, g.oc, "post_add");
  if (has(ins, 4)) check_vec(ins, 4, DType::kF32, g.ic, "threshold");
  const auto& r = req_i8(ins, 5, "residual");
  BF_CHECK(r.shape == Shape({g.n, g.oh, g.ow, g.oc}), "I8BConv2DRes residual shape ", shape_str(r.shape));
  const std::string rc = n.attrs.get_string("r_code", "l8");
  BF_CHECK(rc == "l8" || rc == "linear", "I8BConv2DRes r_code must be l8 or linear");
  if (rc == "l8") {
    const int64_t e = n.attrs.get_int("r_exp", 0);
    BF_CHECK(e >= -100 && e <= 100, "I8BConv2DRes r_exp out of range");
  }
  check_vec(ins, 6, DType::kF32, g.oc, "slope");
  if (has(ins, 7)) check_vec(ins, 7, DType::kF32, g.oc, "out_bias");
  if (has(ins, 9)) {  // out_lut (runtime rewrite, Session: an I8Lut absorbed): output 0 = out_lut[l8 code + 128]
    const auto& t = *ins[9];
    BF_CHECK(t.dtype == DType::kI8 && t.shape == Shape({1, 256}), "I8BConv2DRes out_lut must be i8 [1, 256]");
  }
  // pre stage (int8_ops.bconv_res_pre): pre_slope / pre_mul / pre_add f32[O]?, gate i8 [N, 1, 1, O]? (g_scale > 0, g_zero)
  if (has(ins, 10)) check_vec(ins, 10, DType::kF32, g.oc, "pre_slope");
  if (has(ins, 11)) check_vec(ins, 11, DType::kF32, g.oc, "pre_mul");
  if (has(ins, 12)) check_vec(ins, 12, DType::kF32, g.oc, "pre_add");
  if (has(ins, 13)) {
    const auto& gt = req_i8(ins, 13, "gate");
    BF_CHECK(gt.shape == Shape({g.n, 1, 1, g.oc}), "I8BConv2DRes gate shape ", shape_str(gt.shape));
    BF_CHECK(n.attrs.get_number("g_scale", 0) > 0, "I8BConv2DRes gate needs g_scale > 0");
  }
  set_out(outs, DType::kI8, {g.n, g.oh, g.ow, g.oc});
  if (has(ins, 8)) {
    check_vec(ins, 8, DType::kF32, g.oc, "bit_thr");
    const std::string bm = n.attrs.get_string("bit_binarize", "ge");
    BF_CHECK(bm == "ge" || bm == "gt", "I8BConv2DRes bit_binarize must be ge or gt");
    const std::string bs = n.attrs.get_string("bit_src", "sum");
    BF_CHECK(bs == "sum" || bs == "out", "I8BConv2DRes bit_src must be sum or out");
    BF_CHECK(outs.size() == 2 && outs[1], "I8BConv2DRes with bit_thr needs 2 outputs");
    outs[1]->dtype = DType::kBit;
    outs[1]->shape = {g.n, g.oh, g.ow, g.oc};
  } else {
    BF_CHECK(outs.size() == 1, "I8BConv2DRes without bit_thr has 1 output");
  }
}

void infer_bmatmul(const Node& n, const Ins& ins, Outs& outs) {
  const auto& a = req_i8(ins, 0, "a");
  const auto& b = req_i8(ins, 1, "b");
  auto g = matmul_geometry(n.attrs, a.shape, b.shape);
  parse_bmode(n.attrs.get_string("a_binarize", "sign_zero"));
  parse_bmode(n.attrs.get_string("b_binarize", "sign_zero"));
  BF_CHECK(n.attrs.get_number("a_scale", 0) > 0 && n.attrs.get_number("b_scale", 0) > 0, "I8BMatMul needs a_scale, b_scale > 0");
  BF_CHECK(g.k < (int64_t(1) << 20), "I8BMatMul K too large");
  BF_CHECK(has(ins, 2) && ins[2]->dtype == DType::kF32 && (num_elements(ins[2]->shape) == 1 || num_elements(ins[2]->shape) == g.n),
           "I8BMatMul mul must be f32[1] or f32[N]");
  if (has(ins, 3))
    BF_CHECK(ins[3]->dtype == DType::kF32 && (num_elements(ins[3]->shape) == 1 || num_elements(ins[3]->shape) == g.n),
             "I8BMatMul add must be f32[1] or f32[N]");
  check_q(n);
  set_out(outs, out_dtype(n), g.out);
}

}  // namespace

void register_int8_full_op_defs(OpRegistry& reg) {
  reg.add_op({"I8Quantize", infer_quantize});
  reg.add_op({"I8Dequantize", infer_dequantize});
  reg.add_op({"I8Lut", infer_lut});
  reg.add_op({"I8Add", infer_add_mul});
  reg.add_op({"I8Mul", infer_add_mul});
  reg.add_op({"I8AffineSum", infer_affine_sum});
  reg.add_op({"I8LutN", infer_lutn});
  reg.add_op({"I8AddN", infer_addn});
  reg.add_op({"I8PackBits", infer_packbits});
  reg.add_op({"I8Conv2DN", infer_conv2dn});
  reg.add_op({"I8Conv2D", infer_conv});
  reg.add_op({"I8Dense", infer_dense});
  reg.add_op({"I8MatMul", infer_matmul});
  reg.add_op({"I8LayerNorm", infer_layernorm});
  reg.add_op({"I8LayerNormN", infer_layernormn});
  reg.add_op({"I8Softmax", infer_softmax});
  reg.add_op({"I8Concat", infer_concat});
  reg.add_op({"I8Transpose", infer_transpose});
  reg.add_op({"I8Gather", infer_gather});
  reg.add_op({"I8MaxPool2D", infer_pool});
  reg.add_op({"I8AvgPool2D", infer_pool});
  reg.add_op({"I8GlobalAvgPool", infer_gap});
  reg.add_op({"I8ReduceMean", infer_reduce_mean});
  reg.add_op({"I8Pad", infer_pad});
  reg.add_op({"I8BDense", infer_bdense});
  reg.add_op({"I8BConv2D", infer_bconv});
  reg.add_op({"I8BConv2DRes", infer_bconv_res});
  reg.add_op({"I8BMatMul", infer_bmatmul});
}

}  // namespace bitino
