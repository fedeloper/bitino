// Shared attribute parsing and geometry for operators. Operator contracts are
// documented in docs/operators.md; keep the two in sync.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include "bitino/error.h"
#include "bitino/model.h"

// Loops whose float compares must keep IEEE semantics (binarization of denormals) on 32-bit ARM, where clang
// auto-vectorizes them into AArch32 NEON compares, which flush denormals to zero (-1e-40 >= 0 becomes true).
// Elsewhere vector compares are IEEE and the macro is empty.
#if defined(__arm__) && !defined(__aarch64__) && defined(__clang__)
#define BF_A32_IEEE_LOOP _Pragma("clang loop vectorize(disable)")
#else
#define BF_A32_IEEE_LOOP
#endif

namespace bitino {

enum class Activation { kNone, kRelu, kRelu6 };

inline Activation parse_activation(const Json& attrs) {
  std::string a = attrs.get_string("activation", "none");
  if (a == "none") return Activation::kNone;
  if (a == "relu") return Activation::kRelu;
  if (a == "relu6") return Activation::kRelu6;
  fail("unknown fused activation '", a, "'");
}

inline float apply_activation(Activation a, float v) {
  switch (a) {
    case Activation::kNone: return v;
    case Activation::kRelu: return v > 0.f ? v : 0.f;
    case Activation::kRelu6: return std::min(std::max(v, 0.f), 6.f);
  }
  return v;
}

// How a real-valued activation x (with per-channel threshold t, default 0)
// maps to a binary value.
enum class BinarizeMode {
  kGe,        // +1 if x >= t else -1   (larq ste_sign/approx_sign: sign(0) = +1)
  kGt,        // +1 if x >  t else -1   (sign(0) = -1)
  kSignZero,  // torch.sign semantics: +1 / 0 / -1. Exact zeros contribute 0.
              // Only representable by reference kernels.
};

inline BinarizeMode parse_binarize(const Json& attrs) {
  std::string m = attrs.get_string("input_binarize", "ge");
  if (m == "ge") return BinarizeMode::kGe;
  if (m == "gt") return BinarizeMode::kGt;
  if (m == "sign_zero") return BinarizeMode::kSignZero;
  fail("unknown input_binarize mode '", m, "'");
}

inline int binarize(BinarizeMode m, float x, float t) {
  switch (m) {
    case BinarizeMode::kGe: return x >= t ? 1 : -1;
    case BinarizeMode::kGt: return x > t ? 1 : -1;
    case BinarizeMode::kSignZero: return x > t ? 1 : (x < t ? -1 : 0);
  }
  return 1;
}

struct Conv2DGeom {
  int64_t n = 0, ih = 0, iw = 0, ic = 0;  // input NHWC
  int64_t oc = 0, kh = 0, kw = 0, kic = 0;  // filter OHWI (kic = ic / groups)
  int64_t oh = 0, ow = 0;
  int64_t sh = 1, sw = 1, dh = 1, dw = 1;
  int64_t pt = 0, pb = 0, pl = 0, pr = 0;
  int64_t groups = 1;
};

inline Conv2DGeom conv_geometry(const Json& attrs, const Shape& x, const Shape& w) {
  BF_CHECK(x.size() == 4, "conv input must be NHWC rank 4, got ", shape_str(x));
  BF_CHECK(w.size() == 4, "conv filter must be OHWI rank 4, got ", shape_str(w));
  Conv2DGeom g;
  g.n = x[0]; g.ih = x[1]; g.iw = x[2]; g.ic = x[3];
  g.oc = w[0]; g.kh = w[1]; g.kw = w[2]; g.kic = w[3];
  auto st = attrs.get_ints("strides", {1, 1});
  auto dl = attrs.get_ints("dilations", {1, 1});
  auto pd = attrs.get_ints("pads", {0, 0, 0, 0});
  BF_CHECK(st.size() == 2 && dl.size() == 2 && pd.size() == 4, "strides/dilations need 2 values, pads 4 (top,bottom,left,right)");
  g.sh = st[0]; g.sw = st[1]; g.dh = dl[0]; g.dw = dl[1];
  g.pt = pd[0]; g.pb = pd[1]; g.pl = pd[2]; g.pr = pd[3];
  g.groups = attrs.get_int("groups", 1);
  BF_CHECK(g.groups >= 1 && g.ic % g.groups == 0 && g.oc % g.groups == 0, "invalid groups ", g.groups, " for ic=", g.ic, " oc=", g.oc);
  BF_CHECK(g.kic * g.groups == g.ic, "filter in-channels ", g.kic, " * groups ", g.groups, " != input channels ", g.ic);
  int64_t eh = g.dh * (g.kh - 1) + 1, ew = g.dw * (g.kw - 1) + 1;
  g.oh = (g.ih + g.pt + g.pb - eh) / g.sh + 1;
  g.ow = (g.iw + g.pl + g.pr - ew) / g.sw + 1;
  BF_CHECK(g.oh > 0 && g.ow > 0, "conv output is empty");
  return g;
}

struct Pool2DGeom {
  int64_t n, ih, iw, c, kh, kw, sh, sw, pt, pb, pl, pr, oh, ow;
};

inline Pool2DGeom pool_geometry(const Json& attrs, const Shape& x) {
  BF_CHECK(x.size() == 4, "pool input must be NHWC, got ", shape_str(x));
  auto k = attrs.get_ints("kernel", {});
  BF_CHECK(k.size() == 2, "pool needs kernel [kh,kw]");
  auto st = attrs.get_ints("strides", k);
  auto pd = attrs.get_ints("pads", {0, 0, 0, 0});
  Pool2DGeom g{x[0], x[1], x[2], x[3], k[0], k[1], st[0], st[1], pd[0], pd[1], pd[2], pd[3], 0, 0};
  g.oh = (g.ih + g.pt + g.pb - g.kh) / g.sh + 1;
  g.ow = (g.iw + g.pl + g.pr - g.kw) / g.sw + 1;
  BF_CHECK(g.oh > 0 && g.ow > 0, "pool output is empty");
  return g;
}

}  // namespace bitino
