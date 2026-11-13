// int8_full speed work of 2026-11-12 (docs/int8.md, "int8_full latency: fused element-wise chains"):
//  * new fast paths of the NEON element-wise kernels (I8AffineSum shift requantization, I8Mul int16 products,
//    I8Lut per-channel 64-row tiles) bit-exact with the reference kernels on random parameters, edge values,
//    tails, 1 and 4 threads;
//  * the int8 plan-time rewrites (src/graph_rewrite_i8.cc): every rewritten graph bit-identical to the same graph
//    run without Session rewrites (SessionOptions::fuse off) and to the reference tier, for every rewrite and op
//    combination, odd sizes and tails, 1 and 4 threads.
// On x86-64 the int8_full stage runs the reference / portable kernels; the rewrite tests are exact there too.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <set>

#include "bitino/bitino.h"
#include "kernels/int8_full/q8_math.h"
#include "testing.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

struct Rng {
  std::mt19937 g;
  explicit Rng(uint32_t s) : g(s) {}
  int i(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
  int64_t l(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(g); }
  float f(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g); }
  std::vector<uint8_t> i8(int64_t n) {
    std::vector<uint8_t> v(static_cast<size_t>(n));
    for (auto& x : v) x = static_cast<uint8_t>(i(-128, 127));
    return v;
  }
  std::pair<int32_t, int32_t> mult(double lo, double hi) {
    double r = std::exp(f(static_cast<float>(std::log(lo)), static_cast<float>(std::log(hi))));
    int e = 0;
    double fr = std::frexp(r, &e);
    int64_t m = std::llround(fr * 2147483648.0);
    if (m == (int64_t(1) << 31)) m /= 2, ++e;
    return {static_cast<int32_t>(m), e};
  }
};

const char* kRef = "reference,int8_full_reference";
const char* kFull = "reference,binary_scalar,neon,binary_avx2,float_opt,int8_full_reference,int8_full_opt,int8_full_neon";

std::set<std::string> tiers_of(const std::string& s) {
  std::set<std::string> t;
  size_t p = 0;
  while (p <= s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    t.insert(s.substr(p, q - p));
    p = q + 1;
  }
  return t;
}

struct RunInfo {
  std::map<std::string, std::string> kernels;
  size_t nodes = 0;
};

// All graph outputs (bytes) at the given tiers; fuse = Session rewrites on / off.
std::vector<uint8_t> run_all(const bftest::ModelBuilder& mb, const std::string& tiers, int threads, bool fuse,
                             const std::vector<std::vector<uint8_t>>& inputs, RunInfo* info = nullptr) {
  auto model = mb.build();
  SessionOptions so;
  so.num_threads = threads;
  so.spin_run = threads > 1;
  so.fuse = fuse;
  so.policy = KernelPolicy::tiers(tiers_of(tiers));
  Session s(model, so);
  for (size_t i = 0; i < inputs.size(); ++i) std::memcpy(s.input(i).raw(), inputs[i].data(), inputs[i].size());
  s.run();
  if (info) {
    info->kernels = s.kernel_choices();
    info->nodes = s.kernel_choices().size();
  }
  std::vector<uint8_t> out;
  for (size_t i = 0; i < s.num_outputs(); ++i) {
    const Tensor& y = s.output(i);
    const auto* b = static_cast<const uint8_t*>(y.raw());
    out.insert(out.end(), b, b + y.bytes());
  }
  return out;
}

// Reference tier without rewrites == int8_full stage with and without rewrites, 1 and 4 threads. On AArch64 the
// kernel `neon_kernel` (if not empty) must have been selected by an int8_full run (rewritten or not: a rewrite may
// replace the node under test); `max_nodes` (if >= 0) bounds the node count of the rewritten graph (the rewrite must
// have fired).
int64_t g_cases = 0;
void compare(const bftest::ModelBuilder& mb, const std::vector<std::vector<uint8_t>>& in, const std::string& what,
             const std::string& neon_kernel = "", int max_nodes = -1) {
  const auto ref = run_all(mb, kRef, 1, false, in);
  for (int t : {1, 4}) {
    bool found = neon_kernel.empty();
    for (bool fuse : {false, true}) {
      RunInfo info;
      const auto got = run_all(mb, kFull, t, fuse, in, &info);
      if (got != ref)
        throw bftest::Failure{"int8_full output differs from the reference (" + what + ", threads " + std::to_string(t) +
                              (fuse ? ", rewrites on)" : ", rewrites off)")};
      if (fuse && max_nodes >= 0 && static_cast<int>(info.nodes) > max_nodes)
        throw bftest::Failure{"rewrite did not fire for " + what + ": " + std::to_string(info.nodes) + " nodes"};
      for (const auto& kv : info.kernels) found = found || kv.second == neon_kernel;
    }
#if defined(__aarch64__)
    if (!found) throw bftest::Failure{"expected kernel " + neon_kernel + " for " + what};
#endif
    // the reference tier with the rewrites (the fused reference kernels)
    const auto rr = run_all(mb, kRef, t, true, in);
    if (rr != ref) throw bftest::Failure{"reference tier with rewrites differs (" + what + ")"};
  }
  ++g_cases;
}

Json zeros_json(const std::vector<int64_t>& z) {
  Json::Array a;
  for (auto v : z) a.push_back(Json(v));
  return Json(a);
}

}  // namespace

// I8AffineSum with mult = 2^30, shift <= 0: the shift requantization of AffineSumNeon (bias with the rounding
// constants and the zero point, SHRN #16, int16 shift, prescaled coefficients for s < 16). Every shift 0..-30,
// 1..5 inputs, per-channel and scalar coefficients, channel tails, clamped outputs, accumulators near the int32 bound.
BF_TEST(int8_full_affinesum_shift_requant_bit_exact) {
  Rng r(1112);
  for (int shift = 0; shift >= -30; --shift) {
    for (int nin : {1, 2, 3, 4, 5}) {
      const int64_t c = std::vector<int64_t>{16, 19, 48, 7, 100}[static_cast<size_t>((-shift + nin) % 5)];
      const int64_t rows = 1 + (-shift * 7 + nin) % 41;
      const bool pc = (shift + nin) % 3 != 0;
      const int variant = (-shift + 2 * nin) % 4;  // 0 random, 1 near the bound, 2 clamped, 3 int16 coefficients
      bftest::ModelBuilder mb;
      std::vector<std::string> ins;
      std::vector<std::vector<uint8_t>> data;
      std::vector<int64_t> zs;
      for (int i = 0; i < nin; ++i) {
        mb.input("x" + std::to_string(i), DType::kI8, {rows, c});
        ins.push_back("x" + std::to_string(i));
        auto d = r.i8(rows * c);
        for (int64_t k = 0; k < rows * c; k += 5) d[k] = (k / 5) % 2 ? 0x80 : 0x7F;  // extreme codes
        data.push_back(d);
        zs.push_back(r.i(-128, 127));
      }
      const int64_t kc = pc ? c : 1;
      std::vector<int32_t> K(static_cast<size_t>(nin * kc)), B(static_cast<size_t>(kc));
      // Variants 0, 2, 3 keep (|B'| + sum |K| 128) 2^pre < 2^31 (the shift path, pre = max(0, 16 - s) the prescale
      // exponent); variant 1 is close to the general fast-path bound sum |K| (128 + |z|) + |B| < 2^31 (the shift path's
      // bound fails for most shifts: the SQRDMULH path).
      const int s = -shift + 1, pre = std::max(0, 16 - s);
      const int64_t kshr = (int64_t(1) << (29 - pre)) / (nin * 128);
      const int64_t kmax = variant == 1 ? (int64_t(1) << 31) / (nin * 260) : (variant == 3 ? std::min<int64_t>(32767, kshr) : kshr);
      const int64_t bmax = variant == 1 ? (int64_t(1) << 24) : (int64_t(1) << (28 - pre));
      for (auto& v : K) v = static_cast<int32_t>(r.l(-kmax, kmax));
      for (auto& v : B) v = static_cast<int32_t>(r.l(-bmax, bmax));
      mb.constant("K", DType::kI32, {nin, kc}, K.data());
      mb.constant("B", DType::kI32, {kc}, B.data());
      ins.push_back("K");
      ins.push_back("B");
      Json at = Json::Object{{"zeros", zeros_json(zs)}, {"mult", 1 << 30}, {"shift", shift}, {"y_zero", r.i(-128, 127)}};
      if (variant == 2) {
        at["qmin"] = r.i(-128, -20);
        at["qmax"] = r.i(0, 127);
      }
      mb.node("I8AffineSum", ins, {"y"}, at);
      mb.output("y");
      compare(mb, data, "I8AffineSum shift " + std::to_string(shift) + " nin " + std::to_string(nin), "i8affinesum.neon");
    }
  }
  EXPECT_TRUE(g_cases > 0);
}

// I8AffineSum with mult 2^30 and a positive shift (2026-11-13; QuickNet's binarize-aware twins have shift 1): the shift
// path with coefficients scaled by 2^(shift-1), prescaled (small K) or shifted by 0 (mode 3), near the no-saturation
// bound acc 2^shift < 2^31, and beyond it (saturating: the SQSHL path); clamped outputs, channel tails, 1-4 inputs.
BF_TEST(int8_full_affinesum_positive_shift_bit_exact) {
  Rng r(1114);
  int k = 0;
  for (int shift = 1; shift <= 6; ++shift) {
    for (int nin : {1, 2, 3, 4}) {
      for (int variant = 0; variant < 4; ++variant) {  // 0 small K (prescaled), 1 int16 K, 2 near the bound, 3 saturating
        ++k;
        const int64_t c = std::vector<int64_t>{16, 19, 48, 7, 100}[static_cast<size_t>(k % 5)];
        const int64_t rows = 1 + (k * 7) % 41;
        const bool pc = k % 3 != 0;
        bftest::ModelBuilder mb;
        std::vector<std::string> ins;
        std::vector<std::vector<uint8_t>> data;
        std::vector<int64_t> zs;
        for (int i = 0; i < nin; ++i) {
          mb.input("x" + std::to_string(i), DType::kI8, {rows, c});
          ins.push_back("x" + std::to_string(i));
          auto d = r.i8(rows * c);
          for (int64_t q = 0; q < rows * c; q += 5) d[q] = (q / 5) % 2 ? 0x80 : 0x7F;  // extreme codes
          data.push_back(d);
          zs.push_back(r.i(-128, 127));
        }
        const int64_t kc = pc ? c : 1;
        std::vector<int32_t> K(static_cast<size_t>(nin * kc)), B(static_cast<size_t>(kc));
        const int64_t lim = (int64_t(1) << (31 - shift)) / (nin * 260);  // sum |K| (128 + |z|) 2^shift < 2^31
        // variant 3: |acc| < 2^31 (the quantizer's bound) but acc 2^shift saturates
        const int64_t kmax = variant == 0 ? std::max<int64_t>(1, (int64_t(1) << (14 - shift)) / (nin * 128))
                                          : (variant == 1 ? std::min<int64_t>(32767, lim) : (variant == 2 ? lim : (int64_t(1) << 30) / (nin * 256)));
        const int64_t bmax = variant == 0 ? 64 : (variant == 3 ? (int64_t(1) << 20) : lim * 8);
        for (auto& v : K) v = static_cast<int32_t>(r.l(-kmax, kmax));
        for (auto& v : B) v = static_cast<int32_t>(r.l(-bmax, bmax));
        mb.constant("K", DType::kI32, {nin, kc}, K.data());
        mb.constant("B", DType::kI32, {kc}, B.data());
        ins.push_back("K");
        ins.push_back("B");
        Json at = Json::Object{{"zeros", zeros_json(zs)}, {"mult", 1 << 30}, {"shift", shift}, {"y_zero", r.i(-128, 127)}};
        if (k % 4 == 1) {
          at["qmin"] = r.i(-128, -20);
          at["qmax"] = r.i(0, 127);
        }
        mb.node("I8AffineSum", ins, {"y"}, at);
        mb.output("y");
        compare(mb, data, "I8AffineSum positive shift " + std::to_string(shift) + " nin " + std::to_string(nin) + " variant " +
                              std::to_string(variant), "i8affinesum.neon");
      }
    }
  }
}

// I8Mul with int8 zero points: int16 differences and SMULL products (same shape and per-sample channel broadcast,
// negative / zero / positive shifts, clamped outputs, channel tails), and zero points outside int8 (the int32 path).
BF_TEST(int8_full_mul_s16_bit_exact) {
  Rng r(1113);
  int k = 0;
  for (int64_t c : {16, 37, 64, 5}) {
    for (int bc = 0; bc < 2; ++bc) {
      for (int sv = 0; sv < 3; ++sv) {
        ++k;
        const int64_t n = 2, h = 1 + k % 5, w = 3 + k % 4;
        bftest::ModelBuilder mb;
        mb.input("a", DType::kI8, {n, h, w, c});
        mb.input("b", DType::kI8, bc ? Shape{n, 1, 1, c} : Shape{n, h, w, c});
        auto mm = sv == 0 ? r.mult(1e-5, 0.4) : (sv == 1 ? r.mult(0.5, 0.99) : r.mult(3.0, 70000.0));
        if (sv == 1) mm.second = 0;  // multiplier in [0.5, 1): shift 0
        const bool wide = k % 7 == 0;  // zero points outside int8: the int32 path
        Json at = Json::Object{{"a_zero", wide ? r.i(-1000, -200) : r.i(-128, 127)}, {"b_zero", r.i(-128, 127)},
                               {"mult", mm.first}, {"shift", mm.second}, {"y_zero", r.i(-128, 127)}};
        if (k % 3 == 0) {
          at["qmin"] = r.i(-128, -10);
          at["qmax"] = r.i(10, 127);
        }
        mb.node("I8Mul", {"a", "b"}, {"y"}, at);
        mb.output("y");
        auto a = r.i8(n * h * w * c), b = r.i8(bc ? n * c : n * h * w * c);
        a[0] = 0x80;
        b[0] = 0x80;
        a[1] = 0x7F;
        b[1 % b.size()] = 0x7F;
        compare(mb, {a, b}, "I8Mul case " + std::to_string(k), "i8mul.neon");
      }
    }
  }
}

// I8Add with one input multiplier 2^30 / shift 0 (TFLite's larger input scale): run_half (term (q - z) 2^19 without
// SQRDMULH), a or b the half input, same shapes and a broadcast row, out shift negative / positive, clamped outputs,
// tails; one case with both inputs 2^30 / 0.
BF_TEST(int8_full_add_half_bit_exact) {
  Rng r(1114);
  int k = 0;
  for (int64_t n : {16, 47, 4096, 70001}) {
    for (int side = 0; side < 3; ++side) {
      for (int bmode = 0; bmode < 2; ++bmode) {
        ++k;
        const int64_t c = 32;
        const int64_t rows = (n + c - 1) / c;
        bftest::ModelBuilder mb;
        mb.input("a", DType::kI8, bmode ? Shape{rows, c} : Shape{n});
        mb.input("b", DType::kI8, bmode ? Shape{c} : Shape{n});
        auto other = r.mult(0.05, 0.49);
        auto mo = k % 4 == 0 ? r.mult(1.0, 3.0) : r.mult(1e-7, 1e-4);
        const bool half_a = side != 1, half_b = side != 0;
        Json at = Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"left_shift", 20},
                               {"a_mult", half_a ? (1 << 30) : other.first}, {"a_shift", half_a ? 0 : other.second},
                               {"b_mult", half_b ? (1 << 30) : other.first}, {"b_shift", half_b ? 0 : other.second},
                               {"out_mult", mo.first}, {"out_shift", mo.second}, {"y_zero", r.i(-128, 127)}};
        if (k % 3 == 0) {
          at["qmin"] = r.i(-128, -10);
          at["qmax"] = r.i(10, 127);
        }
        mb.node("I8Add", {"a", "b"}, {"y"}, at);
        mb.output("y");
        const int64_t na = bmode ? rows * c : n;
        compare(mb, {r.i8(na), r.i8(bmode ? c : n)}, "I8Add half case " + std::to_string(k), "i8add.neon");
      }
    }
  }
}

namespace {
std::vector<int8_t> monotone_rows(Rng& r, int64_t rows, bool decreasing_some) {
  std::vector<int8_t> t(static_cast<size_t>(rows * 256));
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<int> v(256);
    for (auto& x : v) x = r.i(-128, 127);
    std::sort(v.begin(), v.end());
    if (decreasing_some && row % 3 == 1) std::reverse(v.begin(), v.end());
    for (int q = 0; q < 256; ++q) t[row * 256 + q] = static_cast<int8_t>(v[q]);
  }
  return t;
}
}  // namespace

// compose_luts: chains of 2-3 I8Lut nodes (one table, per-channel tables on the last or a middle axis, a final f32
// table), and the cases left alone (inner output read twice, inner output a graph output, per-channel tables on
// different axes).
BF_TEST(int8_full_rewrite_compose_luts_bit_exact) {
  Rng r(2001);
  int k = 0;
  for (int len : {2, 3}) {
    for (int rows_a : {1, 2}) {      // 1: one table, 2: per-channel
      for (int rows_b : {1, 2}) {
        for (int variant = 0; variant < 5; ++variant) {  // 0 plain, 1 middle axis, 2 f32 last table, 3 inner read twice, 4 inner output
          ++k;
          const Shape xs = {2, 5, 7, 19};
          const int64_t ax = variant == 1 ? 1 : 3;
          const int64_t ch = xs[ax];
          bftest::ModelBuilder mb;
          mb.input("x", DType::kI8, xs);
          std::string cur = "x";
          for (int i = 0; i < len; ++i) {
            const int64_t rows = (i % 2 ? rows_b : rows_a) == 1 ? 1 : ch;
            const bool f32 = variant == 2 && i == len - 1;
            const std::string tn = "t" + std::to_string(i), yn = "y" + std::to_string(i);
            if (f32) {
              std::vector<float> tab(static_cast<size_t>(rows * 256));
              for (auto& v : tab) v = r.f(-3.f, 3.f);
              mb.constant(tn, DType::kF32, {rows, 256}, tab.data());
            } else {
              auto tab = r.i8(rows * 256);
              mb.constant(tn, DType::kI8, {rows, 256}, tab.data());
            }
            mb.node("I8Lut", {cur, tn}, {yn}, Json::Object{{"axis", ax}});
            cur = yn;
          }
          mb.output(cur);
          if (variant == 3) {  // a second reader of the first LUT's output
            auto tab = r.i8(256);
            mb.constant("tz", DType::kI8, {1, 256}, tab.data());
            mb.node("I8Lut", {"y0", "tz"}, {"z"});
            mb.output("z");
          }
          if (variant == 4) mb.output("y0");
          const int expect = variant >= 3 ? -1 : 1;
          compare(mb, {r.i8(num_elements(xs))}, "compose_luts case " + std::to_string(k), "", expect);
        }
      }
    }
  }
}

// fold_lut_thresholds: I8Lut -> I8BDense / I8BConv2D binarizing it (ge / gt, scalar / per-channel thresholds incl. ties
// and never / always, monotone one-row and per-channel tables, some decreasing rows (not foldable: left alone), a LUT
// with another reader (kept), padding values -1 / 0 / 1).
BF_TEST(int8_full_rewrite_fold_lut_thresholds_bit_exact) {
  Rng r(2002);
  int k = 0;
  for (const char* mode : {"ge", "gt"}) {
    for (int lrows : {1, 2}) {
      for (int variant = 0; variant < 4; ++variant) {  // 0 BDense, 1 BConv2D, 2 + a second reader, 3 decreasing rows
        ++k;
        const bool conv = variant == 1 || (variant == 2 && k % 2);
        const int64_t c = std::vector<int64_t>{7, 32, 70, 100}[static_cast<size_t>(k % 4)];
        bftest::ModelBuilder mb;
        const Shape xs = conv ? Shape{1, 9, 11, c} : Shape{37, c};
        mb.input("x", DType::kI8, xs);
        const int64_t rows = lrows == 1 ? 1 : c;
        auto tab = monotone_rows(r, rows, variant == 3);
        mb.constant("t", DType::kI8, {rows, 256}, tab.data());
        mb.node("I8Lut", {"x", "t"}, {"y"});
        const float s = r.f(0.01f, 0.1f);
        const int z = r.i(-30, 30);
        std::vector<float> thr(static_cast<size_t>(c));
        for (auto& v : thr) v = r.f(-140 * s, 140 * s);
        for (int64_t i = 0; i < c; i += 3) thr[i] = q8::deq(r.i(-128, 127), s, z);  // ties
        thr[0] = 1e9f;                                                           // never
        if (c > 1) thr[1] = -1e9f;                                               // always
        mb.constant("thr", DType::kF32, {c}, thr.data());
        const int64_t o = 21;
        auto pm = std::vector<float>(o), pa = std::vector<float>(o);
        for (auto& v : pm) v = r.f(-3.f, 3.f);
        for (auto& v : pa) v = r.f(-50.f, 50.f);
        mb.constant("pm", DType::kF32, {o}, pm.data());
        mb.constant("pa", DType::kF32, {o}, pa.data());
        Json at = Json::Object{{"input_binarize", mode}, {"x_scale", static_cast<double>(s)}, {"x_zero", z}};
        if (conv) {
          std::vector<int8_t> wv(static_cast<size_t>(o * 9 * c));
          for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
          auto wp = bftest::pack_pm1(wv, o * 9, c);
          mb.constant("w", DType::kBit, {o, 3, 3, c}, wp.data());
          Json::Array pads;
          for (int i = 0; i < 4; ++i) pads.push_back(Json(1));
          at["pads"] = Json(pads);
          at["pad_value"] = (k % 3) - 1;
          mb.node("I8BConv2D", {"y", "w", "pm", "pa", "thr"}, {"b"}, at);
        } else {
          std::vector<int8_t> wv(static_cast<size_t>(o * c));
          for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
          auto wp = bftest::pack_pm1(wv, o, c);
          mb.constant("w", DType::kBit, {o, c}, wp.data());
          if (lrows == 1 && k % 2) mb.node("I8BDense", {"y", "w", "pm", "pa"}, {"b"}, at);  // scalar threshold 0
          else mb.node("I8BDense", {"y", "w", "pm", "pa", "thr"}, {"b"}, at);
        }
        mb.output("b");
        if (variant == 2) {  // the LUT output is also read by another LUT and is a graph output
          auto t2 = r.i8(256);
          mb.constant("t2", DType::kI8, {1, 256}, t2.data());
          mb.node("I8Lut", {"y", "t2"}, {"z"});
          mb.output("z");
        }
        compare(mb, {r.i8(num_elements(xs))}, "fold_lut_thresholds case " + std::to_string(k), "", variant <= 1 ? 1 : -1);
      }
    }
  }
}

// fold_lut_thresholds, interval form (2026-11-13): V-shaped LUT rows (decreasing then increasing, plateaus, ties at the
// threshold), a Lambda-shaped row (must not fold: two rays), BDense and BConv2D consumers, two consumers with equal
// thresholds sharing one I8PackBits and a third with other thresholds, C not a multiple of 16 / 32 / 64.
std::vector<int8_t> v_rows(Rng& r, int64_t rows, int lambda_row) {
  std::vector<int8_t> t(static_cast<size_t>(rows * 256));
  for (int64_t row = 0; row < rows; ++row) {
    const int m = r.i(-140, 140);  // the minimum's position (outside the codes: monotone rows)
    const int lo = r.i(-128, 20);
    const int sa = r.i(0, 3), sb = r.i(0, 3);
    for (int q = -128; q <= 127; ++q) {
      int v = q < m ? lo + ((m - q) * (sa + 1)) / 4 : lo + ((q - m) * (sb + 1)) / 3;
      if (row == lambda_row) v = -v;
      t[row * 256 + q + 128] = static_cast<int8_t>(std::min(127, std::max(-128, v)));
    }
  }
  return t;
}

BF_TEST(int8_full_rewrite_fold_lut_intervals_bit_exact) {
  Rng r(2010);
  int k = 0;
  for (const char* mode : {"ge", "gt"}) {
    for (int variant = 0; variant < 5; ++variant) {  // 0 BDense, 1 BConv2D, 2 two + one consumers, 3 one table, 4 Lambda row
      ++k;
      const bool conv = variant == 1;
      const int64_t c = std::vector<int64_t>{7, 33, 70, 100, 130}[static_cast<size_t>(k % 5)];
      bftest::ModelBuilder mb;
      const Shape xs = conv ? Shape{1, 9, 11, c} : Shape{37, c};
      mb.input("x", DType::kI8, xs);
      const int64_t rows = variant == 3 ? 1 : c;
      auto tab = v_rows(r, rows, variant == 4 ? static_cast<int>(c / 2) : -1);
      mb.constant("t", DType::kI8, {rows, 256}, tab.data());
      mb.node("I8Lut", {"x", "t"}, {"y"});
      const float s = r.f(0.01f, 0.1f);
      const int z = r.i(-30, 30);
      auto thr_vec = [&]() {
        std::vector<float> thr(static_cast<size_t>(c));
        for (auto& v : thr) v = r.f(-140 * s, 140 * s);
        for (int64_t i = 0; i < c; i += 3) thr[i] = q8::deq(r.i(-128, 127), s, z);  // ties
        thr[0] = 1e9f;                                                           // never
        if (c > 1) thr[1] = -1e9f;                                               // always
        return thr;
      };
      auto thr = thr_vec(), thr2 = thr_vec();
      mb.constant("thr", DType::kF32, {c}, thr.data());
      mb.constant("thr2", DType::kF32, {c}, thr2.data());
      const int64_t o = 21;
      auto pm = std::vector<float>(o), pa = std::vector<float>(o);
      for (auto& v : pm) v = r.f(-3.f, 3.f);
      for (auto& v : pa) v = r.f(-50.f, 50.f);
      mb.constant("pm", DType::kF32, {o}, pm.data());
      mb.constant("pa", DType::kF32, {o}, pa.data());
      Json at = Json::Object{{"input_binarize", mode}, {"x_scale", static_cast<double>(s)}, {"x_zero", z}};
      if (conv) {
        std::vector<int8_t> wv(static_cast<size_t>(o * 9 * c));
        for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
        auto wp = bftest::pack_pm1(wv, o * 9, c);
        mb.constant("w", DType::kBit, {o, 3, 3, c}, wp.data());
        Json::Array pads;
        for (int i = 0; i < 4; ++i) pads.push_back(Json(1));
        at["pads"] = Json(pads);
        at["pad_value"] = (k % 3) - 1;
        mb.node("I8BConv2D", {"y", "w", "pm", "pa", "thr"}, {"b"}, at);
        mb.output("b");
      } else {
        std::vector<int8_t> wv(static_cast<size_t>(o * c));
        for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
        auto wp = bftest::pack_pm1(wv, o, c);
        mb.constant("w", DType::kBit, {o, c}, wp.data());
        mb.node("I8BDense", {"y", "w", "pm", "pa", "thr"}, {"b"}, at);
        mb.output("b");
        if (variant == 2) {
          mb.node("I8BDense", {"y", "w", "pa", "pm", "thr"}, {"b2"}, at);   // same bounds: shares the bits
          mb.node("I8BDense", {"y", "w", "pm", "pa", "thr2"}, {"b3"}, at);  // other bounds
          mb.output("b2");
          mb.output("b3");
        }
      }
      // folded: the LUT becomes an I8PackBits (variant 2: two of them); a Lambda row keeps the LUT
      compare(mb, {r.i8(num_elements(xs))}, "fold_lut_intervals case " + std::to_string(k), variant == 4 ? "" : "i8packbits.neon",
              variant == 2 ? 5 : (variant == 4 ? -1 : 2));
    }
  }
}

// merge_twins: 2-3 I8Lut nodes on one input (one-row and per-channel tables mixed; C < 16 runs the reference twin
// kernel), and I8Add nodes with the same inputs and input parameters (half / scaled inputs, same shapes and a broadcast
// row, clamped outputs, positive out shifts; a third Add with other input parameters stays separate).
BF_TEST(int8_full_rewrite_twins_bit_exact) {
  Rng r(2003);
  int k = 0;
  for (int64_t c : {7, 16, 19, 48}) {
    for (int64_t rows : {1, 15, 64, 65, 200}) {
      for (int nt = 2; nt <= 3; ++nt) {
        ++k;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {rows, c});
        for (int j = 0; j < nt; ++j) {
          const int64_t tr = (j + k) % 3 == 0 ? 1 : c;
          auto tab = r.i8(tr * 256);
          mb.constant("t" + std::to_string(j), DType::kI8, {tr, 256}, tab.data());
          mb.node("I8Lut", {"x", "t" + std::to_string(j)}, {"y" + std::to_string(j)});
          mb.output("y" + std::to_string(j));
        }
        compare(mb, {r.i8(rows * c)}, "lut twins case " + std::to_string(k), c >= 16 ? "i8lutn.neon" : "", 1);
      }
    }
  }
  for (int64_t n : {16, 47, 4096, 70001}) {
    for (int half = 0; half < 3; ++half) {
      for (int bmode = 0; bmode < 2; ++bmode) {
        ++k;
        const int64_t c = 32, rows = (n + c - 1) / c;
        bftest::ModelBuilder mb;
        mb.input("a", DType::kI8, bmode ? Shape{rows, c} : Shape{n});
        mb.input("b", DType::kI8, bmode ? Shape{c} : Shape{n});
        auto ma = r.mult(0.05, 0.49), mbb = r.mult(0.05, 0.49);
        Json base = Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"left_shift", 20},
                                 {"a_mult", half == 1 ? (1 << 30) : ma.first}, {"a_shift", half == 1 ? 0 : ma.second},
                                 {"b_mult", half == 2 ? (1 << 30) : mbb.first}, {"b_shift", half == 2 ? 0 : mbb.second}};
        for (int j = 0; j < 3; ++j) {
          Json at = base;
          auto mo = (j + k) % 4 == 0 ? r.mult(1.0, 3.0) : r.mult(1e-7, 1e-4);
          at["out_mult"] = mo.first;
          at["out_shift"] = mo.second;
          at["y_zero"] = r.i(-128, 127);
          if ((j + k) % 3 == 0) {
            at["qmin"] = r.i(-128, -10);
            at["qmax"] = r.i(10, 127);
          }
          if (j == 2) at["b_zero"] = static_cast<int>(base.get_int("b_zero", 0)) / 2 + 1;  // other input parameters
          mb.node("I8Add", {"a", "b"}, {"y" + std::to_string(j)}, at);
          mb.output("y" + std::to_string(j));
        }
        const int64_t na = bmode ? rows * c : n;
        compare(mb, {r.i8(na), r.i8(bmode ? c : n)}, "add twins case " + std::to_string(k), "i8addn.neon", 2);
      }
    }
  }
}

// I8Gather of int8 tables (i8gather.neon, 2026-11-13): row lengths below / at / above the 8- and 16-byte moves, repeated
// and out-of-range indices (fill_q).
BF_TEST(int8_full_gather_rows_bit_exact) {
  Rng r(2012);
  for (int64_t d : {1, 5, 8, 13, 16, 31, 64, 100}) {
    const int64_t v = 37, n = 53;
    bftest::ModelBuilder mb;
    mb.input("t", DType::kI8, {v, d});
    std::vector<int32_t> idx(static_cast<size_t>(n));
    for (auto& x : idx) x = r.i(-3, static_cast<int>(v) + 2);
    mb.constant("idx", DType::kI32, {n}, idx.data());
    mb.node("I8Gather", {"t", "idx"}, {"y"}, Json::Object{{"fill_q", r.i(-128, 127)}});
    mb.output("y");
    compare(mb, {r.i8(v * d)}, "gather d " + std::to_string(d), "i8gather.neon");
  }
}

// merge_twins, I8Conv2D: 2-3 convolutions of one input with byte-identical weights / bias and the same geometry (stem
// 7x7 stride 2 with the SMMLA run layout, 3x3 padded, 1x1 in place, 5x5 stride 2), differing requantizations
// (negative / positive shifts, clamps), with and without bias, channel tails; a convolution with other weights on the
// same input stays separate.
BF_TEST(int8_full_rewrite_conv_twins_bit_exact) {
  Rng r(2004);
  struct Geo {
    int64_t h, w, ic, oc, k, s, p;
  };
  const Geo geos[] = {{23, 21, 3, 32, 7, 2, 3}, {9, 11, 16, 24, 3, 1, 1}, {8, 8, 32, 19, 1, 1, 0}, {13, 10, 5, 40, 5, 2, 2}};
  int k = 0;
  for (const Geo& g : geos) {
    for (int nt = 2; nt <= 3; ++nt) {
      for (int bias = 0; bias < 2; ++bias) {
        ++k;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {1, g.h, g.w, g.ic});
        auto wv = r.i8(g.oc * g.k * g.k * g.ic);
        mb.constant("w", DType::kI8, {g.oc, g.k, g.k, g.ic}, wv.data());
        mb.constant("w2", DType::kI8, {g.oc, g.k, g.k, g.ic}, wv.data());  // byte-identical copy (as the quantizer emits)
        auto wo = r.i8(g.oc * g.k * g.k * g.ic);
        mb.constant("wo", DType::kI8, {g.oc, g.k, g.k, g.ic}, wo.data());
        std::vector<int32_t> bv(static_cast<size_t>(g.oc));
        for (auto& v : bv) v = r.i(-100000, 100000);
        mb.constant("b", DType::kI32, {g.oc}, bv.data());
        mb.constant("b2", DType::kI32, {g.oc}, bv.data());
        const int xz = r.i(-128, 127);
        Json::Array pads;
        for (int i = 0; i < 4; ++i) pads.push_back(Json(g.p));
        Json::Array strides = {Json(g.s), Json(g.s)};
        for (int j = 0; j <= nt; ++j) {  // j == nt: other weights
          std::vector<int32_t> mv(static_cast<size_t>(g.oc)), sv(static_cast<size_t>(g.oc));
          for (int64_t o = 0; o < g.oc; ++o) {
            auto m = (j + k + static_cast<int>(o)) % 7 == 0 ? r.mult(1.5, 4.0) : r.mult(1e-6, 1e-3);
            mv[o] = m.first;
            sv[o] = m.second;
          }
          const std::string js = std::to_string(j);
          mb.constant("m" + js, DType::kI32, {g.oc}, mv.data());
          mb.constant("s" + js, DType::kI32, {g.oc}, sv.data());
          Json at = Json::Object{{"x_zero", xz}, {"pad_q", xz}, {"pads", Json(pads)}, {"strides", Json(strides)}, {"y_zero", r.i(-128, 127)}};
          if ((j + k) % 3 == 0) {
            at["qmin"] = r.i(-128, -20);
            at["qmax"] = r.i(0, 127);
          }
          const std::string wn = j == nt ? "wo" : (j % 2 ? "w2" : "w"), bn = bias ? (j % 2 ? "b2" : "b") : "";
          mb.node("I8Conv2D", {"x", wn, bn, "m" + js, "s" + js}, {"y" + js}, at);
          mb.output("y" + js);
        }
        compare(mb, {r.i8(g.h * g.w * g.ic)}, "conv twins case " + std::to_string(k), "i8conv2dn.neon_sdot", 2);
      }
    }
  }
}

// Depthwise I8Conv2D (groups = C), int8 output: SMLAL accumulation and the vector requantization of DepthwiseNeon
// (3x3 / 5x5, strides 1 / 2, dilation 2, padding with pad_q != x_zero, with / without bias, positive and negative
// shifts, clamps, C multiple of 16 and with tails, extreme weights and codes).
BF_TEST(int8_full_depthwise_bit_exact) {
  Rng r(2005);
  int k = 0;
  // (c, channel multiplier): groups = c / mult input channels (QuickNet-XL: 8 inputs, multiplier 8)
  for (std::pair<int64_t, int64_t> cm : std::vector<std::pair<int64_t, int64_t>>{{16, 1}, {64, 1}, {21, 1}, {64, 8}, {48, 3}, {40, 5}}) {
    const int64_t c = cm.first, mult = cm.second, cin = c / mult;
    for (int64_t ks : {3, 5}) {
      for (int s = 1; s <= 2; ++s) {
        for (int variant = 0; variant < 3; ++variant) {  // 0 plain, 1 dilation 2 + no bias, 2 positive shifts + clamp
          ++k;
          const int64_t h = 11, w = 9, pad = ks / 2 * (variant == 1 ? 2 : 1);
          bftest::ModelBuilder mb;
          mb.input("x", DType::kI8, {1, h, w, cin});
          auto wv = r.i8(c * ks * ks);
          wv[0] = 0x80;
          wv[1] = 0x7F;
          mb.constant("w", DType::kI8, {c, ks, ks, 1}, wv.data());
          std::vector<int32_t> bv(static_cast<size_t>(c)), mv(static_cast<size_t>(c)), sv(static_cast<size_t>(c));
          for (auto& v : bv) v = r.i(-200000, 200000);
          for (int64_t o = 0; o < c; ++o) {
            auto m = variant == 2 && o % 3 == 0 ? r.mult(1.5, 40.0) : r.mult(1e-5, 1e-2);
            mv[o] = m.first;
            sv[o] = m.second;
          }
          mb.constant("b", DType::kI32, {c}, bv.data());
          mb.constant("m", DType::kI32, {c}, mv.data());
          mb.constant("s", DType::kI32, {c}, sv.data());
          Json::Array pads, strides = {Json(s), Json(s)}, dil = {Json(variant == 1 ? 2 : 1), Json(variant == 1 ? 2 : 1)};
          for (int i = 0; i < 4; ++i) pads.push_back(Json(pad));
          const int xz = r.i(-128, 127);
          Json at = Json::Object{{"x_zero", xz}, {"pad_q", r.i(-128, 127)}, {"pads", Json(pads)}, {"strides", Json(strides)},
                                 {"dilations", Json(dil)}, {"groups", cin}, {"y_zero", r.i(-128, 127)}};
          if (variant == 2) {
            at["qmin"] = r.i(-128, -30);
            at["qmax"] = r.i(10, 127);
          }
          mb.node("I8Conv2D", {"x", "w", variant == 1 ? "" : "b", "m", "s"}, {"y"}, at);
          mb.output("y");
          auto x = r.i8(h * w * cin);
          x[0] = 0x80;
          x[1] = 0x7F;
          compare(mb, {x}, "depthwise case " + std::to_string(k), "i8conv2d.neon_depthwise");
        }
      }
    }
  }
}

// collapse_unary: I8Mul(x, Lut(x)) (the GELU-mask pattern), I8Add(Lut1(x), Lut2(x)) with a per-channel table,
// I8AffineSum(x, Lut(x), x) with per-channel and scalar coefficients, clamped outputs; an inner LUT with another reader
// (kept); operands of two different tensors (not collapsed).
BF_TEST(int8_full_rewrite_collapse_unary_bit_exact) {
  Rng r(2006);
  int k = 0;
  for (int64_t c : {7, 32, 50}) {
    for (int variant = 0; variant < 5; ++variant) {
      ++k;
      const int64_t rows = 13;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {rows, c});
      mb.input("x2", DType::kI8, {rows, c});
      auto t1 = r.i8(256), t2 = r.i8(c * 256);
      mb.constant("t1", DType::kI8, {1, 256}, t1.data());
      mb.constant("t2", DType::kI8, {c, 256}, t2.data());
      mb.node("I8Lut", {"x", "t1"}, {"l1"});
      mb.node("I8Lut", {"x", "t2"}, {"l2"});
      int expect = 1;
      if (variant == 0 || variant == 4) {  // Mul(x, Lut(x)); variant 4: Mul(x2, Lut(x)) (two roots)
        auto mm = r.mult(1e-3, 2.0);
        Json at = Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"mult", mm.first}, {"shift", mm.second},
                               {"y_zero", r.i(-128, 127)}};
        if (k % 2) {
          at["qmin"] = -100;
          at["qmax"] = 90;
        }
        mb.node("I8Mul", {variant == 4 ? "x2" : "x", "l1"}, {"y"}, at);
        expect = variant == 4 ? -1 : 1;
      } else if (variant == 1) {  // Add(Lut1(x), Lut2(x))
        auto ma = r.mult(0.05, 0.49), mo = r.mult(1e-7, 1e-4);
        Json at = Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"left_shift", 20}, {"a_mult", ma.first},
                               {"a_shift", ma.second}, {"b_mult", 1 << 30}, {"b_shift", 0}, {"out_mult", mo.first},
                               {"out_shift", mo.second}, {"y_zero", r.i(-128, 127)}};
        mb.node("I8Add", {"l1", "l2"}, {"y"}, at);
      } else {  // AffineSum(x, Lut2(x), x) (variant 3: scalar coefficients, and l2 also a graph output)
        const int64_t kc = variant == 3 ? 1 : c;
        std::vector<int32_t> K(static_cast<size_t>(3 * kc)), B(static_cast<size_t>(kc));
        for (auto& v : K) v = r.i(-(1 << 18), 1 << 18);
        for (auto& v : B) v = r.i(-(1 << 24), 1 << 24);
        mb.constant("K", DType::kI32, {3, kc}, K.data());
        mb.constant("B", DType::kI32, {kc}, B.data());
        Json::Array zs = {Json(r.i(-128, 127)), Json(r.i(-128, 127)), Json(r.i(-128, 127))};
        mb.node("I8AffineSum", {"x", "l2", "x", "K", "B"}, {"y"},
                Json::Object{{"zeros", Json(zs)}, {"mult", 1 << 30}, {"shift", -17}, {"y_zero", r.i(-128, 127)}});
        if (variant == 3) {
          mb.output("l2");
          expect = 2;
        }
      }
      mb.output("y");
      compare(mb, {r.i8(rows * c), r.i8(rows * c)}, "collapse_unary case " + std::to_string(k), "", expect);
    }
  }
}

// I8LayerNorm: the SMLAL row form (full output range) and the clamped form, row lengths with and without 16-element
// tails, extreme zero points and codes, tiny / zero / large gains, constant rows.
// merge_twins, I8LayerNorm (2026-11-13): 2-3 LayerNorms of one input with the same statistics attributes (-> I8LayerNormN),
// full-range and clamped outputs mixed, d with and without 16-element tails, a constant row; a LayerNorm with another
// eps stays separate.
BF_TEST(int8_full_rewrite_layernorm_twins_bit_exact) {
  Rng r(2013);
  int k = 0;
  for (int64_t d : {16, 48, 200, 768}) {
    for (int nt = 2; nt <= 3; ++nt) {
      ++k;
      const int64_t rows = 1 + k % 7;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {rows, d});
      const Json base = Json::Object{{"x_zero", r.i(-128, 127)}, {"x_scale", static_cast<double>(r.f(0.001f, 3.f))}, {"eps", 1e-5}};
      for (int j = 0; j <= nt; ++j) {
        std::vector<float> g(static_cast<size_t>(d)), b(static_cast<size_t>(d));
        for (auto& v : g) v = r.f(-80.f, 80.f);
        for (auto& v : b) v = r.f(-90.f, 90.f);
        mb.constant("g" + std::to_string(j), DType::kF32, {d}, g.data());
        mb.constant("b" + std::to_string(j), DType::kF32, {d}, b.data());
        Json at = base;
        if ((j + k) % 3 == 1) {
          at["qmin"] = r.i(-128, -10);
          at["qmax"] = r.i(10, 127);
        }
        if (j == nt) at["eps"] = 1e-6;  // other statistics: not merged
        mb.node("I8LayerNorm", {"x", "g" + std::to_string(j), "b" + std::to_string(j)}, {"y" + std::to_string(j)}, at);
        mb.output("y" + std::to_string(j));
      }
      auto x = r.i8(rows * d);
      x[0] = 0x80;
      x[1] = 0x7F;
      if (rows > 2) std::fill_n(x.begin() + d, d, x[d]);
      compare(mb, {x}, "layernorm twins case " + std::to_string(k), "i8layernormn.neon", 2);
    }
  }
}

BF_TEST(int8_full_layernorm_bit_exact) {
  Rng r(2007);
  int k = 0;
  for (int64_t d : {16, 48, 200, 768, 1000}) {
    for (int variant = 0; variant < 3; ++variant) {  // 0 full range, 1 clamped, 2 tiny / zero gains
      ++k;
      const int64_t rows = 1 + k % 9;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {rows, d});
      std::vector<float> g(static_cast<size_t>(d)), b(static_cast<size_t>(d));
      for (auto& v : g) v = r.f(-80.f, 80.f);
      for (auto& v : b) v = r.f(-90.f, 90.f);
      if (variant == 2)
        for (int64_t i = 0; i < d; i += 3) g[i] = i % 2 ? 0.f : 1e-30f;
      mb.constant("g", DType::kF32, {d}, g.data());
      mb.constant("b", DType::kF32, {d}, b.data());
      Json at = Json::Object{{"x_zero", k % 2 ? -128 : 127}, {"x_scale", static_cast<double>(r.f(0.001f, 3.f))}, {"eps", 1e-5}};
      if (variant == 1) {
        at["qmin"] = r.i(-128, -10);
        at["qmax"] = r.i(10, 127);
      }
      mb.node("I8LayerNorm", {"x", "g", "b"}, {"y"}, at);
      mb.output("y");
      auto x = r.i8(rows * d);
      x[0] = 0x80;
      x[1] = 0x7F;
      if (rows > 2) std::fill_n(x.begin() + d, d, x[d]);  // a constant row
      compare(mb, {x}, "layernorm case " + std::to_string(k), "i8layernorm.neon");
    }
  }
}

// AArch32 int8 GEMM tiles (a32gemm::tile4x2: VMULL.S8 / VMLAL.S8 / VPADAL.S16, kernels/int8_neon/int8_gemm_a32.h)
// behind i8conv2d / i8dense / i8conv2dn.neon_a32_mull, bit-exact with the reference: row counts that leave 1..7 rows
// in the last 8-row tile (partial quads), odd channel counts (a zero column in the last pair), K with and without a
// 16-byte tail, in-place 1x1 rows, padding taps with pad_q != x_zero, groups (kic 2 and 24), extreme codes and
// weights +-127, i8 (negative / positive shifts, clamps), f32, l8 and bit outputs, Dense with 1..13 rows, twin
// convolutions (I8Conv2DN, two outputs). Weights containing -128 must fall back to the SMLAL kernel (neon_a32), also
// exact. On AArch32 the expected kernel must be selected; elsewhere the cases run on that target's kernels.
namespace {
void expect_a32_kernel(const bftest::ModelBuilder& mb, const std::vector<std::vector<uint8_t>>& in, const std::string& kernel,
                       const std::string& what) {
#if defined(__arm__) && defined(__ARM_NEON) && !defined(__aarch64__)
  RunInfo info;
  run_all(mb, kFull, 1, true, in, &info);
  bool found = false;
  for (const auto& kv : info.kernels) found = found || kv.second == kernel;
  if (!found) throw bftest::Failure{"expected kernel " + kernel + " for " + what};
#else
  (void)mb, (void)in, (void)kernel, (void)what;
#endif
}
std::vector<uint8_t> weights_127(Rng& r, int64_t n, bool extremes) {
  auto w = r.i8(n);
  for (size_t i = 0; i < w.size(); ++i) {
    int v = static_cast<int8_t>(w[i]);
    if (v == -128) v = -127;
    if (extremes && i % 5 == 0) v = i % 2 ? 127 : -127;
    w[i] = static_cast<uint8_t>(v);
  }
  return w;
}
}  // namespace

BF_TEST(int8_full_gemm_a32_mull_bit_exact) {
  Rng r(3232);
  struct C {
    int64_t n, h, w, ic, oc, k, s, pad, groups;
    const char* out;
  };
  int idx = 0;
  for (C c : std::vector<C>{{1, 13, 11, 3, 32, 7, 2, 3, 1, "i8"},    // stem: K 147 (tail), 7x6 = 42 rows (2 in the last tile)
                            {1, 9, 7, 16, 21, 3, 1, 1, 1, "i8"},     // odd channels, K 144
                            {2, 5, 5, 64, 35, 1, 1, 0, 1, "i8"},     // 1x1 in place, 50 rows
                            {1, 6, 5, 24, 16, 1, 2, 0, 1, "i8"},     // 1x1 stride 2, K 24 (tail), 9 rows
                            {1, 7, 6, 8, 32, 3, 1, 1, 4, "i8"},      // groups 4 (kic 2, ocg 8)
                            {1, 5, 7, 48, 18, 3, 2, 1, 2, "i8"},     // groups 2 (kic 24, ocg 9: a zero column per group)
                            {1, 11, 9, 32, 37, 3, 1, 1, 1, "f32"},   // f32 output
                            {1, 9, 10, 40, 64, 3, 2, 1, 1, "l8"},    // l8 output
                            {1, 3, 1, 16, 7, 1, 1, 0, 1, "i8"},      // 3 rows: one partial quad only
                            {1, 4, 5, 128, 70, 3, 1, 1, 1, "i8"}}) { // K 1152 (> 1032: the int32 bound check)
    for (int variant = 0; variant < 2; ++variant) {
      ++idx;
      const int64_t kic = c.ic / c.groups;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {c.n, c.h, c.w, c.ic});
      auto wv = weights_127(r, c.oc * c.k * c.k * kic, variant == 1);
      mb.constant("w", DType::kI8, {c.oc, c.k, c.k, kic}, wv.data());
      const int xz = variant ? (idx % 2 ? -128 : 127) : r.i(-128, 127);
      Json at = Json::Object{{"strides", zeros_json({c.s, c.s})}, {"pads", zeros_json({c.pad, c.pad, c.pad, c.pad})}, {"groups", c.groups},
                             {"x_zero", xz}, {"pad_q", variant ? r.i(-128, 127) : xz}};
      std::vector<std::string> ins = {"x", "w", "b"};
      const std::string out = c.out;
      if (out == "i8") {
        std::vector<int32_t> bq(static_cast<size_t>(c.oc)), m(static_cast<size_t>(c.oc)), sh(static_cast<size_t>(c.oc));
        for (int64_t o = 0; o < c.oc; ++o) {
          bq[o] = r.i(-200000, 200000);
          auto ms = (o + idx) % 5 == 0 ? r.mult(1.5, 6.0) : r.mult(1e-6, 1e-3);
          m[o] = ms.first;
          sh[o] = ms.second;
        }
        mb.constant("b", DType::kI32, {c.oc}, bq.data());
        mb.constant("m", DType::kI32, {c.oc}, m.data());
        mb.constant("sh", DType::kI32, {c.oc}, sh.data());
        ins.insert(ins.end(), {"m", "sh"});
        at["y_zero"] = r.i(-100, 100);
        if (variant) {
          at["qmin"] = r.i(-128, -30);
          at["qmax"] = r.i(20, 127);
        }
      } else {
        std::vector<float> sc(static_cast<size_t>(c.oc)), bi(static_cast<size_t>(c.oc));
        for (int64_t o = 0; o < c.oc; ++o) {
          sc[o] = std::ldexp(r.f(0.5f, 1.f), r.i(-22, -10));
          bi[o] = r.f(-1.f, 1.f) * std::ldexp(1.f, r.i(-8, 2));
        }
        mb.constant("b", DType::kF32, {c.oc}, bi.data());
        mb.constant("s", DType::kF32, {c.oc}, sc.data());
        ins.push_back("s");
        at["output"] = out;
        if (out == "l8") at["y_exp"] = 0;
      }
      std::vector<std::string> outs = {"y"};
      if (out == "i8" && variant == 1) {  // + bit output (thresholds on acc)
        std::vector<int32_t> bt(static_cast<size_t>(c.oc));
        for (auto& v : bt) v = r.i(-3000, 3000);
        mb.constant("bt", DType::kI32, {c.oc}, bt.data());
        ins.push_back("bt");
        outs.push_back("yb");
      }
      mb.node("I8Conv2D", ins, outs, at);
      for (const auto& o : outs) mb.output(o);
      const auto x = r.i8(c.n * c.h * c.w * c.ic);
      compare(mb, {x}, "a32 conv case " + std::to_string(idx));
      expect_a32_kernel(mb, {x}, "i8conv2d.neon_a32_mull", "a32 conv case " + std::to_string(idx));
    }
  }
  // Dense: 1..13 rows, K with / without tails, 1..37 outputs, i8 and f32 outputs
  for (int64_t rows : {1, 2, 3, 5, 8, 9, 13}) {
    for (int64_t k : {16, 100, 1040}) {
      for (int64_t o : {1, 2, 37}) {
        ++idx;
        const bool f32 = idx % 3 == 0;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {rows, k});
        auto wv = weights_127(r, o * k, idx % 2 == 0);
        mb.constant("w", DType::kI8, {o, k}, wv.data());
        Json at = Json::Object{{"x_zero", r.i(-128, 127)}};
        std::vector<std::string> ins = {"x", "w", "b", "m"};
        if (f32) {
          std::vector<float> b(static_cast<size_t>(o)), sc(static_cast<size_t>(o));
          for (int64_t j = 0; j < o; ++j) b[j] = r.f(-1.f, 1.f), sc[j] = r.f(1e-4f, 1e-2f);
          mb.constant("b", DType::kF32, {o}, b.data());
          mb.constant("m", DType::kF32, {o}, sc.data());
          at["output"] = "f32";
        } else {
          std::vector<int32_t> b(static_cast<size_t>(o)), m(static_cast<size_t>(o)), sh(static_cast<size_t>(o));
          for (int64_t j = 0; j < o; ++j) {
            b[j] = r.i(-1000000, 1000000);
            auto mm = r.mult(1e-6, 1e-3);
            m[j] = mm.first;
            sh[j] = mm.second;
          }
          mb.constant("b", DType::kI32, {o}, b.data());
          mb.constant("m", DType::kI32, {o}, m.data());
          mb.constant("s", DType::kI32, {o}, sh.data());
          ins.push_back("s");
          at["y_zero"] = r.i(-128, 127);
        }
        mb.node("I8Dense", ins, {"y"}, at);
        mb.output("y");
        const auto x = r.i8(rows * k);
        compare(mb, {x}, "a32 dense case " + std::to_string(idx));
        expect_a32_kernel(mb, {x}, "i8dense.neon_a32_mull", "a32 dense case " + std::to_string(idx));
      }
    }
  }
  // twin convolutions (merge_twins -> I8Conv2DN, two outputs) and the -128 fallback
  for (int fallback = 0; fallback < 2; ++fallback) {
    for (int64_t oc : {32, 19}) {
      ++idx;
      const int64_t h = 15, w = 13, ic = 3, kk = 7;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {1, h, w, ic});
      auto wv = weights_127(r, oc * kk * kk * ic, true);
      if (fallback) wv[static_cast<size_t>(idx) % wv.size()] = 0x80;  // one weight -128
      mb.constant("w", DType::kI8, {oc, kk, kk, ic}, wv.data());
      mb.constant("w2", DType::kI8, {oc, kk, kk, ic}, wv.data());
      std::vector<int32_t> bv(static_cast<size_t>(oc));
      for (auto& v : bv) v = r.i(-100000, 100000);
      mb.constant("b", DType::kI32, {oc}, bv.data());
      const int xz = r.i(-128, 127);
      for (int j = 0; j < 2; ++j) {
        std::vector<int32_t> mv(static_cast<size_t>(oc)), sv(static_cast<size_t>(oc));
        for (int64_t o = 0; o < oc; ++o) {
          auto m = (j + o) % 4 == 0 ? r.mult(1.5, 4.0) : r.mult(1e-6, 1e-3);
          mv[o] = m.first;
          sv[o] = m.second;
        }
        const std::string js = std::to_string(j);
        mb.constant("m" + js, DType::kI32, {oc}, mv.data());
        mb.constant("s" + js, DType::kI32, {oc}, sv.data());
        Json at = Json::Object{{"x_zero", xz}, {"pad_q", xz}, {"pads", zeros_json({3, 3, 3, 3})}, {"strides", zeros_json({2, 2})},
                               {"y_zero", r.i(-128, 127)}};
        if (j == 1) at["qmin"] = r.i(-128, -20);
        mb.node("I8Conv2D", {"x", j ? "w2" : "w", "b", "m" + js, "s" + js}, {"y" + js}, at);
        mb.output("y" + js);
      }
      const auto x = r.i8(h * w * ic);
      compare(mb, {x}, "a32 conv twins case " + std::to_string(idx), "", 1);
      expect_a32_kernel(mb, {x}, fallback ? "i8conv2dn.neon_a32" : "i8conv2dn.neon_a32_mull", "a32 conv twins case " + std::to_string(idx));
    }
  }
}

// I8Quantize on AArch32 (QuantizeNeon::run_a32: multiply by the reciprocal, IEEE division near half-integers and for
// denormal inputs): exact .5 ties of x / s and their float neighbours over the whole int8 range and beyond, values
// that clamp, +-0, denormals, zero points -256..256 (and one beyond: the generic loop), every tail length.
BF_TEST(int8_full_quantize_reciprocal_bit_exact) {
  Rng r(4321);
  int k = 0;
  for (float s : {0.0137f, 1.f, 3.25f, 0.5f, 7.3e-3f, 1e-30f, 0x1p-126f}) {
    for (int z : {0, -20, 117, -256, 256, 300}) {
      ++k;
      std::vector<float> x;
      for (int j = -420; j <= 420; ++j) {
        const float tie = (static_cast<float>(j) + 0.5f) * s;
        x.push_back(tie);
        x.push_back(std::nextafter(tie, 1e30f));
        x.push_back(std::nextafter(tie, -1e30f));
        x.push_back(static_cast<float>(j) * s);
      }
      for (float v : {0.f, -0.f, 1e-45f, -2e-41f, 9e-39f, 1e20f, -1e20f, 600.f * s, -1000.f * s}) x.push_back(v);
      for (int i = 0; i < 500; ++i) x.push_back(r.f(-300.f, 300.f) * s);
      for (int64_t n : {int64_t(1), int64_t(15), int64_t(16), int64_t(17), static_cast<int64_t>(x.size())}) {
        bftest::ModelBuilder mb;
        mb.input("x", DType::kF32, {n});
        mb.node("I8Quantize", {"x"}, {"y"}, Json::Object{{"scale", static_cast<double>(s)}, {"zero", z}});
        mb.output("y");
        std::vector<uint8_t> xb(static_cast<size_t>(n) * 4);
        std::memcpy(xb.data(), x.data(), xb.size());
        compare(mb, {xb}, "quantize case " + std::to_string(k) + " n " + std::to_string(n), "i8quantize.neon");
      }
    }
  }
}

// I8MatMul on AArch32 (MatMulA32: VMULL.S8 tiles, -128 codes of b packed as -127 and restored): operands dense in
// -128 (every pair a*b = (-128)(-128) appears), row counts with partial quads, odd column counts, K with and without
// 16-byte tails up to the 1024 bound, transposed and plain b, zero points beyond 256 (scalar epilogue), f32 output.
BF_TEST(int8_full_matmul_a32_bit_exact) {
  Rng r(5150);
  int k = 0;
  struct S {
    int64_t bt, m, kk, n;
  };
  for (S sh : std::vector<S>{{3, 7, 49, 49}, {2, 13, 64, 17}, {1, 5, 1024, 3}, {4, 1, 30, 1}, {2, 33, 100, 40}}) {
    for (int tb = 0; tb < 2; ++tb) {
      for (int variant = 0; variant < 3; ++variant) {  // 0 i8, 1 i8 with |z| > 256, 2 f32
        ++k;
        bftest::ModelBuilder mb;
        mb.input("a", DType::kI8, {sh.bt, sh.m, sh.kk});
        mb.input("b", DType::kI8, tb ? Shape{sh.bt, sh.n, sh.kk} : Shape{sh.bt, sh.kk, sh.n});
        Json at = Json::Object{{"a_zero", variant == 1 ? 300 : r.i(-128, 127)}, {"b_zero", variant == 1 ? -290 : r.i(-128, 127)},
                               {"transpose_b", tb}};
        if (variant == 2) {
          at["output"] = "f32";
          at["scale"] = 0.0007;
        } else {
          auto mm = r.mult(1e-6, 1e-3);
          at["mult"] = mm.first;
          at["shift"] = mm.second;
          at["y_zero"] = r.i(-100, 100);
          if (k % 2) at["qmin"] = -20;
        }
        mb.node("I8MatMul", {"a", "b"}, {"y"}, at);
        mb.output("y");
        auto a = r.i8(sh.bt * sh.m * sh.kk), b = r.i8(sh.bt * sh.kk * sh.n);
        for (size_t i = 0; i < a.size(); i += 2) a[i] = 0x80;
        for (size_t i = 0; i < b.size(); ++i)
          if (r.i(0, 1)) b[i] = 0x80;
        compare(mb, {a, b}, "a32 matmul case " + std::to_string(k));
        expect_a32_kernel(mb, {a, b}, "i8matmul.neon_a32", "a32 matmul case " + std::to_string(k));
      }
    }
  }
}
