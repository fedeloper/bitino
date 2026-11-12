// INT8 Conv2D / Dense tests (contract: runtime/kernels/int8_quant.h).
//  (a) every platform-SIMD int8 kernel (AArch64 int8_neon: SDOT, I8MM;
//      x86-64 int8_avx2: VPMADDWD, AVX-VNNI VPDPBUSD) is BIT-EXACT with
//      int8_reference over many shapes, paddings and thread counts, and the
//      AVX2 activation quantizer is bit-exact with the scalar one;
//  (b) int8_reference stays within an analytic error bound of the float
//      reference (derivation at int8_error_bound below and in docs/int8.md);
//  (c) kernel selection follows CPU features (BITINO_DISABLE_FEATURES is
//      read once per process, so masked runs re-execute this binary) and the
//      int8 tiers are opt-in.
#include <cstdlib>
#include <cstring>
#include <set>
#include <random>

#include "bitino/bitino.h"
#include "kernels/int8_quant.h"
#if defined(__x86_64__)
#include "kernels/int8_avx2/int8_avx2.h"
#endif
#include "tools/model_builder.h"
#include "testing.h"

using namespace bitino;

namespace {

Json ints(std::initializer_list<int64_t> v) {
  Json::Array a;
  for (auto x : v) a.push_back(Json(x));
  return Json(a);
}

struct ConvCase {
  int64_t n, h, w, ic, oc, k, stride;
  int64_t pt, pb, pl, pr;
  int64_t dil;
  float pad_value;
  int64_t groups;
  const char* act;
  bool bias;
  int input;  // 0 normal, 1 normal with exact zeros, 2 all zero, 3 relu-like (>= 0)
};

struct ConvModel {
  std::shared_ptr<Model> model;
  std::vector<float> w, b;
};

ConvModel make_conv(const ConvCase& c, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  ConvModel m;
  int64_t kic = c.ic / c.groups;
  m.w.resize(static_cast<size_t>(c.oc * c.k * c.k * kic));
  for (auto& v : m.w) v = nd(rng) * 0.2f;
  if (c.oc > 1) std::fill_n(m.w.begin(), c.k * c.k * kic, 0.f);  // all-zero filter row: scale_w = 1
  m.b.resize(static_cast<size_t>(c.oc));
  for (auto& v : m.b) v = nd(rng);
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kF32, {c.oc, c.k, c.k, kic}, m.w.data());
  if (c.bias) mb.constant("b", DType::kF32, {c.oc}, m.b.data());
  mb.input("x", DType::kF32, {c.n, c.h, c.w, c.ic});
  Json attrs = Json::Object{{"strides", ints({c.stride, c.stride})}, {"dilations", ints({c.dil, c.dil})},
                            {"pads", ints({c.pt, c.pb, c.pl, c.pr})}, {"pad_value", Json(static_cast<double>(c.pad_value))},
                            {"groups", Json(c.groups)}, {"activation", c.act}};
  mb.node("Conv2D", {"x", "w", c.bias ? "b" : ""}, {"y"}, attrs);
  mb.output("y");
  m.model = mb.build();
  if (!c.bias) std::fill(m.b.begin(), m.b.end(), 0.f);
  return m;
}

struct DenseCase {
  Shape x;  // last dim = K
  int64_t o;
  const char* act;
  bool bias;
  int input;
};

ConvModel make_dense(const DenseCase& c, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  ConvModel m;
  int64_t k = c.x.back();
  m.w.resize(static_cast<size_t>(c.o * k));
  for (auto& v : m.w) v = nd(rng) * 0.1f;
  m.b.resize(static_cast<size_t>(c.o));
  for (auto& v : m.b) v = nd(rng);
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kF32, {c.o, k}, m.w.data());
  if (c.bias) mb.constant("b", DType::kF32, {c.o}, m.b.data());
  mb.input("x", DType::kF32, c.x);
  mb.node("Dense", {"x", "w", c.bias ? "b" : ""}, {"y"}, Json(Json::Object{{"activation", c.act}}));
  mb.output("y");
  m.model = mb.build();
  if (!c.bias) std::fill(m.b.begin(), m.b.end(), 0.f);
  return m;
}

std::vector<float> make_input(int64_t size, int kind, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    float v = nd(rng) * 1.5f;
    if (kind == 1 && i % 5 == 0) v = 0.f;
    if (kind == 2) v = 0.f;
    if (kind == 3) v = std::fabs(v);
    x[i] = v;
  }
  return x;
}

struct RunResult {
  std::vector<float> y;
  std::string kernel;
};

RunResult run_model(const std::shared_ptr<Model>& m, const KernelPolicy& policy, int threads, const std::vector<float>& x) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = policy;
  Session s(m, o);
  std::memcpy(s.input(0).raw(), x.data(), x.size() * 4);
  s.run();
  RunResult r;
  r.kernel = s.kernel_choices().begin()->second;
  const Tensor& y = s.output(0);
  r.y.assign(y.data<float>(), y.data<float>() + y.size());
  return r;
}

KernelPolicy forced(const std::string& op, const std::string& kernel) {
  KernelPolicy p;
  p.force[op] = kernel;
  return p;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

std::string first_diff(const std::vector<float>& a, const std::vector<float>& b) {
  for (size_t i = 0; i < a.size() && i < b.size(); ++i)
    if (std::memcmp(&a[i], &b[i], 4) != 0)
      return "index " + std::to_string(i) + ": " + std::to_string(a[i]) + " vs " + std::to_string(b[i]);
  return "size " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
}

// The platform-SIMD int8 kernels of an op (int8_neon on AArch64, int8_avx2
// on x86-64) and whether this CPU must be able to run them (a kernel the CPU
// supports may never be silently skipped).
struct SimdKernel {
  std::string name;
  bool required;
};

std::vector<SimdKernel> simd_kernels(const std::string& prefix) {
  const CpuInfo& cpu = cpu_info();
#if defined(__aarch64__)
  return {{prefix + ".int8_neon_sdot", cpu.has(kFeatNeon | kFeatDotProd)},
          {prefix + ".int8_neon_i8mm", cpu.has(kFeatNeon | kFeatDotProd | kFeatI8mm)}};
#elif defined(__x86_64__)
  return {{prefix + ".int8_avx2", cpu.has(kFeatAvx2)}, {prefix + ".int8_avx2_vnni", cpu.has(kFeatAvx2 | kFeatAvxVnni)}};
#elif defined(__arm__) && defined(__ARM_NEON)
  return {{prefix + ".int8_neon_a32", cpu.has(kFeatNeon)}};  // AArch32: VMULL.S8 tiles (int8_neon_a32.cc)
#else
  (void)cpu;
  return {};
#endif
}

// Policies used by the selection checks: every int8 tier.
const std::set<std::string> kSimdTiers = {"int8_neon", "int8_avx2"};
KernelPolicy tiers_with_simd(std::set<std::string> t) {
  t.insert(kSimdTiers.begin(), kSimdTiers.end());
  return KernelPolicy::tiers(t);
}

int g_simd_comparisons = 0;

// Runs `op` with every platform-SIMD int8 kernel at several thread counts and
// checks bit-exactness against int8_reference.
void check_bit_exact(const std::shared_ptr<Model>& m, const std::string& op, const std::string& prefix,
                     const std::vector<float>& x, const std::string& what) {
  RunResult ref = run_model(m, forced(op, prefix + ".int8_reference"), 1, x);
  RunResult ref3 = run_model(m, forced(op, prefix + ".int8_reference"), 3, x);
  if (!same_bits(ref.y, ref3.y)) throw bftest::Failure{what + ": int8_reference not thread-count invariant"};
  for (const SimdKernel& k : simd_kernels(prefix)) {
    for (int threads : {1, 2, 3, 4}) {
      RunResult r;
      try {
        r = run_model(m, forced(op, k.name), threads, x);
      } catch (const Error& e) {
        if (k.required) throw bftest::Failure{what + ": " + k.name + " unavailable on a CPU that supports it: " + e.what()};
        break;
      }
      EXPECT_EQ(r.kernel, k.name);
      if (!same_bits(ref.y, r.y))
        throw bftest::Failure{what + ": " + k.name + " threads " + std::to_string(threads) + " differs from int8_reference at " +
                              first_diff(ref.y, r.y)};
      ++g_simd_comparisons;
    }
  }
}

const std::vector<ConvCase>& conv_cases() {
  static const std::vector<ConvCase> c = {
      // n  h   w   ic   oc  k  s  pt pb pl pr dil pad_v  g  act      bias input
      {1, 8, 8, 16, 16, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "none", true, 0},     // typical
      {1, 9, 7, 3, 8, 3, 2, 1, 1, 1, 1, 1, 0.f, 1, "relu", true, 0},       // first layer: ic 3 (unaligned K)
      {1, 6, 6, 33, 17, 3, 1, 1, 1, 1, 1, 1, 0.5f, 1, "relu6", true, 1},   // K tail, O tail, pad_value
      {1, 7, 5, 5, 9, 3, 1, 0, 2, 1, 0, 1, -1.3f, 1, "none", false, 0},    // asymmetric pads, no bias
      {1, 10, 10, 32, 20, 3, 1, 2, 2, 2, 2, 2, 0.f, 1, "relu", true, 0},   // dilation 2
      {1, 5, 5, 64, 64, 1, 1, 0, 0, 0, 0, 1, 0.f, 1, "none", true, 0},     // 1x1
      {1, 12, 12, 12, 70, 5, 2, 2, 2, 2, 2, 1, 100.f, 1, "none", true, 0}, // 5x5, saturating pad_value
      {1, 4, 4, 1, 5, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "none", true, 0},       // ic 1
      {1, 3, 3, 8, 1, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "relu", true, 0},       // oc 1
      {2, 6, 6, 8, 24, 3, 1, 1, 1, 1, 1, 1, 0.25f, 1, "none", true, 0},    // batch 2
      {1, 14, 14, 64, 96, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "relu", true, 3},   // K 576
      {1, 7, 7, 256, 40, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "none", true, 0},    // K 2304, 49 pixels
      {1, 6, 6, 16, 16, 3, 1, 1, 1, 1, 1, 1, 0.7f, 1, "none", true, 2},    // all-zero input: scale 1
      {1, 9, 9, 16, 16, 1, 2, 0, 0, 0, 0, 1, 0.f, 1, "none", true, 0},     // strided 1x1
      {1, 1, 1, 7, 3, 3, 1, 1, 1, 1, 1, 1, -0.4f, 1, "none", true, 0},     // 1x1 input, mostly padding
      {1, 6, 6, 4, 12, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "relu6", true, 0},     // ic 4: SDOT-aligned, not I8MM
      {1, 5, 6, 12, 9, 3, 2, 1, 0, 0, 1, 1, 0.3f, 1, "none", true, 1},     // ic 12, stride 2, odd pads
      {1, 11, 13, 24, 31, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "relu", false, 0},  // P tail (143 pixels), O tail
      {1, 11, 9, 3, 16, 3, 1, 2, 2, 2, 2, 2, 0.1f, 1, "none", true, 0},   // ic 3 with dilation 2
      {1, 6, 6, 33, 8, 9, 1, 4, 4, 4, 4, 1, -0.2f, 1, "none", true, 0},   // unaligned K 2673 (> stack row)
      {1, 20, 20, 3, 64, 7, 2, 3, 3, 3, 3, 1, 0.f, 1, "relu", true, 0},   // RGB stem 7x7/2
      // Enough work that the pool really splits (kInlineNs): fused
      // pack+GEMM path, and the two-phase path with several channel chunks.
      {1, 28, 28, 64, 64, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "none", true, 0},
      {1, 7, 7, 256, 256, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "relu", true, 0},
  };
  return c;
}

const std::vector<ConvCase>& grouped_cases() {
  static const std::vector<ConvCase> c = {
      {1, 8, 8, 16, 16, 3, 1, 1, 1, 1, 1, 1, 0.f, 16, "relu", true, 0},  // depthwise
      {1, 7, 7, 12, 18, 3, 2, 1, 1, 1, 1, 1, 0.2f, 3, "none", true, 0},  // grouped
  };
  return c;
}

const std::vector<DenseCase>& dense_cases() {
  static const std::vector<DenseCase> c = {
      {{1, 512}, 1000, "none", true, 0}, {{1, 3}, 7, "relu", true, 0},  {{2, 1}, 9, "none", true, 0},
      {{3, 4}, 8, "none", false, 1},     {{1, 7}, 1, "none", true, 0},  {{9, 8}, 33, "relu6", true, 0},
      {{8, 9}, 16, "none", true, 0},     {{17, 15}, 17, "none", true, 0}, {{5, 16}, 64, "relu", true, 3},
      {{2, 17}, 10, "none", true, 0},    {{4, 100}, 40, "none", true, 1}, {{2, 3, 64}, 24, "none", true, 0},
      {{1, 1000}, 12, "none", true, 0},  {{2, 32}, 8, "none", true, 2},
      {{1100, 1000}, 8, "relu", true, 1},  // 1.1M input elements: parallel quantization path
      {{32, 1024}, 512, "none", true, 0},  // split over channel chunks
  };
  return c;
}

// ---------------------------------------------------------- error bound (b)
// With per-tensor scale sx and per-channel scale sw, every quantized operand
// satisfies x = sx*(qx + dx), w = sw*(qw + dw) with |dx|,|dw| <= 1/2 + 1e-5
// (1/2 from round-to-nearest; the fp32 division x/s adds at most
// u*|x/s| <= 127*2^-24 < 1e-5; for |x| <= max|x| clamping never triggers).
// Writing ex = sx*dx, ew = sw*dw:
//   x*w - sx*sw*qx*qw = x*ew + ex*w - ex*ew
//   => |x*w - sx*sw*qx*qw| <= |x|*|ew| + |w|*|ex| + |ex|*|ew|
// Summed over the K taps (padding taps use x = pad_value, |pad_value| <= max|x|):
//   Q = sum_k |x_k|*ew + |w_k|*ex + ex*ew,  ew = sw*(1/2+1e-5), ex = sx*(1/2+1e-5)
// Floating point: D = sx*sw*acc is evaluated with three roundings and the
// bias add with one (<= 4.02u|D| + u|b|, u = 2^-24); the float reference
// rounds its double sum once (<= u*(S + |b|), S = sum_k |x_k*w_k|). With
// |D| <= S + Q the total bound is
//   |y_int8 - y_float| <= Q + 8u*(S + Q + |b|)
// and relu/relu6 are 1-Lipschitz, so the bound also holds after activation.
double int8_error_bound(double q, double s, double b) { return q + 8.0 * std::ldexp(1.0, -24) * (s + q + std::fabs(b)); }

float maxabs(const std::vector<float>& v) {
  float m = 0.f;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

float scale_of(float m) { return m > 0.f ? m / 127.f : 1.f; }

struct BoundStats {
  double worst_ratio = 0, max_err = 0;
};

void check_conv_bound(const ConvCase& c, const ConvModel& m, const std::vector<float>& x, const std::vector<float>& y8,
                      const std::vector<float>& yf, BoundStats& st) {
  const double e = 0.5 + 1e-5;
  const double ex = scale_of(maxabs(x)) * e;
  int64_t kic = c.ic / c.groups, ocg = c.oc / c.groups, taps = c.k * c.k * kic;
  int64_t oh = (c.h + c.pt + c.pb - c.dil * (c.k - 1) - 1) / c.stride + 1;
  int64_t ow = (c.w + c.pl + c.pr - c.dil * (c.k - 1) - 1) / c.stride + 1;
  for (int64_t o = 0; o < c.oc; ++o) {
    std::vector<float> row(m.w.begin() + o * taps, m.w.begin() + (o + 1) * taps);
    const double ew = scale_of(maxabs(row)) * e;
    for (int64_t n = 0; n < c.n; ++n)
      for (int64_t oy = 0; oy < oh; ++oy)
        for (int64_t ox = 0; ox < ow; ++ox) {
          double q = 0, s = 0;
          for (int64_t ky = 0; ky < c.k; ++ky)
            for (int64_t kx = 0; kx < c.k; ++kx) {
              int64_t iy = oy * c.stride - c.pt + ky * c.dil, ix = ox * c.stride - c.pl + kx * c.dil;
              bool inside = iy >= 0 && iy < c.h && ix >= 0 && ix < c.w;
              for (int64_t ch = 0; ch < kic; ++ch) {
                double xv = inside ? x[((n * c.h + iy) * c.w + ix) * c.ic + (o / ocg) * kic + ch] : c.pad_value;
                double wv = row[(ky * c.k + kx) * kic + ch];
                q += std::fabs(xv) * ew + std::fabs(wv) * ex + ex * ew;
                s += std::fabs(xv * wv);
              }
            }
          size_t idx = static_cast<size_t>(((n * oh + oy) * ow + ox) * c.oc + o);
          double err = std::fabs(static_cast<double>(y8[idx]) - yf[idx]);
          double bound = int8_error_bound(q, s, m.b[o]);
          if (!(err <= bound))
            throw bftest::Failure{"conv bound violated at " + std::to_string(idx) + ": err " + std::to_string(err) + " > " + std::to_string(bound)};
          st.worst_ratio = std::max(st.worst_ratio, err / bound);
          st.max_err = std::max(st.max_err, err);
        }
  }
}

void check_dense_bound(const DenseCase& c, const ConvModel& m, const std::vector<float>& x, const std::vector<float>& y8,
                       const std::vector<float>& yf, BoundStats& st) {
  const double e = 0.5 + 1e-5;
  const double ex = scale_of(maxabs(x)) * e;
  int64_t k = c.x.back(), rows = static_cast<int64_t>(x.size()) / k;
  for (int64_t o = 0; o < c.o; ++o) {
    std::vector<float> row(m.w.begin() + o * k, m.w.begin() + (o + 1) * k);
    const double ew = scale_of(maxabs(row)) * e;
    for (int64_t r = 0; r < rows; ++r) {
      double q = 0, s = 0;
      for (int64_t j = 0; j < k; ++j) {
        q += std::fabs(x[r * k + j]) * ew + std::fabs(row[j]) * ex + ex * ew;
        s += std::fabs(static_cast<double>(x[r * k + j]) * row[j]);
      }
      size_t idx = static_cast<size_t>(r * c.o + o);
      double err = std::fabs(static_cast<double>(y8[idx]) - yf[idx]);
      double bound = int8_error_bound(q, s, m.b[o]);
      if (!(err <= bound))
        throw bftest::Failure{"dense bound violated at " + std::to_string(idx) + ": err " + std::to_string(err) + " > " + std::to_string(bound)};
      st.worst_ratio = std::max(st.worst_ratio, err / bound);
      st.max_err = std::max(st.max_err, err);
    }
  }
}

}  // namespace

// --------------------------------------------------------------- hand checks
BF_TEST(int8_reference_dense_hand_computed) {
  // max|x| = 127 -> scale_x = 1, so q = rne(x): ties go to even.
  float x[4] = {127.f, -2.5f, 3.5f, 0.5f};  // q = 127, -2, 4, 0
  float w[12] = {127.f, 1.5f, -1.5f, 100.4f,  // scale 1:  127, 2, -2, 100
                 0.f, 0.f, 0.f, 0.f,          // all-zero row: scale 1, q 0
                 -254.f, 2.f, 0.f, 0.f};      // scale 2:  -127, 1, 0, 0
  float b[3] = {0.5f, -1.f, 0.25f};
  for (const char* act : {"none", "relu"}) {
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kF32, {3, 4}, w);
    mb.constant("b", DType::kF32, {3}, b);
    mb.input("x", DType::kF32, {1, 4});
    mb.node("Dense", {"x", "w", "b"}, {"y"}, Json(Json::Object{{"activation", act}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<float> xv(x, x + 4);
    // acc = 16129 - 4 - 8 = 16117 | 0 | -16129 - 2 = -16131 (times scale 2)
    float expect[3] = {16117.5f, -1.f, -32261.75f};
    if (std::string(act) == "relu") expect[1] = expect[2] = 0.f;
    std::vector<std::string> kernels = {"dense.int8_reference"};
    for (const auto& k : simd_kernels("dense"))
      if (k.required) kernels.push_back(k.name);
    for (const auto& k : kernels) {
      RunResult r = run_model(m, forced("Dense", k), 1, xv);
      for (int o = 0; o < 3; ++o) EXPECT_EQ(r.y[o], expect[o]);
    }
  }
}

BF_TEST(int8_reference_conv_hand_computed) {
  // 1x1x2 input, 3x3 kernel, pad 1: 8 padding taps with pad_value 2.5 -> qpad
  // rne(2.5) = 2. x = {127, -0.5} -> q = {127, 0}. Filter all ones except the
  // centre tap of channel 0 (127): scale_w = 1.
  float x[2] = {127.f, -0.5f};
  std::vector<float> w(18, 1.f);
  w[4 * 2 + 0] = 127.f;
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kF32, {1, 3, 3, 2}, w.data());
  mb.input("x", DType::kF32, {1, 1, 1, 2});
  mb.node("Conv2D", {"x", "w", ""}, {"y"},
          Json(Json::Object{{"pads", ints({1, 1, 1, 1})}, {"pad_value", Json(2.5)}}));
  mb.output("y");
  auto m = mb.build();
  std::vector<std::string> kernels = {"conv2d.int8_reference"};
  for (const auto& k : simd_kernels("conv2d"))
    if (k.required) kernels.push_back(k.name);
  for (const auto& k : kernels) {
    RunResult r = run_model(m, forced("Conv2D", k), 1, std::vector<float>(x, x + 2));
    EXPECT_EQ(r.y[0], 127.f * 127.f + 8 * (2 + 2));  // 16161
  }
}

// ------------------------------------------------------- (a) bit-exactness
BF_TEST(int8_simd_conv_bit_exact_vs_int8_reference) {
  int idx = 0;
  for (const ConvCase& c : conv_cases()) {
    ConvModel m = make_conv(c, 300 + idx);
    auto x = make_input(c.n * c.h * c.w * c.ic, c.input, 40 + idx);
    check_bit_exact(m.model, "Conv2D", "conv2d", x, "conv case " + std::to_string(idx));
    ++idx;
  }
  std::printf("    int8 conv: %d SIMD int8 runs compared bit-exactly\n", g_simd_comparisons);
}

BF_TEST(int8_simd_dense_bit_exact_vs_int8_reference) {
  int idx = 0, before = g_simd_comparisons;
  for (const DenseCase& c : dense_cases()) {
    ConvModel m = make_dense(c, 500 + idx);
    auto x = make_input(num_elements(c.x), c.input, 60 + idx);
    check_bit_exact(m.model, "Dense", "dense", x, "dense case " + std::to_string(idx));
    ++idx;
  }
  std::printf("    int8 dense: %d SIMD int8 runs compared bit-exactly\n", g_simd_comparisons - before);
}

BF_TEST(int8_simd_rejects_grouped_conv) {
  for (const ConvCase& c : grouped_cases()) {
    ConvModel m = make_conv(c, 9);
    auto x = make_input(c.h * c.w * c.ic, 0, 1);
    for (const auto& k : simd_kernels("conv2d")) EXPECT_THROWS(run_model(m.model, forced("Conv2D", k.name), 1, x));
    // Selector falls back: to float with {reference, int8_neon}, to int8_reference when allowed.
    EXPECT_EQ(run_model(m.model, tiers_with_simd({"reference"}), 1, x).kernel, std::string("conv2d.reference"));
    EXPECT_EQ(run_model(m.model, tiers_with_simd({"int8_reference"}), 1, x).kernel, std::string("conv2d.int8_reference"));
  }
}

// ------------------------------------------- (b) accuracy vs float reference
BF_TEST(int8_reference_within_analytic_bound_of_float) {
  BoundStats st;
  int idx = 0;
  std::vector<ConvCase> cases = conv_cases();
  for (const auto& g : grouped_cases()) cases.push_back(g);
  for (const ConvCase& c : cases) {
    ++idx;
    ConvModel m = make_conv(c, 700 + idx);
    auto x = make_input(c.n * c.h * c.w * c.ic, c.input, 80 + idx);
    if (std::fabs(c.pad_value) > maxabs(x)) continue;  // saturating pad: bound needs |pad_value| <= max|x|
    RunResult r8 = run_model(m.model, KernelPolicy::tiers({"int8_reference"}), 2, x);
    RunResult rf = run_model(m.model, KernelPolicy::tiers({"reference"}), 2, x);
    EXPECT_EQ(r8.kernel, std::string("conv2d.int8_reference"));
    check_conv_bound(c, m, x, r8.y, rf.y, st);
  }
  idx = 0;
  for (const DenseCase& c : dense_cases()) {
    ++idx;
    ConvModel m = make_dense(c, 900 + idx);
    auto x = make_input(num_elements(c.x), c.input, 90 + idx);
    RunResult r8 = run_model(m.model, KernelPolicy::tiers({"int8_reference"}), 1, x);
    RunResult rf = run_model(m.model, KernelPolicy::tiers({"reference"}), 1, x);
    EXPECT_EQ(r8.kernel, std::string("dense.int8_reference"));
    check_dense_bound(c, m, x, r8.y, rf.y, st);
  }
  std::printf("    int8 vs float: max |err| %.4g, worst err/bound %.3f\n", st.max_err, st.worst_ratio);
  EXPECT_TRUE(st.max_err > 0);        // int8 really ran (it is not exact)
  EXPECT_TRUE(st.worst_ratio < 1.0);  // and never exceeded the bound
}

// ----------------------------------------------------- (c) kernel selection
namespace {

struct Expectation {
  std::string conv_a, dense_a;  // policy {int8_reference, int8_neon, int8_avx2}
  std::string conv_b, dense_b;  // policy {reference, int8_neon, int8_avx2}
};

// Expected choices on a CPU with the given features.
Expectation expected_for(uint32_t features) {
  Expectation e{"conv2d.int8_reference", "dense.int8_reference", "conv2d.reference", "dense.reference"};
#if defined(__aarch64__)
  CpuInfo cpu;
  cpu.features = features;
  const char* best = cpu.has(kFeatNeon | kFeatDotProd | kFeatI8mm) ? "int8_neon_i8mm"
                     : cpu.has(kFeatNeon | kFeatDotProd)          ? "int8_neon_sdot"
                                                                  : nullptr;
  if (best) {
    e.conv_a = e.conv_b = std::string("conv2d.") + best;
    e.dense_a = e.dense_b = std::string("dense.") + best;
  }
#elif defined(__x86_64__)
  CpuInfo cpu;
  cpu.features = features;
  const char* best = cpu.has(kFeatAvx2 | kFeatAvxVnni) ? "int8_avx2_vnni" : cpu.has(kFeatAvx2) ? "int8_avx2" : nullptr;
  if (best) {
    e.conv_a = e.conv_b = std::string("conv2d.") + best;
    e.dense_a = e.dense_b = std::string("dense.") + best;
  }
#elif defined(__arm__) && defined(__ARM_NEON)
  CpuInfo cpu;
  cpu.features = features;
  if (cpu.has(kFeatNeon)) {
    e.conv_a = e.conv_b = "conv2d.int8_neon_a32";
    e.dense_a = e.dense_b = "dense.int8_neon_a32";
  }
#else
  (void)features;
#endif
  return e;
}

std::shared_ptr<Model> gating_conv() { return make_conv({1, 6, 6, 16, 16, 3, 1, 1, 1, 1, 1, 1, 0.f, 1, "none", true, 0}, 5).model; }
std::shared_ptr<Model> gating_dense() { return make_dense({{2, 64}, 24, "none", true, 0}, 6).model; }

std::string choice(const std::shared_ptr<Model>& m, const KernelPolicy& p) {
  SessionOptions o;
  o.policy = p;
  Session s(m, o);
  return s.kernel_choices().begin()->second;
}

void check_expectation(const Expectation& e) {
  auto conv = gating_conv(), dense = gating_dense();
  KernelPolicy a = tiers_with_simd({"int8_reference"}), b = tiers_with_simd({"reference"});
  EXPECT_EQ(choice(conv, a), e.conv_a);
  EXPECT_EQ(choice(dense, a), e.dense_a);
  EXPECT_EQ(choice(conv, b), e.conv_b);
  EXPECT_EQ(choice(dense, b), e.dense_b);
}

// Re-executes this test binary running only int8_gating_child with extra
// environment variables; returns its exit status.
int run_child(const std::vector<std::string>& extra_env) { return bftest::run_self("int8_gating_child", extra_env); }

}  // namespace

BF_TEST(int8_tiers_are_opt_in) {
  auto conv = gating_conv(), dense = gating_dense();
  // Default policy (all tiers) keeps float numerics.
  EXPECT_TRUE(choice(conv, KernelPolicy::all()).find("int8") == std::string::npos);
  EXPECT_TRUE(choice(dense, KernelPolicy::all()).find("int8") == std::string::npos);
  // Listing an int8 tier prefers it over float reference.
  EXPECT_EQ(choice(conv, KernelPolicy::tiers({"reference", "int8_reference"})), std::string("conv2d.int8_reference"));
  check_expectation(expected_for(cpu_info().features));
}

BF_TEST(int8_feature_masking_selects_fallbacks) {
  if (std::getenv("BF_INT8_CHILD")) return;  // we are the child of another run
  const uint32_t have = cpu_info().features;
  const char* prev = std::getenv("BITINO_DISABLE_FEATURES");
  std::string base = prev && *prev ? std::string(prev) + "," : "";
  struct Mask {
    const char* name;
    uint32_t bit;
  };
#if defined(__x86_64__)
  // avxvnni masked -> the VPMADDWD kernels; avx2 masked -> no int8_avx2 kernel.
  const std::vector<Mask> masks = {Mask{"avxvnni", kFeatAvxVnni}, Mask{"avx2", kFeatAvx2}};
#elif defined(__arm__) && defined(__ARM_NEON)
  // 32-bit ARM: neon masked -> no int8_neon kernel (the AArch32 kernels need only NEON).
  const std::vector<Mask> masks = {Mask{"neon", kFeatNeon}};
#else
  const std::vector<Mask> masks = {Mask{"i8mm", kFeatI8mm}, Mask{"dotprod", kFeatDotProd}};
#endif
  for (Mask mk : masks) {
    Expectation e = expected_for(have & ~mk.bit);
    // The task's gating rules: i8mm masked -> SDOT kernels (on an AArch64 CPU
    // with dotprod); dotprod masked -> no int8_neon kernel at all. Likewise
    // avxvnni -> int8_avx2 (with avx2), avx2 -> none.
#if defined(__aarch64__)
    if (std::string(mk.name) == "i8mm" && (have & kFeatDotProd)) EXPECT_EQ(e.conv_a, std::string("conv2d.int8_neon_sdot"));
#endif
#if defined(__x86_64__)
    if (std::string(mk.name) == "avxvnni" && (have & kFeatAvx2)) EXPECT_EQ(e.conv_a, std::string("conv2d.int8_avx2"));
#endif
    if (std::string(mk.name) == "dotprod" || std::string(mk.name) == "avx2" || std::string(mk.name) == "neon")
      EXPECT_EQ(e.conv_b, std::string("conv2d.reference"));
    std::vector<std::string> env = {"BITINO_DISABLE_FEATURES=" + base + mk.name, "BF_INT8_CHILD=1", "BF_INT8_MASKED=" + std::string(mk.name),
                                    "BF_INT8_EXPECT=" + e.conv_a + "," + e.dense_a + "," + e.conv_b + "," + e.dense_b};
    int rc = run_child(env);
    if (rc != 0)
      throw bftest::Failure{std::string("masked child failed (") + mk.name + "), rerun: BITINO_DISABLE_FEATURES=" + base + mk.name +
                            " BF_INT8_CHILD=1 BF_INT8_MASKED=" + mk.name + " BF_INT8_EXPECT=" + e.conv_a + "," + e.dense_a + "," +
                            e.conv_b + "," + e.dense_b + " bitino_tests int8_gating_child"};
  }
}

BF_TEST(int8_gating_child) {
  const char* expect = std::getenv("BF_INT8_EXPECT");
  if (!std::getenv("BF_INT8_CHILD") || !expect) return;  // only meaningful when spawned
  std::string masked = std::getenv("BF_INT8_MASKED") ? std::getenv("BF_INT8_MASKED") : "";
  if (masked == "i8mm") EXPECT_TRUE(!cpu_info().has(kFeatI8mm));
  if (masked == "dotprod") EXPECT_TRUE(!cpu_info().has(kFeatDotProd));
  if (masked == "avxvnni") EXPECT_TRUE(!cpu_info().has(kFeatAvxVnni));
  if (masked == "avx2") EXPECT_TRUE(!cpu_info().has(kFeatAvx2));
  if (masked == "neon") EXPECT_TRUE(!cpu_info().has(kFeatNeon));
  std::vector<std::string> parts;
  std::string s = expect;
  size_t b = 0;
  while (b <= s.size()) {
    size_t e = s.find(',', b);
    if (e == std::string::npos) e = s.size();
    parts.push_back(s.substr(b, e - b));
    b = e + 1;
  }
  EXPECT_EQ(parts.size(), size_t(4));
  check_expectation({parts[0], parts[1], parts[2], parts[3]});
}

// ------------------------------------------------ adversarial review cases
namespace {

// Every int8 kernel of `prefix` that this CPU can run (reference first).
std::vector<std::string> runnable_int8(const std::string& prefix) {
  std::vector<std::string> ks = {prefix + ".int8_reference"};
  for (const auto& k : simd_kernels(prefix))
    if (k.required) ks.push_back(k.name);
  return ks;
}

}  // namespace

BF_TEST(int8_session_reuse_refreshes_per_run_state) {
  // One session, several run() calls with inputs whose scale_x and qpad
  // differ: the per-run scale/bias vectors, the padding row and the packed A
  // buffer must not keep anything from the previous run.
  ConvCase c{1, 9, 11, 20, 27, 3, 1, 1, 1, 1, 1, 1, 0.6f, 1, "relu6", true, 0};
  ConvModel m = make_conv(c, 77);
  std::vector<std::vector<float>> xs = {make_input(c.h * c.w * c.ic, 0, 1), make_input(c.h * c.w * c.ic, 2, 2),
                                        make_input(c.h * c.w * c.ic, 3, 3), make_input(c.h * c.w * c.ic, 1, 4)};
  for (auto& v : xs[2]) v *= 1e-3f;  // tiny scale: qpad saturates to 127
  for (const auto& k : runnable_int8("conv2d")) {
    for (int threads : {1, 4}) {
      SessionOptions o;
      o.num_threads = threads;
      o.policy = forced("Conv2D", k);
      Session s(m.model, o);
      for (size_t i = 0; i < xs.size(); ++i) {
        std::memcpy(s.input(0).raw(), xs[i].data(), xs[i].size() * 4);
        s.run();
        const Tensor& y = s.output(0);
        std::vector<float> got(y.data<float>(), y.data<float>() + y.size());
        RunResult fresh = run_model(m.model, forced("Conv2D", "conv2d.int8_reference"), 1, xs[i]);
        if (!same_bits(got, fresh.y))
          throw bftest::Failure{k + " run " + std::to_string(i) + " threads " + std::to_string(threads) +
                                " differs from a fresh session: " + first_diff(fresh.y, got)};
      }
    }
  }
}

BF_TEST(int8_tiny_subnormal_scale_is_well_defined) {
  // max|x| < 127 * 2^-150: max/127 rounds to 0 in fp32. A zero scale would
  // make x/scale = 0/0 = NaN, and converting NaN to int8 is undefined
  // behaviour. The scale must fall back to 1 (all q = 0), so y = act(bias).
  // Same for a filter row of subnormal weights.
  const int64_t k = 37, o = 11;
  std::vector<float> w(static_cast<size_t>(o * k), 0.25f), b(static_cast<size_t>(o));
  for (int64_t j = 0; j < k; ++j) w[j] = j % 3 ? 3e-44f : 0.f;  // row 0: subnormal weights
  for (int64_t i = 0; i < o; ++i) b[i] = 0.5f * static_cast<float>(i) - 2.f;
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kF32, {o, k}, w.data());
  mb.constant("b", DType::kF32, {o}, b.data());
  mb.input("x", DType::kF32, {3, k});
  mb.node("Dense", {"x", "w", "b"}, {"y"}, Json(Json::Object{{"activation", "none"}}));
  mb.output("y");
  auto m = mb.build();
  std::vector<float> x(static_cast<size_t>(3 * k), 0.f);
  for (size_t i = 0; i < x.size(); ++i) x[i] = i % 4 == 0 ? 0.f : (i % 2 ? 5e-44f : -4e-44f);
  EXPECT_EQ(int8_scale(5e-44f), 1.f);
  EXPECT_TRUE(int8_scale(1e-38f) > 0.f && int8_scale(1e-38f) < 1e-39f);  // normal path unchanged
  for (const auto& kn : runnable_int8("dense")) {
    RunResult r = run_model(m, forced("Dense", kn), 1, x);
    for (int64_t row = 0; row < 3; ++row)
      for (int64_t i = 0; i < o; ++i) EXPECT_EQ(r.y[row * o + i], b[i]);
  }
  // Weight row of subnormals with a normal input: row 0 gives bias exactly.
  std::vector<float> xn = make_input(3 * k, 0, 9);
  RunResult ref = run_model(m, forced("Dense", "dense.int8_reference"), 1, xn);
  for (int64_t row = 0; row < 3; ++row) EXPECT_EQ(ref.y[row * o], b[0]);
  for (const auto& kn : runnable_int8("dense")) EXPECT_TRUE(same_bits(run_model(m, forced("Dense", kn), 2, xn).y, ref.y));
}

BF_TEST(int8_dense_k_at_int32_bound) {
  // K = kInt8MaxK with every product at +-127*127: |acc| = K*16129, just
  // below 2^31. One more K must be rejected (selector falls back to float).
  for (int64_t k : {kInt8MaxK, kInt8MaxK + 1}) {
    const int64_t o = 9, rows = 3;
    std::vector<float> w(static_cast<size_t>(o * k));
    for (int64_t i = 0; i < o; ++i)
      for (int64_t j = 0; j < k; ++j) w[i * k + j] = i % 2 ? -1.f : 1.f;
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kF32, {o, k}, w.data());
    mb.input("x", DType::kF32, {rows, k});
    mb.node("Dense", {"x", "w", ""}, {"y"}, Json(Json::Object{{"activation", "none"}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<float> x(static_cast<size_t>(rows * k));
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t j = 0; j < k; ++j) x[r * k + j] = r == 1 ? -2.f : 2.f;
    if (k > kInt8MaxK) {
      for (const auto& kn : runnable_int8("dense")) EXPECT_THROWS(run_model(m, forced("Dense", kn), 1, x));
      EXPECT_EQ(run_model(m, tiers_with_simd({"reference", "int8_reference"}), 1, x).kernel, std::string("dense.reference"));
      continue;
    }
    const int32_t acc = static_cast<int32_t>(k * 16129);
    for (const auto& kn : runnable_int8("dense")) {
      for (int threads : {1, 4}) {
        RunResult r = run_model(m, forced("Dense", kn), threads, x);
        for (int64_t row = 0; row < rows; ++row)
          for (int64_t i = 0; i < o; ++i) {
            int32_t a = (row == 1) != (i % 2 == 1) ? -acc : acc;
            EXPECT_EQ(r.y[row * o + i], int8_epilogue(a, 2.f / 127.f, 1.f / 127.f, 0.f, Activation::kNone));
          }
      }
    }
  }
}

BF_TEST(int8_simd_multichunk_unaligned_bit_exact) {
  // Two-phase path (weights over kWeightChunkBytes -> several channel
  // chunks) combined with unaligned K, pixel/channel tails and enough work to
  // leave the inline threshold; plus Dense with the same properties.
  const std::vector<ConvCase> cs = {
      {1, 9, 9, 33, 72, 9, 1, 4, 4, 4, 4, 1, -0.3f, 1, "relu", true, 0},     // K 2673 (byte scatter), 3 chunks
      {1, 14, 13, 20, 1003, 3, 1, 1, 1, 1, 1, 1, 0.2f, 1, "none", true, 1},  // K 180: stack row, ic%8 != 0, O tail
      {2, 10, 9, 3, 900, 7, 2, 3, 2, 3, 1, 1, 0.f, 1, "relu6", false, 3},     // RGB stem, batch 2, 900 channels
  };
  int idx = 0;
  for (const ConvCase& c : cs) {
    ConvModel m = make_conv(c, 1200 + idx);
    auto x = make_input(c.n * c.h * c.w * c.ic, c.input, 1300 + idx);
    check_bit_exact(m.model, "Conv2D", "conv2d", x, "multichunk conv " + std::to_string(idx++));
  }
  const std::vector<DenseCase> ds = {{{37, 1001}, 300, "relu", true, 0}, {{1, 4099}, 517, "none", true, 1}, {{19, 129}, 2049, "none", false, 0}};
  idx = 0;
  for (const DenseCase& c : ds) {
    ConvModel m = make_dense(c, 1400 + idx);
    auto x = make_input(num_elements(c.x), c.input, 1500 + idx);
    check_bit_exact(m.model, "Dense", "dense", x, "multichunk dense " + std::to_string(idx++));
  }
}

// ------------------------------------------------ x86-64 AVX2 quantizer
// The AVX2 activation quantizer (division, MXCSR round-to-nearest-even,
// clamp, +128 shift for VNNI) must produce exactly int8_quantize() for every
// element, and the AVX2 max-abs exactly the scalar max. Adversarial inputs:
// exact .5 ties of both parities, their float neighbours, +-127*s and beyond,
// +-0, values far above max (as for a saturating pad_value), and every tail
// length.
BF_TEST(int8_avx2_quantizer_bit_exact_vs_scalar) {
#if defined(__x86_64__)
  if (!cpu_info().has(kFeatAvx2)) return;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  int checked = 0;
  for (float scale : {1.f, 0.0137f, 3.25f, 1e-30f}) {
    std::vector<float> x;
    for (int k = -130; k <= 130; ++k) {
      const float tie = (static_cast<float>(k) + 0.5f) * scale;
      x.push_back(tie);
      x.push_back(std::nextafter(tie, 1e30f));
      x.push_back(std::nextafter(tie, -1e30f));
      x.push_back(static_cast<float>(k) * scale);
    }
    for (float v : {0.f, -0.f, 127.f * scale, -127.f * scale, 1e20f, -1e20f, 128.4f * scale, -300.f * scale}) x.push_back(v);
    for (int i = 0; i < 4000; ++i) x.push_back(nd(rng) * 40.f * scale);
    for (size_t n : {size_t(0), size_t(1), size_t(5), size_t(7), size_t(8), size_t(9), size_t(15), size_t(16), size_t(17), size_t(31),
                     size_t(33), x.size()}) {
      std::vector<uint8_t> qu(n + 1, 0xAB);
      std::vector<int16_t> qs(n + 1, 0x5555);
      int8_avx2::quantize_u8(x.data(), static_cast<int64_t>(n), scale, qu.data());
      int8_avx2::quantize_s16(x.data(), static_cast<int64_t>(n), scale, qs.data());
      for (size_t i = 0; i < n; ++i) {
        const int q = int8_quantize(x[i], scale);
        if (qu[i] != q + 128 || qs[i] != q)
          throw bftest::Failure{"quantizer mismatch at x=" + std::to_string(x[i]) + " scale=" + std::to_string(scale) + ": scalar " +
                                std::to_string(q) + ", u8 " + std::to_string(int(qu[i]) - 128) + ", s16 " + std::to_string(qs[i])};
        ++checked;
      }
      EXPECT_EQ(int(qu[n]), 0xAB);  // nothing written past n
      EXPECT_EQ(int(qs[n]), 0x5555);
      float m = 0.f;
      for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
      EXPECT_EQ(int8_avx2::maxabs(x.data(), static_cast<int64_t>(n)), m);
    }
  }
  std::printf("    avx2 quantizer: %d elements bit-exact with int8_quantize\n", checked);
#endif
}

// ------------------------------------------------ 32-bit ARM quantizer
// int8_neon_a32 quantizes with one NEON multiply by 1/scale and falls back to the IEEE division near half-integers,
// for denormal inputs and for scales without a normal reciprocal (int8_neon_a32.cc, quantize_fast). Through a Dense
// of one input feature and weight 1 (y = float(127 q) * scale / 127: distinct codes give distinct outputs) its codes
// must equal int8_reference's for adversarial inputs: exact .5 ties of both parities at the scale of the run, their
// float neighbours, +-max, +-0, denormals next to a normal max (scale 2^-126: a power-of-two scale whose reciprocal
// multiply would flush them), tensors of tiny values (denormal scale), and every tail length.
BF_TEST(int8_neon_a32_quantizer_bit_exact_vs_reference) {
  std::mt19937 rng(17);
  std::normal_distribution<float> nd(0.f, 1.f);
  int checked = 0;
  for (float m : {127.f, 1.74f, 412.75f, 127.f * 0x1p-126f, 3e-38f, 1e-40f}) {
    const float s = int8_scale(m);
    std::vector<float> x = {m, -m, 0.f, -0.f};
    for (int k = -127; k <= 126; ++k) {
      const float tie = (static_cast<float>(k) + 0.5f) * s;
      if (std::fabs(tie) > m) continue;
      x.push_back(tie);
      x.push_back(std::nextafter(tie, 1e30f));
      x.push_back(std::nextafter(tie, -1e30f));
      x.push_back(static_cast<float>(k) * s);
    }
    for (float d : {1e-45f, -3e-42f, 7e-39f, -1.1e-38f}) x.push_back(std::fabs(d) <= m ? d : 0.f);
    for (int i = 0; i < 3000; ++i) x.push_back(std::max(-m, std::min(m, nd(rng) * 0.4f * m)));
    for (size_t n : {size_t(1), size_t(5), size_t(15), size_t(16), size_t(17), size_t(33), x.size()}) {
      std::vector<float> xn(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n));
      xn[0] = m;  // the max stays in every prefix
      const float w = 1.f;
      bftest::ModelBuilder mb;
      mb.constant("w", DType::kF32, {1, 1}, &w);
      mb.input("x", DType::kF32, {static_cast<int64_t>(n), 1});
      mb.node("Dense", {"x", "w", ""}, {"y"}, Json(Json::Object{{"activation", "none"}}));
      mb.output("y");
      auto model = mb.build();
      RunResult ref = run_model(model, forced("Dense", "dense.int8_reference"), 1, xn);
      for (const auto& k : simd_kernels("dense")) {
        if (!k.required) continue;
        RunResult r = run_model(model, forced("Dense", k.name), 1, xn);
        if (!same_bits(ref.y, r.y))
          throw bftest::Failure{k.name + " quantizer differs from int8_reference (max " + std::to_string(m) + ", n " + std::to_string(n) +
                                ") at " + first_diff(ref.y, r.y)};
        checked += static_cast<int>(n);
      }
    }
  }
  std::printf("    simd quantizer (via Dense): %d elements bit-exact with int8_reference\n", checked);
}
