// Full-integer I8* operators (stage int8_full, docs/int8.md):
//  * fixed-point requantization: q8::requant vs an independent 128-bit definition, edge cases, and
//    (AArch64) the SQRDMULH/SRSHL/SQSHL vector form used by the NEON kernels;
//  * binarization thresholds: q >= qthr_ge(t) <=> deq(q) >= t (and gt) for every code, random and
//    adversarial (s, z, t), including ties t == deq(q) and thresholds outside the grid;
//  * hand-computed values of the reference kernels;
//  * every int8_full_neon kernel bit-exact with int8_full_reference on randomized shapes, tails,
//    parameters and 1 / 4 threads (on x86-64 the int8_full stage selects the reference kernels and the
//    comparison is trivially exact; the test then only checks the plumbing).
//  * 8-bit log2 code (l8): scalar encode / decode vs independent definitions, NEON encode16 / decode16 vs scalar,
//    I8BConv2DRes (l8 / linear residuals, bias, bit output ge / gt, int8 / l8 / bit inputs, chained), I8Conv2D l8
//    output and I8BConv2D with a bit input, bit-exact vs the reference (x86-64: the portable wrappers).
//  * I8Conv2D bit output (i8 / l8 / f32 epilogues, groups, ties) and I8MaxPool2D of bit tensors (vs its definition and
//    vs the reference), as used by the l8 stem variants "bits" / "l8_bits" (quantize_l8.py).
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <random>
#include <set>

#include "bitino/bitino.h"
#include "kernels/int8_full/int8_full.h"
#include "kernels/int8_full/l8_neon.h"
#include "kernels/int8_full/q8_math.h"
#include "testing.h"
#include "tools/model_builder.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace bitino;

namespace {

Json ints(std::vector<int64_t> v) {
  Json::Array a;
  for (auto x : v) a.push_back(Json(x));
  return Json(a);
}

// ------------------------------------------------------------------ requant
// Independent definition with 128-bit integers: floor((x*2^l*M + 2^30) / 2^31), then round-half-up >> r.
#if defined(__SIZEOF_INT128__)
int64_t requant_def(int64_t x, int32_t m, int32_t shift) {
  const int l = shift > 0 ? shift : 0, r = shift < 0 ? -shift : 0;
  __int128 xs = static_cast<__int128>(x) * (static_cast<__int128>(1) << l);
  if (xs > INT32_MAX) xs = INT32_MAX;
  if (xs < INT32_MIN) xs = INT32_MIN;
  __int128 p = xs * m + (static_cast<__int128>(1) << 30);
  __int128 h = p >> 31;  // arithmetic shift of a signed 128-bit value = floor
  if (r == 0) return static_cast<int64_t>(h);
  return static_cast<int64_t>((h + (static_cast<__int128>(1) << (r - 1))) >> r);
}
#else
// 32-bit targets have no __int128: the same definition in int64, which holds every intermediate once x * 2^l is
// saturated to int32 (it is formed only when |x * 2^l| <= 2^62, otherwise it saturates by the sign of x).
int64_t requant_def(int64_t x, int32_t m, int32_t shift) {
  const int l = shift > 0 ? shift : 0, r = shift < 0 ? -shift : 0;
  int64_t xs;
  if (x == 0) xs = 0;
  else if (l > 62 || x > (INT64_C(1) << (62 - l)) || x < -(INT64_C(1) << (62 - l))) xs = x > 0 ? INT32_MAX : INT32_MIN;
  else xs = x * (INT64_C(1) << l);
  if (xs > INT32_MAX) xs = INT32_MAX;
  if (xs < INT32_MIN) xs = INT32_MIN;
  const int64_t p = xs * m + (INT64_C(1) << 30);  // |xs * m| < 2^62
  const int64_t h = p >> 31;                        // arithmetic shift = floor
  if (r == 0) return h;
  return (h + (INT64_C(1) << (r - 1))) >> r;
}
#endif

BF_TEST(int8_full_requant_matches_definition_and_edges) {
  std::mt19937_64 rng(7);
  std::vector<int64_t> xs = {0, 1, -1, 2, -2, 127, -128, 255, -255, (1 << 20), -(1 << 20), INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1};
  std::vector<int32_t> ms = {1 << 30, (1 << 30) + 1, INT32_MAX, 1518500250, 1395864372};
  std::vector<int32_t> shifts = {-31, -30, -20, -8, -1, 0, 1, 2, 7, 20};
  int64_t n = 0;
  for (auto x : xs)
    for (auto m : ms)
      for (auto s : shifts) {
        EXPECT_EQ(q8::requant(x, m, s), requant_def(x, m, s));
        ++n;
      }
  for (int i = 0; i < 200000; ++i) {
    const int64_t x = static_cast<int32_t>(rng());
    const int32_t m = static_cast<int32_t>((1u << 30) + (rng() % (1u << 30)));
    const int32_t s = static_cast<int32_t>(rng() % 40) - 31;
    EXPECT_EQ(q8::requant(x, m, s), requant_def(x, m, s));
  }
  // rounding ties: x*M/2^31 = k + 1/2 rounds up for both signs (round half up)
  EXPECT_EQ(q8::requant(1, 1 << 30, 0), 1);    // 0.5 -> 1
  EXPECT_EQ(q8::requant(-1, 1 << 30, 0), 0);   // -0.5 -> 0
  EXPECT_EQ(q8::requant(3, 1 << 30, -1), 1);   // 1.5 -> 2 in mulhi, then 2/2 = 1
  EXPECT_EQ(q8::rshr(-3, 1), -1);              // -1.5 -> -1
  EXPECT_EQ(q8::rshr(3, 1), 2);                // 1.5 -> 2
  EXPECT_EQ(q8::rdiv(-3, 2), -2);              // average pooling: half away from zero
  EXPECT_EQ(q8::rdiv(3, 2), 2);
  EXPECT_TRUE(n > 0);
}

#if defined(__aarch64__)
BF_TEST(int8_full_requant_neon_matches_scalar) {
  std::mt19937_64 rng(11);
  for (int i = 0; i < 100000; ++i) {
    int32_t x[4];
    for (auto& v : x) v = static_cast<int32_t>(rng());
    if (i % 7 == 0) x[0] = INT32_MIN, x[1] = INT32_MAX, x[2] = 0, x[3] = -1;
    const int32_t m = static_cast<int32_t>((1u << 30) + (rng() % (1u << 30)));
    const int32_t s = static_cast<int32_t>(rng() % 40) - 31;
    const int32x4_t r = vrshlq_s32(vqrdmulhq_s32(vqshlq_s32(vld1q_s32(x), vdupq_n_s32(s > 0 ? s : 0)), vdupq_n_s32(m)),
                                   vdupq_n_s32(s < 0 ? s : 0));
    int32_t out[4];
    vst1q_s32(out, r);
    for (int k = 0; k < 4; ++k) EXPECT_EQ(static_cast<int64_t>(out[k]), q8::requant(x[k], m, s));
  }
}
#endif

// --------------------------------------------------------------- thresholds
void check_thresholds(float t, float s, int32_t z) {
  const int ge = q8::qthr_ge(t, s, z), gt = q8::qthr_gt(t, s, z);
  for (int q = -128; q <= 127; ++q) {
    const float d = q8::deq(q, s, z);
    if ((d >= t) != (q >= ge)) throw bftest::Failure{"qthr_ge mismatch"};
    if ((d > t) != (q >= gt)) throw bftest::Failure{"qthr_gt mismatch"};
  }
  // the analytic estimate ceil(t/s + z) is within one code of the exact threshold (when inside the grid)
  const double est = std::ceil(static_cast<double>(t) / s + z);
  if (est > -127 && est < 126) EXPECT_TRUE(std::fabs(est - ge) <= 1.0);
}

BF_TEST(int8_full_binarization_thresholds_equal_dequantize_then_compare) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  int64_t n = 0;
  for (int i = 0; i < 20000; ++i) {
    const float s = std::ldexp(1.f + std::fabs(u(rng)), static_cast<int>(rng() % 40) - 30);
    const int32_t z = (i % 5 == 0) ? static_cast<int32_t>(rng() % 1000000) - 500000 : static_cast<int32_t>(rng() % 256) - 128;
    // thresholds: random, exact grid points (ties), their float neighbours, outside the grid
    const int q0 = static_cast<int>(rng() % 256) - 128;
    const float d0 = q8::deq(q0, s, z);
    for (float t : {u(rng) * 200.f * s + q8::deq(0, s, z), d0, std::nextafter(d0, 1e30f), std::nextafter(d0, -1e30f),
                    q8::deq(-128, s, z) - s, q8::deq(127, s, z) + s, 0.f, -0.f}) {
      check_thresholds(t, s, z);
      ++n;
    }
  }
  // ties: deq(q) == t exactly -> ge includes q, gt excludes it
  EXPECT_EQ(q8::qthr_ge(0.5f, 0.25f, 0), 2);
  EXPECT_EQ(q8::qthr_gt(0.5f, 0.25f, 0), 3);
  EXPECT_EQ(q8::qthr_ge(1e9f, 0.25f, 0), 128);    // never
  EXPECT_EQ(q8::qthr_ge(-1e9f, 0.25f, 0), -128);  // always
  EXPECT_TRUE(n > 100000);
}

// ------------------------------------------------------ hand-computed values
std::vector<int8_t> run_model(const bftest::ModelBuilder& mb, const std::string& stage_tiers, int threads,
                              const std::vector<std::vector<uint8_t>>& inputs, std::map<std::string, std::string>* kernels = nullptr) {
  auto model = mb.build();
  SessionOptions so;
  so.num_threads = threads;
  so.spin_run = threads > 1;  // multi-threaded runs also exercise the hot thread-pool mode
  // kernel tests: no Session rewrites (the int8 rewrites of src/graph_rewrite_i8.h may replace the node under test,
  // e.g. a one-input I8AffineSum by an I8Lut; they are tested in test_int8_fused.cc)
  so.fuse = false;
  std::set<std::string> tiers;
  size_t p = 0;
  while (p <= stage_tiers.size()) {
    size_t q = stage_tiers.find(',', p);
    if (q == std::string::npos) q = stage_tiers.size();
    tiers.insert(stage_tiers.substr(p, q - p));
    p = q + 1;
  }
  so.policy = KernelPolicy::tiers(tiers);
  Session s(model, so);
  for (size_t i = 0; i < inputs.size(); ++i) std::memcpy(s.input(i).raw(), inputs[i].data(), inputs[i].size());
  s.run();
  if (kernels) *kernels = s.kernel_choices();
  const Tensor& y = s.output(0);
  std::vector<int8_t> out(y.bytes());
  std::memcpy(out.data(), y.raw(), y.bytes());
  return out;
}

const char* kRef = "reference,int8_full_reference";
const char* kFull = "reference,binary_scalar,neon,binary_avx2,float_opt,int8_full_reference,int8_full_opt,int8_full_neon";

BF_TEST(int8_full_reference_hand_values) {
  {  // I8Add: a = 0.5*(qa - 0), b = 0.25*(qb - 4), y = 0.5*(qy + 2)
    bftest::ModelBuilder mb;
    mb.input("a", DType::kI8, {1, 4});
    mb.input("b", DType::kI8, {1, 4});
    // twice_max = 1.0: real_a = 0.5, real_b = 0.25, real_out = 1/(2^20 * 0.5)
    Json at = Json::Object{{"a_zero", 0}, {"b_zero", 4}, {"left_shift", 20}, {"a_mult", 1 << 30}, {"a_shift", 0},
                           {"b_mult", 1 << 30}, {"b_shift", -1}, {"out_mult", 1 << 30}, {"out_shift", -18}, {"y_zero", -2}};
    mb.node("I8Add", {"a", "b"}, {"y"}, at);
    mb.output("y");
    std::vector<uint8_t> a = {2, 0, 0xFF, 100}, b = {8, 4, 0, 4};  // a = {1, 0, -0.5, 50}, b = {1, 0, -1, 0}
    auto y = run_model(mb, kRef, 1, {a, b});
    // y_real = {2, 0, -1.5, 50} -> q = y/0.5 - 2 = {2, -2, -5, 98}
    EXPECT_EQ(int(y[0]), 2);
    EXPECT_EQ(int(y[1]), -2);
    EXPECT_EQ(int(y[2]), -5);
    EXPECT_EQ(int(y[3]), 98);
  }
  {  // I8Softmax: equal inputs -> uniform 1/4, output scale 1/256, zero point -128 -> -64
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {1, 4});
    std::vector<int32_t> tab(256);
    for (int d = 0; d < 256; ++d) tab[d] = static_cast<int32_t>(std::lround(std::ldexp(std::exp(-0.1 * d), 20)));
    mb.constant("e", DType::kI32, {256}, tab.data());
    mb.node("I8Softmax", {"x", "e"}, {"y"}, Json::Object{{"inv_scale", 256.0}, {"y_zero", -128}});
    mb.output("y");
    auto y = run_model(mb, kRef, 1, {{5, 5, 5, 5}});
    for (int i = 0; i < 4; ++i) EXPECT_EQ(int(y[i]), -64);
  }
  {  // I8LayerNorm: row {-1, 1} (x_scale 1, x_zero 0), eps 0 -> normalized {-1, +1}; G = 10, B = 0 -> {-10, 10}
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {1, 2});
    float g[2] = {10.f, 10.f}, b[2] = {0.f, 0.f};
    mb.constant("g", DType::kF32, {2}, g);
    mb.constant("b", DType::kF32, {2}, b);
    mb.node("I8LayerNorm", {"x", "g", "b"}, {"y"}, Json::Object{{"x_zero", 0}, {"x_scale", 1.0}, {"eps", 0.0}});
    mb.output("y");
    auto y = run_model(mb, kRef, 1, {{0xFF, 1}});
    EXPECT_EQ(int(y[0]), -10);
    EXPECT_EQ(int(y[1]), 10);
  }
}

// ------------------------------------------------- NEON vs reference (random)
struct Rng {
  std::mt19937 g;
  explicit Rng(uint32_t s) : g(s) {}
  int i(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
  float f(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g); }
  std::vector<uint8_t> i8(int64_t n) {
    std::vector<uint8_t> v(static_cast<size_t>(n));
    for (auto& x : v) x = static_cast<uint8_t>(i(-128, 127));
    return v;
  }
  std::vector<float> f32(int64_t n, float lo, float hi) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = f(lo, hi);
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

std::vector<uint8_t> bytes_of(const std::vector<float>& v) {
  std::vector<uint8_t> b(v.size() * 4);
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// Runs the model at the reference stage and at int8_full (1 and 4 threads); outputs must be identical and,
// on AArch64, the expected NEON kernel must have been selected for op `op`.
int64_t g_compared = 0;
void compare_stages(const bftest::ModelBuilder& mb, const std::vector<std::vector<uint8_t>>& in, const std::string& op,
                    const std::string& neon_kernel) {
  auto ref = run_model(mb, kRef, 1, in);
  for (int t : {1, 4}) {
    std::map<std::string, std::string> ks;
    auto got = run_model(mb, kFull, t, in, &ks);
    if (got != ref) throw bftest::Failure{"int8_full NEON output differs from the reference for " + op + " (" + neon_kernel + ")"};
#if defined(__aarch64__)
    bool found = neon_kernel.empty();
    for (const auto& kv : ks) found = found || kv.second == neon_kernel;
    if (!found) throw bftest::Failure{"expected kernel " + neon_kernel + " to be selected for " + op};
#endif
    ++g_compared;
  }
}

BF_TEST(int8_full_neon_bit_exact_vs_reference) {
  Rng r(2026);
  // I8Quantize
  for (int64_t n : {1, 15, 16, 17, 100, 70000}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kF32, {n});
    mb.node("I8Quantize", {"x"}, {"y"}, Json::Object{{"scale", static_cast<double>(r.f(0.001f, 0.2f))}, {"zero", r.i(-20, 20)}});
    mb.output("y");
    auto x = r.f32(n, -30.f, 30.f);
    for (int64_t i = 0; i < n; i += 7) x[i] = std::round(x[i]) * 0.5f;  // ties after the division
    compare_stages(mb, {bytes_of(x)}, "I8Quantize", "i8quantize.neon");
  }
  // I8Lut (one table)
  for (int64_t n : {1, 16, 33, 1000, 70001}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {n});
    auto tab = r.i8(256);
    mb.constant("t", DType::kI8, {1, 256}, tab.data());
    mb.node("I8Lut", {"x", "t"}, {"y"});
    mb.output("y");
    compare_stages(mb, {r.i8(n)}, "I8Lut", "i8lut.neon");
  }
  // I8Add (same shapes; scalar b of shape [1] / [1, 1]; out multiplier with a positive shift)
  for (int64_t n : {1, 15, 16, 31, 512, 70000}) {
    for (int bmode = 0; bmode < 3; ++bmode) {
      bftest::ModelBuilder mb;
      mb.input("a", DType::kI8, bmode == 2 ? Shape{2, n} : Shape{n});
      mb.input("b", DType::kI8, bmode == 0 ? Shape{n} : (bmode == 1 ? Shape{1} : Shape{1, 1}));
      auto ma = r.mult(0.05, 0.5), mbb = r.mult(0.05, 0.5), mo = n == 31 ? r.mult(1.0, 3.0) : r.mult(1e-7, 1e-4);
      const int zy = r.i(-128, 127);
      Json at = Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"left_shift", 20}, {"a_mult", ma.first},
                             {"a_shift", ma.second}, {"b_mult", mbb.first}, {"b_shift", mbb.second}, {"out_mult", mo.first},
                             {"out_shift", mo.second}, {"y_zero", zy}, {"qmin", std::max(-128, zy - 50)}, {"qmax", 127}};
      mb.node("I8Add", {"a", "b"}, {"y"}, at);
      mb.output("y");
      compare_stages(mb, {r.i8(bmode == 2 ? 2 * n : n), r.i8(bmode == 0 ? n : 1)}, "I8Add", "i8add.neon");
    }
  }
  // I8Add, same shapes, full output range, one input multiplier in [0.5, 1) (shift 0; the specialized loop that skips
  // its rounding shift), out shift negative or positive
  for (int64_t n : {1, 16, 47, 4096, 70000}) {
    for (int side = 0; side < 3; ++side) {
      bftest::ModelBuilder mb;
      mb.input("a", DType::kI8, {n});
      mb.input("b", DType::kI8, {n});
      auto ma = side == 0 ? r.mult(0.5, 0.99) : r.mult(0.05, 0.49), mbb = side == 1 ? r.mult(0.5, 0.99) : r.mult(0.05, 0.49);
      if (side == 2) ma = r.mult(0.5, 0.99), mbb = r.mult(0.5, 0.99);  // both shifts 0
      auto mo = n == 47 ? r.mult(1.0, 3.0) : r.mult(1e-7, 1e-4);
      Json at = Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"left_shift", 20}, {"a_mult", ma.first},
                             {"a_shift", ma.second}, {"b_mult", mbb.first}, {"b_shift", mbb.second}, {"out_mult", mo.first},
                             {"out_shift", mo.second}, {"y_zero", r.i(-128, 127)}, {"qmin", -128}, {"qmax", 127}};
      mb.node("I8Add", {"a", "b"}, {"y"}, at);
      mb.output("y");
      compare_stages(mb, {r.i8(n), r.i8(n)}, "I8Add", "i8add.neon");
    }
  }
  // I8AffineSum (per-channel and scalar coefficients, 1..5 inputs). Variants: 0 int32 coefficients;
  // 1 int16 coefficients (SMLAL path), zero points up to +-2000, clamped output, positive shift; 2 mixed
  // int16 / int32 rows; 3 coefficients too large for the folded-bias bound (general path), data bounded so
  // that the exact sum stays within int32.
  for (int64_t c : {1, 15, 16, 64, 100}) {
    for (int nin : {1, 2, 4, 5}) {
      for (int pc = 0; pc < 2; ++pc) {
        for (int variant = 0; variant < 4; ++variant) {
          const int64_t rows = 37;
          bftest::ModelBuilder mb;
          std::vector<std::string> ins;
          std::vector<std::vector<uint8_t>> data;
          Json::Array zs;
          for (int i = 0; i < nin; ++i) {
            mb.input("x" + std::to_string(i), DType::kI8, {rows, c});
            ins.push_back("x" + std::to_string(i));
            auto d = r.i8(rows * c);
            if (variant == 3)
              for (auto& v : d) v = static_cast<uint8_t>(r.i(-60, 60));
            data.push_back(d);
            const int64_t z = variant == 1 ? r.i(-2000, 2000) : (variant == 3 ? 0 : r.i(-128, 127));
            zs.push_back(Json(z));
          }
          const int64_t kc = pc ? c : 1;
          std::vector<int32_t> K(static_cast<size_t>(nin * kc)), B(static_cast<size_t>(kc));
          for (int i = 0; i < nin; ++i)
            for (int64_t j = 0; j < kc; ++j) {
              int32_t& v = K[i * kc + j];
              if (variant == 1 || (variant == 2 && i % 2 == 0)) v = r.i(-32768, 32767);
              else if (variant == 3) v = r.i(-(1 << 22), 1 << 22);
              else v = r.i(-(1 << 18), 1 << 18);
            }
          if (variant == 1) K[0] = -32768;  // int16 edge
          for (auto& v : B) v = variant == 1 ? r.i(-(1 << 20), 1 << 20) : r.i(-(1 << 30), 1 << 30);
          if (variant == 3)
            for (auto& v : B) v = r.i(-(1 << 26), 1 << 26);
          mb.constant("K", DType::kI32, {nin, kc}, K.data());
          mb.constant("B", DType::kI32, {kc}, B.data());
          ins.push_back("K");
          ins.push_back("B");
          auto mm = variant == 1 && c % 2 ? r.mult(1.5, 6.0) : r.mult(1e-6, 1e-4);
          if (variant == 1 && c % 2) mm = r.mult(1e-12, 1e-11);  // tiny with negative shift > 31
          Json at = Json::Object{{"zeros", Json(zs)}, {"mult", mm.first}, {"shift", mm.second}, {"y_zero", r.i(-128, 127)}};
          if (variant == 1 && !(c % 2)) {
            auto big = r.mult(1.5, 6.0);  // positive shift (SQSHL kept)
            at["mult"] = big.first;
            at["shift"] = big.second;
          }
          if (variant == 1) {
            at["qmin"] = -100;
            at["qmax"] = 90;
          }
          mb.node("I8AffineSum", ins, {"y"}, at);
          mb.output("y");
          compare_stages(mb, data, "I8AffineSum", "i8affinesum.neon");
        }
      }
    }
  }
  // I8Lut per-channel tables (last axis, and another axis), I8Transpose, I8ReduceMean, I8Concat, I8MatMul
  for (int ax : {-1, 1}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {2, 5, 7, 19});
    const int64_t rows = ax == -1 ? 19 : 5;
    auto tab = r.i8(rows * 256);
    mb.constant("t", DType::kI8, {rows, 256}, tab.data());
    mb.node("I8Lut", {"x", "t"}, {"y"}, Json::Object{{"axis", ax}});
    mb.output("y");
    compare_stages(mb, {r.i8(2 * 5 * 7 * 19)}, "I8Lut", ax == -1 ? "i8lut.neon_channels" : "i8lut.neon_rows");
  }
  // per-channel tables: transposed 16x16 TBL blocks (partial row blocks, overlapping channel tail, C < 16
  // scalar), every code of every table exercised
  for (int64_t rows : {1, 15, 16, 17, 40, 300}) {
    for (int64_t c : {3, 16, 19, 33, 64}) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {rows, c});
      auto tab = r.i8(c * 256);
      mb.constant("t", DType::kI8, {c, 256}, tab.data());
      mb.node("I8Lut", {"x", "t"}, {"y"});
      mb.output("y");
      auto x = r.i8(rows * c);
      if (rows == 300)
        for (int64_t i = 0; i < 256 * c; ++i) x[(i % 256) * c + i / 256] = static_cast<uint8_t>(i % 256);  // all codes
      compare_stages(mb, {x}, "I8Lut", "i8lut.neon_channels");
    }
  }
  for (int64_t inner : {5, 16, 19, 40}) {  // tables along a middle axis (TBL per block for inner >= 16)
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {2, 3, inner});
    auto tab = r.i8(3 * 256);
    mb.constant("t", DType::kI8, {3, 256}, tab.data());
    mb.node("I8Lut", {"x", "t"}, {"y"}, Json::Object{{"axis", 1}});
    mb.output("y");
    compare_stages(mb, {r.i8(6 * inner)}, "I8Lut", "i8lut.neon_rows");
  }
  for (auto perm : std::vector<std::vector<int64_t>>{{0, 2, 1, 3}, {0, 1, 3, 2}, {3, 2, 1, 0}, {1, 0, 2, 3}}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {3, 5, 7, 9});
    mb.node("I8Transpose", {"x"}, {"y"}, Json::Object{{"perm", ints(perm)}});
    mb.output("y");
    compare_stages(mb, {r.i8(3 * 5 * 7 * 9)}, "I8Transpose", "i8transpose.neon");
  }
  for (int64_t len : {1, 2, 4, 8, 15, 16, 53, 300}) {
    for (int rq = 0; rq < 4; ++rq) {
      const int64_t rows = rq >= 2 ? 77 : 6;  // 77: 64-byte steps of short rows plus a scalar tail
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {rows, len});
      Json at = Json::Object{{"axis", -1}, {"x_zero", r.i(-128, 127)}};
      if (rq % 2) {
        auto mm = r.mult(0.2 / len, 2.0 / len);
        at["mult"] = mm.first;
        at["shift"] = mm.second;
        at["y_zero"] = r.i(-128, 127);
      }
      mb.node("I8ReduceMean", {"x"}, {"y"}, at);
      mb.output("y");
      const bool neon = len >= 16 || len == 2 || len == 4 || len == 8;
      compare_stages(mb, {r.i8(rows * len)}, "I8ReduceMean", neon ? "i8reducemean.neon" : "");
    }
  }
  // I8ReduceMean on a middle axis and I8GlobalAvgPool (int16 partial sums flushed every 256 rows,
  // overlapping channel tail), requantized and rounding-division forms
  for (int64_t len : {1, 7, 300}) {
    for (int64_t inner : {16, 19, 40}) {
      for (int rq = 0; rq < 2; ++rq) {
        for (int gap = 0; gap < 2; ++gap) {
          bftest::ModelBuilder mb;
          Json at = Json::Object{{"x_zero", r.i(-128, 127)}};
          if (rq) {
            auto mm = r.mult(0.2 / len, 2.0 / len);
            at["mult"] = mm.first;
            at["shift"] = mm.second;
            at["y_zero"] = r.i(-128, 127);
          }
          if (gap) {
            mb.input("x", DType::kI8, {2, len, 1, inner});
            at["keepdims"] = static_cast<int64_t>(len % 2);
            mb.node("I8GlobalAvgPool", {"x"}, {"y"}, at);
          } else {
            mb.input("x", DType::kI8, {2, len, inner});
            at["axis"] = 1;
            mb.node("I8ReduceMean", {"x"}, {"y"}, at);
          }
          mb.output("y");
          auto x = r.i8(2 * len * inner);
          if (len == 300)
            for (int64_t i = 0; i < 256 * inner; ++i) x[i] = 0x80;  // -128 run: int16 partial sum at -32768
          compare_stages(mb, {x}, gap ? "I8GlobalAvgPool" : "I8ReduceMean", gap ? "i8globalavgpool.neon" : "i8reducemean.neon_mid");
        }
      }
    }
  }
  // I8Pad
  for (auto pads : std::vector<std::vector<int64_t>>{{0, 0, 1, 1, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 3, 5}, {1, 0, 0, 2, 1, 1, 2, 2}}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {2, 5, 7, 19});
    mb.node("I8Pad", {"x"}, {"y"}, Json::Object{{"pads", ints(pads)}, {"pad_q", r.i(-128, 127)}});
    mb.output("y");
    compare_stages(mb, {r.i8(2 * 5 * 7 * 19)}, "I8Pad", "i8pad.neon");
  }
  {
    bftest::ModelBuilder mb;
    mb.input("a", DType::kI8, {4, 3, 20});
    mb.input("b", DType::kI8, {4, 3, 37});
    auto mm = r.mult(0.3, 3.0);
    mb.node("I8Concat", {"a", "b"}, {"y"}, Json::Object{{"axis", -1}, {"y_zero", 5}, {"zeros", ints({0, r.i(-128, 127)})},
                                                        {"mults", ints({0, mm.first})}, {"shifts", ints({0, mm.second})}});
    mb.output("y");
    compare_stages(mb, {r.i8(4 * 3 * 20), r.i8(4 * 3 * 37)}, "I8Concat", "i8concat.neon");
  }
  for (int shape = 0; shape < 2; ++shape)
  for (int tb = 0; tb < 2; ++tb) {
    for (int f32 = 0; f32 < 2; ++f32) {
      const int64_t bt = shape ? 2 : 5, m = shape ? 20 : 53, k = shape ? 64 : 53, n = shape ? 48 : 21;
      bftest::ModelBuilder mb;
      mb.input("a", DType::kI8, {bt, m, k});
      mb.input("b", DType::kI8, tb ? Shape{bt, n, k} : Shape{bt, k, n});
      Json at = Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"transpose_b", tb}};
      if (f32) {
        at["output"] = "f32";
        at["scale"] = 0.001;
      } else {
        auto mm = r.mult(1e-5, 1e-3);
        at["mult"] = mm.first;
        at["shift"] = mm.second;
        at["y_zero"] = r.i(-128, 127);
      }
      mb.node("I8MatMul", {"a", "b"}, {"y"}, at);
      mb.output("y");
      compare_stages(mb, {r.i8(bt * m * k), r.i8(bt * k * n)}, "I8MatMul", "i8matmul.neon_sdot");
    }
  }
  {  // I8Add with a broadcast row (additive attention mask)
    bftest::ModelBuilder mb;
    mb.input("a", DType::kI8, {1, 3, 7, 64});
    mb.input("b", DType::kI8, {1, 1, 1, 64});
    auto ma = r.mult(0.05, 0.5), mbb = r.mult(0.05, 0.5), mo = r.mult(1e-7, 1e-4);
    mb.node("I8Add", {"a", "b"}, {"y"}, Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"left_shift", 20},
                                                     {"a_mult", ma.first}, {"a_shift", ma.second}, {"b_mult", mbb.first},
                                                     {"b_shift", mbb.second}, {"out_mult", mo.first}, {"out_shift", mo.second},
                                                     {"y_zero", r.i(-128, 127)}});
    mb.output("y");
    compare_stages(mb, {r.i8(3 * 7 * 64), r.i8(64)}, "I8Add", "i8add.neon");
  }
  // I8Mul (same shape and per-sample channel broadcast), pooling, depthwise conv
  for (int bc = 0; bc < 2; ++bc) {
    bftest::ModelBuilder mb;
    mb.input("a", DType::kI8, {2, 5, 7, 37});
    mb.input("b", DType::kI8, bc ? Shape{2, 1, 1, 37} : Shape{2, 5, 7, 37});
    auto mm = r.mult(1e-4, 4.0);
    mb.node("I8Mul", {"a", "b"}, {"y"}, Json::Object{{"a_zero", r.i(-128, 127)}, {"b_zero", r.i(-128, 127)}, {"mult", mm.first},
                                                     {"shift", mm.second}, {"y_zero", r.i(-128, 127)}});
    mb.output("y");
    compare_stages(mb, {r.i8(2 * 5 * 7 * 37), r.i8(bc ? 2 * 37 : 2 * 5 * 7 * 37)}, "I8Mul", "i8mul.neon");
  }
  for (const char* op : {"I8MaxPool2D", "I8AvgPool2D"}) {
    for (int inc = 0; inc < 2; ++inc) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {1, 9, 11, 35});
      mb.node(op, {"x"}, {"y"}, Json::Object{{"kernel", ints({3, 3})}, {"strides", ints({2, 2})}, {"pads", ints({1, 1, 1, 1})},
                                             {"x_zero", r.i(-128, 127)}, {"count_include_pad", inc}});
      mb.output("y");
      compare_stages(mb, {r.i8(9 * 11 * 35)}, op, std::string(op) == "I8MaxPool2D" ? "i8maxpool2d.neon" : "i8avgpool2d.neon");
    }
  }
  for (int64_t c : {16, 35, 64}) {
    for (int s = 1; s <= 2; ++s) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {1, 10, 9, c});
      auto w = r.i8(c * 9);
      for (auto& v : w) v = static_cast<uint8_t>(std::max<int>(-127, static_cast<int8_t>(v)));
      mb.constant("w", DType::kI8, {c, 3, 3, 1}, w.data());
      std::vector<int32_t> b(c), m(c), sh(c);
      for (int64_t o = 0; o < c; ++o) {
        b[o] = r.i(-5000, 5000);
        auto mm = r.mult(1e-4, 1e-2);
        m[o] = mm.first;
        sh[o] = mm.second;
      }
      mb.constant("b", DType::kI32, {c}, b.data());
      mb.constant("m", DType::kI32, {c}, m.data());
      mb.constant("s", DType::kI32, {c}, sh.data());
      mb.node("I8Conv2D", {"x", "w", "b", "m", "s"}, {"y"},
              Json::Object{{"strides", ints({s, s})}, {"pads", ints({1, 1, 1, 1})}, {"groups", c}, {"x_zero", r.i(-128, 127)},
                           {"pad_q", r.i(-128, 127)}, {"y_zero", r.i(-20, 20)}});
      mb.output("y");
      compare_stages(mb, {r.i8(10 * 9 * c)}, "I8Conv2D", "i8conv2d.neon_depthwise");
    }
  }
  // I8LayerNorm
  for (int64_t d : {1, 7, 16, 64, 100, 192, 768}) {
    for (int64_t rows : {1, 3, 198}) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {rows, d});
      auto g = r.f32(d, -50.f, 50.f), b = r.f32(d, -60.f, 60.f);
      mb.constant("g", DType::kF32, {d}, g.data());
      mb.constant("b", DType::kF32, {d}, b.data());
      mb.node("I8LayerNorm", {"x", "g", "b"}, {"y"},
              Json::Object{{"x_zero", r.i(-128, 127)}, {"x_scale", static_cast<double>(r.f(0.01f, 2.f))}, {"eps", 1e-5}});
      mb.output("y");
      auto x = r.i8(rows * d);
      if (rows > 1) std::fill_n(x.begin(), d, x[0]);  // constant row: variance 0 (eps only)
      compare_stages(mb, {x}, "I8LayerNorm", "i8layernorm.neon");
    }
  }
  // I8Softmax (i8 and f32 output, large zero points of binarize-aware grids)
  for (int64_t n : {1, 5, 16, 53, 198, 300}) {
    for (int mode = 0; mode < 3; ++mode) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {4, n});
      const double sx = r.f(0.01f, 0.3f);
      const int bits = std::min(22, 30 - static_cast<int>(std::ceil(std::log2(std::max<int64_t>(n, 2)))));
      std::vector<int32_t> tab(256);
      for (int d = 0; d < 256; ++d) tab[d] = static_cast<int32_t>(std::llround(std::ldexp(std::exp(-sx * d), bits)));
      mb.constant("e", DType::kI32, {256}, tab.data());
      Json at = Json::Object{};
      if (mode == 0) at = Json::Object{{"inv_scale", 255.0}, {"y_zero", -128}};
      if (mode == 1) at = Json::Object{{"inv_scale", 1.0 / 7.8e-9}, {"y_zero", -617000}};  // grid around a threshold 0.0048
      if (mode == 2) at = Json::Object{{"output", "f32"}};
      mb.node("I8Softmax", {"x", "e"}, {"y"}, at);
      mb.output("y");
      compare_stages(mb, {r.i8(4 * n)}, "I8Softmax", "i8softmax.neon");
    }
  }
  // I8Conv2D / I8Dense (SDOT GEMM), i8 and f32 output
  struct C { int64_t h, w, ic, oc, k, s, pad; };
  for (C c : std::vector<C>{{14, 14, 3, 8, 3, 1, 1}, {32, 32, 3, 19, 16, 16, 0}, {9, 7, 12, 33, 3, 2, 1}, {5, 5, 64, 64, 1, 1, 0}}) {
    for (int f32 = 0; f32 < 2; ++f32) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {1, c.h, c.w, c.ic});
      auto w = r.i8(c.oc * c.k * c.k * c.ic);
      for (auto& v : w) v = static_cast<uint8_t>(std::max<int>(-127, static_cast<int8_t>(v)));
      mb.constant("w", DType::kI8, {c.oc, c.k, c.k, c.ic}, w.data());
      Json at = Json::Object{{"strides", ints({c.s, c.s})}, {"pads", ints({c.pad, c.pad, c.pad, c.pad})}, {"x_zero", r.i(-128, 127)},
                             {"pad_q", r.i(-128, 127)}};
      std::vector<std::string> ins = {"x", "w", "b", "m"};
      if (f32) {
        auto b = r.f32(c.oc, -1.f, 1.f), sc = r.f32(c.oc, 1e-4f, 1e-2f);
        mb.constant("b", DType::kF32, {c.oc}, b.data());
        mb.constant("m", DType::kF32, {c.oc}, sc.data());
        at["output"] = "f32";
      } else {
        std::vector<int32_t> b(c.oc), m(c.oc), sh(c.oc);
        for (int64_t o = 0; o < c.oc; ++o) {
          b[o] = r.i(-100000, 100000);
          auto mm = r.mult(1e-5, 1e-2);
          m[o] = mm.first;
          sh[o] = mm.second;
        }
        mb.constant("b", DType::kI32, {c.oc}, b.data());
        mb.constant("m", DType::kI32, {c.oc}, m.data());
        mb.constant("s", DType::kI32, {c.oc}, sh.data());
        ins.push_back("s");
        at["y_zero"] = r.i(-20, 20);
      }
      mb.node("I8Conv2D", ins, {"y"}, at);
      mb.output("y");
      compare_stages(mb, {r.i8(c.h * c.w * c.ic)}, "I8Conv2D", "i8conv2d.neon_sdot");
    }
  }
  for (int64_t rows : {1, 7, 8, 9, 64}) {
    for (int64_t k : {1, 5, 192, 384}) {
      for (int64_t o : {1, 3, 8, 13, 100}) {
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {rows, k});
        auto w = r.i8(o * k);
        mb.constant("w", DType::kI8, {o, k}, w.data());
        auto b = r.f32(o, -1.f, 1.f), sc = r.f32(o, 1e-4f, 1e-2f);
        mb.constant("b", DType::kF32, {o}, b.data());
        mb.constant("m", DType::kF32, {o}, sc.data());
        mb.node("I8Dense", {"x", "w", "b", "m"}, {"y"}, Json::Object{{"x_zero", r.i(-128, 127)}, {"output", "f32"}});
        mb.output("y");
        compare_stages(mb, {r.i8(rows * k)}, "I8Dense", "i8dense.neon_sdot");
      }
    }
  }
  // I8BDense (ge / gt / sign_zero, scalar and per-channel thresholds, w_zero) and I8BConv2D
  for (const char* mode : {"ge", "gt", "sign_zero"}) {
    for (int64_t k : {7, 64, 100, 192, 768}) {
      for (int pc = 0; pc < 3; ++pc) {  // 2: per-channel thresholds all within the int8 grid (int8 compares), f32 output
        const int64_t rows = 37, o = 21;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {rows, k});
        std::vector<int8_t> wv(static_cast<size_t>(o * k));
        for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
        auto wp = bftest::pack_pm1(wv, o, k);
        mb.constant("w", DType::kBit, {o, k}, wp.data());
        auto pm = r.f32(o, -3.f, 3.f), pa = r.f32(o, -100.f, 100.f);
        mb.constant("pm", DType::kF32, {o}, pm.data());
        mb.constant("pa", DType::kF32, {o}, pa.data());
        std::vector<std::string> ins = {"x", "w", "pm", "pa"};
        const float s = r.f(0.001f, 0.1f);
        const int z = r.i(-30, 30);
        if (pc) {
          auto t = r.f32(k, -100 * s, 100 * s);
          for (int64_t i = 0; i < k; i += 3) t[i] = q8::deq(r.i(-128, 127), s, z);  // ties
          t[0] = pc == 1 ? 1e9f : -1e9f;  // never (T = 128) / always (T = -128)
          mb.constant("t", DType::kF32, {k}, t.data());
          ins.push_back("t");
        } else {
          ins.push_back("");
        }
        if (std::string(mode) == "sign_zero" && k == 192) {
          std::vector<int8_t> zv(static_cast<size_t>(o * k));
          for (auto& v : zv) v = r.i(0, 9) == 0 ? -1 : 1;
          auto zp = bftest::pack_pm1(zv, o, k);
          mb.constant("wz", DType::kBit, {o, k}, zp.data());
          ins.push_back("wz");
        }
        while (!ins.empty() && ins.back().empty()) ins.pop_back();
        Json at = Json::Object{{"input_binarize", mode}, {"x_scale", static_cast<double>(s)}, {"x_zero", z}, {"qmin", -100}, {"qmax", 120}};
        if (pc == 2) at["output"] = "f32";
        mb.node("I8BDense", ins, {"y"}, at);
        mb.output("y");
        compare_stages(mb, {r.i8(rows * k)}, "I8BDense", "i8bdense.neon");
      }
    }
  }
  for (const char* mode : {"ge", "gt"}) {
    for (int64_t ic : {3, 32, 70}) {
      const int64_t h = 9, w = 11, oc = 10;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {1, h, w, ic});
      std::vector<int8_t> wv(static_cast<size_t>(oc * 9 * ic));
      for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
      auto wp = bftest::pack_pm1(wv, oc * 9, ic);
      mb.constant("w", DType::kBit, {oc, 3, 3, ic}, wp.data());
      auto pm = r.f32(oc, -3.f, 3.f), pa = r.f32(oc, -10.f, 10.f), t = r.f32(ic, -1.f, 1.f);
      mb.constant("pm", DType::kF32, {oc}, pm.data());
      mb.constant("pa", DType::kF32, {oc}, pa.data());
      mb.constant("t", DType::kF32, {ic}, t.data());
      mb.node("I8BConv2D", {"x", "w", "pm", "pa", "t"}, {"y"},
              Json::Object{{"input_binarize", mode}, {"x_scale", 0.02}, {"x_zero", 3}, {"pads", ints({1, 1, 1, 1})}, {"pad_value", 1}});
      mb.output("y");
      compare_stages(mb, {r.i8(h * w * ic)}, "I8BConv2D", "i8bconv2d.neon");
    }
  }
  // I8BMatMul: modes, transpose_b, coefficient combinations (incl. c_ab = 0), thresholds, per-column mul
  for (const char* ma : {"sign_zero", "ge", "gt0"}) {
    for (int tb = 0; tb < 2; ++tb) {
      for (int coef = 0; coef < 3; ++coef) {
        const int64_t bt = 3, m = 19, k = 64 + coef * 70, n = 21;
        bftest::ModelBuilder mb;
        mb.input("a", DType::kI8, {bt, m, k});
        mb.input("b", DType::kI8, tb ? Shape{bt, n, k} : Shape{bt, k, n});
        auto mul = r.f32(coef == 2 ? n : 1, -2.f, 2.f), add = r.f32(1, -5.f, 5.f);
        mb.constant("mul", DType::kF32, {static_cast<int64_t>(mul.size())}, mul.data());
        mb.constant("add", DType::kF32, {1}, add.data());
        Json at = Json::Object{{"a_binarize", ma}, {"b_binarize", "sign_zero"}, {"transpose_b", tb}, {"a_scale", 0.01}, {"a_zero", -5},
                               {"a_threshold", 0.05}, {"b_scale", 0.2}, {"b_zero", 1}, {"c_ab", coef == 1 ? 0 : -1},
                               {"c_rs", coef ? 2 : 0}, {"c_cs", coef ? -1 : 0}, {"c0", coef ? 7 : 0}};
        mb.node("I8BMatMul", {"a", "b", "mul", "add"}, {"y"}, at);
        mb.output("y");
        compare_stages(mb, {r.i8(bt * m * k), r.i8(bt * k * n)}, "I8BMatMul", "i8bmatmul.neon");
      }
    }
  }
  EXPECT_TRUE(g_compared > 150);
}

// I8Conv2D / I8Dense on the int32 SDOT GEMM: K > 1032 (formerly the reference kernel), grouped convolutions,
// dilation, in-place rows (1x1 / Dense with K % 16 == 0), positive and negative requantization shifts,
// clamped outputs, f32 outputs, partial row tiles and channel tails.
BF_TEST(int8_full_gemm_general_bit_exact_vs_reference) {
  Rng r(77);
  struct C { int64_t h, w, ic, oc, kh, kw, s, d, pad, groups; };
  for (C c : std::vector<C>{{6, 5, 512, 21, 3, 3, 1, 1, 1, 1},   // K = 4608
                            {8, 8, 300, 40, 2, 2, 2, 1, 0, 1},   // K = 1200, stride 2
                            {9, 9, 160, 17, 3, 3, 1, 2, 2, 1},   // dilation 2, K = 1440
                            {7, 6, 8, 32, 3, 3, 1, 1, 1, 4},     // groups 4, kic 2
                            {7, 6, 32, 64, 3, 3, 2, 1, 1, 8},    // groups 8, kic 4
                            {5, 7, 64, 48, 3, 3, 1, 1, 1, 2},    // groups 2, kic 32
                            {6, 6, 64, 35, 1, 1, 1, 1, 0, 1},    // 1x1 in place
                            {6, 6, 48, 16, 1, 1, 2, 1, 0, 1}}) { // 1x1 stride 2
    for (int f32 = 0; f32 < 2; ++f32) {
      for (int lsh = 0; lsh < 2; ++lsh) {
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {2, c.h, c.w, c.ic});
        const int64_t kic = c.ic / c.groups, k = c.kh * c.kw * kic;
        auto w = r.i8(c.oc * k);
        for (auto& v : w) v = static_cast<uint8_t>(std::max<int>(-127, static_cast<int8_t>(v)));
        mb.constant("w", DType::kI8, {c.oc, c.kh, c.kw, kic}, w.data());
        Json at = Json::Object{{"strides", ints({c.s, c.s})}, {"dilations", ints({c.d, c.d})}, {"pads", ints({c.pad, c.pad, c.pad, c.pad})},
                               {"groups", c.groups}, {"x_zero", r.i(-128, 127)}, {"pad_q", r.i(-128, 127)}};
        std::vector<std::string> ins = {"x", "w", "b", "m"};
        if (f32) {
          auto b = r.f32(c.oc, -1.f, 1.f), sc = r.f32(c.oc, 1e-4f, 1e-2f);
          mb.constant("b", DType::kF32, {c.oc}, b.data());
          mb.constant("m", DType::kF32, {c.oc}, sc.data());
          at["output"] = "f32";
        } else {
          std::vector<int32_t> b(c.oc), m(c.oc), sh(c.oc);
          for (int64_t o = 0; o < c.oc; ++o) {
            b[o] = r.i(-1000000, 1000000);
            auto mm = lsh && o % 3 == 0 ? r.mult(1.0, 8.0) : r.mult(1e-6, 1e-2);
            m[o] = mm.first;
            sh[o] = mm.second;
          }
          mb.constant("b", DType::kI32, {c.oc}, b.data());
          mb.constant("m", DType::kI32, {c.oc}, m.data());
          mb.constant("s", DType::kI32, {c.oc}, sh.data());
          ins.push_back("s");
          const int zy = r.i(-20, 20);
          at["y_zero"] = zy;
          if (lsh) at["qmin"] = zy;  // fused relu
        }
        mb.node("I8Conv2D", ins, {"y"}, at);
        mb.output("y");
        compare_stages(mb, {r.i8(2 * c.h * c.w * c.ic)}, "I8Conv2D", "i8conv2d.neon_sdot");
      }
    }
  }
  for (int64_t rows : {1, 3, 8, 21}) {
    for (int64_t k : {64, 1040, 2048}) {
      for (int f32 = 0; f32 < 2; ++f32) {
        const int64_t o = 37;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {rows, k});
        auto w = r.i8(o * k);
        mb.constant("w", DType::kI8, {o, k}, w.data());
        Json at = Json::Object{{"x_zero", r.i(-128, 127)}};
        std::vector<std::string> ins = {"x", "w", "b", "m"};
        if (f32) {
          auto b = r.f32(o, -1.f, 1.f), sc = r.f32(o, 1e-4f, 1e-2f);
          mb.constant("b", DType::kF32, {o}, b.data());
          mb.constant("m", DType::kF32, {o}, sc.data());
          at["output"] = "f32";
        } else {
          std::vector<int32_t> b(o), m(o), sh(o);
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
        compare_stages(mb, {r.i8(rows * k)}, "I8Dense", "i8dense.neon_sdot");
      }
    }
  }
}


// I8BMatMul on bmatmul.neon's int8 operand mode: every binarization mode for a and b, thresholds at
// ties and outside the grid (T = 128: empty upper set, T = -128), k with partial words, partial column
// blocks, both b layouts, the plain path (c_ab = 1 only) and the general combination, i8 / f32 output.
BF_TEST(int8_full_bmatmul_int8_operands_bit_exact_vs_reference) {
  Rng r(4242);
  const char* modes[] = {"sign_zero", "ge", "gt", "gt0"};
  for (int ia = 0; ia < 4; ++ia) {
    for (int ib = 0; ib < 4; ++ib) {
      for (int tb = 0; tb < 2; ++tb) {
        for (int var = 0; var < 3; ++var) {
          const int64_t bt = 2, m = 9 + var, k = var == 0 ? 37 : (var == 1 ? 64 : 100), n = var == 2 ? 6 : (var == 0 ? 12 : 19);
          bftest::ModelBuilder mb;
          mb.input("a", DType::kI8, {bt, m, k});
          mb.input("b", DType::kI8, tb ? Shape{bt, n, k} : Shape{bt, k, n});
          // per-column mul / add (var 1; the int8 tile epilogue reads them padded to 8-column blocks)
          auto mul = r.f32(var == 1 ? n : 1, -2.f, 2.f), add = r.f32(var == 1 ? n : 1, -5.f, 5.f);
          mb.constant("mul", DType::kF32, {static_cast<int64_t>(mul.size())}, mul.data());
          mb.constant("add", DType::kF32, {static_cast<int64_t>(add.size())}, add.data());
          const double sa = 0.01, sb = 0.02;
          const int za = r.i(-20, 20), zb = r.i(-20, 20);
          // thresholds: a tie (deq of a code), above the grid (T = 128), below it (T = -128)
          const double ta = var == 0 ? q8::deq(r.i(-100, 100), static_cast<float>(sa), za) : (var == 1 ? 50.0 : -50.0);
          const double tbv = var == 2 ? q8::deq(r.i(-100, 100), static_cast<float>(sb), zb) : (var == 0 ? -50.0 : 0.3);
          Json at = Json::Object{{"a_binarize", modes[ia]}, {"b_binarize", modes[ib]}, {"transpose_b", tb}, {"a_scale", sa},
                                 {"a_zero", za}, {"a_threshold", ta}, {"b_scale", sb}, {"b_zero", zb}, {"b_threshold", tbv},
                                 {"c_ab", 1}, {"c_rs", var == 1 ? 3 : 0}, {"c_cs", var == 1 ? -2 : 0}, {"c0", var == 1 ? 5 : 0}};
          if (var == 2) at["output"] = "f32";
          if (var == 0) {  // clamped int8 output
            at["qmin"] = -60;
            at["qmax"] = 90;
          }
          mb.node("I8BMatMul", {"a", "b", "mul", "add"}, {"y"}, at);
          mb.output("y");
          compare_stages(mb, {r.i8(bt * m * k), r.i8(bt * k * n)}, "I8BMatMul", "i8bmatmul.neon");
        }
      }
    }
  }
}


// I8BConv2D / I8BDense with the int8 output written by the binary kernels (bconv2d.neon_opt full / pair /
// partial tiles and channel tails, bdense.neon tiled / pair / GEMV rows, bdense.ternary_neon reading the
// codes): clamped outputs (fused relu / relu6 ranges), f32 output, grouped and strided convolutions,
// pad values 0 / 1, scalar and per-channel thresholds incl. ties and thresholds outside the grid, batch 2.
BF_TEST(int8_full_binary_int8_io_bit_exact_vs_reference) {
  Rng r(9090);
  struct C { int64_t h, w, ic, oc, k, s, groups, pad_value; };
  int idx = 0;
  // dot2 = 1: BITINO_BCONV_DOT2=0 (the single-phase epilogue compute_tile_a instead of compute_dot2)
  for (int dot2_off = 0; dot2_off < 2; ++dot2_off) {
  if (dot2_off) setenv("BITINO_BCONV_DOT2", "0", 1);
  for (C c : std::vector<C>{{9, 11, 32, 10, 3, 1, 1, 0},   // pair mode (k_words 9), partial channel block
                            {8, 7, 64, 21, 3, 2, 1, 1},    // stride 2, pad value 1
                            {6, 6, 64, 16, 3, 1, 2, 0},    // groups 2 (kic 32)
                            {6, 5, 64, 32, 3, 1, 4, 0},    // groups 4 (kic 16, 16-bit units)
                            {7, 7, 130, 12, 1, 1, 1, 0},   // 1x1, K tail
                            {9, 11, 64, 48, 3, 1, 1, 0},   // pair, 3 whole 16-channel units, border corrections
                            {10, 9, 128, 40, 3, 1, 1, 1},  // 2 units + an 8-channel tail, pad value 1
                            {11, 7, 96, 36, 3, 2, 1, 0},   // stride 2, 2 units + a 4-channel tail
                            {5, 6, 256, 64, 1, 1, 1, 0}}) { // 1x1, odd pixel count
    for (int variant = 0; variant < 3; ++variant, ++idx) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {2, c.h, c.w, c.ic});
      const int64_t kic = c.ic / c.groups;
      std::vector<int8_t> wv(static_cast<size_t>(c.oc * c.k * c.k * kic));
      for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
      auto wp = bftest::pack_pm1(wv, c.oc * c.k * c.k, kic);
      mb.constant("w", DType::kBit, {c.oc, c.k, c.k, kic}, wp.data());
      auto pm = r.f32(c.oc, -3.f, 3.f), pa = r.f32(c.oc, -40.f, 40.f);
      mb.constant("pm", DType::kF32, {c.oc}, pm.data());
      mb.constant("pa", DType::kF32, {c.oc}, pa.data());
      const float s = 0.02f;
      const int z = r.i(-10, 10);
      std::vector<std::string> ins = {"x", "w", "pm", "pa"};
      if (variant != 1) {
        auto t = r.f32(c.ic, -1.f, 1.f);
        for (int64_t i = 0; i < c.ic; i += 4) t[i] = q8::deq(r.i(-128, 127), s, z);  // ties
        if (variant == 2) t[1] = 1e9f;  // T = 128 (never +1): the int16 compare path
        mb.constant("t", DType::kF32, {c.ic}, t.data());
        ins.push_back("t");
      }
      Json at = Json::Object{{"input_binarize", idx % 2 ? "gt" : "ge"}, {"x_scale", static_cast<double>(s)}, {"x_zero", z},
                             {"strides", ints({c.s, c.s})}, {"pads", ints({c.k / 2, c.k / 2, c.k / 2, c.k / 2})},
                             {"pad_value", c.pad_value}, {"groups", c.groups}};
      if (variant == 0) {
        at["qmin"] = -5;  // relu-like range
      } else if (variant == 1) {
        at["qmin"] = -128;
        at["qmax"] = 40;  // relu6-like upper clamp
      } else {
        at["output"] = "f32";
      }
      mb.node("I8BConv2D", ins, {"y"}, at);
      mb.output("y");
      compare_stages(mb, {r.i8(2 * c.h * c.w * c.ic)}, "I8BConv2D", "i8bconv2d.neon");
    }
  }
  if (dot2_off) unsetenv("BITINO_BCONV_DOT2");
  }
  // BDense: GEMV-like (rows < 8), tiled, pair mode (K = 64, 192), ternary with clamps
  for (const char* mode : {"ge", "sign_zero"}) {
    for (int64_t rows : {1, 3, 8, 13}) {
      for (int64_t k : {64, 192, 300}) {
        const int64_t o = 18;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {rows, k});
        std::vector<int8_t> wv(static_cast<size_t>(o * k));
        for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
        auto wp = bftest::pack_pm1(wv, o, k);
        mb.constant("w", DType::kBit, {o, k}, wp.data());
        auto pm = r.f32(o, -3.f, 3.f), pa = r.f32(o, -60.f, 60.f);
        mb.constant("pm", DType::kF32, {o}, pm.data());
        mb.constant("pa", DType::kF32, {o}, pa.data());
        const float s = 0.03f;
        const int z = r.i(-20, 20);
        Json at = Json::Object{{"input_binarize", mode}, {"x_scale", static_cast<double>(s)}, {"x_zero", z},
                               {"qmin", rows % 2 ? z : -128}, {"qmax", rows % 2 ? 127 : 60}};
        mb.node("I8BDense", {"x", "w", "pm", "pa"}, {"y"}, at);
        mb.output("y");
        compare_stages(mb, {r.i8(rows * k)}, "I8BDense", "i8bdense.neon");
      }
    }
  }
}

}  // namespace

// ======================================================= 8-bit log2 code (l8) and the fused residual blocks
namespace {

// Independent definitions: decode = fp32 nearest of 2^(e + (j - 127)/8) (long double); encode = number of the snapped
// rounding boundaries 2^o (1 + B_i/128) (exact in fp32) that are <= |w|, mapped to the code.
float l8_decode_def(int c, int e) {
  if (c == 0) return 0.f;
  const int j = std::min(std::abs(c), 127);
  const float v = static_cast<float>(std::exp2l(static_cast<long double>(j - 127) / 8.0L + e));
  return c < 0 ? -v : v;
}
int l8_encode_def(float w) {
  const float a = std::fabs(w);
  // boundaries of octave o (o = -16 .. 0) separate levels 8 (o + 16) + i - 1 and 8 (o + 16) + i (level j: 2^((j - 127)/8),
  // level 0: the zero code); boundaries of lower octaves only separate zero codes, of higher ones only saturated codes
  int n = 0;
  for (int o = -16; o <= 0; ++o)
    for (int b : q8::kL8B) n += a >= std::ldexp(1.f + static_cast<float>(b) / 128.f, o);
  const int j = std::min(std::max(n - 1, 0), 127);
  return std::signbit(w) ? -j : j;
}

BF_TEST(int8_full_l8_code_matches_definition) {
  for (int e = -100; e <= 100; ++e)
    for (int c = -128; c <= 127; ++c) {
      const float d = q8::l8_decode(c, e), ref = l8_decode_def(c, e);
      if (std::memcmp(&d, &ref, 4) != 0) throw bftest::Failure{"l8_decode differs at c=" + std::to_string(c) + " e=" + std::to_string(e)};
      if (c > -128 && q8::l8_decode(c - 1, e) > d) throw bftest::Failure{"l8_decode not monotonic"};
    }
  for (int c = -128; c <= 127; ++c) EXPECT_EQ(static_cast<int>(q8::l8_encode(q8::l8_decode(c, 0))), c == -128 ? -127 : c);
  Rng r(77);
  int64_t n = 0;
  auto check = [&](float w) {
    if (static_cast<int>(q8::l8_encode(w)) != l8_encode_def(w))
      throw bftest::Failure{"l8_encode differs from the definition at " + std::to_string(w)};
    ++n;
  };
  for (int o = -20; o <= 3; ++o)
    for (int b : q8::kL8B)
      for (float sg : {1.f, -1.f}) {
        const float x = sg * std::ldexp(1.f + static_cast<float>(b) / 128.f, o);
        check(x);
        check(std::nextafter(x, 0.f));
        check(std::nextafter(x, 2 * x));
      }
  for (float x : {0.f, -0.f, 1e-45f, -1e-38f, 1.f, -1.f, 1.0442f, 2.f, 1e30f, -1e30f}) check(x);
  for (int i = 0; i < 400000; ++i) check(std::ldexp(r.f(-2.f, 2.f), r.i(-22, 4)));
  // nearest level in the log domain up to the snapping of the boundaries (< 0.0042 in log2)
  for (int i = 0; i < 100000; ++i) {
    const float w = std::ldexp(r.f(1.f, 2.f), r.i(-15, -1));
    const float v = q8::l8_decode(q8::l8_encode(w), 0);
    EXPECT_TRUE(std::fabs(std::log2(static_cast<double>(v) / w)) <= 1.0 / 16 + 0.0042);
  }
  EXPECT_TRUE(n > 400000);
}

#if defined(__aarch64__)
BF_TEST(int8_full_l8_neon_matches_scalar) {
  const l8neon::EncTables et;
  Rng r(78);
  alignas(16) float w[16];
  alignas(16) int8_t got[16];
  auto run = [&]() {
    vst1q_s8(got, l8neon::encode16(vld1q_f32(w), vld1q_f32(w + 4), vld1q_f32(w + 8), vld1q_f32(w + 12), et));
    for (int k = 0; k < 16; ++k)
      if (got[k] != q8::l8_encode(w[k])) throw bftest::Failure{"encode16 differs at " + std::to_string(w[k])};
  };
  for (int it = 0; it < (1 << 20); ++it) {
    for (int k = 0; k < 16; ++k) {
      const uint32_t b = static_cast<uint32_t>(r.g());
      std::memcpy(&w[k], &b, 4);
      if (std::isnan(w[k])) w[k] = 0.f;
    }
    run();
  }
  for (int o = -20; o <= 3; ++o)  // boundaries and their neighbours
    for (int b = 0; b < 8; ++b) {
      const float x = std::ldexp(1.f + static_cast<float>(q8::kL8B[b]) / 128.f, o);
      const float v[16] = {x, -x, std::nextafter(x, 0.f), std::nextafter(x, 2 * x), -std::nextafter(x, 0.f), 0.f, -0.f, 1e-45f,
                           std::ldexp(1.f, o), -std::ldexp(1.f, o), 3e38f, -3e38f, 1.f, 2.f, 0.5f, -1e-40f};
      std::memcpy(w, v, sizeof v);
      run();
    }
  alignas(16) int8_t c[16];
  alignas(16) float d[16];
  for (int e = -110; e <= 100; ++e) {
    const l8neon::DecTables dt(e);
    for (int base = -128; base < 128; base += 16) {
      for (int k = 0; k < 16; ++k) c[k] = static_cast<int8_t>(base + k);
      float32x4_t dv[4];
      l8neon::decode16(vld1q_s8(c), dt, dv);
      for (int k = 0; k < 4; ++k) vst1q_f32(d + 4 * k, dv[k]);
      for (int k = 0; k < 16; ++k) {
        const float ref = q8::l8_decode(c[k], e);
        if (std::memcmp(&d[k], &ref, 4) != 0)
          throw bftest::Failure{"decode16 differs at c=" + std::to_string(c[k]) + " e=" + std::to_string(e)};
      }
    }
  }
}
#endif

bool g_no_fuse = false;  // run_model_all: SessionOptions::fuse off (no Session rewrites)

// All graph outputs (bytes) of a model at the given tiers.
std::vector<uint8_t> run_model_all(const bftest::ModelBuilder& mb, const std::string& stage_tiers, int threads,
                                   const std::vector<std::vector<uint8_t>>& inputs, std::map<std::string, std::string>* kernels = nullptr) {
  auto model = mb.build();
  SessionOptions so;
  so.num_threads = threads;
  so.spin_run = threads > 1;  // multi-threaded runs also exercise the hot thread-pool mode
  so.fuse = !g_no_fuse;
  std::set<std::string> tiers;
  size_t p = 0;
  while (p <= stage_tiers.size()) {
    size_t q = stage_tiers.find(',', p);
    if (q == std::string::npos) q = stage_tiers.size();
    tiers.insert(stage_tiers.substr(p, q - p));
    p = q + 1;
  }
  so.policy = KernelPolicy::tiers(tiers);
  Session s(model, so);
  for (size_t i = 0; i < inputs.size(); ++i) std::memcpy(s.input(i).raw(), inputs[i].data(), inputs[i].size());
  s.run();
  if (kernels) {
    *kernels = s.kernel_choices();
    (*kernels)["#absorbed_luts"] = std::to_string(s.absorbed_luts());
  }
  std::vector<uint8_t> out;
  for (size_t i = 0; i < s.num_outputs(); ++i) {
    const Tensor& y = s.output(i);
    const auto* b = static_cast<const uint8_t*>(y.raw());
    out.insert(out.end(), b, b + y.bytes());
  }
  return out;
}

#if BF_NEON_A32
int g_a32_res_neon = 0;  // AArch32: compare_all runs that selected i8bconv2dres.neon
#endif

void compare_all(const bftest::ModelBuilder& mb, const std::vector<std::vector<uint8_t>>& in, const std::string& what,
                 const std::string& neon_kernel) {
  auto ref = run_model_all(mb, kRef, 1, in);
  for (int t : {1, 4}) {
    std::map<std::string, std::string> ks;
    auto got = run_model_all(mb, kFull, t, in, &ks);
    if (got != ref) throw bftest::Failure{"int8_full output differs from the reference: " + what};
#if defined(__aarch64__)
    bool found = neon_kernel.empty();
    for (const auto& kv : ks) found = found || kv.second == neon_kernel;
    if (!found) throw bftest::Failure{"expected kernel " + neon_kernel + " for " + what};
#elif BF_NEON_A32
    // AArch32: I8BConv2DRes runs the NEON epilogue when no value can be denormal (int8_full::res_epilogue_normal; the
    // configurations with tiny scales or gt thresholds 0 stay on the portable kernel): counted, see a32_res_neon_runs
    for (const auto& kv : ks) g_a32_res_neon += kv.second == neon_kernel && neon_kernel == "i8bconv2dres.neon";
#endif
  }
}

// FtzBound (int8_full.h, 2026-11-13): the proof that no fp32 value of an epilogue chain is denormal, which lets the
// AArch32 (flush-to-zero) NEON epilogues of I8BConv2DRes / I8Conv2D l8 run. Checked against brute force on small
// cases: every value the bound admits is 0 or normal, and the bound rejects chains that do reach a denormal.
BF_TEST(int8_full_ftz_bound) {
  using int8_full::FtzBound;
  const double kMin = FtzBound::kMinNormal;
  // integer * c + d over |acc| <= 4096: every value 0 or normal whenever ok
  Rng r(77);
  int accepted = 0, rejected = 0;
  for (int t = 0; t < 400; ++t) {
    const float c = std::ldexp(r.f(1.f, 2.f), r.i(-140, -90)) * (r.i(0, 1) ? 1.f : -1.f);
    const float d = t % 3 == 0 ? 0.f : std::ldexp(r.f(1.f, 2.f), r.i(-140, -90));
    const float s = std::ldexp(r.f(0.1f, 1.f), r.i(-30, 0));
    FtzBound b;
    b.mul_const(&c, 1);
    b.add_const(&d, 1);
    b.prelu(&s, 1);
    bool any_denormal = false;
#if defined(__clang__)
#pragma clang loop vectorize(disable)  // IEEE scalar arithmetic (AArch32 NEON would flush the denormals it looks for)
#endif
    for (int acc = -4096; acc <= 4096; ++acc) {
      float v = static_cast<float>(acc) * c;
      any_denormal = any_denormal || (v != 0.f && std::fabs(v) < kMin);
      v = v + d;
      any_denormal = any_denormal || (v != 0.f && std::fabs(v) < kMin);
      if (v < 0.f) v = v * s;
      any_denormal = any_denormal || (v != 0.f && std::fabs(v) < kMin);
    }
    if (b.ok) {
      ++accepted;
      if (any_denormal) throw bftest::Failure{"FtzBound accepted a chain with a denormal value (t " + std::to_string(t) + ")"};
    } else {
      ++rejected;
    }
  }
  if (accepted == 0 || rejected == 0) throw bftest::Failure{"FtzBound test cases not mixed"};
  // explicit rejections: denormal constant, product below 2^-126, sum granularity below 2^-126
  FtzBound a;
  const float den = 1e-39f, tiny = 1e-20f, g = std::ldexp(1.f, -110);
  a.mul_const(&den, 1);
  if (a.ok) throw bftest::Failure{"FtzBound: denormal constant accepted"};
  FtzBound b;
  b.mul_const(&tiny, 1);
  b.mul_const(&tiny, 1);
  if (b.ok) throw bftest::Failure{"FtzBound: 1e-40 product accepted"};
  FtzBound c;
  c.mul_const(&g, 1);  // L = 2^-110, G = 2^-133: a sum with another value can cancel to a denormal
  FtzBound one;
  c.add(one);
  if (c.ok) throw bftest::Failure{"FtzBound: sum granularity 2^-133 accepted"};
}

struct ResCfg {
  int64_t n, h, w, ic, oc, k, s, pad_value;
  int xin;  // 0 linear int8 + thresholds, 1 l8 int8 (x_code) + thresholds, 2 bit tensor
  bool r_l8, bias, bits, gt;
  int r_exp;
  bool bit_out = false;  // bit_src "out": the bits binarize w (the block output) instead of u
  int pre = 0;           // pre stage inputs (bit mask): 1 pre_slope, 2 pre_mul, 4 pre_add, 8 gate [N, 1, 1, O], 16 gate_add
  Json extra = Json::Object{};  // attributes set last (int8_full v2: output "i8", y_zero, qmin / qmax)
};

// One I8BConv2DRes node (constants prefixed with p); outs = its output names.
void add_res_node(bftest::ModelBuilder& mb, Rng& r, const ResCfg& c, const std::string& x, const std::string& res, const std::string& p,
                  std::vector<std::string>& outs) {
  std::vector<int8_t> wv(static_cast<size_t>(c.oc * c.k * c.k * c.ic));
  for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
  auto wp = bftest::pack_pm1(wv, c.oc * c.k * c.k, c.ic);
  mb.constant(p + "w", DType::kBit, {c.oc, c.k, c.k, c.ic}, wp.data());
  // magnitudes over many octaves; some channels with pm = 0 and pa on (or next to) a rounding boundary, so that u is
  // exactly there when the residual code is 0, and a bit threshold equal to it (tie)
  std::vector<float> pm(static_cast<size_t>(c.oc)), pa(static_cast<size_t>(c.oc)), sl(static_cast<size_t>(c.oc)),
      bi(static_cast<size_t>(c.oc)), bt(static_cast<size_t>(c.oc));
  for (int64_t o = 0; o < c.oc; ++o) {
    const float sc = std::ldexp(1.f, r.i(-14, -2));
    pm[o] = r.f(-1.f, 1.f) * sc;
    pa[o] = r.f(-20.f, 20.f) * sc;
    if (o % 7 == 3) {
      pm[o] = 0.f;
      pa[o] = (o % 2 ? -1.f : 1.f) * std::ldexp(1.f + static_cast<float>(q8::kL8B[o % 8]) / 128.f, r.i(-12, -1));
      if (o % 3 == 0) pa[o] = std::nextafter(pa[o], 0.f);
    }
    sl[o] = o % 5 == 0 ? 0.f : (o % 5 == 1 ? -r.f(0.f, 2.f) : r.f(0.f, 3.f));
    bi[o] = r.f(-0.05f, 0.05f);
    bt[o] = o % 4 == 0 ? 0.f : r.f(-0.02f, 0.02f);
    if (o % 7 == 3) bt[o] = pa[o];
    if (o % 7 == 3 && c.bit_out) {  // tie of w (residual code 0)
      float wv = pa[o] >= 0.f ? pa[o] : pa[o] * sl[o];
      if (c.bias) wv = wv + bi[o];
      bt[o] = wv;
    }
  }
  mb.constant(p + "pm", DType::kF32, {c.oc}, pm.data());
  mb.constant(p + "pa", DType::kF32, {c.oc}, pa.data());
  mb.constant(p + "sl", DType::kF32, {c.oc}, sl.data());
  std::vector<std::string> ins = {x, p + "w", p + "pm", p + "pa", ""};
  Json at = Json::Object{{"strides", ints({c.s, c.s})}, {"pads", ints({c.k / 2, c.k / 2, c.k / 2, c.k / 2})},
                         {"pad_value", c.pad_value}, {"y_exp", 0}};
  if (c.xin != 2) {
    auto t = r.f32(c.ic, -0.01f, 0.01f);
    for (int64_t i = 0; i < c.ic; i += 5) t[i] = c.xin == 1 ? q8::l8_decode(r.i(-128, 127), -2) : q8::deq(r.i(-128, 127), 0.002f, 3);
    mb.constant(p + "t", DType::kF32, {c.ic}, t.data());
    ins[4] = p + "t";
    at["input_binarize"] = r.i(0, 1) ? "gt" : "ge";
    if (c.xin == 1) {
      at["x_code"] = "l8";
      at["x_exp"] = -2;
    } else {
      at["x_scale"] = 0.002;
      at["x_zero"] = 3;
    }
  }
  ins.push_back(res);
  ins.push_back(p + "sl");
  if (c.r_l8) {
    at["r_code"] = "l8";
    at["r_exp"] = c.r_exp;
  } else {
    at["r_code"] = "linear";
    at["r_scale"] = static_cast<double>(std::ldexp(1.f, c.r_exp) / 100.f);
    at["r_zero"] = r.i(-20, 20);
  }
  if (c.bias) mb.constant(p + "bi", DType::kF32, {c.oc}, bi.data());
  ins.push_back(c.bias ? p + "bi" : "");
  outs = {p + "y"};
  if (c.bits) {
    mb.constant(p + "bt", DType::kF32, {c.oc}, bt.data());
    ins.push_back(p + "bt");
    at["bit_binarize"] = c.gt ? "gt" : "ge";
    if (c.bit_out) at["bit_src"] = "out";
    outs.push_back(p + "b");
  }
  if (c.pre) {  // inputs 9 (out_lut, Session only) .. 13
    while (ins.size() < 10) ins.push_back("");
    std::vector<float> ps(static_cast<size_t>(c.oc)), pmu(static_cast<size_t>(c.oc)), pad(static_cast<size_t>(c.oc));
    for (int64_t o = 0; o < c.oc; ++o) {
      ps[o] = o % 3 == 0 ? 0.f : r.f(-1.5f, 2.5f);                       // ReLU (0) and PReLU slopes
      pmu[o] = o % 11 == 5 ? 0.f : r.f(-2.f, 2.f) * std::ldexp(1.f, r.i(-3, 3));
      pad[o] = o % 4 == 1 ? -0.f : r.f(-0.3f, 0.3f);
    }
    mb.constant(p + "ps", DType::kF32, {c.oc}, ps.data());
    mb.constant(p + "pmu", DType::kF32, {c.oc}, pmu.data());
    mb.constant(p + "pad", DType::kF32, {c.oc}, pad.data());
    ins.push_back(c.pre & 1 ? p + "ps" : "");
    ins.push_back(c.pre & 2 ? p + "pmu" : "");
    ins.push_back(c.pre & 4 ? p + "pad" : "");
    if (c.pre & 8) {
      auto gq = r.i8(c.n * c.oc);
      for (int64_t i = 0; i < c.n * c.oc; i += 13) gq[i] = 0x80;  // gate 0
      mb.constant(p + "g", DType::kI8, {c.n, 1, 1, c.oc}, gq.data());
      ins.push_back(p + "g");
      at["g_scale"] = 1.0 / 255.0;
      at["g_zero"] = -128;
    }
    if (c.pre & 16) {  // gate_add after the gate (2026-11-14)
      while (ins.size() < 14) ins.push_back("");
      std::vector<float> ga(static_cast<size_t>(c.oc));
      for (int64_t o = 0; o < c.oc; ++o) ga[o] = o % 5 == 2 ? -0.f : r.f(-0.4f, 0.4f);
      mb.constant(p + "ga", DType::kF32, {c.oc}, ga.data());
      ins.push_back(p + "ga");
    }
  }
  while (!ins.empty() && ins.back().empty()) ins.pop_back();
  for (const auto& kv : c.extra.as_object()) at[kv.first] = kv.second;
  mb.node("I8BConv2DRes", ins, outs, at);
}

std::vector<uint8_t> res_codes(Rng& r, int64_t n) {
  auto v = r.i8(n);
  for (int64_t i = 0; i < n; i += 9) v[i] = 0;
  for (int64_t i = 4; i < n; i += 23) v[i] = static_cast<uint8_t>(i % 2 ? 127 : 0x80);  // 127 / -128
  return v;
}

BF_TEST(int8_full_bconv_res_bit_exact_vs_reference) {
#if BF_NEON_A32
  const int a32_before = g_a32_res_neon;
  struct A32Check {  // most configurations must take the AArch32 NEON epilogue
    const int& before;
    ~A32Check() noexcept(false) {
      if (!std::uncaught_exceptions() && g_a32_res_neon - before < 20)
        throw bftest::Failure{"AArch32: only " + std::to_string(g_a32_res_neon - before) + " I8BConv2DRes runs on i8bconv2dres.neon"};
    }
  } a32_check{a32_before};
#endif
  Rng r(4242);
  struct G { int64_t n, h, w, ic, oc, k, s, pad_value; };
  int idx = 0;
  for (G g : std::vector<G>{{1, 9, 11, 64, 64, 3, 1, 0},    // pair mode (k_words 18), whole words
                            {2, 7, 6, 128, 40, 3, 1, 0},    // normal mode, oc 40 (partial last word)
                            {1, 8, 7, 64, 21, 3, 2, 1},     // stride 2, pad value 1, channel tail
                            {1, 5, 5, 32, 10, 3, 1, -1},    // pair mode (k_words 9), oc 10, pad -1
                            {1, 6, 6, 130, 12, 1, 1, 0}}) { // 1x1, K tail
    for (int variant = 0; variant < 9; ++variant, ++idx) {
      const int rexp = variant == 5 ? -8 : (variant == 1 ? 7 : r.i(-3, 3));
      ResCfg c{g.n, g.h, g.w, g.ic, g.oc, g.k, g.s, g.pad_value, variant % 2, variant != 3, variant % 3 != 0, variant != 4,
               variant == 2 || variant == 5, rexp};
      if (variant >= 6) {  // bits of the block output w (bit_src "out"): l8 / linear residuals, bias or not, ge / gt
        c.bits = true;
        c.bit_out = true;
        c.xin = (variant - 6) % 2;
        c.r_l8 = variant != 7;
        c.bias = variant != 8;
        c.gt = variant == 7;
      }
      const int64_t oh = (g.h + 2 * (g.k / 2) - g.k) / g.s + 1, ow = (g.w + 2 * (g.k / 2) - g.k) / g.s + 1;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {g.n, g.h, g.w, g.ic});
      mb.input("res", DType::kI8, {g.n, oh, ow, g.oc});
      std::vector<std::string> outs;
      add_res_node(mb, r, c, "x", "res", "a", outs);
      for (auto& o : outs) mb.output(o);
      std::vector<uint8_t> xin = c.xin == 1 ? res_codes(r, g.n * g.h * g.w * g.ic) : r.i8(g.n * g.h * g.w * g.ic);
      compare_all(mb, {xin, res_codes(r, g.n * oh * ow * g.oc)}, "I8BConv2DRes config " + std::to_string(idx), "i8bconv2dres.neon");
    }
  }
  // chains: block b reads block a's bit output (x) and l8 output (residual); a plain I8BConv2D reads b's bits
  for (int64_t c0 : {64, 128}) {
    for (int gt = 0; gt < 2; ++gt) {
      const int64_t h = 9, w = 10;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {1, h, w, c0});
      mb.input("res", DType::kI8, {1, h, w, c0});
      std::vector<std::string> o1, o2;
      ResCfg a{1, h, w, c0, c0, 3, 1, 0, 0, false, false, true, gt == 1, -2};
      add_res_node(mb, r, a, "x", "res", "a", o1);
      ResCfg b{1, h, w, c0, c0, 3, 1, 0, 2, true, true, true, gt == 0, -1};
      b.bit_out = gt == 1;  // the plain I8BConv2D reads bits of b's output w
      add_res_node(mb, r, b, o1[1], o1[0], "b", o2);
      std::vector<int8_t> wv(static_cast<size_t>(16 * 9 * c0));
      for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
      auto wp = bftest::pack_pm1(wv, 16 * 9, c0);
      mb.constant("cw", DType::kBit, {16, 3, 3, c0}, wp.data());
      auto pm = r.f32(16, -1.f, 1.f), pa = r.f32(16, -9.f, 9.f);
      mb.constant("cpm", DType::kF32, {16}, pm.data());
      mb.constant("cpa", DType::kF32, {16}, pa.data());
      mb.node("I8BConv2D", {o2[1], "cw", "cpm", "cpa"}, {"cy"},
              Json::Object{{"strides", ints({1, 1})}, {"pads", ints({1, 1, 1, 1})}, {"pad_value", 0}, {"qmin", -100}, {"qmax", 100}});
      for (auto& o : o1) mb.output(o);
      for (auto& o : o2) mb.output(o);
      mb.output("cy");
      compare_all(mb, {r.i8(h * w * c0), res_codes(r, h * w * c0)}, "I8BConv2DRes chain", "i8bconv2dres.neon");
    }
  }
}

// pre stage of I8BConv2DRes (2026-11-05: QuickNet BConv[relu] -> BN -> Add, SE-gated blocks): every subset of
// pre_slope / pre_mul / pre_add / gate with l8 and linear residuals, bias or not, bits of u / w, ge / gt, batch 2 (the gate
// is per sample), channel tails (scalar epilogue) and 1x1 / strided shapes.
BF_TEST(int8_full_bconv_res_pre_bit_exact_vs_reference) {
  Rng r(4747);
  struct G { int64_t n, h, w, ic, oc, k, s, pad_value; };
  int idx = 0;
  for (G g : std::vector<G>{{2, 9, 7, 64, 64, 3, 1, 0},     // pair mode, whole words, two samples
                            {2, 6, 7, 128, 40, 3, 1, 1},    // oc 40: last unit of 8 channels (scalar tail)
                            {1, 8, 8, 64, 21, 3, 2, -1},    // stride 2, channel tail
                            {2, 5, 6, 130, 16, 1, 1, 0}}) { // 1x1, K tail
    for (int pre : {1, 7, 8, 15, 6, 9, 24, 31}) {
      for (int variant = 0; variant < 3; ++variant, ++idx) {
        ResCfg c{g.n, g.h, g.w, g.ic, g.oc, g.k, g.s, g.pad_value, variant % 2, variant != 1, variant != 2, true, variant == 1,
                 r.i(-3, 3)};
        c.bit_out = variant == 2;
        c.pre = pre;
        const int64_t oh = (g.h + 2 * (g.k / 2) - g.k) / g.s + 1, ow = (g.w + 2 * (g.k / 2) - g.k) / g.s + 1;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {g.n, g.h, g.w, g.ic});
        mb.input("res", DType::kI8, {g.n, oh, ow, g.oc});
        std::vector<std::string> outs;
        add_res_node(mb, r, c, "x", "res", "a", outs);
        for (auto& o : outs) mb.output(o);
        std::vector<uint8_t> xin = c.xin == 1 ? res_codes(r, g.n * g.h * g.w * g.ic) : r.i8(g.n * g.h * g.w * g.ic);
        compare_all(mb, {xin, res_codes(r, g.n * oh * ow * g.oc)}, "I8BConv2DRes pre config " + std::to_string(idx),
                    "i8bconv2dres.neon");
      }
    }
  }
}

// int8_full v2 (2026-11-13): I8BConv2DRes with the linear output code (output "i8": clamp(rne(w) + y_zero, qmin, qmax)),
// linear residuals, with and without bias / bits of u or w / pre stage, y_zero far from 0 and clamped ranges.
BF_TEST(int8_full_bconv_res_linear_output_bit_exact_vs_reference) {
  Rng r(5151);
  struct G { int64_t n, h, w, ic, oc, k, s, pad_value; };
  int idx = 0;
  for (G g : std::vector<G>{{1, 9, 11, 64, 64, 3, 1, 0}, {2, 7, 6, 128, 40, 3, 1, 0}, {1, 8, 7, 64, 21, 3, 2, 1},
                            {1, 6, 6, 130, 16, 1, 1, -1}}) {
    for (int variant = 0; variant < 6; ++variant, ++idx) {
      ResCfg c{g.n, g.h, g.w, g.ic, g.oc, g.k, g.s, g.pad_value, variant % 2, false, variant % 3 == 0, variant != 3, variant == 4,
               r.i(-3, 3)};
      c.bit_out = variant == 5;
      c.pre = variant == 2 ? 7 : 0;
      const int64_t oh = (g.h + 2 * (g.k / 2) - g.k) / g.s + 1, ow = (g.w + 2 * (g.k / 2) - g.k) / g.s + 1;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {g.n, g.h, g.w, g.ic});
      mb.input("res", DType::kI8, {g.n, oh, ow, g.oc});
      std::vector<std::string> outs;
      // linear output; residual units scaled up so that w spans the int8 range
      c.extra = Json::Object{{"output", "i8"}, {"y_zero", variant == 1 ? 100 : r.i(-60, 60)}, {"qmin", variant == 3 ? -100 : -128},
                             {"qmax", variant == 3 ? 90 : 127}, {"r_scale", static_cast<double>(r.f(0.3f, 4.f))}};
      add_res_node(mb, r, c, "x", "res", "a", outs);
      for (auto& o : outs) mb.output(o);
      std::vector<uint8_t> xin = c.xin == 1 ? res_codes(r, g.n * g.h * g.w * g.ic) : r.i8(g.n * g.h * g.w * g.ic);
      compare_all(mb, {xin, res_codes(r, g.n * oh * ow * g.oc)}, "I8BConv2DRes linear output config " + std::to_string(idx),
                  "i8bconv2dres.neon");
    }
  }
}

// I8BConv2DResI (int8_full v2): the integer residual epilogue. Random multipliers within the operator's bound, every
// combination of PReLU (none, slopes in [0, 1) = the kernel's max form, signed / > 1 slopes with a slope shift), out_bias,
// bits of acc / w, the clamp pre stage, y_zero, clamped ranges, shifts 0..16 (fast form) and > 16 (generic form), channel
// tails (scalar epilogue), strides, 1x1, pad values, batch 2, bit and int8 inputs; and a chain reading the bits.
BF_TEST(int8_full_bconv_resi_bit_exact_vs_reference) {
  Rng r(6161);
  struct G { int64_t n, h, w, ic, oc, k, s, pad_value; };
  int idx = 0;
  auto add_node = [&](bftest::ModelBuilder& mb, const G& g, int variant, const std::string& x, const std::string& res,
                      const std::string& p, bool bit_in, std::vector<std::string>& outs, int pmode = 0) {
    const int64_t K = g.k * g.k * g.ic;
    std::vector<int8_t> wv(static_cast<size_t>(g.oc * K));
    for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
    auto wp = bftest::pack_pm1(wv, g.oc * g.k * g.k, g.ic);
    mb.constant(p + "w", DType::kBit, {g.oc, g.k, g.k, g.ic}, wp.data());
    const int slope_mode = variant % 3;            // 0 none, 1 slopes in [0, 1), 2 signed / > 1 with a slope shift
    const bool ob = variant % 2 == 1, pre = variant % 4 == 2, bits = variant % 5 != 4, bit_out = variant % 7 == 3;
    const int shift = variant == 8 ? 0 : (variant == 9 ? 20 : r.i(6, 16));
    const int k = slope_mode == 2 ? r.i(1, 2) : 0;
    // |acc| < 2^30: m1 * K + 128 * m2 + |b| (+ |pre_b|)
    const int64_t m1max = std::min<int64_t>(32767, (int64_t(1) << 28) / K), m2max = std::min<int64_t>(32767, int64_t(1) << 20);
    std::vector<int32_t> m1(static_cast<size_t>(g.oc)), b(static_cast<size_t>(g.oc)), m2(static_cast<size_t>(g.oc)),
        thr(static_cast<size_t>(g.oc)), sl(static_cast<size_t>(g.oc)), obv(static_cast<size_t>(g.oc)), pb(static_cast<size_t>(g.oc)),
        plo(static_cast<size_t>(g.oc)), phi(static_cast<size_t>(g.oc));
    const int32_t m2c = static_cast<int32_t>(r.i(-static_cast<int>(m2max), static_cast<int>(m2max)));
    const int64_t unit = int64_t(1) << std::min(shift, 16);
    for (int64_t o = 0; o < g.oc; ++o) {
      m1[o] = o % 9 == 4 ? 0 : r.i(-static_cast<int>(m1max), static_cast<int>(m1max));
      m2[o] = o % 6 == 1 ? r.i(-static_cast<int>(m2max), static_cast<int>(m2max)) : m2c;
      b[o] = static_cast<int32_t>(r.f(-100.f, 100.f) * static_cast<float>(unit));
      thr[o] = o % 8 == 0 ? b[o] : static_cast<int32_t>(r.f(-30.f, 30.f) * static_cast<float>(unit));  // tie at dot = q = 0
      sl[o] = slope_mode == 1 ? r.i(0, 2147483647) : r.i(-2147483647, 2147483647);
      if (o % 5 == 0) sl[o] = 0;
      obv[o] = static_cast<int32_t>(r.f(-20.f, 20.f) * static_cast<float>(unit));
      pb[o] = static_cast<int32_t>(r.f(-50.f, 50.f) * static_cast<float>(unit));
      const bool neg = o % 3 == 1;
      plo[o] = neg ? INT32_MIN : 0;
      phi[o] = neg ? 0 : INT32_MAX;
      if (variant == 10 && o % 4 == 3) {  // a general clamp (not max(., 0) / min(., 0)): the scalar epilogue
        plo[o] = -static_cast<int32_t>(r.f(0.f, 20.f) * static_cast<float>(unit));
        phi[o] = static_cast<int32_t>(r.f(0.f, 20.f) * static_cast<float>(unit));
      }
    }
    mb.constant(p + "m1", DType::kI32, {g.oc}, m1.data());
    mb.constant(p + "b", DType::kI32, {g.oc}, b.data());
    mb.constant(p + "m2", DType::kI32, {g.oc}, m2.data());
    std::vector<std::string> ins = {x, p + "w", "", "", "", res, p + "m1", p + "b", p + "m2"};
    Json at = Json::Object{{"strides", ints({g.s, g.s})}, {"pads", ints({g.k / 2, g.k / 2, g.k / 2, g.k / 2})},
                           {"pad_value", g.pad_value}, {"shift", shift}, {"y_zero", variant == 6 ? -110 : r.i(-40, 40)},
                           {"qmin", variant == 7 ? -90 : -128}, {"qmax", variant == 7 ? 100 : 127}};
    if (!bit_in) {
      auto t = r.f32(g.ic, -0.01f, 0.01f);
      for (int64_t i = 0; i < g.ic; i += 5) t[i] = q8::deq(r.i(-128, 127), 0.002f, 3);
      mb.constant(p + "t", DType::kF32, {g.ic}, t.data());
      ins[4] = p + "t";
      at["input_binarize"] = r.i(0, 1) ? "gt" : "ge";
      at["x_scale"] = 0.002;
      at["x_zero"] = 3;
    }
    outs = {p + "y"};
    if (bits) {
      mb.constant(p + "bt", DType::kI32, {g.oc}, thr.data());
      ins.push_back(p + "bt");
      if (bit_out) at["bit_src"] = "out";
      outs.push_back(p + "bits");
    } else {
      ins.push_back("");
    }
    if (slope_mode) {
      mb.constant(p + "sl", DType::kI32, {g.oc}, sl.data());
      ins.push_back(p + "sl");
      at["slope_shift"] = k;
    } else {
      ins.push_back("");
    }
    if (ob) mb.constant(p + "ob", DType::kI32, {g.oc}, obv.data());
    ins.push_back(ob ? p + "ob" : "");
    if (pre) {
      mb.constant(p + "pb", DType::kI32, {g.oc}, pb.data());
      mb.constant(p + "plo", DType::kI32, {g.oc}, plo.data());
      mb.constant(p + "phi", DType::kI32, {g.oc}, phi.data());
      ins.push_back(p + "pb");
      ins.push_back(p + "plo");
      ins.push_back(p + "phi");
      if (pmode) {  // PReLU pre stage (2026-11-14): 1 slopes in [0, 1) (max form), 2 signed / > 1 with a shift (select form)
        std::vector<int32_t> ps(static_cast<size_t>(g.oc));
        for (int64_t o = 0; o < g.oc; ++o) ps[o] = o % 7 == 3 ? 0 : (pmode == 1 ? r.i(0, 2147483647) : r.i(-2147483647, 2147483647));
        mb.constant(p + "ps", DType::kI32, {g.oc}, ps.data());
        ins.push_back(p + "ps");
        at["pre_slope_shift"] = pmode == 2 ? 1 : 0;
      }
    }
    while (!ins.empty() && ins.back().empty()) ins.pop_back();
    mb.node("I8BConv2DResI", ins, outs, at);
  };
  for (G g : std::vector<G>{{1, 9, 11, 64, 64, 3, 1, 0},     // pair mode, whole words
                            {2, 7, 6, 128, 40, 3, 1, 0},     // oc 40: a last unit of 8 channels (scalar tail), two samples
                            {1, 8, 7, 64, 21, 3, 2, 1},      // stride 2, channel tail
                            {1, 6, 6, 130, 16, 1, 1, -1},    // 1x1, K tail
                            {1, 10, 9, 32, 48, 3, 1, 0}}) {  // 16-channel input rows
    for (int variant = 0; variant < 12; ++variant, ++idx) {
      const int64_t oh = (g.h + 2 * (g.k / 2) - g.k) / g.s + 1, ow = (g.w + 2 * (g.k / 2) - g.k) / g.s + 1;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {g.n, g.h, g.w, g.ic});
      mb.input("res", DType::kI8, {g.n, oh, ow, g.oc});
      std::vector<std::string> outs;
      add_node(mb, g, variant, "x", "res", "a", false, outs);
      for (auto& o : outs) mb.output(o);
      compare_all(mb, {r.i8(g.n * g.h * g.w * g.ic), res_codes(r, g.n * oh * ow * g.oc)},
                  "I8BConv2DResI config " + std::to_string(idx), "i8bconv2dresi.neon");
    }
  }
  // PReLU pre stage (in(15)): variants with the max(., 0) / min(., 0) pre stage (2, 6), both slope forms
  for (G g : std::vector<G>{{1, 9, 11, 64, 64, 3, 1, 0}, {2, 7, 6, 128, 40, 3, 1, 0}, {1, 8, 7, 64, 21, 3, 2, 1},
                            {1, 6, 6, 130, 16, 1, 1, -1}}) {
    for (int variant : {2, 6}) {
      for (int pmode : {1, 2}) {
        const int64_t oh = (g.h + 2 * (g.k / 2) - g.k) / g.s + 1, ow = (g.w + 2 * (g.k / 2) - g.k) / g.s + 1;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {g.n, g.h, g.w, g.ic});
        mb.input("res", DType::kI8, {g.n, oh, ow, g.oc});
        std::vector<std::string> outs;
        add_node(mb, g, variant, "x", "res", "a", false, outs, pmode);
        for (auto& o : outs) mb.output(o);
        compare_all(mb, {r.i8(g.n * g.h * g.w * g.ic), res_codes(r, g.n * oh * ow * g.oc)},
                    "I8BConv2DResI PReLU pre " + std::to_string(variant) + "/" + std::to_string(pmode), "i8bconv2dresi.neon");
      }
    }
  }
  // chain: block b reads block a's bits (x) and int8 output (residual)
  for (int variant : {0, 1, 2, 5}) {
    const G g{1, 9, 10, 64, 64, 3, 1, 0};
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {1, g.h, g.w, g.ic});
    mb.input("res", DType::kI8, {1, g.h, g.w, g.oc});
    std::vector<std::string> o1, o2;
    add_node(mb, g, 0, "x", "res", "a", false, o1);
    add_node(mb, g, variant, o1[1], o1[0], "b", true, o2);
    for (auto& o : o1) mb.output(o);
    for (auto& o : o2) mb.output(o);
    compare_all(mb, {r.i8(g.h * g.w * g.ic), res_codes(r, g.h * g.w * g.oc)}, "I8BConv2DResI chain " + std::to_string(variant),
                "i8bconv2dresi.neon");
  }
}

// I8BConv2DResIX (EBN expert binary convolution, 2026-11-14): with expert index e it must give the bytes of the
// I8BConv2DResI with the e-th weight set, on every tier (NEON: E prepared instances of the base kernel, a pointer select per
// run), with and without the bit output and the residual; I8ArgMax picks the first maximum.
BF_TEST(int8_full_expert_bconv_equals_selected_conv) {
  Rng r(8181);
  struct G { int64_t h, w, ic, oc, k, s; };
  for (G g : std::vector<G>{{9, 11, 64, 64, 3, 1}, {8, 7, 64, 48, 3, 2}, {6, 6, 128, 32, 1, 1}}) {
    const int64_t E = 3, K = g.k * g.k * g.ic;
    const int64_t oh = (g.h + 2 * (g.k / 2) - g.k) / g.s + 1, ow = (g.w + 2 * (g.k / 2) - g.k) / g.s + 1;
    std::vector<int8_t> wv(static_cast<size_t>(E * g.oc * K));
    for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
    std::vector<int32_t> m1(static_cast<size_t>(g.oc)), b(static_cast<size_t>(g.oc)), m2(static_cast<size_t>(g.oc)), thr(static_cast<size_t>(g.oc));
    for (int64_t o = 0; o < g.oc; ++o) {
      m1[o] = r.i(-2000, 2000);
      m2[o] = r.i(-3000, 3000);
      b[o] = r.i(-100000, 100000);
      thr[o] = r.i(-50000, 50000);
    }
    auto xin = r.i8(g.h * g.w * g.ic), rin = res_codes(r, oh * ow * g.oc);
    for (int bits = 0; bits < 2; ++bits)
      for (int64_t e = 0; e < E; ++e) {
        auto build = [&](bool expert) {
          bftest::ModelBuilder mb;
          mb.input("x", DType::kI8, {1, g.h, g.w, g.ic});
          mb.input("res", DType::kI8, {1, oh, ow, g.oc});
          std::vector<int8_t> ws = expert ? wv : std::vector<int8_t>(wv.begin() + e * g.oc * K, wv.begin() + (e + 1) * g.oc * K);
          const int64_t rows = expert ? E * g.oc : g.oc;
          auto wp = bftest::pack_pm1(ws, rows * g.k * g.k, g.ic);
          mb.constant("w", DType::kBit, {rows, g.k, g.k, g.ic}, wp.data());
          auto t = r.f32(g.ic, -0.01f, 0.01f);
          mb.constant("t", DType::kF32, {g.ic}, t.data());
          mb.constant("m1", DType::kI32, {g.oc}, m1.data());
          mb.constant("b", DType::kI32, {g.oc}, b.data());
          mb.constant("m2", DType::kI32, {g.oc}, m2.data());
          mb.constant("bt", DType::kI32, {g.oc}, thr.data());
          std::vector<std::string> ins = {"x", "w", "", "", "t", "res", "m1", "b", "m2", bits ? "bt" : ""};
          Json at = Json::Object{{"strides", ints({g.s, g.s})}, {"pads", ints({g.k / 2, g.k / 2, g.k / 2, g.k / 2})}, {"pad_value", 0},
                                 {"shift", 10}, {"y_zero", 3}, {"input_binarize", "ge"}, {"x_scale", 0.002}, {"x_zero", 3}};
          std::vector<std::string> outs = {"y"};
          if (bits) outs.push_back("yb");
          if (expert) {
            const int32_t ev = static_cast<int32_t>(e);
            mb.constant("ix", DType::kI32, {1}, &ev);
            while (ins.size() < 16) ins.push_back("");
            ins.push_back("ix");
            at["experts"] = E;
            mb.node("I8BConv2DResIX", ins, outs, at);
          } else {
            while (!ins.empty() && ins.back().empty()) ins.pop_back();
            mb.node("I8BConv2DResI", ins, outs, at);
          }
          for (auto& o : outs) mb.output(o);
          return mb;
        };
        // same thresholds in both builds: reseed the threshold draw
        Rng keep = r;
        auto mx = build(true);
        r = keep;
        auto mp = build(false);
        for (const char* tiers : {kRef, kFull}) {
          for (int t : {1, 4}) {
            auto a = run_model_all(mx, tiers, t, {xin, rin}), c = run_model_all(mp, tiers, t, {xin, rin});
            if (a != c) throw bftest::Failure{"I8BConv2DResIX differs from the selected I8BConv2DResI (expert " + std::to_string(e) + ")"};
          }
        }
      }
  }
}

// I8BConvBP (bit-serial binary 1x1 on integer inputs, 2026-11-14) and I8DctYuv420 (partial 8x8 DCT of a YUV 4:2:0 frame):
// the NEON / fast kernels against the reference, channel counts below / above one 32-bit word, with and without bits.
BF_TEST(int8_full_bconv_bp_and_dct_vs_reference) {
  Rng r(9191);
  struct G { int64_t n, h, w, c, o; bool bits; };
  for (G g : std::vector<G>{{1, 28, 28, 24, 64, true}, {2, 5, 7, 40, 13, false}, {1, 6, 6, 7, 33, true}, {1, 4, 4, 64, 8, true}}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {g.n, g.h, g.w, g.c});
    std::vector<int8_t> wv(static_cast<size_t>(g.o * g.c));
    for (auto& v : wv) v = r.i(0, 1) ? 1 : -1;
    auto wp = bftest::pack_pm1(wv, g.o, g.c);
    mb.constant("w", DType::kBit, {g.o, 1, 1, g.c}, wp.data());
    std::vector<int32_t> m1(static_cast<size_t>(g.o)), b(static_cast<size_t>(g.o)), t(static_cast<size_t>(g.o));
    const int mx = static_cast<int>(std::min<int64_t>(30000, (int64_t(1) << 29) / (255 * g.c)));
    for (int64_t k = 0; k < g.o; ++k) {
      m1[k] = r.i(-mx, mx);
      b[k] = r.i(-100000, 100000);
      t[k] = k % 5 == 0 ? b[k] : r.i(-200000, 200000);
    }
    mb.constant("m1", DType::kI32, {g.o}, m1.data());
    mb.constant("b", DType::kI32, {g.o}, b.data());
    mb.constant("t", DType::kI32, {g.o}, t.data());
    std::vector<std::string> ins = {"x", "w", "m1", "b"}, outs = {"y"};
    if (g.bits) {
      ins.push_back("t");
      outs.push_back("yb");
    }
    mb.node("I8BConvBP", ins, outs, Json::Object{{"x_zero", r.i(-20, 20)}, {"shift", r.i(4, 14)}, {"y_zero", r.i(-30, 30)}});
    for (auto& o : outs) mb.output(o);
    compare_all(mb, {r.i8(g.n * g.h * g.w * g.c)}, "I8BConvBP " + std::to_string(g.c) + "x" + std::to_string(g.o), "i8bconvbp.neon");
  }
  for (int64_t hw : {16, 32, 224}) {
    bftest::ModelBuilder mb;
    mb.input("y", DType::kI8, {1, hw, hw, 1});
    mb.input("c", DType::kI8, {1, hw / 2, hw / 2, 2});
    std::vector<int64_t> qy(64), qc(64), sh;
    for (int k = 0; k < 64; ++k) {
      qy[k] = r.i(1, 40);
      qc[k] = r.i(1, 40);
    }
    std::vector<int64_t> ysel = {0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11}, csel = {0, 1, 8, 16, 9};
    for (int k = 0; k < 24; ++k) sh.push_back(k % 7 == 0 ? 2 : r.i(0, 1));
    mb.node("I8DctYuv420", {"y", "c"}, {"o"},
            Json::Object{{"y_sel", ints(ysel)}, {"c_sel", ints(csel)}, {"qy", ints(qy)}, {"qc", ints(qc)}, {"shifts", ints(sh)}});
    mb.output("o");
    compare_all(mb, {r.i8(hw * hw), r.i8(hw * hw / 2)}, "I8DctYuv420 " + std::to_string(hw), "");
  }
}

// I8GatedAdd (int8_full v2: BNext's x + k * sigmoid(se) * p): random constants within the bound, zero points far from 0,
// 0 / 1 / 2 / 3 bit outputs, shifts 0..16 (fast form) and > 16, clamped ranges, channel counts with tails, batch 2.
BF_TEST(int8_full_gated_add_bit_exact_vs_reference) {
  Rng r(7171);
  int idx = 0;
  for (int64_t c : {16, 32, 40, 64, 7}) {
    for (int variant = 0; variant < 6; ++variant, ++idx) {
      const int64_t n = variant % 2 ? 2 : 1, h = 5, w = 7, nb = variant % 4;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {n, h, w, c});
      mb.input("p", DType::kI8, {n, h, w, c});
      mb.input("g", DType::kI8, {n, 1, 1, c});
      const int shift = variant == 5 ? 20 : (variant == 4 ? 0 : r.i(8, 16));
      std::vector<int32_t> kx(static_cast<size_t>(c)), kp(static_cast<size_t>(c)), b(static_cast<size_t>(c)),
          thr(static_cast<size_t>(std::max<int64_t>(nb, 1) * c));
      for (int64_t i = 0; i < c; ++i) {
        kx[i] = r.i(-20000, 20000);
        kp[i] = i % 7 == 2 ? 0 : r.i(-4000, 4000);
        b[i] = r.i(-(1 << 24), 1 << 24);
      }
      for (auto& t : thr) t = r.i(-(1 << 22), 1 << 22);
      thr[0] = b[0] - 3 * kx[0];  // a tie at x - z_x = 3, p - z_p = 0 is reachable
      mb.constant("kx", DType::kI32, {c}, kx.data());
      mb.constant("kp", DType::kI32, {c}, kp.data());
      mb.constant("b", DType::kI32, {c}, b.data());
      std::vector<std::string> ins = {"x", "p", "g", "kx", "kp", "b"}, outs = {"y"};
      if (nb) {
        mb.constant("t", DType::kI32, {nb, c}, thr.data());
        ins.push_back("t");
        for (int64_t j = 0; j < nb; ++j) outs.push_back("bits" + std::to_string(j));
      }
      mb.node("I8GatedAdd", ins, outs,
              Json::Object{{"x_zero", r.i(-128, 127)}, {"p_zero", variant == 3 ? 120 : r.i(-20, 20)}, {"g_zero", -128}, {"shift", shift},
                           {"y_zero", r.i(-100, 100)}, {"qmin", variant == 2 ? -90 : -128}, {"qmax", variant == 2 ? 100 : 127}});
      for (auto& o : outs) mb.output(o);
      auto gq = r.i8(n * c);
      gq[0] = 0x80;
      compare_all(mb, {r.i8(n * h * w * c), r.i8(n * h * w * c), gq}, "I8GatedAdd config " + std::to_string(idx), "i8gatedadd.neon");
    }
  }
}

// I8GapAffine (int8_full v2: BNext's GAP(mix) without rounding the full-resolution mix): one to three inputs, large and
// odd spatial sizes, shifts up to 40, channel tails, batch 2.
BF_TEST(int8_full_gap_affine_bit_exact_vs_reference) {
  Rng r(8181);
  int idx = 0;
  for (int64_t c : {16, 64, 40, 3}) {
    for (int k : {1, 2, 3}) {
      for (int64_t hw : {1, 49, 3136}) {
        const int64_t n = idx % 2 ? 2 : 1, h = hw == 3136 ? 56 : (hw == 49 ? 7 : 1), w = hw / h;
        bftest::ModelBuilder mb;
        std::vector<std::vector<uint8_t>> in;
        std::vector<std::string> ins;
        Json::Array zs;
        for (int i = 0; i < k; ++i) {
          mb.input("x" + std::to_string(i), DType::kI8, {n, h, w, c});
          ins.push_back("x" + std::to_string(i));
          in.push_back(r.i8(n * h * w * c));
          zs.push_back(Json(r.i(-128, 127)));
        }
        std::vector<int32_t> K(static_cast<size_t>(k * c)), B(static_cast<size_t>(c));
        for (auto& v : K) v = r.i(-(1 << 30), 1 << 30) >> r.i(0, 20);
        for (auto& v : B) v = r.i(-(1 << 30), 1 << 30);
        mb.constant("K", DType::kI32, {k, c}, K.data());
        mb.constant("B", DType::kI32, {c}, B.data());
        ins.push_back("K");
        ins.push_back("B");
        mb.node("I8GapAffine", ins, {"y"}, Json::Object{{"zeros", Json(zs)}, {"shift", r.i(10, 40)}, {"y_zero", r.i(-50, 50)}});
        mb.output("y");
        compare_all(mb, in, "I8GapAffine config " + std::to_string(idx++), "i8gapaffine.neon");
      }
    }
  }
}

BF_TEST(int8_full_gemm_l8_output_bit_exact_vs_reference) {
  Rng r(4343);
  for (int64_t oc : {16, 37, 128}) {
    for (int64_t k : {1, 3}) {
      const int64_t h = 7, w = 9, ic = 48;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {1, h, w, ic});
      auto wq = r.i8(oc * k * k * ic);
      mb.constant("w", DType::kI8, {oc, k, k, ic}, wq.data());
      std::vector<float> sc(static_cast<size_t>(oc)), bi(static_cast<size_t>(oc));
      for (int64_t o = 0; o < oc; ++o) {
        sc[o] = std::ldexp(r.f(0.5f, 1.f), r.i(-24, -12));
        bi[o] = r.f(-1.f, 1.f) * std::ldexp(1.f, r.i(-10, 0));
      }
      mb.constant("b", DType::kF32, {oc}, bi.data());
      mb.constant("s", DType::kF32, {oc}, sc.data());
      mb.node("I8Conv2D", {"x", "w", oc == 37 ? "" : "b", "s"}, {"y"},
              Json::Object{{"strides", ints({1, 1})}, {"pads", ints({k / 2, k / 2, k / 2, k / 2})}, {"x_zero", r.i(-30, 30)},
                           {"pad_q", 0}, {"output", "l8"}, {"y_exp", 0}});
      mb.output("y");
      compare_all(mb, {r.i8(h * w * ic)}, "I8Conv2D l8 output", "i8conv2d.neon_sdot");
    }
  }
}

// I8Conv2D bit output (bit -1 iff !(acc >= bit_thr[o]), acc before the bias) with every epilogue (i8 / l8 / f32), groups
// starting on and off 16-channel boundaries, channel tails, thresholds on reachable accumulator values (small codes:
// many ties) and at the int32 extremes; I8MaxPool2D of the bits (3x3 / 2 / pad 1 and 2x2 / 2) and binary
// convolutions reading the conv bits and the pooled bits.
BF_TEST(int8_full_gemm_bit_output_and_bit_maxpool_bit_exact_vs_reference) {
  Rng r(4444);
  struct G {
    int64_t oc, k, s, groups, ic;
    const char* out;
  };
  int idx = 0;
  for (G g : std::vector<G>{{64, 7, 2, 1, 3, "i8"},    // the ReActNet stem shape (scaled down)
                            {64, 7, 2, 1, 3, "l8"},
                            {37, 3, 1, 1, 16, "i8"},   // channel tail (scalar bits), partial last bit word
                            {48, 3, 1, 2, 8, "i8"},    // groups of 24: the second starts off a 16-channel boundary
                            {64, 1, 1, 4, 32, "f32"},  // groups of 16
                            {128, 3, 2, 1, 24, "l8"}}) {
    const int64_t h = 13, w = 11, kic = g.ic / g.groups;
    const int64_t oh = (h + 2 * (g.k / 2) - g.k) / g.s + 1, ow = (w + 2 * (g.k / 2) - g.k) / g.s + 1;
    bftest::ModelBuilder mb;
    mb.input("x", DType::kI8, {1, h, w, g.ic});
    std::vector<uint8_t> wq(static_cast<size_t>(g.oc * g.k * g.k * kic)), xq(static_cast<size_t>(h * w * g.ic));
    for (auto& v : wq) v = static_cast<uint8_t>(r.i(-2, 2));
    for (auto& v : xq) v = static_cast<uint8_t>(idx % 2 ? r.i(-128, 127) : r.i(-3, 3));
    mb.constant("w", DType::kI8, {g.oc, g.k, g.k, kic}, wq.data());
    const int xz = r.i(-3, 3);
    std::vector<int32_t> bt(static_cast<size_t>(g.oc));
    for (int64_t o = 0; o < g.oc; ++o) {
      bt[o] = idx % 2 ? r.i(-3000, 3000) : r.i(-12, 12);
      if (o % 11 == 5) bt[o] = o % 2 ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::min();
    }
    mb.constant("bt", DType::kI32, {g.oc}, bt.data());
    Json at = Json::Object{{"strides", ints({g.s, g.s})}, {"pads", ints({g.k / 2, g.k / 2, g.k / 2, g.k / 2})}, {"groups", g.groups},
                           {"x_zero", xz}, {"pad_q", r.i(-4, 4)}};
    std::vector<std::string> ins = {"x", "w", "b"};
    const std::string out = g.out;
    if (out == "i8") {
      std::vector<int32_t> bq(static_cast<size_t>(g.oc)), m(static_cast<size_t>(g.oc)), sh(static_cast<size_t>(g.oc));
      for (int64_t o = 0; o < g.oc; ++o) {
        bq[o] = r.i(-50, 50);
        auto ms = r.mult(0.01, 0.5);
        m[o] = ms.first;
        sh[o] = ms.second;
      }
      mb.constant("b", DType::kI32, {g.oc}, bq.data());
      mb.constant("m", DType::kI32, {g.oc}, m.data());
      mb.constant("sh", DType::kI32, {g.oc}, sh.data());
      ins.insert(ins.end(), {"m", "sh"});
      at["y_zero"] = r.i(-10, 10);
    } else {
      auto sc = r.f32(g.oc, 0.001f, 0.01f), bi = r.f32(g.oc, -0.1f, 0.1f);
      mb.constant("b", DType::kF32, {g.oc}, bi.data());
      mb.constant("s", DType::kF32, {g.oc}, sc.data());
      ins.insert(ins.end(), {"s", ""});
      at["output"] = out;
      if (out == "l8") at["y_exp"] = -1;
    }
    ins.push_back("bt");
    mb.node("I8Conv2D", ins, {"y", "yb"}, at);
    mb.node("I8MaxPool2D", {"yb"}, {"p3"}, Json::Object{{"kernel", ints({3, 3})}, {"strides", ints({2, 2})}, {"pads", ints({1, 1, 1, 1})}});
    mb.node("I8MaxPool2D", {"yb"}, {"p2"}, Json::Object{{"kernel", ints({2, 2})}, {"strides", ints({2, 2})}});
    std::vector<int8_t> bw(static_cast<size_t>(8 * 9 * g.oc));
    for (auto& v : bw) v = r.i(0, 1) ? 1 : -1;
    auto bwp = bftest::pack_pm1(bw, 8 * 9, g.oc);
    mb.constant("cw", DType::kBit, {8, 3, 3, g.oc}, bwp.data());
    auto pm = r.f32(8, -1.f, 1.f), pa = r.f32(8, -9.f, 9.f);
    mb.constant("cpm", DType::kF32, {8}, pm.data());
    mb.constant("cpa", DType::kF32, {8}, pa.data());
    for (const char* src : {"yb", "p3"})
      mb.node("I8BConv2D", {src, "cw", "cpm", "cpa"}, {std::string("c_") + src},
              Json::Object{{"strides", ints({1, 1})}, {"pads", ints({1, 1, 1, 1})}, {"pad_value", 0}, {"output", "f32"}});
    for (const char* o : {"y", "yb", "p3", "p2", "c_yb", "c_p3"}) mb.output(o);
    (void)oh;
    (void)ow;
    compare_all(mb, {xq}, "I8Conv2D bit output config " + std::to_string(idx++), "i8conv2d.neon_sdot");
  }
}

// The bit max pool against its definition (+1 iff some element of the window is +1) on hand-made bit patterns.
BF_TEST(int8_full_bit_maxpool_matches_definition) {
  Rng r(4545);
  for (int64_t c : {5, 32, 64, 70}) {
    const int64_t h = 6, w = 7, words = bit_words(c);
    bftest::ModelBuilder mb;
    mb.input("x", DType::kBit, {1, h, w, c});
    mb.node("I8MaxPool2D", {"x"}, {"y"}, Json::Object{{"kernel", ints({3, 3})}, {"strides", ints({2, 2})}, {"pads", ints({1, 1, 1, 1})}});
    mb.output("y");
    std::vector<uint32_t> xw(static_cast<size_t>(h * w * words), 0u);
    std::vector<int> val(static_cast<size_t>(h * w * c));
    for (int64_t p = 0; p < h * w; ++p)
      for (int64_t ch = 0; ch < c; ++ch) {
        val[p * c + ch] = r.i(0, 7) == 0 ? 1 : -1;  // mostly -1: windows of all -1 occur
        if (val[p * c + ch] < 0) xw[p * words + (ch >> 5)] |= 1u << (ch & 31);
      }
    std::vector<uint8_t> xb(xw.size() * 4);
    std::memcpy(xb.data(), xw.data(), xb.size());
    const int64_t oh = (h + 2 - 3) / 2 + 1, ow = (w + 2 - 3) / 2 + 1;
    std::vector<uint32_t> want(static_cast<size_t>(oh * ow * words), 0u);
    for (int64_t oy = 0; oy < oh; ++oy)
      for (int64_t ox = 0; ox < ow; ++ox)
        for (int64_t ch = 0; ch < c; ++ch) {
          int mx = -1;
          for (int64_t iy = oy * 2 - 1; iy <= oy * 2 + 1; ++iy)
            for (int64_t ix = ox * 2 - 1; ix <= ox * 2 + 1; ++ix)
              if (iy >= 0 && iy < h && ix >= 0 && ix < w) mx = std::max(mx, val[(iy * w + ix) * c + ch]);
          if (mx < 0) want[(oy * ow + ox) * words + (ch >> 5)] |= 1u << (ch & 31);
        }
    std::vector<uint8_t> wb(want.size() * 4);
    std::memcpy(wb.data(), want.data(), wb.size());
    for (const char* st : {kRef, kFull})
      for (int t : {1, 4})
        if (run_model_all(mb, st, t, {xb}) != wb) throw bftest::Failure{"bit max pool differs from its definition, c = " + std::to_string(c)};
  }
}


// Fast paths of 2026-11-03 (ReActNet-18 log2 tuning), each against the reference:
//  * I8BConv2DRes two-phase epilogue (bconv2d.neon_opt compute_res2): 16-channel units with a whole-word / half-word
//    bit store, a last unit of 4 / 8 / 12 channels, K > 31 steps (uint8 runs folded twice), odd pixel counts (pair
//    rows and 1-3 single tail rows), pair-mode 3x3 patch rows of interior pixel pairs (stride 1 / 2, odd widths);
//    also with BITINO_BCONV_RES=v1 (the previous single-phase epilogue);
//  * I8Conv2D on SMMLA with K in kernel-row runs (small ic, 7x7 / 5x5 / 3x5 kernels, strides, dilation, padding,
//    tiles of fewer than 8 rows) and with BITINO_GEMM_I8MM=0 (SDOT);
//  * whole-window 3x3 max / 2x2 average pools (zero points at the extremes, ties of the rounding).
BF_TEST(int8_full_fast_paths_bit_exact_vs_reference) {
  Rng r(5151);
  struct G { int64_t n, h, w, ic, oc, s, xin; };
  int idx = 0;
  for (int v1 = 0; v1 < 2; ++v1) {
    if (v1) setenv("BITINO_BCONV_RES", "v1", 1);
    for (G g : std::vector<G>{{1, 9, 11, 64, 48, 1, 2},    // pair mode, units 0-15 / 16-31 (half words) / 32-47 (whole word)
                              {1, 7, 7, 64, 24, 1, 2},     // 49 pixels (odd), last unit of 8 channels
                              {1, 11, 9, 64, 44, 2, 2},    // stride 2 pair rows, last unit of 12
                              {1, 7, 7, 512, 36, 1, 2},    // K = 4608 (36 steps), 49 pixels: tail rows, last unit of 4
                              {2, 6, 5, 128, 80, 1, 0},    // batch 2, int8 input, 5 units
                              {1, 13, 13, 64, 64, 2, 1}}) {  // l8 input, stride 2, odd output width
      for (int variant = 0; variant < 3; ++variant, ++idx) {
        ResCfg c{g.n, g.h, g.w, g.ic, g.oc, 3, g.s, variant == 2 ? 1 : 0, static_cast<int>(g.xin), variant != 1, variant == 1, true,
                 variant == 2, r.i(-3, 3)};
        c.bit_out = idx % 2 == 1;  // bit_src "out" on every other configuration
        const int64_t oh = (g.h + 2 - 3) / g.s + 1, ow = (g.w + 2 - 3) / g.s + 1;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {g.n, g.h, g.w, g.ic});
        mb.input("res", DType::kI8, {g.n, oh, ow, g.oc});
        std::vector<std::string> outs;
        add_res_node(mb, r, c, "x", "res", "a", outs);
        for (auto& o : outs) mb.output(o);
        std::vector<uint8_t> xin = c.xin == 1 ? res_codes(r, g.n * g.h * g.w * g.ic) : r.i8(g.n * g.h * g.w * g.ic);
        if (c.xin == 2) {  // bit input: an int8 tensor of the right byte size is reinterpreted as words
          bftest::ModelBuilder mb2;
          mb2.input("x", DType::kBit, {g.n, g.h, g.w, g.ic});
          mb2.input("res", DType::kI8, {g.n, oh, ow, g.oc});
          std::vector<std::string> o2;
          Rng r2(9000 + idx);
          add_res_node(mb2, r2, c, "x", "res", "a", o2);
          for (auto& o : o2) mb2.output(o);
          auto xb = r.i8(g.n * g.h * g.w * g.ic / 8);
          compare_all(mb2, {xb, res_codes(r, g.n * oh * ow * g.oc)}, "I8BConv2DRes fast path " + std::to_string(idx), "i8bconv2dres.neon");
          continue;
        }
        compare_all(mb, {xin, res_codes(r, g.n * oh * ow * g.oc)}, "I8BConv2DRes fast path " + std::to_string(idx), "i8bconv2dres.neon");
      }
    }
    if (v1) unsetenv("BITINO_BCONV_RES");
  }
  struct C { int64_t h, w, ic, oc, kh, kw, s, d, pad; };
  for (int sdot = 0; sdot < 2; ++sdot) {
    if (sdot) setenv("BITINO_GEMM_I8MM", "0", 1);
    for (C c : std::vector<C>{{23, 21, 3, 64, 7, 7, 2, 1, 3},   // the stem: rows of 21 bytes in runs of 24
                              {9, 10, 5, 20, 5, 5, 1, 1, 2},    // runs of 25 -> 32 bytes
                              {11, 9, 16, 24, 3, 5, 2, 1, 1},   // non-square kernel, runs of 80 bytes
                              {12, 12, 3, 8, 3, 3, 1, 2, 2},    // dilation 2
                              {5, 3, 8, 13, 3, 3, 1, 1, 0}}) {  // no padding, 3 output pixels (one short tile)
      for (const char* out : {"i8", "l8", "f32"}) {
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {1, c.h, c.w, c.ic});
        auto w = r.i8(c.oc * c.kh * c.kw * c.ic);
        for (auto& v : w) v = static_cast<uint8_t>(std::max<int>(-127, static_cast<int8_t>(v)));
        mb.constant("w", DType::kI8, {c.oc, c.kh, c.kw, c.ic}, w.data());
        Json at = Json::Object{{"strides", ints({c.s, c.s})}, {"dilations", ints({c.d, c.d})}, {"pads", ints({c.pad, c.pad, c.pad, c.pad})},
                               {"x_zero", r.i(-128, 127)}, {"pad_q", r.i(-128, 127)}};
        std::vector<std::string> ins = {"x", "w", "b", "m"};
        if (std::string(out) == "i8") {
          std::vector<int32_t> b(c.oc), m(c.oc), sh(c.oc);
          for (int64_t o = 0; o < c.oc; ++o) {
            b[o] = r.i(-100000, 100000);
            auto mm = r.mult(1e-5, 1e-2);
            m[o] = mm.first;
            sh[o] = mm.second;
          }
          mb.constant("b", DType::kI32, {c.oc}, b.data());
          mb.constant("m", DType::kI32, {c.oc}, m.data());
          mb.constant("s", DType::kI32, {c.oc}, sh.data());
          ins.push_back("s");
          at["y_zero"] = r.i(-20, 20);
        } else {
          auto b = r.f32(c.oc, -1.f, 1.f), sc = r.f32(c.oc, 1e-4f, 1e-2f);
          mb.constant("b", DType::kF32, {c.oc}, b.data());
          mb.constant("m", DType::kF32, {c.oc}, sc.data());
          at["output"] = out;
          if (std::string(out) == "l8") at["y_exp"] = 0;
        }
        mb.node("I8Conv2D", ins, {"y"}, at);
        mb.output("y");
        compare_stages(mb, {r.i8(c.h * c.w * c.ic)}, std::string("I8Conv2D runs ") + out, "i8conv2d.neon_sdot");
      }
    }
    for (int64_t rows : {2, 4, 6, 7, 9}) {  // SMMLA tiles of 1-4 row pairs, odd and even
      const int64_t o = 37, k = 72;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {rows, k});
      auto w = r.i8(o * k);
      mb.constant("w", DType::kI8, {o, k}, w.data());
      auto b = r.f32(o, -1.f, 1.f), sc = r.f32(o, 1e-4f, 1e-2f);
      mb.constant("b", DType::kF32, {o}, b.data());
      mb.constant("m", DType::kF32, {o}, sc.data());
      mb.node("I8Dense", {"x", "w", "b", "m"}, {"y"}, Json::Object{{"x_zero", r.i(-128, 127)}, {"output", "f32"}});
      mb.output("y");
      compare_stages(mb, {r.i8(rows * k)}, "I8Dense short tiles", "i8dense.neon_sdot");
    }
    if (sdot) unsetenv("BITINO_GEMM_I8MM");
  }
  // Session rewrite I8BConv2DRes -> I8Lut (out_lut): against the reference without rewrites; not applied when the
  // l8 tensor has a second reader. Two-phase and v1 epilogues.
  for (int v1 = 0; v1 < 2; ++v1) {
    if (v1) setenv("BITINO_BCONV_RES", "v1", 1);
    for (int second = 0; second < 2; ++second) {
      for (int64_t oc : {64, 40}) {
        const int64_t h = 7, w = 9, ic = 64;
        ResCfg c{1, h, w, ic, oc, 3, 1, 0, 2, true, v1 == 0, true, false, r.i(-2, 2)};
        bftest::ModelBuilder mb;
        mb.input("x", DType::kBit, {1, h, w, ic});
        mb.input("res", DType::kI8, {1, h, w, oc});
        std::vector<std::string> outs;
        add_res_node(mb, r, c, "x", "res", "a", outs);
        auto tab = r.i8(256);
        mb.constant("lut", DType::kI8, {1, 256}, tab.data());
        mb.node("I8Lut", {outs[0], "lut"}, {"lin"});
        mb.output("lin");
        mb.output(outs[1]);
        if (second) mb.output(outs[0]);  // the l8 codes are read twice: no rewrite
        const std::vector<std::vector<uint8_t>> in = {r.i8(h * w * ic / 8), res_codes(r, h * w * oc)};
        g_no_fuse = true;
        auto ref = run_model_all(mb, kRef, 1, in);
        g_no_fuse = false;
        for (const char* st : {kRef, kFull})
          for (int t : {1, 4}) {
            std::map<std::string, std::string> ks;
            if (run_model_all(mb, st, t, in, &ks) != ref) throw bftest::Failure{std::string("out_lut rewrite differs at ") + st};
            if (ks["#absorbed_luts"] != (second ? "0" : "1")) throw bftest::Failure{"unexpected out_lut rewrite count " + ks["#absorbed_luts"]};
          }
      }
    }
    if (v1) unsetenv("BITINO_BCONV_RES");
  }
  for (int64_t c : {32, 48, 40}) {
    for (int zi = 0; zi < 4; ++zi) {
      const int z = std::vector<int>{-128, -58, 0, 127}[zi];
      for (int inc = 0; inc < 2; ++inc) {
        const int64_t h = 9, w = 10;
        bftest::ModelBuilder mb;
        mb.input("x", DType::kI8, {2, h, w, c});
        mb.node("I8MaxPool2D", {"x"}, {"mx"}, Json::Object{{"kernel", ints({3, 3})}, {"strides", ints({2, 2})}, {"pads", ints({1, 1, 1, 1})}});
        mb.node("I8AvgPool2D", {"x"}, {"av"},
                Json::Object{{"kernel", ints({2, 2})}, {"strides", ints({2, 2})}, {"x_zero", z}, {"count_include_pad", inc}});
        mb.node("I8AvgPool2D", {"x"}, {"av1"},
                Json::Object{{"kernel", ints({2, 2})}, {"strides", ints({1, 1})}, {"pads", ints({1, 1, 1, 1})}, {"x_zero", z},
                             {"count_include_pad", inc}});
        mb.output("mx");
        mb.output("av");
        mb.output("av1");
        auto x = r.i8(2 * h * w * c);
        for (size_t i = 0; i < x.size(); i += 5) x[i] = static_cast<uint8_t>(i % 2 ? 127 : 0x80);  // extremes: sums at the ends
        for (size_t i = 3; i < x.size(); i += 11) x[i] = static_cast<uint8_t>(z);
        compare_all(mb, {x}, "pools c " + std::to_string(c) + " z " + std::to_string(z), "");
      }
    }
  }
}

}  // namespace
