// Differential tests for the float_opt tier: every kernel is run on the same
// model and inputs as the reference ("oracle") kernel and compared. Each
// float_opt kernel exists per backend: the 4-lane NEON/portable kernels
// ("<op>.float_opt") everywhere and, on x86-64, the AVX2+FMA kernels
// ("<op>.float_opt_avx2"). Every backend the CPU supports is forced and
// checked, and the default selection must pick the best one; a re-executed
// child with BITINO_DISABLE_FEATURES=avx2 checks the fallback selection.
//
// Tolerance. Ops built from the same float operations as the reference (Add,
// ChannelAffine, Activation, MaxPool2D) must match bit for bit. Conv2D, Dense
// and the average pools accumulate in float (and use FMA on NEON) while the
// reference accumulates in double and rounds once, so we accept
//     |opt - ref| <= 1e-4 * (1 + |ref|).
// Justification: summing K products in float has a worst-case error of about
// K·u·Σ|x_i·w_i| (u = 2^-24 ≈ 6e-8), and a typical (random-walk) error of about
// sqrt(K)·u·||x∘w||_2. Inputs are N(0,1) and weights N(0,1)/sqrt(K), so
// outputs are O(1) and ||x∘w||_2 ≈ 1: for the largest K tested (~600) the
// typical error is ~1.5e-6 and even the worst case on these magnitudes stays
// below ~1e-4·(1+|ref|). Real indexing/tail/border bugs produce O(0.1..1)
// errors, three orders of magnitude above the bound. The AVX2 kernels use
// fused multiply-add like NEON (one rounding per tap instead of two), sum in
// a different order (k-split accumulators for short row tiles) and are held
// to the same bound; Add/ChannelAffine/Activation/MaxPool stay bit-exact.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

#include "bitino/bitino.h"
#include "tools/model_builder.h"
#include "testing.h"

#if defined(__aarch64__)
#define BF_FLOAT_BACKEND0 "neon"
#else
#define BF_FLOAT_BACKEND0 "portable"
#endif

using namespace bitino;

namespace {

constexpr double kRelTol = 1e-4;

// Largest observed |opt - ref| / (1 + |ref|) per test and backend (printed,
// to document the margin under kRelTol): [0] NEON/portable, [1] AVX2.
double g_max_rel_err[2] = {0, 0};
int g_backend_runs[2] = {0, 0};
void report_max_err(const char* test) {
  std::printf("    %s: max |opt-ref|/(1+|ref|) = %.3g %s (%d runs)", test, g_max_rel_err[0], BF_FLOAT_BACKEND0, g_backend_runs[0]);
  if (g_backend_runs[1]) std::printf(", %.3g avx2 (%d runs)", g_max_rel_err[1], g_backend_runs[1]);
  std::printf(", tolerance %.0e\n", kRelTol);
  g_max_rel_err[0] = g_max_rel_err[1] = 0;
  g_backend_runs[0] = g_backend_runs[1] = 0;
}

// The float_opt kernels implementing `base` (e.g. "conv2d.float_opt") that
// this CPU can run, lowest priority first: the 4-lane kernel, then (x86-64
// with avx2+fma) the AVX2 kernel "<base>_avx2".
// Ops with a single kernel for every backend (e.g. pad.float_opt) have no
// "_avx2" entry in the registry.
std::vector<std::string> float_opt_variants(const std::string& base) {
  std::vector<std::string> v = {base};
#if defined(__x86_64__)
  bool registered = false;
  for (const auto& k : OpRegistry::global().all_kernels()) registered |= k.name == base + "_avx2";
  if (registered && cpu_info().has(kFeatAvx2 | kFeatFma)) v.push_back(base + "_avx2");
#endif
  return v;
}

Json ints(std::initializer_list<int64_t> v) {
  Json::Array a;
  for (auto x : v) a.push_back(Json(x));
  return Json(a);
}

std::vector<float> randn(size_t n, uint32_t seed, float scale = 1.f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = nd(rng) * scale;
  return v;
}

struct RunResult {
  std::vector<float> out;
  std::string kernel;  // kernel chosen for the (single) node
};

RunResult run_model(const std::shared_ptr<Model>& m, const KernelPolicy& policy, int threads, uint32_t seed) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = policy;
  Session s(m, o);
  for (size_t i = 0; i < s.num_inputs(); ++i) {
    Tensor& t = s.input(i);
    auto v = randn(static_cast<size_t>(t.size()), seed + 1000 * static_cast<uint32_t>(i));
    std::memcpy(t.raw(), v.data(), v.size() * sizeof(float));
  }
  s.run();
  const Tensor& y = s.output(0);
  RunResult r;
  r.out.assign(y.data<float>(), y.data<float>() + y.size());
  r.kernel = s.kernel_choices().begin()->second;
  return r;
}

RunResult run_model(const std::shared_ptr<Model>& m, const std::set<std::string>& tiers, int threads, uint32_t seed) {
  return run_model(m, KernelPolicy::tiers(tiers), threads, seed);
}

KernelPolicy forced(const std::shared_ptr<Model>& m, const std::string& kernel) {
  KernelPolicy p;
  p.force[m->graph().nodes.at(0).op] = kernel;
  return p;
}

// Compares every float_opt backend of `expect_kernel` (see
// float_opt_variants) against reference for the given thread counts, after
// checking that the float_opt policy selects the best one. `exact` demands
// bit-identical (numerically equal) results. Note the kernels run small
// problems inline (see parallel_grain()), so cases meant to exercise the
// multithreaded split must be large enough.
void check_against_reference(const std::shared_ptr<Model>& m, const std::string& what, bool exact,
                             const std::string& expect_kernel, uint32_t seed = 17,
                             std::initializer_list<int> thread_counts = {1, 3, 4}) {
  RunResult ref = run_model(m, {"reference"}, 1, seed);
  const std::vector<std::string> variants = float_opt_variants(expect_kernel);
  const std::string chosen = run_model(m, {"float_opt"}, 1, seed).kernel;
  if (chosen != variants.back()) throw bftest::Failure{what + ": expected kernel " + variants.back() + ", got " + chosen};
  for (const std::string& variant : variants) {
    const int backend = variant.size() > 5 && variant.compare(variant.size() - 5, 5, "_avx2") == 0 ? 1 : 0;
    for (int threads : thread_counts) {
      RunResult opt = run_model(m, forced(m, variant), threads, seed);
      if (opt.kernel != variant) throw bftest::Failure{what + ": forced " + variant + ", got " + opt.kernel};
      if (opt.out.size() != ref.out.size()) throw bftest::Failure{what + ": size mismatch"};
      ++g_backend_runs[backend];
      for (size_t i = 0; i < ref.out.size(); ++i) {
        double r = ref.out[i], o = opt.out[i];
        if (std::isfinite(r)) g_max_rel_err[backend] = std::max(g_max_rel_err[backend], std::fabs(o - r) / (1.0 + std::fabs(r)));
        bool ok = r == o || (exact ? false : std::fabs(o - r) <= kRelTol * (1.0 + std::fabs(r)));
        if (!ok) {
          std::ostringstream os;
          os << what << " (" << opt.kernel << ", threads " << threads << "): element " << i << " opt " << o << " ref " << r;
          throw bftest::Failure{os.str()};
        }
      }
    }
  }
}

// ------------------------------------------------------------------ Conv2D
struct ConvCase {
  int64_t n, h, w, ic, oc, kh, kw;
  int64_t sh, sw, dh, dw;
  int64_t pt, pb, pl, pr;
  bool depthwise;
  std::string act;
  float pad_value;
  bool bias;
  int64_t groups = 1;  // grouped (non-depthwise) convolution when > 1
};

std::string describe(const ConvCase& c) {
  std::ostringstream os;
  os << (c.depthwise ? "dwconv" : c.groups > 1 ? "gconv" + std::to_string(c.groups) : "conv") << " n" << c.n << " " << c.h << "x" << c.w << "x" << c.ic << "->" << c.oc << " k"
     << c.kh << "x" << c.kw << " s" << c.sh << "," << c.sw << " d" << c.dh << "," << c.dw << " p" << c.pt << "," << c.pb << ","
     << c.pl << "," << c.pr << " " << c.act << " pv" << c.pad_value << (c.bias ? " bias" : "");
  return os.str();
}

std::shared_ptr<Model> conv_model(const ConvCase& c, uint32_t seed) {
  int64_t kic = c.depthwise ? 1 : c.ic / c.groups, oc = c.depthwise ? c.ic : c.oc;
  auto w = randn(static_cast<size_t>(oc * c.kh * c.kw * kic), seed, 1.f / std::sqrt(float(c.kh * c.kw * kic)));
  auto b = randn(static_cast<size_t>(oc), seed + 1, 0.5f);
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kF32, {oc, c.kh, c.kw, kic}, w.data());
  if (c.bias) mb.constant("b", DType::kF32, {oc}, b.data());
  mb.input("x", DType::kF32, {c.n, c.h, c.w, c.ic});
  Json attrs = Json::Object{{"strides", ints({c.sh, c.sw})},
                            {"dilations", ints({c.dh, c.dw})},
                            {"pads", ints({c.pt, c.pb, c.pl, c.pr})},
                            {"pad_value", Json(static_cast<double>(c.pad_value))},
                            {"activation", c.act}};
  if (c.depthwise) attrs["groups"] = Json(c.ic);
  else if (c.groups > 1) attrs["groups"] = Json(c.groups);
  mb.node("Conv2D", {"x", "w", c.bias ? "b" : ""}, {"y"}, attrs);
  mb.output("y");
  return mb.build();
}

}  // namespace

#if defined(__aarch64__) || (defined(__arm__) && defined(__ARM_NEON))
BF_TEST(float_conv2d_neon_gemm_tile_order_exact) {
  // conv2d.float_opt on NEON computes every output of its 8-row tiles as one
  // fused multiply-add chain: acc = bias, then acc = fma(a[k], w[k], acc) for
  // k = 0..K-1 in (ky, kx, c) order (K blocks continue the chain through the
  // output), then the activation. The tile loop is inline asm: check it
  // reproduces that chain bit for bit (M a multiple of 8 so no row takes the
  // GEMV path; K = 45 with a scalar k tail, K = 360 with two K blocks, 1x1
  // reading the input in place, N tails of the 8-wide panels).
  // 32-bit ARM (two 4-row passes of VMLA in inline asm): the same chain with
  // acc = acc + a[k] * w[k] rounded after the multiply and after the add.
  struct Case {
    int64_t n, h, w, ic, oc, k, pad;
    std::string act;
  };
  int idx = 0;
  for (const Case& c : {Case{1, 8, 8, 5, 13, 3, 1, "none"}, Case{2, 8, 4, 40, 9, 3, 1, "relu"}, Case{1, 4, 8, 48, 16, 1, 0, "none"},
                        Case{1, 8, 6, 7, 8, 5, 2, "none"}, Case{1, 4, 4, 300, 3, 1, 0, "relu"}}) {
    const int64_t kk = c.k * c.k * c.ic;
    auto w = randn(static_cast<size_t>(c.oc * kk), 50 + idx, 0.3f);
    auto b = randn(static_cast<size_t>(c.oc), 60 + idx, 0.5f);
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kF32, {c.oc, c.k, c.k, c.ic}, w.data());
    mb.constant("b", DType::kF32, {c.oc}, b.data());
    mb.input("x", DType::kF32, {c.n, c.h, c.w, c.ic});
    mb.node("Conv2D", {"x", "w", "b"}, {"y"},
            Json(Json::Object{{"pads", ints({c.pad, c.pad, c.pad, c.pad})}, {"activation", c.act}}));
    mb.output("y");
    auto m = mb.build();
    const RunResult opt = run_model(m, forced(m, "conv2d.float_opt"), 1, 70 + idx);
    EXPECT_EQ(opt.kernel, std::string("conv2d.float_opt"));
    const auto x = randn(static_cast<size_t>(c.n * c.h * c.w * c.ic), 70 + idx);  // run_model's input
    size_t i = 0;
    for (int64_t n = 0; n < c.n; ++n)
      for (int64_t oy = 0; oy < c.h; ++oy)
        for (int64_t ox = 0; ox < c.w; ++ox)
          for (int64_t o = 0; o < c.oc; ++o, ++i) {
            float acc = b[o];
            for (int64_t ky = 0; ky < c.k; ++ky)
              for (int64_t kx = 0; kx < c.k; ++kx)
                for (int64_t ci = 0; ci < c.ic; ++ci) {
                  const int64_t iy = oy - c.pad + ky, ix = ox - c.pad + kx;
                  const float a = iy < 0 || iy >= c.h || ix < 0 || ix >= c.w ? 0.f : x[((n * c.h + iy) * c.w + ix) * c.ic + ci];
#if defined(__aarch64__)
                  acc = std::fmaf(a, w[((o * c.k + ky) * c.k + kx) * c.ic + ci], acc);
#else
                  acc = acc + a * w[((o * c.k + ky) * c.k + kx) * c.ic + ci];  // -ffp-contract=off: two roundings
#endif
                }
            if (c.act == "relu") acc = acc > 0.f ? acc : 0.f;
            if (std::memcmp(&acc, &opt.out[i], 4) != 0)
              throw bftest::Failure{"case " + std::to_string(idx) + " element " + std::to_string(i) + ": " + std::to_string(opt.out[i]) +
                                    " vs fma chain " + std::to_string(acc)};
          }
    ++idx;
  }
}
#endif

BF_TEST(float_conv2d_gemm_matches_reference) {
  // n, h, w, ic, oc, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, dw?, act, pad_value, bias
  std::vector<ConvCase> cases = {
      {1, 7, 7, 16, 24, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "none", 0, true},      // 1x1 direct, M tail
      {1, 5, 9, 5, 13, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "relu", 0, true},       // direct, K and N tails
      {1, 6, 6, 300, 20, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "relu6", 0, true},    // direct, K blocked (300 > 256)
      {1, 9, 8, 12, 17, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, false, "none", 0, false},     // 1x1 stride 2 (im2col)
      {1, 6, 6, 8, 8, 1, 1, 1, 1, 1, 1, 1, 0, 2, 1, false, "none", 0.75f, true},    // 1x1 padded, pad_value
      {1, 10, 11, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "relu", 0, true},    // 3x3 same
      {1, 13, 12, 7, 9, 3, 3, 2, 2, 1, 1, 0, 1, 0, 1, false, "relu6", 0, true},     // asymmetric (TF same, s2)
      {1, 14, 14, 6, 10, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, false, "none", 0, true},     // dilation 2
      {1, 11, 13, 5, 11, 5, 3, 2, 1, 1, 2, 2, 1, 0, 3, false, "relu", 0.5f, true},  // rect kernel, mixed s/d/p
      {1, 33, 31, 3, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, false, "relu", 0, true},      // RGB stem (QuickNet)
      {1, 40, 38, 3, 64, 7, 7, 2, 2, 1, 1, 3, 3, 3, 3, false, "none", 0, true},     // 7x7 stem (Bi-Real)
      {1, 9, 9, 64, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "relu6", -1.f, true},  // K = 576: 3 K blocks
      {2, 6, 5, 9, 12, 3, 3, 1, 2, 1, 1, 1, 1, 1, 1, false, "none", 0, true},       // batch 2
      {1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "none", 0, true},        // M = 6 (masked 8-row tile)
      {1, 1, 3, 7, 9, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "relu", 0, true},        // M = 3 (GEMV rows)
      {1, 3, 3, 4, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, false, "none", 0, true},        // M = 1
      {1, 5, 4, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 2.f, true},      // ic = oc = 1
      {1, 4, 4, 2, 6, 3, 3, 1, 1, 1, 1, 4, 4, 4, 4, false, "relu", 1.f, true},      // pads larger than kernel
      {1, 21, 19, 24, 70, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "relu", 0, true},    // several pixel blocks, N tail
      {1, 8, 9, 100, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0.3f, true},  // K = 900: K blocks start mid-tap
      {1, 9, 7, 37, 10, 5, 5, 1, 2, 1, 1, 2, 2, 3, 1, false, "relu", -0.5f, true},  // K = 925, mid-tap, s_w 2
      {1, 12, 10, 2, 6, 3, 5, 1, 1, 3, 1, 3, 3, 2, 2, false, "relu6", 1.f, true},   // merged runs, dilation_h only
      {1, 7, 7, 256, 516, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "relu", 0, true},    // small M, big N*K: N split, tail chunk
  };
  uint32_t seed = 1;
  for (const auto& c : cases) {
    check_against_reference(conv_model(c, seed), describe(c), false, "conv2d.float_opt", seed + 1);
    seed += 2;
  }
  report_max_err("conv2d_gemm");
}

BF_TEST(float_conv2d_depthwise_matches_reference) {
  std::vector<ConvCase> cases;
  uint32_t i = 0;
  for (int64_t ch : {2, 3, 4, 8, 13, 16, 20, 37, 64}) {
    const char* act = i % 3 == 0 ? "none" : i % 3 == 1 ? "relu" : "relu6";
    cases.push_back({1, 9, 10, ch, ch, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, true, act, 0, true});
    cases.push_back({1, 12, 11, ch, ch, 3, 3, 2, 2, 1, 1, 0, 1, 0, 1, true, act, 0, i % 2 == 0});
    ++i;
  }
  cases.push_back({2, 15, 14, 24, 24, 5, 5, 1, 2, 2, 1, 4, 3, 2, 1, true, "relu", -1.5f, true});  // dilation, rect stride
  cases.push_back({1, 8, 8, 17, 17, 3, 2, 1, 1, 1, 1, 2, 0, 1, 2, true, "relu6", 0.25f, true});   // rect kernel, pad_value
  cases.push_back({1, 3, 2, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, true, "none", 0, true});          // all border
  cases.push_back({1, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 0, 1, 0, 1, true, "relu", 0, true});      // QuickNet shape
  cases.push_back({1, 56, 56, 36, 36, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, true, "relu6", 0, true});     // big enough to split rows
  cases.push_back({1, 13, 12, 12, 12, 3, 3, 2, 1, 2, 2, 2, 2, 2, 2, true, "relu", 0.5f, true});   // 3x3 dilated, s(2,1)
  cases.push_back({1, 10, 17, 6, 6, 3, 3, 1, 3, 1, 1, 0, 0, 2, 0, true, "none", 0, false});      // 3x3 s_w 3, one-sided pad
  uint32_t seed = 100;
  for (const auto& c : cases) {
    check_against_reference(conv_model(c, seed), describe(c), false, "conv2d_depthwise.float_opt", seed + 1);
    seed += 2;
  }
  report_max_err("conv2d_depthwise");
}

BF_TEST(float_conv2d_grouped_matches_reference) {
  // Grouped (not depthwise) convolution: one GEMM per group. Odd channel
  // counts per group (kic, oc/groups), strides, padding (incl. pad_value and
  // asymmetric), dilation, K blocking, the 1x1 direct path with a channel
  // offset per group, the MeliusNet22 shapes (kic 8 / 4, 8 outputs per group:
  // the 8-wide AVX2 panels), batch 2, a depthwise-like layer with channel
  // multiplier 2 (not taken by the depthwise kernel) and the N split; 1-4
  // threads.
  // n, h, w, ic, oc, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, dw?, act, pad_value, bias, groups
  std::vector<ConvCase> cases = {
      {1, 9, 9, 12, 15, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "relu", 0, true, 3},       // kic 4, 5 outputs per group
      {1, 10, 11, 15, 21, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, false, "none", 0, true, 3},     // kic 5, 7 per group, stride 2
      {1, 14, 13, 6, 9, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, false, "relu6", 0, false, 3},     // kic 2, dilation 2, no bias
      {1, 9, 8, 10, 6, 3, 2, 1, 2, 1, 1, 0, 2, 1, 0, false, "relu6", 0.5f, true, 2},    // rect kernel, asymmetric, pad_value
      {1, 11, 13, 9, 12, 5, 3, 2, 1, 1, 2, 2, 1, 0, 3, false, "relu", -0.75f, true, 3}, // 5x3, mixed s/d/p
      {1, 7, 7, 16, 24, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "none", 0, true, 4},       // 1x1 direct (lda = ic)
      {1, 9, 7, 14, 10, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, false, "relu", 0, true, 2},       // 1x1 strided (im2col)
      {1, 20, 20, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0, true, 4},     // MeliusNet22: kic 8, 8 per group
      {1, 20, 20, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0, true, 8},     // MeliusNet22: kic 4, 8 per group
      {1, 8, 8, 128, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "relu", 0.3f, true, 2},   // K = 576: 3 K blocks, mid-tap
      {1, 8, 9, 6, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0, true, 6},        // kic 1, multiplier 2
      {2, 28, 28, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "relu", 0, true, 8},     // batch 2, split over threads
      {1, 5, 5, 64, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, "none", 0, true, 2},      // small M, wide N: N split
      {1, 3, 3, 8, 6, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, false, "none", 0, true, 2},         // M = 1 (GEMV rows)
      {1, 6, 7, 21, 35, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "relu", 0, true, 7},       // 7 groups of 3 -> 5
  };
  uint32_t seed = 50;
  for (const auto& c : cases) {
    check_against_reference(conv_model(c, seed), describe(c), false, "conv2d_grouped.float_opt", seed + 1, {1, 2, 3, 4});
    seed += 2;
  }
  report_max_err("conv2d_grouped");
}

BF_TEST(float_conv2d_kernel_selection) {
  // Grouped convolution is a float_opt kernel (it used to fall back to the
  // reference). Depthwise layers keep the depthwise kernel (higher priority
  // than the grouped GEMM).
  ConvCase grouped{1, 6, 6, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0, false, 2};
  auto m = conv_model(grouped, 5);
  EXPECT_EQ(run_model(m, {"reference", "float_opt"}, 1, 3).kernel, float_opt_variants("conv2d_grouped.float_opt").back());
  EXPECT_EQ(run_model(m, {"reference"}, 1, 3).kernel, std::string("conv2d.reference"));
  ConvCase dw{1, 6, 6, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, true, "none", 0, true};
  EXPECT_EQ(run_model(conv_model(dw, 2), {"float_opt"}, 1, 3).kernel, float_opt_variants("conv2d_depthwise.float_opt").back());
  // Default policy (all tiers) picks float_opt for a plain convolution.
  ConvCase c{1, 6, 6, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0, true};
  Session s(conv_model(c, 1), {});
  EXPECT_EQ(s.kernel_choices().begin()->second, float_opt_variants("conv2d.float_opt").back());
}

// With BITINO_DISABLE_FEATURES=avx2 (or fma) the AVX2 kernels must not be
// selected: a re-executed child checks that the default float_opt selection
// falls back to the portable kernels.
BF_TEST(float_feature_masking_selects_portable) {
#if defined(__x86_64__)
  if (std::getenv("BF_FLOAT_CHILD")) return;
  if (!cpu_info().has(kFeatAvx2 | kFeatFma)) return;  // already the portable path
  for (const char* mask : {"avx2", "fma"}) {
    const char* prev = std::getenv("BITINO_DISABLE_FEATURES");
    std::string dis = (prev && *prev ? std::string(prev) + "," : std::string()) + mask;
    int rc = bftest::run_self("float_gating_child", {"BITINO_DISABLE_FEATURES=" + dis, "BF_FLOAT_CHILD=1"});
    if (rc != 0)
      throw bftest::Failure{std::string("masked child failed, rerun: BITINO_DISABLE_FEATURES=") + dis +
                            " BF_FLOAT_CHILD=1 bitino_tests float_gating_child"};
  }
#endif
}

BF_TEST(float_gating_child) {
  if (!std::getenv("BF_FLOAT_CHILD")) return;  // only meaningful when spawned
  EXPECT_TRUE(!cpu_info().has(kFeatAvx2 | kFeatFma));
  ConvCase c{1, 6, 6, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0, true};
  EXPECT_EQ(run_model(conv_model(c, 1), {"float_opt"}, 1, 3).kernel, std::string("conv2d.float_opt"));
  ConvCase g{1, 6, 6, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, false, "none", 0, true, 2};
  EXPECT_EQ(run_model(conv_model(g, 1), {"float_opt"}, 1, 3).kernel, std::string("conv2d_grouped.float_opt"));
  ConvCase dw{1, 6, 6, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, true, "none", 0, true};
  EXPECT_EQ(run_model(conv_model(dw, 1), {"float_opt"}, 1, 3).kernel, std::string("conv2d_depthwise.float_opt"));
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {1, 9});
  mb.input("b", DType::kF32, {1, 9});
  mb.node("Add", {"a", "b"}, {"y"});
  mb.output("y");
  EXPECT_EQ(run_model(mb.build(), {"float_opt"}, 1, 3).kernel, std::string("add.float_opt"));
}

// -------------------------------------------------------------------- Dense
BF_TEST(float_dense_matches_reference) {
  uint32_t seed = 300;
  struct D {
    Shape x;
    int64_t o;
    bool bias;
    const char* act;
  };
  std::vector<D> cases = {
      {{1, 512}, 1000, true, "none"},  // classifier head (GEMV)
      {{1, 1}, 1, true, "relu"},       {{1, 3}, 7, false, "none"},   {{2, 4}, 8, true, "relu6"},
      {{3, 7}, 9, true, "relu"},       {{5, 64}, 100, true, "none"}, {{8, 100}, 33, false, "relu"},
      {{9, 513}, 17, true, "relu6"},   {{17, 33}, 64, true, "none"}, {{2, 3, 20}, 12, true, "relu"},  // rank-3 input
      {{40, 300}, 70, true, "none"},   // several row tiles, K > 256 (two K blocks)
      {{3, 1024}, 700, true, "relu"},  // GEMV rows with the N split across threads
      {{64, 256}, 300, true, "none"},  // several pixel blocks in parallel
  };
  for (const auto& d : cases) {
    int64_t k = d.x.back();
    auto w = randn(static_cast<size_t>(d.o * k), seed, 1.f / std::sqrt(float(k)));
    auto b = randn(static_cast<size_t>(d.o), seed + 1);
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kF32, {d.o, k}, w.data());
    if (d.bias) mb.constant("b", DType::kF32, {d.o}, b.data());
    mb.input("x", DType::kF32, d.x);
    mb.node("Dense", {"x", "w", d.bias ? "b" : ""}, {"y"}, Json(Json::Object{{"activation", d.act}}));
    mb.output("y");
    std::string what = "dense " + shape_str(d.x) + " -> " + std::to_string(d.o) + " " + d.act;
    check_against_reference(mb.build(), what, false, "dense.float_opt", seed);
    seed += 2;
  }
  report_max_err("dense");
}

// -------------------------------------------------------------- elementwise
BF_TEST(float_add_matches_reference_exactly) {
  uint32_t seed = 400;
  for (Shape s : {Shape{1, 1}, Shape{1, 3}, Shape{2, 5, 7}, Shape{1, 37, 41, 23}, Shape{1, 56, 56, 64}})
    for (const char* act : {"none", "relu", "relu6"}) {
      bftest::ModelBuilder mb;
      mb.input("a", DType::kF32, s);
      mb.input("b", DType::kF32, s);
      mb.node("Add", {"a", "b"}, {"y"}, Json(Json::Object{{"activation", act}}));
      mb.output("y");
      check_against_reference(mb.build(), "add " + shape_str(s) + " " + act, true, "add.float_opt", seed++);
    }
}

BF_TEST(float_channel_affine_matches_reference_exactly) {
  uint32_t seed = 500;
  for (int64_t c : {1, 3, 4, 5, 16, 37})
    for (int mode = 0; mode < 3; ++mode) {  // scale+bias, scale only, bias only
      auto sc = randn(static_cast<size_t>(c), seed), bi = randn(static_cast<size_t>(c), seed + 1);
      bftest::ModelBuilder mb;
      if (mode != 2) mb.constant("s", DType::kF32, {c}, sc.data());
      if (mode != 1) mb.constant("b", DType::kF32, {c}, bi.data());
      mb.input("x", DType::kF32, {2, 9, 11, c});
      mb.node("ChannelAffine", {"x", mode != 2 ? "s" : "", mode != 1 ? "b" : ""}, {"y"});
      mb.output("y");
      check_against_reference(mb.build(), "channel_affine c" + std::to_string(c) + " mode " + std::to_string(mode), true,
                              "channel_affine.float_opt", seed);
      seed += 2;
    }
  // Large enough to be split across threads.
  auto sc = randn(64, 1), bi = randn(64, 2);
  bftest::ModelBuilder mb;
  mb.constant("s", DType::kF32, {64}, sc.data());
  mb.constant("b", DType::kF32, {64}, bi.data());
  mb.input("x", DType::kF32, {1, 56, 56, 64});
  mb.node("ChannelAffine", {"x", "s", "b"}, {"y"});
  mb.output("y");
  check_against_reference(mb.build(), "channel_affine large", true, "channel_affine.float_opt", 3);
}

BF_TEST(float_activation_matches_reference_exactly) {
  uint32_t seed = 600;
  for (Shape s : {Shape{1, 5}, Shape{1, 7, 9, 13}, Shape{1, 40, 40, 32}}) {
    int64_t c = s.back();
    std::vector<Json> attrs = {
        Json(Json::Object{{"type", "relu"}}),
        Json(Json::Object{{"type", "relu6"}}),
        Json(Json::Object{{"type", "hardtanh"}, {"min", Json(-0.5)}, {"max", Json(2.0)}}),
        Json(Json::Object{{"type", "hardtanh"}}),
        Json(Json::Object{{"type", "leaky_relu"}, {"alpha", Json(0.2)}}),
        Json(Json::Object{{"type", "prelu"}}),
    };
    for (const Json& a : attrs) {
      bool prelu = a["type"].as_string() == "prelu";
      auto slope = randn(static_cast<size_t>(c), seed, 0.3f);
      bftest::ModelBuilder mb;
      if (prelu) mb.constant("slope", DType::kF32, {c}, slope.data());
      mb.input("x", DType::kF32, s);
      mb.node("Activation", prelu ? std::vector<std::string>{"x", "slope"} : std::vector<std::string>{"x"}, {"y"}, a);
      mb.output("y");
      check_against_reference(mb.build(), "activation " + a.dump() + " " + shape_str(s), true, "activation.float_opt", seed++);
    }
  }
  // The transcendental types (gelu, sigmoid, tanh, hardswish) and sign / sign_zero
  // are handled by activation_ext.float_opt (tests/cpp/test_transformer_opt.cc).
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, 8});
  mb.node("Activation", {"x"}, {"y"}, Json(Json::Object{{"type", "sigmoid"}}));
  mb.output("y");
  auto m = mb.build();
  EXPECT_EQ(run_model(m, {"reference", "float_opt"}, 1, 1).kernel.rfind("activation_ext.float_opt", 0), size_t(0));
  EXPECT_EQ(run_model(m, {"reference"}, 1, 1).kernel, std::string("activation.reference"));
}

// -------------------------------------------------------------------- pools
namespace {

Json ints(const std::vector<int64_t>& v) {
  Json::Array a;
  for (auto x : v) a.push_back(Json(x));
  return Json(a);
}

std::shared_ptr<Model> pool_model(const char* op, const Shape& x, const std::vector<int64_t>& k, const std::vector<int64_t>& st,
                                  const std::vector<int64_t>& pd, int include_pad) {
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, x);
  mb.node(op, {"x"}, {"y"},
          Json(Json::Object{{"kernel", ints(k)}, {"strides", ints(st)}, {"pads", ints(pd)}, {"count_include_pad", Json(include_pad)}}));
  mb.output("y");
  return mb.build();
}

}  // namespace

BF_TEST(float_pools_match_reference) {
  uint32_t seed = 700;
  struct P {
    Shape x;
    std::vector<int64_t> k, st, pd;
  };
  std::vector<P> cases = {
      {{1, 8, 8, 16}, {2, 2}, {2, 2}, {0, 0, 0, 0}},
      {{1, 9, 11, 21}, {3, 3}, {2, 2}, {1, 1, 1, 1}},
      {{2, 7, 6, 4}, {3, 3}, {1, 1}, {0, 2, 1, 0}},
      {{1, 10, 9, 3}, {3, 2}, {1, 2}, {1, 0, 0, 1}},
      {{1, 5, 5, 37}, {2, 2}, {1, 1}, {2, 2, 2, 2}},  // windows entirely in padding
      {{1, 56, 56, 64}, {3, 3}, {2, 2}, {1, 1, 1, 1}},
  };
  for (const auto& p : cases) {
    std::string geo = shape_str(p.x) + " k" + std::to_string(p.k[0]) + "x" + std::to_string(p.k[1]) + " s" + std::to_string(p.st[0]);
    check_against_reference(pool_model("MaxPool2D", p.x, p.k, p.st, p.pd, 0), "maxpool " + geo, true, "maxpool2d.float_opt", seed++);
    for (int inc : {0, 1})
      check_against_reference(pool_model("AvgPool2D", p.x, p.k, p.st, p.pd, inc), "avgpool " + geo + " inc" + std::to_string(inc),
                              false, "avgpool2d.float_opt", seed++);
  }
  report_max_err("pools");
}

BF_TEST(float_global_avg_pool_matches_reference) {
  uint32_t seed = 800;
  for (Shape s : {Shape{1, 1, 1, 3}, Shape{2, 7, 7, 512}, Shape{1, 5, 3, 37}, Shape{3, 4, 4, 16}, Shape{1, 56, 56, 20}})
    for (int keep : {0, 1}) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kF32, s);
      mb.node("GlobalAvgPool", {"x"}, {"y"}, Json(Json::Object{{"keepdims", Json(keep)}}));
      mb.output("y");
      check_against_reference(mb.build(), "gap " + shape_str(s), false, "global_avg_pool.float_opt", seed++);
    }
  report_max_err("global_avg_pool");
}

BF_TEST(float_pad_matches_reference_exactly) {
  uint32_t seed = 900;
  struct P {
    Shape x;
    std::vector<int64_t> pads;
    double value;
  };
  std::vector<P> cases = {
      {{1, 5, 6, 7}, {0, 0, 1, 1, 1, 1, 0, 0}, 0.0},    // spatial "same" padding (MeliusNet)
      {{2, 4, 3, 5}, {0, 1, 2, 0, 0, 3, 1, 2}, -1.5},   // every dim, asymmetric, batch padding
      {{3, 9}, {1, 2, 0, 4}, 0.25},                     // rank 2
      {{17}, {3, 5}, 2.0},                              // rank 1
      {{1, 6, 6, 8}, {0, 0, 0, 0, 0, 0, 0, 0}, 7.0},    // no padding
      {{1, 56, 56, 64}, {0, 0, 1, 1, 1, 1, 0, 0}, 0.0}, // large enough to split over threads
  };
  for (const auto& c : cases) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kF32, c.x);
    Json::Array pa;
    for (auto p : c.pads) pa.push_back(Json(p));
    mb.node("Pad", {"x"}, {"y"}, Json(Json::Object{{"pads", Json(pa)}, {"value", Json(c.value)}}));
    mb.output("y");
    check_against_reference(mb.build(), "pad " + shape_str(c.x), true, "pad.float_opt", seed++);
  }
}

BF_TEST(float_small_network_matches_reference) {
  // Conv -> depthwise -> 1x1 conv -> add -> pool -> dense: exercises buffer
  // reuse between float_opt kernels in one session.
  auto w1 = randn(16 * 3 * 3 * 3, 1, 0.2f), w2 = randn(16 * 3 * 3, 2, 0.3f), w3 = randn(16 * 16, 3, 0.25f);
  auto b1 = randn(16, 4), b2 = randn(16, 5), b3 = randn(16, 6), wd = randn(10 * 16, 7, 0.25f), bd = randn(10, 8);
  bftest::ModelBuilder mb;
  mb.constant("w1", DType::kF32, {16, 3, 3, 3}, w1.data());
  mb.constant("b1", DType::kF32, {16}, b1.data());
  mb.constant("w2", DType::kF32, {16, 3, 3, 1}, w2.data());
  mb.constant("b2", DType::kF32, {16}, b2.data());
  mb.constant("w3", DType::kF32, {16, 1, 1, 16}, w3.data());
  mb.constant("b3", DType::kF32, {16}, b3.data());
  mb.constant("wd", DType::kF32, {10, 16}, wd.data());
  mb.constant("bd", DType::kF32, {10}, bd.data());
  mb.input("x", DType::kF32, {1, 17, 17, 3});
  mb.node("Conv2D", {"x", "w1", "b1"}, {"c1"},
          Json(Json::Object{{"strides", ints({2, 2})}, {"pads", ints({1, 1, 1, 1})}, {"activation", "relu"}}));
  mb.node("Conv2D", {"c1", "w2", "b2"}, {"c2"},
          Json(Json::Object{{"groups", Json(16)}, {"pads", ints({1, 1, 1, 1})}, {"activation", "relu6"}}));
  mb.node("Conv2D", {"c2", "w3", "b3"}, {"c3"});
  mb.node("Add", {"c1", "c3"}, {"s"}, Json(Json::Object{{"activation", "relu"}}));
  mb.node("MaxPool2D", {"s"}, {"p"}, Json(Json::Object{{"kernel", ints({3, 3})}, {"strides", ints({2, 2})}, {"pads", ints({1, 1, 1, 1})}}));
  mb.node("GlobalAvgPool", {"p"}, {"g"});
  mb.node("Dense", {"g", "wd", "bd"}, {"y"});
  mb.output("y");
  auto m = mb.build();
  RunResult ref = run_model(m, {"reference"}, 1, 9);
  for (int threads : {1, 4}) {
    SessionOptions o;
    o.num_threads = threads;
    o.policy = KernelPolicy::tiers({"float_opt"});
    Session s(m, o);
    for (const auto& kv : s.kernel_choices()) EXPECT_TRUE(kv.second.find("float_opt") != std::string::npos);
    RunResult opt = run_model(m, {"float_opt"}, threads, 9);
    for (size_t i = 0; i < ref.out.size(); ++i) EXPECT_NEAR(opt.out[i], ref.out[i], kRelTol * (1 + std::fabs(ref.out[i])));
  }
}
