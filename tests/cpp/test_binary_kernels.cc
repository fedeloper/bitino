// Level-1/2 differential tests: every optimized BConv2D/BDense kernel must
// agree EXACTLY with the reference kernel (binary accumulation is integer and
// the post-scale is applied identically in float).
#include <cstring>
#include <random>

#include "bitino/bitino.h"
#include "tools/model_builder.h"
#include "testing.h"

using namespace bitino;

namespace {

struct BConvCase {
  int64_t h, w, ic, oc, k, stride, pad, dil;
  int pad_value;
  std::string binarize;
  bool bit_out;
  bool with_threshold;
  std::string activation = "none";
  bool neg_mul = false;  // alternate post_mul signs (exercises -0.0 / activation edge cases)
  bool no_add = false;   // omit post_add
};

std::shared_ptr<Model> make_bconv_model(const BConvCase& c, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> coin(0, 1);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<int8_t> w(static_cast<size_t>(c.oc * c.k * c.k * c.ic));
  for (auto& v : w) v = coin(rng) ? 1 : -1;
  auto wp = bftest::pack_pm1(w, c.oc * c.k * c.k, c.ic);
  std::vector<float> mul(c.oc), add(c.oc), thr(c.ic);
  for (size_t i = 0; i < mul.size(); ++i) mul[i] = (c.neg_mul && i % 2 ? -1.f : 1.f) * (0.01f + std::fabs(nd(rng)) * 0.1f);
  for (auto& v : add) v = nd(rng);
  for (auto& v : thr) v = nd(rng) * 0.3f;
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {c.oc, c.k, c.k, c.ic}, wp.data(), "OHWI_bit");
  mb.constant("mul", DType::kF32, {c.oc}, mul.data());
  mb.constant("add", DType::kF32, {c.oc}, add.data());
  if (c.with_threshold) mb.constant("thr", DType::kF32, {c.ic}, thr.data());
  mb.input("x", DType::kF32, {1, c.h, c.w, c.ic});
  Json attrs = Json::Object{{"strides", Json(Json::Array{Json(c.stride), Json(c.stride)})},
                            {"dilations", Json(Json::Array{Json(c.dil), Json(c.dil)})},
                            {"pads", Json(Json::Array{Json(c.pad), Json(c.pad), Json(c.pad), Json(c.pad)})},
                            {"pad_value", Json(c.pad_value)},
                            {"input_binarize", c.binarize},
                            {"output", c.bit_out ? "bit" : "f32"},
                            {"activation", c.activation}};
  mb.node("BConv2D", {"x", "w", "mul", c.no_add ? "" : "add", c.with_threshold ? "thr" : ""}, {"y"}, attrs);
  mb.output("y");
  return mb.build();
}

void fill_input(Tensor& t, uint32_t seed, bool with_zeros) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  float* p = t.data<float>();
  for (int64_t i = 0; i < t.size(); ++i) p[i] = (with_zeros && i % 7 == 0) ? 0.f : nd(rng);
}

// Runs the model with a given tier set (or, if `force_kernel` is set, that
// kernel forced for `force_op`) and returns raw output bytes.
std::vector<uint8_t> run_with(const std::shared_ptr<Model>& m, const std::set<std::string>& tiers, int threads,
                              uint32_t seed, bool zeros, std::string* kernel = nullptr, const std::string& force_op = "",
                              const std::string& force_kernel = "") {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers(tiers);
  if (!force_kernel.empty()) o.policy.force[force_op] = force_kernel;
  Session s(m, o);
  fill_input(s.input(0), seed, zeros);
  s.run();
  if (kernel) *kernel = s.kernel_choices().begin()->second;
  const Tensor& y = s.output(0);
  std::vector<uint8_t> out(y.bytes());
  std::memcpy(out.data(), y.raw(), out.size());
  return out;
}

// Every optimized BConv2D kernel, forced by name so kernels sharing a tier
// (bconv2d.neon / bconv2d.neon_opt) are all covered.
const std::vector<std::string>& optimized_bconv_kernels() {
  static std::vector<std::string> k = {"bconv2d.packed_naive", "bconv2d.binary_scalar", "bconv2d.neon", "bconv2d.neon_opt",
                                       "bconv2d.avx2"};
  return k;
}

bool kernel_available(const std::string& name) {
  for (const auto& k : OpRegistry::global().all_kernels())
    if (k.name == name) return cpu_info().has(k.required_features);
  return false;
}

// Compares every available optimized BConv2D kernel against the reference.
void check_bconv_cases(const std::vector<BConvCase>& cases, uint32_t seed0, std::initializer_list<int> thread_counts) {
  int idx = 0;
  for (const auto& c : cases) {
    auto m = make_bconv_model(c, seed0 + idx);
    auto ref = run_with(m, {"reference"}, 1, 5 + idx, true);
    for (const auto& name : optimized_bconv_kernels()) {
      if (!kernel_available(name)) continue;  // e.g. NEON kernels on x86
      for (int threads : thread_counts) {
        std::string kernel;
        std::vector<uint8_t> out;
        try {
          out = run_with(m, {}, threads, 5 + idx, true, &kernel, "BConv2D", name);
        } catch (const Error&) {
          continue;  // kernel rejects this shape (supports() == false)
        }
        if (out != ref) {
          throw bftest::Failure{"mismatch case " + std::to_string(idx) + " kernel " + kernel + " threads " + std::to_string(threads)};
        }
      }
    }
    ++idx;
  }
}

}  // namespace

BF_TEST(xnor_popcount_identity) {
  // a·b = K - 2·popcount(a XOR b) for K around word boundaries.
  std::mt19937 rng(1);
  for (int64_t k : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4609}) {
    std::vector<int8_t> a(k), b(k);
    int64_t dot = 0;
    for (int64_t i = 0; i < k; ++i) {
      a[i] = rng() & 1 ? 1 : -1;
      b[i] = rng() & 1 ? 1 : -1;
      dot += a[i] * b[i];
    }
    auto pa = bftest::pack_pm1(a, 1, k), pb = bftest::pack_pm1(b, 1, k);
    int64_t pc = 0;
    for (size_t w = 0; w < pa.size(); ++w) pc += __builtin_popcount(pa[w] ^ pb[w]);
    EXPECT_EQ(dot, k - 2 * pc);
  }
}

BF_TEST(bconv_reference_matches_hand_computation) {
  // 1x1 input with 3x3 kernel and pad 1: only the centre tap is in-bounds.
  // dot = x·w_centre + pad_value * Σ_{other taps} Σ_c w.
  for (int pv : {-1, 0, 1}) {
    BConvCase c{1, 1, 5, 2, 3, 1, 1, 1, pv, "ge", false, false};
    auto m = make_bconv_model(c, 7);
    SessionOptions o;
    o.policy = KernelPolicy::tiers({"reference"});
    Session s(m, o);
    float x[5] = {0.5f, -1.f, 0.f, 2.f, -0.1f};  // ge: +1,-1,+1,+1,-1
    std::memcpy(s.input(0).raw(), x, sizeof x);
    s.run();
    // Recompute independently from the packed constants.
    const auto& g = m->graph();
    const uint32_t* w = g.tensors[g.find("w")].constant.data<uint32_t>();
    const float* mul = g.tensors[g.find("mul")].constant.data<float>();
    const float* add = g.tensors[g.find("add")].constant.data<float>();
    int xs[5] = {1, -1, 1, 1, -1};
    for (int o = 0; o < 2; ++o) {
      int dot = 0;
      for (int t = 0; t < 9; ++t)
        for (int ch = 0; ch < 5; ++ch) {
          int wv = (w[o * 9 + t] >> ch) & 1 ? -1 : 1;
          dot += (t == 4 ? xs[ch] : pv) * wv;
        }
      EXPECT_NEAR(s.output(0).data<float>()[o], dot * mul[o] + add[o], 1e-5);
    }
  }
}

BF_TEST(bconv_optimized_kernels_match_reference_exactly) {
  std::vector<BConvCase> cases = {
      {8, 8, 64, 64, 3, 1, 1, 1, 0, "ge", false, false},     // typical, zero padding
      {9, 7, 32, 17, 3, 2, 1, 1, 0, "ge", false, true},      // odd sizes, stride 2, thresholds
      {6, 6, 33, 8, 3, 1, 1, 1, 1, "ge", false, false},      // ic tail, +1 padding
      {6, 5, 65, 12, 3, 1, 1, 1, -1, "gt", false, false},    // -1 padding, gt
      {7, 7, 128, 32, 1, 1, 0, 1, 0, "ge", false, false},    // 1x1
      {10, 10, 32, 20, 3, 1, 2, 2, 0, "ge", false, false},   // dilation 2
      {5, 5, 256, 64, 3, 2, 1, 1, 0, "ge", true, false},     // bit output
      {4, 4, 1, 5, 3, 1, 1, 1, 0, "ge", false, false},       // ic = 1
      {12, 12, 96, 70, 5, 1, 2, 1, 0, "ge", true, true},     // 5x5, bit out, oc tail
  };
  check_bconv_cases(cases, 100, {1, 2, 3, 4, 8});
}

BF_TEST(bconv_optimized_kernels_match_reference_larger) {
  // Larger shapes: several pixel tiles, K beyond 31 uint8 steps (36 and 72
  // steps), output-channel chunking at 8 threads on small spatial sizes,
  // activations with signed zeros, missing post_add.
  std::vector<BConvCase> cases = {
      {23, 19, 32, 32, 3, 1, 1, 1, 1, "ge", false, false},                         // many tiles, pixel tail
      {28, 28, 64, 64, 3, 1, 1, 1, 0, "ge", false, false, "relu"},                 // quicknet-like
      {7, 7, 512, 200, 3, 1, 1, 1, 0, "ge", true, false},                          // 36 steps, oc chunks, bit out
      {7, 7, 512, 96, 3, 1, 1, 1, -1, "gt", false, true, "relu6", true, true},     // signed zeros, no add
      {9, 9, 1024, 40, 3, 2, 1, 1, 0, "ge", false, false},                         // 72 steps, stride 2
      {14, 14, 256, 70, 3, 1, 1, 1, 1, "ge", true, true, "relu"},                  // bit out with oc tail
      {15, 13, 160, 36, 1, 1, 0, 1, 0, "ge", false, false, "none", true, true},    // 1x1, ic tail
      {16, 16, 64, 128, 3, 2, 1, 1, 0, "ge", false, false, "relu6", true, false},  // stride 2
      {11, 11, 48, 33, 7, 1, 3, 1, -1, "ge", true, false},                         // 7x7 kernel (49 taps)
      {12, 10, 128, 48, 3, 1, 1, 1, 0, "ge", false, false},                        // 9 steps (8..15 reduce path)
      {10, 10, 32, 8, 9, 1, 4, 1, 0, "ge", false, false},                          // 81 taps (> neon_opt limit)
  };
  check_bconv_cases(cases, 300, {1, 2, 4, 8});
}

BF_TEST(bconv_bit_input_chain_matches_reference) {
  // Two BConv2D nodes: the first emits bits, the second consumes them, so the
  // bit-input path of every kernel is exercised.
  for (int64_t ic : {64, 96, 256}) {
    std::mt19937 rng(static_cast<uint32_t>(ic));
    std::normal_distribution<float> nd(0.f, 1.f);
    int64_t oc = ic == 96 ? 72 : ic;
    std::vector<int8_t> w1(static_cast<size_t>(ic * 9 * ic)), w2(static_cast<size_t>(oc * 9 * ic));
    for (auto& v : w1) v = rng() & 1 ? 1 : -1;
    for (auto& v : w2) v = rng() & 1 ? 1 : -1;
    auto wp1 = bftest::pack_pm1(w1, ic * 9, ic), wp2 = bftest::pack_pm1(w2, oc * 9, ic);
    std::vector<float> mul(static_cast<size_t>(std::max(ic, oc))), add(mul.size());
    for (auto& v : mul) v = 0.01f + std::fabs(nd(rng)) * 0.1f;
    for (auto& v : add) v = nd(rng);
    bftest::ModelBuilder mb;
    mb.constant("w1", DType::kBit, {ic, 3, 3, ic}, wp1.data(), "OHWI_bit");
    mb.constant("w2", DType::kBit, {oc, 3, 3, ic}, wp2.data(), "OHWI_bit");
    mb.constant("mul1", DType::kF32, {ic}, mul.data());
    mb.constant("add1", DType::kF32, {ic}, add.data());
    mb.constant("mul2", DType::kF32, {oc}, mul.data());
    mb.constant("add2", DType::kF32, {oc}, add.data());
    mb.input("x", DType::kF32, {1, 13, 11, ic});
    Json a1 = Json::Object{{"pads", Json(Json::Array{Json(1), Json(1), Json(1), Json(1)})}, {"output", "bit"}};
    Json a2 = Json::Object{{"pads", Json(Json::Array{Json(1), Json(1), Json(1), Json(1)})}, {"pad_value", Json(-1)}};
    mb.node("BConv2D", {"x", "w1", "mul1", "add1", ""}, {"h"}, a1);
    mb.node("BConv2D", {"h", "w2", "mul2", "add2", ""}, {"y"}, a2);
    mb.output("y");
    auto m = mb.build();
    auto ref = run_with(m, {"reference"}, 1, 3, true);
    for (const auto& name : optimized_bconv_kernels()) {
      if (!kernel_available(name)) continue;
      for (int threads : {1, 4}) {
        auto out = run_with(m, {}, threads, 3, true, nullptr, "BConv2D", name);
        if (out != ref) throw bftest::Failure{"bit-input chain mismatch ic " + std::to_string(ic) + " kernel " + name};
      }
    }
  }
}

BF_TEST(bconv_sign_zero_only_on_reference) {
  BConvCase c{4, 4, 32, 8, 3, 1, 1, 1, 0, "sign_zero", false, false};
  auto m = make_bconv_model(c, 3);
  run_with(m, {"reference"}, 1, 1, true);
  EXPECT_THROWS(run_with(m, {"binary_scalar"}, 1, 1, true));
}

BF_TEST(bdense_scalar_matches_reference) {
  std::mt19937 rng(9);
  for (int64_t k : {7, 32, 100, 512}) {
    int64_t o = 13;
    std::vector<int8_t> w(o * k);
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    auto wp = bftest::pack_pm1(w, o, k);
    std::vector<float> mul(o, 0.5f), add(o, 0.25f);
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {o, k}, wp.data());
    mb.constant("mul", DType::kF32, {o}, mul.data());
    mb.constant("add", DType::kF32, {o}, add.data());
    mb.input("x", DType::kF32, {2, k});
    mb.node("BDense", {"x", "w", "mul", "add", ""}, {"y"});
    mb.output("y");
    auto m = mb.build();
    auto ref = run_with(m, {"reference"}, 1, 11, false);
    auto opt = run_with(m, {"binary_scalar"}, 2, 11, false);
    EXPECT_TRUE(ref == opt);
  }
}

// SIMD BDense kernels (each tested when the CPU has its features).
const char* const kSimdBDense[] = {"bdense.neon", "bdense.avx2"};

BF_TEST(bdense_simd_matches_reference) {
  struct Case {
    int64_t rows, k, o;
    bool thr, gt;
    std::string act;
  };
  std::vector<Case> cases = {{1, 7, 13, false, false, "none"},   {2, 32, 13, true, false, "relu"},
                             {5, 100, 13, false, true, "relu6"}, {1, 512, 1000, false, false, "none"},
                             {8, 4100, 37, true, true, "none"},  {3, 1024, 64, false, false, "relu"},
                             {2, 8192, 1100, true, false, "relu"},   // large enough to run multi-threaded
                             {196, 256, 256, true, false, "none"},   // BHViT-like token GEMM (row tiles)
                             {67, 1000, 70, false, true, "relu6"},   // row tail, oc tail
                             {4, 33, 9, true, false, "none"},
                             // pair mode (1-2 words mod 4, >= 8 rows): odd row counts, partial words, row tiles
                             {37, 64, 20, true, false, "relu"},     {200, 64, 512, false, false, "none"},
                             {9, 40, 6, false, true, "none"},       {16, 160, 5, true, false, "relu6"},
                             {300, 64, 70, false, false, "relu"},
                             // AArch64 4x8 tile (fp32, >= 8 rows, whole 128-bit steps, 16-channel units): row tails
                             {13, 384, 32, false, false, "relu"},   {29, 96, 48, true, false, "none"},
                             {66, 4096, 16, false, true, "relu6"},  {11, 1024, 64, true, false, "none"}};
  int idx = 0;
  for (const auto& c : cases) {
    std::mt19937 rng(20 + idx);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int8_t> w(c.o * c.k);
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    auto wp = bftest::pack_pm1(w, c.o, c.k);
    std::vector<float> mul(c.o), add(c.o), thr(c.k);
    for (size_t i = 0; i < mul.size(); ++i) mul[i] = (i % 3 == 0 ? -1.f : 1.f) * (0.01f + std::fabs(nd(rng)) * 0.1f);
    for (auto& v : add) v = nd(rng);
    for (auto& v : thr) v = nd(rng) * 0.3f;
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {c.o, c.k}, wp.data());
    mb.constant("mul", DType::kF32, {c.o}, mul.data());
    mb.constant("add", DType::kF32, {c.o}, add.data());
    if (c.thr) mb.constant("thr", DType::kF32, {c.k}, thr.data());
    mb.input("x", DType::kF32, {c.rows, c.k});
    Json attrs = Json::Object{{"input_binarize", c.gt ? "gt" : "ge"}, {"activation", c.act}};
    mb.node("BDense", {"x", "w", "mul", idx % 2 ? "" : "add", c.thr ? "thr" : ""}, {"y"}, attrs);
    mb.output("y");
    auto m = mb.build();
    auto ref = run_with(m, {"reference"}, 1, 11 + idx, true);
    for (const char* name : kSimdBDense) {
      if (!kernel_available(name)) continue;
      for (int threads : {1, 2, 4, 8}) {
        auto opt = run_with(m, {}, threads, 11 + idx, true, nullptr, "BDense", name);
        if (opt != ref)
          throw bftest::Failure{std::string(name) + " mismatch case " + std::to_string(idx) + " threads " + std::to_string(threads)};
      }
    }
    ++idx;
  }
}

BF_TEST(kernel_selector_respects_features) {
  // With NEON disabled via policy, the selector must fall back.
  BConvCase c{4, 4, 32, 8, 3, 1, 1, 1, 0, "ge", false, false};
  auto m = make_bconv_model(c, 3);
  SessionOptions o;
  Session s(m, o);
  std::string chosen = s.kernel_choices().begin()->second;
  if (cpu_info().has(kFeatNeon)) EXPECT_EQ(chosen, std::string("bconv2d.neon_opt"));
  else if (cpu_info().has(kFeatAvx2 | kFeatPopcnt)) EXPECT_EQ(chosen, std::string("bconv2d.avx2"));
  else EXPECT_EQ(chosen, std::string("bconv2d.binary_scalar"));
  SessionOptions forced;
  forced.policy.force["BConv2D"] = "bconv2d.packed_naive";
  Session s2(m, forced);
  EXPECT_EQ(s2.kernel_choices().begin()->second, std::string("bconv2d.packed_naive"));
}

// ------------------------------------------------- adversarial review cases
namespace {

struct GenConv {
  int64_t n, ih, iw, ic, oc, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr;
  int pad_value;
  bool bit_out, thr;
  std::string act;
  int64_t groups = 1;
};

std::shared_ptr<Model> make_gen_conv(const GenConv& c, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int64_t kic = c.ic / c.groups;
  std::vector<int8_t> w(static_cast<size_t>(c.oc * c.kh * c.kw * kic));
  for (auto& v : w) v = rng() & 1 ? 1 : -1;
  auto wp = bftest::pack_pm1(w, c.oc * c.kh * c.kw, kic);
  std::vector<float> mul(c.oc), add(c.oc), thr(c.ic);
  // Includes 0, -0, inf and negative scales so the epilogue sees signed zeros,
  // inf*0 = NaN and sign flips.
  for (size_t i = 0; i < mul.size(); ++i) {
    switch (i % 7) {
      case 0: mul[i] = 0.f; break;
      case 1: mul[i] = -0.f; break;
      case 2: mul[i] = INFINITY; break;
      case 3: mul[i] = -0.05f; break;
      default: mul[i] = 0.01f + std::fabs(nd(rng)) * 0.1f;
    }
  }
  for (size_t i = 0; i < add.size(); ++i) add[i] = i % 5 == 0 ? -0.f : nd(rng);
  for (auto& v : thr) v = nd(rng) * 0.3f;
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {c.oc, c.kh, c.kw, kic}, wp.data(), "OHWI_bit");
  mb.constant("mul", DType::kF32, {c.oc}, mul.data());
  mb.constant("add", DType::kF32, {c.oc}, add.data());
  if (c.thr) mb.constant("thr", DType::kF32, {c.ic}, thr.data());
  mb.input("x", DType::kF32, {c.n, c.ih, c.iw, c.ic});
  Json attrs = Json::Object{{"strides", Json(Json::Array{Json(c.sh), Json(c.sw)})},
                            {"dilations", Json(Json::Array{Json(c.dh), Json(c.dw)})},
                            {"pads", Json(Json::Array{Json(c.pt), Json(c.pb), Json(c.pl), Json(c.pr)})},
                            {"pad_value", Json(c.pad_value)},
                            {"output", c.bit_out ? "bit" : "f32"},
                            {"activation", c.act},
                            {"groups", Json(c.groups)}};
  mb.node("BConv2D", {"x", "w", "mul", "add", c.thr ? "thr" : ""}, {"y"}, attrs);
  mb.output("y");
  return mb.build();
}

// Input with NaN, +-inf, +-0 and values equal to typical thresholds.
void fill_nasty(Tensor& t, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  float* p = t.data<float>();
  for (int64_t i = 0; i < t.size(); ++i) {
    switch (rng() % 11) {
      case 0: p[i] = 0.f; break;
      case 1: p[i] = -0.f; break;
      case 2: p[i] = NAN; break;
      case 3: p[i] = INFINITY; break;
      case 4: p[i] = -INFINITY; break;
      default: p[i] = nd(rng);
    }
  }
}

// Runs a session twice (different inputs) and returns both outputs, so stale
// scratch from the first run cannot hide behind identical inputs.
std::vector<std::vector<uint8_t>> run_twice(const std::shared_ptr<Model>& m, const std::set<std::string>& tiers, int threads,
                                            const std::string& force_op, const std::string& force_kernel,
                                            std::string* kernel) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers(tiers);
  if (!force_kernel.empty()) o.policy.force[force_op] = force_kernel;
  Session s(m, o);
  std::vector<std::vector<uint8_t>> res;
  for (uint32_t seed : {7u, 8u}) {
    fill_nasty(s.input(0), seed);
    s.run();
    const Tensor& y = s.output(0);
    res.emplace_back(y.bytes());
    std::memcpy(res.back().data(), y.raw(), res.back().size());
  }
  if (kernel) *kernel = s.kernel_choices().begin()->second;
  return res;
}

}  // namespace

BF_TEST(bconv_adversarial_geometry_matches_reference) {
  // Batch > 1 (pixel tiles crossing images), rectangular kernels, asymmetric
  // pads/strides/dilations, kernels larger than the input, pair mode with odd
  // pixel counts and > 31 steps, exactly 64 taps, oc < 4.
  std::vector<GenConv> cases = {
      // n  ih  iw   ic   oc kh kw sh sw dh dw pt pb pl pr pv  bit    thr   act
      {3, 5, 5, 32, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, false, "none"},   // pair, k_words 9, 25 px per image
      {2, 7, 6, 64, 33, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, -1, true, false, "relu"},   // pair, 18 words, bit oc tail
      {1, 9, 7, 448, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, true, "relu6"},  // pair, 126 words, 63 steps
      {1, 6, 9, 416, 5, 3, 3, 1, 1, 1, 1, 0, 2, 1, 0, 0, true, false, "none"},    // pair, 117 words, 59 steps
      {2, 5, 8, 480, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, false, "none"},  // 135 words: just above pair limit
      {2, 9, 11, 40, 3, 2, 5, 2, 1, 1, 2, 0, 1, 3, 2, 0, true, true, "none"},     // rect kernel, dw = 2, oc = 3
      {1, 3, 2, 70, 1, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, -1, true, false, "none"},    // kernel > input, oc = 1
      {2, 10, 10, 31, 2, 8, 8, 1, 1, 1, 1, 4, 3, 4, 3, 0, false, false, "relu"},  // 64 taps (mask bit 63), oc = 2
      {1, 12, 12, 33, 64, 8, 8, 1, 1, 1, 1, 3, 4, 3, 4, 1, true, false, "none"},  // 64 taps, pad_value 1, 2 words/tap
      {4, 4, 4, 1, 65, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, true, false, "none"},     // ic = 1, 1x1, pair, oc 65
      {1, 1, 1, 96, 200, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, true, false, "relu6"},  // one output pixel, oc chunks
      {2, 13, 9, 128, 96, 3, 3, 2, 3, 2, 1, 2, 1, 0, 3, -1, false, true, "none"}, // mixed strides/dilations
  };
  int idx = 0;
  for (const auto& c : cases) {
    auto m = make_gen_conv(c, 900 + idx);
    auto ref = run_twice(m, {"reference"}, 1, "", "", nullptr);
    for (const auto& name : optimized_bconv_kernels()) {
      if (!kernel_available(name)) continue;
      for (int threads : {1, 3, 4, 8}) {
        std::string kernel;
        auto out = run_twice(m, {}, threads, "BConv2D", name, &kernel);
        EXPECT_EQ(kernel, name);
        for (size_t r = 0; r < 2; ++r)
          if (out[r] != ref[r])
            throw bftest::Failure{"adversarial mismatch case " + std::to_string(idx) + " kernel " + name + " threads " +
                                  std::to_string(threads) + " run " + std::to_string(r)};
      }
    }
    ++idx;
  }
}

BF_TEST(bconv_bit_output_into_reused_arena_slot) {
  // x -> A (f32) -> B (f32) -> C (bit, oc % 32 != 0) -> D. C's output can
  // reuse A's arena slot, which holds non-zero f32 data, so a kernel that
  // leaves the padding bits of its bit output untouched feeds garbage bits
  // into D.
  const int64_t ic = 64, oc_c = 40;
  std::mt19937 rng(77);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto rand_w = [&](int64_t o, int64_t i) {
    std::vector<int8_t> w(static_cast<size_t>(o * 9 * i));
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    return bftest::pack_pm1(w, o * 9, i);
  };
  auto wa = rand_w(ic, ic), wb = rand_w(ic, ic), wc = rand_w(oc_c, ic), wd = rand_w(16, oc_c);
  std::vector<float> mul(ic), add(ic);
  for (auto& v : mul) v = 0.01f + std::fabs(nd(rng)) * 0.1f;
  for (auto& v : add) v = nd(rng);
  bftest::ModelBuilder mb;
  mb.constant("wa", DType::kBit, {ic, 3, 3, ic}, wa.data(), "OHWI_bit");
  mb.constant("wb", DType::kBit, {ic, 3, 3, ic}, wb.data(), "OHWI_bit");
  mb.constant("wc", DType::kBit, {oc_c, 3, 3, ic}, wc.data(), "OHWI_bit");
  mb.constant("wd", DType::kBit, {16, 3, 3, oc_c}, wd.data(), "OHWI_bit");
  mb.constant("mul", DType::kF32, {ic}, mul.data());
  mb.constant("add", DType::kF32, {ic}, add.data());
  mb.constant("mulc", DType::kF32, {oc_c}, mul.data());
  mb.constant("addc", DType::kF32, {oc_c}, add.data());
  mb.constant("muld", DType::kF32, {16}, mul.data());
  mb.input("x", DType::kF32, {1, 8, 8, ic});
  Json same = Json::Object{{"pads", Json(Json::Array{Json(1), Json(1), Json(1), Json(1)})}};
  Json bit = Json::Object{{"pads", Json(Json::Array{Json(1), Json(1), Json(1), Json(1)})}, {"output", "bit"}};
  mb.node("BConv2D", {"x", "wa", "mul", "add", ""}, {"a"}, same);
  mb.node("BConv2D", {"a", "wb", "mul", "add", ""}, {"b"}, same);
  mb.node("BConv2D", {"b", "wc", "mulc", "addc", ""}, {"c"}, bit);
  mb.node("BConv2D", {"c", "wd", "muld", "", ""}, {"y"}, same);
  mb.output("y");
  auto m = mb.build();
  auto ref = run_with(m, {"reference"}, 1, 3, false);
  for (const auto& name : optimized_bconv_kernels()) {
    if (!kernel_available(name)) continue;
    for (int threads : {1, 4}) {
      auto out = run_with(m, {}, threads, 3, false, nullptr, "BConv2D", name);
      if (out != ref) throw bftest::Failure{"reused-slot bit output mismatch kernel " + name + " threads " + std::to_string(threads)};
    }
  }
}

BF_TEST(bdense_simd_adversarial) {
  // Bit input (k % 32 != 0), rank-3 input, rows % 4 tails, k < 32, nasty
  // input values, two runs per session.
  struct Case {
    std::vector<int64_t> xshape;
    int64_t o;
    bool bit_in, thr;
  };
  std::vector<Case> cases = {{{7, 33}, 5, false, true},   {{2, 3, 70}, 9, false, false}, {{9, 1}, 1, false, false},
                             {{6, 100}, 7, true, false},  {{4, 64}, 3, true, false},     {{1, 2049}, 1001, false, true},
                             {{13, 31}, 4, false, false}, {{5, 4096}, 260, true, false},
                             {{19, 64}, 7, true, false},  {{3, 7, 64}, 10, false, true}};
  int idx = 0;
  for (const auto& c : cases) {
    const int64_t k = c.xshape.back();
    std::mt19937 rng(40 + idx);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int8_t> w(static_cast<size_t>(c.o * k));
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    auto wp = bftest::pack_pm1(w, c.o, k);
    std::vector<float> mul(c.o), add(c.o), thr(k);
    for (size_t i = 0; i < mul.size(); ++i) mul[i] = i % 4 == 0 ? -0.f : (i % 4 == 1 ? INFINITY : nd(rng));
    for (auto& v : add) v = nd(rng);
    for (auto& v : thr) v = nd(rng) * 0.3f;
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {c.o, k}, wp.data());
    mb.constant("mul", DType::kF32, {c.o}, mul.data());
    mb.constant("add", DType::kF32, {c.o}, add.data());
    if (c.thr) mb.constant("thr", DType::kF32, {k}, thr.data());
    Json attrs = Json::Object{{"activation", idx % 3 == 0 ? "relu" : "none"}};
    if (c.bit_in) {
      // Bit input produced by a 1x1 BConv2D (whole-word outputs) on [rows,1,1,k].
      int64_t rows = 1;
      for (size_t i = 0; i + 1 < c.xshape.size(); ++i) rows *= c.xshape[i];
      std::vector<int8_t> w0(static_cast<size_t>(k * k));
      for (auto& v : w0) v = rng() & 1 ? 1 : -1;
      auto wp0 = bftest::pack_pm1(w0, k, k);
      std::vector<float> one(static_cast<size_t>(k), 1.f);
      mb.constant("w0", DType::kBit, {k, 1, 1, k}, wp0.data(), "OHWI_bit");
      mb.constant("one", DType::kF32, {k}, one.data());
      mb.input("x", DType::kF32, {rows, 1, 1, k});
      mb.node("BConv2D", {"x", "w0", "one", "", ""}, {"h"}, Json::Object{{"output", "bit"}});
      mb.node("BDense", {"h", "w", "mul", "add", ""}, {"y"}, attrs);
    } else {
      mb.input("x", DType::kF32, c.xshape);
      mb.node("BDense", {"x", "w", "mul", "add", c.thr ? "thr" : ""}, {"y"}, attrs);
    }
    mb.output("y");
    auto m = mb.build();
    auto ref = run_twice(m, {"reference"}, 1, "", "", nullptr);
    for (const char* name : kSimdBDense) {
      if (!kernel_available(name)) continue;
      for (int threads : {1, 4}) {
        std::string kernel;
        auto out = run_twice(m, {"reference"}, threads, "BDense", name, &kernel);
        for (size_t r = 0; r < 2; ++r)
          if (out[r] != ref[r])
            throw bftest::Failure{std::string(name) + " adversarial mismatch case " + std::to_string(idx) + " threads " +
                                  std::to_string(threads) + " run " + std::to_string(r)};
      }
    }
    ++idx;
  }
}

// ------------------------------------------------------ grouped BConv2D
namespace {

// Kernels that must accept groups > 1 (when the CPU has their features).
bool supports_groups(const std::string& name) { return name != "bconv2d.neon"; }

void check_grouped_cases(const std::vector<GenConv>& cases, uint32_t seed0, std::initializer_list<int> thread_counts) {
  int idx = 0;
  for (const auto& c : cases) {
    auto m = make_gen_conv(c, seed0 + idx);
    auto ref = run_twice(m, {"reference"}, 1, "", "", nullptr);
    for (const auto& name : optimized_bconv_kernels()) {
      if (!kernel_available(name)) continue;
      for (int threads : thread_counts) {
        std::string kernel;
        std::vector<std::vector<uint8_t>> out;
        try {
          out = run_twice(m, {}, threads, "BConv2D", name, &kernel);
        } catch (const Error&) {
          if (supports_groups(name) && c.kh * c.kw <= 64)
            throw bftest::Failure{"grouped case " + std::to_string(idx) + " rejected by " + name};
          continue;
        }
        EXPECT_EQ(kernel, name);
        for (size_t r = 0; r < 2; ++r)
          if (out[r] != ref[r])
            throw bftest::Failure{"grouped mismatch case " + std::to_string(idx) + " kernel " + name + " threads " +
                                  std::to_string(threads) + " run " + std::to_string(r)};
      }
    }
    ++idx;
  }
}

}  // namespace

BF_TEST(bconv_grouped_matches_reference) {
  // Every patch-row layout of BConvGroupLayout: kic % 32 == 0 (direct words),
  // kic % 32 == 16 (dense 16-bit units), other kic (per-group repacking),
  // oc per group not a multiple of 4 / 8 / 32 (bit-output words shared by
  // groups), depthwise (kic = 1), and groups == 1 with ic % 32 == 16.
  std::vector<GenConv> cases = {
      // n  ih  iw   ic   oc kh kw sh sw dh dw pt pb pl pr pv  bit    thr   act      groups
      {1, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, true, "none", 4},     // BHViT: kic 16, dil 1
      {1, 14, 14, 64, 64, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 0, false, true, "none", 4},     // BHViT: dil 3
      {1, 12, 13, 128, 128, 3, 3, 1, 1, 5, 5, 5, 5, 5, 5, 0, false, true, "none", 4},   // BHViT: kic 32, dil 5
      {2, 9, 7, 96, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, -1, true, false, "relu", 2},     // kic 48 (3 units), ocg 20, bit
      {1, 8, 8, 15, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, -1, true, true, "none", 3},       // kic 5 (repack), ocg 3, bit
      {1, 9, 10, 80, 64, 3, 3, 2, 1, 1, 1, 1, 1, 0, 2, 1, false, true, "relu6", 2},    // kic 40 (repack), stride 2x1
      {1, 7, 7, 99, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, true, false, "none", 3},       // kic 33: 2 words per group
      {2, 6, 6, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, false, "none", 8},      // depthwise-like, kic 1
      {1, 10, 9, 128, 96, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 0, true, true, "none", 2},     // kic 64, ocg 48, bit
      {1, 11, 11, 64, 36, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, "none", 4},    // 1x1, kic 16, ocg 9
      {1, 13, 12, 48, 24, 5, 5, 1, 1, 2, 2, 4, 3, 2, 5, -1, false, false, "none", 3},  // 5x5 dil 2, kic 16, asym pads
      {1, 8, 8, 48, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, true, true, "relu", 1},       // groups 1, ic 48 (16-bit units)
      {1, 6, 7, 16, 33, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true, false, "none", 1},      // groups 1, ic 16, oc tail
      {1, 5, 5, 64, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, true, false, "none", 4},     // few pixels: group chunks, bit
      {1, 4, 4, 64, 100, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, true, "none", 4},     // few pixels, f32, ocg 25
      {1, 3, 3, 96, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, -1, true, false, "none", 32},    // kic 3, 32 groups, bit
      {1, 10, 10, 32, 8, 8, 8, 1, 1, 1, 1, 4, 3, 4, 3, 0, false, false, "none", 2},    // 64 taps, kic 16
      {1, 2, 3, 40, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, true, false, "relu", 5},      // kic 8, ocg 16, 6 pixels
  };
  check_grouped_cases(cases, 1300, {1, 2, 3, 4});
}

BF_TEST(bconv_grouped_bit_input_chain_matches_reference) {
  // A non-grouped layer emits bits; a grouped layer consumes them, so the
  // bit-input paths (direct 32/16-bit units and per-group bit repacking) run.
  struct Chain {
    int64_t ic, groups, oc2;
  };
  for (Chain ch : {Chain{64, 4, 64}, Chain{128, 4, 128}, Chain{96, 2, 40}, Chain{15, 3, 9}, Chain{99, 3, 6}, Chain{80, 2, 64}}) {
    std::mt19937 rng(static_cast<uint32_t>(ch.ic * 7 + ch.groups));
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t ic = ch.ic, kic = ic / ch.groups, oc = ch.oc2;
    std::vector<int8_t> w1(static_cast<size_t>(ic * 9 * ic)), w2(static_cast<size_t>(oc * 9 * kic));
    for (auto& v : w1) v = rng() & 1 ? 1 : -1;
    for (auto& v : w2) v = rng() & 1 ? 1 : -1;
    auto wp1 = bftest::pack_pm1(w1, ic * 9, ic), wp2 = bftest::pack_pm1(w2, oc * 9, kic);
    std::vector<float> mul(static_cast<size_t>(std::max(ic, oc))), add(mul.size());
    for (auto& v : mul) v = 0.01f + std::fabs(nd(rng)) * 0.1f;
    for (auto& v : add) v = nd(rng);
    bftest::ModelBuilder mb;
    mb.constant("w1", DType::kBit, {ic, 3, 3, ic}, wp1.data(), "OHWI_bit");
    mb.constant("w2", DType::kBit, {oc, 3, 3, kic}, wp2.data(), "OHWI_bit");
    mb.constant("mul1", DType::kF32, {ic}, mul.data());
    mb.constant("add1", DType::kF32, {ic}, add.data());
    mb.constant("mul2", DType::kF32, {oc}, mul.data());
    mb.constant("add2", DType::kF32, {oc}, add.data());
    mb.input("x", DType::kF32, {2, 9, 11, ic});
    Json a1 = Json::Object{{"pads", Json(Json::Array{Json(1), Json(1), Json(1), Json(1)})}, {"output", "bit"}};
    Json a2 = Json::Object{{"pads", Json(Json::Array{Json(2), Json(2), Json(2), Json(2)})},
                           {"dilations", Json(Json::Array{Json(2), Json(2)})},
                           {"pad_value", Json(-1)},
                           {"groups", Json(ch.groups)}};
    mb.node("BConv2D", {"x", "w1", "mul1", "add1", ""}, {"h"}, a1);
    mb.node("BConv2D", {"h", "w2", "mul2", "add2", ""}, {"y"}, a2);
    mb.output("y");
    auto m = mb.build();
    auto ref = run_with(m, {"reference"}, 1, 3, true);
    for (const auto& name : optimized_bconv_kernels()) {
      if (!kernel_available(name) || !supports_groups(name)) continue;
      for (int threads : {1, 3}) {
        auto out = run_with(m, {}, threads, 3, true, nullptr, "BConv2D", name);
        if (out != ref)
          throw bftest::Failure{"grouped bit-input chain mismatch ic " + std::to_string(ic) + " groups " + std::to_string(ch.groups) +
                                " kernel " + name};
      }
    }
  }
}

BF_TEST(bconv_tile_4x8_matches_reference) {
  // AArch64 fp32 two-phase BConv2D on the 4x8 tile (rows of whole 128-bit steps, groups 1, oc % 8 == 0): step
  // counts 1..72 (uint16 accumulators, no uint8 runs), row tails of 1-3 pixels (single-row path), several pixel
  // tiles, 8-channel units and a unit tail of 8 channels, every pad value and activation.
  struct Shape2 {
    int64_t ic, k;
  };
  // 3x3: 36 words (9 steps), 27 (7), 63 (16), 72 (18), 144 (36), 288 (72); 1x1: 3 (1 step), 4 (1), 7 (2), 31 (8)
  const std::vector<Shape2> shapes = {{128, 3}, {96, 3}, {224, 3}, {256, 3}, {512, 3}, {1024, 3},
                                      {96, 1},  {128, 1}, {224, 1}, {992, 1}};
  const int64_t ocs[] = {8, 16, 24, 40, 64};
  const char* acts[] = {"none", "relu", "relu6"};
  std::vector<GenConv> cases;
  int i = 0;
  for (const Shape2& s : shapes) {
    const int64_t ih = i % 3 == 0 ? 11 : 5, iw = i % 3 == 0 ? 13 : (i % 3 == 1 ? 7 : 6), p = s.k / 2;  // 143, 35, 30 px
    cases.push_back({1, ih, iw, s.ic, ocs[i % 5], s.k, s.k, 1, 1, 1, 1, p, p, p, p, i % 3 - 1, false, i % 4 == 1, acts[i % 3]});
    ++i;
  }
  cases.push_back({1, 7, 7, 512, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, false, "relu"});    // 49 px: 1-row tail
  cases.push_back({2, 9, 9, 256, 48, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, -1, false, false, "none"});   // batch 2, stride 2
  check_grouped_cases(cases, 3100, {1, 3});
}

BF_TEST(bconv_step_classes_match_reference) {
  // bconv2d.neon_opt folds its uint8 popcount accumulators differently for
  // runs of <= 7, <= 15 and <= 31 steps, runs the micro-kernel two steps per
  // iteration (odd and even step counts, one-step runs) and takes a
  // straight-line path for full micro-tiles (4 pixels, 8 in pair mode) with a
  // per-tile border test: sweep step counts across these classes in normal
  // and pair mode, with pixel tails (mod 4 and mod 8), several pixel tiles,
  // channel tails, every activation and pad value.
  struct Shape2 {
    int64_t ic, k;  // k_words = k * k * bit_words(ic) (groups 1)
  };
  // 3x3: 9 words (pair, 5 steps), 18 (pair, 9), 27 (7), 36 (9), 45 (pair, 23), 54 (pair, 27), 63 (16), 72 (18),
  // 90 (pair, 45: two runs), 108 (27), 126 (pair, 63), 144 (36: two runs). 1x1: 1 / 2 (pair, 1 step), 3 (1),
  // 4 (1), 5 (pair, 3), 7 (2), 15 (4), 31 (8), 60 (15), 64 (16), 62 (pair, 31), 124 (31), 125 (pair, 63).
  const std::vector<Shape2> shapes = {{32, 3},   {64, 3},   {96, 3},   {128, 3},  {160, 3},  {192, 3},  {224, 3},
                                      {256, 3},  {320, 3},  {384, 3},  {448, 3},  {512, 3},  {32, 1},   {64, 1},
                                      {96, 1},   {128, 1},  {160, 1},  {224, 1},  {480, 1},  {992, 1},  {1920, 1},
                                      {2048, 1}, {1984, 1}, {3968, 1}, {4000, 1}};
  const char* acts[] = {"none", "relu", "relu6"};
  std::vector<GenConv> cases;
  int i = 0;
  for (const Shape2& s : shapes) {
    const bool big = i % 3 == 0;  // 143 pixels: several 64-pixel tiles for the narrower rows
    const int64_t ih = big ? 11 : 5, iw = big ? 13 : 7, p = s.k / 2;
    const int64_t oc = i % 2 ? 10 : 12;  // channel tail (2 of 4) or full blocks
    const int pv = i % 3 - 1;            // -1, 0, +1 padding
    cases.push_back({1, ih, iw, s.ic, oc, s.k, s.k, 1, 1, 1, 1, p, p, p, p, pv, false, i % 4 == 1, acts[i % 3]});
    ++i;
  }
  // Batch 2 (pixel tiles crossing images), stride 2.
  cases.push_back({2, 9, 9, 64, 16, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 0, false, false, "relu"});
  cases.push_back({2, 9, 9, 128, 16, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, -1, false, false, "none"});
  check_grouped_cases(cases, 2100, {1, 3});
}
