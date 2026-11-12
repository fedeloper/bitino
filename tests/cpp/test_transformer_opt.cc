// Differential tests of the kernels and plan-time rewrites added for the
// transformer / BHViT / binary-CNN models:
//   * plan-time rewrites (src/graph_rewrite.h): FusedElementwise and
//     bias->threshold folding are bit-exact with the unfused reference graph;
//     Reshape outputs alias their input buffer;
//   * float_opt kernels: LayerNorm, ReduceMean, FakeQuant, AddBroadcast,
//     Transpose, Concat, Gather, Mul, ChannelGate, hardswish/sign/sign_zero are
//     bit-exact; GELU / sigmoid / tanh / Softmax / MatMul within documented
//     bounds (GELU additionally sign-exact everywhere);
//   * BMatMul bit-plane kernels (portable, NEON, AVX2): bit-exact;
//   * INT8 MatMul (int8_avx2 / int8_neon): bit-exact with int8_reference,
//     within the analytic bound of the float reference, opt-in and gated.
// Every backend the CPU supports is forced and checked (NEON or portable 4-lane
// "<kernel>", AVX2 "<kernel>_avx2").
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

#include <thread>

#include "bitino/bitino.h"
#include "kernels/float_opt/float_opt.h"
#include "testing.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

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

bool registered(const std::string& name) {
  for (const auto& k : OpRegistry::global().all_kernels())
    if (k.name == name) return true;
  return false;
}

bool available(const std::string& name) {
  for (const auto& k : OpRegistry::global().all_kernels())
    if (k.name == name) return cpu_info().has(k.required_features);
  return false;
}

// float_opt backends of `base` this CPU runs: "<base>" (NEON / portable) and,
// with avx2+fma, "<base>_avx2".
std::vector<std::string> variants(const std::string& base) {
  std::vector<std::string> v = {base};
  if (registered(base + "_avx2") && available(base + "_avx2")) v.push_back(base + "_avx2");
  return v;
}

struct Out {
  std::vector<float> y;
  std::map<std::string, std::string> kernels;
  int fused = 0, folded = 0, aliased = 0;
};

// Runs a model with explicit input data (missing inputs get N(0,1) values).
Out run(const std::shared_ptr<Model>& m, const KernelPolicy& policy, const std::vector<std::vector<float>>& inputs, int threads = 1,
        bool fuse = true, bool reuse = true) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = policy;
  o.fuse = fuse;
  o.reuse_memory = reuse;
  Session s(m, o);
  for (size_t i = 0; i < s.num_inputs(); ++i) {
    Tensor& t = s.input(i);
    if (i < inputs.size()) {
      EXPECT_EQ(static_cast<int64_t>(inputs[i].size()), t.size());
      std::memcpy(t.raw(), inputs[i].data(), inputs[i].size() * sizeof(float));
    } else {
      auto v = randn(static_cast<size_t>(t.size()), 77 + static_cast<uint32_t>(i));
      std::memcpy(t.raw(), v.data(), v.size() * sizeof(float));
    }
  }
  s.run();
  Out r;
  const Tensor& y = s.output(0);
  r.y.assign(y.data<float>(), y.data<float>() + y.size());
  r.kernels = s.kernel_choices();
  r.fused = s.fused_nodes();
  r.folded = s.folded_thresholds();
  r.aliased = s.aliased_reshapes();
  return r;
}

KernelPolicy tiers(std::set<std::string> t) { return KernelPolicy::tiers(std::move(t)); }

KernelPolicy force(const std::string& op, const std::string& kernel, std::set<std::string> t = {"reference", "float_opt"}) {
  KernelPolicy p = KernelPolicy::tiers(std::move(t));
  p.force[op] = kernel;
  return p;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void expect_identical(const std::vector<float>& a, const std::vector<float>& b, const std::string& what) {
  if (same_bits(a, b)) return;
  size_t i = 0;
  while (i < a.size() && i < b.size() && std::memcmp(&a[i], &b[i], 4) == 0) ++i;
  std::ostringstream os;
  os << what << ": not bit-identical (sizes " << a.size() << "/" << b.size() << ")";
  if (i < a.size() && i < b.size()) os << ", first difference at " << i << ": " << a[i] << " vs " << b[i];
  throw bftest::Failure{os.str()};
}

const std::set<std::string> kRef = {"reference"};
const std::set<std::string> kOpt = {"reference", "binary_scalar", "neon", "binary_avx2", "float_opt"};

// Values that stress sign / rounding edge cases, mixed into random inputs.
std::vector<float> with_specials(std::vector<float> v, uint32_t seed) {
#if defined(__arm__) && defined(__ARM_NEON)
  // 32-bit ARM: AArch32 NEON arithmetic flushes denormals to zero (kernels/neon_a32.h), so the float_opt tier
  // (used by the fused epilogues as well) is bit-exact only for inputs whose results stay normal; the denormal
  // specials are replaced by tiny normal values. Compares and selections of denormals are covered by
  // a32_neon_denormal_semantics (test_binary_kernels.cc).
  const float sp[] = {0.f, -0.f, 1.f, -1.f, 0.5f, -0.5f, 1e-30f, -1e-30f, 1e-29f, -1e-29f, 3.f, -3.f};
#else
  const float sp[] = {0.f, -0.f, 1.f, -1.f, 0.5f, -0.5f, 1e-30f, -1e-30f, 1e-40f, -1e-40f, 3.f, -3.f};
#endif
  std::mt19937 rng(seed);
  for (size_t i = 0; i < v.size(); i += 7) v[i] = sp[rng() % (sizeof(sp) / sizeof(sp[0]))];
  return v;
}

}  // namespace

// ============================================================ plan rewrites
BF_TEST(fused_elementwise_bit_exact_with_unfused_graph) {
  // A BNext / ReActNet / BHViT-like elementwise DAG: two binary-conv-free
  // branches combined through ChannelAffine, Add (+relu), prelu, Mul,
  // ChannelGate (sigmoid and none), hardtanh, leaky_relu, relu6, sign, sign_zero.
  for (Shape s : {Shape{1, 3, 5, 7}, Shape{2, 9, 11, 40}, Shape{1, 28, 28, 64}}) {
    const int64_t c = s.back(), n = s[0];
    auto s1 = randn(c, 1), b1 = randn(c, 2), b2 = randn(c, 3), sl = randn(c, 4, 0.25f), s3 = randn(c, 5);
    bftest::ModelBuilder m2;
    m2.constant("s1", DType::kF32, {c}, s1.data());
    m2.constant("b1", DType::kF32, {c}, b1.data());
    m2.constant("b2", DType::kF32, {c}, b2.data());
    m2.constant("sl", DType::kF32, {c}, sl.data());
    m2.constant("s3", DType::kF32, {c}, s3.data());
    m2.input("x", DType::kF32, s);
    m2.input("r", DType::kF32, s);
    m2.input("g", DType::kF32, {n, c});
    m2.input("g4", DType::kF32, {n, 1, 1, c});
    m2.node("ChannelAffine", {"x", "s1", "b1"}, {"a"});
    m2.node("Add", {"a", "r"}, {"b"});
    m2.node("ChannelAffine", {"b", "", "b2"}, {"c"});
    m2.node("Activation", {"c", "sl"}, {"d"}, Json(Json::Object{{"type", "prelu"}}));
    m2.node("ChannelAffine", {"d", "s3", ""}, {"e"});
    m2.node("ChannelGate", {"e", "g"}, {"f"}, Json(Json::Object{{"gate", "sigmoid"}}));
    m2.node("Mul", {"f", "x"}, {"h"});
    m2.node("Activation", {"r"}, {"e1"}, Json(Json::Object{{"type", "hardtanh"}, {"min", Json(-0.5)}, {"max", Json(0.75)}}));
    m2.node("Activation", {"e1"}, {"e2"}, Json(Json::Object{{"type", "leaky_relu"}, {"alpha", Json(0.1)}}));
    m2.node("Add", {"h", "e2"}, {"i"}, Json(Json::Object{{"activation", "relu"}}));
    m2.node("ChannelGate", {"i", "g4"}, {"j"}, Json(Json::Object{{"gate", "none"}}));
    m2.node("Activation", {"j"}, {"k"}, Json(Json::Object{{"type", "relu6"}}));
    m2.node("Add", {"k", "b"}, {"l"}, Json(Json::Object{{"activation", "relu6"}}));  // b has two consumers: a leaf
    m2.node("Activation", {"l"}, {"m"}, Json(Json::Object{{"type", "sign"}}));
    m2.node("ChannelAffine", {"m", "s3", "b2"}, {"n"});
    m2.node("Activation", {"n"}, {"o"}, Json(Json::Object{{"type", "sign_zero"}}));
    m2.node("Add", {"o", "l"}, {"z"});
    m2.output("z");
    auto model = m2.build();
    std::vector<std::vector<float>> in = {with_specials(randn(num_elements(s), 10), 11), with_specials(randn(num_elements(s), 12), 13),
                                          randn(n * c, 14, 2.f), randn(n * c, 15)};
    const Out ref = run(model, tiers(kRef), in);
    EXPECT_EQ(ref.fused, 0);
    for (int threads : {1, 3}) {
      const Out opt = run(model, tiers(kOpt), in, threads);
      EXPECT_TRUE(opt.fused >= 1);
      expect_identical(opt.y, ref.y, "fused default " + shape_str(s));
      for (const std::string& k : {std::string("fused_elementwise.reference"), std::string("fused_elementwise.float_opt"),
                                   std::string("fused_elementwise.float_opt_avx2")}) {
        if (!registered(k) || !available(k)) continue;
        const Out f = run(model, force("FusedElementwise", k), in, threads);
        expect_identical(f.y, ref.y, k + " " + shape_str(s));
      }
      const Out nofuse = run(model, tiers(kOpt), in, threads, /*fuse=*/false);
      EXPECT_EQ(nofuse.fused, 0);
      expect_identical(nofuse.y, ref.y, "unfused float_opt " + shape_str(s));
    }
  }
}

BF_TEST(fused_gather_and_tile_loads_bit_exact) {
  // BHViT patterns: roll as Reshape -> Gather(row ids) -> Reshape feeding an Add (rows of
  // C and of C/4 channels, with out-of-range ids), and Concat([h] * 4, -1) feeding an Add.
  // Both become loads of one FusedElementwise node; results must equal the unfused graph.
  const int64_t h = 6, w = 7, c = 16;
  std::vector<int32_t> rows(h * w), sub(h * w * 4);
  for (int64_t i = 0; i < h * w; ++i) rows[i] = static_cast<int32_t>((i + 5) % (h * w));
  rows[3] = -1;                          // zero row
  rows[7] = static_cast<int32_t>(h * w);  // zero row
  for (int64_t i = 0; i < h * w * 4; ++i) sub[i] = static_cast<int32_t>((i * 7 + 3) % (h * w * 4));
  auto sc = randn(4 * c, 2);
  bftest::ModelBuilder mb;
  mb.constant("rows", DType::kI32, {h * w}, rows.data());
  mb.constant("sub", DType::kI32, {h * w * 4}, sub.data());
  mb.constant("sc", DType::kF32, {4 * c}, sc.data());
  mb.input("x", DType::kF32, {1, h, w, c});
  mb.input("big", DType::kF32, {1, h, w, 4 * c});
  mb.node("Activation", {"x"}, {"xa"}, Json(Json::Object{{"type", "relu"}}));  // multi-use base tensor
  mb.node("Reshape", {"xa"}, {"r1"}, Json(Json::Object{{"shape", ints({h * w, c})}}));
  mb.node("Gather", {"r1", "rows"}, {"g1"});
  mb.node("Reshape", {"g1"}, {"g1r"}, Json(Json::Object{{"shape", ints({1, h, w, c})}}));
  mb.node("Add", {"xa", "g1r"}, {"s1"});
  mb.node("Reshape", {"xa"}, {"r2"}, Json(Json::Object{{"shape", ints({h * w * 4, c / 4})}}));
  mb.node("Gather", {"r2", "sub"}, {"g2"});
  mb.node("Reshape", {"g2"}, {"g2r"}, Json(Json::Object{{"shape", ints({1, h, w, c})}}));
  mb.node("Add", {"s1", "g2r"}, {"s2"});
  mb.node("Concat", {"s2", "s2", "s2", "s2"}, {"rep"}, Json(Json::Object{{"axis", Json(-1)}}));
  mb.node("ChannelAffine", {"big", "sc", ""}, {"bb"});
  mb.node("Add", {"bb", "rep"}, {"y"});
  mb.output("y");
  auto m = mb.build();
  std::vector<std::vector<float>> in = {with_specials(randn(h * w * c, 3), 4), randn(h * w * 4 * c, 5)};
  const Out ref = run(m, tiers(kRef), in);
  for (int threads : {1, 3}) {
    const Out o = run(m, tiers(kOpt), in, threads);
    EXPECT_TRUE(o.fused >= 1);
    expect_identical(o.y, ref.y, "gather/tile loads");
    for (const std::string& k : {std::string("fused_elementwise.reference"), std::string("fused_elementwise.float_opt"),
                                 std::string("fused_elementwise.float_opt_avx2")}) {
      if (!registered(k) || !available(k)) continue;
      expect_identical(run(m, force("FusedElementwise", k), in, threads).y, ref.y, k + " gather/tile loads");
    }
  }
  // The rewritten plan has no Gather / Concat / second Reshape left.
  SessionOptions so;
  so.policy = tiers(kOpt);
  Session s(m, so);
  for (const auto& kv : s.kernel_choices()) {
    EXPECT_TRUE(kv.second.find("gather") == std::string::npos);
    EXPECT_TRUE(kv.second.find("concat") == std::string::npos);
  }
}

BF_TEST(fusion_respects_multi_consumer_and_graph_outputs) {
  // t feeds two nodes and is a graph output: never absorbed; the result must
  // still be exact and every output correct.
  const Shape s = {1, 4, 4, 16};
  bftest::ModelBuilder mb;
  auto b = randn(16, 3);
  mb.constant("b", DType::kF32, {16}, b.data());
  mb.input("x", DType::kF32, s);
  mb.node("ChannelAffine", {"x", "", "b"}, {"t"});
  mb.node("Activation", {"t"}, {"u"}, Json(Json::Object{{"type", "relu"}}));
  mb.node("Add", {"u", "t"}, {"v"});
  mb.node("Activation", {"v"}, {"w"}, Json(Json::Object{{"type", "relu6"}}));
  mb.output("w");
  mb.output("t");
  auto m = mb.build();
  std::vector<std::vector<float>> in = {randn(num_elements(s), 5)};
  SessionOptions o;
  o.policy = tiers(kOpt);
  Session s1(m, o);
  std::memcpy(s1.input(0).raw(), in[0].data(), in[0].size() * 4);
  s1.run();
  SessionOptions r;
  r.policy = tiers(kRef);
  Session s2(m, r);
  std::memcpy(s2.input(0).raw(), in[0].data(), in[0].size() * 4);
  s2.run();
  for (size_t i = 0; i < 2; ++i)
    EXPECT_TRUE(std::memcmp(s1.output(i).raw(), s2.output(i).raw(), s1.output(i).bytes()) == 0);
  EXPECT_TRUE(s1.fused_nodes() == 1);  // relu + add + relu6 (t stays materialized)
}

BF_TEST(threshold_folding_bit_exact) {
  // ChannelAffine(bias only) -> BConv2D / BDense (ge, gt, sign_zero): folded into
  // threshold = -bias. Inputs include exact ties x == -bias, +-0 and values one
  // ulp away, where a wrong fold would flip a sign.
  for (const char* mode : {"ge", "gt", "sign_zero"}) {
    const int64_t c = 40, o = 24;
    auto bias = randn(c, 1);
    bias[3] = 0.f;
    bias[4] = -0.f;
    std::vector<int8_t> w(o * 9 * c), wd(o * c);
    std::mt19937 rng(9);
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    for (auto& v : wd) v = rng() & 1 ? 1 : -1;
    auto wb = bftest::pack_pm1(w, o * 9, c);
    auto wdb = bftest::pack_pm1(wd, o, c);
    auto mul = randn(o, 3);
    const Shape s = {1, 6, 7, c};
    std::vector<float> x = randn(num_elements(s), 4);
    for (size_t i = 0; i < x.size(); ++i) {
      const float b = bias[i % c];
      switch (i % 5) {
        case 0: x[i] = -b; break;                                     // tie: x + b == 0
        case 1: x[i] = std::nextafter(-b, 1e30f); break;              // just above
        case 2: x[i] = std::nextafter(-b, -1e30f); break;             // just below
        case 3: x[i] = (i & 8) ? 0.f : -0.f; break;
        default: break;
      }
    }
    bftest::ModelBuilder mb;
    mb.constant("bias", DType::kF32, {c}, bias.data());
    mb.constant("w", DType::kBit, {o, 3, 3, c}, wb.data());
    mb.constant("wd", DType::kBit, {o, c}, wdb.data());
    mb.constant("mul", DType::kF32, {o}, mul.data());
    mb.input("x", DType::kF32, s);
    mb.node("ChannelAffine", {"x", "", "bias"}, {"t"});
    Json a = Json::Object{{"pads", ints({1, 1, 1, 1})}, {"input_binarize", mode}};
    mb.node("BConv2D", {"t", "w", "mul"}, {"y1"}, a);
    mb.node("BDense", {"t", "wd", "mul"}, {"y2"}, Json(Json::Object{{"input_binarize", mode}}));
    mb.node("Add", {"y1", "y2"}, {"y"});
    mb.output("y");
    auto m = mb.build();
    const Out ref = run(m, tiers(kRef), {x});
    for (int threads : {1, 2}) {
      const Out opt = run(m, tiers(kOpt), {x}, threads);
      EXPECT_EQ(opt.folded, 1);
      expect_identical(opt.y, ref.y, std::string("threshold folding ") + mode);
    }
  }
  // Not folded: a scale, or a non-binary consumer.
  const int64_t c = 8;
  auto sc = randn(c, 1), bias = randn(c, 2);
  std::vector<int8_t> wd(4 * c, 1);
  auto wdb = bftest::pack_pm1(wd, 4, c);
  bftest::ModelBuilder mb;
  mb.constant("s", DType::kF32, {c}, sc.data());
  mb.constant("b", DType::kF32, {c}, bias.data());
  mb.constant("wd", DType::kBit, {4, c}, wdb.data());
  mb.input("x", DType::kF32, {3, c});
  mb.node("ChannelAffine", {"x", "s", "b"}, {"t"});
  mb.node("BDense", {"t", "wd"}, {"y1"});
  mb.node("ChannelAffine", {"x", "", "b"}, {"u"});
  mb.node("BDense", {"u", "wd"}, {"y2"});
  mb.node("Activation", {"u"}, {"y3"}, Json(Json::Object{{"type", "relu"}}));
  mb.node("Add", {"y1", "y2"}, {"y"});
  mb.output("y");
  mb.output("y3");
  const Out r = run(mb.build(), tiers(kOpt), {});
  EXPECT_EQ(r.folded, 0);
}

BF_TEST(bconv_hosted_epilogue_bit_exact) {
  // BConv2D -> prelu -> ChannelAffine -> Add(residual) -> relu6 with the
  // residual computed before the conv: the tree runs inside the BConv2D kernel
  // (per pixel tile, or after the tasks when channels are split across
  // threads). Every BConv2D kernel with epilogue support is exercised through
  // its tier; results must equal the unfused reference graph bit for bit.
  struct HCase {
    Shape x;
    int64_t oc, k, groups;
  };
  uint32_t seed = 900;
  for (const HCase& hc : {HCase{{1, 9, 11, 64}, 64, 3, 1}, HCase{{1, 7, 7, 256}, 256, 3, 1}, HCase{{2, 6, 5, 64}, 64, 3, 4},
                          HCase{{1, 14, 14, 32}, 48, 1, 1}}) {
    const int64_t ic = hc.x[3], oc = hc.oc, kic = ic / hc.groups;
    std::vector<int8_t> w(oc * hc.k * hc.k * kic);
    std::mt19937 rng(seed++);
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    auto wb = bftest::pack_pm1(w, oc * hc.k * hc.k, kic);
    auto mul = randn(oc, seed++, 0.05f), sl = randn(oc, seed++, 0.3f), sc = randn(oc, seed++), bi = randn(oc, seed++);
    const int64_t p = hc.k / 2;
    const Shape ys = {hc.x[0], hc.x[1], hc.x[2], oc};
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {oc, hc.k, hc.k, kic}, wb.data());
    mb.constant("mul", DType::kF32, {oc}, mul.data());
    mb.constant("sl", DType::kF32, {oc}, sl.data());
    mb.constant("sc", DType::kF32, {oc}, sc.data());
    mb.constant("bi", DType::kF32, {oc}, bi.data());
    mb.input("x", DType::kF32, hc.x);
    mb.input("r", DType::kF32, ys);
    mb.node("Activation", {"r"}, {"res"}, Json(Json::Object{{"type", "relu"}}));  // produced before the conv
    mb.node("BConv2D", {"x", "w", "mul"}, {"c"},
            Json(Json::Object{{"pads", ints({p, p, p, p})}, {"groups", Json(hc.groups)}, {"input_binarize", "ge"}}));
    mb.node("Activation", {"c", "sl"}, {"d"}, Json(Json::Object{{"type", "prelu"}}));
    mb.node("ChannelAffine", {"d", "sc", "bi"}, {"e"});
    mb.node("Add", {"e", "res"}, {"f"});
    mb.node("Activation", {"f"}, {"y"}, Json(Json::Object{{"type", "relu6"}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {with_specials(randn(num_elements(hc.x), seed++), 3), randn(num_elements(ys), seed++)};
    const Out ref = run(m, tiers(kRef), in);
    std::vector<std::set<std::string>> tier_sets = {{"reference", "float_opt"}, {"reference", "binary_scalar", "float_opt"}, kOpt};
    for (const auto& ts : tier_sets)
      for (int threads : {1, 4}) {
        SessionOptions o;
        o.num_threads = threads;
        o.policy = tiers(ts);
        Session s(m, o);
        std::memcpy(s.input(0).raw(), in[0].data(), in[0].size() * 4);
        std::memcpy(s.input(1).raw(), in[1].data(), in[1].size() * 4);
        s.run();
        EXPECT_EQ(s.hosted_epilogues(), 1);
        std::string bconv_kernel;
        for (const auto& kv : s.kernel_choices())
          if (kv.first.find("+epilogue") != std::string::npos) bconv_kernel = kv.second;
        EXPECT_TRUE(!bconv_kernel.empty());
        const Tensor& y = s.output(0);
        expect_identical(std::vector<float>(y.data<float>(), y.data<float>() + y.size()), ref.y,
                         "hosted epilogue " + bconv_kernel + " t" + std::to_string(threads) + " " + shape_str(hc.x));
      }
  }
}

BF_TEST(fused_program_compiler_paths_bit_exact) {
  // Exercises every micro-op form of the compiled FusedElementwise evaluator
  // (kernels/float_opt/fused_elementwise.cc): binary ops whose first operand is
  // a load and second a computed subtree (reversed fused loads, with and
  // without activation), both operands computed (spill / pop), bias -> prelu ->
  // bias as one micro-op, tile and gather loads read in place (full and half
  // chunk widths) or through the row buffer (widths that do not fit the
  // chunk), and row tails (chunks, single vectors, single floats).
  struct FCase {
    int64_t c, w;  // channels, tile / gather piece width
  };
  uint32_t seed = 300;
  for (const FCase& fc : {FCase{64, 16}, FCase{96, 12}, FCase{37, 37}, FCase{128, 64}, FCase{40, 8}}) {
    const int64_t h = 5, wd = 7, c = fc.c, k = c / fc.w, rows = h * wd;
    const Shape s = {1, h, wd, c};
    auto s1 = randn(c, seed++), b1 = randn(c, seed++), bb1 = randn(c, seed++), sl = randn(c, seed++, 0.3f), bb2 = randn(c, seed++),
         s2 = randn(c, seed++);
    std::vector<int32_t> ids(static_cast<size_t>(rows * k));
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>((i * 5 + 3) % (ids.size() + 2)) - 1;  // incl. -1, V
    bftest::ModelBuilder mb;
    mb.constant("s1", DType::kF32, {c}, s1.data());
    mb.constant("b1", DType::kF32, {c}, b1.data());
    mb.constant("bb1", DType::kF32, {c}, bb1.data());
    mb.constant("sl", DType::kF32, {c}, sl.data());
    mb.constant("bb2", DType::kF32, {c}, bb2.data());
    mb.constant("s2", DType::kF32, {c}, s2.data());
    mb.constant("ids", DType::kI32, {rows * k}, ids.data());
    mb.input("x", DType::kF32, s);
    mb.input("r", DType::kF32, s);
    mb.input("t", DType::kF32, {1, h, wd, fc.w});
    mb.node("Activation", {"x"}, {"xs"}, Json(Json::Object{{"type", "sign"}}));        // multi-use leaves
    mb.node("Activation", {"r"}, {"rl"}, Json(Json::Object{{"type", "leaky_relu"}, {"alpha", Json(0.2)}}));
    mb.node("Activation", {"t"}, {"tt"}, Json(Json::Object{{"type", "hardtanh"}, {"min", Json(-1.0)}, {"max", Json(1.5)}}));
    mb.node("Activation", {"r"}, {"ra"}, Json(Json::Object{{"type", "relu"}}));
    mb.node("Add", {"xs", "ra"}, {"a"});                                             // load + computed
    mb.node("ChannelAffine", {"a", "s1", "b1"}, {"a2"});
    mb.node("Mul", {"rl", "a2"}, {"b"});                                             // load * computed
    mb.node("Add", {"rl", "b"}, {"c1"}, Json(Json::Object{{"activation", "relu6"}}));  // load + computed, act
    mb.node("ChannelAffine", {"c1", "", "bb1"}, {"d1"});                             // bias -> prelu -> bias
    mb.node("Activation", {"d1", "sl"}, {"d2"}, Json(Json::Object{{"type", "prelu"}}));
    mb.node("ChannelAffine", {"d2", "", "bb2"}, {"d"});
    mb.node("ChannelAffine", {"xs", "s2", ""}, {"e1"});
    mb.node("Activation", {"d"}, {"e2"}, Json(Json::Object{{"type", "sign_zero"}}));
    mb.node("Add", {"e1", "e2"}, {"e"});                                             // computed + computed
    std::vector<std::string> cat(static_cast<size_t>(k), "tt");
    mb.node("Concat", cat, {"rep"}, Json(Json::Object{{"axis", Json(-1)}}));
    mb.node("Add", {"rep", "e"}, {"f"}, Json(Json::Object{{"activation", "relu"}}));  // tile load first, act
    mb.node("Reshape", {"rl"}, {"tab"}, Json(Json::Object{{"shape", ints({rows * k, fc.w})}}));
    mb.node("Gather", {"tab", "ids"}, {"g"});
    mb.node("Reshape", {"g"}, {"gr"}, Json(Json::Object{{"shape", ints({1, h, wd, c})}}));
    mb.node("Mul", {"gr", "f"}, {"m"});                                              // gather load first
    mb.node("Concat", cat, {"rep2"}, Json(Json::Object{{"axis", Json(-1)}}));
    mb.node("Add", {"m", "rep2"}, {"y"});                                            // tile load second
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {with_specials(randn(num_elements(s), seed++), 5), with_specials(randn(num_elements(s), seed++), 6),
                                          with_specials(randn(static_cast<size_t>(rows * fc.w), seed++), 7)};
    const Out ref = run(m, tiers(kRef), in);
    for (int threads : {1, 3}) {
      const Out opt = run(m, tiers(kOpt), in, threads);
      EXPECT_TRUE(opt.fused >= 1);
      expect_identical(opt.y, ref.y, "compiler paths default c=" + std::to_string(c));
      for (const std::string& kn : {std::string("fused_elementwise.float_opt"), std::string("fused_elementwise.float_opt_avx2")}) {
        if (!registered(kn) || !available(kn)) continue;
        expect_identical(run(m, force("FusedElementwise", kn), in, threads).y, ref.y, kn + " c=" + std::to_string(c));
      }
    }
  }
}

BF_TEST(bdense_hosted_epilogue_bit_exact) {
  // BHViT MLP: BDense (K -> 8K) -> ChannelAffine -> + Concat x8 of the block
  // input -> bias -> prelu -> bias. The tree runs inside the BDense kernel (per
  // row tile, or after the product for few rows); every BDense kernel with
  // epilogue support is exercised through its tier, results must equal the
  // unfused reference graph bit for bit.
  struct DCase {
    Shape x;
    bool gt, thr;
  };
  uint32_t seed = 1300;
  for (const DCase& dc : {DCase{{1, 7, 9, 64}, false, false}, DCase{{1, 2, 3, 64}, true, false}, DCase{{2, 5, 5, 32}, false, true},
                          DCase{{1, 14, 14, 128}, false, false}, DCase{{1, 3, 5, 40}, true, true}}) {
    const int64_t k = dc.x.back(), o = 8 * k, rows = num_elements(dc.x) / k;
    std::vector<int8_t> w(static_cast<size_t>(o * k));
    std::mt19937 rng(seed++);
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    auto wb = bftest::pack_pm1(w, o, k);
    auto mul = randn(o, seed++, 0.05f), add = randn(o, seed++), sc = randn(o, seed++), bi = randn(o, seed++), b1 = randn(o, seed++),
         sl = randn(o, seed++, 0.3f), b2 = randn(o, seed++), thr = randn(k, seed++, 0.2f);
    Shape ys = dc.x;
    ys.back() = o;
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {o, k}, wb.data());
    mb.constant("mul", DType::kF32, {o}, mul.data());
    mb.constant("add", DType::kF32, {o}, add.data());
    mb.constant("thr", DType::kF32, {k}, thr.data());
    mb.constant("sc", DType::kF32, {o}, sc.data());
    mb.constant("bi", DType::kF32, {o}, bi.data());
    mb.constant("b1", DType::kF32, {o}, b1.data());
    mb.constant("sl", DType::kF32, {o}, sl.data());
    mb.constant("b2", DType::kF32, {o}, b2.data());
    mb.input("x", DType::kF32, dc.x);
    mb.node("Activation", {"x"}, {"xr"}, Json(Json::Object{{"type", "hardtanh"}, {"min", Json(-1.0)}, {"max", Json(1.0)}}));
    mb.node("BDense", {"xr", "w", "mul", "add", dc.thr ? "thr" : ""}, {"h"},
            Json(Json::Object{{"input_binarize", dc.gt ? "gt" : "ge"}}));
    mb.node("ChannelAffine", {"h", "sc", "bi"}, {"n"});
    mb.node("Concat", std::vector<std::string>(8, "xr"), {"rep"}, Json(Json::Object{{"axis", Json(-1)}}));
    mb.node("Add", {"n", "rep"}, {"a"});
    mb.node("ChannelAffine", {"a", "", "b1"}, {"m1"});
    mb.node("Activation", {"m1", "sl"}, {"p"}, Json(Json::Object{{"type", "prelu"}}));
    mb.node("ChannelAffine", {"p", "", "b2"}, {"y"});
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {with_specials(randn(num_elements(dc.x), seed++), 3)};
    const Out ref = run(m, tiers(kRef), in);
    std::vector<std::set<std::string>> tier_sets = {{"reference", "float_opt"}, {"reference", "binary_scalar", "float_opt"}, kOpt};
    for (const auto& ts : tier_sets)
      for (int threads : {1, 4}) {
        SessionOptions so;
        so.num_threads = threads;
        so.policy = tiers(ts);
        Session s(m, so);
        std::memcpy(s.input(0).raw(), in[0].data(), in[0].size() * 4);
        s.run();
        s.run();  // second run: per-run state (packing, epilogue operand pointers) is refreshed
        EXPECT_EQ(s.hosted_epilogues(), 1);
        std::string kernel;
        for (const auto& kv : s.kernel_choices())
          if (kv.first.find("+epilogue") != std::string::npos) kernel = kv.second;
        EXPECT_TRUE(kernel.find("bdense.") == 0);
        const Tensor& y = s.output(0);
        expect_identical(std::vector<float>(y.data<float>(), y.data<float>() + y.size()), ref.y,
                         "bdense hosted epilogue " + kernel + " t" + std::to_string(threads) + " " + shape_str(dc.x) +
                             " rows " + std::to_string(rows));
      }
    // Forcing a BDense kernel disables hosting in BDense (a forced kernel may lack support).
    SessionOptions so;
    so.policy = force("BDense", "bdense.reference", kOpt);
    Session s(m, so);
    EXPECT_EQ(s.hosted_epilogues(), 0);
  }
}

BF_TEST(unit16_epilogue_chains_bit_exact) {
  // Hosted epilogues that are chains of steps (EltwiseProgram::store_unit16: on AArch64 bconv2d.neon_opt / bdense.neon
  // run them per 16-channel unit in registers): every supported step sequence, after a binary layer with each fused
  // activation, plus chains it does not take (clamp, oc % 16 != 0): bit-identical to the unfused reference graph.
  // Letters: p prelu, r bias -> prelu -> bias, s ChannelAffine, a + residual, o + residual with relu, c clamp (relu6).
  uint32_t seed = 4100;
  for (const char* sig : {"p", "r", "s", "a", "ar", "sa", "so", "sar", "sarsa", "sac", "pc"}) {
    for (int dense = 0; dense < 2; ++dense) {
      for (const char* act : {"none", "relu", "relu6"}) {
        for (int64_t oc : {int64_t(32), int64_t(48), int64_t(40)}) {
          const int64_t ic = 64, k = dense ? 1 : 3;
          const Shape xs = dense ? Shape{37, ic} : Shape{1, 7, 9, ic};
          Shape ys = xs;
          ys.back() = oc;
          std::vector<int8_t> w(static_cast<size_t>(oc * k * k * ic));
          std::mt19937 rng(seed++);
          for (auto& v : w) v = rng() & 1 ? 1 : -1;
          auto wb = bftest::pack_pm1(w, oc * k * k, ic);
          bftest::ModelBuilder mb;
          mb.constant("w", DType::kBit, dense ? Shape{oc, ic} : Shape{oc, k, k, ic}, wb.data());
          auto mul = randn(oc, seed++, 0.05f), add = randn(oc, seed++);
          mb.constant("mul", DType::kF32, {oc}, mul.data());
          mb.constant("add", DType::kF32, {oc}, add.data());
          mb.input("x", DType::kF32, xs);
          mb.input("r", DType::kF32, ys);
          mb.node("Activation", {"r"}, {"res"}, Json(Json::Object{{"type", "leaky_relu"}, {"alpha", Json(0.1)}}));  // before the producer
          Json at = Json::Object{{"input_binarize", "ge"}, {"activation", act}};
          if (!dense) at["pads"] = ints({1, 1, 1, 1});
          mb.node(dense ? "BDense" : "BConv2D", {"x", "w", "mul", "add"}, {"prod"}, at);
          std::string cur = "prod";
          int n = 0;
          auto cst = [&](float scale) {
            const std::string nm = "c" + std::to_string(n++);
            auto v = randn(oc, seed++, scale);
            mb.constant(nm, DType::kF32, {oc}, v.data());
            return nm;
          };
          auto next = [&] { return "t" + std::to_string(n++); };
          for (const char* q = sig; *q; ++q) {
            const std::string out = next();
            switch (*q) {
              case 'p': mb.node("Activation", {cur, cst(0.3f)}, {out}, Json(Json::Object{{"type", "prelu"}})); break;
              case 'r': {
                const std::string b1 = cst(0.5f), sl = cst(0.3f), b2 = cst(0.5f), m1 = next(), m2 = next();
                mb.node("ChannelAffine", {cur, "", b1}, {m1});
                mb.node("Activation", {m1, sl}, {m2}, Json(Json::Object{{"type", "prelu"}}));
                mb.node("ChannelAffine", {m2, "", b2}, {out});
                break;
              }
              case 's': {
                const std::string sc = cst(1.f), bi = cst(1.f);
                mb.node("ChannelAffine", {cur, sc, bi}, {out});
                break;
              }
              case 'a': mb.node("Add", {cur, "res"}, {out}); break;
              case 'o': mb.node("Add", {cur, "res"}, {out}, Json(Json::Object{{"activation", "relu"}})); break;
              default: mb.node("Activation", {cur}, {out}, Json(Json::Object{{"type", "relu6"}})); break;
            }
            cur = out;
          }
          mb.node("ChannelAffine", {cur, "", ""}, {"y"});  // keeps the chain's last op inside the fused tree
          mb.output("y");
          auto m = mb.build();
          std::vector<std::vector<float>> in = {with_specials(randn(num_elements(xs), seed++), 3), with_specials(randn(num_elements(ys), seed++), 4)};
          const Out ref = run(m, tiers(kRef), in);
          for (int threads : {1, 4}) {
            const Out o = run(m, tiers(kOpt), in, threads);
            expect_identical(o.y, ref.y, std::string("unit16 chain ") + sig + (dense ? " bdense " : " bconv ") + act + " oc " +
                                             std::to_string(oc) + " t" + std::to_string(threads));
          }
        }
      }
    }
  }
}

BF_TEST(shared_binarizations_bit_exact) {
  // Bi-ViT qkv: several BDense binarize the same (LayerNorm) output identically -> one PackBits node per
  // (tensor, mode, threshold) group, the BDense read its bits. Groups: 3 x ge (shared), 2 x gt with the same
  // threshold (shared), 1 x gt with another threshold and 1 x ge reading the tensor twice (not shared); the tensor
  // also feeds an Add. Outputs must equal the unfused reference graph bit for bit.
  uint32_t seed = 5200;
  for (int64_t d : {int64_t(64), int64_t(40), int64_t(192)}) {
    const int64_t rows = 37, o = d;
    bftest::ModelBuilder mb;
    auto g = randn(d, seed++), b = randn(d, seed++), t1 = randn(d, seed++, 0.3f), t2 = randn(d, seed++, 0.3f);
    mb.constant("g", DType::kF32, {d}, g.data());
    mb.constant("b", DType::kF32, {d}, b.data());
    mb.constant("t1", DType::kF32, {d}, t1.data());
    mb.constant("t2", DType::kF32, {d}, t2.data());
    mb.input("x", DType::kF32, {rows, d});
    mb.node("ChannelAffine", {"x", "g", "b"}, {"n"});  // exact in every tier (a LayerNorm would select float_fast)
    std::vector<std::string> outs;
    struct U {
      const char* mode;
      const char* thr;
    };
    int k = 0;
    for (const U& u : {U{"ge", ""}, U{"ge", ""}, U{"gt", "t1"}, U{"ge", ""}, U{"gt", "t1"}, U{"gt", "t2"}}) {
      std::vector<int8_t> w(static_cast<size_t>(o * d));
      std::mt19937 rng(seed++);
      for (auto& v : w) v = rng() & 1 ? 1 : -1;
      auto wb = bftest::pack_pm1(w, o, d);
      auto mul = randn(o, seed++, 0.05f);
      const std::string id = std::to_string(k++);
      mb.constant("w" + id, DType::kBit, {o, d}, wb.data());
      mb.constant("m" + id, DType::kF32, {o}, mul.data());
      mb.node("BDense", {"n", "w" + id, "m" + id, "", u.thr}, {"d" + id}, Json(Json::Object{{"input_binarize", u.mode}}));
      outs.push_back("d" + id);
    }
    mb.node("Concat", outs, {"cat"}, Json(Json::Object{{"axis", Json(-1)}}));
    mb.node("Concat", {"n", "n", "n", "n", "n", "n"}, {"rep"}, Json(Json::Object{{"axis", Json(-1)}}));
    mb.node("Add", {"cat", "rep"}, {"y"});
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {with_specials(randn(rows * d, seed++, 2.f), 9)};
    const Out ref = run(m, tiers(kRef), in);
    for (const auto& ts : std::vector<std::set<std::string>>{{"reference", "float_opt"}, {"reference", "binary_scalar", "float_opt"}, kOpt})
      for (int threads : {1, 3}) {
        SessionOptions so;
        so.num_threads = threads;
        so.policy = tiers(ts);
        Session s(m, so);
        std::memcpy(s.input(0).raw(), in[0].data(), in[0].size() * 4);
        s.run();
        s.run();
        EXPECT_EQ(s.shared_binarizations(), 2);
        const Tensor& y = s.output(0);
        expect_identical(std::vector<float>(y.data<float>(), y.data<float>() + y.size()), ref.y,
                         "shared binarizations d=" + std::to_string(d) + " t" + std::to_string(threads));
      }
    SessionOptions so;
    so.policy = force("BDense", "bdense.reference", kOpt);  // a forced BDense kernel: no sharing
    EXPECT_EQ(Session(m, so).shared_binarizations(), 0);
  }
}

BF_TEST(reshape_output_aliases_input_buffer) {
  // x -> Add -> a -> Reshape r -> (other nodes allocate buffers) -> Add(r', ...)
  // The aliased buffer must stay live until the last use of any view.
  const Shape s = {2, 6, 8};
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, s);
  mb.node("Add", {"x", "x"}, {"a"});
  mb.node("Reshape", {"a"}, {"r"}, Json(Json::Object{{"shape", ints({12, 8})}}));
  mb.node("Activation", {"x"}, {"p"}, Json(Json::Object{{"type", "sign"}}));
  mb.node("Add", {"p", "x"}, {"q"});
  mb.node("Reshape", {"q"}, {"q2"}, Json(Json::Object{{"shape", ints({12, 8})}}));
  mb.node("Add", {"r", "q2"}, {"z1"});
  mb.node("Reshape", {"z1"}, {"z"}, Json(Json::Object{{"shape", ints({2, 6, 8})}}));  // graph output: copied
  mb.output("z");
  auto m = mb.build();
  std::vector<std::vector<float>> in = {randn(96, 1)};
  const Out ref = run(m, tiers(kRef), in, 1, true, /*reuse=*/false);
  for (const auto& pol : {tiers(kRef), tiers(kOpt)}) {
    const Out o = run(m, pol, in);
    EXPECT_EQ(o.aliased, 2);
    expect_identical(o.y, ref.y, "reshape aliasing");
  }
  SessionOptions so;
  Session sess(m, so);
  EXPECT_TRUE(sess.tensor("r").raw() == sess.tensor("a").raw());
  EXPECT_TRUE(sess.output(0).raw() != sess.tensor("z1").raw());
}

// ============================================================ activations
namespace {

std::vector<float> activation_ref_values(const std::string& type, const std::vector<float>& x) {
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, static_cast<int64_t>(x.size())});
  mb.node("Activation", {"x"}, {"y"}, Json(Json::Object{{"type", type}}));
  mb.output("y");
  return run(mb.build(), tiers(kRef), {x}).y;
}

std::vector<float> activation_opt_values(const std::string& type, const std::vector<float>& x, const std::string& kernel, int threads) {
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, static_cast<int64_t>(x.size())});
  mb.node("Activation", {"x"}, {"y"}, Json(Json::Object{{"type", type}}));
  mb.output("y");
  return run(mb.build(), force("Activation", kernel), {x}, threads).y;
}

float from_bits(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

int sgn(float v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

// Dense sample of floats in [-lim, lim]: every `stride`-th bit pattern, plus specials.
std::vector<float> float_grid(float lim, uint32_t stride) {
  std::vector<float> v;
  const uint32_t top = [&] {
    uint32_t b;
    std::memcpy(&b, &lim, 4);
    return b;
  }();
  for (uint32_t b = 0; b <= top; b += stride) {
    v.push_back(from_bits(b));
    v.push_back(-from_bits(b));
  }
  for (float s : {0.f, -0.f, 1e-45f, -1e-45f, 1e-38f, -1e-38f, 5.5f, -5.5f, 8.37438965f, -8.37438965f, -8.37438869f, -8.37439060f,
                  10.f, -10.f, 13.f, -13.f, 30.f, -30.f, 87.f, -87.f, 89.f, -89.f})
    v.push_back(s);
  return v;
}

}  // namespace

BF_TEST(activation_ext_gelu_sign_exact_and_bounded) {
  // GELU: sign(opt) == sign(reference) for every input (BiBERT binarizes the
  // GELU output with sign_zero). Value bound, with the double reference's own
  // cancellation error for x < -5.5 (1 + erf rounds to multiples of 2^-53):
  //   |opt - ref| <= 6e-7 |ref| + 2^-52 |x|   (about 10 ulp; measured worst 4.4e-7 relative).
  const std::vector<float> x = float_grid(40.f, 1021);
  const std::vector<float> ref = activation_ref_values("gelu", x);
  for (const std::string& k : variants("activation_ext.float_opt")) {
    const std::vector<float> y = activation_opt_values("gelu", x, k, 2);
    double worst = 0;
    size_t wi = 0;
    int bad_sign = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      if (sgn(y[i]) != sgn(ref[i])) ++bad_sign;
      const double err = std::fabs(double(y[i]) - double(ref[i]));
      const double bound = 6e-7 * std::fabs(double(ref[i])) + std::ldexp(std::fabs(double(x[i])), -52);
      if (err / bound > worst) {
        worst = err / bound;
        wi = i;
      }
    }
    std::printf("    %s gelu: %zu inputs, sign mismatches %d, worst err/bound %.3f (x=%.9g ref=%.9g opt=%.9g)\n", k.c_str(), x.size(),
                bad_sign, worst, x[wi], ref[wi], y[wi]);
    EXPECT_EQ(bad_sign, 0);
    EXPECT_TRUE(worst <= 1.0);
  }
}

// The kernel's GELU path (on AArch64: per-vector skip of the small-|x| branch and
// compaction of the live lanes x > cutoff in blocks of 256) must equal the plain
// per-vector vgelu bit for bit, for every lane pattern, block mix and tail length.
BF_TEST(activation_ext_gelu_fast_path_bit_identical) {
  std::mt19937 rng(4242);
  const float cut = -8.37438965f;
  const float sp[] = {0.f, -0.f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN(), 1e-45f, -1e-45f, 1e-38f, -1e-38f, 3e38f, -3e38f, cut,
                      std::nextafter(cut, 0.f), std::nextafter(cut, -1.f), 10.f, std::nextafter(10.f, 0.f), 0.70710678f,
                      std::nextafter(0.70710678f, 1.f), -0.70710678f, 20.f, -20.f};
  auto check = [](const std::vector<float>& x, const char* what) {
    std::vector<float> a(x.size()), b(x.size());
    float_opt_gelu_simd4(x.data(), a.data(), static_cast<int64_t>(x.size()), true);
    float_opt_gelu_simd4(x.data(), b.data(), static_cast<int64_t>(x.size()), false);
    expect_identical(a, b, std::string("gelu fast path ") + what + " n=" + std::to_string(x.size()));
  };
  int cases = 0;
  for (int64_t n : {0, 1, 3, 4, 5, 255, 256, 257, 511, 512, 513, 1000, 4099, 20000}) {
    for (int kind = 0; kind < 5; ++kind) {
      std::vector<float> x(n);
      std::normal_distribution<float> live(0.f, 3.f), dead(-9.f, 2.f), small(0.f, 0.4f);
      for (int64_t i = 0; i < n; ++i) {
        // 0: live, 1: dead-heavy, 2: small, 3: runs of mixed classes, 4: ~50/50 dead/live (compaction)
        int c = kind < 3 ? kind : (kind == 3 ? static_cast<int>((i / 97) % 3) : static_cast<int>(rng() % 2));
        x[i] = c == 0 ? live(rng) : (c == 1 ? dead(rng) : small(rng));
        if (rng() % 11 == 0) x[i] = sp[rng() % (sizeof(sp) / sizeof(sp[0]))];
      }
      check(x, "random");
      ++cases;
    }
  }
  // every 4-lane pattern of {dead, live, small, NaN}, at every alignment
  const float cls[4] = {-9.f, 1.5f, 0.2f, std::numeric_limits<float>::quiet_NaN()};
  std::vector<float> pat;
  for (int m = 0; m < 256; ++m)
    for (int j = 0; j < 4; ++j) pat.push_back(cls[(m >> (2 * j)) & 3]);
  for (int off = 0; off < 4; ++off) {
    std::vector<float> x(pat.begin() + off, pat.end());
    x.insert(x.end(), pat.begin(), pat.end());
    check(x, "patterns");
    ++cases;
  }
  std::printf("    gelu fast vs plain: %d cases identical\n", cases);
  // Opt-in: every float32 bit pattern (BITINO_EXHAUSTIVE_GELU=1; interleaved so blocks mix value classes).
  if (const char* e = std::getenv("BITINO_EXHAUSTIVE_GELU"); e && std::atoi(e) > 0) {
    constexpr int64_t kChunk = int64_t(1) << 20;
    const int nt = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<int64_t> bad(nt, 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t)
      ts.emplace_back([&, t] {
        std::vector<float> x(kChunk), a(kChunk), b(kChunk);
        for (int64_t c = t; c < (int64_t(1) << 32) / kChunk; c += nt) {
          for (int64_t j = 0; j < kChunk; ++j) {
            const uint32_t u = static_cast<uint32_t>(c * kChunk + ((j * 0x9E3779B1LL) & (kChunk - 1)));
            std::memcpy(&x[j], &u, 4);
          }
          float_opt_gelu_simd4(x.data(), a.data(), kChunk, true);
          float_opt_gelu_simd4(x.data(), b.data(), kChunk, false);
          if (std::memcmp(a.data(), b.data(), kChunk * 4) != 0) ++bad[t];
        }
      });
    for (auto& th : ts) th.join();
    int64_t nbad = 0;
    for (auto v : bad) nbad += v;
    std::printf("    gelu fast vs plain: all 2^32 inputs, %lld differing chunks\n", static_cast<long long>(nbad));
    EXPECT_EQ(nbad, int64_t(0));
  }
}

BF_TEST(activation_ext_sigmoid_tanh_bounded_hardswish_sign_exact) {
  const std::vector<float> x = float_grid(100.f, 4099);
  for (const std::string& k : variants("activation_ext.float_opt")) {
    for (const char* t : {"sigmoid", "tanh"}) {
      const std::vector<float> ref = activation_ref_values(t, x);
      const std::vector<float> y = activation_opt_values(t, x, k, 1);
      double worst = 0;
      for (size_t i = 0; i < x.size(); ++i) {
        // relative bound 4e-7 (a few ulp), absolute floor 1e-37 (sigmoid tails)
        const double err = std::fabs(double(y[i]) - double(ref[i]));
        worst = std::max(worst, err / (4e-7 * std::fabs(double(ref[i])) + 1e-37));
      }
      std::printf("    %s %s: worst err/bound %.3f\n", k.c_str(), t, worst);
      EXPECT_TRUE(worst <= 1.0);
    }
    for (const char* t : {"hardswish", "sign", "sign_zero"}) {
      std::vector<float> xs = x;
      xs.push_back(std::numeric_limits<float>::quiet_NaN());
      const std::vector<float> ref = activation_ref_values(t, xs);
      for (int threads : {1, 3}) expect_identical(activation_opt_values(t, xs, k, threads), ref, k + " " + t);
    }
  }
}

BF_TEST(clamp_activations_keep_signed_zero) {
  // relu6 / hardtanh(0, x) of -0 is -0 in the reference (std::min(std::max(-0, 0), 6));
  // a vector FMAX would return +0 (NEON). Also FakeQuant's clamp to qmin = 0.
  std::vector<float> x = {-0.f, 0.f, -1e-40f, 1e-40f, -0.f, 6.f, 7.f, -3.f, -0.f, -0.f, 0.5f, -0.f, 0.f, -0.f, 2.f, -0.f, -0.f};
  for (const char* t : {"relu6", "relu"}) {
    const std::vector<float> ref = activation_ref_values(t, x);
    for (const std::string& k : variants("activation.float_opt")) expect_identical(activation_opt_values(t, x, k, 1), ref, k + " " + t);
  }
}

// ============================================================ normalization / reductions
BF_TEST(layernorm_reduce_mean_fake_quant_bit_exact) {
  uint32_t seed = 100;
  for (int64_t d : {1, 3, 4, 5, 7, 8, 64, 192, 255, 768}) {
    for (int64_t rows : {1, 2, 3, 5, 9, 15, 64, 198}) {  // NEON groups of 8 / 4 / 2 / 1 rows
      for (int gb = 0; gb < 4; ++gb) {
        auto g = randn(d, seed++), b = randn(d, seed++);
        bftest::ModelBuilder mb;
        if (gb & 1) mb.constant("g", DType::kF32, {d}, g.data());
        if (gb & 2) mb.constant("b", DType::kF32, {d}, b.data());
        mb.input("x", DType::kF32, {rows, d});
        mb.node("LayerNorm", {"x", (gb & 1) ? "g" : "", (gb & 2) ? "b" : ""}, {"y"}, Json(Json::Object{{"eps", Json(1e-12)}}));
        mb.output("y");
        auto m = mb.build();
        std::vector<std::vector<float>> in = {with_specials(randn(rows * d, seed, 3.f), seed + 1)};
        in[0][0] += 1000.f;  // large mean: cancellation in the variance
        const Out ref = run(m, tiers(kRef), in);
        for (const std::string& k : variants("layernorm.float_opt"))
          for (int threads : {1, 3}) expect_identical(run(m, force("LayerNorm", k), in, threads).y, ref.y, k);
      }
    }
  }
  for (Shape s : {Shape{1, 56, 56, 64, 8}, Shape{3, 5, 7}, Shape{2, 9}, Shape{4, 3, 13, 6}}) {
    for (int64_t axis = -1; axis >= -static_cast<int64_t>(s.size()); --axis) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kF32, s);
      mb.node("ReduceMean", {"x"}, {"y"}, Json(Json::Object{{"axis", Json(axis)}, {"keepdims", Json(0)}}));
      mb.output("y");
      auto m = mb.build();
      std::vector<std::vector<float>> in = {with_specials(randn(num_elements(s), seed, 2.f), seed + 1)};
      const Out ref = run(m, tiers(kRef), in);
      for (const std::string& k : variants("reduce_mean.float_opt"))
        for (int threads : {1, 4}) expect_identical(run(m, force("ReduceMean", k), in, threads).y, ref.y, k + " " + shape_str(s));
    }
  }
  // FakeQuant (BHViT attention probabilities: axis 1, pre_scale 3, levels 0..3)
  // with inputs placed exactly on rounding ties after t = x*pre/s + z.
  for (Shape s : {Shape{4, 4, 50, 50}, Shape{3, 7}, Shape{2, 3, 5, 9}}) {
    for (int64_t axis : {int64_t(1), int64_t(-1)}) {
      const int64_t c = s[axis < 0 ? s.size() - 1 : axis];
      std::vector<float> sc(c), zp(c);
      for (int64_t i = 0; i < c; ++i) {
        sc[i] = 0.25f + 0.125f * static_cast<float>(i % 5);
        zp[i] = static_cast<float>(i % 3) * 0.5f;
      }
      bftest::ModelBuilder mb;
      mb.constant("s", DType::kF32, {c}, sc.data());
      mb.constant("z", DType::kF32, {c}, zp.data());
      mb.input("x", DType::kF32, s);
      mb.node("FakeQuant", {"x", "s", "z"}, {"y"},
              Json(Json::Object{{"axis", Json(axis)}, {"pre_scale", Json(3.0)}, {"qmin", Json(0)}, {"qmax", Json(3)}}));
      mb.output("y");
      auto m = mb.build();
      std::vector<float> x = randn(num_elements(s), seed++, 0.6f);
      for (size_t i = 0; i < x.size(); i += 3) x[i] = static_cast<float>(i % 9) * 0.125f;  // dyadic: exact ties
      const Out ref = run(m, tiers(kRef), {x});
      for (const std::string& k : variants("fake_quant.float_opt"))
        for (int threads : {1, 3}) expect_identical(run(m, force("FakeQuant", k), {x}, threads).y, ref.y, k + " " + shape_str(s));
    }
  }
}

BF_TEST(softmax_float_opt_within_bound) {
  // |opt - ref| <= 2e-6 |ref| + 1e-12: float exp (~1 ulp), float sums of <= 1000
  // terms in W lanes, one division (the double reference rounds once).
  uint32_t seed = 300;
  double worst_seen = 0;
  for (int64_t c : {1, 3, 7, 8, 9, 16, 49, 64, 197, 198, 1000}) {
    for (float mag : {0.1f, 3.f, 30.f}) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kF32, {5, c});
      mb.node("Softmax", {"x"}, {"y"});
      mb.output("y");
      auto m = mb.build();
      std::vector<std::vector<float>> in = {randn(5 * c, seed++, mag)};
      const Out ref = run(m, tiers(kRef), in);
      for (const std::string& k : variants("softmax.float_opt")) {
        const Out o = run(m, force("Softmax", k), in, 2);
        for (size_t i = 0; i < o.y.size(); ++i) {
          const double r = std::fabs(double(o.y[i]) - double(ref.y[i])) / (2e-6 * std::fabs(double(ref.y[i])) + 1e-12);
          worst_seen = std::max(worst_seen, r);
          EXPECT_TRUE(r <= 1.0);
        }
      }
    }
  }
  std::printf("    softmax worst err/bound %.3f\n", worst_seen);
}

// ============================================================ float32 fast kernels (stage all / optimized)
namespace {

// Error of y against the reference in units of 2^-23 * scale (scale > 0), and in ulps of the reference value.
struct ErrStats {
  double worst = 0, max_abs = 0, max_ulp = 0;
  void add(float y, float ref, double scale) {
    const double e = std::fabs(double(y) - double(ref));
    worst = std::max(worst, e / (std::ldexp(scale, -23) + 1e-30));
    max_abs = std::max(max_abs, e);
    const double ulp = std::ldexp(1.0, std::max(std::ilogb(ref == 0.f ? 1e-38f : ref), -126) - 23);
    max_ulp = std::max(max_ulp, e / ulp);
  }
};

}  // namespace

BF_TEST(layernorm_float_fast_within_bound) {
  // layernorm.float_fast vs the double reference: with t = (x - mean) / sd the exact normalized value,
  //   |y - ref| <= 4 * 2^-23 * ((|t| + 1) |g| + |b|)   (measured worst 1.56)
  // i.e. a few float roundings of t plus an absolute error of ~2^-23 sd in the mean (float sums of the
  // deviations); also for rows with a large mean (|mean| up to 3000 sd), constant rows and eps = 1e-12.
  uint32_t seed = 900;
  ErrStats st, big;  // big: no gamma / beta, |t| >= 0.25 (ulp of the normalized value)
  for (int64_t d : {1, 3, 4, 5, 7, 8, 15, 16, 17, 64, 192, 255, 384, 768}) {
    for (int64_t rows : {1, 3, 9, 64}) {
      for (int gb = 0; gb < 4; ++gb) {
        for (double eps : {1e-12, 1e-5}) {
          auto g = randn(d, seed++), b = randn(d, seed++);
          bftest::ModelBuilder mb;
          if (gb & 1) mb.constant("g", DType::kF32, {d}, g.data());
          if (gb & 2) mb.constant("b", DType::kF32, {d}, b.data());
          mb.input("x", DType::kF32, {rows, d});
          mb.node("LayerNorm", {"x", (gb & 1) ? "g" : "", (gb & 2) ? "b" : ""}, {"y"}, Json(Json::Object{{"eps", Json(eps)}}));
          mb.output("y");
          auto m = mb.build();
          std::vector<float> x = with_specials(randn(rows * d, seed, 3.f), seed + 1);
          ++seed;
          for (int64_t r = 0; r < rows; ++r) {  // per-row offsets up to ~1e4 (cancellation), one constant row
            const float off = (r % 3 == 0) ? 0.f : (r % 3 == 1 ? 37.5f : -9000.f);
            for (int64_t i = 0; i < d; ++i) x[r * d + i] = (r == 2 ? 0.1f : x[r * d + i]) + off;
          }
          const Out ref = run(m, tiers(kRef), {x});
          for (int threads : {1, 3}) {
            const Out o = run(m, force("LayerNorm", "layernorm.float_fast"), {x}, threads);
            for (int64_t r = 0; r < rows; ++r) {
              double mean = 0, var = 0, mabs = 0;
              for (int64_t i = 0; i < d; ++i) mabs += std::fabs(x[r * d + i]) / double(d);
              for (int64_t i = 0; i < d; ++i) mean += x[r * d + i];
              mean /= double(d);
              for (int64_t i = 0; i < d; ++i) var += (x[r * d + i] - mean) * (x[r * d + i] - mean);
              const double sd = std::sqrt(var / double(d) + eps);
              for (int64_t i = 0; i < d; ++i) {
                const double t = (x[r * d + i] - mean) / sd;
                const double scale = (std::fabs(t) + 1.0) * std::fabs((gb & 1) ? g[i] : 1.f) + ((gb & 2) ? std::fabs(b[i]) : 0.0) + 1e-30;
                const size_t k = static_cast<size_t>(r * d + i);
                const double before = st.worst;
                st.add(o.y[k], ref.y[k], scale);
                if (gb == 0 && std::fabs(t) >= 0.25) big.add(o.y[k], ref.y[k], scale);
                if (st.worst > before && st.worst > 4)
                  std::printf("      d=%lld rows=%lld r=%lld i=%lld gb=%d eps=%g x=%.9g t=%.9g y=%.9g ref=%.9g\n", (long long)d, (long long)rows,
                              (long long)r, (long long)i, gb, eps, x[k], t, o.y[k], ref.y[k]);
              }
            }
          }
        }
      }
    }
  }
  std::printf("    layernorm.float_fast: worst err/(2^-23 ((|t| + 1) |g| + |b|)) %.3f, max abs %.3g; y = t, |t| >= 0.25: max abs %.3g, max %.1f ulp\n",
              st.worst, st.max_abs, big.max_abs, big.max_ulp);
  EXPECT_TRUE(st.worst <= 4.0);
}

BF_TEST(reduce_mean_float_fast_within_bound) {
  // reduce_mean.float_fast (last axis) vs the double reference, with S = mean |x| of the reduced run:
  //   |y - ref| <= (len / 16 + 6) * 2^-23 * S   (float sums in 16 lanes / pairwise trees)
  // Other axes are not supported (reduce_mean.float_opt is selected there).
  uint32_t seed = 950;
  double worst = 0, max_abs = 0;
  for (Shape s : {Shape{1, 56, 56, 64, 8}, Shape{1, 14, 14, 256, 4}, Shape{3, 5, 7}, Shape{2, 9}, Shape{4, 3, 13, 6}, Shape{2, 197},
                  Shape{3, 1000}, Shape{5, 1}}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kF32, s);
    mb.node("ReduceMean", {"x"}, {"y"}, Json(Json::Object{{"axis", Json(-1)}, {"keepdims", Json(0)}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<float> x = with_specials(randn(num_elements(s), seed, 2.f), seed + 1);
    seed += 2;
    for (size_t i = 0; i < x.size(); i += 5) x[i] += 50.f;
    const int64_t len = s.back();
    const Out ref = run(m, tiers(kRef), {x});
    for (int threads : {1, 4}) {
      const Out o = run(m, tiers(kOpt), {x}, threads);
      const Out f = run(m, force("ReduceMean", "reduce_mean.float_fast"), {x}, threads);
      if (!available("reduce_mean.float_opt_avx2")) EXPECT_EQ(o.kernels.begin()->second, std::string("reduce_mean.float_fast"));
      for (size_t k = 0; k < f.y.size(); ++k) {
        double sa = 0;
        for (int64_t i = 0; i < len; ++i) sa += std::fabs(x[k * len + i]);
        const double e = std::fabs(double(f.y[k]) - double(ref.y[k]));
        max_abs = std::max(max_abs, e);
        worst = std::max(worst, e / (std::ldexp((len / 16.0 + 6.0) * sa / double(len), -23) + 1e-30));
      }
    }
  }
  {  // not the last axis: the exact kernel is selected
    bftest::ModelBuilder mb;
    mb.input("x", DType::kF32, {1, 4, 4, 256});
    mb.node("ReduceMean", {"x"}, {"y"}, Json(Json::Object{{"axis", Json(1)}, {"keepdims", Json(0)}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {randn(4 * 4 * 256, 7)};
    const Out o = run(m, tiers(kOpt), in);
    EXPECT_TRUE(o.kernels.begin()->second.rfind("reduce_mean.float_opt", 0) == 0);
    expect_identical(o.y, run(m, tiers(kRef), in).y, "reduce_mean axis 1");
  }
  std::printf("    reduce_mean.float_fast: worst err/bound %.3f, max abs %.3g\n", worst, max_abs);
  EXPECT_TRUE(worst <= 1.0);
}

BF_TEST(softmax_float_fast_within_bound) {
  // softmax.float_fast: exp(x - max) (Cephes polynomial, ~1 ulp), float sums in 16 lanes, y = e * (1 / sum):
  //   |y - ref| <= 2e-6 |ref| + 1e-12   (as softmax.float_opt; measured worst ~0.3 of it)
  uint32_t seed = 980;
  double worst = 0;
  ErrStats st;
  for (int64_t c : {1, 3, 7, 8, 9, 15, 16, 17, 49, 64, 197, 198, 1000}) {
    for (float mag : {0.1f, 3.f, 30.f}) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kF32, {5, c});
      mb.node("Softmax", {"x"}, {"y"});
      mb.output("y");
      auto m = mb.build();
      std::vector<std::vector<float>> in = {randn(5 * c, seed++, mag)};
      in[0][0] = -1e30f;  // masked score (exp underflows to 2^-126 instead of 0)
      const Out ref = run(m, tiers(kRef), in);
      for (int threads : {1, 2}) {
        const Out o = run(m, force("Softmax", "softmax.float_fast"), in, threads);
        for (size_t i = 0; i < o.y.size(); ++i) {
          worst = std::max(worst, std::fabs(double(o.y[i]) - double(ref.y[i])) / (2e-6 * std::fabs(double(ref.y[i])) + 1e-12));
          if (ref.y[i] > 1e-30f) st.add(o.y[i], ref.y[i], std::fabs(ref.y[i]));
        }
      }
    }
  }
  std::printf("    softmax.float_fast: worst err/bound %.3f, max %.1f ulp of ref (ref > 1e-30)\n", worst, st.max_ulp);
  EXPECT_TRUE(worst <= 1.0);
}

BF_TEST(gelu_float_fast_sign_exact_and_bounded) {
  // activation_gelu.float_fast (erfc form for every lane, no small-|x| branch): sign-exact like the exact kernel
  // (-0 for x <= kGeluCutoff, sign(x) elsewhere) and |y - ref| <= 1e-6 |ref| + 2^-52 |x|.
  if (!available("activation_gelu.float_fast")) {   // not registered on 32-bit ARM (denormal flush, see the kernel)
    std::printf("    activation_gelu.float_fast: not available on this target\n");
    return;
  }
  const std::vector<float> x = float_grid(40.f, 1021);
  const std::vector<float> ref = activation_ref_values("gelu", x);
  for (int threads : {1, 2}) {
    const std::vector<float> y = activation_opt_values("gelu", x, "activation_gelu.float_fast", threads);
    double worst = 0, worst_rel = 0;
    int bad_sign = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      if (sgn(y[i]) != sgn(ref[i]) || std::signbit(y[i]) != std::signbit(ref[i])) ++bad_sign;
      const double err = std::fabs(double(y[i]) - double(ref[i]));
      worst = std::max(worst, err / (1e-6 * std::fabs(double(ref[i])) + std::ldexp(std::fabs(double(x[i])), -52)));
      if (x[i] > -5.5f && std::fabs(ref[i]) > 1e-30) worst_rel = std::max(worst_rel, err / std::fabs(double(ref[i])));
    }
    std::printf("    activation_gelu.float_fast: %zu inputs, sign mismatches %d, worst err/bound %.3f, max rel (x > -5.5) %.3g\n", x.size(), bad_sign, worst,
                worst_rel);
    EXPECT_EQ(bad_sign, 0);
    EXPECT_TRUE(worst <= 1.0);
  }
  // non-GELU activations are not taken
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, 64});
  mb.node("Activation", {"x"}, {"y"}, Json(Json::Object{{"type", "tanh"}}));
  mb.output("y");
  EXPECT_TRUE(run(mb.build(), tiers(kOpt), {randn(64, 3)}).kernels.begin()->second.rfind("activation_ext.float_opt", 0) == 0);
}

BF_TEST(float_fast_kernels_only_in_float_opt_stages) {
  // Stage `neon` (no float_opt tier) keeps the bit-exact reference LayerNorm / Softmax / ReduceMean; with float_opt
  // the fast kernels win over the exact float_opt ones (unless an x86-64 AVX2 kernel is available).
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {4, 8, 8});
  mb.node("LayerNorm", {"x", "", ""}, {"a"}, Json(Json::Object{{"eps", Json(1e-5)}}));
  mb.node("Softmax", {"a"}, {"b"});
  mb.node("Activation", {"b"}, {"c"}, Json(Json::Object{{"type", "gelu"}}));
  mb.node("ReduceMean", {"c"}, {"y"}, Json(Json::Object{{"axis", Json(-1)}, {"keepdims", Json(0)}}));
  mb.output("y");
  auto m = mb.build();
  std::vector<std::vector<float>> in = {randn(4 * 8 * 8, 5)};
  const Out neon = run(m, tiers({"reference", "binary_scalar", "neon", "binary_avx2"}), in);
  for (const auto& kv : neon.kernels) EXPECT_TRUE(kv.second.find(".reference") != std::string::npos);
  expect_identical(neon.y, run(m, tiers(kRef), in).y, "stage neon");
  const Out opt = run(m, tiers(kOpt), in);
  const Out all = run(m, KernelPolicy{}, in);
  for (const Out* o : {&opt, &all})
    for (const auto& kv : o->kernels) {
      const bool avx2 = available(kv.second) && kv.second.find("_avx2") != std::string::npos;
      // 32-bit ARM has no fast GELU (denormal flush): its float_opt GELU is expected there
      const bool a32_gelu = !available("activation_gelu.float_fast") && kv.second.rfind("activation", 0) == 0 &&
                            kv.second.find(".float_opt") != std::string::npos;
      EXPECT_TRUE(avx2 || a32_gelu || kv.second.find(".float_fast") != std::string::npos);
    }
}

// ============================================================ data movement
BF_TEST(transpose_concat_gather_addbroadcast_mul_gate_bit_exact) {
  uint32_t seed = 400;
  struct TCase {
    Shape s;
    std::vector<int64_t> perm;
  };
  for (const TCase& t : {TCase{{1, 64, 12, 64}, {0, 2, 1, 3}}, TCase{{1, 8, 7, 8, 7, 16}, {0, 1, 3, 2, 4, 5}}, TCase{{3, 5, 7}, {2, 0, 1}},
                         TCase{{4, 6}, {1, 0}}, TCase{{2, 3, 4}, {0, 1, 2}}, TCase{{2, 3, 4, 5}, {3, 2, 1, 0}}}) {
    Json::Array p;
    for (auto v : t.perm) p.push_back(Json(v));
    bftest::ModelBuilder mb;
    mb.input("x", DType::kF32, t.s);
    mb.node("Transpose", {"x"}, {"y"}, Json(Json::Object{{"perm", Json(p)}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {randn(num_elements(t.s), seed++)};
    const Out ref = run(m, tiers(kRef), in);
    for (int threads : {1, 3}) expect_identical(run(m, force("Transpose", "transpose.float_opt"), in, threads).y, ref.y, "transpose");
  }
  for (int64_t axis : {0, 1, 2, -1}) {
    bftest::ModelBuilder mb;
    mb.input("a", DType::kF32, {2, 3, 4, 5});
    mb.input("b", DType::kF32, {2, 3, 4, 5});
    mb.node("Concat", {"a", "b", "a", "b", "b"}, {"y"}, Json(Json::Object{{"axis", Json(axis)}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {randn(120, seed++), randn(120, seed++)};
    const Out ref = run(m, tiers(kRef), in);
    for (int threads : {1, 4}) expect_identical(run(m, force("Concat", "concat.float_opt"), in, threads).y, ref.y, "concat");
  }
  {
    auto table = randn(7 * 33, seed++);
    std::vector<int32_t> ids = {3, 0, 6, 9, -1, 1, 6, 2};
    bftest::ModelBuilder mb;
    mb.constant("t", DType::kF32, {7, 33}, table.data());
    mb.constant("i", DType::kI32, {2, 4}, ids.data());
    mb.input("x", DType::kF32, {1});
    mb.node("Gather", {"t", "i"}, {"y"});
    mb.output("y");
    auto m = mb.build();
    const Out ref = run(m, tiers(kRef), {{0.f}});
    expect_identical(run(m, force("Gather", "gather.float_opt"), {{0.f}}, 2).y, ref.y, "gather");
  }
  struct BCase {
    Shape a, b;
  };
  for (const BCase& bc : {BCase{{1, 12, 64, 64}, {1, 1, 1, 64}}, BCase{{2, 3, 4, 5}, {3, 1, 5}}, BCase{{2, 3, 4, 5}, {2, 3, 4, 1}},
                          BCase{{2, 3, 4, 5}, {2, 3, 4, 5}}, BCase{{2, 3, 4, 5}, {1}}, BCase{{4, 6}, {4, 1}}}) {
    bftest::ModelBuilder mb;
    mb.input("a", DType::kF32, bc.a);
    mb.input("b", DType::kF32, bc.b);
    mb.node("AddBroadcast", {"a", "b"}, {"y"});
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {randn(num_elements(bc.a), seed++), randn(num_elements(bc.b), seed++)};
    const Out ref = run(m, tiers(kRef), in);
    for (const std::string& k : variants("add_broadcast.float_opt"))
      for (int threads : {1, 3}) expect_identical(run(m, force("AddBroadcast", k), in, threads).y, ref.y, k + " " + shape_str(bc.b));
  }
  for (Shape s : {Shape{1, 3, 5, 7}, Shape{2, 28, 28, 96}}) {
    const int64_t n = s[0], c = s[3];
    for (const char* gate : {"sigmoid", "none"}) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kF32, s);
      mb.input("g", DType::kF32, {n, c});
      mb.node("ChannelGate", {"x", "g"}, {"y"}, Json(Json::Object{{"gate", gate}}));
      mb.output("y");
      auto m = mb.build();
      std::vector<std::vector<float>> in = {with_specials(randn(num_elements(s), seed++), 1), randn(n * c, seed++, 3.f)};
      const Out ref = run(m, tiers(kRef), in);
      for (const std::string& k : variants("channel_gate.float_opt"))
        for (int threads : {1, 3}) expect_identical(run(m, force("ChannelGate", k), in, threads).y, ref.y, k);
    }
    bftest::ModelBuilder mb;
    mb.input("a", DType::kF32, s);
    mb.input("b", DType::kF32, s);
    mb.node("Mul", {"a", "b"}, {"y"});
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {with_specials(randn(num_elements(s), seed++), 2), randn(num_elements(s), seed++)};
    const Out ref = run(m, tiers(kRef), in);
    for (const std::string& k : variants("mul.float_opt"))
      for (int threads : {1, 3}) expect_identical(run(m, force("Mul", k), in, threads).y, ref.y, k);
  }
}

// ============================================================ MatMul (float)
BF_TEST(matmul_float_opt_within_bound) {
  // float GEMM with FMA vs the double reference: |opt - ref| <= 1e-4 (1 + |ref|)
  // (the float_opt GEMM bound, tests/cpp/test_float_kernels.cc).
  struct MCase {
    int64_t batch, m, k, n;
    int tb;
    double scale;
  };
  uint32_t seed = 500;
  double worst = 0;
  for (const MCase& c : {MCase{8, 50, 50, 64, 0, 1.0}, MCase{3, 7, 13, 9, 1, 0.125}, MCase{1, 1, 1, 1, 0, 1.0}, MCase{2, 97, 600, 33, 1, 1.0},
                         MCase{4, 12, 64, 17, 0, -2.0}, MCase{1, 200, 31, 40, 0, 1.0}}) {
    bftest::ModelBuilder mb;
    mb.input("a", DType::kF32, {c.batch, c.m, c.k});
    mb.input("b", DType::kF32, c.tb ? Shape{c.batch, c.n, c.k} : Shape{c.batch, c.k, c.n});
    mb.node("MatMul", {"a", "b"}, {"y"}, Json(Json::Object{{"transpose_b", Json(c.tb)}, {"scale", Json(c.scale)}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<std::vector<float>> in = {randn(c.batch * c.m * c.k, seed++), randn(c.batch * c.k * c.n, seed++, 1.f / std::sqrt(float(c.k)))};
    const Out ref = run(m, tiers(kRef), in);
    for (const std::string& k : variants("matmul.float_opt"))
      for (int threads : {1, 3}) {
        const Out o = run(m, force("MatMul", k), in, threads);
        for (size_t i = 0; i < o.y.size(); ++i) {
          const double r = std::fabs(double(o.y[i]) - double(ref.y[i])) / (1e-4 * (1 + std::fabs(double(ref.y[i]))));
          worst = std::max(worst, r);
          EXPECT_TRUE(r <= 1.0);
        }
      }
  }
  std::printf("    matmul float_opt worst err/bound %.3g\n", worst);
}

// ============================================================ BMatMul
BF_TEST(bmatmul_kernels_bit_exact) {
  std::vector<std::string> kernels;
  for (const char* k : {"bmatmul.bitplane", "bmatmul.neon", "bmatmul.avx2"})
    if (registered(k) && available(k)) kernels.push_back(k);
  EXPECT_TRUE(!kernels.empty());  // bitplane always; the platform kernel unless its features are masked
  uint32_t seed = 600;
  int compared = 0;
  struct BCase {
    int64_t batch, m, k, n;
    int tb;
  };
  const char* modes[] = {"ge", "gt", "sign_zero", "gt0"};
  for (const BCase& c : {BCase{12, 64, 64, 64, 1}, BCase{12, 64, 64, 64, 0}, BCase{2, 5, 1, 3, 0}, BCase{3, 9, 63, 7, 1}, BCase{2, 4, 65, 5, 0},
                         BCase{1, 17, 130, 33, 1}, BCase{2, 3, 200, 2, 0}, BCase{1, 2, 2100, 3, 1}}) {
    for (int ma = 0; ma < 4; ++ma)
      for (int mbm = 0; mbm < 4; ++mbm) {
        if (c.m * c.n > 1000 && (ma + mbm) % 3 != 0) continue;  // keep the big cases to a subset of mode pairs
        bftest::ModelBuilder mb;
        mb.input("a", DType::kF32, {c.batch, c.m, c.k});
        mb.input("b", DType::kF32, c.tb ? Shape{c.batch, c.n, c.k} : Shape{c.batch, c.k, c.n});
        mb.node("BMatMul", {"a", "b"}, {"y"},
                Json(Json::Object{{"a_binarize", modes[ma]}, {"b_binarize", modes[mbm]}, {"transpose_b", Json(c.tb)}, {"scale", Json(0.125)}}));
        mb.output("y");
        auto m = mb.build();
        std::vector<float> a = with_specials(randn(c.batch * c.m * c.k, seed), seed + 1);
        std::vector<float> b = with_specials(randn(c.batch * c.k * c.n, seed + 2), seed + 3);
        a[a.size() / 2] = std::numeric_limits<float>::quiet_NaN();
        const Out ref = run(m, tiers(kRef), {a, b});
        for (const std::string& k : kernels)
          for (int threads : {1, 3}) {
            expect_identical(run(m, force("BMatMul", k, {"reference", "binary_scalar", "neon", "binary_avx2"}), {a, b}, threads).y, ref.y,
                             k + " " + modes[ma] + "x" + modes[mbm]);
            ++compared;
          }
      }
  }
  // Inputs without exact zeros: rows of a / column blocks of b with every bit nonzero take
  // the NEON "full" tiles (both, a only, b only); some zero rows / columns mix the modes.
  // Shapes cover m % 4, n % 8 remainders, k % 32 and k > 15 words (several uint8 rounds).
  for (const BCase& c : {BCase{1, 198, 64, 198, 1}, BCase{1, 198, 198, 64, 0}, BCase{1, 1, 198, 64, 0}, BCase{2, 13, 32, 17, 1},
                         BCase{2, 7, 500, 9, 0}, BCase{1, 6, 1000, 16, 1}, BCase{3, 4, 96, 8, 0}}) {
    for (int ma = 0; ma < 4; ++ma)
      for (int mbm = 0; mbm < 4; ++mbm)
        for (int zeros = 0; zeros < 3; ++zeros) {  // 0: none, 1: a few zero rows of a, 2: a few zero columns of b
          if (c.m * c.n > 1000 && (ma + mbm) % 2 != 0 && !(ma == 2 && mbm == 2)) continue;
          bftest::ModelBuilder mb;
          mb.input("a", DType::kF32, {c.batch, c.m, c.k});
          mb.input("b", DType::kF32, c.tb ? Shape{c.batch, c.n, c.k} : Shape{c.batch, c.k, c.n});
          mb.node("BMatMul", {"a", "b"}, {"y"},
                  Json(Json::Object{{"a_binarize", modes[ma]}, {"b_binarize", modes[mbm]}, {"transpose_b", Json(c.tb)}, {"scale", Json(1.0)}}));
          mb.output("y");
          auto m = mb.build();
          std::vector<float> a = randn(c.batch * c.m * c.k, seed), b = randn(c.batch * c.k * c.n, seed + 2);
          seed += 4;
          if (zeros == 1)
            for (int64_t r = 0; r < c.batch * c.m; r += 5) a[r * c.k + (r * 7) % c.k] = 0.f;
          if (zeros == 2)
            for (int64_t j = 0; j < c.n; j += 3) b[c.tb ? j * c.k + (j * 5) % c.k : ((j * 5) % c.k) * c.n + j] = 0.f;
          const Out ref = run(m, tiers(kRef), {a, b});
          for (const std::string& k : kernels)
            for (int threads : {1, 3}) {
              expect_identical(run(m, force("BMatMul", k, {"reference", "binary_scalar", "neon", "binary_avx2"}), {a, b}, threads).y, ref.y,
                               k + " full-planes " + modes[ma] + "x" + modes[mbm]);
              ++compared;
            }
        }
  }
  std::printf("    bmatmul: %d kernel-vs-reference comparisons (%zu kernels)\n", compared, kernels.size());
  // default selection at stage neon picks the platform kernel
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {1, 4, 8});
  mb.input("b", DType::kF32, {1, 4, 8});
  mb.node("BMatMul", {"a", "b"}, {"y"}, Json(Json::Object{{"transpose_b", Json(1)}}));
  mb.output("y");
  const std::string chosen = run(mb.build(), tiers({"reference", "binary_scalar", "neon", "binary_avx2"}), {}).kernels.begin()->second;
  EXPECT_EQ(chosen, kernels.back());
}

// ============================================================ INT8 MatMul
namespace {

std::vector<std::string> int8_matmul_kernels() {
  std::vector<std::string> v;
  for (const char* k : {"matmul.int8_avx2", "matmul.int8_avx2_vnni", "matmul.int8_neon_sdot", "matmul.int8_neon_i8mm"})
    if (registered(k) && available(k)) v.push_back(k);
  return v;
}

const std::set<std::string> kInt8 = {"reference", "float_opt", "int8_reference", "int8_neon", "int8_avx2"};

}  // namespace

BF_TEST(int8_matmul_hand_computed) {
  // a = [[1, -2.5], [0.5, 4]] (max 4 -> sa = 4/127), b = I (sb = 1/127):
  // qa = rne(a / sa) = [[32, -79], [16, 127]] (31.75 -> 32, -79.375 -> -79, 15.875 -> 16),
  // qb = 127 I, acc = 127 qa, y = acc * (sa * sb) = qa * 4 / 127 (fp32).
  std::vector<float> a = {1.f, -2.5f, 0.5f, 4.f}, b = {1.f, 0.f, 0.f, 1.f};
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {1, 2, 2});
  mb.input("b", DType::kF32, {1, 2, 2});
  mb.node("MatMul", {"a", "b"}, {"y"});
  mb.output("y");
  auto m = mb.build();
  const float sa = 4.f / 127.f, sb = 1.f / 127.f, c = sa * sb;
  const int32_t qa[4] = {32, -79, 16, 127};
  for (const std::string& k : [] {
         auto v = int8_matmul_kernels();
         v.insert(v.begin(), "matmul.int8_reference");
         return v;
       }()) {
    const Out o = run(m, force("MatMul", k, kInt8), {a, b});
    for (int i = 0; i < 4; ++i) EXPECT_EQ(o.y[i], static_cast<float>(qa[i] * 127) * c + 0.f);
  }
}

BF_TEST(int8_matmul_bit_exact_vs_int8_reference) {
  const auto kernels = int8_matmul_kernels();
  struct ICase {
    int64_t batch, m, k, n;
    int tb;
    double scale;
  };
  uint32_t seed = 700;
  int compared = 0;
  double worst_bound = 0;
  for (const ICase& c : {ICase{8, 50, 50, 64, 0, 1.0}, ICase{3, 7, 13, 9, 1, 0.125}, ICase{1, 1, 1, 1, 0, 1.0}, ICase{2, 97, 300, 33, 1, 1.0},
                         ICase{4, 12, 64, 17, 0, -2.0}, ICase{1, 200, 31, 40, 0, 1.0}, ICase{5, 9, 3, 16, 1, 1.0}, ICase{2, 8, 5, 8, 0, 1.0},
                         ICase{1, 3, 6, 15, 0, 1.0}, ICase{2, 6, 7, 31, 1, 1.0}}) {
    bftest::ModelBuilder mb;
    mb.input("a", DType::kF32, {c.batch, c.m, c.k});
    mb.input("b", DType::kF32, c.tb ? Shape{c.batch, c.n, c.k} : Shape{c.batch, c.k, c.n});
    mb.node("MatMul", {"a", "b"}, {"y"}, Json(Json::Object{{"transpose_b", Json(c.tb)}, {"scale", Json(c.scale)}}));
    mb.output("y");
    auto m = mb.build();
    std::vector<float> a = randn(c.batch * c.m * c.k, seed++), b = randn(c.batch * c.k * c.n, seed++, 0.3f);
    for (size_t i = 0; i < a.size(); i += 11) a[i] = std::nearbyint(a[i] * 4.f) * 0.125f;  // values on .5 quantization ties
    const Out ref = run(m, force("MatMul", "matmul.int8_reference", kInt8), {a, b});
    for (const std::string& k : kernels)
      for (int threads : {1, 4}) {
        expect_identical(run(m, force("MatMul", k, kInt8), {a, b}, threads).y, ref.y, k);
        ++compared;
      }
    // analytic bound vs the float reference (docs/int8.md § MatMul):
    //   |y_int8 - y| <= |scale| * (sum_k |a||eb| + |b||ea| + |ea||eb|) + 8u (|scale| S + |y_int8|)
    const Out fl = run(m, tiers(kRef), {a, b});
    float ma = 0, mbv = 0;
    for (float v : a) ma = std::max(ma, std::fabs(v));
    for (float v : b) mbv = std::max(mbv, std::fabs(v));
    const double ea = (ma / 127.0) * (0.5 + 1e-5), eb = (mbv / 127.0) * (0.5 + 1e-5), u = std::ldexp(1.0, -24);
    for (int64_t bt = 0; bt < c.batch; ++bt)
      for (int64_t i = 0; i < c.m; ++i)
        for (int64_t j = 0; j < c.n; ++j) {
          double q = 0, s = 0;
          for (int64_t k = 0; k < c.k; ++k) {
            const double av = a[(bt * c.m + i) * c.k + k];
            const double bv = c.tb ? b[(bt * c.n + j) * c.k + k] : b[(bt * c.k + k) * c.n + j];
            q += std::fabs(av) * eb + std::fabs(bv) * ea + ea * eb;
            s += std::fabs(av * bv);
          }
          const size_t o = static_cast<size_t>((bt * c.m + i) * c.n + j);
          const double bound = std::fabs(c.scale) * q + 8 * u * (std::fabs(c.scale) * s + std::fabs(double(ref.y[o])));
          const double err = std::fabs(double(ref.y[o]) - double(fl.y[o]));
          worst_bound = std::max(worst_bound, err / bound);
          EXPECT_TRUE(err <= bound);
        }
  }
  std::printf("    int8 matmul: %d comparisons vs int8_reference (%zu kernels), worst err/analytic bound %.3f\n", compared, kernels.size(),
              worst_bound);
}

BF_TEST(int8_matmul_is_opt_in_and_gated) {
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {2, 8, 16});
  mb.input("b", DType::kF32, {2, 16, 8});
  mb.node("MatMul", {"a", "b"}, {"y"});
  mb.output("y");
  auto m = mb.build();
  const std::string all = run(m, KernelPolicy::all(), {}).kernels.begin()->second;
  EXPECT_TRUE(all.find("int8") == std::string::npos);
  const auto kernels = int8_matmul_kernels();
  const std::string chosen = run(m, tiers(kInt8), {}).kernels.begin()->second;
  if (!kernels.empty()) EXPECT_EQ(chosen, kernels.back());
  else EXPECT_EQ(chosen.find("float_opt") != std::string::npos || chosen == "matmul.int8_reference", true);
  // allow_int8 = 0 keeps the node in float
  bftest::ModelBuilder m2;
  m2.input("a", DType::kF32, {2, 8, 16});
  m2.input("b", DType::kF32, {2, 16, 8});
  m2.node("MatMul", {"a", "b"}, {"y"}, Json(Json::Object{{"allow_int8", Json(0)}}));
  m2.output("y");
  EXPECT_TRUE(run(m2.build(), tiers(kInt8), {}).kernels.begin()->second.find("int8") == std::string::npos);
}

// Re-executed with masked CPU features (cpu_info() is read once per process).
BF_TEST(int8_matmul_feature_masked_selection_child) {
  const char* expect = std::getenv("BF_EXPECT_MATMUL_KERNEL");
  if (!expect) return;  // only meaningful in the re-executed child
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {2, 8, 16});
  mb.input("b", DType::kF32, {2, 16, 8});
  mb.node("MatMul", {"a", "b"}, {"y"});
  mb.output("y");
  EXPECT_EQ(run(mb.build(), tiers(kInt8), {}).kernels.begin()->second, std::string(expect));
}

BF_TEST(int8_matmul_feature_masking) {
  if (std::getenv("BF_EXPECT_MATMUL_KERNEL")) return;
  std::vector<std::pair<std::string, std::string>> cases;
#if defined(__x86_64__)
  if (available("matmul.int8_avx2_vnni")) cases.push_back({"avxvnni", "matmul.int8_avx2"});
  if (available("matmul.int8_avx2")) cases.push_back({"avx2", "matmul.float_opt"});
#endif
#if defined(__aarch64__)
  if (available("matmul.int8_neon_i8mm")) cases.push_back({"i8mm", "matmul.int8_neon_sdot"});
  if (available("matmul.int8_neon_sdot")) cases.push_back({"dotprod", "matmul.float_opt"});
#endif
  for (const auto& c : cases) {
    const int rc = bftest::run_self("int8_matmul_feature_masked_selection_child",
                                    {"BITINO_DISABLE_FEATURES=" + c.first, "BF_EXPECT_MATMUL_KERNEL=" + c.second});
    if (rc != 0) throw bftest::Failure{"masked " + c.first + ": expected " + c.second + " (child exit " + std::to_string(rc) + ")"};
  }
}
