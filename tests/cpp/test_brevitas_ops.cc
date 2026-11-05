// Brevitas / FINN BNN-PYNQ support: FakeQuant reference kernel vs hand computation,
// attribute validation, and the integer-threshold BN+sign patterns of the converted
// models (bit-exact across every kernel tier and thread count).
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

#include "bitino/bitino.h"
#include "testing.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

std::vector<uint8_t> run_tiers(const std::shared_ptr<Model>& m, const std::vector<float>& x, const std::set<std::string>& tiers,
                               int threads, std::string* kernels = nullptr) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers(tiers);
  Session s(m, o);
  EXPECT_EQ(s.input(0).size(), static_cast<int64_t>(x.size()));
  std::memcpy(s.input(0).raw(), x.data(), x.size() * sizeof(float));
  s.run();
  if (kernels)
    for (const auto& kv : s.kernel_choices()) *kernels += kv.second + ";";
  const Tensor& y = s.output(0);
  std::vector<uint8_t> out(y.bytes());
  std::memcpy(out.data(), y.raw(), out.size());
  return out;
}

std::vector<float> as_f32(const std::vector<uint8_t>& b) {
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

// Generic FakeQuant (docs/operators_bhvit.md) as used for Brevitas CNV's Q1.7 input:
// per-channel scale 1/128, zero point 0, [-128, 127], over the last axis.
void add_q17(bftest::ModelBuilder& mb, int64_t c, const std::string& in, const std::string& out) {
  static std::vector<float> sc, zp;
  sc.assign(c, 1.f / 128.f);
  zp.assign(c, 0.f);
  mb.constant(out + "_scale", DType::kF32, {c}, sc.data());
  mb.constant(out + "_zp", DType::kF32, {c}, zp.data());
  mb.node("FakeQuant", {in, out + "_scale", out + "_zp"}, {out},
          Json(Json::Object{{"axis", -1}, {"pre_scale", 1.0}, {"qmin", -128}, {"qmax", 127}}));
}

std::shared_ptr<Model> fake_quant_model(int64_t n) {
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, n});
  add_q17(mb, n, "x", "y");
  mb.output("y");
  return mb.build();
}

}  // namespace

BF_TEST(fake_quant_q17_ties_clamp_and_signed_zero) {
  // Brevitas CNV input quantizer: scale 1/128, [-128, 127], round half to even.
  const float s = 1.f / 128.f;
  std::vector<float> x = {0.5f * s, 1.5f * s, 2.5f * s, -0.5f * s, -1.5f * s, -2.5f * s, 0.3f * s, -0.3f * s,
                          -1.f, 1.f, 127.5f * s, -128.5f * s, 3.f, -3.f, 0.f, -0.f};
  std::vector<float> want = {0.f, 2.f * s, 2.f * s, -0.f, -2.f * s, -2.f * s, 0.f, -0.f,
                             -1.f, 127.f * s, 127.f * s, -1.f, 127.f * s, -1.f, 0.f, 0.f};
  // Note the last one: x = -0.0 gives t = -0.0 / s + 0 = +0.0 (IEEE: -0 + +0 = +0), as in Brevitas.
  auto y = as_f32(run_tiers(fake_quant_model(static_cast<int64_t>(x.size())), x, {"reference"}, 1));
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(y[i], want[i]);
    EXPECT_EQ(std::signbit(y[i]), std::signbit(want[i]));
  }
}

BF_TEST(fake_quant_q17_all_totensor_pixel_values) {
  // 2*(k/255) - 1 for every uint8 k: exactly clamp(round(128(2k-255)/255), -128, 127)/128.
  std::vector<float> x(256);
  for (int k = 0; k < 256; ++k) x[k] = 2.f * (static_cast<float>(k) / 255.f) - 1.f;
  auto y = as_f32(run_tiers(fake_quant_model(256), x, {}, 2));
  for (int k = 0; k < 256; ++k) {
    double q = std::nearbyint(128.0 * (2 * k - 255) / 255.0);
    q = std::min(std::max(q, -128.0), 127.0);
    EXPECT_EQ(y[k], static_cast<float>(q / 128.0));
  }
}

// CNV-style integer-threshold chain: FakeQuant -> float Conv2D (+-1 weights, bias multiple
// of 1/128) -> BConv2D (post_mul +-1, integer post_add, f32 out) -> MaxPool2D ->
// BConv2D (bit out) -> BConv2D (bit in) -> BDense chain (threshold-free ge) -> logits.
// Every value is exact, so every tier (incl. float_opt Conv2D) must match the reference bit for bit.
BF_TEST(brevitas_cnv_style_chain_bit_exact_all_tiers) {
  std::mt19937 rng(11);
  auto pm = [&](size_t n) {
    std::vector<int8_t> v(n);
    for (auto& e : v) e = rng() & 1 ? 1 : -1;
    return v;
  };
  auto sgn = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& e : v) e = rng() % 5 == 0 ? -1.f : 1.f;
    return v;
  };
  auto ints = [&](size_t n, int lim) {
    std::vector<float> v(n);
    for (auto& e : v) e = static_cast<float>(static_cast<int>(rng() % (2 * lim + 1)) - lim);
    return v;
  };
  const int64_t H = 14, C0 = 3, C1 = 16, C2 = 64, C3 = 40, C4 = 96, F1 = 70, NC = 10;
  bftest::ModelBuilder mb;
  // conv1: float +-1 weights, bias = integer/128.
  std::vector<float> w1(C1 * 9 * C0), b1(C1);
  for (auto& v : w1) v = rng() & 1 ? 1.f : -1.f;
  for (auto& v : b1) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 128.f;
  mb.constant("w1", DType::kF32, {C1, 3, 3, C0}, w1.data());
  mb.constant("b1", DType::kF32, {C1}, b1.data());
  auto w2 = bftest::pack_pm1(pm(C2 * 9 * C1), C2 * 9, C1);
  auto m2 = sgn(C2), a2 = ints(C2, 20);
  mb.constant("w2", DType::kBit, {C2, 3, 3, C1}, w2.data());
  mb.constant("m2", DType::kF32, {C2}, m2.data());
  mb.constant("a2", DType::kF32, {C2}, a2.data());
  auto w3 = bftest::pack_pm1(pm(C3 * 9 * C2), C3 * 9, C2);
  auto m3 = sgn(C3), a3 = ints(C3, 30);
  mb.constant("w3", DType::kBit, {C3, 3, 3, C2}, w3.data());
  mb.constant("m3", DType::kF32, {C3}, m3.data());
  mb.constant("a3", DType::kF32, {C3}, a3.data());
  auto w4 = bftest::pack_pm1(pm(C4 * 9 * C3), C4 * 9, C3);
  auto m4 = sgn(C4), a4 = ints(C4, 30);
  mb.constant("w4", DType::kBit, {C4, 3, 3, C3}, w4.data());
  mb.constant("m4", DType::kF32, {C4}, m4.data());
  mb.constant("a4", DType::kF32, {C4}, a4.data());
  auto wf1 = bftest::pack_pm1(pm(F1 * C4), F1, C4);
  auto mf1 = sgn(F1), af1 = ints(F1, 10);
  mb.constant("wf1", DType::kBit, {F1, C4}, wf1.data());
  mb.constant("mf1", DType::kF32, {F1}, mf1.data());
  mb.constant("af1", DType::kF32, {F1}, af1.data());
  auto wf2 = bftest::pack_pm1(pm(NC * F1), NC, F1);
  std::vector<float> mf2(NC, 0.585f / 122.4f), af2(NC, -0.907f);
  mb.constant("wf2", DType::kBit, {NC, F1}, wf2.data());
  mb.constant("mf2", DType::kF32, {NC}, mf2.data());
  mb.constant("af2", DType::kF32, {NC}, af2.data());
  Json valid = Json::Object{{"strides", Json(Json::Array{Json(1), Json(1)})}, {"pads", Json(Json::Array{Json(0), Json(0), Json(0), Json(0)})}};
  auto conv_attrs = [&](const char* out) {
    Json a = valid;
    a["input_binarize"] = Json("ge");
    a["output"] = Json(out);
    return a;
  };
  mb.input("x", DType::kF32, {1, H, H, C0});
  add_q17(mb, C0, "x", "q");
  mb.node("Conv2D", {"q", "w1", "b1"}, {"c1"}, valid);
  mb.node("BConv2D", {"c1", "w2", "m2", "a2", ""}, {"c2"}, conv_attrs("f32"));
  Json pool = Json::Object{{"kernel", Json(Json::Array{Json(2), Json(2)})}, {"strides", Json(Json::Array{Json(2), Json(2)})}};
  mb.node("MaxPool2D", {"c2"}, {"p2"}, pool);
  mb.node("BConv2D", {"p2", "w3", "m3", "a3", ""}, {"c3"}, conv_attrs("bit"));
  mb.node("BConv2D", {"c3", "w4", "m4", "a4", ""}, {"c4"}, conv_attrs("f32"));
  mb.node("Reshape", {"c4"}, {"r4"}, Json(Json::Object{{"shape", Json(Json::Array{Json(1), Json(-1)})}}));
  // flatten of [1,1,1,C4]: H=14 -> 12 -> 10 -> pool 5 -> 3 -> 1.
  Json dense = Json::Object{{"input_binarize", "ge"}};
  mb.node("BDense", {"r4", "wf1", "mf1", "af1", ""}, {"f1"}, dense);
  mb.node("BDense", {"f1", "wf2", "mf2", "af2", ""}, {"logits"}, dense);
  mb.output("logits");
  auto m = mb.build();
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<float> x(H * H * C0);
    for (auto& v : x) v = 2.f * (static_cast<float>(static_cast<int>(ud(rng) * 255.f)) / 255.f) - 1.f;
    auto ref = run_tiers(m, x, {"reference"}, 1);
    for (const auto& tiers : std::vector<std::set<std::string>>{
             {"reference", "packed_naive"}, {"reference", "binary_scalar"}, {"reference", "binary_scalar", "neon"},
             {"reference", "binary_scalar", "neon", "float_opt"}, {}}) {
      for (int threads : {1, 3}) {
        std::string k;
        auto got = run_tiers(m, x, tiers, threads, &k);
        if (got != ref) throw bftest::Failure{"brevitas chain mismatch, kernels " + k};
      }
    }
  }
}

// FC-style: BDense with input threshold 0.5 (sign(2x-1)) -> integer-threshold hidden layers.
BF_TEST(brevitas_fc_threshold_half_matches_explicit_2x_minus_1) {
  std::mt19937 rng(5);
  const int64_t K = 784, O = 64;
  std::vector<int8_t> w(O * K);
  for (auto& e : w) e = rng() & 1 ? 1 : -1;
  auto wp = bftest::pack_pm1(w, O, K);
  std::vector<float> thr(K, 0.5f), two(K, 2.f), m1(K, -1.f);
  auto build = [&](bool threshold) {
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {O, K}, wp.data());
    mb.input("x", DType::kF32, {1, K});
    if (threshold) {
      mb.constant("t", DType::kF32, {K}, thr.data());
      mb.node("BDense", {"x", "w", "", "", "t"}, {"y"}, Json(Json::Object{{"input_binarize", "ge"}}));
    } else {
      mb.constant("s", DType::kF32, {K}, two.data());
      mb.constant("b", DType::kF32, {K}, m1.data());
      mb.node("ChannelAffine", {"x", "s", "b"}, {"z"});
      mb.node("BDense", {"z", "w", "", "", ""}, {"y"}, Json(Json::Object{{"input_binarize", "ge"}}));
    }
    mb.output("y");
    return mb.build();
  };
  auto a = build(true), b = build(false);
  std::vector<float> x(K);
  for (int k = 0; k < 256; ++k) x[k] = static_cast<float>(k) / 255.f;   // every ToTensor value incl. 128/255
  for (int64_t k = 256; k < K; ++k) x[k] = static_cast<float>(rng() % 256) / 255.f;
  x[300] = 0.5f;                                                          // exact tie: 2x-1 == 0 -> +1
  auto ref = run_tiers(b, x, {"reference"}, 1);
  for (const auto& tiers : std::vector<std::set<std::string>>{{"reference"}, {"reference", "binary_scalar"}, {}})
    for (int threads : {1, 2}) EXPECT_TRUE(run_tiers(a, x, tiers, threads) == ref);
}
