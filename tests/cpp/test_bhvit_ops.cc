// BHViT operators (docs/operators_bhvit.md): FakeQuant, ReduceMean and the
// Activation type sign_zero. Reference kernels vs hand computation (exact).
#include <cmath>
#include <cstring>
#include <random>

#include "bitino/bitino.h"
#include "testing.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

std::vector<float> randn(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = nd(rng);
  return v;
}

std::vector<float> run_ref(const std::shared_ptr<Model>& m, const std::vector<float>& x, int threads = 1) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers({"reference"});
  Session s(m, o);
  EXPECT_EQ(x.size() * sizeof(float), s.input(0).bytes());
  std::memcpy(s.input(0).raw(), x.data(), x.size() * sizeof(float));
  s.run();
  const Tensor& y = s.output(0);
  return std::vector<float>(y.data<float>(), y.data<float>() + y.size());
}

}  // namespace

BF_TEST(fake_quant_per_axis_round_half_even_and_clamp) {
  const int64_t B = 2, C = 3, L = 4, M = 5;
  auto x = randn(B * C * L * M, 11);
  for (auto& v : x) v = std::fabs(v) * 0.2f;  // softmax-like magnitudes
  std::vector<float> s = {0.75f, 0.0337f, 0.0682f}, z = {0.f, -1.379f, 0.3376f};
  // Exact ties for channel 0 (scale 0.75, zp 0, pre 3): (3x)/0.75 = 4x exactly.
  x[0] = 0.125f;  // 0.5 -> 0 (half to even; half away from zero would give 1)
  x[1] = 0.625f;  // 2.5 -> 2
  x[2] = 5.f;     // 20 -> clamped to qmax 3
  bftest::ModelBuilder mb;
  mb.constant("s", DType::kF32, {C}, s.data());
  mb.constant("z", DType::kF32, {C}, z.data());
  mb.input("x", DType::kF32, {B, C, L, M});
  mb.node("FakeQuant", {"x", "s", "z"}, {"y"},
          Json(Json::Object{{"axis", int64_t(1)}, {"pre_scale", 3.0}, {"qmin", int64_t(0)}, {"qmax", int64_t(3)}}));
  mb.output("y");
  auto y = run_ref(mb.build(), x, 3);
  for (int64_t i = 0; i < B * C * L * M; ++i) {
    int64_t c = (i / (L * M)) % C;
    float t = x[i] * 3.f;
    t = t / s[c];
    t = t + z[c];
    float q = std::nearbyint(t);
    q = std::fmin(std::fmax(q, 0.f), 3.f);
    EXPECT_EQ(y[i], q * s[c]);
  }
  EXPECT_EQ(y[0], 0.f);
  EXPECT_EQ(y[1], 2.f * 0.75f);
  EXPECT_EQ(y[2], 3.f * 0.75f);
}

BF_TEST(fake_quant_rejects_bad_scale_shape) {
  std::vector<float> s(4, 1.f);
  bftest::ModelBuilder mb;
  mb.constant("s", DType::kF32, {4}, s.data());
  mb.constant("z", DType::kF32, {4}, s.data());
  mb.input("x", DType::kF32, {1, 3, 2});
  mb.node("FakeQuant", {"x", "s", "z"}, {"y"}, Json(Json::Object{{"axis", int64_t(1)}}));
  mb.output("y");
  auto m = mb.build();
  EXPECT_THROWS(Session(m, SessionOptions{}));
}

BF_TEST(reduce_mean_any_axis_double_accumulation) {
  const int64_t A = 3, B = 7, C = 5;
  auto x = randn(A * B * C, 12);
  for (int64_t axis = 0; axis < 3; ++axis)
    for (int keep = 0; keep < 2; ++keep) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kF32, {A, B, C});
      mb.node("ReduceMean", {"x"}, {"y"}, Json(Json::Object{{"axis", axis == 2 ? int64_t(-1) : axis}, {"keepdims", int64_t(keep)}}));
      mb.output("y");
      auto y = run_ref(mb.build(), x, 2);
      const int64_t dims[3] = {A, B, C};
      int64_t outer = 1, inner = 1, len = dims[axis];
      for (int64_t d = 0; d < axis; ++d) outer *= dims[d];
      for (int64_t d = axis + 1; d < 3; ++d) inner *= dims[d];
      EXPECT_EQ(y.size(), static_cast<size_t>(outer * inner));
      for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
          double acc = 0;
          for (int64_t k = 0; k < len; ++k) acc += x[(o * len + k) * inner + i];
          EXPECT_EQ(y[o * inner + i], static_cast<float>(acc / len));
        }
    }
}

BF_TEST(activation_sign_zero_is_torch_sign) {
  std::vector<float> x = {1.5f, -2.f, 0.f, -0.f, 1e-30f, -1e-30f, 3.f, -7.f};
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {2, 4});
  mb.node("Activation", {"x"}, {"y"}, Json(Json::Object{{"type", "sign_zero"}}));
  mb.output("y");
  auto y = run_ref(mb.build(), x);
  const float want[] = {1.f, -1.f, 0.f, 0.f, 1.f, -1.f, 1.f, -1.f};
  for (size_t i = 0; i < x.size(); ++i) EXPECT_EQ(y[i], want[i]);
  EXPECT_TRUE(!std::signbit(y[3]));  // never -0
}
