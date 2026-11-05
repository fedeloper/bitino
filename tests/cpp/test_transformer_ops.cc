// Transformer operators: reference kernels vs hand computation, and the
// ternary bit-plane BDense kernel vs the reference (bit-exact).
#include <cmath>
#include <cstring>
#include <random>

#include "bitino/bitino.h"
#include "testing.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

Json ints(std::initializer_list<int64_t> v) {
  Json::Array a;
  for (auto x : v) a.push_back(Json(x));
  return Json(a);
}

std::vector<float> randn(size_t n, uint32_t seed, float zero_every = 0) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = (zero_every > 0 && i % static_cast<size_t>(zero_every) == 0) ? 0.f : nd(rng);
  return v;
}

std::vector<float> run1(const std::shared_ptr<Model>& m, const std::vector<std::vector<uint8_t>>& inputs,
                        const std::set<std::string>& tiers = {"reference"}, int threads = 1, std::string* kernel = nullptr) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers(tiers);
  Session s(m, o);
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(inputs[i].size(), s.input(i).bytes());
    std::memcpy(s.input(i).raw(), inputs[i].data(), inputs[i].size());
  }
  s.run();
  if (kernel) {
    for (const auto& kv : s.kernel_choices()) *kernel += kv.second + ";";
  }
  const Tensor& y = s.output(0);
  return std::vector<float>(y.data<float>(), y.data<float>() + y.size());
}

template <typename T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

int sgn0(float x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }

}  // namespace

BF_TEST(gather_f32_and_bit_tables) {
  const int64_t V = 5, D = 37;
  auto tf = randn(V * D, 1);
  std::vector<int8_t> pm(V * D);
  std::vector<int8_t> zm(V * D, 1);
  std::mt19937 rng(2);
  for (auto& v : pm) v = rng() & 1 ? 1 : -1;
  for (size_t i = 0; i < zm.size(); i += 7) zm[i] = -1;  // -1 bit marks an exact zero
  auto tb = bftest::pack_pm1(pm, V, D);
  auto zb = bftest::pack_pm1(zm, V, D);
  std::vector<float> scale = {0.5f, 1.25f, 2.f, 0.125f, 3.f};
  std::vector<int32_t> ids = {3, 0, 4, 9, -1, 1};  // 9 and -1 are out of range -> zero rows
  for (int bit = 0; bit < 2; ++bit) {
    bftest::ModelBuilder mb2;
    if (bit) {
      mb2.constant("t", DType::kBit, {V, D}, tb.data());
      mb2.constant("s", DType::kF32, {V}, scale.data());
      mb2.constant("z", DType::kBit, {V, D}, zb.data());
      mb2.input("ids", DType::kI32, {2, 3});
      mb2.node("Gather", {"t", "ids", "s", "z"}, {"y"});
    } else {
      mb2.constant("t", DType::kF32, {V, D}, tf.data());
      mb2.input("ids", DType::kI32, {2, 3});
      mb2.node("Gather", {"t", "ids"}, {"y"});
    }
    mb2.output("y");
    auto y = run1(mb2.build(), {bytes_of(ids)});
    EXPECT_EQ(y.size(), static_cast<size_t>(6 * D));
    for (size_t i = 0; i < ids.size(); ++i)
      for (int64_t c = 0; c < D; ++c) {
        float want = 0.f;
        if (ids[i] >= 0 && ids[i] < V) {
          int64_t e = ids[i] * D + c;
          want = bit ? (zm[e] < 0 ? 0.f : pm[e] * scale[ids[i]]) : tf[e];
        }
        EXPECT_EQ(y[i * D + c], want);
      }
  }
}

BF_TEST(layernorm_matches_double_formula) {
  const int64_t R = 3, D = 50;
  auto x = randn(R * D, 3);
  for (auto& v : x) v = v * 3.f + 1.5f;
  auto g = randn(D, 4), b = randn(D, 5);
  bftest::ModelBuilder mb;
  mb.constant("g", DType::kF32, {D}, g.data());
  mb.constant("b", DType::kF32, {D}, b.data());
  mb.input("x", DType::kF32, {1, R, D});
  mb.node("LayerNorm", {"x", "g", "b"}, {"y"}, Json(Json::Object{{"eps", 1e-12}}));
  mb.output("y");
  auto y = run1(mb.build(), {bytes_of(x)}, {"reference"}, 2);
  for (int64_t r = 0; r < R; ++r) {
    double s = 0, v = 0;
    for (int64_t i = 0; i < D; ++i) s += x[r * D + i];
    double mean = s / D;
    for (int64_t i = 0; i < D; ++i) v += (x[r * D + i] - mean) * (x[r * D + i] - mean);
    double sd = std::sqrt(v / D + 1e-12);
    for (int64_t i = 0; i < D; ++i) {
      float want = static_cast<float>((x[r * D + i] - mean) / sd * g[i] + b[i]);
      EXPECT_EQ(y[r * D + i], want);
    }
  }
}

BF_TEST(bmatmul_modes_exact) {
  const int64_t B = 2, H = 3, M = 5, K = 70, N = 4;
  auto a = randn(B * H * M * K, 6, 5);  // exact zeros every 5th element
  auto bt = randn(B * H * N * K, 7, 3);
  const char* modes[] = {"ge", "gt", "sign_zero", "gt0"};
  auto apply = [](const std::string& m, float x) {
    if (m == "ge") return x >= 0 ? 1 : -1;
    if (m == "gt") return x > 0 ? 1 : -1;
    if (m == "sign_zero") return sgn0(x);
    return x > 0 ? 1 : 0;
  };
  for (int tb = 0; tb < 2; ++tb)
    for (const char* ma : modes)
      for (const char* mbm : modes) {
        // b given as [..., N, K] (transpose_b=1) or [..., K, N] (transpose_b=0).
        std::vector<float> b(bt.size());
        for (int64_t bh = 0; bh < B * H; ++bh)
          for (int64_t j = 0; j < N; ++j)
            for (int64_t k = 0; k < K; ++k)
              b[tb ? (bh * N + j) * K + k : (bh * K + k) * N + j] = bt[(bh * N + j) * K + k];
        bftest::ModelBuilder mb;
        mb.input("a", DType::kF32, {B, H, M, K});
        if (tb) mb.input("b", DType::kF32, {B, H, N, K});
        else mb.input("b", DType::kF32, {B, H, K, N});
        mb.node("BMatMul", {"a", "b"}, {"y"},
                Json(Json::Object{{"transpose_b", tb}, {"a_binarize", ma}, {"b_binarize", mbm}, {"scale", 0.125}}));
        mb.output("y");
        auto y = run1(mb.build(), {bytes_of(a), bytes_of(b)}, {"reference"}, 3);
        for (int64_t bh = 0; bh < B * H; ++bh)
          for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
              int acc = 0;
              for (int64_t k = 0; k < K; ++k) acc += apply(ma, a[(bh * M + i) * K + k]) * apply(mbm, bt[(bh * N + j) * K + k]);
              EXPECT_EQ(y[(bh * M + i) * N + j], static_cast<float>(acc) * 0.125f);
            }
      }
}

BF_TEST(matmul_double_accumulation) {
  const int64_t B = 2, M = 3, K = 33, N = 6;
  auto a = randn(B * M * K, 8), b = randn(B * K * N, 9);
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {B, M, K});
  mb.input("b", DType::kF32, {B, K, N});
  mb.node("MatMul", {"a", "b"}, {"y"}, Json(Json::Object{{"scale", 0.5}}));
  mb.output("y");
  auto y = run1(mb.build(), {bytes_of(a), bytes_of(b)});
  for (int64_t bt = 0; bt < B; ++bt)
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) {
        double acc = 0;
        for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(a[(bt * M + i) * K + k]) * b[(bt * K + k) * N + j];
        EXPECT_EQ(y[(bt * M + i) * N + j], static_cast<float>(acc * 0.5));
      }
}

BF_TEST(add_broadcast_and_transpose) {
  const int64_t A = 2, B = 3, C = 4, D = 5;
  auto x = randn(A * B * C * D, 10), m = randn(C, 11);
  bftest::ModelBuilder mb;
  mb.constant("m", DType::kF32, {C, 1}, m.data());
  mb.input("x", DType::kF32, {A, B, C, D});
  mb.node("AddBroadcast", {"x", "m"}, {"s"});
  mb.node("Transpose", {"s"}, {"y"}, Json(Json::Object{{"perm", ints({0, 2, 1, 3})}}));
  mb.output("y");
  auto y = run1(mb.build(), {bytes_of(x)});
  for (int64_t a = 0; a < A; ++a)
    for (int64_t b = 0; b < B; ++b)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t d = 0; d < D; ++d)
          EXPECT_EQ(y[((a * C + c) * B + b) * D + d], x[((a * B + b) * C + c) * D + d] + m[c]);
}

BF_TEST(gelu_tanh_double_formula) {
  std::vector<float> x = {-9.f, -6.f, -5.5f, -1.f, -0.f, 0.f, 1e-3f, 0.7f, 3.f};
  for (const char* t : {"gelu", "tanh"}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kF32, {1, static_cast<int64_t>(x.size())});
    mb.node("Activation", {"x"}, {"y"}, Json(Json::Object{{"type", t}}));
    mb.output("y");
    auto y = run1(mb.build(), {bytes_of(x)});
    for (size_t i = 0; i < x.size(); ++i) {
      double d = x[i];
      float want = std::string(t) == "gelu" ? static_cast<float>(d * 0.5 * (1.0 + std::erf(d / std::sqrt(2.0))))
                                            : static_cast<float>(std::tanh(d));
      EXPECT_EQ(y[i], want);
    }
  }
  // GELU in double keeps the sign for x in (-8.3, -5.5] where an fp32 erf underflows.
}

// BDense with ternary (sign_zero) inputs and optional zero weights:
// reference vs bdense.ternary_bitplane, bit-exact, several K (incl. non-multiples of 64).
BF_TEST(bdense_ternary_bitplane_matches_reference) {
  for (int64_t k : {7, 64, 100, 768}) {
    for (int wz = 0; wz < 2; ++wz)
      for (int thr = 0; thr < 2; ++thr) {
        const int64_t rows = 9, o = 37;
        std::mt19937 rng(static_cast<uint32_t>(k * 10 + wz * 2 + thr));
        std::vector<int8_t> w(o * k), z(o * k, 1);
        for (auto& v : w) v = rng() & 1 ? 1 : -1;
        for (size_t i = 0; i < z.size(); i += 11) z[i] = -1;
        auto wp = bftest::pack_pm1(w, o, k), zp = bftest::pack_pm1(z, o, k);
        auto mul = randn(o, 12), add = randn(o, 13), th = randn(k, 14);
        auto x = randn(rows * k, 15, 4);
        if (thr) for (int64_t i = 0; i < rows * k; i += 9) x[i] = th[i % k];  // exact ties with the threshold
        bftest::ModelBuilder mb;
        mb.constant("w", DType::kBit, {o, k}, wp.data());
        mb.constant("mul", DType::kF32, {o}, mul.data());
        mb.constant("add", DType::kF32, {o}, add.data());
        mb.constant("thr", DType::kF32, {k}, th.data());
        mb.constant("z", DType::kBit, {o, k}, zp.data());
        mb.input("x", DType::kF32, {1, rows, k});
        mb.node("BDense", {"x", "w", "mul", "add", thr ? "thr" : "", wz ? "z" : ""}, {"y"},
                Json(Json::Object{{"input_binarize", "sign_zero"}}));
        mb.output("y");
        auto m = mb.build();
        std::string k1, k2;
        auto ref = run1(m, {bytes_of(x)}, {"reference"}, 1, &k1);
        auto opt = run1(m, {bytes_of(x)}, {"reference", "binary_scalar"}, 4, &k2);  // SIMD tiers: test_binary_avx2.cc
        EXPECT_TRUE(k1.find("bdense.reference") != std::string::npos);
        EXPECT_TRUE(k2.find("bdense.ternary_bitplane") != std::string::npos);
        EXPECT_TRUE(std::memcmp(ref.data(), opt.data(), ref.size() * 4) == 0);
        // Hand check of one row.
        for (int64_t oo = 0; oo < o; ++oo) {
          int acc = 0;
          for (int64_t j = 0; j < k; ++j) {
            int wv = (wz && z[oo * k + j] < 0) ? 0 : w[oo * k + j];
            acc += sgn0(x[j] - (thr ? th[j] : 0.f)) * wv;
          }
          EXPECT_EQ(ref[oo], static_cast<float>(acc) * mul[oo] + add[oo]);
        }
      }
  }
}

BF_TEST(bdense_w_zero_requires_sign_zero) {
  std::vector<int8_t> w(4 * 8, 1);
  auto wp = bftest::pack_pm1(w, 4, 8);
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {4, 8}, wp.data());
  mb.constant("z", DType::kBit, {4, 8}, wp.data());
  mb.input("x", DType::kF32, {1, 8});
  mb.node("BDense", {"x", "w", "", "", "", "z"}, {"y"}, Json(Json::Object{{"input_binarize", "ge"}}));
  mb.output("y");
  auto m = mb.build();
  EXPECT_THROWS(Session(m, SessionOptions{}));
}
