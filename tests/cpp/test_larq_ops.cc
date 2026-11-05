// Reference kernels of the Larq-slice operators (QBConv2D, QBDense, ChannelGate)
// against independent hand computation. Contracts: docs/operators_larq.md.
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

std::unique_ptr<Session> session(const std::shared_ptr<Model>& m, int threads = 1) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers({"reference"});
  return std::make_unique<Session>(m, o);
}

}  // namespace

BF_TEST(dorefa_levels_round_half_to_even) {
  // k=2 (n=3): 0.5*3 = 1.5 -> 2 (even); k=1 (n=1): 0.5 -> 0 (even); clip below 0 / above 1.
  // QBDense with K=1 and weight +1 exposes the integer level directly (post_mul 1).
  struct Case {
    int bits;
    float x;
    int level;
  };
  const Case cases[] = {{2, 0.5f, 2}, {1, 0.5f, 0}, {2, -3.f, 0}, {2, 7.f, 3}, {2, 0.1f, 0},
                        {2, 0.2f, 1}, {2, 0.9f, 3}, {3, 0.5f, 4}, {1, 0.75f, 1}, {2, 0.f, 0}};
  for (const auto& c : cases) {
    std::vector<int8_t> w = {1};
    auto wp = bftest::pack_pm1(w, 1, 1);
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {1, 1}, wp.data());
    mb.input("x", DType::kF32, {1, 1});
    mb.node("QBDense", {"x", "w"}, {"y"}, Json(Json::Object{{"act_bits", Json(c.bits)}, {"act_quant", "dorefa"}}));
    mb.output("y");
    auto s = session(mb.build());
    s->input(0).data<float>()[0] = c.x;
    s->run();
    EXPECT_EQ(s->output(0).data<float>()[0], static_cast<float>(c.level));
  }
}

BF_TEST(qbconv2d_reference_matches_hand_computation) {
  // 1x3x3x4 input, 3x3 kernel, stride 1, pad 1 (zero padding of the *quantized*
  // input), 2-bit activations, 3 output channels, post_mul/post_add and relu.
  const int64_t H = 3, W = 3, C = 4, O = 3, K = 3;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> ud(-0.4f, 1.4f);
  std::vector<float> x(H * W * C);
  for (auto& v : x) v = ud(rng);
  x[0] = 0.5f;  // exact tie: level 2
  std::vector<int8_t> w(O * K * K * C);
  for (auto& v : w) v = (rng() & 1) ? 1 : -1;
  auto wp = bftest::pack_pm1(w, O * K * K, C);
  std::vector<float> mul = {0.25f, -0.5f, 1.f / 3.f}, add = {0.1f, 2.f, -1.f};
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {O, K, K, C}, wp.data(), "OHWI_bit");
  mb.constant("mul", DType::kF32, {O}, mul.data());
  mb.constant("add", DType::kF32, {O}, add.data());
  mb.input("x", DType::kF32, {1, H, W, C});
  mb.node("QBConv2D", {"x", "w", "mul", "add"}, {"y"},
          Json(Json::Object{{"strides", ints({1, 1})}, {"pads", ints({1, 1, 1, 1})}, {"act_bits", Json(2)},
                            {"act_quant", "dorefa"}, {"activation", "relu"}}));
  mb.output("y");
  for (int threads : {1, 3}) {
    auto s = session(mb.build(), threads);
    std::memcpy(s->input(0).raw(), x.data(), x.size() * sizeof(float));
    s->run();
    for (int64_t oy = 0; oy < H; ++oy)
      for (int64_t ox = 0; ox < W; ++ox)
        for (int64_t o = 0; o < O; ++o) {
          int acc = 0;
          for (int64_t ky = 0; ky < K; ++ky)
            for (int64_t kx = 0; kx < K; ++kx) {
              int64_t iy = oy - 1 + ky, ix = ox - 1 + kx;
              if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
              for (int64_t c = 0; c < C; ++c) {
                float v = x[(iy * W + ix) * C + c];
                float cl = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                int q = static_cast<int>(std::nearbyint(cl * 3.f));
                acc += q * w[((o * K + ky) * K + kx) * C + c];
              }
            }
          float e = static_cast<float>(acc) * mul[o] + add[o];
          e = e > 0.f ? e : 0.f;
          EXPECT_EQ(s->output(0).data<float>()[(oy * W + ox) * O + o], e);
        }
  }
}

BF_TEST(qbconv2d_grouped_and_strided) {
  // groups=2, stride 2, valid padding: output channel o only sees input group o / (O/groups).
  const int64_t H = 5, W = 5, C = 4, O = 4, K = 3, G = 2, KIC = C / G;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ud(-0.2f, 1.2f);
  std::vector<float> x(H * W * C);
  for (auto& v : x) v = ud(rng);
  std::vector<int8_t> w(O * K * K * KIC);
  for (auto& v : w) v = (rng() & 1) ? 1 : -1;
  auto wp = bftest::pack_pm1(w, O * K * K, KIC);
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {O, K, K, KIC}, wp.data(), "OHWI_bit");
  mb.input("x", DType::kF32, {1, H, W, C});
  mb.node("QBConv2D", {"x", "w"}, {"y"},
          Json(Json::Object{{"strides", ints({2, 2})}, {"groups", Json(G)}, {"act_bits", Json(2)}}));
  mb.output("y");
  auto s = session(mb.build());
  std::memcpy(s->input(0).raw(), x.data(), x.size() * sizeof(float));
  s->run();
  const int64_t OH = 2, OW = 2;
  for (int64_t oy = 0; oy < OH; ++oy)
    for (int64_t ox = 0; ox < OW; ++ox)
      for (int64_t o = 0; o < O; ++o) {
        int acc = 0;
        int64_t grp = o / (O / G);
        for (int64_t ky = 0; ky < K; ++ky)
          for (int64_t kx = 0; kx < K; ++kx)
            for (int64_t c = 0; c < KIC; ++c) {
              float v = x[((oy * 2 + ky) * W + ox * 2 + kx) * C + grp * KIC + c];
              float cl = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
              acc += static_cast<int>(std::nearbyint(cl * 3.f)) * w[((o * K + ky) * K + kx) * KIC + c];
            }
        EXPECT_EQ(s->output(0).data<float>()[(oy * OW + ox) * O + o], static_cast<float>(acc));
      }
}

BF_TEST(qbdense_reference_matches_hand_computation) {
  const int64_t N = 2, K = 37, O = 5;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> ud(-0.5f, 1.5f);
  std::vector<float> x(N * K);
  for (auto& v : x) v = ud(rng);
  std::vector<int8_t> w(O * K);
  for (auto& v : w) v = (rng() & 1) ? 1 : -1;
  auto wp = bftest::pack_pm1(w, O, K);
  std::vector<float> mul(O), add(O);
  for (int64_t o = 0; o < O; ++o) {
    mul[o] = 0.125f * static_cast<float>(o + 1);
    add[o] = -0.5f + static_cast<float>(o);
  }
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {O, K}, wp.data());
  mb.constant("mul", DType::kF32, {O}, mul.data());
  mb.constant("add", DType::kF32, {O}, add.data());
  mb.input("x", DType::kF32, {N, K});
  mb.node("QBDense", {"x", "w", "mul", "add"}, {"y"}, Json(Json::Object{{"act_bits", Json(2)}}));
  mb.output("y");
  auto s = session(mb.build(), 2);
  std::memcpy(s->input(0).raw(), x.data(), x.size() * sizeof(float));
  s->run();
  for (int64_t r = 0; r < N; ++r)
    for (int64_t o = 0; o < O; ++o) {
      int acc = 0;
      for (int64_t j = 0; j < K; ++j) {
        float v = x[r * K + j];
        float cl = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        acc += static_cast<int>(std::nearbyint(cl * 3.f)) * w[o * K + j];
      }
      EXPECT_EQ(s->output(0).data<float>()[r * O + o], static_cast<float>(acc) * mul[o] + add[o]);
    }
}

BF_TEST(channel_gate_matches_hand_computation) {
  const int64_t N = 2, H = 2, W = 3, C = 5;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 2.f);
  std::vector<float> x(N * H * W * C), g(N * C);
  for (auto& v : x) v = nd(rng);
  for (auto& v : g) v = nd(rng);
  for (bool keepdims : {false, true}) {
    for (const char* gate : {"sigmoid", "none"}) {
      bftest::ModelBuilder mb;
      mb.input("x", DType::kF32, {N, H, W, C});
      if (keepdims) mb.input("g", DType::kF32, {N, 1, 1, C});
      else mb.input("g", DType::kF32, {N, C});
      mb.node("ChannelGate", {"x", "g"}, {"y"}, Json(Json::Object{{"gate", gate}}));
      mb.output("y");
      auto s = session(mb.build());
      std::memcpy(s->input(0).raw(), x.data(), x.size() * sizeof(float));
      std::memcpy(s->input(1).raw(), g.data(), g.size() * sizeof(float));
      s->run();
      bool sig = std::string(gate) == "sigmoid";
      for (int64_t b = 0; b < N; ++b)
        for (int64_t p = 0; p < H * W; ++p)
          for (int64_t c = 0; c < C; ++c) {
            double gv = g[b * C + c];
            float sc = sig ? static_cast<float>(1.0 / (1.0 + std::exp(-gv))) : static_cast<float>(gv);
            int64_t i = (b * H * W + p) * C + c;
            EXPECT_EQ(s->output(0).data<float>()[i], x[i] * sc);
          }
    }
  }
}

BF_TEST(larq_ops_reject_invalid_attributes) {
  std::vector<int8_t> w(2 * 9 * 3, 1);
  auto wp = bftest::pack_pm1(w, 18, 3);
  auto build = [&](Json attrs) {
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {2, 3, 3, 3}, wp.data(), "OHWI_bit");
    mb.input("x", DType::kF32, {1, 4, 4, 3});
    mb.node("QBConv2D", {"x", "w"}, {"y"}, attrs);
    mb.output("y");
    return session(mb.build());
  };
  EXPECT_THROWS(build(Json(Json::Object{{"act_bits", Json(0)}})));
  EXPECT_THROWS(build(Json(Json::Object{{"act_bits", Json(2)}, {"pad_value", Json(1.0)}, {"pads", ints({1, 1, 1, 1})}})));
  EXPECT_THROWS(build(Json(Json::Object{{"act_bits", Json(2)}, {"act_quant", "uniform"}})));
  build(Json(Json::Object{{"act_bits", Json(2)}}));
  // ChannelGate shape mismatch.
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, 2, 2, 4});
  mb.input("g", DType::kF32, {1, 3});
  mb.node("ChannelGate", {"x", "g"}, {"y"});
  mb.output("y");
  EXPECT_THROWS(session(mb.build()));
}
