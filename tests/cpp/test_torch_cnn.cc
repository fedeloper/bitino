// Mul reference kernel vs hand computation.
#include <cstring>

#include "bitino/bitino.h"
#include "tools/model_builder.h"
#include "testing.h"

using namespace bitino;

BF_TEST(mul_same_shape_matches_hand_computation) {
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {1, 3, 3, 1001});
  mb.input("b", DType::kF32, {1, 3, 3, 1001});
  mb.node("Mul", {"a", "b"}, {"y"});
  mb.output("y");
  const int n = 3 * 3 * 1001;
  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = 0.37f * static_cast<float>(i % 97) - 11.f;
    b[i] = 1.f / static_cast<float>(1 + i % 13) - 0.25f;
  }
  for (int threads : {1, 4}) {
    SessionOptions o;
    o.num_threads = threads;
    Session s(mb.build(), o);
    std::memcpy(s.input(0).raw(), a.data(), n * sizeof(float));
    std::memcpy(s.input(1).raw(), b.data(), n * sizeof(float));
    s.run();
    const float* y = s.output(0).data<float>();
    for (int i = 0; i < n; ++i) EXPECT_EQ(y[i], a[i] * b[i]);
  }
}

BF_TEST(mul_rejects_shape_mismatch) {
  bftest::ModelBuilder mb;
  mb.input("a", DType::kF32, {1, 2, 2, 3});
  mb.input("b", DType::kF32, {1, 1, 1, 3});
  mb.node("Mul", {"a", "b"}, {"y"});
  mb.output("y");
  EXPECT_THROWS(Session(mb.build(), {}));
}
