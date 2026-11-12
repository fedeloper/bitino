// Image-feature operators (docs/operators_image_features.md): reference kernels vs hand computation
// (gray formula, a direct long-double DFT, scikit-image LBP on a hand-made image), and the optimized kernels
// (fft_log_magnitude.opt / local_binary_pattern.opt; NEON on AArch64) bit for bit against the reference.
#include <cmath>
#include <cstring>
#include <random>

#include "bitino/bitino.h"
#include "kernels/image_features_common.h"
#include "tools/model_builder.h"
#include "testing.h"

using namespace bitino;

namespace {

Json nums(std::initializer_list<double> v) {
  Json::Array a;
  for (double d : v) a.push_back(Json(d));
  return Json(a);
}

std::shared_ptr<Model> feature_model(int64_t h, int64_t w, Json lbp_attrs) {
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, h, w, 3});
  mb.node("ImageGrayU8", {"x"}, {"g"},
          Json(Json::Object{{"coeffs", Json(Json::Array{Json(3735), Json(19235), Json(9798)})}, {"shift", Json(15)}, {"scale", Json(255.0)}}));
  mb.node("FftLogMagnitude", {"g"}, {"f"},
          Json(Json::Object{{"eps", Json(1e-9)}, {"mul", Json(20.0)}, {"div", Json(255.0)}, {"shift", Json(1)}}));
  mb.node("LocalBinaryPattern", {"g"}, {"l"}, lbp_attrs);
  mb.output("g");
  mb.output("f");
  mb.output("l");
  return mb.build();
}

Json lbp_p3r6() {  // skimage offsets for P = 3, R = 6 (np.round(..., 5))
  return Json(Json::Object{{"rp", nums({-0.0, -5.19615, 5.19615})}, {"cp", nums({6.0, -3.0, -3.0})}, {"method", Json("uniform")},
                           {"div", Json(255.0)}});
}

struct Outs {
  std::vector<float> g, f, l;
};

Outs run(const std::shared_ptr<Model>& model, const std::vector<float>& x, const std::string& fft_kernel,
         const std::string& lbp_kernel, int threads) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy.force = {{"FftLogMagnitude", fft_kernel}, {"LocalBinaryPattern", lbp_kernel}};
  Session s(model, o);
  std::memcpy(s.input(0).raw(), x.data(), x.size() * sizeof(float));
  s.run();
  Outs r;
  for (int i = 0; i < 3; ++i) {
    const float* p = s.output(i).data<float>();
    std::vector<float> v(p, p + s.output(i).size());
    (i == 0 ? r.g : i == 1 ? r.f : r.l) = v;
  }
  return r;
}

std::vector<float> random_image(int64_t h, int64_t w, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<float> x(h * w * 3);
  for (auto& v : x) v = static_cast<float>(rng() % 256) * (1.f / 255.f);
  return x;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

BF_TEST(image_gray_u8_matches_opencv_fixed_point_formula) {
  const int64_t h = 1, w = 7;
  std::vector<float> x = {0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f,
                          -3.f, 9.f, NAN, 0.5f, 0.25f, 0.75f};
  Outs r = run(feature_model(h, w, lbp_p3r6()), x, "fft_log_magnitude.reference", "local_binary_pattern.reference", 1);
  // (3735 R + 19235 G + 9798 B + 2^14) >> 15 on trunc(x * 255) (OpenCV BGR2GRAY applied to RGB data)
  const int want[7] = {0, 255, (3735 * 255 + 16384) >> 15, (19235 * 255 + 16384) >> 15, (9798 * 255 + 16384) >> 15,
                       (19235 * 255 + 16384) >> 15, (3735 * 127 + 19235 * 63 + 9798 * 191 + 16384) >> 15};
  for (int i = 0; i < 7; ++i) EXPECT_EQ(r.g[i], static_cast<float>(want[i]));
}

BF_TEST(fft_log_magnitude_reference_matches_direct_dft) {
  for (auto hw : {std::pair<int64_t, int64_t>{6, 10}, {7, 4}, {12, 15}}) {
    const int64_t h = hw.first, w = hw.second;
    std::vector<float> x = random_image(h, w, 7);
    Outs r = run(feature_model(h, w, lbp_p3r6()), x, "fft_log_magnitude.reference", "local_binary_pattern.reference", 1);
    const long double pi = 3.141592653589793238462643383279502884L;
    for (int64_t k1 = 0; k1 < h; ++k1)
      for (int64_t k2 = 0; k2 < w; ++k2) {
        long double re = 0, im = 0;
        for (int64_t a = 0; a < h; ++a)
          for (int64_t b = 0; b < w; ++b) {
            const long double ang = -2 * pi * (static_cast<long double>(k1 * a) / h + static_cast<long double>(k2 * b) / w);
            re += r.g[a * w + b] * std::cos(ang);
            im += r.g[a * w + b] * std::sin(ang);
          }
        const double want = static_cast<double>(20.0L * std::log(std::sqrt(re * re + im * im) + 1e-9L) / 255.0L);
        const int64_t rr = (k1 + h / 2) % h, cc = (k2 + w / 2) % w;
        EXPECT_NEAR(r.f[rr * w + cc], want, 1e-5 * (1 + std::fabs(want)));
      }
  }
}

BF_TEST(local_binary_pattern_reference_matches_hand_computation) {
  // 1 x 14 gray row (P = 3, R = 6): sample 0 at (r, c + 6), samples 1 / 2 at rows r -+ 5.19615 (outside the
  // image -> bilinear mix of 0 and 0 = 0), columns c - 3. bit = (texture - center >= 0); uniform: number of ones.
  const int64_t h = 1, w = 14;
  std::vector<float> x(w * 3);
  const int gray[14] = {10, 0, 50, 7, 7, 0, 3, 200, 9, 0, 1, 2, 3, 0};
  for (int c = 0; c < w; ++c)
    for (int k = 0; k < 3; ++k) x[c * 3 + k] = static_cast<float>(gray[c]) * (1.f / 255.f);
  Outs r = run(feature_model(h, w, lbp_p3r6()), x, "fft_log_magnitude.reference", "local_binary_pattern.reference", 1);
  for (int c = 0; c < w; ++c) {
    EXPECT_EQ(r.g[c], static_cast<float>(gray[c]));
    const double right = c + 6 < w ? gray[c + 6] : 0.0;
    int ones = (right - gray[c] >= 0) + 2 * (0.0 - gray[c] >= 0);
    EXPECT_EQ(r.l[c], static_cast<float>(ones / 255.0));
  }
}

BF_TEST(image_feature_opt_kernels_bit_identical_to_reference) {
  const Json lbp_frac = Json(Json::Object{{"rp", nums({-0.0, -1.76777, -2.5, -1.76777, -0.0, 1.76777, 2.5, 1.76777})},
                                          {"cp", nums({2.5, 1.76777, 0.0, -1.76777, -2.5, -1.76777, -0.0, 1.76777})},
                                          {"method", Json("default")}, {"div", Json(1.0)}});
  for (auto hw : {std::pair<int64_t, int64_t>{224, 224}, {30, 42}, {15, 21}, {1, 9}, {35, 5}}) {
    for (int li = 0; li < 2; ++li) {
      const auto model = feature_model(hw.first, hw.second, li == 0 ? lbp_p3r6() : lbp_frac);
      const std::vector<float> x = random_image(hw.first, hw.second, 3 + li);
      const Outs ref = run(model, x, "fft_log_magnitude.reference", "local_binary_pattern.reference", 1);
      for (int threads : {1, 3}) {
        const Outs opt = run(model, x, "fft_log_magnitude.opt", "local_binary_pattern.opt", threads);
        EXPECT_TRUE(same_bits(ref.f, opt.f));
        EXPECT_TRUE(same_bits(ref.l, opt.l));
      }
    }
  }
}

BF_TEST(fft_log_magnitude_rejects_large_prime_factor) {
  bftest::ModelBuilder mb;
  mb.input("g", DType::kF32, {1, 11, 8, 1});
  mb.node("FftLogMagnitude", {"g"}, {"f"}, Json(Json::Object{}));
  mb.output("f");
  EXPECT_THROWS(Session(mb.build(), {}));
}
