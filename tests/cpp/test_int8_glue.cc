// int8_full ops for accuracy-carrying glue (2026-11-13; docs/int8.md): the NEON kernels of I8ChannelShift, I8BinarizeDyn
// and the full-resolution / table-gated I8GatedAdd bit-exact against the reference kernels on random shapes, channel tails,
// zero points and constants at the int64 / int32 bounds; I8BinarizeDyn also against its definition (q - z) D >= acc,
// I8SimAM (reference only) against hand values. On x86-64 the int8_full stage runs the reference kernels (plumbing only).
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <set>

#include "bitino/bitino.h"
#include "testing.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

const char* kRefTiers = "reference,int8_full_reference";
const char* kFullTiers = "reference,binary_scalar,neon,binary_avx2,float_opt,int8_full_reference,int8_full_opt,int8_full_neon";

struct Rng {
  std::mt19937 g;
  explicit Rng(uint32_t s) : g(s) {}
  int i(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
  std::vector<uint8_t> i8(int64_t n) {
    std::vector<uint8_t> v(static_cast<size_t>(n));
    for (auto& x : v) x = static_cast<uint8_t>(i(-128, 127));
    return v;
  }
};

std::vector<uint8_t> run_all(const bftest::ModelBuilder& mb, const std::string& stage_tiers, int threads,
                             const std::vector<std::vector<uint8_t>>& inputs, std::map<std::string, std::string>* kernels = nullptr) {
  auto model = mb.build();
  SessionOptions so;
  so.num_threads = threads;
  so.spin_run = threads > 1;
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
  std::vector<uint8_t> out;
  for (size_t i = 0; i < s.num_outputs(); ++i) {
    const Tensor& y = s.output(i);
    const auto* b = static_cast<const uint8_t*>(y.raw());
    out.insert(out.end(), b, b + y.bytes());
  }
  return out;
}

std::vector<uint8_t> compare(const bftest::ModelBuilder& mb, const std::vector<std::vector<uint8_t>>& in, const std::string& what,
                             const std::string& neon_kernel) {
  auto ref = run_all(mb, kRefTiers, 1, in);
  for (int t : {1, 4}) {
    std::map<std::string, std::string> ks;
    auto got = run_all(mb, kFullTiers, t, in, &ks);
    if (got != ref) throw bftest::Failure{"int8_full output differs from the reference: " + what};
#if defined(__aarch64__) || (defined(__ARM_NEON) && !defined(__aarch64__))
    bool found = neon_kernel.empty();
    for (const auto& kv : ks) found = found || kv.second == neon_kernel;
    if (!found) throw bftest::Failure{"expected kernel " + neon_kernel + " for " + what};
#endif
  }
  return ref;
}

Json ints(const std::vector<int64_t>& v) {
  Json::Array a;
  for (auto x : v) a.push_back(Json(x));
  return Json(a);
}

int64_t isqrt_def(int64_t v) {  // independent: bisection
  int64_t lo = 0, hi = int64_t(1) << 31;
  while (hi - lo > 1) {
    const int64_t m = (lo + hi) / 2;
    if (m * m <= v) lo = m;
    else hi = m;
  }
  return lo;
}

}  // namespace

BF_TEST(int8_glue_channel_shift_bit_exact_vs_reference) {
  Rng r(5151);
  int idx = 0;
  for (int64_t c : {12, 16, 37, 64, 128}) {
    for (int variant = 0; variant < 4; ++variant, ++idx) {
      const int64_t n = variant % 2 ? 2 : 1, h = variant == 3 ? 1 : 6, w = 7;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {n, h, w, c});
      // ShiftViT: four groups of c / 12 channels shifted by one pixel left / right / up / down; variant 2: shifts of 2..8
      const int64_t g = std::max<int64_t>(1, c / 12);
      const int64_t s = variant == 2 ? r.i(2, 8) : 1;
      std::vector<int64_t> gs = {0, g, 0, s, g, 2 * g, 0, -s, 2 * g, 3 * g, s, 0, 3 * g, 4 * g, -s, 0};
      if (variant == 1) gs = {c / 2, c / 2 + g, 1, -1};
      mb.node("I8ChannelShift", {"x"}, {"y"}, Json::Object{{"groups", ints(gs)}, {"fill_q", r.i(-128, 127)}});
      mb.output("y");
      auto x = r.i8(n * h * w * c);
      auto y = compare(mb, {x}, "I8ChannelShift config " + std::to_string(idx), "i8channelshift.neon");
      // definition spot check: the first group's pixel (0, s) reads x(0, 0)
      if (variant == 0 && g >= 1 && w > 1) {
        EXPECT_EQ(int(y[(0 * w + 1) * c + 0]), int(x[(0 * w + 0) * c + 0]));
      }
    }
  }
}

BF_TEST(int8_glue_binarize_dyn_bit_exact_vs_reference_and_definition) {
  Rng r(6161);
  int idx = 0;
  for (int64_t c : {16, 32, 48, 7, 70}) {
    for (int variant = 0; variant < 5; ++variant, ++idx) {
      const int64_t n = variant % 2 ? 2 : 1, h = variant == 4 ? 14 : 5, w = variant == 4 ? 14 : 3, hw = h * w;
      const int F = variant == 3 ? 0 : r.i(4, 20), G = variant == 3 ? 0 : r.i(0, 8);
      const int xz = variant == 2 ? 127 : r.i(-128, 127);
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {n, h, w, c});
      std::vector<int32_t> t0(static_cast<size_t>(c)), a(static_cast<size_t>(c)), b(static_cast<size_t>(c)), ko(static_cast<size_t>(c));
      const int64_t D = hw << F;
      for (int64_t i = 0; i < c; ++i) {
        t0[i] = static_cast<int32_t>(std::max<int64_t>(-2147483647, std::min<int64_t>(2147483647, int64_t(r.i(-60, 60)) * D)));
        a[i] = r.i(-(1 << 20), 1 << 20) >> r.i(0, 10);
        b[i] = r.i(-(1 << 20), 1 << 20) >> r.i(0, 10);
        ko[i] = r.i(-(1 << 22), 1 << 22);
      }
      mb.constant("t0", DType::kI32, {c}, t0.data());
      mb.constant("a", DType::kI32, {c}, a.data());
      mb.constant("b", DType::kI32, {c}, b.data());
      std::vector<std::string> ins = {"x", "t0", "a", "b"};
      const bool off = variant == 1;
      if (off) {
        mb.input("o", DType::kI8, {n, 1, 1, c});
        mb.constant("ko", DType::kI32, {c}, ko.data());
        ins.push_back("o");
        ins.push_back("ko");
      }
      const int oz = r.i(-20, 20);
      mb.node("I8BinarizeDyn", ins, {"y"}, Json::Object{{"x_zero", xz}, {"o_zero", oz}, {"shift", F}, {"sqrt_shift", G}});
      mb.output("y");
      auto x = r.i8(n * hw * c);
      if (variant == 4)  // a constant channel (V = 0) and a two-valued one
        for (int64_t j = 0; j < hw; ++j) {
          x[j * c + 0] = 5;
          x[j * c + 1] = j % 2 ? 0x80 : 0x7f;
        }
      std::vector<std::vector<uint8_t>> in = {x};
      std::vector<uint8_t> o;
      if (off) in.push_back(o = r.i8(n * c));
      auto y = compare(mb, in, "I8BinarizeDyn config " + std::to_string(idx), "i8binarizedyn.neon");
      // the definition, independently
      const int64_t words = (c + 31) / 32;
      for (int64_t s = 0; s < n; ++s)
        for (int64_t ch = 0; ch < c; ++ch) {
          int64_t s1 = 0, s2 = 0;
          for (int64_t j = 0; j < hw; ++j) {
            const int64_t v = int64_t(static_cast<int8_t>(x[(s * hw + j) * c + ch])) - xz;
            s1 += v;
            s2 += v * v;
          }
          const int64_t V = hw * s2 - s1 * s1;
          int64_t acc = int64_t(t0[ch]) + int64_t(a[ch]) * s1 + int64_t(b[ch]) * isqrt_def(V << (2 * G));
          if (off) acc += int64_t(ko[ch]) * (int64_t(static_cast<int8_t>(o[s * c + ch])) - oz);
          for (int64_t j = 0; j < hw; ++j) {
            const int64_t v = int64_t(static_cast<int8_t>(x[(s * hw + j) * c + ch])) - xz;
            const bool minus = !(v * D >= acc);
            uint32_t wd;
            std::memcpy(&wd, y.data() + ((s * hw + j) * words + ch / 32) * 4, 4);
            if (((wd >> (ch % 32)) & 1u) != (minus ? 1u : 0u)) throw bftest::Failure{"I8BinarizeDyn differs from its definition"};
          }
        }
      // bits beyond C are 0
      for (int64_t row = 0; row < n * hw; ++row) {
        uint32_t wd;
        std::memcpy(&wd, y.data() + (row * words + words - 1) * 4, 4);
        if (c % 32 && (wd >> (c % 32))) throw bftest::Failure{"I8BinarizeDyn: bits beyond C not 0"};
      }
    }
  }
}

BF_TEST(int8_glue_gated_add_full_and_lut_bit_exact_vs_reference) {
  Rng r(7272);
  int idx = 0;
  for (int64_t c : {16, 32, 40, 64, 7}) {
    for (int variant = 0; variant < 8; ++variant, ++idx) {
      const bool full = variant < 6, lut = variant % 2 == 1;
      const int64_t n = variant % 3 ? 2 : 1, h = 5, w = 7, nb = variant % 3;
      bftest::ModelBuilder mb;
      mb.input("x", DType::kI8, {n, h, w, c});
      mb.input("p", DType::kI8, {n, h, w, c});
      mb.input("g", DType::kI8, full ? Shape{n, h, w, c} : Shape{n, 1, 1, c});
      const int shift = variant == 5 ? 20 : (variant == 4 ? 0 : r.i(8, 16));
      std::vector<int32_t> kx(static_cast<size_t>(c)), kp(static_cast<size_t>(c)), b(static_cast<size_t>(c)),
          thr(static_cast<size_t>(std::max<int64_t>(nb, 1) * c));
      for (int64_t i = 0; i < c; ++i) {
        kx[i] = r.i(-20000, 20000);
        kp[i] = i % 7 == 2 ? 0 : r.i(-4000, 4000);
        b[i] = r.i(-(1 << 24), 1 << 24);
      }
      for (auto& t : thr) t = r.i(-(1 << 22), 1 << 22);
      thr[0] = b[0] - 3 * kx[0];
      mb.constant("kx", DType::kI32, {c}, kx.data());
      mb.constant("kp", DType::kI32, {c}, kp.data());
      mb.constant("b", DType::kI32, {c}, b.data());
      std::vector<std::string> ins = {"x", "p", "g", "kx", "kp", "b"}, outs = {"y"};
      if (nb) {
        mb.constant("t", DType::kI32, {nb, c}, thr.data());
        ins.push_back("t");
        for (int64_t j = 0; j < nb; ++j) outs.push_back("bits" + std::to_string(j));
      } else if (lut) {
        ins.push_back("");
      }
      std::vector<int8_t> tab(256);
      for (int k = 0; k < 256; ++k) tab[k] = static_cast<int8_t>(std::lround(255.0 / (1.0 + std::exp(-(k - 128) / 20.0))) - 128);
      if (lut) {
        mb.constant("lut", DType::kI8, {1, 256}, tab.data());
        ins.push_back("lut");
      }
      const int gz = lut ? -128 : r.i(-128, 127);
      mb.node("I8GatedAdd", ins, outs,
              Json::Object{{"x_zero", r.i(-128, 127)}, {"p_zero", variant == 3 ? 120 : r.i(-20, 20)}, {"g_zero", gz}, {"shift", shift},
                           {"y_zero", r.i(-100, 100)}, {"qmin", variant == 2 ? -90 : -128}, {"qmax", variant == 2 ? 100 : 127}});
      for (auto& o : outs) mb.output(o);
      auto gq = r.i8(full ? n * h * w * c : n * c);
      gq[0] = 0x80;
      compare(mb, {r.i8(n * h * w * c), r.i8(n * h * w * c), gq}, "I8GatedAdd full/lut config " + std::to_string(idx), "i8gatedadd.neon");
    }
  }
}

BF_TEST(int8_glue_simam_reference_hand_values) {
  // a 1x2x2x1 map {0, 10, 20, 40}, x_zero 0, identity table tab[k] = k - 128 (G = idx), mult 2^30 / shift -3:
  // S1 = 70, S2 = 2100, V = 3500, mu4 = 280, r = 16 / (64 * 3501); x = 40: dd = 360, dd^2 r = 9.26 -> idx 9,
  // y = rshr(mulhi(360, 2^30), 3) = rshr(180, 3) = 23; x = 0: y = 0
  bftest::ModelBuilder mb;
  mb.input("x", DType::kI8, {1, 2, 2, 1});
  std::vector<int8_t> tab(256);
  for (int k = 0; k < 256; ++k) tab[k] = static_cast<int8_t>(k - 128);
  mb.constant("tab", DType::kI8, {1, 256}, tab.data());
  mb.node("I8SimAM", {"x", "tab"}, {"y"},
          Json::Object{{"x_zero", 0}, {"lam_q", 1}, {"g_zero", -128}, {"mult", int64_t(1) << 30}, {"shift", -3}, {"y_zero", 0}});
  mb.output("y");
  auto y = run_all(mb, kRefTiers, 1, {{0, 10, 20, 40}});
  EXPECT_EQ(int(static_cast<int8_t>(y[0])), 0);
  EXPECT_EQ(int(static_cast<int8_t>(y[3])), 23);
  EXPECT_TRUE(run_all(mb, kFullTiers, 1, {{0, 10, 20, 40}}) == y);
}
