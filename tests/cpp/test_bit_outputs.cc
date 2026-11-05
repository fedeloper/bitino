// Bit-packed outputs of binary layers (kernels/bit_output.h, plan-time rewrite
// bit_outputs in src/graph_rewrite.cc):
//   * every BConv2D / BDense kernel writing "output": "bit" (ge / gt, with and
//     without an output threshold) stores exactly pack_row_f32() of the float
//     output the same node computes with "output": "f32" (NaN, +-inf, +-0
//     included), reference vs binary_scalar / NEON / AVX2, several thread counts;
//   * the rewrite is bit-exact end to end: graphs where a binary layer's
//     (epilogue-fused, threshold-folded) output only feeds the ge / gt
//     binarization of the next binary layer give byte-identical outputs with
//     and without the rewrite and against the unfused reference graph, and the
//     rewrite is skipped where it would not be exact (sign_zero consumer,
//     several uses, graph outputs, a threshold computed after the producer).
#include <cmath>
#include <cstring>
#include <random>

#include "bitino/bitino.h"
#include "kernels/bit_output.h"
#include "kernels/bitpack.h"
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

std::vector<uint32_t> rand_bits(int64_t rows, int64_t c, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<int8_t> w(static_cast<size_t>(rows * c));
  for (auto& v : w) v = rng() & 1 ? 1 : -1;
  return bftest::pack_pm1(w, rows, c);
}

// Input with NaN, +-inf, +-0 and ordinary values.
std::vector<float> nasty(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) {
    switch (rng() % 13) {
      case 0: x = 0.f; break;
      case 1: x = -0.f; break;
      case 2: x = NAN; break;
      case 3: x = INFINITY; break;
      case 4: x = -INFINITY; break;
      default: x = nd(rng);
    }
  }
  return v;
}

// Post-scales including 0, -0, inf and negative values (signed zeros, NaN = inf * 0 outputs).
std::vector<float> nasty_mul(int64_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> v(static_cast<size_t>(n));
  for (size_t i = 0; i < v.size(); ++i) {
    switch (i % 9) {
      case 0: v[i] = 0.f; break;
      case 1: v[i] = -0.f; break;
      case 2: v[i] = INFINITY; break;
      case 3: v[i] = -0.05f; break;
      default: v[i] = 0.01f + std::fabs(nd(rng)) * 0.1f;
    }
  }
  return v;
}

bool available(const std::string& name) {
  for (const auto& k : OpRegistry::global().all_kernels())
    if (k.name == name) return cpu_info().has(k.required_features);
  return false;
}

struct Run {
  std::vector<uint8_t> y;
  std::map<std::string, std::string> kernels;
  int bit_outputs = 0, hosted = 0, folded = 0;
};

Run run(const std::shared_ptr<Model>& m, const KernelPolicy& p, const std::vector<std::vector<float>>& in, int threads, bool bit_outputs = true,
        bool fuse = true) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = p;
  o.bit_outputs = bit_outputs;
  o.fuse = fuse;
  Session s(m, o);
  for (size_t i = 0; i < in.size(); ++i) std::memcpy(s.input(i).raw(), in[i].data(), in[i].size() * 4);
  s.run();
  s.run();  // second run: per-run state (thresholds, packing, epilogue operands) is refreshed
  Run r;
  const Tensor& y = s.output(0);
  r.y.assign(static_cast<const uint8_t*>(y.raw()), static_cast<const uint8_t*>(y.raw()) + y.bytes());
  r.kernels = s.kernel_choices();
  r.bit_outputs = s.bit_outputs();
  r.hosted = s.hosted_epilogues();
  r.folded = s.folded_thresholds();
  return r;
}

KernelPolicy tiers(std::set<std::string> t) { return KernelPolicy::tiers(std::move(t)); }
KernelPolicy forced(const std::string& op, const std::string& k) {
  KernelPolicy p;
  p.force[op] = k;
  return p;
}

const std::set<std::string> kRef = {"reference"};
const std::set<std::string> kOpt = {"reference", "binary_scalar", "neon", "binary_avx2", "float_opt"};

void expect_same(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const std::string& what) {
  if (a == b) return;
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
  throw bftest::Failure{what + ": outputs differ (sizes " + std::to_string(a.size()) + "/" + std::to_string(b.size()) + ", first byte " +
                        std::to_string(i) + ")"};
}

// Expected bit output: pack_row_f32 of the f32 output rows.
std::vector<uint8_t> pack_expected(const std::vector<uint8_t>& f32_bytes, int64_t c, BinarizeMode mode, const float* thr) {
  const int64_t rows = static_cast<int64_t>(f32_bytes.size() / 4) / c, words = bit_words(c);
  std::vector<uint32_t> bits(static_cast<size_t>(rows * words));
  const auto* f = reinterpret_cast<const float*>(f32_bytes.data());
  for (int64_t r = 0; r < rows; ++r) pack_row_f32(f + r * c, c, bits.data() + r * words, mode, thr);
  std::vector<uint8_t> out(bits.size() * 4);
  std::memcpy(out.data(), bits.data(), out.size());
  return out;
}

}  // namespace

// ---------------------------------------------------------------- integer thresholds
BF_TEST(bit_output_integer_thresholds_exact) {
  // IntThresholds (kernels/bit_output.h) must reproduce bit(d) = !(act(d*mul + add) >= / > t) for every
  // reachable dot product d, or report that it cannot (then the kernels keep the float compare).
  const int64_t k = 300, oc = 64;
  int fallbacks = 0, built = 0;
  for (int act = 0; act < 3; ++act)
    for (BinarizeMode mode : {BinarizeMode::kGe, BinarizeMode::kGt}) {
      auto mul = nasty_mul(oc, 11 + act), add = randn(oc, 12 + act), thr = randn(oc, 13 + act, 2.f);
      for (int64_t o = 0; o < oc; o += 5) thr[o] = static_cast<float>(o % 3) - 1.f;  // ties with small integers
      add[7] = INFINITY;
      add[8] = -INFINITY;
      thr[9] = NAN;
      IntThresholds it;
      const Activation a = act == 0 ? Activation::kNone : act == 1 ? Activation::kRelu : Activation::kRelu6;
      if (!it.update(k, oc, mul.data(), add.data(), a, mode, thr.data())) {
        ++fallbacks;  // e.g. relu6(inf * 0 = NaN) between two 0-bits: two flips
        continue;
      }
      ++built;
      for (int64_t o = 0; o < oc; ++o)
        for (int64_t d = -k; d <= k; ++d) {
          float v = static_cast<float>(d) * mul[o];
          v = v + add[o];
          const bool want = output_bit(apply_activation(a, v), mode, thr[o]);
          const bool got = it.neg()[o] ? d > it.lim()[o] : d < it.lim()[o];
          if (want != got) throw bftest::Failure{"integer threshold mismatch o=" + std::to_string(o) + " d=" + std::to_string(d)};
        }
      // Cached: same parameters -> no rebuild, same answer.
      EXPECT_TRUE(it.update(k, oc, mul.data(), add.data(), a, mode, thr.data()));
    }
  EXPECT_TRUE(built > 0);
  // A NaN in the middle of the d range (inf * 0 at d = 0) between two equal bits cannot be one threshold.
  std::vector<float> mul = {INFINITY}, add = {0.f}, thr = {-0.5f};
  IntThresholds it;
  EXPECT_TRUE(!it.update(10, 1, mul.data(), add.data(), Activation::kRelu6, BinarizeMode::kGe, thr.data()));
  (void)fallbacks;
}

// ---------------------------------------------------------------- kernels
BF_TEST(bit_output_kernels_equal_packed_float_output) {
  struct Case {
    bool dense;
    Shape x;         // input (NHWC for BConv2D, [..., K] for BDense)
    int64_t oc, k;   // output channels, kernel size (BConv2D)
    int64_t groups;
    bool gt, thr;
    std::string act;
  };
  const std::vector<Case> cases = {
      {false, {1, 9, 11, 64}, 64, 3, 1, false, true, "none"},    // pair mode, word-aligned oc
      {false, {2, 7, 6, 96}, 40, 3, 1, true, true, "relu"},      // oc tail inside a word, batch 2
      {false, {1, 7, 7, 256}, 256, 3, 1, false, false, "none"},  // few pixel tiles: channel chunks with threads
      {false, {1, 6, 5, 64}, 36, 3, 4, true, true, "relu6"},     // grouped, ocg % 4 != 0 (read-modify-write words)
      {false, {1, 5, 5, 48}, 72, 1, 3, false, true, "none"},     // grouped 1x1, kic 16
      {true, {1, 13, 64}, 96, 1, 1, false, true, "none"},        // tiled rows, pair mode
      {true, {3, 100}, 33, 1, 1, true, true, "relu"},            // GEMV-like (rows < 8), K and O tails
      {true, {2, 9, 256}, 520, 1, 1, false, false, "none"},      // tiled, O > 512
      {true, {1, 197, 192}, 768, 1, 1, true, true, "relu6"},     // Bi-ViT fc1 shape
  };
  const std::vector<std::string> conv_kernels = {"bconv2d.reference", "bconv2d.packed_naive", "bconv2d.binary_scalar", "bconv2d.neon",
                                                 "bconv2d.neon_opt", "bconv2d.avx2"};
  const std::vector<std::string> dense_kernels = {"bdense.reference", "bdense.binary_scalar", "bdense.neon", "bdense.avx2"};
  uint32_t seed = 4100;
  int checked = 0;
  for (const Case& c : cases) {
    const int64_t ic = c.x.back(), kic = ic / c.groups;
    auto w = c.dense ? rand_bits(c.oc, ic, seed++) : rand_bits(c.oc * c.k * c.k, kic, seed++);
    auto mul = nasty_mul(c.oc, seed++), add = randn(c.oc, seed++), othr = randn(c.oc, seed++, 0.5f);
    othr[0] = 0.f;
    othr[1] = -0.f;
    auto build = [&](bool bit) {
      bftest::ModelBuilder mb;
      if (c.dense) mb.constant("w", DType::kBit, {c.oc, ic}, w.data());
      else mb.constant("w", DType::kBit, {c.oc, c.k, c.k, kic}, w.data(), "OHWI_bit");
      mb.constant("mul", DType::kF32, {c.oc}, mul.data());
      mb.constant("add", DType::kF32, {c.oc}, add.data());
      mb.constant("othr", DType::kF32, {c.oc}, othr.data());
      mb.input("x", DType::kF32, c.x);
      Json a = Json::Object{{"activation", c.act}};
      if (!c.dense) {
        const int64_t p = c.k / 2;
        a["pads"] = ints({p, p, p, p});
        a["groups"] = Json(c.groups);
      }
      std::vector<std::string> ins = {"x", "w", "mul", "add", ""};
      if (c.dense) ins.push_back("");
      if (bit) {
        a["output"] = "bit";
        a["output_binarize"] = c.gt ? "gt" : "ge";
        if (c.thr) {
          a["output_threshold_input"] = Json(static_cast<int64_t>(ins.size()));
          ins.push_back("othr");
        }
      }
      mb.node(c.dense ? "BDense" : "BConv2D", ins, {"y"}, a);
      mb.output("y");
      return mb.build();
    };
    auto mf = build(false), mb = build(true);
    std::vector<std::vector<float>> in = {nasty(static_cast<size_t>(num_elements(c.x)), seed++)};
    const Run f32 = run(mf, tiers(kRef), in, 1);
    const auto expect = pack_expected(f32.y, c.oc, c.gt ? BinarizeMode::kGt : BinarizeMode::kGe, c.thr ? othr.data() : nullptr);
    for (const auto& kn : c.dense ? dense_kernels : conv_kernels) {
      if (!available(kn)) continue;
      for (int threads : {1, 3, 4}) {
        Run r;
        try {
          r = run(mb, forced(c.dense ? "BDense" : "BConv2D", kn), in, threads);
        } catch (const Error&) {
          continue;  // kernel rejects this geometry
        }
        expect_same(r.y, expect, "bit output " + kn + " t" + std::to_string(threads) + " " + shape_str(c.x));
        // The same kernel with f32 output equals the reference float output (no hidden drift).
        expect_same(run(mf, forced(c.dense ? "BDense" : "BConv2D", kn), in, threads).y, f32.y, "f32 output " + kn);
        ++checked;
      }
    }
  }
  EXPECT_TRUE(checked > 0);
}

BF_TEST(bconv_integer_bit_output_step_classes) {
  // BConv2D bit output through integer thresholds (finite non-zero post-scales
  // of both signs, so IntThresholds exist): bconv2d.neon_opt binarizes full
  // micro-tiles 4 rows at a time, with the border correction per tile; sweep
  // step-count classes in normal and pair mode, pad values -1 / 0 / +1, pixel
  // tails, blocks starting inside a word (oc 40, grouped ocg 20 / 32).
  struct Case {
    int64_t h, w, ic, oc, k, groups;
    int pad_value;
    bool gt;
    std::string act;
  };
  const std::vector<Case> cases = {
      {7, 9, 32, 64, 3, 1, 0, false, "none"},     // pair, 5 steps
      {9, 7, 64, 40, 3, 1, -1, true, "relu"},     // pair, 9 steps, oc tail inside a word
      {6, 11, 96, 64, 3, 1, 1, false, "none"},    // 7 steps
      {11, 13, 128, 96, 3, 1, 0, true, "relu6"},  // 9 steps, several pixel tiles
      {5, 9, 224, 32, 3, 1, -1, false, "none"},   // 16 steps
      {7, 7, 160, 72, 3, 1, 0, false, "none"},    // pair, 23 steps
      {5, 6, 512, 64, 3, 1, -1, true, "relu"},    // 36 steps: two runs
      {6, 7, 448, 36, 3, 1, 0, false, "none"},    // pair, 63 steps, oc tail
      {9, 9, 128, 64, 1, 1, 0, false, "none"},    // 1x1, 1 step
      {9, 9, 64, 64, 1, 1, 0, true, "none"},      // 1x1 pair, 1 step
      {8, 8, 80, 40, 3, 2, 0, false, "none"},     // grouped kic 40 (repack), ocg 20: blocks at bit 20
      {8, 9, 128, 64, 3, 2, -1, true, "relu"},    // grouped kic 64, ocg 32
  };
  const std::vector<std::string> conv_kernels = {"bconv2d.reference", "bconv2d.binary_scalar", "bconv2d.neon_opt", "bconv2d.avx2"};
  uint32_t seed = 5200;
  int checked = 0;
  for (const Case& c : cases) {
    const int64_t kic = c.ic / c.groups;
    auto w = rand_bits(c.oc * c.k * c.k, kic, seed++);
    auto mul = randn(c.oc, seed++, 0.1f), add = randn(c.oc, seed++), othr = randn(c.oc, seed++, 0.5f);
    for (size_t o = 0; o < mul.size(); ++o)
      if (mul[o] == 0.f) mul[o] = 0.01f;
    auto build = [&](bool bit) {
      bftest::ModelBuilder mb;
      mb.constant("w", DType::kBit, {c.oc, c.k, c.k, kic}, w.data(), "OHWI_bit");
      mb.constant("mul", DType::kF32, {c.oc}, mul.data());
      mb.constant("add", DType::kF32, {c.oc}, add.data());
      mb.constant("othr", DType::kF32, {c.oc}, othr.data());
      mb.input("x", DType::kF32, {1, c.h, c.w, c.ic});
      const int64_t p = c.k / 2;
      Json a = Json::Object{{"activation", c.act}, {"pad_value", Json(c.pad_value)}};
      a["pads"] = ints({p, p, p, p});
      a["groups"] = Json(c.groups);
      std::vector<std::string> ins = {"x", "w", "mul", "add", ""};
      if (bit) {
        a["output"] = "bit";
        a["output_binarize"] = c.gt ? "gt" : "ge";
        a["output_threshold_input"] = Json(static_cast<int64_t>(ins.size()));
        ins.push_back("othr");
      }
      mb.node("BConv2D", ins, {"y"}, a);
      mb.output("y");
      return mb.build();
    };
    auto mf = build(false), mb = build(true);
    std::vector<std::vector<float>> in = {randn(static_cast<size_t>(c.h * c.w * c.ic), seed++)};
    const Run f32 = run(mf, tiers(kRef), in, 1);
    const auto expect = pack_expected(f32.y, c.oc, c.gt ? BinarizeMode::kGt : BinarizeMode::kGe, othr.data());
    for (const auto& kn : conv_kernels) {
      if (!available(kn)) continue;
      for (int threads : {1, 3}) {
        expect_same(run(mb, forced("BConv2D", kn), in, threads).y, expect,
                    "integer bit output " + kn + " t" + std::to_string(threads) + " ic " + std::to_string(c.ic));
        ++checked;
      }
    }
  }
  EXPECT_TRUE(checked > 0);
}

BF_TEST(bdense_bit_input_equals_packed_float_input) {
  // A BDense reading bits equals the same BDense binarizing the float tensor
  // the bits were packed from (ge / gt with thresholds), for every kernel.
  uint32_t seed = 4300;
  for (const Shape& xs : {Shape{1, 13, 64}, Shape{3, 100}, Shape{2, 9, 256}}) {
    const int64_t k = xs.back(), o = 24;
    for (bool gt : {false, true}) {
      auto w = rand_bits(o, k, seed++);
      auto thr = randn(k, seed++, 0.3f);
      auto x = nasty(static_cast<size_t>(num_elements(xs)), seed++);
      const int64_t rows = num_elements(xs) / k, words = bit_words(k);
      std::vector<uint32_t> bits(static_cast<size_t>(rows * words));
      for (int64_t r = 0; r < rows; ++r) pack_row_f32(x.data() + r * k, k, bits.data() + r * words, gt ? BinarizeMode::kGt : BinarizeMode::kGe, thr.data());
      bftest::ModelBuilder ma, mbits;
      for (auto* mb : {&ma, &mbits}) {
        mb->constant("w", DType::kBit, {o, k}, w.data());
        mb->constant("thr", DType::kF32, {k}, thr.data());
      }
      ma.input("x", DType::kF32, xs);
      ma.node("BDense", {"x", "w", "", "", "thr"}, {"y"}, Json(Json::Object{{"input_binarize", gt ? "gt" : "ge"}}));
      ma.output("y");
      mbits.constant("xb", DType::kBit, xs, bits.data());
      mbits.node("BDense", {"xb", "w"}, {"y"});
      mbits.output("y");
      auto mf = ma.build(), mbm = mbits.build();
      const Run ref = run(mf, tiers(kRef), {x}, 1);
      for (const std::string& kn : {"bdense.reference", "bdense.binary_scalar", "bdense.neon", "bdense.avx2"}) {
        if (!available(kn)) continue;
        for (int threads : {1, 4}) expect_same(run(mbm, forced("BDense", kn), {}, threads).y, ref.y, "bit input " + kn);
      }
    }
  }
}

// ---------------------------------------------------------------- rewrite
BF_TEST(bit_output_rewrite_bit_exact) {
  // Producer -> (epilogue tree / bias fold) -> consumer that binarizes with
  // ge / gt. Every variant: the rewrite fires (count checked), the result is
  // byte-identical to the same session without the rewrite, to the unfused
  // reference graph, and across tier sets and thread counts.
  struct Case {
    const char* name;
    int expect_bit;     // bit_outputs expected at kOpt
    int expect_hosted;  // hosted epilogues expected at kOpt
    std::function<std::shared_ptr<Model>(uint32_t)> build;
    std::vector<Shape> inputs;
  };
  auto conv_attrs = [](int64_t p, const char* mode, int64_t groups = 1) {
    return Json(Json::Object{{"pads", ints({p, p, p, p})}, {"input_binarize", mode}, {"groups", Json(groups)}});
  };
  std::vector<Case> cases;
  // 1. BConv2D -> BConv2D (ge, threshold): direct bit store with an output threshold.
  cases.push_back({"conv->conv ge thr", 1, 0, [&](uint32_t s) {
                     const int64_t c = 64, o = 40;
                     auto w1 = rand_bits(c * 9, c, s), w2 = rand_bits(o * 9, c, s + 1);
                     auto m1 = nasty_mul(c, s + 2), a1 = randn(c, s + 3), t2 = randn(c, s + 4, 0.5f), m2 = randn(o, s + 5, 0.1f);
                     bftest::ModelBuilder mb;
                     mb.constant("w1", DType::kBit, {c, 3, 3, c}, w1.data());
                     mb.constant("w2", DType::kBit, {o, 3, 3, c}, w2.data());
                     mb.constant("m1", DType::kF32, {c}, m1.data());
                     mb.constant("a1", DType::kF32, {c}, a1.data());
                     mb.constant("t2", DType::kF32, {c}, t2.data());
                     mb.constant("m2", DType::kF32, {o}, m2.data());
                     mb.input("x", DType::kF32, {1, 9, 8, c});
                     mb.node("BConv2D", {"x", "w1", "m1", "a1"}, {"h"}, conv_attrs(1, "ge"));
                     mb.node("BConv2D", {"h", "w2", "m2", "", "t2"}, {"y"}, conv_attrs(1, "ge"));
                     mb.output("y");
                     return mb.build();
                   },
                   {{1, 9, 8, 64}}});
  // 2. BConv2D -> prelu -> affine -> + residual -> BConv2D (gt): epilogue + bit output; few tiles x 256 channels.
  for (int64_t hw : {9, 7}) {
    cases.push_back({hw == 9 ? "conv+epilogue->conv gt" : "conv+epilogue->conv gt (channel chunks)", 1, 1, [=](uint32_t s) {
                       const int64_t c = hw == 9 ? 64 : 256, o = 32;
                       auto w1 = rand_bits(c * 9, c, s), w2 = rand_bits(o * 9, c, s + 1);
                       auto m1 = randn(c, s + 2, 0.05f), sl = randn(c, s + 3, 0.3f), sc = randn(c, s + 4), bi = randn(c, s + 5);
                       auto m2 = randn(o, s + 6, 0.1f);
                       bftest::ModelBuilder mb;
                       mb.constant("w1", DType::kBit, {c, 3, 3, c}, w1.data());
                       mb.constant("w2", DType::kBit, {o, 3, 3, c}, w2.data());
                       mb.constant("m1", DType::kF32, {c}, m1.data());
                       mb.constant("sl", DType::kF32, {c}, sl.data());
                       mb.constant("sc", DType::kF32, {c}, sc.data());
                       mb.constant("bi", DType::kF32, {c}, bi.data());
                       mb.constant("m2", DType::kF32, {o}, m2.data());
                       mb.input("x", DType::kF32, {1, hw, hw, c});
                       mb.input("r", DType::kF32, {1, hw, hw, c});
                       mb.node("Activation", {"r"}, {"res"}, Json(Json::Object{{"type", "relu"}}));
                       mb.node("BConv2D", {"x", "w1", "m1"}, {"h"}, Json(Json::Object{{"pads", ints({1, 1, 1, 1})}}));
                       mb.node("Activation", {"h", "sl"}, {"p"}, Json(Json::Object{{"type", "prelu"}}));
                       mb.node("ChannelAffine", {"p", "sc", "bi"}, {"e"});
                       mb.node("Add", {"e", "res"}, {"f"});
                       mb.node("BConv2D", {"f", "w2", "m2"}, {"y"}, conv_attrs(1, "gt"));
                       mb.output("y");
                       return mb.build();
                     },
                     {{1, hw, hw, hw == 9 ? 64 : 256}, {1, hw, hw, hw == 9 ? 64 : 256}}});
  }
  // 3. BConv2D -> ChannelAffine(bias) -> grouped BConv2D (ge): bias folded into the consumer's threshold,
  //    which becomes the producer's output threshold; the grouped consumer reads bits (regroup, kic 24).
  cases.push_back({"conv->bias fold->grouped conv", 1, 0, [&](uint32_t s) {
                     const int64_t c = 72, o = 48, g = 3;
                     auto w1 = rand_bits(c * 9, 32, s), w2 = rand_bits(o * 9, c / g, s + 1);
                     auto m1 = randn(c, s + 2, 0.1f), bias = randn(c, s + 3), m2 = randn(o, s + 4, 0.1f);
                     bias[5] = 0.f;
                     bftest::ModelBuilder mb;
                     mb.constant("w1", DType::kBit, {c, 3, 3, 32}, w1.data());
                     mb.constant("w2", DType::kBit, {o, 3, 3, c / g}, w2.data());
                     mb.constant("m1", DType::kF32, {c}, m1.data());
                     mb.constant("bias", DType::kF32, {c}, bias.data());
                     mb.constant("m2", DType::kF32, {o}, m2.data());
                     mb.input("x", DType::kF32, {2, 6, 7, 32});
                     mb.node("BConv2D", {"x", "w1", "m1"}, {"h"}, Json(Json::Object{{"pads", ints({1, 1, 1, 1})}}));
                     mb.node("ChannelAffine", {"h", "", "bias"}, {"t"});
                     mb.node("BConv2D", {"t", "w2", "m2"}, {"y"}, conv_attrs(1, "ge", g));
                     mb.output("y");
                     return mb.build();
                   },
                   {{2, 6, 7, 32}}});
  // 4. BDense chain (Bi-ViT-like rows): fc1 -> affine -> fc2 (ge, thr) -> fc3 (gt): two bit outputs, one with epilogue.
  for (int64_t rows : {197, 5}) {
    cases.push_back({rows > 8 ? "dense chain (tiled)" : "dense chain (GEMV)", 2, 1, [=](uint32_t s) {
                       const int64_t k = 96, h = 160, o = 40;
                       auto w1 = rand_bits(h, k, s), w2 = rand_bits(h, h, s + 1), w3 = rand_bits(o, h, s + 2);
                       auto m1 = nasty_mul(h, s + 3), sc = randn(h, s + 4), bi = randn(h, s + 5), t2 = randn(h, s + 6, 0.5f);
                       auto m2 = randn(h, s + 7, 0.1f), a2 = randn(h, s + 8);
                       bftest::ModelBuilder mb;
                       mb.constant("w1", DType::kBit, {h, k}, w1.data());
                       mb.constant("w2", DType::kBit, {h, h}, w2.data());
                       mb.constant("w3", DType::kBit, {o, h}, w3.data());
                       mb.constant("m1", DType::kF32, {h}, m1.data());
                       mb.constant("sc", DType::kF32, {h}, sc.data());
                       mb.constant("bi", DType::kF32, {h}, bi.data());
                       mb.constant("t2", DType::kF32, {h}, t2.data());
                       mb.constant("m2", DType::kF32, {h}, m2.data());
                       mb.constant("a2", DType::kF32, {h}, a2.data());
                       mb.input("x", DType::kF32, {1, rows, k});
                       mb.node("BDense", {"x", "w1", "m1"}, {"h1"});
                       mb.node("ChannelAffine", {"h1", "sc", "bi"}, {"e"});
                       mb.node("BDense", {"e", "w2", "m2", "a2", "t2"}, {"h2"}, Json(Json::Object{{"activation", "relu"}}));
                       mb.node("BDense", {"h2", "w3"}, {"y"}, Json(Json::Object{{"input_binarize", "gt"}}));
                       mb.output("y");
                       return mb.build();
                     },
                     {{1, rows, 96}}});
  }
  // 5. BDense -> BConv2D in NHWC (1x1 spatial positions as rows): dense producer, conv consumer.
  cases.push_back({"dense->conv", 1, 0, [&](uint32_t s) {
                     const int64_t c = 48, o = 24;
                     auto w1 = rand_bits(c, c, s), w2 = rand_bits(o * 9, c, s + 1);
                     auto m1 = randn(c, s + 2, 0.1f), a1 = randn(c, s + 3), m2 = randn(o, s + 4, 0.1f);
                     bftest::ModelBuilder mb;
                     mb.constant("w1", DType::kBit, {c, c}, w1.data());
                     mb.constant("w2", DType::kBit, {o, 3, 3, c}, w2.data());
                     mb.constant("m1", DType::kF32, {c}, m1.data());
                     mb.constant("a1", DType::kF32, {c}, a1.data());
                     mb.constant("m2", DType::kF32, {o}, m2.data());
                     mb.input("x", DType::kF32, {1, 6, 6, c});
                     mb.node("BDense", {"x", "w1", "m1", "a1"}, {"h"});
                     mb.node("BConv2D", {"h", "w2", "m2"}, {"y"}, conv_attrs(1, "ge"));
                     mb.output("y");
                     return mb.build();
                   },
                   {{1, 6, 6, 48}}});
  uint32_t seed = 5000;
  for (const Case& c : cases) {
    auto m = c.build(seed);
    seed += 100;
    std::vector<std::vector<float>> in;
    for (const Shape& s : c.inputs) in.push_back(nasty(static_cast<size_t>(num_elements(s)), seed++));
    const Run ref = run(m, tiers(kRef), in, 1);  // no float_opt: no plan-time rewrite at all
    EXPECT_EQ(ref.bit_outputs, 0);
    std::vector<std::set<std::string>> tier_sets = {{"reference", "float_opt"}, {"reference", "binary_scalar", "float_opt"}, kOpt};
    for (const auto& ts : tier_sets)
      for (int threads : {1, 4}) {
        const Run on = run(m, tiers(ts), in, threads, true);
        const Run off = run(m, tiers(ts), in, threads, false);
        const std::string what = std::string(c.name) + " t" + std::to_string(threads) + " tiers " + std::to_string(ts.size());
        EXPECT_EQ(on.bit_outputs, c.expect_bit);
        EXPECT_EQ(off.bit_outputs, 0);
        if (ts == kOpt) EXPECT_EQ(on.hosted, c.expect_hosted);
        expect_same(on.y, off.y, what + " (with vs without rewrite)");
        expect_same(on.y, ref.y, what + " (vs unfused reference)");
      }
  }
}

BF_TEST(bit_output_rewrite_skips_inexact_cases) {
  const int64_t c = 32, o = 16;
  auto w1 = rand_bits(c, c, 1), w2 = rand_bits(o, c, 2);
  auto m1 = randn(c, 3, 0.1f), t = randn(c, 4);
  auto base = [&](bftest::ModelBuilder& mb) {
    mb.constant("w1", DType::kBit, {c, c}, w1.data());
    mb.constant("w2", DType::kBit, {o, c}, w2.data());
    mb.constant("m1", DType::kF32, {c}, m1.data());
    mb.constant("t", DType::kF32, {c}, t.data());
    mb.input("x", DType::kF32, {9, c});
  };
  auto count = [&](const std::shared_ptr<Model>& m, KernelPolicy p = tiers(kOpt)) { return run(m, p, {nasty(9 * c, 7)}, 2).bit_outputs; };
  {  // sign_zero consumer: three values, not representable by one bit
    bftest::ModelBuilder mb;
    base(mb);
    mb.node("BDense", {"x", "w1", "m1"}, {"h"});
    mb.node("BDense", {"h", "w2"}, {"y"}, Json(Json::Object{{"input_binarize", "sign_zero"}}));
    mb.output("y");
    EXPECT_EQ(count(mb.build()), 0);
  }
  {  // two uses of the producer's output
    bftest::ModelBuilder mb;
    base(mb);
    mb.node("BDense", {"x", "w1", "m1"}, {"h"});
    mb.node("BDense", {"h", "w2"}, {"y1"});
    mb.node("BDense", {"h", "w2"}, {"y2"});
    mb.node("Add", {"y1", "y2"}, {"y"});
    mb.output("y");
    EXPECT_EQ(count(mb.build()), 0);
  }
  {  // the producer's output is a graph output
    bftest::ModelBuilder mb;
    base(mb);
    mb.node("BDense", {"x", "w1", "m1"}, {"h"});
    mb.node("BDense", {"h", "w2"}, {"y"});
    mb.output("y");
    mb.output("h");
    EXPECT_EQ(count(mb.build()), 0);
  }
  {  // sign_zero producer input (ternary kernels write floats only)
    bftest::ModelBuilder mb;
    base(mb);
    mb.node("BDense", {"x", "w1", "m1"}, {"h"}, Json(Json::Object{{"input_binarize", "sign_zero"}}));
    mb.node("BDense", {"h", "w2"}, {"y"});
    mb.output("y");
    EXPECT_EQ(count(mb.build()), 0);
  }
  {  // consumer threshold computed after the producer runs
    bftest::ModelBuilder mb;
    base(mb);
    mb.input("u", DType::kF32, {c});
    mb.node("BDense", {"x", "w1", "m1"}, {"h"});
    mb.node("Activation", {"u"}, {"tu"}, Json(Json::Object{{"type", "relu"}}));
    mb.node("BDense", {"h", "w2", "", "", "tu"}, {"y"});
    mb.output("y");
    SessionOptions so;
    so.policy = tiers(kOpt);
    Session s(mb.build(), so);
    EXPECT_EQ(s.bit_outputs(), 0);
  }
  {  // positive control, then a forced BDense kernel disables the rewrite
    bftest::ModelBuilder mb;
    base(mb);
    mb.node("BDense", {"x", "w1", "m1"}, {"h"});
    mb.node("BDense", {"h", "w2", "", "", "t"}, {"y"});
    mb.output("y");
    auto m = mb.build();
    EXPECT_EQ(count(m), 1);
    EXPECT_EQ(count(m, forced("BDense", "bdense.reference")), 0);
  }
}
