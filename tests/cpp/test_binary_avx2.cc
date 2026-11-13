// binary_avx2 tier and grouped packed BConv2D:
//  (a) randomized differential test: every packed BConv2D kernel the CPU can
//      run (incl. bconv2d.avx2 / bconv2d.neon_opt) vs the reference over random
//      geometry, groups, pad values, thresholds, output types and threads;
//  (b) kernel selection follows CPU features: BITINO_DISABLE_FEATURES is read
//      once per process, so masked runs re-execute this binary (as in
//      test_int8_kernels.cc) and check the fallbacks and their results.
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <random>

#include "bitino/bitino.h"
#include "testing.h"
#include "tools/model_builder.h"

extern char** environ;

using namespace bitino;

namespace {

struct RandConv {
  int64_t n, ih, iw, ic, oc, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, groups;
  int pad_value;
  bool bit_out, thr, gt;
  std::string act;
};

std::string describe(const RandConv& c) {
  std::ostringstream os;
  os << "n" << c.n << " " << c.ih << "x" << c.iw << "x" << c.ic << " -> " << c.oc << " k" << c.kh << "x" << c.kw << " s" << c.sh << c.sw
     << " d" << c.dh << c.dw << " p" << c.pt << c.pb << c.pl << c.pr << " g" << c.groups << " pv" << c.pad_value
     << (c.bit_out ? " bit" : " f32") << (c.thr ? " thr" : "") << (c.gt ? " gt" : " ge") << " " << c.act;
  return os.str();
}

std::shared_ptr<Model> make_conv(const RandConv& c, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int64_t kic = c.ic / c.groups;
  std::vector<int8_t> w(static_cast<size_t>(c.oc * c.kh * c.kw * kic));
  for (auto& v : w) v = rng() & 1 ? 1 : -1;
  auto wp = bftest::pack_pm1(w, c.oc * c.kh * c.kw, kic);
  std::vector<float> mul(c.oc), add(c.oc), thr(c.ic);
  for (size_t i = 0; i < mul.size(); ++i) mul[i] = (i % 5 == 0 ? -0.f : (i % 3 == 0 ? -1.f : 1.f)) * (0.01f + std::fabs(nd(rng)) * 0.1f);
  for (size_t i = 0; i < add.size(); ++i) add[i] = i % 7 == 0 ? -0.f : nd(rng);
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
                            {"input_binarize", c.gt ? "gt" : "ge"},
                            {"output", c.bit_out ? "bit" : "f32"},
                            {"activation", c.act},
                            {"groups", Json(c.groups)}};
  mb.node("BConv2D", {"x", "w", "mul", "add", c.thr ? "thr" : ""}, {"y"}, attrs);
  mb.output("y");
  return mb.build();
}

// Inputs with exact zeros, values equal to the threshold-free decision point,
// NaN and infinities next to normal values.
void fill(Tensor& t, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  float* p = t.data<float>();
  for (int64_t i = 0; i < t.size(); ++i) {
    switch (rng() % 13) {
      case 0: p[i] = 0.f; break;
      case 1: p[i] = -0.f; break;
      case 2: p[i] = NAN; break;
      case 3: p[i] = INFINITY; break;
      default: p[i] = nd(rng);
    }
  }
}

struct RunOut {
  std::vector<std::vector<uint8_t>> y;
  std::string kernel;
};

// Two runs per session with different inputs (stale scratch cannot hide).
RunOut run(const std::shared_ptr<Model>& m, const std::set<std::string>& tiers, int threads, const std::string& force = "") {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers(tiers);
  if (!force.empty()) o.policy.force["BConv2D"] = force;
  Session s(m, o);
  RunOut r;
  for (uint32_t seed : {11u, 12u}) {
    fill(s.input(0), seed);
    s.run();
    const Tensor& y = s.output(0);
    r.y.emplace_back(y.bytes());
    std::memcpy(r.y.back().data(), y.raw(), r.y.back().size());
  }
  r.kernel = s.kernel_choices().begin()->second;
  return r;
}

bool available(const std::string& name) {
  for (const auto& k : OpRegistry::global().all_kernels())
    if (k.name == name) return cpu_info().has(k.required_features);
  return false;
}

RandConv random_conv(std::mt19937& rng) {
  auto pick = [&](std::initializer_list<int64_t> v) { return *(v.begin() + rng() % v.size()); };
  RandConv c;
  c.groups = pick({1, 1, 2, 3, 4, 8});
  const int64_t kic = pick({1, 5, 8, 16, 16, 31, 32, 32, 33, 40, 48, 64, 70, 96, 128});
  c.ic = kic * c.groups;
  c.oc = c.groups * pick({1, 3, 4, 7, 8, 9, 16, 20, 32, 33, 40});
  c.kh = pick({1, 2, 3, 3, 5});
  c.kw = rng() % 4 == 0 ? pick({1, 3, 5}) : c.kh;
  c.sh = pick({1, 1, 2});
  c.sw = rng() % 3 == 0 ? pick({1, 2, 3}) : c.sh;
  c.dh = pick({1, 1, 2, 3});
  c.dw = rng() % 3 == 0 ? pick({1, 2}) : c.dh;
  c.n = pick({1, 1, 2});
  c.ih = pick({1, 3, 5, 8, 11, 14});
  c.iw = pick({2, 4, 7, 9, 13});
  c.pt = rng() % 4;
  c.pb = rng() % 4;
  c.pl = rng() % 4;
  c.pr = rng() % 4;
  // Keep the output non-empty.
  while (c.ih + c.pt + c.pb < c.dh * (c.kh - 1) + 1) ++c.pb;
  while (c.iw + c.pl + c.pr < c.dw * (c.kw - 1) + 1) ++c.pr;
  c.pad_value = static_cast<int>(rng() % 3) - 1;
  c.bit_out = rng() % 3 == 0;
  c.thr = rng() % 2 == 0;
  c.gt = rng() % 4 == 0;
  c.act = pick({0, 0, 1, 2}) == 0 ? "none" : (rng() % 2 ? "relu" : "relu6");
  return c;
}

const char* const kPacked[] = {"bconv2d.packed_naive", "bconv2d.binary_scalar", "bconv2d.neon_opt", "bconv2d.avx2"};

}  // namespace

BF_TEST(bconv_packed_random_shapes_match_reference) {
  std::mt19937 rng(20261030);
  const int kCases = 120;
  for (int idx = 0; idx < kCases; ++idx) {
    RandConv c = random_conv(rng);
    auto m = make_conv(c, 5000 + idx);
    auto ref = run(m, {"reference"}, 1);
    for (const char* name : kPacked) {
      if (!available(name)) continue;
      const int threads = 1 + idx % 4;
      auto out = run(m, {}, threads, name);
      EXPECT_EQ(out.kernel, std::string(name));
      for (size_t r = 0; r < 2; ++r)
        if (out.y[r] != ref.y[r])
          throw bftest::Failure{"random case " + std::to_string(idx) + " (" + describe(c) + ") kernel " + name + " threads " +
                                std::to_string(threads) + " run " + std::to_string(r)};
    }
  }
}

// ------------------------------------------------------- feature masking
namespace {

// A BHViT-like grouped layer and a BDense; returns "<bconv kernel>,<bdense kernel>".
std::shared_ptr<Model> gating_model() {
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int64_t ic = 64, groups = 4, kic = 16, oc = 64, k = 1024, on = 40;
  std::vector<int8_t> w(static_cast<size_t>(oc * 9 * kic)), wd(static_cast<size_t>(on * k));
  for (auto& v : w) v = rng() & 1 ? 1 : -1;
  for (auto& v : wd) v = rng() & 1 ? 1 : -1;
  auto wp = bftest::pack_pm1(w, oc * 9, kic), wdp = bftest::pack_pm1(wd, on, k);
  std::vector<float> mul(static_cast<size_t>(on + oc)), add(mul.size()), thr(static_cast<size_t>(ic));
  for (auto& v : mul) v = 0.01f + std::fabs(nd(rng)) * 0.1f;
  for (auto& v : add) v = nd(rng);
  for (auto& v : thr) v = nd(rng) * 0.3f;
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {oc, 3, 3, kic}, wp.data(), "OHWI_bit");
  mb.constant("wd", DType::kBit, {on, k}, wdp.data());
  mb.constant("mul", DType::kF32, {oc}, mul.data());
  mb.constant("add", DType::kF32, {oc}, add.data());
  mb.constant("muld", DType::kF32, {on}, mul.data() + oc);
  mb.constant("addd", DType::kF32, {on}, add.data() + oc);
  mb.constant("thr", DType::kF32, {ic}, thr.data());
  mb.input("x", DType::kF32, {1, 4, 4, ic});
  Json a = Json::Object{{"pads", Json(Json::Array{Json(3), Json(3), Json(3), Json(3)})},
                        {"dilations", Json(Json::Array{Json(3), Json(3)})},
                        {"groups", Json(groups)}};
  mb.node("BConv2D", {"x", "w", "mul", "add", "thr"}, {"h"}, a);
  mb.node("Reshape", {"h"}, {"hr"}, Json::Object{{"shape", Json(Json::Array{Json(1), Json(k)})}});
  mb.node("BDense", {"hr", "wd", "muld", "addd", ""}, {"y"});
  mb.output("y");
  return mb.build();
}

std::string choices(const Session& s) {
  std::string r;
  for (const auto& kv : s.kernel_choices())
    if (kv.second.rfind("bconv2d.", 0) == 0) r = kv.second + r;
    else if (kv.second.rfind("bdense.", 0) == 0) r += "," + kv.second;
  return r;
}

// Default stage-neon kernels for a feature set (what Session must pick).
std::string expected_for(uint32_t f) {
  if (f & kFeatNeon) return "bconv2d.neon_opt,bdense.neon";
  if ((f & (kFeatAvx2 | kFeatPopcnt)) == (kFeatAvx2 | kFeatPopcnt)) return "bconv2d.avx2,bdense.avx2";
  return "bconv2d.binary_scalar,bdense.binary_scalar";
}

std::vector<uint8_t> run_gating(const std::set<std::string>& tiers, int threads, std::string* kernels) {
  auto m = gating_model();
  SessionOptions o;
  o.num_threads = threads;
  o.policy = KernelPolicy::tiers(tiers);
  Session s(m, o);
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.f, 1.f);
  float* x = s.input(0).data<float>();
  for (int64_t i = 0; i < s.input(0).size(); ++i) x[i] = nd(rng);
  s.run();
  if (kernels) *kernels = choices(s);
  const Tensor& y = s.output(0);
  std::vector<uint8_t> out(y.bytes());
  std::memcpy(out.data(), y.raw(), out.size());
  return out;
}

// Re-executes this test binary running only binary_avx2_gating_child with
// extra environment variables; returns its exit status. Honors
// BITINO_TEST_LAUNCHER like test_int8_kernels.cc (qemu user mode).
int spawn_child(const std::vector<std::string>& extra_env) {
  char exe[4096];
  ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (len <= 0) throw bftest::Failure{"readlink /proc/self/exe failed"};
  exe[len] = 0;
  std::vector<std::string> env_store;
  for (char** e = environ; *e; ++e) {
    std::string s = *e;
    bool overridden = false;
    for (const auto& x : extra_env) overridden |= s.compare(0, x.find('=') + 1, x, 0, x.find('=') + 1) == 0;
    if (!overridden) env_store.push_back(s);
  }
  for (const auto& x : extra_env) env_store.push_back(x);
  std::vector<char*> envp;
  for (auto& s : env_store) envp.push_back(&s[0]);
  envp.push_back(nullptr);
  std::vector<std::string> launch;
  if (const char* l = std::getenv("BITINO_TEST_LAUNCHER")) {
    std::string cur;
    for (const char* c = l;; ++c) {
      if (*c == '|' || *c == 0) {
        if (!cur.empty()) launch.push_back(cur);
        cur.clear();
        if (*c == 0) break;
      } else {
        cur += *c;
      }
    }
  }
  launch.push_back(exe);
  launch.push_back("binary_avx2_gating_child");
  std::vector<char*> argv;
  for (auto& a : launch) argv.push_back(&a[0]);
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
  pid_t pid;
  int rc = posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) throw bftest::Failure{"posix_spawn failed: " + std::to_string(rc)};
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

const std::set<std::string> kNeonStage = {"reference", "binary_scalar", "neon", "binary_avx2"};

}  // namespace

BF_TEST(binary_avx2_selection_follows_features) {
  std::string k;
  auto ref = run_gating({"reference"}, 1, nullptr);
  auto out = run_gating(kNeonStage, 4, &k);
  EXPECT_EQ(k, expected_for(cpu_info().features));
  EXPECT_TRUE(out == ref);
#if defined(__x86_64__)
  // An x86-64 CPU with AVX2 + POPCNT must get the AVX2 kernels at stage neon.
  if (cpu_info().has(kFeatAvx2 | kFeatPopcnt)) EXPECT_EQ(k, std::string("bconv2d.avx2,bdense.avx2"));
#endif
}

BF_TEST(binary_avx2_feature_masking_selects_fallbacks) {
  if (std::getenv("BF_AVX2_CHILD")) return;  // we are the child of another run
  const uint32_t have = cpu_info().features;
  if (!(have & kFeatAvx2)) return;  // nothing to mask on this CPU
  const char* prev = std::getenv("BITINO_DISABLE_FEATURES");
  std::string base = prev && *prev ? std::string(prev) + "," : "";
  struct Mask {
    const char* name;
    uint32_t bit;
  };
  for (Mask mk : {Mask{"avx2", kFeatAvx2}, Mask{"popcnt", kFeatPopcnt}}) {
    const std::string e = expected_for(have & ~mk.bit);
    EXPECT_EQ(e, std::string("bconv2d.binary_scalar,bdense.binary_scalar"));
    std::vector<std::string> env = {"BITINO_DISABLE_FEATURES=" + base + mk.name, "BF_AVX2_CHILD=1", "BF_AVX2_MASKED=" + std::string(mk.name),
                                    "BF_AVX2_EXPECT=" + e};
    int rc = spawn_child(env);
    if (rc != 0)
      throw bftest::Failure{std::string("masked child failed (") + mk.name + "), rerun: BITINO_DISABLE_FEATURES=" + base + mk.name +
                            " BF_AVX2_CHILD=1 BF_AVX2_MASKED=" + mk.name + " BF_AVX2_EXPECT=" + e + " bitino_tests binary_avx2_gating_child"};
  }
}

BF_TEST(binary_avx2_gating_child) {
  const char* expect = std::getenv("BF_AVX2_EXPECT");
  if (!std::getenv("BF_AVX2_CHILD") || !expect) return;  // only meaningful when spawned
  std::string masked = std::getenv("BF_AVX2_MASKED") ? std::getenv("BF_AVX2_MASKED") : "";
  if (masked == "avx2") EXPECT_TRUE(!cpu_info().has(kFeatAvx2));
  if (masked == "popcnt") EXPECT_TRUE(!cpu_info().has(kFeatPopcnt));
  std::string k;
  auto ref = run_gating({"reference"}, 1, nullptr);
  auto out = run_gating(kNeonStage, 3, &k);
  EXPECT_EQ(k, std::string(expect));
  EXPECT_TRUE(out == ref);  // the fallback kernels (grouped binary_scalar) stay bit-exact
  // Forcing the AVX2 kernel without its features must fail at plan time.
  SessionOptions o;
  o.policy.force["BConv2D"] = "bconv2d.avx2";
  EXPECT_THROWS(Session(gating_model(), o));
}

// ------------------------------------------------------ ternary BDense
BF_TEST(bdense_ternary_simd_matches_reference) {
  struct Case {
    int64_t rows, k, o;
  };
  int idx = 0;
  // 4100 (33 steps: two uint8 runs) and 2000 (16 steps) cover the long-K paths of bdense.ternary_neon's 4-row micro-kernel.
  for (Case c : {Case{9, 7, 37}, Case{1, 768, 768}, Case{3, 31, 5}, Case{64, 100, 40}, Case{130, 768, 70}, Case{2, 3072, 33},
                 Case{17, 64, 8}, Case{5, 129, 9}, Case{8, 4100, 12}, Case{6, 2000, 7},
                 // AArch64 8-channel blocks: 4-row tiles without zero inputs on the binary 4x8 tile (seed 2: zeros only in
                 // rows r % 5 == 2, so tiles of zero-free rows, mixed tiles and row tails)
                 Case{13, 768, 64}, Case{66, 3072, 32}, Case{8, 4100, 16}, Case{23, 96, 24}}) {
    for (int wz = 0; wz < 2; ++wz)
      for (int thr = 0; thr < 2; ++thr) {
        std::mt19937 rng(static_cast<uint32_t>(700 + idx));
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<int8_t> w(static_cast<size_t>(c.o * c.k)), z(w.size(), 1);
        for (auto& v : w) v = rng() & 1 ? 1 : -1;
        for (auto& v : z) v = rng() % 7 == 0 ? -1 : 1;
        auto wp = bftest::pack_pm1(w, c.o, c.k), zp = bftest::pack_pm1(z, c.o, c.k);
        std::vector<float> mul(static_cast<size_t>(c.o)), add(mul.size()), th(static_cast<size_t>(c.k));
        for (size_t i = 0; i < mul.size(); ++i) mul[i] = i % 5 == 0 ? -0.f : (i % 7 == 1 ? INFINITY : nd(rng));
        for (auto& v : add) v = nd(rng);
        for (auto& v : th) v = nd(rng) * 0.3f;
        bftest::ModelBuilder mb;
        mb.constant("w", DType::kBit, {c.o, c.k}, wp.data());
        mb.constant("mul", DType::kF32, {c.o}, mul.data());
        mb.constant("add", DType::kF32, {c.o}, add.data());
        mb.constant("thr", DType::kF32, {c.k}, th.data());
        mb.constant("z", DType::kBit, {c.o, c.k}, zp.data());
        mb.input("x", DType::kF32, {c.rows, c.k});
        mb.node("BDense", {"x", "w", "mul", idx % 3 ? "add" : "", thr ? "thr" : "", wz ? "z" : ""}, {"y"},
                Json(Json::Object{{"input_binarize", "sign_zero"}, {"activation", idx % 3 == 2 ? "relu6" : idx % 2 ? "relu" : "none"}}));
        mb.output("y");
        auto m = mb.build();
        auto run_dense = [&](const std::string& force, int threads, uint32_t seed, std::string* kernel) {
          SessionOptions o;
          o.num_threads = threads;
          o.policy = force.empty() ? KernelPolicy::tiers({"reference"}) : KernelPolicy::all();
          if (!force.empty()) o.policy.force["BDense"] = force;
          Session s(m, o);
          fill(s.input(0), seed);
          // Exact ties with the threshold (sign_zero -> 0).
          float* x = s.input(0).data<float>();
          for (int64_t i = 0; i < s.input(0).size(); i += 9)
            if (seed == 1 || (i / c.k) % 5 == 2) x[i] = thr ? th[i % c.k] : 0.f;
          s.run();
          if (kernel) *kernel = s.kernel_choices().begin()->second;
          const Tensor& y = s.output(0);
          std::vector<uint8_t> out(y.bytes());
          std::memcpy(out.data(), y.raw(), out.size());
          return out;
        };
        for (uint32_t seed : {1u, 2u}) {
          auto ref = run_dense("", 1, seed, nullptr);
          for (const char* name : {"bdense.ternary_avx2", "bdense.ternary_neon"}) {
            if (!available(name)) continue;
            for (int threads : {1, 3, 4}) {
              std::string kernel;
              auto out = run_dense(name, threads, seed, &kernel);
              EXPECT_EQ(kernel, std::string(name));
              if (out != ref)
                throw bftest::Failure{std::string(name) + " mismatch case " + std::to_string(idx) + " wz " + std::to_string(wz) +
                                      " thr " + std::to_string(thr) + " threads " + std::to_string(threads)};
            }
          }
        }
      }
    ++idx;
  }
  // Stage neon prefers the SIMD kernel over bdense.ternary_bitplane.
  std::vector<int8_t> w(8 * 64, 1);
  auto wp = bftest::pack_pm1(w, 8, 64);
  bftest::ModelBuilder mb;
  mb.constant("w", DType::kBit, {8, 64}, wp.data());
  mb.input("x", DType::kF32, {2, 64});
  mb.node("BDense", {"x", "w", "", "", ""}, {"y"}, Json(Json::Object{{"input_binarize", "sign_zero"}}));
  mb.output("y");
  SessionOptions o;
  o.policy = KernelPolicy::tiers(kNeonStage);
  Session s(mb.build(), o);
  const std::string want = cpu_info().has(kFeatNeon)                        ? "bdense.ternary_neon"
                           : cpu_info().has(kFeatAvx2 | kFeatPopcnt) ? "bdense.ternary_avx2"
                                                                     : "bdense.ternary_bitplane";
  EXPECT_EQ(s.kernel_choices().begin()->second, want);
}

// ------------------------------------------- QBConv2D / QBDense (k-bit x binary)
namespace {

// Inputs around every DoReFa level boundary (x·n = m + 0.5 ties round to even),
// below 0, above 1, ±0 and ±inf. NaN is excluded: the reference converts the
// NaN level with a float->int cast (undefined behaviour in C++).
void fill_levels(Tensor& t, uint32_t seed, int bits) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-0.3f, 1.3f);
  const float n = static_cast<float>((1 << bits) - 1);
  float* p = t.data<float>();
  for (int64_t i = 0; i < t.size(); ++i) {
    switch (rng() % 9) {
      case 0: p[i] = (static_cast<float>(rng() % (1 << bits)) + 0.5f) / n; break;  // tie
      case 1: p[i] = -0.f; break;
      case 2: p[i] = INFINITY; break;
      case 3: p[i] = -INFINITY; break;
      default: p[i] = u(rng);
    }
  }
}

std::vector<uint8_t> run_q(const std::shared_ptr<Model>& m, const std::string& op, const std::string& force, int threads, uint32_t seed,
                           int bits, std::string* kernel) {
  SessionOptions o;
  o.num_threads = threads;
  o.policy = force.empty() ? KernelPolicy::tiers({"reference"}) : KernelPolicy::all();
  if (!force.empty()) o.policy.force[op] = force;
  Session s(m, o);
  fill_levels(s.input(0), seed, bits);
  s.run();
  if (kernel) *kernel = s.kernel_choices().begin()->second;
  const Tensor& y = s.output(0);
  std::vector<uint8_t> out(y.bytes());
  std::memcpy(out.data(), y.raw(), out.size());
  return out;
}

}  // namespace

// qbconv2d.avx2 (act_bits <= 4) and qbconv2d.neon (act_bits <= 8: bit-planes
// as rows of the BConv2D micro-kernel with AND, 4-row tiles spanning pixels
// when act_bits does not divide 4).
BF_TEST(qbconv2d_simd_random_shapes_match_reference) {
  for (const std::string name : {"qbconv2d.avx2", "qbconv2d.neon"}) {
  if (!available(name)) continue;
  const int max_bits = name == "qbconv2d.avx2" ? 4 : 8;
  std::mt19937 rng(424242);
  for (int idx = 0; idx < 48; ++idx) {
    RandConv c = random_conv(rng);
    c.groups = 1;
    c.pad_value = 0;
    const int bits = 1 + idx % max_bits;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int8_t> w(static_cast<size_t>(c.oc * c.kh * c.kw * c.ic));
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    auto wp = bftest::pack_pm1(w, c.oc * c.kh * c.kw, c.ic);
    std::vector<float> mul(static_cast<size_t>(c.oc)), add(mul.size());
    for (size_t i = 0; i < mul.size(); ++i) mul[i] = i % 5 == 0 ? -0.f : nd(rng) * 0.1f;
    for (auto& v : add) v = nd(rng);
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {c.oc, c.kh, c.kw, c.ic}, wp.data(), "OHWI_bit");
    mb.constant("mul", DType::kF32, {c.oc}, mul.data());
    mb.constant("add", DType::kF32, {c.oc}, add.data());
    mb.input("x", DType::kF32, {c.n, c.ih, c.iw, c.ic});
    Json attrs = Json::Object{{"strides", Json(Json::Array{Json(c.sh), Json(c.sw)})},
                              {"dilations", Json(Json::Array{Json(c.dh), Json(c.dw)})},
                              {"pads", Json(Json::Array{Json(c.pt), Json(c.pb), Json(c.pl), Json(c.pr)})},
                              {"act_bits", Json(bits)},
                              {"activation", c.act}};
    mb.node("QBConv2D", {"x", "w", "mul", idx % 3 ? "add" : ""}, {"y"}, attrs);
    mb.output("y");
    auto m = mb.build();
    for (uint32_t seed : {1u, 2u}) {
      auto ref = run_q(m, "QBConv2D", "", 1, seed + idx, bits, nullptr);
      const int threads = 1 + idx % 4;
      std::string kernel;
      auto out = run_q(m, "QBConv2D", name, threads, seed + idx, bits, &kernel);
      EXPECT_EQ(kernel, name);
      if (out != ref)
        throw bftest::Failure{name + " mismatch case " + std::to_string(idx) + " (" + describe(c) + ") bits " + std::to_string(bits) +
                              " threads " + std::to_string(threads)};
    }
  }
  }
}

BF_TEST(qbdense_simd_matches_reference) {
  struct Case {
    int64_t rows, k, o;
    int bits;
  };
  for (const std::string name : {"qbdense.avx2", "qbdense.neon"}) {
  if (!available(name)) continue;
  const int max_bits = name == "qbdense.avx2" ? 4 : 8;
  int idx = 0;
  for (Case c : {Case{1, 9216, 70, 2}, Case{1, 4096, 1000, 2}, Case{3, 33, 9, 1}, Case{64, 300, 40, 3}, Case{130, 96, 17, 4},
                 Case{2, 7, 5, 2}, Case{5, 4100, 260, 2}, Case{1, 520, 13, 8}, Case{67, 200, 9, 5}, Case{9, 4000, 33, 7}}) {
    if (c.bits > max_bits) continue;
    std::mt19937 rng(static_cast<uint32_t>(900 + idx));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int8_t> w(static_cast<size_t>(c.o * c.k));
    for (auto& v : w) v = rng() & 1 ? 1 : -1;
    auto wp = bftest::pack_pm1(w, c.o, c.k);
    std::vector<float> mul(static_cast<size_t>(c.o)), add(mul.size());
    for (auto& v : mul) v = nd(rng) * 0.01f;
    for (auto& v : add) v = nd(rng);
    bftest::ModelBuilder mb;
    mb.constant("w", DType::kBit, {c.o, c.k}, wp.data());
    mb.constant("mul", DType::kF32, {c.o}, mul.data());
    mb.constant("add", DType::kF32, {c.o}, add.data());
    mb.input("x", DType::kF32, {c.rows, c.k});
    mb.node("QBDense", {"x", "w", "mul", "add"}, {"y"},
            Json(Json::Object{{"act_bits", Json(c.bits)}, {"activation", idx % 2 ? "relu" : "none"}}));
    mb.output("y");
    auto m = mb.build();
    auto ref = run_q(m, "QBDense", "", 1, 5 + idx, c.bits, nullptr);
    for (int threads : {1, 4}) {
      std::string kernel;
      auto out = run_q(m, "QBDense", name, threads, 5 + idx, c.bits, &kernel);
      EXPECT_EQ(kernel, name);
      if (out != ref) throw bftest::Failure{name + " mismatch case " + std::to_string(idx) + " threads " + std::to_string(threads)};
    }
    ++idx;
  }
  }
}
