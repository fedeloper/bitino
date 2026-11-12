// 32-bit ARM (armeabi-v7a) NEON: the AArch64-only intrinsics emulated by kernels/neon_a32.h against their
// AArch64 definitions, and the denormal handling of the AArch32 kernels (AArch32 NEON flushes denormals to zero;
// compares against zero, bit packing and the selections of float_opt keep the IEEE results). The kernels
// themselves are checked against the reference kernels by the shared tests (test_binary_kernels.cc,
// test_float_kernels.cc, test_transformer_opt.cc ...), which select them on any NEON build. Empty elsewhere.
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#include "testing.h"

#if defined(__arm__) && defined(__ARM_NEON)
#include "bitino/bitino.h"
#include "kernels/binary_neon/bitpack_neon.h"
#include "kernels/float_opt/float_opt.h"
#include "kernels/neon_a32.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

float from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// Bit patterns covering zeros, every denormal boundary, normals, infinities and NaNs of both signs, plus random.
std::vector<float> edge_floats(size_t n_random, uint32_t seed) {
  std::vector<uint32_t> u = {0u,          1u,          2u,          0x007fffffu, 0x00800000u, 0x00800001u, 0x00ffffffu,
                             0x01000000u, 0x017fffffu, 0x01800000u, 0x3f800000u, 0x7f7fffffu, 0x7f800000u, 0x7f800001u,
                             0x7fc00000u, 0x7fffffffu, 0x00012345u, 0x00400000u};
  const size_t base = u.size();
  for (size_t i = 0; i < base; ++i) u.push_back(u[i] | 0x80000000u);
  std::mt19937 rng(seed);
  for (size_t i = 0; i < n_random; ++i) {
    uint32_t r = rng();
    if (i % 3 == 0) r &= 0x807fffffu;  // denormals
    u.push_back(r);
  }
  std::vector<float> f;
  for (uint32_t x : u) f.push_back(from_bits(x));
  return f;
}

// IEEE compares in double (no AArch32 vector unit for doubles, so these cannot be flushed).
bool ge0(float x) { return static_cast<double>(x) >= 0.0; }
bool gt0(float x) { return static_cast<double>(x) > 0.0; }
bool lt0(float x) { return static_cast<double>(x) < 0.0; }

}  // namespace

BF_TEST(a32_neon_flushes_denormals) {
  // Documents why the AArch32 kernels need the bit-pattern compares: the vector unit flushes denormal inputs and
  // results (standard FPSCR value), the scalar VFP unit does not.
  // Inline asm: the compiler would fold or scalarize (VFP, IEEE) intrinsics on constants or single lanes.
  volatile float in[2] = {1e-40f, -1e-40f};
  float32x4_t sum = vdupq_n_f32(in[0]), ge = vdupq_n_f32(in[1]);
  const float32x4_t z = vdupq_n_f32(0.f);
  asm volatile("vadd.f32 %q0, %q0, %q2\n vcge.f32 %q1, %q1, %q2\n" : "+w"(sum), "+w"(ge) : "w"(z));
  float s4[4];
  uint32_t g4[4];
  vst1q_f32(s4, sum);
  vst1q_u32(g4, vreinterpretq_u32_f32(ge));
  EXPECT_EQ(s4[0], 0.f);    // 1e-40 + 0 flushed
  EXPECT_EQ(g4[0], ~0u);    // -1e-40 >= 0 "true"
  EXPECT_TRUE(in[0] + 0.f != 0.f);  // scalar VFP: IEEE
}

BF_TEST(a32_zero_compares_exact) {
  const std::vector<float> x = edge_floats(4000, 1);
  for (size_t i = 0; i + 4 <= x.size(); i += 4) {
    const float32x4_t v = vld1q_f32(&x[i]);
    uint32_t ge[4], gt[4], lt[4];
    vst1q_u32(ge, vcgezq_f32(v));
    vst1q_u32(gt, vcgtzq_f32(v));
    vst1q_u32(lt, vcltzq_f32(v));
    for (int j = 0; j < 4; ++j) {
      const float f = x[i + j];
      if ((ge[j] != 0) != ge0(f) || (gt[j] != 0) != gt0(f) || (lt[j] != 0) != lt0(f))
        throw bftest::Failure{"zero compare of bits " + std::to_string(bits(f))};
      if (ge[j] != 0 && ge[j] != ~0u) throw bftest::Failure{"mask not all ones"};
    }
  }
  bool any = bf_a32_any_denormal(x.data(), 16);
  EXPECT_TRUE(any);
  const float normal[5] = {0.f, -0.f, 1.f, 1e-37f, -std::numeric_limits<float>::infinity()};
  EXPECT_TRUE(!bf_a32_any_denormal(normal, 5));
}

BF_TEST(a32_shim_matches_aarch64_definitions) {
  std::mt19937 rng(5);
  for (int it = 0; it < 2000; ++it) {
    uint8_t a[16], b[16], t[64], idx[16];
    for (auto& v : a) v = static_cast<uint8_t>(rng());
    for (auto& v : b) v = static_cast<uint8_t>(rng());
    for (auto& v : t) v = static_cast<uint8_t>(rng());
    for (auto& v : idx) v = static_cast<uint8_t>(it % 2 ? rng() % 80 : rng());  // in and out of range
    const uint8x16_t va = vld1q_u8(a), vb = vld1q_u8(b), vi = vld1q_u8(idx);
    uint8_t r[16];
    // ADDP
    vst1q_u8(r, vpaddq_u8(va, vb));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], static_cast<uint8_t>(j < 8 ? a[2 * j] + a[2 * j + 1] : b[2 * j - 16] + b[2 * j - 15]));
    uint16_t a16[8], b16[8], r16[8];
    std::memcpy(a16, a, 16);
    std::memcpy(b16, b, 16);
    vst1q_u16(r16, vpaddq_u16(vld1q_u16(a16), vld1q_u16(b16)));
    for (int j = 0; j < 8; ++j)
      EXPECT_EQ(r16[j], static_cast<uint16_t>(j < 4 ? a16[2 * j] + a16[2 * j + 1] : b16[2 * j - 8] + b16[2 * j - 7]));
    uint32_t a32[4], b32[4], r32[4];
    std::memcpy(a32, a, 16);
    std::memcpy(b32, b, 16);
    vst1q_u32(r32, vpaddq_u32(vld1q_u32(a32), vld1q_u32(b32)));
    for (int j = 0; j < 4; ++j) EXPECT_EQ(r32[j], j < 2 ? a32[2 * j] + a32[2 * j + 1] : b32[2 * j - 4] + b32[2 * j - 3]);
    // ADDV / UADDLV
    EXPECT_EQ(vaddvq_u32(vld1q_u32(a32)), a32[0] + a32[1] + a32[2] + a32[3]);
    uint32_t s8 = 0, s16 = 0;
    for (int j = 0; j < 16; ++j) s8 += a[j];
    for (int j = 0; j < 8; ++j) s16 += a16[j];
    EXPECT_EQ(static_cast<uint32_t>(vaddlvq_u8(va)), s8);
    EXPECT_EQ(vaddlvq_u16(vld1q_u16(a16)), s16);
    EXPECT_EQ(vmaxvq_u32(vld1q_u32(a32)), std::max(std::max(a32[0], a32[1]), std::max(a32[2], a32[3])));
    // UZP1 / UZP2 / ZIP1 / ZIP2
    vst1q_u8(r, vuzp1q_u8(va, vb));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], j < 8 ? a[2 * j] : b[2 * j - 16]);
    vst1q_u8(r, vuzp2q_u8(va, vb));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], j < 8 ? a[2 * j + 1] : b[2 * j - 15]);
    vst1q_u16(r16, vuzp1q_u16(vld1q_u16(a16), vld1q_u16(b16)));
    for (int j = 0; j < 8; ++j) EXPECT_EQ(r16[j], j < 4 ? a16[2 * j] : b16[2 * j - 8]);
    vst1q_u16(r16, vuzp2q_u16(vld1q_u16(a16), vld1q_u16(b16)));
    for (int j = 0; j < 8; ++j) EXPECT_EQ(r16[j], j < 4 ? a16[2 * j + 1] : b16[2 * j - 7]);
    vst1q_u8(r, vzip1q_u8(va, vb));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], j % 2 ? b[j / 2] : a[j / 2]);
    vst1q_u8(r, vzip2q_u8(va, vb));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], j % 2 ? b[8 + j / 2] : a[8 + j / 2]);
    vst1q_u16(r16, vzip1q_u16(vld1q_u16(a16), vld1q_u16(b16)));
    for (int j = 0; j < 8; ++j) EXPECT_EQ(r16[j], j % 2 ? b16[j / 2] : a16[j / 2]);
    vst1q_u16(r16, vzip2q_u16(vld1q_u16(a16), vld1q_u16(b16)));
    for (int j = 0; j < 8; ++j) EXPECT_EQ(r16[j], j % 2 ? b16[4 + j / 2] : a16[4 + j / 2]);
    uint64_t a64[2], b64[2], r64[2];
    std::memcpy(a64, a, 16);
    std::memcpy(b64, b, 16);
    vst1q_u64(r64, vzip1q_u64(vld1q_u64(a64), vld1q_u64(b64)));
    EXPECT_TRUE(r64[0] == a64[0] && r64[1] == b64[0]);
    vst1q_u64(r64, vzip2q_u64(vld1q_u64(a64), vld1q_u64(b64)));
    EXPECT_TRUE(r64[0] == a64[1] && r64[1] == b64[1]);
    // TBL / TBX (out-of-range index: 0 / lane kept)
    vst1q_u8(r, vqtbl1q_u8(vld1q_u8(t), vi));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], idx[j] < 16 ? t[idx[j]] : 0);
    uint8x16x2_t t2;
    t2.val[0] = vld1q_u8(t);
    t2.val[1] = vld1q_u8(t + 16);
    vst1q_u8(r, vqtbl2q_u8(t2, vi));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], idx[j] < 32 ? t[idx[j]] : 0);
    uint8x16x4_t t4;
    for (int q = 0; q < 4; ++q) t4.val[q] = vld1q_u8(t + 16 * q);
    vst1q_u8(r, vqtbl4q_u8(t4, vi));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], idx[j] < 64 ? t[idx[j]] : 0);
    vst1q_u8(r, vqtbx4q_u8(va, t4, vi));
    for (int j = 0; j < 16; ++j) EXPECT_EQ(r[j], idx[j] < 64 ? t[idx[j]] : a[j]);
    // high-half widening
    int16_t w16[8];
    vst1q_s16(w16, vmovl_high_s8(vreinterpretq_s8_u8(va)));
    for (int j = 0; j < 8; ++j) EXPECT_EQ(w16[j], static_cast<int8_t>(a[8 + j]));
  }
  // FCVTNS / FRINTN: ties to even, saturation, NaN -> 0, sign of zero
  const std::vector<float> f = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, -0.f, 0.f, 0.49999997f, 8388607.5f, -8388607.5f, 8388609.f,
                                1e10f, -1e10f, 3e9f, -3e9f, 1e-40f, -1e-40f, -0.3f, std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 2147483520.f,
                                -2147483648.f, 123.456f, -77.5f};
  for (size_t i = 0; i < f.size(); i += 4) {
    float in[4] = {0.f, 0.f, 0.f, 0.f}, rn[4];
    int32_t cv[4];
    for (size_t j = 0; j < 4 && i + j < f.size(); ++j) in[j] = f[i + j];
    vst1q_f32(rn, vrndnq_f32(vld1q_f32(in)));
    vst1q_s32(cv, vcvtnq_s32_f32(vld1q_f32(in)));
    for (int j = 0; j < 4; ++j) {
      const float x = std::fpclassify(in[j]) == FP_SUBNORMAL ? std::copysign(0.f, in[j]) : in[j];  // flushed input
      const float want = std::nearbyint(x);
      if (!(std::isnan(want) ? std::isnan(rn[j]) : bits(want) == bits(rn[j])))
        throw bftest::Failure{"vrndnq_f32(" + std::to_string(in[j]) + ") = " + std::to_string(rn[j])};
      const double d = std::nearbyint(static_cast<double>(x));
      const int32_t wi = std::isnan(x) ? 0 : d >= 2147483647.0 ? INT32_MAX : d <= -2147483648.0 ? INT32_MIN : static_cast<int32_t>(d);
      if (cv[j] != wi) throw bftest::Failure{"vcvtnq_s32_f32(" + std::to_string(in[j]) + ") = " + std::to_string(cv[j])};
    }
  }
}

BF_TEST(a32_bit_packing_exact_on_denormals) {
  // Rows with denormals of both signs (and NaN, -0): the NEON packing (exact zero compares; thresholds: the scalar
  // path when a denormal is present) must give the bits of the IEEE definition.
  std::mt19937 rng(9);
  for (int64_t c : {32, 64, 77, 96, 131, 256}) {
    for (int with_thr = 0; with_thr < 3; ++with_thr) {  // none, normal thresholds, thresholds with denormals / zeros
      std::vector<float> x = edge_floats(static_cast<size_t>(c), static_cast<uint32_t>(c + with_thr));
      x.resize(static_cast<size_t>(c));
      std::shuffle(x.begin(), x.end(), rng);
      std::vector<float> thr(static_cast<size_t>(c));
      std::normal_distribution<float> nd(0.f, 1.f);
      for (auto& v : thr) v = nd(rng);
      if (with_thr == 2)
        for (size_t i = 0; i < thr.size(); i += 3) thr[i] = i % 2 ? -0.f : from_bits(static_cast<uint32_t>(rng() & 0x807fffffu));
      const float* tp = with_thr ? thr.data() : nullptr;
      for (BinarizeMode mode : {BinarizeMode::kGe, BinarizeMode::kGt}) {
        std::vector<uint32_t> got(static_cast<size_t>(bit_words(c)) + 1, 0xdeadbeefu), want(got.size(), 0u);
        pack_row_f32_neon(x.data(), c, got.data(), mode, tp);
        for (int64_t i = 0; i < c; ++i) {
          const double v = x[i], t = tp ? static_cast<double>(tp[i]) : 0.0;
          const bool pos = mode == BinarizeMode::kGt ? v > t : v >= t;
          want[i / 32] |= static_cast<uint32_t>(!pos) << (i % 32);
        }
        for (int64_t w = 0; w < bit_words(c); ++w)
          if (got[w] != want[w]) throw bftest::Failure{"pack c=" + std::to_string(c) + " thr=" + std::to_string(with_thr) + " word " + std::to_string(w)};
      }
      std::vector<uint32_t> nz(static_cast<size_t>(bit_words(c))), neg(nz.size()), wnz(nz.size(), 0u), wneg(nz.size(), 0u);
      pack_ternary_row_neon(x.data(), c, nz.data(), neg.data(), tp);
      for (int64_t i = 0; i < c; ++i) {
        const double v = x[i], t = tp ? static_cast<double>(tp[i]) : 0.0;
        wnz[i / 32] |= static_cast<uint32_t>(v > t || v < t) << (i % 32);
        wneg[i / 32] |= static_cast<uint32_t>(v < t) << (i % 32);
      }
      EXPECT_TRUE(nz == wnz);
      EXPECT_TRUE(neg == wneg);
    }
  }
}

BF_TEST(a32_float_opt_selections_exact_on_denormals) {
  // sign, sign_zero, relu, relu6, hardtanh, leaky_relu's select, hardswish and GELU of denormal inputs: the
  // float_opt kernels equal the reference kernels bit for bit (GELU's sign as well: BiBERT binarizes it).
  std::vector<float> x = edge_floats(3000, 3);
  for (float& v : x)
    if (std::isnan(v)) v = 0.25f;  // NaN payloads are not compared here
  for (const char* t : {"sign", "sign_zero", "relu", "relu6", "hardtanh", "hardswish", "gelu"}) {
    bftest::ModelBuilder mb;
    mb.input("x", DType::kF32, {static_cast<int64_t>(x.size())});
    Json::Object attrs{{"type", t}};
    if (std::string(t) == "hardtanh") {
      attrs["min"] = Json(0.0);
      attrs["max"] = Json(1.0);
    }
    mb.node("Activation", {"x"}, {"y"}, Json(attrs));
    mb.output("y");
    auto m = mb.build();
    std::vector<float> out[2];
    std::string kernel[2];
    for (int i = 0; i < 2; ++i) {
      SessionOptions o;
      o.policy = KernelPolicy::tiers(i == 0 ? std::set<std::string>{"reference"} : std::set<std::string>{"reference", "float_opt"});
      Session s(m, o);
      std::memcpy(s.input(0).raw(), x.data(), x.size() * 4);
      s.run();
      out[i].assign(s.output(0).data<float>(), s.output(0).data<float>() + x.size());
      kernel[i] = s.kernel_choices().begin()->second;
    }
    EXPECT_TRUE(kernel[1].find("float_opt") != std::string::npos);
    for (size_t i = 0; i < x.size(); ++i) {
      if (std::string(t) == "gelu") {  // within the GELU bound; exact for |x| < 2^-125 (and the sign always)
        const bool tiny = (bits(x[i]) & 0x7fffffffu) < 0x01000000u;
        const double ref = out[0][i], got = out[1][i];
        const bool sign_ok = (ref > 0) == (got > 0) && (ref < 0) == (got < 0);
        if (!sign_ok || (tiny && bits(out[0][i]) != bits(out[1][i])) || std::fabs(got - ref) > 6e-7 * std::fabs(ref) + 2.3e-16 * std::fabs(x[i]))
          throw bftest::Failure{std::string("gelu of bits ") + std::to_string(bits(x[i])) + ": " + std::to_string(got) + " vs " + std::to_string(ref)};
      } else if (bits(out[0][i]) != bits(out[1][i])) {
        throw bftest::Failure{std::string(t) + " (" + kernel[1] + ") of bits " + std::to_string(bits(x[i])) + ": " + std::to_string(out[1][i]) +
                              " vs " + std::to_string(out[0][i])};
      }
    }
  }
}

#endif
