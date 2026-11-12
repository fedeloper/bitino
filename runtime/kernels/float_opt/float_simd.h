// 4-lane f32 vector abstraction for the float_opt kernels.
//
// On AArch64 every helper is a single NEON intrinsic (baseline ASIMD, no
// extension needed). Elsewhere a plain struct of four floats implements the
// same operations so the *same* kernel code (tiling, packing, tails, borders)
// runs and is tested on the host. The one semantic difference is madd():
// fused (one rounding, vfmaq) on NEON, separately rounded on the portable
// path (-ffp-contract=off keeps the compiler from fusing it).
//
// 32-bit ARM with NEON (BF_FLOAT_OPT_NEON32): NEON intrinsics with the portable
// path's semantics where AArch32 lacks the AArch64 instruction: madd() is VMLA
// (rounded multiply, then add, as the portable path), div() exact (VFP
// divisions, or the exact reciprocal of a power of two), fma_() VFMA when the
// CPU has VFPv4 (kFeatFma; any ARMv8 core) and
// std::fma otherwise, round_ne() / pow2i() / relu() / max_num() without the
// ARMv8 FRINTN / FCVTNS / FMAXNM. AArch32 NEON flushes denormals to zero (see
// kernels/neon_a32.h).
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>

#include "bitino/op.h"
#include "bitino/tensor.h"
#include "ops/ops_common.h"

#include "kernels/neon_a32.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#define BF_FLOAT_OPT_NEON 1
#else
#define BF_FLOAT_OPT_NEON 0
#endif
#if BF_NEON_A32
#define BF_FLOAT_OPT_NEON32 1
#else
#define BF_FLOAT_OPT_NEON32 0
#endif

namespace bitino {
namespace fopt {

#if BF_FLOAT_OPT_NEON
using f32x4 = float32x4_t;
inline f32x4 load(const float* p) { return vld1q_f32(p); }
inline void store(float* p, f32x4 v) { vst1q_f32(p, v); }
inline f32x4 dup(float s) { return vdupq_n_f32(s); }
inline f32x4 add(f32x4 a, f32x4 b) { return vaddq_f32(a, b); }
inline f32x4 mul(f32x4 a, f32x4 b) { return vmulq_f32(a, b); }
inline f32x4 div(f32x4 a, f32x4 b) { return vdivq_f32(a, b); }
inline f32x4 vmax(f32x4 a, f32x4 b) { return vmaxq_f32(a, b); }
inline f32x4 vmin(f32x4 a, f32x4 b) { return vminq_f32(a, b); }
// acc + a * b
inline f32x4 madd(f32x4 acc, f32x4 a, f32x4 b) { return vfmaq_f32(acc, a, b); }
// acc + b * a[L]
template <int L>
inline f32x4 madd_lane(f32x4 acc, f32x4 b, f32x4 a) { return vfmaq_laneq_f32(acc, b, a, L); }
// v >= 0 ? v : v * s  (prelu / leaky_relu)
inline f32x4 select_ge0(f32x4 v, f32x4 neg) { return vbslq_f32(vcgeq_f32(v, vdupq_n_f32(0.f)), v, neg); }
// Reference relu is `v > 0 ? v : 0`, which maps NaN to 0: FMAXNM does the same.
inline f32x4 relu(f32x4 v) { return vmaxnmq_f32(v, vdupq_n_f32(0.f)); }
inline f32x4 sub(f32x4 a, f32x4 b) { return vsubq_f32(a, b); }
inline f32x4 fma_(f32x4 a, f32x4 b, f32x4 c) { return vfmaq_f32(c, a, b); }  // a*b + c, one rounding
inline f32x4 round_ne(f32x4 v) { return vrndnq_f32(v); }
inline f32x4 pow2i(f32x4 n) { return vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(n), vdupq_n_s32(127)), 23)); }
inline f32x4 vabs(f32x4 v) { return vabsq_f32(v); }
inline f32x4 select_lt(f32x4 a, f32x4 b, f32x4 x, f32x4 y) { return vbslq_f32(vcltq_f32(a, b), x, y); }
inline f32x4 select_gt(f32x4 a, f32x4 b, f32x4 x, f32x4 y) { return vbslq_f32(vcgtq_f32(a, b), x, y); }
inline float hsum(f32x4 v) { return vaddvq_f32(v); }
inline float hmax(f32x4 v) { return vmaxvq_f32(v); }
inline f32x4 clamp_std(f32x4 v, f32x4 lo, f32x4 hi) {
  const f32x4 m = vbslq_f32(vcltq_f32(v, lo), lo, v);  // std::max(v, lo) = v < lo ? lo : v
  return vbslq_f32(vcltq_f32(hi, m), hi, m);           // std::min(m, hi) = hi < m ? hi : m
}
inline f32x4 sign_ge(f32x4 v) { return vbslq_f32(vcgeq_f32(v, vdupq_n_f32(0.f)), vdupq_n_f32(1.f), vdupq_n_f32(-1.f)); }
inline f32x4 sign_zero(f32x4 v) {
  const f32x4 z = vdupq_n_f32(0.f);
  return vbslq_f32(vcgtq_f32(v, z), vdupq_n_f32(1.f), vbslq_f32(vcltq_f32(v, z), vdupq_n_f32(-1.f), z));
}
#elif BF_FLOAT_OPT_NEON32
using f32x4 = float32x4_t;
// VFPv4 fused multiply-add available (read once at startup; BITINO_DISABLE_FEATURES=fma masks it).
inline const bool kA32Fma = cpu_info().has(kFeatFma);
inline f32x4 load(const float* p) { return vld1q_f32(p); }
inline void store(float* p, f32x4 v) { vst1q_f32(p, v); }
inline f32x4 dup(float s) { return vdupq_n_f32(s); }
inline f32x4 add(f32x4 a, f32x4 b) { return vaddq_f32(a, b); }
inline f32x4 mul(f32x4 a, f32x4 b) { return vmulq_f32(a, b); }
inline f32x4 div(f32x4 a, f32x4 b) { return vdivq_f32(a, b); }  // neon_a32.h: exact
inline f32x4 vmax(f32x4 a, f32x4 b) { return vmaxq_f32(a, b); }
inline f32x4 vmin(f32x4 a, f32x4 b) { return vminq_f32(a, b); }
// acc + a * b, rounded after the multiply and after the add (VMLA), like the portable path
inline f32x4 madd(f32x4 acc, f32x4 a, f32x4 b) { return vmlaq_f32(acc, a, b); }
// acc + b * a[L]
template <int L>
inline f32x4 madd_lane(f32x4 acc, f32x4 b, f32x4 a) {
  return vmlaq_lane_f32(acc, b, L < 2 ? vget_low_f32(a) : vget_high_f32(a), L & 1);
}
// Compares against zero use the bit-pattern forms of neon_a32.h (vcgezq_f32 ...), which keep the IEEE result
// for denormals; a float compare would see -1e-40 >= 0. General compares are exact unless an operand is a
// denormal: lt() also fixes lanes whose b is +-0 (clamp bounds, the select constants), so of the selections only
// one between a denormal and a nonzero denormal bound can differ from the reference.
inline uint32x4_t lt(f32x4 a, f32x4 b) {
  const uint32x4_t bz = vceqq_u32(vshlq_n_u32(vreinterpretq_u32_f32(b), 1), vdupq_n_u32(0));
  return vbslq_u32(bz, vcltzq_f32(a), vcltq_f32(a, b));
}
inline uint32x4_t gt(f32x4 a, f32x4 b) {
  const uint32x4_t bz = vceqq_u32(vshlq_n_u32(vreinterpretq_u32_f32(b), 1), vdupq_n_u32(0));
  return vbslq_u32(bz, vcgtzq_f32(a), vcgtq_f32(a, b));
}
inline f32x4 select_ge0(f32x4 v, f32x4 neg) { return vbslq_f32(vcgezq_f32(v), v, neg); }
// v > 0 ? v : 0 (NaN -> 0)
inline f32x4 relu(f32x4 v) { return vbslq_f32(vcgtzq_f32(v), v, vdupq_n_f32(0.f)); }
inline f32x4 sub(f32x4 a, f32x4 b) { return vsubq_f32(a, b); }
// a*b + c with one rounding
inline f32x4 fma_(f32x4 a, f32x4 b, f32x4 c) {
#if defined(__ARM_FEATURE_FMA)
  return vfmaq_f32(c, a, b);
#else
  if (kA32Fma) {
    asm(".fpu neon-vfpv4\n vfma.f32 %q0, %q1, %q2\n" : "+w"(c) : "w"(a), "w"(b));
    return c;
  }
  float x[4], y[4], z[4];
  vst1q_f32(x, a);
  vst1q_f32(y, b);
  vst1q_f32(z, c);
  for (int i = 0; i < 4; ++i) z[i] = std::fma(x[i], y[i], z[i]);
  return vld1q_f32(z);
#endif
}
inline f32x4 round_ne(f32x4 v) { return bf_a32_rndn(v); }
// n is integer valued: truncating conversion
inline f32x4 pow2i(f32x4 n) { return vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23)); }
inline f32x4 vabs(f32x4 v) { return vabsq_f32(v); }
inline f32x4 select_lt(f32x4 a, f32x4 b, f32x4 x, f32x4 y) { return vbslq_f32(lt(a, b), x, y); }
inline f32x4 select_gt(f32x4 a, f32x4 b, f32x4 x, f32x4 y) { return vbslq_f32(gt(a, b), x, y); }
inline float hsum(f32x4 v) { return vaddvq_f32(v); }  // (v0 + v1) + (v2 + v3)
inline float hmax(f32x4 v) { return vmaxvq_f32(v); }
inline f32x4 clamp_std(f32x4 v, f32x4 lo, f32x4 hi) {
  const f32x4 m = vbslq_f32(lt(v, lo), lo, v);
  return vbslq_f32(gt(m, hi), hi, m);  // hi < m
}
inline f32x4 sign_ge(f32x4 v) { return vbslq_f32(vcgezq_f32(v), vdupq_n_f32(1.f), vdupq_n_f32(-1.f)); }
inline f32x4 sign_zero(f32x4 v) {
  return vbslq_f32(vcgtzq_f32(v), vdupq_n_f32(1.f), vbslq_f32(vcltzq_f32(v), vdupq_n_f32(-1.f), vdupq_n_f32(0.f)));
}
#else
struct f32x4 {
  float v[4];
};
inline f32x4 load(const float* p) { return {{p[0], p[1], p[2], p[3]}}; }
inline void store(float* p, f32x4 a) {
  for (int i = 0; i < 4; ++i) p[i] = a.v[i];
}
inline f32x4 dup(float s) { return {{s, s, s, s}}; }
#define BF_FOPT_LANEWISE(expr) \
  f32x4 r;                     \
  for (int i = 0; i < 4; ++i) r.v[i] = (expr); \
  return r
inline f32x4 add(f32x4 a, f32x4 b) { BF_FOPT_LANEWISE(a.v[i] + b.v[i]); }
inline f32x4 mul(f32x4 a, f32x4 b) { BF_FOPT_LANEWISE(a.v[i] * b.v[i]); }
inline f32x4 div(f32x4 a, f32x4 b) { BF_FOPT_LANEWISE(a.v[i] / b.v[i]); }
inline f32x4 vmax(f32x4 a, f32x4 b) { BF_FOPT_LANEWISE(std::max(a.v[i], b.v[i])); }
inline f32x4 vmin(f32x4 a, f32x4 b) { BF_FOPT_LANEWISE(std::min(a.v[i], b.v[i])); }
inline f32x4 madd(f32x4 acc, f32x4 a, f32x4 b) { BF_FOPT_LANEWISE(acc.v[i] + a.v[i] * b.v[i]); }
template <int L>
inline f32x4 madd_lane(f32x4 acc, f32x4 b, f32x4 a) { BF_FOPT_LANEWISE(acc.v[i] + b.v[i] * a.v[L]); }
inline f32x4 select_ge0(f32x4 v, f32x4 neg) { BF_FOPT_LANEWISE(v.v[i] >= 0.f ? v.v[i] : neg.v[i]); }
inline f32x4 relu(f32x4 v) { BF_FOPT_LANEWISE(v.v[i] > 0.f ? v.v[i] : 0.f); }
inline f32x4 sub(f32x4 a, f32x4 b) { BF_FOPT_LANEWISE(a.v[i] - b.v[i]); }
inline f32x4 fma_(f32x4 a, f32x4 b, f32x4 c) { BF_FOPT_LANEWISE(std::fma(a.v[i], b.v[i], c.v[i])); }
inline f32x4 round_ne(f32x4 v) { BF_FOPT_LANEWISE(std::nearbyint(v.v[i])); }
inline f32x4 pow2i(f32x4 n) { BF_FOPT_LANEWISE(std::ldexp(1.f, static_cast<int>(n.v[i]))); }
inline f32x4 vabs(f32x4 v) { BF_FOPT_LANEWISE(std::fabs(v.v[i])); }
inline f32x4 select_lt(f32x4 a, f32x4 b, f32x4 x, f32x4 y) { BF_FOPT_LANEWISE(a.v[i] < b.v[i] ? x.v[i] : y.v[i]); }
inline f32x4 select_gt(f32x4 a, f32x4 b, f32x4 x, f32x4 y) { BF_FOPT_LANEWISE(a.v[i] > b.v[i] ? x.v[i] : y.v[i]); }
inline float hsum(f32x4 v) { return (v.v[0] + v.v[1]) + (v.v[2] + v.v[3]); }
inline float hmax(f32x4 v) { return std::max(std::max(v.v[0], v.v[1]), std::max(v.v[2], v.v[3])); }
inline f32x4 clamp_std(f32x4 v, f32x4 lo, f32x4 hi) { BF_FOPT_LANEWISE(std::min(std::max(v.v[i], lo.v[i]), hi.v[i])); }
inline f32x4 sign_ge(f32x4 v) { BF_FOPT_LANEWISE(v.v[i] >= 0.f ? 1.f : -1.f); }
inline f32x4 sign_zero(f32x4 v) { BF_FOPT_LANEWISE(v.v[i] > 0.f ? 1.f : (v.v[i] < 0.f ? -1.f : 0.f)); }
#undef BF_FOPT_LANEWISE
#endif

// Max with the reference's NaN behaviour: std::max(m, x) ignores a NaN x,
// as does FMAXNM.
inline f32x4 max_num(f32x4 m, f32x4 x) {
#if BF_FLOAT_OPT_NEON
  return vmaxnmq_f32(m, x);
#elif BF_FLOAT_OPT_NEON32
  return vbslq_f32(vcltq_f32(m, x), x, m);  // std::max(m, x) = m < x ? x : m (pools: a plain compare)
#else
  f32x4 r;
  for (int i = 0; i < 4; ++i) r.v[i] = std::max(m.v[i], x.v[i]);
  return r;
#endif
}

// Vector policy of the shared leaf routines (float_opt_leaf.h) for this
// 4-lane layer. Only baseline translation units use it.
struct Simd4 {
  using T = f32x4;
  static constexpr int W = 4;
  static T load(const float* p) { return fopt::load(p); }
  static void store(float* p, T v) { fopt::store(p, v); }
  static T dup(float s) { return fopt::dup(s); }
  static T add(T a, T b) { return fopt::add(a, b); }
  static T mul(T a, T b) { return fopt::mul(a, b); }
  static T div(T a, T b) { return fopt::div(a, b); }
  static T madd(T acc, T a, T b) { return fopt::madd(acc, a, b); }
  static T relu(T v) { return fopt::relu(v); }
  // std::min(std::max(v, lo), hi) exactly, including signed zeros (FMAX/FMIN
  // would turn std::max(-0, +0) = -0 into +0) and NaN (kept).
  static T clamp(T v, T lo, T hi) { return clamp_std(v, lo, hi); }
  static T select_ge0(T v, T neg) { return fopt::select_ge0(v, neg); }
  static T max_num(T m, T x) { return fopt::max_num(m, x); }
  static T sign(T v) { return fopt::sign_ge(v); }
  static T sign0(T v) { return fopt::sign_zero(v); }
  static T sub(T a, T b) { return fopt::sub(a, b); }
  static T fma(T a, T b, T c) { return fopt::fma_(a, b, c); }
  static T round(T v) { return fopt::round_ne(v); }
  static T pow2i(T n) { return fopt::pow2i(n); }
  static T min(T a, T b) { return vmin(a, b); }
  static T max(T a, T b) { return vmax(a, b); }
  static T abs(T v) { return fopt::vabs(v); }
  static T select_lt(T a, T b, T x, T y) { return fopt::select_lt(a, b, x, y); }
  static T select_gt(T a, T b, T x, T y) { return fopt::select_gt(a, b, x, y); }
  static float hsum(T v) { return fopt::hsum(v); }
  static float hmax(T v) { return fopt::hmax(v); }
};

// Fused activation on a vector; matches apply_activation() lane by lane
// (including NaN propagation: relu(NaN) = 0, relu6(NaN) = NaN).
inline f32x4 activate(Activation a, f32x4 v) {
  switch (a) {
    case Activation::kNone: return v;
    case Activation::kRelu: return relu(v);
    case Activation::kRelu6: return vmin(vmax(v, dup(0.f)), dup(6.f));
  }
  return v;
}

inline int64_t round_up(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

inline const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

// Grain for ThreadPool::parallel_for over `items` work items totalling
// `total_work` (caller's unit, e.g. MACs). The pool wakes every worker on each
// call and waits for all of them, which costs tens of µs on a phone with idle
// cores, so small jobs run inline on the caller (parallel_for does that when
// n <= grain). Larger jobs use grain 1, letting the pool cut ~4 chunks per
// thread: on big.LITTLE, a few large chunks were measured to be slower than
// either option because the caller waits for a chunk stuck on a slow core
// (ledger: float_opt step 3).
inline int64_t parallel_grain(int64_t items, int64_t total_work, int64_t inline_below) {
  return total_work < inline_below ? std::max<int64_t>(items, 1) : 1;
}

// Per-thread scratch buffers for work done inside parallel_for. A pool with N
// threads runs at most N chunk callbacks concurrently, so with N slots
// acquire() always finds a free one in its first scan. Slots are 64-byte
// aligned and allocated in prepare(); run() never allocates.
class ScratchSlots {
 public:
  void init(int slots, size_t floats_each) {
    slots_ = std::max(slots, 1);
    stride_ = static_cast<size_t>(round_up(static_cast<int64_t>(std::max<size_t>(floats_each, 1)), 16));
    buf_ = aligned_alloc_shared(stride_ * slots_ * sizeof(float));
    busy_.reset(new std::atomic<bool>[slots_]);
    for (int i = 0; i < slots_; ++i) busy_[i].store(false);
  }
  int acquire() {
    for (;;)
      for (int i = 0; i < slots_; ++i) {
        bool expected = false;
        if (busy_[i].compare_exchange_strong(expected, true, std::memory_order_acquire)) return i;
      }
  }
  void release(int i) { busy_[i].store(false, std::memory_order_release); }
  float* get(int i) const { return static_cast<float*>(buf_.get()) + stride_ * i; }
  size_t bytes() const { return stride_ * slots_ * sizeof(float); }

 private:
  int slots_ = 0;
  size_t stride_ = 0;
  std::shared_ptr<void> buf_;
  std::unique_ptr<std::atomic<bool>[]> busy_;
};

// RAII holder for one acquired slot.
class ScratchLease {
 public:
  explicit ScratchLease(ScratchSlots& s) : s_(s), i_(s.acquire()) {}
  ~ScratchLease() { s_.release(i_); }
  float* data() const { return s_.get(i_); }

 private:
  ScratchSlots& s_;
  int i_;
};

}  // namespace fopt
}  // namespace bitino
