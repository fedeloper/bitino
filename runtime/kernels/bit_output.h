// Bit-packed outputs of binary layers (BConv2D / BDense attribute "output": "bit").
//
// The producer binarizes its own final float output v (after post_mul,
// post_add, activation and a fused epilogue, if any) exactly like the next
// layer would binarize it as f32 input:
//   bit = 1 (-1) iff !(v >= t[c])   output_binarize "ge" (default)
//   bit = 1 (-1) iff !(v >  t[c])   output_binarize "gt"
// with t = the f32[C] node input named by the attribute
// "output_threshold_input" (an input index), or 0 without it. This is
// pack_row_f32() of the float output, so a consumer reading the bits sees
// exactly the binary values it would compute from the float tensor (NaN -> 1,
// -0 like +0). The plan-time rewrite in src/graph_rewrite.cc sets it when the
// float output has a single binary consumer (docs/operators.md).
//
// Kernels keep the bit-output paths in their own functions; this header holds
// the shared parsing and the row packing used after a float tile is final.
#pragma once

#include <cstdint>
#include <climits>
#include <cstring>
#include <memory>
#include <vector>

#include "bitino/op.h"
#include "kernels/bitpack.h"
#include "ops/ops_common.h"

namespace bitino {

struct BitOutput {
  bool active = false;
  BinarizeMode mode = BinarizeMode::kGe;  // kGe or kGt
  int thr_input = -1;                     // node input with the per-channel threshold, -1 = threshold 0

  void parse(const KernelContext& ctx) {
    const Json& a = ctx.attrs();
    active = a.get_string("output", "f32") == "bit";
    mode = a.get_string("output_binarize", "ge") == "gt" ? BinarizeMode::kGt : BinarizeMode::kGe;
    thr_input = static_cast<int>(a.get_int("output_threshold_input", -1));
  }
  const float* threshold(const KernelContext& ctx) const {
    return thr_input >= 0 && ctx.has_in(static_cast<size_t>(thr_input)) ? ctx.in(static_cast<size_t>(thr_input)).data<float>() : nullptr;
  }
  // The legacy contract (ge against 0): kernels that only implement that reject other nodes.
  bool plain() const { return mode == BinarizeMode::kGe && thr_input < 0; }
  bool gt() const { return mode == BinarizeMode::kGt; }
};

// Bit of one output value (scalar paths).
inline bool output_bit(float v, BinarizeMode m, float t) { return m == BinarizeMode::kGt ? !(v > t) : !(v >= t); }

// Integer form of a bit output without epilogue (LCE-style fused thresholds).
// The stored bit is a function of the integer dot product d alone:
//   v(d) = act(fl(fl(float(d) * mul[o]) + add[o])),  bit(d) = !(v >= t) (ge) / !(v > t) (gt),
// and every step is monotone in d (rounding is monotone; relu / relu6 too), so over
// the reachable d in [-k, k] the set {bit = 1} is {d < lim} (neg = 0) or {d > lim}
// (neg = -1), possibly empty or full. build() evaluates the kernels' exact float
// expression for every such d (-ffp-contract=off, separate mul and add, as the
// SIMD epilogues) and derives lim / neg per channel; a channel whose bit is not of
// that form (only with NaN scales, e.g. inf * 0) makes build() fail and the kernel
// keeps its float epilogue. The result is cached on the parameter values, so a
// run only pays a memcmp of mul / add / threshold.
class IntThresholds {
 public:
  // mul / add / thr: per output channel (nullptr: 1 / -0 / 0), as the kernels apply them.
  bool update(int64_t k, int64_t oc, const float* mul, const float* add, Activation act, BinarizeMode mode, const float* thr) {
    const bool present[3] = {mul != nullptr, add != nullptr, thr != nullptr};
    const float* p[3] = {mul, add, thr};
    bool hit = valid_ && k == k_ && act == act_ && mode == mode_ && oc == oc_;
    for (int i = 0; i < 3 && hit; ++i)
      hit = present[i] == present_[i] &&
            (!present[i] || std::memcmp(key_.data() + i * oc, p[i], static_cast<size_t>(oc) * sizeof(float)) == 0);
    if (hit) return ok_;
    const float def[3] = {1.f, -0.f, 0.f};  // what an absent mul / add / threshold stands for in the kernels
    key_.resize(static_cast<size_t>(3 * oc));
    for (int i = 0; i < 3; ++i) {
      present_[i] = present[i];
      for (int64_t o = 0; o < oc; ++o) key_[i * oc + o] = present[i] ? p[i][o] : def[i];
    }
    valid_ = true;
    k_ = k;
    oc_ = oc;
    act_ = act;
    mode_ = mode;
    ok_ = build(k, oc);
    ++generation_;
    return ok_;
  }
  bool ok() const { return valid_ && ok_; }
  const int32_t* lim() const { return lim_.data(); }
  const int32_t* neg() const { return neg_.data(); }  // 0 or -1 (all bits set)
  // Changes whenever update() rebuilt the thresholds (kernels re-copy their padded arrays only then).
  uint64_t generation() const { return generation_; }

 private:
  bool build(int64_t k, int64_t oc) {
    lim_.assign(static_cast<size_t>(oc), 0);
    neg_.assign(static_cast<size_t>(oc), 0);
    for (int64_t o = 0; o < oc; ++o) {
      const float m = key_[o], a = key_[oc + o], t = key_[2 * oc + o];
      auto bit = [&](int64_t d) {
        float v = static_cast<float>(d);
        v = v * m;
        v = v + a;
        return output_bit(apply_activation(act_, v), mode_, t);
      };
      const bool first = bit(-k);
      int64_t flips = 0, at = 0;
      bool prev = first;
      for (int64_t d = -k + 1; d <= k; ++d) {
        const bool b = bit(d);
        if (b != prev) {
          ++flips;
          at = d;
          prev = b;
        }
      }
      if (flips == 0) {
        lim_[o] = first ? INT32_MAX : INT32_MIN;  // d < INT32_MAX: always 1; d < INT32_MIN: never
      } else if (flips == 1) {
        lim_[o] = static_cast<int32_t>(first ? at : at - 1);  // 1 for d < at  /  1 for d > at - 1
        neg_[o] = first ? 0 : -1;
      } else {
        return false;
      }
    }
    return true;
  }

  bool valid_ = false, ok_ = false;
  bool present_[3] = {false, false, false};
  int64_t k_ = 0, oc_ = 0;
  uint64_t generation_ = 0;
  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  std::vector<float> key_;
  std::vector<int32_t> lim_, neg_;
};

// Packs float rows [r0, r1) (row r at src + r * c) into bit rows (row r at
// dst + r * bit_words(c)); padding bits of each last word are cleared.
inline void pack_output_rows(const float* src, int64_t r0, int64_t r1, int64_t c, uint32_t* dst, BinarizeMode m, const float* thr) {
  const int64_t words = bit_words(c);
  for (int64_t r = r0; r < r1; ++r) pack_row_f32(src + r * c, c, dst + r * words, m, thr);
}

}  // namespace bitino
