// x86-64 AVX2 BConv2D / BDense ("bconv2d.avx2", "bdense.avx2", tier
// binary_avx2). Baseline-ISA part: everything except the SIMD code, which is
// in binary_avx2_simd.cc (see binary_avx2.h for the split and the data
// layouts). The measurements behind the design are in
// the development log (not published)
//
// BConv2D pipeline per run() (same structure as bconv2d.neon_opt):
//  1. f32 input -> bits (AVX2 compares), or per-group repacking when the
//     group width needs it (BConvGroupLayout); bit input is used in place.
//  2. Work items = (pixel tile) x (output-channel chunk). Each task owns a
//     contiguous item range and a private patch buffer, so im2col rows are
//     built per tile and group and stay in L1. Channel chunks are only split
//     when pixel tiles alone cannot feed ~4 tasks per thread.
//  3. Micro-kernel: 8 output channels x 4 pixels, 64 bits per step,
//     popcount(a XOR w) with VPSHUFB nibble lookups.
//  4. Epilogue: dot = K - 2*popcount (+ border padding correction), separate
//     float mul and add (no FMA, like the reference), activation, f32 store
//     or bit packing.
#include <algorithm>

#include "kernels/bconv_plan.h"
#include "kernels/binary_avx2/binary_avx2.h"
#include "kernels/float_opt/eltwise_program.h"
#include "ops/op_defs.h"

namespace bitino {

#if defined(__x86_64__)
namespace {

constexpr uint32_t kAvx2Features = kFeatAvx2 | kFeatPopcnt;
constexpr int64_t kMaxTaps = 64;  // border tap masks are uint64
// Inputs up to this many floats are packed on the calling thread.
constexpr int64_t kPackSerialFloats = 128 * 1024;

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

int act_code(Activation a) { return a == Activation::kRelu ? 1 : a == Activation::kRelu6 ? 2 : 0; }

// Interleaves `rows` dense rows (row r at src + r * src_stride, kwp words,
// kwp even) into lane quads [rows_pad/4][kwp/2 steps][4][2 words]. `row_of`
// maps a padded row index to a source row (or -1 for a zero row).
template <typename RowOf>
void interleave_quads(const uint32_t* src, int64_t src_stride, int64_t rows_pad, int64_t kwp, RowOf row_of, uint32_t* dst) {
  const int64_t steps = kwp / 2;
  for (int64_t p = 0; p < rows_pad; ++p) {
    const int64_t r = row_of(p);
    for (int64_t s = 0; s < steps; ++s)
      for (int h = 0; h < 2; ++h) dst[(((p / 4) * steps + s) * 4 + p % 4) * 2 + h] = r < 0 ? 0u : src[r * src_stride + 2 * s + h];
  }
}

// Packs a whole f32 tensor [pix][c] into bits [pix][dst_words] (dst_words >= bit_words(c)).
void pack_tensor(const KernelContext& ctx, const float* x, int64_t pix, int64_t c, uint32_t* dst, int64_t dst_words, bool gt,
                 const float* thr) {
  if (c % 32 == 0 && !thr && dst_words == c / 32) {
    // Contiguous: one flat run of words.
    const int64_t total = pix * dst_words;
    ctx.pool->parallel_for((total + 63) / 64, kPackSerialFloats / (64 * 32), [&](int64_t b0, int64_t b1) {
      const int64_t w0 = b0 * 64, w1 = std::min(total, b1 * 64);
      avx2::pack_words(x + w0 * 32, w1 - w0, dst + w0, gt, nullptr);
    });
    return;
  }
  ctx.pool->parallel_for(pix, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(c, 1)), [&](int64_t p0, int64_t p1) {
    for (int64_t p = p0; p < p1; ++p) avx2::pack_row(x + p * c, c, dst + p * dst_words, gt, thr);
  });
}

// ------------------------------------------------------------------ BConv2D
class BConv2DAvx2 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Json& a = ctx.attrs();
    g_ = conv_geometry(a, ctx.in(0).shape(), ctx.in(1).shape());
    L_.init(g_);
    act_ = parse_activation(a);
    mode_ = parse_binarize(a);
    pad_value_ = static_cast<int>(a.get_number("pad_value", 0.0));
    bout_.parse(ctx);
    bit_out_ = bout_.active;
    bit_in_ = ctx.in(0).dtype() == DType::kBit;
    taps_ = L_.taps;
    kwp_ = (L_.k_words + 1) / 2 * 2;  // rows padded to 64 bits
    steps_ = kwp_ / 2;
    ocg_ = L_.ocg;
    ocg_pad_ = (ocg_ + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    oc_pad_ = g_.groups * ocg_pad_;
    pixels_ = g_.n * g_.oh * g_.ow;
    out_words_ = bit_words(g_.oc);

    const uint32_t* w = ctx.in(1).data<uint32_t>();
    std::vector<uint32_t> dense(static_cast<size_t>(g_.oc * kwp_));
    L_.pack_weight_rows(w, g_.oc, dense.data(), kwp_);
    const int64_t ocg = ocg_, ocgp = ocg_pad_;
    auto row_of = [ocg, ocgp](int64_t p) {
      const int64_t gi = p / ocgp, j = p % ocgp;
      return j < ocg ? gi * ocg + j : int64_t(-1);
    };
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(oc_pad_ * kwp_ * 4));
    interleave_quads(dense.data(), kwp_, oc_pad_, kwp_, row_of, static_cast<uint32_t*>(wbuf_.get()));
    // Per-tap channel sums by padded channel index, [tap][oc_pad].
    tap_sum_.assign(static_cast<size_t>(taps_ * oc_pad_), 0);
    for (int64_t p = 0; p < oc_pad_; ++p) {
      const int64_t o = row_of(p);
      if (o >= 0)
        for (int64_t t = 0; t < taps_; ++t) tap_sum_[t * oc_pad_ + p] = L_.tap_sum(w, o, t);
    }
    mul_.assign(static_cast<size_t>(oc_pad_), 1.f);
    add_.assign(static_cast<size_t>(oc_pad_), -0.f);  // -0 is the exact additive identity

    if (!bit_in_ || L_.regroup) xbits_ = aligned_alloc_shared(static_cast<size_t>(g_.n * g_.ih * g_.iw * L_.src_words * 4));

    // Pixel tile: about 16 KB of patch rows, a multiple of the micro-tile rows.
    const int64_t tp = avx2::kRowBlock;
    mt_ = std::max<int64_t>(tp, std::min<int64_t>(64, (16384 / (kwp_ * 4)) / tp * tp));
    n_ptiles_ = (pixels_ + mt_ - 1) / mt_;
    plan_chunks(ctx.pool->num_threads());
    items_ = n_ptiles_ * static_cast<int64_t>(chunks_.size());
    const int64_t threads = ctx.pool->num_threads();
    n_tasks_ = threads > 1 ? std::min<int64_t>(items_, 4 * threads) : 1;
    slot_words_ = mt_ * kwp_;
    patches_ = aligned_alloc_shared(static_cast<size_t>(n_tasks_ * slot_words_ * 4));
    masks_.assign(static_cast<size_t>(n_tasks_ * mt_), 0);
    epi_.prepare_epilogue(ctx, g_.oh * g_.ow);
    prepare_bit_output();
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(oc_pad_ * kwp_ * 4 + tap_sum_.size() * 4); }

  void run(const KernelContext& ctx) override {
    const float* thr = opt_f32(ctx, 4);
    const bool gt = mode_ == BinarizeMode::kGt;
    const uint32_t* xb;
    if (bit_in_ && !L_.regroup) {
      xb = ctx.in(0).data<uint32_t>();
    } else {
      auto* dst = static_cast<uint32_t*>(xbits_.get());
      const int64_t pix = g_.n * g_.ih * g_.iw, sw = L_.src_words;
      if (!L_.regroup) {
        pack_tensor(ctx, ctx.in(0).data<float>(), pix, g_.ic, dst, sw, gt, thr);
      } else {
        const BConvGroupLayout& lay = L_;
        const int64_t ic = g_.ic;
        const uint32_t* xin = bit_in_ ? ctx.in(0).data<uint32_t>() : nullptr;
        const float* x = bit_in_ ? nullptr : ctx.in(0).data<float>();
        ctx.pool->parallel_for(pix, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(ic, 1)), [&](int64_t p0, int64_t p1) {
          for (int64_t p = p0; p < p1; ++p) {
            if (xin) {
              regroup_bit_row(xin + p * bit_words(ic), lay.groups, lay.kic, dst + p * sw);
            } else {
              for (int64_t gi = 0; gi < lay.groups; ++gi)
                avx2::pack_row(x + p * ic + gi * lay.kic, lay.kic, dst + p * sw + gi * lay.ut, gt, thr ? thr + gi * lay.kic : nullptr);
            }
          }
        });
      }
      xb = dst;
    }
    // Post-scale by padded channel index.
    const float* m = opt_f32(ctx, 2);
    const float* d = opt_f32(ctx, 3);
    for (int64_t gi = 0; gi < g_.groups; ++gi) {
      if (m) std::copy_n(m + gi * ocg_, ocg_, mul_.data() + gi * ocg_pad_);
      if (d) std::copy_n(d + gi * ocg_, ocg_, add_.data() + gi * ocg_pad_);
    }
    avx2::TileArgs ta;
    ta.wts = static_cast<const uint8_t*>(wbuf_.get());
    ta.steps = steps_;
    ta.k_bits = static_cast<int32_t>(L_.k_bits);
    ta.tap_sum = tap_sum_.data();
    ta.oc_pad = oc_pad_;
    ta.pad_scale = pad_value_ - 1;
    ta.mul = mul_.data();
    ta.add = add_.data();
    ta.act = act_code(act_);
    ta.yf = direct_bits_ ? nullptr : (bit_out_ ? static_cast<float*>(yf_.get()) : ctx.out(0).data<float>());
    ta.yb = direct_bits_ ? ctx.out(0).data<uint32_t>() : nullptr;
    ta.y_stride = direct_bits_ ? out_words_ : g_.oc;
    ta.oc_total = g_.oc;
    ta.bit_rmw = bit_rmw_;
    if (direct_bits_) set_output_threshold(ctx, ta);
    if (epi_.active()) epi_.begin(ctx);
    ctx.pool->parallel_for(n_tasks_, 1, [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) run_task(ctx, ta, xb, t);
    });
    // Fused epilogue (and bit output after it): applied per pixel tile inside the tasks when every task
    // computes whole output rows (one channel chunk), else after them.
    if (epi_.active() && chunks_.size() > 1) {
      float* y = ta.yf;
      ctx.pool->parallel_for(n_ptiles_, 1, [&](int64_t b0, int64_t b1) {
        const int64_t q0 = b0 * mt_, q1 = std::min(pixels_, b1 * mt_);
        epi_.apply(y, q0, q1);
        if (bit_out_) pack_float_rows(ctx, q0, q1);
      });
    }
  }

 private:
  // ---- bit output (kernels/bit_output.h). Without an epilogue the tile
  // epilogue binarizes each 8-channel block against the output threshold
  // (direct_bits_); with one, the float rows go to yf_, the epilogue runs on
  // them and pack_float_rows() binarizes the finished rows.
  void prepare_bit_output() {
    direct_bits_ = bit_out_ && !epi_.active();
    bit_rmw_ = direct_bits_ && g_.groups > 1 && ocg_ % 8 != 0;
    if (direct_bits_) thr_.assign(static_cast<size_t>(oc_pad_), 0.f);
    if (bit_out_ && !direct_bits_) yf_ = aligned_alloc_shared(static_cast<size_t>(pixels_ * g_.oc * 4));
  }
  // Per run (direct bit output): integer thresholds on the dot product when they represent the float
  // epilogue exactly (IntThresholds), else the output threshold for the float compare.
  void set_output_threshold(const KernelContext& ctx, avx2::TileArgs& ta) {
    const float* t = bout_.threshold(ctx);
    if (ithr_.update(L_.k_bits, g_.oc, opt_f32(ctx, 2), opt_f32(ctx, 3), act_, bout_.mode, t)) {
      if (ithr_.generation() != ithr_gen_) {
        lim_.assign(static_cast<size_t>(oc_pad_), INT32_MIN);  // padded channels: bit 0
        neg_.assign(static_cast<size_t>(oc_pad_), 0);
        for (int64_t gi = 0; gi < g_.groups; ++gi) {
          std::copy_n(ithr_.lim() + gi * ocg_, ocg_, lim_.data() + gi * ocg_pad_);
          std::copy_n(ithr_.neg() + gi * ocg_, ocg_, neg_.data() + gi * ocg_pad_);
        }
        ithr_gen_ = ithr_.generation();
      }
      ta.out_lim = lim_.data();
      ta.out_neg = neg_.data();
      return;
    }
    if (t) {
      for (int64_t gi = 0; gi < g_.groups; ++gi) std::copy_n(t + gi * ocg_, ocg_, thr_.data() + gi * ocg_pad_);
      ta.out_thr = thr_.data();
    }
    ta.out_gt = bout_.gt();
  }
  void pack_float_rows(const KernelContext& ctx, int64_t q0, int64_t q1) const {
    const float* y = static_cast<const float*>(yf_.get());
    const float* thr = bout_.threshold(ctx);
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    for (int64_t q = q0; q < q1; ++q) avx2::pack_row(y + q * g_.oc, g_.oc, yb + q * out_words_, bout_.gt(), thr);
  }

  // Output-channel chunks [c0, c1). Boundaries are group boundaries or
  // multiples of 8 channels inside a group (micro-tile width), and multiples
  // of 32 for bit output so that no two tasks write the same output word.
  void plan_chunks(int64_t threads) {
    chunks_.clear();
    int64_t want = 1;
    if (threads > 1 && n_ptiles_ < 4 * threads) want = std::min((4 * threads + n_ptiles_ - 1) / n_ptiles_, (g_.oc + 7) / 8);
    if (want <= 1) {
      chunks_.push_back({0, g_.oc});
      return;
    }
    const int64_t target = (g_.oc + want - 1) / want;
    int64_t c0 = 0;
    for (int64_t c = 1; c < g_.oc; ++c) {
      const bool boundary = (c % ocg_) % 8 == 0 && (!bit_out_ || c % 32 == 0);
      if (boundary && c - c0 >= target) {
        chunks_.push_back({c0, c});
        c0 = c;
      }
    }
    chunks_.push_back({c0, g_.oc});
  }

  void build_patches(const uint32_t* xb, int64_t gi, int64_t p0, int64_t p1, uint32_t* rows, uint64_t* mask) const {
    int64_t ox = p0 % g_.ow, oy = (p0 / g_.ow) % g_.oh, n = p0 / (g_.ow * g_.oh);
    const auto* src = reinterpret_cast<const uint8_t*>(xb);
    for (int64_t p = p0; p < p1; ++p) {
      mask[p - p0] = build_patch_row(g_, L_, src, gi, n, oy, ox, reinterpret_cast<uint8_t*>(rows + (p - p0) * kwp_));
      if (++ox == g_.ow) {
        ox = 0;
        if (++oy == g_.oh) {
          oy = 0;
          ++n;
        }
      }
    }
  }

  void run_task(const KernelContext& ctx, const avx2::TileArgs& ta, const uint32_t* xb, int64_t task) {
    const int64_t nch = static_cast<int64_t>(chunks_.size());
    const int64_t i0 = task * items_ / n_tasks_, i1 = (task + 1) * items_ / n_tasks_;
    // Patch rows beyond the group's k_units units are never written and keep
    // their initial zeros (all groups share one layout).
    uint32_t* rows = static_cast<uint32_t*>(patches_.get()) + task * slot_words_;
    uint64_t* mask = masks_.data() + task * mt_;
    int64_t cur_pt = -1, cur_g = -1;
    for (int64_t it = i0; it < i1; ++it) {
      const int64_t pt = it / nch;
      const Chunk ch = chunks_[it % nch];
      const int64_t p0 = pt * mt_, p1 = std::min(pixels_, p0 + mt_);
      if (bit_rmw_) {
        const int64_t w0 = ch.c0 >> 5, w1 = (ch.c1 + 31) >> 5;
        for (int64_t q = p0; q < p1; ++q) std::fill(ta.yb + q * out_words_ + w0, ta.yb + q * out_words_ + w1, 0u);
      }
      for (int64_t gi = ch.c0 / ocg_; gi * ocg_ < ch.c1; ++gi) {
        if (pt != cur_pt || gi != cur_g) {
          build_patches(xb, gi, p0, p1, rows, mask);
          cur_pt = pt;
          cur_g = gi;
        }
        const int64_t j0 = std::max<int64_t>(0, ch.c0 - gi * ocg_);
        const int64_t j1 = std::min(ocg_, ch.c1 - gi * ocg_);
        const int64_t j1p = (j1 + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
        avx2::bconv_tile(ta, rows, mask, p0, p1 - p0, gi * ocg_pad_, gi * ocg_, j0, j1p, ocg_);
      }
      if (nch == 1 && epi_.active()) {  // rows still in cache
        epi_.apply(ta.yf, p0, p1);
        if (bit_out_) pack_float_rows(ctx, p0, p1);
      }
    }
  }

  struct Chunk {
    int64_t c0, c1;
  };

  Conv2DGeom g_;
  BConvGroupLayout L_;
  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  int pad_value_ = 0;
  bool bit_out_ = false, bit_in_ = false, bit_rmw_ = false, direct_bits_ = false;
  int64_t taps_ = 0, kwp_ = 0, steps_ = 0, ocg_ = 0, ocg_pad_ = 0, oc_pad_ = 0, pixels_ = 0, out_words_ = 0;
  int64_t mt_ = 4, n_ptiles_ = 0, items_ = 0, n_tasks_ = 1, slot_words_ = 0;
  std::vector<Chunk> chunks_;
  std::shared_ptr<void> wbuf_, xbits_, patches_;
  std::shared_ptr<void> yf_;  // [pixels][oc] float output before binarization (bit output + epilogue)
  std::vector<int32_t> tap_sum_;
  std::vector<uint64_t> masks_;
  std::vector<float> mul_, add_;
  std::vector<float> thr_;  // output threshold by padded channel index (direct bit output)
  std::vector<int32_t> lim_, neg_;  // integer thresholds by padded channel index (direct bit output)
  IntThresholds ithr_;
  uint64_t ithr_gen_ = ~uint64_t(0);
  BitOutput bout_;
  EltwiseProgram epi_;
};

bool bconv_avx2_supported(const KernelContext& ctx) {
  if (!bconv_packed_grouped_supported(ctx)) return false;
  const Shape& w = ctx.in(1).shape();
  return w[1] * w[2] <= kMaxTaps;
}

// ------------------------------------------------------------------- BDense
// A 1x1 convolution over rows without padding: same micro-kernel and epilogue.
class BDenseAvx2 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    mode_ = parse_binarize(ctx.attrs());
    k_ = ctx.in(1).dim(1);
    on_ = ctx.in(1).dim(0);
    words_ = bit_words(k_);
    kwp_ = (words_ + 1) / 2 * 2;
    rows_ = ctx.in(0).size() / k_;
    on_pad_ = (on_ + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    const int64_t on = on_;
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * kwp_ * 4));
    std::vector<uint32_t> dense(static_cast<size_t>(on_ * kwp_), 0u);
    const uint32_t* w = ctx.in(1).data<uint32_t>();
    for (int64_t o = 0; o < on_; ++o) std::copy_n(w + o * words_, words_, dense.data() + o * kwp_);
    interleave_quads(dense.data(), kwp_, on_pad_, kwp_, [on](int64_t p) { return p < on ? p : int64_t(-1); },
                     static_cast<uint32_t*>(wbuf_.get()));
    xbuf_ = aligned_alloc_shared(static_cast<size_t>(rows_ * kwp_ * 4));  // padding words stay 0
    mul_.assign(static_cast<size_t>(on_pad_), 1.f);
    add_.assign(static_cast<size_t>(on_pad_), -0.f);
    // Row tiles x channel chunks; chunks split only when row tiles cannot
    // feed ~4 tasks per thread.
    const int64_t threads = ctx.pool->num_threads();
    rt_ = std::min<int64_t>(rows_, 64);
    n_rt_ = (rows_ + rt_ - 1) / rt_;
    occ_ = on_pad_;
    if (threads > 1 && n_rt_ < 4 * threads) {
      const int64_t want = std::min((4 * threads + n_rt_ - 1) / n_rt_, on_pad_ / avx2::kOcBlock);
      occ_ = ((on_pad_ + want - 1) / want + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    }
    n_occ_ = (on_pad_ + occ_ - 1) / occ_;
    // >= ~16K popcount steps per pool chunk: smaller GEMVs run on the calling
    // thread (a pool dispatch costs a few microseconds).
    const int64_t item_steps = std::max<int64_t>(1, rt_ * (occ_ / 4) * (kwp_ / 2));
    grain_ = std::max<int64_t>(1, 16384 / item_steps);
    const Tensor& y = ctx.out(0);
    epi_.prepare_epilogue(ctx, y.dim(0) > 0 ? rows_ / y.dim(0) : rows_);
    bout_.parse(ctx);
    if (bout_.active) yf_ = aligned_alloc_shared(static_cast<size_t>(rows_ * on_ * 4));
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(on_pad_ * kwp_ * 4); }

  void run(const KernelContext& ctx) override {
    const float* thr = opt_f32(ctx, 4);
    auto* xb = static_cast<uint32_t*>(xbuf_.get());
    if (ctx.in(0).dtype() == DType::kF32) {
      pack_tensor(ctx, ctx.in(0).data<float>(), rows_, k_, xb, kwp_, mode_ == BinarizeMode::kGt, thr);
    } else {
      const uint32_t* x = ctx.in(0).data<uint32_t>();
      for (int64_t r = 0; r < rows_; ++r) std::copy_n(x + r * words_, words_, xb + r * kwp_);
    }
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, on_, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, on_, add_.data());
    avx2::TileArgs ta;
    ta.wts = static_cast<const uint8_t*>(wbuf_.get());
    ta.steps = kwp_ / 2;
    ta.k_bits = static_cast<int32_t>(k_);
    ta.oc_pad = on_pad_;
    ta.mul = mul_.data();
    ta.add = add_.data();
    ta.act = act_code(act_);
    // Bit output: without an epilogue, integer thresholds on the dot product when they represent the
    // float epilogue exactly (bits stored by the tile, one byte per 8-channel block); otherwise the float
    // rows (and the epilogue) go to yf_, then each finished row tile is binarized.
    const bool int_bits = bout_.active && !epi_.active() &&
                          ithr_.update(k_, on_, opt_f32(ctx, 2), opt_f32(ctx, 3), act_, bout_.mode, bout_.threshold(ctx));
    if (int_bits) {
      if (ithr_.generation() != ithr_gen_) {
        lim_.assign(static_cast<size_t>(on_pad_), INT32_MIN);  // padded channels: bit 0
        neg_.assign(static_cast<size_t>(on_pad_), 0);
        std::copy_n(ithr_.lim(), on_, lim_.data());
        std::copy_n(ithr_.neg(), on_, neg_.data());
        ithr_gen_ = ithr_.generation();
      }
      ta.out_lim = lim_.data();
      ta.out_neg = neg_.data();
      ta.yb = ctx.out(0).data<uint32_t>();
      ta.y_stride = bit_words(on_);
    } else {
      ta.yf = bout_.active ? static_cast<float*>(yf_.get()) : ctx.out(0).data<float>();
      ta.y_stride = on_;
    }
    ta.oc_total = on_;
    const int64_t n_items = n_rt_ * n_occ_;
    if (epi_.active()) epi_.begin(ctx);
    ctx.pool->parallel_for(n_items, grain_, [&](int64_t i0, int64_t i1) {
      for (int64_t it = i0; it < i1; ++it) {
        const int64_t r0 = (it / n_occ_) * rt_, c0 = (it % n_occ_) * occ_;
        const int64_t nr = std::min(rt_, rows_ - r0), c1 = std::min(on_pad_, c0 + occ_);
        avx2::bconv_tile(ta, xb + r0 * kwp_, nullptr, r0, nr, 0, 0, c0, c1, on_);
        if (n_occ_ == 1 && !int_bits) {  // rows still in cache
          if (epi_.active()) epi_.apply(ta.yf, r0, r0 + nr);
          if (bout_.active) pack_float_rows(ctx, ta.yf, r0, r0 + nr);
        }
      }
    });
    // Fused epilogue / bit output: per row tile inside the items with one channel chunk, else after them.
    if ((epi_.active() || bout_.active) && n_occ_ > 1 && !int_bits)
      ctx.pool->parallel_for(n_rt_, 1, [&](int64_t t0, int64_t t1) {
        const int64_t r0 = t0 * rt_, r1 = std::min(rows_, t1 * rt_);
        if (epi_.active()) epi_.apply(ta.yf, r0, r1);
        if (bout_.active) pack_float_rows(ctx, ta.yf, r0, r1);
      });
  }

 private:
  // Bit output (kernels/bit_output.h): binarizes finished float rows [r0, r1).
  void pack_float_rows(const KernelContext& ctx, const float* y, int64_t r0, int64_t r1) const {
    const float* thr = bout_.threshold(ctx);
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    const int64_t ow = bit_words(on_);
    for (int64_t r = r0; r < r1; ++r) avx2::pack_row(y + r * on_, on_, yb + r * ow, bout_.gt(), thr);
  }

  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  int64_t k_ = 0, on_ = 0, words_ = 0, kwp_ = 0, rows_ = 0, on_pad_ = 0;
  int64_t rt_ = 1, n_rt_ = 1, occ_ = 8, n_occ_ = 1, grain_ = 1;
  std::shared_ptr<void> wbuf_, xbuf_;
  std::shared_ptr<void> yf_;  // [rows][on] float output before binarization (bit output)
  std::vector<float> mul_, add_;
  BitOutput bout_;
  IntThresholds ithr_;
  uint64_t ithr_gen_ = ~uint64_t(0);
  std::vector<int32_t> lim_, neg_;  // integer thresholds [on_pad] (bit output without epilogue)
  EltwiseProgram epi_;
};

bool bdense_avx2_supported(const KernelContext& ctx) { return ctx.attrs().get_string("input_binarize", "ge") != "sign_zero"; }

// ------------------------------------------------------------ ternary BDense
// sign_zero BDense (x in {-1,0,+1}, w = ±1 or ternary with w_zero) on two
// bit-planes per operand; same contract and exactness argument as
// bdense.ternary_bitplane (ternary_scalar/ternary_kernels.cc).
class BDenseTernaryAvx2 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    const Tensor& w = ctx.in(1);
    on_ = w.dim(0);
    k_ = w.dim(1);
    words_ = bit_words(k_);
    kwp_ = (words_ + 1) / 2 * 2;
    rows_ = ctx.in(0).size() / k_;
    on_pad_ = (on_ + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    const int64_t on = on_;
    auto row_of = [on](int64_t p) { return p < on ? p : int64_t(-1); };
    std::vector<uint32_t> dense(static_cast<size_t>(on_ * kwp_), 0u);
    for (int64_t o = 0; o < on_; ++o) std::copy_n(w.data<uint32_t>() + o * words_, words_, dense.data() + o * kwp_);
    wneg_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * kwp_ * 4));
    interleave_quads(dense.data(), kwp_, on_pad_, kwp_, row_of, static_cast<uint32_t*>(wneg_.get()));
    if (ctx.has_in(5)) {
      // nz = NOT zero-mask on the k valid bits; padding bits stay 0.
      const uint32_t* wz = ctx.in(5).data<uint32_t>();
      for (int64_t o = 0; o < on_; ++o)
        for (int64_t q = 0; q < words_; ++q) {
          const int64_t valid = k_ - q * 32;
          const uint32_t lane = valid >= 32 ? ~0u : ((1u << valid) - 1);
          dense[o * kwp_ + q] = ~wz[o * words_ + q] & lane;
        }
      wnz_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * kwp_ * 4));
      interleave_quads(dense.data(), kwp_, on_pad_, kwp_, row_of, static_cast<uint32_t*>(wnz_.get()));
    }
    xnz_ = aligned_alloc_shared(static_cast<size_t>(rows_ * kwp_ * 4));  // padding words stay 0
    xneg_ = aligned_alloc_shared(static_cast<size_t>(rows_ * kwp_ * 4));
    mul_.assign(static_cast<size_t>(on_pad_), 1.f);
    add_.assign(static_cast<size_t>(on_pad_), -0.f);
    const int64_t threads = ctx.pool->num_threads();
    rt_ = std::min<int64_t>(rows_, 64);
    n_rt_ = (rows_ + rt_ - 1) / rt_;
    occ_ = on_pad_;
    if (threads > 1 && n_rt_ < 4 * threads) {
      const int64_t want = std::min((4 * threads + n_rt_ - 1) / n_rt_, on_pad_ / avx2::kOcBlock);
      occ_ = ((on_pad_ + want - 1) / want + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    }
    n_occ_ = (on_pad_ + occ_ - 1) / occ_;
    const int64_t item_steps = std::max<int64_t>(1, rt_ * (occ_ / 4) * (kwp_ / 2) * (wnz_ ? 2 : 1));
    grain_ = std::max<int64_t>(1, 16384 / item_steps);
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(on_pad_ * kwp_ * 4 * (wnz_ ? 2 : 1)); }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    const float* thr = opt_f32(ctx, 4);
    auto* xn = static_cast<uint32_t*>(xnz_.get());
    auto* xs = static_cast<uint32_t*>(xneg_.get());
    const int64_t k = k_, kwp = kwp_;
    ctx.pool->parallel_for(rows_, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(k, 1)), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) avx2::pack_ternary_row(x + r * k, k, xn + r * kwp, xs + r * kwp, thr);
    });
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, on_, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, on_, add_.data());
    avx2::TernaryArgs ta;
    ta.wneg = static_cast<const uint8_t*>(wneg_.get());
    ta.wnz = static_cast<const uint8_t*>(wnz_.get());
    ta.steps = kwp_ / 2;
    ta.mul = mul_.data();
    ta.add = add_.data();
    ta.act = act_code(act_);
    ta.y = ctx.out(0).data<float>();
    ta.y_stride = on_;
    ctx.pool->parallel_for(n_rt_ * n_occ_, grain_, [&](int64_t i0, int64_t i1) {
      for (int64_t it = i0; it < i1; ++it) {
        const int64_t r0 = (it / n_occ_) * rt_, c0 = (it % n_occ_) * occ_;
        const int64_t nr = std::min(rt_, rows_ - r0), c1 = std::min(on_pad_, c0 + occ_);
        avx2::ternary_tile(ta, xn + r0 * kwp_, xs + r0 * kwp_, r0, nr, c0, c1, on_);
      }
    });
  }

 private:
  Activation act_ = Activation::kNone;
  int64_t on_ = 0, k_ = 0, words_ = 0, kwp_ = 0, rows_ = 0, on_pad_ = 0;
  int64_t rt_ = 1, n_rt_ = 1, occ_ = 8, n_occ_ = 1, grain_ = 1;
  std::shared_ptr<void> wneg_, wnz_, xnz_, xneg_;
  std::vector<float> mul_, add_;
};

// --------------------------------------------- QBConv2D / QBDense (k-bit x binary)
// DoReFa k-bit activations (k <= 4) as k bit-planes, binary weights; see
// binary_avx2.h. Zero padding of the quantized input = zero bits in every
// plane, so no border correction is needed (QBConv2D requires pad_value 0).
float dorefa_levels(const KernelContext& ctx) { return static_cast<float>((1 << ctx.attrs().get_int("act_bits", 0)) - 1); }

class QBConv2DAvx2 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Json& a = ctx.attrs();
    g_ = conv_geometry(a, ctx.in(0).shape(), ctx.in(1).shape());
    L_.init(g_);  // groups == 1 (supports()): 32-bit units, or 16-bit units when ic % 32 == 16
    act_ = parse_activation(a);
    planes_ = static_cast<int>(a.get_int("act_bits", 0));
    n_ = dorefa_levels(ctx);
    kwp_ = (L_.k_words + 1) / 2 * 2;
    oc_pad_ = (g_.oc + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    pixels_ = g_.n * g_.oh * g_.ow;
    const int64_t oc = g_.oc;
    std::vector<uint32_t> dense(static_cast<size_t>(g_.oc * kwp_));
    L_.pack_weight_rows(ctx.in(1).data<uint32_t>(), g_.oc, dense.data(), kwp_);
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(oc_pad_ * kwp_ * 4));
    interleave_quads(dense.data(), kwp_, oc_pad_, kwp_, [oc](int64_t p) { return p < oc ? p : int64_t(-1); },
                     static_cast<uint32_t*>(wbuf_.get()));
    mul_.assign(static_cast<size_t>(oc_pad_), 1.f);
    add_.assign(static_cast<size_t>(oc_pad_), -0.f);
    in_pix_ = g_.n * g_.ih * g_.iw;
    xplanes_ = aligned_alloc_shared(static_cast<size_t>(planes_ * in_pix_ * L_.src_words * 4));
    const int64_t row_bytes = planes_ * kwp_ * 4;
    mt_ = std::max<int64_t>(avx2::kRowBlock, std::min<int64_t>(64, (16384 / row_bytes) / avx2::kRowBlock * avx2::kRowBlock));
    n_ptiles_ = (pixels_ + mt_ - 1) / mt_;
    const int64_t threads = ctx.pool->num_threads();
    occ_ = oc_pad_;
    if (threads > 1 && n_ptiles_ < 4 * threads) {
      const int64_t want = std::min((4 * threads + n_ptiles_ - 1) / n_ptiles_, oc_pad_ / avx2::kOcBlock);
      occ_ = ((oc_pad_ + want - 1) / want + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    }
    n_occ_ = (oc_pad_ + occ_ - 1) / occ_;
    items_ = n_ptiles_ * n_occ_;
    n_tasks_ = threads > 1 ? std::min<int64_t>(items_, 4 * threads) : 1;
    slot_words_ = mt_ * planes_ * kwp_;
    patches_ = aligned_alloc_shared(static_cast<size_t>(n_tasks_ * slot_words_ * 4));  // row tails stay 0
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(oc_pad_ * kwp_ * 4); }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    auto* xp = static_cast<uint32_t*>(xplanes_.get());
    const int64_t ic = g_.ic, sw = L_.src_words, ps = in_pix_ * sw;
    const float n = n_;
    const int planes = planes_;
    ctx.pool->parallel_for(in_pix_, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(ic, 1)), [&](int64_t p0, int64_t p1) {
      for (int64_t p = p0; p < p1; ++p) avx2::pack_levels_row(x + p * ic, ic, n, planes, xp + p * sw, ps);
    });
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, g_.oc, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, g_.oc, add_.data());
    avx2::QTileArgs ta;
    ta.wts = static_cast<const uint8_t*>(wbuf_.get());
    ta.steps = kwp_ / 2;
    ta.planes = planes_;
    ta.mul = mul_.data();
    ta.add = add_.data();
    ta.act = act_code(act_);
    ta.y = ctx.out(0).data<float>();
    ta.y_stride = g_.oc;
    ctx.pool->parallel_for(n_tasks_, 1, [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) run_task(ta, xp, t);
    });
  }

 private:
  void run_task(const avx2::QTileArgs& ta, const uint32_t* xp, int64_t task) {
    const int64_t i0 = task * items_ / n_tasks_, i1 = (task + 1) * items_ / n_tasks_;
    uint32_t* rows = static_cast<uint32_t*>(patches_.get()) + task * slot_words_;
    const int64_t plane_words = in_pix_ * L_.src_words;
    int64_t cur = -1;
    for (int64_t it = i0; it < i1; ++it) {
      const int64_t pt = it / n_occ_, c0 = (it % n_occ_) * occ_;
      const int64_t p0 = pt * mt_, p1 = std::min(pixels_, p0 + mt_);
      if (pt != cur) {
        int64_t ox = p0 % g_.ow, oy = (p0 / g_.ow) % g_.oh, n = p0 / (g_.ow * g_.oh);
        for (int64_t p = p0; p < p1; ++p) {
          for (int b = 0; b < planes_; ++b)
            build_patch_row(g_, L_, reinterpret_cast<const uint8_t*>(xp + b * plane_words), 0, n, oy, ox,
                            reinterpret_cast<uint8_t*>(rows + ((p - p0) * planes_ + b) * kwp_));
          if (++ox == g_.ow) {
            ox = 0;
            if (++oy == g_.oh) {
              oy = 0;
              ++n;
            }
          }
        }
        cur = pt;
      }
      avx2::qbinary_tile(ta, rows, p0, p1 - p0, c0, std::min(oc_pad_, c0 + occ_), g_.oc);
    }
  }

  Conv2DGeom g_;
  BConvGroupLayout L_;
  Activation act_ = Activation::kNone;
  int planes_ = 0;
  float n_ = 1.f;
  int64_t kwp_ = 0, oc_pad_ = 0, pixels_ = 0, in_pix_ = 0;
  int64_t mt_ = 4, n_ptiles_ = 0, occ_ = 8, n_occ_ = 1, items_ = 0, n_tasks_ = 1, slot_words_ = 0;
  std::shared_ptr<void> wbuf_, xplanes_, patches_;
  std::vector<float> mul_, add_;
};

bool qbconv_avx2_supported(const KernelContext& ctx) {
  const Json& a = ctx.attrs();
  return a.get_int("groups", 1) == 1 && a.get_int("act_bits", 0) >= 1 && a.get_int("act_bits", 0) <= avx2::kMaxQPlanes;
}

class QBDenseAvx2 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    planes_ = static_cast<int>(ctx.attrs().get_int("act_bits", 0));
    n_ = dorefa_levels(ctx);
    k_ = ctx.in(1).dim(1);
    on_ = ctx.in(1).dim(0);
    words_ = bit_words(k_);
    kwp_ = (words_ + 1) / 2 * 2;
    rows_ = ctx.in(0).size() / k_;
    on_pad_ = (on_ + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    const int64_t on = on_;
    std::vector<uint32_t> dense(static_cast<size_t>(on_ * kwp_), 0u);
    const uint32_t* w = ctx.in(1).data<uint32_t>();
    for (int64_t o = 0; o < on_; ++o) std::copy_n(w + o * words_, words_, dense.data() + o * kwp_);
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(on_pad_ * kwp_ * 4));
    interleave_quads(dense.data(), kwp_, on_pad_, kwp_, [on](int64_t p) { return p < on ? p : int64_t(-1); },
                     static_cast<uint32_t*>(wbuf_.get()));
    xbuf_ = aligned_alloc_shared(static_cast<size_t>(rows_ * planes_ * kwp_ * 4));  // padding words stay 0
    mul_.assign(static_cast<size_t>(on_pad_), 1.f);
    add_.assign(static_cast<size_t>(on_pad_), -0.f);
    const int64_t threads = ctx.pool->num_threads();
    rt_ = std::min<int64_t>(rows_, 64);
    n_rt_ = (rows_ + rt_ - 1) / rt_;
    occ_ = on_pad_;
    if (threads > 1 && n_rt_ < 4 * threads) {
      const int64_t want = std::min((4 * threads + n_rt_ - 1) / n_rt_, on_pad_ / avx2::kOcBlock);
      occ_ = ((on_pad_ + want - 1) / want + avx2::kOcBlock - 1) / avx2::kOcBlock * avx2::kOcBlock;
    }
    n_occ_ = (on_pad_ + occ_ - 1) / occ_;
    const int64_t item_steps = std::max<int64_t>(1, rt_ * (occ_ / 4) * (kwp_ / 2) * planes_);
    grain_ = std::max<int64_t>(1, 16384 / item_steps);
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>(on_pad_ * kwp_ * 4); }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    auto* xb = static_cast<uint32_t*>(xbuf_.get());
    const int64_t k = k_, rw = planes_ * kwp_;
    const float n = n_;
    const int planes = planes_;
    const int64_t kwp = kwp_;
    ctx.pool->parallel_for(rows_, std::max<int64_t>(1, kPackSerialFloats / std::max<int64_t>(k, 1)), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) avx2::pack_levels_row(x + r * k, k, n, planes, xb + r * rw, kwp);
    });
    if (const float* m = opt_f32(ctx, 2)) std::copy_n(m, on_, mul_.data());
    if (const float* d = opt_f32(ctx, 3)) std::copy_n(d, on_, add_.data());
    avx2::QTileArgs ta;
    ta.wts = static_cast<const uint8_t*>(wbuf_.get());
    ta.steps = kwp_ / 2;
    ta.planes = planes_;
    ta.mul = mul_.data();
    ta.add = add_.data();
    ta.act = act_code(act_);
    ta.y = ctx.out(0).data<float>();
    ta.y_stride = on_;
    ctx.pool->parallel_for(n_rt_ * n_occ_, grain_, [&](int64_t i0, int64_t i1) {
      for (int64_t it = i0; it < i1; ++it) {
        const int64_t r0 = (it / n_occ_) * rt_, c0 = (it % n_occ_) * occ_;
        const int64_t nr = std::min(rt_, rows_ - r0), c1 = std::min(on_pad_, c0 + occ_);
        avx2::qbinary_tile(ta, xb + r0 * rw, r0, nr, c0, c1, on_);
      }
    });
  }

 private:
  Activation act_ = Activation::kNone;
  int planes_ = 0;
  float n_ = 1.f;
  int64_t k_ = 0, on_ = 0, words_ = 0, kwp_ = 0, rows_ = 0, on_pad_ = 0;
  int64_t rt_ = 1, n_rt_ = 1, occ_ = 8, n_occ_ = 1, grain_ = 1;
  std::shared_ptr<void> wbuf_, xbuf_;
  std::vector<float> mul_, add_;
};

bool qbdense_avx2_supported(const KernelContext& ctx) {
  const int64_t k = ctx.attrs().get_int("act_bits", 0);
  return k >= 1 && k <= avx2::kMaxQPlanes;
}

// Same domain as bdense.ternary_bitplane: f32 input with sign_zero binarization.
bool bdense_ternary_avx2_supported(const KernelContext& ctx) {
  return ctx.in(0).dtype() == DType::kF32 && ctx.attrs().get_string("input_binarize", "ge") == "sign_zero" && !has_epilogue(ctx) &&
         ctx.attrs().get_string("output", "f32") == "f32";
}

}  // namespace

void register_binary_avx2_kernels(OpRegistry& reg) {
  reg.add_kernel({"BConv2D", "bconv2d.avx2", "binary_avx2", 40, kAvx2Features, bconv_avx2_supported, make_kernel<BConv2DAvx2>});
  reg.add_kernel({"BDense", "bdense.avx2", "binary_avx2", 30, kAvx2Features, bdense_avx2_supported, make_kernel<BDenseAvx2>});
  reg.add_kernel({"BDense", "bdense.ternary_avx2", "binary_avx2", 35, kAvx2Features, bdense_ternary_avx2_supported,
                  make_kernel<BDenseTernaryAvx2>});
  reg.add_kernel({"QBConv2D", "qbconv2d.avx2", "binary_avx2", 30, kAvx2Features, qbconv_avx2_supported, make_kernel<QBConv2DAvx2>});
  reg.add_kernel({"QBDense", "qbdense.avx2", "binary_avx2", 30, kAvx2Features, qbdense_avx2_supported, make_kernel<QBDenseAvx2>});
}

#else

void register_binary_avx2_kernels(OpRegistry&) {}

#endif

}  // namespace bitino
