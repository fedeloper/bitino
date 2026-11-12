// "int8_neon" tier: dynamic-range INT8 Conv2D (groups == 1) and Dense on
// AArch64, bit-exact with the "int8_reference" tier (kernels/int8_quant.h).
//
// Per run():
//   1. quantize the input tensor: NEON max-abs reduction, then a quantize pass
//      (vdivq_f32 + vcvtnq_s32_f32 = IEEE division and round-to-nearest-even,
//      exactly what the reference computes); both passes are split over the
//      pool for tensors of >= 1M elements.
//   2. pack A: Conv2D im2col straight into the tiled layout of int8_gemm.h
//      (spatial padding taps = qpad); Dense copies rows.
//   3. GEMM over (output-channel chunk, pixel tile) work items with the SDOT
//      or SMMLA micro-kernel; fused epilogue (scale, bias, activation). Steps
//      2 and 3 share one work item per pixel tile when the weights fit one
//      chunk, and run inline when the layer is too small to amortize the pool.
// Weights are quantized and packed once in prepare(). Two kernels per op:
//   *.int8_neon_sdot  (dotprod, C = 4)   *.int8_neon_i8mm  (i8mm, C = 8,
//   higher priority). Depthwise / grouped Conv2D is rejected in supports() so
// the selector falls back.
#include "kernels/int8_neon/int8_gemm.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "kernels/int8_quant.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace bitino {

#if defined(__aarch64__)
namespace int8_neon {
namespace {

// Quantization is memory bound and a pool dispatch costs ~20-100 us on the
// S22, so tensors below kQuantParallelMin elements are quantized inline
// (ledger: "quantization threading").
constexpr int64_t kQuantParallelMin = 1 << 20;
constexpr int64_t kQuantPartElems = 1 << 18;  // elements per max-abs partial
constexpr int64_t kQuantMaxParts = 64;
// Weight bytes one GEMM work item streams for a pixel tile; consecutive work
// items reuse the same weight chunk, so it should stay cache resident.
constexpr int64_t kWeightChunkBytes = 128 * 1024;
// A pool dispatch costs ~20-100 us on the S22 and threads may land on A510
// cores, so a layer whose estimated single-thread time is below kInlineNs runs
// inline. Estimate (ns, calibrated on S22 microbench rows, ledger "inline
// threshold"): 1.25 per output (epilogue), 0.012 (SDOT) / 0.006 (SMMLA) per
// MAC, 20 per im2col pixel.
constexpr double kInlineNs = 80e3;
// Patch rows up to this many bytes are assembled on the stack (im2col).
constexpr int64_t kRowStack = 2048;

float maxabs(const float* x, int64_t n) {
  float32x4_t m0 = vdupq_n_f32(0.f), m1 = m0, m2 = m0, m3 = m0;
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x + i)));
    m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(x + i + 4)));
    m2 = vmaxq_f32(m2, vabsq_f32(vld1q_f32(x + i + 8)));
    m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(x + i + 12)));
  }
  for (; i + 4 <= n; i += 4) m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x + i)));
  float m = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
  for (; i < n; ++i) m = std::max(m, std::fabs(x[i]));
  return m;
}

// q = clamp(rne(x / scale), -127, 127). FCVTNS saturates to int32 and the
// saturating narrows to [-128, 127]; both are monotone, so clamping to -127
// at the end equals clamping the exact rounded value.
void quantize(const float* x, int64_t n, float scale, int8_t* q) {
  const float32x4_t s = vdupq_n_f32(scale);
  const int8x16_t lo = vdupq_n_s8(-127);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    int32x4_t a = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + i), s));
    int32x4_t b = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + i + 4), s));
    int32x4_t c = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + i + 8), s));
    int32x4_t d = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + i + 12), s));
    int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
    int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
    vst1q_s8(q + i, vmaxq_s8(vcombine_s8(vqmovn_s16(ab), vqmovn_s16(cd)), lo));
  }
  for (; i < n; ++i) q[i] = int8_quantize(x[i], scale);
}

}  // namespace

int64_t quant_parts(int64_t n) {
  return n < kQuantParallelMin ? 1 : std::max<int64_t>(1, std::min(kQuantMaxParts, n / kQuantPartElems));
}

float quantize_tensor(const float* x, int64_t n, int8_t* q, float* partial, int64_t parts, ThreadPool& pool) {
  if (n < kQuantParallelMin || pool.num_threads() == 1) {
    const float sx = int8_scale(maxabs(x, n));
    quantize(x, n, sx, q);
    return sx;
  }
  // max is exact and order independent, so any split gives the same scale.
  pool.parallel_for(parts, 1, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) partial[i] = maxabs(x + n * i / parts, n * (i + 1) / parts - n * i / parts);
  });
  float m = 0.f;
  for (int64_t i = 0; i < parts; ++i) m = std::max(m, partial[i]);
  const float sx = int8_scale(m);
  pool.parallel_for(n, kQuantPartElems, [&](int64_t b, int64_t e) { quantize(x + b, e - b, sx, q + b); });
  return sx;
}

namespace {

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

int8_t* bytes_of(const std::shared_ptr<void>& p) { return static_cast<int8_t*>(p.get()); }
float* floats_of(const std::shared_ptr<void>& p) { return static_cast<float*>(p.get()); }

// Copies rows [r0, r1) of a row-major [rows][k] int8 matrix into the tiled
// layout [ceil(rows/8)][kc][8][C]. Padding (rows >= rows, k >= K) is never
// written here: the destination is zeroed once at allocation.
void pack_rows(const int8_t* src, int64_t k, int64_t r0, int64_t r1, int chunk, int64_t kc, int8_t* dst) {
  for (int64_t r = r0; r < r1; ++r) {
    int8_t* tile = dst + (r / kTile) * kc * kTile * chunk + (r % kTile) * chunk;
    const int8_t* s = src + r * k;
    int64_t full = k / chunk;
    for (int64_t c = 0; c < full; ++c) std::memcpy(tile + c * kTile * chunk, s + c * chunk, chunk);
    if (full < kc) std::memcpy(tile + full * kTile * chunk, s + full * chunk, static_cast<size_t>(k - full * chunk));
  }
}

// State shared by the Conv2D and Dense kernels: packed weights, per-run scale
// and bias vectors, packed activations, work decomposition.
class Int8Gemm {
 public:
  // Quantizes and packs W[o][k]; sizes A for p rows.
  void init(const float* w, int64_t o, int64_t k, int64_t p, int chunk, int threads, bool im2col) {
    o_ = o;
    k_ = k;
    p_ = p;
    chunk_ = chunk;
    kc_ = (k + chunk - 1) / chunk;
    ob_ = (o + kTile - 1) / kTile;
    pb_ = (p + kTile - 1) / kTile;
    std::unique_ptr<int8_t[]> wq(new int8_t[static_cast<size_t>(o * k)]);
    sw_.reset(new float[static_cast<size_t>(o)]);
    int8_quantize_rows(w, o, k, wq.get(), sw_.get());
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(ob_ * kc_ * kTile * chunk));
    pack_rows(wq.get(), k, 0, o, chunk, kc_, bytes_of(wbuf_));
    abuf_ = aligned_alloc_shared(static_cast<size_t>(pb_ * kc_ * kTile * chunk));
    scale_ = aligned_alloc_shared(static_cast<size_t>(ob_ * kTile * 4));
    bias_ = aligned_alloc_shared(static_cast<size_t>(ob_ * kTile * 4));
    // Work items = (channel chunk, pixel tile), pixel tile fastest. Channel
    // chunks are capped by kWeightChunkBytes and made small enough to give
    // every thread ~4 items when there are few pixel tiles (Dense, 7x7);
    // layers estimated below kInlineNs run on the calling thread.
    const double est_ns = 1.25 * p * o + (chunk == 8 ? 0.006 : 0.012) * p * o * kc_ * chunk + (im2col ? 20.0 * p : 0.0);
    inline_ = threads == 1 || est_ns < kInlineNs;
    int64_t cache_blocks = std::max<int64_t>(1, kWeightChunkBytes / (kTile * kc_ * chunk));
    int64_t chunks_needed = (4 * (inline_ ? 1 : threads) + pb_ - 1) / pb_;
    obc_ = std::min(cache_blocks, std::max<int64_t>(1, (ob_ + chunks_needed - 1) / chunks_needed));
    nchunks_ = (ob_ + obc_ - 1) / obc_;
  }

  int8_t* a() const { return bytes_of(abuf_); }
  int64_t kc() const { return kc_; }
  int64_t pixel_tiles() const { return pb_; }
  size_t prepacked_bytes() const { return static_cast<size_t>(ob_ * kc_ * kTile * chunk_) + static_cast<size_t>(o_) * 4; }

  // Packs A (pack(r0, r1) writes rows [r0, r1)) and runs the GEMM. When all
  // weights fit one channel chunk, each work item packs its own pixel tiles
  // right before using them (one pool dispatch, A tile hot in L1); otherwise
  // A is packed in a separate parallel pass.
  template <typename Pack>
  void run(const KernelContext& ctx, float sx, const float* bias, Activation act, float* y, const Pack& pack) {
    GemmArgs g = args(sx, bias, act, y);
    auto tiles = chunk_ == 8 ? gemm_tiles_i8mm : gemm_tiles_sdot;
    const int64_t pb = pb_, obc = obc_, ob = ob_, p = p_;
    // Grain = item count runs the whole loop inline (parallel_for, n <= grain).
    if (nchunks_ == 1) {
      ctx.pool->parallel_for(pb, inline_ ? pb : 1, [&](int64_t b0, int64_t b1) {
        for (int64_t b = b0; b < b1; ++b) {
          pack(b * kTile, std::min(p, (b + 1) * kTile));
          tiles(g, b, 0, ob);
        }
      });
      return;
    }
    ctx.pool->parallel_for(pb, inline_ ? pb : 1, [&](int64_t b0, int64_t b1) { pack(b0 * kTile, std::min(p, b1 * kTile)); });
    ctx.pool->parallel_for(nchunks_ * pb, inline_ ? nchunks_ * pb : 1, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        int64_t c = i / pb;
        tiles(g, i % pb, c * obc, std::min(ob, (c + 1) * obc));
      }
    });
  }

 private:
  GemmArgs args(float sx, const float* bias, Activation act, float* y) {
    float* sc = floats_of(scale_);
    float* bi = floats_of(bias_);
    for (int64_t o = 0; o < o_; ++o) {
      sc[o] = sx * sw_[o];
      bi[o] = bias ? bias[o] : 0.f;
    }
    GemmArgs g;
    g.a = a();
    g.w = bytes_of(wbuf_);
    g.scale = sc;
    g.bias = bi;
    g.y = y;
    g.rows = p_;
    g.cols = o_;
    g.kc = kc_;
    g.act = act == Activation::kRelu ? 1 : act == Activation::kRelu6 ? 2 : 0;
    return g;
  }

  int64_t o_ = 0, k_ = 0, p_ = 0, kc_ = 0, ob_ = 0, pb_ = 0, obc_ = 1, nchunks_ = 1;
  bool inline_ = false;
  int chunk_ = 4;
  std::unique_ptr<float[]> sw_;
  std::shared_ptr<void> wbuf_, abuf_, scale_, bias_;
};

template <int kChunk>
class Int8Conv2DNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    act_ = parse_activation(ctx.attrs());
    pad_value_ = static_cast<float>(ctx.attrs().get_number("pad_value", 0.0));
    gemm_.init(ctx.in(1).data<float>(), g_.oc, g_.kh * g_.kw * g_.ic, g_.n * g_.oh * g_.ow, kChunk, ctx.pool->num_threads(), true);
    xlen_ = g_.n * g_.ih * g_.iw * g_.ic;
    xq_ = aligned_alloc_shared(static_cast<size_t>(xlen_));
    parts_ = quant_parts(xlen_);
    partial_ = aligned_alloc_shared(static_cast<size_t>(parts_ * 4));
    padrow_ = aligned_alloc_shared(static_cast<size_t>(g_.ic));
  }
  size_t prepacked_bytes() const override { return gemm_.prepacked_bytes(); }

  void run(const KernelContext& ctx) override {
    int8_t* xq = bytes_of(xq_);
    const float sx = quantize_tensor(ctx.in(0).data<float>(), xlen_, xq, floats_of(partial_), parts_, *ctx.pool);
    int8_t* padrow = bytes_of(padrow_);
    std::memset(padrow, int8_quantize(pad_value_, sx), static_cast<size_t>(g_.ic));
    gemm_.run(ctx, sx, opt_f32(ctx, 2), act_, ctx.out(0).data<float>(),
              [&](int64_t p0, int64_t p1) { im2col(xq, padrow, p0, p1); });
  }

 private:
  // Writes the patch rows of output pixels [p0, p1) into the tiled A buffer.
  void im2col(const int8_t* xq, const int8_t* padrow, int64_t p0, int64_t p1) const {
    const Conv2DGeom& g = g_;
    const int64_t kc = gemm_.kc(), ic = g.ic;
    int8_t* abuf = gemm_.a();
    const bool aligned = ic % kChunk == 0;
    const int64_t kreal = g.kh * g.kw * ic;
    if (!aligned && kc * kChunk <= kRowStack) {
      // Unaligned channels (e.g. RGB input): build the patch row contiguously
      // (one memcpy per kernel row when it is fully inside), then copy chunks.
      int8_t buf[kRowStack];
      std::memset(buf + kreal, 0, static_cast<size_t>(kc * kChunk - kreal));
      const int64_t run = g.kw * ic;
      for (int64_t p = p0; p < p1; ++p) {
        int64_t n = p / (g.oh * g.ow), oy = (p / g.ow) % g.oh, ox = p % g.ow;
        int64_t ix0 = ox * g.sw - g.pl;
        bool row_inside = g.dw == 1 && ix0 >= 0 && ix0 + g.kw <= g.iw;
        for (int64_t ky = 0; ky < g.kh; ++ky) {
          int64_t iy = oy * g.sh - g.pt + ky * g.dh;
          int8_t* d = buf + ky * run;
          if (iy < 0 || iy >= g.ih) {
            std::memset(d, padrow[0], static_cast<size_t>(run));
          } else if (row_inside) {
            std::memcpy(d, xq + ((n * g.ih + iy) * g.iw + ix0) * ic, static_cast<size_t>(run));
          } else {
            for (int64_t kx = 0; kx < g.kw; ++kx) {
              int64_t ix = ix0 + kx * g.dw;
              const int8_t* src = ix >= 0 && ix < g.iw ? xq + ((n * g.ih + iy) * g.iw + ix) * ic : padrow;
              std::memcpy(d + kx * ic, src, static_cast<size_t>(ic));
            }
          }
        }
        int8_t* row = abuf + (p / kTile) * kc * kTile * kChunk + (p % kTile) * kChunk;
        for (int64_t c = 0; c < kc; ++c) std::memcpy(row + c * kTile * kChunk, buf + c * kChunk, kChunk);
      }
      return;
    }
    for (int64_t p = p0; p < p1; ++p) {
      int8_t* row = abuf + (p / kTile) * kc * kTile * kChunk + (p % kTile) * kChunk;
      int64_t n = p / (g.oh * g.ow), oy = (p / g.ow) % g.oh, ox = p % g.ow;
      int64_t k = 0;
      for (int64_t ky = 0; ky < g.kh; ++ky) {
        int64_t iy = oy * g.sh - g.pt + ky * g.dh;
        for (int64_t kx = 0; kx < g.kw; ++kx, k += ic) {
          int64_t ix = ox * g.sw - g.pl + kx * g.dw;
          bool inside = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw;
          const int8_t* src = inside ? xq + ((n * g.ih + iy) * g.iw + ix) * ic : padrow;
          if (aligned) {
            int8_t* d = row + (k / kChunk) * kTile * kChunk;
            for (int64_t c = 0; c < ic; c += kChunk, d += kTile * kChunk) std::memcpy(d, src + c, kChunk);
          } else {
            for (int64_t c = 0; c < ic; ++c) row[((k + c) / kChunk) * kTile * kChunk + (k + c) % kChunk] = src[c];
          }
        }
      }
    }
  }

  Conv2DGeom g_;
  Activation act_ = Activation::kNone;
  float pad_value_ = 0;
  Int8Gemm gemm_;
  int64_t xlen_ = 0, parts_ = 1;
  std::shared_ptr<void> xq_, partial_, padrow_;
};

template <int kChunk>
class Int8DenseNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    const Tensor& w = ctx.in(1);
    k_ = w.dim(1);
    xlen_ = ctx.in(0).size();
    rows_ = xlen_ / k_;
    gemm_.init(w.data<float>(), w.dim(0), k_, rows_, kChunk, ctx.pool->num_threads(), false);
    xq_ = aligned_alloc_shared(static_cast<size_t>(xlen_));
    parts_ = quant_parts(xlen_);
    partial_ = aligned_alloc_shared(static_cast<size_t>(parts_ * 4));
  }
  size_t prepacked_bytes() const override { return gemm_.prepacked_bytes(); }

  void run(const KernelContext& ctx) override {
    int8_t* xq = bytes_of(xq_);
    const float sx = quantize_tensor(ctx.in(0).data<float>(), xlen_, xq, floats_of(partial_), parts_, *ctx.pool);
    const int64_t k = k_, kc = gemm_.kc();
    int8_t* a = gemm_.a();
    gemm_.run(ctx, sx, opt_f32(ctx, 2), act_, ctx.out(0).data<float>(),
              [&](int64_t r0, int64_t r1) { pack_rows(xq, k, r0, r1, kChunk, kc, a); });
  }

 private:
  Activation act_ = Activation::kNone;
  Int8Gemm gemm_;
  int64_t k_ = 0, rows_ = 0, xlen_ = 0, parts_ = 1;
  std::shared_ptr<void> xq_, partial_;
};

bool conv_supported(const KernelContext& ctx) {
  if (!int8_allowed(ctx)) return false;
  const Json& a = ctx.attrs();
  if (a.get_int("groups", 1) != 1) return false;
  const Shape& w = ctx.in(1).shape();
  return w[1] * w[2] * w[3] <= kInt8MaxK;
}

bool dense_supported(const KernelContext& ctx) { return int8_allowed(ctx) && ctx.in(1).dim(1) <= kInt8MaxK; }

}  // namespace
}  // namespace int8_neon

void register_int8_neon_kernels(OpRegistry& reg) {
  using namespace int8_neon;
  const uint32_t sdot = kFeatNeon | kFeatDotProd, i8mm = sdot | kFeatI8mm;
  reg.add_kernel({"Conv2D", "conv2d.int8_neon_sdot", "int8_neon", 140, sdot, conv_supported, make_kernel<Int8Conv2DNeon<4>>});
  reg.add_kernel({"Conv2D", "conv2d.int8_neon_i8mm", "int8_neon", 150, i8mm, conv_supported, make_kernel<Int8Conv2DNeon<8>>});
  reg.add_kernel({"Dense", "dense.int8_neon_sdot", "int8_neon", 140, sdot, dense_supported, make_kernel<Int8DenseNeon<4>>});
  reg.add_kernel({"Dense", "dense.int8_neon_i8mm", "int8_neon", 150, i8mm, dense_supported, make_kernel<Int8DenseNeon<8>>});
}

#elif defined(__ARM_NEON) && defined(__arm__)

// 32-bit ARM: the AArch32 kernels of int8_neon_a32.cc (VMULL.S8 tiles, no dot products).
void register_int8_neon_a32_kernels(OpRegistry& reg);
void register_int8_neon_kernels(OpRegistry& reg) { register_int8_neon_a32_kernels(reg); }

#else

void register_int8_neon_kernels(OpRegistry&) {}

#endif

}  // namespace bitino
