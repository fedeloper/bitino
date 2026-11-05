// "int8_avx2" tier: dynamic-range INT8 Conv2D (groups == 1) and Dense on
// x86-64, bit-exact with the "int8_reference" tier (kernels/int8_quant.h).
// This translation unit is compiled with the baseline x86-64 flags; the
// AVX2 / AVX-VNNI code is in the leaf TUs declared in int8_avx2.h and only
// runs from kernels whose required features the CPU reports.
//
// Per run():
//   1. activation scale: AVX2 max-abs over the whole input (split over the
//      pool for >= 1M elements; max is exact, so any split gives the same
//      scale_x);
//   2. quantize + pack A: Conv2D quantizes the input tensor (AVX2 division +
//      round-to-nearest-even, as the reference) and builds im2col rows
//      (spatial padding taps = qpad); Dense quantizes its rows straight into
//      A inside the GEMM work items;
//   3. GEMM over (panel chunk, block of rows) work items with the VPDPBUSD
//      (AVX-VNNI) or VPMADDWD (AVX2) micro-kernel and a fused epilogue
//      (scale, bias, activation). Packing and GEMM share one work item per
//      row block when the weights fit one chunk; small layers run inline.
// Weights are quantized and packed once in prepare(). Two kernels per op:
//   *.int8_avx2       (avx2, VPMADDWD, int16 operands)       priority 140
//   *.int8_avx2_vnni  (avx2 + avxvnni, VPDPBUSD, u8 x s8)    priority 150
// Depthwise / grouped Conv2D is rejected in supports() so the selector falls
// back (to float_opt, or to int8_reference when allowed and float_opt is not).
#include "kernels/int8_avx2/int8_avx2.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>

#include "kernels/int8_quant.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {

#if defined(__x86_64__)
namespace int8_avx2 {
namespace {

// Quantization is memory bound; tensors below kQuantParallelMin elements are
// handled inline (a pool dispatch costs a few µs on the host).
constexpr int64_t kQuantParallelMin = 1 << 20;
constexpr int64_t kQuantPartElems = 1 << 18;  // elements per max-abs partial
constexpr int64_t kQuantMaxParts = 64;
// Packed weight bytes one work item streams; consecutive items reuse the same
// chunk, so it should stay in L2 (2 MB per P-core on Raptor Lake).
constexpr int64_t kWeightChunkBytes = 256 * 1024;
constexpr int64_t kRowBlock = 4 * kMR;  // A rows per work item
// Layers estimated below kInlineNs on one core run on the calling thread.
// Estimate (ns): 0.5 per output (epilogue) + {0.003 VNNI, 0.006 MADD} per MAC
// + 10 per im2col row.
constexpr double kInlineNs = 30e3;

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }
uint8_t* bytes_of(const std::shared_ptr<void>& p) { return static_cast<uint8_t*>(p.get()); }
float* floats_of(const std::shared_ptr<void>& p) { return static_cast<float*>(p.get()); }

// Operand representation of one variant (see int8_avx2.h).
template <bool kVnni>
struct Variant {
  using T = typename std::conditional<kVnni, uint8_t, int16_t>::type;
  static constexpr int64_t E = kVnni ? 4 : 2;  // k values per 4-byte chunk
  static void quantize(const float* x, int64_t n, float s, T* q) {
    if constexpr (kVnni) quantize_u8(x, n, s, q);
    else quantize_s16(x, n, s, q);
  }
  static T encode(int8_t q) { return static_cast<T>(kVnni ? q + 128 : q); }
  static void gemm(const GemmArgs& g, int64_t r0, int64_t rows, int64_t p0, int64_t p1) {
    if constexpr (kVnni) gemm_vnni(g, r0, rows, p0, p1);
    else gemm_madd(g, r0, rows, p0, p1);
  }
};

int64_t quant_parts(int64_t n) {
  return n < kQuantParallelMin ? 1 : std::max<int64_t>(1, std::min(kQuantMaxParts, n / kQuantPartElems));
}

// scale_x of the whole input (int8_scale(max |x|)).
float tensor_scale(const float* x, int64_t n, float* partial, int64_t parts, ThreadPool& pool) {
  if (parts == 1 || pool.num_threads() == 1) return int8_scale(maxabs(x, n));
  pool.parallel_for(parts, 1, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) partial[i] = maxabs(x + n * i / parts, n * (i + 1) / parts - n * i / parts);
  });
  float m = 0.f;
  for (int64_t i = 0; i < parts; ++i) m = std::max(m, partial[i]);
  return int8_scale(m);
}

// Packed weights, per-run scale/bias, A buffer and work decomposition shared
// by the Conv2D and Dense kernels.
template <bool kVnni>
class GemmX86 {
 public:
  using V = Variant<kVnni>;
  using T = typename V::T;

  void init(const float* w, int64_t o, int64_t k, int64_t p, int threads, bool im2col) {
    o_ = o;
    k_ = k;
    p_ = p;
    kc_ = (k + V::E - 1) / V::E;
    lda_ = 4 * kc_;
    panels_ = (o + kNR - 1) / kNR;
    std::unique_ptr<int8_t[]> wq(new int8_t[static_cast<size_t>(o * k)]);
    sw_.reset(new float[static_cast<size_t>(o)]);
    int8_quantize_rows(w, o, k, wq.get(), sw_.get());
    // W [panels][kc][16][4 bytes]; aligned_alloc_shared zero-fills the padding.
    wbuf_ = aligned_alloc_shared(static_cast<size_t>(panels_ * kc_ * 64));
    corr_ = aligned_alloc_shared(static_cast<size_t>(panels_ * kNR) * sizeof(int32_t));
    int32_t* corr = static_cast<int32_t*>(corr_.get());
    for (int64_t oc = 0; oc < o; ++oc) {
      const int8_t* row = wq.get() + oc * k;
      uint8_t* panel = bytes_of(wbuf_) + (oc / kNR) * kc_ * 64 + (oc % kNR) * 4;
      int64_t sum = 0;
      for (int64_t j = 0; j < k; ++j) {
        const int64_t c = j / V::E, t = j % V::E;
        if constexpr (kVnni) {
          panel[c * 64 + t] = static_cast<uint8_t>(row[j]);
        } else {
          const int16_t v = row[j];
          std::memcpy(panel + c * 64 + t * 2, &v, 2);
        }
        sum += row[j];
      }
      // 128 * sum_k qw modulo 2^32 (|128 * sum| can exceed 2^31 for large K;
      // the GEMM subtracts it with wrapping int32 arithmetic).
      corr[oc] = static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(128 * sum)));
    }
    abuf_ = aligned_alloc_shared(static_cast<size_t>(p * lda_ + 64));
    scale_ = aligned_alloc_shared(static_cast<size_t>(panels_ * kNR) * sizeof(float));
    bias_ = aligned_alloc_shared(static_cast<size_t>(panels_ * kNR) * sizeof(float));
    const double est_ns = 0.5 * p * o + (kVnni ? 0.003 : 0.006) * double(p) * o * k + (im2col ? 10.0 * p : 0.0);
    inline_ = threads == 1 || est_ns < kInlineNs;
    blocks_ = (p + kRowBlock - 1) / kRowBlock;
    const int64_t cache_panels = std::max<int64_t>(1, kWeightChunkBytes / (64 * kc_));
    const int64_t chunks_needed = (4 * (inline_ ? 1 : threads) + blocks_ - 1) / blocks_;
    pc_ = std::min(cache_panels, std::max<int64_t>(1, (panels_ + chunks_needed - 1) / chunks_needed));
    nchunks_ = (panels_ + pc_ - 1) / pc_;
  }

  // A row r as T elements (K padding past k stays 0).
  T* a_row(int64_t r) const { return reinterpret_cast<T*>(bytes_of(abuf_) + r * lda_); }
  size_t prepacked_bytes() const { return static_cast<size_t>(panels_ * kc_ * 64) + static_cast<size_t>(o_) * 8; }

  // pack(r0, r1) fills A rows [r0, r1); then the GEMM. When the weights fit
  // one panel chunk, each work item packs its own row block right before
  // using it (one pool dispatch, A block hot in cache).
  template <typename Pack>
  void run(const KernelContext& ctx, float sx, const float* bias, Activation act, float* y, const Pack& pack) {
    float* sc = floats_of(scale_);
    float* bi = floats_of(bias_);
    for (int64_t o = 0; o < o_; ++o) {
      sc[o] = sx * sw_[o];
      bi[o] = bias ? bias[o] : 0.f;
    }
    GemmArgs g;
    g.a = bytes_of(abuf_);
    g.lda = lda_;
    g.w = reinterpret_cast<const int8_t*>(wbuf_.get());
    g.corr = kVnni ? static_cast<const int32_t*>(corr_.get()) : nullptr;
    g.scale = sc;
    g.bias = bi;
    g.y = y;
    g.cols = o_;
    g.kc = kc_;
    g.act = act == Activation::kRelu ? 1 : act == Activation::kRelu6 ? 2 : 0;
    const int64_t blocks = blocks_, p = p_, panels = panels_, pc = pc_;
    if (nchunks_ == 1) {
      ctx.pool->parallel_for(blocks, inline_ ? blocks : 1, [&](int64_t b0, int64_t b1) {
        for (int64_t b = b0; b < b1; ++b) {
          const int64_t r0 = b * kRowBlock, r1 = std::min(p, r0 + kRowBlock);
          pack(r0, r1);
          V::gemm(g, r0, r1 - r0, 0, panels);
        }
      });
      return;
    }
    ctx.pool->parallel_for(blocks, inline_ ? blocks : 1, [&](int64_t b0, int64_t b1) { pack(b0 * kRowBlock, std::min(p, b1 * kRowBlock)); });
    const int64_t items = nchunks_ * blocks;
    ctx.pool->parallel_for(items, inline_ ? items : 1, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        const int64_t c = i / blocks, b = i % blocks, r0 = b * kRowBlock;
        V::gemm(g, r0, std::min(p, r0 + kRowBlock) - r0, c * pc, std::min(panels, (c + 1) * pc));
      }
    });
  }

 private:
  int64_t o_ = 0, k_ = 0, p_ = 0, kc_ = 0, lda_ = 0, panels_ = 0, blocks_ = 0, pc_ = 1, nchunks_ = 1;
  bool inline_ = false;
  std::unique_ptr<float[]> sw_;
  std::shared_ptr<void> wbuf_, corr_, abuf_, scale_, bias_;
};

template <bool kVnni>
class Int8Conv2DX86 : public Kernel {
 public:
  using V = Variant<kVnni>;
  using T = typename V::T;

  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    act_ = parse_activation(ctx.attrs());
    pad_value_ = static_cast<float>(ctx.attrs().get_number("pad_value", 0.0));
    gemm_.init(ctx.in(1).data<float>(), g_.oc, g_.kh * g_.kw * g_.ic, g_.n * g_.oh * g_.ow, ctx.pool->num_threads(), true);
    xlen_ = g_.n * g_.ih * g_.iw * g_.ic;
    xq_ = aligned_alloc_shared(static_cast<size_t>(xlen_) * sizeof(T));
    parts_ = quant_parts(xlen_);
    partial_ = aligned_alloc_shared(static_cast<size_t>(parts_) * sizeof(float));
    padrow_ = aligned_alloc_shared(static_cast<size_t>(g_.ic) * sizeof(T));
  }
  size_t prepacked_bytes() const override { return gemm_.prepacked_bytes(); }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    T* xq = static_cast<T*>(xq_.get());
    const float sx = tensor_scale(x, xlen_, floats_of(partial_), parts_, *ctx.pool);
    if (xlen_ < kQuantParallelMin || ctx.pool->num_threads() == 1) {
      V::quantize(x, xlen_, sx, xq);
    } else {
      ctx.pool->parallel_for(xlen_, kQuantPartElems, [&](int64_t b, int64_t e) { V::quantize(x + b, e - b, sx, xq + b); });
    }
    T* padrow = static_cast<T*>(padrow_.get());
    std::fill_n(padrow, g_.ic, V::encode(int8_quantize(pad_value_, sx)));
    gemm_.run(ctx, sx, opt_f32(ctx, 2), act_, ctx.out(0).data<float>(), [&](int64_t p0, int64_t p1) { im2col(xq, padrow, p0, p1); });
  }

 private:
  // Patch rows of output pixels [p0, p1) in (ky, kx, c) order.
  void im2col(const T* xq, const T* padrow, int64_t p0, int64_t p1) const {
    const Conv2DGeom& g = g_;
    const int64_t ic = g.ic, run = g.kw * ic;
    const size_t tap_bytes = static_cast<size_t>(ic) * sizeof(T);
    for (int64_t p = p0; p < p1; ++p) {
      T* row = gemm_.a_row(p);
      const int64_t n = p / (g.oh * g.ow), oy = (p / g.ow) % g.oh, ox = p % g.ow;
      const int64_t ix0 = ox * g.sw - g.pl;
      const bool row_inside = g.dw == 1 && ix0 >= 0 && ix0 + g.kw <= g.iw;
      for (int64_t ky = 0; ky < g.kh; ++ky) {
        const int64_t iy = oy * g.sh - g.pt + ky * g.dh;
        T* d = row + ky * run;
        if (iy < 0 || iy >= g.ih) {
          for (int64_t kx = 0; kx < g.kw; ++kx) std::memcpy(d + kx * ic, padrow, tap_bytes);
        } else if (row_inside) {
          std::memcpy(d, xq + ((n * g.ih + iy) * g.iw + ix0) * ic, static_cast<size_t>(run) * sizeof(T));
        } else {
          for (int64_t kx = 0; kx < g.kw; ++kx) {
            const int64_t ix = ix0 + kx * g.dw;
            const T* src = ix >= 0 && ix < g.iw ? xq + ((n * g.ih + iy) * g.iw + ix) * ic : padrow;
            std::memcpy(d + kx * ic, src, tap_bytes);
          }
        }
      }
    }
  }

  Conv2DGeom g_;
  Activation act_ = Activation::kNone;
  float pad_value_ = 0;
  GemmX86<kVnni> gemm_;
  int64_t xlen_ = 0, parts_ = 1;
  std::shared_ptr<void> xq_, partial_, padrow_;
};

template <bool kVnni>
class Int8DenseX86 : public Kernel {
 public:
  using V = Variant<kVnni>;

  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    const Tensor& w = ctx.in(1);
    k_ = w.dim(1);
    xlen_ = ctx.in(0).size();
    rows_ = xlen_ / k_;
    gemm_.init(w.data<float>(), w.dim(0), k_, rows_, ctx.pool->num_threads(), false);
    parts_ = quant_parts(xlen_);
    partial_ = aligned_alloc_shared(static_cast<size_t>(parts_) * sizeof(float));
  }
  size_t prepacked_bytes() const override { return gemm_.prepacked_bytes(); }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    const float sx = tensor_scale(x, xlen_, floats_of(partial_), parts_, *ctx.pool);
    const int64_t k = k_;
    gemm_.run(ctx, sx, opt_f32(ctx, 2), act_, ctx.out(0).data<float>(), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) V::quantize(x + r * k, k, sx, gemm_.a_row(r));
    });
  }

 private:
  Activation act_ = Activation::kNone;
  GemmX86<kVnni> gemm_;
  int64_t k_ = 0, rows_ = 0, xlen_ = 0, parts_ = 1;
  std::shared_ptr<void> partial_;
};

bool conv_supported(const KernelContext& ctx) {
  if (!int8_allowed(ctx)) return false;
  if (ctx.attrs().get_int("groups", 1) != 1) return false;
  const Shape& w = ctx.in(1).shape();
  return w[1] * w[2] * w[3] <= kInt8MaxK;
}

bool dense_supported(const KernelContext& ctx) { return int8_allowed(ctx) && ctx.in(1).dim(1) <= kInt8MaxK; }

}  // namespace
}  // namespace int8_avx2

void register_int8_avx2_kernels(OpRegistry& reg) {
  using namespace int8_avx2;
  const uint32_t madd = kFeatAvx2, vnni = kFeatAvx2 | kFeatAvxVnni;
  reg.add_kernel({"Conv2D", "conv2d.int8_avx2", "int8_avx2", 140, madd, conv_supported, make_kernel<Int8Conv2DX86<false>>});
  reg.add_kernel({"Conv2D", "conv2d.int8_avx2_vnni", "int8_avx2", 150, vnni, conv_supported, make_kernel<Int8Conv2DX86<true>>});
  reg.add_kernel({"Dense", "dense.int8_avx2", "int8_avx2", 140, madd, dense_supported, make_kernel<Int8DenseX86<false>>});
  reg.add_kernel({"Dense", "dense.int8_avx2_vnni", "int8_avx2", 150, vnni, dense_supported, make_kernel<Int8DenseX86<true>>});
}

#else

void register_int8_avx2_kernels(OpRegistry&) {}

#endif

}  // namespace bitino
