// float_opt elementwise and pooling kernels: Add (+fused activation),
// ChannelAffine, Activation (relu, relu6, hardtanh, prelu, leaky_relu; other
// types fall back to the reference kernel), MaxPool2D, AvgPool2D,
// GlobalAvgPool, and Pad (data movement only, one kernel for all backends).
//
// The arithmetic is the shared leaf code of float_opt_leaf.h, vectorized over
// 4 floats (NEON / portable) or 8 (AVX2, float_opt_avx2.cc) with a scalar
// tail; this file splits the work over threads in large contiguous ranges
// (elementwise) or output rows (pools).
// Add, ChannelAffine, Activation and MaxPool2D use exactly the reference's
// float operations (no FMA), so they are bit-identical to it. The average
// pools sum in float instead of double (see tests for the tolerance).
#include <algorithm>
#include <cstring>
#include <vector>

#include "kernels/float_opt/float_opt.h"
#include "kernels/float_opt/float_opt_leaf.h"
#include "kernels/float_opt/float_simd.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

#if defined(__x86_64__)
#include "kernels/float_opt/float_opt_avx2.h"
#endif

namespace bitino {
namespace {

using namespace fopt;

// Elements per parallel work item for streaming ops: large enough that the
// thread-pool dispatch (~µs) is amortized, small enough to balance 4 threads
// on mid-sized tensors.
constexpr int64_t kElemGrain = 16384;

// Leaf entry points per backend.
struct Simd4Ew {
  static constexpr int kLanes = Simd4::W;
  static void add(const float* a, const float* b, float* y, int64_t i0, int64_t i1, int32_t act) { leaf_add<Simd4>(a, b, y, i0, i1, act); }
  static void affine_rows(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t r0, int64_t r1) {
    leaf_affine_rows<Simd4>(x, s, b, y, c, r0, r1);
  }
  static void activation(int32_t t, const float* x, float* y, int64_t i0, int64_t i1, float lo, float hi, float al) {
    leaf_activation<Simd4>(t, x, y, i0, i1, lo, hi, al);
  }
  static void prelu_rows(const float* x, const float* sl, float* y, int64_t c, int64_t r0, int64_t r1) {
    leaf_prelu_rows<Simd4>(x, sl, y, c, r0, r1);
  }
  static void pool_rows(const PoolArgs& g, int64_t r0, int64_t r1) { leaf_pool_rows<Simd4>(g, r0, r1); }
  static void gap(const float* x, float* y, int64_t hw, int64_t c, int64_t t0, int64_t t1) {
    leaf_global_avg_pool<Simd4>(x, y, hw, c, t0, t1);
  }
};

#if defined(__x86_64__)
struct Avx2Ew {
  static constexpr int kLanes = fopt_avx2::kLanes;
  static void add(const float* a, const float* b, float* y, int64_t i0, int64_t i1, int32_t act) { fopt_avx2::add(a, b, y, i0, i1, act); }
  static void affine_rows(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t r0, int64_t r1) {
    fopt_avx2::affine_rows(x, s, b, y, c, r0, r1);
  }
  static void activation(int32_t t, const float* x, float* y, int64_t i0, int64_t i1, float lo, float hi, float al) {
    fopt_avx2::activation(t, x, y, i0, i1, lo, hi, al);
  }
  static void prelu_rows(const float* x, const float* sl, float* y, int64_t c, int64_t r0, int64_t r1) {
    fopt_avx2::prelu_rows(x, sl, y, c, r0, r1);
  }
  static void pool_rows(const PoolArgs& g, int64_t r0, int64_t r1) { fopt_avx2::pool_rows(g, r0, r1); }
  static void gap(const float* x, float* y, int64_t hw, int64_t c, int64_t t0, int64_t t1) { fopt_avx2::global_avg_pool(x, y, hw, c, t0, t1); }
};
#endif

// Runs fn(begin, end) over [0, n) in parallel chunks of kElemGrain elements.
template <typename Fn>
void parallel_elements(ThreadPool* pool, int64_t n, const Fn& fn) {
  const int64_t blocks = (n + kElemGrain - 1) / kElemGrain;
  pool->parallel_for(blocks, 1, [&](int64_t b0, int64_t b1) { fn(b0 * kElemGrain, std::min(n, b1 * kElemGrain)); });
}

// Runs fn(row_begin, row_end) over rows of `c` channels with ~kElemGrain
// elements per work item.
template <typename Fn>
void parallel_rows(ThreadPool* pool, int64_t rows, int64_t c, const Fn& fn) {
  const int64_t grain = std::max<int64_t>(1, kElemGrain / std::max<int64_t>(c, 1));
  const int64_t blocks = (rows + grain - 1) / grain;
  pool->parallel_for(blocks, 1, [&](int64_t b0, int64_t b1) { fn(b0 * grain, std::min(rows, b1 * grain)); });
}

// -------------------------------------------------------------------- Add
template <class B>
class AddOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { act_ = static_cast<int32_t>(parse_activation(ctx.attrs())); }
  void run(const KernelContext& ctx) override {
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    const int32_t act = act_;
    parallel_elements(ctx.pool, ctx.in(0).size(), [&](int64_t i0, int64_t i1) { B::add(a, b, y, i0, i1, act); });
  }

 private:
  int32_t act_ = kActNone;
};

// ---------------------------------------------------------- ChannelAffine
template <class B>
class ChannelAffineOpt : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& xt = ctx.in(0);
    const int64_t c = xt.dim(-1), rows = xt.size() / c;
    const float* s = opt_f32(ctx, 1);
    const float* b = opt_f32(ctx, 2);
    const float* x = xt.data<float>();
    float* y = ctx.out(0).data<float>();
    parallel_rows(ctx.pool, rows, c, [&](int64_t r0, int64_t r1) { B::affine_rows(x, s, b, y, c, r0, r1); });
  }
};

// ------------------------------------------------------------- Activation
enum class ActType { kRelu, kRelu6, kHardtanh, kPrelu, kLeakyRelu };

bool parse_act_type(const std::string& t, ActType* out) {
  if (t == "relu") *out = ActType::kRelu;
  else if (t == "relu6") *out = ActType::kRelu6;
  else if (t == "hardtanh") *out = ActType::kHardtanh;
  else if (t == "prelu") *out = ActType::kPrelu;
  else if (t == "leaky_relu") *out = ActType::kLeakyRelu;
  else return false;
  return true;
}

template <class B>
class ActivationOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    parse_act_type(ctx.attrs().get_string("type", ""), &type_);
    lo_ = static_cast<float>(ctx.attrs().get_number("min", -1.0));
    hi_ = static_cast<float>(ctx.attrs().get_number("max", 1.0));
    alpha_ = static_cast<float>(ctx.attrs().get_number("alpha", 0.01));
    if (type_ == ActType::kRelu6) {
      lo_ = 0.f;
      hi_ = 6.f;
    }
  }
  void run(const KernelContext& ctx) override {
    const Tensor& xt = ctx.in(0);
    const float* x = xt.data<float>();
    float* y = ctx.out(0).data<float>();
    if (type_ == ActType::kPrelu) {
      const int64_t c = xt.dim(-1);
      const float* slope = ctx.in(1).data<float>();
      parallel_rows(ctx.pool, xt.size() / c, c, [&](int64_t r0, int64_t r1) { B::prelu_rows(x, slope, y, c, r0, r1); });
      return;
    }
    const int32_t type = type_ == ActType::kRelu ? kActOpRelu : type_ == ActType::kLeakyRelu ? kActOpLeaky : kActOpClamp;
    const float lo = lo_, hi = hi_, alpha = alpha_;
    parallel_elements(ctx.pool, xt.size(), [&](int64_t i0, int64_t i1) { B::activation(type, x, y, i0, i1, lo, hi, alpha); });
  }

 private:
  ActType type_ = ActType::kRelu;
  float lo_ = -1, hi_ = 1, alpha_ = 0.01f;
};

bool activation_supported(const KernelContext& ctx) {
  ActType t;
  return parse_act_type(ctx.attrs().get_string("type", ""), &t);
}

// ------------------------------------------------------------------ pools
// AvgPool2DPrelu (internal, src/graph_rewrite.h pool_prelu): input 1 = the prelu slope, output 0 = the average of
// prelu(x), output 1 (optional) = the average of x, in one pass over x.
template <class B, bool kMax>
class Pool2DOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = pool_geometry(ctx.attrs(), ctx.in(0).shape());
    include_pad_ = ctx.attrs().get_int("count_include_pad", 0) != 0;
  }
  void run(const KernelContext& ctx) override {
    const Pool2DGeom& g = g_;
    PoolArgs a;
    a.x = ctx.in(0).data<float>();
    a.y = ctx.out(0).data<float>();
    a.ih = g.ih;
    a.iw = g.iw;
    a.c = g.c;
    a.oh = g.oh;
    a.ow = g.ow;
    a.kh = g.kh;
    a.kw = g.kw;
    a.sh = g.sh;
    a.sw = g.sw;
    a.pt = g.pt;
    a.pl = g.pl;
    a.is_max = kMax;
    a.include_pad = include_pad_;
    if (!kMax && ctx.has_in(1)) {  // AvgPool2DPrelu
      a.slope = ctx.in(1).data<float>();
      a.y2 = ctx.outputs.size() > 1 && ctx.outputs[1] ? ctx.out(1).data<float>() : nullptr;
    }
    const int64_t rows = g.n * g.oh;
    ctx.pool->parallel_for(rows, parallel_grain(rows, rows * g.ow * g.c * g.kh * g.kw, 4 * kElemGrain),
                           [&](int64_t r0, int64_t r1) { B::pool_rows(a, r0, r1); });
  }

 private:
  bool include_pad_ = false;
  Pool2DGeom g_{};
};

// ---------------------------------------------------------- GlobalAvgPool
// Work items are (batch, block of 4 vectors of channels); each sums its
// channels over all pixels with 4 vector accumulators.
template <class B>
class GlobalAvgPoolOpt : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& xt = ctx.in(0);
    const int64_t n = xt.dim(0), hw = xt.dim(1) * xt.dim(2), c = xt.dim(3);
    const float* x = xt.data<float>();
    float* y = ctx.out(0).data<float>();
    constexpr int64_t kBlock = 4 * B::kLanes;
    const int64_t cblocks = (c + kBlock - 1) / kBlock;
    const int64_t grain = std::max<int64_t>(1, kElemGrain / std::max<int64_t>(hw * kBlock, 1));
    ctx.pool->parallel_for(n * cblocks, grain, [&](int64_t t0, int64_t t1) { B::gap(x, y, hw, c, t0, t1); });
  }
};

// -------------------------------------------------------------------- Pad
// Constant padding of an f32 tensor of any rank: the output is filled with
// `value`, then every innermost row of the input is copied to its place (pure
// data movement, bit-identical to the reference). Work items are blocks of
// input rows; the fill is split the same way over output elements. One
// kernel for every backend (memcpy / std::fill vectorize by themselves).
class PadOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& y = ctx.out(0);
    pads_ = ctx.attrs().get_ints("pads", {});
    value_ = static_cast<float>(ctx.attrs().get_number("value", 0.0));
    rank_ = x.rank();
    inner_ = x.dim(-1);
    rows_ = inner_ > 0 ? x.size() / inner_ : 0;
    xdims_.assign(x.shape().begin(), x.shape().end());
    ystride_.assign(static_cast<size_t>(rank_), 1);
    for (int d = rank_ - 2; d >= 0; --d) ystride_[d] = ystride_[d + 1] * y.dim(d + 1);
  }
  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    const float value = value_;
    parallel_elements(ctx.pool, ctx.out(0).size(), [&](int64_t i0, int64_t i1) { std::fill(y + i0, y + i1, value); });
    if (rows_ == 0) return;
    parallel_rows(ctx.pool, rows_, inner_, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        // Output offset of input row r: decompose r over the outer dims.
        int64_t rem = r, off = pads_[2 * (rank_ - 1)];
        for (int d = rank_ - 2; d >= 0; --d) {
          off += (rem % xdims_[d] + pads_[2 * d]) * ystride_[d];
          rem /= xdims_[d];
        }
        std::memcpy(y + off, x + r * inner_, static_cast<size_t>(inner_) * sizeof(float));
      }
    });
  }

 private:
  std::vector<int64_t> pads_, xdims_, ystride_;
  float value_ = 0;
  int rank_ = 0;
  int64_t inner_ = 0, rows_ = 0;
};

bool pad_supported(const KernelContext& ctx) {
  if (ctx.in(0).dtype() != DType::kF32 || ctx.in(0).rank() < 1) return false;
  for (int64_t p : ctx.attrs().get_ints("pads", {}))
    if (p < 0) return false;  // cropping is left to the reference kernel
  return true;
}

template <class B>
void register_backend(OpRegistry& reg, const char* suffix, int priority, uint32_t features) {
  auto add = [&](const char* op, const char* base, bool (*supports)(const KernelContext&), std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, std::string(base) + suffix, "float_opt", priority, features, supports, create});
  };
  add("Add", "add.float_opt", nullptr, make_kernel<AddOpt<B>>);
  add("ChannelAffine", "channel_affine.float_opt", nullptr, make_kernel<ChannelAffineOpt<B>>);
  add("Activation", "activation.float_opt", activation_supported, make_kernel<ActivationOpt<B>>);
  add("MaxPool2D", "maxpool2d.float_opt", nullptr, make_kernel<Pool2DOpt<B, true>>);
  add("AvgPool2D", "avgpool2d.float_opt", nullptr, make_kernel<Pool2DOpt<B, false>>);
  add("GlobalAvgPool", "global_avg_pool.float_opt", nullptr, make_kernel<GlobalAvgPoolOpt<B>>);
}

}  // namespace

void register_float_opt_avgpool_prelu_kernels(OpRegistry& reg) {
  reg.add_kernel({"AvgPool2DPrelu", "avgpool2d_prelu.float_opt", "float_opt", 40, kFloatOptFeatures, nullptr,
                  make_kernel<Pool2DOpt<Simd4Ew, false>>});
#if defined(__x86_64__)
  reg.add_kernel({"AvgPool2DPrelu", "avgpool2d_prelu.float_opt_avx2", "float_opt", kFloatOptAvx2Priority, kFloatOptAvx2Features, nullptr,
                  make_kernel<Pool2DOpt<Avx2Ew, false>>});
#endif
}

void register_float_opt_elementwise_kernels(OpRegistry& reg) {
  register_backend<Simd4Ew>(reg, "", 40, kFloatOptFeatures);
  reg.add_kernel({"Pad", "pad.float_opt", "float_opt", 40, kFloatOptFeatures, pad_supported, make_kernel<PadOpt>});
#if defined(__x86_64__)
  register_backend<Avx2Ew>(reg, "_avx2", kFloatOptAvx2Priority, kFloatOptAvx2Features);
#endif
}

}  // namespace bitino
