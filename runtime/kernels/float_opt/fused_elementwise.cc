// float_opt kernels of the plan-time fused elementwise operator
// (FusedElementwise, ops/fused_elementwise.h) plus the standalone Mul and
// ChannelGate kernels that use the same leaves.
//
// FusedElementwise compiles its stack program at prepare() into register
// micro-ops (FeMicro, float_opt_args.h) and evaluates them row by row on
// chunks of 32 floats (leaf_fe_rows, float_opt_leaf.h): the whole program
// runs on one chunk while its values stay in vector registers, so each input
// is read once, the output written once and no intermediate value touches
// memory. Every micro-op is exactly the float operation of the reference
// kernel of the operator it replaces (same vector / scalar semantics as the
// one-pass leaves), so results are bit-identical. Tile / gather loads read
// their source rows in place when the piece width is a multiple of the chunk
// width, else the row is first copied to a per-thread row buffer.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "kernels/float_opt/eltwise_program.h"
#include "kernels/float_opt/float_opt.h"
#include "kernels/float_opt/float_opt_leaf.h"
#include "kernels/float_opt/float_simd.h"
#include "ops/fused_elementwise.h"
#include "ops/op_defs.h"

#if defined(__x86_64__)
#include "kernels/float_opt/float_opt_avx2.h"
#endif

namespace bitino {
namespace {

using namespace fopt;

constexpr int64_t kElemGrain = 16384;   // elements per parallel work item

struct FeSimd4 {
  static constexpr int kChunk = 32, kVec = 4;  // chunk of 8 vectors: stack top + operand in 16 q registers
  static void rows(const FeEvalArgs& e, float* y, int64_t r0, int64_t r1, void* scratch) {
    if (e.chunk == kChunk) leaf_fe_rows<Simd4, kChunk / 4>(e, y, r0, r1, scratch);
    else leaf_fe_rows<Simd4, kChunk / 8>(e, y, r0, r1, scratch);
  }
  static void add(const float* a, const float* b, float* y, int64_t n, int32_t act) { leaf_add<Simd4>(a, b, y, 0, n, act); }
  static void mul(const float* a, const float* b, float* y, int64_t n) { leaf_mul<Simd4>(a, b, y, 0, n); }
  static void affine_rows(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t rows) {
    leaf_affine_rows<Simd4>(x, s, b, y, c, 0, rows);
  }
  static void mul_rows(const float* x, const float* g, float* y, int64_t c, int64_t rows) { leaf_mul_rows<Simd4>(x, g, y, c, 0, rows); }
  static void activation(int32_t t, const float* x, float* y, int64_t n, float lo, float hi, float al) {
    leaf_activation<Simd4>(t, x, y, 0, n, lo, hi, al);
  }
  static void prelu_rows(const float* x, const float* sl, float* y, int64_t c, int64_t rows) { leaf_prelu_rows<Simd4>(x, sl, y, c, 0, rows); }
  static void sign(const float* x, float* y, int64_t n, int32_t zero) { leaf_sign<Simd4>(x, y, 0, n, zero); }
};

#if defined(__x86_64__)
struct FeAvx2 {
  static constexpr int kChunk = fopt_avx2::kFeChunk, kVec = 8;
  static void rows(const FeEvalArgs& e, float* y, int64_t r0, int64_t r1, void* scratch) { fopt_avx2::fe_rows(e, y, r0, r1, scratch); }
  static void add(const float* a, const float* b, float* y, int64_t n, int32_t act) { fopt_avx2::add(a, b, y, 0, n, act); }
  static void mul(const float* a, const float* b, float* y, int64_t n) { fopt_avx2::mul(a, b, y, 0, n); }
  static void affine_rows(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t rows) {
    fopt_avx2::affine_rows(x, s, b, y, c, 0, rows);
  }
  static void mul_rows(const float* x, const float* g, float* y, int64_t c, int64_t rows) { fopt_avx2::mul_rows(x, g, y, c, 0, rows); }
  static void activation(int32_t t, const float* x, float* y, int64_t n, float lo, float hi, float al) {
    fopt_avx2::activation(t, x, y, 0, n, lo, hi, al);
  }
  static void prelu_rows(const float* x, const float* sl, float* y, int64_t c, int64_t rows) { fopt_avx2::prelu_rows(x, sl, y, c, 0, rows); }
  static void sign(const float* x, float* y, int64_t n, int32_t zero) { fopt_avx2::sign(x, y, 0, n, zero); }
};
#endif

// Runs fn(row_begin, row_end) over rows of c channels, ~kElemGrain elements per item.
template <typename Fn>
void parallel_rows(ThreadPool* pool, int64_t rows, int64_t c, const Fn& fn) {
  const int64_t grain = std::max<int64_t>(1, kElemGrain / std::max<int64_t>(c, 1));
  const int64_t blocks = (rows + grain - 1) / grain;
  pool->parallel_for(blocks, 1, [&](int64_t b0, int64_t b1) { fn(b0 * grain, std::min(rows, b1 * grain)); });
}

// ChannelGate's per-(batch, channel) factor: sigmoid evaluated in double and
// rounded once (exactly ChannelGateRef), or the gate itself.
void gate_factors(const float* g, int64_t n, bool sigmoid, float* out) {
  for (int64_t i = 0; i < n; ++i) out[i] = sigmoid ? static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(g[i])))) : g[i];
}

// ------------------------------------------------------- FusedElementwise
// The program compiler / evaluator shared by the FusedElementwise kernels and
// by the epilogues of binary kernels (EltwiseProgram, eltwise_program.h).
// Loads of input kEltwiseInPlace read the buffer the program runs on.
}  // namespace

class FeCoreBase {
 public:
  virtual ~FeCoreBase() = default;
  virtual void apply(float* y, int64_t r0, int64_t r1) = 0;
  std::vector<FeInstr> prog_;
  std::vector<FeMicro> ops_;
  std::vector<int32_t> from_;     // per micro-op: index of its instruction in prog_ (-1: kFePop)
  std::vector<int32_t> row_ops_;  // micro-ops needing per-row pointers
  std::vector<std::vector<float>> gate_;
  std::vector<const float*> in_;
  int64_t c_ = 0, rows_ = 0, rows_per_batch_ = 1;
  FeEvalArgs args_{};
  ScratchSlots scratch_;

  // chunk / vec: the backend's chunk and vector widths in floats (they decide
  // which tile / gather loads can read their source rows in place).
  void prepare(const std::vector<FeInstr>& prog, const KernelContext& ctx, int64_t c, int64_t rows, int64_t rows_per_batch, int chunk,
               int vec) {
    prog_ = prog;
    BF_CHECK(fe_stack_depth(prog_) <= 16, "FusedElementwise: stack deeper than 16");
    c_ = c;
    rows_ = rows;
    rows_per_batch_ = std::max<int64_t>(1, rows_per_batch);
    gate_.assign(prog_.size(), {});
    ops_.clear();
    from_.clear();
    row_ops_.clear();
    // Chunk width: the backend's, or half of it when that lets every tile /
    // gather load read its source in place (e.g. 16-float gather pieces).
    bool full_ok = true, half_ok = true;
    for (const FeInstr& in : prog_)
      if (in.op == FeOp::kLoadTile || in.op == FeOp::kLoadGather) {
        full_ok = full_ok && in.act % chunk == 0;
        half_ok = half_ok && in.act % (chunk / 2) == 0;
      }
    if (!full_ok && half_ok && c >= chunk / 2) chunk /= 2;
    args_.chunk = chunk;
    // Operand tree of the postfix program (children of each instruction).
    std::vector<int32_t> ca(prog_.size(), -1), cb(prog_.size(), -1), stk;
    for (size_t k = 0; k < prog_.size(); ++k) {
      const FeOp op = prog_[k].op;
      if (op == FeOp::kAdd || op == FeOp::kMul) {
        cb[k] = stk.back();
        stk.pop_back();
        ca[k] = stk.back();
        stk.back() = static_cast<int32_t>(k);
      } else if (fe_is_load(op)) {
        stk.push_back(static_cast<int32_t>(k));
      } else {
        ca[k] = stk.back();
        stk.back() = static_cast<int32_t>(k);
      }
    }
    int32_t slots = 0;
    auto emit = [&](const FeMicro& m, int32_t from) {
      ops_.push_back(m);
      from_.push_back(from);
    };
    // Source micro-op (opcode base + addressing variant) for load instruction k.
    auto source = [&](int32_t k, int32_t base) {
      const FeInstr& in = prog_[static_cast<size_t>(k)];
      FeMicro m{};
      int32_t at = kFeAtRow;
      if (in.op == FeOp::kLoad) {
        m.src = kFeSrcRow;
      } else {
        m.w = in.act;
        BF_CHECK(m.w > 0 && c % m.w == 0, "FusedElementwise: bad tile / gather width");
        const bool direct = m.w % vec == 0 && (c < chunk || m.w % chunk == 0);
        if (in.op == FeOp::kLoadGather) {
          m.v = ctx.in(in.a).dim(0);
          m.src = direct ? kFeSrcGather : kFeSrcGatherRow;
          at = direct ? kFeAtGather : kFeAtRow;
        } else {
          m.src = direct ? kFeSrcTile : kFeSrcTileRow;
          at = direct ? kFeAtTile : kFeAtRow;
        }
        if (!direct) m.slot = slots++;
      }
      m.op = base + at;
      row_ops_.push_back(static_cast<int32_t>(ops_.size()));
      emit(m, k);
    };
    auto is_bias = [&](int32_t k) { return k >= 0 && prog_[k].op == FeOp::kAffine && prog_[k].a < 0 && prog_[k].b >= 0; };
    // Emits the micro-ops computing the subtree rooted at instruction k into
    // top; `depth` counts the values held (top + spilled). Loads are free of
    // side effects, so a subtree may be evaluated before a sibling load; every
    // binary op keeps its operand order.
    std::function<void(int32_t, int&)> gen = [&](int32_t k, int& depth) {
      const FeInstr& in = prog_[static_cast<size_t>(k)];
      FeMicro m{};
      m.lo = in.lo;
      m.hi = in.hi;
      m.alpha = in.alpha;
      if (fe_is_load(in.op)) {
        if (depth > 0) {
          FeMicro s{};
          s.op = kFeSpill;
          emit(s, -1);
        }
        source(k, kFePush);
        ++depth;
        return;
      }
      if (in.op == FeOp::kAdd || in.op == FeOp::kMul) {
        const bool add = in.op == FeOp::kAdd, plain = !add || in.act == kActNone;
        m.act = add ? in.act : kActNone;
        const int32_t a = ca[k], b = cb[k];
        if (fe_is_load(prog_[b].op)) {  // top = top op source
          gen(a, depth);
          if (plain) {
            source(b, add ? kFeAddSrc : kFeMulSrc);
          } else {
            source(b, kFeOpnd);
            m.op = kFeAdd;
            emit(m, k);
          }
        } else if (fe_is_load(prog_[a].op)) {  // top = source op top
          gen(b, depth);
          if (plain) {
            source(a, add ? kFeRAddSrc : kFeRMulSrc);
          } else {
            source(a, kFeOpnd);
            m.op = kFeRAdd;
            emit(m, k);
          }
        } else {
          gen(a, depth);
          gen(b, depth);
          FeMicro pop{};
          pop.op = kFePop;
          emit(pop, -1);
          m.op = add ? kFeAdd : kFeMul;
          emit(m, k);
          --depth;
        }
        return;
      }
      // bias -> prelu -> bias (consecutive in the postfix order): one micro-op.
      if (is_bias(k) && prog_[ca[k]].op == FeOp::kPrelu && is_bias(ca[ca[k]])) {
        const int32_t first = ca[ca[k]];
        gen(ca[first], depth);
        m.op = kFeRPrelu;
        emit(m, first);
        return;
      }
      gen(ca[k], depth);
      switch (in.op) {
        case FeOp::kAffine: m.op = in.a >= 0 ? (in.b >= 0 ? kFeAffineSB : kFeAffineS) : (in.b >= 0 ? kFeAffineB : kFeAffineN); break;
        case FeOp::kRelu: m.op = kFeRelu; break;
        case FeOp::kClamp: m.op = kFeClamp; break;
        case FeOp::kLeaky: m.op = kFeLeaky; break;
        case FeOp::kPrelu: m.op = kFePrelu; break;
        case FeOp::kSign: m.op = kFeSign; break;
        case FeOp::kSignZero: m.op = kFeSign0; break;
        case FeOp::kGate:
          m.op = kFeGate;
          gate_[k].assign(static_cast<size_t>(ctx.in(in.a).size()), 0.f);
          row_ops_.push_back(static_cast<int32_t>(ops_.size()));
          break;
        default: BF_CHECK(false, "FusedElementwise: unexpected opcode");
      }
      emit(m, k);
    };
    int depth = 0;
    gen(static_cast<int32_t>(prog_.size()) - 1, depth);
    // Sentinel micro-op ending the program (FeChunk::run_chunks dispatches to it; not counted in n_ops).
    FeMicro end{};
    end.op = kFeNumOps;
    ops_.push_back(end);
    from_.push_back(-1);
    args_.ops = ops_.data();
    args_.n_ops = static_cast<int32_t>(ops_.size()) - 1;
    args_.row_ops = row_ops_.data();
    args_.n_row_ops = static_cast<int32_t>(row_ops_.size());
    args_.n_slots = slots;
    args_.flat = 1;
    for (int32_t j : row_ops_)
      if (ops_[static_cast<size_t>(j)].op == kFeGate || ops_[static_cast<size_t>(j)].src != kFeSrcRow) args_.flat = 0;
    args_.c = c_;
    args_.rows_per_batch = rows_per_batch_;
    const size_t ptr_floats = (2 * ops_.size() * sizeof(void*) + sizeof(float) - 1) / sizeof(float);
    scratch_.init(ctx.pool->num_threads(), ptr_floats + static_cast<size_t>(slots * c_));
  }
  // Per run, single-threaded: operand pointers and ChannelGate factors.
  void begin(const KernelContext& ctx) {
    in_.resize(ctx.inputs.size());
    for (size_t i = 0; i < ctx.inputs.size(); ++i) in_[i] = ctx.inputs[i] ? ctx.inputs[i]->data<float>() : nullptr;
    for (size_t j = 0; j < ops_.size(); ++j) {
      if (from_[j] < 0) continue;
      const FeInstr& in = prog_[static_cast<size_t>(from_[j])];
      FeMicro& m = ops_[j];
      switch (in.op) {
        case FeOp::kLoad: m.p = in.a == kEltwiseInPlace ? nullptr : in_[in.a]; break;
        case FeOp::kLoadTile: m.p = in_[in.a]; break;
        case FeOp::kLoadGather:
          m.p = in_[in.a];
          m.ids = reinterpret_cast<const int32_t*>(in_[in.b]);
          break;
        case FeOp::kAffine:
          if (m.op == kFeRPrelu) {
            const size_t f = static_cast<size_t>(from_[j]);
            m.q = in_[in.b];
            m.p = in_[prog_[f + 1].a];
            m.r = in_[prog_[f + 2].b];
            break;
          }
          m.p = in.a >= 0 ? in_[in.a] : nullptr;
          m.q = in.b >= 0 ? in_[in.b] : nullptr;
          break;
        case FeOp::kPrelu: m.p = in_[in.a]; break;
        case FeOp::kGate: {
          const Tensor& g = ctx.in(in.a);
          std::vector<float>& f = gate_[static_cast<size_t>(from_[j])];
          gate_factors(g.data<float>(), g.size(), in.b != 0, f.data());
          m.p = f.data();
          break;
        }
        default: break;
      }
    }
  }
};

namespace {

template <class B>
class FeCore : public FeCoreBase {
 public:
  void prepare(const std::vector<FeInstr>& prog, const KernelContext& ctx, int64_t c, int64_t rows, int64_t rows_per_batch) {
    FeCoreBase::prepare(prog, ctx, c, rows, rows_per_batch, B::kChunk, B::kVec);
  }
  // Rows [r0, r1) of y (in place for loads of kEltwiseInPlace); thread safe.
  void apply(float* y, int64_t r0, int64_t r1) override {
    if (r0 >= r1) return;
    ScratchLease lease(scratch_);
    B::rows(args_, y, r0, r1, lease.data());
  }
};

template <class B>
class FusedElementwiseOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Tensor& y = ctx.out(0);
    const int64_t c = y.dim(-1), rows = c > 0 ? y.size() / c : 0;
    core_.prepare(parse_fe_program(ctx.attrs()), ctx, c, rows, y.dim(0) > 0 ? rows / y.dim(0) : 1);
  }
  void run(const KernelContext& ctx) override {
    if (core_.rows_ == 0) return;
    core_.begin(ctx);
    float* y = ctx.out(0).data<float>();
    parallel_rows(ctx.pool, core_.rows_, core_.c_, [&](int64_t r0, int64_t r1) { core_.apply(y, r0, r1); });
  }

 private:
  FeCore<B> core_;
};

bool fused_supported(const KernelContext& ctx) { return fe_stack_depth(parse_fe_program(ctx.attrs())) <= 16; }

// -------------------------------------------------------------------- Mul
template <class B>
class MulOpt : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t n = ctx.in(0).size();
    const int64_t blocks = (n + kElemGrain - 1) / kElemGrain;
    ctx.pool->parallel_for(blocks, 1, [&](int64_t b0, int64_t b1) {
      const int64_t i0 = b0 * kElemGrain, i1 = std::min(n, b1 * kElemGrain);
      B::mul(a + i0, b + i0, y + i0, i1 - i0);
    });
  }
};

// ------------------------------------------------------------ ChannelGate
template <class B>
class ChannelGateOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    sigmoid_ = ctx.attrs().get_string("gate", "sigmoid") == "sigmoid";
    factors_.assign(static_cast<size_t>(ctx.in(1).size()), 0.f);
  }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t n = x.dim(0), hw = x.dim(1) * x.dim(2), c = x.dim(3);
    gate_factors(ctx.in(1).data<float>(), n * c, sigmoid_, factors_.data());
    const float* xp = x.data<float>();
    const float* g = factors_.data();
    float* y = ctx.out(0).data<float>();
    parallel_rows(ctx.pool, n * hw, c, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1;) {
        const int64_t b = r / hw, end = std::min(r1, (b + 1) * hw);
        B::mul_rows(xp + r * c, g + b * c, y + r * c, c, end - r);
        r = end;
      }
    });
  }

 private:
  bool sigmoid_ = true;
  std::vector<float> factors_;
};

template <class B>
void register_backend(OpRegistry& reg, const char* suffix, int priority, uint32_t features) {
  auto add = [&](const char* op, const char* base, bool (*supports)(const KernelContext&), std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, std::string(base) + suffix, "float_opt", priority, features, supports, create});
  };
  add("FusedElementwise", "fused_elementwise.float_opt", fused_supported, make_kernel<FusedElementwiseOpt<B>>);
  add("Mul", "mul.float_opt", nullptr, make_kernel<MulOpt<B>>);
  add("ChannelGate", "channel_gate.float_opt", nullptr, make_kernel<ChannelGateOpt<B>>);
}

}  // namespace

// ------------------------------------------------------------ EltwiseProgram
EltwiseProgram::EltwiseProgram() = default;
EltwiseProgram::~EltwiseProgram() = default;

void EltwiseProgram::prepare(const std::vector<FeInstr>& prog, const KernelContext& ctx, int64_t c, int64_t rows, int64_t rows_per_batch) {
#if defined(__x86_64__)
  if (cpu_info().has(kFloatOptAvx2Features)) {
    auto* core = new FeCore<FeAvx2>();
    core_.reset(core);
    core->prepare(prog, ctx, c, rows, rows_per_batch);
    return;
  }
#endif
  auto* core = new FeCore<FeSimd4>();
  core_.reset(core);
  core->prepare(prog, ctx, c, rows, rows_per_batch);
}

void EltwiseProgram::begin(const KernelContext& ctx) { core_->begin(ctx); }

void EltwiseProgram::apply(float* y, int64_t r0, int64_t r1) { core_->apply(y, r0, r1); }

bool EltwiseProgram::prepare_epilogue(const KernelContext& ctx, int64_t rows_per_batch) {
  if (!ctx.attrs().contains("epilogue")) return false;
  const Tensor& y = ctx.out(0);
  const int64_t c = y.dim(-1), rows = c > 0 ? y.size() / c : 0;
  prepare(parse_fe_program(Json(Json::Object{{"program", ctx.attrs()["epilogue"]}})), ctx, c, rows, rows_per_batch);
  return true;
}

void register_float_opt_fused_kernels(OpRegistry& reg) {
  register_backend<FeSimd4>(reg, "", 40, kFloatOptFeatures);
#if defined(__x86_64__)
  register_backend<FeAvx2>(reg, "_avx2", kFloatOptAvx2Priority, kFloatOptAvx2Features);
#endif
}

}  // namespace bitino
