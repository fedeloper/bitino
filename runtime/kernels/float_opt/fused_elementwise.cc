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
#include <string>
#include <tuple>
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
  // chunk of 8 vectors: stack top + operand in 16 q registers (AArch32, 16 q registers in all: 4 vectors)
  static constexpr int kChunk = BF_FLOAT_OPT_NEON32 ? 16 : 32, kVec = 4;
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
  // the GlobalAvgPool kernel's sums (elementwise.cc, global_avg_pool.float_opt) for reduce programs
  static constexpr int kGapLanes = 4;
  static void gap(const float* x, float* y, int64_t hw, int64_t c, int64_t t0, int64_t t1) { leaf_global_avg_pool<Simd4>(x, y, hw, c, t0, t1); }
  static void gap_add_rows(float* acc, const float* x, int64_t rows, int64_t c) { leaf_gap_add_rows<Simd4>(acc, x, rows, c); }
  static void gap_finish(const float* acc, float* y, int64_t c, int64_t hw) { leaf_gap_finish<Simd4>(acc, y, c, hw); }
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
  static constexpr int kGapLanes = 8;
  static void gap(const float* x, float* y, int64_t hw, int64_t c, int64_t t0, int64_t t1) { fopt_avx2::global_avg_pool(x, y, hw, c, t0, t1); }
  static void gap_add_rows(float* acc, const float* x, int64_t rows, int64_t c) { fopt_avx2::gap_add_rows(acc, x, rows, c); }
  static void gap_finish(const float* acc, float* y, int64_t c, int64_t hw) { fopt_avx2::gap_finish(acc, y, c, hw); }
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

// ------------------------------------------------------- fused unit stores
// EltwiseProgram::store_unit16 (eltwise_program.h): the two-phase binary kernels' 16-channel unit (int16 dot
// products of np rows) converted, activated and run through the epilogue program row by row in registers,
// for programs that are a chain of steps on the producer's value (push of the evaluated buffer, then the
// micro-ops below). Each step performs exactly the float operations of the FeChunk handler of its micro-op,
// in program order, so the result is bit-identical to storing the unit and then applying the program. Per
// channel parameters are loaded once per unit. Simd4 policy: NEON on AArch64, portable elsewhere.
using Unit16Fn = void (*)(const FeMicro* ops, const int16_t* db, int64_t np, const float* mul, const float* add, float* y, int64_t ys,
                          int64_t row0, int64_t ch0, int64_t c);

struct U16 {
  using V = Simd4;
  using T = Simd4::T;
  // int16 dots of one row -> 4 float vectors
  static inline void dots(const int16_t* d, T v[4]) {
#if defined(__aarch64__) || BF_FLOAT_OPT_NEON32   // vmovl_high_s16: kernels/neon_a32.h on 32-bit ARM
    const int16x8_t d0 = vld1q_s16(d), d1 = vld1q_s16(d + 8);
    v[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0)));
    v[1] = vcvtq_f32_s32(vmovl_high_s16(d0));
    v[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1)));
    v[3] = vcvtq_f32_s32(vmovl_high_s16(d1));
#else
    for (int k = 0; k < 4; ++k)
      for (int j = 0; j < 4; ++j) v[k].v[j] = static_cast<float>(d[4 * k + j]);
#endif
  }
  // the binary kernels' fused activation (bconv_neon_opt.cc activate(): v > 0 ? v : 0; min(max(v, 0), 6) as selects)
  template <int A>
  static inline T act(T v) {
    if (A == kActRelu) return V::select_gt(v, V::dup(0.f), v, V::dup(0.f));
    if (A == kActRelu6) {
      const T m = V::select_lt(v, V::dup(0.f), V::dup(0.f), v);
      return V::select_lt(V::dup(6.f), m, V::dup(6.f), m);
    }
    return v;
  }
};

// Row / tile source of a step: p + row * stride + off (+ 4k), off = ch0 (row source) or ch0 % w (tile).
struct U16Src {
  const float* base = nullptr;
  int64_t stride = 0;
  void init(const FeMicro& m, int64_t ch0, int64_t c) {
    const bool tile = m.src != kFeSrcRow;
    base = m.p + (tile ? ch0 % m.w : ch0);
    stride = tile ? m.w : c;
  }
  inline void load(int64_t row, U16::T s[4]) const {
    for (int k = 0; k < 4; ++k) s[k] = U16::V::load(base + row * stride + 4 * k);
  }
};

struct U16AffSB {  // kFeAffineSB
  static constexpr int kOps = 1;
  U16::T p[4], q[4];
  void init(const FeMicro* m, int64_t ch0, int64_t) {
    for (int k = 0; k < 4; ++k) {
      p[k] = U16::V::load(m->p + ch0 + 4 * k);
      q[k] = U16::V::load(m->q + ch0 + 4 * k);
    }
  }
  inline void apply(U16::T v[4], int64_t) const {
    for (int k = 0; k < 4; ++k) v[k] = U16::V::add(U16::V::mul(v[k], p[k]), q[k]);
  }
};
struct U16AddSrc {  // kFeAddSrc: top + source
  static constexpr int kOps = 1;
  U16Src s;
  void init(const FeMicro* m, int64_t ch0, int64_t c) { s.init(*m, ch0, c); }
  inline void apply(U16::T v[4], int64_t row) const {
    U16::T o[4];
    s.load(row, o);
    for (int k = 0; k < 4; ++k) v[k] = U16::V::add(v[k], o[k]);
  }
};
struct U16OpndAdd {  // kFeOpnd (source) + kFeAdd: act(top + opnd)
  static constexpr int kOps = 2;
  U16Src s;
  int32_t act = kActNone;
  void init(const FeMicro* m, int64_t ch0, int64_t c) {
    s.init(*m, ch0, c);
    act = m[1].act;
  }
  inline void apply(U16::T v[4], int64_t row) const {
    U16::T o[4];
    s.load(row, o);
    for (int k = 0; k < 4; ++k) v[k] = leaf_activate<U16::V>(act, U16::V::add(v[k], o[k]));
  }
};
struct U16Prelu {  // kFePrelu
  static constexpr int kOps = 1;
  U16::T p[4];
  void init(const FeMicro* m, int64_t ch0, int64_t) {
    for (int k = 0; k < 4; ++k) p[k] = U16::V::load(m->p + ch0 + 4 * k);
  }
  inline void apply(U16::T v[4], int64_t) const {
    for (int k = 0; k < 4; ++k) v[k] = U16::V::select_ge0(v[k], U16::V::mul(v[k], p[k]));
  }
};
struct U16RPrelu {  // kFeRPrelu: bias q -> prelu p -> bias r
  static constexpr int kOps = 1;
  U16::T p[4], q[4], r[4];
  void init(const FeMicro* m, int64_t ch0, int64_t) {
    for (int k = 0; k < 4; ++k) {
      p[k] = U16::V::load(m->p + ch0 + 4 * k);
      q[k] = U16::V::load(m->q + ch0 + 4 * k);
      r[k] = U16::V::load(m->r + ch0 + 4 * k);
    }
  }
  inline void apply(U16::T v[4], int64_t) const {
    for (int k = 0; k < 4; ++k) {
      const U16::T t = U16::V::add(v[k], q[k]);
      v[k] = U16::V::add(U16::V::select_ge0(t, U16::V::mul(t, p[k])), r[k]);
    }
  }
};

template <class S>
inline void u16_init(S& s, const FeMicro*& m, int64_t ch0, int64_t c) {
  s.init(m, ch0, c);
  m += S::kOps;
}

template <int A, class... S>
void unit16_t(const FeMicro* ops, const int16_t* db, int64_t np, const float* mul, const float* add, float* y, int64_t ys, int64_t row0,
              int64_t ch0, int64_t c) {
  using V = Simd4;
  typename U16::T mv[4], av[4];
  for (int k = 0; k < 4; ++k) {
    mv[k] = V::load(mul + 4 * k);
    av[k] = V::load(add + 4 * k);
  }
  std::tuple<S...> st;
  const FeMicro* m = ops + 1;  // ops[0]: push of the evaluated buffer
  std::apply([&](auto&... s) { (u16_init(s, m, ch0, c), ...); }, st);
  for (int64_t i = 0; i < np; ++i) {
    typename U16::T v[4];
    U16::dots(db + i * 16, v);
    for (int k = 0; k < 4; ++k) v[k] = U16::act<A>(V::add(V::mul(v[k], mv[k]), av[k]));
    const int64_t row = row0 + i;
    std::apply([&](const auto&... s) { (s.apply(v, row), ...); }, st);
    float* yr = y + row * ys + ch0;
    for (int k = 0; k < 4; ++k) V::store(yr + 4 * k, v[k]);
  }
}

template <class... S>
struct U16Pattern {
  static Unit16Fn fn(int act) {
    if (act == kActRelu) return unit16_t<kActRelu, S...>;
    if (act == kActRelu6) return unit16_t<kActRelu6, S...>;
    return unit16_t<kActNone, S...>;
  }
};

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
  std::vector<float> zeros_;      // a row of c zeros (whole-row gathers of out-of-range ids)
  std::vector<const float*> in_;
  int64_t c_ = 0, rows_ = 0, rows_per_batch_ = 1;
  FeEvalArgs args_{};
  ScratchSlots scratch_;
  // store_unit16 evaluators by kernel activation (kActNone / kActRelu / kActRelu6); null: not a supported chain
  Unit16Fn unit16_[3] = {nullptr, nullptr, nullptr};

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
    // Also half when only that divides the rows (e.g. 16 channels): rows then run as whole chunks (and flat
    // programs as one run) instead of per-vector dispatch.
    if (half_ok && c >= chunk / 2 && (!full_ok || (c % chunk != 0 && c % (chunk / 2) == 0))) chunk /= 2;
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
        if (in.op == FeOp::kLoadGather && m.w == c) {  // whole rows: a per-row pointer (no per-chunk index lookup)
          m.v = ctx.in(in.a).dim(0);
          m.src = kFeSrcGatherWhole;
          zeros_.assign(static_cast<size_t>(c), 0.f);
        } else if (in.op == FeOp::kLoadGather) {
          m.v = ctx.in(in.a).dim(0);
          m.src = direct ? kFeSrcGather : kFeSrcGatherRow;
          at = direct ? kFeAtGather : kFeAtRow;
        } else {
          m.src = direct ? kFeSrcTile : kFeSrcTileRow;
          at = direct ? kFeAtTile : kFeAtRow;
        }
        if (!direct && m.src != kFeSrcGatherWhole) m.slot = slots++;
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
      if (ops_[static_cast<size_t>(j)].op != kFeGate && ops_[static_cast<size_t>(j)].src != kFeSrcRow) args_.flat = 0;
    args_.c = c_;
    args_.rows_per_batch = rows_per_batch_;
    const size_t ptr_floats = (2 * ops_.size() * sizeof(void*) + sizeof(float) - 1) / sizeof(float);
    scratch_.init(ctx.pool->num_threads(), ptr_floats + static_cast<size_t>(slots * c_));
    match_unit16();
  }
  // Chains for store_unit16: push of the evaluated buffer, then one of the step sequences below (row / tile
  // sources of other inputs; tile widths a multiple of 16, so that a 16-channel unit reads within one piece).
  void match_unit16() {
    for (auto& f : unit16_) f = nullptr;
    const int32_t n = static_cast<int32_t>(ops_.size()) - 1;  // without the end sentinel
    if (n < 2 || c_ % 16 != 0 || ops_[0].op != kFePush + kFeAtRow || from_[0] < 0 ||
        prog_[static_cast<size_t>(from_[0])].a != kEltwiseInPlace)
      return;
    std::string sig;  // one letter per step
    for (int32_t k = 1; k < n; ++k) {
      const FeMicro& m = ops_[static_cast<size_t>(k)];
      const bool src_op = m.op < kFeSpill;
      if (src_op) {
        if (m.src != kFeSrcRow && m.src != kFeSrcTile && m.src != kFeSrcTileRow) return;
        if (m.src != kFeSrcRow && m.w % 16 != 0) return;
        const int32_t f = from_[static_cast<size_t>(k)];
        if (f < 0 || prog_[static_cast<size_t>(f)].a == kEltwiseInPlace) return;
      }
      const int32_t base = src_op ? m.op - m.op % 3 : m.op;
      if (base == kFeAddSrc) {
        sig += 'a';
      } else if (base == kFeOpnd && k + 1 < n && ops_[static_cast<size_t>(k + 1)].op == kFeAdd) {
        sig += 'o';
        ++k;
      } else if (m.op == kFeAffineSB) {
        sig += 's';
      } else if (m.op == kFePrelu) {
        sig += 'p';
      } else if (m.op == kFeRPrelu) {
        sig += 'r';
      } else {
        return;
      }
    }
    Unit16Fn (*make)(int) = nullptr;
    if (sig == "p") make = U16Pattern<U16Prelu>::fn;
    else if (sig == "r") make = U16Pattern<U16RPrelu>::fn;
    else if (sig == "s") make = U16Pattern<U16AffSB>::fn;
    else if (sig == "a") make = U16Pattern<U16AddSrc>::fn;
    else if (sig == "ar") make = U16Pattern<U16AddSrc, U16RPrelu>::fn;
    else if (sig == "sa") make = U16Pattern<U16AffSB, U16AddSrc>::fn;
    else if (sig == "so") make = U16Pattern<U16AffSB, U16OpndAdd>::fn;
    else if (sig == "sar") make = U16Pattern<U16AffSB, U16AddSrc, U16RPrelu>::fn;
    else if (sig == "sarsa") make = U16Pattern<U16AffSB, U16AddSrc, U16RPrelu, U16AffSB, U16AddSrc>::fn;
    if (!make) return;
    for (int a = 0; a < 3; ++a) unit16_[a] = make(a);
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
          if (m.src == kFeSrcGatherWhole) m.q = zeros_.data();
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
    reduce_ = ctx.attrs().contains("reduce");
    // the elementwise result's shape: the output's, or out_shape (reduce: the output holds its channel means)
    const Shape s = reduce_ ? Shape(ctx.attrs().get_ints("out_shape", {})) : ctx.out(0).shape();
    const int64_t n = s.empty() ? 0 : s[0], c = s.empty() ? 0 : s.back(), rows = c > 0 ? num_elements(s) / c : 0;
    core_.prepare(parse_fe_program(ctx.attrs()), ctx, c, rows, n > 0 ? rows / n : 1);
    if (!reduce_ || rows == 0) return;
    BF_CHECK(s.size() == 4, "FusedElementwise reduce: rank 4");
    batches_ = n;
    if (ctx.pool->num_threads() > 1) {
      full_.assign(static_cast<size_t>(rows * c), 0.f);  // threads: materialized, then the GlobalAvgPool kernel's passes
    } else {
      block_rows_ = std::max<int64_t>(1, kReduceBlockFloats / c);
      block_.assign(static_cast<size_t>(block_rows_ * c), 0.f);
      acc_.assign(static_cast<size_t>(c), 0.f);
    }
  }
  void run(const KernelContext& ctx) override {
    if (core_.rows_ == 0) return;
    core_.begin(ctx);
    float* y = ctx.out(0).data<float>();
    if (reduce_) return run_reduce(ctx, y);
    parallel_rows(ctx.pool, core_.rows_, core_.c_, [&](int64_t r0, int64_t r1) { core_.apply(y, r0, r1); });
  }

 private:
  static constexpr int64_t kReduceBlockFloats = 4096;  // rows evaluated at a time (16 KB, L1) before they are summed

  // reduce = global_avg_pool. One thread: per batch, blocks of rows evaluated into an L1 buffer and added to the
  // channel sums in pixel order (B::gap_add_rows, the GlobalAvgPool kernel's per-channel operations), the result never
  // written to memory. Several threads: the rows evaluated in parallel into full_, then the GlobalAvgPool kernel's
  // pass over channel blocks (each channel summed by one thread in pixel order). Same sums either way.
  void run_reduce(const KernelContext& ctx, float* y) {
    const int64_t c = core_.c_, hw = core_.rows_ / batches_;
    if (!full_.empty()) {
      float* f = full_.data();
      parallel_rows(ctx.pool, core_.rows_, c, [&](int64_t r0, int64_t r1) { core_.apply(f, r0, r1); });
      const int64_t cblocks = (c + 4 * B::kGapLanes - 1) / (4 * B::kGapLanes);
      const int64_t grain = std::max<int64_t>(1, kElemGrain / std::max<int64_t>(hw * 4 * B::kGapLanes, 1));
      ctx.pool->parallel_for(batches_ * cblocks, grain, [&](int64_t t0, int64_t t1) { B::gap(f, y, hw, c, t0, t1); });
      return;
    }
    float* acc = acc_.data();
    const uintptr_t blk = reinterpret_cast<uintptr_t>(block_.data());
    for (int64_t b = 0; b < batches_; ++b) {
      std::fill(acc_.begin(), acc_.end(), 0.f);
      for (int64_t r0 = b * hw; r0 < (b + 1) * hw; r0 += block_rows_) {
        const int64_t r1 = std::min((b + 1) * hw, r0 + block_rows_);
        // rows r0..r1 land at block_ (apply writes row r at y + r * c)
        core_.apply(reinterpret_cast<float*>(blk - static_cast<uintptr_t>(r0 * c) * sizeof(float)), r0, r1);
        B::gap_add_rows(acc, block_.data(), r1 - r0, c);
      }
      B::gap_finish(acc, y + b * c, c, hw);
    }
  }

  FeCore<B> core_;
  bool reduce_ = false;
  int64_t batches_ = 1, block_rows_ = 1;
  std::vector<float> full_, block_, acc_;
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

bool EltwiseProgram::unit16() const { return core_ && core_->unit16_[0] != nullptr; }

void EltwiseProgram::store_unit16(const int16_t* db, int64_t np, const float* mul, const float* add, int act, float* y, int64_t ys,
                                  int64_t row0, int64_t ch0) const {
  core_->unit16_[act](core_->ops_.data(), db, np, mul, add, y, ys, row0, ch0, core_->c_);
}

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
  register_float_opt_avgpool_prelu_kernels(reg);
#if defined(__x86_64__)
  register_backend<FeAvx2>(reg, "_avx2", kFloatOptAvx2Priority, kFloatOptAvx2Features);
#endif
}

}  // namespace bitino
