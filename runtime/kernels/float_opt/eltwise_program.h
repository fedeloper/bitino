// Elementwise epilogue programs for producer kernels (bit-exact).
//
// The plan-time rewrite (src/graph_rewrite.cc) can host a fused elementwise
// tree in the BConv2D node that produces one of its operands: the node gets
// the attribute "epilogue" (a FusedElementwise program, ops/fused_elementwise.h)
// in which loads of input kEltwiseInPlace read the convolution's own output,
// and the other operands are extra node inputs. A kernel computes its output
// as usual and then runs the program in place on the rows it just wrote,
// while they are still in cache; the result is the fused tree's output,
// bit-identical to running the nodes separately.
//
// The interpreter is the float_opt one (AVX2 + FMA leaves when the CPU has
// them, else NEON / portable); kernels only call prepare_epilogue() in
// prepare(), begin() once per run and apply() on disjoint row ranges (any
// thread).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "bitino/op.h"
#include "ops/fused_elementwise.h"

namespace bitino {

constexpr int32_t kEltwiseInPlace = -2;

class FeCoreBase;  // fused_elementwise.cc

class EltwiseProgram {
 public:
  EltwiseProgram();
  ~EltwiseProgram();
  EltwiseProgram(const EltwiseProgram&) = delete;
  EltwiseProgram& operator=(const EltwiseProgram&) = delete;

  // Program over `rows` rows of `c` floats (batch of row r = r / rows_per_batch).
  void prepare(const std::vector<FeInstr>& prog, const KernelContext& ctx, int64_t c, int64_t rows, int64_t rows_per_batch);
  // From the node attribute "epilogue" (rows/c from the node's output); false if absent.
  bool prepare_epilogue(const KernelContext& ctx, int64_t rows_per_batch);
  bool active() const { return core_ != nullptr; }
  void begin(const KernelContext& ctx);
  void apply(float* y, int64_t r0, int64_t r1);

 private:
  std::unique_ptr<FeCoreBase> core_;
};

// True if the node carries an epilogue program (kernels without epilogue
// support reject such nodes in supports()).
inline bool has_epilogue(const KernelContext& ctx) { return ctx.attrs().contains("epilogue"); }

}  // namespace bitino
