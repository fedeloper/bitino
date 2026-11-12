// Plan-time graph rewrites applied by Session on its private copy of the
// graph (the Model is never modified). Every rewrite is BIT-EXACT: the
// rewritten graph performs the same IEEE float operations, in the same order
// per element, as the original one (docs/architecture.md § Plan-time rewrites).
//
//  * fold_thresholds: ChannelAffine(x, scale = none, bias b) whose output only
//    feeds the binarized f32 input of BConv2D/BDense nodes (ge, gt, sign_zero,
//    no threshold input) is removed and becomes the consumers' threshold -b.
//    Exact because fl(x + b) and x + b have the same sign (and are zero
//    together) for finite x, b: fl(x + b) >= 0 <=> x >= -b, likewise > 0, < 0.
//    With a scale s (fold_scaled_thresholds; ge / gt consumers): the channel's
//    threshold t, fl(fl(x * s) + b) >= 0 (gt: > 0) <=> x >= t, is found by
//    bisection over the ordered floats (monotone for s != 0); channels with
//    s < 0 are binarized negated and the consumers' weights of that channel
//    flipped (a BConv2D only without groups and with pad_value 0). Finite x.
//  * fuse_elementwise: maximal trees of elementwise nodes (ChannelAffine, Add,
//    Mul, Activation relu/relu6/hardtanh/prelu/leaky_relu/sign/sign_zero,
//    ChannelGate) whose inner results have exactly one consumer become one
//    FusedElementwise node that evaluates the tree in one pass over the data
//    (ops/fused_elementwise.h). Inner tensors are never materialized; operands
//    produced by a row Gather (through Reshapes) or by a Concat of one tensor
//    with itself become gather / tile loads of the program. When
//    one operand is the f32 output of a BConv2D used only by the tree, and the
//    other operands exist when that BConv2D runs, the tree (even a single node)
//    becomes the BConv2D's in-kernel epilogue instead (host_epilogues). The
//    same holds for a BDense with a binary (ge / gt) input and no w_zero.
//  * bit_outputs: a BConv2D, or a BDense with a binary (ge / gt) input and no
//    w_zero, whose (possibly epilogue-fused) f32 output has exactly one use,
//    input 0 of a BConv2D / BDense that binarizes it with ge / gt, stores that
//    binarization itself ("output": "bit", "output_binarize" = the consumer's
//    mode, the consumer's threshold moved to the producer's input named by
//    "output_threshold_input"); the consumer reads the bits. Exact: the stored
//    bit is pack_row_f32() of the float value the consumer would have read
//    (kernels/bit_output.h). sign_zero consumers need three values: skipped.
//  * shared_binarizations (with bit_outputs): an f32 tensor that two or more BDense nodes binarize identically
//    (input 0, same ge / gt mode and threshold input, no w_zero) is packed once by an internal PackBits node
//    (ops/op_defs_fused.cc) inserted before the first of them, which then read the bits. Exact: the packed bits
//    are those each consumer would have packed.
#pragma once

#include <vector>

#include "bitino/model.h"

namespace bitino {

struct RewriteOptions {
  bool fold_thresholds = true;
  bool fuse_elementwise = true;
  // Host a fused tree in the BConv2D / BDense node producing one of its operands
  // (attribute "epilogue", kernels/float_opt/eltwise_program.h) when every other
  // operand is available at that node: its output is never written back separately.
  bool host_epilogues = true;
  bool host_bconv2d = true;  // per producer op (false e.g. when a kernel is forced for it)
  bool host_bdense = true;
  // Bit-packed outputs of binary layers whose only consumer binarizes them (after fusion).
  bool bit_outputs = true;
  bool bit_bconv2d = true;  // per op, as producer or consumer (false e.g. when a kernel is forced for it)
  bool bit_bdense = true;
};

struct RewriteStats {
  int thresholds_folded = 0;  // ChannelAffine nodes removed
  int fused_nodes = 0;        // FusedElementwise nodes created
  int absorbed_nodes = 0;     // original nodes replaced by them
  int hosted_epilogues = 0;   // fused trees moved into the BConv2D / BDense producing an operand
  int bit_outputs = 0;        // binary layers writing bit-packed output for the next binary layer
  int shared_binarizations = 0;  // PackBits nodes inserted (one packing shared by several BDense inputs)
};

// `nodes` and `infos` are the session's copies with shapes already inferred;
// constant tensors created by a rewrite are appended to `infos`.
RewriteStats rewrite_graph(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& graph_outputs,
                           const RewriteOptions& opts);

}  // namespace bitino
