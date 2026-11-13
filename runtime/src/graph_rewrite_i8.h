// Plan-time rewrites of int8_full graphs (I8* operators), applied by Session on its private copy of the graph when
// SessionOptions::fuse is on (the Model is never modified). Every rewrite is BIT-EXACT: the rewritten graph computes
// the same int8 codes as the original one, element for element (docs/int8.md, "int8_full latency").
//
//  * collapse_unary: an I8Mul / I8Add / I8AffineSum whose operands are all x or I8Lut(x) of one tensor x becomes
//    I8Lut(x, T), T[r][q] = the operator's integer formula on the operand codes of q for all 256 codes (e.g. the
//    GELU-mask products x * keep(x) of Bi-ViT / BiBERT); LUTs left without readers are removed.
//  * compose_luts: I8Lut(I8Lut(x, T1), T2) whose inner output has no other use (and is no graph output) becomes
//    I8Lut(x, T) with T[r][q] = T2[r][T1[r][q]] (rows: one table or per-channel tables along the same axis). A lookup
//    of a lookup is a lookup: exact by construction.
//  * fold_lut_thresholds: an I8Lut y = L(x) read by I8BConv2D / I8BDense that binarize it (input 0, ge / gt): the
//    consumer's integer threshold on y, T_y[c] = qthr_ge / qthr_gt of its float threshold on y's grid, becomes a
//    threshold on x when {x : L[c][x] >= T_y[c]} is an upper set {x >= T_x[c]} (checked on all 256 codes of every
//    channel): the consumer then reads x with x_scale 1, x_zero 0 (deq = identity, exact) and the threshold T_x[c]
//    (an integer, as a float: qthr_ge(T, identity) = T for T in [-128, 128]). Otherwise, when the -1 codes
//    {x : L[c][x] < T_y[c]} form an interval [lo_c, hi_c] for every channel (V-shaped LUTs, e.g. PReLU with negative
//    slopes), the consumer reads the bit tensor I8PackBits(x, [lo; hi]) (bit -1 iff lo_c <= x <= hi_c; shared by consumers
//    with equal bounds). The LUT is removed when nothing else reads y.
//  * merge_twins: element-wise nodes that read the same inputs ("twins", e.g. a tensor and its binarize-aware copy):
//    I8Lut nodes on one input -> one I8LutN (one pass, shared loads / transposes); I8Add nodes with the same inputs and
//    input parameters -> one I8AddN (the rescaled input sum computed once, one output requantization per twin);
//    I8Conv2D nodes with the same input, byte-identical weights and bias and the same geometry -> one I8Conv2DN (the
//    int32 accumulators computed once, one requantization per twin); I8LayerNorm nodes with the same x and statistics
//    attributes -> one I8LayerNormN (row statistics and fp32(u) * r once).
//  * bit_outputs (2026-11-13, after merge_twins): an I8Add (-> internal I8AddB), an output of an I8AddN, or an
//    I8AffineSum (-> internal I8AffineSumB) whose output is read only by binary layers binarizing it (input 0, ge / gt,
//    the same thresholds) produces the bit tensor itself: bit -1 iff the integer sum before the output requantization is
//    <= U[c] (U: i32 [1 | C] constant input; I8AddN: one per output listed in attribute bit_outputs, after a and b). The
//    output requantization is non-decreasing (multiplier > 0), so y >= T <=> acc >= A: the same bits.
#pragma once

#include <vector>

#include "bitino/model.h"

namespace bitino {

struct I8RewriteStats {
  int collapsed_ops = 0;   // I8Mul / I8Add / I8AffineSum of one tensor replaced by an I8Lut
  int composed_luts = 0;   // I8Lut nodes removed by composition
  int folded_luts = 0;     // binary-layer inputs re-pointed through a LUT (threshold folding)
  int removed_luts = 0;    // I8Lut nodes removed after folding
  int interval_folds = 0;  // binary-layer inputs read as I8PackBits(x) through a LUT (interval binarization)
  int lut_twins = 0;       // I8LutN nodes created
  int add_twins = 0;       // I8AddN nodes created
  int conv_twins = 0;      // I8Conv2DN nodes created
  int ln_twins = 0;        // I8LayerNormN nodes created
  int merged_nodes = 0;    // nodes replaced by the twin nodes
  int bit_outputs = 0;     // I8Add / I8AddN / I8AffineSum outputs turned into bit tensors (bit_outputs)
};

// collapse_unary + compose_luts + fold_lut_thresholds (before Session's out_lut absorption into I8BConv2DRes)
I8RewriteStats rewrite_int8_luts(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& graph_outputs);
// merge_twins (after it)
I8RewriteStats rewrite_int8_twins(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& graph_outputs);

}  // namespace bitino
