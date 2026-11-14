#pragma once

#include "bitino/op.h"

namespace bitino {

void register_op_defs(OpRegistry& reg);
void register_larq_ops(OpRegistry& reg);  // op_defs_larq.cc

// Kernel families (one function per kernels/ subdirectory).
void register_reference_kernels(OpRegistry& reg);
void register_binary_scalar_kernels(OpRegistry& reg);
void register_binary_neon_kernels(OpRegistry& reg);
void register_binary_neon_opt_kernels(OpRegistry& reg);
void register_binary_avx2_kernels(OpRegistry& reg);  // binary_avx2/ (x86-64 only, empty elsewhere)
void register_int8_kernels(OpRegistry& reg);
void register_int8_neon_kernels(OpRegistry& reg);
void register_int8_avx2_kernels(OpRegistry& reg);  // kernels/int8_avx2 (x86-64; no-op elsewhere)
void register_float_opt_kernels(OpRegistry& reg);
void register_larq_kernels(OpRegistry& reg);  // reference/larq_reference.cc
void register_glue_float(OpRegistry& reg);    // reference/glue_float_reference.cc (InstanceThreshold, ChannelShift)

// Transformer operators (runtime/ops/transformer_op_defs.cc) and their kernels.
void register_transformer_op_defs(OpRegistry& reg);
void register_transformer_reference_kernels(OpRegistry& reg);
void register_ternary_kernels(OpRegistry& reg);

// BHViT operators (runtime/ops/op_defs_bhvit.cc) and their reference kernels.
void register_bhvit_op_defs(OpRegistry& reg);
void register_bhvit_reference_kernels(OpRegistry& reg);

// PyTorch binary-CNN operators (runtime/ops/op_defs_torch_cnn.cc) and reference kernels.
void register_torch_cnn_ops(OpRegistry& reg);      // op_defs_torch_cnn.cc
void register_torch_cnn_kernels(OpRegistry& reg);  // kernels/reference/torch_cnn_reference.cc

// Image-feature operators (runtime/ops/op_defs_image_features.cc, docs/operators_image_features.md).
void register_image_feature_ops(OpRegistry& reg);               // op_defs_image_features.cc
void register_image_feature_reference_kernels(OpRegistry& reg);  // kernels/reference/image_features_reference.cc
void register_image_feature_opt_kernels(OpRegistry& reg);        // kernels/image_features_opt.cc (neon / binary_scalar)

// Internal FusedElementwise operator (created by the session's plan-time rewrite,
// src/graph_rewrite.h): OpDef, reference kernel, float_opt kernels.
void register_fused_ops(OpRegistry& reg);                  // op_defs_fused.cc
void register_fused_reference_kernels(OpRegistry& reg);    // kernels/reference/fused_reference.cc
void register_float_opt_fused_kernels(OpRegistry& reg);    // kernels/float_opt/fused_elementwise.cc (+ Mul, ChannelGate)
void register_int8_matmul_kernels(OpRegistry& reg);  // kernels/int8_reference/int8_matmul.cc (int8_reference, int8_neon, int8_avx2)
void register_bmatmul_kernels(OpRegistry& reg);  // kernels/binary_scalar/bmatmul_bitplane.cc (portable, NEON, AVX2)
void register_float_opt_transformer_kernels(OpRegistry& reg);  // kernels/float_opt/transformer_opt.cc (transformer / BHViT ops, Concat, Gather)

// Full-integer I8* operators (stage int8_full, docs/int8.md): OpDefs, portable reference kernels,
// NEON kernels (AArch64; no-op elsewhere).
void register_int8_full_op_defs(OpRegistry& reg);           // ops/op_defs_int8_full.cc
void register_int8_full_reference_kernels(OpRegistry& reg); // kernels/int8_full/int8_full_reference.cc
void register_int8_full_neon_kernels(OpRegistry& reg);      // kernels/int8_full/int8_full_neon.cc
void register_int8_full_portable_kernels(OpRegistry& reg);  // kernels/int8_full/int8_full_portable.cc (tier int8_full_opt)
void register_int8_glue_op_defs(OpRegistry& reg);           // ops/op_defs_int8_glue.cc (I8ChannelShift, I8BinarizeDyn, I8SimAM)
void register_int8_glue_kernels(OpRegistry& reg);           // kernels/int8_full/int8_full_glue.cc

}  // namespace bitino
