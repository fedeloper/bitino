# Operators added for BHViT

BHViT (IMRL/BHViT, CVPR'25) is a binarized hybrid ViT: grouped dilated binary convolutions in stages 1-2,
windowed binary attention with pooled "merge" tokens in stages 3-4, a 2-bit attention-probability
quantizer, and many full-precision shortcut paths. Almost all of it maps onto existing operators
(`docs/operators.md`); this page lists the additions and how each BHViT construct is expressed.

New: `FakeQuant`, `ReduceMean` (`runtime/ops/op_defs_bhvit.cc`, reference kernels in
`runtime/kernels/reference/bhvit_reference.cc`) and the `Activation` type `sign_zero`.
All three are generic. The numpy executor (`python/bitino/executor.py`) performs the same IEEE
operations in the same order, so the numpy f32 executor and the C++ reference are bit-identical
(`tests/cpp/test_bhvit_ops.cc`, `tests/python/test_bhvit_ops.py`).

| Op | Inputs | Attributes | Contract |
|---|---|---|---|
| `FakeQuant` | `x f32[...]`, `scale f32[C]`, `zero_point f32[C]` with `C = x.shape[axis]` | `axis` (default −1), `pre_scale` (1.0), `qmin` (0), `qmax` (1) | Per-channel uniform quantize-dequantize, every step rounded to f32: `t = x·pre_scale; t = t / scale[c]; t = t + zero_point[c]; q = clamp(nearbyint(t), qmin, qmax); y = q·scale[c]`. `nearbyint` is round-half-to-even (= `torch.round`); the kernel requires the default rounding mode. The zero point is **not** subtracted on dequantization (LSQ-style as in BHViT). |
| `ReduceMean` | `x f32[...]` | `axis` (default −1), `keepdims` (0) | Mean over one axis: double accumulation in index order, one division by the count, rounded once. |
| `Activation type=sign_zero` | `x f32` | — | `torch.sign`: `+1` if `x > 0`, `−1` if `x < 0`, `0` otherwise (never `−0`). Float output, for products with non-binary operands (BHViT `P·V`). `sign` keeps its `x ≥ 0 → +1` meaning. |

Numpy mirror changes made at the same time (outputs unchanged whenever the double sums were already
exact): `Softmax` now evaluates `x − max` in the working precision, `exp` with libm in double and a
sequential double sum, exactly like `SoftmaxRef`; `AvgPool2D` and `GlobalAvgPool` accumulate sequentially
in the reference kernels' order; float `Conv2D` and `Dense` start the double accumulator from the bias and add in
`(ky, kx, c)` / `k` order like `Conv2DRef` / `DenseRef`. Before, numpy used BLAS / pairwise sums and its own `exp`,
which is not guaranteed to round identically (it showed up as 2.4e-7 differences on bhvit-tiny-somefp's fp
2×2 downsampling convs). Stored numpy logits of the earlier models are unchanged (spot-checked).

Optimized kernels: `fake_quant.float_opt[_avx2]` performs the same float steps with vector division and
round-half-even (`VROUNDPS` / `FRINTN`), `reduce_mean.float_opt[_avx2]` the same double sums with several
outputs per vector (each lane in index order), so both are **bit-exact**; `Activation sign_zero` runs in
`activation_ext.float_opt[_avx2]` (bit-exact). Softmax (`softmax.float_opt`) and the `P·V` MatMul
(`matmul.float_opt`, or INT8 at `optimized_int8`) are not bit-exact; see `docs/operators.md`.

## BHViT constructs as IR

| BHViT (transformer/BHViT.py, utils_quant.py) | IR |
|---|---|
| `BatchNormT` (eval) | `ChannelAffine` (folded in fp64, stored f32) |
| `QuantizeLinear(h + move)`: `linear(sign(h+move), mean|W|·sign(W − mean_row W))` | `BDense(threshold = −move, input_binarize ge | sign_zero, post_mul = mean|W| per row)` |
| `QuantizeConv2d(x + move)` (token mixer g=4, dilation 1/3/5, `padding='same'`; downsample 2×2 s2) | `BConv2D(groups, dilations, pads, pad_value 0, threshold = −move)`. Grouped convs (16 or 32 channels per group) run on the packed kernels (`bconv2d.neon_opt`, `bconv2d.avx2`, `bconv2d.binary_scalar`) since 2026-10-30; before, they fell back to `bconv2d.reference` |
| `RPReLU`: `prelu(x − move1) + move2` | `ChannelAffine(bias −move1)` → `Activation prelu` → `ChannelAffine(bias move2)` |
| `LayerScale`: `x·alpha + move` | `ChannelAffine`; the `/3`, `/5` of `Shift`/`Shift2` folded into `alpha` |
| `torch.roll` (`Shift`, `Shift2`, `Shift_channel_mix`) | `Reshape [rows, D]` → `Gather` with constant `i32` row indices → `Reshape`. Indices are traced through `torch.roll` / the official `Shift_channel_mix` module on index tensors |
| `ViTIntermediate` shortcut `concat([h]·r, −1)` | `Concat` of the same tensor `r` times |
| `ViTOutput` shortcut `AvgPool1d(r)` over channels | `Reshape [..., C, r]` → `ReduceMean(axis −1)` |
| `windows_split` (7×7) | `Reshape [1, H/7, 7, W/7, 7, C]` → `Transpose [0,1,3,2,4,5]` → `Reshape [nW, 49, C]` |
| merge tokens `a1·MaxPool(7,7,p3) + (1−a1)·AvgPool(7,7,p3)` | `MaxPool2D` / `AvgPool2D(count_include_pad=1)` → `ChannelAffine` ×2 → `Add` |
| `repeat_interleave(merge, nW, dim 0)`, `cat(windows, merge, dim 1)` | `Concat` axis 0 (same tensor `nW` times), `Concat` axis 1 |
| `nn.LayerNorm` (Token_for_Attention) | `LayerNorm` |
| `sign(q)·sign(k)ᵀ / √dh` | `BMatMul(sign_zero × sign_zero, transpose_b, scale 1/√dh)` (exact for `dh` a power of 4) |
| softmax | `Softmax` |
| `BinaryActivation_Attention(heads, 3)(P·3)` | `FakeQuant(axis 1, pre_scale 3, qmin 0, qmax 3)` with the `grad_scale()` forward values of `alpha`, `zero_point` |
| `probs @ sign(v)` | `Activation sign_zero` on V, `MatMul` (exact: `q·alpha` times `{−1,0,1}` sums exactly in double) |
| `token_split`: `window_reverse`, `window_reverse_high` (mean over windows + nearest upsampling) | `Gather` (window tokens to NHWC), `Gather` + `ReduceMean(axis 1)` + `Gather` (upsampling); indices traced through the official `window_reverse` and `F.interpolate(mode='nearest')` |
| stem `Conv2d k4 s4` + BN + `GELU` | `Conv2D` → `ChannelAffine` → `Activation gelu` (double) |
| position embeddings | `Add` with a constant |
| head: BN, `mean(dim=1)`, `Linear` | `ChannelAffine` → `GlobalAvgPool` → `Dense` |
