# PyTorch binary CNNs (BNext, RBONN, IR-Net family, bitorch, SA-BNN, BBCU/BSRC, AdaBin, FDA-BNN, DoReFa AlexNet, BNAS, ReActNet-A with BN: BURN, S2-BNN, BiMatting): operators and mapping

Adapters: `python/bitino/adapters/torch_bnext.py` (hpi-xnor/BNext `src/bnext.py`) and
`python/bitino/adapters/torch_rbonn.py` (SteveTsui/RBONN `2_step2_rbonn/rbonn.py`).
Converter: `tools/convert_torch_cnn.py` (entry points `tools/convert_bnext.py`, `tools/convert_rbonn.py`).
Integration notes: the development log (not published).

## New operator: `Mul`

Files: `runtime/ops/op_defs_torch_cnn.cc` (OpDef), `runtime/kernels/reference/torch_cnn_reference.cc`
(reference kernel `mul.reference`), numpy mirror in `python/bitino/executor.py`, tests in
`tests/cpp/test_torch_cnn.cc` and `tests/python/test_torch_cnn.py`.

| Inputs | Attributes | Contract |
|---|---|---|
| `a f32[S]`, `b f32[S]` (same shape) | — | `y = a · b`, one fp32 rounding per element |

It is the BNext ELM-attention product `x * residual`, where both operands are runtime tensors. No existing
operator multiplies two runtime tensors of the same shape. `ChannelGate` only takes `[N,C]` / `[N,1,1,C]`
gates, and `ChannelAffine` only takes constant per-channel vectors. `a·b` and `b·a` are the same IEEE
product, so the operand order PyTorch uses does not matter.

An earlier draft of this slice had `ChannelMix` (the info-recoupling mix) and a `MulBroadcast` with a fused
sigmoid. Both were removed in favour of existing operators: `ChannelAffine` + `ChannelAffine` + `Add` gives
the same rounding sequence, and `ChannelGate(gate=sigmoid)` (`docs/operators_larq.md`) already evaluates the
sigmoid in double and rounds once, which keeps it bit-reproducible between the C++ reference and numpy.

`mul.float_opt[_avx2]` and `channel_gate.float_opt[_avx2]` (gate sigmoid in double, then one fp32 product)
are bit-exact; at `optimized` stages both are usually absorbed into `FusedElementwise` nodes or into the
epilogue of the preceding `BConv2D` (`docs/operators.md` § Plan-time rewrites).

## Constructs that reuse existing operators

| Source construct | IR | Exactness |
|---|---|---|
| `LearnableBias` (`x + bias`) | `ChannelAffine(x, —, bias)` | identical fp32 add |
| `torch.sign` through the STE expression `(sign(x) − r).detach() + r` (BNext `HardSign`, `r = clamp(x, ±1.5)`; RBONN `BinaryActivation`, `r` = piecewise-quadratic ApproxSign) | `BConv2D` `input_binarize = sign_zero` (exact) or `ge` (deployable) | `fl(fl(s − r) + r) = s` for `s = sign(r)`, `|r| ≤ 1.5`. For `|r| ≥ ½`, `s − r` is exact (Sterbenz). Below that, the error of `s − r` is at most half an ulp of 1, so the sum rounds back to `s`. Checked numerically with forward hooks on all 128 eval samples, fp32 and fp64 (0 mismatches). |
| BNext `HardBinaryConv` with `temperature < 1e-7` | `BConv2D` with `sign(W)` bits, no `post_mul`, `pad_value 0` | the adapter raises if `temperature ≥ 1e-7` (weights would be `clamp(W/T, ±1)`) or if a latent weight is exactly 0 |
| RBONN `RBOConv` | `BConv2D` with `sign(W)` bits, `post_mul = 0.02·|scale_factor|` | folded in float64 and rounded once |
| `BatchNorm2d` (eval) | `ChannelAffine(scale, shift)`, `scale = γ/√(var+ε)`, `shift = β − mean·scale` in float64 | the fused variant folds BN after `Conv2D`/`BConv2D` into weights or `post_mul`/`post_add` (`passes.fold_affine_into_conv`) |
| BNext info-recoupling `scale·input + (1−scale)·x` | `ChannelAffine(input, scale)`, `ChannelAffine(x, 1−scale)`, `Add` | same rounding sequence as PyTorch eager: `fl(fl(s·a) + fl((1−s)·b))`. `1−s` is folded in float64, and that equals PyTorch's fp32 `1−s` because the difference is exact in float64. `FFN_1x1.forward` clamps `scale` to `[0,1]` even in eval; the adapter mirrors that (the tiny checkpoint has a scale of −8e−8). |
| BNext `SqueezeAndExpand("sigmoid")` | `GlobalAvgPool(keepdims=1)` → `Conv2D 1x1 (relu)` → `Conv2D 1x1` → `ChannelGate(x, gate, gate=sigmoid)` | sigmoid in double, rounded once (existing op) |
| BNext ELM attention `x * residual` (residual = PReLU1(input), pooled if stride 2) | `Mul` (new) | one fp32 product |
| `nn.PReLU(C)` | `Activation prelu` | slopes stored verbatim |
| `AvgPool2d(2,2)` (downsample on residual and on the mix input) | `AvgPool2D` | |
| `torch.cat(dim=1)` (FFN_1 ‖ FFN_2) | `Concat(axis=−1)` | |
| Unused modules (BNext stem PReLU, `activation1` of FFN units, `norm2` of FFN units without recoupling; RBONN `binary_conv.u`) | not converted | listed in `info["unused_in_forward"]`; the adapter checks that every other parameter and buffer is consumed |

## IR-Net family (ReCU, RBNN, SiMaN, BiPer), SNN and Bi-Real-20

No new operator. Adapters `torch_irnet` (ReCU/RBNN/SiMaN/BiPer ImageNet ResNet-18/34, CIFAR ResNet-18/20,
VGG-Small; third-party Bi-Real-20) and `torch_rbonn` (SNN ReActNet-ResNet-18 W1A1 and 0.67-bit);
notes in the development log (not published).

| Source construct | IR | Exactness |
|---|---|---|
| `BinarizeConv2d` weights (ReCU `sign(clamp(standardized w))`, RBNN `sign(rotated standardized w)`, BiPer `sign(sin(freq·w))`, SiMaN `2B − 1`) | `BConv2D` sign bits | evaluated with the source's own fp32 torch expression; cross-checked against the tensor the source passes to `F.conv2d` (0 mismatches). A weight that is not exactly ±1 is rejected |
| `output * alpha` (`alpha` [O,1,1]) | `BConv2D post_mul = alpha` | exact fp32 parameter |
| RBNN / SiMaN activation `sign((a − mean_CHW a) / std)` | `Reshape [1, HWC]` → `ReduceMean(axis −1, keepdims)` → `ChannelAffine(scale −1)` → `AddBroadcast(a, −mean)` → `BConv2D` | the positive per-sample divisor does not change the sign; mean in double, rounded once (torch: fp32 reduction, rounding-level difference) |
| `nn.Hardtanh` / `F.hardtanh` | `Activation hardtanh(−1, 1)` | `clip`, exact |
| `AvgPool2d(7)` / `F.avg_pool2d(out, H)` over the whole map | `GlobalAvgPool` | |
| `BatchNorm1d` before the classifier | `ChannelAffine` on `[1, C]` (float64 fold) | |
| CIFAR option-A shortcut `F.pad(x[:, :, ::2, ::2], channels ±planes/4)` | `AvgPool2D(kernel 1, stride 2)` + `Pad` (channel axis) | exact (1-tap average); the adapter checks the opaque `LambdaLayer` numerically against the formula |
| BiPer CIFAR shortcut `AvgPool2d(2)` + channel `F.pad` | `AvgPool2D(2, 2)` + `Pad` | |
| VGG-Small `x.view(N, −1)` (NCHW flatten) + `Linear` | `Reshape` + `Dense` with columns permuted to NHWC order | |
| SNN / Bi-Real-20 `mean\|W\|·sign(W) − clamp(W) + clamp(W)` | `BConv2D` sign bits, `post_mul = mean\|W\|` (float64, rounded once) | the STE terms cancel up to fp32 rounding (≤ ulp(\|clamp W\|)) in the source; exactly in float64 limit |
| SNN 0.67-bit `Binarize` (nearest of 64 binary 3×3 patterns) | decoded ±1 kernels → `BConv2D` | decoded with the source's `weights2patterns` on fp32 weights; strict W1A1 forward |
| Bi-Real-20 `BiRealConv` bias (passed into `nn.Conv2d`'s `dilation` slot by the source, so the module keeps a trained bias) | `BConv2D post_add` | exact fp32 parameter |

## bitorch model hub (ResNetE-18, BinaryDenseNet-28, MeliusNet-22) and AB-BNN ReActNet-A

No new operator. Adapter `torch_bitorch` (bitorch 0.3.0 model code, the release matching the hub
checkpoints) and `torch_abbnn` extended to ReActNet-A and the CIFAR stems; notes in
the development log (not published).

| Source construct | IR | Exactness |
|---|---|---|
| bitorch `QConv2d` (eval): `conv2d(pad(Sign(x), value −1), Sign(W))`, `Sign` = `torch.sign` with 0 → +1 (also −0.0); `ProgressiveSign` weights are `Sign` when not training | `BConv2D input_binarize = ge`, `pad_value = −1`, `(W ≥ 0 ? +1 : −1)` bits, no `post_mul` | exact: `ge` *is* the source semantics, so there is no `sign_zero` variant. Input and weight quantizers are probed against the live modules (incl. ±0 and ±1e−30) |
| bitorch `DenseLayer` (`cat([x, QConv2d(BN(x))])`) | `ChannelAffine` → `BConv2D` → `Concat(axis −1)` | |
| bitorch MeliusNet `ImprovementBlock`: `cat(x[:, :C−c], x[:, C−c:] + y)` | `Pad(y, C−c channels in front, 0)` + `Add(x, ·)` | adding +0.0 is exact except −0.0 → +0.0, which the following BN maps to the same value |
| bitorch transition `bn,max_pool,relu,fp_conv`; ResNetE downsample `AvgPool2d(2) → Conv2d 1×1 → BN` | `ChannelAffine`, `MaxPool2D`, `Activation relu`, `Conv2D`; `AvgPool2D`, `Conv2D`, `ChannelAffine` | ResNetE `BasicBlock.activation` (ReLU) is constructed but unused (`unused_in_forward`) |
| AB-BNN ReActNet-A block: `x' + α·Bconv3x3(sign(move11(x)))` (`x' = AvgPool2d(2)(x)` at stride 2), then `α·Bconv1x1(sign(move21(out1))) + out1`, or two 1×1 binary convs on the same binarized input + `out1` each, concatenated | `ChannelAffine` → `BConv2D` (`post_mul = gain·mean\|W_std\|·α`) → `Add`; `Concat(axis −1)` (down1 ‖ down2) | as the ReActNet-18 path; a `W_std` that is exactly 0 only in fp32 (ReActNet-A ImageNet: 1 weight) follows the float64 sign |
| AB-BNN CIFAR stems (`ScaledStdConv2d` 3×3 stride 1, `nn.Identity` max-pool; ReActNet-A `firstconv3x3` stride 1) | `Conv2D` (weight standardization folded in float64) | |

## SA-BNN ResNet-18 and IDa-Det BiRes-18 backbone (slice S1 remainder)

No new operator. Adapters `torch_sabnn` (new; anonymous SA-BNN Drive `birealnet.py.txt`) and `torch_rbonn`
(extended with IDa-Det's `BiConv` block); converters `tools/convert_sabnn.py`, `tools/convert_idadet.py`; notes in
the development log (not published).

| Source construct | IR | Exactness |
|---|---|---|
| SA-BNN block: `a = bn2(x)`; `r = downsample(a)` (the BN'd input) or `x`; `h = prelu(bn1(bconv(sign(a))) + r)`; `out = prelu2(bn4(bconv2(sign(bn3(h)))) + r)` (one shortcut used by both adds) | `ChannelAffine` (pre-sign BN, materialized: it also feeds the downsample) → `BConv2D` → `ChannelAffine` → `Add` → `Activation prelu`, twice per block | pre-sign BN folded in float64, stored fp32 (folding = the only rounding difference) |
| SA-BNN `downsample2` (same module object as `downsample`, applied only if `out.size() != residual.size()`) | not converted | never runs: `residual` always has the block output shape; the adapter checks that it is the shared module (`unused_in_forward`) |
| SA-BNN `HardBinaryConv` (flat `(N,1)` weights viewed as `(O,I,3,3)`, `mean\|W\|·sign(W)` STE) | `BConv2D` sign bits, `post_mul = mean\|W\|` (float64, rounded once) | as SNN |
| IDa-Det `BiConv` (`binfunc` = `BinaryActivation` inside the conv; `Binarization` = `sign(W)·0.02·\|scale\|`; trained conv bias) | `BConv2D` sign bits, `post_mul = 0.02·\|scale\|`, `post_add = bias` | as RBONN; `F.conv2d` adds the bias after the scaled sum, `BConv2D` applies `acc·post_mul + post_add` |
| IDa-Det downsample `AvgPool2d(2, 2, ceil_mode=True)` | `AvgPool2D(2, 2)` | equal to floor mode when `(H − 2) % 2 == 0` (56/28/14 at 224×224); the adapter tracks the spatial sizes and rejects other cases |

## Binary image restoration and super-resolution: BBCU, BSRC-2023 (slice S8)

No new operator. PixelShuffle and the nearest / bilinear resizes are exact compositions of existing ops
(`python/bitino/adapters/restoration_common.py`). Adapters `torch_bbcu` (Zj-BinXia/BBCU @c47df53: DnCNN
denoising and JPEG deblocking, BBCU-L / BBCU-M SR) and `torch_bsrc` (ilyazhara/BSRC-2023 @62bbcf2 challenge
baselines); converters `tools/convert_bbcu.py`, `tools/convert_bsrc.py` (shared `tools/convert_restoration.py`);
notes in the development log (not published). The graph output is an image (NHWC), validated with
`tools/validate_model.py --output-kind image` (`docs/numerical-correctness.md`).

| Source construct | IR | Exactness |
|---|---|---|
| `nn.PixelShuffle(r)` (channel `c·r² + i·r + j` → pixel `(h·r+i, w·r+j)`) | `Reshape [1,H,W,C,r,r]` → `Transpose [0,1,4,2,5,3]` → `Reshape [1,rH,rW,C]` | data movement |
| `F.interpolate(mode='nearest', scale_factor=r)` (BBCU-M `BinaryUpConv2d` shortcut) | `Reshape [HW, C]` → `Gather` (constant i32 ids `(y//r)·W + x//r`, shape `[rH, rW]`) → `Reshape` | copy |
| `F.interpolate(mode='bilinear', align_corners=False, scale_factor=r)`, r ∈ {2, 4} (BBCU global skip) | replicate pad 1 (`Gather`, ids `clamp(y−1)·W + clamp(x−1)`) → `Conv2D` 3×3, no padding, `C·r²` phase outputs with the dyadic weights `(−d, 1+d)` / `(1−d, d)`, `d = (i+½)/r − ½`, products of the two axes → PixelShuffle | torch clamps the source coordinate at 0 and the upper neighbour at `H−1`; replicate padding reads the border pixel twice and `(−d)x + (1+d)x = x` exactly. Bit-identical to torch in float64 for float32-valued inputs (test); fp32: ≤ 1 ulp (summation order) |
| BBCU `BinaryConv2d`: `RPReLU(HardBinaryConv(sign(x + move0))) + x` | `ChannelAffine(bias)` → `BConv2D` (sign bits, `post_mul = mean\|W\|` float64, `post_add` = conv bias for deblocking, `pad_value 0`) → `ChannelAffine(bias)` → `Activation prelu` → `ChannelAffine(bias)` → `Add` | `BinaryActivation` = `torch.sign` exactly (RBONN argument, hook-checked): `sign_zero` (exact) / `ge` (deployable) variants |
| BBCU `x·k`, `(x·k − n)/k`, `out/k` (k = 130) | `ChannelAffine(k)`; `Add(xk, ChannelAffine(n, −1))`; `ChannelAffine(1/k)` | `x·k` and the negation are the same fp32 ops; `·fl(1/k)` vs torch's `/k`: ≤ 1 ulp, after the last binarization (float64 constant in the Level-3 check) |
| `nn.LeakyReLU(0.1)` | `Activation leaky_relu(alpha 0.1)` | |
| BSRC `BinaryConv2d` (input `sign(sign(x)+0.1)`, weights `sign(sign(W)+0.1)·mean\|W\|` per output channel, bias) | `BConv2D input_binarize ge` (0 → +1 is the source semantics), `(W ≥ 0 ? +1 : −1)` bits, `post_mul = mean\|W\|`, `post_add = bias`, `pad_value 0` | Parabola / STE / STE-with-clip forward terms `(s − b) + b` return `s` exactly (`\|b\| ≤ 1`, same sign or 0; hook-checked) |
| BSRC `LAB`: depthwise `Conv2d(C → 2C, groups C)`, forward value `argmax over (x0, x1)·2 − 1` | `Conv2D(groups C)` → `Conv2D 1×1 (groups C, weights −1/+1)` → `BConv2D input_binarize gt` | the difference of two rounded floats has the exact sign and is 0 iff `x0 = x1` (argmax tie → index 0 → −1 = `gt`) |
| BSRC `Scale` (`variance·x + mean`), `MeanShift`, `Normalize` (`(x − 127.5)/51`, inverse `x·51 + 127.5`) | `ChannelAffine` | same fp32 ops except `/51` → `·fl(1/51)` (≤ 1 ulp, before the first conv) |
| BSRC grouped / depthwise FP convs (`groups = C/16`, `groups = C`), 9×9 convs, `BatchNorm2d` | `Conv2D(groups)`, `ChannelAffine` (float64 BN fold) | |

## AdaBin, FDA-BNN, tensorpack DoReFa-Net AlexNet (slice S12)

No new operator. Adapters `torch_adabin` (official AdaBin PyTorch model), `mindspore_fdabnn` (FDA-BNN tensors read from
the MindSpore `.ckpt` by `mindspore_ckpt`, a protobuf wire-format reader; no MindSpore) and `tensorpack_dorefa` (npz
tensors); converters `tools/convert_adabin.py`, `tools/convert_fdabnn.py` (shared `tools/convert_s12_cifar.py`) and
`tools/convert_tensorpack_dorefa.py`; notes in the development log (not published).

| Source construct | IR | Exactness |
|---|---|---|
| AdaBin `BinaryActivation`: `a = α·(sign((x−β)/α) + β)` (the `gradient_approx` STE returns `torch.sign` exactly, hook-checked), i.e. `a = p + q·s`, `s = sign(x − β)`, `p = αβ`, `q = α` (levels `α(β±1)`); requires `α > 0` | operand `s`: `BConv2D` `threshold = β`, `sign_zero` (exact; `s = 0` gives `a = αβ = p`) or `ge` | `sign(fl((x−β)/α)) = sign(x − β)` for `α > 0` |
| AdaBin weights `sign((w − mean_c)/std_c)·std_c + mean_c` (population std): two levels `W±` per output channel, `w = c_o + d_o·s_w` | `BConv2D` sign bits | `W±` are evaluated with the source's fp32 expression and checked bit for bit against the tensor passed to `F.conv2d` |
| AdaBin `F.conv2d(a, w, padding=1)` with **zero** padding (0 is not in the binary set) + `BatchNorm2d` | `Σ(c+d s_w)(p+q s) = q d·BConv2D(s_w, s) + q c·Σ_inb s + p(d·Σ_inb s_w + c·K_inb)`: main `BConv2D` (`pad_value 0`, `post_mul = g q d`), one-output-channel all-(+1) `BConv2D` (`Σ_inb s`) → `Conv2D 1×1` (weights `g q c`), `Add` of a constant `[1,OH,OW,O]` map `g p(d Σ_inb s_w + c K_inb) + h` (border taps missing; computed from the padding mask by float64 convs of an all-ones input) | exact decomposition (float64 IR with float64 constants vs the official float64 model: ≤ 3.9e-14) |
| AdaBin `Maxout`: `pos·relu(x) − neg·relu(−x)` (per channel, scales of either sign) | `Activation relu` → `ChannelAffine(pos)`; `ChannelAffine(−1)` → `Activation relu` → `ChannelAffine(−neg)`; `Add` | bit-exact with the source in fp32 (one of the two products is ±0; `fl(−n·r) = −fl(n·r)`) |
| FDA-BNN `QuanConv`: `conv2d(Sign(x), sign(W)·mean\|W\|)` (MindSpore `Sign(0) = 0`, one scale per layer, zero padding) | `BConv2D` sign bits, `post_mul = mean\|W\|` (float64), `sign_zero` / `ge` | a latent weight that is exactly 0 is rejected |
| FDA-BNN `BasicCell`: `bn2(prelu2(conv2(bn1(prelu1(conv1(x)))))) + shortcut`, scalar PReLU, no post-add activation; head `relu → ReduceMean(2,3) → Dense` | `BConv2D` → `Activation prelu` (slope broadcast) → `ChannelAffine`; `Add`; `Activation relu` → `GlobalAvgPool` → `Dense` | |
| DoReFa `fa(clip(x, 0, 1))` with 1 bit = `round(x)` (half to even) ∈ {0, 1}, = 1 iff `x > 0.5` | operand `s = gt(x, 0.5)`, `u = (1 + s)/2` | exact (no tie ambiguity: `gt` is the source rule, single `model.bitino` variant) |
| DoReFa `fw` (1 bit): `where(W == 0, 1, sign(W/E))·E`, `E = mean\|W\|` per layer; SAME zero padding of `u` (0 is a level); grouped convs (`split = 2`) | `BConv2D gt, threshold 0.5, pad_value −1, groups`, `post_mul = E/2`, `post_add = E/2·Σ_taps w` (`Σ u w = (Σ w + Σ s w)/2`; a padded tap has `s = −1`, i.e. `u = 0`) | integer-exact, as the NNCF {0, s} mapping; packed kernels incl. NEON (grouped layouts) |
| DoReFa FC layers (`fc0` + bias, `fc1`) | `BDense gt, threshold 0.5`, `post_add = E/2·Σw (+ bias)`; the NCHW flatten before `fc0` → fc0 rows permuted to NHWC | |
| DoReFa BN (EMA statistics, ε = 1e-4) → `MaxPooling(3, 2, SAME)` → quantizer | `ChannelAffine` → `MaxPool2D` (TF SAME pads bottom/right, padding ignored) → next binary layer | the max-pool acts on the real BN output before the threshold, as the source |
| DoReFa `bnfc1 → clip(0, 1) → fct` | `ChannelAffine` → `Activation hardtanh(0, 1)` → `Dense` | |

## BNAS cell networks (slice S13)

No new operator. Adapter `torch_bnas` (official yonseivnl/bnas `NetworkCIFAR` / `NetworkImageNet` after the official
`BinOp.binarization()`), converter `tools/convert_bnas.py`; notes in the development log (not published).
Reference = the **first release** (commit cdb58f7), the code the checkpoints were trained with: its BinOp also binarizes
the `preprocess_res` convs (stale loop variable in the `'res'` test); the current HEAD (675ecc7, "typo fixes") does not,
and does not match the released weights.

| Source construct | IR | Exactness |
|---|---|---|
| `BinReLUConvBN` / `BinDilConv`: `relu(conv(sign(bn(x))))`, `torch.sign` (0 → 0), zero padding after binarization, dilation 2 for `bin_dil_conv_*`; weights `sign(w')·mean\|w'\|` per output channel, `w' = clamp(w − mean_ci w, −1, 1)` (BinOp) | `ChannelAffine(bias b/s)` → `BConv2D(threshold none, pad_value 0, dilations, post_mul = mean\|w'\|, relu)`, `sign_zero` (exact) or `ge`; `sign(s·x + b) = sign(s)·sign(x + b/s)`: the BN's `sign(s)` is moved into the weight bits of that input channel | weight bits / scale read from the BinOp-binarized tensor (every tap checked to be ±scale, no zero); `fl(x + b/s)` has the exact sign of `x + b/s` (one f32 rounding of the threshold, no `fl(fl(s·x)+b)` near-ties); the plan-time threshold folding then removes the ChannelAffine |
| BinOp counting quirk: the last *#preprocess_res* non-res convs are not weight-binarized (fp32 latent weights) but still get `sign(bn(x))` inputs | `ChannelAffine(bias b/s)` → `Activation(sign_zero)` → `Conv2D(relu)` (input channels with `s < 0` negated) in both variants | exact (float sign_zero kernels are bit-exact) |
| `Zero` op (`x.mul(0.)`) | edge dropped (`h + 0 = h`); an all-zero state that is concatenated → `ChannelAffine(state, scale 0)`; branches feeding only `Zero` ops are not emitted | exact |
| Structurally zero channels reaching `sign()` (layer-0 cell's all-zero states carried along the residual chain, BN with folded shift 0) | `sign_zero`: as is; `ge`: `BConv2D(act none)` + `Add` of the constant map `−m[o]·Σ_{zero channels, in-bounds taps} w` + `relu` | exact correction (BNAS-A `cells.7.preprocess0`, 72 of 144 input channels) |
| `MaxPool2d(3, s, 1)` / `AvgPool2d(3, s, 1, count_include_pad=False)` | `MaxPool2D` / `AvgPool2D(count_include_pad 0)`, identical pools of one state shared | |
| Cell output `cat(states) (+ res1 \| + preprocess_res(res1))`, `preprocess_res = BN + Conv2d(2×2, s2, bias)` | `Concat` (channels) + `Add`; `ChannelAffine` → `Conv2D` | |
| First-release BinOp: `preprocess_res` conv weight-binarized (`sign(w')·mean\|w'\|`), input = real BN output (no sign), bias kept | `Conv2D` with the binarized weight values (exact fp32 values) + bias | exact (binary weights × real input is not a BConv2D) |
| Head `AdaptiveAvgPool2d(1)` / `AvgPool2d(7)` on the 7×7 map + `Linear` | `GlobalAvgPool` + `Dense`; ImageNet: rows permuted to the standard class order (official labels = ImageFolder over folders named `0..999`, sorted as strings) | |

## ReActNet-A with BatchNorm: BURN, S2-BNN, BiMatting backbone (slice S11)

No new operator. `torch_abbnn._convert_reactnet_a` (the AB-BNN ReActNet-A skeleton; AB-BNN graphs unchanged byte for byte)
gained a BN flavour for BURN (`ReActnet` + `BinaryConv2d`), S2-BNN (`ReActNet` + `HardBinaryConv`) and the BiMatting
ImageNet backbone (`BiMobileNet` + `BiConv2d`); converter `tools/convert_reacta_bn.py`; notes in
the development log (not published).

| Source construct | IR | Exactness |
|---|---|---|
| Stem `Conv2d(3, C, 3, s2, p1)` + BN | `Conv2D` + `ChannelAffine` (BN folded in float64; the fused model folds it into the conv) | |
| `sign(move(x))` → binary conv (`torch.sign`, zero padding) → BN → `+ residual` | `ChannelAffine(bias)` → `BConv2D(pad 0, post_mul = mean\|W\|·s_bn, post_add = t_bn)` → `Add` | BN folded in float64, one rounding per constant; `sign_zero` (exact) or `ge` |
| Weights: `BinaryConv2d` `sign(W)·mean\|W\|`; `HardBinaryConv` / `BiConv2d` STE `s·sign(W) − clamp(W) + clamp(W)` (BiConv2d on `W − mean W`) | `sign` bits (float64 centring), `mean\|·\|` in float64 | the STE terms cancel exactly in exact arithmetic; the fp32 source differs from `s·sign` on ~9% of the taps by ≤ 1.2e-7·s (fp32 noise floor) |
| Binarization after a PReLU tail: source `sign(fl(fl(q + b_prev) + b_next))` (q = PReLU output, `move13 → move21`, `move23 → next move11`) | `ChannelAffine(q, bias fl64(b_prev + b_next))` → `BConv2D`; the residual keeps `fl(q + b_prev)`; the plan-time threshold folding makes it the exact comparison `q ≥ −(b_prev + b_next)` | equal in exact arithmetic; removes the source's fp32 ties (channels with tiny PReLU slope and nearly cancelling biases). `literal_thresholds=True` mirrors the two fp32 additions |
| `nn.PReLU(C)` | `Activation prelu` (fp32 slopes) | |
| Channel doubling `cat(Bconv1x1_down1(b) + out1, Bconv1x1_down2(b) + out1)`; BiMatting ×4: four convs | `BConv2D` + `Add` per branch → `Concat(axis −1)` | |
| BiMatting half block: `Bconv1x1(2P → P)(b) + sum(torch.split(out1, P, 1)) / 2` | `Reshape [1, H·W, 2, P]` → `AvgPool2D [1, 2]` → `Reshape [1, H, W, P]`, `Add` | bit-exact: one rounding of `a + b`, halved (tested on numpy and C++ reference / neon / optimized) |
| BiMatting `forward_single_frame` macro residuals `x += AvgPool2d(2, 2, ceil_mode=True)(feature)` after blocks 3, 6, 12 | `AvgPool2D` + `Add` | ceil mode = floor mode for the even map sizes (checked at conversion) |
| Heads: BURN `pool → Linear(1024, 1024) + ReLU` (backbone output used by linear eval); S2-BNN `pool` (fc = Identity, `main_lincls.py` feature); BiMatting `pool → Linear(1024, 1000)` (checkpoint `fc`, not in the released backbone code) | `GlobalAvgPool` (+ `Dense` (+ `relu`)) | |
| Unused parameters: `BiConv2d.channel_threshold` (all 0, never read), S2-BNN blocks' untaken `binary_pw` / `binary_pw_down*` branches | not emitted (`unused_in_forward`) | |

## Binarization zeros

`torch.sign(0) = 0` contributes 0 to the convolution. The deployable `ge` variant maps 0 to +1.
The converter counts exact zeros at every binarization input in the source (fp32 and fp64)
and in the numpy executor. See `conversion_report.json` and the integration notes for the
per-model counts.
