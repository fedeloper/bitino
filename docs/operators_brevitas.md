# Brevitas / FINN BNN-PYNQ models as Bitino IR

Covers the Brevitas BNN-PYNQ CNV/LFC/SFC/TFC 1W1A models (`python/bitino/adapters/brevitas_bnn.py`,
`tools/convert_brevitas.py`). **No new operator is needed.** Every layer maps to existing operators
(`BConv2D`, `BDense`, `Conv2D`, `MaxPool2D`, `ChannelAffine`, `Activation(sign)`, `Reshape`) plus the generic
`FakeQuant` introduced for BHViT (`docs/operators_bhvit.md`). Tests: `tests/cpp/test_brevitas_ops.cc`,
`tests/python/test_brevitas_bnn.py`.

## CNV's 8-bit input as `FakeQuant`

Brevitas `QuantIdentity(bit_width=8, min_val=-1, max_val=1-2^-7, narrow_range=False,
restrict_scaling_type=POWER_OF_TWO)` computes, in `IntQuant` order (`brevitas/core/quant/int_base.py`),
`y = (clamp(round(x/s + zp), qmin, qmax) − zp)·s` with `torch.round` (half to even). Probed values:
`s = 1/128`, `zp = 0`, `[−128, 127]`. The generic `FakeQuant(x, scale[C], zero_point[C])` with `pre_scale = 1`,
`scale = 1/128`, `zero_point = 0`, `qmin = −128`, `qmax = 127` performs `t = x·1; t = t/s; t = t + 0;
q = clamp(nearbyint(t)); y = q·s`, one f32 rounding per step. With `zp = 0` this is the Brevitas expression
bit for bit, including signed zeros (`x = −0.0` → `t = −0.0 + 0 = +0.0` in both).

On ToTensor inputs (`k/255`, then `2x−1`), no value lies within 0.0019 of a rounding tie, so the quantized input is
the integer `clamp(round(128(2k−255)/255), −128, 127)` for every pixel value `k`. The adapter checks this exactly
against the Brevitas module (all 256 values plus a 1/512 grid, float32 and float64). The converter checks it
against FINN's input MultiThreshold (255 thresholds per channel) in the FINN-ONNX export (`--finn-onnx`).
`FakeQuant` has a reference kernel only. It runs once on the 32×32×3 input.

## Mapping of the Brevitas BNN-PYNQ layers

| Brevitas | Bitino (unfused) | Bitino (fused, `model.bitino`) |
|---|---|---|
| `2x - 1` in `forward` | `ChannelAffine(2, -1)` | CNV: same. FC: folded into the first `BDense` threshold 0.5 (exact) |
| `QuantIdentity` 8-bit (CNV input) | `FakeQuant` | `FakeQuant` |
| `QuantIdentity` 1-bit (`ClampedBinaryQuant`, `ge`) | `Activation(sign)` | absorbed: next op binarizes with `ge` |
| `QuantConv2d` W1, first CNV layer (8-bit input) | `Conv2D` with ±1 f32 weights | `Conv2D` with weights `s·w` (±1), bias `−s·T/128` |
| `QuantConv2d` W1A1 (3×3, VALID) | `BConv2D` (`ge`, pads 0) | `BConv2D` with `post_mul = s`, `post_add = −s·T`, `output = bit` when the consumer is a BConv2D |
| `QuantLinear` W1A1 | `BDense` (`ge`) | `BDense` with `post_mul = s`, `post_add = −s·T` |
| `BatchNorm2d/1d` | `ChannelAffine` (float64-folded, f32) | integer threshold `T` per channel (above) |
| `MaxPool2d(2)` on ±1 | `MaxPool2D` | `MaxPool2D` on `s·(acc − T)` (ge-sign commutes with max) |
| `TensorNorm` (scalar affine) | `ChannelAffine` | `post_mul/post_add` of the last `BDense` |
| `Dropout` | (identity, eval) | — |

Integer threshold: with `tau = mean − beta·sqrt(var+eps)/gamma` (float64), `s = sign(gamma)`,
`T = ceil(tau)` for `gamma > 0` and `floor(tau)` for `gamma < 0`, then `s·(acc − T) ≥ 0 ⇔ BN(acc) ≥ 0` for every
integer accumulator. The equivalence is exact in float32 and float64 and is tested exhaustively over all attainable
accumulators in `tests/python/test_brevitas_bnn.py`. The adapter reports, per layer, the distance from `tau` to
the nearest attainable accumulator (parity of K respected). If that distance were ~1e-12, the float64
computation of `tau` itself could pick the wrong integer. The smallest distance observed is 2.5e-4
(accumulator units, CNV `linear_features.1`).
