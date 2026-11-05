# Operator contracts: Larq Zoo extensions

Operators added for the remaining Larq Zoo models (DoReFa-Net, Real-to-Binary Net).
Conventions are those of `operators.md`: NHWC activations, OHWI filters, explicit
`pads [top, bottom, left, right]`, fused `activation ∈ {none, relu, relu6}`.

| Where | File |
|---|---|
| OpDefs (validation, shape inference) | `runtime/ops/op_defs_larq.cc` |
| Reference kernels | `runtime/kernels/reference/larq_reference.cc` |
| numpy mirror | `python/bitino/executor.py` |
| Tests | `tests/cpp/test_larq_ops.cc`, `tests/python/test_larq_ops.py`, `tests/python/test_keras_larq_adapter.py` |

Only reference kernels exist (tier `reference`). They are selected at every benchmark
stage because every stage includes the reference tier.

## QBConv2D (k-bit activations × binary weights)

Inputs: `x f32[N,H,W,C]`, `w bit[O,KH,KW,C/groups]`, optional `post_mul f32[O]`, `post_add f32[O]`.
Attributes: `strides, dilations, pads, groups, activation`, `act_bits k ∈ [1, 8]`,
`act_quant = "dorefa"` (the only mode), `pad_value` (must be 0).

1. Quantize the input to integer levels (DoReFa, larq `DoReFa(k_bit=k, mode="activations")`):
   `n = 2^k − 1`, `level(x) = round_half_even(clip(x, 0, 1) · n)` with the product
   evaluated **in float32** (one rounded multiply; `-ffp-contract=off`), so
   `level ∈ {0, …, n}`. `round_half_even` = `std::nearbyint` (default rounding mode) =
   `tf.round` = `np.round`.
2. `acc = Σ_{taps, c} level · w` in exact integers. Out-of-bounds taps contribute 0
   (zero padding of the *quantized* input, larq `pad_values = 0`).
3. `v = act(acc · post_mul[o] + post_add[o])` in float32, in that order.

The source computes `conv(level / n, ±s)`; the converter puts `s / n` (rounded once from
float64) into `post_mul`. A folded BatchNorm updates `post_mul` / `post_add` exactly as for
`BConv2D` (`passes.fold_affine_into_conv`).

|acc| ≤ n · KH · KW · C, far below 2^24 for every model, so `float(acc)` is exact.

## QBDense

`x f32[..., K]`, `w bit[O, K]`, optional `post_mul f32[O]`, `post_add f32[O]`; attributes
`act_bits`, `act_quant`, `activation`. Same arithmetic as QBConv2D with a single tap.

## ChannelGate (data-dependent channel scaling)

Inputs: `x f32[N,H,W,C]`, `gate f32[N,C]` or `f32[N,1,1,C]`. Attribute `gate ∈ {sigmoid, none}`.

`y[n,h,w,c] = x[n,h,w,c] · s[n,c]`, with `s = float32(1 / (1 + exp(−(double)gate)))` for
`sigmoid` (evaluated in double, rounded once to float32) or `s = gate` for `none`.

Real-to-Binary Net's squeeze-excite scaling `conv_out · Reshape(sigmoid(Dense(z)))` maps to
`Dense(activation=none)` followed by `ChannelGate(gate=sigmoid)`. The sigmoid is evaluated inside the
gate, not by a separate `Activation(sigmoid)` node, for two reasons. The double-precision
evaluation is bit-reproducible between the C++ reference and numpy (`exp` in double, then one
rounding to float32). The float32 `Activation(sigmoid)` reference uses `expf`, and `expf` is not
guaranteed to match numpy's.

Note: `exp` is still a libm call (glibc on the host, bionic on Android). A double result
that differs by 1 ulp would change the float32 gate only if it fell exactly on a
float32 rounding boundary (probability ~2^-29 per value). Host vs emulated-ARM agreement is
checked end-to-end on Real-to-Binary Net.

## Patterns that need no new op

| Source pattern | IR |
|---|---|
| QuantConv2D(input_quantizer=None, kernel_quantizer=ste_sign) (BinaryAlexNet stem) | `Conv2D` with ±1 float weights |
| `xnor_weight_scale` kernels (XNOR-Net) | `BConv2D`, `post_mul[o] = mean|clip(W,−1,1)|` over (kh, kw, ci) |
| `clip_by_value_activation` (DoReFa-Net head) | `Activation(hardtanh, min 0, max 1)` |
| PReLU(shared_axes=[1,2]) | `Activation(prelu, slope[c])` |
| `tf.split(x, G)` → G × Conv2D → `tf.concat` (MeliusNet stem, larq_zoo `TFOpLayer`) | one `Conv2D(groups=G)` |
| MeliusNet improvement merge `x + tf.pad(w, [[0,0],[0,0],[0,0],[C−c, 0]])` | `Pad` (channel axis) + `Add` |
| identity `Lambda` (Real-to-Binary block outputs) | no node |
| legacy QuickNet-XL squeeze-excite: `tf.reshape(sigmoid gate, [-1, 1, 1, C])` and `x *= gate` (`TFOpLambda` `reshape` / `math.multiply`) | `Dense(activation=none)` + `ChannelGate(gate=sigmoid)`, as Real-to-Binary Net (the reshape call arguments are checked) |
| `DepthwiseConv2D(depth_multiplier=8, activation=relu)` (legacy QuickNet-XL stem) | `Conv2D(groups=C, relu)` with the multiplier folded into the output channels |
| `DoReFa(k_bit=1, mode="weights")` as **input** and kernel quantizer (BinaryFaceNet, insightface_larq_keras; slice S6) | `BConv2D(input_binarize=gt, pad_value 0, post_add = bias, groups kept)` on the operand `x − 2^-24·tanh(max\|x\|)`: `MaxPool2D` (full H×W) of `x` and of `ChannelAffine(x, −1)` → `Concat` → `Reshape [1,1,2C,1]` → `MaxPool2D [1,2C]` → `Activation tanh` → `ChannelAffine(−2^-24)` → `AddBroadcast(x, ·)`. In float32, `2·round(tanh(x)/(2·max\|tanh\|) + 0.5) − 1` is exactly `+1 iff x > 2^-24·max\|tanh x\|` (per tensor = per sample at batch 1; tiny positive values of near-dead channels map to −1). Kernels: the live quantizer's ±1 output (no scale) |
| `QuantDepthwiseConv2D` with the same quantizers | `BConv2D(groups = C)` (one input channel per group), same operand |
| 1×1 `QuantSeparableConv2D` with the same quantizers (depthwise then pointwise on one ±1 input, no requantization in between) | one 1×1 `BConv2D` with bits `pointwise[o,c]·depthwise[c]` (products of ±1: exact) |
| `Conv2D(activation=sigmoid)` on a 1×1 map → `Multiply` (BinaryFaceNet attention) | `Conv2D(activation=none)` + `ChannelGate(gate=sigmoid)` with an `[N,1,1,C]` gate |
| `GlobalMaxPooling2D` at a fixed input size (BRILLsson head) | `MaxPool2D(kernel = [H, W])` + `Reshape [N, C]` |

Every Lambda interpretation is checked by the adapter, which runs the Lambda on random inputs
and compares the result exactly. Every quantizer interpretation is also checked: input
quantizers on probe values, including exact zeros and DoReFa ties, and kernel quantizers on the
layer's own kernel.

## Open issues

* QBConv2D / QBDense have bit-packed kernels on x86-64 (`qbconv2d.avx2`, `qbdense.avx2`,
  tier `binary_avx2`, `act_bits ≤ 4`) and AArch64 (`qbconv2d.neon`, `qbdense.neon`, tier `neon`,
  `act_bits ≤ 8`), QBConv2D `groups == 1`: the levels are split into
  bit-planes (`level = Σ_b 2^b · bit_b`) and `Σ level·w = X − 2·D` with
  `X = Σ_b 2^b·popcount(bit_b)` and `D = Σ_b 2^b·popcount(bit_b AND neg_w)`, exact in integers,
  so both are bit-exact with the reference. The NEON kernels run every (pixel or row, plane) as one
  row of the `bconv2d.neon_opt` 4×4 micro-kernel with AND instead of XOR (larq-zoo-dorefanet:
  4.21 G → 0.22 G emulated ARM instructions per inference). ChannelGate has no optimized kernel.
* Grouped `QBConv2D` (`groups > 1`) runs only on the reference kernel. Grouped `BConv2D` is
  supported by the packed kernels since 2026-10-30 (`BConvGroupLayout`, see `operators.md`). No
  model of this slice has a grouped binary convolution, and the Keras adapter still rejects
  grouped binary QuantConv2D.
