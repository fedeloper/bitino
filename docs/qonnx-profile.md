# Bitino QONNX profile, version 1 (`bitino-qonnx/1`)

Status: **the standard** (step 0 of [qonnx-frontend.md § Plan](qonnx-frontend.md#plan), 2026-11-20; since step 6,
2026-11-21, every published file is built through it: [qonnx-standard.md](qonnx-standard.md)). This document is the standard;
`python/bitino/qonnx_profile.py` checks files against it (`bitino check-qonnx FILE`). Nothing here is upstream or
public.

Decision (2026-11-20): **QONNX is the standard format of every Bitino network.** Each of the 132 networks gets one
canonical QONNX file per shipped `.bitino` file (`model.bitino`, `model_int8_full.bitino`, `model_int8_full_log2.bitino`),
and the `.bitino` is derived from it by one importer (`bitino.qonnx_import`). The canonical file uses **standard ONNX
operators and the QONNX quantizers only**: no private domain, no model-local functions. A Bitino kernel that has no
single ONNX operator (the image features, the full-integer epilogues, the log2 residual code) is written as a subgraph
of standard operators, which the importer pattern-matches back onto the fused kernel. For every such subgraph this
document states whether it is bit-exact with the kernel, and why; where that is not established it says so
(§ [Open questions](#open-questions-for-the-user)).

**Acceptance bar (identity).** A `.bitino` rebuilt from the canonical QONNX file must have the **same data SHA-256 and
the same graph SHA-256** as the published file (`registry/recipes.json` → `expected[<file>]`): every constant byte in
the same order and alignment, and the same graph JSON (constants table with names, dtypes, shapes, offsets and layouts;
inputs; nodes with op, name, inputs, outputs and attributes; outputs). The provenance manifest is outside the graph
hash (it holds the conversion date and tool versions) and is regenerated from the file's metadata. **Decided
2026-11-20 (Q11): data + graph SHA-256 identity is sufficient**: tensor data and graph structure, including every layer,
node and constant name, byte for byte; whole-file / manifest identity (converter name, timestamps, provenance fields of
the manifest) is not required. § [Naming and order metadata](#7-naming-and-order-metadata-byte-identity) is how a file carries
what the identity bar needs beyond semantics.

Contents: [container](#1-container) · [exactness classes](#2-exactness-classes-and-evaluation-assumptions) ·
[semantic rules](#3-semantic-rules) · [operator patterns](#4-operator-patterns) ·
[int8_full](#5-int8_full-files) · [metadata](#6-model-metadata) · [naming](#7-naming-and-order-metadata-byte-identity) ·
[verification](#8-verification-contract) · [checker](#9-the-checker) · [open questions](#open-questions-for-the-user).

## 1. Container

| item | rule | why |
|---|---|---|
| ONNX IR version | ≥ 10 | `NodeProto.metadata_props` (the naming metadata) exists from IR 10 / onnx 1.16 |
| opsets | exactly one `ai.onnx` import, version 17–21; `qonnx.custom_op.general` version 1 when a QONNX quantizer is used; **no other domain** (no `ai.bitino`, `onnx.brevitas`, `finn.*`) | 17: `LayerNormalization`; ≥ 20 only for `Gelu` (else its `Erf` form). Each operator must exist at the declared version |
| functions | `model.functions` empty | a local function is a custom operator |
| data | one self-contained file: no external data, no sparse initializers | the file is the unit that is hashed and archived |
| graph inputs | the runtime inputs only (no initializer listed as an input); element type FLOAT, or INT32 / INT64 for token ids; **fully static shapes** (batch 1) | the `.bitino` input shape is fixed |
| graph outputs | exactly the `.bitino` outputs, in order (logits, not `TopK` / `ArgMax`) | outputs map one-to-one |
| order | nodes in topological order | ONNX rule, checked |
| layout | **NCHW** activations, OIHW filters, `[K, N]` MatMul / `[N, K]` Gemm(`transB=1`) weights; `bitino.layout = NCHW` | the importer converts to NHWC / OHWI itself |
| element types | FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, UINT32, UINT64, BOOL | no FLOAT16 / BFLOAT16 / FLOAT8 (no Bitino kernel) |

**Allowed operators** (the checker's `ONNX_OPS` / `QONNX_OPS`; everything else is rejected):

| group | operators |
|---|---|
| linear | `Conv`, `ConvInteger`, `MatMul`, `MatMulInteger`, `Gemm` |
| element-wise | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Relu`, `PRelu`, `LeakyRelu`, `Clip`, `Sigmoid`, `Tanh`, `Gelu`, `Erf`, `Sqrt`, `Pow`, `Exp`, `Log`, `Reciprocal`, `Max`, `Min`, `Sign`, `Round`, `Floor`, `Ceil` |
| comparison / logic | `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `Equal`, `Not`, `And`, `Or`, `Xor`, `Where`, `IsNaN` |
| integer | `Cast`, `BitShift`, `QuantizeLinear`, `DequantizeLinear` |
| pooling / reduction | `MaxPool`, `AveragePool`, `GlobalAveragePool`, `GlobalMaxPool`, `ReduceMean`, `ReduceSum`, `ReduceMax` |
| shape / data movement | `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Transpose`, `Concat`, `Slice`, `Pad`, `Gather`, `Identity`, `Shape`, `Constant` |
| normalization | `LayerNormalization`, `Softmax` |
| image features | none: the FFT feature's DFT is a `MatMul` on DOUBLE cos / sin tables (Q1, decided), `DFT` is rejected |
| QONNX (`qonnx.custom_op.general`) | `BipolarQuant`, `IntQuant` |

Explicitly rejected, with the canonical replacement (the checker prints it): `BatchNormalization` (§ 3 R6), QONNX `Quant`
(old name: write `IntQuant`), FINN `MultiThreshold` (not a QONNX operator: `Where(GreaterOrEqual(x, t), 1, −1)` for one
threshold, `out_scale · ReduceSum(Cast(GreaterOrEqual(Unsqueeze(x, −1), T)), −1) + out_bias` for several), `TopK`, QONNX
`Trunc` / `FloatQuant` (no Bitino operator), `Dropout`, `Resize`.

Attribute rules: no `auto_pad` (explicit `pads`), `ceil_mode = 0`, `MaxPool` without indices output and `storage_order 0`,
`count_include_pad` written explicitly on a padded `AveragePool`, `Pad` mode `constant` only, `Gemm` with `transA = 0`,
`alpha = beta = 1` (scales are `Mul` nodes), `Gelu approximate = none`, `LayerNormalization` / `Softmax` on the last
axis, `IntQuant rounding_mode ROUND` (= `HALF_EVEN`).

## 2. Exactness classes and evaluation assumptions

Each pattern below is classified against the Bitino kernel it maps onto (`docs/operators.md`, `docs/int8.md`,
`docs/operators_image_features.md`):

* **B (bit-exact):** evaluating the pattern with ONNX semantics, in the element types written in the file, gives exactly
  the kernel's values for every finite input in the stated domain (−0 and +0 count as equal: every binarizer of the
  profile maps both alike).
* **F (float contract):** the same real-valued function, but the kernel has its own documented precision (double
  accumulation, a fixed summation order, a double-precision libm function), so an fp32 ONNX evaluation agrees only up
  to rounding. Accepted with the Level-3 rules of `docs/numerical-correctness.md` (float64 reference, first-flip
  tracing). F does not affect byte identity: the importer derives the constants, not the evaluation.
* **O (open):** not established; listed under open questions. No custom operator is introduced instead.

B rests on four assumptions about the reference evaluation:

* **A1** IEEE-754 binary32 / binary64, round to nearest even; `Add`, `Sub`, `Mul`, `Div`, `Sqrt` correctly rounded
  per element; no contraction (FMA) inside a node or across nodes.
* **A2** no graph rewriting: onnxruntime sessions use `ORT_DISABLE_ALL` (its fusions, e.g. `MatMul + Add → Gemm`,
  change rounding); qonnx's `execute_onnx` runs one node per session, which has the same effect for fusions across
  nodes. (Measured in step 3, onnxruntime 1.31: a single-node session still applies its default optimizations to
  the node itself, e.g. the NCHWc kernel of a float `Conv`, so `execute_onnx` and an `ORT_DISABLE_ALL` session
  differ by float32 rounding on F-class nodes; B-class patterns are equal in both.)
* **A3** integers never overflow: each integer pattern states its bound (INT32 < 2^31, INT64 < 2^62 where the
  shift construction of § 5 is used).
* **A4** finite inputs (no NaN / Inf), except where a pattern handles them (`ImageGrayU8`: NaN → 0).

Measured on onnxruntime 1.31.1 (`.cache/venv-qonnx`, host, 2026-11-20). The profile works around these behaviours:

| behaviour | consequence in the profile |
|---|---|
| `Clip`, `Min`, `Max` on **INT64** compare inexactly near 2^31 (2^31 passes through a clip at 2^31 − 1; −2^31 − 1 passes a clip at −2^31). INT32 `Clip` is exact (checked at 2^24 + 1 and 2^31 − 2) | INT64 saturation is `Where(Greater(a, hi), hi, a)` / `Where(Less(a, lo), lo, a)`; the checker rejects INT64 `Clip` / `Min` / `Max` (`E-INT64-CLAMP`) |
| `ConvInteger` padding reads the zero point (a padded tap contributes 0); the ONNX text does not say | `ConvInteger` must have `pads = 0`; padding is an explicit `Pad(constant pad code)` (`E-CONVINT-PAD`) |
| `QuantizeLinear` divides (`x / s`, not `x · (1/s)`): equal to `rne(x / s)` on the 6 of 2·10^6 random values where the two differ | `QuantizeLinear` is B (when its zero point fits INT8) |
| `DFT`, length 224, Gaussian input: max abs error vs numpy fp64 **2.3e-3 in FLOAT**, 2.1e-12 in DOUBLE | the FFT feature is a DOUBLE `MatMul` on float64 cos / sin tables (Q1, decided), not `DFT`; any DFT runs in DOUBLE |
| `Round` is half to even (0.5, 1.5, 2.5, −0.5 → 0, 2, 2, −0); `Cast` FLOAT → INT32 truncates | `Round` is the rounding; a `Cast` to an integer type only ever sees integral values |
| `Sign(−0) = +0`; `Relu(−0)` and `PRelu(−0)` keep −0 | covered by "−0 = +0" in B |
| step 5: `ReduceSum` has no INT8 kernel, `MatMul` none for INT8 (both exist for INT32 / INT64); `ConvInteger` and `MatMulInteger` accept INT8 activations with an INT8 zero point (checked against an int64 reference: grouped, dilated, strided, batched 4-D) | int8 tensors are `Cast` to INT32 / INT64 before a reduction; the integer GEMMs are `ConvInteger` / `MatMulInteger` |
| step 5: a float `Conv` / `MatMul` of ±1 (or small integer) operands sums exactly while every partial sum is an integer below 2^24 (K = 4608 ±1 taps, 56 × 56 window sums of codes, checked); no Winograd / NCHWc rewrite under `ORT_DISABLE_ALL` | binary dot products of int8_full files are float `Conv` / `MatMul` on ±1 FLOAT, cast to INT64 for an integer epilogue |
| step 5: an initializer that no node reads is dropped by the session with a warning | `bitino.unused_constants` (§ 6) initializers change no output |

## 3. Semantic rules

* **R1 `sign(0)`.** The binarizer in the file states the convention and the importer keeps it, never substituting one
  for another (`docs/numerical-correctness.md` § Binarization semantics):

  | pattern | value at the threshold / at 0 | IR mode |
  |---|---|---|
  | `BipolarQuant(x, s)`, `s > 0` | `x ≥ 0 → +s` (also −0), else `−s` | `ge` |
  | `Where(GreaterOrEqual(x, t), 1, −1)` | `x ≥ t → +1` | `ge`, threshold `t` |
  | `Where(Greater(x, t), 1, −1)` | `x > t → +1` | `gt`, threshold `t` |
  | `Sign(x)` (`Sign(Sub(x, t))`) | `0 → 0` | `sign_zero` (threshold `t`) |
  | `Relu(Sign(x))` | `{0, 1}`: `x > 0 → 1`, else 0 | `gt0` (BiBERT's attention binarizer) |

  All five are **B**: a comparison or a sign of an fp32 value is exact, and `fl(x − t) ≥ 0 ⇔ x ≥ t` for finite `x`, `t`
  (an IEEE difference has the exact sign and is 0 only when `x = t`), so a threshold written as `Sub` / `Add` before
  the binarizer is exact too. A binarizer's scale must be a constant > 0 (a negative scale would flip the comparison).
* **R2 binary padding is explicit.** A binarized tensor (a binarizer output, also through `Reshape`, `Flatten`,
  `Squeeze`, `Unsqueeze`, `Transpose`, `Slice`, `Gather`, `Concat`, `Cast`, `MaxPool`) is padded only by a `Pad` node with
  mode `constant` and an explicit constant input `v ∈ {−1, 0, +1}`, and the `Conv` / `ConvInteger` reading it has
  `pads = 0`. `v = 0` is PyTorch / Larq zero padding after the binarizer (IR `pad_value 0`), `v = ±1` the bitorch / daBNN
  / Larq `pad_values` conventions. A binarized `Conv` with its own `pads` is rejected (`E-BINPAD-IMPLICIT`), although the
  phase-1 importer still reads it as `pad_value 0` from non-canonical inputs. `MaxPool` padding is ignored taps in
  ONNX and Bitino alike. **B.**
* **R3 layout.** NCHW in the file; the importer produces NHWC / OHWI. A `Flatten` / `Reshape` of an NCHW tensor has NCHW
  element order and the importer inserts the Transpose (named by `bitino.extra_names`, § 7).
* **R4 binary weights.** Canonical: `BipolarQuant(W_latent, 1)` or a constant in `{−1, +1}`, with the per-channel
  scale α written as `Mul(conv, α[1, O, 1, 1])` after the layer. Then the fp32 `Conv` sums integers (exact while
  `K < 2^24`) and `Mul` / `Add` are the kernel's `fp32(acc) · post_mul + post_add`: **B**. Also accepted, with a warning
  (`W-ALPHA-INSIDE`): `BipolarQuant(W_latent, α[o])` or constants `±α[o]`: the importer extracts α exactly, but the fp32
  `Conv` then sums multiples of α and rounds (**F**). Weights on a binarized input that are not `±α` per output channel
  are a float layer on a binarized tensor (`W-FLOAT-WEIGHTS`); they are never rounded to ±α. Ternary weights (`w_zero`):
  `IntQuant(W, s, 0, 2, signed=1, narrow=1)` or a `{−1, 0, +1}` constant (the plain constant: checker info `I-TERNARY`,
  step 6; before, the checker warned `W-FLOAT-WEIGHTS` on it).
* **R5 rounding.** Half to even only: `Round`, `IntQuant` `ROUND`, `QuantizeLinear`. Other rounding modes are rejected.
  Integer rounding shifts (half up, `docs/int8.md`) are written out with `Add` and a shift (§ 5).
* **R6 no `BatchNormalization`.** ONNX stores BN's epsilon as a float32 attribute and evaluates BN in its own order,
  while the converters fold BN in float64 with the source's epsilon. BN is folded by the exporter, in float64, into
  `Mul(x, s)` → `Add(·, b)`, `s = γ / √(σ² + ε)`, `b = β − μ·s`, with `s`, `b` as float64 carriers (R7). The fp32
  evaluation of the two nodes is the IR `ChannelAffine` (**B**); the importer reads the float64 values for every
  further fold (integer thresholds, `post_mul` / `post_add`).
* **R7 float64 carrier.** A constant whose float64 value enters an importer fold (BN scale / shift, α, merged affines,
  thresholds) is a DOUBLE initializer read through `Cast(c, to = FLOAT)`. fp32 evaluation sees `fl32(c)`, which is what
  Bitino stores when the constant is stored as is; the float64 reference and the importer see `c`. This is the only
  place where the file rounds a constant from DOUBLE to FLOAT.
* **R8 constant folding.** Only input-independent subgraphs may be folded (`Bin(W_latent)` → `±1` constants is fine);
  nothing is folded across a binarizer of a runtime tensor.
* **R9 integer bounds.** Every integer pattern names its bound: INT32 accumulations `|acc| ≤ K · 127 · 127` need
  `K ≤ 133144`; INT64 intermediates of the shift construction stay below 2^62.
* **R10 graph I/O.** As § 1. Token-id inputs declare their range (`bitino.input_domain = ids:<V>`), which makes a plain
  `Gather` exact (Q7).

## 4. Operator patterns

Patterns are written as nested ONNX calls in NCHW. `Bin(x)` is one of the binarizers of R1. `[...]?` is optional.
Every constant is an initializer; `c64` marks a float64 carrier (R7). The IR operators are those of the 132 published
`model.bitino` files (29 types; counts in [qonnx-frontend.md](qonnx-frontend.md#the-kernel-set)).

### 4.1 Binary operators

| IR op | canonical pattern | class | why |
|---|---|---|---|
| `BConv2D` | `[Pad(·, v ∈ {−1,0,1})]? (Bin(x [− t])) → Conv(·, BipolarQuant(W, 1) or ±1, pads 0, group g) → Mul(·, post_mul c64) → Add(·, post_add c64) → [Relu / Clip]?` | B | operands in {−1, 0, +1}: the Conv sum is an integer `|acc| ≤ K < 2^24`, exact in any order; then the same two fp32 operations as the kernel, and an exact activation |
| `BConv2D` with BN + binarizer folded into an integer threshold (Brevitas / FINN style) | either the float chain `Conv → Mul(α) → Mul(s) → Add(b) → Bin` (BN folded per R6) or the IR form `Conv → Mul(±1) → Add(∓T) → BipolarQuant(·, 1)` | B for the IR form; for the float chain B when the importer's exhaustive check over every `acc ∈ [−K, K]` finds the fp32 chain and `T` agree (recorded in the report), else F with the count | the threshold `T` is derived from the float64 chain, as the legacy converters do |
| `BDense` | `Bin(x) → MatMul(·, W[K, O] ±1)` or `Gemm(·, W[O, K] ±1, transB 1)` (no Gemm bias) `→ Mul(post_mul) → Add(post_add)` | B | as BConv2D |
| `BDense` ternary (`sign_zero` × `w_zero`) | `Sign(x [− t]) → MatMul(·, IntQuant(W, 1, 0, 2, signed, narrow) or {−1,0,1})` → `Mul` → `Add` | B | ternary products, integer sums |
| `BMatMul` | `MatMul(Bin_a(a), Bin_b(b))` (`Transpose(b)` for `transpose_b`) `→ Mul(·, scale)` | B | `y = float(acc) · float(scale)`; BiBERT: `Sign × Sign`, `Relu(Sign) × Sign` |
| `QBConv2D` / `QBDense` (DoReFa, k-bit activations) | `Round(Mul(Clip(x, 0, 1), n)) → Conv(·, ±1 weights) → Mul(post_mul) → Add(post_add)`, `n = 2^k − 1` | B | the kernel's level is `rne(fp32(clip(x) · n))`, one rounded product; levels are integers so the sum is exact. **Not** `IntQuant(x, 1/n)`: `x / fl(1/n) ≠ x · n` in fp32 |
| `BConv2D` with an input threshold and input-channel polarity (step 4: the BNN-PYNQ Theano CNV, conv → MaxPool → BN → sign) | `Conv(Bin(x), ±1, pads 0) → MaxPool → Mul(s c64) → Add(b c64) → BipolarQuant(·, 1) → Conv(·, ±1, pads 0)` (the first Conv without epilogue, `--affine fold`) | B | the MaxPool runs on the exact integer accumulator `n`, as the source (pool before BN); with `T` the integer threshold of `brevitas_bnn.integer_threshold` (k_max = K, parity K mod 2), channels with `s > 0` binarize `n ≥ T`, channels with `s < 0` have `+1 ⇔ n ≤ T ⇔ −(n ≥ T + 1)`: the next BConv2D gets the input threshold `T` / `T + 1` and its weights of the `s < 0` input channels negated (exact only without spatial padding of that conv: checked) — the rule of `finn_bnn` `pool="before_bn"` |
| AdaBin conv (step 1b): `BConv2D(1 output, all +1)` + `Conv2D 1×1` + `BConv2D` + `Add` + `Add(map)` | `a = Mul(Add(Bin(Sub(x, β)), β), α)` (`Bin` = `Sign` or `BipolarQuant(·, 1)`; `α > 0` and `β` constant scalars, the same `β` in `Sub` and `Add`) `→ [Pad(·, 0)]? → Conv(a, W, pads)` (W constant, **exactly two values W⁺ > W⁻ per output channel**, group 1, no dilation, no bias) `→ [Mul(s c64) → Add(t c64)]*` (per-channel affines, the BatchNorm) | F | `a ∈ {α(β−1), α(β+1)}` is the source's own pair of fp32 operations on the sign (AdaBin `BinaryActivation`: `α·(sign((x−β)/α) + β)`; the canonical file binarizes `x − β`, the same sign for `α > 0` except where the fp32 quotient `(x−β)/α` underflows to 0, `|x − β| < α·2⁻¹⁵⁰`: the AB-BNN `b/β` precedent of step 1). The importer rebuilds the legacy decomposition of `torch_adabin` in float64 from the file's values, in the adapter's operation order: with `p, q = (A⁺ ± A⁻)/2`, `c, d = (W⁺ ± W⁻)/2` and the affine `g·y + h`, `conv = q·d·Σ s_w s + q·c·Σ_inb s + p·(d·Σ_inb s_w + c·K_inb)`: main `BConv2D` (threshold β, `post_mul = g·q·d`), the per-pixel `Σ_inb s` as a 1-output `BConv2D` with all-+1 weights broadcast by a 1×1 `Conv2D` (`g·q·c`), and the constant border map `g·p·(d·Σ_inb s_w + c·K_inb) + h` (zero padding: out-of-bounds taps contribute 0, which is not in the activation set). The fp32 ONNX evaluation sums non-integer products, the kernels integer sums: F, Level 3. Checked by `E-ADABIN` / `I-ADABIN` |

### 4.2 Float compute

| IR op | pattern | class | why |
|---|---|---|---|
| `Conv2D` | `[Pad(·, v)]? → Conv(x, W, [b]?)`, depthwise `group = C` | F | reference accumulates in double, optimized kernels fp32 GEMM with a bound; ONNX order is implementation-defined |
| `Dense` | `Gemm(x, W[O,K], b, transB 1)` or `MatMul(x, W[K,O]) → Add(b)` | F | as Conv2D |
| `MatMul` | `MatMul(a, b)` (`Transpose(b)` for `transpose_b`) `→ [Mul(scale)]?` | F | the kernel rounds `acc · scale` once in double |

### 4.3 Per-channel and element-wise

| IR op | pattern | class | why |
|---|---|---|---|
| `ChannelAffine` | `Mul(x, s[1,C,1,1] c64) → Add(·, b c64)` (either optional) | B | the same two fp32 operations in that order. `Div` by `c` is not `Mul` by `1/c`: canonical files write the constant Bitino stores |
| `Activation relu`, `relu6` / `hardtanh` | `Relu`, `Clip(x, lo, hi)` | B | exact |
| `Activation prelu` / RPReLU | `PRelu(x, slope[C,1,1])` / `Add → PRelu → Add` | B | one product for `x < 0`, as the kernel |
| `Activation leaky_relu` | `LeakyRelu(alpha)` | B | as PReLU; alpha's float64 value via `bitino.attrs` (§ 7) |
| `Activation sign` / `sign_zero` | `BipolarQuant(x, 1)` / `Sign(x)` (not feeding a binary op) | B | R1 |
| `Activation sigmoid`, `tanh` | `Sigmoid`, `Tanh` | F | the kernels use double libm (`tanh`) or their own approximations within a bound |
| `Activation gelu` | `Gelu` (opset ≥ 20) or `Mul(Mul(x, 0.5), Add(Erf(Div(x, √2)), 1))` | F | Bitino evaluates `x·0.5·(1 + erf(x/√2))` in double and rounds once (no fp32 erf underflow) |
| `Add` / `AddBroadcast` / `Mul` | `Add` (equal shapes / broadcasting), `Mul` | B | one fp32 operation |
| `Add` of a constant map (step 1b) | `Add(x, M c64)`, `M` a constant `[1, C, H, W]` of the tensor's shape that varies over H / W (a per-channel `M` is a `ChannelAffine`) | B | one fp32 addition per element; the IR stores `fl32(M)` in NHWC (BNAS's data-independent zero-channel correction of its `ge` variant) |
| `ChannelGate` | `Mul(x, Cast(Sigmoid(Cast(g, DOUBLE)), FLOAT))`, `g [N,C,1,1]` (plain fp32 `Sigmoid` accepted) | F | Bitino rounds `1/(1 + exp(−g))` from double with libm `exp`; ONNX's double `Sigmoid` is implementation-defined (equal after the fp32 rounding except at double-rounding ties, not guaranteed) |
| `FakeQuant`, zero point 0 (8 published nodes: the FINN / Brevitas CNV and GSB 8-bit input quantizers) | `[Mul(x, pre_scale)]? → IntQuant(·, s, 0, bits, signed, narrow, ROUND)` | B | qonnx computes `x / s` (IEEE division), adds 0, clips, rounds half to even, multiplies by `s`; Bitino rounds then clips, identical with integer bounds |
| `FakeQuant`, zero point ≠ 0 (40 published nodes: BHViT's LSQ attention quantizer, `pre_scale 3`, levels 0..3) | `Mul(Clip(Round(Add(Div(Mul(x, pre_scale), s), z)), qmin, qmax), s)` | B | the kernel's steps, one fp32 operation each, in its order. Not `IntQuant`: it returns `(q − z)·s`, while Bitino's FakeQuant does not subtract the zero point (`docs/operators_bhvit.md`) |

### 4.4 Pooling, shape, normalization

| IR op | pattern | class | why |
|---|---|---|---|
| `MaxPool2D` | `MaxPool` (explicit pads) | B | max is exact; out-of-bounds taps ignored in both |
| `AvgPool2D` | `AveragePool` with explicit `count_include_pad` | F | Bitino sums the taps in its order, then divides (or multiplies by a power-of-two reciprocal, an exact equivalent) |
| `GlobalAvgPool`, `ReduceMean` | `GlobalAveragePool`, `ReduceMean(axes)` | F | summation order |
| `Concat`, `Reshape`, `Transpose`, `Pad` | `Concat`; `Reshape` / `Flatten` / `Squeeze` / `Unsqueeze`; `Transpose`; `Pad` constant | B | data movement |
| `Gather` | `Gather(table, ids)` with ids in `[0, V)` (`bitino.input_domain ids:<V>`) | B on the declared domain | Bitino gives a zero row for out-of-range ids, ONNX wraps negative ids and fails above V (Q7) |
| `LayerNorm` | `LayerNormalization(axis −1, epsilon)` (eps's float64 value via `bitino.attrs`) or its `ReduceMean / Sub / Mul / Sqrt / Div` form | F | Bitino computes mean and variance in double with sequential sums |
| `Softmax` | `Softmax(axis −1)` | F | max-subtracted, double accumulation |

**Image-to-image data movement (step 1b, § 4.6 below).** Image outputs are NCHW `[1, C, H, W]` graph outputs (the IR
output is NHWC); `bitino.labels = none`.

Transformers need nothing beyond these: BiBERT's attention is `BDense(sign_zero)` → `Reshape` / `Transpose` →
`BMatMul(Sign, Sign, scale)` → `Add(mask)` with `mask = Mul/Add(attention_mask)` (`ChannelAffine(m, 10000, −10000)`, exact
for `m ∈ {0, 1}`) → `BMatMul(Relu(Sign(·)), Sign(V))`: all B; embeddings are `Gather` (B on `ids:<V>`), LayerNorm and
GELU are F. Bi-ViT and BHViT use the same set plus `FakeQuant` (BHViT's 2-bit probabilities: the explicit form, B).

### 4.5 Image features (the "Faster Than Lies" detectors, 18 files)

Written as standard-op subgraphs on the NCHW tensor (channel axis 1); the importer matches each subgraph onto the
fused kernel of `docs/operators_image_features.md`.

**`ImageGrayU8`** (coefficients `k`, `shift`, `scale`): **B**.

```
v = Mul(x, scale)                                   # fp32 product, as f32(x_c · scale)
v = Where(IsNaN(v), 0, v)                           # NaN → 0
u = Cast(Floor(Clip(v, 0, 255)), INT32)             # trunc = floor on [0, 255]; integral, so the Cast is exact
s = ReduceSum(Mul(u, k[1,3,1,1] INT32), axes [1], keepdims 1)
y = Cast(BitShift(Cast(Add(s, 2^(shift−1)), UINT32), shift, RIGHT), FLOAT)
```

Exact: every value is an integer below 2^31 and non-negative, so the UINT32 shift is the kernel's `>>`; Clip handles ±Inf.

**`LocalBinaryPattern`** (`rp`, `cp`, `method`, `div`; all 18 detectors use `P = 3`, `rp = [−0, −5.19615, 5.19615]`,
`cp = [6, −3, −3]`, `uniform`, `div 255`): **B** under a condition. In DOUBLE (`X = Cast(x, DOUBLE)`):
for each sample point `i`, four shifted copies of `X` (`Pad` with zeros, then `Slice` at `floor` / `ceil` of the
offsets), the bilinear weights `dr = Sub(rr, Floor(rr))` with `rr = Add(r_index[H,1], rp_i)` (and the same along the
columns), `top = Add(Mul(Sub(1, dc), v00), Mul(dc, v01))`, `bottom` likewise, `t = Add(Mul(Sub(1, dr), top), Mul(dr,
bottom))`, `bit_i = GreaterOrEqual(Sub(t, X), 0)`; `uniform`: `changes = Σ_{i<P−1} Cast(Xor(bit_i, bit_{i+1}))`,
`lbp = Where(LessOrEqual(changes, 2), Σ bit_i, P + 1)`; `default`: `Σ bit_i · 2^i`; `y = Cast(Div(Cast(lbp, DOUBLE),
div), FLOAT)`. Exact because these are the double operations of scikit-image's `bilinear_interpolation` in its order
(A1), computing `rr` per row from a constant row-index tensor reproduces its per-pixel rounding of `r + rp`, and a
division rounded to double and then to float equals the direct float division (53 ≥ 2·24 + 2). **Condition:** the shift
of every sample point is uniform, `floor(r + rp_i) = r + floor(rp_i)` for every row (and the same for columns and
`ceil`). The optimized kernel checks the same condition in `supports()`; the exporter checks it for the stored
offsets and image size. The detectors' offsets satisfy it at every practical image size: the integral ones shift
exactly, and `±5.19615` is 0.196 away from an integer, far beyond the rounding of `r + rp` for `r < 2^20`.

**`FftLogMagnitude`** (`eps`, `mul`, `div`, `shift`): **F** (decided 2026-11-20, Q1). The 2-D DFT is a
double-precision DFT written as matrix products with cos / sin tables computed in float64 and stored as DOUBLE
initializers: with `X = Cast(x, DOUBLE)` per channel, `Re = Sub(MatMul(MatMul(C_H, X), C_W), MatMul(MatMul(S_H, X), S_W))`,
`Im = Neg(Add(MatMul(MatMul(S_H, X), C_W), MatMul(MatMul(C_H, X), S_W)))` (`C_N[k, n] = cos(2πkn/N)`,
`S_N[k, n] = sin(2πkn/N)`), `|X| = Sqrt(Add(Mul(Re, Re), Mul(Im, Im)))`, `fftshift` as `Gather` with the rotated
indices, `y = Cast(Div(Mul(Log(Add(|X|, eps)), mul), div), FLOAT)`. It is **not** bit-exact with Bitino's kernel (a
fixed Stockham FFT with polynomial twiddles and a polynomial `log`), and need not be: the feature is under the float
contract (class F, `docs/numerical-correctness.md` Level 3, with the per-model feature check of today's conversions:
float32 channel values equal to the source on natural images, differences only where `|X|` is at rounding level). No
unrolled exact FFT. ONNX `DFT` is not used (rejected by the checker): onnxruntime's FLOAT DFT is off by 2.3e-3 at
N = 224 (§ 2) and the operator does not fix its algorithm.

### 4.6 Image-to-image data movement (step 1b)

`DepthToSpace` and `Resize` are not profile operators. The super-resolution networks (BBCU, BSRC-2023) and the
BiMatting backbone spell their data movement out with `Reshape` / `Transpose` / `Gather` / `AveragePool` on NCHW
tensors; the importer maps each subgraph onto the legacy NHWC subgraph of `bitino.adapters.restoration_common` /
`torch_abbnn` (the IR node named by `bitino.name` is the last one of the group, the others are `bitino.extra_names` in
order). All are **B** (data movement; the half sum one rounded addition halved exactly):

| subgraph (NCHW) | IR (NHWC) |
|---|---|
| PixelShuffle(r): `Reshape [1, C·r², H, W] → [1, C, r, r, H, W]`, `Transpose [0, 1, 4, 2, 5, 3]`, `Reshape [1, C, H·r, W·r]` | `Reshape [1, H, W, C, r, r]` (extra 0), `Transpose [0, 1, 4, 2, 5, 3]` (extra 1), `Reshape [1, H·r, W·r, C]` |
| row gather (nearest ×r, replicate padding): `Reshape [1, C, H·W]`, `Gather(axis 2, constant integer indices [OH, OW] in [0, H·W))` | `Reshape [H·W, C]` (extra 0), `Gather(i32 indices <group>/indices)` (extra 1), `Reshape [1, OH, OW, C]` |
| bilinear ×r (`align_corners = False`) | replicate padding (row gather) `→ Conv(3×3 phase kernel, pads 0)` (the exact dyadic phase weights of `restoration_common.bilinear_kernel`) `→` PixelShuffle(r); F (the weighted sum in another order than torch's) |
| channel split (BiMatting `(out1[:, :P] + out1[:, P:]) / 2`): `Reshape [1, k, C/k, H·W]`, `Transpose [0, 2, 3, 1]`, then `AveragePool(1 × k, count_include_pad 1)` and `Reshape [1, C/k, H, W]` | `Reshape [1, H·W, k, C/k]`, `AvgPool2D [1, k]`, `Reshape [1, H, W, C/k]` |
| spatial-only `Reshape [1, C, a, b] → [1, C, a', b']` (`a·b = a'·b'`) | `Reshape [1, a', b', C]` (the NHWC order is unchanged) |

## 5. int8_full files

An int8_full file is its own canonical QONNX file (`bitino.variant = int8_full` or `int8_full_<suffix>`, with
`bitino.calibration`), group-mapped (`bitino.import = {"mapping": "groups"}`, § 7): one ONNX group per `I8*` IR node of
the published int8_full graph (step 5, [qonnx-step5.md](qonnx-step5.md); writer and importer
`python/bitino/qonnx_int8.py`). Activations are real INT8 tensors (not fake-quantized floats), so integer GEMMs are
`ConvInteger` / `MatMulInteger`; bit tensors (the `:bits` outputs of the fused binary blocks) are ±1 FLOAT tensors;
f32 tensors are FLOAT. 4-D image tensors are NCHW as in § 3 R3 (spatial `I8*` operators and the per-tensor element-wise
ones between them), every other tensor keeps the IR's shape, with the layout `Transpose`s of step 3. All patterns of
this section are **B**: onnxruntime (`ORT_DISABLE_ALL`, the declared element types) reproduces `int8_ops` (and the C++
`int8_full` kernels) bit for bit; verified on all 101 published int8_full files ([qonnx-step5.md](qonnx-step5.md)).

**Primitives** (`x = s · (q − z)` per tensor; per-channel constants are `[C, 1, 1]` tensors against an NCHW view, `[C]`
otherwise; an attribute value that is not an fp32 number is a float64 carrier, R7):

| name | standard-op form | bound / note |
|---|---|---|
| `QUANT(x; s, z)` (I8Quantize) | `Cast(Clip(Add(Round(Div(x, s)), z), −128, 127), INT8)` (FLOAT ops, `z` an integral float) | = `clamp(rne(fp32(x / s)) + z)`. `QuantizeLinear` is equivalent when `z` fits INT8 (measured, § 2) but is not used: one form for every grid (Q4) |
| `DEQ(q; s, z)` (I8Dequantize, the binarizers' `deq`) | `Mul(Sub(Cast(q, FLOAT), z), s)` | = `fp32(fp32(q − z) · s)`, exact `q − z` for `|z| < 2^24` (binarize-aware grids have `|z| < 2^20`) |
| `SAT(a; lo, hi)` on INT64 | `Where(Less(Where(Greater(a, hi), hi, a), lo), lo, ·)` | not `Clip` / `Min` / `Max` (§ 2); per-channel bounds allowed |
| `ASR(a, n)` = `floor(a / 2^n)` | `Sub(Cast(BitShift(Cast(Add(a, 2^62), UINT64), n, RIGHT), INT64), 2^(62−n))` | `|a| < 2^62`, `0 ≤ n ≤ 62`: `a + 2^62 ∈ [0, 2^63)` so the UINT64 cast is exact, and `floor((a + 2^62)/2^n) = floor(a/2^n) + 2^(62−n)`. ONNX `BitShift` is defined for unsigned types only and the ONNX text does not fix the rounding of integer `Div`, so neither a signed shift nor integer `Div` is used (Q6) |
| `MULHI(a, M)` (SQRDMULH) | `ASR(Add(Mul(a, M), 2^30), 31)` on INT64 | `|a|, M < 2^31` ⇒ `|a·M| < 2^62` |
| `RSHR(a, n)` (rounding shift, half up) | `ASR(Add(a, R), n)`, `R[c] = n[c] > 0 ? 2^(n[c]−1) : 0` | |
| `REQUANT(x; M, shift)` | `RSHR(MULHI(SAT(Mul(x, 2^max(shift,0)), −2^31, 2^31−1), M), max(−shift, 0))` on INT64 | **verified**: equals `int8_ops.requant` on random int32, a dense range around 0 and the int32 extremes, scalar and per-channel `(M, shift)`, shifts −31..3 (`test_int8_requant_shift_construction_exact`) |
| `RNE8(v; qmin, qmax)` | `Cast(Clip(Round(v), qmin, qmax), INT8)` (FLOAT) | = `clamp(rne(v))`: with integral bounds, clamping before or after the rounding is the same |
| `BIN(q; mode, t)` (binarization of int8 codes) | the R1 binarizers on the dequantized value: `BipolarQuant(Sub(DEQ(q), t), 1)` (`ge`), `Where(Greater(Sub(DEQ(q), t), 0), 1, −1)` (`gt`), `Sign(Sub(DEQ(q), t))` (`sign_zero`), `Relu(Sign(Sub(DEQ(q), t)))` (`gt0`); an l8 operand (`x_code l8`) reads `L8DEC(q)` instead of `DEQ`; `Sub` only with a threshold | `deq` is non-decreasing in `q`, so `deq(q) ≥ t ⇔ q ≥ T_ge(t)` and `deq(q) > t ⇔ q ≥ T_gt(t)` by the definition of the kernels' integer thresholds (`int8_ops.qthr_ge / qthr_gt`, `docs/int8.md`); the comparison of an IEEE difference is exact (R1). The file carries `(x_scale, x_zero, threshold)`, which the IR stores, not the derived integer thresholds `T` (several `(s, z, t)` give one `T`, so a `T` form could not be imported to the published bytes) |
| `L8DEC(c; E)` (log2 code → fp32) | `Gather(D_E[256] FLOAT, Add(Cast(c, INT64), 128))`, `D_E = int8_ops.l8_decode(−128..127, E)` | the decode is exact by definition, so the 256-entry table is; the importer requires the table to be `l8_decode(·, E)` bit for bit (`E = log2 D_E[255]`) |
| `L8ENC(w)` (fp32 → log2 code) | step 6, the 7-step branchless bisection: `a = Abs(w)`; `c₆ = Where(GreaterOrEqual(a, Gather(β[127], 63)), 64, 0)`; for j = 5..0: `c_j = Add(c_{j+1}, Where(GreaterOrEqual(a, Gather(β, Add(c_{j+1}, 2^j − 1))), 2^j, 0))` (INT64); `y = Cast(Mul(Cast(Sign(w), INT64), c₀), INT8)` (38 nodes, every intermediate of the shape of `w`) | `β_k` = the smallest fp32 with code ≥ k (`qonnx_int8.l8_boundaries`, bisection over the fp32 bit patterns; `β_1 = 1.74046e−5`, `β_127 = 0.957031`), strictly increasing. The code is non-decreasing in `|w|`, so it is the count `N = #{k : β_k ≤ |w|} ≤ 127` of boundaries passed, and the bisection computes `N` bit by bit: before step j, `c = N` with its bits below j+1 cleared, and `β[c + 2^j − 1] ≤ a ⇔ N ≥ c + 2^j` (β sorted), so step j sets bit j exactly; the index stays ≤ 126. **Proved exhaustively** (`tools/qonnx_l8enc_proof.py`, the development log (not published)): equal to `int8_ops.l8_encode` on every non-NaN fp32 input, all 2³² − 2²⁴ + 2 bit patterns (±0, subnormals, ±inf), evaluated as written in onnxruntime (`ORT_DISABLE_ALL`) and as its numpy mirror (`qonnx_int8.l8enc_bisect`); the 22 files with l8 outputs re-verified bit-exact against the C++ `int8_full` stages. Step 5 wrote the 127-compare count form `Cast(Mul(Cast(Sign(w), INT32), ReduceSum(Cast(GreaterOrEqual(Unsqueeze(Abs(w), −1), β), INT32), [−1])), INT8)` (equal on every input, same proof), whose `[…, 127]` comparison tensor was ~0.5 GB per sample at a 112 × 112 × 64 stem (Q5, decided) |
| `LUT(q; L[T, 256], axis)` (I8Lut) | per tensor `Gather(L[0], Add(Cast(q, INT64), 128))`; per channel `Gather(Reshape(L, [T·256]), Add(Add(Cast(q, INT64), 128), OFF))`, `OFF = 256·c` along the axis | indexing; INT8 or FLOAT tables |

**Operators** (`docs/operators.md` § Full-integer operators; every step one node in the documented order; INT64 for
every integer epilogue):

| IR op | pattern |
|---|---|
| `I8Conv2D` | `[Pad(q, pads, pad_q INT8)]? → ConvInteger(·, W INT8 OIHW, x_zero INT8, pads 0) → Cast INT64 → [Add(bias)] → SAT(int32) → REQUANT(M[o], shift[o]) → Add(y_zero) → SAT(qmin, qmax) → Cast INT8`; bit output `Where(GreaterOrEqual(acc, bit_thr[o]), 1, −1)` on the INT64 accumulator before the bias; f32 output `[Add](Mul(Cast(acc, FLOAT), scale[o]), bias[o])`; l8 output `L8ENC` of that. `K ≤ 133144` |
| `I8Dense` / `I8MatMul` | `MatMulInteger(q, Wᵀ, x_zero)` / `MatMulInteger(a, [Transpose(last two)](b), a_zero, b_zero)` (batched), then as I8Conv2D (I8MatMul: `SAT(int32)` first; f32 output `Mul(Cast(acc, FLOAT), scale)`); all published zero points fit INT8 |
| `I8Add` | `x_a = REQUANT(Mul(Sub(Cast(a, INT64), z_a), 2^ls), a_mult, a_shift)`, `x_b` likewise, `y = SAT(Add(REQUANT(Add(x_a, x_b), out_mult, out_shift), y_zero), qmin, qmax)` (TFLite); an int8 constant operand is an INT8 initializer |
| `I8AffineSum` / `I8Mul` | `Add(Σ_i Mul(Sub(Cast(q_i, INT64), z_i), K[i]), B)` / `Mul(Sub(a, z_a), Sub(b, z_b))` in INT64, then `REQUANT`, `+ y_zero`, `SAT` |
| `I8Lut` | `LUT` |
| `I8LayerNorm` | `t = Sub(Cast(q, INT64), z)`, `S1 = ReduceSum(t)`, `S2 = ReduceSum(Mul(t, t))`, `VN = Sub(Mul(S2, D), Mul(S1, S1))` (exact); `r = Cast(Div(xs, Sqrt(Add(Mul(Mul(xs, xs), Cast(VN, DOUBLE)), Mul(Mul(ε, D), D)))), FLOAT)` in DOUBLE (`Cast` of `VN < 2^53` exact; `xs`, `ε` DOUBLE constants); per element `RNE8(Add(Mul(Mul(Cast(Sub(Mul(t, D), S1), FLOAT), r), G), B))` |
| `I8Softmax` | `q = Cast(q, INT64)`, `e = Gather(E INT32, Sub(ReduceMax(q), q))`, `S = ReduceSum(Cast(e, INT64))`, `r = Cast(Div(inv_scale DOUBLE, Cast(S, DOUBLE)), FLOAT)`, `v = Mul(Cast(e, FLOAT), r)`, `y = Cast(Clip(Add(Round(v), y_zero), qmin, qmax), INT8)` (f32 output: `v` with numerator 1) |
| `I8Concat`, `I8Transpose`, `Reshape`, `I8Pad`, `I8MaxPool2D` | `Concat` (inputs with other parameters: `SAT(Add(REQUANT(Sub(Cast(q, INT64), zeros[i]), mults[i], shifts[i]), y_zero), −128, 127)` first), `Transpose`, `Reshape` (INT8, ±1 FLOAT or FLOAT), `Pad(pad_q)`, `MaxPool` on INT8 or on a ±1 FLOAT bit tensor |
| `I8AvgPool2D` | `S = ConvInteger([Pad(q, x_zero)], ones[C, 1, kh, kw], x_zero, group C)` (`Σ(q − z)` per window), `RDIV(S, count)` with `count` a constant scalar or map, `+ x_zero`, `SAT(−128, 127)`; `RDIV(a, c)` (half away from zero) `= Sign(a) · Floor(Div(Add(Mul(Abs(A), 2), c), 2c))` on `A = Cast(a, DOUBLE)`, exact while `2|a| + c, 2c < 2^26` (the distance of a non-integer quotient to the next integer, ≥ 1/2c, exceeds the double rounding error) |
| `I8GlobalAvgPool` / `I8ReduceMean` | `ReduceSum(Sub(Cast(q, INT64), x_zero))` over H, W (or the axis), `REQUANT`, `+ y_zero`, `SAT(−128, 127)` |
| `I8Gather` | the guarded row gather, exact for every id (no domain assumption, Q7): `ok = And(GreaterOrEqual(ids, 0), Less(ids, V))`, `rows = Gather(T, Where(ok, ids, 0))`, `y = Where(Unsqueeze(ok, −1), rows, fill_q)`; bit table: `rows = Where(Greater(Gather(S ±1 INT8), 0), Slice(Gather(qpm), 0:1), Slice(Gather(qpm), 1:2))`, masked rows `Where(Less(Gather(Z ±1 INT8), 0), fill_q, rows)`. The guard is what carries `fill_q` (identity) |
| `I8BDense` / `I8BConv2D` | `BIN(q; T)` (or a bit tensor as is) `→ [Pad(v)] → MatMul / Conv` on ±1 FLOAT (exact: `|acc| ≤ K < 2^24`) `→ [Mul(post_mul)] → [Add(post_add)] → RNE8` (or `Identity` for the FLOAT output) |
| `I8BMatMul` | `A, B = BIN(a), BIN(b)` (FLOAT `{−1, 0, 1}` / `{0, 1}`), `acc = Add(Add(Add(Mul(MatMul(A, [Transpose]B), c_ab), Mul(ReduceSum(A, −1), c_rs)), Mul(ReduceSum(B, −2), c_cs)), c0)` (exact integers in fp32; the final `+ c0` turns a `−0` into `+0`, as the kernel's integer accumulator), `→ Mul(mul) → [Add(add)] → RNE8` |
| `I8BConv2DRes` | binary conv as I8BConv2D giving `v = Add(Mul(acc, post_mul), post_add)`; pre stage `PRelu(v, pre_slope)`, `Mul(pre_mul)`, `Add(pre_add)`, `Mul(DEQ(gate))`, `Add(gate_add)` (each only with its input); `u = Add(v, R)`, `R = L8DEC(res; r_exp)` (`r_code l8`) or `DEQ(res)` (linear); `w = [Add](PRelu(u, slope), out_bias)`; output `L8ENC(w)` (l8) or `Cast(Clip(Add(Round(w), y_zero), qmin, qmax), INT8)`; bit output `Where(GreaterOrEqual(s, bit_thr), 1, −1)` (`Greater` for `bit_binarize gt`) with `s = u` or `w` (`bit_src`) |
| `I8BConv2DResI` | integer epilogue in INT64: `pre = Mul(Cast(dot, INT64), m1)` (or `SAT(Add(pre, pre_b), pre_lo, pre_hi)`), `acc = Add(pre, b)` `[+ Mul(Cast(res) (Concat × r_tile), m2)]`, `w = Where(Less(acc, 0), Mul(MULHI(acc, slope), 2^slope_shift), acc)`, `[+ out_bias]`, `y = SAT(Add(RSHR(w, shift), y_zero))`, bits `Where(GreaterOrEqual(s, bit_thr), 1, −1)`; exact within `int8_ops.bconv_res_i_bound < 2^31` (checked by the writer). The PReLU pre stage (input 15) has no pattern yet: no published file uses it |
| `I8GatedAdd` / `I8GapAffine` | `acc = Add(Add(Mul(Sub(x, z_x), kx), Mul(Mul(Sub(p, z_p), Sub(g, z_g)), kp)), b)` / `acc = B + Σ_i Mul(ReduceSum_{H,W}(Sub(x_i, z_i)), K[i])` in INT64, `RSHR(acc, shift)`, `+ y_zero`, `SAT`; bits `Where(GreaterOrEqual(acc, bit_thr[j]), 1, −1)`. The gate table input (I8GatedAdd input 7) has no pattern yet: no published file uses it |
| `QBConv2D` / `QBDense` in int8_full files | the float patterns of § 4.1 (`Round(Mul(Clip(x, 0, 1), n)) → Conv / MatMul(±1) → Mul → Add → Identity`), between `I8Dequantize` and `I8Quantize` |

The int8_full plan-time rewrites (`collapse_unary`, twin merging, bit outputs, ...) happen in the runtime session and
are not part of the file.

## 6. Model metadata

`ModelProto.metadata_props`, all values strings. Required unless marked; `bitino.*` keys outside this table are warned
about (typos).

| key | format | content |
|---|---|---|
| `bitino.profile` | `bitino-qonnx/1` | profile version |
| `bitino.model_id` | `[a-z0-9][a-z0-9._-]*` | registry id (`registry/models.json`) |
| `bitino.variant` | `float` \| `int8_full` \| `int8_full_<suffix>` | which `.bitino` file of the model this is |
| `bitino.layout` | `NCHW` | layout of the file's tensors (v1: NCHW only) |
| `bitino.input_domain` | `float32` \| `unit8` (`k/255`) \| `uint8` (`k`) \| `ids:<V>`, or a JSON object `{graph input: domain}` | values the inputs take; rules that are exact only on a discrete domain are checked exhaustively on it (as in phase 1) |
| `bitino.preprocessing` | JSON `{"key": ..., "definition": ...}` | preprocessing key of `registry/eval_sets.json` and its exact text (resize, crop, mean / std, channel order) |
| `bitino.labels` | JSON `{"num_classes": N, "set": "<label set>"}` or `none` | output classes (`none`: restoration / regression outputs) |
| `bitino.source.framework`, `bitino.source.framework_version` | text | e.g. `pytorch`, `2.4.1` |
| `bitino.source.checkpoint_sha256` | 64 lowercase hex, or JSON `{file: sha256}` | the checkpoint(s) the file was exported from |
| `bitino.weights_license` | SPDX id or text | licence of the *weights* (`unknown` warns: not redistributable) |
| `bitino.exporter` | text | exporting tool and versions, e.g. `tools/export_brevitas_qonnx.py@<commit>; brevitas 0.11.0; torch 2.4.1` |
| `bitino.calibration` | JSON `{"set", "split", "n", "tensor_sha256"}` | **required for int8_full variants**; `split` other than `train` warns (calibration must be disjoint from every eval set) |
| `bitino.source.code` (optional) | `<repo url>@<commit>` | model code |
| `bitino.const_order` (optional) | JSON list | § 7 |
| `bitino.import` (optional) | JSON `{"mapping": "groups"}` or `{"mapping": "patterns", <options>}` | § 7: how the file is imported, recorded in the file instead of an import argument. `groups` (step 3): one ONNX group per IR node of the deployed graph, read back operator by operator, no graph pass after the import (docs/qonnx-step3.md). `patterns` (step 4): the pattern importer of steps 1–2 / 4 with its layout options `naming` (index / onnx / exporter), `affine` (fold / keep), `constant_order` (walk / first-use), `cleanup` (none / qonnx), `bit_outputs` (set / none / threshold: § 7), `flatten_dense` (transpose / permute); an import argument that contradicts the file is an error (docs/qonnx-step4.md). An int8_full variant with `groups` is read by the step-5 readers (docs/qonnx-step5.md). **Step 6: required in effect** — every canonical file carries it (steps 1 / 1b / 2 write every option, the values of their former import arguments), so the file alone determines its `.bitino`: with `patterns` an option the file does not list has the importer default and an explicit argument must equal it, with `groups` no import argument applies; the input domain is the file's `bitino.input_domain` (an explicit `--input-domain` must agree). `--no-fold` is not a layout option but the derivation of a pattern-mapped file's `*_unfused.bitino` (docs/qonnx-step6.md) |
| `bitino.unused_constants` (optional) | JSON list of initializer names | step 5: IR constants of the published file that no IR node reads (three int8_full files keep the quantized class / distillation tokens `cls_token_5:q8`, `dist_token_6:q8`); each is an initializer of that name that no node uses, imported as an IR constant with its data and element type (INT8 → i8, INT32 → i32, FLOAT → f32); they take part in the data section (identity) and in no computation |
| `bitino.ir_unique_counter` (optional) | non-negative integer | § 7: start value of the IR graph's unique-name counter (legacy converters that drew names from `Graph.unique()` before the passes, whose folded constants continue the counter: `conv1/folded_18_w`) |
| `bitino.task`, `bitino.notes` (optional) | text | |

`producer_name` / `producer_version` should be set too (warning otherwise). The importer writes the `.bitino` manifest
(`model_id`, `source.*`, `input.preprocessing`, licence) from these keys.

## 7. Naming and order metadata (byte identity)

Semantics come from the graph; names, order and the JSON representation of attributes come from metadata. **Metadata
never changes a value**: the importer rejects a file whose metadata would.

* **IR node names: `bitino.name`** (node `metadata_props`, **required on every node on the activation path**, i.e.
  every node whose output depends on a graph input's values; optional on constant-only nodes). It is the name of the IR
  node that this ONNX node becomes or is folded into. ONNX nodes with the same `bitino.name` form a *group*: a `Conv`,
  its weight `BipolarQuant`, the `Mul(α)`, the folded BN `Mul` / `Add`, the activation and the next binarizer's
  threshold all carry, e.g., `conv_features.1`. A group emits one IR node with that name.
* **Further IR nodes of a group: `bitino.extra_names`**, a JSON list in emission order, for the IR nodes the importer
  adds (the Transpose before an NCHW flatten, a `ChannelAffine` before a `FakeQuant`, ...). Output tensor names follow
  from node names (`<name>:out`, `<name>:out<i>`); IR input names are the ONNX graph input names.
* **Node order: `bitino.order`**, a non-negative integer, the same on every node of a group: the IR position of the
  group (its extra nodes follow it in list order). Without it (on every group) the order is that of each group's first
  node in the ONNX graph. All groups or none.
* **Constant names: `bitino.consts`** on any node of a group, JSON `{"<input index>": "<name>"}` for the group's IR
  node and `{"<extra name>#<input index>": "<name>"}` for an extra node; input indices are the IR operator's
  (`docs/operators.md`, e.g. BConv2D 1 = `w`, 2 = `post_mul`, 3 = `post_add`, 4 = `threshold`). Constants without an
  entry get the importer's default name; constants created by the graph passes are named by the passes. Names are
  unique.
* **Constant order: first use.** The data section lists constants in the order of their first reference, walking the
  final IR nodes in order and each node's inputs by index. `bitino.const_order` (model metadata, a JSON list of every
  constant name in data-section order) overrides this; the importer fails if the list and the constants it built
  differ as sets. (The phase-1 importer writes creation order; first-use order is part of step 1.) The list is the
  order of the final data section, after the importer's passes (their folded constants included); the importer
  applies it last (step 2: BinaryFaceNet, whose converter created the binary conv's constants before its input
  operand's). Step 1b's AdaBin files use it too (the adapter created `threshold` before `ones_bits` and
  the Dense kernel before its bias: neither walk nor first-use order).
* **Declared float layer: `bitino.float_layer`** (step 1b), value `"1"`, on a `Conv` / `Gemm` / `MatMul` whose
  activation input is binarized and whose weights are *not* ±α per output channel: the source convolves the
  binarized activation with float weights (BNAS: the convs the first-release BinOp leaves unbinarized). The importer
  materializes the binarizer (`Activation(sign / sign_zero)`, named by `bitino.extra_names[0]`) and imports a float
  `Conv2D`; the checker reports `I-FLOAT-LAYER` instead of `W-FLOAT-WEIGHTS`. Without the key such a layer still warns
  and the importer rejects it (binary weights are never rounded); on ±α weights, on `BipolarQuant` weights, with
  another value or on a layer without a binarized input it is `E-FLOAT-LAYER` (a declaration must not turn a binary
  layer into a float one). It changes no value.
* **IR operator of a group: `bitino.ir`** (node `metadata_props`, group-mapped files, step 3), JSON
  `{"op": <IR operator>, "n_inputs": <int>, "layouts": {"<input index>": "<layout tag>"}}`: the IR operator the group
  becomes, where the ONNX pattern admits several operators that compute the same values (`Add` of a constant is
  `Add`, `AddBroadcast` or a shift-only `ChannelAffine`; `Mul` by a constant is `Mul` or a scale-only
  `ChannelAffine`); `n_inputs` the IR input count where the IR node has trailing empty inputs; `layouts` the layout
  tags of its constants where they differ from the importer's defaults (`OHWI` / `OHWI_bit` for conv kernels, else
  empty). The importer still checks the whole pattern against the named operator; none of the keys changes a value.
  In a group-mapped file one IR constant may be named by several groups (a shared mask or index table); the importer
  requires their data to be equal.
* **IR output names: `bitino.ir` key `outputs`** (step 5), a JSON list of the IR node's output tensor names where
  they are not `<name>:out` (one output) / `<name>:out<i>` (several): the int8_full rewrites keep the float graph's
  tensor names (`layer1.0.prelu:out`, `conv1:out:bits`). The pattern's outputs are taken in the operator's order
  (`y`, then the bit outputs); the names change no value.
* **Attribute representation: `bitino.attrs`** (the group's IR node) and **`bitino.extra_attrs`** (`{extra name:
  {...}}`), JSON objects whose values replace the importer's attribute values only (a) when they are the same number
  in another JSON form (the published files have both `pad_value: -1` and `pad_value: -1.0`), or (b) as the **float64
  value of a float32 ONNX attribute** whose float32 rounding equals the attribute (LayerNorm `eps = 1e-12`,
  LeakyRelu `alpha`). The value `null` means "attribute absent" and is a case of (a): it is accepted only where the
  importer's value is the operator's default (step 3 adds `Conv2D` `dilations [1, 1]`, `groups 1`, `activation none`,
  `BConv2D` `activation none`, `output f32`, `Dense` `activation none` to `pad_value 0`; docs/operators.md: the Larq converters write no
  `pad_value` on a float `Conv2D`, the importer writes `0.0`; step 2). Step 4 adds `AvgPool2D` `count_include_pad 0`
  to the null defaults, and the reverse case: a key the importer does not write may be *added* with the operator's
  default value (`Add` `activation "none"`, the daBNN converter's form). Step 5 adds, for int8_full groups only
  (`qonnx_int8.attrs_meta_i8`), (c) **informational attributes**: an attribute that the operator does not read in the
  node's configuration, so that no pattern can carry it — `y_exp` of an l8 or linear fused output (the grid's
  exponent, read by no kernel), `pad_value` of an unpadded binary convolution, `count_include_pad` of an unpadded pool,
  `pad_q` of an unpadded I8Conv2D, `bit_src` / `bit_binarize` / `r_tile` / `slope_shift` without the input they
  qualify (and `bit_src` of an I8BConv2DResI without slope and out_bias, where `w = acc`), `axis` of a per-tensor
  I8Lut, `y_zero` of an I8Concat that only copies, and the `zeros` / `shifts` entries of the copied inputs of an
  I8Concat (a list attribute entry by entry: the derived entries must agree). Its value comes from `bitino.attrs`
  (absent there = absent in the IR); this is the only case where metadata supplies an attribute value, and only
  where no computed value depends on it; and (d) the form of a **derived axis** (`I8Lut` per channel: the axis of the
  offset table, written `k`, `k − rank` or absent at the default −1 in the IR). Anything else is an import error.
* **Step 6, group mapping of the converters' pre-pass graphs** (the derived `*_unfused.bitino` files with a canonical file
  of their own, `tools/export_unfused_qonnx.py`): the group writer / reader also map `Activation sign` (a standalone
  `BipolarQuant(x, 1)`, R1), the IR `Pad` (`Pad(mode constant)` on the tensor in IR layout: AdaBin's channel padding),
  a float `Conv2D` on a binarized tensor (explicit `Pad(0)` + `Conv(pads 0)`, R2, with `bitino.float_layer`), `BDense`
  with a fused ReLU; `bitino.attrs` may *add* `BDense` `activation "none"` (the operator default, rule (a): the
  FINN-style converters write it) and supplies the informational `pad_value` of an unpadded `BConv2D` (rule (c) in
  float groups: no tap reads it; the OpenVINO 1×1 binary convolutions carry `pad_value −1`). None changes a value.
* **Constant layout tag: `bitino.const_layouts`** (step 4), node key, JSON `{"<input index>": tag}`: the informational
  layout tag (docs/model-format.md) of the group IR node's constant at that input, allowed only between tags that name
  the same storage: a rank-2 bit tensor packed along its last axis is `"OI_bit"` (the importer's) or untagged `""`
  (the tensorpack converter's BDense kernels). The checker rejects any other tag (`E-NAME-FORMAT`). It changes no value.
* **Bit-output layout** (step 4, `bitino.import` option `bit_outputs`): `set` (default: a BConv2D feeding only `ge`
  BConv2D nodes without threshold emits bits), `none` (every output float32: the Larq, daBNN, NNCF, tensorpack and
  FDA-BNN converters) or `threshold` (the OpenVINO converter: a BConv2D without epilogue whose every consumer binarizes
  it with `gt` and one threshold `t` emits bits with `post_add = −(⌊t⌋ + 1)`, exact on the integer accumulator;
  `openvino_ir.fuse_threshold_into_bit_output`). A layout choice, never a value change.

Who writes the names: an exporter writes them from the source (e.g. the PyTorch module path), and for the published
files a step that reads the published graph annotates the canonical QONNX file (planned with step 1); the phase-1
report shows why this is needed (the Brevitas graphs differed only in names).

## 8. Verification contract

For each canonical file (float or int8_full):

1. **Profile:** `bitino check-qonnx FILE` reports no error (warnings reviewed and justified in the conversion
   report).
2. **Export:** the source framework (fp32, and fp64 where it has a mode) against the QONNX file, in the fp32
   reference (onnxruntime, `ORT_DISABLE_ALL`; qonnx `execute_onnx` as the cross-check, below) and in
   `bitino.onnx_f64.OnnxF64` (fp64): top-1 agreement, max |Δlogit|, first binarization flip for every deviating
   sample (as phase 1). (Steps 1–2 recorded the `execute_onnx` comparison; steps 3–5 the `ORT_DISABLE_ALL` session.)
3. **Identity (the acceptance bar):** the importer's `.bitino` has the data SHA-256 and the graph SHA-256 of
   `registry/recipes.json` → `expected[<file>]`.
4. **Semantics:** float files: IR (numpy executor, fp64) against the QONNX fp64 reference within 1e-6, or deviations
   traced to a first flip at a rounding-scale pre-activation, with the agreement rule of
   `docs/numerical-correctness.md`. int8_full files (all patterns B, apart from the open questions): the QONNX file
   evaluated in its declared types (onnxruntime with `ORT_DISABLE_ALL`, or `execute_onnx`) is **bit-identical** to the
   `.bitino` in the numpy executor (`int8_ops` oracle) on the whole check set. Step 5 compares onnxruntime with
   the C++ runtime's `int8_full` and `int8_full_reference` stages on every sample of the recipe's check set (all
   graph outputs byte for byte) and with the numpy oracle on the first samples; the importer additionally
   re-emits every group from the IR node it extracted and requires the file's nodes to equal the canonical pattern
   (operators, attributes, wiring, constant values), so a group that only computes the same values is rejected.
5. **Runtime:** the C++ runtime bit-exact with the numpy executor at its exact stages (unchanged).

The references (step 6, design decision): **fp32** = one **onnxruntime session with `ORT_DISABLE_ALL`** on the
whole file (`bitino.qonnx_groups.ort_session`: no graph rewrite, A2; the QONNX quantizers replaced by their
definitions in standard operators, `BipolarQuant(x, s) = Where(x ≥ 0, s, −s)`, the constant ternary `IntQuant` of
`{−1, 0, +1}` weights = the constant), the session in which the B-class patterns are bit-exact; **cross-check** = qonnx
`execute_onnx` (one onnxruntime session per standard node with that node's default optimizations, qonnx's numpy for
`BipolarQuant` / `IntQuant`): equal to the reference on B-class patterns, within float32 rounding on F-class nodes
(e.g. the NCHWc kernel of a float `Conv`, § 2 A2); **fp64** = `OnnxF64` with every FLOAT tensor widened to float64,
`Cast(DOUBLE constant, FLOAT)` read as the float64 value (R7) and integer operators exact; **int8_full** = the
declared types, without widening. `OnnxF64` does not yet implement `ConvInteger`, `MatMulInteger`, `BitShift` or
`IsNaN`; step 3 added `IsNaN` and unsigned `BitShift`. The int8_full reference is onnxruntime in the declared types
(the patterns are B; no float64 widening applies), so step 5 did not extend `OnnxF64` with `ConvInteger` /
`MatMulInteger`.

## 9. The checker

`python/bitino/qonnx_profile.py` (needs `onnx`, not qonnx); wrappers `tools/check_qonnx.py` (python `-I`) and
`bitino check-qonnx` (in-process when onnx is importable, else `--python PY` or the checkout's
`.cache/venv-qonnx/bin/python`):

```bash
.cache/venv-qonnx/bin/python -I tools/check_qonnx.py model.onnx [more.onnx ...] [--json OUT|-] [--strict] [--no-info]
./bitino check-qonnx --python .cache/venv-qonnx/bin/python model.onnx
```

Exit code 0 = every file passes, 1 = errors (with `--strict` also warnings), 2 = usage error or unreadable file. It
checks the form, not the identity: whether the importer reproduces the published file is item 3 of § 8.

| code | level | rule |
|---|---|---|
| `E-IR`, `E-OPSET`, `E-DOMAIN`, `E-DOMAIN-UNDECLARED`, `E-FUNCTIONS`, `E-EXTERNAL`, `E-SPARSE`, `E-DTYPE` | error | § 1 container |
| `E-INPUT-INIT`, `E-INPUT-TYPE`, `E-INPUT-SHAPE`, `E-IO`, `E-TOPO` | error | graph inputs / outputs, topological order |
| `E-OP`, `E-OP-OPSET` | error | operator not allowed (with the replacement) / not in the declared opset |
| `E-AUTOPAD`, `E-CEIL`, `E-POOL-PAD`, `E-PAD-MODE`, `E-ATTR`, `E-CONVINT-PAD` | error | attribute rules (§ 1, § 2) |
| `E-ROUNDING`, `E-INTQUANT`, `E-INTQUANT-ZP` | error | R5, `IntQuant` constants, zero point of a runtime `IntQuant` |
| `E-BINARIZER-SCALE`, `E-BINPAD-IMPLICIT`, `E-BINPAD-VALUE` | error | R1, R2 |
| `W-ALPHA-INSIDE`, `W-FLOAT-WEIGHTS` | warning | R4 |
| `E-INT64-CLAMP` | error | INT64 `Clip` / `Min` / `Max` (§ 2) |
| `E-META-MISSING`, `E-META-FORMAT`, `E-META-DUP`; `W-META`, `W-META-UNKNOWN`, `W-PRODUCER` | error / warning | § 6 (step 5: `bitino.unused_constants` must list distinct initializers that no node reads) |
| `E-NAME-MISSING`, `E-NAME-FORMAT`, `E-NAME-DUP`, `E-ORDER`; `W-NAME-UNKNOWN` | error / warning | § 7 (step 5: `bitino.ir` may carry `outputs`, a list of distinct IR output names) |
| `E-FLOAT-LAYER`; `I-FLOAT-LAYER` | error / info | § 7 `bitino.float_layer` (step 1b) |
| `I-TERNARY` | info | R4: ternary weights on a binarized input as a plain `{−1, 0, +1}` constant (step 6) |
| `E-ADABIN`; `I-ADABIN` | error / info | § 4.1 AdaBin conv (step 1b): every `Conv` fed by `Mul(Add(Bin(Sub(x, β)), β), α)` must meet the pattern's conditions |
| `E-ONNX-CHECKER`, `E-SHAPES` | error | `onnx.checker` and strict shape inference (QONNX quantizers treated as shape-preserving) give a static shape and type to every tensor |
| `W-OPEN` | warning | a construct of an open question (none at present: `DFT` is rejected since Q1 was decided) |

Tests: `tests/python/test_qonnx_profile.py` (57 tests on synthetic graphs, 2 of step 1b: the float-layer declaration and the AdaBin pattern; a complete valid file passes with no
finding above info, and each rule is broken once).

### Phase-1 files (2026-11-20, read only)

The 13 ONNX files under `.cache/qonnx_phase1/*/` (8 cleaned `model_clean.onnx`, 5 raw Brevitas exports `model.onnx`)
all **fail**, none for a semantic reason that blocks conversion. What each lacks:

| file group | findings |
|---|---|
| all 13 | `E-META-MISSING` (no `bitino.*` metadata at all); `E-NAME-MISSING` (no node carries `bitino.name`: 18 / 35 / 30 / 52 activation-path nodes); `E-IR` (IR 6–8, metadata_props needs 10) |
| 5 Brevitas `model_clean.onnx` + FINN GTSRB | `E-OP` `BatchNormalization` (8 in CNV, 3 in the MLPs) and `Quant` (CNV input quantizer, should be `IntQuant`); `E-DOMAIN` the file imports `onnx.brevitas` but its nodes use `qonnx.custom_op.general`, which it does not import (`E-DOMAIN-UNDECLARED`): qonnx cleanup renames the nodes' domain without updating `opset_import`. GTSRB also `E-OPSET` (opset 11) |
| 5 Brevitas raw `model.onnx` | as above, with the nodes still in `onnx.brevitas`, and `E-INPUT-INIT` (16 / 36 initializers listed as graph inputs) |
| FINN CNV / TFC releases | `E-OPSET` (opset 12), `E-OP` `MultiThreshold` (9 / 4), `BatchNormalization` (8 / 3), `TopK`; shape inference not reached |

Nothing else fails: binary padding (all CNV convolutions are unpadded), binary weights (`BipolarQuant(W, 1)`: no
`W-ALPHA-INSIDE`), attributes and, where it runs, shape inference pass. Converting them is mechanical: fold BN in
float64 (R6), `Quant` → `IntQuant`, `MultiThreshold` → `Where(GreaterOrEqual)`, drop `TopK`, fix `opset_import`, raise
the IR version, add metadata and names.

## Design questions

1. **Q1 `FftLogMagnitude` (18 detectors). Decided 2026-11-20:** a double-precision DFT as `MatMul` on cos / sin
   tables computed in float64, under the float tolerance contract (F), not bit-exact; no unrolled exact FFT (§ 4.5).
2. **Q2 F-class float operators.** Float `Conv` / `Gemm` / `MatMul`, `AveragePool` / `GlobalAveragePool` / `ReduceMean`,
   `Sigmoid`, `Tanh`, `Gelu`, `LayerNormalization`, `Softmax`, `ChannelGate` are F: Bitino evaluates them with double
   accumulation or double libm functions. Confirm that F + Level 3 (today's rule) is the bar for these, with byte
   identity carried by the constants. Making them B would need DOUBLE subgraphs with a fixed summation order and would
   still depend on the reference runtime's libm for `exp` / `erf` / `tanh`.
3. **Q3 α inside binary weights.** `BipolarQuant(W, α)` / `±α` constants are accepted with a warning (fp32 evaluation
   F). Make it an error (canonical: ±1 weights + `Mul(α)`, B)?
4. **Q4 two quantizer forms.** Activations with zero points outside INT8 (binarize-aware grids, `|z| < 2^20`) need the
   explicit `Div / Round / Add / Clip / Cast` form; `QuantizeLinear` covers the rest. Allow both forms, or require the
   explicit form everywhere for uniformity? (Step 5 writes the explicit form everywhere, for `I8Quantize` and every
   `DEQ`; the binary operands' grids reach `|z| ≈ 5.2·10^5`.)
5. **Q5 `L8ENC` cost.** The exact log2 encoder is a 127-way comparison and a sum per element (no bitcast in standard
   ONNX up to opset 21). Exact and verified; acceptable as the file form? (Step 5 uses it for every l8 output: 59
   `I8Conv2D` stems and 437 `I8BConv2DRes` blocks of 22 files (the 20 log2 files and two model_int8_full files); its
   `[…, 127]` comparison tensor is the largest
   intermediate of those files, about 0.5 GB at a 112 × 112 × 64 stem. An exact cheaper form exists: a 7-step
   branchless bisection, `cnt += 2^j · (|w| ≥ β[cnt + 2^j − 1])` with a `Gather` of the boundaries, j = 6..0.)
   **Decided (step 6):** the 7-step bisection is the file form (§ 5 `L8ENC`), proved equal to `l8_encode` on every
   non-NaN fp32 input; the 22 files re-verified bit-exact against the C++ runtime.
6. **Q6 signed shifts.** Arithmetic right shifts use the UINT64 offset construction (`ASR`), because ONNX `BitShift`
   is unsigned-only and integer `Div` rounding is not specified in the ONNX text. Acceptable, or prefer `Div` with the
   onnxruntime behaviour (truncation) documented as an assumption? (Step 5 uses the construction in every integer
   epilogue: 101 files, bit-exact against the C++ kernels.)
7. **Q7 out-of-range ids.** Bitino's `Gather` returns a zero row (`I8Gather`: `fill_q`) for ids outside `[0, V)`;
   ONNX wraps negative ids and fails above V. Canonical files rely on the declared domain `ids:<V>`. Or should they
   carry the guarded form `Where(in_range, Gather(table, Clip(ids)), 0)`? (Step 5: the int8_full `I8Gather` is
   written in the guarded form, because its `fill_q` is part of the IR node and only the guard carries it; the float
   files of step 3 keep the plain `Gather` on the declared domain.)
8. **Q8 one opset or a range.** v1 allows ai.onnx 17–21 (17 is what the PyTorch / tf2onnx exporters write; `Gelu`
   needs 20). Pin a single version instead (then exporters run `onnx.version_converter`)?
9. **Q9 explicit binary padding.** Canonical files must pad binarized tensors with an explicit `Pad` and `pads = 0` on
   the `Conv` (R2). `torch.onnx.export` puts the padding into the `Conv`, so the PyTorch exporter needs a rewrite.
   Keep R2 strict, or accept `Conv` zero padding of a binarized tensor as an implicit `pad_value 0`?
10. **Q10 FINN / Brevitas inputs.** The canonical file of a FINN release is a rewrite of it (no `MultiThreshold`, no
    `TopK`, BN folded); the release ONNX stays the converter's input and its SHA-256 goes into `bitino.exporter` /
    the manifest. Confirm.
11. **Q11 whole-file identity. Decided 2026-11-20:** data + graph SHA-256 identity is sufficient (tensor data and
    graph structure including layer / node / constant names, byte for byte); the manifest (converter name, timestamps,
    provenance) is regenerated and not compared.
12. **Q12 LBP condition.** `LocalBinaryPattern` is B only when every sample point's shift is uniform over the image
    (true for the offsets of all 18 detectors). For a future model that violates it: reject the file, or carry the
    general per-pixel form (per-pixel floor, gathered with computed indices)?
