# QONNX as Bitino's model interface (proposal)

Status: **proposal**, nothing below is implemented yet except where marked *today*. Decided (2026-11-19): QONNX is the
**default route for new models**; existing families move to it only when the new route reproduces their files. Companion guide:
[converting-a-new-bnn.md](converting-a-new-bnn.md).

## Why

Today every family of published BNNs has its own converter: 30 scripts (`tools/convert_*.py`) that each read one
framework's checkpoint and emit Bitino's graph IR directly. Each one re-implements the same decisions (what `sign(0)`
is, how binary padding works, NCHW → NHWC, folding BatchNorm into thresholds) for one framework's conventions. Adding a
BNN means writing or extending a converter.

The proposal is to put one standard format between the frameworks and Bitino:

```
 checkpoint ──(framework exporter)──► QONNX ──(qonnx cleanup)──► Bitino importer ──► graph IR ──► passes ──► .bitino
  Brevitas: export_qonnx          one file, standard ONNX     pattern rules       existing      existing
  PyTorch:  torch.onnx.export     + BipolarQuant / IntQuant   (one module)        IR + checks   format
  QKeras / HAWQ: their exporters
```

[QONNX](https://github.com/fastmachinelearning/qonnx) (Apache-2.0, v1.0.0, 2026-02) is ONNX plus four quantizer
operators: `BipolarQuant`, `IntQuant` (formerly `Quant`), `FloatQuant` and `Trunc`. Brevitas, QKeras and HAWQ already
export it, and FINN and hls4ml import it. With QONNX in the middle:

* **Framework-specific work shrinks to export.** Brevitas models export with one call; plain PyTorch models export
  with `torch.onnx.export` once their binarizer is an exportable op.
* **One importer owns the semantics.** The rules in the table below live in one place, with one set of tests.
* **The reference is standard tooling.** The source model can be checked with `qonnx-exec` / onnxruntime instead of a
  per-framework harness. The float64 reference becomes a general numpy evaluator of the QONNX graph; *today* a narrow
  one exists, `python/bitino/adapters/finn_onnx.OnnxNumpy`.
* **The environment already exists.** *Today* `.cache/venv-qonnx` has qonnx 1.0.0 and onnxruntime 1.31.1, used by
  `tools/convert_finn_onnx.py` for three FINN/QONNX networks.

Nothing changes for the runtime, the file format, the int8_full quantizer or the published files.

## The kernel set

The importer targets the operators that the 132 converted networks actually use: 29 operator types in total. The
`I8*` full-integer operators are not imported; `tools/quantize_int8_full.py` generates them from the float graph, as
today.

Each IR operator has several kernels, selected at run time by stage and CPU features (`docs/operators.md`). The table
lists, for each operator, how often the zoo uses it and the QONNX / ONNX pattern the importer maps onto it.

Notation: `Bin(x)` is a *binarizer*, any of:
* `BipolarQuant(x, s)`: `x ≥ 0 → +s`, else `−s`. Bitino mode `ge`; Larq `ste_sign` has the same semantics.
* ONNX `Sign(x)`: `0 → 0`. Bitino mode `sign_zero`; this is `torch.sign` in many PyTorch BNNs.
* `Where(GreaterOrEqual(x, t), 1, −1)` / `Where(Greater(x, t), 1, −1)`: modes `ge` / `gt` with threshold `t`.

A per-channel constant subtracted before `Bin` (ReActNet's learned sign threshold, RSign) becomes the operator's
`threshold`.

### Binary operators (the reason the runtime exists)

| IR op | zoo use (nodes / models) | QONNX / ONNX pattern | notes |
|---|---:|---|---|
| `BConv2D` | 3207 / 119 | `Conv(Bin(x), W)` where `W` is `Bin(W_latent)` or a constant whose values per output channel are `±α[o]`; optional `Pad(value v)` before `Conv`; per-channel `Mul`/`Add`/`BatchNormalization` after | `Conv`'s own zero padding of a ±1 tensor = binary `pad_value 0`; an explicit `Pad(1)` / `Pad(−1)` = `pad_value ±1`; `α[o]` and the affine fold into the epilogue `mul`/`add`; groups supported |
| `BDense` | 1084 / 23 | `Gemm` / `MatMul(+Add)` with the same binarized operands | ternary weights (`w_zero`): `IntQuant(W, bitwidth 2, narrow)` |
| `BMatMul` | 642 / 11 | `MatMul(Bin(a), Bin(b))`, both runtime tensors | BiBERT `{0,1}` binarizer: `Relu(Sign(x))` → mode `gt0` |
| `QBConv2D` / `QBDense` | 4 / 1 + 2 / 1 | `Conv(IntQuant(x, bits ≤ 8, signed 0), Bin(W))` | DoReFa k-bit activations |

### Float compute

| IR op | zoo use | pattern | notes |
|---|---:|---|---|
| `Conv2D` | 2192 / 123 | `Conv` with float weights (stem, downsampling, 1×1) | depthwise = `group = C` |
| `Dense` | 180 / 109 | `Gemm`, or `MatMul` + `Add` with a constant weight | classifier |
| `MatMul` | 40 / 4 | `MatMul` of two runtime tensors | attention |

### Per-channel and element-wise

| IR op | zoo use | pattern | notes |
|---|---:|---|---|
| `ChannelAffine` | 8658 / 119 | `BatchNormalization` (inference), `Mul`/`Add` with a `[1,C,1,1]` constant | folded into neighbours by the existing passes |
| `Activation` | 2675 / 101 | `Relu`, `Clip`, `PRelu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `HardSwish`, `Gelu` (opset 20) or the `Erf` decomposition, `Sign` not feeding a binary op | RPReLU = `Add` + `PRelu` + `Add` |
| `Add` | 5453 / 104 | `Add` of equal shapes | residuals |
| `AddBroadcast` | 447 / 19 | `Add` with broadcasting | attention masks |
| `Mul` | 228 / 27 | `Mul` of runtime tensors | |
| `ChannelGate` | 861 / 25 | `Mul(x, Sigmoid(g))` with `g` of shape `[N,C,1,1]` | squeeze-and-excitation |
| `FakeQuant` | 48 / 11 | a standalone `IntQuant` (not feeding a binary op) | quantize-dequantize, e.g. an 8-bit input |

### Pooling, shape, normalization

| IR op | zoo use | pattern |
|---|---:|---|
| `MaxPool2D` / `AvgPool2D` | 313 / 69, 363 / 65 | `MaxPool`, `AveragePool` (`count_include_pad` kept) |
| `GlobalAvgPool` / `ReduceMean` | 902 / 76, 300 / 15 | `GlobalAveragePool`, `ReduceMean` |
| `Concat` / `Reshape` / `Transpose` / `Pad` / `Gather` | 737, 3077, 468, 70, 1017 nodes | `Concat`, `Reshape`/`Flatten`/`Squeeze`/`Unsqueeze`, `Transpose`, `Pad`, `Gather` |
| `LayerNorm` / `Softmax` | 434 / 11, 175 / 22 | `LayerNormalization` (opset 17) or its `ReduceMean`/`Sub`/`Pow`/`Sqrt`/`Div` decomposition, `Softmax` |

### Image features (custom domain)

| IR op | zoo use | pattern |
|---|---:|---|
| `ImageGrayU8`, `FftLogMagnitude`, `LocalBinaryPattern` | 18 / 18 each | nodes in an `ai.bitino` custom domain, with the same attributes as `docs/operators_image_features.md` (the Faster-Than-Lies detectors compute these features inside the model) |

The `ai.bitino` domain is the escape hatch for any operator that standard ONNX + QONNX cannot express. Every node in
it must have a numpy reference implementation registered with the importer, so the QONNX graph stays executable for
verification.

## Semantics the profile fixes

These are the points where frameworks disagree and where today's converters each carry their own logic:

1. **`sign(0)`.** `BipolarQuant` maps 0 to +1 and ONNX `Sign` maps 0 to 0. The importer keeps whichever the model
   uses, and never silently replaces one with the other.
2. **Binary padding.** ONNX `Conv` pads its input with 0. After a binarizer, that is the binary-domain value 0 (the
   PyTorch / Larq default). Padding with ±1 must be an explicit `Pad` node.
3. **Layout.** QONNX graphs are NCHW, and the importer converts to NHWC itself. `qonnx-to-channels-last` is not
   required.
4. **Weights.** Latent float weights followed by `Bin`, or constants already in `{±α[o]}`, are both accepted. The
   importer checks the values and packs them; a constant that is not exactly `±α[o]` per channel is rejected, not
   rounded.
5. **Rounding.** `IntQuant` `ROUND` (= half-even) matches Bitino's `rne`. Other rounding modes are rejected unless a
   kernel implements them.
6. **Constant folding.** `qonnx-cleanup` may fold `Bin(W_latent)` into a ±1 constant, which rule 4 accepts. Folding
   across a binarizer of a *runtime* tensor is not allowed, and the importer checks for it.

## Verification (unchanged rules, new references)

The acceptance rules of `docs/numerical-correctness.md` stay the same: float64 reference, first-flip tracing,
agreement ≥ 98 %, at most the source's own float32 disagreements plus max(1, 1 %). Only the reference changes:

* **float32:** `qonnx.core.onnx_exec.execute_onnx` on the exported graph (onnxruntime for standard nodes, qonnx's
  numpy for its quantizers).
* **float64:** a general numpy evaluator of the QONNX graph (onnxruntime has no float64 Conv), generalizing *today's*
  `OnnxNumpy` to the patterns above.
* **The source framework itself** in float32 (and float64 where it has a mode for it), compared with the exported
  QONNX graph. This checks the *export*, a step that does not exist today.

## Plan

| phase | work | done when |
|---|---|---|
| 0 | this document + guide | agreed |
| 1 (**done** 2026-11-19, [qonnx-phase1.md](qonnx-phase1.md): 3 identical files, 5 identical data + bit-identical outputs, graphs equal up to node names) | `python/bitino/qonnx_import.py` for the binary, float, element-wise and pooling sets; general float64 evaluator; `bitino import-qonnx` CLI | the 3 FINN/QONNX and 5 Brevitas networks re-imported through QONNX give files with the **same data and graph SHA-256** as the published conversions (or, if the graph differs, bit-identical outputs on the check set) |
| 2 | plain-PyTorch export path (exportable binarizer modules, `torch.onnx.export`) | one network per PyTorch family (ReActNet / AB-BNN, Bi-Real, RBONN, BNext, ReCU, AdaBin, BiPer, ...) reproduces its published file or outputs |
| 3 | transformers (BMatMul, LayerNorm, Gather) and the `ai.bitino` image-feature domain | BiBERT, BHViT and a Faster-Than-Lies detector reproduce |
| 4 | Larq / Keras via tf2onnx (`ste_sign` → `BipolarQuant` pattern) | Larq Zoo QuickNet and Bi-Real reproduce |
| 5 | retire each converter whose family reproduces; keep the others as legacy front ends that write QONNX | `tools/convert_*.py` count shrinks; every recipe still rebuilds identical files |

Shipped files never change because of the migration. A family moves to the QONNX path only when the new path
reproduces its files or its outputs exactly.

**Expected to stay outside** (front ends that build the graph themselves, emitting QONNX instead of IR):
* OpenVINO IR: its `BinaryConvolution` has no ONNX exporter.
* MindSpore, tensorpack and Theano/Lasagne checkpoints, which have no live exporter.
