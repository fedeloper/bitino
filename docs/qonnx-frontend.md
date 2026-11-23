# QONNX as Bitino's model interface

Status: **done (step 6, 2026-11-21): QONNX is THE build path.** Every published file of the 132 networks (195 float
files, 101 int8_full files) and every derived file is built checkpoint → canonical QONNX file → `bitino.qonnx_import`
→ `.bitino` by `tools/build_from_qonnx.py` (routes: `registry/qonnx_routes.json`) with the published data and graph
SHA-256; the IR-emitting converters `tools/convert_*.py` are legacy cross-checks (`tools/qonnx_legacy_crosscheck.py`).
Overview of the standard: [qonnx-standard.md](qonnx-standard.md); step-6 report: [qonnx-step6.md](qonnx-step6.md).
History: decided 2026-11-19: QONNX is the default route for new models. Decided 2026-11-20:
**QONNX is the standard format of every Bitino network**: all 132, float files and int8_full files. Each network gets
one canonical QONNX file per shipped `.bitino` file, and the `.bitino` is derived from it by one importer. The standard
is [qonnx-profile.md](qonnx-profile.md) (Bitino QONNX profile v1: standard ONNX + QONNX operators only, required
metadata, naming metadata, verification contract), checked by `bitino check-qonnx`. Phase 1 (importer, 8 networks) is
done: [qonnx-phase1.md](qonnx-phase1.md); phase 2 (plain-PyTorch exporter, 4 families byte-identical): [qonnx-phase2.md](qonnx-phase2.md); step 1 (canonical PyTorch files, 43 networks byte-identical): [qonnx-step1.md](qonnx-step1.md); step 2 (Keras / Larq): [qonnx-step2.md](qonnx-step2.md); step 3 (transformers, image features): [qonnx-step3.md](qonnx-step3.md); step 5 (int8_full files): [qonnx-step5.md](qonnx-step5.md). Companion guide: [converting-a-new-bnn.md](converting-a-new-bnn.md).

**Acceptance bar.** A family moves to the QONNX route only when the `.bitino` rebuilt from its canonical QONNX file is
**byte-identical** to the published one: the same data SHA-256 and graph SHA-256 (`registry/recipes.json` →
`expected`), including node and constant names, node order and constant order. Bit-identical outputs alone are not
enough. Decided 2026-11-20: data + graph SHA-256 identity is sufficient; whole-file / manifest identity (converter name,
timestamps, provenance fields) is not required.

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

Nothing changes for the runtime, the file format or the published files. The int8_full quantizer will write its
graphs as canonical QONNX files too (step 5).

## The kernel set

The importer targets the operators that the 132 converted networks actually use: 29 operator types in total. The
`I8*` full-integer operators come from `tools/quantize_int8_full.py` today; in step 5 they get canonical QONNX files of
their own, written with standard integer operators (`ConvInteger`, `MatMulInteger`, integer epilogues, `Gather` tables),
patterns in [qonnx-profile.md § 5](qonnx-profile.md#5-int8_full-files).

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
| `BConv2D` | 3207 / 119 | `Conv(Bin(x), W)` where `W` is `Bin(W_latent)` or a constant whose values per output channel are `±α[o]`; optional `Pad(value v)` before `Conv`; per-channel `Mul`/`Add` after (BatchNorm folded in float64) | canonical files pad a ±1 tensor only with an explicit `Pad(v ∈ {−1, 0, +1})` (the phase-1 importer also reads `Conv`'s own zero padding as `pad_value 0`); canonical weights are ±1 with `α[o]` as a `Mul` after; groups supported |
| `BDense` | 1084 / 23 | `Gemm` / `MatMul(+Add)` with the same binarized operands | ternary weights (`w_zero`): `IntQuant(W, bitwidth 2, narrow)` |
| `BMatMul` | 642 / 11 | `MatMul(Bin(a), Bin(b))`, both runtime tensors | BiBERT `{0,1}` binarizer: `Relu(Sign(x))` → mode `gt0` |
| `QBConv2D` / `QBDense` | 4 / 1 + 2 / 1 | `Conv(Round(Mul(Clip(x, 0, 1), 2^k − 1)), Bin(W))` | DoReFa k-bit activations as integer levels (bit-exact; `IntQuant(x, 1/n)` is not, since `x / fl(1/n) ≠ x · n` in fp32) |

### Float compute

| IR op | zoo use | pattern | notes |
|---|---:|---|---|
| `Conv2D` | 2192 / 123 | `Conv` with float weights (stem, downsampling, 1×1) | depthwise = `group = C` |
| `Dense` | 180 / 109 | `Gemm`, or `MatMul` + `Add` with a constant weight | classifier |
| `MatMul` | 40 / 4 | `MatMul` of two runtime tensors | attention |

### Per-channel and element-wise

| IR op | zoo use | pattern | notes |
|---|---:|---|---|
| `ChannelAffine` | 8658 / 119 | `Mul`/`Add` with a `[1,C,1,1]` constant (BatchNorm folded in float64 by the exporter; `BatchNormalization` itself only in non-canonical inputs) | folded into neighbours by the existing passes |
| `Activation` | 2675 / 101 | `Relu`, `Clip`, `PRelu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `Gelu` (opset 20) or the `Erf` decomposition, `Sign` not feeding a binary op | RPReLU = `Add` + `PRelu` + `Add` |
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

### Image features (standard-op subgraphs)

| IR op | zoo use | pattern |
|---|---:|---|
| `ImageGrayU8`, `FftLogMagnitude`, `LocalBinaryPattern` | 18 / 18 each | subgraphs of standard ONNX operators that the importer matches back onto the fused kernels (the Faster-Than-Lies detectors compute these features inside the model): gray conversion with `Mul` / `Floor` / `Clip` / integer `Mul` / `ReduceSum` / `BitShift` (bit-exact), LBP with `Pad` / `Slice` / DOUBLE arithmetic / comparisons (bit-exact), the FFT feature as a double-precision DFT (`MatMul` on float64 cos / sin tables) under the float contract, not bit-exact (decided 2026-11-20) |

There is no private domain. Every Bitino operator, including the fused full-integer kernels, is written with
standard ONNX + QONNX operators; [qonnx-profile.md](qonnx-profile.md) gives each pattern and says whether it is
bit-exact with the kernel and why. A construct whose exact standard-op form is not established is an open question of
the profile, not a reason for a custom operator.

## Semantics the profile fixes

These are the points where frameworks disagree and where today's converters each carry their own logic. The binding
rules are [qonnx-profile.md § 3](qonnx-profile.md#3-semantic-rules); in short:

1. **`sign(0)`.** `BipolarQuant` maps 0 to +1 and ONNX `Sign` maps 0 to 0. The importer keeps whichever the model
   uses, and never silently replaces one with the other.
2. **Binary padding.** A binarized tensor is padded only by an explicit `Pad` with a value in {−1, 0, +1}, and the
   `Conv` reading it has `pads = 0` (canonical files). The phase-1 importer also reads `Conv`'s own zero padding of a
   ±1 tensor as the binary value 0 (the PyTorch / Larq default).
3. **Layout.** QONNX graphs are NCHW, and the importer converts to NHWC itself. `qonnx-to-channels-last` is not
   required.
4. **Weights.** Latent float weights followed by `Bin`, or constants already in `{±α[o]}`, are both accepted. The
   importer checks the values and packs them; a constant that is not exactly `±α[o]` per channel is rejected, not
   rounded.
5. **Rounding.** `IntQuant` `ROUND` (= half-even) matches Bitino's `rne`. Other rounding modes are rejected unless a
   kernel implements them.
6. **Constant folding.** `qonnx-cleanup` may fold `Bin(W_latent)` into a ±1 constant, which rule 4 accepts. Folding
   across a binarizer of a *runtime* tensor is not allowed, and the importer checks for it.
7. **BatchNorm** is folded by the exporter in float64 into `Mul` / `Add` whose constants are float64 initializers read
   through `Cast(to FLOAT)` (ONNX `BatchNormalization` stores epsilon as float32).

## Verification (unchanged rules, new references)

The acceptance rules of `docs/numerical-correctness.md` stay the same: float64 reference, first-flip tracing,
agreement ≥ 98 %, at most the source's own float32 disagreements plus max(1, 1 %). Only the reference changes:

* **float32:** `qonnx.core.onnx_exec.execute_onnx` on the exported graph (onnxruntime for standard nodes, qonnx's
  numpy for its quantizers).
* **float64:** `bitino.onnx_f64.OnnxF64`, a general numpy evaluator of the QONNX graph (onnxruntime has no float64
  Conv); float64 carriers `Cast(DOUBLE, FLOAT)` are read at their float64 value.
* **int8_full files:** the QONNX graph in its declared integer / float types must be bit-identical to the `.bitino`
  (all their patterns are exact; [qonnx-profile.md § 8](qonnx-profile.md#8-verification-contract)).
* **The source framework itself** in float32 (and float64 where it has a mode for it), compared with the exported
  QONNX graph. This checks the *export*, a step that does not exist today.

## Plan

| step | work | done when |
|---|---|---|
| done before (phase 1, 2026-11-19) | `python/bitino/qonnx_import.py` (binary, float, element-wise, pooling sets), `OnnxF64`, `bitino import-qonnx`; [qonnx-phase1.md](qonnx-phase1.md) | 3 FINN / QONNX files identical; 5 Brevitas files with identical data, graphs equal up to names |
| **0** (**done** 2026-11-20) | the standard: [qonnx-profile.md](qonnx-profile.md) (opsets, allowed operators and the pattern of every IR operator with its exactness, semantic rules, metadata, naming / order metadata, verification contract, open questions) and its checker `python/bitino/qonnx_profile.py` / `tools/check_qonnx.py` / `bitino check-qonnx` | the profile is agreed (open questions answered); the checker runs on every canonical file |
| 1 (**done** 2026-11-21 with step 1b ([qonnx-step1b.md](qonnx-step1b.md)): all 59 PyTorch CNN networks; first slice 2026-11-20: 43 networks, every published float file byte-identical from a profile-conformant canonical file: IR-Net family (22), AB-BNN ReActNet-18/34 (4), ReActNet-A (5), RBONN / SNN / IDa-Det (4), SA-BNN, BNext (4), bitorch (3), 81 of the 83 files also fully verified (BNext-L: identity only, memory); open: BBCU, BSRC, BNAS, BiMatting, AdaBin: [qonnx-step1.md](qonnx-step1.md)) | PyTorch CNNs: exportable binarizer modules + `torch.onnx.export`, rewritten to the profile (explicit binary `Pad`, BN folded in float64, ±1 weights + `Mul(α)`), names annotated from the published graphs; the importer reads the naming metadata and writes first-use constant order | every PyTorch CNN family (ReActNet / AB-BNN, Bi-Real, RBONN, BNext, ReCU, AdaBin, BiPer, IR-Net family, ...) rebuilds its published `model.bitino` byte for byte from its canonical file; the 8 phase-1 networks get canonical files too |
| 2 (**done** 2026-11-21: all 23 Keras / Larq networks, see [qonnx-step2.md](qonnx-step2.md)) | Larq / Keras: a direct writer from the Keras model to the profile (`ste_sign` → `BipolarQuant`, `pad_values` → explicit `Pad`), no tf2onnx in the identity path: `tools/export_keras_qonnx.py` | Larq Zoo and the other Keras families rebuild byte for byte |
| 3 (**done** 2026-11-21: all 29 networks, 39 float files, see [qonnx-step3.md](qonnx-step3.md)) | transformers (BMatMul, LayerNorm, Gather, BiBERT's `Relu(Sign)`) and the image-feature subgraphs: one ONNX group per IR node of the deployed graph (`python/bitino/qonnx_groups.py`, `tools/export_groups_qonnx.py`), import mode recorded in the file (`bitino.import`) | BiBERT, Bi-ViT, BHViT, GSB and the 18 Faster-Than-Lies detectors rebuild byte for byte (the FFT feature: DOUBLE `MatMul` DFT, float contract, decided 2026-11-20) |
| 4 (**done** 2026-11-21: all 21 networks, 22 float files, see [qonnx-step4.md](qonnx-step4.md)) | front ends without an ONNX exporter write the profile themselves: OpenVINO IR (`BinaryConvolution`), daBNN ONNX, NNCF, MindSpore, tensorpack, Theano / Lasagne, and the FINN / Brevitas networks whose phase-1 files were not canonical: a direct writer from each legacy adapter's parse (`tools/export_ir_qonnx.py`, `tools/qonnx_ir_writer.py`), import layout recorded in the file (`bitino.import` mapping `patterns`) | each of their networks rebuilds byte for byte |
| 5 (**done** 2026-11-21: all 101 published int8_full files, see [qonnx-step5.md](qonnx-step5.md)) | int8_full as QONNX: the writer (`tools/export_int8_qonnx.py`, `python/bitino/qonnx_int8.py`) writes canonical int8_full files from the published int8_full graphs (INT8 tensors, `ConvInteger` / `MatMulInteger`, integer epilogues, `Gather` tables, the l8 code); the importer maps them onto the `I8*` operators | every published `model_int8_full*.bitino` rebuilds byte for byte, and its QONNX file is bit-identical to it on the check sets |
| 6 (**done** 2026-11-21, see [qonnx-step6.md](qonnx-step6.md)) | every recipe builds through checkpoint → QONNX → `.bitino`: one entry point `tools/build_from_qonnx.py` and per-recipe routes (`registry/qonnx_routes.json`: the writer of each file's canonical file, or the derivation of a derived file); the step-1 / 1b / 2 import arguments moved into the files (`bitino.import`), so every canonical file determines its `.bitino` alone; L8ENC as the exact 7-step bisection; the IR-emitting converters kept, marked, as legacy cross-checks (`tools/qonnx_legacy_crosscheck.py`) | all 296 published files and the derived files rebuild through the entry point with the published data and graph SHA-256. `bitino fetch` (public recipes) still runs the legacy converters, which stay as cross-checks; switching it is a later step (qonnx-step6.md) |

Shipped files never change because of the migration: a family switches only when its files reproduce byte for byte.
