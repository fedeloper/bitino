# QONNX route, phase 1: acceptance report

Plan: [qonnx-frontend.md § Plan](qonnx-frontend.md#plan), phase 1. Date: 2026-11-19, host only (x86-64), nothing under
`artifacts/` was written. Machine-readable results: the development log (not published). Re-imported files and reports:
`.cache/qonnx_phase1/<id>/`. Command to reproduce everything:

```bash
scripts/capped.sh 8G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_phase1_acceptance.py
```

## What was built

| file | role |
|---|---|
| `python/bitino/qonnx_import.py` | the importer: pattern rules from a (qonnx-cleaned) QONNX / FINN-ONNX graph to the IR, the Level-3 verification, and the CLI `main` |
| `python/bitino/onnx_f64.py` | general numpy evaluator of ONNX / QONNX graphs in float64 (or float32); generalizes `adapters.finn_onnx.OnnxNumpy`, which is unchanged |
| `tools/import_qonnx.py` | `python -I` wrapper of `qonnx_import.main` (the `.cache` venvs do not have `bitino` installed, so `python -I -m bitino.qonnx_import` cannot find the package) |
| `bitino import-qonnx ...` | CLI subcommand (`python/bitino/cli/main.py`); runs in-process when `onnx` is importable, or `--python .cache/venv-qonnx/bin/python` |
| `tools/export_brevitas_qonnx.py` | Brevitas exporter step: loads the checkpoint with `convert_brevitas.load_model` (SHA-256 checked), `brevitas.export.export_qonnx`, writes `model.onnx`, `export.json` and the source model's own fp32 / fp64 logits on the eval set |
| `tools/qonnx_phase1_acceptance.py` | this acceptance run (export / extract, import, compare with the published files) |
| `tests/python/test_qonnx_import.py` | 22 tests on synthetic QONNX graphs |

Pipeline per network: source ONNX → `qonnx.util.cleanup.cleanup_model(preserve_qnt_ops=True)` → import →
`passes.optimize` + bit outputs → `model.bitino` + `conversion_report.json` (provenance: ONNX SHA-256, producer,
opsets, exporter and versions, checkpoint SHA-256 for Brevitas, cleanup file SHA-256, input domain, preprocessing,
rules applied, input-quantizer / threshold checks, integer-threshold margins) → verification: qonnx `execute_onnx`
(fp32) and `OnnxF64` (fp64) of the cleaned graph vs the numpy executor of the written file (fp32 and fp64), first
binarization flip per deviating sample, the source framework's own logits vs the export (Brevitas), the raw vs cleaned
graph in fp64, and the C++ host runtime (`tools/validate_model.py --require-exact-numpy`, stages `reference` and
`optimized`, `build/host/bitino_bench`, not rebuilt: it is newer than every runtime source).

Input domains (`--input-domain`): `unit8` = float32(k)/255 for the ToTensor eval sets (Brevitas, GTSRB), `uint8` =
float32(k) for the FINN-ONNX graphs whose `Div 255` is inside the graph. The domain is used only where a rule is exact
on a discrete domain and not on all floats (FINN's 255-threshold MultiThreshold input quantizer, and the
`Div 255 → Mul 2 → Sub 1 → binarizer` threshold of FINN TFC); every such use is an exhaustive check over the 256 values,
recorded in the report. The Brevitas TFC threshold 0.5 also verifies on the `float32` domain (same file).

## Acceptance table

Data SHA-256 = `data_sha256` of the data section, graph SHA-256 = `fetch.graph_sha256` (constants table, inputs, nodes,
outputs; not the manifest). "Expected" = `registry/recipes.json` → `expected["model.bitino"]`; for all 8 networks the
published `artifacts/<id>/model.bitino` has exactly the expected hashes. "Up to names" = the graph JSON with positional
names (constants `c<i>`, nodes `n<i>`, outputs `n<i>:o<j>`) is identical, i.e. same ops, order, attributes, dtypes, shapes,
layouts, offsets, sizes and wiring. Outputs = all eval128 samples of the model's check set, numpy executor (fp32) and
C++ runtime (stage `reference`), both files, compared bit for bit.

| network | source ONNX SHA-256 | data SHA-256 (re-import) | data = expected | graph re-import / expected | graph = | up to names | renamed nodes / constants | outputs bit-identical numpy / C++ |
|---|---|---|---|---|---|---|---|---|
| `brevitas-bnn-pynq-cnv-1w1a` | a02a002f4eff (export) | `d7223bccd797d8cc` | yes | `22ae302d08eb2efc` / `920e0b7263483c0a` | no | yes | 14 / 30 | yes / yes |
| `brevitas-bnn-pynq-lfc-1w1a` | 4527391f4831 (export) | `12198dfceb7294ed` | yes | `071fd7ca1210ad84` / `06152823517f4978` | no | yes | 5 / 13 | yes / yes |
| `brevitas-bnn-pynq-sfc-1w1a` | 40bc9952f487 (export) | `aa5848d66e0dd551` | yes | `8425ae740d866673` / `324c7b2789ba64e2` | no | yes | 5 / 13 | yes / yes |
| `brevitas-bnn-pynq-tfc-1w1a` | c08cc653d527 (export) | `3531cef0e4e18773` | yes | `86d500565979dca5` / `c47d1a9bc2e276c1` | no | yes | 5 / 13 | yes / yes |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | 972094148d58 (export) | `829c77f6dcf899c5` | yes | `22ae302d08eb2efc` / `920e0b7263483c0a` | no | yes | 14 / 30 | yes / yes |
| `finn-examples-bnn-pynq-cnv-w1a1` | 47007fe69572 (release) | `602cbf84cacba194` | yes | `22ae302d08eb2efc` / `22ae302d08eb2efc` | **yes** | yes | – | yes / yes |
| `finn-examples-bnn-pynq-tfc-w1a1` | 1706b590920d (release) | `14c00376ac27547d` | yes | `86d500565979dca5` / `86d500565979dca5` | **yes** | yes | – | yes / yes |
| `finn-examples-cnv-w1a1-gtsrb` | 050a1e6e3bf6 (release) | `19d5c56db43d02e3` | yes | `9b7cbbec22623051` / `9b7cbbec22623051` | **yes** | yes | – | yes / yes |

(Full 64-digit hashes in the development log (not published).)

**Result: 8 / 8 pass.** The 3 FINN / QONNX networks reproduce their published files exactly (data and graph). The 5
Brevitas networks reproduce the data section exactly (every weight bit, integer threshold, input affine, quantizer and
final affine constant, in the same order and layout) and a graph that differs from the published one **only in names**;
their outputs are bit-identical on the check set, in the numpy executor and in the C++ runtime.

### Why the Brevitas graphs differ (names only)

`tools/convert_brevitas.py` names IR nodes after the PyTorch module paths and uses some ad-hoc names:
`input_2x_minus_1` with constants `input/scale2` / `input/shift-1`, the FakeQuant `conv_features.0`, layers
`conv_features.1`, `conv_features.4`, ..., `linear_features.0`, pools `conv_features.7`, the flatten `flatten`, the FC input
threshold `features.0/threshold`. The importer's default naming (`--naming index`, the convention of
`adapters.finn_bnn`) gives `input_affine`, `input_quant`, `conv0`..`conv5`, `dense6`.., `conv1/pool`, `dense6/flatten`,
`flatten_input`, `input/threshold`. The module paths are not recoverable from a cleaned QONNX graph
(`GiveUniqueNodeNames` renames every node, e.g. `MatMul_0`), and `--naming onnx` (ONNX node names) does not give them
either. Reproducing them would mean a naming table keyed on the legacy converter, which the importer does not have.
Everything else is identical: the positional-name graphs match (same ops, order, attributes, shapes, dtypes, layouts,
offsets, wiring). A side effect that shows the importer is independent of the source framework: the re-imported Brevitas
CNV / TFC graphs are byte-identical to the published **FINN** CNV / TFC graphs (`22ae302d...`, `86d50056...`), the same
topology exported by a different tool.

## Verification of each re-import (Level 3 + C++)

For all 8: semantic check IR fp64 vs QONNX fp64 within 1e-6 on 128/128 samples (max |Δlogit| 3.4e-8 .. 1.5e-7, the fp32
storage of the final affine), top-1 agreement 1.000, **0 samples with a binarization flip** against the source (every
binarizer of the graph compared, fp64 and fp32), source fp32 vs fp64 top-1 disagreements 0, acceptance rule of
`numerical-correctness.md` passed, C++ runtime bit-exact with the numpy executor at stages `reference` and `optimized`.
Brevitas export check: the source model (Brevitas, fp32) vs the exported QONNX graph (qonnx `execute_onnx`, fp32) is
**bit-identical** on 128/128 samples for all 5; in fp64 they differ by ≤ 1.0e-7, which is exactly the effect of
`qonnx-cleanup` folding `TensorNorm`'s `running_var + eps` in float32 (raw vs cleaned graph in fp64: the same numbers).

Robustness checks done on TFC (same data SHA-256 each time): torch's default constant folding at export instead of
`do_constant_folding=False`, no qonnx cleanup, `--input-domain float32` instead of `unit8`.

## Tests

`.cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_import.py` → **22 passed**. They cover each binarizer
and its sign(0) convention (BipolarQuant `ge` with exact-zero inputs, ONNX `Sign` → `sign_zero`, `Where(GreaterOrEqual)` /
`Where(Greater)` with inputs exactly at the threshold, MultiThreshold BIPOLAR), that `Sign` and `BipolarQuant` stay
different on zero inputs, per-channel ±α accepted and folded, a channel that is not ±α rejected (never rounded), the pad
rules (no pad, Conv zero pad → `pad_value 0`, `Pad(+1)` / `Pad(-1)` → `pad_value ±1`, Pad + Conv padding rejected),
IntQuant → FakeQuant + W1A8 Conv2D + integer thresholds (file round trip), bit outputs between binary convs, a non-ROUND
rounding mode rejected, the exact input threshold 0.5 at the float32 neighbours, the MultiThreshold input quantizer on the
uint8 domain (and its rejection without a discrete domain), the float set (Conv2D + BN + Relu + residual Add + MaxPool,
NCHW flatten through a Transpose, Gemm → Dense), the trailing TopK, the precise error of an unmatched node (name, op,
reason; also for an op the evaluator cannot run), and `OnnxF64` against onnxruntime. Existing tests touched by the CLI
change (`test_zoo_cli.py`, `test_finn_bnn.py`, `test_ir_and_format.py`) still pass.

## Findings and deviations

* **`export_qonnx` does not run in `.cache/venv-brevitas` as installed:** brevitas' ONNX managers require
  `onnxoptimizer`, which the env does not have (and installing it was not allowed). `tools/export_brevitas_qonnx.py`
  supplies a local implementation of exactly the two passes the QONNX manager asks for
  (`extract_constant_to_initializer`, `eliminate_unused_initializer`; anything else raises) and records that in
  `export.json`. The exports are deterministic (same ONNX SHA-256 on repeated runs).
* The guide's `python -I -m bitino.qonnx_import` does not work with the repo's venvs (`bitino` is not installed in
  them; `-I` ignores `PYTHONPATH`); use `tools/import_qonnx.py` or `bitino import-qonnx`. `converting-a-new-bnn.md`
  still marks step 2 "(planned)" and was not edited.
* The importer writes the fused file only (`model.bitino`); the legacy converters also write `model_unfused.bitino`.

## Remaining gaps vs phase 1 of qonnx-frontend.md

Implemented and exercised by the 8 networks: BConv2D, BDense, Conv2D (W1A8), FakeQuant, ChannelAffine (input path),
MaxPool2D, Reshape, BN → integer thresholds, bit outputs. Implemented and tested only on synthetic graphs: `Sign`,
`Where(GreaterOrEqual/Greater)`, per-channel ±α, Conv padding / explicit `Pad(±1)`, float Conv2D / Dense (Gemm, MatMul),
Relu / Clip / PRelu / LeakyRelu / Sigmoid / Tanh / HardSwish / Gelu(opset 20) → Activation, Add / AddBroadcast, AvgPool2D,
GlobalAvgPool, Concat (channels), Softmax, Pad (float), Transpose before an NCHW flatten. Not implemented in phase 1
(each stops the import with a precise message):
* `QBConv2D` / `QBDense` (k-bit activations into binary weights): IntQuant before a binary-weight conv becomes the exact
  W1A8 Conv2D only for a power-of-two scale followed by a binarizer, else a float Conv2D; no QB* emission yet.
* ternary weights (`IntQuant(W, 2 bit, narrow)` → `w_zero`), `ChannelGate`, runtime `Mul`, `MatMul` / `BMatMul` of two
  runtime tensors, `ReduceMean`, `LayerNorm`, `Gather` (phases 2-3), the Erf decomposition of GELU, `HardSigmoid`.
* `Sign` directly on a binary accumulator + affine (exact ternary integer threshold), a binarizer of a float tensor
  scaled by a negative factor (would need input-channel polarity flips), a multi-level MultiThreshold anywhere but the
  input path, IntQuant with a non-zero zero point, ceil-mode pooling, flatten of a binarized tensor with a per-channel
  threshold, rank-3 tensors.
* The `ai.bitino` custom domain (phase 3).
* Graph names of the legacy Brevitas converter (see above): reproducible only with a naming table keyed on that
  converter; not added.
