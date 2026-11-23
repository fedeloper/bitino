# QONNX route, step 4: front ends without an ONNX exporter

Plan: [qonnx-frontend.md § Plan](qonnx-frontend.md#plan), step 4. Date: 2026-11-21, host only (x86-64), nothing under
`artifacts/` was written, registry hashes untouched. Machine-readable results: the development log (not published).
Canonical QONNX files, re-imported files and reports: `.cache/qonnx_step4/<id>/<file stem>/`. Command (the run was
split in shards of 2–4 GB (tensorpack 4 GB), the regression re-check at 7 GB, `--out` one JSON per shard, merged):

```bash
scripts/capped.sh 3G nice -n 10 env OMP_NUM_THREADS=3 .cache/venv-qonnx/bin/python -I tools/qonnx_step4_acceptance.py [--only ID ...] [--regress N]
```

**Identity bar** (design decision 2026-11-20, Q11): the re-imported `.bitino` has the published file's data SHA-256 and
graph SHA-256 (= `registry/recipes.json` → `expected` for `model.bitino`), every layer, node and constant name byte for
byte; the manifest is not compared. **Level 3** is the numerical rule.

**Result: all 21 networks, 22 of 22 published float files, rebuild from their canonical QONNX files with the published
data and graph SHA-256** (the 21 `model.bitino` = the recipes' expected hashes, plus FDA-BNN's
`model_sign_zero.bitino`), every canonical file passes `check-qonnx` with no error and no warning, and **all 22 are fully verified**: outputs bit-identical with the published file in the numpy executor and both C++ stages on the model's check set (128 samples), the C++ reference stage bit-exact with numpy, Level 3 passed.
Nothing needed an operator outside standard ONNX + QONNX; no network was stopped. **Every file determines its
`.bitino` alone**: the import layout is model metadata (`bitino.import`, mapping `patterns`), no import argument is
needed.

## Scope

Every recipe whose converter has no usable ONNX exporter path and is neither a PyTorch CNN (steps 1 / 1b), Keras /
Larq (step 2) nor a transformer / image-feature network (step 3), from `registry/recipes.json` by converter, plus the
other leftover non-step-3 front end found: the FINN / Brevitas networks. Their phase-1 files (`.cache/qonnx_phase1`)
were never canonical (no metadata, `BatchNormalization`, `MultiThreshold`, `Quant`, `TopK`, IR < 10: profile § 9) and
step 1 left them out (its scope was the IR-emitting PyTorch converters), so they are included here.

| front end (legacy converter, adapter) | networks | published float files | stage-1 environment |
|---|---:|---:|---|
| OpenVINO Open Model Zoo IR (`convert_omz.py`, `openvino_ir`): ResNet-18 XNOR, ResNet-50 binary, FP32-INT1 and FP16-INT1 | 4 | 4 | `.cache/venv-openvino` |
| daBNN model zoo ONNX (`convert_dabnn.py`, `onnx_dabnn`): Bi-Real 18, Bi-Real 18 + stem | 2 | 2 | `.cache/venv-qonnx` |
| NNCF (`convert_nncf.py`, `torch_nncf`): ResNet-18 XNOR / DoReFa binarization | 2 | 2 | `.cache/venv-nncf` |
| MindSpore checkpoint (`convert_fdabnn.py`, `mindspore_ckpt` + `mindspore_fdabnn`): FDA-BNN ResNet-20 | 1 | 2 (`ge`, `sign_zero`) | `.cache/venv-qonnx2` (PyTorch port as the reference) |
| tensorpack TF1 npz (`convert_tensorpack_dorefa.py`, `tensorpack_dorefa`): DoReFa AlexNet W1A1 G32 | 1 | 1 | `.cache/venv-qonnx2` |
| Theano / Lasagne npz (`convert_bnn_pynq_npz.py`, `bnn_pynq_npz` + `finn_bnn`): BNN-PYNQ CNV CIFAR-10 / GTSRB, LFC MNIST | 3 | 3 | `.cache/venv-qonnx` + `.cache/env-theano27` (Theano run) |
| Brevitas (`convert_brevitas.py`, `brevitas_bnn`): BNN-PYNQ CNV / LFC / SFC / TFC, the Aliasgharshinwari CNV retrain | 5 | 5 | `.cache/venv-brevitas` |
| FINN-ONNX / QONNX releases (`convert_finn_onnx.py`, `finn_onnx` + `finn_bnn`): finn-examples CNV, TFC, GTSRB CNV | 3 | 3 | `.cache/venv-qonnx` |
| **total** | **21** | **22** | |

No network of these front ends publishes `model_sign_zero.bitino` except FDA-BNN (MindSpore `Sign(0) = 0`): the
OpenVINO, NNCF, tensorpack and FINN-family binarizers are threshold rules (`x > t`, `x ≥ t`) with no `sign(0)`
ambiguity, daBNN's is the sign-bit packing (`ge`). The int8_full files are step 5.

## What was built

| file | change |
|---|---|
| `tools/qonnx_ir_writer.py` | **the direct writers** (new, numpy only, imported by every front end in its own environment). `IRWriter`: a legacy adapter's IR graph → canonical ONNX spec, one pattern per IR node (step 2's Keras writer generalized: input thresholds, `gt` / `ge` / `sign_zero` binarizers, explicit binary `Pad(±1 / 0)`, float64 carriers from the adapter's own float64 values, latent kernels as `BipolarQuant(W, 1)`, Keras-style NHWC flatten, the attribute JSON forms). `LayerWriter`: the FINN-style layer specs (`finn_bnn.InputSpec` + `Layer`: exact ±1 weights, float64 BN affines) → the source's own form (input quantizer or binarizer, `Conv` / `Gemm` → [MaxPool] → `Mul` / `Add` carriers → `BipolarQuant` → [MaxPool]) |
| `tools/export_ir_qonnx.py` | the step-4 exporter (new): stage 1 per front end (checkpoint SHA-256, the legacy adapter's parse, the source framework's fp32 / fp64 logits), stage 2 in `.cache/venv-qonnx` (onnx assembly, metadata, `check-qonnx` must pass) |
| `python/bitino/qonnx_import.py` | step-4 patterns as separate functions (below), entered from one line each in `binarize()`, `conv()`, `dense()`, `_quant()`, `apply_group_attrs()`, `import_model()` and `main()` |
| `python/bitino/qonnx_profile.py`, `docs/qonnx-profile.md` | `bitino.import` mapping `patterns` with its options, node key `bitino.const_layouts`, the pool-before-BN pattern (§ 4.1), attribute-default extensions (§ 7) |
| `python/bitino/qonnx_groups.py` | one line: `import_mode` leaves `mapping: patterns` to the pattern importer |
| `tools/qonnx_step4_acceptance.py` | this acceptance (step 2's flow, per published file), plus `--regress` (identity re-check of earlier steps' canonical files) |
| `tests/python/test_qonnx_step4.py` | 7 tests on synthetic graphs |

### The canonical files, per front end

| front end | source construct → canonical QONNX (standard ONNX + QONNX only) |
|---|---|
| OpenVINO | `FakeQuantize(levels 2, input_low = input_high = t, out {0, 1})` + `BinaryConvolution(xnor-popcount, pad_value −1, u1 LSB-first, bit 1 = +1)` → `Where(Greater(x, t), 1, −1)` → `Pad(−1)` → `Conv(±1 constants)`; every chain of constant `Multiply` / `Add` / `Subtract` merged as the legacy converter merges it (`openvino_ir.merge_affine_chains`: float64 composition, float32 rounding after each merge; FLOAT constants, the values Bitino stores) → `Mul` / `Add`; `AvgPool` over the full map → `GlobalAveragePool`, its `Reshape` → `Transpose` + `Reshape` (keeps the IR's two nodes); `bitino.import` `bit_outputs: threshold` |
| daBNN | onnx2bnn's implicit input binarization of every all-±1 `Conv` → `BipolarQuant(x, 1)` (the aarch64 sign-bit packing for every value but −0.0) → `Pad(−1)` → `Conv(±1 constants)`; `BatchNormalization` (eps the float32 attribute) folded in float64 → `Mul` / `Add` carriers (the adapter's `float64_constants`); the 1×1 classifier `Conv` on the pooled vector → `Gemm` |
| NNCF | `a = s·[x > t]` with `wb = sign(w > 0)·α`: `Σ a·wb = (α s / 2)(acc + Wsum)` (the adapter's exact decomposition) → `Where(Greater(x, t), 1, −1)` → `Pad(−1)` (a padded tap is the 0 level) → `Conv(BipolarQuant(w, 1))` → `Mul(α s / 2)` → `Add(α s Wsum / 2)` (float64 carriers); the threshold `t = fl32(sigmoid · s)` as FLOAT (the value the fp32 source compares with) |
| MindSpore FDA-BNN | `QuanConv` → `Sign(x)` (`sign_zero` file) or `BipolarQuant(x, 1)` (`ge` file) → `Pad(0)` → `Conv(BipolarQuant(W, 1))` → `Mul(mean\|W\| c64)`; PReLU slope broadcast; shortcut `x[:, :, ::2, ::2]` + channel zero pad → `AveragePool(1, stride 2)` + `Pad` |
| tensorpack DoReFa | `round(clip(x, 0, 1)) = [x > 0.5]` → `Where(Greater(x, 0.5), 1, −1)` → `Pad(−1)` → `Conv(BipolarQuant(W, 1), group 2)` → `Mul(E/2)` → `Add(E/2 · Σw [+ bias])` (carriers); the NCHW flatten before `fc0` → `Transpose` + `Reshape` with fc0's rows in (h, w, c) order; BDense kernels untagged (`bitino.const_layouts`) |
| BNN-PYNQ Theano | int8 input `round3((clip(x)+1)·128)/128 − 1` → `Mul(1)` → `Add(0)` → `IntQuant(1/128, 0, 8, signed, ROUND)` (the reader checks half-away = half-even on the input domain); binary weights = Theano's own `quantizeWeights` output (round half to even of `hard_sigmoid(W)`: **±1 constants, not the sign of the latent W**, which differs for tiny weights); conv → MaxPool → BN → sign in that order (pool before BN); BN + biases folded in float64 → carriers before `BipolarQuant` |
| Brevitas | `2x − 1` → `IntQuant(1/128 …)` (CNV) or → `BipolarQuant` (FC); the modules' own `quant_weight()` as ±1 constants; BN (`bn_params`, float64) and the final `TensorNorm` → carriers; MaxPool after the sign |
| FINN releases | the release ONNX rewritten to the profile (Q10): its input quantizer as the reader's verified IR form (`Mul(a) → Add(b) → IntQuant`, checked on all 256 input values against the release's `MultiThreshold` / `Quant`; TFC: `Where(GreaterOrEqual(x, t))`), BN folded → carriers, `MultiThreshold` / `BatchNormalization` / `TopK` removed |

Model metadata (§ 6): all required keys; `bitino.source.checkpoint_sha256` the checkpoint (OpenVINO: `{bin, xml}`);
`bitino.import` `{"mapping": "patterns", "naming": "exporter", "affine": keep | fold, "constant_order": first-use |
walk, "cleanup": "none", "bit_outputs": none | threshold | set}`; `bitino.const_order` where the legacy data order is
not the importer's (the OpenVINO files: creation order with the merged / pruned constants); `bitino.ir_unique_counter`
(OpenVINO: the merges' unique-name counter, continued by the folds and the bit-output fuse). Naming metadata (§ 7):
`bitino.name` / `bitino.order` / `bitino.consts` = the adapter's IR node and constant names, `bitino.attrs` the JSON
forms, `bitino.extra_names` the input `ChannelAffine` before a `FakeQuant`.

### Two layouts, as the legacy converters

* **keep** (OpenVINO, daBNN, NNCF, FDA-BNN, tensorpack: adapter → `passes.optimize(fold=True)`): the file is the
  adapter's graph *before* the passes, one group per IR node; the importer rebuilds that graph node for node and runs
  the same passes (BN folded into the binary epilogues and float convs in float64 of the stored float32 values).
* **fold** (FINN, BNN-PYNQ, Brevitas: the adapters fold BN + sign into integer thresholds themselves, then
  `passes.optimize(fold=False)`): the file carries the source's BN affines as float64 carriers before each binarizer,
  and the importer's fold mode derives the same exact integer thresholds (`brevitas_bnn.integer_threshold`), bit
  outputs (`set_bit_outputs`) and W1A8 first convs as the legacy adapters.

### Importer changes (all on synthetic tests)

* **`bitino.import` mapping `patterns`** (model metadata): the pattern importer's layout options (`naming`, `affine`,
  `constant_order`, `cleanup`, `bit_outputs`, `flatten_dense`) recorded in the file; an import argument that
  contradicts the file is an error (step 3's `{"mapping": "groups"}` is unchanged). The step-4 files need no
  argument.
* **`bit_outputs: threshold`** (OpenVINO): after the passes, `openvino_ir.fuse_threshold_into_bit_output` (a BConv2D
  without epilogue whose consumers all binarize it with `gt` and one threshold emits bits, `post_add = −(⌊t⌋ + 1)`,
  exact); flip tracing follows the fused decision.
* **keep mode, a binarizer of another group's binary conv** (`_step4_keep_lin`): under `affine keep` + exporter
  naming, a binarizer that reads a binary linear operator's output from another naming group binarizes its closed
  float output (both IR nodes stay at their positions) instead of folding into an integer threshold.
* **Pool before BN** (`_step4_pool_threshold`, `_step4_flip_inputs`; profile § 4.1): MaxPool of a binary conv's
  accumulator → BN → `ge` sign becomes the next conv's integer input threshold (`T`, or `T + 1` with the input
  channel's weights negated for a negative BN scale), exact without spatial padding (checked); flip tracing applies
  the polarity.
* **Extra-named input affine** (`_step4_quant_affine`): the `ChannelAffine` before an input `FakeQuant` is named by the
  `IntQuant` group's `bitino.extra_names[0]` and keeps both `Mul` and `Add` (also a scale of 1).
* **`bitino.const_layouts`** (node key): the informational layout tag of a rank-2 bit constant, `OI_bit` or untagged
  (the tensorpack converter's BDense kernels).
* **`bitino.attrs` defaults**: `null` also removes `AvgPool2D` `count_include_pad 0`; an absent attribute may be
  *added* with the operator's default (`Add` `activation "none"`, daBNN).

## Acceptance table

Data / graph SHA-256 of the re-imported file (first 16 hex digits; full hashes in `summary.json`), compared with the
published `artifacts/<id>/<file>` and the recipe's expected hashes (in parentheses, `model.bitino`). Outputs: every
sample of the model's check set (128), numpy executor (fp32) and C++ runtime stages `reference` and `optimized`,
published vs re-imported, bit for bit. **Level 3** (step 1's rule, `FLIP_REL` / `NOFLIP_REL`): IR (numpy, fp64) vs
the QONNX graph in float64 (`OnnxF64`, carriers at their float64 value): samples within 1e-6 · deviating samples whose
first binarization flip is at |pre| ≤ 1e-3 × the median |pre| of that binarizer (largest ratio) · deviating samples
without a flip with max |Δ| ≤ 1e-5 × max |output| (largest ratio). Export check: the source framework in fp32 vs the
canonical file in qonnx `execute_onnx` (top-1 agreement) and in fp64 vs `OnnxF64` (max |Δ|).

**Acceptance of a row** = check-qonnx passes with no warning, data + graph SHA-256 identical (= the recipe), outputs
bit-identical (numpy, C++ reference, C++ optimized), C++ reference bit-exact with numpy (`--require-exact-numpy`),
Level 3 passed.

| network | file | family | check-qonnx | data SHA-256 | graph SHA-256 | = published (recipe) | outputs numpy / C++ ref / C++ opt | Level 3: within · flip (max rel \|pre\|) · no flip (max rel \|Δ\|) | export check fp32 top-1 / fp64 max \|Δ\| | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | model | brevitas | pass | `829c77f6dcf899c5` | `920e0b7263483c0a` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 4.4e-16 | **PASS** |
| `bnn-pynq-cnv-w1a1-cifar10` | model | bnnpynq | pass | `c226b6c7eff56197` | `9aca249e2f8115e9` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 6.7e-16 | **PASS** |
| `bnn-pynq-cnv-w1a1-gtsrb` | model | bnnpynq | pass | `a8158ac8821c13c7` | `7a7f8b8d6bfe6624` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 8.9e-16 | **PASS** |
| `bnn-pynq-lfc-w1a1-mnist` | model | bnnpynq | pass | `8f71f69378aa9618` | `7588343e3efcdd1d` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 8.9e-16 | **PASS** |
| `brevitas-bnn-pynq-cnv-1w1a` | model | brevitas | pass | `d7223bccd797d8cc` | `920e0b7263483c0a` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 2.2e-16 | **PASS** |
| `brevitas-bnn-pynq-lfc-1w1a` | model | brevitas | pass | `12198dfceb7294ed` | `06152823517f4978` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 8.9e-16 | **PASS** |
| `brevitas-bnn-pynq-sfc-1w1a` | model | brevitas | pass | `aa5848d66e0dd551` | `324c7b2789ba64e2` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 4.4e-16 | **PASS** |
| `brevitas-bnn-pynq-tfc-1w1a` | model | brevitas | pass | `3531cef0e4e18773` | `c47d1a9bc2e276c1` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 2.2e-16 | **PASS** |
| `dabnn-bireal18` | model | dabnn | pass | `59a15139520bac8b` | `c4d4777d315f84d5` | yes / yes (yes) | yes / yes / yes | 0 · 4 (2.6e-07) · 124 (1.8e-07) | 0.977 / 1.1e-14 | **PASS** |
| `dabnn-bireal18-stem` | model | dabnn | pass | `cd151e30868279f2` | `43885d3bb9d3a069` | yes / yes (yes) | yes / yes / yes | 0 · 5 (1.4e-07) · 123 (2.2e-07) | 0.961 / 1.1e-14 | **PASS** |
| `fda-bnn-resnet20-cifar10-mindspore` | model | fdabnn | pass | `6f34472c8af160cc` | `516d80ac3825eded` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 1.1e-14 | **PASS** |
| `fda-bnn-resnet20-cifar10-mindspore` | model_sign_zero | fdabnn | pass | `6f34472c8af160cc` | `2726f748d7a0bb48` | yes / yes | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 1.1e-14 | **PASS** |
| `finn-examples-bnn-pynq-cnv-w1a1` | model | finn | pass | `602cbf84cacba194` | `22ae302d08eb2efc` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 4.4e-16 | **PASS** |
| `finn-examples-bnn-pynq-tfc-w1a1` | model | finn | pass | `14c00376ac27547d` | `86d500565979dca5` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 6.7e-16 | **PASS** |
| `finn-examples-cnv-w1a1-gtsrb` | model | finn | pass | `19d5c56db43d02e3` | `9b7cbbec22623051` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 1.000 / 1.8e-15 | **PASS** |
| `nncf-resnet18-binarization-dorefa` | model | nncf | pass | `0d90e07eeeafd5b3` | `46265de5c6a8b583` | yes / yes (yes) | yes / yes / yes | 0 · 5 (2.9e-07) · 123 (2.2e-07) | 0.977 / 1.2 | **PASS** |
| `nncf-resnet18-binarization-xnor` | model | nncf | pass | `911acdfdb4651f79` | `46265de5c6a8b583` | yes / yes (yes) | yes / yes / yes | 0 · 3 (1.5e-07) · 125 (1.8e-07) | 0.984 / 2 | **PASS** |
| `omz-resnet18-xnor-binary-onnx-0001` | model | omz | pass | `552ad64947f41230` | `3f09a56d398c7430` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0.984 / 6.7e-06 | **PASS** |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | model | omz | pass | `b15c813787ac4471` | `b2f9d5790e11a779` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0.992 / 3.3e-06 | **PASS** |
| `omz-resnet50-binary-0001` | model | omz | pass | `9772e363f197d82f` | `2b69b86b7d8e44b1` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0.922 / 4.3 | **PASS** |
| `omz-resnet50-binary-0001-fp16` | model | omz | pass | `c0aa68a1ef2e2559` | `3a3ab86bb477c5f2` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0.945 / 4.3 | **PASS** |
| `tensorpack-dorefa-alexnet-w1a1-g32` | model | tensorpack | pass | `156d8ca9d2b47d34` | `aa64b4fa29e73b35` | yes / yes (yes) | yes / yes / yes | 0 · 0 (-) · 128 (6.2e-08) | 0.992 / 5e-14 | **PASS** |

### Per network

| front end | networks | files | identical (data + graph SHA-256) | verified (outputs bit-identical, Level 3) | status |
|---|---:|---:|---:|---:|---|
| OpenVINO / OMZ | 4 | 4 | 4 / 4 | 4 / 4 | **identical, verified** |
| daBNN | 2 | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| NNCF | 2 | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| MindSpore FDA-BNN | 1 | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| tensorpack DoReFa | 1 | 1 | 1 / 1 | 1 / 1 | **identical, verified** |
| Theano / Lasagne BNN-PYNQ | 3 | 3 | 3 / 3 | 3 / 3 | **identical, verified** |
| Brevitas | 5 | 5 | 5 / 5 | 5 / 5 | **identical, verified** |
| FINN releases | 3 | 3 | 3 / 3 | 3 / 3 | **identical, verified** |
| **total** | **21** | **22** | **22 / 22** | **22 / 22** | nothing not done |

Every network individually: identical and verified (the table above, one row per file). Not done: none.

The export check's float64 column is large (1.2–4.3) for NNCF and the OMZ ResNet-50 and equals the legacy conversion
reports' own numbers (OMZ ResNet-50 fused vs the OpenVINO IR in float64: 4.32; NNCF XNOR / DoReFa: 1.30 / 1.41): the
NNCF float64 source compares with the exact product `sigmoid·s` while the IR (and the canonical file) compares with its
float32 rounding, and the OMZ ResNet-50's binarizations flip on near-ties between the merged and the original affine
chains. Where both sides binarize alike it is at rounding level (OMZ ResNet-18 6.7e-6 / 3.3e-6, the rest ≤ 5e-14).

## Findings

* **Nothing needed a custom operator, and no network was stopped.** Every construct of the 21 networks is standard
  ONNX + the QONNX `BipolarQuant` / `IntQuant`: threshold binarizers as `Where(Greater / GreaterOrEqual)`, the
  OpenVINO / NNCF / tensorpack `{0, 1}` activations through the adapters' exact ±1 decompositions, the FINN input
  quantizers as `IntQuant` (verified on all 256 input values by the reader), `Sign` for MindSpore's `Sign(0) = 0`.
* **The direct writers work from the legacy adapters' parse**, as step 2: one generic IR-graph writer for the five
  converters whose pipeline is adapter → `passes.optimize`, one layer-spec writer for the three FINN-style ones. The
  values in the files are the parse's (the checkpoint tensors, and the adapters' own float64 folds as carriers,
  asserted to round to the IR constants bit for bit), never read from the published files; the legacy graph is used
  for the naming metadata only.
* **Each legacy converter's own pass pipeline is reproduced, not a generic one.** daBNN / NNCF / tensorpack / FDA-BNN:
  `passes.optimize(fold=True)` on the adapter's graph (the importer's own passes). OpenVINO: the adapter-specific
  `merge_affine_chains` (iterated float32 rounding: the file holds the merged constants, the values Bitino stores),
  then the common folds, then `fuse_threshold_into_bit_output` (`bitino.import` `bit_outputs: threshold`), with the
  merges' unique-name counter (`bitino.ir_unique_counter`) and the creation-order data section (`bitino.const_order`).
  FINN / BNN-PYNQ / Brevitas: the adapters' BN + sign → integer-threshold fold, derived by the importer's fold mode
  from the float64 BN affines.
* **The FINN / Brevitas files are now names-identical.** Phase 1 rebuilt the 5 Brevitas files with identical data but
  other node names (index naming); their canonical files carry the module-path names, so all 8 FINN / Brevitas
  networks are now byte-identical in data and graph.
* **Two source quirks are kept exactly.** BNN-PYNQ's Theano weight binarizer rounds `hard_sigmoid(W)` half to even in
  float32, so tiny positive weights are −1: the files hold Theano's own ±1 weights, not `BipolarQuant(W_latent)`.
  NNCF compares the float32 input with `fl32(sigmoid(θ)·s)`: the threshold is a FLOAT constant, not a float64 carrier
  (the IR stores the float32 value; a carrier would make the float64 reference compare with the exact product).
* **The export check (source framework vs canonical file)** is reported per file; the binary networks amplify
  float32 rounding (OpenVINO's CPU plugin, NNCF, tensorpack, Theano run float32 with their own summation orders), as
  in the legacy conversion reports. Level 3 is measured against the canonical file itself, the reference of the
  QONNX route.
* **Disk.** The canonical files are 0.1–244 MB (they carry float ±1 or latent float32 kernels); after verification
  the canonical and re-imported files of each network were deleted to keep the host above 37 GB free (their SHA-256
  are in `summary.json`; the command above rebuilds them in minutes).

## No regressions

The importer changes were re-checked on the canonical files of the earlier steps, re-imported with the step-4 importer (identity only, each with its recorded import arguments; `tools/qonnx_step4_acceptance.py --regress 3 --regress-row ...`, the files under `.cache/qonnx_step1*` / `qonnx_step2` read, not rewritten), against the re-imported hashes of their summaries. The sample covers the rows where the keep-mode binarizer rule could have applied (BinaryFaceNet, XNOR-Net, BinaryAlexNet: BConv2D → BConv2D after the BN fold) and the step-1b patterns:

* **step1:** 5 / 5 identical (`ab-bnn-cifar10-reactnet18:ge`, `bnext-middle:ge`, `recu-resnet18-imagenet:sign_zero`, `bitorch-resnete18-imagenet:ge`, `rbonn-reactnet-resnet18-imagenet:ge`).
* **step1b:** 5 / 5 identical (`bbcu-denoise:ge`, `bbcu-sr-m-x2:sign_zero`, `bnas-a-cifar10:ge`, `adabin-resnet20-cifar10:ge`, `bimatting-bimobilenet-backbone:sign_zero`).
* **step2:** 6 / 6 identical (`brillsson-binarydensenet28`, `larq-zoo-binary-densenet45`, `larq-zoo-quicknet-small`, `larqface-binaryfacenet-msceleb`, `larq-zoo-xnornet`, `larq-zoo-binary-alexnet`).

The same check with a 5-row sample per step ran after the merge of master (step 3) with the identity pass: step 1 5 / 5, step 1b 4 / 4, step 2 5 / 5. Step 3's canonical files are no longer on disk (deleted for space), so step 3 is covered by its synthetic group tests in `test_qonnx_import.py` (all pass); the only step-4 change on its path is that `qonnx_groups.import_mode` leaves `{"mapping": "patterns"}` to the pattern importer. BNext-L is not in the sample (its import needs more than this job's memory budget, as in step 2).

## Design decisions

1. **Import layout in the file.** The step-4 files record their whole import layout in `bitino.import`
   (`{"mapping": "patterns", "naming", "affine", "constant_order", "cleanup", "bit_outputs"}`), following step 3's
   key; an explicit argument that contradicts the file is an error, so each file alone determines its `.bitino`.
   The step 1 / 1b / 2 files still need their recorded import arguments. Re-write them with the same key (no
   value or name changes: identity re-checked), or keep the arguments there?
2. **OpenVINO affine chains merged by the writer.** The legacy converter merges each chain of constant `Multiply` /
   `Add` nodes with a float32 rounding after every pairwise merge; the canonical file writes the merged chain (the
   source's constants composed exactly that way, FLOAT values), so its fp32 evaluation is the Bitino computation
   (B) and differs from the original OpenVINO chain by rounding (in the export check). The alternative is the
   original chain in the file plus an importer option that replays the iterated merge. Accept the merged form?
3. **FINN releases as rewrites (Q10) and their input quantizers in IR form.** The finn-examples files are rewritten
   to the profile; their input quantizer (`MultiThreshold` with 255 thresholds, or `Quant`) is written as the
   reader's `Mul → Add → IntQuant` / `Where(GreaterOrEqual)`, which the reader proves equal on all 256 input values,
   not as the profile's `ReduceSum(GreaterOrEqual(…))` multi-threshold form (no importer pattern for it; it would
   need one). Accept, or require the literal multi-threshold form?
4. **`bitino.const_layouts`** is a new node key for an informational tag (`OI_bit` vs untagged on a rank-2 bit
   constant, the tensorpack converter's BDense kernels). It never changes data or semantics; confirm the key (the
   alternative is no identity for that file).
5. **Writers on the legacy adapters.** As step 2's decision 3: the writers depend on the legacy adapters' parse
   (`openvino_ir`, `onnx_dabnn`, `torch_nncf`, `mindspore_fdabnn`, `tensorpack_dorefa`, `bnn_pynq_npz`, `finn_onnx`,
   `brevitas_bnn`). For step 6 they stay as the writers' parse layer, or the writers absorb them. Which?
6. **FINN / Brevitas in step 4.** They were step-1 / phase-1 networks without canonical files; this step gives them
   canonical files (and graph identity, which phase 1 did not have for Brevitas). Confirm they belong here.

## Tests

`.cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_import.py tests/python/test_qonnx_profile.py
tests/python/test_qonnx_step4.py` → **124 passed** (117 in `test_qonnx_import.py` + `test_qonnx_profile.py`, unchanged by this step, and 7 new in `test_qonnx_step4.py`: the keep layout round trip (daBNN-like: BN carriers, `pad_value -1` in the integer form, `Add.activation` added, `count_include_pad` null), the OpenVINO layout with `bit_outputs: threshold` and `bitino.const_order`, the untagged BDense kernel (`bitino.const_layouts`, a wrong tag rejected by the checker), the FINN-style layer specs with pool before BN and negative BN scales (identity with `finn_bnn.emit`, bit-identical outputs), the Brevitas FC input sign threshold renamed by `bitino.consts`, the polarity rule's padding check, and the file-recorded import options (a contradicting argument and a bad option rejected)).
