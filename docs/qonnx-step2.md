# QONNX route, step 2: acceptance report (Keras / Larq)

Plan: [qonnx-frontend.md § Plan](qonnx-frontend.md#plan), step 2. Date: 2026-11-21, host only (x86-64), nothing under
`artifacts/` was written, registry hashes untouched. Machine-readable results: the development log (not published).
Canonical QONNX files, exports, re-imported files and reports: `.cache/qonnx_step2/<id>/`. Command to reproduce:

```bash
scripts/capped.sh 6G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_step2_acceptance.py [--only ID ...]
```

**Identity bar (design decision 2026-11-20, docs/qonnx-profile.md Q11).** The re-imported `.bitino` has the published
file's data SHA-256 and graph SHA-256 (= `registry/recipes.json` → `expected["model.bitino"]`): every constant byte
in the same order, the same graph JSON, every layer, node and constant name byte for byte. The manifest is not
compared.

**Result: 23 networks, 23 published float files: every file rebuilds from its canonical QONNX file with the published
data and graph SHA-256** (= the recipe's expected hashes), every canonical file passes `check-qonnx` with no error and
no warning, and **all 23 are fully verified**: outputs (every graph output) bit-identical with the published file in
the numpy executor and both C++ stages, the C++ reference stage bit-exact with numpy, Level 3 passed. Nothing is
"not done", and no network needed an operator outside standard ONNX + QONNX.

## Scope

Every recipe of `registry/recipes.json` whose converter reads a Keras / Larq model: the two converters
`tools/convert_larq_zoo.py` and `tools/convert_keras_embedding.py`, both on the adapter
`bitino.adapters.keras_larq` (no other recipe's converter reads a Keras / Larq model: the AB-BNN `.h5` files are
PyTorch checkpoints, the tensorpack DoReFa AlexNet is a TF1 checkpoint read as npz with numpy, step 4). **23 networks, 23 published float files.** Larq's `ste_sign` maps 0 to +1,
which is the source semantics, so each network has one float file, `model.bitino` (no `model_sign_zero.bitino` is
published for any of them). The int8_full files are step 5.

| family (converter) | networks | constructs |
|---|---:|---|
| Larq Zoo (`convert_larq_zoo.py`): QuickNet / -Small / -Large, QuickNet-XL (legacy larq/zoo @40f4cdc), BinaryResNetE18, BiRealNet, BinaryDenseNet28 / 37 / 37-dilated / 45, MeliusNet22, BinaryAlexNet, XNOR-Net, DoReFaNet, Real-to-Binary Net | 15 | `ste_sign`, `MagnitudeAwareSign`, XNOR scale, DoReFa 2-bit activations (QBConv2D / QBDense), `pad_values = 1`, fused ReLU, grouped split convs (MeliusNet), channel-pad merge, SE / data-dependent gates (QuickNet-XL, Real-to-Binary), PReLU, two outputs (logits, probabilities) |
| BRILLsson (`convert_keras_embedding.py --family brillsson`): QuickNet, BinaryDenseNet28, MeliusNet22 audio encoders | 3 | the Larq Zoo backbones on log-mel input `[98, 64, 1]`, GlobalMaxPool head, two outputs (projection embedding, pooled features) |
| LarqFace (`--family larqface`): QuickNet, BinaryDenseNet28 / 37 / 45, BinaryFaceNet (MS-Celeb) | 5 | BinaryFaceNet's DoReFa(k = 1, weights) input quantizer (`gt` on `x − 2⁻²⁴·tanh(max|x|)`), binary depthwise and folded 1×1 separable convs, a sigmoid attention gate, unscaled binary convs with a bias |

## What was built

| file | change |
|---|---|
| `tools/export_keras_qonnx.py` | **the direct writer** (new). Stage 1 (`.cache/venv-larq`, TensorFlow 2.15.1 / larq 0.13.3 / larq_zoo 2.4.0) builds the Keras model exactly as the legacy converter (its own `zoo_factory` / `build_larqface` / `build_brillsson`, with their checkpoint, h5 layer-order and nested-vs-flat checks), writes the source model's fp32 / fp64 outputs, parses it with the legacy adapter (every layer and quantizer interpretation checked against the live layer there) and writes the ONNX graph of that parse with the values taken from the Keras layers; stage 2 (`.cache/venv-qonnx`, onnx) assembles the model and refuses a file that does not pass check-qonnx. No tf2onnx |
| `python/bitino/qonnx_import.py` | step-2 patterns and metadata (below); all phase-1 / step-1 behaviour unchanged |
| `docs/qonnx-profile.md` | § 7: `bitino.attrs` value `null` (attribute absent = its default) and `bitino.const_order` is the final data order |
| `tools/qonnx_step2_acceptance.py` | this acceptance run |
| `tests/python/test_qonnx_import.py` | 7 new tests (42 in the file) |

### The canonical file

| construct (adapter IR op) | canonical QONNX (standard ONNX + QONNX only) | profile rule |
|---|---|---|
| larq sign of a binary conv's input (`BConv2D ge`) | `BipolarQuant(x, 1)` → [`Pad(·, pads, v)`, `v` = `pad_values` ∈ {0, 1}] → `Conv(·, BipolarQuant(W_latent, 1), pads 0)` → [`Mul(alpha)`] → [`Add(bias)`] | R1, R2 (explicit Pad), R4 (±1 weights + `Mul(alpha)`; the Keras latent kernel, checked: its larq sign is the adapter's bits) |
| `MagnitudeAwareSign` / XNOR scale | `Mul(alpha[1, O, 1, 1])` in the conv's group: a FLOAT constant (`mean|W|` is a float32 reduction in larq and in the adapter) / a float64 carrier (`mean|clip(W, −1, 1)|` in float64) | R4, R7 |
| BatchNorm | `Mul(x, Cast(s64))` → `Add(·, Cast(t64))`, `s64 = γ / √(σ² + ε)`, `t64 = β − fl32(μ·γ)·(1/√(σ² + ε))`: exactly the adapter's float64 expression before its float32 rounding (checked bit for bit against the adapter's constants) | R6, R7 |
| DoReFa k-bit activations (`QBConv2D` / `QBDense`) | `Round(Mul(Clip(x, 0, 1), n))` → `Conv` / `Gemm(·, BipolarQuant(W_latent, 1))` → `Mul(mean|W| / n)` (float64 carrier) [→ `Add(bias)`] | § 4.1 QBConv2D (B) |
| BinaryFaceNet DoReFa(k = 1, weights) input quantizer (`BConv2D gt`) | the adapter's operand `x − 2⁻²⁴·tanh(max|x|)` node for node (`MaxPool` over H × W, `Mul(−1)`, `MaxPool`, `Concat`, `Reshape`, `MaxPool`, `Tanh`, `Mul(−2⁻²⁴)`, `Add`), then `Where(Greater(·, 0), 1, −1)`; the kernels are the live quantizer's ±1 values (constants: the quantizer is not the sign of the latent kernel) | R1 (`gt`), R4 |
| float conv / dense, ±1 float stem (BinaryAlexNet) | `Conv` / `Gemm(transB 1)`; the stem's ±1 weights as `BipolarQuant(W_latent, 1)` | § 4.2 |
| Keras flatten / reshape (channels last) | `Transpose(0, 2, 3, 1)` → `Reshape`; a plain `Reshape` where the NCHW and NHWC element orders coincide (C = 1 or H·W = 1) | R3 |
| SE / data-dependent gate (`ChannelGate`) | `Gemm` → `Reshape [N, C, 1, 1]` → `Sigmoid` → `Mul(x, ·)`; the BinaryFaceNet gate `Conv` → `Sigmoid` → `Mul` | § 4.3 ChannelGate |
| fused ReLU (`activation="relu"` of a conv / dense) | a `Relu` node in a group of its own (`<layer>/relu`), folded back by the importer's existing pass | |
| pools, `Concat`, `Add`, channel / spatial `Pad`, `PReLU`, `Clip` (`clip_by_value_activation`), `Tanh`, `Softmax` | themselves, explicit pads, `count_include_pad 0` | § 4.3, § 4.4 |

Model metadata (§ 6): profile, model id, variant `float`, layout, input domain `float32`, preprocessing key +
definition (the eval set's), labels (`imagenet-1k`, `none` for the embeddings), framework `keras+larq` and the TF / larq
/ larq_zoo versions, checkpoint SHA-256, weights licence (registry), exporter, source code (`repo@commit` from the
recipe where it has one), notes. Naming metadata (§ 7): `bitino.name` = the adapter's IR node name (Keras layer
names, the adapter's `…/pre_sigmoid`, `…/dorefa_*`, `…/window` names), `bitino.order` = 2 × its position (2 × + 1 for a
ReLU group), `bitino.consts` = the IR name of every constant input, `bitino.attrs` = `pad_value`'s JSON form (binary
convs) or `null` (float convs: the Larq converters write no `pad_value`), model `bitino.const_order` only for
BinaryFaceNet.

**check-qonnx:** every file passes with no error and no warning (recorded per file in `summary.json` →
`check_qonnx`).

### Importer changes (all on synthetic tests)

* **DoReFa levels → QBConv2D / QBDense**: `Clip(x, 0, 1)` → `Mul(n)` → `Round` (n = 2ᵏ − 1) feeding a Conv / Gemm with
  ±1 weights; the group's `Mul` is post_mul (always present, as the converter's), the IR input is `x`.
* **Keras flatten**: `Transpose(0, 2, 3, 1)` emits nothing (the IR tensor already is NHWC) and the following
  `Reshape` / `Flatten` is an IR `Reshape` of the NHWC tensor; a 4-D → 4-D `Reshape` whose NCHW and NHWC element orders
  coincide is an IR `Reshape` with the NHWC target. A Transpose that feeds anything else is an import error.
* **Gate reshape**: `Reshape [N, C] → [N, C, 1, 1]` before a gate `Sigmoid` emits nothing; `ChannelGate` reads the
  `[N, C]` gate (the converter's layout). Any other consumer of such a tensor is an import error.
* **Several graph outputs** → the same IR outputs in order (verification follows the first: logits / embedding).
* **Unscaled binary operator with a bias**: a BConv2D / BDense with alpha 1, activation scale 1 and no `Mul` in its
  group writes the bias as post_add and leaves post_mul empty (before: a post_mul of ones). Only BinaryFaceNet's
  binary convs have this form; step-1 / phase-1 files are unaffected (below).
* **`bitino.attrs` null**: removes an attribute whose importer value is the operator default (`pad_value 0`); any other
  value is still an import error. **`bitino.const_order`**: the final data order, applied after the passes; the set
  must equal the constants built. **`bitino.consts`** renames are applied all at once (two constants may exchange the
  importer's default names).
* **`--bit-outputs none`** (import argument, recorded in `export.json` → `import_args`): no `set_bit_outputs`. The
  Larq converters never set bit outputs; BinaryAlexNet and XNOR-Net have BConv2D → BConv2D chains after the BN fold,
  where the default would change 4 / 3 nodes.

Import arguments of every step-2 file: `--naming exporter --constant-order first-use --affine keep --cleanup none
--bit-outputs none`.

## Acceptance table

Data / graph SHA-256 of the re-imported file (first 16 hex digits; full hashes in `summary.json`), compared with the
published `artifacts/<id>/model.bitino` and the recipe's expected hashes (in parentheses). Outputs: every sample of
the model's check set (128: `eval128` ImageNet, `eval128_lfw`, `eval128_speechcommands`), all graph outputs (count in
parentheses: Larq Zoo logits + probabilities, BRILLsson embedding + pooled features, LarqFace embedding), numpy
executor (fp32) and C++ runtime stages `reference` and `optimized`, published vs re-imported, bit for bit.
**Level 3** (the step-1 rule, `tools/qonnx_step1_acceptance.py` FLIP_REL / NOFLIP_REL, on the first output): IR
(numpy, fp64) vs the QONNX graph in float64 (`OnnxF64`, carriers at their float64 value): samples within 1e-6 ·
samples whose first binarization flip (vs the QONNX fp64 graph) is at |pre| ≤ 1e-3 × the median |pre| of that
binarizer (largest ratio) · samples without a flip and max |Δ| ≤ 1e-5 × max |output| (largest ratio). Export check:
the Keras model (unmodified, fp32) vs the canonical file in qonnx `execute_onnx` (fp32): top-1 agreement (Larq Zoo) or
minimum cosine (embeddings); Keras fp64 vs `OnnxF64` fp64: max |Δ|.

**Acceptance of a row** = check-qonnx passes with no warning, data + graph SHA-256 identical (= the recipe), outputs
bit-identical (numpy, C++ reference, C++ optimized), C++ reference bit-exact with numpy (`--require-exact-numpy`),
Level 3 passed.

| network | check-qonnx | data SHA-256 | graph SHA-256 | data / graph = published (recipe) | outputs bit-identical numpy / C++ ref / C++ opt (outputs) | Level 3: ≤ 1e-6 · flip (max \|pre\|/typical) · no flip (max \|Δ\|/\|out\|) | export check fp32 top-1 or cos / fp64 max \|Δ\| | status |
|---|---|---|---|---|---|---|---|---|
| `brillsson-binarydensenet28` | pass | `399860f025f41e41` | `44fecf62800a7885` | yes / yes (yes) | yes / yes / yes (2) | 47 · 4 (5.9e-07) · 77 (3.2e-08) | cos 0.99934 / 0.63 | **PASS** |
| `brillsson-meliusnet22` | pass | `f1283652a2f22dc8` | `ce218357492d7b8c` | yes / yes (yes) | yes / yes / yes (2) | 123 · 5 (3.8e-07) · 0 (-) | cos 0.99950 / 0.49 | **PASS** |
| `brillsson-quicknet` | pass | `021ce1d5f2a109e1` | `fd0eac90bffba6c0` | yes / yes (yes) | yes / yes / yes (2) | 1 · 1 (2.1e-07) · 126 (4.5e-08) | cos 0.99980 / 5.4e-07 | **PASS** |
| `larq-zoo-binary-alexnet` | pass | `2232e146633b6082` | `5e2ade63e3b3f929` | yes / yes (yes) | yes / yes / yes (2) | 128 · 0 (-) · 0 (-) | 1.000 / 0 | **PASS** |
| `larq-zoo-binary-densenet28` | pass | `f5e628281f976342` | `bfbbd63590e2cd17` | yes / yes (yes) | yes / yes / yes (2) | 0 · 7 (3.2e-07) · 121 (6.6e-08) | 0.984 / 4.9 | **PASS** |
| `larq-zoo-binary-densenet37` | pass | `2dde4d2048c38c3c` | `4f608bec898eb9a2` | yes / yes (yes) | yes / yes / yes (2) | 1 · 11 (1.9e-06) · 116 (4.2e-08) | 0.969 / 2.6 | **PASS** |
| `larq-zoo-binary-densenet37-dilated` | pass | `9f966d4b22fa3244` | `edacaa0b189dba90` | yes / yes (yes) | yes / yes / yes (2) | 0 · 59 (1.3e-05) · 69 (2.5e-07) | 0.984 / 1.9 | **PASS** |
| `larq-zoo-binary-densenet45` | pass | `0892a677236fa6b5` | `bd3c229da9633c30` | yes / yes (yes) | yes / yes / yes (2) | 0 · 19 (1.9e-06) · 109 (8.0e-08) | 0.961 / 3.3 | **PASS** |
| `larq-zoo-binary-resnet-e18` | pass | `1705b13a82aa1b3d` | `e8a85ef7a5ed9da4` | yes / yes (yes) | yes / yes / yes (2) | 2 · 5 (2.9e-07) · 121 (7.5e-08) | 0.984 / 2.8 | **PASS** |
| `larq-zoo-birealnet` | pass | `4160cf568d937c9e` | `958d4265766604c6` | yes / yes (yes) | yes / yes / yes (2) | 0 · 5 (2.4e-07) · 123 (4.3e-07) | 0.969 / 3.1 | **PASS** |
| `larq-zoo-dorefanet` | pass | `15076b42a514ecd7` | `c9629b80d35f107f` | yes / yes (yes) | yes / yes / yes (2) | 100 · 0 (-) · 28 (1.6e-08) | 0.969 / 5.7e-14 | **PASS** |
| `larq-zoo-meliusnet22` | pass | `1a65fa2286c6aaad` | `fccc078e1ea4574d` | yes / yes (yes) | yes / yes / yes (2) | 0 · 14 (6.9e-07) · 114 (6.1e-08) | 0.977 / 10 | **PASS** |
| `larq-zoo-quicknet` | pass | `91139ed28107cf41` | `acd0c57fcaf29aa1` | yes / yes (yes) | yes / yes / yes (2) | 112 · 7 (2.3e-07) · 9 (5.6e-08) | 0.977 / 1.7 | **PASS** |
| `larq-zoo-quicknet-large` | pass | `cbb4533ac6327f2d` | `25afd44519904ad3` | yes / yes (yes) | yes / yes / yes (2) | 113 · 15 (4.4e-07) · 0 (-) | 0.953 / 3.1 | **PASS** |
| `larq-zoo-quicknet-small` | pass | `4fc3f4c1c5c553cd` | `00ad15ff932deb4a` | yes / yes (yes) | yes / yes / yes (2) | 85 · 3 (3.2e-06) · 40 (9.9e-08) | 1.000 / 3e-07 | **PASS** |
| `larq-zoo-quicknet-xl-legacy` | pass | `d7bd798ce336449a` | `922f596950055a84` | yes / yes (yes) | yes / yes / yes (2) | 123 · 5 (1.0e-07) · 0 (-) | 0.992 / 3.1 | **PASS** |
| `larq-zoo-real-to-binary-net` | pass | `02549d58fc7649c0` | `ae950704fbdea7d0` | yes / yes (yes) | yes / yes / yes (2) | 0 · 7 (3.5e-07) · 121 (2.6e-07) | 0.992 / 2.4 | **PASS** |
| `larq-zoo-xnornet` | pass | `56aaf0d1b44dcce2` | `e98edd168c9f9a73` | yes / yes (yes) | yes / yes / yes (2) | 59 · 0 (-) · 69 (6.2e-08) | 1.000 / 2.5e-14 | **PASS** |
| `larqface-binarydensenet28-msceleb` | pass | `558ff1cb8afdd699` | `86c36b0c2222dad2` | yes / yes (yes) | yes / yes / yes (1) | 126 · 2 (2.8e-08) · 0 (-) | cos 0.99686 / 0.12 | **PASS** |
| `larqface-binarydensenet37-msceleb` | pass | `1432fe7f183c5f6c` | `09ccbff8a2500cba` | yes / yes (yes) | yes / yes / yes (1) | 128 · 0 (-) · 0 (-) | cos 0.98809 / 6.8e-08 | **PASS** |
| `larqface-binarydensenet45-msceleb` | pass | `8f1e36f0acfc168a` | `8ca700953a1032b7` | yes / yes (yes) | yes / yes / yes (1) | 127 · 1 (7.3e-08) · 0 (-) | cos 0.99449 / 0.15 | **PASS** |
| `larqface-binaryfacenet-msceleb` | pass | `8c7586372e6bafd9` | `bfb30307edc98765` | yes / yes (yes) | yes / yes / yes (1) | 126 · 2 (7.8e-08) · 0 (-) | cos 0.99886 / 2.4e-07 | **PASS** |
| `larqface-quicknet-msceleb` | pass | `591f66ea35846119` | `01d1129aeddfc282` | yes / yes (yes) | yes / yes / yes (1) | 124 · 4 (2.5e-07) · 0 (-) | cos 0.99881 / 9.1e-08 | **PASS** |

The export check's float64 column is large (0.5–10) for the networks whose binarizations the float64 Keras model
decides differently at rounding-level pre-activations (Keras folds BatchNorm in its own order; the legacy conversion
reports show the same numbers for these files); where Keras and the canonical file binarize alike it is at rounding
level (5.4e-7 BRILLsson QuickNet, 5.7e-14 DoReFaNet, 2.5e-14 XNOR-Net, 0 BinaryAlexNet). Level 3 is measured against
the canonical file itself, the reference of the QONNX route, as in step 1.

## Findings

* **Nothing needed a custom operator.** Every construct of the 23 networks is standard ONNX + the QONNX
  `BipolarQuant`; no network was stopped for an inexpressible operator. The two non-sign quantizers are written
  exactly in standard operators: DoReFa's `round(clip(x, 0, 1)·n)` (`Clip`, `Mul`, `Round`: half to even, as
  `tf.round`), and BinaryFaceNet's DoReFa(k = 1, weights) input quantizer as the adapter's exact float32 rule
  `x > 2⁻²⁴·tanh(max|x|)` (`MaxPool`, `Mul`, `Concat`, `Reshape`, `Tanh`, `Add`, `Greater`, `Where`).
* **The direct writer works from the adapter's parse.** The writer walks the legacy adapter's IR graph (before the
  passes) and emits, per IR node, its canonical pattern with the values read again from the Keras layers. The
  adapter is where every Keras layer and quantizer interpretation is checked against the live layer (sign
  conventions on probe values, kernel quantizers on the kernels, Lambdas and the separable / depthwise folds end to
  end), so the canonical file inherits those checks instead of a second, divergent interpretation. The values that
  enter the file are the Keras parameters and the adapter's own float64 expressions (asserted to round to the
  published constants bit for bit), not the published file.
* **Byte identity holds through the profile with three layout facts of the Larq converters**, each carried by the
  import arguments or the naming metadata, never by a value: float `Conv2D` nodes have no `pad_value` (the PyTorch
  converters write `0.0`; `bitino.attrs` null), BConv2D outputs are never bit outputs (`--bit-outputs none`; the
  importer's default would change 4 BinaryAlexNet and 3 XNOR-Net nodes), and BinaryFaceNet's data section is not in
  first-use order (`bitino.const_order`). A fused ReLU is a separate `Relu` group folded back by the existing pass, so
  the importer needed no Keras-specific fusion.
* **The source's own float32 noise is visible in the export check** (Keras fp32 vs QONNX fp32, column "export check"):
  the binary networks amplify float32 rounding; the float64 Keras model vs the float64 QONNX graph agree to
  rounding level except where a binarization flips at a |pre-activation| of rounding size (the same flips the
  legacy conversion reports list), and Level 3 traces every deviating sample to such a flip.
* **BinaryFaceNet's kernels are written as ±1 constants**, not `BipolarQuant(W_latent, 1)`: its DoReFa(k = 1,
  weights) kernel quantizer is `x > 2⁻²⁴·max|tanh W|`, not the sign of the latent kernel, so the latent kernel with
  a sign binarizer would be a different function for weights in `(0, 2⁻²⁴·m]`. The live quantizer's ±1 output (what
  the adapter stores, checked to be exactly ±1) is the canonical weight. The same holds for its folded 1×1
  separable convs (products of two ±1 kernels). All other binary layers carry their latent kernel.
* **The importer's Level-4 check does not apply to the embedding models** (LarqFace, BRILLsson: top-1 of an
  embedding is meaningless; tools/validate_model.py's classification acceptance reports `False` for BinaryFaceNet's
  C++ stage while its outputs are bit-exact with numpy). The acceptance uses Level 3 and the bit-exact comparison,
  as step 1.

## Running it

The acceptance runs every Keras / Larq recipe (`networks()`: the registry's recipes with one of the two converters);
`--only` selects, `--no-verify` checks identity only (export + check-qonnx + import, 20–40 s per network),
`--no-export` re-imports the existing `model.onnx` (no TensorFlow run), `--skip-run` only compares. `--cache-dir`,
`--artifacts-dir`, `--bench` point at the checkout's `.cache`, `artifacts` (read only) and the host `bitino_bench`.
This run: one identity pass with the exports (all 23, 7 GB cap, 13 min), then the verified pass with `--no-export`
in shards (`--out` per shard, merged into the development log (not published)): Larq Zoo / BRILLsson at 6 GB, LarqFace
and five Zoo networks at 2 GB, XNOR-Net again at 6 GB (its verification exceeded 2 GB). The verified pass takes 3–35
min per network on the shared host (qonnx `execute_onnx` per node and sample dominates). Disk: 1.8 GB under
`.cache/qonnx_step2` (the canonical files are 2–250 MB: they carry the latent fp32 kernels).

## No regressions

The importer changes were re-checked on the canonical / source files of the earlier steps, re-imported with the
step-2 importer (identity only, each with its recorded import arguments; the files under `.cache/qonnx_phase1`,
`.cache/qonnx_step1` were read, not rewritten):

* **Phase 1:** the 8 Brevitas / FINN files (`model.onnx` / `source.onnx`, `--input-domain` as recorded) give the data
  and graph SHA-256 of the development log (not published) → `reimported`: **8 / 8**.
* **Step 1 (includes the 4 phase-2 networks):** **81 / 83** published float files re-import with the data and graph
  SHA-256 of the development log (not published) → `reimported` (every hook: IR-Net family, AB-BNN, ReActNet-A and its
  BN flavour, RBONN / SNN / IDa-Det, SA-BNN, BNext-T / S / M, bitorch). BNext-L's two files were not re-run: its
  import needs more than this job's 8 GB memory budget (step 1: > 9 GB with verification). It uses the same hook and
  patterns as BNext-T / S / M, whose 6 files re-import identically; not re-run, so not claimed.
* `tests/python/test_qonnx_import.py` and `test_qonnx_profile.py`: 97 passed (below).

## Design decisions

1. **Import arguments vs file metadata.** As in step 1, part of the layout is an import argument recorded in
   `export.json` (`--bit-outputs none` is new; the others are step 1's). The canonical file alone does not fix it.
   Keep the arguments (the recipe will carry them in step 6), or move them into model metadata (e.g.
   `bitino.import = {"bit_outputs": "none", ...}`) so that a file determines its `.bitino` without arguments?
2. **`bitino.attrs` null** (absent = the operator default) is a small extension of § 7 (a), applied in the profile
   text. It is how the float `Conv2D` nodes of every Larq file carry "no `pad_value`". Confirm, or prefer the
   importer to drop default-valued attributes by an import argument?
3. **The writer depends on the legacy adapter's parse** (`bitino.adapters.keras_larq.convert`): the canonical file
   is written from the adapter's IR graph, with the values re-read from the Keras layers. For step 6 ("retire the
   IR-emitting converters") this means the adapter stays as the writer's parse layer for Keras, or the writer is
   rewritten to walk the Keras layers itself (the adapter's checks would move with it). Which?

## Tests

`.cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_import.py tests/python/test_qonnx_profile.py` →
**97 passed** (42 in `test_qonnx_import.py`, 35 of phases 1–2 / step 1 and 7 new; 55 in `test_qonnx_profile.py`). New:
the Keras head (Transpose + Reshape flatten, two outputs, `pad_value` null), null on a non-default value rejected,
`bitino.const_order` (applied; a wrong set rejected), the SE gate reshape → ChannelGate, DoReFa → QBConv2D / QBDense
(values vs the float64 reference), the unscaled binary conv with a bias and `--bit-outputs`, simultaneous consts
renames.
