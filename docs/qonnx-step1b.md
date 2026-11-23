# QONNX route, step 1b: the 16 PyTorch networks step 1 left on legacy converters

Plan: [qonnx-frontend.md § Plan](qonnx-frontend.md#plan), step 1 (remainder; [qonnx-step1.md § Not done](qonnx-step1.md#not-done-and-why)).
Date: 2026-11-21, host only (x86-64), nothing under `artifacts/` was written. Machine-readable results:
the development log (not published). Canonical QONNX files, re-imported files and reports:
`.cache/qonnx_step1b/<id>/<variant>/`. Command (the run was split in shards of 3–4 GB, BNAS-H alone at 8 GB; `--out` one JSON per
shard, merged):

```bash
scripts/capped.sh 3G nice -n 10 env OMP_NUM_THREADS=3 .cache/venv-qonnx/bin/python -I tools/qonnx_step1b_acceptance.py [--only ID ...]
```

**Identity bar** (design decision 2026-11-20, Q11): the re-imported `.bitino` has the published file's data SHA-256 and
graph SHA-256 (= `registry/recipes.json` → `expected` for `model.bitino`), names included byte for byte; the manifest
is not compared. **Level 3** is the numerical rule.

**Result: all 16 networks, 28 of 28 published float files, rebuild from their canonical QONNX files with the published data and graph SHA-256** (the 16 `model.bitino` = the recipes' expected hashes, the 12 `model_sign_zero.bitino` = the published files), every canonical file passes `check-qonnx` with no error and no warning, and every file is fully verified: outputs bit-identical with the published file in the numpy executor and both C++ stages on the model's check set, C++ reference bit-exact with numpy, Level 3 passed (images: the per-pixel rule below). Nothing is left on a legacy converter. The profile needed four extensions, each implemented, documented and enforced by the checker, listed under [Decisions for the user](#decisions-for-the-user).

## Scope

| family (legacy converter, adapter) | networks | files | status |
|---|---:|---:|---|
| BBCU DnCNN denoise / deblocking, BBCU-L / BBCU-M SR x2 / x4 (`convert_bbcu.py`, `torch_bbcu`) | 6 | 12 | **done** (12 files identical and verified) |
| BSRC-2023 binary SR x2 / x4, v0 / v1 (`convert_bsrc.py`, `torch_bsrc`; `ge` is the source semantics: one file each) | 4 | 4 | **done** (4 files identical and verified) |
| ReActNet-A BiMatting backbone (`convert_reacta_bn.py`, `torch_abbnn`) | 1 | 2 | **done** (2 files identical and verified) |
| BNAS-A / D / E / H (`convert_bnas.py`, `torch_bnas`) | 4 | 8 | **done** (8 files identical and verified) |
| AdaBin ResNet-20 (`convert_adabin.py`, `torch_adabin`) | 1 | 2 | **done** (2 files identical and verified) |

## What was built

| file | change |
|---|---|
| `tools/export_torch_qonnx.py` | hooks `bbcu`, `bsrc`, `bnas`, `adabin`; `reacta` extended to BiMatting; image-output families (`Restoration`: outputs NHWC files, the ONNX output NCHW, `bitino.labels none`, `bitino.task`); the input shape is known to the hook (shape-specific index maps); hooks may ask for batch-1 torch evaluation and carry `bitino.const_order`; `canonicalize` treats `Where(Greater(x, t), 1, −1)` as a binarizer (explicit `Pad(0)`, R2) and writes `bitino.float_layer` |
| `python/bitino/qonnx_import.py` | new pattern methods (`_step_image`, `_step_const_map`, `_step_adabin`, `_float_layer_on_bin`), entered from one line at the top of `step()`, one branch in `conv()` and one in `mat()`; `verify` compares rank-4 (image) outputs in NHWC order |
| `python/bitino/qonnx_profile.py`, `docs/qonnx-profile.md` | profile extensions (below): § 4.1 AdaBin conv (`E-ADABIN` / `I-ADABIN`), § 4.3 constant-map `Add`, § 4.6 image data movement, § 7 `bitino.float_layer` (`E-FLOAT-LAYER` / `I-FLOAT-LAYER`) |
| `tools/qonnx_step1b_acceptance.py` | this acceptance (step 1's flow) plus the image-output Level-3 rule and the recipe's `validate_args` for the C++ check |
| `tests/python/test_qonnx_import.py`, `tests/python/test_qonnx_profile.py` | 13 new tests on synthetic graphs |

### Export hooks

| hook | source constructs → canonical QONNX |
|---|---|
| `bbcu` | the legacy adapter runs first (module / parameter coverage, float64 `mean\|W\|`, index maps, phase kernel); `x·k` → `Mul`; BinaryConv2d → `Add(move0) → Bin → Pad(0) → Conv(BipolarQuant(W, 1)) → Mul(mean\|W\| c64) [→ Add(bias)] → Add → PRelu → Add → Add(x)`; DnCNN tail `Sub(xk, n)`, `Mul(fl64(1/k) c64)` (the constant Bitino stores; torch divides: ≤ 1 ulp after the last binarization); PixelShuffle, nearest ×r, bilinear ×r (§ 4.6); LeakyReLU (alpha 0.1 via `bitino.attrs`) |
| `bsrc` | `sign(sign(x) + 0.1)` = `BipolarQuant` (the source semantics, one file); `Scale` = `Mul + Add` (fp32 as the source); LAB = depthwise Conv → 1×1 grouped Conv (−1, +1) → `Where(Greater(d, 0), 1, −1)` → `BConv2D gt`; `Normalize` `Add(−127.5) + Mul(fl64(1/51) c64)`; BN carriers; PixelShuffle (§ 4.6) |
| `reacta` (BiMatting) | `BiConv2d` centred latent `W − nested_mean(W)` (float64, the adapter's), 4C and C/2 blocks (the half sum as § 4.6's channel split), `forward_single_frame` macro residuals (AvgPool2d(2) ceil mode = floor mode on the even maps, checked); a block after a macro residual binarizes the sum literally (no PReLU tail: the adapter's rule) |
| `bnas` | **translates the legacy adapter's IR graph (before the passes) node by node into exportable modules**: the adapter's graph-level decisions (BN sign folded into the next conv's weights, `ChannelAffine(b/s)` thresholds as float64 carriers, the `ge` zero-channel correction map, the BinOp quirk convs, shared pools, dropped dead branches, all-zero cell states `Mul(x, 0)`, the classifier rows in the standard ImageNet order) are reproduced by construction; the export check compares the canonical file with the authors' unmodified model (BinOp applied) |
| `adabin` | `BinaryActivation` = `Sub(β) → Sign / BipolarQuant → Add(β) → Mul(α)` (the source's fp32 operations on the sign; `/α` inside the sign dropped, as the adapter: the same sign for α > 0 except fp32 underflow of `(x−β)/α`); `AdaBin_Conv2d` = `Conv` with the source's fp32 two-level weights (zero pads); BN carriers; Maxout = `Relu(x)·pos + Relu(x·(−1))·(−neg)` (bit-exact); option-A shortcut; the legacy (post-pass) data order as `bitino.const_order` |

### Importer patterns (all on synthetic tests)

* **PixelShuffle**, **row gather** (nearest, replicate padding), **channel split** (BiMatting half sum), **spatial-only
  Reshape** (§ 4.6): pending values of new kinds (`ps6`, `rows`, `split`) carry the data movement to the node that
  closes it; the IR names are `bitino.name` (last node) and `bitino.extra_names` (the others).
* **Constant-map Add** (§ 4.3): `Add(x, M)` with a spatially varying `[1, C, H, W]` constant → IR `Add(x, fl32(M) NHWC)`.
* **Declared float layer** (§ 7 `bitino.float_layer`): `Activation(sign / sign_zero)` + float `Conv2D`.
* **AdaBin conv** (§ 4.1): the legacy five-node decomposition with the BN folded in, emitted at the end of the affine
  chain (the adapter's node order: before the shortcut), every constant in float64 in the adapter's operation order.
* **`bitino.const_order`** (§ 7): step 1b implemented it as the order of the importer's constants before the
  passes; step 2 (merged into master meanwhile) implemented it as the order of the final data section, applied
  after the passes. The merge keeps step 2's implementation (one semantics); the AdaBin hook writes the
  post-pass order (the adapter's graph with the legacy converter's passes applied).

## Acceptance

**Acceptance of a row** = check-qonnx passes with no warning, data + graph SHA-256 identical (and = the recipe for `model.bitino`), outputs bit-identical with the published file (numpy, C++ reference, C++ optimized), the C++ reference stage bit-exact with numpy (`tools/validate_model.py --require-exact-numpy` with the recipe's `validate_args`), and Level 3. **Level 3 for images** (`tools/qonnx_step1b_acceptance.py`): per sample and pixel, IR (numpy fp64) vs the QONNX graph in float64; a sample with a pixel deviating by more than 1e-6 × the output range (BBCU 1, BSRC 255) is accepted when its first binarization flip is at a |pre-activation| ≤ 1e-3 × the median |pre| of that binarizer, or it has no flip and every pixel is within 1e-5 × the range. Classifiers: step 1's rule. Columns: Level 3 = samples within 1e-6 · deviating with a first flip (max |pre| / median |pre|) · deviating without a flip (max |Δ| / range or max |logit|); info = PSNR of the numpy fp32 output vs the fp64 reference and the number of deviating pixels (images), top-1 agreement numpy fp32 vs fp64 (classifiers, Level 4 informational); export check = the authors' model in float64 vs the canonical file in float64 (max |Δ|).

| network | file | check-qonnx | data SHA-256 | graph SHA-256 | = published (recipe) | outputs numpy / C++ ref / C++ opt | Level 3: within · flip (max rel \|pre\|) · no flip (max rel \|Δ\|) | info | export check fp64 max \|Δ\| | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `bbcu-denoise` | model | pass | `b78cd07d3f3d8bdc` | `595484b44d48bbbc` | yes / yes (yes) | yes / yes / yes | 64 · 4 (6.9e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 62.0 dB; deviating pixels 2062/1880064 | 1e-15 | **PASS** |
| `bbcu-denoise` | model_sign_zero | pass | `b78cd07d3f3d8bdc` | `e35fd4a48cf97960` | yes / yes | yes / yes / yes | 64 · 4 (6.9e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 62.0 dB; deviating pixels 2062/1880064 | 1e-15 | **PASS** |
| `bbcu-deblocking` | model | pass | `6b6f9d3421974b59` | `b55294e00bf32f52` | yes / yes (yes) | yes / yes / yes | 61 · 7 (6.1e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 63.6 dB; deviating pixels 2102/626688 | 3.3e-16 | **PASS** |
| `bbcu-deblocking` | model_sign_zero | pass | `6b6f9d3421974b59` | `4b71e1179e7bf92b` | yes / yes | yes / yes / yes | 61 · 7 (6.1e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 63.6 dB; deviating pixels 2102/626688 | 3.3e-16 | **PASS** |
| `bbcu-sr-l-x2` | model | pass | `ae386457eb9ba698` | `4722dc435fe27c05` | yes / yes (yes) | yes / yes / yes | 63 · 1 (2.8e-09) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 50.9 dB; deviating pixels 9525/1769472 | 3e-09 | **PASS** |
| `bbcu-sr-l-x2` | model_sign_zero | pass | `ae386457eb9ba698` | `130e06caeb56eaff` | yes / yes | yes / yes / yes | 63 · 1 (2.8e-09) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 50.9 dB; deviating pixels 9525/1769472 | 3e-09 | **PASS** |
| `bbcu-sr-l-x4` | model | pass | `06e59d1b01cf1b6e` | `5de8ead37059de4c` | yes / yes (yes) | yes / yes / yes | 63 · 1 (2.5e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 54.7 dB; deviating pixels 40048/3145728 | 4.8e-09 | **PASS** |
| `bbcu-sr-l-x4` | model_sign_zero | pass | `06e59d1b01cf1b6e` | `53fe136a5b1240c1` | yes / yes | yes / yes / yes | 63 · 1 (2.5e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 54.7 dB; deviating pixels 40048/3145728 | 4.8e-09 | **PASS** |
| `bbcu-sr-m-x2` | model | pass | `3eb11a461e786add` | `796211b8b0dd9cc2` | yes / yes (yes) | yes / yes / yes | 60 · 4 (3.6e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 49.0 dB; deviating pixels 33189/1769472 | 2.7e-10 | **PASS** |
| `bbcu-sr-m-x2` | model_sign_zero | pass | `3eb11a461e786add` | `857d22e1a5f6ed7a` | yes / yes | yes / yes / yes | 60 · 4 (3.6e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 49.0 dB; deviating pixels 33189/1769472 | 2.7e-10 | **PASS** |
| `bbcu-sr-m-x4` | model | pass | `4b11feb2661e5faa` | `7353cb8a0271cf20` | yes / yes (yes) | yes / yes / yes | 61 · 3 (1.5e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 58.7 dB; deviating pixels 19765/3145728 | 0.0081 | **PASS** |
| `bbcu-sr-m-x4` | model_sign_zero | pass | `4b11feb2661e5faa` | `f1df1e68c761303a` | yes / yes | yes / yes / yes | 61 · 3 (1.5e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 58.7 dB; deviating pixels 19765/3145728 | 0.0081 | **PASS** |
| `bsrc2023-binary-sr-x2-v0` | model | pass | `f3d859ab8ac03650` | `15d7d0d22af1f589` | yes / yes (yes) | yes / yes / yes | 62 · 2 (1.3e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 64.5 dB; deviating pixels 5022/1769472 | 2.6e-13 | **PASS** |
| `bsrc2023-binary-sr-x2-v1` | model | pass | `50dfdf8743ffd15f` | `551cd8c1e71e3ff1` | yes / yes (yes) | yes / yes / yes | 61 · 3 (3.6e-08) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 66.0 dB; deviating pixels 13743/1769472 | 2.3e-13 | **PASS** |
| `bsrc2023-binary-sr-x4-v0` | model | pass | `ad035e0e97ef3094` | `86dc51ea07fc1563` | yes / yes (yes) | yes / yes / yes | 64 · 0 (-) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 148.8 dB; deviating pixels 0/3145728 | 3.1e-13 | **PASS** |
| `bsrc2023-binary-sr-x4-v1` | model | pass | `ad6f72b94ee11e71` | `6090e93a18728c75` | yes / yes (yes) | yes / yes / yes | 64 · 0 (-) · 0 (-) | PSNR numpy fp32 vs fp64 ≥ 63.1 dB; deviating pixels 0/3145728 | 8e-13 | **PASS** |
| `bimatting-bimobilenet-backbone` | model | pass | `86f157b0389c7603` | `ca0d23aa81f1ae95` | yes / yes (yes) | yes / yes / yes | 110 · 10 (3.4e-07) · 8 (9.0e-08) | top-1 numpy fp32 vs fp64 1.000 | 2.1e-14 | **PASS** |
| `bimatting-bimobilenet-backbone` | model_sign_zero | pass | `86f157b0389c7603` | `876516b82ec6d88f` | yes / yes | yes / yes / yes | 110 · 10 (3.4e-07) · 8 (9.0e-08) | top-1 numpy fp32 vs fp64 0.992 | 2.1e-14 | **PASS** |
| `bnas-a-cifar10` | model | pass | `033607629e4de346` | `e54ad33a2eb8f2c5` | yes / yes (yes) | yes / yes / yes | 127 · 1 (1.5e-07) · 0 (-) | top-1 numpy fp32 vs fp64 1.000 | 7.1e-14 | **PASS** |
| `bnas-a-cifar10` | model_sign_zero | pass | `940f478b12ee6bf2` | `6d718a1495289c27` | yes / yes | yes / yes / yes | 127 · 1 (1.5e-07) · 0 (-) | top-1 numpy fp32 vs fp64 1.000 | 7.1e-14 | **PASS** |
| `bnas-d-imagenet` | model | pass | `10286fbf95def160` | `382074cf06ce898a` | yes / yes (yes) | yes / yes / yes | 125 · 3 (7.9e-08) · 0 (-) | top-1 numpy fp32 vs fp64 1.000 | 1.2e-14 | **PASS** |
| `bnas-d-imagenet` | model_sign_zero | pass | `10286fbf95def160` | `e0cce52e172029e5` | yes / yes | yes / yes / yes | 125 · 3 (7.9e-08) · 0 (-) | top-1 numpy fp32 vs fp64 1.000 | 1.2e-14 | **PASS** |
| `bnas-e-imagenet` | model | pass | `65f304d596f75b72` | `050c1c0fced2c654` | yes / yes (yes) | yes / yes / yes | 124 · 4 (1.6e-07) · 0 (-) | top-1 numpy fp32 vs fp64 1.000 | 1.3e-14 | **PASS** |
| `bnas-e-imagenet` | model_sign_zero | pass | `65f304d596f75b72` | `d07ef8c508199481` | yes / yes | yes / yes / yes | 124 · 4 (1.6e-07) · 0 (-) | top-1 numpy fp32 vs fp64 1.000 | 1.3e-14 | **PASS** |
| `bnas-h-imagenet` | model | pass | `e4c7c10f5cfa33d2` | `921e8dcf918aa061` | yes / yes (yes) | yes / yes / yes | 116 · 12 (4.5e-07) · 0 (-) | top-1 numpy fp32 vs fp64 0.977 | 1.6e-14 | **PASS** |
| `bnas-h-imagenet` | model_sign_zero | pass | `e4c7c10f5cfa33d2` | `02d478707b7a93a3` | yes / yes | yes / yes / yes | 116 · 12 (4.5e-07) · 0 (-) | top-1 numpy fp32 vs fp64 0.969 | 1.6e-14 | **PASS** |
| `adabin-resnet20-cifar10` | model | pass | `f5eda3ecf8d2105a` | `b8c4e322cbd7c630` | yes / yes (yes) | yes / yes / yes | 17 · 0 (-) · 111 (3.4e-07) | top-1 numpy fp32 vs fp64 1.000 | 1.6e-14 | **PASS** |
| `adabin-resnet20-cifar10` | model_sign_zero | pass | `f5eda3ecf8d2105a` | `b62e8705f9442490` | yes / yes | yes / yes / yes | 17 · 0 (-) · 111 (3.4e-07) | top-1 numpy fp32 vs fp64 1.000 | 1.6e-14 | **PASS** |

### Per network

| network | files | identical (data + graph SHA-256) | verified (outputs bit-identical, Level 3) | status |
|---|---:|---:|---:|---|
| `bbcu-denoise` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bbcu-deblocking` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bbcu-sr-l-x2` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bbcu-sr-l-x4` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bbcu-sr-m-x2` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bbcu-sr-m-x4` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bsrc2023-binary-sr-x2-v0` | 1 | 1 / 1 | 1 / 1 | **identical, verified** |
| `bsrc2023-binary-sr-x2-v1` | 1 | 1 / 1 | 1 / 1 | **identical, verified** |
| `bsrc2023-binary-sr-x4-v0` | 1 | 1 / 1 | 1 / 1 | **identical, verified** |
| `bsrc2023-binary-sr-x4-v1` | 1 | 1 / 1 | 1 / 1 | **identical, verified** |
| `bimatting-bimobilenet-backbone` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bnas-a-cifar10` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bnas-d-imagenet` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bnas-e-imagenet` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `bnas-h-imagenet` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |
| `adabin-resnet20-cifar10` | 2 | 2 / 2 | 2 / 2 | **identical, verified** |


## Findings

* **The legacy layouts are reachable with standard operators.** Every legacy NHWC-specific construct (6-D reshapes,
  constant row gathers, the replicate-pad + phase-conv bilinear, the BiMatting `[1, HW, 2, P]` average) has an NCHW
  standard-op spelling that the importer maps back node for node, names and constant order included.
* **Constant order.** The image converters and BNAS keep the importer's walk / first-use order; AdaBin's adapter
  creates `threshold` before `ones_bits` (first use: the reverse) and the head's Dense kernel before its bias (the
  importer's walk: the reverse), so its canonical files carry `bitino.const_order` (the final data order of the legacy file; the
  importer checks it names exactly its constants).
* **Image outputs need a family range.** The SR eval sets carry a `[0, 1]` preprocessing (BBCU) and a `[0, 255]` one
  (BSRC); the image Level-3 rule takes the range from the family (`OUTPUT_RANGE`), not from the eval set. BSRC rows were
  first judged against range 1 (one failed: x4-v0, max |Δ| 2.5e-5 on a 0..255 image); they were re-judged with
  `--rejudge` (same files, recorded output comparisons).
* **Export check (source vs canonical file).** In float64 the exportable copy (the canonical graph, in torch) equals
  the authors' model to ≤ 2.6e-13 on every network (the dropped AdaBin `/α`, the bilinear phase conv and the BNAS
  graph-level choices included). The canonical file evaluated by OnnxF64 in float64 agrees with the authors' model to
  ≤ 8e-13 (BSRC, range 255), ≤ 1e-14 (classifiers) and ≤ 1e-15 (BBCU DnCNN); on the BBCU SR models the two float64
  evaluations of the *same* graph (torch vs OnnxF64, different summation orders) differ by up to 5e-9, and on BBCU-M ×4
  one of 64 crops flips one near-tie binarization (max |Δ| 8.1e-3): evaluation-order noise, not a conversion
  difference. In
  float32 the BiMatting backbone, BSRC and BNAS sources are chaotic (BiMatting: 28–29 / 128 top-1 changes between the
  authors' fp32 model and the fp32 canonical file, the 8,735 float32 ties of qonnx-step1.md's ReActNet-A note), as their
  legacy conversion reports already showed.
* **Memory.** The image verifications fit in 3–4 GB per shard; the BSRC shard was OOM-killed once at a 3 GB cap at the
  start of an export (the parent still held its comparison arrays) and rerun at 4 GB. BNAS-D / E run at 4 GB (about
  40 min per file, the float64 reference of 108 binarizers on 224² ImageNet maps); BNAS-H's export does not fit 4 GB
  and ran alone at 8 GB with `--chunk 4` (source logits 4 images at a time), its import with verification peaking at
  ~5.6 GB.
* **Export check of two BNAS rows recomputed.** A disk cleanup during the run removed the source logits of BNAS-D
  `sign_zero` and BNAS-E `ge` before their imports' export check; the export was re-run for those two (the same
  canonical file, SHA-256 checked) to regenerate the source logits, and the export check was recomputed from them
  (`export_check_recomputed` in the summary). Identity, outputs and Level 3 of those rows were not affected.

## Design decisions

Each is implemented (all checks pass with it) and written into docs/qonnx-profile.md and the checker; each can be
reverted or changed without touching the other networks:

1. **`bitino.float_layer` (profile § 7).** BNAS's two BinOp-quirk convs per network convolve a `Sign`ed tensor with
   float weights. Without a declaration the checker warns (`W-FLOAT-WEIGHTS`) and the importer refuses. The extension
   declares such a layer explicitly (error if the weights are ±α, i.e. if it would hide a binary layer). Alternative:
   keep the warning and accept "zero errors, two justified warnings" for the BNAS files.
2. **AdaBin conv pattern (profile § 4.1, class F).** The canonical file states the source's computation (activation
   `α·(Bin(x−β)+β)`, two-level weights, BN); the importer rebuilds the adapter's five-node binary decomposition. Two
   sub-choices: (a) `/α` inside the sign is dropped (exact except fp32 underflow of `(x−β)/α`, the AB-BNN precedent);
   writing `Sign(Div(Sub(x, β), α))` instead would make the file the source bit for bit and needs one more importer
   rule; (b) alternatively the canonical file could spell out the decomposition itself (BConv + 1×1 Conv + map Add),
   which needs a layout override for the 1×1 kernel (the legacy constant has no `OHWI` layout) — not done.
3. **BNAS hook built from the adapter's IR graph.** The canonical BNAS files carry the adapter's graph-level choices
   (BN sign folded into the next conv's bits, `b/s` thresholds, the `ge` zero-channel correction map, zero states as
   `Mul(x, 0)`); they are verified against the authors' model by the export check, but the hook does not re-derive them
   from the source modules. A source-mirroring hook would reproduce the same file.
4. **Image-mode C++ validation interpreter.** `tools/validate_model.py --output-kind image` needs OpenCV (SSIM); the
   acceptance uses the system `python3 -I` (as the legacy S8 runs). Alternative: add OpenCV to a new venv.

## Tests and regressions

* `.cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_import.py tests/python/test_qonnx_profile.py` →
  **109 passed** after the merge of master (step 2): 52 + 57; step 1b added 10 importer tests (PixelShuffle, rank-6
  rejection, nearest row gather, negative index rejection, channel split half sum, constant-map Add, declared float
  layer, AdaBin decomposition ge / sign_zero, three weight levels rejected) and 2 checker tests (float-layer
  declaration, AdaBin pattern conditions).
* **Step-1 sample, before the merge, with this branch's importer / exporter** (separate cache root, nothing of step 1
  overwritten): `recu-resnet18-cifar10`, `rbnn-vgg-small-cifar10` (flatten permute), `ab-bnn-cifar10-reactneta`,
  `siman-resnet20-cifar10` × 2 variants fully re-verified (8 PASS: identical, outputs bit-identical, Level 3);
  `burn-reacta-stage2`, `s2bnn-reacta-distill-only` × 2 (the ReActNet-A hook changed for BiMatting) identical (4).
* **After the merge of master** (step 2: Keras / Larq): AdaBin both files re-run and fully verified (step 2's final
  `bitino.const_order`); identity re-checked for the step-1b sample `bbcu-denoise`, `bbcu-sr-m-x4`,
  `bsrc2023-binary-sr-x2-v1`, `bimatting-bimobilenet-backbone`, `bnas-a-cifar10` (5 identical), the step-1 sample
  `recu-resnet18-cifar10` × 2, `rbnn-vgg-small-cifar10` × 2, `ab-bnn-cifar10-reactneta` × 2,
  `bitorch-resnete18-imagenet` (7 identical) and the step-2 sample `larqface-binaryfacenet-msceleb` (its
  `bitino.const_order` / attrs null), `larq-zoo-dorefanet` (DoReFa), `larq-zoo-quicknet-small`, `brillsson-quicknet`
  (re-import of the step-2 canonical files: 4 identical).
* **Merge notes.** Both branches had implemented `bitino.const_order`: step 1b as the order of the importer's
  constants before the passes, step 2 as the final data order after them. The merge keeps step 2's; AdaBin's hook now
  writes the post-pass order. The step-1b pattern hook runs before step 2's new `Transpose` / DoReFa interceptors
  (step 2's `Transpose` rule accepts only the Keras flatten and would reject the PixelShuffle / channel-split
  transposes).

## Running it

`tools/qonnx_step1b_acceptance.py` runs every network / variant of its `NETWORKS` (`--only`, `--variant`,
`--no-verify`, `--skip-run`, `--cache-dir`, `--artifacts-dir`, `--bench`, `--out`) into `<cache>/qonnx_step1b/`. The
exports run in `.cache/venv-qonnx2`, the imports and checks in `.cache/venv-qonnx`, the image-mode
`tools/validate_model.py` (C++ reference stage, `--require-exact-numpy`, the recipe's `validate_args`) in the system
`python3 -I` (`--image-validate-python`): it needs OpenCV for the SSIM of the task metric, which `.cache/venv-qonnx` does
not have (the legacy S8 validations used the same interpreter).

Disk: to keep the host above 40 GB free while other jobs ran, the intermediates of accepted rows (torch source /
exportable outputs, fp32 / fp64 numpy outputs) and the canonical files of four accepted BNAS rows (`bnas-d-imagenet`
both, `bnas-e-imagenet` ge, `bnas-h-imagenet` ge; 114–640 MB each) were deleted from `.cache/qonnx_step1b/` after their
rows were recorded; each row keeps its re-imported `model.bitino`, `export.json` and `conversion_report.json`, and the
summary records each canonical file's SHA-256 (`export.onnx_sha256`); rerunning the acceptance for the network
regenerates it.
