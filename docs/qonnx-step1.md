# QONNX route, step 1: acceptance report (PyTorch CNNs)

Plan: [qonnx-frontend.md § Plan](qonnx-frontend.md#plan), step 1. Date: 2026-11-20, host only (x86-64), nothing under
`artifacts/` was written. Machine-readable results: the development log (not published). Canonical QONNX files, exports,
re-imported files and reports: `.cache/qonnx_step1/<id>/<variant>/`. Command to reproduce (one shard; the run was split
in four shards of at most 4 GB each, see [Running it](#running-it)):

```bash
scripts/capped.sh 4G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_step1_acceptance.py [--only ID ...]
```

**Identity bar (design decision 2026-11-20).** Data + graph SHA-256 identity is sufficient: the re-imported `.bitino`
has the published file's data SHA-256 and graph SHA-256 (= `registry/recipes.json` → `expected` for `model.bitino`),
i.e. every constant byte in the same order and the same graph JSON, **including every layer, node and constant name,
byte for byte**. Whole-file / manifest identity (converter name, timestamps, provenance fields of the manifest) is not
required and not compared (docs/qonnx-profile.md Q11).

**Result: 43 networks, 83 published float files: every file rebuilds from its canonical QONNX file with the published
data and graph SHA-256** (all 43 `model.bitino` = the recipe's expected hashes; 40 `model_sign_zero.bitino`), and every
canonical file passes `check-qonnx` with no error and no warning. 81 of the 83 files are also fully verified (outputs
bit-identical with the published file in the numpy executor and both C++ stages, C++ reference bit-exact with numpy,
Level 3 passed). BNext-L's two files are identical but not verified: its verification needs more than the ~10 GB memory
budget of this run (below). 16 networks are not done (below), each still on its legacy converter.

## Scope

Every network of `registry/recipes.json` whose converter is a PyTorch converter (`tools/convert_*.py` reading a
PyTorch / Lightning checkpoint), minus the transformers (BiBERT, Bi-ViT, BHViT, GSB) and the 18 Faster-Than-Lies
detectors (step 3): **59 networks**, 111 published float files (`model.bitino`, and `model_sign_zero.bitino` where the
source binarizer is `torch.sign`). The int8_full files are step 5.

| family (converter, adapter) | networks | files | status |
|---|---:|---:|---|
| IR-Net: ReCU, BiPer, RBNN, SiMaN (`convert_irnet.py`, `torch_irnet`): ImageNet ResNet-18/34, CIFAR ResNet-18/20 (RBNN also Bi-Real-20), VGG-Small; ShuaiZ Bi-Real-20 | 22 | 44 | **done** |
| AB-BNN ReActNet-18 / -34 (`convert_abbnn.py`, `torch_abbnn`): ImageNet, CIFAR-10, CIFAR-100 | 4 | 8 | **done** |
| ReActNet-A (`torch_abbnn._convert_reactnet_a`): AB-BNN `Reactnet` ×3 (`convert_abbnn.py`), BURN, S2-BNN (`convert_reacta_bn.py`) | 5 | 10 | **done** |
| ReActNet-A BiMatting backbone (`convert_reacta_bn.py`) | 1 | 2 | not done (below) |
| RBONN adapter users (`torch_rbonn`): RBONN, SNN sub-bit W1A1 / W0.67A1, IDa-Det BiRes-18 backbone | 4 | 8 | **done** |
| SA-BNN ResNet-18 (`convert_sabnn.py`, `torch_sabnn`) | 1 | 2 | **done** |
| BNext-T / S / M / L (`convert_bnext.py`, `torch_bnext`) | 4 | 8 | **done** (BNext-L: identical, verification not run: memory) |
| bitorch ResnetE18, DenseNet28, MeliusNet22 (`convert_bitorch.py`, `torch_bitorch`; `ge` is the source semantics: one file each) | 3 | 3 | **done** |
| BBCU denoise / deblocking / SR-L / SR-M x2 x4 (`convert_bbcu.py`, `torch_bbcu`) | 6 | 12 | not done (below) |
| BSRC-2023 binary SR x2 / x4 v0 / v1 (`convert_bsrc.py`, `torch_bsrc`) | 4 | 4 | not done (below) |
| BNAS-A / D / E / H (`convert_bnas.py`, `torch_bnas`) | 4 | 8 | not done (below) |
| AdaBin ResNet-20 (`convert_adabin.py`, `torch_adabin`) | 1 | 2 | not done (below) |

The four phase-2 networks (AB-BNN ReActNet-18, RBONN, ReCU-18, BiPer-18) are part of the run: their canonical files
are now profile-conformant (phase 2's were not) and still rebuild both published files byte for byte.

## What was built

| file | change |
|---|---|
| `tools/export_torch_qonnx.py` | the exporter now writes **canonical files of the profile** (`bitino-qonnx/1`), checked with `check-qonnx` before it returns (an export that does not pass fails); hooks `abbnn`, `reacta`, `rbonn` (RBONN + SNN + IDa-Det), `irnet` (all 22 IR-Net networks), `sabnn`, `bnext`, `bitorch` |
| `python/bitino/qonnx_import.py` | step-1 patterns and metadata (below); importer version unchanged in name (`qonnx_import 2`), options `--flatten-dense`, the bitino.attrs / bitino.consts / bitino.extra_names readers |
| `python/bitino/onnx_f64.py` | float64 carriers (R7): `Cast(DOUBLE initializer or Constant, to=FLOAT)` keeps the float64 value in a float64 evaluation |
| `python/bitino/qonnx_profile.py`, `docs/qonnx-profile.md` | optional model key `bitino.ir_unique_counter` (§ 6 / § 7) |
| `tools/qonnx_step1_acceptance.py` | this acceptance run (export, check-qonnx, import + Level-3 verification + C++, comparison with the published files and the recipe hashes) |
| `tests/python/test_qonnx_import.py` | 5 new tests on synthetic graphs (34 in total) |

### The canonical file (exporter rewrite to the profile)

`torch.onnx.export` (opset 17, TorchScript, `do_constant_folding=False`) followed by `canonicalize`:

| profile rule | what the exporter writes |
|---|---|
| § 1 container | ONNX IR version 10 (node `metadata_props`); `ai.onnx` 17 + `qonnx.custom_op.general` 1; Constant nodes become initializers; torch's de-duplicated buffers (`Identity` of an initializer) are replaced by the initializer; unused initializers dropped; producer `bitino tools/export_torch_qonnx.py` |
| R1 binarizers | `Sign` (the source's `torch.sign`, variant `sign_zero`) or `BipolarQuant(x, 1)` (`--binarizer ge`, variant `ge`), scale an initializer > 0 |
| R2 binary padding | an explicit `Pad(x_bin, pads, value 0)` before every Conv that pads a binarized tensor, and `pads = 0` on the Conv (bitorch: the source's own `Pad(-1)`) |
| R4 binary weights | `Conv(Bin(x), BipolarQuant(W_latent, 1))` → `Mul(alpha[o])` → [`Add(bias)`] in the Conv's naming group (no `W-ALPHA-INSIDE`); alpha keeps its sign (the legacy bit polarity); a binary-weight conv on a *float* input (AB-BNN ReActNet-18 downsample) keeps `BipolarQuant(W, fl32(alpha))`: it is a float `Conv2D` with ±alpha weights in the legacy file, not a binary layer |
| R6 / R7 BatchNorm and derived constants | `Mul(x, Cast(s64))` + `Add(·, Cast(t64))` with DOUBLE initializers (the adapters' float64 fold); every constant the adapters compute in float64 (alpha, b / beta, 1 − scale, refolded thresholds) is a float64 carrier |
| § 6 model metadata | all required keys: profile, model id, variant `float`, layout, input domain `float32`, preprocessing key + definition (from the eval set), labels (`imagenet-1k` / `cifar10` / `cifar100`, `none` for the BURN / S2-BNN embeddings), framework `pytorch` + version, checkpoint SHA-256, weights licence (registry), exporter (tool@commit, torch, onnx), source code (`repo@commit`), notes (binarizer variant and published file) |
| § 7 naming | `bitino.name` on every activation-path node (none on constant-only nodes), `bitino.order` the same on every node of a group (the exporter position of the group's first node), `bitino.attrs` / `bitino.extra_names` / `bitino.consts` where the legacy layout needs them, model key `bitino.ir_unique_counter` |

The names come from the module path (the default) or from a hook's per-module `bitino_meta`
(`{"name", "extra_names", "consts", "attrs"}` or `{"by_op": {op type: {...}}}`), written in the adapter's creation
order. torch's JIT common-subexpression elimination is switched off during the export: it merged the two identical
activation-centering subgraphs of the RBNN / SiMaN CIFAR ResNet-18 (the legacy graph, like the source, computes the
mean once per binary conv, also for the 1×1 shortcut conv on the same input).

**check-qonnx:** every exported file passes with no error and no warning (findings recorded per file in
`summary.json` → `check_qonnx`).

### Export hooks

Each hook mirrors its adapter node for node, checks every module type it meets (an unknown type stops the export) and
reuses the adapter's own helpers for every derived constant (`binary_weight`, `binary_conv_parts`,
`bn_binary_conv_parts`, `_hard_binary_conv_bits`, `_binary_bits`, `qprelu_slope`, `fold_scaled_std_conv`, ...):

| hook | source constructs → canonical QONNX |
|---|---|
| `irnet` | `BinarizeConv2d`: `Conv(Bin(a), BipolarQuant(bw, 1))·alpha` (bw = the source's fp32 weight function, exactly ±1, checked); RBNN / SiMaN centering `sign((a − mean_CHW a)/σ)` → `ReduceMean(a, [1,2,3])` → `Sub` → Bin (σ > 0 dropped, as the adapter); option-A shortcut `x[:, :, ::2, ::2]` + channel pad → `AveragePool(1, stride 2)` + `Pad` (checked numerically against the module, as the adapter); avgpad shortcut `AvgPool2d(2)` + `Pad`; bconv shortcut; VGG-Small `x.view(N, -1)` + Linear → `Flatten` + `Gemm`, imported with `--flatten-dense permute`; ShuaiZ `BiRealConv` → `·mean|W|` (carrier) `+ bias`; Hardtanh → `Clip`; names drawn from `Graph.unique()` in the adapter's order (`stem_hardtanh_1`, `layer1.0.nonlinear_2`, ...) |
| `abbnn` | as phase 2, with canonical weights (±1 + `Mul(gain·mean|W_std|·alpha)` carrier) and carriers for `b / beta` |
| `reacta` | AB-BNN `Reactnet` (no BN; `beta` dropped as in ReActNet-18) and the BN flavour (BN after each binary conv folded into its epilogue: `Mul(mean|W|·s_bn)` + `Add(t_bn)` carriers); downsample blocks: `Concat` of the per-half `binary_pw_down_j + out1`; the BN flavour's refolded thresholds (below) |
| `rbonn` | RBONN `RBOConv` (`·0.02|sf|`), SNN `HardBinaryConv` (`·mean|W|`; W0.67A1: the source's own pattern decode, the decoded ±1 kernel as the latent), IDa-Det `BiConv` (`·0.02|scale| + bias`), IDa-Det `AvgPool2d(ceil_mode=True)` exported in floor mode (equal on the even maps, checked) |
| `sabnn` | BN before each sign, flat `HardBinaryConv` weights, the shared `downsample` applied to `bn2(x)` |
| `bnext` | `HardSign` → Bin; `HardBinaryConv` (no scale); info recoupling `Mul(x', scale) + Mul(t, 1 − scale)` (carriers, FFN_1x1 clamps scale in eval as the source); squeeze-and-expand `Mul(t, Sigmoid(g))`; ELM attention `Mul(t, residual)` |
| `bitorch` | `QConv2d`: `BipolarQuant(x, 1)` → `Pad(−1)` → `Conv(BipolarQuant(W, 1))` (0 → +1 for x and W: exactly bitorch's SignFunction, checked on the live quantizers); DenseLayer `Concat`; MeliusNet sliced addition `Pad` (channels in front) + `Add`; non-affine BN |

### Importer changes (all on synthetic tests)

* **Group epilogue** (`--affine keep`, exporter naming): a constant `Mul` / `Add` / `Sub` of the binary operator's own
  naming group is its epilogue (post_mul / post_add), not a ChannelAffine: the canonical `Conv → Mul(alpha) → Add(b)`.
  A `Mul` / `Add` of another group (a BatchNorm) stays its own ChannelAffine, as in the legacy files.
* **Explicit binary Pad**: `pad_value` is the Pad's (FLOAT) value in its JSON form (`0.0`, `-1.0`); Conv zero padding
  of a binarized tensor (non-canonical inputs) keeps reading as `0.0`, an unpadded binary conv as `0`.
* **`bitino.attrs`** (§ 7): replaces an attribute's JSON form (`pad_value: 0` → `0.0`, the form every PyTorch
  adapter writes also for unpadded 1×1 binary convs) or a float32 attribute's float64 value; any other change is an
  import error. **`bitino.consts`**: renames the constants of the group's IR nodes (`{"1": "c/minus_one"}`,
  `{"<extra>#1": ...}`), keeping their data-section position. **`bitino.extra_names`**: names the further IR nodes a
  pattern emits.
* **New patterns**: `ReduceMean(x, [1, 2, 3], keepdims)` → `Reshape [N, CHW]` + `ReduceMean(axis −1)`;
  `Sub(u, w)` of runtime tensors → `ChannelAffine(w, −1)` + `Add` / `AddBroadcast` (exact: `fl(−w) = −w`,
  `fl(u + (−w)) = fl(u − w)`); `Pad` of a float tensor on the channel axis (NHWC pads on C); `Mul(x, Sigmoid(g))`
  with `g` `[N, C, 1, 1]` → `ChannelGate(sigmoid)`; `Mul` of equal-shape runtime tensors → `Mul`;
  `--flatten-dense permute`: a flatten of an NCHW tensor feeding a `Gemm` / `MatMul` becomes a Reshape of the NHWC
  tensor with the Dense columns permuted (the legacy PyTorch layout) instead of a Transpose.
* **No qonnx cleanup** for canonical files (`--cleanup none`, recorded in `export.json` → `import_args`): the file is
  self-contained with static shapes, and `FoldConstants` would fold the float64 carriers into FLOAT initializers (the
  float64 reference would lose R7). The fp32 reference is still qonnx `execute_onnx`.

## Acceptance table

Data / graph SHA-256 of the re-imported file (first 16 hex digits; full hashes in `summary.json`), compared with the
published `artifacts/<id>/<file>` and, for `model.bitino`, with the recipe's expected hashes (in parentheses);
`model_sign_zero.bitino` has no recipe entry and is compared with the published file. Outputs: every sample of the
model's check set (`eval128*`: 128 images), numpy executor (fp32) and C++ runtime stages `reference` and `optimized`,
published vs re-imported file, bit for bit. Verification (Level 3, `docs/numerical-correctness.md`): IR (numpy, fp64)
vs the QONNX graph in float64 (`OnnxF64`, carriers at their float64 value), samples within 1e-6; samples whose first
binarization decision differs from the QONNX fp32 graph (qonnx `execute_onnx`), with the largest |pre-activation| at a
flip in fp32 / fp64; export check = the authors' unmodified model vs the canonical file (fp32 top-1 agreement, fp64 max
|Δ|).

**Acceptance of a row** = check-qonnx passes, data + graph SHA-256 identical (and = the recipe for `model.bitino`),
outputs bit-identical with the published file (numpy, C++ reference, C++ optimized), the C++ reference stage bit-exact
with the numpy executor (`--require-exact-numpy`), and **Level 3** of `docs/numerical-correctness.md`: every sample
whose IR (fp64) deviates from the QONNX fp64 graph by more than 1e-6 either has its first binarization flip at a
|pre-activation| ≤ 1e-3 × the median |pre| of that binarizer (the column gives the largest ratio; measured: ≤ 1.2e-4
for the bitorch models with their unscaled activations, ≤ 1.7e-6 for every other network), or has no flip and differs by ≤ 1e-5 × max
|logit| (float32 storage of the folded constants: the fp64 QONNX graph reads the float64 carriers, the IR their float32
roundings). The **Level-4** rule (top-1 agreement with fp64 ≥ 98 %, disagreements ≤ the reference's own fp32 ones +
max(1, 1 %)) is the device-validation rule of `tools/validate_model.py`; it is reported, not required. For the
fp32-chaotic networks (AB-BNN ReActNet-34, SNN, ReCU / SiMaN ResNet-34 `sign_zero`, bitorch, BNext) the published
files themselves do not meet it: the numpy-vs-fp64 disagreement counts here are the legacy conversion reports' counts
for the same files (e.g. 4 / 5 / 4 / 6 / 4 / 2 / 5 for ReCU-34 `sign_zero`, ReActNet-34, SNN W1A1 / W0.67A1 `sign_zero`,
ResnetE18, DenseNet28, MeliusNet22), as they must be for byte-identical files.

| network | file | check-qonnx | data SHA-256 | graph SHA-256 | data / graph = published (recipe) | outputs bit-identical numpy / C++ ref / C++ opt | Level 3: ≤ 1e-6 · flip (max \|pre\|/typical) · no flip (max \|Δ\|/\|logit\|) | Level 4 (info): numpy fp32 vs fp64 disagreements / reference's own | export check fp32 top-1 / fp64 max \|Δ\| | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `ab-bnn-imagenet-reactnet18` | model | pass | `ecce5fe851958c06` | `8fd2a98bd119c235` | yes / yes (yes) | yes / yes / yes | 117 · 10 (1.8e-08) · 1 (3.7e-08) | 1 / 0 | 1.000 / 2.61 | **PASS** |
| `ab-bnn-imagenet-reactnet18` | model_sign_zero | pass | `ecce5fe851958c06` | `888e734a95b0e9c6` | yes / yes | yes / yes / yes | 117 · 10 (1.8e-08) · 1 (3.7e-08) | 0 / 0 | 1.000 / 2.61 | **PASS** |
| `rbonn-reactnet-resnet18-imagenet` | model | pass | `c6bdc47b26fe6784` | `14d7eb5f7845faad` | yes / yes (yes) | yes / yes / yes | 2 · 5 (1.0e-07) · 121 (2.2e-07) | 0 / 0 | 0.984 / 4.97e-14 | **PASS** |
| `rbonn-reactnet-resnet18-imagenet` | model_sign_zero | pass | `c6bdc47b26fe6784` | `7ad97ac3b034a0fa` | yes / yes | yes / yes / yes | 2 · 5 (1.0e-07) · 121 (2.2e-07) | 0 / 0 | 0.984 / 4.97e-14 | **PASS** |
| `recu-resnet18-imagenet` | model | pass | `15848412289bd57f` | `2b3d61bcc3bcdb55` | yes / yes (yes) | yes / yes / yes | 0 · 5 (2.3e-07) · 123 (1.0e-06) | 0 / 0 | 1.000 / 4.17e-14 | **PASS** |
| `recu-resnet18-imagenet` | model_sign_zero | pass | `15848412289bd57f` | `31b75adb9d274afd` | yes / yes | yes / yes / yes | 0 · 5 (2.3e-07) · 123 (1.0e-06) | 1 / 0 | 1.000 / 4.17e-14 | **PASS** |
| `biper-resnet18-imagenet` | model | pass | `a5f2c2ad0720c4b3` | `2b3d61bcc3bcdb55` | yes / yes (yes) | yes / yes / yes | 0 · 3 (6.5e-08) · 125 (1.2e-06) | 0 / 1 | 0.992 / 4.17e-14 | **PASS** |
| `biper-resnet18-imagenet` | model_sign_zero | pass | `a5f2c2ad0720c4b3` | `31b75adb9d274afd` | yes / yes | yes / yes / yes | 0 · 3 (6.5e-08) · 125 (1.2e-06) | 1 / 2 | 1.000 / 4.17e-14 | **PASS** |
| `recu-resnet34-imagenet` | model | pass | `7f8ce2a9e548601a` | `7fedad4ad3f49b27` | yes / yes (yes) | yes / yes / yes | 0 · 6 (2.9e-08) · 122 (1.3e-06) | 2 / 1 | 0.977 / 5.2e-14 | **PASS** |
| `recu-resnet34-imagenet` | model_sign_zero | pass | `7f8ce2a9e548601a` | `45667e9f0126433e` | yes / yes | yes / yes / yes | 0 · 6 (2.9e-08) · 122 (1.3e-06) | 4 / 2 | 0.969 / 5.2e-14 | **PASS** |
| `recu-resnet18-cifar10` | model | pass | `5c52c9171fa2c172` | `776ad19959a59b91` | yes / yes (yes) | yes / yes / yes | 0 · 0 (-) · 128 (2.0e-07) | 0 / 0 | 1.000 / 2.49e-14 | **PASS** |
| `recu-resnet18-cifar10` | model_sign_zero | pass | `5c52c9171fa2c172` | `2e633dfe9a09fe41` | yes / yes | yes / yes / yes | 0 · 0 (-) · 128 (2.0e-07) | 0 / 0 | 1.000 / 2.49e-14 | **PASS** |
| `recu-resnet20-cifar10` | model | pass | `eb6420273e8c3c8d` | `10620643c5c6e0db` | yes / yes (yes) | yes / yes / yes | 66 · 0 (-) · 62 (4.1e-07) | 0 / 0 | 1.000 / 1.42e-14 | **PASS** |
| `recu-resnet20-cifar10` | model_sign_zero | pass | `eb6420273e8c3c8d` | `d40fdc0638d559bd` | yes / yes | yes / yes / yes | 66 · 0 (-) · 62 (4.1e-07) | 0 / 0 | 1.000 / 1.42e-14 | **PASS** |
| `recu-vgg-small-cifar10` | model | pass | `e93213f0e5381c7c` | `d3ee4c5e8894d9e1` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 3.55e-14 | **PASS** |
| `recu-vgg-small-cifar10` | model_sign_zero | pass | `e93213f0e5381c7c` | `713c244066e0660a` | yes / yes | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 3.55e-14 | **PASS** |
| `biper-resnet34-imagenet` | model | pass | `86befff92b43cb22` | `7fedad4ad3f49b27` | yes / yes (yes) | yes / yes / yes | 0 · 7 (9.4e-08) · 121 (1.1e-06) | 0 / 2 | 0.977 / 5.11e-14 | **PASS** |
| `biper-resnet34-imagenet` | model_sign_zero | pass | `86befff92b43cb22` | `45667e9f0126433e` | yes / yes | yes / yes / yes | 0 · 7 (9.4e-08) · 121 (1.1e-06) | 0 / 2 | 0.977 / 5.11e-14 | **PASS** |
| `biper-resnet18-cifar10` | model | pass | `e2c1c36cf076ab16` | `2996d63493a77b15` | yes / yes (yes) | yes / yes / yes | 0 · 1 (1.8e-08) · 127 (1.4e-06) | 0 / 0 | 1.000 / 4.62e-14 | **PASS** |
| `biper-resnet18-cifar10` | model_sign_zero | pass | `e2c1c36cf076ab16` | `eac2b3dec646b7c9` | yes / yes | yes / yes / yes | 0 · 1 (1.8e-08) · 127 (1.4e-06) | 0 / 0 | 1.000 / 4.62e-14 | **PASS** |
| `biper-resnet20-cifar10` | model | pass | `667736f483d23922` | `4121f81ac23dd3b6` | yes / yes (yes) | yes / yes / yes | 0 · 0 (-) · 128 (8.5e-07) | 0 / 0 | 1.000 / 4.26e-14 | **PASS** |
| `biper-resnet20-cifar10` | model_sign_zero | pass | `667736f483d23922` | `df59e779388f78ce` | yes / yes | yes / yes / yes | 0 · 0 (-) · 128 (8.5e-07) | 0 / 0 | 1.000 / 4.26e-14 | **PASS** |
| `biper-vgg-small-cifar10` | model | pass | `f04f6794ee535deb` | `d3ee4c5e8894d9e1` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 2.84e-14 | **PASS** |
| `biper-vgg-small-cifar10` | model_sign_zero | pass | `f04f6794ee535deb` | `713c244066e0660a` | yes / yes | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 2.84e-14 | **PASS** |
| `rbnn-resnet18-imagenet` | model | pass | `e8a74da6060c5952` | `c96680781597e132` | yes / yes (yes) | yes / yes / yes | 0 · 5 (2.5e-07) · 123 (1.3e-06) | 0 / 3 | 0.977 / 5.15e-14 | **PASS** |
| `rbnn-resnet18-imagenet` | model_sign_zero | pass | `e8a74da6060c5952` | `2ac47dd106d2b035` | yes / yes | yes / yes / yes | 0 · 5 (2.5e-07) · 123 (1.3e-06) | 0 / 3 | 0.977 / 5.15e-14 | **PASS** |
| `rbnn-resnet34-imagenet` | model | pass | `5770ad8badb2ac1f` | `c1e68724be7bee40` | yes / yes (yes) | yes / yes / yes | 0 · 4 (2.8e-08) · 124 (1.1e-06) | 2 / 5 | 0.969 / 3.38e-14 | **PASS** |
| `rbnn-resnet34-imagenet` | model_sign_zero | pass | `5770ad8badb2ac1f` | `988ecdd38ea91770` | yes / yes | yes / yes / yes | 0 · 4 (2.8e-08) · 124 (1.1e-06) | 2 / 5 | 0.969 / 3.38e-14 | **PASS** |
| `rbnn-resnet18-cifar10` | model | pass | `84336a83f1fdc161` | `26e2e4df01f608dd` | yes / yes (yes) | yes / yes / yes | 0 · 1 (3.8e-08) · 127 (1.1e-06) | 0 / 0 | 1.000 / 2.13e-14 | **PASS** |
| `rbnn-resnet18-cifar10` | model_sign_zero | pass | `84336a83f1fdc161` | `f9b4d7f6f9e51b52` | yes / yes | yes / yes / yes | 0 · 1 (3.8e-08) · 127 (1.1e-06) | 0 / 0 | 1.000 / 2.13e-14 | **PASS** |
| `rbnn-resnet20-cifar10` | model | pass | `cfea54ade6f56083` | `dc6953d9663d2900` | yes / yes (yes) | yes / yes / yes | 0 · 0 (-) · 128 (3.4e-06) | 0 / 0 | 1.000 / 4.26e-14 | **PASS** |
| `rbnn-resnet20-cifar10` | model_sign_zero | pass | `cfea54ade6f56083` | `60522d427669b289` | yes / yes | yes / yes / yes | 0 · 0 (-) · 128 (3.4e-06) | 0 / 0 | 1.000 / 4.26e-14 | **PASS** |
| `rbnn-resnet20-bireal-cifar10` | model | pass | `c76835af7d6b2120` | `6e2e6826bf967fc3` | yes / yes (yes) | yes / yes / yes | 0 · 0 (-) · 128 (1.7e-06) | 0 / 0 | 1.000 / 3.73e-14 | **PASS** |
| `rbnn-resnet20-bireal-cifar10` | model_sign_zero | pass | `c76835af7d6b2120` | `13a3805181ba9da7` | yes / yes | yes / yes / yes | 0 · 0 (-) · 128 (1.7e-06) | 0 / 0 | 1.000 / 3.73e-14 | **PASS** |
| `rbnn-vgg-small-cifar10` | model | pass | `6d8a8a21ac8fecd9` | `2d6e896a542c62e7` | yes / yes (yes) | yes / yes / yes | 127 · 1 (1.2e-07) · 0 (-) | 0 / 0 | 1.000 / 2.84e-14 | **PASS** |
| `rbnn-vgg-small-cifar10` | model_sign_zero | pass | `6d8a8a21ac8fecd9` | `360780d334b4cdca` | yes / yes | yes / yes / yes | 127 · 1 (1.2e-07) · 0 (-) | 0 / 0 | 1.000 / 2.84e-14 | **PASS** |
| `siman-resnet18-imagenet` | model | pass | `df16ba6469c4003a` | `a86b94a784605eb9` | yes / yes (yes) | yes / yes / yes | 15 · 5 (9.9e-08) · 108 (1.5e-07) | 0 / 10 | 0.922 / 1.78e-14 | **PASS** |
| `siman-resnet18-imagenet` | model_sign_zero | pass | `df16ba6469c4003a` | `ed60a68d0b293233` | yes / yes | yes / yes / yes | 15 · 5 (9.9e-08) · 108 (1.5e-07) | 0 / 13 | 0.898 / 1.78e-14 | **PASS** |
| `siman-resnet34-imagenet` | model | pass | `3e729e54673bf843` | `c58e6938757d9cfd` | yes / yes (yes) | yes / yes / yes | 0 · 11 (1.8e-07) · 117 (2.3e-07) | 2 / 10 | 0.930 / 1.42e-14 | **PASS** |
| `siman-resnet34-imagenet` | model_sign_zero | pass | `3e729e54673bf843` | `0396a52ab4dee997` | yes / yes | yes / yes / yes | 0 · 11 (1.8e-07) · 117 (2.3e-07) | 3 / 12 | 0.914 / 1.42e-14 | **PASS** |
| `siman-resnet18-cifar10` | model | pass | `c0182500c28346bf` | `26e2e4df01f608dd` | yes / yes (yes) | yes / yes / yes | 127 · 1 (9.9e-08) · 0 (-) | 0 / 0 | 1.000 / 7.11e-15 | **PASS** |
| `siman-resnet18-cifar10` | model_sign_zero | pass | `c0182500c28346bf` | `f9b4d7f6f9e51b52` | yes / yes | yes / yes / yes | 127 · 1 (9.9e-08) · 0 (-) | 0 / 0 | 1.000 / 7.11e-15 | **PASS** |
| `siman-resnet20-cifar10` | model | pass | `7537b641cb437b8e` | `7159b7f3ecdee765` | yes / yes (yes) | yes / yes / yes | 125 · 1 (2.6e-09) · 2 (8.7e-08) | 0 / 0 | 1.000 / 1.24e-14 | **PASS** |
| `siman-resnet20-cifar10` | model_sign_zero | pass | `7537b641cb437b8e` | `1dad0b96ec762fee` | yes / yes | yes / yes / yes | 125 · 1 (2.6e-09) · 2 (8.7e-08) | 0 / 0 | 1.000 / 1.24e-14 | **PASS** |
| `siman-vgg-small-cifar10` | model | pass | `8c3ebdf66a56fe6a` | `b56f648b86ea6404` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 1.78e-14 | **PASS** |
| `siman-vgg-small-cifar10` | model_sign_zero | pass | `8c3ebdf66a56fe6a` | `63a319e23c4f20f9` | yes / yes | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 1.78e-14 | **PASS** |
| `shuaiz-bireal20-cifar10` | model | pass | `29f6b642bf855f79` | `fc36e8f1785a79c1` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 1.42e-14 | **PASS** |
| `shuaiz-bireal20-cifar10` | model_sign_zero | pass | `29f6b642bf855f79` | `2381ea34377dafff` | yes / yes | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 1.42e-14 | **PASS** |
| `ab-bnn-imagenet-reactnet34` | model | pass | `df7f4a42dab50a50` | `e03e6082ad5dd466` | yes / yes (yes) | yes / yes / yes | 103 · 24 (1.7e-07) · 1 (1.0e-07) | 5 / 8 | 0.953 / 7.51 | **PASS** |
| `ab-bnn-imagenet-reactnet34` | model_sign_zero | pass | `df7f4a42dab50a50` | `fb8e5ce6fde6abcd` | yes / yes | yes / yes / yes | 103 · 24 (1.7e-07) · 1 (1.0e-07) | 4 / 7 | 0.961 / 7.51 | **PASS** |
| `ab-bnn-cifar10-reactnet18` | model | pass | `241035c006858244` | `336896aa78af7082` | yes / yes (yes) | yes / yes / yes | 127 · 1 (5.1e-08) · 0 (-) | 0 / 0 | 1.000 / 5.63e-07 | **PASS** |
| `ab-bnn-cifar10-reactnet18` | model_sign_zero | pass | `241035c006858244` | `10b480222d21b372` | yes / yes | yes / yes / yes | 127 · 1 (5.1e-08) · 0 (-) | 0 / 0 | 1.000 / 5.63e-07 | **PASS** |
| `ab-bnn-cifar100-reactnet18` | model | pass | `710c0794069a49fd` | `54a7164b555986da` | yes / yes (yes) | yes / yes / yes | 114 · 0 (-) · 14 (3.6e-08) | 0 / 0 | 1.000 / 1.9 | **PASS** |
| `ab-bnn-cifar100-reactnet18` | model_sign_zero | pass | `710c0794069a49fd` | `6cc79fbd7f83a1df` | yes / yes | yes / yes / yes | 114 · 0 (-) · 14 (3.6e-08) | 0 / 0 | 1.000 / 1.9 | **PASS** |
| `ab-bnn-imagenet-reactneta` | model | pass | `672fcbfbf301b753` | `d78441a1155a8a40` | yes / yes (yes) | yes / yes / yes | 7 · 121 (4.1e-08) · 0 (-) | 10 / 12 | 0.930 / 3.53 | **PASS** |
| `ab-bnn-imagenet-reactneta` | model_sign_zero | pass | `672fcbfbf301b753` | `2fba9a593c5554fe` | yes / yes | yes / yes / yes | 7 · 121 (4.1e-08) · 0 (-) | 15 / 14 | 0.953 / 3.53 | **PASS** |
| `ab-bnn-cifar10-reactneta` | model | pass | `f3ec0596d82746b4` | `8d6d3cbd600334f3` | yes / yes (yes) | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 1.32 | **PASS** |
| `ab-bnn-cifar10-reactneta` | model_sign_zero | pass | `f3ec0596d82746b4` | `baa9203bc02810d4` | yes / yes | yes / yes / yes | 128 · 0 (-) · 0 (-) | 0 / 0 | 1.000 / 1.32 | **PASS** |
| `ab-bnn-cifar100-reactneta` | model | pass | `6ec16e690a5afde6` | `001a02c8510f4788` | yes / yes (yes) | yes / yes / yes | 122 · 0 (-) · 6 (2.8e-08) | 0 / 0 | 1.000 / 2.59 | **PASS** |
| `ab-bnn-cifar100-reactneta` | model_sign_zero | pass | `6ec16e690a5afde6` | `4dbf90c3c0181d4e` | yes / yes | yes / yes / yes | 122 · 0 (-) · 6 (2.8e-08) | 0 / 0 | 1.000 / 2.59 | **PASS** |
| `burn-reacta-stage2` | model | pass | `f104a0e47c2092c5` | `87e3658334825397` | yes / yes (yes) | yes / yes / yes | 115 · 13 (2.1e-07) · 0 (-) | 1 / 2 | 0.867 / 1.6e-14 | **PASS** |
| `burn-reacta-stage2` | model_sign_zero | pass | `f104a0e47c2092c5` | `93b302cfc3d00744` | yes / yes | yes / yes / yes | 115 · 13 (2.1e-07) · 0 (-) | 1 / 2 | 0.867 / 1.6e-14 | **PASS** |
| `s2bnn-reacta-distill-only` | model | pass | `a452de2923901a08` | `de5763652639efaa` | yes / yes (yes) | yes / yes / yes | 114 · 14 (2.0e-07) · 0 (-) | 5 / 5 | 0.727 / 6.44e-15 | **PASS** |
| `s2bnn-reacta-distill-only` | model_sign_zero | pass | `a452de2923901a08` | `0f08cab67e05dba6` | yes / yes | yes / yes / yes | 114 · 14 (2.0e-07) · 0 (-) | 4 / 6 | 0.734 / 6.44e-15 | **PASS** |
| `snn-subbit-reactnet18-w1a1-imagenet` | model | pass | `9a879bf0be64eaf0` | `14d7eb5f7845faad` | yes / yes (yes) | yes / yes / yes | 10 · 6 (2.4e-07) · 112 (1.7e-07) | 1 / 2 | 0.953 / 4.26e-14 | **PASS** |
| `snn-subbit-reactnet18-w1a1-imagenet` | model_sign_zero | pass | `9a879bf0be64eaf0` | `7ad97ac3b034a0fa` | yes / yes | yes / yes / yes | 10 · 6 (2.4e-07) · 112 (1.7e-07) | 4 / 5 | 0.992 / 4.26e-14 | **PASS** |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | model | pass | `5818873d8c7efce3` | `14d7eb5f7845faad` | yes / yes (yes) | yes / yes / yes | 0 · 6 (1.9e-07) · 122 (2.4e-07) | 1 / 1 | 0.961 / 4.26e-14 | **PASS** |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | model_sign_zero | pass | `5818873d8c7efce3` | `7ad97ac3b034a0fa` | yes / yes | yes / yes / yes | 0 · 6 (1.9e-07) · 122 (2.4e-07) | 6 / 4 | 0.992 / 4.26e-14 | **PASS** |
| `ida-det-bires18-imagenet-backbone` | model | pass | `640b1c05f8fab91b` | `0b535c7117ed9f1a` | yes / yes (yes) | yes / yes / yes | 0 · 6 (1.2e-07) · 122 (2.0e-07) | 0 / 0 | 1.000 / 7.11e-14 | **PASS** |
| `ida-det-bires18-imagenet-backbone` | model_sign_zero | pass | `640b1c05f8fab91b` | `8151d8807993f4fe` | yes / yes | yes / yes / yes | 0 · 6 (1.2e-07) · 122 (2.0e-07) | 0 / 0 | 1.000 / 7.11e-14 | **PASS** |
| `sa-bnn-resnet18-imagenet` | model | pass | `f71075ea20008bef` | `2d8531a058ef08f1` | yes / yes (yes) | yes / yes / yes | 3 · 4 (2.9e-07) · 121 (3.0e-07) | 1 / 0 | 0.961 / 3.91e-14 | **PASS** |
| `sa-bnn-resnet18-imagenet` | model_sign_zero | pass | `f71075ea20008bef` | `e0bf965e180fbe72` | yes / yes | yes / yes / yes | 3 · 4 (2.9e-07) · 121 (3.0e-07) | 0 / 1 | 0.969 / 3.91e-14 | **PASS** |
| `bnext-tiny` | model | pass | `97b71ad2021f64fc` | `9361af60e1a56662` | yes / yes (yes) | yes / yes / yes | 37 · 21 (1.3e-06) · 70 (4.7e-07) | 2 / 8 | 0.930 / 5.68e-14 | **PASS** |
| `bnext-tiny` | model_sign_zero | pass | `97b71ad2021f64fc` | `718d2384016b381e` | yes / yes | yes / yes / yes | 37 · 21 (1.3e-06) · 70 (4.7e-07) | 2 / 8 | 0.930 / 5.68e-14 | **PASS** |
| `bnext-small` | model | pass | `2a5230ee787493bc` | `5c3a16fcd902836f` | yes / yes (yes) | yes / yes / yes | 60 · 37 (1.0e-06) · 31 (4.7e-07) | 2 / 4 | 0.969 / 5.68e-14 | **PASS** |
| `bnext-small` | model_sign_zero | pass | `2a5230ee787493bc` | `031eb7d92268e4e1` | yes / yes | yes / yes / yes | 60 · 37 (1.0e-06) · 31 (4.7e-07) | 2 / 4 | 0.969 / 5.68e-14 | **PASS** |
| `bnext-middle` | model | pass | `946ce650583d7b66` | `d40f5c1e12cef72b` | yes / yes (yes) | yes / yes / yes | 54 · 45 (1.7e-06) · 29 (4.2e-07) | 2 / 6 | 0.945 / 7.11e-14 | **PASS** |
| `bnext-middle` | model_sign_zero | pass | `946ce650583d7b66` | `d4a5bb0b114f364f` | yes / yes | yes / yes / yes | 54 · 45 (1.7e-06) · 29 (4.2e-07) | 3 / 6 | 0.945 / 7.11e-14 | **PASS** |
| `bnext-large` | model | pass | `94e7456160240ba3` | `5f8d80f879cf8895` | yes / yes (yes) | not run (memory) | - | - | - | identical, not verified |
| `bnext-large` | model_sign_zero | pass | `94e7456160240ba3` | `725a9e261aa4ceff` | yes / yes | not run (memory) | - | - | - | identical, not verified |
| `bitorch-resnete18-imagenet` | model | pass | `e5f9eb912f0aaed2` | `6f52c33bb4236b4f` | yes / yes (yes) | yes / yes / yes | 0 · 48 (1.2e-04) · 80 (4.2e-07) | 4 / 5 | 0.961 / 1.71e-13 | **PASS** |
| `bitorch-densenet28-imagenet` | model | pass | `2cbf621ded3dd9a6` | `6621165121351e64` | yes / yes (yes) | yes / yes / yes | 0 · 42 (1.5e-05) · 86 (8.3e-07) | 2 / 0 | 0.984 / 1.42e-13 | **PASS** |
| `bitorch-meliusnet22-imagenet` | model | pass | `53a94a34aa8ff8da` | `530db90fbb4d2175` | yes / yes (yes) | yes / yes / yes | 0 · 54 (7.1e-05) · 74 (2.7e-07) | 5 / 6 | 0.938 / 8.53e-14 | **PASS** |

BNext-L: the outputs comparison and the Level-3 run were not executed (memory); with data and graph SHA-256 identical,
the re-imported files are the published files byte for byte in everything the runtime reads.

## Findings

* **Byte identity holds through the profile.** The profile-conformant files (±1 weights + `Mul(alpha)`, explicit
  `Pad`, float64 carriers) import to the same bytes as the phase-2 files did: the importer folds the group's `Mul`
  into post_mul with the same single float32 rounding of the float64 value the legacy converters stored.
* **Legacy attribute forms.** Every PyTorch adapter writes `pad_value: 0.0` for every binary conv, also unpadded 1×1
  convs (where the FINN files have `0`): this was phase 2's ReActNet-A blocker; `bitino.attrs` carries it.
* **Legacy layouts that are not module paths** are carried by the naming metadata, not by importer special cases:
  `Graph.unique()` names and the counter continued by the passes (`bitino.ir_unique_counter`), four IR nodes from the
  two ONNX nodes of the RBNN / SiMaN centering (`extra_names`, `consts`), BNext's `scale` / `one_minus_scale`
  constants, bitorch's `_model.` paths.
* **Adapter choices that differ from the source** are now visible in the canonical file (each was already documented
  in its adapter and accepted by the legacy validation; the export check measures them): RBNN / SiMaN binarize
  `a − mean` without the positive divisor; AB-BNN binarizes `x + b/beta` instead of `(x + b)·beta`; the ReActNet-A BN
  flavour (BURN, S2-BNN) binarizes `q + fl64(b_prev + b_next)` (one rounding of a constant) where the source computes
  `fl(fl(q + b_prev) + b_next)`, written as `Add(q, carrier)` beside the residual path's `Add(q, b_prev)`; IDa-Det's
  ceil-mode average pool is written in floor mode (equal on its even maps).
* **torch's JIT CSE** merges identical subgraphs during `torch.onnx.export`; the legacy graphs keep them (RBNN /
  SiMaN CIFAR ResNet-18: the 1×1 shortcut conv centers the same input as conv1). It is disabled during the export.
* **Flip tracing through the passes.** A binarizer's IR carrier can be folded away by the passes (the CIFAR ResNet-18
  stem BN folded into conv1, whose output is binarized directly); the trace now re-reads the binary consumer's input
  after the passes (it crashed on ReCU-18 CIFAR before).
* **Memory.** The verification held four full evaluations per sample (OnnxF64 fp64, qonnx fp32, IR fp32 / fp64); it
  now keeps one at a time plus compact per-binarizer decisions, and `OnnxF64.run(keep=set)` frees every intermediate
  after its last use. BNext-L's canonical file is 1.3 GB: the exporter computes the source logits first (in chunks of
  4 samples, `--chunk`), frees the torch models before `check-qonnx`, and peaks at 5.9 GB. ImageNet imports with
  verification need 2–4 GB, BNext-M / -L more (run alone, ≤ 9 GB cap).

## Not done, and why

**Update (step 1b, [qonnx-step1b.md](qonnx-step1b.md)): all 16 are done** (28 files identical and verified). At the
time of this step each still had its legacy converter, unchanged. None is blocked by the profile itself; each needed
work that this step did not reach:

* **BBCU (6 networks) and BSRC-2023 (4).** Image-to-image networks. The adapters build PixelShuffle, nearest and
  bilinear upsampling from NHWC-specific subgraphs (`restoration_common`: 6-D `Reshape` / `Transpose`, row `Gather`
  with constant indices, a replicate pad by `Gather` and a 3×3 phase `Conv2D`); the importer handles tensors of rank
  2 and 4 only and has none of these patterns (`Resize` / `DepthToSpace` are not profile operators, so the canonical
  file must spell out the same standard-op subgraphs, and the importer must match them back). BSRC's LAB binarizer
  (`argmax` of two depthwise-conv channels → `gt` of their difference) and the `/k`, `/51` divisions written as
  products need the same treatment. The acceptance script also needs an image-output verification (PSNR against the
  float64 reference instead of top-1 agreement). The DnCNN pair (denoise, deblocking) needs none of the upsampling
  patterns and is the natural next one.
* **BNAS (4).** The adapter is not a module-by-module mirror: it moves `sign(s_bn)` of the BN before each sign into
  the next conv's weights (`sign(s·x + b) = sign(s)·sign(x + b/s)`), corrects structurally-zero channels for the `ge`
  variant with a data-independent `Add(−m·Σw)`, maps the BinOp "partial binarization" quirk convs to
  `Activation(sign_zero)` + float `Conv2D`, shares identical pools and drops dead cell branches. A hook has to
  reproduce each of these graph-level decisions; not attempted.
* **ReActNet-A BiMatting backbone (1).** The BN-flavour hook covers BURN and S2-BNN; BiMatting adds the
  `planes = inplanes / 2` block (`(out1[:, :P] + out1[:, P:]) / 2`, which the adapter writes as `Reshape [1, HW, 2, P]`
  + `AvgPool2D [1, 2]` + `Reshape` on the NHWC tensor: no importer pattern), centered `BiConv2d` weights and the
  macro residuals of `forward_single_frame`.
* **AdaBin (1).** Activations binarize to `{α(β−1), α(β+1)}` and weights to `{c − d, c + d}` per channel; the adapter
  decomposes each conv into two BConv2D, a 1×1 Conv2D and a constant border map. The profile's binarizers are `±s`;
  the canonical form (an affine of a `±1` binarizer, and the weight split) and its importer rule are open.

## Running it

The acceptance runs every network and variant listed in `NETWORKS` (`tools/qonnx_step1_acceptance.py`); `--only` /
`--variant` select, `--no-verify` checks identity only (export + check-qonnx + import, ~10–100 s per file).
`--cache-dir`, `--artifacts-dir` and `--bench` point at the checkout's `.cache`, `artifacts` (read only) and the host
`bitino_bench`. The step-1 run used shards in parallel under `scripts/capped.sh` (2–4 GB each, BNext-M and
MeliusNet alone at 6–7 GB; `--out` one JSON per shard, merged into the development log (not published)); the rows verified
before the Level-3 rule was written into the script were recomputed with `--skip-run` (same files, same references).
BNext-L: the export peaks at 5.9 GB; its import with verification exceeds a 9 GB cap, the identity-only import fits.

## Tests

`.cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_import.py tests/python/test_qonnx_profile.py` →
**90 passed** (35 in `test_qonnx_import.py`, 29 of phases 1–2 and 6 new; 55 in `test_qonnx_profile.py`, 1 new: `DFT`
is rejected since Q1 was decided). New in `test_qonnx_import.py`: float64 carriers (OnnxF64 fp64 / fp32, importer storage), the canonical binary
conv group (explicit `Pad(0)` → `pad_value 0.0`, `Mul(alpha)` with negative alpha and `Add(bias)` → post_mul /
post_add, a BN of another group kept), `bitino.attrs` (JSON form applied, a value change rejected), the IR-Net
centering (`ReduceMean` + `Sub` → the four legacy IR nodes, names from `extra_names`, constant from `consts`), the
option-A shortcut (`AveragePool(1, 2)` + channel `Pad`) and `--flatten-dense permute` vs the Transpose form, and
`OnnxF64.run(keep=set)` (same values, intermediates freed).

Phase 1 is unaffected: the 8 phase-1 `model_clean.onnx` files re-imported with this importer give the data and graph
SHA-256 of the development log (not published) (8 / 8). `test_onnx_dabnn.py`, `test_ir_and_format.py`,
`test_finn_bnn.py` pass.

## User decisions applied (2026-11-20)

* **Identity bar (Q11):** data + graph SHA-256 identity is sufficient (names byte for byte); whole-file / manifest
  identity is not required. Stated in docs/qonnx-profile.md, docs/qonnx-frontend.md, this report, the acceptance
  script's docstring and its row status (`PASS: data + graph SHA-256 identical (names included; manifest not
  compared), ...`).
* **FFT feature (Q1, step 3):** a double-precision DFT written as `MatMul` on cos / sin tables computed in float64,
  under the float tolerance contract (class F, not bit-exact); no unrolled exact FFT. docs/qonnx-profile.md § 4.5 gives
  the canonical subgraph; `check-qonnx` now rejects ONNX `DFT` with that replacement (it was a W-OPEN warning).
