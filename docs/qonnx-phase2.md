# QONNX route, phase 2: acceptance report (plain PyTorch)

Plan: [qonnx-frontend.md § Plan](qonnx-frontend.md#plan), phase 2. Date: 2026-11-20, host only (x86-64), nothing under
`artifacts/` was written. Machine-readable results: the development log (not published). Exports, re-imported files
and reports: `.cache/qonnx_phase2/<id>/<variant>/`. Command to reproduce everything:

```bash
scripts/capped.sh 8G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_phase2_acceptance.py
```

## What was built

| file | role |
|---|---|
| `tools/export_torch_qonnx.py` | the exporter: loads a checkpoint exactly as the family's converter does (`tools/convert_*.py` `load_model`, SHA-256 checked), builds an *exportable copy* with a per-family **export hook**, `torch.onnx.export` (opset 17, `do_constant_folding=False`), records node metadata for the importer, writes `model.onnx`, `export.json` and the source model's own fp32 / fp64 logits on the eval set |
| `python/bitino/qonnx_import.py` | importer extensions (below); importer version `qonnx_import 2 (phase 2)` |
| `tools/qonnx_phase2_acceptance.py` | this acceptance run (export, import + verification + C++, comparison with the published files) |
| `tests/python/test_qonnx_import.py` | 7 new tests on synthetic graphs (29 in total) |
| `.cache/venv-qonnx2` | new venv (`python -m venv`, no system site packages): torch 2.6.0+cpu, torchvision 0.21.0+cpu (the versions of the recorded PyTorch conversions), numpy 2.2.6, onnx 1.17.0, timm 1.0.27 (RBONN's source imports it). Freeze: `envs/qonnx2-freeze.txt`; env `qonnx2` in `registry/recipes.json`. Needed because `torch.onnx.export` imports the `onnx` package, which the torch env (base conda) does not have |

Pipeline per network and variant: authors' model (read-only clone + checkpoint) → export hook → `torch.onnx.export` →
`qonnx.util.cleanup.cleanup_model(preserve_qnt_ops=True)` → import (`--naming exporter --constant-order first-use
--affine keep`, from `export.json`) → `passes.optimize` → `model.bitino` + `conversion_report.json` → verification as
in phase 1 (qonnx `execute_onnx` fp32 and `OnnxF64` fp64 of the cleaned graph vs the numpy executor of the written
file, first binarization flip, the source framework's own logits vs the export, raw vs cleaned graph, and the C++ host
runtime `tools/validate_model.py --require-exact-numpy`, stage `reference`, `build/host/bitino_bench` rebuilt from this
branch), then comparison with the published file: data and graph SHA-256, and the outputs of both files on the check
set (numpy executor, C++ stages `reference` and `optimized`), bit for bit.

### Export hooks

A hook mirrors the legacy adapter of its family (`bitino.adapters.torch_abbnn`, `torch_rbonn`, `torch_irnet`): it
checks every module type it meets (an unknown one stops the export), keeps the module tree (so every ONNX node is named
after its module path) and replaces each module by one whose forward is plain ONNX:

| source construct | exported as | why exact |
|---|---|---|
| `torch.sign` activation binarizer (all four families) | ONNX `Sign` (variant `sign_zero`) or, with `--binarizer ge`, `BipolarQuant(x, 1)` (variant `ge`) | `Sign` is the source; `ge` is the legacy converters' deployable choice, recorded in `export.json` and visible in the graph |
| AB-BNN `sign((x + b/β)·β)` (LearnableBias + scaled sign) | `Add(x, fl32(b/β))` → binarizer | β > 0; the legacy adapter drops β the same way (differs only if `fl(x + b/β)` is subnormal) |
| binary weights (`HardBinaryScaledStdConv2d`, `RBOConv`, `BinarizeConv2d`) | `BipolarQuant(W_latent, scale[o])`: AB-BNN `W_std`, scale `gain·mean|W_std|·α`; RBONN `W`, `0.02·|s|`; ReCU / BiPer the source's own fp32 `bw` (±1), scale `α` | no exact zero in `W_latent` (checked), so `BipolarQuant` = `torch.sign`; the scale keeps its sign |
| constants derived from parameters (weight standardization, scales, `b/β`, `2^round(L_α)`) | float32 initializers computed in float64 by the adapters' own helpers, rounded once | the legacy converters' rule |
| `BatchNorm2d` / `BatchNorm1d` (eval) | `Mul(x, s)` + `Add(t)`, `s, t` folded in float64 as the adapters do | ONNX `BatchNormalization.epsilon` is float32 and cannot hold PyTorch's double `1e-5` (see Findings) |
| `Q_PReLU` / `nn.PReLU` | `PRelu` with the float32 slopes (`2^round(L_α)`, exact powers of two) | |
| `AvgPool2d(7)` on the 7×7 map (IR-Net head) | `AdaptiveAvgPool2d(1)` → `GlobalAveragePool` | same window; the adapter's `GlobalAvgPool` |
| `x.view(N, -1)` | `torch.flatten(x, 1)` → `Flatten` | |

### Importer extensions (all covered by synthetic tests)

* **`--naming exporter`**: IR node names from node `metadata_props` (`bitino.name`), which `export_torch_qonnx.py`
  writes (module path of a leaf module, `<module path>.<op>` for an op in a module's own forward, e.g.
  `layer1.0.add`) and which survive qonnx cleanup's `GiveUniqueNodeNames`. When every node carries `bitino.order`,
  the walk follows the exporter's node order (cleanup's shape inference reorders independent branches, e.g. the
  downsample before `move0`). A hook can also write the legacy converter's own names (the IR-Net hook:
  `nonlinear_1`, `layer1.0.add1`, ...) and the start value of the IR graph's unique-name counter (model metadata
  `bitino.ir_unique_counter`), which the passes' constant names continue (`conv1/folded_18_w`).
* **`--affine keep`**: every affine source node (BatchNormalization, `Add` / `Sub` / `Mul` / `Div` by a per-channel
  constant) becomes its own `ChannelAffine` at its position (bias-only for Add/Sub, scale-only for Mul/Div; a `Mul`
  whose only consumer is a constant `Add` merges with it into one `ChannelAffine(scale, shift)` = `fl(fl(x·s) + t)`);
  a linear operator's epilogue closes at its weight scale; binarizers binarize at 0. These are the float32 rounding
  steps of the source and the layout of the legacy PyTorch converters; `passes.fold_affine_into_conv` then folds what
  they folded. A `Div` by a constant whose reciprocal is inexact is rejected in this mode. The default (`fold`) is
  unchanged from phase 1.
* **`--constant-order first-use`**: the data section in first-use order before the passes (the converters'
  insertion order; the passes append their folded constants at the end, as for the legacy files).
* **Signed weight scale**: a binary-weight producer `BipolarQuant(W_latent, scale[o])` gives bits = `Bin(W_latent)`
  and α = `scale` *with its sign*. The plain `±α` rule (phase 1) takes α = |w| and flips the bits of a channel with a
  negative scale: the same outputs bit for bit (`(-acc)·(-m) = acc·m`), but not the legacy file (AB-BNN has negative
  `gain` on some channels).
* Layout rules that phase 1 never met: a bias-only `ChannelAffine` (scale input empty) when every scale is 1;
  `GlobalAveragePool` followed only by a flatten → `GlobalAvgPool(keepdims 0)` without a `Reshape`; `Add` of runtime
  tensors without attributes; `post_add` only when non-zero; `pad_value` `0.0` (float) for Conv zero padding of a
  binarized tensor (`0` when the layer has no padding, as in the FINN files).
* **An affine value with several consumers is materialized once** (before: each consumer chain could merge it with
  its own constants in float64, i.e. `x + fl(c1 + c2)` instead of `fl(fl(x + c1) + c2)`).

Phase 1 is unaffected: the 8 phase-1 ONNX files re-imported with this importer give the same data SHA-256 (= recipe)
and graph SHA-256 as in the development log (not published).

## Acceptance table

Data SHA-256 = `data_sha256` of the data section, graph SHA-256 = `fetch.graph_sha256` (constants table with names,
inputs, nodes with names, outputs; not the manifest). Published = `artifacts/<id>/<file>`; for `model.bitino` it equals
the recipe's `expected` hashes (checked, all 4); `model_sign_zero.bitino` has no recipe entry and is compared with the
published file. Outputs = all 128 samples of the model's check set (`eval128`, `torchvision_imagenet`), numpy executor
(fp32) and C++ runtime stages `reference` and `optimized`, published vs re-imported file, bit for bit.

| network | variant (published file) | exported ONNX SHA-256 | data SHA-256 (re-import) | data = published | graph SHA-256 re-import / published | graph = | outputs bit-identical numpy / C++ ref / C++ opt |
|---|---|---|---|---|---|---|---|
| `ab-bnn-imagenet-reactnet18` | ge (`model.bitino`) | c8e5b529e156 | `ecce5fe851958c06` | **yes** | `8fd2a98bd119c235` / `8fd2a98bd119c235` | **yes** | yes / yes / yes |
| `ab-bnn-imagenet-reactnet18` | sign_zero (`model_sign_zero.bitino`) | 6ca231fff3bb | `ecce5fe851958c06` | **yes** | `888e734a95b0e9c6` / `888e734a95b0e9c6` | **yes** | yes / yes / yes |
| `rbonn-reactnet-resnet18-imagenet` | ge (`model.bitino`) | 4c33eb55ca48 | `c6bdc47b26fe6784` | **yes** | `14d7eb5f7845faad` / `14d7eb5f7845faad` | **yes** | yes / yes / yes |
| `rbonn-reactnet-resnet18-imagenet` | sign_zero (`model_sign_zero.bitino`) | e38252f7b543 | `c6bdc47b26fe6784` | **yes** | `7ad97ac3b034a0fa` / `7ad97ac3b034a0fa` | **yes** | yes / yes / yes |
| `recu-resnet18-imagenet` | ge (`model.bitino`) | 5d7b6d15c55c | `15848412289bd57f` | **yes** | `2b3d61bcc3bcdb55` / `2b3d61bcc3bcdb55` | **yes** | yes / yes / yes |
| `recu-resnet18-imagenet` | sign_zero (`model_sign_zero.bitino`) | 8f159c4fed74 | `15848412289bd57f` | **yes** | `31b75adb9d274afd` / `31b75adb9d274afd` | **yes** | yes / yes / yes |
| `biper-resnet18-imagenet` | ge (`model.bitino`) | dce0cbb6bb94 | `a5f2c2ad0720c4b3` | **yes** | `2b3d61bcc3bcdb55` / `2b3d61bcc3bcdb55` | **yes** | yes / yes / yes |
| `biper-resnet18-imagenet` | sign_zero (`model_sign_zero.bitino`) | 1ac33d8beb81 | `a5f2c2ad0720c4b3` | **yes** | `31b75adb9d274afd` / `31b75adb9d274afd` | **yes** | yes / yes / yes |

(Full 64-digit hashes in the development log (not published). ReCU-18 and BiPer-18 share their graph SHA-256: the same
topology and names, different weights.)

**Result: 8 / 8 files byte-identical in data and graph** (4 networks × 2 binarizer variants), names included; every
`model.bitino` matches its recipe's expected hashes. Two more AB-BNN files were checked by hash only (export + import,
no verification run): `ab-bnn-cifar10-reactnet18` and `ab-bnn-imagenet-reactnet34`, both variants, also identical in
data and graph.

### Verification of each re-import (Level 3 + C++)

| network / variant | IR f64 vs QONNX f64: max abs, samples ≤ 1e-6 | samples with a binarization flip vs the QONNX graph (max \|pre-activation\| fp32 / fp64) | QONNX fp32 vs fp64 top-1 disagreements | export check: source vs QONNX, fp32 top-1 agreement / fp64 max abs | acceptance | C++ reference = numpy |
|---|---|---|---|---|---|---|
| AB-BNN ge | 1.2e-13, 128/128 | 16 (7.0e-06 / –) | 1 | 0.992 / 2.61 | pass | yes |
| AB-BNN sign_zero | 1.2e-13, 128/128 | 21 (7.0e-06 / –) | 0 | 1.000 / 2.61 | pass | yes |
| RBONN ge | 1.76, 16/128 | 12 (4.5e-07 / 3.1e-09) | 0 | 0.984 / 6.58 | pass | yes |
| RBONN sign_zero | 1.76, 16/128 | 15 (4.5e-07 / 3.1e-09) | 0 | 0.984 / 6.58 | pass | yes |
| ReCU ge | 1.90, 123/128 | 16 (1.3e-06 / 4.1e-08) | 0 | 1.000 / 1.46 | pass | yes |
| ReCU sign_zero | 1.90, 123/128 | 16 (1.3e-06 / 4.1e-08) | 0 | 1.000 / 1.46 | pass | yes |
| BiPer ge | 1.32, 97/128 | 21 (4.8e-07 / 1.5e-08) | 2 | 0.984 / 1.93 | pass | yes |
| BiPer sign_zero | 1.32, 97/128 | 24 (4.8e-07 / 1.5e-08) | 2 | 0.984 / 1.93 | pass | yes |

How to read it. The re-imported files *are* the published files, so these numbers are the published files' own
numerical behaviour against the new references. AB-BNN folds nothing after the import, and the IR equals the QONNX graph
in float64 to 1e-13. RBONN, ReCU and BiPer: `passes.fold_affine_into_conv` folds each BatchNorm into the binary conv's
epilogue with one float32 rounding of `post_mul · s` (as the legacy converters did), which the QONNX graph applies as
two separate float32 constants; in float64 that moves pre-activations by ~1e-8 and flips a binarization in a few
samples (the fp64 flips are at |pre| ≤ 4e-8, rounding scale), whence the larger logit differences on those samples.
Every fp32 flip is at |pre| ≤ 7e-6 against typical values 0.4–0.8. Acceptance rule of `numerical-correctness.md`
(top-1 agreement with fp64 ≥ 98 %, disagreements ≤ fp32 noise floor + max(1, 1 %)): passed for all 8. The export
check compares the authors' unmodified model with the exported graph: in float64 its differences are the float32
storage of the folded constants (for AB-BNN exactly the legacy converter's own semantic-check value, 2.612265; RBONN
6.58, ReCU 1.46, BiPer 1.93 are likewise the legacy reports' values for the unfused files); `qonnx` cleanup changes
nothing (raw vs cleaned graph in fp64: 0.0).

## Findings

* **The binarizer variant is an explicit choice.** The four families binarize with `torch.sign` (`sign(0) = 0`). The
  legacy converters publish the deployable `ge` file (`0 → +1`, `model.bitino`, the recipe's expected file) and the
  exact `sign_zero` file (`model_sign_zero.bitino`, reference kernels only). The QONNX route reproduces both, from
  two exports: `Sign` (the source) and `BipolarQuant` (`--binarizer ge`, recorded in `export.json` and visible in the
  graph). The importer never turns one into the other. The two differ only for activations exactly 0 at a
  binarization point: the fp32 AB-BNN source has 82 of them on the 128 check images (0 in fp64).
* **ONNX BatchNormalization cannot carry PyTorch's epsilon.** `epsilon` is a float32 attribute; `float32(1e-5) =
  9.99999974737875e-06`, PyTorch (and the legacy fold) use the double `1e-5`. Exported as `BatchNormalization`, ReCU-18
  imported with 5 of its 1024 head-BatchNorm1d constants 1 ulp off (`bn2/scale` 1 value, `bn2/shift` 4; the running
  variances are small there). The hooks therefore export BatchNorm as `Mul` + `Add` of float64-folded constants; RBONN
  happened to be unaffected either way.
* **Weight polarity.** AB-BNN has channels with a negative `gain` (77 of the 512 output channels of `layer4.0.conv1`).
  The phase-1 `±α` rule imports them as `α = |w| > 0` with flipped bits: bit-identical outputs, different data. The
  signed-scale rule on `BipolarQuant(W, scale)` keeps the source's polarity and gives the legacy bytes.
* **qonnx cleanup reorders nodes** (the downsample branch moved before the block's `move0`) and renames them; both
  are undone by the exporter's node metadata (`bitino.order`, `bitino.name`), which cleanup keeps.
* **Float32 roundings of affine chains.** Phase 1 accumulated consecutive per-channel affines in float64 and, for an
  affine value used twice (ReActNet's `move2` feeds the next `move0` and the residual), let each consumer chain merge
  it with its own constants (`x + fl(c2 + c0)` instead of `fl(fl(x + c2) + c0)`). That is now materialized once in
  both modes; `--affine keep` keeps every source rounding step.
* **AB-BNN's scaled sign.** `sign((x + b/β)·β)` is exported as `Sign(x + fl32(b/β))`, as the legacy adapter computes
  it (β > 0). This equals the fp32 source except where `fl(x + b/β)` is subnormal; `b/β` is rounded once from float64,
  the fp32 source computes it in float32 (part of its noise floor; the f64 export check, max |Δ| 2.6, is exactly the
  legacy converter's own semantic-check value 2.612265, from the float32 storage of the constants).
* **The C++ optimized stage** is not bit-exact with the numpy executor on these ResNets, for the published files as
  well (the legacy validation accepts it through first-flip tracing). The import checks stage `reference` with
  `--require-exact-numpy`; the acceptance compares the optimized outputs of the published and the re-imported file,
  which are bit-identical.
* **torch 2.6's `torch.onnx.export` imports `onnx`** even for the TorchScript exporter
  (`onnx_proto_utils._add_onnxscript_fn`), so the converters' env (base conda, no `onnx`) cannot export; hence
  `.cache/venv-qonnx2` (no package was added to any existing env).
* **ReCU / BiPer graph names.** `adapters.torch_irnet` names the Hardtanh activations with `Graph.unique()` (a global
  counter: `nonlinear_1`, `layer1.0.nonlinear1_2`, ...), the residual adds `add1` / `add2`, and the counter continues
  into the fold pass's constant names (`conv1/folded_18_w`). The ReCU / BiPer hook computes these names
  (`LegacyIrnetNames`, the adapter's rule in forward order) into `bitino.name` and stores the counter's final value
  (17 for ResNet-18) in the model's `metadata_props` (`bitino.ir_unique_counter`, standard ONNX metadata, kept by
  qonnx cleanup); with `--naming exporter` the importer starts the IR graph's counter there. Without this the graphs
  were identical only up to names.

## Tests

`.cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_import.py` → **29 passed** (22 of phase 1, 7 new):
the legacy unique-name counter continuing into the fold pass's constant names (`conv1/folded_18_w`),
exporter names / order / PyTorch layout on a ReActNet-like block (incl. exact zeros at the `Sign`), the signed
BipolarQuant weight scale vs the folded `±α` constant (bit-identical outputs, different polarity), `--affine keep` vs
`fold` (same binary decisions at exact threshold values, `Mul + Add` merged, BN kept), a multi-use affine rounded once
(bit-exact against the source in float32), the inexact `Div` rejection, and the exporter metadata through qonnx
cleanup (same IR graph before and after). `test_onnx_dabnn.py`, `test_zoo_cli.py`, `test_ir_and_format.py`,
`test_finn_bnn.py` still pass (38).

## Remaining gaps vs phase 2 of qonnx-frontend.md

Superseded by [qonnx-step1.md](qonnx-step1.md) (2026-11-20): the exporter now writes canonical profile files, `tools/qonnx_step1_acceptance.py` re-verifies these four networks with them, and BNext, ReActNet-A (except BiMatting), RBNN / SiMaN, SA-BNN, SNN, IDa-Det, bitorch and the other IR-Net / AB-BNN members are done. The list below is the state at the end of phase 2.

Covered: ReActNet / AB-BNN, Bi-Real / RBONN (RBONN's network is the Bi-Real ResNet-18, class `BiRealNet`), ReCU and
BiPer (IR-Net family). Not attempted in this phase (each still has its legacy converter, unchanged):

* **BNext** (`torch_bnext`): squeeze-and-expand gates (`ChannelGate`) and the ELM attention's `x * residual` (runtime
  `Mul`) have no importer pattern yet (phase-1 list); the info-recoupling mix is two `ChannelAffine` + `Add`.
* **AdaBin** (`torch_adabin`): activations binarize to the adaptive set `{α(β-1), α(β+1)}` and weights to
  `{c - d, c + d}` per channel; the legacy adapter decomposes the conv into two BConv2D, a 1×1 Conv2D and a border
  map. The importer only accepts `±s` binarizers and `±α[o]` weights, so this needs a dedicated rule.
* **ReActNet-A** (AB-BNN `Reactnet`, BURN, S2-BNN, BiMatting): the hook is not written. Two known points: the
  `--affine keep` layout of the BN flavour's refolded thresholds (`q + fl64(b_prev + b_next)`, a deliberate deviation
  of the legacy adapter from the source's two roundings) can only be expressed by exporting that constant, and the
  legacy files write `pad_value 0.0` for unpadded 1×1 binary convs while the importer writes `0` (as the FINN files
  need): a difference in an attribute's JSON type only, but one that changes the graph SHA-256, so byte identity will
  need a serialization rule tied to the PyTorch layout.
* **RBNN / SiMaN** (IR-Net family with per-sample activation centering: `ReduceMean` over C·H·W feeding the
  binarizer): `ReduceMean` and `AddBroadcast` patterns are not in the importer yet.
* **SA-BNN, SNN, IDa-Det, bitorch, BNAS, BBCU, BSRC, transformers**: not attempted (the last ones are phase 3).
* The other members of the covered families (ReCU / BiPer ResNet-34, CIFAR models) are not in the acceptance run;
  their hooks are the same classes, but the CIFAR IR-Net ResNets use other block types the hook does not accept yet.
* The int8_full files (`model_int8_full*.bitino`) are produced from the float file by `tools/quantize_int8_full.py`;
  this phase reproduces the float files only.
