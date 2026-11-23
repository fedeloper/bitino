# Converting a new BNN to Bitino

This guide is for contributors who have a trained binary network and want it to run in Bitino. It describes the
QONNX route, which is **the** build path since step 6 (2026-11-21): every published network is built as checkpoint →
canonical QONNX file → `bitino.qonnx_import` → `.bitino` by `tools/build_from_qonnx.py`; overview of the standard
in [qonnx-standard.md](qonnx-standard.md), the profile in [qonnx-profile.md](qonnx-profile.md), history in
[qonnx-frontend.md](qonnx-frontend.md). The IR-emitting `tools/convert_*.py` converters are legacy cross-checks; a new
network does not get one. (The phase-1 / phase-2 notes below remain accurate for the tools they describe; marks
**(planned)** refer to those phases and are all done.)

## Before you start

* **Weights and licence.** You need a checkpoint with its SHA-256, the model code that defines the network, and the
  licence of the *weights*. A converted file is redistributed only under MIT, Apache-2.0 or BSD weights; any other
  model is installed through a fetch recipe that downloads the authors' files. Add the model to the registry
  (`registry/`) with its paper, code, checkpoint URL and licences.
* **Operators.** Check that every layer maps to the kernel set in
  [qonnx-frontend.md § The kernel set](qonnx-frontend.md#the-kernel-set). Typical BNN CNNs (ReActNet, Bi-Real Net,
  QuickNet, XNOR-Net style) need only the binary, float and element-wise sets. If a layer has no pattern, see
  *Unsupported nodes* below.
* **Preprocessing.** Know the evaluation preprocessing exactly: resize size and filter, crop, mean/std, channel order.
  Bitino reproduces it bit for bit on Android, so "roughly ImageNet" is not enough.

## Step 1: export to QONNX

**Brevitas** models export directly:

```python
from brevitas.export import export_qonnx
model.eval()
export_qonnx(model, torch.randn(1, 3, 224, 224), "model.onnx")
```

**Plain PyTorch** models need the binarizer to appear in the graph as an ONNX operator. `torch.sign` exports as ONNX
`Sign` (`sign(0) = 0`). If your binarizer maps 0 to +1, as `x >= 0 ? 1 : -1` does, export it explicitly so that the
semantics survive:

```python
class BinarizeGE(torch.autograd.Function):          # forward only matters for export
    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def symbolic(g, x):                             # emits qonnx BipolarQuant(x, scale=1)
        s = g.op("Constant", value_t=torch.tensor(1.0))
        return g.op("qonnx.custom_op.general::BipolarQuant", x, s)
```

Replace the binarizer of activations *and* weights with it (or with `torch.sign`, if that is what the model trained
with). Then export:

```python
torch.onnx.export(model.eval(), torch.randn(1, 3, 224, 224), "model.onnx", opset_version=17,
                  input_names=["x"], output_names=["logits"], do_constant_folding=False)
```

`do_constant_folding=False` keeps `Bin(W_latent)` visible. Folded `±α` constants are also accepted, but keeping the
binarizer makes the graph self-documenting. Learned sign thresholds (RSign) should appear as `Sub(x, b)` or `Add(x, b)`
before the binarizer, and RPReLU as `Add` / `PRelu` / `Add`. Both are standard ONNX.

**`tools/export_torch_qonnx.py`** (phase 2) does all of this for the families that have an *export hook*: AB-BNN
ReActNet-18/34, RBONN, ReCU and BiPer (ImageNet ResNets). It loads the checkpoint exactly as the family's converter
does (SHA-256 checked), builds an exportable copy with the same module tree, exports it, and writes `export.json`
plus the source model's own fp32 / fp64 logits:

```bash
.cache/venv-qonnx2/bin/python -I tools/export_torch_qonnx.py --model-id ab-bnn-imagenet-reactnet18 \
    --repo .cache/repos/AB-BNN --weights .cache/checkpoints/ab-bnn/imagenet_reactnet18_step2.h5 \
    --eval-dir .cache/data/eval128 --out-dir .cache/export/ab-bnn-imagenet-reactnet18 [--binarizer ge]
```

(`.cache/venv-qonnx2`: torch 2.6.0 + onnx 1.17.0; the base env's torch cannot export without the `onnx` package.)
For a new family, add a hook (a `Family` subclass in that file). What a hook does, learned on the four families:

* **Binarizers.** Keep the source's: `torch.sign` → `Sign`. `--binarizer ge` exports `BipolarQuant` instead, the
  deployable `0 → +1` choice; it is recorded in `export.json` and is visible in the graph, never silent.
* **Binary weights** as `BipolarQuant(W_latent, scale[o])` with the source's per-channel scale *including its sign*;
  the importer then keeps the source's bit polarity (a constant `±α` would be imported with `α > 0` and flipped bits:
  the same outputs, a different file). Check that `W_latent` has no exact zero (`torch.sign` would give a 0 weight).
* **Constants derived from parameters** (weight standardization, `gain·mean|W|·α`, `bias / β`, `2^round(L)`) are
  computed in float64 by the adapter's own helpers and rounded once to float32 (the rule of the legacy converters).
* **BatchNorm** as `x * s + t` with `s, t` folded in float64: ONNX `BatchNormalization` stores `epsilon` as a float32
  attribute, which cannot hold PyTorch's double `eps = 1e-5`. On ReCU-18's head BatchNorm1d (small running
  variances) that changed 5 of 1024 folded constants by 1 ulp; exported as `Mul` / `Add` they are exact.
* **Pooling** that the adapter maps to `GlobalAvgPool` (e.g. `AvgPool2d(7)` on a 7×7 map) as `AdaptiveAvgPool2d(1)`.

**Keras / Larq** (planned, phase 4): tf2onnx, with Larq's `ste_sign` recognized as `BipolarQuant`. Until then, Larq
models use `tools/convert_larq_zoo.py`.

**Check the export before going further.** Compare the exported graph with your framework on a few real inputs:

```bash
.cache/venv-qonnx/bin/qonnx-cleanup model.onnx --out-file model_clean.onnx   # keeps weight quantizers by default
```

```python
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
m = ModelWrapper("model_clean.onnx")
y = execute_onnx(m, {m.graph.input[0].name: x_nchw})[m.graph.output[0].name]
```

(The `qonnx-exec` command needs `tqdm`, which `.cache/venv-qonnx` does not have; the Python API does not.)

The top-1 class should match the framework's on almost every image. A few flips are possible, from activations within
rounding distance of zero. Many more mean the export changed the semantics, most often `sign(0)` or the padding.

## Step 2: import

```bash
.cache/venv-qonnx/bin/python -I tools/import_qonnx.py model_clean.onnx \
    --model-id <registry id> --preprocess <eval preprocessing> \
    --eval-dir .cache/data/eval128 --out-dir .cache/import/<registry id>
# or: bitino import-qonnx --python .cache/venv-qonnx/bin/python model_clean.onnx --model-id ... --out-dir ...
```

(`--help` lists the rest: input domain, cleanup, naming, the stages to check.) Brevitas models can be exported
and imported in one go with `tools/export_brevitas_qonnx.py`. Phase 1 covers the binary, float, element-wise and
pooling operators; ternary weights, ChannelGate, runtime-tensor Mul/MatMul, ReduceMean, LayerNorm, Gather and the
Erf form of GELU still stop the import with a message (later phases).

Three options give the file layout of the legacy PyTorch converters; `tools/export_torch_qonnx.py` records them in
the file itself (model metadata `bitino.import {"mapping": "patterns", ...}`, step 6; also in `export.json`
`import_args`), so the canonical file is imported with **no** layout argument (one that contradicts the file is an
error). None changes a computed value:
* `--naming exporter`: node names from the exporter's `metadata_props` (`bitino.name`: the torch module path), which
  survive qonnx cleanup's renaming; the walk also follows the exporter's node order (`bitino.order`). A hook can
  supply the legacy converter's own names (the IR-Net hook does: `nonlinear_1`, `layer1.0.add1`, ...) and the start
  of the graph's unique-name counter (model metadata `bitino.ir_unique_counter`), so that a migrated family keeps
  byte-identical files.
* `--affine keep`: one `ChannelAffine` per affine node of the source (BN, `Add` / `Mul` by a per-channel constant;
  `Mul` + `Add` as one), i.e. the source's float32 rounding steps, folded afterwards by the existing passes. The
  default (`fold`) accumulates them in float64 into the next epilogue or threshold.
* `--constant-order first-use`: the data section in first-use order (the converters' insertion order).

The importer:
1. maps every node to an IR operator by the pattern rules, or stops with the list of unmatched nodes and the reason
   (e.g. "Conv weight channel 17 is not ±α");
2. converts NCHW to NHWC, folds BatchNorm and the per-channel affine layers into thresholds and epilogues, and packs
   the binary weights;
3. runs the existing graph passes and writes `model.bitino` with its provenance manifest (the checkpoint and ONNX
   SHA-256, the exporter and its version, the preprocessing);
4. runs the checks of step 3 and writes `conversion_report.json`.

## Step 3: verify

The checks are the existing ones (`docs/numerical-correctness.md`), against new references:

* QONNX graph in float32 (`execute_onnx`) and in float64 (the numpy evaluator) vs the converted graph (numpy executor)
  vs the C++ runtime's exact stage, which must be bit-identical with the numpy executor;
* a sample that deviates from float64 is traced to its first binarization flip. It is accepted only at a
  rounding-scale pre-activation (≤ 1e-5 against typical 0.5–1);
* pass: top-1 agreement with float64 ≥ 98 %, and at most the source's own float32 disagreements + max(1, 1 %).

Then validate on a phone: `tools/validate_model.py` and `scripts/device_final_suite.sh`, as for every model.

## Step 4: full-integer file, speed, recipe

* `tools/quantize_int8_full.py` builds `model_int8_full` from the float file, with calibration images disjoint from
  every evaluation set. Report its accuracy change on the larger evaluation set before shipping it.
* Benchmark at 1 thread on a pinned core (`tools/benchmark.py`, `docs/benchmarking.md`).
* Add the fetch recipe (`registry/recipes.json`): the download URL and SHA-256, the code commit, the exporter
  environment, and the expected SHA-256 of the result. `bitino fetch` must rebuild the identical file from an empty
  cache.

## Unsupported nodes

When the importer stops on a node, in this order:
1. **Rewrite it in supported ONNX.** For example, an SE block as `GlobalAveragePool` → `Conv`/`Gemm` → `Sigmoid` →
   `Mul`, or a custom normalization as its arithmetic.
2. **Add a pattern** to `qonnx_import.py`, when the node is a standard op the runtime can already run as a
   composition.
3. **Add an operator** (`docs/operators.md` § Adding an operator): contract, reference kernel, numpy executor, tests,
   then optimized kernels. There is no private ONNX domain: the QONNX file expresses the new operator as a subgraph
   of standard ONNX + QONNX operators that the importer matches onto it, and
   [qonnx-profile.md](qonnx-profile.md) gets the pattern and its exactness (a profile revision).

Check a file against the profile with `bitino check-qonnx model.onnx` (or
`.cache/venv-qonnx/bin/python -I tools/check_qonnx.py model.onnx`).

## Families the QONNX route does not cover yet

Transformers (phase 3), Keras / Larq (phase 4) and PyTorch models whose layers have no pattern yet (e.g. activation
centering by a per-sample `ReduceMean`, SE gates, ternary weights): copy the closest `tools/convert_*.py`. A PyTorch
family: `convert_abbnn.py` or `convert_rbonn.py`; Brevitas: `convert_brevitas.py`; QONNX/FINN:
`convert_finn_onnx.py`. Each builds the IR with `python/bitino` adapters and runs the same checks. The kernel set
and the semantic rules above already apply. Writing your network so that it *would* export cleanly to QONNX keeps the
later migration free.
