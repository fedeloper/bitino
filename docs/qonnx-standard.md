# The QONNX standard of Bitino

Status: **in force since step 6 (2026-11-21).** Every published Bitino network file (the 195 float files and the 101
int8_full files of the 132 networks) is built as

```
source checkpoint ──(front-end writer)──► canonical QONNX file ──(bitino.qonnx_import)──► .bitino
                                                    │
                                                    └──► derived files (*_unfused, the detector sub-models)
```

by one entry point, `tools/build_from_qonnx.py`, and every rebuilt file has the published data and graph SHA-256.
The IR-emitting converters `tools/convert_*.py` are no longer the build path; they are kept, unchanged, as legacy
cross-checks (`tools/qonnx_legacy_crosscheck.py`). Nothing here is upstream (the profile is a profile of QONNX, not a change to it). History and acceptance
reports: [qonnx-frontend.md](qonnx-frontend.md), steps [1](qonnx-step1.md), [1b](qonnx-step1b.md), [2](qonnx-step2.md),
[3](qonnx-step3.md), [4](qonnx-step4.md), [5](qonnx-step5.md), [6](qonnx-step6.md).

## The parts

| part | where | what it fixes |
|---|---|---|
| **profile** `bitino-qonnx/1` | [qonnx-profile.md](qonnx-profile.md) | the file format: standard ONNX (`ai.onnx` 17–21) + the QONNX quantizers `BipolarQuant` / `IntQuant` only, no custom domain, no local functions; the pattern of every Bitino IR operator (float § 4, int8_full § 5) with its exactness class (B bit-exact, F float contract); the semantic rules (sign(0), explicit binary padding, NCHW, ±1 weights + `Mul(α)`, half-even rounding, BatchNorm folded in float64 with float64 carriers); the model metadata (§ 6) and the naming / order metadata that carries byte identity (§ 7); the verification contract (§ 8) |
| **checker** | `python/bitino/qonnx_profile.py`, `tools/check_qonnx.py`, `bitino check-qonnx` | the form of a file against the profile (container, operators, attributes, binarizers and padding, weights, metadata, names); exit 0 / 1 / 2. Every writer refuses a file with a checker error (the int8 writer also with a warning) |
| **writers** (one per front end) | `tools/export_torch_qonnx.py` (PyTorch CNNs, steps 1 / 1b), `tools/export_keras_qonnx.py` (Keras / Larq, step 2), `tools/export_groups_qonnx.py` (transformers and the image-feature detectors, step 3), `tools/export_ir_qonnx.py` + `tools/qonnx_ir_writer.py` (OpenVINO, daBNN, NNCF, MindSpore, tensorpack, Theano / Lasagne, Brevitas, FINN, step 4), `tools/export_int8_qonnx.py` + `python/bitino/qonnx_int8.py` (int8_full files, step 5) | source → canonical file. Each checks the checkpoint SHA-256 against `registry/models.json` and writes the § 6 metadata, including `bitino.import` |
| **importer** | `python/bitino/qonnx_import.py` (pattern mapping), `python/bitino/qonnx_groups.py` (group mapping), `python/bitino/qonnx_int8.py` (int8_full groups); CLI `tools/import_qonnx.py`, `bitino import-qonnx` | canonical file → `.bitino`. **The file determines the result alone**: `bitino.import` says how (`patterns` with its layout options, or `groups`), `bitino.input_domain` the input domain; an import argument that contradicts the file (or, for an option the file does not list, the importer default) is an error, and a group-mapped file takes no import argument at all |
| **routes** | `registry/qonnx_routes.json` (generated: `tools/build_from_qonnx.py --write-routes`) | per recipe: the front end, and per file the writer, its environment and arguments, or the derivation of a derived file; the legacy converter is recorded as cross-check only |
| **entry point** | `tools/build_from_qonnx.py --model-id ID [--file F] \| --all` | runs the route: writer, importer (no import argument), derivations, and the identity comparison with the published file |
| **user build path** | `bitino fetch ID` (`python/bitino/fetch.py`; recipes `registry/recipes.json` from `tools/build_recipes.py`) | the same route from the authors' downloads (SHA-256 checked) in the user's reference environments, the canonical QONNX files kept in the model directory, every file compared with the recipe's data and graph SHA-256 (below) |

Two mappings exist. **Pattern-mapped** files (steps 1, 1b, 2, 4) are the source graph in standard operators; the
importer recognizes each IR operator's pattern, then runs the graph passes (`passes.optimize`: affine and ReLU folds,
pruning) and the layout options of `bitino.import`. **Group-mapped** files (steps 3, 5) carry one ONNX group per IR node
of the deployed graph (`bitino.ir`); the importer reads every group back and, for int8_full, re-emits it and requires
the file's nodes to be exactly the canonical pattern; no pass runs after the import.

## The identity bar

A rebuilt `.bitino` must have the **data SHA-256 and the graph SHA-256** of the published file (`artifacts/<id>/<file>`;
for the files in `registry/recipes.json` → `expected` also the recipe's hashes): every constant byte in the same order
and alignment, and the same graph JSON (constants table with names, dtypes, shapes, offsets, layouts; inputs; nodes with
op, name, inputs, outputs, attributes; outputs), names byte for byte. The provenance manifest is outside the graph hash
and not compared (design decision 2026-11-20, profile Q11). Byte identity is carried by metadata that never changes a
value (§ 7: `bitino.name`, `bitino.order`, `bitino.consts`, `bitino.attrs`, `bitino.const_order`, ...).

The numerical bar is separate and was established per step (the step reports): float files Level 3 of
`docs/numerical-correctness.md` against the float64 reference of the QONNX graph; int8_full files bit for bit (the
canonical file in onnxruntime `ORT_DISABLE_ALL` = the C++ `int8_full` outputs on the check set). The fp32 reference of a
canonical file is one onnxruntime session with `ORT_DISABLE_ALL`; qonnx `execute_onnx` is the cross-check (profile § 8).

## Derived files

The `*_unfused.bitino` files are **not derivations of the published files**: they are the legacy converters' graphs
before `passes.optimize(fold=True)`, and the fold passes are lossy (float32 products), so they cannot be recovered from
the fused `.bitino`. They are built from the QONNX route in one of two ways (`registry/qonnx_routes.json` records which):

| derived file | derivation | when |
|---|---|---|
| `model_unfused.bitino`, `model_sign_zero_unfused.bitino` | the parent's canonical file imported without the fold passes (`tools/import_qonnx.py --no-fold`): the importer's graph before the passes | the importer's pre-pass graph is the converter's (IR-Net family, AB-BNN, ReActNet-A, RBONN / SNN / IDa-Det, SA-BNN, bitorch, BiMatting, BBCU super-resolution, daBNN, NNCF, FDA-BNN, tensorpack, Larq DoReFaNet) |
| the same files | a canonical file of their own: the converter's graph before its fold passes, rebuilt from the checkpoint with the legacy adapter (checked equal to the published file) and written group-mapped (`tools/export_unfused_qonnx.py`; the step-3 networks: `tools/export_groups_qonnx.py --file *_unfused.bitino`) | the legacy adapter's own pre-pass graph has another layout than the importer's: ReLUs the adapter fused itself (BBCU denoise / deblocking, BNAS, Keras / Larq), the data section in the adapter's creation order (BSRC, AdaBin, BNext, ShuaiZ Bi-Real-20), the literal BatchNorm + sign mirror of the FINN-style adapters (Brevitas, FINN, BNN-PYNQ), the unmerged OpenVINO affine chains, or a group-mapped parent (step 3) |
| `model_features.bitino`, `model_network.bitino` (three COCOFake detectors) | `tools/ftl_deepfake_submodels.split` of the QONNX-built `model.bitino` | the two stages of the detector, node for node |

[qonnx-step6.md](qonnx-step6.md) lists the result per file and the derived files whose legacy layout needed more than
these rules.

## `bitino fetch`: the user's build path

Since 2026-11-23 the fetch recipes build through QONNX too (before, they ran the legacy converters). A recipe
(`registry/recipes.json`, generated by `tools/build_recipes.py` from the registry, the published files and the route
table) has `"build": "qonnx"` and:

| key | content |
|---|---|
| `downloads`, `sources` | unchanged: the authors' files with SHA-256, the model code at the recorded commit |
| `steps` | the route, in the recipe's reference environments (`envs`), `python -I`: per published / own-canonical file a **writer step** (`role: qonnx-writer`; the route's writer with the fetch placeholders: `{weights}`, `{download:NAME}`, `{repo}` and the other source trees, `{eval_dir}`, `{python:qonnx}` for stage 2) and an **import step** (`tools/import_qonnx.py {out_dir}/<stem>.qonnx.onnx` with no import argument); the derivations (`role: derivation`: the `--no-fold` import of the parent's canonical file, `tools/ftl_deepfake_submodels.py`); NNCF first extracts the tensors-only state (`role: prepare`) |
| `outputs` (per step) | what the step writes under `{work}` and its name in the model directory: `<stem>.qonnx.onnx` (the canonical file), `<stem>.qonnx.json` (the writer's record), `<stem>.bitino`, `<stem>.import.json` (the importer's report); stale copies are removed before the step, a missing output fails it |
| `expected` | data and graph SHA-256 (and size) of **every** file the route builds: the 296 published files and the 190 derived ones (was: `model.bitino` and the int8 files, 233) |
| `qonnx` | the front end, and per file its canonical file, writer and mapping, or its derivation |
| `crosscheck` | the legacy converter steps (`tools/convert_*.py`, the former `steps`), run only by `fetch --legacy` |
| `int8[]` | the quantizer arguments (output `{work}/int8/<file>`) and `qonnx_steps`: the int8 writer on the quantizer output and its re-import |
| `verified_qonnx` | the end-to-end run of these steps by `fetch` (from `registry/recipes_verified.json`) |

Environments. The writers run in the source framework's environment: `qonnx2` (torch 2.6.0 + onnx, the PyTorch
writer), `larq`, `torch` (step-3 transformers, the PyTorch `*_unfused` writer), `bhvit`, `ftl`, `openvino`, `brevitas`,
`nncf`, `qonnx` (daBNN, FINN, BNN-PYNQ, the importer and every writer's stage 2) and `theano27` (BNN-PYNQ's Theano
binary weights). The writers of steps 3 and 4 and of the own-canonical `*_unfused` files read the checkpoint with the
legacy adapters (as in the dev build); in fetch they compare that parse with the recipe's recorded hashes
(`--expect-recipe`, no published file needed) instead of `artifacts/`.

What the user gets in `$BITINO_HOME/models/<id>/`: every `.bitino` of the route, its canonical
`<stem>.qonnx.onnx` (the QONNX file is a deliverable, not a temporary), and `fetch_record.json` with the downloads,
the commands, the SHA-256 of every canonical file and, per `.bitino`, the data / graph comparison with `expected`.
The canonical files are not byte-compared with the dev sweep's (their metadata records the writer's commit and the
eval set used); the bar is the `.bitino`, data + graph.

**int8_full.** The published int8 files are the quantizer's output on calibration images that are not
redistributed, so fetch cannot write an int8 canonical file from the checkpoint alone and does not pretend to.
`fetch --int8 --calib-dir D` runs: the quantizer on the QONNX-built `model.bitino` with the user's images
(`{work}/int8/<file>`) → `tools/export_int8_qonnx.py --source` writes the int8 canonical file of *that* output
(`<stem>.qonnx.onnx`, metadata from its own manifest) → `tools/import_qonnx.py` re-imports it as `<file>` in the
model directory. The re-import **must** equal the quantizer output (data + graph SHA-256), else the fetch fails and
the re-import is moved aside; the comparison with the published int8 file is **reported**, and holds only with the
recorded calibration samples and quantizer version (checked: `recu-resnet20-cifar10` with the CIFAR-10 calibration
set rebuilt by `tools/eval_sets/cifar10_normalized.py --split train`, tensor SHA-256 `faaf8baf…` as recorded: the
re-import equals the quantizer output and the published `model_int8_full.bitino`, data and graph). The dev build's
int8 canonical files (`tools/build_from_qonnx.py`) are still written from the published int8 graphs (step 5).

**`--legacy`** runs the recipe's legacy converter into `<model dir>/legacy/` after the QONNX build, compares each file
it writes with the expected hashes and with the QONNX-built file (`fetch_record.json` → `crosscheck`), and leaves
the converter's reference outputs (the source framework's logits on the eval set) there: `bitino validate` uses
them when the model directory has none (the QONNX build itself checks identity only).

## Adding a network

1. **Registry.** Add the model to `registry/models.json` (paper, code, checkpoint URL, SHA-256, licences) and its recipe
   (`tools/build_recipes.py`: downloads, sources, eval set).
2. **Writer.** Use the front end's writer. A PyTorch model whose binarizer, padding and scale conventions an existing
   hook of `tools/export_torch_qonnx.py` covers needs only a family entry; otherwise add a hook (an exportable module:
   `BipolarQuant` / `Sign` binarizers, explicit binary `Pad`, ±1 weights with `Mul(α)`, BatchNorm folded in float64).
   A Brevitas model exports with `brevitas.export.export_qonnx` and is rewritten to the profile (step 4's FINN path).
   A framework without an ONNX exporter gets a direct writer on `tools/qonnx_ir_writer.py`. A new operator needs a
   pattern in [qonnx-profile.md](qonnx-profile.md) with its exactness class and a rule in the checker and the importer;
   anything that cannot be written exactly in standard operators is a design decision, never a custom operator.
3. **Check.** `bitino check-qonnx model.onnx` must pass (no error; warnings justified).
4. **Import.** `tools/import_qonnx.py model.onnx --model-id ID --out-dir DIR` with no layout argument: the file's
   `bitino.import` decides. For a new network there is no published file to match: its `.bitino` *is* the import of
   the canonical file; verify it as § 8 says (export check against the framework, Level 3 / bit-exactness, C++ runtime).
5. **Route.** Add the network to the writer's tables, regenerate `registry/qonnx_routes.json`
   (`tools/build_from_qonnx.py --write-routes`) and build it: `tools/build_from_qonnx.py --model-id ID`.
6. **int8_full.** Quantize the float file with `tools/quantize_int8_full.py` and write its canonical file with
   `tools/export_int8_qonnx.py` (the integer graph in the § 5 patterns, re-emission-checked by the importer).

[converting-a-new-bnn.md](converting-a-new-bnn.md) is the step-by-step guide with commands.
