# Adding a model

The zoo is data: a registry entry, a conversion, a fetch recipe and a results file. Adding a model touches each once.

## 1. Registry entry

Add an object to `registry/models.json` → `models` (copy a similar entry). Required: `model_id` (lowercase,
`<family>-<arch>-<dataset>`), `task`, `paper`, `repo_url`, `checkpoint.url`, `checkpoint.sha256`,
`checkpoint.size_bytes`, `precision.categories` (`W1A1`, `W1A1+FP`, …), `strict_bnn`, and `license` with **code and
weights recorded separately** plus `evidence` (where the licence is stated). Write "no explicit weights licence"
when the weights have none: only an explicit MIT / Apache-2.0 / BSD statement for the weights lets the converted
file be redistributed.

## 2. Convert

Use the converter of the closest family (`tools/convert_*.py`; `docs/operators_*.md` lists the semantics each one
handles). They all follow the same pattern:

```bash
python -I tools/convert_<family>.py --model-id <id> --repo <authors' code at a commit> --weights <checkpoint> \
    --eval-dir <eval set> --out-dir artifacts/<id>
```

and write `model.bitino` (fused), `model_unfused.bitino`, the source framework's float32 and float64 reference
outputs, the numpy executor's outputs and `conversion_report.json` (Level-3 semantic check). A new architecture
needs an adapter in `python/bitino/adapters/` that maps the instantiated source model to the IR
(`python/bitino/ir.py`); new operators need an `OpDef`, a scalar reference kernel and tests
(`docs/operators.md` → "Adding an operator").

Never load untrusted checkpoints with pickle: use `torch.load(weights_only=True)` (or the framework's weight-only
readers), in a fresh directory, with `python -I`.

## 3. Validate

```bash
cmake --build build/host
python3 tools/validate_model.py --artifacts artifacts/<id> --eval-dir <eval> --preprocess <key> \
    --bench build/host/bitino_bench --stage neon --accept-exact-numpy
python3 tools/validate_model.py ... --stage all --explain
```

`neon` must be bit-exact with the numpy executor; optimized stages must agree, and every top-1 disagreement must
trace to a rounding-scale first binarization flip (`docs/numerical-correctness.md`). For arm64 without a phone:
`BITINO_STATIC=ON scripts/build_android.sh build/android-arm64-static` and `scripts/bench_emulated.sh` as `--bench`
(correctness only).

## 4. int8 (optional)

```bash
python3 tools/quantize_int8_full.py --model artifacts/<id>/model.bitino --calib-dir <calibration set> \
    --preprocess <key> --n 128 --out artifacts/<id>/model_int8_full.bitino
python3 tools/accuracy_eval.py ...   # paired exact vs int8_full on a larger set (McNemar)
```

The calibration set must come from the training split (disjoint from every eval set); `tools/build_eval_set.py
--split train` records the disjointness check.

## 5. Recipe, results, tables

```bash
python3 tools/build_recipes.py --artifacts <dir containing artifacts/>   # registry/recipes.json
bitino fetch <id> --python <env>=<python>       # run the recipe end to end; then record it in
                                                 # registry/recipes_verified.json (state "verified")
python3 tools/ingest_speed.py --s22 <benchmark output dir>  # measured phone runs (tools/benchmark.py)
python3 tools/build_site.py --readme-only        # README / docs/MODEL_ZOO.md tables
python3 tools/build_site.py --out ../site        # website
```

Speed numbers come only from real hardware, with their thermal state recorded (`docs/benchmarking.md`); emulated
timings are never added.
