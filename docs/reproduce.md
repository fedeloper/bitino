# Reproducing the numbers

Every number in the README, the website and the paper comes from a script in this repository and a data file under
`results/` or `reports/`. This page lists them, claim by claim. Phone measurements need the phone (USB debugging, no
root); everything else runs on a Linux x86-64 host.

| Claim | Data | How to re-run |
|---|---|---|
| A converted model matches its source framework | `results/models/<id>.json` → `accuracy.headline` | `bitino fetch <id>` then `bitino validate <id>` (source model in float64 vs the converted graph vs the C++ runtime) |
| A fetch recipe rebuilds the published file | `registry/recipes.json` → `expected` | `bitino fetch <id>` prints `IDENTICAL` when the data section matches bit for bit |
| ImageNet top-1 on the full validation set, int8_full accuracy change | `results/models/<id>.json` → `accuracy.full_val`; `reports/accuracy_full50k/original_resolution/summary.md` | `tools/accuracy_full_val.py manifest --copy original --val-dir <ILSVRC2012 val>` then `run` and `summarize` (about a day on 8 cores; resumable) |
| Latency of every model on a phone (1 and 4 threads) | `results/speed_<device>.json` | `tools/benchmark.py` with the plan of [benchmarking.md](benchmarking.md); then `tools/report_benchmarks.py` and `tools/ingest_speed.py` |
| Thread scaling (1, 2, 4, 8 threads) | `results/speed_<device>.json`, plans `threads_*` | `ANDROID_SERIAL=<serial> CORES=<cores, fastest first> scripts/device_threads_bench.sh` |
| Faster than Larq Compute Engine and TensorFlow Lite | `reports/h2h/` | `scripts/h2h_lce_tflite.sh` ([comparison-lce-tflite.md](comparison-lce-tflite.md)) |
| Faster than daBNN | `reports/h2h_dabnn/` | `scripts/h2h_dabnn.sh` ([comparison-dabnn.md](comparison-dabnn.md)) |
| Measured energy per inference (OnePlus Open) | `reports/energy_oneplus/combined.md` | `tools/energy_oneplus.py run --out <dir>` (needs the OPLUS battery service; other phones expose no battery current without root) |
| QONNX import reproduces the published files | `reports/qonnx_phase1/summary.json` | `tools/qonnx_phase1_acceptance.py` ([qonnx-phase1.md](qonnx-phase1.md)) |
| Your own phone | your JSON | `scripts/bench_my_phone.sh` ([benchmark-your-phone.md](benchmark-your-phone.md)) |

Rules every number follows:
- **Timings come from real hardware**, at batch 1, with the core, thread count, clock ceilings and thermal status
  recorded per run. Emulators (qemu) are used for correctness and instruction counts only, never for timings.
- **Accuracy uses the original images and each model's own preprocessing.** A copy of the ImageNet validation set
  whose images were already resized to 256 px lowers BNN top-1 by 1.6 pt on average, so the numbers here use the
  original JPEGs.
- **Datasets are not redistributed.** The evaluation manifests (`registry/eval_sets/`) list file names, SHA-256 and
  labels, so a run on your copy checks that it is the same data.

If a number does not reproduce, that is a bug: please open an issue with the **Bug or wrong number** template.
