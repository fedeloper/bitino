# Changelog

All notable changes to Bitino. Dates are UTC.

## Unreleased

### Added
- **Thread pool for phones with big and little cores.** Each thread's share of a layer is sized to its measured
  speed, slow cores stop claiming work when the fast ones would finish first, and each layer's thread count is tuned
  during the first runs of a session (`auto_threads`, on by default; `BITINO_AUTO_THREADS=0` turns it off). Median
  speed-up at 8 threads over 1: 2.19× → 2.56× (OnePlus Open), 3.20× → 3.47× (moto e13), 4.80× → 4.82× (Redmi 9A);
  8 threads is no longer slower than 4 for any model. Outputs stay byte-identical at every thread count.
  ([docs/threads.md](docs/threads.md))
- **1, 2, 4 and 8-thread results** for every model on three phones (`results/speed_<device>.json`, plans `threads_*`).
- **Full ImageNet validation accuracy.** Every ImageNet classifier on all 50,000 validation images
  (`results/models/<id>.json` → `accuracy.full_val`): 53 of 55 within 0.5 pt of their published top-1. The int8_full
  accuracy change of the headline files is now measured on the same 50,000 images.
- **QONNX import.** Brevitas, QONNX and FINN-ONNX models import through one importer
  ([docs/converting-a-new-bnn.md](docs/converting-a-new-bnn.md)).
- README: "Which model should I use?", thread-scaling chart; CONTRIBUTING.md and issue templates.

### Changed
- `bitino_bench --affinity` warms the pinned cores up for 1 s before timing (`--prewarm-ms`), which removes the
  first-run outliers on very small models.

## v0.1.0 (2026-11-18)

First public release: the C++ runtime (NEON for ARMv8 / ARMv7, AVX2 / AVX-VNNI), the Python package and CLI, the
Android library and two example apps, 132 converted BNNs with fetch recipes, 11 redistributable converted files, and
the benchmark results of four phones.
