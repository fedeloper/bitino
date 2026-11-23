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
- **QONNX as the standard format of every network.** All 132 networks, float and int8_full files, have a canonical
  QONNX file written in standard ONNX + QONNX operators only, and `tools/build_from_qonnx.py` builds every file from
  it: all 486 files (296 published, 190 derived) come out with the published data and graph SHA-256
  ([docs/qonnx-standard.md](docs/qonnx-standard.md), profile and checker: [docs/qonnx-profile.md](docs/qonnx-profile.md),
  `tools/check_qonnx.py`).
- **Full ImageNet int8_full accuracy for all 50 ImageNet int8_full files** (`accuracy.full_val.int8_full`): 48 lose a
  median 0.84 pt; BNAS-D and the bitorch MeliusNet-22 int8_full files collapse to chance level and should not be
  used (their float files are unaffected).
- **32-bit Android release binary** (`bitino_bench-android-armv7`, `bitino-sdk-android-armv7.tar.gz`), so
  `scripts/bench_my_phone.sh` runs on armeabi-v7a phones without a local build.
- **Community speed table**: phones measured by users with `scripts/bench_my_phone.sh` and validated by
  `tools/ingest_community.py` appear in the README and on the benchmarks page.
- **Packages**: `pip install bitino` (PyPI) and `io.github.fedeloper:bitino-android` (Maven Central).
- README: "Which model should I use?", thread-scaling chart; CONTRIBUTING.md and issue templates.

### Changed
- **`bitino fetch` builds through QONNX.** Each recipe runs the network's QONNX writer on the authors' checkpoint
  (canonical file `<stem>.qonnx.onnx`, kept in the model directory), imports it with no import argument and derives
  the other files; every built file (all 486 files of the 132 recipes now have recorded data and graph SHA-256) is
  compared with the published one. The per-framework converters are a cross-check (`fetch --legacy`, which also
  computes the source-framework outputs `bitino validate` needs). `fetch --int8` writes the int8 QONNX file of the
  quantizer output and requires its re-import to equal it. The writers need the `qonnx` environment (and `qonnx2` for
  PyTorch networks) next to the source framework's ([docs/qonnx-standard.md](docs/qonnx-standard.md)).
- `bitino_bench --affinity` warms the pinned cores up for 1 s before timing (`--prewarm-ms`), which removes the
  first-run outliers on very small models.

## v0.1.0 (2026-11-18)

First public release: the C++ runtime (NEON for ARMv8 / ARMv7, AVX2 / AVX-VNNI), the Python package and CLI, the
Android library and two example apps, 132 converted BNNs with fetch recipes, 11 redistributable converted files, and
the benchmark results of four phones.
