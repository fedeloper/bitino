<p align="center"><img src="site/assets/logo.svg" width="96" alt="Bitino logo"></p>

<h1 align="center">Bitino</h1>
<p align="center"><i>1-bit neural networks that keep your phone cool.</i><br>
by Fred · <a href="https://federicofontana.dev">federicofontana.dev</a></p>

Bitino runs **binary neural networks** (BNNs) on the CPU of an Android phone or a laptop: a dependency-free C++17
runtime with NEON (ARMv8.2 dotprod, i8mm) and AVX2 / AVX-VNNI kernels, converters that take networks from their
**original** published checkpoints, a full-integer int8 path, and a curated zoo of every BNN we could find with
public weights.

<!-- generated:stats -->
**326** binary networks investigated, **206** with verified public weights, **114** converted and validated against their original framework, **11** redistributable as converted files (explicit permissive weights licence). Fetch recipes: 4 verified end to end, 110 reconstructed from the conversion records.
<!-- /generated:stats -->

*Un bitino* is Italian for “a little bit”, which is all a binary network needs per weight.

## Why binary networks, why Bitino

* **Small and cool.** A binary convolution is `XNOR` + `popcount` on packed bits: 32 weights per 32-bit word,
  ~1/32 of the weight memory, and far less energy per operation than float multiply-adds. Phones run them on the
  CPU, without an NPU driver and without heating up.
* **Real models, not toy ones.** BNN research is spread over a dozen frameworks (Larq/Keras, PyTorch, OpenVINO,
  ONNX, Brevitas, MindSpore, Theano…). Bitino converts the authors' own checkpoints, so you get the published
  networks, verified against their source framework.
* **Checked, not trusted.** Every conversion is compared with the original framework in float32 *and* float64;
  every optimized kernel is checked against a scalar reference; every first disagreement between runtime and
  reference is traced to a rounding-scale sign flip (`docs/numerical-correctness.md`).
* **Honest speed numbers.** Timings come only from real hardware (Galaxy S22, x86-64 host), with date, thread
  setting and thermal state. Emulators are used for correctness and instruction counts, never for timings.

## Five-minute quickstart

```bash
git clone https://github.com/fedeloper/bitino && cd bitino
pip install -e .                      # the `bitino` CLI (needs only numpy)

# 1. the runtime: build it (CMake + a C++17 compiler), or download bitino_bench from the Releases page
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/host

# 2. a model: browse the zoo, then fetch one
bitino list --converted --task imagenet
bitino fetch ab-bnn-imagenet-reactnet18 --prebuilt   # converted file from GitHub Releases (permissive licence)
bitino fetch ab-bnn-imagenet-reactnet18              # or: original weights + code, SHA-256 check, local conversion (needs torch)

# 3. run it on x86 and time it
bitino run ab-bnn-imagenet-reactnet18 --random 2
bitino bench ab-bnn-imagenet-reactnet18 --threads 4
bitino validate ab-bnn-imagenet-reactnet18           # runtime vs numpy executor vs source framework
```

**Deploy on Android** (arm64, no root): take `bitino_bench-android-arm64` from the Releases page (or
`scripts/build_android.sh` with the NDK) and

```bash
adb push bitino_bench-android-arm64 /data/local/tmp/bitino_bench
adb push ~/.cache/bitino/models/ab-bnn-imagenet-reactnet18/model.bitino /data/local/tmp/
adb shell /data/local/tmp/bitino_bench --model /data/local/tmp/model.bitino --threads 4 --stage all
```

In an app, link `libbitino.a` and use the C API (`runtime/include/bitino/bitino_c.h`) or the JNI example
(`examples/android_jni/`). Details: [docs/getting-started.md](docs/getting-started.md),
[docs/android-deployment.md](docs/android-deployment.md).

`bitino fetch` downloads the weights from where the authors published them, into a fresh directory, checks their
SHA-256, downloads the authors' model code pinned to the recorded commit, and runs the converter with `python -I`
(checkpoints are read with `torch.load(weights_only=True)` only). The converted file's data section is compared with
the published conversion: `IDENTICAL` means bit-identical weights and constants. Converters run in the source
framework's environment (`envs/`); `bitino fetch` says which one and how to point it there
(`--python larq=/path/to/python`). Without a dataset it generates a small synthetic eval set, which is enough for
conversion and runtime checks (accuracy numbers need the real one).

## Speed (Galaxy S22, measured)

<!-- generated:speed -->
| model | task | accuracy (reference → Bitino) | S22 1 thread (X2) | S22 4 threads | measured | model file |
|---|---|---|---|---|---|---|
| `ab-bnn-cifar10-reactnet18` | cifar10-classification | 93.8% → 93.8% (cifar10-test-first-128, n=128) | 2.92 ms ⚠ (int8_full) | 2.02 ms ⚠ (int8_full) | 2026-11-04 | model_int8_full_log2.bitino |
| `ab-bnn-cifar10-reactneta` | cifar10-classification | 91.4% → 91.4% (cifar10-test-first-128, n=128) | 2.6 ms ⚠ (float) | 2.48 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `ab-bnn-cifar100-reactnet18` | cifar100-classification | 69.5% → 69.5% (cifar100-test-first-128, n=128) | 3.72 ms ⚠ (float) | 2.56 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `ab-bnn-cifar100-reactneta` | cifar100-classification | 71.9% → 71.9% (cifar100-test-first-128, n=128) | 2.63 ms ⚠ (float) | 2.47 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `ab-bnn-imagenet-reactnet18` | imagenet-classification | 65.6% → 64.8% (imagenet-val-subset-128, n=128) | 10.9 ms ⚠ (int8_full) | 5.44 ms ⚠ (int8_full) | 2026-11-04 | model_int8_full_log2.bitino |
| `ab-bnn-imagenet-reactnet34` | imagenet-classification | 67.2% → 68.0% (imagenet-val-subset-128, n=128) | 27.0 ms ⚠ (float) | 14.6 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `ab-bnn-imagenet-reactneta` | imagenet-classification | 66.4% → 65.6% (imagenet-val-subset-128, n=128) | 19.6 ms ⚠ (float) | 12.8 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `adabin-resnet20-cifar10` | cifar10-classification | 91.4% → 91.4% (cifar10-test-first-128, n=128) | 1.22 ms ⚠ (float) | 9.37 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | cifar10-classification | 89.8% → 89.8% (cifar10-test-first-128, n=128) | 0.57 ms ⚠ (float) | 1.39 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bbcu-deblocking` | jpeg-artifact-reduction (grayscale, Y channel) | 31.33 dB → 31.33 dB (live1-classic5-jpeg-y-q10q40-center96, n=68) | 33.1 ms ⚠ (float) | 16.3 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bbcu-denoise` | color-image-denoising (AWGN blind) | 29.74 dB → 29.74 dB (cbsd68-sigma25-center96, n=68) | 34.0 ms ⚠ (float) | 16.7 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bbcu-sr-l-x2` | single-image-super-resolution-x2 | 32.03 dB → 32.03 dB (set5-set14-b100-x2-lr48, n=64) | 36.6 ms ⚠ (float) | 19.5 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | 27.36 dB → 27.36 dB (set5-set14-b100-x4-lr32, n=64) | 53.5 ms ⚠ (float) | 27.6 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bbcu-sr-m-x2` | single-image-super-resolution-x2 | 31.89 dB → 31.89 dB (set5-set14-b100-x2-lr48, n=64) | 25.6 ms ⚠ (float) | 13.3 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bbcu-sr-m-x4` | single-image-super-resolution-x4 | 27.22 dB → 27.22 dB (set5-set14-b100-x4-lr32, n=64) | 24.4 ms ⚠ (float) | 11.9 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bhvit-small-binary` | imagenet-classification | 68.8% → 65.6% (imagenet-val-subset-128, n=128) | 85.8 ms ⚠ (float) | 59.3 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bhvit-small-somefp` | imagenet-classification | 72.7% → 76.6% (imagenet-val-subset-128, n=128) | 106.9 ms ⚠ (float) | 70.6 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bhvit-tiny-binary` | imagenet-classification | 64.1% → 62.5% (imagenet-val-subset-128, n=128) | 46.6 ms ⚠ (float) | 39.3 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bhvit-tiny-somefp` | imagenet-classification | 65.6% → 65.6% (imagenet-val-subset-128, n=128) | 58.8 ms ⚠ (float) | 37.4 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bi-vit-deit-small` | imagenet-classification | 11.7% → 10.9% (imagenet-val-subset-128, n=128) | 68.6 ms ⚠ (int8_full) | 56.5 ms ⚠ (int8_full) | 2026-11-03 | model_int8_full.bitino |
| `bi-vit-deit-tiny` | imagenet-classification | 15.6% → 15.6% (imagenet-val-subset-128, n=128) | 21.9 ms ⚠ (int8_full) | 13.0 ms ⚠ (int8_full) | 2026-11-04 | model_int8_full.bitino |
| `bibert-bert-base-mnli` | glue-mnli | 61.7% → 61.7% (mnli-dev-matched-128, n=128) | 124.7 ms ⚠ (float) | 54.9 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bibert-bert-base-mrpc` | glue-mrpc | 76.6% → 76.6% (mrpc-dev-128, n=128) | 148.9 ms ⚠ (float) | 68.3 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bibert-bert-base-qqp` | glue-qqp | 83.6% → 83.6% (qqp-dev-128, n=128) | 160.7 ms ⚠ (float) | 73.6 ms ⚠ (float) | 2026-11-02 | model.bitino |
| `bnext-tiny` | imagenet-classification | 69.5% → 68.0% (imagenet-val-subset-128, n=128) | 39.2 ms ⚠ (int8_full) | 20.4 ms ⚠ (int8_full) | 2026-11-04 | model_int8_full.bitino |
| `larq-zoo-quicknet-small` | imagenet-classification | 59.4% → 58.6% (imagenet-val-subset-128, n=128) | 8.14 ms ⚠ (int8_full) | 4.17 ms ⚠ (int8_full) | 2026-11-04 | model_int8_full.bitino |

⚠ = at least one repeat ran with a thermal CPU-frequency cap (phone charging over USB; caps are recorded per run in `results/speed_s22.json`); † = run made before the frequency-cap check existed (caps unknown). Every published S22 number so far is one of the two: the full device suite has not been re-run on the current runtime yet, so each number is for the runtime of its date. "float" = exact binary layers with float glue (stage all); int8_full = full-integer file. Emulated timings are never published.
<!-- /generated:speed -->

All runs and their thermal state: `results/speed_s22.json`; x86-64 host runs (older runtime, labelled):
`results/speed_x86.json`. Method: [docs/benchmarking.md](docs/benchmarking.md).

## Model zoo

Models measured on the phone so far (full table with all converted models: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md);
searchable version on the website):

<!-- generated:zoo_top -->
| model | task | framework | accuracy (reference → Bitino) | eval set (n) | int8_full Δ vs exact | S22 ms (t1) | weights licence | release |
|---|---|---|---|---|---|---|---|---|
| `ab-bnn-cifar10-reactnet18` | cifar10-classification | pytorch | 93.8% → 93.8% | cifar10-test-first-128 (128) | +0.06 pt (cifar10-test-first-10000, n=10000) | 2.92 ⚠ | MIT (HF card) | yes |
| `ab-bnn-cifar10-reactneta` | cifar10-classification | pytorch | 91.4% → 91.4% | cifar10-test-first-128 (128) | -0.22 pt (cifar10-test-first-10000, n=10000) | 2.6 ⚠ | MIT (HF card) | yes |
| `ab-bnn-cifar100-reactnet18` | cifar100-classification | pytorch | 69.5% → 69.5% | cifar100-test-first-128 (128) | -0.56 pt (cifar100-test-10000, n=10000) | 3.72 ⚠ | MIT (HF card) | yes |
| `ab-bnn-cifar100-reactneta` | cifar100-classification | pytorch | 71.9% → 71.9% | cifar100-test-first-128 (128) | -0.31 pt (cifar100-test-10000, n=10000) | 2.63 ⚠ | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactnet18` | imagenet-classification | pytorch | 65.6% → 64.8% | imagenet-val-subset-128 (128) | -0.34 pt (imagenet-val-subset-5000, n=5000) | 10.9 ⚠ | MIT (HF model card metadata 'license: mit'); app | yes |
| `ab-bnn-imagenet-reactnet34` | imagenet-classification | pytorch | 67.2% → 68.0% | imagenet-val-subset-128 (128) | -0.62 pt (imagenet-val-subset-5000, n=5000) | 27.0 ⚠ | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactneta` | imagenet-classification | pytorch | 66.4% → 65.6% | imagenet-val-subset-128 (128) | -0.58 pt (imagenet-val-subset-5000, n=5000) | 19.6 ⚠ | MIT (HF card) | yes |
| `adabin-resnet20-cifar10` | cifar10-classification | pytorch | 91.4% → 91.4% | cifar10-test-first-128 (128) | -0.64 pt (cifar10-test-first-10000, n=10000) | 1.22 ⚠ | MIT by co-location in the same directory (no sep | – |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | cifar10-classification | pytorch+brevitas | 89.8% → 89.8% | cifar10-test-first-128 (128) | +0.06 pt (cifar10-test-first-10000, n=10000) | 0.57 ⚠ | unspecified | – |
| `bbcu-deblocking` | jpeg-artifact-reduction (grayscale, Y channel) | pytorch | 31.33 dB → 31.33 dB | live1-classic5-jpeg-y-q10q40-center96 (68) | – | 33.1 ⚠ | no explicit weights licence (inherits repo licen | – |
| `bbcu-denoise` | color-image-denoising (AWGN blind) | pytorch | 29.74 dB → 29.74 dB | cbsd68-sigma25-center96 (68) | – | 34.0 ⚠ | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-l-x2` | single-image-super-resolution-x2 | pytorch | 32.03 dB → 32.03 dB | set5-set14-b100-x2-lr48 (64) | – | 36.6 ⚠ | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | pytorch | 27.36 dB → 27.36 dB | set5-set14-b100-x4-lr32 (64) | – | 53.5 ⚠ | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-m-x2` | single-image-super-resolution-x2 | pytorch | 31.89 dB → 31.89 dB | set5-set14-b100-x2-lr48 (64) | – | 25.6 ⚠ | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-m-x4` | single-image-super-resolution-x4 | pytorch | 27.22 dB → 27.22 dB | set5-set14-b100-x4-lr32 (64) | – | 24.4 ⚠ | no explicit weights licence (inherits repo licen | – |
| `bhvit-small-binary` | imagenet-classification | pytorch | 68.8% → 65.6% | imagenet-val-subset-128 (128) | – | 85.8 ⚠ | no explicit weights licence (inherits repo MIT?  | – |
| `bhvit-small-somefp` | imagenet-classification | pytorch | 72.7% → 76.6% | imagenet-val-subset-128 (128) | -1.90 pt (imagenet-val-subset-5000_bhvit, n=5000) | 106.9 ⚠ | no explicit weights licence (inherits repo MIT?  | – |
| `bhvit-tiny-binary` | imagenet-classification | pytorch | 64.1% → 62.5% | imagenet-val-subset-128 (128) | -0.74 pt (imagenet-val-subset-5000_bhvit, n=5000) | 46.6 ⚠ | no explicit weights licence (inherits repo MIT?  | – |
| `bhvit-tiny-somefp` | imagenet-classification | pytorch | 65.6% → 65.6% | imagenet-val-subset-128 (128) | -6.20 pt (imagenet-val-subset-5000_bhvit, n=5000) | 58.8 ⚠ | no explicit weights licence (inherits repo MIT?  | – |
| `bi-vit-deit-small` | imagenet-classification | pytorch | 11.7% → 10.9% | imagenet-val-subset-128 (128) | -0.88 pt (imagenet-val-subset-5000_bhvit, n=5000) | 68.6 ⚠ | no explicit weights licence (inherits repo Apach | – |
| `bi-vit-deit-tiny` | imagenet-classification | pytorch | 15.6% → 15.6% | imagenet-val-subset-128 (128) | -0.34 pt (imagenet-val-subset-5000_bhvit, n=5000) | 21.9 ⚠ | no explicit weights licence (inherits repo Apach | – |
| `bibert-bert-base-mnli` | glue-mnli | pytorch | 61.7% → 61.7% | mnli-dev-matched-128 (128) | +0.54 pt (mnli-dev-matched-9815, n=9815) | 124.7 ⚠ | unspecified | – |
| `bibert-bert-base-mrpc` | glue-mrpc | pytorch | 76.6% → 76.6% | mrpc-dev-128 (128) | +0.98 pt (mrpc-dev-408, n=408) | 148.9 ⚠ | no explicit weights licence (inherits repo licen | – |
| `bibert-bert-base-qqp` | glue-qqp | pytorch | 83.6% → 83.6% | qqp-dev-128 (128) | -0.59 pt (qqp-dev-40430, n=40430) | 160.7 ⚠ | unspecified | – |
| `bnext-tiny` | imagenet-classification | pytorch | 69.5% → 68.0% | imagenet-val-subset-128 (128) | -3.84 pt (imagenet-val-subset-5000, n=5000) | 39.2 ⚠ | no explicit weights licence (inherits repo Apach | – |
| `larq-zoo-quicknet-small` | imagenet-classification | larq_zoo | 59.4% → 58.6% | imagenet-val-subset-128 (128) | -1.96 pt (imagenet-val-subset-5000, n=5000) | 8.14 ⚠ | No explicit weights licence (inherits repo licen | – |
<!-- /generated:zoo_top -->

Accuracy columns are conversion checks on small subsets (n = 128 for most classifiers): the original model run by its
own framework (in float64 where the conversion recorded it, else float32) against Bitino's exact stage. They are not
estimates of full-dataset accuracy.

**Weights and licences.** Converted files are attached to GitHub Releases only when the weights licence is
explicitly permissive (Apache-2.0 / MIT / BSD stated for the weights), with the licence text and attribution next to
each file (`bitino list --release`). Everything else is one `bitino fetch` away from its original source; the
registry records code and weights licences separately for every model (`registry/models.json`).

## What is supported

* **Frameworks / checkpoint formats:** Larq Zoo and Keras/Larq `.h5` (TensorFlow), PyTorch state dicts and
  Lightning checkpoints (AB-BNN, ReActNet / Bi-Real / IR-Net families, BNext, bitorch, BNAS, BHViT, Bi-ViT, BiBERT,
  restoration and super-resolution BNNs, …), OpenVINO IR, ONNX (daBNN), FINN-ONNX / QONNX, Brevitas, NNCF,
  MindSpore checkpoints, tensorpack and Theano/Lasagne `.npz`.
* **Operators:** binary Conv2D / Dense / MatMul with exact padding correction and fused thresholds, float
  Conv2D / depthwise / Dense / pooling / elementwise, transformer blocks (attention, LayerNorm, GELU, softmax),
  full-integer `I8*` operators. Contracts: [docs/operators.md](docs/operators.md).
* **Kernel tiers:** scalar reference (every op), NEON (`binary_neon`, `float_opt`, `int8_neon` with SDOT / SMMLA),
  x86-64 AVX2 / AVX-VNNI, selected at run time from the CPU features.
* **Stages:** `reference`, `neon` (exact), `optimized` (float fusion), `optimized_int8`, `int8_full`.
* **Targets:** Android arm64 (API 30+), Linux aarch64, Linux x86-64.

## Add a model

1. Add the network to `registry/models.json` (checkpoint URL, SHA-256, licences, provenance).
2. Convert it with the closest converter in `tools/convert_*.py` (or write an adapter in `python/bitino/adapters/`).
3. Validate: `tools/validate_model.py` on the host, then on a device.
4. `tools/build_recipes.py --artifacts <dir>` writes its fetch recipe; `tools/build_site.py` refreshes the tables.

Full walk-through: [docs/adding-a-model.md](docs/adding-a-model.md).

## Honest limitations

* **Speed data is thin.** The full device suite has not been re-run on the current runtime; every published S22
  number was measured on a thermally capped phone (charging over USB) and is labelled so. x86 host numbers are
  from an older runtime. Slots for both suites are in `results/`.
* **Fetch recipes:** a few are verified end to end; the others are reconstructed from the conversion records
  (state shown by `bitino list`). Some sources need a manual download (Google Drive, Baidu) or a second reference
  environment (MindSpore, tensorpack, Python 2.7 Theano).
* **int8_full** loses accuracy on some networks (see the zoo tables); the exact `neon` / `optimized` stages
  match the source framework. Reproducing an int8 file bit for bit needs the calibration images it was built from
  (training split, not redistributed).
* **Accuracy subsets are small** (mostly 128 images): they validate conversions, they do not re-measure the papers.
* **One phone.** Tuned and measured on one Snapdragon 8 Gen 1. Other arm64 CPUs run the same code; their speed
  is not measured here. No GPU / NPU backends.
* **File format v1** keeps the on-disk magic `BFMODEL1` from the format's original name, so files converted
  before the rename stay readable (`docs/model-format.md`).

## Citation

```bibtex
@software{fontana_bitino_2026,
  author = {Fontana, Federico},
  title  = {Bitino: 1-bit neural networks that keep your phone cool},
  year   = {2026},
  url    = {https://github.com/fedeloper/bitino}
}
```

Please also cite the papers of the networks you use (`bitino info <model-id>` prints each paper).

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE), third-party notes in [THIRD_PARTY.md](THIRD_PARTY.md)).
Model weights are not covered by this licence: each keeps its authors' licence. Datasets are not redistributed.
