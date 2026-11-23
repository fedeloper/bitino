<p align="center"><img src="site/assets/logo.svg" width="110" alt="Bitino logo"></p>

<h1 align="center">Bitino</h1>

<p align="center"><b>1-bit neural networks that keep your phone cool.</b><br>
A fast, dependency-free runtime for binary neural networks on phone and laptop CPUs, with a zoo of 132 published
BNNs read from their original checkpoints and stored in one standard format, QONNX.</p>

<p align="center">
  <a href="https://github.com/fedeloper/bitino/actions/workflows/ci.yml"><img src="https://github.com/fedeloper/bitino/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/fedeloper/bitino/releases"><img src="https://img.shields.io/github/v/release/fedeloper/bitino" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <a href="https://pypi.org/project/bitino/"><img src="https://img.shields.io/pypi/v/bitino" alt="PyPI"></a>
  <a href="https://central.sonatype.com/artifact/io.github.fedeloper/bitino-android"><img src="https://img.shields.io/maven-central/v/io.github.fedeloper/bitino-android" alt="Maven Central"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platforms-Android%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platforms">
  <a href="https://colab.research.google.com/github/fedeloper/bitino/blob/main/examples/colab/bitino_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

<p align="center">
  <a href="https://federicofontana.dev/bitino/">Website</a> ·
  <a href="docs/MODEL_ZOO.md">Model zoo</a> ·
  <a href="docs/SPEED.md">Benchmarks</a> ·
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/python.md">Python API</a> ·
  <a href="docs/android-deployment.md">Android</a>
</p>

---

Bitino runs **binary neural networks** (BNNs: 1-bit weights and activations) on the CPU of an Android phone or a
laptop. It is a C++17 runtime with no dependencies, with hand-written NEON kernels (ARMv8 dotprod / i8mm, ARMv7) and
AVX2 / AVX-VNNI kernels. It reads networks from their authors' **original** checkpoints (Larq, PyTorch, OpenVINO,
ONNX, Brevitas, MindSpore, …) and writes each one as a **standard [QONNX](https://github.com/fastmachinelearning/qonnx)
file**, standard ONNX + QONNX operators only, with no custom operator; one importer compiles every QONNX file into
Bitino's phone format. It has a full-integer int8 path and a curated zoo of every BNN with public weights we could find.

```
original checkpoint ──(front-end writer)──► QONNX file ──(Bitino importer)──► .bitino ──► runtime (NEON / AVX2)
 Larq, PyTorch, OpenVINO,    standard ONNX + BipolarQuant /     one importer, no         phones and
 Brevitas, MindSpore, …      IntQuant, float and int8_full      per-model options        laptops
```

<!-- generated:stats -->
**343** binary networks investigated, **223** with verified public weights, **132** converted and validated against their original framework, **11** redistributable as converted files (explicit permissive weights licence). Fetch recipes: 131 verified end to end, 1 broken upstream.
<!-- /generated:stats -->

## Quickstart

```bash
pip install "bitino[image]"     # prebuilt wheels: Linux x86-64 / aarch64, macOS arm64, Windows x64
```

```python
import bitino

m = bitino.load("ab-bnn-imagenet-reactnet18")   # MIT-licensed weights, downloaded from the release (SHA-256 checked)
logits = m.predict_image("cat.jpg")             # the model's own preprocessing, then the runtime
print(logits.argmax(), m.benchmark(threads=1), "ms on this CPU")
```

Or try it in the browser: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fedeloper/bitino/blob/main/examples/colab/bitino_quickstart.ipynb)
(Colab runs on a cloud x86 server, not a phone; the phone numbers are below). On other platforms pip builds the
runtime from the source distribution (CMake and a C++17 compiler).

The same from a terminal:

```bash
bitino list --converted --task imagenet                     # browse the zoo
bitino run ab-bnn-imagenet-reactnet18 --image cat.jpg --bench
bitino fetch larq-zoo-quicknet                              # any other model: the authors' weights, converted locally
```

Models whose weights licence is not explicitly permissive are not redistributed. `bitino fetch` downloads them from
where the authors published them and converts them on your machine; they then load the same way.

<details>
<summary><b>Build from source</b></summary>

```bash
git clone https://github.com/fedeloper/bitino && cd bitino
pip install -e .                                           # the bitino CLI and Python API
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/host

bitino fetch ab-bnn-imagenet-reactnet18 --prebuilt         # converted file from the release (permissive licence)
bitino fetch ab-bnn-imagenet-reactnet18 --legacy           # or: original weights + code, SHA-256 check, QONNX build (+ legacy cross-check)
bitino bench ab-bnn-imagenet-reactnet18 --threads 4
bitino validate ab-bnn-imagenet-reactnet18                 # runtime vs numpy executor vs source framework
```

How `fetch` works:
- It downloads the weights from where the authors published them, into a fresh directory, and checks their SHA-256.
- It downloads the authors' model code, pinned to the recorded commit.
- It builds through QONNX: the network's writer reads the checkpoint (`python -I`; checkpoints are read with
  `torch.load(weights_only=True)` only) and writes the canonical QONNX file `model.qonnx.onnx`, which
  `bitino import-qonnx` turns into `model.bitino` with no import argument. The QONNX files stay next to the `.bitino`
  files.
- Every built file's data section and graph are compared with the published conversion: `IDENTICAL` means
  bit-identical weights, constants and graph.

Writers run in the source framework's environment, the importer in `qonnx` (`envs/`). Without a dataset, `fetch`
generates a small synthetic eval set: enough for the build, not for accuracy numbers. `--legacy` also runs the original
converter as a cross-check and computes the source framework's outputs that `bitino validate` compares against.
</details>

<details>
<summary><b>Run on an Android phone from the command line</b> (arm64, no root)</summary>

Take `bitino_bench-android-arm64` from the [Releases](https://github.com/fedeloper/bitino/releases) page (or build
it with `scripts/build_android.sh` and the NDK):

```bash
adb push bitino_bench-android-arm64 /data/local/tmp/bitino_bench
adb push ~/.cache/bitino/models/ab-bnn-imagenet-reactnet18/model.bitino /data/local/tmp/
adb shell /data/local/tmp/bitino_bench --model /data/local/tmp/model.bitino --threads 1 --stage all
```
</details>


## Highlights

- **Fast on one core.**
  - QuickNet-Small at 4.7 ms, QuickNet at 7.4 ms and ReActNet-18 at 10.8 ms per ImageNet image on a single phone
    core.
  - Faster than Larq Compute Engine on every model on every core measured: **1.35×** on a Cortex-X3, 1.14× on a
    Cortex-A715, 1.24× on a Cortex-A510, 1.12× on a Cortex-A75, 1.11× on a 32-bit Cortex-A53; and **1.6–2.05×**
    faster than daBNN on its own models.
- **One standard format.** All 132 networks, float and full-integer, are QONNX files in standard operators only.
  One importer builds every file of the zoo from them: all 486 files come out identical in data and graph to the
  published ones ([docs/qonnx-standard.md](docs/qonnx-standard.md)). A new network needs a QONNX file, not a new
  converter.
- **Real networks, verified.** 132 published BNNs from ten frameworks. Each conversion is checked against a float64
  evaluation of the source model (the source framework itself wherever it has a float64 mode). The exact `neon` stage is bit-identical with the reference executor.
- **Matches the papers.** On all 50,000 ImageNet validation images, 53 of the 55 converted classifiers are within
  0.5 pt of their published top-1 (mean −0.06 pt). The other two are Bi-ViT, whose released code and weights do not
  reach their published numbers.
- **Uses every core of a phone.** The thread pool sizes each core's share of the work to its measured speed, so the slow
  little cores help instead of holding the big ones back: the median network runs 2.6× (OnePlus Open), 3.5× (moto
  e13) and 4.8× (Redmi 9A) faster at 8 threads than at 1, and outputs are byte-identical at every thread count.
- **Full-integer path.** `int8_full` runs the whole network in integers (binary convs, int8 everything else). That is
  about 0.85× the float latency on the measured phones, for a median 0.84 pt of top-1 on 48 of the 50 ImageNet
  files (measured per file on the 50,000 validation images; two files collapse, see [Limitations](#limitations)).
- **Honest numbers.** Every timing comes from real hardware: four Android phones and an x86-64 host, with date, thread
  setting and clock state recorded. Emulators are used for correctness only, never for speed.
- **Runs everywhere a CPU does.**
  - Android arm64 and 32-bit armeabi-v7a, Linux aarch64 / x86-64, Python on Linux, macOS and Windows.
  - An Android library (AAR) with a Kotlin API and two example apps.

*Un bitino* is Italian for "a little bit", which is all a binary network needs per weight.

<p align="center"><img src="docs/img/pareto_imagenet.svg" width="760" alt="ImageNet: accuracy vs latency of the converted BNNs on one phone core, with the Pareto frontier"></p>
<p align="center"><sub>ImageNet BNNs on one core of a OnePlus Open (Snapdragon 8 Gen 2), batch 1: accuracy against measured
latency. Every point is a published network converted from its original checkpoint.</sub></p>

## Which model should I use?

ImageNet top-1 measured by Bitino on all 50,000 validation images; latency on one Cortex-X3 core of a OnePlus Open
(1 thread, batch 1, float file). Every model loads the same way: `bitino.load("<id>")`.

| You want | Model id | Top-1 | ms |
|---|---|---:|---:|
| A model with no conversion step (converted file in the release, MIT weights) | `ab-bnn-imagenet-reactnet18` | 61.3 % | 10.8 |
| The fastest ImageNet model | `larq-zoo-quicknet-small` | 59.5 % | 4.7 |
| The best under 10 ms | `larq-zoo-quicknet` | 63.4 % | 7.3 |
| The best under 15 ms | `larq-zoo-quicknet-xl-legacy` | 67.1 % | 13.0 |
| 30 fps on one core at over 70 % | `bnext-tiny` | 72.3 % | 24.0 |
| The most accurate | `bnext-large` | 80.5 % | 182 |
| CIFAR-10, smallest / most accurate | `brevitas-bnn-pynq-cnv-1w1a` / `biper-resnet18-cifar10` | 84.2 / 93.8 % (published) | 0.34 / 1.99 |

Need more speed? Use more cores (below) or the full-integer file (`int8_full`, about 0.85× the latency, at the
accuracy cost listed per model in the zoo tables).

## Android apps

The Android library (`android/bitino`) is an AAR for arm64-v8a and armeabi-v7a, API 24+. It wraps the runtime in a
small Kotlin API, and its preprocessing reproduces each model's Python pipeline bit for bit, Pillow resampling
included.

```kotlin
// build.gradle.kts: implementation("io.github.fedeloper:bitino-android:0.1.0")
// Maven Central (mavenCentral() in the repositories block); AAR for arm64-v8a and armeabi-v7a
import io.github.fedeloper.bitino.Bitino

val model = Bitino.load(context, "ab-bnn-imagenet-reactnet18.bitino", threads = 1)   // asset or file path
val scores = model.run(bitmap)              // resize / crop / normalize exactly as the model's manifest says
val top5 = Bitino.topK(Bitino.softmax(scores), 5)
model.close()
```

| App | What it does |
|---|---|
| [`android/demo`](android/demo) | Classifies the camera feed live with ReActNet-18 (bundled, MIT weights). Shows latency and fps. |
| [`android/deepfake`](android/deepfake) | Finds the images on screen with classical layout analysis and runs a binary deepfake detector ([Faster Than Lies](https://arxiv.org/abs/2406.04932)) on each, flagging the fake ones in place. With an ImageNet model instead, it labels each image with its class. Detectors are not bundled (no redistribution licence yet). |

Native code can link `libbitino.a` and use the C API (`runtime/include/bitino/bitino_c.h`). Details:
[docs/android-deployment.md](docs/android-deployment.md).

## Benchmarks

All numbers: median latency of one inference, batch 1, measured on the phone, pinned to its fastest core.

**Against Larq Compute Engine and TensorFlow Lite.** Same Larq Zoo weights, 1 thread, batch 1, every runtime pinned to
the same core by one launcher.

| Phone (core) | Bitino vs LCE 0.16 |
|---|---|
| OnePlus Open (Cortex-X3) | **1.35× faster** (all 11 models, 1.14–1.74×) |
| OnePlus Open (Cortex-A715) | **1.14× faster** (all 11, 1.05–1.29×) |
| OnePlus Open (Cortex-A510) | **1.24× faster** (all 11, 1.12–1.35×) |
| moto e13 (Cortex-A75) | **1.12× faster** (all 11, 1.03–1.25×) |
| Redmi 9A (Cortex-A53, 32-bit) | **1.11× faster** (5 models, earlier runtime) |

TFLite fp32 is 5–13× slower than Bitino. TFLite int8 is 1.4–4.8× slower, and its top-1 class agrees with the float
model on only 45–66 % of the images.

Tables, method, setup differences and raw data: [docs/comparison-lce-tflite.md](docs/comparison-lce-tflite.md).

**Multiple threads.** Every model at 1, 2, 4 and 8 threads, pinned to the N fastest cores:

<p align="center"><img src="docs/img/threads_scaling.svg" width="640" alt="Median speed-up over 1 thread at 2, 4 and 8 threads on the OnePlus Open, moto e13 and Redmi 9A"></p>

Big and little cores differ by up to 5× in speed; the pool learns each core's speed per layer and gives it a matching
share, and tunes how many threads each layer uses. How it works: [docs/threads.md](docs/threads.md).

**Against daBNN.** On daBNN's own Bi-Real Net-18 models (same weights and input, 1 thread, pinned to the same core),
Bitino is 1.60–2.05× faster on five arm64 cores (Cortex-X3, A715, A510, A75; geometric mean 1.81×) and 6.5–7.5×
faster on a 32-bit Cortex-A53, with less peak memory. Method and data: [docs/comparison-dabnn.md](docs/comparison-dabnn.md).

<!-- generated:community -->
**Your phone.** No community submissions yet. One command measures any Android phone (no root): [docs/benchmark-your-phone.md](docs/benchmark-your-phone.md).
<!-- /generated:community -->

<details>
<summary><b>Latency of every converted model on every phone</b></summary>

<!-- generated:speed -->
Median latency of one inference (batch 1) in ms. Main columns: **OnePlus Open (CPH2551)** (Qualcomm Snapdragon 8 Gen 2 (SM8550)); 1 thread = pinned to the fastest core; int8 = the full-integer file (int8_full) where measured, else the int8 kernels on the exact file. Accuracy: for ImageNet classifiers, published top-1 → Bitino's top-1 on all 50,000 validation images ("val 50k"; original JPEGs, each model's own preprocessing); otherwise the original model in its own framework → Bitino's exact stage on the small check subset. int8_full Δ: the full-integer file's paired difference on the same images (the 50,000 where measured). Fastest first.

### CIFAR-10 (23 models)

![CIFAR-10: accuracy vs latency, Pareto frontier](docs/img/pareto_cifar10.svg)

![CIFAR-10 int8_full: accuracy vs latency, Pareto frontier](docs/img/pareto_cifar10_int8_full.svg)

| model | accuracy | int8_full Δ | OnePlus Open 1 thread | OnePlus Open 4 threads | OnePlus Open int8 1 thread | Galaxy S22 1 thread | moto e13 1 thread | Redmi 9A 1 thread (32-bit) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bnn-pynq-cnv-w1a1-cifar10` | 86.7% → 86.7% | +0.42 pt | 0.318 ◇ | 0.222 ◇ | 0.335 ◇ | – | 1.32 | 3.19 |
| `brevitas-bnn-pynq-cnv-1w1a` | 85.9% → 85.9% | -0.77 pt | 0.339 ◇ | 0.215 ◇ | 0.308 ◇ | – | 1.32 | 3.17 |
| `finn-examples-bnn-pynq-cnv-w1a1` | 85.9% → 85.9% | -0.77 pt | 0.34 ◇ | 0.213 ◇ | 0.335 ◇ | – | 1.32 | 3.15 |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | 89.8% → 89.8% | +0.06 pt | 0.366 ◇ | 0.217 ◇ | 0.335 ◇ | 0.57 ⚠ | 1.32 | 3.13 |
| `shuaiz-bireal20-cifar10` | 84.4% → 84.4% | – | 0.455 ◇ | 0.474 ◇ | 0.426 ◇ | – | 1.64 | 5.86 |
| `fda-bnn-resnet20-cifar10-mindspore` | 89.8% → 89.8% | – | 0.503 ◇ | 0.492 ◇ | 0.5 ◇ | – | 1.86 | 6.72 |
| `recu-resnet20-cifar10` | 88.3% → 88.3% | – | 0.503 ◇ | 0.544 ◇ | 0.448 ◇ | – | 1.84 | 6.65 |
| `biper-resnet20-cifar10` | 86.7% → 86.7% | – | 0.504 ◇ | 0.398 ◇ | 0.455 ◇ | – | 1.86 | 7.09 |
| `siman-resnet20-cifar10` | 87.5% → 87.5% | -2.32 pt | 0.528 ◇ | 0.509 ◇ | 0.636 ◇ | – | 1.99 | 7.38 |
| `rbnn-resnet20-bireal-cifar10` | 89.8% → 89.8% | – | 0.559 ◇ | 0.515 ◇ | 0.508 ◇ | – | 2.02 | 7.6 |
| `rbnn-resnet20-cifar10` | 88.3% → 88.3% | – | 0.591 ◇ | 0.598 ◇ | 0.522 ◇ | – | 2 | 7.25 |
| `adabin-resnet20-cifar10` | 91.4% → 91.4% | -0.64 pt | 0.888 ◇ | 0.753 ◇ | 0.904 ◇ | 1.22 ⚠ | 4.02 | 15.5 |
| `ab-bnn-cifar10-reactneta` | 91.4% → 91.4% | – | 1.55 ◇ | 0.924 ◇ | 1.5 ◇ | 2.6 ⚠ | 8.01 | 20.7 |
| `biper-resnet18-cifar10` | 94.5% → 94.5% | – | 1.99 ◇ | 1.08 ◇ | 1.97 ◇ | – | 9.71 | 23.1 |
| `recu-resnet18-cifar10` | 93.8% → 93.8% | – | 2 ◇ | 1.12 ◇ | 1.97 ◇ | – | 9.88 | 23.4 |
| `biper-vgg-small-cifar10` | 96.1% → 96.1% | -1.40 pt | 2.03 ◇ | 1.06 ◇ | 1.95 ◇ | – | 10.2 | 21.9 |
| `recu-vgg-small-cifar10` | 96.1% → 96.1% | -0.84 pt | 2.04 ◇ | 1.04 ◇ | 1.95 ◇ | – | 10.2 | 21.9 |
| `siman-vgg-small-cifar10` | 95.3% → 95.3% | – | 2.11 ◇ | 1.12 ◇ | 2.06 ◇ | – | 10.9 | 22.9 |
| `rbnn-vgg-small-cifar10` | 91.4% → 91.4% | -0.50 pt | 2.11 ◇ | 1.13 ◇ | 2.05 ◇ | – | 10.9 | 23.3 |
| `siman-resnet18-cifar10` | 95.3% → 95.3% | – | 2.17 ◇ | 1.31 ◇ | 2.08 ◇ | – | 10.7 | 25.7 |
| `rbnn-resnet18-cifar10` | 94.5% → 94.5% | – | 2.17 ◇ | 1.32 ◇ | 2.07 ◇ | – | 10.7 | 25.7 |
| `ab-bnn-cifar10-reactnet18` | 93.8% → 93.8% | +0.06 pt | 2.25 ◇ | 1.23 ◇ | 2.1 ◇ | 3.72 ⚠ | 11.3 | 25.9 |
| `bnas-a-cifar10` | 93.8% → 93.8% | -0.42 pt | 8.64 ◇ | 5.22 ◇ | 7.21 ◇ | – | 50.0 | 123.4 |

### ImageNet (55 models)

![ImageNet: accuracy vs latency, Pareto frontier](docs/img/pareto_imagenet.svg)

![ImageNet int8_full: accuracy vs latency, Pareto frontier](docs/img/pareto_imagenet_int8_full.svg)

| model | accuracy | int8_full Δ | OnePlus Open 1 thread | OnePlus Open 4 threads | OnePlus Open int8 1 thread | Galaxy S22 1 thread | moto e13 1 thread | Redmi 9A 1 thread (32-bit) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `larq-zoo-quicknet-small` | 59.4% → 59.5% (val 50k) | -2.67 pt | 4.65 ◇ | 2.49 ◇ | 4.7 ◇ | 9.26 † | 23.4 | 54.7 |
| `larq-zoo-binary-alexnet` | 36.3% → 36.3% (val 50k) | -0.26 pt | 5.45 ◇ | 2.69 ◇ | 3.06 ◇ | – | 29.3 | 60.0 |
| `larq-zoo-quicknet` | 63.3% → 63.4% (val 50k) | -1.43 pt | 7.34 ◇ | 3.9 ◇ | 7.67 ◇ | – | 38.7 | 93.1 |
| `larq-zoo-xnornet` | 45.0% → 45.0% (val 50k) | -0.37 pt | 7.82 ◇ | 4.16 ◇ | 4.18 ◇ | – | 42.0 | 93.7 |
| `dabnn-bireal18-stem` | 56.4% → 56.1% (val 50k) | -0.06 pt | 7.97 ◇ | 4.25 ◇ | 6.71 ◇ | – | 45.4 | 101.8 |
| `larq-zoo-birealnet` | 57.5% → 57.5% (val 50k) | -0.62 pt | 10.3 ◇ | 5.41 ◇ | 7.06 ◇ | – | 59.6 | 125.3 |
| `dabnn-bireal18` | 56.4% → 55.9% (val 50k) | -0.28 pt | 10.3 ◇ | 5.39 ◇ | 8.5 ◇ | – | 60.0 | 128.0 |
| `larq-zoo-binary-resnet-e18` | 58.3% → 58.3% (val 50k) | -0.89 pt | 10.3 ◇ | 5.44 ◇ | 7.34 ◇ | – | 59.6 | 127.2 |
| `bitorch-resnete18-imagenet` | 60.6% → 60.5% (val 50k) | -0.74 pt | 10.4 ◇ | 5.46 ◇ | 8.08 ◇ | – | 60.1 | 129.0 |
| `nncf-resnet18-binarization-xnor` | 61.7% → 61.7% (val 50k) | -0.26 pt | 10.7 ◇ | 5.54 ◇ | 7.84 ◇ | – | 59.4 | 121.8 |
| `nncf-resnet18-binarization-dorefa` | 61.6% → 61.6% (val 50k) | -0.15 pt | 10.7 ◇ | 5.53 ◇ | 7.81 ◇ | – | 59.7 | 123.0 |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | 61.7% → 61.8% (val 50k) | -0.49 pt | 10.7 ◇ | 5.54 ◇ | 7.57 ◇ | – | 60.0 | 124.0 |
| `omz-resnet18-xnor-binary-onnx-0001` | 61.7% → 61.8% (val 50k) | -0.37 pt | 10.7 ◇ | 5.59 ◇ | 7.57 ◇ | – | 60.1 | 123.8 |
| `snn-subbit-reactnet18-w1a1-imagenet` | 65.7% → 65.8% (val 50k) | -0.91 pt | 10.8 ◇ | 5.55 ◇ | 7.83 ◇ | – | 61.0 | 132.1 |
| `rbonn-reactnet-resnet18-imagenet` | 66.7% → 66.6% (val 50k) | -0.81 pt | 10.8 ◇ | 5.53 ◇ | 7.83 ◇ | – | 61.0 | 132.5 |
| `ab-bnn-imagenet-reactnet18` | 61.4% → 61.3% (val 50k) | -0.95 pt | 10.8 ◇ | 5.53 ◇ | 7.68 ◇ | 16.9 ⚠ | 61.0 | 133.2 |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | 63.4% → 63.4% (val 50k) | -1.12 pt | 10.8 ◇ | 5.52 ◇ | 7.83 ◇ | – | 60.9 | 132.6 |
| `recu-resnet18-imagenet` | 61.2% → 61.2% (val 50k) | -1.44 pt | 10.9 ◇ | 5.74 ◇ | 7.52 ◇ | – | 61.0 | 133.0 |
| `biper-resnet18-imagenet` | 61.4% → 61.3% (val 50k) | -1.08 pt | 11.0 ◇ | 5.75 ◇ | 7.51 ◇ | – | 61.0 | 132.3 |
| `siman-resnet18-imagenet` | 60.1% → 60.2% (val 50k) | -3.21 pt | 11.0 ◇ | 6.05 ◇ | 7.49 ◇ | – | 62.1 | 139.2 |
| `rbnn-resnet18-imagenet` | 59.9% → 59.5% (val 50k) | -0.60 pt | 11.4 ◇ | 6.25 ◇ | 8.29 ◇ | – | 62.9 | 144.1 |
| `sa-bnn-resnet18-imagenet` | 61.7% → 61.7% (val 50k) | -0.48 pt | 11.5 ◇ | 5.79 ◇ | 8.25 ◇ | – | 61.5 | 134.2 |
| `tensorpack-dorefa-alexnet-w1a1-g32` | 48.5% → 48.7% (val 50k) | -0.05 pt | 11.5 ◇ | 5.81 ◇ | 7.63 ◇ | – | 60.8 | 132.5 |
| `larq-zoo-real-to-binary-net` | 65.0% → 65.0% (val 50k) | -1.46 pt | 12.0 ◇ | 6.44 ◇ | 9.23 ◇ | – | 65.6 | 156.0 |
| `larq-zoo-quicknet-xl-legacy` | 67.0% → 67.1% (val 50k) | -1.97 pt | 13.0 ◇ | 7.39 ◇ | 13.2 ◇ | – | 67.4 | 158.9 |
| `larq-zoo-quicknet-large` | 66.9% → 66.7% (val 50k) | -1.62 pt | 13.2 ◇ | 7.2 ◇ | 13.1 ◇ | – | 66.7 | 158.5 |
| `bi-vit-deit-tiny` | 28.7% → 15.6% (val 50k) | +0.76 pt | 13.8 ◇ | 8.06 ◇ | 12.3 ◇ | 42.1 ⚠ | 67.1 | 196.3 |
| `recu-resnet34-imagenet` | 65.2% → 65.1% (val 50k) | -4.27 pt | 17.2 ◇ | 9.12 ◇ | 13.6 ◇ | – | 89.9 | 198.5 |
| `biper-resnet34-imagenet` | 65.6% → 65.4% (val 50k) | -0.74 pt | 17.3 ◇ | 9.11 ◇ | 13.6 ◇ | – | 89.9 | 198.2 |
| `ab-bnn-imagenet-reactnet34` | 65.2% → 64.9% (val 50k) | -0.52 pt | 17.3 ◇ | 8.92 ◇ | 14.5 ◇ | 27.0 ⚠ | 90.6 | 201.6 |
| `siman-resnet34-imagenet` | 63.9% → 63.6% (val 50k) | -14.10 pt | 17.8 ◇ | 9.82 ◇ | 14.7 ◇ | – | 92.6 | 209.9 |
| `rbnn-resnet34-imagenet` | 63.1% → 62.9% (val 50k) | -0.24 pt | 18.1 ◇ | 10.0 ◇ | 13.4 ◇ | – | 94.2 | 215.4 |
| `ab-bnn-imagenet-reactneta` | 66.9% → 66.7% (val 50k) | -0.47 pt | 19.5 ◇ | 10.0 ◇ | 19.5 ◇ | 19.6 ⚠ | 96.1 | 269.0 |
| `larq-zoo-binary-densenet28` | 60.9% → 60.9% (val 50k) | -0.64 pt | 21.1 ◇ | 11.1 ◇ | 14.6 ◇ | – | 113.5 | 249.8 |
| `bitorch-densenet28-imagenet` | 58.7% → 58.6% (val 50k) | -0.94 pt | 22.5 ◇ | 11.8 ◇ | 14.5 ◇ | – | 119.6 | 265.6 |
| `larq-zoo-dorefanet` | 53.4% → 53.4% (val 50k) | -0.65 pt | 23.2 ◇ | 12.0 ◇ | 19.9 ◇ | – | 117.8 | 218.6 |
| `bnext-tiny` | 72.4% → 72.3% (val 50k) | -3.01 pt | 24.0 ◇ | 12.4 ◇ | 22.1 ◇ | – | 115.5 | 342.9 |
| `larq-zoo-binary-densenet37` | 62.9% → 62.9% (val 50k) | -41.28 pt | 24.3 ◇ | 12.9 ◇ | 19.8 ◇ | – | 129.0 | 286.6 |
| `larq-zoo-meliusnet22` | 62.4% → 62.6% (val 50k) | -57.38 pt | 24.8 ◇ | 13.0 ◇ | 23.0 ◇ | – | 127.4 | 311.3 |
| `bnas-d-imagenet` | 57.7% → 57.8% (val 50k) | -57.46 pt | 26.1 ◇ | 14.4 ◇ | 16.0 ◇ | – | 139.1 | 318.7 |
| `bitorch-meliusnet22-imagenet` | 64.3% → 64.4% (val 50k) | -64.10 pt | 26.2 ◇ | 13.9 ◇ | 19.6 ◇ | – | 137.5 | 323.7 |
| `bhvit-tiny-binary` | 64.0% → 63.8% (val 50k) | -1.01 pt | 28.4 ◇ | 14.4 ◇ | 29.0 ◇ | 46.6 ⚠ | 141.1 | 572.8 |
| `bhvit-tiny-somefp` | 66.0% → 65.8% (val 50k) | -4.89 pt | 30.8 ◇ | 15.6 ◇ | 33.9 ◇ | 58.8 ⚠ | 155.0 | 595.9 |
| `bnas-e-imagenet` | 58.3% → 58.2% (val 50k) | -1.71 pt | 32.5 ◇ | 18.3 ◇ | 23.3 ◇ | – | 183.2 | 393.7 |
| `larq-zoo-binary-densenet45` | 64.6% → 64.5% (val 50k) | -43.86 pt | 32.5 ◇ | 17.2 ◇ | 27.0 ◇ | – | 172.4 | 382.8 |
| `omz-resnet50-binary-0001-fp16` | 70.7% → 70.8% (val 50k) | -0.34 pt | 33.6 ◇ | 17.3 ◇ | 19.9 ◇ | – | 181.9 | 439.7 |
| `omz-resnet50-binary-0001` | 70.7% → 70.6% (val 50k) | -0.31 pt | 33.7 ◇ | 17.3 ◇ | 19.9 ◇ | – | 182.0 | 439.6 |
| `bi-vit-deit-small` | 40.9% → 16.8% (val 50k) | -1.39 pt | 35.0 ◇ | 19.6 ◇ | 32.3 ◇ | 113.1 ⚠ | 168.5 | 464.9 |
| `bhvit-small-binary` | 68.4% → 68.5% (val 50k) | -2.70 pt | 43.6 ◇ | 22.2 ◇ | 53.5 ◇ | 85.8 ⚠ | 216.8 | 881.9 |
| `larq-zoo-binary-densenet37-dilated` | 64.3% → 64.3% (val 50k) | -59.00 pt | 44.1 ◇ | 22.7 ◇ | 33.7 ◇ | – | 239.2 | 530.8 |
| `bhvit-small-somefp` | 70.1% → 70.1% (val 50k) | -1.45 pt | 46.0 ◇ | 23.3 ◇ | 52.8 ◇ | 106.9 ⚠ | 230.6 | 906.4 |
| `bnext-small` | 76.1% → 76.0% (val 50k) | -2.01 pt | 46.9 ◇ | 23.3 ◇ | 43.6 ◇ | – | 224.6 | 639.5 |
| `bnext-middle` | 78.3% → 78.1% (val 50k) | -1.10 pt | 80.7 ◇ | 40.2 ◇ | 77.5 ◇ | – | 391.0 | 1075.4 |
| `bnas-h-imagenet` | 63.7% → 63.7% (val 50k) | -0.41 pt | 107.5 ◇ | 53.3 ◇ | 68.1 ◇ | – | 562.3 | 1255.1 |
| `bnext-large` | 80.6% → 80.5% (val 50k) | -0.87 pt | 182.4 ◇ | 90.2 ◇ | 177.8 ◇ | – | 886.3 | 2432.4 |

### Other tasks (54 models)

| model | task | accuracy | int8_full Δ | OnePlus Open 1 thread | OnePlus Open 4 threads | OnePlus Open int8 1 thread | Galaxy S22 1 thread | moto e13 1 thread | Redmi 9A 1 thread (32-bit) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `brevitas-bnn-pynq-tfc-1w1a` | mnist | 96.9% → 96.9% | – | 0.001 ◇ | 0.013 ◇ | 0.006 ◇ | – | 0.004 | 0.012 |
| `finn-examples-bnn-pynq-tfc-w1a1` | mnist | 96.9% → 96.9% | – | 0.001 ◇ | 0.012 ◇ | 0.003 ◇ | – | 0.004 | 0.012 |
| `brevitas-bnn-pynq-sfc-1w1a` | mnist | 100.0% → 100.0% | – | 0.003 ◇ | 0.025 ◇ | 0.003 ◇ | – | 0.013 | 0.033 |
| `bnn-pynq-lfc-w1a1-mnist` | mnist | 100.0% → 100.0% | – | 0.019 ◇ | 0.053 ◇ | 0.018 ◇ | – | 0.077 | 0.193 |
| `brevitas-bnn-pynq-lfc-1w1a` | mnist | 100.0% → 100.0% | – | 0.019 ◇ | 0.052 ◇ | 0.018 ◇ | – | 0.077 | 0.19 |
| `finn-examples-cnv-w1a1-gtsrb` | gtsrb | 93.8% → 93.8% | – | 0.317 ◇ | 0.209 ◇ | 0.253 ◇ | – | 1.31 | 3.14 |
| `bnn-pynq-cnv-w1a1-gtsrb` | gtsrb | 93.8% → 93.8% | – | 0.341 ◇ | 0.209 ◇ | 0.269 ◇ | – | 1.32 | 3.19 |
| `brillsson-quicknet` | non-semantic speech representation | 90.7% → 90.7% | – | 1.03 ◇ | 0.679 ◇ | 0.883 ◇ | – | 5.02 | 13.3 |
| `ab-bnn-cifar100-reactneta` | cifar100 | 71.9% → 71.9% | -0.31 pt | 1.56 ◇ | 0.929 ◇ | 1.56 ◇ | 2.63 ⚠ | 7.98 | 21.0 |
| `larqface-quicknet-msceleb` | face-recognition | 99.0% → 99.0% | – | 1.8 ◇ | 1.15 ◇ | 1.68 ◇ | – | 8.99 | 22.2 |
| `ab-bnn-cifar100-reactnet18` | cifar100 | 69.5% → 69.5% | – | 2.25 ◇ | 1.23 ◇ | 2.11 ◇ | 3.72 ⚠ | 11.3 | 26.0 |
| `brillsson-binarydensenet28` | non-semantic speech representation | 90.5% → 90.5% | – | 2.34 ◇ | 1.55 ◇ | 1.96 ◇ | – | 12.9 | 30.2 |
| `brillsson-meliusnet22` | non-semantic speech representation | 91.9% → 91.9% | – | 3.19 ◇ | 2.15 ◇ | 2.9 ◇ | – | 17.7 | 41.7 |
| `larqface-binarydensenet28-msceleb` | face-recognition | 99.2% → 99.2% | – | 5.18 ◇ | 3.1 ◇ | 3.99 ◇ | – | 29.5 | 65.1 |
| `larqface-binarydensenet37-msceleb` | face-recognition | 99.2% → 99.2% | – | 5.99 ◇ | 3.63 ◇ | 4.79 ◇ | – | 33.4 | 74.0 |
| `larqface-binaryfacenet-msceleb` | face-recognition | 95.1% → 95.1% | – | 6.21 ◇ | 3.87 ◇ | 6.2 ◇ | – | 27.3 | 100.3 |
| `larqface-binarydensenet45-msceleb` | face-recognition | 99.2% → 99.2% | – | 8.29 ◇ | 4.92 ◇ | 6.52 ◇ | – | 44.2 | 98.4 |
| `bimatting-bimobilenet-backbone` | imagenet | 55.5% → 55.5% | -22.86 pt | 8.51 ◇ | 4.8 ◇ | 8.36 ◇ | – | 42.0 | 140.4 |
| `ida-det-bires18-imagenet-backbone` | imagenet | 65.6% → 65.6% | -0.64 pt | 10.8 ◇ | 5.55 ◇ | 7.83 ◇ | – | 61.0 | 131.7 |
| `bbcu-sr-m-x4` | single-image-super-resolution-x4 | 27.22 dB → 27.22 dB | – | 14.5 ◇ | 7.1 ◇ | 16.0 ◇ | 24.4 ⚠ | 75.0 | 196.3 |
| `bbcu-sr-m-x2` | single-image-super-resolution-x2 | 31.89 dB → 31.89 dB | – | 16.4 ◇ | 8.85 ◇ | 16.9 ◇ | 25.6 ⚠ | 85.9 | 223.2 |
| `s2bnn-reacta-distill-only` | imagenet self-supervised representation learning | – → 0.96370 | – | 19.7 ◇ | 10.2 ◇ | 19.9 ◇ | – | 96.2 | 262.1 |
| `burn-reacta-stage2` | imagenet self-supervised representation learning | – → 0.99615 | – | 20.0 ◇ | 10.5 ◇ | 20.0 ◇ | – | 97.9 | 265.4 |
| `ftl-deepfake-bnext-t-cocofake-unfrozen` | deepfake-image-detection | – → – | – | 25.5 ◇ | 13.3 ◇ | 26.1 ◇ | – | 126.4 | 382.1 |
| `ftl-deepfake-bnext-t-dffd` | deepfake-image-detection | – → – | – | 25.6 ◇ | 13.3 ◇ | 26.6 ◇ | – | 126.9 | 383.7 |
| `ftl-deepfake-bnext-t-dffd-unfrozen` | deepfake-image-detection | – → – | – | 25.6 ◇ | 13.3 ◇ | 26.3 ◇ | – | 126.5 | 383.9 |
| `ftl-deepfake-bnext-t-cifake-unfrozen` | deepfake-image-detection | – → – | – | 25.7 ◇ | 13.3 ◇ | 26.3 ◇ | – | 127.8 | 383.0 |
| `ftl-deepfake-bnext-t-cifake` | deepfake-image-detection | – → – | – | 25.7 ◇ | 13.3 ◇ | 26.2 ◇ | – | 126.9 | 382.6 |
| `ftl-deepfake-bnext-t-cocofake` | deepfake-image-detection | – → – | – | 25.8 ◇ | 13.2 ◇ | 26.3 ◇ | – | 127.1 | 382.8 |
| `bsrc2023-binary-sr-x2-v0` | single-image-super-resolution-x2 | 31.88 dB → 31.88 dB | – | 25.8 ◇ | 13.8 ◇ | 15.8 ◇ | – | 144.9 | 329.5 |
| `bibert-bert-base-sst2` | glue-sst2 | 89.8% → 89.8% | -0.34 pt | 27.2 ◇ | 13.7 ◇ | 27.8 ◇ | – | 135.4 | 321.5 |
| `bbcu-deblocking` | jpeg-artifact-reduction | 31.33 dB → 31.33 dB | – | 27.3 ◇ | 13.3 ◇ | 28.2 ◇ | 33.1 ⚠ | 143.9 | 369.1 |
| `bbcu-denoise` | color-image-denoising | 29.74 dB → 29.74 dB | – | 27.9 ◇ | 13.6 ◇ | 28.2 ◇ | 34.0 ⚠ | 147.7 | 373.9 |
| `bbcu-sr-l-x2` | single-image-super-resolution-x2 | 32.03 dB → 32.03 dB | – | 35.4 ◇ | 18.0 ◇ | 19.5 ◇ | 36.6 ⚠ | 197.2 | 437.3 |
| `bsrc2023-binary-sr-x4-v0` | single-image-super-resolution-x4 | 27.22 dB → 27.22 dB | – | 38.6 ◇ | 17.6 ◇ | 25.9 ◇ | – | 224.9 | 549.5 |
| `bsrc2023-binary-sr-x4-v1` | single-image-super-resolution-x4 | 27.32 dB → 27.32 dB | – | 43.6 ◇ | 20.3 ◇ | 29.7 ◇ | – | 250.8 | 652.3 |
| `bibert-bert-base-mrpc` | glue-mrpc | 76.6% → 76.6% | +0.98 pt | 45.2 ◇ | 21.4 ◇ | 49.1 ◇ | 148.9 ⚠ | 225.4 | 592.9 |
| `ftl-deepfake-bnext-s-cifake` | deepfake-image-detection | – → – | – | 48.0 ◇ | 24.1 ◇ | 48.0 ◇ | – | 233.9 | 677.1 |
| `ftl-deepfake-bnext-s-dffd-unfrozen` | deepfake-image-detection | – → – | – | 48.0 ◇ | 24.1 ◇ | 48.5 ◇ | – | 233.3 | 678.3 |
| `ftl-deepfake-bnext-s-cifake-unfrozen` | deepfake-image-detection | – → – | – | 48.0 ◇ | 24.1 ◇ | 48.4 ◇ | – | 233.0 | 677.1 |
| `ftl-deepfake-bnext-s-cocofake-unfrozen` | deepfake-image-detection | – → – | – | 48.0 ⚠ | 24.1 ◇ | 48.2 ◇ | – | 233.1 | 679.9 |
| `ftl-deepfake-bnext-s-dffd` | deepfake-image-detection | – → – | – | 48.1 ◇ | 24.0 ◇ | 48.7 ◇ | – | 233.8 | 678.3 |
| `ftl-deepfake-bnext-s-cocofake` | deepfake-image-detection | – → – | – | 48.2 ◇ | 24.1 ◇ | 48.5 ◇ | – | 233.8 | 676.7 |
| `bibert-bert-base-mnli` | glue-mnli | 61.7% → 61.7% | +0.54 pt | 52.3 ◇ | 24.7 ◇ | 54.8 ◇ | 124.7 ⚠ | 258.8 | 660.1 |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | 27.36 dB → 27.36 dB | – | 52.7 ◇ | 24.6 ◇ | 19.6 ◇ | 53.5 ⚠ | 296.4 | 625.6 |
| `bibert-bert-base-qqp` | glue-qqp | 83.6% → 83.6% | -0.59 pt | 57.9 ◇ | 27.2 ◇ | 60.0 ◇ | 160.7 ⚠ | 288.8 | 730.6 |
| `gsb-deit-s-cifar100-stage2` | cifar100 | 70.3% → 69.5% | -1.99 pt | 75.2 ◇ | 38.2 ◇ | 64.2 ◇ | – | 407.5 | 1533.4 |
| `ftl-deepfake-bnext-m-cocofake` | deepfake-image-detection | – → – | – | 81.9 ◇ | 41.1 ◇ | 82.1 ◇ | – | 399.9 | 1118.9 |
| `ftl-deepfake-bnext-m-cifake` | deepfake-image-detection | – → – | – | 82.0 ◇ | 41.2 ◇ | 82.0 ◇ | – | 400.3 | 1125.2 |
| `ftl-deepfake-bnext-m-dffd` | deepfake-image-detection | – → – | – | 82.0 ◇ | 41.0 ◇ | 82.0 ◇ | – | 400.0 | 1121.6 |
| `ftl-deepfake-bnext-m-dffd-unfrozen` | deepfake-image-detection | – → – | – | 82.2 ◇ | 41.1 ◇ | 82.0 ◇ | – | 399.1 | 1119.2 |
| `ftl-deepfake-bnext-m-cifake-unfrozen` | deepfake-image-detection | – → – | – | 82.2 ◇ | 41.2 ◇ | 82.1 ◇ | – | 400.3 | 1123.8 |
| `ftl-deepfake-bnext-m-cocofake-unfrozen` | deepfake-image-detection | – → – | – | 82.3 ◇ | 41.1 ◇ | 82.1 ◇ | – | 400.0 | 1118.8 |
| `bsrc2023-binary-sr-x2-v1` | single-image-super-resolution-x2 | 32.22 dB → 32.22 dB | – | 91.0 ◇ | 45.0 ◇ | 61.5 ◇ | – | 543.8 | 1336.3 |

Unmarked = clean (every repeat at full clocks). ◇ = the phone's own power policy held the clock ceiling below the maximum with no thermal event (stock settings, screen off; the ceiling is recorded per run); ⚠ = at least one repeat ran under a thermal CPU-frequency cap (e.g. a phone charging over USB); † = run made before the frequency-cap check existed (caps unknown). Runs per device and validity: Galaxy S22: 185 thermal-throttled, 24 unknown-caps; OnePlus Open: 6 thermal-throttled, 4549 vendor-policy; moto e13: 2940 clean; Redmi 9A: 2087 clean. 

Every phone with 1 and 4 threads and int8: [docs/SPEED.md](docs/SPEED.md). On-device correctness: Galaxy S22: 381/381 correctness checks passed on 114 models, 2026-11-02 to 2026-11-12 (outputs byte-identical to a validated emulated-ARM or reference-phone run of the same code); OnePlus Open: 415/415 correctness checks passed on 132 models, 2026-11-14 (outputs byte-identical to a validated emulated-ARM or reference-phone run of the same code); moto e13: 503/503 correctness checks passed on 132 models, 2026-11-14 (outputs byte-identical to a validated emulated-ARM or reference-phone run of the same code); Redmi 9A: 264/264 correctness checks passed on 132 models, 2026-11-14 (exact stage bit-identical to the numpy executor, float stage within tolerance of the float64 reference).
<!-- /generated:speed -->

The phones: `results/devices.json`. All runs with their clock state: `results/speed_<device>.json`; x86-64 host runs
(older runtime, labelled): `results/speed_x86.json`. Method: [docs/benchmarking.md](docs/benchmarking.md).
</details>

**Your phone.** `scripts/bench_my_phone.sh` measures Bitino on any Android phone over adb (no root, about 5 minutes,
outputs checked against the reference) and writes one JSON file for the community table (`results/community/`):
[docs/benchmark-your-phone.md](docs/benchmark-your-phone.md).

## Model zoo

Selected models (all converted models: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md); a searchable version is on the
[website](https://federicofontana.dev/bitino/)):

<!-- generated:zoo_top -->
| model | task | framework | accuracy | eval set (n) | int8_full Δ | fastest phone ms (1 thread) | weights licence | release |
|---|---|---|---|---|---|---|---|---|
| `ab-bnn-cifar10-reactnet18` | cifar10-classification | pytorch | 93.8% → 93.8% | cifar10-test-first-128 (128) | +0.06 pt (cifar10-test-first-10000, n=10000) | 2.25 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactnet18` | imagenet-classification | pytorch | 61.4% → 61.3% (val 50k) | ImageNet val (50000) | -0.95 pt (ImageNet val (full), n=50000) | 10.8 ◇ (OnePlus Open) | MIT (HF model card metadata 'license: mit'); app | yes |
| `ab-bnn-imagenet-reactnet34` | imagenet-classification | pytorch | 65.2% → 64.9% (val 50k) | ImageNet val (50000) | -0.52 pt (ImageNet val (full), n=50000) | 17.3 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactneta` | imagenet-classification | pytorch | 66.9% → 66.7% (val 50k) | ImageNet val (50000) | -0.47 pt (ImageNet val (full), n=50000) | 19.5 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | pytorch | 27.36 dB → 27.36 dB | set5-set14-b100-x4-lr32 (64) | – | 52.7 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bhvit-tiny-binary` | imagenet-classification | pytorch | 64.0% → 63.8% (val 50k) | ImageNet val (50000) | -1.01 pt (ImageNet val (full), n=50000) | 28.4 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `bi-vit-deit-tiny` | imagenet-classification | pytorch | 28.7% → 15.6% (val 50k) | ImageNet val (50000) | +0.76 pt (ImageNet val (full), n=50000) | 13.8 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bibert-bert-base-sst2` | glue-sst2 | pytorch | 89.8% → 89.8% | sst2-dev-128 (128) | -0.34 pt (sst2-dev-872, n=872) | 27.2 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bnext-tiny` | imagenet-classification | pytorch | 72.4% → 72.3% (val 50k) | ImageNet val (50000) | -3.01 pt (ImageNet val (full), n=50000) | 24.0 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `brevitas-bnn-pynq-cnv-1w1a` | cifar10-classification | pytorch+brevitas | 85.9% → 85.9% | cifar10-test-first-128 (128) | -0.77 pt (cifar10-test-first-10000, n=10000) | 0.339 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `larq-zoo-birealnet` | imagenet-classification | larq_zoo | 57.5% → 57.5% (val 50k) | ImageNet val (50000) | -0.62 pt (ImageNet val (full), n=50000) | 10.3 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet` | imagenet-classification | larq_zoo | 63.3% → 63.4% (val 50k) | ImageNet val (50000) | -1.43 pt (ImageNet val (full), n=50000) | 7.34 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet-large` | imagenet-classification | larq_zoo | 66.9% → 66.7% (val 50k) | ImageNet val (50000) | -1.62 pt (ImageNet val (full), n=50000) | 13.2 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet-small` | imagenet-classification | larq_zoo | 59.4% → 59.5% (val 50k) | ImageNet val (50000) | -2.67 pt (ImageNet val (full), n=50000) | 4.65 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `omz-resnet18-xnor-binary-onnx-0001` | imagenet-classification | openvino-ir | 61.7% → 61.8% (val 50k) | ImageNet val (50000) | -0.37 pt (ImageNet val (full), n=50000) | 10.7 ◇ (OnePlus Open) | Apache-2.0 | yes |
| `omz-resnet50-binary-0001` | imagenet-classification | openvino-ir | 70.7% → 70.6% (val 50k) | ImageNet val (50000) | -0.31 pt (ImageNet val (full), n=50000) | 33.7 ◇ (OnePlus Open) | Apache-2.0 (model.yml license field points to th | yes |
<!-- /generated:zoo_top -->

Accuracy: ImageNet classifiers show their published top-1 → Bitino's top-1 on all 50,000 validation images (original
JPEGs, each model's own preprocessing). The other models show conversion checks on small subsets (n = 128 for most):
the original model, run by its own framework, against Bitino's exact stage.

**Weights and licences.**
- Converted files are attached to the releases only when the weights licence is explicitly permissive (Apache-2.0 /
  MIT / BSD stated for the weights). Each file comes with its licence text and attribution (`bitino list --release`).
- Everything else is one `bitino fetch` away from its original source.
- The registry records the code and weights licences of every model separately (`registry/models.json`).

## What is supported

| | |
|---|---|
| **Model format** | QONNX (profile `bitino-qonnx/1`: standard ONNX opsets 17–21 + `BipolarQuant` / `IntQuant`; [docs/qonnx-profile.md](docs/qonnx-profile.md)), checked by `bitino check-qonnx`, compiled by `bitino import-qonnx` into `.bitino` |
| **Checkpoint formats** | Larq Zoo and Keras/Larq `.h5`; PyTorch state dicts and Lightning checkpoints (AB-BNN, ReActNet / Bi-Real / IR-Net families, BNext, bitorch, BNAS, BHViT, Bi-ViT, BiBERT, restoration and super-resolution BNNs, …); OpenVINO IR; ONNX (daBNN); FINN-ONNX / QONNX; Brevitas; NNCF; MindSpore; tensorpack and Theano/Lasagne `.npz` |
| **Operators** | Binary Conv2D / Dense / MatMul with exact padding correction and fused thresholds; float Conv2D / depthwise / Dense / pooling / elementwise; transformer blocks (attention, LayerNorm, GELU, softmax); full-integer `I8*` operators ([docs/operators.md](docs/operators.md)) |
| **Kernels** | Scalar reference for every op; NEON (`binary_neon`, `float_opt`, `int8_neon` with SDOT / SMMLA) on arm64 and AArch32; x86-64 AVX2 / AVX-VNNI. Selected at run time from the CPU features |
| **Stages** | `reference`, `neon` (exact), `optimized` (float fusion), `optimized_int8`, `int8_full` (full integer) |
| **Targets** | Android arm64-v8a and armeabi-v7a (command-line runtime API 30+, AAR API 24+), Linux aarch64 and x86-64. Python on Linux x86-64 / aarch64, macOS arm64, Windows x64 (correctness-tested in CI; speed is measured on the phones and the x86 host only) |

More: [architecture](docs/architecture.md) · [model format](docs/model-format.md) ·
[numerical correctness](docs/numerical-correctness.md) · [int8](docs/int8.md) · [threads](docs/threads.md) ·
[reproducing every number](docs/reproduce.md).

## Add a model

A new network needs a QONNX file, not a converter.

1. Add the network to `registry/models.json`: checkpoint URL, SHA-256, licences, provenance.
2. Write its QONNX file: Brevitas and FINN export QONNX directly; PyTorch, Keras / Larq and the other front ends have a
   writer (`tools/export_*_qonnx.py`). `bitino check-qonnx model.onnx` checks it against the profile.
3. `bitino import-qonnx model.onnx --model-id ID --out-dir DIR` compiles it into `.bitino`; validate it with `tools/validate_model.py` on the host,
   then on a device.
4. Add its route and recipe (`tools/build_from_qonnx.py --write-routes`, `tools/build_recipes.py --artifacts <dir>`);
   `tools/build_site.py` refreshes the tables.

Guides: [docs/qonnx-standard.md](docs/qonnx-standard.md) (the standard),
[docs/converting-a-new-bnn.md](docs/converting-a-new-bnn.md) (step by step),
[docs/adding-a-model.md](docs/adding-a-model.md).

## Contributing

The most useful contributions, in order:
1. **Speed on your phone.** Every new phone makes the benchmark tables more useful, and it takes one command:
   [docs/benchmark-your-phone.md](docs/benchmark-your-phone.md).
2. **A BNN with public weights** that is not in the zoo yet.
3. **Bugs and wrong numbers.** If a model disagrees with its paper or framework, please open an issue.

Questions and ideas: [Discussions](https://github.com/fedeloper/bitino/discussions). Changes: [CHANGELOG.md](CHANGELOG.md).

## Limitations

- **Speed caveats.**
  - The OnePlus Open runs use the phone's stock power policy, which holds the clocks below their maximum with the
    screen off (◇; the ceiling is recorded per run).
  - The Galaxy S22 numbers come from an older runtime on a thermally capped phone (⚠).
  - The x86 host numbers come from an older runtime.
- **Few phones.** The OnePlus Open is the reference; the Galaxy S22, moto e13 and Redmi 9A (32-bit) are also
  measured. Other arm64 CPUs run the same code, but their speed is not measured here. There are no GPU / NPU backends.
- **int8_full** loses accuracy on some networks (see the zoo tables); the exact `neon` / `optimized` stages match the
  source framework. On the full ImageNet validation set, 48 of the 50 ImageNet int8_full files lose a median 0.84 pt
  (29 within 1 pt, 41 within 2 pt). **Do not use the int8_full files of BNAS-D (`bnas-d-imagenet`) and the bitorch
  MeliusNet-22 (`bitorch-meliusnet22-imagenet`)**: they collapse to chance level (0.3 % top-1); their float files are
  unaffected. Reproducing an int8 file bit for bit needs its calibration images (training split, not
  redistributed).
- **Accuracy beyond ImageNet.** ImageNet classifiers are measured on the full validation set; the other tasks
  (CIFAR, restoration, deepfake detection, embeddings) only on small conversion-check subsets (mostly 128 inputs).
- **Fetch recipes.**
  - 131 of 132 ran end to end from an empty cache. One is broken upstream: S2-BNN, whose OneDrive link has expired.
  - All 132 were rebuilt through the QONNX path with the recorded files (S2-BNN from a local copy): 385 of 385 float
    and derived files identical in data and graph. The int8 files need your calibration images; with the recorded
    CIFAR-10 calibration set, ReCU ResNet-20's int8 file came out identical too.
  - Google Drive can refuse a scripted download; `fetch` then prints the link to download by hand.
  - Some recipes need a second environment, listed in `envs/`.
- **File format v1** keeps the on-disk magic `BFMODEL1` from the format's original name, so older files stay readable
  ([docs/model-format.md](docs/model-format.md)).

## Citation

```bibtex
@software{fontana_bitino_2026,
  author = {Fontana, Federico},
  title  = {Bitino: 1-bit neural networks that keep your phone cool},
  year   = {2026},
  url    = {https://github.com/fedeloper/bitino}
}
```

Please also cite the papers of the networks you use (`bitino info <model-id>` prints each one).

## Licence and acknowledgements

Code: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE), third-party notes in [THIRD_PARTY.md](THIRD_PARTY.md)). Model
weights are not covered by this licence: each keeps its authors' licence. Datasets are not redistributed.

Bitino exists because many research groups published their binary networks with weights. The zoo credits each one
(`registry/models.json`, `bitino info <model-id>`). The head-to-head uses [Larq Compute Engine](https://github.com/larq/compute-engine)
and [TensorFlow Lite](https://www.tensorflow.org/lite).
