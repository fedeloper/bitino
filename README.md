<p align="center"><img src="site/assets/logo.svg" width="96" alt="Bitino logo"></p>

<h1 align="center">Bitino</h1>
<p align="center"><i>1-bit neural networks that keep your phone cool.</i><br>
by Fred · <a href="https://federicofontana.dev">federicofontana.dev</a></p>

Bitino runs **binary neural networks** (BNNs) on the CPU of an Android phone or a laptop: a dependency-free C++17
runtime with NEON (ARMv8.2 dotprod, i8mm) and AVX2 / AVX-VNNI kernels, converters that take networks from their
**original** published checkpoints, a full-integer int8 path, and a curated zoo of every BNN we could find with
public weights.

<!-- generated:stats -->
**343** binary networks investigated, **223** with verified public weights, **132** converted and validated against their original framework, **11** redistributable as converted files (explicit permissive weights licence). Fetch recipes: 131 verified end to end, 1 broken upstream.
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
* **Honest speed numbers.** Timings come only from real hardware (the Android phones in `results/devices.json`,
  an x86-64 host), with date, thread setting and clock state. Emulators are used for correctness and instruction
  counts, never for timings.

## Quickstart: Python

```bash
pip install "bitino[image]"           # native runtime inside the wheel; [image] adds Pillow
```

```python
import bitino

m = bitino.load("ab-bnn-imagenet-reactnet18")    # MIT-licensed weights: converted file from the release, SHA-256 checked
logits = m.predict_image("cat.jpg")              # the model's own preprocessing, then the runtime (numpy in, numpy out)
print(logits.argmax(), m.benchmark(threads=1), "ms on this CPU")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fedeloper/bitino/blob/main/examples/colab/bitino_quickstart.ipynb)
classify an image and time it on Colab's CPU (a cloud x86 server, not a phone: the phone numbers are below).
Wheels for Linux x86-64 / aarch64, macOS arm64 and Windows x64, any Python >= 3.10; `predict(x)` takes numpy
arrays in the model's layout (NHWC for images), `stage="neon"` is bit-identical with the reference numpy executor.
From a terminal: `bitino run ab-bnn-imagenet-reactnet18 --image cat.jpg --bench`. API: [docs/python.md](docs/python.md).
Models whose weights licence is not explicitly permissive are converted from their original checkpoint
(`bitino fetch`, below) and then load the same way.

## Five-minute quickstart: from source

```bash
git clone https://github.com/fedeloper/bitino && cd bitino
pip install -e .                      # the `bitino` CLI and Python API (compiles the runtime library with CMake)

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

## In an Android app

<!-- demo GIF: record with android/tools/record_demo.sh (docs/android-deployment.md), then restore:
<p align="center"><img src="docs/img/android_demo.gif" width="300" alt="Bitino demo app: live ImageNet top-5 from the camera with latency and fps"></p> -->

The Android library (`android/bitino`, an AAR for arm64-v8a and armeabi-v7a, API 24+) wraps the runtime in a small
Kotlin API; `android/demo` is the CameraX app above (bundled ReActNet-18, weights MIT).

```kotlin
// build.gradle.kts: implementation("io.github.fedeloper:bitino-android:0.1.0")   (until published: build the AAR,
// android/gradlew -p android :bitino:publishToMavenLocal, and add mavenLocal())
import io.github.fedeloper.bitino.Bitino

val model = Bitino.load(context, "ab-bnn-imagenet-reactnet18.bitino", threads = 1)   // asset or absolute file path
val scores = model.run(bitmap)                  // resize / crop / normalize exactly as the model's manifest says
val top5 = Bitino.topK(Bitino.softmax(scores), 5)
model.close()
```

The preprocessing reproduces the Python pipeline of each model bit for bit (Pillow resampling included; checked
against the evaluation tensors). Native code can instead link `libbitino.a` and use the C API
(`runtime/include/bitino/bitino_c.h`). Details: [docs/android-deployment.md](docs/android-deployment.md),
[docs/getting-started.md](docs/getting-started.md).

`bitino fetch` downloads the weights from where the authors published them, into a fresh directory, checks their
SHA-256, downloads the authors' model code pinned to the recorded commit, and runs the converter with `python -I`
(checkpoints are read with `torch.load(weights_only=True)` only). The converted file's data section is compared with
the published conversion: `IDENTICAL` means bit-identical weights and constants. Converters run in the source
framework's environment (`envs/`); `bitino fetch` says which one and how to point it there
(`--python larq=/path/to/python`). Without a dataset it generates a small synthetic eval set, which is enough for
conversion and runtime checks (accuracy numbers need the real one).

## Speed (measured on phones)

<!-- generated:speed -->
Median latency of one inference (batch 1) in ms. Main columns: **OnePlus Open (CPH2551)** (Qualcomm Snapdragon 8 Gen 2 (SM8550)); 1 thread = pinned to the fastest core; int8 = the full-integer file (int8_full) where measured, else the int8 kernels on the exact file. Accuracy: the original model in its own framework → Bitino's exact stage, on the small check subset; int8_full Δ: the full-integer file's paired difference against the exact stage. Fastest first.

### CIFAR-10 (23 models)

![CIFAR-10: accuracy vs latency, Pareto frontier](docs/img/pareto_cifar10.svg)

![CIFAR-10 int8_full: accuracy vs latency, Pareto frontier](docs/img/pareto_cifar10_int8_full.svg)

| model | accuracy (reference → Bitino) | int8_full Δ | OnePlus Open 1 thread | OnePlus Open 4 threads | OnePlus Open int8 1 thread | Galaxy S22 1 thread | moto e13 1 thread | Redmi 9A 1 thread (32-bit) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `brevitas-bnn-pynq-cnv-1w1a` | 85.9% → 85.9% | -0.77 pt | 0.293 ◇ | 0.192 ◇ | 0.431 ◇ | – | 1.5 | 3.35 |
| `finn-examples-bnn-pynq-cnv-w1a1` | 85.9% → 85.9% | -0.77 pt | 0.293 ◇ | 0.233 ◇ | 0.487 ◇ | – | – | – |
| `bnn-pynq-cnv-w1a1-cifar10` | 86.7% → 86.7% | +0.42 pt | 0.295 ◇ | 0.225 ◇ | 0.398 ◇ | – | – | – |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | 89.8% → 89.8% | +0.06 pt | 0.311 ◇ | 0.196 ◇ | 0.264 ◇ | 0.57 ⚠ | – | – |
| `shuaiz-bireal20-cifar10` | 84.4% → 84.4% | -0.55 pt | 0.425 ◇ | 0.403 ◇ | 0.617 ◇ | – | – | – |
| `fda-bnn-resnet20-cifar10-mindspore` | 89.8% → 89.8% | -0.55 pt | 0.466 ◇ | 0.37 ◇ | 0.532 ◇ | – | – | – |
| `biper-resnet20-cifar10` | 86.7% → 86.7% | -0.63 pt | 0.468 ◇ | 0.369 ◇ | 0.571 ◇ | – | – | – |
| `recu-resnet20-cifar10` | 88.3% → 88.3% | -0.67 pt | 0.471 ◇ | 0.552 ◇ | 0.583 ◇ | – | – | – |
| `siman-resnet20-cifar10` | 87.5% → 87.5% | -2.32 pt | 0.515 ◇ | 0.425 ◇ | 0.728 ◇ | – | – | – |
| `rbnn-resnet20-bireal-cifar10` | 89.8% → 89.8% | -1.13 pt | 0.537 ◇ | 0.562 ◇ | 0.73 ◇ | – | – | – |
| `rbnn-resnet20-cifar10` | 88.3% → 88.3% | -1.01 pt | 0.541 ◇ | 0.616 ◇ | 0.673 ◇ | – | 2.56 | 8.24 |
| `adabin-resnet20-cifar10` | 91.4% → 91.4% | -0.64 pt | 0.89 ◇ | 0.76 ◇ | 1.02 ◇ | 1.22 ⚠ | – | – |
| `ab-bnn-cifar10-reactneta` | 91.4% → 91.4% | -0.50 pt | 1.59 ◇ | 0.935 ◇ | 1.96 ◇ | 2.6 ⚠ | – | – |
| `biper-resnet18-cifar10` | 94.5% → 94.5% | -1.94 pt | 2.01 ◇ | 1.1 ◇ | 2.18 ◇ | – | – | – |
| `recu-resnet18-cifar10` | 93.8% → 93.8% | -1.32 pt | 2.04 ◇ | 1.11 ◇ | 2.19 ◇ | – | – | – |
| `biper-vgg-small-cifar10` | 96.1% → 96.1% | -1.40 pt | 2.08 ◇ | 1.05 ◇ | 2.02 ◇ | – | – | – |
| `recu-vgg-small-cifar10` | 96.1% → 96.1% | -0.84 pt | 2.09 ◇ | 1.07 ◇ | 2.01 ◇ | – | – | – |
| `siman-vgg-small-cifar10` | 95.3% → 95.3% | -0.49 pt | 2.15 ◇ | 1.13 ◇ | 2.29 ◇ | – | – | – |
| `rbnn-vgg-small-cifar10` | 91.4% → 91.4% | -0.50 pt | 2.16 ◇ | 1.18 ◇ | 2.29 ◇ | – | – | – |
| `siman-resnet18-cifar10` | 95.3% → 95.3% | -0.15 pt | 2.2 ◇ | 1.31 ◇ | 2.75 ◇ | – | – | – |
| `rbnn-resnet18-cifar10` | 94.5% → 94.5% | -0.75 pt | 2.22 ◇ | 1.31 ◇ | 2.73 ◇ | – | – | – |
| `ab-bnn-cifar10-reactnet18` | 93.8% → 93.8% | +0.06 pt | 2.29 ◇ | 1.22 ◇ | 2.57 ◇ | 3.72 ⚠ | 12.4 | 27.1 |
| `bnas-a-cifar10` | 93.8% → 93.8% | -0.42 pt | 8.91 ◇ | 5.22 ◇ | 7.11 ◇ | – | – | – |

### ImageNet (55 models)

![ImageNet: accuracy vs latency, Pareto frontier](docs/img/pareto_imagenet.svg)

![ImageNet int8_full: accuracy vs latency, Pareto frontier](docs/img/pareto_imagenet_int8_full.svg)

| model | accuracy (reference → Bitino) | int8_full Δ | OnePlus Open 1 thread | OnePlus Open 4 threads | OnePlus Open int8 1 thread | Galaxy S22 1 thread | moto e13 1 thread | Redmi 9A 1 thread (32-bit) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `larq-zoo-quicknet-small` | 59.4% → 58.6% | -1.96 pt | 4.91 ◇ | 2.6 ◇ | 5.84 ◇ | 9.26 † | 26.6 | 58.7 |
| `larq-zoo-binary-alexnet` | 28.9% → 28.9% | +0.30 pt | 5.45 ◇ | 2.69 ◇ | 3.08 ◇ | – | – | – |
| `larq-zoo-quicknet` | 66.4% → 66.4% | -0.72 pt | 7.65 ◇ | 4.02 ◇ | 8.15 ◇ | – | 43.4 | 94.6 |
| `larq-zoo-xnornet` | 34.4% → 34.4% | -0.36 pt | 7.8 ◇ | 4.25 ◇ | 4.2 ◇ | – | – | – |
| `dabnn-bireal18-stem` | 58.6% → 58.6% | -0.42 pt | 8.23 ◇ | 4.37 ◇ | 8.37 ◇ | – | – | – |
| `larq-zoo-birealnet` | 57.8% → – | -0.20 pt | 10.6 ◇ | 5.53 ◇ | 8.86 ◇ | – | 69.3 | 131.8 |
| `larq-zoo-binary-resnet-e18` | 55.5% → 55.5% | -0.92 pt | 10.6 ◇ | 5.58 ◇ | 8.79 ◇ | – | – | – |
| `dabnn-bireal18` | 58.6% → 58.6% | +0.16 pt | 10.7 ◇ | 5.54 ◇ | 8.85 ◇ | – | – | – |
| `bitorch-resnete18-imagenet` | 64.8% → 62.5% | -0.84 pt | 10.8 ◇ | 5.57 ◇ | 8.42 ◇ | – | – | – |
| `nncf-resnet18-binarization-dorefa` | 62.5% → 62.5% | +0.14 pt | 10.9 ◇ | 5.62 ◇ | 7.99 ◇ | – | – | – |
| `nncf-resnet18-binarization-xnor` | 61.7% → 61.7% | -0.12 pt | 10.9 ◇ | 5.64 ◇ | 8.05 ◇ | – | – | – |
| `omz-resnet18-xnor-binary-onnx-0001` | 59.4% → 59.4% | -1.20 pt | 10.9 ◇ | 5.68 ◇ | 8.26 ◇ | – | 68.4 | 129.1 |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | 64.1% → 64.1% | +0.00 pt | 10.9 ◇ | 5.65 ◇ | 8.26 ◇ | – | – | – |
| `snn-subbit-reactnet18-w1a1-imagenet` | 64.8% → 64.8% | -0.20 pt | 11.0 ◇ | 5.64 ◇ | 8.83 ◇ | – | – | – |
| `rbonn-reactnet-resnet18-imagenet` | 65.6% → 65.6% | -0.56 pt | 11.0 ◇ | 5.65 ◇ | 8.84 ◇ | – | – | – |
| `ab-bnn-imagenet-reactnet18` | 65.6% → 64.8% | -0.34 pt | 11.1 ◇ | 5.64 ◇ | 8.98 ◇ | 16.9 ⚠ | 70.6 | 138.5 |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | 64.1% → 64.8% | -0.68 pt | 11.1 ◇ | 5.64 ◇ | 8.8 ◇ | – | – | – |
| `recu-resnet18-imagenet` | 59.4% → 59.4% | -1.74 pt | 11.2 ◇ | 5.85 ◇ | 8.59 ◇ | – | 70.6 | 139.4 |
| `tensorpack-dorefa-alexnet-w1a1-g32` | 46.1% → 46.1% | -0.32 pt | 11.2 ◇ | 5.84 ◇ | 7.71 ◇ | – | – | – |
| `siman-resnet18-imagenet` | 54.7% → 54.7% | -4.58 pt | 11.3 ◇ | 6.14 ◇ | 9.41 ◇ | – | – | – |
| `biper-resnet18-imagenet` | 59.4% → 59.4% | -1.86 pt | 11.3 ◇ | 5.84 ◇ | 8.55 ◇ | – | – | – |
| `rbnn-resnet18-imagenet` | 53.1% → 53.1% | -0.90 pt | 11.7 ◇ | 6.36 ◇ | 9.54 ◇ | – | – | – |
| `sa-bnn-resnet18-imagenet` | 61.7% → 62.5% | -0.78 pt | 11.7 ◇ | 5.88 ◇ | 9.23 ◇ | – | – | – |
| `larq-zoo-real-to-binary-net` | 64.8% → 64.8% | -1.12 pt | 12.3 ◇ | 6.55 ◇ | 10.3 ◇ | – | – | – |
| `larq-zoo-quicknet-xl-legacy` | 64.1% → 64.8% | -0.92 pt | 13.4 ◇ | 7.53 ◇ | 16.1 ◇ | – | – | – |
| `bi-vit-deit-tiny` | 15.6% → 15.6% | -0.34 pt | 13.6 ◇ | 8.02 ◇ | 15.3 ◇ | 42.1 ⚠ | 69.5 | – |
| `larq-zoo-quicknet-large` | 71.1% → 70.3% | -1.96 pt | 13.6 ◇ | 7.37 ◇ | 15.1 ◇ | – | 74.6 | 157.2 |
| `recu-resnet34-imagenet` | 68.8% → 68.8% | -4.56 pt | 17.4 ◇ | 9.27 ◇ | 15.5 ◇ | – | – | – |
| `ab-bnn-imagenet-reactnet34` | 67.2% → 68.0% | -0.62 pt | 17.6 ◇ | 9.07 ◇ | 16.3 ◇ | 27.0 ⚠ | – | – |
| `biper-resnet34-imagenet` | 66.4% → 66.4% | -1.26 pt | 17.8 ◇ | 9.3 ◇ | 15.5 ◇ | – | – | – |
| `siman-resnet34-imagenet` | 59.4% → 60.9% | -14.10 pt | 17.9 ◇ | 10.0 ◇ | 17.4 ◇ | – | – | – |
| `rbnn-resnet34-imagenet` | 64.1% → 63.3% | -0.52 pt | 18.3 ◇ | 10.1 ◇ | 17.4 ◇ | – | – | – |
| `ab-bnn-imagenet-reactneta` | 66.4% → 65.6% | -0.58 pt | 20.2 ◇ | 10.3 ◇ | 23.1 ◇ | 19.6 ⚠ | – | – |
| `larq-zoo-binary-densenet28` | 60.9% → 60.9% | -0.66 pt | 21.5 ◇ | 11.2 ◇ | 16.7 ◇ | – | – | – |
| `bitorch-densenet28-imagenet` | 59.4% → 59.4% | -0.76 pt | 22.9 ◇ | 12.0 ◇ | 16.7 ◇ | – | – | – |
| `larq-zoo-dorefanet` | 46.1% → 46.1% | -0.64 pt | 23.1 ◇ | 11.9 ◇ | 20.0 ◇ | – | – | – |
| `larq-zoo-binary-densenet37` | 66.4% → 65.6% | -41.28 pt | 24.6 ◇ | 13.0 ◇ | 23.2 ◇ | – | – | – |
| `larq-zoo-meliusnet22` | 65.6% → 66.4% | -57.38 pt | 24.9 ◇ | 13.1 ◇ | 27.0 ◇ | – | – | – |
| `bnext-tiny` | 69.5% → 68.0% | -3.84 pt | 25.0 ◇ | 12.8 ◇ | 27.7 ◇ | – | 137.1 | 419.6 |
| `bnas-d-imagenet` | 60.2% → 60.2% | -54.94 pt | 26.6 ◇ | 14.6 ◇ | 16.0 ◇ | – | – | – |
| `bitorch-meliusnet22-imagenet` | 66.4% → 65.6% | -61.52 pt | 26.7 ◇ | 14.1 ◇ | 22.4 ◇ | – | – | – |
| `bhvit-tiny-binary` | 64.1% → 62.5% | -0.74 pt | 29.0 ◇ | 14.5 ◇ | 38.5 ◇ | 46.6 ⚠ | 148.6 | 636.4 |
| `bhvit-tiny-somefp` | 65.6% → 65.6% | -6.20 pt | 31.3 ◇ | 15.6 ◇ | 37.9 ◇ | 58.8 ⚠ | – | – |
| `bnas-e-imagenet` | 56.2% → 56.2% | -0.92 pt | 33.2 ◇ | 18.4 ◇ | 22.2 ◇ | – | – | – |
| `larq-zoo-binary-densenet45` | 60.9% → 61.7% | -43.86 pt | 33.3 ◇ | 17.4 ◇ | 32.6 ◇ | – | – | – |
| `omz-resnet50-binary-0001-fp16` | 71.9% → 73.4% | +0.26 pt | 34.4 ◇ | 17.5 ◇ | 24.8 ◇ | – | – | – |
| `omz-resnet50-binary-0001` | 72.7% → 73.4% | -0.04 pt | 34.4 ◇ | 17.4 ◇ | 24.9 ◇ | – | – | – |
| `bi-vit-deit-small` | 11.7% → 10.9% | -0.88 pt | 35.1 ◇ | 19.3 ◇ | 39.7 ◇ | 113.1 ⚠ | 182.3 | – |
| `larq-zoo-binary-densenet37-dilated` | 65.6% → 65.6% | -59.00 pt | 44.5 ◇ | 22.9 ◇ | 39.5 ◇ | – | – | – |
| `bhvit-small-binary` | 68.8% → 65.6% | – | 44.5 ◇ | 22.3 ◇ | 59.6 ◇ | 85.8 ⚠ | 228.6 | – |
| `bhvit-small-somefp` | 72.7% → 76.6% | -1.90 pt | 46.6 ◇ | 23.4 ◇ | 59.0 ◇ | 106.9 ⚠ | – | – |
| `bnext-small` | 76.6% → 75.0% | -2.70 pt | 48.0 ◇ | 24.1 ◇ | 52.1 ◇ | – | 270.1 | 787.1 |
| `bnext-middle` | 78.9% → 78.9% | -1.34 pt | 82.8 ◇ | 41.0 ◇ | 90.4 ◇ | – | – | – |
| `bnas-h-imagenet` | 60.2% → 60.2% | +0.08 pt | 108.1 ◇ | 53.8 ◇ | 69.5 ◇ | – | – | – |
| `bnext-large` | 79.7% → 80.5% | -0.88 pt | 188.5 ◇ | 92.8 ◇ | 201.5 ◇ | – | – | – |

### Other tasks (54 models)

| model | task | accuracy (reference → Bitino) | int8_full Δ | OnePlus Open 1 thread | OnePlus Open 4 threads | OnePlus Open int8 1 thread | Galaxy S22 1 thread | moto e13 1 thread | Redmi 9A 1 thread (32-bit) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `brevitas-bnn-pynq-tfc-1w1a` | mnist | 96.9% → 96.9% | – | 0.002 ◇ | 0.005 ◇ | 0.006 ◇ | – | – | – |
| `brevitas-bnn-pynq-sfc-1w1a` | mnist | 100.0% → 100.0% | – | 0.003 ◇ | 0.018 ◇ | 0.003 ◇ | – | – | – |
| `finn-examples-bnn-pynq-tfc-w1a1` | mnist | 96.9% → 96.9% | – | 0.003 ◇ | 0.012 ◇ | 0.003 ◇ | – | – | – |
| `bnn-pynq-lfc-w1a1-mnist` | mnist | 100.0% → 100.0% | – | 0.018 ◇ | 0.033 ◇ | 0.018 ◇ | – | – | – |
| `brevitas-bnn-pynq-lfc-1w1a` | mnist | 100.0% → 100.0% | – | 0.018 ◇ | 0.037 ◇ | 0.018 ◇ | – | – | – |
| `finn-examples-cnv-w1a1-gtsrb` | gtsrb | 93.8% → 93.8% | – | 0.293 ◇ | 0.271 ◇ | 0.253 ◇ | – | – | – |
| `bnn-pynq-cnv-w1a1-gtsrb` | gtsrb | 93.8% → 93.8% | – | 0.294 ◇ | 0.208 ◇ | 0.255 ◇ | – | – | – |
| `brillsson-quicknet` | non-semantic speech representation | 90.7% → 90.7% | – | 0.998 ◇ | 0.688 ◇ | 0.9 ◇ | – | – | – |
| `ab-bnn-cifar100-reactneta` | cifar100 | 71.9% → 71.9% | -0.31 pt | 1.59 ◇ | 0.947 ◇ | 1.99 ◇ | 2.63 ⚠ | – | – |
| `larqface-quicknet-msceleb` | face-recognition | 99.0% → 99.0% | – | 1.86 ◇ | 1.17 ◇ | 1.72 ◇ | – | – | – |
| `ab-bnn-cifar100-reactnet18` | cifar100 | 69.5% → 69.5% | -0.56 pt | 2.29 ◇ | 1.26 ◇ | 2.56 ◇ | 3.72 ⚠ | – | – |
| `brillsson-binarydensenet28` | non-semantic speech representation | 90.5% → 90.5% | – | 2.37 ◇ | 1.56 ◇ | 1.95 ◇ | – | – | – |
| `brillsson-meliusnet22` | non-semantic speech representation | 91.9% → 91.9% | – | 3.19 ◇ | 2.17 ◇ | 2.9 ◇ | – | – | – |
| `larqface-binarydensenet28-msceleb` | face-recognition | 99.2% → 99.2% | – | 5.27 ◇ | 3.12 ◇ | 3.99 ◇ | – | – | – |
| `larqface-binarydensenet37-msceleb` | face-recognition | 99.2% → 99.2% | – | 6.08 ◇ | 3.67 ◇ | 4.77 ◇ | – | – | – |
| `larqface-binarydensenet45-msceleb` | face-recognition | 99.2% → 99.2% | – | 8.43 ◇ | 4.99 ◇ | 6.53 ◇ | – | – | – |
| `bimatting-bimobilenet-backbone` | imagenet | 55.5% → 55.5% | -22.86 pt | 8.83 ◇ | 5.02 ◇ | 10.7 ◇ | – | – | – |
| `ida-det-bires18-imagenet-backbone` | imagenet | 65.6% → 65.6% | -0.64 pt | 11.1 ◇ | 5.65 ◇ | 8.89 ◇ | – | – | – |
| `bbcu-sr-m-x4` | single-image-super-resolution-x4 | 27.22 dB → 27.22 dB | – | 14.6 ◇ | 7.23 ◇ | 15.6 ◇ | 24.4 ⚠ | – | – |
| `larqface-binaryfacenet-msceleb` | face-recognition | 95.1% → 95.1% | – | 16.0 ◇ | 7.91 ◇ | 16.0 ◇ | – | – | – |
| `bbcu-sr-m-x2` | single-image-super-resolution-x2 | 31.89 dB → 31.89 dB | – | 16.4 ◇ | 8.85 ◇ | 16.9 ◇ | 25.6 ⚠ | – | – |
| `s2bnn-reacta-distill-only` | imagenet self-supervised representation learning | – → 0.96370 | – | 20.4 ◇ | 10.5 ◇ | 20.3 ◇ | – | – | – |
| `burn-reacta-stage2` | imagenet self-supervised representation learning | – → 0.99615 | – | 21.0 ◇ | 10.6 ◇ | 20.5 ◇ | – | – | – |
| `bsrc2023-binary-sr-x2-v0` | single-image-super-resolution-x2 | 31.88 dB → 31.88 dB | – | 26.1 ◇ | 13.8 ◇ | 16.0 ◇ | – | – | – |
| `ftl-deepfake-bnext-t-dffd-unfrozen` | deepfake-image-detection | – → – | – | 26.6 ◇ | 13.7 ◇ | 27.3 ◇ | – | – | – |
| `ftl-deepfake-bnext-t-cifake` | deepfake-image-detection | – → – | – | 26.7 ◇ | 13.7 ◇ | 27.1 ◇ | – | – | – |
| `ftl-deepfake-bnext-t-dffd` | deepfake-image-detection | – → – | – | 26.7 ◇ | 13.8 ◇ | 27.1 ◇ | – | – | – |
| `ftl-deepfake-bnext-t-cocofake-unfrozen` | deepfake-image-detection | – → – | – | 26.7 ◇ | 13.7 ◇ | 27.3 ◇ | – | 147.9 | – |
| `ftl-deepfake-bnext-t-cocofake` | deepfake-image-detection | – → – | – | 26.7 ◇ | 13.7 ◇ | 27.2 ◇ | – | – | – |
| `ftl-deepfake-bnext-t-cifake-unfrozen` | deepfake-image-detection | – → – | – | 26.8 ◇ | 13.7 ◇ | 27.1 ◇ | – | – | – |
| `bbcu-deblocking` | jpeg-artifact-reduction | 31.33 dB → 31.33 dB | – | 27.3 ◇ | 13.3 ◇ | 28.0 ◇ | 33.1 ⚠ | – | – |
| `bbcu-denoise` | color-image-denoising | 29.74 dB → 29.74 dB | – | 27.9 ◇ | 13.6 ◇ | 28.2 ◇ | 34.0 ⚠ | – | – |
| `bibert-bert-base-sst2` | glue-sst2 | 89.8% → 89.8% | -0.34 pt | 30.6 ◇ | 15.0 ◇ | 34.9 ◇ | – | 169.1 | 351.3 |
| `bbcu-sr-l-x2` | single-image-super-resolution-x2 | 32.03 dB → 32.03 dB | – | 35.5 ◇ | 17.9 ◇ | 19.4 ◇ | 36.6 ⚠ | – | – |
| `bsrc2023-binary-sr-x4-v0` | single-image-super-resolution-x4 | 27.22 dB → 27.22 dB | – | 38.9 ◇ | 17.6 ◇ | 25.8 ◇ | – | – | – |
| `bsrc2023-binary-sr-x4-v1` | single-image-super-resolution-x4 | 27.32 dB → 27.32 dB | – | 43.9 ◇ | 20.3 ◇ | 29.8 ◇ | – | – | – |
| `ftl-deepfake-bnext-s-dffd-unfrozen` | deepfake-image-detection | – → – | – | 49.2 ◇ | 24.7 ◇ | 49.6 ◇ | – | – | – |
| `ftl-deepfake-bnext-s-cifake` | deepfake-image-detection | – → – | – | 49.5 ◇ | 24.8 ◇ | 50.0 ◇ | – | – | – |
| `ftl-deepfake-bnext-s-cifake-unfrozen` | deepfake-image-detection | – → – | – | 49.5 ◇ | 24.7 ◇ | 49.9 ◇ | – | – | – |
| `ftl-deepfake-bnext-s-cocofake-unfrozen` | deepfake-image-detection | – → – | – | 49.6 ◇ | 24.8 ◇ | 50.0 ◇ | – | 281.5 | – |
| `ftl-deepfake-bnext-s-cocofake` | deepfake-image-detection | – → – | – | 49.6 ◇ | 24.8 ◇ | 49.8 ◇ | – | – | – |
| `ftl-deepfake-bnext-s-dffd` | deepfake-image-detection | – → – | – | 49.6 ◇ | 24.7 ◇ | 49.7 ◇ | – | – | – |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | 27.36 dB → 27.36 dB | – | 52.9 ◇ | 24.5 ◇ | 19.6 ◇ | 53.5 ⚠ | – | – |
| `bibert-bert-base-mrpc` | glue-mrpc | 76.6% → 76.6% | +0.98 pt | 57.5 ◇ | 26.3 ◇ | 69.0 ◇ | 148.9 ⚠ | – | – |
| `bibert-bert-base-mnli` | glue-mnli | 61.7% → 61.7% | +0.54 pt | 59.4 ◇ | 27.3 ◇ | 69.5 ◇ | 124.7 ⚠ | – | – |
| `bibert-bert-base-qqp` | glue-qqp | 83.6% → 83.6% | -0.59 pt | 65.1 ◇ | 30.2 ◇ | 75.0 ◇ | 160.7 ⚠ | – | – |
| `gsb-deit-s-cifar100-stage2` | cifar100 | 70.3% → 69.5% | -1.99 pt | 75.8 ◇ | 38.2 ◇ | 73.9 ◇ | – | – | – |
| `ftl-deepfake-bnext-m-cocofake` | deepfake-image-detection | – → – | – | 83.7 ◇ | 41.7 ◇ | 83.9 ◇ | – | – | – |
| `ftl-deepfake-bnext-m-dffd-unfrozen` | deepfake-image-detection | – → – | – | 83.9 ◇ | 41.8 ◇ | 83.8 ◇ | – | – | – |
| `ftl-deepfake-bnext-m-cocofake-unfrozen` | deepfake-image-detection | – → – | – | 84.0 ◇ | 41.8 ◇ | 83.8 ◇ | – | 470.1 | – |
| `ftl-deepfake-bnext-m-cifake` | deepfake-image-detection | – → – | – | 84.0 ◇ | 41.8 ◇ | 83.7 ◇ | – | – | – |
| `ftl-deepfake-bnext-m-dffd` | deepfake-image-detection | – → – | – | 84.1 ◇ | 41.8 ◇ | 84.1 ◇ | – | – | – |
| `ftl-deepfake-bnext-m-cifake-unfrozen` | deepfake-image-detection | – → – | – | 84.1 ◇ | 41.8 ◇ | 84.0 ◇ | – | – | – |
| `bsrc2023-binary-sr-x2-v1` | single-image-super-resolution-x2 | 32.22 dB → 32.22 dB | – | 91.0 ◇ | 44.4 ◇ | 61.6 ◇ | – | – | – |

Unmarked = clean (every repeat at full clocks). ◇ = the phone's own power policy held the clock ceiling below the maximum with no thermal event (stock settings, screen off; the ceiling is recorded per run); ⚠ = at least one repeat ran under a thermal CPU-frequency cap (e.g. a phone charging over USB); † = run made before the frequency-cap check existed (caps unknown). Runs per device and validity: Galaxy S22: 185 thermal-throttled, 24 unknown-caps; OnePlus Open: 1405 vendor-policy; moto e13: 116 clean; Redmi 9A: 42 clean. 

Every phone with 1 and 4 threads and int8: [docs/SPEED.md](docs/SPEED.md). On-device correctness: Galaxy S22: 381/381 correctness checks passed on 114 models, 2026-11-02 to 2026-11-12 (outputs byte-identical to a validated emulated-ARM or reference-phone run of the same code); OnePlus Open: 460/460 correctness checks passed on 132 models, 2026-11-12 (outputs byte-identical to a validated emulated-ARM or reference-phone run of the same code); moto e13: 503/503 correctness checks passed on 132 models, 2026-11-12 (outputs byte-identical to a validated emulated-ARM or reference-phone run of the same code); Redmi 9A: 228/228 correctness checks passed on 114 models, 2026-11-12 to 2026-11-13 (exact stage bit-identical to the numpy executor, float stage within tolerance of the float64 reference).
<!-- /generated:speed -->

The phones: `results/devices.json`. All runs with their clock state: `results/speed_<device>.json`; x86-64 host
runs (older runtime, labelled): `results/speed_x86.json`. Method: [docs/benchmarking.md](docs/benchmarking.md).

**Against Larq Compute Engine and TensorFlow Lite** (same Larq Zoo weights, 1 thread, batch 1, every runtime pinned
to the same core by one launcher): Bitino is 1.29× faster than LCE 0.16 on a Cortex-X3 (all 11 models, 1.09–1.71×),
1.11× faster on a 32-bit Cortex-A53, and on par on a Cortex-A75, where LCE is 3 % faster on average. TFLite fp32 is
4.7–13× slower than Bitino; TFLite int8 is 1.4–4.8× slower and agrees with the float model on only 45–66 % of top-1
classes. Tables, method, setup differences and raw data: [docs/comparison-lce-tflite.md](docs/comparison-lce-tflite.md).

## Model zoo

Selected models (full table with all converted models: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md); searchable version on
the website):

<!-- generated:zoo_top -->
| model | task | framework | accuracy (reference → Bitino) | eval set (n) | int8_full Δ vs exact | fastest phone ms (1 thread) | weights licence | release |
|---|---|---|---|---|---|---|---|---|
| `ab-bnn-cifar10-reactnet18` | cifar10-classification | pytorch | 93.8% → 93.8% | cifar10-test-first-128 (128) | +0.06 pt (cifar10-test-first-10000, n=10000) | 2.29 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactnet18` | imagenet-classification | pytorch | 65.6% → 64.8% | imagenet-val-subset-128 (128) | -0.34 pt (imagenet-val-subset-5000, n=5000) | 11.1 ◇ (OnePlus Open) | MIT (HF model card metadata 'license: mit'); app | yes |
| `ab-bnn-imagenet-reactnet34` | imagenet-classification | pytorch | 67.2% → 68.0% | imagenet-val-subset-128 (128) | -0.62 pt (imagenet-val-subset-5000, n=5000) | 17.6 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactneta` | imagenet-classification | pytorch | 66.4% → 65.6% | imagenet-val-subset-128 (128) | -0.58 pt (imagenet-val-subset-5000, n=5000) | 19.6 ⚠ (Galaxy S22) | MIT (HF card) | yes |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | pytorch | 27.36 dB → 27.36 dB | set5-set14-b100-x4-lr32 (64) | – | 52.9 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bhvit-tiny-binary` | imagenet-classification | pytorch | 64.1% → 62.5% | imagenet-val-subset-128 (128) | -0.74 pt (imagenet-val-subset-5000_bhvit, n=5000) | 29.0 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `bi-vit-deit-tiny` | imagenet-classification | pytorch | 15.6% → 15.6% | imagenet-val-subset-128 (128) | -0.34 pt (imagenet-val-subset-5000_bhvit, n=5000) | 13.6 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bibert-bert-base-sst2` | glue-sst2 | pytorch | 89.8% → 89.8% | sst2-dev-128 (128) | -0.34 pt (sst2-dev-872, n=872) | 30.6 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bnext-tiny` | imagenet-classification | pytorch | 69.5% → 68.0% | imagenet-val-subset-128 (128) | -3.84 pt (imagenet-val-subset-5000, n=5000) | 25.0 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `brevitas-bnn-pynq-cnv-1w1a` | cifar10-classification | pytorch+brevitas | 85.9% → 85.9% | cifar10-test-first-128 (128) | -0.77 pt (cifar10-test-first-10000, n=10000) | 0.293 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `larq-zoo-birealnet` | imagenet-classification | larq_zoo | 57.8% → – | imagenet-val-subset-128 (128) | -0.20 pt (imagenet-val-subset-5000, n=5000) | 10.6 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet` | imagenet-classification | larq_zoo | 66.4% → 66.4% | imagenet-val-subset-128 (128) | -0.72 pt (imagenet-val-subset-5000, n=5000) | 7.65 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet-large` | imagenet-classification | larq_zoo | 71.1% → 70.3% | imagenet-val-subset-128 (128) | -1.96 pt (imagenet-val-subset-5000, n=5000) | 13.6 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet-small` | imagenet-classification | larq_zoo | 59.4% → 58.6% | imagenet-val-subset-128 (128) | -1.96 pt (imagenet-val-subset-5000, n=5000) | 4.91 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `omz-resnet18-xnor-binary-onnx-0001` | imagenet-classification | openvino-ir | 59.4% → 59.4% | imagenet-val-subset-128 (128) | -1.20 pt (imagenet-val-subset-5000, n=5000) | 10.9 ◇ (OnePlus Open) | Apache-2.0 | yes |
| `omz-resnet50-binary-0001` | imagenet-classification | openvino-ir | 72.7% → 73.4% | imagenet-val-subset-128 (128) | -0.04 pt (imagenet-val-subset-5000, n=5000) | 34.4 ◇ (OnePlus Open) | Apache-2.0 (model.yml license field points to th | yes |
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
* **Targets:** Android arm64 and armeabi-v7a (command-line runtime: API 30+; AAR: API 24+), Linux aarch64,
  Linux x86-64; Python wheels for Linux x86-64 / aarch64, macOS arm64 and Windows x64 (correctness-tested in CI;
  speed is measured on the phones and the x86 host only).

## Add a model

1. Add the network to `registry/models.json` (checkpoint URL, SHA-256, licences, provenance).
2. Convert it with the closest converter in `tools/convert_*.py` (or write an adapter in `python/bitino/adapters/`).
3. Validate: `tools/validate_model.py` on the host, then on a device.
4. `tools/build_recipes.py --artifacts <dir>` writes its fetch recipe; `tools/build_site.py` refreshes the tables.

Full walk-through: [docs/adding-a-model.md](docs/adding-a-model.md).

## Honest limitations

* **Speed data has caveats.** The OnePlus Open numbers are on the current runtime with the phone's stock power
  policy, which holds the clocks below their maximum with the screen off (◇, ceilings recorded per run). The S22
  numbers are from an older runtime on a thermally capped phone (charging over USB) and are labelled so. x86 host
  numbers are from an older runtime.
* **Fetch recipes:** 131 of 132 were run end to end from an empty cache (authors' downloads, SHA-256 checked,
  converted file identical to the published one); one is broken upstream (S2-BNN: the authors' OneDrive link has
  expired). State per model: `bitino list --converted`. Google Drive can refuse a scripted download (daily quota);
  `fetch` then prints the link to download by hand. Some recipes need a second reference environment
  (BHViT's transformers 4.39, OpenCV + scikit-image for the Faster Than Lies detectors, tensorpack/TensorFlow,
  Python 2.7 Theano for BNN-PYNQ), listed in `envs/`.
* **int8_full** loses accuracy on some networks (see the zoo tables); the exact `neon` / `optimized` stages
  match the source framework. Reproducing an int8 file bit for bit needs the calibration images it was built from
  (training split, not redistributed).
* **Accuracy subsets are small** (mostly 128 images): they validate conversions, they do not re-measure the papers.
* **Few phones.** Tuned on a Snapdragon 8 Gen 1 and measured on the phones in `results/devices.json`. Other arm64
  CPUs run the same code; their speed is not measured here. 32-bit Android (armeabi-v7a) runs only a portable
  fallback build without NEON kernels. No GPU / NPU backends.
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
