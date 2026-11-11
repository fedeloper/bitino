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
* **Honest speed numbers.** Timings come only from real hardware (the Android phones in `results/devices.json`,
  an x86-64 host), with date, thread setting and clock state. Emulators are used for correctness and instruction
  counts, never for timings.

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

## Speed (measured on phones)

<!-- generated:speed -->
| model | task | accuracy (reference → Bitino) | Galaxy S22 ms | OnePlus Open ms |
|---|---|---|---|---|
| `ab-bnn-cifar10-reactnet18` | cifar10-classification | 93.8% → 93.8% (cifar10-test-first-128, n=128) | 3.72 ⚠ / 5.25 ⚠<br>int8 2.92 ⚠ / 2.02 ⚠ | 2.32 ◇ / 1.28 ◇<br>int8 2.15 ◇ / – |
| `ab-bnn-cifar10-reactneta` | cifar10-classification | 91.4% → 91.4% (cifar10-test-first-128, n=128) | 2.6 ⚠ / 2.48 ⚠<br>int8 2.57 ⚠ / – | 1.61 ◇ / 0.992 ◇<br>int8 1.59 ◇ / – |
| `ab-bnn-cifar100-reactnet18` | cifar100-classification | 69.5% → 69.5% (cifar100-test-first-128, n=128) | 3.72 ⚠ / 2.56 ⚠<br>int8 3.5 ⚠ / – | 2.32 ◇ / 1.31 ◇<br>int8 2.13 ◇ / – |
| `ab-bnn-cifar100-reactneta` | cifar100-classification | 71.9% → 71.9% (cifar100-test-first-128, n=128) | 2.63 ⚠ / 2.47 ⚠<br>int8 2.58 ⚠ / – | 1.62 ◇ / 0.997 ◇<br>int8 1.6 ◇ / – |
| `ab-bnn-imagenet-reactnet18` | imagenet-classification | 65.6% → 64.8% (imagenet-val-subset-128, n=128) | 16.9 ⚠ / 8.93 ⚠<br>int8 10.9 ⚠ / 5.44 ⚠ | 11.1 ◇ / 5.86 ◇<br>int8 7.81 ◇ / – |
| `ab-bnn-imagenet-reactnet34` | imagenet-classification | 67.2% → 68.0% (imagenet-val-subset-128, n=128) | 27.0 ⚠ / 14.6 ⚠<br>int8 22.9 ⚠ / – | 17.7 ◇ / 9.36 ◇<br>int8 14.3 ◇ / – |
| `ab-bnn-imagenet-reactneta` | imagenet-classification | 66.4% → 65.6% (imagenet-val-subset-128, n=128) | 19.6 ⚠ / 12.8 ⚠<br>int8 19.4 ⚠ / – | 20.2 ◇ / 10.5 ◇<br>int8 19.8 ◇ / – |
| `adabin-resnet20-cifar10` | cifar10-classification | 91.4% → 91.4% (cifar10-test-first-128, n=128) | 1.22 ⚠ / 9.37 ⚠<br>int8 1.25 ⚠ / – | 1.58 ◇ / 1.27 ◇<br>int8 1.71 ◇ / – |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | cifar10-classification | 89.8% → 89.8% (cifar10-test-first-128, n=128) | 0.57 ⚠ / 1.39 ⚠<br>int8 0.511 ⚠ / – | 0.731 ◇ / 0.216 ◇<br>int8 0.695 ◇ / – |
| `bbcu-deblocking` | jpeg-artifact-reduction (grayscale, Y channel) | 31.33 dB → 31.33 dB (live1-classic5-jpeg-y-q10q40-center96, n=68) | 33.1 ⚠ / 16.3 ⚠<br>int8 32.9 ⚠ / – | 28.9 ◇ / 13.8 ◇<br>int8 28.2 ◇ / – |
| `bbcu-denoise` | color-image-denoising (AWGN blind) | 29.74 dB → 29.74 dB (cbsd68-sigma25-center96, n=68) | 34.0 ⚠ / 16.7 ⚠<br>int8 33.6 ⚠ / – | 29.2 ◇ / 14.0 ◇<br>int8 28.2 ◇ / – |
| `bbcu-sr-l-x2` | single-image-super-resolution-x2 | 32.03 dB → 32.03 dB (set5-set14-b100-x2-lr48, n=64) | 36.6 ⚠ / 19.5 ⚠<br>int8 20.6 ⚠ / – | 37.8 ◇ / 19.3 ◇<br>int8 20.4 ◇ / – |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | 27.36 dB → 27.36 dB (set5-set14-b100-x4-lr32, n=64) | 53.5 ⚠ / 27.6 ⚠<br>int8 22.6 ⚠ / – | 56.9 ◇ / 25.7 ◇<br>int8 21.6 ◇ / – |
| `bbcu-sr-m-x2` | single-image-super-resolution-x2 | 31.89 dB → 31.89 dB (set5-set14-b100-x2-lr48, n=64) | 25.6 ⚠ / 13.3 ⚠<br>int8 25.2 ⚠ / – | 18.7 ◇ / 9.61 ◇<br>int8 18.0 ◇ / – |
| `bbcu-sr-m-x4` | single-image-super-resolution-x4 | 27.22 dB → 27.22 dB (set5-set14-b100-x4-lr32, n=64) | 24.4 ⚠ / 11.9 ⚠<br>int8 23.0 ⚠ / – | 18.5 ◇ / 8.48 ◇<br>int8 17.7 ◇ / – |
| `bhvit-small-binary` | imagenet-classification | 68.8% → 65.6% (imagenet-val-subset-128, n=128) | 85.8 ⚠ / 59.3 ⚠<br>int8 85.2 ⚠ / – | 50.0 ◇ / 24.7 ◇<br>int8 49.6 ◇ / – |
| `bhvit-small-somefp` | imagenet-classification | 72.7% → 76.6% (imagenet-val-subset-128, n=128) | 106.9 ⚠ / 70.6 ⚠<br>int8 104.5 ⚠ / – | 52.3 ◇ / 26.0 ◇<br>int8 49.8 ◇ / – |
| `bhvit-tiny-binary` | imagenet-classification | 64.1% → 62.5% (imagenet-val-subset-128, n=128) | 46.6 ⚠ / 39.3 ⚠<br>int8 67.5 ⚠ / – | 32.8 ◇ / 16.3 ◇<br>int8 32.1 ◇ / – |
| `bhvit-tiny-somefp` | imagenet-classification | 65.6% → 65.6% (imagenet-val-subset-128, n=128) | 58.8 ⚠ / 37.4 ⚠<br>int8 67.5 ⚠ / – | 35.1 ◇ / 17.5 ◇<br>int8 32.4 ◇ / – |
| `bi-vit-deit-small` | imagenet-classification | 11.7% → 10.9% (imagenet-val-subset-128, n=128) | 113.1 ⚠ / 82.6 ⚠<br>int8 68.6 ⚠ / 56.5 ⚠ | 41.9 ◇ / 23.8 ◇<br>int8 40.1 ◇ / – |
| `bi-vit-deit-tiny` | imagenet-classification | 15.6% → 15.6% (imagenet-val-subset-128, n=128) | 42.1 ⚠ / 31.4 ⚠<br>int8 21.9 ⚠ / 13.0 ⚠ | 16.8 ◇ / 10.1 ◇<br>int8 16.0 ◇ / – |
| `bibert-bert-base-mnli` | glue-mnli | 61.7% → 61.7% (mnli-dev-matched-128, n=128) | 124.7 ⚠ / 54.9 ⚠<br>int8 124.8 ⚠ / – | 62.6 ◇ / 28.7 ◇<br>int8 62.7 ◇ / – |
| `bibert-bert-base-mrpc` | glue-mrpc | 76.6% → 76.6% (mrpc-dev-128, n=128) | 148.9 ⚠ / 68.3 ⚠<br>int8 148.6 ⚠ / – | 60.6 ◇ / 27.6 ◇<br>int8 60.5 ◇ / – |
| `bibert-bert-base-qqp` | glue-qqp | 83.6% → 83.6% (qqp-dev-128, n=128) | 160.7 ⚠ / 73.6 ⚠<br>int8 160.6 ⚠ / – | 68.2 ◇ / 31.5 ◇<br>int8 68.3 ◇ / – |
| `bibert-bert-base-sst2` | glue-sst2 | 89.8% → 89.8% (sst2-dev-128, n=128) | – | 32.1 ◇ / 15.8 ◇<br>int8 32.2 ◇ / – |
| `bimatting-bimobilenet-backbone` | imagenet-classification (backbone) | 55.5% → 55.5% (imagenet-val-subset-128, n=128) | – | 9.91 ◇ / 5.54 ◇<br>int8 9.62 ◇ / – |
| `biper-resnet18-cifar10` | cifar10-classification | 94.5% → 94.5% (cifar10-test-first-128, n=128) | – | 2.01 ◇ / 1.14 ◇<br>int8 1.99 ◇ / – |
| `biper-resnet18-imagenet` | imagenet-classification | 59.4% → 59.4% (imagenet-val-subset-128, n=128) | – | 11.2 ◇ / 6.01 ◇<br>int8 7.9 ◇ / – |
| `biper-resnet20-cifar10` | cifar10-classification | 86.7% → 86.7% (cifar10-test-first-128, n=128) | – | 1.22 ◇ / 0.631 ◇<br>int8 1.15 ◇ / – |
| `biper-resnet34-imagenet` | imagenet-classification | 66.4% → 66.4% (imagenet-val-subset-128, n=128) | – | 17.5 ◇ / 9.37 ◇<br>int8 14.2 ◇ / – |
| `biper-vgg-small-cifar10` | cifar10-classification | 96.1% → 96.1% (cifar10-test-first-128, n=128) | – | 2.08 ◇ / 1.08 ◇<br>int8 1.99 ◇ / – |
| `bitorch-densenet28-imagenet` | imagenet-classification | 59.4% → 59.4% (imagenet-val-subset-128, n=128) | – | 22.8 ◇ / 12.1 ◇<br>int8 17.8 ◇ / – |
| `bitorch-meliusnet22-imagenet` | imagenet-classification | 66.4% → 65.6% (imagenet-val-subset-128, n=128) | – | 26.9 ◇ / 14.3 ◇<br>int8 22.3 ◇ / – |
| `bitorch-resnete18-imagenet` | imagenet-classification | 64.8% → 62.5% (imagenet-val-subset-128, n=128) | – | 10.8 ◇ / 5.8 ◇<br>int8 7.54 ◇ / – |
| `bnas-a-cifar10` | cifar10-classification | 93.8% → 93.8% (cifar10-test-first-128, n=128) | – | 8.74 ◇ / 5.28 ◇<br>int8 5.67 ◇ / – |
| `bnas-d-imagenet` | imagenet-classification | 60.2% → 60.2% (imagenet-val-subset-128, n=128) | – | 26.4 ◇ / 15.4 ◇<br>int8 15.9 ◇ / – |
| `bnas-e-imagenet` | imagenet-classification | 56.2% → 56.2% (imagenet-val-subset-128, n=128) | – | 32.7 ◇ / 19.1 ◇<br>int8 21.3 ◇ / – |
| `bnas-h-imagenet` | imagenet-classification | 60.2% → 60.2% (imagenet-val-subset-128, n=128) | – | 106.6 ◇ / 56.2 ◇<br>int8 65.6 ◇ / – |
| `bnext-large` | imagenet-classification | 79.7% → 80.5% (imagenet-val-subset-128, n=128) | – | 191.3 ◇ / 94.8 ◇<br>int8 188.1 ◇ / – |
| `bnext-middle` | imagenet-classification | 78.9% → 78.9% (imagenet-val-subset-128, n=128) | – | 86.4 ◇ / 43.3 ◇<br>int8 85.5 ◇ / – |
| `bnext-small` | imagenet-classification | 76.6% → 75.0% (imagenet-val-subset-128, n=128) | – | 51.9 ◇ / 25.8 ◇<br>int8 51.0 ◇ / – |
| `bnext-tiny` | imagenet-classification | 69.5% → 68.0% (imagenet-val-subset-128, n=128) | int8 39.2 ⚠ / 20.4 ⚠ | 25.6 ◇ / 13.3 ◇<br>int8 25.1 ◇ / – |
| `bnn-pynq-cnv-w1a1-cifar10` | cifar10-classification | 86.7% → 86.7% (cifar10-test-first-128, n=128) | – | 0.712 ◇ / 0.235 ◇<br>int8 0.698 ◇ / – |
| `bnn-pynq-cnv-w1a1-gtsrb` | gtsrb-classification (43 classes + junk = 44) | 93.8% → 93.8% (gtsrb-test-udacity32-first-128, n=128) | – | 0.731 ◇ / 0.241 ◇<br>int8 0.698 ◇ / – |
| `bnn-pynq-lfc-w1a1-mnist` | mnist-classification | 100.0% → 100.0% (mnist-test-first-128, n=128) | – | 0.055 ◇ / 0.053 ◇<br>int8 0.054 ◇ / – |
| `brevitas-bnn-pynq-cnv-1w1a` | cifar10-classification | 85.9% → 85.9% (cifar10-test-first-128, n=128) | – | 0.701 ◇ / 0.253 ◇<br>int8 0.696 ◇ / – |
| `brevitas-bnn-pynq-lfc-1w1a` | mnist-classification | 100.0% → 100.0% (mnist-test-first-128, n=128) | – | 0.054 ◇ / 0.053 ◇<br>int8 0.054 ◇ / – |
| `brevitas-bnn-pynq-sfc-1w1a` | mnist-classification | 100.0% → 100.0% (mnist-test-first-128, n=128) | – | 0.03 ◇ / 0.025 ◇<br>int8 0.033 ◇ / – |
| `brevitas-bnn-pynq-tfc-1w1a` | mnist-classification | 96.9% → 96.9% (mnist-test-first-128, n=128) | – | 0.017 ◇ / 0.013 ◇<br>int8 0.019 ◇ / – |
| `brillsson-binarydensenet28` | non-semantic speech representation (embedding; linear-probe on SpeechCommands / emotion / language-ID etc.) | 90.5% → 90.5% (Speech Commands v0.02 test, n=128) | – | 2.52 ◇ / 1.65 ◇<br>int8 2.13 ◇ / – |
| `brillsson-meliusnet22` | non-semantic speech representation (embedding; linear-probe on SpeechCommands / emotion / language-ID etc.) | 91.9% → 91.9% (Speech Commands v0.02 test, n=128) | – | 3.4 ◇ / 2.32 ◇<br>int8 3.1 ◇ / – |
| `brillsson-quicknet` | non-semantic speech representation (embedding; linear-probe on SpeechCommands / emotion / language-ID etc.) | 90.7% → 90.7% (Speech Commands v0.02 test, n=128) | – | 1.02 ◇ / 0.72 ◇<br>int8 0.926 ◇ / – |
| `bsrc2023-binary-sr-x2-v0` | single-image-super-resolution-x2 | 31.88 dB → 31.88 dB (set5-set14-b100-x2-lr48, n=64) | – | 28.2 ◇ / 14.9 ◇<br>int8 16.8 ◇ / – |
| `bsrc2023-binary-sr-x2-v1` | single-image-super-resolution-x2 | 32.22 dB → 32.22 dB (set5-set14-b100-x2-lr48, n=64) | – | 139.6 ◇ / 64.6 ◇<br>int8 108.3 ◇ / – |
| `bsrc2023-binary-sr-x4-v0` | single-image-super-resolution-x4 | 27.22 dB → 27.22 dB (set5-set14-b100-x4-lr32, n=64) | – | 62.2 ◇ / 28.1 ◇<br>int8 28.0 ◇ / – |
| `bsrc2023-binary-sr-x4-v1` | single-image-super-resolution-x4 | 27.32 dB → 27.32 dB (set5-set14-b100-x4-lr32, n=64) | – | 76.0 ◇ / 34.0 ◇<br>int8 40.1 ◇ / – |
| `burn-reacta-stage2` | imagenet self-supervised representation learning (backbone; the ImageNet linear-eval head is NOT included) | – → 0.99615 (imagenet-val-subset-128 / 5000, n=128) | – | 20.8 ◇ / 10.8 ◇<br>int8 20.5 ◇ / – |
| `dabnn-bireal18` | imagenet-classification | 58.6% → 58.6% (imagenet-val-subset-128, n=128) | – | 10.8 ◇ / 5.79 ◇<br>int8 7.46 ◇ / – |
| `dabnn-bireal18-stem` | imagenet-classification | 58.6% → 58.6% (imagenet-val-subset-128, n=128) | – | 8.36 ◇ / 4.5 ◇<br>int8 7.45 ◇ / – |
| `fda-bnn-resnet20-cifar10-mindspore` | cifar10-classification | 89.8% → 89.8% (cifar10-test-first-128, n=128) | – | 1.23 ◇ / 0.501 ◇<br>int8 1.21 ◇ / – |
| `finn-examples-bnn-pynq-cnv-w1a1` | cifar10-classification / mnist-classification | 85.9% → 85.9% (cifar10-test-first-128, n=128) | – | 0.73 ◇ / 0.237 ◇<br>int8 0.695 ◇ / – |
| `finn-examples-bnn-pynq-tfc-w1a1` | cifar10-classification / mnist-classification | 96.9% → 96.9% (mnist-test-first-128, n=128) | – | 0.016 ◇ / 0.012 ◇<br>int8 0.019 ◇ / – |
| `finn-examples-cnv-w1a1-gtsrb` | gtsrb-classification (traffic signs; 44 outputs) | 93.8% → 93.8% (gtsrb-test-udacity32-first-128, n=128) | – | 0.701 ◇ / 0.241 ◇<br>int8 0.696 ◇ / – |
| `gsb-deit-s-cifar100-stage2` | cifar100-classification | 70.3% → 69.5% (cifar100-test-first-128-deit224, n=128) | – | 106.0 ◇ / 51.4 ◇<br>int8 104.0 ◇ / – |
| `ida-det-bires18-imagenet-backbone` | imagenet-classification (backbone for detection) | 65.6% → 65.6% (imagenet-val-subset-128, n=128) | – | 11.1 ◇ / 5.87 ◇<br>int8 7.81 ◇ / – |
| `larq-zoo-binary-alexnet` | imagenet-classification | 28.9% → 28.9% (imagenet-val-subset-128, n=128) | – | 5.48 ◇ / 2.73 ◇<br>int8 3.69 ◇ / – |
| `larq-zoo-binary-densenet28` | imagenet-classification | 60.9% → 60.9% (imagenet-val-subset-128, n=128) | – | 22.8 ◇ / 12.1 ◇<br>int8 17.8 ◇ / – |
| `larq-zoo-binary-densenet37` | imagenet-classification | 66.4% → 65.6% (imagenet-val-subset-128, n=128) | – | 26.4 ◇ / 14.2 ◇<br>int8 21.4 ◇ / – |
| `larq-zoo-binary-densenet37-dilated` | imagenet-classification | 65.6% → 65.6% (imagenet-val-subset-128, n=128) | – | 47.6 ◇ / 24.7 ◇<br>int8 36.5 ◇ / – |
| `larq-zoo-binary-densenet45` | imagenet-classification | 60.9% → 61.7% (imagenet-val-subset-128, n=128) | – | 35.8 ◇ / 19.0 ◇<br>int8 29.4 ◇ / – |
| `larq-zoo-binary-resnet-e18` | imagenet-classification | 55.5% → 55.5% (imagenet-val-subset-128, n=128) | – | 10.8 ◇ / 5.84 ◇<br>int8 7.57 ◇ / – |
| `larq-zoo-birealnet` | imagenet-classification | 57.8% → – (imagenet-val-subset-128, n=128) | – | 10.7 ◇ / 5.78 ◇<br>int8 7.57 ◇ / – |
| `larq-zoo-dorefanet` | imagenet-classification | 46.1% → 46.1% (imagenet-val-subset-128, n=128) | – | 23.1 ◇ / 12.1 ◇<br>int8 19.3 ◇ / – |
| `larq-zoo-meliusnet22` | imagenet-classification | 65.6% → 66.4% (imagenet-val-subset-128, n=128) | – | 26.6 ◇ / 14.0 ◇<br>int8 24.6 ◇ / – |
| `larq-zoo-quicknet` | imagenet-classification | 66.4% → 66.4% (imagenet-val-subset-128, n=128) | – | 7.83 ◇ / 4.16 ◇<br>int8 7.28 ◇ / – |
| `larq-zoo-quicknet-large` | imagenet-classification | 71.1% → 70.3% (imagenet-val-subset-128, n=128) | – | 14.0 ◇ / 7.63 ◇<br>int8 13.4 ◇ / – |
| `larq-zoo-quicknet-small` | imagenet-classification | 59.4% → 58.6% (imagenet-val-subset-128, n=128) | 9.26 † / 6.28 †<br>int8 8.14 ⚠ / 4.17 ⚠ | 5.05 ◇ / 2.7 ◇<br>int8 4.77 ◇ / – |
| `larq-zoo-quicknet-xl-legacy` | imagenet-classification | 64.1% → 64.8% (imagenet-val-subset-128, n=128) | – | 13.7 ◇ / 7.7 ◇<br>int8 13.1 ◇ / – |
| `larq-zoo-real-to-binary-net` | imagenet-classification | 64.8% → 64.8% (imagenet-val-subset-128, n=128) | – | 12.7 ◇ / 6.72 ◇<br>int8 9.03 ◇ / – |
| `larq-zoo-xnornet` | imagenet-classification | 34.4% → 34.4% (imagenet-val-subset-128, n=128) | – | 7.85 ◇ / 4.29 ◇<br>int8 4.7 ◇ / – |
| `larqface-binarydensenet28-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | 99.2% → 99.2% (LFW + AgeDB-30 insightface .bin, n=6000) | – | 5.58 ◇ / 3.38 ◇<br>int8 4.29 ◇ / – |
| `larqface-binarydensenet37-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | 99.2% → 99.2% (LFW + AgeDB-30 insightface .bin, n=6000) | – | 6.5 ◇ / 4.04 ◇<br>int8 5.19 ◇ / – |
| `larqface-binarydensenet45-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | 99.2% → 99.2% (LFW + AgeDB-30 insightface .bin, n=6000) | – | 9.11 ◇ / 5.5 ◇<br>int8 7.13 ◇ / – |
| `larqface-binaryfacenet-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | 95.1% → 95.1% (LFW + AgeDB-30 insightface .bin, n=6000) | – | 16.2 ◇ / 8.63 ◇<br>int8 16.1 ◇ / – |
| `larqface-quicknet-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | 99.0% → 99.0% (LFW + AgeDB-30 insightface .bin, n=6000) | – | 1.92 ◇ / 1.25 ◇<br>int8 1.75 ◇ / – |
| `nncf-resnet18-binarization-dorefa` | imagenet-classification | 62.5% → 62.5% (imagenet-val-subset-128, n=128) | – | 10.9 ◇ / 5.8 ◇<br>int8 7.72 ◇ / – |
| `nncf-resnet18-binarization-xnor` | imagenet-classification | 61.7% → 61.7% (imagenet-val-subset-128, n=128) | – | 10.9 ◇ / 5.8 ◇<br>int8 7.73 ◇ / – |
| `omz-resnet18-xnor-binary-onnx-0001` | imagenet-classification | 59.4% → 59.4% (imagenet-val-subset-128, n=128) | – | 11.0 ◇ / 5.84 ◇<br>int8 7.8 ◇ / – |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | imagenet-classification | 64.1% → 64.1% (imagenet-val-subset-128, n=128) | – | 11.1 ◇ / 5.85 ◇<br>int8 7.8 ◇ / – |
| `omz-resnet50-binary-0001` | imagenet-classification | 72.7% → 73.4% (imagenet-val-subset-128, n=128) | – | 34.4 ◇ / 18.0 ◇<br>int8 21.6 ◇ / – |
| `omz-resnet50-binary-0001-fp16` | imagenet-classification | 71.9% → 73.4% (imagenet-val-subset-128, n=128) | – | 34.4 ◇ / 18.0 ◇<br>int8 21.5 ◇ / – |
| `rbnn-resnet18-cifar10` | cifar10-classification | 94.5% → 94.5% (cifar10-test-first-128, n=128) | – | 2.77 ◇ / 1.86 ◇<br>int8 2.71 ◇ / – |
| `rbnn-resnet18-imagenet` | imagenet-classification | 53.1% → 53.1% (imagenet-val-subset-128, n=128) | – | 13.1 ◇ / 7.86 ◇<br>int8 9.8 ◇ / – |
| `rbnn-resnet20-bireal-cifar10` | cifar10-classification | 89.8% → 89.8% (cifar10-test-first-128, n=128) | – | 0.858 ◇ / 0.715 ◇<br>int8 0.867 ◇ / – |
| `rbnn-resnet20-cifar10` | cifar10-classification | 88.3% → 88.3% (cifar10-test-first-128, n=128) | – | 0.86 ◇ / 0.635 ◇<br>int8 0.878 ◇ / – |
| `rbnn-resnet34-imagenet` | imagenet-classification | 64.1% → 63.3% (imagenet-val-subset-128, n=128) | – | 20.9 ◇ / 12.6 ◇<br>int8 17.5 ◇ / – |
| `rbnn-vgg-small-cifar10` | cifar10-classification | 91.4% → 91.4% (cifar10-test-first-128, n=128) | – | 2.39 ◇ / 1.37 ◇<br>int8 2.29 ◇ / – |
| `rbonn-reactnet-resnet18-imagenet` | imagenet-classification | 65.6% → 65.6% (imagenet-val-subset-128, n=128) | – | 11.1 ◇ / 5.87 ◇<br>int8 7.81 ◇ / – |
| `recu-resnet18-cifar10` | cifar10-classification | 93.8% → 93.8% (cifar10-test-first-128, n=128) | – | 2.05 ◇ / 1.17 ◇<br>int8 2 ◇ / – |
| `recu-resnet18-imagenet` | imagenet-classification | 59.4% → 59.4% (imagenet-val-subset-128, n=128) | – | 11.2 ◇ / 6.01 ◇<br>int8 7.94 ◇ / – |
| `recu-resnet20-cifar10` | cifar10-classification | 88.3% → 88.3% (cifar10-test-first-128, n=128) | – | 1.2 ◇ / 0.646 ◇<br>int8 1.18 ◇ / – |
| `recu-resnet34-imagenet` | imagenet-classification | 68.8% → 68.8% (imagenet-val-subset-128, n=128) | – | 17.6 ◇ / 9.37 ◇<br>int8 14.2 ◇ / – |
| `recu-vgg-small-cifar10` | cifar10-classification | 96.1% → 96.1% (cifar10-test-first-128, n=128) | – | 2.09 ◇ / 1.06 ◇<br>int8 2.01 ◇ / – |
| `s2bnn-reacta-distill-only` | imagenet self-supervised representation learning (backbone + MoCo projection; no ImageNet classifier) | – → 0.96370 (imagenet-val-subset-128 / 5000, n=128) | – | 20.6 ◇ / 10.6 ◇<br>int8 20.5 ◇ / – |
| `sa-bnn-resnet18-imagenet` | imagenet-classification | 61.7% → 62.5% (imagenet-val-subset-128, n=128) | – | 11.9 ◇ / 6.12 ◇<br>int8 8.43 ◇ / – |
| `shuaiz-bireal20-cifar10` | cifar10-classification | 84.4% → 84.4% (cifar10-test-first-128, n=128) | – | 1.11 ◇ / 0.59 ◇<br>int8 1.09 ◇ / – |
| `siman-resnet18-cifar10` | cifar10-classification | 95.3% → 95.3% (cifar10-test-first-128, n=128) | – | 2.75 ◇ / 1.86 ◇<br>int8 2.72 ◇ / – |
| `siman-resnet18-imagenet` | imagenet-classification | 54.7% → 54.7% (imagenet-val-subset-128, n=128) | – | 12.8 ◇ / 7.68 ◇<br>int8 9.49 ◇ / – |
| `siman-resnet20-cifar10` | cifar10-classification | 87.5% → 87.5% (cifar10-test-first-128, n=128) | – | 0.868 ◇ / 0.769 ◇<br>int8 0.877 ◇ / – |
| `siman-resnet34-imagenet` | imagenet-classification | 59.4% → 60.9% (imagenet-val-subset-128, n=128) | – | 20.5 ◇ / 12.4 ◇<br>int8 17.2 ◇ / – |
| `siman-vgg-small-cifar10` | cifar10-classification | 95.3% → 95.3% (cifar10-test-first-128, n=128) | – | 2.4 ◇ / 1.39 ◇<br>int8 2.32 ◇ / – |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | imagenet-classification | 64.1% → 64.8% (imagenet-val-subset-128, n=128) | – | 11.1 ◇ / 5.85 ◇<br>int8 7.82 ◇ / – |
| `snn-subbit-reactnet18-w1a1-imagenet` | imagenet-classification | 64.8% → 64.8% (imagenet-val-subset-128, n=128) | – | 11.1 ◇ / 5.87 ◇<br>int8 7.81 ◇ / – |
| `tensorpack-dorefa-alexnet-w1a1-g32` | imagenet-classification | 46.1% → 46.1% (imagenet-val-subset-128, n=128) | – | 11.4 ◇ / 5.82 ◇<br>int8 7.68 ◇ / – |

Each cell: median latency of one inference, 1 thread pinned to the fastest core / 4 threads; first line exact binary layers with float glue (stage all), "int8" line the full-integer file (int8_full) where measured, else the int8 kernels on the exact file. Galaxy S22: Qualcomm Snapdragon 8 Gen 1 (SM8450), 1 thread on the Cortex-X2; OnePlus Open: Qualcomm Snapdragon 8 Gen 2 (SM8550), 1 thread on the Cortex-X3. Unmarked = clean (every repeat at full clocks). ◇ = the phone's own power policy held the clock ceiling below the maximum with no thermal event (stock settings, screen off; the ceiling is recorded per run); ⚠ = at least one repeat ran under a thermal CPU-frequency cap (e.g. a phone charging over USB); † = run made before the frequency-cap check existed (caps unknown). Runs per device and validity: Galaxy S22: 185 thermal-throttled, 24 unknown-caps; OnePlus Open: 456 vendor-policy. Per-run details (date, runtime, clock ceilings): `results/speed_<device>.json`. Emulated timings are never published.

On-device correctness (every model, every stage, `results/models/<id>.json` → device_verification): Galaxy S22: 342/342 correctness checks passed on 114 models, 2026-11-02 (outputs byte-identical to the emulated-ARM run of the same code); OnePlus Open: 449/449 correctness checks passed on 114 models, 2026-11-11 (outputs byte-identical to the emulated-ARM run of the same code).
<!-- /generated:speed -->

The phones: `results/devices.json`. All runs with their clock state: `results/speed_<device>.json`; x86-64 host
runs (older runtime, labelled): `results/speed_x86.json`. Method: [docs/benchmarking.md](docs/benchmarking.md).

## Model zoo

Models measured on the phone so far (full table with all converted models: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md);
searchable version on the website):

<!-- generated:zoo_top -->
| model | task | framework | accuracy (reference → Bitino) | eval set (n) | int8_full Δ vs exact | fastest phone ms (1 thread) | weights licence | release |
|---|---|---|---|---|---|---|---|---|
| `ab-bnn-cifar10-reactnet18` | cifar10-classification | pytorch | 93.8% → 93.8% | cifar10-test-first-128 (128) | +0.06 pt (cifar10-test-first-10000, n=10000) | 2.32 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-cifar10-reactneta` | cifar10-classification | pytorch | 91.4% → 91.4% | cifar10-test-first-128 (128) | -0.22 pt (cifar10-test-first-10000, n=10000) | 1.61 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-cifar100-reactnet18` | cifar100-classification | pytorch | 69.5% → 69.5% | cifar100-test-first-128 (128) | -0.56 pt (cifar100-test-10000, n=10000) | 2.32 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-cifar100-reactneta` | cifar100-classification | pytorch | 71.9% → 71.9% | cifar100-test-first-128 (128) | -0.31 pt (cifar100-test-10000, n=10000) | 1.62 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactnet18` | imagenet-classification | pytorch | 65.6% → 64.8% | imagenet-val-subset-128 (128) | -0.34 pt (imagenet-val-subset-5000, n=5000) | 11.1 ◇ (OnePlus Open) | MIT (HF model card metadata 'license: mit'); app | yes |
| `ab-bnn-imagenet-reactnet34` | imagenet-classification | pytorch | 67.2% → 68.0% | imagenet-val-subset-128 (128) | -0.62 pt (imagenet-val-subset-5000, n=5000) | 17.7 ◇ (OnePlus Open) | MIT (HF card) | yes |
| `ab-bnn-imagenet-reactneta` | imagenet-classification | pytorch | 66.4% → 65.6% | imagenet-val-subset-128 (128) | -0.58 pt (imagenet-val-subset-5000, n=5000) | 19.6 ⚠ (Galaxy S22) | MIT (HF card) | yes |
| `adabin-resnet20-cifar10` | cifar10-classification | pytorch | 91.4% → 91.4% | cifar10-test-first-128 (128) | -0.64 pt (cifar10-test-first-10000, n=10000) | 1.22 ⚠ (Galaxy S22) | MIT by co-location in the same directory (no sep | – |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | cifar10-classification | pytorch+brevitas | 89.8% → 89.8% | cifar10-test-first-128 (128) | +0.06 pt (cifar10-test-first-10000, n=10000) | 0.57 ⚠ (Galaxy S22) | unspecified | – |
| `bbcu-deblocking` | jpeg-artifact-reduction (grayscale, Y channel) | pytorch | 31.33 dB → 31.33 dB | live1-classic5-jpeg-y-q10q40-center96 (68) | – | 28.9 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bbcu-denoise` | color-image-denoising (AWGN blind) | pytorch | 29.74 dB → 29.74 dB | cbsd68-sigma25-center96 (68) | – | 29.2 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-l-x2` | single-image-super-resolution-x2 | pytorch | 32.03 dB → 32.03 dB | set5-set14-b100-x2-lr48 (64) | – | 36.6 ⚠ (Galaxy S22) | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-l-x4` | single-image-super-resolution-x4 | pytorch | 27.36 dB → 27.36 dB | set5-set14-b100-x4-lr32 (64) | – | 53.5 ⚠ (Galaxy S22) | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-m-x2` | single-image-super-resolution-x2 | pytorch | 31.89 dB → 31.89 dB | set5-set14-b100-x2-lr48 (64) | – | 18.7 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bbcu-sr-m-x4` | single-image-super-resolution-x4 | pytorch | 27.22 dB → 27.22 dB | set5-set14-b100-x4-lr32 (64) | – | 18.5 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bhvit-small-binary` | imagenet-classification | pytorch | 68.8% → 65.6% | imagenet-val-subset-128 (128) | – | 50.0 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `bhvit-small-somefp` | imagenet-classification | pytorch | 72.7% → 76.6% | imagenet-val-subset-128 (128) | -1.90 pt (imagenet-val-subset-5000_bhvit, n=5000) | 52.3 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `bhvit-tiny-binary` | imagenet-classification | pytorch | 64.1% → 62.5% | imagenet-val-subset-128 (128) | -0.74 pt (imagenet-val-subset-5000_bhvit, n=5000) | 32.8 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `bhvit-tiny-somefp` | imagenet-classification | pytorch | 65.6% → 65.6% | imagenet-val-subset-128 (128) | -6.20 pt (imagenet-val-subset-5000_bhvit, n=5000) | 35.1 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `bi-vit-deit-small` | imagenet-classification | pytorch | 11.7% → 10.9% | imagenet-val-subset-128 (128) | -0.88 pt (imagenet-val-subset-5000_bhvit, n=5000) | 41.9 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bi-vit-deit-tiny` | imagenet-classification | pytorch | 15.6% → 15.6% | imagenet-val-subset-128 (128) | -0.34 pt (imagenet-val-subset-5000_bhvit, n=5000) | 16.8 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bibert-bert-base-mnli` | glue-mnli | pytorch | 61.7% → 61.7% | mnli-dev-matched-128 (128) | +0.54 pt (mnli-dev-matched-9815, n=9815) | 62.6 ◇ (OnePlus Open) | unspecified | – |
| `bibert-bert-base-mrpc` | glue-mrpc | pytorch | 76.6% → 76.6% | mrpc-dev-128 (128) | +0.98 pt (mrpc-dev-408, n=408) | 60.6 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bibert-bert-base-qqp` | glue-qqp | pytorch | 83.6% → 83.6% | qqp-dev-128 (128) | -0.59 pt (qqp-dev-40430, n=40430) | 68.2 ◇ (OnePlus Open) | unspecified | – |
| `bibert-bert-base-sst2` | glue-sst2 | pytorch | 89.8% → 89.8% | sst2-dev-128 (128) | -0.34 pt (sst2-dev-872, n=872) | 32.1 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `bimatting-bimobilenet-backbone` | imagenet-classification (backbone) | pytorch | 55.5% → 55.5% | imagenet-val-subset-128 (128) | -22.86 pt (imagenet-val-subset-5000, n=5000) | 9.91 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `biper-resnet18-cifar10` | cifar10-classification | pytorch | 94.5% → 94.5% | cifar10-test-first-128 (128) | -1.94 pt (cifar10-test-first-10000, n=10000) | 2.01 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `biper-resnet18-imagenet` | imagenet-classification | pytorch | 59.4% → 59.4% | imagenet-val-subset-128 (128) | -0.12 pt (imagenet-val-subset-5000, n=5000) | 11.2 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `biper-resnet20-cifar10` | cifar10-classification | pytorch | 86.7% → 86.7% | cifar10-test-first-128 (128) | -0.63 pt (cifar10-test-first-10000, n=10000) | 1.22 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `biper-resnet34-imagenet` | imagenet-classification | pytorch | 66.4% → 66.4% | imagenet-val-subset-128 (128) | -0.12 pt (imagenet-val-subset-5000, n=5000) | 17.5 ◇ (OnePlus Open) | unspecified (MIT repo; no separate weights licen | – |
| `biper-vgg-small-cifar10` | cifar10-classification | pytorch | 96.1% → 96.1% | cifar10-test-first-128 (128) | -1.40 pt (cifar10-test-first-10000, n=10000) | 2.08 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `bitorch-densenet28-imagenet` | imagenet-classification | pytorch | 59.4% → 59.4% | imagenet-val-subset-128 (128) | -0.76 pt (imagenet-val-subset-5000, n=5000) | 22.8 ◇ (OnePlus Open) | unspecified (no licence on the W&B artifacts; GP | – |
| `bitorch-meliusnet22-imagenet` | imagenet-classification | pytorch | 66.4% → 65.6% | imagenet-val-subset-128 (128) | +60.44 pt (imagenet-val-subset-5000, n=5000) | 26.9 ◇ (OnePlus Open) | unspecified (no licence on the W&B artifacts; GP | – |
| `bitorch-resnete18-imagenet` | imagenet-classification | pytorch | 64.8% → 62.5% | imagenet-val-subset-128 (128) | -0.84 pt (imagenet-val-subset-5000, n=5000) | 10.8 ◇ (OnePlus Open) | unspecified (no licence on the W&B artifacts; GP | – |
| `bnas-a-cifar10` | cifar10-classification | pytorch | 93.8% → 93.8% | cifar10-test-first-128 (128) | -0.42 pt (cifar10-test-first-10000, n=10000) | 8.74 ◇ (OnePlus Open) | none stated | – |
| `bnas-d-imagenet` | imagenet-classification | pytorch | 60.2% → 60.2% | imagenet-val-subset-128 (128) | +53.34 pt (imagenet-val-subset-5000, n=5000) | 26.4 ◇ (OnePlus Open) | none stated | – |
| `bnas-e-imagenet` | imagenet-classification | pytorch | 56.2% → 56.2% | imagenet-val-subset-128 (128) | -0.92 pt (imagenet-val-subset-5000, n=5000) | 32.7 ◇ (OnePlus Open) | unspecified | – |
| `bnas-h-imagenet` | imagenet-classification | pytorch | 60.2% → 60.2% | imagenet-val-subset-128 (128) | +0.08 pt (imagenet-val-subset-5000, n=5000) | 106.6 ◇ (OnePlus Open) | unspecified | – |
| `bnext-large` | imagenet-classification | pytorch | 79.7% → 80.5% | imagenet-val-subset-128 (128) | -0.88 pt (imagenet-val-subset-5000, n=5000) | 191.3 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bnext-middle` | imagenet-classification | pytorch | 78.9% → 78.9% | imagenet-val-subset-128 (128) | -1.34 pt (imagenet-val-subset-5000, n=5000) | 86.4 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bnext-small` | imagenet-classification | pytorch | 76.6% → 75.0% | imagenet-val-subset-128 (128) | -2.70 pt (imagenet-val-subset-5000, n=5000) | 51.9 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bnext-tiny` | imagenet-classification | pytorch | 69.5% → 68.0% | imagenet-val-subset-128 (128) | -3.84 pt (imagenet-val-subset-5000, n=5000) | 25.6 ◇ (OnePlus Open) | no explicit weights licence (inherits repo Apach | – |
| `bnn-pynq-cnv-w1a1-cifar10` | cifar10-classification | theano+lasagne (BNN-PYNQ) | 86.7% → 86.7% | cifar10-test-first-128 (128) | +0.42 pt (cifar10-test-first-10000, n=10000) | 0.712 ◇ (OnePlus Open) | BSD-3-Clause by inclusion in the repo tree (no s | – |
| `bnn-pynq-cnv-w1a1-gtsrb` | gtsrb-classification (43 classes + junk = 44) | theano+lasagne (BNN-PYNQ) | 93.8% → 93.8% | gtsrb-test-udacity32-first-128 (128) | – | 0.731 ◇ (OnePlus Open) | BSD-3-Clause by inclusion in the repo tree | – |
| `bnn-pynq-lfc-w1a1-mnist` | mnist-classification | theano+lasagne (BNN-PYNQ) | 100.0% → 100.0% | mnist-test-first-128 (128) | – | 0.055 ◇ (OnePlus Open) | BSD-3-Clause by inclusion in the repo tree | – |
| `brevitas-bnn-pynq-cnv-1w1a` | cifar10-classification | pytorch+brevitas | 85.9% → 85.9% | cifar10-test-first-128 (128) | -0.77 pt (cifar10-test-first-10000, n=10000) | 0.701 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `brevitas-bnn-pynq-lfc-1w1a` | mnist-classification | pytorch+brevitas | 100.0% → 100.0% | mnist-test-first-128 (128) | – | 0.054 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `brevitas-bnn-pynq-sfc-1w1a` | mnist-classification | pytorch+brevitas | 100.0% → 100.0% | mnist-test-first-128 (128) | – | 0.03 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `brevitas-bnn-pynq-tfc-1w1a` | mnist-classification | pytorch+brevitas | 96.9% → 96.9% | mnist-test-first-128 (128) | – | 0.017 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `brillsson-binarydensenet28` | non-semantic speech representation (embedding; linear-probe on SpeechCommands / emotion / language-ID etc.) | keras+larq | 90.5% → 90.5% | Speech Commands v0.02 test (128) | – | 2.52 ◇ (OnePlus Open) | unspecified | – |
| `brillsson-meliusnet22` | non-semantic speech representation (embedding; linear-probe on SpeechCommands / emotion / language-ID etc.) | keras+larq | 91.9% → 91.9% | Speech Commands v0.02 test (128) | – | 3.4 ◇ (OnePlus Open) | unspecified | – |
| `brillsson-quicknet` | non-semantic speech representation (embedding; linear-probe on SpeechCommands / emotion / language-ID etc.) | keras+larq | 90.7% → 90.7% | Speech Commands v0.02 test (128) | – | 1.02 ◇ (OnePlus Open) | unspecified | – |
| `bsrc2023-binary-sr-x2-v0` | single-image-super-resolution-x2 | pytorch | 31.88 dB → 31.88 dB | set5-set14-b100-x2-lr48 (64) | – | 28.2 ◇ (OnePlus Open) | unspecified | – |
| `bsrc2023-binary-sr-x2-v1` | single-image-super-resolution-x2 | pytorch | 32.22 dB → 32.22 dB | set5-set14-b100-x2-lr48 (64) | – | 139.6 ◇ (OnePlus Open) | unspecified | – |
| `bsrc2023-binary-sr-x4-v0` | single-image-super-resolution-x4 | pytorch | 27.22 dB → 27.22 dB | set5-set14-b100-x4-lr32 (64) | – | 62.2 ◇ (OnePlus Open) | unspecified | – |
| `bsrc2023-binary-sr-x4-v1` | single-image-super-resolution-x4 | pytorch | 27.32 dB → 27.32 dB | set5-set14-b100-x4-lr32 (64) | – | 76.0 ◇ (OnePlus Open) | unspecified | – |
| `burn-reacta-stage2` | imagenet self-supervised representation learning (backbone; the ImageNet linear-eval head is NOT included) | pytorch | – → 0.99615 | imagenet-val-subset-128 / 5000 (128) | – | 20.8 ◇ (OnePlus Open) | No separate statement. Distributed via the READM | – |
| `dabnn-bireal18` | imagenet-classification | onnx (daBNN model zoo) | 58.6% → 58.6% | imagenet-val-subset-128 (128) | +0.16 pt (imagenet-val-subset-5000, n=5000) | 10.8 ◇ (OnePlus Open) | No explicit weights licence (inherits repo BSD-3 | – |
| `dabnn-bireal18-stem` | imagenet-classification | onnx (daBNN model zoo) | 58.6% → 58.6% | imagenet-val-subset-128 (128) | -0.42 pt (imagenet-val-subset-5000, n=5000) | 8.36 ◇ (OnePlus Open) | No explicit weights licence (inherits repo BSD-3 | – |
| `fda-bnn-resnet20-cifar10-mindspore` | cifar10-classification | mindspore (checkpoint read without MindSpore) | 89.8% → 89.8% | cifar10-test-first-128 (128) | -0.55 pt (cifar10-test-first-10000, n=10000) | 1.23 ◇ (OnePlus Open) | no explicit weights licence (inherits Apache-2.0 | – |
| `finn-examples-bnn-pynq-cnv-w1a1` | cifar10-classification / mnist-classification | FINN-ONNX/QONNX (qonnx) | 85.9% → 85.9% | cifar10-test-first-128 (128) | -0.77 pt (cifar10-test-first-10000, n=10000) | 0.73 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `finn-examples-bnn-pynq-tfc-w1a1` | cifar10-classification / mnist-classification | FINN-ONNX/QONNX (qonnx) | 96.9% → 96.9% | mnist-test-first-128 (128) | – | 0.016 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
| `finn-examples-cnv-w1a1-gtsrb` | gtsrb-classification (traffic signs; 44 outputs) | FINN-ONNX/QONNX (qonnx) | 93.8% → 93.8% | gtsrb-test-udacity32-first-128 (128) | – | 0.701 ◇ (OnePlus Open) | unspecified separately; QONNX zoo folder ships a | – |
| `gsb-deit-s-cifar100-stage2` | cifar100-classification | pytorch | 70.3% → 69.5% | cifar100-test-first-128-deit224 (128) | -1.99 pt (cifar100-test-first-10000-deit224, n=10000) | 106.0 ◇ (OnePlus Open) | no explicit weights licence (inherits repo MIT?  | – |
| `ida-det-bires18-imagenet-backbone` | imagenet-classification (backbone for detection) | pytorch | 65.6% → 65.6% | imagenet-val-subset-128 (128) | -0.64 pt (imagenet-val-subset-5000, n=5000) | 11.1 ◇ (OnePlus Open) | no explicit weights licence (repo has none) | – |
| `larq-zoo-binary-alexnet` | imagenet-classification | larq_zoo | 28.9% → 28.9% | imagenet-val-subset-128 (128) | +0.30 pt (imagenet-val-subset-5000, n=5000) | 5.48 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-binary-densenet28` | imagenet-classification | larq_zoo | 60.9% → 60.9% | imagenet-val-subset-128 (128) | -0.66 pt (imagenet-val-subset-5000, n=5000) | 22.8 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-binary-densenet37` | imagenet-classification | larq_zoo | 66.4% → 65.6% | imagenet-val-subset-128 (128) | -41.28 pt (imagenet-val-subset-5000, n=5000) | 26.4 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-binary-densenet37-dilated` | imagenet-classification | larq_zoo | 65.6% → 65.6% | imagenet-val-subset-128 (128) | -59.00 pt (imagenet-val-subset-5000, n=5000) | 47.6 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-binary-densenet45` | imagenet-classification | larq_zoo | 60.9% → 61.7% | imagenet-val-subset-128 (128) | -43.86 pt (imagenet-val-subset-5000, n=5000) | 35.8 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-binary-resnet-e18` | imagenet-classification | larq_zoo | 55.5% → 55.5% | imagenet-val-subset-128 (128) | -0.92 pt (imagenet-val-subset-5000, n=5000) | 10.8 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-birealnet` | imagenet-classification | larq_zoo | 57.8% → – | imagenet-val-subset-128 (128) | -0.20 pt (imagenet-val-subset-5000, n=5000) | 10.7 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-dorefanet` | imagenet-classification | larq_zoo | 46.1% → 46.1% | imagenet-val-subset-128 (128) | -0.64 pt (imagenet-val-subset-5000, n=5000) | 23.1 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-meliusnet22` | imagenet-classification | larq_zoo | 65.6% → 66.4% | imagenet-val-subset-128 (128) | -57.38 pt (imagenet-val-subset-5000, n=5000) | 26.6 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet` | imagenet-classification | larq_zoo | 66.4% → 66.4% | imagenet-val-subset-128 (128) | -0.72 pt (imagenet-val-subset-5000, n=5000) | 7.83 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet-large` | imagenet-classification | larq_zoo | 71.1% → 70.3% | imagenet-val-subset-128 (128) | -1.96 pt (imagenet-val-subset-5000, n=5000) | 14.0 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet-small` | imagenet-classification | larq_zoo | 59.4% → 58.6% | imagenet-val-subset-128 (128) | -1.96 pt (imagenet-val-subset-5000, n=5000) | 5.05 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-quicknet-xl-legacy` | imagenet-classification | larq_zoo | 64.1% → 64.8% | imagenet-val-subset-128 (128) | -0.92 pt (imagenet-val-subset-5000, n=5000) | 13.7 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-real-to-binary-net` | imagenet-classification | larq_zoo | 64.8% → 64.8% | imagenet-val-subset-128 (128) | -1.12 pt (imagenet-val-subset-5000, n=5000) | 12.7 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larq-zoo-xnornet` | imagenet-classification | larq_zoo | 34.4% → 34.4% | imagenet-val-subset-128 (128) | -0.36 pt (imagenet-val-subset-5000, n=5000) | 7.85 ◇ (OnePlus Open) | No explicit weights licence (inherits repo licen | – |
| `larqface-binarydensenet28-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | keras+larq | 99.2% → 99.2% | LFW + AgeDB-30 insightface .bin (6000) | – | 5.58 ◇ (OnePlus Open) | non-commercial research only (README: 'training  | – |
| `larqface-binarydensenet37-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | keras+larq | 99.2% → 99.2% | LFW + AgeDB-30 insightface .bin (6000) | – | 6.5 ◇ (OnePlus Open) | non-commercial research only (README: 'training  | – |
| `larqface-binarydensenet45-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | keras+larq | 99.2% → 99.2% | LFW + AgeDB-30 insightface .bin (6000) | – | 9.11 ◇ (OnePlus Open) | non-commercial research only (README: 'training  | – |
| `larqface-binaryfacenet-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | keras+larq | 95.1% → 95.1% | LFW + AgeDB-30 insightface .bin (6000) | – | 16.2 ◇ (OnePlus Open) | non-commercial research only (README: 'training  | – |
| `larqface-quicknet-msceleb` | face-recognition (embedding; verification on LFW/CFP-FP/AgeDB-30, low-resolution face recognition) | keras+larq | 99.0% → 99.0% | LFW + AgeDB-30 insightface .bin (6000) | – | 1.92 ◇ (OnePlus Open) | non-commercial research only (README: 'training  | – |
| `nncf-resnet18-binarization-dorefa` | imagenet-classification | pytorch+nncf | 62.5% → 62.5% | imagenet-val-subset-128 (128) | +0.14 pt (imagenet-val-subset-5000, n=5000) | 10.9 ◇ (OnePlus Open) | Not stated separately. Presumably Apache-2.0 (un | – |
| `nncf-resnet18-binarization-xnor` | imagenet-classification | pytorch+nncf | 61.7% → 61.7% | imagenet-val-subset-128 (128) | -0.12 pt (imagenet-val-subset-5000, n=5000) | 10.9 ◇ (OnePlus Open) | Not stated separately. Presumably Apache-2.0 as  | – |
| `omz-resnet18-xnor-binary-onnx-0001` | imagenet-classification | openvino-ir | 59.4% → 59.4% | imagenet-val-subset-128 (128) | -1.20 pt (imagenet-val-subset-5000, n=5000) | 11.0 ◇ (OnePlus Open) | Apache-2.0 | yes |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | imagenet-classification | openvino-ir | 64.1% → 64.1% | imagenet-val-subset-128 (128) | +0.00 pt (imagenet-val-subset-5000, n=5000) | 11.1 ◇ (OnePlus Open) | Apache-2.0 | yes |
| `omz-resnet50-binary-0001` | imagenet-classification | openvino-ir | 72.7% → 73.4% | imagenet-val-subset-128 (128) | -0.04 pt (imagenet-val-subset-5000, n=5000) | 34.4 ◇ (OnePlus Open) | Apache-2.0 (model.yml license field points to th | yes |
| `omz-resnet50-binary-0001-fp16` | imagenet-classification | openvino-ir | 71.9% → 73.4% | imagenet-val-subset-128 (128) | +0.26 pt (imagenet-val-subset-5000, n=5000) | 34.4 ◇ (OnePlus Open) | Apache-2.0 | yes |
| `rbnn-resnet18-cifar10` | cifar10-classification | pytorch | 94.5% → 94.5% | cifar10-test-first-128 (128) | -0.75 pt (cifar10-test-first-10000, n=10000) | 2.77 ◇ (OnePlus Open) | none stated | – |
| `rbnn-resnet18-imagenet` | imagenet-classification | pytorch | 53.1% → 53.1% | imagenet-val-subset-128 (128) | -0.48 pt (imagenet-val-subset-5000, n=5000) | 13.1 ◇ (OnePlus Open) | none stated | – |
| `rbnn-resnet20-bireal-cifar10` | cifar10-classification | pytorch | 89.8% → 89.8% | cifar10-test-first-128 (128) | -1.13 pt (cifar10-test-first-10000, n=10000) | 0.858 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `rbnn-resnet20-cifar10` | cifar10-classification | pytorch | 88.3% → 88.3% | cifar10-test-first-128 (128) | -1.01 pt (cifar10-test-first-10000, n=10000) | 0.86 ◇ (OnePlus Open) | none stated | – |
| `rbnn-resnet34-imagenet` | imagenet-classification | pytorch | 64.1% → 63.3% | imagenet-val-subset-128 (128) | -0.52 pt (imagenet-val-subset-5000, n=5000) | 20.9 ◇ (OnePlus Open) | none stated | – |
| `rbnn-vgg-small-cifar10` | cifar10-classification | pytorch | 91.4% → 91.4% | cifar10-test-first-128 (128) | -0.50 pt (cifar10-test-first-10000, n=10000) | 2.39 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `rbonn-reactnet-resnet18-imagenet` | imagenet-classification | pytorch | 65.6% → 65.6% | imagenet-val-subset-128 (128) | -0.56 pt (imagenet-val-subset-5000, n=5000) | 11.1 ◇ (OnePlus Open) | none stated | – |
| `recu-resnet18-cifar10` | cifar10-classification | pytorch | 93.8% → 93.8% | cifar10-test-first-128 (128) | -0.29 pt (cifar10-test-first-10000, n=10000) | 2.05 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `recu-resnet18-imagenet` | imagenet-classification | pytorch | 59.4% → 59.4% | imagenet-val-subset-128 (128) | -0.14 pt (imagenet-val-subset-5000, n=5000) | 11.2 ◇ (OnePlus Open) | none stated | – |
| `recu-resnet20-cifar10` | cifar10-classification | pytorch | 88.3% → 88.3% | cifar10-test-first-128 (128) | -0.42 pt (cifar10-test-first-10000, n=10000) | 1.2 ◇ (OnePlus Open) | none stated | – |
| `recu-resnet34-imagenet` | imagenet-classification | pytorch | 68.8% → 68.8% | imagenet-val-subset-128 (128) | -4.56 pt (imagenet-val-subset-5000, n=5000) | 17.6 ◇ (OnePlus Open) | none stated | – |
| `recu-vgg-small-cifar10` | cifar10-classification | pytorch | 96.1% → 96.1% | cifar10-test-first-128 (128) | -0.84 pt (cifar10-test-first-10000, n=10000) | 2.09 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `s2bnn-reacta-distill-only` | imagenet self-supervised representation learning (backbone + MoCo projection; no ImageNet classifier) | pytorch | – → 0.96370 | imagenet-val-subset-128 / 5000 (128) | – | 20.6 ◇ (OnePlus Open) | No statement | – |
| `sa-bnn-resnet18-imagenet` | imagenet-classification | pytorch | 61.7% → 62.5% | imagenet-val-subset-128 (128) | -0.78 pt (imagenet-val-subset-5000, n=5000) | 11.9 ◇ (OnePlus Open) | none stated | – |
| `shuaiz-bireal20-cifar10` | cifar10-classification | pytorch | 84.4% → 84.4% | cifar10-test-first-128 (128) | -0.55 pt (cifar10-test-first-10000, n=10000) | 1.11 ◇ (OnePlus Open) | unspecified | – |
| `siman-resnet18-cifar10` | cifar10-classification | pytorch | 95.3% → 95.3% | cifar10-test-first-128 (128) | -0.15 pt (cifar10-test-first-10000, n=10000) | 2.75 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `siman-resnet18-imagenet` | imagenet-classification | pytorch | 54.7% → 54.7% | imagenet-val-subset-128 (128) | -4.58 pt (imagenet-val-subset-5000, n=5000) | 12.8 ◇ (OnePlus Open) | none stated | – |
| `siman-resnet20-cifar10` | cifar10-classification | pytorch | 87.5% → 87.5% | cifar10-test-first-128 (128) | -2.32 pt (cifar10-test-first-10000, n=10000) | 0.868 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `siman-resnet34-imagenet` | imagenet-classification | pytorch | 59.4% → 60.9% | imagenet-val-subset-128 (128) | -14.10 pt (imagenet-val-subset-5000, n=5000) | 20.5 ◇ (OnePlus Open) | none stated | – |
| `siman-vgg-small-cifar10` | cifar10-classification | pytorch | 95.3% → 95.3% | cifar10-test-first-128 (128) | -0.49 pt (cifar10-test-first-10000, n=10000) | 2.4 ◇ (OnePlus Open) | unspecified (no separate weights licence) | – |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | imagenet-classification | pytorch | 64.1% → 64.8% | imagenet-val-subset-128 (128) | -0.68 pt (imagenet-val-subset-5000, n=5000) | 11.1 ◇ (OnePlus Open) | unspecified (MIT repo) | – |
| `snn-subbit-reactnet18-w1a1-imagenet` | imagenet-classification | pytorch | 64.8% → 64.8% | imagenet-val-subset-128 (128) | -0.20 pt (imagenet-val-subset-5000, n=5000) | 11.1 ◇ (OnePlus Open) | unspecified (MIT repo) | – |
| `tensorpack-dorefa-alexnet-w1a1-g32` | imagenet-classification | tensorpack (TF1) npz, read with numpy | 46.1% → 46.1% | imagenet-val-subset-128 (128) | -0.32 pt (imagenet-val-subset-5000 (tensorpack_bgr), n=5000) | 11.4 ◇ (OnePlus Open) | no explicit weights licence (inherits repo licen | – |
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

* **Speed data has caveats.** The OnePlus Open numbers are on the current runtime with the phone's stock power
  policy, which holds the clocks below their maximum with the screen off (◇, ceilings recorded per run). The S22
  numbers are from an older runtime on a thermally capped phone (charging over USB) and are labelled so. x86 host
  numbers are from an older runtime.
* **Fetch recipes:** a few are verified end to end; the others are reconstructed from the conversion records
  (state shown by `bitino list`). Some sources need a manual download (Google Drive, Baidu) or a second reference
  environment (MindSpore, tensorpack, Python 2.7 Theano).
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
