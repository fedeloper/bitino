# Getting started

## 1. Install

Bitino has two halves: the **runtime** (C++17, no dependencies) that executes models, and the **tools** (Python,
numpy only) that list, fetch, convert, validate and quantize them.

Only to run models from Python: `pip install "bitino[image]"` (binary wheels with the runtime inside; see
[python.md](python.md)). For the whole tool chain (fetch recipes, converters, validation, Android builds):

```bash
git clone https://github.com/fedeloper/bitino && cd bitino
pip install -e .                       # the `bitino` command and the Python API (editable: it reads registry/ and
                                       # results/; compiles the runtime library libbitino_c with CMake)
```

Runtime, either prebuilt or from source:

```bash
# prebuilt: bitino_bench-linux-x86_64 / -linux-aarch64 / -android-arm64 from the GitHub Releases page
chmod +x bitino_bench-linux-x86_64 && export BITINO_BENCH=$PWD/bitino_bench-linux-x86_64

# from source (CMake >= 3.20, a C++17 compiler; Ninja optional; on aarch64 Linux use clang:
# add -DCMAKE_CXX_COMPILER=clang++, since GCC rejects the NEON kernels' inline asm)
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/host
./build/host/bitino_tests              # unit tests of every kernel against its scalar reference
```

The `bitino` command finds the runtime at `$BITINO_BENCH`, `build/host/bitino_bench` or on `PATH`
(or pass `--bench PATH`).

## 2. Pick a model

```bash
bitino list --converted                # every converted model, recipe state, licence, accuracy check
bitino list --task cifar10 --release   # filters: --task --family --framework --strict --release
bitino info ab-bnn-imagenet-reactnet18 # paper, licences, recipe, results
```

## 3. Fetch it

Two ways:

```bash
bitino fetch ab-bnn-imagenet-reactnet18 --prebuilt   # converted file from the GitHub release (permissive weights only)
bitino fetch ab-bnn-imagenet-reactnet18              # from the original source, converted on your machine
```

From the original source, `fetch`

1. downloads the authors' weights into a new empty directory under `~/.cache/bitino/downloads/<model>/`
   (`$BITINO_HOME` moves the cache) and checks the recorded SHA-256 while streaming;
2. downloads the authors' model code at the recorded commit (GitHub archive or PyPI wheel, archive members checked);
3. runs the converter of that framework with `python -I` in an empty work directory; checkpoints are only read
   with `torch.load(weights_only=True)` (or the framework's own weight loader for `.h5`, IR, ONNX);
4. compares the converted file's data section with the published conversion and prints `IDENTICAL` or `DIFFERS`.

Converters need the source framework. `bitino fetch` tells you which environment a model needs (`envs/`) and
how to pass it:

```bash
python3.11 -m venv ~/.venvs/bitino-larq && ~/.venvs/bitino-larq/bin/pip install -r envs/larq-zoo-requirements.txt
bitino fetch larq-zoo-quicknet-small --python larq=$HOME/.venvs/bitino-larq/bin/python
```

The converters also compute reference outputs. Datasets are not redistributed, so by default `fetch` makes a small
synthetic eval set (seeded random inputs of the right shape). That is enough for the conversion and for the
runtime-vs-reference checks; to reproduce the published accuracy, build the real eval set with
`tools/build_eval_set.py` from your copy of the dataset and pass `--eval-dir`.

Other options: `--file weights=PATH` (use a file you downloaded by hand, e.g. from Google Drive; its SHA-256 is
still checked), `--source repo=DIR` (local source tree), `--int8 --calib-dir DIR` (also build the int8_full file
with the recorded quantizer flags), `--threads N` (worker processes and framework threads of the converter,
default 4), `--dry-run` (print the recipe).

## 4. Run, time, validate

```bash
bitino run ab-bnn-imagenet-reactnet18 --random 2              # random inputs, prints top-5 per sample
bitino run ab-bnn-imagenet-reactnet18 --input x.f32 --output y.f32   # raw float32 NHWC inputs
bitino bench ab-bnn-imagenet-reactnet18 --threads 4 --stage all
bitino bench ab-bnn-imagenet-reactnet18 --int8 --stage int8_full
bitino validate ab-bnn-imagenet-reactnet18                    # runtime vs numpy executor vs source framework
```

Stages: `reference` (scalar oracle), `neon` (exact, bit-identical with the numpy executor), `all` / `optimized`
(float fusion and the fastest kernels), `optimized_int8`, `int8_full` (full-integer file). On x86-64 the same
stage names select the AVX2 / AVX-VNNI kernels.

## 5. Deploy on Android

```bash
adb push bitino_bench-android-arm64 /data/local/tmp/bitino_bench      # or build/android-arm64/bitino_bench
adb push ~/.cache/bitino/models/ab-bnn-imagenet-reactnet18/model.bitino /data/local/tmp/
adb shell /data/local/tmp/bitino_bench --model /data/local/tmp/model.bitino --threads 1 --affinity 7   # big core
adb shell /data/local/tmp/bitino_bench --model /data/local/tmp/model.bitino --threads 4
```

Building it yourself: `scripts/build_android.sh` (Android NDK r27c tested; `ANDROID_NDK=...`). No root needed.
For an Android app use the AAR and its Kotlin API (`android/bitino`, see
[android-deployment.md](android-deployment.md)); other native code can link `libbitino.a` and use the C API:

```c
#include "bitino/bitino_c.h"
bf_model* m; bf_session* s;
bf_model_load("model.bitino", 1, &m);                 /* 1 = verify the data SHA-256 */
bf_session_create(m, 4, "all", &s);                    /* threads, stage */
bf_session_run(s, input, bf_session_input_size(s, 0));
bf_session_get_output(s, 0, output, bf_session_output_size(s, 0));
bf_session_free(s); bf_model_free(m);
```

A minimal Java JNI wrapper is in `examples/android_jni/`. More: [android-deployment.md](android-deployment.md),
[architecture.md](architecture.md), [threads.md](threads.md).

## 6. Where the numbers come from

* `results/models/<model>.json`: accuracy checks (eval set, n, Δ vs reference), int8 studies, emulated
  instruction counts, measured phone and x86 speed.
* `results/devices.json`: the phones; `results/speed_<device>.json`, `results/speed_x86.json`: every measured run
  with date, threads, affinity and clock state. Emulated timings are never recorded.
* [benchmarking.md](benchmarking.md): the measurement method; [numerical-correctness.md](numerical-correctness.md):
  how conversions and kernels are validated; [int8.md](int8.md): the int8 paths.
