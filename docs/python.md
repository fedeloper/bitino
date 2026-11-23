# Python API

`pip install bitino` installs the native runtime (the same C++ kernels as `bitino_bench`, compiled into
`libbitino_c`) together with the Python package: model zoo, CLI, converters and the numpy executor.

```bash
pip install "bitino[image]"      # [image] adds Pillow for image decoding
```

```python
import bitino

m = bitino.load("ab-bnn-imagenet-reactnet18")    # zoo id (downloaded if redistributable) or a .bitino path
print(m.inputs, m.outputs)                         # [TensorSpec(name='input', shape=(1, 224, 224, 3), dtype=float32)] ...
logits = m.predict_image("cat.jpg")                # decode + the model's own preprocessing, then predict()
y = m.predict(x)                                   # numpy in (the model's layout: NHWC for images), numpy out
print(m.benchmark(iterations=50, threads=1))       # median latency in ms on this machine
print(bitino.cpu_info())                         # {'arch': 'x86_64', 'features': ['popcnt', 'avx2', ...], ...}
```

Binary wheels: Linux x86-64 and aarch64 (manylinux_2_28, glibc 2.28+), macOS arm64 (11+), Windows x64. One wheel
per platform serves every Python >= 3.10 (the runtime is a plain C library loaded with `ctypes`, so the wheels do
not depend on the CPython ABI). Elsewhere pip builds from the sdist: CMake >= 3.20 and a C++17 compiler (clang on
aarch64 Linux, see below).

## Loading models

`bitino.load(model, *, stage="all", threads=1, affinity=None, variant="float", verify_hash=True, download=True)`

* **A file path** (`*.bitino`) is loaded as is.
* **A zoo id** (`bitino list --converted`) is looked up in `$BITINO_HOME/models/<id>/` (default
  `~/.cache/bitino`), where `bitino fetch` writes converted files. If it is not there:
  * models whose **weights licence is explicitly permissive** (MIT / Apache-2.0 / BSD stated for the weights: the
    rule of `bitino list --release`) are downloaded from the GitHub release, SHA-256 checked;
  * the others are not redistributed. They are built from the authors' original checkpoint by their fetch recipe,
    which needs the QONNX writers of a source checkout and the reference environments of the recipe (the source
    framework, e.g. TensorFlow/Larq or PyTorch, plus `qonnx` for the importer; `recipes.json` → `envs`). In a source
    checkout `load()` runs the recipe like `bitino fetch <id>`; from an installed wheel it raises
    `PermissionError` with the command to run. Once built, `bitino.load("<id>")` finds the file from any
    installation that shares `BITINO_HOME`.

  `bitino fetch <id>` builds every file of the model through QONNX: the network's writer turns the original
  checkpoint into the canonical QONNX file `<stem>.qonnx.onnx` (kept in the model directory, with the writer's
  record `<stem>.qonnx.json`), `tools/import_qonnx.py` imports it with no import argument into `<stem>.bitino`, and
  the derived files follow (`*_unfused`: the `--no-fold` import of the parent's canonical file, or a canonical file
  of its own). Every file's data and graph SHA-256 are compared with the recorded ones (`fetch_record.json`).
  `--legacy` also runs the legacy converter (`tools/convert_*.py`) into `<model dir>/legacy/` as a cross-check; its
  source-framework reference outputs are what `bitino validate` needs. `--int8 --calib-dir D` quantizes with your
  calibration images, writes the int8 canonical QONNX file of the quantizer output and re-imports it (the re-import
  must equal the quantizer output; it equals the published int8 file only with the recorded calibration samples).

  `download=False` never downloads or converts. `variant="int8"` / `"int8_log2"` selects the full-integer files of a
  zoo model (`model_int8_full*.bitino`).
* **stage** selects the kernel tiers: `"all"` (default: fastest exact kernels and float fusion), `"neon"` (binary
  layers on the SIMD kernels, bit-identical with the numpy executor; on x86-64 the same stage runs the AVX2 kernels),
  `"reference"` (scalar oracle), `"optimized_int8"`, `"int8_full"`, ... (`bitino.STAGES`; the stages of
  `bitino_bench`, [benchmarking.md](benchmarking.md)).
* **threads**: worker threads of the session. **affinity**: CPU ids; the calling Python thread is pinned to
  `affinity[0]` and worker *i* to `affinity[i % n]` (Linux / Android only; elsewhere pinning is reported as failed,
  see `describe()["affinity_errors"]`).

## Model

| | |
|---|---|
| `m.inputs`, `m.outputs` | `TensorSpec(name, shape, dtype)`; dim 0 is the batch the model was converted with (1 for the zoo) |
| `m.predict(x, ...)` | one array per input (positionally, by name, or a dict). Shapes: the input's own shape, the shape without the batch dimension, or N x the per-sample shape (run sample by sample). Other dtypes are converted (float64 -> float32). Returns one array (one output) or a list, each with N rows. Outputs are copies: later runs do not change them. |
| `m.predict_image(img)` | path, file object or PIL image -> the zoo model's preprocessing (`m.preprocess`) -> `predict()` |
| `m.benchmark(iterations=50, warmup=10, threads=None, stage=None, affinity=None, inputs=None, full=False)` | median latency (ms) of the native `run()` over `iterations` timed runs, as `bitino_bench` times it; other threads / stage / affinity use a temporary session; `full=True` returns all statistics, the samples and the session description |
| `m.describe()` | kernel chosen for each op (counts), plan rewrites, prepare time, activation / prepacked bytes |
| `m.manifest` | the converter's provenance manifest (source checkpoint, versions, ...) |
| `m.close()` / `with bitino.load(...) as m:` | frees the native session |

Copies: an input is copied once into the session's own buffer and an output once out of it; nothing else is copied
(an input that is already a C-contiguous array of the model's dtype is not converted). A `Model` holds one native
session and serialises calls on it; for parallel inference create one `Model` per thread (the native call releases
the GIL).

## Images

`bitino.images` re-implements the preprocessing of the zoo's image classifiers with Pillow + numpy:
`torchvision_imagenet`, `bitorch_imagenet`, `timm_bicubic_crop095`, `omz_bgr` (OpenVINO models: BGR, 0..255),
and the CIFAR normalisations (any photo is reduced to 32 x 32). Same operations in the same order as the eval-set
builders (Pillow resampling, torchvision crop rounding, float32 normalisation): on ImageNet validation images the
`torchvision_imagenet` and `omz_bgr` tensors are bit-identical to the eval-set files built with torchvision /
Accuracy Checker (Pillow 12.2). Models validated with TensorFlow preprocessing (`larq_zoo`) are not covered: pass
a preprocessed array to `predict()`.

```python
x = bitino.images.load_image("cat.jpg", "torchvision_imagenet")   # float32, 1 x 224 x 224 x 3
```

Class names are not shipped (ImageNet's are in Keras' `imagenet_class_index.json`, for example).

## Command line

The `bitino` command of the package runs models with the same runtime:

```bash
bitino run ab-bnn-imagenet-reactnet18 --image cat.jpg --labels imagenet_class_index.json --bench
bitino run model.bitino --input x.f32 --output y.f32 --stage neon
bitino bench ab-bnn-imagenet-reactnet18 --threads 4      # bitino_bench if found, else the package's runtime
```

`run --bench` (no value) also times the model; `run --bench PATH` runs through that `bitino_bench` binary instead.

## Timings on this machine

`benchmark()` measures the CPU it runs on. A cloud VM (Colab, CI runners) is a shared x86 server core: its numbers
say nothing about a phone. The phone numbers in the README are measured on the phones themselves
(`results/devices.json`, method in [benchmarking.md](benchmarking.md)). Never time under an emulator.

## How it is built

* `runtime/include/bitino/bitino_c.h`: the C API (version 2 adds session options, multi-input models, tensor
  descriptions, direct buffer access, an in-library timing loop and `bf_session_describe`; version 1 calls are
  unchanged). `cmake -DBITINO_BUILD_SHARED=ON` builds it as `libbitino_c` (exports `bf_*` only; the C++ runtime
  is linked in), which `python/bitino/_native.py` loads with `ctypes`.
* `pyproject.toml`: scikit-build-core runs CMake (only the shared library) and puts the library, the Python package
  and the zoo data (`registry/`, `results/`) into a `py3-none-<platform>` wheel.
* aarch64 Linux is built with clang (the NEON kernels' inline asm exceeds GCC's operand limit): the wheel build sets
  it; for an sdist install on aarch64, have `clang++` on `PATH`.
* Windows x64 is built with clang for MinGW-w64 ([llvm-mingw](https://github.com/mstorsjo/llvm-mingw)): the kernels
  use GNU inline asm, builtins and flags, and the DLL links its C++ runtime statically (it imports only
  `KERNEL32` and the UCRT), so it does not have to match the MSVC ABI of CPython. MSVC `cl.exe` and `clang-cl` are
  not supported.
* From a source checkout without installing a wheel: `cmake -S . -B build/host -DBITINO_BUILD_SHARED=ON &&
  cmake --build build/host --target bitino_c`; the package finds `build/*/libbitino_c.*` (or set
  `BITINO_LIB`). `pip install -e .` builds it too.

Wheels are built and tested by `.github/workflows/wheels.yml` (cibuildwheel): Linux x86-64 / aarch64 in the
manylinux_2_28 containers (aarch64 on a native ARM runner), macOS arm64, Windows x64; each wheel is tested with
`tests/python/test_python_api.py` on Python 3.10, 3.12 and 3.14.

## Releasing wheels

The `wheels` workflow builds the sdist and the wheels on every `v*` tag (and on pull requests touching the runtime or
the package) and uploads them as workflow artifacts. It never publishes on its own. To publish a release:

1. One-time setup: on PyPI, add a *trusted publisher* for this repository, workflow `wheels.yml`, environment
   `pypi`; in the repository settings, create the `pypi` environment with required reviewers.
2. Check the artifacts of the tag's run (download, `pip install` one wheel per platform, run a model).
3. Actions -> wheels -> Run workflow on the tag, with `publish` ticked. The build jobs run again; the `pypi` job
   waits for an environment reviewer, then uploads the sdist and all wheels (`pypa/gh-action-pypi-publish`).

Locally: `python -m build` (sdist + wheel for this machine), `cibuildwheel --platform linux` (manylinux wheels, needs
Docker).

## Limitations

* CPU only; no GPU / NPU. Thread pinning on Linux / Android only.
* Windows: model paths go through the C runtime's narrow-character API, so paths outside the system code page
  (non-ASCII on many systems) cannot be opened; copy the file to an ASCII path.
* The wheels' x86-64 kernels need AVX2 for the fast paths (selected at run time from the CPU features; older CPUs run
  the portable kernels).
