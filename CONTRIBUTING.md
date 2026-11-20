# Contributing to Bitino

Thank you for helping. Bitino is small and its numbers are only as good as the hardware and the networks behind
them, so the three most useful contributions are below, most useful first.

## Speed on your phone

The benchmark tables cover four phones. Every other Android phone you add makes them more useful, and all you need is
the phone, a USB cable and `adb`; no root, no build.

1. Enable USB debugging on the phone and connect it (`adb devices` lists it).
2. Run `scripts/bench_my_phone.sh` from a clone of this repository. It downloads the benchmark binary and a few
   models from the latest release, runs them pinned to the fastest core at 1 thread and at 4 threads, and writes one
   JSON file plus a short summary. It changes no setting on the phone and removes its files afterwards. 32-bit (armeabi-v7a) phones
   need a local build for now (the guide shows how). Details: [docs/benchmark-your-phone.md](docs/benchmark-your-phone.md).
3. Open an issue with the **Speed on my phone** template, paste the summary and attach the JSON
   (`bitino-bench-<phone model>-<date>.json`).

The JSON contains the phone model, SoC, Android version, CPU clocks and temperatures, and the timings. It does not
contain serial numbers or account information.

Tips for clean numbers: let the phone cool down first, keep it unplugged from a charger that heats it, close other
apps. A run under a thermal limit is still welcome; it is labelled as such.

## A binary network that is not in the zoo

[docs/wanted-models.md](docs/wanted-models.md) lists networks with public weights that are not converted yet, and
which ones are good first conversions. Something missing from that list too? Open an **Add a model** issue with
the paper, code and checkpoint links.

To convert one:
- **Brevitas, QONNX or FINN-ONNX models** import directly: [docs/converting-a-new-bnn.md](docs/converting-a-new-bnn.md).
- **Other frameworks:** start from the closest `tools/convert_*.py` ([docs/adding-a-model.md](docs/adding-a-model.md)).

A conversion is accepted when it passes the checks of [docs/numerical-correctness.md](docs/numerical-correctness.md):
top-1 agreement with a float64 run of the source model of at least 98 % on the check set, every deviation traced to a
rounding-level binarization flip, and the exact stage bit-identical with the NumPy executor. The pull request should
include the registry entry (with the licence of the *weights*), the converter, the fetch recipe and
`conversion_report.json`. We redistribute converted files only under MIT, Apache-2.0 or BSD weights; anything else
ships as a fetch recipe that downloads the authors' files.

## Bugs, wrong numbers, questions

- **A model disagrees with its paper or its framework:** open a **Bug** issue with the model id, the command and
  what you expected. Wrong numbers are bugs.
- **A crash or wrong output:** include the output of
  `python -c "import bitino; print(bitino.__version__, bitino.cpu_info())"`, the platform, and the smallest input
  that reproduces it.
- **Questions and ideas:** [Discussions](https://github.com/fedeloper/bitino/discussions).

## Code

```bash
pip install -e ".[test]"
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/host
ctest --test-dir build/host --output-on-failure && pytest -q tests/python
```

Rules that keep the numbers trustworthy:
- **Exactness.** The `neon` stage must stay bit-identical with the reference executor, and `int8_full` with its
  NumPy oracle. A new kernel comes with a test against the reference kernel on random shapes.
- **No emulated timings.** Speed claims come from real hardware, with the device, clocks and thread setting
  recorded. Emulators (qemu) are for correctness only.
- **Untrusted downloads.** Checkpoints are loaded with `torch.load(weights_only=True)`, converters run with
  `python -I`, and every download is hash-checked.

By contributing you agree that your contribution is licensed under Apache-2.0, like the rest of the code.
