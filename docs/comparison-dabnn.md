# Bitino vs daBNN: 1-thread latency on phones, same Bi-Real Net-18 models

This page compares the latency of Bitino (this repository) with daBNN (JDAI-CV/dabnn, "daBNN: A Super Fast Inference
Framework for Binary Neural Networks on ARM devices", ACM MM 2019) on daBNN's own published Bi-Real Net-18 models.
Both runtimes run the same weights on the same input, 1 thread, batch 1, pinned to the same core by the same
launcher. Every number was measured on a real phone. No emulated timing appears anywhere.

Reproduce it with `scripts/h2h_dabnn.sh` (steps `src`, `build`, `convert`, `run`, `eval`, `report`). It uses:

* `tools/h2h_dabnn.py`: runs the device benchmarks and the 128-image correctness check, writes the report.
* `tools/dabnn_h2h.cc`: a 170-line timing and output driver linked to daBNN's unmodified library (`libdabnn.a`).
* `tools/h2h_run.c`: the on-device launcher every runtime runs under (the same one as in
  [comparison-lce-tflite.md](comparison-lce-tflite.md)).

The raw data are in `reports/h2h_dabnn/2026-11-19/`:

* One directory per phone and core (`<phone>-cpu<N>/`), holding:
  * `plan.json`: device, binaries and model files with SHA-256, policy, reference clock cap, caps at start and end.
  * `raw/<model>__<runtime>__r<repeat>.json`: the command, launcher report (CPU placement histogram, allowed mask,
    peak RSS), timing (every timed run for daBNN; the full `bitino_bench` JSON next to it for Bitino), cpufreq of
    every CPU before and after the timed loop, cap and thermal state before and after the round, round order.
  * `outputs/`: each runtime's output on the timed input (eval image 0).
* `eval-<phone>/eval.json`: the 128-image correctness runs (commands, placement, SHA-256 of the outputs; the
  128 × 1000 logit files themselves, 512 KB each, stay in the work directory, `.cache/dabnn-h2h/eval-outputs/`).
* `sensitivity-march/`: daBNN rebuilt with Bitino's `-march` (`scripts/h2h_dabnn.sh sens`; see
  [Results](#results)).
* `summary.json` and `summary.md`: the aggregates.

## Contents

1. [Results](#results)
2. [Correctness](#correctness)
3. [What runs: versions, builds, models](#what-runs-versions-builds-models)
4. [Measurement method](#measurement-method)
5. [Differences in setup, caveats](#differences-in-setup-caveats)
6. [What was not done](#what-was-not-done)

## Results

**Summary** (2026-11-19; 5 interleaved rounds per configuration, no throttled round, every run on the pinned
core):

* **arm64 phones (Cortex-X3, A715, A510, A75):** Bitino (stage `all`, float layers in fp32) is faster than daBNN on
  every core and both models, by **1.60–2.05×** (geometric mean **1.81×**). With the full-integer model
  (`int8_full`) the margin is 1.96–2.50× (geometric mean 2.24×), but that model is not numerically equivalent (see
  [Correctness](#correctness)).
* **32-bit Redmi 9A (Cortex-A53, armeabi-v7a):** Bitino is 6.5–7.5× faster. daBNN's 32-bit path has no direct
  binary 3×3 convolution (its direct kernels are AArch64 assembly), so this mostly measures daBNN's portable
  fallback; daBNN's paper and README report arm64 only.
* **daBNN's own harness agrees with our driver**: `benchmark_single_model` is within −0.8 % … +1.9 % of the driver's
  number in every configuration, so the daBNN figures are not an artefact of the driver.
* **Compiler flags do not explain the gap**: daBNN rebuilt with Bitino's `-march=armv8.2-a+dotprod` runs within
  ±1.6 % of the default build on the X3 and the A75 (table below).
* **Peak RSS:** daBNN 26–34 MB, Bitino `all` 17–20 MB, `int8_full` 11–14 MB.

One-sentence summary:

> On daBNN's own Bi-Real Net-18 models and weights (byte-identical to daBNN's released `.dab` files), at 1 thread
> and batch 1 on the same core, Bitino is 1.6–2.1× faster than daBNN on four arm64 cores (Cortex-X3, A715, A510,
> A75; geometric mean 1.81×) with the same float semantics, and 6.5–7.5× faster on a 32-bit Cortex-A53.

**Latency, median ms** (per run: mean of 50 timed runs; per configuration: median of 5 rounds). Ratio = daBNN ÷
Bitino (> 1: Bitino faster). Clock: the pinned core's frequency during the timed loops (read by the daBNN driver
right before and after its loop, the same value in every round) / the core's maximum.

| phone | core (clock / max GHz) | model | daBNN | Bitino `all` | daBNN ÷ Bitino `all` | Bitino `int8_full` | daBNN ÷ `int8_full` | daBNN own harness |
|---|---|---|---:|---:|---:|---:|---:|---:|
| OnePlus Open | cpu7 Cortex-X3 (2.23 / 3.19) | Bi-Real-18 | 21.19 | **10.33** | **2.05** | 8.51 | 2.49 | 21.29 |
| OnePlus Open | cpu7 Cortex-X3 (2.23 / 3.19) | Bi-Real-18 stem | 15.34 | **7.97** | **1.92** | 6.70 | 2.29 | 15.41 |
| OnePlus Open | cpu4 Cortex-A715 (1.79 / 2.80) | Bi-Real-18 | 39.55 | **24.33** | **1.63** | 18.99 | 2.08 | 39.88 |
| OnePlus Open | cpu4 Cortex-A715 (1.79 / 2.80) | Bi-Real-18 stem | 30.08 | **18.75** | **1.60** | 15.37 | 1.96 | 30.51 |
| OnePlus Open | cpu0 Cortex-A510 (2.02 / 2.02) | Bi-Real-18 | 109.44 | **54.19** | **2.02** | 43.77 | 2.50 | 108.52 |
| OnePlus Open | cpu0 Cortex-A510 (2.02 / 2.02) | Bi-Real-18 stem | 82.90 | **40.57** | **2.04** | 33.87 | 2.45 | 84.49 |
| Moto e13 | cpu7 Cortex-A75 (1.61 / 1.61) | Bi-Real-18 | 101.95 | **59.90** | **1.70** | 48.28 | 2.11 | 103.92 |
| Moto e13 | cpu7 Cortex-A75 (1.61 / 1.61) | Bi-Real-18 stem | 72.16 | **45.12** | **1.60** | 34.82 | 2.07 | 72.07 |
| Redmi 9A (32-bit) | cpu0 Cortex-A53 (2.00 / 2.00) | Bi-Real-18 | 826.87 | **128.21** | **6.45** | 112.39 | 7.36 | 841.58 |
| Redmi 9A (32-bit) | cpu0 Cortex-A53 (2.00 / 2.00) | Bi-Real-18 stem | 773.73 | **102.65** | **7.54** | 80.14 | 9.66 | 786.65 |
| geometric mean, arm64 rows | | | | | **1.81** | | 2.24 | |

Run-to-run spread (max ÷ min of the 5 per-run means) is ≤ 1.036 for both drivers in every configuration (≤ 1.09 for
daBNN's own harness); the medians of the per-run medians are within 0.4 % of the medians of the means
(`summary.md`). The OnePlus clocks are the vendor's screen-off caps (X3 2.23 of 3.19 GHz, A715 1.79 of 2.80 GHz),
in force whenever the screen is off; they apply equally to both runtimes and were the same in every round (the
gate's reference cap). For reference, daBNN's README reports 61.3 ms (Bi-Real-18) and 43.2 ms (stem) on a Google
Pixel 1 (Kryo, 1 thread); on the Cortex-A75 at 1.61 GHz we measure 102 / 72 ms.

**Sensitivity: daBNN built with Bitino's `-march`** (`sensitivity-march/`; the same driver and `libdabnn` compiled
with `-march=armv8.2-a+dotprod`; 3 interleaved rounds; median ms):

| phone / core | model | daBNN (default, ARMv8.0-A) | daBNN (`armv8.2-a+dotprod`) | Bitino `all` |
|---|---|---:|---:|---:|
| OnePlus Open cpu7 (X3) | Bi-Real-18 | 21.26 | 21.25 | 10.33 |
| OnePlus Open cpu7 (X3) | Bi-Real-18 stem | 15.43 | 15.44 | 7.97 |
| Moto e13 cpu7 (A75) | Bi-Real-18 | 101.33 | 102.92 | 59.60 |
| Moto e13 cpu7 (A75) | Bi-Real-18 stem | 72.47 | 72.28 | 45.00 |

## Correctness

All 128 images of the eval set (ImageNet val subset, see [numerical-correctness.md](numerical-correctness.md))
went through daBNN and Bitino on each phone (untimed, pinned to the benchmark core; outputs = the 1000 logits).
Reference = the original ONNX evaluated in float64 with daBNN's implicit binarization made explicit
(`reference_logits_f64.f64`, from the model conversion).

| phone | model | top-1 accuracy daBNN / Bitino `all` / `int8_full` / fp64 ref | top-1 agreement with fp64 ref: daBNN / `all` / `int8_full` | Bitino `all` vs daBNN: top-1 agreement, max \|Δ logit\|, median per-sample max \|Δ\| | `int8_full` vs daBNN: top-1 agreement, max \|Δ\| |
|---|---|---|---|---|---|
| OnePlus Open, Moto e13 (arm64; byte-identical outputs on both) | Bi-Real-18 | 58.59 / 58.59 / 54.69 / 58.59 % | 0.961 / 0.969 / 0.781 | 0.961, 3.17, 7.6e-5 | 0.766, 6.06 |
| OnePlus Open, Moto e13 (arm64; byte-identical outputs on both) | Bi-Real-18 stem | 57.03 / 57.81 / 55.47 / 58.59 % | 0.977 / 0.984 / 0.703 | 0.961, 3.26, 7.6e-5 | 0.719, 6.59 |
| Redmi 9A (armeabi-v7a) | Bi-Real-18 | 57.03 / 58.59 / 54.69 / 58.59 % | 0.945 / 0.969 / 0.781 | 0.945, 2.65, 7.7e-5 | 0.750, 6.06 |
| Redmi 9A (armeabi-v7a) | Bi-Real-18 stem | 58.59 / 58.59 / 55.47 / 58.59 % | 0.984 / 0.992 / 0.703 | 0.977, 3.26, 7.6e-5 | 0.719, 6.86 |

* **daBNN and Bitino `all` compute the same network.** On the typical sample their logits differ by ~8e-5 (median
  per-sample max |Δ|: float32 summation order). The maximum differences (2.7–3.3) come from samples where a
  binarized activation sits within float rounding of 0 and flips in one runtime but not in the other; the flip then
  propagates through the following binary layers. Both runtimes have such samples relative to the float64 reference
  (samples deviating by more than 1e-3: daBNN 26–39 of 128, Bitino `all` 17–25), and both keep a top-1 agreement
  with the float64 reference of 0.95–0.99, the same order as onnxruntime float32 vs float64 on these models
  (0.977 / 0.984). Bitino's `all` outputs are bit-identical to its `optimized` stage, whose deviations were traced
  flip by flip in [numerical-correctness.md](numerical-correctness.md); daBNN's were not traced.
* **daBNN on the phones = daBNN under emulation:** the arm64 device outputs are bit-identical to the earlier
  qemu-aarch64 run of the same daBNN sources (`artifacts/<model>/dabnn_runtime_logits_emulated.f32`); the 32-bit
  build takes other code paths and differs from it (top-1 agreement 0.953 / 0.984).
* **`int8_full` is a different model**, not a faster implementation of the same arithmetic: its non-binary layers
  are statically quantized to int8. On these 128 images it changes 22–30 % of the top-1 predictions and costs
  3–4 points of top-1; on 5,000 ImageNet validation images it costs 0.6 / 0.7 points (52.6 → 52.1 %, 53.5 →
  52.7 %) with 73 / 72 % prediction agreement. Its
  timing is reported for completeness; the like-for-like comparison is daBNN vs Bitino `all`.
* **The timed input (eval image 0) of Bi-Real-18 is one of the flip samples**: daBNN returns the reference class
  (134) and Bitino `all` class 86 (`outputs/`, max |Δ| 1.95 vs daBNN). For the stem model both agree with the
  reference to 8e-5. The input is the same for both runtimes, and neither runtime's work depends on which class
  wins.

## What runs: versions, builds, models

### daBNN

* Source: JDAI-CV/dabnn master `d93aa950788319d92ade545eb7fe7ac6f1589fd1` (2019-11-12, the last commit), with its
  submodules at the commits that tree pins (`git ls-tree HEAD third_party/`), fetched as GitHub tarballs and checked
  against the SHA-256 values in `scripts/h2h_dabnn.sh`:

  | submodule | commit | used for |
  |---|---|---|
  | google/glog | `de82428e520f0a1a71089043140c4bfc3b44e554` | runtime (logging) |
  | google/flatbuffers | `9e7e8cbe9f675123dd41b7c62868acad39188cae` | runtime (`.dab` format) |
  | google/benchmark | `f0901417c89d123474e6b91365029cfe32cf89dc` | daBNN's benchmark targets |
  | daquexian/onnx | `9f70c118910f7174f65dac84c90d46d608d9cfd2` | `onnx2bnn` |
  | protocolbuffers/protobuf | `6973c3a5041636c1d8dc5f7f6c8c1f3c15bc63d6` | `onnx2bnn` (daBNN's packed protobuf) |

* Built out of tree from a copy of the source. Changes to the copy, none of which touches daBNN's code or its
  code-generation flags:
  * `-Werror` removed from daBNN's `treat_warnings_as_errors` and from google/benchmark's release flags (NDK r27c
    clang 18 warns where the 2019 compilers did not);
  * `tools/dabnn_h2h.cc` added as one more target of daBNN's `binaries/` directory, linked to `libdabnn.a`;
  * glog configured with `HAVE_EXECINFO_H=OFF` (no `backtrace()` below Android API 33) and the NDK's static
    `libunwind.a` (glog's `find_library` otherwise picks the host one).
* Android builds: NDK r27c (clang 18.0.3), `ANDROID_PLATFORM=android-30`, `ANDROID_STL=c++_static`,
  `CMAKE_BUILD_TYPE=Release`. daBNN's own `CMAKE_CXX_FLAGS_RELEASE` is `-O3 -DNDEBUG`; the NDK adds its defaults.
  * arm64-v8a: no `-march` (NDK default ARMv8.0-A), as daBNN's documented build. daBNN's binary kernels are
    hand-written AArch64 assembly, independent of `-march`.
  * armeabi-v7a: NDK default `-march=armv7-a -mthumb` with NEON (daBNN's CI used "armeabi-v7a with NEON").
  * Targets: `dabnn_h2h` (timing driver), `benchmark_single_model` (daBNN's own single-model benchmark, the binary
    its `tools/quick_benchmark.py` runs) and `dabnn_benchmark` (built, not run: it reads hard-coded
    `/data/local/tmp/model_imagenet*.dab` paths).
* `onnx2bnn` for the host (x86-64, GCC 15.2, packed protobuf, `-include stdexcept` because `common/StrKeyMap.h`
  relies on a transitive include that GCC 15's libstdc++ no longer provides).

### Bitino

* This repository at the commit of this page's data (`a7c17ddd` + the h2h scripts), `bitino_version` 0.1.0,
  `scripts/build_android.sh`: NDK r27c clang 18.0.3, API 30, `c++_static`, Release:
  `-O3 -DNDEBUG -ffp-contract=off`, arm64 `-march=armv8.2-a+dotprod` (i8mm kernels compiled separately and selected
  at run time), armeabi-v7a NDK default (NEON).
* Stages: `all` (the default, best kernels per node; float layers on the optimized float kernels) on
  `model.bitino`, and `int8_full` on `model_int8_full.bitino` (every non-binary layer on int8 tensors, see
  [int8.md](int8.md)).

### Models: the same weights

| model | daBNN ONNX (SHA-256) | `.dab` | Bitino `model.bitino` | Bitino `model_int8_full.bitino` |
|---|---|---|---|---|
| `dabnn-bireal18` (Bi-Real Net-18) | `birealnet18.onnx` `9c3864bd…e988` | `3b43f3c0…d249`, 4,198,320 B | `5f1a21b8…7cc4`, 4,205,568 B | `c279d2d7…6679`, 2,164,480 B |
| `dabnn-bireal18-stem` (with stem module) | `birealnet18stem.onnx` `b25d4e33…ae2f` | `6a3aa841…40c3`, 4,202,880 B | `a6f8a7c6…ea55`, 4,210,432 B | `d3c43508…86eb`, 2,183,104 B |

* The `.dab` files were produced here with daBNN's documented conversion (`onnx2bnn <onnx> <dab> --aggressive`,
  as in daBNN's `ci/download_and_convert_models.sh`). They are **byte-identical** to the `.dab` files daBNN publishes
  next to the ONNX files (same SHA-256), so daBNN runs its own released models.
* The Bitino models were converted from the same ONNX files (`tools/convert_dabnn.py`) and validated against them
  ([numerical-correctness.md](numerical-correctness.md)): 16 binary 3×3 convolutions with input sign `x ≥ 0 → +1`
  and padding −1, as daBNN does, float stem / downsample / classifier.
* Weights are daBNN's; they are not redistributed here (no model files in this repository's reports).

## Measurement method

The protocol is the one of [comparison-lce-tflite.md](comparison-lce-tflite.md):

* **Launcher.** Every runtime runs under `h2h_run --cpus <N> -- <binary> ...`. The launcher sets the affinity with
  `sched_setaffinity()` in the child before `exec` (none of the binaries pins itself), reads the mask back, samples
  every 25 ms the CPU each thread last ran on, and reports peak RSS (`ru_maxrss`). A run counts only if the mask
  read back is `{N}` and every placement sample of every thread is on CPU N; otherwise it is repeated (at most 3
  attempts, all recorded). Additionally, `bitino_bench`'s `affinity_errors` must be 0 and the daBNN driver
  reports its own `sched_getaffinity()` mask, which must be `{N}`. **Every recorded run passed** (placement
  histograms in the raw JSON; 0 off-CPU samples in all configurations).
* **Same policy for both.** 1 thread (daBNN is single-threaded; Bitino `--threads 1`), batch 1, input = eval
  image 0 of the 128-image eval set (ImageNet val, Resize 256 / CenterCrop 224 / ImageNet mean-std, float32 NHWC),
  read from the same file. 10 untimed runs (1 run whose output is written + 9 warm-up), then 50 timed runs. Timing
  covers only the network call: `bnn::Net::run` for daBNN, `Session::run` for Bitino (model load, preparation and
  the input copy are outside; both runtimes read the input from a buffer filled once).
* **Statistic.** Per run, the mean of the 50 timed runs (the median is recorded as well); per configuration, the
  median over 5 rounds of the per-run means.
* **daBNN's own harness** (`benchmark_single_model`, google/benchmark defaults: adaptive iteration count, ≥ 0.5 s,
  mean real time per iteration, input = an uninitialised stack buffer as upstream) runs in every round as a
  cross-check of the driver. It is not the primary number.
* **Interleaving.** A round = one model, every runtime once, in an order rotated by (round + model index). The
  device lock (`/tmp/bitino-device-<serial>.lock`, shared with every other job on that phone) is held for the
  whole round.
* **Clock gate.** Before every round the gate waits (up to 600 s) until the Android thermal status is 0 and the
  `scaling_max_freq` of the pinned CPU's policy equals the reference cap (its value when the invocation started);
  caps, thermal status and skin temperature are recorded before and after the round, and the drivers record the
  cpufreq of every CPU before and after their timed loop. A second pass (`--redo-throttled`) re-measures rounds
  that were off the reference cap. No governor was changed; nothing ran as root.

## Differences in setup, caveats

* **Same core, same clock, one at a time.** The runtimes never ran simultaneously: each round ran all runtimes back
  to back on one core under the device lock. Another head-to-head job (LCE/TFLite on the OnePlus A715) shared the
  OnePlus during these runs; the lock serialized its rounds with ours (waits in each record's
  `round.lock_wait_s`). No round was off the reference clock cap or had a thermal status above 0, so the
  `--redo-throttled` pass had nothing to redo.
* **Clock records.** The daBNN driver reads the cpufreq right before and right after its timed loop: the pinned
  core was at the reference cap in every daBNN run. `bitino_bench` takes its snapshots before loading the model
  and after writing its report, outside the loop, so its values show the governor ramping up or down (e.g. A75
  1.04–1.61 GHz "after"); its round-to-round spread (≤ 1.8 %) shows the loop itself ran at a steady clock.
* **Compiler flags differ.** daBNN is built as documented (arm64: no `-march`, i.e. ARMv8.0-A), Bitino with its
  `-march=armv8.2-a+dotprod`. daBNN's binary kernels are hand-written AArch64 assembly, and daBNN rebuilt with
  Bitino's flags is within ±1.6 % of the default build (sensitivity table above). The kernels Bitino selected are
  in each raw record (`bitino.kernels`; stage `all`: `bconv2d.neon_opt` for the 16 binary layers, `*.float_opt`
  for the rest).
* **Toolchain.** daBNN dates from 2019; here both runtimes are compiled with NDK r27c clang 18. daBNN's hot binary
  loops are assembly, so the compiler version mostly affects its float code (stem, pooling, BN).
* **Network call only.** Both drivers time only the inference call with the input already in place (daBNN: input
  pointer handed to `Net::run`; Bitino: input set before the loop). Load and preparation are excluded for both
  (`load_ms` / `prepare_ms` are recorded).
* **daBNN harness.** `benchmark_single_model` chooses its own iteration count (6–45 per run on the arm64 cores, 1 on
  the A53, where one run exceeds its 0.5 s minimum time) and feeds an uninitialised buffer; it is a cross-check
  only.
* **32-bit.** On armeabi-v7a daBNN's `BinConv` has no direct 3×3 path (`direct_conv_compatible()` is AArch64-only),
  so it runs im2col + its AArch32 NEON binary GEMM micro-kernel. Bitino's 32-bit binary kernels are AArch32 builds
  of its AArch64 ones. The 6.5–7.5× ratio therefore says more about daBNN's 32-bit fallback than about its main
  (arm64) design point.
* **Preprocessing** is not part of either measurement; daBNN documents none, the Bi-Real ImageNet transform is
  assumed (see the model records).
* **Batch 1, 1 thread only.** daBNN has no multi-threading, so no multi-thread comparison is possible.

## What was not done

* **daBNN's `dabnn_benchmark` target** (the micro-benchmarks and the Bi-Real-18 entries behind its README table) was
  built but not run: its network benchmarks read hard-coded `/data/local/tmp/model_imagenet*.dab` paths, and this
  study keeps all its files in its own device directory. `benchmark_single_model` runs the same `Net::run` on the
  same files.
* **Moto e13 little cores (Cortex-A55)** and the OnePlus Cortex-A710 cores were not measured (not requested; the
  same binaries run there).
* **Flips inside daBNN** were not traced to layers (that needs per-layer dumps from daBNN); only logit-level
  agreement is reported.
* **No uncapped OnePlus clocks:** the X3 / A715 ran under the vendor's screen-off caps; lifting them needs the
  screen on, which this study does not touch.
