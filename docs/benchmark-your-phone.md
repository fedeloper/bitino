# Benchmark your phone

Bitino's speed tables come from a handful of phones. One command measures it on yours, and the result can be added
to the community table in `results/community/`.

## Run it

You need:

* an Android phone (arm64-v8a, or 32-bit armeabi-v7a with your own build, see below) with
  [USB debugging](https://developer.android.com/studio/debug/dev-options) enabled, connected by USB;
* on the computer: `adb` ([Android platform-tools](https://developer.android.com/tools/releases/platform-tools)),
  `bash`, `awk`, `curl` (or `wget`) and `sha256sum` (or `shasum`). Linux and macOS have them; on Windows use WSL or
  Git Bash.

No root, no app install, nothing to build.

```bash
git clone https://github.com/fedeloper/bitino && cd bitino
scripts/bench_my_phone.sh
```

Or without cloning:

```bash
curl -fsSLO https://raw.githubusercontent.com/fedeloper/bitino/main/scripts/bench_my_phone.sh
bash bench_my_phone.sh
```

It takes 5–10 minutes and downloads about 40 MB (cached in `~/.cache/bitino/bench`). Leave the phone alone while it
runs: screen on or off is fine (the screen state is recorded), but do not use it, and preferably do not charge it
from a wall charger (the USB connection itself is fine).

| option | effect |
|---|---|
| `--quick` | one model file (ReActNet-18, about a minute) |
| `--all` | all 22 redistributable model files of the release (about 150 MB, 30+ minutes) |
| `--models A,B` | chosen files, by release asset name, e.g. `omz-resnet50-binary-0001_int8_full` |
| `--serial ID` | the phone to use when several are connected (first column of `adb devices`; not recorded) |
| `--binary PATH` | your own `bitino_bench` build instead of the release binary |
| `--local DIR` | model files from `DIR/<model-id>/model.bitino` (or `DIR/<asset>.bitino`) instead of downloading them |
| `--repeats N`, `--iterations N`, `--warmup N` | defaults 3, 50, 10 |
| `--no-check` | skip the output check |
| `--out DIR` | where the result files go (default: current directory) |

**32-bit phones** (Android reports `armeabi-v7a`, e.g. the Redmi 9A): the release has no 32-bit binary yet. Build one
with the Android NDK, `ANDROID_ABI=armeabi-v7a scripts/build_android.sh build/android-armv7`, and run
`scripts/bench_my_phone.sh --binary build/android-armv7/bitino_bench`.

## What it measures

Default set, from the latest [GitHub release](https://github.com/fedeloper/bitino/releases) (MIT / Apache-2.0
weights; the files are checked against the release's `SHA256SUMS` / `MODELS-SHA256SUMS`):

| model | files |
|---|---|
| ReActNet-18, A&B BNN (ImageNet) | `model.bitino` (float stage `all`), `model_int8_full.bitino` (stage `int8_full`) |
| ResNet-50 binary, Intel Open Model Zoo (ImageNet) | `model.bitino`, `model_int8_full.bitino` |

For each file:

1. **Output check.** One inference on a fixed synthetic input (`pattern-v1`: the float32 values 0.5, −1.25, 2, −0.75,
   1.5, −0.25, 1 repeated) at the file's exact stage (`neon` for the float file, `int8_full` for the integer file),
   4 threads. The SHA-256 of the outputs must equal the reference in the script. These stages are bit-exact on every
   platform (x86-64, arm64 and 32-bit ARM give identical bytes; [numerical correctness](numerical-correctness.md)), so
   a mismatch means the runtime computed a wrong result on your phone: please report it.
2. **Timing.** `bitino_bench` with
   * `t1_fast`: 1 thread pinned to the fastest core (the highest-numbered CPU of the cpufreq policy with the highest
     maximum clock);
   * `t4_fast`: 4 threads pinned to the 4 fastest cores, fastest first.

   Each configuration runs 3 times, round-robin over all files and configurations (so drift spreads evenly), each
   time 10 warm-up and 50 timed inferences at batch 1 on a fixed input. The reported number is the median of the 3
   per-repeat medians. [Method](benchmarking.md).
3. **Clock state.** Before and after every run the script reads every cpufreq policy's ceiling
   (`scaling_max_freq` vs `cpuinfo_max_freq`) and Android's thermal status; while the phone is throttling (thermal
   status > 0, or a ceiling below the highest seen during the run) it waits, at most 120 s. Each result is labelled
   like the maintainers' runs: `clean` (every policy at its maximum), `vendor-policy` (a ceiling below the maximum
   with no thermal event, e.g. a screen-off power policy), `thermal-throttled`.

The script never changes governors, clocks or settings, never reboots, installs nothing and does not use the screen
or the camera. It writes only to `/data/local/tmp/bitino-community` on the phone, and deletes that directory at the
end (also when interrupted with Ctrl-C).

## Submit

The script writes two files:

* `bitino-bench-<phone model>-<date>.json`: everything (device block, clock states, output checks, every raw
  `bitino_bench` report with all 50 latencies);
* `bitino-bench-<phone model>-<date>.md`: a short summary table.

Open an issue with the **[Speed on my phone](https://github.com/fedeloper/bitino/issues/new/choose)** template,
paste the `.md` and attach the `.json` (if GitHub refuses the file type, zip it). A maintainer validates it with
`tools/ingest_community.py` (physical phone, consistent raw data, published model files, plausible numbers, matching
outputs) and adds it to `results/community/`, labelled `community`, with the issue number. Run it again after a
system update or on a second phone: every submission is kept.

## Privacy

The JSON contains only:

* the phone model and its product codename (`ro.product.model`, `.manufacturer`, `.brand`, `.device`), the SoC and
  board platform, the hardware name, the Android version, API level and security patch date, the kernel version,
  the ABI;
* the CPU topology: CPU part numbers, cpufreq policies with their maximum clocks, ceilings and governors, the CPU
  features the runtime detected;
* the measurement conditions: clock ceilings and Android thermal status before and after each run, CPU thermal-zone
  temperatures, battery level and temperature, whether it is powered over USB / AC / wireless, and whether the screen
  is on;
* the measurements, the runtime version with the compiler and flags it was built with, the SHA-256 of the binary and
  the model files, the output checksums, and the date and time.

It does **not** contain the adb serial number, IMEI or any other hardware identifier, accounts, phone number,
the build fingerprint (removed from the runtime's reports), the computer's name, user name or paths, or anything
from the phone's storage. You can read the whole file before attaching it; it is plain JSON.
`tools/ingest_community.py` refuses files with an identifying field that carries a value.
