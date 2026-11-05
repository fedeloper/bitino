# Benchmarking methodology

All performance numbers are measured on the physical Galaxy S22 with `bitino_bench`. Host numbers
are never reported as device numbers. The x86-64 host is a separate, explicitly labelled target
(see "Host benchmarks" below); emulated (qemu) timings are never reported.

## What is timed

* **Timed:** `Session::run()` only, i.e. steady-state inference at batch 1 on a fixed input.
* **Reported separately:** model load and hash check (`load_ms`), session preparation including kernel
  selection, weight prepacking and memory planning (`session_prepare_ms`), and memory (`VmRSS` at each
  phase, `VmHWM`, activation arena bytes, prepacked weight bytes).
* Correctness passes (`--inputs/--outputs`) are never timed.
* Each report records the device fingerprint, CPU features, compiler and flags, kernel chosen per
  node, thread and affinity settings, per-CPU frequency before and after, CPU/SoC thermal zones
  before and after, and every per-iteration latency.

## Stages

Every stage runs the **same binary** (except stage 1, which uses the `-O0` build) and the same model
file, restricted to a set of kernel tiers (`--stage`):

| # | stage | binary | tiers / model |
|---|---|---|---|
| 1 | scalar reference | `-O0` | reference, unfused model |
| 2 | compiler-optimized baseline | `-O3` | reference |
| 3 | bit packing | `-O3` | + packed_naive |
| 4 | optimized binary scalar | | + binary_scalar |
| 5 | NEON binary | | + neon |
| 6 | INT8 reference (float layers) | | + int8_reference |
| 7 | ARM-optimized INT8 (SDOT/I8MM) | | + int8_neon |
| 8 | optimized fp32 kernels | | + float_opt |
| 9 | operator fusion | | fused model file; plus the session's bit-exact plan-time rewrites at every stage that allows `float_opt` (`--no-fuse` disables them, `config.plan_rewrites` in the JSON counts them; `docs/operators.md` § Plan-time rewrites) |
| 10 | memory | | `--no-reuse-memory` vs default |
| 11–12 | thread scheduling, final | | 1/2/4/8 threads, affinity variants |
| — | `int8_full_reference` | | reference + int8_full_reference; **full-integer model** `model_int8_full.bitino` (portable oracle kernels of the I8 operators) |
| — | `int8_full` | | `optimized` + int8_full_reference + int8_full_opt + int8_full_neon; full-integer model: every non-binary operator on int8 tensors (`docs/int8.md` § Full-integer execution). Compared with `optimized_int8` (same binary, float model, Conv2D/Dense/MatMul in dynamic INT8) |

Stages that do not apply to a model (e.g. INT8 when it lowers accuracy below the acceptance criterion)
are still measured where possible, and marked.

## Discipline

* `tools/benchmark.py` runs configurations round-robin over repeats (default 3 × 50 iterations after 10
  warmup runs), so thermal drift spreads over all configurations.
* Cool-down on the device is adaptive (default since 2026-11-02, `--cooldown-mode adaptive`). Before every
  run the script waits until the hottest CPU thermal zone (`cpu-*`, `cpuss-*`) is within `--cool-margin`
  (2 °C) of the idle baseline. The wait is at least `--cool-min` (3 s) and at most `--cool-max` (90 s);
  a timeout is recorded, not fatal. The idle baseline is measured once: the phone idles until the CPU
  temperature stops falling (< 0.5 °C over 30 s, at most 300 s). It is cached in
  `.cache/bench_idle_baseline.json` for `--baseline-max-age` hours (24), so every model of a suite run
  uses the same target. Every raw JSON gets a `cooldown` block with the wait, the temperature at the end
  of the previous run and at the start of this one, and the timeout flag; summaries report the median
  and maximum wait, the start-temperature range and the timeouts. A fixed sleep wastes most of its time
  on light models and can be too short after heavy ones. `--cooldown-mode fixed --cooldown S` restores
  the earlier fixed sleep.
* Thermal frequency caps (since 2026-11-02). The vendor thermal engine lowers `scaling_max_freq` when
  the skin gets warm. On the S22 while charging over USB it held the X2 at 1.84 of 2.99 GHz, the A710
  at 1.56 of 2.50 GHz and the A510 at 1.36 of 1.79 GHz, with the CPU zones near idle temperature, so the
  cool-down above cannot see it. Before every run `tools/benchmark.py` waits until every cpufreq
  policy is back at `cpuinfo_max_freq` (at most `--cap-max` s, default 900; 0 records only). It reads
  the caps, the Android thermal status and the SKIN temperature again after the run, and stores them
  in a `throttle` block of the raw JSON. A run capped while it ran is repeated after the cap lifts (up
  to 3 attempts); a run started after a gate timeout is kept and flagged. Summaries count the
  throttled runs and mark the configurations with "⚠ throttled k/n". `scripts/device_final_suite.sh`
  logs the caps and the charging state before the benchmark phase and warns when the phone is
  charging. Benchmark with a charged phone off the charger, in a cool place. Runs found to be invalid
  later are listed in the development log (not published) with the reason; `tools/generate_report.py`
  leaves them out (the data stays in the repository) and marks earlier runs without a cap record with †.
* Single-thread stage comparisons are pinned to the Cortex-X2 (cpu7). Thread-scaling runs include
  unpinned and per-cluster pinned variants. Affinity uses `sched_setaffinity`, needs no root, and
  any failures are counted in the report.
* `tools/report_benchmarks.py` reports the median of per-repeat medians, the min–max of those medians,
  pooled mean/p90/p95, and speedups vs the previous stage and vs stage 1.
* An unavailable measurement is recorded as unavailable. Kernel microbenchmarks
  (`bitino_microbench`) and full-model results are reported separately: a microbenchmark win
  is not evidence of an end-to-end win.

## Native x86-64 host measurements

The x86-64 kernels (`binary_avx2` tier) are measured natively on the development host
(i9-14900K), and those timings are real, but they are numbers for that CPU, not for the phone.
They live in their own ledger, which records the CPU, the
governor and turbo state (`/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor`,
`intel_pstate/no_turbo`) and the host load. Single-thread runs are pinned to a P-core
(`--affinity 2`; CPUs 0–15 are P-core hyperthreads, 16–31 E-cores). Multi-thread runs use 8
threads unpinned. Configurations run round-robin over repeats, and the ledger reports the median
of the per-repeat medians with their min–max. On x86 the stage `neon` selects the AVX2 kernels,
so stage `binary_scalar` vs `neon` on the same binary compares `binary_scalar` with `binary_avx2`.
Emulated (qemu) timings are still never reported.

## Host benchmarks (x86-64)

`tools/benchmark.py --target host` runs `build/host/bitino_bench` locally with the same plans
(`stages`, `threads`, `quick`, `custom`), the same raw JSON schema and the same
`tools/report_benchmarks.py` summaries as the device path (which is unchanged and stays the
default):

```bash
python3 tools/benchmark.py --target host --plan quick --stage optimized \
    --model-dir artifacts/larq-zoo-quicknet-small --out reports/benchmarks_host/larq-zoo-quicknet-small/quick1
python3 tools/report_benchmarks.py reports/benchmarks_host/larq-zoo-quicknet-small/quick1 --baseline t1_pcore
```

* Machine: Intel i9-14900K (Raptor Lake; AVX2, FMA, BMI2, AVX-VNNI, no AVX-512). CPUs 0-15 are the
  8 P-cores' hyperthreads (siblings 0/1, 2/3, ...), CPUs 16-31 the E-cores. Override with
  `--host-pin`, `--host-pcores`, `--host-ecores`.
* Single-thread configurations are pinned to a P-core (CPU 2, `--host-pin`). The `quick` plan runs
  1 thread pinned, 8 threads on the 8 physical P-cores (one hyperthread each), 8 threads unpinned,
  and the same pinned configurations at `optimized_int8`. The `threads` plan adds 1/2/4/8/16
  unpinned, 1/2/4/8 P-cores, 16 P-core hyperthreads, 1/8/16 E-cores, all 24 cores, all 32 threads
  and spin-wait variants. Stage names are the platform-neutral ones (`neon`, `int8_neon` select the
  AVX2 tiers on x86-64).
* Recorded: `plan.json` gets a `host` block (CPU model, cpufreq driver, governor, EPP,
  `scaling_max_freq`, intel_pstate `no_turbo`, sibling list, load average); every raw JSON gets
  `host_env` (load average and governor / current frequency of the pinned CPUs before and after the
  run) next to `bitino_bench`'s own per-CPU frequency and `x86_pkg_temp` snapshots. Summaries are
  headed "Host (not the device)".
* Default cooldown is 2 s (`--cooldown`). The host is shared: record the load average, interleave
  A/B binaries (round-robin repeats) and compare medians of per-repeat medians, as on the device.
  

## Instruction counts under emulation

Emulated timings are meaningless. qemu-aarch64 translates each ARM instruction to x86 in
software, and its cost depends on the instruction type. Float SIMD goes through softfloat
helpers and is 45–80× slower than native; loads, stores and adds are 5–7× slower. QEMU also has
no pipeline or cache model, so it cannot produce cycles either: the guest cycle counter just
reports host time.

What QEMU *can* measure exactly is the number of ARM instructions executed.
`tools/insn_count.py` does this with a small TCG plugin, `tools/qemu_plugins/bf_symcount.c`,
which counts instructions per ELF symbol:

```bash
scripts/build_qemu_plugins.sh        # once: QEMU 10.2.1 with plugins (distro qemu-user has none)
python3 tools/insn_count.py --all --jobs 10 --out reports/insn_counts
```

* One steady-state inference is isolated by subtracting a (warmup 1 + 1 iteration) run from a
  (warmup 1 + 2 iterations) run, symbol by symbol. This cancels model loading, SHA-256
  verification, input generation and session setup. The residual of negative per-symbol
  differences is reported and is ~1e-6 of the total.
* Symbols are demangled and grouped by kernel class (`BConv2DNeonOpt`, `Conv2DGemm`, ...), so
  the counts break down per kernel.
* Counts are for 1 thread and are deterministic and independent of host load. With `--cpu`
  they can be taken for a QEMU CPU model without i8mm or dotprod, to exercise the fallbacks.

Use them to compare the work done by kernel variants, to find where instructions go, and, together
with device latency, to estimate IPC = instructions / (latency × clock). Example: QuickNet-Small
executes 69.9 M instructions per inference. At 8.5 ms on the 2.99 GHz Cortex-X2 that is about
25 M cycles, an IPC of about 2.7 (assuming the core stayed at maximum frequency). They are not a
latency proxy across instruction types: one NEON FMLA does 4 multiply-adds, and a cache miss
costs ~100 cycles while counting as one load.
