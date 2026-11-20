# Threads: the pool, its defaults, and what the S22 does to them

`SessionOptions::num_threads = N` runs every kernel's `parallel_for` on the calling thread plus N − 1 pool workers
(`runtime/include/bitino/thread_pool.h`, `runtime/src/thread_pool.cc`). Threading never changes outputs: a kernel
partitions its output into items, every item is computed the same way whatever thread or chunk runs it, and the
emulated references (4 threads) equal the 1-thread outputs. This page describes the pool as of 2026-11-04 and the
phone measurements behind its defaults.

## How a parallel region runs

1. **Publish.** The caller stores the job (function, item count, chunk size) in one of two job slots and then the
   claim word `claim_ = (generation tag << 32) | next unclaimed item`. Workers spinning on the generation counter see
   it within a few polls; sleeping workers are woken with one `notify_all`.
2. **Claim.** Every thread, the caller included, claims chunks with a compare-and-swap that keeps the tag. A thread
   that read a stale job (it woke up late) fails its CAS and never runs a chunk of a newer job.
3. **Complete.** Each finished chunk adds its item count to `done_`; the caller returns when `done_ == n`.
   **The caller never waits for a worker that has not claimed a chunk** (asleep, not scheduled, migrating, still
   waking up): a slow wake-up costs parallelism, not latency beyond the 1-thread time of that region. Before
   2026-11-04 the caller waited until every worker had checked in, so one descheduled worker stalled every region:
   ReActNet-18 log2 at 8 threads with `--spin-run` took 172–190 ms (1 thread: 10 ms).

Waiting (workers between regions, the caller for chunks in progress) spins on `yield` for `spin_us`, or without a
limit while the pool is **hot**, then blocks on a condition variable. Every 1024 polls a spinning thread calls
`sched_yield()`, so a spinning thread that shares a core with the thread holding a chunk gives the core away.

## Defaults (and the knobs)

| setting | default | meaning |
|---|---|---|
| `SessionOptions::spin_run` | **true** (was false) | hot during `Session::run`: no OS wake-up between the regions of one inference; workers block again after the run (after `spin_us`). `bitino_bench --no-spin-run` turns it off. |
| chunks per thread | **16** (8 until 2026-11-19, 4 before 2026-11-04) | `BITINO_POOL_CHUNKS`: the minimum chunk is n / (16 × active threads) items (or the kernel's grain); weighted claims take larger ones first (below) |
| weighted claims | **on** (per node: on or off, tuned) | `BITINO_POOL_WEIGHTED=0` turns them off everywhere (equal chunks, no speed measurement); `BITINO_POOL_SIZING=0` keeps equal chunk sizes with the skip rule, `BITINO_POOL_SKIP=0` the sizes without it, `BITINO_POOL_SKIP_INFLIGHT=0` counts only unclaimed items in the skip rule, `BITINO_POOL_SKIP_SLACK_PCT` (100) skips only when the chunk would end that much later, `BITINO_POOL_FACTOR` (2) the claim factor |
| `SessionOptions::auto_threads` | **true** | per-node thread counts tuned in the first runs of a session (below); `BITINO_AUTO_THREADS=0` or `bitino_bench --no-auto-threads` turns it off |
| `BITINO_POOL_STATS` | 0 | 1: per-thread busy time, items and chunks, region wall time and the caller's tail wait; `bitino_bench` writes them to `config.pool_stats` (per run) |
| serial threshold while hot | grain / **4** | `BITINO_POOL_HOT_SERIAL_DIV`: a kernel's grain (items below which it runs on the caller) was sized for an OS wake-up per dispatch |
| minimum chunk while hot | grain / **64** | `BITINO_POOL_HOT_GRAIN_DIV` (16, applied after the serial divisor) |
| `SessionOptions::affinity` | none | pins thread i to `affinity[i % size]`; a failed pin is retried once per `Session::run` (`ThreadPool::begin_run`) until it succeeds |

The knobs are environment variables so that a device A/B needs no rebuild; the defaults are the best settings of the
sweep below.

## What the S22 does

* **Paused cores.** core_ctl pauses idle cores (`/sys/devices/system/cpu/cpu7/core_ctl/global_state`: cpu6 and cpu7
  "Paused: 1" on an idle phone; the X2 cluster has `min_cpus 0`). `sched_setaffinity` to a paused core fails with
  EINVAL, so `--affinity 7` usually did **not** pin at session creation (`affinity_errors` 1–2 in most runs of the
  day); the thread stayed unpinned and the scheduler moved the busy thread to the X2 once core_ctl woke it (1-thread
  runs "on the X2" were at X2 speed, 10–11 ms, an A710 gives 21 ms). The pool now retries a failed pin at the start of
  each run (`unpinned_threads_after` in the bench JSON reports threads still unpinned after the runs).
* **Wake-ups and frequency.** Without spinning every region pays the wake-up of the workers, and cores that sleep
  between regions also lose frequency under walt. ReActNet-18 log2, 4 threads pinned 4–7: 17–19 ms without spinning
  vs 10.3–10.9 ms with `--spin-run` (memory profile, X2 capped 1.61 GHz); later the same day t4 without spinning
  ranged 5.4–20 ms between repeats while `--spin-run` stayed at 4.9–5.8 ms.
* **1 X2 + 3 A710.** An A710 thread is 2–2.9× slower than the X2 (21.0 vs 10.0 ms at caps 1.56 / 1.84 GHz; 34.4 vs
  11.7 ms at 1.32 / 1.61 GHz). With dynamic chunks the X2 takes more of them, but a region with few chunks ends when
  the last A710 chunk does: Bi-ViT DeiT-Tiny's 197-token layers had 3 LayerNorm chunks (speedup 1.14 at 4 threads),
  BDense row tiles of 40 rows (5 tiles, speedup 1.49), Add nodes below the serial threshold (0.96). More chunks, a
  lower serial threshold while hot and BDense row tiles of at most rows / (4 × threads) fixed most of it (below).

## Measurements (S22 over USB, throttled; medians of 30 iterations, interleaved)

Chunking sweep, t4 unpinned, `--spin-run` binary of 3eb99e3 before its defaults (X2 1.61 / A710 1.32 GHz), ms:

| C chunks/thread : D hot grain divisor (serial divisor 1) | Bi-ViT DeiT-Tiny | BNext-Tiny | QuickNet-Small | ReActNet-18 log2 |
|---|---|---|---|---|
| 4 : 1 (old) | 16.97, 17.03 | 24.81, 24.80 | 5.05, 4.93 | 5.77, 5.79 |
| 8 : 1 | 16.98, 17.24 | 24.76, 24.71 | 4.98, 5.01 | 5.66, 5.72 |
| 4 : 4 | 15.29, 15.22 | 22.93, 22.94 | 5.19, 4.72 | 5.73, 5.74 |
| 8 : 16 | 14.78, 14.89 | 22.19, 22.19 | 4.58, 4.54 | 5.60, 5.58 |
| 8 : 64 / 8 : 100000 | 14.85, 14.90 | 22.18–22.34 | 4.54–4.70 | – |
| 16 : 16 | 14.89, 14.94 | 22.08, 22.11 | 4.55, 4.62 | 5.59, 5.62 |

Serial threshold divisor (C 8, D 16): Bi-ViT 14.76 / 14.93 (S 1) → 12.67 / 12.90 (S 4) → 15.29 / 12.76 (S 16); the
CNNs unchanged. Final defaults (8 chunks and 8 `bconv2d.neon_opt` tasks per thread) against the same binary with
`BITINO_POOL_CHUNKS=4` (4 of each): BNext-Tiny 22.08 / 22.19 → 21.18 / 21.37, ReActNet-18 log2 5.71 / 5.76 →
5.43 / 5.53, QuickNet-Small 4.63 / 4.64 → 4.51 / 4.54, Bi-ViT 13.06 / 13.04 → 12.96 / 12.77.

Thread-pool barrier alone (threads-only binary vs master fccc6c1, same kernels, ReActNet-18 log2, X2 1.84 / A710
1.56 GHz, 4 repeats): unchanged where nothing stalls (t4 spin-run 4.95–5.20 vs 4.94–5.10 ms, t4 default 5.40–5.44 vs
5.41–5.91), 8 threads with spin 6.6–14.4 ms vs 8.5–10.0 (and 172–190 ms in an earlier pair of runs).

Before / after for the five models of the speed-all slice: the development log (not published)
("Phone").

## Heterogeneous cores (2026-11-19, OnePlus Open)

The SM8550 has 1 Cortex-X3, 2 A715, 2 A710 and 3 A510; on the binary kernels the X3 is ~2x an A715 and ~5x an A510,
so 2, 4 and 8 threads (pinned fastest first: 7,3,4,5,6,0,1,2) can reach at most ~1.5x, ~2.4x and ~3.5x. The thread
suite of 2026-11-19 (123 models) gave medians 1.30x / 1.92x / 2.19x (float) and 8 threads slower than 4 in 38 models.
A per-layer profile with `BITINO_POOL_STATS=1` showed where it went:

* **8 threads: A510 tails.** With equal chunks of n / 64 items, an A510 that claims one of the last chunks holds the
  region for 5 chunk-times of the X3: the caller's wait after its own last chunk was 30–55 % of the parallel time
  (bnas-d int8_full 6.0 of 11.7 ms, Bi-ViT DeiT-Tiny 6.5 of 11.3 ms, ReActNet-18 1.7 of 5.4 ms).
* **Small layers.** CIFAR ResNet-20's binary convs at 16x16 and 8x8 pixels had 4 and 2 work items (64-pixel tiles,
  32-channel chunks): at 4 threads two of the cores had nothing to do.
* **Coarse task partitions.** `bconv2d.neon_opt` split its items into 8 tasks per thread up front; the last task on
  an A715 bounded the region.
* **Nodes that do not scale.** Some nodes are slower with more threads (bhvit I8Gather 1.8 ms at 1 thread, 3.3 ms
  at 8; LayerNorm and ChannelAffine of Bi-ViT at 8 threads).

What changed:

1. **Weighted claims** (`ThreadPool::run_chunks`). Every chunk is timed (`cntvct_el0` on AArch64); after each region
   the caller updates each thread's speed relative to its own (items per tick, exponential average). A claim takes
   about `remaining × w_i / (2 × Σw)` items (weighted factoring, at least the minimum chunk), so every claim lasts
   about as long on every core and the chunks shrink towards the end of the region. A worker skips the rest of a
   region when its chunk would end after the threads at least as fast as itself could finish everything not done yet
   (`(n − done) / Σ w_faster`: the chunks they hold count too, but not before one chunk of the fastest thread); the
   caller always claims. Weights are kept per node (`ThreadPool::set_weight_key`; a new node starts from the weights
   over all regions), since memory-bound and compute-bound nodes see different ratios between the cores. Only the
   partition changes, never what an item computes.
2. **Slots instead of tasks.** `ThreadPool::parallel_for_slot` passes the index of the running thread, so
   `bconv2d.neon_opt` (and `qbconv2d.neon`) keep one patch buffer per thread and let the pool claim their items
   directly instead of a fixed task split.
3. **Smaller tiles on small layers.** With more than one thread `bconv2d.neon_opt` halves its pixel tile (down to 16
   pixels) until there are ~8 tiles per thread; the 1-thread plan is unchanged.
4. **Per-node thread counts** (`SessionOptions::auto_threads`, `Session::tune_run`). After the runs of a session
   have taken 300 ms (frequencies and paused cores settle; at most 2000 runs), it times every node with 1, 2, 4, …,
   N active threads (`ThreadPool::set_active_threads`: the caller and workers 1..k−1, i.e. the first k cores of the
   affinity list), each count > 1 with and without the weighted claims (`ThreadPool::set_weighted`; in Bi-ViT's short
   regions at 4 threads equal chunks were faster), 3–16 rounds, and gives each node the fastest candidate when it
   beats N threads with weighted claims by 3 % and 0.5 µs. The choices stay only if whole runs with them are faster
   than with the default (4–32 interleaved pairs, medians).
   `bitino_bench` extends its warmup until the tuning is over (`config.warmup_runs`, `config.auto_threads`), so
   timed iterations never include tuning runs. `Session::set_node_threads` sets counts directly (tests, A/B).
5. **Pre-warm for pinning** (`bitino_bench --prewarm-ms`, default 1000 with `--affinity`). The nine FINN /
   Brevitas MLPs (~1 µs per inference on the X3) never loaded the phone long enough for core control to wake the
   paused X3: every repeat stayed unpinned (at 16 µs on another core) and was dropped. One busy thread per requested
   core retries the pin every 0.5 ms until all succeed; the session is created right after. Measured: 0 pin
   failures instead of 61 per run, 1.04 µs instead of 16 µs (brevitas-bnn-pynq-tfc-1w1a, 1 thread on cpu 7).

Outputs: `tests/cpp/test_core.cc` (`thread_pool_slots_active_threads_and_weights`,
`session_thread_counts_identical_outputs`: 1/2/3/4/8 threads through tuning and with random per-node counts give the
1-thread bytes), the kernel tests at 1/3/4/8 threads, and `scripts/thread_identity.sh` (models on their eval sets at
1/2/4/8 threads, after tuning, against a master binary's 1-thread output, host or phone).

### Measured (2026-11-19, `scripts/device_threads_bench.sh`, master vs this branch interleaved by model)

Same script, cores and settings as the 123-model suite (3 repeats × 50 iterations, 10 warmup; this branch's bench
also warms up until the per-node tuning is over). Raw data: the development log (not published)
(master 281f453c) and `-mt` (this branch). Speedups use master's 1-thread median as the common reference (the
1-thread code path is unchanged; small CIFAR models vary by ±10 % between 1-thread runs of the suite script).

OnePlus Open (CPH2551, cores 7,3,4,5,6,0,1,2), 17 models picked from the worst cases of the suite plus representatives.
Median / minimum speedup over 15 models: the two FINN/Brevitas TFC MLPs are left out, master's runs on them stayed
unpinned (16–17 µs on another core instead of ~1 µs on the X3; the "16x" in their rows is the pre-warm, not threads):

| stage | 2 threads | 4 threads | 8 threads |
|---|---|---|---|
| all (float) | median 1.23 → 1.30, min 1.05 → 1.07 | median 1.71 → 1.74, min 1.20 → 1.23 | median 1.55 → 2.15, min 0.93 → 1.27 |
| int8_full | median 1.24 → 1.27, min 1.04 → 1.08 | median 1.61 → 1.79, min 1.16 → 1.28 | median 1.62 → 2.08, min 0.86 → 1.30 |

8 threads slower than 4: master in 9 of 17 float and 8 of 15 int8_full models, this branch only in the two ~1 µs
TFC MLPs.
Slower than master: Bi-ViT DeiT-Tiny at 4 threads (float 1.71 → 1.67, int8_full 1.77 → 1.75) and
bhvit-small-somefp float at 4 threads (1.99 → 1.98).

float model, stage `all`; speedup = master's 1-thread median / median at N threads (base → new):

| model | t1 ms base / new | 2t | 4t | 8t | 8t ms base / new |
|---|---|---|---|---|---|
| ab-bnn-imagenet-reactnet18 | 10.803 / 10.777 | 1.30 -> 1.36 | 1.96 -> 2.04 | 2.32 -> 2.75 | 4.650 / 3.923 |
| adabin-resnet20-cifar10 | 0.878 / 0.989 | 1.05 -> 1.07 | 1.20 -> 1.23 | 1.00 -> 1.27 | 0.882 / 0.692 |
| bhvit-small-somefp | 46.071 / 46.137 | 1.36 -> 1.38 | 1.99 -> 1.98 | 1.57 -> 2.41 | 29.377 / 19.095 |
| bi-vit-deit-tiny | 13.775 / 13.760 | 1.23 -> 1.24 | 1.71 -> 1.67 | 1.47 -> 1.88 | 9.380 / 7.346 |
| biper-resnet20-cifar10 | 0.453 / 0.448 | 1.07 -> 1.11 | 1.27 -> 1.37 | 0.94 -> 1.46 | 0.482 / 0.311 |
| bnas-d-imagenet | 26.063 / 26.150 | 1.26 -> 1.30 | 1.84 -> 1.94 | 1.94 -> 2.46 | 13.404 / 10.608 |
| bnas-e-imagenet | 32.544 / 32.651 | 1.28 -> 1.31 | 1.78 -> 1.95 | 1.96 -> 2.45 | 16.595 / 13.310 |
| bnext-tiny | 24.111 / 23.943 | 1.33 -> 1.34 | 1.94 -> 2.01 | 2.05 -> 2.51 | 11.745 / 9.611 |
| brevitas-bnn-pynq-cnv-1w1a | 0.300 / 0.339 | 1.20 -> 1.34 | 1.67 -> 1.74 | 1.55 -> 2.15 | 0.193 / 0.139 |
| brevitas-bnn-pynq-tfc-1w1a | 0.017 / 0.001 | 0.90 -> 16.00 | 1.09 -> 15.61 | 1.10 -> 14.54 | 0.015 / 0.001 |
| finn-examples-bnn-pynq-tfc-w1a1 | 0.016 / 0.001 | 0.99 -> 15.55 | 0.98 -> 15.55 | 0.98 -> 14.15 | 0.017 / 0.001 |
| larq-zoo-quicknet-small | 4.642 / 4.661 | 1.25 -> 1.32 | 1.87 -> 1.93 | 2.14 -> 2.51 | 2.171 / 1.849 |
| larq-zoo-quicknet | 7.348 / 7.352 | 1.28 -> 1.34 | 1.88 -> 2.00 | 2.22 -> 2.62 | 3.304 / 2.802 |
| rbnn-resnet20-cifar10 | 0.525 / 0.489 | 1.12 -> 1.15 | 1.28 -> 1.38 | 0.97 -> 1.43 | 0.543 / 0.366 |
| recu-resnet20-cifar10 | 0.455 / 0.502 | 1.06 -> 1.12 | 1.23 -> 1.38 | 0.93 -> 1.47 | 0.490 / 0.309 |
| shuaiz-bireal20-cifar10 | 0.423 / 0.460 | 1.08 -> 1.16 | 1.27 -> 1.49 | 1.08 -> 1.57 | 0.393 / 0.270 |
| siman-resnet20-cifar10 | 0.528 / 0.544 | 1.12 -> 1.16 | 1.29 -> 1.38 | 1.07 -> 1.44 | 0.493 / 0.366 |


full-integer model, stage `int8_full`; speedup = master's 1-thread median / median at N threads (base → new):

| model | t1 ms base / new | 2t | 4t | 8t | 8t ms base / new |
|---|---|---|---|---|---|
| ab-bnn-imagenet-reactnet18 | 7.688 / 7.683 | 1.34 -> 1.37 | 2.02 -> 2.11 | 2.54 -> 2.87 | 3.025 / 2.679 |
| adabin-resnet20-cifar10 | 0.897 / 0.904 | 1.09 -> 1.11 | 1.30 -> 1.34 | 1.05 -> 1.37 | 0.850 / 0.655 |
| bhvit-small-somefp | 52.339 / 52.987 | 1.28 -> 1.27 | 1.74 -> 1.79 | 1.62 -> 2.13 | 32.400 / 24.618 |
| bi-vit-deit-tiny | 12.292 / 12.308 | 1.28 -> 1.30 | 1.77 -> 1.75 | 1.83 -> 1.92 | 6.720 / 6.386 |
| biper-resnet20-cifar10 | 0.398 / 0.400 | 1.05 -> 1.10 | 1.26 -> 1.41 | 0.86 -> 1.49 | 0.461 / 0.267 |
| bnas-d-imagenet | 16.047 / 16.047 | 1.19 -> 1.27 | 1.61 -> 1.83 | 1.60 -> 2.08 | 10.037 / 7.725 |
| bnas-e-imagenet | 23.311 / 23.389 | 1.24 -> 1.27 | 1.60 -> 1.81 | 1.69 -> 2.11 | 13.760 / 11.060 |
| bnext-tiny | 22.025 / 22.045 | 1.29 -> 1.31 | 1.84 -> 1.95 | 2.23 -> 2.46 | 9.888 / 8.960 |
| brevitas-bnn-pynq-cnv-1w1a | 0.600 / 0.327 | 1.70 -> 2.81 | 2.66 -> 3.54 | 3.44 -> 4.19 | 0.175 / 0.143 |
| larq-zoo-quicknet-small | 4.685 / 4.701 | 1.30 -> 1.35 | 1.98 -> 2.05 | 2.44 -> 2.74 | 1.918 / 1.708 |
| larq-zoo-quicknet | 7.661 / 7.661 | 1.30 -> 1.36 | 1.95 -> 2.06 | 2.38 -> 2.67 | 3.222 / 2.868 |
| rbnn-resnet20-cifar10 | 0.452 / 0.429 | 1.13 -> 1.17 | 1.31 -> 1.48 | 1.08 -> 1.55 | 0.420 / 0.291 |
| recu-resnet20-cifar10 | 0.421 / 0.448 | 1.13 -> 1.23 | 1.43 -> 1.66 | 1.05 -> 1.76 | 0.400 / 0.239 |
| shuaiz-bireal20-cifar10 | 0.401 / 0.427 | 1.12 -> 1.28 | 1.39 -> 1.67 | 1.16 -> 1.80 | 0.346 / 0.223 |
| siman-resnet20-cifar10 | 0.560 / 0.604 | 1.04 -> 1.08 | 1.16 -> 1.28 | 1.02 -> 1.30 | 0.549 / 0.429 |

Moto e13 (2 × Cortex-A75 + 6 × A55, cores 7,6,5,4,3,2,1,0), 8 models; medians over 7 (TFC excluded):

| stage | 2 threads | 4 threads | 8 threads |
|---|---|---|---|
| all (float) | median 1.71 → 1.69, min 1.36 → 1.35 | median 2.20 → 2.28, min 1.11 → 1.48 | median 2.71 → 2.76, min 1.12 → 1.60 |
| int8_full | median 1.63 → 1.67, min 1.36 → 1.48 | median 2.18 → 2.20, min 1.10 → 1.66 | median 2.81 → 2.74, min 1.13 → 1.86 |

Slower than master on the Moto: Bi-ViT DeiT-Tiny int8_full at 8 threads (19.46 → 19.99 ms), adabin float at
2 threads (1.38 → 1.35), Bi-ViT float at 2 threads (1.71 → 1.69).

float model, stage `all`; speedup = master's 1-thread median / median at N threads (base → new):

| model | t1 ms base / new | 2t | 4t | 8t | 8t ms base / new |
|---|---|---|---|---|---|
| ab-bnn-imagenet-reactnet18 | 60.940 / 60.991 | 1.90 -> 1.90 | 2.75 -> 2.83 | 4.44 -> 4.52 | 13.714 / 13.475 |
| adabin-resnet20-cifar10 | 3.979 / 3.995 | 1.38 -> 1.35 | 1.38 -> 1.48 | 1.43 -> 1.60 | 2.782 / 2.486 |
| bi-vit-deit-tiny | 67.168 / 67.202 | 1.71 -> 1.69 | 2.20 -> 2.28 | 2.71 -> 2.76 | 24.784 / 24.378 |
| bnas-d-imagenet | 139.133 / 138.857 | 1.90 -> 1.92 | 2.57 -> 2.72 | 3.66 -> 3.98 | 37.972 / 34.950 |
| brevitas-bnn-pynq-tfc-1w1a | 0.003 / 0.004 | 0.96 -> 0.96 | 0.97 -> 0.97 | 0.99 -> 0.97 | 0.004 / 0.004 |
| larq-zoo-quicknet | 38.992 / 38.616 | 1.95 -> 1.95 | 2.84 -> 2.86 | 4.26 -> 4.34 | 9.154 / 8.979 |
| rbnn-resnet20-cifar10 | 2.000 / 1.994 | 1.36 -> 1.46 | 1.11 -> 1.67 | 1.12 -> 1.87 | 1.793 / 1.069 |
| recu-resnet20-cifar10 | 1.839 / 1.844 | 1.38 -> 1.48 | 1.17 -> 1.71 | 1.17 -> 1.98 | 1.578 / 0.928 |


full-integer model, stage `int8_full`; speedup = master's 1-thread median / median at N threads (base → new):

| model | t1 ms base / new | 2t | 4t | 8t | 8t ms base / new |
|---|---|---|---|---|---|
| ab-bnn-imagenet-reactnet18 | 44.797 / 44.923 | 1.89 -> 1.90 | 2.85 -> 2.94 | 4.61 -> 4.67 | 9.711 / 9.596 |
| adabin-resnet20-cifar10 | 4.133 / 4.161 | 1.36 -> 1.49 | 1.44 -> 1.70 | 1.57 -> 1.86 | 2.636 / 2.226 |
| bi-vit-deit-tiny | 54.686 / 54.760 | 1.63 -> 1.64 | 2.18 -> 2.20 | 2.81 -> 2.74 | 19.461 / 19.989 |
| bnas-d-imagenet | 92.267 / 92.255 | 1.81 -> 1.84 | 2.64 -> 2.75 | 3.94 -> 4.17 | 23.434 / 22.133 |
| larq-zoo-quicknet | 36.354 / 36.400 | 1.85 -> 1.87 | 2.72 -> 2.79 | 4.26 -> 4.25 | 8.530 / 8.551 |
| rbnn-resnet20-cifar10 | 1.896 / 1.889 | 1.37 -> 1.48 | 1.15 -> 1.66 | 1.13 -> 1.97 | 1.673 / 0.962 |
| recu-resnet20-cifar10 | 1.637 / 1.641 | 1.59 -> 1.67 | 1.10 -> 2.01 | 1.37 -> 2.61 | 1.194 / 0.628 |

1-thread check (OnePlus, cpu 7, 6 interleaved rounds × 200 iterations): adabin-resnet20 0.876 → 0.882 ms,
recu-resnet20 int8_full 0.397 → 0.398, shuaiz-bireal20 0.408 → 0.404, siman-resnet20 0.499 → 0.494; Moto (cpu 7,
5 rounds): rbnn 1.991 → 1.988, QuickNet 38.96 → 38.83, QuickNet int8_full 36.40 → 36.40, bnas-a 50.10 → 50.09 ms.

## Not done

* A static work split weighted for 1 X2 + 3 A710 (e.g. 40/20/20/20): the dynamic chunks already give the X2 its
  share, and threads are not reliably on the cores they were meant for (paused cores, unpinned migration), so a
  static ratio would be wrong whenever the placement is.
* Pinning by default: unpinned and pinned 7,4,5,6 measured the same (within 1 %) once spinning was on, and pinning
  to paused cores fails anyway.
* 8 threads: the A510s are much slower than the X2; 8 threads were slower than 4 in every run (ReActNet-18 log2,
  threads-only A/B: 6.4–8.0 ms vs 4.9–5.0 ms). (2026-11-19: fixed by the weighted claims, above.)
* Serial nodes of small CIFAR models: RBNN's per-tensor mean centering (ReduceMean over the whole 32x32x16 tensor,
  then AddBroadcast) and the 16K-element elementwise nodes stay on the caller (one 16384-element block each): about a
  third of ResNet-20's time at 4 threads. Folding `x - mean >= 0` into the binary conv's threshold (`x >= mean`,
  exact in IEEE arithmetic) would remove the AddBroadcast pass; not done (a graph rewrite, not threading).
