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
| chunks per thread | **8** (was 4) | `BITINO_POOL_CHUNKS`; `bconv2d.neon_opt` uses the same number of tasks per thread |
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

## Not done

* A static work split weighted for 1 X2 + 3 A710 (e.g. 40/20/20/20): the dynamic chunks already give the X2 its
  share, and threads are not reliably on the cores they were meant for (paused cores, unpinned migration), so a
  static ratio would be wrong whenever the placement is.
* Pinning by default: unpinned and pinned 7,4,5,6 measured the same (within 1 %) once spinning was on, and pinning
  to paused cores fails anyway.
* 8 threads: the A510s are much slower than the X2; 8 threads were slower than 4 in every run (ReActNet-18 log2,
  threads-only A/B: 6.4–8.0 ms vs 4.9–5.0 ms).
