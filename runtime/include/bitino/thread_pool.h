// Fixed-size thread pool. A pool with `num_threads == N` uses the calling
// thread plus N-1 workers, so N is the number of CPU workers that execute
// kernel work. N == 1 runs everything inline on the caller (no workers).
//
// Waiting: workers (and the caller, while waiting for completion) first spin
// for up to `spin_us` microseconds polling an atomic generation counter, then
// block on a condition variable. Completion is counted in finished items: the
// caller works on the job itself and returns when every chunk is done, without
// waiting for workers that never claimed a chunk (asleep or not scheduled), so
// a slow wake-up costs parallelism but never latency beyond the 1-thread time. spin_us = 0 blocks immediately (lowest CPU
// use; each parallel_for pays an OS wake-up). Spinning trades CPU time for
// lower dispatch latency, which matters when a model issues ~100 short
// parallel regions per inference.
//
// Threading rules: one parallel_for at a time per pool (Session::run is not
// reentrant on the same session); kernels must partition output so that work
// items never write overlapping memory.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace bitino {

class ThreadPool {
 public:
  // `affinity`: optional list of CPU ids; thread i (caller = 0) is pinned to
  // affinity[i % size]. Empty = OS scheduling. Pinning failures are reported
  // by affinity_errors(), never fatal.
  explicit ThreadPool(int num_threads, std::vector<int> affinity = {}, int spin_us = 0);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int num_threads() const { return num_threads_; }
  // Minimum chunk = n / (chunks_per_thread * active threads) items (or the kernel's grain); weighted claims start larger.
  int chunks_per_thread() const { return chunks_per_thread_; }
  int spin_us() const { return spin_us_; }
  int affinity_errors() const { return affinity_errors_.load(); }

  using RangeFn = std::function<void(int64_t, int64_t)>;
  using SlotFn = std::function<void(int, int64_t, int64_t)>;

  // Calls fn(begin, end) over [0, n) split into contiguous chunks of at least
  // `grain` items. Blocks until all chunks completed.
  void parallel_for(int64_t n, int64_t grain, const RangeFn& fn);
  // Same, and fn(slot, begin, end) also gets the index of the thread running the chunk (0 = the caller, always
  // < num_threads()): kernels keep per-thread scratch buffers instead of a fixed task partition, so chunks can be as
  // fine as the load balance needs. No two chunks with the same slot run at the same time.
  void parallel_for_slot(int64_t n, int64_t grain, const SlotFn& fn);

  // Threads taking part in the following parallel regions: the caller and workers 1..k-1 (1 <= k <= num_threads();
  // with an affinity list, the first k entries). Session sets it per node (SessionOptions::auto_threads); call it
  // between parallel regions only.
  void set_active_threads(int k);
  int active_threads() const { return active_; }

  // Pins the *calling* thread (used for the caller slot).
  bool pin_current_thread(int cpu);
  // Start of an inference (Session::run): threads whose pinning failed retry it, the caller here and each worker
  // the next time it wakes for work. On the S22 core_ctl pauses idle cores (the X2 and one A710 when the phone is
  // idle), and sched_setaffinity to a paused core fails (EINVAL); once load has woken the cores the retry succeeds.
  void begin_run();
  // Threads not (yet) on their affinity CPU.
  int unpinned_threads() const { return unpinned_.load(); }

  // Hot mode (Session::run with SessionOptions::spin_run): while on, waiting workers and the waiting caller spin
  // without a time limit instead of blocking, so back-to-back parallel_for calls of one inference pay no OS
  // wake-up; switching it off lets idle workers block again (after spin_us). Called between parallel_for calls only.
  void set_hot(bool on) { hot_.store(on, std::memory_order_release); }
  bool hot() const { return hot_.load(std::memory_order_relaxed); }
  // parallel_for calls that went to the workers (n > grain, more than one thread) since construction.
  uint64_t dispatches() const { return generation_.load(std::memory_order_relaxed); }
  // parallel_for calls (with n > 0) since construction, including those run inline on the caller.
  uint64_t calls() const { return calls_.load(std::memory_order_relaxed); }

  // Diagnostics (BITINO_POOL_STATS=1, off by default: no clock reads then). Per thread (caller = 0): time spent in
  // chunks and items run; per pool: dispatched regions, their wall time (publish to completion, caller clock) and the
  // caller's wait after its own last chunk (the tail other threads hold the region for), inline regions.
  struct Stats {
    uint64_t regions = 0, inline_regions = 0;
    double region_us = 0, tail_us = 0, inline_us = 0;
    std::vector<double> busy_us;
    std::vector<uint64_t> items, chunks;
  };
  bool stats_enabled() const { return stats_on_; }
  // Speed of thread i relative to the caller as measured by the weighted claims (1 until measured), over all regions.
  float speed_weight(int i) const { return wrows_[0][i].load(std::memory_order_relaxed); }
  // Weighted claims: the following regions use (and update) the speed weights of `key` (Session: the node index), so
  // memory-bound and compute-bound nodes each get their own ratios between the cores; a new key starts from the
  // weights over all regions. key < 0: those overall weights. Call between parallel regions only.
  void set_weight_key(int key);
  // Weighted claims for the following regions (default on; Session tunes it per node). Off: chunks of the minimum
  // size, no skipping (the speeds are still measured). No effect when BITINO_POOL_WEIGHTED=0.
  void set_weighted(bool on) { use_weighted_ = on; }
  bool weighting_available() const { return weighted_; }
  Stats stats() const;
  void reset_stats();

 private:
  struct Job {
    std::atomic<const RangeFn*> fn{nullptr};
    std::atomic<const SlotFn*> sfn{nullptr};
    std::atomic<int64_t> n{0};
    std::atomic<int64_t> chunk{1};
    std::atomic<int> active{1};
    std::atomic<float> wsum{1.f}, wmax{1.f};  // speed weights of the active threads (weighted claims)
    std::atomic<std::atomic<float>*> w{nullptr};  // the weight row of the region
    std::atomic<bool> weighted{false};             // weighted claims in this region
    std::atomic<bool> timed{false};                // chunks timed for the speed weights in this region
  };
  void dispatch(int64_t n, int64_t grain, const RangeFn* fn, const SlotFn* sfn);
  void worker_loop(int index);
  // Claims and runs chunks of the job currently published in claim_ until none is left.
  void run_chunks(int self);
  // Spins until pred() or the spin budget expires; returns pred().
  template <typename Pred>
  bool spin_until(Pred pred) const;

  int num_threads_;
  int active_ = 1;  // set_active_threads (caller thread only)
  int spin_us_;
  int chunks_per_thread_ = 16;  // BITINO_POOL_CHUNKS: minimum chunk of weighted claims = n / (this * active)
  int equal_chunks_per_thread_ = 8;  // BITINO_POOL_EQUAL_CHUNKS: equal chunks (weighted claims off)
  int hot_serial_div_ = 4;     // BITINO_POOL_HOT_SERIAL_DIV: serial threshold = grain / this while hot
  int hot_grain_div_ = 16;     // BITINO_POOL_HOT_GRAIN_DIV: minimum chunk = grain / (serial div * this) while hot
  std::vector<int> affinity_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  std::atomic<uint64_t> generation_{0};  // dispatches so far; workers wake when it changes
  std::atomic<bool> stop_{false};
  std::atomic<bool> hot_{false};
  std::atomic<uint64_t> calls_{0};
  int sleeping_workers_ = 0;   // guarded by mu_
  bool caller_sleeping_ = false;  // guarded by mu_

  // Current job. claim_ = (generation tag << 32) | next unclaimed index; jobs_[tag & 1] holds the job of that tag.
  // A chunk is claimed by a CAS that keeps the tag, so a thread that read a stale job can never claim a chunk of
  // a newer one. The caller returns when done_ (items finished) reaches n: it never waits for a worker that has
  // not claimed a chunk (asleep, not scheduled or still waking up), only for chunks in progress.
  Job jobs_[2];
  std::atomic<uint64_t> claim_{0};
  std::atomic<int64_t> done_{0};
  std::atomic<int> affinity_errors_{0};
  std::atomic<int> unpinned_{0};
  std::atomic<uint64_t> run_epoch_{0};  // begin_run() calls
  bool caller_pinned_ = true;           // caller thread only

  struct alignas(64) ThreadStats {
    double busy_us = 0;
    uint64_t items = 0, chunks = 0;
  };
  // Heterogeneous cores (weighted claims, BITINO_POOL_WEIGHTED=0 turns them off). w[i] = speed of thread i
  // relative to the caller, measured: the chunks of a region are timed (every region of a weight row's first 16, then
  // one in time_every_), and after such a region the caller compares the items per tick of each thread with its own
  // (exponential average). A claim takes about remaining * w_i / (factor * sum of w) items (at least the minimum
  // chunk), so that every claim lasts about the same time whatever the core; a worker skips the rest of a region when
  // its chunk would end later than the threads at least as fast as itself (the slower ones skip as well) could finish
  // all that is left. Regions whose active threads all measure within sym_tol_ of each other use equal chunks.
  struct alignas(64) RegionSlot {
    std::atomic<int64_t> items{0};
    std::atomic<uint64_t> ticks{0};
  };
  bool weighted_ = true;      // BITINO_POOL_WEIGHTED: speeds measured, weighted claims possible
  bool use_weighted_ = true;  // set_weighted (caller thread only)
  bool skip_ = true;    // BITINO_POOL_SKIP
  bool inflight_ = true;  // BITINO_POOL_SKIP_INFLIGHT: the skip rule counts every item not done, not only unclaimed ones
  float skip_slack_ = 1.f;  // BITINO_POOL_SKIP_SLACK_PCT / 100 (100): skip only when the chunk would end this many times later
  bool sizing_ = true;  // BITINO_POOL_SIZING: weighted claim sizes (0: fixed chunks, skip rule only)
  int time_every_ = 8;  // BITINO_POOL_TIME_EVERY: after a row's first 16 regions, one in this many is timed
  std::vector<uint32_t> row_visits_;  // regions per weight row (caller thread only)
  float sym_tol_ = 0.2f;  // BITINO_POOL_SYM_PCT / 100 (20): equal chunks when the active threads' speeds are this close
  float factor_ = 2.f;  // BITINO_POOL_FACTOR
  std::vector<std::unique_ptr<std::atomic<float>[]>> wrows_;  // [0]: all regions, [key + 1]: set_weight_key
  int wrow_ = 0;  // current row (caller thread only)
  std::unique_ptr<std::atomic<float>[]> wfast_;  // per thread: sum of the weights of the other active threads at least as fast
  std::unique_ptr<RegionSlot[]> rslot_;
  void update_weights(int active, std::atomic<float>* row);
  std::atomic<uint64_t> weight_updates_{0};

  bool stats_on_ = false;  // BITINO_POOL_STATS
  std::vector<ThreadStats> tstats_;
  uint64_t st_regions_ = 0, st_inline_ = 0;  // caller thread only
  double st_region_us_ = 0, st_tail_us_ = 0, st_inline_us_ = 0;
};

}  // namespace bitino
