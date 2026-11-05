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
  // Target chunks (work items) per thread; kernels with their own task partition use it too (bconv2d.neon_opt).
  int chunks_per_thread() const { return chunks_per_thread_; }
  int spin_us() const { return spin_us_; }
  int affinity_errors() const { return affinity_errors_.load(); }

  // Calls fn(begin, end) over [0, n) split into contiguous chunks of at least
  // `grain` items. Blocks until all chunks completed.
  void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn);

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

 private:
  struct Job {
    std::atomic<const std::function<void(int64_t, int64_t)>*> fn{nullptr};
    std::atomic<int64_t> n{0};
    std::atomic<int64_t> chunk{1};
  };
  void worker_loop(int index);
  // Claims and runs chunks of the job currently published in claim_ until none is left.
  void run_chunks();
  // Spins until pred() or the spin budget expires; returns pred().
  template <typename Pred>
  bool spin_until(Pred pred) const;

  int num_threads_;
  int spin_us_;
  int chunks_per_thread_ = 8;  // BITINO_POOL_CHUNKS
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
};

}  // namespace bitino
