#include "bitino/thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#if defined(__linux__)
#include <sched.h>
#endif

#include "bitino/error.h"

namespace bitino {
namespace {

inline void cpu_relax() {
#if defined(__aarch64__)
  __asm__ __volatile__("yield");
#elif defined(__arm__) && (__ARM_ARCH >= 7)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

}  // namespace

namespace {
int env_int(const char* name, int def, int lo, int hi) {
  const char* e = std::getenv(name);
  if (!e || !*e) return def;
  return std::min(hi, std::max(lo, std::atoi(e)));
}
}  // namespace

ThreadPool::ThreadPool(int num_threads, std::vector<int> affinity, int spin_us)
    : num_threads_(num_threads), spin_us_(spin_us), affinity_(std::move(affinity)) {
  // Tuning knobs (A/B on a device without rebuilding; defaults from the S22 sweep of 2026-11-04, docs/threads.md).
  chunks_per_thread_ = env_int("BITINO_POOL_CHUNKS", chunks_per_thread_, 1, 64);
  hot_grain_div_ = env_int("BITINO_POOL_HOT_GRAIN_DIV", hot_grain_div_, 1, 1 << 20);
  hot_serial_div_ = env_int("BITINO_POOL_HOT_SERIAL_DIV", hot_serial_div_, 1, 1 << 20);
  BF_CHECK(num_threads >= 1, "num_threads must be >= 1, got ", num_threads);
  BF_CHECK(spin_us >= 0, "spin_us must be >= 0");
  if (!affinity_.empty()) {
    caller_pinned_ = pin_current_thread(affinity_[0]);
    if (!caller_pinned_) unpinned_.fetch_add(1);
  }
  for (int i = 1; i < num_threads_; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_.store(true);
  }
  cv_start_.notify_all();
  for (auto& t : workers_) t.join();
}

bool ThreadPool::pin_current_thread(int cpu) {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) == 0) return true;
#else
  (void)cpu;
#endif
  affinity_errors_.fetch_add(1);
  return false;
}

template <typename Pred>
bool ThreadPool::spin_until(Pred pred) const {
  if (pred()) return true;
  if (spin_us_ <= 0 && !hot_.load(std::memory_order_acquire)) return false;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us_);
  for (int i = 1;; ++i) {
    cpu_relax();
    if (pred()) return true;
    if ((i & 63) == 0) {
      if (!hot_.load(std::memory_order_acquire) && std::chrono::steady_clock::now() >= deadline) return pred();
      // Every 1024 polls give the core away if another thread is runnable on it: with more threads than free
      // cores (other processes, or the scheduler packing two of ours on one core) a pure spin would keep the
      // thread that holds a chunk off its core.
      if ((i & 1023) == 0) std::this_thread::yield();
    }
  }
}

void ThreadPool::run_chunks() {
  uint64_t v = claim_.load();
  while (true) {
    const uint64_t tag = v >> 32;
    Job& j = jobs_[tag & 1];
    const int64_t n = j.n.load(), chunk = j.chunk.load();
    const int64_t begin = static_cast<int64_t>(v & 0xffffffffu);
    if (begin >= n) return;
    const int64_t end = std::min(n, begin + chunk);
    // seq_cst: if the loads of j above saw a newer job's values, the tag has moved on and the CAS fails.
    if (!claim_.compare_exchange_weak(v, (tag << 32) | static_cast<uint64_t>(end))) continue;
    (*j.fn.load())(begin, end);  // the job cannot change before this chunk is counted done
    if (done_.fetch_add(end - begin) + (end - begin) == n) {
      std::lock_guard<std::mutex> lk(mu_);
      if (caller_sleeping_) cv_done_.notify_one();
    }
    v = claim_.load();
  }
}

void ThreadPool::begin_run() {
  run_epoch_.fetch_add(1, std::memory_order_relaxed);
  if (!caller_pinned_ && pin_current_thread(affinity_[0])) {
    caller_pinned_ = true;
    unpinned_.fetch_sub(1);
  }
}

void ThreadPool::worker_loop(int index) {
  const int cpu = affinity_.empty() ? -1 : affinity_[index % affinity_.size()];
  bool pinned = cpu < 0 || pin_current_thread(cpu);
  if (!pinned) unpinned_.fetch_add(1);
  uint64_t tried = run_epoch_.load(std::memory_order_relaxed);
  uint64_t seen = 0;
  while (true) {
    auto ready = [&] { return stop_.load(std::memory_order_acquire) || generation_.load(std::memory_order_acquire) != seen; };
    if (!spin_until(ready)) {
      std::unique_lock<std::mutex> lk(mu_);
      ++sleeping_workers_;
      cv_start_.wait(lk, ready);
      --sleeping_workers_;
    }
    if (stop_.load(std::memory_order_acquire)) return;
    seen = generation_.load(std::memory_order_acquire);
    if (!pinned && run_epoch_.load(std::memory_order_relaxed) != tried) {  // once per run until it succeeds
      tried = run_epoch_.load(std::memory_order_relaxed);
      if ((pinned = pin_current_thread(cpu))) unpinned_.fetch_sub(1);
    }
    run_chunks();
  }
}

void ThreadPool::parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn) {
  if (n <= 0) return;
  calls_.fetch_add(1, std::memory_order_relaxed);
  grain = std::max<int64_t>(grain, 1);
  // A kernel's grain amortizes the dispatch cost (its serial threshold and minimum chunk). While the pool is hot
  // (spin_run) a dispatch costs a few microseconds instead of an OS wake-up: the serial threshold is divided by
  // hot_serial_div_ and the minimum chunk by hot_serial_div_ * hot_grain_div_. Chunking only partitions the items:
  // every kernel computes each item the same way whatever its chunk (1 and N threads give identical outputs).
  const bool hot = num_threads_ > 1 && hot_.load(std::memory_order_relaxed);
  if (hot) grain = std::max<int64_t>(1, grain / hot_serial_div_);
  if (num_threads_ == 1 || n <= grain) {
    fn(0, n);
    return;
  }
  BF_CHECK(n < (int64_t(1) << 31), "parallel_for: too many items (", n, ")");
  if (hot) grain = std::max<int64_t>(1, grain / hot_grain_div_);
  // About chunks_per_thread_ chunks per thread for load balance on big.LITTLE cores (1 X2 + 3 A710: the X2
  // takes more of them; the last chunk an A710 holds bounds the region).
  const int64_t per = chunks_per_thread_ * num_threads_;
  int64_t chunk = std::max<int64_t>(grain, (n + per - 1) / per);
  const uint64_t gen = generation_.load(std::memory_order_relaxed) + 1;
  const uint64_t tag = gen & 0xffffffffu;
  Job& j = jobs_[tag & 1];
  j.fn.store(&fn);
  j.n.store(n);
  j.chunk.store(chunk);
  done_.store(0);
  claim_.store(tag << 32);                              // publishes the job to run_chunks()
  generation_.store(gen, std::memory_order_release);    // wakes the workers
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (sleeping_workers_ > 0) cv_start_.notify_all();
  }
  run_chunks();
  auto done = [&] { return done_.load(std::memory_order_acquire) == n; };
  if (!spin_until(done)) {
    std::unique_lock<std::mutex> lk(mu_);
    caller_sleeping_ = true;
    cv_done_.wait(lk, done);
    caller_sleeping_ = false;
  }
}

}  // namespace bitino
