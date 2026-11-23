#include "bitino/thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__arm__) && defined(__linux__)
#include <setjmp.h>
#include <signal.h>

#include <mutex>
#include <string>
#endif

#include "bitino/error.h"

namespace bitino {
namespace {

#if defined(__arm__) && defined(__linux__)
// 32-bit ARM. A 32-bit process on an arm64 kernel without a compat vDSO (Redmi 9A: Android 11, kernel 4.19) reads
// steady_clock with a system call: 430 ns on its Cortex-A53, against ~10 ns for the virtual counter (CNTVCT, MRRC
// p15 c14). The pool times every chunk, so on that phone the clock reads alone made 4-thread transformer runs 4-5 %
// slower than with equal chunks. The counter is used when steady_clock is slow and a read of the counter works:
// ARMv7 cores without the generic timer (Cortex-A9) or kernels that deny EL0 access raise SIGILL, caught once here
// under a temporary handler. BITINO_POOL_CLOCK=steady keeps steady_clock.
sigjmp_buf g_probe_env;
void probe_sigill(int) { siglongjmp(g_probe_env, 1); }

inline uint64_t read_cntvct() {
  uint64_t v;
  __asm__ __volatile__("isb; mrrc p15, 1, %Q0, %R0, c14" : "=r"(v));
  return v;
}

bool probe_cntvct() {
  const char* e = std::getenv("BITINO_POOL_CLOCK");
  if (e && std::string(e) == "steady") return false;
  using C = std::chrono::steady_clock;
  const auto t0 = C::now();
  uint64_t sink = 0;
  for (int i = 0; i < 32; ++i) sink += static_cast<uint64_t>(C::now().time_since_epoch().count());
  const double ns = std::chrono::duration<double, std::nano>(C::now() - t0).count() / 32;
  if (ns < 150 || sink == 1) return false;  // vDSO: cheap enough
  struct sigaction sa = {}, old = {};
  sa.sa_handler = probe_sigill;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGILL, &sa, &old) != 0) return false;
  bool ok = false;
  if (sigsetjmp(g_probe_env, 1) == 0) {
    const uint64_t a = read_cntvct();
    for (volatile int i = 0; i < 20000; ++i) {
    }
    ok = read_cntvct() > a;  // a counter that does not advance (emulated as 0) is useless
  }
  sigaction(SIGILL, &old, nullptr);
  return ok;
}

std::atomic<bool> g_cntvct{false};  // set once (ThreadPool constructor, before its workers start)

void choose_clock() {
  static std::once_flag once;
  std::call_once(once, [] { g_cntvct.store(probe_cntvct(), std::memory_order_relaxed); });
}
#else
void choose_clock() {}
#endif

// Chunk timing for the speed weights: the generic timer on AArch64 (one register read), steady_clock elsewhere
// (32-bit ARM: the virtual counter when steady_clock is a system call, above).
inline uint64_t ticks() {
#if defined(__aarch64__)
  uint64_t v;
  __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(v));
  return v;
#else
#if defined(__arm__) && defined(__linux__)
  if (g_cntvct.load(std::memory_order_relaxed)) return read_cntvct();
#endif
  return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

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
  equal_chunks_per_thread_ = env_int("BITINO_POOL_EQUAL_CHUNKS", equal_chunks_per_thread_, 1, 64);
  hot_grain_div_ = env_int("BITINO_POOL_HOT_GRAIN_DIV", hot_grain_div_, 1, 1 << 20);
  hot_serial_div_ = env_int("BITINO_POOL_HOT_SERIAL_DIV", hot_serial_div_, 1, 1 << 20);
  stats_on_ = env_int("BITINO_POOL_STATS", 0, 0, 1) != 0;
  weighted_ = env_int("BITINO_POOL_WEIGHTED", 1, 0, 1) != 0;
  factor_ = static_cast<float>(env_int("BITINO_POOL_FACTOR", 2, 1, 64));
  skip_ = env_int("BITINO_POOL_SKIP", 1, 0, 1) != 0;
  inflight_ = env_int("BITINO_POOL_SKIP_INFLIGHT", 1, 0, 1) != 0;
  skip_slack_ = static_cast<float>(env_int("BITINO_POOL_SKIP_SLACK_PCT", 100, 100, 1000)) / 100.f;
  sizing_ = env_int("BITINO_POOL_SIZING", 1, 0, 1) != 0;
  sym_tol_ = static_cast<float>(env_int("BITINO_POOL_SYM_PCT", 20, 0, 1000)) / 100.f;
  time_every_ = env_int("BITINO_POOL_TIME_EVERY", time_every_, 1, 1 << 20);
  if (weighted_ && num_threads > 1) choose_clock();
  wrows_.emplace_back(new std::atomic<float>[static_cast<size_t>(std::max(num_threads, 1))]);
  for (int i = 0; i < std::max(num_threads, 1); ++i) wrows_[0][i].store(1.f);
  wfast_.reset(new std::atomic<float>[static_cast<size_t>(std::max(num_threads, 1))]);
  for (int i = 0; i < std::max(num_threads, 1); ++i) wfast_[i].store(0.f);
  rslot_.reset(new RegionSlot[static_cast<size_t>(std::max(num_threads, 1))]);
  BF_CHECK(num_threads >= 1, "num_threads must be >= 1, got ", num_threads);
  tstats_.resize(static_cast<size_t>(num_threads));
  active_ = num_threads;
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

namespace {
inline double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

ThreadPool::Stats ThreadPool::stats() const {
  Stats s;
  s.regions = st_regions_;
  s.inline_regions = st_inline_;
  s.region_us = st_region_us_;
  s.tail_us = st_tail_us_;
  s.inline_us = st_inline_us_;
  for (const ThreadStats& t : tstats_) {
    s.busy_us.push_back(t.busy_us);
    s.items.push_back(t.items);
    s.chunks.push_back(t.chunks);
  }
  return s;
}

void ThreadPool::reset_stats() {
  st_regions_ = st_inline_ = 0;
  st_region_us_ = st_tail_us_ = st_inline_us_ = 0;
  for (ThreadStats& t : tstats_) t = ThreadStats{};
}

void ThreadPool::run_chunks(int self) {
  uint64_t v = claim_.load();
  // Speed timing: one clock read per chunk. A chunk's time runs from the end of this thread's previous chunk (or its
  // first claim) to its own end, so the claims between chunks (a CAS) count too.
  uint64_t k_prev = 0, k_tag = ~uint64_t(0);  // k_tag: the job k_prev belongs to (a worker may run on into the next)
  while (true) {
    const uint64_t tag = v >> 32;
    Job& j = jobs_[tag & 1];
    const bool timed = j.timed.load(std::memory_order_relaxed);
    const int64_t n = j.n.load(), chunk = j.chunk.load();
    const int64_t begin = static_cast<int64_t>(v & 0xffffffffu);
    if (begin >= n || self >= j.active.load()) return;
    int64_t size = chunk;
    if (j.weighted.load()) {
      // Weighted factoring: a claim of about rem * w / (factor * W) items lasts about the same time on every core.
      // A worker whose chunk would end after the threads at least as fast as itself could finish all that is left
      // (not before one chunk of the fastest thread) leaves it to them. The caller always claims: it completes
      // every region.
      const int64_t rem = n - begin;
      const float w = j.w.load()[self].load(std::memory_order_relaxed), ws = j.wsum.load(), wm = j.wmax.load();
      if (sizing_) size = std::max<int64_t>(chunk, static_cast<int64_t>(static_cast<float>(rem) * w / (factor_ * ws)));
      if (self != 0 && skip_) {
        const float c = static_cast<float>(std::min(size, rem)), wf = wfast_[self].load(std::memory_order_relaxed);
        // The faster threads also have to finish the chunks they hold: all items not done yet (inflight_ mode).
        const float left = static_cast<float>(inflight_ ? std::max<int64_t>(rem, n - done_.load(std::memory_order_relaxed)) : rem);
        if (c / w > skip_slack_ * std::max(left / std::max(wf, 1e-3f), c / wm)) return;
      }
    }
    const int64_t end = std::min(n, begin + size);
    if (timed && k_tag != tag) {  // first claim attempt of this thread in the region
      k_prev = ticks();
      k_tag = tag;
    }
    // seq_cst: if the loads of j above saw a newer job's values, the tag has moved on and the CAS fails.
    if (!claim_.compare_exchange_weak(v, (tag << 32) | static_cast<uint64_t>(end))) continue;
    // The job cannot change before this chunk is counted done.
    const double t0 = stats_on_ ? now_us() : 0;
    if (const RangeFn* f = j.fn.load()) (*f)(begin, end);
    else (*j.sfn.load())(self, begin, end);
    if (timed) {  // before done_: the caller reads the slots once every item is done
      RegionSlot& rs = rslot_[self];
      const uint64_t k1 = ticks();
      rs.ticks.store(rs.ticks.load(std::memory_order_relaxed) + (k1 - k_prev), std::memory_order_relaxed);
      k_prev = k1;
      rs.items.store(rs.items.load(std::memory_order_relaxed) + (end - begin), std::memory_order_relaxed);
    }
    if (stats_on_) {
      ThreadStats& ts = tstats_[static_cast<size_t>(self)];
      ts.busy_us += now_us() - t0;
      ts.items += static_cast<uint64_t>(end - begin);
      ts.chunks += 1;
    }
    if (done_.fetch_add(end - begin) + (end - begin) == n) {
      std::lock_guard<std::mutex> lk(mu_);
      if (caller_sleeping_) cv_done_.notify_one();
    }
    v = claim_.load();
  }
}

void ThreadPool::set_weight_key(int key) {
  const int row = key < 0 ? 0 : key + 1;
  if (row >= static_cast<int>(wrows_.size())) wrows_.resize(static_cast<size_t>(row) + 1);
  if (!wrows_[row]) {
    wrows_[row].reset(new std::atomic<float>[static_cast<size_t>(num_threads_)]);
    for (int i = 0; i < num_threads_; ++i) wrows_[row][i].store(wrows_[0][i].load(std::memory_order_relaxed));
  }
  wrow_ = row;
}

// After a dispatched region (caller): the items per tick of every thread that ran chunks, relative to the caller's,
// into its weight in the region's row and in the overall row (exponential average, clamped). Regions where the
// caller ran nothing leave the weights alone.
void ThreadPool::update_weights(int active, std::atomic<float>* row) {
  const int64_t i0 = rslot_[0].items.load(std::memory_order_relaxed);
  const uint64_t k0 = rslot_[0].ticks.load(std::memory_order_relaxed);
  if (i0 > 0 && k0 > 0) {
    const double r0 = static_cast<double>(i0) / static_cast<double>(k0);
    for (int i = 1; i < active; ++i) {
      const int64_t it = rslot_[i].items.load(std::memory_order_relaxed);
      const uint64_t kt = rslot_[i].ticks.load(std::memory_order_relaxed);
      if (it <= 0 || kt == 0) continue;
      const double s = std::min(8.0, std::max(0.05, static_cast<double>(it) / static_cast<double>(kt) / r0));
      for (std::atomic<float>* r : {row, wrows_[0].get()}) {
        const float w = r[i].load(std::memory_order_relaxed);
        r[i].store(static_cast<float>(w + 0.125 * (s - w)), std::memory_order_relaxed);
        if (r == wrows_[0].get()) break;
      }
    }
    weight_updates_.fetch_add(1, std::memory_order_relaxed);
  }
  for (int i = 0; i < active; ++i) {
    rslot_[i].items.store(0, std::memory_order_relaxed);
    rslot_[i].ticks.store(0, std::memory_order_relaxed);
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
    run_chunks(index);
  }
}

void ThreadPool::set_active_threads(int k) { active_ = std::min(num_threads_, std::max(1, k)); }

void ThreadPool::parallel_for(int64_t n, int64_t grain, const RangeFn& fn) { dispatch(n, grain, &fn, nullptr); }

void ThreadPool::parallel_for_slot(int64_t n, int64_t grain, const SlotFn& fn) { dispatch(n, grain, nullptr, &fn); }

void ThreadPool::dispatch(int64_t n, int64_t grain, const RangeFn* fn, const SlotFn* sfn) {
  if (n <= 0) return;
  calls_.fetch_add(1, std::memory_order_relaxed);
  grain = std::max<int64_t>(grain, 1);
  // A kernel's grain amortizes the dispatch cost (its serial threshold and minimum chunk). While the pool is hot
  // (spin_run) a dispatch costs a few microseconds instead of an OS wake-up: the serial threshold is divided by
  // hot_serial_div_ and the minimum chunk by hot_serial_div_ * hot_grain_div_. Chunking only partitions the items:
  // every kernel computes each item the same way whatever its chunk (1 and N threads give identical outputs).
  const int active = active_;
  const bool hot = active > 1 && hot_.load(std::memory_order_relaxed);
  if (hot) grain = std::max<int64_t>(1, grain / hot_serial_div_);
  if (active == 1 || n <= grain) {
    const double t0 = stats_on_ ? now_us() : 0;
    if (fn) (*fn)(0, n);
    else (*sfn)(0, 0, n);
    if (stats_on_) {
      st_inline_us_ += now_us() - t0;
      ++st_inline_;
    }
    return;
  }
  const double st_t0 = stats_on_ ? now_us() : 0;
  BF_CHECK(n < (int64_t(1) << 31), "parallel_for: too many items (", n, ")");
  if (hot) grain = std::max<int64_t>(1, grain / hot_grain_div_);
  // About chunks_per_thread_ chunks per thread for load balance on big.LITTLE cores (1 X2 + 3 A710: the X2
  // takes more of them; the last chunk an A710 holds bounds the region).
  std::atomic<float>* wr = wrows_[wrow_].get();
  float ws = 0, wm = 0, wmin = 0;
  if (weighted_) {
    for (int i = 0; i < active; ++i) {
      const float w = wr[i].load(std::memory_order_relaxed);
      ws += w;
      wm = std::max(wm, w);
      wmin = i == 0 ? w : std::min(wmin, w);
    }
  }
  // Weighted claims only between cores of measurably different speed. When the measured speeds of the active threads
  // are within sym_tol_ of each other (one cluster: the Redmi 9A's four Cortex-A53 at 4 threads, the moto e13's two
  // A75 at 2) the region takes the equal chunks of the pool before weighted claims: there factoring and the skip rule
  // only acted on measurement noise (Bi-ViT DeiT-Tiny on the Redmi at 4 threads ~1 % slower than equal chunks).
  const bool weighted = weighted_ && use_weighted_ && !(sym_tol_ > 0 && wm <= wmin * (1.f + sym_tol_));
  // Weighted claims start large and shrink to the minimum chunk at the end of the region; equal chunks are all of the
  // minimum size, so they use fewer per thread (the pool before weighted claims: 8): on the moto e13 at 8 threads
  // (2 A75 + 6 A55) 16 equal chunks per thread made Bi-ViT-Small int8_full 6 % slower than 8.
  const int64_t per = (weighted ? chunks_per_thread_ : equal_chunks_per_thread_) * active;
  int64_t chunk = std::max<int64_t>(grain, (n + per - 1) / per);
  const uint64_t gen = generation_.load(std::memory_order_relaxed) + 1;
  const uint64_t tag = gen & 0xffffffffu;
  Job& j = jobs_[tag & 1];
  j.fn.store(fn);
  j.sfn.store(sfn);
  j.n.store(n);
  j.chunk.store(chunk);
  j.active.store(active);
  j.weighted.store(weighted);
  // Speed measurement (chunk timing and the caller's weight update after the region): every region of a weight row's
  // first 16, then one in time_every_ (8). The update reads every thread's slot on the caller's critical path; on the
  // Redmi 9A that was ~0.5 % of Bi-ViT's 4-thread time (500 regions per run).
  bool timed = false;
  if (weighted_) {
    if (wrow_ >= static_cast<int>(row_visits_.size())) row_visits_.resize(static_cast<size_t>(wrow_) + 1, 0);
    const uint32_t visit = row_visits_[wrow_]++;
    timed = visit < 16 || visit % static_cast<uint32_t>(time_every_) == 0;
  }
  j.timed.store(timed, std::memory_order_relaxed);
  if (weighted) {
    j.w.store(wr);
    j.wsum.store(ws);
    j.wmax.store(wm);
    for (int i = 1; i < active; ++i) {
      const float wi = wr[i].load(std::memory_order_relaxed);
      float wf = 0;
      for (int k = 0; k < active; ++k) {
        const float wk = wr[k].load(std::memory_order_relaxed);
        if (k != i && wk >= wi) wf += wk;
      }
      wfast_[i].store(wf, std::memory_order_relaxed);
    }
  }
  done_.store(0);
  claim_.store(tag << 32);                              // publishes the job to run_chunks()
  generation_.store(gen, std::memory_order_release);    // wakes the workers
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (sleeping_workers_ > 0) cv_start_.notify_all();
  }
  run_chunks(0);
  const double st_t1 = stats_on_ ? now_us() : 0;
  auto done = [&] { return done_.load(std::memory_order_acquire) == n; };
  if (!spin_until(done)) {
    std::unique_lock<std::mutex> lk(mu_);
    caller_sleeping_ = true;
    cv_done_.wait(lk, done);
    caller_sleeping_ = false;
  }
  if (timed) update_weights(active, wr);
  if (stats_on_) {
    const double st_t2 = now_us();
    ++st_regions_;
    st_region_us_ += st_t2 - st_t0;
    st_tail_us_ += st_t2 - st_t1;
  }
}

}  // namespace bitino
