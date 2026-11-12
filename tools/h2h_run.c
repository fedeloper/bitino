// h2h_run: the one launcher every runtime of the head-to-head benchmark (scripts/h2h_lce_tflite.sh) runs under.
//
//   h2h_run --cpus 7 [--sample-ms 25] [--pin-wait-ms 30000] -- /path/to/benchmark args...
//
// - Pins the child to --cpus with sched_setaffinity() before exec (the same mechanism as taskset; the benchmark
//   binaries are not asked to pin themselves) and reads the mask back; while a paused core refuses the mask it
//   busy-spins and retries (see below), and fails after --pin-wait-ms.
// - While the child runs, samples every --sample-ms the CPU each of its threads last ran on (/proc/<pid>/task/<tid>/stat
//   field 39) and its Cpus_allowed_list, so the report shows where the benchmark actually ran.
// - Reports peak RSS (wait4() ru_maxrss), CPU times, context switches and page faults of the child.
// Prints one final line on stderr:  H2H_RUN {json}
// Build: <ndk>/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang -O2 -o h2h_run tools/h2h_run.c
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAXCPU 64

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int parse_cpus(const char* s, cpu_set_t* set) {
  CPU_ZERO(set);
  int n = 0;
  const char* p = s;
  while (*p) {
    char* e;
    long a = strtol(p, &e, 10);
    if (e == p || a < 0 || a >= MAXCPU) return -1;
    long b = a;
    if (*e == '-') {
      p = e + 1;
      b = strtol(p, &e, 10);
      if (e == p || b < a || b >= MAXCPU) return -1;
    }
    for (long c = a; c <= b; ++c, ++n) CPU_SET((int)c, set);
    p = (*e == ',') ? e + 1 : e;
    if (*e && *e != ',') return -1;
  }
  return n;
}

static void set_to_list(const cpu_set_t* set, char* out, size_t cap) {
  out[0] = 0;
  size_t len = 0;
  for (int c = 0; c < MAXCPU; ++c)
    if (CPU_ISSET(c, set)) len += snprintf(out + len, len < cap ? cap - len : 0, "%s%d", len ? "," : "", c);
}

// Field 39 ("processor") of a /proc stat line; the comm field may contain spaces, so count after the last ')'.
static int stat_cpu(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  char buf[2048];
  size_t n = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  buf[n] = 0;
  char* p = strrchr(buf, ')');
  if (!p) return -1;
  int field = 2;  // p points at the end of field 2
  for (char* q = p + 1; *q; ++q) {
    if (*q == ' ') {
      ++field;
      if (field == 39) return atoi(q + 1);
    }
  }
  return -1;
}

int main(int argc, char** argv) {
  const char* cpus = NULL;
  int sample_ms = 25, i = 1;
  double pin_wait_ms = 30000;
  for (; i < argc; ++i) {
    if (!strcmp(argv[i], "--cpus") && i + 1 < argc) cpus = argv[++i];
    else if (!strcmp(argv[i], "--pin-wait-ms") && i + 1 < argc) pin_wait_ms = atof(argv[++i]);
    else if (!strcmp(argv[i], "--sample-ms") && i + 1 < argc) sample_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--")) { ++i; break; }
    else { fprintf(stderr, "h2h_run: unknown argument %s\n", argv[i]); return 2; }
  }
  if (!cpus || i >= argc) {
    fprintf(stderr, "usage: h2h_run --cpus LIST [--sample-ms N] -- cmd args...\n");
    return 2;
  }
  cpu_set_t want;
  if (parse_cpus(cpus, &want) <= 0) { fprintf(stderr, "h2h_run: bad --cpus %s\n", cpus); return 2; }
  char want_list[256];
  set_to_list(&want, want_list, sizeof want_list);

  // Close-on-exec pipe: the parent starts sampling only once the child has set its affinity and exec'd (EOF).
  int pfd[2];
  if (pipe2(pfd, O_CLOEXEC) != 0) { perror("pipe2"); return 1; }
  double t0 = now_s();
  pid_t pid = fork();
  if (pid < 0) { perror("fork"); return 1; }
  if (pid == 0) {
    close(pfd[0]);
    // Qualcomm core_ctl "pauses" idle prime cores (OnePlus Open: cpu7 with the screen off): sched_setaffinity onto a
    // paused core fails with EINVAL, or succeeds while the kernel substitutes other CPUs (read back as 0-5), until
    // core_ctl sees load and unpauses it. Busy-spin (on whatever core this child runs on) and retry until the mask
    // reads back exactly as requested, for up to --pin-wait-ms; the wait is reported (H2H_PIN_WAIT) for every runtime.
    double tp = now_s();
    long tries = 0;
    cpu_set_t got;
    char got_list[256] = "";
    for (;;) {
      int rc = sched_setaffinity(0, sizeof want, &want), err = errno;
      if (rc == 0) {
        sched_getaffinity(0, sizeof got, &got);
        set_to_list(&got, got_list, sizeof got_list);
        if (CPU_EQUAL(&got, &want)) break;
      } else if (err != EINVAL) {
        fprintf(stderr, "H2H_AFFINITY_ERROR sched_setaffinity(%s): %s\n", want_list, strerror(err));
        _exit(125);
      }
      if ((now_s() - tp) * 1000.0 > pin_wait_ms) {
        fprintf(stderr, "H2H_AFFINITY_ERROR requested %s, last read back '%s' (%s) after %ld tries\n", want_list,
                got_list, rc ? strerror(err) : "mask differs", tries);
        _exit(125);
      }
      ++tries;
      for (double t1 = now_s(); now_s() - t1 < 0.02;) {}
    }
    fprintf(stderr, "H2H_PIN_WAIT {\"tries\": %ld, \"wait_s\": %.3f}\n", tries, now_s() - tp);
    execv(argv[i], argv + i);
    fprintf(stderr, "h2h_run: exec %s: %s\n", argv[i], strerror(errno));
    _exit(127);
  }

  close(pfd[1]);
  char cbuf;
  while (read(pfd[0], &cbuf, 1) < 0 && errno == EINTR) {}
  close(pfd[0]);
  long hist[MAXCPU] = {0};
  long samples = 0, max_threads = 0, allowed_samples = 0, allowed_mismatch = 0;
  char allowed[256] = "";
  int status = 0;
  struct rusage ru;
  memset(&ru, 0, sizeof ru);
  for (;;) {
    pid_t r = wait4(pid, &status, WNOHANG, &ru);
    if (r == pid) break;
    if (r < 0 && errno != EINTR) { perror("wait4"); return 1; }
    char path[128];
    snprintf(path, sizeof path, "/proc/%d/task", (int)pid);
    DIR* d = opendir(path);
    if (d) {
      struct dirent* e;
      long nt = 0;
      while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char sp[192];
        snprintf(sp, sizeof sp, "/proc/%d/task/%s/stat", (int)pid, e->d_name);
        int c = stat_cpu(sp);
        if (c >= 0 && c < MAXCPU) { hist[c]++; samples++; }
        nt++;
      }
      closedir(d);
      if (nt > max_threads) max_threads = nt;
    }
    snprintf(path, sizeof path, "/proc/%d/status", (int)pid);
    FILE* f = fopen(path, "r");
    if (f) {
      char line[512];
      while (fgets(line, sizeof line, f))
        if (!strncmp(line, "Cpus_allowed_list:", 18)) {
          char* v = line + 18;
          while (*v == ' ' || *v == '\t') ++v;
          v[strcspn(v, "\n")] = 0;
          snprintf(allowed, sizeof allowed, "%s", v);
          allowed_samples++;
          if (strcmp(allowed, want_list)) allowed_mismatch++;  // the mask was changed while the benchmark ran
        }
      fclose(f);
    }
    struct timespec ts = {sample_ms / 1000, (sample_ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
  }
  double wall = now_s() - t0;
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  fprintf(stderr, "H2H_RUN {\"cpus_requested\": \"%s\", \"exit\": %d, \"wall_s\": %.3f, \"maxrss_kb\": %ld, "
                  "\"utime_s\": %.3f, \"stime_s\": %.3f, \"nvcsw\": %ld, \"nivcsw\": %ld, \"minflt\": %ld, \"majflt\": %ld, "
                  "\"cpus_allowed_list\": \"%s\", \"allowed_samples\": %ld, \"allowed_mismatch_samples\": %ld, \"max_threads\": %ld, \"cpu_samples\": %ld, \"cpu_hist\": {",
          want_list, code, wall, ru.ru_maxrss, ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6,
          ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6, ru.ru_nvcsw, ru.ru_nivcsw, ru.ru_minflt, ru.ru_majflt, allowed,
          allowed_samples, allowed_mismatch, max_threads, samples);
  int first = 1;
  for (int c = 0; c < MAXCPU; ++c)
    if (hist[c]) { fprintf(stderr, "%s\"%d\": %ld", first ? "" : ", ", c, hist[c]); first = 0; }
  fprintf(stderr, "}}\n");
  return code;
}
