#!/usr/bin/env python3
"""Runs a benchmark matrix on the connected Android device (default) or on this host and collects raw JSON.

Each configuration = (binary, model file, stage, threads, affinity). Every configuration
is repeated --repeats times (round-robin over configurations, so slow drift and thermal
effects spread evenly). All raw bitino_bench JSON reports are pulled to --out;
tools/report_benchmarks.py aggregates them.

Cool-down on the device (--cooldown-mode adaptive, the default): before every run the script waits until the
hottest CPU thermal zone (cpu-*, cpuss-*) is back within --cool-margin degrees C of the idle baseline (at least
--cool-min s, at most --cool-max s; a timeout is recorded, not fatal). The idle baseline is measured once (the
phone idles until the CPU temperature stops falling) and cached in .cache/bench_idle_baseline.json for
--baseline-max-age hours, so every model of a suite run uses the same target. Each raw JSON gains a "cooldown"
block (wait, temperature at the end of the previous run and at the start of this one, timeout flag).
--cooldown-mode fixed sleeps --cooldown s after each run instead (the behaviour before 2026-11-02).

Frequency-cap gate on the device (both cool-down modes): the vendor thermal engine lowers scaling_max_freq when the
skin gets warm (on the S22 while charging over USB: X2 1.84 of 2.99 GHz, A710 1.56 of 2.50, A510 1.36 of 1.79), which
the CPU zones do not show. Before every run the script waits until every cpufreq policy is back at cpuinfo_max_freq
(at most --cap-max s; 0 = record only). Each raw JSON gains a "throttle" block (caps, Android thermal status and SKIN
temperature before and after the run, wait, timeout, throttled flag). A run that got capped while it ran is repeated
after the cap lifts (up to 3 attempts); a run started after a gate timeout is kept and flagged.

  python3 tools/benchmark.py --plan stages --model-dir artifacts/larq-zoo-quicknet-small \
      --out reports/benchmarks/larq-zoo-quicknet-small/run1

Plans:
  stages   optimization stages at 1 thread pinned to the X2 core (cpu7)
  threads  final configuration at 1/2/4/8 threads (unpinned) + pinned cluster variants
  quick    final configuration only (1 thread pinned, 4/8 threads, INT8)
  custom   --configs JSON file with a list of {name, binary, model, stage, threads, affinity, extra}

Host mode (--target host) runs build/host/bitino_bench locally with the same plans, JSON schema and
tools/report_benchmarks.py summaries; no adb involved. Defaults are for the i9-14900K host (Raptor Lake,
CPUs 0-15 = 8 P-cores x 2 hyperthreads, siblings (0,1), (2,3), ...; CPUs 16-31 = E-cores; override with
--host-pin / --host-pcores / --host-ecores): single-thread configs are pinned to a P-core (CPU 2); the
thread plan has unpinned, P-core-only (one thread per physical core, then with hyperthreads), E-core-only
and all-core variants. plan.json records the CPU model, the cpufreq driver/governor/limits, turbo state
and load average; each raw JSON gains a "host_env" block (load average, governor and current frequency of
the pinned CPUs before and after the run). Host timings are host numbers, never device numbers.

  python3 tools/benchmark.py --target host --plan quick --stage optimized \
      --model-dir artifacts/larq-zoo-quicknet-small --out reports/benchmarks_host/larq-zoo-quicknet-small/quick1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADB = os.path.expanduser("~/Android/Sdk/platform-tools/adb")
LOCK = os.environ.get("BITINO_DEVICE_LOCK", "/tmp/bitino-device.lock")
DEV = os.environ.get("BITINO_DEVICE_DIR", "/data/local/tmp/bitino")  # e.g. /data/local/tmp/bitino_perf


def adb(*args, check=True, lock=False):
    cmd = [ADB, *args]
    if lock and os.path.exists(LOCK):
        cmd = ["flock", LOCK] + cmd
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


# ------------------------------------------------------------------ adaptive cool-down (device)
BASELINE_CACHE = os.path.join(ROOT, ".cache", "bench_idle_baseline.json")
_ZONES: list[str] = []


def cpu_temp_c() -> float:
    """Hottest CPU thermal zone (types cpu-* / cpuss-*) in degrees C, read with one adb shell call."""
    if not _ZONES:
        r = adb("shell", "for z in /sys/class/thermal/thermal_zone*; do echo \"$z $(cat $z/type 2>/dev/null)\"; done", check=False)
        _ZONES.extend(z for z, t in (l.split()[:2] for l in r.stdout.splitlines() if len(l.split()) >= 2)
                      if t.startswith(("cpu-", "cpuss-")))
        if not _ZONES:
            raise RuntimeError("no cpu-*/cpuss-* thermal zones found on the device")
    r = adb("shell", "cat " + " ".join(f"{z}/temp" for z in _ZONES) + " 2>/dev/null", check=False)
    vals = [int(v) / 1000.0 for v in r.stdout.split() if v.lstrip("-").isdigit()]
    if not vals:
        raise RuntimeError("could not read CPU temperatures")
    return max(vals)


def idle_baseline_c(max_age_h: float) -> dict:
    """Idle CPU temperature: cached, or measured by idling until the temperature stops falling
    (< 0.5 C drop over 30 s, at most 300 s)."""
    serial = adb("get-serialno", check=False).stdout.strip()
    try:
        c = json.load(open(BASELINE_CACHE))
        if c.get("serial") == serial and time.time() - c["measured_unix"] < max_age_h * 3600:
            return c
    except (OSError, ValueError, KeyError):
        pass
    hist = [(time.time(), cpu_temp_c())]
    while True:
        time.sleep(5)
        hist.append((time.time(), cpu_temp_c()))
        old = [t for ts, t in hist if ts <= hist[-1][0] - 30]
        if (old and old[-1] - hist[-1][1] < 0.5) or hist[-1][0] - hist[0][0] >= 300:
            break
    c = {"serial": serial, "temp_c": hist[-1][1], "measured_unix": hist[-1][0],
         "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(hist[-1][0])),
         "settle_s": round(hist[-1][0] - hist[0][0], 1), "start_temp_c": hist[0][1]}
    os.makedirs(os.path.dirname(BASELINE_CACHE), exist_ok=True)
    json.dump(c, open(BASELINE_CACHE, "w"), indent=1)
    print(f"idle baseline {c['temp_c']:.1f} C (from {c['start_temp_c']:.1f} C after {c['settle_s']} s)", file=sys.stderr)
    return c


def wait_device(max_s: float = 1800.0) -> bool:
    """True once adb sees the device again (a dropped USB link), False after max_s seconds."""
    t0 = time.time()
    while adb("get-state", check=False).stdout.strip() != "device":
        if time.time() - t0 > max_s:
            return False
        time.sleep(10)
    if time.time() - t0 > 1:
        print(f"device back after {time.time() - t0:.0f} s", file=sys.stderr)
    return True


def wait_cool(target_c: float, min_s: float, max_s: float) -> dict:
    t0 = time.time()
    before = cpu_temp_c()
    time.sleep(min_s)
    t = cpu_temp_c()
    while t > target_c and time.time() - t0 < max_s:
        time.sleep(1.0)
        t = cpu_temp_c()
    return {"waited_s": round(time.time() - t0, 1), "temp_before_wait_c": before, "temp_at_start_c": t,
            "target_c": target_c, "timeout": t > target_c}


def freq_caps() -> dict:
    """scaling_max_freq vs cpuinfo_max_freq (kHz) of every cpufreq policy, Android thermal status and SKIN temperature."""
    r = adb("shell", "for p in /sys/devices/system/cpu/cpufreq/policy*; do echo ${p##*/policy} "
            "$(cat $p/scaling_max_freq $p/cpuinfo_max_freq 2>/dev/null); done; "
            "dumpsys thermalservice 2>/dev/null | grep -E 'Thermal Status|mName=SKIN'", check=False)
    pol = {}
    for l in r.stdout.splitlines():
        f = l.split()
        if len(f) == 3 and all(x.isdigit() for x in f):
            pol[f[0]] = [int(f[1]), int(f[2])]
    if not pol:
        raise RuntimeError("could not read cpufreq limits")
    status = re.search(r"Thermal Status: (\d+)", r.stdout)
    skin = re.search(r"mValue=([-0-9.]+)[^}]*mName=SKIN", r.stdout)
    return {"policies": pol, "capped": any(s < m for s, m in pol.values()),
            "thermal_status": int(status.group(1)) if status else None, "skin_c": float(skin.group(1)) if skin else None}


def wait_uncapped(max_s: float) -> dict:
    t0 = time.time()
    st = freq_caps()
    while st["capped"] and time.time() - t0 < max_s:
        time.sleep(5)
        st = freq_caps()
    return {"waited_s": round(time.time() - t0, 1), "before": st, "timeout": st["capped"]}


def stage_plan(model_dir: str) -> list[dict]:
    m, mu = f"{model_dir}/model.bitino", f"{model_dir}/model_unfused.bitino"
    if not os.path.exists(mu):  # no unfused variant (e.g. nothing to fold): stage 9 then equals stage 8
        mu = m
    pin = "7"  # Cortex-X2
    c = []
    # Reference stages are 10-1000x slower: fewer iterations per repeat (recorded in plan.json).
    c.append(dict(name="01_reference_O0", binary="O0", model=mu, stage="reference", threads=1, affinity=pin, warmup=1, iterations=5))
    c.append(dict(name="02_reference_O3", binary="rel", model=mu, stage="reference", threads=1, affinity=pin, warmup=2, iterations=10))
    c.append(dict(name="03_packed_naive", binary="rel", model=mu, stage="packed_naive", threads=1, affinity=pin))
    c.append(dict(name="04_binary_scalar", binary="rel", model=mu, stage="binary_scalar", threads=1, affinity=pin))
    c.append(dict(name="05_neon", binary="rel", model=mu, stage="neon", threads=1, affinity=pin))
    c.append(dict(name="06_int8_reference", binary="rel", model=mu, stage="int8_reference", threads=1, affinity=pin))
    c.append(dict(name="07_int8_neon", binary="rel", model=mu, stage="int8_neon", threads=1, affinity=pin))
    c.append(dict(name="08_float_opt", binary="rel", model=mu, stage="optimized", threads=1, affinity=pin))
    c.append(dict(name="09_fusion", binary="rel", model=m, stage="optimized", threads=1, affinity=pin))
    c.append(dict(name="10_no_memory_reuse", binary="rel", model=m, stage="optimized", threads=1, affinity=pin, extra=["--no-reuse-memory"]))
    c.append(dict(name="11_final_exact", binary="rel", model=m, stage="all", threads=1, affinity=pin))
    c.append(dict(name="12_final_int8", binary="rel", model=m, stage="optimized_int8", threads=1, affinity=pin))
    return c


def quick_plan(model_dir: str, stage: str) -> list[dict]:
    """Final configuration only (for models outside the fully benchmarked priority set)."""
    m = f"{model_dir}/model.bitino"
    return [dict(name="t1_x2", binary="rel", model=m, stage=stage, threads=1, affinity="7"),
            dict(name="t4_unpinned", binary="rel", model=m, stage=stage, threads=4, affinity=""),
            dict(name="t8_unpinned", binary="rel", model=m, stage=stage, threads=8, affinity=""),
            dict(name="t1_x2_int8", binary="rel", model=m, stage="optimized_int8", threads=1, affinity="7")]


# ------------------------------------------------------------------ host mode
HOST_PIN = "2"                      # a P-core (hyperthread 0 of the second P-core)
HOST_PCORES = [0, 2, 4, 6, 8, 10, 12, 14]   # one hyperthread per physical P-core
HOST_ECORES = list(range(16, 32))


def _cpus(lst) -> str:
    return ",".join(str(c) for c in lst)


def host_stage_plan(model_dir: str, pin: str) -> list[dict]:
    """Same stages as the device plan, single thread pinned to a P-core. Stage names are the
    platform-neutral ones: on x86-64 "neon"/"int8_neon" select the AVX2 tiers (bitino_bench)."""
    c = stage_plan(model_dir)
    for cfg in c:
        cfg["affinity"] = pin
    return c


def host_quick_plan(model_dir: str, stage: str, pin: str, pcores: list[int]) -> list[dict]:
    m = f"{model_dir}/model.bitino"
    p8 = [int(pin)] + [c for c in pcores if c != int(pin)]
    return [dict(name="t1_pcore", binary="rel", model=m, stage=stage, threads=1, affinity=pin),
            dict(name="t8_pcore", binary="rel", model=m, stage=stage, threads=8, affinity=_cpus(p8[:8])),
            dict(name="t8_unpinned", binary="rel", model=m, stage=stage, threads=8, affinity=""),
            dict(name="t1_pcore_int8", binary="rel", model=m, stage="optimized_int8", threads=1, affinity=pin),
            dict(name="t8_pcore_int8", binary="rel", model=m, stage="optimized_int8", threads=8, affinity=_cpus(p8[:8]))]


def host_thread_plan(model_dir: str, stage: str, pin: str, pcores: list[int], ecores: list[int]) -> list[dict]:
    m = f"{model_dir}/model.bitino"
    p = [int(pin)] + [c for c in pcores if c != int(pin)]
    ht = p + [c + 1 for c in p]  # P-cores, then their hyperthread siblings
    c = []
    for t in (1, 2, 4, 8, 16):
        c.append(dict(name=f"t{t}_unpinned", binary="rel", model=m, stage=stage, threads=t, affinity=""))
    for t in (1, 2, 4, 8):
        c.append(dict(name=f"t{t}_pcore", binary="rel", model=m, stage=stage, threads=t, affinity=_cpus(p[:t])))
    c.append(dict(name="t16_pcore_ht", binary="rel", model=m, stage=stage, threads=16, affinity=_cpus(ht)))
    c.append(dict(name="t1_ecore", binary="rel", model=m, stage=stage, threads=1, affinity=str(ecores[0])))
    c.append(dict(name="t8_ecore", binary="rel", model=m, stage=stage, threads=8, affinity=_cpus(ecores[:8])))
    c.append(dict(name="t16_ecore", binary="rel", model=m, stage=stage, threads=16, affinity=_cpus(ecores[:16])))
    c.append(dict(name="t24_all_cores", binary="rel", model=m, stage=stage, threads=len(p) + len(ecores),
                  affinity=_cpus(p + ecores)))
    c.append(dict(name="t32_all_threads", binary="rel", model=m, stage=stage, threads=len(ht) + len(ecores),
                  affinity=_cpus(ht + ecores)))
    for t, aff in ((8, _cpus(p[:8])), (8, "")):
        c.append(dict(name=f"t{t}_{'pcore' if aff else 'unpinned'}_spin200", binary="rel", model=m, stage=stage, threads=t,
                      affinity=aff, extra=["--spin-us", "200"]))
    return c


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def host_cpu_state(cpus: list[int]) -> dict:
    out = {"loadavg": _read("/proc/loadavg")}
    for c in cpus:
        base = f"/sys/devices/system/cpu/cpu{c}/cpufreq/"
        out[f"cpu{c}"] = {"governor": _read(base + "scaling_governor"), "cur_khz": _read(base + "scaling_cur_freq")}
    return out


def host_info(pin: str) -> dict:
    model = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    base = f"/sys/devices/system/cpu/cpu{pin}/cpufreq/"
    return {"cpu_model": model, "hostname": os.uname().nodename, "kernel": os.uname().release,
            "online_cpus": _read("/sys/devices/system/cpu/online"),
            "cpufreq_driver": _read(base + "scaling_driver"), "governor_pinned_cpu": _read(base + "scaling_governor"),
            "energy_performance_preference": _read(base + "energy_performance_preference"),
            "scaling_max_khz_pinned_cpu": _read(base + "scaling_max_freq"),
            "cpuinfo_max_khz_pinned_cpu": _read(base + "cpuinfo_max_freq"),
            "intel_pstate_no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            "pinned_cpu_siblings": _read(f"/sys/devices/system/cpu/cpu{pin}/topology/thread_siblings_list"),
            "loadavg_at_start": _read("/proc/loadavg")}


def run_host(a, cfgs: list[dict], binaries: dict) -> int:
    """Host counterpart of the adb loop in main(): same plan.json / <config>__r<rep>.json layout."""
    found = {k: v for k, v in binaries.items() if os.path.exists(v)}
    # Custom configs may name a binary by path (e.g. A/B runs of two builds, interleaved per repeat).
    for c in cfgs:
        if c["binary"] not in binaries and os.path.exists(c["binary"]):
            found[c["binary"]] = os.path.abspath(c["binary"])
    for k, v in binaries.items():
        if k not in found and any(c["binary"] == k for c in cfgs):
            print(f"skip binary {k}: {v} missing", file=sys.stderr)
    cfgs = [c for c in cfgs if c["binary"] in found]
    plan = {"plan": a.plan, "model_dir": a.model_dir, "target": "host", "warmup": a.warmup, "iterations": a.iterations,
            "repeats": a.repeats, "cooldown_s": a.cooldown, "default_cpu": a.host_pin, "configs": cfgs,
            "binaries": found, "host": host_info(a.host_pin)}
    json.dump(plan, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    failures = 0
    for rep in range(a.repeats):
        for c in cfgs:
            tag = f"{c['name']}__r{rep}"
            out_json = os.path.join(a.out, f"{tag}.json")
            cmd = [found[c["binary"]], "--model", c["model"], "--stage", c["stage"], "--threads", str(c["threads"]),
                   "--warmup", str(c.get("warmup", a.warmup)), "--iterations", str(c.get("iterations", a.iterations)),
                   "--json", out_json, "--label", tag]
            if c.get("affinity"):
                cmd += ["--affinity", c["affinity"]]
            cmd += c.get("extra", [])
            cpus = [int(x) for x in (c.get("affinity") or a.host_pin).split(",")]
            before = host_cpu_state(cpus)
            r = subprocess.run(cmd, capture_output=True, text=True)
            after = host_cpu_state(cpus)
            line = (r.stdout.strip().splitlines() or [""])[-1]
            if r.returncode != 0:
                print(f"[{tag}] FAILED: {r.stderr.strip()[:300]}", file=sys.stderr)
                failures += 1
                continue
            rep_json = json.load(open(out_json))
            rep_json["host_env"] = {"before": before, "after": after}
            json.dump(rep_json, open(out_json, "w"), indent=1)
            print(f"[{tag}] {line}")
            time.sleep(a.cooldown)
    return 1 if failures else 0


def thread_plan(model_dir: str, stage: str) -> list[dict]:
    m = f"{model_dir}/model.bitino"
    c = []
    for t in (1, 2, 4, 8):
        c.append(dict(name=f"t{t}_unpinned", binary="rel", model=m, stage=stage, threads=t, affinity=""))
    c.append(dict(name="t1_x2", binary="rel", model=m, stage=stage, threads=1, affinity="7"))
    c.append(dict(name="t1_a710", binary="rel", model=m, stage=stage, threads=1, affinity="4"))
    c.append(dict(name="t1_a510", binary="rel", model=m, stage=stage, threads=1, affinity="0"))
    c.append(dict(name="t2_a710", binary="rel", model=m, stage=stage, threads=2, affinity="4,5"))
    c.append(dict(name="t3_a710", binary="rel", model=m, stage=stage, threads=3, affinity="4,5,6"))
    c.append(dict(name="t4_x2_a710", binary="rel", model=m, stage=stage, threads=4, affinity="7,4,5,6"))
    c.append(dict(name="t4_a510", binary="rel", model=m, stage=stage, threads=4, affinity="0,1,2,3"))
    c.append(dict(name="t8_all_pinned", binary="rel", model=m, stage=stage, threads=8, affinity="7,4,5,6,0,1,2,3"))
    # Thread-pool spin-wait A/B (ledger: threading). Same configs with a 200 us spin budget.
    for t, aff in ((2, ""), (4, ""), (4, "7,4,5,6"), (8, "")):
        c.append(dict(name=f"t{t}_{'pinned' if aff else 'unpinned'}_spin200", binary="rel", model=m, stage=stage, threads=t,
                      affinity=aff, extra=["--spin-us", "200"]))
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["device", "host"], default="device",
                    help="device: push to the Android device with adb (default); host: run build/host/bitino_bench here")
    ap.add_argument("--plan", choices=["stages", "threads", "quick", "custom"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--configs")
    ap.add_argument("--stage", default="all", help="stage for the threads plan")
    ap.add_argument("--rel-binary", help="default: build/android-arm64/bitino_bench (device), build/host/bitino_bench (host)")
    ap.add_argument("--o0-binary", help="default: build/android-arm64-O0/bitino_bench (device), build/host-O0/bitino_bench (host)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cooldown", type=float, help="fixed mode: seconds between runs (default: 20 device, 2 host)")
    ap.add_argument("--cooldown-mode", choices=["adaptive", "fixed"], default="adaptive",
                    help="device: adaptive = wait for the CPU temperature to return to the idle baseline (default)")
    ap.add_argument("--cool-margin", type=float, default=2.0, help="adaptive: degrees C above the idle baseline")
    ap.add_argument("--cool-min", type=float, default=3.0, help="adaptive: minimum wait before each run (s)")
    ap.add_argument("--cool-max", type=float, default=90.0, help="adaptive: maximum wait before each run (s)")
    ap.add_argument("--baseline-max-age", type=float, default=24.0, help="adaptive: hours a cached idle baseline stays valid")
    ap.add_argument("--cap-max", type=float, default=900.0,
                    help="device: maximum wait (s) for the thermal frequency caps to lift before each run; 0 = record only")
    ap.add_argument("--host-pin", default=HOST_PIN, help="host: CPU for single-thread configs (default 2, a P-core)")
    ap.add_argument("--host-pcores", default=_cpus(HOST_PCORES), help="host: one CPU per physical P-core")
    ap.add_argument("--host-ecores", default=_cpus(HOST_ECORES), help="host: E-core CPUs")
    ap.add_argument("--only", help="comma-separated config names to run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    md = os.path.join(ROOT, a.model_dir) if not os.path.isabs(a.model_dir) else a.model_dir
    if a.target == "host":
        a.cooldown = 2.0 if a.cooldown is None else a.cooldown
        binaries = {"rel": a.rel_binary or os.path.join(ROOT, "build/host/bitino_bench"),
                    "O0": a.o0_binary or os.path.join(ROOT, "build/host-O0/bitino_bench")}
        pcores = [int(x) for x in a.host_pcores.split(",")]
        ecores = [int(x) for x in a.host_ecores.split(",")]
        if a.plan == "stages":
            cfgs = host_stage_plan(md, a.host_pin)
        elif a.plan == "threads":
            cfgs = host_thread_plan(md, a.stage, a.host_pin, pcores, ecores)
        elif a.plan == "quick":
            cfgs = host_quick_plan(md, a.stage, a.host_pin, pcores)
        else:
            cfgs = json.load(open(a.configs))
        if a.only:
            keep = set(a.only.split(","))
            cfgs = [c for c in cfgs if c["name"] in keep]
        return run_host(a, cfgs, binaries)
    a.cooldown = 20.0 if a.cooldown is None else a.cooldown
    a.rel_binary = a.rel_binary or os.path.join(ROOT, "build/android-arm64/bitino_bench")
    a.o0_binary = a.o0_binary or os.path.join(ROOT, "build/android-arm64-O0/bitino_bench")
    if a.plan == "stages":
        cfgs = stage_plan(md)
    elif a.plan == "threads":
        cfgs = thread_plan(md, a.stage)
    elif a.plan == "quick":
        cfgs = quick_plan(md, a.stage)
    else:
        cfgs = json.load(open(a.configs))
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c["name"] in keep]

    adb("shell", "mkdir", "-p", f"{DEV}/bench", f"{DEV}/out")
    binaries = {"rel": a.rel_binary, "O0": a.o0_binary}
    pushed = {}
    for c in cfgs:
        if c["binary"] not in pushed:
            src = binaries[c["binary"]]
            if not os.path.exists(src):
                print(f"skip binary {c['binary']}: {src} missing", file=sys.stderr)
                continue
            dst = f"{DEV}/bench/bitino_bench_{c['binary']}"
            adb("push", src, dst)
            pushed[c["binary"]] = dst
        mp = c["model"]
        dst_m = f"{DEV}/bench/{os.path.basename(os.path.dirname(mp))}__{os.path.basename(mp)}"
        if mp not in pushed:
            adb("push", mp, dst_m)
            pushed[mp] = dst_m
    cfgs = [c for c in cfgs if c["binary"] in pushed]

    adaptive = a.cooldown_mode == "adaptive"
    plan = {"plan": a.plan, "model_dir": a.model_dir, "warmup": a.warmup, "iterations": a.iterations,
            "repeats": a.repeats, "cooldown_mode": a.cooldown_mode, "configs": cfgs,
            "binaries": {k: v for k, v in binaries.items() if k in pushed}}
    if adaptive:
        base = idle_baseline_c(a.baseline_max_age)
        target = base["temp_c"] + a.cool_margin
        plan["cooldown"] = {"idle_baseline": base, "margin_c": a.cool_margin, "target_c": target,
                            "min_s": a.cool_min, "max_s": a.cool_max}
    else:
        plan["cooldown_s"] = a.cooldown
    plan["freq_cap_gate"] = {"max_wait_s": a.cap_max, "at_plan_start": freq_caps()}
    json.dump(plan, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    for rep in range(a.repeats):
        for c in cfgs:
            tag = f"{c['name']}__r{rep}"
            # A run interrupted by a dropped USB link is repeated once the device is back; a run capped while it ran
            # is repeated once the cap lifts.
            for attempt in range(3):
                try:
                    gate = wait_uncapped(a.cap_max)
                    cool = wait_cool(target, a.cool_min, a.cool_max) if adaptive else None
                except RuntimeError:  # temperature or limits unreadable: device gone
                    if not wait_device():
                        print("device lost for 30 min; stopping", file=sys.stderr)
                        return 2
                    continue
                res = run_one(a, c, tag, pushed, cool, gate)
                if res == "ok" or (res == "throttled" and (gate["timeout"] or attempt == 2)):
                    break
                if res == "throttled":
                    continue
                if adb("get-state", check=False).stdout.strip() == "device":
                    break  # a genuine failure, not a lost device: do not repeat
                if not wait_device():
                    print("device lost for 30 min; stopping", file=sys.stderr)
                    return 2
            if not adaptive:
                time.sleep(a.cooldown)
    return 0


def run_one(a, c, tag, pushed, cool, gate) -> str:
    """One device run: "ok", "throttled" (a frequency cap was active before or after it) or "failed" (the caller
    checks whether the device is still there)."""
    dev_json = f"{DEV}/out/{tag}.json"
    cmd = [pushed[c["binary"]], "--model", pushed[c["model"]], "--stage", c["stage"], "--threads", str(c["threads"]),
           "--warmup", str(c.get("warmup", a.warmup)), "--iterations", str(c.get("iterations", a.iterations)),
           "--json", dev_json, "--label", tag]
    if c.get("affinity"):
        cmd += ["--affinity", c["affinity"]]
    cmd += c.get("extra", [])
    r = adb("shell", " ".join(cmd), check=False, lock=True)
    line = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode != 0:
        print(f"[{tag}] FAILED: {r.stderr.strip()[:300]}", file=sys.stderr)
        return "failed"
    try:
        after = freq_caps()
    except RuntimeError:
        after = None
    local = os.path.join(a.out, f"{tag}.json")
    if adb("pull", dev_json, local, check=False).returncode != 0:
        print(f"[{tag}] FAILED: pull", file=sys.stderr)
        return "failed"
    throttled = gate["before"]["capped"] or bool(after and after["capped"])
    rep_json = json.load(open(local))
    if cool:
        rep_json["cooldown"] = cool
    rep_json["throttle"] = {**gate, "after": after, "throttled": throttled}
    json.dump(rep_json, open(local, "w"), indent=1)
    x2 = (after or gate["before"])["policies"]
    cap = ", ".join(f"cpu{p} {s / 1e6:.2f}/{m / 1e6:.2f} GHz" for p, (s, m) in sorted(x2.items(), key=lambda kv: int(kv[0])))
    print(f"[{tag}] {line}" + (f" (cool-down {cool['waited_s']} s, start {cool['temp_at_start_c']:.1f} C"
                               + (", TIMEOUT" if cool["timeout"] else "") + ")" if cool else "")
          + (f" THROTTLED ({cap})" if throttled else "") + (f" (cap wait {gate['waited_s']} s)" if gate["waited_s"] > 1 else ""))
    return "throttled" if throttled else "ok"


if __name__ == "__main__":
    sys.exit(main())
