#!/usr/bin/env python3
"""Native x86-64 latency of every converted model on this host (real timings, unlike qemu).

Configurations per model (bitino_bench, stage names as on the device):
  t1        stage optimized (final exact config), 1 thread pinned to a P-core
  t8_pcore  stage optimized, 8 threads, one per physical P-core
  t1_int8   stage optimized_int8, 1 thread pinned
  t1_base   optional --baseline binary (e.g. the build before the x86 kernels), stage optimized, 1 thread pinned
            (--baseline-configs t1,t1_int8,t8_pcore adds the baseline binary for each listed configuration:
            <name>_base, interleaved with the others)
Repeats run round-robin over all configurations of a model; the reported value is the median of the
per-repeat medians, with min-max of those medians. Iterations per repeat are sized from a probe run to
~--target-ms of work (5..200).

  python3 tools/bench_host_all.py --baseline build/host-baseline-81aad90/bitino_bench \
      --out reports/benchmarks_host/all_models_$(date +%Y%m%d_%H%M)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(p: str, default: str = "?") -> str:
    try:
        return open(p).read().strip()
    except OSError:
        return default


def run(bench: str, model: str, stage: str, threads: int, affinity: str, warmup: int, iters: int, js: str) -> float:
    cmd = [bench, "--model", model, "--stage", stage, "--threads", str(threads), "--warmup", str(warmup),
           "--iterations", str(iters), "--affinity", affinity, "--json", js]
    subprocess.run(cmd, check=True, capture_output=True)
    return float(json.load(open(js))["latency"]["median_ms"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(ROOT, "build/host/bitino_bench"))
    ap.add_argument("--baseline", help="second bitino_bench binary for a before/after column")
    ap.add_argument("--baseline-configs", default="t1", help="configurations also run with --baseline (t1,t1_int8,t8_pcore)")
    ap.add_argument("--models", nargs="*", help="default: every artifacts/*/model.bitino with a validation")
    ap.add_argument("--pin", default="2")
    ap.add_argument("--pcores", default="0,2,4,6,8,10,12,14")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--target-ms", type=float, default=1500.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--render-only", action="store_true", help="rewrite summary.md from <out>/results.json")
    a = ap.parse_args()
    if a.render_only:
        d = json.load(open(os.path.join(a.out, "results.json")))
        return render(a, d["host"], d["results"])
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    models = a.models or sorted(d for d in os.listdir(os.path.join(ROOT, "artifacts"))
                                if os.path.isfile(os.path.join(ROOT, "artifacts", d, "model.bitino")))
    configs = [("t1", a.bench, "optimized", 1, a.pin), ("t8_pcore", a.bench, "optimized", 8, a.pcores),
               ("t1_int8", a.bench, "optimized_int8", 1, a.pin)]
    if a.baseline:
        for base in a.baseline_configs.split(","):
            c = next(c for c in configs if c[0] == base)
            configs.append((base + "_base", a.baseline, c[2], c[3], c[4]))
    host = {"cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
            "kernel": platform.release(),
            "governor": read("/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor"),
            "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            "scaling_max_freq_khz": read("/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq"),
            "loadavg_start": read("/proc/loadavg"), "bench": a.bench, "baseline": a.baseline,
            "pin": a.pin, "pcores": a.pcores, "repeats": a.repeats,
            "baseline_configs": a.baseline_configs if a.baseline else ""}
    results = {}
    for m in models:
        mp = os.path.join(ROOT, "artifacts", m, "model.bitino")
        r = {}
        try:
            plan = []
            for name, bench, stage, th, aff in configs:
                probe = run(bench, mp, stage, th, aff, 1, 2, os.path.join(a.out, "raw", f"{m}_{name}_probe.json"))
                plan.append((name, bench, stage, th, aff, max(5, min(200, int(a.target_ms / max(probe, 1e-3))))))
            per = {p[0]: [] for p in plan}
            for rep in range(a.repeats):
                for name, bench, stage, th, aff, it in plan:
                    per[name].append(run(bench, mp, stage, th, aff, max(2, it // 10), it,
                                         os.path.join(a.out, "raw", f"{m}_{name}_r{rep}.json")))
            for name, bench, stage, th, aff, it in plan:
                v = per[name]
                r[name] = {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v), "iterations": it,
                           "stage": stage, "threads": th, "affinity": aff}
        except subprocess.CalledProcessError as e:
            r["error"] = (e.stderr or b"").decode()[-300:]
        results[m] = r
        line = ", ".join(f"{k} {v['median_ms']:.3f}" for k, v in r.items() if isinstance(v, dict))
        print(f"{m}: {line or r.get('error')}", flush=True)
    host["loadavg_end"] = read("/proc/loadavg")
    json.dump({"schema": "bitino.host_latency.v1", "host": host, "results": results},
              open(os.path.join(a.out, "results.json"), "w"), indent=1)
    return render(a, host, results)


def render(a, host: dict, results: dict) -> int:
    a.baseline = a.baseline or host.get("baseline")
    a.pin, a.pcores = host.get("pin", a.pin), host.get("pcores", a.pcores)
    bases = (host.get("baseline_configs") or "t1").split(",") if a.baseline else []
    extra = [b for b in bases if b != "t1"]

    def f(r, k):
        if k not in r:
            return "–"
        v = r[k]["median_ms"]
        return f"{v:,.2f}" if v >= 1 else f"{v:.3g}"
    L = ["# Native x86-64 latency, all converted models", "",
         f"Host: {host['cpu']}, governor {host['governor']}, turbo {'off' if host['no_turbo'] == '1' else 'on'}, "
         f"max {int(host['scaling_max_freq_khz']) / 1e6 if host['scaling_max_freq_khz'].isdigit() else '?'} GHz. "
         f"Load average at start / end: {host['loadavg_start'].split()[0]} / {host['loadavg_end'].split()[0]}."
         + (f" Note: {host['note']}." if host.get("note") else ""), "",
         "Median of per-repeat medians (ms). `optimized` is the final exact configuration; t1 pinned to P-core CPU "
         f"{a.pin}; t8 = one thread per physical P-core ({a.pcores}). These are x86 numbers, not phone numbers.", "",
         "| model | t1 | t8 P-cores | t1 INT8 |" + (" t1 before x86 kernels | speedup |" if "t1" in bases else "")
         + "".join(f" {b} baseline | {b} speedup |" for b in extra),
         "|---|---|---|---|" + ("---|---|" if "t1" in bases else "") + "---|---|" * len(extra)]

    def sp(r, k):
        return f"{r[k + '_base']['median_ms'] / r[k]['median_ms']:.2f}×" if k in r and k + "_base" in r else "–"
    for m, r in sorted(results.items(), key=lambda kv: kv[1].get("t1", {}).get("median_ms", 1e18)):
        row = f"| {m} | {f(r, 't1')} | {f(r, 't8_pcore')} | {f(r, 't1_int8')} |"
        if "t1" in bases:
            row += f" {f(r, 't1_base')} | {sp(r, 't1')} |"
        for b in extra:
            row += f" {f(r, b + '_base')} | {sp(r, b)} |"
        L.append(row)
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
