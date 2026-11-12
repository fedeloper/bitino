#!/usr/bin/env python3
"""Aggregates raw bitino_bench JSON reports (from tools/benchmark.py) into tables.

For each configuration: median of per-repeat medians, pooled mean/p90/p95 over all timed
iterations, spread of per-repeat medians (min..max), speedups vs. the previous
configuration in plan order and vs. the first one, kernels used, memory, temperatures and
frequencies observed. Writes summary.json and summary.md into the run directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def pct(v, p):
    v = sorted(v)
    if not v:
        return None
    i = p * (len(v) - 1)
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", help="config name used as baseline (default: first in plan)")
    a = ap.parse_args()
    plan = json.load(open(os.path.join(a.run_dir, "plan.json")))
    rows = []
    all_reps: list[dict] = []
    for c in plan["configs"]:
        reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.run_dir, f"{c['name']}__r*.json")))]
        # A pinned configuration whose thread was still unpinned in the timed iterations (bitino_bench retries the
        # pin at every inference; more failures than warmup + 1 mean the timed part ran unpinned, e.g. while the
        # vendor kept the requested core paused) measured some other core: such repeats are excluded.
        n_all = len(reps)
        if c.get("affinity"):
            reps = [r for r in reps if (r["config"].get("affinity_errors") or 0) <= r["config"].get("warmup", 0) + 1]
        unpinned = n_all - len(reps)
        all_reps.extend(reps)
        if not reps:
            rows.append({"name": c["name"], "status": "no data" if not n_all else "no pinned repeat", "unpinned_repeats": unpinned})
            continue
        meds = [r["latency"]["median_ms"] for r in reps]
        pooled = [s for r in reps for s in r["latency"]["samples_ms"]]
        temps = [t for r in reps for k in ("thermal_before_c", "thermal_after_c") for t in r.get(k, {}).values()]
        # Default CPU of unpinned configs: the X2 (cpu7) on the device, plan["default_cpu"] for host runs.
        pin = c.get("affinity") or str(plan.get("default_cpu", "7"))
        fx = [f["cur_khz"] for r in reps for f in r.get("cpu_freq_after", []) if str(f["cpu"]) in pin.split(",")]
        r0 = reps[0]
        rows.append({
            "name": c["name"], "stage": c["stage"], "threads": c["threads"], "affinity": c.get("affinity", ""),
            "model_file": os.path.basename(c["model"]), "extra": c.get("extra", []),
            "repeats": len(reps), "iterations_per_repeat": r0["config"]["iterations"], "warmup": r0["config"]["warmup"],
            "median_ms": statistics.median(meds), "median_min_ms": min(meds), "median_max_ms": max(meds),
            "mean_ms": statistics.mean(pooled), "p90_ms": pct(pooled, 0.9), "p95_ms": pct(pooled, 0.95),
            "stdev_ms": statistics.pstdev(pooled), "throughput_per_s": 1000.0 / statistics.median(meds),
            "kernels": r0["kernels"], "prepare_ms": r0["latency"]["session_prepare_ms"], "load_ms": r0["latency"]["load_ms"],
            "activation_bytes": r0["memory"]["activation_bytes"], "prepacked_bytes": r0["memory"]["prepacked_bytes"],
            "vm_hwm_kb": max(r["memory"]["vm_hwm_kb"] for r in reps), "model_file_bytes": r0["model"]["file_bytes"],
            "checkpoint_sha256": r0["model"]["manifest"].get("source", {}).get("checkpoint_sha256"),
            "model_data_sha256": r0["model"]["data_sha256"], "affinity_errors": r0["config"]["affinity_errors"],
            "unpinned_repeats": unpinned,
            "temp_c_range": [min(temps), max(temps)] if temps else None,
            "pinned_cpu_freq_khz_after": [min(fx), max(fx)] if fx else None,
            "build": r0["build"], "device": r0["device"].get("model") or (r0["device"].get("cpu_model") or "").strip(),
            "timestamp_first": r0["timestamp_utc"],
            # tools/benchmark.py frequency-cap gate; None for runs made before the gate existed
            "throttled_repeats": sum(bool(r["throttle"]["throttled"]) for r in reps) if all("throttle" in r for r in reps) else None,
        })
    ok = [r for r in rows if "median_ms" in r]
    base = next((r for r in ok if r["name"] == a.baseline), ok[0] if ok else None)
    prev = None
    for r in ok:
        r["speedup_vs_baseline"] = base["median_ms"] / r["median_ms"]
        r["speedup_vs_previous"] = prev["median_ms"] / r["median_ms"] if prev else 1.0
        prev = r
    json.dump({"plan": plan, "rows": rows}, open(os.path.join(a.run_dir, "summary.json"), "w"), indent=1)

    host = plan.get("host")
    where = (f"Host (not the device): {host.get('cpu_model')}, governor {host.get('governor_pinned_cpu')}, "
             f"no_turbo={host.get('intel_pstate_no_turbo')}, max {host.get('scaling_max_khz_pinned_cpu')} kHz, "
             f"load at start {host.get('loadavg_at_start')}") if host else f"Device: {ok[0]['device'] if ok else '?'}"
    cd = plan.get("cooldown")
    if cd:  # adaptive (tools/benchmark.py --cooldown-mode adaptive)
        waits = [r["cooldown"]["waited_s"] for r in all_reps if "cooldown" in r]
        starts = [r["cooldown"]["temp_at_start_c"] for r in all_reps if "cooldown" in r]
        touts = sum(bool(r["cooldown"]["timeout"]) for r in all_reps if "cooldown" in r)
        cool = (f"adaptive cool-down to {cd['target_c']:.1f} °C (idle baseline {cd['idle_baseline']['temp_c']:.1f} + "
                f"{cd['margin_c']:g}; wait {cd['min_s']:g}–{cd['max_s']:g} s" +
                (f": median {statistics.median(waits):.0f} s, max {max(waits):.0f} s; start temperature "
                 f"{min(starts):.1f}–{max(starts):.1f} °C; {touts} timeouts" if waits else "") + ")")
    else:
        cool = f"cooldown {plan.get('cooldown_s', '?')} s"
    gated = [r["throttle"] for r in all_reps if "throttle" in r]
    if gated:
        n_thr = sum(bool(g["throttled"]) for g in gated)
        waits = [g["waited_s"] for g in gated]
        cool += (f"; frequency-cap gate (wait ≤ {plan['freq_cap_gate']['max_wait_s']:g} s for scaling_max = cpuinfo_max): "
                 f"wait median {statistics.median(waits):.0f} s, max {max(waits):.0f} s, "
                 f"{sum(bool(g['timeout']) for g in gated)} timeouts, **{n_thr} of {len(gated)} runs throttled**")
    elif plan.get("host") is None:
        cool += "; no frequency-cap record (run made before the gate; thermal caps unknown)"
    lines = [f"# Benchmark summary: `{plan['model_dir']}` ({plan['plan']})", "",
             f"{where}; {plan['repeats']} repeats x {plan['iterations']} iterations (per-config overrides in plan.json) "
             f"(warmup {plan['warmup']}), {cool}. First run {ok[0]['timestamp_first'] if ok else '?'}.", "",
             "Median = median of per-repeat medians; range = min..max of per-repeat medians; p90/p95 pooled over all iterations.", "",
             "| config | stage | thr | affinity | median ms | range | mean | p95 | speedup vs prev | vs baseline | kernels | temp °C |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "median_ms" not in r:
            lines.append(f"| {r['name']} | | | | no data | | | | | | | |")
            continue
        ks = ", ".join(f"{k}×{v}" for k, v in sorted(r["kernels"].items()) if not k.endswith(".reference") or "conv" in k)
        t = f"{r['temp_c_range'][0]:.0f}–{r['temp_c_range'][1]:.0f}" if r["temp_c_range"] else "n/a"
        thr = f" ⚠ throttled {r['throttled_repeats']}/{r['repeats']}" if r.get("throttled_repeats") else ""
        lines.append(f"| {r['name']}{thr} | {r['stage']} | {r['threads']} | {r['affinity'] or '-'} | {r['median_ms']:.2f} | "
                     f"{r['median_min_ms']:.2f}–{r['median_max_ms']:.2f} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} | "
                     f"{r['speedup_vs_previous']:.2f}× | {r['speedup_vs_baseline']:.2f}× | {ks} | {t} |")
    if ok:
        lines += ["", f"Build: `{ok[0]['build']['compiler']}`, flags `{ok[0]['build']['flags'].strip()}`.",
                  f"Model data SHA-256 `{ok[0]['model_data_sha256'][:16]}…`, checkpoint SHA-256 `{(ok[0]['checkpoint_sha256'] or '?')[:16]}…`."]
    open(os.path.join(a.run_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
