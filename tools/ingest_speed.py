#!/usr/bin/env python3
"""Adds measured speed runs to results/ (schema in results/README.md).

  python3 tools/ingest_speed.py --s22 <dir written by tools/benchmark.py + tools/report_benchmarks.py> [--model-id ID]
  python3 tools/ingest_speed.py --x86 <results.json written by tools/bench_host_all.py> [--date YYYY-MM-DD]
  python3 tools/build_site.py --readme-only        # then refresh the tables

S22: one record per configuration row of <dir>/summary.json; the throttle state comes from the raw per-repeat
JSONs (<config>__r<k>.json, "throttle" block written by tools/benchmark.py): `clean` only if every repeat ran
uncapped. Runs already present (same model, plan and config) are replaced. Never ingest emulated runs: the script
refuses reports whose device is not a physical phone (no "device.model") or whose path says "emulated".
"""
from __future__ import annotations

import argparse
import ast
import datetime
import glob
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def load(p):
    with open(p) as f:
        return json.load(f)


def lit(v):
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def caps_text(pol):
    if not pol:
        return None
    names = {"0": "A510", "4": "A710", "7": "X2"}
    return ", ".join(f"{names.get(k, 'cpu' + k)} {c / 1e6:.2f}/{m / 1e6:.2f} GHz" for k, (c, m) in sorted(pol.items(), reverse=True))


def s22_runs(d: str, model_id: str | None) -> list[dict]:
    if "emulat" in d.lower():
        raise SystemExit("refusing to ingest an emulated run")
    summ = load(os.path.join(d, "summary.json"))
    mid = model_id or os.path.basename(os.path.normpath(os.path.dirname(os.path.normpath(d))))
    plan = os.path.basename(os.path.normpath(d))
    out = []
    for row in summ.get("rows", []):
        raws = [load(p) for p in sorted(glob.glob(os.path.join(d, row["name"] + "__r*.json")))]
        if raws and not (lit(raws[0].get("device")) or {}).get("model"):
            raise SystemExit(f"{row['name']}: no physical device recorded; refusing")
        ths = [lit(r.get("throttle")) for r in raws]
        checked = bool(ths) and all(isinstance(t, dict) for t in ths)
        throttled = any(t.get("throttled") for t in ths) if checked else None
        caps = sorted({caps_text((t.get("after") or {}).get("policies")) for t in ths if isinstance(t, dict) and t.get("throttled")} - {None})
        out.append({"model_id": mid, "date": (row.get("timestamp_first") or "")[:10], "timestamp_utc": row.get("timestamp_first"),
                    "plan": plan, "config": row["name"], "model_file": row.get("model_file"), "stage": row.get("stage"),
                    "threads": row.get("threads"), "affinity": row.get("affinity") or "unpinned",
                    "median_ms": row.get("median_ms"), "median_range_ms": [row.get("median_min_ms"), row.get("median_max_ms")],
                    "p90_ms": row.get("p90_ms"), "repeats": row.get("repeats"), "iterations": row.get("iterations_per_repeat"),
                    "warmup": row.get("warmup"), "peak_rss_kb": row.get("vm_hwm_kb"),
                    "validity": ("throttled" if throttled else "clean") if checked else "unknown-caps",
                    "caps_seen": caps or None, "notes": None, "model_data_sha256": row.get("model_data_sha256"),
                    "runtime": f"bitino {raws[0].get('bitino_version', '?') if raws else '?'}"})
    return out


def x86_runs(p: str, date: str) -> tuple[dict, list[dict]]:
    d = load(p)
    runs = []
    for mid, cfgs in d.get("results", {}).items():
        for name, r in cfgs.items():
            runs.append({"model_id": mid, "date": date, "suite": os.path.basename(os.path.dirname(p)), "config": name,
                         "model_file": r.get("model_file", "model.bitino"), "stage": r.get("stage"), "threads": r.get("threads"),
                         "affinity": r.get("affinity"), "median_ms": r.get("median_ms"),
                         "median_range_ms": [r.get("min_ms"), r.get("max_ms")], "iterations": r.get("iterations"),
                         "validity": "clean", "notes": None})
    return d.get("host", {}), runs


def merge(path: str, new: list[dict], host: dict | None = None) -> None:
    doc = load(path)
    key = lambda r: (r["model_id"], r.get("plan") or r.get("suite"), r.get("config"))
    keep = {key(r) for r in new}
    doc["runs"] = [r for r in doc.get("runs", []) if key(r) not in keep] + new
    if host:
        doc["host"] = host
    doc["generated"] = datetime.date.today().isoformat()
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    which = "s22" if "s22" in os.path.basename(path) else "x86"
    for mid in {r["model_id"] for r in new}:
        mp = os.path.join(RES, "models", f"{mid}.json")
        if not os.path.exists(mp):
            print(f"warning: no results/models/{mid}.json; run only kept in {os.path.basename(path)}", file=sys.stderr)
            continue
        m = load(mp)
        sp = m.setdefault("speed", {}).setdefault(which, [])
        mine = [r for r in new if r["model_id"] == mid]
        mk = {key(r) for r in mine}
        m["speed"][which] = [r for r in sp if key(r) not in mk] + mine
        with open(mp, "w") as f:
            json.dump(m, f, indent=1, ensure_ascii=False)
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s22", nargs="*", default=[])
    ap.add_argument("--x86")
    ap.add_argument("--model-id")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()
    n = 0
    for d in a.s22:
        runs = s22_runs(d, a.model_id)
        merge(os.path.join(RES, "speed_s22.json"), runs)
        n += len(runs)
    if a.x86:
        host, runs = x86_runs(a.x86, a.date)
        merge(os.path.join(RES, "speed_x86.json"), runs, host)
        n += len(runs)
    print(f"ingested {n} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
