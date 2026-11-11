#!/usr/bin/env python3
"""Adds measured speed runs to results/ (schema in results/README.md).

  python3 tools/ingest_speed.py --device oneplus-open reports/benchmarks/*/quick_20261111_1500_cph2551
  python3 tools/ingest_speed.py --s22 <dir>            # same as --device s22 <dir>
  python3 tools/ingest_speed.py --x86 <results.json written by tools/bench_host_all.py> [--date YYYY-MM-DD]
  python3 tools/build_site.py --readme-only            # then refresh the tables

Phones: one record per configuration row of <dir>/summary.json (dirs written by tools/benchmark.py +
tools/report_benchmarks.py; shell globs and quoted glob patterns both work), stored in results/speed_<device>.json
and in results/models/<model>.json -> speed.<device>. The device must be listed in results/devices.json (id, name,
SoC, cores, cpufreq policy names, ABI, product models); every raw run's recorded `device.model` must be one of that
entry's `product_models`, so runs of two phones cannot be mixed up. An unknown device id is added to devices.json
from the recorded device block (`--add-device`), to be completed by hand.

Validity of a run, from the raw per-repeat JSONs (<config>__r<k>.json, "throttle" block of tools/benchmark.py):
  clean              every repeat ran with every readable cpufreq policy at its maximum frequency
  vendor-policy      a clock ceiling was set while Android reported no thermal event (thermal status 0): the
                     phone's own power policy (e.g. screen off), steady, recorded per run in caps_seen
  thermal-throttled  a frequency cap with a thermal status > 0 (or of unknown cause) during at least one repeat
  unknown-caps       the run predates the frequency-cap check
(`throttled` in older files means thermal-throttled.) Runs already present (same model, plan and config) are
replaced. Never ingest emulated runs: the script refuses reports whose device is not a physical phone (no
"device.model") or whose path says "emulated".
"""
from __future__ import annotations

import argparse
import ast
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
SCHEMA = "bitino.results.v1"
VALIDITY_ORDER = ["clean", "vendor-policy", "thermal-throttled", "unknown-caps"]   # worst repeat wins
PLAN_DIR = re.compile(r"^(?P<plan>.+?)_(?P<ts>\d{8}_\d{4})(?:_(?P<tag>[A-Za-z0-9.-]+))?$")


def load(p, default=None):
    if default is not None and not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def save(p, doc) -> None:
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")


def lit(v):
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


# ----------------------------------------------------------------------------------------------- devices
def load_devices(res: str = RES) -> dict:
    return load(os.path.join(res, "devices.json"), {"schema": SCHEMA + ".devices", "devices": {}})


def caps_text(pol: dict | None, dev: dict | None = None) -> str | None:
    """'X3 2.23/3.19 GHz, A715/A710 1.79/2.80 GHz, ...': [scaling_max, cpuinfo_max] per cpufreq policy, the policy
    named after its cores in devices.json (cpuN when unknown); fastest policy first."""
    if not pol:
        return None
    names = (dev or {}).get("policies") or {}
    return ", ".join(f"{names.get(k, 'cpu' + k)} {c / 1e6:.2f}/{m / 1e6:.2f} GHz"
                     for k, (c, m) in sorted(pol.items(), key=lambda kv: int(kv[0]), reverse=True))


def classify_repeat(th) -> dict:
    """One raw repeat's throttle block -> {checked, validity, policies, thermal_status}."""
    th = lit(th)
    if not isinstance(th, dict):
        return {"checked": False, "validity": "unknown-caps"}
    sides = [s for s in (th.get("before"), th.get("after")) if isinstance(s, dict)]
    capped = bool(th.get("throttled")) or any(s.get("capped") for s in sides)
    status = [s.get("thermal_status") for s in sides if s.get("thermal_status") is not None]
    pol = next((s.get("policies") for s in (th.get("after"), th.get("before")) if isinstance(s, dict) and s.get("policies")), None)
    if not capped:
        v = "clean"
    elif status and max(status) == 0:
        v = "vendor-policy"
    else:
        v = "thermal-throttled"
    return {"checked": True, "validity": v, "policies": pol if capped else None,
            "thermal_status": max(status) if status else None}


def worst(validities) -> str:
    vs = list(validities)
    return max(vs, key=VALIDITY_ORDER.index) if vs else "unknown-caps"


def device_of_raw(raw: dict) -> dict:
    return lit(raw.get("device")) or {}


def match_device(devices: dict, product_model: str | None, tag: str | None) -> str | None:
    """Device id for a run: the recorded product model first, then the DEVICE_TAG suffix of the plan directory."""
    pm = (product_model or "").strip().lower()
    for did, d in devices.items():
        if pm and pm in [x.lower() for x in d.get("product_models", [])]:
            return did
    for did, d in devices.items():
        if tag and tag.lower() in [x.lower() for x in d.get("tags", [])]:
            return did
    return None


def new_device_entry(raw_dev: dict) -> dict:
    feats = raw_dev.get("cpu_features") or ""
    return {"kind": "phone", "name": raw_dev.get("model"), "short": raw_dev.get("model"), "soc": raw_dev.get("soc"),
            "android": raw_dev.get("android_release"), "abi": "arm64-v8a" if feats.startswith("aarch64") else None,
            "kernels": "neon", "isa": feats.split(":", 1)[-1].strip() or None, "cores": [], "policies": {},
            "fastest_cpu": "7", "product_models": [raw_dev.get("model")], "tags": [], "notes": ["added by ingest_speed.py: complete by hand"]}


# ----------------------------------------------------------------------------------------------- phone runs
def phone_runs(d: str, device_id: str, dev: dict | None, model_id: str | None = None) -> list[dict]:
    """Records of one benchmark plan directory measured on the phone `device_id` (metadata `dev`)."""
    if "emulat" in d.lower():
        raise SystemExit("refusing to ingest an emulated run")
    summ = load(os.path.join(d, "summary.json"))
    mid = model_id or os.path.basename(os.path.normpath(os.path.dirname(os.path.normpath(d))))
    plan = os.path.basename(os.path.normpath(d))
    out = []
    for row in summ.get("rows", []):
        if not row.get("median_ms"):
            continue   # listed in the plan, never measured
        raws = [load(p) for p in sorted(glob.glob(os.path.join(d, row["name"] + "__r*.json")))]
        rd = device_of_raw(raws[0]) if raws else {}
        if raws and not rd.get("model"):
            raise SystemExit(f"{d}/{row['name']}: no physical device recorded; refusing")
        if dev and rd.get("model") and dev.get("product_models") and \
                rd["model"].lower() not in [x.lower() for x in dev["product_models"]]:
            raise SystemExit(f"{d}: measured on {rd['model']!r}, not on {device_id} ({dev.get('product_models')})")
        cls = [classify_repeat(r.get("throttle")) for r in raws]
        checked = bool(cls) and all(c["checked"] for c in cls)
        validity = worst(c["validity"] for c in cls) if checked else "unknown-caps"
        caps = sorted({caps_text(c.get("policies"), dev) for c in cls if c.get("policies")} - {None})
        out.append({"model_id": mid, "device": device_id, "date": (row.get("timestamp_first") or "")[:10] or None,
                    "timestamp_utc": row.get("timestamp_first"), "plan": plan, "config": row["name"],
                    "model_file": row.get("model_file"), "stage": row.get("stage"),
                    "threads": row.get("threads"), "affinity": row.get("affinity") or "unpinned",
                    "median_ms": row.get("median_ms"), "median_range_ms": [row.get("median_min_ms"), row.get("median_max_ms")],
                    "p90_ms": row.get("p90_ms"), "repeats": row.get("repeats"), "iterations": row.get("iterations_per_repeat"),
                    "warmup": row.get("warmup"), "peak_rss_kb": row.get("vm_hwm_kb"),
                    "validity": validity, "caps_seen": caps or None, "notes": None,
                    "abi": (dev or {}).get("abi"), "kernels": (dev or {}).get("kernels"),
                    "model_data_sha256": row.get("model_data_sha256"),
                    "runtime": f"bitino {(raws[0].get('bitino_version') or raws[0].get('bitino_version') or '?') if raws else '?'}"})
    return out


def x86_runs(p: str, date: str) -> tuple[dict, list[dict]]:
    d = load(p)
    runs = []
    for mid, cfgs in d.get("results", {}).items():
        for name, r in cfgs.items():
            runs.append({"model_id": mid, "device": "x86", "date": date, "suite": os.path.basename(os.path.dirname(p)), "config": name,
                         "model_file": r.get("model_file", "model.bitino"), "stage": r.get("stage"), "threads": r.get("threads"),
                         "affinity": r.get("affinity"), "median_ms": r.get("median_ms"),
                         "median_range_ms": [r.get("min_ms"), r.get("max_ms")], "iterations": r.get("iterations"),
                         "validity": "clean", "notes": None})
    return d.get("host", {}), runs


def run_key(r: dict) -> tuple:
    return (r["model_id"], r.get("plan") or r.get("suite"), r.get("config"))


def speed_doc(device_id: str, dev: dict) -> dict:
    return {"schema": SCHEMA + ".speed", "device_id": device_id, "device": dev,
            "method": {"tool": "bitino_bench (tools/benchmark.py on the device over adb)",
                       "statistic": "median latency of one inference (batch 1) per repeat; median of repeats",
                       "validity": "see results/README.md"},
            "runs": []}


def merge(res: str, device_id: str, new: list[dict], host: dict | None = None, dev: dict | None = None) -> None:
    path = os.path.join(res, f"speed_{device_id}.json")
    doc = load(path, speed_doc(device_id, dev or {}))
    keep = {run_key(r) for r in new}
    doc["runs"] = [r for r in doc.get("runs", []) if run_key(r) not in keep] + new
    if host:
        doc["host"] = host
    doc["generated"] = datetime.date.today().isoformat()
    save(path, doc)
    for mid in sorted({r["model_id"] for r in new}):
        mp = os.path.join(res, "models", f"{mid}.json")
        if not os.path.exists(mp):
            print(f"warning: no results/models/{mid}.json; run only kept in {os.path.basename(path)}", file=sys.stderr)
            continue
        m = load(mp)
        sp = m.setdefault("speed", {}).setdefault(device_id, [])
        mine = [r for r in new if r["model_id"] == mid]
        mk = {run_key(r) for r in mine}
        m["speed"][device_id] = [r for r in sp if run_key(r) not in mk] + mine
        save(mp, m)


def expand(dirs: list[str]) -> list[str]:
    out = []
    for d in dirs:
        out += sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d]
    return [d for d in out if os.path.isfile(os.path.join(d, "summary.json"))]


def ingest_device(res: str, device_id: str, dirs: list[str], model_id: str | None = None, add: bool = False) -> int:
    devs = load_devices(res)
    dev = devs["devices"].get(device_id)
    dirs = expand(dirs)
    if dev is None:
        raws = [p for d in dirs for p in sorted(glob.glob(os.path.join(d, "*__r*.json")))[:1]]
        if not add or not raws:
            raise SystemExit(f"unknown device {device_id!r}: add it to results/devices.json (or pass --add-device)")
        dev = devs["devices"][device_id] = new_device_entry(device_of_raw(load(raws[0])))
        save(os.path.join(res, "devices.json"), devs)
        print(f"added {device_id} to results/devices.json from the recorded device block: complete name, cores, policies",
              file=sys.stderr)
    runs = [r for d in dirs for r in phone_runs(d, device_id, dev, model_id)]
    merge(res, device_id, runs, dev=dev)
    return len(runs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", nargs="+", metavar=("ID", "DIR"), help="device id (results/devices.json) and plan dirs")
    ap.add_argument("--s22", nargs="*", default=[], help="plan dirs measured on the Galaxy S22 (= --device s22 DIR...)")
    ap.add_argument("--x86")
    ap.add_argument("--model-id")
    ap.add_argument("--add-device", action="store_true", help="create an unknown --device entry from the recorded device block")
    ap.add_argument("--results", default=RES, help="results directory (default: results/ of this tree)")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()
    n = 0
    if a.s22:
        n += ingest_device(a.results, "s22", a.s22, a.model_id)
    if a.device:
        if len(a.device) < 2:
            raise SystemExit("--device ID DIR [DIR ...]")
        n += ingest_device(a.results, a.device[0], a.device[1:], a.model_id, a.add_device)
    if a.x86:
        host, runs = x86_runs(a.x86, a.date)
        merge(a.results, "x86", runs, host)
        n += len(runs)
    print(f"ingested {n} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
