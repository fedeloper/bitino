#!/usr/bin/env python3
"""Validates community speed submissions (the JSON written by scripts/bench_my_phone.sh) and adds them to results/community/.

  python3 tools/ingest_community.py bitino-bench-cph2551-2026-11-20.json --issue 123
  python3 tools/ingest_community.py --check bitino-bench-*.json     # validate and print, write nothing
  python3 tools/build_site.py --readme-only                         # then refresh the generated pages

Checks (any failure refuses the whole file; nothing is written):
  schema        "bitino.community_bench.v1", a known tool version
  device        phone model, SoC (or board platform), Android release, ABI arm64-v8a / armeabi-v7a, cpufreq policies
  physical      no emulator marker (ro.kernel.qemu / ro.boot.qemu, goldfish / ranchu / cutf hardware, sdk_gphone
                models, "emulator" build characteristics); every raw report names the same ro.product.model (read by
                the runtime itself on the phone)
  privacy       no field that could identify the phone or its owner (serial number, IMEI, Android ID, accounts,
                build fingerprint, host name, ...) carries a value
  models        every timed file is a published release file (model.data_sha256 listed in registry/release_models.json
                for that model and file) and the runtime verified its hash
  consistency   each raw report matches its run (stage, threads, affinity, iterations); its median recomputed from
                latency.samples_ms equals latency.median_ms; the submission's summary equals the medians recomputed here
  plausibility  0.05 ms < median < 60 s; 4 threads at most 4.6x faster than 1 thread; int8_full / float of the same
                model within 0.05..20
  outputs       an output check whose output_sha256 differs from its expected_sha256 refuses the file (wrong results)

A repeat of a pinned configuration whose thread stayed unpinned (affinity_errors > warmup + 1) is dropped, as in
tools/ingest_speed.py. Validity per repeat from its throttle block with the rules of tools/ingest_speed.py (clean /
vendor-policy / thermal-throttled); the worst repeat wins. Notes (not refusals): repeats more than 1.5x apart, and
medians more than 2x away from the maintainers' own runs of the same phone model (results/speed_<device>.json).

Output, labelled "community" (schema in results/README.md):
  results/community/speed_<device-id>.json   device id "community-<ro.product.model>"; device block, submissions, one
                                             run record per (file, configuration) per submission; re-ingesting the
                                             same file replaces its runs
  results/community/index.json               devices with submission and run counts and their headline numbers
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import ingest_speed as IS   # noqa: E402  (classify_repeat, worst, caps_text, load, save)

SCHEMA_IN = "bitino.community_bench.v1"
SCHEMA_OUT = "bitino.results.v1.community"
TOOL_VERSIONS = {1}
ABIS = {"arm64-v8a": "neon", "armeabi-v7a": "neon-aarch32"}
CONFIGS = {"t1_fast", "t4_fast"}
HEADLINE = ("ab-bnn-imagenet-reactnet18", "model.bitino")
EMU_HW = re.compile(r"^(goldfish|ranchu|vbox86|cutf)", re.I)
EMU_MODEL = re.compile(r"sdk_gphone|emulator|sdk built for", re.I)
PRIVATE_KEY = re.compile(r"serial|imei|meid|imsi|iccid|android_id|advertising|account|e-?mail|phone_?number|"
                         r"fingerprint|hostname|mac_?addr|bt_?addr|wifi_?addr|user_?name", re.I)
CPU_PARTS = {  # (implementer, part) -> core name; implementer 0x41 = Arm, 0x51 = Qualcomm
    ("0x41", "0xd03"): "A53", ("0x41", "0xd04"): "A35", ("0x41", "0xd05"): "A55", ("0x41", "0xd07"): "A57",
    ("0x41", "0xd08"): "A72", ("0x41", "0xd09"): "A73", ("0x41", "0xd0a"): "A75", ("0x41", "0xd0b"): "A76",
    ("0x41", "0xd0d"): "A77", ("0x41", "0xd41"): "A78", ("0x41", "0xd44"): "X1", ("0x41", "0xd46"): "A510",
    ("0x41", "0xd47"): "A710", ("0x41", "0xd48"): "X2", ("0x41", "0xd4d"): "A715", ("0x41", "0xd4e"): "X3",
    ("0x41", "0xd80"): "A520", ("0x41", "0xd81"): "A720", ("0x41", "0xd82"): "X4", ("0x41", "0xd85"): "X925",
    ("0x41", "0xd87"): "A725", ("0x41", "0xd8e"): "A725",
    ("0x51", "0x800"): "Kryo Gold (A73)", ("0x51", "0x801"): "Kryo Silver (A53)", ("0x51", "0x802"): "Kryo Gold (A75)",
    ("0x51", "0x803"): "Kryo Silver (A55)", ("0x51", "0x804"): "Kryo Gold (A76)", ("0x51", "0x805"): "Kryo Silver (A55)",
}


class Refused(Exception):
    pass


def results_dir() -> str:
    """results/ of this tree (public layout: tools/ next to results/)."""
    for d in (os.path.join(ROOT, "results"), os.path.join(ROOT, "..", "results")):
        if os.path.exists(os.path.join(d, "devices.json")):
            return os.path.abspath(d)
    return os.path.join(ROOT, "results")


def release_files() -> dict:
    """data_sha256 -> (model_id, file) of every published release model file."""
    for p in (os.path.join(ROOT, "registry", "release_models.json"), os.path.join(ROOT, "public", "registry", "release_models.json")):
        if os.path.exists(p):
            rm = IS.load(p)
            return {a["data_sha256"]: (mid, a["file"]) for mid, m in rm.get("models", {}).items() for a in m.get("assets", [])}
    raise SystemExit("registry/release_models.json not found")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def pct50(v: list[float]) -> float:
    """Median with the runtime's interpolation (bitino_bench stats())."""
    v = sorted(v)
    idx = 0.5 * (len(v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)


def check_privacy(obj, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if PRIVATE_KEY.search(k) and v not in (None, "", [], {}):
                raise Refused(f"privacy: field {p} carries a value; submissions must not contain it")
            check_privacy(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_privacy(v, f"{path}[{i}]")


def check_device(dev: dict) -> None:
    for k in ("model", "android_release", "abi"):
        if not dev.get(k):
            raise Refused(f"device: no {k}")
    if not (dev.get("soc_model") or dev.get("board_platform")):
        raise Refused("device: neither soc_model nor board_platform")
    if dev["abi"] not in ABIS:
        raise Refused(f"device: ABI {dev['abi']!r} is not a phone ABI Bitino builds for")
    em = dev.get("emulator_markers") or {}
    if str(em.get("ro.kernel.qemu", "")).strip() == "1" or str(em.get("ro.boot.qemu", "")).strip() == "1" or \
            "emulator" in str(em.get("ro.build.characteristics", "")).lower() or \
            EMU_HW.match(dev.get("hardware") or "") or EMU_MODEL.search(dev.get("model") or ""):
        raise Refused("physical: emulator markers present; the community table takes real phones only")
    if not dev.get("cpufreq_policies"):
        raise Refused("physical: no cpufreq policies recorded (real phones expose them)")


def core_name(dev: dict, cpu: int) -> str:
    for c in dev.get("cpus") or []:
        if c.get("cpu") == cpu:
            return CPU_PARTS.get((str(c.get("implementer", "")).lower(), str(c.get("part", "")).lower()), f"part {c.get('part')}")
    return f"cpu{cpu}"


def device_entry(dev: dict) -> dict:
    """results/devices.json-style entry from the submission's device block."""
    pols = dev["cpufreq_policies"]
    cores, names = [], {}
    for pid, p in sorted(pols.items(), key=lambda kv: -int(kv[1].get("cpuinfo_max_khz") or 0)):
        cpus = p.get("cpus") or []
        groups: dict[str, list[int]] = {}
        for c in cpus:
            groups.setdefault(core_name(dev, c), []).append(c)
        for name, cs in groups.items():
            cores.append({"name": name, "count": len(cs), "max_ghz": round((p.get("cpuinfo_max_khz") or 0) / 1e6, 4),
                          "cpus": ",".join(map(str, cs))})
        names[pid] = "/".join(groups) or f"cpu{pid}"
    soc = " ".join(x for x in (dev.get("soc_manufacturer"), dev.get("soc_model")) if x) or \
        (f"platform {dev['board_platform']}" if dev.get("board_platform") else None)
    return {"kind": "phone", "source": "community", "name": f"{dev.get('manufacturer') or ''} {dev['model']}".strip(),
            "short": dev["model"], "soc": soc, "board_platform": dev.get("board_platform"), "android": dev.get("android_release"),
            "abi": dev["abi"], "kernels": ABIS[dev["abi"]], "isa": dev.get("cpu_features"), "cores": cores,
            "policies": names, "fastest_cpu": dev.get("fastest_cpu"), "product_models": [dev["model"]]}


def runtime_version(rep: dict) -> str:
    for k, v in rep.items():
        if k.endswith("_version") and isinstance(v, str):
            return v
    return "?"


def maintainer_runs(res: str, product_model: str) -> list[dict]:
    """Runs measured by the maintainers on the same phone model (results/speed_<device>.json)."""
    devs = (IS.load(os.path.join(res, "devices.json"), {"devices": {}}) or {}).get("devices", {})
    out = []
    for did, d in devs.items():
        if product_model.lower() in [x.lower() for x in d.get("product_models", [])]:
            out += (IS.load(os.path.join(res, f"speed_{did}.json"), {"runs": []}) or {}).get("runs", [])
    return out


def validate(sub: dict, raw: bytes, res: str, issue: str | None = None) -> tuple[str, dict, dict, list[dict]]:
    """-> (device id, device entry, submission entry, run records); raises Refused."""
    if sub.get("schema") != SCHEMA_IN:
        raise Refused(f"schema: {sub.get('schema')!r}, expected {SCHEMA_IN!r}")
    if (sub.get("tool") or {}).get("version") not in TOOL_VERSIONS:
        raise Refused(f"tool: unknown version {(sub.get('tool') or {}).get('version')!r}")
    check_privacy(sub)
    dev = sub.get("device") or {}
    check_device(dev)
    entry = device_entry(dev)
    did = "community-" + slug(dev["model"])
    rel = release_files()
    created = sub.get("created_utc") or ""
    try:
        date = datetime.date.fromisoformat(created[:10])
    except ValueError:
        raise Refused(f"created_utc {created!r} is not a date")
    if date > datetime.date.today() + datetime.timedelta(days=1):
        raise Refused(f"created_utc {created} is in the future")

    checks = {}
    for c in sub.get("output_checks") or []:
        got, exp = c.get("output_sha256"), c.get("expected_sha256")
        if got and exp and got != exp:
            raise Refused(f"outputs: {c.get('asset')} ({c.get('stage')}) gave {got[:12]}..., expected {exp[:12]}...: "
                          "the runtime computed a wrong result on this phone (please report it as a bug)")
        checks[(c.get("model_id"), c.get("model_file"))] = "match" if got and exp else ("unchecked" if got else "failed")

    groups: dict[tuple, list[dict]] = {}
    dropped = 0
    for i, r in enumerate(sub.get("runs") or []):
        rep = r.get("report") or {}
        where = f"runs[{i}] ({r.get('asset')} {r.get('config')} r{r.get('repeat')})"
        if not str(rep.get("schema", "")).endswith(".benchmark.v1"):
            raise Refused(f"{where}: report schema {rep.get('schema')!r}")
        if r.get("config") not in CONFIGS:
            raise Refused(f"{where}: unknown configuration {r.get('config')!r}")
        if (rep.get("device") or {}).get("model") != dev["model"]:
            raise Refused(f"physical: {where} was recorded on {(rep.get('device') or {}).get('model')!r}, not {dev['model']!r}")
        cfg, lat, m = rep.get("config") or {}, rep.get("latency") or {}, rep.get("model") or {}
        for k, want in (("stage", r.get("stage")), ("threads", r.get("threads")), ("affinity", r.get("affinity")),
                        ("iterations", r.get("iterations"))):
            if cfg.get(k) != want:
                raise Refused(f"consistency: {where}: report config.{k} = {cfg.get(k)!r}, run says {want!r}")
        s = lat.get("samples_ms") or []
        if len(s) != r.get("iterations") or not lat.get("median_ms"):
            raise Refused(f"consistency: {where}: {len(s)} samples for {r.get('iterations')} iterations")
        if abs(pct50(s) - lat["median_ms"]) > 1e-6 * max(1.0, lat["median_ms"]) or abs(lat["median_ms"] - (r.get("median_ms") or 0)) > 1e-6 * lat["median_ms"]:
            raise Refused(f"consistency: {where}: median does not match its samples")
        hit = rel.get(m.get("data_sha256"))
        if not hit or hit != (r.get("model_id"), r.get("model_file")):
            raise Refused(f"models: {where}: data_sha256 {str(m.get('data_sha256'))[:12]}... is not the release file "
                          f"{r.get('model_id')}/{r.get('model_file')}")
        if m.get("hash_verified") is not True:
            raise Refused(f"models: {where}: the runtime did not verify the model hash")
        if IS.unpinned_repeat(rep, r.get("affinity")):
            dropped += 1
            continue
        groups.setdefault((r["model_id"], r["model_file"], r["config"]), []).append(r)
    if not groups:
        raise Refused("no runs")

    summ = {(x.get("model_id"), x.get("model_file"), x.get("config")): x for x in sub.get("summary") or []}
    maint = maintainer_runs(res, dev["model"])
    runtime = runtime_version(groups[next(iter(groups))][0]["report"])
    binary = sub.get("binary") or {}
    sub_id = hashlib.sha256(raw).hexdigest()[:12]
    records = []
    for (mid, mf, cname), rs in sorted(groups.items()):
        meds = [x["report"]["latency"]["median_ms"] for x in rs]
        med = statistics.median(meds)
        for x in meds:
            if not 0.05 < x < 60000:
                raise Refused(f"plausibility: {mid}/{mf} {cname}: median {x} ms")
        cls = [IS.classify_repeat(x.get("throttle")) for x in rs]
        validity = IS.worst(c["validity"] for c in cls) if all(c["checked"] for c in cls) else "unknown-caps"
        caps = sorted({IS.caps_text(c.get("policies"), entry) for c in cls if c.get("policies")} - {None})
        s = summ.get((mid, mf, cname))
        if s and len(rs) == len(meds) and not dropped and abs((s.get("median_ms") or 0) - med) > 1e-3 * max(1.0, med):
            raise Refused(f"consistency: summary median of {mid}/{mf} {cname} ({s.get('median_ms')}) != {med:.6g} from the runs")
        notes = []
        if max(meds) > 1.5 * min(meds):
            notes.append(f"repeats {min(meds):.3f}..{max(meds):.3f} ms: unsteady conditions")
        base = re.sub(r"\.[a-z]+$", "", mf)
        mr = [x for x in maint if x.get("model_id") == mid and re.sub(r"\.[a-z]+$", "", x.get("model_file") or "") == base
              and x.get("stage") == rs[0]["stage"] and x.get("threads") == rs[0]["threads"] and x.get("median_ms")]
        if mr:
            ref = max(mr, key=lambda x: x.get("date") or "")
            ratio = med / ref["median_ms"]
            if not 0.5 <= ratio <= 2.0:
                notes.append(f"{ratio:.2f}x the maintainers' run of this phone model ({ref['median_ms']:.3f} ms, {ref.get('date')})")
        rep0 = rs[0]["report"]
        p90 = [x["report"]["latency"].get("p90_ms") for x in rs if x["report"]["latency"].get("p90_ms")]
        records.append({
            "model_id": mid, "device": did, "source": "community", "submission": sub_id, "issue": issue,
            "date": created[:10], "timestamp_utc": created, "config": cname, "model_file": mf, "stage": rs[0]["stage"],
            "threads": rs[0]["threads"], "affinity": rs[0]["affinity"], "median_ms": med,
            "median_range_ms": [min(meds), max(meds)], "p90_ms": statistics.median(p90) if p90 else None,
            "repeats": len(rs), "iterations": rs[0]["iterations"], "warmup": rs[0].get("warmup"),
            "peak_rss_kb": max(((x["report"].get("memory") or {}).get("vm_hwm_kb") or 0) for x in rs) or None,
            "validity": validity, "caps_seen": caps or None, "notes": notes or None,
            "abi": dev["abi"], "kernels": ABIS[dev["abi"]], "android": dev.get("android_release"),
            "model_data_sha256": rep0["model"]["data_sha256"],
            "runtime": f"bitino {runtime}" + (f" (release {(sub.get('tool') or {}).get('release')})" if binary.get("source") == "release"
                                               else " (local build)"),
            "binary_sha256": binary.get("sha256"), "output_check": checks.get((mid, mf), "not run")})

    # plausibility across configurations / files
    by = {(r["model_id"], r["model_file"], r["config"]): r for r in records}
    for (mid, mf, cname), r in by.items():
        t1 = by.get((mid, mf, "t1_fast"))
        if cname == "t4_fast" and t1 and t1["median_ms"] / r["median_ms"] > 1.15 * r["threads"]:
            raise Refused(f"plausibility: {mid}/{mf}: {r['threads']} threads {t1['median_ms'] / r['median_ms']:.1f}x faster than 1 thread")
        fl = by.get((mid, "model.bitino", cname))
        if mf != "model.bitino" and fl and not 0.05 <= r["median_ms"] / fl["median_ms"] <= 20:
            raise Refused(f"plausibility: {mid}: {mf} / float ratio {r['median_ms'] / fl['median_ms']:.2f}")
    submission = {"id": sub_id, "issue": issue, "created_utc": created, "android": dev.get("android_release"),
                  "security_patch": dev.get("security_patch"), "runtime": records[0]["runtime"],
                  "tool_version": sub["tool"]["version"], "runs": len(records), "dropped_unpinned_repeats": dropped,
                  "start": (sub.get("conditions") or {}).get("start"), "end": (sub.get("conditions") or {}).get("end")}
    return did, entry, submission, records


def merge(res: str, did: str, entry: dict, submission: dict, records: list[dict]) -> str:
    cdir = os.path.join(res, "community")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"speed_{did}.json")
    doc = IS.load(path, {"schema": SCHEMA_OUT + ".speed", "device_id": did, "device": entry,
                         "method": {"tool": "scripts/bench_my_phone.sh (bitino_bench on the phone over adb)",
                                    "statistic": "median latency of one inference (batch 1) per repeat; median of repeats",
                                    "configs": {"t1_fast": "1 thread pinned to the fastest core",
                                                "t4_fast": "4 threads pinned to the 4 fastest cores"},
                                    "validity": "see results/README.md"},
                         "submissions": [], "runs": []})
    doc["device"] = entry   # the latest submission's view of the phone (Android version, clocks)
    doc["submissions"] = [s for s in doc.get("submissions", []) if s["id"] != submission["id"]] + [submission]
    doc["runs"] = [r for r in doc.get("runs", []) if r.get("submission") != submission["id"]] + records
    doc["generated"] = datetime.date.today().isoformat()
    IS.save(path, doc)
    write_index(cdir)
    return path


def write_index(cdir: str) -> None:
    devices = {}
    for fn in sorted(os.listdir(cdir)):
        if not (fn.startswith("speed_") and fn.endswith(".json")):
            continue
        d = IS.load(os.path.join(cdir, fn))
        runs = d.get("runs", [])
        head = {}
        for cname in sorted(CONFIGS):
            for mf in ("model.bitino", "model_int8_full.bitino"):
                xs = [r for r in runs if (r["model_id"], r["model_file"]) == (HEADLINE[0], mf) and r["config"] == cname]
                if xs:
                    head[f"{mf.replace('.bitino', '')}:{cname}"] = statistics.median(r["median_ms"] for r in xs)
        devices[d["device_id"]] = {"name": d["device"].get("name"), "soc": d["device"].get("soc"), "abi": d["device"].get("abi"),
                                   "submissions": len(d.get("submissions", [])), "runs": len(runs),
                                   "latest": max((r["date"] for r in runs), default=None),
                                   "headline_ms": {"model": HEADLINE[0], **head} if head else None}
    IS.save(os.path.join(cdir, "index.json"), {"schema": SCHEMA_OUT + ".index", "generated": datetime.date.today().isoformat(),
                                               "headline": "median over submissions of ab-bnn-imagenet-reactnet18, ms",
                                               "devices": devices})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--issue", help="issue number the file was attached to (recorded with the runs)")
    ap.add_argument("--check", action="store_true", help="validate and print the records; write nothing")
    ap.add_argument("--results", default=None, help="results directory (default: results/ of this tree)")
    a = ap.parse_args()
    res = a.results or results_dir()
    bad = 0
    for p in a.files:
        with open(p, "rb") as f:
            raw = f.read()
        try:
            sub = json.loads(raw)
            did, entry, submission, records = validate(sub, raw, res, a.issue)
        except (Refused, ValueError, KeyError, TypeError) as e:
            print(f"REFUSED {p}: {e}", file=sys.stderr)
            bad += 1
            continue
        print(f"{p}: {did} ({entry['name']}, {entry['soc']}), submission {submission['id']}, {len(records)} runs")
        for r in records:
            print(f"  {r['model_id']:<40} {r['model_file']:<24} {r['config']:<8} {r['median_ms']:9.3f} ms  {r['validity']:<17} "
                  f"output {r['output_check']}" + (f"  notes: {'; '.join(r['notes'])}" if r["notes"] else ""))
        if not a.check:
            print(f"  -> {merge(res, did, entry, submission, records)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
