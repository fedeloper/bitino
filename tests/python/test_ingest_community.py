"""Community speed submissions (scripts/bench_my_phone.sh -> tools/ingest_community.py): validation and storage.

The fixture mimics the JSON written by the script: device block, output checks, one run per repeat with a throttle
block and the raw bench report.
"""
import copy
import importlib.util
import json
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _path(*rel):
    for d in ("", "public"):   # public tree / development tree
        p = os.path.join(ROOT, d, *rel)
        if os.path.exists(p):
            return p
    pytest.skip(f"{os.path.join(*rel)} not found")


def _tool(name):
    p = _path("tools", name + ".py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IC = _tool("ingest_community")
RELEASE = json.load(open(_path("registry", "release_models.json")))
MID = "ab-bnn-imagenet-reactnet18"
ASSETS = {a["file"]: a for a in RELEASE["models"][MID]["assets"]}
CAPS = {"0": [2016000, 2016000], "3": [1785600, 2803200], "7": [2227200, 3187200]}


def side(pol, status=0):
    return {"policies": pol, "capped": any(c < m for c, m in pol.values()), "thermal_status": status}


def report(model, mf, stage, threads, affinity, samples):
    v = sorted(samples)
    i = 0.5 * (len(v) - 1)
    med = v[int(i)] + (v[min(int(i) + 1, len(v) - 1)] - v[int(i)]) * (i - int(i))
    return {"schema": "bitino.benchmark.v1", "bitino_version": "0.1.0",
            "device": {"model": model, "build_fingerprint": None, "cpu_features": "aarch64: neon dotprod"},
            "config": {"stage": stage, "threads": threads, "affinity": affinity, "affinity_errors": 0, "warmup": 10,
                       "iterations": len(samples)},
            "model": {"data_sha256": ASSETS[mf]["data_sha256"], "hash_verified": True, "manifest": None},
            "latency": {"median_ms": med, "p90_ms": v[-2], "samples_ms": samples}, "memory": {"vm_hwm_kb": 20000}}


def submission(model="TESTPHONE-1", status=0, caps=CAPS):
    runs, summary = [], []
    for mf, stage, base in (("model.bitino", "all", 10.0), ("model_int8_full.bitino", "int8_full", 7.0)):
        asset = MID + ("" if mf == "model.bitino" else "_int8_full")
        for cfg, thr, aff, div in (("t1_fast", 1, "7", 1.0), ("t4_fast", 4, "7,3,4,5", 2.0)):
            meds = []
            for rep in range(3):
                s = [base / div + 0.01 * ((k * 7) % 5) + 0.001 * rep for k in range(50)]
                r = report(model, mf, stage, thr, aff, s)
                meds.append(r["latency"]["median_ms"])
                runs.append({"asset": asset, "model_id": MID, "model_file": mf, "config": cfg, "stage": stage,
                             "threads": thr, "affinity": aff, "repeat": rep, "warmup": 10, "iterations": 50,
                             "median_ms": r["latency"]["median_ms"], "validity": "vendor-policy",
                             "throttle": {"mode": "throttle", "waited_s": 0, "timeout": False, "throttled_before": False,
                                          "throttled": status > 0, "before": side(caps, status), "after": side(caps, status)},
                             "report": r})
            summary.append({"model_id": MID, "model_file": mf, "config": cfg, "median_ms": sorted(meds)[1]})
    return {"schema": "bitino.community_bench.v1", "tool": {"name": "bench_my_phone.sh", "version": 1, "release": "v0.1.0"},
            "created_utc": "2026-11-20T10:00:00Z",
            "device": {"model": model, "manufacturer": "Acme", "brand": "acme", "device": "acme1", "soc_manufacturer": "QTI",
                       "soc_model": "SM8550", "board_platform": "kalama", "hardware": "qcom", "android_release": "16",
                       "api_level": "36", "abi": "arm64-v8a", "cpu_features": "aarch64: neon dotprod",
                       "emulator_markers": {"ro.kernel.qemu": "", "ro.boot.qemu": "", "ro.build.characteristics": "nosdcard"},
                       "cpus": [{"cpu": c, "implementer": "0x41", "part": p} for c, p in
                                ((0, "0xd46"), (1, "0xd46"), (2, "0xd46"), (3, "0xd4d"), (4, "0xd4d"), (5, "0xd47"),
                                 (6, "0xd47"), (7, "0xd4e"))],
                       "cpufreq_policies": {"0": {"cpus": [0, 1, 2], "cpuinfo_max_khz": 2016000},
                                            "3": {"cpus": [3, 4, 5, 6], "cpuinfo_max_khz": 2803200},
                                            "7": {"cpus": [7], "cpuinfo_max_khz": 3187200}},
                       "fastest_cpu": "7", "t4_affinity": "7,3,4,5"},
            "binary": {"source": "release", "sha256": "00" * 32, "runtime_version": "0.1.0"},
            "output_checks": [{"asset": MID, "model_id": MID, "model_file": "model.bitino", "stage": "neon",
                               "output_sha256": "ab" * 32, "expected_sha256": "ab" * 32, "result": "match"}],
            "summary": summary, "runs": runs}


def results(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "devices.json").write_text(json.dumps({"devices": {}}))
    return str(res)


def ingest(sub, res):
    raw = json.dumps(sub).encode()
    return IC.validate(json.loads(raw), raw, res, "7")


def test_valid_submission_is_stored_as_community(tmp_path):
    res = results(tmp_path)
    did, entry, subm, recs = ingest(submission(), res)
    assert did == "community-testphone-1" and entry["source"] == "community"
    assert entry["cores"][0]["name"] == "X3" and entry["policies"]["3"] == "A715/A710"
    assert len(recs) == 4 and {r["config"] for r in recs} == {"t1_fast", "t4_fast"}
    r = next(r for r in recs if r["model_file"] == "model.bitino" and r["config"] == "t1_fast")
    assert r["source"] == "community" and r["issue"] == "7" and r["validity"] == "vendor-policy"
    assert r["caps_seen"] and "X3 2.23/3.19 GHz" in r["caps_seen"][0]
    assert r["output_check"] == "match" and abs(r["median_ms"] - 10.02) < 0.05
    p = IC.merge(res, did, entry, subm, recs)
    doc = json.load(open(p))
    assert doc["schema"].endswith(".community.speed") and len(doc["runs"]) == 4
    IC.merge(res, did, entry, subm, recs)   # same file again: replaced, not duplicated
    assert len(json.load(open(p))["runs"]) == 4
    idx = json.load(open(os.path.join(res, "community", "index.json")))
    assert idx["devices"][did]["submissions"] == 1 and idx["devices"][did]["headline_ms"]["model:t1_fast"] > 0


def test_thermal_status_marks_runs_throttled(tmp_path):
    _, _, _, recs = ingest(submission(status=2), results(tmp_path))
    assert {r["validity"] for r in recs} == {"thermal-throttled"}


@pytest.mark.parametrize("mutate,why", [
    (lambda s: s["device"].update(hardware="ranchu"), "emulator"),
    (lambda s: s["device"].update(model="sdk_gphone64_arm64"), "emulator"),
    (lambda s: s["device"].update(serial="ABC123"), "privacy"),
    (lambda s: s["runs"][0]["report"]["device"].update(build_fingerprint="acme/acme1:16/X/1:user/release-keys"), "privacy"),
    (lambda s: s["device"].update(cpufreq_policies={}), "cpufreq"),
    (lambda s: s["device"].update(abi="x86_64"), "ABI"),
    (lambda s: s["runs"][0]["report"]["device"].update(model="OTHER"), "recorded on"),
    (lambda s: s["runs"][0]["report"]["model"].update(data_sha256="cd" * 32), "release file"),
    (lambda s: s["runs"][0]["report"]["latency"].update(median_ms=1.0), "median"),
    (lambda s: s["runs"][0]["report"]["config"].update(threads=2), "config.threads"),
    (lambda s: s["output_checks"][0].update(output_sha256="ef" * 32), "wrong result"),
    (lambda s: s.update(schema="bitino.community_bench.v0"), "schema"),
])
def test_refused(tmp_path, mutate, why):
    s = copy.deepcopy(submission())
    mutate(s)
    with pytest.raises(IC.Refused, match=why):
        ingest(s, results(tmp_path))


def test_implausible_thread_speedup_refused(tmp_path):
    s = submission()
    for r in s["runs"]:
        if r["config"] == "t4_fast":
            r["report"]["latency"]["samples_ms"] = [x / 4 for x in r["report"]["latency"]["samples_ms"]]
            r["report"]["latency"]["median_ms"] /= 4
            r["median_ms"] = r["report"]["latency"]["median_ms"]
    s["summary"] = []
    with pytest.raises(IC.Refused, match="faster than 1 thread"):
        ingest(s, results(tmp_path))


def test_bench_script_stays_in_its_directory_and_records_no_identifier():
    text = open(_path("scripts", "bench_my_phone.sh")).read()
    assert 'DEV_DIR="/data/local/tmp/bitino-community"' in text
    for line in text.splitlines():
        if re.search(r"\brm -rf?\b", line) and "TMP" not in line and ".part" not in line:
            assert "$DEV_DIR" in line, line
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"ro\.serialno|ro\.boot\.serialno|get-serialno|settings get secure|iphonesubinfo|scaling_governor\s*$", code)
    assert not re.search(r">\s*\S*scaling_(governor|max_freq|min_freq)|setprop|reboot|svc |input keyevent|am start", code)
