"""Multi-device speed results: validity classification, ingest of tools/benchmark.py directories, README tables.

The fixture mimics a quick-plan directory written by tools/benchmark.py + tools/report_benchmarks.py
(summary.json rows + <config>__r<k>.json raws with "device" and "throttle" blocks).
"""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _tool(name):
    for d in ("tools", os.path.join("public", "tools")):   # public tree / development tree
        p = os.path.join(ROOT, d, name + ".py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location(name, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip(f"{name}.py not found")


IS = _tool("ingest_speed")
SITE = _tool("build_site")

ONEPLUS = {"kind": "phone", "name": "OnePlus Open (CPH2551)", "short": "OnePlus Open", "soc": "Snapdragon 8 Gen 2 (SM8550)",
           "abi": "arm64-v8a", "kernels": "neon",
           "cores": [{"name": "Cortex-X3", "count": 1, "max_ghz": 3.1872, "cpus": "7"},
                     {"name": "Cortex-A510", "count": 3, "max_ghz": 2.016, "cpus": "0-2"}],
           "policies": {"7": "X3", "3": "A715/A710", "0": "A510"}, "fastest_cpu": "7",
           "product_models": ["CPH2551"], "tags": ["cph2551"]}
S22 = {"kind": "phone", "name": "Samsung Galaxy S22", "short": "Galaxy S22", "abi": "arm64-v8a", "kernels": "neon",
       "cores": [{"name": "Cortex-X2", "count": 1, "max_ghz": 2.9952, "cpus": "7"}],
       "policies": {"7": "X2", "4": "A710", "0": "A510"}, "fastest_cpu": "7", "product_models": ["SM-S901U1"]}
VENDOR_CAPS = {"0": [2016000, 2016000], "3": [1785600, 2803200], "7": [2227200, 3187200]}


def throttle(policies, status):
    capped = any(c < m for c, m in policies.values())
    side = {"policies": policies, "capped": capped, "thermal_status": status, "skin_c": None}
    return {"waited_s": 0.0, "before": side, "after": side, "timeout": capped, "throttled": capped}


def write_plan(base, model, plan, product_model, throttles, configs=("t1_x2", "t4_unpinned", "t1_x2_int8")):
    d = os.path.join(base, model, plan)
    os.makedirs(d)
    rows = []
    for name in configs:
        thr, aff = (4, "") if name.startswith("t4") else (1, "7")
        stage = "optimized_int8" if name.endswith("int8") else "all"
        rows.append({"name": name, "stage": stage, "threads": thr, "affinity": aff, "model_file": "model.bitino",
                     "repeats": len(throttles), "iterations_per_repeat": 50, "warmup": 10,
                     "median_ms": 5.0 / thr, "median_min_ms": 4.9 / thr, "median_max_ms": 5.2 / thr, "p90_ms": 5.3 / thr,
                     "vm_hwm_kb": 11000, "model_data_sha256": "ab" * 32, "timestamp_first": "2026-11-11T15:30:00Z"})
        for k, th in enumerate(throttles):
            raw = {"schema": "bitino.benchmark.v1", "bitino_version": "0.1.0",
                   "device": {"model": product_model, "soc": "SM8550", "cpu_features": "aarch64: neon dotprod i8mm"}}
            if th is not None:
                raw["throttle"] = th
            json.dump(raw, open(os.path.join(d, f"{name}__r{k}.json"), "w"))
    json.dump({"plan": {"plan": "quick"}, "rows": rows}, open(os.path.join(d, "summary.json"), "w"))
    return d


@pytest.fixture
def res(tmp_path):
    r = tmp_path / "results"
    (r / "models").mkdir(parents=True)
    json.dump({"schema": "bitino.results.v1.devices", "devices": {"s22": S22, "oneplus-open": ONEPLUS}},
              open(r / "devices.json", "w"))
    json.dump({"model_id": "m1", "speed": {"s22": []}}, open(r / "models" / "m1.json", "w"))
    return str(r)


def test_classify_repeat():
    assert IS.classify_repeat(throttle({"7": [3187200, 3187200]}, 0))["validity"] == "clean"
    assert IS.classify_repeat(throttle(VENDOR_CAPS, 0))["validity"] == "vendor-policy"
    assert IS.classify_repeat(throttle(VENDOR_CAPS, 1))["validity"] == "thermal-throttled"
    assert IS.classify_repeat(throttle(VENDOR_CAPS, None))["validity"] == "thermal-throttled"   # cause unknown: not excused
    assert IS.classify_repeat(None)["validity"] == "unknown-caps"
    assert IS.worst(["clean", "vendor-policy", "clean"]) == "vendor-policy"
    assert IS.worst(["vendor-policy", "thermal-throttled"]) == "thermal-throttled"


def test_caps_text_uses_device_core_names():
    assert IS.caps_text(VENDOR_CAPS, ONEPLUS) == "X3 2.23/3.19 GHz, A715/A710 1.79/2.80 GHz, A510 2.02/2.02 GHz"
    assert IS.caps_text({"6": [1612000, 1612000]}, None) == "cpu6 1.61/1.61 GHz"


def test_plan_dir_and_device_matching():
    m = IS.PLAN_DIR.match("quick_20261111_1530_moto-e13")
    assert (m.group("plan"), m.group("ts"), m.group("tag")) == ("quick", "20261111_1530", "moto-e13")
    assert IS.PLAN_DIR.match("quick_20261102_1142").group("tag") is None
    devs = {"s22": S22, "oneplus-open": ONEPLUS}
    assert IS.match_device(devs, "cph2551", None) == "oneplus-open"
    assert IS.match_device(devs, None, "cph2551") == "oneplus-open"
    assert IS.match_device(devs, "SM-S901U1", "cph2551") == "s22"   # the recorded model wins over the directory name
    assert IS.match_device(devs, "Pixel 8", None) is None


def test_ingest_device(tmp_path, res):
    bench = str(tmp_path / "benchmarks")
    write_plan(bench, "m1", "quick_20261111_1530_cph2551", "CPH2551", [throttle(VENDOR_CAPS, 0)] * 3)
    n = IS.ingest_device(res, "oneplus-open", [os.path.join(bench, "*", "quick_*_cph2551")])
    assert n == 3
    doc = json.load(open(os.path.join(res, "speed_oneplus-open.json")))
    assert doc["device_id"] == "oneplus-open" and len(doc["runs"]) == 3
    r = next(r for r in doc["runs"] if r["config"] == "t1_x2")
    assert r["validity"] == "vendor-policy" and r["device"] == "oneplus-open" and r["abi"] == "arm64-v8a"
    assert r["caps_seen"] == ["X3 2.23/3.19 GHz, A715/A710 1.79/2.80 GHz, A510 2.02/2.02 GHz"]
    m = json.load(open(os.path.join(res, "models", "m1.json")))
    assert len(m["speed"]["oneplus-open"]) == 3 and m["speed"]["s22"] == []
    # re-ingesting the same plan replaces, never duplicates
    IS.ingest_device(res, "oneplus-open", [os.path.join(bench, "m1", "quick_20261111_1530_cph2551")])
    assert len(json.load(open(os.path.join(res, "speed_oneplus-open.json")))["runs"]) == 3


def test_ingest_refuses_other_phone_and_emulated(tmp_path, res):
    bench = str(tmp_path / "benchmarks")
    d = write_plan(bench, "m1", "quick_20261111_1530", "SM-S901U1", [throttle(VENDOR_CAPS, 1)])
    with pytest.raises(SystemExit, match="not on oneplus-open"):
        IS.ingest_device(res, "oneplus-open", [d])
    e = write_plan(str(tmp_path / "emulated"), "m1", "quick_20261111_1530", "CPH2551", [None])
    with pytest.raises(SystemExit, match="emulated"):
        IS.ingest_device(res, "oneplus-open", [e])
    with pytest.raises(SystemExit, match="unknown device"):
        IS.ingest_device(res, "pixel-8", [d])


def test_ingest_add_device(tmp_path, res):
    bench = str(tmp_path / "benchmarks")
    d = write_plan(bench, "m1", "quick_20261111_1530_moto-e13", "moto e13", [throttle({"6": [1612000, 1612000]}, 0)])
    IS.ingest_device(res, "moto-e13", [d], add=True)
    devs = json.load(open(os.path.join(res, "devices.json")))["devices"]
    assert devs["moto-e13"]["product_models"] == ["moto e13"]
    runs = json.load(open(os.path.join(res, "speed_moto-e13.json")))["runs"]
    assert {r["validity"] for r in runs} == {"clean"}


def test_pick_speed_and_readme_tables():
    def run(cfg, stage, thr, aff, ms, validity, date="2026-11-11"):
        return {"model_id": "m1", "config": cfg, "stage": stage, "threads": thr, "affinity": aff, "median_ms": ms,
                "validity": validity, "date": date, "plan": "quick", "model_file": "model.bitino"}
    op = [run("t1_x2", "all", 1, "7", 4.0, "vendor-policy"), run("t4_unpinned", "all", 4, "unpinned", 1.5, "vendor-policy"),
          run("t1_x2_int8", "optimized_int8", 1, "7", 3.0, "vendor-policy"), run("t1_a510", "all", 1, "0", 20.0, "clean")]
    s22 = [run("t1_x2", "all", 1, "7", 5.0, "throttled", "2026-11-02")]
    p = SITE.pick_speed(op, "7")
    assert p["t1"]["ms"] == 4.0 and p["t4"]["ms"] == 1.5 and p["t1_int8"]["stage"] == "optimized_int8" and "t4_int8" not in p
    q = SITE.pick_speed(s22, "7")
    assert q["t1"]["validity"] == "thermal-throttled"   # legacy label normalized
    devices = {"s22": S22, "oneplus-open": ONEPLUS}
    model = {"id": "m1", "task": "image classification", "framework": "torch", "metric": "top-1", "reference": 0.6,
             "runtime": 0.6, "eval_set": "imagenet", "n": 128, "int8": None, "release": False, "recipe": "reconstructed",
             "licence_weights": "MIT", "phones": {"s22": q, "oneplus-open": p},
             "best": {"t1": {**p["t1"], "device": "oneplus-open", "device_short": "OnePlus Open"}}}
    d = {"models": [model], "devices": devices, "registry_total": 1, "verified_weights": 1,
         "speed": {"s22": {"runs": s22}, "oneplus-open": {"runs": op}}}
    t = SITE.readme_tables(d)
    head = t["speed_full"].splitlines()[0]   # docs/SPEED.md: every phone, 1 and 4 threads and int8
    assert "Galaxy S22 ms" in head and "OnePlus Open ms" in head
    row = next(line for line in t["speed_full"].splitlines() if line.startswith("| `m1`"))
    assert "5 ⚠" in row and "4 ◇ / 1.5 ◇<br>int8 3 ◇ / –" in row
    assert "◇ = the phone's own power policy" in t["speed"] and "⚠ = at least one repeat" in t["speed"]
    # README: tables by task, headline phone = the most recent suite (OnePlus), other phones' 1-thread column after it
    assert "### Other tasks (1 models)" in t["speed"] and "**OnePlus Open" in t["speed"]
    row = next(line for line in t["speed"].splitlines() if line.startswith("| `m1`"))
    assert row.endswith("| 4 ◇ | 1.5 ◇ | 3 ◇ | 5 ⚠ |")
    assert "4 ◇ (OnePlus Open)" in t["zoo"]


def test_pareto_frontier_and_svg():
    sys.path.insert(0, os.path.join(ROOT, "public", "tools"))
    import pareto_plot as PP
    pts = [{"id": "a", "label": "a", "x": 5.0, "y": 59.4}, {"id": "b", "label": "b", "x": 8.0, "y": 63.3},
           {"id": "c", "label": "c", "x": 9.0, "y": 60.0},   # slower and less accurate than b: dominated
           {"id": "d", "label": "d", "x": 11.0, "y": 63.3},  # ties b on accuracy but slower: dominated
           {"id": "e", "label": "e", "x": 25.0, "y": 72.4}]
    assert [p["id"] for p in PP.frontier(pts)] == ["a", "b", "e"]
    svg = PP.pareto_svg(pts, "T", "s", "x", "y")
    assert svg.startswith("<svg") and svg.count('class="dot-f"') == 3 + 1 and "var(" not in svg   # +1: legend mark
