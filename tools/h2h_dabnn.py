#!/usr/bin/env python3
"""Head-to-head on a phone: Bitino (bitino_bench) vs daBNN (JDAI-CV/dabnn) on daBNN's own Bi-Real Net-18 models,
1 thread, batch 1, every runtime pinned to the same core by the same launcher. Method: docs/comparison-dabnn.md;
driver script: scripts/h2h_dabnn.sh.

  python3 tools/h2h_dabnn.py run    --serial e3698ea7 --tag cph2551 --cpu 7 --abi arm64 --work <dabnn-h2h> --out <dir>
  python3 tools/h2h_dabnn.py eval   --serial e3698ea7 --tag cph2551 --cpu 7 --abi arm64 --work <dabnn-h2h> --out <dir>
  python3 tools/h2h_dabnn.py report --in reports/h2h_dabnn/<date> --work <dabnn-h2h> --md reports/h2h_dabnn/<date>/summary.md

Runtimes (all under tools/h2h_run.c `h2h_run --cpus <cpu> -- ...`, as in tools/h2h_bench.py):
- dabnn          tools/dabnn_h2h.cc linked to daBNN's libdabnn: 1 untimed run on eval image 0 (output written),
                 --warmup-1 more untimed runs, --iterations timed runs (steady_clock around bnn::Net::run)
- bitino_all     bitino_bench --stage all, model.bitino, same input, --warmup-1 + 1 correctness run untimed
- bitino_int8_full  bitino_bench --stage int8_full, model_int8_full.bitino (8-bit float parts; not bit-exact to
                 the float model, see docs/comparison-dabnn.md)
- dabnn_gbench   daBNN's own benchmark target benchmark_single_model (google/benchmark, default settings: adaptive
                 iteration count, min time 0.5 s, mean real time per iteration; input = uninitialised stack
                 buffer, as upstream). A cross-check of the driver, not the primary number.
Per run: mean and median of the timed runs; per configuration: median over --repeats rounds of the per-run means.
A round = one model, every runtime once, order rotated by (repeat + model index), device lock held for the round,
throttle gate (tools/h2h_bench.PinnedGate) before, caps recorded before/after; --redo-throttled re-measures rounds
off the reference cap. Placement validity as in tools/h2h_bench.py (mask read back == cpu, every sample on cpu).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import h2h_bench as hb  # noqa: E402  (adb helpers, DeviceLock, device_info, PinnedGate, round_off_reference)

MAIN = hb.MAIN
DEV = "/data/local/tmp/h2h_dabnn"
hb.DEV = DEV
MODELS = {"dabnn-bireal18": {"blob": "188"}, "dabnn-bireal18-stem": {"blob": "216"}}
RUNTIMES = ["dabnn", "bitino_all", "bitino_int8_full", "dabnn_gbench"]
EVAL = os.path.join(MAIN, ".cache/data/eval128/torchvision_imagenet.f32")
PER = 224 * 224 * 3
PARTS = {"0xd03": "Cortex-A53", "0xd05": "Cortex-A55", "0xd0a": "Cortex-A75", "0xd46": "Cortex-A510",
         "0xd4d": "Cortex-A715", "0xd47": "Cortex-A710", "0xd4e": "Cortex-X3"}


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def bins(a) -> dict:
    sub = "arm64" if a.abi == "arm64" else "armv7"
    return {"dabnn_h2h": os.path.join(a.work, f"build-{sub}/binaries/dabnn_h2h"),
            "benchmark_single_model": os.path.join(a.work, f"build-{sub}/benchmark/benchmark_single_model"),
            "bitino_bench": os.path.join(ROOT, f"build/android-{sub}/bitino_bench"),
            "h2h_run": os.path.join(ROOT, f"build/h2h/h2h_run_{sub}"),
            # sensitivity check only: the same driver + libdabnn compiled with Bitino's -march=armv8.2-a+dotprod
            **({"dabnn_h2h_v82": os.path.join(a.work, "build-arm64-v82/binaries/dabnn_h2h")}
               if "dabnn_v82" in getattr(a, "runtimes", "") else {})}


def setup(a) -> dict:
    hb.sh(f"mkdir -p {DEV}/bin {DEV}/m {DEV}/out", check=True)
    pushed = {}
    for name, src in bins(a).items():
        h = sha(src)
        have = hb.sh(f"sha256sum {DEV}/bin/{name} 2>/dev/null").stdout.split()
        if not have or have[0] != h:
            hb.adb("push", src, f"{DEV}/bin/{name}")
            hb.sh(f"chmod 755 {DEV}/bin/{name}")
        pushed[name] = {"local": src, "sha256": h}
    return pushed


def model_files(a, m: str) -> dict:
    inp = os.path.join(a.work, "inputs", "input0.f32")
    if not os.path.exists(inp):
        os.makedirs(os.path.dirname(inp), exist_ok=True)
        with open(EVAL, "rb") as f:
            open(inp, "wb").write(f.read(PER * 4))
    files = {"input0.f32": inp, "model.dab": os.path.join(a.work, "dab", f"{m}.dab"),
             "model.bitino": os.path.join(MAIN, "artifacts", m, "model.bitino")}
    i8 = os.path.join(MAIN, "artifacts", m, "model_int8_full.bitino")
    if os.path.exists(i8):
        files["model_int8_full.bitino"] = i8
    return files


def push_files(mdir: str, files: dict) -> dict:
    hb.sh(f"mkdir -p {mdir}", check=True)
    have = dict(l.split()[::-1] for l in hb.sh(f"cd {mdir} && sha256sum * 2>/dev/null").stdout.splitlines()
                if len(l.split()) == 2)
    out = {}
    for name, path in files.items():
        h = sha(path)
        if have.get(name) != h:
            hb.adb("push", path, f"{mdir}/{name}")
        out[name] = {"local": path, "sha256": h, "bytes": os.path.getsize(path)}
    return out


def runtime_cmd(rt: str, m: str, a, out_file: str, json_file: str, label: str) -> str:
    mdir = f"{DEV}/m/{m}"
    w, it = a.warmup, a.iterations
    if rt in ("dabnn", "dabnn_v82"):
        exe = "dabnn_h2h" if rt == "dabnn" else "dabnn_h2h_v82"
        return (f"{DEV}/bin/{exe} --model {mdir}/model.dab --inputs {mdir}/input0.f32 --blob {MODELS[m]['blob']} "
                f"--outputs {out_file} --warmup {w - 1} --iterations {it} --json {json_file} --label {label}")
    if rt == "dabnn_gbench":
        return f"{DEV}/bin/benchmark_single_model {mdir}/model.dab --benchmark_format=json"
    model, stage = ("model.bitino", "all") if rt == "bitino_all" else ("model_int8_full.bitino", "int8_full")
    return (f"{DEV}/bin/bitino_bench --model {mdir}/{model} --stage {stage} --threads 1 --warmup {w - 1} "
            f"--iterations {it} --inputs {mdir}/input0.f32 --outputs {out_file} --json {json_file} --label {label}")


def parse_gbench(text: str) -> dict:
    m = re.search(r"^\{.*^\}", text, re.S | re.M)
    if not m:
        return {}
    j = json.loads(m.group(0))
    b = j["benchmarks"][0]
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[b.get("time_unit", "ns")]
    return {"n": int(b["iterations"]), "mean_ms": b["real_time"] * scale, "cpu_mean_ms": b["cpu_time"] * scale,
            "gbench_context": j.get("context")}


def run_one(rt: str, m: str, rep: int, order: int, a, outdir: str, pull_outputs: bool) -> dict:
    tag = f"{m}__{rt}__r{rep}"
    out_file, json_file = f"{DEV}/out/{m}__{rt}.out", f"{DEV}/out/{tag}.json"
    cmd = f"{DEV}/bin/h2h_run --cpus {a.cpu} -- " + runtime_cmd(rt, m, a, out_file, json_file, tag)
    t0 = time.time()
    p = hb.sh(cmd, timeout=a.run_timeout)
    text = p.stdout + p.stderr
    rec = {"model": m, "runtime": rt, "repeat": rep, "order_in_round": order, "cmd": cmd, "exit": p.returncode,
           "wall_s_host": round(time.time() - t0, 2), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    mm = re.search(r"H2H_RUN (\{.*\})", text)
    rec["launcher"] = json.loads(mm.group(1)) if mm else None
    mm = re.search(r"H2H_PIN_WAIT (\{.*\})", text)
    rec["pin_wait"] = json.loads(mm.group(1)) if mm else None
    if rt.startswith("bitino"):
        loc = os.path.join(outdir, "raw", f"{tag}.bitino.json")
        if hb.adb("pull", json_file, loc, check=False).returncode == 0:
            j = json.load(open(loc))
            lat = j["latency"]
            rec["timing"] = {"n": lat["n"], "mean_ms": lat["mean_ms"], "median_ms": lat["median_ms"],
                             "min_ms": lat["min_ms"], "max_ms": lat.get("max_ms"), "std_ms": lat.get("stddev_ms"),
                             "load_ms": lat.get("load_ms"), "prepare_ms": lat.get("session_prepare_ms")}
            cfg = j.get("config") or {}
            rec["bitino"] = {"kernels": j.get("kernels"), "memory": j.get("memory"), "config": cfg,
                             "affinity_errors": cfg.get("affinity_errors"),
                             "cpu_freq_before": j.get("cpu_freq_before"), "cpu_freq_after": j.get("cpu_freq_after"),
                             "version": j.get("bitino_version"), "build": j.get("build")}
    elif rt in ("dabnn", "dabnn_v82"):
        mm = re.search(r"DABNN_H2H (\{.*\})", p.stdout)
        if mm:
            j = json.loads(mm.group(1))
            rec["timing"] = {k: j["latency"][k] for k in ("n", "mean_ms", "median_ms", "min_ms", "max_ms")}
            rec["timing"].update(std_ms=j["latency"]["stddev_ms"], load_ms=j["latency"]["load_ms"],
                                 first_run_ms=j["latency"]["first_run_ms"], runs_ms=j["latency"]["runs_ms"])
            rec["dabnn"] = {k: j.get(k) for k in ("affinity", "cpu_at_end", "blob_shape_hwc", "warmup_untimed",
                                                  "cpu_freq_before", "cpu_freq_after")}
    else:
        rec["timing"] = parse_gbench(p.stdout)
    rec["stdout_tail"] = [l for l in text.strip().splitlines() if not l.startswith("{")][-6:]
    rec["ok"] = p.returncode == 0 and (rec.get("timing") or {}).get("mean_ms") is not None
    if rec["ok"] and pull_outputs and rt != "dabnn_gbench":
        loc = os.path.join(outdir, "outputs", f"{m}__{rt}.out")
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        hb.adb("pull", out_file, loc, check=False)
    return rec


def cmd_run(a) -> int:
    os.environ["ANDROID_SERIAL"] = a.serial
    import benchmark as bm  # freq_caps() (reads ANDROID_SERIAL through adb)

    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    models = a.models.split(",")
    with hb.DeviceLock(a.serial):
        pushed = setup(a)
        info = hb.device_info(a.cpu)
        files = {m: push_files(f"{DEV}/m/{m}", model_files(a, m)) for m in models}
    rts = [r for r in a.runtimes.split(",")]
    caps0 = bm.freq_caps()
    plan = {"tag": a.tag, "serial": a.serial, "device": info, "binaries": pushed, "model_files": files,
            "models": models, "runtimes": rts, "warmup": a.warmup, "iterations": a.iterations, "repeats": a.repeats,
            "cpu": a.cpu, "abi": a.abi, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "caps_at_start": caps0}
    ppath = os.path.join(a.out, "plan.json")
    if os.path.exists(ppath):
        old = json.load(open(ppath))
        for k in ("runtimes", "warmup", "iterations", "cpu", "abi"):
            if old.get(k) != plan[k]:
                raise SystemExit(f"{ppath}: {k} differs from this invocation ({old.get(k)} vs {plan[k]})")
        if {k: v["sha256"] for k, v in old["binaries"].items()} != {k: v["sha256"] for k, v in pushed.items()}:
            raise SystemExit(f"{ppath}: binaries differ from this invocation")
        plan["invocations"] = old.get("invocations", []) + [{k: old.get(k) for k in (
            "models", "repeats", "started_utc", "finished_utc", "caps_at_start", "caps_at_end")}]
        plan["models"] = old["models"] + [m for m in models if m not in old["models"]]
        plan["reference_cap_khz"] = old["reference_cap_khz"]
    if a.ref_cap_khz:
        plan["reference_cap_khz"] = a.ref_cap_khz
    plan.setdefault("reference_cap_khz", caps0["policies"][hb.policy_of(caps0, a.cpu)][0])
    gate = hb.PinnedGate(bm, a.cpu, plan["reference_cap_khz"], a.throttle_max)
    plan["throttle_gate"] = {"max_wait_s": a.throttle_max, "poll_s": 2.0, "rule": hb.PinnedGate.RULE,
                             "reference_cap_khz": plan["reference_cap_khz"]}
    json.dump(plan, open(ppath, "w"), indent=1)
    failures = 0
    for rep in range(a.repeats):
        for mi, m in enumerate(models):
            have = [r for r in rts if r != "bitino_int8_full" or "model_int8_full.bitino" in files[m]]
            k = (rep + mi) % len(have)
            order = have[k:] + have[:k]
            done = {os.path.basename(f) for f in glob.glob(os.path.join(a.out, "raw", f"{m}__*__r{rep}.json"))}
            if all(f"{m}__{r}__r{rep}.json" in done for r in order):
                old = [json.load(open(os.path.join(a.out, "raw", f"{m}__{r}__r{rep}.json"))) for r in order]
                if not (a.redo_throttled and any(hb.round_off_reference(o, a.cpu, plan["reference_cap_khz"]) for o in old)):
                    continue
                os.makedirs(os.path.join(a.out, "raw", "throttled"), exist_ok=True)
                for r in order:
                    for suffix in (".json", ".bitino.json"):
                        f = os.path.join(a.out, "raw", f"{m}__{r}__r{rep}{suffix}")
                        if os.path.exists(f):
                            os.replace(f, os.path.join(a.out, "raw", "throttled", os.path.basename(f)))
            g = gate.wait()
            with hb.DeviceLock(a.serial) as lk:
                before = bm.freq_caps()
                recs = []
                for oi, rt in enumerate(order):
                    attempts = []
                    for _ in range(3):
                        rec = run_one(rt, m, rep, oi, a, a.out, pull_outputs=(rep == 0))
                        la = rec.get("launcher") or {}
                        hist = la.get("cpu_hist") or {}
                        rec["placement_ok"] = (bool(hist) and set(hist) == {a.cpu} and la.get("cpus_allowed_list") == a.cpu
                                               and la.get("allowed_mismatch_samples", 0) == 0)
                        if rt.startswith("bitino") and (rec.get("bitino") or {}).get("affinity_errors"):
                            rec["placement_ok"] = False
                        if rt.startswith("dabnn") and (rec.get("dabnn") or {}).get("affinity") not in (None, a.cpu):
                            rec["placement_ok"] = False
                        if rec["ok"] and not rec["placement_ok"]:
                            rec["ok"] = False
                        attempts.append({k: rec.get(k) for k in ("exit", "launcher", "pin_wait", "placement_ok")})
                        if rec["ok"]:
                            break
                    rec["attempts"] = attempts
                    recs.append(rec)
                after = bm.freq_caps()
            throttled = not gate.ok(before) or not gate.ok(after)
            for rec in recs:
                rec["round"] = {"lock_wait_s": round(lk.waited, 1), "gate": g, "caps_before": before,
                                "caps_after": after, "throttled": throttled, "order": order}
                json.dump(rec, open(os.path.join(a.out, "raw", f"{m}__{rec['runtime']}__r{rep}.json"), "w"), indent=1)
                failures += 0 if rec["ok"] else 1
            line = " ".join(f"{r['runtime']}={r['timing'].get('mean_ms', float('nan')):.2f}" if r["ok"]
                            else f"{r['runtime']}=FAIL" for r in recs)
            cpus = {r["runtime"]: ",".join(sorted((r.get("launcher") or {}).get("cpu_hist", {}))) for r in recs}
            off = {k: v for k, v in cpus.items() if v != a.cpu}
            print(f"[{a.tag} cpu{a.cpu} r{rep} {m}] {line} ms" + (" THROTTLED" if throttled else "")
                  + (f" OFF-CPU {off}" if off else "") + f" (lock wait {lk.waited:.0f} s, gate {g['waited_s']} s)",
                  flush=True)
    plan["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    plan["caps_at_end"] = bm.freq_caps()
    json.dump(plan, open(ppath, "w"), indent=1)
    return 1 if failures else 0


def cmd_eval(a) -> int:
    """All 128 eval images through daBNN and Bitino on the device (untimed, pinned), outputs pulled."""
    os.environ["ANDROID_SERIAL"] = a.serial
    os.makedirs(a.out, exist_ok=True)
    odir = os.path.join(a.work, "eval-outputs", a.tag)  # 512 KB per run: kept out of git, SHA-256 in eval.json
    os.makedirs(odir, exist_ok=True)
    rec = {"tag": a.tag, "outputs_dir": odir, "serial": a.serial, "cpu": a.cpu, "abi": a.abi, "eval_file": EVAL, "eval_sha256": sha(EVAL),
           "runs": []}
    with hb.DeviceLock(a.serial):
        rec["binaries"] = setup(a)
        rec["device"] = hb.device_info(a.cpu)
        push_files(f"{DEV}/eval", {"eval128.f32": EVAL})
        for m in a.models.split(","):
            mdir = f"{DEV}/m/{m}"
            rec.setdefault("model_files", {})[m] = push_files(mdir, model_files(a, m))
            cmds = {"dabnn": f"{DEV}/bin/dabnn_h2h --model {mdir}/model.dab --inputs {DEV}/eval/eval128.f32 "
                             f"--blob {MODELS[m]['blob']} --outputs {DEV}/out/{m}__dabnn.eval --eval 128",
                    "bitino_all": f"{DEV}/bin/bitino_bench --model {mdir}/model.bitino --stage all --threads 1 "
                                  f"--warmup 0 --iterations 1 --inputs {DEV}/eval/eval128.f32 --outputs {DEV}/out/{m}__bitino_all.eval"}
            if os.path.exists(os.path.join(MAIN, "artifacts", m, "model_int8_full.bitino")):
                cmds["bitino_int8_full"] = (f"{DEV}/bin/bitino_bench --model {mdir}/model_int8_full.bitino --stage int8_full "
                                            f"--threads 1 --warmup 0 --iterations 1 --inputs {DEV}/eval/eval128.f32 "
                                            f"--outputs {DEV}/out/{m}__bitino_int8_full.eval")
            for rt, c in cmds.items():
                p = hb.sh(f"{DEV}/bin/h2h_run --cpus {a.cpu} -- {c}", timeout=a.run_timeout)
                loc = os.path.join(odir, f"{m}__{rt}.f32")
                ok = p.returncode == 0 and hb.adb("pull", f"{DEV}/out/{m}__{rt}.eval", loc, check=False).returncode == 0
                mm = re.search(r"H2H_RUN (\{.*\})", p.stdout + p.stderr)
                rec["runs"].append({"model": m, "runtime": rt, "cmd": c, "exit": p.returncode, "ok": ok,
                                    "launcher": json.loads(mm.group(1)) if mm else None,
                                    "sha256": sha(loc) if ok else None})
                print(f"[{a.tag} eval {m} {rt}] exit {p.returncode} ok {ok}", flush=True)
    json.dump(rec, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    return 0 if all(r["ok"] for r in rec["runs"]) else 1


# ------------------------------------------------------------------------------------------------ report
def compare(x, y) -> dict:
    import numpy as np

    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return {"top1_agreement": float((x.argmax(1) == y.argmax(1)).mean()), "n_top1_diff": int((x.argmax(1) != y.argmax(1)).sum()),
            "max_abs": float(d.max()), "median_per_sample_max_abs": float(np.median(d.max(1))),
            "n_samples_max_abs_le_1e-3": int((d.max(1) <= 1e-3).sum()), "bit_identical": bool((x == y).all())}


def cmd_report(a) -> int:
    import numpy as np

    rows, meta = [], {}
    for pdir in sorted(glob.glob(os.path.join(a.indir, "*", "plan.json"))):
        d = os.path.dirname(pdir)
        plan = json.load(open(pdir))
        key = f"{plan['tag']}-cpu{plan['cpu']}"
        meta[key] = plan
        recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "raw", "*__r*.json")))
                if not f.endswith(".bitino.json")]
        by = {}
        for r in recs:
            by.setdefault((r["model"], r["runtime"]), []).append(r)
        for (m, rt), rs in sorted(by.items()):
            ok = [r for r in rs if r["ok"]]
            for r in ok:
                r["off_reference"] = hb.round_off_reference(r, plan["cpu"], plan["reference_cap_khz"])
            good = [r for r in ok if not r["off_reference"]]
            use = good or ok
            means = [r["timing"]["mean_ms"] for r in use]
            hist = {}
            for r in use:
                for c, n in ((r.get("launcher") or {}).get("cpu_hist") or {}).items():
                    hist[c] = hist.get(c, 0) + n
            row = {"key": key, "model": m, "runtime": rt, "n_ok": len(use), "n_runs": len(rs),
                   "off_reference_used": 0 if good else len(use),
                   "median_of_means_ms": statistics.median(means) if means else None,
                   "min_mean_ms": min(means) if means else None, "max_mean_ms": max(means) if means else None,
                   "means_ms": means, "cpu_hist": hist, "off_cpu_samples": sum(n for c, n in hist.items() if c != plan["cpu"]),
                   "maxrss_mb": statistics.median([r["launcher"]["maxrss_kb"] / 1024 for r in use]) if use else None}
            if rt != "dabnn_gbench" and use:
                row["median_of_medians_ms"] = statistics.median([r["timing"]["median_ms"] for r in use])
            if rt.startswith("bitino") and use:
                row["affinity_errors_total"] = sum((r.get("bitino") or {}).get("affinity_errors") or 0 for r in use)
            if rt == "dabnn_gbench" and use:
                row["gbench_iterations"] = [r["timing"]["n"] for r in use]
            rows.append(row)
    # timed-sample outputs (repeat 0): daBNN vs Bitino
    sample0 = []
    for key, plan in meta.items():
        d = os.path.join(a.indir, f"{key}")
        for m in plan["models"]:
            outs = {}
            for rt in ("dabnn", "bitino_all", "bitino_int8_full"):
                f = os.path.join(d, "outputs", f"{m}__{rt}.out")
                if os.path.exists(f):
                    outs[rt] = np.fromfile(f, np.float32).reshape(1, -1)
            ref = np.fromfile(os.path.join(MAIN, "artifacts", m, "reference_logits_f64.f64"), np.float64).reshape(128, -1)[:1]
            for rt, o in outs.items():
                e = {"key": key, "model": m, "runtime": rt, "top1": int(o.argmax()), "ref_f64_top1": int(ref.argmax()),
                     "max_abs_vs_ref_f64": float(np.abs(o - ref).max())}
                if rt != "dabnn" and "dabnn" in outs:
                    e["max_abs_vs_dabnn"] = float(np.abs(o.astype(np.float64) - outs["dabnn"]).max())
                sample0.append(e)
    # 128-image eval set
    manifest = json.load(open(os.path.join(os.path.dirname(EVAL), "manifest.json")))
    labels = np.array([s["label"] for s in manifest["samples"]])
    evals = []
    for ej in sorted(glob.glob(os.path.join(a.indir, "eval-*", "eval.json"))):
        e = json.load(open(ej))
        d = os.path.join(a.work, "eval-outputs", e["tag"])
        for m in sorted({r["model"] for r in e["runs"]}):
            ref = np.fromfile(os.path.join(MAIN, "artifacts", m, "reference_logits_f64.f64"), np.float64).reshape(128, -1)
            emu = np.fromfile(os.path.join(MAIN, "artifacts", m, "dabnn_runtime_logits_emulated.f32"), np.float32).reshape(128, -1)
            outs = {}
            for r in e["runs"]:
                if r["model"] == m and r["ok"]:
                    f = os.path.join(d, f"{m}__{r['runtime']}.f32")
                    assert sha(f) == r["sha256"], f"{f}: SHA-256 differs from eval.json"
                    outs[r["runtime"]] = np.fromfile(f, np.float32).reshape(128, -1)
            ent = {"tag": e["tag"], "cpu": e["cpu"], "model": m,
                   "placement": {r["runtime"]: sorted(((r.get("launcher") or {}).get("cpu_hist") or {}).keys())
                                 for r in e["runs"] if r["model"] == m},
                   "top1_acc": {rt: float((o.argmax(1) == labels).mean()) for rt, o in outs.items()},
                   "ref_f64_top1_acc": float((ref.argmax(1) == labels).mean())}
            ent["top1_acc"]["reference_f64"] = ent["ref_f64_top1_acc"]
            if "dabnn" in outs:
                ent["dabnn_device_vs_dabnn_emulated"] = compare(outs["dabnn"], emu)
                for rt in ("bitino_all", "bitino_int8_full"):
                    if rt in outs:
                        ent[f"{rt}_vs_dabnn"] = compare(outs[rt], outs["dabnn"])
            for rt, o in outs.items():
                ent[f"{rt}_vs_ref_f64"] = compare(o, ref)
            evals.append(ent)
    json.dump({"rows": rows, "sample0": sample0, "eval128": evals, "plans": meta},
              open(os.path.join(a.indir, "summary.json"), "w"), indent=1)

    def cell(key, m, rt, field="median_of_means_ms"):
        r = next((r for r in rows if r["key"] == key and r["model"] == m and r["runtime"] == rt), None)
        return None if not r else r.get(field)

    L = ["# daBNN vs Bitino: head-to-head, 1 thread, batch 1\n",
         "Per configuration: median over rounds of the per-run mean of 50 timed runs (ms); `*` = only off-reference-cap "
         "rounds available. Ratio = daBNN / Bitino (> 1: Bitino faster). Method: docs/comparison-dabnn.md.\n"]
    L.append("| phone | core | model | daBNN | Bitino all | daBNN / Bitino all | Bitino int8_full | daBNN / Bitino int8_full "
             "| daBNN own harness (gbench) | rounds ok (dabnn/all/int8) |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for key, plan in meta.items():
        part = plan["device"]["cpu_parts"].get(plan["cpu"], "?")
        core = PARTS.get(part, part)
        for m in plan["models"]:
            dv, bv, iv, gv = (cell(key, m, rt) for rt in ("dabnn", "bitino_all", "bitino_int8_full", "dabnn_gbench"))
            star = lambda rt: " *" if cell(key, m, rt, "off_reference_used") else ""  # noqa: E731
            f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
            n = "/".join(str(cell(key, m, rt, "n_ok") or 0) for rt in ("dabnn", "bitino_all", "bitino_int8_full"))
            L.append(f"| {plan['device'].get('model')} ({plan['tag']}) | cpu{plan['cpu']} ({core}) | {m} | {f(dv)}{star('dabnn')} "
                     f"| {f(bv)}{star('bitino_all')} | {f'{dv / bv:.2f}x' if dv and bv else 'n/a'} | {f(iv)}{star('bitino_int8_full')} "
                     f"| {f'{dv / iv:.2f}x' if dv and iv else 'n/a'} | {f(gv)} | {n} |")
    L.append("")
    L.append("Run-to-run spread (max/min of the per-run means), median-of-medians and peak RSS:\n")
    L.append("| phone/core | model | runtime | median of medians ms | min..max of means ms | peak RSS MB | off-cpu samples |")
    L.append("|---|---|---|---:|---|---:|---:|")
    for r in rows:
        mm = r.get("median_of_medians_ms")
        L.append(f"| {r['key']} | {r['model']} | {r['runtime']} | {'-' if mm is None else f'{mm:.2f}'} "
                 f"| {r['min_mean_ms']:.2f}..{r['max_mean_ms']:.2f} | {r['maxrss_mb']:.1f} | {r['off_cpu_samples']} |"
                 if r["median_of_means_ms"] is not None else f"| {r['key']} | {r['model']} | {r['runtime']} | FAIL | | | |")
    L.append("")
    if evals:
        L.append("### Correctness on the 128-image eval set (device, untimed)\n")
        L.append("| phone | model | top-1 acc daBNN / Bitino all / int8_full / fp64 ref | top-1 agreement with fp64 ref: daBNN / "
                 "Bitino all / int8_full | Bitino all vs daBNN: top-1 agree, max abs, median per-sample max abs "
                 "| int8_full vs daBNN: top-1 agree, max abs | daBNN device vs daBNN emulated |")
        L.append("|---|---|---|---|---|---|---|")
        for e in evals:
            acc = " / ".join(f"{e['top1_acc'][k]:.4f}" if k in e["top1_acc"] else "-"
                             for k in ("dabnn", "bitino_all", "bitino_int8_full", "reference_f64"))
            b = e.get("bitino_all_vs_dabnn")
            i = e.get("bitino_int8_full_vs_dabnn")
            em = e.get("dabnn_device_vs_dabnn_emulated")
            agr = " / ".join(f"{e[k + '_vs_ref_f64']['top1_agreement']:.4f}" if k + "_vs_ref_f64" in e else "-"
                             for k in ("dabnn", "bitino_all", "bitino_int8_full"))
            L.append(f"| {e['tag']} | {e['model']} | {acc} | {agr} | "
                     + (f"{b['top1_agreement']:.4f}, {b['max_abs']:.3g}, {b['median_per_sample_max_abs']:.3g}" if b else "-")
                     + " | " + (f"{i['top1_agreement']:.4f}, {i['max_abs']:.3g}" if i else "-")
                     + " | " + (("bit-identical" if em["bit_identical"] else f"max abs {em['max_abs']:.3g}, top-1 {em['top1_agreement']:.4f}") if em else "-")
                     + " |")
        L.append("")
    if sample0:
        L.append("### Timed input (eval image 0), outputs of repeat 0\n")
        L.append("| phone/core | model | runtime | top-1 (fp64 ref top-1) | max abs vs fp64 ref | max abs vs daBNN |")
        L.append("|---|---|---|---|---:|---:|")
        for e in sample0:
            L.append(f"| {e['key']} | {e['model']} | {e['runtime']} | {e['top1']} ({e['ref_f64_top1']}) | {e['max_abs_vs_ref_f64']:.3g} "
                     f"| {e.get('max_abs_vs_dabnn', float('nan')):.3g} |")
        L.append("")
    open(a.md, "w").write("\n".join(L))
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "eval"):
        r = sub.add_parser(name)
        r.add_argument("--serial", required=True)
        r.add_argument("--tag", required=True)
        r.add_argument("--cpu", required=True, help="the one CPU every runtime is pinned to")
        r.add_argument("--abi", choices=["arm64", "armv7"], default="arm64")
        r.add_argument("--work", required=True, help="scripts/h2h_dabnn.sh work dir (daBNN builds, dab/)")
        r.add_argument("--models", default=",".join(MODELS))
        r.add_argument("--run-timeout", type=float, default=3600.0)
        r.add_argument("--out", required=True)
        if name == "run":
            r.add_argument("--runtimes", default=",".join(RUNTIMES))
            r.add_argument("--repeats", type=int, default=5)
            r.add_argument("--warmup", type=int, default=10)
            r.add_argument("--iterations", type=int, default=50)
            r.add_argument("--throttle-max", type=float, default=600.0)
            r.add_argument("--ref-cap-khz", type=int)
            r.add_argument("--redo-throttled", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--in", dest="indir", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--md", required=True)
    a = ap.parse_args()
    return {"run": cmd_run, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
