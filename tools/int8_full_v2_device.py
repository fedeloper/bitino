#!/usr/bin/env python3
"""Phone runs of the int8_full v2 files (model_int8_full_v2.bitino, docs/int8.md "int8_full v2"): 1-thread latency of
the current int8_full files, the v2 file and the float stage `all`, and the 4-thread eval-set outputs of the v2 files.

    python3 tools/int8_full_v2_device.py bench --serial e3698ea7 --pin 7 --models m1 m2 ... --out reports/int8_full_v2/speed
    python3 tools/int8_full_v2_device.py outputs --serial e3698ea7 --models m1 ... --out reports/int8_full_v2/outputs

bench: per model, in ONE hold of the phone's lock (/tmp/bitino-device-<serial>.lock), --repeats interleaved rounds of
(model_int8_full.bitino [, model_int8_full_log2.bitino], model_int8_full_v2.bitino at stage int8_full, model.bitino at
stage all), each `bitino_bench --threads 1 --affinity PIN`; a run counts only with affinity_errors <= warmup + 1 (else it
is repeated once with more warmup). Writes <out>/<tag>/<model>.json with every run and the medians of the per-run medians.
outputs: stage int8_full of every v2 file on the model's eval128 set (scripts/eval_sets.sh), 4 threads, pulled to
<out>/<tag>/<model>_int8_full_v2.f32 (compared with the numpy oracle and across phones by tools/int8_full_v2.py check).
Device files live in /data/local/tmp/i8v2 (binary, models, data) and are deleted with `cleanup`.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADB = os.environ.get("ADB", os.path.expanduser("~/Android/Sdk/platform-tools/adb"))
D = "/data/local/tmp/i8v2"


def adb(serial, *args, check=True, capture=True):
    r = subprocess.run([ADB, "-s", serial, *args], capture_output=capture, text=True)
    if check and r.returncode:
        raise RuntimeError(f"adb {' '.join(args)}: {r.stderr[-500:]}")
    return r.stdout if capture else ""


def locked(serial, *args):
    """adb shell under the phone's lock (every command that runs code on the phone)."""
    lock = f"/tmp/bitino-device-{serial}.lock"
    r = subprocess.run(["flock", lock, ADB, "-s", serial, "shell", *args], capture_output=True, text=True)
    return r


def eval_of(m):
    out = subprocess.run(["bash", "-c", 'source scripts/eval_sets.sh; eval_of "$0"', m], cwd=ROOT, capture_output=True,
                         text=True).stdout.split()
    edir, pp = out[0], out[1]
    man = json.load(open(os.path.join(ROOT, edir, "manifest.json")))
    return os.path.join(edir, man["preprocessing"][pp]["file"])


def files_of(m):
    a = os.path.join(ROOT, "artifacts", m)
    cfg = []
    for f, tag in (("model_int8_full.bitino", "int8_full"), ("model_int8_full_log2.bitino", "int8_full_log2"),
                   ("model_int8_full_v2.bitino", "int8_full_v2")):
        if os.path.exists(os.path.join(a, f)):
            cfg.append((tag, f, "int8_full"))
    cfg.append(("float_all", "model.bitino", "all"))
    return cfg


def push_binary(a):
    adb(a.serial, "shell", "mkdir", "-p", f"{D}/models", f"{D}/data", f"{D}/out")
    adb(a.serial, "push", os.path.join(ROOT, a.binary), f"{D}/bitino_bench")
    adb(a.serial, "shell", "chmod", "755", f"{D}/bitino_bench")


def push_models(a, m, cfg):
    adb(a.serial, "shell", "mkdir", "-p", f"{D}/models/{m}")
    for _, f, _ in cfg:
        adb(a.serial, "push", os.path.realpath(os.path.join(ROOT, "artifacts", m, f)), f"{D}/models/{m}/{f}")


def drop_models(a, m):
    adb(a.serial, "shell", "rm", "-rf", f"{D}/models/{m}", check=False)


def one_run(a, m, f, stage, warmup, iters):
    js = f"{D}/out/r.json"
    cmd = (f"{D}/bitino_bench --model {D}/models/{m}/{f} --stage {stage} --threads 1 --affinity {a.pin} "
           f"--warmup {warmup} --iterations {iters} --json {js} > /dev/null 2>&1; cat {js}; rm -f {js}")
    r = subprocess.run([ADB, "-s", a.serial, "shell", cmd], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    caps = {c["cpu"]: c.get("max_khz") for c in d.get("cpu_freq_after", [])}
    return {"median_ms": d["latency"]["median_ms"], "min_ms": d["latency"]["min_ms"], "affinity_errors": d["config"]["affinity_errors"],
            "warmup": warmup, "iterations": iters, "cur_khz_pin_after": next((c.get("cur_khz") for c in d.get("cpu_freq_after", [])
                                                                              if c["cpu"] == a.pin), None),
            "max_khz_pin": caps.get(a.pin), "thermal_after_c": d.get("thermal_after_c"), "utc": d.get("timestamp_utc")}


def bench(a):
    tag = a.tag
    od = os.path.join(ROOT, a.out, tag)
    os.makedirs(od, exist_ok=True)
    push_binary(a)
    lock = f"/tmp/bitino-device-{a.serial}.lock"
    for m in a.models:
        outp = os.path.join(od, f"{m}.json")
        if os.path.exists(outp) and not a.force:
            print(f"{m}: done before ({outp})", flush=True)
            continue
        cfg = files_of(m)
        if not any(t == "int8_full_v2" for t, _, _ in cfg):
            print(f"{m}: no v2 file", flush=True)
            continue
        push_models(a, m, cfg)
        # size the iterations by a probe of the float file (outside the lock: a 1-iteration run is not a timing)
        runs = {t: [] for t, _, _ in cfg}
        fd = os.open(lock, os.O_RDONLY | os.O_CREAT)
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            probe = one_run(a, m, "model.bitino", "all", 1, 3)
            ms = probe["median_ms"] if probe else 10.0
            iters = int(min(200, max(10, a.budget_ms / max(ms, 0.01))))
            warmup = int(min(50, max(3, iters // 5)))
            for rep in range(a.repeats):
                for t, f, st in (cfg if rep % 2 == 0 else list(reversed(cfg))):
                    r = one_run(a, m, f, st, warmup, iters)
                    if r is not None and r["affinity_errors"] > warmup + 1:
                        r2 = one_run(a, m, f, st, warmup * 2 + 5, iters)
                        r = r2 if r2 is not None else r
                        r["retried"] = True
                    if r is not None:
                        r["repeat"] = rep
                        runs[t].append(r)
                    if a.cooldown:
                        time.sleep(a.cooldown)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        drop_models(a, m)
        res = {"model": m, "serial": a.serial, "pin": a.pin, "binary": a.binary, "binary_sha256": sha(os.path.join(ROOT, a.binary)),
               "files_sha256": {t: sha(os.path.realpath(os.path.join(ROOT, "artifacts", m, f))) for t, f, _ in cfg},
               "iterations": iters, "warmup": warmup, "runs": runs, "median_ms": {}}
        for t, rr in runs.items():
            ok = [r["median_ms"] for r in rr if r["affinity_errors"] <= r["warmup"] + 1]
            res["median_ms"][t] = statistics.median(ok) if ok else None
            res.setdefault("valid_runs", {})[t] = len(ok)
        json.dump(res, open(outp, "w"), indent=1)
        print(f"{m}: " + ", ".join(f"{t} {v:.3f}" if v else f"{t} -" for t, v in res["median_ms"].items()), flush=True)


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def outputs(a):
    od = os.path.join(ROOT, a.out, a.tag)
    os.makedirs(od, exist_ok=True)
    push_binary(a)
    pushed = {}
    for m in a.models:
        f = os.path.join(ROOT, "artifacts", m, "model_int8_full_v2.bitino")
        if not os.path.exists(f):
            continue
        src = eval_of(m)
        if src not in pushed:
            dst = f"{D}/data/" + src.replace("/", "_").replace(".", "_")
            adb(a.serial, "push", os.path.join(ROOT, src), dst)
            pushed[src] = dst
        adb(a.serial, "shell", "mkdir", "-p", f"{D}/models/{m}")
        adb(a.serial, "push", os.path.realpath(f), f"{D}/models/{m}/model_int8_full_v2.bitino")
        out = f"{D}/out/{m}_int8_full_v2.f32"
        r = locked(a.serial, f"{D}/bitino_bench --model {D}/models/{m}/model_int8_full_v2.bitino --inputs {pushed[src]} "
                             f"--outputs {out} --stage int8_full --threads 4 --warmup 0 --iterations 1 > /dev/null")
        adb(a.serial, "pull", out, os.path.join(od, f"{m}_int8_full_v2.f32"))
        adb(a.serial, "shell", "rm", "-f", out)
        drop_models(a, m)
        open(os.path.join(od, f"{m}_int8_full_v2.f32.bin"), "w").write(sha(os.path.join(ROOT, a.binary)) + "\n")
        print(f"{m}: {'ok' if r.returncode == 0 else 'rc=' + str(r.returncode)}", flush=True)
    for dst in pushed.values():
        adb(a.serial, "shell", "rm", "-f", dst, check=False)


def identity(a):
    """Existing behaviour: the current files (stage int8_full: model_int8_full / model_int8_full_log2; float stages of
    model.bitino) on the eval128 sets, 4 threads, byte-compared with this phone's stored outputs
    (<main checkout>/artifacts/<m>/device/<ref-tag>/<m>_<stage>.f32, written by scripts/device_final_suite.sh)."""
    od = os.path.join(ROOT, a.out, a.tag)
    os.makedirs(od, exist_ok=True)
    push_binary(a)
    pushed, rows = {}, {}
    main_art = os.path.realpath(os.path.join(ROOT, "artifacts"))
    for m in a.models:
        for st in a.stages:
            f = {"int8_full": "model_int8_full.bitino", "int8_full_log2": "model_int8_full_log2.bitino"}.get(st, "model.bitino")
            src_model = os.path.realpath(os.path.join(ROOT, "artifacts", m, f))
            ref = os.path.join(a.ref_root, m, "device", a.ref_tag, f"{m}_{st}.f32")
            if not os.path.exists(src_model) or not os.path.exists(ref):
                continue
            src = eval_of(m)
            if src not in pushed:
                dst = f"{D}/data/" + src.replace("/", "_").replace(".", "_")
                adb(a.serial, "push", os.path.join(ROOT, src), dst)
                pushed[src] = dst
            adb(a.serial, "shell", "mkdir", "-p", f"{D}/models/{m}")
            adb(a.serial, "push", src_model, f"{D}/models/{m}/{f}")
            out = f"{D}/out/{m}_{st}.f32"
            rst = "int8_full" if st.startswith("int8_full") else st
            locked(a.serial, f"{D}/bitino_bench --model {D}/models/{m}/{f} --inputs {pushed[src]} --outputs {out} --stage {rst} "
                             f"--threads 4 --warmup 0 --iterations 1 > /dev/null")
            loc = os.path.join(od, f"{m}_{st}.f32")
            adb(a.serial, "pull", out, loc, check=False)
            adb(a.serial, "shell", "rm", "-f", out, check=False)
            same = os.path.exists(loc) and open(loc, "rb").read() == open(ref, "rb").read()
            rows[f"{m}:{st}"] = "IDENTICAL" if same else "DIFF"
            if os.path.exists(loc):
                os.remove(loc)
            print(f"{m} {st}: {rows[f'{m}:{st}']}", flush=True)
        drop_models(a, m)
    for dst in pushed.values():
        adb(a.serial, "shell", "rm", "-f", dst, check=False)
    json.dump({"serial": a.serial, "ref_tag": a.ref_tag, "binary_sha256": sha(os.path.join(ROOT, a.binary)), "results": rows,
               "identical": sum(v == "IDENTICAL" for v in rows.values()), "total": len(rows)},
              open(os.path.join(od, "identity.json"), "w"), indent=1)
    print(f"{sum(v == 'IDENTICAL' for v in rows.values())} / {len(rows)} identical", flush=True)


def cleanup(a):
    adb(a.serial, "shell", "rm", "-rf", D, check=False)
    print(f"{a.serial}: {D} removed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("bench", "outputs", "identity", "cleanup"))
    ap.add_argument("--stages", nargs="*", default=["int8_full", "int8_full_log2"])
    ap.add_argument("--ref-tag", default="cph2551")
    ap.add_argument("--ref-root", default=os.path.join(ROOT, "..", "..", "..", "artifacts"),
                    help="artifacts directory holding the stored phone outputs (the main checkout's)")
    ap.add_argument("--serial", required=True)
    ap.add_argument("--pin", type=int, default=7)
    ap.add_argument("--binary", default="build/android-arm64/bitino_bench")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out", default="reports/int8_full_v2/speed")
    ap.add_argument("--tag", required=False)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget-ms", type=float, default=1500.0, help="timed milliseconds per run (iterations = budget / latency)")
    ap.add_argument("--cooldown", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.tag = a.tag or a.serial
    a.ref_root = os.path.realpath(a.ref_root)
    return {"bench": bench, "outputs": outputs, "identity": identity, "cleanup": cleanup}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
