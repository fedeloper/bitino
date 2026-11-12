#!/usr/bin/env python3
"""Head-to-head latency benchmark on a phone: Bitino (bitino_bench) vs Larq Compute Engine (lce_benchmark_model)
vs plain TensorFlow Lite (benchmark_model, fp32 and int8), 1 thread, batch 1, same models and weights.

  python3 tools/h2h_bench.py run --serial e3698ea7 --tag cph2551 --cpu 7 --abi arm64 \
      --models-dir <h2h_models> --bins-dir <h2h_dl> --out reports/h2h/<date>/cph2551
  python3 tools/h2h_bench.py report --in reports/h2h/<date> --models-dir <h2h_models> --md reports/h2h/<date>/summary.md

Method (docs/comparison-lce-tflite.md):
- Every runtime runs under tools/h2h_run.c (`h2h_run --cpus <cpu> -- <binary> ...`): affinity set with
  sched_setaffinity() in the launcher's child before exec (none of the benchmark binaries pins itself), the CPU each
  thread actually ran on sampled every 25 ms, peak RSS from wait4() ru_maxrss.
- Same policy for all: 1 thread, 10 untimed runs, then 50 timed runs on eval sample 0 (larq_zoo preprocessing).
  TFLite/LCE: --warmup_runs=10 --warmup_min_secs=0 --num_runs=50 --min_secs=0, input via --input_layer_value_files;
  Bitino: one correctness run on the same input (--inputs) + --warmup 9, --iterations 50. Each tool's default
  float path is kept (the TFLite runtime applies XNNPACK to float ops by default in both TFLite binaries).
- Statistic: per run, the mean of the 50 timed runs (the only per-run statistic benchmark_model prints);
  per configuration, the median over --repeats runs. Bitino's per-run median is kept as well.
- Interleaving: a round = one model, every runtime once, in an order rotated by (repeat + model index); the
  device lock (/tmp/bitino-device-<serial>.lock, flock(2), shared with the other jobs) is held for the whole round.
  Before every round the throttle gate of tools/benchmark.py waits (at most --throttle-max s) while the Android
  thermal status is > 0 or a cpufreq cap is below the highest seen; caps/thermal status/skin temperature are
  recorded before and after every round.
- The outputs of every runtime on the timed sample are pulled in repeat 0 and compared in `report`.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN = os.environ.get("BITINO_MAIN", "~/BNN arm developments")
ADB = os.path.expanduser("~/Android/Sdk/platform-tools/adb")
DEV = "/data/local/tmp/h2h"
RUNTIMES = ["bitino_all", "bitino_neon", "lce", "tflite_fp32", "tflite_int8"]
BINS = {  # local file names in --bins-dir (downloads) / build dirs
    "arm64": {"lce": "lce_benchmark_model_android_arm64", "tflite": "android_aarch64_benchmark_model",
              "bitino": os.path.join(ROOT, "build/android-arm64/bitino_bench"), "h2h_run": os.path.join(ROOT, "build/h2h/h2h_run_arm64")},
    # No Android armeabi-v7a LCE binary is published: the official Linux AArch32 build (glibc, -march=armv7-a
    # -mfpu=neon-vfpv4) runs through Debian bookworm's armhf glibc loader pushed next to it (--glibc-dir).
    "armv7": {"lce": "lce_benchmark_model_aarch32", "tflite": "android_arm_benchmark_model",
              "bitino": os.path.join(ROOT, "build/android-armv7/bitino_bench"), "h2h_run": os.path.join(ROOT, "build/h2h/h2h_run_armv7")},
}


def adb(*args, check=True, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run([ADB, *args], check=check, capture_output=True, text=True, timeout=timeout)


def sh(cmd: str, timeout=None, check=False) -> subprocess.CompletedProcess:
    return adb("shell", cmd, check=check, timeout=timeout)


class DeviceLock:
    def __init__(self, serial: str):
        self.path = f"/tmp/bitino-device-{serial}.lock"

    def __enter__(self):
        self.f = open(self.path, "a")
        t0 = time.time()
        fcntl.flock(self.f, fcntl.LOCK_EX)
        self.waited = time.time() - t0
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def device_info(cpu: str) -> dict:
    q = lambda c: sh(c).stdout.strip()  # noqa: E731
    info = {k: q(f"getprop {p}") for k, p in (("model", "ro.product.model"), ("android", "ro.build.version.release"),
                                               ("sdk", "ro.build.version.sdk"), ("soc", "ro.soc.model"),
                                               ("platform", "ro.board.platform"), ("abilist", "ro.product.cpu.abilist"),
                                               ("build", "ro.build.fingerprint"))}
    info["kernel"] = q("uname -a")
    info["cpus"] = {}
    for c in range(8):
        base = f"/sys/devices/system/cpu/cpu{c}/cpufreq"
        info["cpus"][str(c)] = q(f"cat {base}/cpuinfo_max_freq {base}/scaling_max_freq {base}/scaling_governor 2>/dev/null"
                                 " | tr '\\n' ' '")
    parts = q("grep -E 'processor|CPU part' /proc/cpuinfo | tr '\\n' ' '")
    info["cpu_parts"] = dict(re.findall(r"processor\s*:\s*(\d+)\s+CPU part\s*:\s*(0x[0-9a-f]+)", parts))
    info["pinned_cpu"] = cpu
    info["battery"] = q("dumpsys battery | grep -E 'level|temperature|powered'")
    info["wakefulness"] = q("dumpsys power | grep -m1 mWakefulness=")
    info["core_ctl_cpu"] = q(f"cat /sys/devices/system/cpu/cpu{cpu}/core_ctl/enable 2>/dev/null")
    return info


def runtime_cmd(rt: str, mdir: str, a, out_file: str, json_file: str, label: str) -> str:
    w, it = a.warmup, a.iterations
    if rt.startswith("bitino"):
        stage = rt.split("_", 1)[1]
        # 1 correctness run on the input (outside timing) + warmup-1 = the same 10 untimed runs as the TFLite tools
        return (f"{DEV}/bin/bitino_bench --model {mdir}/model.bitino --stage {stage} --threads 1 --warmup {w - 1} "
                f"--iterations {it} --inputs {mdir}/input0.f32 --outputs {out_file} --json {json_file} --label {label}")
    binary = f"{DEV}/bin/lce_benchmark_model" if rt == "lce" else f"{DEV}/bin/benchmark_model"
    if rt == "lce" and a.abi == "armv7":
        binary = f"{DEV}/glibc/ld-linux-armhf.so.3 --library-path {DEV}/glibc {DEV}/bin/lce_benchmark_model"
    graph = {"lce": "lce.tflite", "tflite_fp32": "tflite_fp32.tflite", "tflite_int8": "tflite_int8.tflite",
             "lcebin_fp32": "tflite_fp32.tflite"}[rt]
    if rt == "lcebin_fp32":
        binary = f"{DEV}/bin/lce_benchmark_model"
    inp = "input0_int8.bin" if rt == "tflite_int8" else "input0.f32"
    name = "serving_default_input_1_0"
    return (f"{binary} --graph={mdir}/{graph} --num_threads=1 --warmup_runs={w} --warmup_min_secs=0 --num_runs={it} "
            f"--min_secs=0 --max_secs=100000 --input_layer={name} --input_layer_shape=1,224,224,3 "
            f"--input_layer_value_files={name}:{mdir}/{inp} --output_filepath={out_file}")


def parse_tflite(text: str) -> dict:
    r = {}
    # "count=50 first=... curr=... min=... max=... avg=... std=... [p5=... median=... p95=...]" (warm-up line, then the
    # timed line); values in us, large ones in %g notation (avg=1.68347e+06)
    num = r"[-+0-9.eE]+"
    counts = [dict(re.findall(rf"(\w+)=({num})", l)) for l in text.splitlines() if re.match(r"(INFO: )?count=\d+ ", l.strip())]
    if len(counts) >= 2:
        c, w = counts[-1], counts[-2]
        r.update(n=int(c["count"]), first_ms=float(c["first"]) / 1e3, min_ms=float(c["min"]) / 1e3,
                 max_ms=float(c["max"]) / 1e3, mean_ms=float(c["avg"]) / 1e3, std_ms=float(c["std"]) / 1e3,
                 warmup_n=int(w["count"]), warmup_mean_ms=float(w["avg"]) / 1e3)
        if "median" in c:  # printed by the 2025 nightly tool, not by LCE 0.16's
            r["median_ms"] = float(c["median"]) / 1e3
    m = re.search(rf"Init: ({num}), First inference: ({num})", text)
    if m:
        r["init_ms"], r["first_inference_ms"] = float(m.group(1)) / 1e3, float(m.group(2)) / 1e3
    m = re.search(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \((\w+)\) node, yielding (\d+) partitions", text)
    if m:
        r["delegate"] = {"name": m.group(3), "nodes_delegated": int(m.group(1)), "nodes_total": int(m.group(2)),
                         "partitions": int(m.group(4))}
    m = re.search(r"Memory footprint delta from the start of the tool \(MB\): init=([\d.]+) overall=([\d.]+)", text)
    if m:
        r["tool_memory_delta_mb"] = {"init": float(m.group(1)), "overall": float(m.group(2))}
    return r


def run_one(rt: str, model: str, rep: int, order: int, a, outdir: str, pull_outputs: bool) -> dict:
    mdir = f"{DEV}/m/{model}"
    tag = f"{model}__{rt}__r{rep}"
    out_file, json_file = f"{DEV}/out/{model}__{rt}.out", f"{DEV}/out/{tag}.json"
    bench = runtime_cmd(rt, mdir, a, out_file, json_file, tag)
    cmd = f"{DEV}/bin/h2h_run --cpus {a.cpu} -- {bench}"
    t0 = time.time()
    p = sh(cmd, timeout=a.run_timeout)
    text = p.stdout + p.stderr
    rec = {"model": model, "runtime": rt, "repeat": rep, "order_in_round": order, "cmd": cmd, "exit": p.returncode,
           "wall_s_host": round(time.time() - t0, 2), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    m = re.search(r"H2H_RUN (\{.*\})", text)
    rec["launcher"] = json.loads(m.group(1)) if m else None
    m = re.search(r"H2H_PIN_WAIT (\{.*\})", text)
    rec["pin_wait"] = json.loads(m.group(1)) if m else None
    if rt.startswith("bitino"):
        loc = os.path.join(outdir, "raw", f"{tag}.bitino.json")
        if adb("pull", json_file, loc, check=False).returncode == 0:
            j = json.load(open(loc))
            lat = j["latency"]
            rec["timing"] = {"n": lat["n"], "mean_ms": lat["mean_ms"], "median_ms": lat["median_ms"], "min_ms": lat["min_ms"],
                             "max_ms": lat.get("max_ms"), "std_ms": lat.get("stddev_ms"), "load_ms": lat.get("load_ms"),
                             "prepare_ms": lat.get("session_prepare_ms")}
            rec["bitino"] = {"kernels": j.get("kernels"), "memory": j.get("memory"), "config": j.get("config"),
                             "cpu_freq_before": j.get("cpu_freq_before"), "cpu_freq_after": j.get("cpu_freq_after"),
                             "version": j.get("bitino_version"), "build": j.get("build")}
        rec["stdout_tail"] = text.strip().splitlines()[-4:]
    else:
        rec["timing"] = parse_tflite(text)
        rec["stdout_tail"] = [l for l in text.strip().splitlines() if not l.startswith("INFO: Log parameter")][-30:]
    ok = p.returncode == 0 and rec.get("timing", {}).get("mean_ms") is not None
    rec["ok"] = ok
    if ok and pull_outputs:
        loc = os.path.join(outdir, "outputs", f"{model}__{rt}.out")
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        adb("pull", out_file, loc, check=False)
    return rec


def push_model(model: str, a, rts: list[str]) -> None:
    src = os.path.join(a.models_dir, model)
    mdir = f"{DEV}/m/{model}"
    sh(f"mkdir -p {mdir}", check=True)
    files = {"input0.f32": os.path.join(src, "input0.f32")}
    if any(r.startswith("bitino") for r in rts):
        files["model.bitino"] = os.path.join(MAIN, "artifacts", model, "model.bitino")
    if "lce" in rts:
        files["lce.tflite"] = os.path.join(src, "lce.tflite")
    if "tflite_fp32" in rts or "lcebin_fp32" in rts:
        files["tflite_fp32.tflite"] = os.path.join(src, "tflite_fp32.tflite")
    if "tflite_int8" in rts:
        files["tflite_int8.tflite"] = os.path.join(src, "tflite_int8.tflite")
        files["input0_int8.bin"] = os.path.join(src, "input0_int8.bin")
    have = dict(l.split()[::-1] for l in sh(f"cd {mdir} && sha256sum * 2>/dev/null").stdout.splitlines() if len(l.split()) == 2)
    import hashlib

    for name, path in files.items():
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if have.get(name) != h:
            adb("push", path, f"{mdir}/{name}")


def setup(a) -> dict:
    b = BINS[a.abi]
    sh(f"mkdir -p {DEV}/bin {DEV}/m {DEV}/out", check=True)
    pushed = {}
    for key, dst in (("bitino", "bitino_bench"), ("h2h_run", "h2h_run"), ("lce", "lce_benchmark_model"),
                     ("tflite", "benchmark_model")):
        src = b[key] if os.path.isabs(b[key]) else os.path.join(a.bins_dir, b[key])
        if not os.path.exists(src):
            print(f"missing {key} binary {src}", file=sys.stderr)
            continue
        adb("push", src, f"{DEV}/bin/{dst}")
        sh(f"chmod 755 {DEV}/bin/{dst}")
        import hashlib

        pushed[key] = {"local": src, "sha256": hashlib.sha256(open(src, "rb").read()).hexdigest()}
    if a.abi == "armv7" and a.glibc_dir:
        import hashlib

        sh(f"mkdir -p {DEV}/glibc", check=True)
        pushed["glibc"] = {}
        for f in sorted(os.listdir(a.glibc_dir)):
            adb("push", os.path.join(a.glibc_dir, f), f"{DEV}/glibc/{f}")
            pushed["glibc"][f] = hashlib.sha256(open(os.path.join(a.glibc_dir, f), "rb").read()).hexdigest()
        sh(f"chmod 755 {DEV}/glibc/*")
    elif a.abi == "armv7":
        pushed.pop("lce", None)
    return pushed


def policy_of(caps: dict, cpu: str) -> str:
    """cpufreq policy (its first CPU) that contains cpu: the largest policy number <= cpu."""
    return str(max(int(p) for p in caps["policies"] if int(p) <= int(cpu)))


class PinnedGate:
    """Before each round: wait (poll 2 s, at most max_s; 10 s after a timeout) until the Android thermal status is 0
    and the scaling_max_freq of the pinned CPU's policy equals the plan's reference cap (the value at the start of the
    first invocation). A different cap either way (a lower thermal/vendor cap, or the higher screen-on cap) is a
    different clock. Caps of the other clusters are recorded, not gated on (1-thread runs on the pinned core)."""
    RULE = "thermal status 0 and pinned-policy scaling_max_freq == reference_cap_khz, before and after the round"

    def __init__(self, bm, cpu: str, ref_khz: int, max_s: float):
        self.bm, self.cpu, self.ref, self.max_s, self.gave_up = bm, cpu, int(ref_khz), max_s, False

    def ok(self, st) -> bool:
        if not st:
            return False
        return (st.get("thermal_status") or 0) == 0 and st["policies"][policy_of(st, self.cpu)][0] == self.ref

    def wait(self) -> dict:
        t0, limit = time.time(), (10.0 if self.gave_up else self.max_s)
        st = self.bm.freq_caps()
        while not self.ok(st) and time.time() - t0 < limit:
            time.sleep(2.0)
            st = self.bm.freq_caps()
        bad = not self.ok(st)
        self.gave_up = bad
        return {"waited_s": round(time.time() - t0, 1), "before": st, "timeout": bad, "mode": "pinned_reference_cap",
                "reference_cap_khz": self.ref, "throttled_before": bad}


def round_off_reference(rec: dict, cpu: str, ref_khz: int) -> bool:
    """The PinnedGate criterion applied to a recorded round (also to rounds recorded before the gate existed)."""
    rd = rec.get("round") or {}
    for st in (rd.get("caps_before"), rd.get("caps_after")):
        if not st or (st.get("thermal_status") or 0) != 0 or st["policies"][policy_of(st, cpu)][0] != int(ref_khz):
            return True
    return False


def cmd_run(a) -> int:
    os.environ["ANDROID_SERIAL"] = a.serial
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import benchmark as bm  # freq_caps() / ThrottleGate (reads ANDROID_SERIAL through adb)

    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    models = a.models.split(",")
    rts = a.runtimes.split(",")
    with DeviceLock(a.serial):
        pushed = setup(a)
        info = device_info(a.cpu)
    if "lce" not in pushed:
        rts = [r for r in rts if r not in ("lce", "lcebin_fp32")]
    caps0 = bm.freq_caps()
    plan = {"tag": a.tag, "serial": a.serial, "device": info, "binaries": pushed, "models": models, "runtimes": rts,
            "warmup": a.warmup, "iterations": a.iterations, "repeats": a.repeats, "cpu": a.cpu, "abi": a.abi,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "caps_at_start": caps0}
    # Several invocations may write to one --out (e.g. models added later): keep the earlier ones in "invocations"
    # and the union of the models in "models" (binaries, runtimes and policy must be the same).
    ppath = os.path.join(a.out, "plan.json")
    if os.path.exists(ppath):
        old = json.load(open(ppath))
        for k in ("runtimes", "warmup", "iterations", "cpu", "abi"):
            if old.get(k) != plan[k]:
                raise SystemExit(f"{ppath}: {k} differs from this invocation ({old.get(k)} vs {plan[k]})")
        if {k: v["sha256"] for k, v in old["binaries"].items() if "sha256" in v} != \
                {k: v["sha256"] for k, v in pushed.items() if "sha256" in v}:
            raise SystemExit(f"{ppath}: binaries differ from this invocation")
        plan["invocations"] = old.get("invocations", []) + [{k: old.get(k) for k in ("models", "repeats", "started_utc",
                                                                                    "finished_utc", "caps_at_start", "caps_at_end")}]
        plan["models"] = old["models"] + [m for m in models if m not in old["models"]]
        if old.get("reference_cap_khz"):
            plan["reference_cap_khz"] = old["reference_cap_khz"]
        else:  # plans written before the reference cap was recorded: the caps at the start of their first invocation
            first = (plan["invocations"][0] or {}).get("caps_at_start") or old["caps_at_start"]
            plan["reference_cap_khz"] = first["policies"][policy_of(first, a.cpu)][0]
    if a.ref_cap_khz:
        plan["reference_cap_khz"] = a.ref_cap_khz
    plan.setdefault("reference_cap_khz", caps0["policies"][policy_of(caps0, a.cpu)][0])
    gate = PinnedGate(bm, a.cpu, plan["reference_cap_khz"], a.throttle_max)
    plan["throttle_gate"] = {"max_wait_s": a.throttle_max, "poll_s": 2.0, "rule": PinnedGate.RULE,
                             "reference_cap_khz": plan["reference_cap_khz"]}
    json.dump(plan, open(ppath, "w"), indent=1)
    for m in models:
        missing = [r for r in rts if r != "lcebin_fp32" and not r.startswith("bitino") and
                   not os.path.exists(os.path.join(a.models_dir, m, {"lce": "lce.tflite"}.get(r, f"{r}.tflite")))]
        if missing:
            print(f"{m}: missing {missing}; skipping those runtimes for this model", file=sys.stderr)
        with DeviceLock(a.serial):
            push_model(m, a, [r for r in rts if r not in missing])
    failures = 0
    for rep in range(a.repeats):
        for mi, m in enumerate(models):
            have = [r for r in rts if r.startswith("bitino") or r == "lcebin_fp32" or
                    os.path.exists(os.path.join(a.models_dir, m, {"lce": "lce.tflite"}.get(r, f"{r}.tflite")))]
            k = (rep + mi) % len(have)
            order = have[k:] + have[:k]
            done = {os.path.basename(f) for f in glob.glob(os.path.join(a.out, "raw", f"{m}__*__r{rep}.json"))}
            if all(f"{m}__{r}__r{rep}.json" in done for r in order):
                old = [json.load(open(os.path.join(a.out, "raw", f"{m}__{r}__r{rep}.json"))) for r in order]
                if not (a.redo_throttled and any(round_off_reference(o, a.cpu, plan["reference_cap_khz"]) for o in old)):
                    continue  # resume
                # --redo-throttled: keep the throttled round under raw/throttled/ and measure it again
                os.makedirs(os.path.join(a.out, "raw", "throttled"), exist_ok=True)
                for r in order:
                    for suffix in (".json", ".bitino.json"):
                        f = os.path.join(a.out, "raw", f"{m}__{r}__r{rep}{suffix}")
                        if os.path.exists(f):
                            os.replace(f, os.path.join(a.out, "raw", "throttled", os.path.basename(f)))
            g = gate.wait()
            with DeviceLock(a.serial) as lk:
                before = bm.freq_caps()
                recs = []
                for oi, rt in enumerate(order):
                    # Validity: the launcher pinned the process (mask read back == --cpu) and every placement sample
                    # of every thread is on --cpu; otherwise the run is repeated (at most 3 attempts, all recorded).
                    attempts = []
                    for _ in range(3):
                        rec = run_one(rt, m, rep, oi, a, a.out, pull_outputs=(rep == 0))
                        la = rec.get("launcher") or {}
                        hist = la.get("cpu_hist") or {}
                        rec["placement_ok"] = (bool(hist) and set(hist) == {a.cpu} and la.get("cpus_allowed_list") == a.cpu
                                               and la.get("allowed_mismatch_samples", 0) == 0)
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
                rec["round"] = {"lock_wait_s": round(lk.waited, 1), "gate": g, "caps_before": before, "caps_after": after,
                                "throttled": throttled, "order": order}
                json.dump(rec, open(os.path.join(a.out, "raw", f"{m}__{rec['runtime']}__r{rep}.json"), "w"), indent=1)
                if not rec["ok"]:
                    failures += 1
            line = " ".join(f"{r['runtime']}={r['timing'].get('mean_ms', float('nan')):.2f}" if r["ok"] else f"{r['runtime']}=FAIL"
                            for r in recs)
            cpus = {r["runtime"]: ",".join(sorted((r.get("launcher") or {}).get("cpu_hist", {}))) for r in recs}
            off = {k: v for k, v in cpus.items() if v != a.cpu}
            print(f"[{a.tag} r{rep} {m}] {line} ms" + (" THROTTLED" if throttled else "")
                  + (f" OFF-CPU {off}" if off else "") + f" (lock wait {lk.waited:.0f} s, gate {g['waited_s']} s)", flush=True)
    plan["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    plan["caps_at_end"] = bm.freq_caps()
    json.dump(plan, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    return 1 if failures else 0


# ------------------------------------------------------------------------------------------------ report
def load_out(path: str, rt: str, rep: dict) -> "np.ndarray | None":
    import numpy as np

    if not os.path.exists(path):
        return None
    if rt == "tflite_int8":
        q = np.fromfile(path, np.int8).astype(np.float32)
        io = rep.get("int8_io", {})
        return (q - io.get("output_zero_point", 0)) * io.get("output_scale", 1.0)
    o = np.fromfile(path, np.float32)
    if rt.startswith("bitino") and o.size % 1000 == 0 and o.size > 1000:
        # the Bitino graphs of the Larq Zoo models have two outputs, [logits, softmax probabilities]: take the probs
        rows = o.reshape(-1, 1000)
        probs = [r for r in rows if r.min() >= 0 and abs(float(r.sum()) - 1.0) < 1e-3]
        return probs[0] if probs else None
    return o


def cmd_report(a) -> int:
    import numpy as np

    rows, agree, meta = [], [], {}
    for pdir in sorted(glob.glob(os.path.join(a.indir, "*", "plan.json"))):
        d = os.path.dirname(pdir)
        plan = json.load(open(pdir))
        tag = plan["tag"]
        meta[tag] = plan
        recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "raw", "*__r*.json"))) if not f.endswith(".bitino.json")]
        for r in recs:
            # Runs recorded before parse_tflite() accepted %g numbers (avg=1.68347e+06, runs > 1 s) were marked failed
            # although they completed with a valid placement: re-parse the kept tool output.
            if not r["ok"] and not r["runtime"].startswith("bitino") and r.get("exit") == 0 and r.get("placement_ok"):
                t = parse_tflite("\n".join(r.get("stdout_tail") or []))
                if t.get("mean_ms") is not None and t.get("n") == plan["iterations"]:
                    r.update(timing=t, ok=True, reparsed=True)
        by = {}
        for r in recs:
            by.setdefault((r["model"], r["runtime"]), []).append(r)
        if "reference_cap_khz" not in plan:
            first = ((plan.get("invocations") or [{}])[0] or {}).get("caps_at_start") or plan["caps_at_start"]
            plan["reference_cap_khz"] = first["policies"][policy_of(first, plan["cpu"])][0]
        for (m, rt), rs in sorted(by.items()):
            ok = [r for r in rs if r["ok"]]
            for r in ok:
                r["off_reference"] = round_off_reference(r, plan["cpu"], plan["reference_cap_khz"])
            # median over the rounds at the reference cap; off-reference rounds only when there is nothing else
            ok = [r for r in ok if not r["off_reference"]] or ok
            means = [r["timing"]["mean_ms"] for r in ok]
            hist = {}
            for r in ok:
                for c, n in ((r.get("launcher") or {}).get("cpu_hist") or {}).items():
                    hist[c] = hist.get(c, 0) + n
            row = {"device": tag, "model": m, "runtime": rt, "n_ok": len(ok), "n_runs": len(rs),
                   "median_of_means_ms": statistics.median(means) if means else None,
                   "min_mean_ms": min(means) if means else None, "max_mean_ms": max(means) if means else None,
                   "means_ms": means, "throttled_runs": sum(1 for r in ok if r["off_reference"]),
                   "off_reference_runs_excluded": sum(1 for r in rs if r["ok"] and r.get("off_reference")) - sum(1 for r in ok if r["off_reference"]),
                   "maxrss_mb": statistics.median([r["launcher"]["maxrss_kb"] / 1024 for r in ok]) if ok else None,
                   "cpu_hist": hist, "off_cpu_samples": sum(n for c, n in hist.items() if c != plan["cpu"]),
                   "pin_wait_max_s": max(((r.get("pin_wait") or {}).get("wait_s", 0) for r in ok), default=None)}
            if rt.startswith("bitino") and ok:
                row["median_of_medians_ms"] = statistics.median([r["timing"]["median_ms"] for r in ok])
            if ok and "delegate" in ok[0]["timing"]:
                row["delegate"] = ok[0]["timing"]["delegate"]
            if not ok:
                row["error"] = (rs[0].get("stdout_tail") or [""])[-3:]
            rows.append(row)
        # output agreement on the timed sample (repeat 0 outputs) vs the host reference / each other
        for m in sorted({r["model"] for r in recs}):
            bpath = os.path.join(a.models_dir, m, "build_report.json")
            brep = json.load(open(bpath)) if os.path.exists(bpath) else {}
            ref = np.fromfile(os.path.join(MAIN, "artifacts", m, "reference_probs.f32"), np.float32).reshape(128, -1)[0]
            host = np.load(os.path.join(a.models_dir, m, "host_outputs.npz")) if os.path.exists(
                os.path.join(a.models_dir, m, "host_outputs.npz")) else {}
            outs = {}
            for rt in plan["runtimes"]:
                o = load_out(os.path.join(d, "outputs", f"{m}__{rt}.out"), rt, brep)
                if o is not None and o.size == ref.size:
                    outs[rt] = o
            for rt, o in outs.items():
                hk = {"bitino_all": None, "bitino_neon": None, "lce": "lce", "tflite_fp32": "tflite_fp32",
                      "tflite_int8": "tflite_int8", "lcebin_fp32": "tflite_fp32"}.get(rt)
                e = {"device": tag, "model": m, "runtime": rt, "top1": int(o.argmax()), "ref_top1": int(ref.argmax()),
                     "max_abs_vs_reference_probs": float(np.abs(o - ref).max())}
                if hk and hk in host:
                    e["max_abs_vs_host_same_model"] = float(np.abs(o - host[hk][0]).max())
                for other in ("bitino_all", "lce", "tflite_fp32"):
                    if other in outs and other != rt:
                        e[f"max_abs_vs_{other}"] = float(np.abs(o - outs[other]).max())
                agree.append(e)
    json.dump({"rows": rows, "agreement_sample0": agree, "plans": meta}, open(os.path.join(a.indir, "summary.json"), "w"), indent=1)
    # markdown: one table per device (rows = models, columns = runtimes)
    lines = []
    for tag, plan in meta.items():
        dev = plan["device"]
        lines.append(f"### {dev.get('model')} ({tag}), cpu{plan['cpu']}, {plan['abi']}\n")
        rts = plan["runtimes"]
        lines.append("| model | " + " | ".join(rts) + " | LCE / Bitino (all) |")
        lines.append("|---|" + "---:|" * (len(rts) + 1))
        for m in plan["models"]:
            cell = {}
            for r in rows:
                if r["device"] == tag and r["model"] == m:
                    v = r["median_of_means_ms"]
                    cell[r["runtime"]] = "n/a" if v is None else f"{v:.2f}" + (" *" if r["throttled_runs"] else "")
            bv = next((r["median_of_means_ms"] for r in rows if r["device"] == tag and r["model"] == m and r["runtime"] == "bitino_all"), None)
            lv = next((r["median_of_means_ms"] for r in rows if r["device"] == tag and r["model"] == m and r["runtime"] == "lce"), None)
            ratio = f"{lv / bv:.2f}x" if bv and lv else "n/a"
            lines.append(f"| {m.replace('larq-zoo-', '')} | " + " | ".join(cell.get(r, "-") for r in rts) + f" | {ratio} |")
        lines.append("")
        # spread over repeats and peak RSS
        lines.append(f"Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on {tag}:\n")
        lines.append("| model | " + " | ".join(f"{r} RSS" for r in rts) + " | worst spread |")
        lines.append("|---|" + "---:|" * (len(rts) + 1))
        for m in plan["models"]:
            rr = {r["runtime"]: r for r in rows if r["device"] == tag and r["model"] == m}
            spread = max((r["max_mean_ms"] / r["min_mean_ms"] for r in rr.values() if r["min_mean_ms"]), default=None)
            lines.append(f"| {m.replace('larq-zoo-', '')} | "
                         + " | ".join(f"{rr[r]['maxrss_mb']:.1f}" if r in rr and rr[r]["maxrss_mb"] else "-" for r in rts)
                         + (f" | {spread:.3f} |" if spread else " | - |"))
        lines.append("")
    # model files and host agreement (from the build reports)
    models = sorted({r["model"] for r in rows})
    lines.append("### Model files and output agreement on the 128-image eval set (host interpreters)\n")
    lines.append("| model | Bitino .bitino MB | LCE .tflite MB | TFLite fp32 MB | TFLite int8 MB | top-1 acc ref / LCE / fp32 / int8 "
                 "| top-1 agreement with ref: LCE / fp32 / int8 | notes |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for m in models:
        bp = os.path.join(a.models_dir, m, "build_report.json")
        if not os.path.exists(bp):
            continue
        b = json.load(open(bp))
        f = b["files"]
        h = b["host_agreement_vs_reference_probs"]
        mb = lambda k: f"{f[k]['bytes'] / 1e6:.2f}" if k in f else "-"  # noqa: E731
        acc = " / ".join(f"{h[k]['top1_acc_out']:.3f}" if k in h else "-" for k in ("keras", "lce", "tflite_fp32", "tflite_int8"))
        ag = " / ".join(f"{h[k]['top1_agreement']:.3f}" if k in h else "-" for k in ("lce", "tflite_fp32", "tflite_int8"))
        notes = []
        if "weights_prequantized" in b:
            notes.append("kernel quantizers pre-applied (" + b["weights_prequantized"]["reason"] + ")")
        if "tflite_int8_unrunnable" in b:
            notes.append("int8 model rejected by the TFLite runtime: " + b["tflite_int8_unrunnable"]["error"][:120])
        nb, nl = f["lce"]["ops"].get("LceBconv2d", 0), b.get("lce_default_conversion", {}).get("binary_quantconv2d_layers")
        if nl and nb < nl:
            notes.append(f"LCE: {nb} of {nl} binary convolutions are LceBconv2d")
        lines.append(f"| {m.replace('larq-zoo-', '')} | {mb('bitino')} | {mb('lce')} | {mb('tflite_fp32')} | {mb('tflite_int8')} "
                     f"| {acc} | {ag} | {'; '.join(notes)} |")
    lines.append("")
    # on-device outputs of the timed sample (eval image 0)
    lines.append("### On-device outputs on the timed input (eval image 0)\n")
    lines.append("max |Δ probability| vs the reference (original Keras/larq model, TF 2.15.1), top-1 class in brackets "
                 "(the reference top-1 is given in the model column); `=host` marks a device output identical within 1e-5 to "
                 "the host interpreter's output of the same .tflite.\n")
    devs = list(meta)
    for tag in devs:
        rts = meta[tag]["runtimes"]
        lines.append(f"{tag}:\n")
        lines.append("| model (ref top-1) | " + " | ".join(rts) + " |")
        lines.append("|---|" + "---|" * len(rts))
        for m in meta[tag]["models"]:
            es = {e["runtime"]: e for e in agree if e["device"] == tag and e["model"] == m}
            if not es:
                continue
            ref1 = next(iter(es.values()))["ref_top1"]
            cells = []
            for r in rts:
                e = es.get(r)
                if not e:
                    cells.append("-")
                    continue
                c = f"{e['max_abs_vs_reference_probs']:.1e} ({e['top1']})"
                if e.get("max_abs_vs_host_same_model") is not None and e["max_abs_vs_host_same_model"] < 1e-5:
                    c += " =host"
                cells.append(c)
            lines.append(f"| {m.replace('larq-zoo-', '')} ({ref1}) | " + " | ".join(cells) + " |")
        lines.append("")
    open(a.md, "w").write("\n".join(lines))
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--serial", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--cpu", required=True, help="the one CPU every runtime is pinned to")
    r.add_argument("--abi", choices=list(BINS), default="arm64")
    r.add_argument("--models-dir", required=True)
    r.add_argument("--bins-dir", required=True)
    r.add_argument("--glibc-dir", help="armv7: dir with ld-linux-armhf.so.3, libc.so.6, libm.so.6, libgcc_s.so.1, "
                                       "libstdc++.so.6 (Debian bookworm armhf) for the Linux AArch32 LCE binary")
    r.add_argument("--models", required=True)
    r.add_argument("--runtimes", default=",".join(RUNTIMES))
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--iterations", type=int, default=50)
    r.add_argument("--throttle-max", type=float, default=120.0)
    r.add_argument("--ref-cap-khz", type=int, help="reference scaling_max_freq of the pinned CPU's policy (default: the "
                                                   "value at the start of the first invocation on this --out)")
    r.add_argument("--run-timeout", type=float, default=3600.0)
    r.add_argument("--redo-throttled", action="store_true",
                   help="re-measure rounds recorded as throttled (the old records move to raw/throttled/)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--in", dest="indir", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--md", required=True)
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
