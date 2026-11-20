#!/usr/bin/env python3
"""Measured (partial) energy per inference on the OnePlus Open with the USB cable plugged in.

  python3 tools/energy_oneplus.py run --out reports/energy_oneplus/<date> [--configs x3,a715,...] [--repeats 2]

Why it works. Plugged into this PC, the phone's USB input is limited; the OPLUS battery service (`dumpsys battery`,
"Battery current", mA, positive = discharging, updated every ~5 s; readable without root) shows the battery current.
When the phone needs more power than the USB input supplies, the input is at its limit and every extra milliwatt
comes from the battery. A constant BALLAST (one shell busy loop per core, pinned with taskset) on the cores the
measured run does not use keeps the phone in that state. Then, alternating windows:
  A = ballast only,  B = ballast + the model (bitino_bench pinned to its cores, an iteration count that lasts --run-s)
  dP = (I_B - mean(I_A before, I_A after)) * V_battery        (the model's extra power: CPU, DRAM, SoC rails)
  energy per inference = dP * T_B / N_B
A window is VALID only if the battery discharges in the ballast-only windows (I_A > --min-discharge-ma): the
input is then saturated in both A and B. Samples in the first --settle-s of every window are dropped (the reading
lags). The sampler (dumpsys, once per second) is pinned to a ballast core, so it costs the same in A and B.
Limits: partial (one phone; configurations that leave enough cores free for the ballast), the gauge's ~5 s update and
mA resolution, and the ballast's heat (thermal status recorded per window; a window at status > 0 is flagged).
The runs hold the phone's device lock (/tmp/bitino-device-<serial>.lock), so other jobs on it pause meanwhile.
"""
import argparse
import fcntl
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADB = os.path.expanduser("~/Android/Sdk/platform-tools/adb")
SERIAL = "e3698ea7"
DEV = "/data/local/tmp/bitino_energy"
ALL = [0, 1, 2, 3, 4, 5, 6, 7]
CORE = {"x3": "7", "a715": "3", "a510": "0", "t2": "7,3", "t4": "7,3,4,5"}
MODELS = ["larq-zoo-quicknet-small", "larq-zoo-quicknet", "ab-bnn-imagenet-reactnet18"]


def sh(cmd, check=False, timeout=None):
    return subprocess.run([ADB, "-s", SERIAL, "shell", cmd], capture_output=True, text=True, check=check, timeout=timeout)


def adb(*a):
    return subprocess.run([ADB, "-s", SERIAL, *a], capture_output=True, text=True, check=True)


def configs(sel):
    out = []
    for m in MODELS:
        for c in sel:
            out.append({"name": f"{m}__{c}", "model": m, "file": "model.bitino", "stage": "all", "aff": CORE[c]})
        if "x3" in sel:
            out.append({"name": f"{m}__x3_int8_full", "model": m, "file": "model_int8_full.bitino", "stage": "int8_full",
                        "aff": "7"})
    return out


def script(c, n, a):
    used = {int(x) for x in c["aff"].split(",")}
    ballast = [x for x in ALL if x not in used]
    samp_core = ballast[0]
    thr = len(used)
    L = ["#!/system/bin/sh", f"D={DEV}", f"O=$D/log/{c['name']}.txt", "rm -f $O",
         "samp() { while [ ! -f $D/log/stop ]; do echo \"S $(cut -d' ' -f1 /proc/uptime) $(dumpsys battery | grep -E "
         "'Battery current|  voltage:|status:' | tr -s ' ' | tr '\\n' '|') T=$(dumpsys thermalservice 2>/dev/null | grep -m1 "
         "'Thermal Status' | grep -o '[0-9]*$') F=$(cat /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq),$(cat /sys/devices/system/cpu/cpufreq/policy3/scaling_max_freq),$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)\" >> $O; sleep 1; done; }",
         "ev() { echo \"E $(cut -d' ' -f1 /proc/uptime) $*\" >> $O; }", "rm -f $D/log/stop",
         "P=''"]
    for b in ballast:
        L.append(f"taskset {1 << b:x} sh -c 'while :; do :; done' & P=\"$P $!\"")
    L += ["samp & SP=$!", f"taskset -p {1 << samp_core:x} $SP >/dev/null", f"sleep {a.settle_s}"]
    for r in range(a.repeats):
        L += [f"ev A_start {r}", f"sleep {a.idle_s}", f"ev A_end {r}", f"ev B_start {r}",
              f"$D/bitino_bench --model $D/m/{c['model']}__{c['file']} --stage {c['stage']} --threads {thr} "
              f"--affinity {c['aff']} --warmup 10 --iterations {n} --json $D/log/{c['name']}_r{r}.json >/dev/null 2>&1",
              f"ev B_end {r}"]
    L += [f"ev A_start {a.repeats}", f"sleep {a.idle_s}", f"ev A_end {a.repeats}",
          "touch $D/log/stop", "kill $P", "wait $SP 2>/dev/null", "ev done"]
    return "\n".join(L) + "\n"


def parse(path):
    S, E = [], []
    for line in open(path):
        p = line.split(None, 2)
        if not p:
            continue
        if p[0] == "S":
            i = re.search(r"Battery current : (-?\d+)", line)
            v = re.search(r"voltage: (\d+)", line)
            t = re.search(r"T=(\d+)", line)
            fq = re.search(r"F=(\d+),(\d+),(\d+)", line)
            if i and v:
                S.append({"t": float(p[1]), "i": int(i.group(1)), "v": int(v.group(1)), "th": int(t.group(1)) if t else None,
                          "caps": tuple(int(x) for x in fq.groups()) if fq else None})
        elif p[0] == "E":
            E.append((float(p[1]), p[2].split()))
    return S, E


def win(S, t0, t1, settle):
    s = [x for x in S if t0 + settle <= x["t"] <= t1]
    if not s:
        return None
    return {"i_ma": statistics.mean(x["i"] for x in s), "v_mv": statistics.mean(x["v"] for x in s), "n": len(s),
            "thermal_max": max((x["th"] or 0) for x in s), "t_s": t1 - t0,
            "caps_min_khz": [min(x["caps"][k] for x in s if x["caps"]) for k in range(3)] if any(x["caps"] for x in s) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="x3,a715,a510,t2,t4")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--idle-s", type=int, default=45)
    ap.add_argument("--run-s", type=int, default=45)
    ap.add_argument("--settle-s", type=int, default=8)
    ap.add_argument("--min-discharge-ma", type=float, default=50.0)
    a = ap.parse_args()
    MODELS[:] = a.models.split(",")
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    cfgs = configs(a.configs.split(","))
    if a.cmd == "run":
        lk = open(f"/tmp/bitino-device-{SERIAL}.lock", "a")
        for c in cfgs:
            if os.path.exists(os.path.join(a.out, "raw", c["name"] + ".txt")):
                continue
            t0 = time.time()
            fcntl.flock(lk, fcntl.LOCK_EX)
            waited = time.time() - t0
            try:
                sh(f"mkdir -p {DEV}/m {DEV}/log", check=True)
                if sh(f"ls {DEV}/bitino_bench").returncode:
                    adb("push", os.path.join(ROOT, "build/android-arm64/bitino_bench"), f"{DEV}/bitino_bench")
                    sh(f"chmod 755 {DEV}/bitino_bench")
                src = os.path.join(ROOT, "artifacts", c["model"], c["file"])
                dst = f"{DEV}/m/{c['model']}__{c['file']}"
                adb("push", src, dst)
                thr = len(c["aff"].split(","))
                p = sh(f"{DEV}/bitino_bench --model {dst} --stage {c['stage']} --threads {thr} --affinity {c['aff']} "
                       f"--warmup 10 --iterations 30 --json {DEV}/log/cal.json >/dev/null 2>&1; cat {DEV}/log/cal.json")
                ms = json.loads(p.stdout[p.stdout.index("{"):])["latency"]["median_ms"]
                n = max(100, int(a.run_s * 1000 / ms))
                open(os.path.join(a.out, "raw", c["name"] + ".sh"), "w").write(script(c, n, a))
                adb("push", os.path.join(a.out, "raw", c["name"] + ".sh"), f"{DEV}/run.sh")
                est = a.settle_s + a.repeats * (a.idle_s + a.run_s) + a.idle_s
                print(f"{c['name']}: {ms:.2f} ms cal, {n} iterations, ~{est} s (lock wait {waited:.0f} s)", flush=True)
                sh(f"sh {DEV}/run.sh", timeout=est * 3 + 120)
                adb("pull", f"{DEV}/log/{c['name']}.txt", os.path.join(a.out, "raw", c["name"] + ".txt"))
                for r in range(a.repeats):
                    adb("pull", f"{DEV}/log/{c['name']}_r{r}.json", os.path.join(a.out, "raw", f"{c['name']}_r{r}.json"))
                json.dump({"cal_ms": ms, "iterations": n, "config": c, "lock_wait_s": waited},
                          open(os.path.join(a.out, "raw", c["name"] + ".meta.json"), "w"), indent=1)
            finally:
                sh("pkill -f 'while :; do :; done'; touch " + DEV + "/log/stop")
                fcntl.flock(lk, fcntl.LOCK_UN)
    rows = []
    for c in cfgs:
        f = os.path.join(a.out, "raw", c["name"] + ".txt")
        if not os.path.exists(f):
            continue
        meta = json.load(open(os.path.join(a.out, "raw", c["name"] + ".meta.json")))
        S, E = parse(f)
        st = {}
        A, B = {}, {}
        for t, ev in E:
            k = ev[0]
            if len(ev) < 2:
                continue
            if k.endswith("_start"):
                st[(k[0], ev[1])] = t
            elif k.endswith("_end"):
                w = win(S, st[(k[0], ev[1])], t, a.settle_s)
                (A if k[0] == "A" else B)[int(ev[1])] = w
        reps = []
        for r in sorted(B):
            j = json.load(open(os.path.join(a.out, "raw", f"{c['name']}_r{r}.json")))
            nb = j["latency"]["n"] + 10
            adj = [A.get(r), A.get(r + 1)]
            adj = [x for x in adj if x]
            ia = statistics.mean(x["i_ma"] for x in adj)
            dp_mw = (B[r]["i_ma"] - ia) * B[r]["v_mv"] / 1000.0
            e_mj = dp_mw * B[r]["t_s"] / nb
            valid = min(x["i_ma"] for x in adj) > a.min_discharge_ma
            reps.append({"repeat": r, "i_a_ma": ia, "i_b_ma": B[r]["i_ma"], "v_mv": B[r]["v_mv"], "dp_mw": dp_mw,
                         "inferences": nb, "t_b_s": B[r]["t_s"], "mj_per_inference": e_mj, "valid": valid,
                         "median_ms": j["latency"]["median_ms"], "affinity_errors": (j.get("config") or {}).get("affinity_errors"),
                         "thermal_max": max(B[r]["thermal_max"], *[x["thermal_max"] for x in adj]),
                         "caps_min_khz_b": B[r].get("caps_min_khz")})
        ok = [x for x in reps if x["valid"]]
        rows.append({"name": c["name"], "model": c["model"], "stage": c["stage"], "affinity": c["aff"], "repeats": reps,
                     "mj_per_inference": statistics.median(x["mj_per_inference"] for x in ok) if ok else None,
                     "dp_mw": statistics.median(x["dp_mw"] for x in ok) if ok else None,
                     "median_ms": statistics.median(x["median_ms"] for x in reps) if reps else None,
                     "valid_repeats": len(ok)})
    json.dump({"method": __doc__.split("Why it works.")[1].strip(), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    L = ["# Measured energy per inference, OnePlus Open (plugged in, ballast method; partial)", "",
         "| config | cores | median ms | extra power (mW) | mJ / inference | valid repeats | ballast-only discharge (mA) | thermal max |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ia = ", ".join(f"{x['i_a_ma']:.0f}" for x in r["repeats"])
        th = max((x["thermal_max"] for x in r["repeats"]), default=None)
        f = lambda v, p: "–" if v is None else f"{v:.{p}f}"  # noqa: E731
        L.append(f"| {r['name']} | {r['affinity']} | {f(r['median_ms'], 2)} | {f(r['dp_mw'], 0)} | {f(r['mj_per_inference'], 2)} | "
                 f"{r['valid_repeats']}/{len(r['repeats'])} | {ia} | {th} |")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
