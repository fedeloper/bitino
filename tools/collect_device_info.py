#!/usr/bin/env python3
"""Collects the Android device configuration over adb (no root needed).

Records model, SoC, Android/API level, ABI, CPU topology (part IDs, clusters,
frequency policies and governors), ISA features, memory, storage, thermal
status and profiling availability. Unavailable values are recorded as null,
never guessed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

CPU_PARTS = {"0xd46": "Cortex-A510", "0xd47": "Cortex-A710", "0xd48": "Cortex-X2", "0xd05": "Cortex-A55",
             "0xd41": "Cortex-A78", "0xd44": "Cortex-X1", "0xd4d": "Cortex-A715", "0xd4e": "Cortex-X3",
             "0xd80": "Cortex-A520", "0xd81": "Cortex-A720", "0xd82": "Cortex-X4"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb", default=os.path.expanduser("~/Android/Sdk/platform-tools/adb"))
    ap.add_argument("--json", action="store_true", help="print JSON only")
    ap.add_argument("--out")
    a = ap.parse_args()

    def sh(cmd: str):
        try:
            r = subprocess.run([a.adb, "shell", cmd], capture_output=True, text=True, timeout=30)
            return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
        except Exception:
            return None

    devs = subprocess.run([a.adb, "devices", "-l"], capture_output=True, text=True).stdout
    if not any(l.split()[1:2] == ["device"] for l in devs.splitlines()[1:] if l.strip()):
        print(json.dumps({"connected": False, "adb_devices": devs}))
        return 1
    prop = lambda k: sh(f"getprop {k}")
    info = {
        "collected_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "connected": True,
        "model": prop("ro.product.model"), "device": prop("ro.product.device"),
        "soc_manufacturer": prop("ro.soc.manufacturer"), "soc_model": prop("ro.soc.model"), "board_platform": prop("ro.board.platform"),
        "android_release": prop("ro.build.version.release"), "api_level": prop("ro.build.version.sdk"),
        "abi": prop("ro.product.cpu.abi"), "abi_list": prop("ro.product.cpu.abilist"),
        "build_fingerprint": prop("ro.build.fingerprint"), "kernel": sh("uname -r"),
    }
    cpuinfo = sh("cat /proc/cpuinfo") or ""
    cpus, cur = [], None
    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        k, v = (s.strip() for s in line.split(":", 1))
        if k == "processor":
            cur = {"cpu": int(v)}
            cpus.append(cur)
        elif cur is not None and k in ("CPU part", "CPU variant", "Features"):
            cur[k.replace(" ", "_").lower()] = v
    for c in cpus:
        c["core"] = CPU_PARTS.get(c.get("cpu_part", ""), "unknown")
        base = f"/sys/devices/system/cpu/cpu{c['cpu']}"
        for f in ("cpuinfo_max_freq", "cpuinfo_min_freq", "scaling_governor", "scaling_cur_freq"):
            c[f] = sh(f"cat {base}/cpufreq/{f}")
        c["capacity"] = sh(f"cat {base}/cpu_capacity")
    info["cpus"] = cpus
    info["features"] = sorted(set((cpus[0].get("features") or "").split())) if cpus else []
    info["isa_summary"] = {f: f in info["features"] for f in ("asimd", "asimddp", "i8mm", "bf16", "asimdhp", "sve", "sve2")}
    info["freq_policies"] = {}
    for p in (sh("ls /sys/devices/system/cpu/cpufreq") or "").split():
        if p.startswith("policy"):
            info["freq_policies"][p] = sh(f"cat /sys/devices/system/cpu/cpufreq/{p}/related_cpus")
    mem = sh("cat /proc/meminfo") or ""
    info["memory_kb"] = {l.split(":")[0]: int(l.split()[1]) for l in mem.splitlines()[:3] if len(l.split()) >= 2}
    info["storage_data_local_tmp"] = sh("df -h /data/local/tmp | tail -1")
    info["thermal_status"] = sh("dumpsys thermalservice | grep 'Thermal Status' | head -1")
    info["perf_event_paranoid"] = sh("cat /proc/sys/kernel/perf_event_paranoid")
    info["simpleperf"] = sh("which simpleperf")
    text = json.dumps(info, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
