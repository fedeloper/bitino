#!/usr/bin/env python3
"""Files slower than float `all` with the current int8_full files and with the proposed v2 files in their place (docs/int8.md
"int8_full v2"). Pairs timed in this work: reports/int8_full_v2/speed (current, v2 and float of one binary, interleaved);
files of models without a v2 file: the round-3 ledger (the development log (not published), its
"after / float"). A current file is replaced when tools/int8_full_v2_table.py proposes the pair.

    python3 tools/int8_full_v2_summary.py [--table-json reports/int8_full_v2/table.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LEDGER = os.path.join(ROOT, "benchmarks", "ledger", "int8_full_raw", "speed_round3_2026-11-13.md")
SECTIONS = {"oneplus-x3": "OnePlus Open (Cortex-X3):", "moto-a75": "Moto e13 (Cortex-A75):", "redmi-a53": "Redmi 9A, armeabi-v7a"}


def ledger():
    out, cur = {}, None
    for line in open(LEDGER):
        for ph, head in SECTIONS.items():
            if line.startswith(head):
                cur = ph
                out[cur] = {}
        m = re.match(r"\| ([\w.:\-]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if cur and m:
            out[cur][m.group(1)] = float(m.group(6))  # after / float
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-json", default=os.path.join(ROOT, "reports", "int8_full_v2", "table.json"))
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = json.load(open(a.table_json))
    led = ledger()
    res = {}
    for ph, files in led.items():
        before = after = 0
        slow_after = []
        for f, ratio in files.items():
            m, _, kind = f.partition(":")
            cur = "int8_full_log2" if kind == "log2" else "int8_full"
            r = next((x for x in rows if x["model"] == m and x["current"] == cur), None)
            t = r.get(ph) if r else None
            if t and t["float"]:   # measured here (same binary for current, v2 and float)
                slow_b = t["current"] >= t["float"]
                use_v2 = r["verdict"] == "propose"
                slow_a = (t["v2"] if use_v2 else t["current"]) >= t["float"]
            else:
                slow_b = slow_a = ratio >= 1.0
            before += slow_b
            after += slow_a
            if slow_a:
                slow_after.append(f)
        res[ph] = {"files": len(files), "slower_than_float_current": before, "slower_than_float_with_v2": after,
                   "still_slower": slow_after}
        print(f"{ph}: {len(files)} files; slower than float: current {before}, with the proposed v2 files {after}: {', '.join(slow_after)}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
