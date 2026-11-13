#!/usr/bin/env python3
"""Tables of the int8_full v2 files (docs/int8.md "int8_full v2"): per (model, current file) pair the accuracy of the float
model, the current int8_full file and the v2 file on the model's large eval set (paired current vs v2:
reports/accuracy/<m>_<current>_vs_int8_full_v2_<tag>.json; float and current from the shipped
reports/accuracy/<m>_exact_vs_<current>_<tag>.json), and the 1-thread latency of current / v2 / float `all` on the phones
(reports/int8_full_v2/speed/<phone>/<m>.json, tools/int8_full_v2_device.py bench). Verdict per pair: "propose" iff v2 is
on par (top-1 >= current - 0.3 pt) and faster than the current file on every phone that timed it; never a v2 file that
keeps a collapse of the current file (top-1 below half the float model's: ReActNet-34's int8_full, whose log2 file is the
working one), nor one whose accuracy report was measured on another file (sha256 mismatch).

    python3 tools/int8_full_v2_table.py [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import int8_full_suite as S  # noqa: E402

PHONES = [("oneplus-x3", "OnePlus X3"), ("moto-a75", "Moto A75"), ("redmi-a53", "Redmi A53")]
PAR = 0.30  # percentage points


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows():
    out = []
    models = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(ROOT, "artifacts", "*", "model_int8_full_v2.bitino")))
    for m in models:
        try:
            _, _, _, tag = S.big_eval_of(m)
        except SystemExit:
            tag = None
        v2_sha = hashlib.sha256(open(os.path.join(ROOT, "artifacts", m, "model_int8_full_v2.bitino"), "rb").read()).hexdigest()
        speed = {ph: load(os.path.join(ROOT, "reports", "int8_full_v2", "speed", ph, f"{m}.json")) for ph, _ in PHONES}
        for cur in ("int8_full", "int8_full_log2"):
            if not os.path.exists(os.path.join(ROOT, "artifacts", m, f"model_{cur}.bitino")):
                continue
            r = {"model": m, "current": cur, "eval": tag}
            pair = load(os.path.join(ROOT, "reports", "accuracy", f"{m}_{cur}_vs_int8_full_v2_{tag}.json")) if tag else None
            ex = load(os.path.join(ROOT, "reports", "accuracy", f"{m}_exact_vs_{cur}_{tag}.json")) if tag else None
            if ex is None and cur == "int8_full_log2" and tag:   # some log2 reports carry the stem variant in the name
                cand = sorted(glob.glob(os.path.join(ROOT, "reports", "accuracy", f"{m}_exact_vs_int8_full_log2*_{tag}.json")))
                ex = load(cand[0]) if cand else None
            if pair:
                r.update(n=pair["n"], acc_current=pair["configs"]["a"]["top1"] * 100, acc_v2=pair["configs"]["b"]["top1"] * 100,
                         diff=pair["paired"]["top1_diff_b_minus_a"] * 100, p=pair["paired"]["mcnemar_exact_p"],
                         agree=pair["paired"]["top1_prediction_agreement"] * 100, ci=[v * 100 for v in pair["paired"]["diff_ci95"]],
                         v2_sha=pair.get("files_sha256", {}).get("b"))
            if ex:
                r["acc_float"] = ex["configs"]["a"]["top1"] * 100
                if "acc_current" in r and abs(ex["configs"]["b"]["top1"] * 100 - r["acc_current"]) > 1e-9:
                    r["note"] = f"shipped report {ex['configs']['b']['top1'] * 100:.2f}"
            for ph, _ in PHONES:
                sp = speed[ph]
                if sp and sp["median_ms"].get(cur) and sp["median_ms"].get("int8_full_v2"):
                    r[ph] = {"current": sp["median_ms"][cur], "v2": sp["median_ms"]["int8_full_v2"], "float": sp["median_ms"].get("float_all")}
            timed = [r[ph] for ph, _ in PHONES if ph in r]
            on_par = "diff" in r and r["diff"] >= -PAR - 1e-9
            faster = bool(timed) and all(t["v2"] < t["current"] for t in timed)
            r["on_par"], r["faster"] = on_par, faster
            r["verdict"] = "propose" if (on_par and faster) else ("pending accuracy" if "diff" not in r else "not proposed")
            if "diff" in r and r.get("acc_float") and r["acc_v2"] < r["acc_float"] / 2:
                r["verdict"] = "not proposed (collapsed)"
            if r.get("v2_sha") and r["v2_sha"] != v2_sha:
                r["verdict"] = "stale accuracy report"
            r["v2_faster_than_float"] = {ph: (r[ph]["v2"] < r[ph]["float"]) for ph, _ in PHONES if ph in r and r[ph]["float"]}
            r["current_faster_than_float"] = {ph: (r[ph]["current"] < r[ph]["float"]) for ph, _ in PHONES if ph in r and r[ph]["float"]}
            out.append(r)
    return out


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def md(rs):
    L = ["| model | current file | eval (n) | top-1 float | top-1 current | top-1 v2 | v2 − current pt [95% CI] (p) | "
         + " | ".join(f"{n} current / v2 / float (ms)" for _, n in PHONES) + " | verdict |",
         "|---|---|---|---|---|---|---|" + "---|" * len(PHONES) + "---|"]
    for r in rs:
        cells = []
        for ph, _ in PHONES:
            t = r.get(ph)
            if not t:
                cells.append("–")
                continue
            mark = lambda a, b: f"**{a:.3f}**" if b is not None and a < b else f"{a:.3f}"  # noqa: E731
            cells.append(f"{t['current']:.3f} / {mark(t['v2'], t['float'])} / {fmt(t['float'], 3)}")
        acc = f"{r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] ({r['p']:.2g})" if "diff" in r else "–"
        L.append(f"| {r['model']} | {r['current'].replace('int8_full_log2', 'log2')} | {r.get('eval') or '–'} ({r.get('n', '–')}) | "
                 f"{fmt(r.get('acc_float'))} | {fmt(r.get('acc_current'))} | {fmt(r.get('acc_v2'))} | {acc} | " + " | ".join(cells)
                 + f" | {r['verdict']} |")
    return "\n".join(L)


def summary(rs):
    s = []
    for ph, name in PHONES:
        rr = [r for r in rs if ph in r and r[ph]["float"]]
        if not rr:
            continue
        cur_slow = sum(r[ph]["current"] >= r[ph]["float"] for r in rr)
        v2_slow = sum(r[ph]["v2"] >= r[ph]["float"] for r in rr)
        s.append(f"{name}: {len(rr)} pairs timed; slower than float: current {cur_slow}, v2 {v2_slow}")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    rs = rows()
    text = "\n".join(summary(rs)) + "\n\n" + md(rs) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    else:
        print(text)
    if a.json:
        json.dump(rs, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
