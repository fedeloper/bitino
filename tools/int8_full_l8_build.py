#!/usr/bin/env python3
"""Builds model_int8_full_log2.bitino (l8 residual stream, fused I8BConv2DRes blocks) for models of the suite.

    python3 tools/int8_full_l8_build.py --models ab-bnn-imagenet-reactnet34 ab-bnn-cifar10-reactnet18 \\
        --artifacts <dir with the float + int8_full models> --stats-dir .cache/int8_full_stats --out-dir artifacts

Same float model, calibration set, sample count, range rule and calibration statistics as the shipped
model_int8_full.bitino (reports/int8_full/<m>.json); first checks that `--residual-code linear` reproduces the
shipped int8_full model byte for byte (the statistics are the right ones), then writes the l8 model with
`--residual-code l8` (tools/quantize_int8_full.py, python/bitino/quantize_l8.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import int8_full_suite as S  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "python"))
from bitino import format as F  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def graph_equal(a, b):
    """Same nodes / constants / tensors (the manifests differ in dates and paths)."""
    ga, gb = F.read(a, verify=False), F.read(b, verify=False)
    if [(n.op, n.inputs, n.outputs, n.attrs, n.name) for n in ga.nodes] != [(n.op, n.inputs, n.outputs, n.attrs, n.name) for n in gb.nodes]:
        return False
    if set(ga.constants) != set(gb.constants):
        return False
    import numpy as np
    for k, c in ga.constants.items():
        d = gb.constants[k]
        if c.dtype != d.dtype or tuple(c.shape) != tuple(d.shape) or not np.array_equal(np.asarray(c.data), np.asarray(d.data)):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--stats-dir", default=os.path.join(ROOT, ".cache", "int8_full_stats"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--no-check", action="store_true", help="skip the linear reproduction check")
    ap.add_argument("--extra", action="extend", nargs="*", default=[], help="extra arguments for quantize_int8_full.py "
                    "(repeatable, e.g. --extra=--l8-range=lohi --extra=--stem=split)")
    a = ap.parse_args()
    rc = 0
    for m in a.models:
        rep = json.load(open(os.path.join(ROOT, "reports", "int8_full", f"{m}.json")))
        cal = rep["calibration"]
        cdir, key = S.calib_of(m)
        assert key == cal["preprocessing"], (m, key, cal["preprocessing"])
        fm = os.path.join(a.artifacts, m, "model.bitino")
        if sha(fm) != rep["source_model"]["sha256"]:
            print(f"{m}: float model changed since int8_full (sha256 differs); skipped")
            rc = 1
            continue
        od = os.path.join(a.out_dir, m)
        os.makedirs(od, exist_ok=True)
        base = [sys.executable, os.path.join(HERE, "quantize_int8_full.py"), "--model", fm, "--calib-dir", os.path.join(ROOT, cdir),
                "--preprocess", key, "--n", str(cal["n"]), "--stats", os.path.join(a.stats_dir, f"{m}.stats.json")]
        if cal["range"].startswith("percentile"):
            base += ["--percentile", cal["range"].split()[1]]
        if not a.no_check:
            lin = os.path.join(od, "model_int8_full.linear_check.bitino")
            r = subprocess.run(base + ["--out", lin, "--report", lin + ".report.json"], capture_output=True, text=True)
            if r.returncode:
                print(f"{m}: linear rebuild FAILED\n{r.stderr[-2000:]}")
                rc = 1
                continue
            same = graph_equal(lin, os.path.join(a.artifacts, m, "model_int8_full.bitino"))
            print(f"{m}: --residual-code linear reproduces the shipped int8_full graph: {same}", flush=True)
            os.remove(lin)
            os.remove(lin + ".report.json")
            if not same:
                rc = 1
                continue
        out = os.path.join(od, "model_int8_full_log2.bitino")
        r = subprocess.run(base + ["--residual-code", "l8", "--out", out] + a.extra, capture_output=True, text=True)
        if r.returncode:
            print(f"{m}: l8 FAILED\n{r.stderr[-3000:]}")
            rc = 1
            continue
        j = json.load(open(out + ".report.json"))
        rcode = j.get("residual_code", {})
        print(f"{m}: {j['nodes']} nodes, fused blocks {rcode.get('fused_blocks')}, gemm_l8 {len(rcode.get('gemm_l8', []))}, "
              f"luts {len(rcode.get('lut_to_linear', []))}, int8 ops {j['int8_ops']}, sanity {j['sanity']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
