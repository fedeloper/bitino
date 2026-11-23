#!/usr/bin/env python3
"""Identity sample of the earlier QONNX steps with the current importer (no regression check of a later step): the
canonical files of phase 2 / steps 1, 1b, 2 that are still on disk (<cache>/qonnx_step*/<id>/[<variant>/]model.onnx,
read only) are re-imported with their recorded import arguments (reports/qonnx_step*/summary.json -> export.import_args)
and the data / graph SHA-256 compared with the recorded re-imported (= published) hashes. A file whose SHA-256 is not
the recorded export.onnx_sha256 is skipped (it was regenerated after the report).

  .cache/venv-qonnx/bin/python -I tools/qonnx_earlier_identity.py --cache-dir .cache [--keys step1:ab-bnn-cifar10-reactnet18:ge ...]
      [--per-step N] [--out .cache/qonnx_step5/earlier_identity.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

STEPS = {"step1": "qonnx_step1", "step1b": "qonnx_step1b", "step2": "qonnx_step2"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates(cache: str, step: str) -> list:
    s = json.load(open(os.path.join(ROOT, "reports", STEPS[step], "summary.json")))
    out = []
    for key, row in s["networks"].items():
        mid, _, var = key.partition(":")
        exp = row.get("export", {})
        rec = row.get("reimported", {})
        if not exp.get("onnx_sha256") or not rec.get("data_sha256"):
            continue
        d = os.path.join(cache, STEPS[step], mid, var) if var else os.path.join(cache, STEPS[step], mid)
        p = os.path.join(d, "model.onnx")
        out.append({"key": f"{step}:{key}", "onnx": p, "onnx_sha256": exp["onnx_sha256"], "import_args": exp.get("import_args", []),
                    "data_sha256": rec["data_sha256"], "graph_sha256": rec["graph_sha256"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--per-step", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    from bitino.fetch import data_sha256, graph_sha256
    res = {}
    for step in STEPS:
        cands = [c for c in candidates(a.cache_dir, step) if os.path.exists(c["onnx"])]
        if a.keys:
            cands = [c for c in cands if c["key"] in a.keys]
        else:
            cands = cands[:: max(1, len(cands) // a.per_step)][:a.per_step]
        for c in cands:
            if sha256_file(c["onnx"]) != c["onnx_sha256"]:
                res[c["key"]] = {"skipped": "model.onnx is not the recorded export"}
                continue
            with tempfile.TemporaryDirectory() as td:
                cmd = [sys.executable, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), c["onnx"], "--model-id",
                       c["key"].split(":")[1], "--out-dir", td] + list(c["import_args"])
                r = subprocess.run(cmd, capture_output=True, text=True)
                bf = [f for f in os.listdir(td) if f.endswith(".bitino")]
                if r.returncode or not bf:
                    res[c["key"]] = {"error": (r.stderr or r.stdout)[-600:]}
                    continue
                p = os.path.join(td, bf[0])
                d, g = data_sha256(p), graph_sha256(p)
            res[c["key"]] = {"data": d == c["data_sha256"], "graph": g == c["graph_sha256"], "import_args": c["import_args"],
                             "data_sha256": d, "graph_sha256": g}
            print(c["key"], "data", res[c["key"]]["data"], "graph", res[c["key"]]["graph"], flush=True)
    ok = sum(1 for v in res.values() if v.get("data") and v.get("graph"))
    print(f"identical {ok} / {len(res)}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"schema": "bitino.qonnx_earlier_identity.v1", "identical": ok, "checked": len(res), "files": res},
                  open(a.out, "w"), indent=1)
    return 0 if ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
