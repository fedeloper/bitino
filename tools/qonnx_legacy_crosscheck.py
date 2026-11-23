#!/usr/bin/env python3
"""Legacy cross-check of the QONNX build path (QONNX step 6, docs/qonnx-step6.md).

Since step 6 the published files are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
bitino.qonnx_import). The IR-emitting converters tools/convert_*.py are kept, unchanged, as LEGACY CROSS-CHECKS: this
tool builds a network both ways from the same local checkpoint and compares the results:

  legacy   the recipe's own steps (registry/recipes.json -> steps: tools/convert_*.py [+ tools/derive_variant.py]) in
           the recipe's reference environment, python -I, into <work>/<id>/legacy/
  qonnx    tools/build_from_qonnx.py (the route of registry/qonnx_routes.json) into <work>/<id>/qonnx/

and for every file both produce: data + graph SHA-256 of legacy vs QONNX vs the published file (artifacts/, read only).
The legacy converters also run their own verification on the eval set (slow); the comparison is identity only.

  scripts/capped.sh 6G nice -n 10 .cache/venv-qonnx/bin/python -I tools/qonnx_legacy_crosscheck.py \\
      [--model-id ID ...] [--cache-dir DIR] [--artifacts-dir DIR] [--python ENV=PY ...] [--out JSON]

Default sample: one small network per legacy converter family of the four commonest front ends (PyTorch AB-BNN,
PyTorch IR-Net, Brevitas, Larq Zoo). Reference environments (local, read only): torch = ~/anaconda3/bin/python (torch
2.6.0, the recorded conversions'), brevitas / larq / qonnx = .cache/venv-*; override with --python ENV=PY.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

SAMPLE = ["ab-bnn-cifar10-reactnet18", "recu-resnet20-cifar10", "brevitas-bnn-pynq-tfc-1w1a", "larq-zoo-quicknet-small"]
ENVS = {"torch": "~/anaconda3/bin/python", "brevitas": "{cache}/venv-brevitas/bin/python", "larq": "{cache}/venv-larq/bin/python",
        "qonnx": "{cache}/venv-qonnx/bin/python", "base": "{cache}/venv-qonnx/bin/python",
        "openvino": "{cache}/venv-openvino/bin/python", "nncf": "{cache}/venv-nncf/bin/python"}


def legacy_build(mid: str, ctx, routes: dict, pyover: dict, out_dir: str, log: str) -> int:
    """Runs the recipe's steps with the local checkpoint / source tree (the paths tools/build_from_qonnx.py uses)."""
    rec = ctx.recipes[mid]
    reg_id = rec.get("registry_model_id", mid)
    lc = ctx.models[reg_id]["checkpoint"]["local_cache_path"]
    weights = os.path.join(ctx.cache, lc[len(".cache/"):]) if lc.startswith(".cache/") else os.path.join(ROOT, lc)
    if len(rec.get("downloads", [])) != 1:
        raise SystemExit(f"{mid}: the cross-check maps one download (the checkpoint) to the local cache; this recipe has "
                         f"{len(rec.get('downloads', []))}")
    route = routes[mid]
    wargs = route["files"]["model.bitino"]["canonical"]["args"]
    fill = {"model_id": mid, "weights": weights, "download:" + rec["downloads"][0]["name"]: weights,
            "sha256": ctx.models[reg_id]["checkpoint"]["sha256"], "eval_dir": os.path.join(ctx.cache, "data", rec["eval"]["dir"]),
            "out_dir": out_dir, "work": os.path.join(os.path.dirname(out_dir), "legacy_work"), "threads": str(ctx.threads)}
    if "--repo" in wargs:
        fill["repo"] = ctx.fill(wargs[wargs.index("--repo") + 1], mid)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fill["work"], exist_ok=True)
    for st in rec["steps"]:
        py = os.path.expanduser((pyover.get(st["env"]) or ENVS[st["env"]]).replace("{cache}", ctx.cache))
        args = []
        for a in st["args"]:
            for k in re.findall(r"\{([\w:-]+)\}", a):
                if k not in fill:
                    raise SystemExit(f"{mid}: recipe placeholder {{{k}}} has no local value")
                a = a.replace("{" + k + "}", fill[k])
            args.append(a)
        from build_from_qonnx import run
        rc = run([py, "-I", os.path.join(ROOT, st["tool"])] + args, log, ctx.env())
        if rc:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", action="append")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--work", help="default <cache>/qonnx_step6/crosscheck")
    ap.add_argument("--python", action="append", default=[], help="ENV=PY: interpreter of a recipe environment")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_step6", "legacy_crosscheck.json"))
    ap.add_argument("--keep-files", action="store_true")
    args = ap.parse_args()
    import build_from_qonnx as B
    from bitino.fetch import data_sha256, graph_sha256
    pyover = dict(kv.split("=", 1) for kv in args.python)
    routes = json.load(open(B.ROUTES))["routes"]
    bargs = argparse.Namespace(cache_dir=args.cache_dir, artifacts_dir=args.artifacts_dir, keep_files=True, threads=args.threads,
                               work=args.work or os.path.join(os.path.abspath(args.cache_dir), "qonnx_step6", "crosscheck"))
    ctx = B.Ctx(bargs)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {"schema": "bitino.qonnx_legacy_crosscheck.v1", "networks": {}}
    ok_all = True
    for mid in args.model_id or SAMPLE:
        t0 = time.time()
        base = os.path.join(ctx.work, mid)
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(base, exist_ok=True)
        log = os.path.join(base, "crosscheck.log")
        row = {"model_id": mid, "legacy_converter": ctx.recipes[mid]["converter"], "front_end": routes[mid]["front_end"],
               "legacy_steps": [s["tool"] for s in ctx.recipes[mid]["steps"]]}
        leg_dir = os.path.join(base, "legacy")
        rc = legacy_build(mid, ctx, routes, pyover, leg_dir, log)
        row["legacy_returncode"] = rc
        # the QONNX route, keeping its built files: build_from_qonnx.build_model writes under <work>/<id>/<stem>/
        qctx = B.Ctx(argparse.Namespace(**{**vars(bargs), "work": os.path.join(base, "qonnx")}))
        qfiles = [f for f in routes[mid]["files"] if f.endswith(".bitino") and not f.startswith("model_int8")]
        qrows = B.build_model(qctx, mid, routes[mid], qfiles, base)
        files = {}
        for f in qfiles:
            qr = qrows.get(f"{mid}/{f}", {})
            lp = os.path.join(leg_dir, f)
            pub = os.path.join(ctx.artifacts, mid, f)
            leg = {"data_sha256": data_sha256(lp), "graph_sha256": graph_sha256(lp)} if os.path.exists(lp) else None
            q = {k: qr["built"][k] for k in ("data_sha256", "graph_sha256")} if qr.get("built") else None
            p = {"data_sha256": data_sha256(pub), "graph_sha256": graph_sha256(pub)}
            files[f] = {"legacy": leg, "qonnx": q, "published": p, "legacy_equals_qonnx": leg is not None and leg == q,
                        "qonnx_equals_published": q == p, "legacy_equals_published": leg == p}
        row["files"] = files
        row["agree"] = bool(files) and all(v["legacy_equals_qonnx"] and v["qonnx_equals_published"] for v in files.values())
        row["seconds"] = round(time.time() - t0, 1)
        ok_all &= row["agree"]
        print(f"[{mid}] legacy rc {rc}; " + "; ".join(f"{f}: legacy=qonnx {v['legacy_equals_qonnx']} qonnx=published "
                                                       f"{v['qonnx_equals_published']}" for f, v in files.items()), flush=True)
        res["networks"][mid] = row
        res["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        if not args.keep_files:
            shutil.rmtree(base, ignore_errors=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
