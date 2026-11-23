#!/usr/bin/env python3
"""Step-5 acceptance of the QONNX route (docs/qonnx-frontend.md § Plan, step 5; report docs/qonnx-step5.md): every
published int8_full file of the registry (registry/recipes.json -> expected: model_int8_full.bitino,
model_int8_full_log2.bitino) written to a canonical QONNX file of the Bitino profile by the int8 group writer
(tools/export_int8_qonnx.py, python/bitino/qonnx_int8.py), checked with check-qonnx, imported (the group importer of
qonnx_groups dispatches to qonnx_int8 for bitino.variant int8_full*) and compared with the published file.

Runs in .cache/venv-qonnx with python -I:

  scripts/capped.sh 8G nice -n 10 env OMP_NUM_THREADS=1 .cache/venv-qonnx/bin/python -I tools/qonnx_step5_acceptance.py \
      [--only ID[/FILE] ...] [--no-verify] [--samples N] [--numpy-samples K] [--keep-files] [--work DIR] \
      [--cache-dir DIR] [--artifacts-dir DIR] [--bench build/host/bitino_bench] [--out shard.json]

Per file (work files under --work, default the scratch tmpfs; per-file reports under <cache>/qonnx_step5/<id>/):
  1. export: the published file's data / graph SHA-256 = the recipe's expected hashes; canonical model.onnx written
     from it (refused unless check-qonnx gives no error and no warning);
  2. import: data / graph SHA-256 of the re-imported .bitino vs the published file and the recipe (the identity bar,
     docs/qonnx-profile.md Q11: data + graph identity, names byte for byte; the manifest is not compared);
  3. bit-exactness: the canonical file in onnxruntime (one session, ORT_DISABLE_ALL, the declared element types: INT8
     activations, INT64 epilogues; assumption A2) on every sample of the recipe's check set against the C++ runtime
     (bitino_bench stages int8_full_reference and int8_full) on the published file, every output byte for byte; the
     numpy oracle (bitino.int8_ops via qonnx_int8.run_ir) on the first --numpy-samples samples; the C++ reference
     stage on the re-imported file; qonnx execute_onnx (one session per node) on the first sample of the smaller files.
Unless --keep-files, model.onnx and the re-imported file are deleted after the row (their SHA-256 are in the report).
Writes reports/qonnx_step5/summary.json (merged with earlier rows).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import export_int8_qonnx as ei  # noqa: E402
from bitino import format as bff, qonnx_groups as qg, qonnx_int8 as qi  # noqa: E402
from bitino.fetch import data_sha256, graph_sha256, sha256_file  # noqa: E402

XCHECK_MAX_ONNX = 40e6          # qonnx execute_onnx cross-check (one session per node, every activation kept): files up
XCHECK_MAX_NODES = 6000         # to this size / node count whose IR activations of one sample take at most
XCHECK_MAX_ACT = 2e6            # this many elements and whose input has at most XCHECK_MAX_IN elements (CIFAR-size)
XCHECK_MAX_IN = 64 * 64 * 3


def bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint8),
                                                                              np.ascontiguousarray(b).view(np.uint8)))


def eval_inputs(mid: str, cache: str, g) -> tuple[dict, str, dict]:
    rec = ei.recipes()[mid]
    ev = rec["eval"]
    d = os.path.join(cache, "data", ev["dir"])
    em = json.load(open(os.path.join(d, "manifest.json")))
    X = qg.load_eval_inputs(d, ev["preprocess"], g)
    return X, os.path.join(d, em["preprocessing"][ev["preprocess"]]["file"]), em


def out_layout(g, shapes: dict, kinds: dict) -> list:
    return [(o, tuple(shapes[o]), {"f": np.float32, "i8": np.int8, "i32": np.int32}[kinds[o]]) for o in g.outputs]


def split_bench(raw: bytes, layout: list, n: int) -> list:
    """bitino_bench --outputs: per sample the raw bytes of every graph output in order."""
    per = [int(np.prod(s)) * np.dtype(dt).itemsize for _, s, dt in layout]
    tot = sum(per)
    if len(raw) != tot * n:
        raise ValueError(f"bench outputs {len(raw)} bytes, expected {tot * n}")
    buf = np.frombuffer(raw, np.uint8).reshape(n, tot)
    outs, off = [], 0
    for (_, s, dt), b in zip(layout, per):
        outs.append(np.ascontiguousarray(buf[:, off:off + b]).view(dt).reshape((n,) + tuple(s[1:])))
        off += b
    return outs


def run_bench(bench: str, model: str, inputs: str, out: str, stage: str) -> bytes:
    subprocess.run([bench, "--model", model, "--inputs", inputs, "--outputs", out, "--stage", stage, "--threads", "1",
                    "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
    raw = open(out, "rb").read()
    os.remove(out)
    return raw


def xcheck_execute_onnx(model, g, feeds_ir: dict, ref: list) -> dict:
    """qonnx execute_onnx (one onnxruntime session per node) on one sample."""
    try:
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.core.onnx_exec import execute_onnx
    except Exception as e:      # pragma: no cover
        return {"skipped": f"qonnx not importable: {e}"}
    from qonnx.transformation.infer_shapes import InferShapes
    t0 = time.time()
    mw = ModelWrapper(model).transform(InferShapes())
    out = execute_onnx(mw, qg.onnx_feeds(g, feeds_ir))
    ys = [np.asarray(out[o.name]) for o in model.graph.output]
    return {"bit_identical": all(bytes_equal(np.asarray(a).reshape(np.shape(b)), b) for a, b in zip(ys, ref)),
            "samples": 1, "seconds": round(time.time() - t0, 1)}


def ort_session_lean(model):
    """qonnx_groups.ort_session (ORT_DISABLE_ALL, one thread) without the memory arena and the memory pattern
    planner: the INT64 epilogues of the 224 x 224 networks otherwise keep several GB reserved. Values are unchanged."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    return ort.InferenceSession(qg.ort_model(model).SerializeToString(), so, providers=["CPUExecutionProvider"])


def run_file(mid: str, fn: str, args) -> dict:
    t_all = time.time()
    tag = fn.replace(".bitino", "")
    wd = os.path.join(args.work, mid, tag)
    os.makedirs(wd, exist_ok=True)
    pub = os.path.join(args.artifacts_dir, mid, fn)
    exp = ei.recipes()[mid]["expected"][fn]
    row = {"model_id": mid, "file": fn, "recipe_expected": {"data_sha256": exp["data_sha256"], "graph_sha256": exp["graph_sha256"]}}
    g = bff.read(pub)
    t0 = time.time()
    shapes, kinds = qi.ir_shapes_kinds(g)
    onnx_path = os.path.join(wd, "model.onnx")
    rec = ei.export(mid, fn, onnx_path, args.artifacts_dir, args.cache_dir, (shapes, kinds))
    row["export"] = {k: rec[k] for k in ("published", "writer", "ir_nodes", "ir_constants", "onnx_sha256", "onnx_bytes",
                                         "onnx_nodes", "profile_check")}
    row["export"]["seconds"] = round(time.time() - t0, 1)
    row["check_qonnx"] = "pass" if rec["profile_check"]["passed"] else "fail"
    import onnx
    model = onnx.load(onnx_path)
    t0 = time.time()
    g2, info = qg.import_groups(model)
    g2.manifest = {"schema": "bitino.manifest.v1", "model_id": mid, "variant": ei.FILES[fn],
                   "source": {"framework": "QONNX", "onnx_sha256": rec["onnx_sha256"]},
                   "conversion": {"tool": "bitino.qonnx_import (group mapping, bitino.qonnx_int8)", "rules": info["rules"],
                                  "python": platform.python_version(), "numpy": np.__version__, "onnx": onnx.__version__}}
    re_path = os.path.join(wd, fn)
    bff.write(g2, re_path)
    del g2
    row["import"] = {"data_sha256": data_sha256(re_path), "graph_sha256": graph_sha256(re_path),
                     "file_sha256": sha256_file(re_path), "seconds": round(time.time() - t0, 1),
                     "rules": {k: v for k, v in info["rules"].items() if not k.startswith("re-emission")},
                     "reemission_checked_groups": sum(v for k, v in info["rules"].items() if k.startswith("re-emission"))}
    pd, pg = rec["published"]["data_sha256"], rec["published"]["graph_sha256"]
    row["identical"] = {"data": row["import"]["data_sha256"] == pd, "graph": row["import"]["graph_sha256"] == pg,
                        "recipe_data": row["import"]["data_sha256"] == exp["data_sha256"],
                        "recipe_graph": row["import"]["graph_sha256"] == exp["graph_sha256"]}
    ident = all(row["identical"].values())
    print(f"[{mid}/{fn}] check-qonnx {row['check_qonnx']} | data {row['identical']['data']} graph {row['identical']['graph']} "
          f"| {rec['onnx_bytes'] / 1e6:.1f} MB, {rec['onnx_nodes']} nodes", flush=True)
    if not args.no_verify:
        row["bit_exact"] = verify(mid, fn, g, model, pub, re_path, shapes, kinds, rec, args)
    ok = ident and row["check_qonnx"] == "pass" and (args.no_verify or row["bit_exact"]["all_bit_identical"])
    row["status"] = "PASS" if ok else "FAIL"
    row["seconds"] = round(time.time() - t_all, 1)
    import resource
    row["peak_rss_mb_process"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    del model
    if not args.keep_files:
        shutil.rmtree(wd, ignore_errors=True)
    gc.collect()
    return row


def verify(mid, fn, g, model, pub, re_path, shapes, kinds, rec, args) -> dict:
    X, inp_file, em = eval_inputs(mid, args.cache_dir, g)
    n = len(next(iter(X.values())))
    if args.samples:
        n = min(n, args.samples)
    layout = out_layout(g, shapes, kinds)
    res = {"check_set": em.get("name"), "samples": n, "outputs": [o for o, _, _ in layout]}
    inp = inp_file
    if args.samples and args.samples < len(next(iter(X.values()))):
        per = os.path.getsize(inp_file) // len(next(iter(X.values())))
        inp = os.path.join(os.path.dirname(re_path), "inputs.bin")
        with open(inp_file, "rb") as f, open(inp, "wb") as o:
            o.write(f.read(per * n))
    cpp = {}
    t0 = time.time()
    n_ref = n if not args.ref_samples else min(n, args.ref_samples)
    per = os.path.getsize(inp) // n
    inp_ref = os.path.join(os.path.dirname(re_path), "inputs_ref.bin")
    with open(inp, "rb") as f, open(inp_ref, "wb") as o:
        o.write(f.read(per * n_ref))
    cpp["int8_full"] = split_bench(run_bench(args.bench, pub, inp, os.path.join(os.path.dirname(re_path), "cpp_opt.bin"),
                                             "int8_full"), layout, n)
    cpp["int8_full_reference"] = split_bench(run_bench(args.bench, pub, inp_ref, os.path.join(os.path.dirname(re_path), "cpp_ref.bin"),
                                                       "int8_full_reference"), layout, n_ref)
    res["cpp_reference_samples"] = n_ref
    n_re = min(n, args.reimported_samples)          # the re-imported file is data + graph identical: a sample suffices
    per = os.path.getsize(inp) // n
    inp_re = os.path.join(os.path.dirname(re_path), "inputs_re.bin")
    with open(inp, "rb") as f, open(inp_re, "wb") as o:
        o.write(f.read(per * n_re))
    cpp_re = split_bench(run_bench(args.bench, re_path, inp_re, os.path.join(os.path.dirname(re_path), "cpp_re.bin"),
                                   "int8_full_reference"), layout, n_re)
    os.remove(inp_re)
    res["cpp_reimported_samples"] = n_re
    res["cpp_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    sess = ort_session_lean(model)
    ort_out = [np.empty_like(a) for a in cpp["int8_full"]]
    for k in range(n):
        feeds = qg.onnx_feeds(g, {kk: v[k:k + 1] for kk, v in X.items()})
        ys = sess.run(None, feeds)
        for j, y in enumerate(ys):
            ort_out[j][k] = np.asarray(y).reshape(ort_out[j].shape[1:])
    del sess
    res["ort_seconds"] = round(time.time() - t0, 1)
    res["ort_vs_cpp_int8_full_reference"] = all(bytes_equal(a[:n_ref], b) for a, b in zip(ort_out, cpp["int8_full_reference"]))
    res["ort_vs_cpp_int8_full"] = all(bytes_equal(a, b) for a, b in zip(ort_out, cpp["int8_full"]))
    res["cpp_reimported_vs_published"] = all(bytes_equal(a, b[:n_re]) for a, b in zip(cpp_re, cpp["int8_full_reference"]))
    for st in ("int8_full_reference", "int8_full"):
        bad = sorted({int(k) for a, b in zip(ort_out, cpp[st]) for k in range(len(b)) if not bytes_equal(a[k], b[k])})
        res[f"samples_differing_vs_{st}"] = bad[:20]
    os.remove(inp_ref)
    k_np = min(n, args.numpy_samples)
    t0 = time.time()
    ok_np = True
    for k in range(k_np):
        ref = qi.run_ir(g, {kk: v[k:k + 1] for kk, v in X.items()})
        for j, (o, s, dt) in enumerate(layout):
            if not bytes_equal(np.asarray(ref[o], dt).reshape(s[1:]), ort_out[j][k]):
                ok_np = False
    res["numpy_oracle_vs_ort"] = {"samples": k_np, "bit_identical": ok_np, "seconds": round(time.time() - t0, 1)}
    act = sum(int(np.prod(shapes[o])) for nd in g.nodes for o in nd.outputs)
    in_max = max(int(np.prod(i["shape"])) for i in g.inputs)
    if rec["onnx_bytes"] <= XCHECK_MAX_ONNX and rec["onnx_nodes"] <= XCHECK_MAX_NODES and act <= XCHECK_MAX_ACT \
            and in_max <= XCHECK_MAX_IN \
            and not args.no_xcheck:
        try:
            res["qonnx_execute_onnx_vs_ort"] = xcheck_execute_onnx(model, g, {kk: v[0:1] for kk, v in X.items()},
                                                                   [a[0:1].reshape((1,) + a.shape[1:]) for a in ort_out])
        except Exception as e:
            res["qonnx_execute_onnx_vs_ort"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    else:
        res["qonnx_execute_onnx_vs_ort"] = {"skipped": f"file > {XCHECK_MAX_ONNX / 1e6:.0f} MB, > {XCHECK_MAX_NODES} nodes or "
                                                       f"> {XCHECK_MAX_ACT / 1e6:.0f}M activation elements per sample or an input above 64 x 64 x 3 "
                                                       "(execute_onnx keeps every activation)"}
    res["all_bit_identical"] = bool(res["ort_vs_cpp_int8_full_reference"] and res["ort_vs_cpp_int8_full"] and ok_np
                                    and res["cpp_reimported_vs_published"])
    print(f"[{mid}/{fn}] {n} samples: ORT vs C++ ref {res['ort_vs_cpp_int8_full_reference']} / opt {res['ort_vs_cpp_int8_full']} "
          f"| numpy {k_np} {ok_np} | re-imported C++ {res['cpp_reimported_vs_published']} | xcheck "
          f"{res['qonnx_execute_onnx_vs_ort'].get('bit_identical', res['qonnx_execute_onnx_vs_ort'])} | ORT {res['ort_seconds']} s",
          flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", help="model id or id/file (repeatable)")
    ap.add_argument("--no-verify", action="store_true", help="identity only (export + check-qonnx + import)")
    ap.add_argument("--no-xcheck", action="store_true", help="skip the qonnx execute_onnx cross-check")
    ap.add_argument("--samples", type=int, default=0, help="first N samples of the check set (0: all)")
    ap.add_argument("--numpy-samples", type=int, default=4)
    ap.add_argument("--reimported-samples", type=int, default=8, help="C++ reference stage on the re-imported file")
    ap.add_argument("--ref-samples", type=int, default=32,
                    help="C++ int8_full_reference stage on the first N samples (0: all; the int8_full stage always runs on all)")
    ap.add_argument("--keep-files", action="store_true")
    ap.add_argument("--work", default=os.path.join(ROOT, ".cache", "qonnx_step5", "_work"),
                    help="work directory for model.onnx / the re-imported file (deleted per row unless --keep-files)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--bench", default=os.path.join(ROOT, "build", "host", "bitino_bench"))
    ap.add_argument("--out", help="shard JSON (default: merge into reports/qonnx_step5/summary.json)")
    ap.add_argument("--skip-done", action="store_true", help="skip rows already PASS in the output file")
    args = ap.parse_args()
    files = ei.int8_files()
    if args.only:
        sel = set(args.only)
        files = [(m, f) for m, f in files if m in sel or f"{m}/{f}" in sel]
    out = args.out or os.path.join(ROOT, "reports", "qonnx_step5", "summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summ = json.load(open(out)) if os.path.exists(out) else {}
    rows = summ.setdefault("rows", {})
    summ["bench"] = {"path": os.path.relpath(args.bench, ROOT) if args.bench.startswith(ROOT) else args.bench,
                     "sha256": sha256_file(args.bench)}
    import onnx
    import onnxruntime
    summ["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "onnx": onnx.__version__,
                           "onnxruntime": onnxruntime.__version__, "host": platform.machine()}
    for mid, fn in files:
        key = f"{mid}/{fn}"
        if args.skip_done and rows.get(key, {}).get("status") == "PASS":
            continue
        try:
            row = run_file(mid, fn, args)
        except Exception as e:
            import traceback
            row = {"model_id": mid, "file": fn, "status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:800]}",
                   "traceback": traceback.format_exc()[-2000:]}
            print(f"[{key}] FAIL {row['error']}", flush=True)
            shutil.rmtree(os.path.join(args.work, mid), ignore_errors=True)
        rows[key] = row
        rd = os.path.join(args.cache_dir, "qonnx_step5", mid)
        os.makedirs(rd, exist_ok=True)
        json.dump(row, open(os.path.join(rd, fn.replace(".bitino", ".json")), "w"), indent=1, default=str)
        summ["counts"] = {"files": len(rows), "pass": sum(1 for r in rows.values() if r.get("status") == "PASS")}
        json.dump(summ, open(out, "w"), indent=1, default=str)
    print(f"summary: {summ['counts']} -> {out}")
    return 0 if all(r.get("status") == "PASS" for k, r in rows.items() if not args.only or k in {f"{m}/{f}" for m, f in files}) else 1


if __name__ == "__main__":
    sys.exit(main())
