#!/usr/bin/env python3
"""Step-4 acceptance of the QONNX route (docs/qonnx-frontend.md § Plan, step 4; report docs/qonnx-step4.md): every
network of the front ends without an ONNX exporter (OpenVINO / OMZ, daBNN, NNCF, MindSpore, tensorpack, Theano /
Lasagne) and the FINN / Brevitas networks whose phase-1 files were not canonical, written to a canonical QONNX file
of the Bitino profile by the direct writer (tools/export_ir_qonnx.py), checked with check-qonnx, imported
(tools/import_qonnx.py) and compared with the published files.

Runs in .cache/venv-qonnx with python -I; the exports run in each front end's environment (export_ir_qonnx.VENVS),
subprocesses with python -I:

  scripts/capped.sh 8G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_step4_acceptance.py \
      [--only ID ...] [--no-verify] [--no-export] [--skip-run] [--regress N] [--out JSON] [--cache-dir DIR] \
      [--artifacts-dir DIR] [--bench build/host/bitino_bench]

Per network, into <cache>/qonnx_step4/<id>/<file stem>/ (never artifacts/):
  1. tools/export_ir_qonnx.py: checkpoint SHA-256 check, the canonical model.onnx per published float file (the
     exporter refuses a file that does not pass check-qonnx), export.json, the source framework's fp32 / fp64 logits
     on the check set (<cache>/qonnx_step4/<id>/source_logits*.f*);
  2. tools/check_qonnx.py on model.onnx (findings by level);
  3. tools/import_qonnx.py with the import arguments export.json records (the bit-output layout is the file's
     bitino.bit_outputs): import, Level-3 verification (qonnx execute_onnx fp32, OnnxF64 fp64 with the float64
     carriers, first binarization flip), C++ host runtime stage reference (--require-exact-numpy);
  4. comparison with artifacts/<id>/<file>: data SHA-256, graph SHA-256, the recipe's expected hashes (model.bitino).
     Identity bar (user decision 2026-11-20, docs/qonnx-profile.md Q11): data + graph SHA-256 identity, names byte for
     byte; the manifest is NOT compared. Outputs of both files on the check set: numpy executor (float32) and C++
     runtime (stages reference and optimized), bit for bit.
Level 3 as step 1 / 2 (FLIP_REL, NOFLIP_REL). --regress N re-imports N canonical files of each of steps 1, 1b and 2
(their recorded import arguments) and compares with their summaries' re-imported hashes (the importer changes of this
step must not change them). Writes reports/qonnx_step4/summary.json (merged with earlier runs).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.fetch import data_sha256, graph_sha256  # noqa: E402
from export_ir_qonnx import FAMILIES, VENVS  # noqa: E402

FLIP_REL = 1e-3
NOFLIP_REL = 1e-5
SCHEMA = "bitino.qonnx_step4.v1"


def run_bench(bench: str, model: str, inputs: str, out: str, stage: str = "reference") -> np.ndarray:
    subprocess.run([bench, "--model", model, "--inputs", inputs, "--outputs", out, "--stage", stage, "--threads", "1",
                    "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
    return np.fromfile(out, np.float32)


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32)))


def all_outputs(g, x: np.ndarray) -> np.ndarray:
    return np.stack([np.concatenate([r[o].reshape(-1) for o in g.outputs])
                     for r in (executor.run(g, {"input": x[i:i + 1]}) for i in range(len(x)))])


def networks() -> list:
    recs = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    return [k for k, v in recs.items() if v.get("converter") in FAMILIES]


def published_files(artifacts: str, mid: str) -> list:
    """The published float files of a network: model.bitino, and model_sign_zero.bitino where published."""
    return [f for f in ("model.bitino", "model_sign_zero.bitino") if os.path.exists(os.path.join(artifacts, mid, f))]


def check_qonnx(venv_q: str, path: str) -> dict:
    chk = subprocess.run([venv_q, "-I", os.path.join(ROOT, "tools", "check_qonnx.py"), path, "--json", "-"], capture_output=True, text=True)
    cj = json.loads(chk.stdout)
    cj = cj[0] if isinstance(cj, list) else cj
    return {"returncode": chk.returncode, "passed": cj.get("passed"),
            "warnings": sorted({f["code"] for f in cj.get("findings", []) if f.get("level") == "warning"}),
            "errors": sorted({f["code"] for f in cj.get("findings", []) if f.get("level") == "error"}),
            "op_counts": (cj.get("summary") or {}).get("op_counts")}


def level3(crep: dict, d: str, gb, x: np.ndarray) -> dict:
    v = crep.get("verification", {})
    fl = v.get("binarization_flips_vs_source", {})
    ref64 = np.fromfile(os.path.join(d, "reference_logits_f64.f64"), np.float64).reshape(len(x), -1)
    ir64 = np.stack([executor.run(gb, {"input": x[i:i + 1]}, float_dtype=np.float64)[gb.outputs[0]].reshape(-1) for i in range(len(x))])
    per = np.abs(ir64 - ref64).max(axis=1)
    scale = np.abs(ref64).max(axis=1)
    fls = fl.get("samples", [])
    complete = len(fls) == fl.get("n_samples_with_flips", 0)
    f64 = {q["sample"]: q["ir_f64_vs_source_f64"] for q in fls if "ir_f64_vs_source_f64" in q}
    dev = [int(i) for i in np.where(per > 1e-6)[0]]
    flip_rel = {i: f64[i]["max_abs_pre_source"] / max(f64[i]["typical_abs_pre_source"], 1e-30) for i in dev if i in f64}
    noflip_rel = {i: float(per[i] / max(scale[i], 1e-30)) for i in dev if i not in f64}
    out = {"rule": f"deviation > 1e-6: first flip at |pre| <= {FLIP_REL:g} x median |pre| of its binarizer, "
                   f"or no flip and max |delta| <= {NOFLIP_REL:g} x max |output|",
           "samples_le_1e-6": int(np.sum(per <= 1e-6)), "deviating": len(dev),
           "with_flip": len(flip_rel), "max_rel_pre_at_first_flip": max(flip_rel.values(), default=None),
           "max_abs_pre_at_first_flip": max((f64[i]["max_abs_pre_source"] for i in flip_rel), default=None),
           "without_flip": len(noflip_rel), "max_rel_delta_without_flip": max(noflip_rel.values(), default=None),
           "max_abs_delta_without_flip": max((float(per[i]) for i in noflip_rel), default=None),
           "flip_list_complete": complete}
    out["passed"] = bool(complete and all(r <= FLIP_REL for r in flip_rel.values()) and all(r <= NOFLIP_REL for r in noflip_rel.values()))
    return out


def regress(cache: str, venv_q: str, n: int, extra: list | None = None) -> dict:
    """Re-import a sample of the canonical files of steps 1, 1b and 2 with their recorded import arguments (identity
    only) and compare with the re-imported hashes of their summaries."""
    out = {}
    for step in ("step1", "step1b", "step2"):
        rows = json.load(open(os.path.join(ROOT, "reports", f"qonnx_{step}", "summary.json")))["networks"]
        keys = [k for k, r in rows.items() if r.get("reimported") and r.get("data_identical") and r.get("graph_identical")
                and not k.startswith("bnext-large")]   # BNext-L: its import needs more than this job's memory budget
        pick = keys[:: max(1, len(keys) // n)][:n] if n < len(keys) else keys
        pick += [k.split("/", 1)[1] for k in (extra or []) if k.split("/", 1)[0] == step and k.split("/", 1)[1] not in pick and k.split("/", 1)[1] in rows]
        res = []
        for key in pick:
            r = rows[key]
            mid, var = (key.split(":") + [None])[:2]
            d = os.path.join(cache, f"qonnx_{step}", mid, var) if var else os.path.join(cache, f"qonnx_{step}", mid)
            onnx_path = os.path.join(d, "model.onnx")
            exp = json.load(open(os.path.join(d, "export.json"))) if os.path.exists(os.path.join(d, "export.json")) else {}
            args = list(exp.get("import_args") or (r.get("export") or {}).get("import_args") or [])
            if not os.path.exists(onnx_path):
                res.append({"row": key, "status": "canonical file not on disk"})
                continue
            tmp = tempfile.mkdtemp(prefix=f"regress_{step}_", dir=os.path.join(cache, "qonnx_step4"))
            p = subprocess.run([venv_q, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), onnx_path, "--model-id", mid, "--out-dir", tmp] + args,
                               capture_output=True, text=True)
            new = os.path.join(tmp, "model.bitino")
            if p.returncode or not os.path.exists(new):
                res.append({"row": key, "status": "import failed", "stderr": p.stderr.strip().splitlines()[-2:]})
            else:
                got = {"data_sha256": data_sha256(new), "graph_sha256": graph_sha256(new)}
                want = {k: r["reimported"][k] for k in ("data_sha256", "graph_sha256")}
                res.append({"row": key, "import_args": args, "identical": got == want, "data_sha256": got["data_sha256"],
                            "graph_sha256": got["graph_sha256"]})
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"regress {step} {key}: {res[-1].get('identical', res[-1].get('status'))}", flush=True)
        out[step] = {"checked": len(res), "identical": sum(1 for q in res if q.get("identical")), "rows": res}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-run", action="store_true", help="only compare (exports / imports already in <cache>/qonnx_step4)")
    ap.add_argument("--only", action="append", help="model id (repeatable)")
    ap.add_argument("--no-verify", action="store_true", help="identity only: import without the eval set (no Level-3 run)")
    ap.add_argument("--no-export", action="store_true", help="re-import the existing model.onnx files (no source run)")
    ap.add_argument("--regress", type=int, default=0, help="also re-check N canonical files of each of steps 1, 1b, 2 (identity)")
    ap.add_argument("--regress-row", action="append", help="also re-check this row, '<step>/<summary key>' (repeatable)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"), help="published conversions (read only)")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build", "host", "bitino_bench"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_step4", "summary.json"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cache = os.path.abspath(args.cache_dir)
    venv_q = os.path.join(cache, "venv-qonnx", "bin", "python")
    recipes = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    os.makedirs(args.out and os.path.dirname(args.out), exist_ok=True)
    old = json.load(open(args.out)) if os.path.exists(args.out) else {"networks": {}}
    rows = old.get("networks", {})
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), TF_CPP_MIN_LOG_LEVEL="2")
    tmpd = os.path.join(cache, "qonnx_step4", "tmp")
    os.makedirs(tmpd, exist_ok=True)
    failed = []
    for mid in networks():
        if args.only and mid not in args.only:
            continue
        rec = recipes[mid]
        fam = FAMILIES[rec["converter"]]
        ev = rec["eval"]
        eval_dir = os.path.join(cache, "data", ev["dir"])
        nd = os.path.join(cache, "qonnx_step4", mid)
        os.makedirs(nd, exist_ok=True)
        files = published_files(args.artifacts_dir, mid)
        t0 = time.time()
        export_rc = 0
        if not args.skip_run and not args.no_export:
            cmd = [os.path.join(cache, VENVS[fam], "bin", "python"), "-I", os.path.join(ROOT, "tools", "export_ir_qonnx.py"),
                   "--model-id", mid, "--out-dir", nd, "--eval-dir", eval_dir, "--cache-dir", cache, "--qonnx-python", venv_q,
                   "--threads", str(args.threads), "--tmp-dir", tmpd] + (["--no-source-logits"] if args.no_verify else [])
            export_rc = subprocess.run(cmd, env=env).returncode
        for pub_name in files:
            stem = pub_name[: -len(".bitino")]
            key = mid if stem == "model" else f"{mid}:{stem}"
            d = os.path.join(nd, stem)
            row = {"model_id": mid, "published_file": pub_name, "family": fam, "legacy_converter": rec["converter"],
                   "eval_dir": ev["dir"], "preprocess": ev["preprocess"]}
            if export_rc or not os.path.exists(os.path.join(d, "export.json")):
                row.update({"status": "FAIL", "reason": f"export failed (rc {export_rc})"})
                rows[key] = row
                failed.append(key)
                continue
            exp = json.load(open(os.path.join(d, "export.json")))
            row["export"] = {k: exp.get(k) for k in ("checkpoint_sha256", "exporter", "onnx_sha256", "onnx_bytes", "import_args", "import_options", "writer",
                                                     "const_order", "ir_unique_counter", "output_widths", "ir_nodes", "onnx_nodes",
                                                     "source_top1", "source_f32_vs_f64_max_abs", "source_logits", "legacy_passes")}
            onnx_path = os.path.join(d, "model.onnx")
            row["check_qonnx"] = check_qonnx(venv_q, onnx_path)
            width = int(exp["output_widths"][0])
            if not args.skip_run:
                cmd = [venv_q, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), onnx_path, "--model-id", mid, "--out-dir", d,
                       "--exporter", exp["exporter"], "--source-json", os.path.join(d, "export.json"), "--file-name", "model.bitino"] \
                    + list(exp["import_args"])
                if not args.no_verify:
                    cmd += ["--eval-dir", eval_dir, "--preprocess", ev["preprocess"], "--input-domain", "float32", "--bench", args.bench,
                            "--stages", "reference", "--num-classes", str(width)]
                if os.path.exists(os.path.join(d, "conversion_report.json")):
                    os.remove(os.path.join(d, "conversion_report.json"))       # never read a stale report
                row["import_returncode"] = subprocess.run(cmd, env=env).returncode
                if not os.path.exists(os.path.join(d, "conversion_report.json")):
                    row.update({"status": "FAIL", "reason": f"import crashed (rc {row['import_returncode']})"})
                    rows[key] = row
                    failed.append(key)
                    continue
            crep = json.load(open(os.path.join(d, "conversion_report.json")))
            if "import_error" in crep:
                row.update({"status": "FAIL", "reason": f"import error: {crep['import_error']}"})
                rows[key] = row
                failed.append(key)
                continue
            new = os.path.join(d, "model.bitino")
            pub = os.path.join(args.artifacts_dir, mid, pub_name)
            row["published"] = {"data_sha256": data_sha256(pub), "graph_sha256": graph_sha256(pub)}
            e = rec["expected"].get(pub_name)
            row["recipe_expected"] = {"data_sha256": e["data_sha256"], "graph_sha256": e["graph_sha256"]} if e else None
            row["published_matches_recipe"] = (row["published"] == row["recipe_expected"]) if e else None
            row["reimported"] = {"data_sha256": data_sha256(new), "graph_sha256": graph_sha256(new), "bytes": os.path.getsize(new)}
            row["data_identical"] = row["reimported"]["data_sha256"] == row["published"]["data_sha256"]
            row["graph_identical"] = row["reimported"]["graph_sha256"] == row["published"]["graph_sha256"]
            row["recipe_identical"] = (row["reimported"]["data_sha256"] == e["data_sha256"] and
                                       row["reimported"]["graph_sha256"] == e["graph_sha256"]) if e else None
            if not args.no_verify:
                pre = json.load(open(os.path.join(eval_dir, "manifest.json")))["preprocessing"][ev["preprocess"]]
                inp = os.path.join(eval_dir, pre["file"])
                x = np.fromfile(inp, np.float32).reshape(pre["shape"])
                ga, gb = bfformat.read(pub), bfformat.read(new)
                ya, yb = all_outputs(ga, x), all_outputs(gb, x)
                outs = {}
                for tag, path in (("published", pub), ("reimported", new)):
                    for st in ("reference", "optimized"):
                        outs[(tag, st)] = run_bench(args.bench, path, inp, os.path.join(d, f"cpp_{tag}_{st}.f32"), st)
                        os.remove(os.path.join(d, f"cpp_{tag}_{st}.f32"))
                row["outputs_check_set"] = {"n_samples": int(len(x)), "eval_set": ev["dir"], "graph_outputs": list(gb.outputs),
                                            "numpy_f32_bit_identical": bits_equal(ya, yb),
                                            "cpp_reference_bit_identical": bits_equal(outs[("published", "reference")], outs[("reimported", "reference")]),
                                            "cpp_optimized_bit_identical": bits_equal(outs[("published", "optimized")], outs[("reimported", "optimized")]),
                                            "cpp_vs_numpy_reimported_bit_identical": bits_equal(outs[("reimported", "reference")], yb.reshape(-1))}
                v = crep.get("verification", {})
                sem = v.get("semantic_check_ir_f64_vs_reference_f64", {})
                ec32 = v.get("source_framework_f32_vs_qonnx_f32 (export check)", {})
                ec64 = v.get("source_framework_f64_vs_qonnx_f64 (export check)", {})
                fl = v.get("binarization_flips_vs_source", {})
                l3 = level3(crep, d, gb, x)
                l4 = v.get("acceptance") or {}
                row["verification"] = {
                    "acceptance": {"rule": "Level 3 (docs/numerical-correctness.md) + C++ reference stage bit-exact with numpy",
                                   "passed": l3["passed"], "level3": l3,
                                   "level4_informational": {"criterion": l4.get("criterion"), "passed": l4.get("passed")}},
                    "semantic_f64_max_abs": sem.get("max_abs_error"), "semantic_f64_top1": sem.get("top1_agreement"),
                    "binarization_flip_samples": fl.get("n_samples_with_flips"),
                    "qonnx_f32_vs_f64_top1_disagreements": v.get("reference_f32_vs_f64", {}).get("n_top1_disagreements"),
                    "numpy_f32_vs_qonnx_f64": {k: v.get("numpy_f32_vs_reference_f64", {}).get(k) for k in ("top1_agreement", "n_top1_disagreements")},
                    "export_check_source_f32_vs_qonnx_f32": {k: ec32.get(k) for k in ("max_abs_error", "top1_agreement", "n_top1_disagreements")},
                    "export_check_source_f64_vs_qonnx_f64": {k: ec64.get(k) for k in ("max_abs_error", "top1_agreement", "n_top1_disagreements")},
                    "cpp_host_reference_exact_vs_numpy": [c.get("exact_vs_numpy") for c in crep.get("cpp_host", []) if c.get("stage") == "reference"],
                }
                for f in ("numpy_logits_fused.f32", "numpy_logits_f64.f64", "reference_logits.f32"):
                    if os.path.exists(os.path.join(d, f)):
                        os.remove(os.path.join(d, f))
            row["import_rules"] = crep.get("import", {}).get("rules")
            row["import_passes"] = crep.get("import", {}).get("passes")
            row["seconds"] = round(time.time() - t0, 1)
            ident = row["data_identical"] and row["graph_identical"] and row["recipe_identical"] is not False
            ok_chk = bool(row["check_qonnx"]["passed"]) and not row["check_qonnx"]["warnings"]
            if args.no_verify:
                row["status"] = "IDENTICAL (data + graph SHA-256, names included; manifest not compared), not verified" \
                    if ident and ok_chk else "FAIL"
            else:
                o = row["outputs_check_set"]
                ver = row["verification"]
                acc_ok = bool(ver["acceptance"]["passed"]) and all(ver["cpp_host_reference_exact_vs_numpy"] or [False])
                outs_ok = o["numpy_f32_bit_identical"] and o["cpp_reference_bit_identical"] and o["cpp_optimized_bit_identical"]
                row["status"] = "PASS: data + graph SHA-256 identical (names included; manifest not compared), outputs " \
                                "bit-identical, Level 3 passed" if ident and ok_chk and outs_ok and acc_ok else "FAIL"
            if row["status"] == "FAIL":
                failed.append(key)
            rows[key] = row
            print(f"{key:52s} check {'ok' if ok_chk else 'FAIL'} data {'=' if row['data_identical'] else '!='} "
                  f"graph {'=' if row['graph_identical'] else '!='} -> {row['status']} ({row['seconds']} s)", flush=True)
            json.dump({**old, "schema": SCHEMA, "networks": rows}, open(args.out, "w"), indent=1)
    res = {**old, "schema": SCHEMA, "networks": rows}
    if args.regress or args.regress_row:
        res["regression"] = regress(cache, venv_q, args.regress, args.regress_row)
    json.dump(res, open(args.out, "w"), indent=1)
    shutil.rmtree(tmpd, ignore_errors=True)
    if failed:
        print("FAILED:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
