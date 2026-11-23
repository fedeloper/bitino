#!/usr/bin/env python3
"""Step-2 acceptance of the QONNX route (docs/qonnx-frontend.md § Plan, step 2; report docs/qonnx-step2.md): every
Keras / Larq network of the registry written to a canonical QONNX file of the Bitino profile by the direct writer
(tools/export_keras_qonnx.py), checked with check-qonnx, imported (tools/import_qonnx.py) and compared with the
published files.

Runs in .cache/venv-qonnx with python -I; the exports run in .cache/venv-larq (stage 1, TensorFlow) and
.cache/venv-qonnx (stage 2, onnx), subprocesses with python -I:

  scripts/capped.sh 8G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_step2_acceptance.py \
      [--only ID ...] [--skip-run] [--no-verify] [--cache-dir DIR] [--artifacts-dir DIR] [--bench build/host/bitino_bench]

Scope: every recipe of registry/recipes.json whose converter reads a Keras / Larq model (tools/convert_larq_zoo.py,
tools/convert_keras_embedding.py; the adapter bitino.adapters.keras_larq): 23 networks, one published float file each
(model.bitino: larq's sign is 0 -> +1, the source semantics, so there is no model_sign_zero.bitino). The int8_full
files are step 5.

Per network, into <cache>/qonnx_step2/<id>/ (never artifacts/):
  1. tools/export_keras_qonnx.py: checkpoint SHA-256 check, the canonical model.onnx (the exporter refuses a file that
     does not pass check-qonnx), export.json, the source model's own fp32 / fp64 outputs on the check set;
  2. tools/check_qonnx.py on model.onnx (findings by level);
  3. tools/import_qonnx.py with the import arguments export.json records (--naming exporter --constant-order first-use
     --affine keep --cleanup none --bit-outputs none): import, Level-3 verification (qonnx execute_onnx fp32, OnnxF64
     fp64 with the float64 carriers, first binarization flip) on the first output (logits / embedding), C++ host runtime
     stage reference (--require-exact-numpy);
  4. comparison with artifacts/<id>/model.bitino: data SHA-256, graph SHA-256, the recipe's expected hashes. The
     identity bar (user decision 2026-11-20, docs/qonnx-profile.md Q11): data + graph SHA-256 identity -- tensor data and
     graph structure including every layer / node / constant name, byte for byte; the manifest is NOT compared;
     outputs of both files on the check set (every graph output): numpy executor (float32) and C++ runtime (stages
     reference and optimized), bit for bit.
Level 3 as step 1 (tools/qonnx_step1_acceptance.py: FLIP_REL, NOFLIP_REL). Writes reports/qonnx_step2/summary.json
(merged with earlier runs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.fetch import data_sha256, graph_sha256  # noqa: E402

CONVERTERS = ("tools/convert_larq_zoo.py", "tools/convert_keras_embedding.py")
REPOS = {"larq-zoo-quicknet-xl-legacy": "larq-zoo-40f4cdc"}      # {repo} of the recipe (read-only source tree)
FLIP_REL = 1e-3
NOFLIP_REL = 1e-5
PUB = "model.bitino"


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
    return [k for k, v in recs.items() if v.get("converter") in CONVERTERS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-run", action="store_true", help="only compare (exports / imports already in <cache>/qonnx_step2)")
    ap.add_argument("--only", action="append", help="model id (repeatable)")
    ap.add_argument("--no-verify", action="store_true", help="identity only: import without the eval set (no Level-3 run)")
    ap.add_argument("--no-export", action="store_true", help="re-import the existing model.onnx (no TensorFlow run)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"), help="published conversions (read only)")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build", "host", "bitino_bench"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_step2", "summary.json"))
    ap.add_argument("--exporter", default=os.path.join(ROOT, "tools", "export_keras_qonnx.py"))
    args = ap.parse_args()
    cache = os.path.abspath(args.cache_dir)
    venv_e = os.path.join(cache, "venv-larq", "bin", "python")
    venv_q = os.path.join(cache, "venv-qonnx", "bin", "python")
    recipes = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    models = {m["model_id"]: m for m in json.load(open(os.path.join(ROOT, "registry", "models.json")))["models"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    old = json.load(open(args.out)) if os.path.exists(args.out) else {"networks": {}}
    rows = old.get("networks", {})
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TF_CPP_MIN_LOG_LEVEL="2")
    failed = []
    for mid in networks():
        if args.only and mid not in args.only:
            continue
        rec = recipes[mid]
        ev = rec["eval"]
        eval_dir = os.path.join(cache, "data", ev["dir"])
        lc = models[mid]["checkpoint"]["local_cache_path"]
        weights = os.path.join(cache, lc[len(".cache/"):]) if lc.startswith(".cache/") else os.path.join(ROOT, lc)
        d = os.path.join(cache, "qonnx_step2", mid)
        os.makedirs(d, exist_ok=True)
        row = {"model_id": mid, "published_file": PUB, "legacy_converter": rec["converter"], "eval_dir": ev["dir"],
               "preprocess": ev["preprocess"]}
        t0 = time.time()
        if not args.skip_run and not args.no_export:
            cmd = [venv_e, "-I", args.exporter, "--model-id", mid, "--weights", weights, "--out-dir", d, "--eval-dir", eval_dir,
                   "--cache-dir", cache, "--qonnx-python", venv_q]
            if mid in REPOS:
                cmd += ["--repo", os.path.join(cache, "repos", REPOS[mid])]
            r = subprocess.run(cmd, env=env)
            if r.returncode:
                row.update({"status": "FAIL", "reason": f"export failed (rc {r.returncode})"})
                rows[mid] = row
                failed.append(mid)
                continue
        exp = json.load(open(os.path.join(d, "export.json")))
        row["export"] = {k: exp.get(k) for k in ("checkpoint_sha256", "exporter", "onnx_sha256", "onnx_bytes", "import_args",
                                                 "source_top1", "source_f32_vs_f64_max_abs", "writer", "const_order",
                                                 "output_widths", "ir_nodes", "onnx_nodes")}
        onnx_path = os.path.join(d, "model.onnx")
        chk = subprocess.run([venv_q, "-I", os.path.join(ROOT, "tools", "check_qonnx.py"), onnx_path, "--json", "-"],
                             capture_output=True, text=True)
        cj = json.loads(chk.stdout)
        cj = cj[0] if isinstance(cj, list) else cj
        row["check_qonnx"] = {"returncode": chk.returncode, "passed": cj.get("passed"),
                              "warnings": sorted({f["code"] for f in cj.get("findings", []) if f.get("level") == "warning"}),
                              "errors": sorted({f["code"] for f in cj.get("findings", []) if f.get("level") == "error"}),
                              "op_counts": (cj.get("summary") or {}).get("op_counts")}
        width = int(exp["output_widths"][0])
        if not args.skip_run:
            cmd = [venv_q, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), onnx_path, "--model-id", mid, "--out-dir", d,
                   "--exporter", exp["exporter"], "--source-json", os.path.join(d, "export.json")] + list(exp["import_args"])
            if not args.no_verify:
                cmd += ["--eval-dir", eval_dir, "--preprocess", ev["preprocess"], "--input-domain", "float32", "--bench", args.bench,
                        "--stages", "reference", "--num-classes", str(width)]
            r = subprocess.run(cmd, env=env)
            row["import_returncode"] = r.returncode
        crep = json.load(open(os.path.join(d, "conversion_report.json")))
        if "import_error" in crep:
            row.update({"status": "FAIL", "reason": f"import error: {crep['import_error']}"})
            rows[mid] = row
            failed.append(mid)
            continue
        new = os.path.join(d, "model.bitino")
        pub = os.path.join(args.artifacts_dir, mid, PUB)
        row["published"] = {"data_sha256": data_sha256(pub), "graph_sha256": graph_sha256(pub)}
        e = rec["expected"][PUB]
        row["recipe_expected"] = {"data_sha256": e["data_sha256"], "graph_sha256": e["graph_sha256"]}
        row["published_matches_recipe"] = row["published"] == row["recipe_expected"]
        row["reimported"] = {"data_sha256": data_sha256(new), "graph_sha256": graph_sha256(new), "bytes": os.path.getsize(new)}
        row["data_identical"] = row["reimported"]["data_sha256"] == row["published"]["data_sha256"]
        row["graph_identical"] = row["reimported"]["graph_sha256"] == row["published"]["graph_sha256"]
        row["recipe_identical"] = row["reimported"]["data_sha256"] == e["data_sha256"] and \
            row["reimported"]["graph_sha256"] == e["graph_sha256"]
        if not args.no_verify:
            pre = json.load(open(os.path.join(eval_dir, "manifest.json")))["preprocessing"][ev["preprocess"]]
            inp = os.path.join(eval_dir, pre["file"])
            x = np.fromfile(inp, np.float32).reshape(pre["shape"])
            ga, gb = bfformat.read(pub), bfformat.read(new)
            ya, yb = all_outputs(ga, x), all_outputs(gb, x)
            ca = run_bench(args.bench, pub, inp, os.path.join(d, "cpp_published.f32"))
            cb = run_bench(args.bench, new, inp, os.path.join(d, "cpp_reimported.f32"))
            oa = run_bench(args.bench, pub, inp, os.path.join(d, "cpp_published_optimized.f32"), "optimized")
            ob = run_bench(args.bench, new, inp, os.path.join(d, "cpp_reimported_optimized.f32"), "optimized")
            row["outputs_check_set"] = {"n_samples": int(len(x)), "eval_set": ev["dir"], "graph_outputs": list(gb.outputs),
                                        "numpy_f32_bit_identical": bits_equal(ya, yb),
                                        "cpp_reference_bit_identical": bits_equal(ca, cb),
                                        "cpp_optimized_bit_identical": bits_equal(oa, ob),
                                        "cpp_vs_numpy_reimported_bit_identical": bits_equal(cb, yb.reshape(-1))}
            for f in ("cpp_published.f32", "cpp_reimported.f32", "cpp_published_optimized.f32", "cpp_reimported_optimized.f32"):
                os.remove(os.path.join(d, f))
            v = crep.get("verification", {})
            sem = v.get("semantic_check_ir_f64_vs_reference_f64", {})
            ec32 = v.get("source_framework_f32_vs_qonnx_f32 (export check)", {})
            ec64 = v.get("source_framework_f64_vs_qonnx_f64 (export check)", {})
            fl = v.get("binarization_flips_vs_source", {})
            mx32 = [s.get("ir_f32_vs_source_f32", {}).get("max_abs_pre_source") for s in fl.get("samples", [])]
            mx64 = [s.get("ir_f64_vs_source_f64", {}).get("max_abs_pre_source") for s in fl.get("samples", [])]
            ref64 = np.fromfile(os.path.join(d, "reference_logits_f64.f64"), np.float64).reshape(len(x), -1)
            ir64 = np.stack([executor.run(gb, {"input": x[i:i + 1]}, float_dtype=np.float64)[gb.outputs[0]].reshape(-1)
                             for i in range(len(x))])
            per = np.abs(ir64 - ref64).max(axis=1)
            scale = np.abs(ref64).max(axis=1)
            fls = fl.get("samples", [])
            complete = len(fls) == fl.get("n_samples_with_flips", 0)
            f64 = {q["sample"]: q["ir_f64_vs_source_f64"] for q in fls if "ir_f64_vs_source_f64" in q}
            dev = [int(i) for i in np.where(per > 1e-6)[0]]
            flip_rel = {i: f64[i]["max_abs_pre_source"] / max(f64[i]["typical_abs_pre_source"], 1e-30) for i in dev if i in f64}
            noflip_rel = {i: float(per[i] / max(scale[i], 1e-30)) for i in dev if i not in f64}
            level3 = {"rule": f"deviation > 1e-6: first flip at |pre| <= {FLIP_REL:g} x median |pre| of its binarizer, "
                              f"or no flip and max |delta| <= {NOFLIP_REL:g} x max |output|",
                      "samples_le_1e-6": int(np.sum(per <= 1e-6)), "deviating": len(dev),
                      "with_flip": len(flip_rel), "max_rel_pre_at_first_flip": max(flip_rel.values(), default=None),
                      "max_abs_pre_at_first_flip": max((f64[i]["max_abs_pre_source"] for i in flip_rel), default=None),
                      "without_flip": len(noflip_rel), "max_rel_delta_without_flip": max(noflip_rel.values(), default=None),
                      "max_abs_delta_without_flip": max((float(per[i]) for i in noflip_rel), default=None),
                      "flip_list_complete": complete}
            level3["passed"] = bool(complete and all(r_ <= FLIP_REL for r_ in flip_rel.values())
                                    and all(r_ <= NOFLIP_REL for r_ in noflip_rel.values()))
            l4 = v.get("acceptance") or {}
            acc = {"rule": "Level 3 (docs/numerical-correctness.md) + C++ reference stage bit-exact with numpy",
                   "passed": level3["passed"], "level3": level3,
                   "level4_informational": {"criterion": l4.get("criterion"), "passed": l4.get("passed")}}
            row["verification"] = {
                "acceptance": acc,
                "semantic_f64_max_abs": sem.get("max_abs_error"), "semantic_f64_top1": sem.get("top1_agreement"),
                "samples_le_1e-6": level3["samples_le_1e-6"],
                "binarization_flip_samples": fl.get("n_samples_with_flips"),
                "max_abs_pre_at_flip_f32": max([m for m in mx32 if m is not None], default=None),
                "max_abs_pre_at_flip_f64": max([m for m in mx64 if m is not None], default=None),
                "qonnx_f32_vs_f64_top1_disagreements": v.get("reference_f32_vs_f64", {}).get("n_top1_disagreements"),
                "numpy_f32_vs_qonnx_f64": {k: v.get("numpy_f32_vs_reference_f64", {}).get(k) for k in ("top1_agreement", "n_top1_disagreements")},
                "export_check_source_f32_vs_qonnx_f32": {k: ec32.get(k) for k in ("max_abs_error", "top1_agreement", "n_top1_disagreements",
                                                                                  "cosine_min")},
                "export_check_source_f64_vs_qonnx_f64": {k: ec64.get(k) for k in ("max_abs_error", "top1_agreement", "n_top1_disagreements",
                                                                                  "cosine_min")},
                "cpp_host_reference_exact_vs_numpy": [c.get("exact_vs_numpy") for c in crep.get("cpp_host", []) if c.get("stage") == "reference"],
            }
        row["import_rules"] = crep.get("import", {}).get("rules")
        row["seconds"] = round(time.time() - t0, 1)
        ident = row["data_identical"] and row["graph_identical"] and row["recipe_identical"]
        ok_chk = bool(row["check_qonnx"]["passed"]) and not row["check_qonnx"]["warnings"]
        if args.no_verify:
            row["status"] = "IDENTICAL (data + graph SHA-256, names included; manifest not compared), not verified" \
                if ident and ok_chk else "FAIL"
        else:
            o = row["outputs_check_set"]
            ver = row["verification"]
            acc_ok = bool((ver.get("acceptance") or {}).get("passed")) and all(ver["cpp_host_reference_exact_vs_numpy"] or [False])
            outs = o["numpy_f32_bit_identical"] and o["cpp_reference_bit_identical"] and o["cpp_optimized_bit_identical"]
            row["status"] = "PASS: data + graph SHA-256 identical (names included; manifest not compared), outputs " \
                            "bit-identical, Level 3 passed" if ident and ok_chk and outs and acc_ok else "FAIL"
        if row["status"] == "FAIL":
            failed.append(mid)
        rows[mid] = row
        print(f"{mid:44s} check {'ok' if ok_chk else 'FAIL'} data {'=' if row['data_identical'] else '!='} "
              f"graph {'=' if row['graph_identical'] else '!='} -> {row['status']} ({row['seconds']} s)", flush=True)
        json.dump({"schema": "bitino.qonnx_step2.v1", "networks": rows}, open(args.out, "w"), indent=1)
    json.dump({"schema": "bitino.qonnx_step2.v1", "networks": rows}, open(args.out, "w"), indent=1)
    if failed:
        print("FAILED:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
