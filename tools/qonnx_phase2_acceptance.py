#!/usr/bin/env python3
"""Phase-2 acceptance of the QONNX route (docs/qonnx-frontend.md § Plan, phase 2): plain-PyTorch networks exported with
torch.onnx.export (tools/export_torch_qonnx.py), cleaned with qonnx and imported (tools/import_qonnx.py), compared with
the files the legacy converters published.

Runs in .cache/venv-qonnx with python -I; the exports run in .cache/venv-qonnx2 (subprocess, python -I):

  scripts/capped.sh 8G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_phase2_acceptance.py \
      [--skip-run] [--only ID] [--cache-dir DIR] [--artifacts-dir DIR] [--bench build/host/bitino_bench]

Per network and binarizer variant, into <cache>/qonnx_phase2/<id>/<variant>/ (never artifacts/):
  variant ge         export --binarizer ge     (BipolarQuant; the published deployable file model.bitino)
  variant sign_zero  export --binarizer source (ONNX Sign = torch.sign; the published model_sign_zero.bitino)
  1. tools/export_torch_qonnx.py: checkpoint SHA-256 check, model.onnx, export.json, the source model's own fp32 / fp64
     logits on the model's eval128 check set;
  2. tools/import_qonnx.py with the import arguments export.json records (--naming exporter --constant-order first-use
     --affine keep): qonnx cleanup, import, Level-3 verification, C++ host runtime stage reference (--require-exact-numpy;
     the optimized stage of these ResNets is not bit-exact with the numpy executor, for the published files either:
     their float convolutions sum in another order, accepted by the legacy validation through flip tracing);
  3. comparison with the published file artifacts/<id>/<file>: data SHA-256, graph SHA-256 (fetch.graph_sha256), for
     model.bitino also the recipe's expected hashes; when the graph differs, the graph with positional names
     (tools/qonnx_phase1_acceptance.canonical) and the renamed items; outputs of both files on the check set: numpy
     executor (float32) and C++ runtime (stages reference and optimized), bit for bit.
Writes reports/qonnx_phase2/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.fetch import data_sha256, graph_sha256  # noqa: E402
from qonnx_phase1_acceptance import canonical  # noqa: E402

# id -> (repo under <cache>/repos, checkpoint under <cache>/checkpoints)
NETWORKS = {
    "ab-bnn-imagenet-reactnet18": ("AB-BNN", "ab-bnn/imagenet_reactnet18_step2.h5"),
    "rbonn-reactnet-resnet18-imagenet": ("SteveTsui_RBONN", "rbonn-reactnet-resnet18-imagenet/RBONN/model_best.pth.tar"),
    "recu-resnet18-imagenet": ("yuanchunyu_ReCU", "recu-resnet18-imagenet/ResNet-18_finetune_61.20/model_best.pth.tar"),
    "biper-resnet18-imagenet": ("edmav4_BiPer", "biper-resnet18-imagenet/model_best.pth.tar"),
}
# variant -> (exporter --binarizer, published file)
VARIANTS = {"ge": ("ge", "model.bitino"), "sign_zero": ("source", "model_sign_zero.bitino")}


def run_bench(bench: str, model: str, inputs: str, out: str, stage: str = "reference") -> np.ndarray:
    subprocess.run([bench, "--model", model, "--inputs", inputs, "--outputs", out, "--stage", stage, "--threads", "1",
                    "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
    return np.fromfile(out, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-run", action="store_true", help="only compare (exports / imports already in <cache>/qonnx_phase2)")
    ap.add_argument("--only", action="append", help="model id (repeatable)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"), help="published conversions (read only)")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build", "host", "bitino_bench"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_phase2", "summary.json"))
    args = ap.parse_args()
    cache = os.path.abspath(args.cache_dir)
    venv_e = os.path.join(cache, "venv-qonnx2", "bin", "python")
    venv_q = os.path.join(cache, "venv-qonnx", "bin", "python")
    recipes = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    old = json.load(open(args.out)) if os.path.exists(args.out) else {"networks": {}}
    rows = old.get("networks", {})
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for mid, (repo, ck) in NETWORKS.items():
        if args.only and mid not in args.only:
            continue
        rec = recipes[mid]
        ev = rec["eval"]
        eval_dir = os.path.join(cache, "data", ev["dir"])
        for var, (binarizer, pub_name) in VARIANTS.items():
            if args.variant and var not in args.variant:
                continue
            key = f"{mid}:{var}"
            d = os.path.join(cache, "qonnx_phase2", mid, var)
            os.makedirs(d, exist_ok=True)
            row = {"model_id": mid, "variant": var, "published_file": pub_name, "eval_dir": ev["dir"], "preprocess": ev["preprocess"]}
            if not args.skip_run:
                subprocess.run([venv_e, "-I", os.path.join(ROOT, "tools", "export_torch_qonnx.py"), "--model-id", mid,
                                "--repo", os.path.join(cache, "repos", repo), "--weights", os.path.join(cache, "checkpoints", ck),
                                "--out-dir", d, "--eval-dir", eval_dir, "--binarizer", binarizer], check=True, env=env)
            exp = json.load(open(os.path.join(d, "export.json")))
            row["export"] = {k: exp.get(k) for k in ("checkpoint_sha256", "exporter", "onnx_sha256", "binarizer", "import_args",
                                                     "source_top1", "source_exact_zeros", "source_f32_vs_f64_top1_disagreements",
                                                     "exportable_vs_source")}
            if not args.skip_run:
                r = subprocess.run([venv_q, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), os.path.join(d, "model.onnx"),
                                    "--model-id", mid, "--out-dir", d, "--eval-dir", eval_dir, "--preprocess", ev["preprocess"],
                                    "--input-domain", "float32", "--bench", args.bench, "--source-json", os.path.join(d, "export.json"),
                                    "--exporter", exp["exporter"], "--stages", "reference"] + list(exp["import_args"]), env=env)
                row["import_returncode"] = r.returncode
            crep = json.load(open(os.path.join(d, "conversion_report.json")))
            if "import_error" in crep:
                row.update({"status": "FAIL", "reason": f"import error: {crep['import_error']}"})
                rows[key] = row
                continue
            new = os.path.join(d, "model.bitino")
            pub = os.path.join(args.artifacts_dir, mid, pub_name)
            row["published"] = {"data_sha256": data_sha256(pub), "graph_sha256": graph_sha256(pub)}
            if pub_name in rec.get("expected", {}):
                e = rec["expected"][pub_name]
                row["recipe_expected"] = {"data_sha256": e["data_sha256"], "graph_sha256": e["graph_sha256"]}
                row["published_matches_recipe"] = row["published"] == row["recipe_expected"]
            row["reimported"] = {"data_sha256": data_sha256(new), "graph_sha256": graph_sha256(new), "bytes": os.path.getsize(new)}
            row["data_identical"] = row["reimported"]["data_sha256"] == row["published"]["data_sha256"]
            row["graph_identical"] = row["reimported"]["graph_sha256"] == row["published"]["graph_sha256"]
            cn, nn_ = canonical(new)
            cp, np_ = canonical(pub)
            row["graph_identical_up_to_names"] = cn == cp
            if not row["graph_identical"]:
                ren_t = [[a, b] for a, b in zip(np_["tensors"], nn_["tensors"]) if a != b]
                ren_n = [[a, b] for a, b in zip(np_["nodes"], nn_["nodes"]) if a != b]
                row["renamed"] = {"constants": len(ren_t), "nodes": len(ren_n), "examples": ren_n[:4] + ren_t[:4]}
            pre = json.load(open(os.path.join(eval_dir, "manifest.json")))["preprocessing"][ev["preprocess"]]
            inp = os.path.join(eval_dir, pre["file"])
            x = np.fromfile(inp, np.float32).reshape(pre["shape"])
            ga, gb = bfformat.read(pub), bfformat.read(new)
            ya = np.stack([executor.run(ga, {"input": x[i:i + 1]})[ga.outputs[0]].reshape(-1) for i in range(len(x))])
            yb = np.stack([executor.run(gb, {"input": x[i:i + 1]})[gb.outputs[0]].reshape(-1) for i in range(len(x))])
            ca = run_bench(args.bench, pub, inp, os.path.join(d, "cpp_published.f32"))
            cb = run_bench(args.bench, new, inp, os.path.join(d, "cpp_reimported.f32"))
            oa = run_bench(args.bench, pub, inp, os.path.join(d, "cpp_published_optimized.f32"), "optimized")
            ob = run_bench(args.bench, new, inp, os.path.join(d, "cpp_reimported_optimized.f32"), "optimized")
            row["outputs_check_set"] = {"n_samples": int(len(x)), "eval_set": ev["dir"],
                                        "numpy_f32_bit_identical": bool(np.array_equal(ya.view(np.uint32), yb.view(np.uint32))),
                                        "cpp_reference_bit_identical": bool(np.array_equal(ca.view(np.uint32), cb.view(np.uint32))),
                                        "cpp_optimized_bit_identical": bool(np.array_equal(oa.view(np.uint32), ob.view(np.uint32))),
                                        "cpp_vs_numpy_reimported_bit_identical": bool(np.array_equal(cb.view(np.uint32),
                                                                                                     yb.reshape(-1).view(np.uint32)))}
            v = crep.get("verification", {})
            sem = v.get("semantic_check_ir_f64_vs_reference_f64", {})
            ec32 = v.get("source_framework_f32_vs_qonnx_f32 (export check)", {})
            ec64 = v.get("source_framework_f64_vs_qonnx_f64 (export check)", {})
            row["verification"] = {
                "acceptance": v.get("acceptance"),
                "semantic_f64_max_abs": sem.get("max_abs_error"), "semantic_f64_top1": sem.get("top1_agreement"),
                "samples_le_1e-6": sem.get("n_samples_max_abs_le_1e-6"),
                "binarization_flip_samples": v.get("binarization_flips_vs_source", {}).get("n_samples_with_flips"),
                "qonnx_f32_vs_f64_top1_disagreements": v.get("reference_f32_vs_f64", {}).get("n_top1_disagreements"),
                "numpy_f32_vs_qonnx_f32": {k: v.get("numpy_f32_vs_reference_f32", {}).get(k) for k in ("max_abs_error", "top1_agreement")},
                "export_check_source_f32_vs_qonnx_f32": {k: ec32.get(k) for k in ("max_abs_error", "top1_agreement", "n_top1_disagreements")},
                "export_check_source_f64_vs_qonnx_f64": {k: ec64.get(k) for k in ("max_abs_error", "top1_agreement", "n_top1_disagreements")},
                "cleanup_effect_raw_f64_vs_clean_f64_max_abs": v.get("cleanup_effect_raw_f64_vs_clean_f64", {}).get("max_abs_error"),
                # stage reference must equal the numpy executor bit for bit; the optimized stage is compared with the
                # published file's optimized outputs instead (outputs_check_set)
                "cpp_host_reference_exact_vs_numpy": [c.get("exact_vs_numpy") for c in crep.get("cpp_host", []) if c.get("stage") == "reference"],
            }
            row["import_rules"] = crep.get("import", {}).get("rules")
            o = row["outputs_check_set"]
            if row["data_identical"] and row["graph_identical"]:
                row["status"] = "IDENTICAL FILE (data + graph)"
            elif row["data_identical"] and row["graph_identical_up_to_names"] and o["numpy_f32_bit_identical"] and o["cpp_reference_bit_identical"] \
                    and o["cpp_optimized_bit_identical"]:
                row["status"] = "DATA IDENTICAL; GRAPH IDENTICAL UP TO NAMES; OUTPUTS BIT-IDENTICAL"
            elif o["numpy_f32_bit_identical"] and o["cpp_reference_bit_identical"] and o["cpp_optimized_bit_identical"]:
                row["status"] = "GRAPH/DATA DIFFER; OUTPUTS BIT-IDENTICAL"
            else:
                row["status"] = "FAIL"
            rows[key] = row
            print(f"{key:48s} data {'=' if row['data_identical'] else '!='} graph {'=' if row['graph_identical'] else '!='} "
                  f"(names-only {row['graph_identical_up_to_names']}) outputs numpy {o['numpy_f32_bit_identical']} "
                  f"cpp ref {o['cpp_reference_bit_identical']} opt {o['cpp_optimized_bit_identical']} -> {row['status']}", flush=True)
            json.dump({"schema": "bitino.qonnx_phase2.v1", "networks": rows}, open(args.out, "w"), indent=1)
    json.dump({"schema": "bitino.qonnx_phase2.v1", "networks": rows}, open(args.out, "w"), indent=1)
    return 0 if all(r.get("status", "FAIL") != "FAIL" for r in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
