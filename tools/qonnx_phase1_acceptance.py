#!/usr/bin/env python3
"""Phase-1 acceptance of the QONNX route (docs/qonnx-frontend.md § Plan): re-import the 5 Brevitas and 3 FINN/QONNX
networks through QONNX and compare each re-imported model.bitino with the published conversion.

Runs in .cache/venv-qonnx with python -I; the Brevitas exports run in .cache/venv-brevitas (subprocess, python -I):

  scripts/capped.sh 8G nice -n 10 .cache/venv-qonnx/bin/python -I tools/qonnx_phase1_acceptance.py [--skip-run] [--only ID]

Per network, into .cache/qonnx_phase1/<id>/ (never artifacts/):
  1. source: Brevitas -> tools/export_brevitas_qonnx.py (model.onnx + source logits); FINN/QONNX -> the ONNX member
     read from the registry-verified zip (zip and member SHA-256 checked) -> source.onnx;
  2. tools/import_qonnx.py (qonnx cleanup, import, Level-3 verification, C++ host runtime exact stages);
  3. comparison with the published file artifacts/<id>/model.bitino and the recipe's expected hashes: data SHA-256,
     graph SHA-256 (fetch.graph_sha256: constants table, inputs, nodes, outputs), and, when the graph differs, the
     graph with positional names (constants c<i>, nodes n<i>, outputs n<i>:o<j>) and a list of the renamed items;
     outputs of both files on the model's eval128 check set: numpy executor (float32) and C++ runtime (stage
     reference), compared bit for bit.
Writes reports/qonnx_phase1/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.fetch import _header, data_sha256, graph_sha256  # noqa: E402

VENV_B = os.path.join(ROOT, ".cache", "venv-brevitas", "bin", "python")
VENV_Q = os.path.join(ROOT, ".cache", "venv-qonnx", "bin", "python")
BENCH = os.path.join(ROOT, "build", "host", "bitino_bench")
CK = os.path.join(ROOT, ".cache", "checkpoints")

# id -> source kind, source file (checkpoint / zip), zip member + its SHA-256 (tools/convert_finn_onnx.py), input domain
NETWORKS = {
    "brevitas-bnn-pynq-cnv-1w1a": ("brevitas", "brevitas-bnn-pynq-cnv-1w1a/cnv_1w1a-758c8fef.pth", None, None, "unit8"),
    "brevitas-bnn-pynq-lfc-1w1a": ("brevitas", "brevitas-bnn-pynq-lfc-1w1a/lfc_1w1a-318d8bd9.pth", None, None, "unit8"),
    "brevitas-bnn-pynq-sfc-1w1a": ("brevitas", "brevitas-bnn-pynq-sfc-1w1a/sfc_1w1a-fa4b2f7f.pth", None, None, "unit8"),
    "brevitas-bnn-pynq-tfc-1w1a": ("brevitas", "brevitas-bnn-pynq-tfc-1w1a/tfc_1w1a-45185b4d.pth", None, None, "unit8"),
    "aliasgharshinwari-finn-cnv-w1a1-cifar10": ("brevitas", "aliasgharshinwari-finn-cnv-w1a1-cifar10/cnv_w1a1_best.pt", None, None, "unit8"),
    "finn-examples-bnn-pynq-cnv-w1a1": ("finn", "finn-examples-bnn-pynq-onnx/onnx-models-bnn-pynq.zip", "cnv-w1a1.onnx",
                                        "47007fe69572a1c49b3d2be2607e21aaba92e2e7b8c29cb72254f22e2476c1d7", "uint8"),
    "finn-examples-bnn-pynq-tfc-w1a1": ("finn", "finn-examples-bnn-pynq-onnx/onnx-models-bnn-pynq.zip", "tfc-w1a1.onnx",
                                        "1706b590920df2a73dc5f256725292b9a8640f904e0e0345547722b607dea6a0", "uint8"),
    "finn-examples-cnv-w1a1-gtsrb": ("finn", "finn-examples-cnv-w1a1-gtsrb/onnx-models-gtsrb.zip", "onnx-models-gtsrb/cnv_1w1a_gtsrb.onnx",
                                     "050a1e6e3bf65dd67a6bd024805da41e16156302ac6bb51bdf2cb76b935a7f90", "unit8"),
}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(path: str) -> tuple[str, dict]:
    """Graph JSON with positional names, and the name map (published name -> positional name)."""
    h = _header(path)
    names = {}
    for i, t in enumerate(h["tensors"]):
        names[t["name"]] = f"c{i}"
    for i, n in enumerate(h["nodes"]):
        for j, o in enumerate(n["outputs"]):
            names[o] = f"n{i}:o{j}"
    for i, t in enumerate(h["inputs"]):
        names[t["name"]] = f"in{i}"
    doc = {"tensors": [{**t, "name": names[t["name"]]} for t in h["tensors"]],
           "inputs": [{**t, "name": names[t["name"]]} for t in h["inputs"]],
           "nodes": [{**n, "name": f"n{i}", "inputs": [names.get(x, x) if x else None for x in n["inputs"]],
                      "outputs": [names[o] for o in n["outputs"]]} for i, n in enumerate(h["nodes"])],
           "outputs": [names[o] for o in h["outputs"]]}
    js = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(js.encode()).hexdigest(), {"tensors": [t["name"] for t in h["tensors"]],
                                                    "nodes": [n["name"] for n in h["nodes"]]}


def run_bench(model: str, inputs: str, out: str) -> np.ndarray:
    subprocess.run([BENCH, "--model", model, "--inputs", inputs, "--outputs", out, "--stage", "reference", "--threads", "1",
                    "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
    return np.fromfile(out, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-run", action="store_true", help="only compare (re-imports already in .cache/qonnx_phase1)")
    ap.add_argument("--only", action="append")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_phase1", "summary.json"))
    args = ap.parse_args()
    recipes = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    old = json.load(open(args.out)) if os.path.exists(args.out) else {"networks": {}}
    rows = old.get("networks", {})
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for mid, (kind, src, member, member_sha, domain) in NETWORKS.items():
        if args.only and mid not in args.only:
            continue
        rec = recipes[mid]
        ev = rec["eval"]
        eval_dir = os.path.join(ROOT, ".cache", "data", ev["dir"])
        n_cls = int(ev["validate_args"][ev["validate_args"].index("--num-classes") + 1])
        d = os.path.join(ROOT, ".cache", "qonnx_phase1", mid)
        os.makedirs(d, exist_ok=True)
        row = {"kind": kind, "eval_dir": ev["dir"], "preprocess": ev["preprocess"], "input_domain": domain}
        src_path = os.path.join(CK, src)
        if kind == "brevitas":
            onnx_path = os.path.join(d, "model.onnx")
            if not args.skip_run:
                subprocess.run([VENV_B, "-I", os.path.join(ROOT, "tools", "export_brevitas_qonnx.py"), "--model-id", mid,
                                "--weights", src_path, "--out-dir", d, "--eval-dir", eval_dir], check=True, env=env)
            exp = json.load(open(os.path.join(d, "export.json")))
            row["source"] = {"checkpoint": src, "checkpoint_sha256": exp["checkpoint_sha256"], "exporter": exp["exporter"],
                             "onnx_sha256": exp["onnx_sha256"], "onnxoptimizer_shim": bool(exp.get("onnxoptimizer_shim"))}
            extra = ["--source-json", os.path.join(d, "export.json"), "--exporter", exp["exporter"]]
        else:
            raw = open(src_path, "rb").read()
            zsha = sha256(raw)
            want = next(x["sha256"] for x in rec["downloads"] if x["file"] == os.path.basename(src))
            if zsha != want:
                raise SystemExit(f"{mid}: zip SHA-256 {zsha} != recipe {want}")
            mb = zipfile.ZipFile(io.BytesIO(raw)).read(member)
            if sha256(mb) != member_sha:
                raise SystemExit(f"{mid}: member {member} SHA-256 mismatch")
            onnx_path = os.path.join(d, "source.onnx")
            open(onnx_path, "wb").write(mb)
            row["source"] = {"zip": src, "zip_sha256": zsha, "member": member, "onnx_sha256": member_sha}
            extra = ["--exporter", "finn-examples release file (FINN-ONNX / Brevitas QONNX export, as published)"]
        if not args.skip_run:
            r = subprocess.run([VENV_Q, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), onnx_path, "--model-id", mid,
                                "--out-dir", d, "--eval-dir", eval_dir, "--preprocess", ev["preprocess"], "--input-domain", domain,
                                "--bench", BENCH, "--num-classes", str(n_cls)] + extra, env=env)
            row["import_returncode"] = r.returncode
        crep = json.load(open(os.path.join(d, "conversion_report.json")))
        if "import_error" in crep:
            row.update({"status": "FAIL", "reason": f"import error: {crep['import_error']}"})
            rows[mid] = row
            continue
        new = os.path.join(d, "model.bitino")
        pub = os.path.join(ROOT, "artifacts", mid, "model.bitino")
        exp_h = rec["expected"]["model.bitino"]
        row["expected"] = {"data_sha256": exp_h["data_sha256"], "graph_sha256": exp_h["graph_sha256"]}
        row["published_file"] = {"data_sha256": data_sha256(pub), "graph_sha256": graph_sha256(pub)}
        row["reimported"] = {"data_sha256": data_sha256(new), "graph_sha256": graph_sha256(new), "bytes": os.path.getsize(new)}
        row["data_identical"] = row["reimported"]["data_sha256"] == exp_h["data_sha256"]
        row["graph_identical"] = row["reimported"]["graph_sha256"] == exp_h["graph_sha256"]
        row["published_matches_recipe"] = row["published_file"] == row["expected"]
        cn, nn = canonical(new)
        cp, npub = canonical(pub)
        row["graph_identical_up_to_names"] = cn == cp
        if not row["graph_identical"]:
            ren_t = [[a, b] for a, b in zip(npub["tensors"], nn["tensors"]) if a != b]
            ren_n = [[a, b] for a, b in zip(npub["nodes"], nn["nodes"]) if a != b]
            row["renamed"] = {"constants": len(ren_t), "nodes": len(ren_n), "examples": (ren_n[:4] + ren_t[:4])}
        # outputs on the check set: numpy executor (float32) + C++ runtime (stage reference), bit for bit
        pre = json.load(open(os.path.join(eval_dir, "manifest.json")))["preprocessing"][ev["preprocess"]]
        inp = os.path.join(eval_dir, pre["file"])
        x = np.fromfile(inp, np.float32).reshape(pre["shape"])
        ga, gb = bfformat.read(pub), bfformat.read(new)
        ya = np.stack([executor.run(ga, {"input": x[i:i + 1]})[ga.outputs[0]].reshape(-1) for i in range(len(x))])
        yb = np.stack([executor.run(gb, {"input": x[i:i + 1]})[gb.outputs[0]].reshape(-1) for i in range(len(x))])
        ca = run_bench(pub, inp, os.path.join(d, "cpp_published.f32"))
        cb = run_bench(new, inp, os.path.join(d, "cpp_reimported.f32"))
        row["outputs_check_set"] = {"n_samples": int(len(x)), "eval_set": ev["dir"],
                                    "numpy_f32_bit_identical": bool(np.array_equal(ya.view(np.uint32), yb.view(np.uint32))),
                                    "cpp_reference_bit_identical": bool(np.array_equal(ca.view(np.uint32), cb.view(np.uint32))),
                                    "cpp_vs_numpy_reimported_bit_identical": bool(np.array_equal(cb.view(np.uint32), yb.reshape(-1).view(np.uint32)))}
        v = crep.get("verification", {})
        row["verification"] = {
            "acceptance": v.get("acceptance"),
            "semantic_f64_max_abs": v.get("semantic_check_ir_f64_vs_reference_f64", {}).get("max_abs_error"),
            "semantic_f64_top1": v.get("semantic_check_ir_f64_vs_reference_f64", {}).get("top1_agreement"),
            "samples_le_1e-6": v.get("semantic_check_ir_f64_vs_reference_f64", {}).get("n_samples_max_abs_le_1e-6"),
            "binarization_flip_samples": v.get("binarization_flips_vs_source", {}).get("n_samples_with_flips"),
            "source_f32_vs_f64_top1_disagreements": v.get("reference_f32_vs_f64", {}).get("n_top1_disagreements"),
            "numpy_f32_vs_qonnx_f32_max_abs": v.get("numpy_f32_vs_reference_f32", {}).get("max_abs_error"),
            "export_check_source_f32_vs_qonnx_f32": {k: v.get("source_framework_f32_vs_qonnx_f32 (export check)", {}).get(k)
                                                    for k in ("max_abs_error", "top1_agreement", "exact_match")}
            if "source_framework_f32_vs_qonnx_f32 (export check)" in v else None,
            "export_check_source_f64_vs_qonnx_f64": {k: v.get("source_framework_f64_vs_qonnx_f64 (export check)", {}).get(k)
                                                    for k in ("max_abs_error", "top1_agreement")}
            if "source_framework_f64_vs_qonnx_f64 (export check)" in v else None,
            "cleanup_effect_raw_f64_vs_clean_f64_max_abs": v.get("cleanup_effect_raw_f64_vs_clean_f64", {}).get("max_abs_error"),
            "cpp_host": [{k: c.get(k) for k in ("stage", "exact_vs_numpy")} for c in crep.get("cpp_host", [])],
        }
        o = row["outputs_check_set"]
        if row["data_identical"] and row["graph_identical"]:
            row["status"] = "IDENTICAL FILE (data + graph)"
        elif row["data_identical"] and row["graph_identical_up_to_names"] and o["numpy_f32_bit_identical"] and o["cpp_reference_bit_identical"]:
            row["status"] = "DATA IDENTICAL; GRAPH IDENTICAL UP TO NAMES; OUTPUTS BIT-IDENTICAL"
        elif o["numpy_f32_bit_identical"] and o["cpp_reference_bit_identical"]:
            row["status"] = "GRAPH/DATA DIFFER; OUTPUTS BIT-IDENTICAL"
        else:
            row["status"] = "FAIL"
        rows[mid] = row
        print(f"{mid:42s} data {'=' if row['data_identical'] else '!='} graph {'=' if row['graph_identical'] else '!='} "
              f"(names-only {row['graph_identical_up_to_names']}) outputs numpy {o['numpy_f32_bit_identical']} cpp {o['cpp_reference_bit_identical']} "
              f"-> {row['status']}")
    json.dump({"schema": "bitino.qonnx_phase1.v1", "networks": rows}, open(args.out, "w"), indent=1)
    return 0 if all(r.get("status", "FAIL") != "FAIL" for r in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
