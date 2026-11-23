#!/usr/bin/env python3
"""Step-1b acceptance of the QONNX route (docs/qonnx-step1b.md): the PyTorch networks step 1 left on their legacy
converters (BBCU, BSRC-2023, BNAS, BiMatting, AdaBin) exported to canonical QONNX files of the Bitino profile
(tools/export_torch_qonnx.py), checked with check-qonnx, imported (tools/import_qonnx.py) and compared with the
published files. Same flow and identity bar as tools/qonnx_step1_acceptance.py, plus image-to-image outputs.

Runs in .cache/venv-qonnx with python -I; the exports run in .cache/venv-qonnx2 (subprocess, python -I):

  scripts/capped.sh 4G nice -n 10 env OMP_NUM_THREADS=4 .cache/venv-qonnx/bin/python -I tools/qonnx_step1b_acceptance.py \
      [--only ID ...] [--variant ge|sign_zero] [--skip-run] [--no-verify] [--cache-dir DIR] [--artifacts-dir DIR]
      [--bench build/host/bitino_bench] [--out reports/qonnx_step1b/summary.json]

Per network and binarizer variant, into <cache>/qonnx_step1b/<id>/<variant>/ (never artifacts/):
  1. export (checkpoint SHA-256 check, canonical model.onnx, export.json, source / exportable outputs);
  2. tools/check_qonnx.py on model.onnx (no error and no warning required);
  3. tools/import_qonnx.py with export.json's import arguments; with verification: Level-3 data (qonnx execute_onnx
     fp32, OnnxF64 fp64 with the float64 carriers, first binarization flip); tools/validate_model.py with the recipe's
     validate_args (image models: --output-kind image) and --require-exact-numpy for the C++ reference stage;
  4. comparison with artifacts/<id>/<file>: data + graph SHA-256 (names included; manifest not compared), the recipe's
     expected hashes for model.bitino; outputs of both files on the check set (numpy executor fp32, C++ reference and
     optimized stages), bit for bit.

Level 3 (docs/numerical-correctness.md § Level 3), per sample, on the IR in float64 vs the QONNX graph in float64:
  classifiers   as step 1: deviation > 1e-6 -> first flip at |pre| <= 1e-3 x median |pre| of its binarizer, or no flip
                and max |delta logit| <= 1e-5 x max |logit|;
  images        the same rule per pixel with the image's output range R (1 or 255) as the scale: a sample with a
                pixel deviating by more than 1e-6 x R is accepted when its first binarization flip is at a rounding-
                scale pre-activation (as above), or it has no flip and every pixel is within 1e-5 x R. Reported: the
                number of deviating pixels, PSNR of the numpy fp32 output vs the fp64 reference (information).
Writes reports/qonnx_step1b/summary.json (merged with earlier runs).
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
from qonnx_step1_acceptance import FLIP_REL, NOFLIP_REL, VARIANTS, bits_equal, run_bench  # noqa: E402

IMG_DEV_REL = 1e-6        # deviation threshold x output range (the legacy S8 converters' rule)
CHUNK = {"bnas-h-imagenet": 4}   # exporter --chunk: source / exportable logits a few samples at a time (memory)
BOTH = ("ge", "sign_zero")
# the BNAS checkpoints were trained / evaluated with the first-release BinOp (commit cdb58f7, the recipe's source); the
# clone's head (675ecc7) selects other convs (adapters.torch_bnas.binop_targets)
BNAS_REPO = "bnas-s13/src-cdb58f7"
# model id -> (variants, repository clone under <cache>/repos, output kind)
NETWORKS = {
    "bbcu-denoise": (BOTH, "BBCU", "image"), "bbcu-deblocking": (BOTH, "BBCU", "image"),
    "bbcu-sr-l-x2": (BOTH, "BBCU", "image"), "bbcu-sr-l-x4": (BOTH, "BBCU", "image"),
    "bbcu-sr-m-x2": (BOTH, "BBCU", "image"), "bbcu-sr-m-x4": (BOTH, "BBCU", "image"),
    "bsrc2023-binary-sr-x2-v0": (("ge",), "BSRC-2023-62bbcf2", "image"),
    "bsrc2023-binary-sr-x2-v1": (("ge",), "BSRC-2023-62bbcf2", "image"),
    "bsrc2023-binary-sr-x4-v0": (("ge",), "BSRC-2023-62bbcf2", "image"),
    "bsrc2023-binary-sr-x4-v1": (("ge",), "BSRC-2023-62bbcf2", "image"),
    "bimatting-bimobilenet-backbone": (BOTH, "BiMatting", "logits"),
    "bnas-a-cifar10": (BOTH, BNAS_REPO, "logits"), "bnas-d-imagenet": (BOTH, BNAS_REPO, "logits"),
    "bnas-e-imagenet": (BOTH, BNAS_REPO, "logits"), "bnas-h-imagenet": (BOTH, BNAS_REPO, "logits"),
    "adabin-resnet20-cifar10": (BOTH, "Efficient-Computing", "logits"),
}


# output range of the image models (tools/convert_restoration.py FAMILIES[...]["output_range"]): the SR eval sets carry
# a [0, 1] (BBCU) and a [0, 255] (BSRC) preprocessing, so the range belongs to the family, not to the eval set
OUTPUT_RANGE = {"bbcu-": 1.0, "bsrc2023-": 255.0}


def output_range(model_id: str) -> float:
    return next(v for k, v in OUTPUT_RANGE.items() if model_id.startswith(k))


def level3(crep: dict, per: np.ndarray, scale: np.ndarray, dev_thr: np.ndarray, noflip_thr: np.ndarray) -> dict:
    """Level-3 decision per sample: per = max |IR fp64 - QONNX fp64|; deviating when per > dev_thr; accepted with a
    first flip at a rounding-scale |pre| (<= FLIP_REL x median) or without a flip when per <= noflip_thr."""
    v = crep.get("verification", {})
    fl = v.get("binarization_flips_vs_source", {})
    fls = fl.get("samples", [])
    complete = len(fls) == fl.get("n_samples_with_flips", 0)
    f64 = {q["sample"]: q["ir_f64_vs_source_f64"] for q in fls if "ir_f64_vs_source_f64" in q}
    dev = [int(i) for i in np.where(per > dev_thr)[0]]
    flip_rel = {i: f64[i]["max_abs_pre_source"] / max(f64[i]["typical_abs_pre_source"], 1e-30) for i in dev if i in f64}
    noflip_rel = {i: float(per[i] / max(scale[i], 1e-30)) for i in dev if i not in f64}
    noflip_ok = all(per[i] <= noflip_thr[i] for i in noflip_rel)
    out = {"samples_within_dev_threshold": int(np.sum(per <= dev_thr)), "deviating": len(dev),
           "with_flip": len(flip_rel), "max_rel_pre_at_first_flip": max(flip_rel.values(), default=None),
           "max_abs_pre_at_first_flip": max((f64[i]["max_abs_pre_source"] for i in flip_rel), default=None),
           "without_flip": len(noflip_rel), "max_rel_delta_without_flip": max(noflip_rel.values(), default=None),
           "max_abs_delta_without_flip": max((float(per[i]) for i in noflip_rel), default=None),
           "flip_list_complete": complete}
    out["passed"] = bool(complete and all(r_ <= FLIP_REL for r_ in flip_rel.values()) and noflip_ok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-run", action="store_true", help="only compare (exports / imports already in <cache>/qonnx_step1b)")
    ap.add_argument("--rejudge", action="store_true", help="with --skip-run: reuse the recorded output comparisons of the same "
                                                           "re-imported file, recompute Level 3 from the saved fp64 outputs")
    ap.add_argument("--only", action="append", help="model id (repeatable)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    ap.add_argument("--no-verify", action="store_true", help="identity only: import without the eval set (no Level-3 run)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"), help="published conversions (read only)")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build", "host", "bitino_bench"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_step1b", "summary.json"))
    ap.add_argument("--exporter", default=os.path.join(ROOT, "tools", "export_torch_qonnx.py"))
    ap.add_argument("--image-validate-python", default="/usr/bin/python3",
                    help="interpreter for tools/validate_model.py --output-kind image (needs OpenCV for SSIM, as the legacy "
                         "S8 validations; .cache/venv-qonnx has none)")
    args = ap.parse_args()
    cache = os.path.abspath(args.cache_dir)
    venv_e = os.path.join(cache, "venv-qonnx2", "bin", "python")
    venv_q = os.path.join(cache, "venv-qonnx", "bin", "python")
    recipes = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    models = {m["model_id"]: m for m in json.load(open(os.path.join(ROOT, "registry", "models.json")))["models"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    old = json.load(open(args.out)) if os.path.exists(args.out) else {"networks": {}}
    rows = old.get("networks", {})
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    failed = []

    def save():
        json.dump({"schema": "bitino.qonnx_step1b.v1", "networks": rows}, open(args.out, "w"), indent=1)

    for mid, (variants, repo_dir, kind) in NETWORKS.items():
        if args.only and mid not in args.only:
            continue
        rec = recipes[mid]
        ev = rec["eval"]
        eval_dir = os.path.join(cache, "data", ev["dir"])
        lc = models[mid]["checkpoint"]["local_cache_path"]
        weights = os.path.join(cache, lc[len(".cache/"):]) if lc.startswith(".cache/") else os.path.join(ROOT, lc)
        for var in variants:
            if args.variant and var not in args.variant:
                continue
            binarizer, pub_name = VARIANTS[var]
            key = f"{mid}:{var}"
            d = os.path.join(cache, "qonnx_step1b", mid, var)
            os.makedirs(d, exist_ok=True)
            row = {"model_id": mid, "variant": var, "published_file": pub_name, "eval_dir": ev["dir"], "preprocess": ev["preprocess"],
                   "output_kind": kind}
            t0 = time.time()
            if not args.skip_run:
                r = subprocess.run([venv_e, "-I", args.exporter, "--model-id", mid, "--repo", os.path.join(cache, "repos", repo_dir),
                                    "--weights", weights, "--out-dir", d, "--eval-dir", eval_dir, "--binarizer", binarizer] + (["--chunk", str(CHUNK[mid])] if mid in CHUNK else []), env=env)
                if r.returncode:
                    row.update({"status": "FAIL", "reason": f"export failed (rc {r.returncode})"})
                    rows[key] = row
                    failed.append(key)
                    save()
                    continue
            exp = json.load(open(os.path.join(d, "export.json")))
            row["export"] = {k: exp.get(k) for k in ("checkpoint_sha256", "exporter", "onnx_sha256", "binarizer", "import_args",
                                                     "source_top1", "source_exact_zeros", "source_f32_vs_f64_top1_disagreements",
                                                     "source_f32_vs_f64_max_abs", "exportable_vs_source", "canonical")}
            onnx_path = os.path.join(d, "model.onnx")
            prev_chk = (rows.get(key) or {}).get("check_qonnx")
            if args.rejudge and not os.path.exists(onnx_path) and prev_chk and \
                    (rows.get(key) or {}).get("export", {}).get("onnx_sha256") == exp.get("onnx_sha256"):
                # the canonical file was deleted after its row was recorded (disk): its recorded check of the same SHA-256
                row["check_qonnx"] = dict(prev_chk, note="recorded (--rejudge; canonical file removed after the run, SHA-256 in export)")
            else:
                chk = subprocess.run([venv_q, "-I", os.path.join(ROOT, "tools", "check_qonnx.py"), onnx_path, "--json", "-"],
                                     capture_output=True, text=True)
                cj = json.loads(chk.stdout)
                cj = cj[0] if isinstance(cj, list) else cj
                row["check_qonnx"] = {"returncode": chk.returncode, "passed": cj.get("passed"), "summary": cj.get("summary"),
                                      "warnings": sorted({f["code"] for f in cj.get("findings", []) if f.get("level") == "warning"}),
                                      "errors": sorted({f["code"] for f in cj.get("findings", []) if f.get("level") == "error"})}
            if not args.skip_run:
                cmd = [venv_q, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), onnx_path, "--model-id", mid, "--out-dir", d,
                       "--exporter", exp["exporter"], "--source-json", os.path.join(d, "export.json")] + list(exp["import_args"])
                if not args.no_verify:
                    cmd += ["--eval-dir", eval_dir, "--preprocess", ev["preprocess"], "--input-domain", "float32"]
                r = subprocess.run(cmd, env=env)
                row["import_returncode"] = r.returncode
            crep = json.load(open(os.path.join(d, "conversion_report.json")))
            if "import_error" in crep:
                row.update({"status": "FAIL", "reason": f"import error: {crep['import_error']}"})
                rows[key] = row
                failed.append(key)
                save()
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
            if "recipe_expected" in row:
                row["recipe_identical"] = row["reimported"]["data_sha256"] == row["recipe_expected"]["data_sha256"] and \
                    row["reimported"]["graph_sha256"] == row["recipe_expected"]["graph_sha256"]
            if not args.no_verify:
                pre = json.load(open(os.path.join(eval_dir, "manifest.json")))["preprocessing"][ev["preprocess"]]
                inp = os.path.join(eval_dir, pre["file"])
                x = np.fromfile(inp, np.float32).reshape(pre["shape"])
                n = len(x)
                if not args.skip_run:      # C++ reference stage vs the numpy executor (the recipe's validation arguments)
                    js = os.path.join(d, "validation_host_reference.json")
                    vpy = args.image_validate_python if kind == "image" else venv_q
                    vr = subprocess.run([vpy, "-I", os.path.join(ROOT, "tools", "validate_model.py"), "--artifacts", d,
                                         "--eval-dir", eval_dir, "--preprocess", ev["preprocess"], "--bench", args.bench,
                                         "--stage", "reference", "--threads", "1", "--label", "qonnx_host_reference", "--json", js,
                                         "--require-exact-numpy"] + list(ev.get("validate_args", [])), capture_output=True, text=True, env=env)
                    row["validate_model"] = {"python": vpy, "returncode": vr.returncode,
                                             "stderr_tail": vr.stderr.strip().splitlines()[-3:] if vr.returncode else []}
                vj = os.path.join(d, "validation_host_reference.json")
                vrep = json.load(open(vj)) if os.path.exists(vj) else {}
                prev = rows.get(key, {}) if args.rejudge else {}
                if prev.get("outputs_check_set") and prev.get("reimported") == row["reimported"]:
                    # --rejudge: the recorded bit-for-bit comparisons of the same file; the fp32 / fp64 IR outputs the
                    # importer's verification wrote (numpy executor of the same file)
                    row["outputs_check_set"] = dict(prev["outputs_check_set"], note="recorded (--rejudge)")
                    yb = np.fromfile(os.path.join(d, "numpy_logits_fused.f32"), np.float32).reshape(n, -1)
                    ir64 = np.fromfile(os.path.join(d, "numpy_logits_f64.f64"), np.float64).reshape(n, -1)
                else:
                    ga, gb = bfformat.read(pub), bfformat.read(new)
                    ya = np.stack([executor.run(ga, {"input": x[i:i + 1]})[ga.outputs[0]].reshape(-1) for i in range(n)])
                    yb = np.stack([executor.run(gb, {"input": x[i:i + 1]})[gb.outputs[0]].reshape(-1) for i in range(n)])
                    ca = run_bench(args.bench, pub, inp, os.path.join(d, "cpp_published.f32"))
                    cb = run_bench(args.bench, new, inp, os.path.join(d, "cpp_reimported.f32"))
                    oa = run_bench(args.bench, pub, inp, os.path.join(d, "cpp_published_optimized.f32"), "optimized")
                    ob = run_bench(args.bench, new, inp, os.path.join(d, "cpp_reimported_optimized.f32"), "optimized")
                    for f in ("cpp_published.f32", "cpp_reimported.f32", "cpp_published_optimized.f32", "cpp_reimported_optimized.f32"):
                        os.remove(os.path.join(d, f))       # disk: the comparison is recorded below
                    row["outputs_check_set"] = {"n_samples": int(n), "eval_set": ev["dir"],
                                                "numpy_f32_bit_identical": bits_equal(ya, yb),
                                                "cpp_reference_bit_identical": bits_equal(ca, cb),
                                                "cpp_optimized_bit_identical": bits_equal(oa, ob),
                                                "cpp_vs_numpy_reimported_bit_identical": bits_equal(cb, yb.reshape(-1))}
                    f64p = os.path.join(d, "numpy_logits_f64.f64")
                    if not args.skip_run and os.path.exists(f64p):
                        # the importer's verification of this run wrote the numpy executor's fp64 outputs of this file
                        ir64 = np.fromfile(f64p, np.float64).reshape(n, -1)
                    else:
                        ir64 = np.stack([executor.run(gb, {"input": x[i:i + 1]}, float_dtype=np.float64)[gb.outputs[0]].reshape(-1)
                                         for i in range(n)])
                v = crep.get("verification", {})
                ref64 = np.fromfile(os.path.join(d, "reference_logits_f64.f64"), np.float64).reshape(n, -1)
                per = np.abs(ir64 - ref64).max(axis=1)
                if kind == "image":
                    rng = output_range(mid)
                    scale = np.full(n, rng)
                    l3 = level3(crep, per, scale, IMG_DEV_REL * scale, NOFLIP_REL * scale)
                    pix = np.abs(ir64 - ref64)
                    l3.update({"rule": f"images: a pixel deviating by > {IMG_DEV_REL:g} x range ({rng:g}) needs a first flip at "
                                       f"|pre| <= {FLIP_REL:g} x median |pre| of its binarizer, or (no flip) |delta| <= "
                                       f"{NOFLIP_REL:g} x range", "output_range": rng,
                               "pixels": int(pix.size), "pixels_deviating": int(np.sum(pix > IMG_DEV_REL * rng)),
                               "pixels_deviating_in_samples_without_flip": int(sum(np.sum(pix[i] > IMG_DEV_REL * rng)
                                                                                  for i in np.where(per > IMG_DEV_REL * rng)[0]
                                                                                  if i not in {q["sample"] for q in v.get(
                                                                                      "binarization_flips_vs_source", {}).get("samples", [])
                                                                                      if "ir_f64_vs_source_f64" in q})),
                               "max_abs_ir_f64_vs_qonnx_f64": float(per.max())})
                    from bitino.validation.image_metrics import compare_images
                    yb32 = yb.astype(np.float64)
                    ci = compare_images(ref64, yb32, rng)
                    info = {"numpy_f32_vs_qonnx_f64_psnr_min_db": ci["psnr_vs_reference_min_db"],
                            "numpy_f32_vs_qonnx_f64_max_abs": ci["max_abs_error"],
                            "validate_model": {k: vrep.get(k) for k in ("acceptance",)} if vrep else None}
                else:
                    scale = np.abs(ref64).max(axis=1)
                    l3 = level3(crep, per, scale, np.full(n, 1e-6), NOFLIP_REL * scale)
                    l3["rule"] = (f"deviation > 1e-6: first flip at |pre| <= {FLIP_REL:g} x median |pre| of its binarizer, "
                                  f"or no flip and max |delta| <= {NOFLIP_REL:g} x max |logit|")
                    l4 = v.get("acceptance") or {}
                    info = {"level4_informational": {"criterion": l4.get("criterion"), "passed": l4.get("passed")},
                            "numpy_f32_vs_qonnx_f64": {k: v.get("numpy_f32_vs_reference_f64", {}).get(k)
                                                       for k in ("top1_agreement", "n_top1_disagreements")}}
                ec32 = v.get("source_framework_f32_vs_qonnx_f32 (export check)", {})
                ec64 = v.get("source_framework_f64_vs_qonnx_f64 (export check)", {})
                cpp_exact = (vrep.get("vs_numpy_executor_f32") or {}).get("exact_match")
                row["verification"] = {
                    "acceptance": {"rule": "Level 3 (docs/numerical-correctness.md) + C++ reference stage bit-exact with numpy",
                                   "passed": bool(l3["passed"]), "level3": l3},
                    "informational": info,
                    "binarization_flip_samples": v.get("binarization_flips_vs_source", {}).get("n_samples_with_flips"),
                    "export_check_source_f32_vs_qonnx_f32_max_abs": ec32.get("max_abs_error"),
                    "export_check_source_f64_vs_qonnx_f64_max_abs": ec64.get("max_abs_error"),
                    "export_check_source_f64_vs_qonnx_f64_top1": None if kind == "image" else ec64.get("top1_agreement"),
                    "cpp_host_reference_exact_vs_numpy": cpp_exact,
                }
            row["import_rules"] = crep.get("import", {}).get("rules")
            row["seconds"] = round(time.time() - t0, 1)
            ident = row["data_identical"] and row["graph_identical"] and row.get("recipe_identical", True)
            ok_chk = bool(row["check_qonnx"]["passed"]) and not row["check_qonnx"]["warnings"]
            if args.no_verify:
                row["status"] = "IDENTICAL (data + graph SHA-256, names included; manifest not compared), not verified" \
                    if ident and ok_chk else "FAIL"
            else:
                o = row["outputs_check_set"]
                ver = row["verification"]
                acc = bool(ver["acceptance"]["passed"]) and bool(ver["cpp_host_reference_exact_vs_numpy"])
                outs = o["numpy_f32_bit_identical"] and o["cpp_reference_bit_identical"] and o["cpp_optimized_bit_identical"]
                row["status"] = "PASS: data + graph SHA-256 identical (names included; manifest not compared), outputs " \
                                "bit-identical, Level 3 passed" if ident and ok_chk and outs and acc else "FAIL"
            if row["status"] == "FAIL":
                failed.append(key)
            rows[key] = row
            print(f"{key:52s} check {'ok' if ok_chk else 'FAIL'} data {'=' if row['data_identical'] else '!='} "
                  f"graph {'=' if row['graph_identical'] else '!='} -> {row['status']} ({row['seconds']} s)", flush=True)
            save()
    save()
    if failed:
        print("FAILED:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
