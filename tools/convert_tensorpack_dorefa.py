#!/usr/bin/env python3
"""Converter for the tensorpack DoReFa-Net AlexNet W1A1 (``AlexNet-1,1,32.npz``, slice S12).

  python3 -I tools/convert_tensorpack_dorefa.py --model-id tensorpack-dorefa-alexnet-w1a1-g32 \
      --npz ".cache/checkpoints/tensorpack-dorefa-alexnet-w1a1-g32/AlexNet-1,1,32.npz" --repo .cache/repos/tensorpack \
      --eval-dir .cache/data/eval128_tensorpack_bgr --tensorpack-logits artifacts/<id>/tensorpack_logits_eval128.f32 \
      --big-eval-dir .cache/data/eval5000_tensorpack_bgr --tensorpack-logits-big artifacts/<id>/tensorpack_logits_eval5000.f32 \
      --out-dir artifacts/tensorpack-dorefa-alexnet-w1a1-g32

References: reference_logits.f32 = the official tensorpack graph in TF float32 (tools/tensorpack_dorefa_reference.py,
run in a TF env); reference_logits_f64.f64 = the cited PyTorch port tools/dorefa_alexnet_reference.py in float64 (E in
float64, as the IR's exact constants); the port in float32 is compared with the official output. Outputs model.bitino /
model_unfused.bitino (binarization ``gt`` = the exact source rule u = [x > 0.5]; no sign_zero variant),
numpy_logits_{fused,unfused}.f32, conversion_report.json (Level 3 with float64 constants and the first-flip analysis
of every sample deviating by more than 1e-6) and, with --big-eval-dir, reference_logits_port_eval<N>.f32.
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import platform
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import tensorpack_dorefa  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402
from convert_torch_cnn import repo_commit, sha256_file  # noqa: E402
import dorefa_alexnet_reference as port  # noqa: E402

PREPROCESS = "tensorpack_bgr"
LAYERS = ["conv1", "conv2", "conv3", "conv4", "fc0", "fc1"]
_G: dict = {}
_EXACT: dict = {}


def _init_worker(paths: dict) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
        if k in _EXACT:
            gx = bfformat.read(p)
            for name, v in _EXACT[k].items():
                assert np.array_equal(v.astype(np.float32), gx.constants[name].data.reshape(v.shape)), name
                gx.constants[name].data = v.reshape(gx.constants[name].data.shape)
            _G[k + "#exact"] = gx


def _run_one(job):
    key, x = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    bins = [n for n in g.nodes if n.op in ("BConv2D", "BDense")]
    ties = sum(int(np.sum(env[n.inputs[0]] == np.float32(0.5))) for n in bins)
    env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    pre64 = {n.name: np.asarray(env64[n.inputs[0]]) - 0.5 for n in bins}
    outx = executor.run(_G[key + "#exact"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1) if key + "#exact" in _G else None
    return env[g.outputs[0]].reshape(-1), env64[g.outputs[0]].reshape(-1), ties, outx, pre64


def port_logits(model, x_nhwc: np.ndarray, dtype, batch: int = 32) -> np.ndarray:
    import torch
    model.to_dtype(dtype)
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), batch):
            out.append(model(torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch].transpose(0, 3, 1, 2))).to(dtype)).numpy())
    model.to_dtype(torch.float32)
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=["tensorpack-dorefa-alexnet-w1a1-g32"])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--repo", required=True, help="clone of tensorpack (examples/DoReFa-Net)")
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--tensorpack-logits", required=True, help="official TF fp32 logits on --eval-dir")
    ap.add_argument("--big-eval-dir")
    ap.add_argument("--tensorpack-logits-big")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=4)
    args = ap.parse_args()

    reg = json.load(open(args.registry))
    entry = next(m for m in reg["models"] if m.get("model_id") == args.model_id)
    actual = sha256_file(args.npz)
    if actual != entry["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch: expected {entry['checkpoint']['sha256']}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}")
    os.makedirs(args.out_dir, exist_ok=True)
    import torch
    torch.set_num_threads(args.torch_threads)
    params = port.load(args.npz)
    model = port.DoReFaAlexNet(params)
    ex = os.path.join(args.repo, "examples", "DoReFa-Net")
    src = {f"examples/DoReFa-Net/{f}": sha256_file(os.path.join(ex, f)) for f in ("alexnet-dorefa.py", "dorefa.py", "imagenet_utils.py")}
    src["tools/dorefa_alexnet_reference.py (port)"] = sha256_file(port.__file__)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])
    tp32 = np.fromfile(args.tensorpack_logits, np.float32).reshape(len(x), -1)
    tp_info = json.load(open(args.tensorpack_logits + ".json")) if os.path.exists(args.tensorpack_logits + ".json") else {}
    p32 = port_logits(model, x, torch.float32)
    p64 = port_logits(model, x, torch.float64)
    tp32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    p64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    p32.astype(np.float32).tofile(os.path.join(args.out_dir, "reference_logits_port_f32.f32"))
    top5 = lambda z, y: float(np.mean([y[i] in np.argsort(-z[i])[:5] for i in range(len(y))]))  # noqa: E731
    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_preprocessing": PREPROCESS,
              "reference": {"framework_f32": f"official tensorpack graph: TF {tp_info.get('tensorflow')} + tensorpack {tp_info.get('tensorpack')} (CPU)",
                            "framework_f64": f"PyTorch port tools/dorefa_alexnet_reference.py, torch {torch.__version__} (CPU)",
                            "repo_commit": repo_commit(args.repo), "source_sha256": src, "official_run": tp_info,
                            "top1_accuracy": float(np.mean(tp32.argmax(1) == labels)), "top5_accuracy": top5(tp32, labels),
                            "top1_accuracy_port_f32": float(np.mean(p32.argmax(1) == labels)),
                            "top1_accuracy_f64": float(np.mean(p64.argmax(1) == labels)), "top5_accuracy_f64": top5(p64, labels),
                            "official_f32_vs_port_f32": compare_outputs(tp32, p32, labels),
                            "latent_weight_exact_zeros": {k: int(np.sum(v == 0)) for k, v in params.items() if k.endswith("W:0")}},
              "reference_f32_vs_f64": compare_outputs(p64, tp32, labels), "port_f32_vs_f64": compare_outputs(p64, p32, labels), "variants": {}}
    print(json.dumps({k: (v if not isinstance(v, dict) or len(v) < 8 else {kk: v[kk] for kk in list(v)[:8]})
                      for k, v in report["reference"].items() if k != "source_sha256"}, indent=1, default=str))

    files = {}
    for variant, fold in (("unfused", False), ("fused", True)):
        graph, info = tensorpack_dorefa.convert(params, args.model_id)
        applied = passes.optimize(graph, fold=fold)
        graph.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": f"{variant}, input_binarize=gt (threshold 0.5)",
            "source": {"framework": "tensorpack (TF1) npz, read with numpy", "class": "examples/DoReFa-Net/alexnet-dorefa.py Model (--dorefa 1,1,32)",
                       "repo": "https://github.com/tensorpack/tensorpack", "repo_commit": report["reference"]["repo_commit"], "source_sha256": src,
                       "checkpoint_file": os.path.basename(args.npz), "checkpoint_sha256": actual},
            "conversion": {"tool": "tools/convert_tensorpack_dorefa.py", "adapter": tensorpack_dorefa.__name__, "passes": applied,
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": platform.python_version(),
                           "input_binarize": "gt", "binarize_note": "exact: DoReFa round(clip(x,0,1)) = 1 iff x > 0.5 (half to even)"},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32", "preprocessing": pre["definition"]},
            "outputs": ["logits"], "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"], "weight_diagnostics": info["weight_diagnostics"], "unused_in_forward": []}
        if not fold:
            _EXACT[variant] = info["float64_constants"]
        path = os.path.join(args.out_dir, "model.bitino" if fold else "model_unfused.bitino")
        file_sha = bfformat.write(graph, path)
        files[variant] = path
        report["variants"][variant] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": "gt", "op_counts": graph.op_counts(), "passes": applied}
        print(f"[{variant}] wrote {path}: {graph.op_counts()} passes={applied}")

    jobs = [(k, x[i:i + 1]) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
        results = pool.map(_run_one, jobs, chunksize=2)
    by_key = {k: results[j * len(x):(j + 1) * len(x)] for j, k in enumerate(files)}
    np_logits = {}
    for k, res in by_key.items():
        out32 = np.stack([r[0] for r in res]).astype(np.float32)
        out64 = np.stack([r[1] for r in res])
        np_logits[k] = out32
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        sem = compare_outputs(p64, out64, labels)
        per = np.abs(out64 - p64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per <= 1e-6))
        sem["n_samples_max_abs_le_1e-4"] = int(np.sum(per <= 1e-4))
        sem["per_sample_max_abs"] = [float(v) for v in per]
        if res[0][3] is not None:
            outx = np.stack([r[3] for r in res])
            px = np.abs(outx - p64).max(axis=1)
            sem["float64_constants"] = {"max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-6": int(np.sum(px <= 1e-6)),
                                        "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                                        "top1_agreement": float(np.mean(outx.argmax(1) == p64.argmax(1)))}
        if k == "unfused":
            ff = []
            for i in np.where(per > 1e-6)[0]:
                cap: dict = {}
                model.to_dtype(torch.float64)
                with torch.no_grad():
                    model(torch.from_numpy(np.ascontiguousarray(x[i:i + 1].transpose(0, 3, 1, 2))).double(), capture=cap)
                model.to_dtype(torch.float32)
                first = None
                for name in LAYERS:
                    a = cap[name] - 0.5
                    b = res[i][4][name].reshape(a.shape)
                    flips = (a > 0) != (b > 0)
                    if flips.any():
                        typ = float(np.median(np.abs(a)))
                        mx = float(np.abs(a[flips]).max())
                        first = {"layer": name, "n_flips": int(flips.sum()), "max_abs_pre_at_flip": mx, "typical_abs_pre": typ,
                                 "relative_to_typical": mx / typ}
                        break
                ff.append({"sample": int(i), "max_abs_logit_diff": float(per[i]), "top1_changed": bool(out64[i].argmax() != p64[i].argmax()),
                           "first_flip": first})
            wf = [d for d in ff if d["first_flip"]]
            sem["first_flip_analysis"] = {
                "pre": "x - 0.5 at the {0,1} quantizer (flip = (x > 0.5) differs)", "n_analyzed": len(ff),
                "n_without_flip": len(ff) - len(wf), "max_abs_logit_diff_without_flip": max([d["max_abs_logit_diff"] for d in ff if not d["first_flip"]], default=0.0),
                "n_with_flip": len(wf), "max_abs_pre_at_first_flip": max([d["first_flip"]["max_abs_pre_at_flip"] for d in wf], default=None),
                "max_relative_to_typical": max([d["first_flip"]["relative_to_typical"] for d in wf], default=None),
                "first_flip_layers": sorted({d["first_flip"]["layer"] for d in wf}), "n_top1_changes": int(sum(d["top1_changed"] for d in ff)),
                "samples": ff}
        report["variants"][k].update({"numpy_vs_reference_f32": compare_outputs(tp32, out32, labels),
                                      "numpy_vs_port_f32": compare_outputs(p32, out32, labels),
                                      "numpy_vs_reference_f64": compare_outputs(p64, out32, labels),
                                      "semantic_check_ir_f64_vs_reference_f64": sem,
                                      "exact_ties_x_eq_0.5_numpy_f32": int(sum(r[2] for r in res))})
        print(f"[{k}] vs official f32: top1 agree {report['variants'][k]['numpy_vs_reference_f32']['top1_agreement']:.4f} | SEMANTIC f64: max_abs "
              f"{sem['max_abs_error']:.3e} <=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{len(x)} top1 {sem['top1_agreement']:.4f}"
              + (f" | f64 consts max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else "")
              + (f" | first-flip: {sem['first_flip_analysis']['n_with_flip']} with flip, max rel {sem['first_flip_analysis']['max_relative_to_typical']}"
                 if "first_flip_analysis" in sem else ""))
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(np_logits["unfused"], np_logits["fused"], labels)

    if args.big_eval_dir:
        bm = json.load(open(os.path.join(args.big_eval_dir, "manifest.json")))
        bp = bm["preprocessing"][PREPROCESS]
        xb = np.fromfile(os.path.join(args.big_eval_dir, bp["file"]), np.float32).reshape(bp["shape"])
        yb = np.array([s["label"] for s in bm["samples"]][: len(xb)])
        lb = port_logits(model, xb, torch.float32, batch=50)
        lb.astype(np.float32).tofile(os.path.join(args.out_dir, f"reference_logits_port_eval{len(xb)}.f32"))
        big = {"eval_set": bm["name"], "n": int(len(xb)), "port_fp32_top1": float(np.mean(lb.argmax(1) == yb)), "port_fp32_top5": top5(lb, yb)}
        if args.tensorpack_logits_big:
            tb = np.fromfile(args.tensorpack_logits_big, np.float32).reshape(lb.shape)
            big.update({"official_fp32_top1": float(np.mean(tb.argmax(1) == yb)), "official_fp32_top5": top5(tb, yb),
                        "official_vs_port_fp32": compare_outputs(tb, lb, yb)})
        report["large_eval_source"] = big
        print("large eval:", json.dumps({k: v for k, v in big.items() if k != "official_vs_port_fp32"}))
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
