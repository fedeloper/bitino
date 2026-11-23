#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert the original BNN-PYNQ Theano/Lasagne checkpoints (cifar10-1w-1a, gtsrb-1w-1a, mnist-1w-1a .npz) to
.bitino and validate them against the official Theano/Lasagne network (slice S7).

  python3 -I tools/convert_bnn_pynq_npz.py --model-id bnn-pynq-cnv-w1a1-cifar10 \
      --npz .cache/checkpoints/bnn-pynq-cnv-w1a1-cifar10/cifar10-1w-1a.npz \
      --eval-dir .cache/data/eval128_cifar10_finn --out-dir artifacts/bnn-pynq-cnv-w1a1-cifar10

The trusted reference is tools/bnn_pynq_theano_reference.py (Python 2.7, Theano 1.0.4, Lasagne 0.2.dev1, the
official BNN-PYNQ cnv.py / lfc.py), run three times as a subprocess: float32 (as trained), float64 with the
float32-derived binary weights (primary float64 reference, reference_logits_f64.f64) and float64 with the
latent weights (reference_logits_f64_naive.f64; Theano's float64 weight binarization differs for |W| < ~6e-8,
see bitino.adapters.bnn_pynq_npz).

Outputs: model.bitino (fused: BN+sign -> integer thresholds; pool-before-BN thresholds in the next conv),
model_unfused.bitino, reference logits, numpy_logits_{fused,unfused}.f32, conversion_report.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import bnn_pynq_npz, finn_bnn  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

MODELS = {
    "bnn-pynq-cnv-w1a1-cifar10": dict(arch="cnv", n_out=10, pre="cifar10_bnnpynq",
                                      claim="BNN-PYNQ training README: CNV W1A1 test error 20.46% (79.54% top-1, CIFAR-10 test)", claim_top1=0.7954),
    "bnn-pynq-cnv-w1a1-gtsrb": dict(arch="cnv", n_out=44, pre="gtsrb_bnnpynq",
                                    claim="BNN-PYNQ training README: test error 3.37% (own random split of GTSRB Final_Training with random rotations/crops + junk class)",
                                    claim_top1=0.9663),
    "bnn-pynq-lfc-w1a1-mnist": dict(arch="lfc", n_out=10, pre="mnist_bnnpynq",
                                    claim="BNN-PYNQ training README: LFC W1A1 test error 1.65% (98.35% top-1, MNIST test)", claim_top1=0.9835),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def theano_run(args, cfg, x_file, shape, floatx, out, binary_weights=None):
    env = dict(os.environ)
    env["THEANO_FLAGS"] = f"base_compiledir={args.theano_compiledir},blas.ldflags="
    cmd = [args.theano_python, "-E", "-s", os.path.join(ROOT, "tools", "bnn_pynq_theano_reference.py"), "--arch", cfg["arch"],
           "--num-outputs", str(cfg["n_out"]), "--npz", os.path.abspath(args.npz), "--inputs", os.path.abspath(x_file),
           "--shape", ",".join(str(s) for s in shape), "--floatx", floatx, "--lasagne-dir", os.path.abspath(args.lasagne_dir),
           "--bnnpynq-dir", os.path.abspath(args.bnnpynq_dir), "--out", out]
    if binary_weights:
        cmd += ["--binary-weights", binary_weights]
    r = subprocess.run(cmd, env=env, cwd=args.theano_compiledir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Theano reference failed:\n{r.stdout}\n{r.stderr[-4000:]}")
    return dict(np.load(out, allow_pickle=False))


def src_act(pre: np.ndarray, dt) -> np.ndarray:
    """FixedHardTanH(QuantizationBinary) decision in the given precision: +1 iff round3(clip((clip(x)+1)/2)) = 1."""
    x = np.clip(np.asarray(pre, dt), dt(-1), dt(1))
    h = np.clip((x + dt(1)) / dt(2), dt(0), dt(1)).astype(dt)
    return np.where(h >= dt(0.5), 1, -1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--npz", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--theano-python", default=os.path.join(ROOT, ".cache", "env-theano27", "bin", "python"))
    ap.add_argument("--lasagne-dir", default=os.path.join(ROOT, ".cache", "downloads-s7", "lasagne", "Lasagne-master"))
    ap.add_argument("--bnnpynq-dir", default=os.path.join(ROOT, ".cache", "repos", "bnn-pynq", "bnn", "src", "training"))
    ap.add_argument("--theano-compiledir", default=os.path.join(tempfile.gettempdir(), "bitino-theanoc"),
                    help="Theano compile cache (a path without spaces)")
    args = ap.parse_args()
    cfg = MODELS[args.model_id]
    os.makedirs(args.theano_compiledir, exist_ok=True)

    reg = json.load(open(args.registry))
    entry = next(m for m in reg["models"] if m.get("model_id") == args.model_id)
    actual = sha256_file(args.npz)
    if actual != entry["checkpoint"]["sha256"]:
        print(f"ERROR: npz SHA-256 {actual} != registry {entry['checkpoint']['sha256']}", file=sys.stderr)
        return 3
    print(f"npz SHA-256 OK: {actual}")
    os.makedirs(args.out_dir, exist_ok=True)
    with np.load(args.npz, allow_pickle=False) as f:
        arrs = [np.asarray(f[f"arr_{i}"]) for i in range(len(f.files))]

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][cfg["pre"]]
    x_file = os.path.join(args.eval_dir, pre["file"])
    x = np.fromfile(x_file, np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])
    n = len(x)

    tmp = tempfile.mkdtemp(prefix="bnnpynq_ref_")
    d32 = theano_run(args, cfg, x_file, pre["shape"], "float32", os.path.join(tmp, "f32.npz"))
    d64 = theano_run(args, cfg, x_file, pre["shape"], "float64", os.path.join(tmp, "f64.npz"), binary_weights=os.path.join(tmp, "f32.npz"))
    d64n = theano_run(args, cfg, x_file, pre["shape"], "float64", os.path.join(tmp, "f64n.npz"))
    ref32 = d32["logits"].astype(np.float32)
    ref64 = d64["logits"].astype(np.float64)
    ref64n = d64n["logits"].astype(np.float64)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    ref64n.tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))
    n_pts = len([k for k in d32 if k.startswith("pre_")])
    pre32 = [d32[f"pre_{k}"].reshape(n, -1).astype(np.float64) for k in range(n_pts)]
    pre64 = [d64[f"pre_{k}"].reshape(n, -1) for k in range(n_pts)]
    wq32 = [d32[f"wq_{k}"] for k in range(len([k for k in d32 if k.startswith("wq_")]))]
    wq64n = [d64n[f"wq_{k}"] for k in range(len(wq32))]

    inp, layers, rinfo = bnn_pynq_npz.read(arrs, wq32, cfg["arch"], cfg["n_out"])
    if "qin" in d32:   # Theano's own int8 input levels vs the IR FakeQuant on the same inputs
        fq = np.clip(np.rint(x.astype(np.float64) * 128), -128, 127) / 128
        rinfo["input_quantizer_check"]["theano_qin_equals_ir_fakequant_on_eval_inputs"] = bool(
            np.array_equal(d32["qin"].transpose(0, 2, 3, 1).astype(np.float64), fq))
    rinfo["binary_weights_f64_latent_vs_f32"] = [int(np.sum(a != b)) for a, b in zip(wq32, wq64n)]

    def ties(prem, dt):
        out = {}
        for k, v in enumerate(prem):
            out[f"pre_{k}"] = {"exact_zeros": int(np.sum(v == 0)), "min_abs": float(np.abs(v).min()), "n_values": int(v.size),
                               "n_in_rounding_band_below_0": int(np.sum((v < 0) & (src_act(v, dt) > 0)))}
        return out

    report = {"model_id": args.model_id, "n_samples": n, "eval_set": em["name"], "eval_preprocessing": cfg["pre"],
              "source": {"npz": os.path.basename(args.npz), "sha256": actual, "n_arrays": len(arrs),
                         "upstream_hash": entry["checkpoint"].get("upstream_hash_published")},
              "reference": {"framework": " / ".join(str(v) for v in d32["versions"].tolist()),
                            "code": "Xilinx/BNN-PYNQ bnn/src/training/{cnv,lfc,quantization,quantized_net}.py (official, unmodified), "
                                    "lasagne.layers.get_output(net, deterministic=True)",
                            "lasagne_layers": [str(v) for v in d32["layer_names"].tolist()],
                            "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)),
                            "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                            "top1_accuracy_f64_naive_latent_weights": float(np.mean(ref64n.argmax(1) == labels)),
                            "claim": cfg["claim"],
                            "binarization_inputs_f32": ties(pre32, np.float32), "binarization_inputs_f64": ties(pre64, np.float64)},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "reference_f64_vs_f64_naive": compare_outputs(ref64, ref64n, labels),
              "reader": rinfo, "variants": {},
              "tied_top1_logits_f32": [int(i) for i in range(n) if np.sum(ref32[i] == ref32[i].max()) > 1]}
    print(json.dumps({k: report["reference"][k] for k in ("top1_accuracy", "top1_accuracy_f64", "top1_accuracy_f64_naive_latent_weights")}),
          "| weights f64-naive != f32:", rinfo["binary_weights_f64_latent_vs_f32"], "| vs finnthesizer ge0:", rinfo["weights_vs_finnthesizer_ge0"])

    for variant, fused in (("unfused", False), ("fused", True)):
        graph, info = finn_bnn.emit(inp, layers, fused=fused)
        applied = passes.optimize(graph, fold=False)
        graph.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": variant,
            "source": {"framework": "theano+lasagne (BNN-PYNQ)", "repo": "https://github.com/Xilinx/BNN-PYNQ/tree/master/bnn/src/training",
                       "checkpoint_file": os.path.basename(args.npz), "checkpoint_sha256": actual},
            "conversion": {"tool": "tools/convert_bnn_pynq_npz.py", "adapter": "bitino.adapters.bnn_pynq_npz + finn_bnn",
                           "passes": {"bn_sign_to_integer_threshold (adapter)": fused, **applied},
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "numpy": np.__version__},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32", "preprocessing": pre["definition"]},
            "outputs": ["logits"], "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"], "threshold_margins": info["threshold_margins"],
        }
        path = os.path.join(args.out_dir, "model.bitino" if fused else "model_unfused.bitino")
        file_sha = bfformat.write(graph, path)
        g = bfformat.read(path)
        pts = info["binarization_points"]
        assert len(pts) == n_pts, (len(pts), n_pts)
        g64c = bfformat.read(path)
        for l in layers:
            if l.final:
                names = (f"{l.name}/post_mul", f"{l.name}/post_add") if fused else (f"{l.name}/affine_scale", f"{l.name}/affine_shift")
                g64c.constants[names[0]].data = np.asarray(l.scale, np.float64)
                g64c.constants[names[1]].data = np.asarray(l.shift, np.float64)
            elif not fused:
                g64c.constants[f"{l.name}/bn_scale"].data = np.asarray(l.scale, np.float64)
                g64c.constants[f"{l.name}/bn_shift"].data = np.asarray(l.shift, np.float64)
        out32, out64, out64c, flips = [], [], [], []
        for i in range(n):
            env = executor.run(g, {"input": x[i:i + 1]}, keep_intermediates=True)
            env64 = executor.run(g, {"input": x[i:i + 1]}, keep_intermediates=True, float_dtype=np.float64)
            out32.append(env[g.outputs[0]].reshape(-1))
            out64.append(env64[g.outputs[0]].reshape(-1))
            out64c.append(executor.run(g64c, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]].reshape(-1))
            rec = {"sample": i}
            for tag, envx, prem, dt in (("ir_f64_vs_source_f64", env64, pre64, np.float64), ("ir_f32_vs_source_f32", env, pre32, np.float32)):
                for k, pt in enumerate(pts):
                    a_ir = finn_bnn.ir_activation(envx, pt)[0]
                    a_src = src_act(prem[k][i], dt)
                    dd = np.where(a_ir != a_src)[0]
                    if dd.size:
                        rec[tag] = {"point": pt["point"], "n_flips": int(dd.size), "max_abs_pre_source": float(np.abs(prem[k][i][dd]).max()),
                                    "typical_abs_pre_source": float(np.median(np.abs(prem[k][i])))}
                        break
            if len(rec) > 1:
                flips.append(rec)
        out32 = np.stack(out32).astype(np.float32)
        out64, out64c = np.stack(out64), np.stack(out64c)
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{variant}.f32"))
        sem = compare_outputs(ref64, out64, labels)
        per = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per <= 1e-6))
        sem["deviating_samples_gt_1e-6"] = [int(i) for i in np.where(per > 1e-6)[0]]
        semc = compare_outputs(ref64, out64c, labels)
        report["variants"][variant] = {
            "file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path), "op_counts": g.op_counts(),
            "n_binary_layers": info["n_binary_layers"], "float_compute_layers": info["float_compute_layers"],
            "bit_output_layers": info["bit_output_layers"], "threshold_margins": info["threshold_margins"],
            "layer_precision": info["layer_precision"],
            "numpy_vs_reference_f32": compare_outputs(ref32, out32, labels),
            "numpy_vs_reference_f64": compare_outputs(ref64, out32, labels),
            "semantic_check_ir_f64_vs_reference_f64": sem,
            "semantic_check_float64_constants": {"max_abs_error": semc["max_abs_error"], "top1_agreement": semc["top1_agreement"]},
            "binarization_flips_vs_source": {"n_samples_with_flips": len(flips), "samples": flips},
        }
        print(f"[{variant}] {g.op_counts()} | SEMANTIC f64: max_abs {sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} "
              f"<=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{n} | f64 consts max_abs {semc['max_abs_error']:.3e} | numpy f32 vs ref f32 "
              f"max_abs {report['variants'][variant]['numpy_vs_reference_f32']['max_abs_error']:.3e} top1 "
              f"{report['variants'][variant]['numpy_vs_reference_f32']['top1_agreement']:.4f} | samples with flips {len(flips)}")
    a = np.fromfile(os.path.join(args.out_dir, "numpy_logits_unfused.f32"), np.float32).reshape(n, -1)
    b = np.fromfile(os.path.join(args.out_dir, "numpy_logits_fused.f32"), np.float32).reshape(n, -1)
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(a, b, labels)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)
    print("fused vs unfused numpy f32: max_abs %.3e top1 %.4f" % (report["fused_vs_unfused_numpy_f32"]["max_abs_error"],
                                                                 report["fused_vs_unfused_numpy_f32"]["top1_agreement"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
