#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert a "Faster Than Lies" deepfake detector checkpoint (Lanzino et al., CVPRW 2024, arXiv:2406.04932;
https://github.com/fedeloper/binary_deepfake_detection, BNext4DFR in model.py) to .bitino and validate it.

The FFT-magnitude and LBP input channels are computed inside the model by the IR operators ImageGrayU8,
FftLogMagnitude and LocalBinaryPattern (docs/operators_image_features.md), so the deployed model takes RGB in
[0, 1] (eval preprocessing ftl_rgb01, tools/eval_sets/ftl_deepfake.py) and runs the whole method on the device.

The authors' model.py is imported from a clone (read only, no bytecode written) and run unmodified for the
reference outputs. Its training-only imports are replaced by inert stand-ins in this process: lightning
(LightningModule = nn.Module + no-op save_hyperparameters / log_dict + a .device property), fvcore and
torchmetrics (FLOP counting / metrics, not used by forward). BNext4DFR.__init__ loads the ImageNet BNext
checkpoint from pretrained/<size>_checkpoint.pth.tar; that torch.load is redirected to --bnext-weights with
weights_only=True (the backbone is then overwritten by the detector checkpoint, load_state_dict strict).
The detector checkpoint (Lightning .ckpt) is read with torch.load(weights_only=True).

  python -I tools/convert_ftl_deepfake.py --model-id ftl-deepfake-bnext-t-cocofake-unfrozen \
      --repo .cache/repos/fedeloper_binary_deepfake_detection --bnext-repo .cache/repos/BNext \
      --bnext-weights .cache/checkpoints/bnext-tiny/checkpoint.pth.tar \
      --weights .cache/checkpoints/ftl-deepfake/coco_fake_T_unfrozen.ckpt \
      --eval-dir .cache/data/eval128_cifake_ftl --out-dir artifacts/ftl-deepfake-bnext-t-cocofake-unfrozen

References: float32 = the authors' forward on float32 input; float64 = the same module in float64 (the FFT /
LBP channels are computed by the authors' numpy code on the float32-valued image and rounded to float32 by
their .float(), then promoted). The single output logit (positive = real) is reported as the two logits
[0, logit] (same as the IR head), so top-1 is the real/fake decision and labels are 1 = real, 0 = fake.

Outputs in --out-dir as tools/convert_torch_cnn.py (model[_sign_zero][_unfused].bitino, reference_logits*.f*,
numpy_logits_<variant>.f32, conversion_report.json) plus feature_check: the IR's gray / FFT / LBP channels vs the
authors' cv2 / numpy.fft / scikit-image channels on every eval image.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import io
import json
import multiprocessing as mp
import os
import platform
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_torch_cnn as ctc  # noqa: E402
from bitino import format as bfformat, image_ops, passes  # noqa: E402
from bitino.adapters import torch_ftl_deepfake  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

REPO_URL = "https://github.com/fedeloper/binary_deepfake_detection"
SIZES = {"BNext-T": "tiny", "BNext-S": "small", "BNext-M": "middle", "BNext-L": "large"}
PREPROCESS = "ftl_rgb01"


def _install_stubs(bnext_repo: str):
    import torch

    class LightningModule(torch.nn.Module):
        def save_hyperparameters(self, *a, **k):
            pass

        def log_dict(self, *a, **k):
            pass

        @property
        def device(self):
            p = next(self.parameters(), None)
            return p.device if p is not None else torch.device("cpu")

    def _unused(*a, **k):
        raise RuntimeError("training/metrics helper not available in the conversion process")

    lightning = types.ModuleType("lightning")
    lightning.LightningModule = LightningModule
    fv = types.ModuleType("fvcore")
    fvnn = types.ModuleType("fvcore.nn")
    fvnn.FlopCountAnalysis = fvnn.parameter_count = _unused
    fv.nn = fvnn
    tm = types.ModuleType("torchmetrics")
    tmf = types.ModuleType("torchmetrics.functional")
    tmc = types.ModuleType("torchmetrics.functional.classification")
    tmc.accuracy = tmc.auroc = _unused
    pkg, src = types.ModuleType("BNext"), types.ModuleType("BNext.src")
    pkg.__path__, src.__path__ = [], []
    spec = importlib.util.spec_from_file_location("BNext.src.bnext", os.path.join(bnext_repo, "src", "bnext.py"))
    bn = importlib.util.module_from_spec(spec)
    sys.modules.update({"lightning": lightning, "fvcore": fv, "fvcore.nn": fvnn, "torchmetrics": tm,
                        "torchmetrics.functional": tmf, "torchmetrics.functional.classification": tmc,
                        "BNext": pkg, "BNext.src": src, "BNext.src.bnext": bn})
    spec.loader.exec_module(bn)
    return bn


def load_model(repo: str, bnext_repo: str, bnext_weights: str, weights: str):
    import warnings

    import torch

    sys.dont_write_bytecode = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bn = _install_stubs(bnext_repo)
        spec = importlib.util.spec_from_file_location("ftl_model", os.path.join(repo, "model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    hp = dict(ck["hyper_parameters"])
    size = SIZES[hp["backbone"]]
    orig = torch.load

    def redirected(path, map_location=None, **kw):
        assert path == f"pretrained/{size}_checkpoint.pth.tar", path
        return orig(bnext_weights, map_location="cpu", weights_only=True)

    torch.load = redirected
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = mod.BNext4DFR(**hp)
    finally:
        torch.load = orig
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    meta = {"hyper_parameters": hp, "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
            "pytorch_lightning_version": ck.get("pytorch-lightning_version"),
            "class": f"model.BNext4DFR(**hyper_parameters) + BNext.src.bnext.BNext(size={size!r})",
            "source_sha256": {"model.py": ctc.sha256_file(os.path.join(repo, "model.py")),
                              "BNext/src/bnext.py": ctc.sha256_file(bn.__file__)},
            "bnext_repo_commit": ctc.repo_commit(bnext_repo)}
    return model, mod, meta


def feature_check(model, x_nhwc: np.ndarray, ops: dict) -> dict:
    """The authors' _add_new_channels_worker (cv2 / numpy.fft / skimage, float32 outputs) vs the IR operators."""
    import torch

    gray_mis = lbp_mis = fft_mis = 0
    fft_max = 0.0
    lbp_attrs = {k: ops["lbp"][k] for k in ("rp", "cp", "method", "div")}
    for i in range(len(x_nhwc)):
        x = x_nhwc[i:i + 1]
        with torch.no_grad():
            src = model._add_new_channels_worker(torch.from_numpy(x[0].copy())).numpy()   # [H, W, 2]: fft, lbp
        gray = image_ops.image_gray_u8(x, ops["gray"])
        f = image_ops.fft_log_magnitude(gray, ops["fft"])[0, :, :, 0]
        lb = image_ops.local_binary_pattern(gray, lbp_attrs)[0, :, :, 0]
        import cv2
        gsrc = cv2.cvtColor((x[0] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray_mis += int(np.sum(gray[0, :, :, 0] != gsrc))
        fft_mis += int(np.sum(f != src[..., 0]))
        fft_max = max(fft_max, float(np.max(np.abs(f - src[..., 0]))))
        lbp_mis += int(np.sum(lb != src[..., 1]))
    n = int(x_nhwc.shape[0] * x_nhwc.shape[1] * x_nhwc.shape[2])
    return {"n_images": int(len(x_nhwc)), "values_per_channel": n, "gray_mismatches": gray_mis, "lbp_mismatches": lbp_mis,
            "fft_values_differing": fft_mis, "fft_max_abs_diff": fft_max,
            "note": "gray vs cv2.cvtColor(BGR2GRAY), FFT / LBP vs the authors' _add_new_channels_worker float32 outputs"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--repo", required=True, help=f"clone of {REPO_URL}")
    ap.add_argument("--bnext-repo", required=True, help="clone of https://github.com/hpi-xnor/BNext (model definition)")
    ap.add_argument("--bnext-weights", required=True, help="ImageNet BNext checkpoint of the same size (only for BNext4DFR.__init__)")
    ap.add_argument("--weights", required=True, help="detector checkpoint (.ckpt)")
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--sha256", help="expected SHA-256 (overrides --registry)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="use N evenly spaced samples of the eval set (both classes; the indices are recorded in the report)")
    ap.add_argument("--variants", default="ge,sign_zero")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()

    expected = args.sha256
    if not expected:
        reg = json.load(open(args.registry))
        entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
        if entry is None:
            print(f"ERROR: {args.model_id} not in {args.registry}", file=sys.stderr)
            return 2
        expected = entry["checkpoint"]["sha256"]
    actual = ctc.sha256_file(args.weights)
    if actual != expected:
        print(f"ERROR: checkpoint hash mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}")
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    import torchvision

    torch.set_num_threads(args.torch_threads)
    model, _, meta = load_model(args.repo, args.bnext_repo, args.bnext_weights, args.weights)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    sel = np.arange(len(x))
    if args.max_samples and args.max_samples < len(x):
        sel = np.unique(np.round(np.linspace(0, len(x) - 1, args.max_samples)).astype(int))
        x = np.ascontiguousarray(x[sel])
    labels = np.array([em["samples"][i]["label"] for i in sel])

    def two(logit):
        logit = np.asarray(logit, np.float64).reshape(len(x), 1)
        return np.concatenate([np.zeros_like(logit), logit], 1)

    class Logits(torch.nn.Module):   # forward() returns {"logits": ...}
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, xb):
            return self.m(xb)["logits"]

    r32, zeros32 = ctc.reference_outputs(Logits(model), x, torch.float32, "HardSign")
    r64, zeros64 = ctc.reference_outputs(Logits(model), x, torch.float64, "HardSign")
    ref32, ref64 = two(r32).astype(np.float32), two(r64)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    g0, info0 = torch_ftl_deepfake.convert(model, args.model_id, binarize="sign_zero")
    fcheck = feature_check(model, x, info0["preprocessing_ops"])
    print("feature check:", fcheck)

    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_set": em.get("name"), "eval_preprocessing": PREPROCESS,
              "sample_indices": None if len(sel) == len(em["samples"]) else [int(i) for i in sel],
              "reference": {"framework": f"torch {torch.__version__} (CPU), torchvision {torchvision.__version__}",
                            "repo": REPO_URL, "repo_commit": ctc.repo_commit(args.repo), **meta,
                            "accuracy_fp32": float(np.mean(ref32.argmax(1) == labels)),
                            "accuracy_fp64": float(np.mean(ref64.argmax(1) == labels)),
                            "exact_zeros_at_binarization_f32": zeros32, "exact_zeros_at_binarization_f64": zeros64},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "feature_check": fcheck, "variants": {}}
    print(json.dumps({k: v for k, v in report["reference"].items() if k != "hyper_parameters"}, indent=1))

    files = {}
    for binarize in args.variants.split(","):
        for variant, fold in (("unfused", False), ("fused", True)):
            graph, info = torch_ftl_deepfake.convert(model, args.model_id, binarize=binarize)
            applied = passes.optimize(graph, fold=fold)
            stem = "model" if binarize == "ge" else "model_sign_zero"
            key = variant if binarize == "ge" else f"sign_zero_{variant}"
            graph.manifest = {
                "schema": "bitino.manifest.v1", "model_id": args.model_id,
                "variant": f"{variant}, input_binarize={binarize}",
                "source": {"framework": "pytorch", "class": meta["class"], "repo": REPO_URL,
                           "repo_commit": report["reference"]["repo_commit"], "source_sha256": meta["source_sha256"],
                           "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual,
                           "hyper_parameters": meta["hyper_parameters"]},
                "conversion": {"tool": "tools/convert_ftl_deepfake.py", "adapter": torch_ftl_deepfake.__name__, "passes": applied,
                               "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "python": platform.python_version(), "torch": torch.__version__, "input_binarize": binarize},
                "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                          "preprocessing": pre["definition"]},
                "outputs": ["logits [constant 0, real-vs-fake logit]: argmax 1 = real, sigmoid(logit) = P(real)"],
                "op_versions": {"format": 1},
                "layer_precision": info["layer_precision"],
                "weight_diagnostics": info["weight_diagnostics"],
                "unused_in_forward": info["unused_in_forward"],
                "preprocessing_ops": info["preprocessing_ops"],
                "licence": "no LICENSE file in the source repository: weights not redistributable (fetch recipe only)",
            }
            if not fold:
                ctc._EXACT[key] = info["float64_constants"]
            path = os.path.join(args.out_dir, f"{stem}.bitino" if fold else f"{stem}_unfused.bitino")
            file_sha = bfformat.write(graph, path)
            files[key] = path
            report["variants"][key] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": binarize, "op_counts": graph.op_counts(), "passes": applied}
            print(f"[{key}] wrote {path}: {graph.op_counts()} passes={applied}")

    jobs = [(k, x[i:i + 1], True) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=ctc._init_worker, initargs=(files,)) as pool:
        results = pool.map(ctc._run_one, jobs, chunksize=2)
    np_logits = {}
    for j, k in enumerate(files):
        res = results[j * len(x):(j + 1) * len(x)]
        out32 = np.stack([r[0] for r in res]).astype(np.float32)
        out64 = np.stack([r[1] for r in res])
        zeros = np.array([r[2] for r in res])
        zeros64 = np.array([r[3] for r in res])
        np_logits[k] = out32
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        sem = compare_outputs(ref64, out64, labels)
        per_sample = np.abs(out64 - ref64).max(axis=1)
        sem.update({"n_samples_max_abs_le_1e-6": int(np.sum(per_sample <= 1e-6)), "n_samples_max_abs_le_1e-4": int(np.sum(per_sample <= 1e-4)),
                    "deviating_samples_gt_1e-4": [int(i) for i in np.where(per_sample > 1e-4)[0]],
                    "per_sample_max_abs": [float(v) for v in per_sample]})
        if res[0][4] is not None:
            outx = np.stack([r[4] for r in res])
            px = np.abs(outx - ref64).max(axis=1)
            sem["float64_constants"] = {"max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-6": int(np.sum(px <= 1e-6)),
                                        "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                                        "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1)))}
        cmp, cmp64 = compare_outputs(ref32, out32, labels), compare_outputs(ref64, out32, labels)
        report["variants"][k].update({
            "numpy_vs_reference_f32": cmp, "numpy_vs_reference_f64": cmp64, "semantic_check_ir_f64_vs_reference_f64": sem,
            "accuracy_numpy_f32": float(np.mean(out32.argmax(1) == labels)),
            "exact_zeros_at_bconv_inputs_numpy_f32": {"total": int(zeros.sum()), "samples_with_zeros": int(np.sum(zeros.sum(1) > 0))},
            "exact_zeros_at_bconv_inputs_numpy_f64": {"total": int(zeros64.sum()), "samples_with_zeros": int(np.sum(zeros64.sum(1) > 0))}})
        print(f"[{k}] vs f32 ref: decision agree {cmp['top1_agreement']:.4f} max_abs {cmp['max_abs_error']:.3e} | vs f64 ref: agree "
              f"{cmp64['top1_agreement']:.4f} max_abs {cmp64['max_abs_error']:.3e} | SEMANTIC f64: max_abs {sem['max_abs_error']:.3e} "
              f"agree {sem['top1_agreement']:.4f} <=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{len(x)}"
              + (f" | f64 constants: max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else ""))
    if "sign_zero_fused" in np_logits and "fused" in np_logits:
        report["ge_vs_sign_zero_numpy_f32"] = {v: compare_outputs(np_logits[f"sign_zero_{v}"], np_logits[v], labels) for v in ("unfused", "fused")}
    if "fused" in np_logits:
        report["fused_vs_unfused_numpy_f32"] = compare_outputs(np_logits["unfused"], np_logits["fused"], labels)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
