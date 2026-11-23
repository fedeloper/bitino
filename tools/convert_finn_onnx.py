#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert the finn-examples FINN-ONNX (cnv-w1a1 / tfc-w1a1) and QONNX (cnv_1w1a_gtsrb) CNV / TFC graphs to
.bitino and validate them (slice S7).

Runs in the project venv .cache/venv-qonnx (qonnx 1.0.0, onnx, onnxruntime; envs/qonnx-*.txt):

  .cache/venv-qonnx/bin/python -I tools/convert_finn_onnx.py --model-id finn-examples-bnn-pynq-cnv-w1a1 \
      --zip .cache/checkpoints/finn-examples-bnn-pynq-onnx/onnx-models-bnn-pynq.zip \
      --eval-dir .cache/data/eval128_cifar10_finn --out-dir artifacts/finn-examples-bnn-pynq-cnv-w1a1

The ONNX member is read directly from the registry-verified zip (SHA-256 of the zip and of the member are
checked). Trusted reference: qonnx's ``execute_onnx`` on the original graph (float32; the custom ops
MultiThreshold / Quant / BipolarQuant run in qonnx's numpy implementations, the standard nodes in
onnxruntime). The graph ends in TopK (class index); the reference logits are the TopK input. float64
reference: ``finn_onnx.OnnxNumpy`` (numpy evaluation of the same graph in float64; onnxruntime has no
float64 Conv), cross-checked against qonnx in float32.

Outputs in --out-dir: model.bitino (fused: BN + sign -> integer thresholds), model_unfused.bitino,
reference_logits.f32 / reference_logits_f64.f64, numpy_logits_{fused,unfused}.f32, conversion_report.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import platform
import sys
import zipfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import finn_bnn, finn_onnx  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

# model_id -> registry record, zip member, member sha256, eval preprocessing key, input domain, classes,
#             published claim, Brevitas artifact with the same trained weights (cross-check) and its eval key
MODELS = {
    "finn-examples-bnn-pynq-cnv-w1a1": dict(
        registry="finn-examples-bnn-pynq-onnx", member="cnv-w1a1.onnx",
        member_sha256="47007fe69572a1c49b3d2be2607e21aaba92e2e7b8c29cb72254f22e2476c1d7",
        pre="cifar10_pixels", domain="pixels", classes=10,
        claim="not stated in the release; same weights as brevitas cnv_1w1a (84.22% CIFAR-10 test, Brevitas README/log)",
        claim_top1=0.8422, brevitas="brevitas-bnn-pynq-cnv-1w1a"),
    "finn-examples-bnn-pynq-tfc-w1a1": dict(
        registry="finn-examples-bnn-pynq-onnx", member="tfc-w1a1.onnx",
        member_sha256="1706b590920df2a73dc5f256725292b9a8640f904e0e0345547722b607dea6a0",
        pre="mnist_pixels", domain="pixels", classes=10,
        claim="not stated in the release; same weights as brevitas tfc_1w1a r1 (92.98% MNIST test, eval log)",
        claim_top1=0.9298, brevitas="brevitas-bnn-pynq-tfc-1w1a"),
    "finn-examples-cnv-w1a1-gtsrb": dict(
        registry="finn-examples-cnv-w1a1-gtsrb", member="onnx-models-gtsrb/cnv_1w1a_gtsrb.onnx",
        member_sha256="050a1e6e3bf65dd67a6bd024805da41e16156302ac6bb51bdf2cb76b935a7f90",
        pre="gtsrb_totensor", domain="unit", classes=44,
        claim="QONNX model zoo README: 96.93% GTSRB top-1 (author claim, 'provided by the trainer')",
        claim_top1=0.9693, brevitas=None),
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def binarizer_nodes(model, read_info) -> list:
    """Source activation binarizers in graph order (MultiThreshold BIPOLAR / BipolarQuant whose input is an
    activation), excluding the input quantizer of TFC, which is handled as the input threshold."""
    from onnx import numpy_helper

    init = {i.name for i in model.graph.initializer}
    produced_by_init = set(init)
    pts = []
    for nd in model.graph.node:
        if all(i in produced_by_init for i in nd.input if i):
            produced_by_init.update(nd.output)
            continue
        if nd.op_type == "BipolarQuant" or (nd.op_type == "MultiThreshold" and finn_onnx._attrs(nd).get("out_dtype") == "BIPOLAR"):
            pts.append(nd)
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--zip", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = MODELS[args.model_id]

    import onnx
    import onnxruntime
    import qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx

    reg = json.load(open(args.registry))
    entry = next(m for m in reg["models"] if m.get("model_id") == cfg["registry"])
    zraw = open(args.zip, "rb").read()
    zsha = sha256_bytes(zraw)
    if zsha != entry["checkpoint"]["sha256"]:
        print(f"ERROR: zip SHA-256 {zsha} != registry {entry['checkpoint']['sha256']}", file=sys.stderr)
        return 3
    raw = zipfile.ZipFile(io.BytesIO(zraw)).read(cfg["member"])
    msha = sha256_bytes(raw)
    if msha != cfg["member_sha256"]:
        print(f"ERROR: member {cfg['member']} SHA-256 {msha} != {cfg['member_sha256']}", file=sys.stderr)
        return 3
    print(f"zip SHA-256 OK ({zsha}); member {cfg['member']} OK ({msha})")
    os.makedirs(args.out_dir, exist_ok=True)
    model = onnx.load_from_string(raw)
    # (no onnx.checker: FINN-ONNX uses custom-domain nodes without an opset import)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][cfg["pre"]]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])
    n = len(x)

    inp, layers, rinfo = finn_onnx.read(model, cfg["domain"])
    logits_name = rinfo["logits_onnx_tensor"]
    bins = binarizer_nodes(model, rinfo)
    if inp.kind == "binary_threshold":
        bins = bins[1:]                      # TFC: the first BIPOLAR quantizer is the input threshold
    n_pts = sum(1 for l in layers if not l.final)
    assert len(bins) == n_pts, (len(bins), n_pts)

    # ---------------------------------------------------------------- references
    mw = ModelWrapper(model)
    if not all(vi.type.tensor_type.HasField("shape") for vi in mw.graph.value_info) or len(mw.graph.value_info) == 0:
        # qonnx's execute_onnx needs every intermediate shape: run qonnx's own InferShapes (shape annotation only)
        from qonnx.transformation.infer_shapes import InferShapes
        mw = mw.transform(InferShapes())
    ev = finn_onnx.OnnxNumpy(model)
    ref32, ref64, own32 = [], [], []
    pre32 = {nd.name: [] for nd in bins}
    pre64 = {nd.name: [] for nd in bins}
    topk_match = []
    for i in range(n):
        xi = np.ascontiguousarray(x[i:i + 1].transpose(0, 3, 1, 2))
        ctx = execute_onnx(mw, {ev.input_name: xi}, return_full_exec_context=True)
        ref32.append(np.asarray(ctx[logits_name], np.float32).reshape(-1))
        if ev.output_name != logits_name:
            topk_match.append(int(np.asarray(ctx[ev.output_name]).reshape(-1)[0]) == int(np.argmax(ref32[-1])))
        for nd in bins:
            pre32[nd.name].append(np.asarray(ctx[nd.input[0]], np.float64).reshape(-1))
        e64 = ev.run(xi, np.float64, keep=True)
        ref64.append(np.asarray(e64[logits_name], np.float64).reshape(-1))
        for nd in bins:
            pre64[nd.name].append(np.asarray(e64[nd.input[0]], np.float64).reshape(-1))
        own32.append(np.asarray(ev.run(xi, np.float32, outputs=[logits_name])[logits_name], np.float32).reshape(-1))
    ref32, ref64, own32 = np.stack(ref32), np.stack(ref64), np.stack(own32)
    pre32 = {k: np.stack(v) for k, v in pre32.items()}
    pre64 = {k: np.stack(v) for k, v in pre64.items()}
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    def ties(p):
        return {k: {"exact_zeros": int(np.sum(v == 0)), "min_abs": float(np.abs(v).min()), "n_values": int(v.size)} for k, v in p.items()}

    report = {
        "model_id": args.model_id, "registry_model_id": cfg["registry"], "n_samples": n, "eval_set": em["name"],
        "eval_preprocessing": cfg["pre"],
        "source": {"zip": os.path.basename(args.zip), "zip_sha256": zsha, "member": cfg["member"], "member_sha256": msha,
                   "onnx_producer": f"{model.producer_name} {model.producer_version}".strip(),
                   "opsets": [(o.domain or "ai.onnx", o.version) for o in model.opset_import]},
        "reference": {"framework": f"qonnx {getattr(qonnx, '__version__', '1.0.0')} execute_onnx (onnxruntime {onnxruntime.__version__}, onnx {onnx.__version__}), float32",
                      "float64": "bitino.adapters.finn_onnx.OnnxNumpy (numpy float64 evaluation of the same graph)",
                      "logits_tensor": logits_name, "graph_output": ev.output_name,
                      "graph_topk_equals_argmax_of_logits": (all(topk_match) if topk_match else None),
                      "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)), "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                      "claim": cfg["claim"], "binarization_inputs_f32": ties(pre32), "binarization_inputs_f64": ties(pre64)},
        "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
        "onnxnumpy_f32_vs_qonnx_f32": compare_outputs(ref32, own32, labels),
        "reader": rinfo, "variants": {},
        "tied_top1_logits_f32": [int(i) for i in range(n) if np.sum(ref32[i] == ref32[i].max()) > 1],
    }
    print(json.dumps({k: report["reference"][k] for k in ("top1_accuracy", "top1_accuracy_f64", "graph_topk_equals_argmax_of_logits")}),
          "| onnxnumpy f32 vs qonnx max_abs %.3e" % report["onnxnumpy_f32_vs_qonnx_f32"]["max_abs_error"])

    import onnx as _onnx  # noqa: F401
    for variant, fused in (("unfused", False), ("fused", True)):
        graph, info = finn_bnn.emit(inp, layers, fused=fused)
        applied = passes.optimize(graph, fold=False)
        graph.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": variant,
            "source": {"framework": "FINN-ONNX/QONNX (qonnx)", "registry_model_id": cfg["registry"], "file": cfg["member"],
                       "file_sha256": msha, "container_sha256": zsha, "repo": "https://github.com/Xilinx/finn-examples"},
            "conversion": {"tool": "tools/convert_finn_onnx.py", "adapter": "bitino.adapters.finn_onnx + finn_bnn",
                           "passes": {"bn_sign_to_integer_threshold (adapter)": fused, **applied},
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "numpy": np.__version__},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32", "preprocessing": pre["definition"],
                      "domain_note": "exact for the declared input domain (uint8 pixel values k, as k or float32(k)/255); "
                                     "the input quantizer was verified on all 256 values"},
            "outputs": ["logits"], "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"], "threshold_margins": info["threshold_margins"],
        }
        path = os.path.join(args.out_dir, "model.bitino" if fused else "model_unfused.bitino")
        file_sha = bfformat.write(graph, path)
        g = bfformat.read(path)
        pts = info["binarization_points"]
        # float64-constants copy: final affine (and input affine) in float64
        g64c = bfformat.read(path)
        for l in layers:
            if l.final:
                names = (f"{l.name}/post_mul", f"{l.name}/post_add") if fused else (f"{l.name}/affine_scale", f"{l.name}/affine_shift")
                g64c.constants[names[0]].data = np.asarray(l.scale, np.float64)
                g64c.constants[names[1]].data = np.asarray(l.shift, np.float64)
            elif not fused:
                g64c.constants[f"{l.name}/bn_scale"].data = np.asarray(l.scale, np.float64)
                g64c.constants[f"{l.name}/bn_shift"].data = np.asarray(l.shift, np.float64)
        if inp.kind == "int8":
            g64c.constants["input/scale"].data = np.full(inp.shape[-1], inp.affine[0], np.float64)
            g64c.constants["input/shift"].data = np.full(inp.shape[-1], inp.affine[1], np.float64)
        out32, out64, out64c, flips = [], [], [], []
        for i in range(n):
            env = executor.run(g, {"input": x[i:i + 1]}, keep_intermediates=True)
            env64 = executor.run(g, {"input": x[i:i + 1]}, keep_intermediates=True, float_dtype=np.float64)
            out32.append(env[g.outputs[0]].reshape(-1))
            out64.append(env64[g.outputs[0]].reshape(-1))
            out64c.append(executor.run(g64c, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]].reshape(-1))
            rec = {"sample": i}
            for tag, envx, prem in (("ir_f64_vs_source_f64", env64, pre64), ("ir_f32_vs_source_f32", env, pre32)):
                for nd, pt in zip(bins, pts):
                    a_ir = finn_bnn.ir_activation(envx, pt)[0]
                    a_src = np.where(prem[nd.name][i] >= 0, 1, -1)
                    d = np.where(a_ir != a_src)[0]
                    if d.size:
                        rec[tag] = {"point": pt["point"], "onnx_node": nd.name, "n_flips": int(d.size),
                                    "max_abs_pre_source": float(np.abs(prem[nd.name][i][d]).max()),
                                    "typical_abs_pre_source": float(np.median(np.abs(prem[nd.name][i])))}
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
              f"{report['variants'][variant]['numpy_vs_reference_f32']['top1_agreement']:.4f} | flips {len(flips)}")
    a = np.fromfile(os.path.join(args.out_dir, "numpy_logits_unfused.f32"), np.float32).reshape(n, -1)
    b = np.fromfile(os.path.join(args.out_dir, "numpy_logits_fused.f32"), np.float32).reshape(n, -1)
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(a, b, labels)

    # Cross-check with the Brevitas conversion of the same trained weights (different serialisation and reader).
    if cfg["brevitas"]:
        bdir = os.path.join(ROOT, "artifacts", cfg["brevitas"])
        bp = os.path.join(bdir, "model.bitino")
        if os.path.exists(bp):
            gb = bfformat.read(bp)
            gf = bfformat.read(os.path.join(args.out_dir, "model.bitino"))
            cb = [gb.constants[n_.inputs[1]] for n_ in gb.nodes if n_.op in ("Conv2D", "BConv2D", "BDense")]
            cf = [gf.constants[n_.inputs[1]] for n_ in gf.nodes if n_.op in ("Conv2D", "BConv2D", "BDense")]
            pb = [(gb.constants[n_.inputs[2]].data if n_.inputs[2] else None, gb.constants[n_.inputs[3]].data if len(n_.inputs) > 3 and n_.inputs[3] else None)
                  for n_ in gb.nodes if n_.op in ("Conv2D", "BConv2D", "BDense")]
            pf = [(gf.constants[n_.inputs[2]].data if n_.inputs[2] else None, gf.constants[n_.inputs[3]].data if len(n_.inputs) > 3 and n_.inputs[3] else None)
                  for n_ in gf.nodes if n_.op in ("Conv2D", "BConv2D", "BDense")]
            same_w = len(cb) == len(cf) and all(np.array_equal(p.data, q.data) for p, q in zip(cb, cf))
            same_thr = all((u[0] is None and v[0] is None) or (u[0] is not None and v[0] is not None and np.array_equal(u[0], v[0])) for u, v in zip(pb[:-1], pf[:-1])) \
                and all((u[1] is None and v[1] is None) or (u[1] is not None and v[1] is not None and np.array_equal(u[1], v[1])) for u, v in zip(pb[:-1], pf[:-1]))
            bl = np.fromfile(os.path.join(bdir, "numpy_logits_fused.f32"), np.float32).reshape(n, -1)
            report["brevitas_crosscheck"] = {
                "artifact": cfg["brevitas"], "weight_tensors_identical": bool(same_w), "n_weight_tensors": len(cf),
                "integer_thresholds_identical_hidden_layers": bool(same_thr),
                "final_affine_max_abs_diff": [float(np.max(np.abs(pb[-1][k].astype(np.float64) - pf[-1][k].astype(np.float64)))) for k in (0, 1)],
                "numpy_logits_vs_brevitas_numpy_logits_same_images": compare_outputs(bl, b, labels)}
            print("Brevitas cross-check:", {k: v for k, v in report["brevitas_crosscheck"].items() if k != "numpy_logits_vs_brevitas_numpy_logits_same_images"},
                  "logits max_abs %.3e top1 %.4f" % (report["brevitas_crosscheck"]["numpy_logits_vs_brevitas_numpy_logits_same_images"]["max_abs_error"],
                                                     report["brevitas_crosscheck"]["numpy_logits_vs_brevitas_numpy_logits_same_images"]["top1_agreement"]))
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)
    print("fused vs unfused numpy f32: max_abs %.3e" % report["fused_vs_unfused_numpy_f32"]["max_abs_error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
