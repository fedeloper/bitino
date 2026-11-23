#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert an OpenVINO Open Model Zoo binary classifier (IR .xml + .bin) to .bitino and validate it.

Conversion itself needs only numpy (bitino.adapters.openvino_ir parses the IR with
ElementTree). The trusted fp32 reference is the OpenVINO runtime (CPU plugin,
INFERENCE_PRECISION_HINT=f32), imported only for that step. Run in the pinned
project-local env (envs/openvino-requirements.txt):

  .cache/venv-openvino/bin/python tools/convert_omz.py --model-id omz-resnet50-binary-0001 \\
      --checkpoint-dir .cache/checkpoints --registry registry/models.json \\
      --eval-dir .cache/data/eval128_omz_bgr --out-dir artifacts/omz-resnet50-binary-0001

High-precision reference: OpenVINO only runs float32, so reference_logits_f64.f64 is a numpy
float64 evaluation of the IR graph itself (openvino_ir.evaluate: every IR layer implemented
directly from the opset-1 spec, independent of the Bitino adapter/executor). It is checked
against the OpenVINO fp32 outputs (reported as reference_f32_vs_f64, the fp32 noise floor).

Produces in --out-dir:
  model.bitino / model_unfused.bitino   fused (affine chains merged + folded, BConv2D->FakeQuantize
                                          thresholds fused into bit outputs) / 1:1 mapping of the IR
  reference_logits.f32                    OpenVINO CPU plugin, float32
  reference_logits_f64.f64                numpy float64 evaluation of the IR
  ir_numpy_logits_f32.f32                 numpy float32 evaluation of the IR (diagnostic)
  numpy_logits_fused.f32 / numpy_logits_unfused.f32   Bitino numpy executor (float32)
  conversion_report.json                  semantic check, first-flip analysis, accuracies
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process; parallelism is per sample

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import openvino_ir as oi  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

MODELS = {  # model_id -> (checkpoint subdir, precision dir, file stem)
    "omz-resnet18-xnor-binary-onnx-0001": ("omz-resnet18-xnor-binary-onnx-0001", "FP32-INT1", "resnet18-xnor-binary-onnx-0001"),
    "omz-resnet18-xnor-binary-onnx-0001-fp16": ("omz-resnet18-xnor-binary-onnx-0001", "FP16-INT1", "resnet18-xnor-binary-onnx-0001"),
    "omz-resnet50-binary-0001": ("omz-resnet50-binary-0001", "FP32-INT1", "resnet50-binary-0001"),
    "omz-resnet50-binary-0001-fp16": ("omz-resnet50-binary-0001", "FP16-INT1", "resnet50-binary-0001"),
}
PREPROCESS = "omz_bgr"
SEM_TOL = 1e-6


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_graphs(ir: oi.IRModel, model_id: str):
    unfused, info = oi.convert(ir, model_id)
    applied_u = {"prune_unused_constants": passes.prune_unused_constants(unfused)}
    fused, _ = oi.convert(ir, model_id)
    applied = {"merge_affine_chains": oi.merge_affine_chains(fused)}
    applied.update(passes.optimize(fused, fold=True))
    applied["fuse_threshold_into_bit_output"] = oi.fuse_threshold_into_bit_output(fused)
    applied["prune_unused_constants"] += passes.prune_unused_constants(fused)
    return {"unfused": (unfused, applied_u), "fused": (fused, applied)}, info


# ------------------------------------------------------------------ first-flip analysis
def _ir_binarizer_inputs(ir: oi.IRModel, env: dict) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """For every BinaryConvolution (IR topological order): (bitino node name, FQ input x,
    threshold t broadcast to x, binarized bool = FQ output == output_high), NHWC."""
    out = []
    for lid in ir.topo:
        layer = ir.layers[lid]
        if layer.type != "BinaryConvolution":
            continue
        fq_t = ir.inputs_of(lid)[0]
        fq = ir.layers[fq_t[0]]
        fins = ir.inputs_of(fq.id)
        x = env[fins[0]]
        t = np.broadcast_to(ir.const_value(fins[1]).astype(x.dtype), x.shape)
        b = env[fq_t] == np.broadcast_to(ir.const_value(fins[4]).astype(x.dtype), x.shape)
        tr = (0, 2, 3, 1)
        out.append((f"{layer.name}#{layer.id}", x.transpose(tr), t.transpose(tr), b.transpose(tr)))
    return out


def first_flip(ir: oi.IRModel, ir_env: dict, g, bf_env: dict) -> dict | None:
    """First BinaryConvolution whose binarized input differs between the IR evaluation and the
    Bitino executor. Margins |x - t| are reported on the IR side (and on the Bitino side
    when its input is f32 with a threshold)."""
    nodes = {n.name: n for n in g.nodes if n.op == "BConv2D"}
    prod = {o: n for n in g.nodes for o in n.outputs}
    for name, x, t, b_ir in _ir_binarizer_inputs(ir, ir_env):
        n = nodes[name]
        xb = bf_env[n.inputs[0]]
        p = prod.get(n.inputs[0])
        if p is not None and p.op == "BConv2D" and p.attrs.get("output", "f32") == "bit":
            b_bf = xb > 0                          # executor stores bit outputs as +-1
            pre_bf = None
        else:
            thr = bf_env[n.inputs[4]] if len(n.inputs) > 4 and n.inputs[4] else 0.0
            mode = n.attrs.get("input_binarize", "ge")
            b_bf = xb > thr if mode == "gt" else xb >= thr
            pre_bf = xb - thr
        flips = b_ir != b_bf
        if flips.any():
            pre = np.abs(x - t)
            r = {"bconv": name, "n_flips": int(flips.sum()), "n_bits": int(flips.size),
                 "max_abs_pre_ir": float(pre[flips].max()), "typical_abs_pre_ir": float(np.median(pre)),
                 "pre_ir_at_flips": [float(v) for v in (x - t)[flips][:8]]}
            if pre_bf is not None:
                r["max_abs_pre_bitino"] = float(np.abs(pre_bf[flips]).max())
                r["max_abs_diff_binarizer_input"] = float(np.abs(xb.astype(np.float64) - x).max())
            return r
    return None


def ties_at_thresholds(ir: oi.IRModel, env: dict) -> int:
    """Number of binarizer inputs exactly equal to their threshold (x == t -> output_low)."""
    return int(sum(np.sum(x == t) for _, x, t, _ in _ir_binarizer_inputs(ir, env)))


# ------------------------------------------------------------------ workers
_G: dict = {}


def _init_worker(ir_paths, graph_paths) -> None:
    _G["ir"] = oi.IRModel(*ir_paths)
    _G["graphs"] = {k: bfformat.read(p) for k, p in graph_paths.items()}


def _run_one(x: np.ndarray):
    ir = _G["ir"]
    x_nchw = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    r64 = oi.evaluate(ir, x_nchw, np.float64, keep=True)
    r32 = oi.evaluate(ir, x_nchw, np.float32, keep=True)
    res = {"ir64": r64["outputs"][0].reshape(-1), "ir32": r32["outputs"][0].reshape(-1),
           "ties_ir64": ties_at_thresholds(ir, r64["env"]), "ties_ir32": ties_at_thresholds(ir, r32["env"])}
    del r32
    for k, g in _G["graphs"].items():
        out32 = executor.run(g, {"input": x})[g.outputs[0]].reshape(-1)
        env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        out64 = env64[g.outputs[0]].reshape(-1)
        res[f"{k}32"], res[f"{k}64"] = out32, out64
        dev = float(np.abs(out64 - res["ir64"]).max())
        res[f"{k}_flip"] = first_flip(ir, r64["env"], g, env64) if dev > SEM_TOL else None
        del env64
    return res


def openvino_reference(xml: str, binf: str, x: np.ndarray) -> tuple[np.ndarray, dict]:
    import openvino as ov

    core = ov.Core()
    cm = core.compile_model(core.read_model(xml, binf), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    req = cm.create_infer_request()
    out = []
    for i in range(len(x)):
        r = req.infer({0: np.ascontiguousarray(x[i:i + 1].transpose(0, 3, 1, 2))})
        out.append(np.asarray(r[cm.outputs[0]], np.float32).reshape(-1))
    meta = {"framework": f"openvino {ov.__version__}", "device": "CPU",
            "device_name": core.get_property("CPU", "FULL_DEVICE_NAME"),
            "inference_precision_hint": str(cm.get_property("INFERENCE_PRECISION_HINT")),
            "execution_mode_hint": str(cm.get_property("EXECUTION_MODE_HINT")),
            "input_layout": "NCHW (eval tensors are NHWC BGR 0..255, transposed)"}
    return np.stack(out), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--checkpoint-dir", required=True, help="holds <model>/<FP32-INT1|FP16-INT1>/<stem>.{xml,bin}")
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    sub, prec, stem = MODELS[args.model_id]
    xml = os.path.join(args.checkpoint_dir, sub, prec, stem + ".xml")
    binf = os.path.join(args.checkpoint_dir, sub, prec, stem + ".bin")
    reg = json.load(open(args.registry))
    entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
    if entry is None:
        print(f"ERROR: {args.model_id} not in {args.registry}", file=sys.stderr)
        return 2
    want_bin = entry["checkpoint"]["sha256"]
    m = re.search(r"sha256 ([0-9a-f]{64})", entry["checkpoint"].get("download_status", ""))
    want_xml = m.group(1) if m else None
    got_bin, got_xml = sha256_file(binf), sha256_file(xml)
    if got_bin != want_bin or (want_xml and got_xml != want_xml):
        print(f"ERROR: checkpoint hash mismatch: bin {got_bin} (want {want_bin}), xml {got_xml} (want {want_xml})", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: bin {got_bin} xml {got_xml}")
    os.makedirs(args.out_dir, exist_ok=True)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    n = len(x)
    labels = np.array([s["label"] for s in em["samples"]][:n])

    ir = oi.IRModel(xml, binf)
    graphs, info = build_graphs(ir, args.model_id)
    files = {}
    report = {"model_id": args.model_id, "n_samples": int(n), "eval_preprocessing": PREPROCESS,
              "eval_preprocessing_definition": pre["definition"],
              "source": {"xml": os.path.relpath(xml, ROOT) if xml.startswith(ROOT) else xml, "xml_sha256": got_xml,
                         "bin_sha256": got_bin, "ir_version": ir.version, "ir_op_counts": ir.op_counts(),
                         "u1_order": ir.u1_order, "license": entry.get("license"),
                         "claimed_metrics": entry.get("reported_metrics")},
              "layer_precision": info["layer_precision"], "variants": {}}
    for key, (g, applied) in graphs.items():
        g.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": key,
            "source": {"framework": "openvino-ir", "class": f"OMZ {stem} ({prec})",
                       "repo": entry.get("repo_url", ""), "checkpoint_file": os.path.basename(binf),
                       "checkpoint_sha256": got_bin, "xml_file": os.path.basename(xml), "xml_sha256": got_xml,
                       "license": "Apache-2.0 (Open Model Zoo)"},
            "conversion": {"tool": "tools/convert_omz.py", "adapter": "bitino.adapters.openvino_ir", "passes": applied,
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "numpy": np.__version__,
                           "binarization": "FakeQuantize levels=2 -> BConv2D threshold, input_binarize gt (x > t -> +1); "
                                           "pad_value -1; u1 weights LSB-first, bit 1 = +1"},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                      "preprocessing": pre["definition"]},
            "outputs": ["logits"], "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"],
        }
        path = os.path.join(args.out_dir, "model.bitino" if key == "fused" else "model_unfused.bitino")
        file_sha = bfformat.write(g, path)
        files[key] = path
        report["variants"][key] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                   "op_counts": g.op_counts(), "passes": applied}
        print(f"[{key}] wrote {path}: {g.op_counts()} passes={applied}")

    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=((xml, binf), files)) as pool:
        res = pool.map(_run_one, [x[i:i + 1] for i in range(n)], chunksize=1)

    ir64 = np.stack([r["ir64"] for r in res]).astype(np.float64)
    ir32 = np.stack([r["ir32"] for r in res]).astype(np.float32)
    ov32, ov_meta = openvino_reference(xml, binf, x)
    ov32.astype(np.float32).tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ir64.tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    ir32.tofile(os.path.join(args.out_dir, "ir_numpy_logits_f32.f32"))

    def top5(z):
        return float(np.mean([labels[i] in np.argsort(-z[i])[:5] for i in range(n)]))

    report["reference"] = dict(ov_meta, top1_accuracy=float(np.mean(ov32.argmax(1) == labels)), top5_accuracy=top5(ov32),
                               top1_accuracy_f64=float(np.mean(ir64.argmax(1) == labels)), top5_accuracy_f64=top5(ir64),
                               f64_definition="numpy float64 evaluation of the IR graph (bitino.adapters.openvino_ir.evaluate); "
                                              "OpenVINO itself is float32-only",
                               ties_at_threshold_ir_f64=int(sum(r["ties_ir64"] for r in res)),
                               ties_at_threshold_ir_f32=int(sum(r["ties_ir32"] for r in res)))
    report["reference_f32_vs_f64"] = compare_outputs(ir64, ov32, labels)
    report["ir_numpy_f32_vs_openvino_f32"] = compare_outputs(ov32, ir32, labels)
    report["ir_numpy_f32_vs_ir_f64"] = compare_outputs(ir64, ir32, labels)
    print("reference:", json.dumps({k: v for k, v in report["reference"].items() if "accuracy" in k}))
    print(f"OpenVINO fp32 vs IR fp64: max_abs {report['reference_f32_vs_f64']['max_abs_error']:.3e} "
          f"top1 agree {report['reference_f32_vs_f64']['top1_agreement']:.4f}")

    for key in files:
        out32 = np.stack([r[f"{key}32"] for r in res]).astype(np.float32)
        out64 = np.stack([r[f"{key}64"] for r in res])
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{key}.f32"))
        sem = compare_outputs(ir64, out64, labels)
        per = np.abs(out64 - ir64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per <= SEM_TOL))
        sem["deviating_samples"] = [int(i) for i in np.where(per > SEM_TOL)[0]]
        sem["per_sample_max_abs"] = [float(v) for v in per]
        flips = {int(i): res[i][f"{key}_flip"] for i in sem["deviating_samples"]}
        sem["first_flips"] = flips
        unexplained = [i for i, f in flips.items() if f is None]
        rel = [f["max_abs_pre_ir"] / f["typical_abs_pre_ir"] for f in flips.values() if f]
        sem["first_flip_summary"] = {"n_deviating": len(flips), "n_without_flip": len(unexplained),
                                     "max_abs_pre_at_first_flip": max((f["max_abs_pre_ir"] for f in flips.values() if f), default=None),
                                     "max_rel_pre_at_first_flip": max(rel, default=None),
                                     "max_n_flips_at_first_layer": max((f["n_flips"] for f in flips.values() if f), default=None)}
        report["variants"][key].update({
            "semantic_check_bitino_f64_vs_ir_f64": sem,
            "numpy_f32_vs_reference_f32": compare_outputs(ov32, out32, labels),
            "numpy_f32_vs_reference_f64": compare_outputs(ir64, out32, labels),
            "numpy_f32_vs_ir_numpy_f32": compare_outputs(ir32, out32, labels)})
        s = sem["first_flip_summary"]
        print(f"[{key}] SEMANTIC (Bitino f64 vs IR f64): max_abs {sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} "
              f"samples<=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{n}; deviating {s['n_deviating']} (no flip: {s['n_without_flip']}, "
              f"max |pre| at first flip {s['max_abs_pre_at_first_flip']}) | numpy f32 vs OV f32 top1 "
              f"{report['variants'][key]['numpy_f32_vs_reference_f32']['top1_agreement']:.4f}, vs f64 "
              f"{report['variants'][key]['numpy_f32_vs_reference_f64']['top1_agreement']:.4f}")
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
