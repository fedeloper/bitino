#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert a verified Brevitas BNN-PYNQ checkpoint (CNV/LFC/SFC/TFC 1W1A) to .bitino and validate it.

Runs in the pinned reference venv (.cache/venv-brevitas, envs/brevitas-*.txt).
The official model definitions come from the brevitas wheel (brevitas_examples.bnn_pynq.models,
model_with_cfg(name, pretrained=False)); the checkpoint is loaded from the local verified cache
with torch.load(weights_only=True) and load_state_dict(strict=True). Nothing is downloaded.

  .cache/venv-brevitas/bin/python tools/convert_brevitas.py --model-id brevitas-bnn-pynq-tfc-1w1a \
      --weights .cache/checkpoints/brevitas-bnn-pynq-tfc-1w1a/tfc_1w1a-45185b4d.pth \
      --eval-dir .cache/data/eval128_mnist_brevitas --out-dir artifacts/brevitas-bnn-pynq-tfc-1w1a

Produces in --out-dir:
  model.bitino / model_unfused.bitino     fused: BN+sign -> integer thresholds, bit outputs;
                                            unfused: literal BN (ChannelAffine) + Activation(sign)
  reference_logits.f32 / reference_logits_f64.f64   official Brevitas model in float32 / float64
  numpy_logits_fused.f32 / numpy_logits_unfused.f32  numpy executor (float32) per variant
  conversion_report.json                    Level-3 strict float64 semantic check, binarization
                                            ties/margins, first-flip analysis of deviating samples
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import brevitas_bnn  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

MODELS = {  # model_id -> (brevitas cfg name, eval preprocessing key, eval-log path relative to the checkpoint dir)
    "brevitas-bnn-pynq-cnv-1w1a": ("cnv_1w1a", "cifar10_totensor", "cnv_1w1a_eval-ea2f0427.txt"),
    "brevitas-bnn-pynq-lfc-1w1a": ("lfc_1w1a", "mnist_totensor", "lfc_1w1a_eval-318d8bd9.txt"),
    "brevitas-bnn-pynq-sfc-1w1a": ("sfc_1w1a", "mnist_totensor", "sfc_1w1a_eval-fa4b2f7f.txt"),
    "brevitas-bnn-pynq-tfc-1w1a": ("tfc_1w1a", "mnist_totensor", "tfc_1w1a_eval-45185b4d.txt"),
    # third-party retrain of the same CNV_1W1A recipe (checkpoint dict {epoch, state_dict, val_acc}; no eval log)
    "aliasgharshinwari-finn-cnv-w1a1-cifar10": ("cnv_1w1a", "cifar10_totensor", None),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(cfg_name: str, weights: str):
    import torch
    from brevitas_examples.bnn_pynq import models as bm

    model, cfg = bm.model_with_cfg(cfg_name, pretrained=False)   # official definition, no download
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    meta = {}
    if "state_dict" in sd:   # training checkpoint {epoch, state_dict, val_acc}
        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in sd.items() if k != "state_dict"}
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model._bitino_ckpt_meta = meta
    model.eval()
    srcdir = os.path.dirname(bm.__file__)
    src = {f: sha256_file(os.path.join(srcdir, f)) for f in ("CNV.py", "FC.py", "common.py", "tensor_norm.py")}
    cfg_dict = {s: dict(cfg.items(s)) for s in cfg.sections()}
    return model, src, cfg_dict, len(sd)


def binarization_points(model):
    """(module path, module) of every binary QuantIdentity, in forward order."""
    pts = []
    for name, m in model.named_modules():
        if type(m).__name__ == "QuantIdentity" and type(m.act_quant.fused_activation_quant_proxy.tensor_quant).__name__ == "ClampedBinaryQuant":
            pts.append((name, m))
    return pts


def reference_run(model, x_nhwc: np.ndarray, dtype):
    """Logits + the pre-sign value at every binarization point (per sample), float32 or float64."""
    import torch

    m = model.to(dtype)
    pts = binarization_points(m)
    pre = {n: [] for n, _ in pts}
    hooks = [mod.register_forward_pre_hook(lambda _m, inp, n=n: pre[n].append(inp[0].detach().to(torch.float64).numpy().copy()))
             for n, mod in pts]
    with torch.no_grad():
        out = m(torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))).to(dtype))
    out = getattr(out, "value", out)
    for h in hooks:
        h.remove()
    model.to(torch.float32)
    return out.detach().numpy(), {n: np.concatenate(v) for n, v in pre.items()}


def ir_binary_points(graph, fused: bool) -> dict:
    """Source binarization point -> (IR tensor name, threshold) whose ge-sign is that activation."""
    pts = {}
    for n in graph.nodes:
        if not fused and n.op == "Activation" and n.attrs.get("type") == "sign":
            pts[n.name] = (n.inputs[0], 0.0)          # ChannelAffine (BN) output
    if fused:
        for n in graph.nodes:
            if n.op in ("Conv2D", "BConv2D", "BDense") and n.inputs[2]:
                # producer name conv_features.i / features.i  ->  binary QuantIdentity at i+2
                head, idx = n.name.rsplit(".", 1)
                pts[f"{head}.{int(idx) + 2}"] = (n.outputs[0], 0.0)
    return pts


def to_nchw_flat(a: np.ndarray) -> np.ndarray:
    return a.transpose(0, 3, 1, 2).reshape(len(a), -1) if a.ndim == 4 else a.reshape(len(a), -1)


def first_flip(src_pre: dict, ir_env: dict, ir_pts: dict, order: list, sample: int) -> dict | None:
    """First binarization point (forward order) where ge-sign(source pre) != ge-sign(IR value)."""
    for name in order:
        if name not in ir_pts:
            continue
        t, thr = ir_pts[name]
        v_ir = to_nchw_flat(ir_env[t])[0]
        v_src = src_pre[name][sample].reshape(-1)
        if v_src.size != v_ir.size:
            continue
        d = np.where((v_src >= 0) != (v_ir >= thr))[0]
        if d.size:
            return {"point": name, "n_flips": int(d.size), "max_abs_pre_source": float(np.abs(v_src[d]).max()),
                    "typical_abs_pre_source": float(np.median(np.abs(v_src)))}
    return None


def finn_onnx_cross_check(onnx_path: str, unfused_graph, pre_key: str) -> dict:
    """Second view (not the trusted reference): compare the unfused IR with the FINN-ONNX export.

    * every Conv / MatMul initializer (OIHW / [in, out], exactly +-1) vs the IR weights in node order;
    * CNV: FINN's input path (Div 255, Mul 2, Sub 1, MultiThreshold UINT8 with 255 thresholds per
      channel, Add -128, Mul 1/128) vs the IR's ChannelAffine(2,-1) + FakeQuant for all 256 pixel values."""
    import onnx
    from onnx import numpy_helper

    from bitino.ir import unpack_bits

    m = onnx.load(onnx_path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    finn_w = [init[n.input[1]] for n in m.graph.node if n.op_type in ("Conv", "MatMul")]
    ir_w = []
    for n in unfused_graph.nodes:
        if n.op in ("Conv2D", "BConv2D", "BDense"):
            c = unfused_graph.constants[n.inputs[1]]
            w = unpack_bits(c.data, c.shape[-1]).reshape(c.shape).astype(np.float32) if c.dtype == "bit" else c.data
            ir_w.append(np.transpose(w, (0, 3, 1, 2)) if w.ndim == 4 else w.T)   # OHWI -> OIHW, [O,K] -> [K,O]
    res = {"file": os.path.basename(onnx_path), "sha256": sha256_file(onnx_path), "n_weight_tensors_finn": len(finn_w),
           "n_weight_tensors_ir": len(ir_w)}
    res["weights_identical"] = bool(len(finn_w) == len(ir_w) and all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(finn_w, ir_w)))
    mts = [n for n in m.graph.node if n.op_type == "MultiThreshold"]
    if pre_key == "cifar10_totensor" and mts and mts[0].input[1] in init:
        thr = init[mts[0].input[1]]                         # [C, 255], value counted iff x >= thr
        k = np.arange(256, dtype=np.float32)
        xin = (k / np.float32(255)) * np.float32(2) - np.float32(1)
        finn = (np.stack([np.sum(xin[:, None] >= thr[c][None, :], axis=1) for c in range(thr.shape[0])]) - 128) * np.float32(1 / 128)
        fq = next(n for n in unfused_graph.nodes if n.op == "FakeQuant")
        sc, zp = (unfused_graph.constants[n].data[0] for n in fq.inputs[1:3])
        ours = executor._fake_quant(xin, np.full(xin.size, sc, np.float32), np.full(xin.size, zp, np.float32), fq.attrs, np.float32)
        res["input_quantizer_identical_all_256_pixel_values"] = bool(all(np.array_equal(finn[c], ours) for c in range(thr.shape[0])))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--sha256", help="expected SHA-256 (overrides --registry)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--finn-onnx", help="optional second view: the finn-examples FINN-ONNX export of the same checkpoint "
                                        "(cnv-w1a1.onnx / tfc-w1a1.onnx); weights and the input quantizer are compared")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*[Nn]amed tensors.*")

    cfg_name, pre_key, log_name = MODELS[args.model_id]
    expected = args.sha256
    if not expected:
        reg = json.load(open(args.registry))
        entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
        if entry is None:
            print(f"ERROR: {args.model_id} not in {args.registry}", file=sys.stderr)
            return 2
        expected = entry["checkpoint"]["sha256"]
    actual = sha256_file(args.weights)
    if actual != expected:
        print(f"ERROR: checkpoint hash mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}")
    os.makedirs(args.out_dir, exist_ok=True)

    import brevitas
    import torch

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    model, src_hashes, cfg_dict, n_tensors = load_model(cfg_name, args.weights)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][pre_key]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    ref32, pre32 = reference_run(model, x, torch.float32)
    ref64, pre64 = reference_run(model, x, torch.float64)
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    order = [n for n, _ in binarization_points(model)]

    # Published per-batch accuracy (batch size 100, shuffle=False): batch 0 == our first 100 samples.
    log_path = os.path.join(os.path.dirname(args.weights), log_name) if log_name else ""
    log_check = None
    if log_name and os.path.exists(log_path) and len(x) >= 100:
        line = next(l for l in open(log_path) if "Test: [0/" in l)
        p1 = float(line.split("Prec@1")[1].split()[0])
        # trainer.accuracy() uses output.topk(k, 1, True, True); with exactly tied logits (the logits are
        # an affine map of integer accumulators, so ties happen) torch.topk may rank a higher index first,
        # whereas Bitino reports argmax (lowest index). Both are recorded.
        topk_pred = torch.from_numpy(ref32[:100]).topk(1, 1, True, True)[1][:, 0].numpy()
        ours_topk = 100.0 * float(np.mean(topk_pred == labels[:100]))
        ours_argmax = 100.0 * float(np.mean(ref32[:100].argmax(1) == labels[:100]))
        ties = [int(i) for i in range(100) if np.sum(ref32[i] == ref32[i].max()) > 1]
        log_check = {"eval_log": log_name, "eval_log_sha256": sha256_file(log_path), "log_batch0_prec1": p1,
                     "ours_first100_prec1_f32_torch_topk": ours_topk, "ours_first100_prec1_f32_argmax": ours_argmax,
                     "first100_samples_with_tied_top1_logits": ties, "match_torch_topk": abs(p1 - ours_topk) < 1e-9}

    def tie_stats(prem):
        return {n: {"exact_zeros": int(np.sum(v == 0)), "min_abs": float(np.abs(v).min()), "n_values": int(v.size)} for n, v in prem.items()}

    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_set": em["name"], "eval_preprocessing": pre_key,
              "reference": {"framework": f"brevitas {brevitas.__version__}, torch {torch.__version__} (CPU)",
                            "class": f"brevitas_examples.bnn_pynq.models.model_with_cfg('{cfg_name}')", "cfg": cfg_dict,
                            "source_sha256": src_hashes, "checkpoint_tensors": n_tensors,
                            "checkpoint_metadata": getattr(model, "_bitino_ckpt_meta", {}),
                            "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)),
                            "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                            "first100_vs_published_eval_log": log_check,
                            "binarization_inputs_f32": tie_stats(pre32), "binarization_inputs_f64": tie_stats(pre64)},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "tied_top1_logits": {"f32": [int(i) for i in range(len(x)) if np.sum(ref32[i] == ref32[i].max()) > 1],
                                   "f64": [int(i) for i in range(len(x)) if np.sum(ref64[i] == ref64[i].max()) > 1],
                                   "note": "top-1 = argmax (lowest index on exact ties) everywhere in Bitino reports"},
              "variants": {}}
    print(json.dumps({k: report["reference"][k] for k in ("top1_accuracy", "top1_accuracy_f64", "first100_vs_published_eval_log")}, indent=1))

    for variant, fused in (("unfused", False), ("fused", True)):
        graph, info = brevitas_bnn.convert(model, args.model_id, fused=fused)
        applied = passes.optimize(graph, fold=False)   # all folding is explicit in the adapter
        graph.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": variant,
            "source": {"framework": "pytorch+brevitas", "brevitas": brevitas.__version__, "class": report["reference"]["class"],
                       "repo": "https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/bnn_pynq",
                       "source_sha256": src_hashes, "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
            "conversion": {"tool": "tools/convert_brevitas.py", "adapter": "bitino.adapters.brevitas_bnn",
                           "passes": {"bn_sign_to_integer_threshold (adapter)": fused, **applied},
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                      "preprocessing": pre["definition"]},
            "outputs": ["logits"], "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"], "threshold_margins": info["threshold_margins"],
        }
        path = os.path.join(args.out_dir, "model.bitino" if fused else "model_unfused.bitino")
        file_sha = bfformat.write(graph, path)
        g = bfformat.read(path)     # validate what was written
        out32, out64, flips = [], [], []
        pts = ir_binary_points(g, fused)
        for i in range(len(x)):
            env = executor.run(g, {"input": x[i:i + 1]}, keep_intermediates=True)
            env64 = executor.run(g, {"input": x[i:i + 1]}, keep_intermediates=True, float_dtype=np.float64)
            out32.append(env[g.outputs[0]].reshape(-1))
            out64.append(env64[g.outputs[0]].reshape(-1))
            ff64 = first_flip(pre64, env64, pts, order, i)
            ff32 = first_flip(pre32, env, pts, order, i)
            if ff64 or ff32:
                flips.append({"sample": i, "ir_f64_vs_source_f64": ff64, "ir_f32_vs_source_f32": ff32})
        out32 = np.stack(out32).astype(np.float32)
        out64 = np.stack(out64)
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{variant}.f32"))
        sem = compare_outputs(ref64, out64, labels)
        per_sample = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per_sample <= 1e-6))
        sem["deviating_samples_gt_1e-6"] = [int(i) for i in np.where(per_sample > 1e-6)[0]]
        sem["max_rel_logit_error"] = float((np.abs(out64 - ref64) / np.maximum(np.abs(ref64), 1e-30)).max())
        report["variants"][variant] = {
            "file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
            "op_counts": g.op_counts(), "n_binary_layers": info["n_binary_layers"], "float_compute_layers": info["float_compute_layers"],
            "bit_output_layers": [n.name for n in g.nodes if n.attrs.get("output") == "bit"],
            "passes": graph.manifest["conversion"]["passes"], "threshold_margins": info["threshold_margins"],
            "numpy_vs_reference_f32": compare_outputs(ref32, out32, labels),
            "numpy_vs_reference_f64": compare_outputs(ref64, out32, labels),
            "semantic_check_ir_f64_vs_reference_f64": sem,
            "binarization_flips_vs_source": {"n_samples_with_flips": len(flips), "samples": flips},
        }
        print(f"[{variant}] {path}: {g.op_counts()} | SEMANTIC f64: max_abs {sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} "
              f"samples<=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{len(x)} | numpy f32 vs ref f32: max_abs "
              f"{report['variants'][variant]['numpy_vs_reference_f32']['max_abs_error']:.3e} | samples with binarization flips: {len(flips)}")
    if args.finn_onnx:
        report["finn_onnx_second_view"] = finn_onnx_cross_check(args.finn_onnx, brevitas_bnn.convert(model, fused=False)[0], pre_key)
        print("FINN-ONNX second view:", report["finn_onnx_second_view"])
    a = np.fromfile(os.path.join(args.out_dir, "numpy_logits_unfused.f32"), np.float32).reshape(len(x), -1)
    b = np.fromfile(os.path.join(args.out_dir, "numpy_logits_fused.f32"), np.float32).reshape(len(x), -1)
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(a, b, labels)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    print("fused vs unfused (numpy f32): max_abs %.3e top1 %.4f" % (report["fused_vs_unfused_numpy_f32"]["max_abs_error"],
                                                                    report["fused_vs_unfused_numpy_f32"]["top1_agreement"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
