#!/usr/bin/env python3
"""Convert a verified Larq Zoo checkpoint to .bitino and produce reference outputs.

Runs inside the Larq reference environment (envs/larq-zoo-requirements.txt):

  .cache/venv-larq/bin/python tools/convert_larq_zoo.py --model-id larq-zoo-quicknet-small \
      --zoo-class sota.QuickNetSmall --weights .cache/checkpoints/.../quicknet_small_weights.h5 \
      --sha256 <expected> --eval-dir .cache/data/eval128 --out-dir artifacts/larq-zoo-quicknet-small

Produces in --out-dir:
  model.bitino / model_unfused.bitino   converted graphs (fused = BN/ReLU folding passes applied)
  reference_logits.f32 / reference_probs.f32   source-framework outputs on the eval set
  reference_logits_f64.f64                source model evaluated in float64
  numpy_logits_{unfused,fused}.f32        numpy executor (float32) of each converted graph
  ir_f64_logits_{unfused,fused}.f64       numpy executor in float64 (strict semantic check)
  conversion_report.json                  Level-3 comparison (source vs numpy IR executor)

--zoo-class examples: sota.QuickNetLarge, literature.BinaryResNetE18, literature.BinaryDenseNet28,
literature.MeliusNet22, literature.BinaryAlexNet, literature.XNORNet, literature.RealToBinaryNet,
literature.DoReFaNet.

Legacy models removed from larq_zoo (QuickNet-XL v0.1.1): --zoo-module-file points at the model file of a
read-only larq/zoo source tree at the old commit; it is executed in-process against the installed larq_zoo core
(ModelFactory, utils) and --zoo-class names its factory function, e.g.
  --zoo-module-file .cache/repos/larq-zoo-40f4cdc/larq_zoo/sota/quicknet.py --zoo-class sota.QuickNetXL
The h5 weights are loaded by topological order (weights-only file), so the converter also checks that the
weighted-layer names stored in the h5 equal those of the rebuilt model, in order.

Level-3 extras (all models): exact zeros at every binarization input (source fp32/fp64, numpy fp32) and a
float64-constants check (unfused IR with the folded BatchNorm scale/shift recomputed in float64 vs the float64
source; all other IR constants are the source's own fp32 parameters).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import keras_larq  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zoo_factory(zoo_class: str, module_file: str | None):
    """larq_zoo.<module>.<cls>, or <cls> from a legacy larq_zoo model file executed in-process (read only)."""
    import larq_zoo

    module, cls = zoo_class.split(".")
    if not module_file:
        return getattr(getattr(larq_zoo, module), cls), None
    import importlib.util

    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(f"bitino_legacy_{module}_{cls}", module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = os.path.abspath(os.path.join(os.path.dirname(module_file), "..", ".."))
    sha_file = os.path.join(root, "COMMIT_SHA")
    commit = open(sha_file).read().strip() if os.path.exists(sha_file) else "unknown"
    return getattr(mod, cls), {"module_file": os.path.relpath(module_file, root), "module_sha256": sha256_file(module_file),
                               "commit": commit}


def check_h5_layer_order(model, weights: str) -> dict:
    """Weights-only h5 files are loaded by layer order: the names of the weighted layers stored in the file must
    equal those of the instantiated model, in order (op layers such as tf_op_layer_* carry no weights)."""
    import h5py

    with h5py.File(weights, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["layer_names"]]
        stored = [n for n in names if len(f[n].attrs.get("weight_names", [])) > 0]
        keras_version = f.attrs.get("keras_version", b"")
    mine = [l.name for l in model.layers if l.weights]
    ok = stored == mine
    if not ok:
        raise RuntimeError(f"h5 weighted-layer order differs from the model: {list(zip(stored, mine))[:5]}")
    return {"n_weighted_layers": len(mine), "names_identical_in_order": ok,
            "keras_version": keras_version.decode() if isinstance(keras_version, bytes) else str(keras_version)}


def quant_input_zeros(model, x, dtype) -> dict:
    """Exact zeros at the inputs of the quantized (binarizing) layers of the Keras model."""
    import tensorflow as tf

    qls = [l for l in model.layers if type(l).__name__ in ("QuantConv2D", "QuantDense")
           and getattr(l, "input_quantizer", None) is not None]
    ins = [l._inbound_nodes[0].keras_inputs for l in qls]
    ins = [t[0] if isinstance(t, (list, tuple)) else t for t in ins]
    probe = tf.keras.Model(model.inputs, ins)
    z = tot = 0
    for i in range(0, len(x), 8):
        outs = probe(x[i:i + 8].astype(dtype), training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for o in outs:
            o = o.numpy()
            z += int(np.sum(o == 0))
            tot += int(o.size)
    return {"exact_zero": z, "total": tot, "n_quantized_layers": len(qls)}


def bn_float64_constants(model) -> dict:
    """Folded BatchNorm scale/shift exactly as keras_larq computes them, but kept in float64."""
    out = {}
    for layer in model.layers:
        if type(layer).__name__ != "BatchNormalization":
            continue
        gamma = layer.gamma.numpy().astype(np.float64) if layer.scale else 1.0
        beta = layer.beta.numpy().astype(np.float64) if layer.center else 0.0
        mean = layer.moving_mean.numpy().astype(np.float64)
        inv = 1.0 / np.sqrt(layer.moving_variance.numpy().astype(np.float64) + layer.epsilon)
        out[f"{layer.name}/scale"] = gamma * inv
        out[f"{layer.name}/shift"] = beta - mean * gamma * inv
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--zoo-class", required=True, help="e.g. sota.QuickNetSmall or literature.BiRealNet")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sha256", required=True, help="expected SHA-256 of the weights file")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--zoo-module-file", help="legacy larq_zoo model file (see module docstring)")
    args = ap.parse_args()

    actual = sha256_file(args.weights)
    if actual != args.sha256:
        print(f"ERROR: checkpoint hash mismatch: expected {args.sha256}, got {actual}", file=sys.stderr)
        return 3
    os.makedirs(args.out_dir, exist_ok=True)

    import larq
    import larq_zoo
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    factory, legacy = zoo_factory(args.zoo_class, args.zoo_module_file)
    model = factory(weights=args.weights, include_top=True)
    h5_check = check_h5_layer_order(model, args.weights)

    # Logits = input of the final softmax layer.
    last = model.layers[-1]
    assert (type(last).__name__ == "Activation" and last.activation.__name__ == "softmax") or \
        type(last).__name__ == "Softmax", "expected softmax head"
    logits_model = tf.keras.Model(model.inputs, [last.input, model.output])

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"]["larq_zoo"]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    ref_logits, ref_probs = [], []
    for i in range(0, len(x), 16):
        lg, pr = logits_model(x[i:i + 16], training=False)
        ref_logits.append(lg.numpy())
        ref_probs.append(pr.numpy())
    ref_logits = np.concatenate(ref_logits).astype(np.float32)
    ref_probs = np.concatenate(ref_probs).astype(np.float32)
    ref_logits.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref_probs.tofile(os.path.join(args.out_dir, "reference_probs.f32"))

    # Float64 evaluation of the same source model: the high-precision reference.
    # Binary networks amplify float32 rounding (sign flips near 0), so the
    # float32 reference has its own noise floor; see docs/numerical-correctness.md.
    tf.keras.backend.set_floatx("float64")
    model64 = factory(weights=None, include_top=True)
    model64.set_weights([w.astype(np.float64) for w in model.get_weights()])
    logits64_model = tf.keras.Model(model64.inputs, model64.layers[-1].input)
    ref_logits64 = np.concatenate([logits64_model(x[i:i + 16].astype(np.float64), training=False).numpy()
                                   for i in range(0, len(x), 16)])
    zeros_src = {"source_fp32": quant_input_zeros(model, x, np.float32), "source_fp64": quant_input_zeros(model64, x, np.float64)}
    tf.keras.backend.set_floatx("float32")
    ref_logits64.tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    report = {"model_id": args.model_id, "n_samples": int(len(x)),
              "reference": {"framework": f"tensorflow {tf.__version__}, larq {larq.__version__}, larq_zoo {larq_zoo.__version__}",
                            "top1_accuracy": float(np.mean(ref_logits.argmax(1) == labels)),
                            "top5_accuracy": float(np.mean([labels[i] in np.argsort(-ref_logits[i])[:5] for i in range(len(x))]))},
              "reference_f32_vs_f64": compare_outputs(ref_logits64, ref_logits, labels),
              "h5_layer_order_check": h5_check,
              "exact_zeros_at_binarization_inputs": zeros_src,
              "variants": {}}
    if legacy:
        report["reference"]["legacy_source"] = legacy

    for variant, fold in (("unfused", False), ("fused", True)):
        graph, info = keras_larq.convert(model, args.model_id)
        # Expose logits (softmax input) as an extra output for sensitive comparison.
        sm = graph.nodes[-1]
        assert sm.op == "Softmax"
        graph.set_outputs(sm.inputs[0], sm.outputs[0])
        applied = passes.optimize(graph, fold=fold)
        graph.manifest = {
            "schema": "bitino.manifest.v1",
            "model_id": args.model_id,
            "variant": variant,
            "source": {"framework": "larq_zoo", "class": args.zoo_class, "checkpoint_file": os.path.basename(args.weights),
                       "checkpoint_sha256": actual, **({"legacy_source": legacy} if legacy else {})},
            "conversion": {"tool": "tools/convert_larq_zoo.py", "adapter": "bitino.adapters.keras_larq",
                           "passes": applied, "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
                           "python": platform.python_version(), "tensorflow": tf.__version__, "larq": larq.__version__,
                           "larq_zoo": larq_zoo.__version__},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                      "preprocessing": pre["definition"]},
            "outputs": ["logits", "probabilities"],
            "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"],
        }
        path = os.path.join(args.out_dir, "model.bitino" if fold else "model_unfused.bitino")
        file_sha = bfformat.write(graph, path)
        # Level 3: numpy executor of the converted graph vs source framework.
        g2 = bfformat.read(path)
        out_logits, nz = [], 0
        bin_inputs = [(n.inputs[0], n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else None)
                      for n in g2.nodes if n.op in ("BConv2D", "BDense")]
        for i in range(len(x)):
            res = executor.run(g2, {"input": x[i:i + 1]}, keep_intermediates=True)
            out_logits.append(res[g2.outputs[0]].reshape(1, -1))
            nz += sum(int(np.sum(res[a] == (res[t] if t else 0.0))) for a, t in bin_inputs)
        out_logits = np.concatenate(out_logits)
        cmp = compare_outputs(ref_logits, out_logits, labels)
        cmp64 = compare_outputs(ref_logits64, out_logits, labels)
        # Strict semantic check: IR evaluated with float64 intermediates vs float64 source model.
        sem = np.concatenate([executor.run(g2, {"input": x[i:i + 1]}, float_dtype=np.float64)[g2.outputs[0]].reshape(1, -1)
                              for i in range(len(x))])
        cmp_sem = compare_outputs(ref_logits64, sem, labels)
        dev = np.max(np.abs(sem - ref_logits64), axis=1)
        cmp_sem["n_samples_within_1e-6"] = int(np.sum(dev <= 1e-6))
        cmp_sem["deviating_samples_gt_1e-6"] = [int(i) for i in np.where(dev > 1e-6)[0]]
        np.asarray(sem, np.float64).tofile(os.path.join(args.out_dir, f"ir_f64_logits_{variant}.f64"))
        if not fold:
            # float64-constants check: same unfused IR, folded BN constants recomputed in float64.
            gx = bfformat.read(path)
            f64c = bn_float64_constants(model)
            for name, v in f64c.items():
                # keras_larq forms mean*gamma in float32 before the float64 product, so the stored shift may differ
                # from float32(exact) by an ulp; anything larger would be a mapping error
                d = gx.constants[name].data.astype(np.float64)
                assert np.allclose(d, v, rtol=2.0 ** -21, atol=1e-30 + 2.0 ** -21 * np.abs(v).max()), f"{name}: bfmodel differs"
                gx.constants[name].data = np.asarray(v, np.float64)
            semx = np.concatenate([executor.run(gx, {"input": x[i:i + 1]}, float_dtype=np.float64)[gx.outputs[0]].reshape(1, -1)
                                   for i in range(len(x))])
            px = np.abs(semx - ref_logits64).max(axis=1)
            cmp_sem["float64_constants_check"] = {
                "description": "unfused IR with the folded BatchNorm scale/shift in float64 vs float64 source",
                "n_constants_replaced": len(f64c), "max_abs_error": float(px.max()),
                "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                "top1_agreement": float(np.mean(semx.argmax(1) == ref_logits64.argmax(1)))}
            print(f"[{variant}] float64 constants: max_abs {px.max():.3e}, <=1e-9 {int(np.sum(px <= 1e-9))}/{len(x)}")
        report["variants"][variant] = {"file": os.path.basename(path), "file_sha256": file_sha,
                                       "exact_zeros_at_binarization_inputs_numpy_f32": nz,
                                       "file_bytes": os.path.getsize(path), "op_counts": g2.op_counts(),
                                       "passes": applied, "numpy_vs_reference_f32": cmp, "numpy_vs_reference_f64": cmp64,
                                       "semantic_check_ir_f64_vs_reference_f64": cmp_sem}
        np.asarray(out_logits, np.float32).tofile(os.path.join(args.out_dir, f"numpy_logits_{variant}.f32"))
        print(f"[{variant}] {path}: {g2.op_counts()} | vs f32 ref: top1 agree {cmp['top1_agreement']:.4f} "
              f"max_abs {cmp['max_abs_error']:.3e} | vs f64 ref: top1 agree {cmp64['top1_agreement']:.4f} "
              f"max_abs {cmp64['max_abs_error']:.3e} cos {cmp64['cosine_min']:.6f} | SEMANTIC (f64 IR vs f64 ref): "
              f"max_abs {cmp_sem['max_abs_error']:.3e} top1 {cmp_sem['top1_agreement']:.4f} "
              f"within1e-6 {cmp_sem['n_samples_within_1e-6']}/{len(x)}")

    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    print(json.dumps(report["reference"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
