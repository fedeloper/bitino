#!/usr/bin/env python3
"""Locates the FIRST binarization disagreement between a Larq/Keras model and
its converted IR, both evaluated with float64 intermediates.

For each requested sample, walks BConv2D nodes in topological order and finds
the first one whose binarized input differs between source and IR. Reports the
largest |pre-binarization value| among the flipped positions there. If that
value is at float32-parameter rounding scale (e.g. < 1e-4), the downstream
divergence is a rounding cascade (IR constants such as folded BatchNorm
scale/shift are stored in float32), not a semantic conversion error.

Quantizing nodes covered: BConv2D / BDense (sign: flip = different sign) and
QBConv2D / QBDense (k-bit DoReFa: flip = different integer level). For k-bit
nodes the analogue of |pre| is the distance of x to the nearest rounding
boundary (i + 1/2) / n; "typical" is the median of that distance.

--indices accepts a comma-separated list, "all", or "report:<variant>" (the
deviating samples recorded in <bfmodel dir>/conversion_report.json).

Run inside the Larq reference env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from bitino import executor, format as bfformat  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo-class", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bfmodel", required=True, help="UNFUSED converted model (node names = Keras layer names)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--indices", required=True, help="comma-separated sample indices")
    ap.add_argument("--json", required=True)
    ap.add_argument("--zoo-module-file", help="legacy larq_zoo model file (tools/convert_larq_zoo.py)")
    args = ap.parse_args()

    import tensorflow as tf

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_larq_zoo import zoo_factory

    tf.config.set_visible_devices([], "GPU")
    factory, _ = zoo_factory(args.zoo_class, args.zoo_module_file)
    model32 = factory(weights=args.weights)
    tf.keras.backend.set_floatx("float64")
    model = factory(weights=None)
    model.set_weights([w.astype(np.float64) for w in model32.get_weights()])

    g = bfformat.read(args.bitino)
    producers = {o: n for n in g.nodes for o in n.outputs}
    # Layer names of the float64 rebuild are uniquified by Keras; map by position.
    keras_layers = {l32.name: l64 for l32, l64 in zip(model32.layers, model.layers)}
    bnodes = [n for n in g.nodes if n.op in ("BConv2D", "BDense", "QBConv2D", "QBDense")]
    # Keras tensors feeding each binary conv = output of the IR producer's Keras layer.
    feeders = [producers[n.inputs[0]].name for n in bnodes]
    probe = tf.keras.Model(model.inputs, [keras_layers[f].output for f in feeders])
    logits_model = tf.keras.Model(model.inputs, model.layers[-1].input)

    def logits64(x):
        return logits_model(x, training=False).numpy()

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"]["larq_zoo"]
    x_all = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.indices == "all":
        indices = list(range(len(x_all)))
    elif args.indices.startswith("report:"):
        rep = json.load(open(os.path.join(os.path.dirname(os.path.abspath(args.bitino)), "conversion_report.json")))
        indices = rep["variants"][args.indices.split(":", 1)[1]]["semantic_check_ir_f64_vs_reference_f64"]["deviating_samples_gt_1e-6"]
    else:
        indices = [int(i) for i in args.indices.split(",") if i]
    results = []
    for idx in indices:
        x = x_all[idx:idx + 1].astype(np.float64)
        k_outs = [o.numpy() for o in probe(x, training=False)]
        env = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        rec = {"sample": idx, "first_flip_node": None}
        for n, kv in zip(bnodes, k_outs):
            iv = env[n.inputs[0]].reshape(kv.shape)
            if n.op in ("QBConv2D", "QBDense"):
                nl = float(2 ** int(n.attrs["act_bits"]) - 1)
                lk, li = np.round(np.clip(kv, 0, 1) * nl), np.round(np.clip(iv, 0, 1) * nl)
                flips = lk != li

                def dist(v):  # distance to the nearest level boundary (i + 1/2) / n inside (0, 1)
                    t = np.clip(v, 0, 1) * nl
                    return np.abs(t - (np.floor(t) + 0.5)) / nl
                pre_k, pre_i = dist(kv), dist(iv)
                kind = f"{int(n.attrs['act_bits'])}-bit level"
            else:
                flips = (kv >= 0) != (iv >= 0)
                pre_k, pre_i = np.abs(kv), np.abs(iv)
                kind = "sign"
            if flips.any():
                rec.update({"first_flip_node": n.name, "op": n.op, "flip_kind": kind, "n_flips": int(flips.sum()),
                            "n_bits": int(flips.size),
                            "max_abs_pre_at_flip_keras": float(np.max(pre_k[flips])),
                            "max_abs_pre_at_flip_ir": float(np.max(pre_i[flips])),
                            "max_abs_diff_feeding_tensor": float(np.max(np.abs(kv - iv))),
                            "typical_abs_pre": float(np.median(pre_k))})
                break
        if rec["first_flip_node"] is None:
            out_name = g.outputs[0]
            rec["max_abs_logit_diff_no_flip"] = float(np.max(np.abs(env[out_name].reshape(-1) - logits64(x).reshape(-1))))
        results.append(rec)
        print(json.dumps(rec))
    json.dump(results, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
