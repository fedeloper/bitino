#!/usr/bin/env python3
"""Linear-probe sanity check of BRILLsson embeddings: Bitino runtime vs the fp32 Keras source.

NOT the paper's protocol (eval.py trains a QuantDense probe on the Speech Commands *train* split for 100 epochs;
that split is 2.4 GB and was not downloaded). Here: the 4,890-clip official v0.02 test set
(tools/eval_sets/speech_commands_logmel.py, 12 classes), a closed-form ridge classifier (one-hot targets,
features standardised on the training folds, lambda = 1e-2 * mean feature variance * n_train), 5 folds by
index % 5. The same folds and probe are applied to every config, so the numbers compare Bitino with the
source on a task metric; they are not comparable to the paper's accuracies.

Configs: NAME=bench:<model.bitino>:<stage> | NAME=keras:<weights h5>:<sha256>:<zoo class>   (record = [projection
embedding (hidden), pooled encoder features]); --features pooled|projection|both.

  python3 tools/speech_probe_eval.py --eval-dir .cache/data/eval4890_speechcommands --out-dir artifacts/<id> \
      --bench build-s6/host/bitino_bench --config bitino_neon=bench:artifacts/<id>/model.bitino:neon \
      --json artifacts/<id>/speech_probe_sc_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ridge_cv(f, y, n_cls, folds=5):
    f = np.asarray(f, np.float64)
    pred = np.empty(len(y), int)
    for k in range(folds):
        te = np.arange(len(y)) % folds == k
        tr = ~te
        mu, sd = f[tr].mean(0), f[tr].std(0) + 1e-8
        a, b = (f[tr] - mu) / sd, (f[te] - mu) / sd
        a1, b1 = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
        t = np.eye(n_cls)[y[tr]]
        lam = 1e-2 * len(a)
        w = np.linalg.solve(a1.T @ a1 + lam * np.eye(a1.shape[1]), a1.T @ t)
        pred[te] = np.argmax(b1 @ w, 1)
    return float(np.mean(pred == y)), pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--bench", default=os.path.join(ROOT, "build", "host", "bitino_bench"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"]["brillsson_logmel"]
    n = pre["shape"][0]
    y = np.array([s["label"] for s in em["samples"]])
    n_cls = len(em["class_index"]["classes"])
    inp = os.path.join(args.eval_dir, pre["file"])
    report = {"eval_set": em["name"], "n": int(n), "n_classes": n_cls, "protocol": __doc__.split("\n\n")[1].strip(),
              "configs": {}, "paired": {}}
    preds = {}
    for spec in args.config:
        name, rest = spec.split("=", 1)
        kind, rest = rest.split(":", 1)
        out = os.path.join(args.out_dir, f"sc_test_{name}.f32")
        if not os.path.exists(out):
            if kind == "bench":
                model, stage = rest.rsplit(":", 1)
                subprocess.run([args.bench, "--model", model, "--inputs", inp, "--outputs", out, "--stage", stage,
                                "--threads", str(args.threads), "--warmup", "0", "--iterations", "1"], check=True,
                               capture_output=True)
            else:
                h5, sha, zoo = rest.split(":")
                sys.path.insert(0, os.path.join(ROOT, "tools"))
                import tensorflow as tf
                from convert_keras_embedding import build_brillsson, sha256_file, run_keras
                tf.config.set_visible_devices([], "GPU")
                tf.config.threading.set_intra_op_parallelism_threads(4)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                assert sha256_file(h5) == sha, "sha256 mismatch"
                model, _, _ = build_brillsson(h5, zoo, int(pre["shape"][1]), args.hidden)
                x = np.fromfile(inp, np.float32).reshape(pre["shape"])
                run_keras(model, x, np.float32, batch=64).astype(np.float32).tofile(out)
        rec = np.fromfile(out, np.float32).reshape(n, -1)
        r = {"spec": spec, "finite": bool(np.isfinite(rec).all())}
        for feat, sl in (("pooled", slice(args.hidden, None)), ("projection", slice(0, args.hidden))):
            acc, p = ridge_cv(rec[:, sl], y, n_cls)
            r[f"probe_accuracy_{feat}"] = acc
            preds[(name, feat)] = p
        report["configs"][name] = r
        print(f"{name}: probe acc pooled {r['probe_accuracy_pooled'] * 100:.2f}% projection {r['probe_accuracy_projection'] * 100:.2f}%")
    names = list(report["configs"])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report["paired"][f"{a}__vs__{b}"] = {f: {"prediction_agreement": float(np.mean(preds[(a, f)] == preds[(b, f)])),
                                                     "discordant_correct_a_only": int(np.sum((preds[(a, f)] == y) & (preds[(b, f)] != y))),
                                                     "discordant_correct_b_only": int(np.sum((preds[(a, f)] != y) & (preds[(b, f)] == y)))}
                                                 for f in ("pooled", "projection")}
    json.dump(report, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
