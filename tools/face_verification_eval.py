#!/usr/bin/env python3
"""Face-verification accuracy (LFW / AgeDB-30 insightface .bin pairs) of embedding models: Bitino vs source.

Embeddings of every image and of its horizontal flip (eval sets from tools/eval_sets/face_verification_bin.py,
keys `insightface` / `insightface_flip`) are produced by
  * the Bitino runtime:  --config NAME=bench:<model.bitino>:<stage>     (bitino_bench, --bench)
  * the fp32 Keras source: --config NAME=keras:<h5>:<sha256>               (Larq reference env; batch 128 as the
                                                                           authors' eval_callback)
and cached as <out-dir>/<set>_<NAME>{,_flip}.f32. Metrics per config (pairs: images 2k, 2k+1):

  max_accuracy_flip  insightface_larq_keras evals.py eval_callback.__eval_func__ (the authors' number, also the score in
                     the checkpoint file names): emb + emb_flip, L2-normalise, cosine; accuracy maximised over the
                     thresholds given by the top 10% of impostor scores (the threshold is chosen ON the test pairs).
  tenfold_flip       standard LFW protocol as in insightface verification.py evaluate(): squared L2 distance of the
                     normalised (emb + emb_flip), thresholds arange(0, 4, 0.01), 10 contiguous folds (KFold without
                     shuffle), threshold picked on 9 folds, accuracy on the held-out fold; mean +- std.
  *_noflip           the same without the flip embedding.
Paired: for every pair of configs, the number of pairs whose same/different decision differs at the first config's
best (max-accuracy) threshold, and the per-image cosine between the two configs' embeddings.

  python3 tools/face_verification_eval.py --eval-dir .cache/data/eval12000_lfw --set lfw --out-dir artifacts/<id> \
      --bench build-s6/host/bitino_bench --threads 4 \
      --config bitino_neon=bench:artifacts/<id>/model.bitino:neon --json artifacts/<id>/face_verification_lfw.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def l2n(e):
    e = np.asarray(e, np.float64)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


def max_accuracy(emb, issame):
    """evals.py eval_callback.__eval_func__ (sklearn normalize == l2n)."""
    e = l2n(emb)
    dists = (e[::2] * e[1::2]).sum(1)
    tt = np.sort(dists[issame])
    ff = np.sort(dists[np.logical_not(issame)])
    t_steps = int(0.1 * ff.shape[0])
    acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
    i = int(np.argmax(acc_count))
    return float(acc_count[i] / dists.shape[0]), float(ff[i - t_steps])


def tenfold(emb, issame, n_folds=10):
    """insightface verification.py calculate_roc accuracy (squared L2 on normalised embeddings, KFold no shuffle)."""
    e = l2n(emb)
    dist = np.sum((e[::2] - e[1::2]) ** 2, 1)
    thresholds = np.arange(0, 4, 0.01)
    n = len(issame)
    folds = np.array_split(np.arange(n), n_folds)   # == sklearn KFold(n_splits, shuffle=False) for n % 10 == 0
    accs, thr = [], []
    for k in range(n_folds):
        test = folds[k]
        train = np.setdiff1d(np.arange(n), test)
        acc_train = [np.mean((dist[train] < t) == issame[train]) for t in thresholds]
        best = int(np.argmax(acc_train))
        thr.append(float(thresholds[best]))
        accs.append(float(np.mean((dist[test] < thresholds[best]) == issame[test])))
    return float(np.mean(accs)), float(np.std(accs)), thr


def embed_bench(bench, model, stage, threads, inputs, out):
    subprocess.run([bench, "--model", model, "--inputs", inputs, "--outputs", out, "--stage", stage,
                    "--threads", str(threads), "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)


def embed_keras(h5, sha, inputs, shape, out):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import tensorflow as tf
    from convert_keras_embedding import build_larqface, sha256_file
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if sha256_file(h5) != sha:
        raise SystemExit(f"{h5}: sha256 mismatch")
    model, _, _ = build_larqface(h5)
    x = np.memmap(inputs, np.float32, "r", shape=tuple(shape))
    with open(out, "wb") as f:
        for i in range(0, shape[0], 128):
            f.write(np.asarray(model(np.asarray(x[i:i + 128]), training=False), np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--set", required=True, help="tag for cache files, e.g. lfw / agedb30")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", action="append", required=True, help="NAME=bench:<bfmodel>:<stage> | NAME=keras:<h5>:<sha256>")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build", "host", "bitino_bench"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    issame = np.array(em["pairs"]["issame"], bool)
    n_img = 2 * len(issame)
    shape = em["preprocessing"]["insightface"]["shape"]
    assert shape[0] == n_img
    report = {"eval_set": em["name"], "n_pairs": int(len(issame)), "n_same": int(issame.sum()),
              "source_files": em["source_files"], "configs": {}, "paired": {}}
    embs = {}
    for spec in args.config:
        name, kind_rest = spec.split("=", 1)
        kind, rest = kind_rest.split(":", 1)
        files = {}
        for key in ("insightface", "insightface_flip"):
            out = os.path.join(args.out_dir, f"{args.set}_{name}{'_flip' if key.endswith('flip') else ''}.f32")
            if not os.path.exists(out):
                inp = os.path.join(args.eval_dir, em["preprocessing"][key]["file"])
                if kind == "bench":
                    model, stage = rest.rsplit(":", 1)
                    embed_bench(args.bench, model, stage, args.threads, inp, out)
                elif kind == "keras":
                    h5, sha = rest.rsplit(":", 1)
                    embed_keras(h5, sha, inp, shape, out)
                else:
                    raise SystemExit(f"unknown config kind {kind}")
            files[key] = np.fromfile(out, np.float32).reshape(n_img, -1)
        e, ef = files["insightface"], files["insightface_flip"]
        embs[name] = (e, ef)
        both = e.astype(np.float64) + ef.astype(np.float64)
        ma, mt = max_accuracy(both, issame)
        ma0, mt0 = max_accuracy(e, issame)
        tf_m, tf_s, tf_thr = tenfold(both, issame)
        t0_m, t0_s, _ = tenfold(e, issame)
        report["configs"][name] = {"spec": spec, "max_accuracy_flip": ma, "max_accuracy_flip_threshold": mt,
                                   "max_accuracy_noflip": ma0, "tenfold_flip_mean": tf_m, "tenfold_flip_std": tf_s,
                                   "tenfold_flip_thresholds": tf_thr, "tenfold_noflip_mean": t0_m, "tenfold_noflip_std": t0_s,
                                   "finite": bool(np.isfinite(both).all())}
        print(f"{args.set} {name}: max-acc(flip) {ma * 100:.2f}% | 10-fold(flip) {tf_m * 100:.2f} +- {tf_s * 100:.2f}% | "
              f"max-acc(noflip) {ma0 * 100:.2f}% | 10-fold(noflip) {t0_m * 100:.2f}%")
    names = list(embs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ea = l2n(embs[a][0].astype(np.float64) + embs[a][1])
            eb = l2n(embs[b][0].astype(np.float64) + embs[b][1])
            thr = report["configs"][a]["max_accuracy_flip_threshold"]
            da = (ea[::2] * ea[1::2]).sum(1) > thr
            db = (eb[::2] * eb[1::2]).sum(1) > thr
            cos = (ea * eb).sum(1)
            report["paired"][f"{a}__vs__{b}"] = {"pair_decisions_differing_at_threshold_of_first": int(np.sum(da != db)),
                                                 "image_cosine_min": float(cos.min()), "image_cosine_mean": float(cos.mean()),
                                                 "n_images_cosine_below_0.999": int(np.sum(cos < 0.999))}
    json.dump(report, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
