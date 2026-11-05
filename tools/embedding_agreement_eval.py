#!/usr/bin/env python3
"""Feature-level agreement of a headless backbone (no classifier released) on a large eval set.

Runs the host runtime on the eval set for one or more stages and compares the output features with the official
model's float32 (and, if present, float64) features of the same images: per-sample cosine similarity and relative L2
distance. There is no accuracy here by construction (slice S11: BURN / S2-BNN self-supervised ReActNet-A backbones).

  python3 tools/embedding_agreement_eval.py --artifacts artifacts/burn-reacta-stage2 --eval-dir .cache/data/eval5000 \\
      --preprocess torchvision_imagenet --bench build/host/bitino_bench --stages neon optimized \\
      --ref-f32 reference_logits_eval5000.f32 --ref-f64 reference_logits_f64_eval5000.f64 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np


def cosines(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)


def stats(ref, cand) -> dict:
    c = cosines(cand, ref)
    rel = np.linalg.norm(np.asarray(cand, np.float64) - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-300)
    q = np.quantile(c, [0.001, 0.01, 0.5])
    return {"cosine_min": float(c.min()), "cosine_q0.001": float(q[0]), "cosine_q0.01": float(q[1]), "cosine_median": float(q[2]),
            "cosine_mean": float(c.mean()), "n_cosine_lt_0.99999": int(np.sum(c < 0.99999)), "n_cosine_lt_0.999": int(np.sum(c < 0.999)),
            "n_cosine_lt_0.99": int(np.sum(c < 0.99)), "rel_l2_max": float(rel.max()), "rel_l2_mean": float(rel.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--model", default="model.bitino")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--preprocess", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--stages", nargs="+", default=["neon"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ref-f32", required=True, help="official float32 features (file in --artifacts)")
    ap.add_argument("--ref-f64", help="official float64 features (file in --artifacts)")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][args.preprocess]
    n = int(pre["shape"][0])
    ref32 = np.fromfile(os.path.join(args.artifacts, args.ref_f32), np.float32).reshape(n, -1)
    ref64 = np.fromfile(os.path.join(args.artifacts, args.ref_f64), np.float64).reshape(n, -1) if args.ref_f64 else None
    rep = {"eval_set": em["name"], "n": n, "model": os.path.join(args.artifacts, args.model), "feature_dim": int(ref32.shape[1]),
           "note": "feature-level agreement only: no classifier was released with this checkpoint, so no accuracy",
           "stages": {}}
    if ref64 is not None:
        rep["official_f32_vs_f64"] = stats(ref64, ref32)
    outs = {}
    for st in args.stages:
        with tempfile.TemporaryDirectory() as d:
            o = os.path.join(d, "o.f32")
            subprocess.run([args.bench, "--model", os.path.join(args.artifacts, args.model), "--inputs",
                            os.path.join(args.eval_dir, pre["file"]), "--outputs", o, "--stage", st, "--threads", str(args.threads),
                            "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
            outs[st] = np.fromfile(o, np.float32).reshape(n, -1)
        r = {"vs_official_f32": stats(ref32, outs[st])}
        if ref64 is not None:
            r["vs_official_f64"] = stats(ref64, outs[st])
        rep["stages"][st] = r
        print(st, json.dumps({k: {kk: v[kk] for kk in ("cosine_min", "cosine_median", "n_cosine_lt_0.999")} for k, v in r.items()}))
    if len(outs) > 1:
        a, b = args.stages[0], args.stages[1]
        rep[f"{b}_vs_{a}"] = stats(outs[a], outs[b])
    json.dump(rep, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
