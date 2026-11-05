#!/usr/bin/env python3
"""Paired accuracy comparison of two runtime configurations on a (larger) eval set.

Runs the host runtime for configuration A and B on the same images and reports top-1/top-5
accuracy with 95% Wilson intervals, the paired difference with a 95% CI, and an exact McNemar
test on the discordant pairs. Used to quantify the accuracy cost of lossy execution paths
(e.g. INT8), where the 128-sample smoke set is too small.

  python3 tools/accuracy_eval.py --eval-dir .cache/data/eval5000 \
      --a artifacts/X/model.bitino:optimized --b artifacts/X/model.bitino:optimized_int8 --json out.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def run(bench: str, spec: str, inputs: str, n: int, threads: int, jobs: int = 1, tmp_dir: str | None = None) -> np.ndarray:
    """Outputs of `bench` on all n samples. jobs > 1 splits the samples into that many contiguous chunks run
    concurrently (e.g. several qemu processes for scripts/bench_emulated.sh) and concatenates the outputs."""
    model, stage = spec.rsplit(":", 1)
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as d:  # input chunks: GBs for image sets, keep them off a RAM-backed /tmp
        jobs = max(1, min(jobs, n))
        if jobs == 1:
            parts = [(inputs, os.path.join(d, "o.f32"))]
        else:
            raw = np.memmap(inputs, np.uint8, mode="r")
            rec = raw.size // n
            assert rec * n == raw.size, f"{inputs}: size is not a multiple of {n} records"
            bounds = [n * j // jobs for j in range(jobs + 1)]
            parts = []
            for j in range(jobs):
                p = os.path.join(d, f"in{j}.bin")
                raw[bounds[j] * rec:bounds[j + 1] * rec].tofile(p)
                parts.append((p, os.path.join(d, f"o{j}.f32")))

        def one(part):
            subprocess.run([bench, "--model", model, "--inputs", part[0], "--outputs", part[1], "--stage", stage,
                            "--threads", str(threads), "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)

        with concurrent.futures.ThreadPoolExecutor(len(parts)) as ex:
            list(ex.map(one, parts))
        return np.concatenate([np.fromfile(o, np.float32) for _, o in parts]).reshape(n, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--preprocess", default="larq_zoo")
    ap.add_argument("--a", required=True, help="model.bitino:stage")
    ap.add_argument("--b", required=True, help="model.bitino:stage")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build/host/bitino_bench"))
    ap.add_argument("--bench-b", help="different runner for config B (e.g. scripts/bench_emulated.sh to use the ARM "
                                      "INT8 kernels the device selects); defaults to --bench")
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=1, help="run each config as this many concurrent sample chunks (e.g. qemu processes)")
    ap.add_argument("--save-logits", metavar="PREFIX", help="also write PREFIX_a.f32 / PREFIX_b.f32 (raw outputs)")
    ap.add_argument("--tmp-dir", default=os.path.join(ROOT, "build", "tmp"), help="where --jobs writes the input chunks")
    ap.add_argument("--shards", dest="jobs", type=int, help="alias of --jobs; each shard gets --threads // shards threads")
    ap.add_argument("--shards-b", "--jobs-b", dest="jobs_b", type=int, help="--jobs for config B (default --jobs), e.g. <= 6 qemu processes")
    ap.add_argument("--cache-dir", help="reuse the top-5 predictions of a configuration (model file bytes, stage, eval set, "
                                        "preprocessing, classes) from this directory; store new ones there")
    ap.add_argument("--json")
    a = ap.parse_args()
    a.shards_mode = any(x.startswith("--shards") for x in sys.argv[1:])
    em = json.load(open(os.path.join(a.eval_dir, "manifest.json")))
    pre = em["preprocessing"][a.preprocess]
    n = pre["shape"][0]
    labels = np.array([s["label"] for s in em["samples"]][:n])
    inputs = os.path.join(a.eval_dir, pre["file"])
    res = {"eval_set": em["name"], "n": int(n), "limitations": em["limitations"], "configs": {}}
    correct = {}
    for key, spec in (("a", a.a), ("b", a.b)):
        bench = a.bench_b if key == "b" and a.bench_b else a.bench
        jobs = (a.jobs_b or a.jobs) if key == "b" else a.jobs
        threads = max(1, a.threads // jobs) if a.shards_mode else a.threads
        ck = None
        if a.cache_dir and not a.save_logits:  # top-5 class indices (best first); the outputs are deterministic
            mpath, stage = spec.rsplit(":", 1)
            import hashlib
            h = hashlib.sha256(open(mpath, "rb").read()).hexdigest()[:24]
            ck = os.path.join(a.cache_dir, f"{h}_{stage}_{em['name']}_{a.preprocess}_{n}_{a.classes}_top5.npy")
        if ck and os.path.exists(ck):
            top5i = np.load(ck)
        else:
            full = run(bench, spec, inputs, n, threads, jobs, a.tmp_dir)
            if a.save_logits:
                full.tofile(f"{a.save_logits}_{key}.f32")
            logits = full[:, :a.classes]
            k5 = min(5, logits.shape[1])
            top5i = np.argsort(-logits, axis=1, kind="stable")[:, :k5].astype(np.int32)
            assert np.array_equal(top5i[:, 0], logits.argmax(1))
            if ck:
                os.makedirs(a.cache_dir, exist_ok=True)
                np.save(ck, top5i)
        pred = top5i[:, 0]
        top1 = pred == labels
        # top-5 (with <= 5 classes, e.g. SST-2, every label is in the top 5)
        top5 = (np.array([labels[i] in top5i[i] for i in range(n)]) if top5i.shape[1] >= 5 and a.classes > 5
                else np.ones(n, dtype=bool))
        correct[key] = (top1, pred)
        res["configs"][key] = {"spec": spec, "runner": os.path.relpath(bench, ROOT) if os.path.isabs(bench) else bench, "top1": float(top1.mean()), "top1_ci95": wilson(int(top1.sum()), n),
                               "top5": float(top5.mean()), "top5_ci95": wilson(int(top5.sum()), n)}
    ta, tb = correct["a"][0], correct["b"][0]
    b_only_a = int(np.sum(ta & ~tb))
    b_only_b = int(np.sum(~ta & tb))
    diff = tb.astype(float) - ta.astype(float)
    se = diff.std(ddof=1) / math.sqrt(n)
    res["paired"] = {"top1_diff_b_minus_a": float(diff.mean()), "diff_ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                     "a_correct_b_wrong": b_only_a, "a_wrong_b_correct": b_only_b,
                     "mcnemar_exact_p": mcnemar_exact(b_only_a, b_only_b),
                     "top1_prediction_agreement": float(np.mean(correct["a"][1] == correct["b"][1]))}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
