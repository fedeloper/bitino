#!/usr/bin/env python3
"""Top-1 / top-5 of every converted ImageNet classifier on the full ILSVRC2012 validation set (50,000 images).

The 5,000-image regression set (.cache/data/eval5000, tools/build_eval_set.py + tools/accuracy_eval.py) covers only
the 100 ImageNet-100 classes; this tool repeats the measurement on all 50,000 validation images (1,000 classes x 50),
with each model's own evaluation preprocessing (registry/recipes.json eval.preprocess), on the x86-64 host runtime.

Configurations (one result per model and configuration):
  all        artifacts/<m>/model.bitino at stage `all` (the deployed float path), every ImageNet model of
             registry/published_accuracy.json
  int8_full  artifacts/<m>/model_int8_full.bitino at stage `int8_full` (the shipped full-integer file) where it
             exists; paired difference vs `all` with a 95 % CI and an exact McNemar test (as tools/accuracy_eval.py)
  neon       artifacts/<m>/model.bitino at the exact stage `neon`, for the 11 headline networks (paper Table tab:acc),
             on a slice of the first --neon-images images (the exact stage costs far more than 3x `all` on the host);
             reported as exact-vs-all top-1 agreement and paired accuracy difference on that slice

Order and labels: the selection rule of tools/build_eval_set.py (sample i = class i % 1000 of the sorted synset
directories, file i // 1000 of the sorted names), so every block of 1,000 consecutive samples holds one image per
class; labels from the sorted-synset convention via .cache/data/imagenet_class_index.json. The manifest (file names,
JPEG SHA-256, labels; no image data) is registry/eval_sets/imagenet-val-50000.json.

Resource model (shared host): the images are processed in shards of --shard images (default 1,000). For a shard, each
needed preprocessing is computed once (in the model family's own reference env, by the same code as the 5,000-image
set: tools/build_eval_set.py make_preprocessor for larq_zoo / torchvision_imagenet, tools/eval_sets/<p>.py for the
others), every configuration that uses it runs on it (one bitino_bench process per configuration, 1 thread), only
the top-5 class indices and logits are kept (.cache/full50k/results/<config>/sNNN.npz), and the shard tensor is
deleted once nothing pending needs it. At most --workers processes run at a time (preprocessing included). Results
are checkpointed per shard and configuration, so `run` resumes where it stopped. Run the whole tree under one
memory cap:

  scripts/capped.sh 24G nice -n 10 ~/anaconda3/bin/python -I tools/accuracy_full_val.py run --workers 8 \
      --val-dir <ILSVRC2012 validation directory> --passes pilot,float,neon,int8_headline,int8_rest,int8_fp16

Subcommands:
  manifest    build registry/eval_sets/imagenet-val-50000.json from --val-dir (hashes every JPEG)
  verify-prep check that this tool's preprocessing reproduces the recorded 5,000-image tensors bit for bit
              (same code path, applied to the original-resolution ImageNet-100 images of .cache/data/eval5000*)
  run         compute the missing (shard, configuration) results of the given passes
  summarize   write reports/accuracy_full50k/<model>__<config>.json, summary.json and summary.md
  prep        (internal) one shard of one preprocessing, run inside that preprocessing's env
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
HOME = os.path.expanduser("~")
DEFAULT_MANIFEST = os.path.join(ROOT, "registry", "eval_sets", "imagenet-val-50000.json")
DEFAULT_WORK = os.path.join(ROOT, ".cache", "full50k")
DEFAULT_OUT = os.path.join(ROOT, "reports", "accuracy_full50k")
CLASS_INDEX = os.path.join(ROOT, ".cache", "data", "imagenet_class_index.json")
CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
BENCH = os.path.join(ROOT, "build", "host", "bitino_bench")
N_CLASSES = 1000

# The 11 headline networks of paper/tomm/main.tex Table tab:acc (row order).
HEADLINE = ["larq-zoo-quicknet-small", "larq-zoo-quicknet", "larq-zoo-quicknet-large", "larq-zoo-birealnet",
            "dabnn-bireal18", "ab-bnn-imagenet-reactnet18", "ab-bnn-imagenet-reactnet34", "ab-bnn-imagenet-reactneta",
            "bhvit-tiny-binary", "omz-resnet50-binary-0001", "bnext-tiny"]

# Headline networks whose exact stage cost <= 3x the `all` cost in the 1,000-image pilot (cpu s/img, 1 thread:
# QuickNet-Large neon 0.081 / all 0.029 = 2.8x); the other ten are 3.2-13x (all 11 together 7.3x) -> neon slice only.
NEON_FULL = ["larq-zoo-quicknet-large"]

# Reference env (read-only, never modified) and implementation of each preprocessing. "script" = the recorded
# tools/eval_sets/<name>.py run on a per-shard manifest; "inproc" = tools/build_eval_set.py make_preprocessor.
PREPROC = {
    "larq_zoo": {"python": os.path.join(ROOT, ".cache/venv-larq/bin/python"), "impl": "inproc"},
    "torchvision_imagenet": {"python": os.path.join(HOME, "anaconda3/bin/python"), "impl": "inproc"},
    "bitorch_imagenet": {"python": os.path.join(HOME, "anaconda3/bin/python"), "impl": "script",
                         "script": "tools/eval_sets/bitorch_imagenet.py", "ref_flag": "--ref-manifest"},
    "timm_bicubic_crop095": {"python": os.path.join(HOME, "anaconda3/bin/python"), "impl": "script",
                             "script": "tools/eval_sets/timm_bicubic_crop095.py", "ref_flag": "--ref-manifest"},
    "bhvit": {"python": os.path.join(HOME, "anaconda3/envs/bhvit/bin/python"), "impl": "script",
              "script": "tools/eval_sets/bhvit.py", "ref_flag": "--base-manifest",
              "extra": ["--repo", os.path.join(ROOT, ".cache/repos/IMRL_BHViT")]},
    "omz_bgr": {"python": os.path.join(ROOT, ".cache/venv-openvino/bin/python"), "impl": "script",
                "script": "tools/eval_sets/omz_bgr.py", "ref_flag": "--base-manifest"},
    "tensorpack_bgr": {"python": os.path.join(ROOT, ".cache/venv-tensorpack-s12/bin/python"), "impl": "script",
                       "script": "tools/eval_sets/tensorpack_bgr.py", "ref_flag": "--base-manifest"},
}
# Recorded 5,000-image tensors (original-resolution ImageNet-100 val images) used by verify-prep.
EVAL5000 = {"larq_zoo": "eval5000", "torchvision_imagenet": "eval5000", "bitorch_imagenet": "eval5000_bitorch",
            "timm_bicubic_crop095": "eval5000_timm_bicubic_crop095", "bhvit": "eval5000_bhvit",
            "omz_bgr": "eval5000_omz_bgr", "tensorpack_bgr": "eval5000_tensorpack_bgr"}
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
                     "CUDA_VISIBLE_DEVICES": ""}
REC_BYTES = 224 * 224 * 3 * 4


# ----------------------------------------------------------------------------------------------------- statistics
sys.path.insert(0, TOOLS)
from accuracy_eval import mcnemar_exact, wilson  # noqa: E402  (same definitions as the 5,000-image reports)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def bf_header(path: str) -> dict:
    """JSON header of a .bitino (as tools/build_results.py header())."""
    with open(path, "rb") as f:
        h = f.read(64)
        assert h[:8] == b"BFMODEL1", path
        jo, js = struct.unpack("<QQ", h[16:32])
        f.seek(jo)
        return json.loads(f.read(js))


# ------------------------------------------------------------------------------------------------------- manifest
def select(val_dir: str, n: int) -> list[tuple[str, str]]:
    sys.path.insert(0, TOOLS)
    from build_eval_set import select as _select  # the 5,000-image set's rule
    return _select(val_dir, n)


SOURCE_RESIZED = ("ImageNet ILSVRC2012 validation set, all 50,000 images (1,000 classes x 50), from a locally RESIZED "
                  "copy (short side 256, 3 images 255; re-encoded JPEG, all RGB; the JPEG SHA-256 below are of that copy, "
                  "not of the original ILSVRC2012_img_val.tar files); images not redistributed")
LIMIT_RESIZED = ("Full validation set, but from a copy whose short side was already resized to 256 and re-encoded: every "
                 "evaluation pipeline sees different pixels than on the original JPEGs (resampling + JPEG re-encoding "
                 "before the model's own resize/crop); on the first 1,000 images this lowers top-1 by 1.6 pt on average "
                 "over the 55 models and changes ~25 % of the top-1 predictions (reports/accuracy_full50k/"
                 "diagnostic_original_resolution_1000/). Do not compare these numbers with published ones without that caveat.")
SOURCE_ORIGINAL = ("ImageNet ILSVRC2012 validation set, all 50,000 images (1,000 classes x 50), original JPEGs (byte-identical "
                   "to the ImageNet-100 val files of .cache/data/eval5000 where they overlap); images not redistributed")
LIMIT_ORIGINAL = "Full validation set at original resolution; each model's own evaluation preprocessing is applied."


def cmd_manifest(a) -> int:
    idx = json.load(open(a.class_index))
    synset_to_label = {v[0]: int(k) for k, v in idx.items()}
    classes = sorted(d for d in os.listdir(a.val_dir) if os.path.isdir(os.path.join(a.val_dir, d)))
    assert len(classes) == N_CLASSES, len(classes)
    assert [idx[str(i)][0] for i in range(N_CLASSES)] == classes, "class index is not the sorted-synset order"
    per = {c: len(os.listdir(os.path.join(a.val_dir, c))) for c in classes}
    assert set(per.values()) == {50}, "expected 50 images per class"
    picks = select(a.val_dir, a.n)
    samples = []
    for c, f in picks:
        samples.append({"file": f"{c}/{f}", "jpeg_sha256": sha256_file(os.path.join(a.val_dir, c, f)),
                        "label": synset_to_label[c]})
    assert len({s["file"] for s in samples}) == len(samples)
    head = {
        "schema": "bitino.eval_set.v1",
        "name": f"imagenet-val-{a.n}",
        "split": "val",
        "source": (SOURCE_RESIZED if a.copy == "resized" else SOURCE_ORIGINAL),
        "selection_rule": "class i % n_classes (sorted synsets), file i // n_classes (sorted names), i = 0..n-1 "
                          "(tools/build_eval_set.py select); each block of 1,000 consecutive samples has one image per class",
        "n": len(samples),
        "n_classes_covered": len({s["label"] for s in samples}),
        "class_index": {"url": CLASS_INDEX_URL, "sha256": sha256_file(a.class_index),
                        "convention": "sorted synsets (Keras / larq_zoo / torchvision ImageFolder)"},
        "class_names": [idx[str(i)][1] for i in range(N_CLASSES)],
        "preprocessing": {},
        "preprocessing_note": ("tensors are not stored; tools/accuracy_full_val.py computes them per shard with each "
                               "model's own preprocessing (registry/recipes.json eval.preprocess) and deletes them"),
        "limitations": (LIMIT_RESIZED if a.copy == "resized" else LIMIT_ORIGINAL),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:  # one sample per line: diffable, ~6 MB
        f.write(json.dumps(head, indent=1)[:-2] + ',\n "samples": [\n')
        f.write(",\n".join("  " + json.dumps(s, separators=(", ", ": ")) for s in samples))
        f.write("\n ]\n}\n")
    json.load(open(a.out))
    print(f"wrote {len(samples)} samples -> {a.out} (sha256 {sha256_file(a.out)})")
    return 0


# ------------------------------------------------------------------------------------------- preprocessing (env)
def cmd_prep(a) -> int:
    """One shard of one preprocessing (runs inside that preprocessing's env). Writes --out (.f32) and --out.json."""
    man = json.load(open(a.manifest))
    samples = man["samples"][a.start:a.start + a.count]
    spec = PREPROC[a.preprocess]
    work = a.out + ".d"
    if spec["impl"] == "inproc":
        from build_eval_set import make_preprocessor
        fn, definition = make_preprocessor(a.preprocess)
        out = np.empty((len(samples), 224, 224, 3), np.float32)
        for i, s in enumerate(samples):
            raw = open(os.path.join(a.val_dir, s["file"]), "rb").read()
            if hashlib.sha256(raw).hexdigest() != s["jpeg_sha256"]:
                print(f"ERROR: JPEG hash mismatch for {s['file']}", file=sys.stderr)
                return 3
            out[i] = fn(raw)
        out.tofile(a.out)
        entry = {"definition": definition, "script": "tools/build_eval_set.py make_preprocessor"}
    else:
        os.makedirs(work, exist_ok=True)
        ref = os.path.join(work, "shard_manifest.json")
        json.dump({k: v for k, v in man.items() if k not in ("samples", "class_names")} | {"samples": samples},
                  open(ref, "w"))
        cmd = [sys.executable, "-I", "-B", os.path.join(ROOT, spec["script"]), spec["ref_flag"], ref,
               "--val-dir", a.val_dir, "--out", work] + spec.get("extra", [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        em = json.load(open(os.path.join(work, "manifest.json")))["preprocessing"][a.preprocess]
        os.replace(os.path.join(work, em["file"]), a.out)
        entry = {"definition": em["definition"], "script": spec["script"]}
    shutil.rmtree(work, ignore_errors=True)
    n = os.path.getsize(a.out) // REC_BYTES
    assert n * REC_BYTES == os.path.getsize(a.out) and n == len(samples)
    entry.update({"preprocess": a.preprocess, "start": a.start, "count": len(samples), "shape": [n, 224, 224, 3],
                  "dtype": "float32", "layout": "NHWC", "sha256": sha256_file(a.out), "python": sys.executable,
                  "val_dir": a.val_dir})
    json.dump(entry, open(a.out + ".json", "w"), indent=1)
    return 0


def prep_cmd(p: str, manifest: str, val_dir: str, start: int, count: int, out: str) -> list[str]:
    return [PREPROC[p]["python"], "-I", "-B", os.path.abspath(__file__), "prep", "--preprocess", p, "--manifest",
            manifest, "--val-dir", val_dir, "--start", str(start), "--count", str(count), "--out", out]


def child_env() -> dict:
    env = dict(os.environ)
    env.update(SINGLE_THREAD_ENV)
    return env


def cmd_verify_prep(a) -> int:
    """This tool's preprocessing path vs the recorded 5,000-image tensors (original-resolution ImageNet-100 images)."""
    out_dir = os.path.join(a.work, "verify")
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for p in a.preprocess or list(PREPROC):
        d = os.path.join(ROOT, ".cache", "data", EVAL5000[p])
        man = json.load(open(os.path.join(d, "manifest.json")))
        ent = man["preprocessing"][p]
        sub = os.path.join(out_dir, f"{p}_manifest.json")
        json.dump({k: v for k, v in man.items() if k != "samples"} | {"samples": man["samples"][:a.n]}, open(sub, "w"))
        out = os.path.join(out_dir, f"{p}.f32")
        subprocess.run(prep_cmd(p, sub, a.ref_val_dir, 0, a.n, out), check=True, env=child_env())
        mine = np.fromfile(out, np.float32)
        theirs = np.fromfile(os.path.join(d, ent["file"]), np.float32, count=mine.size)
        res[p] = {"n": a.n, "identical": bool(np.array_equal(mine, theirs)),
                  "max_abs_diff": float(np.max(np.abs(mine - theirs))), "reference": os.path.relpath(d, ROOT)}
        print(p, res[p], flush=True)
        os.remove(out)
    json.dump(res, open(os.path.join(out_dir, "verify_prep.json"), "w"), indent=1)
    return 0 if all(r["identical"] for r in res.values()) else 1


# ------------------------------------------------------------------------------------------------- configurations
def imagenet_models() -> list[dict]:
    pub = json.load(open(os.path.join(ROOT, "registry", "published_accuracy.json")))["models"]
    rec = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    out = []
    for m, v in pub.items():
        if v.get("dataset") != "imagenet":
            continue
        e = rec[m]["eval"]
        assert e["input_shape"] == [1, 224, 224, 3], (m, e["input_shape"])
        out.append({"model": m, "preprocess": e["preprocess"], "eval_dir": e["dir"], "published_top1": v["top1"],
                    "published_note": v.get("note"), "published_source": v.get("source_text")})
    return out


def configs(passes: list[str], only: list[str] | None = None) -> list[dict]:
    """(pass, model, file, stage, shards-selector) in scheduling order."""
    ms = [m for m in imagenet_models() if not only or m["model"] in only]
    out = []
    for ps in passes:
        for m in ms:
            name = m["model"]
            fl = f"artifacts/{name}/model.bitino"
            i8 = f"artifacts/{name}/model_int8_full.bitino"
            has8 = os.path.exists(os.path.join(ROOT, i8))
            fp16 = name.endswith("-fp16")
            if ps == "pilot":  # first shard, every configuration (the pilot is shard 0 of the full run)
                out.append(dict(m, pass_=ps, file=fl, stage="all", shards="pilot"))
                if has8:
                    out.append(dict(m, pass_=ps, file=i8, stage="int8_full", shards="pilot"))
                if name in HEADLINE:
                    out.append(dict(m, pass_=ps, file=fl, stage="neon", shards="pilot"))
            elif ps == "float":
                out.append(dict(m, pass_=ps, file=fl, stage="all", shards="all"))
            elif ps == "neon" and name in HEADLINE:  # full set where the exact stage costs <= 3x `all` (pilot)
                out.append(dict(m, pass_=ps, file=fl, stage="neon", shards="all" if name in NEON_FULL else "neon"))
            elif ps == "int8_headline" and has8 and name in HEADLINE:
                out.append(dict(m, pass_=ps, file=i8, stage="int8_full", shards="all"))
            elif ps == "int8_rest" and has8 and name not in HEADLINE and not fp16:
                out.append(dict(m, pass_=ps, file=i8, stage="int8_full", shards="all"))
            elif ps == "int8_fp16" and has8 and fp16:
                out.append(dict(m, pass_=ps, file=i8, stage="int8_full", shards="all"))
    return out


def cfg_key(c: dict) -> str:
    return f"{c['model']}__{c['stage']}"


# ------------------------------------------------------------------------------------------------------- the run
class Runner:
    def __init__(self, a):
        self.a = a
        self.man = json.load(open(a.manifest))
        self.n = len(self.man["samples"]) if a.limit is None else min(a.limit, len(self.man["samples"]))
        self.nshards = math.ceil(self.n / a.shard)
        self.labels = np.array([s["label"] for s in self.man["samples"]], np.int32)
        self.work = a.work
        self.tdir = os.path.join(self.work, "tensors")
        self.rdir = os.path.join(self.work, "results")
        os.makedirs(self.tdir, exist_ok=True)
        os.makedirs(self.rdir, exist_ok=True)
        self.log = open(os.path.join(self.work, "run.log"), "a")
        self.bench_sha = sha256_file(a.bench)
        self.file_sha = {}

    def workers(self) -> int:
        """--workers, overridable while running by an integer in <work>/workers (e.g. to share the CPUs)."""
        try:
            w = int(open(os.path.join(self.work, "workers")).read().strip())
            if w != getattr(self, "_w", None):
                self.say(f"workers -> {w} (from {os.path.join(self.work, 'workers')})")
            self._w = w
            return max(0, min(w, 8))
        except (OSError, ValueError):
            return self.a.workers

    def say(self, msg: str):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def shard_range(self, s: int) -> tuple[int, int]:
        lo = s * self.a.shard
        return lo, min(self.n, lo + self.a.shard) - lo

    def shards_for(self, sel: str) -> list[int]:
        if sel == "pilot":
            return [0]
        if sel == "neon":
            return list(range(min(self.nshards, math.ceil(self.a.neon_images / self.a.shard))))
        return list(range(self.nshards))

    def tensor_path(self, p: str, s: int) -> str:
        return os.path.join(self.tdir, f"{p}_s{s:03d}.f32")

    def result_path(self, c: dict, s: int) -> str:
        return os.path.join(self.rdir, cfg_key(c), f"s{s:03d}.npz")

    def disk_free_gb(self) -> float:
        st = os.statvfs(self.work)
        return st.f_bavail * st.f_frsize / 1e9

    def run_passes(self, passes: list[str]) -> None:
        for ps in passes:
            jobs = [(s, c) for c in configs([ps], self.a.models) for s in self.shards_for(c["shards"])
                    if not os.path.exists(self.result_path(c, s))]
            jobs.sort(key=lambda j: j[0])  # shard-major: a shard's tensors are freed as soon as possible
            self.say(f"pass {ps}: {len(jobs)} (shard, config) results to compute")
            if jobs:
                self.schedule(jobs, ps)
            self.cleanup_tensors(keep=set())
            if self.a.summarize_after:
                summarize(self.a)

    def cleanup_tensors(self, keep: set) -> None:
        for f in os.listdir(self.tdir):
            if f.endswith(".f32"):
                p, s = f[:-4].rsplit("_s", 1)
                if (p, int(s)) not in keep:
                    os.remove(os.path.join(self.tdir, f))

    def schedule(self, jobs: list[tuple[int, dict]], ps: str) -> None:
        a = self.a
        pending = list(jobs)
        running = {}  # pid -> (kind, info, Popen, t0)
        prepping = set()
        ready = {(f[:-4].rsplit("_s", 1)[0], int(f[:-4].rsplit("_s", 1)[1])) for f in os.listdir(self.tdir)
                 if f.endswith(".f32") and os.path.exists(os.path.join(self.tdir, f + ".json"))}
        failed = []
        while pending or running:
            # free tensors that no pending/running job needs
            need = {(c["preprocess"], s) for s, c in pending} | {(i[1]["preprocess"], i[0]) for k, i, _, _ in
                                                                 running.values() if k == "bench"}
            for t in list(ready):
                if t not in need:
                    for ext in ("", ".json"):
                        try:
                            os.remove(self.tensor_path(*t) + ext)
                        except FileNotFoundError:
                            pass
                    ready.discard(t)
            # launch: preparations first (short; they keep the next tensors ready), then bench jobs
            while len(running) < self.workers() and pending:
                shards_pending = sorted({s for s, _ in pending})
                window = set(shards_pending[:a.lookahead])
                want = []
                for s, c in pending:
                    t = (c["preprocess"], s)
                    if s in window and t not in ready and t not in prepping and t not in want:
                        want.append(t)
                n_prep = sum(1 for k, *_ in running.values() if k == "prep")
                need_gb = self.a.shard * REC_BYTES / 1e9
                disk_ok = self.disk_free_gb() - need_gb > a.min_free_gb
                if want and n_prep < a.prep_workers and disk_ok:
                    prepping.add(want[0])
                    self.launch_prep(want[0], running)
                    continue
                job = next((j for j, (s, c) in enumerate(pending) if (c["preprocess"], s) in ready), None)
                if job is not None:  # earliest shard first
                    s, c = pending.pop(job)
                    self.launch_bench(s, c, running)
                    continue
                if want and not disk_ok and not running:
                    self.say(f"disk guard: {self.disk_free_gb():.1f} GB free (< {a.min_free_gb} GB after the next "
                             f"tensor); waiting")
                    time.sleep(120)
                    continue
                break
            if not running and pending and self.workers() == 0:  # paused via <work>/workers
                time.sleep(60)
                continue
            if not running:
                if pending:
                    self.say(f"ERROR: {len(pending)} jobs cannot be scheduled (failed preprocessing?)")
                    failed.extend(pending)
                break
            pid, status, ru = os.wait4(-1, 0)
            if pid not in running:
                continue
            kind, info, proc, t0 = running.pop(pid)
            wall = time.time() - t0
            rc = os.waitstatus_to_exitcode(status)
            proc.returncode = rc  # reaped here by wait4
            err = b""
            if proc.stderr:
                err = proc.stderr.read()
                proc.stderr.close()
            if proc.stdout:
                proc.stdout.close()
            if kind == "prep":
                prepping.discard(info)
                if rc == 0:
                    ready.add(info)
                    self.record_tensor(info, ru.ru_utime + ru.ru_stime, wall)
                else:
                    self.say(f"ERROR prep {info} rc={rc}; dropping its jobs")
                    failed.extend([(s, c) for s, c in pending if (c["preprocess"], s) == info])
                    pending = [(s, c) for s, c in pending if (c["preprocess"], s) != info]
            else:
                s, c, outp = info
                if rc == 0:
                    self.finish_bench(s, c, outp, ru.ru_utime + ru.ru_stime, wall)
                else:
                    self.say(f"ERROR bench {cfg_key(c)} shard {s} rc={rc}: {err.decode(errors='replace')[-500:]}")
                    failed.append((s, c))
                if os.path.exists(outp):
                    os.remove(outp)
        if failed:
            self.say(f"pass {ps}: {len(failed)} jobs failed")

    def launch_prep(self, t, running) -> None:
        p, s = t
        lo, cnt = self.shard_range(s)
        out = self.tensor_path(p, s)
        with open(os.path.join(self.work, "prep.log"), "a") as log:
            proc = subprocess.Popen(prep_cmd(p, self.a.manifest, self.a.val_dir, lo, cnt, out), env=child_env(),
                                    stdout=log, stderr=log)
        running[proc.pid] = ("prep", t, proc, time.time())
        self.say(f"prep {p} shard {s} [{lo}, {lo + cnt})")

    def launch_bench(self, s: int, c: dict, running) -> None:
        tmp = os.path.join(self.work, "out")
        os.makedirs(tmp, exist_ok=True)
        outp = os.path.join(tmp, f"{cfg_key(c)}_s{s:03d}.f32")
        cmd = [self.a.bench, "--model", os.path.join(ROOT, c["file"]), "--inputs", self.tensor_path(c["preprocess"], s),
               "--outputs", outp, "--stage", c["stage"], "--threads", "1", "--warmup", "0", "--iterations", "1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        running[proc.pid] = ("bench", (s, c, outp), proc, time.time())

    def record_tensor(self, t, cpu: float, wall: float) -> None:
        p, s = t
        ent = json.load(open(self.tensor_path(p, s) + ".json"))
        rec = os.path.join(self.work, "tensor_sha", p)
        os.makedirs(rec, exist_ok=True)
        rp = os.path.join(rec, f"s{s:03d}.json")
        if os.path.exists(rp):  # re-prepared in a later pass: must be bit-identical
            old = json.load(open(rp))
            if old["sha256"] != ent["sha256"]:
                self.say(f"ERROR: {p} shard {s} tensor differs from the earlier preparation "
                         f"({old['sha256'][:16]} vs {ent['sha256'][:16]})")
                ent["mismatch_with_previous"] = old["sha256"]
        ent["prep_cpu_s"] = cpu
        ent["prep_wall_s"] = wall
        json.dump(ent, open(rp, "w"), indent=1)
        self.say(f"ready {p} shard {s} ({wall:.0f} s, sha {ent['sha256'][:12]})")

    def finish_bench(self, s: int, c: dict, outp: str, cpu: float, wall: float) -> None:
        lo, cnt = self.shard_range(s)
        raw = np.fromfile(outp, np.float32)
        assert raw.size % cnt == 0, (outp, raw.size, cnt)
        logits = raw.reshape(cnt, -1)[:, :N_CLASSES]  # Larq models emit 2 outputs: the first 1000 values are used
        top5i = np.argsort(-logits, axis=1, kind="stable")[:, :5].astype(np.int32)  # as tools/accuracy_eval.py
        assert np.array_equal(top5i[:, 0], logits.argmax(1))
        top5v = np.take_along_axis(logits, top5i, 1).astype(np.float32)
        fp = os.path.join(ROOT, c["file"])
        if fp not in self.file_sha:
            self.file_sha[fp] = (sha256_file(fp), bf_header(fp).get("data_sha256"))
        tens = json.load(open(self.tensor_path(c["preprocess"], s) + ".json"))
        d = os.path.dirname(self.result_path(c, s))
        os.makedirs(d, exist_ok=True)
        tmp = self.result_path(c, s) + ".tmp.npz"
        np.savez(tmp, top5i=top5i, top5v=top5v, start=lo, count=cnt, outputs_per_sample=raw.size // cnt,
                 nonfinite=int((~np.isfinite(logits)).sum()),
                 meta=json.dumps({"model": c["model"], "file": c["file"], "stage": c["stage"],
                                  "preprocess": c["preprocess"], "file_sha256": self.file_sha[fp][0],
                                  "data_sha256": self.file_sha[fp][1], "bench": os.path.relpath(self.a.bench, ROOT),
                                  "bench_sha256": self.bench_sha, "tensor_sha256": tens["sha256"],
                                  "cpu_s": cpu, "wall_s": wall, "threads": 1,
                                  "date": datetime.datetime.now().isoformat(timespec="seconds")}))
        os.replace(tmp, self.result_path(c, s))
        acc = float(np.mean(top5i[:, 0] == self.labels[lo:lo + cnt]))
        self.say(f"done {cfg_key(c)} shard {s}: top1 {100 * acc:.1f} ({cpu / cnt * 1000:.0f} ms cpu/img, {wall:.0f} s)")


def cmd_run(a) -> int:
    for p, spec in PREPROC.items():
        assert os.path.exists(spec["python"]), spec["python"]
    r = Runner(a)
    r.say(f"run: n={r.n} shards={r.nshards}x{a.shard} workers={a.workers} passes={a.passes} val_dir={a.val_dir} "
          f"bench sha {r.bench_sha[:16]}")
    t0 = time.time()
    r.run_passes(a.passes.split(","))
    r.say(f"run finished in {(time.time() - t0) / 3600:.2f} h")
    return 0


# --------------------------------------------------------------------------------------------------- summarizing
def load_config(rdir: str, key: str, man_n: int):
    d = os.path.join(rdir, key)
    if not os.path.isdir(d):
        return None
    parts = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            z = np.load(os.path.join(d, f))
            parts.append((int(z["start"]), int(z["count"]), z["top5i"], json.loads(str(z["meta"])), int(z["nonfinite"])))
    if not parts:
        return None
    parts.sort(key=lambda x: x[0])
    return parts


def summarize(a) -> int:
    man = json.load(open(a.manifest))
    labels = np.array([s["label"] for s in man["samples"]], np.int32)
    n_full = len(labels)
    rdir = os.path.join(a.work, "results")
    os.makedirs(a.out, exist_ok=True)
    man_sha = sha256_file(a.manifest)
    models = imagenet_models()
    stats_by = {}
    tensor_sha = {}
    tsd = os.path.join(a.work, "tensor_sha")
    for p in os.listdir(tsd) if os.path.isdir(tsd) else []:
        for f in sorted(os.listdir(os.path.join(tsd, p))):
            e = json.load(open(os.path.join(tsd, p, f)))
            tensor_sha.setdefault(p, {})[f[:-5]] = e

    def gather(key, rd=rdir):
        parts = load_config(rd, key, n_full)
        if parts is None:
            return None
        idx = np.concatenate([np.arange(s, s + c) for s, c, *_ in parts])
        top5 = np.concatenate([t for _, _, t, _, _ in parts])
        metas = [m for *_, m, _ in parts]
        return {"idx": idx, "top5": top5, "metas": metas, "nonfinite": sum(x[4] for x in parts)}

    def acc_block(g):
        n = len(g["idx"])
        lab = labels[g["idx"]]
        t1 = g["top5"][:, 0] == lab
        t5 = (g["top5"] == lab[:, None]).any(1)
        return t1, t5, {"n": int(n), "top1": float(t1.mean()), "top1_ci95": wilson(int(t1.sum()), n),
                        "top5": float(t5.mean()), "top5_ci95": wilson(int(t5.sum()), n)}

    def paired(ta, tb, pa, pb):
        n = len(ta)
        b_only_a = int(np.sum(ta & ~tb))
        b_only_b = int(np.sum(~ta & tb))
        diff = tb.astype(float) - ta.astype(float)
        se = diff.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        return {"n": int(n), "top1_diff_b_minus_a": float(diff.mean()),
                "diff_ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                "a_correct_b_wrong": b_only_a, "a_wrong_b_correct": b_only_b,
                "mcnemar_exact_p": mcnemar_exact(b_only_a, b_only_b),
                "top1_prediction_agreement": float(np.mean(pa == pb))}

    def file_info(metas):
        m0 = metas[0]
        assert all(m["file_sha256"] == m0["file_sha256"] for m in metas), "results from different files mixed"
        return {"file": m0["file"], "file_sha256": m0["file_sha256"], "data_sha256": m0["data_sha256"],
                "stage": m0["stage"], "runner": m0["bench"], "runner_sha256": sorted({m["bench_sha256"] for m in metas}),
                "threads": 1, "cpu_s_per_image": sum(m["cpu_s"] for m in metas) / 1.0}

    summary = {"schema": "bitino.accuracy_full50k.v1", "generated": datetime.date.today().isoformat(),
               "generated_by": "tools/accuracy_full_val.py summarize",
               "eval_set": {"name": man["name"], "n": n_full, "manifest": os.path.relpath(a.manifest, ROOT),
                            "manifest_sha256": man_sha, "source": man["source"], "limitations": man["limitations"]},
               "models": {}}
    for m in models:
        name = m["model"]
        prep = m["preprocess"]
        row = {"preprocess": prep, "published_top1": m["published_top1"], "headline": name in HEADLINE}
        gf = gather(f"{name}__all")
        g8 = gather(f"{name}__int8_full")
        gn = gather(f"{name}__neon")
        base = {"model": name, "eval_set": man["name"], "manifest_sha256": man_sha, "preprocess": prep,
                "preprocess_definition": None, "published": {"top1": m["published_top1"], "note": m["published_note"],
                                                             "source_text": m["published_source"]}}
        tsh = tensor_sha.get(prep, {})
        if tsh:
            any_e = next(iter(tsh.values()))
            base["preprocess_definition"] = any_e["definition"]
            base["preprocess_script"] = any_e["script"]
        res_f = None
        for key, g in (("all", gf), ("int8_full", g8), ("neon", gn)):
            if g is None:
                continue
            t1, t5, acc = acc_block(g)
            fi = file_info(g["metas"])
            fi["cpu_s_per_image"] = fi["cpu_s_per_image"] / acc["n"]
            shard_starts = sorted({int(i) for i in g["idx"][::a.shard]})
            out = dict(base, configuration=key, **fi, **acc, complete=acc["n"] == n_full,
                       nonfinite_logits=g["nonfinite"],
                       samples=("all" if acc["n"] == n_full else f"{acc['n']} samples: shards starting at {shard_starts}"),
                       tensor_sha256_per_shard={k: v["sha256"] for k, v in tsh.items()
                                                if int(k[1:]) * a.shard in set(g["idx"][::a.shard].tolist())})
            if key == "all":
                out["published_minus_measured_pt"] = round(m["published_top1"] - 100 * acc["top1"], 3)
                out["published_in_ci95"] = bool(acc["top1_ci95"][0] * 100 <= m["published_top1"] <= acc["top1_ci95"][1] * 100)
                res_f = (g, t1)
            if key in ("int8_full", "neon") and res_f is not None:
                gfa, t1f = res_f
                common, ia, ib = np.intersect1d(gfa["idx"], g["idx"], return_indices=True)
                if len(common):
                    pr = paired(t1f[ia], t1[ib], gfa["top5"][ia, 0], g["top5"][ib, 0])
                    pr["a"] = f"{name} model.bitino:all"
                    pr["b"] = f"{name} {os.path.basename(fi['file'])}:{key}"
                    pr["a_top1_on_common"] = float(t1f[ia].mean())
                    pr["b_top1_on_common"] = float(t1[ib].mean())
                    out["paired_vs_all"] = pr
            json.dump(out, open(os.path.join(a.out, f"{name}__{key}.json"), "w"), indent=1)
            row[key] = {k: out[k] for k in ("n", "top1", "top1_ci95", "top5", "top5_ci95", "data_sha256", "complete",
                                            "cpu_s_per_image")}
            if "paired_vs_all" in out:
                row[key]["paired_vs_all"] = out["paired_vs_all"]
        # the same configuration on the other copy of the images (original JPEGs vs the resized copy)
        if a.other_work and os.path.isdir(os.path.join(a.other_work, "results")) and gf is not None:
            go = gather(f"{name}__all", os.path.join(a.other_work, "results"))
            if go is not None:
                t1o, _, acco = acc_block(go)
                common, ia, ib = np.intersect1d(gf["idx"], go["idx"], return_indices=True)
                t1f = gf["top5"][:, 0] == labels[gf["idx"]]
                pr = paired(t1f[ia], t1o[ib], gf["top5"][ia, 0], go["top5"][ib, 0])
                pr["a"] = f"this copy ({os.path.relpath(a.manifest, ROOT)})"
                pr["b"] = f"other copy ({os.path.relpath(a.other_work, ROOT)})"
                pr["a_top1_on_common"] = float(t1f[ia].mean())
                pr["b_top1_on_common"] = float(t1o[ib].mean())
                row["other_copy_all"] = dict(acco, complete=acco["n"] == n_full, paired_vs_this_copy=pr)
        summary["models"][name] = row
    if a.other_work:
        summary["other_copy"] = {"work": os.path.relpath(a.other_work, ROOT), "label": a.other_label}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    write_md(summary, os.path.join(a.out, "summary.md"), a)
    print(f"wrote {len(summary['models'])} models -> {a.out}")
    return 0


def cmd_compare_copies(a) -> int:
    """Paired comparison of the same configuration on two copies of the same images (e.g. the resized copy vs the
    original-resolution JPEGs): same file names and labels, different JPEG bytes. Writes --json."""
    ma = json.load(open(a.manifest))["samples"]
    mb = json.load(open(a.manifest_b))["samples"]
    n = min(len(mb), len(ma))
    assert [(x["file"], x["label"]) for x in ma[:n]] == [(x["file"], x["label"]) for x in mb[:n]]
    labels = np.array([x["label"] for x in ma[:n]])
    out = {"a": {"manifest": os.path.relpath(a.manifest, ROOT), "results": os.path.relpath(a.work, ROOT)},
           "b": {"manifest": os.path.relpath(a.manifest_b, ROOT), "results": os.path.relpath(a.work_b, ROOT)},
           "n": n, "configs": {}}
    for key in sorted(os.listdir(os.path.join(a.work_b, "results"))):
        pa = load_config(os.path.join(a.work, "results"), key, n)
        pb = load_config(os.path.join(a.work_b, "results"), key, n)
        if not pa or not pb:
            continue
        ta = np.concatenate([t for _, _, t, _, _ in pa])[:n]
        tb = np.concatenate([t for _, _, t, _, _ in pb])[:n]
        if len(ta) < n or len(tb) < n:
            continue
        ca, cb = ta[:, 0] == labels, tb[:, 0] == labels
        b_only_a, b_only_b = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        diff = cb.astype(float) - ca.astype(float)
        se = diff.std(ddof=1) / math.sqrt(n)
        out["configs"][key] = {"top1_a": float(ca.mean()), "top1_b": float(cb.mean()),
                               "top1_diff_b_minus_a": float(diff.mean()),
                               "diff_ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                               "a_correct_b_wrong": b_only_a, "a_wrong_b_correct": b_only_b,
                               "mcnemar_exact_p": mcnemar_exact(b_only_a, b_only_b),
                               "top1_prediction_agreement": float(np.mean(ta[:, 0] == tb[:, 0]))}
    d = np.array([v["top1_diff_b_minus_a"] for v in out["configs"].values()])
    out["mean_diff_over_configs"] = float(d.mean()) if len(d) else None
    json.dump(out, open(a.json, "w"), indent=1)
    for k, v in out["configs"].items():
        print(f"{k:50s} a {100 * v['top1_a']:6.2f} b {100 * v['top1_b']:6.2f} d {100 * v['top1_diff_b_minus_a']:+6.2f} "
              f"p {v['mcnemar_exact_p']:.2g} agree {100 * v['top1_prediction_agreement']:.1f}")
    print("mean diff (pt)", None if out["mean_diff_over_configs"] is None else 100 * out["mean_diff_over_configs"])
    return 0


REBUILD = r"""
P="~/anaconda3/bin/python -I -B"; R=<directory holding imagenet/val>
$P tools/accuracy_full_val.py verify-prep          # 7 preprocessings == recorded eval5000 tensors, bit for bit
$P tools/accuracy_full_val.py manifest --val-dir $R/imagenet_resize/val      # registry/eval_sets/imagenet-val-50000.json
# resized copy (the registry manifest): float, exact slice, full-integer
scripts/capped.sh 24G nice -n 10 $P tools/accuracy_full_val.py run --workers 8 --val-dir $R/imagenet_resize/val \
    --passes float,neon,int8_headline,int8_rest,int8_fp16
# original JPEGs (diagnostic; stage all)
$P tools/accuracy_full_val.py manifest --copy original --val-dir $R/imagenet/val --out .cache/full50k_orig/manifest50k.json
scripts/capped.sh 7G nice -n 10 $P tools/accuracy_full_val.py run --workers 3 --passes float \
    --manifest .cache/full50k_orig/manifest50k.json --work .cache/full50k_orig \
    --out reports/accuracy_full50k/original_resolution --val-dir $R/imagenet/val
$P tools/accuracy_full_val.py summarize --other-work .cache/full50k_orig
$P tools/accuracy_full_val.py summarize --manifest .cache/full50k_orig/manifest50k.json --work .cache/full50k_orig \
    --out reports/accuracy_full50k/original_resolution --other-work .cache/full50k --other-label "resized copy"
"""


def pct(x):
    return "–" if x is None else f"{100 * x:.2f}"


def write_md(s: dict, path: str, a) -> None:
    L = [f"# Full ImageNet-1k validation accuracy ({s['eval_set']['n']:,} images)", "",
         f"Generated {s['generated']} by `tools/accuracy_full_val.py summarize` from `{a.work}/results/`. "
         f"Eval set `{s['eval_set']['name']}` (`{s['eval_set']['manifest']}`, SHA-256 `{s['eval_set']['manifest_sha256'][:16]}…`). "
         "Host runtime `build/host/bitino_bench`, 1 thread per process. 95 % Wilson intervals; int8 Δ = paired "
         "difference int8_full − all on the same images, 95 % CI, exact McNemar p.", "",
         f"**Data caveat.** {s['eval_set']['limitations']}", "",
         "| Model | Preprocessing | Published | all top-1 [95 % CI] | top-5 | Δ pub (pt) | int8_full top-1 | Δ int8 (pt) [95 % CI] | McNemar p | n |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in s["models"].items():
        f = r.get("all")
        i = r.get("int8_full")
        hl = " ★" if r["headline"] else ""
        if f:
            ci = f"{pct(f['top1'])} [{100 * f['top1_ci95'][0]:.1f}, {100 * f['top1_ci95'][1]:.1f}]"
            dpub = f"{100 * f['top1'] - r['published_top1']:+.2f}"
            t5 = pct(f["top5"])
            n = f"{f['n']:,}"
        else:
            ci, dpub, t5, n = "–", "–", "–", "0"
        if i:
            p = i.get("paired_vs_all")
            i1 = pct(i["top1"]) + ("" if i["complete"] else f" (n={i['n']:,})")
            di = (f"{100 * p['top1_diff_b_minus_a']:+.2f} [{100 * p['diff_ci95'][0]:+.2f}, {100 * p['diff_ci95'][1]:+.2f}]"
                  if p else "–")
            mp = f"{p['mcnemar_exact_p']:.2g}" if p else "–"
        else:
            i1, di, mp = ("no file" if not os.path.exists(os.path.join(ROOT, f"artifacts/{name}/model_int8_full.bitino"))
                          else "not run"), "–", "–"
        L.append(f"| `{name}`{hl} | {r['preprocess']} | {r['published_top1']:.2f} | {ci} | {t5} | {dpub} | {i1} | {di} | {mp} | {n} |")
    L += ["", "★ headline network (paper Table tab:acc).", "",
          "## Exact stage (`neon`) vs `all` on a slice", "",
          "| Model | n | neon top-1 | all top-1 (same images) | top-1 agreement | Δ neon − all (pt) [95 % CI] | McNemar p | cpu s/img neon / all |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in s["models"].items():
        nn = r.get("neon")
        if not nn or "paired_vs_all" not in nn:
            continue
        p = nn["paired_vs_all"]
        ratio = f"{nn['cpu_s_per_image']:.3f} / {r['all']['cpu_s_per_image']:.3f}" if r.get("all") else "–"
        L.append(f"| `{name}` | {p['n']:,} | {pct(p['b_top1_on_common'])} | {pct(p['a_top1_on_common'])} | "
                 f"{100 * p['top1_prediction_agreement']:.2f} % | {100 * p['top1_diff_b_minus_a']:+.2f} "
                 f"[{100 * p['diff_ci95'][0]:+.2f}, {100 * p['diff_ci95'][1]:+.2f}] | {p['mcnemar_exact_p']:.2g} | {ratio} |")
    if s.get("other_copy"):
        lab = s["other_copy"]["label"]
        L += ["", f"## Stage `all`: this copy vs {lab}", "",
              f"Same file names and labels; `{s['other_copy']['work']}` holds the other copy's results. Paired on the "
              "images both runs have.", "",
              f"| Model | Published | this copy | {lab} | n | Δ (other − this, pt) [95 % CI] | McNemar p | top-1 agreement | Δ pub (other) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        ds = []
        for name, r in s["models"].items():
            o = r.get("other_copy_all")
            if not o:
                continue
            p = o["paired_vs_this_copy"]
            ds.append(p["top1_diff_b_minus_a"])
            L.append(f"| `{name}` | {r['published_top1']:.2f} | {pct(p['a_top1_on_common'])} | {pct(p['b_top1_on_common'])} | "
                     f"{p['n']:,} | {100 * p['top1_diff_b_minus_a']:+.2f} [{100 * p['diff_ci95'][0]:+.2f}, "
                     f"{100 * p['diff_ci95'][1]:+.2f}] | {p['mcnemar_exact_p']:.2g} | {100 * p['top1_prediction_agreement']:.1f} % | "
                     f"{100 * p['b_top1_on_common'] - r['published_top1']:+.2f} |")
        if ds:
            L += ["", f"Mean Δ over {len(ds)} models: {100 * float(np.mean(ds)):+.2f} pt."]
    notes = os.path.join(a.out, "NOTES.md")
    if os.path.exists(notes):
        L += ["", open(notes).read().rstrip()]
    L += ["", "## Rebuild", "", "```", REBUILD.strip(), "```", ""]
    open(path, "w").write("\n".join(L))


# ------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--val-dir", required=True)
    m.add_argument("--class-index", default=CLASS_INDEX)
    m.add_argument("--n", type=int, default=50000)
    m.add_argument("--out", default=DEFAULT_MANIFEST)
    m.add_argument("--copy", choices=["resized", "original"], default="resized",
                   help="which copy --val-dir is (only changes the source/limitations text)")
    p = sub.add_parser("prep")
    p.add_argument("--preprocess", required=True, choices=list(PREPROC))
    p.add_argument("--manifest", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", required=True)
    v = sub.add_parser("verify-prep")
    v.add_argument("--ref-val-dir", default=os.path.join(HOME, "noise/imagenet100/val"),
                   help="original-resolution images of the recorded 5,000-image tensors")
    v.add_argument("--n", type=int, default=16)
    v.add_argument("--preprocess", action="append", choices=list(PREPROC))
    v.add_argument("--work", default=DEFAULT_WORK)
    cc = sub.add_parser("compare-copies")
    cc.add_argument("--manifest", default=DEFAULT_MANIFEST)
    cc.add_argument("--work", default=DEFAULT_WORK)
    cc.add_argument("--manifest-b", required=True)
    cc.add_argument("--work-b", required=True)
    cc.add_argument("--json", required=True)
    for name in ("run", "summarize"):
        r = sub.add_parser(name)
        r.add_argument("--manifest", default=DEFAULT_MANIFEST)
        r.add_argument("--work", default=DEFAULT_WORK)
        r.add_argument("--out", default=DEFAULT_OUT)
        r.add_argument("--shard", type=int, default=1000)
        r.add_argument("--other-work", help="work dir of the same run on another copy of the images (summary table)")
        r.add_argument("--other-label", default="original JPEGs")
        if name == "run":
            r.add_argument("--val-dir", required=True)
            r.add_argument("--passes", default="pilot,float,neon,int8_headline,int8_rest,int8_fp16")
            r.add_argument("--workers", type=int, default=8, help="max concurrent processes (1 thread each); an "
                           "integer in <work>/workers overrides it while running (capped at 8)")
            r.add_argument("--prep-workers", type=int, default=2)
            r.add_argument("--lookahead", type=int, default=2, help="shards whose tensors may exist at once")
            r.add_argument("--min-free-gb", type=float, default=32.0)
            r.add_argument("--neon-images", type=int, default=2000)
            r.add_argument("--limit", type=int, help="only the first N samples")
            r.add_argument("--models", type=lambda x: x.split(","), help="comma-separated subset of models")
            r.add_argument("--bench", default=BENCH)
            r.add_argument("--no-summarize", dest="summarize_after", action="store_false")
    a = ap.parse_args()
    if a.cmd == "manifest":
        return cmd_manifest(a)
    if a.cmd == "prep":
        return cmd_prep(a)
    if a.cmd == "verify-prep":
        return cmd_verify_prep(a)
    if a.cmd == "compare-copies":
        return cmd_compare_copies(a)
    if a.cmd == "run":
        os.makedirs(a.work, exist_ok=True)
        return cmd_run(a)
    return summarize(a)


if __name__ == "__main__":
    sys.exit(main())
