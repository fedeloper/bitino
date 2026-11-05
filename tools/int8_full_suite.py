#!/usr/bin/env python3
"""Full-integer stage (int8_full) for a set of converted models: quantize, then paired accuracy vs the exact stage.

  python3 tools/int8_full_suite.py quantize --models bi-vit-deit-tiny bhvit-tiny-binary
  python3 tools/int8_full_suite.py accuracy --models bi-vit-deit-tiny --emulated build/android-arm64-static

quantize: tools/quantize_int8_full.py with the calibration set of the model's family (ImageNet-100 TRAIN images,
GLUE / SST-2 TRAIN examples, CIFAR TRAIN images; every set is disjoint from the evaluation sets, see its
manifest's `disjointness`), writes artifacts/<m>/model_int8_full.bitino and reports/int8_full/<m>.json.
accuracy: tools/accuracy_eval.py on the family's large eval set (the sets of scripts/int8_accuracy_all.sh), A =
model.bitino at stage neon (exact, host runtime), B = model_int8_full.bitino at stage int8_full; with --emulated
the B side runs the static ARM build under qemu-aarch64 (the NEON kernels the phone runs; at most 6 processes).
Results: reports/accuracy/<m>_exact_vs_int8_full_<eval>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMNET = ".cache/data/calib512_imagenet100train"


def calib_of(m: str):
    """(calibration dir, preprocessing key) of a model family."""
    if m.startswith(("bi-vit-", "bhvit-")):
        return IMNET, "bhvit"
    if m == "bibert-bert-base-sst2":
        return ".cache/data/sst2_calib512", "bibert"
    if m.startswith("bibert-bert-base-"):
        return f".cache/data/glue_calib512/{m.rsplit('-', 1)[1]}", "bibert"
    if m.startswith(("larq-zoo-", "larqface-")):
        return IMNET, "larq_zoo"
    if m.startswith("bnext-"):
        return IMNET, "timm_bicubic_crop095"
    if m.startswith("bitorch-"):
        return IMNET, "bitorch_imagenet"
    if m.startswith("omz-"):
        return IMNET, "omz_bgr"
    if m.startswith("tensorpack-"):
        return IMNET, "tensorpack_bgr"
    if m.startswith(("ab-bnn-imagenet", "rbonn-", "sa-bnn-", "ida-det-", "dabnn-", "nncf-", "bimatting-")) or \
            (m.endswith("-imagenet") and m.startswith(("recu-", "rbnn-", "siman-", "biper-", "snn-subbit-", "bnas-"))):
        return IMNET, "torchvision_imagenet"
    if m.startswith(("recu-", "rbnn-", "siman-", "biper-", "adabin-", "fda-bnn-", "ab-bnn-cifar10-")) and "cifar10" in m:
        return ".cache/data/calib512_cifar10train", "cifar10_imagenet_norm"
    if m == "bnas-a-cifar10":
        return ".cache/data/calib512_cifar10train", "cifar10_darts_norm"
    if m == "shuaiz-bireal20-cifar10":
        return ".cache/data/calib512_cifar10train", "cifar10_bireal_norm"
    if m.startswith(("brevitas-bnn-pynq-cnv-", "aliasgharshinwari-finn-cnv-")):
        return ".cache/data/calib512_cifar10train", "cifar10_totensor"
    if m.startswith("finn-examples-bnn-pynq-cnv-"):
        return ".cache/data/calib512_cifar10train", "cifar10_pixels"
    if m == "bnn-pynq-cnv-w1a1-cifar10":
        return ".cache/data/calib512_cifar10train", "cifar10_bnnpynq"
    if m.startswith("ab-bnn-cifar100-"):
        return ".cache/data/calib512_cifar100train", "cifar100_imagenet_norm"
    if m.startswith("gsb-deit-s-cifar100"):
        return ".cache/data/calib512_cifar100train", "cifar100_deit"
    raise SystemExit(f"{m}: no calibration set for this family")


def big_eval_of(m: str):
    """(eval dir, preprocessing key, classes, tag) -- the sets of scripts/int8_accuracy_all.sh."""
    out = subprocess.run(["bash", "-c", "eval \"$(sed -n '/^big_eval_of()/,/^}/p' scripts/int8_accuracy_all.sh)\"; big_eval_of \"$0\"", m],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    if len(out) != 4:
        raise SystemExit(f"{m}: no large eval set")
    return out[0], out[1], int(out[2]), out[3]


def quantize(m: str, a) -> None:
    cdir, key = calib_of(m)
    out = os.path.join(ROOT, "artifacts", m, "model_int8_full.bitino")
    rep = os.path.join(ROOT, "reports", "int8_full", f"{m}.json")
    os.makedirs(os.path.dirname(rep), exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "quantize_int8_full.py"), "--model", os.path.join(ROOT, "artifacts", m, "model.bitino"),
           "--calib-dir", os.path.join(ROOT, cdir), "--preprocess", key, "--n", str(a.n), "--jobs", str(a.jobs), "--out", out, "--report", rep]
    if a.percentile:
        cmd += ["--percentile", str(a.percentile)]
    cached = os.path.join(ROOT, ".cache", "int8_full_stats", f"{m}.stats.json")
    if a.reuse_stats and os.path.exists(cached):  # same calibration samples and range rule as the run that wrote it
        cmd += ["--stats", cached]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"{m}: quantize FAILED\n{r.stderr[-1500:]}")
        return
    st = out + ".stats.json"
    if os.path.exists(st) and not (a.reuse_stats and os.path.exists(cached)):
        os.makedirs(os.path.join(ROOT, ".cache", "int8_full_stats"), exist_ok=True)
        shutil.move(st, os.path.join(ROOT, ".cache", "int8_full_stats", f"{m}.stats.json"))
    j = json.load(open(rep))
    print(f"{m}: {j['nodes']} nodes (float {j['float_nodes']}), int8 ops {sum(j['int8_ops'].values())}, left float {j['left_float'] or 'none'}")


def accuracy(m: str, a, pair: bool = False) -> None:
    edir, key, ncls, tag = big_eval_of(m)
    variant = getattr(a, "variant", "int8_full")
    a_dir = getattr(a, "a_artifacts", None) or os.path.join(ROOT, "artifacts")
    if pair:  # the new variant paired against the shipped int8_full model (same eval set)
        out = os.path.join(ROOT, "reports", "accuracy", f"{m}_int8_full_vs_{variant}_{tag}.json")
        spec_a = os.path.join(ROOT, "artifacts", m, "model_int8_full.bitino") + ":int8_full"
    else:
        out = os.path.join(ROOT, "reports", "accuracy", f"{m}_exact_vs_{variant}_{tag}.json")
        spec_a = os.path.join(a_dir, m, "model.bitino") + ":neon"
    cmd = [sys.executable, os.path.join(ROOT, "tools", "accuracy_eval.py"), "--eval-dir", os.path.join(ROOT, edir), "--preprocess", key,
           "--a", spec_a,
           "--b", os.path.join(ROOT, "artifacts", m, f"model_{variant}.bitino") + ":int8_full",
           "--bench", os.path.join(ROOT, a.bench), "--classes", str(ncls), "--threads", str(a.threads),
           "--shards", str(a.shards), "--json", out]
    if getattr(a, "cache_dir", None):
        cmd += ["--cache-dir", a.cache_dir]
    env = dict(os.environ)
    if a.emulated:
        cmd += ["--bench-b", os.path.join(ROOT, "scripts", "bench_emulated.sh"), "--shards-b", str(min(6, a.shards_b))]
        env["BITINO_STATIC_BUILD"] = os.path.join(ROOT, a.emulated)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        print(f"{m}: accuracy FAILED\n{r.stderr[-1500:]}")
        return
    d = json.load(open(out))
    for k in ("a", "b"):  # repository-relative paths
        d["configs"][k]["spec"] = d["configs"][k]["spec"].replace(a_dir + os.sep, "artifacts" + os.sep).replace(ROOT + os.sep, "")
        d["configs"][k]["runner"] = d["configs"][k]["runner"].replace(ROOT + os.sep, "")
    d["int8_full_runner"] = "qemu-aarch64 static ARM build (int8_full NEON kernels)" if a.emulated else "host runtime"
    json.dump(d, open(out, "w"), indent=1)
    ca, cb, p = d["configs"]["a"], d["configs"]["b"], d["paired"]
    print(f"{m}: {'int8_full' if pair else 'exact'} {ca['top1'] * 100:.2f}% {variant} {cb['top1'] * 100:.2f}% diff "
          f"{p['top1_diff_b_minus_a'] * 100:+.2f} pt p={p['mcnemar_exact_p']:.2g} agree {p['top1_prediction_agreement'] * 100:.1f}% n={d['n']}",
          flush=True)
    if getattr(a, "pair_shipped", False) and not pair and variant != "int8_full":
        accuracy(m, a, pair=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["quantize", "accuracy"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=256, help="calibration samples")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--reuse-stats", action="store_true", help="reuse .cache/int8_full_stats/<m>.stats.json (skip calibration)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--shards-b", type=int, default=6)
    ap.add_argument("--bench", default="build/host/bitino_bench", help="host bitino_bench (A side; B side unless --emulated)")
    ap.add_argument("--emulated", help="static ARM build dir for the int8_full side (qemu-aarch64)")
    ap.add_argument("--variant", choices=("int8_full", "int8_full_log2"), default="int8_full",
                    help="accuracy: B side model_int8_full.bitino (default) or model_int8_full_log2.bitino (l8 residual stream), "
                         "written to reports/accuracy/<m>_exact_vs_<variant>_<set>.json")
    ap.add_argument("--a-artifacts", help="accuracy: directory holding <m>/model.bitino for the A side (default ROOT/artifacts, "
                                          "e.g. a separate checkout)")
    ap.add_argument("--pair-shipped", action="store_true", help="accuracy of a variant: also pair it against the shipped "
                    "model_int8_full.bitino (reports/accuracy/<m>_int8_full_vs_<variant>_<set>.json)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache", "int8_full_eval_preds"),
                    help="accuracy_eval.py --cache-dir (top-5 predictions per model file / stage / eval set; '' = off)")
    a = ap.parse_args()
    for m in a.models:
        (quantize if a.what == "quantize" else accuracy)(m, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
