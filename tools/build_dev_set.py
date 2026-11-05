#!/usr/bin/env python3
"""Labeled development sets from TRAINING images, disjoint from the int8_full calibration sets and the eval sets.

Used to choose quantization schemes without looking at the evaluation sets (docs/int8.md, int8_full section,
"Investigation: binary CNNs"). Two families:

  imagenet: ImageNet-100 TRAIN split (<synset>/<n*_*.JPEG>); sample i = class i % 100 (sorted synsets), file
            --offset + i // 100 (sorted names). The calibration set uses files 0..5 of every class
            (calib512_imagenet100train: file i // 100, i < 512), so --offset >= 6 is disjoint by construction;
            the build also checks file names and JPEG SHA-256 against the calibration manifest and every
            --disjoint-from eval manifest and fails on any overlap. Preprocessings: torchvision_imagenet,
            bitorch_imagenet (std 0.255 typo of bitorch, tools/eval_sets/bitorch_imagenet.py), bhvit (bicubic
            Resize(256), timm mean/std; built with the base env's torchvision, not the BHViT repo env), larq_zoo
            (larq_zoo.training.data.preprocess_image_bytes, the calibration / eval sets' function; run this
            script with the larq env's python, e.g. .cache/venv-larq/bin/python, and only --preprocess larq_zoo).
  cifar10:  CIFAR-10 TRAIN rows --offset .. --offset + n - 1 (torchvision order, data_batch_1..5); calibration
            uses the first 512 rows; pixel SHA-256 checked against the calibration and eval manifests.
            Preprocessing cifar10_imagenet_norm (tools/eval_sets/cifar10_normalized.py).
  cifar100: the same for CIFAR-100 TRAIN (cifar-100-python/train, fine labels); preprocessing
            cifar100_imagenet_norm (tools/eval_sets/cifar100_normalized.py: the same normalization).

  python3 tools/build_dev_set.py imagenet --train-dir ~/noise/imagenet100/train --n 3000 --offset 100 \\
      --class-index .cache/data/imagenet_class_index.json --calib-manifest <calib512_imagenet100train/manifest.json> \\
      --disjoint-from .cache/data/eval5000/manifest.json --disjoint-from .cache/data/eval128/manifest.json --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval_sets"))

MEAN = (0.485, 0.456, 0.406)
STDS = {"torchvision_imagenet": (0.229, 0.224, 0.225), "bitorch_imagenet": (0.229, 0.224, 0.255), "bhvit": (0.229, 0.224, 0.225)}
INTERP = {"bhvit": 3}  # PIL BICUBIC (IMRL/BHViT build_transform(is_train=False), tools/eval_sets/bhvit.py); others bilinear


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def imagenet(a):
    idx = json.load(open(a.class_index))
    syn2lab = {v[0]: int(k) for k, v in idx.items()}
    classes = sorted(d for d in os.listdir(a.train_dir) if os.path.isdir(os.path.join(a.train_dir, d)))
    files = {c: sorted(os.listdir(os.path.join(a.train_dir, c))) for c in classes}
    picks = [(classes[i % len(classes)], files[classes[i % len(classes)]][a.offset + i // len(classes)]) for i in range(a.n)]
    raws = {p: open(os.path.join(a.train_dir, *p), "rb").read() for p in picks}
    samples = [{"file": f"{c}/{f}", "jpeg_sha256": sha(raws[(c, f)]), "synset": c, "label": syn2lab[c]} for c, f in picks]
    checked = []
    for mp in [a.calib_manifest] + a.disjoint_from:
        ev = json.load(open(mp))["samples"]
        hf = {s["file"] for s in samples} & {e["file"] for e in ev}
        hn = {os.path.basename(s["file"]) for s in samples} & {os.path.basename(e["file"]) for e in ev}
        hs = {s["jpeg_sha256"] for s in samples} & {e["jpeg_sha256"] for e in ev}
        if hf or hn or hs:
            raise SystemExit(f"ERROR: overlap with {mp}: {sorted(hf)[:3]} {sorted(hn)[:3]} {sorted(hs)[:3]}")
        checked.append({"manifest": mp, "manifest_sha256": sha(open(mp, "rb").read()), "samples": len(ev),
                        "overlap_file": 0, "overlap_basename": 0, "overlap_jpeg_sha256": 0})
    os.makedirs(a.out, exist_ok=True)
    pre = {}
    for key in a.preprocess:
        if key == "larq_zoo":  # needs the larq env (TensorFlow + larq_zoo), e.g. .cache/venv-larq/bin/python
            from build_eval_set import make_preprocessor
            fn, definition = make_preprocessor("larq_zoo")
            arr = np.stack([fn(raws[p]) for p in picks]).astype(np.float32)
            arr.tofile(os.path.join(a.out, key + ".f32"))
            pre[key] = {"shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": key + ".f32", "sha256": sha(arr.tobytes()),
                        "definition": definition + " (same function as calib512_imagenet100train / eval5000 larq_zoo)"}
            continue
        if key == "omz_bgr":  # OpenVINO OMZ classification preprocessing (tools/eval_sets/omz_bgr.py, the eval sets' function)
            from omz_bgr import preprocess as omz_pre
            arr = np.stack([omz_pre(os.path.join(a.train_dir, *p)) for p in picks]).astype(np.float32)
            arr.tofile(os.path.join(a.out, key + ".f32"))
            pre[key] = {"shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": key + ".f32", "sha256": sha(arr.tobytes()),
                        "definition": "tools/eval_sets/omz_bgr.preprocess (Resize 256 aspect greater, Pillow bilinear, crop 224, "
                                      "RGB to BGR, 0..255; same function as eval5000_omz_bgr)"}
            continue
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        if key == "timm_bicubic_crop095":  # BNext's eval transform (tools/eval_sets/timm_bicubic_crop095.py, same steps)
            import torch
            tf = transforms.Compose([transforms.Resize(235, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224)])
            mean = torch.tensor([x * 255 for x in MEAN], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([x * 255 for x in STDS["torchvision_imagenet"]], dtype=torch.float32).view(1, 3, 1, 1)
            xs = []
            for p in picks:
                u8 = np.asarray(tf(Image.open(io.BytesIO(raws[p])).convert("RGB")), dtype=np.uint8)
                chw = torch.from_numpy(np.ascontiguousarray(u8.transpose(2, 0, 1)))[None]
                xs.append(chw.float().sub_(mean).div_(std).permute(0, 2, 3, 1).contiguous().numpy())
            arr = np.concatenate(xs).astype(np.float32)
            arr.tofile(os.path.join(a.out, key + ".f32"))
            pre[key] = {"shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": key + ".f32", "sha256": sha(arr.tobytes()),
                        "definition": "Resize(235, BICUBIC) -> CenterCrop(224) -> uint8 -> float32 (x - 255 mean) / (255 std), "
                                      "CHW -> HWC (tools/eval_sets/timm_bicubic_crop095.py, the eval sets' steps)"}
            continue
        interp = InterpolationMode.BICUBIC if INTERP.get(key) == 3 else InterpolationMode.BILINEAR
        tf = transforms.Compose([transforms.Resize(256, interpolation=interp), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(MEAN, STDS[key])])
        arr = np.stack([tf(Image.open(io.BytesIO(raws[p])).convert("RGB")).permute(1, 2, 0).contiguous().numpy()
                        for p in picks]).astype(np.float32)
        arr.tofile(os.path.join(a.out, key + ".f32"))
        pre[key] = {"shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": key + ".f32", "sha256": sha(arr.tobytes()),
                    "definition": f"Resize(256, {interp.value}) -> CenterCrop(224) -> ToTensor() -> Normalize({MEAN}, {STDS[key]}), CHW -> HWC"}
    return samples, checked, pre, f"class i % {len(classes)} (sorted synsets), file {a.offset} + i // {len(classes)} (sorted names)"


def cifar(a):
    from _brevitas_common import to_tensor_f32
    import _cifar_train
    data, labels, names, _ = _cifar_train._load(a.family, a.train_dir)
    # rows from --offset on, skipping rows whose pixels occur in a listed manifest (CIFAR-100 train has a few images
    # identical to test images); the checks below then pass by construction and are kept as a guard
    banned = set()
    for mp in [a.calib_manifest] + a.disjoint_from:
        banned |= {e["image_sha256"] for e in json.load(open(mp))["samples"]}
    rows, skipped, r = [], [], a.offset
    while len(rows) < a.n:
        (skipped if sha(np.ascontiguousarray(data[r]).tobytes()) in banned else rows).append(r)
        r += 1
    a.skipped_rows = skipped
    sel = np.ascontiguousarray(data[rows])
    samples = [{"file": names[j], "image_sha256": sha(sel[i].tobytes()), "label": labels[j]} for i, j in enumerate(rows)]
    checked = []
    for mp in [a.calib_manifest] + a.disjoint_from:
        ev = json.load(open(mp))["samples"]
        hs = {s["image_sha256"] for s in samples} & {e["image_sha256"] for e in ev}
        hf = {s["file"] for s in samples} & {e["file"] for e in ev}
        if hs or hf:
            raise SystemExit(f"ERROR: overlap with {mp}: {len(hs)} images, {len(hf)} rows")
        checked.append({"manifest": mp, "manifest_sha256": sha(open(mp, "rb").read()), "samples": len(ev), "overlap_image_sha256": 0,
                        "overlap_row": 0})
    os.makedirs(a.out, exist_ok=True)
    t = to_tensor_f32(sel)
    pre = {}
    keys = [k for k in a.preprocess if k.startswith(a.family + "_")] or [f"{a.family}_imagenet_norm"]
    for key in keys:
        if key == f"{a.family}_imagenet_norm":
            mean, std, what = MEAN, STDS["torchvision_imagenet"], "imagenet mean/std"
        else:  # e.g. cifar10_bireal_norm (tools/eval_sets/cifar10_normalized.py VARIANTS, the eval sets' constants)
            from cifar10_normalized import VARIANTS
            mean, std, what = VARIANTS[key][0], VARIANTS[key][1], VARIANTS[key][2]
        m = np.asarray(mean, np.float32).reshape(1, 3, 1, 1)
        s = np.asarray(std, np.float32).reshape(1, 3, 1, 1)
        x = ((t - m) / s).astype(np.float32).transpose(0, 2, 3, 1).copy()
        x.tofile(os.path.join(a.out, key + ".f32"))
        pre[key] = {"shape": list(x.shape), "dtype": "float32", "layout": "NHWC", "file": key + ".f32",
                    "sha256": sha(x.tobytes()), "definition": f"ToTensor() then Normalize({what}) in float32"}
    src = "data_batch_1..5" if a.family == "cifar10" else "cifar-100-python/train, fine labels"
    sk = f", skipping rows {skipped} (pixels identical to a listed manifest's image)" if skipped else ""
    return samples, checked, pre, f"{a.family.upper()} train rows {a.offset}..{rows[-1]} (torchvision order, {src}){sk}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", choices=["imagenet", "cifar10", "cifar100"])
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--class-index")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--calib-manifest", required=True)
    ap.add_argument("--disjoint-from", action="append", default=[])
    ap.add_argument("--preprocess", nargs="+", default=["torchvision_imagenet", "bitorch_imagenet"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, checked, pre, rule = (imagenet if a.family == "imagenet" else cifar)(a)
    man = {"schema": "bitino.eval_set.v1", "name": f"{a.family}-train-dev-{a.n}", "split": "train", "purpose": "development",
           "selection_rule": rule, "disjointness": {"rule": "no sample in the calibration set or any eval manifest (file, basename, "
                                                             "image / JPEG SHA-256)", "checked": checked, "passed": True},
           "limitations": "Development set for choosing int8_full quantization schemes (labeled training images, disjoint from "
                          "the calibration samples and the eval sets). Not an accuracy estimate of the model.",
           "preprocessing": pre, "samples": samples}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"{a.out}: {len(samples)} samples, {list(pre)}; disjoint from {[c['manifest'] for c in checked]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
