#!/usr/bin/env python3
"""Eval-set preprocessing "bitorch_imagenet" (hpi-xnor/bitorch model hub: ResNetE-18, BinaryDenseNet-28, MeliusNet-22).

Same images (same order, same labels) as the reference ImageNet eval set manifest (.cache/data/eval128 or
eval5000); only the normalisation differs from "torchvision_imagenet".

bitorch 0.3.0 examples/image_classification/datasets/imagenet.py (the training/validation code of the hub
checkpoints, Lightning 1.6.4, mid-2022) defines

  test_transform = Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(mean, std_dev)
  mean = (0.485, 0.456, 0.406), std_dev = (0.229, 0.224, 0.255)

std_dev[2] is 0.255, not the usual 0.225 (a typo in bitorch that the hub models were trained and evaluated
with; it is still present at bitorch HEAD f2ac727). This script reproduces that transform with torchvision on
PIL exactly like the torchvision_imagenet entry (with --std 0.229 0.224 0.225 it reproduces
torchvision_imagenet.f32 bit for bit, which the script checks when that file exists next to the manifest).
Output NHWC float32.

  ~/anaconda3/bin/python -I tools/eval_sets/bitorch_imagenet.py --ref-manifest .cache/data/eval128/manifest.json \
      --val-dir ~/noise/imagenet100/val --out .cache/data/eval128_bitorch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

NAME = "bitorch_imagenet"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.255)   # bitorch ImageNet.std_dev (sic)


def preprocess(paths, std, check_sha=None):
    import torch  # noqa: F401
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize(MEAN, std)])
    out = []
    for i, path in enumerate(paths):
        if check_sha is not None:
            if hashlib.sha256(open(path, "rb").read()).hexdigest() != check_sha[i]:
                raise RuntimeError(f"JPEG hash mismatch for {path}")
        out.append(tf(Image.open(path).convert("RGB")).permute(1, 2, 0).contiguous().numpy())
    return np.stack(out).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-manifest", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-to-manifest", action="store_true",
                    help="add the entry to the existing manifest in --out (same samples as --ref-manifest), e.g. "
                         "the train-split calibration set .cache/data/calib512_imagenet100train")
    args = ap.parse_args()

    import PIL
    import torch
    import torchvision

    ref = json.load(open(args.ref_manifest))
    paths = [os.path.join(args.val_dir, s["file"]) for s in ref["samples"]]
    shas = [s["jpeg_sha256"] for s in ref["samples"]]
    tv_entry = ref["preprocessing"].get("torchvision_imagenet")
    tv_file = os.path.join(os.path.dirname(args.ref_manifest), tv_entry["file"]) if tv_entry else ""
    check = None
    if tv_entry and os.path.exists(tv_file):
        k = min(16, len(paths))
        mine = preprocess(paths[:k], (0.229, 0.224, 0.225), shas[:k])
        theirs = np.fromfile(tv_file, np.float32, count=mine.size).reshape(mine.shape)
        check = {"n_checked": k, "identical": bool(np.array_equal(mine, theirs))}
        if not check["identical"]:
            print("ERROR: the pipeline does not reproduce torchvision_imagenet.f32", file=sys.stderr)
            return 4
    out = preprocess(paths, STD, shas)
    os.makedirs(args.out, exist_ok=True)
    fname = f"{NAME}.f32"
    out.tofile(os.path.join(args.out, fname))
    entry = {
        "shape": list(out.shape), "dtype": "float32", "layout": "NHWC", "file": fname,
        "sha256": hashlib.sha256(out.tobytes()).hexdigest(),
        "definition": (f"torchvision {torchvision.__version__} / PIL {PIL.__version__} / torch {torch.__version__}: "
                       "PIL.Image.open(f).convert('RGB') -> transforms.Resize(256) [InterpolationMode.BILINEAR, antialias=True] -> "
                       "transforms.CenterCrop(224) -> transforms.ToTensor() -> transforms.Normalize((0.485, 0.456, 0.406), "
                       "(0.229, 0.224, 0.255)); CHW -> HWC (bitorch 0.3.0 examples/image_classification/datasets/imagenet.py "
                       "ImageNet.test_transform; std_dev[2] = 0.255 as in bitorch)"),
    }
    if args.add_to_manifest:
        entry["torchvision_imagenet_reproduced_with_std_0.225"] = check
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _manifest_merge import add_preprocessing
        add_preprocessing(args.out, args.ref_manifest, NAME, entry)
    else:
        manifest = {k: v for k, v in ref.items() if k != "preprocessing"}
        manifest["derived_from"] = {"manifest": os.path.abspath(args.ref_manifest),
                                    "manifest_sha256": hashlib.sha256(open(args.ref_manifest, "rb").read()).hexdigest(),
                                    "torchvision_imagenet_reproduced_with_std_0.225": check}
        manifest["preprocessing"] = {NAME: entry}
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {out.shape} -> {os.path.join(args.out, fname)}; torchvision_imagenet check {check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
