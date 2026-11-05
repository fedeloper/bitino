#!/usr/bin/env python3
"""BHViT eval preprocessing for the shared 128-image ImageNet subset.

Same 128 images as .cache/data/eval128/manifest.json (each JPEG re-hashed and checked against
that manifest), preprocessed with the official IMRL/BHViT eval transform. The transform object
is built by the repo's own datasets.build_transform(is_train=False, args) with the default
--input-size 224 of main_new.py (datasets.py:109-119):

  PIL.Image.open(f).convert('RGB')                       (torchvision ImageFolder default_loader)
  -> transforms.Resize(int(256/224*224)=256, interpolation=3)   3 = PIL BICUBIC, short side 256
  -> transforms.CenterCrop(224)
  -> transforms.ToTensor()
  -> transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)   (timm constants)
  -> CHW to HWC (Bitino activations are NHWC)

Writes <out>/bhvit.f32 and <out>/manifest.json (schema of the base manifest, same samples and
labels). Must run with the read-only 'bhvit' conda env (torchvision 0.20, timm 1.0), e.g.:

  ~/anaconda3/envs/bhvit/bin/python tools/eval_sets/bhvit.py --repo .cache/repos/IMRL_BHViT \
      --base-manifest .cache/data/eval128/manifest.json --val-dir ~/noise/imagenet100/val \
      --out .cache/data/eval128_bhvit

--add-to-manifest adds the entry to the manifest already in --out (which must list the same samples as
--base-manifest) instead of writing a derived manifest, e.g. for the train-split calibration set:

  ~/anaconda3/envs/bhvit/bin/python tools/eval_sets/bhvit.py --repo .cache/repos/IMRL_BHViT \
      --base-manifest .cache/data/calib512_imagenet100train/manifest.json --val-dir ~/noise/imagenet100/train \
      --out .cache/data/calib512_imagenet100train --add-to-manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np

NAME = "bhvit"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="clone of github.com/IMRL/BHViT (read only)")
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-to-manifest", action="store_true",
                    help="add the entry to the existing manifest in --out (same samples as --base-manifest)")
    args = ap.parse_args()

    import PIL
    import timm
    import torchvision
    from PIL import Image

    sys.dont_write_bytecode = True  # never write __pycache__ into the source clone
    sys.path.insert(0, os.path.abspath(args.repo))
    import datasets as bhvit_datasets  # official datasets.py

    tf = bhvit_datasets.build_transform(False, types.SimpleNamespace(input_size=224))
    base = json.load(open(args.base_manifest))
    os.makedirs(args.out, exist_ok=True)
    tensors = []
    for s in base["samples"]:
        path = os.path.join(args.val_dir, s["file"])
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != s["jpeg_sha256"]:
            print(f"ERROR: JPEG hash mismatch for {s['file']}", file=sys.stderr)
            return 3
        with Image.open(path) as im:
            t = tf(im.convert("RGB")).numpy()
        tensors.append(np.transpose(t, (1, 2, 0))[None].astype(np.float32))
    arr = np.ascontiguousarray(np.concatenate(tensors).astype(np.float32))
    bin_path = os.path.join(args.out, f"{NAME}.f32")
    arr.tofile(bin_path)
    entry = {
        "shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": os.path.basename(bin_path),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "definition": (f"torchvision {torchvision.__version__} / PIL {PIL.__version__} / timm {timm.__version__}: "
                       "IMRL/BHViT datasets.build_transform(is_train=False, input_size=224) = "
                       + repr(tf).replace("\n", " ") + " applied to PIL.Image.open(f).convert('RGB'); CHW -> HWC"),
        "script": "tools/eval_sets/bhvit.py"}
    if args.add_to_manifest:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _manifest_merge import add_preprocessing
        add_preprocessing(args.out, args.base_manifest, NAME, entry)
    else:
        manifest = {k: v for k, v in base.items() if k != "preprocessing"}
        manifest["name"] = base.get("name", "eval128") + "_bhvit"
        manifest["base_manifest"] = {"path": ".cache/data/eval128/manifest.json",
                                     "sha256": hashlib.sha256(open(args.base_manifest, "rb").read()).hexdigest()}
        manifest["preprocessing"] = {NAME: entry}
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {arr.shape} -> {bin_path} sha256 {entry['sha256']}")
    print(repr(tf))
    return 0


if __name__ == "__main__":
    sys.exit(main())
