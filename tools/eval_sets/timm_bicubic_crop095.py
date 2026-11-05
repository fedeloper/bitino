#!/usr/bin/env python3
"""Eval-set preprocessing "timm_bicubic_crop095" (BNext, hpi-xnor/BNext).

Same 128 images (same order, same labels) as the reference eval set manifest
(.cache/data/eval128/manifest.json); only the preprocessing differs.

BNext's official training script (src/train_assistant_group_amp.py:496-572, run
scripts pass --dali_cpu but NOT --dali, so the timm loader path is used) builds
its validation loader with timm 0.6.11 (src/requirements.txt):

  create_loader(dataset_eval, input_size=[3,224,224], is_training=False,
                use_prefetcher=not --no-prefetcher (=True), interpolation='bicubic'
                (--train-interpolation default), mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD, crop_pct=0.95)

timm 0.6.11 transforms_imagenet_eval(crop_pct=0.95, use_prefetcher=True) is
  Resize(int(floor(224/0.95)) = 235 [short side], BICUBIC) -> CenterCrop(224) -> ToNumpy (uint8 CHW)
and PrefetchLoader then normalises on device in float32:
  x.float().sub_(mean*255).div_(std*255)
This script re-implements exactly that (torchvision Resize/CenterCrop on PIL,
which is what timm calls), because timm >= 0.9 changed the eval resize code path.
Output is NHWC float32.

Run with the base conda env (torch/torchvision, read-only):
  ~/anaconda3/bin/python -I tools/eval_sets/timm_bicubic_crop095.py \
     --ref-manifest .cache/data/eval128/manifest.json --val-dir ~/noise/imagenet100/val \
     --out .cache/data/eval128_timm_bicubic_crop095
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

NAME = "timm_bicubic_crop095"
MEAN = (0.485, 0.456, 0.406)   # timm IMAGENET_DEFAULT_MEAN
STD = (0.229, 0.224, 0.225)    # timm IMAGENET_DEFAULT_STD


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
    from PIL import Image
    from torchvision import transforms

    ref = json.load(open(args.ref_manifest))
    scale = int(math.floor(224 / 0.95))
    assert scale == 235
    tf = transforms.Compose([transforms.Resize(scale, interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.CenterCrop(224)])
    mean = torch.tensor([x * 255 for x in MEAN], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([x * 255 for x in STD], dtype=torch.float32).view(1, 3, 1, 1)

    tensors = []
    for s in ref["samples"]:
        path = os.path.join(args.val_dir, s["file"])
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != s["jpeg_sha256"]:
            print(f"ERROR: JPEG hash mismatch for {s['file']}", file=sys.stderr)
            return 3
        img = Image.open(path).convert("RGB")   # timm ImageDataset loader: img.convert('RGB')
        arr = np.asarray(tf(img), dtype=np.uint8)           # HWC uint8 (timm ToNumpy gives CHW)
        chw = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]
        x = chw.float().sub_(mean).div_(std)                # PrefetchLoader normalisation
        tensors.append(x.permute(0, 2, 3, 1).contiguous().numpy())
    out = np.concatenate(tensors).astype(np.float32)
    os.makedirs(args.out, exist_ok=True)
    fname = f"{NAME}.f32"
    out.tofile(os.path.join(args.out, fname))
    entry = {
        "shape": list(out.shape), "dtype": "float32", "layout": "NHWC", "file": fname,
        "sha256": hashlib.sha256(out.tobytes()).hexdigest(),
        "definition": (f"torchvision {torchvision.__version__} / PIL {PIL.__version__} / torch {torch.__version__}: "
                       "PIL.Image.open(f).convert('RGB') -> transforms.Resize(235) [short side, InterpolationMode.BICUBIC] -> "
                       "transforms.CenterCrop(224) -> uint8 -> float32 (x - 255*mean)/(255*std), timm IMAGENET_DEFAULT mean/std; "
                       "CHW -> HWC. Re-implements timm 0.6.11 transforms_imagenet_eval(crop_pct=0.95, interpolation='bicubic', "
                       "use_prefetcher=True) + PrefetchLoader as used by hpi-xnor/BNext src/train_assistant_group_amp.py:556-572"),
    }
    if args.add_to_manifest:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _manifest_merge import add_preprocessing
        add_preprocessing(args.out, args.ref_manifest, NAME, entry)
    else:
        manifest = {k: v for k, v in ref.items() if k != "preprocessing"}
        manifest["derived_from"] = {"manifest": os.path.abspath(args.ref_manifest),
                                    "manifest_sha256": hashlib.sha256(open(args.ref_manifest, "rb").read()).hexdigest()}
        manifest["preprocessing"] = {NAME: entry}
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {out.shape} -> {os.path.join(args.out, fname)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
