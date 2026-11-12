#!/usr/bin/env python3
"""Eval-set preprocessing "ftl_rgb01" ("Faster Than Lies" deepfake detector, fedeloper/binary_deepfake_detection).

The authors' test loaders (coco_fake_dataset.py / dffd_dataset.py / cifake_dataset.py, read_image, non-train
split) feed the model RGB in [0, 1]; normalisation happens inside the model, after the FFT / LBP channels:

  PIL.Image.open(f).convert('RGB') -> torchvision.transforms.v2:
  Resize(252 [= 224 + 224 // 8, short side], BILINEAR [antialias default True]) -> CenterCrop(224)
  -> ToImage() -> ToDtype(float32, scale=True)   (= uint8 * fl32(1/255)); CHW -> HWC

Labels follow the source (is_real): 1 = real, 0 = fake (the IR's logit row 1, row 0 is a constant 0).

Sources (images are not redistributed; only their hashes are recorded):
  --source cifake   : a local download of the CIFAKE test split (REAL/ and FAKE/ subdirectories, e.g. the
                      Hugging Face mirror batgre/CIFAKE; ftl_download_manifest.json written by the download step)
  --source imagenet : the 128 ImageNet validation images of an existing eval manifest (all real photos, label 1)

  python -I tools/eval_sets/ftl_deepfake.py --source cifake --raw-dir .cache/downloads-ftl-cifake \
      --out .cache/data/eval128_cifake_ftl
  python -I tools/eval_sets/ftl_deepfake.py --source imagenet --ref-manifest .cache/data/eval128/manifest.json \
      --raw-dir ~/noise/imagenet100/val --out .cache/data/eval128_imagenet_ftl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

NAME = "ftl_rgb01"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("cifake", "imagenet"), required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--ref-manifest", help="imagenet: eval manifest whose samples are used")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import PIL
    import torch
    import torchvision
    import torchvision.transforms.v2 as T
    from PIL import Image

    tf = T.Compose([T.Resize(224 + 224 // 8, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224), T.ToImage(),
                    T.ToDtype(torch.float32, scale=True)])
    samples = []
    if args.source == "cifake":
        dl = json.load(open(os.path.join(args.raw_dir, "ftl_download_manifest.json")))
        for f in dl["files"]:
            samples.append({"file": os.path.join(f["label"], os.path.basename(f["path"])), "jpeg_sha256": f["sha256"],
                            "label": 1 if f["label"] == "REAL" else 0, "class_name": f["label"].lower()})
        meta = {"name": "cifake-test-subset-128",
                "source": f"CIFAKE test split (Bird & Lotfi, IEEE Access 2024; real = CIFAR-10 test, fake = Stable Diffusion 1.4), "
                          f"Hugging Face mirror {dl['repo']}@{dl['revision']} (original file names; the mirror has "
                          f"{dl.get('n_REAL_in_split')} REAL / {dl.get('n_FAKE_in_split')} FAKE test files); images not redistributed",
                "selection_rule": "per class: sorted file names, every (n_files // 64)-th, first 64; REAL then FAKE",
                "limitations": "Smoke/regression subset: 128 of 20000 CIFAKE test images (32x32 upscaled to 224 as in the "
                               "source); accuracy on it is NOT an estimate of the full test accuracy."}
    else:
        ref = json.load(open(args.ref_manifest))
        for s in ref["samples"]:
            samples.append({"file": s["file"], "jpeg_sha256": s["jpeg_sha256"], "label": 1, "class_name": "real",
                            "imagenet_label": s["label"]})
        meta = {"name": "imagenet-val-subset-128-as-real",
                "source": "the 128 ImageNet ILSVRC2012 validation images of imagenet-val-subset-128 (all real photographs, "
                          "label 1 = real); images not redistributed",
                "selection_rule": f"samples of {os.path.basename(os.path.dirname(args.ref_manifest))}/manifest.json, same order",
                "limitations": "Real photographs only (no fakes): conversion-fidelity set, the fraction classified real is "
                               "not a detector accuracy."}
    xs = []
    for s in samples:
        path = os.path.join(args.raw_dir, s["file"])
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != s["jpeg_sha256"]:
            print(f"ERROR: hash mismatch for {s['file']}", file=sys.stderr)
            return 3
        x = tf(Image.open(path).convert("RGB"))
        xs.append(x.permute(1, 2, 0).contiguous().numpy()[None])
    arr = np.ascontiguousarray(np.concatenate(xs), np.float32)
    os.makedirs(args.out, exist_ok=True)
    arr.tofile(os.path.join(args.out, f"{NAME}.f32"))
    man = {"schema": "bitino.eval_set.v1", **meta, "n_classes_covered": len({s["label"] for s in samples}),
           "labels": {"0": "fake", "1": "real"}, "samples": samples,
           "preprocessing": {NAME: {"shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": f"{NAME}.f32",
                                    "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                                    "definition": f"torchvision {torchvision.__version__} / PIL {PIL.__version__} / torch {torch.__version__}: "
                                                  "PIL.Image.open(f).convert('RGB') -> v2.Resize(252, BILINEAR, antialias=True) -> "
                                                  "v2.CenterCrop(224) -> v2.ToImage() -> v2.ToDtype(float32, scale=True); CHW -> HWC; "
                                                  "RGB in [0, 1], no normalisation (inside the model). Re-implements read_image() of "
                                                  "fedeloper/binary_deepfake_detection {coco_fake,dffd,cifake}_dataset.py (non-train split)"}}}
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {args.out}: {arr.shape}, labels {np.bincount([s['label'] for s in samples]).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
