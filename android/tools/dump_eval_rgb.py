#!/usr/bin/env python3
"""Decodes the images of an evaluation set to raw RGB, for the optional real-image check of PreprocessingTest.

The Kotlin preprocessing of PIL.Image.open(f).convert("RGB") of each eval image must reproduce the eval set's
tensors byte for byte (torchvision_imagenet.f32 of .cache/data/eval128, omz_bgr.f32 of .cache/data/eval128_omz_bgr).
ImageNet images are not redistributable, so this runs only where the validation images are available:

  python3 android/tools/dump_eval_rgb.py --val-dir <imagenet val dir> --out <dir> \
      --set .cache/data/eval128:torchvision_imagenet --set .cache/data/eval128_omz_bgr:omz_bgr
  BITINO_EVAL_RGB_DIR=<dir> android/gradlew -p android :bitino:testDebugUnitTest

Writes <dir>/images.txt ("<w> <h> <file>.rgb" per sample, eval-set order), the .rgb files (h*w*3 bytes) and
<dir>/sets.txt ("<preprocessing key>\t<tensor file>\t<definition>").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True, help="<eval dir>:<preprocessing key>")
    a = ap.parse_args()
    from PIL import Image

    os.makedirs(a.out, exist_ok=True)
    samples = None
    sets = []
    for spec in a.set:
        d, key = spec.rsplit(":", 1)
        man = json.load(open(os.path.join(d, "manifest.json")))
        files = [s["file"] for s in man["samples"]]
        if samples is None:
            samples = man["samples"]
        elif files != [s["file"] for s in samples]:
            print(f"{d}: different samples than the first set", file=sys.stderr)
            return 2
        pre = man["preprocessing"][key]
        sets.append(f"{key}\t{os.path.abspath(os.path.join(d, pre['file']))}\t{pre['definition']}")
    lines = []
    for i, s in enumerate(samples):
        raw = open(os.path.join(a.val_dir, s["file"]), "rb").read()
        if s.get("jpeg_sha256") and hashlib.sha256(raw).hexdigest() != s["jpeg_sha256"]:
            print(f"hash mismatch: {s['file']}", file=sys.stderr)
            return 3
        img = np.array(Image.open(os.path.join(a.val_dir, s["file"])).convert("RGB"))
        name = f"{i:05d}.rgb"
        img.tofile(os.path.join(a.out, name))
        lines.append(f"{img.shape[1]} {img.shape[0]} {name}")
    open(os.path.join(a.out, "images.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(a.out, "sets.txt"), "w").write("\n".join(sets) + "\n")
    print(f"{len(lines)} images, {len(sets)} sets -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
