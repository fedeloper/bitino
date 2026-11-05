#!/usr/bin/env python3
"""Eval-set tensors with the tensorpack ImageNet evaluation preprocessing (DoReFa-Net AlexNet, slice S12).

Same images as a base ImageNet eval set (``.cache/data/eval128`` or ``eval5000``), re-preprocessed exactly as
tensorpack's evaluation path does (examples/DoReFa-Net, tensorpack 0.11):

  read:   dataset.ILSVRC12Files -> cv2.imread(f, cv2.IMREAD_COLOR)          (BGR uint8)
  resize: imagenet_utils.fbresnet_augmentor(isTrain=False) l.60-63:
          imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC): scale = 256/min(h, w); if h < w:
          (newh, neww) = (256, int(scale*w + 0.5)) else (int(scale*h + 0.5), 256); cv2.resize on uint8
          (dataflow/imgaug/misc.py:73-80, transform.py:118-125)
  crop:   imgaug.CenterCrop(224): h0 = int((H - 224)*0.5), w0 = int((W - 224)*0.5)    (imgaug/crop.py:52-58)
  model:  ImageNetModel.image_preprocess (imagenet_utils.py:386-398), image_bgr=True: float32 image,
          mean = [0.485, 0.456, 0.406][::-1] * 255., std = [0.229, 0.224, 0.225][::-1] * 255. (float32 products),
          (image - mean) / std in float32, which TF grappler (default on) rewrites to (image - mean) * fl32(1/std).

Writes ``tensorpack_bgr.f32`` (normalized, NHWC BGR float32: the Bitino model input) and
``tensorpack_bgr_u8.u8`` (the uint8 crops fed to the TensorFlow reference, whose own image_preprocess the
converter checks to be bit-identical to the float file). Recorded runs used OpenCV 4.8.1 (the tensorpack venv).

  python -I tools/eval_sets/tensorpack_bgr.py --base-manifest .cache/data/eval128/manifest.json \
      --val-dir ~/noise/imagenet100/val --out .cache/data/eval128_tensorpack_bgr
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

PREPROC = "tensorpack_bgr"
MEAN = [0.485, 0.456, 0.406][::-1]
STD = [0.229, 0.224, 0.225][::-1]


def crop_u8(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, path
    h, w = img.shape[:2]
    scale = 256 * 1.0 / min(h, w)
    if h < w:
        newh, neww = 256, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), 256
    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_CUBIC)
    h0, w0 = int((newh - 224) * 0.5), int((neww - 224) * 0.5)
    out = img[h0:h0 + 224, w0:w0 + 224]
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8, out.shape
    return out


def normalize(u8: np.ndarray) -> np.ndarray:
    """What the TF graph computes at run time: grappler's arithmetic optimizer (on by default) rewrites the division
    by the constant std into a multiplication by its float32 reciprocal (checked: 0 differences with grappler on;
    251,271 of 1.2M values differ by 1 ulp from a true division, which is what grappler-off computes)."""
    mean = np.asarray(MEAN, np.float32) * np.float32(255.0)
    std = np.asarray(STD, np.float32) * np.float32(255.0)
    return ((u8.astype(np.float32) - mean) * (np.float32(1.0) / std)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-to-manifest", action="store_true",
                    help="add the entry to the existing manifest in --out (same samples as --base-manifest), e.g. "
                         "the train-split calibration set .cache/data/calib512_imagenet100train")
    args = ap.parse_args()
    import cv2

    base = json.load(open(args.base_manifest))
    os.makedirs(args.out, exist_ok=True)
    crops = []
    for s in base["samples"]:
        path = os.path.join(args.val_dir, s["file"])
        if hashlib.sha256(open(path, "rb").read()).hexdigest() != s["jpeg_sha256"]:
            print(f"ERROR: JPEG hash mismatch for {s['file']}", file=sys.stderr)
            return 3
        crops.append(crop_u8(path)[None])
    u8 = np.ascontiguousarray(np.concatenate(crops))
    f32 = np.ascontiguousarray(normalize(u8))
    f32.tofile(os.path.join(args.out, f"{PREPROC}.f32"))
    u8.tofile(os.path.join(args.out, f"{PREPROC}_u8.u8"))
    man = copy.deepcopy(base)
    man["name"] = base["name"] + f" ({PREPROC})"
    man["preprocessing"] = {PREPROC: {
        "shape": list(f32.shape), "dtype": "float32", "layout": "NHWC", "file": f"{PREPROC}.f32",
        "sha256": hashlib.sha256(f32.tobytes()).hexdigest(),
        "uint8_file": f"{PREPROC}_u8.u8", "uint8_sha256": hashlib.sha256(u8.tobytes()).hexdigest(),
        "definition": (f"tensorpack 0.11 DoReFa-Net eval path, OpenCV {cv2.__version__}: cv2.imread BGR -> ResizeShortestEdge(256, "
                       "INTER_CUBIC, int(scale*side + 0.5)) on uint8 -> CenterCrop 224 (int((H-224)*0.5)) -> float32 "
                       "(x - [0.406,0.456,0.485]*255) * fl32(1 / ([0.225,0.224,0.229]*255)) (BGR order; = the TF graph division after grappler rewrite), NHWC"),
        "script": "tools/eval_sets/tensorpack_bgr.py"}}
    man["derived_from"] = {"manifest": os.path.abspath(args.base_manifest),
                           "sha256": hashlib.sha256(open(args.base_manifest, "rb").read()).hexdigest()}
    if args.add_to_manifest:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _manifest_merge import add_preprocessing
        add_preprocessing(args.out, args.base_manifest, PREPROC, man["preprocessing"][PREPROC])
    else:
        json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {f32.shape} -> {args.out} (sha256 {man['preprocessing'][PREPROC]['sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
