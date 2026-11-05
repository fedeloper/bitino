#!/usr/bin/env python3
"""Eval-set tensors with the OpenVINO Open Model Zoo classification preprocessing.

Same 128 images as the reference eval set (``.cache/data/eval128/manifest.json``),
re-preprocessed exactly as the OMZ ``accuracy-check.yml`` of resnet50-binary-0001 and
resnet18-xnor-binary-onnx-0001 prescribes (Accuracy Checker, open_model_zoo master):

  reader: pillow_imread         -> np.array(PIL.Image.open(f).convert('RGB'))      (image_readers.py)
  resize: size 256, aspect_ratio_scale greater, use_pillow, BILINEAR
          scale_greater(): if h > w: (W, H) = (256, int(256*h/w)) else (int(256*w/h), 256)
          -> Image.fromarray(img).resize((W, H), Image.BILINEAR)                   (resize.py)
  crop:   size 224, use_pillow  -> i = int(round((H-224)/2.)), j = int(round((W-224)/2.))
          -> Image.fromarray(img).crop((j, i, j+224, i+224))                        (crop.py)
  rgb_to_bgr                    -> cv2.cvtColor(RGB2BGR) == channel reversal         (color_space_conversion.py)

The result is BGR, 0..255 (integers stored as float32), NHWC. Mean subtraction is
embedded in the IR (Subtract node); no scaling is applied here.

Usage (any env with numpy + Pillow; recorded run used .cache/venv-openvino):
  python tools/eval_sets/omz_bgr.py --base-manifest .cache/data/eval128/manifest.json \
      --val-dir ~/noise/imagenet100/val --out .cache/data/eval128_omz_bgr
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

PREPROC = "omz_bgr"


def preprocess(path: str):
    from PIL import Image

    with open(path, "rb") as f:
        img = np.array(Image.open(f).convert("RGB"))
    h, w = img.shape[:2]
    if h > w:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    img = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    i = int(round((nh - 224) / 2.0))
    j = int(round((nw - 224) / 2.0))
    img = np.array(Image.fromarray(img).crop((j, i, j + 224, i + 224)))
    assert img.shape == (224, 224, 3), img.shape
    return img[..., ::-1].astype(np.float32)  # RGB -> BGR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-to-manifest", action="store_true",
                    help="add the entry to the existing manifest in --out (same samples as --base-manifest), e.g. "
                         "the train-split calibration set .cache/data/calib512_imagenet100train")
    args = ap.parse_args()

    import PIL

    base = json.load(open(args.base_manifest))
    os.makedirs(args.out, exist_ok=True)
    tensors = []
    for s in base["samples"]:
        path = os.path.join(args.val_dir, s["file"])
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != s["jpeg_sha256"]:
            print(f"ERROR: JPEG hash mismatch for {s['file']}", file=sys.stderr)
            return 3
        tensors.append(preprocess(path)[None])
    arr = np.ascontiguousarray(np.concatenate(tensors).astype(np.float32))
    bin_path = os.path.join(args.out, f"{PREPROC}.f32")
    arr.tofile(bin_path)
    man = copy.deepcopy(base)
    man["preprocessing"] = {PREPROC: {
        "shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": os.path.basename(bin_path),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "definition": (f"OMZ accuracy-check.yml (resnet50-binary-0001 / resnet18-xnor-binary-onnx-0001), Pillow {PIL.__version__}: "
                       "PIL.Image.open(f).convert('RGB') -> resize short side to 256 with Accuracy Checker scale_greater "
                       "(int truncation), Image.BILINEAR -> center crop 224 (offsets int(round((H-224)/2)), Python round) -> "
                       "RGB->BGR; values 0..255 as float32, NHWC. Mean subtraction (103.5,116.3,123.7 BGR) is inside the IR."),
        "script": "tools/eval_sets/omz_bgr.py"}}
    man["derived_from"] = {"manifest": os.path.abspath(args.base_manifest),
                           "sha256": hashlib.sha256(open(args.base_manifest, "rb").read()).hexdigest()}
    if args.add_to_manifest:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _manifest_merge import add_preprocessing
        add_preprocessing(args.out, args.base_manifest, PREPROC, man["preprocessing"][PREPROC])
    else:
        json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {arr.shape} -> {bin_path} (sha256 {man['preprocessing'][PREPROC]['sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
