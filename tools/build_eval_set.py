#!/usr/bin/env python3
"""Builds the deterministic ImageNet evaluation subset used for validation.

Selection: classes = sorted synset directories of an ImageNet-1k *validation*
subset (e.g. an ImageNet-100 split organised as <synset>/<ILSVRC2012_val_*.JPEG>);
sample i takes class i % n_classes and the (i // n_classes)-th file in sorted
order. Labels are ImageNet-1k indices from Keras' imagenet_class_index.json
(sorted-synset convention used by larq_zoo / Keras / torchvision).

Outputs (in --out):
  manifest.json       file names, JPEG SHA-256, labels  (small; committed copy in registry/eval_sets/)
  <preproc>.f32       float32 NHWC tensors, concatenated  (local only, never committed)

Images are NOT redistributed; anyone with ImageNet access can rebuild the set.

Calibration sets (--split train): the same selection rule applied to an ImageNet *training*
subset (<synset>/<nXXXXXXXX_*.JPEG>), for static INT8 calibration. The manifest additionally
records split "train", purpose "calibration" and a "disjointness" entry: every --disjoint-from
eval manifest is checked and the build fails if any selected file (synset/file or bare name) or
JPEG SHA-256 occurs in it. Without --split the output is unchanged (byte-identical manifests).

  ~/anaconda3/bin/python -I tools/build_eval_set.py --split train --val-dir ~/noise/imagenet100/train \
      --class-index .cache/data/imagenet_class_index.json --n 512 --preprocess torchvision_imagenet \
      --disjoint-from .cache/data/eval128/manifest.json --disjoint-from .cache/data/eval5000/manifest.json \
      --out .cache/data/calib512_imagenet100train

Several preprocessings of the SAME images can live in one eval dir: if
manifest.json already exists, the new preprocessing entry is added to it after
checking that the selected samples (file names, JPEG SHA-256, labels) are
identical. Existing entries are kept unchanged.

Preprocessings:
  larq_zoo              larq_zoo.training.data.preprocess_image_bytes(is_training=False)
                        (center crop 224/(224+32) of the short side, bicubic resize to 224,
                        mean/std normalisation). Must run inside the Larq reference env.
  torchvision_imagenet  the standard torchvision ImageNet eval transform, as used by
                        AB-BNN Inference/ImageNet.ipynb (datasets.ImageFolder default PIL loader):
                        PIL.Image.open(f).convert("RGB") -> transforms.Resize(256) (bilinear,
                        antialias, short side 256) -> transforms.CenterCrop(224) ->
                        transforms.ToTensor() -> transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)), then CHW -> HWC. Needs torch + torchvision + PIL.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

TORCHVISION_IMAGENET_DEF = (
    "torchvision {tv} / PIL {pil}: PIL.Image.open(f).convert('RGB') -> transforms.Resize(256) "
    "[InterpolationMode.BILINEAR, antialias=True, short side 256] -> transforms.CenterCrop(224) -> "
    "transforms.ToTensor() -> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)); "
    "CHW -> HWC (AB-BNN Inference/ImageNet.ipynb transform_test)")


def select(val_dir: str, n: int) -> list[tuple[str, str]]:
    classes = sorted(d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
    files = {c: sorted(os.listdir(os.path.join(val_dir, c))) for c in classes}
    out = []
    for i in range(n):
        c = classes[i % len(classes)]
        out.append((c, files[c][i // len(classes)]))
    return out


def disjointness(samples: list[dict], manifests: list[str]) -> dict:
    """Fails (SystemExit 3) if any sample's synset/file, bare file name or JPEG SHA-256 occurs in an eval manifest."""
    files = {s["file"] for s in samples}
    names = {os.path.basename(s["file"]) for s in samples}
    shas = {s["jpeg_sha256"] for s in samples}
    checked = []
    for mp in manifests:
        ev = json.load(open(mp))["samples"]
        hit_f = files & {e["file"] for e in ev}
        hit_n = names & {os.path.basename(e["file"]) for e in ev}
        hit_s = shas & {e["jpeg_sha256"] for e in ev}
        if hit_f or hit_n or hit_s:
            print(f"ERROR: calibration samples overlap {mp}: files {sorted(hit_f)[:5]} names {sorted(hit_n)[:5]} "
                  f"sha256 {sorted(hit_s)[:5]}", file=sys.stderr)
            raise SystemExit(3)
        checked.append({"manifest": os.path.relpath(os.path.abspath(mp), os.getcwd()),
                        "manifest_sha256": hashlib.sha256(open(mp, "rb").read()).hexdigest(),
                        "eval_samples": len(ev), "overlap_file": 0, "overlap_basename": 0, "overlap_jpeg_sha256": 0})
    return {"rule": "no selected sample's synset/file, bare file name or JPEG SHA-256 occurs in any listed eval manifest",
            "calib_samples": len(samples), "unique_jpeg_sha256": len(shas), "checked": checked, "passed": True}


def make_preprocessor(name: str):
    """Returns (fn(raw JPEG bytes) -> float32 HWC array, definition string)."""
    if name == "larq_zoo":
        import tensorflow as tf
        from larq_zoo.training.data import preprocess_image_bytes

        def fn(raw: bytes) -> np.ndarray:
            return preprocess_image_bytes(tf.constant(raw), is_training=False).numpy().astype(np.float32)
        return fn, "larq_zoo.training.data.preprocess_image_bytes(is_training=False)"
    if name == "torchvision_imagenet":
        import PIL
        import torchvision
        from PIL import Image
        from torchvision import transforms

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        def fn(raw: bytes) -> np.ndarray:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return transform_test(img).numpy().transpose(1, 2, 0).astype(np.float32)
        return fn, TORCHVISION_IMAGENET_DEF.format(tv=torchvision.__version__, pil=PIL.__version__)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--class-index", required=True, help="imagenet_class_index.json")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--preprocess", default="larq_zoo", choices=["larq_zoo", "torchvision_imagenet"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val",
                    help="train: calibration set from a training split (needs --disjoint-from); default val (eval set)")
    ap.add_argument("--disjoint-from", action="append", default=[], metavar="MANIFEST",
                    help="eval manifest the selected samples must not overlap (repeatable; --split train)")
    args = ap.parse_args()
    if args.split == "train" and not args.disjoint_from:
        ap.error("--split train needs at least one --disjoint-from eval manifest")

    idx = json.load(open(args.class_index))
    synset_to_label = {v[0]: int(k) for k, v in idx.items()}
    os.makedirs(args.out, exist_ok=True)
    picks = select(args.val_dir, args.n)
    if args.split == "train":  # check before the (slow) preprocessing
        disjoint = disjointness([{"file": f"{c}/{f}", "jpeg_sha256": hashlib.sha256(
            open(os.path.join(args.val_dir, c, f), "rb").read()).hexdigest()} for c, f in picks], args.disjoint_from)
    preprocess, definition = make_preprocessor(args.preprocess)

    samples, tensors = [], []
    for synset, fname in picks:
        path = os.path.join(args.val_dir, synset, fname)
        raw = open(path, "rb").read()
        img = preprocess(raw)
        tensors.append(img[None])
        samples.append({"file": f"{synset}/{fname}", "jpeg_sha256": hashlib.sha256(raw).hexdigest(),
                        "synset": synset, "label": synset_to_label[synset], "class_name": idx[str(synset_to_label[synset])][1]})
    arr = np.concatenate(tensors).astype(np.float32)
    entry = {"shape": list(arr.shape), "dtype": "float32", "layout": "NHWC", "file": f"{args.preprocess}.f32",
             "sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "definition": definition}
    man_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(man_path):
        manifest = json.load(open(man_path))
        old = [(s["file"], s["jpeg_sha256"], s["label"]) for s in manifest["samples"]]
        new = [(s["file"], s["jpeg_sha256"], s["label"]) for s in samples]
        if old != new:
            print(f"ERROR: {man_path} lists different samples; refusing to mix eval sets", file=sys.stderr)
            return 3
        if manifest.get("split", "val") != args.split:
            print(f"ERROR: {man_path} is a {manifest.get('split', 'val')} set, not {args.split}", file=sys.stderr)
            return 3
        manifest["preprocessing"][args.preprocess] = entry
    elif args.split == "train":
        manifest = {
            "schema": "bitino.eval_set.v1",
            "name": f"imagenet100-train-calib-{args.n}",
            "split": "train",
            "purpose": "calibration",
            "source": "ImageNet ILSVRC2012 training images (local ImageNet-100 train split); images not redistributed",
            "selection_rule": "class i % n_classes (sorted synsets), file i // n_classes (sorted names), i = 0..n-1",
            "n_classes_covered": len({s['synset'] for s in samples}),
            "class_index": {"url": "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
                            "sha256": hashlib.sha256(open(args.class_index, "rb").read()).hexdigest()},
            "disjointness": disjoint,
            "preprocessing": {args.preprocess: entry},
            "limitations": "Calibration set for static INT8 quantization (activation statistics), drawn from the ImageNet-100 "
                           "TRAIN split and disjoint from the evaluation sets listed under disjointness; labels are recorded "
                           "for completeness only. Never report accuracy on it.",
            "samples": samples,
        }
    else:
        manifest = {
            "schema": "bitino.eval_set.v1",
            "name": f"imagenet-val-subset-{args.n}",
            "source": "ImageNet ILSVRC2012 validation images (local ImageNet-100 split); images not redistributed",
            "selection_rule": "class i % n_classes (sorted synsets), file i // n_classes (sorted names)",
            "n_classes_covered": len({s['synset'] for s in samples}),
            "class_index": {"url": "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
                            "sha256": hashlib.sha256(open(args.class_index, "rb").read()).hexdigest()},
            "preprocessing": {args.preprocess: entry},
            "limitations": "Small smoke/regression set: only classes present in the ImageNet-100 split are covered; "
                           "accuracy on it is NOT an estimate of full ImageNet accuracy.",
            "samples": samples,
        }
    bin_path = os.path.join(args.out, entry["file"])
    arr.tofile(bin_path)
    json.dump(manifest, open(man_path, "w"), indent=1)
    print(f"wrote {len(samples)} samples, tensor {arr.shape} -> {bin_path} ({args.preprocess})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
