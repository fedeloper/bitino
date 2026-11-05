#!/usr/bin/env python3
"""CIFAR-100 test images (first N) with the DeiT-style eval transform of IMRL/GSB-Vision-Transformer.

The transform object is built by the repo's own datasets.build_transform(is_train=False, args) with
--input-size 224 (datasets.py build_transform; test.py uses datasets.CIFAR100(train=False, transform=...)):

  PIL.Image.fromarray(uint8 HWC 32x32)                          (torchvision CIFAR100.__getitem__)
  -> transforms.Resize(int(256/224*224) = 256, InterpolationMode.BICUBIC)   upsampled 32 -> 256
  -> transforms.CenterCrop(224)
  -> transforms.ToTensor()
  -> transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)       (timm constants)
  -> CHW to HWC (Bitino activations are NHWC)

The CIFAR-100 python-version `test` pickle is read with the restricted unpickler of _brevitas_common (numpy
reconstruction globals only) and checked against torchvision's md5. Selection: the first --n rows of `test`
(torchvision CIFAR100(train=False) order), fine labels. Data are never committed.

  python3 -I tools/eval_sets/cifar100_deit.py --repo .cache/repos/IMRL_GSB-Vision-Transformer \
      --cifar-dir .cache/data/cifar100 --n 128 --out .cache/data/eval128_cifar100_deit

--split train --train-dir DIR --disjoint-from MANIFEST... [--add-to-manifest]: same preprocessing on the first
--n training rows instead (calibration set; see tools/eval_sets/_cifar_train.py). Default output unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _brevitas_common import md5_file, safe_unpickle, sha256_bytes  # noqa: E402
import _cifar_train  # noqa: E402

NAME = "cifar100_deit"
TEST_MD5 = "f0ef6b0ae62326f3e7ffdfab6717acfc"   # torchvision.datasets.CIFAR100.test_list
META_MD5 = "7973b15100ade9c7d40fb424638fde48"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="clone of github.com/IMRL/GSB-Vision-Transformer (read only)")
    ap.add_argument("--cifar-dir", required=True, help="directory containing cifar-100-python/")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    _cifar_train.add_args(ap)
    args = ap.parse_args()

    import PIL
    import timm
    import torchvision
    from PIL import Image

    tp = os.path.join(args.cifar_dir, "cifar-100-python", "test")
    mp_ = os.path.join(args.cifar_dir, "cifar-100-python", "meta")
    if md5_file(tp) != TEST_MD5 or md5_file(mp_) != META_MD5:
        print("ERROR: CIFAR-100 test/meta md5 differs from torchvision's", file=sys.stderr)
        return 3
    raw = open(tp, "rb").read()
    d = safe_unpickle(raw)
    meta = safe_unpickle(open(mp_, "rb").read())
    classes = [c.decode() for c in meta[b"fine_label_names"]]
    data = np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)   # NHWC, torchvision order
    labels = [int(v) for v in d[b"fine_labels"]]
    names, calib = [f"test[{i}]" for i in range(len(labels))], None
    if args.split == "train":
        sel, labels, names, calib = _cifar_train.load_selection("cifar100", args)
        data = sel.transpose(0, 2, 3, 1)
    n = min(args.n, len(labels))

    sys.dont_write_bytecode = True  # never write __pycache__ into the source clone
    sys.path.insert(0, os.path.abspath(args.repo))
    import datasets as gsb_datasets  # official datasets.py
    tf = gsb_datasets.build_transform(False, types.SimpleNamespace(input_size=224))

    os.makedirs(args.out, exist_ok=True)
    bin_path = os.path.join(args.out, f"{NAME}.f32")
    h = hashlib.sha256()
    with open(bin_path, "wb") as f:
        for i in range(n):
            t = tf(Image.fromarray(data[i])).numpy()
            a = np.ascontiguousarray(np.transpose(t, (1, 2, 0)).astype(np.float32))
            b = a.tobytes()
            h.update(b)
            f.write(b)
    samples = [{"file": names[i], "image_sha256": sha256_bytes(data[i].transpose(2, 0, 1).tobytes()),
                "label": labels[i], "class_name": classes[labels[i]]} for i in range(n)]
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": f"cifar100-test-first-{n}-deit224",
        "source": "CIFAR-100 python version (https://www.cs.toronto.edu/~kriz/cifar.html), test; images not redistributed. "
                  "Terms: no formal licence; cite Krizhevsky 2009 (TR, U. Toronto).",
        "source_files": {"test": {"md5": TEST_MD5, "sha256": sha256_bytes(raw)}, "meta": {"md5": META_MD5}},
        "selection_rule": f"first {n} rows of test (torchvision CIFAR100(train=False) order, shuffle=False), fine labels",
        "n_classes_covered": len({s['label'] for s in samples}),
        "class_index": {"classes": classes},
        "preprocessing": {NAME: {
            "shape": [n, 224, 224, 3], "dtype": "float32", "layout": "NHWC", "file": os.path.basename(bin_path),
            "sha256": h.hexdigest(),
            "definition": (f"torchvision {torchvision.__version__} / PIL {PIL.__version__} / timm {timm.__version__}: "
                           "IMRL/GSB-Vision-Transformer datasets.build_transform(is_train=False, input_size=224) = "
                           + repr(tf).replace("\n", " ") + " applied to PIL.Image.fromarray(CIFAR-100 32x32 uint8); CHW -> HWC"),
            "script": "tools/eval_sets/cifar100_deit.py"}},
        "limitations": ("Full CIFAR-100 test set." if n == len(labels) else
                        f"Smoke/regression subset: {n} of {len(labels)} CIFAR-100 test images; accuracy on it is NOT an estimate of full test accuracy."),
        "samples": samples,
    }
    _cifar_train.write_manifest(args, manifest, calib)
    print(f"wrote {n} samples -> {args.out} ({manifest['n_classes_covered']} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
