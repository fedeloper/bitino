#!/usr/bin/env python3
"""CIFAR-10 test subset (first N samples) with ToTensor + Normalize(mean, std) eval preprocessing.

Two variants are written into one eval-set directory (same images, same labels):
  * cifar10_imagenet_norm : Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]); the IR-Net
    family code bases (ReCU, RBNN, SiMaN, BiPer: cifar*/dataset/dataset.py, param['cifar10'])
  * cifar10_bireal_norm   : Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010));
    ShuaiZ1037/bnn-xnor-bireal train.py:117-122 (Bi-Real-20)
Both test transforms are ToTensor() then Normalize (no resize/crop). torchvision computes
float32(k)/255, then tensor.sub_(mean).div_(std) in float32 with float32 mean/std; reproduced
in numpy float32 (identical IEEE operations).

Selection rule: the first --n rows of `test_batch` (torchvision CIFAR10(train=False) order,
shuffle=False), as tools/eval_sets/cifar10_brevitas.py. Output: NHWC float32. Data are never
committed; anyone with CIFAR-10 can rebuild the set (hashes in the manifest).

  python3 tools/eval_sets/cifar10_normalized.py --cifar-dir .cache/data/cifar10 --out .cache/data/eval128_cifar10_norm

--split train --train-dir DIR --disjoint-from MANIFEST... [--add-to-manifest]: same preprocessing on the first
--n training rows instead (calibration set; see tools/eval_sets/_cifar_train.py). Default output unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _brevitas_common import md5_file, safe_unpickle, sha256_bytes, to_tensor_f32  # noqa: E402
from cifar10_brevitas import CLASSES, TEST_BATCH_MD5  # noqa: E402
import _cifar_train  # noqa: E402

VARIANTS = {
    "cifar10_imagenet_norm": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                              "IR-Net family (ReCU/RBNN/SiMaN/BiPer) cifar dataset.py param['cifar10']['normalize']"),
    "cifar10_bireal_norm": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],
                            "ShuaiZ1037/bnn-xnor-bireal train.py:117-122"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cifar-dir", required=True, help="directory containing cifar-10-batches-py/")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    _cifar_train.add_args(ap)
    args = ap.parse_args()

    tb = os.path.join(args.cifar_dir, "cifar-10-batches-py", "test_batch")
    md5 = md5_file(tb)
    if md5 != TEST_BATCH_MD5:
        print(f"ERROR: test_batch md5 {md5} != torchvision's {TEST_BATCH_MD5}", file=sys.stderr)
        return 3
    raw = open(tb, "rb").read()
    d = safe_unpickle(raw)
    data = np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32)  # NCHW, R/G/B planes
    labels = [int(v) for v in d[b"labels"]]
    names, calib = [f"test_batch[{i}]" for i in range(len(data))], None
    if args.split == "train":
        data, labels, names, calib = _cifar_train.load_selection("cifar10", args)
    sel = data[: args.n]
    t = to_tensor_f32(sel)                                           # NCHW float32 k/255
    os.makedirs(args.out, exist_ok=True)
    pre = {}
    for key, (mean, std, src) in VARIANTS.items():
        m = np.asarray(mean, np.float32).reshape(1, 3, 1, 1)
        s = np.asarray(std, np.float32).reshape(1, 3, 1, 1)
        x = ((t - m) / s).astype(np.float32).transpose(0, 2, 3, 1).copy()   # NHWC
        fname = f"{key}.f32"
        x.tofile(os.path.join(args.out, fname))
        pre[key] = {"shape": list(x.shape), "dtype": "float32", "layout": "NHWC", "file": fname,
                    "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                    "definition": f"torchvision ToTensor() then Normalize(mean={mean}, std={std}) in float32, CHW->HWC ({src})"}
    samples = [{"file": names[i], "image_sha256": sha256_bytes(sel[i].tobytes()),
                "label": labels[i], "class_name": CLASSES[labels[i]]} for i in range(args.n)]
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": f"cifar10-test-first-{args.n}",
        "source": "CIFAR-10 python version (https://www.cs.toronto.edu/~kriz/cifar.html), test_batch; "
                  "images not redistributed. Terms: no formal licence; cite Krizhevsky 2009 (TR, U. Toronto).",
        "source_files": {"test_batch": {"md5": md5, "sha256": sha256_bytes(raw)}},
        "selection_rule": f"first {args.n} rows of test_batch (torchvision CIFAR10(train=False) order, shuffle=False)",
        "n_classes_covered": len({s['label'] for s in samples}),
        "class_index": {"classes": CLASSES},
        "preprocessing": pre,
        "limitations": f"Smoke/regression subset: {args.n} of 10000 CIFAR-10 test images; accuracy on it is NOT an "
                       "estimate of full test accuracy.",
        "samples": samples,
    }
    _cifar_train.write_manifest(args, manifest, calib)
    print(f"wrote {list(pre)} -> {args.out}; label histogram {np.bincount([s['label'] for s in samples], minlength=10).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
