#!/usr/bin/env python3
"""CIFAR-100 test subset (first N samples) with ToTensor + Normalize(ImageNet mean/std) eval preprocessing.

Key cifar100_imagenet_norm: ToTensor() then Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), the
AB-BNN test transform (Inference/CIFAR100.ipynb, transform_test). torchvision computes float32(k)/255, then
tensor.sub_(mean).div_(std) in float32 with float32 mean/std; reproduced in numpy float32 (identical IEEE
operations), as tools/eval_sets/cifar10_normalized.py.

Selection rule: the first --n rows (0 = all 10000) of cifar-100-python/test (torchvision CIFAR100(train=False)
order, shuffle=False); labels = fine_labels (100 classes). The test file is md5-checked against torchvision's
value and read with the restricted unpickler of _brevitas_common (numpy reconstruction globals only). Output:
NHWC float32. Data are never committed.

  curl -LO https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz   (md5 eb9058c3a382ffc7106e4002c42a8d85)
  python3 tools/eval_sets/cifar100_normalized.py --cifar-dir .cache/data/cifar100 --out .cache/data/eval128_cifar100_norm
  python3 tools/eval_sets/cifar100_normalized.py --cifar-dir .cache/data/cifar100 --n 0 --out .cache/data/eval10000_cifar100_norm

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
import _cifar_train  # noqa: E402

TEST_MD5 = "f0ef6b0ae62326f3e7ffdfab6717acfc"   # torchvision.datasets.CIFAR100.test_list
META_MD5 = "7973b15100ade9c7d40fb424638fde48"   # torchvision.datasets.CIFAR100.meta
VARIANTS = {
    "cifar100_imagenet_norm": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                               "AB-BNN Inference/CIFAR100.ipynb transform_test"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cifar-dir", required=True, help="directory containing cifar-100-python/")
    ap.add_argument("--n", type=int, default=128, help="first n test images (0 = all 10000)")
    ap.add_argument("--out", required=True)
    _cifar_train.add_args(ap)
    args = ap.parse_args()

    tb = os.path.join(args.cifar_dir, "cifar-100-python", "test")
    mt = os.path.join(args.cifar_dir, "cifar-100-python", "meta")
    for path, want in ((tb, TEST_MD5), (mt, META_MD5)):
        got = md5_file(path)
        if got != want:
            print(f"ERROR: {path} md5 {got} != torchvision's {want}", file=sys.stderr)
            return 3
    raw = open(tb, "rb").read()
    d = safe_unpickle(raw)
    classes = [c.decode() for c in safe_unpickle(open(mt, "rb").read())[b"fine_label_names"]]
    data = np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32)  # NCHW, R/G/B planes
    labels = [int(v) for v in d[b"fine_labels"]]
    names, calib = [f"test[{i}]" for i in range(len(data))], None
    if args.split == "train":
        data, labels, names, calib = _cifar_train.load_selection("cifar100", args)
    n = args.n or len(data)
    sel = data[:n]
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
                "label": labels[i], "class_name": classes[labels[i]]} for i in range(n)]
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": f"cifar100-test-first-{n}" if n < len(data) else f"cifar100-test-{n}",
        "source": "CIFAR-100 python version (https://www.cs.toronto.edu/~kriz/cifar.html), test; images not "
                  "redistributed. Terms: no formal licence; cite Krizhevsky 2009 (TR, U. Toronto).",
        "source_files": {"test": {"md5": TEST_MD5, "sha256": sha256_bytes(raw)}, "meta": {"md5": META_MD5}},
        "selection_rule": (f"first {n} rows of test (torchvision CIFAR100(train=False) order, shuffle=False), fine labels"
                           if n < len(data) else "all 10000 test images, fine labels"),
        "n_classes_covered": len({s['label'] for s in samples}),
        "class_index": {"classes": classes},
        "preprocessing": pre,
        "limitations": (f"Smoke/regression subset: {n} of 10000 CIFAR-100 test images ({len({s['label'] for s in samples})} "
                        "of 100 classes); accuracy on it is NOT an estimate of full test accuracy." if n < len(data)
                        else "Full CIFAR-100 test set."),
        "samples": samples,
    }
    _cifar_train.write_manifest(args, manifest, calib)
    print(f"wrote {list(pre)} -> {args.out}; n={n}, classes covered {manifest['n_classes_covered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
