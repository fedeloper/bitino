#!/usr/bin/env python3
"""CIFAR-10 test subset (first N samples) with the Brevitas BNN-PYNQ eval preprocessing.

Brevitas bnn_pynq trainer.py builds the test set as
  torchvision.datasets.CIFAR10(train=False, transform=transforms.Compose([transforms.ToTensor()]))
with shuffle=False, so sample i is row i of `test_batch`. ToTensor maps uint8 k -> float32 k/255;
there is NO mean/std normalisation (the model's forward does 2x-1 and an 8-bit input quantizer).

Selection rule: the first --n rows of `test_batch` (torchvision order). Output: NHWC float32.
Data are never committed; anyone with CIFAR-10 can rebuild the set (hashes in the manifest).

  python3 tools/eval_sets/cifar10_brevitas.py --cifar-dir ~/datasets/CIFAR10 \
      --out .cache/data/eval128_cifar10_brevitas

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

CIFAR_TGZ_MD5 = "c58f30108f718f92721af3b95e74349a"  # torchvision.datasets.CIFAR10.tgz_md5
TEST_BATCH_MD5 = "40351d587109b95175f43aff81a1287e"  # torchvision.datasets.CIFAR10.test_list
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


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
    x = to_tensor_f32(sel).transpose(0, 2, 3, 1).copy()  # NHWC
    os.makedirs(args.out, exist_ok=True)
    fname = "cifar10_totensor.f32"
    x.tofile(os.path.join(args.out, fname))
    samples = [{"file": names[i], "image_sha256": sha256_bytes(sel[i].tobytes()),
                "label": labels[i], "class_name": CLASSES[labels[i]]} for i in range(args.n)]
    tgz = os.path.join(args.cifar_dir, "cifar-10-python.tar.gz")
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": f"cifar10-test-first-{args.n}",
        "source": "CIFAR-10 python version (https://www.cs.toronto.edu/~kriz/cifar.html), test_batch; "
                  "images not redistributed. Terms: no formal licence; cite Krizhevsky 2009 (TR, U. Toronto).",
        "source_files": {"test_batch": {"md5": md5, "sha256": sha256_bytes(raw)},
                         **({"cifar-10-python.tar.gz": {"md5": md5_file(tgz), "expected_md5": CIFAR_TGZ_MD5}}
                            if os.path.exists(tgz) else {})},
        "selection_rule": f"first {args.n} rows of test_batch (torchvision CIFAR10(train=False) order, shuffle=False)",
        "n_classes_covered": len({s['label'] for s in samples}),
        "class_index": {"classes": CLASSES},
        "preprocessing": {"cifar10_totensor": {
            "shape": list(x.shape), "dtype": "float32", "layout": "NHWC", "file": fname,
            "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "definition": "torchvision transforms.ToTensor(): float32(uint8)/255, CHW->HWC; no normalisation "
                          "(brevitas_examples/bnn_pynq/trainer.py transform_to_tensor)"}},
        "limitations": f"Smoke/regression subset: {args.n} of 10000 CIFAR-10 test images; accuracy on it is NOT an "
                       "estimate of full test accuracy.",
        "samples": samples,
    }
    _cifar_train.write_manifest(args, manifest, calib)
    print(f"wrote {x.shape} -> {os.path.join(args.out, fname)}; label histogram "
          f"{np.bincount([s['label'] for s in samples], minlength=10).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
