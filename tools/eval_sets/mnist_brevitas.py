#!/usr/bin/env python3
"""MNIST test subset (first N samples) with the Brevitas BNN-PYNQ eval preprocessing.

Brevitas bnn_pynq trainer.py builds the test set as
  MirrorMNIST(train=False, transform=transforms.Compose([transforms.ToTensor()]))
(MirrorMNIST only changes download mirrors), shuffle=False, so sample i is image i of
t10k-images-idx3-ubyte. ToTensor maps uint8 k -> float32 k/255; no normalisation (the FC
model's forward flattens, computes 2x-1 and applies a 1-bit input quantizer).

Selection rule: the first --n test images. Output: NHWC float32 [N, 28, 28, 1].

  python3 tools/eval_sets/mnist_brevitas.py --mnist-raw ~/data/MNIST/raw \
      --out .cache/data/eval128_mnist_brevitas
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _brevitas_common import md5_file, read_idx, read_maybe_gz, sha256_bytes, to_tensor_f32  # noqa: E402

# torchvision.datasets.MNIST.resources
MD5 = {"t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
       "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mnist-raw", required=True, help="directory with t10k-*-ubyte.gz (MNIST/raw)")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = {}
    for name, want in MD5.items():
        p = os.path.join(args.mnist_raw, name)
        got = md5_file(p)
        if got != want:
            print(f"ERROR: {name} md5 {got} != torchvision's {want}", file=sys.stderr)
            return 3
        files[name] = {"md5": got, "sha256": sha256_bytes(open(p, "rb").read())}
    imgs = read_idx(read_maybe_gz(os.path.join(args.mnist_raw, "t10k-images-idx3-ubyte.gz")))
    labs = read_idx(read_maybe_gz(os.path.join(args.mnist_raw, "t10k-labels-idx1-ubyte.gz")))
    assert imgs.shape == (10000, 28, 28) and labs.shape == (10000,)
    sel = imgs[: args.n]
    x = to_tensor_f32(sel)[..., None].copy()  # NHWC, C=1
    os.makedirs(args.out, exist_ok=True)
    fname = "mnist_totensor.f32"
    x.tofile(os.path.join(args.out, fname))
    samples = [{"file": f"t10k-images-idx3-ubyte[{i}]", "image_sha256": sha256_bytes(sel[i].tobytes()),
                "label": int(labs[i]), "class_name": str(int(labs[i]))} for i in range(args.n)]
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": f"mnist-test-first-{args.n}",
        "source": "MNIST test set (LeCun, Cortes, Burges; torchvision mirror files); images not redistributed. "
                  "Terms: no formal licence stated by the authors (CC BY-SA 3.0 often cited, unverified).",
        "source_files": files,
        "selection_rule": f"first {args.n} images of t10k-images-idx3-ubyte (torchvision MNIST(train=False) order)",
        "n_classes_covered": len({s['label'] for s in samples}),
        "class_index": {"classes": [str(i) for i in range(10)]},
        "preprocessing": {"mnist_totensor": {
            "shape": list(x.shape), "dtype": "float32", "layout": "NHWC", "file": fname,
            "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "definition": "torchvision transforms.ToTensor(): float32(uint8)/255; no normalisation "
                          "(brevitas_examples/bnn_pynq/trainer.py transform_to_tensor)"}},
        "limitations": f"Smoke/regression subset: {args.n} of 10000 MNIST test images; accuracy on it is NOT an "
                       "estimate of full test accuracy.",
        "samples": samples,
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {x.shape} -> {os.path.join(args.out, fname)}; label histogram "
          f"{np.bincount([s['label'] for s in samples], minlength=10).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
