#!/usr/bin/env python3
"""Eval subsets for the FINN-ONNX / QONNX / BNN-PYNQ (Theano) models (slice S7).

Three datasets, first --n samples in the source order, NHWC float32, several preprocessing keys each:

  cifar10 (test_batch, torchvision order, md5-checked):
    cifar10_pixels   float32(k), k = uint8 pixel. Input of the finn-examples FINN-ONNX cnv-w1a1.onnx, whose
                     graph starts with Div 255 (finn-examples v0.0.1a, "global_in" = raw pixels).
    cifar10_bnnpynq  float32(2/255) * float32(k) - 1 in float32: BNN-PYNQ cifar10.py
                     `np.subtract(np.multiply(2./255., X), 1.)` on pylearn2's float32 X (numpy keeps float32
                     for a Python-float scalar), i.e. the input of cnv.genCnv (its first layer is the 8-bit
                     input quantizer).
  mnist (t10k, torchvision order, md5-checked):
    mnist_pixels     float32(k): input of finn-examples tfc-w1a1.onnx (graph starts with Div 255).
    mnist_bnnpynq    +-1: BNN-PYNQ mnist.py `np.where(2*X - 1 < 0, -1, 1)` with X = k/255 (pylearn2
                     MNIST(center=False)); 2k/255 - 1 < 0 <=> k <= 127 in any float precision (k = 127.5 is
                     never hit), so the value is -1 for k <= 127 and +1 for k >= 128.
  gtsrb (test.p of the Udacity "traffic-signs-data.zip", the set finn-examples'
         7_traffic_sign_recognition_gtsrb.ipynb validates the cnv_w1a1_gtsrb accelerator on: 12,630 GTSRB
         test images resized to 32x32 RGB uint8, labels = GTSRB class ids 0..42):
    gtsrb_totensor   float32(k)/255: QONNX cnv_1w1a_gtsrb.onnx input (finn-examples build.py prepends
                     Div 255 to the QONNX graph, "GTSRB data with raw uint8 pixels is divided by 255").
    gtsrb_bnnpynq    float32(2*(k/255.0) - 1) (gtsrb.py resize_and_format, float64 then stored in a float32
                     array). NOTE: BNN-PYNQ trained on aspect-preserving 40-px resizes with random 32-px crops
                     and tested on a random split of Final_Training with random rotations/crops; the Udacity
                     32x32 test images are a different (whole-image) framing, so accuracy on them is an
                     approximation of the published test error, not a reproduction.

test.p is a pickle: it is read with the restricted unpickler of _brevitas_common (numpy reconstruction
globals only), so no code can run. Data are never committed; hashes are in the manifests.

  python3 tools/eval_sets/finn_bnn_pynq.py cifar10 --cifar-dir .cache/data/cifar10 --n 128 --out .cache/data/eval128_cifar10_finn
  python3 tools/eval_sets/finn_bnn_pynq.py mnist --mnist-raw ~/data/MNIST/raw --n 128 --out .cache/data/eval128_mnist_finn
  python3 tools/eval_sets/finn_bnn_pynq.py gtsrb --zip .cache/data/gtsrb_udacity_raw/traffic-signs-data.zip --n 128 \
      --out .cache/data/eval128_gtsrb

cifar10 also takes --split train --train-dir DIR --disjoint-from MANIFEST... [--add-to-manifest]: the same keys on the
first --n training rows (calibration set; see tools/eval_sets/_cifar_train.py). Default output unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _brevitas_common import md5_file, read_idx, read_maybe_gz, safe_unpickle, sha256_bytes  # noqa: E402
import _cifar_train  # noqa: E402

TEST_BATCH_MD5 = "40351d587109b95175f43aff81a1287e"
MNIST_MD5 = {"t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
             "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c"}
GTSRB_ZIP_SHA256 = "0ee14ebdd48b07d73ab9f717c902a233f360eb52c8bf68b554747089809f9c75"
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _key(x: np.ndarray, fname: str, definition: str) -> dict:
    return {"shape": list(x.shape), "dtype": "float32", "layout": "NHWC", "file": fname,
            "sha256": hashlib.sha256(x.tobytes()).hexdigest(), "definition": definition}


def _write(out: str, keys: dict, manifest: dict, args=None, calib=None) -> None:
    os.makedirs(out, exist_ok=True)
    pre = {}
    for name, (x, definition) in keys.items():
        x = np.ascontiguousarray(x, np.float32)
        fname = f"{name}.f32"
        x.tofile(os.path.join(out, fname))
        pre[name] = _key(x, fname, definition)
    manifest["preprocessing"] = pre
    if calib is not None:
        _cifar_train.write_manifest(args, manifest, calib)
    else:
        json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"wrote {out}: {', '.join(f'{k} {v['shape']}' for k, v in pre.items())}")


def cifar10(args) -> int:
    tb = os.path.join(args.cifar_dir, "cifar-10-batches-py", "test_batch")
    md5 = md5_file(tb)
    if md5 != TEST_BATCH_MD5:
        print(f"ERROR: test_batch md5 {md5} != {TEST_BATCH_MD5}", file=sys.stderr)
        return 3
    raw = open(tb, "rb").read()
    d = safe_unpickle(raw)
    data = np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32)[: args.n]
    labels = [int(v) for v in d[b"labels"]][: args.n]
    names, calib = [f"test_batch[{i}]" for i in range(len(labels))], None
    if getattr(args, "split", "test") == "train":
        data, labels, names, calib = _cifar_train.load_selection("cifar10", args)
    k = data.transpose(0, 2, 3, 1).astype(np.float32)
    bnn = (np.float32(2.0 / 255.0) * k - np.float32(1.0)).astype(np.float32)
    samples = [{"file": names[i], "image_sha256": sha256_bytes(data[i].tobytes()), "label": labels[i],
                "class_name": CIFAR_CLASSES[labels[i]]} for i in range(len(labels))]
    _write(args.out, {
        "cifar10_pixels": (k, "float32(k) for uint8 pixel k, NHWC RGB (finn-examples FINN-ONNX cnv-w1a1: Div 255 is in the graph)"),
        "cifar10_bnnpynq": (bnn, "float32(2/255)*float32(k) - 1 in float32, NHWC RGB (BNN-PYNQ cifar10.py on pylearn2 float32 X)"),
    }, {"schema": "bitino.eval_set.v1", "name": f"cifar10-test-first-{len(labels)}",
        "source": "CIFAR-10 python version (https://www.cs.toronto.edu/~kriz/cifar.html), test_batch; images not redistributed.",
        "source_files": {"test_batch": {"md5": md5, "sha256": sha256_bytes(raw)}},
        "selection_rule": f"first {len(labels)} rows of test_batch (torchvision / pylearn2 order)",
        "n_classes_covered": len(set(labels)), "class_index": {"classes": CIFAR_CLASSES},
        "limitations": f"{len(labels)} of 10000 CIFAR-10 test images; a 128-image subset is a smoke test, not an accuracy estimate.",
        "samples": samples}, args, calib)
    return 0


def mnist(args) -> int:
    files = {}
    for name, want in MNIST_MD5.items():
        p = os.path.join(args.mnist_raw, name)
        got = md5_file(p)
        if got != want:
            print(f"ERROR: {name} md5 {got} != {want}", file=sys.stderr)
            return 3
        files[name] = {"md5": got, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    img = read_idx(read_maybe_gz(os.path.join(args.mnist_raw, "t10k-images-idx3-ubyte.gz")))[: args.n]
    lab = read_idx(read_maybe_gz(os.path.join(args.mnist_raw, "t10k-labels-idx1-ubyte.gz")))[: args.n]
    k = img[..., None].astype(np.float32)
    pm1 = np.where(img[..., None] >= 128, 1.0, -1.0).astype(np.float32)
    # The definition above (2k/255 - 1 < 0 <=> k <= 127) is checked in float32 and float64.
    for dt in (np.float32, np.float64):
        kk = np.arange(256, dtype=dt)
        assert np.array_equal(np.where(dt(2) * (kk / dt(255)) - dt(1) < 0, -1, 1), np.where(np.arange(256) >= 128, 1, -1))
    samples = [{"file": f"t10k-images-idx3-ubyte[{i}]", "image_sha256": sha256_bytes(img[i].tobytes()), "label": int(lab[i]),
                "class_name": str(int(lab[i]))} for i in range(len(lab))]
    _write(args.out, {
        "mnist_pixels": (k, "float32(k) for uint8 pixel k, NHWC [N,28,28,1] (finn-examples FINN-ONNX tfc-w1a1: Div 255 is in the graph)"),
        "mnist_bnnpynq": (pm1, "+1 if k >= 128 else -1 (BNN-PYNQ mnist.py np.where(2*X-1 < 0, -1, 1), X = k/255)"),
    }, {"schema": "bitino.eval_set.v1", "name": f"mnist-test-first-{len(lab)}",
        "source": "MNIST test set (torchvision mirror files); images not redistributed.", "source_files": files,
        "selection_rule": f"first {len(lab)} images of t10k-images-idx3-ubyte", "n_classes_covered": len(set(lab.tolist())),
        "class_index": {"classes": [str(i) for i in range(10)]},
        "limitations": f"{len(lab)} of 10000 MNIST test images; a 128-image subset is a smoke test, not an accuracy estimate.",
        "samples": samples})
    return 0


def gtsrb(args) -> int:
    zsha = hashlib.sha256(open(args.zip, "rb").read()).hexdigest()
    if zsha != GTSRB_ZIP_SHA256:
        print(f"ERROR: traffic-signs-data.zip sha256 {zsha} != {GTSRB_ZIP_SHA256}", file=sys.stderr)
        return 3
    raw = zipfile.ZipFile(args.zip).read("test.p")
    d = safe_unpickle(raw)
    d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in d.items()}
    feats = np.asarray(d["features"])
    labs = np.asarray(d["labels"]).astype(np.int64)
    assert feats.dtype == np.uint8 and feats.shape[1:] == (32, 32, 3), (feats.dtype, feats.shape)
    feats, labs = feats[: args.n], labs[: args.n]
    tot = (feats.astype(np.float32) / np.float32(255.0)).astype(np.float32)
    bnn = (2.0 * (feats.astype(np.float64) / 255.0) - 1.0).astype(np.float32)
    samples = [{"file": f"test.p[features][{i}]", "image_sha256": sha256_bytes(feats[i].tobytes()), "label": int(labs[i]),
                "class_name": str(int(labs[i]))} for i in range(len(labs))]
    _write(args.out, {
        "gtsrb_totensor": (tot, "float32(k)/255, NHWC RGB (finn-examples build.py custom_step_add_preproc: Div 255 before the QONNX graph)"),
        "gtsrb_bnnpynq": (bnn, "float32(2*(k/255.0)-1) computed in float64, NHWC RGB (BNN-PYNQ gtsrb.py normalisation); "
                               "framing differs from BNN-PYNQ training (40-px aspect-preserving resize + 32-px crops)"),
    }, {"schema": "bitino.eval_set.v1", "name": f"gtsrb-test-udacity32-first-{len(labs)}",
        "source": "GTSRB final test set (Stallkamp et al. 2012) as resized to 32x32 RGB in Udacity's traffic-signs-data.zip "
                  "test.p (https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip, "
                  "the file finn-examples' GTSRB notebook downloads); not redistributed. Licence of GTSRB: research use (unverified).",
        "source_files": {"traffic-signs-data.zip": {"sha256": zsha, "s3_etag": "133a4ee03b43b5a33f94b8f5f5e0ac4f-12 (multipart, not a content hash)"},
                         "test.p": {"sha256": sha256_bytes(raw)}},
        "selection_rule": f"first {len(labs)} entries of test.p (file order)", "n_classes_covered": len(set(labs.tolist())),
        "class_index": {"classes": [str(i) for i in range(43)] + ["junk (model output 43, never a label)"]},
        "limitations": f"{len(labs)} of 12630 GTSRB test images (Udacity 32x32 resize); a 128-image subset is a smoke test.",
        "samples": samples})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="ds", required=True)
    a = sub.add_parser("cifar10"); a.add_argument("--cifar-dir", required=True)
    _cifar_train.add_args(a)
    b = sub.add_parser("mnist"); b.add_argument("--mnist-raw", required=True)
    c = sub.add_parser("gtsrb"); c.add_argument("--zip", required=True)
    for p in (a, b, c):
        p.add_argument("--n", type=int, default=128)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"cifar10": cifar10, "mnist": mnist, "gtsrb": gtsrb}[args.ds](args)


if __name__ == "__main__":
    sys.exit(main())
