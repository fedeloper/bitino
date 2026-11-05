#!/usr/bin/env python3
"""Runs the OFFICIAL tensorpack DoReFa-Net AlexNet (examples/DoReFa-Net/alexnet-dorefa.py, --dorefa 1,1,32) on CPU
and writes its float32 logits for an eval set (slice S12 reference; TensorFlow cannot run this graph in float64,
so the float64 reference is the PyTorch port tools/dorefa_alexnet_reference.py, checked against this output).

Run in an environment with TensorFlow 1.x-compatible tensorpack (recorded: tensorpack 0.11 + TF 2.15.1 CPU,
``.cache/venv-tensorpack-s12``), with ``python -I``:

  .cache/venv-tensorpack-s12/bin/python -I tools/tensorpack_dorefa_reference.py --repo .cache/repos/tensorpack \
      --npz ".cache/checkpoints/tensorpack-dorefa-alexnet-w1a1-g32/AlexNet-1,1,32.npz" \
      --eval-dir .cache/data/eval128_tensorpack_bgr --out tensorpack_logits_eval128.f32

The example files dorefa.py, imagenet_utils.py and alexnet-dorefa.py are executed as modules from the clone (not put
on sys.path), BITW/BITA/BITG set to 1/1/32 as the script's ``--dorefa 1,1,32`` does, and the model is evaluated with
tensorpack's PredictConfig/OfflinePredictor (the path of ``--run``). The npz is read with
np.load(allow_pickle=False) and restored with DictRestore (what SmartInit does for .npz). The uint8 BGR crops of the
eval set are fed to the graph's ``input``; the graph's own image_preprocess output is fetched and compared bit for bit
with the eval set's float tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys

import numpy as np


def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--preprocess", default="tensorpack_bgr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    sys.dont_write_bytecode = True
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf
    ex = os.path.join(args.repo, "examples", "DoReFa-Net")
    _load("dorefa", os.path.join(ex, "dorefa.py"))
    _load("imagenet_utils", os.path.join(ex, "imagenet_utils.py"))
    mod = _load("alexnet_dorefa", os.path.join(ex, "alexnet-dorefa.py"))
    mod.BITW, mod.BITA, mod.BITG = 1, 1, 32            # alexnet-dorefa.py:205-211 with --dorefa 1,1,32
    import tensorpack
    from tensorpack import OfflinePredictor, PredictConfig
    from tensorpack.tfutils.sessinit import DictRestore

    with np.load(args.npz, allow_pickle=False) as z:
        params = {k: z[k] for k in z.files}
    cfg = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.threads, inter_op_parallelism_threads=1)
    from tensorpack.tfutils.sesscreate import NewSessionCreator
    pc = PredictConfig(model=mod.Model(), session_init=DictRestore(params), input_names=["input"],
                       output_names=["fct/output", "image_preprocess/truediv"], session_creator=NewSessionCreator(config=cfg))
    pred = OfflinePredictor(pc)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][args.preprocess]
    u8 = np.fromfile(os.path.join(args.eval_dir, pre["uint8_file"]), np.uint8).reshape(pre["shape"])
    xf = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(u8)])
    out, mism = [], 0
    for i in range(0, len(u8), args.batch):
        logits, img = pred(u8[i:i + args.batch])
        out.append(np.asarray(logits, np.float32))
        mism += int(np.sum(np.asarray(img, np.float32).view(np.uint32) != xf[i:i + args.batch].view(np.uint32)))
    logits = np.concatenate(out)
    logits.tofile(args.out)
    info = {"tensorflow": tf.__version__, "tensorpack": tensorpack.__version__, "n": int(len(u8)),
            "top1": float(np.mean(logits.argmax(1) == labels)),
            "top5": float(np.mean([labels[i] in np.argsort(-logits[i])[:5] for i in range(len(u8))])),
            "image_preprocess_vs_eval_tensor_mismatched_values": mism,
            "source_sha256": {f: hashlib.sha256(open(os.path.join(ex, f), "rb").read()).hexdigest()
                              for f in ("alexnet-dorefa.py", "dorefa.py", "imagenet_utils.py")}}
    json.dump(info, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(info))
    return 0 if mism == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
