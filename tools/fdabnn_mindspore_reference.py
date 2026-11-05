#!/usr/bin/env python3
"""Runs the OFFICIAL FDA-BNN MindSpore model (research/cv/FDA-BNN src/resnet.py + src/quan.py, read only) on CPU
and writes its float32 logits for an eval set (slice S12 reference cross-check; MindSpore has no float64 CPU conv,
so the float64 reference is the PyTorch port tools/fdabnn_reference.py).

Must run in an environment with MindSpore (e.g. a venv; MindSpore 2.7.2 CPU was used), with ``python -I``:

  .cache/venv-mindspore-s12/bin/python -I tools/fdabnn_mindspore_reference.py \
      --repo .cache/repos/mindspore-models-FDA-BNN --ckpt .cache/checkpoints/fda-bnn-resnet20-cifar10-mindspore/fdabnn.ckpt \
      --eval-dir .cache/data/eval128_cifar10_norm --preprocess cifar10_imagenet_norm --out mindspore_logits_eval128.f32

The two source files are executed as modules ``src.quan`` / ``src.resnet`` (resnet.py imports ``src.quan``)
without putting the clone on sys.path. The checkpoint is loaded with MindSpore's own load_checkpoint +
load_param_into_net (protobuf); every network parameter must be loaded (checked). Evaluation as eval.py:
``net.set_train(False)``, CPU, batches of --batch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import types

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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--preprocess", default="cifar10_imagenet_norm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--mode", choices=("pynative", "graph"), default="pynative")
    args = ap.parse_args()
    sys.dont_write_bytecode = True

    import mindspore as ms
    from mindspore import context
    from mindspore.train.serialization import load_checkpoint, load_param_into_net

    context.set_context(mode=context.PYNATIVE_MODE if args.mode == "pynative" else context.GRAPH_MODE, device_target="CPU")
    pkg = types.ModuleType("src")
    pkg.__path__ = []
    sys.modules["src"] = pkg
    quan = _load("src.quan", os.path.join(args.repo, "quan.py"))
    pkg.quan = quan
    resnet = _load("src.resnet", os.path.join(args.repo, "resnet.py"))
    net = resnet.resnet20()
    ck = load_checkpoint(args.ckpt)
    net_params = {p.name for p in net.get_parameters()}
    renamed = {}
    # MindSpore >= 2.x keeps an explicit Parameter(name='a') of resnet.PReLU (l.102) instead of the attribute name 'w'
    # used by the MindSpore 1.3 checkpoint; map '<cell>.w' -> the network's name for that parameter.
    for k in list(ck):
        if k not in net_params and k.endswith(".w"):
            cand = [n for n in net_params if n.rsplit(".", 1)[0] == k.rsplit(".", 1)[0]]
            if len(cand) == 1:
                renamed[k] = cand[0]
    ck2 = {renamed.get(k, k): v for k, v in ck.items()}
    not_loaded, ck_not_loaded = load_param_into_net(net, ck2, strict_load=True)
    assert not not_loaded and not ck_not_loaded, (not_loaded, ck_not_loaded)
    net.set_train(False)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][args.preprocess]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])
    out = []
    for i in range(0, len(x), args.batch):
        xb = ms.Tensor(np.ascontiguousarray(x[i:i + args.batch].transpose(0, 3, 1, 2)), ms.float32)
        out.append(net(xb).asnumpy().astype(np.float32))
    logits = np.concatenate(out)
    logits.tofile(args.out)
    acc = float(np.mean(logits.argmax(1) == labels))
    info = {"mindspore": ms.__version__, "mode": args.mode, "n": int(len(x)), "top1": acc,
            "top1_first_9984": float(np.mean(logits[:9984].argmax(1) == labels[:9984])) if len(x) >= 9984 else None,
            "renamed_checkpoint_keys": renamed,
            "source_sha256": {f: hashlib.sha256(open(os.path.join(args.repo, f), "rb").read()).hexdigest() for f in ("quan.py", "resnet.py")}}
    json.dump(info, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
