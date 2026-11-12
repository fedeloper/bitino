#!/usr/bin/env python3
"""Splits a converted "Faster Than Lies" detector (tools/convert_ftl_deepfake.py) into its two stages so the
preprocessing and the network can be timed separately on a phone (same kernels, same session options):

  model_features.bitino : input RGB [1,224,224,3] -> ImageGrayU8 -> FftLogMagnitude, LocalBinaryPattern -> Concat
                           = the 5-channel tensor "features" [1,224,224,5] (nodes features.*)
  model_network.bitino  : input "features" [1,224,224,5] -> adapter Conv2D -> BNext -> logits (every other node)

The composition is the full model node for node (bit-identical outputs; checked here with the numpy executor on
--check-input when given).

  python3 tools/ftl_deepfake_submodels.py --artifacts artifacts/ftl-deepfake-bnext-t-cocofake-unfrozen [--model model.bitino]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402


def split(g):
    feat = [n for n in g.nodes if n.name.startswith("features.")]
    net = [n for n in g.nodes if not n.name.startswith("features.")]
    assert feat and g.nodes[:len(feat)] == feat, "features.* nodes must come first"
    cat = feat[-1]
    assert cat.op == "Concat", cat.op
    h, w = g.inputs[0]["shape"][1:3]
    gf = copy.deepcopy(g)
    gf.nodes = copy.deepcopy(feat)
    gf.outputs = [cat.outputs[0]]
    gn = copy.deepcopy(g)
    gn.nodes = copy.deepcopy(net)
    gn.inputs = [{"name": "features", "dtype": "f32", "shape": [1, h, w, 5]}]
    for n in gn.nodes:
        n.inputs = ["features" if i == cat.outputs[0] else i for i in n.inputs]
    for sub, what in ((gf, "features"), (gn, "network")):
        passes.prune_unused_constants(sub)
        sub.manifest = dict(sub.manifest, variant=f"{sub.manifest.get('variant', '')}; {what} stage only (tools/ftl_deepfake_submodels.py)")
    return gf, gn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--model", default="model.bitino")
    ap.add_argument("--check-input", help="raw float32 NHWC RGB file: check features + network == full (numpy, first sample)")
    args = ap.parse_args()
    g = bfformat.read(os.path.join(args.artifacts, args.model))
    gf, gn = split(g)
    stem = os.path.splitext(args.model)[0]
    for sub, suffix in ((gf, "features"), (gn, "network")):
        p = os.path.join(args.artifacts, f"{stem}_{suffix}.bitino")
        bfformat.write(sub, p)
        print(f"wrote {p}: {sub.op_counts()}")
    if args.check_input:
        h, w = g.inputs[0]["shape"][1:3]
        x = np.fromfile(args.check_input, np.float32, count=h * w * 3).reshape(1, h, w, 3)
        full = executor.run(g, {"input": x})[g.outputs[0]]
        f = executor.run(gf, {"input": x})[gf.outputs[0]]
        y = executor.run(gn, {"features": f})[gn.outputs[0]]
        assert np.array_equal(full.view(np.uint32), y.view(np.uint32)), (full, y)
        print("features + network == full model (numpy executor, bit-exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
