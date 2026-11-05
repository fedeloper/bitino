#!/usr/bin/env python3
"""Static consistency check of an int8_full model: the zero point a consumer assumes for each int8 input vs the zero
point the producer actually writes (y_zero of the producing operator, propagated through Reshape / Transpose /
max and average pooling / Pad / plain-copy Concat). A mismatch means the consumer misreads the codes (e.g. the stale
grid of a Reshape after an AvgPool2D, docs/int8.md log2 section, "Larq models").

    python3 tools/int8_full_grid_check.py artifacts/*/model_int8_full.bitino
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from bitino import format as F  # noqa: E402

KEEP = {"Reshape", "I8Transpose", "I8MaxPool2D", "I8AvgPool2D", "I8Pad"}


def check(path: str) -> list[str]:
    g = F.read(path, verify=False)
    zero = {}
    bad = []
    for n in g.nodes:
        a = n.attrs
        # what this node assumes about its inputs
        exp = {}
        if "x_zero" in a and n.inputs:
            exp[0] = int(a["x_zero"])
        if n.op in ("I8Add", "I8Mul"):
            exp[0], exp[1] = int(a.get("a_zero", 0)), int(a.get("b_zero", 0))
        if n.op in ("I8AffineSum", "I8Concat") and "zeros" in a:
            k = list(range(len(n.inputs) - 2)) if n.op == "I8AffineSum" else list(range(len(n.inputs)))  # AffineSum: ..., K, B
            for j, z in zip(k, a["zeros"]):
                exp[j] = int(z)
        for j, z in exp.items():
            t = n.inputs[j]
            if t in zero and zero[t] is not None and zero[t] != z and a.get("x_code", "linear") != "l8":
                bad.append(f"{n.name} ({n.op}) input {j} '{t}': assumes zero {z}, producer writes {zero[t]}")
        # what this node writes
        for o in n.outputs:
            if n.op in KEEP:
                zero[o] = zero.get(n.inputs[0])
            elif n.op == "I8Concat" and all(int(m) == 0 for m in a.get("mults", [1])):
                zero[o] = int(a["y_zero"])
            elif "y_zero" in a and a.get("output", "i8") == "i8":
                zero[o] = int(a["y_zero"])
            elif n.op == "I8Quantize":
                zero[o] = int(a["zero"])
            else:
                zero[o] = None  # LUT tables, l8 codes, bits, f32: not checked
    # the grid (scale, zero) the quantizer recorded for a tensor (what its consumers were built for) vs the grid of the
    # tensor its codes come from through Reshape / transpose / pooling / Pad
    tp = g.manifest.get("quantization", {}).get("tensor_params", {})
    src = {}
    for n in g.nodes:
        for o in n.outputs:
            src[o] = src.get(n.inputs[0], n.inputs[0]) if n.op in KEEP else o
    for t, s in src.items():
        if s != t and t in tp and s in tp and [float(v) for v in tp[t]] != [float(v) for v in tp[s]]:
            bad.append(f"'{t}' recorded grid {tp[t]} but holds the codes of '{s}' on grid {tp[s]}")
    return bad


def main() -> int:
    tot = 0
    for p in sys.argv[1:]:
        b = check(p)
        tot += len(b)
        name = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
        print(f"{name}: {'OK' if not b else str(len(b)) + ' MISMATCH'}")
        for line in b[:5]:
            print("   " + line)
    return 1 if tot else 0


if __name__ == "__main__":
    sys.exit(main())
