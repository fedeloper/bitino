#!/usr/bin/env python3
"""First binarization disagreement between the C++ runtime and the numpy executor.

Sites, in node order: BConv2D/BDense inputs (relative to their per-channel threshold), BMatMul
operands, Activation sign/sign_zero inputs and FakeQuant rounding (|pre| = distance of the
pre-rounding argument to the nearest k + 0.5 boundary, in quantization steps).

Used when an inexact kernel tier (e.g. float_opt) changes end-to-end outputs: proves
whether the divergence starts as a near-zero sign flip (float rounding, acceptable)
or as a large error (kernel bug).

  bitino_bench --model M --inputs X --outputs /dev/null --no-reuse-memory \
      --dump-dir D --dump-sample K --stage optimized --iterations 1 --warmup 0
  python3 tools/first_flip_runtime.py --model M --inputs X --sample K --dump-dir D
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from bitino import executor, format as bfformat  # noqa: E402
from bitino.ir import unpack_bits  # noqa: E402


def load_dump(d: str, name: str, index: dict) -> np.ndarray | None:
    if name not in index:
        return None
    e = index[name]
    raw = np.fromfile(os.path.join(d, e["file"]), np.uint32 if e["dtype"] == "bit" else np.float32)
    shape = tuple(e["shape"])
    if e["dtype"] == "bit":
        return unpack_bits(raw.reshape(shape[:-1] + (-1,)), shape[-1]).astype(np.float32)
    return raw.reshape(shape)


def binarized(x, mode: str):
    x = np.asarray(x)
    if mode == "sign_zero":
        return np.sign(x)
    if mode == "gt0" or mode == "gt":
        return x > 0
    return x >= 0


def fq_levels(x, node, env):
    """FakeQuant integer levels (same float32 steps as the kernel) and the distance of the
    pre-rounding argument to the nearest rounding boundary k + 0.5 (the analogue of |pre|)."""
    a = node.attrs
    ax = a.get("axis", -1) % np.ndim(x)
    shp = [1] * np.ndim(x)
    shp[ax] = -1
    s = np.asarray(env[node.inputs[1]], np.float32).reshape(shp)
    z = np.asarray(env[node.inputs[2]], np.float32).reshape(shp)
    t = ((np.asarray(x, np.float32) * np.float32(a.get("pre_scale", 1.0))) / s + z).astype(np.float32)
    q = np.clip(np.rint(t), a.get("qmin", 0), a.get("qmax", 1))
    dist = np.abs(t.astype(np.float64) - (np.floor(t.astype(np.float64)) + 0.5))
    return q, dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    g = bfformat.read(a.model)
    # Sample record = every graph input's raw bytes in graph input order (bitino_bench --inputs).
    dts = {"f32": np.float32, "i32": np.int32, "i8": np.int8, "u8": np.uint8}
    sizes = [int(np.prod(i["shape"])) * np.dtype(dts[i.get("dtype", "f32")]).itemsize for i in g.inputs]
    off = a.sample * sum(sizes)
    feeds = {}
    for i, nb in zip(g.inputs, sizes):
        dt = dts[i.get("dtype", "f32")]
        feeds[i["name"]] = np.fromfile(a.inputs, dt, count=nb // np.dtype(dt).itemsize, offset=off).reshape(i["shape"])
        off += nb
    env = executor.run(g, feeds, keep_intermediates=True)
    index = json.load(open(os.path.join(a.dump_dir, "index.json")))
    max_float_diff_before = 0.0
    res = {"sample": a.sample, "first_flip": None}
    for n in g.nodes:
        sites = []  # (tensor, mode, threshold tensor or None)
        if n.op in ("BConv2D", "BDense"):
            thr = n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else None
            sites = [(n.inputs[0], n.attrs.get("input_binarize", "ge"), thr)]
        elif n.op == "BMatMul":
            sites = [(n.inputs[0], n.attrs.get("a_binarize", "sign_zero"), None), (n.inputs[1], n.attrs.get("b_binarize", "sign_zero"), None)]
        elif n.op == "Activation" and n.attrs.get("type") in ("sign", "sign_zero"):
            sites = [(n.inputs[0], "ge" if n.attrs["type"] == "sign" else "sign_zero", None)]
        elif n.op == "FakeQuant":
            sites = [(n.inputs[0], "fake_quant", None)]
        elif n.op in ("QBConv2D", "QBDense"):
            sites = [(n.inputs[0], "kbit", None)]
        for src, mode, thr in sites:
            cpp = load_dump(a.dump_dir, src, index)
            ref = env[src]
            if cpp is not None and cpp.shape != np.shape(ref) and cpp.size == np.size(ref):
                cpp = cpp.reshape(np.shape(ref))  # dumps keep the runtime's shape (e.g. bit tensors)
            if cpp is not None and res["first_flip"] is None:
                if mode == "fake_quant":
                    (qc, dc), (qr, dr) = fq_levels(cpp, n, env), fq_levels(ref, n, env)
                    flips = qc != qr
                    pre_c, pre_r = dc, dr
                elif mode == "kbit":
                    # k-bit DoReFa input: a "flip" is a different integer level; |pre| is the
                    # distance to the nearest level boundary (i + 1/2) / n.
                    nl = np.float32(2 ** int(n.attrs["act_bits"]) - 1)
                    tc, tr = np.clip(cpp, 0, 1).astype(np.float32) * nl, np.clip(ref, 0, 1).astype(np.float32) * nl
                    flips = np.round(tc) != np.round(tr)
                    pre_c = np.abs(tc - (np.floor(tc) + 0.5)).astype(np.float64) / nl
                    pre_r = np.abs(tr - (np.floor(tr) + 0.5)).astype(np.float64) / nl
                else:
                    # value relative to the threshold; f32 - f32 is exact in float64, so the sign test is exact
                    t = 0.0 if thr is None else env[thr].astype(np.float64)
                    pre_c, pre_r = cpp.astype(np.float64) - t, ref.astype(np.float64) - t
                    flips = binarized(pre_c, mode) != binarized(pre_r, mode)
                if flips.any():
                    res["first_flip"] = {"node": n.name, "op": n.op, "mode": mode, "n_flips": int(flips.sum()), "n_bits": int(flips.size),
                                         "max_abs_pre_numpy": float(np.abs(pre_r[flips]).max()),
                                         "max_abs_pre_cpp": float(np.abs(pre_c[flips]).max()),
                                         "typical_abs_pre": float(np.median(np.abs(pre_r))),
                                         "max_abs_diff_feeding_tensor": float(np.abs(cpp - ref).max()),
                                         "max_float_diff_before": max_float_diff_before}
        if res["first_flip"] is not None:
            break
        out = n.outputs[0]
        cpp = load_dump(a.dump_dir, out, index)
        if cpp is not None and cpp.shape == np.shape(env[out]):
            max_float_diff_before = max(max_float_diff_before, float(np.abs(cpp - env[out]).max()))
    if res["first_flip"] is None:
        res["max_float_diff_overall"] = max_float_diff_before
    print(json.dumps(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
