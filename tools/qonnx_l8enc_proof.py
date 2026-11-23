#!/usr/bin/env python3
"""Exhaustive check of the step-6 L8ENC pattern (docs/qonnx-profile.md § 5, docs/qonnx-step6.md): the 7-step
branchless bisection that python/bitino/qonnx_int8.Int8Writer.l8enc writes (fp32 -> int8 log2 code, exponent 0)
equals the kernels' encoder bitino.int8_ops.l8_encode on EVERY fp32 input that is not a NaN (profile assumption A4):
all 2^32 - 2^24 + 2 bit patterns (both signs, +-0, subnormals, +-inf). The pattern is evaluated as written (the
writer's nodes in onnxruntime, ORT_DISABLE_ALL, the profile's fp32 reference) and as its numpy mirror
(qonnx_int8.l8enc_bisect); the step-5 127-compare count form is compared too.

  .cache/venv-qonnx/bin/python -I tools/qonnx_l8enc_proof.py [--chunk-bits 22] [--out reports/qonnx_step6/l8enc_proof.json]
      [--quick N]   (only every N-th chunk: a smoke run)

Why exhaustive is also a proof sketch: l8_encode reads only the sign, the exponent and the top 7 mantissa bits
(b >> 16), so every boundary beta_k (the smallest fp32 with code >= k) has its low 16 bits zero and the count of
boundaries <= |w| is the same for all 2^16 low halves; the run checks every pattern anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import int8_ops, qonnx_int8 as qi8  # noqa: E402
from bitino.ir import Graph  # noqa: E402


def l8enc_session(n: int):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    wr = qi8.Int8Writer(Graph(), {}, {})
    wr.cur = (type("N", (), {"name": "l8"})(), 0)
    wr._casts = set()
    y = wr.l8enc("w", out="y")
    nodes = [helper.make_node(nd["op"], nd["inputs"], nd["outputs"], **nd["attrs"]) for nd in wr.s.nodes]
    inits = [numpy_helper.from_array(np.asarray(v), k) for k, v in wr.s.inits.items()]
    gr = helper.make_graph(nodes, "l8enc", [helper.make_tensor_value_info("w", TensorProto.FLOAT, [n])],
                           [helper.make_tensor_value_info(y, TensorProto.INT8, [n])], inits)
    mo = helper.make_model(gr, opset_imports=[helper.make_opsetid("", 17)])
    mo.ir_version = 10
    onnx.checker.check_model(mo, full_check=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1
    ops = [nd["op"] for nd in wr.s.nodes]
    return ort.InferenceSession(mo.SerializeToString(), so, providers=["CPUExecutionProvider"]), ops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-bits", type=int, default=22)
    ap.add_argument("--quick", type=int, default=1, help="check every N-th chunk only (smoke run; 1 = all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_step6", "l8enc_proof.json"))
    args = ap.parse_args()
    import onnxruntime as ort
    n = 1 << args.chunk_bits
    sess, ops = l8enc_session(n)
    b = qi8.beta()
    t0 = time.time()
    checked = nan_skipped = 0
    mism = {"onnxruntime": 0, "numpy_mirror": 0, "count_form": 0}
    first = {}
    for c in range(0, 1 << (32 - args.chunk_bits), args.quick):
        pats = np.arange(c * n, (c + 1) * n, dtype=np.uint64).astype(np.uint32)
        w = pats.view(np.float32)
        keep = ~np.isnan(w)
        nan_skipped += int(n - keep.sum())
        ref = int8_ops.l8_encode(w)
        got = sess.run(None, {"w": np.where(keep, w, np.float32(0))})[0]
        mir = qi8.l8enc_bisect(np.where(keep, w, np.float32(0)))
        a = np.abs(w[keep])
        cnt = np.searchsorted(b, a, side="right")          # #{k : beta_k <= |w|} (the 127-compare count form)
        cf = (np.sign(w[keep]).astype(np.int64) * cnt).astype(np.int8)
        for k, v in (("onnxruntime", got[keep]), ("numpy_mirror", mir[keep]), ("count_form", cf)):
            bad = np.nonzero(v != ref[keep])[0]
            mism[k] += int(bad.size)
            if bad.size and k not in first:
                first[k] = int(pats[keep][bad[0]])
        checked += int(keep.sum())
        if c % 64 == 0:
            print(f"chunk {c}: {checked} checked, mismatches {mism}, {time.time() - t0:.0f} s", flush=True)
    full = args.quick == 1
    res = {"pattern": "L8ENC 7-step bisection (qonnx_int8.Int8Writer.l8enc)", "onnx_ops": ops, "n_nodes": len(ops),
           "reference": "bitino.int8_ops.l8_encode", "evaluations": ["onnxruntime ORT_DISABLE_ALL", "numpy mirror qonnx_int8.l8enc_bisect",
                                                                     "127-compare count form (step 5)"],
           "onnxruntime": ort.__version__, "inputs_checked": checked, "nan_patterns_skipped": nan_skipped,
           "exhaustive": full, "expected_non_nan_patterns": (1 << 32) - (1 << 24) + 2 if full else None,
           "mismatches": mism, "first_mismatch_pattern": first, "seconds": round(time.time() - t0, 1),
           "equal_on_every_input": full and not any(mism.values()) and checked == (1 << 32) - (1 << 24) + 2}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if not any(mism.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
