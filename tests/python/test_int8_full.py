"""Full-integer stage (int8_full): numpy oracle (python/bitino/int8_ops.py) vs the C++ runtime, bit-exact.

* fixed-point requantization and binarization thresholds against their definitions;
* the static quantizer on the tiny BiBERT-style layer of test_transformer_ops (Gather bit+f32 tables,
  LayerNorm, ternary BDense with zero weights, heads Reshape/Transpose, BMatMul sign_zero / gt0 with the
  additive mask, GELU, tanh, MatMul, Dense) and on a small CNN (Conv2D, BConv2D, ChannelAffine, PReLU,
  pooling, Add, Dense, Softmax), calibrated on random inputs; the quantized graph's outputs from the numpy
  oracle and from bitino_bench (stages int8_full_reference and int8_full) must be identical bytes.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(__file__))

from bitino import executor, format as bfformat, int8_ops, quantize_int8 as QI  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))


def requant_def(x: int, m: int, shift: int) -> int:
    l, r = max(shift, 0), max(-shift, 0)
    xs = max(-(1 << 31), min((1 << 31) - 1, x << l))
    h = (xs * m + (1 << 30)) // (1 << 31)
    return h if r == 0 else (h + (1 << (r - 1))) // (1 << r)


def test_requant_matches_definition():
    rng = np.random.default_rng(0)
    xs = rng.integers(-(1 << 31), 1 << 31, 20000)
    ms = rng.integers(1 << 30, 1 << 31, 20000)
    sh = rng.integers(-31, 9, 20000)
    got = int8_ops.requant(xs, ms, sh)
    for i in range(0, 20000, 7):
        assert int(got[i]) == requant_def(int(xs[i]), int(ms[i]), int(sh[i]))
    # quantize_multiplier round trip
    for r in (1e-9, 3e-5, 0.37, 0.5, 0.999, 1.0, 3.7, 1234.5):
        m, s = int8_ops.quantize_multiplier(r)
        assert (1 << 30) <= m < (1 << 31)
        assert abs(m * 2.0 ** s / 2 ** 31 - r) <= r * 2 ** -30


def test_thresholds_equal_dequantize_then_compare():
    rng = np.random.default_rng(1)
    codes = np.arange(-128, 128)
    for _ in range(3000):
        s = float(np.float32(2.0 ** rng.uniform(-30, 5)))
        z = int(rng.integers(-128, 128)) if rng.random() < 0.8 else int(rng.integers(-500000, 500000))
        d = int8_ops.deq(codes, s, z)
        ts = np.array([rng.uniform(d[0] - s, d[-1] + s), d[rng.integers(256)], np.nextafter(d[rng.integers(256)], np.float32(1e30)), 0.0],
                      np.float32)
        ge, gt = int8_ops.qthr_ge(ts, s, z), int8_ops.qthr_gt(ts, s, z)
        for t, a, b in zip(ts, ge, gt):
            assert np.array_equal(d >= t, codes >= a)
            assert np.array_equal(d > t, codes >= b)


def _run_bench(model: str, record: bytes, stage: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        open(d + "/i.bin", "wb").write(record)
        subprocess.run([BENCH, "--model", model, "--inputs", d + "/i.bin", "--outputs", d + "/o.bin", "--stage", stage,
                        "--warmup", "0", "--iterations", "1", "--threads", "2"], check=True, capture_output=True)
        return open(d + "/o.bin", "rb").read()


def _records(g: Graph, n: int, seed: int) -> list[bytes]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        parts = []
        for inp in g.inputs:
            shp = inp["shape"]
            if inp.get("dtype") == "i32":
                parts.append(rng.integers(0, 50, shp).astype(np.int32).tobytes())
            elif inp["name"] == "attention_mask":
                m = np.ones(shp, np.float32)
                m[..., int(rng.integers(3, shp[-1])):] = 0
                parts.append(m.tobytes())
            else:
                parts.append(rng.normal(size=shp).astype(np.float32).tobytes())
        out.append(b"".join(parts))
    return out


def _quantize_and_check(g: Graph, tmp: str, n_cal: int = 24):
    path = os.path.join(tmp, "f.bitino")
    bfformat.write(g, path)
    cal = _records(g, n_cal, 1)
    stats = QI.calibrate(path, cal, None, jobs=2)
    shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
    qg, rep = QI.quantize(g, stats, shapes)
    assert not rep["left_float"], rep["left_float"]
    qpath = os.path.join(tmp, "q.bitino")
    bfformat.write(qg, qpath)
    qg = bfformat.read(qpath)
    for rec in _records(g, 3, 99):
        env = executor.run(qg, QI._feeds(qg, rec))
        ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
        for stage in ("int8_full_reference", "int8_full"):
            assert _run_bench(qpath, rec, stage) == ref, stage
    return rep


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
def test_tiny_bert_int8_full_bit_exact():
    from test_transformer_ops import tiny_bert
    with tempfile.TemporaryDirectory() as tmp:
        rep = _quantize_and_check(tiny_bert(3), tmp)
        for op in ("I8Gather", "I8LayerNorm", "I8BDense", "I8BMatMul", "I8Transpose", "I8Lut", "I8MatMul", "I8Dense", "I8Add"):
            assert rep["int8_ops"].get(op, 0) > 0, op


def tiny_cnn(seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 12, 12, 3])
    c = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(8, 3, 3, 3)) * 0.3), g.add_constant("b0", rng.normal(size=8))],
                   {"pads": [1, 1, 1, 1], "activation": "relu"})
    c = g.add_node("MaxPool2D", [c], {"kernel": [2, 2], "strides": [2, 2]})
    a = g.add_node("ChannelAffine", [c, g.add_constant("s1", rng.random(8) + 0.5), g.add_constant("t1", rng.normal(size=8))])
    w = np.where(rng.random((16, 3, 3, 8)) < 0.5, -1, 1).astype(np.int8)
    b = g.add_node("BConv2D", [a, g.add_constant("bw", w, "bit"), g.add_constant("pm", rng.random(16) * 0.1),
                               g.add_constant("pa", rng.normal(size=16)), g.add_constant("thr", rng.normal(size=8) * 0.1)],
                   {"pads": [1, 1, 1, 1], "pad_value": 1, "input_binarize": "ge"})
    b = g.add_node("Activation", [b, g.add_constant("sl", rng.random(16) * 0.3)], {"type": "prelu"})
    b2 = g.add_node("Conv2D", [b, g.add_constant("w2", rng.normal(size=(16, 1, 1, 16)) * 0.2)])
    s = g.add_node("Add", [b, b2])
    p = g.add_node("AvgPool2D", [s], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 1, 0, 1], "count_include_pad": 0})
    gp = g.add_node("GlobalAvgPool", [p])
    d = g.add_node("Dense", [gp, g.add_constant("dw", rng.normal(size=(5, 16))), g.add_constant("db", rng.normal(size=5))])
    g.set_outputs(g.add_node("Softmax", [d]))
    return g


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
def test_tiny_cnn_int8_full_bit_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rep = _quantize_and_check(tiny_cnn(5), tmp)
        for op in ("I8Conv2D", "I8BConv2D", "I8Lut", "I8MaxPool2D", "I8AvgPool2D", "I8GlobalAvgPool", "I8Add", "I8Dense", "I8Softmax"):
            assert rep["int8_ops"].get(op, 0) > 0, op


# ----------------------------------------------------------------------------- l8 residual stream (quantize_l8)
def test_l8_code_definitions():
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    codes = np.arange(-128, 128)
    for e in (-20, -3, 0, 5, 40):
        d = int8_ops.l8_decode(codes, e)
        assert np.all(np.diff(d) >= 0)
        for c in (1, 2, 7, 8, 9, 100, 127, -1, -127, -128):
            j = min(abs(c), 127)
            ref = np.float32(float((Decimal(2) ** (Decimal(j - 127) / 8 + e))))  # double -> fp32 (no tie here)
            assert d[c + 128] == (ref if c > 0 else -ref)
        assert d[128] == 0
    assert np.array_equal(int8_ops.l8_encode(int8_ops.l8_decode(codes, 0)), np.where(codes == -128, -127, codes))
    # encode = count of the snapped boundaries 2^o (1 + B/128) below |w|
    bnd = np.array(sorted(np.ldexp(1 + b / 128, o) for o in range(-16, 1) for b in int8_ops.L8_B), np.float64)
    rng = np.random.default_rng(3)
    w = np.ldexp(rng.uniform(-2, 2, 50000), rng.integers(-22, 4, 50000)).astype(np.float32)
    w = np.concatenate([w, bnd.astype(np.float32), np.nextafter(bnd.astype(np.float32), np.float32(0)), -bnd.astype(np.float32),
                        np.array([0.0, -0.0, 1e-45, 3e38], np.float32)])
    n = np.searchsorted(bnd, np.abs(w).astype(np.float64), side="right")
    ref = np.clip(n - 1, 0, 127) * np.where(np.signbit(w), -1, 1)
    assert np.array_equal(int8_ops.l8_encode(w).astype(np.int64), ref)


def tiny_reactnet(seed: int, slope_lo: float = 0.05) -> Graph:
    """ReActNet-style residual blocks: move0 -> BConv -> Add(residual) -> move1 -> PReLU -> move2, a downsample block
    (AvgPool -> 1x1 Conv residual), stem Conv + MaxPool, GlobalAvgPool + Dense."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 16, 16, 3])
    x = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(8, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=8))],
                   {"pads": [1, 1, 1, 1]})
    x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2]})
    c = 8
    for i, (co, ds) in enumerate([(8, False), (8, False), (16, True), (16, False)]):
        m0 = g.add_node("ChannelAffine", [x, "", g.add_constant(f"m0_{i}", rng.normal(size=c) * 0.2)])
        w = np.where(rng.random((co, 3, 3, c)) < 0.5, -1, 1).astype(np.int8)
        b = g.add_node("BConv2D", [m0, g.add_constant(f"bw{i}", w, "bit"), g.add_constant(f"pm{i}", rng.random(co) * 0.2 + 0.05), "", ""],
                       {"pads": [1, 1, 1, 1], "strides": [2, 2] if ds else [1, 1], "pad_value": 0, "input_binarize": "ge", "output": "f32"})
        if ds:
            p = g.add_node("AvgPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "count_include_pad": 1})
            r = g.add_node("Conv2D", [p, g.add_constant(f"dw{i}", rng.normal(size=(co, 1, 1, c)) * 0.3), g.add_constant(f"db{i}", rng.normal(size=co))])
        else:
            r = x
        a = g.add_node("Add", [b, r])
        a = g.add_node("ChannelAffine", [a, "", g.add_constant(f"m1_{i}", rng.normal(size=co) * 0.3)])
        a = g.add_node("Activation", [a, g.add_constant(f"sl{i}", rng.uniform(slope_lo, 2.5, co))], {"type": "prelu"})
        x = g.add_node("ChannelAffine", [a, "", g.add_constant(f"m2_{i}", rng.normal(size=co) * 0.3)])
        c = co
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
def test_tiny_reactnet_l8_bit_exact():
    from bitino import quantize_l8 as QL
    g = tiny_reactnet(11)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.bitino")
        bfformat.write(g, path)
        cal = _records(g, 24, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        Q = QI.rewrite(g, stats, shapes, {})
        E = QI._Emitter(Q, {})
        qg0, _ = E.emit()
        ref_q, _ = QI.quantize(g, stats, shapes)  # the post-pass input is the regular int8_full graph
        assert [(n.op, n.inputs, n.outputs) for n in qg0.nodes] == [(n.op, n.inputs, n.outputs) for n in ref_q.nodes]
        qg, rep = QL.fuse(qg0, Q, E)
        assert rep["fused_blocks"] == 4 and rep["int8_ops"]["I8BConv2DRes"] == 4, rep
        assert rep["gemm_l8"] and "I8AffineSum" not in rep["int8_ops"] and "I8Add" not in rep["int8_ops"]
        assert sum(len(n.outputs) == 2 for n in qg.nodes if n.op == "I8BConv2DRes") == 3  # the last block has no bit consumer
        qpath = os.path.join(tmp, "q.bitino")
        bfformat.write(qg, qpath)
        qg = bfformat.read(qpath)
        for rec in _records(g, 4, 99):
            env = executor.run(qg, QI._feeds(qg, rec))
            ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
            for stage in ("int8_full_reference", "int8_full"):
                assert _run_bench(qpath, rec, stage) == ref, stage


def test_l8_stem_acc_thresholds():
    from bitino.quantize_l8 import acc_thresholds
    rng = np.random.default_rng(5)
    s = np.exp(rng.uniform(-12, 2, 200))
    b = rng.normal(size=200) * 3
    t = rng.normal(size=200) * 3
    s[:40] = 2.0 ** rng.integers(-12, 2, 40)  # exact ties: t on a reachable value, all quantities dyadic
    b[:40] = s[:40] * rng.integers(-100, 100, 40)
    t[:40] = b[:40] + s[:40] * rng.integers(-50, 50, 40)
    a = np.arange(-3000, 3001)[:, None]
    for mode in ("ge", "gt"):
        T = acc_thresholds(t, mode, b, s)
        v = s * a + b
        assert np.array_equal(a >= T, (v > t) if mode == "gt" else (v >= t)), mode
    # s == 0 (all-zero weights): the bias decides; NaN / infinite thresholds; int32 range
    T = acc_thresholds([1.0, 1.0, np.nan, np.inf, -np.inf, 1e30, -1e30], "ge", [2.0, 0.0, 0, 0, 0, 0, 0], [0.0, 0.0, 1, 1, 1, 1e-9, 1e-9])
    assert T.dtype == np.int32 and list(T) == [0, 1, 2 ** 31 - 1, 2 ** 31 - 1, -(2 ** 31), 2 ** 31 - 1, -(2 ** 31)]


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
@pytest.mark.parametrize("stem", ["l8", "bits", "l8_bits"])
def test_tiny_reactnet_l8_stem_bit_exact(stem):
    """The stem without its binarize-aware twin (quantize_l8 opts["stem"]): one GEMM; the oracle == the runtime."""
    from bitino import quantize_l8 as QL
    g = tiny_reactnet(11)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.bitino")
        bfformat.write(g, path)
        cal = _records(g, 24, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        Q = QI.rewrite(g, stats, shapes, {})
        E = QI._Emitter(Q, {})
        qg0, _ = E.emit()
        assert sum(n.name.endswith("/bin") and n.op == "I8Conv2D" for n in qg0.nodes) == 1  # the stem's twin
        qg, rep = QL.fuse(qg0, Q, E, {"stem": stem})
        assert rep["stem"] == {"mode": stem, "convs": ["conv2d_1"]} and rep["fused_blocks"] == 4, rep
        assert not any(n.name.endswith("/bin") for n in qg.nodes)
        assert rep["int8_ops"]["I8Conv2D"] == 2  # stem + downsample
        conv = next(n for n in qg.nodes if n.name == "conv2d_1")
        assert (conv.attrs.get("output") == "l8") == stem.startswith("l8") and len(conv.outputs) == (2 if "bits" in stem else 1)
        blk = next(n for n in qg.nodes if n.op == "I8BConv2DRes")
        assert blk.attrs["r_code"] == ("l8" if stem.startswith("l8") else "linear")
        if stem == "l8":
            assert blk.attrs["x_code"] == "l8" and blk.inputs[0] == conv.outputs[0].replace("conv2d_1", "maxpool2d_2") and blk.inputs[4]
        else:
            assert blk.inputs[0].endswith(":bits") and "input_binarize" not in blk.attrs and not blk.inputs[4]
        qpath = os.path.join(tmp, "q.bitino")
        bfformat.write(qg, qpath)
        qg = bfformat.read(qpath)
        for rec in _records(g, 4, 99):
            env = executor.run(qg, QI._feeds(qg, rec))
            ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
            for st in ("int8_full_reference", "int8_full"):
                assert _run_bench(qpath, rec, st) == ref, st


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
def test_tiny_reactnet_l8_bit_src_out_bit_exact():
    """PReLU slopes <= 0 in some channels: no R3b, the next binary layer binarizes the block output y, so the fused
    blocks write bits of w (bit_src "out"); the binarize-aware LUT copies of y are dropped."""
    from bitino import quantize_l8 as QL
    g = tiny_reactnet(12, slope_lo=-1.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.bitino")
        bfformat.write(g, path)
        cal = _records(g, 24, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        Q = QI.rewrite(g, stats, shapes, {})
        E = QI._Emitter(Q, {})
        qg0, _ = E.emit()
        qg, rep = QL.fuse(qg0, Q, E)
        assert rep["fused_blocks"] == 4, rep
        outs = [n for n in qg.nodes if n.op == "I8BConv2DRes" and len(n.outputs) == 2]
        assert len(outs) == 3 and all(n.attrs.get("bit_src") == "out" for n in outs)
        assert not any(o.endswith(":bin") for n in qg.nodes for o in n.outputs if o.startswith(tuple(n2.outputs[0] for n2 in outs)))
        qpath = os.path.join(tmp, "q.bitino")
        bfformat.write(qg, qpath)
        qg = bfformat.read(qpath)
        for rec in _records(g, 4, 99):
            env = executor.run(qg, QI._feeds(qg, rec))
            ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
            for stage in ("int8_full_reference", "int8_full"):
                assert _run_bench(qpath, rec, stage) == ref, stage


# ----------------------------------------------------------------------------- 2026-11-04: more block shapes / grids
def tiny_bireal(seed: int, relu: bool = False) -> Graph:
    """Bi-Real / ResNet-E style blocks without an activation chain: BConv2D (BatchNorm folded) -> Add(residual); with
    relu=True the sum has a fused ReLU (NNCF / OMZ ResNets) and the next layer binarizes it with thresholds."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 16, 16, 3])
    x = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(8, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=8))],
                   {"pads": [1, 1, 1, 1]})
    x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2]})
    c = 8
    for i, (co, ds) in enumerate([(8, False), (8, False), (16, True), (16, False)]):
        w = np.where(rng.random((co, 3, 3, c)) < 0.5, -1, 1).astype(np.int8)
        thr = g.add_constant(f"t{i}", rng.normal(size=c) * 0.2) if relu else ""
        b = g.add_node("BConv2D", [x, g.add_constant(f"bw{i}", w, "bit"), g.add_constant(f"pm{i}", rng.random(co) * 0.2 + 0.05),
                                   g.add_constant(f"pa{i}", rng.normal(size=co) * 0.3), thr],
                       {"pads": [1, 1, 1, 1], "strides": [2, 2] if ds else [1, 1], "pad_value": 0, "input_binarize": "gt" if relu else "ge"})
        if ds:
            p = g.add_node("AvgPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "count_include_pad": 1})
            r = g.add_node("Conv2D", [p, g.add_constant(f"dw{i}", rng.normal(size=(co, 1, 1, c)) * 0.3), g.add_constant(f"db{i}", rng.normal(size=co))])
        else:
            r = x
        x = g.add_node("Add", [b, r], {"activation": "relu"} if relu else {})
        c = co
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


def _l8_check(g: Graph, opts: dict, tmp: str):
    from bitino import quantize_l8 as QL
    path = os.path.join(tmp, "f.bitino")
    bfformat.write(g, path)
    cal = _records(g, 24, 1)
    stats = QI.calibrate(path, cal, 99.99, jobs=2)
    shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
    Q = QI.rewrite(g, stats, shapes, {})
    E = QI._Emitter(Q, {})
    qg, rep = QL.fuse(E.emit()[0], Q, E, opts)
    qpath = os.path.join(tmp, "q.bitino")
    bfformat.write(qg, qpath)
    qg = bfformat.read(qpath)
    for rec in _records(g, 4, 99):
        env = executor.run(qg, QI._feeds(qg, rec))
        ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
        for stage in ("int8_full_reference", "int8_full"):
            assert _run_bench(qpath, rec, stage) == ref, stage
    return rep, qg


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
@pytest.mark.parametrize("relu", [False, True])
def test_tiny_bireal_l8_identity_blocks_bit_exact(relu):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _l8_check(tiny_bireal(13, relu), {}, tmp)
        assert rep["fused_blocks"] == 4 and rep["int8_ops"]["I8BConv2DRes"] == 4, rep
        assert "I8Add" not in rep["int8_ops"] and "I8AffineSum" not in rep["int8_ops"], rep
        res = [n for n in qg.nodes if n.op == "I8BConv2DRes"]
        assert all(n.attrs.get("bit_src", "sum") == ("out" if relu else "sum") for n in res if len(n.outputs) == 2)
        slopes = {float(v) for n in res for v in np.asarray(qg.constants[n.inputs[6]].data).reshape(-1)}
        assert slopes == ({0.0} if relu else {1.0})
    with tempfile.TemporaryDirectory() as tmp:  # off: nothing fused
        from bitino import quantize_l8 as QL
        g = tiny_bireal(13, relu)
        path = os.path.join(tmp, "f.bitino")
        bfformat.write(g, path)
        cal = _records(g, 8, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        Q = QI.rewrite(g, stats, shapes, {})
        E = QI._Emitter(Q, {})
        assert QL.fuse(E.emit()[0], Q, E, {"identity_blocks": "off"})[1]["fused_blocks"] == 0


def tiny_densenet(seed: int, melius: bool, stem2: bool = False) -> Graph:
    """DenseNet / MeliusNet-style: stem Conv(relu) + MaxPool, dense layers Concat(x, BConv(BN(x))) whose channels have very
    different ranges; MeliusNet improvement blocks Add(x, Pad(BConv(BN(x)))) on the last channels; head BN, ReLU, pool, Dense.
    stem2: a second ReLU convolution (grouped) before the max pool (Larq MeliusNet's stem has three)."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 16, 16, 3])
    x = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(8, 3, 3, 3)) * 0.2), g.add_constant("b0", rng.normal(size=8) * 0.1)],
                   {"pads": [1, 1, 1, 1], "activation": "relu"})
    if stem2:
        x = g.add_node("Conv2D", [x, g.add_constant("w1", rng.normal(size=(8, 3, 3, 4)) * np.repeat(rng.uniform(0.05, 2.0, 8), 36).reshape(8, 3, 3, 4)),
                                  g.add_constant("b1", rng.normal(size=8) * 0.1)], {"pads": [1, 1, 1, 1], "groups": 2, "activation": "relu"})
    x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2]})
    c = 8
    for i in range(3):
        bn = g.add_node("ChannelAffine", [x, g.add_constant(f"s{i}", rng.uniform(-2, 2, c)), g.add_constant(f"t{i}", rng.normal(size=c))])
        w = np.where(rng.random((8, 3, 3, c)) < 0.5, -1, 1).astype(np.int8)
        b = g.add_node("BConv2D", [bn, g.add_constant(f"bw{i}", w, "bit"), "", "", ""], {"pads": [1, 1, 1, 1], "pad_value": 0, "input_binarize": "ge"})
        x = g.add_node("Concat", [x, b], {"axis": -1})
        c += 8
        if melius:
            bn = g.add_node("ChannelAffine", [x, g.add_constant(f"si{i}", rng.uniform(-2, 2, c)), g.add_constant(f"ti{i}", rng.normal(size=c))])
            w = np.where(rng.random((8, 3, 3, c)) < 0.5, -1, 1).astype(np.int8)
            b = g.add_node("BConv2D", [bn, g.add_constant(f"bwi{i}", w, "bit"), "", "", ""], {"pads": [1, 1, 1, 1], "pad_value": 0, "input_binarize": "ge"})
            p = g.add_node("Pad", [b], {"pads": [0, 0, 0, 0, 0, 0, c - 8, 0], "value": 0.0})
            x = g.add_node("Add", [x, p])
    bn = g.add_node("ChannelAffine", [x, g.add_constant("sh", rng.uniform(0.5, 2, c)), g.add_constant("th", rng.normal(size=c))])
    r = g.add_node("Activation", [bn], {"type": "relu"})
    gp = g.add_node("GlobalAvgPool", [r])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
@pytest.mark.parametrize("melius", [False, True])
def test_tiny_densenet_concat_parts_bit_exact(melius):
    g = tiny_densenet(17, melius)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.bitino")
        bfformat.write(g, path)
        cal = _records(g, 24, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        qg, rep = QI.quantize(g, stats, shapes, {"concat_parts": True})
        assert rep["rewrites"].get("per-part grids of Concat trees (concat_parts)", 0) == (5 if melius else 3), rep["rewrites"]  # the last Add joins the head BN (R7)
        cats = [n for n in qg.nodes if n.op == "I8Concat"]
        assert cats and all(all(m == 0 for m in n.attrs["mults"]) for n in cats)   # plain copies
        qpath = os.path.join(tmp, "q.bitino")
        bfformat.write(qg, qpath)
        qg = bfformat.read(qpath)
        for rec in _records(g, 3, 99):
            env = executor.run(qg, QI._feeds(qg, rec))
            ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
            for stage in ("int8_full_reference", "int8_full"):
                assert _run_bench(qpath, rec, stage) == ref, stage


# ----------------------------------------------------------------------------- 2026-11-05: MeliusNet improvement slices
def test_pc_slice_stats_match_numpy_percentile():
    """pc_slice_stats (order statistics from the k extreme values per record) == np.percentile over all values."""
    g = tiny_densenet(17, True)
    cal = _records(g, 6, 1)
    sl = QI.improvement_slices(g)
    assert len(sl) == 3
    vals = {y: [] for y in sl}

    def hook(t, v):
        if t in vals:
            vals[t].append(np.asarray(v, np.float64).reshape(-1, v.shape[-1]))
        return v
    for r in cal:
        executor.run(g, QI._feeds(g, r), post_hook=hook)
    for p in (99.0, None):
        stats = {}
        QI.pc_slice_stats(g, cal, stats, p)
        q = p if p else 100.0
        for y, (c0, b) in sl.items():
            x = np.concatenate(vals[y])[:, c0:c0 + 8]
            np.testing.assert_allclose(stats[y]["slice_pc_hi"], np.percentile(x, q, axis=0), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(stats[y]["slice_pc_lo"], np.percentile(x, 100 - q, axis=0), rtol=1e-6, atol=1e-6)
            assert abs(stats[y]["slice_hi"] - np.percentile(x, q)) < 1e-5
            assert abs(stats[y]["slice_lo"] - np.percentile(x, 100 - q)) < 1e-5
            np.testing.assert_array_equal(stats[y]["slice_pc_min"], x.min(axis=0))
            np.testing.assert_array_equal(stats[y]["slice_pc_max"], x.max(axis=0))
            assert stats[y]["slice_min"] == x.min() and stats[y]["slice_max"] == x.max()


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
@pytest.mark.parametrize("mode", ["channel", "slice", "channel_minmax", "slice_minmax"])
def test_tiny_meliusnet_pc_slices_bit_exact(mode):
    """--pc-slices: the improvement slices get grids from their own calibration (per channel / per slice); the
    I8AffineSum coefficients change, the channels outside the slice stay copies, the runtime matches the oracle."""
    g = tiny_densenet(17, True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.bitino")
        bfformat.write(g, path)
        cal = _records(g, 24, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        q0, _ = QI.quantize(g, stats, shapes, {"concat_parts": True})
        QI.pc_slice_stats(g, cal, stats, 99.99)
        q1, _ = QI.quantize(g, stats, shapes, {"concat_parts": True})
        assert [n.attrs for n in q1.nodes] == [n.attrs for n in q0.nodes]   # the new statistics alone change nothing
        qg, rep = QI.quantize(g, stats, shapes, {"concat_parts": True, "pc_slices": mode})
        k0 = {n.name: np.asarray(q0.constants[n.inputs[2]].data) for n in q0.nodes if n.op == "I8AffineSum"}
        sums = [n for n in qg.nodes if n.op == "I8AffineSum" and n.name in k0]
        assert sums and any(not np.array_equal(np.asarray(qg.constants[n.inputs[2]].data), k0[n.name]) for n in sums)
        for n in sums:   # x's channels before the slice: copies (K = 2^F, B = 0 for a zero-point-0 copy or exact z 2^F)
            K = np.asarray(qg.constants[n.inputs[2]].data).reshape(2, -1)
            c0 = QI.improvement_slices(g).get(n.outputs[0], (None,))[0]
            if c0:
                assert np.all(K[0, :c0] == K[0, 0]) and K[0, 0] == 2 ** int(round(math.log2(K[0, 0])))
        qpath = os.path.join(tmp, "q.bitino")
        bfformat.write(qg, qpath)
        qg = bfformat.read(qpath)
        for rec in _records(g, 3, 99):
            env = executor.run(qg, QI._feeds(qg, rec))
            ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
            for stage in ("int8_full_reference", "int8_full"):
                assert _run_bench(qpath, rec, stage) == ref, stage


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
def test_tiny_meliusnet_pc_stem_pooled_bit_exact():
    """--pc-stem pooled: per-channel grids for the ReLU convolution before the first max pool (not the one reading the input),
    computed by tools/quantize_int8_full.py pc_stem_stats; its I8Conv2D gets per-channel multipliers, the max pool and
    the per-part readers keep them; the runtime matches the oracle."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import quantize_int8_full as QF
    g = tiny_densenet(19, True, stem2=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.bitino")
        bfformat.write(g, path)
        cal = _records(g, 24, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        st_in = {k: dict(v) for k, v in stats.items()}
        QF.pc_stem_stats(g, cal, st_in, 99.99, "input")
        QF.pc_stem_stats(g, cal, stats, 99.99, "pooled")
        convs = [n for n in g.nodes if n.op == "Conv2D"]
        assert "pc_hi" in st_in[convs[0].outputs[0]] and "pc_hi" not in st_in[convs[1].outputs[0]]
        assert "pc_hi" in stats[convs[1].outputs[0]] and "pc_hi" not in stats[convs[0].outputs[0]]
        qg, rep = QI.quantize(g, stats, shapes, {"concat_parts": True, "pc_stem": "pooled", "pc_slices": "channel"})
        mult = [np.asarray(qg.constants[n.inputs[3]].data) for n in qg.nodes if n.op == "I8Conv2D" and n.name == convs[1].name][0]
        assert len(set(mult.tolist())) > 1   # per-channel output scales
        qpath = os.path.join(tmp, "q.bitino")
        bfformat.write(qg, qpath)
        qg = bfformat.read(qpath)
        for rec in _records(g, 3, 99):
            env = executor.run(qg, QI._feeds(qg, rec))
            ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
            for stage in ("int8_full_reference", "int8_full"):
                assert _run_bench(qpath, rec, stage) == ref, stage


def tiny_quicknet(seed: int) -> Graph:
    """QuickNet-style blocks BConv[relu] -> BatchNorm (some negative scales) -> Add(residual) (I8BConv2DRes pre stage,
    quantize_l8 res_pre), the last block of a stage Add[relu] -> MaxPool -> depthwise -> 1x1 Conv (the next residual,
    binarized by the next block: a "stem" of the l8 pass), GlobalAvgPool + Dense."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 16, 16, 3])
    x = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(8, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=8))],
                   {"pads": [1, 1, 1, 1], "strides": [2, 2], "activation": "relu"})
    x = g.add_node("Conv2D", [x, g.add_constant("w1", rng.normal(size=(8, 1, 1, 8)) * 0.4), g.add_constant("b1", rng.normal(size=8))])
    c = 8
    for i, last in enumerate([False, True, False, False]):
        w = np.where(rng.random((c, 3, 3, c)) < 0.5, -1, 1).astype(np.int8)
        b = g.add_node("BConv2D", [x, g.add_constant(f"bw{i}", w, "bit"), g.add_constant(f"pm{i}", rng.random(c) * 0.2 + 0.05),
                                   g.add_constant(f"pa{i}", rng.normal(size=c) * 0.3), ""],
                       {"pads": [1, 1, 1, 1], "pad_value": 1, "input_binarize": "ge", "activation": "relu"})
        b = g.add_node("ChannelAffine", [b, g.add_constant(f"s{i}", rng.uniform(-1.5, 2.0, c)), g.add_constant(f"t{i}", rng.normal(size=c) * 0.2)])
        x = g.add_node("Add", [b, x], {"activation": "relu"} if last else {})
        if last:
            x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2]})
            x = g.add_node("Conv2D", [x, g.add_constant(f"dw{i}", rng.normal(size=(c, 3, 3, 1)) * 0.4), g.add_constant(f"dwb{i}", rng.normal(size=c))],
                           {"pads": [1, 1, 1, 1], "groups": c})
            x = g.add_node("Conv2D", [x, g.add_constant(f"tw{i}", rng.normal(size=(16, 1, 1, c)) * 0.3), g.add_constant(f"tb{i}", rng.normal(size=16))])
            c = 16
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
def test_tiny_quicknet_l8_res_pre_bit_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _l8_check(tiny_quicknet(19), {"res_pre": True}, tmp)
        assert rep["fused_blocks"] == 4 and len(rep["pre_blocks"]) == 4, rep
        assert "I8AffineSum" not in rep["int8_ops"] and "I8Add" not in rep["int8_ops"], rep
        res = [n for n in qg.nodes if n.op == "I8BConv2DRes"]
        assert all(len(n.inputs) == 13 and n.inputs[9] == "" for n in res)
        assert {float(v) for n in res for v in np.asarray(qg.constants[n.inputs[10]].data).reshape(-1)} == {0.0}  # ReLU
        assert any(float(v) < 0 for n in res for v in np.asarray(qg.constants[n.inputs[11]].data).reshape(-1))   # BN scale < 0
    with tempfile.TemporaryDirectory() as tmp:  # without res_pre the ReLU blocks are left alone (the shipped default)
        rep, _ = _l8_check(tiny_quicknet(19), {}, tmp)
        assert not rep.get("pre_blocks"), rep


def tiny_se(seed: int, kind: str) -> Graph:
    """Squeeze-excitation gated residual blocks. kind "r2b" (Real-to-Binary): BN(x) -> BConv -> ChannelGate(., sigmoid(g(BN(x))))
    -> PReLU -> Add(x); kind "xl" (QuickNet-XL legacy): BConv[relu](x) -> BN -> ChannelGate(., sigmoid(g(x))) -> Add(x); g =
    GlobalAvgPool -> Reshape -> Dense[relu] -> Dense."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [2, 12, 12, 3])
    x = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(16, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=16))],
                   {"pads": [1, 1, 1, 1]})
    c = 16
    for i in range(3):
        src = g.add_node("ChannelAffine", [x, g.add_constant(f"bs{i}", rng.uniform(0.5, 1.5, c)), g.add_constant(f"bt{i}", rng.normal(size=c) * 0.2)]) \
            if kind == "r2b" else x
        gp = g.add_node("GlobalAvgPool", [src])
        gp = g.add_node("Reshape", [gp], {"shape": [2, c]})
        h = g.add_node("Dense", [gp, g.add_constant(f"d1w{i}", rng.normal(size=(4, c)) * 0.5), g.add_constant(f"d1b{i}", rng.normal(size=4) * 0.1)],
                       {"activation": "relu"})
        gt = g.add_node("Dense", [h, g.add_constant(f"d2w{i}", rng.normal(size=(c, 4)) * 0.8), g.add_constant(f"d2b{i}", rng.normal(size=c) * 0.3)])
        w = np.where(rng.random((c, 3, 3, c)) < 0.5, -1, 1).astype(np.int8)
        b = g.add_node("BConv2D", [src, g.add_constant(f"bw{i}", w, "bit"), g.add_constant(f"pm{i}", rng.random(c) * 0.2 + 0.05),
                                     g.add_constant(f"pa{i}", rng.normal(size=c) * 0.3), ""],
                       {"pads": [1, 1, 1, 1], "pad_value": 0, "input_binarize": "ge", "activation": "relu" if kind == "xl" else "none"})
        if kind == "xl":
            b = g.add_node("ChannelAffine", [b, g.add_constant(f"s{i}", rng.uniform(-1.5, 2.0, c)), g.add_constant(f"t{i}", rng.normal(size=c) * 0.2)])
            b = g.add_node("ChannelGate", [b, gt], {"gate": "sigmoid"})
        else:
            b = g.add_node("ChannelGate", [b, gt], {"gate": "sigmoid"})
            b = g.add_node("Activation", [b, g.add_constant(f"sl{i}", rng.uniform(-0.5, 1.5, c))], {"type": "prelu"})
        x = g.add_node("Add", [b, x])
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


@pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
@pytest.mark.parametrize("kind", ["r2b", "xl"])
def test_tiny_se_l8_res_pre_gate_bit_exact(kind):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _l8_check(tiny_se(23, kind), {"res_pre": True, "identity_blocks": "all"}, tmp)
        assert rep["fused_blocks"] == 3 and len(rep["pre_blocks"]) == 3, rep
        assert "I8Mul" not in rep["int8_ops"] and "I8Add" not in rep["int8_ops"] and "I8AffineSum" not in rep["int8_ops"], rep
        res = [n for n in qg.nodes if n.op == "I8BConv2DRes"]
        assert all(len(n.inputs) == 14 and n.inputs[13] and n.attrs["g_zero"] == -128 for n in res)
