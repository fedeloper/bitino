"""int8_full ops for accuracy-carrying glue (docs/int8.md "int8_full ops for accuracy-carrying glue", 2026-11-13): the numpy
definitions of I8ChannelShift, I8BinarizeDyn, I8SimAM and the full-resolution / table-gated I8GatedAdd, and tiny models with
the float ops ChannelShift, InstanceThreshold, SimAM and star / GLU gating quantized through the int8_full v2 path: the
numpy oracle, the C++ reference tier and the host int8_full stage give the same bytes."""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_int8_full import BENCH, _records, _run_bench  # noqa: E402

from bitino import executor, int8_ops  # noqa: E402
from bitino import format as bfformat  # noqa: E402
from bitino import quantize_int8 as QI  # noqa: E402
from bitino import quantize_l8 as QL  # noqa: E402
from bitino.ir import Graph  # noqa: E402

needs_bench = pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")


def test_channel_shift_definition():
    rng = np.random.default_rng(1)
    x = rng.integers(-128, 128, (2, 4, 5, 13)).astype(np.int8)
    y = int8_ops._channel_shift([x], {"groups": [0, 2, 0, 1, 2, 4, 1, 0, 5, 6, -2, 3], "fill_q": 7})
    for n in range(2):
        for i in range(4):
            for j in range(5):
                for c in range(13):
                    dy, dx = (0, 1) if c < 2 else (1, 0) if c < 4 else (-2, 3) if c == 5 else (0, 0)
                    si, sj = i - dy, j - dx
                    want = x[n, si, sj, c] if 0 <= si < 4 and 0 <= sj < 5 else 7
                    assert y[n, i, j, c] == want
    xf = x.astype(np.float32)
    yf = executor.channel_shift(xf, [[0, 2, 0, 1], [2, 4, 1, 0], [5, 6, -2, 3]], 0.0)
    assert np.array_equal(yf[:, :, 1:, :2], xf[:, :, :-1, :2]) and np.all(yf[:, :, 0, :2] == 0)


def test_binarize_dyn_definition_and_threshold_semantics():
    rng = np.random.default_rng(2)
    x = rng.integers(-128, 128, (2, 3, 4, 6)).astype(np.int8)
    c, hw, F, G, z = 6, 12, 10, 4, -3
    t0, A, B = rng.integers(-2 ** 20, 2 ** 20, c), rng.integers(-2 ** 12, 2 ** 12, c), rng.integers(-2 ** 10, 2 ** 10, c)
    o, ko = rng.integers(-128, 128, (2, 1, 1, c)).astype(np.int8), rng.integers(-2 ** 16, 2 ** 16, c)
    y = int8_ops._binarize_dyn([x, t0, A, B, o, ko], {"x_zero": z, "o_zero": 5, "shift": F, "sqrt_shift": G})
    for n in range(2):
        for ch in range(c):
            q = x[n, :, :, ch].astype(np.int64).reshape(-1) - z
            s1, s2 = int(q.sum()), int((q * q).sum())
            acc = int(t0[ch]) + int(A[ch]) * s1 + int(B[ch]) * math.isqrt((hw * s2 - s1 * s1) << (2 * G)) + int(ko[ch]) * (int(o[n, 0, 0, ch]) - 5)
            want = np.where(q * (hw << F) >= acc, 1.0, -1.0)
            assert np.array_equal(y[n, :, :, ch].reshape(-1), want)


def test_simam_definition_close_to_float():
    """I8SimAM with the quantizer's table follows the float SimAM to within the gate's 1/16 index step."""
    rng = np.random.default_rng(3)
    xf = rng.normal(size=(1, 7, 7, 8)).astype(np.float32) * 2
    s, z = 8.0 / 255, 0
    q = np.clip(np.rint(xf / s) + z, -128, 127).astype(np.int8)
    k = np.arange(256, dtype=np.float64)
    gate = 1.0 / (1.0 + np.exp(-(0.5 + (k + 0.5) / 16.0)))
    gate[255] = 1.0
    tab = (np.clip(np.rint(gate * 255.0), 0, 255) - 128).astype(np.int8)[None, :]
    sy = s
    m, sh = int8_ops.quantize_multiplier(s / (255.0 * sy))
    L = max(1, round(1e-4 * 49 * 49 / (s * s)))
    y = int8_ops._simam([q, tab], {"x_zero": z, "lam_q": L, "g_zero": -128, "mult": m, "shift": sh, "y_zero": 0})
    yf = executor.simam((q.astype(np.float32) - z) * s, 1e-4, np.float32)
    assert np.max(np.abs(y.astype(np.float64) * sy - yf)) < 0.05 * np.abs(yf).max() + 2 * sy


def test_gated_add_full_gate_and_lut_definition():
    rng = np.random.default_rng(4)
    x, p, g = (rng.integers(-128, 128, (2, 3, 4, 5)).astype(np.int8) for _ in range(3))
    kx, kp, b = rng.integers(-3000, 3000, 5), rng.integers(-500, 500, 5), rng.integers(-10 ** 6, 10 ** 6, 5)
    tab = rng.integers(-128, 128, (1, 256)).astype(np.int8)
    a = {"x_zero": 4, "p_zero": -7, "g_zero": -100, "shift": 9, "y_zero": 5}
    y = int8_ops._gated_add([x, p, g, kx, kp, b, None, tab], a)
    gv = tab[0][g.astype(np.int64) + 128].astype(np.int64)
    acc = (x.astype(np.int64) - 4) * kx + (p.astype(np.int64) + 7) * (gv + 100) * kp + b
    assert np.array_equal(y, np.clip(((acc + 256) >> 9) + 5, -128, 127).astype(np.int8))


# ------------------------------------------------------------------------------------------------ tiny models
def _bconv(g, x, rng, ci, co, k, name, thr=""):
    w = np.where(rng.random((co, k, k, ci)) < 0.5, -1, 1).astype(np.int8)
    p = (k - 1) // 2
    return g.add_node("BConv2D", [x, g.add_constant(f"{name}w", w, "bit"), g.add_constant(f"{name}m", rng.random(co) * 0.2 + 0.05),
                                  g.add_constant(f"{name}a", rng.normal(size=co) * 0.3), thr],
                      {"pads": [p, p, p, p], "pad_value": 0, "input_binarize": "ge"})


def _stem(g, rng, c, n=1, hw=10):
    x = g.add_input("input", [n, hw, hw, 3])
    return g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(c, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=c))],
                      {"pads": [1, 1, 1, 1]})


def _head(g, rng, x, c):
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


def tiny_shift(seed: int) -> Graph:
    """Binary MetaFormer with a ShiftViT token mixer: s = ChannelShift(x); y = PReLU(s + BConv1x1(move(s)) ...) twice."""
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 24
    x = _stem(g, rng, c)
    for i in range(2):
        s = g.add_node("ChannelShift", [x], {"groups": [[0, 2, 0, 1], [2, 4, 0, -1], [4, 6, 1, 0], [6, 8, -1, 0]]})
        mv = g.add_node("ChannelAffine", [s, "", g.add_constant(f"mv{i}", rng.normal(size=c) * 0.2)])
        b = _bconv(g, mv, rng, c, c, 1, f"b{i}")
        a = g.add_node("Add", [b, s])
        x = g.add_node("Activation", [a, g.add_constant(f"sl{i}", rng.uniform(0.05, 0.9, c))], {"type": "prelu"})
    return _head(g, rng, x, c)


def tiny_insta(seed: int, offset: bool = False) -> Graph:
    """Residual blocks whose binarization threshold is instance-aware: y = x + BConv3x3(InstanceThreshold(x)) (two readers
    with different thresholds in the second block)."""
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 16
    x = _stem(g, rng, c, n=2)
    for i in range(2):
        cs = lambda nm, sc: g.add_constant(f"it_{nm}{i}", rng.normal(size=c) * sc)   # noqa: E731
        ins = [x, cs("t", 0.2), cs("a", 0.5), cs("b", 0.3)]
        if offset:
            gp = g.add_node("GlobalAvgPool", [x], {"keepdims": 1})
            ins.append(g.add_node("ChannelAffine", [gp, cs("os", 0.5), ""]))
        t = g.add_node("InstanceThreshold", ins)
        b = _bconv(g, t, rng, c, c, 3, f"b{i}")
        if i == 1:
            b2 = _bconv(g, t, rng, c, c, 3, f"c{i}", g.add_constant(f"thr{i}", rng.normal(size=c) * 0.1))
            b = g.add_node("Add", [b, b2])
        x = g.add_node("Add", [b, x])
    return _head(g, rng, x, c)


def tiny_star(seed: int, sigmoid: bool = True) -> Graph:
    """Star / GLU gating at the residual: y = x + norm(PReLU(BConv(x)) * sigmoid(BConv(x))); the next block binarizes y."""
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 16
    x = _stem(g, rng, c, n=2)
    for i in range(3):
        mv = g.add_node("ChannelAffine", [x, "", g.add_constant(f"mv{i}", rng.normal(size=c) * 0.2)])
        p = g.add_node("Activation", [_bconv(g, mv, rng, c, c, 3, f"p{i}"), g.add_constant(f"sl{i}", rng.uniform(-0.3, 1.2, c))],
                       {"type": "prelu"})
        q = _bconv(g, mv, rng, c, c, 1, f"q{i}")
        gq = g.add_node("Activation", [q], {"type": "sigmoid"}) if sigmoid else q
        m = g.add_node("Mul", [p, gq])
        nm = g.add_node("ChannelAffine", [m, g.add_constant(f"ns{i}", rng.uniform(0.5, 3, c)), g.add_constant(f"nb{i}", rng.normal(size=c) * 0.2)])
        x = g.add_node("Add", [nm, x])
    return _head(g, rng, x, c)


def tiny_simam(seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 16
    x = _stem(g, rng, c)
    for i in range(2):
        mv = g.add_node("ChannelAffine", [x, "", g.add_constant(f"mv{i}", rng.normal(size=c) * 0.2)])
        b = g.add_node("SimAM", [_bconv(g, mv, rng, c, c, 3, f"b{i}")], {"lambda": 1e-4})
        x = g.add_node("Add", [b, x])
    return _head(g, rng, x, c)


def _check(g: Graph, tmp: str, opts: dict):
    path = os.path.join(tmp, "f.bitino")
    bfformat.write(g, path)
    cal = _records(g, 24, 1)
    stats = QI.calibrate(path, cal, 99.99, jobs=2)
    shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
    Q = QI.rewrite(g, stats, shapes, {})
    E = QI._Emitter(Q, {})
    o = dict(opts, code="lin8", int_epilogue=True, res_pre=True, identity_blocks="all")
    qg, rep = QL.fuse(E.emit()[0], Q, E, o)
    qg, grep = QL.fuse_gated(qg, Q, E, o)
    rep["gated"] = grep
    qpath = os.path.join(tmp, "q.bitino")
    bfformat.write(qg, qpath)
    qg = bfformat.read(qpath)
    fl = []
    for rec in _records(g, 4, 99):
        env = executor.run(qg, QI._feeds(qg, rec))
        ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
        for stage in ("int8_full_reference", "int8_full"):
            assert _run_bench(qpath, rec, stage) == ref, stage
        yf = executor.run(g, QI._feeds(g, rec))[g.outputs[0]].reshape(-1)
        yq = env[qg.outputs[0]].reshape(-1)
        fl.append(np.corrcoef(yf, yq)[0, 1])
    return rep, qg, float(np.mean(fl))


@needs_bench
def test_tiny_shift_bit_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg, r = _check(tiny_shift(41), tmp, {})
        # the shifted tensor is both the residual and binarized: a binarize-aware second version (quantize_int8 plan_splits)
        assert [n.op for n in qg.nodes].count("I8ChannelShift") in (2, 4)
        assert r > 0.9, r


@needs_bench
@pytest.mark.parametrize("offset", [False, True])
def test_tiny_insta_bit_exact(offset):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg, r = _check(tiny_insta(43, offset), tmp, {})
        ops = [n.op for n in qg.nodes]
        assert ops.count("I8BinarizeDyn") == 3, ops   # one per distinct (mode, thresholds) reader binarization
        dyn = {o for n in qg.nodes if n.op == "I8BinarizeDyn" for o in n.outputs}
        readers = [n for n in qg.nodes if n.inputs and n.inputs[0] in dyn]
        assert len(readers) == 3 and all("x_scale" not in n.attrs for n in readers)
        if offset:
            assert all(len(n.inputs) == 6 for n in qg.nodes if n.op == "I8BinarizeDyn")
        assert r > 0.8, r


@needs_bench
@pytest.mark.parametrize("sigmoid", [True, False])
def test_tiny_star_bit_exact(sigmoid):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg, r = _check(tiny_star(47, sigmoid), tmp, {"star": True})
        assert len(rep["gated"].get("star_adds", [])) == 3, rep["gated"]
        ops = [n.op for n in qg.nodes]
        assert "I8Mul" not in ops and ops.count("I8GatedAdd") == 3
        ga = [n for n in qg.nodes if n.op == "I8GatedAdd"]
        assert all((len(n.inputs) == 8) == sigmoid for n in ga)   # the sigmoid table fused as g_lut
        assert sum(len(n.outputs) > 1 for n in ga) == 2           # bits for the next blocks
        if sigmoid:
            assert "I8Lut" not in ops
        assert r > 0.8, r
    with tempfile.TemporaryDirectory() as tmp:   # off by default
        rep, qg, _ = _check(tiny_star(47, sigmoid), tmp, {})
        assert "star_adds" not in rep["gated"] and "I8Mul" in [n.op for n in qg.nodes]


@needs_bench
def test_tiny_simam_bit_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg, r = _check(tiny_simam(53), tmp, {})
        assert [n.op for n in qg.nodes].count("I8SimAM") == 2
        assert r > 0.8, r
