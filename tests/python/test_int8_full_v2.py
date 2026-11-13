"""int8_full v2 (docs/int8.md "int8_full v2"): the lin8 post-pass (fused residual blocks on the int8_full linear grids, fp32 or
integer epilogue), the BNext pass (I8BConv2DResI without a residual, I8GatedAdd, I8GapAffine) and the operators' numpy
definitions. Every quantized tiny model must give the same bytes in the numpy oracle, the C++ reference tier and the host
int8_full stage."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_int8_full import BENCH, _records, _run_bench, tiny_bireal, tiny_quicknet, tiny_reactnet, tiny_se  # noqa: E402

from bitino import executor, int8_ops  # noqa: E402
from bitino import format as bfformat  # noqa: E402
from bitino import quantize_int8 as QI  # noqa: E402
from bitino import quantize_l8 as QL  # noqa: E402
from bitino.ir import Graph  # noqa: E402

needs_bench = pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")


def test_lin_encode_definition():
    w = np.array([-1e9, -200.5, -0.5, 0.5, 1.5, 2.5, 126.4, 1e9, -2.0 ** 20 - 3], np.float32)
    y = int8_ops.lin_encode(w, {"y_zero": 3, "qmin": -100, "qmax": 120})
    ref = np.clip(np.rint(w.astype(np.float64)) + 3, -100, 120)
    assert np.array_equal(y.astype(np.int64), ref.astype(np.int64))


def test_gated_add_and_gap_affine_definitions():
    rng = np.random.default_rng(3)
    x, p = (rng.integers(-128, 128, (2, 3, 4, 5)).astype(np.int8) for _ in range(2))
    g = rng.integers(-128, 128, (2, 1, 1, 5)).astype(np.int8)
    kx, kp, b = rng.integers(-3000, 3000, 5), rng.integers(-500, 500, 5), rng.integers(-10 ** 6, 10 ** 6, 5)
    thr = rng.integers(-10 ** 6, 10 ** 6, (2, 5))
    a = {"x_zero": 4, "p_zero": -7, "g_zero": -128, "shift": 9, "y_zero": 5}
    y, b0, b1 = int8_ops._gated_add([x, p, g, kx, kp, b, thr], a)
    acc = (x.astype(np.int64) - 4) * kx + (p.astype(np.int64) + 7) * (g.astype(np.int64) + 128) * kp + b
    assert np.array_equal(y, np.clip(((acc + 256) >> 9) + 5, -128, 127).astype(np.int8))
    assert np.array_equal(b1, np.where(acc >= thr[1], 1.0, -1.0))
    K = rng.integers(-2 ** 20, 2 ** 20, (2, 5))
    B = rng.integers(-2 ** 28, 2 ** 28, 5)
    yg = int8_ops._gap_affine([x, p, K, B], {"zeros": [1, -2], "shift": 12, "y_zero": -3})
    s = (x.astype(np.int64) - 1).sum(axis=(1, 2), keepdims=True) * K[0] + (p.astype(np.int64) + 2).sum(axis=(1, 2), keepdims=True) * K[1] + B
    assert np.array_equal(yg, np.clip(((s + 2048) >> 12) - 3, -128, 127).astype(np.int8))


def _v2_check(g: Graph, opts: dict, tmp: str, gated: bool = False):
    path = os.path.join(tmp, "f.bitino")
    bfformat.write(g, path)
    cal = _records(g, 24, 1)
    stats = QI.calibrate(path, cal, 99.99, jobs=2)
    shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
    Q = QI.rewrite(g, stats, shapes, {})
    E = QI._Emitter(Q, {})
    o = dict(opts, code="lin8")
    qg, rep = QL.fuse(E.emit()[0], Q, E, o)
    if gated:
        qg, grep = QL.fuse_gated(qg, Q, E, o)
        rep["gated"] = grep
    qpath = os.path.join(tmp, "q.bitino")
    bfformat.write(qg, qpath)
    qg = bfformat.read(qpath)
    for rec in _records(g, 4, 99):
        env = executor.run(qg, QI._feeds(qg, rec))
        ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
        for stage in ("int8_full_reference", "int8_full"):
            assert _run_bench(qpath, rec, stage) == ref, stage
    return rep, qg


@needs_bench
@pytest.mark.parametrize("int_epi", [False, True])
def test_v2_tiny_reactnet_bit_exact(int_epi):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_reactnet(11), {"int_epilogue": int_epi}, tmp)
        assert rep["fused_blocks"] == 4 and rep["code"] == "lin8", rep
        ops = [n.op for n in qg.nodes]
        assert ops.count("I8BConv2DResI" if int_epi else "I8BConv2DRes") == 4
        assert "I8AffineSum" not in ops and "I8Add" not in ops
        if not int_epi:
            assert all(n.attrs["output"] == "i8" and n.attrs["r_code"] == "linear" for n in qg.nodes if n.op == "I8BConv2DRes")
        # no l8 tensor anywhere: the fused outputs keep the int8_full grids
        assert not any(n.attrs.get("output") == "l8" or n.attrs.get("x_code") == "l8" for n in qg.nodes)


@needs_bench
@pytest.mark.parametrize("relu", [False, True])
def test_v2_tiny_bireal_identity_blocks_bit_exact(relu):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_bireal(13, relu), {"int_epilogue": True}, tmp)
        assert rep["fused_blocks"] == 4 and len(rep["int_epilogue"]) == 4, rep
        res = [n for n in qg.nodes if n.op == "I8BConv2DResI"]
        assert all(n.attrs.get("bit_src", "sum") == ("out" if relu else "sum") for n in res if len(n.outputs) == 2)


@needs_bench
def test_v2_tiny_quicknet_pre_clamp_bit_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_quicknet(19), {"res_pre": True, "int_epilogue": True}, tmp)
        assert rep["fused_blocks"] == 4 and len(rep["int_epilogue"]) == 4, rep
        res = [n for n in qg.nodes if n.op == "I8BConv2DResI"]
        assert all(len(n.inputs) == 15 for n in res)  # the clamp pre stage (inputs 12 .. 14)
        lo = np.concatenate([np.asarray(qg.constants[n.inputs[13]].data).reshape(-1) for n in res])
        assert set(lo.tolist()) <= {0, -2 ** 31} and (lo < 0).any()  # BN scales of both signs: max(., 0) and min(., 0)


@needs_bench
@pytest.mark.parametrize("kind", ["r2b", "xl"])
def test_v2_tiny_se_gates_keep_fp32_epilogue(kind):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_se(23, kind), {"res_pre": True, "identity_blocks": "all", "int_epilogue": True}, tmp)
        assert rep["fused_blocks"] == 3 and not rep.get("int_epilogue"), rep  # gated pre stages: fp32 epilogue, output i8
        assert all(n.attrs["output"] == "i8" for n in qg.nodes if n.op == "I8BConv2DRes")


def tiny_bnext(seed: int) -> Graph:
    """BNext-style blocks: p = PReLU(BConv(x + move)); mix = a x + (1 - a) p; G = sigmoid(Conv(relu(Conv(GAP(mix)))));
    y = norm(p * G) + x; the next block binarizes y (twice, with different move biases, like BNext's two FFN branches)."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [2, 10, 10, 3])
    x = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(16, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=16))],
                   {"pads": [1, 1, 1, 1]})
    c = 16
    for i in range(3):
        mv = g.add_node("ChannelAffine", [x, "", g.add_constant(f"mv{i}", rng.normal(size=c) * 0.2)])
        w = np.where(rng.random((c, 3, 3, c)) < 0.5, -1, 1).astype(np.int8)
        b = g.add_node("BConv2D", [mv, g.add_constant(f"bw{i}", w, "bit"), g.add_constant(f"pm{i}", rng.random(c) * 0.2 + 0.05),
                                    g.add_constant(f"pa{i}", rng.normal(size=c) * 0.3), ""],
                       {"pads": [1, 1, 1, 1], "pad_value": 0, "input_binarize": "ge"})
        p = g.add_node("Activation", [b, g.add_constant(f"sl{i}", rng.uniform(-0.3, 1.2, c))], {"type": "prelu"})
        s = rng.uniform(0, 1, c)
        mix = g.add_node("Add", [g.add_node("ChannelAffine", [x, g.add_constant(f"ms{i}", s), ""]),
                                 g.add_node("ChannelAffine", [p, g.add_constant(f"mp{i}", 1 - s), ""])])
        gp = g.add_node("GlobalAvgPool", [mix], {"keepdims": 1})
        h = g.add_node("Conv2D", [gp, g.add_constant(f"s1w{i}", rng.normal(size=(4, 1, 1, c)) * 0.5), g.add_constant(f"s1b{i}", rng.normal(size=4) * 0.1)],
                       {"activation": "relu"})
        gt = g.add_node("Conv2D", [h, g.add_constant(f"s2w{i}", rng.normal(size=(c, 1, 1, 4)) * 0.8), g.add_constant(f"s2b{i}", rng.normal(size=c) * 0.3)])
        se = g.add_node("ChannelGate", [p, gt], {"gate": "sigmoid"})
        nm = g.add_node("ChannelAffine", [se, g.add_constant(f"ns{i}", rng.uniform(0.5, 3, c)), g.add_constant(f"nb{i}", rng.normal(size=c) * 0.2)])
        x = g.add_node("Add", [nm, x])
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


@needs_bench
def test_v2_tiny_bnext_gated_bit_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_bnext(29), {"int_epilogue": True}, tmp, gated=True)
        gr = rep["gated"]
        assert len(gr["prelu_convs"]) == 3 and len(gr["gated_adds"]) == 3 and len(gr["gap_affines"]) == 3, gr
        ops = [n.op for n in qg.nodes]
        assert "I8Mul" not in ops and "I8AffineSum" not in ops
        assert all(n.inputs[5] == "" and n.inputs[8] == "" and n.inputs[10] for n in qg.nodes if n.op == "I8BConv2DResI")  # no residual, PReLU
        assert sum(len(n.outputs) > 1 for n in qg.nodes if n.op == "I8GatedAdd") == 2  # bits for the next blocks


def tiny_bhvit(seed: int, rank3: bool = False) -> Graph:
    """BHViT-style token MLP on [N, H, W, C]: h = BDense(x) (C -> 4C) + repeat(x) (4 copies) -> move1 -> PReLU -> move2 (read
    by the pooling of the 4 groups), PReLU output binarized by BDense (4C -> C) + pool -> PReLU -> next block."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 6, 6, 3])
    x = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(16, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=16))],
                   {"pads": [1, 1, 1, 1]})
    c = 16
    if rank3:   # [N, T, C] tokens (BHViT's attention projections): the fused op works on [N, T, 1, C] views
        x = g.add_node("Reshape", [x], {"shape": [1, 36, c]})
    for i in range(2):
        w1 = np.where(rng.random((4 * c, c)) < 0.5, -1, 1).astype(np.int8)
        h = g.add_node("BDense", [x, g.add_constant(f"dw{i}", w1, "bit"), g.add_constant(f"dm{i}", rng.random(4 * c) * 0.2 + 0.05),
                                  g.add_constant(f"da{i}", rng.normal(size=4 * c) * 0.3), ""], {"input_binarize": "ge"})
        h = g.add_node("Add", [h, g.add_node("Concat", [x, x, x, x], {"axis": -1})])
        h = g.add_node("ChannelAffine", [h, "", g.add_constant(f"m1_{i}", rng.normal(size=4 * c) * 0.2)])
        p = g.add_node("Activation", [h, g.add_constant(f"sl{i}", rng.uniform(-0.5, 3.0, 4 * c))], {"type": "prelu"})
        m2 = g.add_node("ChannelAffine", [p, "", g.add_constant(f"m2_{i}", rng.normal(size=4 * c) * 0.2)])
        pool = g.add_node("ReduceMean", [g.add_node("Reshape", [m2], {"shape": [1, 36, c, 4] if rank3 else [1, 6, 6, c, 4]})],
                          {"axis": -1, "keepdims": 0})
        w2 = np.where(rng.random((c, 4 * c)) < 0.5, -1, 1).astype(np.int8)
        o = g.add_node("BDense", [p, g.add_constant(f"ow{i}", w2, "bit"), g.add_constant(f"om{i}", rng.random(c) * 0.1 + 0.02),
                                  g.add_constant(f"oa{i}", rng.normal(size=c) * 0.3), ""], {"input_binarize": "ge"})
        o = g.add_node("Add", [o, pool])
        x = g.add_node("Activation", [o, g.add_constant(f"so{i}", rng.uniform(0.0, 1.5, c))], {"type": "prelu"})
    if rank3:
        x = g.add_node("Reshape", [x], {"shape": [1, 6, 6, c]})
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


@needs_bench
@pytest.mark.parametrize("rank3", [False, True])
def test_v2_tiny_bhvit_dense_blocks_bit_exact(rank3):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.bitino")
        g = tiny_bhvit(31, rank3)
        bfformat.write(g, path)
        cal = _records(g, 24, 1)
        stats = QI.calibrate(path, cal, 99.99, jobs=2)
        shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
        Q = QI.rewrite(g, stats, shapes, {})
        E = QI._Emitter(Q, {})
        o = {"code": "lin8", "int_epilogue": True}
        qg, rep = QL.fuse(E.emit()[0], Q, E, o)
        qg, drep = QL.fuse_dense(qg, Q, E, o)
        assert len(drep["dense_blocks"]) == 4, drep
        res = [n for n in qg.nodes if n.op == "I8BConv2DResI"]
        assert sorted(int(n.attrs.get("r_tile", 1)) for n in res) == [1, 1, 4, 4]
        assert any(len(n.outputs) == 2 and n.attrs.get("bit_src") == "out" for n in res)   # bits of y for the second BDense
        assert not any(n.op in ("I8BDense", "I8Concat") for n in qg.nodes)
        if rank3:   # int8 and bit Reshape views around the fused ops
            assert any(n.op == "Reshape" and n.outputs[0].endswith(":bits") for n in qg.nodes)
        qpath = os.path.join(tmp, "q.bitino")
        bfformat.write(qg, qpath)
        qg = bfformat.read(qpath)
        for rec in _records(g, 4, 99):
            env = executor.run(qg, QI._feeds(qg, rec))
            ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in qg.outputs)
            for stage in ("int8_full_reference", "int8_full"):
                assert _run_bench(qpath, rec, stage) == ref, stage
