"""int8_full v2 quantizer gaps closed on 2026-11-14 (the research notes (not published)):
* BNext units PReLU(BN(BConv)) + r with the integer epilogue (opts int_pre_prelu: I8BConv2DResI in(15) pre_slope);
* RPReLU (a bias) after a squeeze-excitation gate in a fused block (I8BConv2DRes in(14) gate_add);
* the pre stage of a merged pair (BNext's FFN_1 / FFN_2 doubling without SE).
Every tiny model must give the same bytes in the numpy oracle, the C++ reference tier and the host int8_full stage."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_int8_full import BENCH  # noqa: E402
from test_int8_full_v2 import _v2_check  # noqa: E402

from bitino import int8_ops  # noqa: E402
from bitino.ir import Graph  # noqa: E402

needs_bench = pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
V2 = {"res_pre": True, "identity_blocks": "all", "int_epilogue": True}


def _stem(g, rng, c, hw=10, n=2):
    x = g.add_input("input", [n, hw, hw, 3])
    return g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(c, 3, 3, 3)) * 0.4), g.add_constant("b0", rng.normal(size=c))],
                      {"pads": [1, 1, 1, 1]})


def _bconv(g, rng, x, c_in, c_out, k, name, move=True):
    if move:
        x = g.add_node("ChannelAffine", [x, "", g.add_constant(f"{name}mv", rng.normal(size=c_in) * 0.2)])
    w = np.where(rng.random((c_out, k, k, c_in)) < 0.5, -1, 1).astype(np.int8)
    p = k // 2
    return g.add_node("BConv2D", [x, g.add_constant(f"{name}w", w, "bit"), g.add_constant(f"{name}pm", rng.random(c_out) * 0.2 + 0.05),
                                  g.add_constant(f"{name}pa", rng.normal(size=c_out) * 0.3), ""],
                      {"pads": [p, p, p, p], "pad_value": 0, "input_binarize": "ge"})


def _prelu(g, rng, x, c, name, lo=-0.3, hi=1.2):
    return g.add_node("Activation", [x, g.add_constant(f"{name}sl", rng.uniform(lo, hi, c))], {"type": "prelu"})


def _head(g, rng, x, c):
    gp = g.add_node("GlobalAvgPool", [x])
    g.set_outputs(g.add_node("Dense", [gp, g.add_constant("fw", rng.normal(size=(6, c))), g.add_constant("fb", rng.normal(size=6))]))
    return g


def tiny_bnext_units(seed: int, slopes=(-0.3, 1.2), neg_bn: bool = False) -> Graph:
    """BNext FFN units without SE / ELM: y = PReLU(BN(BConv(x + move))) [* BN after the PReLU, signs mixed] + x."""
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 16
    x = _stem(g, rng, c)
    for i in range(3):
        t = _prelu(g, rng, _bconv(g, rng, x, c, c, 3 if i % 2 == 0 else 1, f"u{i}"), c, f"u{i}", *slopes)
        if neg_bn:
            t = g.add_node("ChannelAffine", [t, g.add_constant(f"n{i}s", rng.uniform(-2, 2, c)), g.add_constant(f"n{i}b", rng.normal(size=c) * 0.1)])
        x = g.add_node("Add", [t, x])
    return _head(g, rng, x, c)


def tiny_se_rprelu(seed: int) -> Graph:
    """Real-to-Binary / basic units with an SE gate and RPReLU after the residual add:
    y = move2(PReLU(move1(sigmoid-gate(BConv(x)) + x)))."""
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 16
    x = _stem(g, rng, c)
    for i in range(3):
        t = _bconv(g, rng, x, c, c, 3, f"u{i}")
        gp = g.add_node("GlobalAvgPool", [x], {"keepdims": 1})
        h = g.add_node("Conv2D", [gp, g.add_constant(f"s1w{i}", rng.normal(size=(4, 1, 1, c)) * 0.5), g.add_constant(f"s1b{i}", rng.normal(size=4) * 0.1)],
                       {"activation": "relu"})
        z = g.add_node("Conv2D", [h, g.add_constant(f"s2w{i}", rng.normal(size=(c, 1, 1, 4)) * 0.8), g.add_constant(f"s2b{i}", rng.normal(size=c) * 0.3)])
        t = g.add_node("ChannelGate", [t, z], {"gate": "sigmoid"})
        y = g.add_node("Add", [t, x])
        y = g.add_node("ChannelAffine", [y, "", g.add_constant(f"m1{i}", rng.normal(size=c) * 0.2)])
        y = _prelu(g, rng, y, c, f"r{i}", 0.05, 0.4)
        x = g.add_node("ChannelAffine", [y, "", g.add_constant(f"m2{i}", rng.normal(size=c) * 0.1)])
    return _head(g, rng, x, c)


def tiny_bnext_pair(seed: int) -> Graph:
    """BNext doubling without SE: y_j = PReLU(BN(BConv1x1_j(x + move))) + x, j = 1, 2, Concat(y1, y2) read by the next ELM
    unit's PReLU (residual) and by its binary convolution."""
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 16
    x = _stem(g, rng, c)
    mv = g.add_node("ChannelAffine", [x, "", g.add_constant("pmv", rng.normal(size=c) * 0.2)])
    ys = []
    for j in (1, 2):
        t = _prelu(g, rng, _bconv(g, rng, mv, c, c, 1, f"f{j}", move=False), c, f"f{j}")
        ys.append(g.add_node("Add", [t, x]))
    x = g.add_node("Concat", ys, {"axis": -1})
    c = 2 * c
    r = _prelu(g, rng, x, c, "act1", 0.05, 0.9)
    t = _prelu(g, rng, _bconv(g, rng, x, c, c, 3, "elm"), c, "elm")
    t = g.add_node("Mul", [t, r])
    t = g.add_node("ChannelAffine", [t, g.add_constant("bn2s", rng.uniform(0.5, 2, c)), g.add_constant("bn2b", rng.normal(size=c) * 0.1)])
    x = g.add_node("Add", [t, r])
    return _head(g, rng, x, c)


def test_bconv_res_i_prelu_pre_definition():
    """in(15): t = sigma (dot m1 + pb), t = t >= 0 ? t : mulhi(t, P) << kp, pre = sigma t."""
    rng = np.random.default_rng(1)
    o, k = 6, 3
    x = np.where(rng.random((1, 4, 4, k)) < 0.5, -1.0, 1.0).astype(np.float32)   # a bit tensor (+-1 view)
    w = np.where(rng.random((o, 1, 1, k)) < 0.5, -1, 1).astype(np.int8)
    m1 = rng.integers(-3000, 3000, o)
    b = rng.integers(-10 ** 5, 10 ** 5, o)
    pb = rng.integers(-10 ** 4, 10 ** 4, o)
    neg = np.arange(o) % 2 == 1
    lo = np.where(neg, -2 ** 31, 0)
    hi = np.where(neg, 0, 2 ** 31 - 1)
    P = rng.integers(-2 ** 31 + 1, 2 ** 31 - 1, o)
    ins = [x, w, None, None, None, None, m1, b, None, None, None, None, pb, lo, hi, P]
    y = int8_ops._bconv_res_i(ins, {"strides": [1, 1], "pads": [0, 0, 0, 0], "shift": 6, "pre_slope_shift": 1})
    dot = np.einsum("nhwk,ok->nhwo", x.astype(np.int64), w.reshape(o, k).astype(np.int64))
    sg = np.where(neg, -1, 1)
    t = sg * (dot * m1 + pb)
    t = np.where(t >= 0, t, ((t * P + 2 ** 30) >> 31) * 2)
    acc = sg * t + b
    assert np.array_equal(y, np.clip(((acc + 32) >> 6), -128, 127).astype(np.int8))


@needs_bench
@pytest.mark.parametrize("variant", ["trained", "unit", "neg_bn"])
def test_bnext_units_integer_prelu_pre_stage(variant):
    g = tiny_bnext_units(31, (0.05, 0.9) if variant == "unit" else (-0.3, 1.2), neg_bn=variant == "neg_bn")
    with tempfile.TemporaryDirectory() as tmp:
        rep0, qg0 = _v2_check(g, V2, tmp)   # default: the fp32 epilogue (shipped files keep their bytes)
        assert len(rep0.get("pre_blocks", [])) == 3 and not rep0.get("int_epilogue"), rep0
        rep, qg = _v2_check(g, dict(V2, int_pre_prelu=True), tmp)
        assert len(rep.get("int_epilogue", [])) == 3, rep
        ri = [n for n in qg.nodes if n.op == "I8BConv2DResI"]
        assert len(ri) == 3 and all(len(n.inputs) == 16 and n.inputs[15] for n in ri)
        assert not any(n.op == "I8BConv2DRes" for n in qg.nodes)


@needs_bench
def test_rprelu_after_se_gate():
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_se_rprelu(37), V2, tmp)
        assert rep["fused_blocks"] == 3 and len(rep.get("pre_blocks", [])) == 3, rep
        rs = [n for n in qg.nodes if n.op == "I8BConv2DRes"]
        assert len(rs) == 3 and all(len(n.inputs) == 15 and n.inputs[13] and n.inputs[14] for n in rs)


@needs_bench
@pytest.mark.parametrize("int_pre", [False, True])
def test_merged_pair_pre_stage(int_pre):
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_bnext_pair(41), dict(V2, int_pre_prelu=int_pre), tmp)
        ops = [n.op for n in qg.nodes]
        assert ("I8BConv2DResI" if int_pre else "I8BConv2DRes") in ops
        merged = [n for n in qg.nodes if n.op.startswith("I8BConv2DRes") and qg.constants[n.inputs[1]].shape[0] == 32 and
                  qg.constants[n.inputs[1]].shape[1] == 1]
        assert len(merged) == 1, [(n.op, n.name) for n in qg.nodes]


def tiny_shift_mlp(seed: int) -> Graph:
    """binary MetaFormer blocks with a ShiftViT mixer: s = ChannelShift(x); y = BConv1x1(PReLU(BConv1x1(s))) + s (the
    shifted tensor is both the residual and the binarized input: its binarize-aware copy reads the previous block's
    binarize-aware sum, which the fusion replaces by an I8Lut from the block output)."""
    rng = np.random.default_rng(seed)
    g = Graph()
    c = 16
    x = _stem(g, rng, c)
    for i in range(3):
        s = g.add_node("ChannelShift", [x], {"groups": [0, 2, 0, 1, 2, 4, 0, -1, 4, 6, 1, 0, 6, 8, -1, 0]})
        h = _prelu(g, rng, _bconv(g, rng, s, c, 4 * c, 1, f"a{i}"), 4 * c, f"a{i}", 0.05, 0.9)
        t = _bconv(g, rng, h, 4 * c, c, 1, f"b{i}")
        x = g.add_node("Add", [t, s])
    return _head(g, rng, x, c)


@needs_bench
def test_shift_mixer_identity_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        rep, qg = _v2_check(tiny_shift_mlp(43), dict(V2, int_pre_prelu=True), tmp)
        ops = [n.op for n in qg.nodes]
        assert "I8ChannelShift" in ops and rep["fused_blocks"] >= 3, (rep, ops)
