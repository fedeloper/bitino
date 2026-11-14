"""JPEG-DCT input ops (2026-11-14, the research notes (not published) section 15): I8DctYuv420 (partial 8x8 DCT of a YUV
4:2:0 frame, q90 quantization) and I8BConvBP (bit-serial binary 1x1 on integer inputs): the numpy oracle equals the host
C++ reference and int8_full stages byte for byte; the oracle's definitions checked directly."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_int8_full import BENCH, _run_bench  # noqa: E402

from bitino import executor, int8_ops  # noqa: E402
from bitino import format as bfformat  # noqa: E402
from bitino.ir import Graph  # noqa: E402

needs_bench = pytest.mark.skipif(not os.path.exists(BENCH), reason="build/host/bitino_bench missing")
ZZ = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11]


def _q(g, x):
    return g.add_node("I8Quantize", [x], {"scale": 1.0, "zero": 0})


def _check(g, feeds, tmp):
    p = os.path.join(tmp, "m.bitino")
    bfformat.write(g, p)
    g = bfformat.read(p)
    env = executor.run(g, feeds)
    ref = b"".join(np.ascontiguousarray(env[o]).tobytes() for o in g.outputs)
    rec = b"".join(np.ascontiguousarray(feeds[i["name"]], np.float32).tobytes() for i in g.inputs)
    for st in ("int8_full_reference", "int8_full"):
        assert _run_bench(p, rec, st) == ref, st


def test_bconv_bp_definition():
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, (1, 3, 3, 24)).astype(np.int8)
    w = np.where(rng.random((8, 1, 1, 24)) < 0.5, -1, 1).astype(np.int8)
    m1, b = rng.integers(-300, 300, 8), rng.integers(-1000, 1000, 8)
    y = int8_ops._bconv_bp([x, w, m1, b], {"x_zero": 5, "shift": 6, "y_zero": 3})
    acc = (x.astype(np.int64) - 5).reshape(-1, 24) @ w.reshape(8, 24).T.astype(np.int64)
    ref = np.clip(((acc * m1 + b + 32) >> 6) + 3, -128, 127).reshape(1, 3, 3, 8)
    assert np.array_equal(y, ref)


def test_dct_definition_matches_float_dct():
    """the fixed-point codes are the q-quantized JPEG DCT within 1 code"""
    rng = np.random.default_rng(1)
    blk = rng.integers(-128, 128, (1, 16, 16)).astype(np.int64)
    c = rng.integers(-128, 128, (1, 8, 8, 2)).astype(np.int64)
    q = np.full(64, 4)
    out = int8_ops._dct_yuv420([blk[..., None].astype(np.int8), c.astype(np.int8)],
                               {"y_sel": ZZ, "c_sel": ZZ[:5], "qy": q, "qc": q, "shifts": [0] * 24})
    k = np.arange(8)
    C = 0.5 * np.cos((2 * k[None, :] + 1) * k[:, None] * np.pi / 16)
    C[0] /= np.sqrt(2)
    b = blk[0].reshape(2, 8, 2, 8).transpose(0, 2, 1, 3)
    F = np.einsum("vy,ijyx,ux->ijvu", C, b, C)
    for i, n in enumerate(ZZ):
        exact = F[..., n // 8, n % 8] / 4
        assert np.max(np.abs(out[0, :, :, i] - np.clip(np.round(exact), -128, 127))) <= 1


@needs_bench
def test_bconv_bp_oracle_equals_host():
    rng = np.random.default_rng(2)
    g = Graph()
    x = _q(g, g.add_input("x", [1, 7, 6, 24]))
    w = np.where(rng.random((40, 1, 1, 24)) < 0.5, -1, 1).astype(np.int8)
    y = g.add_node("I8BConvBP", [x, g.add_constant("w", w, "bit"), g.add_constant("m1", rng.integers(-2000, 2000, 40).astype(np.int32), "i32"),
                                 g.add_constant("b", rng.integers(-10 ** 5, 10 ** 5, 40).astype(np.int32), "i32"),
                                 g.add_constant("t", rng.integers(-10 ** 5, 10 ** 5, 40).astype(np.int32), "i32")],
                   {"x_zero": -3, "shift": 9, "y_zero": 2}, n_outputs=2)
    # the bit output read by a binary 1x1 (as the next layer would), so both outputs reach the compared int8 tensors
    w2 = np.where(rng.random((16, 1, 1, 40)) < 0.5, -1, 1).astype(np.int8)
    z = g.add_node("I8BConv2D", [y[1], g.add_constant("w2", w2, "bit"), g.add_constant("pm", np.full(16, 0.5, np.float32)),
                                 g.add_constant("pa", np.zeros(16, np.float32))],
                   {"strides": [1, 1], "pads": [0, 0, 0, 0], "dilations": [1, 1], "groups": 1, "pad_value": 0.0, "qmin": -128, "qmax": 127})
    g.set_outputs(y[0], z)
    with tempfile.TemporaryDirectory() as tmp:
        _check(g, {"x": rng.integers(-128, 128, (1, 7, 6, 24)).astype(np.float32)}, tmp)


@needs_bench
def test_dct_oracle_equals_host():
    rng = np.random.default_rng(3)
    g = Graph()
    yi, ci = g.add_input("y", [1, 32, 48, 1]), g.add_input("c", [1, 16, 24, 2])
    q = [int(v) for v in rng.integers(1, 30, 64)]
    o = g.add_node("I8DctYuv420", [_q(g, yi), _q(g, ci)], {"y_sel": ZZ, "c_sel": ZZ[:5], "qy": q, "qc": q[::-1],
                                                             "shifts": [int(v) for v in rng.integers(0, 3, 24)]})
    g.set_outputs(o)
    with tempfile.TemporaryDirectory() as tmp:
        _check(g, {"y": rng.integers(-128, 128, (1, 32, 48, 1)).astype(np.float32),
                   "c": rng.integers(-128, 128, (1, 16, 24, 2)).astype(np.float32)}, tmp)
