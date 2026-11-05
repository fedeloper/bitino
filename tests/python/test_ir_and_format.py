"""Python-side tests: packing, serialization, numpy executor vs C++ runtime."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.ir import Graph, pack_bits, unpack_bits  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))


@pytest.mark.parametrize("c", [1, 31, 32, 33, 64, 100])
def test_pack_roundtrip(c):
    rng = np.random.default_rng(c)
    v = np.where(rng.random((3, 5, c)) < 0.5, -1, 1).astype(np.int8)
    p = pack_bits(v)
    assert p.dtype == np.uint32 and p.shape == (3, 5, (c + 31) // 32)
    assert np.array_equal(unpack_bits(p, c), v)
    # Padding bits are zero.
    if c % 32:
        assert np.all(p[..., -1] >> np.uint32(c % 32) == 0)


def test_pack_rejects_non_binary():
    with pytest.raises(ValueError):
        pack_bits(np.array([1, 0, -1]))


def random_graph(seed: int, fold: bool) -> tuple[Graph, tuple]:
    """Small graph touching every CNN op the Larq adapter emits."""
    rng = np.random.default_rng(seed)
    g = Graph()
    h = 9
    x = g.add_input("input", [1, h, h, 3])
    w0 = rng.normal(size=(16, 3, 3, 3)).astype(np.float32)
    y = g.add_node("Conv2D", [x, g.add_constant("w0", w0), g.add_constant("b0", rng.normal(size=16))],
                   {"strides": [2, 2], "pads": [0, 1, 0, 1]})
    y = g.add_node("ChannelAffine", [y, g.add_constant("s0", rng.random(16) + 0.5), g.add_constant("t0", rng.normal(size=16))])
    y = g.add_node("Activation", [y], {"type": "relu"})
    wb = np.where(rng.random((32, 3, 3, 16)) < 0.5, -1, 1).astype(np.int8)
    b = g.add_node("BConv2D", [y, g.add_constant("wb", wb, "bit"), g.add_constant("m", rng.random(32)), g.add_constant("a", rng.normal(size=32)), ""],
                   {"pads": [1, 1, 1, 1], "pad_value": 0.0, "input_binarize": "ge"})
    b = g.add_node("ChannelAffine", [b, g.add_constant("s1", rng.random(32) + 0.5), g.add_constant("t1", rng.normal(size=32))])
    wb2 = np.where(rng.random((32, 3, 3, 32)) < 0.5, -1, 1).astype(np.int8)
    b2 = g.add_node("BConv2D", [b, g.add_constant("wb2", wb2, "bit"), "", "", g.add_constant("thr", rng.normal(size=32) * 0.1)],
                    {"pads": [1, 1, 1, 1], "pad_value": 1.0, "input_binarize": "gt", "activation": "relu"})
    s = g.add_node("Add", [b, b2])
    p = g.add_node("MaxPool2D", [s], {"kernel": [2, 2], "strides": [1, 1]})
    dw = rng.normal(size=(32, 3, 3, 1)).astype(np.float32)
    p = g.add_node("Conv2D", [p, g.add_constant("dw", dw), ""], {"strides": [2, 2], "pads": [1, 1, 1, 1], "groups": 32})
    a = g.add_node("AvgPool2D", [p], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 1, 0, 1], "count_include_pad": 0})
    c = g.add_node("Concat", [a, a], {"axis": -1})
    gp = g.add_node("GlobalAvgPool", [c])
    logits = g.add_node("Dense", [gp, g.add_constant("fc", rng.normal(size=(10, 64))), g.add_constant("fcb", rng.normal(size=10))])
    probs = g.add_node("Softmax", [logits])
    g.set_outputs(logits, probs)
    applied = passes.optimize(g, fold=fold)
    return g, applied


@pytest.mark.parametrize("fold", [False, True])
def test_format_roundtrip_and_fold_preserves_semantics(tmp_path, fold):
    g, applied = random_graph(0, fold)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    assert [n.op for n in g2.nodes] == [n.op for n in g.nodes]
    x = np.random.default_rng(1).normal(size=(1, 9, 9, 3)).astype(np.float32)
    ref, _ = random_graph(0, False)
    out_ref = executor.run(ref, {"input": x})[ref.outputs[0]]
    out = executor.run(g2, {"input": x})[g2.outputs[0]]
    np.testing.assert_allclose(out, out_ref, rtol=1e-4, atol=1e-4)
    if fold:
        assert applied["fold_affine_into_conv"] >= 2


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage", ["reference", "binary_scalar", "all"])
def test_numpy_executor_matches_cpp_runtime(tmp_path, stage):
    g, _ = random_graph(3, True)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    rng = np.random.default_rng(7)
    xs = rng.normal(size=(4, 1, 9, 9, 3)).astype(np.float32)
    xs.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(4, 20)
    for i in range(4):
        res = executor.run(g, {"input": xs[i]})
        np.testing.assert_allclose(got[i, :10], res[g.outputs[0]].ravel(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(got[i, 10:], res[g.outputs[1]].ravel(), rtol=1e-5, atol=1e-6)
