"""Larq-slice operators (QBConv2D, QBDense, ChannelGate; docs/operators_larq.md):
numpy executor vs hand computation, BN folding, and numpy executor vs C++ runtime."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))


def _pm1(rng, shape):
    return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)


def larq_graph(seed: int, fold: bool) -> Graph:
    """Small graph with every op/pattern the Larq adapter emits for the remaining zoo models."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 8, 8, 4])
    # Grouped float conv (MeliusNet stem), PReLU (Real-to-Binary), hardtanh (DoReFa).
    w0 = rng.normal(size=(8, 3, 3, 2)).astype(np.float32)
    y = g.add_node("Conv2D", [x, g.add_constant("w0", w0), ""], {"pads": [1, 1, 1, 1], "groups": 2})
    y = g.add_node("Activation", [y, g.add_constant("slope", rng.random(8) * 0.3)], {"type": "prelu"})
    y = g.add_node("Activation", [y], {"type": "hardtanh", "min": 0.0, "max": 1.0})
    # k-bit conv + BN (foldable), grouped k-bit conv.
    q = g.add_node("QBConv2D", [y, g.add_constant("wq", _pm1(rng, (16, 3, 3, 8)), "bit", "OHWI_bit"),
                                g.add_constant("mq", np.full(16, 0.37 / 3)), ""],
                   {"pads": [1, 1, 1, 1], "act_bits": 2, "act_quant": "dorefa"})
    q = g.add_node("ChannelAffine", [q, g.add_constant("s1", rng.random(16) * 0.2 + 0.05), g.add_constant("t1", rng.random(16))])
    q = g.add_node("QBConv2D", [q, g.add_constant("wq2", _pm1(rng, (16, 3, 3, 4)), "bit", "OHWI_bit"), "",
                                g.add_constant("aq2", rng.normal(size=16))],
                   {"strides": [2, 2], "pads": [0, 1, 0, 1], "groups": 4, "act_bits": 3, "activation": "relu"})
    q = g.add_node("ChannelAffine", [q, g.add_constant("s2", rng.random(16) * 0.05), g.add_constant("t2", rng.random(16) * 0.5)])
    # Squeeze-excite gating (Real-to-Binary): GAP -> Dense relu -> Dense -> ChannelGate(sigmoid).
    z = g.add_node("GlobalAvgPool", [q])
    z = g.add_node("Dense", [z, g.add_constant("d1", rng.normal(size=(4, 16))), ""], {"activation": "relu"})
    z = g.add_node("Dense", [z, g.add_constant("d2", rng.normal(size=(16, 4))), ""])
    q = g.add_node("ChannelGate", [q, z], {"gate": "sigmoid"})
    # MeliusNet improvement merge: x + pad(w, channels before).
    b = g.add_node("BConv2D", [q, g.add_constant("wb", _pm1(rng, (8, 3, 3, 16)), "bit", "OHWI_bit"), "", "", ""],
                   {"pads": [1, 1, 1, 1], "pad_value": 0.0, "input_binarize": "ge"})
    b = g.add_node("Pad", [b], {"pads": [0, 0, 0, 0, 0, 0, 8, 0], "value": 0.0})
    q = g.add_node("Add", [q, b])
    f = g.add_node("Reshape", [q], {"shape": [1, 4 * 4 * 16]})
    d = g.add_node("QBDense", [f, g.add_constant("wd", _pm1(rng, (10, 256)), "bit"), g.add_constant("md", np.full(10, 0.01)), ""],
                   {"act_bits": 2})
    d = g.add_node("ChannelAffine", [d, g.add_constant("s3", rng.random(10) + 0.5), g.add_constant("t3", rng.normal(size=10))])
    probs = g.add_node("Softmax", [d])
    g.set_outputs(d, probs)
    passes.optimize(g, fold=fold)
    return g


def test_qbdense_matches_hand_computation():
    g = Graph()
    x = g.add_input("input", [2, 5])
    w = np.array([[1, -1, 1, 1, -1], [-1, -1, 1, -1, 1]], np.int8)
    y = g.add_node("QBDense", [x, g.add_constant("w", w, "bit"), g.add_constant("m", [0.5, 2.0]), g.add_constant("a", [1.0, -1.0])],
                   {"act_bits": 2})
    g.set_outputs(y)
    xv = np.array([[0.5, 1 / 6, -1.0, 2.0, 0.49], [0.0, 0.8, 0.25, 1.0, 0.5]], np.float32)
    lv = np.round(np.clip(xv, 0, 1) * np.float32(3))  # 0.5*3 = 1.5 -> 2 (half to even)
    assert lv[0, 0] == 2 and lv[1, 4] == 2
    want = (lv @ w.T.astype(np.float32)) * np.array([0.5, 2.0], np.float32) + np.array([1.0, -1.0], np.float32)
    got = executor.run(g, {"input": xv})[y]
    np.testing.assert_array_equal(got, want.astype(np.float32))


def test_channel_gate_sigmoid():
    g = Graph()
    x = g.add_input("input", [1, 2, 2, 3])
    gate = g.add_constant("gate", np.array([[-2.0, 0.0, 3.0]], np.float32))
    y = g.add_node("ChannelGate", [x, gate], {"gate": "sigmoid"})
    g.set_outputs(y)
    xv = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    s = (1 / (1 + np.exp(-np.array([-2.0, 0.0, 3.0])))).astype(np.float32)
    np.testing.assert_array_equal(executor.run(g, {"input": xv})[y], xv * s)


def test_fold_into_qb_ops_preserves_semantics():
    unf, fused = larq_graph(0, False), larq_graph(0, True)
    assert sum(n.op == "ChannelAffine" for n in fused.nodes) < sum(n.op == "ChannelAffine" for n in unf.nodes)
    for n in fused.nodes:
        if n.op in ("QBConv2D", "QBDense"):
            assert len(n.inputs) == 4
    x = np.random.default_rng(1).random((1, 8, 8, 4)).astype(np.float32)
    a = executor.run(unf, {"input": x}, float_dtype=np.float64)[unf.outputs[0]]
    b = executor.run(fused, {"input": x}, float_dtype=np.float64)[fused.outputs[0]]
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("stage", ["reference", "neon", "optimized"])
def test_larq_ops_numpy_matches_cpp(tmp_path, stage, fold):
    g = larq_graph(3, fold)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    rng = np.random.default_rng(7)
    xs = (rng.random(size=(4, 1, 8, 8, 4)) * 1.4 - 0.2).astype(np.float32)
    xs[0, 0, 0, 0, :] = [0.5, 1 / 6, 0.0, 1.0]  # level ties / boundaries
    xs.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(4, 20)
    for i in range(4):
        res = executor.run(g, {"input": xs[i]})
        if stage == "optimized":  # float_opt kernels are not bit-exact (docs/numerical-correctness.md)
            np.testing.assert_allclose(got[i, :10], res[g.outputs[0]].ravel(), rtol=1e-3, atol=1e-3)
        else:
            np.testing.assert_array_equal(got[i, :10], res[g.outputs[0]].ravel())
            # Softmax: exp() of two libms may differ by 1 ulp (same tolerance as test_ir_and_format).
            np.testing.assert_allclose(got[i, 10:], res[g.outputs[1]].ravel(), rtol=1e-5, atol=1e-6)
