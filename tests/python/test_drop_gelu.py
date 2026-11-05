"""passes.drop_gelu_before_sign: GELU used only through sign binarizations.

  * the plain rewrite feeds the GELU input to the binarizations: same binary values except where the evaluated
    GELU underflows to (-)0 (x <= -8.37438965 for the IR's double GELU), where sign_zero now gives -1;
  * keep_zero_below=F64_GELU_NONZERO_FROM keeps that zero set with an exact mask: bit-identical outputs (numpy
    executor here; C++ stages in tests/cpp and on the converted models);
  * GELUs with any other use, a thresholded binarization, or a non-sign-preserving clip are left alone.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, passes  # noqa: E402
from bitino.ir import Graph  # noqa: E402


def mlp(seed: int, consumer_mode: str = "sign_zero", clip: tuple | None = None, extra_use: bool = False, thr: bool = False) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 20, 32])
    w1 = np.where(rng.random((64, 32)) < 0.5, -1, 1).astype(np.int8)
    # A large post_mul spreads fc1 outputs well below the GELU cut-off (as in Bi-ViT) and gives exact zeros.
    h = g.add_node("BDense", [x, g.add_constant("w1", w1, "bit"), g.add_constant("s1", np.full(64, 1.5, np.float32))],
                   {"input_binarize": "ge"}, name="fc1")
    y = g.add_node("Activation", [h], {"type": "gelu"}, name="gelu")
    if clip:
        y = g.add_node("Activation", [y], {"type": "hardtanh", "min": clip[0], "max": clip[1]}, name="clip")
    w2 = np.where(rng.random((16, 64)) < 0.5, -1, 1).astype(np.int8)
    ins = [y, g.add_constant("w2", w2, "bit"), "", ""]
    if thr:
        ins.append(g.add_constant("t2", np.full(64, 0.25, np.float32)))
    o = g.add_node("BDense", ins, {"input_binarize": consumer_mode}, name="fc2")
    if extra_use:
        o = g.add_node("Add", [o, g.add_node("Dense", [y, g.add_constant("wd", rng.normal(size=(16, 64)).astype(np.float32))])])
    g.set_outputs(o)
    return g


def inputs(seed: int) -> dict:
    return {"input": np.random.default_rng(seed).normal(size=(1, 20, 32)).astype(np.float32)}


def gelu64(x: float) -> float:
    return float(np.float32(x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))))


def test_cutoff_constant_matches_double_gelu():
    t = np.float32(passes.F64_GELU_NONZERO_FROM)
    assert gelu64(float(t)) < 0                                      # first non-zero
    assert gelu64(float(np.nextafter(t, np.float32(-1e30)))) == 0    # largest zero (-0)
    for x in np.linspace(-30.0, float(t), 2000, dtype=np.float32)[:-1]:
        assert gelu64(float(x)) == 0


def test_plain_rewrite_removes_gelu_and_matches_exact_sign():
    for mode in ("sign_zero", "ge", "gt"):
        g = mlp(1, mode, clip=(-10.0, 10.0))
        ref = executor.run(g, inputs(2), keep_intermediates=True)
        assert passes.drop_gelu_before_sign(g) == 1
        assert not any(n.attrs.get("type") == "gelu" for n in g.nodes)
        out = executor.run(g, inputs(2), keep_intermediates=True)
        h = ref["fc1:out"]
        # The binarized fc2 input changes only where the double GELU is (-)0 for a negative input.
        changed = np.sign(np.clip(ref["gelu:out"], -10, 10)) != np.sign(h)
        assert np.all(h[changed] <= -8.37438965)
        if not changed.any():
            assert np.array_equal(out[g.outputs[0]], ref[g.outputs[0]])


def test_zero_mask_rewrite_is_bit_identical():
    for seed in range(4):
        g = mlp(seed, "sign_zero")
        ref = executor.run(g, inputs(10 + seed))[g.outputs[0]]
        assert (executor.run(g, inputs(10 + seed), keep_intermediates=True)["fc1:out"] <= -8.37438965).any()
        assert passes.drop_gelu_before_sign(g, keep_zero_below=passes.F64_GELU_NONZERO_FROM) == 1
        assert not any(n.attrs.get("type") == "gelu" for n in g.nodes)
        assert [n.op for n in g.nodes].count("Mul") == 1
        assert np.array_equal(executor.run(g, inputs(10 + seed))[g.outputs[0]], ref)


def test_not_rewritten():
    for g in (mlp(3, extra_use=True), mlp(3, thr=True), mlp(3, clip=(0.0, 10.0)), mlp(3, clip=(-10.0, 0.0))):
        n = len(g.nodes)
        assert passes.drop_gelu_before_sign(g) == 0
        assert len(g.nodes) == n
