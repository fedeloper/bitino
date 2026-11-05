"""finn_bnn emitter (shared by the FINN-ONNX / QONNX and BNN-PYNQ Theano readers): random small CNV-like networks
with negative BN scales, conv biases folded in the shift, pool before BN (Theano) or after the sign (FINN), are
emitted fused and unfused and compared with a direct float64 implementation of the source semantics.
The fused pool-before-BN path (integer thresholds + input-channel polarity flips in the next conv) must give the
same binary activations as BN -> sign, so the logits agree to float32 rounding of the last affine."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor  # noqa: E402
from bitino.adapters import finn_bnn  # noqa: E402


def _conv_valid(x, w):  # x [H, W, C] float64, w [O, C, 3, 3] -> [H-2, W-2, O]
    h, wd, _ = x.shape
    out = np.zeros((h - 2, wd - 2, w.shape[0]))
    for ky in range(3):
        for kx in range(3):
            out += x[ky:ky + h - 2, kx:kx + wd - 2, :] @ w[:, :, ky, kx].T
    return out


def _pool(x):
    h, w, c = x.shape
    return x[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


def _source(x, layers):
    """Direct semantics: acc -> [pool before BN] -> y = acc*scale + shift -> sign (ge) -> [pool after sign]."""
    a = x[0].astype(np.float64)
    for l in layers:
        if l.kind == "dense":
            a = a.reshape(-1)
            acc = l.w.astype(np.float64) @ a
        else:
            acc = _conv_valid(a, l.w.astype(np.float64))
            if l.pool == "before_bn":
                acc = _pool(acc)
        y = acc * l.scale + l.shift
        if l.final:
            return y
        a = np.where(y >= 0, 1.0, -1.0)
        if l.pool == "after_sign":
            a = _pool(a)
    raise AssertionError


def _rand_layer(rng, name, kind, o, i, pool=None, final=False, k=None):
    w = rng.choice([-1, 1], size=(o, i, 3, 3) if kind == "conv" else (o, i)).astype(np.int8)
    k = k or (i * 9 if kind == "conv" else i)
    scale = rng.uniform(0.2, 2.0, o) * rng.choice([-1.0, 1.0], o)
    # thresholds between attainable integers (tau = -shift/scale = m + 0.37 with parity-free offsets)
    tau = rng.integers(-k // 3, k // 3, o) + 0.37
    shift = -tau * scale
    return finn_bnn.Layer(name, kind, w, scale, shift, pool=pool, final=final)


@pytest.mark.parametrize("pool_kind", ["before_bn", "after_sign"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_emit_matches_source(pool_kind, seed):
    rng = np.random.default_rng(seed)
    layers = [_rand_layer(rng, "conv0", "conv", 8, 3, pool=pool_kind),
              _rand_layer(rng, "conv1", "conv", 16, 8),
              _rand_layer(rng, "conv2", "conv", 12, 16),
              _rand_layer(rng, "dense0", "dense", 10, 12),
              _rand_layer(rng, "dense1", "dense", 5, 10, final=True)]
    inp = finn_bnn.InputSpec("pm1", [1, 12, 12, 3])
    for fused in (False, True):
        g, info = finn_bnn.emit(inp, layers, fused=fused)
        if fused and pool_kind == "before_bn":
            conv1 = next(n for n in g.nodes if n.name == "conv1")
            assert conv1.inputs[4], "pool-before-BN must give the next conv an input threshold"
        for s in range(4):
            x = rng.choice([-1.0, 1.0], size=(1, 12, 12, 3)).astype(np.float32)
            ref = _source(x, layers)
            out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
            np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-5)
            env = executor.run(g, {"input": x}, keep_intermediates=True)
            for pt in info["binarization_points"]:
                assert finn_bnn.ir_activation(env, pt).shape[0] == 1
