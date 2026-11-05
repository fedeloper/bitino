"""NNCF binarization mapping (bitino.adapters.torch_nncf): a {0, s}-activation x {+-alpha}-weight conv with
zero padding equals BConv2D(input_binarize=gt, threshold=t, pad_value=-1) with post_mul = alpha*s/2 and
post_add = alpha*s*Wsum/2 (numpy executor, float64; exact up to float64 rounding). The end-to-end check
against official NNCF is tools/convert_nncf.py (Level 3, float64 constants <= 1e-13)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor  # noqa: E402
from bitino.ir import Graph  # noqa: E402


def _direct(x, t, s, sign, alpha, stride, pad):
    """NNCF semantics in float64: a = s * (x > t_c), zero padding, conv with alpha_o * sign."""
    a = np.where(x > t, s, 0.0)
    n, h, w, c = a.shape
    o, kh, kw, _ = sign.shape
    ap = np.zeros((n, h + 2 * pad, w + 2 * pad, c))
    ap[:, pad:pad + h, pad:pad + w] = a
    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    out = np.zeros((n, oh, ow, o))
    wb = sign * alpha[:, None, None, None]
    for y in range(oh):
        for xx in range(ow):
            patch = ap[:, y * stride:y * stride + kh, xx * stride:xx * stride + kw, :]
            out[:, y, xx, :] = np.einsum("nhwc,ohwc->no", patch, wb)
    return out


@pytest.mark.parametrize("stride,pad,per_channel", [(1, 1, True), (2, 1, True), (1, 1, False), (1, 0, True)])
def test_zero_s_activation_mapping(stride, pad, per_channel):
    rng = np.random.default_rng(stride * 10 + pad + per_channel)
    c, o = 37, 9
    x = np.maximum(rng.normal(size=(2, 7, 6, c)), 0.0)                 # post-ReLU input
    t = rng.uniform(0.05, 1.0, c)
    x[0, 1, 1, :] = t                                                   # exact ties map to 0 (x > t is false)
    s = 1.7
    sign = np.where(rng.normal(size=(o, 3, 3, c)) > 0, 1, -1).astype(np.int8)
    alpha = rng.uniform(0.01, 0.1, o) if per_channel else np.full(o, 0.03)
    wsum = sign.reshape(o, -1).astype(np.int64).sum(1)
    g = Graph()
    xi = g.add_input("input", list(x.shape))
    y = g.add_node("BConv2D", [xi, g.add_constant("w", sign, dtype="bit"), g.add_constant("mul", alpha * s / 2),
                               g.add_constant("add", alpha * s * wsum / 2), g.add_constant("thr", t)],
                   {"strides": [stride, stride], "dilations": [1, 1], "pads": [pad] * 4, "groups": 1, "pad_value": -1,
                    "input_binarize": "gt", "activation": "none", "output": "f32"})
    g.set_outputs(y)
    for name, v in {"mul": alpha * s / 2, "add": alpha * s * wsum / 2, "thr": t}.items():
        g.constants[name].data = np.asarray(v, np.float64)              # float64 constants
    got = executor.run(g, {"input": x}, float_dtype=np.float64)[y]
    want = _direct(x, t, s, sign, alpha, stride, pad)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())
