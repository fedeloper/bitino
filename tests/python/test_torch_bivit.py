"""Bi-ViT / GSB adapter building blocks (no checkpoint needed).

* actbi_threshold / min_true: the float32 thresholds reproduce sign(fl32(fl32(x / a) + r)) exactly, checked on
  every float32 around the threshold and on random values.
* code_product: sum_k alpha_a (s_a - r_a) * alpha_b (s_b - r_b) expanded into exact BMatMul counts; numpy executor
  (float64) vs the direct formula, and (if the host bitino_bench is built) C++ reference vs numpy bit-exact.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

torch = pytest.importorskip("torch")
from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters import torch_bivit as tb  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))


@pytest.mark.parametrize("alpha,r", [(0.002437, -1), (0.004818, 1), (1.3759664, -2), (0.7981307, -1), (0.0123, 2), (5e-5, 1)])
def test_actbi_threshold_exact(alpha, r):
    t = tb.actbi_threshold(alpha, r)   # also verifies the floats adjacent to the thresholds
    a = np.float32(alpha)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-r * alpha, abs(alpha) * 1e-5, 20000), rng.normal(0, 3 * abs(alpha) * abs(r), 20000)]).astype(np.float32)
    s = np.sign(((x / a).astype(np.float32) + np.float32(r)).astype(np.float32))
    exp = np.where(x >= np.float32(t["t_hi"]), 1.0, np.where(x >= np.float32(t["t_lo"]), 0.0, -1.0))
    assert np.array_equal(s, exp)


def _operand(b, name, d, alpha, r, rng_x):
    """Operand like _BiViT.actbi (without the module bookkeeping)."""
    sig = 1 if alpha > 0 else -1
    if r == 0:
        return {"u": name, "sigma": sig, "mode": "sign_zero", "r": 0, "alpha32": alpha, "alpha64": alpha}
    t = tb.actbi_threshold(alpha, r)
    mode, tt = ("sign_zero", t["t_lo"]) if t["n_tie_floats"] == 1 else ("ge", t["t_hi"])
    u = b.affine(name, d, scale=float(sig) if sig < 0 else None, bias=-tt, tag=name + "/u")
    return {"u": u, "sigma": 1, "mode": mode, "r": r, "alpha32": alpha, "alpha64": alpha}


def _codes(x, alpha, r):
    a = np.float32(alpha)
    s = np.sign(((x / a).astype(np.float32) + np.float32(r)).astype(np.float32)).astype(np.float64)
    return (s - r) * float(a)


@pytest.mark.parametrize("ra,rb,aa,ab,tb_", [(0, 0, 1.43, -0.0069, 1), (-1, 0, 0.0024, 6.48, 0), (1, -2, -0.0040, 1.376, 0),
                                            (-1, -1, 0.9, 0.8, 1), (2, 1, -0.3, -0.7, 1)])
def test_code_product_exact(ra, rb, aa, ab, tb_):
    m_, k_, n_ = 7, 33, 5
    rng = np.random.default_rng(1)
    xa = (rng.normal(-ra * aa, abs(aa), (m_, k_))).astype(np.float32)
    xb = (rng.normal(-rb * ab, abs(ab), (n_, k_) if tb_ else (k_, n_))).astype(np.float32)
    xa[0, :4] = np.float32(-ra * aa)            # values on the rounding boundary
    b = tb._BiViT(torch.nn.Module(), "sign_zero")
    g = b.g
    ia = g.add_input("a", list(xa.shape))
    ib = g.add_input("b", list(xb.shape))
    opa = _operand(b, ia, k_, aa, ra, rng)
    opb = _operand(b, ib, xb.shape[-1], ab, rb, rng)
    out = b.code_product(opa, opb, k_, (m_, n_), tb_, 0.125 if tb_ else 1.0, 0.125 if tb_ else 1.0, tag="p")
    g.set_outputs(out)
    ca, cb = _codes(xa, aa, ra), _codes(xb, ab, rb)
    exp = (ca @ (cb.T if tb_ else cb)) * (0.125 if tb_ else 1.0)
    got64 = executor.run(g, {"a": xa, "b": xb}, float_dtype=np.float64)[out]
    scale32 = float(np.float32(np.float32(aa) * np.float32(ab)) * np.float32(0.125 if tb_ else 1.0))
    # integer counts are exact; only the final scale differs (float32-rounded alpha_a*alpha_b)
    ref = exp * scale32 / (float(np.float32(aa)) * float(np.float32(ab)) * (0.125 if tb_ else 1.0))
    assert np.allclose(got64, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
    if os.path.exists(BENCH):
        with tempfile.TemporaryDirectory() as d:
            mp, ip, op = os.path.join(d, "m.bitino"), os.path.join(d, "in.f32"), os.path.join(d, "out.f32")
            bfformat.write(g, mp)
            np.concatenate([xa.ravel(), xb.ravel()]).astype(np.float32).tofile(ip)
            want = executor.run(g, {"a": xa, "b": xb})[out]
            for stage in ("reference", "neon", "optimized"):
                subprocess.run([BENCH, "--model", mp, "--inputs", ip, "--outputs", op, "--stage", stage, "--warmup", "0",
                                "--iterations", "1"], check=True, capture_output=True)
                got = np.fromfile(op, np.float32).reshape(want.shape)
                assert np.array_equal(got.view(np.uint32), want.astype(np.float32).view(np.uint32)), stage
