"""Brevitas BNN-PYNQ adapter (CNV / FC): FakeQuant numpy vs C++, integer-threshold BN folding,
float64 semantic equivalence with the official Brevitas classes (random weights, random BN stats).

The adapter tests need brevitas (pinned venv .cache/venv-brevitas, envs/
brevitas-requirements.txt) and are skipped elsewhere:
  .cache/venv-brevitas/bin/python -m pytest tests/python/test_brevitas_bnn.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters import brevitas_bnn  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))


# ----------------------------------------------------------------------------- FakeQuant (Q1.7)
# The generic FakeQuant (docs/operators_bhvit.md) with scale 1/128, zero point 0, [-128, 127]
# is Brevitas CNV's input quantizer.
Q17 = {"axis": -1, "pre_scale": 1.0, "qmin": -128, "qmax": 127}


def _q17(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    return executor._fake_quant(x, np.full(n, 1 / 128, np.float32), np.zeros(n, np.float32), Q17, np.float32)


def _fq_graph(n: int) -> Graph:
    g = Graph()
    x = g.add_input("input", [1, n])
    g.set_outputs(g.add_node("FakeQuant", [x, g.add_constant("s", np.full(n, 1 / 128, np.float32)),
                                           g.add_constant("z", np.zeros(n, np.float32))], Q17))
    return g


def test_fake_quant_q17_numpy_semantics():
    s = 1.0 / 128
    x = np.array([0.5 * s, 1.5 * s, 2.5 * s, -0.5 * s, -2.5 * s, 3.0, -3.0, 127.5 * s, -0.0], np.float32)
    y = _q17(x)
    np.testing.assert_array_equal(y, np.array([0, 2 * s, 2 * s, 0, -2 * s, 127 * s, -1, 127 * s, 0], np.float32))
    assert y.dtype == np.float32
    # all 256 ToTensor values through 2x-1: exactly clamp(round(128*(2k-255)/255), -128, 127)/128
    k = np.arange(256)
    xin = (np.float32(2) * (k.astype(np.float32) / np.float32(255)) - np.float32(1)).astype(np.float32)
    np.testing.assert_array_equal(_q17(xin) * 128, np.clip(np.rint(128 * (2 * k - 255) / 255.0), -128, 127))


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
def test_fake_quant_q17_numpy_bit_exact_with_cpp(tmp_path):
    rng = np.random.default_rng(0)
    n, ns = 300, 4
    x = rng.uniform(-2.5, 2.5, size=(ns, n)).astype(np.float32)
    x[0, :64] = (np.arange(64, dtype=np.float32) - 32 + 0.5) / np.float32(128)  # ties
    g = _fq_graph(n)
    path = str(tmp_path / "fq.bitino")
    bfformat.write(g, path)
    x.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", path, "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--stage", "all", "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(ns, n)
    want = np.stack([executor.run(g, {"input": x[i:i + 1]})[g.outputs[0]].reshape(-1) for i in range(ns)])
    assert got.tobytes() == want.tobytes()


# ----------------------------------------------------------------------------- integer thresholds
@pytest.mark.parametrize("seed", range(5))
def test_integer_threshold_equals_exact_bn_sign_for_every_accumulator(seed):
    rng = np.random.default_rng(seed)
    k, c = 63, 200
    g = rng.normal(size=c) * np.where(rng.random(c) < 0.3, -1, 1)
    mean, var, beta, eps = rng.normal(size=c) * 10, rng.random(c) * 50 + 0.1, rng.normal(size=c), 1e-4
    sd = np.sqrt(var + eps)
    bnp = {"scale": g / sd, "shift": beta - mean * g / sd, "tau": mean - beta * sd / g, "sign": np.sign(g), "eps": eps}
    # put some thresholds exactly on attainable integers and some outside the range
    bnp["tau"][:5] = np.array([1.0, -3.0, 5.0, k + 10.0, -k - 10.0])
    mul, add, info = brevitas_bnn.integer_threshold(bnp, k, parity=k % 2)
    acc = np.arange(-k, k + 1, 2, dtype=np.float64)[:, None]          # attainable +-1 dot products
    exact = (acc - bnp["tau"]) * np.sign(g) >= 0                        # BN(acc) >= 0 in exact arithmetic
    fused32 = (acc.astype(np.float32) * mul + add) >= 0
    fused64 = (acc * mul.astype(np.float64) + add.astype(np.float64)) >= 0
    np.testing.assert_array_equal(fused32, exact)
    np.testing.assert_array_equal(fused64, exact)
    assert info["n_channels_tau_integer"] >= 1 and info["n_channels_constant"] >= 2


# ----------------------------------------------------------------------------- adapter vs Brevitas
def _need_brevitas():
    return pytest.importorskip("brevitas_examples.bnn_pynq.models")


def _randomize(model, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                c = m.num_features
                m.weight.copy_(torch.randn(c, generator=gen) * torch.where(torch.rand(c, generator=gen) < 0.25, -1.0, 1.0))
                m.bias.copy_(torch.randn(c, generator=gen))
                m.running_mean.copy_(torch.randn(c, generator=gen) * 5)
                m.running_var.copy_(torch.rand(c, generator=gen) * 30 + 0.5)
            if type(m).__name__ == "TensorNorm":
                m.weight.fill_(0.7)
                m.bias.fill_(-0.3)
                m.running_mean.fill_(2.0)
                m.running_var.fill_(40.0)
    model.eval()
    return model


def _ref(model, x_nhwc, dtype):
    import torch

    m = model.to(dtype)
    with torch.no_grad():
        y = m(torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))).to(dtype))
    model.to(torch.float32)
    return getattr(y, "value", y).numpy()


def _check_model(model, x, tol=1e-6):
    import torch

    ref64 = _ref(model, x, torch.float64)
    outs = {}
    for fused in (False, True):
        g, info = brevitas_bnn.convert(model, fused=fused)
        o64 = np.stack([executor.run(g, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]].reshape(-1) for i in range(len(x))])
        np.testing.assert_allclose(o64, ref64, rtol=0, atol=tol)
        outs[fused] = np.stack([executor.run(g, {"input": x[i:i + 1]})[g.outputs[0]].reshape(-1) for i in range(len(x))])
        for m in info["threshold_margins"]:
            assert m["min_margin_acc_units"] > 1e-9
    np.testing.assert_array_equal(outs[False], outs[True])
    return info


def test_fc_random_weights_float64_semantics():
    bm = _need_brevitas()
    import torch

    torch.manual_seed(0)
    model = _randomize(bm.FC.FC(num_classes=10, weight_bit_width=1, act_bit_width=1, in_bit_width=1, in_channels=1,
                                out_features=[96, 33]), 1)
    x = (np.random.default_rng(1).integers(0, 256, size=(6, 28, 28, 1)).astype(np.float32) / np.float32(255))
    x[0, 0, :8, 0] = 0.5   # exact tie: 2x-1 == 0 -> +1
    info = _check_model(model, x)
    assert info["op_counts"] == {"Reshape": 1, "BDense": 3}
    assert info["float_compute_layers"] == []


def test_cnv_random_weights_float64_semantics():
    bm = _need_brevitas()
    import torch

    torch.manual_seed(0)
    model = _randomize(bm.CNV.CNV(num_classes=10, weight_bit_width=1, act_bit_width=1, in_bit_width=8, in_ch=3), 2)
    x = np.random.default_rng(2).integers(0, 256, size=(3, 32, 32, 3)).astype(np.float32) / np.float32(255)
    info = _check_model(model, x)
    assert info["op_counts"]["BConv2D"] == 5 and info["op_counts"]["BDense"] == 3
    assert info["float_compute_layers"] == ["conv_features.1"]


def test_unknown_module_and_unsupported_quantizer_raise():
    bm = _need_brevitas()
    import torch

    model = bm.FC.FC(num_classes=10, weight_bit_width=1, act_bit_width=1, in_bit_width=1, in_channels=1, out_features=[16]).eval()
    model.features.insert(3, torch.nn.ReLU())
    with pytest.raises(NotImplementedError, match="unsupported type ReLU"):
        brevitas_bnn.convert(model)
    model2 = bm.FC.FC(num_classes=10, weight_bit_width=2, act_bit_width=1, in_bit_width=1, in_channels=1, out_features=[16]).eval()
    with pytest.raises(NotImplementedError, match="BinaryQuant"):
        brevitas_bnn.convert(model2)
    model3 = bm.CNV.CNV(num_classes=10, weight_bit_width=1, act_bit_width=2, in_bit_width=8, in_ch=3).eval()
    with pytest.raises(NotImplementedError, match="ClampedBinaryQuant"):
        brevitas_bnn.convert(model3)
