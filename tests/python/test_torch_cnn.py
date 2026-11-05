"""PyTorch binary-CNN path (BNext, RBONN): Mul op + adapters.

* A synthetic graph with every op on the BNext path (BConv2D, BN ChannelAffine, PReLU,
  info-recoupling mix as ChannelAffine + ChannelAffine + Add, SE gate GlobalAvgPool(keepdims) ->
  Conv2D(relu) -> Conv2D -> ChannelGate(sigmoid), ELM Mul (same shape), AvgPool2D,
  Concat): numpy executor vs C++ runtime, bit-exact (reference: sign_zero and ge; neon: ge).
* If the official clones are available (BITINO_BNEXT_REPO / BITINO_RBONN_REPO, defaults under
  .cache/repos) and torch/timm/einops/torchinfo import, the official BNext-Tiny and RBONN-18 with
  random weights and random BN statistics are converted and compared with the float64 source
  using exact float64 constants (must agree to ~1e-9 relative).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
H = W = 8
C = 16


def bnext_like(seed: int, binarize: str) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, H, W, C])

    def cst(name, arr, dtype="f32", layout=""):
        return g.add_constant(name, np.asarray(arr), dtype, layout)

    def pm1(shape):
        return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)

    def unit(x, tag, k, stride, elm):
        res = g.add_node("Activation", [x, cst(tag + "a1", rng.random(C) * 0.5)], {"type": "prelu"}) if elm else x
        inp = x
        if stride == 2:
            res = g.add_node("AvgPool2D", [res], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1})
            inp = g.add_node("AvgPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1})
        # quantize the bias to 1/8 so that x + bias hits exact zeros (sign_zero vs ge matters)
        t = g.add_node("ChannelAffine", [x, "", cst(tag + "mv", np.round(rng.normal(size=C) * 4) / 8)])
        t = g.add_node("BConv2D", [t, cst(tag + "w", pm1((C, k, k, C)), "bit", "OHWI_bit"), "", "", ""],
                       {"strides": [stride, stride], "dilations": [1, 1], "pads": [k // 2] * 4, "groups": 1, "pad_value": 0.0,
                        "input_binarize": binarize, "activation": "none", "output": "f32"})
        t = g.add_node("ChannelAffine", [t, cst(tag + "s1", rng.random(C) * 0.1), cst(tag + "b1", rng.normal(size=C))])
        t = g.add_node("Activation", [t, cst(tag + "a2", rng.random(C) * 0.5)], {"type": "prelu"})
        sc = rng.random(C)
        mix = g.add_node("Add", [g.add_node("ChannelAffine", [inp, cst(tag + "sc", sc), ""]),
                                 g.add_node("ChannelAffine", [t, cst(tag + "omsc", 1.0 - sc), ""])])
        gt = g.add_node("GlobalAvgPool", [mix], {"keepdims": 1})
        gt = g.add_node("Conv2D", [gt, cst(tag + "se1", rng.normal(size=(C // 4, 1, 1, C)), layout="OHWI"), cst(tag + "se1b", rng.normal(size=C // 4))],
                        {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "activation": "relu"})
        gt = g.add_node("Conv2D", [gt, cst(tag + "se3", rng.normal(size=(C, 1, 1, C // 4)), layout="OHWI"), cst(tag + "se3b", rng.normal(size=C))],
                        {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "activation": "none"})
        t = g.add_node("ChannelGate", [t, gt], {"gate": "sigmoid"})
        if elm:
            t = g.add_node("Mul", [t, res])
        t = g.add_node("ChannelAffine", [t, cst(tag + "s2", rng.random(C)), cst(tag + "b2", rng.normal(size=C))])
        return g.add_node("Add", [t, res])

    x = unit(x, "u0", 3, 1, False)
    x = unit(x, "u1", 3, 2, True)
    y1 = unit(x, "u2", 1, 1, False)
    y2 = unit(x, "u3", 1, 1, False)
    x = g.add_node("Concat", [y1, y2], {"axis": -1})
    logits = g.add_node("Dense", [g.add_node("GlobalAvgPool", [x]), cst("fw", rng.normal(size=(10, 2 * C))), cst("fb", rng.normal(size=10))])
    g.set_outputs(logits, x)
    return g


def test_mul_numpy_contract():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, 3, 3, 5)).astype(np.float32)
    b = rng.normal(size=(1, 3, 3, 5)).astype(np.float32)
    g = Graph()
    g.add_input("a", [1, 3, 3, 5])
    g.add_input("b", [1, 3, 3, 5])
    g.set_outputs(g.add_node("Mul", ["a", "b"]))
    y = executor.run(g, {"a": a, "b": b})[g.outputs[0]]
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y, a * b)


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("reference", "ge"), ("neon", "ge")])
def test_numpy_executor_bit_exact_with_cpp(tmp_path, stage, binarize):
    g = bnext_like(3, binarize)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    rng = np.random.default_rng(5)
    xs = [np.round(rng.normal(size=(1, H, W, C)) * 8 * s) / 8 for s in (0.5, 1.0, 2.0)]
    xs = [v.astype(np.float32) for v in xs]
    np.concatenate(xs).tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(len(xs), -1)
    zeros = 0
    for i, xin in enumerate(xs):
        res = executor.run(g2, {"input": xin}, keep_intermediates=True)
        zeros += sum(int(np.sum(res[n.inputs[0]] == 0)) for n in g2.nodes if n.op == "BConv2D")
        want = np.concatenate([res[g2.outputs[0]].ravel(), res[g2.outputs[1]].ravel()])
        np.testing.assert_array_equal(got[i], want)
    assert zeros > 0, "test inputs should produce exact zeros at binarization"


def _official(family: str):
    torch = pytest.importorskip("torch")
    for mod in ("timm", "einops", "torchinfo"):
        pytest.importorskip(mod)
    from convert_torch_cnn import FAMILIES  # noqa: F401
    import collections
    import collections.abc
    import contextlib
    import importlib
    import io
    import warnings
    sys.dont_write_bytecode = True
    if family == "bnext":
        repo = os.environ.get("BITINO_BNEXT_REPO", os.path.join(MAIN, ".cache", "repos", "BNext"))
        sub = os.path.join(repo, "src")
    else:
        repo = os.environ.get("BITINO_RBONN_REPO", os.path.join(MAIN, ".cache", "repos", "SteveTsui_RBONN"))
        sub = os.path.join(repo, "2_step2_rbonn")
    if not os.path.isdir(sub):
        pytest.skip(f"official {family} clone not available")
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    sys.path.insert(0, sub)
    torch.manual_seed(0)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", FutureWarning)
        if family == "bnext":
            model = importlib.import_module("bnext").BNext(num_classes=1000, size="tiny")
        else:
            model = importlib.import_module("rbonn").rbonn18()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(("move.bias", "move0.bias", "move1.bias", "move2.bias")) or (n.endswith(".bias") and "se." in n):
                p.copy_(torch.randn_like(p) * 0.1)
            elif n.endswith(".scale"):
                p.copy_(torch.rand_like(p))
        for n, b in model.named_buffers():
            if n.endswith("temperature"):
                b.fill_(7e-24)   # trained checkpoints: ~7e-24 (sign regime)
        for m in model.modules():
            if type(m).__name__ == "BatchNorm2d":
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
                m.weight.copy_(torch.rand_like(m.weight) + 0.5)
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)
    model.eval()
    return model, torch


@pytest.mark.parametrize("family", ["bnext", "rbonn"])
def test_official_random_weights_float64_exact(family):
    model, torch = _official(family)
    from bitino.adapters import torch_bnext, torch_rbonn
    adapter = torch_bnext if family == "bnext" else torch_rbonn
    hw = 64
    g, info = adapter.convert(model, family, input_hw=(hw, hw))
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    x = np.random.default_rng(0).normal(size=(1, hw, hw, 3))
    with torch.no_grad():
        ref = model.double()(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy()
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())


def test_adapter_rejects_non_binary_weights():
    model, torch = _official("bnext")
    from bitino.adapters import torch_bnext
    model.feature[1].Attention.binary_conv.temperature.fill_(1.0)   # weights become clamp(W/T, -1, 1)
    with pytest.raises(NotImplementedError, match="temperature"):
        torch_bnext.convert(model, "bnext-tiny", input_hw=(64, 64))
