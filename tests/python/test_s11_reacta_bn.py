"""Slice S11 (existing ops): ReActNet-A with BatchNorm -- BURN, S2-BNN and the BiMatting ImageNet backbone through the
BN flavour of torch_abbnn._convert_reactnet_a (converter tools/convert_reacta_bn.py). No new operator.

* Official model classes (BURN reactnet.py with the mmcv registry stubbed, S2-BNN step2 ReActNet.py, BiMatting
  mobilenet.py BiMobileNet + the checkpoint head) with random weights / BN statistics / biases / PReLU slopes at 64x64:
  IR in float64 with float64 constants == the official float64 forward (sign_zero). BiMobileNet exercises the
  4x block, the half block (split-sum as Reshape + AvgPool2D[1,2] + Reshape) and the forward_single_frame macro residuals.
* numpy vs C++ bit-exact (reference sign_zero / ge, neon ge) on the small converted BiMobileNet and BURN networks.
* The split-sum decomposition equals the source's (a + b) / 2 bit for bit (numpy and C++ reference / neon / optimized).
* bn_binary_conv_parts (BiConv2d centred weights, BinaryConv2d zero-weight rejection), unused parameters, ceil_mode check.
The tests that need the official code skip when the clones (.cache/repos/{BiMatting,burn,S2-BNN}) are absent.
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
from bitino.adapters import torch_abbnn  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
REPOS = {"bimatting-bimobilenet-backbone": os.environ.get("BITINO_BIMATTING_REPO", os.path.join(MAIN, ".cache", "repos", "BiMatting")),
         "burn-reacta-stage2": os.environ.get("BITINO_BURN_REPO", os.path.join(MAIN, ".cache", "repos", "burn")),
         "s2bnn-reacta-distill-only": os.environ.get("BITINO_S2BNN_REPO", os.path.join(MAIN, ".cache", "repos", "S2-BNN"))}


def _randomize(model, seed=0):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            k = type(m).__name__
            if k == "BatchNorm2d":
                m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.3)
                m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                m.weight.copy_(torch.randn(m.num_features, generator=g))   # negative scales included
                m.bias.copy_(torch.randn(m.num_features, generator=g) * 0.3)
            elif k in ("Conv2d", "BinaryConv2d", "BiConv2d", "HardBinaryConv"):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.2)
            elif k == "LearnableBias":
                m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.2)
            elif k == "PReLU":
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
            elif k == "Linear":
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.05)
                m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
    return model


def _source(model_id: str):
    pytest.importorskip("torch")
    from convert_reacta_bn import MODELS, load_source
    if not os.path.exists(os.path.join(REPOS[model_id], MODELS[model_id][2][0])):
        pytest.skip(f"{model_id} source clone not available")
    model, info = load_source(model_id, REPOS[model_id], None)
    return _randomize(model), info


@pytest.mark.parametrize("literal", [False, True])
@pytest.mark.parametrize("model_id", sorted(REPOS))
def test_float64_equivalence_with_official_model(model_id, literal):
    """Default (refolded thresholds q + fl64(b_prev + b_next)) and literal fl(fl(q + b_prev) + b_next) forms."""
    torch = pytest.importorskip("torch")
    from convert_reacta_bn import run_reference
    model, _ = _source(model_id)
    g, info = torch_abbnn.convert(model, model_id, binarize="sign_zero", input_hw=(64, 64), literal_thresholds=literal)
    n_tail = len(info["binarization_points"]) - 1 - (3 if model_id == "bimatting-bimobilenet-backbone" else 0)
    assert len(info["refolded_thresholds"]) == (0 if literal else n_tail)   # all but block 1 (+ BiMatting post-macro blocks)
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v.reshape(g.constants[name].data.shape)
    x = np.random.default_rng(1).normal(size=(2, 64, 64, 3)).astype(np.float32)
    ref64, st = run_reference(model, x, torch.float64, info["binarization_points"])
    assert st["binary_activation_ne_torch_sign"] == 0
    for i in range(2):
        out = executor.run(g, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]].ravel()
        assert out.shape == ref64[i].shape
        assert np.abs(out - ref64[i]).max() < 1e-9 * max(1.0, np.abs(ref64).max())
    ops = g.op_counts()
    if model_id == "bimatting-bimobilenet-backbone":
        assert ops["Reshape"] == 10 and ops["Concat"] == 10 and ops["Dense"] == 1     # 5 half blocks, 9x 2x + 1x 4x blocks
        assert any(n.name == "macro12.add" for n in g.nodes) and not any(n.name == "macro9.add" for n in g.nodes)
    elif model_id == "burn-reacta-stage2":
        assert ops["Dense"] == 1 and any(n.op == "Activation" and n.attrs.get("type") == "relu" for n in g.nodes)
    else:
        assert "Dense" not in ops and g.nodes[-1].op == "GlobalAvgPool"


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("reference", "ge"), ("neon", "ge")])
@pytest.mark.parametrize("model_id", ["bimatting-bimobilenet-backbone", "burn-reacta-stage2"])
def test_numpy_bit_exact_with_cpp(tmp_path, model_id, stage, binarize):
    model, _ = _source(model_id)
    g, _ = torch_abbnn.convert(model, model_id, binarize=binarize, input_hw=(64, 64))
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    xs = np.random.default_rng(3).normal(size=(2, 64, 64, 3)).astype(np.float32)
    xs.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(2, -1)
    for i in range(2):
        np.testing.assert_array_equal(got[i], executor.run(g2, {"input": xs[i:i + 1]})[g2.outputs[0]].ravel())


def _split_sum_graph(h, w, p):
    g = Graph()
    x = g.add_input("input", [1, h, w, 2 * p])
    r = g.add_node("Reshape", [x], {"shape": [1, h * w, 2, p]})
    r = g.add_node("AvgPool2D", [r], {"kernel": [1, 2], "strides": [1, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1})
    g.set_outputs(g.add_node("Reshape", [r], {"shape": [1, h, w, p]}))
    return g


@pytest.mark.parametrize("stage", ["numpy", "reference", "neon", "optimized"])
def test_split_sum_is_exact(tmp_path, stage):
    """BiMobileNet half block: sum(torch.split(out1, P, dim=1)) / 2 == Reshape + AvgPool2D[1,2] + Reshape, bit for bit."""
    h, w, p = 7, 5, 24
    rng = np.random.default_rng(5)
    x = (rng.normal(size=(2, h, w, 2 * p)) * np.exp(rng.normal(size=(2, h, w, 2 * p)) * 4)).astype(np.float32)
    want = ((np.float32(0) + x[..., :p]) + x[..., p:]) / np.float32(2)          # Python sum() starts at 0, then / 2
    g = _split_sum_graph(h, w, p)
    if stage == "numpy":
        got = np.stack([executor.run(g, {"input": x[i:i + 1]})[g.outputs[0]][0] for i in range(2)])
    else:
        if not os.path.exists(BENCH):
            pytest.skip("host bitino_bench not built")
        path = tmp_path / "s.bitino"
        bfformat.write(g, str(path))
        x.tofile(tmp_path / "in.f32")
        subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                        "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "2"], check=True, capture_output=True)
        got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(want.shape)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def test_bn_binary_conv_parts_centred_and_plain():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class BiConv2d(nn.Conv2d):   # mobilenet.py:8-26 semantics (eval)
        def forward(self, x):
            w = self.weight - self.weight.mean(3, True).mean(2, True).mean(1, True)
            s = w.abs().mean(3, True).mean(2, True).mean(1, True)
            return nn.functional.conv2d(x, s * torch.sign(w), None, self.stride, self.padding)

    c = BiConv2d(6, 4, 3, padding=1, bias=False)
    with torch.no_grad():
        c.weight.copy_(torch.randn(c.weight.shape) + 0.7)   # off-centre: centring matters
    parts = torch_abbnn.bn_binary_conv_parts(c)
    w64 = c.weight.detach().double()
    w64 = w64 - w64.mean(3, True).mean(2, True).mean(1, True)
    np.testing.assert_array_equal(parts["sign"], torch.sign(w64).numpy().astype(np.int8))
    np.testing.assert_allclose(parts["scale64"], w64.abs().mean((1, 2, 3)).numpy(), rtol=1e-14)
    assert (parts["sign"] != np.sign(c.weight.detach().numpy())).any()

    class BinaryConv2d(nn.Conv2d):
        pass
    b = BinaryConv2d(3, 2, 1, bias=False)
    with torch.no_grad():
        b.weight.zero_()
    with pytest.raises(NotImplementedError, match="exactly-zero"):
        torch_abbnn.bn_binary_conv_parts(b)


def test_unused_parameters_listed():
    model, _ = _source("s2bnn-reacta-distill-only")
    _, info = torch_abbnn.convert(model, "s2", binarize="ge", input_hw=(64, 64))
    u = info["unused_in_forward"]
    # stride-1 same-width blocks keep binary_pw / bn2, the doubling blocks binary_pw_down* / bn2_*; the rest is unused
    assert "feature.3.binary_pw_down1.weight" in u and "feature.3.bn2_1.running_var" in u
    assert "feature.1.binary_pw.weight" in u and "feature.1.bn2.weight" in u
    assert not any(".binary_3x3." in n or ".bn1." in n for n in u)
    model, _ = _source("bimatting-bimobilenet-backbone")
    _, info = torch_abbnn.convert(model, "bm", binarize="ge", input_hw=(64, 64))
    assert len(info["unused_in_forward"]) == 46 and all(n.endswith("channel_threshold") for n in info["unused_in_forward"])


def test_ceil_mode_partial_window_rejected():
    model, _ = _source("bimatting-bimobilenet-backbone")
    with pytest.raises(NotImplementedError, match="ceil_mode"):
        torch_abbnn.convert(model, "bm", binarize="ge", input_hw=(36, 36))   # 18 -> 9 -> odd map at the next pooling
