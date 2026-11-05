"""SA-BNN ResNet-18 (adapter torch_sabnn) and IDa-Det BiRes-18 backbone (adapter torch_rbonn, BiConv branch).

* A synthetic graph with every op on both paths (pre-sign BN ChannelAffine feeding a BConv2D *and* an
  AvgPool2D -> Conv2D -> ChannelAffine downsample, BConv2D with post_mul only and with post_mul + post_add
  (BiConv bias), LearnableBias ChannelAffine, Add, PReLU, a shortcut reused by two adds, MaxPool2D,
  GlobalAvgPool -> Dense): numpy executor vs C++ runtime, bit-exact (reference: sign_zero and ge; neon: ge).
* If the source files are available (SA-BNN: the Drive birealnet.py.txt under
  .cache/checkpoints/sa-bnn-resnet18-imagenet or BITINO_SABNN_DIR; IDa-Det: .cache/repos/IDa-Det or
  BITINO_IDADET_REPO), the source model with random weights and random BN statistics is converted and
  compared with the float64 source using exact float64 constants (must agree to ~1e-9 relative), and the
  adapters' guards (non-shared downsample2, ceil_mode AvgPool2d on an odd feature map) are exercised.
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
SABNN_DIR = os.environ.get("BITINO_SABNN_DIR", os.path.join(MAIN, ".cache", "checkpoints", "sa-bnn-resnet18-imagenet"))
IDADET_REPO = os.environ.get("BITINO_IDADET_REPO", os.path.join(MAIN, ".cache", "repos", "IDa-Det"))
H = W = 8
C = 16


def sabnn_like(seed: int, binarize: str) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, H, W, C])

    def cst(name, arr, dtype="f32", layout=""):
        return g.add_constant(name, np.asarray(arr), dtype, layout)

    def pm1(shape):
        return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)

    def bconv(t, tag, cin, cout, stride, bias):
        return g.add_node("BConv2D", [t, cst(tag + "w", pm1((cout, 3, 3, cin)), "bit", "OHWI_bit"), cst(tag + "pm", rng.random(cout) * 0.1),
                                      cst(tag + "pa", rng.normal(size=cout)) if bias else "", ""],
                          {"strides": [stride, stride], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
                           "input_binarize": binarize, "activation": "none", "output": "f32"})

    def bn(t, tag, c, quant=False):
        # power-of-two scale and 1/8-quantized shift: exact zeros reach the binarization (sign_zero vs ge)
        scale = np.where(rng.random(c) < 0.5, 0.5, -1.0) if quant else rng.random(c) + 0.5
        shift = np.round(rng.normal(size=c) * 4) / 8 if quant else rng.normal(size=c)
        return g.add_node("ChannelAffine", [t, cst(tag + "s", scale), cst(tag + "b", shift)])

    def prelu(t, tag, c):
        return g.add_node("Activation", [t, cst(tag, rng.random(c) * 0.5)], {"type": "prelu"})

    # SA-BNN block with a downsample: a = bn2(x); r = ds(a); h = prelu(bn1(bconv(a)) + r); out = prelu2(bn4(bconv(bn3(h))) + r)
    a = bn(x, "bn2", C, quant=True)
    r = g.add_node("AvgPool2D", [a], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1})
    r = g.add_node("Conv2D", [r, cst("dsw", rng.normal(size=(2 * C, 1, 1, C)), layout="OHWI"), ""],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0, "activation": "none"})
    r = bn(r, "dsbn", 2 * C)
    t = bn(bconv(a, "c1", C, 2 * C, 2, False), "bn1", 2 * C)
    h = prelu(g.add_node("Add", [t, r]), "p1", 2 * C)
    u = bn(bconv(bn(h, "bn3", 2 * C, quant=True), "c2", 2 * C, 2 * C, 1, False), "bn4", 2 * C)
    x = prelu(g.add_node("Add", [u, r]), "p2", 2 * C)
    # IDa-Det block: BiConv with bias -> post_add; LearnableBias moves
    t = g.add_node("ChannelAffine", [x, "", cst("move0", np.round(rng.normal(size=2 * C) * 4) / 8)])
    t = bn(bconv(t, "c3", 2 * C, 2 * C, 1, True), "bn5", 2 * C)
    t = g.add_node("Add", [t, x])
    t = g.add_node("ChannelAffine", [t, "", cst("move1", rng.normal(size=2 * C))])
    t = prelu(t, "p3", 2 * C)
    x = g.add_node("ChannelAffine", [t, "", cst("move2", rng.normal(size=2 * C))])
    logits = g.add_node("Dense", [g.add_node("GlobalAvgPool", [x]), cst("fw", rng.normal(size=(10, 2 * C))), cst("fb", rng.normal(size=10))])
    g.set_outputs(logits, x)
    return g


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("reference", "ge"), ("neon", "ge")])
def test_numpy_executor_bit_exact_with_cpp(tmp_path, stage, binarize):
    g = sabnn_like(7, binarize)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    rng = np.random.default_rng(11)
    xs = [(np.round(rng.normal(size=(1, H, W, C)) * 8 * s) / 8).astype(np.float32) for s in (0.5, 1.0, 2.0)]
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


def _randomize(model, torch):
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".weights") or (n.endswith("conv.weight") and p.dim() == 4):
                p.copy_(torch.randn_like(p) * 0.05)          # latent binary weights of both signs
            elif n.endswith((".scale", "prelu.weight", "prelu2.weight")):
                p.copy_(torch.rand_like(p))
            elif n.endswith(("move0.bias", "move1.bias", "move2.bias", "conv.bias")):
                p.copy_(torch.randn_like(p) * 0.1)
        for m in model.modules():
            if type(m).__name__ == "BatchNorm2d":
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
                m.weight.copy_(torch.randn_like(m.weight))      # both signs (a negative pre-sign BN flips the sign)
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)
    model.eval()


def _source(family: str):
    torch = pytest.importorskip("torch")
    from convert_torch_cnn import load_model
    if family == "sabnn":
        if not os.path.exists(os.path.join(SABNN_DIR, "birealnet.py.txt")):
            pytest.skip("SA-BNN birealnet.py.txt not available")
        model, _, _ = load_model("sabnn", SABNN_DIR, "sa-bnn-resnet18-imagenet", None)
    else:
        if not os.path.exists(os.path.join(IDADET_REPO, "mmdet", "models", "backbones", "resnet_react.py")):
            pytest.skip("IDa-Det clone not available")
        model, _, _ = load_model("idadet", IDADET_REPO, "ida-det-bires18-imagenet-backbone", None)
    _randomize(model, torch)
    return model, torch


@pytest.mark.parametrize("family", ["sabnn", "idadet"])
def test_source_random_weights_float64_exact(family):
    model, torch = _source(family)
    from convert_torch_cnn import convert_graph
    hw = 64
    adapter = convert_graph.__globals__["FAMILIES"][family]["adapter"]
    g, info = adapter.convert(model, family, input_hw=(hw, hw))
    assert g.op_counts()["BConv2D"] == 16
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    x = np.random.default_rng(0).normal(size=(1, hw, hw, 3))
    with torch.no_grad():
        ref = model.double()(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy()
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())
    if family == "idadet":
        assert sum(1 for n in g.nodes if n.op == "BConv2D" and n.inputs[3]) == 16   # BiConv bias -> post_add


def test_sabnn_rejects_unshared_downsample2():
    model, torch = _source("sabnn")
    import copy
    from bitino.adapters import torch_sabnn
    blk = model.layer2[0]
    blk.downsample2 = copy.deepcopy(blk.downsample)
    with pytest.raises(NotImplementedError, match="downsample2"):
        torch_sabnn.convert(model, "sa-bnn", input_hw=(64, 64))


def test_idadet_rejects_ceil_mode_on_odd_feature_map():
    model, torch = _source("idadet")
    from bitino.adapters import torch_rbonn
    # 72x72 input: layer3's downsample sees a 9x9 map, where AvgPool2d(2, 2, ceil_mode=True) keeps a partial window
    with pytest.raises(NotImplementedError, match="ceil_mode"):
        torch_rbonn.convert(model, "ida-det", input_hw=(72, 72))
