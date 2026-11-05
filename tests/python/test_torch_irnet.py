"""IR-Net family (ReCU, RBNN, SiMaN, BiPer; adapter torch_irnet) and SNN ReActNet-18 (torch_rbonn).

* A synthetic graph with every op on the IR-Net path (per-sample centering Reshape -> ReduceMean ->
  ChannelAffine(-1) -> AddBroadcast, BConv2D with alpha post_mul, BN ChannelAffine, Bi-Real Add,
  Hardtanh and PReLU, FP conv1x1 stride-2 downsample, MaxPool2D, GlobalAvgPool -> BatchNorm1d
  ChannelAffine on [1, C] -> Dense): numpy executor vs C++ runtime, bit-exact
  (reference: sign_zero and ge; neon: ge).
* If the official code bases are available under .cache/repos (or BITINO_IRNET_REPOS=<dir>),
  each family's official ResNet-18 with random weights and random BN statistics is converted and
  compared with the float64 source (fp32-derived binary weights substituted at F.conv2d) using
  exact float64 constants (must agree to ~1e-9 relative). Same for the CIFAR-10 ResNet-18/20 and
  VGG-Small variants, the third-party Bi-Real-20, and the SNN W1A1 / 0.67-bit ReActNet-18 through
  torch_rbonn.
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
REPOS = os.environ.get("BITINO_IRNET_REPOS", os.path.join(MAIN, ".cache", "repos"))
H = W = 8
C = 16


def irnet_like(seed: int, binarize: str) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, H, W, C])

    def cst(name, arr, dtype="f32", layout=""):
        return g.add_constant(name, np.asarray(arr), dtype, layout)

    def pm1(shape):
        return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)

    def bconv(t, tag, cin, cout, stride, hw, center):
        if center:
            r = g.add_node("Reshape", [t], {"shape": [1, hw * hw * cin]})
            mu = g.add_node("ReduceMean", [r], {"axis": -1, "keepdims": 1})
            neg = g.add_node("ChannelAffine", [mu, cst(tag + "m1", [-1.0]), ""])
            t = g.add_node("AddBroadcast", [t, neg])
        y = g.add_node("BConv2D", [t, cst(tag + "w", pm1((cout, 3, 3, cin)), "bit", "OHWI_bit"), cst(tag + "alpha", rng.random(cout) * 0.1 + 0.01), "", ""],
                       {"strides": [stride, stride], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
                        "input_binarize": binarize, "activation": "none", "output": "f32"})
        return g.add_node("ChannelAffine", [y, cst(tag + "s", rng.random(cout) + 0.5), cst(tag + "b", rng.normal(size=cout) * 0.25)])

    def act(t, tag, kind, c):
        if kind == "hardtanh":
            return g.add_node("Activation", [t], {"type": "hardtanh", "min": -1.0, "max": 1.0})
        return g.add_node("Activation", [t, cst(tag + "slope", rng.random(c) * 0.5)], {"type": "prelu"})

    def block(t, tag, cin, cout, stride, hw, center, kind):
        u = bconv(t, tag + "c1", cin, cout, stride, hw, center)
        res = t
        if stride != 1 or cin != cout:
            res = g.add_node("Conv2D", [t, cst(tag + "dw", rng.normal(size=(cout, 1, 1, cin)) * 0.3, layout="OHWI"), ""],
                             {"strides": [stride, stride], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "activation": "none"})
            res = g.add_node("ChannelAffine", [res, cst(tag + "ds", rng.random(cout) + 0.5), cst(tag + "db", rng.normal(size=cout) * 0.1)])
        u = act(g.add_node("Add", [u, res]), tag + "a1", kind, cout)
        v = bconv(u, tag + "c2", cout, cout, 1, hw // stride, center)
        return act(g.add_node("Add", [v, u]), tag + "a2", kind, cout)

    x = block(x, "b0", C, C, 1, H, False, "hardtanh")       # ReCU / BiPer style
    x = block(x, "b1", C, 2 * C, 2, H, True, "prelu")       # SiMaN style (centered, PReLU)
    x = block(x, "b2", 2 * C, 2 * C, 1, H // 2, True, "hardtanh")   # RBNN style
    x = g.add_node("MaxPool2D", [x], {"kernel": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]})
    p = g.add_node("GlobalAvgPool", [x], {"keepdims": 0})
    p = g.add_node("ChannelAffine", [p, cst("bn1d_s", rng.random(2 * C) + 0.5), cst("bn1d_b", rng.normal(size=2 * C))])
    logits = g.add_node("Dense", [p, cst("fw", rng.normal(size=(10, 2 * C))), cst("fb", rng.normal(size=10))])
    g.set_outputs(logits, x)
    return g


def test_centering_numpy_contract():
    """Reshape -> ReduceMean -> ChannelAffine(-1) -> AddBroadcast == x - mean_HWC(x) (mean in double, rounded once)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 4, 5, 6)).astype(np.float32)
    g = Graph()
    g.add_input("input", [1, 4, 5, 6])
    r = g.add_node("Reshape", ["input"], {"shape": [1, 120]})
    mu = g.add_node("ReduceMean", [r], {"axis": -1, "keepdims": 1})
    neg = g.add_node("ChannelAffine", [mu, g.add_constant("m1", np.array([-1.0], np.float32)), ""])
    g.set_outputs(g.add_node("AddBroadcast", ["input", neg]))
    y = executor.run(g, {"input": x})[g.outputs[0]]
    mean = np.float32(sum(float(v) for v in x.reshape(-1)) / 120.0)
    np.testing.assert_array_equal(y, x - mean)


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("reference", "ge"), ("neon", "ge")])
def test_numpy_executor_bit_exact_with_cpp(tmp_path, stage, binarize):
    g = irnet_like(7, binarize)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    rng = np.random.default_rng(11)
    xs = [(np.round(rng.normal(size=(1, H, W, C)) * 8 * s) / 8).astype(np.float32) for s in (0.25, 1.0, 2.0)]
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


IRNET_REPOS = {"recu": "yuanchunyu_ReCU", "rbnn": "RBNN", "siman": "lmbxmu_SiMaN", "biper": "edmav4_BiPer"}


def _randomize(model, torch):
    with torch.no_grad():
        for m in model.modules():
            cls = type(m).__name__
            if cls in ("BatchNorm2d", "BatchNorm1d"):
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
                m.weight.copy_(torch.rand_like(m.weight) + 0.5)
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)
            elif cls == "BinarizeConv2d":
                m.alpha.copy_(torch.rand_like(m.alpha) * 0.5 + 0.05)
            elif cls == "PReLU":
                m.weight.copy_(torch.rand_like(m.weight) * 0.5)
            elif cls == "LearnableBias":
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)


def _fp64_with_fp32_weights(model, x, conv_cls, irnet, torch):
    from convert_torch_cnn import _install_proxy
    cap32, cap64 = {}, {}
    for dt, cap in ((torch.float32, cap32), (torch.float64, cap64)):
        _, remove = _install_proxy(model.to(dt), conv_cls, capture=cap)
        with torch.no_grad():
            model(torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(dt))
        remove()
    override = {k: (v.double() if irnet else torch.sign(v).double() * cap64[k].abs()) for k, v in cap32.items()}
    _, remove = _install_proxy(model.to(torch.float64), conv_cls, override=override)
    with torch.no_grad():
        ref = model(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy()
    remove()
    model.to(torch.float32)
    return ref


@pytest.mark.parametrize("family", sorted(IRNET_REPOS))
def test_official_irnet_random_weights_float64_exact(family):
    torch = pytest.importorskip("torch")
    repo = os.path.join(REPOS, IRNET_REPOS[family])
    if not os.path.isdir(os.path.join(repo, "imagenet")):
        pytest.skip(f"official {family} code not available")
    from convert_torch_cnn import _load_irnet
    from bitino.adapters import torch_irnet
    torch.manual_seed(0)
    model, _, _ = _load_irnet(family, repo, "resnet18_1w1a", None)
    _randomize(model, torch)
    x = np.random.default_rng(0).normal(size=(1, 224, 224, 3))
    g, info = torch_irnet.convert(model, family, binarize="sign_zero")
    assert info["op_counts"]["BConv2D"] == 16
    if torch_irnet.FAMILIES[family]["center"]:
        assert info["op_counts"]["AddBroadcast"] == 16 and info["op_counts"]["ReduceMean"] == 16
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    ref = _fp64_with_fp32_weights(model, x, "BinarizeConv2d", True, torch)
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())


def test_irnet_adapter_rejects_stage1_biper():
    torch = pytest.importorskip("torch")
    repo = os.path.join(REPOS, IRNET_REPOS["biper"])
    if not os.path.isdir(os.path.join(repo, "imagenet")):
        pytest.skip("official BiPer code not available")
    from convert_torch_cnn import _load_irnet
    from bitino.adapters import torch_irnet
    model, _, _ = _load_irnet("biper", repo, "resnet18_1w1a", None)
    model.layer1[0].conv1.stage2 = False   # stage 1: weights sin(freq*w), not binary
    with pytest.raises(NotImplementedError, match="stage2"):
        torch_irnet.convert(model, "biper")


@pytest.mark.parametrize("arch", ["w1a1", "w0.67a1"])
def test_official_snn_random_weights_float64_exact(arch):
    torch = pytest.importorskip("torch")
    repo = os.path.join(REPOS, "yikaiw_SNN")
    if not os.path.isdir(os.path.join(repo, "step2")):
        pytest.skip("yikaiw/SNN code not available")
    import importlib
    from convert_torch_cnn import _purge
    from bitino.adapters import torch_rbonn
    _purge(("birealnet", "modules"))
    sub = os.path.join(repo, "step1" if arch == "w1a1" else "step2")
    sys.path.insert(0, sub)
    sys.dont_write_bytecode = True
    try:
        mod = importlib.import_module("birealnet")
        torch.manual_seed(0)
        import random
        random.seed(0)
        if arch == "w1a1":
            model = mod.birealnet18()
            for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
                for blk in layer:
                    c = blk.binary_conv
                    blk.binary_conv = mod.HardBinaryConv(c.in_channels, c.out_channels, stride=c.stride[0])
        else:
            model = mod.birealnet18(bit_num=6)
    finally:
        sys.path.remove(sub)
    with torch.no_grad():
        for m in model.modules():
            if type(m).__name__ == "HardBinaryConv":
                m.weight.copy_(torch.randn_like(m.weight) * 0.5)
    _randomize(model, torch)
    model.eval()
    hw = 64
    x = np.random.default_rng(0).normal(size=(1, hw, hw, 3))
    g, info = torch_rbonn.convert(model, "snn", input_hw=(hw, hw))
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    ref = _fp64_with_fp32_weights(model, x, "HardBinaryConv", False, torch)
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())


CIFAR_CASES = [("recu_cifar", "resnet20_1w1a"), ("rbnn_cifar", "resnet18_1w1a"), ("rbnn_cifar", "resnet20_bireal_1w1a"),
               ("siman_cifar", "resnet20_1w1a"), ("siman_cifar", "vgg_small_1w1a"), ("biper_cifar", "resnet18_1w1a"),
               ("biper_cifar", "resnet20_1w1a"), ("recu_cifar", "vgg_small_1w1a")]


@pytest.mark.parametrize("family,arch", CIFAR_CASES)
def test_official_irnet_cifar_random_weights_float64_exact(family, arch):
    """CIFAR ResNet-18/20 (option-A / AvgPool+pad / binary 1x1 shortcuts) and VGG-Small, random weights."""
    torch = pytest.importorskip("torch")
    from convert_torch_cnn import FAMILIES, _load_irnet
    from bitino.adapters import torch_irnet
    fam = FAMILIES[family]
    repo = os.path.join(REPOS, IRNET_REPOS[fam["irnet"]])
    if not os.path.isdir(os.path.join(repo, fam["subdir"])):
        pytest.skip(f"official {family} code not available")
    torch.manual_seed(1)
    model, _, _ = _load_irnet(family, repo, arch, None)
    _randomize(model, torch)
    x = np.random.default_rng(1).normal(size=(1, 32, 32, 3))
    g, info = torch_irnet.convert(model, fam["irnet"], binarize="sign_zero", input_hw=(32, 32))
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    ref = _fp64_with_fp32_weights(model, x, "BinarizeConv2d", True, torch)
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())


def test_official_bireal20_random_weights_float64_exact():
    torch = pytest.importorskip("torch")
    repo = os.path.join(REPOS, "ShuaiZ1037_bnn-xnor-bireal")
    if not os.path.isdir(os.path.join(repo, "binary_model")):
        pytest.skip("ShuaiZ1037/bnn-xnor-bireal code not available")
    from convert_torch_cnn import _load_irnet
    from bitino.adapters import torch_irnet
    torch.manual_seed(2)
    model, _, _ = _load_irnet("shuaiz", repo, "birealnet20", None)
    _randomize(model, torch)
    with torch.no_grad():
        for m in model.modules():
            if type(m).__name__ == "BiRealConv":
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)
    x = np.random.default_rng(2).normal(size=(1, 32, 32, 3))
    g, info = torch_irnet.convert(model, "shuaiz", binarize="sign_zero", input_hw=(32, 32))
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    ref = _fp64_with_fp32_weights(model, x, "BiRealConv", False, torch)
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())
