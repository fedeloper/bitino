"""AB-BNN PyTorch adapter tests (python/bitino/adapters/torch_abbnn.py).

The synthetic modules below re-implement the documented forward semantics of the official
AB-BNN classes (Network/BNN.py, Network/ReActNet_18_Qaw.py at commit 3485ea74) under the
same class names, so the adapter can be tested without the source repository. If a clone
is available (BITINO_ABBNN_REPO, default .cache/repos/AB-BNN), the official ResNet18 is
also converted with random weights at CIFAR resolution and compared in float64.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor  # noqa: E402
from bitino.adapters import torch_abbnn  # noqa: E402


# ----------------------------------------------------------------- synthetic official-like modules
class ScaledStdConv2d(nn.Conv2d):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.gain = nn.Parameter(torch.ones(cout, 1, 1, 1))
        self.scale = self.weight[0].numel() ** -0.5
        self.eps = 1e-5
        self.use_layernorm = False

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        return self.gain * (self.scale * (self.weight - mean) / (std + self.eps))

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), None, self.stride, self.padding)


class HardBinaryScaledStdConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, padding=1):
        super().__init__()
        self.stride, self.padding, self.shape = stride, padding, (cout, cin, k, k)
        self.weight = nn.Parameter(torch.randn(self.shape))
        self.gain = nn.Parameter(torch.ones(cout, 1, 1, 1))
        self.scale = self.weight[0].numel() ** -0.5
        self.eps = 1e-5
        self.use_layernorm = False

    def forward(self, x):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = self.scale * (self.weight - mean) / (std + self.eps)
        sf = w.abs().mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
        return F.conv2d(x, self.gain * sf * torch.sign(w), stride=self.stride, padding=self.padding)


class LearnableBias(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x, beta=1):
        return x + self.bias.expand_as(x) / beta


class BinaryActivation(nn.Module):
    def forward(self, x):
        return torch.sign(x)


class Q_PReLU(nn.Module):  # noqa: N801  (official class name)
    def __init__(self, c):
        super().__init__()
        self.L_alpha = nn.Parameter(-3 * torch.ones(c))

    def forward(self, x):
        return F.prelu(x, torch.pow(torch.tensor(2), torch.round(self.L_alpha)).to(x.dtype))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, alpha, beta, stride=1, downsample=None):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.move0, self.act1 = LearnableBias(cin), BinaryActivation()
        self.conv1 = HardBinaryScaledStdConv2d(cin, cout, 3, stride, 1)
        self.move1, self.prelu, self.move2 = LearnableBias(cout), Q_PReLU(cout), LearnableBias(cout)
        self.downsample, self.stride = downsample, stride

    def forward(self, x):
        out = self.conv1(self.act1(self.move0(x, self.beta) * self.beta))
        residual = self.downsample(x) if self.downsample is not None else x
        return self.move2(self.prelu(self.move1(out * self.alpha + residual, 1)), 1)


class ResNet(nn.Module):
    """Two stages (8 -> 16 channels) on a 12x12 input, maxpool stem, one downsample block."""

    def __init__(self, n_cls=10):
        super().__init__()
        self.conv1 = ScaledStdConv2d(3, 8, 3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(8, 8, 0.25, 1.0), BasicBlock(8, 8, 0.25, 1 / 1.0625 ** 0.5))
        ds = nn.Sequential(nn.AvgPool2d(2, 2), HardBinaryScaledStdConv2d(8, 16, 1, 1, 0))
        self.layer2 = nn.Sequential(BasicBlock(8, 16, 0.25, 1 / 1.125 ** 0.5, 2, ds))
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, n_cls)

    def forward(self, x):
        x = self.layer2(self.layer1(self.maxpool(self.conv1(x))))
        return self.fc(self.avgpool(x).flatten(1))


def _randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("L_alpha"):
                p.copy_(torch.randint(-6, 3, p.shape, generator=g).double() + 0.3 * (torch.rand(p.shape, generator=g) - 0.5))
            elif n.endswith("gain"):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            else:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
    return model


def _exact_constants(g, info):
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    return g


def _compare(model, x_nhwc, binarize="sign_zero"):
    """Converts `model` and returns max |logit difference| of the IR (exact float64 constants) vs torch float64."""
    g, info = torch_abbnn.convert(model, "test", input_hw=x_nhwc.shape[1:3], binarize=binarize)
    _exact_constants(g, info)
    m64 = model.to(torch.float64)
    with torch.no_grad():
        ref = m64(torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2)))).numpy()
    model.to(torch.float32)
    out = executor.run(g, {"input": x_nhwc}, float_dtype=np.float64)[g.outputs[0]]
    return float(np.max(np.abs(out - ref))), g


# ----------------------------------------------------------------- tests
def test_scaled_std_conv_folding_matches_get_weight():
    torch.manual_seed(1)
    conv = ScaledStdConv2d(3, 5, 3).double()
    with torch.no_grad():
        conv.gain.copy_(torch.rand(5, 1, 1, 1, dtype=torch.float64) + 0.5)
    w = torch_abbnn.fold_scaled_std_conv(conv)
    assert np.allclose(w, conv.get_weight().detach().numpy(), rtol=0, atol=1e-15)


def test_qprelu_slope_is_exact_power_of_two_half_to_even():
    q = Q_PReLU(6)
    with torch.no_grad():
        q.L_alpha.copy_(torch.tensor([-2.5, -1.5, 0.5, 1.5, -19.2, 6.4]))
    s = torch_abbnn.qprelu_slope(q)
    assert s.dtype == np.float32
    assert np.array_equal(s, np.ldexp(np.float32(1), np.array([-2, -2, 0, 2, -19, 6])))


def test_binary_conv_parts_scale_and_sign():
    torch.manual_seed(2)
    c = HardBinaryScaledStdConv2d(4, 3, 3)
    with torch.no_grad():
        c.gain.copy_(torch.tensor([0.5, -1.0, 2.0]).reshape(3, 1, 1, 1))
    parts = torch_abbnn.binary_conv_parts(c)
    w = c.weight.detach().double()
    std, mean = torch.std_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    ws = c.scale * (w - mean) / (std + c.eps)
    assert np.array_equal(parts["sign"], torch.sign(ws).numpy().astype(np.int8))
    assert np.allclose(parts["scale64"], (c.gain.detach().double().reshape(-1) * ws.abs().mean(dim=(1, 2, 3))).numpy(), atol=1e-15)


def test_binary_conv_with_zero_standardized_weight_raises():
    c = HardBinaryScaledStdConv2d(1, 1, 1, 1, 0)
    with torch.no_grad():
        c.weight.copy_(torch.zeros(1, 1, 1, 1))     # W - mean == 0 -> torch.sign gives 0
    with pytest.raises(NotImplementedError):
        torch_abbnn.binary_conv_parts(c)


@pytest.mark.parametrize("binarize", ["sign_zero", "ge"])
def test_synthetic_model_float64_semantics(binarize):
    model = _randomize(ResNet()).eval()
    x = np.random.default_rng(0).standard_normal((1, 12, 12, 3))
    err, g = _compare(model, x, binarize)
    assert err < 1e-10, err
    ops = g.op_counts()
    assert ops["BConv2D"] == 3 and ops["Conv2D"] == 2 and ops["Activation"] == 3 and ops["AvgPool2D"] == 1


def test_sign_zero_vs_ge_differ_only_on_exact_zero():
    model = _randomize(ResNet()).eval()
    x = np.random.default_rng(1).standard_normal((1, 12, 12, 3))
    gz, _ = torch_abbnn.convert(model, "t", input_hw=(12, 12), binarize="sign_zero")
    gg, _ = torch_abbnn.convert(model, "t", input_hw=(12, 12), binarize="ge")
    assert np.array_equal(executor.run(gz, {"input": x})[gz.outputs[0]], executor.run(gg, {"input": x})[gg.outputs[0]])
    # Force exact zeros at the first binarization input: zero stem channel 0 and layer1.0.move0 bias.
    for g in (gz, gg):
        g.constants["conv1/kernel"].data = g.constants["conv1/kernel"].data.copy()
        g.constants["conv1/kernel"].data[0] = 0
        g.constants["layer1.0.move0/bias"].data = np.zeros(8, np.float32)
    oz = executor.run(gz, {"input": x}, keep_intermediates=True)
    og = executor.run(gg, {"input": x}, keep_intermediates=True)
    assert np.sum(oz["layer1.0.move0:out"] == 0) == 36   # 6x6 positions of channel 0
    assert not np.array_equal(oz[gz.outputs[0]], og[gg.outputs[0]])


def test_unknown_module_raises():
    model = _randomize(ResNet()).eval()
    model.layer1[0].prelu = nn.PReLU(8)
    with pytest.raises(NotImplementedError):
        torch_abbnn.convert(model, "t", input_hw=(12, 12))


MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
REPO = os.environ.get("BITINO_ABBNN_REPO", os.path.join(MAIN, ".cache", "repos", "AB-BNN"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "Network", "ReActNet_18_Qaw.py")), reason="AB-BNN clone not available")
def test_official_resnet18_random_weights_cifar_float64():
    import importlib

    sys.dont_write_bytecode = True
    sys.path.insert(0, REPO)
    mod = importlib.import_module("Network.ReActNet_18_Qaw")
    torch.manual_seed(0)
    model = _randomize(mod.ResNet18(num_classes=10, imagenet=False), seed=3).eval()
    x = np.random.default_rng(2).standard_normal((1, 32, 32, 3))
    err, g = _compare(model, x)
    assert err < 1e-9, err
    assert g.op_counts()["BConv2D"] == 16


# ----------------------------------------------------------------- ReActNet-A (Network/ReActNet_A_Qaw.py)
class firstconv3x3(nn.Module):  # noqa: N801  (official class name)
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = ScaledStdConv2d(cin, cout, 3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv1(x)


class _ABlockA(nn.Module):
    """ReActNet-A BasicBlock: 3x3 binary (+ AvgPool shortcut if stride 2), then 1x1 binary (two + concat if
    planes = 2 * inplanes)."""

    def __init__(self, cin, cout, alpha, beta1, beta2, stride=1):
        super().__init__()
        self.alpha, self.beta1, self.beta2, self.stride, self.inplanes, self.planes = alpha, beta1, beta2, stride, cin, cout
        self.move11, self.binary_3x3 = LearnableBias(cin), HardBinaryScaledStdConv2d(cin, cin, 3, stride, 1)
        self.move12, self.prelu1, self.move13, self.move21 = LearnableBias(cin), Q_PReLU(cin), LearnableBias(cin), LearnableBias(cin)
        if cin == cout:
            self.binary_pw = HardBinaryScaledStdConv2d(cin, cout, 1, 1, 0)
        else:
            self.binary_pw_down1 = HardBinaryScaledStdConv2d(cin, cin, 1, 1, 0)
            self.binary_pw_down2 = HardBinaryScaledStdConv2d(cin, cin, 1, 1, 0)
        self.move22, self.prelu2, self.move23 = LearnableBias(cout), Q_PReLU(cout), LearnableBias(cout)
        self.binary_activation = BinaryActivation()
        if cin != cout:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x_in):
        out1 = self.binary_3x3(self.binary_activation(self.move11(x_in, self.beta1) * self.beta1))
        x = self.pooling(x_in) if self.stride == 2 else x_in
        out1 = self.move13(self.prelu1(self.move12(x + out1 * self.alpha)))
        out2 = self.binary_activation(self.move21(out1, self.beta2) * self.beta2)
        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2) * self.alpha + out1
        else:
            out2 = torch.cat([self.binary_pw_down1(out2) * self.alpha + out1, self.binary_pw_down2(out2) * self.alpha + out1], dim=1)
        return self.move23(self.prelu2(self.move22(out2)))


_ABlockA.__name__ = "BasicBlock"


class Reactnet(nn.Module):
    """firstconv (4 ch, stride 1) -> 4 -> 8 (stride 1, concat) -> 8 -> 16 (stride 2, concat) -> head."""

    def __init__(self, n_cls=6):
        super().__init__()
        self.feature = nn.ModuleList([firstconv3x3(3, 4, 1), _ABlockA(4, 8, 0.25, 1.0, 1 / 1.0625 ** 0.5, 1),
                                      _ABlockA(8, 8, 0.25, 1 / 1.125 ** 0.5, 1 / 1.1875 ** 0.5, 1),
                                      _ABlockA(8, 16, 0.25, 1 / 1.25 ** 0.5, 1 / 1.3125 ** 0.5, 2)])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, n_cls)

    def forward(self, x):
        for blk in self.feature:
            x = blk(x)
        return self.fc(self.pool1(x).flatten(1))


@pytest.mark.parametrize("binarize", ["sign_zero", "ge"])
def test_synthetic_reactnet_a_float64_semantics(binarize):
    model = _randomize(Reactnet(), seed=5).eval()
    err, g = _compare(model, np.random.default_rng(5).standard_normal((1, 8, 8, 3)), binarize)
    assert err < 1e-10, err
    ops = g.op_counts()
    assert ops["BConv2D"] == 8 and ops["Concat"] == 2 and ops["AvgPool2D"] == 1 and ops["Activation"] == 6


def test_fp32_only_zero_standardized_weight_is_reported_not_raised():
    """W - mean that is 0 only in float32 (cancellation): the IR follows the float64 sign, zeros_f32 is reported."""
    c = HardBinaryScaledStdConv2d(1, 1, 1, 1, 0)
    with torch.no_grad():   # mean of [1, 1+2^-23, 1, 1] rounds to 1 in fp32, so W0 - mean is exactly 0 there
        c.weight = nn.Parameter(torch.tensor([1.0, 1.0 + 2.0 ** -23, 1.0, 1.0]).reshape(1, 4, 1, 1))
    parts = torch_abbnn.binary_conv_parts(c)
    assert parts["zeros_f64"] == 0 and parts["zeros_f32"] >= 1
    assert set(np.unique(parts["sign"])) <= {-1, 1}


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "Network", "ReActNet_A_Qaw.py")), reason="AB-BNN clone not available")
def test_official_reactnet_a_random_weights_cifar_float64():
    import importlib

    sys.dont_write_bytecode = True
    sys.path.insert(0, REPO)
    mod = importlib.import_module("Network.ReActNet_A_Qaw")
    torch.manual_seed(0)
    model = _randomize(mod.Reactnet(num_classes=10, imagenet=False), seed=4).eval()
    err, g = _compare(model, np.random.default_rng(4).standard_normal((1, 32, 32, 3)))
    assert err < 1e-9, err
    assert g.op_counts()["BConv2D"] == 31 and g.op_counts()["Concat"] == 5
