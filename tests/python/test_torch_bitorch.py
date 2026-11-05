"""bitorch model-hub adapter tests (python/bitino/adapters/torch_bitorch.py).

The synthetic modules below re-implement the documented eval forward of the bitorch 0.3.0 classes the
adapter maps (QConv2d = QActivation(sign, 0 -> +1) + pad -1 + conv with sign(W); DenseLayer; ImprovementBlock;
ResNetE BasicBlock; BaseNetDense / _ResnetE containers) under the same class names, so the adapter is tested
without the GPL source. If the bitorch 0.3.0 release is extracted under the main checkout
(.cache/repos/bitorch-pypi/bitorch-0.3.0, or BITINO_BITORCH_REPO), the official ResnetE18, DenseNet28 and
MeliusNet22 are also converted with random weights and compared in float64.
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor  # noqa: E402
from bitino.adapters import torch_bitorch  # noqa: E402

MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
REPO = os.environ.get("BITINO_BITORCH_REPO", os.path.join(MAIN, ".cache", "repos", "bitorch-pypi", "bitorch-0.3.0"))


# ----------------------------------------------------------------- synthetic bitorch-like modules
class Sign(nn.Module):
    def forward(self, x):
        s = torch.sign(x)
        return torch.where(s == 0, torch.ones_like(s), s)


class QActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = Sign()

    def forward(self, x):
        return self.activation_function(x)


class QConv2dBase(nn.Conv2d):
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0, pad_value=-1.0):
        super().__init__(cin, cout, kernel_size, stride=stride, padding=padding, bias=False)
        self._weight_quantize = Sign()
        self._pad_value = pad_value
        self.activation = QActivation()

    def forward(self, x):
        a = F.pad(self.activation(x), self._reversed_padding_repeated_twice, mode="constant", value=self._pad_value)
        return F.conv2d(a, self._weight_quantize(self.weight), None, self.stride, 0, self.dilation, self.groups)


class DenseLayer(nn.Module):
    def __init__(self, c, growth):
        super().__init__()
        self.dropout = 0
        self.features = nn.Sequential(nn.BatchNorm2d(c), QConv2dBase(c, growth, 3, padding=1))

    def forward(self, x):
        return torch.cat([x, self.features(x)], dim=1)


class ImprovementBlock(nn.Module):
    def __init__(self, channels, in_channels):
        super().__init__()
        self.body = nn.Sequential(nn.BatchNorm2d(in_channels), QConv2dBase(in_channels, channels, 3, padding=1))
        self.use_sliced_addition = channels != in_channels
        if self.use_sliced_addition:
            self.slices = [0, in_channels - channels, in_channels]
            self.slices_add_x = [False, True]

    def forward(self, x):
        y = self.body(x)
        if not self.use_sliced_addition:
            return x + y
        a, b = x[:, :self.slices[1]], x[:, self.slices[1]:]
        return torch.cat([a, b + y], dim=1)


class _MeliusNet(nn.Module):
    """Stem (conv/BN/ReLU/maxpool) -> DenseBlock_1 (DenseLayer + ImprovementBlock x2) -> transition
    (bn, max_pool, relu, fp_conv) -> DenseBlock_2 -> finalize -> output."""

    def __init__(self, n_cls=7):
        super().__init__()
        f = [nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
        self.features = nn.Sequential(*f)
        b1 = nn.Sequential()
        b1.add_module("DenseLayer_1", DenseLayer(8, 4))
        b1.add_module("ImprovementBlock1", ImprovementBlock(4, 12))
        b1.add_module("DenseLayer_2", DenseLayer(12, 4))
        b1.add_module("ImprovementBlock2", ImprovementBlock(4, 16))
        self.features.add_module("DenseBlock_1", b1)
        self.features.add_module("Transition_1", nn.Sequential(nn.BatchNorm2d(16), nn.MaxPool2d(2, 2), nn.ReLU(),
                                                               nn.Conv2d(16, 8, 1, bias=False)))
        b2 = nn.Sequential()
        b2.add_module("DenseLayer_1", DenseLayer(8, 4))
        b2.add_module("ImprovementBlock1", ImprovementBlock(4, 12))
        self.features.add_module("DenseBlock_2", b2)
        self.current_dense_block = b2
        self.finalize = nn.Sequential(nn.BatchNorm2d(12), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.output = nn.Linear(12, n_cls)

    def forward(self, x):
        return self.output(self.finalize(self.features(x)))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.shall_downsample = cin != cout
        self.downsample = (nn.Sequential(nn.AvgPool2d(2, stride), nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
                           if self.shall_downsample else nn.Module())
        self.body = nn.Sequential(QConv2dBase(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm2d(cout))
        self.activation = nn.ReLU()

    def forward(self, x):
        res = self.downsample(x) if self.shall_downsample else x
        return self.body(x) + res


class _ResnetE(nn.Module):
    def __init__(self, n_cls=5):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8),
                                      nn.Sequential(BasicBlock(8, 8, 1), BasicBlock(8, 8, 1)),
                                      nn.Sequential(BasicBlock(8, 16, 2), BasicBlock(16, 16, 1)),
                                      nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.output_layer = nn.Linear(16, n_cls)

    def forward(self, x):
        return self.output_layer(self.features(x))


def _randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(0.3 * torch.randn(m.running_mean.shape, generator=g))
                m.running_var.copy_(0.5 + torch.rand(m.running_var.shape, generator=g))
    return model.eval()


def _compare(model, x_nhwc, exact=True):
    g, info = torch_bitorch.convert(model, "test", input_hw=x_nhwc.shape[1:3])
    if exact:
        for name, v in info["float64_constants"].items():
            g.constants[name].data = v
    m64 = model.to(torch.float64)
    with torch.no_grad():
        ref = m64(torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2)))).numpy()
    model.to(torch.float32)
    out = executor.run(g, {"input": x_nhwc}, float_dtype=np.float64)[g.outputs[0]]
    return float(np.max(np.abs(out - ref))), g, info


# ----------------------------------------------------------------- tests
def test_meliusnet_like_float64_semantics():
    model = _randomize(_MeliusNet())
    err, g, info = _compare(model, np.random.default_rng(0).standard_normal((1, 16, 16, 3)))
    assert err < 1e-10, err
    ops = g.op_counts()
    assert ops["BConv2D"] == 6 and ops["Concat"] == 3 and ops["Pad"] == 3 and ops["Add"] == 3
    assert all(n.attrs["pad_value"] == -1.0 and n.attrs["input_binarize"] == "ge" for n in g.nodes if n.op == "BConv2D")


def test_resnete_like_float64_semantics_and_unused_relu():
    model = _randomize(_ResnetE(), seed=1)
    err, g, info = _compare(model, np.random.default_rng(1).standard_normal((1, 8, 8, 3)))
    assert err < 1e-10, err
    assert g.op_counts()["BConv2D"] == 4 and g.op_counts()["AvgPool2D"] == 1
    assert sum("activation" in u for u in info["unused_in_forward"]) == 4


def test_exact_zero_inputs_binarize_to_plus_one():
    """bitorch sign(0) = +1 (also -0.0): the ge IR must match the source on inputs with exact zeros at every
    binarization point, with the float32 constants as stored."""
    model = _randomize(_ResnetE(), seed=2)
    with torch.no_grad():
        model.features[1].weight.zero_()     # stem BN gamma 0 -> every BN output equals beta ...
        model.features[1].bias[:4] = 0.0     # ... and channels 0-3 are exactly 0
    x = np.random.default_rng(2).standard_normal((1, 8, 8, 3))
    g, _ = torch_bitorch.convert(model, "t", input_hw=(8, 8))
    env = executor.run(g, {"input": x.astype(np.float32)}, keep_intermediates=True)
    first = next(n for n in g.nodes if n.op == "BConv2D")
    assert np.sum(env[first.inputs[0]][..., :4] == 0) == 8 * 8 * 4
    with torch.no_grad():
        ref = model(torch.from_numpy(np.ascontiguousarray(x.astype(np.float32).transpose(0, 3, 1, 2)))).numpy()
    assert np.max(np.abs(env[g.outputs[0]] - ref)) < 1e-4


def test_unsupported_settings_raise():
    model = _randomize(_ResnetE())
    with pytest.raises(ValueError):
        torch_bitorch.convert(model, "t", input_hw=(8, 8), binarize="sign_zero")
    model.features[2][0].body[0]._pad_value = 0.0
    with pytest.raises(NotImplementedError):
        torch_bitorch.convert(model, "t", input_hw=(8, 8))
    model = _randomize(_ResnetE())
    model.features[2][1].body.append(nn.PReLU())
    with pytest.raises(NotImplementedError):
        torch_bitorch.convert(model, "t", input_hw=(8, 8))


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "bitorch", "models", "meliusnet.py")), reason="bitorch 0.3.0 not available")
@pytest.mark.parametrize("arch,n_bconv", [("ResnetE18", 16), ("DenseNet28", 23), ("MeliusNet22", 34)])
def test_official_bitorch_random_weights_float64(arch, n_bconv):
    from convert_torch_cnn import _load_bitorch

    model, _, _ = _load_bitorch(REPO, arch, None, input_hw=(32, 32))
    _randomize(model, seed=3)
    err, g, _ = _compare(model, np.random.default_rng(3).standard_normal((1, 32, 32, 3)))
    assert err < 1e-9, err
    assert g.op_counts()["BConv2D"] == n_bconv
