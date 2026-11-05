"""Adapter: FDA-BNN CIFAR-10 ResNet-20 (MindSpore model zoo research/cv/FDA-BNN, ``fdabnn.ckpt``) -> Bitino IR.

Reads the checkpoint tensors directly (``mindspore_ckpt.read``; no MindSpore). Semantics from the official
``src/quan.py`` / ``src/resnet.py`` (see tools/fdabnn_reference.py for the cited port used as reference):

  * QuanConv: ``conv2d(sign(x), sign(W)*mean|W|)``, zero padding of the binarized input, one positive scale per
    layer -> BConv2D (sign bits of W, pad_value 0, post_mul = mean|W| computed in float64, rounded once).
    MindSpore Sign(0) = 0 for activations -> ``sign_zero`` (exact) / ``ge`` (deployable). A latent weight that is
    exactly 0 would be a zero weight (sign 0): rejected.
  * BasicCell: ``out = bn2(prelu2(conv2(bn1(prelu1(conv1(x)))))) + shortcut`` -> BConv2D, Activation prelu
    (the scalar slope broadcast to the channels), ChannelAffine (BN, float64 fold), Add; shortcut
    ``x[:, :, ::2, ::2]`` + channel zero pad -> AvgPool2D(1, stride 2) + Pad.
  * Stem ``bn1(conv1(x))`` (no activation) -> Conv2D + ChannelAffine; head ``relu`` -> GlobalAvgPool -> Dense.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

BINARIZE_MODES = ("sign_zero", "ge")
EPS = 1e-5


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)


def convert(params: dict, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (32, 32),
            binarize: str = "sign_zero") -> tuple[Graph, dict]:
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    used: set[str] = set()
    binary_weights: dict[str, np.ndarray] = {}

    def p(name: str) -> np.ndarray:
        _expect(name in params, f"checkpoint has no {name}")
        used.add(name)
        return params[name]

    def folded(name: str, v64) -> str:
        v = np.asarray(v64, np.float64)
        exact[name] = v
        return g.add_constant(name, v.astype(np.float32))

    def bn(x: str, name: str, c: int) -> str:
        gamma, beta = p(f"{name}.gamma").astype(np.float64), p(f"{name}.beta").astype(np.float64)
        mean, var = p(f"{name}.moving_mean").astype(np.float64), p(f"{name}.moving_variance").astype(np.float64)
        _expect(gamma.shape == (c,), f"{name}: {gamma.shape} vs {c} channels")
        scale = gamma / np.sqrt(var + EPS)
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", beta - mean * scale)], name=name)

    def prelu(x: str, name: str, c: int) -> str:
        w = p(f"{name}.w").astype(np.float32).reshape(-1)
        _expect(w.size == 1, f"{name}: expected one PReLU slope, got {w.size}")
        return g.add_node("Activation", [x, g.add_constant(f"{name}/slope", np.full(c, w[0], np.float32))], {"type": "prelu"}, name=name)

    def quanconv(x: str, name: str, cin: int, cout: int, stride: int) -> str:
        w = p(f"{name}.weight")
        _expect(w.shape == (cout, cin, 3, 3) and w.dtype == np.float32, f"{name}: weight {w.shape} {w.dtype}")
        n_zero = int(np.sum(w == 0))
        _expect(n_zero == 0, f"{name}: {n_zero} latent weights are exactly 0 (Sign gives a zero weight)")
        sign = np.sign(w).astype(np.int8)
        binary_weights[name] = sign
        scale = np.mean(np.abs(w.astype(np.float64)))        # ReduceMean(Abs(w)) over the whole tensor, float64
        precision.append({"layer": name, "op": "BConv2D", "source": "QuanConv (WQuan sign(W)*mean|W|, AQuan Sign)",
                          "weights": "binary sign(W), one scale mean|W| per layer", "activations": "binary", "input_binarize": binarize,
                          "source_binarize": "mindspore Sign(x) (Sign(0)=0)", "pad_value": 0.0, "kernel": [3, 3], "stride": stride})
        diag.append({"layer": name, "n_weights": int(sign.size), "n_plus": int(np.sum(sign > 0)), "mean_abs_w": float(scale)})
        return g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", np.ascontiguousarray(sign.transpose(0, 2, 3, 1)), dtype="bit",
                                                        layout="OHWI_bit"), folded(f"{name}/post_mul", np.full(cout, scale)), "", ""],
                          {"strides": [stride, stride], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
                           "input_binarize": binarize, "activation": "none", "output": "f32"}, name=name)

    h, w_ = input_hw
    x = g.add_input(input_name, [1, h, w_, 3])
    k = p("conv1.weight")
    _expect(k.shape == (16, 3, 3, 3), f"conv1: {k.shape}")
    x = g.add_node("Conv2D", [x, g.add_constant("conv1/kernel", np.ascontiguousarray(k.astype(np.float32).transpose(0, 2, 3, 1)), layout="OHWI"), ""],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0, "activation": "none"}, name="conv1")
    x = bn(x, "bn1", 16)
    precision.append({"layer": "conv1", "op": "Conv2D+ChannelAffine", "source": "nn.Conv2d 3x3 (no bias) + BatchNorm2d (no activation)",
                      "weights": "fp32", "activations": "fp32"})
    c = 16
    for li, planes in enumerate((16, 32, 64), start=1):
        for bi in range(3):
            pth = f"layer{li}.{bi}"
            st = 2 if (bi == 0 and li > 1) else 1
            ho, wo = (h - 1) // st + 1, (w_ - 1) // st + 1
            t = bn(prelu(quanconv(x, f"{pth}.conv1", c, planes, st), f"{pth}.relu1", planes), f"{pth}.bn1", planes)
            t = bn(prelu(quanconv(t, f"{pth}.conv2", planes, planes, 1), f"{pth}.relu2", planes), f"{pth}.bn2", planes)
            r = x
            if st != 1 or c != planes:
                r = g.add_node("AvgPool2D", [x], {"kernel": [1, 1], "strides": [st, st], "pads": [0, 0, 0, 0], "count_include_pad": 1},
                               name=f"{pth}.shortcut.subsample")
                pad = planes // 4
                _expect(c + 2 * pad == planes, f"{pth}: channel pad {c} + 2*{pad} != {planes}")
                r = g.add_node("Pad", [r], {"pads": [0, 0, 0, 0, 0, 0, pad, pad], "value": 0.0}, name=f"{pth}.shortcut.pad")
                precision.append({"layer": f"{pth}.shortcut", "op": "AvgPool2D+Pad", "weights": "none", "activations": "fp32",
                                  "source": "x[:, :, ::2, ::2] + nn.Pad channels"})
            x = g.add_node("Add", [t, r], name=f"{pth}.add")
            h, w_, c = ho, wo, planes
    x = g.add_node("Activation", [x], {"type": "relu"}, name="relu")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="ap")
    lw, lb = p("linear.weight"), p("linear.bias")
    _expect(lw.shape == (10, 64) and lb.shape == (10,), f"linear: {lw.shape}")
    x = g.add_node("Dense", [x, g.add_constant("linear/kernel", lw.astype(np.float32)), g.add_constant("linear/bias", lb.astype(np.float32))],
                   {"activation": "none"}, name="linear")
    precision.append({"layer": "head", "op": "Activation(relu)+GlobalAvgPool+Dense", "source": "ReLU + ReduceMean(2,3) + Dense",
                      "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)
    unused = sorted(set(params) - used)
    _expect(not unused, f"checkpoint tensors not used by the adapter: {unused[:6]}")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": [], "float64_constants": exact, "binary_weights": binary_weights}
    return g, info
