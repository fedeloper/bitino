"""Adapter: NNCF-binarized torchvision ResNet-18 (OpenVINO NNCF <= 2.9 ``binarization`` algorithm) -> Bitino IR.

Source: nncf 2.9.0 (PyPI; ``nncf/torch/binarization/{layers,binarize_functions}.py``; the binarization
code is functionally identical to the v2.5.0 tag the checkpoints were published with, see
the research notes (not published)). The adapter reads the *instantiated* NNCFNetwork
(torchvision ``resnet18`` wrapped by ``create_compressed_model``, checkpoint state loaded with
``load_state``) plus the binarizer constants captured at the binarizer call sites of an fp32 forward
(``capture``, see tools/convert_nncf.py): the effective binary weight ``wb`` returned by the weight
binarizer, and the ``scale`` / ``threshold`` arguments the activation binarizer passes to
``ActivationBinarizationScaleThresholdFn`` (``self.scale.exp()``, ``self.threshold.sigmoid()``).

Semantics (nncf 2.9.0 CPU forward, binarize_functions.py):
  * weights (XNOR): ``wb = ((w > 0) * 2 - 1) * mean|W|[o]`` (mean over dims 1-3: one scale per *output*
    channel); DoReFa: one scalar for the whole tensor. NNCF's CPU path uses mean|W| of the whole tensor, but
    the CUDA kernel the zoo checkpoint was trained/evaluated with uses mean|W| of output filter 0 only
    (``wb_cuda_forward``: elements_per_scale = numel / size(0) also when per_channel is false); the converter
    captures whichever scale the reference forward used (tools/convert_nncf.py --dorefa-weight-scale).
    sign(0) = -1 (no latent weight is 0 here).
  * activations: ``a = (x > t_c) * s`` with ``t = (sigmoid(theta) * s).view(1, C, 1, 1)`` computed in the
    input dtype; x exactly at the threshold maps to 0. a in {0, s}.
  * NNCFConv2d pads with ``nncf_padding_value`` (all 0.0 here), i.e. a padded tap is the "0" level.
  * conv1 (stem), downsample 1x1 convs and fc are not binarized (ignored scopes).

Mapping to existing ops (no new runtime op): with ``b = gt-binarize(x, t) in {-1, +1}``, ``a = s (b + 1) / 2``
and a padded tap has a = 0 <=> b = -1, so over all taps (padded ones included)

    sum a * wb = alpha_o * s * sum (b + 1)/2 * w = alpha_o * s / 2 * (acc + Wsum_o),

where ``acc = BConv2D(x, sign(w), input_binarize=gt, threshold=t, pad_value=-1)`` (exact integer) and
``Wsum_o = sum_{taps, c} sign(w)[o]``. Hence ``post_mul = alpha_o * s / 2``, ``post_add = alpha_o * s * Wsum_o / 2``.
Everything after the conv (BN, ReLU, residual add) is plain float. The thresholds stored in the IR are the
fp32 values the fp32 source compares against (``fl32(sigmoid32 * s32)``); ``float64_constants`` holds the
exact products (the product of two fp32 numbers is exact in fp64) and the exact post_mul/post_add.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph


def _cls(m) -> str:
    return type(m).__name__


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)


def _pair(v) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        assert len(v) == 2
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _to_ohwi(w_oihw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(w_oihw, (0, 2, 3, 1)))


def _f64(t) -> np.ndarray:
    import torch
    return t.detach().cpu().to(torch.float64).reshape(-1).numpy()


BINARIZED = [f"layer{li}.{bi}.conv{ci}" for li in range(1, 5) for bi in range(2) for ci in (1, 2)]


def binarizer_modules(conv):
    """(weight binarizer op, activation binarizer op) of an NNCFConv2d, or (None, None)."""
    ops = list(conv.pre_ops.values())
    if not ops:
        return None, None
    _expect(len(ops) == 2 and _cls(ops[0]) == "UpdateWeight" and _cls(ops[1]) == "UpdateInputs",
            f"unexpected pre_ops {[ _cls(o) for o in ops]}")
    wb, ab = ops[0].op, ops[1].op
    _expect(_cls(wb) in ("XNORBinarize", "DOREFABinarize"), f"unexpected weight binarizer {_cls(wb)}")
    _expect(_cls(ab) == "ActivationBinarizationScaleThreshold", f"unexpected activation binarizer {_cls(ab)}")
    _expect(bool(wb.is_enabled()) and bool(ab.is_enabled()), "binarizer not enabled")
    return wb, ab


def convert(model, capture: dict, mode: str, model_id: str = "", input_name: str = "input",
            input_hw: tuple[int, int] = (224, 224)) -> tuple[Graph, dict]:
    """model: NNCFNetwork around torchvision ResNet (eval). capture[name] = {"wb": fp32 [O,I,kh,kw],
    "scale": fp32 scalar s, "sigmoid": fp32 [C]} from the fp32 forward. mode: "xnor" | "dorefa"."""
    _expect(mode in ("xnor", "dorefa"), f"unknown mode {mode}")
    _expect(not model.training, "model must be in eval mode")
    _expect(hasattr(model, "layer4") and hasattr(model, "maxpool"), "expected a torchvision ResNet")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    binary_weights: dict[str, np.ndarray] = {}

    def folded(name: str, value64) -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, np.asarray(value64).astype(np.float32))

    def bn(m, x: str, name: str) -> str:
        _expect(_cls(m) in ("BatchNorm2d", "NNCFBatchNorm2d") and m.affine and m.running_mean is not None, f"{name}: expected BatchNorm2d")
        _expect(not list(getattr(m, "pre_ops", {}).values()) and not list(getattr(m, "post_ops", {}).values()), f"{name}: NNCF ops on BN")
        scale = _f64(m.weight) / np.sqrt(_f64(m.running_var) + float(m.eps))
        shift = _f64(m.bias) - _f64(m.running_mean) * scale
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", shift)], name=name)

    def fconv(m, x: str, name: str) -> str:
        _expect(_cls(m) == "NNCFConv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros" and float(m.get_padding_value_ref()) == 0.0, f"{name}: unexpected conv")
        _expect(binarizer_modules(m) == (None, None), f"{name}: expected a float (non-binarized) conv")
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", _to_ohwi(m.weight.detach().cpu().numpy().astype(np.float32)), layout="OHWI"), ""],
                          {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                           "activation": "none"}, name=name)

    def bconv(m, x: str, name: str) -> str:
        _expect(_cls(m) == "NNCFConv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros", f"{name}: unexpected conv")
        _expect(float(m.get_padding_value_ref()) == 0.0, f"{name}: nncf_padding_value != 0")
        wbm, abm = binarizer_modules(m)
        _expect(wbm is not None, f"{name}: expected binarizers")
        _expect(_cls(wbm) == {"xnor": "XNORBinarize", "dorefa": "DOREFABinarize"}[mode], f"{name}: weight binarizer {_cls(wbm)} != {mode}")
        c = capture[name]
        wb = np.asarray(c["wb"], np.float32)
        lat = m.weight.detach().cpu().numpy().astype(np.float32)
        _expect(int(np.sum(lat == 0)) == 0, f"{name}: exact-zero latent weights (sign(0) = -1 would need care)")
        sign = np.where(wb > 0, 1, -1).astype(np.int8)
        _expect(np.array_equal(sign, np.where(lat > 0, 1, -1)), f"{name}: binarized weight sign != (w > 0)")
        o = wb.shape[0]
        alpha = np.abs(wb).reshape(o, -1)
        _expect(np.all(alpha == alpha[:, :1]), f"{name}: |wb| not constant per output channel")
        alpha = alpha[:, 0]
        if mode == "dorefa":
            _expect(np.all(alpha == alpha[0]), f"{name}: DoReFa scale not a single scalar")
        _expect(np.all(alpha > 0), f"{name}: zero weight scale")
        s = np.float32(c["scale"])
        sig = np.asarray(c["sigmoid"], np.float32).reshape(-1)
        cin = lat.shape[1]
        _expect(sig.size == cin and np.isfinite(s) and s > 0, f"{name}: activation binarizer constants")
        thr32 = (sig * s).astype(np.float32)                 # fl32(sigmoid * s): the fp32 source's comparison value
        thr64 = sig.astype(np.float64) * np.float64(s)       # exact product (24 x 24 bit mantissas fit in fp64)
        wsum = sign.reshape(o, -1).astype(np.int64).sum(1)
        mul64 = alpha.astype(np.float64) * np.float64(s) / 2.0
        add64 = mul64 * wsum.astype(np.float64)
        binary_weights[name] = sign
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        exact[f"{name}/threshold"] = thr64
        y = g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", _to_ohwi(sign), dtype="bit", layout="OHWI_bit"),
                                   folded(f"{name}/post_mul", mul64), folded(f"{name}/post_add", add64),
                                   g.add_constant(f"{name}/threshold", thr32)],
                       {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": -1,
                        "input_binarize": "gt", "activation": "none", "output": "f32"}, name=name)
        precision.append({"layer": name, "op": "BConv2D", "source": f"NNCFConv2d + {_cls(wbm)} + ActivationBinarizationScaleThreshold",
                          "weights": "binary sign(w>0), " + ("per-output-channel mean|W|" if mode == "xnor" else "per-tensor mean|W|"),
                          "activations": "{0, s}: s * (x > s*sigmoid(theta_c)) -> gt-binarize(x, t_c) in +-1, a = s(b+1)/2",
                          "input_binarize": "gt", "pad_value": -1, "kernel": list(m.kernel_size), "stride": sh,
                          "post": "post_mul = alpha*s/2, post_add = alpha*s*Wsum/2"})
        diag.append({"layer": name, "n_weights": int(sign.size), "n_plus": int(np.sum(sign > 0)), "s": float(s),
                     "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
                     "threshold_min": float(thr32.min()), "threshold_max": float(thr32.max()),
                     "wsum_abs_max": int(np.abs(wsum).max())})
        return y

    def relu(x: str, name: str) -> str:
        return g.add_node("Activation", [x], {"type": "relu"}, name=name)

    h, w_ = input_hw
    x = g.add_input(input_name, [1, h, w_, 3])
    x = fconv(model.conv1, x, "conv1")
    x = bn(model.bn1, x, "bn1")
    x = relu(x, "relu")
    mp = model.maxpool
    _expect(_cls(mp) == "MaxPool2d" and _pair(mp.kernel_size) == (3, 3) and _pair(mp.stride) == (2, 2) and _pair(mp.padding) == (1, 1)
            and _pair(mp.dilation) == (1, 1) and not mp.ceil_mode, "maxpool: expected 3x3/2 pad 1")
    # zero-padded max pool on a post-ReLU (>= 0) tensor: padding never wins, any pad semantics give the same result
    x = g.add_node("MaxPool2D", [x], {"kernel": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]}, name="maxpool")
    precision.append({"layer": "stem", "op": "Conv2D+ChannelAffine+relu+MaxPool2D", "source": "conv1 7x7/2 + bn1 + relu + maxpool",
                      "weights": "fp32", "activations": "fp32"})
    seen = []
    for li in range(1, 5):
        layer = getattr(model, f"layer{li}")
        for bi, blk in enumerate(layer):
            pth = f"layer{li}.{bi}"
            _expect(_cls(blk) == "BasicBlock", f"{pth}: unexpected block {_cls(blk)}")
            t = bconv(blk.conv1, x, f"{pth}.conv1")
            t = bn(blk.bn1, t, f"{pth}.bn1")
            t = relu(t, f"{pth}.relu1")
            u = bconv(blk.conv2, t, f"{pth}.conv2")
            u = bn(blk.bn2, u, f"{pth}.bn2")
            seen += [f"{pth}.conv1", f"{pth}.conv2"]
            residual = x
            if blk.downsample is not None:
                ds = blk.downsample
                _expect(_cls(ds) == "Sequential" and len(ds) == 2, f"{pth}.downsample: unexpected structure")
                r = fconv(ds[0], x, f"{pth}.downsample.0")
                residual = bn(ds[1], r, f"{pth}.downsample.1")
                precision.append({"layer": f"{pth}.downsample", "op": "Conv2D+ChannelAffine", "source": "conv1x1 stride s + BN",
                                  "weights": "fp32", "activations": "fp32"})
            x = g.add_node("Add", [u, residual], name=f"{pth}.add")
            x = relu(x, f"{pth}.relu2")
    _expect(seen == BINARIZED, "unexpected binarized conv set")
    _expect(set(capture) == set(BINARIZED), f"capture covers {sorted(capture)}")
    ap = model.avgpool
    _expect(_cls(ap) == "AdaptiveAvgPool2d" and _pair(ap.output_size) == (1, 1), "avgpool: expected AdaptiveAvgPool2d(1)")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="avgpool")
    fc = model.fc
    _expect(_cls(fc) == "NNCFLinear" and fc.bias is not None and not list(fc.pre_ops.values()), f"fc: unexpected {_cls(fc)}")
    x = g.add_node("Dense", [x, g.add_constant("fc/kernel", fc.weight.detach().cpu().numpy().astype(np.float32)),
                             g.add_constant("fc/bias", fc.bias.detach().cpu().numpy().astype(np.float32))],
                   {"activation": "none"}, name="fc")
    precision.append({"layer": "head", "op": "GlobalAvgPool+Dense", "source": "AdaptiveAvgPool2d(1) + flatten + Linear",
                      "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "mode": mode,
            "float64_constants": exact, "binary_weights": binary_weights}
    return g, info
