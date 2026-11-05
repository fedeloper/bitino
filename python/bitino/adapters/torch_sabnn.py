"""Adapter: instantiated SA-BNN ResNet-18 (anonymous1443/SA-BNN, AAAI'21 review release) -> Bitino IR.

Source: the anonymous repo's Google Drive folder "Training Techniques/resnet18_61.696", file
``birealnet.py.txt`` (sha256 7570f986dabb0538aca4c6c676d48d5593c384d16978e63ebfe776e826b4ab4e; the
anonymous GitHub repo has no code, only README links). ``BiRealNet(BasicBlock, [2, 2, 2, 2])`` from
``birealnet18()``. The adapter reads the *instantiated* nn.Module built from that file (weights loaded
with torch.load(weights_only=True), strict=True); it mirrors BiRealNet.forward and BasicBlock.forward
explicitly and raises on any module type or setting it does not know.

Semantics captured (birealnet.py.txt line numbers):
  * Stem (118-123, 148-152): Conv2d 7x7 s2 p3 (no bias) + BN + PReLU(64) + MaxPool2d(3, 2, 1).
  * BasicBlock (62-110), two binary convs, BN *before* each sign, one shortcut reused twice:
        a  = bn2(x)
        r  = downsample(a)  if downsample is not None else x        (note: the BN'd input is downsampled)
        h  = prelu (bn1(conv1(sign(a))) + r)
        out = prelu2(bn4(conv2(sign(bn3(h)))) + r)
    ``downsample2`` (82, 105-106) is the *same* module object as ``downsample`` and is applied only when
    ``out.size() != residual.size()``; ``residual`` already has the block's output shape (it is the
    downsampled one whenever the shape changes), so the branch never runs (asserted from the shapes).
  * BinaryActivation (20-37), eval: sign(x).detach() - out3.detach() + out3 with the piecewise-quadratic
    ApproxSign out3 (|out3| <= 1, sign(out3) = sign(x)) -> exactly torch.sign(x), sign(0) = 0 (same
    argument as in torch_rbonn; the converter checks it with forward hooks) -> BConv2D input_binarize
    "sign_zero" (exact) or "ge" (deployable, 0 -> +1).
  * HardBinaryConv (40-59), eval: weights stored flat (N, 1), viewed as (out, in, 3, 3);
    F.conv2d(x, s*sign(W) - clamp(W) + clamp(W)) with s = mean|W| per output channel (nested torch.mean
    over kw, kh, cin). The STE terms cancel up to fp32 rounding -> sign(W) bits, post_mul = s (float64,
    rounded once). Zero padding of the binarized input -> pad_value 0. A latent weight that is exactly 0
    (torch.sign -> 0) is rejected.
  * BN (eval, running stats) -> ChannelAffine(scale, shift) folded in float64. The pre-sign BN output
    is a materialized ChannelAffine (it also feeds the downsample in the first block of a stage).
  * Downsample (_make_layer 131-146): AvgPool2d(2, stride) + FP conv1x1 + BN -> AvgPool2D + Conv2D +
    ChannelAffine.
  * Head (159-161): AdaptiveAvgPool2d(1) + Linear -> GlobalAvgPool + Dense.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph
from .torch_rbonn import BINARIZE_MODES, _cls, _expect, _f64, _pair, _to_ohwi


def _binary_bits(bc, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(sign bits OIHW int8, float64 per-out-channel mean|W|, diagnostics) of an SA-BNN HardBinaryConv."""
    import torch

    _expect(_cls(bc) == "HardBinaryConv", f"{name}: unexpected {_cls(bc)}")
    o, i, kh, kw = (int(v) for v in bc.shape)
    _expect((kh, kw) == (3, 3) and int(bc.padding) == 1 and int(bc.stride) in (1, 2), f"{name}: unexpected kernel/stride/padding")
    _expect(tuple(bc.weights.shape) == (o * i * kh * kw, 1) and bc.number_of_weights == o * i * kh * kw, f"{name}: flat weight shape")
    w = bc.weights.detach().cpu().view(bc.shape)            # birealnet.py.txt:50
    zeros = int((w == 0).sum())
    _expect(zeros == 0, f"{name}: {zeros} latent weights are exactly 0 (torch.sign gives a 0 weight)")
    w64 = w.to(torch.float64)
    s = torch.mean(torch.mean(torch.mean(abs(w64), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
    return (w.sign().numpy().astype(np.int8), s.reshape(-1).numpy(),
            {"layer": name, "latent_weight_zeros": zeros, "n_weights": int(w.numel()),
             "latent_abs_gt_1": int((w.abs() > 1).sum())})


def convert(model, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (224, 224),
            binarize: str = "sign_zero") -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths (e.g. layer2.0.bn2)."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    _expect(_cls(model) == "BiRealNet", f"unsupported top-level model class {_cls(model)} (expected SA-BNN BiRealNet)")
    _expect(not model.training, "model must be in eval mode")
    g = Graph()
    binary_weights: dict[str, np.ndarray] = {}
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    visited: set[str] = set()

    def folded(name: str, value64: np.ndarray) -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, np.asarray(value64).astype(np.float32))

    def bn(m, x: str, name: str) -> str:
        _expect(_cls(m) == "BatchNorm2d" and m.affine and m.running_mean is not None, f"{name}: expected BatchNorm2d with running stats")
        scale = _f64(m.weight) / np.sqrt(_f64(m.running_var) + float(m.eps))
        shift = _f64(m.bias) - _f64(m.running_mean) * scale
        visited.add(name)
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", shift)], name=name)

    def prelu(m, x: str, name: str, c: int) -> str:
        _expect(_cls(m) == "PReLU", f"{name}: unexpected {_cls(m)}")
        slope = m.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
        _expect(slope.size in (1, c), f"{name}: {slope.size} slopes for {c} channels")
        visited.add(name)
        return g.add_node("Activation", [x, g.add_constant(f"{name}/slope", np.broadcast_to(slope, (c,)).copy())], {"type": "prelu"}, name=name)

    def conv(m, x: str, name: str) -> str:
        _expect(_cls(m) == "Conv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros", f"{name}: unexpected Conv2d")
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        visited.add(name)
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", _to_ohwi(m.weight.detach().cpu().numpy().astype(np.float32)), layout="OHWI"), ""],
                          {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                           "activation": "none"}, name=name)

    def bconv(blk, x: str, p: str, attr: str, act_attr: str) -> str:
        bc = getattr(blk, attr)
        name = f"{p}.{attr}"
        _expect(_cls(getattr(blk, act_attr)) == "BinaryActivation", f"{p}.{act_attr}: unexpected")
        visited.update({name, f"{p}.{act_attr}"})
        sign, post_mul, d = _binary_bits(bc, name)
        binary_weights[name] = sign
        diag.append(d)
        s = int(bc.stride)
        t = g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", _to_ohwi(sign), dtype="bit", layout="OHWI_bit"),
                                   folded(f"{name}/post_mul", post_mul), "", ""],
                       {"strides": [s, s], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
                        "input_binarize": binarize, "activation": "none", "output": "f32"}, name=name)
        precision.append({"layer": name, "op": "BConv2D", "source": "BN -> BinaryActivation + HardBinaryConv",
                          "weights": "binary sign(W), per-channel mean|W|", "activations": "binary (sign of the preceding BN output)",
                          "input_binarize": binarize, "source_binarize": "torch.sign (sign(0)=0)", "pad_value": 0.0,
                          "kernel": [3, 3], "stride": s})
        return t

    h, w = input_hw
    x = g.add_input(input_name, [1, h, w, 3])
    x = conv(model.conv1, x, "conv1")
    x = bn(model.bn1, x, "bn1")
    x = prelu(model.prelu, x, "prelu", 64)
    mp = model.maxpool
    _expect(_cls(mp) == "MaxPool2d" and _pair(mp.dilation) == (1, 1) and not mp.ceil_mode, "maxpool: unexpected")
    k, s, pd = _pair(mp.kernel_size), _pair(mp.stride), _pair(mp.padding)
    x = g.add_node("MaxPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [pd[0], pd[0], pd[1], pd[1]]}, name="maxpool")
    precision.append({"layer": "conv1", "op": "Conv2D+ChannelAffine+PReLU+MaxPool2D", "source": "Conv2d 7x7 + BN + PReLU + MaxPool2d",
                      "weights": "fp32", "activations": "fp32"})
    unused: list[str] = []

    for lname in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, lname)
        _expect(_cls(layer) == "Sequential", f"{lname}: unexpected {_cls(layer)}")
        for bi, blk in enumerate(layer):
            p = f"{lname}.{bi}"
            _expect(_cls(blk) == "BasicBlock" and hasattr(blk, "binary_conv2"), f"{p}: unexpected block {_cls(blk)}")
            cin, cout = int(blk.binary_conv.shape[1]), int(blk.binary_conv.shape[0])
            stride = int(blk.binary_conv.stride)
            _expect(int(blk.binary_conv2.stride) == 1 and tuple(blk.binary_conv2.shape[:2]) == (cout, cout), f"{p}.binary_conv2: unexpected")
            a = bn(blk.bn2, x, f"{p}.bn2")
            t = bconv(blk, a, p, "binary_conv", "binary_activation")
            t = bn(blk.bn1, t, f"{p}.bn1")
            if blk.downsample is not None:
                ds = blk.downsample
                _expect(_cls(ds) == "Sequential" and len(ds) == 3, f"{p}.downsample: unexpected structure")
                ap = ds[0]
                _expect(_cls(ap) == "AvgPool2d" and not ap.ceil_mode and ap.divisor_override is None and _pair(ap.padding) == (0, 0),
                        f"{p}.downsample.0: unexpected {_cls(ap)}")
                r = g.add_node("AvgPool2D", [a], {"kernel": list(_pair(ap.kernel_size)), "strides": list(_pair(ap.stride)),
                                                  "pads": [0, 0, 0, 0], "count_include_pad": 1 if ap.count_include_pad else 0},
                               name=f"{p}.downsample.0")
                r = conv(ds[1], r, f"{p}.downsample.1")
                residual = bn(ds[2], r, f"{p}.downsample.2")
                visited.add(f"{p}.downsample")
                precision.append({"layer": f"{p}.downsample", "op": "AvgPool2D+Conv2D+ChannelAffine",
                                  "source": "AvgPool2d + conv1x1 + BN applied to bn2(x)", "weights": "fp32", "activations": "fp32"})
            else:
                # identity shortcut: the source would crash (residual size mismatch) unless shapes agree
                _expect(stride == 1 and cin == cout, f"{p}: identity shortcut with stride {stride}, {cin}->{cout}")
                residual = x
            # downsample2 is applied only if out.size() != residual.size(); residual already has the output
            # shape [cout, H/stride, W/stride] here, so it never runs.
            _expect(blk.downsample2 is blk.downsample, f"{p}.downsample2 is not the shared downsample module")
            if blk.downsample2 is not None:
                unused.append(f"{p}.downsample2 (same module as {p}.downsample; never applied: residual already has the output shape)")
            t = g.add_node("Add", [t, residual], name=f"{p}.add")
            t = prelu(blk.prelu, t, f"{p}.prelu", cout)
            u = bn(blk.bn3, t, f"{p}.bn3")
            u = bconv(blk, u, p, "binary_conv2", "binary_activation2")
            u = bn(blk.bn4, u, f"{p}.bn4")
            u = g.add_node("Add", [u, residual], name=f"{p}.add2")
            x = prelu(blk.prelu2, u, f"{p}.prelu2", cout)

    _expect(_cls(model.avgpool) == "AdaptiveAvgPool2d" and _pair(model.avgpool.output_size) == (1, 1), "avgpool: expected AdaptiveAvgPool2d(1)")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="avgpool")
    fc = model.fc
    _expect(_cls(fc) == "Linear" and fc.bias is not None, f"fc: unexpected {_cls(fc)}")
    visited.add("fc")
    x = g.add_node("Dense", [x, g.add_constant("fc/kernel", fc.weight.detach().cpu().numpy().astype(np.float32)),
                             g.add_constant("fc/bias", fc.bias.detach().cpu().numpy().astype(np.float32))],
                   {"activation": "none"}, name="fc")
    precision.append({"layer": "fc", "op": "Dense", "source": "Linear", "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)

    known = {"BiRealNet", "Sequential", "BasicBlock", "Conv2d", "BatchNorm2d", "MaxPool2d", "AvgPool2d", "BinaryActivation",
             "HardBinaryConv", "PReLU", "AdaptiveAvgPool2d", "Linear"}
    for n, m in model.named_modules(remove_duplicate=False):
        _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        ok = any(n == v or n.startswith(v + ".") for v in visited)
        _expect(ok, f"parameter/buffer {n} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": unused, "float64_constants": exact, "binary_weights": binary_weights}
    return g, info
