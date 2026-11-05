"""Adapter: instantiated official BBCU models (Zj-BinXia/BBCU, ICLR'23 "Basic Binary Convolution Unit for
Binarized Image Restoration Network") -> Bitino IR. Image-to-image: the graph output is the restored /
super-resolved image (NHWC), not logits.

Sources (repo commit c47df53): BBCU-denoiseAndblocking/models/network_dncnn_bin.py (color denoising, DnCNN,
nb=20, k=130), models/network_dncnn3_bin.py (grayscale JPEG deblocking, same with conv biases),
BBCU-SR/bbcu/archs/bbcul_arch.py (BBCU-L) and bbcum_arch.py (BBCU-M); binary units in quants/quant_convs_bbcu.py
(denoise) and BBCU-SR/quant_convs_bbcu.py (SR; lines 1-99 identical). The adapter reads the *instantiated*
nn.Module (weights loaded with torch.load(weights_only=True), strict=True), mirrors each forward explicitly and
raises on any module type or setting it does not know.

Semantics captured:
  * ``x * k`` (k = 130 in every test script / YAML) -> ChannelAffine(scale k): the same single fp32 product.
  * BinaryConv2d.forward (quant_convs_bbcu.py:92-98): ``out = RPReLU(HardBinaryConv(sign(x + move0))) + x``.
      - LearnableBias move0 -> ChannelAffine(bias only) feeding the BConv2D (the runtime folds it into a
        threshold at plan time).
      - BinaryActivation (27-44): ``sign(x).detach() - out3.detach() + out3`` with |out3| <= 1, sign(out3) =
        sign(x) -> exactly torch.sign(x) (the RBONN/SA-BNN argument; the converter checks it with forward
        hooks), sign(0) = 0 -> input_binarize "sign_zero" (exact) or "ge" (deployable, 0 -> +1).
      - HardBinaryConv (55-75): ``F.conv2d(x, s*sign(W) - clamp(W) + clamp(W), bias)``, s = mean|W| per output
        channel (nested torch.mean over kw, kh, cin); the STE terms cancel up to fp32 rounding -> sign(W) bits,
        post_mul = s (float64, rounded once), post_add = conv bias if any (deblocking). Zero padding of the
        binarized input -> pad_value 0. A latent weight that is exactly 0 (torch.sign -> 0) is rejected.
      - RPReLU (16-25): ``bias1(prelu(bias0(x)))`` -> ChannelAffine(bias) + Activation prelu + ChannelAffine(bias).
      - residual ``+ x`` -> Add.
  * DnCNN (network_dncnn_bin.py:40-73 / network_dncnn3_bin.py): head Conv2d 3x3 (+bias for deblocking) + ReLU,
    18 BinaryConv2d, tail Conv2d 3x3; forward ``x = x*k; n = model(x); x = x - n; x = x / k`` ->
    Add(xk, ChannelAffine(n, scale -1)) (exact negation) and ChannelAffine(scale fl(1/k)) (one fp32 product
    instead of torch's division: <= 1 ulp, after the last binarization).
  * BBCUL (bbcul_arch.py:53-68): conv_first + LeakyReLU(0.1), 16 BinaryBlock (two BinaryConv2d each, no block
    residual: ``#res += x``), upconv1 [+ upconv2 for x4] (FP Conv2d) + PixelShuffle + LeakyReLU, conv_hr +
    LeakyReLU, conv_last, ``+ F.interpolate(x*k, scale, 'bilinear', align_corners=False)``, ``/ k``.
  * BBCUM (bbcum_arch.py:57-72): as BBCUL, but upconv1/2 are BinaryUpConv2d (SR quant_convs_bbcu.py:100-131:
    move0, sign, HardBinaryConv 64->256 (no bias), RPReLU(256), PixelShuffle, ``+ F.interpolate(x, nearest)``)
    and conv_hr is a BinaryConv2d (with its own ``+ x``), followed by LeakyReLU.
  * PixelShuffle / nearest / bilinear: restoration_common (Reshape/Transpose, row Gather, Gather + Conv2D).
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph
from . import restoration_common as rc
from .torch_rbonn import BINARIZE_MODES, _cls, _expect, _f64, _pair, _to_ohwi

ARCHS = ("dncnn", "bbcul", "bbcum")


def _binary_bits(bc, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(sign bits OIHW int8, float64 per-out-channel mean|W|, diagnostics) of a BBCU HardBinaryConv."""
    import torch

    _expect(_cls(bc) == "HardBinaryConv", f"{name}: unexpected {_cls(bc)}")
    _expect(bc.groups == 1 and _pair(bc.dilation) == (1, 1) and _pair(bc.stride) == (1, 1) and bc.padding_mode == "zeros",
            f"{name}: unexpected conv settings")
    w = bc.weight.detach().cpu()
    o, i, kh, kw = (int(v) for v in w.shape)
    _expect((kh, kw) == (3, 3) and _pair(bc.padding) == (1, 1), f"{name}: unexpected kernel/padding")
    zeros = int((w == 0).sum())
    _expect(zeros == 0, f"{name}: {zeros} latent weights are exactly 0 (torch.sign gives a 0 weight)")
    w64 = w.to(torch.float64)
    s = torch.mean(torch.mean(torch.mean(abs(w64), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
    return (w.sign().numpy().astype(np.int8), s.reshape(-1).numpy(),
            {"layer": name, "latent_weight_zeros": zeros, "n_weights": int(w.numel()), "latent_abs_gt_1": int((w.abs() > 1).sum())})


def convert(model, arch: str, model_id: str = "", input_hw: tuple[int, int] = (96, 96), in_ch: int | None = None,
            binarize: str = "sign_zero") -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths (e.g. model.3.binary_conv)."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    _expect(arch in ARCHS, f"unknown arch {arch}")
    _expect(not model.training, "model must be in eval mode")
    want = {"dncnn": "DnCNN", "bbcul": "BBCUL", "bbcum": "BBCUM"}[arch]
    _expect(_cls(model) == want, f"unsupported top-level model class {_cls(model)} (expected {want})")
    g = Graph()
    binary_weights: dict[str, np.ndarray] = {}
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    visited: set[str] = set()
    k = float(model.k)
    h, w = input_hw

    def folded(name: str, value64: np.ndarray) -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, np.asarray(value64).astype(np.float32))

    def vec(t) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def bias_node(m, x: str, name: str, c: int) -> str:
        _expect(_cls(m) == "LearnableBias" and tuple(m.bias.shape) == (1, c, 1, 1), f"{name}: unexpected {_cls(m)}")
        visited.add(name)
        return g.add_node("ChannelAffine", [x, "", g.add_constant(f"{name}/bias", vec(m.bias))], name=name)

    def conv(m, x: str, name: str, act: str = "none") -> str:
        _expect(_cls(m) == "Conv2d" and m.groups == 1 and _pair(m.dilation) == (1, 1) and _pair(m.stride) == (1, 1)
                and m.padding_mode == "zeros", f"{name}: unexpected Conv2d")
        ph, pw = _pair(m.padding)
        kh, kw = _pair(m.kernel_size)
        _expect((ph, pw) == (kh // 2, kw // 2), f"{name}: not a 'same' conv")
        visited.add(name)
        b = g.add_constant(f"{name}/bias", vec(m.bias)) if m.bias is not None else ""
        precision.append({"layer": name, "op": "Conv2D", "source": f"Conv2d {kh}x{kw}", "weights": "fp32", "activations": "fp32"})
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", _to_ohwi(m.weight.detach().cpu().numpy().astype(np.float32)), layout="OHWI"), b],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                           "activation": act}, name=name)

    def lrelu(m, x: str, name: str) -> str:
        _expect(_cls(m) == "LeakyReLU", f"{name}: unexpected {_cls(m)}")
        visited.add(name)
        return g.add_node("Activation", [x], {"type": "leaky_relu", "alpha": float(m.negative_slope)}, name=name)

    def rprelu(m, x: str, name: str, c: int) -> str:
        _expect(_cls(m) == "RPReLU", f"{name}: unexpected {_cls(m)}")
        t = bias_node(m.pr_bias0, x, f"{name}.pr_bias0", c)
        pr = m.pr_prelu
        _expect(_cls(pr) == "PReLU", f"{name}.pr_prelu: unexpected {_cls(pr)}")
        slope = vec(pr.weight)
        _expect(slope.size in (1, c), f"{name}.pr_prelu: {slope.size} slopes for {c} channels")
        visited.add(f"{name}.pr_prelu")
        t = g.add_node("Activation", [t, g.add_constant(f"{name}.pr_prelu/slope", np.broadcast_to(slope, (c,)).copy())],
                       {"type": "prelu"}, name=f"{name}.pr_prelu")
        return bias_node(m.pr_bias1, t, f"{name}.pr_bias1", c)

    def binary_core(m, x: str, name: str) -> tuple[str, int]:
        """move0 -> sign -> HardBinaryConv -> RPReLU; returns (output, out channels)."""
        bc = m.binary_conv
        cin, cout = int(bc.weight.shape[1]), int(bc.weight.shape[0])
        _expect(_cls(m.binary_activation) == "BinaryActivation", f"{name}.binary_activation: unexpected")
        visited.add(f"{name}.binary_activation")
        t = bias_node(m.move0, x, f"{name}.move0", cin)
        sign, post_mul, d = _binary_bits(bc, f"{name}.binary_conv")
        binary_weights[f"{name}.binary_conv"] = sign
        diag.append(d)
        visited.add(f"{name}.binary_conv")
        pa = g.add_constant(f"{name}.binary_conv/post_add", vec(bc.bias)) if bc.bias is not None else ""
        t = g.add_node("BConv2D", [t, g.add_constant(f"{name}.binary_conv/kernel_bits", _to_ohwi(sign), dtype="bit", layout="OHWI_bit"),
                                   folded(f"{name}.binary_conv/post_mul", post_mul), pa, ""],
                       {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
                        "input_binarize": binarize, "activation": "none", "output": "f32"}, name=f"{name}.binary_conv")
        precision.append({"layer": f"{name}.binary_conv", "op": "BConv2D", "source": "LearnableBias -> BinaryActivation + HardBinaryConv",
                          "weights": "binary sign(W), per-channel mean|W|" + (" + fp bias" if bc.bias is not None else ""),
                          "activations": "binary (sign of x + move0)", "input_binarize": binarize,
                          "source_binarize": "torch.sign (sign(0)=0)", "pad_value": 0.0, "kernel": [3, 3]})
        return rprelu(m.relu, t, f"{name}.relu", cout), cout

    def binary_conv2d(m, x: str, name: str) -> str:
        _expect(_cls(m) == "BinaryConv2d", f"{name}: unexpected {_cls(m)}")
        visited.add(name)
        t, cout = binary_core(m, x, name)
        _expect(cout == int(m.binary_conv.weight.shape[1]), f"{name}: residual needs cin == cout")
        return g.add_node("Add", [t, x], name=f"{name}.add")

    c0 = in_ch or int((model.model[0] if arch == "dncnn" else model.conv_first).weight.shape[1])
    x = g.add_input("input", [1, h, w, c0])
    xk = g.add_node("ChannelAffine", [x, g.add_constant("k/scale", np.full(c0, k, np.float32)), ""], name="x_times_k")
    unused: list[str] = []
    out_c = c0

    if arch == "dncnn":
        seq = model.model
        _expect(_cls(seq) == "Sequential", "model.model: expected Sequential")
        mods = list(seq.named_children())
        _expect(len(mods) == 21 and _cls(mods[0][1]) == "Conv2d" and _cls(mods[1][1]) == "ReLU"
                and all(_cls(m) == "BinaryConv2d" for _, m in mods[2:20]) and _cls(mods[20][1]) == "Conv2d",
                f"unexpected DnCNN layout {[_cls(m) for _, m in mods]}")
        t = conv(mods[0][1], xk, "model.0", act="relu")
        visited.add("model.1")
        for nm, m in mods[2:20]:
            t = binary_conv2d(m, t, f"model.{nm}")
        n = conv(mods[20][1], t, "model.20")
        out_c = int(mods[20][1].weight.shape[0])
        _expect(out_c == c0, "DnCNN residual needs in_nc == out_nc")
        neg = g.add_node("ChannelAffine", [n, g.add_constant("neg/scale", np.full(out_c, -1.0, np.float32)), ""], name="neg_n")
        t = g.add_node("Add", [xk, neg], name="x_minus_n")
    else:
        up = int(model.upscale)
        _expect(up in (2, 4), f"upscale {up} not supported (2, 4)")
        t = lrelu(model.lrelu, conv(model.conv_first, xk, "conv_first"), "lrelu.conv_first")
        body = model.body
        _expect(_cls(body) == "Sequential", "body: expected Sequential")
        visited.add("body")
        for bi, blk in enumerate(body):
            _expect(_cls(blk) == "BinaryBlock" and _cls(blk.body) == "Sequential" and len(blk.body) == 2, f"body.{bi}: unexpected block")
            for ci, cm in enumerate(blk.body):
                t = binary_conv2d(cm, t, f"body.{bi}.body.{ci}")
        nf = int(model.conv_first.weight.shape[0])
        stages = ["upconv1"] + (["upconv2"] if up == 4 else [])
        r = 2 if up == 4 else up
        ch, cw = h, w
        for st in stages:
            m = getattr(model, st)
            if arch == "bbcul":
                _expect(_cls(model.pixel_shuffle) == "PixelShuffle" and int(model.pixel_shuffle.upscale_factor) == r, "pixel_shuffle: unexpected")
                visited.add("pixel_shuffle")
                u = conv(m, t, st)
                u = rc.pixel_shuffle(g, u, ch, cw, int(m.weight.shape[0]), r, f"{st}.pixel_shuffle")
                t = lrelu(model.lrelu, u, f"lrelu.{st}")
            else:
                # BinaryUpConv2d(upscale=u): PixelShuffle(u) and nearest x u for u in (2, 3), PixelShuffle(2) / nearest x 2
                # for u == 4 (quant_convs_bbcu.py:115-131); BBCUM builds upscale=2 modules for x4
                _expect(_cls(m) == "BinaryUpConv2d" and (int(m.upscale) if int(m.upscale) in (2, 3) else 2) == r
                        and _cls(m.pixel_shuffle) == "PixelShuffle"
                        and int(m.pixel_shuffle.upscale_factor) == r, f"{st}: unexpected {_cls(m)}")
                visited.update({st, f"{st}.pixel_shuffle"})
                u, cout = binary_core(m, t, st)
                u = rc.pixel_shuffle(g, u, ch, cw, cout, r, f"{st}.pixel_shuffle")
                nn_up = rc.nearest_upsample(g, t, ch, cw, nf, r, f"{st}.nearest")
                t = g.add_node("Add", [u, nn_up], name=f"{st}.add")
                precision.append({"layer": f"{st}.nearest", "op": "Reshape+Gather+Reshape", "source": "F.interpolate(nearest)",
                                  "weights": "-", "activations": "fp32 (copy)"})
            ch, cw = ch * r, cw * r
        if arch == "bbcul":
            t = lrelu(model.lrelu, conv(model.conv_hr, t, "conv_hr"), "lrelu.conv_hr")
        else:
            t = lrelu(model.lrelu, binary_conv2d(model.conv_hr, t, "conv_hr"), "lrelu.conv_hr")
        t = conv(model.conv_last, t, "conv_last")
        out_c = int(model.conv_last.weight.shape[0])
        _expect(out_c == c0, "global bilinear skip needs num_in_ch == num_out_ch")
        base = rc.bilinear_upsample(g, xk, h, w, c0, up, "base_bilinear")
        precision.append({"layer": "base_bilinear", "op": "Gather+Conv2D+Reshape/Transpose", "source": "F.interpolate(bilinear, align_corners=False)",
                          "weights": "fixed dyadic phase weights", "activations": "fp32"})
        t = g.add_node("Add", [t, base], name="out_plus_base")
        h, w = ch, cw
    y = g.add_node("ChannelAffine", [t, folded("inv_k/scale", np.full(out_c, 1.0 / k)), ""], name="div_k")
    g.set_outputs(y)

    known = {"DnCNN", "BBCUL", "BBCUM", "Sequential", "Conv2d", "ReLU", "LeakyReLU", "BinaryConv2d", "BinaryUpConv2d", "BinaryBlock",
             "LearnableBias", "BinaryActivation", "HardBinaryConv", "RPReLU", "PReLU", "PixelShuffle"}
    for n_, m in model.named_modules(remove_duplicate=False):
        _expect(_cls(m) in known, f"module {n_ or '<root>'}: unsupported type {_cls(m)}")
    for n_, _ in list(model.named_parameters()) + list(model.named_buffers()):
        ok = any(n_ == v or n_.startswith(v + ".") for v in visited)
        _expect(ok, f"parameter/buffer {n_} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": unused, "float64_constants": exact, "binary_weights": binary_weights,
            "output_shape": [1, h, w, out_c], "k": k}
    return g, info


__all__ = ["convert", "ARCHS", "_f64"]
