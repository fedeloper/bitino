"""Adapter: official AdaBin CIFAR-10 ResNet-20 (PyTorch, huawei-noah/Efficient-Computing) -> Bitino IR.

Source (BinaryNetworks/AdaBin/cifar10, read only):
  * utils/binarylib.py
      - BinaryActivation.forward (l.64-67): ``x = (x - beta_a)/alpha_a; x = gradient_approx(x);
        return alpha_a*(x + beta_a)``; gradient_approx (l.48-62) returns ``sign(x).detach() - out3.detach() + out3``,
        which is torch.sign(x) exactly in float32/float64 (|out3| <= 1 with the sign of x; checked by a hook).
        With alpha_a > 0 (asserted) the binarized activation is ``a = alpha_a*(s + beta_a)``,
        ``s = sign(x - beta_a)`` in {-1, 0, +1}: the adaptive binary set {alpha(beta-1), alpha(beta+1)}.
      - AdaBin_Conv2d.forward (l.94-111): per output channel ``beta_w = mean(w)``,
        ``alpha_w = sqrt(sum((w - beta_w)^2)/filter_size)``, ``w_b = sign((w - beta_w)/alpha_w)*alpha_w + beta_w``,
        then ``F.conv2d(a, w_b, padding=1)`` with ZERO padding of the binarized input (0 is not in the set).
      - Maxout.forward (l.38-41): ``pos_scale*relu(x) - neg_scale*relu(-x)`` (per channel).
  * nets/resnet20.py BasicBlock_1w1a.forward (l.54-64): ``out = maxout1(bn1(conv1(x)) + shortcut(x));
    out = maxout2(bn2(conv2(out)) + out)``; option-A shortcut ``F.pad(x[:, :, ::2, ::2], channels planes//4)``.
    ResNet.forward (l.88-97): ``maxout(bn1(conv1(x)))`` (FP stem), 3 stages, ``F.avg_pool2d(out, 8)``,
    BatchNorm1d, Linear.

Mapping of one AdaBin conv + its BatchNorm (existing ops only, exact decomposition):
  With A+ / A- the two activation values, p = (A+ + A-)/2, q = (A+ - A-)/2 (so a = p + q*s), and per
  output channel W+ / W- the two weight values, c_o = (W+ + W-)/2, d_o = (W+ - W-)/2 (w_b = c_o + d_o*s_w):

      conv[o] = sum_{in-bounds taps} (c_o + d_o s_w)(p + q s)
              = q d_o * sum s_w s  +  q c_o * sum_inb s  +  p (d_o * sum_inb s_w + c_o * K_inb)

  * ``sum s_w s``: BConv2D (sign bits of w, threshold beta_a, pad_value 0: out-of-bounds taps add 0),
  * ``sum_inb s``: BConv2D with ONE output channel and all-(+1) weights (same threshold / padding),
    broadcast to the O channels by a 1x1 Conv2D with weights q c_o,
  * the last term depends only on the position (border taps are missing) and the channel: a constant
    map ``M[oh, ow, o]`` computed from the padding mask (float64 conv of an all-ones input), added by Add.
  BatchNorm (eval) is folded in: post_mul = g q d, 1x1 weights g q c, map g*M + h.
  ``s = 0`` (x == beta_a exactly) gives a = alpha*beta = p exactly in real arithmetic, so the
  ``sign_zero`` variant (s in {-1,0,+1}) is the exact source semantics; ``ge`` maps it to +1.
Maxout -> relu / ChannelAffine(pos) and ChannelAffine(-1) / relu / ChannelAffine(-neg), Add: the same IEEE
operations as the source (exactly one of the two products is nonzero), so Maxout is bit-exact.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

BINARIZE_MODES = ("sign_zero", "ge")


def _cls(m) -> str:
    return type(m).__name__


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)


def _pair(v) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _f64(t) -> np.ndarray:
    import torch
    return t.detach().cpu().to(torch.float64).reshape(-1).numpy()


def adabin_weight_levels(m):
    """(sign bits int8 [O,I,kh,kw], W+ [O], W- [O]) exactly as AdaBin_Conv2d.forward computes w_b in the
    dtype of the module (float32 for the converted model): w_b = sign(wn)*alpha_w + beta_w."""
    import torch

    with torch.no_grad():
        w = m.weight.detach()
        beta_w = w.mean((1, 2, 3)).view(-1, 1, 1, 1)
        alpha_w = torch.sqrt(((w - beta_w) ** 2).sum((1, 2, 3)) / m.filter_size).view(-1, 1, 1, 1)
        wn = (w - beta_w) / alpha_w
        sign = torch.sign(wn)
        n_zero = int((sign == 0).sum())
        _expect(n_zero == 0, f"{n_zero} normalized weights are exactly 0 (w_b = beta_w is a third level)")
        wp = (alpha_w + beta_w).reshape(-1)          # sign * alpha_w is exact; then + beta_w, as the source
        wm = (-alpha_w + beta_w).reshape(-1)
        wb = sign * alpha_w + beta_w
        assert torch.equal(torch.where(sign > 0, wp.view(-1, 1, 1, 1), wm.view(-1, 1, 1, 1)), wb)
    return sign.numpy().astype(np.int8), wp.numpy(), wm.numpy()


def border_maps(sign_oihw: np.ndarray, hw: tuple[int, int], stride: int, pad: int):
    """(sum over in-bounds taps of s_w [OH,OW,O], number of in-bounds taps x C [OH,OW]) via float64 convs
    of an all-ones input with zero padding."""
    import torch
    import torch.nn.functional as F

    o, c = sign_oihw.shape[:2]
    ones = torch.ones(1, c, hw[0], hw[1], dtype=torch.float64)
    sw = F.conv2d(ones, torch.from_numpy(sign_oihw.astype(np.float64)), stride=stride, padding=pad)[0].permute(1, 2, 0).numpy()
    k = F.conv2d(ones, torch.ones(1, c, *sign_oihw.shape[2:], dtype=torch.float64), stride=stride, padding=pad)[0, 0].numpy()
    return sw, k


def convert(model, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (32, 32),
            binarize: str = "sign_zero") -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    _expect(not model.training, "model must be in eval mode")
    _expect(_cls(model) == "ResNet" and hasattr(model, "layer3") and not hasattr(model, "layer4") and _cls(model.nonlinear1) == "Maxout",
            f"unsupported top-level model {_cls(model)} (expected AdaBin CIFAR ResNet-20)")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    visited: set[str] = set()
    binary_weights: dict[str, np.ndarray] = {}
    weight_levels: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def folded(name: str, value64, dtype_shape=None) -> str:
        v = np.asarray(value64, np.float64)
        exact[name] = v
        return g.add_constant(name, v.astype(np.float32))

    def bn_params(m, name: str, cls: str = "BatchNorm2d"):
        _expect(_cls(m) == cls and m.affine and m.running_mean is not None, f"{name}: expected {cls} with running stats")
        scale = _f64(m.weight) / np.sqrt(_f64(m.running_var) + float(m.eps))
        shift = _f64(m.bias) - _f64(m.running_mean) * scale
        visited.add(name)
        return scale, shift

    def bn(m, x: str, name: str, cls: str = "BatchNorm2d") -> str:
        scale, shift = bn_params(m, name, cls)
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", shift)], name=name)

    def maxout(m, x: str, name: str, c: int) -> str:
        _expect(_cls(m) == "Maxout" and tuple(m.pos_scale.shape) == (c,) and tuple(m.neg_scale.shape) == (c,), f"{name}: expected Maxout({c})")
        pos = m.pos_scale.detach().cpu().numpy().astype(np.float32)
        neg = m.neg_scale.detach().cpu().numpy().astype(np.float32)
        visited.add(name)
        r1 = g.add_node("Activation", [x], {"type": "relu"}, name=f"{name}/relu_pos")
        a = g.add_node("ChannelAffine", [r1, g.add_constant(f"{name}/pos_scale", pos), ""], name=f"{name}/pos")
        t = g.add_node("ChannelAffine", [x, g.add_constant(f"{name}/minus_one", -np.ones(c, np.float32)), ""], name=f"{name}/negate")
        r2 = g.add_node("Activation", [t], {"type": "relu"}, name=f"{name}/relu_neg")
        b = g.add_node("ChannelAffine", [r2, g.add_constant(f"{name}/minus_neg_scale", -neg), ""], name=f"{name}/neg")
        precision.append({"layer": name, "op": "Activation(relu)x2+ChannelAffine x3+Add", "source": "Maxout pos*relu(x) - neg*relu(-x)",
                          "weights": "fp32 per-channel scales", "activations": "fp32 (bit-exact with the source)"})
        return g.add_node("Add", [a, b], name=name)

    def conv(m, x: str, name: str) -> str:
        _expect(_cls(m) == "Conv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros", f"{name}: unexpected Conv2d")
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        visited.add(name)
        w = np.ascontiguousarray(m.weight.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1))
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", w, layout="OHWI"), ""],
                          {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                           "activation": "none"}, name=name)

    def adabin_conv_bn(m, bnm, x: str, name: str, bn_name: str, hw: tuple[int, int], cin: int) -> str:
        _expect(_cls(m) == "AdaBin_Conv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros" and m.a_bit == 1 and m.w_bit == 1, f"{name}: unexpected AdaBin_Conv2d settings")
        _expect(_cls(m.binary_a) == "BinaryActivation", f"{name}: unexpected activation quantizer")
        (sh, sw_), (ph, pw) = _pair(m.stride), _pair(m.padding)
        _expect(sh == sw_ and ph == pw, f"{name}: anisotropic stride/padding")
        kh, kw = _pair(m.kernel_size)
        o = m.out_channels
        alpha = float(m.binary_a.alpha_a.detach())
        beta = float(m.binary_a.beta_a.detach())
        _expect(alpha > 0, f"{name}: alpha_a = {alpha} <= 0 (the binarization would flip sign)")
        a_plus, a_minus = alpha * (beta + 1.0), alpha * (beta - 1.0)       # float64 from the fp32 parameters
        p, q = (a_plus + a_minus) / 2.0, (a_plus - a_minus) / 2.0
        sign, wp, wm = adabin_weight_levels(m)
        binary_weights[name] = sign
        weight_levels[name] = (wp, wm)
        wp64, wm64 = wp.astype(np.float64), wm.astype(np.float64)
        c_o, d_o = (wp64 + wm64) / 2.0, (wp64 - wm64) / 2.0
        gsc, gsh = bn_params(bnm, bn_name)
        ho, wo = (hw[0] + 2 * ph - kh) // sh + 1, (hw[1] + 2 * pw - kw) // sh + 1
        sw_map, k_map = border_maps(sign, hw, sh, ph)
        assert sw_map.shape == (ho, wo, o) and k_map.shape == (ho, wo)
        const_map = gsc * p * (d_o * sw_map + c_o * k_map[..., None]) + gsh
        attrs = {"strides": [sh, sh], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                 "input_binarize": binarize, "activation": "none", "output": "f32"}
        thr = g.add_constant(f"{name}/threshold", np.full(cin, np.float32(m.binary_a.beta_a.detach().numpy())))
        visited.add(name)
        # sum over in-bounds taps and channels of s (one output channel, all weights +1)
        cnt = g.add_node("BConv2D", [x, g.add_constant(f"{name}/ones_bits", np.ones((1, kh, kw, cin), np.int8), dtype="bit", layout="OHWI_bit"),
                                     "", "", thr], attrs, name=f"{name}/sum_s")
        bsum = g.add_node("Conv2D", [cnt, folded(f"{name}/sum_s_scale", (gsc * q * c_o).reshape(o, 1, 1, 1)), ""],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0, "activation": "none"},
                          name=f"{name}/sum_s_broadcast")
        main = g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", np.ascontiguousarray(sign.transpose(0, 2, 3, 1)), dtype="bit",
                                                        layout="OHWI_bit"),
                                      folded(f"{name}/post_mul", gsc * q * d_o), "", thr], attrs, name=name)
        y = g.add_node("Add", [main, bsum], name=f"{name}/add_sum_s")
        y = g.add_node("Add", [y, folded(f"{name}/border_map", const_map.reshape(1, ho, wo, o))], name=f"{bn_name}")
        precision.append({"layer": name, "op": "BConv2D(O) + BConv2D(1)+Conv2D(1x1) + Add(const map)", "source": "AdaBin_Conv2d + BatchNorm2d",
                          "weights": "binary sign((w-mean)/std), per-channel levels W+/W- (c_o + d_o*s)",
                          "activations": "binary {alpha(beta-1), alpha(beta+1)} (p + q*s)", "input_binarize": binarize,
                          "source_binarize": "torch.sign((x - beta_a)/alpha_a) (sign(0)=0), alpha_a > 0", "threshold": beta,
                          "pad_value": 0.0, "padding_correction": "constant border map (zero padding is not in the binary set)",
                          "kernel": [kh, kw], "stride": sh})
        diag.append({"layer": name, "alpha_a": alpha, "beta_a": beta, "p": p, "q": q, "n_weights": int(sign.size),
                     "n_plus": int(np.sum(sign > 0)), "c_over_d_absmax": float(np.max(np.abs(c_o / d_o)))})
        return y

    def shortcut(blk, x: str, pth: str, cin: int, cout: int, stride: int, h: int, w: int) -> str:
        sc = blk.shortcut
        if _cls(sc) == "Sequential" and len(sc) == 0:
            _expect(stride == 1 and cin == cout, f"{pth}: identity shortcut with a shape change")
            return x
        _expect(_cls(sc) == "LambdaLayer", f"{pth}.shortcut: only option A is supported, got {_cls(sc)}")
        import torch
        pad = (cout - cin) // 2
        probe = torch.randn(1, cin, h, w, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            got = sc(probe)
            want = torch.nn.functional.pad(probe[:, :, ::2, ::2], (0, 0, 0, 0, pad, pad))
        _expect(tuple(got.shape) == tuple(want.shape) and bool(torch.equal(got, want)), f"{pth}.shortcut does not match option A")
        r = g.add_node("AvgPool2D", [x], {"kernel": [1, 1], "strides": [2, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1},
                       name=f"{pth}.shortcut.subsample")
        if pad:
            r = g.add_node("Pad", [r], {"pads": [0, 0, 0, 0, 0, 0, pad, pad], "value": 0.0}, name=f"{pth}.shortcut.pad")
        precision.append({"layer": f"{pth}.shortcut", "op": "AvgPool2D+Pad", "weights": "none", "activations": "fp32",
                          "source": "x[:, :, ::2, ::2] + F.pad channels"})
        return r

    hw = _pair(input_hw)
    x = g.add_input(input_name, [1, hw[0], hw[1], 3])
    x = conv(model.conv1, x, "conv1")
    x = bn(model.bn1, x, "bn1")
    c = model.conv1.out_channels
    x = maxout(model.nonlinear1, x, "nonlinear1", c)
    precision.append({"layer": "conv1", "op": "Conv2D+ChannelAffine", "source": "Conv2d 3x3 + BN (+ Maxout)", "weights": "fp32", "activations": "fp32"})
    h, w_ = hw
    for lname in ("layer1", "layer2", "layer3"):
        for bi, blk in enumerate(getattr(model, lname)):
            pth = f"{lname}.{bi}"
            _expect(_cls(blk) == "BasicBlock_1w1a", f"{pth}: unexpected block {_cls(blk)}")
            cout, st = blk.conv1.out_channels, _pair(blk.conv1.stride)[0]
            ho, wo = (h - 1) // st + 1, (w_ - 1) // st + 1
            u = adabin_conv_bn(blk.conv1, blk.bn1, x, f"{pth}.conv1", f"{pth}.bn1", (h, w_), c)
            u = g.add_node("Add", [u, shortcut(blk, x, pth, c, cout, st, h, w_)], name=f"{pth}.add1")
            u = maxout(blk.nonlinear1, u, f"{pth}.nonlinear1", cout)
            v = adabin_conv_bn(blk.conv2, blk.bn2, u, f"{pth}.conv2", f"{pth}.bn2", (ho, wo), cout)
            v = g.add_node("Add", [v, u], name=f"{pth}.add2")
            x = maxout(blk.nonlinear2, v, f"{pth}.nonlinear2", cout)
            h, w_, c = ho, wo, cout
    _expect((h, w_) == (8, 8), f"head: F.avg_pool2d(out, out.size()[3]) on a {h}x{w_} map (expected the full 8x8 map)")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="avgpool")
    x = bn(model.bn2, x, "bn2", cls="BatchNorm1d")
    fc = model.linear
    _expect(_cls(fc) == "Linear" and fc.bias is not None, f"linear: unexpected {_cls(fc)}")
    visited.add("linear")
    x = g.add_node("Dense", [x, g.add_constant("linear/kernel", fc.weight.detach().cpu().numpy().astype(np.float32)),
                             g.add_constant("linear/bias", fc.bias.detach().cpu().numpy().astype(np.float32))], {"activation": "none"}, name="linear")
    precision.append({"layer": "head", "op": "GlobalAvgPool+ChannelAffine+Dense", "source": "F.avg_pool2d (full map) + BatchNorm1d + Linear",
                      "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)

    known = {"ResNet", "BasicBlock_1w1a", "Sequential", "LambdaLayer", "Conv2d", "BatchNorm2d", "BatchNorm1d", "AdaBin_Conv2d",
             "BinaryActivation", "Maxout", "ReLU", "Linear"}
    for n, m in model.named_modules():
        _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        _expect(any(n == v or n.startswith(v + ".") for v in visited), f"parameter/buffer {n} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": [], "float64_constants": exact, "binary_weights": binary_weights, "weight_levels": weight_levels}
    return g, info
