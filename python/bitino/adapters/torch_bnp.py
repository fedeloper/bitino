"""Adapter: an eval-mode BNP model (research/bnp/model.py; the research notes (not published)) -> Bitino IR.

The graph has the op pattern of the latency generator (tools/hybrid_family.py, block "bnp"), so the int8_full v2 file of
a trained model has the layers that were timed:
  * stem: Conv2D (BN folded in float64) [+ MaxPool2D 3x3 / 2];
  * binarization: ChannelAffine(x, '', bias) of LearnableThreshold (sign(x + bias), "ge": 0 -> +1) or InstanceThreshold
    (INSTA-BNN, inputs t0 / a / b);
  * BinaryConv2d -> BConv2D with post_mul = alpha * BN scale, post_add = BN shift (float64, rounded once), pad_value 0;
    the SE FCs' BN and ReLU go into their BConv2D (activation relu);
  * UnitPReLU -> Activation prelu with slopes sigmoid(theta);
  * recoupling mix s * x + (1 - s) * t -> ChannelAffine(x, s) + ChannelAffine(t, 1 - s) -> Add; GlobalAvgPool keepdims;
    the gate -> ChannelGate(t, z, sigmoid); ELM product -> Mul; BN2 -> ChannelAffine; residual -> Add;
  * int8 downsampling: AvgPool2D 2x2 -> Conv2D 1x1 (BN folded);
  * head: PReLU, GlobalAvgPool, Dense.
Every module and parameter must be covered (raises otherwise). Layouts: NCHW -> NHWC, OIHW -> OHWI."""
from __future__ import annotations

import numpy as np

from ..ir import Graph


def _np64(t):
    return t.detach().cpu().double().numpy()


def _ohwi(w):
    return np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1)))


def _bn(m):
    var, mean = _np64(m.running_var), _np64(m.running_mean)
    s = _np64(m.weight) / np.sqrt(var + float(m.eps))
    return s, _np64(m.bias) - mean * s


def convert(model, model_id: str = "bnp", input_name: str = "input") -> tuple[Graph, dict]:
    import torch
    assert type(model).__name__ == "BNP", type(model).__name__
    assert not model.training, "model must be in eval mode"
    g = Graph()
    covered: set[str] = set()
    f64: dict[str, np.ndarray] = {}

    def c32(name, v64, dtype="f32", layout=""):
        v64 = np.asarray(v64)
        if dtype == "f32":
            f64[name] = np.asarray(v64, np.float64)
            v64 = v64.astype(np.float32)
        return g.add_constant(name, v64, dtype, layout)

    def mark(p):
        covered.add(p)

    def thr(m, x, p):
        mark(p)
        if type(m).__name__ == "InstaThreshold":
            return g.add_node("InstanceThreshold", [x, c32(f"{p}/t0", _np64(m.t0)), c32(f"{p}/a", _np64(m.a)), c32(f"{p}/b", _np64(m.b))], name=p)
        assert type(m).__name__ == "LearnableThreshold", p
        return g.add_node("ChannelAffine", [x, "", c32(f"{p}/bias", _np64(m.bias))], name=p)

    def bconv(m, bn, x, p, act="none"):
        """BinaryConv2d (sign weights, alpha) + BatchNorm folded into post_mul / post_add"""
        w = m.weight.detach().cpu()
        alpha = w.abs().mean(dim=(1, 2, 3)).double().numpy()
        wb = torch.where(w >= 0, 1, -1).numpy().astype(np.int8)
        s, b = _bn(bn)
        k = m.kernel_size[0]
        st = m.stride[0]
        pd = k // 2
        mark(p)
        mark(p.rsplit(".", 1)[0] + "." + [n for n, mm in model.named_modules() if mm is bn][0].rsplit(".", 1)[1])
        return g.add_node("BConv2D", [x, c32(f"{p}/kernel_bits", _ohwi(wb), "bit", "OHWI_bit"), c32(f"{p}/post_mul", alpha * s),
                                      c32(f"{p}/post_add", b), ""],
                          {"strides": [st, st], "dilations": [1, 1], "pads": [pd, pd, pd, pd], "groups": 1, "pad_value": 0.0,
                           "input_binarize": "ge", "activation": act, "output": "f32"}, name=p)

    def conv_bn(m, bn, x, p):
        w = _np64(m.weight)
        s, b = _bn(bn)
        k, st, pd = m.kernel_size[0], m.stride[0], m.padding[0]
        mark(p)
        mark(p.replace("conv", "bn") if "conv" in p else p + "_bn")
        return g.add_node("Conv2D", [x, c32(f"{p}/kernel", _ohwi(w * s[:, None, None, None]), layout="OHWI"), c32(f"{p}/bias", b)],
                          {"strides": [st, st], "dilations": [1, 1], "pads": [pd, pd, pd, pd], "groups": 1, "pad_value": 0.0,
                           "activation": "none"}, name=p)

    def prelu(m, x, p):
        mark(p)
        return g.add_node("Activation", [x, c32(f"{p}/slope", _np64(m.slope()))], {"type": "prelu"}, name=p)

    def affine(x, p, scale=None, shift=None):
        return g.add_node("ChannelAffine", [x, c32(f"{p}/scale", scale) if scale is not None else "",
                                            c32(f"{p}/shift", shift) if shift is not None else ""], name=p)

    def avgpool(x, p):
        return g.add_node("AvgPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1}, name=p)

    def unit(u, x, p):
        if getattr(u, "experts", 1) > 1:
            raise NotImplementedError(f"{p}: expert binary convs (EBN) are not converted yet (no float expert op / quantizer lowering; "
                                      "the runtime op I8BConv2DResIX exists)")
        if u.down:
            r = avgpool(x, f"{p}.pool") if u.stride == 2 else x
            w = _np64(u.down_conv.weight)
            s, b = _bn(u.down_bn)
            mark(f"{p}.down_conv")
            mark(f"{p}.down_bn")
            mark(f"{p}.pool")
            r = g.add_node("Conv2D", [r, c32(f"{p}.down/kernel", _ohwi(w * s[:, None, None, None]), layout="OHWI"), c32(f"{p}.down/bias", b)],
                           {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0,
                            "activation": "none"}, name=f"{p}.down")
        else:
            r = prelu(u.act1, x, f"{p}.act1") if u.elm else x
            if u.stride == 2:
                r = avgpool(r, f"{p}.pool_r")
        t = bconv(u.conv, u.bn1, thr(u.thr, x, f"{p}.thr"), f"{p}.conv")
        t = prelu(u.act2, t, f"{p}.act2")
        if u.se is not None:
            if u.recouple:
                mi = avgpool(x, f"{p}.pool_i") if u.stride == 2 else x
                sc = _np64(u.scale)
                mark(f"{p}.scale")
                src = g.add_node("Add", [affine(mi, f"{p}.mix_in", scale=sc), affine(t, f"{p}.mix_x", scale=1.0 - sc)], name=f"{p}.mix")
            else:
                src = t
            se = u.se
            mark(f"{p}.se")
            z = g.add_node("GlobalAvgPool", [src], {"keepdims": 1}, name=f"{p}.se.gap")
            z = bconv(se.fc1, se.bn1, thr(se.thr1, z, f"{p}.se.thr1"), f"{p}.se.fc1", act="relu")
            z = bconv(se.fc2, se.bn2, thr(se.thr2, z, f"{p}.se.thr2"), f"{p}.se.fc2")
            t = g.add_node("ChannelGate", [t, z], {"gate": "sigmoid"}, name=f"{p}.se.gate")
        if u.elm or u.se is not None:
            if u.elm:
                t = g.add_node("Mul", [t, r], name=f"{p}.elm")
            s, b = _bn(u.bn2)
            mark(f"{p}.bn2")
            t = affine(t, f"{p}.bn2", s, b)
        return g.add_node("Add", [t, r], name=f"{p}.add")

    def basic(u, x, p):
        """BasicUnit: thr -> BConv3x3 (BN folded) [ELM: * PReLU(x), BN2] -> + r -> move1 -> PReLU -> move2 (the generator's
        basic unit with glue rprelu; tools/hybrid_family.py Builder.unit / glue_out)"""
        if u.down:
            r = avgpool(x, f"{p}.pool") if u.stride == 2 else x
            w = _np64(u.down_conv.weight)
            s, b = _bn(u.down_bn)
            for m in ("down_conv", "down_bn", "pool"):
                mark(f"{p}.{m}")
            r = g.add_node("Conv2D", [r, c32(f"{p}.down/kernel", _ohwi(w * s[:, None, None, None]), layout="OHWI"), c32(f"{p}.down/bias", b)],
                           {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0,
                            "activation": "none"}, name=f"{p}.down")
        else:
            r = x
        t = bconv(u.conv, u.bn1, thr(u.thr, x, f"{p}.thr"), f"{p}.conv")
        if getattr(u, "elmb", False):   # BNext-form ELM: PReLU2, product with PReLU1(x), BN2, + r; no RPReLU
            t = prelu(u.act2, t, f"{p}.act2")
            gt = prelu(u.act1, x, f"{p}.act1")
            t = g.add_node("Mul", [t, gt], name=f"{p}.elm")
            s, b = _bn(u.bn2)
            mark(f"{p}.bn2")
            return g.add_node("Add", [affine(t, f"{p}.bn2", s, b), r], name=f"{p}.add")
        if u.elm:
            gt = prelu(u.act1, x, f"{p}.act1")
            t = g.add_node("Mul", [t, gt], name=f"{p}.elm")
            s, b = _bn(u.bn2)
            mark(f"{p}.bn2")
            t = affine(t, f"{p}.bn2", s, b)
        y = g.add_node("Add", [t, r], name=f"{p}.add")
        mark(f"{p}.move1")
        y = g.add_node("ChannelAffine", [y, "", c32(f"{p}.move1/bias", _np64(u.move1.bias))], name=f"{p}.move1")
        y = prelu(u.act, y, f"{p}.act")
        mark(f"{p}.move2")
        return g.add_node("ChannelAffine", [y, "", c32(f"{p}.move2/bias", _np64(u.move2.bias))], name=f"{p}.move2")

    res = int(model.spec.get("res", 224))
    x = g.add_input(input_name, [1, res, res, 3])
    w = _np64(model.stem.weight)
    s, b = _bn(model.stem_bn)
    k, st, pd = model.stem.kernel_size[0], model.stem.stride[0], model.stem.padding[0]
    mark("stem")
    mark("stem_bn")
    x = g.add_node("Conv2D", [x, c32("stem/kernel", _ohwi(w * s[:, None, None, None]), layout="OHWI"), c32("stem/bias", b)],
                   {"strides": [st, st], "dilations": [1, 1], "pads": [pd, pd, pd, pd], "groups": 1, "pad_value": 0.0,
                    "activation": "none"}, name="stem")
    if model.stem_type == "c7pool":
        x = g.add_node("MaxPool2D", [x], {"kernel": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]}, name="stem.pool")
    for i, blk in enumerate(model.blocks):
        if type(blk).__name__ == "BasicBlock":
            x = basic(blk.u0, x, f"blocks.{i}.u0")
            x = basic(blk.u1, x, f"blocks.{i}.u1")
            continue
        x = unit(blk.A, x, f"blocks.{i}.A")
        if blk.F is not None:
            x = unit(blk.F, x, f"blocks.{i}.F")
    if model.head_act is not None:
        x = prelu(model.head_act, x, "head_act")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="pool")
    mark("fc")
    x = g.add_node("Dense", [x, c32("fc/kernel", _np64(model.fc.weight)), c32("fc/bias", _np64(model.fc.bias))], {"activation": "none"}, name="fc")
    g.set_outputs(x)
    # coverage: every parameter / buffer belongs to a converted module
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        mod = n.rsplit(".", 1)[0]
        ok = any(n == c or n.startswith(c + ".") or mod == c for c in covered)
        if not ok:
            raise NotImplementedError(f"parameter / buffer {n} not covered by the BNP adapter")
    return g, {"model_id": model_id, "op_counts": g.op_counts(), "float64_constants": f64}
