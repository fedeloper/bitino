"""Adapter: instantiated official BNext PyTorch model (hpi-xnor/BNext) -> Bitino IR.

Source: https://github.com/hpi-xnor/BNext (commit dfcf347), src/bnext.py. The adapter reads the
*instantiated* nn.Module (official class definitions, weights loaded with
torch.load(weights_only=True)); it mirrors BNext.forward and the forward() of every block
class explicitly and raises on any module type, attribute or setting it does not know.

Semantics captured (src/bnext.py line numbers at dfcf347):
  * Stem firstconv3x3 (43-56): Conv2d 3x3 (stride 2 for 1000 classes, pad 1, no bias) + BN.
    Its PReLU is constructed but NOT used in forward (recorded as unused).
  * LearnableBias (59-66): x + bias -> ChannelAffine(bias only).
  * HardSign (69-86), eval: sign(clamp(x / max(T, 1e-8), -1, 1)) returned through
    (sign - clamp(x, -1.5, 1.5)) + clamp(x, -1.5, 1.5). T > 0 so sign(x / T) = sign(x), and
    fl(fl(s - r) + r) == s exactly for s = sign(r) and |r| <= 1.5 (Sterbenz for |r| >= 0.5,
    |error| <= half an ulp of 1 below that) -> torch.sign(x) with sign(0) = 0 -> BConv2D
    input_binarize "sign_zero" (exact) or "ge" (deployable, 0 -> +1).
  * HardBinaryConv (89-120), eval with temperature < 1e-7: F.conv2d(x, sign(W)), padding
    kernel//2 (the constructor's padding argument is ignored), zero padding of the
    binarized input -> BConv2D pad_value 0, no scaling factor. The adapter checks
    temperature < 1e-7 (otherwise the weights would be clamp(W/T, -1, 1), not binary) and
    that no latent weight is exactly 0 (torch.sign -> 0 weight, not representable).
  * BatchNorm2d (eval, running stats) -> ChannelAffine(scale, shift) folded in float64.
  * nn.PReLU(C) -> Activation prelu (slopes stored verbatim).
  * Info-recoupling (Attention 189-197, FFN_3x3 243-251, FFN_1x1 297-301):
    mix = scale*input + (1-scale)*x -> ChannelAffine(input, scale) + ChannelAffine(x, 1-scale)
    -> Add (fp32 rounding after each product and after the sum, as PyTorch eager). FFN_1x1
    clamps scale to [0, 1] in forward even in eval mode (line 298), the other two only in
    training: mirrored exactly. 1 - scale is folded in float64 and rounded once (equals the
    fp32 1 - scale PyTorch computes, the difference being exact in float64).
  * SqueezeAndExpand (123-145) with attention_mode "sigmoid": AdaptiveAvgPool2d(1) ->
    Conv2d 1x1 + bias -> ReLU -> Conv2d 1x1 + bias -> sigmoid; the gate multiplies x:
    GlobalAvgPool(keepdims) -> Conv2D(relu) -> Conv2D -> ChannelGate(x, gate, gate=sigmoid)
    (existing op, sigmoid in double, rounded once; docs/operators_larq.md).
  * ELM attention (Attention 177-203): residual = PReLU1(input) (AvgPool2d(2) if stride 2),
    x = x * residual (Mul, same shape), BN2, + residual. FFN_3x3/FFN_1x1: residual
    = input (pooled if stride 2), BN2 only when info-recoupling is on, + residual.
  * BasicBlock family (310-426): 3x3 unit (attribute "Attention" or "FFN_3x3") then FFN_1x1,
    or FFN_1 and FFN_2 concatenated on channels when planes = 2 * inplanes.
  * Head (474-482): PReLU -> AdaptiveAvgPool2d(1) -> Linear.

Constants derived from parameters (BN scale/shift, 1 - scale) are computed in float64 and
rounded once to float32; their float64 values are returned in info["float64_constants"]
(analysis only). Layouts: NCHW -> NHWC activations, OIHW -> OHWI filters.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

BINARIZE_MODES = ("sign_zero", "ge")
BLOCK_CLASSES = ("BasicBlock", "BasicBlock_No_ELM_Attention", "BasicBlock_No_Infor_Recoupling", "BasicBlock_No_Extra_Design")
UNIT_CLASSES = ("Attention", "FFN_3x3", "FFN_1x1")


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


def _vec(t) -> np.ndarray:
    return t.detach().cpu().to(dtype=__import__("torch").float64).reshape(-1).numpy()


def convert(model, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (224, 224),
            binarize: str = "sign_zero") -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths (e.g. feature.3.Attention.move),
    so tools can map IR tensors back to module outputs."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    _expect(_cls(model) == "BNext", f"unsupported top-level model class {_cls(model)} (expected BNext)")
    _expect(not model.training, "model must be in eval mode")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    unused: list[str] = []
    exact: dict[str, np.ndarray] = {}
    visited: set[str] = set()

    def folded(name: str, value64: np.ndarray, layout: str = "") -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, np.asarray(value64).astype(np.float32), layout=layout)

    def mark(path: str) -> None:
        visited.add(path)

    def bn(m, x: str, name: str) -> str:
        _expect(_cls(m) == "BatchNorm2d", f"{name}: expected BatchNorm2d, got {_cls(m)}")
        _expect(m.affine and m.track_running_stats and m.running_mean is not None, f"{name}: BN without affine/running stats")
        var, mean = _vec(m.running_var), _vec(m.running_mean)
        scale = _vec(m.weight) / np.sqrt(var + float(m.eps))
        shift = _vec(m.bias) - mean * scale
        mark(name)
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", shift)], name=name)

    def prelu(m, x: str, name: str, c: int) -> str:
        _expect(_cls(m) == "PReLU", f"{name}: expected PReLU, got {_cls(m)}")
        w = m.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
        _expect(w.size in (1, c), f"{name}: PReLU has {w.size} slopes for {c} channels")
        mark(name)
        return g.add_node("Activation", [x, g.add_constant(f"{name}/slope", np.broadcast_to(w, (c,)).copy())],
                          {"type": "prelu"}, name=name)

    def conv1x1_bias(m, x: str, name: str, act: str) -> str:
        _expect(_cls(m) == "Conv2d" and m.kernel_size == (1, 1) and _pair(m.stride) == (1, 1) and m.groups == 1
                and _pair(m.padding) == (0, 0) and m.bias is not None, f"{name}: expected 1x1 Conv2d with bias")
        w = m.weight.detach().cpu().numpy().astype(np.float32)
        b = m.bias.detach().cpu().numpy().astype(np.float32)
        mark(name)
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", _to_ohwi(w), layout="OHWI"), g.add_constant(f"{name}/bias", b)],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0,
                           "activation": act}, name=name)

    def avgpool2(m, x: str, name: str) -> str:
        _expect(_cls(m) == "AvgPool2d" and _pair(m.kernel_size) == (2, 2) and _pair(m.stride) == (2, 2)
                and _pair(m.padding) == (0, 0) and not m.ceil_mode and m.divisor_override is None,
                f"{name}: expected AvgPool2d(2, 2)")
        return g.add_node("AvgPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0],
                                             "count_include_pad": 1 if m.count_include_pad else 0}, name=name)

    def unit(u, x: str, p: str, c: int) -> str:
        cls = _cls(u)
        _expect(cls in UNIT_CLASSES, f"{p}: unsupported unit {cls}")
        stride = int(u.stride)
        _expect(stride in (1, 2), f"{p}: stride {stride}")
        _expect(not (cls == "FFN_1x1" and stride != 1), f"{p}: FFN_1x1 with stride 2 is broken in the source (shape mismatch)")
        _expect(u.inplanes == u.planes == c, f"{p}: expected {c}->{c} unit, got {u.inplanes}->{u.planes}")
        # residual branch
        if cls == "Attention":
            residual = prelu(u.activation1, x, f"{p}.activation1", c)
        else:
            residual = x
            unused.append(f"{p}.activation1 (PReLU constructed, unused in {cls}.forward)")
        if stride == 2:
            residual = avgpool2(u.pooling, residual, f"{p}.pooling.residual")
        # binary branch: move -> HardSign -> HardBinaryConv -> norm1 -> activation2
        _expect(_cls(u.move) == "LearnableBias", f"{p}.move: unexpected {_cls(u.move)}")
        mark(f"{p}.move")
        t = g.add_node("ChannelAffine", [x, "", g.add_constant(f"{p}.move/bias", u.move.bias.detach().cpu().numpy().astype(np.float32).reshape(-1))],
                       name=f"{p}.move")
        ha = u.binary_activation
        _expect(_cls(ha) == "HardSign" and not ha.progressive, f"{p}.binary_activation: expected HardSign(progressive=False)")
        _expect(float(ha.temperature.clamp(min=1e-8)) > 0, f"{p}.binary_activation: temperature must be > 0")
        _expect(list(ha.range) == [-1.5, 1.5], f"{p}.binary_activation: range {ha.range} (exactness proof assumes |r| <= 1.5)")
        mark(f"{p}.binary_activation")
        bc = u.binary_conv
        _expect(_cls(bc) == "HardBinaryConv" and bc.groups == 1, f"{p}.binary_conv: expected HardBinaryConv(groups=1)")
        _expect(bool(bc.temperature < 1e-7), f"{p}.binary_conv: temperature {float(bc.temperature):.3g} >= 1e-7, weights are not sign(W)")
        w = bc.weight.detach().cpu()
        o, i, kh, kw = w.shape
        _expect(kh == kw and i == c and o == c, f"{p}.binary_conv: unexpected weight shape {tuple(w.shape)}")
        zeros = int((w == 0).sum())
        _expect(zeros == 0, f"{p}.binary_conv: {zeros} latent weights are exactly 0 (torch.sign gives a 0 weight)")
        sign = w.sign().numpy().astype(np.int8)
        pad, s = kh // 2, int(bc.stride)
        _expect(s == stride, f"{p}.binary_conv: stride {s} != unit stride {stride}")
        mark(f"{p}.binary_conv")
        t = g.add_node("BConv2D", [t, g.add_constant(f"{p}.binary_conv/kernel_bits", _to_ohwi(sign), dtype="bit", layout="OHWI_bit"), "", "", ""],
                       {"strides": [s, s], "dilations": [1, 1], "pads": [pad, pad, pad, pad], "groups": 1, "pad_value": 0.0,
                        "input_binarize": binarize, "activation": "none", "output": "f32"}, name=f"{p}.binary_conv")
        precision.append({"layer": f"{p}.binary_conv", "op": "BConv2D", "source": "HardSign + HardBinaryConv",
                          "weights": "binary sign(W), no scaling factor", "activations": "binary", "input_binarize": binarize,
                          "source_binarize": "torch.sign (sign(0)=0)", "pad_value": 0.0, "kernel": [kh, kw], "stride": s})
        diag.append({"layer": f"{p}.binary_conv", "latent_weight_zeros": zeros, "n_weights": int(w.numel()),
                     "temperature": float(bc.temperature), "act_temperature": float(ha.temperature)})
        t = bn(u.norm1, t, f"{p}.norm1")
        t = prelu(u.activation2, t, f"{p}.activation2", c)
        if u.infor_recoupling:
            sc = u.scale.detach().cpu().double().reshape(-1)
            if cls == "FFN_1x1":
                sc = sc.clamp(0, 1)   # FFN_1x1.forward clamps in eval mode too (bnext.py:298)
            sc = sc.numpy()
            mark(f"{p}.scale")
            mix_in = x
            if stride == 2:
                mix_in = avgpool2(u.pooling, x, f"{p}.pooling.input")
            a = g.add_node("ChannelAffine", [mix_in, folded(f"{p}.scale", sc), ""], name=f"{p}.mix_input")
            b = g.add_node("ChannelAffine", [t, folded(f"{p}.one_minus_scale", 1.0 - sc), ""], name=f"{p}.mix_x")
            mix = g.add_node("Add", [a, b], name=f"{p}.mix")
            se = u.se
            _expect(_cls(se) == "SqueezeAndExpand" and _cls(se.attention) == "Sigmoid",
                    f"{p}.se: only attention_mode='sigmoid' is supported (got {_cls(se.attention)})")
            seq = se.se
            _expect(_cls(seq) == "Sequential" and len(seq) == 4 and _cls(seq[0]) == "AdaptiveAvgPool2d"
                    and _pair(seq[0].output_size) == (1, 1) and _cls(seq[2]) == "ReLU", f"{p}.se: unexpected structure")
            gt = g.add_node("GlobalAvgPool", [mix], {"keepdims": 1}, name=f"{p}.se.se.0")
            gt = conv1x1_bias(seq[1], gt, f"{p}.se.se.1", "relu")
            gt = conv1x1_bias(seq[3], gt, f"{p}.se.se.3", "none")
            t = g.add_node("ChannelGate", [t, gt], {"gate": "sigmoid"}, name=f"{p}.se_mul")
            precision.append({"layer": f"{p}.se", "op": "GlobalAvgPool+Conv2D+Conv2D+ChannelGate(sigmoid)",
                              "source": "SqueezeAndExpand(sigmoid) * x", "weights": "fp32", "activations": "fp32"})
            if cls == "Attention":
                t = g.add_node("Mul", [t, residual], name=f"{p}.elm_mul")
            t = bn(u.norm2, t, f"{p}.norm2")
        else:
            if cls == "Attention":
                t = g.add_node("Mul", [t, residual], name=f"{p}.elm_mul")
                t = bn(u.norm2, t, f"{p}.norm2")
            else:
                unused.append(f"{p}.norm2 (BatchNorm2d unused in {cls}.forward without info-recoupling)")
        return g.add_node("Add", [t, residual], name=f"{p}.add")

    x = g.add_input(input_name, [1, input_hw[0], input_hw[1], 3])

    # ---------------------------------------------------------------- stem
    feats = model.feature
    _expect(_cls(feats) == "ModuleList" and len(feats) >= 2, "feature: expected ModuleList")
    stem = feats[0]
    _expect(_cls(stem) == "firstconv3x3", f"feature.0: unexpected {_cls(stem)}")
    cv = stem.conv1
    _expect(_cls(cv) == "Conv2d" and cv.bias is None and cv.groups == 1 and _pair(cv.dilation) == (1, 1), "feature.0.conv1: unexpected conv")
    sh, sw = _pair(cv.stride)
    ph, pw = _pair(cv.padding)
    mark("feature.0.conv1")
    x = g.add_node("Conv2D", [x, g.add_constant("feature.0.conv1/kernel", _to_ohwi(cv.weight.detach().cpu().numpy().astype(np.float32)), layout="OHWI"), ""],
                   {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0, "activation": "none"},
                   name="feature.0.conv1")
    x = bn(stem.bn1, x, "feature.0.bn1")
    unused.append("feature.0.prelu (PReLU constructed, unused in firstconv3x3.forward)")
    precision.append({"layer": "feature.0", "op": "Conv2D+ChannelAffine", "source": "firstconv3x3 (Conv2d 3x3 + BN)",
                      "weights": "fp32", "activations": "fp32"})
    c = int(cv.out_channels)

    # ---------------------------------------------------------------- body
    for bi in range(1, len(feats)):
        blk = feats[bi]
        p = f"feature.{bi}"
        _expect(_cls(blk) in BLOCK_CLASSES, f"{p}: unsupported block {_cls(blk)}")
        _expect(blk.inplanes == c and blk.planes in (c, 2 * c), f"{p}: {blk.inplanes}->{blk.planes} with {c} input channels")
        first = "Attention" if hasattr(blk, "Attention") else "FFN_3x3"
        x = unit(getattr(blk, first), x, f"{p}.{first}", c)
        if blk.inplanes == blk.planes:
            x = unit(blk.FFN, x, f"{p}.FFN", c)
        else:
            y1 = unit(blk.FFN_1, x, f"{p}.FFN_1", c)
            y2 = unit(blk.FFN_2, x, f"{p}.FFN_2", c)
            x = g.add_node("Concat", [y1, y2], {"axis": -1}, name=f"{p}.cat")
            c = 2 * c

    # ---------------------------------------------------------------- head
    x = prelu(model.prelu, x, "prelu", c)
    _expect(_cls(model.pool1) == "AdaptiveAvgPool2d" and _pair(model.pool1.output_size) == (1, 1), "pool1: expected AdaptiveAvgPool2d(1)")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="pool1")
    fc = model.fc
    _expect(_cls(fc) == "Linear" and fc.bias is not None and fc.in_features == c, f"fc: unexpected {_cls(fc)}")
    mark("fc")
    x = g.add_node("Dense", [x, g.add_constant("fc/kernel", fc.weight.detach().cpu().numpy().astype(np.float32)),
                             g.add_constant("fc/bias", fc.bias.detach().cpu().numpy().astype(np.float32))],
                   {"activation": "none"}, name="fc")
    precision.append({"layer": "fc", "op": "Dense", "source": "Linear", "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)

    # Every module must be of a known type, every parameter/buffer must belong to a converted
    # module or be listed as unused in forward.
    known = {"BNext", "ModuleList", "Sequential", "firstconv3x3", *BLOCK_CLASSES, *UNIT_CLASSES, "Conv2d", "BatchNorm2d",
             "PReLU", "LearnableBias", "HardSign", "HardBinaryConv", "AvgPool2d", "SqueezeAndExpand", "AdaptiveAvgPool2d",
             "ReLU", "Sigmoid", "Linear"}
    for n, m in model.named_modules():
        _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
    unused_roots = [u.split(" ")[0] for u in unused]
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        ok = any(n == v or n.startswith(v + ".") for v in visited | set(unused_roots))
        _expect(ok, f"parameter/buffer {n} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": unused, "float64_constants": exact}
    return g, info
