"""Adapter: instantiated official AB-BNN PyTorch model (ReActNet-18/34 and ReActNet-A) -> Bitino IR.

Source: https://github.com/Ruichen0424/AB-BNN (code commit 3485ea74), Network/BNN.py,
Network/ReActNet_18_Qaw.py and Network/ReActNet_A_Qaw.py (ImageNet and CIFAR stems; the CIFAR
ResNet stem is a 3x3 stride-1 ScaledStdConv2d with nn.Identity in place of the max-pool). The adapter reads the *instantiated* nn.Module (official
class definitions, weights loaded with torch.load(weights_only=True)); it mirrors
ResNet.forward / BasicBlock.forward explicitly and raises on any module type or
attribute it does not know.

Semantics captured (file:line refer to the commit above):
  * Activation binarization: BasicBlock computes sign((x + bias/beta) * beta) with
    torch.sign (BNN.py:12-13,29-35; ReActNet_18_Qaw.py:55-56). beta > 0, so the sign
    equals sign(x + bias/beta) -> ChannelAffine(bias = bias/beta) feeding a BConv2D.
    torch.sign(0) = 0: input_binarize "sign_zero" is exact (reference kernels only);
    "ge" (0 -> +1) is the deployable approximation, identical unless an activation is
    exactly 0 at a binarization point.
  * Binary conv (BNN.py:105-122): W_std = scale * (W - mean) / (std + eps), biased std
    over (C,H,W), scale = fan_in^-0.5; forward weight = gain * mean|W_std| * sign(W_std);
    F.conv2d zero padding of the *binarized* input -> BConv2D pad_value 0 with
    sign(W_std) bits and post_mul = gain * mean|W_std| * alpha (alpha from
    `out*alpha + residual`). A W_std that is exactly 0 has torch.sign = 0, which a bit
    tensor cannot hold: the adapter raises if one occurs in float64. Signs are taken from the
    float64 W_std; a W_std that is 0 only in float32 (ReActNet-A ImageNet: 1 weight) is reported
    (zeros_f32) and belongs to the fp32 source's noise floor, like the BHViT weight centring.
  * Downsample (ReActNet_18_Qaw.py:95-99): AvgPool2d(2) + binary-weight 1x1 conv on the
    real-valued input (W1A-fp) -> AvgPool2D + Conv2D with weights +-(gain*mean|W_std|).
  * Stem ScaledStdConv2d (BNN.py:65-89): weight standardization and gain folded into a
    plain Conv2D weight.
  * LearnableBias -> ChannelAffine(bias only); Q_PReLU (BNN.py:46-54) -> Activation prelu
    with slope 2^round(L_alpha) (torch.round = half to even), exact powers of two.
  * Head: AdaptiveAvgPool2d(1) + Linear -> GlobalAvgPool + Dense. No BatchNorm anywhere.

Also (slice S11) ReActNet-A *with BatchNorm*, through the same ReActNet-A skeleton (_convert_reactnet_a,
REACTA_BN_MODELS): BURN (naver-ai/burn, ReActnet + BinaryConv2d), S2-BNN (szq0214/S2-BNN, ReActNet +
HardBinaryConv) and the BiMatting ImageNet backbone (htqin/BiMatting, BiMobileNet + BiConv2d with centred weights,
extra 4x / half blocks and forward_single_frame macro residuals). BN after each binary conv is folded into the
BConv2D epilogue (post_mul = mean|W| * s_bn, post_add = t_bn, float64); see _convert_reactnet_a.

Constant folding: every constant the source derives from parameters at run time
(standardized weights, scaling factors, bias/beta) is computed in float64 and rounded
once to float32. Layouts: NCHW -> NHWC activations, OIHW -> OHWI filters.
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
        assert len(v) == 2
        return int(v[0]), int(v[1])
    return int(v), int(v)


def standardized_weight(conv, dtype):
    """W_std exactly as BNN.py:84-85 / 109-110 compute it (use_layernorm=False), in `dtype`."""
    import torch

    w = conv.weight.detach().to(dtype)
    std, mean = torch.std_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    return conv.scale * (w - mean) / (std + conv.eps)


def fold_scaled_std_conv(conv) -> np.ndarray:
    """ScaledStdConv2d.get_weight() in float64 (OIHW)."""
    import torch

    _expect(not conv.use_layernorm, f"ScaledStdConv2d use_layernorm=True not supported")
    w = conv.gain.detach().to(torch.float64) * standardized_weight(conv, torch.float64)
    return w.numpy()


def binary_conv_parts(conv) -> dict:
    """Splits HardBinaryScaledStdConv2d.get_weight() into sign bits and a per-channel scale.

    Returns sign (OIHW int8, from the float64 W_std), scale64 = gain * mean|W_std| (float64,
    per output channel) and diagnostics: exact zeros of W_std in float32/float64 and the
    number of weights whose sign differs between the float32 and float64 evaluation."""
    import torch

    _expect(not conv.use_layernorm, "HardBinaryScaledStdConv2d use_layernorm=True not supported")
    w64 = standardized_weight(conv, torch.float64)
    w32 = standardized_weight(conv, torch.float32)
    zeros64 = int((w64 == 0).sum())
    zeros32 = int((w32 == 0).sum())
    if zeros64:
        raise NotImplementedError(f"binary conv has exactly-zero standardized weights (f64: {zeros64}, f32: {zeros32}); "
                                  "torch.sign gives a 0 weight that a bit tensor cannot represent")
    # A W_std that is 0 only in float32 (cancellation in W - mean) is a 0 weight in the fp32 source but not in the
    # float64 source; the IR follows the float64 sign (reported as zeros_f32: part of the source's fp32 noise floor).
    sign = torch.sign(w64)
    mismatch = int((torch.sign(w32).to(torch.float64) != sign).sum())
    # BNN.py:112: nested means over dims 3, 2, 1.
    sf = torch.mean(torch.mean(torch.mean(w64.abs(), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
    scale64 = (conv.gain.detach().to(torch.float64) * sf).reshape(-1)
    return {"sign": sign.numpy().astype(np.int8), "scale64": scale64.numpy(),
            "zeros_f32": zeros32, "zeros_f64": zeros64, "sign_mismatch_f32_vs_f64": mismatch}


def qprelu_slope(prelu) -> np.ndarray:
    """2^round(L_alpha) as BNN.py:52-53 compute it; checked to be exact powers of two."""
    import torch

    e = torch.round(prelu.L_alpha.detach().to(torch.float64))
    slope = torch.pow(torch.tensor(2, dtype=torch.float64), e).numpy()
    ei = e.numpy().astype(np.int64)
    assert np.array_equal(slope, np.ldexp(1.0, ei)), "Q_PReLU slope is not an exact power of two"
    s32 = slope.astype(np.float32)
    assert np.array_equal(s32.astype(np.float64), slope), "Q_PReLU slope not representable in float32"
    return s32


def _to_ohwi(w_oihw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(w_oihw, (0, 2, 3, 1)))


def convert(model, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (224, 224),
            binarize: str = "sign_zero", literal_thresholds: bool = False) -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths (e.g. layer2.0.move0),
    so tools can map IR tensors back to module outputs. info carries per-layer precision
    and weight diagnostics. literal_thresholds: ReActNet-A-with-BN only (see _convert_reactnet_a)."""
    import torch

    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    if _cls(model) == "Reactnet" or _cls(model) in REACTA_BN_MODELS:
        return _convert_reactnet_a(model, input_name, input_hw, binarize, literal_thresholds)
    _expect(_cls(model) == "ResNet", f"unsupported top-level model class {_cls(model)} (expected AB-BNN ResNet or Reactnet, "
                                     f"or a ReActNet-A-with-BN class {REACTA_BN_MODELS})")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}   # float64 values of constants that are rounded to float32

    def folded(name: str, value64: np.ndarray, layout: str = "") -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, value64.astype(np.float32), layout=layout)
    x = g.add_input(input_name, [1, input_hw[0], input_hw[1], 3])

    def bias_of(lb, beta: float, name: str) -> str:
        _expect(_cls(lb) == "LearnableBias", f"{name}: expected LearnableBias, got {_cls(lb)}")
        b = lb.bias.detach().to(torch.float64).reshape(-1) / beta     # ReActNet_18_Qaw.py:26
        return folded(f"{name}/bias", b.numpy())

    # ---------------------------------------------------------------- stem
    conv1 = model.conv1
    _expect(_cls(conv1) == "ScaledStdConv2d", f"conv1: unexpected {_cls(conv1)}")
    _expect(conv1.bias is None and conv1.groups == 1, "conv1: bias/groups not supported")
    kh, kw = _pair(conv1.kernel_size)
    sh, sw = _pair(conv1.stride)
    ph, pw = _pair(conv1.padding)
    dh, dw = _pair(conv1.dilation)
    w = _to_ohwi(fold_scaled_std_conv(conv1))
    x = g.add_node("Conv2D", [x, folded("conv1/kernel", w, layout="OHWI"), ""],
                   {"strides": [sh, sw], "dilations": [dh, dw], "pads": [ph, ph, pw, pw], "groups": 1,
                    "pad_value": 0.0, "activation": "none"}, name="conv1")
    precision.append({"layer": "conv1", "op": "Conv2D", "source": "ScaledStdConv2d", "weights": "fp32 (weight standardization + gain folded)",
                      "activations": "fp32"})

    mp = model.maxpool
    if _cls(mp) == "MaxPool2d":
        _expect(_pair(mp.dilation) == (1, 1) and not mp.ceil_mode, "maxpool: dilation/ceil_mode not supported")
        k, s, p = _pair(mp.kernel_size), _pair(mp.stride), _pair(mp.padding)
        x = g.add_node("MaxPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [p[0], p[0], p[1], p[1]]}, name="maxpool")
    else:
        _expect(_cls(mp) == "Identity", f"maxpool: unexpected {_cls(mp)}")

    # ---------------------------------------------------------------- body
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, lname)
        _expect(_cls(layer) == "Sequential", f"{lname}: unexpected {_cls(layer)}")
        for bi, blk in enumerate(layer):
            p = f"{lname}.{bi}"
            _expect(_cls(blk) == "BasicBlock", f"{p}: unexpected block {_cls(blk)}")
            _expect(_cls(blk.act1) == "BinaryActivation", f"{p}.act1: unexpected {_cls(blk.act1)}")
            _expect(_cls(blk.conv1) == "HardBinaryScaledStdConv2d", f"{p}.conv1: unexpected {_cls(blk.conv1)}")
            _expect(_cls(blk.prelu) == "Q_PReLU", f"{p}.prelu: unexpected {_cls(blk.prelu)}")
            alpha, beta = float(blk.alpha), float(blk.beta)
            _expect(beta > 0, f"{p}: beta must be > 0 for sign((x+b/beta)*beta) == sign(x+b/beta)")
            residual = x

            # move0 -> sign -> binary 3x3 conv (ReActNet_18_Qaw.py:55-57)
            t = g.add_node("ChannelAffine", [x, "", bias_of(blk.move0, beta, f"{p}.move0")], name=f"{p}.move0")
            bc = blk.conv1
            parts = binary_conv_parts(bc)
            o, i, kh, kw = parts["sign"].shape
            s, pd = int(bc.stride), int(bc.padding)
            post_mul = parts["scale64"] * alpha
            t = g.add_node("BConv2D", [t, g.add_constant(f"{p}.conv1/kernel_bits", _to_ohwi(parts["sign"]), dtype="bit", layout="OHWI_bit"),
                                       folded(f"{p}.conv1/post_mul", post_mul), "", ""],
                           {"strides": [s, s], "dilations": [1, 1], "pads": [pd, pd, pd, pd], "groups": 1,
                            "pad_value": 0.0, "input_binarize": binarize, "activation": "none", "output": "f32"},
                           name=f"{p}.conv1")
            precision.append({"layer": f"{p}.conv1", "op": "BConv2D", "source": "HardBinaryScaledStdConv2d",
                              "weights": "binary (sign(W_std), per-channel gain*mean|W_std|*alpha)", "activations": "binary",
                              "input_binarize": binarize, "source_binarize": "torch.sign (sign(0)=0)", "pad_value": 0.0,
                              "kernel": [kh, kw], "stride": s})
            diag.append({"layer": f"{p}.conv1", **{k: parts[k] for k in ("zeros_f32", "zeros_f64", "sign_mismatch_f32_vs_f64")}})

            # downsample: AvgPool2d + binary-weight 1x1 conv on the real-valued x (ReActNet_18_Qaw.py:59-60,95-99)
            if blk.downsample is not None:
                ds = blk.downsample
                _expect(_cls(ds) == "Sequential" and len(ds) == 2, f"{p}.downsample: unexpected structure")
                ap, dc = ds[0], ds[1]
                _expect(_cls(ap) == "AvgPool2d", f"{p}.downsample.0: unexpected {_cls(ap)}")
                _expect(not ap.ceil_mode and ap.divisor_override is None and _pair(ap.padding) == (0, 0),
                        f"{p}.downsample.0: ceil_mode/padding/divisor_override not supported")
                _expect(_cls(dc) == "HardBinaryScaledStdConv2d", f"{p}.downsample.1: unexpected {_cls(dc)}")
                _expect(int(dc.padding) == 0 and int(dc.stride) == 1 and dc.shape[2:] == (1, 1),
                        f"{p}.downsample.1: expected 1x1 stride 1 pad 0")
                k, st = _pair(ap.kernel_size), _pair(ap.stride)
                r = g.add_node("AvgPool2D", [x], {"kernel": list(k), "strides": list(st), "pads": [0, 0, 0, 0],
                                                  "count_include_pad": 1 if ap.count_include_pad else 0},
                               name=f"{p}.downsample.0")
                dparts = binary_conv_parts(dc)
                wd = dparts["sign"].astype(np.float64) * dparts["scale64"][:, None, None, None]
                residual = g.add_node("Conv2D", [r, folded(f"{p}.downsample.1/kernel", _to_ohwi(wd), layout="OHWI"), ""],
                                      {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1,
                                       "pad_value": 0.0, "activation": "none"}, name=f"{p}.downsample.1")
                precision.append({"layer": f"{p}.downsample.1", "op": "Conv2D", "source": "HardBinaryScaledStdConv2d (1x1, after AvgPool2d)",
                                  "weights": "binary (+-gain*mean|W_std| per channel, stored as fp32)", "activations": "fp32 (W1A-fp)"})
                diag.append({"layer": f"{p}.downsample.1", **{k: dparts[k] for k in ("zeros_f32", "zeros_f64", "sign_mismatch_f32_vs_f64")}})

            # out*alpha + residual -> move1 -> Q_PReLU -> move2 (ReActNet_18_Qaw.py:62-65)
            t = g.add_node("Add", [t, residual], name=f"{p}.add")
            t = g.add_node("ChannelAffine", [t, "", bias_of(blk.move1, 1.0, f"{p}.move1")], name=f"{p}.move1")
            t = g.add_node("Activation", [t, g.add_constant(f"{p}.prelu/slope", qprelu_slope(blk.prelu))], {"type": "prelu"},
                           name=f"{p}.prelu")
            x = g.add_node("ChannelAffine", [t, "", bias_of(blk.move2, 1.0, f"{p}.move2")], name=f"{p}.move2")
            precision.append({"layer": f"{p}.prelu", "op": "Activation(prelu)", "source": "Q_PReLU",
                              "weights": "fp32 slopes, exact powers of two", "activations": "fp32"})

    # ---------------------------------------------------------------- head
    _expect(_cls(model.avgpool) == "AdaptiveAvgPool2d" and _pair(model.avgpool.output_size) == (1, 1), "avgpool: expected AdaptiveAvgPool2d(1)")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="avgpool")
    fc = model.fc
    _expect(_cls(fc) == "Linear" and fc.bias is not None, f"fc: unexpected {_cls(fc)}")
    x = g.add_node("Dense", [x, g.add_constant("fc/kernel", fc.weight.detach().numpy().astype(np.float32)),
                             g.add_constant("fc/bias", fc.bias.detach().numpy().astype(np.float32))],
                   {"activation": "none"}, name="fc")
    precision.append({"layer": "fc", "op": "Dense", "source": "Linear", "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)

    # Every module must be of a known type and every parameter must belong to a visited module.
    known = {"ResNet", "Sequential", "BasicBlock", "ScaledStdConv2d", "HardBinaryScaledStdConv2d", "LearnableBias",
             "BinaryActivation", "Q_PReLU", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "Linear", "Identity"}
    for n, m in model.named_modules():
        _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
    seen = {"conv1", "fc"} | {f"{l}.{i}" for l in ("layer1", "layer2", "layer3", "layer4") for i in range(len(getattr(model, l)))}
    for n, _ in model.named_parameters():
        root = n.split(".")[0] if n.split(".")[0] in ("conv1", "fc") else ".".join(n.split(".")[:2])
        _expect(root in seen, f"parameter {n} not covered by the adapter")
    # float64_constants: analysis only (never serialized); swapping them into the graph gives the
    # IR with exact folded constants, isolating the effect of float32 constant storage.
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "float64_constants": exact}
    return g, info



# ===================================================================== ReActNet-A with BatchNorm (slice S11)
# Top-level classes of the ReActNet-A-with-BN sources handled by _convert_reactnet_a (besides AB-BNN's "Reactnet"):
#   "ReActnet"              BURN (naver-ai/burn openselfsup/models/backbones/reactnet.py, ReActnet(use_mlp=True))
#   "ReActNet"              S2-BNN (szq0214/S2-BNN Distillation_only/step2/ReActNet.py; fc = Identity for features)
#   "BiMobileNetClassifier" BiMatting ImageNet backbone (htqin/BiMatting model/mobilenet.py BiMobileNet) + the
#                           checkpoint's AdaptiveAvgPool2d + Linear head (wrapper built by tools/convert_reacta_bn.py)
REACTA_BN_MODELS = ("ReActnet", "ReActNet", "BiMobileNetClassifier")
# BiMobileNet.forward_single_frame (mobilenet.py:206-217): macro blocks [first, last]; after block `last` the output of
# block `first` is added through AvgPool2d(2, 2, ceil_mode=True) when the spatial size changed in between.
BIMOBILENET_MACRO_BLOCKS = ((1, 3), (4, 6), (7, 9), (10, 12), (13, 15))


def _nested_mean(t, dims=(3, 2, 1)):
    """torch.mean(torch.mean(torch.mean(t, dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)."""
    import torch

    for d in dims:
        t = torch.mean(t, dim=d, keepdim=True)
    return t


def bn_binary_conv_parts(conv) -> dict:
    """Sign bits and per-output-channel scale of the binary convs of the ReActNet-A-with-BN sources (eval forward):
      * BinaryConv2d (BURN reactnet.py:120-146): F.conv2d(sign(x), sign(W) * mean|W|), mean over (in, kh, kw) as
        nested means; a product of +-1 and the scale, exact in float32.
      * HardBinaryConv (S2-BNN ReActNet.py:68-86): scale * sign(W) - clamp(W) + clamp(W) (STE); the STE terms cancel
        up to float32 rounding (exactly in exact arithmetic) -> same bits and scale (SNN / RBONN precedent).
      * BiConv2d (BiMatting mobilenet.py:8-26): W' = W - nested_mean(W) (centred per output channel), then
        scale * sign(W') - clamp(W') + clamp(W') with scale = nested_mean(|W'|); signs from the float64 W' (a W' that
        is 0 only in float32 is reported as zeros_f32, sign differences as sign_mismatch_f32_vs_f64).
    Zero padding of the binarized input, no bias, no dilation / groups (checked). A weight whose sign is 0 in float64
    raises (torch.sign would give a 0 weight that a bit tensor cannot hold)."""
    import torch

    cls = _cls(conv)
    _expect(cls in ("BinaryConv2d", "HardBinaryConv", "BiConv2d"), f"unsupported binary conv {cls}")
    if cls in ("BinaryConv2d", "BiConv2d"):   # nn.Conv2d subclasses
        _expect(conv.bias is None and conv.groups == 1 and _pair(conv.dilation) == (1, 1) and conv.padding_mode == "zeros",
                f"{cls}: bias/groups/dilation/padding_mode not supported")
    w64 = conv.weight.detach().to(torch.float64)
    w32 = conv.weight.detach().to(torch.float32)
    if cls == "BiConv2d":
        w64 = w64 - _nested_mean(w64)
        w32 = w32 - _nested_mean(w32)
    zeros64, zeros32 = int((w64 == 0).sum()), int((w32 == 0).sum())
    if zeros64:
        raise NotImplementedError(f"{cls} has exactly-zero (effective) weights (f64: {zeros64}, f32: {zeros32}); "
                                  "torch.sign gives a 0 weight that a bit tensor cannot represent")
    sign = torch.sign(w64)
    mismatch = int((torch.sign(w32).to(torch.float64) != sign).sum())
    scale64 = _nested_mean(w64.abs()).reshape(-1)
    return {"sign": sign.numpy().astype(np.int8), "scale64": scale64.numpy(), "zeros_f32": zeros32, "zeros_f64": zeros64,
            "sign_mismatch_f32_vs_f64": mismatch, "kind": cls,
            "stride": _pair(conv.stride), "padding": _pair(conv.padding)}


def bn_affine64(bn) -> tuple[np.ndarray, np.ndarray]:
    """BatchNorm2d in eval mode (running statistics) as y = s*x + t, folded in float64."""
    import torch

    _expect(_cls(bn) == "BatchNorm2d" and bn.affine and bn.track_running_stats, f"unsupported norm layer {bn}")
    var = bn.running_var.detach().to(torch.float64)
    s = bn.weight.detach().to(torch.float64) / torch.sqrt(var + bn.eps)
    t = bn.bias.detach().to(torch.float64) - bn.running_mean.detach().to(torch.float64) * s
    return s.numpy(), t.numpy()


def _convert_reactnet_a(model, input_name: str, input_hw: tuple[int, int], binarize: str,
                        literal_thresholds: bool = False) -> tuple[Graph, dict]:
    """ReActNet-A (MobileNet-v1-like) in two flavours sharing one block skeleton.

    AB-BNN (Network/ReActNet_A_Qaw.py: Reactnet, firstconv3x3, BasicBlock), no BatchNorm; mirrors Reactnet.forward /
    BasicBlock.forward (ReActNet_A_Qaw.py:87-131, 166-172):
      stem   firstconv3x3 = ScaledStdConv2d(3, 32, 3, stride 2 (ImageNet) / 1 (CIFAR), pad 1), no activation
      block  out1 = x' + alpha * Bconv3x3(sign(move11(x, beta1) * beta1))   with x' = AvgPool2d(2)(x) if stride 2
             out1 = move13(prelu1(move12(out1)))
             b    = sign(move21(out1, beta2) * beta2)
             out2 = alpha * Bconv1x1(b) + out1                                 if inplanes == planes
                  = concat(alpha * Bconv1x1_down1(b) + out1, alpha * Bconv1x1_down2(b) + out1)   otherwise
             out  = move23(prelu2(move22(out2)))
      head   AdaptiveAvgPool2d(1) + Linear
    Binarization, binary-conv folding and Q_PReLU are the ReActNet-18 path's (module docstring).

    With BatchNorm (slice S11; classes REACTA_BN_MODELS): BURN, S2-BNN and the BiMatting backbone. Same skeleton with
      stem   Conv2d(3, C, 3, s2, p1, no bias) + BN                      -> Conv2D + ChannelAffine (BN folded in float64)
      block  BN after every binary conv, before the residual add       -> folded into the BConv2D epilogue (float64):
             post_mul = mean|W| * s_bn, post_add = t_bn
             sign(move(x)) (torch.sign, sign(0) = 0)                   -> ChannelAffine(bias) + BConv2D input binarization
             nn.PReLU(C) (per channel)                                 -> Activation prelu
      BiMatting adds two block types (mobilenet.py:95-107, 152-172): planes == 4*inplanes (four 1x1 binary convs,
      each + out1, concatenated) and planes == inplanes/2 (one 1x1 binary conv inplanes -> planes, + (out1[:, :P] +
      out1[:, P:]) / 2 -> Reshape [1, H*W, 2, P] + AvgPool2D [1, 2] + Reshape, exact: one rounding of a + b, halved),
      and the macro residuals of forward_single_frame (BIMOBILENET_MACRO_BLOCKS; AvgPool2d(2, 2, ceil_mode=True) is
      floor mode for the even map sizes, checked).
      heads  BURN: AdaptiveAvgPool2d + fc = Sequential(Linear(1024, 1024), ReLU) (use_mlp=True; the backbone output
             that linear evaluation consumes); S2-BNN: AdaptiveAvgPool2d + fc = Identity (main_lincls.py feature);
             BiMatting: AdaptiveAvgPool2d + Linear(1024, 1000).
    Concat order: down1, down2(, down3, down4) along channels (torch.cat dim=1 -> NHWC axis -1).

    Binarization thresholds (BN flavour): the source binarizes sign(fl(fl(q + b_prev) + b_next)), where q is the
    PReLU output and b_prev / b_next the LearnableBias before and after it (move13 -> move21 inside a block, move23 ->
    next block's move11). On ~2% of the channels the PReLU slope on the negative side is tiny (|slope| ~1e-6..1e-4)
    and b_prev + b_next nearly cancel (~|slope * y|), so the decision variable q + b_prev + b_next (~1e-5) is lost in
    the rounding of fl(q + b_prev) (ulp ~1e-8): float32 ties (exact zeros) and a strongly chaotic float32 source. By
    default the IR binarizes q + fl64(b_prev + b_next) instead (one rounding of a constant; the runtime's plan-time
    threshold folding turns it into the exact comparison q >= -(b_prev + b_next)), equal to the source in exact
    arithmetic; the residual path keeps fl(q + b_prev) as the source. Not applied where the binarized tensor is not a
    PReLU tail (block 1 after the stem BN, blocks after a BiMatting macro residual). literal_thresholds=True mirrors
    the source's two float32 additions instead (comparison only)."""
    import torch

    top = _cls(model)
    bn_flavour = top in REACTA_BN_MODELS
    _expect(bn_flavour or top == "Reactnet", f"unsupported ReActNet-A class {top}")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    unused: list[str] = []
    covered: set[str] = set()   # module paths whose parameters the IR uses (BN flavour)
    tails: dict[str, tuple] = {}  # move13 / move23 node name -> (PReLU output tensor, its bias in float64)
    refolded: list[str] = []

    def folded(name: str, value64: np.ndarray, layout: str = "") -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, value64.astype(np.float32), layout=layout)

    def bias_of(lb, beta: float, name: str) -> str:
        _expect(_cls(lb) == "LearnableBias", f"{name}: expected LearnableBias, got {_cls(lb)}")
        b = lb.bias.detach().to(torch.float64).reshape(-1) / beta     # ReActNet_A_Qaw.py:27-29 (beta = 1 with BN: exact)
        return folded(f"{name}/bias", b.numpy())

    def bconv(src: str, conv, alpha: float, name: str, bn=None) -> str:
        if not bn_flavour:
            _expect(_cls(conv) == "HardBinaryScaledStdConv2d", f"{name}: unexpected {_cls(conv)}")
            parts = binary_conv_parts(conv)
            s, pd = int(conv.stride), int(conv.padding)
            post_mul, post_add = parts["scale64"] * alpha, None
            wdesc, src_desc = "binary (sign(W_std), per-channel gain*mean|W_std|*alpha)", "HardBinaryScaledStdConv2d"
        else:
            parts = bn_binary_conv_parts(conv)
            (s, s2), (pd, pd2) = parts["stride"], parts["padding"]
            _expect(s == s2 and pd == pd2, f"{name}: non-square stride/padding")
            sbn, tbn = bn_affine64(bn)
            post_mul, post_add = parts["scale64"] * sbn, tbn
            wdesc = ("binary (sign(W - mean W), per-channel mean|W - mean W| * BN scale)" if parts["kind"] == "BiConv2d"
                     else "binary (sign(W), per-channel mean|W| * BN scale)")
            src_desc = f"{parts['kind']} + BatchNorm2d"
        o, i, kh, kw = parts["sign"].shape
        ins = [src, g.add_constant(f"{name}/kernel_bits", _to_ohwi(parts["sign"]), dtype="bit", layout="OHWI_bit"),
               folded(f"{name}/post_mul", post_mul), folded(f"{name}/post_add", post_add) if post_add is not None else "", ""]
        out = g.add_node("BConv2D", ins, {"strides": [s, s], "dilations": [1, 1], "pads": [pd, pd, pd, pd], "groups": 1,
                                          "pad_value": 0.0, "input_binarize": binarize, "activation": "none", "output": "f32"},
                         name=name)
        precision.append({"layer": name, "op": "BConv2D", "source": src_desc, "weights": wdesc, "activations": "binary",
                          "input_binarize": binarize, "source_binarize": "torch.sign (sign(0)=0)", "pad_value": 0.0,
                          "kernel": [kh, kw], "stride": s})
        diag.append({"layer": name, **{k: parts[k] for k in ("zeros_f32", "zeros_f64", "sign_mismatch_f32_vs_f64")}})
        return out

    def act_tail(t: str, blk, a: str, b: str, c: str, p: str) -> str:
        pr = getattr(blk, b)
        t = g.add_node("ChannelAffine", [t, "", bias_of(getattr(blk, a), 1.0, f"{p}.{a}")], name=f"{p}.{a}")
        if _cls(pr) == "Q_PReLU" and not bn_flavour:
            slope, wd, sd = qprelu_slope(pr), "fp32 slopes, exact powers of two", "Q_PReLU"
        else:
            _expect(bn_flavour and _cls(pr) == "PReLU", f"{p}.{b}: unexpected {_cls(pr)}")
            slope = pr.weight.detach().to(torch.float32).numpy().reshape(-1).copy()
            _expect(slope.size == int(blk.planes if c.endswith("23") else blk.inplanes), f"{p}.{b}: expected one slope per channel")
            wd, sd = "fp32 per-channel slopes", "nn.PReLU"
        t = g.add_node("Activation", [t, g.add_constant(f"{p}.{b}/slope", slope)], {"type": "prelu"}, name=f"{p}.{b}")
        precision.append({"layer": f"{p}.{b}", "op": "Activation(prelu)", "source": sd, "weights": wd, "activations": "fp32"})
        tails[p + "." + c] = (t, getattr(blk, c).bias.detach().to(torch.float64).reshape(-1).numpy())
        return g.add_node("ChannelAffine", [t, "", bias_of(getattr(blk, c), 1.0, f"{p}.{c}")], name=f"{p}.{c}")

    def threshold(src: str, lb, beta: float, name: str) -> str:
        """ChannelAffine(bias) feeding a binarization: literal fl(src + b), or q + fl64(b_prev + b) on a PReLU tail."""
        _expect(_cls(lb) == "LearnableBias", f"{name}: expected LearnableBias, got {_cls(lb)}")
        prod = next((nd for nd in g.nodes if src in nd.outputs), None)
        if not bn_flavour or literal_thresholds or prod is None or prod.name not in tails:
            return g.add_node("ChannelAffine", [src, "", bias_of(lb, beta, name)], name=name)
        q, b_prev = tails[prod.name]
        c64 = b_prev + lb.bias.detach().to(torch.float64).reshape(-1).numpy()
        refolded.append(name)
        return g.add_node("ChannelAffine", [q, "", folded(f"{name}/bias", c64)], name=name)

    def avgpool(src: str, ap, name: str, hw: tuple[int, int]) -> str:
        _expect(_cls(ap) == "AvgPool2d" and ap.divisor_override is None and _pair(ap.padding) == (0, 0), f"{name}: unexpected {ap}")
        k, st = _pair(ap.kernel_size), _pair(ap.stride)
        if ap.ceil_mode:   # identical to floor mode when every window fits (checked on the tracked map size)
            _expect((hw[0] - k[0]) % st[0] == 0 and (hw[1] - k[1]) % st[1] == 0,
                    f"{name}: ceil_mode=True with a partial window at map size {hw} (not supported)")
        return g.add_node("AvgPool2D", [src], {"kernel": list(k), "strides": list(st), "pads": [0, 0, 0, 0],
                                               "count_include_pad": 1 if ap.count_include_pad else 0}, name=name)

    # ------------------------------------------------------------------ top level
    head_kind = "linear"
    macro: tuple = ()
    if top == "BiMobileNetClassifier":
        net = model.backbone
        _expect(_cls(net) == "BiMobileNet", f"backbone: unexpected {_cls(net)}")
        feats, macro = net.feature, BIMOBILENET_MACRO_BLOCKS
        pool1, fc = model.pool1, model.fc
    elif top == "ReActnet":   # BURN: forward returns (pool -> fc MLP)(feature[-1]) as a 1-tuple
        feats = model.feature
        oi = list(model.out_indices) if isinstance(model.out_indices, (list, tuple)) else [model.out_indices]
        _expect(oi == [len(feats) - 1], f"ReActnet: out_indices {model.out_indices} (only the last block is supported)")
        _expect(bool(model.use_mlp), "ReActnet: use_mlp=False (no pooled head) not supported")
        pool1, fc, head_kind = model.pool1, model.fc, "mlp"
    else:
        feats, pool1, fc = model.feature, model.pool1, model.fc
    x = g.add_input(input_name, [1, input_hw[0], input_hw[1], 3])
    _expect(_cls(feats) == "ModuleList" and len(feats) > 1, "feature: expected a ModuleList")
    stem = feats[0]
    _expect(_cls(stem) == "firstconv3x3", f"feature.0: unexpected {_cls(stem)}")
    c1 = stem.conv1
    _expect(_cls(c1) == ("Conv2d" if bn_flavour else "ScaledStdConv2d"), f"feature.0.conv1: unexpected {_cls(c1)}")
    _expect(c1.bias is None and c1.groups == 1 and _pair(c1.dilation) == (1, 1), "feature.0.conv1: bias/groups/dilation not supported")
    (sh, sw), (ph, pw), (kh, kw) = _pair(c1.stride), _pair(c1.padding), _pair(c1.kernel_size)
    if bn_flavour:
        w0 = c1.weight.detach().to(torch.float64).numpy()
        x = g.add_node("Conv2D", [x, folded("feature.0.conv1/kernel", _to_ohwi(w0), layout="OHWI"), ""],
                       {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1,
                        "pad_value": 0.0, "activation": "none"}, name="feature.0.conv1")
        s0, t0 = bn_affine64(stem.bn1)
        x = g.add_node("ChannelAffine", [x, folded("feature.0.bn1/scale", s0), folded("feature.0.bn1/shift", t0)], name="feature.0.bn1")
        precision.append({"layer": "feature.0.conv1", "op": "Conv2D + ChannelAffine", "source": "Conv2d + BatchNorm2d (firstconv3x3)",
                          "weights": "fp32", "activations": "fp32"})
    else:
        x = g.add_node("Conv2D", [x, folded("feature.0.conv1/kernel", _to_ohwi(fold_scaled_std_conv(c1)), layout="OHWI"), ""],
                       {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1,
                        "pad_value": 0.0, "activation": "none"}, name="feature.0.conv1")
        precision.append({"layer": "feature.0.conv1", "op": "Conv2D", "source": "ScaledStdConv2d (firstconv3x3)",
                          "weights": "fp32 (weight standardization + gain folded)", "activations": "fp32"})
    hw = ((input_hw[0] + 2 * ph - kh) // sh + 1, (input_hw[1] + 2 * pw - kw) // sw + 1)

    block_cls = ("BasicBlock", "ReActNetBlock") if bn_flavour else ("BasicBlock",)
    macro_first = {a: b for a, b in macro}
    macro_feat: dict[int, tuple[str, tuple[int, int]]] = {}
    for bi in range(1, len(feats)):
        blk, p = feats[bi], f"feature.{bi}"
        _expect(_cls(blk) in block_cls, f"{p}: unexpected block {_cls(blk)}")
        if bn_flavour:
            alpha = beta1 = beta2 = 1.0
            for an in ("binary_activation", "binary_activation1", "binary_activation2"):
                if hasattr(blk, an):
                    _expect(_cls(getattr(blk, an)) == "BinaryActivation", f"{p}.{an}: unexpected type")
        else:
            _expect(_cls(blk.binary_activation) == "BinaryActivation", f"{p}.binary_activation: unexpected type")
            alpha, beta1, beta2 = float(blk.alpha), float(blk.beta1), float(blk.beta2)
            _expect(beta1 > 0 and beta2 > 0, f"{p}: beta must be > 0 for sign((x+b/beta)*beta) == sign(x+b/beta)")
        stride, cin, cout = int(blk.stride), int(blk.inplanes), int(blk.planes)
        x_in = x
        t = threshold(x_in, blk.move11, beta1, f"{p}.move11")
        t = bconv(t, blk.binary_3x3, alpha, f"{p}.binary_3x3", getattr(blk, "bn1", None) if bn_flavour else None)
        res = x_in
        if stride == 2:
            res = avgpool(x_in, blk.pooling, f"{p}.pooling", hw)
            hw = (hw[0] // 2, hw[1] // 2)
        else:
            _expect(stride == 1, f"{p}: stride {stride}")
        t = g.add_node("Add", [t, res], name=f"{p}.add1")
        out1 = act_tail(t, blk, "move12", "prelu1", "move13", p)
        b = threshold(out1, blk.move21, beta2, f"{p}.move21")
        if bn_flavour:
            used_convs = {"binary_3x3"}
        if cin == cout:
            t = bconv(b, blk.binary_pw, alpha, f"{p}.binary_pw", blk.bn2 if bn_flavour else None)
            t = g.add_node("Add", [t, out1], name=f"{p}.add2")
            if bn_flavour:
                used_convs |= {"binary_pw"}
        elif cout in (2 * cin, 4 * cin):
            _expect(bn_flavour or cout == 2 * cin, f"{p}: planes {cout} != 2 * inplanes {cin}")
            parts = []
            for j in range(1, cout // cin + 1):
                parts.append(g.add_node("Add", [bconv(b, getattr(blk, f"binary_pw_down{j}"), alpha, f"{p}.binary_pw_down{j}",
                                                      getattr(blk, f"bn2_{j}") if bn_flavour else None), out1], name=f"{p}.add2_{j}"))
                if bn_flavour:
                    used_convs |= {f"binary_pw_down{j}"}
            t = g.add_node("Concat", parts, {"axis": -1}, name=f"{p}.cat")
        else:   # BiMatting mobilenet.py:166-172: planes == inplanes // 2
            _expect(bn_flavour and 2 * cout == cin and _cls(blk.binary_pw_down1) == "BiConv2d",
                    f"{p}: planes {cout} vs inplanes {cin} not supported")
            n_pix = hw[0] * hw[1]
            r = g.add_node("Reshape", [out1], {"shape": [1, n_pix, 2, cout]}, name=f"{p}.split")
            r = g.add_node("AvgPool2D", [r], {"kernel": [1, 2], "strides": [1, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1},
                           name=f"{p}.split_sum_half")
            half = g.add_node("Reshape", [r], {"shape": [1, hw[0], hw[1], cout]}, name=f"{p}.split_sum")
            t = bconv(b, blk.binary_pw_down1, alpha, f"{p}.binary_pw_down1", blk.bn2_1)
            t = g.add_node("Add", [t, half], name=f"{p}.add2")
            used_convs |= {"binary_pw_down1"}
            precision.append({"layer": f"{p}.split_sum", "op": "Reshape + AvgPool2D[1,2] + Reshape",
                              "source": "sum(torch.split(out1, planes, dim=1)) / 2", "weights": "-", "activations": "fp32"})
        x = act_tail(t, blk, "move22", "prelu2", "move23", p)
        if bn_flavour:
            covered |= {f"{p}.{m}" for m in ("move11", "move12", "move13", "move21", "move22", "move23", "prelu1", "prelu2", "bn1")}
            covered |= {f"{p}.{c}" for c in used_convs}
            covered |= {f"{p}.{c.replace('binary_pw_down', 'bn2_').replace('binary_pw', 'bn2')}" for c in used_convs if c != "binary_3x3"}
        # BiMobileNet.forward_single_frame macro residuals (mobilenet.py:206-217)
        if bi in macro_first:
            macro_feat[macro_first[bi]] = (x, hw)
        elif bi in macro_feat:
            f, fhw = macro_feat.pop(bi)
            if fhw[0] != hw[0]:
                fp = avgpool(f, net.pooling, f"macro{bi}.pooling", fhw)
                _expect((fhw[0] // 2, fhw[1] // 2) == hw, f"macro residual into {p}: size {fhw} -> {hw}")
                x = g.add_node("Add", [x, fp], name=f"macro{bi}.add")
                precision.append({"layer": f"macro{bi}.add", "op": "AvgPool2D + Add",
                                  "source": "BiMobileNet.forward_single_frame x += pooling(feature)", "weights": "-", "activations": "fp32"})

    _expect(_cls(pool1) == "AdaptiveAvgPool2d" and _pair(pool1.output_size) == (1, 1), "pool1: expected AdaptiveAvgPool2d(1)")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="pool1")
    if head_kind == "mlp":
        _expect(_cls(fc) == "Sequential" and len(fc) == 2 and _cls(fc[0]) == "Linear" and _cls(fc[1]) == "ReLU", "fc: expected Linear + ReLU")
        lin = fc[0]
        x = g.add_node("Dense", [x, g.add_constant("fc.0/kernel", lin.weight.detach().numpy().astype(np.float32)),
                                 g.add_constant("fc.0/bias", lin.bias.detach().numpy().astype(np.float32))],
                       {"activation": "none"}, name="fc.0")
        x = g.add_node("Activation", [x], {"type": "relu"}, name="fc.1")
        precision.append({"layer": "fc", "op": "Dense + relu", "source": "Linear + ReLU (BURN backbone MLP)", "weights": "fp32", "activations": "fp32"})
    elif _cls(fc) == "Identity":
        _expect(bn_flavour, "fc: Identity only for the feature (linear-eval) output of a backbone")
        precision.append({"layer": "pool1", "op": "GlobalAvgPool", "source": "AdaptiveAvgPool2d (feature output; fc = Identity)",
                          "weights": "-", "activations": "fp32"})
    else:
        _expect(_cls(fc) == "Linear" and fc.bias is not None, f"fc: unexpected {_cls(fc)}")
        x = g.add_node("Dense", [x, g.add_constant("fc/kernel", fc.weight.detach().numpy().astype(np.float32)),
                                 g.add_constant("fc/bias", fc.bias.detach().numpy().astype(np.float32))],
                       {"activation": "none"}, name="fc")
        precision.append({"layer": "fc", "op": "Dense", "source": "Linear", "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)

    if bn_flavour:
        known = set(REACTA_BN_MODELS) | {"BiMobileNet", "ModuleList", "firstconv3x3", "BasicBlock", "ReActNetBlock", "Conv2d",
                                         "BatchNorm2d", "BinaryConv2d", "HardBinaryConv", "BiConv2d", "LearnableBias",
                                         "BinaryActivation", "PReLU", "AvgPool2d", "AdaptiveAvgPool2d", "Linear", "ReLU",
                                         "Sequential", "Identity"}
        for n, m in model.named_modules():
            _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
        pre = "backbone." if top == "BiMobileNetClassifier" else ""
        covered |= {"feature.0.conv1", "feature.0.bn1"}
        head_mods = {"fc.0"} if head_kind == "mlp" else ({"fc"} if _cls(fc) == "Linear" else set())
        for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
            if n.endswith("num_batches_tracked"):
                continue
            if n.rsplit(".", 1)[0] in head_mods:
                continue
            _expect(n.startswith(pre + "feature."), f"parameter/buffer {n} not covered by the adapter")
            n = n[len(pre):]
            mod_path, leaf = n.rsplit(".", 1)
            q = mod_path.split(".")
            if mod_path in covered and leaf != "channel_threshold":
                continue
            # never read by the eval forward: BiConv2d.channel_threshold (mobilenet.py:12, unused in forward) and, in
            # S2-BNN ReActNetBlock, the binary_pw / binary_pw_down* convs + BNs of the branch the block does not take
            _expect(leaf == "channel_threshold" or (len(q) == 3 and (q[2].startswith("binary_pw") or q[2].startswith("bn2"))),
                    f"parameter/buffer {pre}{n} not covered by the adapter")
            unused.append(n)
    else:
        known = {"Reactnet", "ModuleList", "firstconv3x3", "BasicBlock", "ScaledStdConv2d", "HardBinaryScaledStdConv2d",
                 "LearnableBias", "BinaryActivation", "Q_PReLU", "AvgPool2d", "AdaptiveAvgPool2d", "Linear"}
        for n, m in model.named_modules():
            _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
        used = {"binary_3x3", "binary_pw", "binary_pw_down1", "binary_pw_down2", "move11", "move12", "move13", "move21",
                "move22", "move23", "prelu1", "prelu2", "conv1"}
        for n, _ in model.named_parameters():
            q = n.split(".")
            _expect(q[0] == "fc" or (q[0] == "feature" and q[2] in used), f"parameter {n} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "float64_constants": exact}
    if bn_flavour:
        info["unused_in_forward"] = unused
        info["refolded_thresholds"] = refolded
        info["binarization_points"] = [n.name for n in g.nodes if n.op == "ChannelAffine" and n.name.endswith((".move11", ".move21"))]
    return g, info
