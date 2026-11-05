"""Adapter: instantiated official RBONN PyTorch model (SteveTsui/RBONN, ECCV'22) -> Bitino IR.

Source: https://github.com/SteveTsui/RBONN (commit c3ae0ba), 2_step2_rbonn/rbonn.py, class
BiRealNet built by rbonn18() (ReActNet/Bi-Real ResNet-18 backbone, [4,4,4,4] single-conv
basic blocks). The adapter reads the *instantiated* nn.Module (official class definitions,
weights loaded with torch.load(weights_only=True)); it mirrors BiRealNet.forward and
BasicBlock.forward explicitly and raises on any module type or setting it does not know.

Semantics captured (rbonn.py line numbers at c3ae0ba):
  * Stem (181-184, 216-218): Conv2d 7x7 s2 p3 (no bias) + BN + MaxPool2d(3, 2, 1); no activation.
  * BasicBlock (140-173): out = BN(RBOConv(BinaryActivation(x + move0))) + residual, then
    + move1, PReLU, + move2. LearnableBias -> ChannelAffine(bias only).
  * BinaryActivation (19-34), eval: sign(x).detach() - out3.detach() + out3 with out3 the
    piecewise-quadratic ApproxSign of x, |out3| <= 1 and sign(out3) = sign(x); hence
    fl(fl(s - out3) + out3) == s exactly (Sterbenz for |out3| >= 1/2, below that the error of
    s - out3 is at most half an ulp of 1) -> torch.sign(x), sign(0) = 0 -> BConv2D
    input_binarize "sign_zero" (exact) or "ge" (deployable, 0 -> +1). The converter also
    checks this numerically with forward hooks.
  * RBOConv (89-137), eval: F.conv2d(x, sign(W) * 0.02 * |scale_factor|) (the recurrent
    module is training-only), zero padding of the binarized input -> BConv2D pad_value 0,
    sign(W) bits, post_mul = 0.02 * |scale_factor| (float64, rounded once). A latent weight
    that is exactly 0 (torch.sign -> 0 weight) is rejected.
  * Downsample (_make_layer 192-213): AvgPool2d(2, 2) + FP conv1x1 + BN -> AvgPool2D +
    Conv2D + ChannelAffine (or conv1x1 + BN without pooling).
  * BatchNorm2d (eval, running stats) -> ChannelAffine(scale, shift) folded in float64.
  * Head (225-227): AdaptiveAvgPool2d(1) + Linear -> GlobalAvgPool + Dense.

Also converts the same backbone from yikaiw/SNN (ICCV'21; commit 073c3b5), whose blocks use
ReActNet's ``HardBinaryConv`` (step1/birealnet.py:48-69, step2/birealnet.py:44-64):
  * W1A1: F.conv2d(x, s * sign(W) - clamp(W) + clamp(W)) with s = mean|W| per output channel
    (nested torch.mean over kw, kh, cin). The STE terms cancel up to fp32 rounding (exactly in
    the limit of exact arithmetic) -> sign(W) bits, post_mul = s (float64, rounded once).
  * 0.67-bit (step2, ``Binarize`` with 2^bit_num 3x3 patterns): every 3x3 kernel is replaced by
    its nearest stored pattern (``weights2patterns``: L2 argmin over sign(patterns)); the decoded
    kernels are +-1, so the forward is a strict W1A1 conv with post_mul = s. The decode calls the
    source's own ``weights2patterns`` on the fp32 weights. Requires |pattern| >= 1e-3 everywhere
    (else ``memory`` falls back to a random init), distinct patterns and |pattern| <= 3 (else the
    every-25-steps ``check_conpensate_patterns`` in forward would modify them).

Also converts the IDa-Det BiRes-18 ImageNet backbone (SteveTsui/IDa-Det, ECCV'22; commit 2a00945):
``mmdet/models/backbones/resnet_react.py`` ``ResNet_REACT(depth=18)`` with
``conv_cfg=dict(type='BiConv')`` (configs/voc_faster_rcnn_bir18.py), executed from the source file with
the mmcv/mmdet registry plumbing stubbed (tools/convert_torch_cnn.py ``_load_idadet``), plus the
checkpoint's classifier (AdaptiveAvgPool2d(1) + Linear(512, 1000), not part of the detection backbone;
class ``ResNetREACTClassifier``). Differences to RBONN:
  * the binary conv is ``block.conv``, a ``BiConv`` (mmcv-0.5.1/mmcv/cnn/bricks/BiConv.py:79-111) that
    binarizes its own input with the same BinaryActivation (``binfunc``) and convolves with
    ``Binarization`` = sign(W) * 0.02 * |scale| (BiConv.py:43-58), **with** a trained conv bias ->
    BConv2D post_add = bias (applied after post_mul, as F.conv2d adds it to the scaled sum);
  * downsample AvgPool2d(2, 2, ceil_mode=True) (resnet_react.py:285-311): identical to floor mode for
    the even feature sizes of a 224x224 input (checked from the tracked shapes).
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


def _to_ohwi(w_oihw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(w_oihw, (0, 2, 3, 1)))


def _f64(t) -> np.ndarray:
    import torch
    return t.detach().cpu().to(torch.float64).reshape(-1).numpy()


def convert(model, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (224, 224),
            binarize: str = "sign_zero") -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths (e.g. layer2.0.move0)."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    _expect(_cls(model) in ("BiRealNet", "ResNetREACTClassifier"),
            f"unsupported top-level model class {_cls(model)} (expected RBONN BiRealNet or IDa-Det ResNetREACTClassifier)")
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

    def bias(m, x: str, name: str) -> str:
        _expect(_cls(m) == "LearnableBias", f"{name}: expected LearnableBias, got {_cls(m)}")
        visited.add(name)
        return g.add_node("ChannelAffine", [x, "", g.add_constant(f"{name}/bias", m.bias.detach().cpu().numpy().astype(np.float32).reshape(-1))],
                          name=name)

    def conv(m, x: str, name: str) -> str:
        _expect(_cls(m) == "Conv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros", f"{name}: unexpected Conv2d")
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        visited.add(name)
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", _to_ohwi(m.weight.detach().cpu().numpy().astype(np.float32)), layout="OHWI"), ""],
                          {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                           "activation": "none"}, name=name)

    x = g.add_input(input_name, [1, input_hw[0], input_hw[1], 3])

    def out_hw(hw, k, s, p):   # spatial size after a conv / pool (floor mode)
        return tuple((v + 2 * pp - kk) // ss + 1 for v, kk, ss, pp in zip(hw, k, s, p))

    hw = out_hw(input_hw, _pair(model.conv1.kernel_size), _pair(model.conv1.stride), _pair(model.conv1.padding))
    x = conv(model.conv1, x, "conv1")
    x = bn(model.bn1, x, "bn1")
    mp = model.maxpool
    _expect(_cls(mp) == "MaxPool2d" and _pair(mp.dilation) == (1, 1) and not mp.ceil_mode, "maxpool: unexpected")
    k, s, p = _pair(mp.kernel_size), _pair(mp.stride), _pair(mp.padding)
    x = g.add_node("MaxPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [p[0], p[0], p[1], p[1]]}, name="maxpool")
    hw = out_hw(hw, k, s, p)
    precision.append({"layer": "conv1", "op": "Conv2D+ChannelAffine+MaxPool2D", "source": "Conv2d 7x7 + BN + MaxPool2d",
                      "weights": "fp32", "activations": "fp32"})

    for lname in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, lname)
        _expect(_cls(layer) == "Sequential", f"{lname}: unexpected {_cls(layer)}")
        for bi, blk in enumerate(layer):
            p = f"{lname}.{bi}"
            _expect(_cls(blk) == "BasicBlock", f"{p}: unexpected block {_cls(blk)}")
            idadet = _cls(getattr(blk, "conv", None)) == "BiConv"
            act = blk.conv.binfunc if idadet else blk.binary_activation
            act_name = f"{p}.conv.binfunc" if idadet else f"{p}.binary_activation"
            _expect(_cls(act) == "BinaryActivation", f"{act_name}: unexpected")
            visited.add(act_name)
            residual = x
            hw_in = hw
            t = bias(blk.move0, x, f"{p}.move0")
            bc = blk.conv if idadet else blk.binary_conv
            bc_name = f"{p}.conv" if idadet else f"{p}.binary_conv"
            w = bc.weight.detach().cpu()
            zeros = int((w == 0).sum())
            extra_diag = {}
            post_add = None
            if idadet:   # BiConv.py:79-111 (Binarization.forward 46-58: sign(W) * 0.02 * |scale|) + conv bias
                _expect(bc.bias is not None and bc.groups == 1 and tuple(bc.dilation) == (1, 1) and bc.padding_mode == "zeros"
                        and not bc.transposed and tuple(bc.output_padding) == (0, 0), f"{bc_name}: unexpected BiConv settings")
                _expect(_cls(bc.binfunc) == "BinaryActivation", f"{bc_name}.binfunc: unexpected")
                _expect(zeros == 0, f"{bc_name}: {zeros} latent weights are exactly 0 (torch.sign gives a 0 weight)")
                _expect(tuple(bc.scale.shape) == (w.shape[0], 1, 1, 1), f"{bc_name}.scale: unexpected shape")
                sign = w.sign().numpy().astype(np.int8)
                post_mul = 0.02 * np.abs(_f64(bc.scale))
                post_add = _f64(bc.bias)
                wsrc, scale_src = "sign(W)", "0.02*|scale| (+ FP conv bias)"
                visited.add(bc_name)
                extra_diag = {"scale_zeros": int((bc.scale == 0).sum())}
            elif _cls(bc) == "RBOConv":
                _expect(bc.bias is None and bc.groups == 1 and tuple(bc.dilation) == (1, 1)
                        and bc.padding_mode == "zeros" and not bc.transposed, f"{p}.binary_conv: unexpected RBOConv settings")
                _expect(zeros == 0, f"{p}.binary_conv: {zeros} latent weights are exactly 0 (torch.sign gives a 0 weight)")
                sign = w.sign().numpy().astype(np.int8)
                post_mul = 0.02 * np.abs(_f64(bc.scale_factor))   # rbonn.py:57-63 (bin = 0.02), 125-135
                wsrc, scale_src = "sign(W)", "0.02*|scale_factor|"
                visited.update({f"{p}.binary_conv.weight", f"{p}.binary_conv.scale_factor"})
            else:
                _expect(_cls(bc) == "HardBinaryConv" and tuple(w.shape[2:]) == (3, 3), f"{p}.binary_conv: unexpected {_cls(bc)}")
                sign, extra_diag = _hard_binary_conv_bits(bc, f"{p}.binary_conv", zeros)
                import torch
                w64 = bc.weight.detach().cpu().to(torch.float64)
                post_mul = torch.mean(torch.mean(torch.mean(abs(w64), dim=3, keepdim=True), dim=2, keepdim=True),
                                      dim=1, keepdim=True).reshape(-1).numpy()
                wsrc = "sign(W)" if not hasattr(bc, "binarize") else "nearest of 2^bit_num binary 3x3 patterns"
                scale_src = "mean|W| per out-channel"
                visited.add(f"{p}.binary_conv")
            binary_weights[bc_name] = sign
            _expect(post_mul.size == w.shape[0], f"{bc_name}: scale shape")
            (sh, sw), (ph, pw) = _pair(bc.stride), _pair(bc.padding)
            t = g.add_node("BConv2D", [t, g.add_constant(f"{bc_name}/kernel_bits", _to_ohwi(sign), dtype="bit", layout="OHWI_bit"),
                                       folded(f"{bc_name}/post_mul", post_mul),
                                       "" if post_add is None else folded(f"{bc_name}/post_add", post_add), ""],
                           {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                            "input_binarize": binarize, "activation": "none", "output": "f32"}, name=bc_name)
            hw = out_hw(hw, tuple(w.shape[2:]), (sh, sw), (ph, pw))
            precision.append({"layer": bc_name, "op": "BConv2D", "source": f"BinaryActivation + {_cls(bc)}",
                              "weights": f"binary {wsrc}, per-channel {scale_src}", "activations": "binary",
                              "input_binarize": binarize, "source_binarize": "torch.sign (sign(0)=0)", "pad_value": 0.0,
                              "kernel": [int(v) for v in w.shape[2:]], "stride": sh})
            if _cls(bc) == "RBOConv":
                extra_diag = {"scale_factor_zeros": int((bc.scale_factor == 0).sum()), "unused_training_only": [f"{p}.binary_conv.u"]}
            diag.append({"layer": bc_name, "latent_weight_zeros": zeros, "n_weights": int(w.numel()), **extra_diag})
            t = bn(blk.bn1, t, f"{p}.bn1")
            if blk.downsample is not None:
                ds = blk.downsample
                _expect(_cls(ds) == "Sequential" and len(ds) in (2, 3), f"{p}.downsample: unexpected structure")
                r = x
                if len(ds) == 3:
                    ap = ds[0]
                    _expect(_cls(ap) == "AvgPool2d" and ap.divisor_override is None and _pair(ap.padding) == (0, 0),
                            f"{p}.downsample.0: unexpected {_cls(ap)}")
                    # ceil_mode (IDa-Det) equals floor mode when no window is cut off: (H - k) % s == 0
                    ck, cs = _pair(ap.kernel_size), _pair(ap.stride)
                    _expect(not ap.ceil_mode or all((v - kk) % ss == 0 for v, kk, ss in zip(hw_in, ck, cs)),
                            f"{p}.downsample.0: ceil_mode AvgPool2d on input {hw_in} would differ from floor mode")
                    r = g.add_node("AvgPool2D", [r], {"kernel": list(_pair(ap.kernel_size)), "strides": list(_pair(ap.stride)),
                                                      "pads": [0, 0, 0, 0], "count_include_pad": 1 if ap.count_include_pad else 0},
                                   name=f"{p}.downsample.0")
                r = conv(ds[-2], r, f"{p}.downsample.{len(ds) - 2}")
                residual = bn(ds[-1], r, f"{p}.downsample.{len(ds) - 1}")
                precision.append({"layer": f"{p}.downsample", "op": "AvgPool2D+Conv2D+ChannelAffine", "source": "AvgPool2d + conv1x1 + BN",
                                  "weights": "fp32", "activations": "fp32"})
            t = g.add_node("Add", [t, residual], name=f"{p}.add")
            t = bias(blk.move1, t, f"{p}.move1")
            pr = blk.prelu
            _expect(_cls(pr) == "PReLU", f"{p}.prelu: unexpected {_cls(pr)}")
            slope = pr.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
            c = sign.shape[0]
            _expect(slope.size in (1, c), f"{p}.prelu: {slope.size} slopes for {c} channels")
            visited.add(f"{p}.prelu")
            t = g.add_node("Activation", [t, g.add_constant(f"{p}.prelu/slope", np.broadcast_to(slope, (c,)).copy())], {"type": "prelu"},
                           name=f"{p}.prelu")
            x = bias(blk.move2, t, f"{p}.move2")

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

    known = {"BiRealNet", "Sequential", "BasicBlock", "Conv2d", "BatchNorm2d", "MaxPool2d", "AvgPool2d", "LearnableBias",
             "BinaryActivation", "RBOConv", "HardBinaryConv", "Binarize", "PReLU", "AdaptiveAvgPool2d", "Linear",
             "ResNetREACTClassifier", "BiConv"}
    for n, m in model.named_modules():
        _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
    unused = sorted(n for n, _ in model.named_parameters() if n.endswith("binary_conv.u"))
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        ok = n in unused or any(n == v or n.startswith(v + ".") for v in visited)
        _expect(ok, f"parameter/buffer {n} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": [f"{u} (RBOConv recurrent-module step size, training only)" for u in unused],
            "float64_constants": exact, "binary_weights": binary_weights}
    return g, info


def _hard_binary_conv_bits(bc, name: str, zeros: int) -> tuple[np.ndarray, dict]:
    """+-1 kernel bits of a ReActNet/SNN HardBinaryConv (OIHW int8), plus diagnostics."""
    import torch

    w = bc.weight.detach().cpu()
    _expect(int(bc.stride) in (1, 2) and int(bc.padding) == 1, f"{name}: unexpected stride/padding")
    if not hasattr(bc, "binarize"):
        _expect(zeros == 0, f"{name}: {zeros} latent weights are exactly 0 (torch.sign gives a 0 weight)")
        return w.sign().numpy().astype(np.int8), {}
    bz = bc.binarize
    _expect(_cls(bz) == "Binarize", f"{name}.binarize: unexpected {_cls(bz)}")
    pat = bz.patterns.detach().cpu()
    _expect(bool(torch.all(pat.abs() >= 1e-3)), f"{name}: |pattern| < 1e-3 (memory would keep a random init)")
    _expect(bool(torch.all(pat.abs() <= 3.0)), f"{name}: |pattern| > 3 (forward clamps patterns every 25 steps)")
    gl = type(bz).forward.__globals__
    bnum = gl["patterns2bnum"](pat)
    _expect(len(set(bnum.tolist())) == pat.shape[0], f"{name}: repeated sign patterns (forward would re-draw them)")
    memory = torch.sign(pat)
    with torch.no_grad():
        bw, idxs = gl["weights2patterns"](w.view(-1, 9), memory)   # source decode, fp32
    bw = bw.view(w.shape)
    _expect(bool(torch.all(bw.abs() == 1)), f"{name}: decoded weights not +-1")
    return bw.numpy().astype(np.int8), {"bit_num": int(bz.bit_num), "n_patterns": int(pat.shape[0]),
                                        "n_patterns_used": int(torch.unique(idxs).numel()),
                                        "decoded_vs_sign_latent_differences": int((bw != torch.sign(w)).sum())}
