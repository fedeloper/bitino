"""Adapter: instantiated IR-Net-family PyTorch models (ReCU, RBNN, SiMaN, BiPer) -> Bitino IR.

The four official code bases share one model zoo (``models_imagenet/resnet.py`` with
``resnet18_1w1a`` / ``resnet34_1w1a``; ``models_cifar`` with ``resnet20_1w1a``,
``resnet18_1w1a`` and ``vgg_small_1w1a``). They differ only in the weight function of
``BinarizeConv2d``, in activation centering and in the nonlinearity. The adapter reads the
*instantiated* nn.Module (official class definitions, weights loaded with
torch.load(weights_only=True)), mirrors the forward of the published code explicitly and
raises on any module type, attribute or option it does not know.

Sources (repo, file:line at the commits recorded by the converter):
  * ReCU  (yuanchunyu/ReCU mirror; official z-hXu/ReCU is 404), imagenet/modules/binarized_modules.py:15-38
  * RBNN  (lmbxmu/RBNN), imagenet/modules/binarized_modules.py:30-63
  * SiMaN (lmbxmu/SiMaN), imagenet/modules/binarized_modules.py:22-66
  * BiPer (edmav4/BiPer), imagenet/modules/binarized_modules.py:17-38

BinarizeConv2d, eval mode (output = F.conv2d(ba, bw, padding=1) * alpha):
  * Weights ``bw`` are computed from parameters only. The adapter evaluates exactly the source's
    fp32 torch expression (same ops in the same order), so the IR holds the binary weights the
    fp32 source uses:
      ReCU : sign(clamp(w1, -Q_tau, Q_tau)), w1 = (w - mean_c w) / (sqrt(var_c + 1e-5) / 2 / sqrt 2),
             Q_tau = -mean|w1| * log(2 - 2 tau)
      RBNN : sign(w2 + |sin(rotate)| * (R1^T X R2 - w2)), w2 = (w - mean_c w) / std_c, X = w2 as [O, a, b]
      SiMaN: 2B - 1 from the stored mask B (the latent weight only enters through X2 - X2.detach() = 0)
      BiPer: sign(sin(freq * w)) (stage-2 square wave; ``freq`` is a module attribute set from args)
    A bw that is not exactly +-1 (e.g. an fp32 exact zero) cannot be held in a bit tensor: the
    adapter raises. ``alpha`` [O,1,1] is an fp32 parameter -> post_mul (exact, no folding).
  * Activations: ReCU / BiPer binarize the input directly, ``sign(a)``. RBNN binarizes
    ``(a - mean_CHW a) / std_CHW a`` and SiMaN ``(a - mean_CHW a) / sqrt(var_CHW a + 1e-5)``,
    i.e. per *sample* centering; the division by a positive scalar does not change the sign, so
    ``sign(a - mean_CHW(a))`` -> Reshape [1, HWC] -> ReduceMean(axis -1, keepdims) ->
    ChannelAffine(scale -1) -> AddBroadcast(a, -mean) feeding the BConv2D. torch.sign(0) = 0:
    input_binarize "sign_zero" (exact) or "ge" (deployable, 0 -> +1).
  * Zero padding of the binarized input (padding=1) -> BConv2D pad_value 0.

Topology (published forward, verified against the module structure):
  * ImageNet ResNet-18/34 (``ResNet``): Conv 7x7 s2 + BN + act + MaxPool(3,2,1); BasicBlock
    out = act1(BN1(bconv1(x)) + downsample(x)); out = act2(BN2(bconv2(out)) + out)
    (Bi-Real shortcut around every binary conv); downsample = FP conv1x1 stride s + BN;
    head AvgPool2d(7) -> BatchNorm1d -> Linear. act: ReCU/BiPer Hardtanh (stem Hardtanh),
    RBNN Hardtanh (shared module; stem ``nonlinear1``), SiMaN PReLU (stem F.relu).
  * BatchNorm (eval, running stats) -> ChannelAffine folded in float64 (BN1d on [1, C]).
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

BINARIZE_MODES = ("sign_zero", "ge")
VGG_CLASSES = ("VGG", "VGG_SMALL_1W1A")

# CIFAR ResNets (models_cifar/resnet.py = ResNet-20, resnet2.py = ResNet-18, RBNN resnet_bireal.py),
# keyed by (family, defining module). Published forwards:
#   standard  : t = hardtanh(BN1(bconv1(x))); out = hardtanh(BN2(bconv2(t)) + shortcut(x))
#   bireal    : u = act1(BN1(bconv1(x)) + shortcut(x)); out = act2(BN2(bconv2(u)) + u)
#   bireal_pre: u = BN1(bconv1(x)) + shortcut(x); out = prelu(BN2(bconv2(hardtanh(u))) + u)   (ReCU-20)
# shortcut: "A" = x[:, :, ::2, ::2] + zero channel pad planes//4 per side; "avgpad" = AvgPool2d(2) +
# zero channel pad; "bconv" = BinarizeConv2d 1x1 stride s + BN. Stem: Conv 3x3 + BN (+ act).
CIFAR_RESNETS = {
    ("recu", "resnet"): {"block": "bireal_pre", "stem": "hardtanh", "shortcut": "A", "act": "hardtanh"},
    ("rbnn", "resnet"): {"block": "standard", "stem": "hardtanh", "shortcut": "A", "act": "hardtanh"},
    ("rbnn", "resnet_bireal"): {"block": "bireal", "stem": "hardtanh", "shortcut": "A", "act": "hardtanh"},
    ("siman", "resnet"): {"block": "bireal", "stem": "relu", "shortcut": "A", "act": "prelu"},
    ("biper", "resnet"): {"block": "bireal", "stem": "hardtanh", "shortcut": "avgpad", "act": "hardtanh"},
    ("recu", "resnet2"): {"block": "standard", "stem": None, "shortcut": "bconv", "act": "hardtanh"},
    ("rbnn", "resnet2"): {"block": "standard", "stem": None, "shortcut": "bconv", "act": "hardtanh"},
    ("siman", "resnet2"): {"block": "standard", "stem": None, "shortcut": "bconv", "act": "hardtanh"},
    ("biper", "resnet2"): {"block": "standard", "stem": None, "shortcut": "avgpad", "act": "hardtanh"},
}


def _cifar_spec(family: str, model):
    mod = type(model).__module__.rsplit(".", 1)[-1]
    return CIFAR_RESNETS.get((family, mod)), mod   # vgg_small_1w1a(): ReCU/BiPer class VGG, RBNN/SiMaN class VGG_SMALL_1W1A

# per family: activation centering, nonlinearity class of the blocks, stem activation
FAMILIES = {
    "recu": {"center": False, "block_act": "Hardtanh", "stem": "nonlinear"},
    "biper": {"center": False, "block_act": "Hardtanh", "stem": "nonlinear"},
    "rbnn": {"center": True, "block_act": "Hardtanh", "stem": "nonlinear1"},
    "siman": {"center": True, "block_act": "PReLU", "stem": "relu"},
    # third-party Bi-Real-20 (ShuaiZ1037/bnn-xnor-bireal, binary_model/{resnet_bireal,binary_modules}.py):
    # BiRealConv = F.conv2d(sign(x), mean|W| * sign(W) - clamp(W) + clamp(W)); block out = BN(conv(x)) + shortcut
    "shuaiz": {"center": False, "block_act": None, "stem": None},
}


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


def source_args(m):
    """The module-global ``args`` (utils.options) the source BinarizeConv2d reads at run time."""
    return type(m).forward.__globals__.get("args")


def binary_weight(family: str, m):
    """bw exactly as the source BinarizeConv2d.forward computes it in eval mode, in the dtype of
    the module's parameters (fp32 for the converted model). Returns a torch tensor [O,I,kh,kw]."""
    import torch

    with torch.no_grad():
        w = m.weight.detach()
        if family == "recu":
            w0 = w - w.mean([1, 2, 3], keepdim=True)
            w1 = w0 / (torch.sqrt(w0.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
            ew = torch.mean(torch.abs(w1))
            q_tau = (- ew * torch.log(2 - 2 * m.tau)).detach().cpu().item()
            return torch.sign(torch.clamp(w1, -q_tau, q_tau))
        if family == "biper":
            _expect(bool(m.stage2), "BiPer BinarizeConv2d.stage2 is False (stage-1 weights are sin(freq*w), not binary)")
            return torch.sign(torch.sin(m.freq * w))
        if family == "rbnn":
            _expect(m.epoch == -1, "RBNN BinarizeConv2d.epoch != -1 (rotation update would run in forward)")
            w1 = w - w.mean([1, 2, 3], keepdim=True)
            w2 = w1 / w1.std([1, 2, 3], keepdim=True)
            x = w2.view(w.shape[0], m.a, m.b)
            rweight = ((m.R1.t()) @ x @ (m.R2)).view_as(w)
            delta = rweight.detach() - w2
            w3 = w2 + torch.abs(torch.sin(m.rotate)) * delta
            return torch.sign(w3)
        if family == "siman":
            x2 = w.abs().view(w.shape[0], -1).view_as(w)   # X2 - X2.detach() == 0 for finite X2
            return x2 - x2 + m.B.mul(2).sub(1).view_as(w)
    raise ValueError(family)


def convert(model, family: str, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (224, 224),
            binarize: str = "sign_zero") -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths (BConv2D = the BinarizeConv2d path)."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    _expect(family in FAMILIES, f"unknown IR-Net family {family!r}")
    spec = FAMILIES[family]
    _expect(not model.training, "model must be in eval mode")
    cifar_spec, cifar_mod = _cifar_spec(family, model) if _cls(model) == "ResNet" and not hasattr(model, "maxpool") else (None, None)
    _expect((_cls(model) == "ResNet" and hasattr(model, "layer4") and hasattr(model, "maxpool")) or _cls(model) in VGG_CLASSES
            or cifar_spec is not None or family == "shuaiz",
            f"unsupported top-level model {_cls(model)} (expected ImageNet ResNet, CIFAR ResNet-18/20 or VGG-Small)")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    visited: set[str] = set()
    binary_weights: dict[str, np.ndarray] = {}

    def folded(name: str, value64: np.ndarray) -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, np.asarray(value64).astype(np.float32))

    def bn(m, x: str, name: str, cls: str = "BatchNorm2d") -> str:
        _expect(_cls(m) == cls and m.affine and m.running_mean is not None, f"{name}: expected {cls} with running stats")
        scale = _f64(m.weight) / np.sqrt(_f64(m.running_var) + float(m.eps))
        shift = _f64(m.bias) - _f64(m.running_mean) * scale
        visited.add(name)
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", shift)], name=name)

    def conv(m, x: str, name: str) -> str:
        _expect(_cls(m) == "Conv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros", f"{name}: unexpected Conv2d")
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        visited.add(name)
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", _to_ohwi(m.weight.detach().cpu().numpy().astype(np.float32)), layout="OHWI"), ""],
                          {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                           "activation": "none"}, name=name)

    def act(m, x: str, name: str, channels: int, expect: str | None = None) -> str:
        _expect(_cls(m) == (expect or spec["block_act"]), f"{name}: expected {expect or spec['block_act']}, got {_cls(m)}")
        if _cls(m) == "Hardtanh":
            _expect(_cls(m) == "Hardtanh" and float(m.min_val) == -1.0 and float(m.max_val) == 1.0, f"{name}: expected Hardtanh(-1, 1), got {_cls(m)}")
            visited.add(name)
            return g.add_node("Activation", [x], {"type": "hardtanh", "min": -1.0, "max": 1.0}, name=g.unique(name))
        _expect(_cls(m) == "PReLU", f"{name}: expected PReLU, got {_cls(m)}")
        slope = m.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
        _expect(slope.size in (1, channels), f"{name}: {slope.size} slopes for {channels} channels")
        visited.add(name)
        return g.add_node("Activation", [x, g.add_constant(f"{name}/slope", np.broadcast_to(slope, (channels,)).copy())], {"type": "prelu"},
                          name=name)

    def bconv(m, x: str, name: str, hw: tuple[int, int], cin: int) -> str:
        if family == "shuaiz":
            return bireal_conv(m, x, name)
        _expect(_cls(m) == "BinarizeConv2d" and m.bias is None and m.groups == 1 and _pair(m.dilation) == (1, 1)
                and m.padding_mode == "zeros", f"{name}: unexpected BinarizeConv2d settings")
        a = source_args(m)
        if family == "rbnn":
            _expect(a is not None and not getattr(a, "a32", False), f"{name}: RBNN args.a32 is set (W1A32)")
        t = x
        if spec["center"]:
            r = g.add_node("Reshape", [x], {"shape": [1, hw[0] * hw[1] * cin]}, name=f"{name}/flatten")
            mu = g.add_node("ReduceMean", [r], {"axis": -1, "keepdims": 1}, name=f"{name}/mean")
            neg = g.add_node("ChannelAffine", [mu, g.add_constant(f"{name}/minus_one", np.array([-1.0], np.float32)), ""], name=f"{name}/neg_mean")
            t = g.add_node("AddBroadcast", [x, neg], name=f"{name}/center")
        bw = binary_weight(family, m)
        bwn = bw.detach().cpu().numpy()
        n_zero = int(np.sum(bwn == 0))
        n_other = int(np.sum((bwn != 0) & (np.abs(bwn) != 1)))
        _expect(n_zero == 0 and n_other == 0, f"{name}: binary weights not all +-1 ({n_zero} zeros, {n_other} other values)")
        sign = bwn.astype(np.int8)
        binary_weights[name] = sign
        alpha = m.alpha.detach().cpu().numpy().astype(np.float32).reshape(-1)
        _expect(alpha.size == sign.shape[0], f"{name}: alpha shape {tuple(m.alpha.shape)}")
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        visited.add(name)
        y = g.add_node("BConv2D", [t, g.add_constant(f"{name}/kernel_bits", _to_ohwi(sign), dtype="bit", layout="OHWI_bit"),
                                   g.add_constant(f"{name}/post_mul", alpha), "", ""],
                       {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                        "input_binarize": binarize, "activation": "none", "output": "f32"}, name=name)
        src_w = {"recu": "sign(clamp(standardized w, +-Q_tau))", "biper": "sign(sin(freq * w))",
                 "rbnn": "sign(rotated standardized w)", "siman": "2B - 1 (stored mask)"}[family]
        precision.append({"layer": name, "op": "BConv2D" if not spec["center"] else "Reshape+ReduceMean+ChannelAffine+AddBroadcast+BConv2D",
                          "source": "BinarizeConv2d", "weights": f"binary {src_w}, per-channel alpha", "activations": "binary",
                          "input_binarize": binarize,
                          "source_binarize": "torch.sign(a - mean_CHW(a)) (sign(0)=0)" if spec["center"] else "torch.sign(a) (sign(0)=0)",
                          "pad_value": 0.0, "kernel": list(m.kernel_size), "stride": sh})
        d = {"layer": name, "n_weights": int(sign.size), "n_plus": int(np.sum(sign > 0))}
        if family == "biper":
            d["freq"] = float(m.freq)
            d["latent_sign_differs_from_bw"] = int((torch_sign_np(m.weight) != sign).sum())
        if family == "recu":
            d["tau"] = float(m.tau)
        diag.append(d)
        return y

    hw = _pair(input_hw)

    def bireal_conv(m, x: str, name: str) -> str:
        import torch
        # BiRealConv passes ``bias`` positionally into nn.Conv2d's ``dilation`` slot, so the module keeps a
        # (trained) bias and dilation=False; forward calls F.conv2d(bx, bw, self.bias, stride, padding)
        # without dilation (default 1). bias -> post_add (exact fp32 parameter).
        _expect(_cls(m) == "BiRealConv" and m.groups == 1 and m.padding_mode == "zeros", f"{name}: unexpected BiRealConv")
        w = m.weight.detach().cpu()
        zeros = int((w == 0).sum())
        _expect(zeros == 0, f"{name}: {zeros} latent weights are exactly 0 (torch.sign gives a 0 weight)")
        sign = w.sign().numpy().astype(np.int8)
        binary_weights[name] = sign
        w64 = w.to(torch.float64)   # alpha = mean|W| (nested means, binary_modules.py BiRealConv.forward), float64 rounded once
        alpha = torch.mean(torch.mean(torch.mean(abs(w64), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True).reshape(-1).numpy()
        (sh, sw), (ph, pw) = _pair(m.stride), _pair(m.padding)
        visited.add(name)
        precision.append({"layer": name, "op": "BConv2D", "source": "BiRealConv (BinActiveBiReal STE = torch.sign)",
                          "weights": "binary sign(W), per-channel mean|W|, conv bias -> post_add", "activations": "binary", "input_binarize": binarize,
                          "source_binarize": "torch.sign(x) (sign(0)=0)", "pad_value": 0.0, "kernel": [int(v) for v in w.shape[2:]], "stride": sh})
        diag.append({"layer": name, "latent_weight_zeros": zeros, "n_weights": int(w.numel())})
        add = g.add_constant(f"{name}/post_add", m.bias.detach().cpu().numpy().astype(np.float32)) if m.bias is not None else ""
        return g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", _to_ohwi(sign), dtype="bit", layout="OHWI_bit"),
                                      folded(f"{name}/post_mul", alpha), add, ""],
                          {"strides": [sh, sw], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0,
                           "input_binarize": binarize, "activation": "none", "output": "f32"}, name=name)

    def fact(x: str, kind: str, name: str) -> str:   # functional activation (no module)
        if kind == "hardtanh":
            return g.add_node("Activation", [x], {"type": "hardtanh", "min": -1.0, "max": 1.0}, name=g.unique(name))
        _expect(kind == "relu", kind)
        return g.add_node("Activation", [x], {"type": "relu"}, name=g.unique(name))

    def shortcut(blk, x: str, pth: str, cin: int, cout: int, stride: int, h: int, w: int) -> str:
        sc = blk.shortcut if hasattr(blk, "shortcut") else blk.downsample
        if sc is None:
            _expect(stride == 1 and cin == cout, f"{pth}: no shortcut module with a shape change")
            return x
        if _cls(sc) == "Sequential" and len(sc) == 0:
            _expect(stride == 1 and cin == cout, f"{pth}: identity shortcut with a shape change")
            return x
        kind = cifar_spec["shortcut"]
        if kind == "bconv":
            _expect(_cls(sc) == "Sequential" and len(sc) == 2 and _cls(sc[0]) == "BinarizeConv2d" and _pair(sc[0].kernel_size) == (1, 1)
                    and _pair(sc[0].padding) == (0, 0), f"{pth}.shortcut: expected BinarizeConv2d 1x1 + BN")
            r = bconv(sc[0], x, f"{pth}.shortcut.0", (h, w), cin)
            return bn(sc[1], r, f"{pth}.shortcut.1")
        # parameter-free shortcuts: check the module numerically against the documented formula
        import torch
        probe = torch.randn(1, cin, h, w, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        pad = (cout - cin) // 2
        with torch.no_grad():
            got = sc.double()(probe) if kind == "avgpad" else sc(probe)
            if kind == "A":
                want = torch.nn.functional.pad(probe[:, :, ::2, ::2], (0, 0, 0, 0, pad, pad))
            else:
                want = torch.nn.functional.pad(torch.nn.functional.avg_pool2d(probe, 2), (0, 0, 0, 0, pad, pad))
        _expect(tuple(got.shape) == tuple(want.shape) and bool(torch.equal(got, want)), f"{pth}.shortcut does not match option {kind}")
        k = 1 if kind == "A" else 2
        r = g.add_node("AvgPool2D", [x], {"kernel": [k, k], "strides": [2, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1},
                       name=f"{pth}.shortcut.{'subsample' if kind == 'A' else 'avgpool'}")
        if pad:
            r = g.add_node("Pad", [r], {"pads": [0, 0, 0, 0, 0, 0, pad, pad], "value": 0.0}, name=f"{pth}.shortcut.pad")
        precision.append({"layer": f"{pth}.shortcut", "op": "AvgPool2D+Pad", "weights": "none", "activations": "fp32",
                          "source": "x[:, :, ::2, ::2] + F.pad channels" if kind == "A" else "AvgPool2d(2) + F.pad channels"})
        return r

    if family == "shuaiz":
        _expect(_cls(model) == "BiRealNetCifar10", f"unexpected model {_cls(model)}")
        cifar_spec = {"shortcut": "A"}
        x = g.add_input(input_name, [1, hw[0], hw[1], 3])
        x = bn(model.bn1, conv(model.conv1, x, "conv1"), "bn1")   # forward: no activation after the stem (commented out)
        precision.append({"layer": "conv1", "op": "Conv2D+ChannelAffine", "source": "Conv2d 3x3 + BN", "weights": "fp32", "activations": "fp32"})
        h, w_, c = hw[0], hw[1], model.conv1.out_channels
        for lname in ("layer1", "layer2", "layer3"):
            for bi, blk in enumerate(getattr(model, lname)):
                pth = f"{lname}.{bi}"
                _expect(_cls(blk) == "BasicBlock", f"{pth}: unexpected block {_cls(blk)}")
                cout, st = blk.binary_conv.out_channels, _pair(blk.binary_conv.stride)[0]
                u = bn(blk.bn1, bconv(blk.binary_conv, x, f"{pth}.binary_conv", (h, w_), c), f"{pth}.bn1")
                x = g.add_node("Add", [u, shortcut(blk, x, pth, c, cout, st, h, w_)], name=f"{pth}.add")
                h, w_, c = (h - 1) // st + 1, (w_ - 1) // st + 1, cout
        _expect(_cls(model.avgpool) == "AdaptiveAvgPool2d" and _pair(model.avgpool.output_size) == (1, 1), "avgpool")
        x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="avgpool")
        fc = model.fc
        visited.add("fc")
        x = g.add_node("Dense", [x, g.add_constant("fc/kernel", fc.weight.detach().cpu().numpy().astype(np.float32)),
                                 g.add_constant("fc/bias", fc.bias.detach().cpu().numpy().astype(np.float32))], {"activation": "none"}, name="fc")
        precision.append({"layer": "head", "op": "GlobalAvgPool+Dense", "source": "AdaptiveAvgPool2d(1) + Linear", "weights": "fp32", "activations": "fp32"})
    elif cifar_spec is not None:
        cs = cifar_spec
        x = g.add_input(input_name, [1, hw[0], hw[1], 3])
        x = conv(model.conv1, x, "conv1")
        x = bn(model.bn1, x, "bn1")
        if cs["stem"]:
            x = fact(x, cs["stem"], "stem_" + cs["stem"])
        precision.append({"layer": "conv1", "op": "Conv2D+ChannelAffine" + ("+Activation" if cs["stem"] else ""),
                          "source": f"Conv2d 3x3 + BN{' + F.' + cs['stem'] if cs['stem'] else ''}", "weights": "fp32", "activations": "fp32"})
        h, w_, c = hw[0], hw[1], model.conv1.out_channels
        layers = [n for n in ("layer1", "layer2", "layer3", "layer4") if hasattr(model, n)]
        for lname in layers:
            for bi, blk in enumerate(getattr(model, lname)):
                pth = f"{lname}.{bi}"
                _expect(_cls(blk) in ("BasicBlock_1w1a", "BasicBlock") and _cls(blk.conv1) == "BinarizeConv2d", f"{pth}: unexpected block {_cls(blk)}")
                cout, st = blk.conv1.out_channels, _pair(blk.conv1.stride)[0]
                ho, wo = (h - 1) // st + 1, (w_ - 1) // st + 1
                u = bn(blk.bn1, bconv(blk.conv1, x, f"{pth}.conv1", (h, w_), c), f"{pth}.bn1")
                res = shortcut(blk, x, pth, c, cout, st, h, w_)
                if cs["block"] == "standard":
                    t = fact(u, "hardtanh", f"{pth}.hardtanh1")
                    v = g.add_node("Add", [bn(blk.bn2, bconv(blk.conv2, t, f"{pth}.conv2", (ho, wo), cout), f"{pth}.bn2"), res], name=f"{pth}.add")
                    x = fact(v, "hardtanh", f"{pth}.hardtanh2")
                elif cs["block"] == "bireal_pre":
                    u = g.add_node("Add", [u, res], name=f"{pth}.add1")
                    t = fact(u, "hardtanh", f"{pth}.hardtanh")
                    v = g.add_node("Add", [bn(blk.bn2, bconv(blk.conv2, t, f"{pth}.conv2", (ho, wo), cout), f"{pth}.bn2"), u], name=f"{pth}.add2")
                    x = act(blk.prelu, v, f"{pth}.prelu", cout, expect="PReLU")
                else:   # bireal
                    u = g.add_node("Add", [u, res], name=f"{pth}.add1")
                    u = act(blk.nonlinear1, u, f"{pth}.nonlinear1", cout) if cs["act"] == "prelu" else fact(u, "hardtanh", f"{pth}.hardtanh1")
                    v = g.add_node("Add", [bn(blk.bn2, bconv(blk.conv2, u, f"{pth}.conv2", (ho, wo), cout), f"{pth}.bn2"), u], name=f"{pth}.add2")
                    x = act(blk.nonlinear2, v, f"{pth}.nonlinear2", cout) if cs["act"] == "prelu" else fact(v, "hardtanh", f"{pth}.hardtanh2")
                h, w_, c = ho, wo, cout
        if len(layers) == 4:
            _expect((h, w_) == (4, 4), f"ResNet-18 head: F.avg_pool2d(out, 4) expects a 4x4 map, got {h}x{w_}")
        x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="avgpool")
        x = bn(model.bn2, x, "bn2", cls="BatchNorm1d")
        fc = model.linear
        _expect(_cls(fc) == "Linear" and fc.bias is not None, f"linear: unexpected {_cls(fc)}")
        visited.add("linear")
        x = g.add_node("Dense", [x, g.add_constant("linear/kernel", fc.weight.detach().cpu().numpy().astype(np.float32)),
                                 g.add_constant("linear/bias", fc.bias.detach().cpu().numpy().astype(np.float32))],
                       {"activation": "none"}, name="linear")
        precision.append({"layer": "head", "op": "GlobalAvgPool+ChannelAffine+Dense", "source": "F.avg_pool2d (full map) + BatchNorm1d + Linear",
                          "weights": "fp32", "activations": "fp32"})
    elif _cls(model) in VGG_CLASSES:
        x = g.add_input(input_name, [1, hw[0], hw[1], 3])
        x = conv(model.conv0, x, "conv0")
        x = bn(model.bn0, x, "bn0")
        acts = ([(model.nonlinear, "nonlinear")] * 6 if family == "rbnn" else
                [(getattr(model, f"nonlinear{i}"), f"nonlinear{i}") for i in range(6)])
        x = act(acts[0][0], x, acts[0][1], model.conv0.out_channels)
        precision.append({"layer": "conv0", "op": "Conv2D+ChannelAffine+Activation", "source": "Conv2d 3x3 + BN + nonlinear0",
                          "weights": "fp32", "activations": "fp32"})
        h, w_, c = hw[0], hw[1], model.conv0.out_channels
        pool = model.pooling
        _expect(_cls(pool) == "MaxPool2d" and _pair(pool.kernel_size) == (2, 2) and _pair(pool.stride) == (2, 2)
                and _pair(pool.padding) == (0, 0) and not pool.ceil_mode, "pooling: expected MaxPool2d(2, 2)")
        visited.add("pooling")
        for i in range(1, 6):
            m = getattr(model, f"conv{i}")
            _expect(_pair(m.stride) == (1, 1) and _pair(m.padding) == (1, 1), f"conv{i}: unexpected stride/padding")
            x = bconv(m, x, f"conv{i}", (h, w_), c)
            c = m.out_channels
            if i in (1, 3, 5):   # VGG-Small forward: conv -> pooling -> bn -> nonlinear
                x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name=f"pooling_{i}")
                h, w_ = h // 2, w_ // 2
            x = bn(getattr(model, f"bn{i}"), x, f"bn{i}")
            x = act(acts[i][0], x, acts[i][1], c)
        fc = model.fc
        _expect(_cls(fc) == "Linear" and fc.bias is not None and fc.in_features == c * h * w_, f"fc: unexpected {_cls(fc)}")
        visited.add("fc")
        # x.view(N, -1) flattens NCHW (c, h, w); the IR flattens NHWC -> permute the fc columns
        wfc = fc.weight.detach().cpu().numpy().astype(np.float32).reshape(-1, c, h, w_).transpose(0, 2, 3, 1).reshape(fc.out_features, -1)
        x = g.add_node("Reshape", [x], {"shape": [1, h * w_ * c]}, name="flatten")
        x = g.add_node("Dense", [x, g.add_constant("fc/kernel", np.ascontiguousarray(wfc)),
                                 g.add_constant("fc/bias", fc.bias.detach().cpu().numpy().astype(np.float32))],
                       {"activation": "none"}, name="fc")
        precision.append({"layer": "head", "op": "Reshape+Dense", "source": "view(N, -1) (NCHW order) + Linear; fc columns permuted to NHWC",
                          "weights": "fp32", "activations": "fp32"})
    else:
        x = g.add_input(input_name, [1, hw[0], hw[1], 3])
        x = conv(model.conv1, x, "conv1")
        x = bn(model.bn1, x, "bn1")
        if spec["stem"] == "relu":
            x = g.add_node("Activation", [x], {"type": "relu"}, name="stem_relu")
            src_stem = "F.relu"
        else:
            x = act(getattr(model, spec["stem"]), x, spec["stem"], 64)
            src_stem = spec["stem"]
        mp = model.maxpool
        _expect(_cls(mp) == "MaxPool2d" and _pair(mp.dilation) == (1, 1) and not mp.ceil_mode, "maxpool: unexpected")
        k, s, p = _pair(mp.kernel_size), _pair(mp.stride), _pair(mp.padding)
        x = g.add_node("MaxPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [p[0], p[0], p[1], p[1]]}, name="maxpool")
        precision.append({"layer": "conv1", "op": "Conv2D+ChannelAffine+Activation+MaxPool2D", "source": f"Conv2d 7x7 + BN + {src_stem} + MaxPool2d",
                          "weights": "fp32", "activations": "fp32"})
        h = (hw[0] + 2 * 3 - 7) // 2 + 1
        h = (h + 2 * p[0] - k[0]) // s[0] + 1
        w_ = (hw[1] + 2 * 3 - 7) // 2 + 1
        w_ = (w_ + 2 * p[1] - k[1]) // s[1] + 1
        c = 64

        for lname in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(model, lname)
            _expect(_cls(layer) == "Sequential", f"{lname}: unexpected {_cls(layer)}")
            for bi, blk in enumerate(layer):
                pth = f"{lname}.{bi}"
                _expect(_cls(blk) == "BasicBlock", f"{pth}: unexpected block {_cls(blk)}")
                if spec["block_act"] == "Hardtanh" and family == "rbnn":
                    a1, a2, n1, n2 = blk.nonlinear, blk.nonlinear, f"{pth}.nonlinear", f"{pth}.nonlinear"
                else:
                    a1, a2, n1, n2 = blk.nonlinear1, blk.nonlinear2, f"{pth}.nonlinear1", f"{pth}.nonlinear2"
                cout = blk.conv1.out_channels
                st = _pair(blk.conv1.stride)[0]
                ho, wo = (h + 2 - 3) // st + 1, (w_ + 2 - 3) // st + 1
                t = bconv(blk.conv1, x, f"{pth}.conv1", (h, w_), c)
                t = bn(blk.bn1, t, f"{pth}.bn1")
                residual = x
                if blk.downsample is not None:
                    ds = blk.downsample
                    _expect(_cls(ds) == "Sequential" and len(ds) == 2, f"{pth}.downsample: unexpected structure (avg_down not supported)")
                    r = conv(ds[0], x, f"{pth}.downsample.0")
                    residual = bn(ds[1], r, f"{pth}.downsample.1")
                    precision.append({"layer": f"{pth}.downsample", "op": "Conv2D+ChannelAffine", "source": "conv1x1 stride s + BN",
                                      "weights": "fp32", "activations": "fp32"})
                else:
                    _expect(st == 1 and c == cout, f"{pth}: shape change without downsample")
                t = g.add_node("Add", [t, residual], name=f"{pth}.add1")
                t = act(a1, t, n1, cout)
                residual = t
                u = bconv(blk.conv2, t, f"{pth}.conv2", (ho, wo), cout)
                u = bn(blk.bn2, u, f"{pth}.bn2")
                u = g.add_node("Add", [u, residual], name=f"{pth}.add2")
                x = act(a2, u, n2, cout)
                h, w_, c = ho, wo, cout

        ap = model.avgpool
        _expect(_cls(ap) == "AvgPool2d" and _pair(ap.kernel_size) == (h, w_) and not ap.ceil_mode and _pair(ap.padding) == (0, 0),
                f"avgpool: expected AvgPool2d over the full {h}x{w_} map")
        visited.add("avgpool")
        x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="avgpool")
        x = bn(model.bn2, x, "bn2", cls="BatchNorm1d")
        fc = model.fc
        _expect(_cls(fc) == "Linear" and fc.bias is not None, f"fc: unexpected {_cls(fc)}")
        visited.add("fc")
        x = g.add_node("Dense", [x, g.add_constant("fc/kernel", fc.weight.detach().cpu().numpy().astype(np.float32)),
                                 g.add_constant("fc/bias", fc.bias.detach().cpu().numpy().astype(np.float32))],
                       {"activation": "none"}, name="fc")
        precision.append({"layer": "head", "op": "GlobalAvgPool+ChannelAffine+Dense", "source": "AvgPool2d(7) + BatchNorm1d + Linear",
                          "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)

    known = {"ResNet", *VGG_CLASSES, "BiRealNetCifar10", "BiRealConv", "AdaptiveAvgPool2d", "Sequential", "BasicBlock", "BasicBlock_1w1a", "LambdaLayer", "Conv2d", "BatchNorm2d", "BatchNorm1d", "MaxPool2d", "AvgPool2d",
             "BinarizeConv2d", "Hardtanh", "PReLU", "Linear"}
    for n, m in model.named_modules():
        _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
    # parameters/buffers not read by the published forward (constructed but unused modules)
    unused = []
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if any(n == v or n.startswith(v + ".") for v in visited):
            continue
        if family == "biper" and n.endswith(".beta") and n.rsplit(".", 1)[0] in visited:
            unused.append(n)            # BiPer CIFAR: BinarizeConv2d.beta (requires_grad=False) is not used in forward
            continue
        if family == "rbnn" and _cls(model) == "ResNet" and n.startswith("bn3."):
            unused.append(n)            # RBNN: ResNet.bn3 (BatchNorm1d(num_classes)) is constructed but not used in forward
            continue
        _expect(False, f"parameter/buffer {n} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": unused, "float64_constants": exact, "binary_weights": binary_weights}
    return g, info


def torch_sign_np(t) -> np.ndarray:
    import torch
    return torch.sign(t.detach().cpu()).numpy().astype(np.int8)
