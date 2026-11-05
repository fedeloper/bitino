"""Adapter: instantiated bitorch model-hub model (hpi-xnor/bitorch 0.3.0) -> Bitino IR.

Models: ResnetE18, DenseNet28 (BinaryDenseNet-28) and MeliusNet22, as published on the bitorch model hub
(Lightning 1.6.4 checkpoints; the model code is bitorch 0.3.0, the release whose BaseNetDense keeps the
``current_dense_block`` attribute that the checkpoints contain). bitorch is GPL-3.0: this adapter does not
copy its code, it reads the *instantiated* nn.Module and re-implements each class's forward as IR nodes,
raising on any module type, attribute or setting it does not know.

Semantics captured (bitorch 0.3.0 file names):
  * QConv2d (layers/qconv2d.py, QConv2dBase + QConv2d_NoAct), eval:
      a = QActivation(x) = Sign(x)                      (layers/qactivation.py; gradient cancellation is
                                                          an identity in forward)
      y = conv2d(pad(a, value=-1), Sign(W), stride, padding=0, dilation, groups), no bias, no scale
    with Sign = quantizations/sign.py SignFunction: torch.sign then 0 -> +1, i.e. "x >= 0 ? +1 : -1"
    (also for -0.0). ProgressiveSign (the training-time weight quantizer of ResNetE-18 / MeliusNet-22)
    returns the same SignFunction when not training. The padding value is applied *after* binarization:
    -> BConv2D(input_binarize "ge", pad_value -1, weights = (W >= 0 ? +1 : -1)). This is exactly the
    deployable Bitino contract: there is no sign_zero variant, because "ge" *is* the source semantics.
  * BatchNorm2d (eval, running stats) -> ChannelAffine(scale, shift), folded in float64, rounded once.
  * Conv2d (fp32 stem / transition 1x1 / ResNetE downsample 1x1), ReLU, MaxPool2d, AvgPool2d,
    AdaptiveAvgPool2d(1) + Flatten, Linear: direct mappings.
  * ResNetE BasicBlock (models/resnet_e.py): body = QConv2d(3x3, stride) -> BN;
    residual = AvgPool2d(2, stride) -> Conv2d 1x1 -> BN if in != out channels else x; out = body + residual.
    Its ``activation`` (nn.ReLU) is constructed but not used in forward.
  * DenseLayer (models/densenet.py, bn_size 0): out = cat([x, QConv2d(BN(x))], channels) -> Concat.
  * ImprovementBlock (models/meliusnet.py): y = QConv2d(BN(x)); out = x + y when channels match, else
    cat(x[:, :C-c], x[:, C-c:] + y) = x + pad(y, C-c channels in front, value 0) -> Pad + Add (adding +0.0
    leaves every value unchanged except -0.0 -> +0.0, which the following BatchNorm maps to the same value).
  * Transition (DOWNSAMPLE_STRUCT "bn,max_pool,relu,fp_conv"): BN -> MaxPool2d(2) -> ReLU -> Conv2d 1x1.

Node names are the module paths of the wrapper Model (e.g. ``_model.features.DenseBlock_1.DenseLayer_1.features.1``),
so tools can hook the source module that feeds each BConv2D.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

BINARIZE_MODES = ("ge",)
SIGN_CLASSES = ("Sign", "ProgressiveSign")


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


def _is_qconv(m) -> bool:
    return any(c.__name__ == "QConv2dBase" for c in type(m).__mro__)


def convert(model, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (224, 224),
            binarize: str = "ge") -> tuple[Graph, dict]:
    """model: the bitorch Model wrapper (ResnetE18 / DenseNet28 / MeliusNet22) or its ``_model``."""
    import torch

    if binarize not in BINARIZE_MODES:
        raise ValueError(f"bitorch binarization is sign(0) = +1: binarize must be one of {BINARIZE_MODES}")
    _expect(not model.training, "model must be in eval mode")
    prefix = "_model." if hasattr(model, "_model") else ""
    net = model._model if prefix else model
    top = _cls(net)
    _expect(top in ("_ResnetE", "_DenseNet", "_MeliusNet"), f"unsupported bitorch network class {top}")

    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    visited: set[int] = set()
    unused: list[str] = []

    def folded(name: str, value64: np.ndarray, layout: str = "") -> str:
        exact[name] = np.asarray(value64, np.float64)
        return g.add_constant(name, np.asarray(value64, np.float64).astype(np.float32), layout=layout)

    def bn(x: str, m, name: str) -> str:
        _expect(_cls(m) == "BatchNorm2d" and m.track_running_stats and m.running_mean is not None, f"{name}: unexpected {m}")
        gamma = m.weight.detach().to(torch.float64) if m.affine else torch.ones_like(m.running_mean, dtype=torch.float64)
        beta = m.bias.detach().to(torch.float64) if m.affine else torch.zeros_like(m.running_mean, dtype=torch.float64)
        inv = 1.0 / torch.sqrt(m.running_var.detach().to(torch.float64) + m.eps)
        scale = (gamma * inv).numpy()
        shift = (beta - m.running_mean.detach().to(torch.float64) * gamma * inv).numpy()
        precision.append({"layer": name, "op": "BatchNorm", "weights": "fp32 (folded in float64)", "activations": "fp32"})
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", shift)], name=name)

    def conv(x: str, m, name: str) -> str:
        _expect(_cls(m) == "Conv2d" and m.padding_mode == "zeros", f"{name}: unexpected {m}")
        (sh, sw), (ph, pw), (dh, dw) = _pair(m.stride), _pair(m.padding), _pair(m.dilation)
        w = _to_ohwi(m.weight.detach().numpy().astype(np.float32))
        b = g.add_constant(f"{name}/bias", m.bias.detach().numpy().astype(np.float32)) if m.bias is not None else ""
        precision.append({"layer": name, "op": "Conv2D", "weights": "fp32", "activations": "fp32", "groups": int(m.groups)})
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", w, layout="OHWI"), b],
                          {"strides": [sh, sw], "dilations": [dh, dw], "pads": [ph, ph, pw, pw], "groups": int(m.groups),
                           "pad_value": 0.0, "activation": "none"}, name=name)

    def qconv(x: str, m, name: str) -> str:
        _expect(_is_qconv(m), f"{name}: expected a bitorch QConv2d, got {_cls(m)}")
        act = m.activation
        _expect(_cls(act) == "QActivation" and _cls(act.activation_function) in SIGN_CLASSES,
                f"{name}: input quantization {_cls(getattr(act, 'activation_function', None))} is not sign")
        _expect(_cls(m._weight_quantize) in SIGN_CLASSES and not m._weight_quantize.training,
                f"{name}: weight quantization {_cls(m._weight_quantize)} is not sign (eval)")
        _expect(m.bias is None and m.padding_mode == "zeros" and float(m._pad_value) == -1.0,
                f"{name}: bias / padding_mode / pad_value {m._pad_value} not supported")
        visited.update({id(act), id(act.activation_function), id(m._weight_quantize)})
        w = m.weight.detach()
        # Sign(W) exactly as the source (SignFunction: torch.sign, 0 -> +1); checked against the live quantizer
        bits = torch.where(w >= 0, 1.0, -1.0)
        with torch.no_grad():
            src = m._weight_quantize(w)
        _expect(bool(torch.equal(src, bits)), f"{name}: weight quantizer output is not (W >= 0 ? +1 : -1)")
        probe = torch.tensor([-2.0, -1e-30, -0.0, 0.0, 1e-30, 3.0])
        with torch.no_grad():
            _expect(torch.equal(act(probe), torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])),
                    f"{name}: input quantizer is not (x >= 0 ? +1 : -1)")
        (sh, sw), (ph, pw), (dh, dw) = _pair(m.stride), _pair(m.padding), _pair(m.dilation)
        o, i, kh, kw = w.shape
        node = g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", _to_ohwi(bits.numpy().astype(np.int8)), dtype="bit",
                                                        layout="OHWI_bit"), "", "", ""],
                          {"strides": [sh, sw], "dilations": [dh, dw], "pads": [ph, ph, pw, pw], "groups": int(m.groups),
                           "pad_value": -1.0, "input_binarize": binarize, "activation": "none", "output": "f32"}, name=name)
        precision.append({"layer": name, "op": "BConv2D", "source": "bitorch QConv2d", "weights": "binary (sign, 0 -> +1), no scale",
                          "activations": "binary (sign, 0 -> +1)", "pad_value": -1.0, "kernel": [kh, kw], "stride": sh,
                          "input_binarize": binarize})
        diag.append({"layer": name, "n_weights": int(w.numel()), "exact_zero_latent_weights": int((w == 0).sum()),
                     "weight_quantizer": _cls(m._weight_quantize)})
        return node

    def seq(x: str, mods, name: str) -> str:
        for k, sub in mods.named_children():
            x = layer(x, sub, f"{name}.{k}")
        return x

    def layer(x: str, m, name: str) -> str:
        visited.add(id(m))
        if _cls(m) == "LayerContainer":   # bitorch layer registry wrapper (layers/extensions/layer_container.py)
            m = m._layer_implementation
            visited.add(id(m))
        c = _cls(m)
        if c == "Sequential":
            return seq(x, m, name)
        if _is_qconv(m):
            return qconv(x, m, name)
        if c == "Conv2d":
            return conv(x, m, name)
        if c == "BatchNorm2d":
            return bn(x, m, name)
        if c == "ReLU":
            return g.add_node("Activation", [x], {"type": "relu"}, name=name)
        if c in ("MaxPool2d", "AvgPool2d"):
            k, s, p = _pair(m.kernel_size), _pair(m.stride if m.stride is not None else m.kernel_size), _pair(m.padding)
            _expect(not m.ceil_mode, f"{name}: ceil_mode not supported")
            if c == "MaxPool2d":
                _expect(_pair(m.dilation) == (1, 1), f"{name}: dilation not supported")
                return g.add_node("MaxPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [p[0], p[0], p[1], p[1]]}, name=name)
            _expect(m.divisor_override is None, f"{name}: divisor_override not supported")
            return g.add_node("AvgPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [p[0], p[0], p[1], p[1]],
                                                 "count_include_pad": 1 if m.count_include_pad else 0}, name=name)
        if c == "AdaptiveAvgPool2d":
            _expect(_pair(m.output_size) == (1, 1), f"{name}: output_size {m.output_size}")
            return g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name=name)
        if c == "Flatten":
            return x          # after GlobalAvgPool(keepdims=0) the tensor is already [N, C]
        if c == "DenseLayer":
            _expect(not m.dropout, f"{name}: dropout")
            y = seq(x, m.features, f"{name}.features")
            return g.add_node("Concat", [x, y], {"axis": -1}, name=f"{name}.cat")
        if c == "ImprovementBlock":
            y = seq(x, m.body, f"{name}.body")
            if not m.use_sliced_addition:
                return g.add_node("Add", [x, y], name=f"{name}.add")
            c_in = int(m.slices[-1])
            c_y = c_in - int(m.slices[1])
            _expect(list(m.slices) == [0, c_in - c_y, c_in] and list(m.slices_add_x) == [False, True],
                    f"{name}: unexpected slices {m.slices} / {m.slices_add_x}")
            yp = g.add_node("Pad", [y], {"pads": [0, 0, 0, 0, 0, 0, c_in - c_y, 0], "value": 0.0}, name=f"{name}.pad")
            return g.add_node("Add", [x, yp], name=f"{name}.add")
        if c == "BasicBlock":
            res = x
            if m.shall_downsample:
                res = seq(x, m.downsample, f"{name}.downsample")
            else:
                visited.add(id(m.downsample))
                unused.append(f"{name}.downsample (empty nn.Module)")
            visited.add(id(m.activation))
            unused.append(f"{name}.activation (nn.ReLU, not called in BasicBlock.forward)")
            y = seq(x, m.body, f"{name}.body")
            return g.add_node("Add", [y, res], name=f"{name}.add")
        raise NotImplementedError(f"{name}: unsupported module type {c}")

    x = g.add_input(input_name, [1, input_hw[0], input_hw[1], 3])
    x = layer(x, net.features, f"{prefix}features")
    if top == "_ResnetE":
        head = net.output_layer
        hname = f"{prefix}output_layer"
    else:
        x = layer(x, net.finalize, f"{prefix}finalize")
        head = net.output
        hname = f"{prefix}output"
    _expect(_cls(head) == "Linear" and head.bias is not None, f"{hname}: unexpected {head}")
    visited.add(id(head))
    x = g.add_node("Dense", [x, g.add_constant(f"{hname}/kernel", head.weight.detach().numpy().astype(np.float32)),
                             g.add_constant(f"{hname}/bias", head.bias.detach().numpy().astype(np.float32))],
                   {"activation": "none"}, name=hname)
    precision.append({"layer": hname, "op": "Dense", "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)

    # every module with parameters or buffers must have been mapped (current_dense_block aliases the last DenseBlock)
    for n, m in model.named_modules():
        if id(m) in visited:
            continue
        own = list(m.parameters(recurse=False)) + [b for b in m.buffers(recurse=False) if b is not None]
        _expect(not own and _cls(m) in ("Sequential", "Module", "QActivation", "Sign", "ProgressiveSign", "ResnetE18", "DenseNet28",
                                        "MeliusNet22", top), f"module {n or '<root>'} ({_cls(m)}) not covered by the adapter")
    info ={"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "float64_constants": exact, "unused_in_forward": unused}
    return g, info
