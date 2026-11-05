"""Adapter: instantiated official Brevitas BNN-PYNQ model (CNV / FC, 1-bit weights) -> Bitino IR.

Source: brevitas 0.11.0 (PyPI wheel, sha256 0350a551...), ``brevitas_examples/bnn_pynq/models/{CNV,FC,
common,tensor_norm}.py``. The adapter reads the *instantiated* nn.Module (official class
definitions, checkpoint loaded with ``torch.load(weights_only=True)``), mirrors ``CNV.forward`` /
``FC.forward`` explicitly and raises on any module type, quantizer or attribute it does not know.
Quantizers are not trusted by name only: each one is *probed* (called on fixed inputs) and its
output must match the semantics below exactly.

Semantics captured (brevitas 0.11.0):
  * Input: ``x -> 2x - 1`` (``CNV.forward`` / ``FC.forward``, after ``x.view(N, -1)`` for FC).
  * Binary activation (``QuantIdentity``, ``ClampedBinaryQuant``, CONST scale 1.0):
    ``clamp(x, -1, 1)`` then ``binary_sign`` = ``torch.ge(x, 0) * 2 - 1`` (function/ops.py), so
    sign(0) = sign(-0.0) = +1 -> Bitino ``ge``.
  * Binary weights (``QuantConv2d`` / ``QuantLinear`` with ``BinaryQuant``, CONST scale 1.0, per
    tensor): ``binary_sign(W)`` -> bit tensor; the adapter takes the module's own
    ``quant_weight()`` values and asserts they are exactly +-1 and equal to ge-sign(W).
  * CNV 8-bit input (``QuantIdentity`` with ``RescalingIntQuant``, POWER_OF_TWO scale): probed to be
    ``clamp(round_half_even(x * 128), -128, 127) / 128`` (Q1.7, narrow_range False) -> the generic
    ``FakeQuant`` (docs/operators_bhvit.md) with scale 1/128, zero point 0, qmin -128, qmax 127.
  * Conv: 3x3, stride 1, **no padding** (VALID), no bias; MaxPool2d(2) after conv 2 and 4,
    applied to +-1 values. BatchNorm2d/1d (eps 1e-4 CNV, 1e-5 FC) after every conv/linear except
    the last; ``TensorNorm`` (scalar affine) after the last linear.
  * Dropout (FC) is the identity in eval mode.

Two graph variants (``fused`` argument):
  * unfused: literal mirror. Conv/BConv2D/BDense produce exact integer accumulators, BN is a
    ``ChannelAffine`` (scale = gamma/sqrt(var+eps), shift = beta - mean*scale, computed in float64,
    rounded once to float32), binary activations are ``Activation(sign)`` (ge), MaxPool2D on +-1.
  * fused: every ``BN -> sign`` becomes an **integer threshold** on the exact accumulator (FINN's
    MultiThreshold idea): with tau = mean - beta*sqrt(var+eps)/gamma (float64),
    gamma > 0: BN(acc) >= 0 <=> acc >= T = ceil(tau)  -> post_mul = +1, post_add = -T;
    gamma < 0: BN(acc) >= 0 <=> acc <= T = floor(tau) -> post_mul = -1, post_add = +T.
    acc and T are small integers, so the decision is exact in float32 and float64 and equals the
    exact-real-arithmetic BN decision of the source (up to the float64 error of tau, reported as
    the per-layer margin). A BConv2D emits bits directly when its consumer is a BConv2D; otherwise
    (MaxPool2D, flatten, and every BDense, which has no bit output) the producer emits the exact
    f32 integer s*(acc - T), whose ge-sign is the binary
    activation (ge-sign is monotone, so max-pooling commutes with it). The FC input
    ``sign(2x - 1)`` becomes the BDense threshold 0.5 (exact: 2x is exact and fl(2x - 1) >= 0 <=>
    x >= 0.5). For CNV conv1 (W1A8) the threshold is applied to 128*acc (an integer) and folded
    into the float Conv2D: weights s*w (still +-1) and bias -s*T/128; every partial sum is a
    multiple of 1/128 below 2^7, so float32 accumulation in any order is exact.
    TensorNorm folds into the last BDense: post_mul = w/sqrt(rv+eps), post_add = b - rm*post_mul.

Layouts: NCHW -> NHWC activations, OIHW -> OHWI filters. The CNV flatten happens at 1x1 spatial
size (asserted), where NCHW and NHWC flattening coincide; the FC input has C = 1.
"""
from __future__ import annotations

import math

import numpy as np

from ..ir import Graph

KNOWN_MODULES = {"CNV", "FC", "ModuleList", "QuantIdentity", "QuantConv2d", "QuantLinear", "BatchNorm1d",
                 "BatchNorm2d", "MaxPool2d", "Dropout", "TensorNorm"}


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


def _tensor_quant(mod):
    return mod.act_quant.fused_activation_quant_proxy.tensor_quant


def _check_layer_quant_off(mod, name: str) -> None:
    """QuantConv2d / QuantLinear: only the weight quantizer may be active."""
    for p in ("input_quant", "output_quant", "bias_quant"):
        proxy = getattr(mod, p, None)
        _expect(proxy is None or not proxy.is_quant_enabled, f"{name}: {p} is enabled (not supported)")
    _expect(mod.bias is None, f"{name}: bias not supported")


def probe_binary_act(mod, name: str) -> None:
    """QuantIdentity with ClampedBinaryQuant: must map x -> +1 if x >= 0 (incl. -0.0) else -1."""
    import torch

    tq = _tensor_quant(mod)
    _expect(_cls(tq) == "ClampedBinaryQuant", f"{name}: expected ClampedBinaryQuant, got {_cls(tq)}")
    _expect(_cls(mod.act_quant.fused_activation_quant_proxy.activation_impl) == "Identity", f"{name}: activation_impl must be Identity")
    for dt in (torch.float32, torch.float64):
        x = torch.tensor([-3.0, -1.0, -1e-30, -0.0, 0.0, 1e-30, 0.5, 1.0, 7.0], dtype=dt)
        with torch.no_grad():
            y = mod(x)
        y = getattr(y, "value", y)
        want = torch.tensor([-1, -1, -1, 1, 1, 1, 1, 1, 1], dtype=dt)
        _expect(torch.equal(y, want), f"{name}: binary activation probe mismatch ({dt}): {y.tolist()}")


def probe_int8_input_quant(mod, name: str) -> dict:
    """QuantIdentity with RescalingIntQuant: must equal clamp(round_half_even(x/s), qmin, qmax)*s."""
    import torch

    tq = _tensor_quant(mod)
    _expect(_cls(tq) == "RescalingIntQuant", f"{name}: expected RescalingIntQuant, got {_cls(tq)}")
    _expect(_cls(tq.int_quant.float_to_int_impl) == "RoundSte", f"{name}: rounding must be RoundSte (round half to even)")
    was = mod.return_quant_tensor
    mod.return_quant_tensor = True
    with torch.no_grad():
        qt = mod(torch.zeros(4))
    mod.return_quant_tensor = was
    scale = float(qt.scale)
    zp = float(qt.zero_point)
    bw = int(qt.bit_width)
    _expect(bool(qt.signed) and zp == 0.0, f"{name}: expected signed, zero-point 0 quantizer")
    _expect(scale > 0 and math.frexp(scale)[0] == 0.5, f"{name}: scale {scale} is not a power of two")
    qmin, qmax = -(2 ** (bw - 1)), 2 ** (bw - 1) - 1   # narrow_range=False, checked by the probe
    # Probe: ties (half-even), clamping, and every value on a 1/4-step grid.
    grid = np.concatenate([np.arange(-2.0, 2.0, scale / 4), [qmin * scale - scale / 2, qmax * scale + scale / 2]])
    for dt in (torch.float32, torch.float64):
        x = torch.tensor(grid, dtype=dt)
        with torch.no_grad():
            y = mod(x)
        y = getattr(y, "value", y).to(torch.float64).numpy()
        want = np.clip(np.rint(grid / scale), qmin, qmax) * scale
        _expect(np.array_equal(y, want), f"{name}: integer input quantizer probe mismatch ({dt})")
    return {"scale": scale, "zero_point": 0, "qmin": qmin, "qmax": qmax, "bit_width": bw, "rounding": "half_even"}


def binary_weight(mod, name: str) -> np.ndarray:
    """+-1 weights exactly as the module quantizes them (BinaryQuant, CONST scale 1)."""
    import torch

    wq = mod.weight_quant.tensor_quant
    _expect(_cls(wq) == "BinaryQuant", f"{name}: expected BinaryQuant weights, got {_cls(wq)}")
    with torch.no_grad():
        qw = mod.quant_weight()
    _expect(torch.all(qw.scale == 1.0).item(), f"{name}: weight scale {qw.scale.flatten().tolist()} != 1 (not supported)")
    v = qw.value.detach().to(torch.float64).numpy()
    _expect(np.all((v == 1.0) | (v == -1.0)), f"{name}: quantized weights are not +-1")
    latent = mod.weight.detach().to(torch.float64).numpy()
    _expect(np.array_equal(v, np.where(latent >= 0, 1.0, -1.0)), f"{name}: quantized weights != ge-sign(latent)")
    return v.astype(np.int8)


def bn_params(bn) -> dict:
    """Float64 BN folding: y = acc*scale + shift; tau = acc value where y = 0."""
    import torch

    _expect(bn.affine and bn.track_running_stats, "BatchNorm must be affine with running stats")
    g = bn.weight.detach().to(torch.float64).numpy()
    b = bn.bias.detach().to(torch.float64).numpy()
    m = bn.running_mean.detach().to(torch.float64).numpy()
    v = bn.running_var.detach().to(torch.float64).numpy()
    _expect(np.all(g != 0), "BatchNorm gamma == 0 (constant channel) not supported")
    sd = np.sqrt(v + float(bn.eps))
    scale = g / sd
    shift = b - m * scale
    tau = m - b * sd / g
    return {"scale": scale, "shift": shift, "tau": tau, "sign": np.sign(g), "eps": float(bn.eps)}


def integer_threshold(bnp: dict, k_max: int, acc_scale: int = 1, parity: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(post_mul, post_add, margin info) for BN -> ge-sign on an integer accumulator n = acc*acc_scale.

    n ranges over integers with |n| <= k_max (and n = parity mod 2 if given: a +-1 dot product of
    length K without padding has the parity of K). Returns post_mul = s and post_add = -s*T so that
    s*n - s*T >= 0 <=> BN(n) >= 0 in exact arithmetic.

    Margin: distance between tau and the closest attainable n, in n units and in BN-output units
    (|scale| * distance). `fp32_fragile_channels` counts channels where that BN-output margin is below
    2^-22 * (|n*scale| + |shift|), i.e. where a float32 evaluation of BN (as in the fp32 source or the
    unfused graph) could in principle round to the other side for that n; the fused integer
    threshold decides exactly there."""
    s = bnp["sign"]
    tau = bnp["tau"] * acc_scale
    t = np.where(s > 0, np.ceil(tau), np.floor(tau))
    t = np.clip(t, -k_max - 1, k_max + 1)          # same decisions for every attainable n
    if parity is None:
        near = np.rint(tau)
    else:
        near = 2.0 * np.rint((tau - parity) / 2.0) + parity
    near = np.clip(near, -k_max, k_max)
    dist = np.abs(near - tau)
    scale_n = np.abs(bnp["scale"]) / acc_scale
    margin_bn = dist * scale_n
    bound = 2.0 ** -22 * (np.abs(near) * scale_n + np.abs(bnp["shift"]))
    info = {"min_margin_acc_units": float(dist.min()), "min_margin_bn_units": float(margin_bn.min()),
            "n_channels_tau_integer": int(np.sum(dist == 0)),
            "n_channels_constant": int(np.sum((tau > k_max) | (tau < -k_max))),
            "fp32_fragile_channels": int(np.sum(margin_bn < bound)), "parity": parity}
    return s.astype(np.float32), (-s * t).astype(np.float32), info


def _to_ohwi(w_oihw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(w_oihw, (0, 2, 3, 1)))


def _check_modules(model) -> None:
    for n, m in model.named_modules():
        top = _cls(m)
        if n.count(".") > 1:   # brevitas internals (proxies, quantizers) below a known layer
            continue
        _expect(top in KNOWN_MODULES, f"module {n or '<root>'}: unsupported type {top}")
    _expect(not model.training, "model must be in eval mode")


def convert(model, model_id: str = "", fused: bool = True, input_name: str = "input") -> tuple[Graph, dict]:
    """Returns (graph, info). Node names are the PyTorch module paths (e.g. conv_features.4)."""
    _check_modules(model)
    if _cls(model) == "CNV":
        return _convert_cnv(model, fused, input_name)
    if _cls(model) == "FC":
        return _convert_fc(model, fused, input_name)
    raise NotImplementedError(f"unsupported top-level class {_cls(model)}")


class _Builder:
    """Shared emission of binary layers + BN + sign in both variants."""

    def __init__(self, fused: bool):
        self.g = Graph()
        self.fused = fused
        self.precision: list[dict] = []
        self.margins: list[dict] = []

    def bn_sign(self, x: str, bn, bn_name: str, act_name: str, k_max: int, producer, acc_scale: int = 1, parity: int | None = None):
        """Unfused: ChannelAffine(BN) + Activation(sign). Fused: thresholds into `producer`.
        Returns the tensor that carries the binary activation (as +-1 or a ge-sign carrier)."""
        bnp = bn_params(bn)
        mul, add, minfo = integer_threshold(bnp, k_max, acc_scale, parity)
        self.margins.append({"layer": bn_name, "k_max": k_max, **minfo})
        if not self.fused:
            g = self.g
            t = g.add_node("ChannelAffine", [x, g.add_constant(f"{bn_name}/scale", bnp["scale"].astype(np.float32)),
                                             g.add_constant(f"{bn_name}/shift", bnp["shift"].astype(np.float32))], name=bn_name)
            return g.add_node("Activation", [t], {"type": "sign"}, name=act_name)
        producer(mul, add)
        return x


def _bconv(g: Graph, x: str, w_oihw: np.ndarray, name: str) -> str:
    return g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", _to_ohwi(w_oihw), dtype="bit", layout="OHWI_bit"), "", "", ""],
                      {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0,
                       "input_binarize": "ge", "activation": "none", "output": "f32"}, name=name)


def _bdense(g: Graph, x: str, w: np.ndarray, name: str, thr: str = "") -> str:
    return g.add_node("BDense", [x, g.add_constant(f"{name}/kernel_bits", w, dtype="bit", layout="OI_bit"), "", "", thr],
                      {"input_binarize": "ge", "activation": "none"}, name=name)


def _set_post(g: Graph, name: str, mul: np.ndarray, add: np.ndarray) -> None:
    node = next(n for n in g.nodes if n.name == name)
    node.inputs[2] = g.add_constant(f"{name}/post_mul", mul)
    node.inputs[3] = g.add_constant(f"{name}/post_add", add)


def _tensornorm(b: _Builder, x: str, tn, last: str, name: str) -> str:
    import torch

    _expect(_cls(tn) == "TensorNorm", f"{name}: expected TensorNorm, got {_cls(tn)}")
    _expect(tn.weight.numel() == 1 and tn.running_var.numel() == 1, f"{name}: TensorNorm must be scalar")
    w, bb, rm, rv = (float(p.detach().to(torch.float64)) for p in (tn.weight, tn.bias, tn.running_mean, tn.running_var))
    mul = w / math.sqrt(rv + float(tn.eps))
    add = bb - rm * mul
    nout = next(n for n in b.g.nodes if n.name == last)
    o = b.g.constants[nout.inputs[1]].shape[0]
    b.precision.append({"layer": name, "op": "post_mul/post_add of " + last if b.fused else "ChannelAffine",
                        "source": "TensorNorm (scalar affine, eps %g)" % tn.eps, "weights": "fp32 scalar affine", "activations": "fp32 logits",
                        "tensornorm": {"weight": w, "bias": bb, "running_mean": rm, "running_var": rv, "positive_scale": w > 0}})
    if b.fused:
        _set_post(b.g, last, np.full(o, mul, np.float32), np.full(o, add, np.float32))
        return x
    g = b.g
    return g.add_node("ChannelAffine", [x, g.add_constant(f"{name}/scale", np.full(o, mul, np.float32)),
                                        g.add_constant(f"{name}/shift", np.full(o, add, np.float32))], name=name)


def _fc_layers(b: _Builder, x: str, mods, start: int, prefix: str, first_thr: str = "") -> str:
    """mods[start:] = [QuantLinear, BatchNorm1d, QuantIdentity(binary), (Dropout)]* + QuantLinear + TensorNorm."""
    g = b.g
    i = start
    first = True
    while i < len(mods):
        m = mods[i]
        if _cls(m) == "Dropout":
            i += 1
            continue
        name = f"{prefix}.{i}"
        _expect(_cls(m) == "QuantLinear", f"{name}: expected QuantLinear, got {_cls(m)}")
        _check_layer_quant_off(m, name)
        w = binary_weight(m, name)
        y = _bdense(g, x, w, name, first_thr if first else "")
        b.precision.append({"layer": name, "op": "BDense", "source": "QuantLinear (BinaryQuant, scale 1)", "weights": "binary +-1",
                            "activations": "binary (ge, sign(0)=+1)", "in_features": int(w.shape[1]), "out_features": int(w.shape[0])})
        first = False
        if i + 1 < len(mods) and _cls(mods[i + 1]) == "TensorNorm":
            y = _tensornorm(b, y, mods[i + 1], name, f"{prefix}.{i + 1}")
            _expect(i + 2 == len(mods), f"{prefix}: modules after TensorNorm")
            return y
        bn, act = mods[i + 1], mods[i + 2]
        _expect(_cls(bn) == "BatchNorm1d" and bn.num_features == w.shape[0], f"{prefix}.{i + 1}: expected BatchNorm1d({w.shape[0]})")
        _expect(_cls(act) == "QuantIdentity", f"{prefix}.{i + 2}: expected QuantIdentity")
        probe_binary_act(act, f"{prefix}.{i + 2}")

        def prod(mul, add, name=name):
            # BDense has no bit output (op contract): it emits s*(acc - T), an exact small
            # integer in f32 whose ge-sign the next BDense takes.
            _set_post(g, name, mul, add)
        x = b.bn_sign(y, bn, f"{prefix}.{i + 1}", f"{prefix}.{i + 2}", int(w.shape[1]), prod, parity=int(w.shape[1]) % 2)
        i += 3
    raise NotImplementedError(f"{prefix}: missing final QuantLinear + TensorNorm")


def _convert_fc(model, fused: bool, input_name: str):
    b = _Builder(fused)
    g = b.g
    feats = model.features
    k_in = int(feats[2].in_features) if len(feats) > 2 and _cls(feats[2]) == "QuantLinear" else 0
    _expect(k_in == 784, f"FC: expected 784 input features (MNIST 28x28x1), got {k_in}")
    x = g.add_input(input_name, [1, 28, 28, 1])
    x = g.add_node("Reshape", [x], {"shape": [1, k_in]}, name="flatten")
    _expect(_cls(feats[0]) == "QuantIdentity", "features.0: expected QuantIdentity (input binarizer)")
    probe_binary_act(feats[0], "features.0")
    b.precision.append({"layer": "features.0", "op": "BDense threshold 0.5" if fused else "ChannelAffine(2,-1)+Activation(sign)",
                        "source": "2x-1 then QuantIdentity (ClampedBinaryQuant)", "weights": "-", "activations": "binary input"})
    if fused:
        thr = g.add_constant("features.0/threshold", np.full(k_in, 0.5, np.float32))
        return _finish(b, _fc_layers(b, x, list(feats), 1, "features", thr), "FC")
    x = g.add_node("ChannelAffine", [x, g.add_constant("input/scale2", np.full(k_in, 2.0, np.float32)),
                                     g.add_constant("input/shift-1", np.full(k_in, -1.0, np.float32))], name="input_2x_minus_1")
    x = g.add_node("Activation", [x], {"type": "sign"}, name="features.0")
    return _finish(b, _fc_layers(b, x, list(feats), 1, "features"), "FC")


def _finish(b: _Builder, out: str, arch: str):
    b.g.set_outputs(out)
    ops = b.g.op_counts()
    info = {"arch": arch, "fused": b.fused, "layer_precision": b.precision, "threshold_margins": b.margins, "op_counts": ops,
            "n_binary_layers": ops.get("BConv2D", 0) + ops.get("BDense", 0),
            "float_compute_layers": [n.name for n in b.g.nodes if n.op in ("Conv2D", "Dense")]}
    return b.g, info


def _convert_cnv(model, fused: bool, input_name: str):
    b = _Builder(fused)
    g = b.g
    cf, lf = list(model.conv_features), list(model.linear_features)
    _expect(_cls(cf[0]) == "QuantIdentity", "conv_features.0: expected QuantIdentity (input quantizer)")
    q = probe_int8_input_quant(cf[0], "conv_features.0")
    first_conv = cf[1]
    _expect(_cls(first_conv) == "QuantConv2d", "conv_features.1: expected QuantConv2d")
    in_ch = int(first_conv.in_channels)
    hw = 32
    x = g.add_input(input_name, [1, hw, hw, in_ch])
    x = g.add_node("ChannelAffine", [x, g.add_constant("input/scale2", np.full(in_ch, 2.0, np.float32)),
                                     g.add_constant("input/shift-1", np.full(in_ch, -1.0, np.float32))], name="input_2x_minus_1")
    # Generic FakeQuant (docs/operators_bhvit.md): y = clamp(rint(x/scale + zp), qmin, qmax) * scale in f32 steps.
    # With zp = 0 this is Brevitas IntQuant's (clamp(round(x/scale + 0)) - 0) * scale bit for bit.
    x = g.add_node("FakeQuant", [x, g.add_constant("conv_features.0/scale", np.full(in_ch, q["scale"], np.float32)),
                                 g.add_constant("conv_features.0/zero_point", np.zeros(in_ch, np.float32))],
                   {"axis": -1, "pre_scale": 1.0, "qmin": q["qmin"], "qmax": q["qmax"]}, name="conv_features.0")
    b.precision.append({"layer": "conv_features.0", "op": "FakeQuant", "source": "QuantIdentity (RescalingIntQuant, 8-bit, POWER_OF_TWO)",
                        "weights": "-", "activations": f"int{q['bit_width']} Q1.7 (scale {q['scale']}, [{q['qmin']}, {q['qmax']}], round half-even)"})
    i, conv_idx = 1, 0
    while i < len(cf):
        m = cf[i]
        name = f"conv_features.{i}"
        _expect(_cls(m) == "QuantConv2d", f"{name}: expected QuantConv2d, got {_cls(m)}")
        _check_layer_quant_off(m, name)
        _expect(_pair(m.kernel_size) == (3, 3) and _pair(m.stride) == (1, 1) and _pair(m.padding) == (0, 0)
                and _pair(m.dilation) == (1, 1) and m.groups == 1 and m.padding_mode == "zeros",
                f"{name}: only 3x3 stride-1 VALID convs are supported")
        w = binary_weight(m, name)
        o, ci, kh, kw = w.shape
        k = ci * kh * kw
        bn, act = cf[i + 1], cf[i + 2]
        _expect(_cls(bn) == "BatchNorm2d" and bn.num_features == o, f"conv_features.{i + 1}: expected BatchNorm2d({o})")
        _expect(_cls(act) == "QuantIdentity", f"conv_features.{i + 2}: expected QuantIdentity")
        probe_binary_act(act, f"conv_features.{i + 2}")
        pool = i + 3 < len(cf) and _cls(cf[i + 3]) == "MaxPool2d"
        if pool:
            mp = cf[i + 3]
            _expect(_pair(mp.kernel_size) == (2, 2) and _pair(mp.stride) == (2, 2) and _pair(mp.padding) == (0, 0)
                    and _pair(mp.dilation) == (1, 1) and not mp.ceil_mode, f"conv_features.{i + 3}: only MaxPool2d(2) supported")
        last_conv = i + (4 if pool else 3) >= len(cf)
        if conv_idx == 0:
            # W1A8: float Conv2D with +-1 weights on Q1.7 inputs (exact: multiples of 1/128, |sum| <= 27).
            wname = f"{name}/kernel"
            y = g.add_node("Conv2D", [x, g.add_constant(wname, _to_ohwi(w).astype(np.float32), layout="OHWI"), ""],
                           {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0,
                            "activation": "none"}, name=name)
            b.precision.append({"layer": name, "op": "Conv2D (float, exact)", "source": "QuantConv2d (BinaryQuant, scale 1) on 8-bit input",
                                "weights": "binary +-1 (stored as f32)", "activations": "int8 Q1.7 input (W1A8)",
                                "note": "all products/partial sums are multiples of 1/128 below 2^7: exact in f32 and f64"})
            inv = 1.0 / q["scale"]
            _expect(inv == int(inv), "input scale must be 1/integer")

            def prod(mul, add, name=name, wname=wname, w=w, inv=int(inv)):
                node = next(n for n in g.nodes if n.name == name)
                s = mul.astype(np.float64)
                node.inputs[1] = g.add_constant(wname + "_signed", _to_ohwi(w * s[:, None, None, None].astype(np.int8)).astype(np.float32), layout="OHWI")
                node.inputs[2] = g.add_constant(name + "/bias", (add.astype(np.float64) / inv).astype(np.float32))
            x = b.bn_sign(y, bn, f"conv_features.{i + 1}", f"conv_features.{i + 2}", k * (-q["qmin"]), prod, acc_scale=int(inv))
        else:
            y = _bconv(g, x, w, name)
            b.precision.append({"layer": name, "op": "BConv2D", "source": "QuantConv2d (BinaryQuant, scale 1)", "weights": "binary +-1",
                                "activations": "binary (ge, sign(0)=+1)", "kernel": [kh, kw], "pads": [0, 0, 0, 0]})

            def prod(mul, add, name=name, bit=not pool and not last_conv):
                _set_post(g, name, mul, add)
                if bit:
                    next(n for n in g.nodes if n.name == name).attrs["output"] = "bit"
            x = b.bn_sign(y, bn, f"conv_features.{i + 1}", f"conv_features.{i + 2}", k, prod, parity=k % 2)
        if pool:
            # sign is monotone (ge), so MaxPool(sign(v)) == sign(MaxPool(v)): exact in the fused variant.
            x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name=f"conv_features.{i + 3}")
            i += 4
        else:
            i += 3
        conv_idx += 1
        hw = (hw - 2) // (2 if pool else 1)
    _expect(hw == 1, f"CNV: flatten expected at 1x1 spatial size, got {hw}x{hw}")
    c_last = int(cf[-3].out_channels) if _cls(cf[-3]) == "QuantConv2d" else 0
    _expect(c_last == int(lf[0].in_features), "CNV: flatten size mismatch")
    x = g.add_node("Reshape", [x], {"shape": [1, c_last]}, name="flatten")
    return _finish(b, _fc_layers(b, x, lf, 0, "linear_features"), "CNV")
