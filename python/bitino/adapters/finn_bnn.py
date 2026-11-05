"""Shared IR emitter for the FINN / BNN-PYNQ CNV and FC (LFC/TFC) topologies, whatever the serialisation.

The readers (``finn_onnx`` for FINN-ONNX MultiThreshold and QONNX BipolarQuant/Quant graphs, ``bnn_pynq_npz``
for the original Theano/Lasagne npz files) translate their source into a list of ``Layer`` specs with exact
±1 weights and float64 affine parameters; this module turns the specs into Bitino IR with the same rules
as ``brevitas_bnn`` (docs/operators_brevitas.md):

* Binary layers have ±1 weights, scale 1, VALID 3x3 convs (no padding) or dense layers, and integer
  accumulators ``acc``. The value fed to the binarizer is ``y = v * scale + shift`` with ``v = acc *
  acc_scale`` (``acc_scale`` = 128 for the W1A8 first conv, whose inputs are multiples of 1/128) and
  (``scale``, ``shift``) the float64 affine of BN (plus any conv/dense bias and scalar ``Mul``).
* fused: ``y >= 0`` becomes an integer threshold on ``acc`` (``brevitas_bnn.integer_threshold``: s = sign(scale),
  T = ceil/floor(tau)); the producer emits the exact integer ``s*(acc - T)`` (bits when its consumer is a
  BConv2D) whose ge-sign is the activation. The W1A8 first conv folds the threshold into a float Conv2D
  (weights s*w, bias -s*T/128; every partial sum is a multiple of 1/128 below 2^12: exact in f32).
* ``pool="before_bn"`` (Theano/Lasagne CNV: conv -> MaxPool -> BN -> sign): MaxPool runs on the raw integer
  accumulator, then the NEXT layer binarizes with a per-input-channel threshold (``ge``): channels with
  scale > 0 use T = ceil(tau) (act = +1 <=> n >= T); channels with scale < 0 have act = +1 <=> n <= floor(tau)
  = -(ge(n, floor(tau)+1)), so the next layer's weights of that input channel are negated. That rewrite is
  exact only without spatial padding, which holds for every CNV conv (asserted).
* ``pool="after_sign"`` (Brevitas/FINN CNV): MaxPool on the ±1 activation commutes with the monotone ge-sign,
  so it runs on the integer carrier.
* The last layer gets ``post_mul = scale``, ``post_add = shift`` (float32 rounding of the float64 affine).

unfused: literal mirror — BConv2D/BDense/Conv2D accumulators, [MaxPool], ChannelAffine(scale, shift), Activation(sign).

Layouts: weights come in OIHW / [O, I]; the IR uses OHWI / [O, I] and NHWC activations. Flattening happens
at 1x1 spatial size (asserted) or on a C = 1 input, where NCHW and NHWC orders coincide.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..ir import Graph
from .brevitas_bnn import integer_threshold


@dataclasses.dataclass
class InputSpec:
    kind: str                      # "int8" (ChannelAffine + FakeQuant) | "binary_threshold" | "pm1"
    shape: list                    # NHWC [1, H, W, C]
    affine: tuple | None = None    # (a, b): x*a + b before the int8 quantizer
    q: dict | None = None          # FakeQuant params {scale, qmin, qmax}
    threshold: float | None = None  # binary_threshold: ge threshold on the (flattened) input
    source: str = ""


@dataclasses.dataclass
class Layer:
    name: str
    kind: str                      # "conv" | "dense"
    w: np.ndarray                  # exact ±1, OIHW (conv) or [O, I] (dense)
    scale: np.ndarray              # float64 [O]: binarizer input y = (acc*acc_scale)*scale + shift
    shift: np.ndarray              # float64 [O]
    pool: str | None = None        # None | "before_bn" | "after_sign"
    final: bool = False
    w1a8: bool = False             # first conv on an int8 (multiple of 1/128) input
    source: str = ""
    extra: dict = dataclasses.field(default_factory=dict)


def _ohwi(w: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1)))


def _bnp(scale: np.ndarray, shift: np.ndarray) -> dict:
    scale = np.asarray(scale, np.float64)
    shift = np.asarray(shift, np.float64)
    if np.any(scale == 0):
        raise NotImplementedError("binarizer affine with zero scale (constant channel) not supported")
    return {"scale": scale, "shift": shift, "tau": -shift / scale, "sign": np.sign(scale)}


def emit(inp: InputSpec, layers: list[Layer], fused: bool = True, input_name: str = "input"):
    """Returns (graph, info). info["binarization_points"]: per source binarizer, the IR tensor + per-channel
    threshold + polarity whose ge-decision equals that activation (for first-flip analysis)."""
    for l in layers:
        if not np.all((l.w == 1) | (l.w == -1)):
            raise NotImplementedError(f"{l.name}: weights are not exactly ±1")
    if not layers[-1].final or any(l.final for l in layers[:-1]):
        raise NotImplementedError("exactly the last layer must be final")
    g = Graph()
    x = g.add_input(input_name, inp.shape)
    prec, margins, points = [], [], []
    shape = list(inp.shape)            # current NHWC (or [1, K]) shape
    pending_thr = None                 # (threshold vector, flip vector) for the next layer's input
    first_thr_const = ""
    if inp.kind == "int8":
        c = inp.shape[-1]
        a, b = inp.affine
        x = g.add_node("ChannelAffine", [x, g.add_constant("input/scale", np.full(c, a, np.float32)),
                                         g.add_constant("input/shift", np.full(c, b, np.float32))], name="input_affine")
        q = inp.q
        x = g.add_node("FakeQuant", [x, g.add_constant("input_quant/scale", np.full(c, q["scale"], np.float32)),
                                     g.add_constant("input_quant/zero_point", np.zeros(c, np.float32))],
                       {"axis": -1, "pre_scale": 1.0, "qmin": q["qmin"], "qmax": q["qmax"]}, name="input_quant")
        prec.append({"layer": "input", "op": "ChannelAffine + FakeQuant", "source": inp.source,
                     "activations": f"int8 (scale {q['scale']}, [{q['qmin']}, {q['qmax']}], round half-even)"})
    elif inp.kind in ("binary_threshold", "pm1"):
        k = int(np.prod(inp.shape[1:]))
        if layers[0].kind == "dense":
            x = g.add_node("Reshape", [x], {"shape": [1, k]}, name="flatten_input")
            shape = [1, k]
        if inp.kind == "binary_threshold":
            first_thr_const = g.add_constant("input/threshold", np.full(shape[-1], inp.threshold, np.float32))
        prec.append({"layer": "input", "op": ("ge threshold %g in the first binary layer" % inp.threshold) if inp.kind == "binary_threshold"
                     else "±1 input, ge (exact)", "source": inp.source, "activations": "binary"})
    else:
        raise NotImplementedError(inp.kind)

    for li, l in enumerate(layers):
        nxt = layers[li + 1] if li + 1 < len(layers) else None
        w = np.asarray(l.w, np.int8).copy()
        if l.kind == "dense" and len(shape) == 4:
            if not (shape[1] == 1 and shape[2] == 1):
                raise NotImplementedError(f"{l.name}: flatten at {shape[1]}x{shape[2]} (NCHW vs NHWC order) not supported")
            x = g.add_node("Reshape", [x], {"shape": [1, shape[3]]}, name=f"{l.name}/flatten")
            shape = [1, shape[3]]
        thr_name = ""
        n_flipped = 0
        if pending_thr is not None:
            thr, flip = pending_thr
            w = (w * flip.reshape((1, -1) + (1,) * (w.ndim - 2))).astype(np.int8)   # input-channel polarity
            n_flipped = int(np.sum(flip < 0))
            thr_name = g.add_constant(f"{l.name}/threshold", thr.astype(np.float32))
            pending_thr = None
        elif li == 0 and first_thr_const:
            thr_name = first_thr_const
        bnp = _bnp(l.scale, l.shift)
        o = w.shape[0]
        if l.kind == "conv":
            if w.shape[2:] != (3, 3):
                raise NotImplementedError(f"{l.name}: only 3x3 convs")
            k = int(np.prod(w.shape[1:]))
            if l.w1a8:
                if inp.kind != "int8" or li != 0 or thr_name:
                    raise NotImplementedError(f"{l.name}: W1A8 conv must be the first layer on an int8 input")
                wname = f"{l.name}/kernel"
                y = g.add_node("Conv2D", [x, g.add_constant(wname, _ohwi(w).astype(np.float32), layout="OHWI"), ""],
                               {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0,
                                "activation": "none"}, name=l.name)
                inv = int(round(1.0 / inp.q["scale"]))
                if inv * inp.q["scale"] != 1.0:
                    raise NotImplementedError("int8 input scale must be 1/integer")
                kmax = k * max(-inp.q["qmin"], inp.q["qmax"])
                op = "Conv2D (float, exact: multiples of 1/%d)" % inv
            else:
                y = g.add_node("BConv2D", [x, g.add_constant(f"{l.name}/kernel_bits", _ohwi(w), dtype="bit", layout="OHWI_bit"),
                                           "", "", thr_name],
                               {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0,
                                "input_binarize": "ge", "activation": "none", "output": "f32"}, name=l.name)
                inv, kmax = 1, k
                op = "BConv2D"
            shape = [1, shape[1] - 2, shape[2] - 2, o]
        else:
            k = int(w.shape[1])
            y = g.add_node("BDense", [x, g.add_constant(f"{l.name}/kernel_bits", w, dtype="bit", layout="OI_bit"), "", "", thr_name],
                           {"input_binarize": "ge", "activation": "none"}, name=l.name)
            inv, kmax, op = 1, k, "BDense"
            shape = [1, o]
        prec.append({"layer": l.name, "op": op, "source": l.source, "weights": "binary ±1 (scale 1)",
                     "activations": "int8 input (W1A8)" if l.w1a8 else "binary (ge)", "in": k, "out": o,
                     "input_threshold": bool(thr_name), "input_channels_negated": n_flipped})
        node = g.nodes[-1]
        parity = None if l.w1a8 else k % 2

        if l.final:
            if fused:
                node.inputs[2] = g.add_constant(f"{l.name}/post_mul", bnp["scale"].astype(np.float32))
                node.inputs[3] = g.add_constant(f"{l.name}/post_add", bnp["shift"].astype(np.float32))
                x = y
            else:
                x = g.add_node("ChannelAffine", [y, g.add_constant(f"{l.name}/affine_scale", bnp["scale"].astype(np.float32)),
                                                 g.add_constant(f"{l.name}/affine_shift", bnp["shift"].astype(np.float32))], name=f"{l.name}/affine")
            continue

        mul, add, minfo = integer_threshold(bnp, kmax, inv, parity)
        margins.append({"layer": l.name, "k_max": kmax, **minfo})
        if not fused:
            t = y
            if l.pool == "before_bn":
                t = g.add_node("MaxPool2D", [t], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name=f"{l.name}/pool")
            t = g.add_node("ChannelAffine", [t, g.add_constant(f"{l.name}/bn_scale", bnp["scale"].astype(np.float32)),
                                             g.add_constant(f"{l.name}/bn_shift", bnp["shift"].astype(np.float32))], name=f"{l.name}/bn")
            points.append({"point": l.name, "tensor": t, "threshold": None, "flip": None})
            x = g.add_node("Activation", [t], {"type": "sign"}, name=f"{l.name}/sign")
            if l.pool == "after_sign":
                x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name=f"{l.name}/pool")
        elif l.pool == "before_bn":
            if l.w1a8:
                raise NotImplementedError(f"{l.name}: pool before BN on the W1A8 conv not supported")
            if nxt is None or nxt.kind != "conv":
                raise NotImplementedError(f"{l.name}: pool-before-BN must feed a VALID conv (polarity folding)")
            x = g.add_node("MaxPool2D", [y], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name=f"{l.name}/pool")
            s = mul.astype(np.float64)
            t_int = -add.astype(np.float64) * s
            thr = np.where(s > 0, t_int, t_int + 1.0)
            flip = np.where(s > 0, 1, -1).astype(np.int8)
            pending_thr = (thr, flip)
            points.append({"point": l.name, "tensor": x, "threshold": thr.astype(np.float32).tolist(), "flip": flip.tolist()})
        else:
            if l.w1a8:
                s = mul.astype(np.float64)
                node.inputs[1] = g.add_constant(f"{l.name}/kernel_signed", _ohwi(w * s[:, None, None, None].astype(np.int8)).astype(np.float32), layout="OHWI")
                node.inputs[2] = g.add_constant(f"{l.name}/bias", (add.astype(np.float64) / inv).astype(np.float32))
            else:
                node.inputs[2] = g.add_constant(f"{l.name}/post_mul", mul)
                node.inputs[3] = g.add_constant(f"{l.name}/post_add", add)
                if l.kind == "conv" and l.pool is None and nxt is not None and nxt.kind == "conv":
                    node.attrs["output"] = "bit"
            points.append({"point": l.name, "tensor": y, "threshold": None, "flip": None})
            x = y
            if l.pool == "after_sign":
                x = g.add_node("MaxPool2D", [y], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name=f"{l.name}/pool")
        if l.pool:
            if l.kind != "conv":
                raise NotImplementedError(f"{l.name}: pooling after a dense layer")
            shape = [1, shape[1] // 2, shape[2] // 2, shape[3]]
    g.set_outputs(x)
    ops = g.op_counts()
    info = {"fused": fused, "layer_precision": prec, "threshold_margins": margins, "op_counts": ops,
            "n_binary_layers": ops.get("BConv2D", 0) + ops.get("BDense", 0),
            "float_compute_layers": [n.name for n in g.nodes if n.op in ("Conv2D", "Dense")],
            "bit_output_layers": [n.name for n in g.nodes if n.attrs.get("output") == "bit"],
            "binarization_points": points}
    return g, info


def ir_activation(env: dict, point: dict) -> np.ndarray:
    """+1/-1 activation (NCHW-flattened per sample) of a binarization point from an executor env."""
    t = env[point["tensor"]]
    thr = 0.0 if point["threshold"] is None else np.asarray(point["threshold"], t.dtype)
    a = np.where(t >= thr, 1, -1)
    if point["flip"] is not None:
        a = a * np.asarray(point["flip"])
    if a.ndim == 4:
        a = a.transpose(0, 3, 1, 2)
    return a.reshape(a.shape[0], -1)


def ir_margin(env: dict, point: dict) -> np.ndarray:
    t = env[point["tensor"]]
    thr = 0.0 if point["threshold"] is None else np.asarray(point["threshold"], np.float64)
    m = np.abs(np.asarray(t, np.float64) - thr)
    if m.ndim == 4:
        m = m.transpose(0, 3, 1, 2)
    return m.reshape(m.shape[0], -1)
