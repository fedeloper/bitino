"""Adapter: live tf.keras / Larq functional model -> Bitino IR.

Run inside the Larq reference environment (research/reference_envs). The
adapter reads the *instantiated* Keras model (architecture from the official
larq_zoo code, weights from the verified .h5), so no pickled objects are
loaded. Every layer type must be mapped explicitly; unknown layers raise.

Semantics captured (see docs/numerical-correctness.md):
  * larq sign: sign(x) = tf.sign(tf.sign(x) + 0.1)  ->  0 maps to +1 ("ge").
  * QuantConv2D(pad_values=v): padded inputs take value v *after* the input
    quantizer, i.e. pad_value in the binary domain.
  * TF "same" padding is asymmetric (extra pad at bottom/right).
  * MagnitudeAwareSign: sign(W) * mean(|W|) per output channel -> post_mul.
  * Activation on QuantConv2D is applied before any following BatchNorm.
  * xnor_weight_scale (XNOR-Net): sign(clip(W, -1, 1)) * mean|clip(W, -1, 1)| per
    output channel -> post_mul.
  * DoReFa activation quantizer (k bits): round_half_even(clip(x, 0, 1) * n) / n,
    n = 2^k - 1, with magnitude_aware_sign_unclipped weights (sign(W) * mean|W| over
    the whole layer) -> QBConv2D / QBDense with post_mul = mean|W| / n (integer
    levels x binary weights, see docs/operators_larq.md). Not a strict BNN.
  * Quantized conv with float input and sign weights (BinaryAlexNet stem) -> Conv2D
    with +-1 float weights (W1, A fp32).
  * Dense(activation=sigmoid) -> Reshape -> Multiply (Real-to-Binary Net data-dependent
    scaling) -> ChannelGate(gate=sigmoid) on the pre-sigmoid Dense output. The same with
    tf.reshape(gate, [-1, 1, 1, C]) and `x *= gate` TFOpLambdas (legacy QuickNet-XL
    squeeze-and-excite, larq/zoo @40f4cdc quicknet.py:35-58, 97-121).
  * larq_zoo TFOpLayer Lambdas (MeliusNet): tf.split + per-split Conv2D + tf.concat ->
    one grouped Conv2D; tf.concat -> Concat; the improvement-block merge
    x + pad(w, channels before) -> Pad + Add; identity Lambdas -> no node.
    Every Lambda interpretation is checked numerically against the Lambda itself.
  * Every quantizer interpretation is checked numerically against the live quantizer
    (kernel quantizers on the layer kernel, input quantizers on probe values).
  * DoReFa(k_bit=1, mode="weights") used as an *input* quantizer (BinaryFaceNet,
    insightface_larq_keras): q(x) = 2*round(tanh(x) / (2*max|tanh(x)|) + 0.5) - 1 in {-1, +1}; in float32
    this is exactly +1 iff x > 2^-24 * max|tanh(x)| (per tensor, i.e. per sample at batch 1): tiny positive inputs
    map to -1 (whole near-dead channels in BinaryFaceNet). Mapped to BConv2D input_binarize "gt" on the operand
    x - 2^-24 * tanh(max|x|) (MaxPool2D / Concat / Reshape / Activation tanh / ChannelAffine / AddBroadcast, see
    _dorefa_w1_operand); checked on the float32 neighbours of the threshold for m = 1 and m < 1. The kernel
    quantizer (same formula on the kernel) is evaluated live and must give exactly +-1 (no scale). Grouped
    convs keep `groups`; QuantDepthwiseConv2D -> BConv2D(groups=C); a 1x1 QuantSeparableConv2D (depthwise
    then pointwise on the same +-1 input, no requantization in between) -> one 1x1 BConv2D with weight
    bits pointwise[o, c] * depthwise[c] (exact: products of +-1). Each such layer is also checked end to end
    against the live layer on random inputs.
  * Conv2D(activation=sigmoid) feeding Multiply (BinaryFaceNet attention) -> Conv2D(activation=none) +
    ChannelGate(gate=sigmoid), like the Dense(sigmoid) gate.
  * GlobalMaxPooling2D at a fixed input size -> MaxPool2D over the full H x W window (+ Reshape to [N, C]).
"""
from __future__ import annotations

import inspect

import numpy as np

from ..ir import Graph

SIGN_QUANTIZERS = {"SteSign", "ApproxSign", "SwishSign", "ste_sign", "approx_sign", "swish_sign"}
SCALED_SIGN_QUANTIZERS = {"MagnitudeAwareSign", "magnitude_aware_sign"}
XNOR_QUANTIZERS = {"xnor_weight_scale"}                      # larq_zoo.literature.xnornet
LAYER_SCALED_SIGN_QUANTIZERS = {"magnitude_aware_sign_unclipped"}  # larq_zoo.literature.dorefanet
KBIT_ACT_QUANTIZERS = {"DoReFa", "DoReFaQuantizer"}

# Probe values for input-quantizer checks: exact zeros, tiny magnitudes and DoReFa ties.
_PROBE = np.array([-2.0, -1.0, -0.5, -1e-30, -0.0, 0.0, 1e-30, 1e-8, 1 / 6, 0.25, 0.5, 2 / 3, 0.75, 5 / 6, 1.0, 1.5],
                  np.float32)


def _qname(q) -> str | None:
    if q is None:
        return None
    if isinstance(q, str):
        return q
    if hasattr(q, "__name__"):
        return q.__name__
    return type(q).__name__


def _same_pads(in_size: int, k: int, s: int, d: int) -> tuple[int, int]:
    eff = d * (k - 1) + 1
    out = -(-in_size // s)
    total = max((out - 1) * s + eff - in_size, 0)
    return total // 2, total - total // 2


def _pads(padding: str, h: int, w: int, kh: int, kw: int, sh: int, sw: int, dh: int = 1, dw: int = 1) -> list[int]:
    if padding == "valid":
        return [0, 0, 0, 0]
    if padding == "same":
        pt, pb = _same_pads(h, kh, sh, dh)
        pl, pr = _same_pads(w, kw, sw, dw)
        return [pt, pb, pl, pr]
    raise ValueError(f"unsupported padding {padding}")


def _activation_name(layer) -> str:
    act = getattr(layer, "activation", None)
    name = getattr(act, "__name__", None) if act is not None else None
    if name in (None, "linear"):
        return "none"
    if name == "relu":
        return "relu"
    raise NotImplementedError(f"{layer.name}: fused activation {name} not supported")


def _call_quantizer(q, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    return np.asarray(q(tf.constant(x)).numpy())


def _check_input_quantizer(lname: str, q, kind: str, k_bits: int = 0) -> None:
    """Asserts that the live input quantizer computes what the IR op assumes."""
    got = _call_quantizer(q, _PROBE)
    if kind == "sign":
        want = np.where(_PROBE >= 0, 1.0, -1.0)
    else:  # dorefa activations
        n = np.float32(2 ** k_bits - 1)
        want = np.round(np.clip(_PROBE, 0, 1) * n) / n
    if not np.array_equal(got.astype(np.float64), want.astype(np.float32).astype(np.float64)):
        raise NotImplementedError(f"{lname}: input quantizer {_qname(q)} does not match '{kind}' semantics: {got} vs {want}")


def _check_kernel_quantizer(lname: str, q, kernel: np.ndarray, w_pm1: np.ndarray, scale) -> None:
    """Asserts kernel_quantizer(kernel) == w_pm1 * scale (scale broadcast over the last axis)."""
    got = _call_quantizer(q, kernel).astype(np.float64)
    want = w_pm1.astype(np.float64) * (1.0 if scale is None else np.asarray(scale, np.float64))
    if not np.array_equal(np.sign(got), np.sign(want)):
        raise NotImplementedError(f"{lname}: kernel quantizer {_qname(q)}: sign pattern differs from the IR interpretation")
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    # Scales are float32 reductions on both sides (TF vs numpy summation order): allow a
    # relative float32-reduction tolerance; the strict check is the float64 semantic check.
    if err > 2e-5 * max(1e-3, float(np.max(np.abs(want)))):
        raise NotImplementedError(f"{lname}: kernel quantizer {_qname(q)}: magnitude differs from the IR interpretation ({err})")


def _is_dorefa_w1(q) -> bool:
    """DoReFa(k_bit=1, mode='weights'): output in {-1, +1} (BinaryFaceNet input and kernel quantizer)."""
    return (q is not None and _qname(q) in KBIT_ACT_QUANTIZERS and int(getattr(q, "precision", 0)) == 1
            and getattr(q, "mode", "") == "weights")


# DoReFa(k=1, 'weights') on float32 activations: q = 2 * round(fl(t / (2m)) + 0.5) - 1 with t = tanh(x),
# m = max|tanh| over the tensor. fl(t/(2m)) + 0.5 rounds above 0.5 iff t/(2m) > 2^-25 + 2^-49 (real), i.e.
# t > m * 2^-24 * (1 + 2^-24); for float32 t this equals t > 2^-24 * m (no float32 lies in between, see
# the research notes (not published)), and TF returns tanh(x) == x at these magnitudes. So
#   q(x) = +1  iff  x > T,  T = 2^-24 * m  (per sample / per tensor; x == T -> -1)
# i.e. tiny positive inputs (e.g. channels whose folded BN scale is ~0) binarize to -1 in the float32 network the
# authors trained and evaluated; the threshold shrinks to ~2^-53 * m in float64, which is NOT the trained function.
DOREFA_W1_TSCALE = 2.0 ** -24


def _check_dorefa_w1_input(lname: str, q) -> None:
    """Checks q(x) == (x > 2^-24 * m) on tensors with m = 1 and m < 1, probing the float32 neighbours of T."""
    import tensorflow as tf
    for big in (20.0, 2.0, 0.5):
        m = float(np.max(np.abs(tf.math.tanh(tf.constant([big], tf.float32)).numpy())))
        t = np.float32(DOREFA_W1_TSCALE * m)
        near = [t]
        lo = hi = t
        for _ in range(4):
            lo, hi = np.nextafter(lo, np.float32(0)), np.nextafter(hi, np.float32(1))
            near += [lo, hi]
        v = np.array([big, -big / 2] + near + [np.float32(t * 2), np.float32(t / 2), 0.0, -0.0, 1e-9, -1e-9, 1e-3, -1e-3, -t],
                     np.float32)
        got = _call_quantizer(q, v).astype(np.float64)
        want = np.where(v > t, 1.0, -1.0)
        if not np.array_equal(got, want):
            raise NotImplementedError(f"{lname}: DoReFa(k=1, weights) input quantizer is not 'x > 2^-24 * max|tanh x|' "
                                      f"(m={m}): {got} vs {want} on {v}")


def _dorefa_w1_operand(g: Graph, x: str, shape: list, lname: str) -> str:
    """x - 2^-24 * tanh(max|x|) (per sample), so that BConv2D 'gt' reproduces the float32 DoReFa(k=1) quantizer.
    max|x| = max(MaxPool_HxW(x), MaxPool_HxW(-x)) over channels (Concat -> Reshape -> MaxPool2D); tanh is the
    existing Activation (double libm, rounded once); x - T via AddBroadcast (the sign of a float32 difference is
    exact)."""
    _, h, w, c = shape
    full = {"kernel": [h, w], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    mx = g.add_node("MaxPool2D", [x], full, name=f"{lname}/dorefa_max")
    neg = g.add_node("ChannelAffine", [x, g.add_constant(f"{lname}/dorefa_neg1", -np.ones(c, np.float32)), ""],
                     name=f"{lname}/dorefa_neg")
    mn = g.add_node("MaxPool2D", [neg], full, name=f"{lname}/dorefa_negmax")
    cat = g.add_node("Concat", [mx, mn], {"axis": -1}, name=f"{lname}/dorefa_cat")
    r = g.add_node("Reshape", [cat], {"shape": [1, 1, 2 * c, 1]}, name=f"{lname}/dorefa_flat")
    amax = g.add_node("MaxPool2D", [r], {"kernel": [1, 2 * c], "strides": [1, 1], "pads": [0, 0, 0, 0]},
                      name=f"{lname}/dorefa_absmax")
    m = g.add_node("Activation", [amax], {"type": "tanh"}, name=f"{lname}/dorefa_m")
    nt = g.add_node("ChannelAffine", [m, g.add_constant(f"{lname}/dorefa_tscale", np.array([-DOREFA_W1_TSCALE], np.float32)), ""],
                    name=f"{lname}/dorefa_negT")
    return g.add_node("AddBroadcast", [x, nt], name=f"{lname}/dorefa_operand")


def _dorefa_w1_kernel_bits(lname: str, q, kernel: np.ndarray) -> np.ndarray:
    """Live DoReFa(k=1, 'weights') kernel quantizer -> int8 +-1 (must be exactly +-1)."""
    got = _call_quantizer(q, kernel)
    if not np.all(np.abs(got) == 1.0):
        raise NotImplementedError(f"{lname}: DoReFa(k=1) kernel quantizer output is not +-1")
    return np.where(got > 0, 1, -1).astype(np.int8)


def _verify_binary_layer(layer, ins, op: str, consts: dict, attrs: dict, n: int = 2) -> None:
    """Runs the live (quantized) layer on random inputs and compares with one IR node evaluated by the numpy
    executor in float64 (exact: integer accumulations, bias added once). Inputs are drawn away from the DoReFa
    rounding band (|x| >= 1e-3)."""
    import tensorflow as tf
    from .. import executor
    rng = np.random.default_rng(1)
    shp = [n] + [int(d) for d in ins[0].shape[1:]]
    x = rng.standard_normal(shp).astype(np.float32)
    x = np.where(np.abs(x) < 1e-3, np.float32(0.5), x)
    want = np.asarray(layer(tf.constant(x)).numpy(), np.float64)
    g = Graph()
    g.add_input("x", [1] + shp[1:])
    names = ["x"]
    for k, (v, kw) in consts.items():
        names.append(g.add_constant(k, v, **kw) if v is not None else "")
    out = g.add_node(op, names, attrs, name="node")
    g.set_outputs(out)
    for i in range(n):
        got = executor.run(g, {"x": x[i:i + 1]}, float_dtype=np.float64)[out]
        err = float(np.max(np.abs(got.reshape(-1) - want[i].reshape(-1))))
        if err > 1e-5 * max(1.0, float(np.max(np.abs(want[i])))):
            raise NotImplementedError(f"{layer.name}: live layer differs from its {op} interpretation ({err})")


def _verify_lambda(layer, inputs, np_fn) -> None:
    """Runs the Lambda on random inputs of the layer's input shapes and compares with np_fn."""
    import tensorflow as tf
    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal([1] + [int(d) for d in t.shape[1:]]).astype(np.float32) for t in inputs]
    arg = [tf.constant(a) for a in arrs] if len(arrs) > 1 else tf.constant(arrs[0])
    got = layer.function(arg)
    got = [np.asarray(t.numpy()) for t in got] if isinstance(got, (list, tuple)) else [np.asarray(got.numpy())]
    want = np_fn(*arrs)
    want = list(want) if isinstance(want, (list, tuple)) else [want]
    if len(got) != len(want) or not all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(got, want)):
        raise NotImplementedError(f"{layer.name}: Lambda does not match its IR interpretation")


def _grouped_conv_from_splits(g: Graph, convs: list[dict], lname: str, precision: list) -> str:
    """tf.split(x, G) -> G x Conv2D -> tf.concat  ==>  Conv2D(groups=G)."""
    groups = convs[0]["piece"]["groups"]
    src = convs[0]["piece"]["src"]
    ok = (len(convs) == groups and [c["piece"]["index"] for c in convs] == list(range(groups))
          and all(c["piece"]["src"] == src and c["attrs"] == convs[0]["attrs"] and c["w"].shape == convs[0]["w"].shape
                  and (c["bias"] is None) == (convs[0]["bias"] is None) for c in convs))
    if not ok:
        raise NotImplementedError(f"{lname}: concat of split convolutions is not a plain group convolution")
    w = np.concatenate([c["w"] for c in convs], axis=0)
    bias = np.concatenate([c["bias"] for c in convs]) if convs[0]["bias"] is not None else None
    attrs = dict(convs[0]["attrs"], groups=groups)
    wname = g.add_constant(f"{lname}/kernel", w, layout="OHWI")
    bname = g.add_constant(f"{lname}/bias", bias) if bias is not None else ""
    precision.append({"layer": lname, "op": "Conv2D (grouped)", "groups": groups, "weights": "fp32", "activations": "fp32",
                      "source_layers": [c["layer"] for c in convs]})
    return g.add_node("Conv2D", [src, wname, bname], attrs, name=lname)


def convert(model, model_id: str = "", input_name: str = "input") -> tuple[Graph, dict]:
    """Returns (graph, info). info records per-layer precision for the registry."""
    import tensorflow as tf  # noqa: F401  (only needed in the reference env)

    g = Graph()
    names: dict[int, str] = {}   # id(KerasTensor) -> IR tensor name
    precision: list[dict] = []
    # Pending structures that only become IR nodes at their consumer:
    split_pieces: dict[int, dict] = {}   # id(KerasTensor) -> output i of a tf.split Lambda
    split_convs: dict[int, dict] = {}    # id(KerasTensor) -> Conv2D applied to one split piece
    lazy_sigmoid: dict[int, str] = {}    # id(KerasTensor) -> IR name of the pre-sigmoid tensor

    def tname(t) -> str:
        if id(t) not in names:
            kind = ("tf.split piece" if id(t) in split_pieces else "per-split conv output" if id(t) in split_convs
                    else "sigmoid gate" if id(t) in lazy_sigmoid else "unknown tensor")
            raise NotImplementedError(f"{t.name}: {kind} consumed by an unsupported pattern")
        return names[id(t)]

    def inputs_of(layer):
        node = layer._inbound_nodes[0]
        ins = node.keras_inputs
        return list(ins) if isinstance(ins, (list, tuple)) else [ins]

    def bind(layer, name: str):
        out = layer.output
        names[id(out)] = name

    for layer in model.layers:
        cls = type(layer).__name__
        if cls == "InputLayer":
            shape = [1] + [int(s) for s in layer.output.shape[1:]]
            g.add_input(input_name, shape)
            bind(layer, input_name)
            continue

        ins = inputs_of(layer)
        x_shape = [int(s) if s is not None else 1 for s in ins[0].shape]
        lname = layer.name

        if cls in ("QuantConv2D", "Conv2D"):
            kernel = layer.kernel.numpy()            # HWIO
            kh, kw, kic, oc = kernel.shape
            sh, sw = layer.strides
            dh, dw = layer.dilation_rate
            pads = _pads(layer.padding, x_shape[1], x_shape[2], kh, kw, sh, sw, dh, dw)
            groups = getattr(layer, "groups", 1)
            iq_obj = getattr(layer, "input_quantizer", None)
            kq_obj = getattr(layer, "kernel_quantizer", None)
            iq = _qname(iq_obj)
            kq = _qname(kq_obj)
            if (cls == "Conv2D" and getattr(layer.activation, "__name__", None) == "sigmoid"
                    and id(ins[0]) not in split_pieces):
                # Sigmoid gate (BinaryFaceNet attention: 1x1 conv on a 1x1 map -> Multiply): the sigmoid is
                # evaluated inside ChannelGate; any other consumer of the gate raises (tname).
                bias = layer.bias.numpy() if layer.use_bias else None
                attrs = {"strides": list(layer.strides), "dilations": list(layer.dilation_rate),
                         "pads": _pads(layer.padding, x_shape[1], x_shape[2], kh, kw, sh, sw, dh, dw),
                         "groups": groups, "activation": "none"}
                wname = g.add_constant(f"{lname}/kernel", np.transpose(kernel, (3, 0, 1, 2)), layout="OHWI")
                bname = g.add_constant(f"{lname}/bias", bias) if bias is not None else ""
                pre = g.add_node("Conv2D", [tname(ins[0]), wname, bname], attrs, name=f"{lname}/pre_sigmoid")
                precision.append({"layer": lname, "op": "Conv2D (pre-sigmoid gate)", "weights": "fp32", "activations": "fp32"})
                lazy_sigmoid[id(layer.output)] = pre
                continue
            act = _activation_name(layer)
            bias = layer.bias.numpy() if getattr(layer, "use_bias", False) and layer.bias is not None else None
            attrs = {"strides": [sh, sw], "dilations": [dh, dw], "pads": pads, "groups": groups, "activation": act}
            w_ohwi = np.transpose(kernel, (3, 0, 1, 2))
            if id(ins[0]) in split_pieces:
                # One branch of a split -> conv -> concat group convolution (MeliusNet stem);
                # emitted as a single grouped Conv2D at the concat.
                if iq is not None or kq is not None or groups != 1:
                    raise NotImplementedError(f"{lname}: quantized/grouped conv on a tf.split piece")
                split_convs[id(layer.output)] = {"piece": split_pieces[id(ins[0])], "w": w_ohwi, "bias": bias,
                                                 "attrs": attrs, "layer": lname}
                continue
            if iq is None and kq is None:
                wname = g.add_constant(f"{lname}/kernel", w_ohwi, layout="OHWI")
                bname = g.add_constant(f"{lname}/bias", bias) if bias is not None else ""
                out = g.add_node("Conv2D", [tname(ins[0]), wname, bname], attrs, name=lname)
                precision.append({"layer": lname, "op": "Conv2D", "weights": "fp32", "activations": "fp32"})
            elif iq is None and kq in SIGN_QUANTIZERS:
                # Binary (+-1) weights on a real-valued input (BinaryAlexNet first layer): an
                # ordinary float convolution with +-1 weights.
                w_pm1 = np.where(w_ohwi >= 0, 1.0, -1.0).astype(np.float32)
                _check_kernel_quantizer(lname, kq_obj, kernel, np.where(kernel >= 0, 1.0, -1.0), None)
                pad_value = float(getattr(layer, "pad_values", 0.0))
                if pad_value != 0.0:
                    attrs["pad_value"] = pad_value
                wname = g.add_constant(f"{lname}/kernel_pm1", w_pm1, layout="OHWI")
                bname = g.add_constant(f"{lname}/bias", bias) if bias is not None else ""
                out = g.add_node("Conv2D", [tname(ins[0]), wname, bname], attrs, name=lname)
                precision.append({"layer": lname, "op": "Conv2D", "weights": "binary (+-1, stored as fp32)",
                                  "activations": "fp32", "kernel_quantizer": kq,
                                  "exact_zero_latent_weights": int(np.sum(kernel == 0))})
            elif iq in KBIT_ACT_QUANTIZERS and kq in LAYER_SCALED_SIGN_QUANTIZERS:
                # DoReFa: k-bit activations x binary weights with one per-layer scale.
                assert groups == 1, "grouped k-bit conv not supported"
                k_bits = int(iq_obj.precision)
                if getattr(iq_obj, "mode", "activations") != "activations":
                    raise NotImplementedError(f"{lname}: DoReFa input quantizer in mode {iq_obj.mode}")
                _check_input_quantizer(lname, iq_obj, "dorefa", k_bits)
                pad_value = float(getattr(layer, "pad_values", 0.0))
                if pad_value != 0.0:
                    raise NotImplementedError(f"{lname}: QBConv2D supports only zero padding, got pad_values={pad_value}")
                scale = float(np.mean(np.abs(kernel.astype(np.float64))))
                _check_kernel_quantizer(lname, kq_obj, kernel, np.where(kernel >= 0, 1.0, -1.0), scale)
                n_lv = 2 ** k_bits - 1
                mul = np.full(oc, scale / n_lv, np.float64).astype(np.float32)
                attrs.update({"act_bits": k_bits, "act_quant": "dorefa"})
                wname = g.add_constant(f"{lname}/kernel_bits", np.where(w_ohwi >= 0, 1, -1).astype(np.int8),
                                       dtype="bit", layout="OHWI_bit")
                mname = g.add_constant(f"{lname}/post_mul", mul)
                aname = g.add_constant(f"{lname}/post_add", bias) if bias is not None else ""
                out = g.add_node("QBConv2D", [tname(ins[0]), wname, mname, aname], attrs, name=lname)
                precision.append({"layer": lname, "op": "QBConv2D", "weights": "binary (per-layer mean|W| scale)",
                                  "activations": f"{k_bits}-bit unsigned (DoReFa, [0,1])", "pad_value": 0.0,
                                  "input_quantizer": iq, "kernel_quantizer": kq,
                                  "exact_zero_latent_weights": int(np.sum(kernel == 0))})
            elif iq in SIGN_QUANTIZERS and (kq in SIGN_QUANTIZERS or kq in SCALED_SIGN_QUANTIZERS or kq in XNOR_QUANTIZERS):
                assert groups == 1, "grouped binary conv not supported yet"
                _check_input_quantizer(lname, iq_obj, "sign")
                w_bin = np.where(w_ohwi >= 0, 1, -1).astype(np.int8)  # larq sign: 0 -> +1
                mul = None
                if kq in SCALED_SIGN_QUANTIZERS:
                    mul = np.mean(np.abs(kernel), axis=(0, 1, 2)).astype(np.float32)
                elif kq in XNOR_QUANTIZERS:
                    mul = np.mean(np.abs(np.clip(kernel.astype(np.float64), -1.0, 1.0)), axis=(0, 1, 2)).astype(np.float32)
                _check_kernel_quantizer(lname, kq_obj, kernel, np.where(kernel >= 0, 1.0, -1.0), mul)
                pad_value = float(getattr(layer, "pad_values", 0.0))
                attrs.update({"pad_value": pad_value, "input_binarize": "ge", "output": "f32"})
                wname = g.add_constant(f"{lname}/kernel_bits", w_bin, dtype="bit", layout="OHWI_bit")
                mname = g.add_constant(f"{lname}/post_mul", mul) if mul is not None else ""
                aname = g.add_constant(f"{lname}/post_add", bias) if bias is not None else ""
                out = g.add_node("BConv2D", [tname(ins[0]), wname, mname, aname, ""], attrs, name=lname)
                n_zero_w = int(np.sum(kernel == 0))
                precision.append({"layer": lname, "op": "BConv2D", "weights": "binary" + (" (per-channel mean|W| scale)" if mul is not None else ""),
                                  "activations": "binary", "pad_value": pad_value, "input_quantizer": iq, "kernel_quantizer": kq,
                                  "exact_zero_latent_weights": n_zero_w})
            elif _is_dorefa_w1(iq_obj) and _is_dorefa_w1(kq_obj):
                # BinaryFaceNet: DoReFa(k=1, 'weights') on inputs and kernels -> +-1 x +-1, no scale; grouped allowed.
                _check_dorefa_w1_input(lname, iq_obj)
                w_bin = np.transpose(_dorefa_w1_kernel_bits(lname, kq_obj, kernel), (3, 0, 1, 2))
                pad_value = float(getattr(layer, "pad_values", 0.0))
                if pad_value not in (-1.0, 0.0, 1.0):
                    raise NotImplementedError(f"{lname}: pad_values={pad_value}")
                attrs.update({"pad_value": pad_value, "input_binarize": "gt", "output": "f32"})
                consts = {f"{lname}/kernel_bits": (w_bin, {"dtype": "bit", "layout": "OHWI_bit"}), "pm": (None, {}),
                          f"{lname}/post_add": (bias, {})}
                _verify_binary_layer(layer, ins, "BConv2D", consts, attrs)
                wname = g.add_constant(f"{lname}/kernel_bits", w_bin, dtype="bit", layout="OHWI_bit")
                aname = g.add_constant(f"{lname}/post_add", bias) if bias is not None else ""
                out = g.add_node("BConv2D", [_dorefa_w1_operand(g, tname(ins[0]), x_shape, lname), wname, "", aname, ""],
                             attrs, name=lname)
                precision.append({"layer": lname, "op": "BConv2D" + (f" (groups={groups})" if groups > 1 else ""),
                                  "weights": "binary (DoReFa k=1 weights mode, no scale)",
                                  "activations": "binary (DoReFa k=1 weights mode: x > 2^-24 * tanh(max|x|), per sample)", "pad_value": pad_value,
                                  "input_quantizer": "DoReFa(k_bit=1, mode=weights)",
                                  "kernel_quantizer": "DoReFa(k_bit=1, mode=weights)"})
            else:
                raise NotImplementedError(f"{lname}: unsupported quantizer combination input={iq} kernel={kq}")
            bind(layer, out)

        elif cls in ("QuantDepthwiseConv2D", "QuantSeparableConv2D") and \
                _is_dorefa_w1(getattr(layer, "input_quantizer", None)):
            # BinaryFaceNet binary depthwise (-> BConv2D groups=C) and 1x1 separable (-> one 1x1 BConv2D).
            iq_obj, dq_obj = layer.input_quantizer, layer.depthwise_quantizer
            pq_obj = getattr(layer, "pointwise_quantizer", None)
            if not _is_dorefa_w1(dq_obj) or (cls == "QuantSeparableConv2D" and not _is_dorefa_w1(pq_obj)):
                raise NotImplementedError(f"{lname}: DoReFa(k=1) input with other kernel quantizers")
            _check_dorefa_w1_input(lname, iq_obj)
            dk = layer.depthwise_kernel.numpy()        # [kh, kw, C, mult]
            kh, kw, c, mult = dk.shape
            sh, sw = layer.strides
            dh, dw = layer.dilation_rate
            pad_value = float(getattr(layer, "pad_values", 0.0))
            if pad_value not in (-1.0, 0.0, 1.0):
                raise NotImplementedError(f"{lname}: pad_values={pad_value}")
            d_bits = _dorefa_w1_kernel_bits(lname, dq_obj, dk)
            bias = layer.bias.numpy() if layer.use_bias else None
            attrs = {"strides": [sh, sw], "dilations": [dh, dw],
                     "pads": _pads(layer.padding, x_shape[1], x_shape[2], kh, kw, sh, sw, dh, dw),
                     "activation": _activation_name(layer), "pad_value": pad_value, "input_binarize": "gt",
                     "output": "f32"}
            if cls == "QuantDepthwiseConv2D":
                w_bin = np.transpose(d_bits, (2, 3, 0, 1)).reshape(c * mult, kh, kw, 1)
                attrs["groups"] = c
                kind = f"BConv2D (depthwise, groups={c})"
            else:
                if (kh, kw, mult, sh, sw, dh, dw) != (1, 1, 1, 1, 1, 1, 1):
                    raise NotImplementedError(f"{lname}: only 1x1, stride-1, multiplier-1 separable convs are folded")
                p_bits = _dorefa_w1_kernel_bits(lname, pq_obj, layer.pointwise_kernel.numpy())   # [1, 1, C, O]
                w_bin = (p_bits[0, 0].T.astype(np.int32) * d_bits[0, 0, :, 0].astype(np.int32)[None, :])
                w_bin = w_bin.astype(np.int8).reshape(-1, 1, 1, c)
                attrs["groups"] = 1
                kind = "BConv2D (1x1 separable folded: pointwise * depthwise signs)"
            consts = {f"{lname}/kernel_bits": (w_bin, {"dtype": "bit", "layout": "OHWI_bit"}), "pm": (None, {}),
                      f"{lname}/post_add": (bias, {})}
            _verify_binary_layer(layer, ins, "BConv2D", consts, attrs)
            wname = g.add_constant(f"{lname}/kernel_bits", w_bin, dtype="bit", layout="OHWI_bit")
            aname = g.add_constant(f"{lname}/post_add", bias) if bias is not None else ""
            out = g.add_node("BConv2D", [_dorefa_w1_operand(g, tname(ins[0]), x_shape, lname), wname, "", aname, ""],
                             attrs, name=lname)
            precision.append({"layer": lname, "op": kind, "weights": "binary (DoReFa k=1 weights mode, no scale)",
                              "activations": "binary (DoReFa k=1 weights mode: x > 2^-24 * tanh(max|x|), per sample)", "pad_value": pad_value,
                              "input_quantizer": "DoReFa(k_bit=1, mode=weights)"})
            bind(layer, out)

        elif cls in ("QuantDepthwiseConv2D", "DepthwiseConv2D"):
            iq = _qname(getattr(layer, "input_quantizer", None))
            kq = _qname(getattr(layer, "depthwise_quantizer", None))
            if iq is not None or kq is not None:
                raise NotImplementedError(f"{lname}: quantized depthwise conv not supported")
            k = layer.depthwise_kernel.numpy()        # [kh, kw, C, mult]
            kh, kw, c, mult = k.shape
            w = np.transpose(k, (2, 3, 0, 1)).reshape(c * mult, kh, kw, 1)
            sh, sw = layer.strides
            dh, dw = layer.dilation_rate
            pads = _pads(layer.padding, x_shape[1], x_shape[2], kh, kw, sh, sw, dh, dw)
            bias = layer.bias.numpy() if layer.use_bias else None
            wname = g.add_constant(f"{lname}/kernel", w, layout="OHWI")
            bname = g.add_constant(f"{lname}/bias", bias) if bias is not None else ""
            out = g.add_node("Conv2D", [tname(ins[0]), wname, bname],
                             {"strides": [sh, sw], "dilations": [dh, dw], "pads": pads, "groups": c, "activation": _activation_name(layer)}, name=lname)
            precision.append({"layer": lname, "op": "DepthwiseConv2D", "weights": "fp32", "activations": "fp32"})
            bind(layer, out)

        elif cls == "BatchNormalization":
            axis = layer.axis if isinstance(layer.axis, (list, tuple)) else [layer.axis]
            if [int(a) % len(x_shape) for a in axis] != [len(x_shape) - 1]:
                raise NotImplementedError(f"{lname}: BatchNormalization over axis {layer.axis} (only the last axis)")
            gamma = layer.gamma.numpy() if layer.scale else 1.0
            beta = layer.beta.numpy() if layer.center else 0.0
            mean = layer.moving_mean.numpy()
            var = layer.moving_variance.numpy()
            inv = 1.0 / np.sqrt(var.astype(np.float64) + layer.epsilon)
            scale = (gamma * inv).astype(np.float32)
            shift = (beta - mean * gamma * inv).astype(np.float32)
            out = g.add_node("ChannelAffine", [tname(ins[0]), g.add_constant(f"{lname}/scale", scale),
                                               g.add_constant(f"{lname}/shift", shift)], name=lname)
            precision.append({"layer": lname, "op": "BatchNorm", "weights": "fp32", "activations": "fp32"})
            bind(layer, out)

        elif cls == "Activation":
            fn = layer.activation.__name__
            if fn == "relu":
                out = g.add_node("Activation", [tname(ins[0])], {"type": "relu"}, name=lname)
            elif fn == "softmax":
                out = g.add_node("Softmax", [tname(ins[0])], name=lname)
            elif fn == "linear":
                out = tname(ins[0])
            elif fn == "clip_by_value_activation":  # larq_zoo DoReFaNet: tf.clip_by_value(x, 0, 1)
                if not np.array_equal(_call_quantizer(layer.activation, _PROBE), np.clip(_PROBE, 0, 1)):
                    raise NotImplementedError(f"{lname}: clip_by_value_activation is not clip(x, 0, 1)")
                out = g.add_node("Activation", [tname(ins[0])], {"type": "hardtanh", "min": 0.0, "max": 1.0}, name=lname)
            else:
                raise NotImplementedError(f"{lname}: activation {fn}")
            bind(layer, out)

        elif cls == "Softmax":
            if int(layer.axis) not in (-1, len(ins[0].shape) - 1):
                raise NotImplementedError(f"{lname}: softmax over axis {layer.axis}")
            out = g.add_node("Softmax", [tname(ins[0])], name=lname)
            bind(layer, out)

        elif cls == "PReLU":
            alpha = layer.alpha.numpy()
            c = x_shape[-1]
            if list(layer.shared_axes or []) != [1, 2] or alpha.shape != (1, 1, c):
                raise NotImplementedError(f"{lname}: PReLU with shared_axes={layer.shared_axes}, alpha {alpha.shape}")
            out = g.add_node("Activation", [tname(ins[0]), g.add_constant(f"{lname}/slope", alpha.reshape(c))],
                             {"type": "prelu"}, name=lname)
            precision.append({"layer": lname, "op": "PReLU", "weights": "fp32", "activations": "fp32"})
            bind(layer, out)

        elif cls == "TFOpLambda" and getattr(layer, "symbol", "") == "reshape" and id(ins[0]) in lazy_sigmoid:
            # Legacy QuickNet-XL squeeze-and-excite: tf.reshape(sigmoid gate [N, C], [-1, 1, 1, C]).
            c_args = layer._inbound_nodes[0].call_args
            c = int(ins[0].shape[-1])
            if len(c_args) != 2 or [int(v) for v in c_args[1]] != [-1, 1, 1, c] or layer._inbound_nodes[0].call_kwargs:
                raise NotImplementedError(f"{lname}: tf.reshape of a sigmoid gate with args {c_args[1:]}")
            lazy_sigmoid[id(layer.output)] = lazy_sigmoid[id(ins[0])]

        elif cls == "Multiply" or (cls == "TFOpLambda" and getattr(layer, "symbol", "") == "math.multiply"):
            # Keras Multiply layer, or `x *= gate` (TFOpLambda tf.math.multiply, legacy QuickNet-XL SE).
            if cls == "TFOpLambda" and set(layer._inbound_nodes[0].call_kwargs) - {"name"}:
                raise NotImplementedError(f"{lname}: tf.math.multiply kwargs {layer._inbound_nodes[0].call_kwargs}")
            gates = [i for i, t in enumerate(ins) if id(t) in lazy_sigmoid]
            if len(ins) != 2 or len(gates) != 1:
                raise NotImplementedError(f"{lname}: Multiply is only supported as x * sigmoid(dense(.)) channel gating")
            x_t, gate = ins[1 - gates[0]], lazy_sigmoid[id(ins[gates[0]])]
            if len(x_t.shape) != 4 or [int(d) for d in ins[gates[0]].shape[1:]] != [1, 1, int(x_t.shape[-1])]:
                raise NotImplementedError(f"{lname}: unsupported gating shapes {x_t.shape} x {ins[gates[0]].shape}")
            out = g.add_node("ChannelGate", [tname(x_t), gate], {"gate": "sigmoid"}, name=lname)
            bind(layer, out)

        elif cls == "Lambda":
            fn = layer.function
            cv = inspect.getclosurevars(fn).nonlocals
            if "tf_op" in cv:  # larq_zoo.core.utils.TFOpLayer(tf_op, *args, **kwargs)
                op_name = getattr(cv["tf_op"], "__name__", "")
                args_, kw = tuple(cv.get("args", ())), dict(cv.get("kwargs", {}))
                axis = kw.get("axis", None)
                if axis not in (-1, 3):
                    raise NotImplementedError(f"{lname}: {op_name} over axis {axis}")
                if op_name == "split" and len(args_) == 1 and isinstance(args_[0], int):
                    groups = int(args_[0])
                    _verify_lambda(layer, ins, lambda a: np.split(a, groups, axis=-1))
                    for i, t in enumerate(layer.output):
                        split_pieces[id(t)] = {"src": tname(ins[0]), "groups": groups, "index": i}
                    continue
                if op_name == "concat" and not args_:
                    _verify_lambda(layer, ins, lambda *a: np.concatenate(a, axis=-1))
                    if all(id(t) in split_convs for t in ins):
                        out = _grouped_conv_from_splits(g, [split_convs[id(t)] for t in ins], lname, precision)
                    else:
                        out = g.add_node("Concat", [tname(t) for t in ins], {"axis": -1}, name=lname)
                    bind(layer, out)
                    continue
                raise NotImplementedError(f"{lname}: TFOpLayer {op_name} args={args_} kwargs={kw}")
            if set(cv) == {"f_in"} and len(ins) == 2:
                # MeliusNet improvement block: x + tf.pad(w, [[0,0],[0,0],[0,0],[f_in - C_w, 0]]).
                f_in, cw = int(cv["f_in"]), int(ins[1].shape[-1])
                if int(ins[0].shape[-1]) != f_in or cw > f_in:
                    raise NotImplementedError(f"{lname}: merge Lambda with f_in={f_in}, shapes {ins[0].shape}, {ins[1].shape}")
                _verify_lambda(layer, ins, lambda a, b: a + np.pad(b, [(0, 0), (0, 0), (0, 0), (f_in - cw, 0)]))
                padded = g.add_node("Pad", [tname(ins[1])], {"pads": [0, 0, 0, 0, 0, 0, f_in - cw, 0], "value": 0.0},
                                    name=f"{lname}/pad")
                out = g.add_node("Add", [tname(ins[0]), padded], name=lname)
                bind(layer, out)
                continue
            if not cv and len(ins) == 1:
                _verify_lambda(layer, ins, lambda a: a)  # identity (Real-to-Binary block outputs)
                bind(layer, tname(ins[0]))
                continue
            raise NotImplementedError(f"{lname}: unsupported Lambda (closure {sorted(cv)})")

        elif cls in ("Add",) or (cls == "TFOpLambda" and getattr(layer, "symbol", "") == "__operators__.add"):
            assert len(ins) == 2, f"{lname}: add with {len(ins)} inputs"
            out = g.add_node("Add", [tname(ins[0]), tname(ins[1])], name=lname)
            bind(layer, out)

        elif cls in ("MaxPooling2D", "AveragePooling2D"):
            kh, kw = layer.pool_size
            sh, sw = layer.strides
            pads = _pads(layer.padding, x_shape[1], x_shape[2], kh, kw, sh, sw)
            op = "MaxPool2D" if cls == "MaxPooling2D" else "AvgPool2D"
            attrs = {"kernel": [kh, kw], "strides": [sh, sw], "pads": pads}
            if op == "AvgPool2D":
                attrs["count_include_pad"] = 0  # TF excludes padding from the average
            out = g.add_node(op, [tname(ins[0])], attrs, name=lname)
            bind(layer, out)

        elif cls == "GlobalAveragePooling2D":
            keep = 1 if getattr(layer, "keepdims", False) else 0
            out = g.add_node("GlobalAvgPool", [tname(ins[0])], {"keepdims": keep}, name=lname)
            bind(layer, out)

        elif cls == "GlobalMaxPooling2D":
            # Fixed input size: max over the full H x W window (BRILLsson head after the larq_zoo backbone).
            if getattr(layer, "data_format", "channels_last") != "channels_last":
                raise NotImplementedError(f"{lname}: GlobalMaxPooling2D data_format {layer.data_format}")
            h, w = x_shape[1], x_shape[2]
            out = g.add_node("MaxPool2D", [tname(ins[0])], {"kernel": [h, w], "strides": [1, 1], "pads": [0, 0, 0, 0]},
                             name=lname if getattr(layer, "keepdims", False) else f"{lname}/window")
            if not getattr(layer, "keepdims", False):
                out = g.add_node("Reshape", [out], {"shape": [1, x_shape[3]]}, name=lname)
            bind(layer, out)

        elif cls == "Reshape" and id(ins[0]) in lazy_sigmoid:
            # [N, C] gate -> [N, 1, 1, C]: ChannelGate accepts the [N, C] gate directly.
            if [int(s) for s in layer.target_shape] != [1, 1, int(ins[0].shape[-1])]:
                raise NotImplementedError(f"{lname}: reshape of a sigmoid gate to {layer.target_shape}")
            lazy_sigmoid[id(layer.output)] = lazy_sigmoid[id(ins[0])]

        elif cls in ("Reshape", "Flatten"):
            target = [1] + [int(s) for s in layer.output.shape[1:]]
            out = g.add_node("Reshape", [tname(ins[0])], {"shape": target}, name=lname)
            bind(layer, out)

        elif cls in ("Dense", "QuantDense"):
            iq = _qname(getattr(layer, "input_quantizer", None))
            kq = _qname(getattr(layer, "kernel_quantizer", None))
            kernel = layer.kernel.numpy()             # [K, O]
            bias = layer.bias.numpy() if layer.use_bias else None
            if iq is None and kq is None and getattr(layer.activation, "__name__", None) == "sigmoid":
                # Sigmoid gate (Real-to-Binary Net): the sigmoid is evaluated inside ChannelGate.
                pre = g.add_node("Dense", [tname(ins[0]), g.add_constant(f"{lname}/kernel", kernel.T),
                                           g.add_constant(f"{lname}/bias", bias) if bias is not None else ""],
                                 {"activation": "none"}, name=f"{lname}/pre_sigmoid")
                precision.append({"layer": lname, "op": "Dense", "weights": "fp32", "activations": "fp32"})
                lazy_sigmoid[id(layer.output)] = pre
                continue
            if iq is None and kq is None:
                out = g.add_node("Dense", [tname(ins[0]), g.add_constant(f"{lname}/kernel", kernel.T),
                                           g.add_constant(f"{lname}/bias", bias) if bias is not None else ""],
                                 {"activation": _activation_name(layer)}, name=lname)
                precision.append({"layer": lname, "op": "Dense", "weights": "fp32", "activations": "fp32"})
            elif iq in SIGN_QUANTIZERS and kq in SIGN_QUANTIZERS | SCALED_SIGN_QUANTIZERS:
                w_bin = np.where(kernel.T >= 0, 1, -1).astype(np.int8)
                mul = np.mean(np.abs(kernel), axis=0).astype(np.float32) if kq in SCALED_SIGN_QUANTIZERS else None
                out = g.add_node("BDense", [tname(ins[0]), g.add_constant(f"{lname}/kernel_bits", w_bin, dtype="bit"),
                                            g.add_constant(f"{lname}/post_mul", mul) if mul is not None else "",
                                            g.add_constant(f"{lname}/post_add", bias) if bias is not None else "", ""],
                                 {"activation": _activation_name(layer), "input_binarize": "ge"}, name=lname)
                precision.append({"layer": lname, "op": "BDense", "weights": "binary", "activations": "binary"})
            elif iq in KBIT_ACT_QUANTIZERS and kq in LAYER_SCALED_SIGN_QUANTIZERS:
                iq_obj, kq_obj = layer.input_quantizer, layer.kernel_quantizer
                k_bits = int(iq_obj.precision)
                if getattr(iq_obj, "mode", "activations") != "activations":
                    raise NotImplementedError(f"{lname}: DoReFa input quantizer in mode {iq_obj.mode}")
                _check_input_quantizer(lname, iq_obj, "dorefa", k_bits)
                scale = float(np.mean(np.abs(kernel.astype(np.float64))))
                _check_kernel_quantizer(lname, kq_obj, kernel, np.where(kernel >= 0, 1.0, -1.0), scale)
                mul = np.full(kernel.shape[1], scale / (2 ** k_bits - 1), np.float64).astype(np.float32)
                out = g.add_node("QBDense", [tname(ins[0]),
                                             g.add_constant(f"{lname}/kernel_bits", np.where(kernel.T >= 0, 1, -1).astype(np.int8), dtype="bit"),
                                             g.add_constant(f"{lname}/post_mul", mul),
                                             g.add_constant(f"{lname}/post_add", bias) if bias is not None else ""],
                                 {"activation": _activation_name(layer), "act_bits": k_bits, "act_quant": "dorefa"}, name=lname)
                precision.append({"layer": lname, "op": "QBDense", "weights": "binary (per-layer mean|W| scale)",
                                  "activations": f"{k_bits}-bit unsigned (DoReFa, [0,1])", "input_quantizer": iq,
                                  "kernel_quantizer": kq, "exact_zero_latent_weights": int(np.sum(kernel == 0))})
            else:
                raise NotImplementedError(f"{lname}: dense quantizers input={iq} kernel={kq}")
            bind(layer, out)

        elif cls == "ZeroPadding2D":
            (pt, pb), (pl, pr) = layer.padding
            out = g.add_node("Pad", [tname(ins[0])], {"pads": [0, 0, pt, pb, pl, pr, 0, 0], "value": 0.0}, name=lname)
            bind(layer, out)

        elif cls == "Concatenate":
            out = g.add_node("Concat", [tname(t) for t in ins], {"axis": int(layer.axis)}, name=lname)
            bind(layer, out)

        else:
            raise NotImplementedError(f"layer {lname}: unsupported Keras layer type {cls}")

    outs = model.outputs
    g.set_outputs(*[tname(t) for t in outs])
    info = {"layer_precision": precision, "op_counts": g.op_counts()}
    return g, info
