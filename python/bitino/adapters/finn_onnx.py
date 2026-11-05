"""Reader: FINN-ONNX (MultiThreshold) and QONNX (Quant / BipolarQuant) CNV / FC graphs -> Bitino IR.

Sources handled (slice S7): finn-examples v0.0.1a ``cnv-w1a1.onnx`` / ``tfc-w1a1.onnx`` (FINN-ONNX after
FINN's ``streamline``-free export: Div 255 -> Mul 2 -> Sub 1 -> MultiThreshold, Conv/MatMul with ±1 float
initializers, ``Mul 1.0``, BatchNormalization (FC: Unsqueeze/BN/Squeeze), MultiThreshold BIPOLAR, MaxPool, a
scalar Sub/Div/Mul/Add head and TopK) and finn-examples v0.0.7a / QONNX zoo ``cnv_1w1a_gtsrb.onnx`` (Brevitas
QONNX export: Mul 2 -> Sub 1 -> Quant(8 bit), BipolarQuant on every weight and activation, Conv, BN, MaxPool,
Shape/Gather/Unsqueeze/Concat/Reshape flatten, Transpose+MatMul, scalar Sub/Pow/Div/Mul/Add head).

Semantics (qonnx 1.0.0 = the reference executor, ``qonnx.core.onnx_exec.execute_onnx``):
  * MultiThreshold: out = out_scale * #{i : x >= thr[c, i]} + out_bias (``vm >= thresholds``); BIPOLAR with
    one threshold 0, scale 2, bias -1 is ge-sign (sign(0) = +1).
  * BipolarQuant(scale 1): +1 if x >= 0 else -1 (``np.where(y >= 0, 1, -1) * scale``).
  * Quant: clamp(round(x/scale + zp), min_int, max_int) - zp) * scale, ROUND = numpy round (half to even).
  * The other nodes are standard ONNX (opset 11/12).

The reader never trusts names: every node on the activation path is matched against an explicit pattern
and anything else raises. Weight tensors (initializers, or BipolarQuant of initializers) must be exactly ±1.
The input quantizer is verified exhaustively: for every value of the declared input domain (uint8 pixels k,
given to the graph either as k or as float32(k)/255) the IR's ChannelAffine + FakeQuant (or ge threshold)
output must equal the graph's own quantizer output computed by the numpy interpreter below in float32.

``OnnxNumpy`` is a small numpy interpreter of exactly these op types (float32 or float64), used for constant
folding, for that exhaustive check, and by tools/convert_finn_onnx.py as the float64 evaluation of the
original graph (onnxruntime has no float64 Conv). Custom ops use qonnx's own numpy functions when qonnx is
importable (``multithreshold``, ``int_quant``, ``binary_quant``), else identical local implementations.
"""
from __future__ import annotations

import numpy as np

from .finn_bnn import InputSpec, Layer

try:  # the reference implementations (Apache-2.0); optional so the adapter itself only needs onnx + numpy
    from qonnx.custom_op.general.bipolar_quant import binary_quant as _q_binary_quant
    from qonnx.custom_op.general.intquant import int_quant as _q_int_quant
    from qonnx.custom_op.general.multithreshold import multithreshold as _q_multithreshold
except Exception:  # pragma: no cover
    _q_binary_quant = _q_int_quant = _q_multithreshold = None

_CONST_FOLDABLE = {"BipolarQuant", "Transpose", "Pow", "Shape", "Gather", "Unsqueeze", "Concat", "Mul", "Div", "Add", "Sub",
                   "Reshape", "Squeeze", "Identity"}


def _attrs(node) -> dict:
    import onnx

    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        out[a.name] = v.decode() if isinstance(v, bytes) else v
    return out


def _multithreshold(v, thr, out_scale, out_bias):
    if _q_multithreshold is not None:
        return _q_multithreshold(v, thr, out_scale, out_bias)
    # identical to qonnx: thresholds [C, T] (or [1, T] broadcast), channel axis 1, count v >= thr
    vm = v.reshape(v.shape[0], v.shape[1], -1)
    c = vm.shape[1]
    t = np.broadcast_to(thr, (c, thr.shape[1])) if thr.shape[0] == 1 else thr
    cnt = (vm[:, :, :, None] >= t[None, :, None, :]).sum(-1).astype(v.dtype)
    return (out_scale * cnt.reshape(v.shape) + out_bias).astype(v.dtype)


def _int_quant(x, scale, zp, bw, signed, narrow, mode):
    if _q_int_quant is not None:
        return _q_int_quant(x, scale, zp, bw, signed, narrow, mode)
    assert mode == "ROUND"
    lo = -(2 ** (bw - 1)) + (1 if narrow else 0) if signed else 0
    hi = 2 ** (bw - 1) - 1 if signed else 2 ** bw - 1 - (1 if narrow else 0)
    return (np.clip(np.round(x / scale + zp), lo, hi) - zp) * scale


def _binary_quant(x, scale):
    if _q_binary_quant is not None:
        return _q_binary_quant(x, scale)
    return np.where(x >= 0.0, 1.0, -1.0).astype(x.dtype) * scale


def _conv_nchw(x, w, attrs, ft):
    if attrs.get("group", 1) != 1 or list(attrs.get("dilations", [1, 1])) != [1, 1]:
        raise NotImplementedError("Conv: groups/dilations")
    pt, pl, pb, pr = (list(attrs.get("pads", [0, 0, 0, 0])) + [0, 0, 0, 0])[:4]
    sh, sw = attrs.get("strides", [1, 1])
    x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    n, c, h, wd = x.shape
    o, ci, kh, kw = w.shape
    oh, ow = (h - kh) // sh + 1, (wd - kw) // sw + 1
    cols = np.empty((n, oh, ow, c, kh, kw), ft)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, :, :, ky, kx] = x[:, :, ky:ky + sh * (oh - 1) + 1:sh, kx:kx + sw * (ow - 1) + 1:sw].transpose(0, 2, 3, 1)
    y = cols.reshape(n * oh * ow, -1).astype(np.float64) @ w.reshape(o, -1).astype(np.float64).T
    return y.astype(ft).reshape(n, oh, ow, o).transpose(0, 3, 1, 2)


class OnnxNumpy:
    """Numpy evaluation of a FINN-ONNX / QONNX graph in float32 or float64 (all float tensors in that dtype).

    Accumulations (Conv / MatMul) are done in float64 and rounded once: on these graphs every Conv/MatMul
    operand is ±1 or a multiple of 1/128, so the sums are exact in any order and precision."""

    def __init__(self, model):
        from onnx import numpy_helper

        self.model = model
        self.init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        self.nodes = list(model.graph.node)
        self.input_name = next(i.name for i in model.graph.input if i.name not in self.init)
        self.output_name = model.graph.output[0].name

    def run(self, x: np.ndarray, ft=np.float32, keep: bool = False, outputs: list | None = None) -> dict:
        env = {k: (v.astype(ft) if v.dtype.kind == "f" else v) for k, v in self.init.items()}
        env[self.input_name] = np.asarray(x, ft)
        for nd in self.nodes:
            a = _attrs(nd)
            ins = [env[i] if i else None for i in nd.input]
            op = nd.op_type
            if op == "Div":
                y = (ins[0] / ins[1]).astype(ft)
            elif op == "Mul":
                y = (ins[0] * ins[1]).astype(ins[0].dtype if ins[0].dtype.kind != "f" else ft)
            elif op == "Sub":
                y = (ins[0] - ins[1]).astype(ft)
            elif op == "Add":
                y = (ins[0] + ins[1]).astype(ft)
            elif op == "Pow":
                y = np.power(ins[0].astype(np.float64), ins[1].astype(np.float64)).astype(ft)
            elif op == "Reshape":
                shp = [int(s) for s in ins[1]]
                shp = [ins[0].shape[i] if s == 0 else s for i, s in enumerate(shp)]
                y = ins[0].reshape(shp)
            elif op == "Shape":
                y = np.array(ins[0].shape, np.int64)
            elif op == "Gather":
                y = np.take(ins[0], ins[1], axis=a.get("axis", 0))
            elif op == "Unsqueeze":
                y = ins[0]
                for ax in sorted(a["axes"]):
                    y = np.expand_dims(y, ax)
            elif op == "Squeeze":
                y = np.squeeze(ins[0], axis=tuple(a["axes"]))
            elif op == "Concat":
                y = np.concatenate([np.atleast_1d(v) for v in ins], axis=a["axis"])
            elif op == "Transpose":
                y = np.transpose(ins[0], a["perm"])
            elif op == "MatMul":
                y = (ins[0].astype(np.float64) @ ins[1].astype(np.float64)).astype(ft)
            elif op == "Conv":
                y = _conv_nchw(ins[0], ins[1], a, ft)
            elif op == "BatchNormalization":
                xx, gam, bet, mu, var = ins
                shp = (1, -1) + (1,) * (xx.ndim - 2)
                eps = ft(a.get("epsilon", 1e-5))
                y = ((xx - mu.reshape(shp)) / np.sqrt(var.reshape(shp) + eps) * gam.reshape(shp) + bet.reshape(shp)).astype(ft)
            elif op == "MaxPool":
                if list(a.get("pads", [0, 0, 0, 0])) != [0, 0, 0, 0] or list(a["kernel_shape"]) != [2, 2] or list(a.get("strides", [1, 1])) != [2, 2]:
                    raise NotImplementedError("MaxPool: only 2x2/2 VALID")
                n, c, h, w = ins[0].shape
                y = ins[0][:, :, :h // 2 * 2, :w // 2 * 2].reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))
            elif op == "MultiThreshold":
                y = _multithreshold(ins[0], ins[1].astype(ft), ft(a.get("out_scale", 1.0)), ft(a.get("out_bias", 0.0))).astype(ft)
            elif op == "BipolarQuant":
                y = _binary_quant(ins[0], ins[1].astype(ft)).astype(ft)
            elif op == "Quant":
                y = _int_quant(ins[0], ins[1].astype(ft), ins[2].astype(ft), ins[3], bool(a.get("signed", 1)), bool(a.get("narrow", 0)),
                               a.get("rounding_mode", "ROUND")).astype(ft)
            elif op == "TopK":
                k = int(ins[1][0])
                idx = np.argsort(-ins[0], axis=a.get("axis", -1), kind="stable")[..., :k]
                y = [np.take_along_axis(ins[0], idx, axis=-1), idx.astype(np.int64)]
            elif op == "Identity":
                y = ins[0]
            else:
                raise NotImplementedError(f"OnnxNumpy: op {op}")
            if isinstance(y, list):
                for o, v in zip(nd.output, y):
                    env[o] = v
            else:
                env[nd.output[0]] = y
        if keep:
            return env
        return {o: env[o] for o in (outputs or [self.output_name])}


def _input_dependent(model, input_name: str) -> set:
    dep = {input_name}
    for nd in model.graph.node:
        if any(i in dep for i in nd.input) and nd.op_type != "Shape":
            dep.update(nd.output)
    return dep


class _Chain:
    """The activation path as a list of nodes, each with exactly one input-dependent input."""

    def __init__(self, model, consts: dict, dep: set):
        self.nodes = [nd for nd in model.graph.node if any(o in dep for o in nd.output)]
        self.consts = consts
        self.i = 0

    def peek(self, k: int = 0):
        j = self.i + k
        return self.nodes[j] if j < len(self.nodes) else None

    def take(self, op: str | tuple):
        nd = self.peek()
        ops = (op,) if isinstance(op, str) else op
        if nd is None or nd.op_type not in ops:
            raise NotImplementedError(f"expected {ops} at chain position {self.i}, got {nd.op_type if nd else None}")
        self.i += 1
        return nd

    def const(self, nd, idx: int) -> np.ndarray:
        name = nd.input[idx]
        if name not in self.consts:
            raise NotImplementedError(f"{nd.op_type} {nd.name}: input {idx} is not constant")
        return self.consts[name]

    def scalar(self, nd, idx: int = 1) -> float:
        v = np.asarray(self.const(nd, idx), np.float64)
        if v.size != 1:
            raise NotImplementedError(f"{nd.op_type} {nd.name}: expected a scalar constant, got shape {v.shape}")
        return float(v.reshape(-1)[0])


def _pm1(w: np.ndarray, what: str) -> np.ndarray:
    w = np.asarray(w, np.float64)
    if not np.all((w == 1.0) | (w == -1.0)):
        raise NotImplementedError(f"{what}: weights are not exactly ±1")
    return w.astype(np.int8)


def _is_bipolar(nd, chain) -> bool:
    if nd is None:
        return False
    if nd.op_type == "MultiThreshold":
        a = _attrs(nd)
        t = np.asarray(chain.const(nd, 1), np.float64)
        return a.get("out_dtype") == "BIPOLAR" and t.size == 1 and float(t.reshape(-1)[0]) == 0.0 \
            and float(a.get("out_scale", 1.0)) == 2.0 and float(a.get("out_bias", 0.0)) == -1.0
    if nd.op_type == "BipolarQuant":
        return chain.scalar(nd, 1) == 1.0
    return False


def read(model, input_domain: str):
    """Returns (InputSpec, [Layer], info). input_domain: "pixels" (graph input = float32(k)) or "unit"
    (graph input = float32(k)/255), k = 0..255; used for the exhaustive input-quantizer check."""
    ev = OnnxNumpy(model)
    in_shape = [d.dim_value for d in next(i for i in model.graph.input if i.name == ev.input_name).type.tensor_type.shape.dim]
    if len(in_shape) != 4 or in_shape[0] != 1:
        raise NotImplementedError(f"input shape {in_shape}")
    _, c_in, h_in, w_in = in_shape
    # constant folding in float64 (weights are ±1 either way; BN / head constants keep the graph's own
    # float64 semantics, e.g. QONNX's Pow(running_var + eps, 0.5))
    env64 = ev.run(np.zeros(in_shape, np.float64), np.float64, keep=True)
    dep = _input_dependent(model, ev.input_name)
    consts = {k: v for k, v in env64.items() if k not in dep}
    chain = _Chain(model, consts, dep)
    info = {"chain_ops": [nd.op_type for nd in chain.nodes], "input_shape_nchw": in_shape}

    # ---------------------------------------------------------------- input path
    pre = []
    while chain.peek() is not None and chain.peek().op_type in ("Div", "Mul", "Sub", "Add", "Reshape"):
        nd = chain.take(("Div", "Mul", "Sub", "Add", "Reshape"))
        if nd.op_type != "Reshape":
            pre.append((nd.op_type, chain.scalar(nd)))
    first_q = chain.peek()
    dom = np.arange(256, dtype=np.float64)
    x_dom = dom if input_domain == "pixels" else dom / 255.0

    def src_pre(xv, ft):
        y = np.asarray(xv, ft)
        for op, c in pre:
            y = {"Div": y / ft(c), "Mul": y * ft(c), "Sub": y - ft(c), "Add": y + ft(c)}[op].astype(ft)
        return y

    # composite affine of the scalar pre-chain, as a single x*a + b (float64, rounded once to float32)
    a64, b64 = 1.0, 0.0
    for op, c in pre:
        a64, b64 = {"Div": (a64 / c, b64 / c), "Mul": (a64 * c, b64 * c), "Sub": (a64, b64 - c), "Add": (a64, b64 + c)}[op]
    # the graph input values of the domain, as the eval sets store them (float32)
    xin32 = dom.astype(np.float32) if input_domain == "pixels" else (dom.astype(np.float32) / np.float32(255.0)).astype(np.float32)
    x_dom = xin32.astype(np.float64)

    if first_q.op_type == "MultiThreshold" and _attrs(first_q).get("out_dtype") == "UINT8":
        mt = chain.take("MultiThreshold")
        thr = np.asarray(chain.const(mt, 1), np.float32)
        add = chain.take("Add")
        mul = chain.take("Mul")
        off, sc = chain.scalar(add), chain.scalar(mul)
        if thr.shape != (c_in, 255):
            raise NotImplementedError(f"input MultiThreshold thresholds {thr.shape}")
        src_q = []
        for ch in range(c_in):
            v = src_pre(xin32, np.float32)
            cnt = (v[:, None] >= thr[ch][None, :]).sum(1).astype(np.float32)
            src_q.append((cnt + np.float32(off)) * np.float32(sc))
        qmin, qmax = int(round(0 + off)), int(round(255 + off))
        q = {"scale": sc, "qmin": qmin, "qmax": qmax}
        src_desc = f"FINN MultiThreshold UINT8 ({c_in}x255 thresholds) + Add {off:g} + Mul {sc:g}"
    elif first_q.op_type == "Quant":
        qn = chain.take("Quant")
        qa = _attrs(qn)
        sc, zp, bw = chain.scalar(qn, 1), chain.scalar(qn, 2), int(chain.scalar(qn, 3))
        if zp != 0.0 or qa.get("rounding_mode", "ROUND") != "ROUND" or not qa.get("signed", 1) or qa.get("narrow", 0):
            raise NotImplementedError(f"input Quant {qa} zp {zp}")
        qmin, qmax = -(2 ** (bw - 1)), 2 ** (bw - 1) - 1
        v = src_pre(xin32, np.float32)
        src_q = [_int_quant(v, np.float32(sc), np.float32(0), np.float32(bw), True, False, "ROUND").astype(np.float32)] * c_in
        q = {"scale": sc, "qmin": qmin, "qmax": qmax}
        src_desc = f"QONNX Quant (signed {bw} bit, scale {sc:g}, zp 0, ROUND = half to even)"
    elif _is_bipolar(first_q, chain):
        bq = chain.take(("MultiThreshold", "BipolarQuant"))
        v = src_pre(xin32, np.float32)
        src_b = np.where(v >= 0, 1, -1)
        pos = np.where(src_b > 0)[0]
        if pos.size == 0 or not np.all(src_b[pos[0]:] == 1) or not np.all(src_b[:pos[0]] == -1):
            raise NotImplementedError("binary input quantizer is not a single threshold on the input domain")
        t = float((x_dom[pos[0]] + x_dom[pos[0] - 1]) / 2.0) if pos[0] > 0 else float(x_dom[0] - 1.0)
        ours = np.where(xin32.astype(np.float32) >= np.float32(t), 1, -1)
        if not np.array_equal(ours, src_b):
            raise NotImplementedError("binary input threshold check failed")
        info["input_quantizer_check"] = {"domain": input_domain, "n_values": 256, "identical": True, "threshold": t,
                                         "source": f"{' -> '.join(f'{o} {c:g}' for o, c in pre)} -> {bq.op_type} (ge 0)"}
        inp = InputSpec("binary_threshold", [1, h_in, w_in, c_in], threshold=t,
                        source=f"input pre-chain {pre} then {bq.op_type} bipolar: exactly 'pixel >= {t}' on uint8 pixels")
        return inp, _read_layers(chain, c_in, h_in, w_in, False, info), info
    else:
        raise NotImplementedError(f"unsupported input quantizer {first_q.op_type}")

    # int8 input: IR = ChannelAffine(a, b) + FakeQuant(scale, 0, qmin, qmax); verify on all 256 inputs.
    a32, b32 = np.float32(a64), np.float32(b64)
    from ..executor import _fake_quant
    xa = (xin32 * a32 + b32).astype(np.float32)
    ours = _fake_quant(xa, np.full(256, sc, np.float32), np.zeros(256, np.float32),
                       {"axis": -1, "pre_scale": 1.0, "qmin": qmin, "qmax": qmax}, np.float32)
    same = all(np.array_equal(ours, s) for s in src_q)
    # also in float64 (the float64 IR evaluation sees the same quantized values)
    xa64 = np.asarray(x_dom, np.float64) * float(a32) + float(b32)
    ours64 = np.clip(np.rint(xa64 / sc), qmin, qmax) * sc
    same64 = bool(np.array_equal(ours64, ours.astype(np.float64)))
    info["input_quantizer_check"] = {"domain": input_domain, "n_values": 256, "identical_f32": bool(same), "identical_f64_ir": same64,
                                     "affine": [float(a32), float(b32)], "source": f"{' -> '.join(f'{o} {c:g}' for o, c in pre)} -> {src_desc}",
                                     "min_distance_to_rounding_tie": float(np.min(np.abs(np.abs(xa64 / sc - np.floor(xa64 / sc)) - 0.5)))}
    if not (same and same64):
        raise NotImplementedError(f"input quantizer check failed: {info['input_quantizer_check']}")
    inp = InputSpec("int8", [1, h_in, w_in, c_in], affine=(float(a32), float(b32)), q=q, source=src_desc)
    return inp, _read_layers(chain, c_in, h_in, w_in, True, info), info


def _read_layers(chain: _Chain, c, h, w, int8_input: bool, info: dict) -> list:
    layers = []
    li = 0
    while chain.peek() is not None:
        nd = chain.peek()
        if nd.op_type in ("Reshape",):
            chain.take("Reshape")
            continue
        lin = chain.take(("Conv", "MatMul"))
        wt = chain.const(lin, 1)
        if lin.op_type == "Conv":
            a = _attrs(lin)
            if list(a.get("pads", [0, 0, 0, 0])) != [0, 0, 0, 0] or list(a.get("strides", [1, 1])) != [1, 1] \
                    or list(a.get("kernel_shape", [3, 3])) != [3, 3] or a.get("group", 1) != 1 or list(a.get("dilations", [1, 1])) != [1, 1]:
                raise NotImplementedError(f"{lin.name}: only 3x3 stride-1 VALID convs")
            wq = _pm1(wt, lin.name)
            kind = "conv"
        else:
            wq = _pm1(np.asarray(wt).T, lin.name)   # MatMul weight [in, out] -> [O, I]
            kind = "dense"
        name = f"{kind}{li}"
        acc_scale = 1.0
        while chain.peek() is not None and chain.peek().op_type == "Mul" and np.asarray(chain.const(chain.peek(), 1)).size == 1:
            acc_scale *= chain.scalar(chain.take("Mul"))
        nd = chain.peek()
        if nd is not None and nd.op_type in ("Unsqueeze", "BatchNormalization"):
            if nd.op_type == "Unsqueeze":
                chain.take("Unsqueeze")
            bn = chain.take("BatchNormalization")
            if chain.peek() is not None and chain.peek().op_type == "Squeeze":
                chain.take("Squeeze")
            gam, bet, mu, var = (np.asarray(chain.const(bn, i), np.float64) for i in (1, 2, 3, 4))
            eps = float(np.float32(_attrs(bn).get("epsilon", 1e-5)))
            sd = np.sqrt(var + eps)
            scale = acc_scale * gam / sd
            shift = bet - mu * gam / sd
            src = f"{lin.op_type} (±1) -> " + (f"Mul {acc_scale:g} -> " if acc_scale != 1.0 else "") + f"BatchNormalization (eps {eps:g})"
            if not _is_bipolar(chain.peek(), chain):
                raise NotImplementedError(f"{name}: BN not followed by a bipolar quantizer")
            q = chain.take(("MultiThreshold", "BipolarQuant"))
            pool = None
            if chain.peek() is not None and chain.peek().op_type == "MaxPool":
                mp = chain.take("MaxPool")
                pa = _attrs(mp)
                if list(pa["kernel_shape"]) != [2, 2] or list(pa.get("strides", [1, 1])) != [2, 2] or list(pa.get("pads", [0, 0, 0, 0])) != [0, 0, 0, 0]:
                    raise NotImplementedError(f"{mp.name}: only MaxPool 2x2/2")
                pool = "after_sign"
            layers.append(Layer(name, kind, wq, scale, shift, pool=pool, w1a8=(li == 0 and int8_input and kind == "conv"),
                                source=src + f" -> {q.op_type} (ge-sign)" + (" -> MaxPool 2x2" if pool else ""),
                                extra={"onnx_nodes": [lin.name, bn.name, q.name], "bn_eps": eps, "negative_scale_channels": int(np.sum(scale < 0))}))
        else:
            # final layer: scalar affine head (Sub / Div / Mul / Add / Pow-folded constants), optional TopK
            m, b = acc_scale, 0.0
            ops = []
            while chain.peek() is not None and chain.peek().op_type in ("Sub", "Div", "Mul", "Add"):
                o = chain.take(("Sub", "Div", "Mul", "Add"))
                cst = chain.scalar(o)
                ops.append(f"{o.op_type} {cst:.9g}")
                m, b = {"Sub": (m, b - cst), "Div": (m / cst, b / cst), "Mul": (m * cst, b * cst), "Add": (m, b + cst)}[o.op_type]
            top = None
            if chain.peek() is not None and chain.peek().op_type == "TopK":
                top = chain.take("TopK")
            if chain.peek() is not None:
                raise NotImplementedError(f"unexpected node after the head: {chain.peek().op_type}")
            o = wq.shape[0]
            layers.append(Layer(name, kind, wq, np.full(o, m), np.full(o, b), final=True,
                                source=f"{lin.op_type} (±1) -> scalar affine [{', '.join(ops)}]" + (" (graph output = TopK index; IR outputs the logits before it)" if top else ""),
                                extra={"logits_tensor": (lin.output[0] if not ops else None), "topk_dropped": top is not None}))
            info["logits_onnx_tensor"] = (top.input[0] if top is not None else chain.nodes[-1].output[0])
            break
        li += 1
    return layers
