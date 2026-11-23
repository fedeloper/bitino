"""General numpy evaluator of ONNX / QONNX graphs in float64 (or float32), the float64 reference of the QONNX route.

docs/qonnx-frontend.md § Verification: onnxruntime has no float64 Conv, so the float64 reference of an exported
QONNX graph is this evaluator. It generalizes ``bitino.adapters.finn_onnx.OnnxNumpy`` (which stays as it is for
tools/convert_finn_onnx.py) to the operator set of the phase-1 importer (``bitino.qonnx_import``).

* Every floating-point tensor is held in the working dtype ``ft`` (float64 for the reference, float32 for a
  cross-check against qonnx's ``execute_onnx``); initializers are cast to it once. Integer tensors (shapes,
  indices, TopK indices) stay integer.
* Conv / MatMul / Gemm accumulate in float64 (BLAS order) and round once to ``ft``.
* Quantizers: ``BipolarQuant``, ``Quant`` / ``IntQuant``, ``MultiThreshold`` (qonnx / finn domains, also the legacy
  ``onnx.brevitas`` domain of Brevitas exports) use qonnx's own numpy functions when qonnx is importable, else
  identical local implementations. ``Trunc`` and ``FloatQuant`` are not implemented (raise).
* Opset differences handled: Unsqueeze / Squeeze / ReduceMean axes as attribute or input, Clip min/max as
  attributes (opset < 11) or inputs, Pad pads as attribute or input, Softmax axis default (1 before opset 13,
  -1 from 13, with the pre-13 flatten semantics).

Unknown operators raise ``NotImplementedError`` naming the node.
"""
from __future__ import annotations

import math

import numpy as np

try:  # qonnx reference implementations (Apache-2.0); optional
    from qonnx.custom_op.general.bipolar_quant import binary_quant as _q_binary_quant
    from qonnx.custom_op.general.intquant import int_quant as _q_int_quant
    from qonnx.custom_op.general.multithreshold import multithreshold as _q_multithreshold
except Exception:  # pragma: no cover - qonnx not installed
    _q_binary_quant = _q_int_quant = _q_multithreshold = None

_ERF = np.vectorize(math.erf, otypes=[np.float64])


class OnnxF64Unsupported(NotImplementedError):
    """An operator (or attribute) this evaluator does not implement; .node is the ONNX node."""

    def __init__(self, node, msg: str):
        self.node = node
        super().__init__(msg)


def attrs(node) -> dict:
    import onnx

    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, onnx.TensorProto):
            from onnx import numpy_helper
            v = numpy_helper.to_array(v)
        out[a.name] = v
    return out


def opset_version(model, domain: str = "") -> int:
    for o in model.opset_import:
        if (o.domain or "") in (domain, "ai.onnx" if domain == "" else domain):
            return int(o.version)
    return 13


# ------------------------------------------------------------------------------------------------ quantizers
def binary_quant(x, scale):
    if _q_binary_quant is not None:
        return _q_binary_quant(x, scale)
    return np.where(x >= 0.0, 1.0, -1.0).astype(x.dtype) * scale


def int_quant(x, scale, zp, bw, signed, narrow, mode):
    if _q_int_quant is not None:
        return _q_int_quant(x, scale, zp, bw, signed, narrow, mode)
    if mode != "ROUND":
        raise NotImplementedError(f"IntQuant rounding_mode {mode}")
    bw = float(np.asarray(bw).reshape(-1)[0])
    lo = -(2 ** (bw - 1)) + (1 if narrow else 0) if signed else 0
    hi = 2 ** (bw - 1) - 1 if signed else 2 ** bw - 1 - (1 if narrow else 0)
    return (np.clip(np.round(x / scale + zp), lo, hi) - zp) * scale


def multithreshold(v, thr, out_scale, out_bias, data_layout="NCHW"):
    if data_layout != "NCHW":
        raise NotImplementedError(f"MultiThreshold data_layout {data_layout}")
    if _q_multithreshold is not None:
        return _q_multithreshold(v, thr, out_scale, out_bias)
    vm = v.reshape(v.shape[0], v.shape[1], -1)
    c = vm.shape[1]
    t = np.broadcast_to(thr, (c, thr.shape[1])) if thr.shape[0] == 1 else thr
    cnt = (vm[:, :, :, None] >= t[None, :, None, :]).sum(-1).astype(v.dtype)
    return (out_scale * cnt.reshape(v.shape) + out_bias).astype(v.dtype)


# ------------------------------------------------------------------------------------------------ helpers
def _np_dtype(onnx_type: int):
    import onnx

    return onnx.helper.tensor_dtype_to_np_dtype(onnx_type)


def _pads_nd(p, nd: int) -> tuple[list, list]:
    p = list(p) if p is not None else [0] * (2 * nd)
    return p[:nd], p[nd:]


def conv_nchw(x, w, b, a, ft):
    """ONNX Conv (2-D, NCHW, OIHW), zero padding, groups, dilations; float64 accumulation."""
    if a.get("auto_pad", "NOTSET") not in ("NOTSET", ""):
        raise NotImplementedError(f"Conv auto_pad {a['auto_pad']}")
    if x.ndim != 4:
        raise NotImplementedError("Conv: only 2-D convolutions")
    g = int(a.get("group", 1))
    dh, dw = a.get("dilations", [1, 1])
    sh, sw = a.get("strides", [1, 1])
    (pt, pl), (pb, pr) = _pads_nd(a.get("pads"), 2)
    x = np.pad(np.asarray(x, np.float64), ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    n, c, h, wd = x.shape
    o, cg, kh, kw = w.shape
    oh = (h - dh * (kh - 1) - 1) // sh + 1
    ow = (wd - dw * (kw - 1) - 1) // sw + 1
    out = np.empty((n, o, oh, ow), np.float64)
    og = o // g
    w64 = np.asarray(w, np.float64)
    for gi in range(g):
        xs = x[:, gi * cg:(gi + 1) * cg]
        cols = np.empty((n, oh, ow, cg, kh, kw), np.float64)
        for ky in range(kh):
            for kx in range(kw):
                y0, x0 = ky * dh, kx * dw
                cols[:, :, :, :, ky, kx] = xs[:, :, y0:y0 + sh * (oh - 1) + 1:sh, x0:x0 + sw * (ow - 1) + 1:sw].transpose(0, 2, 3, 1)
        wg = w64[gi * og:(gi + 1) * og].reshape(og, -1)
        out[:, gi * og:(gi + 1) * og] = (cols.reshape(n * oh * ow, -1) @ wg.T).reshape(n, oh, ow, og).transpose(0, 3, 1, 2)
    if b is not None:
        out = out + np.asarray(b, np.float64).reshape(1, -1, 1, 1)
    return out.astype(ft)


def pool_nchw(x, a, kind: str, ft):
    k = list(a["kernel_shape"])
    if len(k) != 2:
        raise NotImplementedError(f"{kind}Pool: only 2-D")
    if a.get("auto_pad", "NOTSET") not in ("NOTSET", ""):
        raise NotImplementedError(f"{kind}Pool auto_pad {a['auto_pad']}")
    if list(a.get("dilations", [1, 1])) != [1, 1]:
        raise NotImplementedError(f"{kind}Pool dilations")
    kh, kw = k
    sh, sw = a.get("strides", [1, 1])
    (pt, pl), (pb, pr) = _pads_nd(a.get("pads"), 2)
    n, c, h, w = x.shape
    ceil = bool(a.get("ceil_mode", 0))
    rnd = math.ceil if ceil else math.floor
    oh = int(rnd((h + pt + pb - kh) / sh)) + 1
    ow = int(rnd((w + pl + pr - kw) / sw)) + 1
    out = np.empty((n, c, oh, ow), ft)
    include_pad = bool(a.get("count_include_pad", 0))
    for oy in range(oh):
        for ox in range(ow):
            y0, x0 = oy * sh - pt, ox * sw - pl
            ys, ye, xs, xe = max(y0, 0), min(y0 + kh, h), max(x0, 0), min(x0 + kw, w)
            win = x[:, :, ys:ye, xs:xe]
            if kind == "max":
                out[:, :, oy, ox] = win.max(axis=(2, 3))
            else:
                if include_pad:
                    cnt = (min(y0 + kh, h + pb) - y0) * (min(x0 + kw, w + pr) - x0)
                else:
                    cnt = (ye - ys) * (xe - xs)
                out[:, :, oy, ox] = (np.asarray(win, np.float64).sum(axis=(2, 3)) / cnt).astype(ft)
    return out


def _axes(node_attrs, ins, idx=1):
    if "axes" in node_attrs:
        return [int(v) for v in node_attrs["axes"]]
    if len(ins) > idx and ins[idx] is not None:
        return [int(v) for v in np.asarray(ins[idx]).reshape(-1)]
    return None


# ------------------------------------------------------------------------------------------------ evaluator
class OnnxF64:
    """Numpy evaluation of an ONNX / QONNX graph. ``run`` returns the requested tensors (default: graph outputs)."""

    def __init__(self, model):
        from onnx import numpy_helper

        self.model = model
        self.opset = opset_version(model)
        self.init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        self.nodes = list(model.graph.node)
        self.inputs = [i.name for i in model.graph.input if i.name not in self.init]
        self.input_name = self.inputs[0]
        self.output_names = [o.name for o in model.graph.output]
        self.output_name = self.output_names[0]
        # float64 carriers (docs/qonnx-profile.md R7): a DOUBLE initializer (or Constant) read through
        # Cast(to=FLOAT) keeps its float64 value in a float64 evaluation; a float32 evaluation sees fl32(c)
        self.double_consts = {k for k, v in self.init.items() if v.dtype == np.float64}
        for nd in self.nodes:
            if nd.op_type == "Constant":
                v = attrs(nd).get("value")
                if isinstance(v, np.ndarray) and v.dtype == np.float64:
                    self.double_consts.add(nd.output[0])

    def input_shape(self, name: str | None = None) -> list:
        name = name or self.input_name
        vi = next(i for i in self.model.graph.input if i.name == name)
        return [d.dim_value for d in vi.type.tensor_type.shape.dim]

    def run(self, x, ft=np.float64, keep=False, outputs: list | None = None, shapes: dict | None = None) -> dict:
        """keep: True = return every tensor; a set of names = return those plus the outputs, freeing every other
        intermediate after its last use (peak memory); False = the outputs only."""
        env = {k: (v.astype(ft) if v.dtype.kind == "f" else v) for k, v in self.init.items()}
        feeds = x if isinstance(x, dict) else {self.input_name: x}
        for k, v in feeds.items():
            v = np.asarray(v)
            env[k] = v if v.dtype.kind in "iub" else v.astype(ft)      # token ids stay integer
        free_after: dict = {}
        if keep is not True and keep is not False:
            retain = set(keep) | set(outputs or self.output_names) | set(self.init) | set(feeds)
            last: dict = {}
            for j, nd in enumerate(self.nodes):
                for i in nd.input:
                    if i:
                        last[i] = j
            for t, j in last.items():
                if t not in retain:
                    free_after.setdefault(j, []).append(t)
        for j, nd in enumerate(self.nodes):
            try:
                ys = self._node(nd, [env[i] if i else None for i in nd.input], ft)
            except KeyError as e:
                raise OnnxF64Unsupported(nd, f"OnnxF64: node {nd.name!r} ({nd.op_type}): missing input {e}") from None
            except OnnxF64Unsupported:
                raise
            except NotImplementedError as e:
                raise OnnxF64Unsupported(nd, str(e)) from None
            for o, y in zip(nd.output, ys):
                if o:
                    env[o] = y
                    if shapes is not None:
                        shapes[o] = tuple(np.shape(y))
            for t in free_after.get(j, ()):
                env.pop(t, None)
        if keep is not False:          # True or a set of names (possibly empty: the retained constants)
            return env
        return {o: env[o] for o in (outputs or self.output_names)}

    def _node(self, nd, ins, ft) -> list:
        a = attrs(nd)
        op = nd.op_type

        def f(v):
            v = np.asarray(v)
            return v.astype(ft) if v.dtype.kind == "f" else v

        x = ins[0] if ins else None
        if op == "Constant":
            if "value" in a:
                return [f(a["value"])]
            for k in ("value_float", "value_int"):
                if k in a:
                    return [f(np.array(a[k]))]
            for k in ("value_floats", "value_ints"):
                if k in a:
                    return [f(np.array(a[k]))]
            raise NotImplementedError(f"Constant {nd.name}: {list(a)}")
        if op in ("Identity",):
            return [x]
        if op == "Dropout":
            return [x, np.ones(np.shape(x), bool)]
        if op == "Cast":
            dt = _np_dtype(a["to"])
            if ft == np.float64 and dt == np.float32 and nd.input[0] in self.double_consts:
                return [np.asarray(x, np.float64)]          # R7 float64 carrier: read at its float64 value
            return [f(np.asarray(x).astype(dt))]
        if op in ("Add", "Sub", "Mul", "Div"):
            u, v = ins
            if np.asarray(u).dtype.kind in "iub" and np.asarray(v).dtype.kind in "iub":
                r = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply}.get(op)
                return [r(u, v) if r else (np.asarray(u) // np.asarray(v))]
            r = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}[op](f(u).astype(ft), f(v).astype(ft))
            return [r.astype(ft)]
        if op == "Pow":
            return [np.power(np.asarray(ins[0], np.float64), np.asarray(ins[1], np.float64)).astype(ft)]
        if op in ("Sqrt", "Neg", "Abs", "Exp", "Log", "Reciprocal", "Floor", "Ceil", "Round"):
            fn = {"Sqrt": np.sqrt, "Neg": np.negative, "Abs": np.abs, "Exp": np.exp, "Log": np.log,
                  "Reciprocal": np.reciprocal, "Floor": np.floor, "Ceil": np.ceil, "Round": np.round}[op]
            return [f(fn(np.asarray(x, np.float64) if np.asarray(x).dtype.kind == "f" else x))]
        if op == "Erf":
            return [_ERF(np.asarray(x, np.float64)).astype(ft)]
        if op in ("Max", "Min", "Sum", "Mean"):
            r = np.asarray(ins[0], np.float64)
            for v in ins[1:]:
                r = {"Max": np.maximum, "Min": np.minimum, "Sum": np.add, "Mean": np.add}[op](r, np.asarray(v, np.float64))
            if op == "Mean":
                r = r / len(ins)
            return [r.astype(ft)]
        if op in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal"):
            fn = {"Greater": np.greater, "GreaterOrEqual": np.greater_equal, "Less": np.less,
                  "LessOrEqual": np.less_equal, "Equal": np.equal}[op]
            return [fn(ins[0], ins[1])]
        if op == "Not":
            return [np.logical_not(x)]
        if op in ("And", "Or", "Xor"):
            fn = {"And": np.logical_and, "Or": np.logical_or, "Xor": np.logical_xor}[op]
            return [fn(np.asarray(ins[0], bool), np.asarray(ins[1], bool))]
        if op == "IsNaN":
            return [np.isnan(np.asarray(x))]
        if op == "BitShift":
            u, n = np.asarray(ins[0]), np.asarray(ins[1])
            if u.dtype.kind != "u" or n.dtype.kind != "u":
                raise NotImplementedError(f"BitShift {nd.name}: unsigned integer operands only (ONNX)")
            d = a.get("direction", "RIGHT")
            d = d.decode() if isinstance(d, bytes) else d
            return [(np.right_shift(u, n) if d == "RIGHT" else np.left_shift(u, n)).astype(u.dtype)]
        if op == "Where":
            c, u, v = ins
            r = np.where(c, u, v)
            return [f(r)]
        if op == "Sign":
            return [np.sign(x).astype(x.dtype)]
        if op == "Relu":
            return [np.maximum(x, ft(0)).astype(ft)]
        if op == "LeakyRelu":
            al = ft(a.get("alpha", 0.01))
            return [np.where(x >= 0, x, x * al).astype(ft)]
        if op == "PRelu":
            return [np.where(x >= 0, x, x * f(ins[1])).astype(ft)]
        if op == "Sigmoid":
            return [(1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))).astype(ft)]
        if op == "Tanh":
            return [np.tanh(np.asarray(x, np.float64)).astype(ft)]
        if op == "HardSigmoid":
            al, be = a.get("alpha", 0.2), a.get("beta", 0.5)
            return [np.clip(np.asarray(x, np.float64) * al + be, 0, 1).astype(ft)]
        if op == "HardSwish":
            x64 = np.asarray(x, np.float64)
            return [(x64 * np.clip(x64 / 6.0 + 0.5, 0, 1)).astype(ft)]
        if op == "Gelu":
            x64 = np.asarray(x, np.float64)
            if a.get("approximate", "none") == "tanh":
                y = 0.5 * x64 * (1 + np.tanh(math.sqrt(2 / math.pi) * (x64 + 0.044715 * x64 ** 3)))
            else:
                y = x64 * 0.5 * (1.0 + _ERF(x64 / math.sqrt(2.0)))
            return [y.astype(ft)]
        if op == "Clip":
            lo = a.get("min", None) if self.opset < 11 else (ins[1] if len(ins) > 1 else None)
            hi = a.get("max", None) if self.opset < 11 else (ins[2] if len(ins) > 2 else None)
            y = x
            if lo is not None:
                y = np.maximum(y, f(lo))
            if hi is not None:
                y = np.minimum(y, f(hi))
            return [np.asarray(y).astype(ft)]
        if op == "Softmax":
            ax = int(a.get("axis", 1 if self.opset < 13 else -1))
            x64 = np.asarray(x, np.float64)
            if self.opset < 13:
                shp = x64.shape
                ax %= x64.ndim
                x64 = x64.reshape(int(np.prod(shp[:ax])), -1)
                e = np.exp(x64 - x64.max(axis=1, keepdims=True))
                return [(e / e.sum(axis=1, keepdims=True)).reshape(shp).astype(ft)]
            e = np.exp(x64 - x64.max(axis=ax, keepdims=True))
            return [(e / e.sum(axis=ax, keepdims=True)).astype(ft)]
        if op == "Shape":
            s = np.array(np.shape(x), np.int64)
            st, en = a.get("start", 0), a.get("end", None)
            return [s[st:en]]
        if op == "Size":
            return [np.array(np.size(x), np.int64)]
        if op == "Reshape":
            shp = [int(s) for s in np.asarray(ins[1]).reshape(-1)]
            if not a.get("allowzero", 0):
                shp = [np.shape(x)[i] if s == 0 else s for i, s in enumerate(shp)]
            return [np.reshape(x, shp)]
        if op == "Flatten":
            ax = int(a.get("axis", 1)) % max(np.ndim(x), 1)
            s = np.shape(x)
            return [np.reshape(x, (int(np.prod(s[:ax])), -1))]
        if op == "Unsqueeze":
            y = x
            for ax in sorted(v % (np.ndim(x) + len(_axes(a, ins))) for v in _axes(a, ins)):
                y = np.expand_dims(y, ax)
            return [y]
        if op == "Squeeze":
            ax = _axes(a, ins)
            return [np.squeeze(x, axis=tuple(ax)) if ax is not None else np.squeeze(x)]
        if op == "Concat":
            return [np.concatenate([np.atleast_1d(v) for v in ins], axis=int(a["axis"]))]
        if op == "Gather":
            return [np.take(x, np.asarray(ins[1]).astype(np.int64), axis=int(a.get("axis", 0)))]
        if op == "Transpose":
            return [np.transpose(x, a.get("perm"))]
        if op == "Slice":
            starts, ends = np.asarray(ins[1]).reshape(-1), np.asarray(ins[2]).reshape(-1)
            axes = np.asarray(ins[3]).reshape(-1) if len(ins) > 3 and ins[3] is not None else np.arange(len(starts))
            steps = np.asarray(ins[4]).reshape(-1) if len(ins) > 4 and ins[4] is not None else np.ones(len(starts), np.int64)
            sl = [slice(None)] * np.ndim(x)
            for s, e, ax, st in zip(starts, ends, axes, steps):
                sl[int(ax)] = slice(int(s), int(e), int(st))
            return [np.asarray(x)[tuple(sl)]]
        if op == "MatMul":
            return [(np.asarray(ins[0], np.float64) @ np.asarray(ins[1], np.float64)).astype(ft)]
        if op == "Gemm":
            A = np.asarray(ins[0], np.float64)
            B = np.asarray(ins[1], np.float64)
            if a.get("transA", 0):
                A = A.T
            if a.get("transB", 0):
                B = B.T
            y = float(a.get("alpha", 1.0)) * (A @ B)
            if len(ins) > 2 and ins[2] is not None:
                y = y + float(a.get("beta", 1.0)) * np.asarray(ins[2], np.float64)
            return [y.astype(ft)]
        if op == "Conv":
            return [conv_nchw(x, ins[1], ins[2] if len(ins) > 2 else None, a, ft)]
        if op == "BatchNormalization":
            xx, gam, bet, mu, var = (np.asarray(v, ft) for v in ins[:5])
            if a.get("training_mode", 0):
                raise NotImplementedError(f"BatchNormalization {nd.name}: training_mode")
            shp = (1, -1) + (1,) * (xx.ndim - 2)
            eps = ft(a.get("epsilon", 1e-5))
            y = (xx - mu.reshape(shp)) / np.sqrt(var.reshape(shp) + eps) * gam.reshape(shp) + bet.reshape(shp)
            return [y.astype(ft)]
        if op == "LayerNormalization":
            ax = int(a.get("axis", -1))
            eps = float(a.get("epsilon", 1e-5))
            x64 = np.asarray(x, np.float64)
            axes = tuple(range(ax % x64.ndim, x64.ndim))
            m = x64.mean(axis=axes, keepdims=True)
            v = ((x64 - m) ** 2).mean(axis=axes, keepdims=True)
            y = (x64 - m) / np.sqrt(v + eps) * np.asarray(ins[1], np.float64)
            if len(ins) > 2 and ins[2] is not None:
                y = y + np.asarray(ins[2], np.float64)
            return [y.astype(ft)]
        if op == "MaxPool":
            if len(nd.output) > 1 and nd.output[1]:
                raise NotImplementedError(f"MaxPool {nd.name}: Indices output")
            return [pool_nchw(np.asarray(x, ft), a, "max", ft)]
        if op == "AveragePool":
            return [pool_nchw(np.asarray(x, ft), a, "avg", ft)]
        if op == "GlobalAveragePool":
            return [np.asarray(x, np.float64).mean(axis=tuple(range(2, np.ndim(x))), keepdims=True).astype(ft)]
        if op == "GlobalMaxPool":
            return [np.asarray(x).max(axis=tuple(range(2, np.ndim(x))), keepdims=True).astype(ft)]
        if op in ("ReduceSum", "ReduceMax", "ReduceMin") and np.asarray(x).dtype.kind in "iu":
            ax = _axes(a, ins)
            kd = bool(a.get("keepdims", 1))
            fn = {"ReduceSum": np.sum, "ReduceMax": np.max, "ReduceMin": np.min}[op]
            return [np.asarray(fn(np.asarray(x), axis=tuple(ax) if ax else None, keepdims=kd)).astype(np.asarray(x).dtype)]
        if op in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"):
            ax = _axes(a, ins)
            kd = bool(a.get("keepdims", 1))
            fn = {"ReduceMean": np.mean, "ReduceSum": np.sum, "ReduceMax": np.max, "ReduceMin": np.min}[op]
            return [np.asarray(fn(np.asarray(x, np.float64), axis=tuple(ax) if ax else None, keepdims=kd)).astype(ft)]
        if op == "Pad":
            mode = a.get("mode", "constant")
            if mode != "constant":
                raise NotImplementedError(f"Pad {nd.name}: mode {mode}")
            p = a.get("pads") if "pads" in a else [int(v) for v in np.asarray(ins[1]).reshape(-1)]
            val = a.get("value", 0.0) if "pads" in a else (float(np.asarray(ins[2]).reshape(-1)[0]) if len(ins) > 2 and ins[2] is not None else 0.0)
            nd_ = np.ndim(x)
            if len(ins) > 3 and ins[3] is not None:
                raise NotImplementedError(f"Pad {nd.name}: axes input")
            return [np.pad(x, [(int(p[i]), int(p[i + nd_])) for i in range(nd_)], constant_values=val).astype(np.asarray(x).dtype)]
        if op == "TopK":
            k = int(np.asarray(ins[1]).reshape(-1)[0])
            ax = int(a.get("axis", -1))
            if not a.get("largest", 1):
                raise NotImplementedError(f"TopK {nd.name}: largest=0")
            idx = np.argsort(-np.asarray(x), axis=ax, kind="stable")
            idx = np.take(idx, np.arange(k), axis=ax)
            return [np.take_along_axis(x, idx, axis=ax), idx.astype(np.int64)]
        if op == "BipolarQuant":
            return [binary_quant(np.asarray(x, ft), np.asarray(ins[1], ft)).astype(ft)]
        if op in ("Quant", "IntQuant"):
            return [int_quant(np.asarray(x, ft), np.asarray(ins[1], ft), np.asarray(ins[2], ft), np.asarray(ins[3], ft),
                              bool(a.get("signed", 1)), bool(a.get("narrow", 0)), a.get("rounding_mode", "ROUND")).astype(ft)]
        if op == "MultiThreshold":
            return [multithreshold(np.asarray(x, ft), np.asarray(ins[1], ft), ft(a.get("out_scale", 1.0)),
                                   ft(a.get("out_bias", 0.0)), a.get("data_layout", "NCHW")).astype(ft)]
        raise OnnxF64Unsupported(nd, f"OnnxF64: node {nd.name!r}: op {op} (domain {nd.domain!r}) is not implemented")


def input_dependent(model, inputs) -> set:
    """Tensors that depend on a graph input (a Shape node's output is static and breaks the dependency)."""
    dep = set(inputs)
    for nd in model.graph.node:
        if nd.op_type in ("Shape", "Size"):
            continue
        if any(i in dep for i in nd.input if i):
            dep.update(o for o in nd.output if o)
    return dep
