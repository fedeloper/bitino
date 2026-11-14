"""Independent numpy implementation of the Bitino operator contracts.

Used for Level-3 validation (source framework vs converted IR) so that
conversion bugs can be separated from C++ runtime bugs. Float Conv2D / Dense /
pooling / Softmax / LayerNorm / ReduceMean accumulate in float64 in exactly the
order of the C++ reference kernels (bit-identical float32 results); binary ops
accumulate in exact integers. MatMul still uses float64 BLAS.
"""
from __future__ import annotations

import math

import numpy as np

from . import image_ops, int8_ops
from .ir import Graph, unpack_bits

# libm (glibc) erf / tanh in double, element-wise: the C++ reference kernels
# evaluate GELU and tanh in double with the same libm functions and round once.
_ERF = np.frompyfunc(math.erf, 1, 1)
_TANH = np.frompyfunc(math.tanh, 1, 1)
_EXP = np.frompyfunc(math.exp, 1, 1)


def _libm(f, x: np.ndarray) -> np.ndarray:
    return np.asarray(f(np.asarray(x, np.float64)), dtype=np.float64).reshape(np.shape(x))


def _bmode(x: np.ndarray, mode: str) -> np.ndarray:
    """BMatMul operand binarization (exact small integers, as float64 for BLAS-exact matmul)."""
    if mode == "ge":
        return np.where(x >= 0, 1.0, -1.0)
    if mode == "gt":
        return np.where(x > 0, 1.0, -1.0)
    if mode == "sign_zero":
        return np.sign(x).astype(np.float64)
    if mode == "gt0":
        return np.where(x > 0, 1.0, 0.0)
    raise ValueError(mode)


def _layernorm(x: np.ndarray, gamma, beta, eps: float, ft) -> np.ndarray:
    """Mirrors runtime/kernels/reference/transformer_reference.cc exactly (sequential double sums)."""
    d = x.shape[-1]
    x64 = np.asarray(x, np.float64).reshape(-1, d)
    s = np.zeros(x64.shape[0])
    for i in range(d):
        s += x64[:, i]
    mean = s / float(d)
    v = np.zeros(x64.shape[0])
    for i in range(d):
        t = x64[:, i] - mean
        v += t * t
    sd = np.sqrt(v / float(d) + float(eps))
    y = (x64 - mean[:, None]) / sd[:, None]
    if gamma is not None:
        y = y * np.asarray(gamma, np.float64)
    if beta is not None:
        y = y + np.asarray(beta, np.float64)
    return y.astype(ft).reshape(x.shape)


def _gather(table: np.ndarray, idx: np.ndarray, scale, zmask, ft) -> np.ndarray:
    v = table.shape[0]
    ids = np.asarray(idx, np.int64)
    ok = (ids >= 0) & (ids < v)
    safe = np.where(ok, ids, 0)
    rows = table[safe].astype(ft)
    if scale is not None:
        rows = rows * np.asarray(scale, ft)[safe][..., None]
    if zmask is not None:
        rows = np.where(zmask[safe] < 0, ft(0), rows)
    return np.where(ok[..., None], rows, ft(0)).astype(ft)


def _act(x: np.ndarray, attrs: dict) -> np.ndarray:
    a = attrs.get("activation", "none")
    if a == "none":
        return x
    if a == "relu":
        return np.maximum(x, 0)
    if a == "relu6":
        return np.clip(x, 0, 6)
    raise ValueError(a)


def _binarize(x: np.ndarray, mode: str, thr) -> np.ndarray:
    t = 0.0 if thr is None else thr
    if mode == "ge":
        return np.where(x >= t, 1, -1).astype(np.int32)
    if mode == "gt":
        return np.where(x > t, 1, -1).astype(np.int32)
    if mode == "sign_zero":
        return np.sign(x - t).astype(np.int32)
    raise ValueError(mode)


def _im2col(x: np.ndarray, kh, kw, sh, sw, dh, dw, pads, pad_value):
    n, h, w, c = x.shape
    pt, pb, pl, pr = pads
    xp = np.full((n, h + pt + pb, w + pl + pr, c), pad_value, dtype=x.dtype)
    xp[:, pt:pt + h, pl:pl + w, :] = x
    oh = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    ow = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    cols = np.empty((n, oh, ow, kh, kw, c), dtype=x.dtype)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, :, ky, kx, :] = xp[:, ky * dh: ky * dh + sh * (oh - 1) + 1: sh, kx * dw: kx * dw + sw * (ow - 1) + 1: sw, :]
    return cols, oh, ow


def _conv(x, w, attrs, pad_value, acc_dtype):
    sh, sw = attrs.get("strides", [1, 1])
    dh, dw = attrs.get("dilations", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    groups = attrs.get("groups", 1)
    o, kh, kw, kic = w.shape
    cols, oh, ow = _im2col(x, kh, kw, sh, sw, dh, dw, pads, pad_value)
    n = x.shape[0]
    ocg = o // groups
    out = np.empty((n, oh, ow, o), dtype=acc_dtype)
    for g in range(groups):
        cg = cols[..., g * kic:(g + 1) * kic].reshape(n * oh * ow, kh * kw * kic).astype(acc_dtype)
        wg = w[g * ocg:(g + 1) * ocg].reshape(ocg, kh * kw * kic).astype(acc_dtype)
        out[..., g * ocg:(g + 1) * ocg] = (cg @ wg.T).reshape(n, oh, ow, ocg)
    return out


def _conv_float_seq(x, w, bias, attrs, pad_value):
    """Float Conv2D exactly like Conv2DRef: per output, double accumulator initialised with the bias,
    then += x*w in (ky, kx, c) order; rounded once by the caller."""
    sh, sw = attrs.get("strides", [1, 1])
    dh, dw = attrs.get("dilations", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    groups = attrs.get("groups", 1)
    o, kh, kw, kic = w.shape
    cols, oh, ow = _im2col(x, kh, kw, sh, sw, dh, dw, pads, pad_value)
    n = x.shape[0]
    ocg = o // groups
    out = np.empty((n * oh * ow, o), dtype=np.float64)
    for g in range(groups):
        cg = cols[..., g * kic:(g + 1) * kic].reshape(n * oh * ow, kh * kw * kic).astype(np.float64)
        wg = w[g * ocg:(g + 1) * ocg].reshape(ocg, kh * kw * kic).astype(np.float64)
        acc = np.zeros((n * oh * ow, ocg)) + (0.0 if bias is None else np.asarray(bias, np.float64)[g * ocg:(g + 1) * ocg])
        for t in range(kh * kw * kic):
            acc += cg[:, t:t + 1] * wg[:, t]
        out[:, g * ocg:(g + 1) * ocg] = acc
    return out.reshape(n, oh, ow, o)


def _dense_float_seq(x, w, bias):
    """Float Dense exactly like DenseRef: double accumulator initialised with the bias, += x*w in k order."""
    x64 = np.asarray(x, np.float64)
    w64 = np.asarray(w, np.float64)
    acc = np.zeros(x64.shape[:-1] + (w64.shape[0],)) + (0.0 if bias is None else np.asarray(bias, np.float64))
    for j in range(w64.shape[1]):
        acc += x64[..., j:j + 1] * w64[:, j]
    return acc


def _pool(x, attrs, kind, ft=np.float32):
    kh, kw = attrs["kernel"]
    sh, sw = attrs.get("strides", [kh, kw])
    pt, pb, pl, pr = attrs.get("pads", [0, 0, 0, 0])
    n, h, w, c = x.shape
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    out = np.empty((n, oh, ow, c), ft)
    include_pad = bool(attrs.get("count_include_pad", 0))
    for oy in range(oh):
        for ox in range(ow):
            y0, x0 = oy * sh - pt, ox * sw - pl
            ys, ye, xs, xe = max(y0, 0), min(y0 + kh, h), max(x0, 0), min(x0 + kw, w)
            win = x[:, ys:ye, xs:xe, :]
            if kind == "max":
                out[:, oy, ox, :] = win.max(axis=(1, 2))
            else:
                cnt = kh * kw if include_pad else (ye - ys) * (xe - xs)
                s = np.zeros((n, c))
                for iy in range(ye - ys):  # sequential double sum in (ky, kx) order, like the C++ reference
                    for ix in range(xe - xs):
                        s += win[:, iy, ix, :]
                out[:, oy, ox, :] = (s / max(cnt, 1)).astype(ft)
    return out


def _seq_mean(x: np.ndarray, axis: int) -> np.ndarray:
    """Mean over one axis: sequential double sum in index order, one division, like the C++ reference."""
    x64 = np.moveaxis(np.asarray(x, np.float64), axis, 0)
    s = np.zeros(x64.shape[1:])
    for i in range(x64.shape[0]):
        s += x64[i]
    return s / float(x64.shape[0])


def _softmax(x: np.ndarray, ft) -> np.ndarray:
    """Mirrors SoftmaxRef: m = max (input precision), d = x - m (input precision), e = libm exp(double d),
    sequential double sum, y = round(e / sum)."""
    m = x.max(axis=-1, keepdims=True)
    e = _libm(_EXP, (x - m).astype(ft))
    s = np.zeros(e.shape[:-1])
    for i in range(e.shape[-1]):
        s += e[..., i]
    return (e / s[..., None]).astype(ft)


def _fake_quant(x: np.ndarray, scale: np.ndarray, zp: np.ndarray, attrs: dict, ft) -> np.ndarray:
    """FakeQuant: clamp(rint(x*pre/scale[c] + zp[c]), qmin, qmax) * scale[c], c along `axis`; every
    step rounded in the working precision (round half to even)."""
    axis = attrs.get("axis", -1) % x.ndim
    shp = [1] * x.ndim
    shp[axis] = -1
    s = np.asarray(scale, ft).reshape(shp)
    z = np.asarray(zp, ft).reshape(shp)
    t = (x * ft(attrs.get("pre_scale", 1.0))).astype(ft)
    t = (t / s).astype(ft)
    t = (t + z).astype(ft)
    r = np.clip(np.rint(t), ft(attrs.get("qmin", 0)), ft(attrs.get("qmax", 1))).astype(ft)
    return (r * s).astype(ft)


def shift_groups(groups) -> list:
    """ChannelShift groups as [(c0, c1, dy, dx), ...] from [[c0, c1, dy, dx], ...] or the flat list."""
    g = list(groups)
    if g and not isinstance(g[0], (list, tuple)):
        g = [g[k:k + 4] for k in range(0, len(g), 4)]
    return [tuple(int(v) for v in t) for t in g]


def channel_shift(x: np.ndarray, groups, fill) -> np.ndarray:
    """ChannelShift (ShiftViT; docs/int8.md "int8_full ops for accuracy-carrying glue"): y[n, h, w, c] = x[n, h - dy,
    w - dx, c] for c in [c0, c1) of a group when inside the map, else fill (0); other channels copied."""
    y = np.array(x, copy=True)
    _, h, w, _ = x.shape
    for c0, c1, dy, dx in shift_groups(groups):
        part = np.full(x[..., c0:c1].shape, fill, x.dtype)
        hs, he, ws, we = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
        if hs < he and ws < we:
            part[:, hs:he, ws:we, :] = x[:, hs - dy:he - dy, ws - dx:we - dx, c0:c1]
        y[..., c0:c1] = part
    return y


def instance_threshold(x, t0, a, b, off, ft) -> np.ndarray:
    """InstanceThreshold (INSTA-BNN, arXiv 2204.07439): y = x - (t0 + a * mean_hw(x) + b * std_hw(x) [+ off]) per (n, c),
    population std; read only by binary layers. Exact definition (2026-11-14; the C++ float kernel
    runtime/kernels/reference/glue_float_reference.cc is bit-identical): in double, s1 / s2 the sums of x and x * x over
    the H x W positions in row-major order, mu = s1 / HW, var = s2 / HW - mu * mu, sd = sqrt(max(var, 0)),
    th = t0 + a * mu + b * sd (+ off), y = float(double(x) - th)."""
    x64 = np.asarray(x, np.float64)
    n, h, w, c = x64.shape
    xs = x64.reshape(n, h * w, c)
    s1 = np.zeros((n, c))
    s2 = np.zeros((n, c))
    for p in range(h * w):   # sequential sums, as the C++ kernel
        v = xs[:, p, :]
        s1 = s1 + v
        s2 = s2 + v * v
    mu = s1 / (h * w)
    var = s2 / (h * w) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    th = np.asarray(t0, np.float64) + np.asarray(a, np.float64) * mu + np.asarray(b, np.float64) * sd
    if off is not None:
        th = th + np.asarray(off, np.float64).reshape(n, c)
    return (x64 - th.reshape(n, 1, 1, c)).astype(ft)


def simam(x, lam: float, ft) -> np.ndarray:
    """SimAM (parameter-free attention): y = x * sigmoid((x - mu)^2 / (4 (var + lambda)) + 0.5), mu / var per (n, c) over
    H, W (population variance), in float64, rounded once."""
    x64 = np.asarray(x, np.float64)
    mu = x64.mean(axis=(1, 2), keepdims=True)
    var = ((x64 - mu) ** 2).mean(axis=(1, 2), keepdims=True)
    e = (x64 - mu) ** 2 / (4.0 * (var + lam)) + 0.5
    return (x64 / (1.0 + np.exp(-e))).astype(ft)


def run(graph: Graph, feeds: dict[str, np.ndarray], keep_intermediates: bool = False,
        float_dtype=np.float32, post_hook=None) -> dict[str, np.ndarray]:
    """float_dtype=np.float64 keeps all intermediates in float64 (semantic check against a
    float64 evaluation of the source model); np.float32 mirrors the C++ runtime."""
    ft = float_dtype
    env: dict[str, np.ndarray] = {}
    # Unpacked / converted constants are cached on the graph object (per float dtype) and
    # revalidated by identity of the source array, so callers that replace Constant.data
    # see the new values. Large transformer graphs are run once per sample.
    cache = graph.__dict__.setdefault("_executor_const_cache", {})
    for name, c in graph.constants.items():
        key = (name, np.dtype(ft).str if c.dtype == "f32" else c.dtype)
        hit = cache.get(key)
        if hit is None or hit[0] is not c.data:
            hit = (c.data, unpack_bits(c.data, c.shape[-1]).reshape(c.shape) if c.dtype == "bit" else (c.data.astype(ft) if c.dtype == "f32" else c.data))
            cache[key] = hit
        env[name] = hit[1]
    for inp in graph.inputs:
        idt = ft if inp.get("dtype", "f32") == "f32" else {"i32": np.int32, "i8": np.int8, "u8": np.uint8}[inp["dtype"]]
        env[inp["name"]] = np.asarray(feeds[inp["name"]], idt).reshape(inp["shape"])

    def get(name):
        return env[name] if name else None

    for node in graph.nodes:
        a = node.attrs
        ins = [get(i) for i in node.inputs]
        op = node.op
        if op.startswith("I8"):  # full-integer operators (stage int8_full): python/bitino/int8_ops.py
            r = int8_ops.run_op(op, ins, a, cache, node.name)
            if isinstance(r, tuple):  # multi-output operators (I8BConv2DRes with its bit output)
                for o, v in zip(node.outputs, r):
                    env[o] = v
            else:
                env[node.outputs[0]] = r
            continue
        if op == "Reshape" and ins[0].dtype == np.int8:
            env[node.outputs[0]] = ins[0].reshape(a["shape"])
            continue
        if op == "Conv2D":
            y = _conv_float_seq(ins[0], ins[1], ins[2] if len(ins) > 2 else None, a, a.get("pad_value", 0.0))
            y = _act(y.astype(ft), a)
        elif op == "BConv2D":
            x = ins[0]
            thr = ins[4] if len(ins) > 4 else None
            xb = _binarize(x, a.get("input_binarize", "ge"), thr)
            # {-1,0,+1} operands: float64 BLAS sums small integers exactly (|acc| <= K < 2^53), identical to an
            # int64 accumulation and much faster for the large spatial maps of the restoration models
            acc = _conv(xb, ins[1].astype(np.int32), a, int(a.get("pad_value", 0)), np.float64)
            y = acc.astype(ft)
            if len(ins) > 2 and ins[2] is not None:
                y = y * ins[2]
            if len(ins) > 3 and ins[3] is not None:
                y = y + ins[3]
            y = _act(y.astype(ft), a)
            if a.get("output", "f32") == "bit":
                y = np.where(y >= 0, 1.0, -1.0).astype(ft)  # unpacked view of the bit tensor
        elif op in ("QBConv2D", "QBDense"):
            # k-bit (DoReFa) activations x binary weights, docs/operators_larq.md:
            # level = round_half_even(clip(x, 0, 1) * n) in the float dtype, integer accumulation.
            n_lv = ft(2 ** int(a["act_bits"]) - 1)
            lv = np.round(np.clip(ins[0], ft(0), ft(1)).astype(ft) * n_lv).astype(np.int32)
            w = ins[1].astype(np.int32)
            if op == "QBConv2D":
                acc = _conv(lv, w, a, 0, np.int64)
            else:
                acc = lv.astype(np.int64) @ w.astype(np.int64).T
            y = acc.astype(ft)
            if len(ins) > 2 and ins[2] is not None:
                y = y * ins[2]
            if len(ins) > 3 and ins[3] is not None:
                y = y + ins[3]
            y = _act(y.astype(ft), a)
        elif op == "ChannelGate":
            x, gate = ins[0], ins[1]
            gv = gate.reshape(x.shape[0], 1, 1, x.shape[3])
            if a.get("gate", "sigmoid") == "sigmoid":
                # Evaluated in float64 and rounded once (mirrors the C++ reference kernel).
                gv = 1.0 / (1.0 + np.exp(-gv.astype(np.float64)))
            y = (x * gv.astype(ft)).astype(ft)
        elif op in ("Dense", "BDense"):
            x = ins[0]
            if op == "Dense":
                y = _dense_float_seq(x, ins[1], ins[2] if len(ins) > 2 else None).astype(ft)
            else:
                thr = ins[4] if len(ins) > 4 else None
                xb = _binarize(x, a.get("input_binarize", "ge"), thr)
                wkey = ("bdense_wT", node.name)
                wz = ins[5] if len(ins) > 5 else None
                hit = cache.get(wkey)
                if hit is None or hit[0] is not ins[1] or hit[1] is not wz:
                    w = ins[1].astype(np.float32)
                    if wz is not None:  # w_zero: "-1" bits mark exact zero weights
                        w = np.where(wz < 0, np.float32(0), w)
                    hit = (ins[1], wz, np.ascontiguousarray(w.T))
                    cache[wkey] = hit
                w = hit[2].T
                # float32 BLAS on {-1,0,+1} operands is exact: every partial sum is an integer < 2^24.
                y = (xb.astype(np.float32) @ w.T).astype(ft)
                if len(ins) > 2 and ins[2] is not None:
                    y = y * ins[2]
                if len(ins) > 3 and ins[3] is not None:
                    y = y + ins[3]
            y = _act(y, a)
        elif op == "ChannelAffine":
            y = ins[0]
            if len(ins) > 1 and ins[1] is not None:
                y = y * ins[1]
            if len(ins) > 2 and ins[2] is not None:
                y = y + ins[2]
            y = y.astype(ft)
        elif op == "Activation":
            t, x = a["type"], ins[0]
            if t == "relu":
                y = np.maximum(x, 0)
            elif t == "relu6":
                y = np.clip(x, 0, 6)
            elif t == "hardtanh":
                y = np.clip(x, a.get("min", -1.0), a.get("max", 1.0))
            elif t == "prelu":
                y = np.where(x >= 0, x, x * ins[1])
            elif t == "leaky_relu":
                y = np.where(x >= 0, x, x * a.get("alpha", 0.01))
            elif t == "sigmoid":
                y = 1 / (1 + np.exp(-x))
            elif t == "tanh":  # double (libm), rounded once, like the C++ reference
                y = _libm(_TANH, x)
            elif t == "gelu":  # torch/BERT formula in double: x*0.5*(1+erf(x/sqrt(2)))
                x64 = np.asarray(x, np.float64)
                y = x64 * 0.5 * (1.0 + _libm(_ERF, x64 / math.sqrt(2.0)))
            elif t == "hardswish":
                y = x * np.clip(x + 3, 0, 6) / 6
            elif t == "sign":
                y = np.where(x >= 0, 1.0, -1.0)
            elif t == "sign_zero":  # torch.sign: +1 / 0 / -1 (never -0)
                y = np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0))
            else:
                raise ValueError(t)
            y = y.astype(ft)
        elif op == "Add":
            y = _act((ins[0] + ins[1]).astype(ft), a)
        elif op == "Concat":
            y = np.concatenate(ins, axis=a.get("axis", -1))
        elif op == "MaxPool2D":
            y = _pool(ins[0], a, "max", ft)
        elif op == "AvgPool2D":
            y = _pool(ins[0], a, "avg", ft)
        elif op == "GlobalAvgPool":
            x0 = ins[0]
            y = _seq_mean(x0.reshape(x0.shape[0], -1, x0.shape[3]), 1).astype(ft)
            if a.get("keepdims", 0):
                y = y.reshape(x0.shape[0], 1, 1, x0.shape[3])
        elif op == "Reshape":
            y = ins[0].reshape(a["shape"])
        elif op == "Softmax":
            y = _softmax(ins[0], ft)
        elif op == "Gather":
            y = _gather(ins[0], ins[1], ins[2] if len(ins) > 2 else None, ins[3] if len(ins) > 3 else None, ft)
        elif op == "LayerNorm":
            y = _layernorm(ins[0], ins[1] if len(ins) > 1 else None, ins[2] if len(ins) > 2 else None, a.get("eps", 1e-5), ft)
        elif op in ("MatMul", "BMatMul"):
            tb = bool(a.get("transpose_b", 0))
            if op == "MatMul":
                p = np.matmul(ins[0].astype(np.float64), np.swapaxes(ins[1], -1, -2).astype(np.float64) if tb else ins[1].astype(np.float64))
                y = (p * float(a.get("scale", 1.0))).astype(ft)
            else:
                xa = _bmode(ins[0], a.get("a_binarize", "sign_zero"))
                xb = _bmode(ins[1], a.get("b_binarize", "sign_zero"))
                acc = np.matmul(xa, np.swapaxes(xb, -1, -2) if tb else xb)  # exact integers
                y = acc.astype(ft) * ft(a.get("scale", 1.0))
        elif op == "AddBroadcast":
            y = (ins[0] + ins[1]).astype(ft)
        elif op == "Transpose":
            y = np.ascontiguousarray(np.transpose(ins[0], a["perm"]))
        elif op == "ReduceMean":
            x0 = ins[0]
            ax = a.get("axis", -1) % x0.ndim
            y = _seq_mean(x0, ax).astype(ft)
            if a.get("keepdims", 0):
                y = np.expand_dims(y, ax)
        elif op == "FakeQuant":
            y = _fake_quant(ins[0], ins[1], ins[2], a, ft)
        elif op == "Mul":  # docs/operators_torch_cnn.md: same shapes, one fp32 product
            y = (ins[0] * ins[1]).astype(ft)
        elif op == "ImageGrayU8":  # docs/operators_image_features.md (python/bitino/image_ops.py)
            y = image_ops.image_gray_u8(np.asarray(ins[0], np.float32), a).astype(ft)
        elif op == "FftLogMagnitude":
            for d in ins[0].shape[1:3]:
                assert max(image_ops.fft_factors(int(d)) or [1]) <= 7, f"FftLogMagnitude: length {d} has a prime factor > 7"
            y = image_ops.fft_log_magnitude(np.asarray(ins[0], np.float32), a).astype(ft)
        elif op == "LocalBinaryPattern":
            y = image_ops.local_binary_pattern(np.asarray(ins[0], np.float32), a).astype(ft)
        elif op == "Pad":
            p = a["pads"]
            y = np.pad(ins[0], [(p[2 * d], p[2 * d + 1]) for d in range(ins[0].ndim)], constant_values=a.get("value", 0.0))
        elif op == "ChannelShift":
            y = channel_shift(ins[0], a.get("groups", []), 0.0)
        elif op == "InstanceThreshold":
            y = instance_threshold(ins[0], ins[1], ins[2], ins[3], ins[4] if len(ins) > 4 else None, ft)
        elif op == "SimAM":
            y = simam(ins[0], float(a.get("lambda", 1e-4)), ft)
        else:
            raise NotImplementedError(f"numpy executor: op {op}")
        if post_hook is not None:  # diagnostics (e.g. simulated quantization of selected tensors)
            y = post_hook(node.outputs[0], y)
        env[node.outputs[0]] = y
    if keep_intermediates:
        return env
    return {o: env[o] for o in graph.outputs}
