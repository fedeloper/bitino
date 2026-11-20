"""QONNX -> Bitino graph IR importer (docs/qonnx-frontend.md, phase 1).

    .cache/venv-qonnx/bin/python -I tools/import_qonnx.py model.onnx --model-id ID --out-dir DIR \
        --eval-dir .cache/data/eval128_x --preprocess KEY [--input-domain uint8|unit8|float32]
    bitino import-qonnx model.onnx ...            (same arguments; CLI subcommand)

The importer walks the (optionally qonnx-cleaned) ONNX graph once, in topological order, and maps every node that
depends on the graph input onto IR operators by pattern rules. Input-independent subgraphs (weights, BipolarQuant /
IntQuant of weights, Transpose, Shape arithmetic, ``Pow(var + eps, 0.5)``) are folded in float64 by
``bitino.onnx_f64.OnnxF64``; folding never crosses a binarizer of a runtime tensor (rule 6: only tensors that do
not depend on the graph input are constants). An unmatched node stops the import with ``QonnxImportError``
(node name, op type, reason).

Semantics (docs/qonnx-frontend.md § Semantics the profile fixes):

* Binarizers: ``BipolarQuant(x, s)`` (``x >= 0 -> +s``: mode ``ge``), ONNX ``Sign`` (``sign(0) = 0``: mode
  ``sign_zero``), ``Where(GreaterOrEqual(x, t), 1, -1)`` / ``Where(Greater(x, t), 1, -1)`` (modes ``ge`` / ``gt``
  with threshold ``t``), and ``MultiThreshold`` with one threshold per channel, ``out_scale 2``, ``out_bias -1``
  (FINN BIPOLAR: ``x >= t -> +1``, mode ``ge``). The mode of the source is kept; it is never replaced by another.
* Binary weights: a Conv / MatMul / Gemm whose activation input is binarized must have weights that are exactly
  ``±α[o]`` per output channel (folded ``BipolarQuant(W)`` or a constant); anything else is rejected, never rounded.
  ``α[o]`` and the activation scale fold into the epilogue.
* Padding: ``Conv``'s own zero padding of a binarized tensor is the binary ``pad_value 0``; an explicit ``Pad`` of
  the binarized tensor with ``+1`` / ``-1`` is ``pad_value ±1``. Mixing both is rejected.
* Layout: NCHW graph tensors become NHWC IR tensors (OIHW filters -> OHWI); a flatten of a ``[N, C, H, W]`` tensor
  with H*W > 1 and C > 1 gets an explicit Transpose to NCHW order first.
* Per-channel affine (``BatchNormalization`` in inference mode, ``Mul`` / ``Add`` / ``Sub`` / ``Div`` by a constant
  that varies at most along the channel axis) after a linear operator is accumulated in float64 and folded:
  before a ``ge`` / ``gt`` binarizer of a binary (or W1A8) accumulator it becomes an exact integer threshold on the
  accumulator (``brevitas_bnn.integer_threshold``, the same rule as the legacy converters: post_mul = ±1,
  post_add = ∓T, bit output when the only consumers are BConv2D); elsewhere it becomes the binary operator's
  post_mul / post_add (one float32 rounding of the float64 values) or a ChannelAffine folded into a float Conv2D
  by the existing pass ``passes.fold_affine_into_conv``.
* A scalar / per-channel affine of a float tensor followed by a binarizer becomes the consumer's input threshold
  ``t = fl32(-b / s)``; it is verified on the declared input domain (exhaustively for a discrete domain; for
  float32 inputs by monotonicity at the neighbouring float32 values), else the exact float32 boundary (float32
  domain) or the midpoint between neighbouring domain values (discrete domain) is used.
* ``IntQuant`` / ``Quant`` (``rounding_mode ROUND`` = half-even only, zero point 0) of a runtime tensor is a
  ``FakeQuant``, preceded by a ``ChannelAffine`` for a pending affine. The pair is bit-exact for every float32
  input when the affine is one multiplication then one addition; any other chain is accepted only after an
  exhaustive check on a discrete input domain. A binary-weight Conv on a FakeQuant tensor with a power-of-two
  scale (W1A8) folds the next BatchNorm + binarizer into an integer threshold of the exact ``acc / scale``:
  float Conv2D with weights ``s·w`` and bias ``-s·T·scale``.
* A ``MultiThreshold`` with several thresholds on the input path (FINN's input quantizer) followed by a scalar
  ``Add`` / ``Mul`` becomes ``ChannelAffine + FakeQuant`` when that equals the source on every value of the
  declared discrete input domain (checked exhaustively, float32 and float64).
* A trailing ``TopK`` whose index output is the graph output is dropped; the IR outputs its input (the logits).

Naming (graph only; the data section does not depend on it): ``--naming index`` (default) names layers
``conv{i}`` / ``dense{i}`` by their index among the linear operators, the input path ``input_affine`` /
``input_quant`` / ``input/threshold``, pools ``{layer}/pool`` and flattens ``{next layer}/flatten`` /
``flatten_input`` (the convention of ``adapters.finn_bnn``); ``--naming onnx`` uses the ONNX node names.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

from . import executor, format as bfformat, passes
from .adapters.brevitas_bnn import integer_threshold
from .ir import Graph, Node
from .onnx_f64 import OnnxF64, OnnxF64Unsupported, attrs as _attrs, input_dependent

IMPORTER_VERSION = "qonnx_import 1 (phase 1)"
_BINARY_LIN = ("BConv2D", "BDense")


class QonnxImportError(NotImplementedError):
    def __init__(self, node, reason: str):
        self.node_name = getattr(node, "name", str(node))
        self.op = getattr(node, "op_type", "?")
        super().__init__(f"node {self.node_name!r} ({self.op}): {reason}")


# ------------------------------------------------------------------------------------------------ input domains
@dataclasses.dataclass
class Domain:
    """Values the graph input can take (per element). ``values`` None = every float32 value."""
    name: str
    values: np.ndarray | None = None

    @staticmethod
    def parse(spec: str) -> "Domain":
        k = np.arange(256, dtype=np.float32)
        if spec in ("float32", "float", "any"):
            return Domain("float32")
        if spec == "uint8":   # float32(k), k = 0..255
            return Domain("uint8", k)
        if spec == "unit8":   # float32(k) / 255 (torchvision ToTensor)
            return Domain("unit8", (k / np.float32(255)).astype(np.float32))
        raise ValueError(f"unknown input domain {spec!r} (float32 | uint8 | unit8)")


def _prev32(t: np.float32) -> np.float32:
    return np.nextafter(np.float32(t), np.float32(-np.inf), dtype=np.float32)


# ------------------------------------------------------------------------------------------------ values
@dataclasses.dataclass
class V:
    """State of one ONNX tensor on the activation path."""
    kind: str                       # float | affine | lin | bin | quant | mtcount | cmp
    t: str = ""                     # IR tensor (float/quant: the value; bin: carrier; lin: accumulator; affine: base)
    shape: tuple = ()               # ONNX (NCHW) shape
    layer: str = ""                 # IR layer that produced t (for names)
    from_input: bool = False        # elementwise / reshape function of the graph input only
    # affine (y = base*s + b per channel) and lin (pending epilogue)
    s: np.ndarray | None = None
    b: np.ndarray | None = None
    chain: list = dataclasses.field(default_factory=list)   # source ops [(op, per-channel vector, const_on_left)]
    # lin
    node: Node | None = None
    lk: str = ""                    # binary | w1a8 | float
    kmax: int = 0
    parity: int | None = None
    inv: int = 1
    w_sign: np.ndarray | None = None   # w1a8: ±1 weights OIHW
    onnx_node: str = ""
    # bin
    thr: np.ndarray | None = None
    thr_name: str = ""
    mode: str = "ge"
    scale: float = 1.0
    pad: tuple | None = None        # pending explicit Pad of the binary tensor: (pads ONNX, value)
    # quant
    q: float = 0.0
    qmin: int = 0
    qmax: int = 0
    # mtcount / cmp
    base: "V | None" = None
    extra: dict = dataclasses.field(default_factory=dict)


def _ir_shape(shape) -> list:
    s = [int(d) for d in shape]
    if len(s) == 4:
        return [s[0], s[2], s[3], s[1]]
    while len(s) > 2 and s[-1] == 1:
        s = s[:-1]
    if len(s) != 2:
        raise NotImplementedError(f"tensor rank {len(shape)} {shape} is not supported")
    return s


def _channels(shape) -> int:
    return int(shape[1]) if len(shape) > 1 else 1


# ------------------------------------------------------------------------------------------------ importer
class Importer:
    def __init__(self, model, domain: Domain | None = None, naming: str = "index", input_name: str = "input"):
        self.model = model
        self.domain = domain or Domain("float32")
        self.naming = naming
        self.ev = OnnxF64(model)
        if len(self.ev.inputs) != 1:
            raise NotImplementedError(f"exactly one graph input expected, got {self.ev.inputs}")
        self.in_onnx = self.ev.input_name
        shp = self.ev.input_shape()
        if any(d <= 0 for d in shp):
            raise NotImplementedError(f"graph input {self.in_onnx} needs a static shape, got {shp}")
        self.in_shape = tuple(shp)
        # constant folding + static shapes: one float64 evaluation on a zero input
        try:
            self.env0 = self.ev.run(np.zeros(shp, np.float64), np.float64, keep=True)
        except OnnxF64Unsupported as e:
            raise QonnxImportError(e.node, f"no float64 reference for this node (bitino.onnx_f64): {e}") from None
        self.dep = input_dependent(model, [self.in_onnx])
        self.consumers: dict[str, list] = {}
        for nd in model.graph.node:
            for i in nd.input:
                self.consumers.setdefault(i, []).append(nd)
        self.g = Graph()
        self.input_name = input_name
        self.vals: dict[str, V] = {}
        self.n_lin = 0
        self.used_names: set = set()
        self.precision: list = []
        self.margins: list = []
        self.binarizers: list = []        # flip tracing: source binarizer -> IR decision
        self.rules: dict[str, int] = {}
        self.checks: list = []
        self.notes: list = []
        self.dropped: list = []
        self.logits_onnx = None
        self.src_thr: dict = {}            # binarizer node -> source threshold (broadcastable against its input)

    # ---------------------------------------------------------------- helpers
    def rule(self, name: str) -> None:
        self.rules[name] = self.rules.get(name, 0) + 1

    def const(self, name: str):
        if not name or name in self.dep:
            return None
        return self.env0[name]

    def shape(self, name: str) -> tuple:
        return tuple(np.shape(self.env0[name]))

    def uniq(self, name: str) -> str:
        base, k = name, 1
        while name in self.used_names:
            k += 1
            name = f"{base}_{k}"
        self.used_names.add(name)
        return name

    def onnx_name(self, nd) -> str:
        n = nd.name or f"{nd.op_type}_{nd.output[0]}"
        return n.lstrip("/")

    def layer_name(self, kind: str, nd) -> str:
        idx = self.n_lin
        self.n_lin += 1
        return self.uniq(f"{kind}{idx}" if self.naming == "index" else self.onnx_name(nd))

    def node_name(self, default: str, nd) -> str:
        return self.uniq(default if self.naming == "index" else self.onnx_name(nd))

    def per_channel(self, c, shape, nd, what: str) -> np.ndarray:
        """Constant c (numpy-broadcast against an NCHW / NC tensor of `shape`) as a per-channel float64 vector."""
        c = np.asarray(c, np.float64)
        try:
            out = np.broadcast_shapes(c.shape, tuple(shape))
        except ValueError:
            raise QonnxImportError(nd, f"{what}: constant shape {c.shape} does not broadcast to {tuple(shape)}") from None
        if tuple(out) != tuple(shape):
            raise QonnxImportError(nd, f"{what}: constant shape {c.shape} would broadcast the tensor {tuple(shape)} to {out}")
        full = np.broadcast_to(c, shape)
        ch = _channels(shape)
        idx = [0] * len(shape)
        if len(shape) > 1:
            idx[1] = slice(None)
        vec = np.asarray(full[tuple(idx)], np.float64).reshape(-1) if len(shape) > 1 else full.reshape(-1)[:1]
        expect = np.broadcast_to(vec.reshape((1, ch) + (1,) * (len(shape) - 2)) if len(shape) > 1 else vec, shape)
        if not np.array_equal(full, expect):
            raise QonnxImportError(nd, f"{what}: constant {c.shape} varies along a non-channel axis (only per-channel "
                                       "constants are supported)")
        return vec.copy()

    # ---------------------------------------------------------------- materialization
    def mat(self, v: V, nd=None) -> str:
        """IR tensor holding the plain (float) value of v; emits ChannelAffine / epilogues as needed."""
        g = self.g
        if v.kind in ("float", "quant"):
            return v.t
        if v.kind == "affine":
            c = _channels(v.shape)
            if v.from_input and self.naming == "index" and not self._has("input_affine"):
                name = self.uniq("input_affine")
                pre = "input"
            else:
                name = self.node_name(f"{v.layer or 'x'}/affine", nd) if nd is not None else self.uniq(f"{v.layer or 'x'}/affine")
                pre = name
            sc = g.add_constant(self.uniq(f"{pre}/scale"), np.broadcast_to(np.asarray(v.s, np.float64), (c,)).astype(np.float32))
            sh = g.add_constant(self.uniq(f"{pre}/shift"), np.broadcast_to(np.asarray(v.b, np.float64), (c,)).astype(np.float32))
            self.rule("affine -> ChannelAffine")
            v.kind, v.t, v.layer = "float", g.add_node("ChannelAffine", [v.t, sc, sh], name=name), name
            v.extra["chain_materialized"] = v.chain
            return v.t
        if v.kind == "lin":
            return self._finish_lin(v)
        if v.kind == "bin":
            if v.thr is not None:
                raise QonnxImportError(nd, "a binarized tensor with an input threshold is consumed by a non-binary operator "
                                           "(materializing sign(x - t) is not supported)")
            if v.pad is not None:
                raise QonnxImportError(nd, "explicit Pad of a binarized tensor not followed by a Conv")
            name = self.node_name(f"{v.layer}/sign", nd) if nd is not None else self.uniq(f"{v.layer}/sign")
            t = g.add_node("Activation", [v.t], {"type": {"ge": "sign", "sign_zero": "sign_zero"}.get(v.mode, "")}, name=name) \
                if v.mode in ("ge", "sign_zero") else None
            if t is None:
                raise QonnxImportError(nd, f"materializing a binarized tensor with mode {v.mode}")
            if v.scale != 1.0:
                c = _channels(v.shape)
                t = g.add_node("ChannelAffine", [t, g.add_constant(self.uniq(f"{name}/scale"), np.full(c, v.scale, np.float32)), ""],
                               name=self.uniq(f"{name}/scaled"))
            self.rule("binary tensor -> Activation(sign)")
            v.kind, v.t = "float", t
            return t
        if v.kind == "mtcount":
            return self._mt_to_quant(v, nd).t
        raise QonnxImportError(nd, f"value of kind {v.kind} cannot be used here")

    def _has(self, name: str) -> bool:
        return name in self.used_names

    def _finish_lin(self, v: V) -> str:
        """Close the pending epilogue of a linear operator (float32 rounding of the float64 affine)."""
        g = self.g
        node = v.node
        ident = np.all(v.s == 1.0) and np.all(v.b == 0.0)
        if node.op in _BINARY_LIN:
            if not ident:
                node.inputs[2] = g.add_constant(self.uniq(f"{node.name}/post_mul"), v.s.astype(np.float32))
                node.inputs[3] = g.add_constant(self.uniq(f"{node.name}/post_add"), v.b.astype(np.float32))
                self.rule("affine -> binary epilogue post_mul/post_add")
            t = node.outputs[0]
        else:
            t = node.outputs[0]
            if not ident:
                t = g.add_node("ChannelAffine", [t, g.add_constant(self.uniq(f"{node.name}/affine_scale"), v.s.astype(np.float32)),
                                                  g.add_constant(self.uniq(f"{node.name}/affine_shift"), v.b.astype(np.float32))],
                               name=self.uniq(f"{node.name}/affine"))
                self.rule("affine after float linear op -> ChannelAffine (folded by passes.fold_affine_into_conv)")
        v.kind, v.t, v.layer = "float", t, node.name
        return t

    # ---------------------------------------------------------------- domain checks
    def _replay(self, chain, x: np.ndarray, c: int, ft) -> np.ndarray:
        y = np.asarray(x, ft)
        for op, vec, left in chain:
            k = ft(vec[c] if vec.size > 1 else vec[0])
            if op == "Mul":
                y = (y * k).astype(ft)
            elif op == "Div":
                y = ((k / y) if left else (y / k)).astype(ft)
            elif op == "Add":
                y = (y + k).astype(ft)
            elif op == "Sub":
                y = ((k - y) if left else (y - k)).astype(ft)
            else:
                raise AssertionError(op)
        return y

    def _monotone_chain(self, chain, c) -> bool:
        for op, vec, left in chain:
            k = vec[c] if vec.size > 1 else vec[0]
            if left or (op in ("Mul", "Div") and not k > 0):
                return False
        return True

    def _fold_threshold(self, v: V, mode: str, nd) -> tuple[np.ndarray, dict]:
        """Input threshold t[c] with  binarize(affine_chain(x)) == (x >= t[c])  (or > for gt) on the domain."""
        s, b = np.broadcast_to(v.s, (_channels(v.shape),)), np.broadcast_to(v.b, (_channels(v.shape),))
        if np.any(s <= 0):
            raise QonnxImportError(nd, "binarizer of a float tensor scaled by a non-positive factor (would need an "
                                       "input-channel polarity flip)")
        dom = self.domain.values if v.from_input else None
        thr = np.empty(len(s), np.float32)
        how = {}
        cache = {}
        cmp = (lambda a, t: a >= t) if mode == "ge" else (lambda a, t: a > t)
        for c in range(len(s)):
            key = (float(s[c]), float(b[c]), tuple((op, float(vec[c] if vec.size > 1 else vec[0]), l) for op, vec, l in v.chain))
            if key in cache:
                thr[c] = cache[key]
                continue
            t = np.float32(-b[c] / s[c])
            if not self._monotone_chain(v.chain, c):
                raise QonnxImportError(nd, f"channel {c}: affine chain {[(o, l) for o, _, l in v.chain]} before the binarizer is not monotone")
            if dom is not None:
                g32 = cmp(self._replay(v.chain, dom, c, np.float32), 0)
                g64 = cmp(self._replay(v.chain, dom.astype(np.float64), c, np.float64), 0)
                ok = np.array_equal(cmp(dom, t), g32) and np.array_equal(g32, g64)
                kind = "fl32(-b/s), exhaustive on the domain"
                if not ok:
                    pos = np.where(g32)[0]
                    if pos.size == 0 or not np.all(g32[pos[0]:]) or not np.array_equal(g32, g64):
                        raise QonnxImportError(nd, f"channel {c}: the binarizer is not a single threshold on the {self.domain.name} "
                                                   "domain (or float32 and float64 disagree)")
                    k = pos[0]
                    t = np.float32((float(dom[k]) + float(dom[k - 1])) / 2.0) if k > 0 else np.float32(float(dom[0]) - 1.0)
                    kind = "midpoint between neighbouring domain values"
                    if not np.array_equal(cmp(dom, t), g32):
                        raise QonnxImportError(nd, f"channel {c}: threshold check failed on the domain")
            else:
                def g(xv, ft):
                    return bool(cmp(self._replay(v.chain, np.array([xv], ft), c, ft)[0], 0))
                if mode == "ge":
                    ok = g(t, np.float32) and not g(_prev32(t), np.float32) and g(np.float64(t), np.float64) \
                        and not g(np.float64(_prev32(t)), np.float64)
                else:
                    nxt = np.nextafter(t, np.float32(np.inf), dtype=np.float32)
                    ok = not g(t, np.float32) and g(nxt, np.float32) and not g(np.float64(t), np.float64) and g(np.float64(nxt), np.float64)
                kind = "fl32(-b/s), verified at the neighbouring float32 values (monotone chain)"
                if not ok:
                    raise QonnxImportError(nd, f"channel {c}: fl32(-b/s) = {float(t)!r} is not the exact float32 decision boundary "
                                               "of the source chain; declare a discrete --input-domain or keep the affine")
            thr[c] = t
            cache[key] = t
            how[kind] = how.get(kind, 0) + 1
        info = {"rule": "affine + binarizer -> input threshold", "domain": self.domain.name if dom is not None else "float32",
                "chain": [(o, [float(x) for x in np.unique(vec)][:4], l) for o, vec, l in v.chain],
                "thresholds": [float(x) for x in np.unique(thr)][:8], "how": how}
        self.checks.append({"node": nd.name, **info})
        return thr, info

    def _affine_exact(self, chain) -> bool:
        """ChannelAffine (fl(fl(x*a)+b)) equals the source chain for every float32 input."""
        ops = [op for op, _, left in chain if not left]
        if any(left for _, _, left in chain):
            return False
        pat = ["Mul", "Add"], ["Mul", "Sub"], ["Mul"], ["Add"], ["Sub"], []
        if ops not in [list(p) for p in pat]:
            return False
        for op, vec, _ in chain:
            if not np.array_equal(np.asarray(vec, np.float32).astype(np.float64), vec):
                return False
        return True

    # ---------------------------------------------------------------- quantizers on the input path
    def _quant(self, v: V, nd, scale, zp, bw, signed, narrow, rounding) -> V:
        if rounding != "ROUND":
            raise QonnxImportError(nd, f"IntQuant rounding_mode {rounding} (only ROUND = half-even matches the IR's rne)")
        zp = np.asarray(zp, np.float64).reshape(-1)
        if np.any(zp != 0):
            raise QonnxImportError(nd, "IntQuant with a non-zero zero point is not supported")
        sc = np.asarray(scale, np.float64).reshape(-1)
        bwi = float(np.asarray(bw).reshape(-1)[0])
        if bwi != int(bwi) or bwi < 2 or bwi > 8:
            raise QonnxImportError(nd, f"IntQuant bit width {bwi} (2..8 supported as FakeQuant)")
        bwi = int(bwi)
        qmin = -(2 ** (bwi - 1)) + (1 if narrow else 0) if signed else 0
        qmax = 2 ** (bwi - 1) - 1 if signed else 2 ** bwi - 1 - (1 if narrow else 0)
        c = _channels(v.shape)
        scv = self.per_channel(np.asarray(scale, np.float64), v.shape, nd, "IntQuant scale") if sc.size > 1 else np.full(c, sc[0])
        chain = list(v.chain) if v.kind == "affine" else []
        from_input = v.from_input
        if v.kind == "affine":
            exact = self._affine_exact(v.chain)
            dom = self.domain.values if v.from_input else None
            if not exact and dom is None:
                raise QonnxImportError(nd, f"affine chain {[o for o, _, _ in v.chain]} before IntQuant is not a single Mul+Add; "
                                           "it can be accepted only after an exhaustive check on a declared discrete --input-domain")
            base_v = V("affine", v.t, v.shape, v.layer, v.from_input, s=np.broadcast_to(v.s, (c,)).copy(),
                       b=np.broadcast_to(v.b, (c,)).copy(), chain=chain)
            x = self.mat(base_v, None)
        else:
            exact = True
            x = self.mat(v, nd)
        g = self.g
        name = self.node_name("input_quant" if from_input else f"{v.layer}/quant", nd)
        fq = g.add_node("FakeQuant", [x, g.add_constant(self.uniq(f"{name}/scale"), scv.astype(np.float32)),
                                      g.add_constant(self.uniq(f"{name}/zero_point"), np.zeros(c, np.float32))],
                        {"axis": -1, "pre_scale": 1.0, "qmin": qmin, "qmax": qmax}, name=name)
        self.rule("IntQuant -> FakeQuant")
        if v.kind == "affine" and not exact:
            self._check_quant_on_domain(chain, base_v, scv, qmin, qmax, nd,
                                        lambda xin, c_, ft: _int_quant_ref(self._replay(chain, xin, c_, ft), scv[c_], ft, qmin, qmax))
        self.precision.append({"layer": name, "op": "ChannelAffine + FakeQuant" if v.kind == "affine" else "FakeQuant",
                               "source": f"IntQuant ({'signed' if signed else 'unsigned'} {bwi} bit, scale {np.unique(scv).tolist()[:4]}, "
                                         f"zp 0, ROUND = half-even)" + (f" after {[o for o, _, _ in chain]}" if chain else ""),
                               "activations": f"int{bwi} [{qmin}, {qmax}]", "exact_for_all_float32_inputs": bool(exact)})
        q = float(scv[0]) if np.all(scv == scv[0]) else 0.0
        return V("quant", fq, v.shape, name, False, q=q, qmin=qmin, qmax=qmax, extra={"scale_vec": scv})

    def _check_quant_on_domain(self, chain, base_v, scv, qmin, qmax, nd, src_fn) -> None:
        dom = self.domain.values
        c = _channels(base_v.shape)
        a32 = np.broadcast_to(base_v.s, (c,)).astype(np.float32)
        b32 = np.broadcast_to(base_v.b, (c,)).astype(np.float32)
        min_tie = np.inf
        for ch in range(c):
            xa = (dom * a32[ch] + b32[ch]).astype(np.float32)
            ours = executor._fake_quant(xa, np.full(dom.size, scv[ch], np.float32), np.zeros(dom.size, np.float32),
                                        {"axis": -1, "pre_scale": 1.0, "qmin": qmin, "qmax": qmax}, np.float32)
            src = np.asarray(src_fn(dom, ch, np.float32), np.float32)
            xa64 = dom.astype(np.float64) * float(a32[ch]) + float(b32[ch])
            ours64 = np.clip(np.rint(xa64 / float(np.float32(scv[ch]))), qmin, qmax) * float(np.float32(scv[ch]))
            src64 = np.asarray(src_fn(dom.astype(np.float64), ch, np.float64), np.float64)
            if not (np.array_equal(ours, src) and np.array_equal(ours64, src64) and np.array_equal(ours64, ours.astype(np.float64))):
                raise QonnxImportError(nd, f"input quantizer: ChannelAffine + FakeQuant differs from the source on the "
                                           f"{self.domain.name} domain (channel {ch})")
            r = xa64 / float(np.float32(scv[ch]))
            min_tie = min(min_tie, float(np.min(np.abs(np.abs(r - np.floor(r)) - 0.5))))
        self.checks.append({"node": nd.name, "rule": "input quantizer verified exhaustively", "domain": self.domain.name,
                            "n_values": int(dom.size), "channels": c, "identical_f32": True, "identical_f64": True,
                            "affine": [np.unique(a32).tolist()[:4], np.unique(b32).tolist()[:4]],
                            "min_distance_to_rounding_tie": min_tie})

    def _mt_to_quant(self, v: V, nd) -> V:
        """MultiThreshold (k thresholds) [+ scalar affine on its output] -> ChannelAffine + FakeQuant, domain-checked."""
        if self.domain.values is None or not v.from_input:
            raise QonnxImportError(nd, "a multi-level MultiThreshold is supported only on the input path with a declared "
                                       "discrete --input-domain (exhaustive check)")
        mt = v.extra
        base = v.base
        thr = np.asarray(mt["thr"], np.float32)
        c = _channels(base.shape)
        nthr = thr.shape[1]
        A, B = float(mt["os"]) * float(v.s[0]), float(mt["ob"]) * float(v.s[0]) + float(v.b[0])
        if not (np.all(v.s == v.s[0]) and np.all(v.b == v.b[0])) or A <= 0:
            raise QonnxImportError(nd, "MultiThreshold output affine must be a positive scalar")
        off = B / A
        if off != round(off):
            raise QonnxImportError(nd, f"MultiThreshold levels are not integers times a scale (offset {off})")
        qmin, qmax = int(round(off)), int(round(off)) + nthr
        if qmin < -128 or qmax > 255:
            raise QonnxImportError(nd, f"MultiThreshold levels [{qmin}, {qmax}] do not fit 8 bits")
        scv = np.full(c, A)
        chain = base.chain if base.kind == "affine" else []
        base_v = V("affine", base.t, base.shape, base.layer, True, s=np.broadcast_to(base.s if base.kind == "affine" else 1.0, (c,)).copy(),
                   b=np.broadcast_to(base.b if base.kind == "affine" else 0.0, (c,)).copy(), chain=list(chain))
        post = v.chain

        def src(xin, ch, ft):
            y = self._replay(chain, xin, ch, ft)
            cnt = (y[:, None] >= thr[ch][None, :].astype(ft)).sum(1).astype(ft)
            yy = (ft(mt["os"]) * cnt + ft(mt["ob"])).astype(ft)
            return self._replay(post, yy, ch, ft)
        x = self.mat(base_v, None)
        g = self.g
        name = self.node_name("input_quant", nd)
        fq = g.add_node("FakeQuant", [x, g.add_constant(self.uniq(f"{name}/scale"), scv.astype(np.float32)),
                                      g.add_constant(self.uniq(f"{name}/zero_point"), np.zeros(c, np.float32))],
                        {"axis": -1, "pre_scale": 1.0, "qmin": qmin, "qmax": qmax}, name=name)
        self._check_quant_on_domain(chain, base_v, scv, qmin, qmax, nd, src)
        self.rule("MultiThreshold (input quantizer) -> ChannelAffine + FakeQuant")
        self.precision.append({"layer": name, "op": "ChannelAffine + FakeQuant",
                               "source": f"MultiThreshold ({c}x{nthr} thresholds, out_scale {mt['os']}, out_bias {mt['ob']}) + "
                                         f"{[o for o, _, _ in post]} after {[o for o, _, _ in chain]}",
                               "activations": f"[{qmin}, {qmax}] x {A:g}", "exact_on_domain": self.domain.name})
        v.kind, v.t, v.layer, v.q, v.qmin, v.qmax, v.from_input = "quant", fq, name, A, qmin, qmax, False
        v.extra = {"scale_vec": scv}
        return v

    # ---------------------------------------------------------------- binarizers
    def binarize(self, v: V, nd, mode: str, thr_src=None, act_scale: float = 1.0, pre: str | None = None) -> V:
        """Apply a binarizer (mode ge | gt | sign_zero, optional per-channel threshold) to value v."""
        g = self.g
        c = _channels(v.shape)
        thr_vec = None if thr_src is None else self.per_channel(thr_src, v.shape, nd, "binarizer threshold")
        if not act_scale > 0:
            raise QonnxImportError(nd, f"binarizer scale {act_scale} must be > 0")
        pre = pre or nd.input[0]
        if v.kind == "lin" and not self.single_use(pre):
            self.mat(v, nd)
        rec = {"onnx_node": nd.name, "onnx_op": nd.op_type, "pre": pre, "mode": mode,
               "source_threshold": None if thr_vec is None else [float(t) for t in np.unique(thr_vec)][:4]}
        self.src_thr[nd.name] = 0.0 if thr_vec is None else thr_vec.reshape((1, c) + (1,) * (len(v.shape) - 2))
        if v.kind == "lin" and v.lk in ("binary", "w1a8") and mode in ("ge", "gt"):
            s = v.s.copy()
            b = v.b - (thr_vec if thr_vec is not None else 0.0)
            if np.any(s == 0):
                raise QonnxImportError(nd, f"{int(np.sum(s == 0))} channel(s) with zero scale before the binarizer (constant output)")
            tau = -b / s
            if mode == "gt":   # y > 0 on integers: n >= floor(tau) + 1 (s > 0) / n <= ceil(tau) - 1 (s < 0)
                tau = np.where(s > 0, np.floor(tau) + 1.0, np.ceil(tau) - 1.0)
            bnp = {"scale": s, "shift": b, "tau": tau, "sign": np.sign(s)}
            node = v.node
            if v.lk == "binary":
                mul, add, minfo = integer_threshold(bnp, v.kmax, 1, v.parity if mode == "ge" else None)
                node.inputs[2] = g.add_constant(self.uniq(f"{node.name}/post_mul"), mul)
                node.inputs[3] = g.add_constant(self.uniq(f"{node.name}/post_add"), add)
                self.rule("BN/affine + binarizer -> integer threshold (binary accumulator)")
            else:
                mul, add, minfo = integer_threshold(bnp, v.kmax, v.inv, None)
                sgn = mul.astype(np.float64)
                w = (v.w_sign * sgn[:, None, None, None].astype(np.int8)).astype(np.int8)
                node.inputs[1] = g.add_constant(self.uniq(f"{node.name}/kernel_signed"),
                                                np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1))).astype(np.float32), layout="OHWI")
                bias = (add.astype(np.float64) / v.inv).astype(np.float32)
                if not np.array_equal(bias.astype(np.float64) * v.inv, add.astype(np.float64)):
                    raise QonnxImportError(nd, "W1A8 integer threshold bias is not exact in float32")
                node.inputs[2] = g.add_constant(self.uniq(f"{node.name}/bias"), bias)
                self.rule("BN/affine + binarizer -> integer threshold (W1A8 float Conv2D, exact)")
            self.margins.append({"layer": node.name, "k_max": v.kmax, "mode": mode, **minfo})
            out = V("bin", node.outputs[0], v.shape, node.name, False, mode="ge", scale=act_scale)
            rec.update({"ir_tensor": out.t, "ir_threshold": None, "ir_mode": "ge", "fused": True})
        elif v.kind == "lin" and v.lk in ("binary", "w1a8") and mode == "sign_zero":
            raise QonnxImportError(nd, "Sign (sign(0) = 0) directly on a binary accumulator + affine: exact ternary "
                                       "threshold not implemented in phase 1")
        elif v.kind == "affine" and mode in ("ge", "gt"):
            vv = v
            if thr_vec is not None:
                vv = dataclasses.replace(v, b=np.broadcast_to(v.b, (c,)) - thr_vec, chain=v.chain + [("Sub", thr_vec, False)])
            t32, info = self._fold_threshold(vv, mode, nd)
            name = self.uniq("input/threshold" if (v.from_input and self.naming == "index") else f"{self.onnx_name(nd)}/threshold")
            out = V("bin", v.t, v.shape, v.layer, False, mode=mode, scale=act_scale, thr=t32,
                    thr_name=g.add_constant(name, t32))
            self.rule("affine + binarizer -> input threshold")
            rec.update({"ir_tensor": out.t, "ir_threshold": name, "ir_mode": mode, "fused": True, "threshold_check": info["how"]})
        else:
            x = self.mat(v, nd)
            thr_name, t32 = "", None
            if thr_vec is not None and np.any(thr_vec != 0):
                t32 = thr_vec.astype(np.float32)
                if not np.array_equal(t32.astype(np.float64), thr_vec):
                    raise QonnxImportError(nd, "binarizer threshold is not a float32 value")
                thr_name = g.add_constant(self.uniq(f"{self.onnx_name(nd)}/threshold"), t32)
            out = V("bin", x, v.shape, v.layer, False, mode=mode, scale=act_scale, thr=t32, thr_name=thr_name)
            self.rule(f"binarizer ({mode}) of a float tensor")
            rec.update({"ir_tensor": x, "ir_threshold": thr_name or None, "ir_mode": mode, "fused": False})
        rec["ir_layout"] = "nhwc" if len(v.shape) == 4 else "nc"
        self.binarizers.append(rec)
        return out

    # ---------------------------------------------------------------- linear operators
    def _weight_alpha(self, w: np.ndarray, nd) -> tuple[np.ndarray, np.ndarray] | None:
        """(signs int8, alpha[o]) when every output channel of w is exactly ±alpha[o], else None."""
        w = np.asarray(w, np.float64)
        flat = w.reshape(w.shape[0], -1)
        alpha = np.abs(flat[:, :1]).reshape(-1)
        if np.any(alpha == 0) or not np.all(np.abs(flat) == alpha[:, None]):
            return None
        return np.where(flat > 0, 1, -1).astype(np.int8).reshape(w.shape), alpha

    def _binary_weights_or_fail(self, w, nd, what):
        r = self._weight_alpha(w, nd)
        if r is None:
            flat = np.asarray(w, np.float64).reshape(np.shape(w)[0], -1)
            bad = next(o for o in range(flat.shape[0]) if not (flat[o, 0] != 0 and np.all(np.abs(flat[o]) == abs(flat[o, 0]))))
            vals = np.unique(flat[bad])
            raise QonnxImportError(nd, f"{what}: weight channel {bad} is not ±alpha (values {vals[:6].tolist()}"
                                       f"{'...' if vals.size > 6 else ''}); binary weights are never rounded")
        return r

    def conv(self, nd) -> V:
        a = _attrs(nd)
        xv = self.vals[nd.input[0]]
        w = self.const(nd.input[1])
        if w is None:
            raise QonnxImportError(nd, "Conv weight depends on the input (runtime weights not supported)")
        bias = self.const(nd.input[2]) if len(nd.input) > 2 and nd.input[2] else None
        if len(nd.input) > 2 and nd.input[2] and bias is None:
            raise QonnxImportError(nd, "Conv bias is not constant")
        if a.get("auto_pad", "NOTSET") not in ("NOTSET", ""):
            raise QonnxImportError(nd, f"auto_pad {a['auto_pad']}")
        w = np.asarray(w, np.float64)
        if w.ndim != 4:
            raise QonnxImportError(nd, "only 2-D Conv")
        o, cg, kh, kw = w.shape
        groups = int(a.get("group", 1))
        p = list(a.get("pads", [0, 0, 0, 0]))
        pads = [int(p[0]), int(p[2]), int(p[1]), int(p[3])]        # ONNX [t, l, b, r] -> IR [t, b, l, r]
        attrs = {"strides": [int(s) for s in a.get("strides", [1, 1])], "dilations": [int(d) for d in a.get("dilations", [1, 1])],
                 "pads": pads, "groups": groups}
        oshape = self.shape(nd.output[0])
        g = self.g
        k = cg * kh * kw
        if xv.kind == "bin":
            signs, alpha = self._binary_weights_or_fail(w, nd, "Conv on a binarized tensor")
            pad_value = 0
            if xv.pad is not None:
                (pp, pv) = xv.pad
                if any(pads):
                    raise QonnxImportError(nd, "explicit Pad (±1) of the binarized input and Conv zero padding together")
                pads = [int(pp[2]), int(pp[6]), int(pp[3]), int(pp[7])]
                attrs["pads"] = pads
                pad_value = int(pv)
                self.rule(f"explicit Pad({pv}) -> binary pad_value {pv}")
            elif any(pads):
                self.rule("Conv zero padding of a binarized tensor -> binary pad_value 0")
            name = self.layer_name("conv", nd)
            kb = g.add_constant(self.uniq(f"{name}/kernel_bits"), np.ascontiguousarray(np.transpose(signs, (0, 2, 3, 1))),
                                dtype="bit", layout="OHWI_bit")
            y = g.add_node("BConv2D", [xv.t, kb, "", "", xv.thr_name],
                           {**attrs, "pad_value": pad_value, "input_binarize": xv.mode, "activation": "none", "output": "f32"}, name=name)
            parity = (k % 2) if (pad_value != 0 or not any(pads)) and xv.mode != "sign_zero" else None
            s = alpha * xv.scale
            v = V("lin", y, oshape, name, False, s=s, b=np.zeros(o), node=g.nodes[-1], lk="binary", kmax=k, parity=parity,
                  onnx_node=nd.name)
            self.precision.append({"layer": name, "op": "BConv2D", "source": f"Conv ({nd.name}) on a {xv.mode} binarized tensor",
                                   "weights": "binary ±1" + ("" if np.all(alpha == 1) else " x alpha[o]"),
                                   "activations": f"binary ({xv.mode})", "kernel": [kh, kw], "pads": pads, "groups": groups,
                                   "pad_value": pad_value, "input_threshold": bool(xv.thr_name)})
            self.rule("Conv(Bin(x), ±alpha) -> BConv2D")
        else:
            if xv.kind == "mtcount":
                self._mt_to_quant(xv, nd)
            r = self._weight_alpha(w, nd)
            q = xv.q if xv.kind == "quant" else 0.0
            inv = int(round(1.0 / q)) if q > 0 else 0
            w1a8 = (r is not None and xv.kind == "quant" and q > 0 and inv * q == 1.0 and (inv & (inv - 1)) == 0
                    and bias is None and not any(pads) and groups == 1
                    and k * max(-xv.qmin, xv.qmax) < 2 ** 24)
            x = self.mat(xv, nd)
            name = self.layer_name("conv", nd)
            if w1a8:
                signs, alpha = r
                kern = np.ascontiguousarray(np.transpose(signs, (0, 2, 3, 1))).astype(np.float32)
                y = g.add_node("Conv2D", [x, g.add_constant(self.uniq(f"{name}/kernel"), kern, layout="OHWI"), ""],
                               {**attrs, "pad_value": 0.0, "activation": "none"}, name=name)
                v = V("lin", y, oshape, name, False, s=alpha.copy(), b=np.zeros(o), node=g.nodes[-1], lk="w1a8",
                      kmax=k * max(-xv.qmin, xv.qmax), inv=inv, w_sign=signs, onnx_node=nd.name)
                self.precision.append({"layer": name, "op": "Conv2D (float, exact: multiples of 1/%d)" % inv,
                                       "source": f"Conv ({nd.name}) with ±alpha weights on an int input [{xv.qmin}, {xv.qmax}] x 1/{inv}",
                                       "weights": "binary ±1 (stored as f32)" + ("" if np.all(alpha == 1) else " x alpha[o]"),
                                       "activations": "int8 input (W1A8)", "in": k, "out": o})
                self.rule("Conv(FakeQuant(x), ±alpha) -> W1A8 Conv2D")
            else:
                kern = np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1))).astype(np.float32)
                bname = g.add_constant(self.uniq(f"{name}/bias"), np.asarray(bias, np.float32)) if bias is not None else ""
                y = g.add_node("Conv2D", [x, g.add_constant(self.uniq(f"{name}/kernel"), kern, layout="OHWI"), bname],
                               {**attrs, "pad_value": 0.0, "activation": "none"}, name=name)
                v = V("lin", y, oshape, name, False, s=np.ones(o), b=np.zeros(o), node=g.nodes[-1], lk="float", onnx_node=nd.name)
                self.precision.append({"layer": name, "op": "Conv2D", "source": f"Conv ({nd.name})", "weights": "float32",
                                       "activations": "float32"})
                self.rule("Conv (float) -> Conv2D")
        return v

    def dense(self, nd, w_ko: np.ndarray, bias, xv: V) -> V:
        """x [N, K] @ w_ko [K, O] (+ bias[O])."""
        g = self.g
        oshape = self.shape(nd.output[0])
        k, o = w_ko.shape
        w_ok = np.asarray(w_ko, np.float64).T
        if xv.kind == "bin":
            if xv.pad is not None:
                raise QonnxImportError(nd, "Pad before a dense layer")
            signs, alpha = self._binary_weights_or_fail(w_ok, nd, f"{nd.op_type} on a binarized tensor")
            name = self.layer_name("dense", nd)
            kb = g.add_constant(self.uniq(f"{name}/kernel_bits"), signs, dtype="bit", layout="OI_bit")
            y = g.add_node("BDense", [xv.t, kb, "", "", xv.thr_name], {"input_binarize": xv.mode, "activation": "none"}, name=name)
            v = V("lin", y, oshape, name, False, s=alpha * xv.scale, b=np.zeros(o) if bias is None else np.asarray(bias, np.float64).copy(),
                  node=g.nodes[-1], lk="binary", kmax=k, parity=(k % 2) if xv.mode != "sign_zero" else None, onnx_node=nd.name)
            self.precision.append({"layer": name, "op": "BDense", "source": f"{nd.op_type} ({nd.name}) on a {xv.mode} binarized tensor",
                                   "weights": "binary ±1" + ("" if np.all(alpha == 1) else " x alpha[o]"),
                                   "activations": f"binary ({xv.mode})", "in": k, "out": o, "input_threshold": bool(xv.thr_name)})
            self.rule(f"{nd.op_type}(Bin(x), ±alpha) -> BDense")
            return v
        x = self.mat(xv, nd)
        name = self.layer_name("dense", nd)
        bname = g.add_constant(self.uniq(f"{name}/bias"), np.asarray(bias, np.float32)) if bias is not None else ""
        y = g.add_node("Dense", [x, g.add_constant(self.uniq(f"{name}/kernel"), w_ok.astype(np.float32)), bname],
                       {"activation": "none"}, name=name)
        self.precision.append({"layer": name, "op": "Dense", "source": f"{nd.op_type} ({nd.name})", "weights": "float32"})
        self.rule(f"{nd.op_type} (float) -> Dense")
        return V("float", y, oshape, name, False)

    # ---------------------------------------------------------------- reshape-like
    def reshape(self, nd, v: V, new_shape: tuple) -> V:
        old = tuple(v.shape)
        new_shape = tuple(int(d) for d in new_shape)
        if _ir_shape(old) == _ir_shape(new_shape) and (len(old) != 4 or len(new_shape) == 4):
            return dataclasses.replace(v, shape=new_shape)
        if v.kind == "affine":
            if np.unique(v.s).size == 1 and np.unique(v.b).size == 1 and all(np.unique(vec).size == 1 for _, vec, _ in v.chain):
                base = self.reshape(nd, V("float", v.t, v.shape, v.layer, v.from_input), new_shape)
                cn = _channels(new_shape)
                return dataclasses.replace(v, t=base.t, shape=new_shape, layer=base.layer,
                                           s=np.full(cn, v.s.reshape(-1)[0]), b=np.full(cn, v.b.reshape(-1)[0]),
                                           chain=[(o, np.full(1, vec.reshape(-1)[0]), l) for o, vec, l in v.chain])
            self.mat(v, nd)
        if v.kind not in ("float", "bin", "quant"):
            self.mat(v, nd)
        if v.kind == "bin" and (v.thr is not None and np.unique(v.thr).size > 1 or v.pad is not None):
            raise QonnxImportError(nd, "flatten of a binarized tensor with a per-channel input threshold / pending pad")
        if not (len(new_shape) == 2 and new_shape[0] == old[0]):
            raise QonnxImportError(nd, f"reshape {old} -> {new_shape}: only flattening to [N, K] is supported")
        g = self.g
        x = v.t
        cons = [c.op_type for c in self.consumers.get(nd.output[0], [])]
        nxt = "dense" if any(c in ("MatMul", "Gemm") for c in cons) else "conv" if "Conv" in cons else "reshape"
        if v.t == self.input_name or (v.from_input and v.layer == "input"):
            default = "flatten_input"
        else:
            default = f"{nxt}{self.n_lin}/flatten"
        if len(old) == 4 and old[1] > 1 and old[2] * old[3] > 1:
            x = g.add_node("Transpose", [x], {"perm": [0, 3, 1, 2]}, name=self.node_name(default + "_nchw", nd))
            self.rule("flatten of NCHW [C>1, HW>1] -> Transpose + Reshape")
        name = self.node_name(default, nd)
        y = g.add_node("Reshape", [x], {"shape": [int(d) for d in new_shape]}, name=name)
        out = dataclasses.replace(v, t=y, shape=new_shape, layer=name if v.kind != "bin" else v.layer)
        if v.kind == "bin" and v.thr is not None:
            out.thr = np.full(new_shape[1], v.thr.reshape(-1)[0], np.float32)
            out.thr_name = g.add_constant(self.uniq(f"{name}/threshold"), out.thr)
        self.rule("Reshape/Flatten -> Reshape")
        return out

    # ---------------------------------------------------------------- the walk
    def run(self) -> Graph:
        g = self.g
        if len(self.in_shape) not in (2, 4):
            raise NotImplementedError(f"graph input rank {len(self.in_shape)}")
        g.add_input(self.input_name, _ir_shape(self.in_shape))
        self.used_names.add(self.input_name)
        self.vals[self.in_onnx] = V("float", self.input_name, self.in_shape, "input", True)
        out_names = [o.name for o in self.model.graph.output]
        nodes = [nd for nd in self.model.graph.node if any(o in self.dep for o in nd.output if o)]
        for i, nd in enumerate(nodes):
            if nd.op_type == "TopK" and nd.output[1] in out_names and i == len(nodes) - 1 \
                    and nd.output[0] not in out_names and not self.consumers.get(nd.output[0]):
                self.logits_onnx = nd.input[0]
                self.dropped.append({"node": nd.name, "op": "TopK", "reason": "trailing TopK (class index output); the IR outputs its input (logits)"})
                self.rule("trailing TopK dropped")
                continue
            self.step(nd)
        out = self.logits_onnx or out_names[0]
        if len(out_names) > 1 and not self.logits_onnx:
            raise NotImplementedError(f"graphs with several outputs ({out_names}) are not supported")
        if out not in self.vals:
            raise NotImplementedError(f"graph output {out} does not depend on the input")
        y = self.mat(self.vals[out], None)
        if self.logits_onnx is None:
            self.logits_onnx = out
        g.set_outputs(y)
        return g

    def single_use(self, name: str) -> bool:
        return len(self.consumers.get(name, [])) == 1 and name not in {o.name for o in self.model.graph.output}

    def dyn(self, nd) -> list:
        return [i for i in nd.input if i and i in self.dep]

    def step(self, nd) -> None:
        op = nd.op_type
        a = _attrs(nd)
        g = self.g
        dyn = self.dyn(nd)
        out = nd.output[0]
        oshape = self.shape(out) if out in self.env0 and isinstance(self.env0[out], np.ndarray) else ()

        def setv(v):
            self.vals[out] = v

        if op in ("Identity", "Dropout"):
            setv(self.vals[nd.input[0]])
            return
        if op in ("Mul", "Div", "Add", "Sub") and len(dyn) == 1:
            left = nd.input[0] not in self.dep
            vin = self.vals[dyn[0]]
            cst = self.const(nd.input[0] if left else nd.input[1])
            vec = self.per_channel(cst, vin.shape, nd, op)
            if tuple(oshape) != tuple(vin.shape):
                raise QonnxImportError(nd, f"constant broadcasting changes the shape {vin.shape} -> {oshape}")
            if vin.kind == "bin":
                if op == "Mul" and np.all(vec == vec[0]) and vec[0] > 0 and vin.thr is None:
                    setv(dataclasses.replace(vin, scale=vin.scale * float(vec[0])))
                    self.rule("Mul of a binarized tensor by a positive scalar -> activation scale")
                    return
                vin = dataclasses.replace(vin)
                self.mat(vin, nd)
            if vin.kind == "mtcount":
                if op == "Div" and left or op == "Sub" and left:
                    raise QonnxImportError(nd, "constant on the left of Div/Sub after MultiThreshold")
                v = dataclasses.replace(vin, chain=vin.chain + [(op, vec, left)])
                v.s, v.b = _affine_update(op, vec, left, vin.s, vin.b, nd)
                setv(v)
                return
            if vin.kind == "lin" and not self.single_use(dyn[0]):
                self.mat(vin, nd)
            if vin.kind in ("float", "quant"):
                vin = V("affine", self.mat(vin, nd), vin.shape, vin.layer, vin.from_input, s=np.ones(1), b=np.zeros(1))
            if vin.kind not in ("affine", "lin"):
                vin = V("affine", self.mat(vin, nd), vin.shape, vin.layer, vin.from_input, s=np.ones(1), b=np.zeros(1))
            s, b = _affine_update(op, vec, left, vin.s, vin.b, nd)
            setv(dataclasses.replace(vin, s=s, b=b, chain=vin.chain + [(op, vec, left)]))
            return
        if op == "BatchNormalization":
            vin = self.vals[nd.input[0]]
            if a.get("training_mode", 0):
                raise QonnxImportError(nd, "training_mode")
            gam, bet, mu, var = (np.asarray(self.const(nd.input[i]), np.float64).reshape(-1) for i in (1, 2, 3, 4))
            eps = float(np.float32(a.get("epsilon", 1e-5)))
            k = gam / np.sqrt(var + eps)
            if vin.kind == "lin" and not self.single_use(nd.input[0]):
                self.mat(vin, nd)
            if vin.kind in ("float", "quant", "bin", "mtcount"):
                vin = V("affine", self.mat(vin, nd), vin.shape, vin.layer, vin.from_input, s=np.ones(1), b=np.zeros(1))
            s = np.broadcast_to(vin.s, k.shape) * k
            b = (np.broadcast_to(vin.b, k.shape) - mu) * k + bet
            self.rule("BatchNormalization -> per-channel affine (float64)")
            setv(dataclasses.replace(vin, s=s, b=b, chain=vin.chain + [("BN", np.zeros(1), True)]))
            return
        if op in ("Unsqueeze", "Squeeze", "Reshape", "Flatten"):
            vin = self.vals[nd.input[0]]
            setv(self.reshape(nd, vin, oshape))
            return
        if op == "BipolarQuant":
            sc = np.asarray(self.const(nd.input[1]), np.float64).reshape(-1)
            if sc.size != 1:
                raise QonnxImportError(nd, "per-channel BipolarQuant scale on a runtime tensor")
            setv(self.binarize(self.vals[nd.input[0]], nd, "ge", None, float(sc[0])))
            return
        if op == "Sign":
            setv(self.binarize(self.vals[nd.input[0]], nd, "sign_zero"))
            return
        if op in ("GreaterOrEqual", "Greater"):
            if len(dyn) != 1 or nd.input[0] not in self.dep:
                raise QonnxImportError(nd, "comparison must be (runtime tensor, constant)")
            setv(V("cmp", "", self.vals[nd.input[0]].shape, base=self.vals[nd.input[0]],
                   extra={"mode": "ge" if op == "GreaterOrEqual" else "gt", "t": self.const(nd.input[1]), "node": nd}))
            return
        if op == "Where":
            cv = self.vals.get(nd.input[0])
            u, w_ = self.const(nd.input[1]), self.const(nd.input[2])
            if cv is None or cv.kind != "cmp" or u is None or w_ is None or np.asarray(u).size != 1 or np.asarray(w_).size != 1 \
                    or float(np.asarray(w_).reshape(-1)[0]) != -float(np.asarray(u).reshape(-1)[0]) or float(np.asarray(u).reshape(-1)[0]) <= 0:
                raise QonnxImportError(nd, "only Where(GreaterOrEqual/Greater(x, t), s, -s) with s > 0 is a binarizer")
            setv(self.binarize(cv.base, nd, cv.extra["mode"], cv.extra["t"], float(np.asarray(u).reshape(-1)[0]),
                               pre=cv.extra["node"].input[0]))
            return
        if op == "MultiThreshold":
            vin = self.vals[nd.input[0]]
            thr = self.const(nd.input[1])
            if thr is None:
                raise QonnxImportError(nd, "runtime thresholds")
            if a.get("data_layout", "NCHW") != "NCHW":
                raise QonnxImportError(nd, f"data_layout {a['data_layout']}")
            thr = np.asarray(thr, np.float64)
            os_, ob = float(np.float32(a.get("out_scale", 1.0))), float(np.float32(a.get("out_bias", 0.0)))
            if thr.ndim == 2 and thr.shape[1] == 1 and os_ == 2.0 and ob == -1.0:
                c = _channels(vin.shape)
                tv = thr[:, 0] if thr.shape[0] == c else np.full(c, thr[0, 0])
                if thr.shape[0] not in (1, c):
                    raise QonnxImportError(nd, f"threshold shape {thr.shape} for {c} channels")
                self.rule("MultiThreshold BIPOLAR -> binarizer ge")
                setv(self.binarize(vin, nd, "ge", tv.reshape((1, c) + (1,) * (len(vin.shape) - 2)) if len(vin.shape) > 1 else tv))
                return
            c = _channels(vin.shape)
            if thr.ndim != 2 or thr.shape[0] not in (1, c):
                raise QonnxImportError(nd, f"threshold shape {thr.shape}")
            if vin.kind not in ("float", "affine") or not vin.from_input:
                raise QonnxImportError(nd, "multi-level MultiThreshold is supported only as the input quantizer")
            setv(V("mtcount", "", vin.shape, "input", True, s=np.ones(1), b=np.zeros(1), base=vin,
                   extra={"thr": np.broadcast_to(thr, (c, thr.shape[1])).copy(), "os": os_, "ob": ob}))
            return
        if op in ("Quant", "IntQuant"):
            if len(dyn) != 1 or nd.input[0] not in self.dep:
                raise QonnxImportError(nd, "quantizer parameters must be constant")
            setv(self._quant(self.vals[nd.input[0]], nd, self.const(nd.input[1]), self.const(nd.input[2]), self.const(nd.input[3]),
                             bool(a.get("signed", 1)), bool(a.get("narrow", 0)), a.get("rounding_mode", "ROUND")))
            return
        if op == "Conv":
            setv(self.conv(nd))
            return
        if op in ("MatMul", "Gemm"):
            if len(dyn) != 1 or nd.input[0] not in self.dep:
                raise QonnxImportError(nd, "MatMul of two runtime tensors (BMatMul / MatMul) is phase 3")
            xv = self.vals[nd.input[0]]
            if len(xv.shape) != 2:
                raise QonnxImportError(nd, f"input rank {len(xv.shape)} (only [N, K])")
            w = np.asarray(self.const(nd.input[1]), np.float64)
            bias = None
            if op == "Gemm":
                if a.get("transA", 0):
                    raise QonnxImportError(nd, "transA")
                if a.get("transB", 0):
                    w = w.T
                al, be = float(a.get("alpha", 1.0)), float(a.get("beta", 1.0))
                w = w * al
                if len(nd.input) > 2 and nd.input[2]:
                    bias = self.const(nd.input[2])
                    if bias is None:
                        raise QonnxImportError(nd, "Gemm bias is not constant")
                    bias = np.broadcast_to(np.asarray(bias, np.float64) * be, (w.shape[1],)).copy()
            if xv.kind == "mtcount":
                self._mt_to_quant(xv, nd)
            setv(self.dense(nd, w, bias, xv))
            return
        if op == "MaxPool":
            vin = self.vals[nd.input[0]]
            if a.get("ceil_mode", 0) or list(a.get("dilations", [1, 1])) != [1, 1] or a.get("auto_pad", "NOTSET") not in ("NOTSET", "") \
                    or (len(nd.output) > 1 and nd.output[1]):
                raise QonnxImportError(nd, "ceil_mode / dilations / auto_pad / Indices output")
            p = list(a.get("pads", [0, 0, 0, 0]))
            at = {"kernel": [int(k) for k in a["kernel_shape"]], "strides": [int(s) for s in a.get("strides", [1, 1])],
                  "pads": [int(p[0]), int(p[2]), int(p[1]), int(p[3])]}
            if vin.kind == "bin":
                if vin.pad is not None:
                    raise QonnxImportError(nd, "Pad then MaxPool of a binarized tensor")
                name = self.node_name(f"{vin.layer}/pool", nd)
                y = g.add_node("MaxPool2D", [vin.t], at, name=name)
                self.rule("MaxPool of a binarized tensor -> MaxPool2D on its carrier (ge-sign is monotone)")
                setv(dataclasses.replace(vin, t=y, shape=oshape))
                return
            x = self.mat(vin, nd)
            name = self.node_name(f"{vin.layer}/pool", nd)
            setv(V("float", g.add_node("MaxPool2D", [x], at, name=name), oshape, name, False))
            self.rule("MaxPool -> MaxPool2D")
            return
        if op == "AveragePool":
            vin = self.vals[nd.input[0]]
            if a.get("ceil_mode", 0) or a.get("auto_pad", "NOTSET") not in ("NOTSET", ""):
                raise QonnxImportError(nd, "ceil_mode / auto_pad")
            p = list(a.get("pads", [0, 0, 0, 0]))
            at = {"kernel": [int(k) for k in a["kernel_shape"]], "strides": [int(s) for s in a.get("strides", [1, 1])],
                  "pads": [int(p[0]), int(p[2]), int(p[1]), int(p[3])], "count_include_pad": int(a.get("count_include_pad", 0))}
            x = self.mat(vin, nd)
            name = self.node_name(f"{vin.layer}/avgpool", nd)
            setv(V("float", g.add_node("AvgPool2D", [x], at, name=name), oshape, name, False))
            self.rule("AveragePool -> AvgPool2D")
            return
        if op == "GlobalAveragePool":
            vin = self.vals[nd.input[0]]
            x = self.mat(vin, nd)
            name = self.node_name(f"{vin.layer}/gap", nd)
            setv(V("float", g.add_node("GlobalAvgPool", [x], {"keepdims": 1}, name=name), oshape, name, False))
            self.rule("GlobalAveragePool -> GlobalAvgPool")
            return
        if op == "Pad":
            vin = self.vals[nd.input[0]]
            mode = a.get("mode", "constant")
            pads = a.get("pads") if "pads" in a else [int(v) for v in np.asarray(self.const(nd.input[1])).reshape(-1)]
            val = float(a.get("value", 0.0)) if "pads" in a else (
                float(np.asarray(self.const(nd.input[2])).reshape(-1)[0]) if len(nd.input) > 2 and nd.input[2] else 0.0)
            if mode != "constant" or len(pads) != 8 or any(pads[i] for i in (0, 1, 4, 5)):
                raise QonnxImportError(nd, "only constant spatial Pad of an NCHW tensor")
            if vin.kind == "bin":
                if val not in (-1.0, 0.0, 1.0) or vin.pad is not None:
                    raise QonnxImportError(nd, f"Pad value {val} of a binarized tensor (must be -1, 0 or +1)")
                setv(dataclasses.replace(vin, shape=oshape, pad=(pads, int(val))))
                return
            x = self.mat(vin, nd)
            name = self.node_name(f"{vin.layer}/pad", nd)
            at = {"pads": [0, 0, int(pads[2]), int(pads[6]), int(pads[3]), int(pads[7]), 0, 0], "value": val}
            setv(V("float", g.add_node("Pad", [x], at, name=name), oshape, name, False))
            self.rule("Pad -> Pad")
            return
        act = {"Relu": {"type": "relu"}, "Sigmoid": {"type": "sigmoid"}, "Tanh": {"type": "tanh"}, "HardSwish": {"type": "hardswish"},
               "LeakyRelu": {"type": "leaky_relu", "alpha": float(np.float32(a.get("alpha", 0.01)))}}
        if op in act or op == "Clip" or op == "PRelu" or op == "Gelu":
            vin = self.vals[nd.input[0]]
            x = self.mat(vin, nd)
            name = self.node_name(f"{vin.layer}/{op.lower()}", nd)
            ins = [x]
            if op == "Clip":
                lo = a.get("min") if "min" in a else (self.const(nd.input[1]) if len(nd.input) > 1 and nd.input[1] else None)
                hi = a.get("max") if "max" in a else (self.const(nd.input[2]) if len(nd.input) > 2 and nd.input[2] else None)
                lo = None if lo is None else float(np.asarray(lo).reshape(-1)[0])
                hi = None if hi is None else float(np.asarray(hi).reshape(-1)[0])
                if lo == 0.0 and hi is None:
                    at = {"type": "relu"}
                elif lo == 0.0 and hi == 6.0:
                    at = {"type": "relu6"}
                elif lo is not None and hi is not None:
                    at = {"type": "hardtanh", "min": lo, "max": hi}
                else:
                    raise QonnxImportError(nd, f"Clip({lo}, {hi})")
            elif op == "PRelu":
                slope = self.const(nd.input[1])
                if slope is None:
                    raise QonnxImportError(nd, "runtime PRelu slope")
                ins.append(g.add_constant(self.uniq(f"{name}/slope"), self.per_channel(slope, vin.shape, nd, "PRelu slope").astype(np.float32)))
                at = {"type": "prelu"}
            elif op == "Gelu":
                if a.get("approximate", "none") != "none":
                    raise QonnxImportError(nd, "tanh-approximate Gelu")
                at = {"type": "gelu"}
            else:
                at = act[op]
            setv(V("float", g.add_node("Activation", ins, at, name=name), oshape, name, False))
            self.rule(f"{op} -> Activation({at['type']})")
            return
        if op == "Add" and len(dyn) == 2:
            u, w_ = self.vals[nd.input[0]], self.vals[nd.input[1]]
            xu, xw = self.mat(u, nd), self.mat(w_, nd)
            name = self.node_name(f"{u.layer}/add", nd)
            irop = "Add" if tuple(u.shape) == tuple(w_.shape) else "AddBroadcast"
            setv(V("float", g.add_node(irop, [xu, xw], {"activation": "none"} if irop == "Add" else {}, name=name), oshape, name, False))
            self.rule(f"Add of runtime tensors -> {irop}")
            return
        if op == "Concat" and all(i in self.dep for i in nd.input):
            vs = [self.vals[i] for i in nd.input]
            ax = int(a["axis"]) % len(oshape)
            if ax != 1:
                raise QonnxImportError(nd, "Concat along a non-channel axis")
            xs = [self.mat(v, nd) for v in vs]
            name = self.node_name(f"{vs[0].layer}/concat", nd)
            setv(V("float", g.add_node("Concat", xs, {"axis": -1}, name=name), oshape, name, False))
            self.rule("Concat (channels) -> Concat")
            return
        if op == "Softmax":
            vin = self.vals[nd.input[0]]
            ax = int(a.get("axis", -1)) % len(vin.shape)
            if len(vin.shape) != 2 or ax != 1:
                raise QonnxImportError(nd, "Softmax only over the last axis of [N, K]")
            name = self.node_name(f"{vin.layer}/softmax", nd)
            setv(V("float", g.add_node("Softmax", [self.mat(vin, nd)], {}, name=name), oshape, name, False))
            self.rule("Softmax -> Softmax")
            return
        raise QonnxImportError(nd, "no pattern matches this node (op type not in the phase-1 kernel set or unsupported "
                                   "operand kinds: " + ", ".join(f"{i}={self.vals[i].kind}" for i in dyn if i in self.vals) + ")")


def _affine_update(op, vec, left, s, b, nd):
    """(s, b) of y = x*s + b after applying `op` with a per-channel constant vec (float64)."""
    s = np.asarray(s, np.float64)
    b = np.asarray(b, np.float64)
    if op == "Mul":
        return s * vec, b * vec
    if op == "Div":
        if left:
            raise QonnxImportError(nd, "constant / runtime tensor is not affine")
        return s / vec, b / vec
    if op == "Add":
        return s * np.ones_like(vec), b + vec
    if op == "Sub":
        return (-s * np.ones_like(vec), vec - b) if left else (s * np.ones_like(vec), b - vec)
    raise AssertionError(op)


def _int_quant_ref(x, scale, ft, qmin, qmax):
    s = ft(scale)
    return (np.clip(np.round((np.asarray(x, ft) / s).astype(ft)), qmin, qmax) * s).astype(ft)


# ------------------------------------------------------------------------------------------------ post passes
def set_bit_outputs(g: Graph) -> int:
    """BConv2D whose output feeds only BConv2D nodes (ge binarize, no threshold) emits bits (same ge decision)."""
    cons: dict = {}
    for n in g.nodes:
        for i in n.inputs:
            if i:
                cons.setdefault(i, []).append(n)
    outs = set(g.outputs)
    k = 0
    for n in g.nodes:
        if n.op != "BConv2D" or n.attrs.get("output", "f32") != "f32" or n.attrs.get("activation", "none") != "none":
            continue
        c = cons.get(n.outputs[0], [])
        if n.outputs[0] in outs or not c:
            continue
        if all(m.op == "BConv2D" and m.inputs[0] == n.outputs[0] and m.attrs.get("input_binarize") == "ge"
               and not (len(m.inputs) > 4 and m.inputs[4]) for m in c):
            n.attrs["output"] = "bit"
            k += 1
    return k


def import_model(model, domain: Domain | None = None, naming: str = "index", fold: bool = True) -> tuple[Graph, dict]:
    """Returns (graph, info). info: rules applied, binarizer map, checks, precision, margins, passes."""
    imp = Importer(model, domain, naming)
    g = imp.run()
    applied = passes.optimize(g, fold=fold)
    applied["set_bit_outputs"] = set_bit_outputs(g)
    info = {"rules": imp.rules, "binarizers": imp.binarizers, "checks": imp.checks, "layer_precision": imp.precision,
            "threshold_margins": imp.margins, "passes": applied, "dropped": imp.dropped, "logits_onnx_tensor": imp.logits_onnx,
            "notes": imp.notes, "input_shape_onnx": list(imp.in_shape), "op_counts": g.op_counts(), "_src_thr": imp.src_thr}
    return g, info


# ------------------------------------------------------------------------------------------------ verification
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ir_decision(env, rec, g) -> np.ndarray:
    t = env[rec["ir_tensor"]]
    thr = g.constants[rec["ir_threshold"]].data if rec.get("ir_threshold") else None
    d = executor._binarize(t, rec["ir_mode"], thr)
    if d.ndim == 4:
        d = d.transpose(0, 3, 1, 2)
    return d.reshape(d.shape[0], -1)


def _src_decision(pre: np.ndarray, thr, mode: str) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pre, np.float64) - np.asarray(thr, np.float64)
    d = np.where(p >= 0, 1, -1) if mode == "ge" else np.where(p > 0, 1, -1) if mode == "gt" else np.sign(p).astype(int)
    return d.reshape(d.shape[0], -1), p.reshape(p.shape[0], -1)


def _logits_of(model, x_nhwc, logits: str, ft) -> np.ndarray:
    ev = OnnxF64(model)
    out = []
    for i in range(len(x_nhwc)):
        xi = np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2)) if x_nhwc.ndim == 4 else x_nhwc[i:i + 1]
        out.append(np.asarray(ev.run(xi, ft, outputs=[logits])[logits], ft).reshape(-1))
    return np.stack(out)


def verify(model, g: Graph, info: dict, x_nhwc: np.ndarray, labels: np.ndarray, out_dir: str, raw_model=None) -> dict:
    """Level-3 checks of the QONNX route on the imported graph: qonnx execute_onnx (float32 reference) vs OnnxF64
    (float64 reference) vs the numpy executor of the written file (float32 / float64), plus the first binarization
    flip of every sample that deviates (source pre-activation |x - t| at the flipped positions)."""
    from .validation.metrics import compare_outputs

    ev = OnnxF64(model)
    logits = info["logits_onnx_tensor"]
    n = len(x_nhwc)
    rep = {}
    try:
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.core.onnx_exec import execute_onnx
        from qonnx.transformation.infer_shapes import InferShapes
        mw = ModelWrapper(model).transform(InferShapes())
        have_qonnx = True
    except Exception as e:  # pragma: no cover
        have_qonnx = False
        rep["qonnx_unavailable"] = repr(e)
    bins = info["binarizers"]
    src_thr = info.get("_src_thr", {})
    ref32, ref64, own32, ir32, ir64, flips = [], [], [], [], [], []
    for i in range(n):
        xi = np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2)) if x_nhwc.ndim == 4 else x_nhwc[i:i + 1]
        e64 = ev.run(xi, np.float64, keep=True)
        ref64.append(np.asarray(e64[logits], np.float64).reshape(-1))
        own32.append(np.asarray(ev.run(xi, np.float32, outputs=[logits])[logits], np.float32).reshape(-1))
        ctx = None
        if have_qonnx:
            ctx = execute_onnx(mw, {ev.input_name: xi.astype(np.float32)}, return_full_exec_context=True)
            ref32.append(np.asarray(ctx[logits], np.float32).reshape(-1))
        env32 = executor.run(g, {"input": x_nhwc[i:i + 1]}, keep_intermediates=True)
        env64 = executor.run(g, {"input": x_nhwc[i:i + 1]}, keep_intermediates=True, float_dtype=np.float64)
        ir32.append(env32[g.outputs[0]].reshape(-1))
        ir64.append(env64[g.outputs[0]].reshape(-1))
        rec = {"sample": i}
        for tag, envx, src in (("ir_f64_vs_source_f64", env64, e64), ("ir_f32_vs_source_f32", env32, ctx)):
            if src is None:
                continue
            for r in bins:
                dsrc, pv = _src_decision(src[r["pre"]], src_thr.get(r["onnx_node"], 0.0), r["mode"])
                dir_ = _ir_decision(envx, r, g)
                if dsrc.shape != dir_.shape:
                    raise AssertionError(f"binarizer {r['onnx_node']}: source {dsrc.shape} vs IR {dir_.shape}")
                d = np.where(dsrc[0] != dir_[0])[0]
                if d.size:
                    rec[tag] = {"onnx_node": r["onnx_node"], "n_flips": int(d.size), "max_abs_pre_source": float(np.abs(pv[0, d]).max()),
                                "typical_abs_pre_source": float(np.median(np.abs(pv[0])))}
                    break
        if len(rec) > 1:
            flips.append(rec)
    ref64, own32, ir32, ir64 = np.stack(ref64), np.stack(own32), np.stack(ir32).astype(np.float32), np.stack(ir64)
    ref32 = np.stack(ref32) if ref32 else own32
    ref32.tofile(os.path.join(out_dir, "reference_logits.f32"))
    ref64.tofile(os.path.join(out_dir, "reference_logits_f64.f64"))
    ir32.tofile(os.path.join(out_dir, "numpy_logits_fused.f32"))
    sem = compare_outputs(ref64, ir64, labels)
    per = np.abs(ir64 - ref64).max(axis=1)
    sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per <= 1e-6))
    sem["deviating_samples_gt_1e-6"] = [int(i) for i in np.where(per > 1e-6)[0]]
    floor = compare_outputs(ref64, ref32, labels)
    vs64 = compare_outputs(ref64, ir32, labels)
    slack = max(1, int(round(0.01 * n)))
    rep.update({
        "n_samples": int(n),
        "reference_f32": "qonnx.core.onnx_exec.execute_onnx (float32)" if have_qonnx else "bitino.onnx_f64.OnnxF64 (float32)",
        "reference_f64": "bitino.onnx_f64.OnnxF64 (float64)",
        "reference_f32_vs_f64": floor,
        "onnxf64_in_f32_vs_execute_onnx_f32": compare_outputs(ref32, own32, labels),
        "semantic_check_ir_f64_vs_reference_f64": sem,
        "numpy_f32_vs_reference_f32": compare_outputs(ref32, ir32, labels),
        "numpy_f32_vs_reference_f64": vs64,
        "binarization_flips_vs_source": {"n_binarizers": len(bins), "n_samples_with_flips": len(flips), "samples": flips[:64]},
        "acceptance": {"criterion": f"numpy executor (fp32) top-1 agreement vs fp64 >= 0.98 and disagreements <= source fp32 noise "
                                    f"floor ({floor['n_top1_disagreements']}) + {slack}",
                       "passed": bool(vs64["top1_agreement"] >= 0.98 and vs64["n_top1_disagreements"] <= floor["n_top1_disagreements"] + slack)},
    })
    if raw_model is not None:
        raw64 = _logits_of(raw_model, x_nhwc, logits if logits in {o for nd in raw_model.graph.node for o in nd.output} else
                           raw_model.graph.output[0].name, np.float64)
        rep["cleanup_effect_raw_f64_vs_clean_f64"] = compare_outputs(raw64, ref64, labels) if raw64.shape == ref64.shape else \
            {"note": f"output tensor {logits} not found in the raw graph"}
    return rep


def run_cpp(out_dir: str, eval_dir: str, preprocess: str, bench: str, stage: str, n_classes: int, root: str) -> dict:
    """C++ host runtime vs the numpy executor (tools/validate_model.py, bit-exact required)."""
    js = os.path.join(out_dir, f"validation_host_{stage}.json")
    cmd = [sys.executable, "-I", os.path.join(root, "tools", "validate_model.py"), "--artifacts", out_dir, "--eval-dir", eval_dir,
           "--preprocess", preprocess, "--bench", bench, "--stage", stage, "--threads", "1", "--label", f"qonnx_host_{stage}",
           "--json", js, "--require-exact-numpy"] + (["--num-classes", str(n_classes)] if n_classes else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = {"stage": stage, "returncode": r.returncode, "stdout": r.stdout.strip().splitlines()[-1:] if r.stdout else [],
           "stderr_tail": r.stderr.strip().splitlines()[-3:] if r.stderr else []}
    if os.path.exists(js):
        v = json.load(open(js))
        out.update({"exact_vs_numpy": v.get("vs_numpy_executor_f32", {}).get("exact_match"),
                    "acceptance": v.get("acceptance"), "report": os.path.basename(js)})
    return out


# ------------------------------------------------------------------------------------------------ CLI
def build_parser(prog: str = "import-qonnx") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Import a QONNX / FINN-ONNX graph into a Bitino .bitino and verify it "
                                                         "(docs/qonnx-frontend.md, phase 1).")
    ap.add_argument("onnx", help="exported QONNX / FINN-ONNX file")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-dir", help="eval set directory (manifest.json); enables verification")
    ap.add_argument("--preprocess", help="preprocessing key of the eval set (also recorded in the manifest)")
    ap.add_argument("--input-domain", default="float32", choices=("float32", "uint8", "unit8"),
                    help="values of the graph input: any float32 | float32(k) | float32(k)/255 for uint8 pixels k")
    ap.add_argument("--cleanup", choices=("none", "qonnx"), default="qonnx",
                    help="qonnx: run qonnx's cleanup_model (InferShapes, FoldConstants keeping quant ops, ...) first")
    ap.add_argument("--naming", choices=("index", "onnx"), default="index")
    ap.add_argument("--no-fold", action="store_true", help="do not run passes.fold_affine_into_conv / fold_relu_into_producer")
    ap.add_argument("--exporter", default="", help="exporter and version, e.g. 'brevitas 0.11.0 export_qonnx (torch 2.4.1)'")
    ap.add_argument("--source-json", help="exporter provenance JSON (checkpoint SHA-256, framework versions, source logits files)")
    ap.add_argument("--bench", help="host bitino_bench: run the C++ runtime (tools/validate_model.py, exact stages)")
    ap.add_argument("--stages", default="reference,optimized")
    ap.add_argument("--num-classes", type=int, default=0)
    ap.add_argument("--file-name", default="model.bitino")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import onnx

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    os.makedirs(args.out_dir, exist_ok=True)
    onnx_sha = sha256_file(args.onnx)
    model = onnx.load(args.onnx)
    prov = {"onnx_file": os.path.basename(args.onnx), "onnx_sha256": onnx_sha,
            "onnx_producer": f"{model.producer_name} {model.producer_version}".strip(),
            "opsets": [[o.domain or "ai.onnx", int(o.version)] for o in model.opset_import], "exporter": args.exporter}
    src = json.load(open(args.source_json)) if args.source_json else {}
    if src:
        prov["source"] = {k: v for k, v in src.items() if k not in ("logits_f32", "logits_f64")}
    clean_info = {"cleanup": args.cleanup}
    raw_model = None
    if args.cleanup == "qonnx":
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.util.cleanup import cleanup_model
        raw_model = model
        mw = cleanup_model(ModelWrapper(onnx.load(args.onnx)), preserve_qnt_ops=True)
        model = mw.model
        cpath = os.path.join(args.out_dir, "model_clean.onnx")
        onnx.save(model, cpath)
        clean_info.update({"tool": "qonnx.util.cleanup.cleanup_model(preserve_qnt_ops=True)", "clean_file": os.path.basename(cpath),
                           "clean_sha256": sha256_file(cpath)})
    domain = Domain.parse(args.input_domain)
    try:
        g, info = import_model(model, domain, args.naming, fold=not args.no_fold)
    except QonnxImportError as e:
        print(f"IMPORT FAILED: {e}", file=sys.stderr)
        json.dump({"model_id": args.model_id, "import_error": str(e), "node": e.node_name, "op": e.op},
                  open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
        return 4
    pre_def, x, labels, em = None, None, None, None
    if args.eval_dir:
        em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
        pre = em["preprocessing"][args.preprocess]
        pre_def = pre["definition"]
        x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
        labels = np.array([s["label"] for s in em["samples"]][: len(x)])
    try:
        import qonnx
        qv = getattr(qonnx, "__version__", None)
        if qv is None:
            from importlib.metadata import version
            qv = version("qonnx")
    except Exception:
        qv = None
    g.manifest = {
        "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": "fused",
        "source": {"framework": "QONNX", **prov},
        "conversion": {"tool": "bitino.qonnx_import", "importer": IMPORTER_VERSION, "input_domain": domain.name,
                       "naming": args.naming, "preprocessing_cleanup": clean_info, "rules": info["rules"], "passes": info["passes"],
                       "dropped_nodes": info["dropped"], "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "python": platform.python_version(), "numpy": np.__version__, "onnx": onnx.__version__, "qonnx": qv},
        "input": {"name": "input", "shape": g.inputs[0]["shape"], "layout": "NHWC" if len(g.inputs[0]["shape"]) == 4 else "NC",
                  "dtype": "f32", "preprocessing": pre_def or "", "preprocessing_key": args.preprocess or ""},
        "outputs": ["logits"], "op_versions": {"format": 1},
        "layer_precision": info["layer_precision"], "threshold_margins": info["threshold_margins"],
        "checks": info["checks"],
    }
    path = os.path.join(args.out_dir, args.file_name)
    file_sha = bfformat.write(g, path)
    g = bfformat.read(path)
    from .fetch import data_sha256, graph_sha256
    report = {"model_id": args.model_id, "file": args.file_name, "file_sha256": file_sha, "data_sha256": data_sha256(path),
              "graph_sha256": graph_sha256(path), "file_bytes": os.path.getsize(path), "op_counts": g.op_counts(),
              "provenance": prov, "cleanup": clean_info, "input_domain": domain.name, "import": {k: v for k, v in info.items()
                                                                                                 if k not in ("layer_precision", "_src_thr")}}
    print(f"[import] {args.model_id}: {g.op_counts()} data {report['data_sha256'][:16]} graph {report['graph_sha256'][:16]}")
    if x is not None:
        report["eval_set"] = em["name"]
        report["eval_preprocessing"] = args.preprocess
        report["verification"] = verify(model, g, info, x, labels, args.out_dir, raw_model=raw_model)
        v = report["verification"]
        if src.get("logits_f32") and os.path.exists(src["logits_f32"]):
            from .validation.metrics import compare_outputs
            s32 = np.fromfile(src["logits_f32"], np.float32).reshape(len(x), -1)
            r32 = np.fromfile(os.path.join(args.out_dir, "reference_logits.f32"), np.float32).reshape(len(x), -1)
            v["source_framework_f32_vs_qonnx_f32 (export check)"] = compare_outputs(s32, r32, labels)
            if src.get("logits_f64") and os.path.exists(src["logits_f64"]):
                s64 = np.fromfile(src["logits_f64"], np.float64).reshape(len(x), -1)
                r64 = np.fromfile(os.path.join(args.out_dir, "reference_logits_f64.f64"), np.float64).reshape(len(x), -1)
                v["source_framework_f64_vs_qonnx_f64 (export check)"] = compare_outputs(s64, r64, labels)
        sem = v["semantic_check_ir_f64_vs_reference_f64"]
        print(f"[verify] f64 semantic: max_abs {sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} "
              f"<=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{len(x)} | flips {v['binarization_flips_vs_source']['n_samples_with_flips']} | "
              f"numpy f32 vs qonnx f32 max_abs {v['numpy_f32_vs_reference_f32']['max_abs_error']:.3e} | acceptance {v['acceptance']['passed']}")
        if args.bench:
            report["cpp_host"] = [run_cpp(args.out_dir, args.eval_dir, args.preprocess, args.bench, st, args.num_classes, root)
                                  for st in args.stages.split(",") if st]
            for c in report["cpp_host"]:
                print(f"[cpp] stage {c['stage']}: exact vs numpy {c.get('exact_vs_numpy')} rc {c['returncode']}")
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=_json_default)
    ok = True
    if x is not None:
        ok = report["verification"]["acceptance"]["passed"] and all(c.get("exact_vs_numpy") for c in report.get("cpp_host", []))
    return 0 if ok else 1


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
