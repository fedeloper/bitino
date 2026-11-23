"""QONNX -> Bitino graph IR importer (docs/qonnx-frontend.md, phases 1 and 2; docs/qonnx-phase1.md, qonnx-phase2.md).

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
``flatten_input`` (the convention of ``adapters.finn_bnn``); ``--naming onnx`` uses the ONNX node names;
``--naming exporter`` the names tools/export_torch_qonnx.py records in node metadata (``bitino.name``: torch module
paths), which survive qonnx cleanup. Nodes carrying ``bitino.order`` are walked in the exporter's order.

Phase 2 (plain PyTorch, tools/export_torch_qonnx.py): a binary-weight producer ``BipolarQuant(W_latent, scale[o])``
gives bits = Bin(W_latent) and alpha = scale with its sign (the source's polarity); ``--affine keep`` gives every
affine source node its own ChannelAffine (a constant Mul + Add pair one), i.e. the source's float32 rounding steps,
and ``--constant-order first-use`` the data order of the legacy PyTorch converters (import_model docstring). An
affine value with several consumers is materialized once. None of the layout options changes a computed value.

Step 2 (Keras / Larq, tools/export_keras_qonnx.py, docs/qonnx-step2.md): DoReFa levels ``Round(Mul(Clip(x, 0, 1),
n))`` feeding a Conv / Gemm with ±1 weights -> QBConv2D / QBDense; ``Transpose(0, 2, 3, 1)`` + Reshape (Keras
flatten, channels last) and NCHW reshapes that keep the NHWC element order -> one IR Reshape; ``Reshape [N, C] ->
[N, C, 1, 1]`` + Sigmoid + Mul -> ChannelGate on the [N, C] gate; several graph outputs; a binary operator without a
scale (no group Mul, alpha 1) and a bias -> post_add with post_mul empty; ``bitino.attrs`` null (an attribute at its
default is absent), ``bitino.const_order`` (final data order), ``--bit-outputs none`` (no bit outputs, the Larq
layout); bitino.consts renames are applied all at once.
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

IMPORTER_VERSION = "qonnx_import 2 (phase 2)"
NAMINGS = ("index", "onnx", "exporter")
AFFINE_MODES = ("fold", "keep")
FLATTEN_MODES = ("transpose", "permute")
SOURCE_NAME_KEY = "bitino.name"     # node metadata_props written by tools/export_torch_qonnx.py (survives qonnx cleanup)
SOURCE_ORDER_KEY = "bitino.order"   # position of the node in the exported (uncleaned) graph
FLOAT_LAYER_KEY = "bitino.float_layer"   # node metadata: a declared float layer on a binarized tensor (profile § 7)
UNIQUE_COUNTER_KEY = "bitino.ir_unique_counter"   # model metadata: start value of the IR graph's unique-name counter


def _meta(nd, key: str):
    for p in getattr(nd, "metadata_props", ()):
        if p.key == key:
            return p.value
    return None
_BINARY_LIN = ("BConv2D", "BDense")
_QB_LIN = ("QBConv2D", "QBDense")      # k-bit DoReFa activations x binary weights (docs/qonnx-profile.md § 4.1)
CONST_ORDER_KEY = "bitino.const_order"   # model metadata: data-section order of the final constants (§ 7)
BIT_OUTPUTS = ("set", "none", "threshold")
IMPORT_KEY = "bitino.import"   # model metadata: {"mapping": "groups"} (step 3) | {"mapping": "patterns", <options>} (step 4)
# step 4: the pattern importer's layout options a file may record in bitino.import (mapping "patterns"): option ->
# (argument it replaces, default)
PATTERN_IMPORT_OPTIONS = {"naming": ("naming", "index"), "affine": ("affine", "fold"), "constant_order": ("constant_order", "walk"),
                          "cleanup": ("cleanup", "qonnx"), "bit_outputs": ("bit_outputs", "set"),
                          "flatten_dense": ("flatten_dense", "transpose")}
# bitino.attrs value null = "attribute absent": allowed only where the importer's value is the operator's default
# (docs/operators.md: Conv2D / BConv2D pad_value 0), i.e. the same value in another JSON form (§ 7 (a))
_ATTR_DEFAULTS = {"pad_value": 0.0}
# step 4 (docs/qonnx-step4.md): AvgPool2D count_include_pad 0 is the operator default (docs/operators.md), so null
# may remove it too (the daBNN converter writes no count_include_pad on its unpadded pools); and an attribute the
# importer does not write may be *added* with the operator's default value (the daBNN converter writes
# Add.activation "none"). Both are the same value in another JSON form (§ 7 (a)).
_ATTR_DEFAULTS_STEP4 = {"count_include_pad": 0}
_ATTR_ADD_DEFAULTS_STEP4 = {"activation": "none"}


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
    src_nd: object = None           # affine: the ONNX node that last extended the chain (names under --naming onnx / exporter)


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
    def __init__(self, model, domain: Domain | None = None, naming: str = "index", input_name: str = "input",
                 affine: str = "fold", flatten: str = "transpose"):
        self.model = model
        self.domain = domain or Domain("float32")
        if naming not in NAMINGS:
            raise ValueError(f"naming must be one of {NAMINGS}")
        if affine not in AFFINE_MODES:
            raise ValueError(f"affine must be one of {AFFINE_MODES}")
        self.naming = naming
        self.keep = affine == "keep"
        if flatten not in FLATTEN_MODES:
            raise ValueError(f"flatten must be one of {FLATTEN_MODES}")
        self.flatten = flatten
        self.ev = OnnxF64(model)
        if len(self.ev.inputs) != 1:
            raise NotImplementedError(f"exactly one graph input expected, got {self.ev.inputs}")
        self.in_onnx = self.ev.input_name
        shp = self.ev.input_shape()
        if any(d <= 0 for d in shp):
            raise NotImplementedError(f"graph input {self.in_onnx} needs a static shape, got {shp}")
        self.in_shape = tuple(shp)
        # constant folding + static shapes: one float64 evaluation on a zero input that keeps the input-independent
        # tensors (the constants) and records the shape of every tensor (activations are freed after their last use)
        self.dep = input_dependent(model, [self.in_onnx])
        self.shapes: dict = {}
        consts = {o for nd in model.graph.node for o in nd.output if o and o not in self.dep}
        try:
            self.env0 = self.ev.run(np.zeros(shp, np.float64), np.float64, keep=consts, shapes=self.shapes)
        except OnnxF64Unsupported as e:
            raise QonnxImportError(e.node, f"no float64 reference for this node (bitino.onnx_f64): {e}") from None
        self.consumers: dict[str, list] = {}
        self.producer: dict = {}
        self.by_name = {nd.name: nd for nd in model.graph.node if nd.name}
        for nd in model.graph.node:
            for i in nd.input:
                self.consumers.setdefault(i, []).append(nd)
            for o in nd.output:
                self.producer[o] = nd
        self.g = Graph()
        if naming == "exporter":
            # names the legacy converter drew from Graph.unique() before the passes ran (the passes' constant names
            # continue that counter); recorded by tools/export_torch_qonnx.py in the model's metadata_props
            c = next((p.value for p in getattr(model, "metadata_props", ()) if p.key == UNIQUE_COUNTER_KEY), None)
            if c is not None:
                self.g._counter = int(c)
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
        return self.shapes[name] if name in self.shapes else tuple(np.shape(self.env0[name]))

    def uniq(self, name: str) -> str:
        base, k = name, 1
        while name in self.used_names:
            k += 1
            name = f"{base}_{k}"
        self.used_names.add(name)
        return name

    def onnx_name(self, nd) -> str:
        if self.naming == "exporter" and _meta(nd, SOURCE_NAME_KEY):
            return _meta(nd, SOURCE_NAME_KEY)
        n = nd.name or f"{nd.op_type}_{nd.output[0]}"
        return n.lstrip("/")

    def extra_name(self, nd, i: int, default: str) -> str:
        """Name of the i-th further IR node an ONNX node emits: bitino.extra_names[i] (exporter naming), else default."""
        raw = _meta(nd, "bitino.extra_names") if self.naming == "exporter" else None
        if raw is not None:
            names = json.loads(raw)
            if i >= len(names):
                raise QonnxImportError(nd, f"bitino.extra_names has {len(names)} name(s), the pattern emits more")
            return self.uniq(names[i])
        return self.uniq(default)

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
    def mat(self, v: V, nd=None, parts: str = "auto", name: str | None = None) -> str:
        """IR tensor holding the plain (float) value of v; emits ChannelAffine / epilogues as needed. parts (affine):
        auto = scale and shift, or the shift alone when every scale is 1; bias / scale / both (--affine keep: the
        inputs a ChannelAffine for that one source node has)."""
        g = self.g
        if v.kind in ("float", "quant"):
            return v.t
        if v.kind == "ab_lin":                  # step 1b: AdaBin conv (docs/qonnx-profile.md § 4.1)
            return self._finish_adabin(v)
        if v.kind == "affine":
            c = _channels(v.shape)
            if name is not None:                     # step 4: an extra IR node named by bitino.extra_names
                pre = name
            elif v.from_input and self.naming == "index" and not self._has("input_affine"):
                name = self.uniq("input_affine")
                pre = "input"
            else:
                src = v.src_nd if (self.naming != "index" and v.src_nd is not None) else nd
                name = self.node_name(f"{v.layer or 'x'}/affine", src) if src is not None else self.uniq(f"{v.layer or 'x'}/affine")
                pre = name
            s64 = np.broadcast_to(np.asarray(v.s, np.float64), (c,))
            b64 = np.broadcast_to(np.asarray(v.b, np.float64), (c,))
            if parts == "auto":
                parts = "bias" if np.all(s64 == 1.0) else "both"
            if parts == "bias" and np.all(s64 == 1.0):     # x + b (fl(x*1 + b) = fl(x + b)), the legacy layout
                sc = ""
                sh = g.add_constant(self.uniq(f"{pre}/bias"), b64.astype(np.float32))
                self.rule("affine (scale 1) -> ChannelAffine(bias)")
            elif parts == "scale" and np.all(b64 == 0.0):  # x * s
                sc = g.add_constant(self.uniq(f"{pre}/scale"), s64.astype(np.float32))
                sh = ""
                self.rule("affine (shift 0) -> ChannelAffine(scale)")
            else:
                sc = g.add_constant(self.uniq(f"{pre}/scale"), s64.astype(np.float32))
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
        if node.op in _BINARY_LIN or node.op in _QB_LIN:
            # post_mul is written unless the operator is unscaled: no group Mul, alpha 1, activation scale 1
            # (``extra["scaled"]`` False) and every factor 1 -- then a bias alone is post_add with post_mul empty, the
            # layout of the Larq converter (BinaryFaceNet); a QB operator always has its post_mul (mean|W| / n)
            scaled = node.op in _QB_LIN or v.extra.get("scaled", True) or not np.all(v.s == 1.0)
            if not ident or node.op in _QB_LIN:
                if scaled:
                    node.inputs[2] = g.add_constant(self.uniq(f"{node.name}/post_mul"), np.broadcast_to(v.s, v.b.shape).astype(np.float32))
                if np.any(v.b != 0.0):
                    node.inputs[3] = g.add_constant(self.uniq(f"{node.name}/post_add"), v.b.astype(np.float32))
                self.rule("affine -> binary epilogue post_mul" + ("/post_add" if np.any(v.b != 0.0) else ""))
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
            x = self.mat(base_v, None, **self._step4_quant_affine(nd, chain))
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
        if self._step4_keep_lin(v, nd):
            self.mat(v, nd)
        pooled = self._step4_pool_threshold(v, nd, mode, thr_vec, act_scale, pre)
        if pooled is not None:
            return pooled
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
        elif v.kind == "affine" and mode in ("ge", "gt") and not self.keep:
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

    def _bipolar_weight_factors(self, name: str, w: np.ndarray):
        """(signs, alpha[o]) read from a constant ``BipolarQuant(W_latent, scale)`` weight producer with a scalar or
        per-output-channel scale: bits = Bin(W_latent), alpha = scale *with its sign* (a negative scale keeps the
        source's bit polarity and a negative epilogue factor, the legacy layout; the plain ±alpha rule would flip
        both). Checked to reproduce the folded weight exactly. None if the producer is anything else."""
        nd = self.producer.get(name)
        if nd is None or nd.op_type != "BipolarQuant" or any(i in self.dep for i in nd.input if i):
            return None
        lat = np.asarray(self.env0[nd.input[0]], np.float64)
        sc = np.asarray(self.env0[nd.input[1]], np.float64)
        o = lat.shape[0]
        if lat.shape != np.shape(w):
            return None
        if sc.size == 1:
            alpha = np.full(o, float(sc.reshape(-1)[0]))
        elif sc.size == o and sc.shape[0] == o and all(d == 1 for d in sc.shape[1:]):
            alpha = sc.reshape(-1).copy()
        else:
            return None
        signs = np.where(lat >= 0, 1, -1).astype(np.int8)
        if np.any(alpha == 0) or not np.array_equal(signs * alpha.reshape((o,) + (1,) * (lat.ndim - 1)), np.asarray(w, np.float64)):
            return None
        self.rule("BipolarQuant(W, scale[o]) weights -> bits + signed alpha")
        return signs, alpha

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
        if xv.kind == "bin" and _meta(nd, FLOAT_LAYER_KEY) is not None:   # step 1b: declared float layer (profile § 7)
            return self._float_layer_on_bin(nd, xv, w, bias, attrs, oshape)
        if xv.kind == "bin":
            r = self._bipolar_weight_factors(nd.input[1], w)
            signs, alpha = r if r is not None else self._binary_weights_or_fail(w, nd, "Conv on a binarized tensor")
            signs = self._step4_flip_inputs(xv, signs, pads, nd)
            pad_value = 0
            if any(pads):
                pad_value = 0.0      # Conv zero padding of the binarized tensor (serialized as 0.0, like the legacy converters)
            if xv.pad is not None:
                (pp, pv) = xv.pad
                if any(pads):
                    raise QonnxImportError(nd, "explicit Pad (±1) of the binarized input and Conv zero padding together")
                pads = [int(pp[2]), int(pp[6]), int(pp[3]), int(pp[7])]
                attrs["pads"] = pads
                pad_value = float(pv)      # the Pad's (FLOAT) constant; bitino.attrs can give another JSON form
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
                  onnx_node=nd.name, extra={"scaled": bool(not np.all(alpha == 1.0) or xv.scale != 1.0)})
            self.precision.append({"layer": name, "op": "BConv2D", "source": f"Conv ({nd.name}) on a {xv.mode} binarized tensor",
                                   "weights": "binary ±1" + ("" if np.all(alpha == 1) else " x alpha[o]"),
                                   "activations": f"binary ({xv.mode})", "kernel": [kh, kw], "pads": pads, "groups": groups,
                                   "pad_value": pad_value, "input_threshold": bool(xv.thr_name)})
            self.rule("Conv(Bin(x), ±alpha) -> BConv2D")
        elif xv.kind == "dq":
            # DoReFa k-bit levels Round(Mul(Clip(x, 0, 1), n)) x binary weights -> QBConv2D (profile § 4.1); the
            # Conv's own zero padding is level 0 (the quantized value 0), as the kernel's
            r = self._bipolar_weight_factors(nd.input[1], w)
            signs, alpha = r if r is not None else self._binary_weights_or_fail(w, nd, "Conv on DoReFa levels")
            name = self.layer_name("conv", nd)
            kb = g.add_constant(self.uniq(f"{name}/kernel_bits"), np.ascontiguousarray(np.transpose(signs, (0, 2, 3, 1))),
                                dtype="bit", layout="OHWI_bit")
            y = g.add_node("QBConv2D", [xv.t, kb, "", ""], {**attrs, "activation": "none", "act_bits": xv.extra["bits"],
                                                            "act_quant": "dorefa"}, name=name)
            v = V("lin", y, oshape, name, False, s=alpha.copy(), b=np.zeros(o), node=g.nodes[-1], lk="qb", kmax=k,
                  onnx_node=nd.name, extra={"scaled": True})
            self.precision.append({"layer": name, "op": "QBConv2D", "source": f"Conv ({nd.name}) on DoReFa {xv.extra['bits']}-bit levels",
                                   "weights": "binary ±1" + ("" if np.all(alpha == 1) else " x alpha[o]"),
                                   "activations": f"{xv.extra['bits']}-bit unsigned (DoReFa, [0, 1])", "kernel": [kh, kw], "pads": pads})
            self.rule("Conv(Round(Clip(x, 0, 1) * n), ±alpha) -> QBConv2D")
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
        if xv.extra.get("nhwc_flat"):
            c_, h_, w_ = xv.extra["nhwc_flat"]
            w_ok = np.ascontiguousarray(w_ok.reshape(o, c_, h_, w_).transpose(0, 2, 3, 1).reshape(o, k))
        if xv.kind == "bin":
            if xv.pad is not None:
                raise QonnxImportError(nd, "Pad before a dense layer")
            if xv.extra.get("flip") is not None:
                raise QonnxImportError(nd, "input-channel polarity (pool before BN) before a dense layer")
            signs, alpha = self._binary_weights_or_fail(w_ok, nd, f"{nd.op_type} on a binarized tensor")
            name = self.layer_name("dense", nd)
            kb = g.add_constant(self.uniq(f"{name}/kernel_bits"), signs, dtype="bit", layout="OI_bit")
            y = g.add_node("BDense", [xv.t, kb, "", "", xv.thr_name], {"input_binarize": xv.mode, "activation": "none"}, name=name)
            v = V("lin", y, oshape, name, False, s=alpha * xv.scale, b=np.zeros(o) if bias is None else np.asarray(bias, np.float64).copy(),
                  node=g.nodes[-1], lk="binary", kmax=k, parity=(k % 2) if xv.mode != "sign_zero" else None, onnx_node=nd.name,
                  extra={"scaled": bool(not np.all(alpha == 1.0) or xv.scale != 1.0)})
            self.precision.append({"layer": name, "op": "BDense", "source": f"{nd.op_type} ({nd.name}) on a {xv.mode} binarized tensor",
                                   "weights": "binary ±1" + ("" if np.all(alpha == 1) else " x alpha[o]"),
                                   "activations": f"binary ({xv.mode})", "in": k, "out": o, "input_threshold": bool(xv.thr_name)})
            self.rule(f"{nd.op_type}(Bin(x), ±alpha) -> BDense")
            return v
        if xv.kind == "dq":
            if bias is not None:
                raise QonnxImportError(nd, f"{nd.op_type} bias on DoReFa levels (the profile writes it as the epilogue Add)")
            signs, alpha = self._binary_weights_or_fail(w_ok, nd, f"{nd.op_type} on DoReFa levels")
            name = self.layer_name("dense", nd)
            kb = g.add_constant(self.uniq(f"{name}/kernel_bits"), signs, dtype="bit")    # layout "" (the Larq converter's)
            y = g.add_node("QBDense", [xv.t, kb, "", ""], {"activation": "none", "act_bits": xv.extra["bits"], "act_quant": "dorefa"},
                           name=name)
            self.precision.append({"layer": name, "op": "QBDense", "source": f"{nd.op_type} ({nd.name}) on DoReFa {xv.extra['bits']}-bit levels",
                                   "weights": "binary ±1" + ("" if np.all(alpha == 1) else " x alpha[o]"), "in": k, "out": o})
            self.rule(f"{nd.op_type}(Round(Clip(x, 0, 1) * n), ±alpha) -> QBDense")
            return V("lin", y, oshape, name, False, s=alpha.copy(), b=np.zeros(o), node=g.nodes[-1], lk="qb", kmax=k,
                     onnx_node=nd.name, extra={"scaled": True})
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
        if v.extra.get("gap_flat") and new_shape == old[:2]:      # the GlobalAvgPool already outputs [N, C]
            return dataclasses.replace(v, shape=new_shape, extra={})

        def order_kept(s):      # an NCHW tensor whose element order is also its NHWC order (C == 1 or H * W == 1)
            return len(s) == 4 and (s[1] == 1 or s[2] * s[3] == 1)

        if v.extra.get("nhwc") or (len(old) == 4 and len(new_shape) == 4 and order_kept(old) and order_kept(new_shape)
                                   and _ir_shape(old) != _ir_shape(new_shape)):
            # v.extra nhwc: the ONNX tensor is the NHWC IR tensor itself (Transpose(0, 2, 3, 1) before): a flatten to
            # [N, K] or a reshape to an order-keeping NCHW shape is an IR Reshape of it (Keras Flatten / Reshape,
            # channels last). Otherwise both NCHW shapes keep the NHWC element order: an IR Reshape as well.
            if len(new_shape) == 2 and new_shape[0] == old[0]:
                ir_target = list(new_shape)
            elif order_kept(new_shape):
                ir_target = _ir_shape(new_shape)
            else:
                raise QonnxImportError(nd, f"reshape of an NHWC tensor {old} -> {new_shape}: only [N, K] or an NCHW shape with "
                                           "C == 1 or H * W == 1")
            if int(np.prod(old)) != int(np.prod(new_shape)):
                raise QonnxImportError(nd, f"reshape {old} -> {new_shape} changes the element count")
            x = self.mat(v, nd) if v.kind != "float" else v.t
            name = self.node_name(f"{v.layer}/reshape", nd)
            y = self.g.add_node("Reshape", [x], {"shape": [int(d) for d in ir_target]}, name=name)
            self.rule("Reshape of an NHWC-ordered tensor (Keras flatten / reshape) -> Reshape")
            return V("float", y, new_shape, name, False)
        if len(old) == 2 and len(new_shape) == 4 and new_shape == (old[0], old[1], 1, 1):
            # [N, C] -> [N, C, 1, 1] before a gate Sigmoid (the Keras SE reshape): ChannelGate takes the [N, C] gate
            self.rule("Reshape [N, C] -> [N, C, 1, 1] of a gate -> no IR node")
            return dataclasses.replace(v, shape=new_shape, extra={**v.extra, "nc_as_nc11": True})
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
        nhwc_flat = None
        if len(old) == 4 and old[1] > 1 and old[2] * old[3] > 1:
            cons_nd = self.consumers.get(nd.output[0], [])
            if self.flatten == "permute" and len(cons_nd) == 1 and cons_nd[0].op_type in ("MatMul", "Gemm") \
                    and cons_nd[0].input[0] == nd.output[0] and v.kind != "bin":
                nhwc_flat = (int(old[1]), int(old[2]), int(old[3]))
                self.rule("flatten of NCHW [C>1, HW>1] -> Reshape of NHWC + Dense columns permuted (--flatten-dense permute)")
            else:
                x = g.add_node("Transpose", [x], {"perm": [0, 3, 1, 2]}, name=self.node_name(default + "_nchw", nd))
                self.rule("flatten of NCHW [C>1, HW>1] -> Transpose + Reshape")
        name = self.node_name(default, nd)
        y = g.add_node("Reshape", [x], {"shape": [int(d) for d in new_shape]}, name=name)
        out = dataclasses.replace(v, t=y, shape=new_shape, layer=name if v.kind != "bin" else v.layer,
                                  extra={**v.extra, "nhwc_flat": nhwc_flat} if nhwc_flat else v.extra)
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
        self.out_names = set(out_names)
        nodes = [nd for nd in self.model.graph.node if any(o in self.dep for o in nd.output if o)]
        order = [_meta(nd, SOURCE_ORDER_KEY) for nd in nodes]
        if nodes and all(o is not None and o.isdigit() for o in order):
            # the exporter's node order (a topological order of the exported graph, of which the cleaned graph's
            # nodes are a subset): qonnx cleanup reorders independent branches, the IR keeps the source order
            nodes = [nd for _, nd in sorted(zip((int(o) for o in order), nodes), key=lambda t: t[0])]
            self.notes.append(f"nodes walked in the exporter's order ({SOURCE_ORDER_KEY})")
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
            # several graph outputs (Larq Zoo: logits and probabilities; BRILLsson: embedding and pooled features): the IR
            # outputs are the same tensors in the same order; verification follows the first (the logits)
            missing = [o for o in out_names if o not in self.vals]
            if missing:
                raise NotImplementedError(f"graph outputs {missing} do not depend on the input")
            ys = [self.mat(self.vals[o], None) for o in out_names]
            self.logits_onnx = out_names[0]
            self.rule(f"{len(out_names)} graph outputs -> {len(out_names)} IR outputs")
            g.set_outputs(*ys)
            return g
        if out not in self.vals:
            raise NotImplementedError(f"graph output {out} does not depend on the input")
        y = self.mat(self.vals[out], None)
        if self.logits_onnx is None:
            self.logits_onnx = out
        g.set_outputs(y)
        return g

    def _const_add_follows(self, name: str) -> bool:
        """The only consumer of `name` is an Add / Sub of `name` (left) and a per-channel constant (right)."""
        if not self.single_use(name):
            return False
        c = self.consumers[name][0]
        return (c.op_type in ("Add", "Sub") and len(c.input) == 2 and c.input[0] == name and c.input[1] not in self.dep
                and self.dyn(c) == [name])

    def _same_group(self, nd, onnx_node_name: str) -> bool:
        """nd and the ONNX node named onnx_node_name carry the same exporter group name (bitino.name)."""
        if self.naming != "exporter":
            return False
        other = self.by_name.get(onnx_node_name)
        a, b = _meta(nd, SOURCE_NAME_KEY), _meta(other, SOURCE_NAME_KEY) if other is not None else None
        return a is not None and a == b

    def apply_group_attrs(self) -> int:
        """bitino.attrs (docs/qonnx-profile.md § 7): the JSON form of the group's IR node attributes. A value may only
        replace the importer's value when it is the same number (0 vs 0.0) or the float64 value whose float32
        rounding the importer holds; anything else is an import error (metadata never changes a value)."""
        if self.naming != "exporter":
            return 0
        by_ir = {n.name: n for n in self.g.nodes}
        k = 0
        for nd in self.model.graph.node:
            raw = _meta(nd, "bitino.attrs")
            if raw is None:
                continue
            name = _meta(nd, SOURCE_NAME_KEY)
            ir = by_ir.get(name)
            if ir is None:
                raise QonnxImportError(nd, f"bitino.attrs for group {name!r}, which emits no IR node of that name")
            for key, val in json.loads(raw).items():
                if key not in ir.attrs and _step4_attr_add(ir, key, val):
                    k += 1
                    continue
                if key not in ir.attrs:
                    raise QonnxImportError(nd, f"bitino.attrs: {name} has no attribute {key!r}")
                cur = ir.attrs[key]
                if val is None and _step4_attr_null(ir, key, cur):
                    k += 1
                    continue
                if val is None:
                    # null: the attribute is absent in the IR node, allowed only when the importer's value is the
                    # operator's default (absence = the default, the same value in another JSON form)
                    if key not in _ATTR_DEFAULTS or isinstance(cur, bool) or not isinstance(cur, (int, float)) \
                            or float(cur) != _ATTR_DEFAULTS[key]:
                        raise QonnxImportError(nd, f"bitino.attrs {key}=null would change the value {cur!r} (null only "
                                                   f"removes a default-valued attribute: {sorted(_ATTR_DEFAULTS)})")
                    del ir.attrs[key]
                    k += 1
                    continue
                same = (isinstance(val, (int, float)) and not isinstance(val, bool) and isinstance(cur, (int, float))
                        and (float(val) == float(cur) or float(np.float32(val)) == float(cur)))
                if not same and val != cur:
                    raise QonnxImportError(nd, f"bitino.attrs {key}={val!r} would change the value {cur!r}")
                if type(ir.attrs[key]) is not type(val) or ir.attrs[key] != val:
                    k += 1
                ir.attrs[key] = val
        return k

    def apply_group_consts(self) -> int:
        """bitino.consts (docs/qonnx-profile.md § 7): {"<input index>": name} for the group's IR node,
        {"<extra name>#<input index>": name} for an extra node; renames the constants the importer created."""
        if self.naming != "exporter":
            return 0
        g = self.g
        by_ir = {n.name: n for n in g.nodes}
        ren: dict = {}           # old constant name -> new (all renames are applied at once, so a new name may be one
        src: dict = {}           # the importer gave another constant that is renamed too)
        for nd in self.model.graph.node:
            raw = _meta(nd, "bitino.consts")
            if raw is None:
                continue
            group = _meta(nd, SOURCE_NAME_KEY)
            for key, new in json.loads(raw).items():
                node_name, _, slot = key.rpartition("#")
                ir = by_ir.get(node_name or group)
                if ir is None or int(slot) >= len(ir.inputs) or not ir.inputs[int(slot)] or ir.inputs[int(slot)] not in g.constants:
                    raise QonnxImportError(nd, f"bitino.consts {key!r}: no constant at that IR input")
                old = ir.inputs[int(slot)]
                if old in ren and ren[old] != new:
                    raise QonnxImportError(nd, f"bitino.consts: constant {old!r} renamed to {ren[old]!r} and {new!r}")
                ren[old] = new
                src[old] = nd
        ren = {o: n for o, n in ren.items() if o != n}
        self.const_renames = dict(ren)          # step 4: flip tracing follows renamed thresholds
        if not ren:
            return 0
        final = [ren.get(kk, kk) for kk in g.constants]
        if len(set(final)) != len(final):
            dup = sorted({x for x in final if final.count(x) > 1})[:5]
            raise QonnxImportError(src[next(o for o in ren if ren[o] in dup)], f"bitino.consts: name(s) {dup} already used")
        clash = set(ren.values()) & set(by_ir)
        if clash:
            raise QonnxImportError(src[next(o for o in ren if ren[o] in clash)], f"bitino.consts: {sorted(clash)[:5]} are node names")
        new_consts = {}
        for kk, vv in g.constants.items():          # keeps every constant's position
            if kk in ren:
                vv.name = ren[kk]
            new_consts[vv.name] = vv
        g.constants = new_consts
        for n in g.nodes:
            n.inputs = [ren.get(i, i) for i in n.inputs]
        self.used_names.update(ren.values())
        return len(ren)

    def _step4_quant_affine(self, nd, chain) -> dict:
        """Step 4: the ChannelAffine before an input FakeQuant is the IntQuant group's extra IR node, named by its
        bitino.extra_names[0] (docs/qonnx-profile.md § 7: "a ChannelAffine before a FakeQuant"), and holds both the
        Mul and the Add of the file (also a scale of 1: the BNN-PYNQ CNV writes ChannelAffine(1, 0)). Without the
        key: the default naming and parts (phase 1)."""
        if self.naming != "exporter" or _meta(nd, "bitino.extra_names") is None:
            return {}
        ops = [op for op, _, left in chain if not left]
        return {"name": self.extra_name(nd, 0, "input_affine"), "parts": "both" if ops == ["Mul", "Add"] else "auto"}

    def _step4_pool_threshold(self, v: V, nd, mode: str, thr_vec, act_scale: float, pre: str):
        """Step 4 (BNN-PYNQ Theano CNV, finn_bnn pool="before_bn"): conv -> MaxPool -> BN -> sign, i.e. the ge-sign of
        a per-channel affine of the max-pooled exact integer accumulator of a binary conv (no epilogue, VALID). The
        legacy rule, exact: s = sign(scale), T = the integer threshold of brevitas_bnn.integer_threshold on the
        conv's accumulator (k_max = K, parity K mod 2); channels with s > 0 binarize n >= T, channels with s < 0
        have +1 <=> n <= T, i.e. -(n >= T + 1): the next layer's input threshold is T (s > 0) / T + 1 (s < 0) and its
        weights of the s < 0 input channels are negated (exact without spatial padding: checked in conv()).
        None when the pattern does not apply (fold mode only)."""
        if self.keep or mode != "ge" or thr_vec is not None or v.kind != "affine" or v.from_input \
                or [(op, left) for op, _, left in v.chain] != [("Mul", False), ("Add", False)]:
            return None
        prod = {o: n for n in self.g.nodes for o in n.outputs}
        pool = prod.get(v.t)
        conv = prod.get(pool.inputs[0]) if pool is not None and pool.op == "MaxPool2D" else None
        if conv is None or conv.op != "BConv2D" or any(conv.inputs[2:4]) or conv.attrs.get("output") != "f32" \
                or any(conv.attrs.get("pads", [0])) or conv.attrs.get("input_binarize") == "sign_zero" \
                or any(p for p in pool.attrs.get("pads", [0])):
            return None
        g = self.g
        c = _channels(v.shape)
        s, b = np.broadcast_to(v.s, (c,)).astype(np.float64), np.broadcast_to(v.b, (c,)).astype(np.float64)
        if np.any(s == 0):
            raise QonnxImportError(nd, "zero BN scale before the binarizer (constant channel)")
        k = int(np.prod(g.constants[conv.inputs[1]].shape[1:]))
        mul, add, minfo = integer_threshold({"scale": s, "shift": b, "tau": -b / s, "sign": np.sign(s)}, k, 1, k % 2)
        sg = mul.astype(np.float64)
        t_int = -add.astype(np.float64) * sg
        thr = np.where(sg > 0, t_int, t_int + 1.0).astype(np.float32)
        flip = np.where(sg > 0, 1, -1).astype(np.int8)
        name = g.add_constant(self.uniq(f"{self.onnx_name(nd)}/threshold"), thr)
        self.margins.append({"layer": conv.name, "k_max": k, "mode": mode, "pool_before_bn": True, **minfo})
        self.src_thr[nd.name] = 0.0
        self.binarizers.append({"onnx_node": nd.name, "onnx_op": nd.op_type, "pre": pre, "mode": mode, "source_threshold": None,
                                "ir_tensor": v.t, "ir_threshold": name, "ir_mode": "ge", "fused": True, "ir_layout": "nhwc",
                                "ir_polarity": flip.tolist()})
        self.rule("MaxPool + BN + binarizer of a binary accumulator -> integer input threshold + weight polarity (pool before BN)")
        return V("bin", v.t, v.shape, v.layer, False, mode="ge", scale=act_scale, thr=thr, thr_name=name, extra={"flip": flip})

    def _step4_flip_inputs(self, xv: V, signs: np.ndarray, pads, nd) -> np.ndarray:
        """Negate the binary weights of the input channels a pool-before-BN threshold flipped (step 4)."""
        flip = xv.extra.get("flip") if xv.kind == "bin" else None
        if flip is None:
            return signs
        if any(pads) or xv.pad is not None:
            raise QonnxImportError(nd, "input-channel polarity with spatial padding (not exact)")
        return (signs * flip.reshape(1, -1, 1, 1)).astype(np.int8)

    def _step4_keep_lin(self, v: V, nd) -> bool:
        """Step 4 (docs/qonnx-step4.md): under --affine keep + exporter naming, a binarizer that reads a binary linear
        operator's output from *another* naming group binarizes the operator's (closed) float output: the IR keeps
        both nodes at their positions (the OpenVINO BinaryConvolution -> FakeQuantize -> BinaryConvolution chains,
        bitino.bit_outputs threshold fuses them afterwards). Not the integer threshold of the fold mode, which merges
        the next layer's binarizer into this operator's epilogue."""
        return (self.keep and self.naming == "exporter" and v.kind == "lin" and bool(v.onnx_node)
                and not self._same_group(nd, v.onnx_node))

    def apply_group_const_layouts(self) -> int:
        """bitino.const_layouts (step 4, docs/qonnx-profile.md § 7): {"<input index>": tag} sets the informational
        layout tag of the group IR node's constant at that input; only between tags that name the same storage (a
        rank-2 bit tensor: "OI_bit" or untagged; the tensorpack converter writes its BDense kernels untagged)."""
        if self.naming != "exporter":
            return 0
        by_ir = {n.name: n for n in self.g.nodes}
        k = 0
        for nd in self.model.graph.node:
            raw = _meta(nd, "bitino.const_layouts")
            if raw is None:
                continue
            ir = by_ir.get(_meta(nd, SOURCE_NAME_KEY))
            for slot, tag in json.loads(raw).items():
                cname = ir.inputs[int(slot)] if ir is not None and int(slot) < len(ir.inputs) else ""
                c = self.g.constants.get(cname)
                if c is None:
                    raise QonnxImportError(nd, f"bitino.const_layouts {slot!r}: no constant at that IR input")
                if not (c.dtype == "bit" and len(c.shape) == 2 and {c.layout, tag} <= {"", "OI_bit"}):
                    raise QonnxImportError(nd, f"bitino.const_layouts: {cname} ({c.dtype} {list(c.shape)}, {c.layout!r}) -> "
                                               f"{tag!r} is not the same storage")
                if c.layout != tag:
                    c.layout = tag
                    k += 1
        return k

    def _dorefa_levels(self, nd):
        """n when nd is the Clip(x, 0, 1) of a DoReFa level Round(Mul(Clip(x, 0, 1), n)), n = 2^k - 1 (k = 1..8),
        whose levels feed only Conv / MatMul / Gemm (as their data input); else None."""
        a = _attrs(nd)
        lo = a.get("min") if "min" in a else (self.const(nd.input[1]) if len(nd.input) > 1 and nd.input[1] else None)
        hi = a.get("max") if "max" in a else (self.const(nd.input[2]) if len(nd.input) > 2 and nd.input[2] else None)
        if lo is None or hi is None or float(np.asarray(lo).reshape(-1)[0]) != 0.0 or float(np.asarray(hi).reshape(-1)[0]) != 1.0:
            return None
        if not self.single_use(nd.output[0]):
            return None
        mul = self.consumers[nd.output[0]][0]
        if mul.op_type != "Mul" or len(self.dyn(mul)) != 1 or not self.single_use(mul.output[0]):
            return None
        c = self.const(mul.input[1] if mul.input[0] == nd.output[0] else mul.input[0])
        if c is None or np.asarray(c).size != 1:
            return None
        n = float(np.asarray(c).reshape(-1)[0])
        if n != round(n) or int(n) not in [2 ** k - 1 for k in range(1, 9)]:
            return None
        rnd = self.consumers[mul.output[0]][0]
        if rnd.op_type != "Round" or rnd.output[0] in self.out_names:
            return None
        cons = self.consumers.get(rnd.output[0], [])
        if not cons or any(cc.op_type not in ("Conv", "MatMul", "Gemm") or cc.input[0] != rnd.output[0] for cc in cons):
            return None
        return int(n)

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
        oshape = self.shape(out) if out in self.shapes or (out in self.env0 and isinstance(self.env0[out], np.ndarray)) else ()

        def setv(v):
            self.vals[out] = v

        if self._step_image(nd) or self._step_const_map(nd) or self._step_adabin(nd):    # step 1b (docs/qonnx-step1b.md)
            return
        for i in dyn:
            ex = self.vals[i].extra if i in self.vals else {}
            if ex.get("nhwc") and op not in ("Reshape", "Flatten"):
                raise QonnxImportError(nd, "a Transpose to NHWC is supported only before a Reshape / Flatten (Keras flatten)")
            if ex.get("nc_as_nc11") and op != "Sigmoid":
                raise QonnxImportError(nd, "an [N, C] -> [N, C, 1, 1] reshape is supported only before a gate Sigmoid")
        if op == "Transpose" and len(dyn) == 1:
            # Keras flatten (channels last): Transpose(x, [0, 2, 3, 1]) -> Reshape: the IR tensor already is NHWC, so
            # the Transpose emits nothing and the Reshape reads the IR tensor (element order H, W, C)
            vin = self.vals[nd.input[0]]
            perm = [int(p) for p in a.get("perm", [])]
            if len(vin.shape) != 4 or perm != [0, 2, 3, 1] or vin.kind == "bin":
                raise QonnxImportError(nd, f"Transpose perm {perm} of a {len(vin.shape)}-D {vin.kind} tensor (only NCHW -> NHWC "
                                           "before a flatten)")
            cons = self.consumers.get(out, [])
            if not cons or any(c.op_type not in ("Reshape", "Flatten") for c in cons) or out in self.out_names:
                raise QonnxImportError(nd, "Transpose to NHWC must feed Reshape / Flatten only")
            setv(V("float", self.mat(vin, nd), tuple(oshape), vin.layer, vin.from_input, extra={"nhwc": True}))
            self.rule("Transpose(NCHW -> NHWC) before a flatten -> no IR node")
            return
        if op == "Clip" and self._dorefa_levels(nd) is not None:
            vin = self.vals[nd.input[0]]
            setv(V("dqclip", self.mat(vin, nd), vin.shape, vin.layer, False, extra={"n": self._dorefa_levels(nd)}))
            return
        if op == "Mul" and len(dyn) == 1 and self.vals[dyn[0]].kind == "dqclip":
            vin = self.vals[dyn[0]]
            setv(dataclasses.replace(vin, kind="dqmul"))
            return
        if op == "Round":
            vin = self.vals[nd.input[0]]
            if vin.kind != "dqmul":
                raise QonnxImportError(nd, "Round is supported only as the DoReFa level Round(Mul(Clip(x, 0, 1), n))")
            n = int(vin.extra["n"])
            setv(dataclasses.replace(vin, kind="dq", extra={"n": n, "bits": int(round(np.log2(n + 1)))}))
            self.rule("Round(Mul(Clip(x, 0, 1), n)) -> DoReFa levels (QBConv2D / QBDense input)")
            return
        if op in ("Identity", "Dropout"):
            setv(self.vals[nd.input[0]])
            return
        if op == "Sigmoid":
            # squeeze-and-excitation gate: Mul(x, Sigmoid(g)) with g [N, C, 1, 1] -> ChannelGate(x, g, sigmoid) (profile § 4.3)
            cons = self.consumers.get(out, [])
            vin = self.vals[nd.input[0]]
            if len(cons) == 1 and cons[0].op_type == "Mul" and len(self.dyn(cons[0])) == 2 and out not in self.out_names \
                    and len(vin.shape) == 4 and tuple(vin.shape[2:]) == (1, 1):
                setv(V("gate", self.mat(vin, nd), vin.shape, vin.layer, False))
                return
            if vin.extra.get("nc_as_nc11"):
                raise QonnxImportError(nd, "Sigmoid of a reshaped [N, C] tensor that is not a channel gate Mul(x, Sigmoid(g))")
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
            # --affine keep + exporter naming: a constant Mul / Add / Sub of the binary linear operator's own naming group
            # (docs/qonnx-profile.md § 4.1: Conv -> Mul(post_mul) -> Add(post_add)) is its epilogue, not a ChannelAffine
            if self.keep and vin.kind == "lin" and vin.lk in ("binary", "qb") and not left and op in ("Mul", "Add", "Sub") \
                    and self.single_use(dyn[0]) and self._same_group(nd, vin.onnx_node):
                s, b = _affine_update(op, vec, left, vin.s, vin.b, nd)
                setv(dataclasses.replace(vin, s=s, b=b, extra={**vin.extra, "scaled": vin.extra.get("scaled", True) or op == "Mul"}))
                self.rule(f"{op} of the binary operator's naming group -> epilogue " + ("post_mul" if op == "Mul" else "post_add"))
                return
            # --affine keep: a constant Mul whose only consumer is a constant Add / Sub stays pending and the pair becomes
            # one ChannelAffine(scale, shift) = fl(fl(x * s) + t), the same two float32 roundings (e.g. an exported BN)
            mul_then_add = (self.keep and vin.kind == "affine" and op in ("Add", "Sub") and not left and self.single_use(dyn[0])
                            and [(o, lf) for o, _, lf in vin.chain] == [("Mul", False)])
            if vin.kind in ("lin", "affine") and not mul_then_add and (self.keep or not self.single_use(dyn[0])):
                self.mat(vin, nd)      # the value has other consumers: they and this chain share its float32 rounding
            if vin.kind in ("float", "quant"):
                vin = V("affine", self.mat(vin, nd), vin.shape, vin.layer, vin.from_input, s=np.ones(1), b=np.zeros(1))
            if vin.kind not in ("affine", "lin"):
                vin = V("affine", self.mat(vin, nd), vin.shape, vin.layer, vin.from_input, s=np.ones(1), b=np.zeros(1))
            s, b = _affine_update(op, vec, left, vin.s, vin.b, nd)
            v = dataclasses.replace(vin, s=s, b=b, chain=vin.chain + [(op, vec, left)], src_nd=nd)
            if self.keep and op == "Div" and not np.array_equal((1.0 / vec).astype(np.float32).astype(np.float64) * vec, np.ones_like(vec)):
                raise QonnxImportError(nd, "--affine keep: x / c is a ChannelAffine x * fl32(1/c) only when 1/c is exact "
                                           "(power of two); use --affine fold")
            if self.keep and not (op == "Mul" and not left and self._const_add_follows(out)):
                # one ChannelAffine per source node, at its position (Add/Sub: bias only, Mul/Div: scale only)
                self.mat(v, nd, parts="both" if mul_then_add else
                         "bias" if op in ("Add", "Sub") and not left else "scale" if op in ("Mul", "Div") else "both")
                if mul_then_add:
                    self.rule("Mul + Add by per-channel constants -> one ChannelAffine (keep)")
            setv(v)
            return
        if op == "BatchNormalization":
            vin = self.vals[nd.input[0]]
            if a.get("training_mode", 0):
                raise QonnxImportError(nd, "training_mode")
            gam, bet, mu, var = (np.asarray(self.const(nd.input[i]), np.float64).reshape(-1) for i in (1, 2, 3, 4))
            eps = float(np.float32(a.get("epsilon", 1e-5)))
            k = gam / np.sqrt(var + eps)
            if vin.kind in ("lin", "affine") and (self.keep or not self.single_use(nd.input[0])):
                self.mat(vin, nd)
            if vin.kind in ("float", "quant", "bin", "mtcount"):
                vin = V("affine", self.mat(vin, nd), vin.shape, vin.layer, vin.from_input, s=np.ones(1), b=np.zeros(1))
            s = np.broadcast_to(vin.s, k.shape) * k
            b = (np.broadcast_to(vin.b, k.shape) - mu) * k + bet
            self.rule("BatchNormalization -> per-channel affine (float64)")
            v = dataclasses.replace(vin, s=s, b=b, chain=vin.chain + [("BN", np.zeros(1), True)], src_nd=nd)
            if self.keep:
                self.mat(v, nd, parts="both")
            setv(v)
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
            cons = self.consumers.get(out, [])
            flat = (len(cons) == 1 and cons[0].op_type in ("Flatten", "Reshape", "Squeeze") and out not in self.out_names
                    and tuple(self.shape(cons[0].output[0])) == tuple(oshape[:2]))
            setv(V("float", g.add_node("GlobalAvgPool", [x], {"keepdims": 0 if flat else 1}, name=name), oshape, name, False,
                   extra={"gap_flat": flat}))
            self.rule("GlobalAveragePool + flatten -> GlobalAvgPool(keepdims 0)" if flat else "GlobalAveragePool -> GlobalAvgPool")
            return
        if op == "Pad":
            vin = self.vals[nd.input[0]]
            mode = a.get("mode", "constant")
            pads = a.get("pads") if "pads" in a else [int(v) for v in np.asarray(self.const(nd.input[1])).reshape(-1)]
            val = float(a.get("value", 0.0)) if "pads" in a else (
                float(np.asarray(self.const(nd.input[2])).reshape(-1)[0]) if len(nd.input) > 2 and nd.input[2] else 0.0)
            if mode != "constant" or len(pads) != 8 or pads[0] or pads[4]:
                raise QonnxImportError(nd, "only constant Pad of the C / H / W axes of an NCHW tensor")
            if vin.kind == "bin" and (pads[1] or pads[5]):
                raise QonnxImportError(nd, "channel Pad of a binarized tensor")
            if vin.kind == "bin":
                if val not in (-1.0, 0.0, 1.0) or vin.pad is not None:
                    raise QonnxImportError(nd, f"Pad value {val} of a binarized tensor (must be -1, 0 or +1)")
                setv(dataclasses.replace(vin, shape=oshape, pad=(pads, val)))
                self.rule(f"explicit Pad({val:g}) of a binarized tensor -> binary pad_value")
                return
            x = self.mat(vin, nd)
            name = self.node_name(f"{vin.layer}/pad", nd)
            at = {"pads": [0, 0, int(pads[2]), int(pads[6]), int(pads[3]), int(pads[7]), int(pads[1]), int(pads[5])], "value": val}
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
        if op == "Mul" and len(dyn) == 2:
            u, w_ = self.vals[nd.input[0]], self.vals[nd.input[1]]
            if w_.kind == "gate" or u.kind == "gate":
                xv, gv = (u, w_) if w_.kind == "gate" else (w_, u)
                name = self.node_name(f"{xv.layer}/gate", nd)
                setv(V("float", g.add_node("ChannelGate", [self.mat(xv, nd), gv.t], {"gate": "sigmoid"}, name=name), oshape, name, False))
                self.rule("Mul(x, Sigmoid(g [N, C, 1, 1])) -> ChannelGate(sigmoid)")
                return
            if tuple(u.shape) != tuple(w_.shape):
                raise QonnxImportError(nd, f"Mul of runtime tensors with different shapes {u.shape} x {w_.shape}")
            xu, xw = self.mat(u, nd), self.mat(w_, nd)
            name = self.node_name(f"{u.layer}/mul", nd)
            setv(V("float", g.add_node("Mul", [xu, xw], {}, name=name), oshape, name, False))
            self.rule("Mul of runtime tensors -> Mul")
            return
        if op == "ReduceMean":
            vin = self.vals[nd.input[0]]
            axes = a.get("axes")
            if axes is None and len(nd.input) > 1 and nd.input[1]:
                axes = [int(t) for t in np.asarray(self.const(nd.input[1])).reshape(-1)]
            r = len(vin.shape)
            if r != 4 or axes is None or sorted(int(t) % r for t in axes) != [1, 2, 3] or not a.get("keepdims", 1):
                raise QonnxImportError(nd, "ReduceMean: only the per-sample mean over C, H, W of an NCHW tensor (keepdims 1)")
            x = self.mat(vin, nd)
            n = int(vin.shape[1] * vin.shape[2] * vin.shape[3])
            name = self.node_name(f"{vin.layer}/flatten", nd)
            rr = g.add_node("Reshape", [x], {"shape": [int(vin.shape[0]), n]}, name=name)
            name2 = self.extra_name(nd, 0, f"{name}/mean")
            setv(V("float", g.add_node("ReduceMean", [rr], {"axis": -1, "keepdims": 1}, name=name2), oshape, name2, False))
            self.rule("ReduceMean over C, H, W -> Reshape [N, CHW] + ReduceMean(axis -1)")
            return
        if op == "Sub" and len(dyn) == 2:
            u, w_ = self.vals[nd.input[0]], self.vals[nd.input[1]]
            xu, xw = self.mat(u, nd), self.mat(w_, nd)
            name = self.node_name(f"{w_.layer}/neg", nd)
            c = _channels(w_.shape)
            neg = g.add_node("ChannelAffine", [xw, g.add_constant(self.uniq(f"{name}/scale"), np.full(c, -1.0, np.float32)), ""], name=name)
            name2 = self.extra_name(nd, 0, f"{name}/add")
            irop = "Add" if tuple(u.shape) == tuple(w_.shape) else "AddBroadcast"
            setv(V("float", g.add_node(irop, [xu, neg], {}, name=name2), oshape, name2, False))
            self.rule(f"Sub of runtime tensors -> ChannelAffine(-1) + {irop} (exact: fl(-w) = -w)")
            return
        if op == "Add" and len(dyn) == 2:
            u, w_ = self.vals[nd.input[0]], self.vals[nd.input[1]]
            xu, xw = self.mat(u, nd), self.mat(w_, nd)
            name = self.node_name(f"{u.layer}/add", nd)
            irop = "Add" if tuple(u.shape) == tuple(w_.shape) else "AddBroadcast"
            setv(V("float", g.add_node(irop, [xu, xw], {}, name=name), oshape, name, False))
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

    # ---------------------------------------------------------------- step 1b: image-to-image data movement
    # (docs/qonnx-profile.md § 4.6, docs/qonnx-step1b.md). The IR keeps the legacy NHWC subgraphs of
    # bitino.adapters.restoration_common; the canonical file spells out the same data movement on NCHW tensors.
    _PS_PERM = [0, 1, 4, 2, 5, 3]

    def _step_image(self, nd) -> bool:
        """PixelShuffle(r): Reshape [1, C*r*r, H, W] -> [1, C, r, r, H, W], Transpose perm [0, 1, 4, 2, 5, 3],
        Reshape -> [1, C, H*r, W*r]  ==>  IR Reshape [1, H, W, C, r, r] + Transpose (same perm) + Reshape
        [1, H*r, W*r, C] (restoration_common.pixel_shuffle; the IR node of the group is the last one, the first two
        are bitino.extra_names[0, 1]).
        Row gather (nearest upsampling, replicate padding): Reshape [1, C, H, W] -> [1, C, H*W], Gather(axis 2,
        constant indices [OH, OW] in [0, H*W))  ==>  IR Reshape [H*W, C] + Gather(i32 indices) + Reshape
        [1, OH, OW, C] (restoration_common._row_gather; extra_names[0, 1] = the Reshape and the Gather).
        Both are exact data movement (class B). Returns True when nd was handled."""
        op = nd.op_type
        if op not in ("Reshape", "Transpose", "Gather") or not nd.input or nd.input[0] not in self.vals:
            return False
        vin = self.vals[nd.input[0]]
        out = nd.output[0]
        oshape = tuple(int(d) for d in self.shape(out))
        ishape = tuple(int(d) for d in vin.shape)
        cons = self.consumers.get(out, [])
        g = self.g
        pending = ("ps6", "ps6t", "rows", "split")
        # channel split (BiMatting half block): Reshape [1, C, H, W] -> [1, k, C/k, H*W], Transpose [0, 2, 3, 1] ->
        # [1, C/k, H*W, k]  ==>  IR Reshape [1, H*W, k, C/k] of the NHWC tensor (the same elements in the same order)
        if op == "Reshape" and len(ishape) == 4 and len(oshape) == 4 and vin.kind not in pending and ishape[0] == 1 \
                and oshape[0] == 1 and oshape[1] > 1 and oshape[1] * oshape[2] == ishape[1] and oshape[3] == ishape[2] * ishape[3] \
                and len(cons) == 1 and cons[0].op_type == "Transpose" and [int(p) for p in _attrs(cons[0]).get("perm", [])] == [0, 2, 3, 1]:
            if vin.kind == "bin" or out in self.out_names:
                raise QonnxImportError(nd, "channel split of a binarized tensor / as a graph output")
            self.vals[out] = V("split", self.mat(vin, nd), oshape, vin.layer, False, extra={"k": oshape[1], "p": oshape[2], "hw": oshape[3]})
            return True
        if op == "Transpose" and vin.kind == "split":
            if [int(p) for p in _attrs(nd).get("perm", [])] != [0, 2, 3, 1]:
                raise QonnxImportError(nd, "channel split: Transpose perm must be [0, 2, 3, 1]")
            e = vin.extra
            name = self.node_name(f"{vin.layer}/split", nd)
            y = g.add_node("Reshape", [vin.t], {"shape": [1, e["hw"], e["k"], e["p"]]}, name=name)
            self.vals[out] = V("float", y, oshape, name, False)
            self.rule("channel split (Reshape [1,k,C/k,HW] + Transpose [0,2,3,1]) -> Reshape [1,HW,k,C/k]")
            return True
        # spatial-only reshape [1, C, a, b] -> [1, C, a', b'] (a*b = a'*b'): the NHWC element order is unchanged
        if op == "Reshape" and len(ishape) == 4 and len(oshape) == 4 and vin.kind not in pending and ishape != oshape \
                and ishape[:2] == oshape[:2] and ishape[2] * ishape[3] == oshape[2] * oshape[3]:
            if vin.kind == "bin":
                raise QonnxImportError(nd, "spatial reshape of a binarized tensor")
            name = self.node_name(f"{vin.layer}/reshape", nd)
            y = g.add_node("Reshape", [self.mat(vin, nd)], {"shape": [oshape[0], oshape[2], oshape[3], oshape[1]]}, name=name)
            self.vals[out] = V("float", y, oshape, name, False)
            self.rule("spatial-only Reshape [1,C,a,b] -> [1,C,a',b'] -> Reshape (NHWC order unchanged)")
            return True
        if op == "Reshape" and len(ishape) == 4 and len(oshape) == 6 and vin.kind not in pending:
            n, cin, h, w = ishape
            _, c, r1, r2, h2, w2 = oshape
            if not (n == 1 and oshape[0] == 1 and r1 == r2 and r1 > 1 and c * r1 * r2 == cin and (h2, w2) == (h, w)):
                raise QonnxImportError(nd, f"rank-6 Reshape {ishape} -> {oshape} is not the PixelShuffle split [1, C, r, r, H, W]")
            if len(cons) != 1 or cons[0].op_type != "Transpose" or [int(p) for p in _attrs(cons[0]).get("perm", [])] != self._PS_PERM \
                    or out in self.out_names:
                raise QonnxImportError(nd, f"rank-6 Reshape not followed by the PixelShuffle Transpose {self._PS_PERM}")
            if vin.kind == "bin":
                raise QonnxImportError(nd, "PixelShuffle of a binarized tensor")
            self.vals[out] = V("ps6", self.mat(vin, nd), oshape, vin.layer, False, extra={"r": r1, "c": c, "h": h, "w": w})
            return True
        if op == "Transpose" and vin.kind == "ps6":
            if [int(p) for p in _attrs(nd).get("perm", [])] != self._PS_PERM or len(cons) != 1 or cons[0].op_type != "Reshape" \
                    or out in self.out_names:
                raise QonnxImportError(nd, "PixelShuffle: Transpose perm / consumer")
            self.vals[out] = dataclasses.replace(vin, kind="ps6t", shape=oshape)
            return True
        if op == "Reshape" and vin.kind == "ps6t":
            e = vin.extra
            r, c, h, w = e["r"], e["c"], e["h"], e["w"]
            if oshape != (1, c, h * r, w * r):
                raise QonnxImportError(nd, f"PixelShuffle: final Reshape to {oshape}, expected {(1, c, h * r, w * r)}")
            name = self.node_name(f"{vin.layer}/pixel_shuffle", nd)
            t = g.add_node("Reshape", [vin.t], {"shape": [1, h, w, c, r, r]}, name=self.extra_name(nd, 0, f"{name}/reshape_in"))
            t = g.add_node("Transpose", [t], {"perm": list(self._PS_PERM)}, name=self.extra_name(nd, 1, f"{name}/transpose"))
            y = g.add_node("Reshape", [t], {"shape": [1, h * r, w * r, c]}, name=name)
            self.vals[out] = V("float", y, oshape, name, False)
            self.rule("PixelShuffle (Reshape rank 6 + Transpose + Reshape) -> Reshape [1,H,W,C,r,r] + Transpose + Reshape")
            return True
        if op == "Reshape" and len(ishape) == 4 and len(oshape) == 3 and vin.kind not in pending:
            n, c, h, w = ishape
            if oshape != (1, c, h * w) or n != 1:
                raise QonnxImportError(nd, f"rank-3 Reshape {ishape} -> {oshape}: only [1, C, H*W] (row gather) is supported")
            if not cons or out in self.out_names or any(cn.op_type != "Gather" or int(_attrs(cn).get("axis", 0)) != 2
                                                        or cn.input[0] != out or cn.input[1] in self.dep for cn in cons):
                raise QonnxImportError(nd, "rank-3 Reshape [1, C, H*W] must feed Gather(axis 2) with constant indices only")
            if vin.kind == "bin":
                raise QonnxImportError(nd, "row gather of a binarized tensor")
            self.vals[out] = V("rows", self.mat(vin, nd), oshape, vin.layer, False, extra={"c": c, "h": h, "w": w})
            return True
        if op == "Gather" and vin.kind == "rows":
            e = vin.extra
            c, h, w = e["c"], e["h"], e["w"]
            idx = self.const(nd.input[1])
            if idx is None or int(_attrs(nd).get("axis", 0)) != 2:
                raise QonnxImportError(nd, "row gather: Gather(axis 2) with constant indices only")
            idx = np.asarray(idx)
            if idx.ndim != 2 or idx.dtype.kind not in "iu" or idx.size == 0 or idx.min() < 0 or idx.max() >= h * w:
                raise QonnxImportError(nd, f"row gather: indices must be an integer [OH, OW] tensor in [0, {h * w}) "
                                           f"(got {idx.dtype} {idx.shape})")
            oh, ow = (int(d) for d in idx.shape)
            if oshape != (1, c, oh, ow):
                raise QonnxImportError(nd, f"row gather: output {oshape}, expected {(1, c, oh, ow)}")
            name = self.node_name(f"{vin.layer}/gather", nd)
            t = g.add_node("Reshape", [vin.t], {"shape": [h * w, c]}, name=self.extra_name(nd, 0, f"{name}/rows"))
            ids = g.add_constant(self.uniq(f"{name}/indices"), idx.astype(np.int32), dtype="i32")
            t = g.add_node("Gather", [t, ids], name=self.extra_name(nd, 1, f"{name}/gather"))
            y = g.add_node("Reshape", [t], {"shape": [1, oh, ow, c]}, name=name)
            self.vals[out] = V("float", y, oshape, name, False)
            self.rule("row gather (Reshape [1,C,H*W] + Gather axis 2) -> Reshape [H*W,C] + Gather(i32) + Reshape")
            return True
        return False

    def _step_const_map(self, nd) -> bool:
        """Add(x, M) with a constant M [1, C, H, W] that varies over H / W (not a per-channel constant): IR Add(x, M in
        NHWC), one fp32 addition per element (class B). M's float64 value (a float64 carrier) is stored as fl32(M).
        BNAS's data-independent zero-channel correction of its ge variant."""
        if nd.op_type != "Add" or len(self.dyn(nd)) != 1 or len(nd.input) != 2:
            return False
        left = nd.input[0] not in self.dep
        c = self.const(nd.input[0] if left else nd.input[1])
        vin = self.vals[self.dyn(nd)[0]]
        oshape = tuple(int(d) for d in self.shape(nd.output[0]))
        c = None if c is None else np.asarray(c, np.float64)
        if c is None or c.ndim != 4 or tuple(c.shape) != oshape or tuple(vin.shape) != oshape or len(oshape) != 4:
            return False
        if np.array_equal(c, np.broadcast_to(c[:, :, :1, :1], c.shape)):
            return False                                        # per-channel: the ChannelAffine rules
        if vin.kind == "bin":
            raise QonnxImportError(nd, "Add of a constant map to a binarized tensor")
        x = self.mat(vin, nd)
        name = self.node_name(f"{vin.layer}/add_map", nd)
        m = self.g.add_constant(self.uniq(f"{name}/map"), np.ascontiguousarray(c.transpose(0, 2, 3, 1)).astype(np.float32))
        y = self.g.add_node("Add", [m, x] if left else [x, m], name=name)
        self.vals[nd.output[0]] = V("float", y, oshape, name, False)
        self.rule("Add of a constant [1, C, H, W] map -> Add(x, map NHWC)")
        return True

    def _float_layer_on_bin(self, nd, xv: V, w: np.ndarray, bias, attrs: dict, oshape) -> V:
        """Conv on a binarized tensor declared a float layer (node metadata bitino.float_layer, docs/qonnx-profile.md
        § 7): the source convolves the binarized activation with float weights (BNAS: convs BinOp does not binarize).
        IR: Activation(sign / sign_zero) of the binarizer input (bitino.extra_names[0]) + float Conv2D on it, zero
        padding from the explicit Pad(0). Weights that are +-alpha per output channel are rejected (the declaration
        must not turn a binary layer into a float one)."""
        if self.naming != "exporter":
            raise QonnxImportError(nd, f"{FLOAT_LAYER_KEY} needs --naming exporter")
        if self._weight_alpha(w, nd) is not None:
            raise QonnxImportError(nd, f"{FLOAT_LAYER_KEY} on a Conv whose weights are +-alpha per output channel (a binary layer)")
        if xv.mode not in ("ge", "sign_zero") or xv.scale != 1.0 or (xv.thr is not None and np.any(xv.thr != 0)):
            raise QonnxImportError(nd, f"{FLOAT_LAYER_KEY}: binarizer {xv.mode} with scale {xv.scale} / threshold")
        pads = attrs["pads"]
        if xv.pad is not None:
            (pp, pv) = xv.pad
            if any(pads) or float(pv) != 0.0:
                raise QonnxImportError(nd, f"{FLOAT_LAYER_KEY}: only an explicit Pad(0) of the binarized tensor")
            attrs = {**attrs, "pads": [int(pp[2]), int(pp[6]), int(pp[3]), int(pp[7])]}
        elif any(pads):
            raise QonnxImportError(nd, f"{FLOAT_LAYER_KEY}: implicit Conv padding of a binarized tensor")
        g = self.g
        name = self.layer_name("conv", nd)
        s = g.add_node("Activation", [xv.t], {"type": {"ge": "sign", "sign_zero": "sign_zero"}[xv.mode]},
                       name=self.extra_name(nd, 0, f"{name}/sign"))
        kern = np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1))).astype(np.float32)
        bname = g.add_constant(self.uniq(f"{name}/bias"), np.asarray(bias, np.float32)) if bias is not None else ""
        y = g.add_node("Conv2D", [s, g.add_constant(self.uniq(f"{name}/kernel"), kern, layout="OHWI"), bname],
                       {**attrs, "pad_value": 0.0, "activation": "none"}, name=name)
        o = w.shape[0]
        self.precision.append({"layer": name, "op": "Activation(sign) + Conv2D", "source": f"Conv ({nd.name}) with float weights on a "
                               f"{xv.mode} binarized tensor ({FLOAT_LAYER_KEY})", "weights": "float32", "activations": f"binary ({xv.mode}) as float"})
        self.rule(f"Conv(Bin(x), float W) declared {FLOAT_LAYER_KEY} -> Activation(sign) + Conv2D")
        return V("lin", y, oshape, name, False, s=np.ones(o), b=np.zeros(o), node=g.nodes[-1], lk="float", onnx_node=nd.name)

    # ---------------------------------------------------------------- step 1b: AdaBin binary conv (profile § 4.1)
    def _scalar_const(self, name, nd, what):
        c = self.const(name)
        v = None if c is None else np.unique(np.asarray(c, np.float64))
        if v is None or v.size != 1:
            raise QonnxImportError(nd, f"AdaBin conv: {what} must be one constant value")
        return float(v[0])

    def _adabin_chain(self, nd) -> bool:
        """nd = Sub(x, beta) starting Bin(.) -> Add(., beta) -> Mul(., alpha) -> [Pad(., 0)] -> Conv (single consumers)."""
        def one(t, ops):
            c = self.consumers.get(t, [])
            return c[0] if len(c) == 1 and c[0].op_type in ops and t not in self.out_names else None
        if nd.input[1] in self.dep:
            return False
        b = one(nd.output[0], ("Sign", "BipolarQuant"))
        a = b is not None and one(b.output[0], ("Add",))
        m = a and one(a.output[0], ("Mul",))
        if not m:
            return False
        p = one(m.output[0], ("Pad", "Conv"))
        return p is not None and (p.op_type == "Conv" or one(p.output[0], ("Conv",)) is not None)

    def _step_adabin(self, nd) -> bool:
        """AdaBin (huawei-noah AdaBin_Conv2d): the activation a = alpha * (Bin(x - beta) + beta) in {alpha(beta - 1),
        alpha(beta + 1)} (the source's two fp32 operations on the sign; the source binarizes (x - beta) / alpha, the same
        sign for alpha > 0 except where the fp32 division underflows to 0), the conv with per-output-channel two-level
        weights W in {W-, W+}, zero padding (0 is not in the activation set) and the affine (BatchNorm) after it ==> the
        legacy decomposition of bitino.adapters.torch_adabin: BConv2D(1 output, all +1 weights) + Conv2D 1x1 (the sum of
        s per pixel times g*q*c), the main BConv2D (post_mul g*q*d), Add, Add(constant border map g*p*(d*sum_w + c*K) + h),
        every constant in float64 from the file's values in the adapter's order. Emitted at the end of the affine chain."""
        op = nd.op_type
        dyn = self.dyn(nd)
        if not dyn or dyn[0] not in self.vals:
            return False
        vin = self.vals[dyn[0]]
        out = nd.output[0]
        e = dict(vin.extra) if vin.kind.startswith("ab_") else {}
        if op == "Sub" and len(dyn) == 1 and nd.input[0] == dyn[0] and not vin.kind.startswith("ab_") and vin.kind != "bin" \
                and len(vin.shape) == 4 and self._adabin_chain(nd):
            beta = self._scalar_const(nd.input[1], nd, "the threshold beta (Sub)")
            self.vals[out] = V("ab_sub", self.mat(vin, nd), vin.shape, vin.layer, False, extra={"beta": beta, "sub": nd})
            return True
        if vin.kind == "ab_sub" and op in ("Sign", "BipolarQuant"):
            if op == "BipolarQuant" and self._scalar_const(nd.input[1], nd, "the BipolarQuant scale") != 1.0:
                raise QonnxImportError(nd, "AdaBin conv: BipolarQuant scale must be 1")
            e.update(mode="sign_zero" if op == "Sign" else "ge", bin=nd)
            self.vals[out] = dataclasses.replace(vin, kind="ab_bin", extra=e)
            return True
        if vin.kind == "ab_bin" and op == "Add" and len(dyn) == 1:
            b2 = self._scalar_const(nd.input[1] if nd.input[0] == dyn[0] else nd.input[0], nd, "beta (Add)")
            if b2 != e["beta"]:
                raise QonnxImportError(nd, f"AdaBin conv: Add constant {b2!r} differs from the threshold beta {e['beta']!r}")
            self.vals[out] = dataclasses.replace(vin, kind="ab_add", extra=e)
            return True
        if vin.kind == "ab_add" and op == "Mul" and len(dyn) == 1:
            alpha = self._scalar_const(nd.input[1] if nd.input[0] == dyn[0] else nd.input[0], nd, "alpha (Mul)")
            if not alpha > 0:
                raise QonnxImportError(nd, f"AdaBin conv: alpha {alpha} must be > 0")
            e.update(alpha=alpha, hw=tuple(int(d) for d in vin.shape[2:]))      # the unpadded map
            self.vals[out] = dataclasses.replace(vin, kind="ab_act", extra=e)
            return True
        if vin.kind == "ab_act" and op == "Pad":
            a = _attrs(nd)
            pads = [int(v) for v in np.asarray(self.const(nd.input[1])).reshape(-1)]
            val = float(np.asarray(self.const(nd.input[2])).reshape(-1)[0]) if len(nd.input) > 2 and nd.input[2] else 0.0
            if a.get("mode", "constant") != "constant" or val != 0.0 or len(pads) != 8 or pads[0] or pads[1] or pads[4] or pads[5] \
                    or "pad" in e:
                raise QonnxImportError(nd, "AdaBin conv: only one constant Pad(0) of H / W")
            e["pad"] = [pads[2], pads[3], pads[6], pads[7]]          # t, l, b, r
            self.vals[out] = dataclasses.replace(vin, shape=self.shape(out), extra=e)
            return True
        if vin.kind == "ab_act" and op == "Conv":
            a = _attrs(nd)
            w = self.const(nd.input[1])
            if w is None or (len(nd.input) > 2 and nd.input[2]) or int(a.get("group", 1)) != 1 \
                    or [int(d) for d in a.get("dilations", [1, 1])] != [1, 1]:
                raise QonnxImportError(nd, "AdaBin conv: constant weights, no bias, group 1, no dilation")
            p = [int(v) for v in a.get("pads", [0, 0, 0, 0])]            # t, l, b, r
            if "pad" in e:
                if any(p):
                    raise QonnxImportError(nd, "AdaBin conv: explicit Pad and Conv pads together")
                p = e["pad"]
            st = [int(s) for s in a.get("strides", [1, 1])]
            if p[0] != p[2] or p[1] != p[3] or p[0] != p[1] or st[0] != st[1]:
                raise QonnxImportError(nd, "AdaBin conv: symmetric square padding and stride only")
            w = np.asarray(w, np.float64)
            o = w.shape[0]
            flat = w.reshape(o, -1)
            wp, wm = flat.max(1), flat.min(1)
            if not np.all((flat == wp[:, None]) | (flat == wm[:, None])) or not np.all(wp > wm):
                raise QonnxImportError(nd, "AdaBin conv: the weights must take exactly two values per output channel")
            e.update(conv=nd, w=w, wp=wp, wm=wm, pad=p[0], stride=st[0])
            v = V("ab_lin", vin.t, self.shape(out), vin.layer, False, s=np.ones(o), b=np.zeros(o), extra=e)
            return self._adabin_continue(v, out)
        if vin.kind == "ab_lin" and op in ("Mul", "Add") and len(dyn) == 1 and nd.input[0] == dyn[0]:
            vec = self.per_channel(self.const(nd.input[1]), vin.shape, nd, op)
            s, b = _affine_update(op, vec, False, vin.s, vin.b, nd)
            return self._adabin_continue(dataclasses.replace(vin, s=s, b=b, src_nd=nd), out)
        return False

    def _adabin_continue(self, v: V, out: str) -> bool:
        """Keep accumulating while the only consumer is a per-channel constant Mul / Add, else emit the IR nodes now
        (the adapter emits the decomposition at the conv + BatchNorm, before the shortcut)."""
        c = self.consumers.get(out, [])
        nxt = c[0] if len(c) == 1 and out not in self.out_names else None
        more = nxt is not None and nxt.op_type in ("Mul", "Add") and len(self.dyn(nxt)) == 1 and nxt.input[0] == out
        if more:
            try:
                self.per_channel(self.const(nxt.input[1]), v.shape, nxt, nxt.op_type)
            except QonnxImportError:
                more = False
        self.vals[out] = v
        if not more:
            self.vals[out] = V("float", self._finish_adabin(v), v.shape, v.layer, False)
        return True

    def _finish_adabin(self, v: V) -> str:
        e = v.extra
        g = self.g
        cnd, sub, bnd = e["conv"], e["sub"], e["bin"]
        w = e["w"]
        o, cin, kh, kw = w.shape
        h, wd = e["hw"]
        pd, st = e["pad"], e["stride"]
        ho, wo = (h + 2 * pd - kh) // st + 1, (wd + 2 * pd - kw) // st + 1
        alpha, beta, mode = e["alpha"], e["beta"], e["mode"]
        sign = np.where(w == e["wp"][:, None, None, None], 1, -1).astype(np.int8)
        # the adapter's float64 algebra (torch_adabin.adabin_conv_bn), in its operation order
        a_plus, a_minus = alpha * (beta + 1.0), alpha * (beta - 1.0)
        p, q = (a_plus + a_minus) / 2.0, (a_plus - a_minus) / 2.0
        c_o, d_o = (e["wp"] + e["wm"]) / 2.0, (e["wp"] - e["wm"]) / 2.0
        gsc, gsh = np.broadcast_to(v.s, (o,)).astype(np.float64), np.broadcast_to(v.b, (o,)).astype(np.float64)
        inb = np.zeros((kh, kw, ho, wo))           # in-bounds taps (zero padding: out-of-bounds taps contribute 0)
        for ky in range(kh):
            for kx in range(kw):
                iy, ix = np.arange(ho) * st - pd + ky, np.arange(wo) * st - pd + kx
                inb[ky, kx] = ((iy >= 0) & (iy < h))[:, None] & ((ix >= 0) & (ix < wd))[None, :]
        sw_map = np.einsum("yxhw,oyx->hwo", inb, sign.sum(axis=1).astype(np.float64))    # integer sums: exact
        k_map = inb.sum(axis=(0, 1)) * cin
        const_map = gsc * p * (d_o * sw_map + c_o * k_map[..., None]) + gsh
        name = self.layer_name("conv", cnd)
        attrs = {"strides": [st, st], "dilations": [1, 1], "pads": [pd, pd, pd, pd], "groups": 1, "pad_value": 0.0,
                 "input_binarize": mode, "activation": "none", "output": "f32"}
        thr = g.add_constant(self.uniq(f"{name}/threshold"), np.full(cin, np.float32(beta)))
        cnt = g.add_node("BConv2D", [v.t, g.add_constant(self.uniq(f"{name}/ones_bits"), np.ones((1, kh, kw, cin), np.int8),
                                                         dtype="bit", layout="OHWI_bit"), "", "", thr], attrs,
                         name=self.extra_name(cnd, 0, f"{name}/sum_s"))
        bsum = g.add_node("Conv2D", [cnt, g.add_constant(self.uniq(f"{name}/sum_s_scale"), (gsc * q * c_o).reshape(o, 1, 1, 1).astype(np.float32)), ""],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0, "activation": "none"},
                          name=self.extra_name(cnd, 1, f"{name}/sum_s_broadcast"))
        main = g.add_node("BConv2D", [v.t, g.add_constant(self.uniq(f"{name}/kernel_bits"), np.ascontiguousarray(sign.transpose(0, 2, 3, 1)),
                                                          dtype="bit", layout="OHWI_bit"),
                                      g.add_constant(self.uniq(f"{name}/post_mul"), (gsc * q * d_o).astype(np.float32)), "", thr], attrs, name=name)
        y = g.add_node("Add", [main, bsum], name=self.extra_name(cnd, 2, f"{name}/add_sum_s"))
        last = self.node_name(f"{name}/affine", v.src_nd) if v.src_nd is not None else self.extra_name(cnd, 3, f"{name}/affine")
        y = g.add_node("Add", [y, g.add_constant(self.uniq(f"{name}/border_map"), const_map.reshape(1, ho, wo, o).astype(np.float32))],
                       name=last)
        self.src_thr[bnd.name] = 0.0
        self.binarizers.append({"onnx_node": bnd.name, "onnx_op": bnd.op_type, "pre": sub.output[0], "mode": mode,
                                "source_threshold": None, "ir_tensor": v.t, "ir_threshold": thr, "ir_mode": mode, "fused": True,
                                "ir_layout": "nhwc"})
        self.precision.append({"layer": name, "op": "BConv2D(O) + BConv2D(1) + Conv2D(1x1) + Add(const map)",
                               "source": f"AdaBin conv ({cnd.name}): alpha*(Bin(x - beta) + beta) * two-level weights + affine",
                               "weights": "binary (W+ / W- per output channel)", "activations": f"binary {{alpha(beta-1), alpha(beta+1)}} ({mode})",
                               "threshold": beta})
        self.rule("AdaBin conv (Bin(x - beta) + beta) * alpha, two-level weights [+ affine] -> BConv2D x2 + Conv2D 1x1 + Add + Add(map)")
        v.kind, v.t = "float", y
        return y


def _step4_attr_add(ir, key: str, val) -> bool:
    """bitino.attrs (step 4): an attribute the importer did not write, added with the operator's default value."""
    dflt = _ATTR_ADD_DEFAULTS_STEP4.get(key, ())
    if dflt != () and type(val) is type(dflt) and val == dflt:
        ir.attrs[key] = val
        return True
    return False


def _step4_attr_null(ir, key: str, cur) -> bool:
    """bitino.attrs null (step 4): removes count_include_pad when the importer's value is its default 0."""
    if key in _ATTR_DEFAULTS_STEP4 and not isinstance(cur, bool) and isinstance(cur, (int, float)) \
            and float(cur) == float(_ATTR_DEFAULTS_STEP4[key]):
        del ir.attrs[key]
        return True
    return False


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


def step4_import_options(model) -> dict:
    """The pattern importer's options recorded in the file (bitino.import {"mapping": "patterns", ...}, step 4), {}
    without them."""
    raw = next((p.value for p in getattr(model, "metadata_props", ()) if p.key == IMPORT_KEY), None)
    if raw is None:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict) or obj.get("mapping") != "patterns":
        return {}
    opts = {k: v for k, v in obj.items() if k != "mapping"}
    bad = sorted(set(opts) - set(PATTERN_IMPORT_OPTIONS))
    if bad:
        raise ValueError(f"{IMPORT_KEY}: unknown pattern-import option(s) {bad}")
    return opts


def _step4_bit_outputs(model, cli: str | None) -> str:
    """The bit-output layout: the file's bitino.import bit_outputs (step 4) or the argument (default set). Both given
    and different is an error: the file determines its .bitino."""
    meta = step4_import_options(model).get("bit_outputs")
    if meta is None and has_pattern_import(model):
        meta = PATTERN_IMPORT_OPTIONS["bit_outputs"][1]     # step 6: the file determines it; absent = the default
    if meta is None:
        return cli or "set"
    if meta not in BIT_OUTPUTS:
        raise ValueError(f"{IMPORT_KEY} bit_outputs {meta!r} (one of {BIT_OUTPUTS})")
    if cli is not None and cli != meta:
        raise ValueError(f"bit_outputs {cli} contradicts the file's {IMPORT_KEY} bit_outputs {meta}")
    return meta


def has_pattern_import(model) -> bool:
    """The file records its pattern-import layout (bitino.import {"mapping": "patterns", ...}: steps 1, 1b, 2 since
    step 6, and step 4)."""
    raw = next((p.value for p in getattr(model, "metadata_props", ()) if p.key == IMPORT_KEY), None)
    try:
        obj = json.loads(raw) if raw is not None else None
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("mapping") == "patterns"


def _step4_resolve_args(args, model, explicit: set) -> dict:
    """Apply the file's pattern-import options (bitino.import, step 4; steps 1, 1b, 2 since step 6) to the parsed
    arguments. With bitino.import the file determines the import: an option it does not list has the importer's default,
    and an explicit argument that differs from the file's value (or from that default) is an error. Returns the options
    the file gave."""
    opts = step4_import_options(model)
    if not has_pattern_import(model):
        return opts
    for k, (attr, default) in PATTERN_IMPORT_OPTIONS.items():
        v = opts.get(k, default)
        if attr in explicit and getattr(args, attr) != v:
            raise SystemExit(f"--{attr.replace('_', '-')} {getattr(args, attr)} contradicts the file's {IMPORT_KEY} {k} {v}"
                             f"{'' if k in opts else ' (the default: the option is not listed)'}")
        setattr(args, attr, v)
    return opts


SIMPLE_DOMAINS = ("float32", "uint8", "unit8")


def file_input_domain(model) -> str | None:
    """The file's bitino.input_domain (model metadata, docs/qonnx-profile.md § 6), None without the key."""
    return next((p.value for p in getattr(model, "metadata_props", ()) if p.key == "bitino.input_domain"), None)


def resolve_input_domain(model, cli: str | None, groups: bool) -> str:
    """The input domain of the import (step 6: the file's bitino.input_domain). An explicit --input-domain must equal
    it. A pattern-mapped file declares one of SIMPLE_DOMAINS; a group-mapped file may declare a per-input JSON object or
    ids:<V> (read by the group importer; the pattern domain is then float32 and no argument applies)."""
    fd = file_input_domain(model)
    if fd is None:
        return cli or "float32"
    if fd in SIMPLE_DOMAINS:
        if cli is not None and cli != fd:
            raise SystemExit(f"--input-domain {cli} contradicts the file's bitino.input_domain {fd}")
        return fd
    if not groups:
        raise SystemExit(f"bitino.input_domain {fd!r}: a pattern-mapped file declares one of {SIMPLE_DOMAINS}")
    if cli is not None:
        raise SystemExit(f"--input-domain {cli}: the group-mapped file declares bitino.input_domain {fd!r}")
    return "float32"


def _step4_threshold_bit_outputs(g: Graph) -> dict:
    """bit_outputs threshold (step 4, the OpenVINO layout of tools/convert_omz.py): a BConv2D without epilogue whose
    every consumer binarizes it with ``gt`` and the same threshold t emits bits, post_add = -(floor(t) + 1)
    (openvino_ir.fuse_threshold_into_bit_output: exact on the integer accumulator, float32 and float64), then the
    consumers' thresholds are pruned."""
    from .adapters.openvino_ir import fuse_threshold_into_bit_output
    return {"fuse_threshold_into_bit_output": fuse_threshold_into_bit_output(g),
            "prune_unused_constants (after the fuse)": passes.prune_unused_constants(g)}


def order_constants_by_first_use(g: Graph) -> bool:
    """Reorder the constants table (= the data section) by first use: nodes in order, inputs in order. The layout of
    the PyTorch converters' files; the walk order of the importer is the layout of the FINN / Brevitas ones."""
    order = []
    seen = set()
    for n in g.nodes:
        for i in n.inputs:
            if i and i in g.constants and i not in seen:
                seen.add(i)
                order.append(i)
    rest = [k for k in g.constants if k not in seen]
    new = {k: g.constants[k] for k in order + rest}
    changed = list(new) != list(g.constants)
    g.constants = new
    return changed


CONSTANT_ORDERS = ("walk", "first-use")


def apply_const_order(model, g: Graph) -> bool | None:
    """bitino.const_order (model metadata, docs/qonnx-profile.md § 7): the data-section order of the final constants
    (after the passes). The list and the constants the importer built must be equal as sets. None without the key."""
    raw = next((p.value for p in getattr(model, "metadata_props", ()) if p.key == CONST_ORDER_KEY), None)
    if raw is None:
        return None
    order = json.loads(raw)
    if not isinstance(order, list) or len(set(order)) != len(order) or set(order) != set(g.constants):
        extra = sorted(set(order) - set(g.constants))[:5] if isinstance(order, list) else []
        missing = sorted(set(g.constants) - set(order))[:5] if isinstance(order, list) else []
        raise QonnxImportError(type("M", (), {"name": CONST_ORDER_KEY, "op_type": "model"})(),
                               f"{CONST_ORDER_KEY} and the imported constants differ (not built: {extra}, not listed: {missing})")
    changed = order != list(g.constants)
    g.constants = {k: g.constants[k] for k in order}
    return changed


def import_model(model, domain: Domain | None = None, naming: str = "index", fold: bool = True,
                 affine: str = "fold", constant_order: str = "walk", flatten: str = "transpose",
                 bit_outputs: str | None = None) -> tuple[Graph, dict]:
    """Returns (graph, info). info: rules applied, binarizer map, checks, precision, margins, passes.

    affine: "fold" (default) accumulates per-channel affine nodes in float64 into the next epilogue, integer
    threshold or input threshold (one float32 rounding); "keep" gives every affine source node (BatchNormalization,
    Add / Sub / Mul / Div by a per-channel constant) its own ChannelAffine at its position, closes a linear operator's
    epilogue at its weight scale, and binarizes at 0: the float32 rounding steps of the source, which is the layout
    of the legacy PyTorch converters (passes.fold_affine_into_conv then folds what they folded). constant_order:
    "walk" (default) or "first-use" (order_constants_by_first_use, applied before the passes, as the legacy
    converters' insertion order); a file's bitino.const_order then fixes the final data-section order. bit_outputs:
    "set" (default) gives a BConv2D feeding only ge BConv2D nodes a bit output (set_bit_outputs); "none" keeps every
    f32 output (the Larq converters' files); "threshold" fuses gt-threshold binarizers into bit outputs (the OpenVINO
    converter's files, step 4). A file's bitino.import {"mapping": "patterns", ...} (step 4) gives these options."""
    from . import qonnx_groups
    if qonnx_groups.import_mode(model):          # step 3: one ONNX group per IR node (model metadata bitino.import)
        return qonnx_groups.import_groups(model)
    if constant_order not in CONSTANT_ORDERS:
        raise ValueError(f"constant_order must be one of {CONSTANT_ORDERS}")
    bit_outputs = _step4_bit_outputs(model, bit_outputs)
    if bit_outputs not in BIT_OUTPUTS:
        raise ValueError(f"bit_outputs must be one of {BIT_OUTPUTS}")
    imp = Importer(model, domain, naming, affine=affine, flatten=flatten)
    g = imp.run()
    n_attrs = imp.apply_group_attrs()
    n_consts = imp.apply_group_consts()
    n_layouts = imp.apply_group_const_layouts()     # step 4
    for r in imp.binarizers:                        # step 4: a threshold renamed by bitino.consts
        if r.get("ir_threshold") in getattr(imp, "const_renames", {}):
            r["ir_threshold"] = imp.const_renames[r["ir_threshold"]]
    if n_layouts:
        imp.notes.append(f"bitino.const_layouts: {n_layouts} constant layout tag(s) in the exporter's form")
    if n_consts:
        imp.notes.append(f"bitino.consts: {n_consts} constant(s) renamed")
    if n_attrs:
        imp.notes.append(f"bitino.attrs: {n_attrs} attribute(s) in the exporter's JSON form")
    reordered = order_constants_by_first_use(g) if constant_order == "first-use" else None
    # flip tracing reads each binarizer's IR carrier tensor; a pass may fold its producer away (e.g. a BatchNorm into
    # the preceding conv) and rewire the binary consumer: remember the consumer and re-read its input afterwards
    users = {}
    for r in imp.binarizers:
        c = next((n for n in g.nodes if n.op in _BINARY_LIN and n.inputs and n.inputs[0] == r["ir_tensor"]), None)
        if c is not None:
            users[id(r)] = c.name
    applied = passes.optimize(g, fold=fold)
    by_name = {n.name: n for n in g.nodes}
    produced = {o for n in g.nodes for o in n.outputs} | {i["name"] for i in g.inputs}
    for r in imp.binarizers:
        if r["ir_tensor"] not in produced and users.get(id(r)) in by_name:
            r["ir_tensor"] = by_name[users[id(r)]].inputs[0]
    if bit_outputs == "threshold":
        applied.update(_step4_threshold_bit_outputs(g))
        for r in imp.binarizers:       # a fused gt threshold: the decision is the producer's bit output (±1), ge 0
            if r.get("ir_threshold") and r["ir_threshold"] not in g.constants:
                r["ir_threshold"], r["ir_mode"] = None, "ge"
    else:
        applied["set_bit_outputs"] = set_bit_outputs(g) if bit_outputs == "set" else "off (--bit-outputs none)"
    if reordered is not None:
        applied["order_constants_by_first_use (before the passes)"] = int(reordered)
    co = apply_const_order(model, g)
    if co is not None:
        applied["bitino.const_order (after the passes)"] = int(co)
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
    if rec.get("ir_polarity") is not None:      # step 4: pool-before-BN channels decided as -(n >= T + 1)
        d = d * np.asarray(rec["ir_polarity"], d.dtype)
    if d.ndim == 4:
        d = d.transpose(0, 3, 1, 2)
    return d.reshape(d.shape[0], -1)


def _src_decision(pre: np.ndarray, thr, mode: str) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pre, np.float64) - np.asarray(thr, np.float64)
    d = np.where(p >= 0, 1, -1) if mode == "ge" else np.where(p > 0, 1, -1) if mode == "gt" else np.sign(p).astype(int)
    return d.reshape(d.shape[0], -1), p.reshape(p.shape[0], -1)


def _out_flat(a: np.ndarray) -> np.ndarray:
    """One sample's ONNX output flattened in the IR's element order: an image output [1, C, H, W] (NCHW) in NHWC
    order (the IR output layout), anything else as is."""
    return (a.transpose(0, 2, 3, 1) if a.ndim == 4 else a).reshape(-1)


def _logits_of(model, x_nhwc, logits: str, ft) -> np.ndarray:
    ev = OnnxF64(model)
    out = []
    for i in range(len(x_nhwc)):
        xi = np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2)) if x_nhwc.ndim == 4 else x_nhwc[i:i + 1]
        out.append(_out_flat(np.asarray(ev.run(xi, ft, outputs=[logits])[logits], ft)))
    return np.stack(out)


def verify(model, g: Graph, info: dict, x_nhwc: np.ndarray, labels: np.ndarray, out_dir: str, raw_model=None) -> dict:
    """Level-3 checks of the QONNX route on the imported graph: the float32 reference (step 6, docs/qonnx-profile.md
    § 8: one onnxruntime session with ORT_DISABLE_ALL, the QONNX quantizers replaced by their definitions; qonnx
    execute_onnx where a quantizer has no standard-op replacement, and as the cross-check on the first samples) vs
    OnnxF64 (float64 reference) vs the numpy executor of the written file (float32 / float64), plus the first
    binarization flip of every sample that deviates (source pre-activation |x - t| at the flipped positions)."""
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
    sess = None
    try:                       # step 6: the profile's fp32 reference, one ORT_DISABLE_ALL session
        from .qonnx_groups import ort_session
        sess = ort_session(model, extra_outputs=sorted({r["pre"] for r in bins} - {logits}) + [logits])
        sess_out = [o.name for o in sess.get_outputs()]
    except Exception as e:  # pragma: no cover - a quantizer without a standard-op definition (runtime IntQuant)
        rep["ort_reference_unavailable"] = f"{type(e).__name__}: {str(e)[:200]}"
    n_x = 4 if sess is not None else n            # execute_onnx: the cross-check on the first samples
    xck = []
    ref32, ref64, own32, ir32, ir64, flips = [], [], [], [], [], []
    def src_dec(env):          # compact per-binarizer decisions + |pre| (the full environment is freed right after)
        out = []
        for r in bins:
            dsrc, pv = _src_decision(env[r["pre"]], src_thr.get(r["onnx_node"], 0.0), r["mode"])
            out.append((dsrc.astype(np.int8), np.abs(pv).astype(np.float32)))
        return out

    def ir_dec(env):
        return [_ir_decision(env, r, g).astype(np.int8) for r in bins]

    for i in range(n):
        # one full evaluation alive at a time (peak memory: one environment, not four)
        xi = np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2)) if x_nhwc.ndim == 4 else x_nhwc[i:i + 1]
        e64 = ev.run(xi, np.float64, keep={r["pre"] for r in bins}, outputs=[logits])
        ref64.append(_out_flat(np.asarray(e64[logits], np.float64)))
        s64 = src_dec(e64)
        del e64
        own32.append(_out_flat(np.asarray(ev.run(xi, np.float32, outputs=[logits])[logits], np.float32)))
        s32 = None
        if sess is not None:
            ctx = dict(zip(sess_out, sess.run(None, {ev.input_name: xi.astype(np.float32)})))
            ref32.append(_out_flat(np.asarray(ctx[logits], np.float32)))
            s32 = src_dec(ctx)
            del ctx
        if have_qonnx and i < n_x:
            ctx = execute_onnx(mw, {ev.input_name: xi.astype(np.float32)}, return_full_exec_context=True)
            if sess is None:
                ref32.append(_out_flat(np.asarray(ctx[logits], np.float32)))
                s32 = src_dec(ctx)
            else:
                xck.append(_out_flat(np.asarray(ctx[logits], np.float32)))
            del ctx
        env32 = executor.run(g, {"input": x_nhwc[i:i + 1]}, keep_intermediates=True)
        ir32.append(env32[g.outputs[0]].reshape(-1))
        d32 = ir_dec(env32)
        del env32
        env64 = executor.run(g, {"input": x_nhwc[i:i + 1]}, keep_intermediates=True, float_dtype=np.float64)
        ir64.append(env64[g.outputs[0]].reshape(-1))
        d64 = ir_dec(env64)
        del env64
        rec = {"sample": i}
        for tag, dirs, src in (("ir_f64_vs_source_f64", d64, s64), ("ir_f32_vs_source_f32", d32, s32)):
            if src is None:
                continue
            for r, dir_, (dsrc, apv) in zip(bins, dirs, src):
                if dsrc.shape != dir_.shape:
                    raise AssertionError(f"binarizer {r['onnx_node']}: source {dsrc.shape} vs IR {dir_.shape}")
                d = np.where(dsrc[0] != dir_[0])[0]
                if d.size:
                    rec[tag] = {"onnx_node": r["onnx_node"], "n_flips": int(d.size), "max_abs_pre_source": float(apv[0, d].max()),
                                "typical_abs_pre_source": float(np.median(apv[0]))}
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
    sem["per_sample_max_abs"] = [float(v) for v in per]
    ir64.tofile(os.path.join(out_dir, "numpy_logits_f64.f64"))
    floor = compare_outputs(ref64, ref32, labels)
    vs64 = compare_outputs(ref64, ir32, labels)
    slack = max(1, int(round(0.01 * n)))
    rep.update({
        "n_samples": int(n),
        "reference_f32": ("onnxruntime ORT_DISABLE_ALL session (float32; bitino.qonnx_groups.ort_session)" if sess is not None else
                          "qonnx.core.onnx_exec.execute_onnx (float32)" if have_qonnx else "bitino.onnx_f64.OnnxF64 (float32)"),
        "execute_onnx_cross_check_f32": ({"samples": len(xck), "max_abs_vs_reference_f32": float(np.max(np.abs(np.stack(xck) - ref32[:len(xck)]))),
                                          "bit_identical": bool(np.array_equal(np.stack(xck).view(np.uint32), ref32[:len(xck)].view(np.uint32)))}
                                         if xck else None),
        "reference_f64": "bitino.onnx_f64.OnnxF64 (float64)",
        "reference_f32_vs_f64": floor,
        "onnxf64_in_f32_vs_reference_f32": compare_outputs(ref32, own32, labels),
        "semantic_check_ir_f64_vs_reference_f64": sem,
        "numpy_f32_vs_reference_f32": compare_outputs(ref32, ir32, labels),
        "numpy_f32_vs_reference_f64": vs64,
        "binarization_flips_vs_source": {"n_binarizers": len(bins), "n_samples_with_flips": len(flips), "samples": flips},
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
    ap.add_argument("--input-domain", default=None, choices=("float32", "uint8", "unit8"),
                    help="values of the graph input: any float32 | float32(k) | float32(k)/255 for uint8 pixels k "
                         "(default: the file's bitino.input_domain, else float32; an argument that contradicts the file "
                         "is an error)")
    ap.add_argument("--cleanup", choices=("none", "qonnx"), default=None,
                    help="qonnx: run qonnx's cleanup_model (InferShapes, FoldConstants keeping quant ops, ...) first")
    ap.add_argument("--naming", choices=NAMINGS, default=None,
                    help="index: conv{i}/dense{i} (FINN convention); onnx: ONNX node names; exporter: the exporter's names "
                         "recorded in node metadata_props 'bitino.name' (tools/export_torch_qonnx.py: torch module paths)")
    ap.add_argument("--affine", choices=AFFINE_MODES, default=None,
                    help="fold: per-channel affine nodes accumulate in float64 into epilogues / thresholds (default); "
                         "keep: one ChannelAffine per affine node, the source's float32 rounding steps (legacy PyTorch "
                         "layout, folded afterwards by the passes)")
    ap.add_argument("--flatten-dense", choices=FLATTEN_MODES, default=None,
                    help="flatten of an NCHW tensor feeding a Dense: transpose (default: Transpose to NCHW order) or "
                         "permute (Reshape the NHWC tensor and permute the Dense weight columns: the legacy PyTorch layout)")
    ap.add_argument("--constant-order", choices=CONSTANT_ORDERS, default=None,
                    help="data section order: importer walk (default; FINN/Brevitas layout) or first use (PyTorch layout)")
    ap.add_argument("--bit-outputs", choices=BIT_OUTPUTS, default=None,
                    help="set: BConv2D nodes feeding only ge BConv2D nodes output bits (default); none: keep f32 outputs "
                         "(the Larq converters' layout); threshold: a BConv2D feeding only gt-threshold BConv2D nodes "
                         "outputs bits (the OpenVINO converter's layout). A file's bitino.bit_outputs metadata gives the "
                         "value; an argument that contradicts it is an error")
    ap.add_argument("--no-fold", action="store_true",
                    help="do not run passes.fold_affine_into_conv / fold_relu_into_producer: the derived *_unfused.bitino "
                         "of a pattern-mapped file (step 6; a derivation, not a layout option: the published file is the "
                         "import without it)")
    ap.add_argument("--exporter", default="", help="exporter and version, e.g. 'brevitas 0.11.0 export_qonnx (torch 2.4.1)'")
    ap.add_argument("--source-json", help="exporter provenance JSON (checkpoint SHA-256, framework versions, source logits files)")
    ap.add_argument("--bench", help="host bitino_bench: run the C++ runtime (tools/validate_model.py, exact stages)")
    ap.add_argument("--stages", default="reference,optimized")
    ap.add_argument("--num-classes", type=int, default=0)
    ap.add_argument("--file-name", default="model.bitino")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    explicit = {a for a, _ in PATTERN_IMPORT_OPTIONS.values() if getattr(args, a) is not None}
    for a, d in PATTERN_IMPORT_OPTIONS.values():
        if getattr(args, a) is None and a != "bit_outputs":
            setattr(args, a, d)
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
    from . import qonnx_groups
    groups = bool(qonnx_groups.import_mode(model))
    args.input_domain = resolve_input_domain(model, args.input_domain, groups)   # step 6: the file's input domain
    if groups:                                   # step 3: the file says how it is imported (bitino.import)
        bad = sorted(f"--{a.replace('_', '-')}" for a in explicit) + (["--no-fold"] if args.no_fold else [])
        if bad:                                  # step 6: a group-mapped file determines its import alone
            raise SystemExit(f"{bad}: the file is group-mapped ({IMPORT_KEY} mapping groups: one ONNX group per IR "
                             "node, no graph pass); no import argument applies")
        args.cleanup = "none"
        return qonnx_groups.main_groups(args, model, prov)
    file_opts = _step4_resolve_args(args, model, explicit)      # steps 1-2 / 4: the file's bitino.import options
    clean_info = {"cleanup": args.cleanup, "import_options_from_file": file_opts or None}
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
        g, info = import_model(model, domain, args.naming, fold=not args.no_fold, affine=args.affine,
                               constant_order=args.constant_order, flatten=args.flatten_dense, bit_outputs=args.bit_outputs)
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
                       "naming": args.naming, "affine": args.affine, "constant_order": args.constant_order,
                       "flatten_dense": args.flatten_dense,
                       "preprocessing_cleanup": clean_info, "rules": info["rules"], "passes": info["passes"],
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
              f"numpy f32 vs fp32 reference max_abs {v['numpy_f32_vs_reference_f32']['max_abs_error']:.3e} | acceptance {v['acceptance']['passed']}")
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
