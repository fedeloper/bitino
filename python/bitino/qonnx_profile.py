"""Checker of the Bitino QONNX profile v1 (docs/qonnx-profile.md).

    .cache/venv-qonnx/bin/python -I tools/check_qonnx.py MODEL.onnx [MODEL2.onnx ...] [--json OUT] [--strict]
    bitino check-qonnx MODEL.onnx ...            (same arguments; CLI subcommand)

Validates one ONNX file against the profile and prints a report: container (IR version, opsets, domains, no local
functions / external data / custom domains), operators (the profile's allow-list, checked against the declared
opset), attributes that the profile fixes (explicit pads, no auto_pad / ceil_mode, IntQuant rounding and zero point,
Gemm alpha / beta, Cast targets), the binarizer and binary-padding conventions (explicit Pad of a binarized tensor
with a value in {-1, 0, +1}, no implicit Conv padding of a binarized tensor, unit-scale binary weights), the required
model metadata (``bitino.*`` metadata_props, well-formed), the naming / order metadata that carries byte identity
(``bitino.name`` on every activation-path node, ``bitino.order``, ``bitino.extra_names``, ``bitino.consts``,
``bitino.attrs``) and static shape inference.

The checker validates the *form* of a file. Whether the importer maps it onto the published ``.bitino`` byte for
byte is the importer's acceptance test (docs/qonnx-profile.md § Verification contract), not this module's.

Needs ``onnx`` (``.cache/venv-qonnx``); qonnx is not needed. This module does not import ``bitino.qonnx_import``.
Exit codes of ``main``: 0 every file passes, 1 a file has errors (or warnings with ``--strict``), 2 usage error or
unreadable file.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import re
import sys

import numpy as np

PROFILE_ID = "bitino-qonnx/1"
QONNX_DOMAIN = "qonnx.custom_op.general"
ONNX_DOMAINS = ("", "ai.onnx")
OPSET_MIN, OPSET_MAX = 17, 21
MIN_IR_VERSION = 10           # NodeProto.metadata_props (bitino.name, ...) exists from IR version 10 (onnx 1.16)

# ------------------------------------------------------------------------------------------------ operator sets
# The allow-list of docs/qonnx-profile.md § Allowed operators (one source of truth: the doc quotes these sets).
ONNX_OPS = {
    "linear": {"Conv", "ConvInteger", "MatMul", "MatMulInteger", "Gemm"},
    "elementwise": {"Add", "Sub", "Mul", "Div", "Neg", "Abs", "Relu", "PRelu", "LeakyRelu", "Clip", "Sigmoid",
                    "Tanh", "Gelu", "Erf", "Sqrt", "Pow", "Exp", "Log", "Reciprocal", "Max", "Min", "Sign", "Round",
                    "Floor", "Ceil"},
    "compare": {"Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal", "Not", "And", "Or", "Xor", "Where",
                "IsNaN"},
    "integer": {"Cast", "BitShift", "QuantizeLinear", "DequantizeLinear"},
    "pool_reduce": {"MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "ReduceMean", "ReduceSum",
                    "ReduceMax"},
    "shape": {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Concat", "Slice", "Pad", "Gather",
              "Identity", "Shape", "Constant"},
    "normalization": {"LayerNormalization", "Softmax"},
    "image_features": set(),     # Q1 decided: the FFT feature's DFT is a DOUBLE MatMul on cos / sin tables
}
ALLOWED_ONNX_OPS = set().union(*ONNX_OPS.values())
QONNX_OPS = {"BipolarQuant", "IntQuant"}

# Operators that appear in today's exports / releases, with the canonical replacement (error messages).
REJECTED_OPS = {
    "BatchNormalization": "fold in float64 into Mul / Add (scale, shift as Cast(DOUBLE initializer, to=FLOAT)): "
                          "ONNX BatchNormalization stores epsilon as float32",
    "Quant": "rename to IntQuant (QONNX >= 0.4 name, same semantics)",
    "MultiThreshold": "FINN-internal op, not QONNX: write Where(GreaterOrEqual(x, t), 1, -1) (one threshold) or "
                      "out_scale * ReduceSum(Cast(GreaterOrEqual(Unsqueeze(x), T))) + out_bias (several)",
    "TopK": "the canonical file outputs what the .bitino outputs (the logits); drop the TopK",
    "DFT": "write the double-precision DFT as MatMul on DOUBLE cos / sin tables computed in float64 (docs/qonnx-profile.md "
           "§ 4.5, Q1 decided 2026-11-20; float contract, not bit-exact)",
    "Trunc": "QONNX Trunc has no Bitino operator in profile v1",
    "FloatQuant": "QONNX FloatQuant has no Bitino operator in profile v1",
    "Dropout": "remove (inference graph)",
    "Resize": "no Bitino operator in profile v1",
}

# Ops through which a binarized tensor stays binarized: data movement, and MaxPool (max commutes with a
# non-decreasing binarizer: max(bin(x)) = bin(max(x)); out-of-bounds taps are ignored in ONNX and in Bitino).
_DATA_MOVEMENT = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Slice", "Gather", "Identity", "Pad",
                  "Concat", "Cast", "MaxPool"}
_POOLS = {"MaxPool", "AveragePool"}
_PADDED = {"Conv", "ConvInteger", "MaxPool", "AveragePool"}

# Graph-input element types the runtime has, and the tensor element types allowed anywhere in a file.
_INPUT_TYPES = {"FLOAT", "INT32", "INT64"}
_TENSOR_TYPES = {"FLOAT", "DOUBLE", "INT8", "UINT8", "INT16", "INT32", "INT64", "UINT32", "UINT64", "BOOL"}

# ------------------------------------------------------------------------------------------------ metadata
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
VARIANT = re.compile(r"^(float|int8_full(_[a-z0-9]+)?)$")
INPUT_DOMAINS = ("float32", "unit8", "uint8")
IDS_DOMAIN = re.compile(r"^ids:[1-9][0-9]*$")
CODE_REF = re.compile(r"^\S+@[0-9a-f]{7,40}$")
CONST_KEY = re.compile(r"^(?:(?P<node>.+)#)?(?P<slot>[0-9]+)$")
ORDER = re.compile(r"^(0|[1-9][0-9]*)$")
_BAD_NAME = re.compile(r"[\x00-\x1f\x7f]")

REQUIRED_MODEL_KEYS = (
    "bitino.profile", "bitino.model_id", "bitino.variant", "bitino.layout", "bitino.input_domain",
    "bitino.preprocessing", "bitino.labels", "bitino.source.framework", "bitino.source.framework_version",
    "bitino.source.checkpoint_sha256", "bitino.weights_license", "bitino.exporter",
)
OPTIONAL_MODEL_KEYS = ("bitino.calibration", "bitino.source.code", "bitino.const_order", "bitino.task",
                       "bitino.notes", "bitino.ir_unique_counter", "bitino.import", "bitino.unused_constants")
NODE_KEYS = ("bitino.name", "bitino.order", "bitino.extra_names", "bitino.consts", "bitino.attrs",
             "bitino.extra_attrs", "bitino.float_layer", "bitino.ir", "bitino.const_layouts")
# step 4 (§ 7 bitino.const_layouts): the informational layout tag of a group constant, {"<input index>": tag}; only
# tags that name the same storage (docs/model-format.md: a rank-2 bit tensor packed along its last axis, tagged
# "OI_bit" or untagged) are allowed
CONST_LAYOUT_TAGS = ("", "OI_bit")
# step 4 (§ 6 bitino.import, mapping "patterns"): the pattern importer's layout options, recorded in the file
PATTERN_IMPORT_OPTIONS = {"naming": ("index", "onnx", "exporter"), "affine": ("fold", "keep"),
                          "constant_order": ("walk", "first-use"), "cleanup": ("none", "qonnx"),
                          "bit_outputs": ("set", "none", "threshold"), "flatten_dense": ("transpose", "permute")}
FLOAT_LAYER_KEY = "bitino.float_layer"   # § 7: a declared float layer on a binarized tensor (value "1")
IMPORT_MAPPINGS = ("groups",)     # bitino.import {"mapping": ...}: step 3, one ONNX group per IR node (§ 7)


# ------------------------------------------------------------------------------------------------ report
@dataclasses.dataclass
class Finding:
    level: str          # error | warning | info
    code: str           # short id, e.g. E-OP
    message: str
    nodes: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Report:
    path: str = ""
    findings: list = dataclasses.field(default_factory=list)
    summary: dict = dataclasses.field(default_factory=dict)

    def add(self, level: str, code: str, message: str, nodes=()) -> None:
        self.findings.append(Finding(level, code, message, [n for n in nodes]))

    def of(self, level: str) -> list:
        return [f for f in self.findings if f.level == level]

    @property
    def errors(self) -> list:
        return self.of("error")

    @property
    def warnings(self) -> list:
        return self.of("warning")

    def passed(self, strict: bool = False) -> bool:
        return not self.errors and not (strict and self.warnings)

    def codes(self, level: str | None = None) -> set:
        return {f.code for f in self.findings if level is None or f.level == level}

    def to_dict(self) -> dict:
        return {"path": self.path, "profile": PROFILE_ID, "passed": self.passed(), "summary": self.summary,
                "findings": [dataclasses.asdict(f) for f in self.findings]}

    def to_text(self, strict: bool = False) -> str:
        s = self.summary
        lines = [f"check-qonnx {self.path or '<model>'}  (profile {PROFILE_ID})"]
        if s:
            ops = ", ".join(f"{k} {v}" for k, v in sorted(s.get("op_counts", {}).items(), key=lambda kv: (-kv[1], kv[0])))
            lines.append(f"  ONNX IR {s.get('ir_version')}, opsets {s.get('opsets')}, {s.get('nodes')} nodes, "
                         f"{s.get('initializers')} initializers" + (f", sha256 {s['sha256'][:16]}" if s.get("sha256") else ""))
            lines.append(f"  ops: {ops}")
            lines.append(f"  binarizers {s.get('binarizers', {})}, binary layers {s.get('binary_layers', 0)}, "
                         f"activation-path nodes {s.get('activation_nodes', 0)}, named {s.get('named_nodes', 0)}, "
                         f"groups {s.get('groups', 0)}")
        for level, title in (("error", "ERRORS"), ("warning", "WARNINGS"), ("info", "INFO")):
            fs = self.of(level)
            if not fs:
                continue
            lines.append(f"{title} ({len(fs)})")
            for f in fs:
                where = ""
                if f.nodes:
                    shown = ", ".join(str(n) for n in f.nodes[:5])
                    where = f"  [{shown}{', ...' if len(f.nodes) > 5 else ''}]"
                lines.append(f"  [{f.code}] {f.message}{where}")
        ok = self.passed(strict)
        lines.append(f"RESULT: {'PASS' if ok else 'FAIL'} ({len(self.errors)} errors, {len(self.warnings)} warnings)")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ helpers
def _onnx():
    import onnx  # noqa: F401  (lazy: the bitino package does not depend on onnx)

    return onnx


def _attrs(node) -> dict:
    onnx = _onnx()
    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode()
        out[a.name] = v
    return out


def _props(entries) -> tuple[dict, list]:
    """metadata_props -> dict, plus the duplicated keys."""
    d, dup = {}, []
    for p in entries:
        if p.key in d:
            dup.append(p.key)
        d[p.key] = p.value
    return d, dup


def _json(value: str):
    try:
        return json.loads(value), None
    except (TypeError, ValueError) as e:
        return None, str(e)


def _nname(node, i: int) -> str:
    return node.name or f"#{i}:{node.op_type}"


def _dtype_name(t: int) -> str:
    onnx = _onnx()
    try:
        return onnx.TensorProto.DataType.Name(t)
    except ValueError:
        return str(t)


class _Graph:
    """Producer / consumer maps, constant values and the activation path of one graph."""

    def __init__(self, model):
        onnx = _onnx()
        from onnx import numpy_helper

        g = model.graph
        self.g = g
        self.inits = {t.name: t for t in g.initializer}
        self.inputs = [i for i in g.input if i.name not in self.inits]
        self.producer = {}
        self.consumers: dict = {}
        for i, nd in enumerate(g.node):
            for o in nd.output:
                if o:
                    self.producer[o] = (i, nd)
            for x in nd.input:
                if x:
                    self.consumers.setdefault(x, []).append((i, nd))
        self._np = numpy_helper
        self._onnx = onnx
        # activation path: tensors depending on a graph input's *values* (Shape outputs are static here)
        self.dynamic = {i.name for i in self.inputs}
        self.order_errors = []
        known = set(self.inits) | {i.name for i in g.input} | {""}
        for i, nd in enumerate(g.node):
            for x in nd.input:
                if x not in known:
                    self.order_errors.append(_nname(nd, i))
                    break
            known.update(nd.output)
            if nd.op_type != "Shape" and any(x in self.dynamic for x in nd.input if x):
                self.dynamic.update(o for o in nd.output if o)

    def is_dynamic_node(self, nd) -> bool:
        return any(o in self.dynamic for o in nd.output if o)

    def const(self, name: str):
        """Value of an initializer or Constant-node output (one level), else None."""
        if name in self.inits:
            return self._np.to_array(self.inits[name])
        p = self.producer.get(name)
        if p and p[1].op_type == "Constant":
            a = _attrs(p[1])
            if "value" in a:
                return self._np.to_array(a["value"])
            for k in ("value_float", "value_int"):
                if k in a:
                    return np.asarray(a[k])
            for k in ("value_floats", "value_ints"):
                if k in a:
                    return np.asarray(a[k])
        if p and p[1].op_type == "Cast" and p[1].input:   # float64 carrier: Cast(DOUBLE initializer, to=FLOAT)
            v = self.const(p[1].input[0])
            if v is None:
                return None
            try:
                return v.astype(self._onnx.helper.tensor_dtype_to_np_dtype(_attrs(p[1]).get("to", 1)))
            except Exception:
                return None
        return None


# ------------------------------------------------------------------------------------------------ the checks
def _check_container(model, rep: Report) -> dict:
    onnx = _onnx()
    opsets = {}
    for o in model.opset_import:
        dom = o.domain or ""
        if dom in opsets:
            rep.add("error", "E-OPSET", f"domain {dom or 'ai.onnx'!r} imported twice")
        opsets[dom] = o.version
    if model.ir_version < MIN_IR_VERSION:
        rep.add("error", "E-IR", f"ONNX IR version {model.ir_version} < {MIN_IR_VERSION}: node metadata_props "
                                 "(bitino.name, ...) need IR >= 10 (onnx >= 1.16)")
    onnx_v = [opsets[d] for d in ONNX_DOMAINS if d in opsets]
    if len(onnx_v) != 1:
        rep.add("error", "E-OPSET", "exactly one ai.onnx opset import is required")
        onnx_ver = max(onnx_v) if onnx_v else OPSET_MIN
    else:
        onnx_ver = onnx_v[0]
        if not OPSET_MIN <= onnx_ver <= OPSET_MAX:
            rep.add("error", "E-OPSET", f"ai.onnx opset {onnx_ver} outside {OPSET_MIN}..{OPSET_MAX}")
    for dom, v in opsets.items():
        if dom in ONNX_DOMAINS:
            continue
        if dom == QONNX_DOMAIN:
            if v != 1:
                rep.add("error", "E-OPSET", f"{QONNX_DOMAIN} imported with version {v}, profile v1 needs 1")
        else:
            rep.add("error", "E-DOMAIN", f"opset import of domain {dom!r}: only ai.onnx and {QONNX_DOMAIN} are allowed "
                                         "(no private / custom / legacy domains)")
    if len(model.functions):
        rep.add("error", "E-FUNCTIONS", f"{len(model.functions)} model-local function(s): not allowed (a function is a "
                                        "custom operator)", [f"{f.domain}:{f.name}" for f in model.functions])
    g = model.graph
    ext = [t.name for t in g.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    if ext:
        rep.add("error", "E-EXTERNAL", f"{len(ext)} initializer(s) with external data: the canonical file is one "
                                       "self-contained .onnx", ext)
    if len(g.sparse_initializer):
        rep.add("error", "E-SPARSE", "sparse initializers are not allowed")
    if not model.producer_name or not model.producer_version:
        rep.add("warning", "W-PRODUCER", "producer_name / producer_version are empty (provenance)")
    bad_t = sorted({_dtype_name(t.data_type) for t in g.initializer if _dtype_name(t.data_type) not in _TENSOR_TYPES})
    if bad_t:
        rep.add("error", "E-DTYPE", f"initializer element types {bad_t} are not allowed (allowed: {sorted(_TENSOR_TYPES)})")
    return {"opsets": opsets, "onnx_opset": onnx_ver}


def _check_io(G: _Graph, rep: Report) -> None:
    g = G.g
    init_inputs = [i.name for i in g.input if i.name in G.inits]
    if init_inputs:
        rep.add("error", "E-INPUT-INIT", f"{len(init_inputs)} initializer(s) also listed as graph inputs: graph inputs "
                                         "must be the runtime inputs only", init_inputs)
    if not G.inputs:
        rep.add("error", "E-IO", "the graph has no runtime input")
    if not len(g.output):
        rep.add("error", "E-IO", "the graph has no output")
    for i in G.inputs:
        tt = i.type.tensor_type
        et = _dtype_name(tt.elem_type)
        if et not in _INPUT_TYPES:
            rep.add("error", "E-INPUT-TYPE", f"graph input {i.name!r} has element type {et} (allowed {sorted(_INPUT_TYPES)})")
        if not tt.HasField("shape") or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in tt.shape.dim):
            rep.add("error", "E-INPUT-SHAPE", f"graph input {i.name!r} has no fully static shape (the .bitino input "
                                              "shape is fixed)")
    if G.order_errors:
        rep.add("error", "E-TOPO", "nodes are not in topological order (an input is used before it is produced)",
                G.order_errors)


def _check_ops(model, G: _Graph, rep: Report, onnx_ver: int, declared: dict) -> dict:
    onnx = _onnx()
    counts: dict = {}
    for i, nd in enumerate(G.g.node):
        dom = nd.domain or ""
        key = nd.op_type if dom in ONNX_DOMAINS else f"{dom}:{nd.op_type}"
        counts[key] = counts.get(key, 0) + 1
        name = _nname(nd, i)
        if dom in ONNX_DOMAINS:
            if nd.op_type in REJECTED_OPS:
                rep.add("error", "E-OP", f"{nd.op_type}: {REJECTED_OPS[nd.op_type]}", [name])
                continue
            if nd.op_type not in ALLOWED_ONNX_OPS:
                rep.add("error", "E-OP", f"{nd.op_type} is not in the profile's operator set", [name])
                continue
            try:
                onnx.defs.get_schema(nd.op_type, onnx_ver, "")
            except Exception:
                rep.add("error", "E-OP-OPSET", f"{nd.op_type} does not exist at ai.onnx opset {onnx_ver}", [name])
        elif dom == QONNX_DOMAIN or dom in ("onnx.brevitas", "finn.custom_op.general", "finn"):
            if dom != QONNX_DOMAIN:
                rep.add("error", "E-DOMAIN", f"{nd.op_type} in legacy domain {dom!r}: canonical files use {QONNX_DOMAIN}",
                        [name])
            elif dom not in declared:
                rep.add("error", "E-DOMAIN-UNDECLARED", f"{nd.op_type} uses domain {dom!r}, which the file does not "
                                                         "import (opset_import)", [name])
            if nd.op_type in REJECTED_OPS:
                rep.add("error", "E-OP", f"{nd.op_type}: {REJECTED_OPS[nd.op_type]}", [name])
            elif nd.op_type not in QONNX_OPS:
                rep.add("error", "E-OP", f"{dom}:{nd.op_type} is not a QONNX operator of the profile "
                                         f"({sorted(QONNX_OPS)})", [name])
        else:
            rep.add("error", "E-DOMAIN", f"node in custom domain {dom!r}: standard ONNX + QONNX operators only", [name])
    return counts


def _merge_findings(rep: Report) -> None:
    merged: dict = {}
    out = []
    for f in rep.findings:
        k = (f.level, f.code, f.message)
        if k in merged:
            merged[k].nodes.extend(f.nodes)
        else:
            merged[k] = f
            out.append(f)
    for f in out:
        if len(f.nodes) > 1 and not f.message.startswith(f"{len(f.nodes)} "):
            f.message = f"{len(f.nodes)} nodes: {f.message}"
    rep.findings = out


def _binarizer(G: _Graph, nd) -> str | None:
    """Binarizer kind of a node (docs/qonnx-profile.md § Binarizers), else None."""
    op = nd.op_type
    if op == "BipolarQuant":
        return "ge"
    if op == "Sign":
        return "sign_zero"
    if op == "Relu" and nd.input and nd.input[0] in G.producer and G.producer[nd.input[0]][1].op_type == "Sign":
        return "gt0"                                    # BiBERT's {0, 1} map: Relu(Sign(x))
    if op == "Where" and len(nd.input) == 3:
        p = G.producer.get(nd.input[0])
        a, b = G.const(nd.input[1]), G.const(nd.input[2])
        if p and p[1].op_type in ("GreaterOrEqual", "Greater") and a is not None and b is not None \
                and a.size and b.size and np.all(a == 1) and (np.all(b == -1) or np.all(b == 0)):
            mode = "ge" if p[1].op_type == "GreaterOrEqual" else "gt"
            return mode if np.all(b == -1) else f"{mode}01"
    return None


def _pad_spec(G: _Graph, nd) -> tuple:
    """(pads or None, constant value or None, mode, value given explicitly) of a Pad node."""
    a = _attrs(nd)
    mode = a.get("mode", "constant")
    pads = G.const(nd.input[1]) if len(nd.input) > 1 and nd.input[1] else (np.asarray(a["pads"]) if "pads" in a else None)
    val = None
    if len(nd.input) > 2 and nd.input[2]:
        v = G.const(nd.input[2])
        val = None if v is None else float(np.asarray(v).reshape(-1)[0]) if np.asarray(v).size else None
    elif "value" in a:
        val = float(a["value"])
    return pads, val, mode, len(nd.input) > 2 and bool(nd.input[2]) or "value" in a


def _check_attrs_and_binary(G: _Graph, rep: Report) -> dict:
    stats = {"binarizers": {}, "binary_layers": 0}
    binarized: set = set()
    for i, nd in enumerate(G.g.node):
        name, op, a = _nname(nd, i), nd.op_type, _attrs(nd)
        dyn = G.is_dynamic_node(nd)
        # ---- attributes the profile fixes
        if op in _PADDED:
            if a.get("auto_pad", "NOTSET") not in ("NOTSET", ""):
                rep.add("error", "E-AUTOPAD", f"{op} with auto_pad={a['auto_pad']}: pads must be explicit", [name])
            if a.get("ceil_mode", 0):
                rep.add("error", "E-CEIL", f"{op} with ceil_mode=1: no Bitino kernel (profile v1)", [name])
        if op == "MaxPool" and (a.get("storage_order", 0) or len([o for o in nd.output if o]) > 1):
            rep.add("error", "E-ATTR", "MaxPool: storage_order 0 and a single output (no indices) only", [name])
        if op == "AveragePool" and any(a.get("pads", [])) and "count_include_pad" not in a:
            rep.add("error", "E-POOL-PAD", "padded AveragePool without an explicit count_include_pad attribute", [name])
        if op == "Pad" and _pad_spec(G, nd)[2] != "constant":
            rep.add("error", "E-PAD-MODE", f"Pad mode {_pad_spec(G, nd)[2]!r}: only constant padding", [name])
        if op == "Gemm" and (a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0):
            rep.add("error", "E-ATTR", "Gemm: transA 0, alpha 1, beta 1 only (write scales as Mul)", [name])
        if op == "Gelu" and a.get("approximate", "none") != "none":
            rep.add("error", "E-ATTR", "Gelu approximate='tanh' has no Bitino kernel", [name])
        if op == "Cast" and _dtype_name(a.get("to", 0)) not in _TENSOR_TYPES:
            rep.add("error", "E-DTYPE", f"Cast to {_dtype_name(a.get('to', 0))} is not allowed", [name])
        if op == "LayerNormalization" and a.get("axis", -1) != -1:
            rep.add("error", "E-ATTR", "LayerNormalization over the last axis only (axis = -1)", [name])
        if op == "Softmax" and a.get("axis", -1) != -1:
            rep.add("error", "E-ATTR", "Softmax over the last axis only (axis = -1)", [name])
        if op == "ConvInteger":
            if any(a.get("pads", [])):
                rep.add("error", "E-CONVINT-PAD", "ConvInteger with pads: ONNX does not define the padding code; pad "
                                                  "with an explicit Pad(constant pad code) and pads = 0", [name])
            if len(nd.input) > 3 and nd.input[3]:
                wz = G.const(nd.input[3])
                if wz is None or np.any(wz != 0):
                    rep.add("error", "E-ATTR", "ConvInteger weight zero point must be absent or 0 (symmetric weights)",
                            [name])
        if op == "IntQuant":
            rm = str(a.get("rounding_mode", "ROUND")).upper()
            if rm not in ("ROUND", "HALF_EVEN"):
                rep.add("error", "E-ROUNDING", f"IntQuant rounding_mode {rm}: only ROUND (half to even)", [name])
            vals = [G.const(x) if x else None for x in list(nd.input[1:4])]
            if len(nd.input) < 4 or any(v is None for v in vals):
                rep.add("error", "E-INTQUANT", "IntQuant scale / zeropt / bitwidth must be constants", [name])
            elif dyn:
                s, z, bw = vals
                if np.any(z != 0):
                    rep.add("error", "E-INTQUANT-ZP", "IntQuant of a runtime tensor with a non-zero zero point returns "
                                                      "(q - z)*s, Bitino's FakeQuant q*s (LSQ): write "
                                                      "Mul(Clip(Round(Add(Div(x, s), z)), qmin, qmax), s)", [name])
                if np.any(s <= 0):
                    rep.add("error", "E-INTQUANT", "IntQuant scale must be > 0", [name])
                b = float(np.asarray(bw).reshape(-1)[0])
                if b != int(b) or not 1 <= b <= 16:
                    rep.add("error", "E-INTQUANT", f"IntQuant bitwidth {b}: an integer in 1..16", [name])
                elif b == 1 and a.get("signed", 1):
                    rep.add("error", "E-INTQUANT", "signed 1-bit IntQuant is a bipolar binarizer: write BipolarQuant",
                            [name])
        # ---- binarizers and binarized tensors
        kind = _binarizer(G, nd) if dyn else None
        if kind:
            stats["binarizers"][kind] = stats["binarizers"].get(kind, 0) + 1
            binarized.update(o for o in nd.output if o)
            if op == "BipolarQuant":
                s = G.const(nd.input[1]) if len(nd.input) > 1 else None
                if s is None or np.any(s <= 0):
                    rep.add("error", "E-BINARIZER-SCALE", "BipolarQuant of a runtime tensor needs a constant scale > 0",
                            [name])
        elif dyn and op in _DATA_MOVEMENT and nd.input and nd.input[0] in binarized and \
                (op != "Concat" or all(x in binarized for x in nd.input if x in G.dynamic)):
            binarized.update(o for o in nd.output if o)
            if op == "Pad":
                pads, val, mode, explicit = _pad_spec(G, nd)
                allowed = {-1.0, 0.0, 1.0}
                if not explicit or val is None or val not in allowed:
                    rep.add("error", "E-BINPAD-VALUE", "Pad of a binarized tensor needs an explicit constant value in "
                                                       "{-1, 0, +1}", [name])
        declared = _props(nd.metadata_props)[0].get(FLOAT_LAYER_KEY)
        if declared is not None and (declared != "1" or not dyn or op not in ("Conv", "Gemm", "MatMul")
                                     or not nd.input or nd.input[0] not in binarized):
            rep.add("error", "E-FLOAT-LAYER", f"{FLOAT_LAYER_KEY} must be \"1\" on a Conv / Gemm / MatMul whose activation "
                                              "input is binarized", [name])
            declared = None
        if not dyn or op not in ("Conv", "ConvInteger", "MatMul", "MatMulInteger", "Gemm"):
            continue
        bin_in = [x for x in nd.input[:2] if x in binarized]
        if not bin_in:
            continue
        stats["binary_layers"] += 1
        if op in ("Conv", "ConvInteger") and nd.input[0] in binarized and any(a.get("pads", [])):
            rep.add("error", "E-BINPAD-IMPLICIT", f"{op} pads a binarized tensor implicitly: write an explicit "
                                                  "Pad(value in {-1, 0, +1}) before it and pads = 0", [name])
        w_name = nd.input[1]
        if w_name in G.dynamic:
            continue                                    # BMatMul of two runtime tensors
        p = G.producer.get(w_name)
        if p and p[1].op_type == "BipolarQuant" and declared is not None:
            rep.add("error", "E-FLOAT-LAYER", f"{FLOAT_LAYER_KEY} on BipolarQuant weights (a binary layer)", [name])
            continue
        if p and p[1].op_type == "BipolarQuant":
            s = G.const(p[1].input[1]) if len(p[1].input) > 1 else None
            if s is None:
                rep.add("info", "I-WEIGHTS", "binary weight scale not a direct constant (not checked)", [name])
            elif not np.all(s == 1):
                rep.add("warning", "W-ALPHA-INSIDE", "binary weights BipolarQuant(W, alpha) with alpha != 1: canonical is "
                                                     "BipolarQuant(W, 1) followed by Mul(alpha) (bit-exact fp32 "
                                                     "evaluation)", [name])
            continue
        w = G.const(w_name)
        if w is None:
            rep.add("info", "I-WEIGHTS", "weights computed by a constant subgraph (not checked)", [name])
            continue
        if op == "Gemm" and not _attrs(nd).get("transB", 0) or op in ("MatMul", "MatMulInteger"):
            per = np.abs(w.reshape(-1, w.shape[-1]).astype(np.float64)).T       # [K, N] -> per output column
        else:
            per = np.abs(w.reshape(w.shape[0], -1).astype(np.float64))           # [O, ...]
        alpha = per.max(axis=1, keepdims=True)
        if declared is not None:            # § 7 bitino.float_layer: the source's float weights on a binarized tensor
            if np.all(per == alpha) and np.all(alpha > 0):
                rep.add("error", "E-FLOAT-LAYER", f"{FLOAT_LAYER_KEY} on weights that are +-alpha per output channel (a "
                                                  "binary layer must not be declared float)", [name])
            else:
                rep.add("info", "I-FLOAT-LAYER", "declared float layer on a binarized tensor (imported as "
                                                 "Activation(sign) + float layer)", [name])
            continue
        if not np.all(per == alpha) and np.all(np.isin(w, (-1, 0, 1))):
            # R4: ternary weights (w_zero) as a plain {-1, 0, +1} constant (step 6), the constant form of
            # IntQuant(W, 1, 0, 2, signed, narrow); the products are in {-1, 0, +1}, the sums integers (B)
            rep.add("info", "I-TERNARY", "ternary weights: a {-1, 0, +1} constant (R4, w_zero)", [name])
        elif not np.all(per == alpha):
            rep.add("warning", "W-FLOAT-WEIGHTS", "weights on a binarized input are not +-alpha per output channel: "
                                                  "imported as a float layer on a binarized tensor", [name])
        elif not np.all(alpha == 1):
            rep.add("warning", "W-ALPHA-INSIDE", "binary weights stored as +-alpha constants with alpha != 1: canonical "
                                                 "is +-1 weights followed by Mul(alpha)", [name])
    return stats


def _check_model_metadata(model, G: _Graph, rep: Report) -> dict:
    md, dup = _props(model.metadata_props)
    if dup:
        rep.add("error", "E-META-DUP", f"duplicated model metadata keys {sorted(set(dup))}")
    missing = [k for k in REQUIRED_MODEL_KEYS if k not in md]
    variant = md.get("bitino.variant", "")
    if variant.startswith("int8_full") and "bitino.calibration" not in md:
        missing.append("bitino.calibration")
    if missing:
        rep.add("error", "E-META-MISSING", f"missing required model metadata: {', '.join(missing)}")

    def bad(key, why):
        rep.add("error", "E-META-FORMAT", f"{key}: {why}")

    for k, v in md.items():
        if not k.startswith("bitino."):
            continue
        if k not in REQUIRED_MODEL_KEYS and k not in OPTIONAL_MODEL_KEYS:
            rep.add("warning", "W-META-UNKNOWN", f"unknown model metadata key {k!r} (typo?)")
            continue
        if k == "bitino.profile" and v != PROFILE_ID:
            bad(k, f"{v!r} (expected {PROFILE_ID!r})")
        elif k == "bitino.model_id" and not MODEL_ID.match(v):
            bad(k, f"{v!r} is not a registry id ([a-z0-9][a-z0-9._-]*)")
        elif k == "bitino.variant" and not VARIANT.match(v):
            bad(k, f"{v!r} (float | int8_full | int8_full_<suffix>)")
        elif k == "bitino.layout" and v != "NCHW":
            bad(k, f"{v!r}: profile v1 files are NCHW")
        elif k == "bitino.input_domain":
            names = [i.name for i in G.inputs]
            if v.startswith("{"):
                obj, err = _json(v)
                if not isinstance(obj, dict):
                    bad(k, f"not a JSON object ({err})")
                else:
                    for n in names:
                        if n not in obj:
                            bad(k, f"no domain for graph input {n!r}")
                    for n, d in obj.items():
                        if n not in names:
                            bad(k, f"{n!r} is not a graph input")
                        elif not (d in INPUT_DOMAINS or (isinstance(d, str) and IDS_DOMAIN.match(d))):
                            bad(k, f"{n}: {d!r} (float32 | unit8 | uint8 | ids:<V>)")
            elif not (v in INPUT_DOMAINS or IDS_DOMAIN.match(v)):
                bad(k, f"{v!r} (float32 | unit8 | uint8 | ids:<V>, or a JSON object per graph input)")
            elif len(names) > 1:
                rep.add("warning", "W-META", f"{k}: one domain for {len(names)} graph inputs (give a JSON object per input)")
        elif k == "bitino.preprocessing":
            obj, err = _json(v)
            if not isinstance(obj, dict) or not all(isinstance(obj.get(x), str) and obj.get(x).strip()
                                                     for x in ("key", "definition")):
                bad(k, "JSON object with non-empty strings 'key' and 'definition' expected")
        elif k == "bitino.labels":
            if v != "none":
                obj, err = _json(v)
                if not isinstance(obj, dict) or not isinstance(obj.get("num_classes"), int) or obj["num_classes"] < 1 \
                        or isinstance(obj.get("num_classes"), bool) or not (isinstance(obj.get("set"), str) and obj["set"].strip()):
                    bad(k, "'none' or a JSON object {\"num_classes\": int >= 1, \"set\": label-set id} expected")
        elif k == "bitino.source.checkpoint_sha256":
            if v.startswith("{"):
                obj, err = _json(v)
                if not isinstance(obj, dict) or not obj or not all(isinstance(h, str) and HEX64.match(h) for h in obj.values()):
                    bad(k, "JSON object {file: 64 lowercase hex digits} expected")
            elif not HEX64.match(v):
                bad(k, "64 lowercase hex digits (or a JSON object per checkpoint file) expected")
        elif k in ("bitino.source.framework", "bitino.source.framework_version", "bitino.exporter",
                   "bitino.weights_license") and not v.strip():
            bad(k, "empty")
        elif k == "bitino.weights_license" and v.strip().lower() == "unknown":
            rep.add("warning", "W-META", "bitino.weights_license is 'unknown': the converted file cannot be redistributed")
        elif k == "bitino.source.code" and not CODE_REF.match(v):
            rep.add("warning", "W-META", f"bitino.source.code {v!r} is not '<repo url>@<commit>'")
        elif k == "bitino.calibration":
            obj, err = _json(v)
            if not isinstance(obj, dict) or not all(x in obj for x in ("set", "split", "n", "tensor_sha256")):
                bad(k, "JSON object with set, split, n, tensor_sha256 expected")
            else:
                if not isinstance(obj["n"], int) or obj["n"] < 1:
                    bad(k, "n must be an integer >= 1")
                if not (isinstance(obj["tensor_sha256"], str) and HEX64.match(obj["tensor_sha256"])):
                    bad(k, "tensor_sha256 must be 64 lowercase hex digits")
                if obj["split"] != "train":
                    rep.add("warning", "W-META", f"bitino.calibration split {obj['split']!r}: calibration data must be "
                                                 "disjoint from every evaluation set (train split)")
        elif k == "bitino.import":
            obj, err = _json(v)
            if isinstance(obj, dict) and obj.get("mapping") == "patterns":
                badk = [x for x in obj if x != "mapping" and (x not in PATTERN_IMPORT_OPTIONS or obj[x] not in PATTERN_IMPORT_OPTIONS[x])]
                if badk:
                    bad(k, f"mapping patterns: unknown option(s) or value(s) {badk} (options {PATTERN_IMPORT_OPTIONS})")
            elif not isinstance(obj, dict) or obj.get("mapping") not in IMPORT_MAPPINGS or set(obj) - {"mapping"}:
                bad(k, f"JSON object {{\"mapping\": one of {list(IMPORT_MAPPINGS) + ['patterns']}}} expected")
        elif k == "bitino.ir_unique_counter" and not ORDER.match(v):
            bad(k, f"{v!r} is not a non-negative integer")
        elif k == "bitino.unused_constants":       # step 5: IR constants no node uses (data from initializers)
            obj, err = _json(v)
            inits = {t.name for t in model.graph.initializer}
            used = {i for nd in model.graph.node for i in nd.input}
            if not isinstance(obj, list) or not all(isinstance(x, str) and x in inits and x not in used for x in obj) \
                    or len(set(obj)) != len(obj):
                bad(k, "JSON list of distinct initializer names that no node uses expected")
        elif k == "bitino.const_order":
            obj, err = _json(v)
            if not isinstance(obj, list) or not all(isinstance(x, str) and x for x in obj):
                bad(k, "JSON list of constant names expected")
            elif len(set(obj)) != len(obj):
                bad(k, "duplicated constant names")
    return md


def _check_naming(G: _Graph, rep: Report) -> dict:
    groups: dict = {}         # bitino.name -> set of orders
    missing, extras, const_names = [], {}, {}
    named = act = 0
    for i, nd in enumerate(G.g.node):
        name = _nname(nd, i)
        md, dup = _props(nd.metadata_props)
        if dup:
            rep.add("error", "E-NAME-FORMAT", f"duplicated node metadata keys {sorted(set(dup))}", [name])
        for k in md:
            if k.startswith("bitino.") and k not in NODE_KEYS:
                rep.add("warning", "W-NAME-UNKNOWN", f"unknown node metadata key {k!r}", [name])
        dyn = G.is_dynamic_node(nd)
        act += dyn
        ir = md.get("bitino.name")
        if ir is None:
            if dyn:
                missing.append(name)
            for k in ("bitino.order", "bitino.extra_names", "bitino.consts", "bitino.attrs", "bitino.extra_attrs", "bitino.ir",
                      "bitino.const_layouts"):
                if k in md:
                    rep.add("error", "E-NAME-FORMAT", f"{k} without bitino.name", [name])
            continue
        if "bitino.const_layouts" in md:
            obj, err = _json(md["bitino.const_layouts"])
            if not isinstance(obj, dict) or not all(isinstance(k, str) and k.isdigit() and v in CONST_LAYOUT_TAGS
                                                    for k, v in obj.items()):
                rep.add("error", "E-NAME-FORMAT", "bitino.const_layouts must be a JSON object {'<input index>': layout tag} "
                                                  f"with tags in {list(CONST_LAYOUT_TAGS)}", [name])
        named += 1
        if not ir or ir != ir.strip() or _BAD_NAME.search(ir):
            rep.add("error", "E-NAME-FORMAT", f"bitino.name {ir!r}: non-empty, no surrounding whitespace or control "
                                              "characters", [name])
        order = md.get("bitino.order")
        if order is not None and not ORDER.match(order):
            rep.add("error", "E-NAME-FORMAT", f"bitino.order {order!r} is not a non-negative integer", [name])
            order = None
        groups.setdefault(ir, set()).add(int(order) if order is not None else None)
        if "bitino.extra_names" in md:
            obj, err = _json(md["bitino.extra_names"])
            if not isinstance(obj, list) or not all(isinstance(x, str) and x and not _BAD_NAME.search(x) for x in obj) \
                    or len(set(obj)) != len(obj):
                rep.add("error", "E-NAME-FORMAT", "bitino.extra_names must be a JSON list of distinct names", [name])
                obj = []
            for x in obj:
                if x in extras and extras[x] != ir:
                    rep.add("error", "E-NAME-DUP", f"extra IR node name {x!r} used by groups {extras[x]!r} and {ir!r}",
                            [name])
                extras[x] = ir
        else:
            obj = []
        own_extra = set(obj)
        for k in ("bitino.attrs", "bitino.extra_attrs"):
            if k in md:
                v, err = _json(md[k])
                if not isinstance(v, dict) or (k == "bitino.extra_attrs" and not all(
                        isinstance(x, dict) and n in own_extra for n, x in v.items())):
                    rep.add("error", "E-NAME-FORMAT", f"{k} must be a JSON object"
                            + (" {extra name: {attr: value}} over this node's bitino.extra_names" if k == "bitino.extra_attrs" else ""),
                            [name])
        if "bitino.ir" in md:
            obj, err = _json(md["bitino.ir"])
            ok = isinstance(obj, dict) and isinstance(obj.get("op"), str) and obj["op"] and \
                not set(obj) - {"op", "n_inputs", "layouts", "outputs"}
            ok = ok and (("n_inputs" not in obj) or (isinstance(obj["n_inputs"], int) and not isinstance(obj["n_inputs"], bool)
                                                     and obj["n_inputs"] >= 0))
            ok = ok and (("layouts" not in obj) or (isinstance(obj["layouts"], dict) and all(
                ORDER.match(str(a)) and isinstance(b, str) for a, b in obj["layouts"].items())))
            ok = ok and (("outputs" not in obj) or (isinstance(obj["outputs"], list) and obj["outputs"] and all(
                isinstance(x, str) and x and not _BAD_NAME.search(x) for x in obj["outputs"])
                and len(set(obj["outputs"])) == len(obj["outputs"])))      # step 5: IR output tensor names
            if not ok:
                rep.add("error", "E-NAME-FORMAT", "bitino.ir must be a JSON object {\"op\": IR operator, \"n_inputs\": int, "
                                                  "\"layouts\": {input index: layout tag}, \"outputs\": [IR output names]}",
                        [name])
        if "bitino.consts" in md:
            obj, err = _json(md["bitino.consts"])
            if not isinstance(obj, dict):
                rep.add("error", "E-NAME-FORMAT", "bitino.consts must be a JSON object {slot or 'node#slot': name}", [name])
                continue
            for key, cname in obj.items():
                m = CONST_KEY.match(key)
                if not m or (m.group("node") and m.group("node") not in own_extra and m.group("node") != ir):
                    rep.add("error", "E-NAME-FORMAT", f"bitino.consts key {key!r}: '<input index>' or "
                                                      "'<IR node name of this group>#<input index>'", [name])
                if not isinstance(cname, str) or not cname or _BAD_NAME.search(cname):
                    rep.add("error", "E-NAME-FORMAT", f"bitino.consts[{key!r}] must be a non-empty name", [name])
                    continue
                prev = const_names.get(cname)
                if prev is not None and prev != (ir, key) and not getattr(G, "group_mapped", False):
                    rep.add("error", "E-NAME-DUP", f"IR constant name {cname!r} assigned twice", [name])
                const_names[cname] = (ir, key)
    if missing:
        rep.add("error", "E-NAME-MISSING", f"{len(missing)} of {act} activation-path nodes have no bitino.name "
                                           "(byte identity needs the IR name of every node)", missing)
    for x, ir in extras.items():
        if x in groups:
            rep.add("error", "E-NAME-DUP", f"extra IR node name {x!r} (group {ir!r}) is also a bitino.name group")
    with_order = {n for n, o in groups.items() if o != {None}}
    for n, o in groups.items():
        if len(o) > 1:
            rep.add("error", "E-ORDER", f"group {n!r}: its nodes carry different bitino.order values {sorted(map(str, o))}")
    if with_order and len(with_order) != len(groups):
        rep.add("error", "E-ORDER", f"bitino.order on {len(with_order)} of {len(groups)} groups: all or none")
    seen: dict = {}
    for n, o in groups.items():
        for v in o:
            if v is not None:
                if v in seen and seen[v] != n:
                    rep.add("error", "E-ORDER", f"bitino.order {v} used by groups {seen[v]!r} and {n!r}")
                seen[v] = n
    return {"named_nodes": named, "activation_nodes": act, "groups": len(groups)}


def _check_shapes(model, rep: Report) -> None:
    onnx = _onnx()
    m = copy.deepcopy(model)
    g = m.graph
    for k, nd in enumerate(list(g.node)):        # qonnx quantizers keep their input's shape and type
        if (nd.domain or "") not in ONNX_DOMAINS and nd.op_type in ("BipolarQuant", "IntQuant", "Quant") and nd.input:
            new = onnx.helper.make_node("Identity", [nd.input[0]], [nd.output[0]], name=nd.name)
            g.node.remove(nd)
            g.node.insert(k, new)
    keep = [o for o in m.opset_import if (o.domain or "") in ONNX_DOMAINS]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    if any((nd.domain or "") not in ONNX_DOMAINS for nd in g.node):
        rep.add("info", "I-SHAPES", "shape inference skipped (nodes outside ai.onnx / the QONNX quantizers)")
        return
    try:
        onnx.checker.check_model(m)
    except Exception as e:
        rep.add("error", "E-ONNX-CHECKER", f"onnx.checker: {str(e).splitlines()[0][:300]}")
    try:
        m = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True, data_prop=True)
    except Exception as e:
        rep.add("error", "E-SHAPES", f"shape inference failed: {str(e).splitlines()[0][:300]}")
        return
    known, etype = {}, {t.name: t.data_type for t in m.graph.initializer}
    for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
        tt = vi.type.tensor_type
        static = tt.elem_type != 0 and tt.HasField("shape") and all(d.HasField("dim_value") for d in tt.shape.dim)
        known[vi.name] = static or known.get(vi.name, False)
        etype.setdefault(vi.name, tt.elem_type)
    i64 = [nd.name or nd.op_type for nd in m.graph.node if nd.op_type in ("Clip", "Min", "Max") and nd.input
           and etype.get(nd.input[0]) == onnx.TensorProto.INT64]
    if i64:
        rep.add("error", "E-INT64-CLAMP", f"{len(i64)} Clip / Min / Max on INT64: onnxruntime 1.31 compares INT64 there "
                                          "inexactly (2^31 passes a clip at 2^31-1); saturate with Where(Greater / Less)",
                i64)
    inits = {t.name for t in m.graph.initializer}
    bad = [o for nd in m.graph.node for o in nd.output if o and o not in inits and not known.get(o, False)]
    if bad:
        rep.add("error", "E-SHAPES", f"{len(bad)} tensor(s) without an inferable static shape and type", bad)


def _check_adabin(G: _Graph, rep: Report) -> dict:
    """§ 4.1 AdaBin binary conv: Conv([Pad(.., 0)](Mul(Add(Bin(Sub(x, beta)), beta), alpha)), W) with Bin = Sign or
    BipolarQuant(., 1). The importer decomposes it into binary operators, so every instance must meet the pattern's
    conditions (E-ADABIN otherwise): alpha a constant scalar > 0, the same constant scalar beta in Sub and Add, an
    explicit Pad only with value 0 (or the Conv's own zero pads), group 1, no dilation, no bias, constant weights with
    exactly two values per output channel."""
    n = 0

    def prod(t, ops):
        p = G.producer.get(t)
        return p[1] if p and p[1].op_type in ops else None

    def scalar(t):
        c = G.const(t)
        v = None if c is None else np.unique(np.asarray(c, np.float64))
        return float(v[0]) if v is not None and v.size == 1 else None

    for i, nd in enumerate(G.g.node):
        if nd.op_type != "Conv" or not G.is_dynamic_node(nd) or not nd.input:
            continue
        x, pad = nd.input[0], None
        pd = prod(x, ("Pad",))
        if pd is not None:
            pad, x = pd, pd.input[0]
        mul = prod(x, ("Mul",))
        add = mul is not None and prod(next((t for t in mul.input if t in G.dynamic), ""), ("Add",))
        bin_ = add and prod(next((t for t in add.input if t in G.dynamic), ""), ("Sign", "BipolarQuant"))
        sub = bin_ and prod(bin_.input[0], ("Sub",))
        if not sub:
            continue
        n += 1
        name = _nname(nd, i)
        why = []
        alpha = scalar(next((t for t in mul.input if t not in G.dynamic), ""))
        b_add = scalar(next((t for t in add.input if t not in G.dynamic), ""))
        b_sub = scalar(sub.input[1]) if len(sub.input) > 1 and sub.input[1] not in G.dynamic and sub.input[0] in G.dynamic else None
        if alpha is None or not alpha > 0:
            why.append("alpha must be a constant scalar > 0")
        if b_sub is None or b_add is None or b_sub != b_add:
            why.append("beta must be the same constant scalar in Sub(x, beta) and Add(., beta)")
        if bin_.op_type == "BipolarQuant" and scalar(bin_.input[1]) != 1.0:
            why.append("BipolarQuant scale must be 1")
        if pad is not None:
            pads, val, mode, _ = _pad_spec(G, pad)
            if mode != "constant" or val != 0.0 or any(_attrs(nd).get("pads", [])):
                why.append("an explicit Pad must have value 0 and the Conv pads 0")
        a = _attrs(nd)
        if int(a.get("group", 1)) != 1 or any(int(d) != 1 for d in a.get("dilations", [1, 1])) or (len(nd.input) > 2 and nd.input[2]):
            why.append("group 1, no dilation, no bias")
        w = G.const(nd.input[1]) if len(nd.input) > 1 else None
        if w is None:
            why.append("constant weights")
        else:
            flat = np.asarray(w, np.float64).reshape(np.shape(w)[0], -1)
            hi, lo = flat.max(1, keepdims=True), flat.min(1, keepdims=True)
            if not (np.all((flat == hi) | (flat == lo)) and np.all(hi > lo)):
                why.append("the weights must take exactly two values per output channel")
        if why:
            rep.add("error", "E-ADABIN", "AdaBin binary conv (§ 4.1): " + "; ".join(why), [name])
        else:
            rep.add("info", "I-ADABIN", "AdaBin binary conv (decomposed by the importer into binary operators)", [name])
    return {"adabin_convs": n}


def check_model(model, path: str = "", sha256: str = "") -> Report:
    """Validates an onnx.ModelProto against profile v1; returns the Report."""
    rep = Report(path=path)
    info = _check_container(model, rep)
    G = _Graph(model)
    _check_io(G, rep)
    counts = _check_ops(model, G, rep, info["onnx_opset"], info["opsets"])
    stats = _check_attrs_and_binary(G, rep)
    ab = _check_adabin(G, rep)
    if ab["adabin_convs"]:
        stats.update(ab)
    _check_model_metadata(model, G, rep)
    md_model, _ = _props(model.metadata_props)
    G.group_mapped = "bitino.import" in md_model      # one ONNX group per IR node (step 3): shared IR constants allowed
    names = _check_naming(G, rep)
    _check_shapes(model, rep)
    _merge_findings(rep)
    rep.summary = {"ir_version": int(model.ir_version),
                   "opsets": {(d or "ai.onnx"): v for d, v in info["opsets"].items()},
                   "nodes": len(model.graph.node), "initializers": len(model.graph.initializer),
                   "op_counts": counts, **stats, **names}
    if sha256:
        rep.summary["sha256"] = sha256
    return rep


def check_file(path: str) -> Report:
    onnx = _onnx()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    model = onnx.load(path, load_external_data=False)
    return check_model(model, path=path, sha256=h.hexdigest())


def build_parser(prog: str = "check-qonnx") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=f"check ONNX files against the Bitino QONNX profile "
                                                        f"({PROFILE_ID}, docs/qonnx-profile.md)")
    ap.add_argument("files", nargs="+", help="ONNX / QONNX files")
    ap.add_argument("--json", metavar="OUT", help="also write the reports as JSON ('-' = stdout instead of text)")
    ap.add_argument("--strict", action="store_true", help="warnings fail too")
    ap.add_argument("--no-info", action="store_true", help="omit info findings from the text report")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _onnx()
    except ImportError:
        print("check-qonnx needs the onnx package (e.g. .cache/venv-qonnx/bin/python)", file=sys.stderr)
        return 2
    reports, rc = [], 0
    for p in args.files:
        try:
            rep = check_file(p)
        except Exception as e:   # unreadable / not ONNX
            print(f"check-qonnx {p}: cannot read: {e}", file=sys.stderr)
            rc = max(rc, 2)
            continue
        if args.no_info:
            rep.findings = [f for f in rep.findings if f.level != "info"]
        reports.append(rep)
        if args.json != "-":
            print(rep.to_text(args.strict))
            print()
        if not rep.passed(args.strict):
            rc = max(rc, 1)
    if args.json:
        doc = [r.to_dict() for r in reports]
        txt = json.dumps(doc if len(doc) != 1 else doc[0], indent=1, default=str)
        if args.json == "-":
            print(txt)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w") as f:
                f.write(txt + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
