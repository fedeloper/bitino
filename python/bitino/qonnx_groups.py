"""Step 3 of the QONNX route (docs/qonnx-frontend.md § Plan, docs/qonnx-step3.md): one ONNX *group* per IR node.

The transformer networks (BiBERT, Bi-ViT, GSB, BHViT) and the "Faster Than Lies" detectors with their image-feature
front end are written as canonical files of the Bitino profile (docs/qonnx-profile.md) in which every IR operator of
the deployed graph is one group of standard ONNX + QONNX nodes (the nodes sharing one ``bitino.name``), in the
pattern docs/qonnx-profile.md § 4 gives for that operator. The importer (``import_groups``) matches each group's
subgraph back onto its fused IR operator: the operator, its attributes and its constants are read from the ONNX
nodes and initializers of the group (each reader checks the full pattern and fails on anything else). The naming
metadata (§ 7) carries what the identity bar needs beyond semantics; this step adds

* model key ``bitino.import`` = ``{"mapping": "groups"}``: the file is imported group by group (this module) and
  no graph pass runs after the import (the groups are the deployed graph). The choice is recorded in the file, not
  given as an import argument (the open design point of step 2);
* node key ``bitino.ir`` = ``{"op": <IR operator>, "n_inputs": <count>, "layouts": {<input index>: <layout tag>}}``:
  the IR operator of the group where the ONNX pattern admits several IR operators that compute the same values
  (``Add`` of a constant is ``Add``, ``AddBroadcast`` or a shift-only ``ChannelAffine``; ``Mul`` by a constant is
  ``Mul`` or a scale-only ``ChannelAffine``), the IR input count where the IR node has trailing empty inputs, and
  the constant layout tags of the published file (``OI_bit``, ``VD_bit``, ...). None of them changes a value: the
  reader of the named operator still checks the pattern, and the operators it may name compute the same values.

Layout (docs/qonnx-profile.md R3). 4-D image tensors are NCHW in the file; every other tensor (tokens, attention
maps, the 5-D / 6-D window tensors of BHViT, 4-D tensors that the IR reshapes from tokens) has the IR's own shape.
Each ONNX tensor therefore has a *tag*: ``C`` (an NCHW view: the IR tensor is its (0, 2, 3, 1) transpose) or ``I``
(the IR tensor itself). The graph input of an image model and the outputs of spatial operators (Conv, pools, the
feature subgraphs) are ``C``; per-channel / element-wise groups keep the tag of their inputs; all other groups read
and write ``I``. A change of tag is an explicit ``Transpose(0, 2, 3, 1)`` (C -> I) or ``Transpose(0, 3, 1, 2)``
(I -> C), written as the last node of the producer's group (of the first consumer's group for a graph input); it
emits no IR node.

Writer: ``write_spec(graph, shapes)`` (numpy only, no onnx package: it runs in the legacy converters' environments)
turns an IR graph into a spec (nodes, initializers, inputs, outputs, metadata) that ``assemble`` (onnx) writes.
Importer: ``import_groups(model)`` -> (Graph, info); ``qonnx_import.import_model`` dispatches here when the file
carries ``bitino.import``.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from .ir import Graph, pack_bits, unpack_bits

IMPORT_KEY = "bitino.import"
IR_KEY = "bitino.ir"
NAME_KEY = "bitino.name"
ORDER_KEY = "bitino.order"
CONSTS_KEY = "bitino.consts"
ATTRS_KEY = "bitino.attrs"
CONST_ORDER_KEY = "bitino.const_order"
QD = "qonnx.custom_op.general"
MAPPING = "groups"

# ONNX TensorProto element types
FLOAT, UINT8, INT8, INT32, INT64, BOOL, DOUBLE, UINT32 = 1, 2, 3, 6, 7, 9, 11, 12
_NP_OF = {FLOAT: np.float32, INT8: np.int8, INT32: np.int32, INT64: np.int64, BOOL: np.bool_, DOUBLE: np.float64,
          UINT32: np.uint32, UINT8: np.uint8}

C2I = (0, 2, 3, 1)     # NCHW view -> IR (NHWC) tensor
I2C = (0, 3, 1, 2)
SPATIAL = ("Conv2D", "BConv2D", "MaxPool2D", "AvgPool2D", "GlobalAvgPool", "ImageGrayU8", "FftLogMagnitude",
           "LocalBinaryPattern", "ChannelGate")
# operators whose attributes a reader derives; absent attributes of the published file (bitino.attrs null) must
# have the operator's default value (docs/operators.md)
ATTR_DEFAULTS = {("Conv2D", "dilations"): [1, 1], ("Conv2D", "groups"): 1, ("Conv2D", "pad_value"): 0.0,
                 ("Conv2D", "activation"): "none", ("BConv2D", "activation"): "none", ("BConv2D", "output"): "f32",
                 ("BConv2D", "pad_value"): 0.0, ("Dense", "activation"): "none"}
# step 6 (docs/qonnx-profile.md § 7 (a), the step-4 rule in group mapping): an attribute the reader does not derive may be
# ADDED by bitino.attrs with exactly the operator's default value (the FINN-style converters write BDense
# activation "none", the step-3 graphs omit it)
ATTR_ADD_DEFAULTS = {("BDense", "activation"): "none"}
DEFAULT_LAYOUTS = {("Conv2D", 1): "OHWI", ("BConv2D", 1): "OHWI_bit"}


class GroupImportError(NotImplementedError):
    def __init__(self, group: str, reason: str, op: str = "?"):
        self.node_name = group
        self.op = op
        super().__init__(f"group {group!r} ({op}): {reason}")


# ================================================================================================ shared helpers
def dft_tables(n: int) -> tuple[np.ndarray, np.ndarray]:
    """C[k, m] = cos(2 pi ((k m) mod n) / n), S[k, m] = sin(...) in float64 (numpy libm): the DFT matrices of the
    FftLogMagnitude pattern (docs/qonnx-profile.md § 4.5, Q1). Writer and importer compute them with this one
    function; the importer requires the file's tables to be these values bit for bit."""
    k = np.arange(n, dtype=np.int64)
    ph = 2.0 * np.pi * ((k[:, None] * k[None, :]) % n).astype(np.float64) / float(n)
    return np.cos(ph), np.sin(ph)


def fftshift_index(n: int) -> np.ndarray:
    """Indices of np.roll(v, n // 2) along one axis: v_new[i] = v[(i - n // 2) mod n]."""
    return ((np.arange(n, dtype=np.int64) - n // 2) % n).astype(np.int64)


def lbp_shift_ok(rp: float, n: int) -> bool:
    """The uniform-shift condition of the LocalBinaryPattern pattern (§ 4.5, Q12): floor(r + rp) = r + floor(rp) and
    ceil(r + rp) = r + ceil(rp) for every index r < n, evaluated in float64 as the kernel does."""
    r = np.arange(n, dtype=np.float64)
    return bool(np.all(np.floor(r + rp) == r + math.floor(rp)) and np.all(np.ceil(r + rp) == r + math.ceil(rp)))


def _json_num_same(a, b) -> bool:
    return (isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float))
            and not isinstance(b, bool) and float(a) == float(b))


def _informational(op: str, k: str, canon: dict) -> bool:
    """docs/qonnx-profile.md § 7 (c) in float group mapping (step 6): an attribute no computation reads in the node's
    configuration, whose value bitino.attrs supplies: pad_value of an unpadded binary convolution (the OpenVINO
    BinaryConvolution's pad_value -1 on its 1x1 convs)."""
    return op == "BConv2D" and k == "pad_value" and not any(canon.get("pads", [0, 0, 0, 0]))


def attrs_meta(op: str, ir_attrs: dict, canon: dict) -> dict:
    """bitino.attrs entries (docs/qonnx-profile.md § 7) that turn the reader's canonical attributes `canon` into the
    IR node's `ir_attrs`: null for an absent attribute (only at its default), the IR's JSON form of the same number,
    or the float64 value of a float32 attribute. Raises if the two differ in value."""
    out = {}
    for k in sorted(set(canon) | set(ir_attrs)):
        if k not in ir_attrs:
            d = ATTR_DEFAULTS.get((op, k), ())
            if d == () or canon[k] != d or type(canon[k]) is not type(d):
                raise AssertionError(f"{op}: attribute {k!r} absent in the IR node but the reader gives {canon[k]!r} "
                                     "(not the operator default)")
            out[k] = None
            continue
        if k not in canon:
            if (op, k) in ATTR_ADD_DEFAULTS and json.dumps(ir_attrs[k]) == json.dumps(ATTR_ADD_DEFAULTS[(op, k)]):
                out[k] = ir_attrs[k]
                continue
            raise AssertionError(f"{op}: IR attribute {k!r} is not derived by the reader")
        a, c = ir_attrs[k], canon[k]
        if json.dumps(a) == json.dumps(c):
            continue
        if _json_num_same(a, c) or (isinstance(a, float) and isinstance(c, float) and float(np.float32(a)) == c):
            out[k] = a
            continue
        if _informational(op, k, canon):
            out[k] = a
            continue
        raise AssertionError(f"{op}: attribute {k}: IR {a!r} vs reader {c!r}")
    return out


def apply_attrs_meta(op: str, canon: dict, meta: dict, group: str) -> dict:
    """The importer side of attrs_meta: the same three rules, anything else is an import error."""
    out = dict(canon)
    for k, v in meta.items():
        if k not in out:
            if (op, k) in ATTR_ADD_DEFAULTS and json.dumps(v) == json.dumps(ATTR_ADD_DEFAULTS[(op, k)]):
                out[k] = v              # the operator default, added (§ 7 (a))
                continue
            raise GroupImportError(group, f"bitino.attrs: the reader derives no attribute {k!r}", op)
        cur = out[k]
        if v is None:
            d = ATTR_DEFAULTS.get((op, k), ())
            if d == () or cur != d or type(cur) is not type(d):
                raise GroupImportError(group, f"bitino.attrs {k}=null would change the value {cur!r}", op)
            del out[k]
            continue
        if _json_num_same(v, cur) or (isinstance(v, float) and isinstance(cur, float) and float(np.float32(v)) == cur):
            out[k] = v
            continue
        if _informational(op, k, canon):
            out[k] = v
            continue
        raise GroupImportError(group, f"bitino.attrs {k}={v!r} would change the value {cur!r}", op)
    return out


def _f(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _onnx_pads(p) -> list:          # IR [t, b, l, r] -> ONNX [t, l, b, r]
    return [int(p[0]), int(p[2]), int(p[1]), int(p[3])]


def _ir_pads(p) -> list:            # ONNX [t, l, b, r] -> IR [t, b, l, r]
    return [int(p[0]), int(p[2]), int(p[1]), int(p[3])]


# ================================================================================================ writer
class Spec:
    """ONNX graph under construction without the onnx package: nodes as dicts, initializers as numpy arrays."""

    def __init__(self) -> None:
        self.nodes: list = []
        self.inits: dict = {}
        self.inputs: list = []           # (name, elem type, shape)
        self.outputs: list = []
        self.meta: dict = {}
        self.k = 0

    def init(self, name: str, arr) -> str:
        arr = np.asarray(arr)
        if name in self.inits:
            old = self.inits[name]
            if not (old.dtype == arr.dtype and old.shape == arr.shape and np.array_equal(old, arr)):
                raise AssertionError(f"initializer {name} defined twice with different values")
            return name
        self.inits[name] = np.array(arr, order="C", copy=True)
        return name

    def node(self, op: str, ins: list, attrs: dict | None = None, out: str | None = None, meta: dict | None = None,
             domain: str = "") -> str:
        self.k += 1
        out = out or f"t{self.k}"
        self.nodes.append({"op": op, "inputs": list(ins), "outputs": [out], "attrs": dict(attrs or {}),
                           "name": f"n{self.k}_{op}", "domain": domain, "meta": dict(meta or {})})
        return out


class GroupWriter:
    """IR graph (the deployed graph) -> canonical spec, one group per IR node (module docstring)."""

    def __init__(self, g: Graph, shapes: dict):
        self.g = g
        self.shapes = {k: tuple(int(d) for d in v) for k, v in shapes.items()}
        for c in g.constants.values():
            self.shapes.setdefault(c.name, tuple(int(d) for d in c.shape))
        self.s = Spec()
        self.report = {"groups": 0, "onnx_nodes": 0, "layout_transposes": 0, "carriers": 0, "groups_by_op": {},
                       "attrs_meta": 0}
        self.consts = g.constants
        self.prod = {n.outputs[0]: n for n in g.nodes}
        self.cons: dict = {}
        for n in g.nodes:
            for j, i in enumerate(n.inputs):
                if i and i not in self.consts:
                    self.cons.setdefault(i, []).append((n, j))
        self.inputs = {i["name"]: i for i in g.inputs}
        bad = [n.name for n in g.nodes if n.outputs != [f"{n.name}:out"]]
        if bad:
            raise AssertionError(f"IR nodes with outputs other than '<name>:out': {bad[:5]}")
        used = {i for n in g.nodes for i in n.inputs if i}
        unused = [k for k in self.consts if k not in used]
        if unused:
            raise AssertionError(f"unused constants {unused[:5]}")
        s = self.s
        self.ONE = s.init("bitino/one", np.array(1.0, np.float32))
        self.ZERO = s.init("bitino/zero", np.array(0.0, np.float32))
        self.M1 = s.init("bitino/minus_one", np.array(-1.0, np.float32))

    # ---------------------------------------------------------------- layout tags
    def rank(self, t: str) -> int:
        return len(self.shapes[t])

    def _cmode_const(self, c: str) -> bool:
        return self.rank(c) in (1, 4)

    def plan(self) -> None:
        """Primary tag of every runtime IR tensor and the mode (C / I) of every node."""
        prim: dict = {}
        for i in self.g.inputs:
            prim[i["name"]] = "C" if (len(i["shape"]) == 4 and i["dtype"] == "f32") else "I"
        mode: dict = {}
        for n in self.g.nodes:
            rt = [i for i in n.inputs if i and i not in self.consts]
            cs = [i for i in n.inputs if i and i in self.consts]
            op = n.op
            if op in SPATIAL:
                m = "C"
            elif op in ("ChannelAffine", "Activation", "Add", "AddBroadcast", "Mul", "Concat") and rt and \
                    all(self.rank(t) == 4 and prim[t] == "C" for t in rt) and all(self._cmode_const(c) for c in cs):
                m = "C"
                if op == "Concat" and (cs or int(n.attrs.get("axis", -1)) != -1):
                    m = "I"
                if op == "Activation" and n.attrs["type"] == "prelu" and self.rank(n.inputs[1]) != 1:
                    m = "I"
                if op == "ChannelAffine" and any(self.rank(c) != 1 for c in cs):
                    m = "I"
                if op in ("Add", "AddBroadcast", "Mul") and any(self.rank(c) == 4 and self.shapes[c][0] != 1 for c in cs):
                    m = "I"
            else:
                m = "I"
            mode[n.name] = m
            out = n.outputs[0]
            if op == "GlobalAvgPool" and not int(n.attrs.get("keepdims", 0)):
                prim[out] = "I"
            else:
                prim[out] = m if self.rank(out) == 4 else "I"
        self.prim, self.mode = prim, mode
        need: dict = {}
        for n in self.g.nodes:
            for j, t in enumerate(n.inputs):
                if not t or t in self.consts:
                    continue
                want = self.want_tag(n, j)
                if want != prim[t]:
                    need.setdefault(t, set()).add(want)
        for t in self.g.outputs:
            if prim[t] != "I":
                need.setdefault(t, set()).add("I")
        self.need = need

    def want_tag(self, n, j) -> str:
        t = n.inputs[j]
        if self.rank(t) != 4:
            return "I"
        if n.op == "ChannelGate" and j == 1:
            return "C"
        return self.mode[n.name]

    # ---------------------------------------------------------------- emission
    def meta(self, extra: dict | None = None) -> dict:
        n, i = self.cur
        m = {NAME_KEY: n.name, ORDER_KEY: str(i)}
        if extra:
            m.update(extra)
        return m

    def emit(self, op, ins, attrs=None, out=None, domain=""):
        self.report["onnx_nodes"] += 1
        return self.s.node(op, ins, attrs, out=out, meta=self.meta(), domain=domain)

    def rt(self, t: str, tag: str) -> str:
        """ONNX tensor of runtime IR tensor t in tag `tag` (written earlier: primary or its conversion)."""
        if self.rank(t) != 4:
            tag = "I"
        r = self.onnx.get((t, tag))
        if r is None:
            if t in self.inputs and (t, self.prim[t]) in self.onnx:
                # graph input: the conversion is a node of the first consumer's group
                src = self.onnx[(t, self.prim[t])]
                r = self.emit("Transpose", [src], {"perm": list(C2I if self.prim[t] == "C" else I2C)},
                              out=f"{t}:{'nhwc' if tag == 'I' else 'nchw'}")
                self.report["layout_transposes"] += 1
                self.onnx[(t, tag)] = r
            else:
                raise AssertionError(f"tensor {t} in tag {tag} was not written")
        return r

    def cdata(self, name: str) -> np.ndarray:
        c = self.consts[name]
        if c.dtype == "bit":
            return unpack_bits(c.data, c.shape[-1]).reshape(c.shape)
        return np.asarray(c.data)

    def cinit(self, name: str, key: str, arr) -> str:
        return self.s.init(f"{name}{key}", arr)

    def c_f32(self, name: str, shape=None, key: str = "") -> str:
        a = np.asarray(self.cdata(name), np.float32)
        if shape is not None:
            a = a.reshape(shape)
        return self.cinit(name, key, a)

    def c_pm1(self, name: str, arr: np.ndarray, key: str) -> str:
        """±1 / ternary weights: an INT8 initializer read through Cast(to FLOAT) (a constant-only node)."""
        i8 = self.cinit(name, key + ":i8", np.asarray(arr, np.int8))
        out = f"{name}{key}"
        if out not in self._casts:
            self._casts.add(out)
            self.s.node("Cast", [i8], {"to": FLOAT}, out=out)
        return out

    def c_ternary(self, name: str, arr: np.ndarray, key: str) -> str:
        """{-1, 0, +1} weights on a binarized input (w_zero): IntQuant(Cast(INT8), scale 1, zero point 0, 2 bits,
        signed, narrow, ROUND), the canonical ternary form of docs/qonnx-profile.md R4 (constant-only nodes)."""
        src = self.c_pm1(name, arr, key + ":t")
        out = f"{name}{key}"
        if out not in self._casts:
            self._casts.add(out)
            s = self.s
            s.node("IntQuant", [src, self.ONE, self.ZERO, s.init("bitino/two", np.array(2.0, np.float32))],
                   {"signed": 1, "narrow": 1, "rounding_mode": "ROUND"}, out=out, domain=QD)
        return out

    def carrier(self, tag: str, v: float) -> str:
        """A float64 attribute value as a DOUBLE scalar initializer read through Cast(to FLOAT) (R7)."""
        key = f"bitino/f64/{float(v).hex()}"
        if key + ":f32" not in self._casts:
            self._casts.add(key + ":f32")
            d = self.s.init(key, np.array(float(v), np.float64))
            self.s.node("Cast", [d], {"to": FLOAT}, out=key + ":f32")
            self.report["carriers"] += 1
        return key + ":f32"

    def scalar(self, v: float) -> str:
        return self.s.init(f"bitino/c/{float(v).hex()}", np.array(v, np.float32))

    def write(self) -> Spec:
        self.plan()
        self.onnx: dict = {}
        self._casts: set = set()
        s = self.s
        for i in self.g.inputs:
            et = {"f32": FLOAT, "i32": INT32}[i["dtype"]]
            shp = list(i["shape"])
            if self.prim[i["name"]] == "C":
                shp = [shp[0], shp[3], shp[1], shp[2]]
            s.inputs.append((i["name"], et, shp))
            self.onnx[(i["name"], self.prim[i["name"]])] = i["name"]
        for idx, n in enumerate(self.g.nodes):
            self.cur = (n, idx)
            self.report["groups"] += 1
            self.report["groups_by_op"][n.op] = self.report["groups_by_op"].get(n.op, 0) + 1
            first = len(s.nodes)
            fn = getattr(self, f"w_{n.op}", None)
            if fn is None:
                raise AssertionError(f"{n.name}: IR operator {n.op} has no step-3 pattern")
            out, tag, canon = fn(n)
            if out != n.outputs[0]:
                raise AssertionError(f"{n.name}: group output tensor {out}")
            self.onnx[(n.outputs[0], tag)] = out
            for want in sorted(self.need.get(n.outputs[0], ())):
                if want != tag:
                    r = self.emit("Transpose", [out], {"perm": list(C2I if tag == "C" else I2C)},
                                  out=f"{out}:{'nhwc' if want == 'I' else 'nchw'}")
                    self.report["layout_transposes"] += 1
                    self.onnx[(n.outputs[0], want)] = r
            # group metadata on the group's first named node
            gm = {}
            consts = {str(j): c for j, c in enumerate(n.inputs) if c and c in self.consts}
            if consts:
                gm[CONSTS_KEY] = json.dumps(consts, sort_keys=True)
            am = attrs_meta(n.op, n.attrs, canon)
            if am:
                gm[ATTRS_KEY] = json.dumps(am, sort_keys=True)
                self.report["attrs_meta"] += 1
            ir = {"op": n.op}
            na = len(n.inputs)
            trimmed = len(n.inputs)
            while trimmed and not n.inputs[trimmed - 1]:
                trimmed -= 1
            if na != trimmed:
                ir["n_inputs"] = na
            lay = {str(j): self.consts[c].layout for j, c in enumerate(n.inputs) if c in self.consts
                   and self.consts[c].layout != DEFAULT_LAYOUTS.get((n.op, j), "")}
            if lay:
                ir["layouts"] = lay
            gm[IR_KEY] = json.dumps(ir, sort_keys=True)
            target = next(nd for nd in s.nodes[first:] if nd["meta"].get(NAME_KEY) == n.name)
            target["meta"].update(gm)
        for o in self.g.outputs:
            s.outputs.append((self.onnx[(o, "I")], FLOAT, list(self.shapes[o])))
        return s

    # ---------------------------------------------------------------- binarizers
    def binarize(self, x: str, mode: str, thr: str | None, cshape) -> str:
        """Bin(x [- t]) of docs/qonnx-profile.md R1: BipolarQuant (ge), Sign (sign_zero), Where(Greater) (gt),
        Relu(Sign) (gt0); a threshold is a Sub before the binarizer (exact)."""
        if thr:
            x = self.emit("Sub", [x, self.c_f32(thr, cshape, "" if cshape is None else ":c11")])
        if mode == "ge":
            return self.emit("BipolarQuant", [x, self.ONE], domain=QD)
        if mode == "sign_zero":
            return self.emit("Sign", [x])
        if mode == "gt":
            return self.emit("Where", [self.emit("Greater", [x, self.ZERO]), self.ONE, self.M1])
        if mode == "gt0":
            return self.emit("Relu", [self.emit("Sign", [x])])
        raise AssertionError(f"binarize mode {mode}")

    # ---------------------------------------------------------------- operators (each returns out, tag, canon attrs)
    def w_Conv2D(self, n):
        x, w = n.inputs[0], n.inputs[1]
        b = n.inputs[2] if len(n.inputs) > 2 and n.inputs[2] else ""
        a = n.attrs
        if float(a.get("pad_value", 0.0)) != 0.0:
            raise AssertionError(f"{n.name}: float Conv2D with pad_value {a['pad_value']}")
        wd = self.cdata(w)
        o, kh, kw, ci = wd.shape
        wt = self.cinit(w, ":oihw", np.ascontiguousarray(np.transpose(wd, (0, 3, 1, 2))).astype(np.float32))
        xc = self.rt(x, "C")
        pads = list(a.get("pads", [0, 0, 0, 0]))
        prod = getattr(self, "_prod", None)
        if prod is None:
            prod = self._prod = {o_: q for q in self.g.nodes for o_ in q.outputs}
        p_ = prod.get(x)
        on_bin = p_ is not None and p_.op == "Activation" and p_.attrs.get("type") in ("sign", "sign_zero")
        extra = None
        if on_bin:
            # step 6: a float conv on a binarized tensor (BNAS, the convs the first-release BinOp leaves float): R2 explicit
            # binary padding (Pad(0), the IR's pad_value 0) and the declared float layer (§ 7 bitino.float_layer)
            if any(pads):
                t_, b2, l_, r_ = pads
                xc = self.emit("Pad", [xc, self.s.init(f"bitino/pads/{t_}_{l_}_{b2}_{r_}", np.array([0, 0, t_, l_, 0, 0, b2, r_], np.int64)),
                                       self.ZERO], {"mode": "constant"})
            extra = {"bitino.float_layer": "1"}
        ins = [xc, wt] + ([self.c_f32(b)] if b else [])
        act = a.get("activation", "none")
        dil = list(a.get("dilations", [1, 1]))
        self.report["onnx_nodes"] += 1
        y = self.s.node("Conv", ins, {"kernel_shape": [kh, kw], "strides": list(a.get("strides", [1, 1])),
                                      "pads": [0, 0, 0, 0] if on_bin else _onnx_pads(pads), "dilations": dil,
                                      "group": int(a.get("groups", 1))}, out=n.outputs[0] if act == "none" else None,
                        meta=self.meta(extra))
        if act == "relu":
            y = self.emit("Relu", [y], out=n.outputs[0])
        elif act != "none":
            raise AssertionError(f"{n.name}: activation {act}")
        canon = {"strides": list(a.get("strides", [1, 1])), "pads": list(a.get("pads", [0, 0, 0, 0])), "dilations": dil,
                 "groups": int(a.get("groups", 1)), "pad_value": 0.0, "activation": act}
        return y, "C", canon

    def w_BConv2D(self, n):
        x, w, pm, pa, thr = (list(n.inputs) + [""] * 5)[:5]
        a = n.attrs
        if a.get("output", "f32") != "f32":
            raise AssertionError(f"{n.name}: bit output")
        mode = a["input_binarize"]
        c_in = self.shapes[x][3]
        xb = self.binarize(self.rt(x, "C"), mode, thr, (c_in, 1, 1))
        pads = list(a.get("pads", [0, 0, 0, 0]))
        pv = float(a.get("pad_value", 0.0))
        if any(pads):
            t, b_, l, r = pads
            xb = self.emit("Pad", [xb, self.s.init(f"bitino/pads/{t}_{l}_{b_}_{r}", np.array([0, 0, t, l, 0, 0, b_, r], np.int64)),
                                   self.scalar(pv)], {"mode": "constant"})
        # pv != 0.0 without padding: informational (no tap reads it), carried by bitino.attrs (_informational)
        bits = self.cdata(w).astype(np.int8)                      # OHWI
        o, kh, kw, ci = bits.shape
        wt = self.c_pm1(w, np.ascontiguousarray(np.transpose(bits, (0, 3, 1, 2))), ":oihw")
        act = a.get("activation", "none")
        last = not pm and not pa and act == "none"
        y = self.emit("Conv", [xb, wt], {"kernel_shape": [kh, kw], "strides": list(a["strides"]), "pads": [0, 0, 0, 0],
                                         "dilations": list(a["dilations"]), "group": int(a["groups"])},
                      out=n.outputs[0] if last else None)
        if pm:
            y = self.emit("Mul", [y, self.c_f32(pm, (o, 1, 1), ":c11")], out=n.outputs[0] if not pa and act == "none" else None)
        if pa:
            y = self.emit("Add", [y, self.c_f32(pa, (o, 1, 1), ":c11")], out=n.outputs[0] if act == "none" else None)
        if act == "relu":
            y = self.emit("Relu", [y], out=n.outputs[0])
        elif act != "none":
            raise AssertionError(f"{n.name}: activation {act}")
        canon = {"strides": list(a["strides"]), "pads": pads, "dilations": list(a["dilations"]), "groups": int(a["groups"]),
                 "pad_value": pv if any(pads) else 0.0, "input_binarize": mode, "activation": act, "output": "f32"}
        return y, "C", canon

    def w_BDense(self, n):
        x, w, pm, pa, thr, wz = (list(n.inputs) + [""] * 6)[:6]
        mode = n.attrs["input_binarize"]
        xb = self.binarize(self.rt(x, "I"), mode, thr, None)
        bits = self.cdata(w).astype(np.int8)                      # [O, K]
        if wz:
            z = self.cdata(wz)
            if np.any(bits[z < 0] != 1):
                raise AssertionError(f"{n.name}: w_sign bits at zero weights are not +1")
            bits = np.where(z < 0, 0, bits).astype(np.int8)
        wt = (self.c_ternary if wz else self.c_pm1)(w, np.ascontiguousarray(bits.T), ":ko")
        act = n.attrs.get("activation", "none")
        if act not in ("none", "relu"):
            raise AssertionError(f"{n.name}: activation {act}")
        y = self.emit("MatMul", [xb, wt], out=n.outputs[0] if not pm and not pa and act == "none" else None)
        if pm:
            y = self.emit("Mul", [y, self.c_f32(pm)], out=n.outputs[0] if not pa and act == "none" else None)
        if pa:
            y = self.emit("Add", [y, self.c_f32(pa)], out=n.outputs[0] if act == "none" else None)
        if act == "relu":
            y = self.emit("Relu", [y], out=n.outputs[0])
        return y, "I", {"input_binarize": mode, **({"activation": act} if act != "none" else {})}

    def _bm_operand(self, t: str, mode: str) -> str:
        if t in self.consts:
            src = self.c_f32(t)
            return self.binarize(src, mode, None, None)
        return self.binarize(self.rt(t, "I"), mode, None, None)

    def _swap_last(self, t: str, rank: int) -> str:
        perm = list(range(rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        return self.emit("Transpose", [t], {"perm": perm})

    def w_BMatMul(self, n):
        a_, b_ = n.inputs
        at = n.attrs
        A = self._bm_operand(a_, at["a_binarize"])
        B = self._bm_operand(b_, at["b_binarize"])
        tb = int(at.get("transpose_b", 0))
        if tb:
            B = self._swap_last(B, self.rank(b_))
        sc = float(at.get("scale", 1.0))
        y = self.emit("MatMul", [A, B], out=n.outputs[0] if sc == 1.0 else None)
        if sc != 1.0:
            y = self.emit("Mul", [y, self.carrier("scale", sc)], out=n.outputs[0])
        return y, "I", {"a_binarize": at["a_binarize"], "b_binarize": at["b_binarize"], "scale": sc, "transpose_b": tb}

    def w_MatMul(self, n):
        a_, b_ = n.inputs
        at = n.attrs
        A = self.rt(a_, "I") if a_ not in self.consts else self.c_f32(a_)
        B = self.rt(b_, "I") if b_ not in self.consts else self.c_f32(b_)
        tb = int(at.get("transpose_b", 0))
        if tb:
            B = self._swap_last(B, self.rank(b_))
        sc = float(at.get("scale", 1.0))
        y = self.emit("MatMul", [A, B], out=n.outputs[0] if sc == 1.0 else None)
        if sc != 1.0:
            y = self.emit("Mul", [y, self.carrier("scale", sc)], out=n.outputs[0])
        return y, "I", {"scale": sc, "transpose_b": tb}

    def w_Dense(self, n):
        x, w = n.inputs[0], n.inputs[1]
        b = n.inputs[2] if len(n.inputs) > 2 and n.inputs[2] else ""
        act = n.attrs.get("activation", "none")
        wd = np.asarray(self.cdata(w), np.float32)           # [O, K]
        last = act == "none"
        if self.rank(x) == 2:
            ins = [self.rt(x, "I"), self.c_f32(w)] + ([self.c_f32(b)] if b else [])
            y = self.emit("Gemm", ins, {"transB": 1}, out=n.outputs[0] if last else None)
        else:
            y = self.emit("MatMul", [self.rt(x, "I"), self.cinit(w, ":ko", np.ascontiguousarray(wd.T))],
                          out=n.outputs[0] if last and not b else None)
            if b:
                y = self.emit("Add", [y, self.c_f32(b)], out=n.outputs[0] if last else None)
        if act == "relu":
            y = self.emit("Relu", [y], out=n.outputs[0])
        elif act != "none":
            raise AssertionError(f"{n.name}: activation {act}")
        return y, "I", {"activation": act}

    def _vec(self, c: str, mode: str) -> str:
        """A per-channel constant in the node's mode: verbatim (I), [C, 1, 1] (C, 1-D) or NCHW (C, 4-D)."""
        if mode == "I":
            return self.c_f32(c)
        d = np.asarray(self.cdata(c), np.float32)
        if d.ndim == 1:
            return self.cinit(c, ":c11", d.reshape(-1, 1, 1))
        if d.ndim == 4:
            return self.cinit(c, ":nchw", np.ascontiguousarray(np.transpose(d, I2C)))
        raise AssertionError(f"constant {c} of rank {d.ndim} in an NCHW group")

    def w_ChannelAffine(self, n):
        x, sc, sh = (list(n.inputs) + ["", ""])[:3]
        m = self.mode[n.name]
        y = self.rt(x, m)
        if not sc and not sh:
            raise AssertionError(f"{n.name}: ChannelAffine without constants")
        if sc:
            y = self.emit("Mul", [y, self._vec(sc, m)], out=n.outputs[0] if not sh else None)
        if sh:
            y = self.emit("Add", [y, self._vec(sh, m)], out=n.outputs[0])
        return y, m, {}

    def w_Activation(self, n):
        t = n.attrs["type"]
        m = self.mode[n.name]
        x = self.rt(n.inputs[0], m)
        o = n.outputs[0]
        if t == "relu":
            y = self.emit("Relu", [x], out=o)
        elif t == "sign_zero":
            y = self.emit("Sign", [x], out=o)
        elif t == "sign":       # step 6: a standalone sign(0) = +1 binarizer (the FINN-style literal mirrors)
            y = self.emit("BipolarQuant", [x, self.ONE], out=o, domain=QD)
        elif t == "tanh":
            y = self.emit("Tanh", [x], out=o)
        elif t == "sigmoid":
            y = self.emit("Sigmoid", [x], out=o)
        elif t == "prelu":
            y = self.emit("PRelu", [x, self._vec(n.inputs[1], m)], out=o)
        elif t == "gelu":       # x * 0.5 * (1 + erf(x / sqrt 2)), the opset-17 form (docs/qonnx-profile.md § 4.3)
            e = self.emit("Erf", [self.emit("Div", [x, self.carrier("sqrt2", math.sqrt(2.0))])])
            y = self.emit("Mul", [self.emit("Mul", [x, self.scalar(0.5)]), self.emit("Add", [e, self.ONE])], out=o)
        else:
            raise AssertionError(f"{n.name}: activation {t}")
        return y, m, {"type": t}

    def _binary(self, n, op):
        m = self.mode[n.name]
        ins = [(self._vec(i, m) if i in self.consts else self.rt(i, m)) for i in n.inputs]
        return self.emit(op, ins, out=n.outputs[0]), m, {}

    def w_Add(self, n):
        return self._binary(n, "Add")

    def w_AddBroadcast(self, n):
        return self._binary(n, "Add")

    def w_Mul(self, n):
        return self._binary(n, "Mul")

    def w_Concat(self, n):
        m = self.mode[n.name]
        ax = int(n.attrs.get("axis", -1))
        ins = [(self.c_f32(i) if i in self.consts else self.rt(i, m)) for i in n.inputs]
        y = self.emit("Concat", ins, {"axis": 1 if m == "C" else ax}, out=n.outputs[0])
        return y, m, {"axis": ax}

    def w_Reshape(self, n):
        shp = [int(d) for d in n.attrs["shape"]]
        c = self.s.init(f"bitino/shape/{'_'.join(map(str, shp))}", np.array(shp, np.int64))
        return self.emit("Reshape", [self.rt(n.inputs[0], "I"), c], out=n.outputs[0]), "I", {"shape": shp}

    def w_Transpose(self, n):
        perm = [int(p) for p in n.attrs["perm"]]
        return self.emit("Transpose", [self.rt(n.inputs[0], "I")], {"perm": perm}, out=n.outputs[0]), "I", {"perm": perm}

    def w_Gather(self, n):
        tab, ids = n.inputs[0], n.inputs[1]
        scale = n.inputs[2] if len(n.inputs) > 2 and n.inputs[2] else ""
        zm = n.inputs[3] if len(n.inputs) > 3 and n.inputs[3] else ""
        idx = self.c_int(ids) if ids in self.consts else self.rt(ids, "I")
        if tab in self.consts and self.consts[tab].dtype == "bit":
            bits = self.cdata(tab).astype(np.int8)
            if zm:
                z = self.cdata(zm)
                if np.any(bits[z < 0] != 1):
                    raise AssertionError(f"{n.name}: sign bits at zero rows are not +1")
                bits = np.where(z < 0, 0, bits).astype(np.int8)
            t = self.c_pm1(tab, bits, ":tab")
            rows = self.emit("Gather", [t, idx], {"axis": 0}, out=n.outputs[0] if not scale else None)
            if scale:
                sc = self.emit("Gather", [self.c_f32(scale), idx], {"axis": 0})
                ax = self.s.init("bitino/axes/-1", np.array([-1], np.int64))
                rows = self.emit("Mul", [rows, self.emit("Unsqueeze", [sc, ax])], out=n.outputs[0])
            return rows, "I", {}
        if scale or zm:
            raise AssertionError(f"{n.name}: scaled Gather of a float table")
        src = self.c_f32(tab) if tab in self.consts else self.rt(tab, "I")
        return self.emit("Gather", [src, idx], {"axis": 0}, out=n.outputs[0]), "I", {}

    def c_int(self, name: str) -> str:
        c = self.consts[name]
        if c.dtype != "i32":
            raise AssertionError(f"index constant {name} of dtype {c.dtype}")
        return self.cinit(name, "", np.asarray(c.data, np.int32))

    def w_LayerNorm(self, n):
        x, gm, bt = (list(n.inputs) + ["", ""])[:3]
        if not gm or not bt:
            raise AssertionError(f"{n.name}: LayerNorm without gamma / beta")
        eps = float(n.attrs.get("eps", 1e-5))
        y = self.emit("LayerNormalization", [self.rt(x, "I"), self.c_f32(gm), self.c_f32(bt)],
                      {"axis": -1, "epsilon": float(np.float32(eps))}, out=n.outputs[0])
        return y, "I", {"eps": float(np.float32(eps))}

    def w_Softmax(self, n):
        return self.emit("Softmax", [self.rt(n.inputs[0], "I")], {"axis": -1}, out=n.outputs[0]), "I", {}

    def w_ReduceMean(self, n):
        ax, kd = int(n.attrs.get("axis", -1)), int(n.attrs.get("keepdims", 0))
        y = self.emit("ReduceMean", [self.rt(n.inputs[0], "I")], {"axes": [ax], "keepdims": kd}, out=n.outputs[0])
        return y, "I", {"axis": ax, "keepdims": kd}

    def w_GlobalAvgPool(self, n):
        kd = int(n.attrs.get("keepdims", 0))
        x = self.rt(n.inputs[0], "C")
        if kd:
            return self.emit("GlobalAveragePool", [x], out=n.outputs[0]), "C", {"keepdims": 1}
        y = self.emit("Flatten", [self.emit("GlobalAveragePool", [x])], {"axis": 1}, out=n.outputs[0])
        return y, "I", {"keepdims": 0}

    def _pool(self, n, op):
        a = n.attrs
        at = {"kernel_shape": list(a["kernel"]), "strides": list(a.get("strides", a["kernel"])),
              "pads": _onnx_pads(a.get("pads", [0, 0, 0, 0]))}
        canon = {"kernel": list(a["kernel"]), "strides": list(a.get("strides", a["kernel"])), "pads": list(a.get("pads", [0, 0, 0, 0]))}
        if op == "AveragePool":
            at["count_include_pad"] = int(a.get("count_include_pad", 0))
            canon["count_include_pad"] = at["count_include_pad"]
        return self.emit(op, [self.rt(n.inputs[0], "C")], at, out=n.outputs[0]), "C", canon

    def w_Pad(self, n):
        """IR Pad (pads = [begin_0, end_0, begin_1, end_1, ...] per IR axis, constant value; step 6: AdaBin's channel
        padding of the shortcut) -> ONNX Pad(mode constant) on the tensor in IR layout (data movement, B)."""
        p = [int(v) for v in n.attrs["pads"]]
        r = len(p) // 2
        po = [p[2 * d] for d in range(r)] + [p[2 * d + 1] for d in range(r)]
        P = self.s.init(f"bitino/pads/{'_'.join(map(str, po))}", np.array(po, np.int64))
        v = float(n.attrs.get("value", 0.0))
        y = self.emit("Pad", [self.rt(n.inputs[0], "I"), P, self.scalar(v)], {"mode": "constant"}, out=n.outputs[0])
        return y, "I", {"pads": p, "value": v}

    def w_MaxPool2D(self, n):
        return self._pool(n, "MaxPool")

    def w_AvgPool2D(self, n):
        return self._pool(n, "AveragePool")

    def w_FakeQuant(self, n):
        x, s_, z_ = n.inputs
        a = n.attrs
        r = self.rank(x)
        ax = int(a.get("axis", -1)) % r
        sv = np.asarray(self.cdata(s_), np.float32).reshape(-1)
        zv = np.asarray(self.cdata(z_), np.float32).reshape(-1)
        if sv.size != zv.size or (sv.size > 1 and self.shapes[x][ax] != sv.size):
            raise AssertionError(f"{n.name}: FakeQuant constants {sv.size} / {zv.size} along axis {ax}")
        shp = [1] * r
        shp[ax] = sv.size
        canon_axis = -1 if (ax == r - 1 or sv.size == 1) else ax
        S = self.cinit(s_, ":b", sv.reshape(shp))
        Z = self.cinit(z_, ":b", zv.reshape(shp))
        pre = float(a.get("pre_scale", 1.0))
        t = self.emit("Mul", [self.rt(x, "I"), self.carrier("pre_scale", pre)])
        t = self.emit("Add", [self.emit("Div", [t, S]), Z])
        q = self.emit("Clip", [self.emit("Round", [t]), self.scalar(float(a["qmin"])), self.scalar(float(a["qmax"]))])
        y = self.emit("Mul", [q, S], out=n.outputs[0])
        return y, "I", {"axis": canon_axis, "pre_scale": pre, "qmin": int(a["qmin"]), "qmax": int(a["qmax"])}

    def w_ChannelGate(self, n):
        x, gate = n.inputs
        if n.attrs.get("gate", "sigmoid") != "sigmoid":
            raise AssertionError(f"{n.name}: gate {n.attrs}")
        xs, gs = self.shapes[x], self.shapes[gate]
        if len(gs) == 4:
            g = self.rt(gate, "C")
        elif len(gs) == 2:
            g = self.emit("Reshape", [self.rt(gate, "I"), self.s.init(f"bitino/shape/{gs[0]}_{gs[1]}_1_1",
                                                                     np.array([gs[0], gs[1], 1, 1], np.int64))])
        else:
            raise AssertionError(f"{n.name}: gate shape {gs}")
        sg = self.emit("Cast", [self.emit("Sigmoid", [self.emit("Cast", [g], {"to": DOUBLE})])], {"to": FLOAT})
        return self.emit("Mul", [self.rt(x, "C"), sg], out=n.outputs[0]), "C", {"gate": "sigmoid"}

    # ---------------------------------------------------------------- image features (docs/qonnx-profile.md § 4.5)
    def w_ImageGrayU8(self, n):
        a = n.attrs
        coeffs = [int(v) for v in a["coeffs"]]
        shift = int(a["shift"])
        sc = float(a.get("scale", 255.0))
        x = self.rt(n.inputs[0], "C")
        v = self.emit("Mul", [x, self.scalar(float(np.float32(sc)))])
        v = self.emit("Where", [self.emit("IsNaN", [v]), self.ZERO, v])
        u = self.emit("Cast", [self.emit("Floor", [self.emit("Clip", [v, self.ZERO, self.scalar(255.0)])])], {"to": INT32})
        k = self.s.init(f"{n.name}/coeffs:i32", np.array(coeffs, np.int32).reshape(1, 3, 1, 1))
        sm = self.emit("ReduceSum", [self.emit("Mul", [u, k]), self.s.init("bitino/axes/1", np.array([1], np.int64))], {"keepdims": 1})
        sm = self.emit("Add", [sm, self.s.init(f"{n.name}/round:i32", np.array(1 << (shift - 1), np.int32))])
        sh = self.emit("BitShift", [self.emit("Cast", [sm], {"to": UINT32}),
                                    self.s.init(f"{n.name}/shift:u32", np.array(shift, np.uint32))], {"direction": "RIGHT"})
        y = self.emit("Cast", [sh], {"to": FLOAT}, out=n.outputs[0])
        return y, "C", {"coeffs": coeffs, "scale": float(np.float32(sc)), "shift": shift}

    def w_FftLogMagnitude(self, n):
        a = n.attrs
        eps, mul, div, shift = float(a.get("eps", 1e-9)), float(a.get("mul", 20.0)), float(a.get("div", 255.0)), int(a.get("shift", 1))
        _, h, w, c = self.shapes[n.inputs[0]]
        if c != 1:
            raise AssertionError(f"{n.name}: FftLogMagnitude of {c} channels")
        ch, sh_ = dft_tables(h)
        cw, sw = dft_tables(w)
        CH = self.s.init(f"bitino/dft/cos{h}", ch)
        SH = self.s.init(f"bitino/dft/sin{h}", sh_)
        CW = self.s.init(f"bitino/dft/cos{w}", cw)
        SW = self.s.init(f"bitino/dft/sin{w}", sw)
        X = self.emit("Cast", [self.rt(n.inputs[0], "C")], {"to": DOUBLE})
        a1 = self.emit("MatMul", [CH, X])
        a2 = self.emit("MatMul", [SH, X])
        re = self.emit("Sub", [self.emit("MatMul", [a1, CW]), self.emit("MatMul", [a2, SW])])
        im = self.emit("Neg", [self.emit("Add", [self.emit("MatMul", [a2, CW]), self.emit("MatMul", [a1, SW])])])
        mag = self.emit("Sqrt", [self.emit("Add", [self.emit("Mul", [re, re]), self.emit("Mul", [im, im])])])
        if shift:
            mag = self.emit("Gather", [mag, self.s.init(f"bitino/fftshift/{h}", fftshift_index(h))], {"axis": 2})
            mag = self.emit("Gather", [mag, self.s.init(f"bitino/fftshift/{w}", fftshift_index(w))], {"axis": 3})
        d = lambda v: self.s.init(f"bitino/d/{float(v).hex()}", np.array(v, np.float64))   # noqa: E731
        lg = self.emit("Log", [self.emit("Add", [mag, d(eps)])])
        y = self.emit("Cast", [self.emit("Div", [self.emit("Mul", [lg, d(mul)]), d(div)])], {"to": FLOAT}, out=n.outputs[0])
        return y, "C", {"eps": eps, "mul": mul, "div": div, "shift": shift}

    def w_LocalBinaryPattern(self, n):
        a = n.attrs
        rp, cp = [float(v) for v in a["rp"]], [float(v) for v in a["cp"]]
        method, div = a.get("method", "uniform"), float(a.get("div", 1.0))
        _, h, w, c = self.shapes[n.inputs[0]]
        if c != 1 or method not in ("uniform", "default"):
            raise AssertionError(f"{n.name}: LocalBinaryPattern {c} channels / {method}")
        for v in rp:
            if not lbp_shift_ok(v, h):
                raise AssertionError(f"{n.name}: row offset {v} violates the uniform-shift condition (Q12)")
        for v in cp:
            if not lbp_shift_ok(v, w):
                raise AssertionError(f"{n.name}: column offset {v} violates the uniform-shift condition (Q12)")
        d = lambda v: self.s.init(f"bitino/d/{float(v).hex()}", np.array(v, np.float64))   # noqa: E731
        X = self.emit("Cast", [self.rt(n.inputs[0], "C")], {"to": DOUBLE})
        fr = [math.floor(v) for v in rp] + [math.ceil(v) for v in rp]
        fc = [math.floor(v) for v in cp] + [math.ceil(v) for v in cp]
        pt, pb, pl, pr = max(0, -min(fr)), max(0, max(fr)), max(0, -min(fc)), max(0, max(fc))
        Xp = self.emit("Pad", [X, self.s.init(f"{n.name}/pads", np.array([0, 0, pt, pl, 0, 0, pb, pr], np.int64)), d(0.0)],
                       {"mode": "constant"})
        shifted: dict = {}

        def V(dy, dx):
            if (dy, dx) not in shifted:
                st = self.s.init(f"{n.name}/starts/{dy}_{dx}", np.array([pt + dy, pl + dx], np.int64))
                en = self.s.init(f"{n.name}/ends/{dy}_{dx}", np.array([pt + dy + h, pl + dx + w], np.int64))
                shifted[(dy, dx)] = self.emit("Slice", [Xp, st, en, self.s.init("bitino/axes/2_3", np.array([2, 3], np.int64))])
            return shifted[(dy, dx)]
        ridx = self.s.init(f"bitino/index/rows{h}", np.arange(h, dtype=np.float64).reshape(h, 1))
        cidx = self.s.init(f"bitino/index/cols{w}", np.arange(w, dtype=np.float64).reshape(1, w))
        bits = []
        for i in range(len(rp)):
            rr = self.emit("Add", [ridx, d(rp[i])])
            dr = self.emit("Sub", [rr, self.emit("Floor", [rr])])
            cc = self.emit("Add", [cidx, d(cp[i])])
            dc = self.emit("Sub", [cc, self.emit("Floor", [cc])])
            omdc = self.emit("Sub", [d(1.0), dc])
            omdr = self.emit("Sub", [d(1.0), dr])
            r0, r1, c0, c1 = math.floor(rp[i]), math.ceil(rp[i]), math.floor(cp[i]), math.ceil(cp[i])
            top = self.emit("Add", [self.emit("Mul", [omdc, V(r0, c0)]), self.emit("Mul", [dc, V(r0, c1)])])
            bot = self.emit("Add", [self.emit("Mul", [omdc, V(r1, c0)]), self.emit("Mul", [dc, V(r1, c1)])])
            t = self.emit("Add", [self.emit("Mul", [omdr, top]), self.emit("Mul", [dr, bot])])
            bits.append(self.emit("GreaterOrEqual", [self.emit("Sub", [t, X]), d(0.0)]))
        i32 = lambda v: self.s.init(f"bitino/i32/{v}", np.array(v, np.int32))   # noqa: E731
        P = len(rp)
        if method == "uniform":
            ch = None
            for i in range(P - 1):
                xi = self.emit("Cast", [self.emit("Xor", [bits[i], bits[i + 1]])], {"to": INT32})
                ch = xi if ch is None else self.emit("Add", [ch, xi])
            ones = None
            for i in range(P):
                bi = self.emit("Cast", [bits[i]], {"to": INT32})
                ones = bi if ones is None else self.emit("Add", [ones, bi])
            lbp = self.emit("Where", [self.emit("LessOrEqual", [ch, i32(2)]), ones, i32(P + 1)])
        else:
            lbp = None
            for i in range(P):
                bi = self.emit("Mul", [self.emit("Cast", [bits[i]], {"to": INT32}), i32(1 << i)])
                lbp = bi if lbp is None else self.emit("Add", [lbp, bi])
        y = self.emit("Cast", [self.emit("Div", [self.emit("Cast", [lbp], {"to": DOUBLE}), d(div)])], {"to": FLOAT},
                      out=n.outputs[0])
        return y, "C", {"rp": rp, "cp": cp, "method": method, "div": div}


def first_use_order(g: Graph) -> list:
    order, seen = [], set()
    for n in g.nodes:
        for i in n.inputs:
            if i and i in g.constants and i not in seen:
                seen.add(i)
                order.append(i)
    return order + [k for k in g.constants if k not in seen]


def ir_shapes(g: Graph) -> dict:
    """Static shape of every IR tensor: one numpy-executor run on a zero input (ids 0)."""
    from . import executor
    feeds = {}
    for i in g.inputs:
        dt = {"f32": np.float32, "i32": np.int32}[i["dtype"]]
        feeds[i["name"]] = np.zeros(i["shape"], dt)
    shapes = {i["name"]: tuple(i["shape"]) for i in g.inputs}

    def hook(name, y):
        shapes[name] = tuple(np.shape(y))
        return y
    executor.run(g, feeds, post_hook=hook)
    return shapes


def write_spec(g: Graph, shapes: dict | None = None) -> tuple[Spec, dict]:
    """IR graph -> (spec, writer report). The model metadata (§ 6) is added by the caller; this function adds the
    import mode and, where the data section is not in first-use order, bitino.const_order."""
    shapes = shapes if shapes is not None else ir_shapes(g)
    w = GroupWriter(g, shapes)
    spec = w.write()
    spec.meta[IMPORT_KEY] = json.dumps({"mapping": MAPPING}, sort_keys=True)
    if first_use_order(g) != list(g.constants):
        spec.meta[CONST_ORDER_KEY] = json.dumps(list(g.constants))
    w.report["const_order"] = CONST_ORDER_KEY in spec.meta
    return spec, w.report


# ================================================================================================ spec I/O (numpy only)
def save_spec(spec: Spec, d: str) -> None:
    import os
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "spec.npz"), **{f"i{j}": v for j, v in enumerate(spec.inits.values())})
    json.dump({"nodes": spec.nodes, "init_names": list(spec.inits), "inputs": spec.inputs, "outputs": spec.outputs,
               "meta": spec.meta}, open(os.path.join(d, "spec.json"), "w"))


def assemble(d: str, out_path: str, producer: str, opset: int = 17, ir_version: int = 10, extra_meta: dict | None = None):
    """spec directory -> ONNX model file (onnx package). Returns the ModelProto."""
    import os

    import onnx
    from onnx import helper, numpy_helper
    sp = json.load(open(os.path.join(d, "spec.json")))
    z = np.load(os.path.join(d, "spec.npz"))
    inits = [numpy_helper.from_array(z[f"i{j}"], name) for j, name in enumerate(sp["init_names"])]
    nodes = []
    for nd in sp["nodes"]:
        n = helper.make_node(nd["op"], nd["inputs"], nd["outputs"], name=nd["name"], domain=nd["domain"] or None, **nd["attrs"])
        for k, v in nd["meta"].items():
            n.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
        nodes.append(n)
    ins = [helper.make_tensor_value_info(n, et, s) for n, et, s in sp["inputs"]]
    outs = [helper.make_tensor_value_info(n, et, s) for n, et, s in sp["outputs"]]
    meta = dict(sp["meta"])
    meta.update(extra_meta or {})
    graph = helper.make_graph(nodes, meta.get("bitino.model_id", "model"), ins, outs, inits)
    uses_qonnx = any(nd["domain"] == QD for nd in sp["nodes"])
    ops = [helper.make_opsetid("", opset)] + ([helper.make_opsetid(QD, 1)] if uses_qonnx else [])
    m = helper.make_model(graph, opset_imports=ops, producer_name=producer, producer_version="bitino-qonnx/1")
    m.ir_version = ir_version
    for k, v in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(m, out_path)
    return m


# ================================================================================================ importer
def _meta(nd, key):
    for p in getattr(nd, "metadata_props", ()):
        if p.key == key:
            return p.value
    return None


def import_mode(model) -> dict | None:
    raw = next((p.value for p in getattr(model, "metadata_props", ()) if p.key == IMPORT_KEY), None)
    if raw is None:
        return None
    obj = json.loads(raw)
    if isinstance(obj, dict) and obj.get("mapping") == "patterns":   # step 4: the pattern importer's options
        return None
    if not isinstance(obj, dict) or obj.get("mapping") != MAPPING or set(obj) - {"mapping"}:
        raise GroupImportError(IMPORT_KEY, f"unsupported import mode {raw!r} (expected {{\"mapping\": \"groups\"}})", "model")
    return obj


class _Group:
    def __init__(self, name: str, nodes: list):
        self.name = name
        self.nodes = nodes
        self.meta: dict = {}
        for nd in nodes:
            for p in nd.metadata_props:
                if p.key in (CONSTS_KEY, ATTRS_KEY, IR_KEY):
                    if p.key in self.meta and self.meta[p.key] != p.value:
                        raise GroupImportError(name, f"{p.key} given twice with different values")
                    self.meta[p.key] = p.value
        orders = {_meta(nd, ORDER_KEY) for nd in nodes}
        if len(orders) != 1 or None in orders:
            raise GroupImportError(name, f"bitino.order {sorted(map(str, orders))}: one value per group required")
        self.order = int(orders.pop())
        ir = json.loads(self.meta.get(IR_KEY, "{}"))
        if "op" not in ir:
            raise GroupImportError(name, "bitino.ir with the IR operator is required on group-mapped files")
        self.op = ir["op"]
        self.n_inputs = ir.get("n_inputs")
        self.layouts = {int(k): v for k, v in ir.get("layouts", {}).items()}
        self.consts = {int(k): v for k, v in json.loads(self.meta.get(CONSTS_KEY, "{}")).items()}
        self.attrs = json.loads(self.meta.get(ATTRS_KEY, "{}"))


class GroupImporter:
    def __init__(self, model):
        from onnx import numpy_helper
        self.model = model
        gr = model.graph
        self.inits = {t.name: numpy_helper.to_array(t) for t in gr.initializer}
        self.carrier: set = set()        # constant tensors that are float64 carriers (value kept in float64)
        self.cval: dict = dict(self.inits)
        self.prod = {}
        groups: dict = {}
        for nd in gr.node:
            for o in nd.output:
                self.prod[o] = nd
            name = _meta(nd, NAME_KEY)
            if name is None and nd.op_type == "IntQuant" and nd.domain == QD and all(i in self.cval for i in nd.input):
                # ternary weights: IntQuant(W, 1, 0, 2, signed, narrow, ROUND) of {-1, 0, +1} values is the identity
                from onnx import helper
                at = {a.name: helper.get_attribute_value(a) for a in nd.attribute}
                w, sc, zp, bw = (np.asarray(self.cval[i], np.float64) for i in nd.input)
                rm = at.get("rounding_mode", b"ROUND")
                rm = rm.decode() if isinstance(rm, bytes) else rm
                if not (sc.size == zp.size == bw.size == 1 and _f(sc) == 1.0 and _f(zp) == 0.0 and _f(bw) == 2.0
                        and int(at.get("signed", 1)) == 1 and int(at.get("narrow", 0)) == 1 and rm == "ROUND"
                        and np.all(np.isin(w, (-1.0, 0.0, 1.0)))):
                    raise GroupImportError(nd.name, "constant IntQuant: only the ternary weight form "
                                                    "IntQuant(W in {-1, 0, +1}, 1, 0, 2, signed, narrow, ROUND)", nd.op_type)
                self.cval[nd.output[0]] = w.astype(np.float32)
                continue
            if name is None:
                # constant-only preparation node: Cast of an initializer (float64 carrier, ±1 INT8 weights)
                if nd.op_type != "Cast" or nd.input[0] not in self.cval:
                    raise GroupImportError(nd.name, "an unnamed node that is not a Cast of a constant (group mapping: "
                                                    "every other node carries bitino.name)", nd.op_type)
                src = self.cval[nd.input[0]]
                to = int(next(a.i for a in nd.attribute if a.name == "to"))
                if to == FLOAT and src.dtype == np.float64:
                    self.cval[nd.output[0]] = src.astype(np.float64)
                    self.carrier.add(nd.output[0])
                else:
                    self.cval[nd.output[0]] = src.astype(_NP_OF[to])
                continue
            groups.setdefault(name, []).append(nd)
        self.groups = [_Group(k, v) for k, v in groups.items()]
        self.groups.sort(key=lambda gp: gp.order)
        if len({gp.order for gp in self.groups}) != len(self.groups):
            raise GroupImportError("bitino.order", "two groups with the same bitino.order")
        self.g = Graph()
        self.rt: dict = {}               # ONNX tensor -> (IR tensor, tag)
        self.oshape: dict = {}
        self.ir_const_src: dict = {}     # IR constant name -> (ONNX source, data) for consistency checks
        self.binarizers: list = []
        self.rules: dict = {}
        self.out_names = [o.name for o in gr.output]

    def rule(self, r: str) -> None:
        self.rules[r] = self.rules.get(r, 0) + 1

    # ---------------------------------------------------------------- shapes (static, from shape inference)
    def infer_shapes(self) -> None:
        """Static shapes by onnx shape inference, the QONNX quantizers read as shape-preserving (as check-qonnx)."""
        import copy

        import onnx
        m = copy.deepcopy(self.model)
        g = m.graph
        for k, nd in enumerate(list(g.node)):
            if nd.domain == QD and nd.op_type in ("BipolarQuant", "IntQuant"):
                g.node.remove(nd)
                g.node.insert(k, onnx.helper.make_node("Identity", [nd.input[0]], [nd.output[0]], name=nd.name))
        keep = [o for o in m.opset_import if (o.domain or "") in ("", "ai.onnx")]
        del m.opset_import[:]
        m.opset_import.extend(keep)
        m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
        for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
            t = vi.type.tensor_type
            self.oshape[vi.name] = tuple(int(d.dim_value) for d in t.shape.dim)
        for k, v in self.cval.items():
            self.oshape[k] = tuple(np.shape(v))
        del m

    # ---------------------------------------------------------------- IR constants
    def add_const(self, gp: _Group, slot: int, data: np.ndarray, dtype: str, src: str, layout: str | None = None) -> str:
        name = gp.consts.get(slot)
        if name is None:
            raise GroupImportError(gp.name, f"no bitino.consts name for IR input {slot}", gp.op)
        lay = gp.layouts.get(slot, DEFAULT_LAYOUTS.get((gp.op, slot), "")) if layout is None else layout
        if name in self.g.constants:
            c = self.g.constants[name]
            new = pack_bits(data) if dtype == "bit" else np.asarray(data, {"f32": np.float32, "i32": np.int32}[dtype])
            if c.dtype != dtype or c.layout != lay or tuple(c.shape) != tuple(np.shape(data)) or \
                    not np.array_equal(np.asarray(c.data), new):
                raise GroupImportError(gp.name, f"constant {name!r} used twice with different values / layouts", gp.op)
            return name
        self.g.add_constant(name, data, dtype=dtype, layout=lay)
        return name

    # ---------------------------------------------------------------- walk
    def run(self) -> Graph:
        gr = self.model.graph
        self.infer_shapes()
        for i in gr.input:
            if i.name in self.inits:
                continue
            et = i.type.tensor_type.elem_type
            shp = [int(d.dim_value) for d in i.type.tensor_type.shape.dim]
            if et == FLOAT:
                dt = "f32"
            elif et in (INT32, INT64):
                dt = "i32"
            else:
                raise GroupImportError(i.name, f"graph input element type {et}", "input")
            tag = "C" if (len(shp) == 4 and dt == "f32") else "I"
            ir_shape = [shp[0], shp[2], shp[3], shp[1]] if tag == "C" else shp
            self.g.add_input(i.name, ir_shape, dtype=dt)
            self.rt[i.name] = (i.name, tag)
        self.graph_inputs = {i.name for i in gr.input if i.name not in self.inits}
        for gp in self.groups:
            self.group(gp)
        outs = []
        for o in self.out_names:
            if o not in self.rt:
                raise GroupImportError(o, "graph output is not produced by a group", "output")
            t, tag = self.rt[o]
            if tag != "I" and len(self.oshape.get(o, ())) == 4:
                raise GroupImportError(o, "graph output in an NCHW view", "output")
            outs.append(t)
        self.g.set_outputs(*outs)
        return self.g

    def group(self, gp: _Group) -> None:
        nodes = list(gp.nodes)
        # graph-input layout conversions at the start of a consumer group
        rest = []
        for nd in nodes:
            if nd.op_type == "Transpose" and nd.input[0] in self.graph_inputs and self._is_conv(nd, self.rt[nd.input[0]][1]):
                ir, tag = self.rt[nd.input[0]]
                self.rt[nd.output[0]] = (ir, "I" if tag == "C" else "C")
                self.rule("layout Transpose (graph input)")
            else:
                rest.append(nd)
        post = None
        if len(rest) >= 2 and rest[-1].op_type == "Transpose" and rest[-1].input[0] == rest[-2].output[0]:
            post = rest.pop()
        reader = getattr(self, f"r_{gp.op}", None)
        if reader is None:
            raise GroupImportError(gp.name, f"no step-3 reader for IR operator {gp.op}", gp.op)
        self.cur = _View(self, gp, rest)
        ins, canon, out, tag = reader(self.cur)
        self.cur.check_all_used()
        attrs = apply_attrs_meta(gp.op, canon, gp.attrs, gp.name)
        if gp.n_inputs is not None:
            if gp.n_inputs < len(ins) and any(ins[gp.n_inputs:]):
                raise GroupImportError(gp.name, f"bitino.ir n_inputs {gp.n_inputs} drops non-empty inputs", gp.op)
            ins = (list(ins) + [""] * gp.n_inputs)[:gp.n_inputs]
        else:
            while ins and not ins[-1]:
                ins = ins[:-1]
        extra = set(gp.consts) - {j for j, i in enumerate(ins) if i and i in self.g.constants}
        if extra:
            raise GroupImportError(gp.name, f"bitino.consts names IR inputs {sorted(extra)} that are not constants", gp.op)
        y = self.g.add_node(gp.op, ins, attrs, name=gp.name)
        if out not in self.rest_outputs(rest):
            raise GroupImportError(gp.name, f"pattern output {out} is not produced by the group", gp.op)
        self.rt[out] = (y, tag)
        if post is not None:
            if post.input[0] != out or not self._is_conv(post, tag):
                raise GroupImportError(gp.name, "the group's last Transpose is neither the pattern nor a layout conversion "
                                                "of its output", gp.op)
            self.rt[post.output[0]] = (y, "I" if tag == "C" else "C")
            self.rule("layout Transpose (group output)")
        self.rule(f"group -> {gp.op}")

    @staticmethod
    def rest_outputs(rest) -> set:
        return {o for nd in rest for o in nd.output}

    def _is_conv(self, nd, tag) -> bool:
        perm = tuple(next((list(a.ints) for a in nd.attribute if a.name == "perm"), []))
        rank = len(self.oshape.get(nd.input[0], ()))
        return rank == 4 and ((tag == "C" and perm == C2I) or (tag == "I" and perm == I2C))

    # ================================================================ readers
    # each returns (IR inputs, canonical attrs, ONNX output tensor of the pattern, its tag)
    def _bin(self, V, t: str, cshape):
        """Bin(x [- t]) -> (x tensor or constant, mode, threshold array or None, pre tensor, binarizer node)."""
        nd = V.prod(t)
        if nd.op_type == "BipolarQuant":
            if V.const(nd.input[1]) is None or _f(V.const(nd.input[1])) != 1.0 or np.asarray(V.const(nd.input[1])).size != 1:
                raise V.err("BipolarQuant scale must be the scalar 1")
            mode, pre = "ge", nd.input[0]
        elif nd.op_type == "Sign":
            mode, pre = "sign_zero", nd.input[0]
        elif nd.op_type == "Where":
            cmp_ = V.prod(nd.input[0], "Greater")
            if not (V.is_const_val(nd.input[1], 1.0) and V.is_const_val(nd.input[2], -1.0) and V.is_const_val(cmp_.input[1], 0.0)):
                raise V.err("Where(Greater(x, 0), 1, -1) expected")
            mode, pre = "gt", cmp_.input[0]
        elif nd.op_type == "Relu":
            sg = V.prod(nd.input[0], "Sign")
            mode, pre = "gt0", sg.input[0]
        else:
            raise V.err(f"binarizer expected, got {nd.op_type}")
        x, thr = pre, None
        p = V.prod_or_none(pre)
        if p is not None and p.op_type == "Sub" and V.const(p.input[1]) is not None:
            x = p.input[0]
            thr = V.const(p.input[1])
            if cshape is not None:
                thr = np.asarray(thr).reshape(-1)
        return x, mode, thr, pre, nd

    def _record_bin(self, V, pre, mode, x_ir, thr_name, tag):
        self.binarizers.append({"onnx_node": V.gp.name, "pre": pre, "mode": mode, "ir_tensor": x_ir,
                                "ir_threshold": thr_name or None, "ir_mode": mode, "onnx_tag": tag})

    def _pm1(self, V, t: str, what: str) -> np.ndarray:
        w = V.const(t)
        if w is None:
            raise V.err(f"{what}: constant weights expected")
        w = np.asarray(w, np.float64)
        if not np.all(np.isin(w, (-1.0, 0.0, 1.0))):
            raise V.err(f"{what}: weights are not in {{-1, 0, +1}} (binary weights are never rounded)")
        return w.astype(np.int8)

    def r_Conv2D(self, V):
        conv = V.first("Conv")
        a = dict(V.attrs(conv))
        pad = V.prod_or_none(conv.input[0])
        if pad is not None and pad.op_type == "Pad" and V.in_group(pad):     # step 6: explicit Pad(0) of a binarized input
            po = [int(v) for v in np.asarray(V.const(pad.input[1])).reshape(-1)]
            val = float(np.asarray(V.const(pad.input[2])).reshape(()))
            if V.attrs(pad).get("mode", "constant") != "constant" or val != 0.0 or po[0] or po[1] or po[4] or po[5] \
                    or any(a.get("pads", [0, 0, 0, 0])):
                raise V.err("float Conv on a binarized tensor: Pad(0) of H / W and Conv pads 0 expected")
            V.use(pad)
            a["pads"] = [po[2], po[3], po[6], po[7]]
            x, xt = V.runtime(pad.input[0])
        else:
            x, xt = V.runtime(conv.input[0])
        if xt != "C":
            raise V.err("Conv input must be an NCHW view")
        w = V.const(conv.input[1])
        if w is None:
            raise V.err("Conv weights must be constant")
        ins = [x, self.add_const(V.gp, 1, np.ascontiguousarray(np.transpose(np.asarray(w, np.float32), (0, 2, 3, 1))), "f32", conv.input[1])]
        if len(conv.input) > 2 and conv.input[2]:
            ins.append(self.add_const(V.gp, 2, np.asarray(V.const(conv.input[2]), np.float32), "f32", conv.input[2]))
        V.use(conv)
        out, act = conv.output[0], "none"
        nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Relu":
            V.use(nxt)
            out, act = nxt.output[0], "relu"
        canon = {"strides": list(a.get("strides", [1, 1])), "pads": _ir_pads(a.get("pads", [0, 0, 0, 0])),
                 "dilations": list(a.get("dilations", [1, 1])), "groups": int(a.get("group", 1)), "pad_value": 0.0,
                 "activation": act}
        return ins, canon, out, "C"

    def r_BConv2D(self, V):
        conv = V.first("Conv")
        V.use(conv)
        a = V.attrs(conv)
        if any(a.get("pads", [])):
            raise V.err("binary Conv with implicit padding (canonical: explicit Pad, pads 0)")
        src = conv.input[0]
        pads_ir, pv = [0, 0, 0, 0], 0.0
        p = V.prod(src)
        if p.op_type == "Pad":
            V.use(p)
            pp = [int(v) for v in np.asarray(V.const(p.input[1])).reshape(-1)]
            if len(pp) != 8 or pp[0] or pp[1] or pp[4] or pp[5]:
                raise V.err(f"binary Pad {pp}: spatial padding only")
            pads_ir = [pp[2], pp[6], pp[3], pp[7]]
            pv = _f(V.const(p.input[2]))
            if pv not in (-1.0, 0.0, 1.0):
                raise V.err(f"binary pad value {pv}")
            src = p.input[0]
        x, mode, thr, pre, bnd = self._bin(V, src, True)
        V.use_chain(bnd, pre, x)
        xi, xt = V.runtime(x)
        if xt != "C":
            raise V.err("BConv2D input must be an NCHW view")
        w = self._pm1(V, conv.input[1], "BConv2D")
        if np.any(w == 0):
            raise V.err("BConv2D ternary weights")
        kb = self.add_const(V.gp, 1, np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1))), "bit", conv.input[1])
        o = w.shape[0]
        out = conv.output[0]
        ins = [xi, kb, "", "", ""]
        nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Mul" and V.const(nxt.input[1]) is not None:
            V.use(nxt)
            ins[2] = self.add_const(V.gp, 2, np.asarray(V.const(nxt.input[1]), np.float32).reshape(o), "f32", nxt.input[1])
            out = nxt.output[0]
            nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Add" and V.const(nxt.input[1]) is not None:
            V.use(nxt)
            ins[3] = self.add_const(V.gp, 3, np.asarray(V.const(nxt.input[1]), np.float32).reshape(o), "f32", nxt.input[1])
            out = nxt.output[0]
            nxt = V.next_of(out)
        act = "none"
        if nxt is not None and nxt.op_type == "Relu":
            V.use(nxt)
            out, act = nxt.output[0], "relu"
        if thr is not None:
            ins[4] = self.add_const(V.gp, 4, np.asarray(thr, np.float32).reshape(-1), "f32", "thr")
        self._record_bin(V, pre, mode, xi, ins[4], "C")
        canon = {"strides": list(a.get("strides", [1, 1])), "pads": pads_ir, "dilations": list(a.get("dilations", [1, 1])),
                 "groups": int(a.get("group", 1)), "pad_value": pv if any(pads_ir) else 0.0, "input_binarize": mode,
                 "activation": act, "output": "f32"}
        return ins, canon, out, "C"

    def r_BDense(self, V):
        mm = V.first("MatMul")
        V.use(mm)
        x, mode, thr, pre, bnd = self._bin(V, mm.input[0], None)
        V.use_chain(bnd, pre, x)
        xi, xt = V.runtime(x)
        if xt != "I":
            raise V.err("BDense input in an NCHW view")
        w = self._pm1(V, mm.input[1], "BDense")              # [K, O]
        wo = np.ascontiguousarray(w.T)
        ins = [xi, self.add_const(V.gp, 1, np.where(wo < 0, -1, 1).astype(np.int8), "bit", mm.input[1]), "", "", "", ""]
        if np.any(wo == 0):
            ins[5] = self.add_const(V.gp, 5, np.where(wo == 0, -1, 1).astype(np.int8), "bit", mm.input[1])
            self.rule("ternary weights {-1, 0, +1} -> w_sign + w_zero")
        out = mm.output[0]
        nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Mul" and V.const(nxt.input[1]) is not None:
            V.use(nxt)
            ins[2] = self.add_const(V.gp, 2, np.asarray(V.const(nxt.input[1]), np.float32), "f32", nxt.input[1])
            out = nxt.output[0]
            nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Add" and V.const(nxt.input[1]) is not None:
            V.use(nxt)
            ins[3] = self.add_const(V.gp, 3, np.asarray(V.const(nxt.input[1]), np.float32), "f32", nxt.input[1])
            out = nxt.output[0]
        act = "none"
        nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Relu":
            V.use(nxt)
            out, act = nxt.output[0], "relu"
        if thr is not None:
            ins[4] = self.add_const(V.gp, 4, np.asarray(thr, np.float32), "f32", "thr")
        self._record_bin(V, pre, mode, xi, ins[4], "I")
        return ins, {"input_binarize": mode, **({"activation": act} if act != "none" else {})}, out, "I"

    def _bm_operand(self, V, t: str, slot: int):
        x, mode, thr, pre, bnd = self._bin(V, t, None)
        if thr is not None:
            raise V.err("BMatMul operand with a threshold")
        V.use_chain(bnd, pre, x)
        c = V.const(x)
        if c is not None and x in self.cval:
            return self.add_const(V.gp, slot, np.asarray(c, np.float32), "f32", x), mode
        xi, xt = V.runtime(x)
        if xt != "I":
            raise V.err("BMatMul operand in an NCHW view")
        self._record_bin(V, pre, mode, xi, None, "I")
        return xi, mode

    def _matmul_core(self, V):
        mm = V.first("MatMul")
        V.use(mm)
        b_src = mm.input[1]
        tb = 0
        p = V.prod_or_none(b_src)
        if p is not None and p.op_type == "Transpose" and V.in_group(p):
            perm = list(V.attrs(p)["perm"])
            r = len(perm)
            want = list(range(r))
            want[-1], want[-2] = want[-2], want[-1]
            if perm != want:
                raise V.err(f"MatMul operand Transpose {perm} is not a swap of the last two axes")
            V.use(p)
            b_src, tb = p.input[0], 1
        out = mm.output[0]
        sc = 1.0
        nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Mul":
            c = V.carrier_val(nxt.input[1])
            if c is None or np.asarray(c).size != 1:
                raise V.err("MatMul scale must be a scalar float64 carrier")
            V.use(nxt)
            sc, out = float(np.asarray(c).reshape(-1)[0]), nxt.output[0]
        return mm, mm.input[0], b_src, tb, sc, out

    def r_BMatMul(self, V):
        mm, a_src, b_src, tb, sc, out = self._matmul_core(V)
        a, am = self._bm_operand(V, a_src, 0)
        b, bm = self._bm_operand(V, b_src, 1)
        return [a, b], {"a_binarize": am, "b_binarize": bm, "scale": sc, "transpose_b": tb}, out, "I"

    def r_MatMul(self, V):
        mm, a_src, b_src, tb, sc, out = self._matmul_core(V)
        ins = []
        for slot, t in enumerate((a_src, b_src)):
            c = V.const(t)
            if c is not None:
                ins.append(self.add_const(V.gp, slot, np.asarray(c, np.float32), "f32", t))
            else:
                xi, xt = V.runtime(t)
                if xt != "I":
                    raise V.err("MatMul operand in an NCHW view")
                ins.append(xi)
        return ins, {"scale": sc, "transpose_b": tb}, out, "I"

    def r_Dense(self, V):
        nd = V.nodes_left()[0]
        V.use(nd)
        xi, xt = V.runtime(nd.input[0])
        if xt != "I":
            raise V.err("Dense input in an NCHW view")
        if nd.op_type == "Gemm":
            a = V.attrs(nd)
            if not a.get("transB", 0) or a.get("transA", 0):
                raise V.err("Gemm(transB 1) expected")
            w = np.asarray(V.const(nd.input[1]), np.float32)
            ins = [xi, self.add_const(V.gp, 1, w, "f32", nd.input[1])]
            if len(nd.input) > 2 and nd.input[2]:
                ins.append(self.add_const(V.gp, 2, np.asarray(V.const(nd.input[2]), np.float32), "f32", nd.input[2]))
            out = nd.output[0]
        elif nd.op_type == "MatMul":
            w = np.asarray(V.const(nd.input[1]), np.float32)
            ins = [xi, self.add_const(V.gp, 1, np.ascontiguousarray(w.T), "f32", nd.input[1])]
            out = nd.output[0]
            nxt = V.next_of(out)
            if nxt is not None and nxt.op_type == "Add" and V.const(nxt.input[1]) is not None:
                V.use(nxt)
                ins.append(self.add_const(V.gp, 2, np.asarray(V.const(nxt.input[1]), np.float32), "f32", nxt.input[1]))
                out = nxt.output[0]
        else:
            raise V.err(f"Dense: Gemm or MatMul expected, got {nd.op_type}")
        act = "none"
        nxt = V.next_of(out)
        if nxt is not None and nxt.op_type == "Relu":
            V.use(nxt)
            out, act = nxt.output[0], "relu"
        return ins, {"activation": act}, out, "I"

    def _vec(self, V, t: str, tag: str, slot: int) -> str:
        c = np.asarray(V.const(t), np.float32)
        if tag == "C":
            if c.ndim == 3 and c.shape[1:] == (1, 1):
                c = c.reshape(-1)
            elif c.ndim == 4:
                c = np.ascontiguousarray(np.transpose(c, C2I))
            else:
                raise V.err(f"constant of shape {c.shape} in an NCHW group ([C, 1, 1] or 4-D)")
        return self.add_const(V.gp, slot, c, "f32", t)

    def r_ChannelAffine(self, V):
        nodes = V.nodes_left()
        ins = ["", "", ""]
        first = nodes[0]
        x = None
        if first.op_type == "Mul":
            V.use(first)
            x, tag = V.runtime(first.input[0])
            ins[1] = self._vec(V, first.input[1], tag, 1)
            out = first.output[0]
            nxt = V.next_of(out)
            if nxt is not None and nxt.op_type == "Add":
                V.use(nxt)
                ins[2] = self._vec(V, nxt.input[1], tag, 2)
                out = nxt.output[0]
        elif first.op_type == "Add":
            V.use(first)
            x, tag = V.runtime(first.input[0])
            ins[2] = self._vec(V, first.input[1], tag, 2)
            out = first.output[0]
        else:
            raise V.err(f"ChannelAffine: Mul / Add expected, got {first.op_type}")
        ins[0] = x
        return ins, {}, out, tag

    def r_Activation(self, V):
        nodes = V.nodes_left()
        nd = nodes[0]
        if nd.op_type in ("Relu", "Sign", "Tanh", "Sigmoid") and len(nodes) == 1:
            V.use(nd)
            x, tag = V.runtime(nd.input[0])
            t = {"Relu": "relu", "Sign": "sign_zero", "Tanh": "tanh", "Sigmoid": "sigmoid"}[nd.op_type]
            return [x], {"type": t}, nd.output[0], tag
        if nd.op_type == "BipolarQuant" and len(nodes) == 1:     # step 6: Activation sign (sign(0) = +1)
            if not V.is_const_val(nd.input[1], 1.0):
                raise V.err("Activation sign: BipolarQuant(x, 1) expected")
            V.use(nd)
            x, tag = V.runtime(nd.input[0])
            return [x], {"type": "sign"}, nd.output[0], tag
        if nd.op_type == "PRelu":
            V.use(nd)
            x, tag = V.runtime(nd.input[0])
            return [x, self._vec(V, nd.input[1], tag, 1)], {"type": "prelu"}, nd.output[0], tag
        if nd.op_type == "Div":         # gelu: Mul(Mul(x, 0.5), Add(Erf(Div(x, sqrt 2)), 1))
            V.use(nd)
            x = nd.input[0]
            c = V.carrier_val(nd.input[1])
            if c is None or np.asarray(c).size != 1 or float(np.asarray(c).reshape(-1)[0]) != math.sqrt(2.0):
                raise V.err("gelu: Div(x, sqrt 2) with sqrt 2 as a float64 carrier expected")
            erf = V.next_of(nd.output[0], "Erf")
            add = V.next_of(erf.output[0], "Add")
            if not V.is_const_val(add.input[1], 1.0):
                raise V.err("gelu: Add(erf, 1) expected")
            m1 = next(q for q in nodes if q.op_type == "Mul" and q.input[0] == x)
            if not V.is_const_val(m1.input[1], 0.5):
                raise V.err("gelu: Mul(x, 0.5) expected")
            m2 = next(q for q in nodes if q.op_type == "Mul" and q.input == [m1.output[0], add.output[0]])
            for q in (erf, add, m1, m2):
                V.use(q)
            xi, tag = V.runtime(x)
            self.rule("Erf form -> Activation gelu")
            return [xi], {"type": "gelu"}, m2.output[0], tag
        raise V.err(f"Activation: unsupported pattern {[q.op_type for q in nodes]}")

    def _binary(self, V, op):
        nodes = V.nodes_left()
        if len(nodes) != 1 or nodes[0].op_type != op:
            raise V.err(f"{op} expected, got {[q.op_type for q in nodes]}")
        nd = nodes[0]
        V.use(nd)
        tags = set()
        ins = []
        rts = [i for i in nd.input if V.const(i) is None]
        for i in rts:
            tags.add(V.runtime(i)[1])
        if len(tags) != 1:
            raise V.err("element-wise group with inputs in different layouts")
        tag = tags.pop()
        for j, i in enumerate(nd.input):
            if V.const(i) is not None:
                ins.append(self._vec(V, i, tag, j) if tag == "C" else self.add_const(V.gp, j, np.asarray(V.const(i), np.float32), "f32", i))
            else:
                ins.append(V.runtime(i)[0])
        return ins, {}, nd.output[0], tag

    def r_Add(self, V):
        return self._binary(V, "Add")

    def r_AddBroadcast(self, V):
        return self._binary(V, "Add")

    def r_Mul(self, V):
        return self._binary(V, "Mul")

    def r_Concat(self, V):
        nd = V.single("Concat")
        ax = int(V.attrs(nd)["axis"])
        tags, ins = set(), []
        for j, i in enumerate(nd.input):
            c = V.const(i)
            if c is not None:
                ins.append(self.add_const(V.gp, j, np.asarray(c, np.float32), "f32", i))
                tags.add("I")
            else:
                xi, tg = V.runtime(i)
                ins.append(xi)
                tags.add(tg)
        if len(tags) != 1:
            raise V.err("Concat of inputs in different layouts")
        tag = tags.pop()
        if tag == "C":
            if ax != 1:
                raise V.err("Concat of NCHW views must be on the channel axis")
            ax = -1
        return ins, {"axis": ax}, nd.output[0], tag

    def r_Reshape(self, V):
        nd = V.single("Reshape")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("Reshape of an NCHW view")
        shp = [int(v) for v in np.asarray(V.const(nd.input[1])).reshape(-1)]
        return [xi], {"shape": shp}, nd.output[0], "I"

    def r_Transpose(self, V):
        nd = V.single("Transpose")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("IR Transpose of an NCHW view")
        return [xi], {"perm": [int(p) for p in V.attrs(nd)["perm"]]}, nd.output[0], "I"

    def _ids(self, V, t: str, slot: int) -> str:
        c = V.const(t)
        if c is not None:
            if np.asarray(c).dtype != np.int32:
                raise V.err("Gather index constants must be INT32 (the IR's i32)")
            return self.add_const(V.gp, slot, np.asarray(c, np.int32), "i32", t)
        xi, tag = V.runtime(t)
        if tag != "I":
            raise V.err("Gather ids in an NCHW view")
        return xi

    def r_Gather(self, V):
        nodes = V.nodes_left()
        gathers = [q for q in nodes if q.op_type == "Gather"]
        if any(int(V.attrs(q).get("axis", 0)) != 0 for q in gathers):
            raise V.err("Gather on axis 0 only (the IR's row gather)")
        if len(nodes) == 1:
            nd = nodes[0]
            V.use(nd)
            c = V.const(nd.input[0])
            if c is not None and nd.input[0] not in self.inits:      # Cast(INT8 table): a bit / ternary table
                return self._bit_gather(V, nd, None)
            if c is not None:
                tab = self.add_const(V.gp, 0, np.asarray(c, np.float32), "f32", nd.input[0])
            else:
                tab, tag = V.runtime(nd.input[0])
                if tag != "I":
                    raise V.err("Gather of an NCHW view")
            return [tab, self._ids(V, nd.input[1], 1)], {}, nd.output[0], "I"
        mul = next((q for q in nodes if q.op_type == "Mul"), None)
        if mul is None or len(gathers) != 2:
            raise V.err(f"Gather: unsupported pattern {[q.op_type for q in nodes]}")
        rows = V.prod(mul.input[0], "Gather")
        uns = V.prod(mul.input[1], "Unsqueeze")
        sg = V.prod(uns.input[0], "Gather")
        if [int(v) for v in np.asarray(V.const(uns.input[1])).reshape(-1)] != [-1] or sg.input[1] != rows.input[1]:
            raise V.err("scaled bit Gather: Mul(Gather(T, ids), Unsqueeze(Gather(alpha, ids), -1)) expected")
        for q in (mul, rows, uns, sg):
            V.use(q)
        return self._bit_gather(V, rows, sg, out=mul.output[0])

    def _bit_gather(self, V, rows, sg, out=None):
        if rows.input[0] in self.inits:
            raise V.err("bit table expected as Cast(INT8 initializer)")
        t = self._pm1(V, rows.input[0], "Gather table")
        ins = [self.add_const(V.gp, 0, np.where(t < 0, -1, 1).astype(np.int8), "bit", rows.input[0]),
               self._ids(V, rows.input[1], 1), "", ""]
        if sg is not None:
            ins[2] = self.add_const(V.gp, 2, np.asarray(V.const(sg.input[0]), np.float32), "f32", sg.input[0])
        if np.any(t == 0):
            ins[3] = self.add_const(V.gp, 3, np.where(t == 0, -1, 1).astype(np.int8), "bit", rows.input[0])
        self.rule("bit / ternary embedding table -> Gather(w_sign, ids, alpha, w_zero)")
        return ins, {}, out or rows.output[0], "I"

    def r_LayerNorm(self, V):
        nd = V.single("LayerNormalization")
        a = V.attrs(nd)
        if int(a.get("axis", -1)) != -1 or len(nd.input) != 3:
            raise V.err("LayerNormalization(x, gamma, beta, axis -1) expected")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("LayerNorm of an NCHW view")
        ins = [xi, self.add_const(V.gp, 1, np.asarray(V.const(nd.input[1]), np.float32), "f32", nd.input[1]),
               self.add_const(V.gp, 2, np.asarray(V.const(nd.input[2]), np.float32), "f32", nd.input[2])]
        return ins, {"eps": float(np.float32(a.get("epsilon", 1e-5)))}, nd.output[0], "I"

    def r_Softmax(self, V):
        nd = V.single("Softmax")
        if int(V.attrs(nd).get("axis", -1)) != -1:
            raise V.err("Softmax over the last axis only")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("Softmax of an NCHW view")
        return [xi], {}, nd.output[0], "I"

    def r_ReduceMean(self, V):
        nd = V.single("ReduceMean")
        a = V.attrs(nd)
        axes = list(a.get("axes", []))
        if len(axes) != 1:
            raise V.err("ReduceMean over one axis")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("ReduceMean of an NCHW view")
        return [xi], {"axis": int(axes[0]), "keepdims": int(a.get("keepdims", 1))}, nd.output[0], "I"

    def r_GlobalAvgPool(self, V):
        gap = V.first("GlobalAveragePool")
        V.use(gap)
        xi, tag = V.runtime(gap.input[0])
        if tag != "C":
            raise V.err("GlobalAveragePool of a non-NCHW tensor")
        nxt = V.next_of(gap.output[0])
        if nxt is not None and nxt.op_type == "Flatten":
            if int(V.attrs(nxt).get("axis", 1)) != 1:
                raise V.err("Flatten(axis 1) expected")
            V.use(nxt)
            return [xi], {"keepdims": 0}, nxt.output[0], "I"
        return [xi], {"keepdims": 1}, gap.output[0], "C"

    def _pool(self, V, op):
        nd = V.single(op)
        a = V.attrs(nd)
        if a.get("ceil_mode", 0) or a.get("auto_pad", "NOTSET") not in ("NOTSET", ""):
            raise V.err("ceil_mode / auto_pad")
        xi, tag = V.runtime(nd.input[0])
        if tag != "C":
            raise V.err(f"{op} of a non-NCHW tensor")
        k = list(a["kernel_shape"])
        canon = {"kernel": k, "strides": list(a.get("strides", k)), "pads": _ir_pads(a.get("pads", [0, 0, 0, 0]))}
        if op == "AveragePool":
            canon["count_include_pad"] = int(a.get("count_include_pad", 0))
        return [xi], canon, nd.output[0], "C"

    def r_Pad(self, V):
        nd = V.single("Pad")
        if V.attrs(nd).get("mode", "constant") != "constant":
            raise V.err("Pad mode must be constant")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("IR Pad of an NCHW view")
        po = [int(v) for v in np.asarray(V.const(nd.input[1])).reshape(-1)]
        r = len(po) // 2
        val = V.const(nd.input[2]) if len(nd.input) > 2 and nd.input[2] else np.float32(0)
        return [xi], {"pads": [po[d + k * r] for d in range(r) for k in (0, 1)], "value": float(np.asarray(val).reshape(()))}, nd.output[0], "I"

    def r_MaxPool2D(self, V):
        return self._pool(V, "MaxPool")

    def r_AvgPool2D(self, V):
        return self._pool(V, "AveragePool")

    def r_FakeQuant(self, V):
        m0 = V.first("Mul")
        V.use(m0)
        xi, tag = V.runtime(m0.input[0])
        if tag != "I":
            raise V.err("FakeQuant of an NCHW view")
        pre = V.carrier_val(m0.input[1])
        if pre is None:
            raise V.err("FakeQuant pre_scale must be a float64 carrier")
        dv = V.next_of(m0.output[0], "Div")
        ad = V.next_of(dv.output[0], "Add")
        rd = V.next_of(ad.output[0], "Round")
        cl = V.next_of(rd.output[0], "Clip")
        m1 = V.next_of(cl.output[0], "Mul")
        for q in (dv, ad, rd, cl, m1):
            V.use(q)
        s, z = V.const(dv.input[1]), V.const(ad.input[1])
        if s is None or z is None or m1.input[1] != dv.input[1]:
            raise V.err("FakeQuant: Mul(Clip(Round(Add(Div(Mul(x, pre), s), z)), lo, hi), s) expected")
        s, z = np.asarray(s, np.float32), np.asarray(z, np.float32)
        r = len(s.shape)
        big = [i for i, d in enumerate(s.shape) if d != 1]
        if len(big) > 1:
            raise V.err("FakeQuant constants vary along several axes")
        axis = -1 if (not big or big[0] == r - 1) else big[0]
        ins = [xi, self.add_const(V.gp, 1, s.reshape(-1), "f32", dv.input[1]), self.add_const(V.gp, 2, z.reshape(-1), "f32", ad.input[1])]
        lo, hi = _f(V.const(cl.input[1])), _f(V.const(cl.input[2]))
        if lo != int(lo) or hi != int(hi):
            raise V.err("FakeQuant bounds must be integers")
        return ins, {"axis": axis, "pre_scale": float(np.asarray(pre).reshape(-1)[0]), "qmin": int(lo), "qmax": int(hi)}, m1.output[0], "I"

    def r_ChannelGate(self, V):
        mul = [q for q in V.nodes_left() if q.op_type == "Mul"]
        if len(mul) != 1:
            raise V.err("ChannelGate: one Mul expected")
        mul = mul[0]
        c2 = V.prod(mul.input[1], "Cast")
        sg = V.prod(c2.input[0], "Sigmoid")
        c1 = V.prod(sg.input[0], "Cast")
        if V.attrs(c2)["to"] != FLOAT or V.attrs(c1)["to"] != DOUBLE:
            raise V.err("ChannelGate: Cast(Sigmoid(Cast(g, DOUBLE)), FLOAT) expected")
        for q in (mul, c2, sg, c1):
            V.use(q)
        g_src = c1.input[0]
        p = V.prod_or_none(g_src)
        if p is not None and p.op_type == "Reshape" and V.in_group(p):
            V.use(p)
            gi, gt = V.runtime(p.input[0])
            if gt != "I" or len(self.oshape[p.input[0]]) != 2:
                raise V.err("ChannelGate gate reshape of a non-[N, C] tensor")
        else:
            gi, gt = V.runtime(g_src)
            if gt != "C":
                raise V.err("ChannelGate 4-D gate must be an NCHW view")
        xi, xt = V.runtime(mul.input[0])
        if xt != "C":
            raise V.err("ChannelGate input must be an NCHW view")
        return [xi, gi], {"gate": "sigmoid"}, mul.output[0], "C"

    # ---------------------------------------------------------------- image features
    def r_ImageGrayU8(self, V):
        last = V.last()
        cf = V.prod(last.output[0], "Cast") if last.op_type == "Cast" else None
        if cf is None or V.attrs(cf)["to"] != FLOAT:
            raise V.err("ImageGrayU8: final Cast(FLOAT) expected")
        bs = V.prod(cf.input[0], "BitShift")
        if V.attrs(bs).get("direction") != "RIGHT":
            raise V.err("BitShift RIGHT expected")
        shift = int(np.asarray(V.const(bs.input[1])).reshape(-1)[0])
        cu = V.prod(bs.input[0], "Cast")
        ad = V.prod(cu.input[0], "Add")
        rnd = int(np.asarray(V.const(ad.input[1])).reshape(-1)[0])
        if rnd != 1 << (shift - 1) or V.attrs(cu)["to"] != UINT32:
            raise V.err("ImageGrayU8 rounding constant / UINT32 cast")
        rs = V.prod(ad.input[0], "ReduceSum")
        if [int(v) for v in np.asarray(V.const(rs.input[1])).reshape(-1)] != [1] or int(V.attrs(rs).get("keepdims", 1)) != 1:
            raise V.err("ReduceSum over the channel axis (keepdims) expected")
        mk = V.prod(rs.input[0], "Mul")
        k = np.asarray(V.const(mk.input[1]))
        if k.dtype != np.int32 or k.shape != (1, 3, 1, 1):
            raise V.err("ImageGrayU8 coefficients: INT32 [1, 3, 1, 1]")
        ci = V.prod(mk.input[0], "Cast")
        fl = V.prod(ci.input[0], "Floor")
        cl = V.prod(fl.input[0], "Clip")
        if not (V.is_const_val(cl.input[1], 0.0) and V.is_const_val(cl.input[2], 255.0)) or V.attrs(ci)["to"] != INT32:
            raise V.err("Clip(v, 0, 255) -> Floor -> Cast(INT32) expected")
        wh = V.prod(cl.input[0], "Where")
        nan = V.prod(wh.input[0], "IsNaN")
        if not V.is_const_val(wh.input[1], 0.0) or wh.input[2] != nan.input[0]:
            raise V.err("Where(IsNaN(v), 0, v) expected")
        mv = V.prod(nan.input[0], "Mul")
        sc = _f(V.const(mv.input[1]))
        for q in (cf, bs, cu, ad, rs, mk, ci, fl, cl, wh, nan, mv):
            V.use(q)
        xi, tag = V.runtime(mv.input[0])
        if tag != "C":
            raise V.err("ImageGrayU8 input must be an NCHW view")
        self.rule("standard-op gray subgraph -> ImageGrayU8 (bit-exact)")
        return [xi], {"coeffs": [int(v) for v in k.reshape(-1)], "scale": float(np.float32(sc)), "shift": shift}, cf.output[0], "C"

    def r_FftLogMagnitude(self, V):
        cf = V.last()
        if cf.op_type != "Cast" or V.attrs(cf)["to"] != FLOAT:
            raise V.err("FftLogMagnitude: final Cast(FLOAT) expected")
        dv = V.prod(cf.input[0], "Div")
        ml = V.prod(dv.input[0], "Mul")
        lg = V.prod(ml.input[0], "Log")
        ae = V.prod(lg.input[0], "Add")
        div, mul, eps = _f(V.const(dv.input[1])), _f(V.const(ml.input[1])), _f(V.const(ae.input[1]))
        mag = ae.input[0]
        shift = 0
        g2 = V.prod(mag)
        if g2.op_type == "Gather":
            g1 = V.prod(g2.input[0], "Gather")
            if int(V.attrs(g1)["axis"]) != 2 or int(V.attrs(g2)["axis"]) != 3:
                raise V.err("fftshift: Gather on axes 2 then 3")
            V.use(g1)
            V.use(g2)
            shift, ih, iw, mag = 1, np.asarray(V.const(g1.input[1])), np.asarray(V.const(g2.input[1])), g1.input[0]
        sq = V.prod(mag, "Sqrt")
        s2 = V.prod(sq.input[0], "Add")
        mre, mim = V.prod(s2.input[0], "Mul"), V.prod(s2.input[1], "Mul")
        if mre.input[0] != mre.input[1] or mim.input[0] != mim.input[1]:
            raise V.err("|X| = Sqrt(Re*Re + Im*Im) expected")
        re = V.prod(mre.input[0], "Sub")
        ng = V.prod(mim.input[0], "Neg")
        imadd = V.prod(ng.input[0], "Add")
        p1, p2 = V.prod(re.input[0], "MatMul"), V.prod(re.input[1], "MatMul")
        p3, p4 = V.prod(imadd.input[0], "MatMul"), V.prod(imadd.input[1], "MatMul")
        a1, a2 = V.prod(p1.input[0], "MatMul"), V.prod(p2.input[0], "MatMul")
        if p4.input[0] != a1.output[0] or p3.input[0] != a2.output[0] or a1.input[1] != a2.input[1]:
            raise V.err("DFT matrix products: Re = C X C - S X S, Im = -(S X C + C X S)")
        X = a1.input[1]
        cx = V.prod(X, "Cast")
        if V.attrs(cx)["to"] != DOUBLE:
            raise V.err("DFT in DOUBLE expected")
        xi, tag = V.runtime(cx.input[0])
        if tag != "C":
            raise V.err("FftLogMagnitude input must be an NCHW view")
        _, c, h, w = self.oshape[cx.input[0]]
        ch, sh_ = dft_tables(h)
        cw, sw = dft_tables(w)
        tabs = {"C_H": (a1.input[0], ch), "S_H": (a2.input[0], sh_), "C_W": (p1.input[1], cw), "S_W": (p2.input[1], sw)}
        if p3.input[1] != p1.input[1] or p4.input[1] != p2.input[1]:
            raise V.err("DFT: the W tables of Re and Im differ")
        for k, (t, ref) in tabs.items():
            v = V.const(t)
            if v is None or np.asarray(v).dtype != np.float64 or not np.array_equal(np.asarray(v), ref):
                raise V.err(f"DFT table {k} is not cos / sin(2 pi k n / N) in float64")
        if shift and not (np.array_equal(ih, fftshift_index(h)) and np.array_equal(iw, fftshift_index(w))):
            raise V.err("fftshift indices")
        for q in (cf, dv, ml, lg, ae, sq, s2, mre, mim, re, ng, imadd, p1, p2, p3, p4, a1, a2, cx):
            V.use(q)
        self.rule("DOUBLE MatMul DFT subgraph -> FftLogMagnitude (float contract, Q1)")
        return [xi], {"eps": eps, "mul": mul, "div": div, "shift": shift}, cf.output[0], "C"

    def r_LocalBinaryPattern(self, V):
        cf = V.last()
        if cf.op_type != "Cast" or V.attrs(cf)["to"] != FLOAT:
            raise V.err("LocalBinaryPattern: final Cast(FLOAT) expected")
        dv = V.prod(cf.input[0], "Div")
        div = _f(V.const(dv.input[1]))
        cd = V.prod(dv.input[0], "Cast")
        lbp = cd.input[0]
        V.use(cf)
        V.use(dv)
        V.use(cd)
        wh = V.prod_or_none(lbp)
        bits = []
        if wh is not None and wh.op_type == "Where":
            method = "uniform"
            le = V.prod(wh.input[0], "LessOrEqual")
            if int(np.asarray(V.const(le.input[1])).reshape(-1)[0]) != 2:
                raise V.err("uniform: LessOrEqual(changes, 2) expected")
            V.use(wh)
            V.use(le)
            ones = V.sum_chain(wh.input[1])
            for t in ones:
                c = V.prod(t, "Cast")
                V.use(c)
                bits.append(c.input[0])
            P = len(bits)
            if int(np.asarray(V.const(wh.input[2])).reshape(-1)[0]) != P + 1:
                raise V.err("uniform: P + 1 for non-uniform patterns expected")
            ch = V.sum_chain(le.input[0])
            if len(ch) != P - 1:
                raise V.err("uniform: P - 1 transitions expected")
            for i, t in enumerate(ch):
                c = V.prod(t, "Cast")
                xo = V.prod(c.input[0], "Xor")
                if xo.input != [bits[i], bits[i + 1]]:
                    raise V.err("uniform: Xor(bit_i, bit_i+1) in order expected")
                V.use(c)
                V.use(xo)
        else:
            method = "default"
            for i, t in enumerate(V.sum_chain(lbp)):
                m = V.prod(t, "Mul")
                if int(np.asarray(V.const(m.input[1])).reshape(-1)[0]) != 1 << i:
                    raise V.err("default: bit_i * 2^i expected")
                c = V.prod(m.input[0], "Cast")
                V.use(m)
                V.use(c)
                bits.append(c.input[0])
        rp, cp = [], []
        X = None
        for bt in bits:
            ge = V.prod(bt, "GreaterOrEqual")
            if not V.is_const_val(ge.input[1], 0.0):
                raise V.err("bit = GreaterOrEqual(t - X, 0) expected")
            sb = V.prod(ge.input[0], "Sub")
            t = V.prod(sb.input[0], "Add")
            if X is None:
                X = sb.input[1]
            elif sb.input[1] != X:
                raise V.err("all sample points compare against the same X")
            m_top, m_bot = V.prod(t.input[0], "Mul"), V.prod(t.input[1], "Mul")
            omdr, top = m_top.input
            dr, bot = m_bot.input
            ri, r0, r1 = self._frac(V, dr, omdr, "rows")
            top_n, bot_n = V.prod(top, "Add"), V.prod(bot, "Add")
            vals = []
            for nd_ in (top_n, bot_n):
                ma, mb = V.prod(nd_.input[0], "Mul"), V.prod(nd_.input[1], "Mul")
                vals.append((ma, mb))
            (ma, mb), (mc, md) = vals
            omdc, dc = ma.input[0], mb.input[0]
            if mc.input[0] != omdc or md.input[0] != dc:
                raise V.err("bilinear weights: the same (1 - dc), dc in top and bottom")
            ci, c0, c1 = self._frac(V, dc, omdc, "cols")
            offs = {}
            for (mm_, dy, dx) in ((ma, r0, c0), (mb, r0, c1), (mc, r1, c0), (md, r1, c1)):
                offs[(dy, dx)] = mm_.input[1]
            for (dy, dx), s in offs.items():
                self._check_slice(V, s, X, dy, dx)
            for q in (ge, sb, t, m_top, m_bot, top_n, bot_n, ma, mb, mc, md):
                V.use(q)
            rp.append(ri)
            cp.append(ci)
        cx = V.prod(X, "Cast")
        if V.attrs(cx)["to"] != DOUBLE:
            raise V.err("LBP in DOUBLE expected")
        V.use(cx)
        xi, tag = V.runtime(cx.input[0])
        if tag != "C":
            raise V.err("LocalBinaryPattern input must be an NCHW view")
        _, _, h, w = self.oshape[cx.input[0]]
        for v in rp:
            if not lbp_shift_ok(v, h):
                raise V.err(f"row offset {v}: uniform-shift condition (Q12) violated")
        for v in cp:
            if not lbp_shift_ok(v, w):
                raise V.err(f"column offset {v}: uniform-shift condition (Q12) violated")
        V.use_all_named_rest()
        self.rule("DOUBLE bilinear subgraph -> LocalBinaryPattern (bit-exact)")
        return [xi], {"rp": rp, "cp": cp, "method": method, "div": div}, cf.output[0], "C"

    def _frac(self, V, d: str, omd: str, kind: str):
        """d = Sub(rr, Floor(rr)), rr = Add(index, offset), omd = Sub(1, d): returns (offset, floor, ceil)."""
        sb = V.prod(d, "Sub")
        rr = V.prod(sb.input[0], "Add")
        fl = V.prod(sb.input[1], "Floor")
        if fl.input[0] != rr.output[0]:
            raise V.err("bilinear weight: Sub(rr, Floor(rr)) expected")
        idx = np.asarray(V.const(rr.input[0]))
        off = np.asarray(V.const(rr.input[1]))
        n = idx.size
        want = np.arange(n, dtype=np.float64).reshape((n, 1) if kind == "rows" else (1, n))
        if idx.dtype != np.float64 or not np.array_equal(idx, want) or off.dtype != np.float64 or off.size != 1:
            raise V.err(f"{kind} index tensor arange / DOUBLE scalar offset expected")
        om = V.prod(omd, "Sub")
        if om.input[1] != d or not V.is_const_val(om.input[0], 1.0):
            raise V.err("Sub(1, d) expected")
        for q in (sb, rr, fl, om):
            V.use(q)
        o = float(off.reshape(-1)[0])
        return o, math.floor(o), math.ceil(o)

    def _check_slice(self, V, s: str, X: str, dy: int, dx: int) -> None:
        sl = V.prod(s, "Slice")
        pd = V.prod(sl.input[0], "Pad")
        if pd.input[0] != X:
            raise V.err("shifted copies: Slice(Pad(X)) expected")
        pp = [int(v) for v in np.asarray(V.const(pd.input[1])).reshape(-1)]
        if len(pp) != 8 or pp[0] or pp[1] or pp[4] or pp[5] or _f(V.const(pd.input[2])) != 0.0:
            raise V.err("LBP Pad: zero spatial padding expected")
        pt, pl = pp[2], pp[3]
        _, _, h, w = self.oshape[X]
        st = [int(v) for v in np.asarray(V.const(sl.input[1])).reshape(-1)]
        en = [int(v) for v in np.asarray(V.const(sl.input[2])).reshape(-1)]
        ax = [int(v) for v in np.asarray(V.const(sl.input[3])).reshape(-1)]
        if ax != [2, 3] or st != [pt + dy, pl + dx] or en != [pt + dy + h, pl + dx + w] or len(sl.input) > 4:
            raise V.err(f"shifted copy ({dy}, {dx}): Slice offsets do not match the sample point")
        V.use(sl)
        V.use(pd)


class _View:
    """The pattern nodes of one group, with lookups (producer / consumer inside the group, constants, runtime inputs)."""

    def __init__(self, imp: GroupImporter, gp: _Group, nodes: list):
        self.importer, self.gp, self.nodes = imp, gp, nodes
        self.names = {id(nd) for nd in nodes}
        self.prod_in = {o: nd for nd in nodes for o in nd.output}
        self.cons_in: dict = {}
        for nd in nodes:
            for i in nd.input:
                self.cons_in.setdefault(i, []).append(nd)
        self.used: set = set()

    def err(self, why: str) -> GroupImportError:
        return GroupImportError(self.gp.name, why, self.gp.op)

    def attrs(self, nd) -> dict:
        from onnx import helper
        out = {}
        for a in nd.attribute:
            v = helper.get_attribute_value(a)
            out[a.name] = v.decode() if isinstance(v, bytes) else v
        return out

    def const(self, t: str):
        """Value of an initializer or of an unnamed constant Cast (group nodes are never folded)."""
        return self.importer.cval.get(t)

    def carrier_val(self, t: str):
        return self.importer.cval[t] if t in self.importer.carrier else None

    def is_const_val(self, t: str, v: float) -> bool:
        c = self.const(t)
        return c is not None and np.asarray(c).size == 1 and float(np.asarray(c).reshape(-1)[0]) == v

    def runtime(self, t: str):
        r = self.importer.rt.get(t)
        if r is None:
            raise self.err(f"tensor {t!r} is not a runtime tensor of an earlier group")
        return r

    def prod_or_none(self, t: str):
        return self.prod_in.get(t)

    def prod(self, t: str, op: str | None = None):
        nd = self.prod_in.get(t)
        if nd is None or (op is not None and nd.op_type != op):
            raise self.err(f"tensor {t!r}: produced by {nd.op_type if nd is not None else 'nothing in the group'}, "
                           f"{op or 'a group node'} expected")
        return nd

    def in_group(self, nd) -> bool:
        return id(nd) in self.names

    def next_of(self, t: str, op: str | None = None):
        c = self.cons_in.get(t, [])
        if op is not None:
            if len(c) != 1 or c[0].op_type != op:
                raise self.err(f"tensor {t!r}: {op} expected as its only consumer in the group")
            return c[0]
        return c[0] if len(c) == 1 else None

    def first(self, op: str):
        nd = next((q for q in self.nodes if q.op_type == op), None)
        if nd is None:
            raise self.err(f"no {op} node")
        return nd

    def single(self, op: str):
        if len(self.nodes) != 1 or self.nodes[0].op_type != op:
            raise self.err(f"one {op} node expected, got {[q.op_type for q in self.nodes]}")
        self.use(self.nodes[0])
        return self.nodes[0]

    def last(self):
        return self.nodes[-1]

    def nodes_left(self) -> list:
        return [q for q in self.nodes if id(q) not in self.used]

    def chain_ops(self) -> list:
        return [q.op_type for q in self.nodes]

    def use(self, nd) -> None:
        self.used.add(id(nd))

    def use_chain(self, bnd, pre, x) -> None:
        """Mark the binarizer node and its Sub / Greater / Sign helpers."""
        self.use(bnd)
        if bnd.op_type == "Where":
            self.use(self.prod(bnd.input[0]))
        if bnd.op_type == "Relu":
            self.use(self.prod(bnd.input[0]))
        if pre != x:
            self.use(self.prod(pre))

    def sum_chain(self, t: str) -> list:
        """Terms of a left-nested Add chain (a + b) + c ... in order."""
        nd = self.prod_or_none(t)
        if nd is not None and nd.op_type == "Add" and self.const(nd.input[1]) is None and self.const(nd.input[0]) is None:
            self.use(nd)
            return self.sum_chain(nd.input[0]) + [nd.input[1]]
        return [t]

    def use_all_named_rest(self) -> None:
        pass

    def check_all_used(self) -> None:
        left = [q for q in self.nodes if id(q) not in self.used]
        if left:
            raise self.err(f"{len(left)} node(s) of the group are not part of the {self.gp.op} pattern: "
                           f"{[q.op_type for q in left][:8]}")


def import_groups(model) -> tuple[Graph, dict]:
    """Group-mapped canonical file -> (IR graph, info). No graph pass runs (the groups are the deployed graph); the
    data section is in first-use order, or the file's bitino.const_order. An int8_full file (bitino.variant
    int8_full*) is read by the step-5 readers (bitino.qonnx_int8)."""
    variant = next((p.value for p in model.metadata_props if p.key == "bitino.variant"), "")
    if variant.startswith("int8_full"):
        from .qonnx_int8 import import_int8
        return import_int8(model)
    imp = GroupImporter(model)
    g = imp.run()
    order, seen = [], set()
    for n in g.nodes:
        for i in n.inputs:
            if i and i in g.constants and i not in seen:
                seen.add(i)
                order.append(i)
    g.constants = {k: g.constants[k] for k in order + [k for k in g.constants if k not in seen]}
    raw = next((p.value for p in model.metadata_props if p.key == CONST_ORDER_KEY), None)
    const_order = None
    if raw is not None:
        lst = json.loads(raw)
        if set(lst) != set(g.constants) or len(lst) != len(set(lst)):
            raise GroupImportError(CONST_ORDER_KEY, "bitino.const_order and the imported constants differ", "model")
        const_order = lst != list(g.constants)
        g.constants = {k: g.constants[k] for k in lst}
    info = {"rules": imp.rules, "binarizers": imp.binarizers, "mapping": MAPPING, "groups": len(imp.groups),
            "passes": {"none": "group mapping: the groups are the deployed graph",
                       "bitino.const_order (applied)": const_order},
            "op_counts": g.op_counts(), "logits_onnx_tensor": imp.out_names[0], "onnx_outputs": imp.out_names,
            "onnx_tags": {t: tag for t, (_, tag) in imp.rt.items()}}
    return g, info


# ================================================================================================ evaluation helpers
def ort_model(model, extra_outputs: list | None = None, float_outputs: bool = False):
    """A copy of a canonical file that onnxruntime runs as one session: the QONNX quantizers replaced by their
    definitions in standard operators (BipolarQuant(x, s) = Where(x >= 0, s, -s); the constant ternary IntQuant of
    weights in {-1, 0, +1} = Identity), the QONNX opset dropped; extra_outputs become graph outputs. Used for the
    fast float32 reference (the same values as qonnx execute_onnx, which evaluates these definitions per node)."""
    import copy

    import onnx
    from onnx import helper
    m = copy.deepcopy(model)
    g = m.graph
    k = 0
    if any(nd.domain == QD and nd.op_type == "BipolarQuant" for nd in g.node) and \
            not any(t.name == "bitino/zero" for t in g.initializer):
        from onnx import numpy_helper
        g.initializer.append(numpy_helper.from_array(np.array(0.0, np.float32), "bitino/zero"))   # int8 files: created on use
    for nd in list(g.node):
        if nd.domain != QD:
            continue
        if nd.op_type == "BipolarQuant":
            k += 1
            c, ns, ps = f"bq{k}/ge", f"bq{k}/neg", nd.input[1]
            new = [helper.make_node("GreaterOrEqual", [nd.input[0], "bitino/zero"], [c], name=f"bq{k}_ge"),
                   helper.make_node("Neg", [ps], [ns], name=f"bq{k}_neg"),
                   helper.make_node("Where", [c, ps, ns], [nd.output[0]], name=f"bq{k}_where")]
        elif nd.op_type == "IntQuant":
            new = [helper.make_node("Identity", [nd.input[0]], [nd.output[0]], name=nd.name)]
        else:
            raise NotImplementedError(f"ort_model: {nd.op_type}")
        pos = list(g.node).index(nd)
        g.node.remove(nd)
        for j, q in enumerate(new):
            g.node.insert(pos + j, q)
    keep = [o for o in m.opset_import if (o.domain or "") in ("", "ai.onnx")]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    if extra_outputs:
        have = {o.name for o in g.output}
        if float_outputs:       # FLOAT tensors (binarizer inputs): no shape-inference copy of a large model
            for t in extra_outputs:
                if t not in have:
                    g.output.append(helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
        else:
            inf = onnx.shape_inference.infer_shapes(m)
            vi = {v.name: v for v in inf.graph.value_info}
            for t in extra_outputs:
                if t not in have and t in vi:
                    g.output.append(vi[t])
            del inf
    return m


def ort_session(model, extra_outputs: list | None = None, threads: int = 1, float_outputs: bool = False):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL     # A2: no fusion
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    blob = ort_model(model, extra_outputs, float_outputs).SerializeToString()
    return ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])


def load_eval_inputs(eval_dir: str, key: str, g: Graph, n: int | None = None) -> dict:
    """{IR input name: array [N, ...]} of an eval set: a single tensor file (images, NHWC) or a per-sample record of
    several inputs (the BiBERT sets: input_ids i32 | token_type_ids i32 | attention_mask f32)."""
    import os
    em = json.load(open(os.path.join(eval_dir, "manifest.json")))
    pre = em["preprocessing"][key]
    if "record" in pre:
        rec = pre["record"]
        raw = np.fromfile(os.path.join(eval_dir, pre["file"]), np.uint8)
        per = sum(int(np.prod(r["shape"])) * 4 for r in rec)
        cnt = raw.size // per
        raw = raw[: cnt * per].reshape(cnt, per)
        out, off = {}, 0
        for r in rec:
            sz = int(np.prod(r["shape"])) * 4
            dt = {"i32": np.int32, "f32": np.float32}[r["dtype"]]
            out[r["name"]] = np.ascontiguousarray(raw[:, off:off + sz]).view(dt).reshape([cnt] + list(r["shape"])[1:])
            off += sz
    else:
        x = np.fromfile(os.path.join(eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
        out = {g.inputs[0]["name"]: x}
    names = [i["name"] for i in g.inputs]
    if sorted(out) != sorted(names):
        raise ValueError(f"eval set inputs {sorted(out)} vs graph inputs {names}")
    if n is not None:
        out = {k: v[:n] for k, v in out.items()}
    return out


def onnx_feeds(g: Graph, feeds_ir: dict) -> dict:
    """IR feeds -> the canonical file's feeds (a 4-D float input is NCHW in the file)."""
    out = {}
    for i in g.inputs:
        v = np.asarray(feeds_ir[i["name"]])
        if len(i["shape"]) == 4 and i["dtype"] == "f32":
            v = np.ascontiguousarray(v.transpose(0, 3, 1, 2))
        out[i["name"]] = v.astype(np.float32 if i["dtype"] == "f32" else np.int32)
    return out


# ================================================================================================ verification (Level 3)
def _src_decision(pre: np.ndarray, mode: str, tag: str):
    """Decision of a source binarizer from its input `pre` (the threshold is a Sub before it), IR layout."""
    p = np.asarray(pre, np.float64)
    if tag == "C" and p.ndim == 4:
        p = p.transpose(0, 2, 3, 1)
    if mode == "ge":
        d = np.where(p >= 0, 1, -1)
    elif mode == "gt":
        d = np.where(p > 0, 1, -1)
    elif mode == "sign_zero":
        d = np.sign(p)
    elif mode == "gt0":
        d = np.where(p > 0, 1, 0)
    else:
        raise ValueError(mode)
    return d.astype(np.int8).reshape(-1), np.abs(p).astype(np.float32).reshape(-1)


def _ir_decision(x: np.ndarray, mode: str, thr) -> np.ndarray:
    from . import executor
    if mode == "gt0":
        return np.where(np.asarray(x) > 0, 1, 0).astype(np.int8).reshape(-1)
    return executor._binarize(np.asarray(x), mode, thr).astype(np.int8).reshape(-1)


XCHECK_MAX_BYTES = 1.0e9      # qonnx execute_onnx cross-check: only graphs whose float32 activations fit this


def verify_groups(model, g: Graph, info: dict, X: dict, labels: np.ndarray, out_dir: str, xcheck: int = 2) -> dict:
    """Level-3 checks of a group-mapped file (docs/numerical-correctness.md), the same report as
    qonnx_import.verify: float32 reference = the file in onnxruntime (one session, ORT_DISABLE_ALL, the QONNX
    quantizers in their standard-op definition; cross-checked bit for bit against qonnx execute_onnx on the first
    `xcheck` samples), float64 reference = OnnxF64 (float64 carriers at their float64 value), the numpy executor of
    the imported graph in float32 / float64, and the first binarization flip of every sample (source pre-activation
    |x - t| at the flipped positions)."""
    import os

    from . import executor
    from .onnx_f64 import OnnxF64
    from .validation.metrics import compare_outputs
    ev = OnnxF64(model)
    out_onnx = info["logits_onnx_tensor"]
    bins = info["binarizers"]
    pres = sorted({b["pre"] for b in bins})
    sess = ort_session(model, extra_outputs=pres, float_outputs=True)
    onames = [o.name for o in sess.get_outputs()]
    ir_keep = {b["ir_tensor"] for b in bins}
    thr = {b["ir_threshold"]: np.asarray(g.constants[b["ir_threshold"]].data) for b in bins if b.get("ir_threshold")}
    n = len(next(iter(X.values())))
    ref32, ref64, ir32, ir64, flips = [], [], [], [], []
    xc = {"n_samples": 0, "logits_bit_identical": True, "max_abs_logits": 0.0, "first_differing_ops": [],
          "note": "qonnx execute_onnx runs each node in its own onnxruntime session with the default graph "
                  "optimizations (e.g. the NCHWc Conv kernel), this reference one session with ORT_DISABLE_ALL (A2)"}
    mw = None
    sess_all = None
    # the first differing node is located only for the token models (a full execution context of a 224 x 224 image
    # model does not fit the memory budget of the run); the image models report the logits gap
    localize = not any(len(i["shape"]) == 4 and i["shape"][1] * i["shape"][2] > 64 * 64 for i in g.inputs) \
        and model.ByteSize() < 50_000_000          # a second session of a large model (BiBERT) does not fit either
    xc["first_differing_op_located"] = localize
    if xcheck:
        try:
            from qonnx.core.modelwrapper import ModelWrapper
            from qonnx.core.onnx_exec import execute_onnx
            from qonnx.transformation.infer_shapes import InferShapes
            mw = ModelWrapper(model).transform(InferShapes())
        except Exception as e:  # pragma: no cover
            xc = {"unavailable": repr(e)}

    def src_dec(env):
        return {p: _src_decision(env[p], next(b["mode"] for b in bins if b["pre"] == p),
                                 next(b["onnx_tag"] for b in bins if b["pre"] == p)) for p in pres}

    def ir_dec(env):
        return [_ir_decision(env[b["ir_tensor"]], b["ir_mode"], thr.get(b.get("ir_threshold"))) for b in bins]

    for i in range(n):
        f_ir = {k: v[i:i + 1] for k, v in X.items()}
        f_onnx = onnx_feeds(g, f_ir)
        shp = {} if i == 0 else None
        e64 = ev.run(f_onnx, np.float64, keep=set(pres), outputs=[out_onnx], shapes=shp)
        if i == 0:
            # qonnx execute_onnx keeps every tensor of the graph (float32): run the cross-check only where that fits
            act = int(sum(4 * int(np.prod(v)) for v in shp.values()))
            xc["activation_bytes_f32"] = act
            if act > XCHECK_MAX_BYTES or model.ByteSize() > 50_000_000:
                xcheck = 0
                xc["skipped"] = (f"all activations of one sample take {act / 1e9:.1f} GB in float32, the file "
                                 f"{model.ByteSize() / 1e6:.0f} MB (budget: {XCHECK_MAX_BYTES / 1e9:.1f} GB of activations, "
                                 "50 MB of file: execute_onnx copies the model and keeps every tensor)")
            del shp
        ref64.append(np.asarray(e64[out_onnx], np.float64).reshape(-1))
        s64 = src_dec(e64)
        del e64
        o32 = dict(zip(onames, sess.run(None, f_onnx)))
        ref32.append(np.asarray(o32[out_onnx], np.float32).reshape(-1))
        s32 = src_dec(o32)
        if mw is not None and i < xcheck:
            ctx = execute_onnx(mw, f_onnx, return_full_exec_context=localize)
            y = np.asarray(ctx[out_onnx], np.float32).reshape(-1)
            same = np.array_equal(y.view(np.uint32), ref32[-1].view(np.uint32))
            xc["n_samples"] += 1
            xc["logits_bit_identical"] = bool(xc["logits_bit_identical"] and same)
            xc["max_abs_logits"] = max(xc["max_abs_logits"], float(np.max(np.abs(y.astype(np.float64) - ref32[-1]))))
            if not same and localize:
                if sess_all is None:
                    sess_all = ort_session(model, extra_outputs=[nd.output[0] for nd in model.graph.node])
                full = dict(zip([o.name for o in sess_all.get_outputs()], sess_all.run(None, f_onnx)))
                for nd in model.graph.node:
                    t = nd.output[0]
                    if t in full and t in ctx and not np.array_equal(np.asarray(full[t]), np.asarray(ctx[t])):
                        xc["first_differing_ops"].append(nd.op_type)
                        break
                del full
            del ctx
        del o32
        env32 = executor.run(g, f_ir, keep=ir_keep)
        ir32.append(np.asarray(env32[g.outputs[0]], np.float32).reshape(-1))
        d32 = ir_dec(env32)
        del env32
        env64 = executor.run(g, f_ir, keep=ir_keep, float_dtype=np.float64)
        ir64.append(np.asarray(env64[g.outputs[0]], np.float64).reshape(-1))
        d64 = ir_dec(env64)
        del env64
        rec = {"sample": i}
        for tag, dirs, src in (("ir_f64_vs_source_f64", d64, s64), ("ir_f32_vs_source_f32", d32, s32)):
            for b, dir_ in zip(bins, dirs):
                dsrc, apv = src[b["pre"]]
                if dsrc.shape != dir_.shape:
                    raise AssertionError(f"binarizer {b['onnx_node']}: source {dsrc.shape} vs IR {dir_.shape}")
                d = np.where(dsrc != dir_)[0]
                if d.size:
                    rec[tag] = {"onnx_node": b["onnx_node"], "n_flips": int(d.size), "max_abs_pre_source": float(apv[d].max()),
                                "typical_abs_pre_source": float(np.median(apv))}
                    break
        if len(rec) > 1:
            flips.append(rec)
        if (i + 1) % 16 == 0 or i + 1 == n:
            print(f"[verify] {i + 1}/{n} samples, {len(flips)} with a binarization flip", flush=True)
    ref64, ref32, ir32, ir64 = np.stack(ref64), np.stack(ref32), np.stack(ir32), np.stack(ir64)
    ref32.tofile(os.path.join(out_dir, "reference_logits.f32"))
    ref64.tofile(os.path.join(out_dir, "reference_logits_f64.f64"))
    ir32.tofile(os.path.join(out_dir, "numpy_logits_fused.f32"))
    ir64.tofile(os.path.join(out_dir, "numpy_logits_f64.f64"))
    sem = compare_outputs(ref64, ir64, labels)
    per = np.abs(ir64 - ref64).max(axis=1)
    sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per <= 1e-6))
    sem["deviating_samples_gt_1e-6"] = [int(k) for k in np.where(per > 1e-6)[0]]
    sem["per_sample_max_abs"] = [float(v) for v in per]
    floor = compare_outputs(ref64, ref32, labels)
    vs64 = compare_outputs(ref64, ir32, labels)
    slack = max(1, int(round(0.01 * n)))
    return {
        "n_samples": int(n),
        "reference_f32": "the canonical file in onnxruntime (one session, ORT_DISABLE_ALL; BipolarQuant / the constant "
                         "ternary IntQuant by their standard-op definitions)",
        "reference_f32_vs_qonnx_execute_onnx": xc,
        "reference_f64": "bitino.onnx_f64.OnnxF64 (float64)",
        "reference_f32_vs_f64": floor,
        "semantic_check_ir_f64_vs_reference_f64": sem,
        "numpy_f32_vs_reference_f32": compare_outputs(ref32, ir32, labels),
        "numpy_f32_vs_reference_f64": vs64,
        "binarization_flips_vs_source": {"n_binarizers": len(bins), "n_samples_with_flips": len(flips), "samples": flips},
        "acceptance": {"criterion": f"numpy executor (fp32) top-1 agreement vs fp64 >= 0.98 and disagreements <= source fp32 noise "
                                    f"floor ({floor['n_top1_disagreements']}) + {slack}",
                       "passed": bool(vs64["top1_agreement"] >= 0.98 and vs64["n_top1_disagreements"] <= floor["n_top1_disagreements"] + slack)},
    }


def run_bench(bench: str, model: str, inputs: str, out: str, stage: str) -> np.ndarray:
    import subprocess
    subprocess.run([bench, "--model", model, "--inputs", inputs, "--outputs", out, "--stage", stage, "--threads", "1",
                    "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
    return np.fromfile(out, np.float32)


def main_groups(args, model, prov: dict) -> int:
    """tools/import_qonnx.py / bitino import-qonnx for a group-mapped file (dispatched by qonnx_import.main)."""
    import datetime
    import os
    import platform

    import onnx

    from . import format as bfformat
    from .fetch import data_sha256, graph_sha256
    if args.cleanup != "none":
        print("note: group-mapped file: qonnx cleanup is not run (the file is the deployed graph; --cleanup none)", file=sys.stderr)
    try:
        g, info = import_groups(model)
    except GroupImportError as e:
        print(f"IMPORT FAILED: {e}", file=sys.stderr)
        json.dump({"model_id": args.model_id, "import_error": str(e), "node": e.node_name, "op": e.op},
                  open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
        return 4
    md = {p.key: p.value for p in model.metadata_props}
    g.manifest = {
        "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": "fused",
        "source": {"framework": "QONNX", **prov},
        "conversion": {"tool": "bitino.qonnx_import (group mapping, bitino.qonnx_groups)", "mapping": MAPPING,
                       "rules": info["rules"], "passes": info["passes"],
                       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "python": platform.python_version(), "numpy": np.__version__, "onnx": onnx.__version__},
        "input": {"names": [i["name"] for i in g.inputs], "shapes": [i["shape"] for i in g.inputs],
                  "dtypes": [i["dtype"] for i in g.inputs], "preprocessing": md.get("bitino.preprocessing", ""),
                  "input_domain": md.get("bitino.input_domain", "")},
        "outputs": ["logits"], "op_versions": {"format": 1},
    }
    path = os.path.join(args.out_dir, args.file_name)
    file_sha = bfformat.write(g, path)
    g = bfformat.read(path)
    report = {"model_id": args.model_id, "file": args.file_name, "file_sha256": file_sha, "data_sha256": data_sha256(path),
              "graph_sha256": graph_sha256(path), "file_bytes": os.path.getsize(path), "op_counts": g.op_counts(),
              "provenance": prov, "cleanup": {"cleanup": "none"}, "import": {k: v for k, v in info.items() if k != "onnx_tags"}}
    print(f"[import] {args.model_id}: {len(g.nodes)} IR nodes data {report['data_sha256'][:16]} graph {report['graph_sha256'][:16]}")
    ok = True
    if args.eval_dir:
        em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
        X = load_eval_inputs(args.eval_dir, args.preprocess, g)
        labels = np.array([s["label"] for s in em["samples"]][: len(next(iter(X.values())))])
        report["eval_set"] = em["name"]
        report["eval_preprocessing"] = args.preprocess
        v = verify_groups(model, g, info, X, labels, args.out_dir)
        report["verification"] = v
        src = json.load(open(args.source_json)) if args.source_json else {}
        from .validation.metrics import compare_outputs
        n = len(labels)
        for key, fname, tag in (("logits_f32", "reference_logits.f32", "source_framework_f32_vs_qonnx_f32 (export check)"),
                                ("logits_f64", "reference_logits_f64.f64", "source_framework_f64_vs_qonnx_f64 (export check)")):
            if src.get(key) and os.path.exists(src[key]):
                dt = np.float32 if key == "logits_f32" else np.float64
                s_ = np.fromfile(src[key], dt)
                r_ = np.fromfile(os.path.join(args.out_dir, fname), dt).reshape(n, -1)
                sel = src.get("source_sample_indices")     # the source outputs cover a subset of the check set
                lab = labels
                s_ = s_.reshape(n, -1) if sel is None else s_
                if sel is not None:
                    idx = np.asarray(sel, int)
                    s_, r_, lab = s_.reshape(len(idx), -1), r_[idx], labels[idx]
                if s_.shape[1] == 1 and r_.shape[1] == 2:
                    # a single-logit source (the FTL detectors): the IR head outputs [0, logit], as the converter
                    s_ = np.concatenate([np.zeros_like(s_), s_], axis=1)
                try:
                    v[tag] = compare_outputs(s_, r_, lab)
                except AssertionError as e:
                    v[tag] = {"error": f"source outputs {s_.shape} vs QONNX {r_.shape}: {e}"}
        sem = v["semantic_check_ir_f64_vs_reference_f64"]
        print(f"[verify] f64 semantic: max_abs {sem['max_abs_error']:.3e} <=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{n} | "
              f"flips {v['binarization_flips_vs_source']['n_samples_with_flips']} | ort vs execute_onnx "
              f"{v['reference_f32_vs_qonnx_execute_onnx']}")
        if args.bench:
            pre = em["preprocessing"][args.preprocess]
            inp = os.path.join(args.eval_dir, pre["file"])
            ir32 = np.fromfile(os.path.join(args.out_dir, "numpy_logits_fused.f32"), np.float32)
            report["cpp_host"] = []
            for st in [s for s in args.stages.split(",") if s]:
                o = os.path.join(args.out_dir, f"cpp_{st}.f32")
                y = run_bench(args.bench, path, inp, o, st)
                os.remove(o)
                exact = y.shape == ir32.shape and bool(np.array_equal(y.view(np.uint32), ir32.view(np.uint32)))
                report["cpp_host"].append({"stage": st, "exact_vs_numpy": exact if st == "reference" else None,
                                           "max_abs_vs_numpy": float(np.max(np.abs(y - ir32))) if y.shape == ir32.shape else None})
                print(f"[cpp] stage {st}: exact vs numpy {exact}")
            ok = all(c["exact_vs_numpy"] for c in report["cpp_host"] if c["stage"] == "reference")
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=_json_default)
    return 0 if ok else 1


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)
