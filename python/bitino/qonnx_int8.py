"""Step 5 of the QONNX route (docs/qonnx-frontend.md § Plan, docs/qonnx-step5.md): int8_full files as canonical QONNX.

An int8_full file (``model_int8_full*.bitino``, stage ``int8_full``: the ``I8*`` operators of python/bitino/int8_ops.py)
is written as a canonical file of the Bitino profile (docs/qonnx-profile.md § 5) with the group mapping of step 3
(python/bitino/qonnx_groups.py, ``bitino.import = {"mapping": "groups"}``): every IR node of the published int8 graph
is one group of standard ONNX + QONNX nodes in the pattern of its operator, evaluated in the element types written in
the file (INT8 activations, ``ConvInteger`` / ``MatMulInteger``, INT64 integer epilogues with the UINT64 ``BitShift``
shift construction, FLOAT epilogues one IEEE operation per node, the log2 code as a 256-entry ``Gather`` table (decode)
and the 127-compare form (encode)). Every pattern is class B (bit-exact with int8_ops, hence with the C++ int8_full
kernels): onnxruntime execution of the canonical file reproduces the int8_full outputs bit for bit.

Additions to the step-3 conventions (all naming metadata, none changes a value):

* ``bitino.ir`` key ``outputs``: the IR output tensor names when they are not ``<name>:out`` (one output) /
  ``<name>:out<i>`` (several): the int8 rewrites name outputs after the float graph (``layer1.0.prelu:out``,
  ``conv1:out:bits``);
* ``bitino.attrs`` rule (c), informational attributes: an attribute that the operator does not read in the node's
  configuration (``y_exp`` of an l8 output, ``pad_value`` of an unpadded binary convolution, ``count_include_pad``
  of an unpadded pool, the ``zeros`` / ``shifts`` entries of an I8Concat input that is copied, ``axis`` of a
  per-tensor I8Lut) cannot be derived from the semantics; its value is carried by ``bitino.attrs`` (a list attribute
  entry by entry: the derived entries must agree). It is the only metadata that supplies an attribute value, and only
  where no computed value depends on it.

Writer: ``write_spec_int8(graph)`` (numpy only). Importer: ``import_int8(model)`` (``qonnx_groups.import_groups``
dispatches here for ``bitino.variant`` int8_full*). Every reader extracts the IR node from its group and then
**re-emits** the group with the writer from the extracted node and requires the re-emitted ONNX nodes to equal the
file's (operators, attributes, wiring, constant values): a file is imported only if each group is exactly the
canonical pattern of the IR node it becomes.
"""
from __future__ import annotations

import json

import numpy as np

from . import int8_ops
from .ir import Constant, Graph, Node, unpack_bits
from .qonnx_groups import (ATTRS_KEY, BOOL, C2I, CONSTS_KEY, DEFAULT_LAYOUTS, DOUBLE, FLOAT, I2C, INT8, INT32, INT64,
                           IR_KEY, NAME_KEY, QD, UINT32, GroupImporter, GroupImportError, GroupWriter, Spec, _View,
                           _json_num_same, _onnx_pads)

UINT64 = 13
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1

SPATIAL_I8 = ("I8Conv2D", "I8BConv2D", "I8BConv2DRes", "I8BConv2DResI", "I8MaxPool2D", "I8AvgPool2D",
              "I8GlobalAvgPool", "I8GapAffine", "I8GatedAdd", "QBConv2D")
ELEMWISE_I8 = ("I8Quantize", "I8Dequantize", "I8Lut", "I8Add", "I8Mul", "I8AffineSum", "I8Concat", "I8Pad")
UNUSED_KEY = "bitino.unused_constants"
OUT_KEY = "outputs"


class _Info:
    """An attribute (or list entry) the operator does not read in this configuration (bitino.attrs rule (c))."""

    def __repr__(self):
        return "INFO"


INFO = _Info()


class _Axis:
    """An axis derived from the pattern (its non-negative index k in a tensor of rank r): the IR may write it as k or
    k - r, or leave it absent at its default (-1); the form is carried by bitino.attrs like a JSON number form."""

    def __init__(self, k: int, r: int):
        self.k, self.r = int(k), int(r)

    def same(self, v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool) and -self.r <= v < self.r and v % self.r == self.k

    def __repr__(self):
        return f"axis({self.k} of rank {self.r})"

# attributes an int8_full IR node may omit: int8_ops' defaults (a.get(key, default))
I8_DEFAULTS = {
    ("I8Conv2D", "dilations"): [1, 1], ("I8Conv2D", "groups"): 1, ("I8Conv2D", "output"): "i8",
    ("I8Conv2D", "qmin"): -128, ("I8Conv2D", "qmax"): 127,
    ("I8Dense", "output"): "i8", ("I8Dense", "qmin"): -128, ("I8Dense", "qmax"): 127,
    ("I8MatMul", "output"): "i8", ("I8MatMul", "qmin"): -128, ("I8MatMul", "qmax"): 127, ("I8MatMul", "transpose_b"): 0,
    ("I8BDense", "output"): "i8", ("I8BDense", "qmin"): -128, ("I8BDense", "qmax"): 127,
    ("I8BConv2D", "output"): "i8", ("I8BConv2D", "qmin"): -128, ("I8BConv2D", "qmax"): 127,
    ("I8BConv2D", "dilations"): [1, 1], ("I8BConv2D", "groups"): 1,
    ("I8BConv2DRes", "output"): "l8", ("I8BConv2DRes", "bit_binarize"): "ge", ("I8BConv2DRes", "bit_src"): "sum",
    ("I8BConv2DRes", "r_code"): "l8", ("I8BConv2DRes", "dilations"): [1, 1], ("I8BConv2DRes", "groups"): 1,
    ("I8BConv2DRes", "qmin"): -128, ("I8BConv2DRes", "qmax"): 127, ("I8BConv2DRes", "y_zero"): 0,
    ("I8BConv2DResI", "bit_src"): "sum", ("I8BConv2DResI", "slope_shift"): 0, ("I8BConv2DResI", "r_tile"): 1,
    ("I8BConv2DResI", "dilations"): [1, 1], ("I8BConv2DResI", "groups"): 1,
    ("I8BMatMul", "a_threshold"): 0.0, ("I8BMatMul", "b_threshold"): 0.0, ("I8BMatMul", "output"): "i8",
    ("I8BMatMul", "qmin"): -128, ("I8BMatMul", "qmax"): 127,
    ("I8LayerNorm", "qmin"): -128, ("I8LayerNorm", "qmax"): 127,
    ("I8Softmax", "output"): "i8", ("I8Softmax", "qmin"): -128, ("I8Softmax", "qmax"): 127,
    ("I8AvgPool2D", "count_include_pad"): 0,
    ("I8GlobalAvgPool", "keepdims"): 0, ("I8ReduceMean", "keepdims"): 0,
    ("I8Lut", "output"): "i8", ("I8Lut", "axis"): -1,
}
for _op in ("I8Conv2D", "I8Dense", "I8MatMul", "I8BDense", "I8BConv2D", "I8BConv2DRes", "I8BConv2DResI", "I8BMatMul",
            "I8LayerNorm", "I8Softmax", "I8Add", "I8Mul", "I8AffineSum", "I8GatedAdd", "I8GapAffine"):
    I8_DEFAULTS.setdefault((_op, "qmin"), -128)
    I8_DEFAULTS.setdefault((_op, "qmax"), 127)
for _op in ("I8Conv2D", "I8BConv2D", "I8BConv2DRes", "I8BConv2DResI"):
    I8_DEFAULTS.setdefault((_op, "strides"), [1, 1])
    I8_DEFAULTS.setdefault((_op, "pads"), [0, 0, 0, 0])
for _op in ("I8BDense", "I8BConv2D", "I8BConv2DRes", "I8BConv2DResI"):
    I8_DEFAULTS.setdefault((_op, "input_binarize"), "ge")
for _k, _ops in (("x_zero", ("I8Conv2D", "I8Dense", "I8AvgPool2D", "I8GlobalAvgPool", "I8ReduceMean", "I8GatedAdd")),
                 ("y_zero", ("I8Conv2D", "I8Dense", "I8MatMul", "I8BConv2DResI", "I8GatedAdd", "I8GapAffine")),
                 ("a_zero", ("I8Add", "I8Mul", "I8MatMul")), ("b_zero", ("I8Add", "I8Mul", "I8MatMul")),
                 ("p_zero", ("I8GatedAdd",)), ("g_zero", ("I8GatedAdd", "I8BConv2DRes")), ("shift", ("I8BConv2DResI", "I8GatedAdd", "I8GapAffine")),
                 ("pad_q", ("I8Pad",)), ("fill_q", ("I8Gather",)), ("r_zero", ("I8BConv2DRes",)), ("c_rs", ("I8BMatMul",)),
                 ("c_cs", ("I8BMatMul",)), ("c0", ("I8BMatMul",)), ("transpose_b", ("I8BMatMul",))):
    for _op in _ops:
        I8_DEFAULTS.setdefault((_op, _k), 0)
I8_DEFAULTS.update({("I8Add", "left_shift"): 20, ("I8BMatMul", "c_ab"): 1, ("I8BMatMul", "a_binarize"): "sign_zero",
                    ("I8BMatMul", "b_binarize"): "sign_zero", ("I8LayerNorm", "eps"): 1e-5, ("I8Concat", "axis"): -1,
                    ("I8ReduceMean", "axis"): -1, ("I8AvgPool2D", "pads"): [0, 0, 0, 0], ("I8MaxPool2D", "pads"): [0, 0, 0, 0],
                    # defaults that depend on another attribute (int8_ops: strides = kernel, zeros = 0 per input)
                    ("I8AvgPool2D", "strides"): lambda c: c.get("kernel"), ("I8MaxPool2D", "strides"): lambda c: c.get("kernel"),
                    ("I8GapAffine", "zeros"): lambda c: [0] * len(c.get("zeros") or [])})


def _default(op: str, k: str, canon: dict):
    d = I8_DEFAULTS.get((op, k), ())
    return d(canon) if callable(d) else d
DEFAULT_LAYOUTS_I8 = dict(DEFAULT_LAYOUTS)
DEFAULT_LAYOUTS_I8.update({("I8BConv2D", 1): "OHWI_bit", ("I8BConv2DRes", 1): "OHWI_bit",
                           ("I8BConv2DResI", 1): "OHWI_bit", ("QBConv2D", 1): "OHWI_bit"})


def _has_info(v) -> bool:
    return v is INFO or (isinstance(v, list) and any(x is INFO for x in v))


def _same(a, b) -> bool:
    return json.dumps(a) == json.dumps(b) or _json_num_same(a, b)


def attrs_meta_i8(op: str, ir_attrs: dict, canon: dict) -> dict:
    """bitino.attrs of an int8 group (docs/qonnx-profile.md § 7): (a) null for an absent attribute at its default and
    the IR's JSON form of the same number, (b) float64 values of float32 attributes, (c) the IR value of an
    informational attribute (INFO in canon; list entries: the derived entries must agree). Raises on a value change."""
    out = {}
    for k in sorted(set(canon) | set(ir_attrs)):
        c = canon.get(k, None)
        if k not in canon:
            raise AssertionError(f"{op}: IR attribute {k!r} is not derived by the reader")
        if c is INFO:
            if k in ir_attrs:
                out[k] = ir_attrs[k]
            continue
        if isinstance(c, _Axis):
            a = ir_attrs.get(k, I8_DEFAULTS.get((op, k)))
            if not c.same(a):
                raise AssertionError(f"{op}: attribute {k}: IR {a!r} vs derived {c!r}")
            out[k] = ir_attrs[k] if k in ir_attrs else None
            continue
        if isinstance(c, list) and any(x is INFO for x in c):
            a = ir_attrs.get(k)
            if not isinstance(a, list) or len(a) != len(c) or not all(x is INFO or _same(x, y) for x, y in zip(c, a)):
                raise AssertionError(f"{op}: attribute {k}: IR {a!r} vs derived {c!r}")
            out[k] = a
            continue
        if k not in ir_attrs:
            d = _default(op, k, canon)
            if d == () or not _same(c, d) or type(c) is not type(d):
                raise AssertionError(f"{op}: attribute {k!r} absent in the IR node but the reader gives {c!r} "
                                     "(not the operator default)")
            out[k] = None
            continue
        a = ir_attrs[k]
        if json.dumps(a) == json.dumps(c):
            continue
        if _json_num_same(a, c) or (isinstance(a, float) and isinstance(c, float) and float(np.float32(a)) == c):
            out[k] = a
            continue
        raise AssertionError(f"{op}: attribute {k}: IR {a!r} vs reader {c!r}")
    return out


def apply_attrs_meta_i8(op: str, canon: dict, meta: dict, group: str) -> dict:
    out = dict(canon)
    for k, v in meta.items():
        if k not in out:
            raise GroupImportError(group, f"bitino.attrs: the reader derives no attribute {k!r}", op)
        cur = out[k]
        if cur is INFO:
            if v is None:
                del out[k]
            else:
                out[k] = v
            continue
        if isinstance(cur, _Axis):
            d = I8_DEFAULTS.get((op, k)) if v is None else v
            if not cur.same(d):
                raise GroupImportError(group, f"bitino.attrs {k}={v!r} is not the derived {cur!r}", op)
            if v is None:
                del out[k]
            else:
                out[k] = v
            continue
        if isinstance(cur, list) and any(x is INFO for x in cur):
            if not isinstance(v, list) or len(v) != len(cur) or not all(x is INFO or _same(x, y) for x, y in zip(cur, v)):
                raise GroupImportError(group, f"bitino.attrs {k}={v!r} contradicts the derived entries {cur!r}", op)
            out[k] = v
            continue
        if v is None:
            d = _default(op, k, canon)
            if d == () or not _same(cur, d) or type(cur) is not type(d):
                raise GroupImportError(group, f"bitino.attrs {k}=null would change the value {cur!r}", op)
            del out[k]
            continue
        if _json_num_same(v, cur) or (isinstance(v, float) and isinstance(cur, float) and float(np.float32(v)) == cur):
            out[k] = v
            continue
        raise GroupImportError(group, f"bitino.attrs {k}={v!r} would change the value {cur!r}", op)
    for k in [k for k, v in out.items() if v is INFO]:
        del out[k]                      # informational and absent in the IR node
    for k in [k for k, v in out.items() if isinstance(v, _Axis)]:
        raise GroupImportError(group, f"axis attribute {k} without its bitino.attrs form", op)
    left = [k for k, v in out.items() if _has_info(v)]
    if left:
        raise GroupImportError(group, f"informational list entries of {left} without a bitino.attrs value", op)
    return out


# ================================================================================================ the l8 code
def l8_boundaries() -> np.ndarray:
    """beta[k - 1] = the smallest positive fp32 whose l8 code (exponent 0) is >= k, k = 1..127 (L8ENC, docs/qonnx-profile.md
    § 5). int8_ops.l8_encode is non-decreasing in |w| (in the fp32 bit pattern of positive w), so the code of w is the
    number of boundaries <= |w|: each boundary is found by bisection over the positive fp32 bit patterns."""
    out = np.empty(127, np.float32)
    lo_all = np.uint32(1)
    for k in range(1, 128):
        lo, hi = int(lo_all), 0x7F800000            # code(hi = +inf) = 127 >= k
        while lo < hi:
            mid = (lo + hi) // 2
            c = int(int8_ops.l8_encode(np.array([mid], np.uint32).view(np.float32))[0])
            if c >= k:
                hi = mid
            else:
                lo = mid + 1
        out[k - 1] = np.array([lo], np.uint32).view(np.float32)[0]
        lo_all = np.uint32(lo)
    return out


_BETA = None
L8ENC_STEPS = 7          # 2^7 = 128 > 127 boundaries: the bisection's count has 7 bits


def l8enc_bisect(w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """numpy mirror of the L8ENC pattern (Int8Writer.l8enc): sign(w) * the 7-step bisection count, as int8."""
    b = beta() if b is None else b
    a = np.abs(np.asarray(w, np.float32))
    cnt = np.zeros(a.shape, np.int64)
    for j in range(L8ENC_STEPS - 1, -1, -1):
        cnt += np.where(a >= b[cnt + (1 << j) - 1], 1 << j, 0)
    return (np.sign(w).astype(np.int64) * cnt).astype(np.int8)


def beta() -> np.ndarray:
    global _BETA
    if _BETA is None:
        _BETA = l8_boundaries()
    return _BETA


def l8_table(e: int) -> np.ndarray:
    """D_E[q + 128] = l8_decode(q, E), q = -128..127 (L8DEC)."""
    return int8_ops.l8_decode(np.arange(-128, 128, dtype=np.int64), int(e)).astype(np.float32)


def _l8_exp_of(tab: np.ndarray):
    """E of a 256-entry l8 decode table (D_E[255] = 2^E), None if the table is not an l8 table."""
    t = np.asarray(tab, np.float32).reshape(-1)
    if t.size != 256 or not np.isfinite(t[255]) or t[255] <= 0:
        return None
    m, e = np.frexp(np.float64(t[255]))
    if m != 0.5:
        return None
    e = int(e) - 1
    try:
        return e if np.array_equal(l8_table(e).view(np.uint32), t.view(np.uint32)) else None
    except ValueError:
        return None


def pool_counts(h: int, w: int, oh: int, ow: int, k, st, pads, inc: int) -> np.ndarray:
    """Taps per output position of an I8AvgPool2D window (int8_ops._pool: kh * kw with count_include_pad, else the
    in-bounds taps), at least 1."""
    cnt = np.empty((oh, ow), np.int64)
    for oy in range(oh):
        for ox in range(ow):
            y0, x0 = oy * st[0] - pads[0], ox * st[1] - pads[2]
            ys, ye, xs, xe = max(y0, 0), min(y0 + k[0], h), max(x0, 0), min(x0 + k[1], w)
            cnt[oy, ox] = k[0] * k[1] if inc else (ye - ys) * (xe - xs)
    return np.maximum(cnt, 1)


def _fp32_exact(v: float) -> bool:
    return float(np.float32(v)) == float(v)


# ================================================================================================ writer
class Int8Writer(GroupWriter):
    """Published int8_full IR graph -> canonical spec, one group per IR node (module docstring)."""

    def __init__(self, g: Graph, shapes: dict, kinds: dict):
        self.g = g
        self.shapes = {k: tuple(int(d) for d in v) for k, v in shapes.items()}
        for c in g.constants.values():
            self.shapes.setdefault(c.name, tuple(int(d) for d in c.shape))
        self.kinds = dict(kinds)
        self.s = Spec()
        self.report = {"groups": 0, "onnx_nodes": 0, "layout_transposes": 0, "carriers": 0, "groups_by_op": {},
                       "attrs_meta": 0, "informational_attrs": 0}
        self.consts = g.constants
        self.prod = {o: n for n in g.nodes for o in n.outputs}
        self.inputs = {i["name"]: i for i in g.inputs}
        used = {i for n in g.nodes for i in n.inputs if i}
        self.unused = [k for k in self.consts if k not in used]
        for k in self.unused:
            c = self.consts[k]
            if c.dtype not in ("i8", "i32", "f32") or c.layout:
                raise AssertionError(f"unused constant {k} ({c.dtype}, layout {c.layout!r}) has no initializer form")
            self.s.init(k, np.asarray(c.data))

    # shared scalars, created on first use (no unused initializers)
    @property
    def ONE(self) -> str:
        return self.s.init("bitino/one", np.array(1.0, np.float32))

    @property
    def ZERO(self) -> str:
        return self.s.init("bitino/zero", np.array(0.0, np.float32))

    @property
    def M1(self) -> str:
        return self.s.init("bitino/minus_one", np.array(-1.0, np.float32))

    # ---------------------------------------------------------------- layout plan
    def node_mode(self, n, prim) -> str:
        rt = [i for i in n.inputs if i and i not in self.consts]
        cs = [i for i in n.inputs if i and i in self.consts]
        op = n.op
        if op in SPATIAL_I8:
            return "C"
        if op not in ELEMWISE_I8 or not rt or not all(self.rank(t) == 4 and prim[t] == "C" for t in rt):
            return "I"
        if op == "I8Lut" and np.asarray(self.cdata(n.inputs[1])).shape[0] != 1:
            return "I"
        if op == "I8Concat" and (cs or int(n.attrs.get("axis", -1)) not in (-1, 3)):
            return "I"
        if op in ("I8Add", "I8AffineSum") and any(self.rank(c) not in (4,) and c in self._act_consts(n) for c in cs):
            return "I"
        return "C"

    def _act_consts(self, n) -> set:
        """Constant inputs that stand for an activation operand (an int8 tensor broadcast against the runtime one)."""
        if n.op == "I8Add":
            return {i for i in n.inputs[:2] if i in self.consts}
        if n.op == "I8AffineSum":
            return {i for i in n.inputs[:-2] if i in self.consts}
        return set()

    def plan(self) -> None:
        prim: dict = {}
        for i in self.g.inputs:
            prim[i["name"]] = "C" if (len(i["shape"]) == 4 and i["dtype"] == "f32") else "I"
        mode: dict = {}
        for n in self.g.nodes:
            m = self.node_mode(n, prim)
            mode[n.name] = m
            for o in n.outputs:
                if n.op == "I8GlobalAvgPool" and not int(n.attrs.get("keepdims", 0)):
                    prim[o] = "I"
                else:
                    prim[o] = m if self.rank(o) == 4 else "I"
        self.prim, self.mode = prim, mode
        need: dict = {}
        for n in self.g.nodes:
            for t in n.inputs:
                if not t or t in self.consts:
                    continue
                want = mode[n.name] if self.rank(t) == 4 else "I"
                if want != prim[t]:
                    need.setdefault(t, set()).add(want)
        for t in self.g.outputs:
            if prim[t] != "I":
                need.setdefault(t, set()).add("I")
        self.need = need

    # ---------------------------------------------------------------- emission loop
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
            self.emit_node(n, idx)
        for o in self.g.outputs:
            et = {"i8": INT8, "f": FLOAT, "i32": INT32}[self.kinds[o]]
            s.outputs.append((self.onnx[(o, "I")], et, list(self.shapes[o])))
        return s

    def emit_node(self, n, idx: int) -> None:
        s = self.s
        self.cur = (n, idx)
        self.report["groups"] += 1
        self.report["groups_by_op"][n.op] = self.report["groups_by_op"].get(n.op, 0) + 1
        first = len(s.nodes)
        fn = getattr(self, f"w_{n.op}", None)
        if fn is None:
            raise AssertionError(f"{n.name}: IR operator {n.op} has no step-5 pattern")
        res = fn(n)
        if n.op.startswith("I8") or n.op == "QBConv2D" or n.op == "QBDense":
            outs, canon = res
        else:                                   # step-3 writers (Reshape): (out, tag, canon)
            outs, canon = [(res[0], res[1])], res[2]
        if len(outs) != len(n.outputs):
            raise AssertionError(f"{n.name}: {len(outs)} pattern outputs for {len(n.outputs)} IR outputs")
        for (t, tag), o in zip(outs, n.outputs):
            self.onnx[(o, tag)] = t
        for (t, tag), o in zip(outs, n.outputs):
            for want in sorted(self.need.get(o, ())):
                if want != tag:
                    r = self.emit("Transpose", [t], {"perm": list(C2I if tag == "C" else I2C)},
                                  out=f"{t}:{'nhwc' if want == 'I' else 'nchw'}")
                    self.report["layout_transposes"] += 1
                    self.onnx[(o, want)] = r
        gm = {}
        consts = {str(j): c for j, c in enumerate(n.inputs) if c and c in self.consts}
        if consts:
            gm[CONSTS_KEY] = json.dumps(consts, sort_keys=True)
        am = attrs_meta_i8(n.op, n.attrs, canon)
        if am:
            gm[ATTRS_KEY] = json.dumps(am, sort_keys=True)
            self.report["attrs_meta"] += 1
            self.report["informational_attrs"] += sum(1 for k in am if _has_info(canon.get(k)))
        ir = {"op": n.op}
        na = len(n.inputs)
        trimmed = na
        while trimmed and not n.inputs[trimmed - 1]:
            trimmed -= 1
        if na != trimmed:
            ir["n_inputs"] = na
        lay = {str(j): self.consts[c].layout for j, c in enumerate(n.inputs) if c in self.consts
               and self.consts[c].layout != DEFAULT_LAYOUTS_I8.get((n.op, j), "")}
        if lay:
            ir["layouts"] = lay
        default = [f"{n.name}:out"] if len(n.outputs) == 1 else [f"{n.name}:out{i}" for i in range(len(n.outputs))]
        if list(n.outputs) != default:
            ir[OUT_KEY] = list(n.outputs)
        gm[IR_KEY] = json.dumps(ir, sort_keys=True)
        target = next(nd for nd in s.nodes[first:] if nd["meta"].get(NAME_KEY) == n.name)
        target["meta"].update(gm)

    # ---------------------------------------------------------------- constants
    def i64(self, v: int) -> str:
        return self.s.init(f"bitino/i64/{int(v)}", np.array(int(v), np.int64))

    def u64(self, v: int) -> str:
        return self.s.init(f"bitino/u64/{int(v)}", np.array(int(v), np.uint64))

    def i8s(self, v: int) -> str:
        if not -128 <= int(v) <= 127:
            raise AssertionError(f"INT8 scalar {v} out of range")
        return self.s.init(f"bitino/i8/{int(v)}", np.array(int(v), np.int8))

    def f32s(self, v: float) -> str:
        if not _fp32_exact(v):
            raise AssertionError(f"FLOAT scalar {v!r} is not an fp32 value")
        return self.s.init(f"bitino/c/{float(v).hex()}", np.array(v, np.float32))

    def fval(self, v: float) -> str:
        """An fp32-evaluated attribute value: a FLOAT scalar when it is an fp32 value, else a float64 carrier (R7)."""
        return self.f32s(v) if _fp32_exact(v) else self.carrier("attr", float(v))

    def dbl(self, v: float) -> str:
        return self.s.init(f"bitino/d/{float(v).hex()}", np.array(float(v), np.float64))

    def axes(self, ax) -> str:
        ax = [int(a) for a in ax]
        return self.s.init(f"bitino/axes/{'_'.join(map(str, ax))}", np.array(ax, np.int64))

    def chan(self, arr, mode: str, rank: int):
        """A per-channel vector (IR last axis) shaped for broadcasting: [C, 1, 1] against an NCHW view, else [C]."""
        a = np.asarray(arr).reshape(-1)
        if mode == "C" and rank == 4:
            return a.reshape(-1, 1, 1)
        return a

    def cchan(self, name: str, key: str, arr, mode: str, rank: int, dtype) -> str:
        return self.cinit(name, key, np.ascontiguousarray(self.chan(arr, mode, rank)).astype(dtype))

    def c_i8(self, name: str, mode: str, key: str = ":i8") -> str:
        """An int8 IR constant (an activation operand) as an INT8 initializer, NCHW in an NCHW group."""
        d = np.asarray(self.cdata(name), np.int8)
        if mode == "C" and d.ndim == 4:
            d = np.ascontiguousarray(np.transpose(d, I2C))
            key = key + ":nchw"
        return self.cinit(name, key, d)

    # ---------------------------------------------------------------- integer primitives (docs/qonnx-profile.md § 5)
    def asr(self, a: str, n, key: str | None = None) -> str:
        """floor(a / 2^n) = Sub(Cast(BitShift(Cast(Add(a, 2^62), UINT64), n, RIGHT), INT64), 2^(62 - n)), |a| < 2^62."""
        if np.ndim(n) == 0:
            n = int(n)
            N, OFF = self.u64(n), self.i64(1 << (62 - n))
        else:
            nn = np.asarray(n, np.int64)
            N = self.s.init(f"{key}:n", nn.astype(np.uint64))
            OFF = self.s.init(f"{key}:off", np.left_shift(np.int64(1), 62 - nn).astype(np.int64))
        t = self.emit("Add", [a, self.i64(1 << 62)])
        t = self.emit("Cast", [t], {"to": UINT64})
        t = self.emit("BitShift", [t, N], {"direction": "RIGHT"})
        t = self.emit("Cast", [t], {"to": INT64})
        return self.emit("Sub", [t, OFF])

    def sat(self, a: str, lo, hi, key: str | None = None) -> str:
        """clamp on INT64: Where(Less(Where(Greater(a, hi), hi, a), lo), lo, .) (no INT64 Clip, docs/qonnx-profile.md § 2)."""
        if np.ndim(lo) == 0:
            LO, HI = self.i64(lo), self.i64(hi)
        else:
            LO = self.s.init(f"{key}:lo", np.asarray(lo, np.int64))
            HI = self.s.init(f"{key}:hi", np.asarray(hi, np.int64))
        t = self.emit("Where", [self.emit("Greater", [a, HI]), HI, a])
        return self.emit("Where", [self.emit("Less", [t, LO]), LO, t])

    def mulhi(self, a: str, M, key: str | None = None) -> str:
        """(a * M + 2^30) >> 31 (SQRDMULH), |a|, M < 2^31."""
        Mc = self.i64(M) if np.ndim(M) == 0 else self.s.init(f"{key}:M", np.asarray(M, np.int64))
        return self.asr(self.emit("Add", [self.emit("Mul", [a, Mc]), self.i64(1 << 30)]), 31)

    def rshr(self, a: str, n, key: str | None = None) -> str:
        """(a + 2^(n-1)) >> n, identity rounding for n = 0."""
        if np.ndim(n) == 0:
            n = int(n)
            R = self.i64((1 << (n - 1)) if n > 0 else 0)
        else:
            nn = np.asarray(n, np.int64)
            R = self.s.init(f"{key}:R", np.where(nn > 0, np.left_shift(np.int64(1), np.maximum(nn - 1, 0)), 0).astype(np.int64))
        return self.asr(self.emit("Add", [a, R]), n, key)

    def requant(self, x: str, M, shift, key: str | None = None) -> str:
        """int8_ops.requant: rshr(mulhi(sat32(x * 2^max(shift, 0)), M), max(-shift, 0)) on INT64."""
        if np.ndim(shift) == 0:
            sh = int(shift)
            L = self.i64(1 << max(sh, 0))
            right = max(-sh, 0)
        else:
            sh = np.asarray(shift, np.int64)
            L = self.s.init(f"{key}:L", np.left_shift(np.int64(1), np.maximum(sh, 0)).astype(np.int64))
            right = np.maximum(-sh, 0)
        t = self.sat(self.emit("Mul", [x, L]), I32_MIN, I32_MAX)
        return self.rshr(self.mulhi(t, M, key), right, key)

    def to_i8(self, v: str, lo: int, hi: int, out: str | None = None) -> str:
        """INT64 value -> clamp(v, lo, hi) -> INT8."""
        return self.emit("Cast", [self.sat(v, lo, hi)], {"to": INT8}, out=out)

    def i64of(self, q: str) -> str:
        return self.emit("Cast", [q], {"to": INT64})

    def centered(self, q: str, z: int) -> str:
        """q - z on INT64."""
        return self.emit("Sub", [self.i64of(q), self.i64(z)])

    # ---------------------------------------------------------------- float primitives
    def deq(self, q: str, s: float, z: int, out: str | None = None) -> str:
        """fp32(fp32(q - z) * s) (I8Dequantize, the binarizers' deq)."""
        if abs(int(z)) >= 1 << 24:
            raise AssertionError(f"zero point {z} not exact in fp32")
        t = self.emit("Sub", [self.emit("Cast", [q], {"to": FLOAT}), self.f32s(float(int(z)))])
        return self.emit("Mul", [t, self.fval(s)], out=out)

    def l8dec(self, q: str, e: int) -> str:
        """L8DEC: Gather(D_E[256], Cast(q, INT64) + 128) -> FLOAT."""
        tab = self.s.init(f"bitino/l8dec/{int(e)}", l8_table(e))
        return self.emit("Gather", [tab, self.emit("Add", [self.i64of(q), self.i64(128)])], {"axis": 0})

    def l8enc(self, w: str, out: str | None = None) -> str:
        """L8ENC: sign(w) * #{k : |w| >= beta_k} as INT8, the count by an exact 7-step branchless bisection over the
        sorted boundaries (step 6, docs/qonnx-profile.md § 5): a = Abs(w); for j = 6..0:
        cnt += Where(GreaterOrEqual(a, Gather(beta, cnt + 2^j - 1)), 2^j, 0) (INT64; the first step reads beta[63]),
        y = Cast(Mul(Cast(Sign(w), INT64), cnt), INT8). Intermediates have the shape of w (the 127-compare form of
        step 5 had a [..., 127] comparison tensor)."""
        B = self.s.init("bitino/l8enc/beta", beta())
        a = self.emit("Abs", [w])
        cnt = None
        for j in range(L8ENC_STEPS - 1, -1, -1):
            idx = self.i64((1 << j) - 1) if cnt is None else self.emit("Add", [cnt, self.i64((1 << j) - 1)])
            b = self.emit("Gather", [B, idx], {"axis": 0})
            add = self.emit("Where", [self.emit("GreaterOrEqual", [a, b]), self.i64(1 << j), self.i64(0)])
            cnt = add if cnt is None else self.emit("Add", [cnt, add])
        sg = self.emit("Cast", [self.emit("Sign", [w])], {"to": INT64})
        return self.emit("Cast", [self.emit("Mul", [sg, cnt])], {"to": INT8}, out=out)

    def rne8(self, v: str, qmin: int, qmax: int, out: str | None = None) -> str:
        """clamp(rne(v)) -> INT8 (FLOAT ops; integral bounds)."""
        c = self.emit("Clip", [self.emit("Round", [v]), self.f32s(float(qmin)), self.f32s(float(qmax))])
        return self.emit("Cast", [c], {"to": INT8}, out=out)

    def bits_of(self, s: str, thr: str, strict: bool = False, out: str | None = None) -> str:
        """A bit tensor (+-1 FLOAT): +1 iff s >= thr (s > thr)."""
        c = self.emit("Greater" if strict else "GreaterOrEqual", [s, thr])
        return self.emit("Where", [c, self.ONE, self.M1], out=out)

    # ---------------------------------------------------------------- binary operands
    def bin_input(self, n, x: str, mode: str, thr: str | None, cshape) -> tuple:
        """The +-1 operand of a binary layer: a bit tensor as is, or Bin(deq(q) [- t]) of int8 codes (linear grid:
        x_scale / x_zero; l8 grid: x_code l8, x_exp). Returns (onnx tensor, canon attrs)."""
        a = n.attrs
        xo = self.rt(x, mode)
        if self.kinds[x] != "i8":
            return xo, {}
        canon = {"input_binarize": a.get("input_binarize", "ge")}
        if a.get("x_code", "linear") == "l8":
            v = self.l8dec(xo, int(a["x_exp"]))
            canon.update({"x_code": "l8", "x_exp": int(a["x_exp"])})
        else:
            v = self.deq(xo, float(a["x_scale"]), int(a["x_zero"]))
            canon.update({"x_scale": float(a["x_scale"]), "x_zero": int(a["x_zero"])})
        return self.binarize(v, canon["input_binarize"], thr, cshape), canon

    def bin_scalar(self, x: str, mode: str, scale: float, zero: int, t, cst: bool) -> str:
        """Bin(deq(q) [- t]) with a scalar threshold attribute (I8BMatMul); cst: x is an int8 constant."""
        src = self.c_i8(x, "I") if cst else self.rt(x, "I")
        v = self.deq(src, scale, zero)
        if t is not None:
            v = self.emit("Sub", [v, self.f32s(float(t))])
        return self.binarize(v, mode, None, None)

    def pad_binary(self, xb: str, a: dict) -> tuple:
        """Explicit binary Pad(v) (R2); returns (tensor, canon pads / pad_value)."""
        pads = [int(p) for p in a.get("pads", [0, 0, 0, 0])]
        pv = a.get("pad_value", 0)
        if any(pads):
            t, b_, l, r = pads
            if float(pv) not in (-1.0, 0.0, 1.0):
                raise AssertionError(f"binary pad value {pv}")
            xb = self.emit("Pad", [xb, self.s.init(f"bitino/pads/{t}_{l}_{b_}_{r}", np.array([0, 0, t, l, 0, 0, b_, r], np.int64)),
                                   self.f32s(float(pv))], {"mode": "constant"})
            return xb, {"pads": pads, "pad_value": float(pv)}
        return xb, {"pads": pads, "pad_value": INFO}

    def bconv(self, n, xb: str, w: str) -> tuple:
        """Conv(xb, +-1 OIHW weights, pads 0) on FLOAT: the exact binary dot product (|acc| <= K < 2^24)."""
        a = n.attrs
        bits = self.cdata(w).astype(np.int8)                      # OHWI
        o, kh, kw, ci = bits.shape
        wt = self.c_pm1(w, np.ascontiguousarray(np.transpose(bits, (0, 3, 1, 2))), ":oihw")
        y = self.emit("Conv", [xb, wt], {"kernel_shape": [kh, kw], "strides": list(a.get("strides", [1, 1])), "pads": [0, 0, 0, 0],
                                         "dilations": list(a.get("dilations", [1, 1])), "group": int(a.get("groups", 1))})
        return y, {"strides": list(a.get("strides", [1, 1])), "dilations": list(a.get("dilations", [1, 1])),
                   "groups": int(a.get("groups", 1))}

    def _thr(self, n, j: int):
        return n.inputs[j] if len(n.inputs) > j and n.inputs[j] else None

    def _opt(self, n, j: int):
        return n.inputs[j] if len(n.inputs) > j and n.inputs[j] else None

    # ================================================================ operators: each returns ([(out, tag)], canon)
    def w_I8Quantize(self, n):
        m = self.mode[n.name]
        a = n.attrs
        x = self.rt(n.inputs[0], m)
        t = self.emit("Add", [self.emit("Round", [self.emit("Div", [x, self.fval(float(a["scale"]))])]), self.f32s(float(int(a["zero"])))])
        t = self.emit("Clip", [t, self.f32s(-128.0), self.f32s(127.0)])
        y = self.emit("Cast", [t], {"to": INT8}, out=n.outputs[0])
        return [(y, m)], {"scale": float(a["scale"]), "zero": int(a["zero"])}

    def w_I8Dequantize(self, n):
        m = self.mode[n.name]
        a = n.attrs
        y = self.deq(self.rt(n.inputs[0], m), float(a["scale"]), int(a["zero"]), out=n.outputs[0])
        return [(y, m)], {"scale": float(a["scale"]), "zero": int(a["zero"])}

    def w_I8Lut(self, n):
        m = self.mode[n.name]
        q, tab = n.inputs[0], n.inputs[1]
        L = np.asarray(self.cdata(tab))
        T = L.shape[0]
        canon = {}
        if L.dtype == np.float32:
            canon["output"] = "f32"
        elif L.dtype != np.int8:
            raise AssertionError(f"{n.name}: I8Lut table dtype {L.dtype}")
        qo = self.rt(q, m)
        idx = self.emit("Add", [self.i64of(qo), self.i64(128)])
        if T == 1:
            src = self.cinit(tab, ":row", np.ascontiguousarray(L[0]))
            if "axis" in n.attrs:
                canon["axis"] = INFO
        else:
            r = self.rank(q)
            ax = int(n.attrs.get("axis", -1)) % r
            if m != "I" or self.shapes[q][ax] != T:
                raise AssertionError(f"{n.name}: per-channel I8Lut along axis {ax} of {self.shapes[q]} ({T} rows)")
            shp = [1] * r
            shp[ax] = T
            off = self.s.init(f"{tab}:off", (np.arange(T, dtype=np.int64) * 256).reshape(shp))
            idx = self.emit("Add", [idx, off])
            src = self.cinit(tab, ":flat", np.ascontiguousarray(L.reshape(-1)))
            canon["axis"] = _Axis(ax, r)
        y = self.emit("Gather", [src, idx], {"axis": 0}, out=n.outputs[0])
        return [(y, m)], canon

    def _operand(self, t: str, mode: str) -> str:
        """An int8 operand (runtime tensor or int8 constant) in the group's mode."""
        if t in self.consts:
            return self.c_i8(t, mode)
        return self.rt(t, mode)

    def w_I8Add(self, n):
        m = self.mode[n.name]
        a = n.attrs
        ls = int(a.get("left_shift", 20))
        qa = self.centered(self._operand(n.inputs[0], m), int(a.get("a_zero", 0)))
        qb = self.centered(self._operand(n.inputs[1], m), int(a.get("b_zero", 0)))
        xa = self.requant(self.emit("Mul", [qa, self.i64(1 << ls)]), int(a["a_mult"]), int(a["a_shift"]))
        xb = self.requant(self.emit("Mul", [qb, self.i64(1 << ls)]), int(a["b_mult"]), int(a["b_shift"]))
        y = self.emit("Add", [self.requant(self.emit("Add", [xa, xb]), int(a["out_mult"]), int(a["out_shift"])), self.i64(int(a["y_zero"]))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        canon = {k: int(a.get(k, d)) for k, d in (("a_zero", 0), ("b_zero", 0), ("left_shift", 20), ("a_mult", None),
                                                   ("a_shift", None), ("b_mult", None), ("b_shift", None),
                                                   ("out_mult", None), ("out_shift", None), ("y_zero", None))}
        canon.update({"qmin": qmin, "qmax": qmax})
        return [(y, m)], canon

    def w_I8Mul(self, n):
        m = self.mode[n.name]
        a = n.attrs
        qa = self.centered(self._operand(n.inputs[0], m), int(a.get("a_zero", 0)))
        qb = self.centered(self._operand(n.inputs[1], m), int(a.get("b_zero", 0)))
        y = self.emit("Add", [self.requant(self.emit("Mul", [qa, qb]), int(a["mult"]), int(a["shift"])), self.i64(int(a["y_zero"]))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        return [(y, m)], {"a_zero": int(a.get("a_zero", 0)), "b_zero": int(a.get("b_zero", 0)), "mult": int(a["mult"]),
                          "shift": int(a["shift"]), "y_zero": int(a["y_zero"]), "qmin": qmin, "qmax": qmax}

    def w_I8AffineSum(self, n):
        m = self.mode[n.name]
        a = n.attrs
        k = len(n.inputs) - 2
        Kn, Bn = n.inputs[k], n.inputs[k + 1]
        K = np.asarray(self.cdata(Kn), np.int64)
        B = np.asarray(self.cdata(Bn), np.int64)
        r = self.rank(n.outputs[0])
        acc = None
        for i in range(k):
            Ki = self.cchan(Kn, f":row{i}", K[i], m, r, np.int64)
            t = self.emit("Mul", [self.centered(self._operand(n.inputs[i], m), int(a["zeros"][i])), Ki])
            acc = t if acc is None else self.emit("Add", [acc, t])
        acc = self.emit("Add", [acc, self.cchan(Bn, ":b", B, m, r, np.int64)])
        y = self.emit("Add", [self.requant(acc, int(a["mult"]), int(a["shift"])), self.i64(int(a["y_zero"]))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        return [(y, m)], {"zeros": [int(z) for z in a["zeros"]], "mult": int(a["mult"]), "shift": int(a["shift"]),
                          "y_zero": int(a["y_zero"]), "qmin": qmin, "qmax": qmax}

    def w_I8Concat(self, n):
        m = self.mode[n.name]
        a = n.attrs
        yz = int(a["y_zero"])
        parts, zeros, shifts = [], [], []
        any_req = False
        for k, x in enumerate(n.inputs):
            src = self._operand(x, m)
            mk = int(a["mults"][k])
            if mk == 0:
                parts.append(src)
                zeros.append(INFO)
                shifts.append(INFO)
            else:
                any_req = True
                v = self.emit("Add", [self.requant(self.centered(src, int(a["zeros"][k])), mk, int(a["shifts"][k])), self.i64(yz)])
                parts.append(self.to_i8(v, -128, 127))
                zeros.append(int(a["zeros"][k]))
                shifts.append(int(a["shifts"][k]))
        ax = int(a.get("axis", -1))
        y = self.emit("Concat", parts, {"axis": 1 if m == "C" else ax}, out=n.outputs[0])
        canon = {"axis": ax if m == "I" else (-1 if ax == -1 else 3), "mults": [int(v) for v in a["mults"]], "zeros": zeros,
                 "shifts": shifts, "y_zero": yz if any_req else INFO}
        return [(y, m)], canon

    def w_I8Transpose(self, n):
        perm = [int(p) for p in n.attrs["perm"]]
        return [(self.emit("Transpose", [self.rt(n.inputs[0], "I")], {"perm": perm}, out=n.outputs[0]), "I")], {"perm": perm}

    def w_I8Pad(self, n):
        m = self.mode[n.name]
        p = [int(v) for v in n.attrs["pads"]]
        r = self.rank(n.inputs[0])
        dims = list(I2C) if (m == "C" and r == 4) else list(range(r))
        begins = [p[2 * d] for d in dims]
        ends = [p[2 * d + 1] for d in dims]
        P = self.s.init(f"bitino/pads/{'_'.join(map(str, begins + ends))}", np.array(begins + ends, np.int64))
        y = self.emit("Pad", [self.rt(n.inputs[0], m), P, self.i8s(int(n.attrs.get("pad_q", 0)))], {"mode": "constant"},
                      out=n.outputs[0])
        return [(y, m)], {"pads": p, "pad_q": int(n.attrs.get("pad_q", 0))}

    def w_I8MaxPool2D(self, n):
        a = n.attrs
        k = [int(v) for v in a["kernel"]]
        st = [int(v) for v in a.get("strides", k)]
        pads = [int(v) for v in a.get("pads", [0, 0, 0, 0])]
        y = self.emit("MaxPool", [self.rt(n.inputs[0], "C")], {"kernel_shape": k, "strides": st, "pads": _onnx_pads(pads)},
                      out=n.outputs[0])
        return [(y, "C")], {"kernel": k, "strides": st, "pads": pads}

    def w_I8AvgPool2D(self, n):
        a = n.attrs
        x = n.inputs[0]
        k = [int(v) for v in a["kernel"]]
        st = [int(v) for v in a.get("strides", k)]
        pads = [int(v) for v in a.get("pads", [0, 0, 0, 0])]
        z = int(a.get("x_zero", 0))
        inc = int(a.get("count_include_pad", 0))
        _, h, w, c = self.shapes[x]
        _, oh, ow, _ = self.shapes[n.outputs[0]]
        xo = self.rt(x, "C")
        if any(pads):
            t, b_, l, r = pads
            xo = self.emit("Pad", [xo, self.s.init(f"bitino/pads/{t}_{l}_{b_}_{r}", np.array([0, 0, t, l, 0, 0, b_, r], np.int64)),
                                   self.i8s(z)], {"mode": "constant"})
        ones = self.s.init(f"bitino/ones/{c}_{k[0]}_{k[1]}", np.ones((c, 1, k[0], k[1]), np.int8))
        S = self.emit("ConvInteger", [xo, ones, self.i8s(z)], {"kernel_shape": k, "strides": st, "pads": [0, 0, 0, 0],
                                                               "dilations": [1, 1], "group": c})
        cnt = pool_counts(h, w, oh, ow, k, st, pads, inc)
        other = pool_counts(h, w, oh, ow, k, st, pads, 1 - inc)
        if np.all(cnt == cnt.flat[0]):
            C = self.dbl(float(cnt.flat[0]))
            C2 = self.dbl(float(2 * cnt.flat[0]))
        else:
            C = self.cinit(n.name, "/count", cnt.astype(np.float64).reshape(1, 1, oh, ow))
            C2 = self.cinit(n.name, "/count2", (2 * cnt).astype(np.float64).reshape(1, 1, oh, ow))
        A = self.emit("Cast", [S], {"to": DOUBLE})
        num = self.emit("Add", [self.emit("Mul", [self.emit("Abs", [A]), self.dbl(2.0)]), C])
        q = self.emit("Mul", [self.emit("Sign", [A]), self.emit("Floor", [self.emit("Div", [num, C2])])])
        y = self.emit("Add", [self.emit("Cast", [q], {"to": INT64}), self.i64(z)])
        y = self.to_i8(y, -128, 127, out=n.outputs[0])
        canon = {"kernel": k, "strides": st, "pads": pads, "x_zero": z,
                 "count_include_pad": inc if not np.array_equal(cnt, other) else INFO}   # no window reaches the padding
        return [(y, "C")], canon

    def _mean_epilogue(self, n, acc):
        a = n.attrs
        if "mult" not in a:
            raise AssertionError(f"{n.name}: {n.op} without mult (rdiv form) has no step-5 pattern")
        y = self.emit("Add", [self.requant(acc, int(a["mult"]), int(a["shift"])), self.i64(int(a["y_zero"]))])
        return self.to_i8(y, -128, 127, out=n.outputs[0])

    def w_I8GlobalAvgPool(self, n):
        a = n.attrs
        kd = int(a.get("keepdims", 0))
        acc = self.emit("ReduceSum", [self.centered(self.rt(n.inputs[0], "C"), int(a.get("x_zero", 0))), self.axes([2, 3])],
                        {"keepdims": kd})
        y = self._mean_epilogue(n, acc)
        return [(y, "C" if kd else "I")], {"keepdims": kd, "x_zero": int(a.get("x_zero", 0)), "mult": int(a["mult"]),
                                            "shift": int(a["shift"]), "y_zero": int(a["y_zero"])}

    def w_I8ReduceMean(self, n):
        a = n.attrs
        kd = int(a.get("keepdims", 0))
        ax = int(a.get("axis", -1))
        acc = self.emit("ReduceSum", [self.centered(self.rt(n.inputs[0], "I"), int(a.get("x_zero", 0))), self.axes([ax])],
                        {"keepdims": kd})
        y = self._mean_epilogue(n, acc)
        return [(y, "I")], {"axis": ax, "keepdims": kd, "x_zero": int(a.get("x_zero", 0)), "mult": int(a["mult"]),
                            "shift": int(a["shift"]), "y_zero": int(a["y_zero"])}

    # ---------------------------------------------------------------- integer GEMMs
    def _gemm_epilogue(self, n, acc32: str, mode: str, rank: int, has_bits: bool = False):
        """I8Conv2D / I8Dense epilogue (int8_ops._gemm_out) of the INT32 accumulator; returns (outs, canon)."""
        a = n.attrs
        out = a.get("output", "i8")
        bias = self._opt(n, 2)
        canon = {"output": out}
        if out in ("f32", "l8"):
            if has_bits:
                raise AssertionError(f"{n.name}: bit output with output {out}")
            y = self.emit("Mul", [self.emit("Cast", [acc32], {"to": FLOAT}),
                                  self.cchan(n.inputs[3], ":c", self.cdata(n.inputs[3]), mode, rank, np.float32)],
                          out=n.outputs[0] if (out == "f32" and not bias) else None)
            if bias:
                y = self.emit("Add", [y, self.cchan(bias, ":c", self.cdata(bias), mode, rank, np.float32)],
                              out=n.outputs[0] if out == "f32" else None)
            if out == "l8":
                y = self.l8enc(y, out=n.outputs[0])
                canon["y_exp"] = INFO
            return [(y, mode)], canon
        a64 = self.i64of(acc32)
        t = a64
        if bias:
            t = self.emit("Add", [t, self.cchan(bias, ":c", self.cdata(bias), mode, rank, np.int64)])
        t = self.sat(t, I32_MIN, I32_MAX)
        M = self.chan(self.cdata(n.inputs[3]), mode, rank).astype(np.int64)
        sh = self.chan(self.cdata(n.inputs[4]), mode, rank).astype(np.int64)
        y = self.emit("Add", [self.requant(t, M, sh, key=n.inputs[3]), self.i64(int(a.get("y_zero", 0)))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        canon.update({"y_zero": int(a.get("y_zero", 0)), "qmin": qmin, "qmax": qmax})
        outs = [(y, mode)]
        if has_bits:
            bt = self.cchan(n.inputs[5], ":c", self.cdata(n.inputs[5]), mode, rank, np.int64)
            outs.append((self.bits_of(a64, bt, out=n.outputs[1]), mode))
        return outs, canon

    def w_I8Conv2D(self, n):
        a = n.attrs
        x, w = n.inputs[0], n.inputs[1]
        z = int(a.get("x_zero", 0))
        pq = int(a.get("pad_q", z))
        pads = [int(v) for v in a.get("pads", [0, 0, 0, 0])]
        wd = np.asarray(self.cdata(w), np.int8)                 # OHWI
        o, kh, kw, ci = wd.shape
        xo = self.rt(x, "C")
        if any(pads):
            t, b_, l, r = pads
            xo = self.emit("Pad", [xo, self.s.init(f"bitino/pads/{t}_{l}_{b_}_{r}", np.array([0, 0, t, l, 0, 0, b_, r], np.int64)),
                                   self.i8s(pq)], {"mode": "constant"})
        wt = self.cinit(w, ":oihw", np.ascontiguousarray(np.transpose(wd, (0, 3, 1, 2))))
        dil = [int(v) for v in a.get("dilations", [1, 1])]
        st = [int(v) for v in a.get("strides", [1, 1])]
        g = int(a.get("groups", 1))
        acc = self.emit("ConvInteger", [xo, wt, self.i8s(z)], {"kernel_shape": [kh, kw], "strides": st, "pads": [0, 0, 0, 0],
                                                               "dilations": dil, "group": g})
        has_bits = len(n.inputs) > 5 and bool(n.inputs[5])
        outs, canon = self._gemm_epilogue(n, acc, "C", 4, has_bits)
        canon.update({"x_zero": z, "pad_q": pq if any(pads) else INFO, "pads": pads, "strides": st, "dilations": dil,
                      "groups": g})
        return outs, canon

    def w_I8Dense(self, n):
        a = n.attrs
        x, w = n.inputs[0], n.inputs[1]
        z = int(a.get("x_zero", 0))
        wd = np.asarray(self.cdata(w), np.int8)                 # [O, K]
        wt = self.cinit(w, ":ko", np.ascontiguousarray(wd.T))
        acc = self.emit("MatMulInteger", [self.rt(x, "I"), wt, self.i8s(z)])
        outs, canon = self._gemm_epilogue(n, acc, "I", self.rank(n.outputs[0]))
        canon["x_zero"] = z
        return outs, canon

    def w_I8MatMul(self, n):
        a = n.attrs
        A = self._operand(n.inputs[0], "I")
        B = self._operand(n.inputs[1], "I")
        tb = int(a.get("transpose_b", 0))
        if tb:
            B = self._swap_last(B, self.rank(n.inputs[1]))
        za, zb = int(a.get("a_zero", 0)), int(a.get("b_zero", 0))
        acc = self.emit("MatMulInteger", [A, B, self.i8s(za), self.i8s(zb)])
        canon = {"a_zero": za, "b_zero": zb, "transpose_b": tb}
        if a.get("output", "i8") == "f32":
            y = self.emit("Mul", [self.emit("Cast", [acc], {"to": FLOAT}), self.fval(float(a["scale"]))], out=n.outputs[0])
            canon.update({"output": "f32", "scale": float(a["scale"])})
            return [(y, "I")], canon
        t = self.sat(self.i64of(acc), I32_MIN, I32_MAX)
        y = self.emit("Add", [self.requant(t, int(a["mult"]), int(a["shift"])), self.i64(int(a.get("y_zero", 0)))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        canon.update({"output": "i8", "mult": int(a["mult"]), "shift": int(a["shift"]), "y_zero": int(a.get("y_zero", 0)),
                      "qmin": qmin, "qmax": qmax})
        return [(y, "I")], canon

    # ---------------------------------------------------------------- normalization
    def w_I8LayerNorm(self, n):
        a = n.attrs
        x, G, Bt = n.inputs[0], n.inputs[1], n.inputs[2]
        d = self.shapes[x][-1]
        z = int(a["x_zero"])
        xs, eps = float(a["x_scale"]), float(a.get("eps", 1e-5))
        t = self.centered(self.rt(x, "I"), z)
        S1 = self.emit("ReduceSum", [t, self.axes([-1])], {"keepdims": 1})
        S2 = self.emit("ReduceSum", [self.emit("Mul", [t, t]), self.axes([-1])], {"keepdims": 1})
        D64 = self.i64(d)
        VN = self.emit("Sub", [self.emit("Mul", [S2, D64]), self.emit("Mul", [S1, S1])])
        XS, EPS, DD = self.dbl(xs), self.dbl(eps), self.dbl(float(d))
        c2 = self.emit("Mul", [self.emit("Mul", [EPS, DD]), DD])
        den = self.emit("Sqrt", [self.emit("Add", [self.emit("Mul", [self.emit("Mul", [XS, XS]), self.emit("Cast", [VN], {"to": DOUBLE})]), c2])])
        r = self.emit("Cast", [self.emit("Div", [XS, den])], {"to": FLOAT})
        u = self.emit("Cast", [self.emit("Sub", [self.emit("Mul", [t, D64]), S1])], {"to": FLOAT})
        v = self.emit("Add", [self.emit("Mul", [self.emit("Mul", [u, r]), self.c_f32(G)]), self.c_f32(Bt)])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.rne8(v, qmin, qmax, out=n.outputs[0])
        return [(y, "I")], {"x_zero": z, "x_scale": xs, "eps": eps, "qmin": qmin, "qmax": qmax}

    def w_I8Softmax(self, n):
        a = n.attrs
        q = self.i64of(self.rt(n.inputs[0], "I"))
        mx = self.emit("ReduceMax", [q], {"axes": [-1], "keepdims": 1})
        E = self.cinit(n.inputs[1], "", np.asarray(self.cdata(n.inputs[1]), np.int32))
        e = self.emit("Gather", [E, self.emit("Sub", [mx, q])], {"axis": 0})
        S = self.emit("ReduceSum", [self.emit("Cast", [e], {"to": INT64}), self.axes([-1])], {"keepdims": 1})
        f32 = a.get("output", "i8") == "f32"
        num = self.dbl(1.0 if f32 else float(a["inv_scale"]))
        r = self.emit("Cast", [self.emit("Div", [num, self.emit("Cast", [S], {"to": DOUBLE})])], {"to": FLOAT})
        v = self.emit("Mul", [self.emit("Cast", [e], {"to": FLOAT}), r], out=n.outputs[0] if f32 else None)
        if f32:
            return [(v, "I")], {"output": "f32"}
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        yz = int(a["y_zero"])
        if abs(yz) >= 1 << 24:
            raise AssertionError(f"{n.name}: y_zero {yz}")
        c = self.emit("Clip", [self.emit("Add", [self.emit("Round", [v]), self.f32s(float(yz))]), self.f32s(float(qmin)), self.f32s(float(qmax))])
        y = self.emit("Cast", [c], {"to": INT8}, out=n.outputs[0])
        return [(y, "I")], {"output": "i8", "inv_scale": float(a["inv_scale"]), "y_zero": yz, "qmin": qmin, "qmax": qmax}

    def w_I8Gather(self, n):
        a = n.attrs
        tab, ids = n.inputs[0], n.inputs[1]
        fill = int(a.get("fill_q", 0))
        if ids in self.consts:
            idv = np.asarray(self.cdata(ids), np.int32)
            io = self.cinit(ids, "", idv)
        else:
            io = self.rt(ids, "I")
        if tab in self.consts:
            T = self.consts[tab]
            V = T.shape[0]
        else:
            V = self.shapes[tab][0]
        i64 = self.i64of(io)
        ok = self.emit("And", [self.emit("GreaterOrEqual", [i64, self.i64(0)]), self.emit("Less", [i64, self.i64(V)])])
        safe = self.emit("Where", [ok, i64, self.i64(0)])
        if tab in self.consts and self.consts[tab].dtype == "bit":
            sb = self.cinit(tab, ":pm1", self.cdata(tab).astype(np.int8))
            qpm = self.cinit(n.inputs[2], "", np.asarray(self.cdata(n.inputs[2]), np.int8))
            gq = self.emit("Gather", [qpm, safe], {"axis": 0})
            qp = self.emit("Slice", [gq, self.axes([0]), self.axes([1]), self.axes([-1])])
            qn = self.emit("Slice", [gq, self.axes([1]), self.axes([2]), self.axes([-1])])
            rows = self.emit("Where", [self.emit("Greater", [self.emit("Gather", [sb, safe], {"axis": 0}), self.i8s(0)]), qp, qn])
            zm = self._opt(n, 3)
            if zm:
                zb = self.cinit(zm, ":pm1", self.cdata(zm).astype(np.int8))
                rows = self.emit("Where", [self.emit("Less", [self.emit("Gather", [zb, safe], {"axis": 0}), self.i8s(0)]),
                                           self.i8s(fill), rows])
        elif tab in self.consts:
            if self.consts[tab].dtype != "i8" or len(n.inputs) > 2:
                raise AssertionError(f"{n.name}: I8Gather table {self.consts[tab].dtype}")
            rows = self.emit("Gather", [self.cinit(tab, "", np.asarray(self.cdata(tab), np.int8)), safe], {"axis": 0})
        else:
            rows = self.emit("Gather", [self.rt(tab, "I"), safe], {"axis": 0})
        y = self.emit("Where", [self.emit("Unsqueeze", [ok, self.axes([-1])]), rows, self.i8s(fill)], out=n.outputs[0])
        return [(y, "I")], {"fill_q": fill}

    # ---------------------------------------------------------------- binary layers
    def w_I8BDense(self, n):
        a = n.attrs
        x, w = n.inputs[0], n.inputs[1]
        pm, pa, thr, wz = self._opt(n, 2), self._opt(n, 3), self._opt(n, 4), self._opt(n, 5)
        xb, canon = self.bin_input(n, x, "I", thr, None)
        bits = self.cdata(w).astype(np.int8)                      # [O, K]
        if wz:
            zz = self.cdata(wz)
            if np.any(bits[zz < 0] != 1):
                raise AssertionError(f"{n.name}: w_sign bits at zero weights are not +1")
            bits = np.where(zz < 0, 0, bits).astype(np.int8)
        wt = (self.c_ternary if wz else self.c_pm1)(w, np.ascontiguousarray(bits.T), ":ko")
        out = a.get("output", "i8")
        y = self.emit("MatMul", [xb, wt])
        if pm:
            y = self.emit("Mul", [y, self.c_f32(pm)])
        if pa:
            y = self.emit("Add", [y, self.c_f32(pa)])
        canon["output"] = out
        if out == "f32":
            y = self.emit("Identity", [y], out=n.outputs[0])
        else:
            qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
            y = self.rne8(y, qmin, qmax, out=n.outputs[0])
            canon.update({"qmin": qmin, "qmax": qmax})
        return [(y, "I")], canon

    def w_I8BConv2D(self, n):
        a = n.attrs
        x, w = n.inputs[0], n.inputs[1]
        pm, pa, thr = self._opt(n, 2), self._opt(n, 3), self._opt(n, 4)
        c_in = self.shapes[x][3]
        xb, canon = self.bin_input(n, x, "C", thr, (c_in, 1, 1))
        xb, pc = self.pad_binary(xb, a)
        canon.update(pc)
        y, cc = self.bconv(n, xb, w)
        canon.update(cc)
        o = self.consts[w].shape[0]
        if pm:
            y = self.emit("Mul", [y, self.c_f32(pm, (o, 1, 1), ":c11")])
        if pa:
            y = self.emit("Add", [y, self.c_f32(pa, (o, 1, 1), ":c11")])
        out = a.get("output", "i8")
        canon["output"] = out
        if out == "f32":
            y = self.emit("Identity", [y], out=n.outputs[0])
        else:
            qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
            y = self.rne8(y, qmin, qmax, out=n.outputs[0])
            canon.update({"qmin": qmin, "qmax": qmax})
        return [(y, "C")], canon

    def _residual(self, n, r: str, mode: str) -> tuple:
        a = n.attrs
        ro = self.rt(r, mode)
        if a.get("r_code", "l8") == "l8":
            return self.l8dec(ro, int(a["r_exp"])), {"r_code": "l8", "r_exp": int(a["r_exp"])}
        return self.deq(ro, float(a["r_scale"]), int(a.get("r_zero", 0))), \
            {"r_code": "linear", "r_scale": float(a["r_scale"]), "r_zero": int(a.get("r_zero", 0))}

    def w_I8BConv2DRes(self, n):
        a = n.attrs
        ins = list(n.inputs) + [""] * 15
        x, w, pm, pa, thr, res, slope, ob, bt = ins[:9]
        if ins[9]:
            raise AssertionError(f"{n.name}: I8BConv2DRes out_lut (input 9) is Session-only")
        ps, pmul, padd, gate, gadd = ins[10:15]
        o = self.consts[w].shape[0]
        c_in = self.shapes[x][3]
        xb, canon = self.bin_input(n, x, "C", thr or None, (c_in, 1, 1))
        xb, pc = self.pad_binary(xb, a)
        canon.update(pc)
        acc, cc = self.bconv(n, xb, w)
        canon.update(cc)
        c11 = lambda c: self.c_f32(c, (o, 1, 1), ":c11")      # noqa: E731
        v = self.emit("Add", [self.emit("Mul", [acc, c11(pm)]), c11(pa)])
        if ps:
            v = self.emit("PRelu", [v, c11(ps)])
        if pmul:
            v = self.emit("Mul", [v, c11(pmul)])
        if padd:
            v = self.emit("Add", [v, c11(padd)])
        if gate:
            v = self.emit("Mul", [v, self.deq(self.rt(gate, "C"), float(a["g_scale"]), int(a.get("g_zero", 0)))])
            canon.update({"g_scale": float(a["g_scale"]), "g_zero": int(a.get("g_zero", 0))})
        if gadd:
            v = self.emit("Add", [v, c11(gadd)])
        R, rc = self._residual(n, res, "C")
        canon.update(rc)
        u = self.emit("Add", [v, R])
        wv = self.emit("PRelu", [u, c11(slope)])
        if ob:
            wv = self.emit("Add", [wv, c11(ob)])
        out = a.get("output", "l8")
        canon["output"] = out
        if out == "l8":
            y = self.l8enc(wv, out=n.outputs[0])
            canon["y_exp"] = INFO
        else:
            qmin, qmax, yz = int(a.get("qmin", -128)), int(a.get("qmax", 127)), int(a.get("y_zero", 0))
            c = self.emit("Clip", [self.emit("Add", [self.emit("Round", [wv]), self.f32s(float(yz))]),
                                   self.f32s(float(qmin)), self.f32s(float(qmax))])
            y = self.emit("Cast", [c], {"to": INT8}, out=n.outputs[0])
            canon.update({"qmin": qmin, "qmax": qmax, "y_zero": yz})
            if "y_exp" in a:
                canon["y_exp"] = INFO
        outs = [(y, "C")]
        if bt:
            src = a.get("bit_src", "sum")
            mode = a.get("bit_binarize", "ge")
            s = wv if src == "out" else u
            outs.append((self.bits_of(s, c11(bt), strict=(mode == "gt"), out=n.outputs[1]), "C"))
            canon.update({"bit_src": src, "bit_binarize": mode})
        else:
            for k in ("bit_src", "bit_binarize"):
                if k in a:
                    canon[k] = INFO
        return outs, canon

    def w_I8BConv2DResI(self, n):
        a = n.attrs
        ins = list(n.inputs) + [""] * 16
        x, w, _u2, _u3, thr, res, m1, b, m2, bt, sl, ob, pb, plo, phi, pslope = ins[:16]
        if _u2 or _u3:
            raise AssertionError(f"{n.name}: I8BConv2DResI inputs 2 / 3 must be empty")
        if pslope:
            raise AssertionError(f"{n.name}: I8BConv2DResI PReLU pre stage (input 15) has no step-5 pattern yet")
        wsh = self.consts[w].shape
        o = wsh[0]
        K = int(np.prod(wsh[1:]))
        cd = lambda c: np.asarray(self.cdata(c), np.int64) if c else None      # noqa: E731
        if int8_ops.bconv_res_i_bound(K, cd(m1), cd(m2) if res else np.zeros(1, np.int64), cd(b), cd(sl),
                                      int(a.get("slope_shift", 0)), cd(ob), cd(pb), None, 0) >= 2 ** 31:
            raise AssertionError(f"{n.name}: constants exceed the int32 bound")
        c_in = self.shapes[x][3]
        xb, canon = self.bin_input(n, x, "C", thr or None, (c_in, 1, 1))
        xb, pc = self.pad_binary(xb, a)
        canon.update(pc)
        dot, cc = self.bconv(n, xb, w)
        canon.update(cc)
        ci = lambda c, key=":c": self.cchan(c, key, self.cdata(c), "C", 4, np.int64)   # noqa: E731
        pre = self.emit("Mul", [self.emit("Cast", [dot], {"to": INT64}), ci(m1)])
        if pb:
            lo = self.chan(self.cdata(plo), "C", 4).astype(np.int64)
            hi = self.chan(self.cdata(phi), "C", 4).astype(np.int64)
            pre = self.sat(self.emit("Add", [pre, ci(pb)]), lo, hi, key=pb)
        acc = self.emit("Add", [pre, ci(b)])
        rt = int(a.get("r_tile", 1))
        if res:
            r = self.i64of(self.rt(res, "C"))
            if rt > 1:
                r = self.emit("Concat", [r] * rt, {"axis": 1})
            acc = self.emit("Add", [acc, self.emit("Mul", [r, ci(m2)])])
        canon["r_tile"] = rt if res else (INFO if "r_tile" in a else 1)
        wv = acc
        ssh = int(a.get("slope_shift", 0))
        if sl:
            neg = self.emit("Mul", [self.mulhi(acc, self.chan(self.cdata(sl), "C", 4).astype(np.int64), key=sl), self.i64(1 << ssh)])
            wv = self.emit("Where", [self.emit("Less", [acc, self.i64(0)]), neg, acc])
            canon["slope_shift"] = ssh
        elif "slope_shift" in a:
            canon["slope_shift"] = INFO
        else:
            canon["slope_shift"] = 0
        if ob:
            wv = self.emit("Add", [wv, ci(ob)])
        shift = int(a.get("shift", 0))
        y = self.emit("Add", [self.rshr(wv, shift), self.i64(int(a.get("y_zero", 0)))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        canon.update({"shift": shift, "y_zero": int(a.get("y_zero", 0)), "qmin": qmin, "qmax": qmax})
        outs = [(y, "C")]
        if bt:
            src = a.get("bit_src", "sum")
            s = wv if src == "out" else acc
            outs.append((self.bits_of(s, ci(bt), out=n.outputs[1]), "C"))
            canon["bit_src"] = src if wv != acc else INFO      # without slope / out_bias, w = acc: either source
        elif "bit_src" in a:
            canon["bit_src"] = INFO
        return outs, canon

    def w_I8GatedAdd(self, n):
        a = n.attrs
        ins = list(n.inputs) + [""] * 8
        x, p, g, kx, kp, b, bt, glut = ins[:8]
        if glut:
            raise AssertionError(f"{n.name}: I8GatedAdd g_lut (input 7) has no step-5 pattern yet")
        m = self.mode[n.name]
        if m != "C":
            raise AssertionError(f"{n.name}: I8GatedAdd outside an NCHW group")
        ci = lambda c, key=":c": self.cchan(c, key, self.cdata(c), "C", 4, np.int64)   # noqa: E731
        zx, zp, zg = int(a.get("x_zero", 0)), int(a.get("p_zero", 0)), int(a.get("g_zero", 0))
        tx = self.emit("Mul", [self.centered(self.rt(x, "C"), zx), ci(kx)])
        tp = self.emit("Mul", [self.emit("Mul", [self.centered(self.rt(p, "C"), zp), self.centered(self.rt(g, "C"), zg)]), ci(kp)])
        acc = self.emit("Add", [self.emit("Add", [tx, tp]), ci(b)])
        y = self.emit("Add", [self.rshr(acc, int(a.get("shift", 0))), self.i64(int(a.get("y_zero", 0)))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        outs = [(y, "C")]
        if bt:
            T = np.asarray(self.cdata(bt), np.int64)
            c = self.shapes[x][3]
            T = T.reshape(-1, c)
            for j in range(T.shape[0]):
                Tj = self.cinit(bt, f":row{j}", T[j].reshape(-1, 1, 1))
                outs.append((self.bits_of(acc, Tj, out=n.outputs[1 + j]), "C"))
        return outs, {"x_zero": zx, "p_zero": zp, "g_zero": zg, "shift": int(a.get("shift", 0)),
                      "y_zero": int(a.get("y_zero", 0)), "qmin": qmin, "qmax": qmax}

    def w_I8GapAffine(self, n):
        a = n.attrs
        k = len(n.inputs) - 2
        Kn, Bn = n.inputs[k], n.inputs[k + 1]
        K = np.asarray(self.cdata(Kn), np.int64)
        B = np.asarray(self.cdata(Bn), np.int64)
        zs = [int(v) for v in a.get("zeros", [0] * k)]
        acc = self.cinit(Bn, ":c", B.reshape(-1, 1, 1))
        for i in range(k):
            S = self.emit("ReduceSum", [self.centered(self.rt(n.inputs[i], "C"), zs[i]), self.axes([2, 3])], {"keepdims": 1})
            acc = self.emit("Add", [acc, self.emit("Mul", [S, self.cinit(Kn, f":row{i}", K[i].reshape(-1, 1, 1))])])
        y = self.emit("Add", [self.rshr(acc, int(a.get("shift", 0))), self.i64(int(a.get("y_zero", 0)))])
        qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
        y = self.to_i8(y, qmin, qmax, out=n.outputs[0])
        return [(y, "C")], {"zeros": zs, "shift": int(a.get("shift", 0)), "y_zero": int(a.get("y_zero", 0)),
                            "qmin": qmin, "qmax": qmax}

    def w_I8BMatMul(self, n):
        a = n.attrs
        A_, B_ = n.inputs[0], n.inputs[1]
        mul, add = self._opt(n, 2), self._opt(n, 3)
        am, bm = a.get("a_binarize", "sign_zero"), a.get("b_binarize", "sign_zero")
        A = self.bin_scalar(A_, am, float(a["a_scale"]), int(a["a_zero"]), a.get("a_threshold"), A_ in self.consts)
        B = self.bin_scalar(B_, bm, float(a["b_scale"]), int(a["b_zero"]), a.get("b_threshold"), B_ in self.consts)
        tb = int(a.get("transpose_b", 0))
        if tb:
            B = self._swap_last(B, self.rank(B_))
        c_ab, c_rs, c_cs, c0 = (int(a.get(k, d)) for k, d in (("c_ab", 1), ("c_rs", 0), ("c_cs", 0), ("c0", 0)))
        for v in (c_ab, c_rs, c_cs, c0):
            if abs(v) >= 1 << 24:
                raise AssertionError(f"{n.name}: I8BMatMul coefficient {v}")
        t = self.emit("Mul", [self.emit("MatMul", [A, B]), self.f32s(float(c_ab))])
        t = self.emit("Add", [t, self.emit("Mul", [self.emit("ReduceSum", [A, self.axes([-1])], {"keepdims": 1}), self.f32s(float(c_rs))])])
        t = self.emit("Add", [t, self.emit("Mul", [self.emit("ReduceSum", [B, self.axes([-2])], {"keepdims": 1}), self.f32s(float(c_cs))])])
        acc = self.emit("Add", [t, self.f32s(float(c0))])
        v = self.emit("Mul", [acc, self.c_f32(mul)])
        if add:
            v = self.emit("Add", [v, self.c_f32(add)])
        canon = {"a_binarize": am, "b_binarize": bm, "a_scale": float(a["a_scale"]), "a_zero": int(a["a_zero"]),
                 "b_scale": float(a["b_scale"]), "b_zero": int(a["b_zero"]), "transpose_b": tb, "c_ab": c_ab, "c_rs": c_rs,
                 "c_cs": c_cs, "c0": c0}
        if "a_threshold" in a:
            canon["a_threshold"] = float(a["a_threshold"])
        if "b_threshold" in a:
            canon["b_threshold"] = float(a["b_threshold"])
        out = a.get("output", "i8")
        canon["output"] = out
        if out == "f32":
            y = self.emit("Identity", [v], out=n.outputs[0])
        else:
            qmin, qmax = int(a.get("qmin", -128)), int(a.get("qmax", 127))
            y = self.rne8(v, qmin, qmax, out=n.outputs[0])
            canon.update({"qmin": qmin, "qmax": qmax})
        return [(y, "I")], canon

    # ---------------------------------------------------------------- k-bit (DoReFa) layers between I8Dequantize / I8Quantize
    def _levels(self, n, x: str) -> str:
        nl = 2 ** int(n.attrs["act_bits"]) - 1
        return self.emit("Round", [self.emit("Mul", [self.emit("Clip", [x, self.ZERO, self.ONE]), self.f32s(float(nl))])])

    def _qb_epilogue(self, n, y: str, mode: str, rank: int) -> str:
        pm, pa = self._opt(n, 2), self._opt(n, 3)
        if n.attrs.get("activation", "none") != "none":
            raise AssertionError(f"{n.name}: activation {n.attrs['activation']}")
        o = self.consts[n.inputs[1]].shape[0]
        shp = (o, 1, 1) if mode == "C" else None
        if pm:
            y = self.emit("Mul", [y, self.c_f32(pm, shp, ":c11" if shp else "")])
        if pa:
            y = self.emit("Add", [y, self.c_f32(pa, shp, ":c11" if shp else "")])
        return self.emit("Identity", [y], out=n.outputs[0])

    def w_QBConv2D(self, n):
        a = n.attrs
        if a.get("act_quant", "dorefa") != "dorefa":
            raise AssertionError(f"{n.name}: act_quant {a.get('act_quant')}")
        lv = self._levels(n, self.rt(n.inputs[0], "C"))
        bits = self.cdata(n.inputs[1]).astype(np.int8)
        o, kh, kw, ci = bits.shape
        wt = self.c_pm1(n.inputs[1], np.ascontiguousarray(np.transpose(bits, (0, 3, 1, 2))), ":oihw")
        pads = [int(v) for v in a.get("pads", [0, 0, 0, 0])]
        st = [int(v) for v in a.get("strides", [1, 1])]
        dil = [int(v) for v in a.get("dilations", [1, 1])]
        g = int(a.get("groups", 1))
        y = self.emit("Conv", [lv, wt], {"kernel_shape": [kh, kw], "strides": st, "pads": _onnx_pads(pads), "dilations": dil,
                                         "group": g})
        y = self._qb_epilogue(n, y, "C", 4)
        return [(y, "C")], {"act_bits": int(a["act_bits"]), "act_quant": "dorefa", "activation": "none", "pads": pads,
                            "strides": st, "dilations": dil, "groups": g}

    def w_QBDense(self, n):
        a = n.attrs
        if a.get("act_quant", "dorefa") != "dorefa":
            raise AssertionError(f"{n.name}: act_quant {a.get('act_quant')}")
        lv = self._levels(n, self.rt(n.inputs[0], "I"))
        bits = self.cdata(n.inputs[1]).astype(np.int8)               # [O, K]
        wt = self.c_pm1(n.inputs[1], np.ascontiguousarray(bits.T), ":ko")
        y = self._qb_epilogue(n, self.emit("MatMul", [lv, wt]), "I", 2)
        return [(y, "I")], {"act_bits": int(a["act_bits"]), "act_quant": "dorefa", "activation": "none"}


def run_ir(g: Graph, feeds: dict, record=None, keep: set | None = None) -> dict:
    """The numpy executor's int8_full semantics node by node (int8_ops for I8*, executor for the rest), freeing every
    tensor after its last use; record(name, value) sees every output. Returns the graph outputs (+ keep)."""
    from . import executor
    env: dict = {}
    for name, c in g.constants.items():
        env[name] = unpack_bits(c.data, c.shape[-1]).reshape(c.shape) if c.dtype == "bit" else \
            (np.asarray(c.data, np.float32) if c.dtype == "f32" else np.asarray(c.data))
    for i in g.inputs:
        env[i["name"]] = np.asarray(feeds[i["name"]], {"f32": np.float32, "i32": np.int32}[i["dtype"]]).reshape(i["shape"])
    last: dict = {}
    for j, n in enumerate(g.nodes):
        for i in n.inputs:
            if i:
                last[i] = j
    retain = set(keep or ()) | set(g.outputs) | set(g.constants)
    cache: dict = {}
    for j, n in enumerate(g.nodes):
        ins = [env[i] if i else None for i in n.inputs]
        if n.op.startswith("I8"):
            r = int8_ops.run_op(n.op, ins, n.attrs, cache, n.name)
            vals = list(r) if isinstance(r, tuple) else [r]
        elif n.op == "Reshape":
            vals = [np.asarray(ins[0]).reshape(n.attrs["shape"])]
        else:
            sub = Graph()
            for i, v in zip(n.inputs, ins):
                if not i:
                    continue
                if i in g.constants:
                    sub.constants[i] = g.constants[i]
                else:
                    sub.add_input(i, np.shape(v), "f32" if np.asarray(v).dtype == np.float32 else "i32")
            sub.nodes = [Node(n.op, list(n.inputs), list(n.outputs), dict(n.attrs), n.name)]
            sub.set_outputs(*n.outputs)
            out = executor.run(sub, {i: v for i, v in zip(n.inputs, ins) if i and i not in g.constants})
            vals = [out[o] for o in n.outputs]
        for o, v in zip(n.outputs, vals):
            env[o] = v
            if record is not None:
                record(o, v)
        for i in set(n.inputs):
            if i and last.get(i) == j and i not in retain:
                env.pop(i, None)
    return {k: v for k, v in env.items() if k in g.outputs or (keep and k in keep)}


def ir_shapes_kinds(g: Graph) -> tuple[dict, dict]:
    """Static shape and kind ("i8": int8 codes, "f": FLOAT (f32 values or a +-1 bit tensor), "i32") of every IR tensor:
    one run on zero inputs."""
    feeds = {}
    for i in g.inputs:
        dt = {"f32": np.float32, "i32": np.int32}[i["dtype"]]
        feeds[i["name"]] = np.zeros(i["shape"], dt)
    shapes = {i["name"]: tuple(i["shape"]) for i in g.inputs}
    kinds = {i["name"]: ("f" if i["dtype"] == "f32" else "i32") for i in g.inputs}

    def rec(name, y):
        dt = np.asarray(y).dtype
        shapes[name] = tuple(np.shape(y))
        kinds[name] = "i8" if dt == np.int8 else ("i32" if dt.kind in "iu" else "f")
    run_ir(g, feeds, rec)
    return shapes, kinds


def write_spec_int8(g: Graph, shapes: dict | None = None, kinds: dict | None = None):
    """IR graph of an int8_full file -> (spec, writer report); the caller adds the model metadata (§ 6)."""
    from .qonnx_groups import CONST_ORDER_KEY, IMPORT_KEY, MAPPING, first_use_order
    if shapes is None or kinds is None:
        shapes, kinds = ir_shapes_kinds(g)
    w = Int8Writer(g, shapes, kinds)
    spec = w.write()
    spec.meta[IMPORT_KEY] = json.dumps({"mapping": MAPPING}, sort_keys=True)
    if w.unused:
        spec.meta[UNUSED_KEY] = json.dumps(w.unused)
    w.report["unused_constants"] = len(w.unused)
    if first_use_order(g) != list(g.constants):
        spec.meta[CONST_ORDER_KEY] = json.dumps(list(g.constants))
    w.report["const_order"] = CONST_ORDER_KEY in spec.meta
    return spec, w.report


# ================================================================================================ importer
def _pyattrs(nd) -> dict:
    from onnx import helper
    out = {}
    for a in nd.attribute:
        v = helper.get_attribute_value(a)
        out[a.name] = v.decode() if isinstance(v, bytes) else (list(v) if isinstance(v, (list, tuple)) else v)
    return out


def _ctoken(v, carrier: bool) -> tuple:
    import hashlib
    a = np.ascontiguousarray(np.asarray(v))
    return ("c", a.dtype.str, tuple(a.shape), hashlib.sha256(a.tobytes()).hexdigest(), bool(carrier))


def _spec_consts(spec: Spec) -> tuple[dict, set]:
    """Values of a scratch spec's constants: initializers and the unnamed constant-preparation nodes (Cast of an
    initializer, the ternary IntQuant), as the importer reads them (GroupImporter.__init__)."""
    from .qonnx_groups import _NP_OF
    val, car = dict(spec.inits), set()
    for nd in spec.nodes:
        if nd["meta"].get(NAME_KEY) is not None:
            continue
        src = val[nd["inputs"][0]]
        if nd["op"] == "Cast":
            to = int(nd["attrs"]["to"])
            if to == FLOAT and np.asarray(src).dtype == np.float64:
                val[nd["outputs"][0]] = np.asarray(src, np.float64)
                car.add(nd["outputs"][0])
            else:
                val[nd["outputs"][0]] = np.asarray(src).astype(_NP_OF[to])
        elif nd["op"] == "IntQuant":
            val[nd["outputs"][0]] = np.asarray(src, np.float32)
        else:
            raise AssertionError(f"scratch constant node {nd['op']}")
    return val, car


class _View8(_View):
    """_View that records the runtime inputs a reader takes (ONNX tensor -> IR tensor, tag) for the re-emission."""

    def __init__(self, imp, gp, nodes):
        super().__init__(imp, gp, nodes)
        self.uses: dict = {}
        self.mode = None

    def runtime(self, t: str):
        r = super().runtime(t)
        self.uses[t] = r
        return r

    def cons(self, t: str, op: str | None = None):
        c = [q for q in self.cons_in.get(t, []) if op is None or q.op_type == op]
        return c[0] if len(c) == 1 else None

    def cval(self, t: str):
        c = self.const(t)
        if c is None:
            raise self.err(f"tensor {t!r}: constant expected")
        return np.asarray(c)

    def scalar(self, t: str):
        c = self.cval(t)
        if c.size != 1:
            raise self.err(f"tensor {t!r}: scalar constant expected")
        return c.reshape(-1)[0].item()

    def ival(self, t: str) -> int:
        v = self.scalar(t)
        if float(v) != int(v):
            raise self.err(f"tensor {t!r}: integral constant expected")
        return int(v)

    def fval(self, t: str) -> float:
        """An attribute value from a FLOAT scalar (its fp32 value) or a float64 carrier (its float64 value)."""
        return float(self.scalar(t))


def _cast_to(V, nd):
    return V.attrs(nd).get("to")


class Int8Importer(GroupImporter):
    """Group-mapped int8_full file -> IR graph (module docstring): extraction, then the re-emission check per group."""

    def infer_shapes(self) -> None:
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
        self.otype = {}
        for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
            t = vi.type.tensor_type
            self.oshape[vi.name] = tuple(int(d.dim_value) for d in t.shape.dim)
            self.otype[vi.name] = int(t.elem_type)
        for k, v in self.cval.items():
            self.oshape[k] = tuple(np.shape(v))
        del m

    def add_const(self, gp, slot: int, data, dtype: str, src: str, layout: str | None = None) -> str:
        from .ir import pack_bits
        name = gp.consts.get(slot)
        if name is None:
            raise GroupImportError(gp.name, f"no bitino.consts name for IR input {slot}", gp.op)
        lay = gp.layouts.get(slot, DEFAULT_LAYOUTS_I8.get((gp.op, slot), "")) if layout is None else layout
        np_t = {"f32": np.float32, "i32": np.int32, "i8": np.int8}
        if dtype != "bit":
            a = np.asarray(data)
            if not np.array_equal(a.astype(np_t[dtype]).astype(np.float64), a.astype(np.float64)):
                raise GroupImportError(gp.name, f"constant {name!r}: values do not fit {dtype}", gp.op)
        if name in self.g.constants:
            c = self.g.constants[name]
            new = pack_bits(data) if dtype == "bit" else np.asarray(data, np_t[dtype])
            if c.dtype != dtype or c.layout != lay or tuple(c.shape) != tuple(np.shape(data)) or \
                    not np.array_equal(np.asarray(c.data), new):
                raise GroupImportError(gp.name, f"constant {name!r} used twice with different values / layouts", gp.op)
            return name
        self.g.add_constant(name, np.asarray(data) if dtype == "bit" else np.asarray(data, np_t[dtype]), dtype=dtype, layout=lay)
        return name

    # ---------------------------------------------------------------- one group
    def group(self, gp) -> None:
        nodes = list(gp.nodes)
        rest = []
        for nd in nodes:
            if nd.op_type == "Transpose" and nd.input[0] in self.graph_inputs and self._is_conv(nd, self.rt[nd.input[0]][1]):
                ir, tag = self.rt[nd.input[0]]
                self.rt[nd.output[0]] = (ir, "I" if tag == "C" else "C")
                self.rule("layout Transpose (graph input)")
            else:
                rest.append(nd)
        post = []
        while len(rest) >= 2:
            nd = rest[-1]
            perm = tuple(next((list(a.ints) for a in nd.attribute if a.name == "perm"), []))
            inner = {o for q in rest[:-1] for o in q.output}
            if nd.op_type == "Transpose" and nd.input[0] in inner and perm in (C2I, I2C) and \
                    len(self.oshape.get(nd.input[0], ())) == 4 and not any(nd.output[0] in q.input for q in rest[:-1]):
                post.insert(0, rest.pop())
            else:
                break
        reader = getattr(self, f"r_{gp.op}", None)
        if reader is None:
            raise GroupImportError(gp.name, f"no step-5 reader for IR operator {gp.op}", gp.op)
        V = _View8(self, gp, rest)
        self.cur = V
        try:
            ins, canon, outs = reader(V)
        except GroupImportError:
            raise
        except Exception as e:          # an unexpected structure: not the canonical pattern
            raise GroupImportError(gp.name, f"not the canonical {gp.op} pattern ({type(e).__name__}: {e})", gp.op)
        V.check_all_used()
        attrs = apply_attrs_meta_i8(gp.op, canon, gp.attrs, gp.name)
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
        ir = json.loads(gp.meta.get(IR_KEY, "{}"))
        names = ir.get(OUT_KEY) or ([f"{gp.name}:out"] if len(outs) == 1 else [f"{gp.name}:out{i}" for i in range(len(outs))])
        if len(names) != len(outs):
            raise GroupImportError(gp.name, f"bitino.ir outputs names {len(names)} IR outputs, the pattern has {len(outs)}",
                                   gp.op)
        produced = self.rest_outputs(rest)
        for t, _tag in outs:
            if t not in produced:
                raise GroupImportError(gp.name, f"pattern output {t} is not produced by the group", gp.op)
        node = Node(gp.op, list(ins), list(names), dict(attrs), gp.name)
        self.reemit(gp, V, rest, node, outs)
        self.g.nodes.append(node)
        for (t, tag), nm in zip(outs, names):
            self.rt[t] = (nm, tag)
        for nd in post:
            src = next(((nm, tag) for (t, tag), nm in zip(outs, names) if t == nd.input[0]), None)
            if src is None or not self._is_conv(nd, src[1]):
                raise GroupImportError(gp.name, "a trailing Transpose is neither the pattern nor a layout conversion of an "
                                                "output", gp.op)
            self.rt[nd.output[0]] = (src[0], "I" if src[1] == "C" else "C")
            self.rule("layout Transpose (group output)")
        self.rule(f"group -> {gp.op}")

    # ---------------------------------------------------------------- re-emission
    def _kind_of(self, t: str) -> str:
        et = self.otype.get(t)
        if et == INT8:
            return "i8"
        if et in (INT32, INT64):
            return "i32"
        return "f"

    def reemit(self, gp, V, rest, node, outs) -> None:
        """Re-emit the group from the extracted IR node with the writer; the file's group must equal it node for node
        (operator, attributes, inputs: the same external tensors, the same constant values, the same internal wiring)."""
        mini = Graph()
        shapes, kinds, onnx_map, prim = {}, {}, {}, {}

        def ir_shape(t, tag):
            shp = self.oshape[t]
            return (shp[0], shp[2], shp[3], shp[1]) if (tag == "C" and len(shp) == 4) else tuple(shp)
        for t, (ir, tag) in V.uses.items():
            shapes[ir] = ir_shape(t, tag)
            kinds[ir] = self._kind_of(t)
            onnx_map[(ir, tag)] = t
            prim.setdefault(ir, tag)
        for i in node.inputs:
            if i and i in self.g.constants:
                mini.constants[i] = self.g.constants[i]
        for (t, tag), nm in zip(outs, node.outputs):
            shapes[nm] = ir_shape(t, tag)
            kinds[nm] = self._kind_of(t)
        mini.nodes = [node]
        w = Int8Writer.__new__(Int8Writer)
        w.g = mini
        w.shapes = dict(shapes)
        for c in mini.constants.values():
            w.shapes.setdefault(c.name, tuple(int(d) for d in c.shape))
        w.kinds = kinds
        w.s = Spec()
        w.report = {"groups": 0, "onnx_nodes": 0, "layout_transposes": 0, "carriers": 0, "groups_by_op": {}, "attrs_meta": 0,
                    "informational_attrs": 0}
        w.consts = mini.constants
        w.prod = {}
        w.inputs = {}
        w.prim = prim
        w.mode = {node.name: V.mode or (outs[0][1] if outs else "I")}
        w.need = {}
        w.onnx = dict(onnx_map)
        w._casts = set()
        w.cur = (node, gp.order)
        try:
            res = getattr(w, f"w_{node.op}")(node)
        except Exception as e:
            raise GroupImportError(gp.name, f"the extracted IR node has no canonical pattern ({type(e).__name__}: {e})", gp.op)
        w_outs = res[0] if (node.op.startswith("I8") or node.op in ("QBConv2D", "QBDense")) else [(res[0], res[1])]
        sval, scar = _spec_consts(w.s)
        named = [nd for nd in w.s.nodes if nd["meta"].get(NAME_KEY) is not None]
        if len(named) != len(rest):
            raise GroupImportError(gp.name, f"the group has {len(rest)} nodes, the canonical {node.op} pattern of the "
                                            f"extracted node {len(named)}", gp.op)
        sprod = {o: (k, j) for k, nd in enumerate(named) for j, o in enumerate(nd["outputs"])}
        fprod = {o: (k, j) for k, nd in enumerate(rest) for j, o in enumerate(nd.output)}

        def stok(t):
            if t in sval:
                return _ctoken(sval[t], t in scar)
            if t in sprod:
                return ("n",) + sprod[t]
            return ("x", t)

        def ftok(t):
            if t in self.cval:
                return _ctoken(self.cval[t], t in self.carrier)
            if t in fprod:
                return ("n",) + fprod[t]
            return ("x", t)
        for k, (a, b) in enumerate(zip(named, rest)):
            if a["op"] != b.op_type or (a["domain"] or "") != (b.domain or "") or \
                    len(a["outputs"]) != len([o for o in b.output if o]):
                raise GroupImportError(gp.name, f"node {k}: {b.op_type} where the canonical pattern has {a['op']}", gp.op)
            fa = _pyattrs(b)
            sa = {kk: (list(v) if isinstance(v, (list, tuple)) else v) for kk, v in a["attrs"].items()}
            if json.dumps(fa, sort_keys=True, default=str) != json.dumps(sa, sort_keys=True, default=str):
                raise GroupImportError(gp.name, f"node {k} ({b.op_type}): attributes {fa} vs canonical {sa}", gp.op)
            if [stok(t) for t in a["inputs"]] != [ftok(t) for t in b.input]:
                raise GroupImportError(gp.name, f"node {k} ({b.op_type}): inputs differ from the canonical pattern", gp.op)
        if [sprod[t] + (tag,) for t, tag in w_outs] != [fprod[t] + (tag,) for t, tag in outs]:
            raise GroupImportError(gp.name, "the pattern outputs differ from the canonical pattern's", gp.op)
        self.rule(f"re-emission check {node.op}")

    # ---------------------------------------------------------------- extraction helpers (writer primitives, backwards)
    def p_asr(self, V, t):
        sub = V.prod(t, "Sub")
        ci = V.prod(sub.input[0], "Cast")
        bs = V.prod(ci.input[0], "BitShift")
        cu = V.prod(bs.input[0], "Cast")
        ad = V.prod(cu.input[0], "Add")
        for q in (sub, ci, bs, cu, ad):
            V.use(q)
        n = V.cval(bs.input[1]).astype(np.int64)
        return ad.input[0], (int(n) if n.ndim == 0 else n)

    def p_rshr(self, V, t):
        a, n = self.p_asr(V, t)
        ad = V.prod(a, "Add")
        V.use(ad)
        return ad.input[0], n

    def p_mulhi(self, V, t):
        a, _n = self.p_asr(V, t)
        ad = V.prod(a, "Add")
        mu = V.prod(ad.input[0], "Mul")
        V.use(ad)
        V.use(mu)
        M = V.cval(mu.input[1]).astype(np.int64)
        return mu.input[0], (int(M) if M.ndim == 0 else M)

    def p_sat(self, V, t):
        w2 = V.prod(t, "Where")
        ls = V.prod(w2.input[0], "Less")
        w1 = V.prod(ls.input[0], "Where")
        gt = V.prod(w1.input[0], "Greater")
        for q in (w2, ls, w1, gt):
            V.use(q)
        lo, hi = V.cval(ls.input[1]).astype(np.int64), V.cval(gt.input[1]).astype(np.int64)
        return gt.input[0], (int(lo) if lo.ndim == 0 else lo), (int(hi) if hi.ndim == 0 else hi)

    def p_requant(self, V, t):
        a, right = self.p_rshr(V, t)
        b, M = self.p_mulhi(V, a)
        c, _lo, _hi = self.p_sat(V, b)
        mu = V.prod(c, "Mul")
        V.use(mu)
        L = V.cval(mu.input[1]).astype(np.int64)
        left = np.round(np.log2(L.astype(np.float64))).astype(np.int64)
        shift = np.where(left > 0, left, -np.asarray(right, np.int64))
        return mu.input[0], M, (int(shift) if np.ndim(shift) == 0 else shift)

    def p_to_i8(self, V, t):
        c = V.prod(t, "Cast")
        V.use(c)
        return self.p_sat(V, c.input[0])

    def p_centered(self, V, t):
        """Sub(Cast(q, INT64), z) -> (q, z)."""
        sub = V.prod(t, "Sub")
        ci = V.prod(sub.input[0], "Cast")
        V.use(sub)
        V.use(ci)
        return ci.input[0], V.ival(sub.input[1])

    def p_deq(self, V, t):
        """Mul(Sub(Cast(q, FLOAT), z), s) -> (q, s, z)."""
        mu = V.prod(t, "Mul")
        sub = V.prod(mu.input[0], "Sub")
        ci = V.prod(sub.input[0], "Cast")
        for q in (mu, sub, ci):
            V.use(q)
        return ci.input[0], V.fval(mu.input[1]), V.ival(sub.input[1])

    def p_rne8(self, V, t):
        c = V.prod(t, "Cast")
        cl = V.prod(c.input[0], "Clip")
        rd = V.prod(cl.input[0], "Round")
        for q in (c, cl, rd):
            V.use(q)
        return rd.input[0], V.ival(cl.input[1]), V.ival(cl.input[2])

    def p_l8enc(self, V, t):
        """Cast(Mul(Cast(Sign(w), INT64), cnt), INT8) -> w, cnt the 7-step bisection of Int8Writer.l8enc (step 6):
        cnt_j = Add(cnt_{j+1}, Where(GreaterOrEqual(Abs(w), Gather(beta, Add(cnt_{j+1}, 2^j - 1))), 2^j, 0)), the first
        step Where(GreaterOrEqual(Abs(w), Gather(beta, 63)), 64, 0). The constants are checked by the re-emission."""
        c = V.prod(t, "Cast")
        mu = V.prod(c.input[0], "Mul")
        cs = V.prod(mu.input[0], "Cast")
        sg = V.prod(cs.input[0], "Sign")
        used = [c, mu, cs, sg]
        cnt, ab = mu.input[1], None
        for j in range(L8ENC_STEPS):
            if j < L8ENC_STEPS - 1:
                ad = V.prod(cnt, "Add")
                wh = V.prod(ad.input[1], "Where")
                used += [ad, wh]
            else:
                wh = V.prod(cnt, "Where")
                used.append(wh)
            ge = V.prod(wh.input[0], "GreaterOrEqual")
            ga = V.prod(ge.input[1], "Gather")
            a = V.prod(ge.input[0], "Abs")
            used += [ge, ga]
            if ab is None:
                ab = a
                used.append(a)
            elif a is not ab:
                raise V.err("L8ENC: the bisection steps read different Abs nodes")
            if j < L8ENC_STEPS - 1:
                idx = V.prod(ga.input[1], "Add")
                used.append(idx)
                if idx.input[0] != ad.input[0]:
                    raise V.err("L8ENC: a bisection index is not the running count")
                cnt = ad.input[0]
        for q in used:
            V.use(q)
        if ab.input[0] != sg.input[0]:
            raise V.err("L8ENC: Sign and Abs of different tensors")
        return sg.input[0]

    def p_l8dec(self, V, t):
        """Gather(D_E, Add(Cast(q, INT64), 128)) -> (q, E)."""
        ga = V.prod(t, "Gather")
        ad = V.prod(ga.input[1], "Add")
        ci = V.prod(ad.input[0], "Cast")
        for q in (ga, ad, ci):
            V.use(q)
        e = _l8_exp_of(V.cval(ga.input[0]))
        if e is None:
            raise V.err("L8DEC table is not an l8 decode table")
        return ci.input[0], e

    def i8_operand(self, V, t: str, slot: int, mode: str):
        """An int8 operand: a runtime INT8 tensor (IR name) or an INT8 initializer (an int8 IR constant)."""
        c = V.const(t)
        if c is not None:
            d = np.asarray(c)
            if d.dtype != np.int8:
                raise V.err("int8 constant operand expected")
            if mode == "C" and d.ndim == 4:
                d = np.ascontiguousarray(np.transpose(d, C2I))
            return self.add_const(V.gp, slot, d, "i8", t)
        return V.runtime(t)[0]

    def _mode_of(self, V, tensors) -> str:
        tags = {self.rt[t][1] for t in tensors if t in self.rt and V.const(t) is None}
        if len(tags) > 1:
            raise V.err("inputs in different layouts")
        return tags.pop() if tags else "I"

    @staticmethod
    def _unchan(a) -> np.ndarray:
        return np.asarray(a).reshape(-1)

    def bin_operand(self, V, t: str, cshape):
        """Binary-layer operand: a runtime FLOAT bit tensor, or Bin(deq(q) [- t]) / Bin(L8DEC(q) [- t]).
        Returns (IR input, canon attrs, threshold array or None)."""
        if V.prod_or_none(t) is None:
            return V.runtime(t)[0], {}, None
        x, mode, thr, pre, bnd = self._bin(V, t, cshape)
        V.use_chain(bnd, pre, x)
        p = V.prod(x)
        if p.op_type == "Gather":
            q, e = self.p_l8dec(V, x)
            canon = {"input_binarize": mode, "x_code": "l8", "x_exp": e}
        else:
            q, s, z = self.p_deq(V, x)
            canon = {"input_binarize": mode, "x_scale": s, "x_zero": z}
        return V.runtime(q)[0], canon, thr

    def _binary_pad(self, V, t: str):
        p = V.prod_or_none(t)
        if p is not None and p.op_type == "Pad":
            V.use(p)
            pp = [int(v) for v in V.cval(p.input[1]).reshape(-1)]
            if len(pp) != 8 or pp[0] or pp[1] or pp[4] or pp[5]:
                raise V.err(f"binary Pad {pp}: spatial padding only")
            return p.input[0], {"pads": [pp[2], pp[6], pp[3], pp[7]], "pad_value": V.fval(p.input[2])}
        return t, {"pads": [0, 0, 0, 0], "pad_value": INFO}

    def _conv_w_bits(self, V, t: str):
        w = self._pm1(V, t, V.gp.op)
        if np.any(w == 0):
            raise V.err("binary convolution with zero weights")
        return np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1)))

    def _bconv_core(self, V, conv):
        """Conv(Pad?(Bin(x)), W) -> (x IR, w IR const, canon, thr)."""
        V.use(conv)
        a = V.attrs(conv)
        if any(a.get("pads", [])):
            raise V.err("binary Conv with implicit padding")
        src, pc = self._binary_pad(V, conv.input[0])
        xi, canon, thr = self.bin_operand(V, src, True)
        canon.update(pc)
        canon.update({"strides": list(a.get("strides", [1, 1])), "dilations": list(a.get("dilations", [1, 1])),
                      "groups": int(a.get("group", 1))})
        w = self.add_const(V.gp, 1, self._conv_w_bits(V, conv.input[1]), "bit", conv.input[1])
        return xi, w, canon, thr

    def _last_i8(self, V):
        return [q for q in V.nodes if q.op_type == "Cast" and _cast_to(V, q) == INT8][-1].output[0]

    # ================================================================ readers: (IR inputs, canon attrs, [(out, tag)])
    def r_I8Quantize(self, V):
        c = V.last()
        V.use(c)
        cl = V.prod(c.input[0], "Clip")
        ad = V.prod(cl.input[0], "Add")
        rd = V.prod(ad.input[0], "Round")
        dv = V.prod(rd.input[0], "Div")
        for q in (cl, ad, rd, dv):
            V.use(q)
        xi, tag = V.runtime(dv.input[0])
        V.mode = tag
        return [xi], {"scale": V.fval(dv.input[1]), "zero": V.ival(ad.input[1])}, [(c.output[0], tag)]

    def r_I8Dequantize(self, V):
        out = V.last().output[0]
        q, s, z = self.p_deq(V, out)
        xi, tag = V.runtime(q)
        V.mode = tag
        return [xi], {"scale": s, "zero": z}, [(out, tag)]

    def r_I8Lut(self, V):
        ga = V.first("Gather")
        V.use(ga)
        ad = V.prod(ga.input[1], "Add")
        V.use(ad)
        off = V.cval(ad.input[1])
        canon = {}
        if off.size == 1:
            ci = V.prod(ad.input[0], "Cast")
            V.use(ci)
            tab = V.cval(ga.input[0]).reshape(1, -1)
            canon["axis"] = INFO
        else:
            inner = V.prod(ad.input[0], "Add")
            ci = V.prod(inner.input[0], "Cast")
            V.use(inner)
            V.use(ci)
            big = [i for i, d in enumerate(off.shape) if d != 1]
            if len(big) != 1:
                raise V.err("per-channel I8Lut offsets along one axis")
            canon["axis"] = _Axis(big[0], off.ndim)
            tab = V.cval(ga.input[0]).reshape(int(off.size), -1)
        xi, tag = V.runtime(ci.input[0])
        V.mode = tag
        dt = "f32" if tab.dtype == np.float32 else "i8"
        if dt == "f32":
            canon["output"] = "f32"
        return [xi, self.add_const(V.gp, 1, tab, dt, ga.input[0])], canon, [(ga.output[0], tag)]

    def r_I8Add(self, V):
        out = V.last().output[0]
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        s_, om, osh = self.p_requant(V, ad.input[0])
        sm = V.prod(s_, "Add")
        V.use(sm)
        canon = {"y_zero": V.ival(ad.input[1]), "out_mult": om, "out_shift": osh, "qmin": qmin, "qmax": qmax}
        qs = []
        for t, pfx in ((sm.input[0], "a"), (sm.input[1], "b")):
            x, M, sh = self.p_requant(V, t)
            ml = V.prod(x, "Mul")
            V.use(ml)
            ls = int(round(np.log2(float(V.ival(ml.input[1])))))
            q, z = self.p_centered(V, ml.input[0])
            canon.update({f"{pfx}_mult": M, f"{pfx}_shift": sh, f"{pfx}_zero": z, "left_shift": ls})
            qs.append(q)
        mode = self._mode_of(V, qs)
        V.mode = mode
        return [self.i8_operand(V, t, j, mode) for j, t in enumerate(qs)], canon, [(out, mode)]

    def r_I8Mul(self, V):
        out = V.last().output[0]
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        x, M, sh = self.p_requant(V, ad.input[0])
        mu = V.prod(x, "Mul")
        V.use(mu)
        qa, za = self.p_centered(V, mu.input[0])
        qb, zb = self.p_centered(V, mu.input[1])
        mode = self._mode_of(V, [qa, qb])
        V.mode = mode
        irs = [self.i8_operand(V, qa, 0, mode), self.i8_operand(V, qb, 1, mode)]
        return irs, {"a_zero": za, "b_zero": zb, "mult": M, "shift": sh, "y_zero": V.ival(ad.input[1]), "qmin": qmin,
                     "qmax": qmax}, [(out, mode)]

    def r_I8AffineSum(self, V):
        out = V.last().output[0]
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        acc, M, sh = self.p_requant(V, ad.input[0])
        ab = V.prod(acc, "Add")
        V.use(ab)
        B = V.cval(ab.input[1])
        terms = []
        t = ab.input[0]
        while True:
            p = V.prod(t)
            if p.op_type == "Mul":
                terms.insert(0, p)
                V.use(p)
                break
            if p.op_type != "Add":
                raise V.err("I8AffineSum: Add chain of Mul terms expected")
            V.use(p)
            m = V.prod(p.input[1], "Mul")
            V.use(m)
            terms.insert(0, m)
            t = p.input[0]
        qs, zs, Ks = [], [], []
        for m in terms:
            q, z = self.p_centered(V, m.input[0])
            qs.append(q)
            zs.append(z)
            Ks.append(self._unchan(V.cval(m.input[1])))
        mode = self._mode_of(V, qs)
        V.mode = mode
        k = len(qs)
        irs = [self.i8_operand(V, q, j, mode) for j, q in enumerate(qs)]
        irs.append(self.add_const(V.gp, k, np.stack(Ks).astype(np.int64), "i32", terms[0].input[1]))
        irs.append(self.add_const(V.gp, k + 1, self._unchan(B).astype(np.int64), "i32", ab.input[1]))
        return irs, {"zeros": zs, "mult": M, "shift": sh, "y_zero": V.ival(ad.input[1]), "qmin": qmin, "qmax": qmax}, \
            [(out, mode)]

    def r_I8Concat(self, V):
        cc = V.last()
        if cc.op_type != "Concat":
            raise V.err("I8Concat: final Concat expected")
        V.use(cc)
        srcs, mults, zeros, shifts, yz = [], [], [], [], INFO
        for t in cc.input:
            if V.prod_or_none(t) is None:
                srcs.append(t)
                mults.append(0)
                zeros.append(INFO)
                shifts.append(INFO)
                continue
            v, lo, hi = self.p_to_i8(V, t)
            ad = V.prod(v, "Add")
            V.use(ad)
            yz = V.ival(ad.input[1])
            x, M, sh = self.p_requant(V, ad.input[0])
            q, z = self.p_centered(V, x)
            srcs.append(q)
            mults.append(M)
            zeros.append(z)
            shifts.append(sh)
        mode = self._mode_of(V, srcs)
        V.mode = mode
        ax = int(V.attrs(cc)["axis"])
        if mode == "C":
            if ax != 1:
                raise V.err("I8Concat of NCHW views on the channel axis only")
            ax = -1
        irs = [self.i8_operand(V, t, j, mode) for j, t in enumerate(srcs)]
        return irs, {"axis": ax, "mults": mults, "zeros": zeros, "shifts": shifts, "y_zero": yz}, [(cc.output[0], mode)]

    def r_I8Transpose(self, V):
        nd = V.single("Transpose")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("I8Transpose of an NCHW view")
        V.mode = "I"
        return [xi], {"perm": [int(p) for p in V.attrs(nd)["perm"]]}, [(nd.output[0], "I")]

    def r_I8Pad(self, V):
        nd = V.single("Pad")
        xi, tag = V.runtime(nd.input[0])
        V.mode = tag
        pp = [int(v) for v in V.cval(nd.input[1]).reshape(-1)]
        r = len(pp) // 2
        dims = list(I2C) if (tag == "C" and r == 4) else list(range(r))
        ir = [0] * (2 * r)
        for k, d in enumerate(dims):
            ir[2 * d], ir[2 * d + 1] = pp[k], pp[r + k]
        return [xi], {"pads": ir, "pad_q": V.ival(nd.input[2])}, [(nd.output[0], tag)]

    def r_I8MaxPool2D(self, V):
        nd = V.single("MaxPool")
        a = V.attrs(nd)
        xi, tag = V.runtime(nd.input[0])
        if tag != "C":
            raise V.err("I8MaxPool2D of a non-NCHW tensor")
        V.mode = "C"
        k = list(a["kernel_shape"])
        p = list(a.get("pads", [0, 0, 0, 0]))
        return [xi], {"kernel": k, "strides": list(a.get("strides", k)), "pads": [int(p[i]) for i in (0, 2, 1, 3)]}, \
            [(nd.output[0], "C")]

    def r_I8AvgPool2D(self, V):
        out = V.last().output[0]
        v, lo, hi = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        cq = V.prod(ad.input[0], "Cast")
        mq = V.prod(cq.input[0], "Mul")
        sg = V.prod(mq.input[0], "Sign")
        fl = V.prod(mq.input[1], "Floor")
        dv = V.prod(fl.input[0], "Div")
        num = V.prod(dv.input[0], "Add")
        m2 = V.prod(num.input[0], "Mul")
        ab = V.prod(m2.input[0], "Abs")
        ca = V.prod(ab.input[0], "Cast")
        ci = V.prod(ca.input[0], "ConvInteger")
        for q in (ad, cq, mq, sg, fl, dv, num, m2, ab, ca, ci):
            V.use(q)
        a = V.attrs(ci)
        src = ci.input[0]
        pads = [0, 0, 0, 0]
        p = V.prod_or_none(src)
        if p is not None and p.op_type == "Pad":
            V.use(p)
            pp = [int(x) for x in V.cval(p.input[1]).reshape(-1)]
            pads = [pp[2], pp[6], pp[3], pp[7]]
            src = p.input[0]
        xi, tag = V.runtime(src)
        V.mode = "C"
        k = list(a["kernel_shape"])
        st = list(a.get("strides", k))
        n_, c_, h, w = self.oshape[src]
        oh, ow = self.oshape[out][2:]
        c0, c1 = (pool_counts(h, w, oh, ow, k, st, pads, i) for i in (0, 1))
        inc = INFO if np.array_equal(c0, c1) else (1 if V.cval(num.input[1]).size == 1 else 0)
        return [xi], {"kernel": k, "strides": list(a.get("strides", k)), "pads": pads, "x_zero": V.ival(ci.input[2]),
                      "count_include_pad": inc}, [(out, "C")]

    def _mean(self, V):
        out = V.last().output[0]
        v, lo, hi = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        acc, M, sh = self.p_requant(V, ad.input[0])
        rs = V.prod(acc, "ReduceSum")
        V.use(rs)
        q, z = self.p_centered(V, rs.input[0])
        xi, tag = V.runtime(q)
        axes = [int(x) for x in V.cval(rs.input[1]).reshape(-1)]
        kd = int(V.attrs(rs).get("keepdims", 1))
        return xi, tag, axes, kd, {"x_zero": z, "mult": M, "shift": sh, "y_zero": V.ival(ad.input[1]), "keepdims": kd}, out

    def r_I8GlobalAvgPool(self, V):
        xi, tag, axes, kd, canon, out = self._mean(V)
        if tag != "C" or axes != [2, 3]:
            raise V.err("I8GlobalAvgPool: ReduceSum over H, W of an NCHW view")
        V.mode = "C"
        return [xi], canon, [(out, "C" if kd else "I")]

    def r_I8ReduceMean(self, V):
        xi, tag, axes, kd, canon, out = self._mean(V)
        if tag != "I" or len(axes) != 1:
            raise V.err("I8ReduceMean over one axis of an IR-layout tensor")
        V.mode = "I"
        canon["axis"] = axes[0]
        return [xi], canon, [(out, "I")]

    # ---------------------------------------------------------------- GEMMs
    def _gemm_tail(self, V, acc: str, mode: str):
        """Epilogue of an INT32 accumulator -> (IR inputs 2.., canon, outs)."""
        casts = [q for q in V.cons_in.get(acc, []) if q.op_type == "Cast"]
        tf = next((q for q in casts if _cast_to(V, q) == FLOAT), None)
        if tf is not None:
            V.use(tf)
            mu = V.cons(tf.output[0], "Mul")
            V.use(mu)
            scale = self._unchan(V.cval(mu.input[1])).astype(np.float32)
            y = mu.output[0]
            bias = ""
            ad = V.cons(y, "Add")
            if ad is not None:
                V.use(ad)
                bias = self.add_const(V.gp, 2, self._unchan(V.cval(ad.input[1])).astype(np.float32), "f32", ad.input[1])
                y = ad.output[0]
            ins = [bias, self.add_const(V.gp, 3, scale, "f32", mu.input[1])]
            canon = {"output": "f32"}
            if V.cons(y, "Sign") is not None:
                out = V.last().output[0]
                if self.p_l8enc(V, out) != y:
                    raise V.err("l8 output: L8ENC of the f32 value expected")
                canon.update({"output": "l8", "y_exp": INFO})
                y = out
            return ins, canon, [(y, mode)]
        c64 = next(q for q in casts if _cast_to(V, q) == INT64)
        V.use(c64)
        a64 = c64.output[0]
        outs_tail, bt = [], None
        for q in V.cons_in.get(a64, []):
            if q.op_type == "GreaterOrEqual":
                wh = V.cons(q.output[0], "Where")
                V.use(q)
                V.use(wh)
                bt = self._unchan(V.cval(q.input[1]))
                outs_tail.append((wh.output[0], mode))
        out = self._last_i8(V)
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        x, M, sh = self.p_requant(V, ad.input[0])
        x, _lo, _hi = self.p_sat(V, x)
        bias = ""
        if x != a64:
            b = V.prod(x, "Add")
            V.use(b)
            if b.input[0] != a64:
                raise V.err("bias Add of the accumulator expected")
            bias = self.add_const(V.gp, 2, self._unchan(V.cval(b.input[1])), "i32", b.input[1])
        ins = [bias, self.add_const(V.gp, 3, self._unchan(M), "i32", "mult"), self.add_const(V.gp, 4, self._unchan(sh), "i32", "shift")]
        if bt is not None:
            ins.append(self.add_const(V.gp, 5, bt, "i32", "bit_thr"))
        return ins, {"output": "i8", "y_zero": V.ival(ad.input[1]), "qmin": qmin, "qmax": qmax}, [(out, mode)] + outs_tail

    def r_I8Conv2D(self, V):
        ci = V.first("ConvInteger")
        V.use(ci)
        a = V.attrs(ci)
        src = ci.input[0]
        pads, pq = [0, 0, 0, 0], INFO
        p = V.prod_or_none(src)
        if p is not None and p.op_type == "Pad":
            V.use(p)
            pp = [int(x) for x in V.cval(p.input[1]).reshape(-1)]
            pads = [pp[2], pp[6], pp[3], pp[7]]
            pq = V.ival(p.input[2])
            src = p.input[0]
        xi, tag = V.runtime(src)
        if tag != "C":
            raise V.err("I8Conv2D input must be an NCHW view")
        V.mode = "C"
        W = np.ascontiguousarray(np.transpose(V.cval(ci.input[1]).astype(np.int8), (0, 2, 3, 1)))
        w = self.add_const(V.gp, 1, W, "i8", ci.input[1])
        rest_ins, canon, outs = self._gemm_tail(V, ci.output[0], "C")
        canon.update({"x_zero": V.ival(ci.input[2]), "pad_q": pq, "pads": pads, "strides": list(a.get("strides", [1, 1])),
                      "dilations": list(a.get("dilations", [1, 1])), "groups": int(a.get("group", 1))})
        return [xi, w] + rest_ins, canon, outs

    def r_I8Dense(self, V):
        mm = V.first("MatMulInteger")
        V.use(mm)
        xi, tag = V.runtime(mm.input[0])
        if tag != "I":
            raise V.err("I8Dense input in an NCHW view")
        V.mode = "I"
        w = self.add_const(V.gp, 1, np.ascontiguousarray(V.cval(mm.input[1]).astype(np.int8).T), "i8", mm.input[1])
        rest_ins, canon, outs = self._gemm_tail(V, mm.output[0], "I")
        canon["x_zero"] = V.ival(mm.input[2])
        return [xi, w] + rest_ins, canon, outs

    def r_I8MatMul(self, V):
        mm = V.first("MatMulInteger")
        V.use(mm)
        V.mode = "I"
        b_src, tb = mm.input[1], 0
        p = V.prod_or_none(b_src)
        if p is not None and p.op_type == "Transpose":
            V.use(p)
            b_src, tb = p.input[0], 1
        A = self.i8_operand(V, mm.input[0], 0, "I")
        B = self.i8_operand(V, b_src, 1, "I")
        canon = {"a_zero": V.ival(mm.input[2]), "b_zero": V.ival(mm.input[3]), "transpose_b": tb}
        cf = V.cons(mm.output[0], "Cast")
        if _cast_to(V, cf) == FLOAT:
            V.use(cf)
            mu = V.cons(cf.output[0], "Mul")
            V.use(mu)
            canon.update({"output": "f32", "scale": V.fval(mu.input[1])})
            return [A, B], canon, [(mu.output[0], "I")]
        out = V.last().output[0]
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        x, M, sh = self.p_requant(V, ad.input[0])
        x, _lo, _hi = self.p_sat(V, x)
        V.use(V.prod(x, "Cast"))
        canon.update({"output": "i8", "mult": M, "shift": sh, "y_zero": V.ival(ad.input[1]), "qmin": qmin, "qmax": qmax})
        return [A, B], canon, [(out, "I")]

    def r_I8LayerNorm(self, V):
        out = V.last().output[0]
        v, qmin, qmax = self.p_rne8(V, out)
        ad = V.prod(v, "Add")
        mg = V.prod(ad.input[0], "Mul")
        mr = V.prod(mg.input[0], "Mul")
        rc = V.prod(mr.input[1], "Cast")
        dv = V.prod(rc.input[0], "Div")
        sq = V.prod(dv.input[1], "Sqrt")
        a2 = V.prod(sq.input[0], "Add")
        c2 = V.prod(a2.input[1], "Mul")
        c1 = V.prod(c2.input[0], "Mul")
        uc = V.prod(mr.input[0], "Cast")
        su = V.prod(uc.input[0], "Sub")
        mt = V.prod(su.input[0], "Mul")
        sub = V.prod(mt.input[0], "Sub")
        ci = V.prod(sub.input[0], "Cast")
        for q in V.nodes:
            V.use(q)
        xs, eps, z = float(V.cval(dv.input[0])), float(V.cval(c1.input[0])), V.ival(sub.input[1])
        xi, tag = V.runtime(ci.input[0])
        if tag != "I":
            raise V.err("I8LayerNorm of an NCHW view")
        V.mode = "I"
        G = self.add_const(V.gp, 1, V.cval(mg.input[1]).astype(np.float32), "f32", mg.input[1])
        B = self.add_const(V.gp, 2, V.cval(ad.input[1]).astype(np.float32), "f32", ad.input[1])
        return [xi, G, B], {"x_zero": z, "x_scale": xs, "eps": eps, "qmin": qmin, "qmax": qmax}, [(out, "I")]

    def r_I8Softmax(self, V):
        for q in V.nodes:
            V.use(q)
        ga = V.first("Gather")
        E = self.add_const(V.gp, 1, V.cval(ga.input[0]).astype(np.int32), "i32", ga.input[0])
        mx = V.first("ReduceMax")
        ci = V.prod(mx.input[0], "Cast")
        xi, tag = V.runtime(ci.input[0])
        if tag != "I":
            raise V.err("I8Softmax of an NCHW view")
        V.mode = "I"
        num = float(V.cval(V.first("Div").input[0]))
        out = V.last().output[0]
        if V.last().op_type == "Mul":
            return [xi, E], {"output": "f32"}, [(out, "I")]
        cl = V.first("Clip")
        add = V.prod(cl.input[0], "Add")
        return [xi, E], {"output": "i8", "inv_scale": num, "y_zero": V.ival(add.input[1]), "qmin": V.ival(cl.input[1]),
                         "qmax": V.ival(cl.input[2])}, [(out, "I")]

    def r_I8Gather(self, V):
        for q in V.nodes:
            V.use(q)
        last = V.last()
        fill = V.ival(last.input[2])
        ci = V.first("Cast")
        if V.const(ci.input[0]) is not None:
            ids = self.add_const(V.gp, 1, V.cval(ci.input[0]).astype(np.int32), "i32", ci.input[0])
        else:
            ids = V.runtime(ci.input[0])[0]
        V.mode = "I"
        rows = V.prod(last.input[1])
        if rows.op_type == "Gather":
            c = V.const(rows.input[0])
            tab = self.add_const(V.gp, 0, np.asarray(c, np.int8), "i8", rows.input[0]) if c is not None \
                else V.runtime(rows.input[0])[0]
            return [tab, ids], {"fill_q": fill}, [(last.output[0], "I")]
        zm = None
        if rows.op_type == "Where" and V.prod(rows.input[0]).op_type == "Less":
            zg = V.prod(V.prod(rows.input[0]).input[0], "Gather")
            zm = V.cval(zg.input[0])
            rows = V.prod(rows.input[2], "Where")
        sg = V.prod(V.prod(rows.input[0], "Greater").input[0], "Gather")
        qpm = V.prod(V.prod(rows.input[1], "Slice").input[0], "Gather")
        sv = V.cval(sg.input[0])
        ins = [self.add_const(V.gp, 0, np.where(sv < 0, -1, 1).astype(np.int8), "bit", sg.input[0]), ids,
               self.add_const(V.gp, 2, V.cval(qpm.input[0]).astype(np.int8), "i8", qpm.input[0])]
        if zm is not None:
            ins.append(self.add_const(V.gp, 3, np.where(zm < 0, -1, 1).astype(np.int8), "bit", "zero_mask"))
        return ins, {"fill_q": fill}, [(last.output[0], "I")]

    # ---------------------------------------------------------------- binary layers
    def _post_affine(self, V, t: str, slots=(2, 3)):
        ins = ["", ""]
        mu = V.cons(t, "Mul")
        if mu is not None and V.const(mu.input[1]) is not None:
            V.use(mu)
            ins[0] = self.add_const(V.gp, slots[0], self._unchan(V.cval(mu.input[1])).astype(np.float32), "f32", mu.input[1])
            t = mu.output[0]
        ad = V.cons(t, "Add")
        if ad is not None and V.const(ad.input[1]) is not None:
            V.use(ad)
            ins[1] = self.add_const(V.gp, slots[1], self._unchan(V.cval(ad.input[1])).astype(np.float32), "f32", ad.input[1])
            t = ad.output[0]
        return ins, t

    def _out_i8_or_f32(self, V, t: str, mode: str):
        idn = V.cons(t, "Identity")
        if idn is not None:
            V.use(idn)
            return {"output": "f32"}, [(idn.output[0], mode)]
        out = V.last().output[0]
        v, qmin, qmax = self.p_rne8(V, out)
        if v != t:
            raise V.err("RNE8 of the epilogue value expected")
        return {"output": "i8", "qmin": qmin, "qmax": qmax}, [(out, mode)]

    def r_I8BDense(self, V):
        mm = V.first("MatMul")
        V.use(mm)
        V.mode = "I"
        xi, canon, thr = self.bin_operand(V, mm.input[0], None)
        w = self._pm1(V, mm.input[1], "I8BDense")              # [K, O]
        wo = np.ascontiguousarray(w.T)
        ins = [xi, self.add_const(V.gp, 1, np.where(wo < 0, -1, 1).astype(np.int8), "bit", mm.input[1]), "", "", "", ""]
        if np.any(wo == 0):
            ins[5] = self.add_const(V.gp, 5, np.where(wo == 0, -1, 1).astype(np.int8), "bit", mm.input[1])
        (ins[2], ins[3]), t = self._post_affine(V, mm.output[0])
        if thr is not None:
            ins[4] = self.add_const(V.gp, 4, np.asarray(thr, np.float32).reshape(-1), "f32", "thr")
        oc, outs = self._out_i8_or_f32(V, t, "I")
        canon.update(oc)
        return ins, canon, outs

    def r_I8BConv2D(self, V):
        conv = V.first("Conv")
        xi, w, canon, thr = self._bconv_core(V, conv)
        V.mode = "C"
        ins = [xi, w, "", "", ""]
        (ins[2], ins[3]), t = self._post_affine(V, conv.output[0])
        if thr is not None:
            ins[4] = self.add_const(V.gp, 4, np.asarray(thr, np.float32).reshape(-1), "f32", "thr")
        oc, outs = self._out_i8_or_f32(V, t, "C")
        canon.update(oc)
        return ins, canon, outs

    def _residual_r(self, V, t: str):
        if V.prod(t).op_type == "Gather":
            q, e = self.p_l8dec(V, t)
            return q, {"r_code": "l8", "r_exp": e}
        q, s, z = self.p_deq(V, t)
        return q, {"r_code": "linear", "r_scale": s, "r_zero": z}

    def r_I8BConv2DRes(self, V):
        conv = V.first("Conv")
        xi, w, canon, thr = self._bconv_core(V, conv)
        V.mode = "C"
        ins = [xi, w] + [""] * 13
        vec = lambda t: self._unchan(V.cval(t)).astype(np.float32)     # noqa: E731
        mu = V.cons(conv.output[0], "Mul")
        V.use(mu)
        ad = V.cons(mu.output[0], "Add")
        V.use(ad)
        ins[2] = self.add_const(V.gp, 2, vec(mu.input[1]), "f32", mu.input[1])
        ins[3] = self.add_const(V.gp, 3, vec(ad.input[1]), "f32", ad.input[1])
        v = ad.output[0]
        stage = 0                   # pre stage in order: pre_slope, pre_mul, pre_add, gate, gate_add
        while True:
            nx = V.cons(v)
            if nx is None:
                raise V.err("I8BConv2DRes: epilogue chain expected")
            const1 = V.const(nx.input[1]) is not None
            if nx.op_type == "PRelu" and stage == 0:
                ins[10] = self.add_const(V.gp, 10, vec(nx.input[1]), "f32", nx.input[1])
                stage = 1
            elif nx.op_type == "Mul" and stage <= 1 and const1:
                ins[11] = self.add_const(V.gp, 11, vec(nx.input[1]), "f32", nx.input[1])
                stage = 2
            elif nx.op_type == "Add" and stage <= 2 and const1:
                ins[12] = self.add_const(V.gp, 12, vec(nx.input[1]), "f32", nx.input[1])
                stage = 3
            elif nx.op_type == "Mul" and stage <= 3 and not const1:
                g, gs, gz = self.p_deq(V, nx.input[1])
                ins[13] = V.runtime(g)[0]
                canon.update({"g_scale": gs, "g_zero": gz})
                stage = 4
            elif nx.op_type == "Add" and stage == 4 and const1:
                ins[14] = self.add_const(V.gp, 14, vec(nx.input[1]), "f32", nx.input[1])
                stage = 5
            elif nx.op_type == "Add" and not const1:
                break
            else:
                raise V.err(f"I8BConv2DRes: unexpected {nx.op_type} in the epilogue")
            V.use(nx)
            v = nx.output[0]
        ua = V.cons(v, "Add")
        V.use(ua)
        rq, rc = self._residual_r(V, ua.input[1])
        canon.update(rc)
        ins[5] = V.runtime(rq)[0]
        u = ua.output[0]
        pr = V.cons(u, "PRelu")
        V.use(pr)
        ins[6] = self.add_const(V.gp, 6, vec(pr.input[1]), "f32", pr.input[1])
        wv = pr.output[0]
        ob = V.cons(wv, "Add")
        if ob is not None and V.const(ob.input[1]) is not None:
            V.use(ob)
            ins[7] = self.add_const(V.gp, 7, vec(ob.input[1]), "f32", ob.input[1])
            wv = ob.output[0]
        outs_bits = []
        for s_, src in ((u, "sum"), (wv, "out")):
            for q in V.cons_in.get(s_, []):
                if q.op_type in ("GreaterOrEqual", "Greater"):
                    wh = V.cons(q.output[0], "Where")
                    V.use(q)
                    V.use(wh)
                    ins[8] = self.add_const(V.gp, 8, vec(q.input[1]), "f32", q.input[1])
                    canon.update({"bit_src": src, "bit_binarize": "gt" if q.op_type == "Greater" else "ge"})
                    outs_bits.append((wh.output[0], "C"))
        if not outs_bits:
            canon.update({"bit_src": INFO, "bit_binarize": INFO})
        if thr is not None:
            ins[4] = self.add_const(V.gp, 4, np.asarray(thr, np.float32).reshape(-1), "f32", "thr")
        out = self._last_i8(V)
        if V.cons(wv, "Sign") is not None:
            if self.p_l8enc(V, out) != wv:
                raise V.err("L8ENC of w expected")
            canon.update({"output": "l8", "y_exp": INFO})
        else:
            c = V.prod(out, "Cast")
            cl = V.prod(c.input[0], "Clip")
            a_ = V.prod(cl.input[0], "Add")
            rd = V.prod(a_.input[0], "Round")
            for q in (c, cl, a_, rd):
                V.use(q)
            if rd.input[0] != wv:
                raise V.err("linear output: Round of w expected")
            canon.update({"output": "i8", "y_zero": V.ival(a_.input[1]), "qmin": V.ival(cl.input[1]),
                          "qmax": V.ival(cl.input[2]), "y_exp": INFO})
        return ins, canon, [(out, "C")] + outs_bits

    def r_I8BConv2DResI(self, V):
        conv = V.first("Conv")
        xi, w, canon, thr = self._bconv_core(V, conv)
        V.mode = "C"
        ins = [xi, w] + [""] * 14
        vec = lambda t: self._unchan(V.cval(t)).astype(np.int64)      # noqa: E731
        cd = V.cons(conv.output[0], "Cast")
        V.use(cd)
        m1 = V.cons(cd.output[0], "Mul")
        V.use(m1)
        ins[6] = self.add_const(V.gp, 6, vec(m1.input[1]), "i32", m1.input[1])
        t = m1.output[0]
        nx = V.cons(t, "Add")
        if V.cons(nx.output[0], "Greater") is not None:       # pre stage: SAT(Add(pre, pre_b), pre_lo, pre_hi)
            V.use(nx)
            ins[12] = self.add_const(V.gp, 12, vec(nx.input[1]), "i32", nx.input[1])
            w1 = V.cons(nx.output[0], "Where")
            w2 = V.cons(V.cons(w1.output[0], "Less").output[0], "Where")
            _x, lo, hi = self.p_sat(V, w2.output[0])
            ins[13] = self.add_const(V.gp, 13, self._unchan(lo), "i32", "pre_lo")
            ins[14] = self.add_const(V.gp, 14, self._unchan(hi), "i32", "pre_hi")
            nx = V.cons(w2.output[0], "Add")
        V.use(nx)
        ins[7] = self.add_const(V.gp, 7, vec(nx.input[1]), "i32", nx.input[1])
        acc = nx.output[0]
        ar = next((q for q in V.cons_in.get(acc, []) if q.op_type == "Add" and V.const(q.input[1]) is None), None)
        if ar is not None:
            V.use(ar)
            mr = V.prod(ar.input[1], "Mul")
            V.use(mr)
            ins[8] = self.add_const(V.gp, 8, vec(mr.input[1]), "i32", mr.input[1])
            r = mr.input[0]
            rt = 1
            p = V.prod(r)
            if p.op_type == "Concat":
                V.use(p)
                rt = len(p.input)
                r = p.input[0]
            cr = V.prod(r, "Cast")
            V.use(cr)
            ins[5] = V.runtime(cr.input[0])[0]
            acc = ar.output[0]
            canon["r_tile"] = rt
        else:
            canon["r_tile"] = INFO
        wv = acc
        wh = next((q for q in V.cons_in.get(acc, []) if q.op_type == "Where" and q.input[2] == acc), None)
        if wh is not None:
            V.use(wh)
            V.use(V.prod(wh.input[0], "Less"))
            ms = V.prod(wh.input[1], "Mul")
            V.use(ms)
            ssh = int(round(np.log2(float(V.ival(ms.input[1])))))
            _a, sl = self.p_mulhi(V, ms.input[0])
            ins[10] = self.add_const(V.gp, 10, self._unchan(sl), "i32", "slope")
            canon["slope_shift"] = ssh
            wv = wh.output[0]
        else:
            canon["slope_shift"] = INFO
        ob = next((q for q in V.cons_in.get(wv, []) if q.op_type == "Add" and V.const(q.input[1]) is not None
                   and np.asarray(V.const(q.input[1])).size > 1), None)
        if ob is not None:
            V.use(ob)
            ins[11] = self.add_const(V.gp, 11, vec(ob.input[1]), "i32", ob.input[1])
            wv = ob.output[0]
        outs_bits = []
        for s_, src in (((acc, "sum"), (wv, "out")) if wv != acc else ((acc, INFO),)):
            for q in V.cons_in.get(s_, []):
                if q.op_type == "GreaterOrEqual":
                    whb = V.cons(q.output[0], "Where")
                    V.use(q)
                    V.use(whb)
                    ins[9] = self.add_const(V.gp, 9, vec(q.input[1]), "i32", q.input[1])
                    canon["bit_src"] = src
                    outs_bits.append((whb.output[0], "C"))
        if not outs_bits:
            canon["bit_src"] = INFO
        out = self._last_i8(V)
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        x, shift = self.p_rshr(V, ad.input[0])
        if x != wv:
            raise V.err("RSHR of w expected")
        if thr is not None:
            ins[4] = self.add_const(V.gp, 4, np.asarray(thr, np.float32).reshape(-1), "f32", "thr")
        canon.update({"shift": int(shift), "y_zero": V.ival(ad.input[1]), "qmin": qmin, "qmax": qmax})
        return ins, canon, [(out, "C")] + outs_bits

    def r_I8GatedAdd(self, V):
        out = self._last_i8(V)
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        acc, shift = self.p_rshr(V, ad.input[0])
        ab = V.prod(acc, "Add")
        a2 = V.prod(ab.input[0], "Add")
        tx = V.prod(a2.input[0], "Mul")
        tp = V.prod(a2.input[1], "Mul")
        pg = V.prod(tp.input[0], "Mul")
        for q in (ab, a2, tx, tp, pg):
            V.use(q)
        x, zx = self.p_centered(V, tx.input[0])
        p, zp = self.p_centered(V, pg.input[0])
        g, zg = self.p_centered(V, pg.input[1])
        V.mode = "C"
        vec = lambda t: self._unchan(V.cval(t)).astype(np.int64)      # noqa: E731
        ins = [V.runtime(x)[0], V.runtime(p)[0], V.runtime(g)[0], self.add_const(V.gp, 3, vec(tx.input[1]), "i32", tx.input[1]),
               self.add_const(V.gp, 4, vec(tp.input[1]), "i32", tp.input[1]),
               self.add_const(V.gp, 5, vec(ab.input[1]), "i32", ab.input[1])]
        rows, outs_bits = [], []
        for q in V.nodes:
            if q.op_type == "GreaterOrEqual" and q.input[0] == acc:
                wh = V.cons(q.output[0], "Where")
                V.use(q)
                V.use(wh)
                rows.append(vec(q.input[1]))
                outs_bits.append((wh.output[0], "C"))
        if rows:
            ins.append(self.add_const(V.gp, 6, np.stack(rows), "i32", "bit_thr"))
        return ins, {"x_zero": zx, "p_zero": zp, "g_zero": zg, "shift": int(shift), "y_zero": V.ival(ad.input[1]),
                     "qmin": qmin, "qmax": qmax}, [(out, "C")] + outs_bits

    def r_I8GapAffine(self, V):
        out = V.last().output[0]
        v, qmin, qmax = self.p_to_i8(V, out)
        ad = V.prod(v, "Add")
        V.use(ad)
        acc, shift = self.p_rshr(V, ad.input[0])
        terms = []
        t = acc
        while V.prod_or_none(t) is not None:
            a_ = V.prod(t, "Add")
            m = V.prod(a_.input[1], "Mul")
            V.use(a_)
            V.use(m)
            terms.insert(0, m)
            t = a_.input[0]
        B = self._unchan(V.cval(t)).astype(np.int64)
        qs, zs, Ks = [], [], []
        for m in terms:
            rs = V.prod(m.input[0], "ReduceSum")
            V.use(rs)
            q, z = self.p_centered(V, rs.input[0])
            qs.append(V.runtime(q)[0])
            zs.append(z)
            Ks.append(self._unchan(V.cval(m.input[1])).astype(np.int64))
        V.mode = "C"
        k = len(qs)
        ins = qs + [self.add_const(V.gp, k, np.stack(Ks), "i32", "K"), self.add_const(V.gp, k + 1, B, "i32", t)]
        return ins, {"zeros": zs, "shift": int(shift), "y_zero": V.ival(ad.input[1]), "qmin": qmin, "qmax": qmax}, [(out, "C")]

    def r_I8BMatMul(self, V):
        mm = V.first("MatMul")
        V.use(mm)
        V.mode = "I"
        b_src, tb = mm.input[1], 0
        p = V.prod_or_none(b_src)
        if p is not None and p.op_type == "Transpose":
            V.use(p)
            b_src, tb = p.input[0], 1
        canon = {"transpose_b": tb}
        ins = []
        for j, (t, pfx) in enumerate(((mm.input[0], "a"), (b_src, "b"))):
            x, mode, thr, pre, bnd = self._bin(V, t, None)
            V.use_chain(bnd, pre, x)
            q, s, z = self.p_deq(V, x)
            canon.update({f"{pfx}_binarize": mode, f"{pfx}_scale": s, f"{pfx}_zero": z})
            if thr is not None:
                canon[f"{pfx}_threshold"] = float(np.asarray(thr).reshape(-1)[0])
            ins.append(self.i8_operand(V, q, j, "I"))
        mab = V.cons(mm.output[0], "Mul")
        V.use(mab)
        canon["c_ab"] = V.ival(mab.input[1])
        t = mab.output[0]
        for key in ("c_rs", "c_cs"):
            a_ = V.cons(t, "Add")
            m = V.prod(a_.input[1], "Mul")
            rs = V.prod(m.input[0], "ReduceSum")
            for q in (a_, m, rs):
                V.use(q)
            canon[key] = V.ival(m.input[1])
            t = a_.output[0]
        a0 = V.cons(t, "Add")
        V.use(a0)
        canon["c0"] = V.ival(a0.input[1])
        pa_ins, t = self._post_affine(V, a0.output[0])
        ins += pa_ins
        oc, outs = self._out_i8_or_f32(V, t, "I")
        canon.update(oc)
        return ins, canon, outs

    def _levels_r(self, V, t: str):
        rd = V.prod(t, "Round")
        mu = V.prod(rd.input[0], "Mul")
        cl = V.prod(mu.input[0], "Clip")
        for q in (rd, mu, cl):
            V.use(q)
        return cl.input[0], int(round(np.log2(V.ival(mu.input[1]) + 1)))

    def r_QBConv2D(self, V):
        conv = V.first("Conv")
        V.use(conv)
        a = V.attrs(conv)
        x, bits = self._levels_r(V, conv.input[0])
        xi, tag = V.runtime(x)
        V.mode = "C"
        w = self.add_const(V.gp, 1, self._conv_w_bits(V, conv.input[1]), "bit", conv.input[1])
        pa_ins, t = self._post_affine(V, conv.output[0])
        idn = V.cons(t, "Identity")
        V.use(idn)
        pads = list(a.get("pads", [0, 0, 0, 0]))
        return [xi, w] + pa_ins, {"act_bits": bits, "act_quant": "dorefa", "activation": "none",
                                  "pads": [int(pads[i]) for i in (0, 2, 1, 3)], "strides": list(a.get("strides", [1, 1])),
                                  "dilations": list(a.get("dilations", [1, 1])), "groups": int(a.get("group", 1))}, \
            [(idn.output[0], "C")]

    def r_QBDense(self, V):
        mm = V.first("MatMul")
        V.use(mm)
        x, bits = self._levels_r(V, mm.input[0])
        xi, tag = V.runtime(x)
        V.mode = "I"
        w = self._pm1(V, mm.input[1], "QBDense")
        wb = self.add_const(V.gp, 1, np.ascontiguousarray(w.T), "bit", mm.input[1])
        pa_ins, t = self._post_affine(V, mm.output[0])
        idn = V.cons(t, "Identity")
        V.use(idn)
        return [xi, wb] + pa_ins, {"act_bits": bits, "act_quant": "dorefa", "activation": "none"}, [(idn.output[0], "I")]

    def r_Reshape(self, V):
        nd = V.single("Reshape")
        xi, tag = V.runtime(nd.input[0])
        if tag != "I":
            raise V.err("Reshape of an NCHW view")
        V.mode = "I"
        return [xi], {"shape": [int(v) for v in np.asarray(V.const(nd.input[1])).reshape(-1)]}, [(nd.output[0], "I")]


def import_int8(model) -> tuple[Graph, dict]:
    """Group-mapped int8_full canonical file -> (IR graph, info) (qonnx_groups.import_groups dispatches here)."""
    from .qonnx_groups import CONST_ORDER_KEY, MAPPING
    imp = Int8Importer(model)
    g = imp.run()
    raw_u = next((p.value for p in model.metadata_props if p.key == UNUSED_KEY), None)
    for k in (json.loads(raw_u) if raw_u else []):
        used_onnx = {i for nd in model.graph.node for i in nd.input}
        if k not in imp.inits or k in used_onnx or k in g.constants:
            raise GroupImportError(UNUSED_KEY, f"{k!r} is not an unused initializer of the file", "model")
        a = np.asarray(imp.inits[k])
        dt = {np.dtype(np.int8): "i8", np.dtype(np.int32): "i32", np.dtype(np.float32): "f32"}.get(a.dtype)
        if dt is None:
            raise GroupImportError(UNUSED_KEY, f"{k!r}: element type {a.dtype}", "model")
        g.add_constant(k, a, dtype=dt)
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
            "passes": {"none": "group mapping: the groups are the deployed int8_full graph",
                       "bitino.const_order (applied)": const_order},
            "op_counts": g.op_counts(), "logits_onnx_tensor": imp.out_names[0], "onnx_outputs": imp.out_names,
            "onnx_tags": {t: tag for t, (_, tag) in imp.rt.items()}}
    return g, info


def build_model(spec: Spec, producer: str, opset: int = 17, ir_version: int = 10, extra_meta: dict | None = None):
    """Spec -> ONNX ModelProto in memory (qonnx_groups.assemble without the spec files on disk)."""
    import onnx
    from onnx import helper, numpy_helper
    inits = [numpy_helper.from_array(np.asarray(v), k) for k, v in spec.inits.items()]
    nodes = []
    for nd in spec.nodes:
        n = helper.make_node(nd["op"], nd["inputs"], nd["outputs"], name=nd["name"], domain=nd["domain"] or None,
                             **nd["attrs"])
        for k, v in nd["meta"].items():
            n.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
        nodes.append(n)
    ins = [helper.make_tensor_value_info(n, et, s) for n, et, s in spec.inputs]
    outs = [helper.make_tensor_value_info(n, et, s) for n, et, s in spec.outputs]
    meta = dict(spec.meta)
    meta.update(extra_meta or {})
    graph = helper.make_graph(nodes, meta.get("bitino.model_id", "model"), ins, outs, inits)
    uses_qonnx = any(nd["domain"] == QD for nd in spec.nodes)
    ops = [helper.make_opsetid("", opset)] + ([helper.make_opsetid(QD, 1)] if uses_qonnx else [])
    m = helper.make_model(graph, opset_imports=ops, producer_name=producer, producer_version="bitino-qonnx/1")
    m.ir_version = ir_version
    for k, v in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    return m
