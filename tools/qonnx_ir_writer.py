"""Generic writer: a legacy adapter's Bitino IR graph -> canonical QONNX spec of the Bitino profile
(docs/qonnx-profile.md, ``bitino-qonnx/1``); QONNX route step 4 (docs/qonnx-step4.md).

numpy only (imported by the step-4 front ends in their own environments: OpenVINO, NNCF, PyTorch, ...). The writer
walks the adapter's IR graph (the graph the legacy converter builds *before* its passes, or after the adapter-specific
rewrites it applies before the common ``passes.optimize``) and emits, per IR node, its canonical pattern of standard
ONNX + QONNX operators in NCHW, with every constant taken from the parse:

  BConv2D   binarizer [-> Pad(v)] -> Conv(., BipolarQuant(W_latent, 1) | ±1, pads 0, group g) [-> Mul(post_mul)]
            [-> Add(post_add)]; binarizer = BipolarQuant(x, 1) (ge), Where(Greater(x, t), 1, -1) (gt, threshold t),
            Where(GreaterOrEqual(x, t), 1, -1) (ge with a threshold), Sign(x) (sign_zero)
  BDense    binarizer -> Gemm(., W[O, K] ±1, transB 1) [-> Mul] [-> Add]
  ChannelAffine  Mul(x, s) [-> Add(., b)]       Conv2D / Dense  Conv / Gemm(transB 1)
  MaxPool2D / AvgPool2D / GlobalAvgPool / Concat / Add / Pad / Relu / PRelu / Clip / Softmax  as themselves
  Reshape   [Transpose(0, 2, 3, 1)] -> Reshape (an NHWC flatten; plain Reshape where NCHW and NHWC orders agree)

Constants: a FLOAT initializer holding the IR value, or a float64 carrier ``Cast(DOUBLE, FLOAT)`` (profile R7) when
the front end gives the float64 value the adapter rounded (asserted: its float32 rounding is the IR constant, bit for
bit). Binary weights are ``BipolarQuant(W_latent, 1)`` of the source's latent kernel when the front end gives one
(asserted: its sign, 0 -> +1, is the adapter's bit), else ±1 constants.

Naming metadata (§ 7): bitino.name = the IR node name, bitino.order = 2 x the IR position (2 x + 1 for a ReLU group),
bitino.consts = the IR name of every constant input, bitino.attrs = the IR attribute's JSON form where the importer's
differs (pad_value -1 vs -1.0, absent attributes).
"""
from __future__ import annotations

import json

import numpy as np

QD = "qonnx.custom_op.general"


class Spec:
    """ONNX graph under construction, without the onnx package: nodes as dicts, initializers as numpy arrays."""

    def __init__(self) -> None:
        self.nodes: list = []
        self.inits: dict = {}
        self.k = 0
        self.inputs: list = []
        self.outputs: list = []

    def init(self, name: str, arr) -> str:
        arr = np.asarray(arr)
        if name in self.inits:
            if not (self.inits[name].dtype == arr.dtype and np.array_equal(self.inits[name], arr)):
                raise AssertionError(f"initializer {name} defined twice with different values")
            return name
        self.inits[name] = np.array(arr, order="C", copy=True)
        return name

    def node(self, op: str, ins: list, attrs: dict | None = None, out: str | None = None, meta: dict | None = None,
             domain: str = "") -> str:
        self.k += 1
        out = out or f"t{self.k}:{op.lower()}"
        self.nodes.append({"op": op, "inputs": list(ins), "outputs": [out], "attrs": dict(attrs or {}),
                           "name": f"n{self.k}_{op}", "domain": domain, "meta": dict(meta or {})})
        return out


def nchw(shape) -> list:
    s = [int(d) for d in shape]
    return [s[0], s[3], s[1], s[2]] if len(s) == 4 else s


def onnx_pads(p) -> list:          # IR [t, b, l, r] -> ONNX [t, l, b, r]
    return [int(p[0]), int(p[2]), int(p[1]), int(p[3])]


def ir_shapes(g) -> dict:
    """IR tensor -> shape (one float32 run of the numpy executor on a zero input)."""
    from bitino import executor
    shp = list(g.inputs[0]["shape"])
    env = executor.run(g, {g.inputs[0]["name"]: np.zeros(shp, np.float32)}, keep_intermediates=True)
    out = {k: list(np.shape(v)) for k, v in env.items() if k not in g.constants}
    out[g.inputs[0]["name"]] = shp
    return out


def first_use_order(g) -> list:
    order, seen = [], set()
    for n in g.nodes:
        for i in n.inputs:
            if i and i in g.constants and i not in seen:
                seen.add(i)
                order.append(i)
    return order + [k for k in g.constants if k not in seen]


class IRWriter:
    """IR graph -> canonical ONNX spec. c64: {constant: float64 value} (carriers); latent: {bit constant: latent
    kernel OIHW / [O, K] float32 whose sign (0 -> +1) is the constant}; relu_groups: write a node's fused relu as its
    own group (the importer's fold pass puts it back)."""

    def __init__(self, g0, shapes: dict, c64: dict | None = None, latent: dict | None = None):
        self.g0 = g0
        self.shapes = shapes
        self.c64 = dict(c64 or {})
        self.latent = dict(latent or {})
        self.s = Spec()
        self.report: dict = {"carriers": 0, "latent_weights": 0, "pm1_weights": 0, "relu_groups": 0, "transposes": 0,
                             "explicit_binary_pads": 0, "thresholds": 0}
        s = self.s
        self.ONE = s.init("bitino/one", np.array(1.0, np.float32))
        self.ZERO = s.init("bitino/zero", np.array(0.0, np.float32))
        self.M1 = s.init("bitino/minus_one", np.array(-1.0, np.float32))
        self.used_c64: set = set()

    # ---------------------------------------------------------------- constants
    def cdata(self, name: str) -> np.ndarray:
        from bitino.ir import unpack_bits
        c = self.g0.constants[name]
        if c.dtype == "bit":
            return unpack_bits(c.data, c.shape[-1]).reshape(c.shape)
        return np.asarray(c.data)

    def const(self, cname: str, shape, carrier: bool = True) -> str:
        """ONNX tensor holding IR constant cname reshaped to `shape`: a float64 carrier when c64 has the constant,
        else a FLOAT initializer. carrier=False: always the FLOAT value (a binarizer threshold: the source compares
        the float32 tensor with the float32 threshold, which is what the IR stores)."""
        v32 = np.asarray(self.cdata(cname), np.float32)
        if cname not in self.c64 or not carrier:
            return self.s.init(cname, v32.reshape(shape))
        c64 = np.broadcast_to(np.asarray(self.c64[cname], np.float64), v32.shape)
        if not np.array_equal(c64.astype(np.float32).view(np.uint32), v32.view(np.uint32)):
            raise AssertionError(f"{cname}: float32 rounding of the float64 carrier differs from the adapter's constant")
        self.report["carriers"] += 1
        self.used_c64.add(cname)
        d = self.s.init(f"{cname}:f64", np.ascontiguousarray(c64.reshape(shape)))
        return self.s.node("Cast", [d], {"to": 1}, out=cname)

    def weights_pm1(self, cname: str, kind: str) -> str:
        """Binary weights of a BConv2D (OHWI bits -> OIHW) / BDense ([O, K]): BipolarQuant(W_latent, 1) or ±1."""
        bits = self.cdata(cname).astype(np.int8)
        b = np.transpose(bits, (0, 3, 1, 2)) if kind == "conv" else bits
        lat = self.latent.get(cname)
        if lat is not None:
            lat = np.asarray(lat, np.float32)
            if lat.shape != b.shape or not np.array_equal(np.where(lat >= 0, 1, -1), b):
                raise AssertionError(f"{cname}: the adapter's binary weights are not the sign (0 -> +1) of the latent kernel")
            self.report["latent_weights"] += 1
            w = self.s.init(f"{cname}:latent", np.ascontiguousarray(lat))
            return self.s.node("BipolarQuant", [w, self.ONE], domain=QD, out=f"{cname}:pm1")
        self.report["pm1_weights"] += 1
        return self.s.init(f"{cname}:pm1", b.astype(np.float32))

    # ---------------------------------------------------------------- walk
    def write(self, outputs=None):
        g0, s = self.g0, self.s
        names = {n.name for n in g0.nodes}
        inp = g0.inputs[0]
        if len(g0.inputs) != 1 or inp["name"] != "input":
            raise AssertionError("one graph input named 'input' expected")
        s.inputs = [("input", nchw(inp["shape"]))]
        for i, n in enumerate(g0.nodes):
            if f"{n.name}/relu" in names:
                raise AssertionError(f"{n.name}/relu is an IR node name")
            meta = {"bitino.name": n.name, "bitino.order": str(2 * i)}
            consts = {str(j): c for j, c in enumerate(n.inputs) if c and c in g0.constants}
            self.cur = (n, meta, consts, i)
            self.attrs_meta = None
            self.layouts_meta = None
            fn = getattr(self, f"op_{n.op}", None)
            if fn is None:
                raise AssertionError(f"{n.name}: IR op {n.op} is not written by this writer")
            fn(n)
        s.outputs = [(o, nchw(self.shapes[o])) for o in (outputs or g0.outputs)]
        unused = sorted(set(self.c64) - self.used_c64 - set(k for k in self.c64 if k not in g0.constants))
        self.report["c64_not_written"] = [k for k in unused if k in g0.constants]
        return s

    def emit(self, op, ins, attrs=None, out=None, anchor=False, domain=""):
        """A node of the current IR node's group; the anchor carries bitino.consts / bitino.attrs."""
        n, meta, consts, _ = self.cur
        m = dict(meta)
        if anchor:
            if consts:
                m["bitino.consts"] = json.dumps(consts, sort_keys=True)
            if self.attrs_meta:
                m["bitino.attrs"] = json.dumps(self.attrs_meta, sort_keys=True)
            if getattr(self, "layouts_meta", None):
                m["bitino.const_layouts"] = json.dumps(self.layouts_meta, sort_keys=True)
        return self.s.node(op, ins, attrs, out=out, meta=m, domain=domain)

    def relu(self, n, x: str) -> str:
        _, _, _, i = self.cur
        self.report["relu_groups"] += 1
        return self.s.node("Relu", [x], out=n.outputs[0], meta={"bitino.name": f"{n.name}/relu", "bitino.order": str(2 * i + 1)})

    def fused_act(self, n, x: str) -> str:
        act = n.attrs.get("activation", "none")
        if act in ("none", None):
            return x
        if act == "relu":
            return self.relu(n, x)
        raise AssertionError(f"{n.name}: fused activation {act}")

    def _out(self, n):
        return n.outputs[0] if n.attrs.get("activation", "none") in ("none", None) else None

    def _check_attrs(self, n, allowed):
        extra = set(n.attrs) - set(allowed)
        if extra:
            raise AssertionError(f"{n.name} ({n.op}): attributes {sorted(extra)} are not written by this writer")

    def _cs(self, x):
        shp = self.shapes[x]
        return [1, shp[-1], 1, 1] if len(shp) == 4 else [1, shp[-1]]

    # ---------------------------------------------------------------- operators
    def op_ChannelAffine(self, n):
        x, sc, sh = (n.inputs + ["", ""])[:3]
        if n.attrs:
            raise AssertionError(f"{n.name}: ChannelAffine attributes {n.attrs}")
        cs = self._cs(x)
        y = x
        if sc:
            y = self.emit("Mul", [y, self.const(sc, cs)], out=None if sh else n.outputs[0], anchor=True)
        if sh:
            y = self.emit("Add", [y, self.const(sh, cs)], out=n.outputs[0], anchor=not sc)
        if not sc and not sh:
            raise AssertionError(f"{n.name}: ChannelAffine without constants")

    def op_Conv2D(self, n):
        self._check_attrs(n, ("strides", "dilations", "pads", "groups", "activation", "pad_value"))
        if "pad_value" in n.attrs:
            pv = n.attrs["pad_value"]
            if float(pv) != 0.0:
                raise AssertionError(f"{n.name}: float Conv2D pad_value {pv}")
            self.attrs_meta = {"pad_value": pv} if not (isinstance(pv, float) and pv == 0.0) else None
        else:
            self.attrs_meta = {"pad_value": None}      # the IR node has no pad_value (the importer writes 0.0)
        x, w, b = (n.inputs + ["", ""])[:3]
        wd = self.cdata(w)                             # OHWI
        oihw = np.ascontiguousarray(np.transpose(wd, (0, 3, 1, 2))).astype(np.float32)
        ins = [x, self.s.init(w, oihw)] + ([self.const(b, [wd.shape[0]])] if b else [])
        y = self.emit("Conv", ins, {"kernel_shape": list(wd.shape[1:3]), "strides": list(n.attrs["strides"]),
                                    "dilations": list(n.attrs["dilations"]), "pads": onnx_pads(n.attrs["pads"]),
                                    "group": int(n.attrs["groups"])}, out=self._out(n), anchor=True)
        self.fused_act(n, y)

    def binarizer(self, n, x: str, mode: str, thr: str) -> str:
        if thr:
            self.report["thresholds"] += 1
            t = self.const(thr, self._cs(x), carrier=False)
            if mode == "gt":
                return self.emit("Where", [self.emit("Greater", [x, t]), self.ONE, self.M1])
            if mode == "ge":
                return self.emit("Where", [self.emit("GreaterOrEqual", [x, t]), self.ONE, self.M1])
            raise AssertionError(f"{n.name}: input_binarize {mode} with a threshold")
        if mode == "ge":
            return self.emit("BipolarQuant", [x, self.ONE], domain=QD)
        if mode == "gt":
            return self.emit("Where", [self.emit("Greater", [x, self.ZERO]), self.ONE, self.M1])
        if mode == "sign_zero":
            return self.emit("Sign", [x])
        raise AssertionError(f"{n.name}: input_binarize {mode}")

    def op_BConv2D(self, n):
        self._check_attrs(n, ("strides", "dilations", "pads", "groups", "activation", "pad_value", "input_binarize", "output"))
        x, w, pm, pa, thr = (n.inputs + [""] * 5)[:5]
        if n.attrs.get("output", "f32") != "f32":
            raise AssertionError(f"{n.name}: bit output (a pass's rewrite) is not written by this writer")
        pv = n.attrs.get("pad_value", 0.0)
        self.attrs_meta = {"pad_value": pv}            # the IR's JSON form of the value the Pad (or its absence) states
        xb = self.binarizer(n, x, n.attrs["input_binarize"], thr)
        pads = list(n.attrs["pads"])
        if any(pads) or float(pv) != 0.0:
            t, b_, l, r = pads
            self.report["explicit_binary_pads"] += 1
            xb = self.emit("Pad", [xb, self.s.init(f"{n.name}/pads:onnx", np.array([0, 0, t, l, 0, 0, b_, r], np.int64)),
                                   self.s.init(f"bitino/pad_value_{float(pv):g}", np.array(float(pv), np.float32))],
                           {"mode": "constant"})
        else:
            self.attrs_meta = {"pad_value": pv} if not (isinstance(pv, int) and pv == 0) else None
        o, kh, kw, _ = self.cdata(w).shape
        y = self.emit("Conv", [xb, self.weights_pm1(w, "conv")],
                      {"kernel_shape": [kh, kw], "strides": list(n.attrs["strides"]), "dilations": list(n.attrs["dilations"]),
                       "pads": [0, 0, 0, 0], "group": int(n.attrs["groups"])},
                      out=n.outputs[0] if not pm and not pa and self._out(n) else None, anchor=True)
        y = self._epilogue(n, y, pm, pa, [1, o, 1, 1])
        self.fused_act(n, y)

    def _epilogue(self, n, y, pm, pa, cs):
        out = self._out(n)
        if pm:
            y = self.emit("Mul", [y, self.const(pm, cs)], out=out if not pa else None)
        if pa:
            y = self.emit("Add", [y, self.const(pa, cs)], out=out)
        return y

    def op_BDense(self, n):
        self._check_attrs(n, ("activation", "input_binarize"))
        x, w, pm, pa, thr = (n.inputs + [""] * 5)[:5]
        xb = self.binarizer(n, x, n.attrs["input_binarize"], thr)
        o = self.cdata(w).shape[0]
        lay = self.g0.constants[w].layout
        self.layouts_meta = {"1": lay} if lay != "OI_bit" else None   # the importer tags BDense kernels OI_bit
        y = self.emit("Gemm", [xb, self.weights_pm1(w, "dense")], {"transB": 1},
                      out=n.outputs[0] if not pm and not pa and self._out(n) else None, anchor=True)
        self.fused_act(n, self._epilogue(n, y, pm, pa, [1, o]))

    def op_Dense(self, n):
        self._check_attrs(n, ("activation",))
        x, w, b = (n.inputs + ["", ""])[:3]
        wd = self.cdata(w)
        ins = [x, self.s.init(w, wd.astype(np.float32))] + ([self.const(b, [wd.shape[0]])] if b else [])
        y = self.emit("Gemm", ins, {"transB": 1}, out=self._out(n), anchor=True)
        self.fused_act(n, y)

    def _pool(self, n, op, extra=None):
        self._check_attrs(n, ("kernel", "strides", "pads", "count_include_pad"))
        at = {"kernel_shape": list(n.attrs["kernel"]), "strides": list(n.attrs["strides"]), "pads": onnx_pads(n.attrs["pads"])}
        at.update(extra or {})
        self.emit(op, [n.inputs[0]], at, out=n.outputs[0], anchor=True)

    def op_MaxPool2D(self, n):
        if "count_include_pad" in n.attrs:
            raise AssertionError(n.name)
        self._pool(n, "MaxPool")

    def op_AvgPool2D(self, n):
        if "count_include_pad" not in n.attrs:
            self.attrs_meta = {"count_include_pad": None}    # absent in the IR node (= the default 0)
        self._pool(n, "AveragePool", {"count_include_pad": int(n.attrs.get("count_include_pad", 0))})

    def op_GlobalAvgPool(self, n):
        if int(n.attrs.get("keepdims", 0)):
            self.emit("GlobalAveragePool", [n.inputs[0]], out=n.outputs[0], anchor=True)
        else:
            self.emit("Flatten", [self.emit("GlobalAveragePool", [n.inputs[0]], anchor=True)], {"axis": 1}, out=n.outputs[0])

    def op_Reshape(self, n):
        x = n.inputs[0]
        src, dst = list(self.shapes[x]), [int(d) for d in n.attrs["shape"]]
        if list(self.shapes[n.outputs[0]]) != dst:
            raise AssertionError(f"{n.name}: reshape target {dst} vs executor shape {self.shapes[n.outputs[0]]}")
        kept = lambda s: len(s) != 4 or s[3] == 1 or s[1] * s[2] == 1   # noqa: E731  NHWC order = NCHW order
        if len(src) == 4:
            if not kept(dst):
                raise AssertionError(f"{n.name}: reshape {src} -> {dst} (target does not keep the element order)")
            tgt = nchw(dst)
            prod = next((p for p in self.g0.nodes if x in p.outputs), None)
            gap_keep = prod is not None and prod.op == "GlobalAvgPool" and int(prod.attrs.get("keepdims", 0))
            # a GlobalAvgPool (keepdims 1) + Reshape pair of the IR (OpenVINO) is written as GlobalAveragePool ->
            # Transpose -> Reshape: a plain Reshape / Flatten right after the pool is the importer's GlobalAvgPool
            # keepdims 0 (the PyTorch flatten), one IR node instead of two
            if kept(src) and nchw(src) != tgt and len(dst) == 2 and not gap_keep:
                y = x
            else:
                self.report["transposes"] += 1
                y = self.emit("Transpose", [x], {"perm": [0, 2, 3, 1]})
            self.emit("Reshape", [y, self.s.init(f"{n.name}/shape:onnx", np.array(tgt, np.int64))], out=n.outputs[0], anchor=True)
        elif len(src) == 2 and len(dst) == 2 and src != dst:
            self.emit("Reshape", [x, self.s.init(f"{n.name}/shape:onnx", np.array(dst, np.int64))], out=n.outputs[0], anchor=True)
        else:
            raise AssertionError(f"{n.name}: reshape {src} -> {dst}")

    def op_Concat(self, n):
        if int(n.attrs.get("axis", -1)) not in (-1, len(self.shapes[n.inputs[0]]) - 1):
            raise AssertionError(f"{n.name}: concat axis {n.attrs}")
        self.emit("Concat", list(n.inputs), {"axis": 1}, out=n.outputs[0], anchor=True)

    def op_Add(self, n):
        self._check_attrs(n, ("activation",))
        if self.shapes[n.inputs[0]] != self.shapes[n.inputs[1]]:
            raise AssertionError(f"{n.name}: Add of different shapes")
        if "activation" in n.attrs and n.attrs["activation"] == "none":
            self.attrs_meta = {"activation": "none"}       # present in the IR node with its default value
        self.emit("Add", list(n.inputs), out=self._out(n), anchor=True)
        if n.attrs.get("activation", "none") not in ("none", None):
            self.fused_act(n, self.s.nodes[-1]["outputs"][0])

    def op_Pad(self, n):
        p = [int(v) for v in n.attrs["pads"]]
        if len(p) != 8 or p[0] or p[1]:
            raise AssertionError(f"{n.name}: pads {p}")
        _, _, t, b, l, r, c0, c1 = p
        val = float(n.attrs.get("value", 0.0))
        self.emit("Pad", [n.inputs[0], self.s.init(f"{n.name}/pads:onnx", np.array([0, c0, t, l, 0, c1, b, r], np.int64)),
                          self.s.init(f"bitino/pad_value_{val:g}", np.array(val, np.float32))],
                  {"mode": "constant"}, out=n.outputs[0], anchor=True)

    def op_Activation(self, n):
        t = n.attrs["type"]
        x = n.inputs[0]
        if t == "relu":
            self.emit("Relu", [x], out=n.outputs[0], anchor=True)
        elif t == "hardtanh":
            lo, hi = float(n.attrs["min"]), float(n.attrs["max"])
            if float(np.float32(lo)) != lo or float(np.float32(hi)) != hi:
                raise AssertionError(f"{n.name}: hardtanh bounds are not float32 values")
            self.emit("Clip", [x, self.s.init(f"bitino/clip_{lo:g}", np.array(lo, np.float32)),
                               self.s.init(f"bitino/clip_{hi:g}", np.array(hi, np.float32))], out=n.outputs[0], anchor=True)
        elif t == "prelu":
            shp = self.shapes[x]
            self.emit("PRelu", [x, self.const(n.inputs[1], [shp[-1], 1, 1] if len(shp) == 4 else [shp[-1]])],
                      out=n.outputs[0], anchor=True)
        else:
            raise AssertionError(f"{n.name}: activation {t}")

    def op_Softmax(self, n):
        if len(self.shapes[n.inputs[0]]) != 2:
            raise AssertionError(n.name)
        self.emit("Softmax", [n.inputs[0]], {"axis": -1}, out=n.outputs[0], anchor=True)


# ================================================================================================ FINN / BNN-PYNQ / Brevitas
class LayerWriter:
    """The FINN-style layer specs (bitino.adapters.finn_bnn InputSpec + [Layer]: exact ±1 weights, float64 binarizer
    affines) -> canonical ONNX spec in the source's own form: [Mul(a) -> Add(b) -> IntQuant] input quantizer or an
    input binarizer, then per layer Conv / Gemm (±1 constants, VALID) -> [MaxPool before BN] -> Mul(scale c64) ->
    Add(shift c64) -> BipolarQuant(., 1) -> [MaxPool after the sign]; the last layer ends at its affine. The importer
    (--affine fold) folds every BN + sign into an exact integer threshold, as the legacy converters do.

    gf: the legacy fused IR graph (the published structure), read only for the naming metadata: its nodes in order
    are the roles [input ChannelAffine, input FakeQuant], then per layer [flatten Reshape], the layer, [pool]."""

    def __init__(self, inp, layers, gf, input_affine_sign: tuple | None = None):
        self.inp, self.layers, self.gf = inp, layers, gf
        self.affine_sign = input_affine_sign        # (a, b): the input sign(a x + b) of the Brevitas FC (threshold 0.5)
        self.s = Spec()
        self.ONE = self.s.init("bitino/one", np.array(1.0, np.float32))
        self.M1 = self.s.init("bitino/minus_one", np.array(-1.0, np.float32))
        self.report = {"carriers": 0, "pm1_weights": 0, "layers": len(layers)}
        self.order = 0

    def _c64(self, name, v64, shape):
        self.report["carriers"] += 1
        d = self.s.init(f"{name}:f64", np.ascontiguousarray(np.broadcast_to(np.asarray(v64, np.float64).reshape(-1), (int(np.prod(shape)),)).reshape(shape)))
        return self.s.node("Cast", [d], {"to": 1}, out=f"{name}:c64")

    @staticmethod
    def _meta(node, order, consts=None, extra=None):
        m = {"bitino.name": node.name, "bitino.order": str(order)}
        if consts:
            m["bitino.consts"] = json.dumps(consts, sort_keys=True)
        if extra:
            m.update(extra)
        return m

    def _next_order(self):
        self.order += 1
        return 2 * (self.order - 1)

    def write(self):
        s, inp, gf = self.s, self.inp, self.gf
        nodes = list(gf.nodes)
        pos = 0

        def take(op):
            nonlocal pos
            n = nodes[pos]
            if n.op not in (op if isinstance(op, tuple) else (op,)):
                raise AssertionError(f"role {op}: the legacy graph has {n.op} {n.name} at position {pos}")
            pos += 1
            return n

        def consts_of(n):
            return {str(j): c for j, c in enumerate(n.inputs) if c and c in gf.constants}

        h, w, c = inp.shape[1], inp.shape[2], inp.shape[3]
        s.inputs = [("input", [1, c, h, w])]
        x = "input"
        shape = [1, c, h, w]
        in_bin = None                       # pending input binarizer: ("thr", t) | ("pm1",) | ("affine", a, b)
        if inp.kind == "int8":
            ca, fq = take("ChannelAffine"), take("FakeQuant")
            a, b = inp.affine
            mt = self._meta(fq, self._next_order(), {**{f"{ca.name}#{k}": v for k, v in consts_of(ca).items()}, **consts_of(fq)},
                            {"bitino.extra_names": json.dumps([ca.name])})
            y = s.node("Mul", [x, s.init("input/a:onnx", np.array(a, np.float32))], meta=dict(mt))
            y = s.node("Add", [y, s.init("input/b:onnx", np.array(b, np.float32))], meta=dict(mt))
            q = inp.q
            bw = int(round(np.log2(q["qmax"] - q["qmin"] + 1)))
            if q["qmin"] != -(2 ** (bw - 1)) or q["qmax"] != 2 ** (bw - 1) - 1:
                raise AssertionError(f"input quantizer [{q['qmin']}, {q['qmax']}] is not a signed {bw}-bit grid")
            x = s.node("IntQuant", [y, s.init("input_quant/scale:onnx", np.array(q["scale"], np.float32)),
                                    s.init("bitino/zero", np.array(0.0, np.float32)), s.init(f"bitino/bits_{bw}", np.array(float(bw), np.float32))],
                       {"signed": 1, "narrow": 0, "rounding_mode": "ROUND"}, meta=dict(mt), domain=QD)
        elif inp.kind == "binary_threshold":
            in_bin = ("thr", float(inp.threshold))
        elif inp.kind == "pm1":
            in_bin = ("pm1",)
        elif inp.kind == "affine_sign":
            in_bin = ("affine",) + tuple(self.affine_sign)
        else:
            raise AssertionError(inp.kind)
        pending = None                      # BN of a pool-before-BN layer: emitted with its sign in the next layer's group
        for li, l in enumerate(self.layers):
            if (l.kind == "dense" and len(shape) == 4) or (li == 0 and nodes[pos].op == "Reshape"):
                fl = take("Reshape")
                k = int(np.prod(shape[1:]))
                if not (shape[2] * shape[3] == 1 or shape[1] == 1):
                    raise AssertionError(f"{l.name}: flatten at {shape} (NCHW and NHWC orders differ)")
                x = s.node("Reshape", [x, s.init(f"{fl.name}/shape:onnx", np.array([1, k], np.int64))], meta=self._meta(fl, self._next_order()))
                shape = [1, k]
            node = take(("BConv2D", "BDense", "Conv2D"))
            mt = self._meta(node, self._next_order(), consts_of(node))
            ch = shape[1]
            cin = [1, ch, 1, 1] if len(shape) == 4 else [1, ch]
            if pending is not None:          # the previous layer's BN + sign (pool before BN): this layer's threshold
                sc, sh = pending
                y = s.node("Mul", [x, self._c64(f"{node.name}/in_scale", sc, cin)], meta=dict(mt))
                y = s.node("Add", [y, self._c64(f"{node.name}/in_shift", sh, cin)], meta=dict(mt))
                x = s.node("BipolarQuant", [y, self.ONE], meta=dict(mt), domain=QD)
                pending = None
            if li == 0 and in_bin is not None:
                if in_bin[0] == "thr":
                    t = s.init(f"{node.name}/input_threshold:onnx", np.array(in_bin[1], np.float32))
                    x = s.node("Where", [s.node("GreaterOrEqual", [x, t], meta=dict(mt)), self.ONE, self.M1], meta=dict(mt))
                elif in_bin[0] == "pm1":
                    x = s.node("BipolarQuant", [x, self.ONE], meta=dict(mt), domain=QD)
                else:
                    y = s.node("Mul", [x, s.init("input/a:onnx", np.array(in_bin[1], np.float32))], meta=dict(mt))
                    y = s.node("Add", [y, s.init("input/b:onnx", np.array(in_bin[2], np.float32))], meta=dict(mt))
                    x = s.node("BipolarQuant", [y, self.ONE], meta=dict(mt), domain=QD)
            wgt = np.asarray(l.w, np.float32)
            self.report["pm1_weights"] += 1
            o = wgt.shape[0]
            if l.kind == "conv":
                y = s.node("Conv", [x, s.init(f"{l.name}/weight:pm1", wgt)],
                           {"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "group": 1}, meta=dict(mt))
                shape = [1, o, shape[2] - 2, shape[3] - 2]
                cs = [1, o, 1, 1]
            else:
                y = s.node("Gemm", [x, s.init(f"{l.name}/weight:pm1", wgt)], {"transB": 1}, meta=dict(mt))
                shape = [1, o]
                cs = [1, o]
            if l.final:
                y = s.node("Mul", [y, self._c64(f"{l.name}/scale", l.scale, cs)], meta=dict(mt))
                x = s.node("Add", [y, self._c64(f"{l.name}/shift", l.shift, cs)], out=node.outputs[0], meta=dict(mt))
                continue
            if l.pool == "before_bn":
                pn = take("MaxPool2D")
                x = s.node("MaxPool", [y], {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, meta=self._meta(pn, self._next_order()))
                shape = [1, shape[1], shape[2] // 2, shape[3] // 2]
                pending = (l.scale, l.shift)
                continue
            y = s.node("Mul", [y, self._c64(f"{l.name}/scale", l.scale, cs)], meta=dict(mt))
            y = s.node("Add", [y, self._c64(f"{l.name}/shift", l.shift, cs)], meta=dict(mt))
            x = s.node("BipolarQuant", [y, self.ONE], meta=dict(mt), domain=QD)
            if l.pool == "after_sign":
                pn = take("MaxPool2D")
                x = s.node("MaxPool", [x], {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, meta=self._meta(pn, self._next_order()))
                shape = [1, shape[1], shape[2] // 2, shape[3] // 2]
        if pos != len(nodes):
            raise AssertionError(f"legacy graph nodes not covered: {[n.name for n in nodes[pos:]]}")
        s.outputs = [(gf.outputs[0], shape)]
        return s
