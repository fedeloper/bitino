"""Adapter: OpenVINO IR (v10/v11 .xml + .bin) -> Bitino IR.

Parses the IR with xml.etree + numpy only (no OpenVINO dependency), so conversion
is independent of the runtime that serves as the trusted reference.

Scope: the Open Model Zoo binary classifiers (resnet50-binary-0001,
resnet18-xnor-binary-onnx-0001; FP32-INT1 and FP16-INT1). Every IR layer type is
mapped explicitly; anything else raises NotImplementedError.

Semantics captured (OpenVINO opset1 specs; the OpenVINO CPU plugin is the trusted
reference and every point below was checked against it numerically, see
the research notes (not published)):

* FakeQuantize(levels=2, input_low == input_high = t per channel):
  x <= t -> output_low, x > t -> output_high. With {output_low, output_high} = {0, 1}
  this is a binarizer, consumed by BinaryConvolution which reads 0 as -1 and 1 as +1.
  Bitino: BConv2D threshold input t with input_binarize="gt" (x > t -> +1).
  Channels with output_low=1, output_high=0 would be inverted binarizers; they are folded
  into the weights (w[:, :, :, c] *= -1), which is only exact without spatial padding
  or with pad_value 0, so other cases raise. (The OMZ classifiers have none.)
* BinaryConvolution(mode="xnor-popcount"): out = sum over taps of b * w with b, w in
  {-1,+1}; u1 weight bit 1 = +1. pad_value=-1: padded taps read b = -1 (bit 0, i.e. 0 in
  the {0,1} activation domain), so they contribute -w = Bitino pad_value=-1. Other
  pad_values with nonzero padding raise (the OMZ classifiers only use -1, the only value
  verified against the CPU plugin).
* u1 weight constants are read LSB-first: element i is bit (i % 8) of byte i // 8.
  This is what the CPU plugin's BinaryConvolution does with the IR bytes. Note that
  ov.op.Constant(u1) packs MSB-first (element 0 -> bit 7), but decoding the OMZ weights
  MSB-first gives logits off by 15-38 and wrong top-1 versus the CPU plugin, while
  LSB-first matches it to <= 3.1e-5 (the research notes (not published)).
  The pad semantics above were checked the same way: "-1 per padded tap" is off by 7.7.
  BinaryConvolution input channel counts that are not a multiple of 8 raise: on synthetic
  IRs they disagree with the CPU plugin (cause not determined; the OMZ models are unaffected).
* Elementwise Multiply/Add/Subtract with a constant that broadcasts per channel
  become ChannelAffine; Add of two activations becomes Add.
* Layout: OpenVINO is NCHW; Bitino is NHWC. The graph input becomes [N, H, W, C]
  with the IR's channel order (BGR for the OMZ classifiers).
"""
from __future__ import annotations

import collections
import xml.etree.ElementTree as ET

import numpy as np

from ..ir import Graph

_NP_TYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i64": np.int64, "i32": np.int32,
             "u8": np.uint8, "i8": np.int8}


def _ints(s: str | None) -> list[int]:
    if s is None or s.strip() == "":
        return []
    return [int(v) for v in s.replace(" ", "").split(",")]


class IRLayer:
    def __init__(self, el: ET.Element):
        self.id = el.get("id")
        self.name = el.get("name")
        self.type = el.get("type")
        self.version = el.get("version")
        d = el.find("data")
        self.attrs = dict(d.attrib) if d is not None else {}
        self.in_ports: dict[int, list[int]] = {}
        self.out_ports: dict[int, list[int]] = {}
        for tag, dst in (("input", self.in_ports), ("output", self.out_ports)):
            sec = el.find(tag)
            if sec is None:
                continue
            for p in sec.findall("port"):
                dst[int(p.get("id"))] = [int(x.text) for x in p.findall("dim")]

    def __repr__(self) -> str:
        return f"<{self.type} #{self.id} {self.name}>"


class IRModel:
    """Parsed OpenVINO IR. Tensors are addressed as (layer_id, out_port)."""

    def __init__(self, xml_path: str, bin_path: str, u1_order: str = "lsb"):
        root = ET.parse(xml_path).getroot()
        self.version = int(root.get("version", "0"))
        if self.version not in (10, 11):
            raise NotImplementedError(f"IR version {self.version} not supported (need 10 or 11)")
        self.layers: dict[str, IRLayer] = {}
        self.order: list[str] = []
        for el in root.find("layers").findall("layer"):
            layer = IRLayer(el)
            self.layers[layer.id] = layer
            self.order.append(layer.id)
        # (to_layer, to_port) -> (from_layer, from_port)
        self.src: dict[tuple[str, int], tuple[str, int]] = {}
        self.consumers: dict[tuple[str, int], list[tuple[str, int]]] = collections.defaultdict(list)
        for e in root.find("edges").findall("edge"):
            f = (e.get("from-layer"), int(e.get("from-port")))
            t = (e.get("to-layer"), int(e.get("to-port")))
            self.src[t] = f
            self.consumers[f].append(t)
        self.blob = np.fromfile(bin_path, dtype=np.uint8)
        if u1_order not in ("msb", "lsb"):
            raise ValueError(u1_order)
        self.u1_order = u1_order
        self.topo = self._toposort()

    # ------------------------------------------------------------------ graph
    def inputs_of(self, lid: str) -> list[tuple[str, int]]:
        layer = self.layers[lid]
        return [self.src[(lid, p)] for p in sorted(layer.in_ports)]

    def _toposort(self) -> list[str]:
        indeg = {lid: len(self.layers[lid].in_ports) for lid in self.order}
        ready = [lid for lid in self.order if indeg[lid] == 0]
        out = []
        pos = {lid: i for i, lid in enumerate(self.order)}
        while ready:
            ready.sort(key=lambda x: pos[x])
            lid = ready.pop(0)
            out.append(lid)
            for p in self.layers[lid].out_ports:
                for (t, _) in self.consumers.get((lid, p), []):
                    indeg[t] -= 1
                    if indeg[t] == 0:
                        ready.append(t)
        if len(out) != len(self.order):
            raise ValueError("IR graph has a cycle or dangling inputs")
        return out

    def op_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for layer in self.layers.values():
            c[layer.type] = c.get(layer.type, 0) + 1
        return c

    # -------------------------------------------------------------- constants
    def is_const(self, t: tuple[str, int]) -> bool:
        layer = self.layers[t[0]]
        if layer.type == "Const":
            return True
        if layer.type == "Convert":
            return self.is_const(self.inputs_of(layer.id)[0])
        return False

    def const_value(self, t: tuple[str, int]) -> np.ndarray:
        """Value of a Const (optionally behind Convert, as in FP16-INT1 IRs).
        u1 constants are returned as uint8 {0, 1} arrays of the logical shape."""
        layer = self.layers[t[0]]
        if layer.type == "Convert":
            dst = layer.attrs["destination_type"]
            v = self.const_value(self.inputs_of(layer.id)[0])
            if dst not in _NP_TYPES:
                raise NotImplementedError(f"Convert to {dst}")
            return v.astype(_NP_TYPES[dst])  # f16 -> f32 is exact
        if layer.type != "Const":
            raise ValueError(f"{layer} is not a constant")
        a = layer.attrs
        et, off, size = a["element_type"], int(a["offset"]), int(a["size"])
        shape = _ints(a.get("shape", ""))
        raw = self.blob[off: off + size]
        if raw.size != size:
            raise ValueError(f"{layer}: constant out of range of .bin")
        n = int(np.prod(shape)) if shape else 1
        if et == "u1":
            bits = np.unpackbits(raw, bitorder="little" if self.u1_order == "lsb" else "big")
            if bits.size < n:
                raise ValueError(f"{layer}: u1 blob too small")
            return bits[:n].reshape(shape).astype(np.uint8)
        if et not in _NP_TYPES:
            raise NotImplementedError(f"{layer}: element type {et}")
        v = np.frombuffer(raw.tobytes(), dtype=_NP_TYPES[et])
        if v.size != n:
            raise ValueError(f"{layer}: {v.size} elements, shape {shape}")
        return v.reshape(shape).copy()

    def shape_of(self, t: tuple[str, int]) -> list[int]:
        return self.layers[t[0]].out_ports[t[1]]


# ============================================================ independent IR evaluation
def _conv_nchw(x: np.ndarray, w: np.ndarray, strides, dilations, pads_hw, pad_value: float) -> np.ndarray:
    """Direct NCHW convolution: x [N,C,H,W], w [O,C,KH,KW], pads_hw = (top, bottom, left, right).
    Accumulates with one matrix product in x.dtype (float64 -> BLAS dgemm)."""
    n, c, h, wd = x.shape
    o, _, kh, kw = w.shape
    (sh, sw), (dh, dw) = strides, dilations
    pt, pb, pl, pr = pads_hw
    xp = np.full((n, c, h + pt + pb, wd + pl + pr), pad_value, dtype=x.dtype)
    xp[:, :, pt:pt + h, pl:pl + wd] = x
    oh = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    ow = (wd + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    cols = np.empty((n, c, kh, kw, oh, ow), dtype=x.dtype)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, ky, kx] = xp[:, :, ky * dh: ky * dh + sh * (oh - 1) + 1: sh, kx * dw: kx * dw + sw * (ow - 1) + 1: sw]
    out = w.reshape(o, -1).astype(x.dtype) @ cols.reshape(n, c * kh * kw, oh * ow)
    return out.reshape(n, o, oh, ow)


def _pool_nchw(x: np.ndarray, kernel, strides, pads_hw, kind: str, exclude_pad: bool) -> np.ndarray:
    n, c, h, w = x.shape
    (kh, kw), (sh, sw), (pt, pb, pl, pr) = kernel, strides, pads_hw
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    out = np.empty((n, c, oh, ow), x.dtype)
    for oy in range(oh):
        for ox in range(ow):
            y0, x0 = oy * sh - pt, ox * sw - pl
            ys, ye, xs, xe = max(y0, 0), min(y0 + kh, h), max(x0, 0), min(x0 + kw, w)
            win = x[:, :, ys:ye, xs:xe]
            if kind == "max":
                out[:, :, oy, ox] = win.max(axis=(2, 3))
            else:
                cnt = (ye - ys) * (xe - xs) if exclude_pad else kh * kw
                out[:, :, oy, ox] = win.sum(axis=(2, 3)) / cnt
    return out


def evaluate(ir: IRModel, x_nchw: np.ndarray, dtype=np.float64, keep: bool = False) -> dict:
    """Evaluates the OpenVINO IR graph itself with numpy, following the opset-1 operation
    specifications, in NCHW and in `dtype` (float64: the high-precision reference; OpenVINO
    itself only runs float32). Independent of the Bitino adapter and executor: it reads the
    same IR layers but implements each IR op directly.

    BinaryConvolution (xnor-popcount): the input must be {0,1} (0 read as -1); out =
    sum over taps of b*w with b, w in {-1,+1}; padded taps read b = pad_value (the CPU plugin
    semantics; pad_value=-1 is zero padding in the {0,1} domain). Integer sums are exact in
    float64 (and in float32 below 2^24).
    Returns {"outputs": [arrays], "env": {(layer_id, port): array} if keep}."""
    env: dict[tuple[str, int], np.ndarray] = {}
    outs = []
    for lid in ir.topo:
        layer = ir.layers[lid]
        typ, a = layer.type, layer.attrs
        port = min(layer.out_ports) if layer.out_ports else None
        if typ in ("Const", "Convert"):
            continue
        ins = ir.inputs_of(lid)

        def val(t):
            return ir.const_value(t).astype(dtype) if ir.is_const(t) else env[t]

        if typ == "Parameter":
            y = np.asarray(x_nchw, dtype=dtype).reshape(_ints(a["shape"]))
        elif typ in ("Add", "Subtract", "Multiply"):
            if a.get("auto_broadcast", "numpy") != "numpy":
                raise NotImplementedError(f"{layer}: auto_broadcast")
            p, q = val(ins[0]), val(ins[1])
            y = p + q if typ == "Add" else (p - q if typ == "Subtract" else p * q)
        elif typ == "ReLU":
            y = np.maximum(val(ins[0]), 0)
        elif typ == "Convolution":
            y = _conv_nchw(val(ins[0]), val(ins[1]), _ints(a["strides"]), _ints(a["dilations"]), _pads(layer), 0.0)
        elif typ == "BinaryConvolution":
            if a.get("mode") != "xnor-popcount":
                raise NotImplementedError(f"{layer}: mode {a.get('mode')}")
            xin = val(ins[0])
            if not np.all((xin == 0) | (xin == 1)):
                raise ValueError(f"{layer}: BinaryConvolution input is not in {{0,1}}")
            w = np.where(ir.const_value(ins[1]) == 1, 1.0, -1.0).astype(dtype)
            y = _conv_nchw(2 * xin - 1, w, _ints(a["strides"]), _ints(a["dilations"]), _pads(layer), float(a.get("pad_value", "0")))
        elif typ == "FakeQuantize":
            xin, il, ih, ol, oh = (val(t) for t in ins)
            lv = int(a["levels"])
            lo, hi = np.minimum(il, ih), np.maximum(il, ih)
            with np.errstate(divide="ignore", invalid="ignore"):
                mid = np.round((xin - il) / (ih - il) * (lv - 1)) / (lv - 1) * (oh - ol) + ol
            y = np.where(xin <= lo, ol, np.where(xin > hi, oh, mid)).astype(dtype)
        elif typ in ("MaxPool", "AvgPool"):
            if typ == "MaxPool" and any(v != 1 for v in _ints(a.get("dilations", "1,1"))):
                raise NotImplementedError(f"{layer}: dilated MaxPool")
            if a.get("rounding_type", "floor") != "floor":
                raise NotImplementedError(f"{layer}: rounding_type {a.get('rounding_type')}")
            y = _pool_nchw(val(ins[0]), _ints(a["kernel"]), _ints(a["strides"]), _pads(layer),
                           "max" if typ == "MaxPool" else "avg", a.get("exclude-pad", "false") == "true")
        elif typ == "ReduceMean":
            axes = tuple(int(v) for v in ir.const_value(ins[1]).reshape(-1))
            y = val(ins[0]).mean(axis=axes, keepdims=a.get("keep_dims", "false") == "true")
        elif typ == "Reshape":
            xin = val(ins[0])
            tgt = [int(v) for v in ir.const_value(ins[1]).reshape(-1)]
            if a.get("special_zero", "false") == "true":
                tgt = [xin.shape[i] if v == 0 else v for i, v in enumerate(tgt)]
            y = xin.reshape(tgt)
        elif typ == "MatMul":
            p, q = val(ins[0]), val(ins[1])
            if a.get("transpose_a", "false") == "true":
                p = np.swapaxes(p, -1, -2)
            if a.get("transpose_b", "false") == "true":
                q = np.swapaxes(q, -1, -2)
            y = p @ q
        elif typ == "SoftMax":
            z = val(ins[0])
            ax = int(a.get("axis", "1"))
            z = np.exp(z - z.max(axis=ax, keepdims=True))
            y = z / z.sum(axis=ax, keepdims=True)
        elif typ == "Result":
            outs.append(env[ins[0]])
            continue
        else:
            raise NotImplementedError(f"IR evaluation: unsupported layer type {typ} ({layer})")
        y = np.asarray(y, dtype=dtype)
        want = layer.out_ports[port]
        if list(y.shape) != want:
            raise ValueError(f"{layer}: evaluated shape {list(y.shape)} != IR shape {want}")
        env[(lid, port)] = y
    return {"outputs": outs, "env": env if keep else None}


# ====================================================================== adapter
def _per_channel(v: np.ndarray, c: int, rank: int, what: str) -> np.ndarray:
    """Constant that broadcasts along the channel axis (axis 1 for NCHW / last for 2-D)."""
    v = np.asarray(v)
    if v.size == 1:
        return np.full(c, v.reshape(-1)[0], dtype=np.float64)
    if rank == 4 and v.ndim == 4 and v.shape[0] == 1 and v.shape[1] == c and v.shape[2] == 1 and v.shape[3] == 1:
        return v.reshape(c).astype(np.float64)
    if rank == 2 and v.ndim == 2 and v.shape == (1, c):
        return v.reshape(c).astype(np.float64)
    if v.ndim == 1 and v.shape[0] == c and rank == 2:
        return v.astype(np.float64)
    raise NotImplementedError(f"{what}: constant of shape {list(v.shape)} does not broadcast per channel (C={c}, rank {rank})")


def _pads(layer: IRLayer) -> list[int]:
    ap = layer.attrs.get("auto_pad", "explicit")
    if ap not in ("explicit", "notset", ""):
        # The OMZ classifiers use explicit pads everywhere; resolving same_upper/lower
        # needs the input size and is not required here.
        raise NotImplementedError(f"{layer}: auto_pad={ap}")
    pb, pe = _ints(layer.attrs.get("pads_begin", "0,0")), _ints(layer.attrs.get("pads_end", "0,0"))
    if len(pb) != 2 or len(pe) != 2:
        raise NotImplementedError(f"{layer}: only 2-D pads supported")
    return [pb[0], pe[0], pb[1], pe[1]]


def convert(ir: IRModel, model_id: str = "", input_name: str = "input") -> tuple[Graph, dict]:
    """Returns (graph, info). The graph is a 1:1 mapping (no fusion); info holds
    per-layer precision and binarization records."""
    g = Graph()
    tmap: dict[tuple[str, int], str] = {}            # IR tensor -> Bitino tensor
    rank: dict[tuple[str, int], int] = {}
    pending_fq: dict[tuple[str, int], dict] = {}     # FQ output -> binarizer record
    precision: list[dict] = []
    results: list[str] = []

    def name_of(layer: IRLayer) -> str:
        return f"{layer.name}#{layer.id}"

    for lid in ir.topo:
        layer = ir.layers[lid]
        typ = layer.type
        out = (lid, min(layer.out_ports)) if layer.out_ports else None
        if typ in ("Const", "Convert"):
            if typ == "Convert" and not ir.is_const(ir.inputs_of(lid)[0]):
                raise NotImplementedError(f"{layer}: Convert of a non-constant tensor")
            continue
        ins = ir.inputs_of(lid)
        nm = name_of(layer)

        if typ == "Parameter":
            shp = _ints(layer.attrs["shape"])
            if len(shp) != 4 or layer.attrs.get("element_type") != "f32":
                raise NotImplementedError(f"{layer}: expected f32 NCHW 4-D input, got {shp}")
            n, c, h, w = shp
            g.add_input(input_name, [n, h, w, c])
            tmap[out], rank[out] = input_name, 4

        elif typ in ("Subtract", "Add", "Multiply"):
            if layer.attrs.get("auto_broadcast", "numpy") != "numpy":
                raise NotImplementedError(f"{layer}: auto_broadcast={layer.attrs.get('auto_broadcast')}")
            a, b = ins
            ca, cb = ir.is_const(a), ir.is_const(b)
            if not ca and not cb:
                if typ != "Add" or ir.shape_of(a) != ir.shape_of(b):
                    raise NotImplementedError(f"{layer}: tensor-tensor {typ} with shapes {ir.shape_of(a)}, {ir.shape_of(b)}")
                tmap[out] = g.add_node("Add", [tmap[a], tmap[b]], name=nm)
                rank[out] = rank[a]
            else:
                if ca and cb:
                    raise NotImplementedError(f"{layer}: constant-constant op (should be folded)")
                if ca:
                    if typ == "Subtract":
                        raise NotImplementedError(f"{layer}: const - tensor")
                    a, b = b, a
                shp = ir.shape_of(a)
                r = rank[a]
                ch = shp[1] if r == 4 else shp[-1]
                v = _per_channel(ir.const_value(b), ch, r, str(layer))
                if typ == "Multiply":
                    tmap[out] = g.add_node("ChannelAffine", [tmap[a], g.add_constant(nm + "/scale", v.astype(np.float32)), ""], name=nm)
                else:
                    bias = -v if typ == "Subtract" else v
                    tmap[out] = g.add_node("ChannelAffine", [tmap[a], "", g.add_constant(nm + "/bias", bias.astype(np.float32))], name=nm)
                rank[out] = r

        elif typ == "Convolution":
            x, wt = ins
            if not ir.is_const(wt):
                raise NotImplementedError(f"{layer}: non-constant filter")
            w = ir.const_value(wt).astype(np.float32)          # OIHW
            if w.ndim != 4:
                raise NotImplementedError(f"{layer}: filter rank {w.ndim}")
            attrs = {"strides": _ints(layer.attrs["strides"]), "dilations": _ints(layer.attrs["dilations"]),
                     "pads": _pads(layer), "groups": 1, "activation": "none"}
            wname = g.add_constant(nm + "/kernel", np.transpose(w, (0, 2, 3, 1)), layout="OHWI")
            tmap[out] = g.add_node("Conv2D", [tmap[x], wname, ""], attrs, name=nm)
            rank[out] = 4
            precision.append({"layer": layer.name, "ir_type": typ, "op": "Conv2D", "weights": f"fp32 {list(w.shape)}",
                              "activations": "fp32"})

        elif typ == "FakeQuantize":
            levels = int(layer.attrs["levels"])
            if levels != 2:
                raise NotImplementedError(f"{layer}: FakeQuantize levels={levels} (only binarizing levels=2 supported)")
            x = ins[0]
            if rank[x] != 4:
                raise NotImplementedError(f"{layer}: FakeQuantize on rank-{rank[x]} tensor")
            ch = ir.shape_of(x)[1]
            il, ih, ol, oh = (_per_channel(ir.const_value(t), ch, 4, f"{layer} input {i}") for i, t in enumerate(ins[1:], 1))
            if not np.array_equal(il, ih):
                raise NotImplementedError(f"{layer}: input_low != input_high; not a pure threshold")
            ok = ((ol == 0) & (oh == 1)) | ((ol == 1) & (oh == 0))
            if not ok.all():
                raise NotImplementedError(f"{layer}: output levels {set(zip(ol.tolist(), oh.tolist()))} are not {{0,1}}")
            cons = ir.consumers[out]
            for (t, _) in cons:
                if ir.layers[t].type != "BinaryConvolution":
                    raise NotImplementedError(f"{layer}: binarizer consumed by {ir.layers[t]} (only BinaryConvolution supported)")
            pending_fq[out] = {"name": layer.name, "x": tmap[x], "threshold": il.astype(np.float32),
                               "inverted": (ol == 1) & (oh == 0)}

        elif typ == "BinaryConvolution":
            x, wt = ins
            if layer.attrs.get("mode") != "xnor-popcount":
                raise NotImplementedError(f"{layer}: mode {layer.attrs.get('mode')}")
            if x not in pending_fq:
                raise NotImplementedError(f"{layer}: input is not a levels=2 FakeQuantize binarizer")
            fq = pending_fq[x]
            pv = float(layer.attrs.get("pad_value", "0"))
            if pv != -1.0 and any(_pads(layer)):
                # Only pad_value=-1 (padded tap = bit 0 = -1) is verified against the CPU plugin.
                raise NotImplementedError(f"{layer}: pad_value {pv} with padding (unverified semantics)")
            pv = -1.0 if not any(_pads(layer)) else pv
            bits = ir.const_value(wt)                          # OIHW u1 {0,1}
            if bits.shape[1] % 8:
                # Synthetic IRs with IC % 8 != 0 differ from the CPU plugin (by 0.25-0.96 on the
                # logits, cause not determined); the OMZ models have IC % 64 == 0.
                raise NotImplementedError(f"{layer}: BinaryConvolution with {bits.shape[1]} input channels "
                                          "(not a multiple of 8; semantics unverified against the CPU plugin)")
            if ir.layers[ir.inputs_of(lid)[1][0]].attrs.get("element_type") != "u1":
                raise NotImplementedError(f"{layer}: weights are not u1")
            wpm = np.where(bits == 1, 1, -1).astype(np.int8)  # u1 bit 1 = +1
            pads = _pads(layer)
            if fq["inverted"].any():
                if pv != 0.0 and any(pads):
                    raise NotImplementedError(f"{layer}: inverted binarizer channels with padding pad_value={pv}")
                wpm[:, fq["inverted"], :, :] *= -1
            attrs = {"strides": _ints(layer.attrs["strides"]), "dilations": _ints(layer.attrs["dilations"]),
                     "pads": pads, "groups": 1, "pad_value": pv, "input_binarize": "gt", "output": "f32",
                     "activation": "none"}
            wname = g.add_constant(nm + "/kernel_bits", np.transpose(wpm, (0, 2, 3, 1)), dtype="bit", layout="OHWI_bit")
            tname = g.add_constant(nm + "/threshold", fq["threshold"])
            tmap[out] = g.add_node("BConv2D", [fq["x"], wname, "", "", tname], attrs, name=nm)
            rank[out] = 4
            precision.append({"layer": layer.name, "ir_type": typ, "op": "BConv2D",
                              "weights": f"binary u1 {list(bits.shape)} (bit1=+1)",
                              "activations": f"binary via FakeQuantize '{fq['name']}' (levels=2, per-channel threshold, x>t -> +1)",
                              "pad_value": pv, "inverted_binarizer_channels": int(fq["inverted"].sum()),
                              "threshold_range": [float(fq["threshold"].min()), float(fq["threshold"].max())]})

        elif typ == "ReLU":
            tmap[out] = g.add_node("Activation", [tmap[ins[0]]], {"type": "relu"}, name=nm)
            rank[out] = rank[ins[0]]

        elif typ == "MaxPool":
            if any(v != 1 for v in _ints(layer.attrs.get("dilations", "1,1"))):
                raise NotImplementedError(f"{layer}: dilated MaxPool")
            for p in layer.out_ports:
                if p != min(layer.out_ports) and ir.consumers.get((lid, p)):
                    raise NotImplementedError(f"{layer}: MaxPool indices output is used")
            attrs = {"kernel": _ints(layer.attrs["kernel"]), "strides": _ints(layer.attrs["strides"]), "pads": _pads(layer)}
            tmap[out] = g.add_node("MaxPool2D", [tmap[ins[0]]], attrs, name=nm)
            rank[out] = 4
            _check_pool_shape(layer, ir.shape_of(ins[0]), attrs)

        elif typ == "AvgPool":
            k = _ints(layer.attrs["kernel"])
            pads = _pads(layer)
            shp = ir.shape_of(ins[0])
            if k == shp[2:] and not any(pads):
                tmap[out] = g.add_node("GlobalAvgPool", [tmap[ins[0]]], {"keepdims": 1}, name=nm)
            else:
                excl = layer.attrs.get("exclude-pad", "false") == "true"
                attrs = {"kernel": k, "strides": _ints(layer.attrs["strides"]), "pads": pads, "count_include_pad": 0 if excl else 1}
                _check_pool_shape(layer, shp, attrs)
                tmap[out] = g.add_node("AvgPool2D", [tmap[ins[0]]], attrs, name=nm)
            rank[out] = 4

        elif typ == "ReduceMean":
            axes = sorted(int(a) % 4 for a in ir.const_value(ins[1]).reshape(-1))
            if rank[ins[0]] != 4 or axes != [2, 3]:
                raise NotImplementedError(f"{layer}: ReduceMean over axes {axes}")
            keep = layer.attrs.get("keep_dims", "false") == "true"
            tmap[out] = g.add_node("GlobalAvgPool", [tmap[ins[0]]], {"keepdims": 1 if keep else 0}, name=nm)
            rank[out] = 4 if keep else 2

        elif typ == "Reshape":
            src_shape = ir.shape_of(ins[0])
            dst_shape = ir.shape_of(out)
            # NCHW -> flat is only layout-independent when the spatial dims are 1.
            if rank[ins[0]] == 4 and not (src_shape[2] == 1 and src_shape[3] == 1):
                raise NotImplementedError(f"{layer}: reshape of a spatial NCHW tensor {src_shape}")
            if int(np.prod(src_shape)) != int(np.prod(dst_shape)) or len(dst_shape) != 2:
                raise NotImplementedError(f"{layer}: reshape {src_shape} -> {dst_shape}")
            tmap[out] = g.add_node("Reshape", [tmap[ins[0]]], {"shape": dst_shape}, name=nm)
            rank[out] = 2

        elif typ == "MatMul":
            x, wt = ins
            if not ir.is_const(wt) or rank[x] != 2:
                raise NotImplementedError(f"{layer}: only 2-D activation x constant weight")
            if layer.attrs.get("transpose_a", "false") == "true":
                raise NotImplementedError(f"{layer}: transpose_a")
            w = ir.const_value(wt).astype(np.float32)
            if layer.attrs.get("transpose_b", "false") != "true":
                w = w.T                                           # [K, O] -> [O, K]
            tmap[out] = g.add_node("Dense", [tmap[x], g.add_constant(nm + "/kernel", np.ascontiguousarray(w)), ""],
                                   {"activation": "none"}, name=nm)
            rank[out] = 2
            precision.append({"layer": layer.name, "ir_type": typ, "op": "Dense", "weights": f"fp32 {list(w.shape)}", "activations": "fp32"})

        elif typ in ("SoftMax",):
            axis = int(layer.attrs.get("axis", "1"))
            if rank[ins[0]] != 2 or axis not in (1, -1):
                raise NotImplementedError(f"{layer}: SoftMax axis {axis} on rank {rank[ins[0]]}")
            tmap[out] = g.add_node("Softmax", [tmap[ins[0]]], name=nm)
            rank[out] = 2

        elif typ == "Result":
            results.append(tmap[ins[0]])

        else:
            raise NotImplementedError(f"layer {layer}: unsupported IR layer type {typ}")

    g.set_outputs(*results)
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "ir_op_counts": ir.op_counts(),
            "ir_version": ir.version, "u1_order": ir.u1_order}
    return g, info


def _check_pool_shape(layer: IRLayer, in_shape: list[int], attrs: dict) -> None:
    (kh, kw), (sh, sw), (pt, pb, pl, pr) = attrs["kernel"], attrs["strides"], attrs["pads"]
    oh = (in_shape[2] + pt + pb - kh) // sh + 1
    ow = (in_shape[3] + pl + pr - kw) // sw + 1
    want = layer.out_ports[min(layer.out_ports)]
    if [oh, ow] != want[2:]:
        raise NotImplementedError(f"{layer}: floor pooling gives {[oh, ow]} but IR says {want[2:]} (rounding_type={layer.attrs.get('rounding_type')})")


# ============================================================ fused-variant passes
def merge_affine_chains(g: Graph) -> int:
    """ChannelAffine -> ChannelAffine (single consumer) ==> one ChannelAffine whose
    scale/bias are composed in float64 and rounded once to float32. The OMZ IRs
    carry 4-6 consecutive constant Multiply/Add nodes after each binary conv."""
    count = 0
    changed = True
    while changed:
        changed = False
        cons: dict[str, list] = {}
        for n in g.nodes:
            for i in n.inputs:
                if i:
                    cons.setdefault(i, []).append(n)
        for n in g.outputs:
            cons.setdefault(n, []).append(None)
        prod = {o: n for n in g.nodes for o in n.outputs}
        for second in g.nodes:
            if second.op != "ChannelAffine":
                continue
            first = prod.get(second.inputs[0])
            if first is None or first.op != "ChannelAffine" or len(cons.get(first.outputs[0], [])) != 1:
                continue

            def vec(node, i):
                nm = node.inputs[i] if len(node.inputs) > i else ""
                return g.constants[nm].data.astype(np.float64) if nm else None

            s1, b1, s2, b2 = vec(first, 1), vec(first, 2), vec(second, 1), vec(second, 2)
            c = next(v.shape[0] for v in (s1, b1, s2, b2) if v is not None)
            s1 = np.ones(c) if s1 is None else s1
            s2 = np.ones(c) if s2 is None else s2
            b1 = np.zeros(c) if b1 is None else b1
            b2 = np.zeros(c) if b2 is None else b2
            scale = s1 * s2
            bias = b1 * s2 + b2
            tag = g.unique(second.name + "/merged")
            sname = g.add_constant(tag + "_scale", scale.astype(np.float32)) if not np.all(scale == 1) else ""
            bname = g.add_constant(tag + "_bias", bias.astype(np.float32)) if not np.all(bias == 0) else ""
            second.inputs = [first.inputs[0], sname, bname]
            g.nodes.remove(first)
            count += 1
            changed = True
            break
    return count


def fuse_threshold_into_bit_output(g: Graph) -> int:
    """BConv2D(output f32, no post_mul/post_add/activation) whose every consumer is a
    BConv2D binarizing it with input_binarize='gt' and the same threshold t
    ==> producer emits bits directly.

    The producer's value is an exact integer acc (|acc| <= K < 2^24, exact in f32).
    acc > t  <=>  acc >= floor(t) + 1  <=>  acc - (floor(t)+1) >= 0, and a bit output
    stores -1 iff (acc*post_mul + post_add) < 0. With post_mul = 1 and
    post_add = -(floor(t)+1) (an integer, exact in f32 for |t| < 2^24) both are exact,
    so the rewrite is bit-identical to the unfused graph in float32 AND float64."""
    count = 0
    cons: dict[str, list] = {}
    for n in g.nodes:
        for i in n.inputs:
            if i:
                cons.setdefault(i, []).append(n)
    outputs = set(g.outputs)
    for prod in g.nodes:
        if prod.op != "BConv2D" or prod.attrs.get("output", "f32") != "f32" or prod.attrs.get("activation", "none") != "none":
            continue
        if any(len(prod.inputs) > i and prod.inputs[i] for i in (2, 3)):
            continue
        out = prod.outputs[0]
        users = cons.get(out, [])
        if not users or out in outputs:
            continue
        thr = None
        ok = True
        for u in users:
            if u.op != "BConv2D" or u.inputs[0] != out or u.attrs.get("input_binarize") != "gt" or len(u.inputs) < 5 or not u.inputs[4]:
                ok = False
                break
            t = g.constants[u.inputs[4]].data
            if thr is not None and not np.array_equal(thr, t):
                ok = False
                break
            thr = t
        if not ok:
            continue
        k = int(np.prod(g.constants[prod.inputs[1]].shape[1:]))
        if np.any(np.abs(thr) >= 2 ** 24) or k >= 2 ** 24:
            thr = np.clip(thr, -float(k) - 1, float(k) + 1)  # same decisions: |acc| <= k
        add = -(np.floor(thr.astype(np.float64)) + 1.0)
        tag = g.unique(prod.name + "/bitout")
        prod.inputs = [prod.inputs[0], prod.inputs[1], "", g.add_constant(tag + "_post_add", add.astype(np.float32)),
                       prod.inputs[4] if len(prod.inputs) > 4 else ""]
        prod.attrs["output"] = "bit"
        for u in users:
            u.inputs = [u.inputs[0], u.inputs[1], u.inputs[2], u.inputs[3], ""]
            u.attrs["input_binarize"] = "ge"  # irrelevant for bit input; kept valid
        count += 1
    return count
