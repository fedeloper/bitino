#!/usr/bin/env python3
"""Direct writer: a verified Keras / Larq checkpoint -> canonical QONNX file of the Bitino profile
(docs/qonnx-profile.md, ``bitino-qonnx/1``); step 2 of the QONNX route (docs/qonnx-frontend.md § Plan,
docs/qonnx-step2.md). No tf2onnx: Keras has no ONNX exporter for the Larq quantizers, so the file is written from the
Keras model and the parsed weights of the legacy converter.

Two stages, one file:

  stage 1 (.cache/venv-larq: TensorFlow 2.15.1, larq 0.13.3, larq_zoo 2.4.0; python -I)
      .cache/venv-larq/bin/python -I tools/export_keras_qonnx.py --model-id larq-zoo-quicknet --weights <h5> \
          --out-dir .cache/qonnx_step2/larq-zoo-quicknet [--eval-dir .cache/data/eval128] [--repo <source tree>] \
          [--qonnx-python .cache/venv-qonnx/bin/python]
    builds the Keras model exactly as the legacy converter does (tools/convert_larq_zoo.py zoo_factory /
    tools/convert_keras_embedding.py build_larqface / build_brillsson, with their checkpoint and layer-order checks),
    writes the source model's fp32 / fp64 outputs on the check set, parses it with the legacy adapter
    (bitino.adapters.keras_larq.convert: every layer and quantizer interpretation checked numerically against the
    live layer there) and writes the ONNX graph of that parse as a spec (spec.json + spec.npz, numpy only), with the
    values taken from the Keras layers: BatchNorm folded in float64 exactly as the adapter folds it (float64 carriers,
    R6 / R7), binary weights as BipolarQuant(W_latent, 1) of the Keras latent kernel (checked: the adapter's bits are
    its signs) with the per-channel scale as a separate Mul (R4; the XNOR-Net and DoReFa scales as float64 carriers).
  stage 2 (.cache/venv-qonnx: onnx; started by stage 1, or ``--assemble DIR``)
      assembles the ONNX model (IR 10, ai.onnx 17 + qonnx.custom_op.general 1), the model and naming metadata
      (§ 6 / § 7) and checks it with bitino.qonnx_profile (the export fails unless check-qonnx passes with no error).

Patterns written (IR operator of the adapter -> standard ONNX + QONNX):
  BConv2D ge   BipolarQuant(x, 1) [-> Pad(v)] -> Conv(., BipolarQuant(W, 1) | ±1, pads 0) [-> Mul(alpha)] [-> Add(b)]
  BConv2D gt   Where(Greater(x, 0), 1, -1) -> ... (BinaryFaceNet's DoReFa(k=1, weights) input quantizer, whose
               operand x - 2^-24 tanh(max|x|) is the adapter's MaxPool / Mul(-1) / Concat / Reshape / Tanh / Mul / Add
               subgraph written node for node; its kernels are the live quantizer's ±1, written as constants)
  QBConv2D / QBDense  Round(Mul(Clip(x, 0, 1), n)) -> Conv / Gemm(., BipolarQuant(W, 1)) -> Mul(mean|W| / n) [-> Add]
  ChannelAffine  Mul(x, s) [-> Add(., b)] (BatchNorm: float64 carriers)    Conv2D / Dense  Conv / Gemm(transB 1)
  ChannelGate    [Reshape [N, C, 1, 1]] -> Sigmoid -> Mul(x, .)          Reshape (Keras flatten, channels last)
               [Transpose(0, 2, 3, 1)] -> Reshape                         MaxPool2D / AvgPool2D / Concat / Add / Pad /
               Relu / PRelu / Clip / Tanh / Softmax  as themselves (explicit pads, count_include_pad 0)
  a fused activation (relu) is a Relu node in a group of its own (``<node>/relu``), folded back by the importer's pass.
Naming metadata: bitino.name = the adapter's IR node name, bitino.order = 2 x its position (2 x + 1 for the Relu
group), bitino.consts = every constant input's IR name, bitino.attrs = pad_value's JSON form (BConv2D) / null (Conv2D:
no pad_value), model bitino.const_order where the adapter's data order is not first use (BinaryFaceNet).

Writes into --out-dir: model.onnx, export.json (checkpoint / ONNX SHA-256, versions, import arguments, check-qonnx
result, the source outputs below), source_logits.f32 / source_logits_f64.f64 (first output, the one the importer
verifies), source_outputs.f32 / source_outputs_f64.f64 (all outputs, concatenated per sample).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "python"))

PROFILE = "bitino-qonnx/1"
IR_VERSION = 10
OPSET = 17
QD = "qonnx.custom_op.general"
# import arguments of the Keras / Larq file layout: graph names from the metadata, data in first-use order (the
# adapter's creation order; bitino.const_order where it is not), one ChannelAffine per affine group, no qonnx cleanup
# (it would fold the float64 carriers), no bit outputs (the Larq converters never set them)
IMPORT_ARGS_KERAS = ["--naming", "exporter", "--constant-order", "first-use", "--affine", "keep", "--cleanup", "none",
                     "--bit-outputs", "none"]
# step 6: the same layout as model metadata (bitino.import {"mapping": "patterns", ...}, docs/qonnx-profile.md § 6): the
# file alone determines its .bitino, an import argument that contradicts it is an error
IMPORT_META_KERAS = json.dumps({"mapping": "patterns", "naming": "exporter", "constant_order": "first-use", "affine": "keep",
                                "cleanup": "none", "bit_outputs": "none", "flatten_dense": "transpose"}, sort_keys=True)
CONVERTERS = {"tools/convert_larq_zoo.py": "zoo", "tools/convert_keras_embedding.py": "embedding"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def recipe_of(model_id: str) -> dict:
    rec = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"].get(model_id)
    if rec is None or rec.get("converter") not in CONVERTERS:
        raise SystemExit(f"{model_id}: not a Keras / Larq recipe (converters {sorted(CONVERTERS)})")
    return rec


def recipe_args(rec: dict) -> dict:
    a = rec["steps"][0]["args"]
    out = {}
    i = 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a) and not a[i + 1].startswith("--"):
            out[a[i]] = a[i + 1]
            i += 2
        else:
            out[a[i]] = True
            i += 1
    return out


def keras_models() -> list:
    """Model ids of every Keras / Larq recipe (registry/recipes.json, by converter)."""
    recs = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    return [k for k, v in recs.items() if v.get("converter") in CONVERTERS]


# ================================================================================================ spec (numpy only)
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
        self.inits[name] = np.array(arr, order="C", copy=True)       # keeps 0-d scalars 0-d
        return name

    def node(self, op: str, ins: list, attrs: dict | None = None, out: str | None = None, meta: dict | None = None,
             domain: str = "") -> str:
        self.k += 1
        out = out or f"t{self.k}:{op.lower()}"
        self.nodes.append({"op": op, "inputs": list(ins), "outputs": [out], "attrs": dict(attrs or {}),
                           "name": f"n{self.k}_{op}", "domain": domain, "meta": dict(meta or {})})
        return out


def _nchw(shape) -> list:
    s = [int(d) for d in shape]
    return [s[0], s[3], s[1], s[2]] if len(s) == 4 else s


def _onnx_pads(p) -> list:          # IR [t, b, l, r] -> ONNX [t, l, b, r]
    return [int(p[0]), int(p[2]), int(p[1]), int(p[3])]


class KerasWriter:
    """The adapter's IR graph (before the passes) -> canonical ONNX spec, with values from the Keras layers."""

    def __init__(self, g0, model, shapes: dict):
        self.g0 = g0
        self.model = model
        self.layers = {l.name: l for l in model.layers}
        self.shapes = shapes                  # IR tensor -> NHWC / NC shape
        self.s = Spec()
        self.report: dict = {"carriers": 0, "latent_weights": 0, "pm1_weights": 0, "relu_groups": 0, "transposes": 0,
                             "explicit_binary_pads": 0}
        s = self.s
        self.ONE = s.init("bitino/one", np.array(1.0, np.float32))
        self.ZERO = s.init("bitino/zero", np.array(0.0, np.float32))
        self.M1 = s.init("bitino/minus_one", np.array(-1.0, np.float32))

    # ---------------------------------------------------------------- constants
    def cdata(self, name: str) -> np.ndarray:
        from bitino.ir import unpack_bits
        c = self.g0.constants[name]
        if c.dtype == "bit":
            return unpack_bits(c.data, c.shape[-1]).reshape(c.shape)
        return np.asarray(c.data)

    def const(self, cname: str, shape, c64=None) -> str:
        """ONNX tensor holding IR constant cname reshaped to `shape`: a FLOAT initializer, or a float64 carrier
        Cast(DOUBLE, FLOAT) when the source computes the value in float64 (its float32 rounding must be the IR's)."""
        v32 = np.asarray(self.cdata(cname), np.float32)
        if c64 is None:
            return self.s.init(cname, v32.reshape(shape))
        c64 = np.broadcast_to(np.asarray(c64, np.float64), v32.shape)
        if not np.array_equal(c64.astype(np.float32).view(np.uint32), v32.view(np.uint32)):
            raise AssertionError(f"{cname}: float32 rounding of the float64 carrier differs from the adapter's constant")
        self.report["carriers"] += 1
        d = self.s.init(f"{cname}:f64", np.ascontiguousarray(c64.reshape(shape)))
        return self.s.node("Cast", [d], {"to": 1}, out=cname)          # constant-only node: no naming metadata

    def bn64(self, node):
        """float64 (scale, shift) of a BatchNormalization layer, exactly the adapter's expression before its float32
        rounding, or None when the ChannelAffine is not a Keras BatchNormalization."""
        layer = self.layers.get(node.name)
        if layer is None or type(layer).__name__ != "BatchNormalization" or node.inputs[1:] != [f"{node.name}/scale", f"{node.name}/shift"]:
            return None
        gamma = layer.gamma.numpy() if layer.scale else 1.0
        beta = layer.beta.numpy() if layer.center else 0.0
        mean = layer.moving_mean.numpy()
        var = layer.moving_variance.numpy()
        inv = 1.0 / np.sqrt(var.astype(np.float64) + layer.epsilon)
        return np.asarray(gamma * inv, np.float64), np.asarray(beta - mean * gamma * inv, np.float64)

    def latent(self, node, bits_oihw: np.ndarray, kind: str):
        """The Keras latent kernel (OIHW / [O, K]) whose larq sign (0 -> +1) is the adapter's ±1, or None."""
        from bitino.adapters import keras_larq as kl
        layer = self.layers.get(node.name)
        if layer is None or type(layer).__name__ not in ("QuantConv2D", "QuantDense") or getattr(layer, "kernel", None) is None:
            return None
        kq = kl._qname(getattr(layer, "kernel_quantizer", None))
        if kq not in kl.SIGN_QUANTIZERS | kl.SCALED_SIGN_QUANTIZERS | kl.XNOR_QUANTIZERS | kl.LAYER_SCALED_SIGN_QUANTIZERS:
            return None
        k = layer.kernel.numpy()
        lat = np.transpose(k, (3, 2, 0, 1)) if kind == "conv" else k.T
        if lat.shape != bits_oihw.shape or not np.array_equal(np.where(lat >= 0, 1, -1), bits_oihw):
            raise AssertionError(f"{node.name}: the adapter's binary weights are not the larq sign of the Keras kernel")
        return np.ascontiguousarray(lat.astype(np.float32))

    def weights_pm1(self, node, cname: str, kind: str) -> str:
        bits = self.cdata(cname).astype(np.int8)
        b = np.transpose(bits, (0, 3, 1, 2)) if kind == "conv" else bits          # OHWI -> OIHW; dense [O, K]
        lat = self.latent(node, b, kind)
        if lat is not None:
            self.report["latent_weights"] += 1
            w = self.s.init(f"{cname}:latent", lat)
            return self.s.node("BipolarQuant", [w, self.ONE], domain=QD, out=f"{cname}:pm1")
        self.report["pm1_weights"] += 1
        return self.s.init(f"{cname}:pm1", b.astype(np.float32))

    def scale64(self, node):
        """float64 post_mul of an XNOR-Net (mean |clip(W, -1, 1)|) or DoReFa (mean |W| / n) layer; None otherwise."""
        from bitino.adapters import keras_larq as kl
        layer = self.layers.get(node.name)
        if layer is None or getattr(layer, "kernel", None) is None:
            return None
        kq = kl._qname(getattr(layer, "kernel_quantizer", None))
        k = layer.kernel.numpy()
        if node.op == "BConv2D" and kq in kl.XNOR_QUANTIZERS:
            return np.mean(np.abs(np.clip(k.astype(np.float64), -1.0, 1.0)), axis=(0, 1, 2))
        if node.op in ("QBConv2D", "QBDense") and kq in kl.LAYER_SCALED_SIGN_QUANTIZERS:
            n_lv = 2 ** int(node.attrs["act_bits"]) - 1
            scale = float(np.mean(np.abs(k.astype(np.float64))))
            return np.full(k.shape[-1], scale / n_lv, np.float64)
        return None

    # ---------------------------------------------------------------- nodes
    def write(self):
        g0, s = self.g0, self.s
        names = {n.name for n in g0.nodes}
        inp = g0.inputs[0]
        if len(g0.inputs) != 1 or inp["name"] != "input":
            raise AssertionError("one graph input named 'input' expected")
        s.inputs = [("input", _nchw(inp["shape"]))]
        for i, n in enumerate(g0.nodes):
            if f"{n.name}/relu" in names:
                raise AssertionError(f"{n.name}/relu is an IR node name")
            meta = {"bitino.name": n.name, "bitino.order": str(2 * i)}
            consts = {str(j): c for j, c in enumerate(n.inputs) if c and c in g0.constants}
            self.cur = (n, meta, consts, i)
            self.attrs_meta = None
            getattr(self, f"op_{n.op}")(n)
        s.outputs = [(o, _nchw(self.shapes[o])) for o in g0.outputs]
        return s

    def emit(self, op, ins, attrs=None, out=None, anchor=False, domain=""):
        """A node of the current IR node's group; the anchor carries bitino.consts / bitino.attrs."""
        n, meta, consts, _ = self.cur
        m = dict(meta)
        if anchor:
            if consts:
                m["bitino.consts"] = json.dumps(consts, sort_keys=True)
            if getattr(self, "attrs_meta", None) is not None:
                m["bitino.attrs"] = json.dumps(self.attrs_meta, sort_keys=True)
        return self.s.node(op, ins, attrs, out=out, meta=m, domain=domain)

    def relu(self, n, x: str) -> str:
        _, _, _, i = self.cur
        self.report["relu_groups"] += 1
        return self.s.node("Relu", [x], out=n.outputs[0], meta={"bitino.name": f"{n.name}/relu", "bitino.order": str(2 * i + 1)})

    def fused_act(self, n, x: str, last: bool = True) -> str:
        act = n.attrs.get("activation", "none")
        if act == "none":
            return x
        if act == "relu":
            return self.relu(n, x)
        raise AssertionError(f"{n.name}: fused activation {act}")

    def _out(self, n):
        return n.outputs[0] if n.attrs.get("activation", "none") == "none" else None

    def _check_attrs(self, n, allowed):
        extra = set(n.attrs) - set(allowed)
        if extra:
            raise AssertionError(f"{n.name} ({n.op}): attributes {sorted(extra)} are not written by this exporter")

    def op_ChannelAffine(self, n):
        self.attrs_meta = None
        x, sc, sh = n.inputs
        shp = self.shapes[x]
        cs = [1, shp[-1], 1, 1] if len(shp) == 4 else [1, shp[-1]]
        f64 = self.bn64(n)
        y = x
        if sc:
            y = self.emit("Mul", [y, self.const(sc, cs, f64[0] if f64 else None)], out=None if sh else n.outputs[0], anchor=True)
        if sh:
            y = self.emit("Add", [y, self.const(sh, cs, f64[1] if f64 else None)], out=n.outputs[0], anchor=not sc)
        if not sc and not sh:
            raise AssertionError(f"{n.name}: ChannelAffine without constants")

    def op_Conv2D(self, n):
        self._check_attrs(n, ("strides", "dilations", "pads", "groups", "activation"))
        self.attrs_meta = {"pad_value": None}         # the IR node has no pad_value (the importer writes 0.0)
        x, w, b = (n.inputs + ["", ""])[:3]
        wd = self.cdata(w)                             # OHWI
        oihw = np.ascontiguousarray(np.transpose(wd, (0, 3, 1, 2)))
        if np.all(np.abs(oihw) == 1.0):                # ±1 float weights (BinaryAlexNet stem): the latent kernel
            wt = self.weights_pm1(n, w, "conv") if self.layers.get(n.name) is not None and \
                self.latent(n, oihw.astype(np.int8), "conv") is not None else self.s.init(w, oihw.astype(np.float32))
        else:
            wt = self.s.init(w, oihw.astype(np.float32))
        ins = [x, wt] + ([self.const(b, [wd.shape[0]])] if b else [])
        y = self.emit("Conv", ins, {"kernel_shape": list(wd.shape[1:3]), "strides": list(n.attrs["strides"]),
                                    "dilations": list(n.attrs["dilations"]), "pads": _onnx_pads(n.attrs["pads"]),
                                    "group": int(n.attrs["groups"])}, out=self._out(n), anchor=True)
        self.fused_act(n, y)

    def op_BConv2D(self, n):
        self._check_attrs(n, ("strides", "dilations", "pads", "groups", "activation", "pad_value", "input_binarize", "output"))
        x, w, pm, pa, thr = n.inputs
        if thr or n.attrs["output"] != "f32":
            raise AssertionError(f"{n.name}: threshold / bit output not written by this exporter")
        pv = n.attrs["pad_value"]
        self.attrs_meta = {"pad_value": pv}            # the IR's JSON form of the value the Pad (or its absence) states
        mode = n.attrs["input_binarize"]
        if mode == "ge":
            xb = self.emit("BipolarQuant", [x, self.ONE], domain=QD)
        elif mode == "gt":
            xb = self.emit("Where", [self.emit("Greater", [x, self.ZERO]), self.ONE, self.M1])
        else:
            raise AssertionError(f"{n.name}: input_binarize {mode}")
        pads = list(n.attrs["pads"])
        if any(pads) or float(pv) != 0.0:
            t, b_, l, r = pads
            self.report["explicit_binary_pads"] += 1
            xb = self.emit("Pad", [xb, self.s.init(f"{n.name}/pads:onnx", np.array([0, 0, t, l, 0, 0, b_, r], np.int64)),
                                   self.s.init(f"bitino/pad_value_{float(pv):g}", np.array(float(pv), np.float32))],
                           {"mode": "constant"})
        wd = self.cdata(w)
        o, kh, kw, _ = wd.shape
        y = self.emit("Conv", [xb, self.weights_pm1(n, w, "conv")],
                      {"kernel_shape": [kh, kw], "strides": list(n.attrs["strides"]), "dilations": list(n.attrs["dilations"]),
                       "pads": [0, 0, 0, 0], "group": int(n.attrs["groups"])},
                      out=n.outputs[0] if not pm and not pa and self._out(n) else None, anchor=True)
        y = self._epilogue(n, y, pm, pa, [1, o, 1, 1])
        self.fused_act(n, y)

    def _epilogue(self, n, y, pm, pa, cs):
        out = self._out(n)
        if pm:
            y = self.emit("Mul", [y, self.const(pm, cs, self.scale64(n))], out=out if not pa else None)
        if pa:
            y = self.emit("Add", [y, self.const(pa, cs)], out=out)
        return y

    def _dorefa_levels(self, n, x):
        nlev = 2 ** int(n.attrs["act_bits"]) - 1
        if n.attrs.get("act_quant") != "dorefa":
            raise AssertionError(f"{n.name}: act_quant {n.attrs.get('act_quant')}")
        c = self.emit("Clip", [x, self.ZERO, self.ONE])
        m = self.emit("Mul", [c, self.s.init(f"bitino/levels_{nlev}", np.array(float(nlev), np.float32))])
        return self.emit("Round", [m])

    def op_QBConv2D(self, n):
        self._check_attrs(n, ("strides", "dilations", "pads", "groups", "activation", "act_bits", "act_quant"))
        self.attrs_meta = None
        x, w, pm, pa = n.inputs
        q = self._dorefa_levels(n, x)
        o, kh, kw, _ = self.cdata(w).shape
        y = self.emit("Conv", [q, self.weights_pm1(n, w, "conv")],
                      {"kernel_shape": [kh, kw], "strides": list(n.attrs["strides"]), "dilations": list(n.attrs["dilations"]),
                       "pads": _onnx_pads(n.attrs["pads"]), "group": int(n.attrs["groups"])}, anchor=True)
        self.fused_act(n, self._epilogue(n, y, pm, pa, [1, o, 1, 1]))

    def op_QBDense(self, n):
        self._check_attrs(n, ("activation", "act_bits", "act_quant"))
        self.attrs_meta = None
        x, w, pm, pa = n.inputs
        q = self._dorefa_levels(n, x)
        o = self.cdata(w).shape[0]
        y = self.emit("Gemm", [q, self.weights_pm1(n, w, "dense")], {"transB": 1}, anchor=True)
        self.fused_act(n, self._epilogue(n, y, pm, pa, [1, o]))

    def op_Dense(self, n):
        self._check_attrs(n, ("activation",))
        self.attrs_meta = None
        x, w, b = (n.inputs + ["", ""])[:3]
        wd = self.cdata(w)
        ins = [x, self.s.init(w, wd.astype(np.float32))] + ([self.const(b, [wd.shape[0]])] if b else [])
        y = self.emit("Gemm", ins, {"transB": 1}, out=self._out(n), anchor=True)
        self.fused_act(n, y)

    def _pool(self, n, op, extra=None):
        self._check_attrs(n, ("kernel", "strides", "pads", "count_include_pad"))
        self.attrs_meta = None
        at = {"kernel_shape": list(n.attrs["kernel"]), "strides": list(n.attrs["strides"]), "pads": _onnx_pads(n.attrs["pads"])}
        at.update(extra or {})
        self.emit(op, [n.inputs[0]], at, out=n.outputs[0], anchor=True)

    def op_MaxPool2D(self, n):
        if "count_include_pad" in n.attrs:
            raise AssertionError(n.name)
        self._pool(n, "MaxPool")

    def op_AvgPool2D(self, n):
        self._pool(n, "AveragePool", {"count_include_pad": int(n.attrs.get("count_include_pad", 0))})

    def op_GlobalAvgPool(self, n):
        self.attrs_meta = None
        if int(n.attrs.get("keepdims", 0)):
            self.emit("GlobalAveragePool", [n.inputs[0]], out=n.outputs[0], anchor=True)
        else:
            self.emit("Flatten", [self.emit("GlobalAveragePool", [n.inputs[0]], anchor=True)], {"axis": 1}, out=n.outputs[0])

    def op_Reshape(self, n):
        self.attrs_meta = None
        x = n.inputs[0]
        src, dst = list(self.shapes[x]), [int(d) for d in n.attrs["shape"]]
        if list(self.shapes[n.outputs[0]]) != dst:
            raise AssertionError(f"{n.name}: reshape target {dst} vs executor shape {self.shapes[n.outputs[0]]}")
        kept = lambda s: len(s) != 4 or s[3] == 1 or s[1] * s[2] == 1   # noqa: E731  NHWC order = NCHW order
        if len(src) == 4:
            if not kept(dst):
                raise AssertionError(f"{n.name}: reshape {src} -> {dst} (target does not keep the element order)")
            tgt = _nchw(dst)
            if kept(src) and _nchw(src) != tgt and len(dst) == 2:
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
        self.attrs_meta = None
        if int(n.attrs.get("axis", -1)) not in (-1, len(self.shapes[n.inputs[0]]) - 1):
            raise AssertionError(f"{n.name}: concat axis {n.attrs}")
        self.emit("Concat", list(n.inputs), {"axis": 1}, out=n.outputs[0], anchor=True)

    def op_Add(self, n):
        self.attrs_meta = None
        if self.shapes[n.inputs[0]] != self.shapes[n.inputs[1]]:
            raise AssertionError(f"{n.name}: Add of different shapes")
        self.emit("Add", list(n.inputs), out=n.outputs[0], anchor=True)

    def op_AddBroadcast(self, n):
        self.attrs_meta = None
        if self.shapes[n.inputs[0]] == self.shapes[n.inputs[1]]:
            raise AssertionError(f"{n.name}: AddBroadcast of equal shapes")
        self.emit("Add", list(n.inputs), out=n.outputs[0], anchor=True)

    def op_Pad(self, n):
        self.attrs_meta = None
        p = [int(v) for v in n.attrs["pads"]]
        if len(p) != 8 or p[0] or p[1]:
            raise AssertionError(f"{n.name}: pads {p}")
        _, _, t, b, l, r, c0, c1 = p
        self.emit("Pad", [n.inputs[0], self.s.init(f"{n.name}/pads:onnx", np.array([0, c0, t, l, 0, c1, b, r], np.int64)),
                          self.s.init(f"bitino/pad_value_{float(n.attrs.get('value', 0.0)):g}",
                                      np.array(float(n.attrs.get("value", 0.0)), np.float32))],
                  {"mode": "constant"}, out=n.outputs[0], anchor=True)

    def op_Activation(self, n):
        self.attrs_meta = None
        t = n.attrs["type"]
        x = n.inputs[0]
        if t == "relu":
            self.emit("Relu", [x], out=n.outputs[0], anchor=True)
        elif t == "tanh":
            self.emit("Tanh", [x], out=n.outputs[0], anchor=True)
        elif t == "hardtanh":
            lo = self.s.init(f"bitino/clip_{float(n.attrs['min']):g}", np.array(float(n.attrs["min"]), np.float32))
            hi = self.s.init(f"bitino/clip_{float(n.attrs['max']):g}", np.array(float(n.attrs["max"]), np.float32))
            if float(np.float32(n.attrs["min"])) != float(n.attrs["min"]) or float(np.float32(n.attrs["max"])) != float(n.attrs["max"]):
                raise AssertionError(f"{n.name}: hardtanh bounds are not float32 values")
            self.emit("Clip", [x, lo, hi], out=n.outputs[0], anchor=True)
        elif t == "prelu":
            shp = self.shapes[x]
            self.emit("PRelu", [x, self.const(n.inputs[1], [shp[-1], 1, 1] if len(shp) == 4 else [shp[-1]])],
                      out=n.outputs[0], anchor=True)
        else:
            raise AssertionError(f"{n.name}: activation {t}")

    def op_Softmax(self, n):
        self.attrs_meta = None
        if len(self.shapes[n.inputs[0]]) != 2:
            raise AssertionError(n.name)
        self.emit("Softmax", [n.inputs[0]], {"axis": -1}, out=n.outputs[0], anchor=True)

    def op_ChannelGate(self, n):
        self.attrs_meta = None
        if n.attrs.get("gate") != "sigmoid":
            raise AssertionError(f"{n.name}: gate {n.attrs}")
        x, gate = n.inputs
        gs, xs = self.shapes[gate], self.shapes[x]
        if len(xs) != 4 or tuple(gs) not in ((xs[0], xs[3]), (xs[0], 1, 1, xs[3])):
            raise AssertionError(f"{n.name}: gate shape {gs} for {xs}")
        if len(gs) == 2:
            gate = self.emit("Reshape", [gate, self.s.init(f"{n.name}/gate_shape:onnx", np.array([gs[0], gs[1], 1, 1], np.int64))])
        sg = self.emit("Sigmoid", [gate])
        self.emit("Mul", [x, sg], out=n.outputs[0], anchor=True)


def first_use_order(g) -> list:
    order, seen = [], set()
    for n in g.nodes:
        for i in n.inputs:
            if i and i in g.constants and i not in seen:
                seen.add(i)
                order.append(i)
    return order + [k for k in g.constants if k not in seen]


# ================================================================================================ stage 1 (TensorFlow)
def build_source(model_id: str, rec: dict, weights: str, repo: str | None, x: np.ndarray):
    """(model, fp32 outputs [n, ...], fp64 outputs, first-output width, info) exactly as the legacy converter."""
    import tensorflow as tf
    ra = recipe_args(rec)
    fam = CONVERTERS[rec["converter"]]
    info: dict = {"legacy_converter": rec["converter"]}
    if fam == "zoo":
        import convert_larq_zoo as cz
        mf = ra.get("--zoo-module-file")
        if mf:
            if not repo:
                raise SystemExit(f"{model_id}: --repo (the larq/zoo source tree) is needed for {mf}")
            mf = mf.replace("{repo}", repo)
        factory, legacy = cz.zoo_factory(ra["--zoo-class"], mf)
        model = factory(weights=weights, include_top=True)
        info["h5_layer_order_check"] = cz.check_h5_layer_order(model, weights)
        info["zoo_class"] = ra["--zoo-class"]
        if legacy:
            info["legacy_source"] = legacy
        last = model.layers[-1]
        logits_model = tf.keras.Model(model.inputs, [last.input, model.output])
        o32 = []
        for i in range(0, len(x), 16):
            lg, pr = logits_model(x[i:i + 16], training=False)
            o32.append(np.concatenate([lg.numpy(), pr.numpy()], axis=1))
        o32 = np.concatenate(o32).astype(np.float32)
        tf.keras.backend.set_floatx("float64")
        try:
            model64 = factory(weights=None, include_top=True)
            model64.set_weights([w.astype(np.float64) for w in model.get_weights()])
            lm64 = tf.keras.Model(model64.inputs, [model64.layers[-1].input, model64.output])
            o64 = []
            for i in range(0, len(x), 16):
                lg, pr = lm64(x[i:i + 16].astype(np.float64), training=False)
                o64.append(np.concatenate([lg.numpy(), pr.numpy()], axis=1))
            o64 = np.concatenate(o64).astype(np.float64)
            del model64, lm64
        finally:
            tf.keras.backend.set_floatx("float32")
        width = int(last.input.shape[-1])
    else:
        import convert_keras_embedding as ce
        family = ra["--family"]
        if family == "larqface":
            model, build64, src_info = ce.build_larqface(weights)
        else:
            model, build64, src_info = ce.build_brillsson(weights, ra["--zoo-class"], int(x.shape[1]), int(ra.get("--hidden", 1024)))
        info.update({"family": family, **{k: v for k, v in src_info.items() if k not in ("h5_layer_names",)}})
        o32 = ce.run_keras(model, x, np.float32).astype(np.float32)
        try:
            m64 = build64()
            o64 = ce.run_keras(m64, x, np.float64)
            del m64
        finally:
            tf.keras.backend.set_floatx("float32")
        outs = model.outputs
        width = int(np.prod([int(d) for d in outs[0].shape[1:]]))
    return model, o32, o64, width, info


def stage1(args) -> int:
    rec = recipe_of(args.model_id)
    reg = json.load(open(os.path.join(ROOT, "registry", "models.json")))
    entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
    ck = sha256_file(args.weights)
    if entry is None or ck != entry["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch / unknown model: {ck}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {ck}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    import larq
    import larq_zoo
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from bitino import executor, passes
    from bitino.adapters import keras_larq

    ev = rec["eval"]
    eval_dir = args.eval_dir or os.path.join(args.cache_dir, "data", ev["dir"])
    em = json.load(open(os.path.join(eval_dir, "manifest.json")))
    pre = em["preprocessing"][ev["preprocess"]]
    x = np.fromfile(os.path.join(eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    model, o32, o64, width, info = build_source(args.model_id, rec, args.weights, args.repo, x)
    files = {"source_outputs.f32": o32, "source_outputs_f64.f64": o64,
             "source_logits.f32": np.ascontiguousarray(o32[:, :width]), "source_logits_f64.f64": np.ascontiguousarray(o64[:, :width])}
    for k, v in files.items():
        v.tofile(os.path.join(args.out_dir, k))

    g0, _ = keras_larq.convert(model, args.model_id)
    if CONVERTERS[rec["converter"]] == "zoo":       # tools/convert_larq_zoo.py: logits (softmax input) + probabilities
        sm = g0.nodes[-1]
        assert sm.op == "Softmax"
        g0.set_outputs(sm.inputs[0], sm.outputs[0])
    shp = list(g0.inputs[0]["shape"])
    env = executor.run(g0, {"input": np.zeros(shp, np.float32)}, keep_intermediates=True)
    shapes = {k: list(np.shape(v)) for k, v in env.items() if k not in g0.constants}
    shapes["input"] = shp
    del env
    w = KerasWriter(g0, model, shapes)
    spec = w.write()
    # data order: the adapter's creation order; the importer writes first use, so a model whose creation order is not
    # first use (BinaryFaceNet's DoReFa operand constants) carries the published final order as bitino.const_order
    import copy
    gp = copy.deepcopy(g0)
    applied = passes.optimize(gp, fold=True)
    const_order = None
    if first_use_order(g0) != list(g0.constants):
        const_order = list(gp.constants)

    ra = recipe_args(rec)
    lic = (entry.get("license") or {}).get("weights") or "unknown"
    labels = json.dumps({"num_classes": 1000, "set": "imagenet-1k"}) if "imagenet" in em.get("name", "") else "none"
    src = (rec.get("sources") or [{}])[0]
    commit = _git_commit()
    fw = f"tensorflow {tf.__version__}, larq {larq.__version__}, larq_zoo {larq_zoo.__version__}"
    meta = {"bitino.profile": PROFILE, "bitino.model_id": args.model_id, "bitino.variant": "float", "bitino.layout": "NCHW",
            "bitino.input_domain": "float32",
            "bitino.preprocessing": json.dumps({"key": ev["preprocess"], "definition": pre["definition"]}),
            "bitino.labels": labels, "bitino.source.framework": "keras+larq", "bitino.source.framework_version": fw,
            "bitino.source.checkpoint_sha256": ck, "bitino.weights_license": lic,
            "bitino.exporter": f"tools/export_keras_qonnx.py@{commit or 'unknown'} (direct writer from the Keras model, "
                               f"adapter bitino.adapters.keras_larq); {fw}",
            "bitino.notes": "larq sign binarizers (0 -> +1) written as BipolarQuant(x, 1)"
                            + ("; DoReFa(k=1, weights) input quantizers as Where(Greater(x - 2^-24 tanh(max|x|), 0), 1, -1)"
                               if any(n.attrs.get("input_binarize") == "gt" for n in g0.nodes) else "")
                            + f"; outputs {list(g0.outputs)}; published file model.bitino"}
    if src.get("repo") and src.get("commit"):
        meta["bitino.source.code"] = f"{src['repo']}@{src['commit']}"
    if const_order is not None:
        meta["bitino.const_order"] = json.dumps(const_order)
    meta["bitino.import"] = IMPORT_META_KERAS            # step 6: the file determines its import (no import argument)
    labels_list = [s.get("label") for s in em["samples"]][: len(x)]
    sdir = os.path.join(args.out_dir, "spec")
    os.makedirs(sdir, exist_ok=True)
    np.savez(os.path.join(sdir, "spec.npz"), **{f"i{j}": v for j, v in enumerate(spec.inits.values())})
    rec_out = {
        "model_id": args.model_id, "family": info.get("family", "larq-zoo"), "legacy_converter": rec["converter"],
        "adapter": "bitino.adapters.keras_larq", "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": ck,
        "framework": f"{fw}, numpy {np.__version__}, python {platform.python_version()}",
        "exporter": f"tools/export_keras_qonnx.py@{commit or 'unknown'} direct writer (no tf2onnx)",
        "source": {k: v for k, v in info.items()}, "import_args": IMPORT_ARGS_KERAS,
        "writer": w.report, "ir_nodes": len(g0.nodes), "ir_constants": len(g0.constants),
        "published_passes": applied, "const_order": const_order is not None,
        "eval_set": em["name"], "eval_preprocessing": ev["preprocess"], "n_samples": int(len(x)),
        "output_widths": [int(np.prod(shapes[o][1:])) for o in g0.outputs],
        "source_top1": (float(np.mean(o32[:, :width].argmax(1) == np.array(labels_list))) if labels != "none" else None),
        "source_f32_vs_f64_max_abs": float(np.abs(o32.astype(np.float64) - o64).max()),
        "logits_f32": os.path.abspath(os.path.join(args.out_dir, "source_logits.f32")),
        "logits_f64": os.path.abspath(os.path.join(args.out_dir, "source_logits_f64.f64")),
        "venv": "stage 1 .cache/venv-larq, stage 2 .cache/venv-qonnx (read only, python -I)",
    }
    json.dump({"nodes": spec.nodes, "init_names": list(spec.inits), "inputs": spec.inputs, "outputs": spec.outputs,
               "meta": meta, "export": rec_out}, open(os.path.join(sdir, "spec.json"), "w"))
    del model, g0, gp, spec, w
    import gc
    gc.collect()
    print(f"spec written: {len(json.load(open(os.path.join(sdir, 'spec.json')))['nodes'])} ONNX nodes", flush=True)
    py = args.qonnx_python or os.path.join(args.cache_dir, "venv-qonnx", "bin", "python")
    r = subprocess.run([py, "-I", os.path.abspath(__file__), "--assemble", args.out_dir] + (["--keep-spec"] if args.keep_spec else []))
    return r.returncode


# ================================================================================================ stage 2 (onnx)
def assemble(out_dir: str, keep_spec: bool) -> int:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from bitino.qonnx_profile import check_file

    sdir = os.path.join(out_dir, "spec")
    sp = json.load(open(os.path.join(sdir, "spec.json")))
    z = np.load(os.path.join(sdir, "spec.npz"))
    inits = [numpy_helper.from_array(z[f"i{j}"], name) for j, name in enumerate(sp["init_names"])]
    nodes = []
    for nd in sp["nodes"]:
        n = helper.make_node(nd["op"], nd["inputs"], nd["outputs"], name=nd["name"], domain=nd["domain"] or None, **nd["attrs"])
        for k, v in nd["meta"].items():
            n.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
        nodes.append(n)
    ins = [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in sp["inputs"]]
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in sp["outputs"]]
    graph = helper.make_graph(nodes, sp["meta"]["bitino.model_id"], ins, outs, inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid(QD, 1)],
                          producer_name="bitino tools/export_keras_qonnx.py", producer_version=PROFILE)
    m.ir_version = IR_VERSION
    meta = dict(sp["meta"])
    meta["bitino.exporter"] += f"; onnx {onnx.__version__}"
    for k, v in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    path = os.path.join(out_dir, "model.onnx")
    onnx.save(m, path)
    del m, graph, nodes, inits, z
    rep = check_file(path)
    prof = rep.to_dict() if hasattr(rep, "to_dict") else {"passed": rep.passed}
    if not prof.get("passed"):
        print(json.dumps(prof, indent=1)[:4000], file=sys.stderr)
        print("ERROR: the written file does not pass check-qonnx (profile bitino-qonnx/1)", file=sys.stderr)
        return 5
    exp = sp["export"]
    exp.update({"onnx_file": "model.onnx", "onnx_sha256": sha256_file(path), "onnx_bytes": os.path.getsize(path),
                "onnx_producer": f"bitino tools/export_keras_qonnx.py {PROFILE}", "opsets": [["ai.onnx", OPSET], [QD, 1]],
                "onnx_nodes": len(sp["nodes"]), "onnx": onnx.__version__,
                "profile_check": {k: prof.get(k) for k in ("profile", "passed", "summary")} | {
                    "warnings": [f for f in prof.get("findings", []) if f.get("level") == "warning"]}})
    json.dump(exp, open(os.path.join(out_dir, "export.json"), "w"), indent=1)
    if not keep_spec:
        shutil.rmtree(sdir)
    print(f"exported {exp['model_id']} -> {path} ({exp['onnx_bytes']} bytes, sha256 {exp['onnx_sha256'][:16]}), "
          f"{exp['onnx_nodes']} nodes; check-qonnx {prof.get('summary')}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id")
    ap.add_argument("--weights")
    ap.add_argument("--out-dir")
    ap.add_argument("--eval-dir", help="check set (default: <cache>/data/<recipe eval dir>)")
    ap.add_argument("--repo", help="read-only source tree for {repo} in the recipe (QuickNet-XL legacy: larq/zoo @40f4cdc)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--qonnx-python", help="interpreter of stage 2 (default <cache>/venv-qonnx/bin/python)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep-spec", action="store_true")
    ap.add_argument("--assemble", metavar="DIR", help="stage 2 only (venv-qonnx): assemble DIR/spec into DIR/model.onnx")
    ap.add_argument("--list", action="store_true", help="print the Keras / Larq model ids of the registry")
    args = ap.parse_args()
    if args.list:
        print("\n".join(keras_models()))
        return 0
    if args.assemble:
        return assemble(args.assemble, args.keep_spec)
    if not (args.model_id and args.weights and args.out_dir):
        ap.error("--model-id, --weights and --out-dir are required")
    return stage1(args)


if __name__ == "__main__":
    sys.exit(main())
