"""OpenVINO IR adapter tests (python/bitino/adapters/openvino_ir.py).

A small synthetic IR v11 (.xml + .bin) with the same layer types and attributes as the OMZ
binary classifiers is generated here: Subtract (mean), Convolution, per-channel Add/Multiply,
ReLU, FakeQuantize levels=2 (per-channel thresholds, output {0,1}), BinaryConvolution
(xnor-popcount, pad_value -1, u1 weights), a
BinaryConvolution -> FakeQuantize -> BinaryConvolution chain (threshold fusion), residual Add,
MaxPool, ReduceMean, Reshape and MatMul.

Checked: the independent IR evaluator against a hand-written loop; the converted graphs
(unfused 1:1 and fused) against the IR evaluator in float64; unknown ops raise; and, when the
packages/binaries are available, the IR evaluator against the OpenVINO CPU plugin (u1 bit order
and pad semantics) and the C++ runtime against the numpy executor (bit-exact).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import openvino_ir as oi  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))


class IRBuilder:
    """Writes a minimal OpenVINO IR v11 (.xml + .bin)."""

    def __init__(self):
        self.layers, self.edges, self.blob = [], [], bytearray()
        self.shapes: dict[int, list[int]] = {}
        self._out_port: dict[int, int] = {}

    def _dims(self, shape):
        return "".join(f"<dim>{d}</dim>" for d in shape)

    def _add(self, typ, version, data, in_ids, out_shape, name=None, out_prec="FP32"):
        lid = len(self.layers)
        attrs = " ".join(f'{k}="{v}"' for k, v in data.items())
        ins = "".join(f'<port id="{i}" precision="FP32">{self._dims(self.shapes[s])}</port>' for i, s in enumerate(in_ids))
        out_port = len(in_ids)
        xml = (f'<layer id="{lid}" name="{name or typ.lower() + str(lid)}" type="{typ}" version="{version}">'
               f'<data {attrs}/>' + (f"<input>{ins}</input>" if in_ids else "")
               + (f'<output><port id="{out_port}" precision="{out_prec}">{self._dims(out_shape)}</port></output>' if out_shape is not None else "")
               + "</layer>")
        self.layers.append(xml)
        for i, s in enumerate(in_ids):
            self.edges.append(f'<edge from-layer="{s}" from-port="{self._out_port[s]}" to-layer="{lid}" to-port="{i}"/>')
        self.shapes[lid] = out_shape
        self._out_port[lid] = out_port
        return lid

    def param(self, shape):
        return self._add("Parameter", "opset1", {"shape": ",".join(map(str, shape)), "element_type": "f32"}, [], shape)

    def const(self, arr, et="f32"):
        arr = np.asarray(arr)
        if et == "u1":
            raw = np.packbits(arr.reshape(-1).astype(np.uint8), bitorder="little").tobytes()  # LSB-first
        elif et == "i64":
            raw = arr.astype(np.int64).tobytes()
        else:
            raw = arr.astype(np.float32).tobytes()
        off = len(self.blob)
        self.blob.extend(raw)
        return self._add("Const", "opset1", {"element_type": et, "shape": ",".join(map(str, arr.shape)), "offset": off,
                                             "size": len(raw)}, [], list(arr.shape), out_prec="BIN" if et == "u1" else "FP32")

    def op(self, typ, ins, out_shape, version="opset1", **data):
        return self._add(typ, version, data, ins, out_shape)

    def write(self, path_stem):
        xml = ('<?xml version="1.0"?><net name="synthetic" version="11"><layers>' + "".join(self.layers)
               + "</layers><edges>" + "".join(self.edges) + "</edges></net>")
        open(path_stem + ".xml", "w").write(xml)
        open(path_stem + ".bin", "wb").write(bytes(self.blob))
        return path_stem + ".xml", path_stem + ".bin"


def build_ir(tmp_path, seed=0, ic=16, mid=24):
    """Returns (xml, bin). Thresholds are offset by +0.25 from integers where they act on
    binary-conv outputs, and float thresholds are random, so no value sits on a threshold."""
    rng = np.random.default_rng(seed)
    b = IRBuilder()
    h = w = 8
    x = b.param([1, 3, h, w])
    x = b.op("Subtract", [x, b.const(rng.normal(100, 10, (1, 3, 1, 1)))], [1, 3, h, w], auto_broadcast="numpy")
    x = b.op("Convolution", [x, b.const(rng.normal(0, 0.05, (ic, 3, 3, 3)))], [1, ic, h, w], strides="1, 1", dilations="1, 1",
             pads_begin="1, 1", pads_end="1, 1", auto_pad="explicit")
    x = b.op("Add", [x, b.const(rng.normal(0, 0.5, (1, ic, 1, 1)))], [1, ic, h, w], auto_broadcast="numpy")
    stem = b.op("ReLU", [x], [1, ic, h, w])

    def fq(src, c, thr):
        return b.op("FakeQuantize", [src, b.const(thr.reshape(1, c, 1, 1)), b.const(thr.reshape(1, c, 1, 1)),
                                     b.const(np.zeros((1, 1, 1, 1))), b.const(np.ones((1, 1, 1, 1)))],
                    b.shapes[src], levels=2, auto_broadcast="numpy")

    def bconv(src, cin, cout, k, pad):
        wbits = rng.integers(0, 2, (cout, cin, k, k))
        oh = b.shapes[src][2]
        return b.op("BinaryConvolution", [src, b.const(wbits, "u1")], [1, cout, oh, oh], strides="1, 1", dilations="1, 1",
                    pads_begin=f"{pad}, {pad}", pads_end=f"{pad}, {pad}", mode="xnor-popcount", pad_value="-1", auto_pad="explicit")

    a = fq(stem, ic, rng.uniform(0.0, 0.6, ic))
    y = bconv(a, ic, mid, 3, 1)                                     # 3x3 pad 1: border taps contribute -w
    y = fq(y, mid, rng.integers(-20, 20, mid) + 0.25)             # threshold on the integer conv output
    y = bconv(y, mid, ic, 1, 0)                                     # 1x1
    y = b.op("Multiply", [y, b.const(np.full((1, 1, 1, 1), 0.5))], [1, ic, h, w], auto_broadcast="numpy")
    y = b.op("Add", [y, b.const(rng.normal(0, 3, (1, ic, 1, 1)))], [1, ic, h, w], auto_broadcast="numpy")
    y = b.op("Multiply", [y, b.const(rng.uniform(0.05, 0.2, (1, ic, 1, 1)))], [1, ic, h, w], auto_broadcast="numpy")
    y = b.op("Add", [y, b.const(rng.normal(0, 0.3, (1, ic, 1, 1)))], [1, ic, h, w], auto_broadcast="numpy")
    y = b.op("Add", [y, stem], [1, ic, h, w], auto_broadcast="numpy")
    y = b.op("ReLU", [y], [1, ic, h, w])
    y = b.op("MaxPool", [y], [1, ic, 4, 4], strides="2, 2", pads_begin="1, 1", pads_end="1, 1", kernel="3, 3",
             rounding_type="floor", auto_pad="explicit")
    y = b.op("ReduceMean", [y, b.const(np.array([2, 3]), "i64")], [1, ic, 1, 1], keep_dims="true")
    y = b.op("Reshape", [y, b.const(np.array([1, ic]), "i64")], [1, ic], special_zero="true")
    y = b.op("MatMul", [y, b.const(rng.normal(0, 1, (10, ic)))], [1, 10], transpose_a="false", transpose_b="true")
    y = b.op("Add", [y, b.const(rng.normal(0, 1, (1, 10)))], [1, 10], auto_broadcast="numpy")
    b.op("Result", [y], None)
    return b.write(str(tmp_path / f"synthetic{seed}"))


def inputs(n, seed=1):
    return np.random.default_rng(seed).uniform(0, 255, (n, 1, 8, 8, 3)).astype(np.float32)  # NHWC BGR 0..255


def test_binary_convolution_against_loop():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, (1, 5, 4, 4)).astype(np.float64)        # FQ output {0,1}
    w = np.where(rng.integers(0, 2, (3, 5, 3, 3)) == 1, 1.0, -1.0)
    got = oi._conv_nchw(2 * a - 1, w, [1, 1], [1, 1], [1, 1, 1, 1], -1.0)
    ref = np.zeros((1, 3, 4, 4))
    for o in range(3):
        for y in range(4):
            for x in range(4):
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = y + ky - 1, x + kx - 1
                        inside = 0 <= iy < 4 and 0 <= ix < 4
                        b = (2 * a[0, :, iy, ix] - 1) if inside else -np.ones(5)   # padded tap: bit 0 = -1
                        ref[0, o, y, x] += np.dot(b, w[o, :, ky, kx])
    np.testing.assert_array_equal(got, ref)


def test_u1_lsb_first(tmp_path):
    b = IRBuilder()
    b.param([1, 1, 1, 1])
    bits = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], np.uint8)
    cid = b.const(bits, "u1")
    xml, binf = b.write(str(tmp_path / "u1"))
    assert open(binf, "rb").read() == bytes([0b11000001, 0b00000101])
    ir = oi.IRModel(xml, binf)
    np.testing.assert_array_equal(ir.const_value((str(cid), 0)), bits)


@pytest.mark.parametrize("seed", [0, 1])
def test_converted_graphs_match_ir_evaluation_f64(tmp_path, seed):
    xml, binf = build_ir(tmp_path, seed)
    ir = oi.IRModel(xml, binf)
    unfused, info = oi.convert(ir)
    fused, _ = oi.convert(ir)
    assert oi.merge_affine_chains(fused) >= 2
    applied = passes.optimize(fused, fold=True)
    assert oi.fuse_threshold_into_bit_output(fused) == 1
    assert applied["fold_affine_into_conv"] >= 1
    assert sum(1 for n in fused.nodes if n.op == "BConv2D" and n.attrs.get("output") == "bit") == 1
    assert [p["op"] for p in info["layer_precision"]].count("BConv2D") == 2
    for g, tol in ((unfused, 1e-12), (fused, 1e-5)):
        path = tmp_path / "m.bitino"
        bfformat.write(g, str(path))
        g2 = bfformat.read(str(path))
        for x in inputs(6, seed + 10):
            ref = oi.evaluate(ir, x.transpose(0, 3, 1, 2), np.float64)["outputs"][0]
            got = executor.run(g2, {"input": x}, float_dtype=np.float64)[g2.outputs[0]]
            np.testing.assert_allclose(got, ref, rtol=0, atol=tol)


def test_binary_conv_input_channels_not_multiple_of_8_raise(tmp_path):
    # The CPU plugin disagrees with the spec-level evaluation for such IRs (see adapter docstring).
    ir = oi.IRModel(*build_ir(tmp_path, 0, ic=12, mid=24))
    with pytest.raises(NotImplementedError, match="multiple of 8"):
        oi.convert(ir)


def test_unknown_layer_raises(tmp_path):
    b = IRBuilder()
    x = b.param([1, 2, 4, 4])
    y = b.op("Sigmoid", [x], [1, 2, 4, 4])
    b.op("Result", [y], None)
    ir = oi.IRModel(*b.write(str(tmp_path / "bad")))
    with pytest.raises(NotImplementedError, match="Sigmoid"):
        oi.convert(ir)
    with pytest.raises(NotImplementedError, match="Sigmoid"):
        oi.evaluate(ir, np.zeros((1, 2, 4, 4)))


def test_matches_openvino_cpu_plugin(tmp_path):
    ov = pytest.importorskip("openvino")
    xml, binf = build_ir(tmp_path, 0)
    core = ov.Core()
    cm = core.compile_model(core.read_model(xml, binf), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    ir = oi.IRModel(xml, binf)
    ir_msb = oi.IRModel(xml, binf, u1_order="msb")
    worst_msb = 0.0
    for x in inputs(6, 5):
        xn = x.transpose(0, 3, 1, 2).copy()
        r = np.asarray(cm(xn)[0])
        np.testing.assert_allclose(oi.evaluate(ir, xn, np.float64)["outputs"][0], r, rtol=1e-4, atol=1e-4)
        worst_msb = max(worst_msb, float(np.abs(oi.evaluate(ir_msb, xn, np.float64)["outputs"][0] - r).max()))
    assert worst_msb > 1e-2  # the bit order is actually discriminated by this IR


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage", ["reference", "binary_scalar", "neon"])
def test_cpp_runtime_bit_exact_with_numpy(tmp_path, stage):
    xml, binf = build_ir(tmp_path, 1)
    ir = oi.IRModel(xml, binf)
    g, _ = oi.convert(ir)
    oi.merge_affine_chains(g)
    passes.optimize(g, fold=True)
    oi.fuse_threshold_into_bit_output(g)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    xs = inputs(4, 9)
    xs.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(4, 10)
    for i in range(4):
        np.testing.assert_array_equal(got[i], executor.run(g, {"input": xs[i]})[g.outputs[0]].ravel())
