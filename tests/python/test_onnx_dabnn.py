"""daBNN ONNX adapter: a synthetic Bi-Real-style ONNX (fp stem conv + BN + maxpool, two all-+-1 convs with
BN and real shortcut, avgpool downsample, global pool, 1x1 classifier) converted to Bitino IR, checked in
float64 against tools/convert_dabnn.py's graph surgery reference (numpy ONNX evaluator), for pad_value -1 / 0.

Needs onnx (and onnxruntime for the surgery cross-check), e.g. the project venv .cache/venv-qonnx:
  .cache/venv-qonnx/bin/python -m pytest tests/python/test_onnx_dabnn.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from bitino import executor  # noqa: E402
from bitino.adapters import onnx_dabnn  # noqa: E402

_spec = importlib.util.spec_from_file_location("convert_dabnn", os.path.join(ROOT, "tools", "convert_dabnn.py"))
convert_dabnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_dabnn)


def _model(rng, c=32, hw=16, ncls=10):
    inits, nodes = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), name))
        return name

    def bn(x, ch, out):
        nodes.append(helper.make_node("BatchNormalization", [x, const(out + "_g", rng.uniform(0.5, 2, ch) * rng.choice([-1, 1], ch)),
                                                            const(out + "_b", rng.normal(0, 1, ch)), const(out + "_m", rng.normal(0, 1, ch)),
                                                            const(out + "_v", rng.uniform(0.5, 2, ch))], [out], epsilon=1e-5))
        return out

    nodes.append(helper.make_node("Conv", ["in", const("w0", rng.normal(0, 0.3, (c, 3, 3, 3)))], ["c0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    x = bn("c0", c, "b0")
    nodes.append(helper.make_node("MaxPool", [x], ["p0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]))
    x = "p0"
    for i in range(2):
        nodes.append(helper.make_node("Conv", [x, const(f"wb{i}", rng.choice([-1.0, 1.0], (c, c, 3, 3)))], [f"cb{i}"], kernel_shape=[3, 3],
                                      pads=[1, 1, 1, 1], strides=[1, 1]))
        y = bn(f"cb{i}", c, f"bb{i}")
        nodes.append(helper.make_node("Add", [y, x], [f"a{i}"]))
        x = f"a{i}"
    nodes.append(helper.make_node("Pad", [x], ["pz"], mode="constant", pads=[0] * 8, value=0.0))
    nodes.append(helper.make_node("AveragePool", ["pz"], ["ap"], kernel_shape=[2, 2], strides=[2, 2]))
    nodes.append(helper.make_node("Conv", ["ap", const("wd", rng.normal(0, 0.2, (c, c, 1, 1)))], ["cd"], kernel_shape=[1, 1]))
    x = bn("cd", c, "bd")
    nodes.append(helper.make_node("Relu", [x], ["r"]))
    s = hw // 4
    nodes.append(helper.make_node("AveragePool", ["r"], ["gp"], kernel_shape=[s, s], strides=[1, 1]))
    nodes.append(helper.make_node("Conv", ["gp", const("wf", rng.normal(0, 0.2, (ncls, c, 1, 1))), const("bf", rng.normal(0, 0.1, ncls))],
                                  ["out"], kernel_shape=[1, 1]))
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 3, hw, hw])],
                          [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, ncls, 1, 1])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 9)])
    m.ir_version = 4
    onnx.checker.check_model(m)
    return m


@pytest.mark.parametrize("pad_value", [-1, 0])
def test_adapter_matches_surgery_reference_f64(pad_value):
    rng = np.random.default_rng(3 + pad_value)
    m = _model(rng)
    ref_model, probes = convert_dabnn.surgery(m, "ge", pad_value)
    assert len(probes) == 2
    g, info = onnx_dabnn.convert(m, binarize="ge", pad_value=pad_value)
    assert g.op_counts()["BConv2D"] == 2 and g.op_counts()["GlobalAvgPool"] == 1
    for c, v in info["float64_constants"].items():
        g.constants[c].data = v
    for _ in range(4):
        x = rng.normal(0, 1, (1, 16, 16, 3)).astype(np.float32)
        ref = convert_dabnn.evaluate_onnx(ref_model, x.transpose(0, 3, 1, 2), np.float64)["out"].reshape(-1)
        got = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
        assert np.max(np.abs(got - ref)) < 1e-9


def test_surgery_onnxruntime_matches_numpy_f32():
    ort = pytest.importorskip("onnxruntime")  # noqa: F841
    rng = np.random.default_rng(11)
    m = _model(rng)
    ref_model, _ = convert_dabnn.surgery(m, "ge", -1)
    x = rng.normal(0, 1, (3, 16, 16, 3)).astype(np.float32)
    lg, stats, _ = convert_dabnn.ort_run(ref_model, x, 1)
    for i in range(3):
        r = convert_dabnn.evaluate_onnx(ref_model, x[i:i + 1].transpose(0, 3, 1, 2), np.float64)["out"].reshape(-1)
        assert np.max(np.abs(lg[i] - r)) < 1e-3


def test_rejects_nonzero_pad_node():
    rng = np.random.default_rng(0)
    m = _model(rng)
    for n in m.graph.node:
        if n.op_type == "Pad":
            for a in n.attribute:
                if a.name == "pads":
                    a.ints[:] = [0, 0, 1, 1, 0, 0, 1, 1]
    with pytest.raises(NotImplementedError):
        onnx_dabnn.convert(m)
