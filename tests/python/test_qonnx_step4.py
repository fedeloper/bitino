"""QONNX route step 4 (docs/qonnx-step4.md): the direct writers of tools/qonnx_ir_writer.py and the step-4 importer
patterns, on small synthetic graphs. Each test writes a legacy-style IR graph (or FINN-style layer specs) to a canonical
QONNX model, checks it against the profile, imports it and compares with the graph the legacy converter's passes give
(data + graph SHA-256), and the IR's float64 outputs with the ONNX graph's (OnnxF64, the carriers at their float64
value: agreement to rounding level, 1e-5 relative).

Covered: the keep layout of the daBNN / NNCF / tensorpack / FDA-BNN converters (BN carriers, explicit binary Pad with
pad_value -1 in the integer JSON form, gt thresholds, Add.activation "none" added, AvgPool2D without
count_include_pad), the OpenVINO layout (binary conv -> gt binarizer of another group, bitino.import bit_outputs
threshold), BDense kernels without a layout tag (bitino.const_layouts), the FINN / BNN-PYNQ / Brevitas layer specs
(fold mode, the extra-named input ChannelAffine before the FakeQuant, pool before BN with negative BN scales) and the
file-recorded import options (bitino.import mapping patterns).

  .cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_step4.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import finn_bnn, openvino_ir  # noqa: E402
from bitino.fetch import data_sha256, graph_sha256  # noqa: E402
from bitino.ir import Graph  # noqa: E402
from bitino.onnx_f64 import OnnxF64  # noqa: E402
from bitino.qonnx_import import QonnxImportError, _step4_resolve_args, import_model, step4_import_options  # noqa: E402
from bitino.qonnx_profile import check_model  # noqa: E402
from qonnx_ir_writer import QD, IRWriter, LayerWriter, ir_shapes  # noqa: E402

META = {"bitino.profile": "bitino-qonnx/1", "bitino.model_id": "t", "bitino.variant": "float", "bitino.layout": "NCHW",
        "bitino.input_domain": "float32", "bitino.preprocessing": json.dumps({"key": "k", "definition": "d"}),
        "bitino.labels": "none", "bitino.source.framework": "test", "bitino.source.framework_version": "0",
        "bitino.source.checkpoint_sha256": "0" * 64, "bitino.weights_license": "MIT", "bitino.exporter": "test"}
KEEP = {"mapping": "patterns", "naming": "exporter", "constant_order": "first-use", "affine": "keep", "cleanup": "none",
        "bit_outputs": "none"}
FOLD = {"mapping": "patterns", "naming": "exporter", "constant_order": "walk", "affine": "fold", "cleanup": "none",
        "bit_outputs": "set"}


def to_model(spec, opts, extra_meta=None):
    inits = [numpy_helper.from_array(v, k) for k, v in spec.inits.items()]
    nodes = []
    for nd in spec.nodes:
        n = helper.make_node(nd["op"], nd["inputs"], nd["outputs"], name=nd["name"], domain=nd["domain"] or None, **nd["attrs"])
        for k, v in nd["meta"].items():
            n.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
        nodes.append(n)
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in spec.inputs],
                          [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in spec.outputs], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(QD, 1)], producer_name="test", producer_version="0")
    m.ir_version = 10
    for k, v in {**META, "bitino.import": json.dumps(opts), **(extra_meta or {})}.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    return m


def do_import(m):
    o = step4_import_options(m)
    return import_model(m, naming=o["naming"], affine=o["affine"], constant_order=o["constant_order"])[0]


def same_file(ga, gb, tmp_path):
    pa, pb = str(tmp_path / "a.bitino"), str(tmp_path / "b.bitino")
    bfformat.write(ga, pa)
    bfformat.write(gb, pb)
    return data_sha256(pa) == data_sha256(pb), graph_sha256(pa) == graph_sha256(pb)


def f64_close(m, g, x_nhwc, tol=1e-5):
    ev = OnnxF64(m)
    ref = np.asarray(ev.run(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2)), np.float64)[ev.output_names[0]]).reshape(-1)
    ir = executor.run(g, {"input": x_nhwc}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
    return float(np.max(np.abs(ref - ir))) <= tol * max(1.0, float(np.max(np.abs(ref))))


def _bits(rng, shape):
    return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)


def _dabnn_like(rng, c=8):
    """BConv2D(ge, pad_value -1 int) chains with BN ChannelAffines (float64 values), residual Add with activation none,
    an unpadded AvgPool2D without count_include_pad, GlobalAvgPool + Dense: the daBNN converter's pre-pass graph."""
    g = Graph()
    x = g.add_input("input", [1, 8, 8, c])
    c64 = {}

    def bn(t, name):
        s, b = rng.uniform(0.2, 2, c), rng.normal(0, 1, c)
        c64[f"{name}/scale"], c64[f"{name}/shift"] = s, b
        return g.add_node("ChannelAffine", [t, g.add_constant(f"{name}/scale", s.astype(np.float32)),
                                            g.add_constant(f"{name}/shift", b.astype(np.float32))], name=name)
    y = g.add_node("Conv2D", [x, g.add_constant("stem/kernel", rng.normal(0, 1, (c, 3, 3, c)).astype(np.float32), layout="OHWI"), ""],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "activation": "none", "pad_value": 0.0}, name="stem")
    y = bn(y, "stem_bn")
    for i in range(2):
        z = g.add_node("BConv2D", [y, g.add_constant(f"b{i}/kernel_bits", _bits(rng, (c, 3, 3, c)), dtype="bit", layout="OHWI_bit"), "", "", ""],
                       {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "activation": "none", "pad_value": -1,
                        "input_binarize": "ge", "output": "f32"}, name=f"b{i}")
        z = bn(z, f"b{i}_bn")
        y = g.add_node("Add", [z, y], {"activation": "none"}, name=f"add{i}")
    y = g.add_node("AvgPool2D", [y], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name="pool")
    y = g.add_node("GlobalAvgPool", [y], {"keepdims": 0}, name="gap")
    y = g.add_node("Dense", [y, g.add_constant("fc/kernel", rng.normal(0, 1, (5, c)).astype(np.float32)),
                             g.add_constant("fc/bias", rng.normal(0, 1, 5).astype(np.float32))], {"activation": "none"}, name="fc")
    g.set_outputs(y)
    return g, c64


def test_keep_layout_roundtrip_dabnn_like(tmp_path):
    rng = np.random.default_rng(0)
    g0, c64 = _dabnn_like(rng)
    w = IRWriter(g0, ir_shapes(g0), c64)
    m = to_model(w.write(), KEEP)
    rep = check_model(m)
    assert rep.passed() and not [f for f in rep.findings if f.level == "warning"], [f.code for f in rep.findings]
    gp = copy.deepcopy(g0)
    passes.optimize(gp, fold=True)
    g = do_import(m)
    assert same_file(gp, g, tmp_path) == (True, True)
    assert f64_close(m, g, rng.normal(0, 1, (1, 8, 8, 8)).astype(np.float32))
    assert w.report["carriers"] == 6 and w.report["explicit_binary_pads"] == 2


def _omz_like(rng, c=16):
    """OpenVINO layout: Conv2D, BConv2D(gt, threshold, pad -1.0) -> BConv2D(gt, the same kind of threshold) directly
    (no epilogue: the fuse makes it a bit output), then a scale ChannelAffine, GlobalAvgPool keepdims 1 + Reshape."""
    g = Graph()
    x = g.add_input("input", [1, 6, 6, c])
    y = g.add_node("Conv2D", [x, g.add_constant("c0#1/kernel", rng.normal(0, 1, (c, 3, 3, c)).astype(np.float32), layout="OHWI"), ""],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "activation": "none"}, name="c0#1")
    for i in range(2):
        t = rng.normal(0, 3, c).astype(np.float32)
        y = g.add_node("BConv2D", [y, g.add_constant(f"b{i}#2/kernel_bits", _bits(rng, (c, 3, 3, c)), dtype="bit", layout="OHWI_bit"), "", "",
                                   g.add_constant(f"b{i}#2/threshold", t)],
                       {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": -1.0,
                        "input_binarize": "gt", "output": "f32", "activation": "none"}, name=f"b{i}#2")
    y = g.add_node("ChannelAffine", [y, g.add_constant("m#3/scale", rng.uniform(0.5, 1, c).astype(np.float32)), ""], name="m#3")
    y = g.add_node("GlobalAvgPool", [y], {"keepdims": 1}, name="gap#4")
    y = g.add_node("Reshape", [y], {"shape": [1, c]}, name="flat#5")
    g.set_outputs(y)
    return g


def test_threshold_bit_outputs_omz_like(tmp_path):
    rng = np.random.default_rng(1)
    g0 = _omz_like(rng)
    m = to_model(IRWriter(g0, ir_shapes(g0)).write(), {**KEEP, "bit_outputs": "threshold"})
    assert check_model(m).passed()
    gp = copy.deepcopy(g0)
    passes.optimize(gp, fold=True)
    assert openvino_ir.fuse_threshold_into_bit_output(gp) == 1
    passes.prune_unused_constants(gp)
    g = do_import(m)
    assert [n.attrs.get("output") for n in g.nodes if n.op == "BConv2D"] == ["bit", "f32"]
    assert set(g.constants) == set(gp.constants)          # the data order (the legacy creation order) is bitino.const_order
    m2 = to_model(IRWriter(g0, ir_shapes(g0)).write(), {**KEEP, "bit_outputs": "threshold"},
                  {"bitino.const_order": json.dumps(list(gp.constants))})
    g2 = do_import(m2)
    assert same_file(gp, g2, tmp_path) == (True, True)
    x = rng.normal(0, 1, (1, 6, 6, 16)).astype(np.float32)
    assert np.array_equal(executor.run(gp, {"input": x})[gp.outputs[0]], executor.run(g2, {"input": x})[g2.outputs[0]])
    # an explicit bit-output layout that contradicts the file's bitino.import is an error
    with pytest.raises(ValueError):
        import_model(m2, naming="exporter", affine="keep", constant_order="first-use", bit_outputs="none")


def test_bdense_untagged_kernel_const_layouts(tmp_path):
    rng = np.random.default_rng(2)
    g0 = Graph()
    x = g0.add_input("input", [1, 12])
    k = 12
    y = g0.add_node("BDense", [x, g0.add_constant("d/kernel_bits", _bits(rng, (4, k)), dtype="bit"),
                               g0.add_constant("d/post_mul", np.full(4, 0.25, np.float32)), g0.add_constant("d/post_add", rng.normal(0, 1, 4).astype(np.float32)),
                               g0.add_constant("d/threshold", np.full(k, 0.5, np.float32))], {"input_binarize": "gt", "activation": "none"}, name="d")
    g0.set_outputs(y)
    m = to_model(IRWriter(g0, ir_shapes(g0)).write(), KEEP)
    assert check_model(m).passed()
    g = do_import(m)
    assert g.constants["d/kernel_bits"].layout == ""
    assert same_file(g0, g, tmp_path) == (True, True)
    bad = to_model(IRWriter(g0, ir_shapes(g0)).write(), KEEP)
    for nd in bad.graph.node:
        for p in nd.metadata_props:
            if p.key == "bitino.const_layouts":
                p.value = json.dumps({"1": "OHWI"})
    assert not check_model(bad).passed()


def _bnnpynq_like(rng):
    """FINN-style layer specs of the Theano CNV: int8 input (ChannelAffine(1, 0) + FakeQuant 1/128), W1A8 conv0,
    conv1 -> MaxPool -> BN (negative scales) -> sign (pool before BN), conv2, flatten at 1x1, final dense."""
    from bitino.adapters.finn_bnn import InputSpec, Layer
    c = 8
    inp = InputSpec("int8", [1, 10, 10, 3], affine=(1.0, 0.0), q={"scale": 1.0 / 128, "qmin": -128, "qmax": 127})

    def aff(o):
        s = rng.uniform(0.05, 0.5, o) * np.where(rng.random(o) < 0.4, -1, 1)
        return s, rng.normal(0, 1.5, o)
    layers = [Layer("conv0", "conv", _bits(rng, (c, 3, 3, 3)), *aff(c), w1a8=True),
              Layer("conv1", "conv", _bits(rng, (c, c, 3, 3)), *aff(c), pool="before_bn"),
              Layer("conv2", "conv", _bits(rng, (c, c, 3, 3)), *aff(c)),
              Layer("dense0", "dense", _bits(rng, (5, c)), np.full(5, 0.3), np.full(5, -0.1), final=True)]
    return inp, layers


def test_layer_specs_pool_before_bn_fold(tmp_path):
    rng = np.random.default_rng(3)
    inp, layers = _bnnpynq_like(rng)
    gf, _ = finn_bnn.emit(inp, layers, fused=True)
    passes.optimize(gf, fold=False)
    w = LayerWriter(inp, layers, gf)
    m = to_model(w.write(), FOLD)
    rep = check_model(m)
    assert rep.passed() and not [f for f in rep.findings if f.level == "warning"], [f.code for f in rep.findings]
    g = do_import(m)
    assert same_file(gf, g, tmp_path) == (True, True)
    assert g.nodes[0].name == "input_affine" and list(g.constants)[:2] == ["input/scale", "input/shift"]
    thr = [g.constants[n.inputs[4]].data for n in g.nodes if n.op == "BConv2D" and n.inputs[4]]
    assert len(thr) == 1 and np.array_equal(thr[0], np.round(thr[0]))      # the pool-before-BN integer threshold
    x = (np.round(rng.uniform(-1, 1, (1, 10, 10, 3)) * 127) / 128).astype(np.float32)
    assert np.array_equal(executor.run(gf, {"input": x})[gf.outputs[0]], executor.run(g, {"input": x})[g.outputs[0]])


def test_layer_specs_fc_input_sign_threshold_renamed(tmp_path):
    """Brevitas FC: sign(2x - 1) on the flattened input -> the first BDense's input threshold 0.5, named by bitino.consts
    (the flip-tracing record follows the rename)."""
    from bitino.adapters.finn_bnn import InputSpec, Layer
    rng = np.random.default_rng(6)
    spec = InputSpec("binary_threshold", [1, 4, 4, 1], threshold=0.5)
    layers = [Layer("d0", "dense", _bits(rng, (6, 16)), rng.uniform(0.1, 1, 6) * np.where(rng.random(6) < 0.5, -1, 1), rng.normal(0, 2, 6)),
              Layer("d1", "dense", _bits(rng, (3, 6)), np.full(3, 0.5), np.full(3, 0.25), final=True)]
    gf, _ = finn_bnn.emit(spec, layers, fused=True)
    w = LayerWriter(InputSpec("affine_sign", [1, 4, 4, 1]), layers, gf, input_affine_sign=(2.0, -1.0))
    m = to_model(w.write(), FOLD)
    assert check_model(m).passed()
    o = step4_import_options(m)
    g, info = import_model(m, naming=o["naming"], affine=o["affine"], constant_order=o["constant_order"])
    assert same_file(gf, g, tmp_path) == (True, True)
    assert [r["ir_threshold"] for r in info["binarizers"] if r["ir_threshold"]] == ["input/threshold"]


def test_pool_before_bn_polarity_needs_valid_conv():
    from types import SimpleNamespace

    from bitino.qonnx_import import Importer, V
    signs = np.ones((2, 3, 3, 3), np.int8)
    xv = V("bin", "t", (1, 3, 4, 4), extra={"flip": np.array([1, -1, 1], np.int8)})
    out = Importer._step4_flip_inputs(None, xv, signs, [0, 0, 0, 0], SimpleNamespace(name="c", op_type="Conv"))
    assert np.array_equal(out[:, 1], -signs[:, 1]) and np.array_equal(out[:, [0, 2]], signs[:, [0, 2]])
    with pytest.raises(QonnxImportError):          # a padded tap would read -1 in the flipped channels: not exact
        Importer._step4_flip_inputs(None, xv, signs, [1, 1, 1, 1], SimpleNamespace(name="c", op_type="Conv"))


def test_import_options_in_the_file():
    rng = np.random.default_rng(5)
    g0, c64 = _dabnn_like(rng)
    m = to_model(IRWriter(g0, ir_shapes(g0), c64).write(), KEEP)
    import argparse
    args = argparse.Namespace(naming="index", affine="fold", constant_order="walk", cleanup="qonnx", bit_outputs=None, flatten_dense="transpose")
    opts = _step4_resolve_args(args, m, explicit=set())
    assert (args.naming, args.affine, args.constant_order, args.cleanup) == ("exporter", "keep", "first-use", "none") and opts["bit_outputs"] == "none"
    args = argparse.Namespace(naming="index", affine="fold", constant_order="walk", cleanup="qonnx", bit_outputs=None, flatten_dense="transpose")
    with pytest.raises(SystemExit):
        _step4_resolve_args(args, m, explicit={"naming"})
    bad = to_model(IRWriter(g0, ir_shapes(g0), c64).write(), {**KEEP, "affine": "maybe"})
    assert "E-META-FORMAT" in {f.code for f in check_model(bad).findings}
