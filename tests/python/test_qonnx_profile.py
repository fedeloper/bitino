"""bitino.qonnx_profile: the Bitino QONNX profile v1 checker (docs/qonnx-profile.md) on synthetic graphs.

A complete, valid file (binary conv with an explicit Pad(-1), unit-scale BipolarQuant weights, alpha as a Mul fed by
the float64 carrier Cast(DOUBLE), float head, every metadata key, bitino.name / bitino.order on every node) passes
with no error and no warning; each test then breaks one rule and checks that exactly that finding appears.

Runs in .cache/venv-qonnx (needs onnx):  .cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_profile.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from bitino import qonnx_profile as qp  # noqa: E402

QD = qp.QONNX_DOMAIN
SHA = "a" * 64

META = {
    "bitino.profile": qp.PROFILE_ID,
    "bitino.model_id": "test-bnn-tiny",
    "bitino.variant": "float",
    "bitino.layout": "NCHW",
    "bitino.input_domain": "float32",
    "bitino.preprocessing": json.dumps({"key": "cifar10_imagenet_norm", "definition": "ToTensor then Normalize"}),
    "bitino.labels": json.dumps({"num_classes": 10, "set": "cifar10"}),
    "bitino.source.framework": "pytorch",
    "bitino.source.framework_version": "2.4.1",
    "bitino.source.checkpoint_sha256": SHA,
    "bitino.weights_license": "MIT",
    "bitino.exporter": "test-exporter 1.0",
    "bitino.source.code": "https://example.org/repo@0123456789abcdef0123456789abcdef01234567",
}


class B:
    """Graph builder: every node gets bitino.name = its group (default: its own name) and bitino.order."""

    def __init__(self):
        self.nodes, self.inits, self.k = [], [], 0
        self.order = {}

    def c(self, arr, dtype=np.float32, name=None):
        self.k += 1
        n = name or f"c{self.k}"
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype), n))
        return n

    def op(self, op, ins, group=None, domain="", meta=None, **attrs):
        self.k += 1
        name = f"{op}_{self.k}"
        out = f"t{self.k}"
        nd = helper.make_node(op, ins, [out], name=name, domain=domain, **attrs)
        group = group or name
        md = {"bitino.name": group, "bitino.order": str(self.order.setdefault(group, len(self.order)))}
        md.update(meta or {})
        for k, v in md.items():
            if v is not None:
                nd.metadata_props.add(key=k, value=v)
        self.nodes.append(nd)
        return out

    def model(self, out, out_shape, ir=10, opset=17, meta=None, in_shape=(1, 3, 8, 8)):
        g = helper.make_graph(self.nodes, "g", [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(in_shape))],
                              [helper.make_tensor_value_info(out, TensorProto.FLOAT, out_shape)], self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset), helper.make_opsetid(QD, 1)],
                              producer_name="test", producer_version="1")
        m.ir_version = ir
        for k, v in (META if meta is None else meta).items():
            m.metadata_props.add(key=k, value=v)
        return m


def valid_parts(pad_value=-1.0, conv_pads=(0, 0, 0, 0), w_scale=1.0, w_values=None):
    """input -> affine -> BipolarQuant -> Pad(v) -> Conv(BipolarQuant(W, s)) -> Mul(alpha) -> Add -> Relu -> GAP ->
    Flatten -> Gemm. Returns (builder, output name, output shape)."""
    rng = np.random.default_rng(0)
    b = B()
    x = b.op("Mul", ["input", b.c(rng.uniform(0.5, 2, (1, 3, 1, 1)))], group="input_affine")
    x = b.op("Add", [x, b.c(rng.normal(0, 1, (1, 3, 1, 1)))], group="input_affine")
    x = b.op("BipolarQuant", [x, b.c(1.0)], group="conv0", domain=QD)
    x = b.op("Pad", [x, b.c([0, 0, 1, 1, 0, 0, 1, 1], np.int64), b.c(pad_value)], group="conv0", mode="constant")
    if w_values is None:
        w = b.op("BipolarQuant", [b.c(rng.normal(0, 1, (4, 3, 3, 3))), b.c(w_scale)], group="conv0", domain=QD)
    else:
        w = b.c(w_values)
    x = b.op("Conv", [x, w], group="conv0", pads=list(conv_pads), kernel_shape=[3, 3],
             meta={"bitino.consts": json.dumps({"1": "conv0/w", "2": "conv0/post_mul"}),
                   "bitino.attrs": json.dumps({"pad_value": -1})})
    alpha = b.op("Cast", [b.c(rng.uniform(0.1, 1, (1, 4, 1, 1)), np.float64)], group="conv0", to=TensorProto.FLOAT)
    x = b.op("Mul", [x, alpha], group="conv0")
    x = b.op("Add", [x, b.c(rng.normal(0, 1, (1, 4, 1, 1)))], group="conv0")
    x = b.op("Relu", [x], group="conv0")
    x = b.op("GlobalAveragePool", [x], group="pool")
    x = b.op("Flatten", [x], group="flatten", axis=1)
    x = b.op("Gemm", [x, b.c(rng.normal(0, 1, (10, 4))), b.c(rng.normal(0, 1, 10))], group="fc", transB=1)
    return b, x, [1, 10]


def check(m) -> qp.Report:
    return qp.check_model(m)


def errors(rep) -> set:
    return rep.codes("error")


def warnings(rep) -> set:
    return rep.codes("warning")


def test_valid_file_passes():
    b, out, shp = valid_parts()
    rep = check(b.model(out, shp))
    assert rep.passed(strict=True), rep.to_text()
    assert rep.summary["binarizers"] == {"ge": 1} and rep.summary["binary_layers"] == 1
    assert rep.summary["activation_nodes"] == 11 and rep.summary["named_nodes"] == 13   # + 2 constant-path nodes
    assert "PASS" in rep.to_text()


def test_valid_file_round_trips_through_disk_and_cli(tmp_path, capsys):
    b, out, shp = valid_parts()
    p = tmp_path / "ok.onnx"
    onnx.save(b.model(out, shp), str(p))
    assert qp.main([str(p)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    js = tmp_path / "r.json"
    assert qp.main([str(p), "--json", str(js)]) == 0
    doc = json.load(open(js))
    assert doc["passed"] and doc["summary"]["sha256"] and doc["profile"] == qp.PROFILE_ID
    bad = valid_parts(pad_value=0.5)
    pb = tmp_path / "bad.onnx"
    onnx.save(bad[0].model(bad[1], bad[2]), str(pb))
    assert qp.main([str(pb)]) == 1
    assert qp.main([str(tmp_path / "missing.onnx")]) == 2


def test_cli_subcommand(tmp_path, capsys):
    pytest.importorskip("bitino.cli.main")
    from bitino.cli import main as cli
    b, out, shp = valid_parts()
    p = tmp_path / "ok.onnx"
    onnx.save(b.model(out, shp), str(p))
    assert cli.main(["check-qonnx", str(p)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ container / ops
def test_custom_domain_rejected():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    m.graph.node.append(helper.make_node("ImageGrayU8", [out], ["gray"], name="g", domain="ai.bitino"))
    m.opset_import.append(helper.make_opsetid("ai.bitino", 1))
    rep = check(m)
    assert "E-DOMAIN" in errors(rep)
    assert any("ai.bitino" in f.message for f in rep.errors)


def test_local_function_rejected():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    m.functions.append(helper.make_function("ai.bitino", "Feat", ["x"], ["y"],
                                            [helper.make_node("Relu", ["x"], ["y"])], [helper.make_opsetid("", 17)]))
    assert "E-FUNCTIONS" in errors(check(m))


@pytest.mark.parametrize("op,domain,code", [("BatchNormalization", "", "E-OP"), ("MultiThreshold", QD, "E-OP"),
                                            ("Quant", QD, "E-OP"), ("TopK", "", "E-OP"), ("Resize", "", "E-OP"),
                                            ("LpNormalization", "", "E-OP"), ("DFT", "", "E-OP")])
def test_rejected_and_unknown_ops(op, domain, code):
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    m.graph.node.append(helper.make_node(op, [out], ["extra"], name="extra", domain=domain))
    rep = check(m)
    assert code in errors(rep)
    assert any(op in f.message for f in rep.errors if f.code == code)


def test_op_not_in_declared_opset():
    b, out, shp = valid_parts()
    out = b.op("Gelu", [out], group="gelu")
    rep = check(b.model(out, shp))                   # Gelu exists from opset 20
    assert "E-OP-OPSET" in errors(rep)
    b2, out2, shp2 = valid_parts()
    out2 = b2.op("Gelu", [out2], group="gelu")
    assert "E-OP-OPSET" not in errors(check(b2.model(out2, shp2, opset=20)))


def test_ir_and_opset_versions():
    b, out, shp = valid_parts()
    rep = check(b.model(out, shp, ir=8, opset=13))
    assert {"E-IR", "E-OPSET"} <= errors(rep)


def test_undeclared_and_legacy_domains():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    del m.opset_import[1]                            # qonnx domain used but not imported
    assert "E-DOMAIN-UNDECLARED" in errors(check(m))
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    m.graph.node[2].domain = "onnx.brevitas"
    m.opset_import.append(helper.make_opsetid("onnx.brevitas", 1))
    assert "E-DOMAIN" in errors(check(m))


def test_external_data_and_initializer_inputs():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    m.graph.input.append(helper.make_tensor_value_info(m.graph.initializer[0].name, TensorProto.FLOAT, [1, 3, 1, 1]))
    assert "E-INPUT-INIT" in errors(check(m))
    m = b.model(out, shp)
    t = m.graph.initializer[0]
    t.data_location = TensorProto.EXTERNAL
    t.external_data.add(key="location", value="w.bin")
    assert "E-EXTERNAL" in errors(qp.check_model(m))


def test_dynamic_input_shape_rejected():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    m.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    assert "E-INPUT-SHAPE" in errors(check(m))


def test_shapes_must_be_inferable():
    b, out, shp = valid_parts()
    out = b.op("Add", [out, b.c(np.zeros((1, 7), np.float32))], group="bad_add")   # [1,10] + [1,7]
    rep = check(b.model(out, shp))
    assert "E-SHAPES" in errors(rep)


# ------------------------------------------------------------------------------------------------ attributes
def test_auto_pad_and_ceil_mode_rejected():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    conv = next(n for n in m.graph.node if n.op_type == "Conv")
    conv.attribute.append(helper.make_attribute("auto_pad", "SAME_UPPER"))
    assert "E-AUTOPAD" in errors(check(m))
    b, out, shp = valid_parts()
    x = b.op("MaxPool", [b.nodes[-4].output[0]], group="mp", kernel_shape=[2, 2], ceil_mode=1)
    assert "E-CEIL" in errors(check(b.model(out, shp)))


def test_intquant_rounding_and_zero_point():
    b, out, shp = valid_parts()
    b.op("IntQuant", [b.nodes[0].output[0], b.c(0.25), b.c(0.0), b.c(8.0)], group="q", domain=QD,
         signed=1, narrow=0, rounding_mode="FLOOR")
    assert "E-ROUNDING" in errors(check(b.model(out, shp)))
    b, out, shp = valid_parts()
    b.op("IntQuant", [b.nodes[0].output[0], b.c(0.25), b.c(3.0), b.c(8.0)], group="q", domain=QD,
         signed=1, narrow=0, rounding_mode="ROUND")
    assert "E-INTQUANT-ZP" in errors(check(b.model(out, shp)))
    b, out, shp = valid_parts()                       # ROUND, zero point 0: fine
    b.op("IntQuant", [b.nodes[0].output[0], b.c(0.25), b.c(0.0), b.c(8.0)], group="q", domain=QD,
         signed=1, narrow=0, rounding_mode="ROUND")
    assert not errors(check(b.model(out, shp)))


def test_gemm_scaling_rejected():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    next(n for n in m.graph.node if n.op_type == "Gemm").attribute.append(helper.make_attribute("alpha", 0.5))
    assert "E-ATTR" in errors(check(m))


def test_int64_clamp_rejected():
    b = B()
    q = b.op("Cast", [b.op("Round", ["input"])], to=TensorProto.INT64)
    y = b.op("Clip", [q, b.c(-2 ** 31, np.int64), b.c(2 ** 31 - 1, np.int64)])
    y = b.op("Cast", [y], to=TensorProto.FLOAT)
    assert "E-INT64-CLAMP" in errors(check(b.model(y, [1, 3, 8, 8])))
    b = B()                                           # the exact form: Where(Greater / Less)
    q = b.op("Cast", [b.op("Round", ["input"])], to=TensorProto.INT64)
    hi, lo = b.c(2 ** 31 - 1, np.int64), b.c(-2 ** 31, np.int64)
    y = b.op("Where", [b.op("Greater", [q, hi]), hi, q])
    y = b.op("Where", [b.op("Less", [y, lo]), lo, y])
    y = b.op("Cast", [y], to=TensorProto.FLOAT)
    assert check(b.model(y, [1, 3, 8, 8])).passed(strict=True)


def test_convinteger_padding_must_be_explicit():
    b = B()
    q = b.op("Cast", [b.op("Round", ["input"])], to=TensorProto.INT8)
    y = b.op("ConvInteger", [q, b.c(np.ones((2, 3, 3, 3)), np.int8), b.c(0, np.int8)], pads=[1, 1, 1, 1],
             kernel_shape=[3, 3])
    y = b.op("Cast", [y], to=TensorProto.FLOAT)
    rep = check(b.model(y, [1, 2, 8, 8]))
    assert "E-CONVINT-PAD" in errors(rep)


# ------------------------------------------------------------------------------------------------ binary conventions
def test_implicit_binary_padding_rejected():
    b, out, shp = valid_parts(conv_pads=(1, 1, 1, 1))
    rep = check(b.model(out, [1, 10]))
    assert "E-BINPAD-IMPLICIT" in errors(rep)


@pytest.mark.parametrize("value", [0.5, 2.0])
def test_binary_pad_value_must_be_pm1_or_0(value):
    b, out, shp = valid_parts(pad_value=value)
    assert "E-BINPAD-VALUE" in errors(check(b.model(out, shp)))


def test_binary_pad_value_must_be_explicit():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    pad = next(n for n in m.graph.node if n.op_type == "Pad")
    del pad.input[2]
    assert "E-BINPAD-VALUE" in errors(check(m))


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_binary_pad_values_accepted(value):
    b, out, shp = valid_parts(pad_value=value)
    assert not errors(check(b.model(out, shp)))


def test_alpha_inside_weights_warns():
    b, out, shp = valid_parts(w_scale=0.25)
    rep = check(b.model(out, shp))
    assert not errors(rep) and "W-ALPHA-INSIDE" in warnings(rep) and not rep.passed(strict=True)
    rng = np.random.default_rng(1)
    alpha = rng.uniform(0.1, 1, (4, 1, 1, 1))
    w = (np.where(rng.random((4, 3, 3, 3)) < 0.5, -1, 1) * alpha).astype(np.float32)
    b, out, shp = valid_parts(w_values=w)
    assert "W-ALPHA-INSIDE" in warnings(check(b.model(out, shp)))
    w[1, 0, 0, 0] *= 0.5                              # not +-alpha on channel 1
    b, out, shp = valid_parts(w_values=w)
    assert "W-FLOAT-WEIGHTS" in warnings(check(b.model(out, shp)))
    b, out, shp = valid_parts(w_values=np.sign(w))    # +-1 constants: canonical
    assert not check(b.model(out, shp)).findings or not warnings(check(b.model(out, shp)))


def test_plain_ternary_weight_constant_accepted():
    """R4 (step 6): ternary weights (w_zero) as a plain {-1, 0, +1} constant pass with an info, no warning; a constant
    with any other value (e.g. 0.5, or 2) still warns W-FLOAT-WEIGHTS."""
    rng = np.random.default_rng(3)
    w = rng.integers(-1, 2, (4, 3, 3, 3)).astype(np.float32)
    w[0, 0, 0, 0], w[1, 0, 0, 0] = 0.0, 1.0               # zeros and both signs present
    b, out, shp = valid_parts(w_values=w)
    rep = check(b.model(out, shp))
    assert rep.passed(strict=True), rep.to_text()
    assert "I-TERNARY" in rep.codes("info") and "W-FLOAT-WEIGHTS" not in warnings(rep)
    for bad in (0.5, 2.0):
        w2 = w.copy()
        w2[2, 0, 0, 0] = bad
        assert "W-FLOAT-WEIGHTS" in warnings(check(valid_parts(w_values=w2)[0].model(out, shp)))


def test_binarizer_patterns_recognized():
    b = B()
    x = b.op("Sign", ["input"], group="s")
    g = b.op("Relu", [x], group="s")                                     # {0,1}: gt0
    t = b.op("GreaterOrEqual", ["input", b.c(np.zeros((1, 3, 1, 1)))], group="w")
    w = b.op("Where", [t, b.c(1.0), b.c(-1.0)], group="w")               # ge with threshold
    y = b.op("Add", [g, w], group="sum")
    rep = check(b.model(y, [1, 3, 8, 8]))
    assert rep.summary["binarizers"] == {"sign_zero": 1, "gt0": 1, "ge": 1}
    assert not errors(rep), rep.to_text()


def test_binarizer_scale_must_be_positive():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    bq = next(n for n in m.graph.node if n.op_type == "BipolarQuant" and n.input[0].startswith("t"))
    s = next(t for t in m.graph.initializer if t.name == bq.input[1])
    s.CopyFrom(numpy_helper.from_array(np.asarray(-1.0, np.float32), s.name))
    assert "E-BINARIZER-SCALE" in errors(check(m))


# ------------------------------------------------------------------------------------------------ metadata
def test_missing_metadata():
    b, out, shp = valid_parts()
    meta = {k: v for k, v in META.items() if k not in ("bitino.source.checkpoint_sha256", "bitino.weights_license")}
    rep = check(b.model(out, shp, meta=meta))
    f = next(f for f in rep.errors if f.code == "E-META-MISSING")
    assert "bitino.source.checkpoint_sha256" in f.message and "bitino.weights_license" in f.message


@pytest.mark.parametrize("key,value", [
    ("bitino.profile", "bitino-qonnx/0"),
    ("bitino.model_id", "Bad Id"),
    ("bitino.variant", "int4"),
    ("bitino.layout", "NHWC"),
    ("bitino.input_domain", "bytes"),
    ("bitino.input_domain", json.dumps({"other": "float32"})),
    ("bitino.preprocessing", "torchvision_imagenet"),
    ("bitino.labels", json.dumps({"num_classes": 0, "set": "x"})),
    ("bitino.labels", "10"),
    ("bitino.source.checkpoint_sha256", "abc"),
    ("bitino.source.checkpoint_sha256", json.dumps({"w.pth": "xyz"})),
    ("bitino.exporter", " "),
    ("bitino.const_order", json.dumps(["a", "a"])),
])
def test_malformed_metadata(key, value):
    b, out, shp = valid_parts()
    rep = check(b.model(out, shp, meta={**META, key: value}))
    assert "E-META-FORMAT" in errors(rep), rep.to_text()
    assert any(key in f.message for f in rep.errors)


def test_metadata_variants_accepted():
    b, out, shp = valid_parts()
    meta = {**META, "bitino.labels": "none", "bitino.input_domain": json.dumps({"input": "unit8"}),
            "bitino.source.checkpoint_sha256": json.dumps({"a.pth": SHA, "b.pth": "b" * 64}),
            "bitino.const_order": json.dumps(["conv0/w", "conv0/post_mul"]), "other.key": "free"}
    assert check(b.model(out, shp, meta=meta)).passed(strict=True)


def test_int8_full_needs_calibration():
    b, out, shp = valid_parts()
    rep = check(b.model(out, shp, meta={**META, "bitino.variant": "int8_full"}))
    assert any("bitino.calibration" in f.message for f in rep.errors if f.code == "E-META-MISSING")
    cal = {"set": "cifar10-train-calib-512", "split": "train", "n": 128, "tensor_sha256": SHA}
    rep = check(b.model(out, shp, meta={**META, "bitino.variant": "int8_full", "bitino.calibration": json.dumps(cal)}))
    assert rep.passed(strict=True)
    rep = check(b.model(out, shp, meta={**META, "bitino.variant": "int8_full",
                                         "bitino.calibration": json.dumps({**cal, "split": "val"})}))
    assert "W-META" in warnings(rep)


def test_unknown_metadata_key_warns():
    b, out, shp = valid_parts()
    rep = check(b.model(out, shp, meta={**META, "bitino.modelid": "x"}))
    assert "W-META-UNKNOWN" in warnings(rep) and not errors(rep)


# ------------------------------------------------------------------------------------------------ naming / order
def _strip(m, idx, key):
    nd = m.graph.node[idx]
    keep = [(p.key, p.value) for p in nd.metadata_props if p.key != key]
    del nd.metadata_props[:]
    for k, v in keep:
        nd.metadata_props.add(key=k, value=v)


def test_missing_node_name():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    _strip(m, 5, "bitino.order")
    _strip(m, 5, "bitino.name")
    rep = check(m)
    assert "E-NAME-MISSING" in errors(rep)
    f = next(f for f in rep.errors if f.code == "E-NAME-MISSING")
    assert f.nodes == [m.graph.node[5].name]


def test_constant_path_nodes_need_no_name():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    for i, nd in enumerate(m.graph.node):
        if nd.op_type in ("Cast",) or (nd.op_type == "BipolarQuant" and not nd.input[0].startswith("t")):
            _strip(m, i, "bitino.order")
            _strip(m, i, "bitino.name")
    assert check(m).passed(strict=True)


def test_order_rules():
    b, out, shp = valid_parts()
    m = b.model(out, shp)
    _strip(m, 0, "bitino.order")                   # group input_affine: one node without order -> inconsistent
    assert "E-ORDER" in errors(check(m))
    m = b.model(out, shp)
    for i in range(len(m.graph.node)):            # none at all: fine (ONNX order)
        _strip(m, i, "bitino.order")
    assert check(m).passed(strict=True)
    m = b.model(out, shp)
    m.graph.node[-1].metadata_props[1].value = "0"   # fc reuses input_affine's order
    assert "E-ORDER" in errors(check(m))
    m = b.model(out, shp)
    m.graph.node[-1].metadata_props[1].value = "x1"
    assert "E-NAME-FORMAT" in errors(check(m))


def test_extra_names_consts_attrs():
    b, out, shp = valid_parts()
    out = b.op("Flatten", [out], group="tail", axis=1,
               meta={"bitino.extra_names": json.dumps(["tail/transpose"]),
                     "bitino.consts": json.dumps({"tail/transpose#1": "tail/perm"}),
                     "bitino.extra_attrs": json.dumps({"tail/transpose": {"perm": [0, 1]}})})
    assert check(b.model(out, shp)).passed(strict=True)
    for meta in ({"bitino.extra_names": "tail/transpose"},                         # not a JSON list
                 {"bitino.extra_names": json.dumps(["conv0"])},                    # collides with a group
                 {"bitino.consts": json.dumps({"w": "x"})},                        # key is not an input index
                 {"bitino.consts": json.dumps({"1": "conv0/w"})},                  # constant name used twice
                 {"bitino.extra_attrs": json.dumps({"nope": {"a": 1}})},           # not one of its extra names
                 {"bitino.attrs": "[1]"}):
        b, out, shp = valid_parts()
        out = b.op("Flatten", [out], group="tail", axis=1, meta=meta)
        rep = check(b.model(out, shp))
        assert errors(rep) & {"E-NAME-FORMAT", "E-NAME-DUP"}, (meta, rep.to_text())


def test_unknown_node_key_warns():
    b, out, shp = valid_parts()
    out = b.op("Relu", [out], group="r", meta={"bitino.nmae": "x"})
    assert "W-NAME-UNKNOWN" in warnings(check(b.model(out, shp)))


# ------------------------------------------------------------------------------------------------ step 3 (group mapping)
def test_import_mapping_and_ir_node_key():
    """bitino.import = {"mapping": "groups"} (model) and bitino.ir = {"op", "n_inputs", "layouts"} (node), step 3:
    accepted without warnings; malformed values are E-META-FORMAT / E-NAME-FORMAT."""
    b, out, shp = valid_parts()
    for nd in b.nodes:
        if nd.op_type == "Conv":
            nd.metadata_props.add(key="bitino.ir", value=json.dumps({"op": "BConv2D", "n_inputs": 5, "layouts": {"1": "OHWI_bit"}}))
    meta = dict(META, **{"bitino.import": json.dumps({"mapping": "groups"})})
    rep = check(b.model(out, shp, meta=meta))
    assert rep.passed(strict=True), rep.to_text()
    rep = check(b.model(out, shp, meta=dict(META, **{"bitino.import": json.dumps({"mapping": "other"})})))
    assert "E-META-FORMAT" in errors(rep)
    for bad in ('{"n_inputs": 2}', '{"op": "Conv2D", "n_inputs": -1}', '{"op": "Conv2D", "layouts": {"x": "OHWI"}}', '[1]'):
        b2, out2, shp2 = valid_parts()
        next(nd for nd in b2.nodes if nd.op_type == "Conv").metadata_props.add(key="bitino.ir", value=bad)
        assert "E-NAME-FORMAT" in errors(check(b2.model(out2, shp2, meta=meta))), bad


def test_shared_constant_name_only_with_group_mapping():
    """One IR constant used by several IR nodes (a shared mask, an index table) is named by each group: allowed
    with the group mapping (the importer requires equal data), an E-NAME-DUP otherwise."""
    b, out, shp = valid_parts()
    gemm = next(nd for nd in b.nodes if nd.op_type == "Gemm")
    gemm.metadata_props.add(key="bitino.consts", value=json.dumps({"1": "conv0/w"}))
    assert "E-NAME-DUP" in errors(check(b.model(out, shp)))
    meta = dict(META, **{"bitino.import": json.dumps({"mapping": "groups"})})
    assert "E-NAME-DUP" not in errors(check(b.model(out, shp, meta=meta)))


# ------------------------------------------------------------------------------------------------ step 1b
def _float_layer_model(declare: str | None, w):
    b = B()
    x = b.op("Sign", ["input"], group="q")
    x = b.op("Pad", [x, b.c([0, 0, 1, 1, 0, 0, 1, 1], np.int64), b.c(0.0)], group="q", mode="constant")
    meta = {"bitino.extra_names": json.dumps(["q/sign"])}
    if declare is not None:
        meta["bitino.float_layer"] = declare
    x = b.op("Conv", [x, b.c(w)], group="q", pads=[0, 0, 0, 0], kernel_shape=[3, 3], meta=meta)
    x = b.op("GlobalAveragePool", [x], group="pool")
    return b.model(x, [1, 4, 1, 1])


def test_float_layer_declaration():
    """§ 7 bitino.float_layer: float weights on a binarized tensor warn (W-FLOAT-WEIGHTS) unless declared (then info);
    a declaration on +-alpha weights, with another value, or on a layer without a binarized input is an error."""
    rng = np.random.default_rng(7)
    w = rng.normal(0, 1, (4, 3, 3, 3))
    assert "W-FLOAT-WEIGHTS" in warnings(check(_float_layer_model(None, w)))
    rep = check(_float_layer_model("1", w))
    assert not warnings(rep) and not errors(rep) and "I-FLOAT-LAYER" in rep.codes("info")
    assert "E-FLOAT-LAYER" in errors(check(_float_layer_model("1", np.sign(w) * 0.5)))
    assert "E-FLOAT-LAYER" in errors(check(_float_layer_model("yes", w)))
    b = B()
    x = b.op("Conv", ["input", b.c(w)], group="c", pads=[1, 1, 1, 1], kernel_shape=[3, 3], meta={"bitino.float_layer": "1"})
    assert "E-FLOAT-LAYER" in errors(check(b.model(x, [1, 4, 8, 8])))


def _adabin_model(beta_add=0.25, alpha=0.7, levels=2, pad_value=None):
    rng = np.random.default_rng(8)
    b = B()
    x = b.op("Sub", ["input", b.c(0.25)], group="ab")
    x = b.op("Sign", [x], group="ab")
    x = b.op("Add", [x, b.c(beta_add)], group="ab")
    x = b.op("Mul", [x, b.c(alpha)], group="ab")
    pads = [1, 1, 1, 1]
    if pad_value is not None:
        x = b.op("Pad", [x, b.c([0, 0, 1, 1, 0, 0, 1, 1], np.int64), b.c(pad_value)], group="ab", mode="constant")
        pads = [0, 0, 0, 0]
    s = np.where(rng.random((4, 3, 3, 3)) < 0.5, 1.0, -1.0)
    w = np.where(s > 0, 0.3, -0.2)
    if levels == 3:
        w[0, 0, 0, 0] = 0.05
    x = b.op("Conv", [x, b.c(w)], group="ab", pads=pads, kernel_shape=[3, 3])
    x = b.op("GlobalAveragePool", [x], group="pool")
    return b.model(x, [1, 4, 1, 1])


def test_adabin_pattern_conditions():
    """§ 4.1 AdaBin binary conv: Conv(Mul(Add(Sign(Sub(x, beta)), beta), alpha), two-level W) is recognized (I-ADABIN);
    a beta mismatch, a non-positive alpha, a third weight level or a non-zero Pad value is E-ADABIN."""
    rep = check(_adabin_model())
    assert not errors(rep) and not warnings(rep) and "I-ADABIN" in rep.codes("info")
    assert rep.summary.get("adabin_convs") == 1
    assert not errors(check(_adabin_model(pad_value=0.0)))
    for kw in ({"beta_add": 0.5}, {"alpha": -0.7}, {"levels": 3}, {"pad_value": 1.0}):
        assert "E-ADABIN" in errors(check(_adabin_model(**kw))), kw
