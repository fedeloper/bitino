"""bitino.qonnx_import + bitino.onnx_f64 on small synthetic QONNX graphs (docs/qonnx-frontend.md phase 1).

Each test builds an ONNX graph with onnx.helper, imports it, and compares the numpy executor of the IR (float64)
with the float64 evaluation of the source graph (OnnxF64). Covered: the binarizers and their sign(0) conventions
(BipolarQuant ge, Sign sign_zero, Where(GreaterOrEqual / Greater), MultiThreshold BIPOLAR), the binary padding rules
(Conv zero padding = pad_value 0, explicit Pad(±1) = pad_value ±1, both together rejected), the ±alpha weight rule
(per-channel alpha accepted, anything else rejected, never rounded), BN -> integer thresholds, IntQuant -> FakeQuant
and the W1A8 conv, the input threshold fold, the MultiThreshold input quantizer on a uint8 domain, the float set
(Conv2D, Dense, Relu, Add, GlobalAveragePool), NCHW flatten, trailing TopK, and the precise error of an unmatched node.

Runs in .cache/venv-qonnx (needs onnx; qonnx optional):  .cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_import.py
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

from bitino import executor, format as bfformat  # noqa: E402
from bitino.onnx_f64 import OnnxF64  # noqa: E402
from bitino.qonnx_import import Domain, QonnxImportError, import_model  # noqa: E402

QD = "qonnx.custom_op.general"


class G:
    """Tiny graph builder: chain-style helpers returning tensor names."""

    def __init__(self, in_shape):
        self.nodes, self.inits, self.k = [], [], 0
        self.in_shape = list(in_shape)

    def name(self, p="t"):
        self.k += 1
        return f"{p}{self.k}"

    def c(self, arr, dtype=np.float32):
        n = self.name("c")
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype), n))
        return n

    def op(self, op, ins, domain="", n_out=1, **attrs):
        outs = [self.name(op.lower()) for _ in range(n_out)]
        self.nodes.append(helper.make_node(op, ins, outs, name=self.name(op), domain=domain, **attrs))
        return outs[0] if n_out == 1 else outs

    def bq(self, x, s=1.0):
        return self.op("BipolarQuant", [x, self.c(s)], domain=QD)

    def bn(self, x, c, rng, neg=True):
        gam = rng.uniform(0.5, 2.0, c) * (np.where(rng.random(c) < 0.3, -1, 1) if neg else 1)
        return self.op("BatchNormalization", [x, self.c(gam), self.c(rng.normal(0, 1, c)), self.c(rng.normal(0, 2, c)),
                                              self.c(rng.uniform(0.5, 3.0, c))], epsilon=1e-4)

    def model(self, out, out_shape=None, extra_outputs=()):
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, self.in_shape)
        outs = [helper.make_tensor_value_info(out, TensorProto.FLOAT, out_shape)] + list(extra_outputs)
        g = helper.make_graph(self.nodes, "t", [inp], outs, self.inits)
        return helper.make_model(g, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(QD, 1)], producer_name="test")


def run_both(model, x_nchw, domain="float32"):
    g, info = import_model(model, Domain.parse(domain))
    ev = OnnxF64(model)
    ref = [np.asarray(ev.run(xi[None], np.float64)[ev.output_names[0] if not info["dropped"] else info["logits_onnx_tensor"]],
                      np.float64).reshape(-1) if not info["dropped"] else
           np.asarray(ev.run(xi[None], np.float64, outputs=[info["logits_onnx_tensor"]])[info["logits_onnx_tensor"]]).reshape(-1)
           for xi in x_nchw]
    xs = [np.transpose(xi[None], (0, 2, 3, 1)) if xi.ndim == 3 else xi[None] for xi in x_nchw]
    ir = [executor.run(g, {"input": xi}, float_dtype=np.float64)[g.outputs[0]].reshape(-1) for xi in xs]
    ir32 = [executor.run(g, {"input": xi})[g.outputs[0]].reshape(-1) for xi in xs]
    return g, info, np.stack(ref), np.stack(ir), np.stack(ir32)


def node(g, op):
    return [n for n in g.nodes if n.op == op]


def roundtrip(g, tmp_path):
    p = str(tmp_path / "m.bitino")
    bfformat.write(g, p)
    return bfformat.read(p)


# ------------------------------------------------------------------------------------------------ binarizers
def _dense_net(binarizer, rng, k=24, o=16, alpha=None, zeros=True):
    gb = G([1, k])
    x = "x"
    if binarizer == "bq":
        b0 = gb.bq(x)
    elif binarizer == "sign":
        b0 = gb.op("Sign", [x])
    elif binarizer in ("ge", "gt"):
        t = gb.c(rng.normal(0, 0.3, (1, k)))
        cmp = gb.op("GreaterOrEqual" if binarizer == "ge" else "Greater", [x, t])
        b0 = gb.op("Where", [cmp, gb.c(1.0), gb.c(-1.0)])
    elif binarizer == "mt":
        b0 = gb.op("MultiThreshold", [x, gb.c(np.zeros((1, 1)))], domain=QD, out_scale=2.0, out_bias=-1.0, out_dtype="BIPOLAR")
    w = np.where(rng.random((k, o)) < 0.5, -1.0, 1.0)
    if alpha is not None:
        w = w * alpha[None, :]
    h = gb.op("MatMul", [b0, gb.c(w)])
    h = gb.bn(h, o, rng)
    h = gb.bq(h)
    w2 = gb.bq(gb.c(rng.normal(0, 1, (o, 10))))          # latent weights -> Bin(W)
    y = gb.op("MatMul", [h, w2])
    y = gb.op("Add", [gb.op("Mul", [y, gb.c(0.37)]), gb.c(rng.normal(0, 1, 10))])
    return gb.model(y, [1, 10])


@pytest.mark.parametrize("binarizer,mode", [("bq", "ge"), ("sign", "sign_zero"), ("ge", "ge"), ("gt", "gt"), ("mt", "ge")])
def test_binarizers_and_sign0(binarizer, mode):
    rng = np.random.default_rng(1)
    m = _dense_net(binarizer, rng)
    x = rng.normal(0, 1, (12, 24)).astype(np.float32)
    x[:, :6] = 0.0                       # exact zeros: ge -> +1, sign_zero -> 0
    if binarizer in ("ge", "gt"):
        t = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == m.graph.node[0].input[1])).reshape(-1)
        x[:, 6:12] = t[6:12]             # exactly at the threshold: ge -> +1, gt -> -1
    g, info, ref, ir, _ = run_both(m, list(x))
    d = node(g, "BDense")
    assert len(d) == 2 and d[0].attrs["input_binarize"] == mode
    assert info["rules"].get("BN/affine + binarizer -> integer threshold (binary accumulator)") == 1
    # hidden layer: exact integer threshold (post_mul = ±1); logits within float32 rounding of the final affine
    assert set(np.unique(g.constants[d[0].inputs[2]].data)) <= {-1.0, 1.0}
    np.testing.assert_allclose(ir, ref, rtol=2e-6, atol=2e-6)


def test_sign_zero_is_not_ge():
    """The same graph with Sign and with BipolarQuant differs on zero inputs: the importer keeps each convention."""
    rng = np.random.default_rng(2)
    x = np.zeros((1, 24), np.float32)
    outs = {}
    for b in ("bq", "sign"):
        rng = np.random.default_rng(2)
        _, _, ref, ir, _ = run_both(_dense_net(b, rng), list(x))
        np.testing.assert_allclose(ir, ref, rtol=2e-6, atol=2e-6)
        outs[b] = ir
    assert not np.allclose(outs["bq"], outs["sign"])


def test_per_channel_alpha_accepted_and_folded():
    rng = np.random.default_rng(3)
    alpha = rng.uniform(0.1, 2.0, 16)
    m = _dense_net("bq", rng, alpha=alpha)
    x = rng.normal(0, 1, (8, 24)).astype(np.float32)
    g, info, ref, ir, _ = run_both(m, list(x))
    assert info["layer_precision"][0]["weights"].endswith("alpha[o]")
    np.testing.assert_allclose(ir, ref, rtol=2e-6, atol=2e-6)


def test_non_alpha_weights_rejected_not_rounded():
    rng = np.random.default_rng(4)
    gb = G([1, 8])
    w = np.where(rng.random((8, 4)) < 0.5, -1.0, 1.0)
    w[3, 2] = 0.999                      # channel 2 is not ±alpha
    y = gb.op("MatMul", [gb.bq("x"), gb.c(w)])
    with pytest.raises(QonnxImportError) as e:
        import_model(gb.model(y, [1, 4]))
    msg = str(e.value)
    assert "weight channel 2 is not ±alpha" in msg and "MatMul" in msg and "never rounded" in msg


def test_unmatched_node_error_is_precise():
    gb = G([1, 8])
    h = gb.op("Exp", [gb.op("Relu", ["x"])])           # evaluable, but no IR pattern
    with pytest.raises(QonnxImportError) as e:
        import_model(gb.model(h, [1, 8]))
    assert e.value.op == "Exp" and e.value.node_name.startswith("Exp") and "no pattern matches" in str(e.value)
    gb = G([1, 8])
    h = gb.op("Hardmax", [gb.op("Relu", ["x"])])       # not even evaluable
    with pytest.raises(QonnxImportError) as e:
        import_model(gb.model(h, [1, 8]))
    assert e.value.op == "Hardmax" and "no float64 reference" in str(e.value)


# ------------------------------------------------------------------------------------------------ padding
def _conv_net(rng, pad_kind, c=5, o=6, hw=6):
    gb = G([1, c, hw, hw])
    b = gb.bq("x")
    conv_pads = [0, 0, 0, 0]
    if pad_kind == "conv0":
        conv_pads = [1, 1, 1, 1]
    elif pad_kind in ("pad+1", "pad-1", "both"):
        v = 1.0 if pad_kind != "pad-1" else -1.0
        b = gb.op("Pad", [b, gb.c(np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.int64), gb.c(v)], mode="constant")
        if pad_kind == "both":
            conv_pads = [1, 1, 1, 1]
    w = gb.c(np.where(rng.random((o, c, 3, 3)) < 0.5, -1.0, 1.0))
    h = gb.op("Conv", [b, w], kernel_shape=[3, 3], pads=conv_pads, strides=[1, 1])
    h = gb.bq(gb.bn(h, o, rng))
    h = gb.op("Conv", [h, gb.c(np.where(rng.random((4, o, 3, 3)) < 0.5, -1.0, 1.0))], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    y = gb.op("GlobalAveragePool", [h])
    y = gb.op("Flatten", [y], axis=1)
    return gb.model(y, [1, 4])


@pytest.mark.parametrize("pad_kind,pad_value", [("none", 0), ("conv0", 0), ("pad+1", 1), ("pad-1", -1)])
def test_binary_padding_rules(pad_kind, pad_value):
    rng = np.random.default_rng(5)
    m = _conv_net(rng, pad_kind)
    x = list(rng.normal(0, 1, (6, 5, 6, 6)).astype(np.float32))
    g, info, ref, ir, _ = run_both(m, x)
    bc = node(g, "BConv2D")
    assert bc[0].attrs["pad_value"] == pad_value
    assert bc[0].attrs["pads"] == ([0, 0, 0, 0] if pad_kind == "none" else [1, 1, 1, 1])
    assert bc[1].attrs["pad_value"] == 0 and bc[1].attrs["pads"] == [1, 1, 1, 1]
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_pad_and_conv_padding_together_rejected():
    rng = np.random.default_rng(6)
    with pytest.raises(QonnxImportError) as e:
        import_model(_conv_net(rng, "both"))
    assert "Pad" in str(e.value) and "Conv" in str(e.value)


# ------------------------------------------------------------------------------------------------ IntQuant / W1A8 / input path
def _cnv_like(rng, input_q="quant"):
    gb = G([1, 3, 8, 8])
    x = gb.op("Sub", [gb.op("Mul", ["x", gb.c(2.0)]), gb.c([1.0])])
    if input_q == "quant":
        x = gb.op("Quant", [x, gb.c(1 / 128), gb.c(0.0), gb.c(8.0)], domain=QD, signed=1, narrow=0, rounding_mode="ROUND")
    else:
        x = gb.op("Quant", [x, gb.c(1 / 128), gb.c(0.0), gb.c(8.0)], domain=QD, signed=1, narrow=0, rounding_mode=input_q)
    h = gb.op("Conv", [x, gb.bq(gb.c(rng.normal(0, 1, (8, 3, 3, 3))))], kernel_shape=[3, 3])
    h = gb.bq(gb.bn(h, 8, rng))
    h = gb.op("Conv", [h, gb.bq(gb.c(rng.normal(0, 1, (8, 8, 3, 3))))], kernel_shape=[3, 3])
    h = gb.bq(gb.bn(h, 8, rng))
    h = gb.op("MaxPool", [h], kernel_shape=[2, 2], strides=[2, 2])
    h = gb.op("Conv", [h, gb.bq(gb.c(rng.normal(0, 1, (8, 8, 2, 2))))], kernel_shape=[2, 2])
    h = gb.bq(gb.bn(h, 8, rng))
    h = gb.op("Reshape", [h, gb.c(np.array([1, -1]), np.int64)])
    y = gb.op("MatMul", [h, gb.op("Transpose", [gb.bq(gb.c(rng.normal(0, 1, (5, 8))))], perm=[1, 0])])
    y = gb.op("Div", [gb.op("Sub", [y, gb.c([0.3])]), gb.op("Pow", [gb.c([4.0]), gb.c(0.5)])])
    return gb.model(y, [1, 5])


def test_intquant_w1a8_and_integer_thresholds(tmp_path):
    rng = np.random.default_rng(7)
    m = _cnv_like(rng)
    x = list(rng.integers(0, 256, (10, 3, 8, 8)).astype(np.float32) / np.float32(255))
    g, info, ref, ir, ir32 = run_both(m, x, "unit8")
    ops = g.op_counts()
    assert ops == {"ChannelAffine": 1, "FakeQuant": 1, "Conv2D": 1, "BConv2D": 2, "MaxPool2D": 1, "Reshape": 1, "BDense": 1}
    conv = node(g, "Conv2D")[0]
    assert conv.inputs[1].endswith("kernel_signed") and conv.inputs[2].endswith("bias")
    bc = node(g, "BConv2D")
    assert bc[0].attrs["output"] == "f32" and bc[1].attrs["output"] == "f32"   # pool / flatten follow
    fq = node(g, "FakeQuant")[0]
    assert (fq.attrs["qmin"], fq.attrs["qmax"]) == (-128, 127)
    assert np.all(g.constants[fq.inputs[1]].data == np.float32(1 / 128))
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)
    g2 = roundtrip(g, tmp_path)
    xs = np.transpose(np.stack(x), (0, 2, 3, 1))
    assert np.array_equal(np.stack([executor.run(g2, {"input": xi[None]})[g2.outputs[0]].reshape(-1) for xi in xs]), ir32)


def test_bit_output_between_binary_convs():
    rng = np.random.default_rng(8)
    gb = G([1, 4, 7, 7])
    h = gb.bq("x")
    for _ in range(3):
        h = gb.bq(gb.bn(gb.op("Conv", [h, gb.c(np.where(rng.random((4, 4, 3, 3)) < 0.5, -1.0, 1.0))], kernel_shape=[3, 3],
                              pads=[1, 1, 1, 1]), 4, rng))
    y = gb.op("Flatten", [gb.op("GlobalAveragePool", [h])])
    m = gb.model(y, [1, 4])
    g, info, ref, ir, _ = run_both(m, list(rng.normal(0, 1, (4, 4, 7, 7)).astype(np.float32)))
    assert [n.attrs["output"] for n in node(g, "BConv2D")] == ["bit", "bit", "f32"]
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_rounding_mode_other_than_round_rejected():
    with pytest.raises(QonnxImportError) as e:
        import_model(_cnv_like(np.random.default_rng(9), input_q="FLOOR"), Domain.parse("unit8"))
    assert "rounding_mode FLOOR" in str(e.value)


def test_input_threshold_fold_exact_boundary():
    """2x - 1 >= 0  <=>  x >= 0.5 exactly in float32 (verified at the neighbouring float32 values)."""
    gb = G([1, 6])
    h = gb.bq(gb.op("Sub", [gb.op("Mul", ["x", gb.c(2.0)]), gb.c([1.0])]))
    y = gb.op("MatMul", [h, gb.c(np.where(np.random.default_rng(10).random((6, 3)) < 0.5, -1.0, 1.0))])
    g, info, ref, ir, _ = run_both(gb.model(y, [1, 3]), [np.array([0.5, np.nextafter(np.float32(0.5), np.float32(0)), 0, 1, 0.25, 0.75],
                                                                  np.float32)])
    d = node(g, "BDense")[0]
    assert np.all(g.constants[d.inputs[4]].data == np.float32(0.5))
    assert np.array_equal(ir, ref)


def test_multithreshold_input_quantizer_uint8_domain():
    """FINN input path: Div 255, Mul 2, Sub 1, MultiThreshold (255 thresholds), Add -128, Mul 1/128 -> ChannelAffine +
    FakeQuant, verified on all 256 pixel values."""
    rng = np.random.default_rng(11)
    gb = G([1, 2, 5, 5])
    x = gb.op("Sub", [gb.op("Mul", [gb.op("Div", ["x", gb.c(255.0)]), gb.c(2.0)]), gb.c([1.0])])
    k = np.arange(255, dtype=np.float64)
    thr = np.tile(((k - 127.5) / 128.0)[None, :], (2, 1)).astype(np.float32)   # round-half boundaries of x*128
    q = gb.op("MultiThreshold", [x, gb.c(thr)], domain=QD, out_dtype="UINT8")
    q = gb.op("Mul", [gb.op("Add", [q, gb.c(-128.0)]), gb.c(1 / 128)])
    h = gb.op("Conv", [q, gb.c(np.where(rng.random((3, 2, 3, 3)) < 0.5, -1.0, 1.0))], kernel_shape=[3, 3])
    h = gb.op("MultiThreshold", [gb.bn(h, 3, rng), gb.c(np.zeros((1, 1)))], domain=QD, out_scale=2.0, out_bias=-1.0, out_dtype="BIPOLAR")
    y = gb.op("Flatten", [gb.op("GlobalAveragePool", [h])])
    m = gb.model(y, [1, 3])
    xs = list(rng.integers(0, 256, (6, 2, 5, 5)).astype(np.float32))
    g, info, ref, ir, _ = run_both(m, xs, "uint8")
    assert info["rules"]["MultiThreshold (input quantizer) -> ChannelAffine + FakeQuant"] == 1
    assert any(c["rule"] == "input quantizer verified exhaustively" and c["n_values"] == 256 for c in info["checks"])
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)
    with pytest.raises(QonnxImportError):          # without a discrete domain the quantizer cannot be verified
        import_model(m, Domain.parse("float32"))


# ------------------------------------------------------------------------------------------------ float set, layout, TopK
def test_float_set_and_nchw_flatten():
    rng = np.random.default_rng(12)
    gb = G([1, 3, 6, 6])
    h = gb.op("Conv", ["x", gb.c(rng.normal(0, 0.3, (4, 3, 3, 3))), gb.c(rng.normal(0, 0.1, 4))], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    h = gb.op("Relu", [gb.bn(h, 4, rng, neg=False)])
    r = gb.op("Conv", [h, gb.c(rng.normal(0, 0.3, (4, 4, 1, 1)))], kernel_shape=[1, 1])
    h = gb.op("Add", [h, r])
    h = gb.op("MaxPool", [h], kernel_shape=[2, 2], strides=[2, 2])
    f = gb.op("Flatten", [h], axis=1)                                   # [1, 4*3*3]: NCHW order needs a Transpose
    y = gb.op("Gemm", [f, gb.c(rng.normal(0, 0.3, (5, 36))), gb.c(rng.normal(0, 0.1, 5))], transB=1)
    m = gb.model(y, [1, 5])
    g, info, ref, ir, _ = run_both(m, list(rng.normal(0, 1, (5, 3, 6, 6)).astype(np.float32)))
    ops = g.op_counts()
    assert ops.get("Transpose") == 1 and ops.get("Dense") == 1 and ops.get("Conv2D") == 2 and "ChannelAffine" not in ops
    np.testing.assert_allclose(ir, ref, rtol=1e-5, atol=1e-5)


def test_trailing_topk_dropped():
    rng = np.random.default_rng(13)
    gb = G([1, 8])
    y = gb.op("MatMul", [gb.bq("x"), gb.c(np.where(rng.random((8, 4)) < 0.5, -1.0, 1.0))])
    y = gb.op("Mul", [y, gb.c(0.5)])
    vals, idx = gb.op("TopK", [y, gb.c(np.array([1]), np.int64)], n_out=2, axis=-1)
    m = gb.model(idx, [1, 1])
    m.graph.output[0].type.tensor_type.elem_type = TensorProto.INT64
    g, info = import_model(m)
    assert info["dropped"] and info["logits_onnx_tensor"] == y
    assert node(g, "BDense") and g.outputs == [node(g, "BDense")[0].outputs[0]]


def test_onnxf64_matches_onnxruntime_float32():
    ort = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(14)
    gb = G([1, 3, 6, 6])
    h = gb.op("Conv", ["x", gb.c(rng.normal(0, 0.3, (4, 3, 3, 3)))], kernel_shape=[3, 3], pads=[1, 0, 1, 2], strides=[1, 2])
    h = gb.op("AveragePool", [gb.op("Relu", [gb.bn(h, 4, rng)])], kernel_shape=[2, 2], strides=[1, 1], pads=[1, 1, 0, 0],
              count_include_pad=1)
    y = gb.op("Softmax", [gb.op("Flatten", [gb.op("GlobalAveragePool", [h])])], axis=1)
    m = gb.model(y, [1, 4])
    x = rng.normal(0, 1, (1, 3, 6, 6)).astype(np.float32)
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    a = s.run(None, {"x": x})[0]
    b = OnnxF64(m).run(x, np.float64)[y]
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------------------------------ phase 2: PyTorch layout
def _set_meta(nd, **kv):
    for k, v in kv.items():
        p = nd.metadata_props.add()
        p.key, p.value = k, str(v)


def _get_meta(nd, key):
    return next((p.value for p in nd.metadata_props if p.key == key), None)


def _torch_like_block(rng, neg_scale=True):
    """A ReActNet-like block as tools/export_torch_qonnx.py emits it, with exporter metadata (bitino.name / order):
    move0 Add -> Sign -> Conv(BipolarQuant(W, s[o])) ; downsample 1x1 Conv on x ; Add ; move1 Add ; PRelu ; move2 Add ;
    GlobalAveragePool ; Flatten ; Gemm."""
    c = 4
    gb = G([1, c, 6, 6])
    names = []

    def op(name, *a, **k):
        t = gb.op(*a, **k)
        names.append(name)
        return t
    t = op("b.move0", "Add", ["x", gb.c(rng.normal(0, 0.2, (1, c, 1, 1)))])
    s = op("b.act", "Sign", [t])
    sc = rng.uniform(0.2, 1.0, (c, 1, 1, 1)) * (np.where(np.arange(c) % 2 == 0, -1.0, 1.0)[:, None, None, None] if neg_scale else 1.0)
    w = op("b.conv", "BipolarQuant", [gb.c(rng.normal(0, 1, (c, c, 3, 3))), gb.c(sc)], domain=QD)
    h = op("b.conv", "Conv", [s, w], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    r = op("b.ds", "Conv", ["x", gb.c(rng.normal(0, 0.3, (c, c, 1, 1)))], kernel_shape=[1, 1])
    h = op("b.add", "Add", [h, r])
    h = op("b.move1", "Add", [h, gb.c(rng.normal(0, 0.2, (1, c, 1, 1)))])
    h = op("b.prelu", "PRelu", [h, gb.c(rng.uniform(0.05, 0.3, (c, 1, 1)))])
    h = op("b.move2", "Add", [h, gb.c(rng.normal(0, 0.2, (1, c, 1, 1)))])
    h = op("avgpool", "GlobalAveragePool", [h])
    h = op("flatten", "Flatten", [h], axis=1)
    y = op("fc", "Gemm", [h, gb.c(rng.normal(0, 0.3, (3, c))), gb.c(rng.normal(0, 0.1, 3))], transB=1)
    m = gb.model(y, [1, 3])
    for i, (nd, nm) in enumerate(zip(m.graph.node, names)):
        _set_meta(nd, **{"bitino.name": nm, "bitino.order": i})
    return m


def test_exporter_names_order_and_torch_layout():
    """--naming exporter + the exporter's node order (cleanup may reorder independent branches) + --affine keep +
    --constant-order first-use: the legacy PyTorch layout (bias-only ChannelAffine, GlobalAvgPool keepdims 0 before a
    flatten, Add without attributes, no post_add, pad_value 0.0)."""
    rng = np.random.default_rng(20)
    m = _torch_like_block(rng)
    nodes = [onnx.NodeProto.FromString(nd.SerializeToString()) for nd in m.graph.node]
    i_ds = next(i for i, nd in enumerate(nodes) if nd.op_type == "Conv" and nd.input[0] == "x")
    del m.graph.node[:]
    m.graph.node.extend([nodes[i_ds]] + [nd for i, nd in enumerate(nodes) if i != i_ds])     # ds first: still topological
    g, info = import_model(m, naming="exporter", affine="keep", constant_order="first-use")
    assert [(n.op, n.name) for n in g.nodes] == [
        ("ChannelAffine", "b.move0"), ("BConv2D", "b.conv"), ("Conv2D", "b.ds"), ("Add", "b.add"), ("ChannelAffine", "b.move1"),
        ("Activation", "b.prelu"), ("ChannelAffine", "b.move2"), ("GlobalAvgPool", "avgpool"), ("Dense", "fc")]
    n = {x.name: x for x in g.nodes}
    assert n["b.move0"].inputs[1] == "" and n["b.move0"].inputs[2] == "b.move0/bias"
    assert n["b.add"].attrs == {} and n["avgpool"].attrs == {"keepdims": 0}
    bc = n["b.conv"]
    assert bc.attrs["input_binarize"] == "sign_zero" and bc.attrs["pad_value"] == 0.0 and isinstance(bc.attrs["pad_value"], float)
    assert bc.inputs[3] == "" and bc.inputs[2] == "b.conv/post_mul"
    first_use = []
    for x in g.nodes:
        for i in x.inputs:
            if i in g.constants and i not in first_use:
                first_use.append(i)
    assert list(g.constants) == first_use
    xs = list(rng.normal(0, 1, (5, 4, 6, 6)).astype(np.float32))
    b0 = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == nodes[0].input[1])).reshape(4)
    xs[0][:, :2, :2] = -b0[:, None, None]                                   # exact zeros at the Sign
    ev = OnnxF64(m)
    ref = np.stack([np.asarray(ev.run(xi[None], np.float64)[ev.output_names[0]]).reshape(-1) for xi in xs])
    ir = np.stack([executor.run(g, {"input": np.transpose(xi[None], (0, 2, 3, 1))}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
                   for xi in xs])
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_bipolar_weight_scale_sign_kept():
    """BipolarQuant(W, s[o]) weights with negative s[o]: bits = Bin(W), post_mul = s (negative kept), the legacy polarity;
    the plain ±alpha rule (folded constant) flips both. Same outputs, bit for bit."""
    from bitino.ir import unpack_bits

    rng = np.random.default_rng(21)
    m = _torch_like_block(rng, neg_scale=True)
    g, info = import_model(m, naming="exporter", affine="keep")
    bc = node(g, "BConv2D")[0]
    assert np.any(g.constants[bc.inputs[2]].data < 0)
    assert info["rules"]["BipolarQuant(W, scale[o]) weights -> bits + signed alpha"] == 1
    bq = next(nd for nd in m.graph.node if nd.op_type == "BipolarQuant")
    lat = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == bq.input[0]))
    bits = unpack_bits(g.constants[bc.inputs[1]].data, 4)                    # OHWI
    assert np.array_equal(bits, np.where(np.transpose(lat, (0, 2, 3, 1)) >= 0, 1, -1))
    # the same weights as a folded constant: the ±|alpha| rule (positive post_mul, flipped bits)
    w = OnnxF64(m).run(np.zeros((1, 4, 6, 6), np.float32), np.float64, keep=True)[bq.output[0]]
    m.graph.initializer.append(numpy_helper.from_array(np.asarray(w, np.float32), "wfold"))
    conv = next(nd for nd in m.graph.node if nd.op_type == "Conv" and nd.input[1] == bq.output[0])
    conv.input[1] = "wfold"
    m.graph.node.remove(bq)
    g2, _ = import_model(m, naming="exporter", affine="keep")
    assert np.all(g2.constants[node(g2, "BConv2D")[0].inputs[2]].data > 0)
    for xi in rng.normal(0, 1, (4, 6, 6, 4)).astype(np.float32):
        a = executor.run(g, {"input": xi[None]})[g.outputs[0]]
        b = executor.run(g2, {"input": xi[None]})[g2.outputs[0]]
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_affine_keep_vs_fold():
    """--affine keep: BatchNormalization -> its own ChannelAffine, Mul + Add (an exported BN) -> one ChannelAffine, an
    Add before a BipolarQuant -> ChannelAffine + BConv2D(ge) at 0; fold: the same decision as an input threshold.
    The binarized path is bit-identical between the two (fl(x + b) >= 0 <=> x >= -b)."""
    rng = np.random.default_rng(22)
    gb = G([1, 4, 5, 5])
    bias = rng.normal(0, 0.3, (1, 4, 1, 1)).astype(np.float32)
    t = gb.bq(gb.op("Add", ["x", gb.c(bias)]))
    h = gb.op("Conv", [t, gb.c(np.where(rng.random((4, 4, 3, 3)) < 0.5, -1.0, 1.0))], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    h = gb.op("Add", [gb.op("Mul", [h, gb.c(rng.uniform(0.5, 2, (1, 4, 1, 1)))]), gb.c(rng.normal(0, 1, (1, 4, 1, 1)))])
    h = gb.bn(gb.op("Relu", [h]), 4, rng)
    y = gb.op("Flatten", [gb.op("GlobalAveragePool", [h])])
    m = gb.model(y, [1, 4])
    gk, ik = import_model(m, affine="keep", fold=False)
    gf, _ = import_model(m, affine="fold", fold=False)
    assert gk.op_counts() == {"ChannelAffine": 3, "BConv2D": 1, "Activation": 1, "GlobalAvgPool": 1}
    assert ik["rules"]["Mul + Add by per-channel constants -> one ChannelAffine (keep)"] == 1
    bk, bf = node(gk, "BConv2D")[0], node(gf, "BConv2D")[0]
    assert bk.attrs["input_binarize"] == "ge" and not bk.inputs[4] and bf.inputs[4]
    xs = rng.normal(0, 0.5, (6, 5, 5, 4)).astype(np.float32)
    xs[0, :2, :2, :] = -bias.reshape(4)                                     # exactly at the threshold
    for xi in xs:
        ek = executor.run(gk, {"input": xi[None]}, keep_intermediates=True)
        ef = executor.run(gf, {"input": xi[None]}, keep_intermediates=True)
        dk = executor._binarize(ek[bk.inputs[0]], "ge", None)
        df = executor._binarize(ef[bf.inputs[0]], "ge", gf.constants[bf.inputs[4]].data)
        assert np.array_equal(dk, df)                                       # same binary input
    ev = OnnxF64(m)
    ref = np.stack([np.asarray(ev.run(np.transpose(xi[None], (0, 3, 1, 2)), np.float64)[y]).reshape(-1) for xi in xs])
    ir = np.stack([executor.run(gk, {"input": xi[None]}, float_dtype=np.float64)[gk.outputs[0]].reshape(-1) for xi in xs])
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_multi_use_affine_rounded_once():
    """t = x + c1 used twice (Relu and t + c2): t is one ChannelAffine (bias c1) and t + c2 another (bias c2), never
    x + fl(c1 + c2); the IR equals the source in float32 bit for bit."""
    rng = np.random.default_rng(23)
    gb = G([1, 3, 4, 4])
    c1, c2 = rng.normal(0, 1, (1, 3, 1, 1)), rng.normal(0, 1, (1, 3, 1, 1))
    t = gb.op("Add", ["x", gb.c(c1)])
    u = gb.op("Add", [t, gb.c(c2)])
    y = gb.op("Add", [gb.op("Relu", [t]), gb.op("Relu", [u])])
    m = gb.model(y, [1, 3, 4, 4])
    g, info = import_model(m)
    ca = node(g, "ChannelAffine")
    assert len(ca) == 2 and all(not x.inputs[1] for x in ca)
    assert np.array_equal(g.constants[ca[0].inputs[2]].data, c1.reshape(-1).astype(np.float32))
    assert np.array_equal(g.constants[ca[1].inputs[2]].data, c2.reshape(-1).astype(np.float32))
    ev = OnnxF64(m)
    for xi in rng.normal(0, 1, (4, 3, 4, 4)).astype(np.float32):
        ref = np.asarray(ev.run(xi[None], np.float32)[y], np.float32)
        ir = executor.run(g, {"input": np.transpose(xi[None], (0, 2, 3, 1))})[g.outputs[0]]
        assert np.array_equal(np.ascontiguousarray(np.transpose(ir, (0, 3, 1, 2))).view(np.uint32), ref.view(np.uint32))


def test_affine_keep_rejects_inexact_div():
    gb = G([1, 3, 4, 4])
    y = gb.op("Relu", [gb.op("Div", ["x", gb.c(np.full((1, 3, 1, 1), 3.0))])])
    with pytest.raises(QonnxImportError) as e:
        import_model(gb.model(y, [1, 3, 4, 4]), affine="keep")
    assert "power of two" in str(e.value)
    gb = G([1, 3, 4, 4])
    y = gb.op("Relu", [gb.op("Div", ["x", gb.c(np.full((1, 3, 1, 1), 4.0))])])
    g, _ = import_model(gb.model(y, [1, 3, 4, 4]), affine="keep")
    assert g.op_counts() == {"ChannelAffine": 1, "Activation": 1}


def test_exporter_metadata_survives_qonnx_cleanup():
    """qonnx cleanup renames nodes (GiveUniqueNodeNames) and may reorder them; the importer's names and order come from
    the metadata_props tools/export_torch_qonnx.py writes, so the IR graph is the same before and after cleanup."""
    pytest.importorskip("qonnx")
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup_model

    m = _torch_like_block(np.random.default_rng(24))
    p = m.metadata_props.add()
    p.key, p.value = "bitino.ir_unique_counter", "5"
    clean = cleanup_model(ModelWrapper(onnx.ModelProto.FromString(m.SerializeToString())), preserve_qnt_ops=True).model
    assert any(nd.name != _get_meta(nd, "bitino.name") for nd in clean.graph.node)
    assert [(q.key, q.value) for q in clean.metadata_props] == [("bitino.ir_unique_counter", "5")]
    a, _ = import_model(m, naming="exporter", affine="keep", constant_order="first-use")
    b, _ = import_model(clean, naming="exporter", affine="keep", constant_order="first-use")
    assert [(n.op, n.name, n.inputs, n.attrs) for n in a.nodes] == [(n.op, n.name, n.inputs, n.attrs) for n in b.nodes]
    assert list(a.constants) == list(b.constants)
    assert all(np.array_equal(a.constants[k].data, b.constants[k].data) for k in a.constants)


def test_exporter_unique_counter_continues_into_pass_names():
    """Model metadata ``bitino.ir_unique_counter`` (names the legacy converter drew from Graph.unique(), e.g.
    torch_irnet's ``nonlinear_1``) starts the IR graph's counter, so the fold pass names its constants as the legacy
    file does (``conv1/folded_18_w``); without --naming exporter it is ignored."""
    rng = np.random.default_rng(25)
    gb = G([1, 3, 5, 5])
    h = gb.op("Conv", ["x", gb.c(rng.normal(0, 0.3, (4, 3, 3, 3)))], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    h = gb.op("Add", [gb.op("Mul", [h, gb.c(rng.uniform(0.5, 2, (1, 4, 1, 1)))]), gb.c(rng.normal(0, 1, (1, 4, 1, 1)))])
    y = gb.op("Clip", [h, gb.c(-1.0), gb.c(1.0)])
    m = gb.model(y, [1, 4, 5, 5])
    for nd, nm in zip(m.graph.node, ["conv1", "bn1", "bn1", "nonlinear_1"]):
        _set_meta(nd, **{"bitino.name": nm})
    p = m.metadata_props.add()
    p.key, p.value = "bitino.ir_unique_counter", "17"
    g, _ = import_model(m, naming="exporter", affine="keep")
    assert [(n.op, n.name) for n in g.nodes] == [("Conv2D", "conv1"), ("Activation", "nonlinear_1")]
    assert node(g, "Conv2D")[0].inputs[1:] == ["conv1/folded_18_w", "conv1/folded_18_b"]
    g2, _ = import_model(m, affine="keep")
    assert node(g2, "Conv2D")[0].inputs[1].endswith("folded_1_w")


# ------------------------------------------------------------------------------------------------ step 1: canonical files
def _named_model(gb, y, out_shape, names, opset=17):
    """Model with exporter metadata: names[i] = (bitino.name, {extra node metadata}) for node i; bitino.order = the
    position of the group's first node (the same on every node of a group, docs/qonnx-profile.md § 7)."""
    m = gb.model(y, out_shape)
    m.opset_import[0].version = opset
    first = {}
    for i, (nm, _) in enumerate(names):
        first.setdefault(nm, i)
    for nd, (nm, kv) in zip(m.graph.node, names):
        _set_meta(nd, **{"bitino.name": nm, "bitino.order": first[nm], **kv})
    return m


def _ir_vs_ref(m, g, xs):
    ev = OnnxF64(m)
    ref = np.stack([np.asarray(ev.run(xi[None], np.float64)[ev.output_names[0]]).reshape(-1) for xi in xs])
    ir = np.stack([executor.run(g, {"input": np.transpose(xi[None], (0, 2, 3, 1))}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
                   for xi in xs])
    return ir, ref


def test_float64_carrier():
    """R7: Cast(DOUBLE initializer, to=FLOAT) is read at its float64 value by OnnxF64 in float64 and at fl32 in float32;
    the importer stores fl32 of the float64 value (the legacy converters' 'float64 then one rounding')."""
    c64 = np.array([1.0 / 3.0, -2.0 / 7.0, 0.1], np.float64).reshape(1, 3, 1, 1)
    gb = G([1, 3, 2, 2])
    s = gb.op("Cast", [gb.c(c64, np.float64)], to=TensorProto.FLOAT)
    y = gb.op("Mul", ["x", s])
    y = gb.op("GlobalAveragePool", [y])
    y = gb.op("Flatten", [y], axis=1)
    m = gb.model(y, [1, 3])
    x = np.ones((1, 3, 2, 2))
    ev = OnnxF64(m)
    np.testing.assert_array_equal(np.asarray(ev.run(x, np.float64)[y]).reshape(-1), c64.reshape(-1))
    np.testing.assert_array_equal(np.asarray(ev.run(x, np.float32)[y]).reshape(-1), c64.reshape(-1).astype(np.float32))
    g, _ = import_model(m, naming="index", affine="keep")
    ca = node(g, "ChannelAffine")[0]
    np.testing.assert_array_equal(g.constants[ca.inputs[1]].data, c64.reshape(-1).astype(np.float32))


def _canonical_conv_net(rng, attrs=None, pad=True, c=4, o=5):
    """Canonical binary conv group (profile § 4.1): explicit Pad(0) of the binarized tensor, Conv(BipolarQuant(W, 1),
    pads 0), Mul(alpha float64 carrier), Add(bias) -- all in group 'conv' -- then a BN (Mul + Add carriers) in group 'bn'."""
    gb = G([1, c, 5, 5])
    names = []

    def op(nm, *a, kv=None, **k):
        names.append((nm, kv or {}))
        return gb.op(*a, **k)
    t = op("conv", "BipolarQuant", ["x", gb.c(1.0)], domain=QD)
    if pad:
        t = op("conv", "Pad", [t, gb.c(np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.int64), gb.c(0.0)], mode="constant")
    w = gb.bq(gb.c(rng.normal(0, 1, (o, c, 3, 3))))
    names.append(("conv", {}))     # weight BipolarQuant (constant-only node)
    h = op("conv", "Conv", [t, w], kernel_shape=[3, 3], pads=[0, 0, 0, 0], kv={"bitino.attrs": json.dumps(attrs)} if attrs else None)
    a64 = rng.uniform(0.2, 1.0, (1, o, 1, 1)) * np.where(np.arange(o) % 2 == 0, -1.0, 1.0).reshape(1, o, 1, 1)
    h = op("conv", "Mul", [h, op("conv", "Cast", [gb.c(a64, np.float64)], to=TensorProto.FLOAT)])
    h = op("conv", "Add", [h, gb.c(rng.normal(0, 0.1, (1, o, 1, 1)))])
    s64, t64 = rng.uniform(0.5, 2.0, (1, o, 1, 1)), rng.normal(0, 1, (1, o, 1, 1))
    h = op("bn", "Mul", [h, op("bn", "Cast", [gb.c(s64, np.float64)], to=TensorProto.FLOAT)])
    h = op("bn", "Add", [h, op("bn", "Cast", [gb.c(t64, np.float64)], to=TensorProto.FLOAT)])
    h = op("avgpool", "GlobalAveragePool", [h])
    y = op("flatten", "Flatten", [h], axis=1)
    return _named_model(gb, y, [1, o], names), a64.reshape(-1)


def test_canonical_conv_group_epilogue_and_explicit_pad():
    """Conv(BipolarQuant(W, 1)) -> Mul(alpha) -> Add(bias) of one naming group = the BConv2D's post_mul / post_add (alpha
    signed); explicit Pad(0) = pad_value 0.0 (float JSON form); a BN of another group stays its own ChannelAffine."""
    rng = np.random.default_rng(30)
    m, a = _canonical_conv_net(rng)
    g, info = import_model(m, naming="exporter", affine="keep", constant_order="first-use", fold=False)
    assert [(n.op, n.name) for n in g.nodes] == [("BConv2D", "conv"), ("ChannelAffine", "bn"), ("GlobalAvgPool", "avgpool")]
    bc = g.nodes[0]
    assert bc.attrs["pad_value"] == 0.0 and isinstance(bc.attrs["pad_value"], float) and bc.attrs["pads"] == [1, 1, 1, 1]
    np.testing.assert_array_equal(g.constants[bc.inputs[2]].data, a.astype(np.float32))
    assert bc.inputs[3] == "conv/post_add"
    xs = list(rng.normal(0, 1, (4, 4, 5, 5)).astype(np.float32))
    ir, ref = _ir_vs_ref(m, g, xs)
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-5)


def test_group_attrs_json_form_and_value_change_rejected():
    """bitino.attrs: the legacy JSON form of an attribute (pad_value 0.0 of an unpadded binary conv; the importer's
    default is 0) is applied; an attribute value that differs is an import error (metadata never changes a value)."""
    rng = np.random.default_rng(31)
    m, _ = _canonical_conv_net(rng, attrs={"pad_value": 0.0}, pad=False)
    g, _ = import_model(m, naming="exporter", affine="keep")
    bc = node(g, "BConv2D")[0]
    assert bc.attrs["pad_value"] == 0.0 and isinstance(bc.attrs["pad_value"], float)
    m0, _ = _canonical_conv_net(rng, pad=False)
    g0, _ = import_model(m0, naming="exporter", affine="keep")
    assert node(g0, "BConv2D")[0].attrs["pad_value"] == 0 and isinstance(node(g0, "BConv2D")[0].attrs["pad_value"], int)
    m1, _ = _canonical_conv_net(rng, attrs={"pad_value": -1.0}, pad=False)
    with pytest.raises(QonnxImportError) as e:
        import_model(m1, naming="exporter", affine="keep")
    assert "would change the value" in str(e.value)


def test_reducemean_centering_names_and_consts():
    """IR-Net RBNN / SiMaN activation centering: ReduceMean(x, [1, 2, 3]) -> Sub(x, mean) -> Sign -> Conv = the legacy
    Reshape [1, CHW] + ReduceMean(axis -1) + ChannelAffine(-1) + AddBroadcast + BConv2D, named by bitino.name /
    bitino.extra_names, the constant by bitino.consts; exact decisions (fl(-m) = -m, fl(x + (-m)) = fl(x - m))."""
    rng = np.random.default_rng(32)
    c, o = 3, 4
    gb = G([1, c, 4, 4])
    names = []

    def op(nm, *a, kv=None, **k):
        names.append((nm, kv or {}))
        return gb.op(*a, **k)
    mu = op("c/flatten", "ReduceMean", ["x"], axes=[1, 2, 3], keepdims=1, kv={"bitino.extra_names": '["c/mean"]'})
    t = op("c/neg_mean", "Sub", ["x", mu], kv={"bitino.extra_names": '["c/center"]', "bitino.consts": '{"1": "c/minus_one"}'})
    t = op("c", "Sign", [t])
    w = gb.bq(gb.c(rng.normal(0, 1, (o, c, 3, 3))))
    names.append(("c", {}))
    h = op("c", "Conv", [t, w], kernel_shape=[3, 3])
    h = op("gap", "GlobalAveragePool", [h])
    y = op("flatten", "Flatten", [h], axis=1)
    m = _named_model(gb, y, [1, o], names)
    g, _ = import_model(m, naming="exporter", affine="keep", constant_order="first-use")
    assert [(n.op, n.name) for n in g.nodes][:5] == [("Reshape", "c/flatten"), ("ReduceMean", "c/mean"),
                                                    ("ChannelAffine", "c/neg_mean"), ("AddBroadcast", "c/center"), ("BConv2D", "c")]
    assert g.nodes[0].attrs == {"shape": [1, c * 16]} and g.nodes[1].attrs == {"axis": -1, "keepdims": 1}
    assert g.nodes[2].inputs[1] == "c/minus_one" and g.constants["c/minus_one"].data.tolist() == [-1.0]
    assert g.nodes[3].inputs == ["input", "c/neg_mean:out"]
    xs = list(rng.normal(0, 1, (4, c, 4, 4)).astype(np.float32))
    ir, ref = _ir_vs_ref(m, g, xs)
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_channel_pad_shortcut_and_flatten_permute():
    """CIFAR ResNet option-A shortcut: AveragePool(1, stride 2) + Pad of the channels -> AvgPool2D + Pad (NHWC pads on C);
    VGG-Small head: Flatten of [1, C, H, W] + Gemm with --flatten-dense permute -> Reshape of the NHWC tensor and the
    Dense columns permuted (no Transpose), the same function as the default Transpose form."""
    rng = np.random.default_rng(33)
    c = 2
    gb = G([1, c, 4, 4])
    t = gb.op("AveragePool", ["x"], kernel_shape=[1, 1], strides=[2, 2], count_include_pad=1)
    t = gb.op("Pad", [t, gb.c(np.array([0, 1, 0, 0, 0, 1, 0, 0]), np.int64), gb.c(0.0)], mode="constant")
    t = gb.op("Relu", [t])
    f = gb.op("Flatten", [t], axis=1)
    y = gb.op("Gemm", [f, gb.c(rng.normal(0, 1, (3, (c + 2) * 4))), gb.c(rng.normal(0, 1, 3))], transB=1)
    m = gb.model(y, [1, 3])
    g, _ = import_model(m, flatten="permute")
    ap, pd = node(g, "AvgPool2D")[0], node(g, "Pad")[0]
    assert ap.attrs == {"kernel": [1, 1], "strides": [2, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1}
    assert pd.attrs == {"pads": [0, 0, 0, 0, 0, 0, 1, 1], "value": 0.0}
    assert not node(g, "Transpose")
    g2, _ = import_model(m)
    assert node(g2, "Transpose")
    xs = list(rng.normal(0, 1, (3, c, 4, 4)).astype(np.float32))
    for gg in (g, g2):
        ir, ref = _ir_vs_ref(m, gg, xs)
        np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_onnxf64_keep_set_frees_intermediates():
    """OnnxF64.run(keep={names}): the requested tensors and the outputs are returned with the same values as keep=True,
    every other intermediate is freed after its last use (the verification's peak memory)."""
    rng = np.random.default_rng(34)
    m, _ = _canonical_conv_net(rng)
    ev = OnnxF64(m)
    x = rng.normal(0, 1, (1, 4, 5, 5))
    full = ev.run(x, np.float64, keep=True)
    pre = m.graph.node[0].output[0]
    part = ev.run(x, np.float64, keep={pre})
    assert pre in part and ev.output_names[0] in part
    np.testing.assert_array_equal(part[pre], full[pre])
    np.testing.assert_array_equal(part[ev.output_names[0]], full[ev.output_names[0]])
    assert len(part) < len(full)


# ------------------------------------------------------------------------------------------------ step 1b
def _img_ir_vs_ref(m, g, xs):
    """Image outputs: the ONNX output [1, C, H, W] in NHWC order vs the IR output (NHWC), both float64."""
    ev = OnnxF64(m)
    ref = np.stack([np.asarray(ev.run(xi[None], np.float64)[ev.output_names[0]]).transpose(0, 2, 3, 1).reshape(-1) for xi in xs])
    ir = np.stack([executor.run(g, {"input": np.transpose(xi[None], (0, 2, 3, 1))}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
                   for xi in xs])
    return ir, ref


def _pixel_shuffle_np(x, r):
    n, cr, h, w = x.shape
    c = cr // (r * r)
    return x.reshape(n, c, r, r, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(n, c, h * r, w * r)


def test_pixel_shuffle_pattern_names_and_values():
    """PixelShuffle(2) as Reshape [1, C, r, r, H, W] + Transpose [0, 1, 4, 2, 5, 3] + Reshape (profile § 4.6) -> the
    legacy NHWC triple Reshape [1, H, W, C, r, r] / Transpose / Reshape [1, H*r, W*r, C], named by bitino.name and
    bitino.extra_names; values equal torch's pixel_shuffle (bit for bit: data movement)."""
    rng = np.random.default_rng(40)
    gb = G([1, 8, 3, 4])
    t = gb.op("Reshape", ["x", gb.c(np.array([1, 2, 2, 2, 3, 4]), np.int64)])
    t = gb.op("Transpose", [t], perm=[0, 1, 4, 2, 5, 3])
    y = gb.op("Reshape", [t, gb.c(np.array([1, 2, 6, 8]), np.int64)])
    ex = {"bitino.extra_names": json.dumps(["ps/reshape_in", "ps/transpose"])}
    m = _named_model(gb, y, [1, 2, 6, 8], [("ps", ex)] * 3)
    g, _ = import_model(m, naming="exporter", affine="keep")
    assert [(n.op, n.name, n.attrs) for n in g.nodes] == [
        ("Reshape", "ps/reshape_in", {"shape": [1, 3, 4, 2, 2, 2]}), ("Transpose", "ps/transpose", {"perm": [0, 1, 4, 2, 5, 3]}),
        ("Reshape", "ps", {"shape": [1, 6, 8, 2]})]
    xs = list(rng.normal(0, 1, (3, 8, 3, 4)).astype(np.float32))
    ir, ref = _img_ir_vs_ref(m, g, xs)
    np.testing.assert_array_equal(ir, ref)
    np.testing.assert_array_equal(ref[0], _pixel_shuffle_np(xs[0][None], 2).transpose(0, 2, 3, 1).reshape(-1))


def test_rank6_reshape_without_pixel_shuffle_transpose_rejected():
    gb = G([1, 8, 3, 4])
    t = gb.op("Reshape", ["x", gb.c(np.array([1, 2, 2, 2, 3, 4]), np.int64)])
    t = gb.op("Transpose", [t], perm=[0, 1, 2, 3, 5, 4])
    y = gb.op("Reshape", [t, gb.c(np.array([1, 2, 6, 8]), np.int64)])
    with pytest.raises(QonnxImportError, match="PixelShuffle"):
        import_model(gb.model(y, [1, 2, 6, 8]))


def test_row_gather_nearest_upsampling():
    """Nearest x2 as Reshape [1, C, H*W] + Gather(axis 2, constant [OH, OW] indices) -> IR Reshape [H*W, C] + Gather
    (i32 indices, named <group>/indices) + Reshape [1, OH, OW, C]; values = the nearest upsampling (exact copy)."""
    rng = np.random.default_rng(41)
    c, h, w, r = 3, 4, 5, 2
    yy, xx = np.meshgrid(np.arange(h * r) // r, np.arange(w * r) // r, indexing="ij")
    gb = G([1, c, h, w])
    t = gb.op("Reshape", ["x", gb.c(np.array([1, c, h * w]), np.int64)])
    y = gb.op("Gather", [t, gb.c(yy * w + xx, np.int64)], axis=2)
    ex = {"bitino.extra_names": json.dumps(["up/rows", "up/gather"])}
    m = _named_model(gb, y, [1, c, h * r, w * r], [("up", ex)] * 2)
    g, _ = import_model(m, naming="exporter", affine="keep")
    assert [(n.op, n.name) for n in g.nodes] == [("Reshape", "up/rows"), ("Gather", "up/gather"), ("Reshape", "up")]
    assert g.nodes[0].attrs == {"shape": [h * w, c]} and g.nodes[2].attrs == {"shape": [1, h * r, w * r, c]}
    ids = g.constants["up/indices"]
    assert ids.dtype == "i32" and ids.shape == (h * r, w * r)
    xs = list(rng.normal(0, 1, (2, c, h, w)).astype(np.float32))
    ir, ref = _img_ir_vs_ref(m, g, xs)
    np.testing.assert_array_equal(ir, ref)
    np.testing.assert_array_equal(ref[0], np.repeat(np.repeat(xs[0], r, 1), r, 2).transpose(1, 2, 0).reshape(-1))


def test_row_gather_negative_index_rejected():
    gb = G([1, 2, 2, 2])
    t = gb.op("Reshape", ["x", gb.c(np.array([1, 2, 4]), np.int64)])
    y = gb.op("Gather", [t, gb.c(np.array([[0, -1], [1, 2]]), np.int64)], axis=2)
    with pytest.raises(QonnxImportError, match="indices"):
        import_model(gb.model(y, [1, 2, 2, 2]))


def test_channel_split_half_sum():
    """BiMatting (out1[:, :P] + out1[:, P:]) / 2 as Reshape [1, 2, P, H*W] + Transpose [0, 2, 3, 1] + AveragePool(1 x 2,
    count_include_pad 1) + Reshape [1, P, H, W] -> IR Reshape [1, H*W, 2, P] + AvgPool2D + Reshape (the legacy layout);
    float32 value = fl(a + b) / 2."""
    rng = np.random.default_rng(42)
    p, h, w = 3, 2, 4
    gb = G([1, 2 * p, h, w])
    t = gb.op("Reshape", ["x", gb.c(np.array([1, 2, p, h * w]), np.int64)])
    t = gb.op("Transpose", [t], perm=[0, 2, 3, 1])
    t = gb.op("AveragePool", [t], kernel_shape=[1, 2], strides=[1, 2], pads=[0, 0, 0, 0], count_include_pad=1)
    y = gb.op("Reshape", [t, gb.c(np.array([1, p, h, w]), np.int64)])
    m = _named_model(gb, y, [1, p, h, w], [("b.split", {}), ("b.split", {}), ("b.split_sum_half", {}), ("b.split_sum", {})])
    g, _ = import_model(m, naming="exporter", affine="keep")
    assert [(n.op, n.name, n.attrs) for n in g.nodes] == [
        ("Reshape", "b.split", {"shape": [1, h * w, 2, p]}),
        ("AvgPool2D", "b.split_sum_half", {"kernel": [1, 2], "strides": [1, 2], "pads": [0, 0, 0, 0], "count_include_pad": 1}),
        ("Reshape", "b.split_sum", {"shape": [1, h, w, p]})]
    x = rng.normal(0, 1, (1, 2 * p, h, w)).astype(np.float32)
    y32 = executor.run(g, {"input": x.transpose(0, 2, 3, 1)})[g.outputs[0]]
    want = ((x[:, :p] + x[:, p:]) / np.float32(2)).astype(np.float32).transpose(0, 2, 3, 1)
    np.testing.assert_array_equal(y32, want)


def test_const_map_add_and_carrier():
    """Add(x, M) with a constant [1, C, H, W] map that varies spatially (a float64 carrier) -> IR Add(x, fl32(M) NHWC)."""
    rng = np.random.default_rng(43)
    m64 = rng.normal(0, 1, (1, 3, 2, 2))
    gb = G([1, 3, 2, 2])
    y = gb.op("Add", ["x", gb.op("Cast", [gb.c(m64, np.float64)], to=TensorProto.FLOAT)])
    m = _named_model(gb, y, [1, 3, 2, 2], [("corr", {}), ("corr", {"bitino.consts": json.dumps({"1": "corr/map_c"})})])
    g, _ = import_model(m, naming="exporter", affine="keep")
    assert [(n.op, n.name) for n in g.nodes] == [("Add", "corr")]
    np.testing.assert_array_equal(g.constants["corr/map_c"].data, m64.transpose(0, 2, 3, 1).astype(np.float32))
    xs = list(rng.normal(0, 1, (2, 3, 2, 2)).astype(np.float32))
    ir, ref = _img_ir_vs_ref(m, g, xs)
    np.testing.assert_allclose(ir, ref, rtol=0, atol=1e-6)


def _float_layer_net(rng, declare=True, weights="float"):
    """Sign -> Pad(0) -> Conv(W) -> GAP -> Flatten: a float layer on a binarized tensor (BNAS BinOp quirk conv)."""
    gb = G([1, 3, 5, 5])
    t = gb.op("Sign", ["x"])
    t = gb.op("Pad", [t, gb.c(np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.int64), gb.c(0.0)], mode="constant")
    w = rng.normal(0, 1, (4, 3, 3, 3))
    if weights == "alpha":
        w = np.sign(w) * 0.5
    h = gb.op("Conv", [t, gb.c(w)], kernel_shape=[3, 3], pads=[0, 0, 0, 0])
    h = gb.op("GlobalAveragePool", [h])
    y = gb.op("Flatten", [h], axis=1)
    kv = {"bitino.extra_names": json.dumps(["q/sign"])}
    if declare:
        kv["bitino.float_layer"] = "1"
    return _named_model(gb, y, [1, 4], [("q", {}), ("q", {}), ("q", kv), ("pool", {}), ("flat", {})])


def test_declared_float_layer_on_binarized_tensor():
    """bitino.float_layer (profile § 7): Conv(Sign(x), float W) -> Activation(sign_zero) + Conv2D (zero padding of the
    sign tensor); undeclared it stays an import error (binary weights are never rounded); declared on +-alpha weights
    it is rejected (a binary layer must not be declared float)."""
    rng = np.random.default_rng(44)
    m = _float_layer_net(rng)
    g, _ = import_model(m, naming="exporter", affine="keep", fold=False)
    assert [(n.op, n.name) for n in g.nodes][:2] == [("Activation", "q/sign"), ("Conv2D", "q")]
    assert g.nodes[0].attrs == {"type": "sign_zero"} and g.nodes[1].attrs["pads"] == [1, 1, 1, 1] and g.nodes[1].attrs["pad_value"] == 0.0
    xs = list(np.round(rng.normal(0, 1, (3, 3, 5, 5)) * 2).astype(np.float32) / 2)      # exact zeros: sign(0) = 0
    ir, ref = _ir_vs_ref(m, g, xs)
    np.testing.assert_allclose(ir, ref, rtol=1e-12, atol=1e-12)
    with pytest.raises(QonnxImportError, match="weight channel"):
        import_model(_float_layer_net(rng, declare=False), naming="exporter", affine="keep")
    with pytest.raises(QonnxImportError, match="binary layer"):
        import_model(_float_layer_net(rng, weights="alpha"), naming="exporter", affine="keep")


def _adabin_net(rng, mode="sign_zero", levels=2, c=3, o=4, hw=5):
    """AdaBin conv (profile § 4.1): Sub(beta) -> Sign / BipolarQuant -> Add(beta) -> Mul(alpha) -> Conv(two-level W,
    zero pads) -> BN (Mul + Add carriers, group 'bn') -> Add(residual-like runtime operand) -> GAP -> Flatten."""
    gb = G([1, c, hw, hw])
    names = []

    def op(nm, *a, kv=None, **k):
        names.append((nm, kv or {}))
        return gb.op(*a, **k)
    beta, alpha = np.float32(0.1875), np.float32(0.75)
    t = op("ab", "Sub", ["x", gb.c(beta)])
    t = op("ab", "Sign", [t]) if mode == "sign_zero" else op("ab", "BipolarQuant", [t, gb.c(1.0)], domain=QD)
    t = op("ab", "Add", [t, gb.c(beta)])
    t = op("ab", "Mul", [t, gb.c(alpha)])
    s = np.where(rng.random((o, c, 3, 3)) < 0.5, 1, -1)
    wp, wm = rng.uniform(0.1, 0.5, o).astype(np.float32), -rng.uniform(0.1, 0.5, o).astype(np.float32)
    w = np.where(s > 0, wp[:, None, None, None], wm[:, None, None, None]).astype(np.float32)
    if levels == 3:
        w[0, 0, 0, 0] = np.float32(0.01)
    ex = {"bitino.extra_names": json.dumps(["ab/sum_s", "ab/sum_s_broadcast", "ab/add_sum_s"])}
    h = op("ab", "Conv", [t, gb.c(w)], kernel_shape=[3, 3], pads=[1, 1, 1, 1], kv=ex)
    s64, t64 = rng.uniform(0.5, 2.0, (1, o, 1, 1)), rng.normal(0, 1, (1, o, 1, 1))
    h = op("bn", "Mul", [h, op("bn", "Cast", [gb.c(s64, np.float64)], to=TensorProto.FLOAT)])
    h = op("bn", "Add", [h, op("bn", "Cast", [gb.c(t64, np.float64)], to=TensorProto.FLOAT)])
    h = op("res", "Add", [h, op("side", "Conv", ["x", gb.c(rng.normal(0, 1, (o, c, 1, 1)))], kernel_shape=[1, 1])])
    h = op("pool", "GlobalAveragePool", [h])
    y = op("flat", "Flatten", [h], axis=1)
    return _named_model(gb, y, [1, o], names)


@pytest.mark.parametrize("mode", ["sign_zero", "ge"])
def test_adabin_conv_decomposition(mode):
    """The AdaBin conv + its BN -> the legacy decomposition (torch_adabin): BConv2D(1, all +1) 'ab/sum_s' + Conv2D 1x1
    'ab/sum_s_broadcast' (constant without a layout), main BConv2D 'ab' (threshold beta, post_mul), Add 'ab/add_sum_s',
    Add of the border map named by the BN group; emitted before the next group; IR fp64 = the QONNX graph fp64."""
    rng = np.random.default_rng(46)
    m = _adabin_net(rng, mode)
    g, info = import_model(m, naming="exporter", affine="keep", fold=False)
    ops = [(n.op, n.name) for n in g.nodes]
    assert ops[:5] == [("BConv2D", "ab/sum_s"), ("Conv2D", "ab/sum_s_broadcast"), ("BConv2D", "ab"), ("Add", "ab/add_sum_s"), ("Add", "bn")]
    assert list(g.constants)[:6] == ["ab/threshold", "ab/ones_bits", "ab/sum_s_scale", "ab/kernel_bits", "ab/post_mul", "ab/border_map"]
    assert g.constants["ab/sum_s_scale"].layout == "" and g.constants["ab/border_map"].shape == (1, 5, 5, 4)
    assert g.nodes[0].attrs["input_binarize"] == mode and g.nodes[2].inputs[4] == "ab/threshold"
    assert info["binarizers"][0]["ir_threshold"] == "ab/threshold"
    xs = list(rng.normal(0, 1, (4, 3, 5, 5)).astype(np.float32))
    ir, ref = _ir_vs_ref(m, g, xs)
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_adabin_three_weight_levels_rejected():
    with pytest.raises(QonnxImportError, match="two values"):
        import_model(_adabin_net(np.random.default_rng(47), levels=3), naming="exporter", affine="keep")


# ------------------------------------------------------------------------------------------------ step 2 (Keras / Larq)
def _named_model_outs(gb, outs, names, opset=17, model_meta=None):
    """_named_model with several graph outputs [(name, shape)], nodes without a name (None: constant-only) and
    optional model metadata."""
    m = gb.model(outs[0][0], outs[0][1], extra_outputs=[helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outs[1:]])
    m.opset_import[0].version = opset
    first = {}
    for i, (nm, _) in enumerate(names):
        first.setdefault(nm, i)
    for nd, (nm, kv) in zip(m.graph.node, names):
        if nm is not None:
            _set_meta(nd, **{"bitino.name": nm, "bitino.order": first[nm], **kv})
    for k, v in (model_meta or {}).items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    return m


def _keras_head(rng, c=3, h=4, w=4, o=5, const_order=None, gate=False):
    """Keras layout: Conv (no pad_value: bitino.attrs null) -> Transpose(0, 2, 3, 1) + Reshape (Keras flatten,
    channels last) -> Gemm -> Softmax, two graph outputs (logits, probabilities); optionally a squeeze-and-excite gate
    Gemm -> Reshape [N, C, 1, 1] -> Sigmoid -> Mul(x, .) before the flatten."""
    gb = G([1, c, h, w])
    names = []

    def op(nm, *a, kv=None, **k):
        names.append((nm, kv or {}))
        return gb.op(*a, **k)
    wc = gb.c(rng.normal(0, 1, (c, c, 3, 3)))
    t = op("conv", "Conv", ["x", wc], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
           kv={"bitino.attrs": json.dumps({"pad_value": None}), "bitino.consts": json.dumps({"1": "conv/kernel"})})
    if gate:
        p = op("se/pool", "GlobalAveragePool", [t])
        p = op("se/pool", "Flatten", [p], axis=1)
        gl = op("se/pre_sigmoid", "Gemm", [p, gb.c(rng.normal(0, 1, (c, c))), gb.c(rng.normal(0, 1, c))], transB=1)
        gl = op("se", "Reshape", [gl, gb.c(np.array([1, c, 1, 1]), np.int64)])
        sg = op("se", "Sigmoid", [gl])
        t = op("se", "Mul", [t, sg])
    tt = op("flatten", "Transpose", [t], perm=[0, 2, 3, 1])
    f = op("flatten", "Reshape", [tt, gb.c(np.array([1, c * h * w]), np.int64)])
    wd = gb.c(rng.normal(0, 1, (o, c * h * w)))
    y = op("dense", "Gemm", [f, wd, gb.c(rng.normal(0, 1, o))], transB=1,
           kv={"bitino.consts": json.dumps({"1": "dense/kernel", "2": "dense/bias"})})
    p = op("softmax", "Softmax", [y], axis=-1)
    meta = {"bitino.const_order": json.dumps(const_order)} if const_order is not None else None
    return _named_model_outs(gb, [(y, [1, o]), (p, [1, o])], names, model_meta=meta), (c, h, w, o)


def test_keras_flatten_two_outputs_attrs_null():
    """Keras head: Transpose(0, 2, 3, 1) + Reshape = one IR Reshape of the NHWC tensor (no Transpose; the Dense weights
    are the Keras order), two graph outputs = two IR outputs (logits, probabilities), bitino.attrs pad_value null =
    a float Conv2D without pad_value (the Larq converters' layout); the values equal the float64 reference."""
    rng = np.random.default_rng(40)
    m, (c, h, w, o) = _keras_head(rng)
    g, info = import_model(m, naming="exporter", affine="keep", constant_order="first-use")
    assert [(n.op, n.name) for n in g.nodes] == [("Conv2D", "conv"), ("Reshape", "flatten"), ("Dense", "dense"), ("Softmax", "softmax")]
    assert "pad_value" not in g.nodes[0].attrs and g.nodes[1].attrs == {"shape": [1, c * h * w]}
    assert g.outputs == ["dense:out", "softmax:out"]
    assert list(g.constants) == ["conv/kernel", "dense/kernel", "dense/bias"]
    ev = OnnxF64(m)
    for xi in rng.normal(0, 1, (3, c, h, w)):
        r = ev.run(xi[None], np.float64)
        e = executor.run(g, {"input": np.transpose(xi[None], (0, 2, 3, 1))}, float_dtype=np.float64)
        for on, io in zip(ev.output_names, g.outputs):
            np.testing.assert_allclose(e[io].reshape(-1), np.asarray(r[on]).reshape(-1), rtol=1e-9, atol=1e-9)


def test_attrs_null_rejected_for_a_non_default_value():
    """bitino.attrs null removes an attribute only when the importer's value is the operator default (pad_value 0):
    on a BConv2D padded with -1 it would change the value and is rejected."""
    rng = np.random.default_rng(41)
    gb = G([1, 2, 4, 4])
    t = gb.bq("x")
    t = gb.op("Pad", [t, gb.c(np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.int64), gb.c(-1.0)], mode="constant")
    h = gb.op("Conv", [t, gb.bq(gb.c(rng.normal(0, 1, (3, 2, 3, 3))))], kernel_shape=[3, 3])
    h = gb.op("GlobalAveragePool", [h])
    y = gb.op("Flatten", [h], axis=1)
    names = [("c", {}), ("c", {}), (None, {}), ("c", {"bitino.attrs": json.dumps({"pad_value": None})}), ("gap", {}), ("gap", {})]
    m = _named_model_outs(gb, [(y, [1, 3])], names)
    with pytest.raises(QonnxImportError) as e:
        import_model(m, naming="exporter", affine="keep")
    assert "null would change the value" in str(e.value)


def test_const_order_metadata():
    """bitino.const_order fixes the final data-section order (after the passes); a list that is not the set of
    constants the importer built is an import error."""
    rng = np.random.default_rng(42)
    want = ["dense/bias", "conv/kernel", "dense/kernel"]
    m, _ = _keras_head(rng, const_order=want)
    g, info = import_model(m, naming="exporter", affine="keep", constant_order="first-use")
    assert list(g.constants) == want and info["passes"]["bitino.const_order (after the passes)"] == 1
    m2, _ = _keras_head(rng, const_order=["conv/kernel", "dense/kernel"])
    with pytest.raises(QonnxImportError) as e:
        import_model(m2, naming="exporter", affine="keep", constant_order="first-use")
    assert "bitino.const_order" in str(e.value) and "dense/bias" in str(e.value)


def test_keras_se_gate_reshape_sigmoid():
    """Squeeze-and-excite of the Keras models (Real-to-Binary Net, legacy QuickNet-XL): Gemm -> Reshape [N, C, 1, 1] ->
    Sigmoid -> Mul(x, .) = ChannelGate(x, gate [N, C]) on the Dense output, no IR Reshape (the converters' layout)."""
    rng = np.random.default_rng(43)
    m, (c, h, w, o) = _keras_head(rng, gate=True)
    g, _ = import_model(m, naming="exporter", affine="keep", constant_order="first-use")
    ops = [(n.op, n.name) for n in g.nodes]
    assert ops[:4] == [("Conv2D", "conv"), ("GlobalAvgPool", "se/pool"), ("Dense", "se/pre_sigmoid"), ("ChannelGate", "se")]
    assert g.nodes[3].inputs == ["conv:out", "se/pre_sigmoid:out"]
    ev = OnnxF64(m)
    for xi in rng.normal(0, 1, (3, c, h, w)):
        r = ev.run(xi[None], np.float64)[ev.output_names[0]]
        e = executor.run(g, {"input": np.transpose(xi[None], (0, 2, 3, 1))}, float_dtype=np.float64)[g.outputs[0]]
        np.testing.assert_allclose(e.reshape(-1), np.asarray(r).reshape(-1), rtol=1e-9, atol=1e-9)


def _dorefa_net(rng, c=3, o=4, k_bits=2):
    """DoReFa (Larq Zoo DoReFaNet): Round(Mul(Clip(x, 0, 1), n)) -> Conv(BipolarQuant(W, 1), own zero padding) ->
    Mul(mean|W| / n, float64 carrier) -> Transpose + Reshape -> the same quantizer -> Gemm(BipolarQuant(W, 1)) -> Mul."""
    n = 2 ** k_bits - 1
    gb = G([1, c, 4, 4])
    names = []

    def op(nm, *a, kv=None, **k):
        names.append((nm, kv or {}))
        return gb.op(*a, **k)

    def q(nm, x):
        t = op(nm, "Clip", [x, gb.c(0.0), gb.c(1.0)])
        t = op(nm, "Mul", [t, gb.c(float(n))])
        return op(nm, "Round", [t])
    t = q("qconv", "x")
    w = op(None, "BipolarQuant", [gb.c(rng.normal(0, 1, (o, c, 3, 3))), gb.c(1.0)], domain=QD)
    t = op("qconv", "Conv", [t, w], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    pm = rng.uniform(0.1, 0.5) / n
    s = op(None, "Cast", [gb.c(np.full((1, o, 1, 1), pm), np.float64)], to=TensorProto.FLOAT)
    t = op("qconv", "Mul", [t, s])
    tt = op("flatten", "Transpose", [t], perm=[0, 2, 3, 1])
    f = op("flatten", "Reshape", [tt, gb.c(np.array([1, o * 16]), np.int64)])
    f = q("qdense", f)
    w2 = op(None, "BipolarQuant", [gb.c(rng.normal(0, 1, (5, o * 16))), gb.c(1.0)], domain=QD)
    y = op("qdense", "Gemm", [f, w2], transB=1)
    y = op("qdense", "Mul", [y, gb.c(np.full((1, 5), 0.03))])
    return _named_model_outs(gb, [(y, [1, 5])], names), pm


def test_dorefa_qbconv_qbdense():
    """Round(Mul(Clip(x, 0, 1), n)) feeding a Conv / Gemm with ±1 weights -> QBConv2D / QBDense (act_bits, act_quant
    dorefa) with the group Mul as post_mul (fl32 of the float64 carrier) and no post_add; the values equal the float64
    reference (integer levels x ±1, then one product)."""
    rng = np.random.default_rng(44)
    m, pm = _dorefa_net(rng)
    g, _ = import_model(m, naming="exporter", affine="keep", constant_order="first-use", bit_outputs="none")
    assert [(n.op, n.name) for n in g.nodes] == [("QBConv2D", "qconv"), ("Reshape", "flatten"), ("QBDense", "qdense")]
    qc, qd = g.nodes[0], g.nodes[2]
    assert qc.attrs["act_bits"] == 2 and qc.attrs["act_quant"] == "dorefa" and qc.attrs["pads"] == [1, 1, 1, 1]
    assert qc.inputs[0] == "input" and qc.inputs[3] == "" and g.constants[qc.inputs[1]].layout == "OHWI_bit"
    np.testing.assert_array_equal(g.constants[qc.inputs[2]].data, np.full(4, pm, np.float32))
    assert g.constants[qd.inputs[1]].layout == "" and qd.attrs == {"activation": "none", "act_bits": 2, "act_quant": "dorefa"}
    xs = list(rng.uniform(-0.2, 1.2, (4, 3, 4, 4)).astype(np.float32))
    ir, ref = _ir_vs_ref(m, g, xs)
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_unscaled_binary_conv_bias_has_no_post_mul():
    """BinaryFaceNet's binary conv: Where(Greater(x, 0), 1, -1) (gt) -> Conv(±1 constants) -> Add(bias), no Mul in the
    group: post_add = the bias, post_mul empty (not a vector of ones). A BatchNorm of its own group and a ge BConv2D
    follow: after the passes fold the BN into the first BConv2D it feeds the second directly, and with --bit-outputs
    none it keeps its f32 output (the Larq converters), with the default it outputs bits."""
    rng = np.random.default_rng(45)
    gb = G([1, 2, 4, 4])
    names = []

    def op(nm, *a, kv=None, **k):
        names.append((nm, kv or {}))
        return gb.op(*a, **k)
    t = op("b1", "Greater", ["x", gb.c(0.0)])
    t = op("b1", "Where", [t, gb.c(1.0), gb.c(-1.0)])
    h = op("b1", "Conv", [t, gb.c(np.where(rng.random((3, 2, 1, 1)) < 0.5, -1.0, 1.0))], kernel_shape=[1, 1])
    h = op("b1", "Add", [h, gb.c(rng.normal(0, 1, (1, 3, 1, 1)))])
    h = op("bn", "Mul", [h, gb.c(rng.uniform(0.5, 2.0, (1, 3, 1, 1)))])
    h = op("bn", "Add", [h, gb.c(rng.normal(0, 1, (1, 3, 1, 1)))])
    t2 = op("b2", "BipolarQuant", [h, gb.c(1.0)], domain=QD)
    h2 = op("b2", "Conv", [t2, gb.c(np.where(rng.random((2, 3, 1, 1)) < 0.5, -1.0, 1.0))], kernel_shape=[1, 1])
    h2 = op("gap", "GlobalAveragePool", [h2])
    y = op("gap", "Flatten", [h2], axis=1)
    m = _named_model_outs(gb, [(y, [1, 2])], names)
    g, _ = import_model(m, naming="exporter", affine="keep", fold=False)
    b1 = g.nodes[0]
    assert b1.op == "BConv2D" and b1.attrs["input_binarize"] == "gt" and b1.inputs[2] == "" and b1.inputs[3] == "b1/post_add"
    assert [n.op for n in g.nodes][:3] == ["BConv2D", "ChannelAffine", "BConv2D"]
    for bo, want in (("none", "f32"), ("set", "bit")):
        g, _ = import_model(m, naming="exporter", affine="keep", bit_outputs=bo)
        assert [n.op for n in g.nodes][:2] == ["BConv2D", "BConv2D"] and g.nodes[0].attrs["output"] == want
    xs = list(rng.normal(0, 1, (4, 2, 4, 4)).astype(np.float32))
    g, _ = import_model(m, naming="exporter", affine="keep", bit_outputs="none")
    ir, ref = _ir_vs_ref(m, g, xs)
    np.testing.assert_allclose(ir, ref, rtol=1e-6, atol=1e-6)


def test_consts_renamed_at_once():
    """bitino.consts renames are applied together: two constants may exchange the importer's default names."""
    rng = np.random.default_rng(46)
    m, _ = _keras_head(rng)
    for nd in m.graph.node:
        if _get_meta(nd, "bitino.name") == "dense" and nd.op_type == "Gemm":
            del nd.metadata_props[:]
            _set_meta(nd, **{"bitino.name": "dense", "bitino.order": 3, "bitino.consts": json.dumps({"1": "dense/bias", "2": "dense/kernel"})})
    g, _ = import_model(m, naming="exporter", affine="keep", constant_order="first-use")
    d = node(g, "Dense")[0]
    assert d.inputs[1:] == ["dense/bias", "dense/kernel"] and g.constants["dense/kernel"].shape == (5,)


# ================================================================================================ step 3: group mapping
# bitino.qonnx_groups (docs/qonnx-step3.md): an IR graph written as one ONNX group per IR node by the direct writer,
# checked with check-qonnx, imported back; the published-file identity (data + graph SHA-256) is checked on synthetic
# graphs with every step-3 pattern, the B-class patterns bit for bit in onnxruntime (fp32) and the whole graph in
# float64 (OnnxF64 vs the numpy executor).
from bitino import qonnx_groups as qgr  # noqa: E402
from bitino.ir import Graph  # noqa: E402
from bitino.qonnx_profile import check_model as _check_model  # noqa: E402

_META3 = {"bitino.profile": "bitino-qonnx/1", "bitino.model_id": "test-groups", "bitino.variant": "float",
          "bitino.layout": "NCHW", "bitino.preprocessing": json.dumps({"key": "t", "definition": "test"}),
          "bitino.labels": "none", "bitino.source.framework": "test", "bitino.source.framework_version": "1",
          "bitino.source.checkpoint_sha256": "a" * 64, "bitino.weights_license": "MIT", "bitino.exporter": "test"}


def _groups_roundtrip(g: Graph, tmp_path, domain="float32"):
    """IR graph -> spec -> model.onnx (check-qonnx: no error, no warning) -> import; returns (onnx model, imported
    graph, info, data / graph SHA-256 equal)."""
    from bitino.fetch import data_sha256, graph_sha256
    spec, rep = qgr.write_spec(g)
    spec.meta.update(dict(_META3, **{"bitino.input_domain": domain}))
    qgr.save_spec(spec, str(tmp_path / "spec"))
    m = qgr.assemble(str(tmp_path / "spec"), str(tmp_path / "model.onnx"), "test")
    r = _check_model(onnx.load(str(tmp_path / "model.onnx")))
    assert r.passed(strict=True), r.to_text()
    g2, info = import_model(onnx.load(str(tmp_path / "model.onnx")))
    g.manifest, g2.manifest = {}, {}
    bfformat.write(g, str(tmp_path / "a.bitino"))
    bfformat.write(g2, str(tmp_path / "b.bitino"))
    same = (data_sha256(str(tmp_path / "a.bitino")) == data_sha256(str(tmp_path / "b.bitino")),
            graph_sha256(str(tmp_path / "a.bitino")) == graph_sha256(str(tmp_path / "b.bitino")))
    return m, g2, info, same, rep


def _ort_vs_executor(m, g, feeds_ir, ops_exact, tags):
    """Every group output in onnxruntime (fp32) vs the numpy executor (fp32): bit-identical for the IR operators in
    ops_exact (their inputs are bit-identical by construction of the test graphs)."""
    sess = qgr.ort_session(m, extra_outputs=[n.outputs[0] for n in g.nodes])
    names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(names, sess.run(None, qgr.onnx_feeds(g, feeds_ir))))
    env = executor.run(g, feeds_ir, keep_intermediates=True)
    checked = 0
    for n in g.nodes:
        if n.op not in ops_exact or n.outputs[0] not in outs:
            continue
        a, b = env[n.outputs[0]], outs[n.outputs[0]]
        if tags[n.outputs[0]] == "C":
            b = b.transpose(0, 2, 3, 1)
        assert a.shape == b.shape and np.array_equal(np.asarray(a, np.float32).view(np.uint32),
                                                     np.asarray(b, np.float32).view(np.uint32)), n.name
        checked += 1
    return checked


def _bits(rng, shape):
    return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)


def _transformer_ir(exact_only=False):
    rng = np.random.default_rng(3)
    L, D, H, V = 8, 16, 2, 20
    g = Graph()
    g.add_input("input_ids", [1, L], "i32")
    g.add_input("token_type_ids", [1, L], "i32")
    g.add_input("attention_mask", [1, L], "f32")
    tern = rng.choice([-1, 0, 1], size=(V, D))
    e = g.add_node("Gather", [g.add_constant("emb.w_sign", np.where(tern < 0, -1, 1), "bit", "VD_bit"), "input_ids",
                              g.add_constant("emb.alpha", rng.uniform(0.1, 1, V)),
                              g.add_constant("emb.w_zero", np.where(tern == 0, -1, 1), "bit", "VD_bit")], name="emb")
    p = g.add_node("Gather", [g.add_constant("pos.w", rng.normal(size=(L, D))), g.add_constant("pos.ids", np.arange(L).reshape(1, L), "i32")],
                   name="pos")
    t = g.add_node("Gather", [g.add_constant("type.w", rng.normal(size=(2, D))), "token_type_ids"], name="type")
    x = g.add_node("Add", [g.add_node("Add", [e, p], name="add_pos"), t], name="add_type")
    if not exact_only:
        x = g.add_node("LayerNorm", [x, g.add_constant("ln.g", rng.uniform(0.5, 1.5, D)), g.add_constant("ln.b", rng.normal(size=D))],
                       {"eps": 1e-12}, name="ln")
    m = g.add_node("ChannelAffine", ["attention_mask", g.add_constant("m.s", np.full(L, 10000.0)), g.add_constant("m.b", np.full(L, -10000.0))],
                   name="mask")
    m4 = g.add_node("Reshape", [m], {"shape": [1, 1, 1, L]}, name="mask4")
    tw = rng.choice([-1, 0, 1], size=(D, D))
    q = g.add_node("BDense", [x, g.add_constant("q.w", np.where(tw < 0, -1, 1), "bit", "OI_bit"), g.add_constant("q.a", rng.uniform(0.1, 1, D)),
                              g.add_constant("q.b", rng.normal(size=D)), g.add_constant("q.t", rng.normal(0, 0.1, D)),
                              g.add_constant("q.z", np.where(tw == 0, -1, 1), "bit", "OI_bit")], {"input_binarize": "sign_zero"}, name="q")
    k = g.add_node("BDense", [x, g.add_constant("k.w", _bits(rng, (D, D)), "bit"), g.add_constant("k.a", rng.uniform(0.1, 1, D)), "", ""],
                   {"input_binarize": "ge"}, name="k")
    v = g.add_node("BDense", [x, g.add_constant("v.w", _bits(rng, (D, D)), "bit")], {"input_binarize": "gt"}, name="v")
    heads = []
    for nm, s in (("q", q), ("k", k), ("v", v)):
        r = g.add_node("Reshape", [s], {"shape": [1, L, H, D // H]}, name=f"{nm}/split")
        heads.append(g.add_node("Transpose", [r], {"perm": [0, 2, 1, 3]}, name=f"{nm}/heads"))
    s = g.add_node("BMatMul", heads[:2], {"a_binarize": "sign_zero", "b_binarize": "sign_zero", "scale": 0.125, "transpose_b": 1}, name="scores")
    s = g.add_node("AddBroadcast", [s, m4], name="masked")
    c = g.add_node("BMatMul", [s, heads[2]], {"a_binarize": "gt0", "b_binarize": "sign_zero", "scale": -1.0, "transpose_b": 0}, name="ctx")
    c = g.add_node("Reshape", [g.add_node("Transpose", [c], {"perm": [0, 2, 1, 3]}, name="merge")], {"shape": [1, L, D]}, name="flat")
    sh = g.add_node("ChannelAffine", [c, "", g.add_constant("zm.shift", np.full(D, 8.0))], name="zm/shift")
    sg = g.add_node("Activation", [sh], {"type": "sign_zero"}, name="zm/sign")
    half = g.add_constant("zm.half", np.full(D, 0.5))
    kp = g.add_node("ChannelAffine", [sg, half, half], name="zm/keep")
    y = g.add_node("Mul", [c, kp], name="zm/mul")
    rows = g.add_node("Reshape", [y], {"shape": [L, D]}, name="rows")
    cst = g.add_node("BMatMul", [g.add_constant("ones", np.ones((L, L))), rows], {"a_binarize": "sign_zero", "b_binarize": "sign_zero",
                                                                                 "scale": 2.0, "transpose_b": 0}, name="colsum")
    cls = g.add_node("Gather", [cst, g.add_constant("cls.idx", np.zeros(1, np.int64), "i32")], name="cls")
    if exact_only:
        g.set_outputs(cls)
        return g, L, V
    a = g.add_node("Activation", [cls], {"type": "gelu"}, name="gelu")
    a = g.add_node("Dense", [a, g.add_constant("fc.w", rng.normal(size=(D, D))), g.add_constant("fc.b", rng.normal(size=D))], name="fc")
    a = g.add_node("Activation", [a], {"type": "tanh"}, name="tanh")
    p2 = g.add_node("Softmax", [g.add_node("MatMul", [a, g.add_constant("mm.w", rng.normal(size=(D, 3)))], {"scale": 0.5, "transpose_b": 0},
                                           name="mm")], name="softmax")
    g.set_outputs(p2)
    return g, L, V


def _transformer_feeds(L, V, seed=0):
    rng = np.random.default_rng(seed)
    return {"input_ids": rng.integers(0, V, (1, L)).astype(np.int32), "token_type_ids": rng.integers(0, 2, (1, L)).astype(np.int32),
            "attention_mask": (np.arange(L) < L - 2).astype(np.float32).reshape(1, L)}


def test_groups_transformer_identity_and_float64(tmp_path):
    g, L, V = _transformer_ir()
    dom = json.dumps({"input_ids": f"ids:{V}", "token_type_ids": "ids:2", "attention_mask": "float32"})
    m, g2, info, same, rep = _groups_roundtrip(g, tmp_path, dom)
    assert same == (True, True)
    assert info["rules"]["ternary weights {-1, 0, +1} -> w_sign + w_zero"] == 1 and info["passes"]["none"]
    modes = sorted({b["mode"] for b in info["binarizers"]})
    assert modes == ["ge", "gt", "gt0", "sign_zero"]
    ev = OnnxF64(m)
    for seed in range(3):
        f = _transformer_feeds(L, V, seed)
        y64 = ev.run(qgr.onnx_feeds(g2, f), np.float64)[m.graph.output[0].name]
        ir64 = executor.run(g2, f, float_dtype=np.float64)[g2.outputs[0]]
        assert np.max(np.abs(np.asarray(y64) - ir64)) < 1e-12


def test_groups_transformer_b_class_bit_exact(tmp_path):
    g, L, V = _transformer_ir(exact_only=True)
    dom = json.dumps({"input_ids": f"ids:{V}", "token_type_ids": "ids:2", "attention_mask": "float32"})
    m, g2, info, same, _ = _groups_roundtrip(g, tmp_path, dom)
    assert same == (True, True)
    n = _ort_vs_executor(m, g2, _transformer_feeds(L, V), {n.op for n in g2.nodes}, info["onnx_tags"])
    assert n == len(g2.nodes)


def _image_ir(exact_only=False):
    rng = np.random.default_rng(5)
    Hh, C = 16, 8
    g = Graph()
    g.add_input("input", [1, Hh, Hh, 3])
    gray = g.add_node("ImageGrayU8", ["input"], {"coeffs": [9798, 19235, 3735], "scale": 255.0, "shift": 15}, name="gray")
    rp, cp = [-0.0, -1.73205, 1.73205], [2.0, -1.0, -1.0]
    lbp = g.add_node("LocalBinaryPattern", [gray], {"rp": rp, "cp": cp, "method": "uniform", "div": 255.0}, name="lbp")
    lbp2 = g.add_node("LocalBinaryPattern", [gray], {"rp": rp, "cp": cp, "method": "default", "div": 255.0}, name="lbp_default")
    feats = [g.add_node("ChannelAffine", ["input", g.add_constant("in.s", rng.uniform(0.5, 2, 3))], name="in_scale"), lbp, lbp2]
    if not exact_only:
        feats.append(g.add_node("FftLogMagnitude", [gray], {"eps": 1e-9, "mul": 20.0, "div": 255.0, "shift": 1}, name="fft"))
    x = g.add_node("Concat", feats, {"axis": -1}, name="cat")
    cin = 3 + len(feats) - 1
    x = g.add_node("BConv2D", [x, g.add_constant("c1.w", _bits(rng, (C, 3, 3, cin)), "bit", "OHWI_bit"), g.add_constant("c1.m", rng.uniform(0.1, 1, C)),
                               g.add_constant("c1.b", rng.normal(size=C)), g.add_constant("c1.t", rng.normal(0, 0.2, cin))],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0, "input_binarize": "ge"}, name="c1")
    x = g.add_node("Activation", [x, g.add_constant("pr.s", rng.uniform(0, 0.3, C))], {"type": "prelu"}, name="prelu")
    r = g.add_node("ChannelAffine", [x, g.add_constant("bn.s", rng.uniform(0.5, 2, C)), g.add_constant("bn.b", rng.normal(size=C))], name="bn")
    x2 = g.add_node("BConv2D", [r, g.add_constant("c2.w", _bits(rng, (C, 3, 3, C // 2)), "bit", "OHWI_bit"), g.add_constant("c2.m", rng.uniform(0.1, 1, C)),
                                "", ""], {"strides": [1, 1], "dilations": [2, 2], "pads": [2, 2, 2, 2], "groups": 2, "pad_value": 0.0,
                                          "input_binarize": "sign_zero", "activation": "none", "output": "f32"}, name="c2")
    x = g.add_node("Add", [x2, r], name="res")
    x = g.add_node("MaxPool2D", [x], {"kernel": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}, name="mp")
    tok = g.add_node("Reshape", [x], {"shape": [1, (Hh // 2) ** 2, C]}, name="tokens")
    fq = g.add_node("FakeQuant", [tok, g.add_constant("fq.s", np.full(1, 0.25)), g.add_constant("fq.z", np.zeros(1))],
                    {"axis": -1, "pre_scale": 1.0, "qmin": -8, "qmax": 7}, name="fq")
    back = g.add_node("Reshape", [fq], {"shape": [1, Hh // 2, Hh // 2, C]}, name="back")
    x = g.add_node("ChannelAffine", [back, "", g.add_constant("mv.b", rng.normal(size=C))], name="move")
    if exact_only:
        g.set_outputs(g.add_node("Reshape", [x], {"shape": [1, (Hh // 2) ** 2 * C]}, name="flat"))
        return g
    gp = g.add_node("GlobalAvgPool", [x], {"keepdims": 1}, name="se/gap")
    s1 = g.add_node("Conv2D", [gp, g.add_constant("se1.w", rng.normal(size=(4, 1, 1, C)), layout="OHWI"), g.add_constant("se1.b", rng.normal(size=4))],
                    {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0, "activation": "relu"}, name="se1")
    s2 = g.add_node("Conv2D", [s1, g.add_constant("se2.w", rng.normal(size=(C, 1, 1, 4)), layout="OHWI"), g.add_constant("se2.b", rng.normal(size=C))],
                    {"strides": [1, 1], "pads": [0, 0, 0, 0]}, name="se2")
    x = g.add_node("ChannelGate", [x, s2], {"gate": "sigmoid"}, name="gate")
    x = g.add_node("AvgPool2D", [x], {"kernel": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1], "count_include_pad": 1}, name="ap")
    x = g.add_node("GlobalAvgPool", [x], {"keepdims": 0}, name="gap")
    g.set_outputs(g.add_node("Dense", [x, g.add_constant("fc.w", rng.normal(size=(2, C))), g.add_constant("fc.b", rng.normal(size=2))],
                             {"activation": "none"}, name="fc"))
    return g


def test_groups_image_features_identity_and_float64(tmp_path):
    g = _image_ir()
    m, g2, info, same, rep = _groups_roundtrip(g, tmp_path)
    assert same == (True, True)
    assert rep["layout_transposes"] >= 2
    for r in ("standard-op gray subgraph -> ImageGrayU8 (bit-exact)", "DOUBLE bilinear subgraph -> LocalBinaryPattern (bit-exact)",
              "DOUBLE MatMul DFT subgraph -> FftLogMagnitude (float contract, Q1)"):
        assert info["rules"][r] >= 1, r
    ev = OnnxF64(m)
    x = np.random.default_rng(1).random((1, 16, 16, 3)).astype(np.float32)
    y64 = ev.run(qgr.onnx_feeds(g2, {"input": x}), np.float64)[m.graph.output[0].name]
    ir64 = executor.run(g2, {"input": x}, float_dtype=np.float64)[g2.outputs[0]]
    assert np.max(np.abs(np.asarray(y64) - ir64)) < 1e-6        # the gate's double sigmoid is rounded to fp32 (F)


def test_groups_image_b_class_bit_exact(tmp_path):
    """ImageGrayU8, both LocalBinaryPattern methods, binary convs with threshold / explicit Pad / groups / dilation,
    PReLU, ChannelAffine, MaxPool, FakeQuant and the layout changes: onnxruntime fp32 = numpy executor, bit for bit
    (NaN, +-Inf and out-of-range pixels included)."""
    g = _image_ir(exact_only=True)
    m, g2, info, same, _ = _groups_roundtrip(g, tmp_path)
    assert same == (True, True)
    x = np.random.default_rng(2).uniform(-0.2, 1.2, (1, 16, 16, 3)).astype(np.float32)
    x[0, 0, 0, 0], x[0, 1, 1, 1], x[0, 2, 2, 2] = np.nan, np.inf, -np.inf
    x[0, 3, 3, :] = 1.0
    n = _ort_vs_executor(m, g2, {"input": x}, {n.op for n in g2.nodes}, info["onnx_tags"])
    assert n == len(g2.nodes)


def test_groups_rejections(tmp_path):
    """Metadata never changes a value; a group is exactly its operator's pattern; the DFT tables are the definition."""
    g = _image_ir()
    spec, _ = qgr.write_spec(g)
    spec.meta.update(dict(_META3, **{"bitino.input_domain": "float32"}))
    qgr.save_spec(spec, str(tmp_path / "spec"))
    base = qgr.assemble(str(tmp_path / "spec"), str(tmp_path / "model.onnx"), "test")

    def first(m, op):
        return next(nd for nd in m.graph.node if nd.op_type == op and any(p.key == "bitino.ir" for p in nd.metadata_props))

    m = onnx.load(str(tmp_path / "model.onnx"))          # a stride change through bitino.attrs
    nd = next(nd for nd in m.graph.node if nd.op_type == "MaxPool")
    nd.metadata_props.add(key="bitino.attrs", value=json.dumps({"strides": [1, 1]}))
    with pytest.raises(qgr.GroupImportError, match="would change the value"):
        import_model(m)
    m = onnx.load(str(tmp_path / "model.onnx"))          # no IR operator
    nd = first(m, "Conv")
    keep = [p for p in nd.metadata_props if p.key != "bitino.ir"]
    del nd.metadata_props[:]
    nd.metadata_props.extend(keep)
    with pytest.raises(qgr.GroupImportError, match="bitino.ir"):
        import_model(m)
    m = onnx.load(str(tmp_path / "model.onnx"))          # a DFT table off by one ulp
    t = next(i for i in m.graph.initializer if i.name.startswith("bitino/dft/cos"))
    a = numpy_helper.to_array(t).copy()
    a[1, 1] = np.nextafter(a[1, 1], 2.0)
    t.CopyFrom(numpy_helper.from_array(a, t.name))
    with pytest.raises(qgr.GroupImportError, match="DFT table"):
        import_model(m)
    m = onnx.load(str(tmp_path / "model.onnx"))          # an extra node in a group
    mp = next(nd for nd in m.graph.node if nd.op_type == "MaxPool")
    extra = helper.make_node("Relu", [mp.output[0]], ["extra_relu"], name="extra")
    for p in mp.metadata_props:
        if p.key in ("bitino.name", "bitino.order"):
            extra.metadata_props.add(key=p.key, value=p.value)
    m.graph.node.insert(list(m.graph.node).index(mp) + 1, extra)
    with pytest.raises(qgr.GroupImportError):
        import_model(m)
    del base


def test_executor_keep_frees_intermediates():
    g = _image_ir(exact_only=True)
    x = np.random.default_rng(0).random((1, 16, 16, 3)).astype(np.float32)
    full = executor.run(g, {"input": x}, keep_intermediates=True)
    part = executor.run(g, {"input": x}, keep={"c1:out"})
    assert set(part) == {"c1:out", g.outputs[0]}
    assert np.array_equal(part["c1:out"], full["c1:out"]) and np.array_equal(part[g.outputs[0]], full[g.outputs[0]])


# ================================================================================================ step 5 (int8_full)
# docs/qonnx-step5.md: int8_full graphs (the I8* operators) as canonical group-mapped files. The tiny models of the
# int8_full tests are quantized with the real quantizer (bitino.quantize_int8 / quantize_l8) into every published
# variant (v1, the l8 residual code, lin8 with fp32 or integer epilogues, the QuickNet pre clamp, gated SE stages,
# BNext's GatedAdd / GapAffine, BHViT's token blocks, a BiBERT layer); each is written, checked (no error, no warning),
# imported with the re-emission check (data + graph SHA-256 identical) and evaluated in onnxruntime in the file's
# element types against the numpy oracle (bitino.int8_ops), bit for bit.
from bitino import int8_ops, qonnx_int8 as qi8  # noqa: E402

_META5 = dict(_META3, **{"bitino.variant": "int8_full", "bitino.model_id": "test-int8",
                         "bitino.calibration": json.dumps({"set": "test", "split": "train", "n": 24, "tensor_sha256": "b" * 64})})


def _int8_variants():
    sys.path.insert(0, os.path.join(ROOT, "tests", "python"))
    import test_int8_full as T1
    import test_int8_full_v2 as T2
    from test_transformer_ops import tiny_bert
    return {"v1-cnn": (T1.tiny_cnn, (5,), None), "v1-bert": (tiny_bert, (3,), None),
            "v1-densenet": (T1.tiny_densenet, (7, True), None),
            "l8-reactnet": (T1.tiny_reactnet, (11,), {"code": "l8"}),
            "lin8-reactnet": (T1.tiny_reactnet, (11,), {"code": "lin8"}),
            "lin8-int-reactnet": (T1.tiny_reactnet, (11,), {"code": "lin8", "int_epilogue": True}),
            "lin8-bireal-relu": (T1.tiny_bireal, (13, True), {"code": "lin8", "int_epilogue": True}),
            "lin8-quicknet-pre": (T1.tiny_quicknet, (19,), {"code": "lin8", "res_pre": True, "int_epilogue": True}),
            "lin8-se-r2b": (T1.tiny_se, (23, "r2b"), {"code": "lin8", "res_pre": True, "identity_blocks": "all", "int_epilogue": True}),
            "gated-bnext": (T2.tiny_bnext, (29,), {"code": "lin8", "int_epilogue": True, "gated": True}),
            "lin8-bhvit-rank3": (T2.tiny_bhvit, (31, True), {"code": "lin8", "int_epilogue": True})}


def _quantize_tiny(name, tmp):
    from bitino import quantize_int8 as QI
    from bitino import quantize_l8 as QL
    sys.path.insert(0, os.path.join(ROOT, "tests", "python"))
    from test_int8_full import _records
    fn, a, opts = _int8_variants()[name]
    g = fn(*a)
    from bitino.ir import pack_bits, unpack_bits
    for n in g.nodes:      # canonical ternary weights (R4): the sign bit of an exact zero weight is +1
        if n.op in ("BDense", "Gather") and len(n.inputs) > (5 if n.op == "BDense" else 3) and n.inputs[5 if n.op == "BDense" else 3]:
            w, z = g.constants[n.inputs[1 if n.op == "BDense" else 0]], g.constants[n.inputs[5 if n.op == "BDense" else 3]]
            bits = unpack_bits(w.data, w.shape[-1]).reshape(w.shape)
            zero = unpack_bits(z.data, z.shape[-1]).reshape(z.shape) < 0
            w.data = pack_bits(np.where(zero, 1, bits).astype(np.int8))
    path = os.path.join(tmp, "f.bitino")
    bfformat.write(g, path)
    cal = _records(g, 24, 1)
    stats = QI.calibrate(path, cal, 99.99 if opts else None, jobs=2)
    shapes = QI._float_shapes(g, QI._feeds(g, cal[0]))
    if opts is None:
        qg, _ = QI.quantize(g, stats, shapes)
    else:
        Q = QI.rewrite(g, stats, shapes, {})
        E = QI._Emitter(Q, {})
        o = {k: v for k, v in opts.items() if k != "gated"}
        qg, _ = QL.fuse(E.emit()[0], Q, E, o)
        if opts.get("gated"):
            qg, _ = QL.fuse_gated(qg, Q, E, o)
    qpath = os.path.join(tmp, "q.bitino")
    bfformat.write(qg, qpath)
    return bfformat.read(qpath), _records(g, 3, 99)


def _int8_roundtrip(qg, tmp):
    from bitino.fetch import data_sha256, graph_sha256
    spec, rep = qi8.write_spec_int8(qg)
    dom = "float32" if len(qg.inputs) == 1 else json.dumps({i["name"]: "float32" if i["dtype"] == "f32" else "ids:50" for i in qg.inputs})
    spec.meta.update(dict(_META5, **{"bitino.input_domain": dom}))
    m = qi8.build_model(spec, "test")
    r = _check_model(m)
    assert r.passed(strict=True), r.to_text()
    g2, info = qgr.import_groups(m)
    qg.manifest, g2.manifest = {}, {}
    bfformat.write(qg, os.path.join(tmp, "a.bitino"))
    bfformat.write(g2, os.path.join(tmp, "b.bitino"))
    same = (data_sha256(os.path.join(tmp, "a.bitino")) == data_sha256(os.path.join(tmp, "b.bitino")),
            graph_sha256(os.path.join(tmp, "a.bitino")) == graph_sha256(os.path.join(tmp, "b.bitino")))
    return m, info, same, rep


def _feeds_of(qg, rec: bytes) -> dict:
    from bitino import quantize_int8 as QI
    return QI._feeds(qg, rec)


@pytest.mark.parametrize("name", sorted(["v1-cnn", "v1-bert", "v1-densenet", "l8-reactnet", "lin8-reactnet", "lin8-int-reactnet",
                                         "lin8-bireal-relu", "lin8-quicknet-pre", "lin8-se-r2b", "gated-bnext", "lin8-bhvit-rank3"]))
def test_int8_full_groups_identity_and_bit_exact(name, tmp_path):
    pytest.importorskip("onnxruntime")
    qg, recs = _quantize_tiny(name, str(tmp_path))
    m, info, same, rep = _int8_roundtrip(qg, str(tmp_path))
    assert same == (True, True), name
    assert sum(v for k, v in info["rules"].items() if k.startswith("re-emission")) == len(qg.nodes)
    sess = qgr.ort_session(m)
    for rec in recs:
        feeds = _feeds_of(qg, rec)
        ref = executor.run(qg, feeds)
        own = qi8.run_ir(qg, feeds)
        ys = sess.run(None, qgr.onnx_feeds(qg, feeds))
        for o, y in zip(qg.outputs, ys):
            a = np.ascontiguousarray(ref[o])
            assert np.array_equal(a.view(np.uint8), np.ascontiguousarray(own[o]).view(np.uint8)), (name, "run_ir")
            assert a.shape == y.shape and np.array_equal(a.view(np.uint8), np.ascontiguousarray(y).view(np.uint8)), (name, o)


def test_int8_l8enc_count_form_exact():
    """L8ENC = sign(w) * #{k : |w| >= beta_k} equals int8_ops.l8_encode: every 4099th positive fp32 pattern, the
    negatives, +-0, +-inf, every boundary and its predecessor; the step-6 7-step bisection (numpy mirror and the
    writer's L8ENC nodes in onnxruntime) gives the same count. Every fp32 input: tools/qonnx_l8enc_proof.py."""
    ort = pytest.importorskip("onnxruntime")
    b = qi8.beta()
    assert b.shape == (127,) and np.all(np.diff(b.astype(np.float64)) > 0)
    pats = np.arange(0, 0x7F800001, 4099, dtype=np.int64).astype(np.uint32)
    bb = b.view(np.uint32)
    pats = np.concatenate([pats, bb, bb - 1, bb + 1, np.array([0, 0x7F800000], np.uint32)])
    w = pats.view(np.float32)
    w = np.concatenate([w, -w])
    cnt = (np.abs(w)[:, None] >= b[None, :]).sum(axis=1)
    assert np.array_equal((np.sign(w).astype(np.int64) * cnt).astype(np.int8), int8_ops.l8_encode(w))
    assert np.array_equal(qi8.l8enc_bisect(w), int8_ops.l8_encode(w))
    wr = qi8.Int8Writer(Graph(), {}, {})
    wr.cur = (type("N", (), {"name": "l8"})(), 0)
    wr._casts = set()
    y = wr.l8enc("w", out="y")
    nodes = [helper.make_node(nd["op"], nd["inputs"], nd["outputs"], **nd["attrs"]) for nd in wr.s.nodes]
    inits = [numpy_helper.from_array(np.asarray(v), k) for k, v in wr.s.inits.items()]
    gr = helper.make_graph(nodes, "l8", [helper.make_tensor_value_info("w", TensorProto.FLOAT, [w.size])],
                           [helper.make_tensor_value_info(y, TensorProto.INT8, [w.size])], inits)
    mo = helper.make_model(gr, opset_imports=[helper.make_opsetid("", 17)])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    out = ort.InferenceSession(mo.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, {"w": w})[0]
    assert np.array_equal(out, int8_ops.l8_encode(w))
    assert np.array_equal(qi8.l8_table(3), int8_ops.l8_decode(np.arange(-128, 128), 3)) and qi8._l8_exp_of(qi8.l8_table(-2)) == -2


def test_int8_requant_shift_construction_exact():
    """REQUANT (Mul 2^left, INT64 Where saturation, SQRDMULH and the rounding shift as UINT64 BitShift offsets) in
    onnxruntime equals int8_ops.requant: random int32, a dense range around 0, the int32 extremes; scalar and per-channel
    (M, shift), shifts -31..3."""
    ort = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.integers(-2 ** 31, 2 ** 31, 20000), np.arange(-3000, 3000),
                        [-2 ** 31, -2 ** 31 + 1, 2 ** 31 - 1, 2 ** 31 - 2, 0, 1, -1]]).astype(np.int64)
    x = x[: (x.size // 4) * 4].reshape(-1, 4)
    cases = [(int8_ops.quantize_multiplier(v)) for v in (0.7, 1e-6, 3.9e-3, 1.7, 0.25)] + [(1 << 30, -31), (2 ** 31 - 1, 3)]
    per_ch = (np.array([m for m, _ in cases[:4]], np.int64), np.array([s for _, s in cases[:4]], np.int64))
    for M, sh in cases + [per_ch]:
        wr = qi8.Int8Writer(Graph(), {}, {})
        wr.cur = (type("N", (), {"name": "rq"})(), 0)
        wr._casts = set()
        y = wr.requant("x", M, sh, key="rq")
        nodes = [helper.make_node(nd["op"], nd["inputs"], nd["outputs"], **nd["attrs"]) for nd in wr.s.nodes]
        inits = [numpy_helper.from_array(np.asarray(v), k) for k, v in wr.s.inits.items()]
        gr = helper.make_graph(nodes, "rq", [helper.make_tensor_value_info("x", TensorProto.INT64, list(x.shape))],
                               [helper.make_tensor_value_info(y, TensorProto.INT64, list(x.shape))], inits)
        mo = helper.make_model(gr, opset_imports=[helper.make_opsetid("", 17)])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        out = ort.InferenceSession(mo.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        assert np.array_equal(out, int8_ops.requant(x, M, sh)), (M, sh)


def test_int8_rejections(tmp_path):
    """A group that is not exactly the canonical pattern of the node extracted from it is rejected: a changed rounding
    constant, an INT64 Clip in place of the Where saturation, an informational attribute contradicting a derived list
    entry; a changed multiplier is a different IR node (accepted, other hashes)."""
    qg, _ = _quantize_tiny("lin8-int-reactnet", str(tmp_path))
    spec, _rep = qi8.write_spec_int8(qg)
    spec.meta.update(dict(_META5, **{"bitino.input_domain": "float32"}))
    base = qi8.build_model(spec, "test")
    qgr.import_groups(base)
    import copy
    m = copy.deepcopy(base)                     # the rounding term 2^(n-1) of a shift changed
    cons = {i: nd for nd in m.graph.node for i in nd.input}
    two62 = "bitino/i64/4611686018427387904"
    ad = next(nd for nd in m.graph.node if nd.op_type == "Add" and nd.input[1].startswith("bitino/i64/")
              and nd.input[1] not in (two62, "bitino/i64/1073741824") and nd.output[0] in cons
              and two62 in cons[nd.output[0]].input)
    m.graph.initializer.append(numpy_helper.from_array(np.array(12345, np.int64), "test/r"))
    ad.input[1] = "test/r"
    with pytest.raises(qgr.GroupImportError):
        qgr.import_groups(m)
    m = copy.deepcopy(base)                     # an INT64 saturation written as Clip
    w = next(nd for nd in m.graph.node if nd.op_type == "Greater" and _get_meta(nd, "bitino.name"))
    wh = next(nd for nd in m.graph.node if nd.op_type == "Where" and nd.input[0] == w.output[0])
    clip = helper.make_node("Clip", [w.input[0], "", w.input[1]], [wh.output[0]], name="clip")
    for p in wh.metadata_props:
        clip.metadata_props.add(key=p.key, value=p.value)
    pos = list(m.graph.node).index(wh)
    m.graph.node.remove(wh)
    m.graph.node.insert(pos, clip)
    m.graph.node.remove(w)
    with pytest.raises(qgr.GroupImportError):
        qgr.import_groups(m)
    m = copy.deepcopy(base)                     # a multiplier changed: another IR node, different data hash
    mi = next(t for t in m.graph.initializer if t.name.endswith(":M"))
    arr = numpy_helper.to_array(mi).copy()
    arr.flat[0] += 1
    mi.CopyFrom(numpy_helper.from_array(arr, mi.name))
    g2, _ = qgr.import_groups(m)
    k = mi.name[:-2]
    assert not np.array_equal(np.asarray(g2.constants[k].data), np.asarray(qg.constants[k].data))
    with pytest.raises(qgr.GroupImportError):   # informational list entries: the derived ones must agree
        qi8.apply_attrs_meta_i8("I8Concat", {"zeros": [qi8.INFO, 5], "mults": [0, 7]}, {"zeros": [3, 6]}, "c")
    assert qi8.apply_attrs_meta_i8("I8Concat", {"zeros": [qi8.INFO, 5], "mults": [0, 7]}, {"zeros": [3, 5]}, "c")["zeros"] == [3, 5]
