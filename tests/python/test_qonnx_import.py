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
