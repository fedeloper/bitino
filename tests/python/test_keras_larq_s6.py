"""Slice S6 (Larq face recognition + BRILLsson speech) patterns.

Part 1 (any env with numpy + a host bitino_bench): the op mix of BinaryFaceNet / BRILLsson -- grouped BConv2D
with input_binarize "gt", zero padding and post_add, depthwise BConv2D (groups = C, one channel per group), the
folded 1x1 separable BConv2D, Conv2D pre-sigmoid gate + ChannelGate on an [N,1,1,C] gate, PReLU, Add, and the
GlobalMaxPool as a full-window MaxPool2D + Reshape -- numpy executor vs C++ (reference / neon stages), bit-exact,
with exact zeros at the binarization inputs (gt maps them to -1).

Part 2 (Larq reference env; also runs as a script: .cache/venv-larq/bin/python tests/python/test_keras_larq_s6.py):
tiny Keras models with the BinaryFaceNet layers (DoReFa(k_bit=1, mode="weights") QuantConv2D with groups,
QuantDepthwiseConv2D, 1x1 QuantSeparableConv2D, Conv2D(sigmoid) * x attention, BN on a 2-D Dense output) and a
GlobalMaxPooling2D head, converted by keras_larq and compared in float64 with the float64 Keras model.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters.keras_larq import _dorefa_w1_operand  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))

try:
    import larq as lq
    import tensorflow as tf
    HAVE_LARQ = True
except ImportError:  # pragma: no cover - outside the Larq env
    HAVE_LARQ = False

try:
    import pytest
except ImportError:  # the Larq env has no pytest
    pytest = None


def _skipif(cond, reason):
    if pytest is None:
        return lambda f: f
    return pytest.mark.skipif(cond, reason=reason)


# ---------------------------------------------------------------- part 1: numpy executor vs C++ runtime

def binaryfacenet_like(seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 12, 12, 8])

    def cst(name, arr, dtype="f32", layout=""):
        return g.add_constant(name, np.asarray(arr), dtype, layout)

    def pm1(shape):
        return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)

    def bconv(t, tag, cout, cin_g, k, stride, groups, pads, bias=True):
        return g.add_node("BConv2D", [t, cst(tag + "w", pm1((cout, k, k, cin_g)), "bit", "OHWI_bit"), "",
                                      cst(tag + "pa", rng.normal(size=cout)) if bias else "", ""],
                          {"strides": [stride, stride], "dilations": [1, 1], "pads": pads, "groups": groups,
                           "pad_value": 0.0, "input_binarize": "gt", "activation": "none", "output": "f32"})

    def bn(t, tag, c, quant=False):
        scale = np.where(rng.random(c) < 0.5, 0.5, -1.0) if quant else rng.random(c) + 0.5
        shift = np.round(rng.normal(size=c) * 4) / 8 if quant else rng.normal(size=c)
        return g.add_node("ChannelAffine", [t, cst(tag + "s", scale), cst(tag + "b", shift)])

    def prelu(t, tag, c):
        return g.add_node("Activation", [t, cst(tag, rng.random(c) * 0.5)], {"type": "prelu"})

    a = prelu(bn(x, "bn0", 8, quant=True), "p0", 8)                                 # exact zeros reach "gt"
    # one near-dead channel (scale 1e-9): its values sit around the DoReFa float32 threshold 2^-24 * tanh(max|x|)
    a = g.add_node("ChannelAffine", [a, cst("dead_s", np.r_[np.ones(7), 1e-9 * 2 ** 1.5]), cst("dead_b", np.zeros(8))])
    a = _dorefa_w1_operand(g, a, [1, 12, 12, 8], "op0")
    t = prelu(bn(bconv(a, "g4", 16, 2, 3, 2, 4, [0, 1, 0, 1], bias=False), "bn1", 16), "p1", 16)   # 6x6, groups 4
    gate = g.add_node("Conv2D", [g.add_node("MaxPool2D", [t], {"kernel": [6, 6], "strides": [1, 1], "pads": [0, 0, 0, 0]}),
                                 cst("gw", rng.normal(size=(16, 1, 1, 16)), layout="OHWI"), cst("gb", rng.normal(size=16))],
                      {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "activation": "none"})
    t = g.add_node("ChannelGate", [t, gate], {"gate": "sigmoid"})
    d = bconv(_dorefa_w1_operand(g, t, [1, 6, 6, 16], "op1"), "dw", 16, 1, 1, 1, 16, [0, 0, 0, 0])  # depthwise, groups = C
    t = prelu(g.add_node("Add", [t, d]), "p2", 16)
    s = bconv(bn(t, "bn2", 16, quant=True), "sep", 16, 16, 1, 1, 1, [0, 0, 0, 0])     # folded 1x1 separable
    t = prelu(g.add_node("Add", [t, s]), "p3", 16)
    t = bconv(t, "g2", 32, 8, 3, 1, 2, [1, 1, 1, 1])                                  # same padding, groups 2
    pooled = g.add_node("Reshape", [g.add_node("MaxPool2D", [t], {"kernel": [6, 6], "strides": [1, 1], "pads": [0, 0, 0, 0]})],
                        {"shape": [1, 32]})
    emb = g.add_node("Dense", [pooled, cst("fw", rng.normal(size=(12, 32))), ""])
    emb = g.add_node("ChannelAffine", [emb, cst("es", rng.random(12) + 0.5), cst("eb", rng.normal(size=12))])
    g.set_outputs(emb, pooled)
    return g


@_skipif(not os.path.exists(BENCH), "host bitino_bench not built")
def test_numpy_executor_bit_exact_with_cpp():
    g = binaryfacenet_like(5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.bitino")
        bfformat.write(g, path)
        g2 = bfformat.read(path)
        rng = np.random.default_rng(3)
        xs = [(np.round(rng.normal(size=(1, 12, 12, 8)) * 8 * s) / 8).astype(np.float32) for s in (0.5, 1.0, 2.0)]
        np.concatenate(xs).tofile(os.path.join(d, "in.f32"))
        zeros = near = 0
        wants = []
        for xin in xs:
            res = executor.run(g2, {"input": xin}, keep_intermediates=True)
            zeros += sum(int(np.sum(res[n.inputs[0]] == 0)) for n in g2.nodes if n.op in ("BConv2D", "AddBroadcast"))
            # dead-channel values that the threshold flips (0 < x <= T): the operand changes their sign
            op0 = [n for n in g2.nodes if n.name == "op0/dorefa_operand"][0]
            near += int(np.sum((res[op0.inputs[0]] > 0) & (res[op0.outputs[0]] <= 0)))
            wants.append(np.concatenate([res[g2.outputs[0]].ravel(), res[g2.outputs[1]].ravel()]))
        assert zeros > 0, "test inputs should produce exact zeros at binarization"
        assert near > 0, "the near-dead channel should put values below the DoReFa threshold"
        for stage in ("reference", "neon"):
            out = os.path.join(d, f"out_{stage}.f32")
            subprocess.run([BENCH, "--model", path, "--inputs", os.path.join(d, "in.f32"), "--outputs", out,
                            "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"],
                           check=True, capture_output=True)
            got = np.fromfile(out, np.float32).reshape(len(xs), -1)
            for i in range(len(xs)):
                np.testing.assert_array_equal(got[i], wants[i], err_msg=stage)


# ---------------------------------------------------------------- part 2: Keras/Larq patterns (Larq env)

def _randomize_bn(model, rng):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            ws = layer.get_weights()
            c = ws[0].shape[0]
            new = []
            if layer.scale:
                new.append(rng.uniform(0.5, 1.5, c))
            if layer.center:
                new.append(rng.normal(0, 0.3, c))
            new += [rng.normal(0, 0.3, c), rng.uniform(0.5, 2.0, c)]
            layer.set_weights([w.astype(ws[0].dtype) for w in new])


def _check(build, input_shape, ops_expected, seed=0, tol=1e-6):
    from bitino.adapters import keras_larq
    rng = np.random.default_rng(seed)
    tf.keras.backend.set_floatx("float32")
    tf.random.set_seed(seed)
    m32 = build()
    _randomize_bn(m32, rng)
    graph, _ = keras_larq.convert(m32)
    ops = graph.op_counts()
    for op, cnt in ops_expected.items():
        assert ops.get(op, 0) == cnt, (op, ops)
    tf.keras.backend.set_floatx("float64")
    try:
        m64 = build()
        m64.set_weights([w.astype(np.float64) for w in m32.get_weights()])
        x = rng.standard_normal((3,) + tuple(input_shape))
        want = [m64(x[i:i + 1], training=False) for i in range(len(x))]
        want = [[np.asarray(t) for t in (w if isinstance(w, (list, tuple)) else [w])] for w in want]
    finally:
        tf.keras.backend.set_floatx("float32")
    for i in range(len(x)):
        res = executor.run(graph, {"input": x[i:i + 1]}, float_dtype=np.float64)
        for k, name in enumerate(graph.outputs):
            err = float(np.max(np.abs(res[name].reshape(-1) - want[i][k].reshape(-1))))
            assert err < tol * max(1.0, float(np.max(np.abs(want[i][k])))), (i, k, err)
    return graph


def _dorefa_w1():
    return lq.quantizers.DoReFa(k_bit=1, mode="weights")


@_skipif(not HAVE_LARQ, "Larq reference env not available")
def test_binaryfacenet_layers():
    def build():
        q = lambda: dict(input_quantizer=_dorefa_w1(), kernel_quantizer=_dorefa_w1())  # noqa: E731
        inp = tf.keras.Input((12, 12, 3))
        x = tf.keras.layers.Conv2D(8, 1, use_bias=False)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x0 = tf.keras.layers.PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.RandomUniform(0, .5))(x)
        x = lq.layers.QuantConv2D(8, 3, strides=3, padding="valid", groups=4, use_bias=False, **q())(x0)    # 4x4
        x = tf.keras.layers.BatchNormalization()(x)
        x = lq.layers.QuantConv2D(8, 1, strides=4, padding="valid", use_bias=False, **q())(x)                # 1x1
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Reshape((1, 1, 8))(x)
        x = tf.keras.layers.Conv2D(2, 1, use_bias=True)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        gate = tf.keras.layers.Conv2D(8, 1, activation="sigmoid", use_bias=True)(x)
        x = tf.keras.layers.Multiply()([x0, gate])
        x = lq.layers.QuantConv2D(16, 3, strides=2, padding="same", groups=4, use_bias=False, **q())(x)     # 6x6
        x = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        d = lq.layers.QuantDepthwiseConv2D(1, padding="same", use_bias=True, input_quantizer=_dorefa_w1(),
                                           depthwise_quantizer=_dorefa_w1())(x1)
        x = tf.keras.layers.Add()([x1, d])
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        s = lq.layers.QuantSeparableConv2D(16, 1, padding="same", use_bias=True, input_quantizer=_dorefa_w1(),
                                           depthwise_quantizer=_dorefa_w1(), pointwise_quantizer=_dorefa_w1())(x)
        x = tf.keras.layers.Add()([x, s])
        x = tf.keras.layers.Conv2D(8, 3, strides=3, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.Model(inp, tf.keras.layers.Activation("linear")(x))
    g = _check(build, (12, 12, 3), {"BConv2D": 5, "ChannelGate": 1})
    groups = sorted(n.attrs["groups"] for n in g.nodes if n.op == "BConv2D")
    assert groups == [1, 1, 4, 4, 16], groups
    assert all(n.attrs["input_binarize"] == "gt" for n in g.nodes if n.op == "BConv2D")


@_skipif(not HAVE_LARQ, "Larq reference env not available")
def test_dorefa_w1_rounding_band():
    """The live quantizer maps tiny positive inputs to -1 (float32 rounding band); the IR uses gt 0."""
    q = _dorefa_w1()
    x = np.array([2.0, 1e-9, 0.0, -0.0, -1e-9, 1e-3, -1e-3], np.float32)
    got = q(tf.constant(x)).numpy()
    assert list(got) == [1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0], got


@_skipif(not HAVE_LARQ, "Larq reference env not available")
def test_dorefa_w1_threshold_rule_live():
    from bitino.adapters import keras_larq
    keras_larq._check_dorefa_w1_input("probe", _dorefa_w1())   # raises if q(x) != (x > 2^-24 * max|tanh x|)


@_skipif(not HAVE_LARQ, "Larq reference env not available")
def test_dorefa_w1_band_matches_float32_source():
    """Near-dead input channels (values ~1e-8): the IR operand reproduces the float32 layer exactly."""
    from bitino.adapters import keras_larq
    tf.keras.backend.set_floatx("float32")
    inp = tf.keras.Input((6, 6, 4))
    y = lq.layers.QuantConv2D(8, 3, padding="same", groups=2, use_bias=False, input_quantizer=_dorefa_w1(),
                              kernel_quantizer=_dorefa_w1())(inp)
    model = tf.keras.Model(inp, y)
    g, _ = keras_larq.convert(model)
    rng = np.random.default_rng(4)
    for big in (20.0, 0.7):
        x = rng.standard_normal((1, 6, 6, 4)).astype(np.float32) * np.float32(big / 4)
        x[..., 1] = (rng.standard_normal((1, 6, 6)) * 6e-8).astype(np.float32)     # around 2^-24 * m
        x[0, 0, 0, 0] = big
        want = model(x, training=False).numpy()
        got = executor.run(g, {"input": x})[g.outputs[0]]
        np.testing.assert_array_equal(got.reshape(want.shape), want)
        gt0 = np.where(x > 0, 1, -1)
        assert np.any((x > 0) & (_dorefa_w1()(tf.constant(x)).numpy() < 0)), "test must exercise the band"
        del gt0


@_skipif(not HAVE_LARQ, "Larq reference env not available")
def test_global_max_pool_head():
    def build():
        inp = tf.keras.Input((10, 8, 1))
        x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = lq.layers.QuantConv2D(8, 3, padding="same", pad_values=1.0, input_quantizer="ste_sign",
                                  kernel_quantizer="ste_sign", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        pooled = tf.keras.layers.GlobalMaxPool2D()(x)
        emb = lq.layers.QuantDense(6, activation=None)(pooled)
        return tf.keras.Model(inp, [emb, pooled])
    g = _check(build, (10, 8, 1), {"MaxPool2D": 1, "Reshape": 1, "BConv2D": 1})
    mp = [n for n in g.nodes if n.op == "MaxPool2D"][0]
    assert mp.attrs["kernel"] == [5, 4]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if name == "test_numpy_executor_bit_exact_with_cpp" and not os.path.exists(BENCH):
                print("SKIP", name)
                continue
            fn()
            print("PASS", name)
