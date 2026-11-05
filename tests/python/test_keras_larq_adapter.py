"""Keras/Larq adapter patterns of the remaining Larq Zoo models, on tiny random models.

Needs the Larq reference env (tensorflow, larq, larq_zoo). That env has no pytest,
so the file also runs as a script:
  .cache/venv-larq/bin/python tests/python/test_keras_larq_adapter.py
Each test builds a small model with the official larq_zoo building blocks / custom
quantizers, converts it, and compares the numpy executor (float64) with the Keras
model evaluated in float64 (the strict semantic check of docs/numerical-correctness.md).
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

try:
    import larq as lq
    import tensorflow as tf
    from larq_zoo.core import utils as zoo_utils
    from larq_zoo.literature import dorefanet, xnornet
    HAVE_LARQ = True
except ImportError:  # pragma: no cover - outside the Larq env
    HAVE_LARQ = False

if "pytest" in sys.modules and not HAVE_LARQ:
    import pytest
    pytest.skip("Larq reference env not available", allow_module_level=True)

from bitino import executor  # noqa: E402
from bitino.adapters import keras_larq  # noqa: E402


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


def _check(build, input_shape, ops_expected, seed=0, tol=1e-5):
    """build() -> keras model (logits output). Converts the float32 model; compares
    the float64 numpy executor with a float64 rebuild carrying the same weights."""
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
        x = rng.random((3,) + tuple(input_shape))
        want = m64(x, training=False).numpy()
    finally:
        tf.keras.backend.set_floatx("float32")
    for i in range(len(x)):
        got = executor.run(graph, {"input": x[i:i + 1]}, float_dtype=np.float64)[graph.outputs[0]]
        err = float(np.max(np.abs(got.reshape(-1) - want[i].reshape(-1))))
        assert err < tol, (i, err)
    return graph


def test_dorefa_qbconv_qbdense():
    def build():
        inp = tf.keras.Input((10, 10, 3))
        q = dict(input_quantizer=lq.quantizers.DoReFaQuantizer(k_bit=2),
                 kernel_quantizer=dorefanet.magnitude_aware_sign_unclipped, use_bias=False)
        y = tf.keras.layers.Conv2D(8, 3, strides=1, padding="valid", use_bias=True)(inp)
        y = lq.layers.QuantConv2D(16, 3, padding="same", **q)(y)
        y = tf.keras.layers.BatchNormalization(scale=False, epsilon=1e-4)(y)
        y = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(y)
        y = tf.keras.layers.Flatten()(y)
        y = lq.layers.QuantDense(12, **q)(y)
        y = tf.keras.layers.BatchNormalization(scale=False, epsilon=1e-4)(y)
        y = tf.keras.layers.Activation("clip_by_value_activation")(y)
        y = tf.keras.layers.Dense(5)(y)
        return tf.keras.Model(inp, y)
    _check(build, (10, 10, 3), {"QBConv2D": 1, "QBDense": 1, "Activation": 1})


def test_xnor_and_binary_weight_stem():
    def build():
        inp = tf.keras.Input((9, 9, 3))
        y = lq.layers.QuantConv2D(8, 3, strides=2, padding="same", input_quantizer=None,
                                  kernel_quantizer="ste_sign", kernel_constraint="weight_clip", use_bias=False)(inp)
        y = tf.keras.layers.BatchNormalization(scale=False)(y)
        y = lq.layers.QuantConv2D(16, 3, padding="same", input_quantizer="ste_sign",
                                  kernel_quantizer=xnornet.xnor_weight_scale, use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization(scale=False)(y)
        y = tf.keras.layers.Activation("relu")(y)
        y = tf.keras.layers.Flatten()(y)
        return tf.keras.Model(inp, tf.keras.layers.Dense(4, use_bias=False)(y))
    g = _check(build, (9, 9, 3), {"Conv2D": 1, "BConv2D": 1})
    stem = g.constants[g.nodes[0].inputs[1]].data
    assert set(np.unique(stem)) <= {-1.0, 1.0}


def test_real_to_binary_gating_prelu_identity_softmax():
    def build():
        inp = tf.keras.Input((8, 8, 8))
        conv_in = tf.keras.layers.BatchNormalization()(inp)
        conv_out = lq.layers.QuantConv2D(8, 3, padding="same", input_quantizer="ste_sign", kernel_quantizer="ste_sign",
                                         use_bias=False)(conv_in)
        z = zoo_utils.global_pool(conv_in, name="gp")
        z = tf.keras.layers.Dense(2, activation="relu", use_bias=False)(z)
        z = tf.keras.layers.Dense(8, activation="sigmoid", use_bias=False)(z)
        z = tf.keras.layers.Reshape((1, 1, 8))(z)
        y = tf.keras.layers.Multiply()([conv_out, z])
        y = tf.keras.layers.PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.RandomUniform(0, 0.5))(y)
        y = tf.keras.layers.Add()([y, inp])
        y = tf.keras.layers.Lambda(lambda t: t, name="blk_out")(y)
        y = zoo_utils.global_pool(y, name="head")
        logits = tf.keras.layers.Dense(5, name="logits")(y)
        return tf.keras.Model(inp, tf.keras.layers.Softmax()(logits))
    _check(build, (8, 8, 8), {"ChannelGate": 1, "BConv2D": 1, "Add": 1, "Softmax": 1})


def test_meliusnet_lambdas():
    def build():
        inp = tf.keras.Input((8, 8, 8))
        parts = zoo_utils.TFOpLayer(tf.split, 4, axis=-1, name="g_split")(inp)
        ys = [tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=False, name=f"g_conv{i}")(p) for i, p in enumerate(parts)]
        x = zoo_utils.TFOpLayer(tf.concat, axis=-1, name="g_concat")(ys)
        w = tf.keras.layers.BatchNormalization()(x)
        w = lq.layers.QuantConv2D(8, 3, padding="same", input_quantizer=lq.quantizers.SteSign(1.3),
                                  kernel_quantizer=lq.quantizers.SteSign(1.3), use_bias=False)(w)
        x = zoo_utils.TFOpLayer(tf.concat, axis=-1, name="d_concat")([x, w])
        w = tf.keras.layers.BatchNormalization()(x)
        w = lq.layers.QuantConv2D(8, 3, padding="same", input_quantizer=lq.quantizers.SteSign(1.3),
                                  kernel_quantizer=lq.quantizers.SteSign(1.3), use_bias=False)(w)
        f_in = int(x.shape[-1])
        x = tf.keras.layers.Lambda(lambda x_: x_[0] + tf.pad(x_[1], [[0, 0], [0, 0], [0, 0], [f_in - 8, 0]]),
                                   name="i_merge")([x, w])
        return tf.keras.Model(inp, zoo_utils.global_pool(x, name="head"))
    g = _check(build, (8, 8, 8), {"Conv2D": 1, "Concat": 1, "Pad": 1, "Add": 1, "BConv2D": 2})
    conv = next(n for n in g.nodes if n.op == "Conv2D")
    assert conv.attrs["groups"] == 4 and conv.name == "g_concat"


def test_legacy_quicknet_xl_squeeze_excite_tfoplambda():
    """Legacy QuickNet-XL (larq/zoo @40f4cdc quicknet.py): depthwise x8 stem with ReLU, squeeze-and-excite gate
    via tf.reshape(gate, [-1, 1, 1, C]) and `x *= gate` (TFOpLambda layers), pad_values=1 binary conv."""
    def build():
        inp = tf.keras.Input((8, 8, 3))
        x = tf.keras.layers.Conv2D(2, 3, strides=2, padding="same", use_bias=False)(inp)
        x = tf.keras.layers.DepthwiseConv2D(3, depth_multiplier=8, padding="same", use_bias=False, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
        y = zoo_utils.global_pool(x)
        y = tf.keras.layers.Dense(1, activation="relu", use_bias=False)(y)
        y = tf.keras.layers.Dense(16, activation="sigmoid", use_bias=False)(y)
        y = tf.reshape(y, [-1, 1, 1, 16])
        z = lq.layers.QuantConv2D(16, 3, padding="same", pad_values=1.0, input_quantizer=lq.quantizers.SteSign(1.25),
                                  kernel_quantizer=lq.quantizers.SteSign(1.25), use_bias=False, activation="relu")(x)
        z = tf.keras.layers.BatchNormalization(epsilon=1e-5)(z)
        z *= y
        x = tf.keras.layers.add([z, x])
        x = zoo_utils.global_pool(tf.keras.layers.Activation("relu")(x))
        logits = tf.keras.layers.Dense(5)(x)
        return tf.keras.Model(inp, tf.keras.layers.Activation("softmax")(logits))
    g = _check(build, (8, 8, 3), {"ChannelGate": 1, "BConv2D": 1, "Conv2D": 2, "Add": 1})
    dw = [n for n in g.nodes if n.op == "Conv2D"][1]
    assert dw.attrs["groups"] == 2 and dw.attrs["activation"] == "relu"


def test_unknown_lambda_rejected():
    inp = tf.keras.Input((4, 4, 2))
    y = tf.keras.layers.Lambda(lambda t: t * 2.0)(inp)
    try:
        keras_larq.convert(tf.keras.Model(inp, y))
    except NotImplementedError:
        return
    raise AssertionError("an unknown Lambda must be rejected")


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print(f"[ PASS ] {n}")
    print(f"{len(names)} passed")
