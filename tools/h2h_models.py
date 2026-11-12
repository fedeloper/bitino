#!/usr/bin/env python3
"""Head-to-head (Bitino vs LCE vs TFLite): rebuild the Larq Zoo models in Keras/larq, load the original .h5 weights
with h5py, check them against Bitino's reference outputs, and convert them to .tflite.

Runs in the LCE reference env (conda `lce`: tensorflow 2.16.1 + tf_keras 2.16.0, larq 0.14.0,
larq_compute_engine 0.16.0), read-only:

  TF_USE_LEGACY_KERAS=1 ~/anaconda3/envs/lce/bin/python tools/h2h_models.py build --out DIR [--models a,b]

larq_zoo's own code is never imported or executed (project rule, the research notes (not published)). The topologies below
are re-written from the larq_zoo 2.4.0 source (read only, .cache/repos/larq-zoo @ a4b933dc), using only the larq
package's layers and quantizers. The .h5 files are weights-only: they are read with h5py and assigned layer by layer
in the stored `layer_names` order; the names of the weighted layers of the rebuilt model must equal the stored ones,
in order, and every tensor shape must match (else the build fails).

Per model, written to DIR/<model_id>/:
  lce.tflite          lce.convert_keras_model(model)  (LCE 0.16.0 converter, target "arm")
  tflite_fp32.tflite  tf.lite.TFLiteConverter.from_keras_model(model): binarization stays as float SIGN ops
  tflite_int8.tflite  the same with full-integer post-training quantization (int8 in/out kernels; float fallback only
                      where TFLite has no int8 kernel, listed in the report), calibrated on N training images of
                      .cache/data/dev1000_imagenet100train_larq (disjoint from the eval set)
  input0.f32          eval sample 0 (larq_zoo preprocessing, NHWC float32), the input every runtime is timed on
  input0_int8.bin     the same sample quantized with the int8 model's input scale / zero point
  host_outputs.npz    probabilities of every model variant on the 128 eval samples (host interpreters)
  build_report.json   h5 check, Keras-vs-reference agreement, converter op counts, file sizes / SHA-256, host agreement
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2h_tflite_names  # noqa: E402
MAIN = os.environ.get("BITINO_MAIN", "~/BNN arm developments")  # artifacts/.cache live here

# model_id -> (builder name, h5 file under .cache/checkpoints/<model_id>/, expected sha256)
MODELS = {
    "larq-zoo-quicknet-small": ("quicknet_small", "quicknet_small_weights.h5",
                                "1ac3b07df7f5a911dd0b49febb2486428ddf1ca130297c403815dfae5a1c71a2"),
    "larq-zoo-quicknet": ("quicknet", "quicknet_weights.h5",
                          "8aba9e4e5f8d342faef04a0b2ae8e562da57dbb7d15162e8b3e091c951ba756c"),
    "larq-zoo-quicknet-large": ("quicknet_large", "quicknet_large_weights.h5",
                                "c5158e8a59147b31370becd937825f4db8a5cdf308314874f678f596629be45c"),
    "larq-zoo-birealnet": ("birealnet", "birealnet_weights.h5",
                           "6e6efac1584fcd60dd024198c87f42eb53b5ec719a5ca1f527e1fe7e8b997117"),
    "larq-zoo-binary-resnet-e18": ("resnet_e18", "resnet_e_18_weights.h5",
                                   "bde4a64d42c164a7b10a28debbe1ad5b287c499bc0247ecb00449e6e89f3bf5b"),
    "larq-zoo-xnornet": ("xnornet", "xnornet_weights.h5",
                         "e6ba24f785655260ae76a2ef1fab520e3528243d9c8fac430299cd81dbeabe10"),
    "larq-zoo-binary-alexnet": ("binary_alexnet", "binary_alexnet_weights.h5",
                                "7fc065c47c5c1d92389e0bb988ce6df6a4fa09d803b866e2ba648069d6652d63"),
    "larq-zoo-binary-densenet28": ("densenet28", "binary_densenet_28_weights.h5",
                                   "21fe3ca03eed244df9c41a2219876fcf03e73800932ec96a3e2a76af4747ac53"),
    "larq-zoo-binary-densenet37": ("densenet37", "binary_densenet_37_weights.h5",
                                   "8056a5d52c3ed86a934893987d09a06f59a5166aa9bddcaedb050f111d0a7d76"),
    "larq-zoo-binary-densenet37-dilated": ("densenet37_dilated", "binary_densenet_37_dilated_weights.h5",
                                           "15c1bcd79b8dc22971382fbf79acf364a3f51049d0e584a11533e6fdbb7363d3"),
    "larq-zoo-binary-densenet45": ("densenet45", "binary_densenet_45_weights.h5",
                                   "d00a0d26fbd2dba1bfba8c0306c770f3aeea5c370e99f963bb239bd916f72c37"),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- topologies
# Each builder mirrors the layer creation order of the larq_zoo factory (so Keras auto-names and the h5
# `layer_names` coincide) and ends with the same float32 softmax activation.

def _builders():
    import larq as lq
    import tensorflow as tf

    L = tf.keras.layers

    def global_pool(x):  # larq_zoo.core.utils.global_pool: AveragePooling2D(full map) + Flatten
        x = L.AveragePooling2D(pool_size=x.shape[1:3])(x)
        return L.Flatten()(x)

    def head_softmax(x):
        return L.Activation("softmax", dtype="float32")(x)

    # ---- QuickNet v1.0 (sota/quicknet.py)
    def quicknet(blocks, filters):
        q = lambda: lq.quantizers.SteSign(clip_value=1.25)  # noqa: E731
        inp = L.Input((224, 224, 3))
        x = lq.layers.QuantConv2D(filters[0] // 4, (3, 3), padding="same", strides=2, use_bias=False)(inp)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)
        x = lq.layers.QuantDepthwiseConv2D((3, 3), padding="same", strides=2, use_bias=False)(x)
        x = L.BatchNormalization(scale=False, center=False)(x)
        x = lq.layers.QuantConv2D(filters[0], 1, use_bias=False)(x)
        x = L.BatchNormalization()(x)
        for nb, f in zip(blocks, filters):
            for _ in range(nb):
                if f != x.shape[-1]:  # transition block
                    x = L.Activation("relu")(x)
                    x = L.MaxPool2D(pool_size=2, strides=1)(x)
                    x = L.DepthwiseConv2D((3, 3), padding="same", strides=2, trainable=False, use_bias=False)(x)
                    x = lq.layers.QuantConv2D(f, (1, 1), use_bias=False)(x)
                    x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
                r = x
                x = lq.layers.QuantConv2D(int(x.shape[-1]), (3, 3), activation="relu", input_quantizer=q(),
                                          kernel_quantizer=q(), kernel_constraint=lq.constraints.WeightClip(1.25),
                                          padding="same", pad_values=1.0, use_bias=False)(x)
                x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
                x = x + r
        x = L.Activation("relu")(x)
        x = global_pool(x)
        x = lq.layers.QuantDense(1000)(x)
        return tf.keras.Model(inp, head_softmax(x))

    # ---- Bi-Real Net (literature/birealnet.py)
    def birealnet():
        inp = L.Input((224, 224, 3))
        kw = dict(input_quantizer="approx_sign", kernel_quantizer="magnitude_aware_sign", kernel_constraint="weight_clip")

        def block(x, double=False, filters=None):
            fin = x.shape[-1]
            fout = filters or fin if not double else 2 * fin
            sc = x
            if fin != fout:
                sc = L.AvgPool2D(2, strides=2, padding="same")(sc)
                sc = L.Conv2D(fout, (1, 1), use_bias=False)(sc)
                sc = L.BatchNormalization(momentum=0.8)(sc)
            x = lq.layers.QuantConv2D(fout, (3, 3), strides=1 if fout == fin else 2, padding="same", use_bias=False, **kw)(x)
            x = L.BatchNormalization(momentum=0.8)(x)
            return L.add([x, sc])

        x = L.Conv2D(64, (7, 7), strides=2, padding="same", use_bias=False)(inp)
        x = L.BatchNormalization(momentum=0.8)(x)
        x = L.MaxPool2D((3, 3), strides=2, padding="same")(x)
        x = block(x, filters=64)
        for _ in range(3):
            x = block(x)
        for _ in range(3):
            x = block(x, double=True)
            for _ in range(3):
                x = block(x)
        x = global_pool(x)
        x = L.Dense(1000)(x)
        return tf.keras.Model(inp, head_softmax(x))

    # ---- Binary ResNet-E 18 (literature/resnet_e.py)
    def resnet_e18():
        q = lambda: lq.quantizers.SteSign(clip_value=1.25)  # noqa: E731
        inp = L.Input((224, 224, 3))

        def block(x, filters, strides):
            if x.shape[-1] != filters:
                r = L.AvgPool2D(pool_size=2, strides=2)(x)
                r = L.Conv2D(filters, kernel_size=1, use_bias=False)(r)
                r = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(r)
            else:
                r = x
            x = lq.layers.QuantConv2D(filters, kernel_size=3, strides=strides, padding="same", input_quantizer=q(),
                                      kernel_quantizer=q(), kernel_constraint=lq.constraints.WeightClip(1.25),
                                      use_bias=False)(x)
            x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            return L.add([x, r])

        x = L.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(inp)
        x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = L.Activation("relu")(x)
        x = L.MaxPool2D(3, strides=2, padding="same")(x)
        x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        for b, (n, f) in enumerate(zip([2, 2, 2, 2], [64, 128, 256, 512])):
            for i in range(n * 2):
                x = block(x, f, 1 if b == 0 or i != 0 else 2)
        x = L.Activation("relu")(x)
        x = global_pool(x)
        x = L.Dense(1000)(x)
        return tf.keras.Model(inp, head_softmax(x))

    # ---- XNOR-Net (literature/xnornet.py)
    @lq.utils.set_precision(1)
    def xnor_weight_scale(x):  # same math as larq_zoo's xnor_weight_scale (per-output-filter mean |W|)
        x = tf.clip_by_value(x, -1, 1)
        alpha = tf.reduce_mean(tf.abs(x), axis=[0, 1, 2], keepdims=True)
        return alpha * lq.quantizers.ste_sign(x)

    def xnornet():
        inp = L.Input((224, 224, 3))
        kw = dict(kernel_quantizer=xnor_weight_scale, input_quantizer="ste_sign", kernel_constraint="weight_clip",
                  use_bias=False)
        bn = lambda e: L.BatchNormalization(momentum=0.9, scale=False, epsilon=e)  # noqa: E731
        x = L.Conv2D(96, (11, 11), strides=(4, 4), padding="same", use_bias=False)(inp)
        x = bn(1e-5)(x)
        x = L.Activation("relu")(x)
        x = L.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = bn(1e-4)(x)
        x = lq.layers.QuantConv2D(256, (5, 5), padding="same", **kw)(x)
        x = L.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = bn(1e-4)(x)
        x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kw)(x)
        x = bn(1e-4)(x)
        x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kw)(x)
        x = bn(1e-4)(x)
        x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kw)(x)
        x = L.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = bn(1e-4)(x)
        x = lq.layers.QuantConv2D(4096, (6, 6), padding="valid", **kw)(x)
        x = bn(1e-4)(x)
        x = lq.layers.QuantConv2D(4096, (1, 1), strides=(1, 1), padding="valid", **kw)(x)
        x = bn(1e-3)(x)
        x = L.Activation("relu")(x)
        x = L.Flatten()(x)
        x = L.Dense(1000, use_bias=False)(x)
        return tf.keras.Model(inp, head_softmax(x))

    # ---- Binary AlexNet v0.3.0 (literature/binary_alex_net.py)
    def binary_alexnet():
        inp = L.Input((224, 224, 3))

        def conv_block(x, f, k, strides=1, pool=False, first=False):
            x = lq.layers.QuantConv2D(f, kernel_size=k, strides=strides, padding="same",
                                      input_quantizer=None if first else "ste_sign", kernel_quantizer="ste_sign",
                                      kernel_constraint="weight_clip", use_bias=False)(x)
            if pool:
                x = L.MaxPool2D(pool_size=3, strides=2)(x)
            return L.BatchNormalization(scale=False, momentum=0.9)(x)

        def dense_block(x, units):
            x = lq.layers.QuantConv2D(units, kernel_size=1, input_quantizer="ste_sign", kernel_quantizer="ste_sign",
                                      kernel_constraint="weight_clip", use_bias=False)(x)
            return L.BatchNormalization(scale=False, momentum=0.9)(x)

        x = conv_block(inp, 64, 11, strides=4, pool=True, first=True)
        x = conv_block(x, 192, 5, pool=True)
        x = conv_block(x, 384, 3)
        x = conv_block(x, 384, 3)
        x = conv_block(x, 256, 3, pool=True)
        x = L.Reshape((1, 1, x.shape[-1] * x.shape[-2] * x.shape[-3]))(x)
        x = dense_block(x, 4096)
        x = dense_block(x, 4096)
        x = dense_block(x, 1000)
        x = L.Flatten()(x)
        return tf.keras.Model(inp, head_softmax(x))

    # ---- Binary DenseNet 28/37/37-dilated/45 (literature/densenet.py)
    def densenet(reduction, dilation, layers):
        q = lambda: lq.quantizers.SteSign(clip_value=1.3)  # noqa: E731
        inp = L.Input((224, 224, 3))
        x = L.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(inp)
        x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = L.Activation("relu")(x)
        x = L.MaxPool2D(3, strides=2, padding="same")(x)
        for b, n in enumerate(layers):
            for _ in range(n):
                y = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
                y = lq.layers.QuantConv2D(filters=64, kernel_size=3, dilation_rate=dilation[b], input_quantizer=q(),
                                          kernel_quantizer=q(), kernel_constraint=lq.constraints.WeightClip(1.3),
                                          padding="same", use_bias=False)(y)
                x = L.concatenate([x, y])
            if b < len(layers) - 1:
                x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
                if dilation[b + 1] == 1:
                    x = L.MaxPooling2D(2, strides=2)(x)
                x = L.Activation("relu")(x)
                x = L.Conv2D(round(x.shape.as_list()[-1] // reduction[b] / 32) * 32, kernel_size=1, use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = L.Activation("relu")(x)
        x = global_pool(x)
        x = L.Dense(1000)(x)
        return tf.keras.Model(inp, head_softmax(x))

    return {
        "quicknet_small": lambda: quicknet((4, 4, 4, 4), (32, 64, 256, 512)),
        "quicknet": lambda: quicknet((4, 4, 4, 4), (64, 128, 256, 512)),
        "quicknet_large": lambda: quicknet((6, 8, 12, 6), (64, 128, 256, 512)),
        "birealnet": birealnet,
        "resnet_e18": resnet_e18,
        "xnornet": xnornet,
        "binary_alexnet": binary_alexnet,
        "densenet28": lambda: densenet((2.7, 2.7, 2.2), (1, 1, 1, 1), (6, 6, 6, 5)),
        "densenet37": lambda: densenet((3.3, 3.3, 4), (1, 1, 1, 1), (6, 8, 12, 6)),
        "densenet37_dilated": lambda: densenet((3.3, 3.3, 4), (1, 1, 2, 4), (6, 8, 12, 6)),
        "densenet45": lambda: densenet((2.7, 3.3, 4), (1, 1, 1, 1), (6, 12, 14, 8)),
    }


def load_h5_weights(model, path: str) -> dict:
    """Assign the weights of a Keras weights-only .h5 (read with h5py) layer by layer, in stored order."""
    import h5py

    with h5py.File(path, "r") as f:
        dec = lambda n: n.decode() if isinstance(n, bytes) else n  # noqa: E731
        names = [dec(n) for n in f.attrs["layer_names"]]
        stored = []
        for n in names:
            wn = [dec(w) for w in f[n].attrs.get("weight_names", [])]
            if wn:
                stored.append((n, [np.asarray(f[n][w]) for w in wn]))
        kv = f.attrs.get("keras_version", b"")
    mine = [l for l in model.layers if l.weights]
    # Keras auto-names carry a session counter (QuickNetLarge's file was saved after other models had been built in
    # the same session: quant_conv2d_63 ...), so the layer type part of every name must match in order; the
    # full names usually match too (recorded).
    base = lambda n: re.sub(r"_\d+$", "", n)  # noqa: E731
    identical = [n for n, _ in stored] == [l.name for l in mine]
    if len(stored) != len(mine) or [base(n) for n, _ in stored] != [base(l.name) for l in mine]:
        diff = [(a, l.name) for (a, _), l in zip(stored, mine) if base(a) != base(l.name)][:5]
        raise RuntimeError(f"weighted-layer types differ from the h5 ({len(stored)} vs {len(mine)}): {diff}")
    n_tensors = 0
    for (n, ws), layer in zip(stored, mine):
        cur = layer.get_weights()
        if [w.shape for w in ws] != [w.shape for w in cur]:
            raise RuntimeError(f"{n}: shapes {[w.shape for w in ws]} != {[w.shape for w in cur]}")
        layer.set_weights(ws)
        n_tensors += len(ws)
    return {"n_weighted_layers": len(mine), "n_tensors": n_tensors, "names_identical_in_order": identical,
            "layer_types_identical_in_order": True,
            "keras_version_in_file": kv.decode() if isinstance(kv, bytes) else str(kv)}


def freeze_kernel_quantizers(model) -> list:
    """Replace every quantized kernel by its quantized value (the layer's own kernel quantizer, applied once) and drop
    the quantizer, so the converters see a constant filter. Returns per layer the distinct |values| per output channel."""
    import tensorflow as tf
    from larq import context

    out = []
    for layer in model.layers:
        k = getattr(layer, "kernel", None)
        q = getattr(k, "quantizer", None)
        if k is None or q is None:
            continue
        with context.quantized_scope(True):
            qk = q(tf.convert_to_tensor(k.latent_variable)).numpy()
        k.latent_variable.assign(qk)
        k.quantizer = None
        layer.kernel_quantizer = None
        a = np.abs(qk.reshape(-1, qk.shape[-1]))
        out.append({"layer": layer.name, "shape": list(qk.shape),
                    "binary_per_out_channel": bool(np.all(a.max(0) == a.min(0)))})
    return out


def compare(ref: np.ndarray, out: np.ndarray, labels: np.ndarray | None = None) -> dict:
    ref = ref.reshape(len(ref), -1).astype(np.float64)
    out = out.reshape(len(out), -1).astype(np.float64)
    d = np.abs(ref - out)
    top1 = ref.argmax(1) == out.argmax(1)
    t5 = [len(set(np.argsort(-ref[i])[:5]) & set(np.argsort(-out[i])[:5])) / 5 for i in range(len(ref))]
    r = {"n": int(len(ref)), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
         "top1_agreement": float(top1.mean()), "n_top1_disagreements": int((~top1).sum()), "top5_overlap": float(np.mean(t5))}
    if labels is not None:
        r["top1_acc_ref"] = float(np.mean(ref.argmax(1) == labels))
        r["top1_acc_out"] = float(np.mean(out.argmax(1) == labels))
    return r


def tflite_ops(buf: bytes) -> dict:
    """Operator histogram of a .tflite flatbuffer (builtin names; custom ops by their custom code)."""
    import tensorflow as tf

    try:
        from tensorflow.lite.python import schema_py_generated as sch
    except ImportError:  # pragma: no cover
        return {}
    m = sch.ModelT.InitFromObj(sch.Model.GetRootAsModel(buf, 0))
    names = {v: k for k, v in vars(sch.BuiltinOperator).items() if not k.startswith("_")}
    hist: dict[str, int] = {}
    for op in m.subgraphs[0].operators:
        oc = m.operatorCodes[op.opcodeIndex]
        code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
        n = oc.customCode.decode() if code == sch.BuiltinOperator.CUSTOM else names.get(code, str(code))
        hist[n] = hist.get(n, 0) + 1
    del tf
    return dict(sorted(hist.items()))


def tflite_tensor_types(buf: bytes) -> dict:
    """Per-operator input dtype histogram (to see which ops run in float inside the int8 model)."""
    from tensorflow.lite.python import schema_py_generated as sch

    m = sch.ModelT.InitFromObj(sch.Model.GetRootAsModel(buf, 0))
    names = {v: k for k, v in vars(sch.BuiltinOperator).items() if not k.startswith("_")}
    tnames = {v: k for k, v in vars(sch.TensorType).items() if not k.startswith("_")}
    sg = m.subgraphs[0]
    hist: dict[str, int] = {}
    for op in sg.operators:
        oc = m.operatorCodes[op.opcodeIndex]
        code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
        n = oc.customCode.decode() if code == sch.BuiltinOperator.CUSTOM else names.get(code, str(code))
        t = tnames.get(sg.tensors[op.inputs[0]].type, "?") if len(op.inputs) and op.inputs[0] >= 0 else "?"
        key = f"{n}:{t}"
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items()))


def run_tflite(buf: bytes, x: np.ndarray, lce_interp: bool) -> np.ndarray:
    """Host interpreter, 1 thread, sample by sample (batch 1, as on the phones)."""
    if lce_interp:
        import larq_compute_engine as lce

        it = lce.testing.Interpreter(buf, num_threads=1)
        return np.concatenate([np.asarray(it.predict(x[i:i + 1])).reshape(1, -1) for i in range(len(x))])
    import tensorflow as tf

    it = tf.lite.Interpreter(model_content=buf, num_threads=1)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    outs = []
    for i in range(len(x)):
        xi = x[i:i + 1]
        if ind["dtype"] == np.int8:
            s, z = ind["quantization"]
            xi = np.clip(np.round(xi / s + z), -128, 127).astype(np.int8)
        it.set_tensor(ind["index"], xi)
        it.invoke()
        o = it.get_tensor(outd["index"])
        if outd["dtype"] == np.int8:
            s, z = outd["quantization"]
            o = (o.astype(np.float32) - z) * s
        outs.append(o.reshape(1, -1))
    return np.concatenate(outs)


def build_one(model_id: str, out_root: str, x: np.ndarray, labels: np.ndarray, calib: np.ndarray, do_int8: bool,
              prequantize: str = "auto") -> dict:
    import tensorflow as tf
    import larq as lq
    import larq_compute_engine as lce

    builder, h5name, sha = MODELS[model_id]
    out = os.path.join(out_root, model_id)
    os.makedirs(out, exist_ok=True)
    h5 = os.path.join(MAIN, ".cache", "checkpoints", model_id, h5name)
    rep = {"model_id": model_id, "h5": {"file": os.path.relpath(h5, MAIN), "sha256": sha256_file(h5), "expected_sha256": sha},
           "env": {"tensorflow": tf.__version__, "tf_keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "",
                   "larq": lq.__version__, "larq_compute_engine": getattr(lce, "__version__", "0.16.0"),
                   "python": sys.version.split()[0]}}
    if rep["h5"]["sha256"] != sha:
        raise RuntimeError(f"{model_id}: h5 hash mismatch")
    tf.keras.backend.clear_session()
    model = _builders()[builder]()
    rep["h5"].update(load_h5_weights(model, h5))
    rep["keras_params"] = int(model.count_params())

    # 1) Keras model vs Bitino's reference outputs (produced by the original larq_zoo model, TF 2.15.1 / larq 0.13.3)
    art = os.path.join(MAIN, "artifacts", model_id)
    ref_probs = np.fromfile(os.path.join(art, "reference_probs.f32"), np.float32).reshape(128, -1)[: len(x)]
    ref_logits = np.fromfile(os.path.join(art, "reference_logits.f32"), np.float32).reshape(128, -1)[: len(x)]
    lm = tf.keras.Model(model.inputs, [model.layers[-1].input, model.output])
    lg, pr = [], []
    for i in range(0, len(x), 16):
        a, b = lm(x[i:i + 16], training=False)
        lg.append(a.numpy())
        pr.append(b.numpy())
    lg, pr = np.concatenate(lg), np.concatenate(pr)
    rep["keras_vs_reference"] = {"logits": compare(ref_logits, lg, labels), "probs": compare(ref_probs, pr, labels)}
    if rep["keras_vs_reference"]["logits"]["top1_agreement"] < 0.98:
        raise RuntimeError(f"{model_id}: rebuilt Keras model does not reproduce the reference: {rep['keras_vs_reference']}")

    # 2) conversions
    outputs = {"keras": pr}
    files = {}
    t0 = time.time()
    buf = lce.convert_keras_model(model)
    n_bin = sum(1 for l in model.layers if type(l).__name__ == "QuantConv2D" and l.input_quantizer is not None
                and l.kernel_quantizer is not None)
    n_bconv = tflite_ops(buf).get("LceBconv2d", 0)
    rep["lce_default_conversion"] = {"bytes": len(buf), "ops": tflite_ops(buf), "binary_quantconv2d_layers": n_bin}
    lce_s = time.time() - t0
    t0 = time.time()
    fp32 = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    fp32_s = time.time() - t0
    # Every binarized input costs two SIGN ops (larq.math.sign = sign(sign(x) + 0.1)); more SIGN ops, or a TRANSPOSE,
    # mean a weight quantizer was left in the graph and runs at every inference.
    n_inq = sum(1 for l in model.layers if getattr(l, "input_quantizer", None) is not None)
    fops = tflite_ops(fp32)
    rep["tflite_fp32_default_conversion"] = {"bytes": len(fp32), "ops": fops, "layers_with_input_quantizer": n_inq}
    lce_left = n_bconv < n_bin
    tfl_left = fops.get("SIGN", 0) > 2 * n_inq or "TRANSPOSE" in fops
    if prequantize == "always" or (prequantize == "auto" and (lce_left or tfl_left)):
        # The TFLite/LCE converters do not constant-fold the kernel quantizer of very large weight tensors (XNORNet's
        # 6x6x256x4096 conv, BinaryAlexNet's 9216x4096 one: clip/abs/mean/sign/transpose stay in the graph and run at
        # every inference; LCE then cannot see a binary filter). Apply each kernel quantizer once (same function, same
        # weights) and store the quantized kernel as the layer's constant with kernel_quantizer=None: identical math,
        # what Bitino's converter does offline too. Used for all three .tflite variants of this model (recorded).
        frozen = freeze_kernel_quantizers(model)
        why = []
        if lce_left:
            why.append(f"default LCE conversion has {n_bconv} LceBconv2d for {n_bin} binary QuantConv2D layers")
        if tfl_left:
            why.append(f"default TFLite fp32 conversion has {fops.get('SIGN', 0)} SIGN for {n_inq} binarized inputs"
                       + (" and a TRANSPOSE" if "TRANSPOSE" in fops else ""))
        rep["weights_prequantized"] = {"reason": "; ".join(why) + " (weight quantizer not constant-folded)",
                                       "layers": frozen}
        chk = np.concatenate([model(x[i:i + 16], training=False).numpy() for i in range(0, len(x), 16)])
        rep["weights_prequantized"]["keras_vs_reference_probs"] = compare(ref_probs, chk, labels)
        t0 = time.time()
        buf = lce.convert_keras_model(model)
        lce_s = time.time() - t0
        t0 = time.time()
        fp32 = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        fp32_s = time.time() - t0
    files["lce"] = ("lce.tflite", buf, lce_s)
    files["tflite_fp32"] = ("tflite_fp32.tflite", fp32, fp32_s)
    if do_int8:
        t0 = time.time()
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([calib[i:i + 1]] for i in range(len(calib)))
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        try:  # full-integer only first; fall back to int8 + float builtins if some op has no int8 kernel
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            files["tflite_int8"] = ("tflite_int8.tflite", conv.convert(), time.time() - t0)
            rep["int8_mode"] = "supported_ops=TFLITE_BUILTINS_INT8"
        except Exception as e:  # noqa: BLE001
            rep["int8_int8only_error"] = str(e).strip().splitlines()[-1][:400] if str(e).strip() else repr(e)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
            files["tflite_int8"] = ("tflite_int8.tflite", conv.convert(), time.time() - t0)
            rep["int8_mode"] = "supported_ops=TFLITE_BUILTINS_INT8+TFLITE_BUILTINS (float fallback)"
        rep["int8_calibration"] = {"set": ".cache/data/dev1000_imagenet100train_larq (ImageNet-100 TRAIN images, "
                                          "larq_zoo preprocessing, disjoint from eval128)", "n": int(len(calib))}
    rep["files"] = {}
    for k, (fn, b, dt) in files.items():
        p = os.path.join(out, fn)
        with open(p, "wb") as f:
            f.write(b)
        rep["files"][k] = {"file": fn, "bytes": len(b), "sha256_converter": hashlib.sha256(b).hexdigest(),
                           "convert_s": round(dt, 1), "ops": tflite_ops(b)}
        # LCE 0.16's benchmark tool cannot address "serving_default_input_1:0": rename the input in place (same length)
        fixed = h2h_tflite_names.fix(p)
        rep["files"][k].update(sha256=fixed["sha256"], input_renamed=fixed["renamed"])
        if k == "tflite_int8":
            rep["files"][k]["op_types"] = tflite_tensor_types(b)
            rep["int8_float_ops"] = {kk: v for kk, v in rep["files"][k]["op_types"].items()
                                     if kk.endswith(":FLOAT32") and not kk.startswith("QUANTIZE")}
    bf = os.path.join(art, "model.bitino")
    rep["files"]["bitino"] = {"file": os.path.relpath(bf, MAIN), "bytes": os.path.getsize(bf), "sha256": sha256_file(bf)}

    # 3) host agreement on the 128 eval samples (host x86 interpreters, batch 1, 1 thread)
    for k, (fn, b, _) in list(files.items()):
        try:
            outputs[k] = run_tflite(b, x, lce_interp=(k == "lce"))
        except Exception as e:  # noqa: BLE001
            if k != "tflite_int8":
                raise
            # e.g. BinaryAlexNet: the int8 converter emits a CONV_2D whose per-channel weight zero point is -1, which
            # the TFLite runtime rejects at prepare time. Keep the evidence, drop the file (nothing to benchmark).
            rep["tflite_int8_unrunnable"] = {"error": str(e).strip().splitlines()[0][:400], "bytes": len(b),
                                             "sha256": rep["files"]["tflite_int8"]["sha256"]}
            os.rename(os.path.join(out, fn), os.path.join(out, fn + ".unrunnable"))
            del files[k]
    rep["host_agreement_vs_reference_probs"] = {k: compare(ref_probs, v, labels) for k, v in outputs.items()}
    rep["host_agreement_vs_keras_probs"] = {k: compare(outputs["keras"], v, labels) for k, v in outputs.items() if k != "keras"}
    np.savez_compressed(os.path.join(out, "host_outputs.npz"), **{k: v.astype(np.float32) for k, v in outputs.items()})

    # 4) timing input: eval sample 0 (float32 NHWC), and its int8 quantization for the int8 model
    x[0:1].astype(np.float32).tofile(os.path.join(out, "input0.f32"))
    if "tflite_int8" in files:
        import tensorflow as tf2  # noqa: F401

        it = tf.lite.Interpreter(model_content=files["tflite_int8"][1])
        ind, outd = it.get_input_details()[0], it.get_output_details()[0]
        s, z = ind["quantization"]
        np.clip(np.round(x[0:1] / s + z), -128, 127).astype(np.int8).tofile(os.path.join(out, "input0_int8.bin"))
        rep["int8_io"] = {"input_scale": float(s), "input_zero_point": int(z),
                          "output_scale": float(outd["quantization"][0]), "output_zero_point": int(outd["quantization"][1])}
    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--models", default=",".join(MODELS))
    b.add_argument("--calib-n", type=int, default=128)
    b.add_argument("--no-int8", action="store_true")
    b.add_argument("--prequantize-weights", choices=["auto", "always", "never"], default="auto",
                   help="auto: only when the default LCE conversion leaves binary convs unconverted")
    b.add_argument("--max-samples", type=int, default=128)
    a = ap.parse_args()

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    if not tf.keras.__name__.startswith("tf_keras") and "keras.src" not in tf.keras.__name__ and \
            os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        print("set TF_USE_LEGACY_KERAS=1 (larq needs tf_keras)", file=sys.stderr)
        return 2
    ev = os.path.join(MAIN, ".cache", "data", "eval128")
    em = json.load(open(os.path.join(ev, "manifest.json")))
    pre = em["preprocessing"]["larq_zoo"]
    x = np.fromfile(os.path.join(ev, pre["file"]), np.float32).reshape(pre["shape"])[: a.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])
    dv = os.path.join(MAIN, ".cache", "data", "dev1000_imagenet100train_larq")
    dm = json.load(open(os.path.join(dv, "manifest.json")))
    dp = dm["preprocessing"]["larq_zoo"] if "preprocessing" in dm else {"file": "larq_zoo.f32", "shape": [1000, 224, 224, 3]}
    calib = np.array(np.memmap(os.path.join(dv, dp["file"]), np.float32, "r", shape=tuple(dp["shape"]))[: a.calib_n])
    os.makedirs(a.out, exist_ok=True)
    status = 0
    for mid in a.models.split(","):
        t0 = time.time()
        try:
            r = build_one(mid, a.out, x, labels, calib, not a.no_int8, a.prequantize_weights)
            ka = r["keras_vs_reference"]["logits"]
            ha = r["host_agreement_vs_reference_probs"]
            print(f"{mid}: keras vs ref top1 {ka['top1_agreement']:.4f} max|dlogit| {ka['max_abs']:.2e}; host top1 agreement "
                  + " ".join(f"{k}={v['top1_agreement']:.3f}" for k, v in ha.items())
                  + f"; sizes " + " ".join(f"{k}={v['bytes'] / 1e6:.2f}MB" for k, v in r["files"].items())
                  + f" ({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"{mid}: FAILED: {e}", flush=True)
            with open(os.path.join(a.out, f"{mid}.FAILED.txt"), "w") as f:
                f.write(traceback.format_exc())
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
