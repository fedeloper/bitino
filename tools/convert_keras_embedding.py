#!/usr/bin/env python3
"""Convert Keras/Larq *embedding* models (slice S6) to .bitino and produce Level-3 references.

Families (run inside the Larq reference env, `python -I`):

  larqface   insightface_larq_keras full-model h5 (model_config + weights, Keras 2.8 / larq 0.12.2): the architecture
             is rebuilt from the JSON config with larq's registered layers. The config is checked BEFORE
             deserialization: only a whitelist of layer classes is accepted, `Lambda` (marshalled Python code) is
             rejected, and TFOpLambda may only name `__operators__.add`. Output: the 128-D embedding.
  brillsson  HarlinLee/brillsson weights-only h5: the official BinaryModel.create_model structure without the
             Spectrogram front-end (log-mel features are computed on the host, tools/eval_sets/speech_commands_logmel.py)
             is rebuilt from the installed larq_zoo (`--zoo-class`, input (T, 64, 1)), the weights are loaded by
             layer order (weight names/shapes checked), then copied into a flat model
             [larq_zoo backbone -> GlobalMaxPool2D -> QuantDense(hidden)] whose outputs are
             (projection embedding [hidden], pooled encoder features [C]); the copy is checked bit-exact against the
             official nested model.

Produces in --out-dir (file names follow tools/convert_larq_zoo.py; "logits" = the embedding output record):
  model.bitino / model_unfused.bitino, reference_logits.f32 (source fp32, batch 1), reference_logits_f64.f64,
  numpy_logits_{unfused,fused}.f32, ir_f64_logits_{unfused,fused}.f64, conversion_report.json (Level 3 incl. the
  float64-constants check and a first-binarization-flip analysis of every deviating sample).

  .cache/venv-larq/bin/python -I tools/convert_keras_embedding.py --family larqface --model-id larqface-quicknet-msceleb \
      --weights <h5> --sha256 <registry> --eval-dir .cache/data/eval128_lfw --preprocess insightface --out-dir artifacts/<id>
  .cache/venv-larq/bin/python -I tools/convert_keras_embedding.py --family brillsson --model-id brillsson-quicknet \
      --zoo-class sota.QuickNet --weights <h5> --sha256 <registry> --eval-dir .cache/data/eval128_speechcommands \
      --preprocess brillsson_logmel --out-dir artifacts/<id>
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import platform
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import keras_larq  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

LARQFACE_CLASSES = {"InputLayer", "Conv2D", "BatchNormalization", "Activation", "MaxPooling2D", "QuantConv2D",
                    "Concatenate", "Flatten", "Dense", "PReLU", "Reshape", "Multiply", "QuantDepthwiseConv2D", "Add",
                    "QuantSeparableConv2D", "TFOpLambda", "DepthwiseConv2D"}
TFOP_ALLOWED = {"__operators__.add"}
QUANT_LAYERS = ("QuantConv2D", "QuantDense", "QuantDepthwiseConv2D", "QuantSeparableConv2D")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ model building

def _larqface_config(weights: str) -> dict:
    import h5py
    with h5py.File(weights, "r") as f:
        cfg = json.loads(f.attrs["model_config"])
        keras_version = f.attrs.get("keras_version", b"")
    if cfg["class_name"] not in ("Functional", "Model"):
        raise RuntimeError(f"model_config class {cfg['class_name']}")
    for l in cfg["config"]["layers"]:
        cls = l["class_name"]
        if cls not in LARQFACE_CLASSES:
            raise RuntimeError(f"layer class {cls} ({l['config'].get('name')}) not in the whitelist (Lambda code is never loaded)")
        if cls == "TFOpLambda" and l["config"].get("function") not in TFOP_ALLOWED:
            raise RuntimeError(f"TFOpLambda {l['config'].get('function')} not allowed")
    # Initializers never run (every weight is loaded from the h5 afterwards); string references to functions of the
    # training code (QuickNet's larq_zoo 'blurpool_initializer') cannot be deserialized, so all string initializers
    # become 'zeros'.
    replaced = []
    for l in cfg["config"]["layers"]:
        for k, v in list(l["config"].items()):
            if k.endswith("_initializer") and isinstance(v, str) and v != "zeros":
                replaced.append(f"{l['config']['name']}.{k}={v}")
                l["config"][k] = "zeros"
    cfg["_keras_version"] = keras_version.decode() if isinstance(keras_version, bytes) else str(keras_version)
    cfg["_replaced_initializers"] = replaced
    return cfg


def _with_dtype(cfg: dict, dtype: str) -> dict:
    def walk(o):
        if isinstance(o, dict):
            return {k: (dtype if k == "dtype" and v == "float32" else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(copy.deepcopy(cfg))


def build_larqface(weights: str):
    import larq  # noqa: F401  (registers the Quant* layers and quantizers as Keras custom objects)
    import tensorflow as tf
    import h5py
    cfg = _larqface_config(weights)
    kv = cfg.pop("_keras_version")
    replaced = cfg.pop("_replaced_initializers")
    model = tf.keras.models.model_from_json(json.dumps(cfg))
    model.load_weights(weights)
    with h5py.File(weights, "r") as f:
        g = f["model_weights"]
        names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]
        stored = [n for n in names if len(g[n].attrs.get("weight_names", [])) > 0]
    mine = [l.name for l in model.layers if l.weights]
    if stored != mine:
        raise RuntimeError("h5 weighted-layer order differs from the model")

    def build64(pure: bool = False):
        tf.keras.backend.set_floatx("float64")
        m = tf.keras.models.model_from_json(json.dumps(_with_dtype(cfg, "float64")))
        m.set_weights([w.astype(np.float64) for w in model.get_weights()])
        if not pure:
            fp32_dorefa_inputs(m)
        return m
    info = {"source_format": "Keras full-model h5 (model_config deserialized with a class whitelist, no Lambda)",
            "keras_version_in_file": kv, "n_weighted_layers": len(mine), "h5_names_identical_in_order": True,
            "string_initializers_replaced_by_zeros (weights loaded afterwards)": replaced,
            "layer_classes": sorted({l["class_name"] for l in cfg["config"]["layers"]})}
    return model, build64, info


def fp32_dorefa_inputs(model64) -> int:
    """Float64 model whose DoReFa(k=1, 'weights') INPUT quantizers run the original larq quantizer in float32 on the
    float32-cast input (the float32 network's binarization rule; in pure float64 the tanh/round band shrinks from
    ~2^-24 to ~2^-53 and near-dead channels binarize differently, i.e. a different function than the trained one)."""
    import larq as lq
    import tensorflow as tf
    n = 0
    for layer in model64.layers:
        if keras_larq._is_dorefa_w1(getattr(layer, "input_quantizer", None)):
            q32 = lq.quantizers.DoReFa(k_bit=1, mode="weights", dtype="float32")
            layer.input_quantizer = (lambda q: (lambda x: tf.cast(q(tf.cast(x, tf.float32)), x.dtype)))(q32)
            n += 1
    return n


def _canon(name: str) -> str:
    """'quant_conv2d_12/kernel:0' -> 'quant_conv2d/kernel' (Keras name counters differ between sessions)."""
    layer, _, var = name.partition("/")
    return re.sub(r"_\d+$", "", layer) + "/" + var.split(":")[0]


def build_brillsson(weights: str, zoo_class: str, frames: int, hidden: int):
    import larq as lq
    import larq_zoo
    import tensorflow as tf
    import h5py
    module, cls = zoo_class.split(".")
    factory = getattr(getattr(larq_zoo, module), cls)

    def official():
        # BinaryModel.create_model(input_shape, None, model_type, hidden_size) without the Spectrogram layer.
        # model.py passes input_tensor=encoder_inputs and then calls the model on it (a nested model); larq_zoo
        # 2.4 + TF 2.15 rejects a KerasTensor for input_tensor (zookeeper type check), so the backbone is built with
        # input_shape and called on the encoder input: the same nested structure and weight order.
        enc_in = tf.keras.layers.Input((frames, 64, 1))
        backbone = factory(input_shape=(frames, 64, 1), weights=None, include_top=False)
        enc_out = tf.keras.layers.GlobalMaxPool2D()(backbone(enc_in))
        encoder = tf.keras.Model(inputs=enc_in, outputs=enc_out)
        inp = tf.keras.layers.Input((frames, 64, 1))
        x = encoder(inp)
        x = lq.layers.QuantDense(hidden, activation=None)(x)
        return tf.keras.Model(inp, x), encoder

    def flat():
        backbone = factory(input_shape=(frames, 64, 1), weights=None, include_top=False)
        pooled = tf.keras.layers.GlobalMaxPool2D()(backbone.output)
        emb = lq.layers.QuantDense(hidden, activation=None)(pooled)
        return tf.keras.Model(backbone.inputs, [emb, pooled])

    off, encoder = official()
    with h5py.File(weights, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["layer_names"]]
        stored = {n: [w.decode() if isinstance(w, bytes) else w for w in f[n].attrs.get("weight_names", [])] for n in names}
        stored_shapes = {n: [f[n][w].shape for w in stored[n]] for n in names}
    weighted = [n for n in names if stored[n]]
    mine = [l for l in off.layers if l.weights]
    if len(weighted) != len(mine):
        raise RuntimeError(f"weighted layers: file {weighted} vs model {[l.name for l in mine]}")
    for n, l in zip(weighted, mine):
        lw = l.trainable_weights + l.non_trainable_weights   # Keras legacy h5 order
        a = [(_canon(w), tuple(s)) for w, s in zip(stored[n], stored_shapes[n])]
        b = [(_canon(v.name), tuple(v.shape)) for v in lw]
        if a != b:
            bad = next(i for i, (p, q) in enumerate(zip(a, b)) if p != q) if len(a) == len(b) else -1
            raise RuntimeError(f"{n}: weight names/shapes differ from the rebuilt model (first at {bad}: "
                               f"{a[bad] if bad >= 0 else len(a)} vs {b[bad] if bad >= 0 else len(b)})")
    off.load_weights(weights)
    model = flat()
    model.set_weights(off.get_weights())
    rng = np.random.default_rng(0)
    xr = (rng.standard_normal((2, frames, 64, 1)) * 3 - 6).astype(np.float32)
    e1, p1 = [t.numpy() for t in model(xr, training=False)]
    e0, p0 = off(xr, training=False).numpy(), encoder(xr, training=False).numpy()
    if not (np.array_equal(e0, e1) and np.array_equal(p0, p1)):
        raise RuntimeError("flat model differs from the official nested model")

    def build64(pure: bool = False):
        tf.keras.backend.set_floatx("float64")
        m = flat()
        m.set_weights([w.astype(np.float64) for w in model.get_weights()])
        return m
    info = {"source_format": "Keras weights-only h5 loaded into the rebuilt official structure (Spectrogram front-end on host)",
            "zoo_class": zoo_class, "larq_zoo": larq_zoo.__version__, "frames": frames, "hidden": hidden,
            "h5_layer_names": names, "weight_names_and_shapes_match_canonical": True,
            "flat_vs_official_nested_bit_exact": True, "outputs": ["projection embedding (QuantDense, distillation target)",
                                                                   "GlobalMaxPool2D encoder features (eval.py linear probe input)"]}
    return model, build64, info


# ------------------------------------------------------------------------------------------------ helpers

def run_keras(model, x, dtype, batch=1):
    outs = []
    for i in range(0, len(x), batch):
        o = model(x[i:i + batch].astype(dtype), training=False)
        o = o if isinstance(o, (list, tuple)) else [o]
        outs.append(np.concatenate([np.asarray(t).reshape(len(x[i:i + batch]), -1) for t in o], axis=1))
    return np.concatenate(outs)


def quant_input_stats(model, x, dtype) -> dict:
    """Exact zeros (and, for DoReFa(k=1, weights) inputs, values in the float rounding band) at the inputs of the
    binarizing Keras layers, batch 1."""
    import tensorflow as tf
    qls = [l for l in model.layers if type(l).__name__ in QUANT_LAYERS and getattr(l, "input_quantizer", None) is not None]
    ins = [l._inbound_nodes[0].keras_inputs for l in qls]
    ins = [t[0] if isinstance(t, (list, tuple)) else t for t in ins]
    dorefa = [keras_larq._is_dorefa_w1(l.input_quantizer) for l in qls]
    probe = tf.keras.Model(model.inputs, ins)
    eps = np.finfo(np.dtype(dtype)).eps / 2           # 2^-24 (fp32) / 2^-53 (fp64): ulp(0.5) / 2
    z = band = tot = mism = 0
    for i in range(len(x)):
        outs = probe(x[i:i + 1].astype(dtype), training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for li, (o, d) in enumerate(zip(outs, dorefa)):
            o = np.asarray(o)
            z += int(np.sum(o == 0))
            tot += int(o.size)
            if d and dtype == np.float32:
                # live float32 quantizer vs the IR rule x > 2^-24 * float32(libm tanh(max|x|)) (the IR's operand)
                ql = np.asarray(qls[li].input_quantizer(tf.constant(o)))
                m_ir = np.float32(executor._libm(executor._TANH, np.array([np.abs(o).max()], np.float64))[0])
                rule = o > np.float32(keras_larq.DOREFA_W1_TSCALE * m_ir)
                mism += int(np.sum((ql > 0) != rule))
                band += int(np.sum((o > 0) & ~rule))
            elif d:
                m = float(np.max(np.abs(np.tanh(o.astype(np.float64)))))
                band += int(np.sum((o > 0) & (np.tanh(o.astype(np.float64)) <= 2 * eps * m)))
    r = {"exact_zero": z, "total": tot, "n_quantized_layers": len(qls)}
    if any(dorefa):
        r["dorefa_w1_values_0_lt_x_le_T"] = band
        r["dorefa_w1_definition"] = ("values with 0 < x <= T that binarize to -1; T = 2^-24 * float32(libm tanh(max|x|)) "
                                     "(float32 model, = the IR operand) or ~2^-53 * max|tanh| (pure float64 model)")
        if dtype == np.float32:
            r["dorefa_w1_ir_rule_vs_live_quantizer_mismatches"] = mism
    return r


def bn_float64_constants(model) -> dict:
    out = {}
    for layer in model.layers:
        if type(layer).__name__ != "BatchNormalization":
            continue
        gamma = layer.gamma.numpy().astype(np.float64) if layer.scale else 1.0
        beta = layer.beta.numpy().astype(np.float64) if layer.center else 0.0
        mean = layer.moving_mean.numpy().astype(np.float64)
        inv = 1.0 / np.sqrt(layer.moving_variance.numpy().astype(np.float64) + layer.epsilon)
        out[f"{layer.name}/scale"] = gamma * inv
        out[f"{layer.name}/shift"] = beta - mean * gamma * inv
    return out


def first_flip(model64, model32, g, x_all, indices) -> list:
    """First binarization difference (float64 Keras vs float64 IR, unfused graph) for each sample."""
    import tensorflow as tf
    producers = {o: n for n in g.nodes for o in n.outputs}
    l64 = {a.name: b for a, b in zip(model32.layers, model64.layers)}
    bnodes = [n for n in g.nodes if n.op in ("BConv2D", "BDense")]
    feeders, dorefa = [], []
    for n in bnodes:
        p = producers.get(n.inputs[0])
        d = p is not None and p.op == "AddBroadcast" and p.name.endswith("/dorefa_operand")
        if d:
            p = producers.get(p.inputs[0])
        dorefa.append(d)
        feeders.append(l64[p.name].output if p is not None and p.name in l64 else model64.inputs[0])
    probe = tf.keras.Model(model64.inputs, feeders)
    import larq as lq
    q32 = lq.quantizers.DoReFa(k_bit=1, mode="weights", dtype="float32")
    res = []
    for idx in indices:
        x = x_all[idx:idx + 1].astype(np.float64)
        k_outs = probe(x, training=False)
        k_outs = [np.asarray(o) for o in (k_outs if isinstance(k_outs, (list, tuple)) else [k_outs])]
        env = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        rec = {"sample": int(idx), "first_flip_node": None}
        for n, kv, d in zip(bnodes, k_outs, dorefa):
            if d:
                # source: the float32 larq quantizer on float32(kv); IR: sign of the operand x - T
                k32 = kv.astype(np.float32)
                fk = np.asarray(q32(tf.constant(k32))) > 0
                tk = keras_larq.DOREFA_W1_TSCALE * float(np.max(np.abs(np.asarray(tf.math.tanh(tf.constant(k32))))))
                xi = env[producers[n.inputs[0]].inputs[0]].reshape(kv.shape)
                iv = env[n.inputs[0]].reshape(kv.shape)
                fi = iv > 0
                pre_k, pre_i, feed = np.abs(kv - tk), np.abs(iv), np.abs(kv - xi)
            else:
                iv = env[n.inputs[0]].reshape(kv.shape)
                thr = env[n.inputs[4]] if len(n.inputs) > 4 and n.inputs[4] else 0.0
                mode = n.attrs.get("input_binarize", "ge")
                fk = (kv > 0) if mode == "gt" else (kv >= 0)
                fi = (iv - thr > 0) if mode == "gt" else (iv - thr >= 0)
                pre_k, pre_i, feed = np.abs(kv), np.abs(iv - thr), np.abs(kv - iv)
            flips = fk != fi
            if flips.any():
                rec.update({"first_flip_node": n.name, "n_flips": int(flips.sum()), "n_bits": int(flips.size),
                            "dorefa_threshold_site": bool(d),
                            "max_abs_pre_at_flip_keras": float(np.max(pre_k[flips])),
                            "max_abs_pre_at_flip_ir": float(np.max(pre_i[flips])),
                            "max_abs_diff_feeding_tensor": float(np.max(feed)),
                            "typical_abs_pre": float(np.median(pre_k))})
                break
        res.append(rec)
    return res


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("larqface", "brillsson"))
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--zoo-class", help="brillsson: larq_zoo backbone, e.g. sota.QuickNet")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--source-repo", help="read-only source tree (provenance: COMMIT_SHA and file hashes)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--preprocess", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4, help="TensorFlow intra-op threads (shared host)")
    args = ap.parse_args()

    actual = sha256_file(args.weights)
    if actual != args.sha256:
        print(f"ERROR: checkpoint hash mismatch: expected {args.sha256}, got {actual}", file=sys.stderr)
        return 3
    os.makedirs(args.out_dir, exist_ok=True)
    import larq
    import larq_zoo
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][args.preprocess]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    if args.family == "larqface":
        model, build64, src_info = build_larqface(args.weights)
    else:
        model, build64, src_info = build_brillsson(args.weights, args.zoo_class, int(pre["shape"][1]), args.hidden)
    if args.source_repo:
        sha_file = os.path.join(args.source_repo, "COMMIT_SHA")
        src_info["source_repo_commit"] = open(sha_file).read().strip() if os.path.exists(sha_file) else "unknown"
        files = {"brillsson": ["model.py", "eval.py"], "larqface": ["evals.py", "backbones/BinaryFaceNet.py"]}[args.family]
        tree = [d for d in os.listdir(args.source_repo) if os.path.isdir(os.path.join(args.source_repo, d))]
        root = os.path.join(args.source_repo, tree[0]) if len(tree) == 1 else args.source_repo
        src_info["source_files_sha256"] = {f: sha256_file(os.path.join(root, f)) for f in files if os.path.exists(os.path.join(root, f))}

    ref = run_keras(model, x, np.float32).astype(np.float32)
    ref.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    zeros_src = {"source_fp32": quant_input_stats(model, x, np.float32)}
    model64 = build64()
    has_dorefa = any(keras_larq._is_dorefa_w1(getattr(l, "input_quantizer", None)) for l in model.layers)
    try:
        ref64 = run_keras(model64, x, np.float64)
        if has_dorefa:
            pure64 = build64(pure=True)
            ref64_pure = run_keras(pure64, x, np.float64)
            zeros_src["source_fp64_pure"] = quant_input_stats(pure64, x, np.float64)
        else:
            zeros_src["source_fp64"] = quant_input_stats(model64, x, np.float64)
    finally:
        tf.keras.backend.set_floatx("float32")
    ref64.tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    n_out = ref.shape[1]

    report = {"model_id": args.model_id, "family": args.family, "n_samples": int(len(x)), "eval_set": em["name"],
              "output_record": n_out, "source": src_info,
              "reference": {"framework": f"tensorflow {tf.__version__}, larq {larq.__version__}, larq_zoo {larq_zoo.__version__}",
                            "batch": 1},
              "reference_f32_vs_f64": compare_outputs(ref64, ref),
              "exact_zeros_at_binarization_inputs": zeros_src, "variants": {}}
    cos = lambda a, b: np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)  # noqa: E731
    report["reference_f32_vs_f64"]["cosine_per_sample_min_index"] = int(np.argmin(cos(ref.astype(np.float64), ref64)))
    if has_dorefa:
        report["reference"]["float64_reference"] = ("float64 Keras model with every DoReFa(k=1, weights) INPUT quantizer run in "
                                                    "float32 on the float32-cast input (the trained float32 binarization rule)")
        report["reference_f64_pure_vs_f64_primary"] = compare_outputs(ref64, ref64_pure)
        report["reference_f32_vs_f64_pure"] = compare_outputs(ref64_pure, ref)

    for variant, fold in (("unfused", False), ("fused", True)):
        graph, info = keras_larq.convert(model, args.model_id)
        applied = passes.optimize(graph, fold=fold)
        graph.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": variant,
            "source": {"framework": "keras+larq", "family": args.family, "checkpoint_file": os.path.basename(args.weights),
                       "checkpoint_sha256": actual, **{k: v for k, v in src_info.items() if k in ("zoo_class", "source_repo_commit")}},
            "conversion": {"tool": "tools/convert_keras_embedding.py", "adapter": "bitino.adapters.keras_larq",
                           "passes": applied, "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
                           "python": platform.python_version(), "tensorflow": tf.__version__, "larq": larq.__version__,
                           "larq_zoo": larq_zoo.__version__},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                      "preprocessing": pre["definition"]},
            "outputs": (["embedding (128-D, not normalized)"] if args.family == "larqface"
                        else [f"embedding (projection, {args.hidden})", "pooled encoder features"]),
            "op_versions": {"format": 1}, "layer_precision": info["layer_precision"],
        }
        path = os.path.join(args.out_dir, "model.bitino" if fold else "model_unfused.bitino")
        file_sha = bfformat.write(graph, path)
        g2 = bfformat.read(path)
        bin_inputs = [(n.inputs[0], n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else None)
                      for n in g2.nodes if n.op in ("BConv2D", "BDense")]
        outs, nz = [], 0
        for i in range(len(x)):
            res = executor.run(g2, {"input": x[i:i + 1]}, keep_intermediates=True)
            outs.append(np.concatenate([res[o].reshape(-1) for o in g2.outputs]))
            nz += sum(int(np.sum(res[a] == (res[t] if t else 0.0))) for a, t in bin_inputs)
        outs = np.asarray(outs, np.float32)
        assert outs.shape == ref.shape, (outs.shape, ref.shape)
        cmp = compare_outputs(ref, outs)
        cmp64 = compare_outputs(ref64, outs)
        sem = np.asarray([np.concatenate([r.reshape(-1) for r in
                                          (lambda rr: [rr[o] for o in g2.outputs])(executor.run(g2, {"input": x[i:i + 1]}, float_dtype=np.float64))])
                          for i in range(len(x))])
        cmp_sem = compare_outputs(ref64, sem)
        dev = np.max(np.abs(sem - ref64), axis=1)
        cmp_sem["n_samples_within_1e-6"] = int(np.sum(dev <= 1e-6))
        cmp_sem["deviating_samples_gt_1e-6"] = [int(i) for i in np.where(dev > 1e-6)[0]]
        cmp_sem["cosine_min"] = float(np.min(cos(sem, ref64)))
        sem.astype(np.float64).tofile(os.path.join(args.out_dir, f"ir_f64_logits_{variant}.f64"))
        if not fold:
            gx = bfformat.read(path)
            f64c = bn_float64_constants(model)
            for name, v in f64c.items():
                if name not in gx.constants:
                    continue
                d = gx.constants[name].data.astype(np.float64)
                assert np.allclose(d, v, rtol=2.0 ** -21, atol=1e-30 + 2.0 ** -21 * np.abs(v).max()), f"{name}: bfmodel differs"
                gx.constants[name].data = np.asarray(v, np.float64)
            semx = np.asarray([np.concatenate([executor.run(gx, {"input": x[i:i + 1]}, float_dtype=np.float64)[o].reshape(-1)
                                               for o in gx.outputs]) for i in range(len(x))])
            px = np.abs(semx - ref64).max(axis=1)
            rel = px / np.maximum(np.abs(ref64).max(axis=1), 1e-30)
            cmp_sem["float64_constants_check"] = {
                "description": "unfused IR with the folded BatchNorm scale/shift in float64 vs float64 source",
                "n_constants_replaced": sum(1 for k in f64c if k in gx.constants), "max_abs_error": float(px.max()),
                "max_rel_to_output_max": float(rel.max()), "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                "n_samples_rel_le_1e-12": int(np.sum(rel <= 1e-12))}
            dev_idx = cmp_sem["deviating_samples_gt_1e-6"]
            if dev_idx:
                tf.keras.backend.set_floatx("float64")
                try:
                    ff = first_flip(model64, model, bfformat.read(path), x, dev_idx)
                finally:
                    tf.keras.backend.set_floatx("float32")
                withf = [r for r in ff if r["first_flip_node"]]
                cmp_sem["first_flip_analysis"] = {
                    "n_analyzed": len(ff), "n_with_flip": len(withf),
                    "max_abs_pre_at_flip": max([max(r["max_abs_pre_at_flip_keras"], r["max_abs_pre_at_flip_ir"]) for r in withf], default=None),
                    "max_ratio_pre_to_typical": max([max(r["max_abs_pre_at_flip_keras"], r["max_abs_pre_at_flip_ir"]) / r["typical_abs_pre"]
                                                     for r in withf], default=None),
                    "all_flips_within_feeding_deviation": all(max(r["max_abs_pre_at_flip_keras"], r["max_abs_pre_at_flip_ir"])
                                                              <= r["max_abs_diff_feeding_tensor"] for r in withf),
                    "max_abs_dev_without_flip": float(max([dev[i] for i in dev_idx if i not in {r["sample"] for r in withf}], default=0.0)),
                    "records": ff}
                json.dump(ff, open(os.path.join(args.out_dir, "first_flip.json"), "w"), indent=1)
            print(f"[{variant}] float64 constants: max_abs {px.max():.3e}, rel<=1e-12 {int(np.sum(rel <= 1e-12))}/{len(x)}")
        report["variants"][variant] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "exact_zeros_at_binarization_inputs_numpy_f32": nz, "op_counts": g2.op_counts(),
                                       "passes": applied, "numpy_vs_reference_f32": cmp, "numpy_vs_reference_f64": cmp64,
                                       "semantic_check_ir_f64_vs_reference_f64": cmp_sem}
        outs.tofile(os.path.join(args.out_dir, f"numpy_logits_{variant}.f32"))
        print(f"[{variant}] {g2.op_counts()} | vs f32 ref: max_abs {cmp['max_abs_error']:.3e} cos_min {cmp['cosine_min']:.6f} | "
              f"vs f64 ref: max_abs {cmp64['max_abs_error']:.3e} cos_min {cmp64['cosine_min']:.6f} | SEMANTIC: max_abs "
              f"{cmp_sem['max_abs_error']:.3e} within1e-6 {cmp_sem['n_samples_within_1e-6']}/{len(x)} cos_min {cmp_sem['cosine_min']:.9f}")
    r = report["reference_f32_vs_f64"]
    print(f"source fp32 vs fp64: max_abs {r['max_abs_error']:.3e} cos_min {r['cosine_min']:.6f}; zeros {json.dumps(zeros_src)}")
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
