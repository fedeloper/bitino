#!/usr/bin/env python3
"""Convert a daBNN model-zoo Bi-Real ONNX (dabnn-bireal18, dabnn-bireal18-stem) to .bitino and validate it.

Run in the project-local env .cache/venv-qonnx (onnx + onnxruntime + numpy), with `python -I`:

  .cache/venv-qonnx/bin/python -I tools/convert_dabnn.py --model-id dabnn-bireal18 \\
      --onnx .cache/checkpoints/dabnn-bireal18/birealnet18.onnx --dab .cache/checkpoints/dabnn-bireal18/birealnet18.dab \\
      --eval-dir .cache/data/eval128 --out-dir artifacts/dabnn-bireal18 --jobs 4

Trusted reference. The ONNX files carry no Sign op: daBNN's onnx2bnn binarizes the *input* of every
all-+-1 Conv implicitly, and its runtime pads the binarized input with -1. Running the raw ONNX is
therefore not the BNN. The reference is onnxruntime running the ORIGINAL graph with only that implicit
step made explicit (graph surgery, `surgery()`): before each all-+-1 Conv, `Where(Less(x, 0), -1, +1)`
(`ge`: daBNN aarch64 sign-bit packing for every value except -0.0) or `Where(Greater(x, 0), +1, -1)`
(`gt`: daBNN portable packing), then `Pad(value = pad_value)` with the conv's own pads, and the conv
pads set to 0. The opset-6 stem file is first upgraded with onnx.version_converter to opset 9
(onnxruntime has no BatchNormalization-6 kernel); the upgrade is checked numerically (raw graph, no
surgery, numpy float32 evaluation of the original opset-6 graph vs onnxruntime on the upgraded one).

onnxruntime's CPU Conv is float32-only, so reference_logits_f64.f64 is an independent numpy float64
evaluation of the same surgically modified ONNX graph (`evaluate_onnx`, written from the ONNX operator
specs, independent of the Bitino adapter and executor); its float32 twin is compared with onnxruntime.

Produces in --out-dir:
  model.bitino / model_unfused.bitino     fused (BN folded into BConv2D post_mul/post_add and stem convs) / 1:1
  reference_logits.f32                      onnxruntime fp32, surgically binarized original ONNX (primary semantics)
  reference_logits_f64.f64                  numpy float64 evaluation of the same graph
  numpy_logits_fused.f32 / numpy_logits_unfused.f32   Bitino numpy executor (float32)
  semantics_study.json (inside conversion_report.json)  onnxruntime top-1 for pad -1/0 x ge/gt
  variant_pad0/                             same files for pad_value 0 (zero padding of the binarized input)
  conversion_report.json                    Level-3 strict float64 semantic check + first-flip analysis
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import onnx_dabnn  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

MODELS = {"dabnn-bireal18": "Bi-Real Net 18 (daBNN model zoo)", "dabnn-bireal18-stem": "Bi-Real Net 18 + stem module (daBNN model zoo)"}
PREPROCESS = "torchvision_imagenet"
SEM_TOL = 1e-6


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ reference: graph surgery
def upgrade(model):
    """opset < 9 -> 9 (onnx.version_converter); returns (model, note)."""
    import onnx
    from onnx import version_converter

    op = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    if op >= 9:
        return model, None
    m = version_converter.convert_version(model, 9)
    onnx.checker.check_model(m)
    return m, f"opset {op} -> 9 with onnx.version_converter (onnx {onnx.__version__})"


def surgery(model, binarize: str, pad_value: float, probe: bool = False):
    """Makes daBNN's implicit input binarization of every all-+-1 Conv explicit. Returns (model, probes)
    where probes maps each binary conv output name to the name of its (pre-binarization) input tensor."""
    import onnx
    from onnx import helper, numpy_helper

    m, _ = upgrade(copy.deepcopy(model))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    g = m.graph
    for nm, v in (("bf_zero", 0.0), ("bf_pos", 1.0), ("bf_neg", -1.0)):
        g.initializer.append(numpy_helper.from_array(np.array(v, np.float32), nm))
        if m.ir_version < 4:   # IR 3: every initializer must also be a graph input
            g.input.append(helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, []))
    nodes, probes = [], {}
    for n in g.node:
        if n.op_type == "Conv" and n.input[1] in init and onnx_dabnn.is_binary_weight(init[n.input[1]]):
            x, o = n.input[0], n.output[0]
            probes[o] = x
            if binarize == "ge":
                nodes.append(helper.make_node("Less", [x, "bf_zero"], [o + "_bf_neg"]))
                nodes.append(helper.make_node("Where", [o + "_bf_neg", "bf_neg", "bf_pos"], [o + "_bf_bin"]))
            else:
                nodes.append(helper.make_node("Greater", [x, "bf_zero"], [o + "_bf_pos"]))
                nodes.append(helper.make_node("Where", [o + "_bf_pos", "bf_pos", "bf_neg"], [o + "_bf_bin"]))
            pads = [0, 0, 0, 0]
            for a in n.attribute:
                if a.name == "pads":
                    pads = list(a.ints)
                    a.ints[:] = [0, 0, 0, 0]
            # Pad (opset 2..10 attribute form): [n_b, c_b, h_b, w_b, n_e, c_e, h_e, w_e]
            nodes.append(helper.make_node("Pad", [o + "_bf_bin"], [o + "_bf_pad"], mode="constant",
                                          pads=[0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]], value=float(pad_value)))
            n.input[0] = o + "_bf_pad"
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    if probe:
        for x in probes.values():
            g.output.append(helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, [None, None, None, None]))
    onnx.checker.check_model(m)
    return m, probes


def ort_run(model, x_nhwc: np.ndarray, threads: int = 4, probe_names=()):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    outs = [model.graph.output[0].name] + list(probe_names)
    logits, stats = [], {"exact_zeros": 0, "negative_zeros": 0, "n_values": 0}
    for i in range(len(x_nhwc)):
        r = sess.run(outs, {iname: np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2))})
        logits.append(np.asarray(r[0], np.float32).reshape(-1))
        for p in r[1:]:
            stats["exact_zeros"] += int(np.sum(p == 0))
            stats["negative_zeros"] += int(np.sum((p == 0) & np.signbit(p)))
            stats["n_values"] += int(p.size)
    import onnxruntime
    return np.stack(logits), stats, f"onnxruntime {onnxruntime.__version__} (CPUExecutionProvider, {threads} intra-op threads)"


# ------------------------------------------------------------------ reference: independent numpy ONNX evaluator
def _np_conv(x, w, b, pads, strides, dt):
    n, c, h, wd = x.shape
    o, ci, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    oh = (xp.shape[2] - kh) // strides[0] + 1
    ow = (xp.shape[3] - kw) // strides[1] + 1
    cols = np.empty((n, oh, ow, ci, kh, kw), dt)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, :, :, ky, kx] = xp[:, :, ky:ky + strides[0] * (oh - 1) + 1:strides[0], kx:kx + strides[1] * (ow - 1) + 1:strides[1]].transpose(0, 2, 3, 1)
    y = cols.reshape(n * oh * ow, ci * kh * kw) @ w.reshape(o, -1).astype(dt).T
    if b is not None:
        y = y + b.astype(dt)
    return y.reshape(n, oh, ow, o).transpose(0, 3, 1, 2).astype(dt)


def _np_pool(x, ks, st, pads, kind, dt):
    fill = -np.inf if kind == "max" else 0.0
    xp = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])), constant_values=fill)
    oh = (xp.shape[2] - ks[0]) // st[0] + 1
    ow = (xp.shape[3] - ks[1]) // st[1] + 1
    win = np.stack([xp[:, :, ky:ky + st[0] * (oh - 1) + 1:st[0], kx:kx + st[1] * (ow - 1) + 1:st[1]]
                    for ky in range(ks[0]) for kx in range(ks[1])])
    if kind == "max":
        return win.max(axis=0).astype(dt)
    assert not any(pads), "padded AveragePool not needed"
    return (win.sum(axis=0) / (ks[0] * ks[1])).astype(dt)


def evaluate_onnx(model, x_nchw: np.ndarray, dt=np.float64) -> dict:
    """Evaluates the (surgically modified) ONNX graph with numpy in dtype dt. Returns the full env."""
    from onnx import helper, numpy_helper

    env = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    env = {k: (v.astype(dt) if v.dtype.kind == "f" else v) for k, v in env.items()}
    gin = [i.name for i in model.graph.input if i.name not in env][0]
    env[gin] = x_nchw.astype(dt)
    for n in model.graph.node:
        a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
        ins = [env[i] if i else None for i in n.input]
        t = n.op_type
        if t == "Conv":
            assert a.get("group", 1) == 1 and list(a.get("dilations", [1, 1])) == [1, 1]
            y = _np_conv(ins[0], ins[1], ins[2] if len(ins) > 2 else None, list(a.get("pads", [0, 0, 0, 0])), list(a.get("strides", [1, 1])), dt)
        elif t == "BatchNormalization":
            sc, b, mu, var = ins[1:5]
            eps = dt(np.float32(a.get("epsilon", 1e-5)))
            shp = (1, -1, 1, 1)
            y = ((ins[0] - mu.reshape(shp)) / np.sqrt(var.reshape(shp) + eps) * sc.reshape(shp) + b.reshape(shp)).astype(dt)
        elif t == "Add":
            y = (ins[0] + ins[1]).astype(dt)
        elif t == "Relu":
            y = np.maximum(ins[0], dt(0))
        elif t == "Pad":
            p = list(a["pads"])
            k = len(p) // 2
            assert a.get("mode", b"constant") in (b"constant", "constant")
            y = np.pad(ins[0], [(p[d], p[d + k]) for d in range(k)], constant_values=a.get("value", 0.0)).astype(dt)
        elif t in ("MaxPool", "AveragePool"):
            assert not a.get("ceil_mode", 0)
            y = _np_pool(ins[0], list(a["kernel_shape"]), list(a.get("strides", a["kernel_shape"])), list(a.get("pads", [0, 0, 0, 0])),
                         "max" if t == "MaxPool" else "avg", dt)
        elif t == "Concat":
            y = np.concatenate(ins, axis=a["axis"])
        elif t == "Less":
            y = ins[0] < ins[1]
        elif t == "Greater":
            y = ins[0] > ins[1]
        elif t == "Where":
            y = np.where(ins[0], ins[1], ins[2]).astype(dt)
        else:
            raise NotImplementedError(t)
        env[n.output[0]] = y
    return env


# ------------------------------------------------------------------ workers
_G: dict = {}
_EXACT: dict[str, dict] = {}


def _init_worker(paths: dict, ref_model_bytes: bytes, probes: dict) -> None:
    import onnx
    _G["ref"] = onnx.load_from_string(ref_model_bytes)
    _G["probes"] = probes
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
        if k in _EXACT:
            gx = bfformat.read(p)
            for name, v in _EXACT[k].items():
                assert np.array_equal(v.astype(np.float32), gx.constants[name].data), name
                gx.constants[name].data = v
            _G[k + "#exact"] = gx


def _bin_inputs(env_ir, g) -> dict:
    """BConv2D node name -> its f32 input tensor (NHWC)."""
    return {n.name: env_ir[n.inputs[0]] for n in g.nodes if n.op == "BConv2D"}


def first_flip(ref_env: dict, probes: dict, ir_env: dict, g) -> dict | None:
    """First binary conv (graph order) whose ge-binarized input differs between the float64 reference
    and the Bitino float64 executor."""
    ir_in = _bin_inputs(ir_env, g)
    for conv_out, xname in probes.items():
        name = f"conv_{conv_out}"
        xr = ref_env[xname].transpose(0, 2, 3, 1)
        xi = ir_in[name]
        fl = (xr >= 0) != (xi >= 0)
        if fl.any():
            return {"bconv": name, "n_flips": int(fl.sum()), "max_abs_pre_ref": float(np.abs(xr[fl]).max()),
                    "max_abs_pre_bitino": float(np.abs(xi[fl]).max()), "typical_abs_pre_ref": float(np.median(np.abs(xr))),
                    "max_abs_diff_binarizer_input": float(np.abs(xr - xi).max())}
    return None


def _run_one(job):
    keys, x = job
    ref = _G["ref"]
    renv = evaluate_onnx(ref, x.transpose(0, 3, 1, 2), np.float64)
    res = {"ref64": renv[ref.graph.output[0].name].reshape(-1)}
    r32 = evaluate_onnx(ref, x.transpose(0, 3, 1, 2), np.float32)
    res["refnp32"] = r32[ref.graph.output[0].name].reshape(-1)
    del r32
    for k in keys:
        g = _G[k]
        env = executor.run(g, {"input": x}, keep_intermediates=True)
        bi = _bin_inputs(env, g)
        res[k + "32"] = env[g.outputs[0]].reshape(-1)
        res[k + "_zeros32"] = int(sum(np.sum(v == 0) for v in bi.values()))
        res[k + "_negzeros32"] = int(sum(np.sum((v == 0) & np.signbit(v)) for v in bi.values()))
        del env, bi
        env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        res[k + "64"] = env64[g.outputs[0]].reshape(-1)
        dev = float(np.abs(res[k + "64"] - res["ref64"]).max())
        res[k + "_flip"] = first_flip(renv, _G["probes"], env64, g) if dev > SEM_TOL else None
        del env64
        if k + "#exact" in _G:
            res[k + "_exact64"] = executor.run(_G[k + "#exact"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
    return res


def build_variant(model, args, entry, onnx_sha, dab_sha, binarize: str, pad_value: int, out_dir: str, x, labels, study, notes):
    import onnx
    import onnxruntime

    os.makedirs(out_dir, exist_ok=True)
    n = len(x)
    ref_model, probes = surgery(model, binarize, pad_value)
    ref32, zstats, ort_meta = ort_run(ref_model, x, args.ort_threads, ())
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(out_dir, "reference_logits.f32"))

    files, report_v = {}, {}
    for variant, fold in (("unfused", False), ("fused", True)):
        g, info = onnx_dabnn.convert(model, binarize=binarize, pad_value=pad_value)
        applied = passes.optimize(g, fold=fold)
        g.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": f"{variant}, input_binarize={binarize}, pad_value={pad_value}",
            "source": {"framework": "onnx (daBNN model zoo)", "class": MODELS[args.model_id], "repo": entry.get("repo_url", ""),
                       "checkpoint_file": os.path.basename(args.onnx), "checkpoint_sha256": onnx_sha,
                       "dab_file": os.path.basename(args.dab) if args.dab else None, "dab_sha256": dab_sha},
            "conversion": {"tool": "tools/convert_dabnn.py", "adapter": "bitino.adapters.onnx_dabnn", "passes": applied,
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "numpy": np.__version__, "onnx": onnx.__version__,
                           "binarization": f"all-+-1 Conv -> BConv2D, input_binarize {binarize}, pad_value {pad_value}, no weight scale"},
            "input": {"name": "input", "shape": [1, 224, 224, 3], "layout": "NHWC", "dtype": "f32",
                      "preprocessing": notes["preprocessing"]},
            "outputs": ["logits"], "op_versions": {"format": 1}, "layer_precision": info["layer_precision"],
        }
        if not fold:
            _EXACT[variant] = info["float64_constants"]
        path = os.path.join(out_dir, "model.bitino" if fold else "model_unfused.bitino")
        sha = bfformat.write(g, path)
        files[variant] = path
        report_v[variant] = {"file": os.path.basename(path), "file_sha256": sha, "file_bytes": os.path.getsize(path),
                             "op_counts": g.op_counts(), "passes": applied, "unused_initializers": info["unused_initializers"]}
        print(f"[{variant}] wrote {path}: {g.op_counts()} passes={applied}")

    jobs = [(list(files), x[i:i + 1]) for i in range(n)]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files, ref_model.SerializeToString(), probes)) as pool:
        res = pool.map(_run_one, jobs, chunksize=1)
    _EXACT.clear()
    ref64 = np.stack([r["ref64"] for r in res])
    refnp32 = np.stack([r["refnp32"] for r in res]).astype(np.float32)
    ref64.tofile(os.path.join(out_dir, "reference_logits_f64.f64"))

    def top5(z):
        return float(np.mean([labels[i] in np.argsort(-z[i])[:5] for i in range(n)]))

    rep = {"binarize": binarize, "pad_value": pad_value, "n_samples": n,
           "reference": {"framework": ort_meta, "definition": "original ONNX + explicit input binarization (Where/Less|Greater) + "
                         f"Pad(value={pad_value}) before each all-+-1 Conv (conv pads set to 0)",
                         "f64_definition": "numpy float64 evaluation of the same modified ONNX graph (tools/convert_dabnn.py evaluate_onnx); "
                                           "onnxruntime CPU Conv is float32-only",
                         "binary_convs": list(probes), "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)), "top5_accuracy": top5(ref32),
                         "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)), "top5_accuracy_f64": top5(ref64),
                         "binarization_inputs_onnxruntime_f32": study.get((binarize, pad_value), {}).get("binarization_inputs")},
           "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
           "numpy_onnx_f32_vs_onnxruntime_f32": compare_outputs(ref32, refnp32, labels),
           "variants": report_v}
    for k in files:
        out32 = np.stack([r[k + "32"] for r in res]).astype(np.float32)
        out64 = np.stack([r[k + "64"] for r in res])
        out32.tofile(os.path.join(out_dir, f"numpy_logits_{k}.f32"))
        sem = compare_outputs(ref64, out64, labels)
        per = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per <= SEM_TOL))
        sem["deviating_samples"] = [int(i) for i in np.where(per > SEM_TOL)[0]]
        sem["per_sample_max_abs"] = [float(v) for v in per]
        flips = {int(i): res[i][k + "_flip"] for i in sem["deviating_samples"]}
        sem["first_flips"] = flips
        fl = [f for f in flips.values() if f]
        nofl = [i for i, f in flips.items() if f is None]
        sem["first_flip_summary"] = {
            "n_deviating": len(flips), "n_without_flip": len(nofl),
            "max_abs_logit_dev_without_flip": float(max((per[i] for i in nofl), default=0.0)),
            "n_with_flip": len(fl), "max_abs_pre_at_first_flip": max((f["max_abs_pre_ref"] for f in fl), default=None),
            "max_rel_pre_at_first_flip": max((f["max_abs_pre_ref"] / f["typical_abs_pre_ref"] for f in fl), default=None),
            "max_n_flips_at_first_layer": max((f["n_flips"] for f in fl), default=None),
            "top1_changes": int(np.sum(out64.argmax(1) != ref64.argmax(1)))}
        if (k + "_exact64") in res[0]:
            outx = np.stack([r[k + "_exact64"] for r in res])
            px = np.abs(outx - ref64).max(axis=1)
            sem["float64_constants"] = {"max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-6": int(np.sum(px <= 1e-6)),
                                        "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                                        "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1)))}
        rep["variants"][k].update({
            "semantic_check_ir_f64_vs_reference_f64": sem,
            "numpy_vs_reference_f32": compare_outputs(ref32, out32, labels),
            "numpy_vs_reference_f64": compare_outputs(ref64, out32, labels),
            "exact_zeros_at_bconv_inputs_numpy_f32": {"total": int(sum(r[k + "_zeros32"] for r in res)),
                                                      "negative_zeros": int(sum(r[k + "_negzeros32"] for r in res))}})
        s = sem["first_flip_summary"]
        print(f"[{binarize} pad {pad_value} {k}] SEMANTIC f64: max_abs {sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} "
              f"<=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{n}; deviating {s['n_deviating']} (no flip {s['n_without_flip']}, max dev "
              f"{s['max_abs_logit_dev_without_flip']:.2e}; with flip {s['n_with_flip']}, max |pre| {s['max_abs_pre_at_first_flip']})"
              + (f" | f64 constants max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else "")
              + f" | numpy f32 vs ORT f32 top1 {rep['variants'][k]['numpy_vs_reference_f32']['top1_agreement']:.4f}")
    a = np.fromfile(os.path.join(out_dir, "numpy_logits_unfused.f32"), np.float32).reshape(n, -1)
    b = np.fromfile(os.path.join(out_dir, "numpy_logits_fused.f32"), np.float32).reshape(n, -1)
    rep["fused_vs_unfused_numpy_f32"] = compare_outputs(a, b, labels)
    print(f"[{binarize} pad {pad_value}] reference: ORT top1 {rep['reference']['top1_accuracy']:.4f}, f64 {rep['reference']['top1_accuracy_f64']:.4f}; "
          f"ORT f32 vs f64 top1 {rep['reference_f32_vs_f64']['top1_agreement']:.4f}; numpy-ONNX f32 vs ORT max_abs "
          f"{rep['numpy_onnx_f32_vs_onnxruntime_f32']['max_abs_error']:.3e}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dab", help="the matching daBNN .dab (hash recorded only)")
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ort-threads", type=int, default=4)
    ap.add_argument("--study-only", action="store_true", help="only the onnxruntime semantics study (pad -1/0 x ge/gt)")
    ap.add_argument("--no-pad0-variant", action="store_true")
    args = ap.parse_args()

    import onnx

    reg = json.load(open(args.registry))
    entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
    if entry is None:
        print(f"ERROR: {args.model_id} not in registry", file=sys.stderr)
        return 2
    onnx_sha = sha256_file(args.onnx)
    if onnx_sha != entry["checkpoint"]["sha256"]:
        print(f"ERROR: ONNX hash mismatch {onnx_sha} != {entry['checkpoint']['sha256']}", file=sys.stderr)
        return 3
    dab_sha = None
    if args.dab:
        dab_sha = sha256_file(args.dab)
        m = re.search(r"\.dab \(sha256 ([0-9a-f]{64})", entry["checkpoint"].get("inspection", ""))
        if m and m.group(1) != dab_sha:
            print(f"ERROR: .dab hash mismatch {dab_sha} != {m.group(1)}", file=sys.stderr)
            return 3
    print(f"checkpoint SHA-256 OK: onnx {onnx_sha} dab {dab_sha}")
    os.makedirs(args.out_dir, exist_ok=True)
    model = onnx.load(args.onnx)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    n = len(x)
    labels = np.array([s["label"] for s in em["samples"]][:n])
    notes = {"preprocessing": pre["definition"] + " (daBNN documents no preprocessing; the official Bi-Real ImageNet "
                                                   "validation transform is assumed)"}

    # Upgrade check (stem file): raw graph (no surgery) numpy fp32 on the original opset vs onnxruntime on the upgraded graph.
    up_model, up_note = upgrade(copy.deepcopy(model))
    upgrade_check = None
    if up_note:
        raw_ort, _, _ = ort_run(up_model, x[:8], args.ort_threads)
        raw_np = np.stack([evaluate_onnx(model, x[i:i + 1].transpose(0, 3, 1, 2), np.float64)[model.graph.output[0].name].reshape(-1) for i in range(8)])
        upgrade_check = {"note": up_note, "samples": 8, "raw_graph_onnxruntime_upgraded_f32_vs_numpy_original_f64": compare_outputs(raw_np, raw_ort)}
        print("upgrade check:", up_note, upgrade_check["raw_graph_onnxruntime_upgraded_f32_vs_numpy_original_f64"]["max_abs_error"])

    # Semantics study: onnxruntime fp32 top-1 for the plausible binarization conventions.
    study = {}
    for binarize, pad_value in (("ge", -1), ("gt", -1), ("ge", 0), ("gt", 0)):
        rm, probes = surgery(model, binarize, pad_value, probe=True)
        lg, st, meta = ort_run(rm, x, args.ort_threads, list(probes.values()))
        study[(binarize, pad_value)] = {"top1": float(np.mean(lg.argmax(1) == labels)),
                                        "top5": float(np.mean([labels[i] in np.argsort(-lg[i])[:5] for i in range(n)])),
                                        "binarization_inputs": st, "logits": lg}
        print(f"study: binarize {binarize} pad {pad_value}: top1 {study[(binarize, pad_value)]['top1']:.4f} "
              f"top5 {study[(binarize, pad_value)]['top5']:.4f} zeros {st}")
    base = study[("ge", -1)]["logits"]
    study_json = {f"{b}_pad{p}": {k: v for k, v in d.items() if k != "logits"} | {"vs_ge_pad-1": compare_outputs(base, d["logits"], labels)}
                  for (b, p), d in study.items()}
    if args.study_only:
        print(json.dumps(study_json, indent=1)[:4000])
        return 0

    report = {"model_id": args.model_id, "n_samples": n, "eval_set": em["name"], "eval_preprocessing": PREPROCESS,
              "eval_preprocessing_note": notes["preprocessing"],
              "source": {"onnx": os.path.basename(args.onnx), "onnx_sha256": onnx_sha, "dab": os.path.basename(args.dab) if args.dab else None,
                         "dab_sha256": dab_sha, "ir_version": model.ir_version,
                         "opset": [(o.domain, o.version) for o in model.opset_import], "producer": f"{model.producer_name} {model.producer_version}",
                         "binary_convs": onnx_dabnn.binary_convs(model), "license": entry.get("license"),
                         "claimed_metrics": entry.get("reported_metrics")},
              "opset_upgrade_check": upgrade_check, "semantics_study_onnxruntime_f32": study_json}
    report["primary"] = build_variant(model, args, entry, onnx_sha, dab_sha, "ge", -1, args.out_dir, x, labels, study, notes)
    if not args.no_pad0_variant:
        report["variant_pad0"] = build_variant(model, args, entry, onnx_sha, dab_sha, "ge", 0, os.path.join(args.out_dir, "variant_pad0"),
                                               x, labels, study, notes)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
