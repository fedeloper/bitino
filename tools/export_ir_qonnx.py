#!/usr/bin/env python3
"""Direct writer for the front ends without an ONNX exporter: a verified checkpoint -> canonical QONNX file of the
Bitino profile (docs/qonnx-profile.md, ``bitino-qonnx/1``); step 4 of the QONNX route (docs/qonnx-frontend.md § Plan,
docs/qonnx-step4.md).

Front ends (registry/recipes.json, by legacy converter):

  dabnn       tools/convert_dabnn.py            daBNN model-zoo Bi-Real ONNX (opset 6 / 9)          .cache/venv-qonnx
  nncf        tools/convert_nncf.py             NNCF-binarized torchvision ResNet-18 (XNOR, DoReFa)  .cache/venv-nncf
  tensorpack  tools/convert_tensorpack_dorefa.py  tensorpack DoReFa-Net AlexNet (TF1 npz)          .cache/venv-qonnx2
  fdabnn      tools/convert_fdabnn.py           FDA-BNN MindSpore checkpoint (read without MindSpore) .cache/venv-qonnx2
  omz         tools/convert_omz.py              OpenVINO Open Model Zoo IR (BinaryConvolution)       .cache/venv-openvino

Two stages, one file per published float file (``model.bitino``, ``model_sign_zero.bitino``):

  stage 1 (the front end's environment; python -I)
      <venv>/bin/python -I tools/export_ir_qonnx.py --model-id ID --out-dir .cache/qonnx_step4/ID [--eval-dir DIR]
    checks the checkpoint SHA-256 (registry/models.json), parses the source with the legacy adapter exactly as the
    legacy converter does (the adapter re-checks every layer against the source; the front ends below add their own
    checks), writes the source framework's fp32 / fp64 logits on the check set, and writes the ONNX graph of the parse
    (tools/qonnx_ir_writer.py) as a spec (spec.json + spec.npz) per published file, with the float64 values the adapter
    rounded (BatchNorm folds, binary scales, NNCF / DoReFa epilogues) as float64 carriers and the latent kernels of the
    source as BipolarQuant(W_latent, 1);
  stage 2 (.cache/venv-qonnx: onnx; started by stage 1, or ``--assemble DIR``)
      assembles each ONNX model (IR 10, ai.onnx 17 + qonnx.custom_op.general 1), the model and naming metadata
      (§ 6 / § 7) and checks it with bitino.qonnx_profile (the export fails unless check-qonnx passes with no error).

What the importer runs after the written graph is the legacy converter's own pass pipeline (passes.optimize with the
converter's fold flag); the data order the legacy converter wrote goes into bitino.const_order when it is not the
importer's first-use order. Writes per file into --out-dir/<file stem>/: model.onnx, export.json; and into --out-dir:
source_logits.f32 / source_logits_f64.f64 (the source framework, first output).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "python"))

from qonnx_ir_writer import IRWriter, LayerWriter, first_use_order, ir_shapes  # noqa: E402

PROFILE = "bitino-qonnx/1"
IR_VERSION = 10
OPSET = 17
QD = "qonnx.custom_op.general"
FAMILIES = {"tools/convert_dabnn.py": "dabnn", "tools/convert_nncf.py": "nncf",
            "tools/convert_tensorpack_dorefa.py": "tensorpack", "tools/convert_fdabnn.py": "fdabnn",
            "tools/convert_omz.py": "omz", "tools/convert_finn_onnx.py": "finn", "tools/convert_bnn_pynq_npz.py": "bnnpynq",
            "tools/convert_brevitas.py": "brevitas"}
VENVS = {"dabnn": "venv-qonnx", "nncf": "venv-nncf", "tensorpack": "venv-qonnx2", "fdabnn": "venv-qonnx2",
         "omz": "venv-openvino", "finn": "venv-qonnx", "bnnpynq": "venv-qonnx", "brevitas": "venv-brevitas"}
# the import layout, recorded in the file (model metadata bitino.import, mapping "patterns"): names from the
# metadata, data in first-use order (bitino.const_order where the legacy order is not), one ChannelAffine per affine
# group, no qonnx cleanup (it would fold the float64 carriers), and the bit-output layout: none (no converter here
# sets bit outputs with set_bit_outputs) or threshold (tools/convert_omz.py: fuse_threshold_into_bit_output). The
# file alone determines its .bitino: no import argument is needed.
IMPORT_OPTIONS = {"mapping": "patterns", "naming": "exporter", "constant_order": "first-use", "affine": "keep", "cleanup": "none",
                  "bit_outputs": "none"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def recipes() -> dict:
    return json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]


def family_of(model_id: str) -> str:
    rec = recipes().get(model_id)
    if rec is None or rec.get("converter") not in FAMILIES:
        raise SystemExit(f"{model_id}: not a step-4 recipe (converters {sorted(FAMILIES)})")
    return FAMILIES[rec["converter"]]


def step4_models() -> list:
    return [k for k, v in recipes().items() if v.get("converter") in FAMILIES]


def local_checkpoint(model_id: str, cache: str) -> str:
    rec = recipes()[model_id]
    models = {m["model_id"]: m for m in json.load(open(os.path.join(ROOT, "registry", "models.json")))["models"]}
    lc = models[rec.get("registry_model_id", model_id)]["checkpoint"]["local_cache_path"]
    return os.path.join(cache, lc[len(".cache/"):]) if lc.startswith(".cache/") else os.path.join(ROOT, lc)


# ================================================================================================ front ends
# Each returns {"variants": [{"file", "g0", "c64", "latent", "passes", "notes", "const_order"?}], "o32", "o64",
# "framework", "framework_version", "checkpoint_sha256", "source": {...}}; g0 is the adapter's graph before the
# legacy converter's passes, `passes(g)` applies them (the importer runs the same).

def _optimize(fold: bool):
    def run(g):
        from bitino import passes
        return passes.optimize(g, fold=fold)
    return run


def src_dabnn(args, rec, entry, x):
    import onnx

    import convert_dabnn as cd
    from bitino.adapters import onnx_dabnn
    ck = local_checkpoint(args.model_id, args.cache_dir)
    model = onnx.load(ck)
    g0, info = onnx_dabnn.convert(model, binarize="ge", pad_value=-1)        # the published variant (convert_dabnn main)
    out = {"framework": "onnx (daBNN model zoo)", "framework_version": f"onnx {onnx.__version__}",
           "checkpoint_sha256": sha256_file(ck),
           "source": {"onnx_file": os.path.basename(ck), "ir_version": model.ir_version,
                      "opset": [(o.domain, o.version) for o in model.opset_import],
                      "producer": f"{model.producer_name} {model.producer_version}", "binary_convs": len(onnx_dabnn.binary_convs(model)),
                      "unused_initializers": info["unused_initializers"]},
           "variants": [{"file": "model.bitino", "g0": g0, "c64": info["float64_constants"], "latent": {},
                         "passes": _optimize(True),
                         "notes": "daBNN onnx2bnn implicit input binarization of every all-+-1 Conv written as BipolarQuant(x, 1) "
                                  "(ge: the aarch64 sign-bit packing for every value except -0.0) + Pad(-1) (daBNN pads the packed "
                                  "input with bit 0 = -1); the ±1 weights as constants (no scale); BatchNorm folded in float64 "
                                  "(eps as the float32 attribute) into Mul / Add carriers"}]}
    if not args.no_source_logits:
        ref, _ = cd.surgery(model, "ge", -1)
        o32, _, meta = cd.ort_run(ref, x, args.threads, ())
        o64 = np.stack([cd.evaluate_onnx(ref, x[i:i + 1].transpose(0, 3, 1, 2), np.float64)[ref.graph.output[0].name].reshape(-1)
                        for i in range(len(x))])
        out.update({"o32": o32.astype(np.float32), "o64": o64,
                    "source_logits": "the daBNN ONNX with its implicit binarization made explicit (tools/convert_dabnn.py surgery: "
                                     f"ge, Pad -1): fp32 {meta}; fp64 the numpy evaluator convert_dabnn.evaluate_onnx"})
    return out


def src_fdabnn(args, rec, entry, x):
    import torch

    import convert_s12_cifar as cs
    from bitino.adapters import mindspore_ckpt, mindspore_fdabnn
    torch.set_num_threads(args.threads)
    ck = local_checkpoint(args.model_id, args.cache_dir)
    repo = os.path.join(args.cache_dir, "repos", "mindspore-models-FDA-BNN")
    params = mindspore_ckpt.read(ck)
    want = {d["file"]: d["sha256"] for d in rec.get("downloads", []) if d["name"] in ("quan", "resnet")}
    got = {f: sha256_file(os.path.join(repo, f)) for f in want}
    if got != want:
        raise SystemExit(f"FDA-BNN source files {got} != recipe {want}")
    variants = []
    for binarize, file in (("ge", "model.bitino"), ("sign_zero", "model_sign_zero.bitino")):
        g0, info = mindspore_fdabnn.convert(params, args.model_id, binarize=binarize)
        lat = {f"{k}/kernel_bits": np.asarray(params[f"{k}.weight"], np.float32) for k in info["binary_weights"]}
        variants.append({"file": file, "g0": g0, "c64": info["float64_constants"], "latent": lat, "passes": _optimize(True),
                         "notes": f"QuanConv: AQuan Sign(x) (MindSpore Sign(0) = 0) written as "
                                  + ("Sign(x) (sign_zero, exact)" if binarize == "sign_zero" else "BipolarQuant(x, 1) (ge, the deployable variant)")
                                  + ", zero padding of the binarized input as Pad(0), WQuan sign(W) * mean|W| as BipolarQuant(W, 1) "
                                    "+ Mul(mean|W| float64 carrier); BatchNorm (eps 1e-5) folded in float64; PReLU slope broadcast "
                                    "to the channels; shortcut x[:, :, ::2, ::2] + channel zero pad as AveragePool(1, stride 2) + Pad"})
    out = {"framework": "mindspore (checkpoint read without MindSpore: bitino.adapters.mindspore_ckpt)",
           "framework_version": f"mindspore checkpoint protobuf; reference PyTorch port tools/fdabnn_reference.py, torch {torch.__version__}",
           "checkpoint_sha256": sha256_file(ck), "source": {"source_files": got}, "variants": variants}
    if not args.no_source_logits:
        model, _, _, notes = cs.load_source("fdabnn", repo, ck)
        o32, _ = cs.run_reference("fdabnn", model, x, torch.float32)
        o64, _ = cs.run_reference("fdabnn", model, x, torch.float64)
        out.update({"o32": o32.astype(np.float32), "o64": o64.astype(np.float64),
                    "source_logits": "the cited PyTorch port of the official src/resnet.py + src/quan.py (tools/fdabnn_reference.py, "
                                     "MindSpore Sign: sign(0) = 0) in float32 / float64"})
    return out


def src_tensorpack(args, rec, entry, x):
    import torch

    import convert_tensorpack_dorefa as ct
    import dorefa_alexnet_reference as port
    from bitino.adapters import tensorpack_dorefa
    torch.set_num_threads(args.threads)
    ck = local_checkpoint(args.model_id, args.cache_dir)
    params = port.load(ck)
    g0, info = tensorpack_dorefa.convert(params, args.model_id)
    p = {k[:-2] if k.endswith(":0") else k: v for k, v in params.items()}
    lat = {}
    for name, sign in info["binary_weights"].items():
        w = p[f"{name}/W"]
        if w.ndim == 4:                                            # HWIO -> OIHW
            lat[f"{name}/kernel_bits"] = np.ascontiguousarray(w.transpose(3, 2, 0, 1))
        elif name == "fc0":                                        # rows (c, h, w) -> (h, w, c), as the adapter: [O, K]
            lat[f"{name}/kernel_bits"] = np.ascontiguousarray(w.reshape(256, 6, 6, 4096).transpose(1, 2, 0, 3).reshape(9216, 4096).T)
        else:
            lat[f"{name}/kernel_bits"] = np.ascontiguousarray(w.T)
    out = {"framework": "tensorpack (TF1 DoReFa-Net, npz read with numpy)",
           "framework_version": f"npz; reference PyTorch port tools/dorefa_alexnet_reference.py, torch {torch.__version__}",
           "checkpoint_sha256": sha256_file(ck), "source": {"npz_tensors": len(params)},
           "variants": [{"file": "model.bitino", "g0": g0, "c64": info["float64_constants"], "latent": lat, "passes": _optimize(True),
                         "notes": "DoReFa W1A1: u = round(clip(x, 0, 1)) = [x > 0.5] written as Where(Greater(x, 0.5), 1, -1) "
                                  "(u = (1 + s) / 2) + Pad(-1) (a SAME-padded tap is u = 0, s = -1), binary weights "
                                  "where(W == 0, 1, sign(W)) as BipolarQuant(W, 1), E = mean|W| folded into the float64 epilogue "
                                  "post_mul = E / 2, post_add = E / 2 * sum w (+ bias); grouped convs (split 2) as Conv group 2; "
                                  "BatchNorm (eps 1e-4) folded in float64; the NCHW flatten before fc0 as Transpose + Reshape "
                                  "with fc0's rows in (h, w, c) order"}]}
    if not args.no_source_logits:
        model = port.DoReFaAlexNet(params)
        o32 = ct.port_logits(model, x, torch.float32)
        o64 = ct.port_logits(model, x, torch.float64)
        out.update({"o32": o32.astype(np.float32), "o64": o64.astype(np.float64),
                    "source_logits": "the PyTorch port of alexnet-dorefa.py --dorefa 1,1,32 (tools/dorefa_alexnet_reference.py) "
                                     "in float32 / float64"})
    return out


def src_nncf(args, rec, entry, x):
    import nncf
    import torch

    import convert_nncf as cn
    from bitino.adapters import torch_nncf
    torch.set_num_threads(args.threads)
    mode = cn.MODELS[args.model_id]
    ck = local_checkpoint(args.model_id, args.cache_dir)
    ck_sha = sha256_file(ck)
    state = os.path.join(args.cache_dir, "derived-nncf", f"resnet18_{mode}.state.pt")
    cfg = os.path.join(args.cache_dir, "repos", "nncf-v2.5.0", "examples", "torch", "classification", "configs", "binarization",
                       f"resnet18_imagenet_binarization_{mode}.json")
    model, meta, n_loaded, n_state = cn.build_model(state, cfg)
    if meta["source_sha256"] != ck_sha:
        raise SystemExit("derived NNCF state was not extracted from the checkpoint")
    ins = cn.Instrument(model, mode == "dorefa")          # --dorefa-weight-scale cuda (the converter's default)
    ref32, _ = cn.reference_outputs(model, ins, x, torch.float32)
    cap = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in ins.capture.items()}
    capnp = {k: {"wb": v["wb"].numpy(), "scale": float(v["scale"].reshape(-1)[0]), "sigmoid": v["sigmoid"].numpy()} for k, v in cap.items()}
    g0, info = torch_nncf.convert(model, capnp, mode, args.model_id)
    lat = {f"{k}/kernel_bits": model.get_submodule(k).weight.detach().cpu().numpy().astype(np.float32) for k in torch_nncf.BINARIZED}
    out = {"framework": "pytorch+nncf", "framework_version": f"nncf {nncf.__version__}, torch {torch.__version__}",
           "checkpoint_sha256": ck_sha,
           "source": {"mode": mode, "config": os.path.basename(cfg), "config_sha256": sha256_file(cfg),
                      "derived_state": os.path.basename(state), "derived_state_sha256": sha256_file(state),
                      "dorefa_weight_scale": "cuda" if mode == "dorefa" else None, "n_loaded": n_loaded, "n_state": n_state},
           "variants": [{"file": "model.bitino", "g0": g0, "c64": info["float64_constants"], "latent": lat, "passes": _optimize(True),
                         "notes": "NNCF ActivationBinarizationScaleThreshold a = s * [x > t] (t = fl32(sigmoid(theta) * s)) with the "
                                  "binary weights wb = sign(w > 0) * alpha: sum a * wb = alpha s / 2 (acc + Wsum), acc the ±1 conv "
                                  "of Where(Greater(x, t), 1, -1) padded with -1 (a padded tap is the 0 level); written as "
                                  "Where(Greater) + Pad(-1) + Conv(BipolarQuant(w, 1)) + Mul(alpha s / 2) + Add(alpha s Wsum / 2), "
                                  "float64 carriers; BatchNorm folded in float64"
                                  + ("; DoReFa weight scale mean|W[0]| (NNCF's CUDA kernel, the trained network)" if mode == "dorefa" else "")}]}
    if not args.no_source_logits:
        override = {k: {"wb": v["wb"].double(), "scale": v["scale"].double(), "sigmoid": v["sigmoid"].double()} for k, v in cap.items()}
        ins.capture = {}
        o64, _ = cn.reference_outputs(model, ins, x, torch.float64, override=override)
        out.update({"o32": ref32.astype(np.float32), "o64": o64.astype(np.float64),
                    "source_logits": "official NNCF (create_compressed_model + load_state) in float32; float64 with the fp32 binarizer "
                                     "constants substituted (tools/convert_nncf.py primary reference)"})
    return out


def src_omz(args, rec, entry, x):
    import re

    import convert_omz as co
    from bitino import passes
    from bitino.adapters import openvino_ir as oi
    sub, prec, stem = co.MODELS[args.model_id]
    d = os.path.join(args.cache_dir, "checkpoints", sub, prec)
    xml, binf = os.path.join(d, stem + ".xml"), os.path.join(d, stem + ".bin")
    m = re.search(r"sha256 ([0-9a-f]{64})", entry["checkpoint"].get("download_status", ""))
    xml_sha, bin_sha = sha256_file(xml), sha256_file(binf)
    if m and m.group(1) != xml_sha:
        raise SystemExit(f"IR xml hash mismatch {xml_sha} (registry {m.group(1)})")
    ir = oi.IRModel(xml, binf)
    g0, info = oi.convert(ir, args.model_id)
    # the adapter-specific rewrite before the common passes (tools/convert_omz.py build_graphs): each chain of
    # constant Multiply / Add nodes merged into one ChannelAffine, composed pairwise in float64 and rounded to float32
    # after every merge (the published constants); the canonical file writes the merged affines
    merges = oi.merge_affine_chains(g0)

    def legacy(g):
        applied = passes.optimize(g, fold=True)
        applied["fuse_threshold_into_bit_output"] = oi.fuse_threshold_into_bit_output(g)
        applied["prune_unused_constants"] += passes.prune_unused_constants(g)
        return applied
    out = {"framework": "openvino-ir", "framework_version": f"OpenVINO IR v{ir.version} ({prec}), parsed with numpy (xml.etree)",
           "checkpoint_sha256": {os.path.basename(binf): bin_sha, os.path.basename(xml): xml_sha},
           "source": {"xml": os.path.basename(xml), "xml_sha256": xml_sha, "bin_sha256": bin_sha, "ir_version": ir.version,
                      "ir_op_counts": ir.op_counts(), "u1_order": ir.u1_order, "merge_affine_chains": merges},
           "variants": [{"file": "model.bitino", "g0": g0, "c64": {}, "latent": {}, "passes": legacy,
                         "import_options": {**IMPORT_OPTIONS, "bit_outputs": "threshold"},
                         "notes": "OpenVINO FakeQuantize(levels 2, input_low = input_high = t, outputs {0, 1}) + BinaryConvolution "
                                  "(xnor-popcount, pad_value -1, u1 weights LSB-first, bit 1 = +1) written as Where(Greater(x, t), 1, -1) "
                                  "+ Pad(-1) + Conv(±1 constants); every chain of constant Multiply / Add nodes merged as the legacy "
                                  "converter does (float64 composition, float32 rounding after each merge: FLOAT constants); "
                                  "bitino.import bit_outputs threshold: a binary conv feeding only gt binarizers emits bits"}]}
    if not args.no_source_logits:
        o32, meta = co.openvino_reference(xml, binf, x)
        import multiprocessing as mp
        with mp.get_context("fork").Pool(max(1, args.threads)) as pool:
            o64 = np.stack(pool.map(_omz_f64, [(xml, binf, x[i:i + 1]) for i in range(len(x))], chunksize=1))
        out.update({"o32": o32.astype(np.float32), "o64": o64,
                    "source_logits": f"OpenVINO runtime ({meta['framework']}, CPU, INFERENCE_PRECISION_HINT f32) in float32; the "
                                     "numpy float64 evaluation of the IR graph (bitino.adapters.openvino_ir.evaluate, OpenVINO is "
                                     "float32 only)"})
    return out


def _omz_f64(job):
    from bitino.adapters import openvino_ir as oi
    xml, binf, x = job
    ir = oi.IRModel(xml, binf)
    return oi.evaluate(ir, np.ascontiguousarray(x.transpose(0, 3, 1, 2)), np.float64)["outputs"][0].reshape(-1)


# ------------------------------------------------------------------------------------------------ FINN-style layer specs
# The FINN / BNN-PYNQ / Brevitas converters fold BN + sign into integer thresholds inside the adapter (fused
# variant) and run passes.optimize(fold=False) only; their canonical files carry the source's BN affines as float64
# carriers before a ge binarizer, and the importer's fold mode derives the same integer thresholds.
IMPORT_OPTIONS_LAYERS = {"mapping": "patterns", "naming": "exporter", "constant_order": "walk", "affine": "fold", "cleanup": "none",
                         "bit_outputs": "set"}
LAYER_NOTES = ("binary layers as Conv / Gemm with ±1 constants (VALID), BatchNorm (and conv / dense biases, scalar Mul) folded in "
               "float64 into Mul(scale) -> Add(shift) carriers before BipolarQuant(x, 1) (sign(0) = +1); the importer folds "
               "BN + sign into the exact integer thresholds of the legacy converter (brevitas_bnn.integer_threshold); "
               "MaxPool after the sign on the ±1 activation")


def src_finn(args, rec, entry, x):
    import io
    import zipfile

    import onnx

    import convert_finn_onnx as cf
    from bitino import passes
    from bitino.adapters import finn_bnn, finn_onnx
    cfg = cf.MODELS[args.model_id]
    ck = local_checkpoint(args.model_id, args.cache_dir)
    raw = zipfile.ZipFile(io.BytesIO(open(ck, "rb").read())).read(cfg["member"])
    if hashlib.sha256(raw).hexdigest() != cfg["member_sha256"]:
        raise SystemExit(f"{cfg['member']}: SHA-256 differs from {cfg['member_sha256']}")
    model = onnx.load_from_string(raw)
    inp, layers, rinfo = finn_onnx.read(model, cfg["domain"])
    gf, _ = finn_bnn.emit(inp, layers, fused=True)
    passes.optimize(gf, fold=False)
    out = {"framework": "onnx (FINN-ONNX / QONNX release of finn-examples)", "framework_version": f"onnx {onnx.__version__}",
           "checkpoint_sha256": sha256_file(ck),
           "source": {"zip_member": cfg["member"], "member_sha256": cfg["member_sha256"], "input_quantizer_check": rinfo.get("input_quantizer_check"),
                      "producer": f"{model.producer_name} {model.producer_version}".strip()},
           "variants": [{"file": "model.bitino", "writer": "layers", "inp": inp, "layers": layers, "gf": gf, "import_options": IMPORT_OPTIONS_LAYERS,
                         "notes": f"rewrite of the release ONNX to the profile (Q10): {inp.source} written as "
                                  + ("Mul(a) -> Add(b) -> IntQuant(scale, 0, 8, signed, ROUND) (verified exhaustively on the 256 input values "
                                     "against the release's own quantizer by bitino.adapters.finn_onnx)" if inp.kind == "int8" else
                                     f"Where(GreaterOrEqual(x, {inp.threshold}), 1, -1) (verified on the 256 input values)")
                                  + f"; {LAYER_NOTES}; MultiThreshold / BatchNormalization / TopK removed (TopK: the IR outputs the logits)"}]}
    if not args.no_source_logits:
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.core.onnx_exec import execute_onnx
        from qonnx.transformation.infer_shapes import InferShapes
        mw = ModelWrapper(model)
        if not all(vi.type.tensor_type.HasField("shape") for vi in mw.graph.value_info) or len(mw.graph.value_info) == 0:
            mw = mw.transform(InferShapes())
        ev = finn_onnx.OnnxNumpy(model)
        lg = rinfo["logits_onnx_tensor"]
        o32, o64 = [], []
        for i in range(len(x)):
            xi = np.ascontiguousarray(x[i:i + 1].transpose(0, 3, 1, 2))
            o32.append(np.asarray(execute_onnx(mw, {ev.input_name: xi}, return_full_exec_context=True)[lg], np.float32).reshape(-1))
            o64.append(np.asarray(ev.run(xi, np.float64, outputs=[lg])[lg], np.float64).reshape(-1))
        out.update({"o32": np.stack(o32), "o64": np.stack(o64),
                    "source_logits": "the release ONNX itself: qonnx execute_onnx (float32) and bitino.adapters.finn_onnx.OnnxNumpy "
                                     "(float64) at the logits tensor before TopK"})
    return out


def src_bnnpynq(args, rec, entry, x):
    import tempfile

    import convert_bnn_pynq_npz as cb
    from bitino import passes
    from bitino.adapters import bnn_pynq_npz, finn_bnn
    cfg = cb.MODELS[args.model_id]
    ck = local_checkpoint(args.model_id, args.cache_dir)
    with np.load(ck, allow_pickle=False) as f:
        arrs = [np.asarray(f[f"arr_{i}"]) for i in range(len(f.files))]
    ev = rec["eval"]
    eval_dir = args.eval_dir or os.path.join(args.cache_dir, "data", ev["dir"])
    pre = json.load(open(os.path.join(eval_dir, "manifest.json")))["preprocessing"][cfg["pre"]]
    x_file = os.path.join(eval_dir, pre["file"])
    tdir = tempfile.mkdtemp(prefix="bnnpynq_s4_", dir=args.tmp_dir)
    targs = argparse.Namespace(theano_python=os.path.join(args.cache_dir, "env-theano27", "bin", "python"),
                               theano_compiledir=os.path.join(args.tmp_dir, "theanoc"), npz=ck,
                               lasagne_dir=os.path.join(args.cache_dir, "downloads-s7", "lasagne", "Lasagne-master"),
                               bnnpynq_dir=os.path.join(args.cache_dir, "repos", "bnn-pynq", "bnn", "src", "training"))
    os.makedirs(targs.theano_compiledir, exist_ok=True)
    d32 = cb.theano_run(targs, cfg, x_file, pre["shape"], "float32", os.path.join(tdir, "f32.npz"))
    wq32 = [d32[f"wq_{k}"] for k in range(len([k for k in d32 if k.startswith("wq_")]))]
    inp, layers, rinfo = bnn_pynq_npz.read(arrs, wq32, cfg["arch"], cfg["n_out"])
    gf, _ = finn_bnn.emit(inp, layers, fused=True)
    passes.optimize(gf, fold=False)
    versions = " / ".join(str(v) for v in d32["versions"].tolist())
    out = {"framework": "theano+lasagne (BNN-PYNQ npz)", "framework_version": versions, "checkpoint_sha256": sha256_file(ck),
           "source": {"n_arrays": len(arrs), "weights_vs_finnthesizer_ge0": rinfo["weights_vs_finnthesizer_ge0"],
                      "input_quantizer_check": rinfo.get("input_quantizer_check")},
           "variants": [{"file": "model.bitino", "writer": "layers", "inp": inp, "layers": layers, "gf": gf, "import_options": IMPORT_OPTIONS_LAYERS,
                         "notes": f"{inp.source}; binary weights = Theano's own QuantizationBinary.quantizeWeights output (float32, "
                                  f"round half to even of hard_sigmoid(W): written as ±1 constants, not as the sign of the latent W); "
                                  f"{LAYER_NOTES}; conv -> MaxPool -> BN -> sign (pool before BN, Lasagne order) written in that order"}]}
    if not args.no_source_logits:
        d64 = cb.theano_run(targs, cfg, x_file, pre["shape"], "float64", os.path.join(tdir, "f64.npz"), binary_weights=os.path.join(tdir, "f32.npz"))
        out.update({"o32": d32["logits"].astype(np.float32), "o64": d64["logits"].astype(np.float64),
                    "source_logits": f"the official BNN-PYNQ Theano / Lasagne networks ({versions}), deterministic=True, float32; "
                                     "float64 with the float32 binary weights"})
    shutil.rmtree(tdir, ignore_errors=True)
    return out


def _brevitas_layers(model):
    """The Brevitas BNN-PYNQ CNV / FC module (checked by the legacy adapter's probes) as FINN-style layer specs."""
    from bitino.adapters import brevitas_bnn as bb
    from bitino.adapters.finn_bnn import InputSpec, Layer
    import torch
    layers = []

    def fc(mods, start, prefix):
        i = start
        while i < len(mods):
            m = mods[i]
            if type(m).__name__ == "Dropout":
                i += 1
                continue
            name = f"{prefix}.{i}"
            bb._check_layer_quant_off(m, name)
            w = bb.binary_weight(m, name)
            if i + 1 < len(mods) and type(mods[i + 1]).__name__ == "TensorNorm":
                tn = mods[i + 1]
                wt, bt, rm, rv = (float(p.detach().to(torch.float64)) for p in (tn.weight, tn.bias, tn.running_mean, tn.running_var))
                mul = wt / np.sqrt(rv + float(tn.eps))
                add = bt - rm * mul
                o = w.shape[0]
                layers.append(Layer(name, "dense", w, np.full(o, mul), np.full(o, add), final=True, source="QuantLinear + TensorNorm"))
                return
            bn, act = mods[i + 1], mods[i + 2]
            bb.probe_binary_act(act, f"{prefix}.{i + 2}")
            p = bb.bn_params(bn)
            layers.append(Layer(name, "dense", w, p["scale"], p["shift"], source="QuantLinear + BatchNorm1d + binary QuantIdentity"))
            i += 3
        raise NotImplementedError(f"{prefix}: no final QuantLinear + TensorNorm")

    if type(model).__name__ == "FC":
        bb.probe_binary_act(model.features[0], "features.0")
        inp = InputSpec("affine_sign", [1, 28, 28, 1], source="x -> 2x - 1 -> QuantIdentity (ClampedBinaryQuant, ge)")
        fc(list(model.features), 1, "features")
        return inp, layers, (2.0, -1.0)
    cf, lf = list(model.conv_features), list(model.linear_features)
    q = bb.probe_int8_input_quant(cf[0], "conv_features.0")
    inp = InputSpec("int8", [1, 32, 32, int(cf[1].in_channels)], affine=(2.0, -1.0), q={"scale": q["scale"], "qmin": q["qmin"], "qmax": q["qmax"]},
                    source="x -> 2x - 1 -> QuantIdentity (RescalingIntQuant, 8 bit, scale 1/128, round half to even)")
    i, ci = 1, 0
    while i < len(cf):
        m = cf[i]
        name = f"conv_features.{i}"
        bb._check_layer_quant_off(m, name)
        w = bb.binary_weight(m, name)
        bn, act = cf[i + 1], cf[i + 2]
        bb.probe_binary_act(act, f"conv_features.{i + 2}")
        p = bb.bn_params(bn)
        pool = i + 3 < len(cf) and type(cf[i + 3]).__name__ == "MaxPool2d"
        layers.append(Layer(name, "conv", w, p["scale"], p["shift"], pool="after_sign" if pool else None, w1a8=(ci == 0),
                            source="QuantConv2d + BatchNorm2d + binary QuantIdentity" + (" + MaxPool2d" if pool else "")))
        i += 4 if pool else 3
        ci += 1
    fc(lf, 0, "linear_features")
    return inp, layers, None


def src_brevitas(args, rec, entry, x):
    import brevitas
    import torch

    import convert_brevitas as cv
    from bitino import passes
    from bitino.adapters import brevitas_bnn
    torch.set_num_threads(args.threads)
    cfg_name, _, _ = cv.MODELS[args.model_id]
    ck = local_checkpoint(args.model_id, args.cache_dir)
    model, src_hashes, cfg_dict, n_tensors = cv.load_model(cfg_name, ck)
    gf, _ = brevitas_bnn.convert(model, args.model_id, fused=True)       # also the adapter's module / quantizer checks
    passes.optimize(gf, fold=False)
    inp, layers, aff = _brevitas_layers(model)
    out = {"framework": "pytorch+brevitas", "framework_version": f"brevitas {brevitas.__version__}, torch {torch.__version__}",
           "checkpoint_sha256": sha256_file(ck), "source": {"cfg": cfg_name, "source_sha256": src_hashes, "checkpoint_tensors": n_tensors},
           "variants": [{"file": "model.bitino", "writer": "layers", "inp": inp, "layers": layers, "gf": gf, "affine_sign": aff,
                         "import_options": IMPORT_OPTIONS_LAYERS,
                         "notes": f"{inp.source} written as " + ("Mul(2) -> Add(-1) -> IntQuant(1/128, 0, 8, signed, ROUND)" if inp.kind == "int8"
                                                                 else "Mul(2) -> Add(-1) -> BipolarQuant(x, 1)")
                                  + f"; {LAYER_NOTES}; the binary weights are the modules' own quant_weight() (±1 constants, = ge-sign of the "
                                    "latent weights); the final TensorNorm folded in float64 into the last layer's affine"}]}
    if not args.no_source_logits:
        o32, _ = cv.reference_run(model, x, torch.float32)
        o64, _ = cv.reference_run(model, x, torch.float64)
        out.update({"o32": o32.astype(np.float32), "o64": o64.astype(np.float64),
                    "source_logits": f"the official Brevitas model (brevitas_examples.bnn_pynq model_with_cfg('{cfg_name}')) in float32 / float64"})
    return out


SOURCES = {"finn": src_finn, "bnnpynq": src_bnnpynq, "brevitas": src_brevitas, "omz": src_omz, "dabnn": src_dabnn, "fdabnn": src_fdabnn, "tensorpack": src_tensorpack, "nncf": src_nncf}


# ================================================================================================ stage 1
def stage1(args) -> int:
    fam = family_of(args.model_id)
    rec = recipes()[args.model_id]
    reg = json.load(open(os.path.join(ROOT, "registry", "models.json")))
    entry = next(m for m in reg["models"] if m.get("model_id") == rec.get("registry_model_id", args.model_id))
    os.makedirs(args.out_dir, exist_ok=True)
    ev = rec["eval"]
    eval_dir = args.eval_dir or os.path.join(args.cache_dir, "data", ev["dir"])
    em = json.load(open(os.path.join(eval_dir, "manifest.json")))
    pre = em["preprocessing"][ev["preprocess"]]
    x = np.fromfile(os.path.join(eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    src = SOURCES[fam](args, rec, entry, x)
    ck = src["checkpoint_sha256"]
    want = entry["checkpoint"]["sha256"]
    if (ck if isinstance(ck, str) else None) != want and not (isinstance(ck, dict) and want in ck.values()):
        print(f"ERROR: checkpoint hash mismatch: {ck} (registry {want})", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {ck}", flush=True)
    labels_list = [s.get("label") for s in em["samples"]][: len(x)]
    if "o32" in src:
        src["o32"].astype(np.float32).tofile(os.path.join(args.out_dir, "source_logits.f32"))
        if src.get("o64") is not None:
            np.asarray(src["o64"], np.float64).tofile(os.path.join(args.out_dir, "source_logits_f64.f64"))
    lic = (entry.get("license") or {}).get("weights") or "unknown"
    nc = int(rec["eval"].get("validate_args", ["--num-classes", "1000"])[1]) if "--num-classes" in rec["eval"].get("validate_args", []) else 1000
    lset = {10: "cifar10" if "cifar" in ev["dir"] else "mnist", 1000: "imagenet-1k", 44: "gtsrb-44"}.get(nc, f"{nc}-classes")
    commit = _git_commit()
    sc = (rec.get("sources") or [{}])[0]
    rc = 0
    py2 = args.qonnx_python or os.path.join(args.cache_dir, "venv-qonnx", "bin", "python")
    for var in src["variants"]:
        if args.only_file and var["file"] not in args.only_file:
            continue
        stem = var["file"][: -len(".bitino")]
        vdir = os.path.join(args.out_dir, stem)
        os.makedirs(vdir, exist_ok=True)
        if var.get("writer") == "layers":
            # FINN-style layer specs; the legacy graph gf (adapter fused + passes.optimize(fold=False)) names the groups
            g0 = var["gf"]
            w = LayerWriter(var["inp"], var["layers"], g0, var.get("affine_sign"))
            spec = w.write()
            shapes = ir_shapes(g0)
            counter0, applied, const_order = 0, {"passes.optimize(fold=False)": "in the adapter"}, var.get("const_order")
        else:
            g0 = var["g0"]
            shapes = ir_shapes(g0)
            w = IRWriter(g0, shapes, var.get("c64"), var.get("latent"))
            spec = w.write()
            if w.report["c64_not_written"]:
                print(f"note: float64 values not written as carriers (thresholds: the float32 IR value): {w.report['c64_not_written'][:5]}")
            gp = copy.deepcopy(g0)
            counter0 = int(getattr(g0, "_counter", 0))
            applied = var["passes"](gp)
            const_order = var.get("const_order")
            if const_order is None and first_use_order(g0) != list(g0.constants):
                const_order = list(gp.constants)
        meta = {"bitino.profile": PROFILE, "bitino.model_id": args.model_id, "bitino.variant": "float", "bitino.layout": "NCHW",
                "bitino.input_domain": "float32",
                "bitino.preprocessing": json.dumps({"key": ev["preprocess"], "definition": pre["definition"]}),
                "bitino.labels": json.dumps({"num_classes": nc, "set": lset}),
                "bitino.source.framework": src["framework"], "bitino.source.framework_version": src["framework_version"],
                "bitino.source.checkpoint_sha256": ck if isinstance(ck, str) else json.dumps(ck, sort_keys=True),
                "bitino.weights_license": lic,
                "bitino.exporter": f"tools/export_ir_qonnx.py@{commit or 'unknown'} (direct writer from the {fam} parse, legacy "
                                   f"converter {rec['converter']}, adapter {rec.get('adapter')}); numpy {np.__version__}",
                "bitino.notes": var["notes"] + f"; published file {var['file']}"}
        if sc.get("repo") and sc.get("commit"):
            meta["bitino.source.code"] = f"{sc['repo']}@{sc['commit']}"
        opts = var.get("import_options", IMPORT_OPTIONS)
        meta["bitino.import"] = json.dumps(opts, sort_keys=True)
        if counter0:
            meta["bitino.ir_unique_counter"] = str(counter0)
        if const_order is not None:
            meta["bitino.const_order"] = json.dumps(const_order)
        sdir = os.path.join(vdir, "spec")
        os.makedirs(sdir, exist_ok=True)
        np.savez(os.path.join(sdir, "spec.npz"), **{f"i{j}": v for j, v in enumerate(spec.inits.values())})
        exp = {"model_id": args.model_id, "family": fam, "file": var["file"], "legacy_converter": rec["converter"],
               "adapter": rec.get("adapter"), "checkpoint_file": os.path.basename(local_checkpoint(args.model_id, args.cache_dir)),
               "checkpoint_sha256": ck, "framework": f"{src['framework_version']}, numpy {np.__version__}, python {platform.python_version()}",
               "exporter": f"tools/export_ir_qonnx.py@{commit or 'unknown'} direct writer ({fam})",
               "source": src.get("source"), "import_args": [], "import_options": opts, "writer": w.report, "ir_nodes": len(g0.nodes),
               "ir_constants": len(g0.constants), "legacy_passes": applied, "const_order": const_order is not None,
               "ir_unique_counter": counter0, "eval_set": em["name"], "eval_preprocessing": ev["preprocess"], "n_samples": int(len(x)),
               "output_widths": [int(np.prod(shapes[o][1:])) for o in g0.outputs],
               "source_logits": src.get("source_logits")}
        if "o32" in src:
            exp.update({"logits_f32": os.path.abspath(os.path.join(args.out_dir, "source_logits.f32")),
                        "source_top1": float(np.mean(src["o32"].argmax(1) == np.array(labels_list)))})
            if src.get("o64") is not None:
                exp.update({"logits_f64": os.path.abspath(os.path.join(args.out_dir, "source_logits_f64.f64")),
                            "source_f32_vs_f64_max_abs": float(np.abs(src["o32"].astype(np.float64) - src["o64"]).max())})
        json.dump({"nodes": spec.nodes, "init_names": list(spec.inits), "inputs": spec.inputs, "outputs": spec.outputs,
                   "meta": meta, "export": exp}, open(os.path.join(sdir, "spec.json"), "w"))
        print(f"[{var['file']}] spec written: {len(spec.nodes)} ONNX nodes, writer {w.report}", flush=True)
        r = subprocess.run([py2, "-I", os.path.abspath(__file__), "--assemble", vdir] + (["--keep-spec"] if args.keep_spec else []))
        rc = rc or r.returncode
    return rc


# ================================================================================================ stage 2 (onnx)
def assemble(out_dir: str, keep_spec: bool) -> int:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from bitino.qonnx_profile import check_file

    sdir = os.path.join(out_dir, "spec")
    sp = json.load(open(os.path.join(sdir, "spec.json")))
    z = np.load(os.path.join(sdir, "spec.npz"))
    inits = [numpy_helper.from_array(z[f"i{j}"], name) for j, name in enumerate(sp["init_names"])]
    nodes = []
    for nd in sp["nodes"]:
        n = helper.make_node(nd["op"], nd["inputs"], nd["outputs"], name=nd["name"], domain=nd["domain"] or None, **nd["attrs"])
        for k, v in nd["meta"].items():
            n.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
        nodes.append(n)
    ins = [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in sp["inputs"]]
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in sp["outputs"]]
    graph = helper.make_graph(nodes, sp["meta"]["bitino.model_id"], ins, outs, inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid(QD, 1)],
                          producer_name="bitino tools/export_ir_qonnx.py", producer_version=PROFILE)
    m.ir_version = IR_VERSION
    meta = dict(sp["meta"])
    meta["bitino.exporter"] += f"; onnx {onnx.__version__}"
    for k, v in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    path = os.path.join(out_dir, "model.onnx")
    onnx.save(m, path)
    del m, graph, nodes, inits, z
    rep = check_file(path)
    prof = rep.to_dict() if hasattr(rep, "to_dict") else {"passed": rep.passed}
    if not prof.get("passed"):
        print(json.dumps(prof, indent=1)[:4000], file=sys.stderr)
        print("ERROR: the written file does not pass check-qonnx (profile bitino-qonnx/1)", file=sys.stderr)
        return 5
    exp = sp["export"]
    exp.update({"onnx_file": "model.onnx", "onnx_sha256": sha256_file(path), "onnx_bytes": os.path.getsize(path),
                "onnx_producer": f"bitino tools/export_ir_qonnx.py {PROFILE}", "opsets": [["ai.onnx", OPSET], [QD, 1]],
                "onnx_nodes": len(sp["nodes"]), "onnx": onnx.__version__,
                "profile_check": {k: prof.get(k) for k in ("profile", "passed", "summary")} | {
                    "warnings": [f for f in prof.get("findings", []) if f.get("level") == "warning"]}})
    json.dump(exp, open(os.path.join(out_dir, "export.json"), "w"), indent=1)
    if not keep_spec:
        shutil.rmtree(sdir)
    print(f"exported {exp['model_id']} {exp['file']} -> {path} ({exp['onnx_bytes']} bytes, sha256 {exp['onnx_sha256'][:16]}), "
          f"{exp['onnx_nodes']} nodes; check-qonnx {prof.get('summary')}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id")
    ap.add_argument("--out-dir")
    ap.add_argument("--eval-dir", help="check set (default: <cache>/data/<recipe eval dir>)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--qonnx-python", help="interpreter of stage 2 (default <cache>/venv-qonnx/bin/python)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-source-logits", action="store_true", help="skip the source framework's logits (identity runs)")
    ap.add_argument("--only-file", action="append", help="published file to write (repeatable; default all)")
    ap.add_argument("--keep-spec", action="store_true")
    ap.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/tmp"), help="scratch directory (Theano compile cache: no spaces)")
    ap.add_argument("--assemble", metavar="DIR", help="stage 2 only (venv-qonnx): assemble DIR/spec into DIR/model.onnx")
    ap.add_argument("--list", action="store_true", help="print the step-4 model ids and their stage-1 environments")
    args = ap.parse_args()
    if args.list:
        for m in step4_models():
            print(m, VENVS[family_of(m)])
        return 0
    if args.assemble:
        return assemble(args.assemble, args.keep_spec)
    if not (args.model_id and args.out_dir):
        ap.error("--model-id and --out-dir are required")
    return stage1(args)


if __name__ == "__main__":
    sys.exit(main())
