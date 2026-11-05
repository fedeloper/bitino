#!/usr/bin/env python3
"""Builds registry/recipes.json: one fetch-and-convert recipe per converted model, as data.

  python3 tools/build_recipes.py --artifacts <dir holding artifacts/<model>/> [--eval-root <dir with eval-set manifests>]

A recipe says where the ORIGINAL weights (and the original model code, pinned to a commit) are downloaded
from, their SHA-256, which reference environment the converter needs, the converter command, the expected
data_sha256 of the converted file, and how the int8 file was quantized. `bitino fetch <model-id>`
(python/bitino/fetch.py) executes it. Everything is reconstructed from records that already exist:

  registry/models.json                  checkpoint URL, SHA-256, file name, licences
  <artifacts>/<model>/model*.bitino    manifest (converter tool, source repo + commit, checkpoint file + hash,
                                        input shape, preprocessing) and data_sha256 of each shipped file
  <artifacts>/<model>/*.report.json     quantizer options / calibration set of the int8 files
  reports/int8_full_all/summary.json    quantizer arguments of accepted int8 variants
  scripts/eval_sets.sh (ported below)   eval set + preprocessing key per model
  registry/recipes_verified.json        log of recipes run end to end (state "verified")

Recipe state: "verified" = run end to end with `fetch` and the converted file's data section matched the
recorded data_sha256; "reconstructed" = assembled from the records above, not run as a recipe.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import glob
import json
import os
import re
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# scripts/eval_sets.sh eval_of(), as data: (glob, eval dir name, preprocessing key, extra validate_model.py args)
EVAL_OF = [
    ("ab-bnn-cifar10-*", "eval128_cifar10_norm", "cifar10_imagenet_norm", "--num-classes 10"),
    ("ab-bnn-cifar100-*", "eval128_cifar100_norm", "cifar100_imagenet_norm", "--num-classes 100"),
    ("ab-bnn-*", "eval128", "torchvision_imagenet", ""),
    ("bitorch-*", "eval128_bitorch", "bitorch_imagenet", "--flip-rule feed"),
    ("rbonn-*", "eval128", "torchvision_imagenet", ""),
    ("sa-bnn-resnet18-imagenet", "eval128", "torchvision_imagenet", ""),
    ("ida-det-bires18-imagenet-backbone", "eval128", "torchvision_imagenet", ""),
    ("*-imagenet", "eval128", "torchvision_imagenet", ""),   # recu/rbnn/siman/biper/snn/bnas ImageNet
    ("recu-*-cifar10", "eval128_cifar10_norm", "cifar10_imagenet_norm", "--num-classes 10"),
    ("rbnn-*-cifar10", "eval128_cifar10_norm", "cifar10_imagenet_norm", "--num-classes 10"),
    ("siman-*-cifar10", "eval128_cifar10_norm", "cifar10_imagenet_norm", "--num-classes 10"),
    ("biper-*-cifar10", "eval128_cifar10_norm", "cifar10_imagenet_norm", "--num-classes 10"),
    ("adabin-resnet20-cifar10", "eval128_cifar10_norm", "cifar10_imagenet_norm", "--num-classes 10"),
    ("fda-bnn-resnet20-cifar10-mindspore", "eval128_cifar10_norm", "cifar10_imagenet_norm", "--num-classes 10"),
    ("tensorpack-dorefa-alexnet-*", "eval128_tensorpack_bgr", "tensorpack_bgr", ""),
    ("bnas-a-cifar10", "eval128_cifar10_darts", "cifar10_darts_norm", "--num-classes 10"),
    ("bimatting-bimobilenet-backbone", "eval128", "torchvision_imagenet", ""),
    ("burn-reacta-stage2", "eval128", "torchvision_imagenet", "--output-kind embedding --embedding-cos 0.99999"),
    ("s2bnn-reacta-distill-only", "eval128", "torchvision_imagenet", "--output-kind embedding --embedding-cos 0.99999"),
    ("shuaiz-bireal20-cifar10", "eval128_cifar10_norm", "cifar10_bireal_norm", "--num-classes 10"),
    ("bnext-*", "eval128_timm_bicubic_crop095", "timm_bicubic_crop095", ""),
    ("bhvit-*", "eval128_bhvit", "bhvit", ""),
    ("bibert-bert-base-mrpc", "glue/mrpc/eval128", "bibert", "--numpy-logits numpy_logits.f32 --num-classes 2"),
    ("bibert-bert-base-mnli", "glue/mnli/eval128", "bibert", "--numpy-logits numpy_logits.f32 --num-classes 3"),
    ("bibert-bert-base-qqp", "glue/qqp/eval128", "bibert", "--numpy-logits numpy_logits.f32 --num-classes 2"),
    ("bi-vit-deit-*", "eval128_bhvit", "bhvit", ""),
    ("gsb-deit-s-cifar100-*", "eval128_cifar100_deit", "cifar100_deit", "--num-classes 100"),
    ("bibert-*", "sst2/eval128", "bibert", "--numpy-logits numpy_logits.f32 --num-classes 2"),
    ("brevitas-bnn-pynq-cnv-*", "eval128_cifar10_brevitas", "cifar10_totensor", "--num-classes 10 --require-exact-numpy"),
    ("aliasgharshinwari-finn-cnv-*", "eval128_cifar10_brevitas", "cifar10_totensor", "--num-classes 10 --require-exact-numpy"),
    ("brevitas-bnn-pynq-*", "eval128_mnist_brevitas", "mnist_totensor", "--num-classes 10 --require-exact-numpy"),
    ("dabnn-*", "eval128", "torchvision_imagenet", ""),
    ("nncf-resnet18-binarization-*", "eval128", "torchvision_imagenet", ""),
    ("finn-examples-bnn-pynq-cnv-*", "eval128_cifar10_finn", "cifar10_pixels", "--num-classes 10 --require-exact-numpy"),
    ("finn-examples-bnn-pynq-tfc-*", "eval128_mnist_finn", "mnist_pixels", "--num-classes 10 --require-exact-numpy"),
    ("finn-examples-cnv-w1a1-gtsrb", "eval128_gtsrb", "gtsrb_totensor", "--num-classes 44 --require-exact-numpy"),
    ("bnn-pynq-cnv-w1a1-cifar10", "eval128_cifar10_finn", "cifar10_bnnpynq", "--num-classes 10 --require-exact-numpy"),
    ("bnn-pynq-cnv-w1a1-gtsrb", "eval128_gtsrb", "gtsrb_bnnpynq", "--num-classes 44 --require-exact-numpy"),
    ("bnn-pynq-lfc-w1a1-mnist", "eval128_mnist_finn", "mnist_bnnpynq", "--num-classes 10 --require-exact-numpy"),
    ("bbcu-denoise", "eval68_cbsd68_sigma25_c96", "bbcu_denoise_s25", "--output-kind image"),
    ("bbcu-deblocking", "eval68_jpeg_y_q10q40_c96", "bbcu_jpeg_y", "--output-kind image"),
    ("bbcu-sr-*-x2", "eval64_sr_x2_lr48", "sr_x2_unit", "--output-kind image"),
    ("bbcu-sr-*-x4", "eval64_sr_x4_lr32", "sr_x4_unit", "--output-kind image"),
    ("bsrc2023-binary-sr-x2-*", "eval64_sr_x2_lr48", "sr_x2_255", "--output-kind image --image-metric bsrc_y"),
    ("bsrc2023-binary-sr-x4-*", "eval64_sr_x4_lr32", "sr_x4_255", "--output-kind image --image-metric bsrc_y"),
    ("omz-*", "eval128_omz_bgr", "omz_bgr", ""),
    ("larqface-*", "eval128_lfw", "insightface", "--output-kind embedding --embedding-cos 0.99999"),
    ("brillsson-*", "eval128_speechcommands", "brillsson_logmel", "--output-kind embedding --embedding-cos 0.99999"),
    ("*", "eval128", "larq_zoo", ""),
]

# Reference environments (requirement files are published under envs/).
ENVS = {
    "base": {"python": ">=3.10", "requirements": "envs/base-requirements.txt",
             "note": "numpy only (the runtime-side tooling)"},
    "torch": {"python": "3.12", "requirements": "envs/torch-requirements.txt",
              "note": "PyTorch CPU; the recorded conversions used torch 2.6.0 / torchvision 0.21.0"},
    "larq": {"python": "3.11", "requirements": "envs/larq-zoo-requirements.txt",
             "note": "TensorFlow 2.15.1 + larq 0.13.3 + larq-zoo 2.4.0"},
    "openvino": {"python": "3.12", "requirements": "envs/openvino-requirements.txt", "note": "OpenVINO runtime (fp32 reference)"},
    "brevitas": {"python": "3.12", "requirements": "envs/brevitas-requirements.txt", "note": "brevitas 0.11.0 + torch"},
    "nncf": {"python": "3.10", "requirements": "envs/nncf-requirements.txt", "note": "nncf 2.9.0 + torch 2.1.2"},
    "qonnx": {"python": "3.12", "requirements": "envs/qonnx-freeze.txt", "note": "qonnx 1.0.0 + onnx + onnxruntime"},
    "theano27": {"python": "2.7", "requirements": "envs/theano27-conda-export.txt",
                 "note": "Python 2.7 + Theano 1.0.4 + Lasagne (BNN-PYNQ reference)"},
    "mindspore": {"python": "3.11", "requirements": None, "note": "MindSpore CPU (FDA-BNN official logits step)"},
    "tensorpack": {"python": "3.11", "requirements": None, "note": "TensorFlow 1.x-compatible tensorpack (official logits step)"},
}

TOOL_ENV = {
    "tools/convert_larq_zoo.py": "larq", "tools/convert_keras_embedding.py": "larq", "tools/convert_omz.py": "openvino",
    "tools/convert_brevitas.py": "brevitas", "tools/convert_nncf.py": "nncf", "tools/convert_finn_onnx.py": "qonnx",
    "tools/convert_dabnn.py": "qonnx", "tools/convert_bnn_pynq_npz.py": "base",
}

GH = re.compile(r"https://(github\.com|gitee\.com)/([^/\s]+)/([^/\s(]+)")


def load(p):
    with open(p) as f:
        return json.load(f)


def header(path):
    with open(path, "rb") as f:
        h = f.read(64)
        if h[:8] != b"BFMODEL1":
            return None
        jo, js = struct.unpack("<QQ", h[16:32])
        f.seek(jo)
        return json.loads(f.read(js))


def eval_of(mid):
    for pat, d, pre, extra in EVAL_OF:
        if fnmatch.fnmatchcase(mid, pat):
            return d, pre, extra
    raise KeyError(mid)


def archive_source(repo: str, commit: str, subdir: str = "") -> dict | None:
    m = GH.match(repo or "")
    if not m or not re.fullmatch(r"[0-9a-f]{7,40}", commit or ""):
        return None
    host, owner, name = m.groups()
    name = name.removesuffix(".git")
    if host != "github.com":
        return None
    return {"kind": "github-archive", "url": f"https://github.com/{owner}/{name}/archive/{commit}.tar.gz",
            "repo": f"https://github.com/{owner}/{name}", "commit": commit, "subdir": subdir}


def weights_download(ck: dict, name="weights") -> dict:
    url = ck.get("url", "")
    d = {"name": name, "url": url, "sha256": (ck.get("sha256") or "").lower() or None,
         "file": ck.get("filename") or os.path.basename(url.split("?")[0]), "size_bytes": ck.get("size_bytes")}
    if "drive.google" in url or "drive.usercontent" in url:
        d["manual"] = "Google Drive: download in a browser if the scripted download is refused, then pass --file weights=PATH"
    if "pan.baidu" in url:
        d["manual"] = "Baidu Pan: manual download only; pass --file weights=PATH"
    if d["file"] and " " in d["file"]:
        d["file"] = d["file"].split(" ")[0]
    return d


def irnet_family(mid: str) -> str:
    p = mid.split("-")[0]
    if p == "shuaiz":
        return "shuaiz"
    return p + ("_cifar" if "cifar" in mid else "")


def build_recipe(mid, st, reg, man, adir, int8_args):
    conv = man.get("conversion", {})
    src = man.get("source", {})
    tool = conv.get("tool") or ""
    rid = st.get("registry_model_id", mid)
    rm = reg.get(rid, {})
    ck = rm.get("checkpoint", {})
    edir, pre, extra = eval_of(mid)
    env = TOOL_ENV.get(tool, "torch")
    notes = []
    downloads = [weights_download(ck)] if ck.get("url") else []
    if src.get("checkpoint_sha256") and downloads and downloads[0]["sha256"] and src["checkpoint_sha256"] != downloads[0]["sha256"]:
        notes.append("the converted file records checkpoint_sha256 " + src["checkpoint_sha256"]
                     + " (a file inside the registry download); the fetch checks the download's own SHA-256")
    sources = []
    s = archive_source(src.get("repo", ""), src.get("repo_commit", ""))
    if s:
        sources.append({"name": "repo", **s})
    args: list = ["--model-id", "{model_id}"]
    steps = []
    short = os.path.basename(tool)
    if short == "convert_larq_zoo.py":
        args += ["--zoo-class", src.get("class", "?"), "--weights", "{weights}", "--sha256", "{sha256}"]
        leg = src.get("legacy_source")
        if isinstance(leg, str):
            try:
                import ast
                leg = ast.literal_eval(leg)
            except Exception:
                leg = None
        if isinstance(leg, dict) and leg.get("commit"):
            sources.append({"name": "repo", **archive_source("https://github.com/larq/zoo", leg["commit"])})
            args += ["--zoo-module-file", "{repo}/" + leg.get("module_file", "larq_zoo/sota/quicknet.py")]
    elif short == "convert_keras_embedding.py":
        args += ["--family", src.get("family", "?"), "--weights", "{weights}", "--sha256", "{sha256}"]
        if src.get("zoo_class"):
            args += ["--zoo-class", src["zoo_class"]]
        args += ["--preprocess", pre]
        rc = src.get("source_repo_commit")
        s2 = archive_source(rm.get("repo_url", ""), rc or "")
        if s2:
            sources.append({"name": "repo", **s2})
            args += ["--source-repo", "{repo}"]
    elif short == "convert_omz.py":
        sub, prec, stem = {"omz-resnet18-xnor-binary-onnx-0001": ("omz-resnet18-xnor-binary-onnx-0001", "FP32-INT1", "resnet18-xnor-binary-onnx-0001"),
                           "omz-resnet18-xnor-binary-onnx-0001-fp16": ("omz-resnet18-xnor-binary-onnx-0001", "FP16-INT1", "resnet18-xnor-binary-onnx-0001"),
                           "omz-resnet50-binary-0001": ("omz-resnet50-binary-0001", "FP32-INT1", "resnet50-binary-0001"),
                           "omz-resnet50-binary-0001-fp16": ("omz-resnet50-binary-0001", "FP16-INT1", "resnet50-binary-0001")}[mid]
        base = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{stem}/{prec}/{stem}"
        downloads = [{"name": "bin", "url": base + ".bin", "sha256": src.get("checkpoint_sha256"), "file": f"{sub}/{prec}/{stem}.bin"},
                     {"name": "xml", "url": base + ".xml", "sha256": src.get("xml_sha256"), "file": f"{sub}/{prec}/{stem}.xml"}]
        args += ["--checkpoint-dir", "{downloads}"]
    elif short == "convert_abbnn.py":
        args += ["--repo", "{repo}", "--weights", "{weights}"]
    elif short in ("convert_bitorch.py",):
        rc = src.get("repo_commit", "")
        m = re.search(r"(bitorch-[\d.]+-py2\.py3-none-any\.whl) sha256:([0-9a-f]+)", rc)
        sources = [{"name": "repo", "kind": "pypi-wheel", "package": "bitorch", "version": "0.3.0",
                    "file": m.group(1) if m else "bitorch-0.3.0-py2.py3-none-any.whl",
                    "sha256_prefix": m.group(2) if m else None, "subdir": ""}]
        args += ["--repo", "{repo}", "--weights", "{weights}"]
    elif short == "convert_irnet.py":
        args = [irnet_family(mid)] + args + ["--repo", "{repo}", "--weights", "{weights}"]
    elif short in ("convert_bnext.py", "convert_rbonn.py", "convert_snn.py", "convert_idadet.py", "convert_bhvit.py",
                   "convert_bivit.py", "convert_bnas.py", "convert_reacta_bn.py", "convert_bbcu.py", "convert_bsrc.py",
                   "convert_adabin.py"):
        args += ["--repo", "{repo}", "--weights", "{weights}"]
        if short == "convert_bsrc.py":
            notes.append("BSRC-2023 code is not a git checkout in the records (source files are hashed in the manifest)")
    elif short == "convert_sabnn.py":
        args += ["--repo", "{downloads}", "--weights", "{weights}"]
        notes.append("the model definition birealnet.py.txt is part of the same Google Drive release as the weights")
    elif short == "convert_fdabnn.py":
        args += ["--repo", "{repo}", "--weights", "{weights}", "--mindspore-logits", "{out_dir}/mindspore_logits_eval.f32"]
        notes.append("needs the official MindSpore fp32 logits first (tools/fdabnn_mindspore_reference.py in a MindSpore env)")
    elif short == "convert_brevitas.py":
        args += ["--weights", "{weights}"]
        notes.append("the converter also reads the Brevitas eval log next to the checkpoint when the model has one")
    elif short == "convert_finn_onnx.py":
        args += ["--zip", "{weights}"]
    elif short == "convert_dabnn.py":
        downloads.append({"name": "dab", "url": ck.get("url", "").replace(".onnx", ".dab"), "sha256": src.get("dab_sha256"),
                          "file": src.get("dab_file")})
        args += ["--onnx", "{weights}", "--dab", "{download:dab}"]
    elif short == "convert_nncf.py":
        mode = "xnor" if "xnor" in mid else "dorefa"
        sources = [{"name": "repo", **archive_source("https://github.com/openvinotoolkit/nncf", "f4b73bf")}]
        sources[0]["commit_note"] = "tag v2.5.0 (abbreviated commit f4b73bf in the records)"
        steps.append({"env": "base", "tool": "tools/nncf_extract_state.py", "args": ["{weights}", "{work}/state.pt"]})
        args += ["--weights", "{weights}", "--state", "{work}/state.pt", "--config",
                 "{repo}/examples/torch/classification/configs/binarization/resnet18_imagenet_binarization_" + mode + ".json"]
    elif short == "convert_bibert.py":
        args += ["--repo", "{repo}", "--ckpt-dir", "{downloads}"]
        notes.append("the checkpoint is a directory (config + weights); download every file of the release folder")
    elif short == "convert_bnn_pynq_npz.py":
        args += ["--npz", "{weights}", "--theano-python", "{python:theano27}"]
        notes.append("the reference runs the official Theano/Lasagne network in Python 2.7 (env theano27) with the "
                     "BNN-PYNQ and Lasagne sources (--lasagne-dir, --bnnpynq-dir)")
    elif short == "convert_tensorpack_dorefa.py":
        args += ["--npz", "{weights}", "--repo", "{repo}", "--tensorpack-logits", "{work}/tensorpack_logits.f32"]
        steps.append({"env": "tensorpack", "tool": "tools/tensorpack_dorefa_reference.py", "args": ["see the tool's --help"]})
        notes.append("needs the official tensorpack fp32 logits first (tools/tensorpack_dorefa_reference.py)")
    else:
        args += ["--repo", "{repo}", "--weights", "{weights}"]
        notes.append(f"converter {short}: arguments reconstructed generically")
    args += ["--eval-dir", "{eval_dir}", "--out-dir", "{out_dir}"]
    # Bound the converter's parallelism by `fetch --threads` (worker processes / framework threads).
    try:
        src_text = open(os.path.join(ROOT, tool)).read()
        if short in ("convert_irnet.py", "convert_bitorch.py", "convert_bnext.py", "convert_rbonn.py", "convert_snn.py",
                     "convert_idadet.py", "convert_sabnn.py"):
            src_text = open(os.path.join(ROOT, "tools", "convert_torch_cnn.py")).read()
        elif short in ("convert_bbcu.py", "convert_bsrc.py"):
            src_text = open(os.path.join(ROOT, "tools", "convert_restoration.py")).read()
        elif short in ("convert_adabin.py", "convert_fdabnn.py"):
            src_text = open(os.path.join(ROOT, "tools", "convert_s12_cifar.py")).read()
    except OSError:
        src_text = ""
    for flag, ph in (("--jobs", "{threads}"), ("--torch-threads", "{threads}"), ("--threads", "{threads}"),
                     ("--ort-threads", "{threads}")):
        if f'"{flag}"' in src_text:
            args += [flag, ph]
    steps.append({"env": env, "tool": tool, "args": args})
    if any("{repo}" in a for st_ in steps for a in st_["args"]) and not sources:
        notes.append("source tree not pinned in the records (no commit); clone " + (rm.get("repo_url") or "the upstream repo")
                     + " and pass --source repo=PATH")
    expected = {}
    for fn in ("model.bitino", "model_int8_full.bitino", "model_int8_full_log2.bitino"):
        p = os.path.join(adir, mid, fn)
        if os.path.exists(p):
            h = header(p) or {}
            g = json.dumps({k: h.get(k) for k in ("tensors", "inputs", "nodes", "outputs")}, sort_keys=True, separators=(",", ":"))
            expected[fn] = {"data_sha256": h.get("data_sha256"), "graph_sha256": hashlib.sha256(g.encode()).hexdigest(),
                            "bytes": os.path.getsize(p)}
    int8 = []
    for fn in ("model_int8_full.bitino", "model_int8_full_log2.bitino"):
        p = os.path.join(adir, mid, fn)
        if not os.path.exists(p):
            continue
        q = (header(p) or {}).get("manifest", {}).get("quantization", {})
        cal = q.get("calibration") or {}
        opts = q.get("options") or {}
        flags = []
        if q.get("residual_code"):
            flags += ["--residual-code", q["residual_code"].get("code", "l8")]
        for k, flag in (("pc_stem", "--pc-stem"), ("concat_parts", "--concat-parts"), ("bin_clip_far", "--bin-clip-far"),
                        ("bin_relu_clip", "--bin-relu-clip"), ("pc_bin", "--pc-bin")):
            if opts.get(k):
                flags.append(flag)
        if "pool_grid_fix" not in opts:
            flags.append("--no-pool-grid-fix")
        rng = str(cal.get("range") or "")
        if rng.startswith("percentile"):
            flags += ["--percentile", rng.split()[1]]
        for extra_arg in int8_args.get((mid, fn), []):
            toks = extra_arg.split(" (default)")[0].replace("=", " ").split()
            if toks and toks[0].startswith("--") and toks[0] not in flags:
                flags += toks
        int8.append({"file": fn, "date": q.get("date"),
                     "calibration": {k: cal.get(k) for k in ("set", "split", "preprocessing", "n", "range", "tensor_sha256")
                                     if k in cal},
                     "calibration_sample_ids": "listed in the shipped file's manifest.quantization.calibration.sample_ids",
                     "options": opts,
                     "args": ["--model", "{out_dir}/model.bitino", "--calib-dir", "{calib_dir}", "--preprocess",
                              cal.get("preprocessing", pre), "--n", str(cal.get("n", 128)), "--out", "{out_dir}/" + fn, "--jobs", "{threads}"] + flags,
                     "state": "reconstructed",
                     "note": "the quantizer reads calibration images that are not redistributed (training split of the "
                             "dataset); identical bytes need the same samples (ids in the manifest) and the quantizer "
                             "version of the date above"})
    return {
        "model_id": mid, "registry_model_id": rid, "state": "reconstructed",
        "framework": src.get("framework"), "adapter": conv.get("adapter"), "converter": tool,
        "env": env, "downloads": downloads, "sources": sources, "steps": steps,
        "eval": {"dir": edir, "preprocess": pre, "validate_args": extra.split(),
                 "input_shape": (man.get("input") or {}).get("shape"),
                 "synthetic_ok": not any(k in extra for k in ("image", "numpy-logits")) and pre not in (
                     "bibert", "brillsson_logmel", "insightface")},
        "expected": expected, "int8": int8, "notes": notes,
        "licence": {"code": rm.get("license", {}).get("code"), "weights": rm.get("license", {}).get("weights")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--eval-root", help="directory holding the eval-set manifests (<eval-root>/<dir>/manifest.json); "
                                        "their preprocessing definitions are copied into registry/eval_sets.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "registry", "recipes.json"))
    a = ap.parse_args()
    adir = a.artifacts if os.path.basename(os.path.normpath(a.artifacts)) == "artifacts" else os.path.join(a.artifacts, "artifacts")
    reg = {m["model_id"]: m for m in load(os.path.join(ROOT, "registry", "models.json"))["models"]}
    status = load(os.path.join(ROOT, "registry", "bitino_status.json"))
    int8_args = {}
    sp = os.path.join(ROOT, "reports", "int8_full_all", "summary.json")
    if os.path.exists(sp):
        for mid, rows in load(sp).get("models", {}).items():
            for r in rows:
                if str(r.get("verdict", "")).startswith(("accepted", "fixed")) and r.get("file"):
                    int8_args[(mid, os.path.basename(r["file"]))] = r.get("args") or []
    vpath = os.path.join(ROOT, "registry", "recipes_verified.json")
    verified = load(vpath).get("verified", {}) if os.path.exists(vpath) else {}
    recipes = {}
    for mid, st in sorted(status.items()):
        if st.get("conversion_status") != "converted":
            continue
        p = os.path.join(adir, mid, "model.bitino")
        if not os.path.exists(p):
            print(f"WARNING: {mid}: no {p}", file=sys.stderr)
            continue
        r = build_recipe(mid, st, reg, (header(p) or {}).get("manifest", {}), adir, int8_args)
        if mid in verified:
            r["state"] = "verified"
            r["verified"] = verified[mid]
            for i in r["int8"]:
                if i["file"] in verified[mid].get("int8_verified", []):
                    i["state"] = "verified"
        recipes[mid] = r
    eval_sets = {}
    if a.eval_root:
        for r in recipes.values():
            d = r["eval"]["dir"]
            mp = os.path.join(a.eval_root, d, "manifest.json")
            if d in eval_sets or not os.path.exists(mp):
                continue
            m = load(mp)
            eval_sets[d] = {"name": m.get("name"), "source": m.get("source"), "limitations": m.get("limitations"),
                            "n_classes_covered": m.get("n_classes_covered"),
                            "preprocessing": {k: {kk: v.get(kk) for kk in ("shape", "dtype", "layout", "definition")}
                                              for k, v in (m.get("preprocessing") or {}).items()}}
        with open(os.path.join(ROOT, "registry", "eval_sets.json"), "w") as f:
            json.dump({"schema": "bitino.eval_sets.v1",
                       "note": "preprocessing definitions of the evaluation sets (no data: the images are not redistributed)",
                       "eval_sets": eval_sets}, f, indent=1, ensure_ascii=False)
            f.write("\n")
    counts = {}
    for r in recipes.values():
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    doc = {"schema": "bitino.recipes.v1", "generator": "tools/build_recipes.py",
           "states": {"verified": "run end to end with `fetch`; the converted data section matched expected.data_sha256",
                      "reconstructed": "assembled from the conversion records; not run as a recipe yet"},
           "envs": ENVS, "counts": counts, "recipes": recipes}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
