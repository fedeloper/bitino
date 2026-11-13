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
recorded data_sha256; "needs-manual-step" = verified with a file the script cannot download (passed with --file);
"broken" = cannot be run today (registry/recipes_verified.json says why); "reconstructed" = assembled from the
records above, not run as a recipe.
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
    ("ftl-deepfake-*", "eval128_cifake_ftl", "ftl_rgb01", "--num-classes 2"),
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
    "bhvit": {"python": "3.11", "requirements": "envs/bhvit-requirements.txt",
              "note": "PyTorch 2.5.1 CPU + transformers 4.39.2 + timm 1.0.27 (the BHViT model code needs transformers < 4.40)"},
    "ftl": {"python": "3.11", "requirements": "envs/ftl-requirements.txt",
            "note": "PyTorch 2.9.0 CPU + OpenCV 4.12 + scikit-image 0.25.2 + timm 0.6.11 + einops + torchinfo (the authors' "
                    "Faster Than Lies model.py and the BNext definition import them)"},
    "nncf": {"python": "3.10", "requirements": "envs/nncf-requirements.txt", "note": "nncf 2.9.0 + torch 2.1.2"},
    "qonnx": {"python": "3.12", "requirements": "envs/qonnx-freeze.txt", "note": "qonnx 1.0.0 + onnx + onnxruntime"},
    "theano27": {"python": "2.7", "requirements": "envs/theano27-conda-export.txt",
                 "note": "Python 2.7 + Theano 1.0.4 + Lasagne (BNN-PYNQ reference)"},
    "mindspore": {"python": "3.11", "requirements": None, "note": "MindSpore CPU (FDA-BNN official logits step)"},
    "tensorpack": {"python": "3.11", "requirements": "envs/tensorpack-requirements.txt",
                   "note": "TensorFlow 2.15.1 (tf.compat.v1) + tensorpack 0.11 (official logits step)"},
}

TOOL_ENV = {
    "tools/convert_larq_zoo.py": "larq", "tools/convert_keras_embedding.py": "larq", "tools/convert_omz.py": "openvino",
    "tools/convert_brevitas.py": "brevitas", "tools/convert_nncf.py": "nncf", "tools/convert_finn_onnx.py": "qonnx",
    "tools/convert_dabnn.py": "qonnx", "tools/convert_bnn_pynq_npz.py": "base", "tools/convert_bhvit.py": "bhvit",
    "tools/convert_ftl_deepfake.py": "ftl",
}

BNEXT_COMMIT = "dfcf347a30e3bc08606b8cad2c8d4a329d5a5b28"   # hpi-xnor/BNext, the commit every BNext conversion used
MINDSPORE_MODELS_COMMIT = "1139ddcdcfcc2bda71dbe54b1b27da2027a638ea"   # gitee mindspore/models master (2025-12-19)
BSRC_COMMIT = "62bbcf29cc95fbb7e6a8bc9d705d07abe8831d9e"       # ilyazhara/BSRC-2023 (weights and code)
# Weights inside a repository without a recorded source commit: the commit that last changed the file (= main on
# 2026-11-12).
BRANCH_PINS = {"Aliasgharshinwari/finn-cnv-w1a1-pynq-z1": "c155dac9ac3b9f25cd48dd14a6de5b738b517bea"}
LASAGNE_COMMIT ="5d3c63cb315c50b1cbd27a6bc8664b406f34dd99"     # Lasagne master (master.zip used by the conversions)
BNN_PYNQ_COMMIT = "d39c1c75403db571f54dd0f0cf27ae5104ac968e"    # Xilinx/BNN-PYNQ master

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


# The authors' public Google Drive folders whose registry entry names only the folder: the checkpoint's file id
# inside it (listed 2026-11-12 with drive.google.com/embeddedfolderview, no sign-in; the downloaded file's SHA-256
# is the registry's).
DRIVE_FOLDER_FILES = {
    "1dABYp66jArFjTQxodpDZ88KCLdGSraY0": "1_KN-BXE11CASzAS_WZ7NtO2Wj6nvRtj-",   # RBNN resnet18_1w1a (CIFAR-10)
    "14jxaB8SwohsxYXJdjQJERw80IQVV7-Ll": "1sr7WD_M6fopG-jX9IcZwJiCwasFiDH9V",   # RBNN resnet18 ImageNet
    "1aykx5Kar2Y-8mYf13LqvThdTuSlkgtmQ": "1j8rstvTy6jZBox8nVWBsV70PQKEbsq-B",   # RBNN resnet20_1w1a
    "18IxJlYp4i032188hfE3ETJOKH4WxgffF": "1a3xyNUgNYNmgkGGJc3zJ-4NDTxn9oMpZ",   # RBNN resnet34 ImageNet
    "1ZHRLyQ4ZkrhCPT2fITKq47ZLwSlMZWFx": "1leIisPklKZIPPsfZsdXp-z3cilw42mxM",   # RBONN
    "1fv98WNR503iFRwIHrhFUnQT_bQSwpWFN": "1tbUnv0yzPVR_9BVcvSmDJb4_mlPex_bJ",   # ReCU ResNet-18 ImageNet
    "1X3DspRZPKum-dH4Z52M5jt5zNaXv-o1X": "1IUu1LP7Qpqg6przG4Y89RFp-Xz7_b393",   # ReCU ResNet-20
    "1CmApN8sgGuM3zkQbCfsAUqv0w5flHk-p": "14agUliel9WdPEwWPlEMkIDw3n9YWg2AO",   # ReCU ResNet-34 ImageNet
    "15pwL5UeJGFHNwHNFh7Yl6dFgAX9QvxE5": "1SUK8CyN7jzG5jKPhR98ayD9G7DqNVn-x",   # SiMaN resnet18 ImageNet
    "1vhl1Q9ulTfqMy27Gn5lFIMgop24UIRh8": "1xfbIE2jEVv-M8-EUyY0-azMAf8DlZmT-",   # SiMaN resnet34 ImageNet
}
# Other files of a release the converter reads next to the weights (authors' Drive folder of BiBERT,
# 1xEEIynvsYuqqG6wRlMhSySUusZWoR1FL, models/<task>/config.json: num_labels). SHA-256 = the files the published
# conversions read.
EXTRA_DOWNLOADS = {
    "bibert-bert-base-mnli": [("config", "1MeobahnU3XTxBOF0bPf6aCbH2ClVedDK", "config.json",
                               "9071b6497ed732ab01f4dde5988a1e74d645325e26ab2b27e9e92c61884eadf6")],
    "bibert-bert-base-mrpc": [("config", "1sYrWv8fj99fZdTkp8WxpX9jwBweqj29T", "config.json",
                               "25d0f2b404e935b886cb1b8fa94aa2fb5738c545e3e5f3342a36f44f7b0c9b1e")],
    "bibert-bert-base-qqp": [("config", "1NvaoTqEKZngorys6Cn-HrW1F0ZQyvjfI", "config.json",
                              "f397195ba2fedc8376ed798659637b492189137e31097e34b5c2ea83acb4fbce")],
    "bibert-bert-base-sst2": [("config", "11d7PABeK8-NI2AXfqFReSKpthZO4SVMf", "config.json",
                               "1cd92e9b33bc6371b93dc40ffca9acf35d3a0479f1fc3c2a02f01aa711129170")],
    # SA-BNN: the model definition next to the weights in the authors' Drive release (folder
    # 1tlyEXKhE6ViFtMWB7iMAKZpoGXHEaeLs, "Training Techniques/resnet18_61.696/birealnet.py"); saved as .txt, its
    # SHA-256 pins the exact file the published conversion executed.
    "sa-bnn-resnet18-imagenet": [("code", "1VFtvDp8znUOAyaCaldz8JLBqaXFr7hzo", "birealnet.py.txt",
                                  "7570f986dabb0538aca4c6c676d48d5593c384d16978e63ebfe776e826b4ab4e")],
}
_DRIVE_FILE = re.compile(r"(?:[?&]id=|/file/d/|file id |model_best\.pth\.tar id )([\w-]{20,})")
_DRIVE_FOLDER = re.compile(r"/drive/folders/([\w-]{20,})")


def drive_file_url(text: str) -> tuple[str | None, str | None]:
    """(direct uc?export=download URL, Drive file id) for a registry URL text naming a Drive file or folder."""
    m = _DRIVE_FILE.search(text)
    fid = m.group(1) if m else None
    if not fid:
        f = _DRIVE_FOLDER.search(text)
        fid = DRIVE_FOLDER_FILES.get(f.group(1)) if f else None
    return (f"https://drive.google.com/uc?export=download&id={fid}", fid) if fid else (None, None)


def drive_manual(page: str, file: str, name: str = "weights") -> str:
    return (f"Google Drive (authors' upload): if the scripted download is refused (daily download quota, sign-in "
            f"page), open {page} in a browser, download {file} and pass --file {name}=PATH (the SHA-256 is still checked)")


def weights_download(ck: dict, name="weights") -> dict:
    raw = ck.get("url", "")
    url = raw.split(" ")[0]
    d = {"name": name, "url": url, "sha256": (ck.get("sha256") or "").lower() or None,
         "file": ck.get("filename") or os.path.basename(url.split("?")[0]), "size_bytes": ck.get("size_bytes")}
    if d["file"] and " " in d["file"]:
        d["file"] = d["file"].split(" ")[0]
    if raw != url:
        d["page"] = raw   # the registry's description (folder, other files of the release)
    if "drive.google" in raw or "drive.usercontent" in raw:
        direct, fid = drive_file_url(raw)
        if direct:
            d["url"] = direct
            d["page"] = raw if raw.startswith("https://drive.google.com/drive/folders/") or " " in raw else \
                f"https://drive.google.com/file/d/{fid}/view"
        d["file"] = os.path.basename(d["file"])
        d["manual"] = drive_manual(d.get("page") or raw, d["file"], name)
    if "pan.baidu" in raw:
        d["manual"] = "Baidu Pan: manual download only; pass --file weights=PATH"
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)$", d["url"])
    if m:   # a GitHub file page serves HTML; /raw/ serves the bytes
        d["url"] = f"https://github.com/{m.group(1)}/{m.group(2)}/raw/{m.group(3)}"
    m = re.search(r"\(zip, ([\d,]+) B, sha256 ([0-9a-f]{64}); member (\S+?)\)", raw)
    if m:   # one member of a zip release: fetch downloads the archive (SHA-256) and extracts the member (SHA-256)
        d.update(archive_file=os.path.basename(d["url"].split("?")[0]), archive_sha256=m.group(2),
                 archive_bytes=int(m.group(1).replace(",", "")), member=m.group(3))
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
    for name, fid, fn, sha in EXTRA_DOWNLOADS.get(mid, []):
        page = f"https://drive.google.com/file/d/{fid}/view"
        downloads.append({"name": name, "url": f"https://drive.google.com/uc?export=download&id={fid}", "sha256": sha,
                          "file": fn, "page": page, "manual": drive_manual(page, fn, name)})
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
        if short == "convert_bsrc.py":   # the checkpoints live in the repo: its commit pins the code too
            sources = [{"name": "repo", **archive_source("https://github.com/ilyazhara/BSRC-2023", BSRC_COMMIT)}]
    elif short == "convert_ftl_deepfake.py":
        # Faster Than Lies: the authors' model.py + the BNext definition; BNext4DFR.__init__ loads the ImageNet BNext
        # checkpoint of the same size (then overwritten by the detector checkpoint)
        size = {"t": "tiny", "s": "small", "m": "middle", "l": "large"}[mid.split("-")[3]]
        bn = reg.get(f"bnext-{size}", {}).get("checkpoint", {})
        downloads.append(weights_download(bn, "bnext_weights"))
        sources.append({"name": "bnext", **archive_source("https://github.com/hpi-xnor/BNext", BNEXT_COMMIT)})
        args += ["--repo", "{repo}", "--bnext-repo", "{bnext}", "--bnext-weights", "{download:bnext_weights}", "--weights", "{weights}"]
        notes.append("the eval set (CIFAKE test subset, tools/eval_sets/ftl_deepfake.py) is not redistributed; with the "
                     "synthetic eval set the conversion checks run on random inputs")
        if size == "middle":
            notes.append("memory: the conversion checks fork one numpy-executor worker per thread (--threads); BNext-M "
                         "needed 6.2 GB at --threads 1 and more than 14 GB at --threads 4 (verification runs used --threads 1)")
    elif short == "convert_sabnn.py":
        args += ["--repo", "{downloads}", "--weights", "{weights}"]
        notes.append("the model definition birealnet.py.txt is part of the same Google Drive release as the weights")
    elif short == "convert_fdabnn.py":
        # The converter reads the MindSpore checkpoint without MindSpore (protobuf) and hashes the official model
        # files (src/quan.py, src/resnet.py of mindspore/models research/cv/FDA-BNN, pinned by commit + SHA-256;
        # they are not executed). The official MindSpore logits are an optional cross-check (not in the recipe).
        for fn, sha in (("quan.py", "f7f9d9048cb5ae6db5100884c11f2a90bba36dc61b417f12843ef5eb7d9e0aa6"),
                        ("resnet.py", "574989caf4c6d0470b81a24e58595d952179e6a134a441749383fca0691c187f")):
            downloads.append({"name": fn.removesuffix(".py"), "file": fn, "sha256": sha,
                              "url": f"https://gitee.com/mindspore/models/raw/{MINDSPORE_MODELS_COMMIT}/research/cv/FDA-BNN/src/{fn}"})
        args += ["--repo", "{downloads}", "--weights", "{weights}"]
        notes.append("optional cross-check with the official MindSpore network: tools/fdabnn_mindspore_reference.py in a "
                     "MindSpore 2.7 CPU env, then pass its logits with --mindspore-logits (not needed for the converted file)")
    elif short == "convert_brevitas.py":
        args += ["--weights", "{weights}"]
        notes.append("the converter also reads the Brevitas eval log next to the checkpoint when the model has one")
    elif short == "convert_finn_onnx.py":
        args += ["--zip", "{weights}"]
    elif short == "convert_dabnn.py":
        raw = ck.get("url", "")
        dab_url, dab_id = drive_file_url(raw.split("(ONNX);", 1)[1]) if "(ONNX);" in raw else (raw.replace(".onnx", ".dab"), None)
        downloads[0]["page"] = f"https://drive.google.com/file/d/{drive_file_url(raw)[1]}/view"
        downloads[0]["manual"] = drive_manual(downloads[0]["page"], downloads[0]["file"])
        dab_page = f"https://drive.google.com/file/d/{dab_id}/view" if dab_id else None
        downloads.append({"name": "dab", "url": dab_url, "sha256": src.get("dab_sha256"), "file": src.get("dab_file"),
                          **({"page": dab_page, "manual": drive_manual(dab_page, src.get("dab_file"), "dab")} if dab_id else {})})
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
        # BNN-PYNQ master and Lasagne master at the commits the published conversions used (both unchanged since)
        sources = [{"name": "lasagne", **archive_source("https://github.com/Lasagne/Lasagne", LASAGNE_COMMIT)},
                   {"name": "bnnpynq", **archive_source("https://github.com/Xilinx/BNN-PYNQ", BNN_PYNQ_COMMIT,
                                                        "bnn/src/training")}]
        for d in downloads:
            d["url"] = d["url"].replace("/BNN-PYNQ/master/", f"/BNN-PYNQ/{BNN_PYNQ_COMMIT}/")
        args += ["--npz", "{weights}", "--theano-python", "{python:theano27}", "--lasagne-dir", "{lasagne}",
                 "--bnnpynq-dir", "{bnnpynq}"]
        notes.append("the reference runs the official Theano/Lasagne network in Python 2.7 (env theano27, "
                     "envs/theano27-conda-export.txt) with the BNN-PYNQ training code and Lasagne")
    elif short == "convert_tensorpack_dorefa.py":
        args += ["--npz", "{weights}", "--repo", "{repo}", "--tensorpack-logits", "{work}/tensorpack_logits.f32"]
        steps.append({"env": "tensorpack", "tool": "tools/tensorpack_dorefa_reference.py",
                      "args": ["--repo", "{repo}", "--npz", "{weights}", "--eval-dir", "{eval_dir}", "--preprocess", pre,
                               "--out", "{work}/tensorpack_logits.f32", "--threads", "{threads}"]})
        notes.append("first runs the official tensorpack AlexNet-DoReFa graph (TensorFlow, env tensorpack) for the fp32 "
                     "reference logits the converter compares against")
    else:
        args += ["--repo", "{repo}", "--weights", "{weights}"]
        notes.append(f"converter {short}: arguments reconstructed generically")
    # Converter options recorded in the published file (BiBERT / Bi-ViT GELU handling)
    if short in ("convert_bibert.py", "convert_bivit.py"):
        if str(man.get("gelu") or "").startswith("fp32 zero set"):
            args += ["--gelu-fp32-zero-set"]
        if (conv.get("passes") or {}).get("drop_gelu_before_sign"):
            args += ["--drop-gelu-before-sign"]
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
    # Converters whose constants depend on the CPU thread count (fp32 reductions in the source's weight quantizers,
    # recorded as reference.torch_threads) run with the recorded count, not `fetch --threads`.
    try:
        ref_threads = (load(os.path.join(adir, mid, "conversion_report.json")).get("reference") or {}).get("torch_threads")
    except (OSError, ValueError):
        ref_threads = None
    for flag, ph in (("--jobs", "{threads}"), ("--torch-threads", "{threads}"), ("--threads", "{threads}"),
                     ("--ort-threads", "{threads}")):
        if f'"{flag}"' in src_text:
            if flag == "--torch-threads" and ref_threads:
                ph = str(ref_threads)
                notes.append(f"runs the source model with {ref_threads} torch threads as recorded: its fp32 weight-quantizer "
                             f"reductions differ by an ulp with the thread count")
            args += [flag, ph]
    steps.append({"env": env, "tool": tool, "args": args})
    # Published files rewritten after the conversion by tools/derive_variant.py (recorded in conversion.derived):
    # the same IR passes, in place, on the converter's output.
    for dv in conv.get("derived") or []:
        if dv.get("tool") != "tools/derive_variant.py":
            continue
        dargs = []
        for p in dv.get("passes", {}):
            if p not in ("keep_zero_below", "prune_unused_constants"):
                dargs += ["--pass", p]
        if "keep_zero_below" in dv.get("passes", {}):
            dargs += ["--keep-zero-below", repr(float(dv["passes"]["keep_zero_below"]))]
        variant = (man.get("variant") or "").rsplit(", ", 1)[-1] if dv.get("passes", {}).get("keep_zero_below") is not None else None
        src_file = dv.get("source_file", "model.bitino")
        steps.append({"env": "base", "tool": "tools/derive_variant.py",
                      "args": dargs + ["--in", "{out_dir}/" + src_file, "--out", "{out_dir}/" + src_file]
                      + (["--variant", variant] if variant else [])})
        notes.append(f"the published file is the converter's {src_file} rewritten by tools/derive_variant.py "
                     f"({', '.join(k for k in dv.get('passes', {}) if k != 'prune_unused_constants')})")
    if any("{repo}" in a for st_ in steps for a in st_["args"]) and not sources:
        notes.append("source tree not pinned in the records (no commit); clone " + (rm.get("repo_url") or "the upstream repo")
                     + " and pass --source repo=PATH")
    # A file inside the pinned source repo, linked on a branch (main/master): pin it to the same commit.
    for d in downloads:
        m = re.match(r"https://(?:github\.com/([^/]+)/([^/]+)/raw|raw\.githubusercontent\.com/([^/]+)/([^/]+))/(main|master)/(.+)$",
                     d.get("url") or "")
        if not m:
            continue
        owner, name = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        pin = next((s["commit"] for s in sources if s.get("kind") == "github-archive"
                    and s["repo"].lower() == f"https://github.com/{owner}/{name}".lower()), None) or BRANCH_PINS.get(f"{owner}/{name}")
        if pin:
            d["url"] = f"https://github.com/{owner}/{name}/raw/{pin}/{m.group(6)}"
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
        if (q.get("residual_code") or {}).get("code") == "lin8":
            # int8_full v2 (lin8 residual code, promoted 2026-11-13): the quantizer arguments recorded at build time
            # (reports/int8_full_v2/build/<m>.json; --stats is a local cache, --n is set above)
            br = os.path.join(ROOT, "reports", "int8_full_v2", "build", f"{mid}.json")
            bargs = load(br).get("args", []) if os.path.exists(br) else []
            flags = [f for i, f in enumerate(flags) if f != "--residual-code" and not (i and flags[i - 1] == "--residual-code")]
            skip = False
            for i, t in enumerate(bargs):
                if skip:
                    skip = False
                    continue
                if t in ("--stats", "--n"):
                    skip = True
                    continue
                if t.startswith("--") and t in flags:
                    nxt = bargs[i + 1] if i + 1 < len(bargs) and not bargs[i + 1].startswith("--") else None
                    skip = nxt is not None
                    continue
                flags.append(t)
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
    vdoc = load(vpath) if os.path.exists(vpath) else {}
    verified = vdoc.get("verified", {})
    manual = vdoc.get("needs_manual_step", {})
    broken = vdoc.get("broken", {})
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
                if i["file"] in [f.replace(".bitino", ".bitino") for f in verified[mid].get("int8_verified", [])]:
                    i["state"] = "verified"
        elif mid in manual:
            r["state"] = "needs-manual-step"
            r["manual_step"] = manual[mid]
        elif mid in broken:
            r["state"] = "broken"
            r["broken"] = broken[mid]
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
                      "needs-manual-step": "the recipe works (the converted data section matched) once a file the script "
                                           "cannot download is supplied with --file NAME=PATH; see manual_step",
                      "broken": "cannot be run today (upstream file gone or changed); see broken",
                      "reconstructed": "assembled from the conversion records; not run as a recipe yet"},
           "envs": ENVS, "counts": counts, "recipes": recipes}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
