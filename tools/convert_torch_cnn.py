#!/usr/bin/env python3
"""Shared converter for PyTorch binary CNNs: BNext (hpi-xnor/BNext), RBONN (SteveTsui/RBONN), the
SNN ReActNet-ResNet-18 (yikaiw/SNN), the IR-Net family (ReCU, RBNN, SiMaN, BiPer), the bitorch model hub
(ResNetE-18, BinaryDenseNet-28, MeliusNet-22), SA-BNN ResNet-18 (anonymous Drive birealnet.py) and the
IDa-Det BiRes-18 ImageNet backbone (SteveTsui/IDa-Det).

Entry points: tools/convert_bnext.py, tools/convert_rbonn.py, tools/convert_snn.py,
tools/convert_irnet.py, tools/convert_bitorch.py, tools/convert_sabnn.py and tools/convert_idadet.py
(thin wrappers selecting the family). Runs in a plain PyTorch environment (CPU; the base conda env has torch 2.6, timm,
einops and torchinfo, which the official sources import). The official model definition is
imported from a clone of the official repo (read only, no bytecode written into it); the
checkpoint is loaded with torch.load(weights_only=True) and load_state_dict(strict=True).

  python3 tools/convert_bnext.py --model-id bnext-tiny \
      --repo .cache/repos/BNext --weights .cache/checkpoints/bnext-tiny/checkpoint.pth.tar \
      --eval-dir .cache/data/eval128_timm_bicubic_crop095 --out-dir artifacts/bnext-tiny
  python3 tools/convert_rbonn.py --model-id rbonn-reactnet-resnet18-imagenet \
      --repo .cache/repos/SteveTsui_RBONN \
      --weights .cache/checkpoints/rbonn-reactnet-resnet18-imagenet/RBONN/model_best.pth.tar \
      --eval-dir .cache/data/eval128 --out-dir artifacts/rbonn-reactnet-resnet18-imagenet
  python3 tools/convert_irnet.py recu --model-id recu-resnet18-imagenet --repo .cache/repos/yuanchunyu_ReCU \
      --weights .cache/checkpoints/recu-resnet18-imagenet/ResNet-18_finetune_61.20/model_best.pth.tar \
      --eval-dir .cache/data/eval128 --out-dir artifacts/recu-resnet18-imagenet
  python3 -I tools/convert_sabnn.py --model-id sa-bnn-resnet18-imagenet \
      --repo .cache/checkpoints/sa-bnn-resnet18-imagenet \
      --weights .cache/checkpoints/sa-bnn-resnet18-imagenet/resnet18_61.696.tar \
      --eval-dir .cache/data/eval128 --out-dir artifacts/sa-bnn-resnet18-imagenet
  python3 -I tools/convert_idadet.py --model-id ida-det-bires18-imagenet-backbone --repo .cache/repos/IDa-Det \
      --weights .cache/checkpoints/ida-det-faster-rcnn-bir18-voc/bires18.pth \
      --eval-dir .cache/data/eval128 --out-dir artifacts/ida-det-bires18-imagenet-backbone

IR-Net family and SNN-0.67-bit: the binary weights are a function of the parameters (centering,
rotation, sin, pattern decoding). The IR holds the weights the *fp32* source computes. Redoing that
function in float64 can change a few signs (reported as weight_derivation.fp64_vs_fp32_differences),
so the primary float64 reference (reference_logits_f64.f64) runs the official model in float64 with
the fp32-derived binary weights substituted at the F.conv2d call of each binary conv (in-process
proxy for the module-global ``F``; the clone is not modified). model.double() as is is reported
as reference_f64_naive.

Produces in --out-dir:
  model.bitino / model_unfused.bitino                      input_binarize "ge" (deployable, 0 -> +1)
  model_sign_zero.bitino / model_sign_zero_unfused.bitino  input_binarize "sign_zero" (exact torch.sign,
                                                             reference kernels only)
  reference_logits.f32 / reference_logits_f64.f64            official PyTorch model, float32 and float64
  numpy_logits_<variant>.f32                                 numpy executor (float32) per variant
  conversion_report.json                                     Level-3 comparison, exact-zero counts,
                                                             float64-constants semantic check
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import torch_bitorch, torch_bnext, torch_irnet, torch_rbonn, torch_sabnn  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

FAMILIES = {
    "bnext": {
        "archs": {"bnext-tiny": "tiny", "bnext-small": "small", "bnext-middle": "middle", "bnext-large": "large"},
        "preprocess": "timm_bicubic_crop095",     # tools/eval_sets/timm_bicubic_crop095.py
        "repo_url": "https://github.com/hpi-xnor/BNext",
        "adapter": torch_bnext,
        "sign_module": "HardSign",                # module whose output is torch.sign(input) in eval
    },
    "rbonn": {
        "archs": {"rbonn-reactnet-resnet18-imagenet": "rbonn18"},
        "preprocess": "torchvision_imagenet",     # tools/build_eval_set.py (train.py:140-145 is the same transform)
        "repo_url": "https://github.com/SteveTsui/RBONN",
        "adapter": torch_rbonn,
        "sign_module": "BinaryActivation",
    },
    # yikaiw/SNN (ICCV'21): W1A1 ReActNet-ResNet-18 = step1/birealnet.py with HardBinaryConv binary convs
    "snn": {
        "archs": {"snn-subbit-reactnet18-w1a1-imagenet": "w1a1", "snn-subbit-reactnet18-w0.67a1-imagenet": "w0.67a1"},
        "preprocess": "torchvision_imagenet",     # step1/2 train.py: Resize(256), CenterCrop(224), ImageNet mean/std
        "repo_url": "https://github.com/yikaiw/SNN",
        "adapter": torch_rbonn,
        "sign_module": "BinaryActivation",
        "conv_module": "HardBinaryConv",
    },
    # SA-BNN (AAAI'21) ResNet-18 from the anonymous repo's Drive folder: birealnet.py.txt (no git repo;
    # --repo is the directory holding birealnet.py.txt). Val transform of the Drive train.py.txt:
    # Resize(256), CenterCrop(224), ToTensor, Normalize(ImageNet) = torchvision_imagenet.
    "sabnn": {
        "archs": {"sa-bnn-resnet18-imagenet": "birealnet18"},
        "preprocess": "torchvision_imagenet",
        "repo_url": "https://github.com/anonymous1443/SA-BNN (Google Drive 'Training Techniques/resnet18_61.696/birealnet.py.txt')",
        "adapter": torch_sabnn,
        "sign_module": "BinaryActivation",
        "conv_module": "HardBinaryConv",
    },
    # IDa-Det (ECCV'22) BiRes-18 ImageNet backbone: mmdet/models/backbones/resnet_react.py ResNet_REACT(18) with
    # BiConv (mmcv-0.5.1/mmcv/cnn/bricks/BiConv.py) + the checkpoint's AvgPool/fc head. The repo has no ImageNet
    # eval script; torchvision_imagenet (Resize 256, CenterCrop 224, ImageNet mean/std) is an assumption.
    "idadet": {
        "archs": {"ida-det-bires18-imagenet-backbone": "resnet_react18"},
        "preprocess": "torchvision_imagenet",
        "repo_url": "https://github.com/SteveTsui/IDa-Det",
        "adapter": torch_rbonn,
        "sign_module": "BinaryActivation",
        "conv_module": "BiConv",
    },
}

# IR-Net family (one shared model zoo, four weight functions): torch_irnet
_IRNET = {
    "recu": ("https://github.com/yuanchunyu/ReCU", [], {
        "recu-resnet18-imagenet": "resnet18_1w1a", "recu-resnet34-imagenet": "resnet34_1w1a"}),
    "rbnn": ("https://github.com/lmbxmu/RBNN", [], {
        "rbnn-resnet18-imagenet": "resnet18_1w1a", "rbnn-imagenet-resnet18": "resnet18_1w1a",
        "rbnn-resnet34-imagenet": "resnet34_1w1a", "rbnn-imagenet-resnet34": "resnet34_1w1a"}),
    "siman": ("https://github.com/lmbxmu/SiMaN", [], {
        "siman-resnet18-imagenet": "resnet18_1w1a", "siman-resnet34-imagenet": "resnet34_1w1a"}),
    # BiPer: stage-2 checkpoints (square-wave weights); ImageNet trained with --freq 20 (README, run_stg*.sh)
    "biper": ("https://github.com/edmav4/BiPer", ["--stage2", "--freq", "20"], {
        "biper-resnet18-imagenet": "resnet18_1w1a", "biper-resnet34-imagenet": "resnet34_1w1a"}),
}
for _f, (_url, _argv, _archs) in _IRNET.items():
    FAMILIES[_f] = {"archs": _archs, "preprocess": "torchvision_imagenet",   # dataset/imagenet.py get_imagenet_iter_torch (val)
                    "repo_url": _url, "adapter": torch_irnet, "irnet": _f, "argv": _argv, "subdir": "imagenet",
                    "conv_module": "BinarizeConv2d"}
# CIFAR-10 (models_cifar; test transform ToTensor + Normalize(ImageNet mean/std), cifar*/dataset/dataset.py)
_IRNET_CIFAR = {
    "recu": ("cifar", [], {"recu-vgg-small-cifar10": "vgg_small_1w1a", "recu-resnet20-cifar10": "resnet20_1w1a",
                           "recu-resnet18-cifar10": "resnet18_1w1a"}),
    "rbnn": ("cifar", [], {"rbnn-vgg-small-cifar10": "vgg_small_1w1a", "rbnn-resnet20-cifar10": "resnet20_1w1a",
                           "rbnn-resnet18-cifar10": "resnet18_1w1a", "rbnn-resnet20-bireal-cifar10": "resnet20_bireal_1w1a"}),
    "siman": ("cifar", [], {"siman-vgg-small-cifar10": "vgg_small_1w1a", "siman-resnet20-cifar10": "resnet20_1w1a",
                            "siman-resnet18-cifar10": "resnet18_1w1a"}),
    # BiPer CIFAR: freq is read from the stage-1 config.txt (not published with the checkpoints);
    # README/run_stg1.sh train with --freq 20 (see the integration notes for the check)
    "biper": ("cifar10", ["--stage2", "--freq", "20"], {"biper-vgg-small-cifar10": "vgg_small_1w1a", "biper-resnet20-cifar10": "resnet20_1w1a",
                                                        "biper-resnet18-cifar10": "resnet18_1w1a"}),
}
for _f, (_sub, _argv, _archs) in _IRNET_CIFAR.items():
    FAMILIES[_f + "_cifar"] = {**FAMILIES[_f], "archs": _archs, "preprocess": "cifar10_imagenet_norm",   # tools/eval_sets/cifar10_normalized.py
                               "argv": _argv, "subdir": _sub, "zoo": "models_cifar", "num_classes": 10, "input_hw": (32, 32)}
# third-party Bi-Real-20 CIFAR-10 (ShuaiZ1037/bnn-xnor-bireal; train.py:117-122 Normalize CIFAR mean/std)
FAMILIES["shuaiz"] = {"archs": {"shuaiz-bireal20-cifar10": "birealnet20"}, "preprocess": "cifar10_bireal_norm",
                      "repo_url": "https://github.com/ShuaiZ1037/bnn-xnor-bireal", "adapter": torch_irnet, "irnet": "shuaiz",
                      "argv": [], "subdir": "binary_model", "zoo": "resnet_bireal", "num_classes": 10, "input_hw": (32, 32),
                      "conv_module": "BiRealConv", "scaled_weights": True, "no_num_classes_arg": True}
# bitorch model hub (hpi-xnor/bitorch 0.3.0 model code; Lightning checkpoints). Binarization happens inside
# QConv2d (QActivation, sign with 0 -> +1): only "ge" variants (= the exact source semantics).
FAMILIES["bitorch"] = {"archs": {"bitorch-resnete18-imagenet": "ResnetE18", "bitorch-densenet28-imagenet": "DenseNet28",
                                 "bitorch-meliusnet22-imagenet": "MeliusNet22"},
                       "preprocess": "bitorch_imagenet",          # tools/eval_sets/bitorch_imagenet.py
                       "repo_url": "https://github.com/hpi-xnor/bitorch (release 0.3.0, PyPI wheel)",
                       "adapter": torch_bitorch, "sign_module": "QActivation", "binarize_modes": ("ge",),
                       "conv_binarizes": True}
# Checkpoint keys the published code does not construct (loaded with strict=False only for these).
# SiMaN ImageNet: stem PReLU slope (all 64 values ~1.5e-9) from a code version with nn.PReLU(64)
# instead of the published F.relu; the published forward is converted.
ALLOWED_UNEXPECTED = {"siman": {"nonlinear.weight"}, "siman_cifar": set()}


def convert_graph(family: str, model, model_id: str, binarize: str):
    fam = FAMILIES[family]
    if "irnet" in fam:
        return fam["adapter"].convert(model, fam["irnet"], model_id, binarize=binarize, input_hw=fam.get("input_hw", (224, 224)))
    return fam["adapter"].convert(model, model_id, binarize=binarize)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit(repo: str) -> str:
    if not os.path.exists(os.path.join(repo, ".git")) and os.path.exists(os.path.join(repo, "COMMIT_SHA")):
        return open(os.path.join(repo, "COMMIT_SHA")).read().strip()   # tarball of a pinned commit
    if not os.path.exists(os.path.join(repo, ".git")):
        return "none (not a git checkout; see source_sha256)"
    head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    p = os.path.join(repo, ".git", ref)
    if os.path.exists(p):
        return open(p).read().strip()
    for line in open(os.path.join(repo, ".git", "packed-refs")):
        if line.strip().endswith(ref):
            return line.split()[0]
    return "unknown"


def load_model(family: str, repo: str, model_id: str, weights: str):
    import collections
    import collections.abc
    import contextlib
    import importlib
    import io
    import warnings

    import torch

    sys.dont_write_bytecode = True   # never write __pycache__ into the source clone
    arch = FAMILIES[family]["archs"][model_id]
    if "irnet" in FAMILIES[family]:
        return _load_irnet(family, repo, arch, weights)
    if family == "snn":
        return _load_snn(repo, arch, weights)
    if family == "bitorch":
        return _load_bitorch(repo, arch, weights)
    if family == "sabnn":
        return _load_sabnn(repo, arch, weights)
    if family == "idadet":
        return _load_idadet(repo, arch, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # timm.models.layers / registry deprecation shims
        if family == "bnext":
            sys.path.insert(0, os.path.join(os.path.abspath(repo), "src"))
            module = importlib.import_module("bnext")
            with contextlib.redirect_stdout(io.StringIO()):     # BNext.__init__ prints the variant
                model = module.BNext(num_classes=1000, size=arch, ELM_Attention=True, Infor_Recoupling=True)
            cls = f"bnext.BNext(size={arch!r}, ELM_Attention=True, Infor_Recoupling=True)"
            sd = torch.load(weights, map_location="cpu", weights_only=True)   # flat state_dict, no optimizer
        else:
            # rbonn.py:44-49 uses collections.Iterable, removed in Python 3.10; alias it to
            # collections.abc.Iterable in this process (the clone is not modified).
            if not hasattr(collections, "Iterable"):
                collections.Iterable = collections.abc.Iterable
            sys.path.insert(0, os.path.join(os.path.abspath(repo), "2_step2_rbonn"))
            module = importlib.import_module("rbonn")
            model = getattr(module, arch)()
            cls = f"rbonn.{arch}()"
            ck = torch.load(weights, map_location="cpu", weights_only=True)
            assert set(ck) == {"state_dict"}, f"unexpected checkpoint keys {sorted(ck)}"
            sd = ck["state_dict"]
    assert all(isinstance(v, torch.Tensor) for v in sd.values()), "checkpoint is not a flat state_dict"
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    src = {os.path.relpath(module.__file__, repo): sha256_file(module.__file__)}
    return model, cls, src


def _purge(names) -> None:
    for k in list(sys.modules):
        if k.split(".")[0] in names:
            del sys.modules[k]


def _load_irnet(family: str, repo: str, arch: str, weights: str):
    """Official ReCU/RBNN/SiMaN/BiPer model. utils.options parses sys.argv at import time, so the
    import runs with a stub argv (family options, e.g. BiPer --stage2 --freq 20). RBNN's
    BinarizeConv2d.__init__ calls .cuda() and scipy.stats.ortho_group (random init, overwritten by
    the checkpoint): .cuda is a no-op and, if scipy cannot be imported, ortho_group is stubbed."""
    import importlib
    import types

    import torch

    fam = FAMILIES[family]
    sub = os.path.join(os.path.abspath(repo), fam["subdir"])
    pkgs = ("modules", "utils", "models_imagenet", "models_cifar", "dataset", "resnet_bireal", "binary_modules")
    pkgs_files = [f + ".py" for f in pkgs]
    _purge(pkgs)
    saved = {k: sys.modules.get(k) for k in ("scipy", "scipy.stats")}
    try:
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):   # numpy 1.x/2.x ABI warning of a broken scipy
            import scipy.stats  # noqa: F401
        stubbed = False
    except Exception:   # base env: scipy built against an older numpy
        st = types.ModuleType("scipy.stats")
        st.ortho_group = types.SimpleNamespace(rvs=lambda dim: np.eye(dim))
        sc = types.ModuleType("scipy")
        sc.stats = st
        sys.modules["scipy"], sys.modules["scipy.stats"] = sc, st
        stubbed = True
    old_argv, old_cuda = sys.argv, torch.Tensor.cuda
    sys.argv = ["main.py"] + list(fam["argv"])
    torch.Tensor.cuda = lambda self, *a, **k: self
    # RBNN's utils/ has no __init__.py (namespace package): any plain utils.py / modules.py elsewhere on
    # sys.path would shadow it, so the import runs with such entries removed
    saved_path = list(sys.path)
    sys.path[:] = [sub] + [q for q in sys.path if not any(os.path.exists(os.path.join(q or '.', f)) for f in pkgs_files)]
    try:
        zoo = importlib.import_module(fam.get("zoo", "models_imagenet"))
        if fam.get("no_num_classes_arg"):
            model = getattr(zoo, arch)()
            bm = importlib.import_module("binary_modules")
        else:
            model = getattr(zoo, arch)(num_classes=fam.get("num_classes", 1000))
            bm = importlib.import_module("modules.binarized_modules")
    finally:
        sys.argv, torch.Tensor.cuda = old_argv, old_cuda
        sys.path[:] = saved_path
        if stubbed:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    src = {os.path.relpath(f, repo): sha256_file(f) for f in (sys.modules[type(model).__module__].__file__, bm.__file__)}
    cls = f"{fam['subdir']}/{fam.get('zoo', 'models_imagenet')}.{arch}(num_classes={fam.get('num_classes', 1000)}) argv={fam['argv']}"
    if weights is None:   # tests: random initialization
        return model.eval(), cls, src
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    assert "state_dict" in ck, f"unexpected checkpoint keys {sorted(ck)}"
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in ck["state_dict"].items()}
    res = model.load_state_dict(sd, strict=False)
    # SiMaN CIFAR checkpoints predate the ``percent`` buffer (registered as 0 = no clipping, which only
    # rewrites the latent weight and never reaches bw = 2B - 1); keep the constructor value 0.
    missing = [k for k in res.missing_keys if not (family == "siman_cifar" and k.endswith(".percent"))]
    assert not missing, f"missing keys {missing}"
    if family == "siman_cifar":
        assert all(float(b) == 0.0 for n, b in model.named_buffers() if n.endswith(".percent"))
    extra = set(res.unexpected_keys) - ALLOWED_UNEXPECTED.get(family, set())
    assert not extra, f"unexpected checkpoint keys {sorted(extra)}"
    model.eval()
    model._bitino_load_notes = {"unexpected_keys_ignored": sorted(res.unexpected_keys), "missing_keys_default": sorted(res.missing_keys),
                                  "scipy_stubbed": stubbed,
                                  "argv": fam["argv"], "checkpoint_meta": {k: v for k, v in ck.items() if isinstance(v, (int, float, str))}}
    return model, cls, src


def _load_snn(repo: str, arch: str, weights: str):
    """yikaiw/SNN ReActNet-ResNet-18. The W1A1 checkpoint (no ``patterns``) is the ReActNet binary
    model: step1/birealnet.py BiRealNet with every BasicBlock.binary_conv = HardBinaryConv (the class
    defined in that file; the published step1 file builds a real-valued conv3x3 there, which is
    ReActNet's step-1 W32A1 setting). Latent weights cluster at +-1 (clamp STE), consistent with
    HardBinaryConv training. The 0.67-bit checkpoint uses step2/birealnet.py (bit_num = 6)."""
    import importlib

    import torch

    _purge(("birealnet", "modules"))
    step = "step1" if arch == "w1a1" else "step2"
    sub = os.path.join(os.path.abspath(repo), step)
    sys.path.insert(0, sub)
    try:
        mod = importlib.import_module("birealnet")
        if arch == "w1a1":
            model = mod.birealnet18()
            for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
                for blk in layer:
                    c = blk.binary_conv
                    blk.binary_conv = mod.HardBinaryConv(c.in_channels, c.out_channels, kernel_size=3, stride=c.stride[0], padding=1)
            cls = "step1/birealnet.birealnet18() with binary_conv = birealnet.HardBinaryConv"
        else:
            model = mod.birealnet18(bit_num=6)
            cls = "step2/birealnet.birealnet18(bit_num=6)"
    finally:
        sys.path.remove(sub)
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    model._bitino_load_notes = {"checkpoint_meta": {"epoch": ck.get("epoch"), "best_top1_acc": float(ck["best_top1_acc"])}}
    files = [mod.__file__] + ([os.path.join(sub, "modules", "binarize.py"), os.path.join(sub, "modules", "patterns.py")] if step == "step2" else [])
    return model, cls, {os.path.relpath(f, repo): sha256_file(f) for f in files}


def _load_bitorch(repo: str, arch: str, weights: str, input_hw: tuple[int, int] = (224, 224)):
    """bitorch 0.3.0 model (Model wrapper with ``_model``) from the extracted release (read only). bitorch's
    model_hub imports pandas at import time (only for downloading from the hub; the base env's pandas is built
    against numpy 1.x): a stub module stands in for it in this process. The Lightning checkpoint's state_dict
    keys are "model.<key>" (the wrapper) plus "teacher.*" (the FP ResNet-50 used for distillation), which is
    dropped; the rest is loaded with strict=True, as bitorch's lightning_checkpoint_to_state_dict does."""
    import importlib
    import importlib.util
    import types

    import torch

    if "pandas" not in sys.modules:
        pd = types.ModuleType("pandas")
        pd.__spec__ = importlib.util.spec_from_loader("pandas", loader=None)
        pd.DataFrame = type("DataFrame", (), {})
        pd.Series = type("Series", (), {})
        sys.modules["pandas"] = pd
    _purge(("bitorch",))
    sys.path.insert(0, os.path.abspath(repo))
    try:
        models = importlib.import_module("bitorch.models")
    finally:
        sys.path.remove(os.path.abspath(repo))
    model = getattr(models, arch)(input_shape=(1, 3) + tuple(input_hw), num_classes=1000)
    cls = f"bitorch.models.{arch}(input_shape={(1, 3) + tuple(input_hw)}, num_classes=1000) [bitorch 0.3.0]"
    src = {os.path.relpath(f, repo): sha256_file(f) for f in (
        sys.modules[type(model).__module__].__file__, os.path.join(repo, "bitorch", "models", "densenet.py"),
        os.path.join(repo, "bitorch", "layers", "qconv2d.py"), os.path.join(repo, "bitorch", "layers", "qactivation.py"),
        os.path.join(repo, "bitorch", "quantizations", "sign.py"), os.path.join(repo, "bitorch", "quantizations", "progressive_sign.py"))}
    if weights is None:   # tests: random initialization
        return model.eval(), cls, src
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    dropped = sorted(k for k in sd if not k.startswith("model."))
    assert all(k.startswith("teacher.") for k in dropped), f"unexpected checkpoint keys {dropped[:5]}"
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
    model.eval()
    hp = ck.get("hyper_parameters", {})
    model._bitino_load_notes = {
        "dropped_teacher_tensors": len(dropped), "pytorch_lightning_version": ck.get("pytorch-lightning_version"),
        "epoch": ck.get("epoch"), "pandas_stubbed": True,
        "hyper_parameters": {k: hp.get(k) for k in ("model", "teacher", "input_quantization", "weight_quantization", "padding_value",
                                                     "max_epochs", "optimizer", "lr", "dataset")}}
    return model, cls, src


def _load_source_file(name: str, path: str):
    """Executes one source file as module ``name`` without putting its directory on sys.path (works for a
    non-.py file such as the SA-BNN Drive ``birealnet.py.txt``; no bytecode is written)."""
    import importlib.machinery
    import importlib.util

    sys.dont_write_bytecode = True
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def _load_sabnn(repo: str, arch: str, weights: str):
    """SA-BNN ResNet-18 from the anonymous Drive folder: ``birealnet.py.txt`` (read, not modified) and the
    legacy (non-zip) pickle checkpoint dict{epoch, state_dict ('module.' prefix), best_top1_acc, optimizer}.
    ``downsample2`` is the same module object as ``downsample`` (birealnet.py.txt:78/82), so the checkpoint
    stores it twice; both copies are checked to be identical before the strict load."""
    import torch

    path = os.path.join(os.path.abspath(repo), "birealnet.py.txt")
    mod = _load_source_file("sabnn_birealnet", path)
    model = getattr(mod, arch)()
    if weights is None:   # tests: random initialization
        return model.eval(), f"birealnet.py.txt {arch}()", {"birealnet.py.txt": sha256_file(path)}
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    assert set(ck) == {"epoch", "state_dict", "best_top1_acc", "optimizer"}, f"unexpected checkpoint keys {sorted(ck)}"
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in ck["state_dict"].items()}
    dup = {k: k.replace(".downsample2.", ".downsample.") for k in sd if ".downsample2." in k}
    assert dup and all(torch.equal(sd[k], sd[v]) for k, v in dup.items()), "downsample2 tensors differ from downsample"
    model.load_state_dict(sd, strict=True)
    model.eval()
    model._bitino_load_notes = {"checkpoint_meta": {"epoch": int(ck["epoch"]), "best_top1_acc": float(ck["best_top1_acc"])},
                                  "downsample2_tensors_identical_to_downsample": len(dup),
                                  "optimizer_state_ignored": True}
    return model, f"birealnet.py.txt {arch}() [anonymous Drive release]", {"birealnet.py.txt": sha256_file(path)}


def _load_idadet(repo: str, arch: str, weights: str):
    """IDa-Det BiRes-18 backbone + ImageNet head. The backbone classes are executed from the official files
    (read only): ``mmdet/models/backbones/resnet_react.py`` (BasicBlock, make_res_layer_bin, ResNet_REACT) and
    ``mmcv-0.5.1/mmcv/cnn/bricks/BiConv.py`` (BiConv, Binarization, BinaryActivation). mmcv/mmdet themselves
    are not built or imported: resnet_react.py's imports are satisfied by stub modules providing
    build_conv_layer / build_norm_layer with the semantics of mmcv-0.5.1/mmcv/cnn/bricks/{conv,norm}.py
    (Conv/Conv2d -> nn.Conv2d, BiConv -> BiConv; BN -> ('bn' + postfix, nn.BatchNorm2d(eps=1e-5))) and
    registry/init/logging helpers that raise if called (ResNet_REACT's constructor and forward never call
    them). Backbone arguments from configs/voc_faster_rcnn_bir18.py. The checkpoint has fc.weight/bias
    (1000 x 512) but the backbone has no head: ``ResNetREACTClassifier`` adds AdaptiveAvgPool2d(1) +
    Linear(512, 1000) on the layer4 output (the standard ResNet head, as in RBONN rbonn.py:225-227) -- an
    assumption, validated only by the accuracy it gives."""
    import collections
    import collections.abc
    import types

    import torch
    import torch.nn as nn

    if not hasattr(collections, "Iterable"):   # BiConv.py:14 uses collections.Iterable (removed in Python 3.10)
        collections.Iterable = collections.abc.Iterable
    root = os.path.abspath(repo)
    biconv_path = os.path.join(root, "mmcv-0.5.1", "mmcv", "cnn", "bricks", "BiConv.py")
    react_path = os.path.join(root, "mmdet", "models", "backbones", "resnet_react.py")
    biconv = _load_source_file("idadet_biconv", biconv_path)

    def build_conv_layer(cfg, *args, **kwargs):          # mmcv-0.5.1/mmcv/cnn/bricks/conv.py build_conv_layer
        cfg_ = dict(type="Conv2d") if cfg is None else dict(cfg)
        layer = {"Conv": nn.Conv2d, "Conv2d": nn.Conv2d, "BiConv": biconv.BiConv}[cfg_.pop("type")]
        return layer(*args, **kwargs, **cfg_)

    def build_norm_layer(cfg, num_features, postfix=""):  # mmcv-0.5.1/mmcv/cnn/bricks/norm.py build_norm_layer (BN)
        cfg_ = dict(cfg)
        assert cfg_.pop("type") == "BN"
        requires_grad = cfg_.pop("requires_grad", True)
        cfg_.setdefault("eps", 1e-5)
        layer = nn.BatchNorm2d(num_features, **cfg_)
        for prm in layer.parameters():
            prm.requires_grad = requires_grad
        return "bn" + str(postfix), layer

    def _unused(*a, **k):
        raise RuntimeError("stubbed mmcv/mmdet helper called")

    stubs = {}
    for name in ("mmcv", "mmcv.cnn", "mmcv.runner", "mmdet", "mmdet.ops", "mmdet.utils", "mmdet.models",
                 "mmdet.models.builder", "mmdet.models.utils", "mmdet.models.backbones"):
        m = types.ModuleType(name)
        m.__path__ = []
        stubs[name] = m
    stubs["mmcv.cnn"].__dict__.update(build_conv_layer=build_conv_layer, build_norm_layer=build_norm_layer,
                                      constant_init=_unused, kaiming_init=_unused)
    stubs["mmcv.runner"].load_checkpoint = _unused
    stubs["mmdet.ops"].build_plugin_layer = _unused
    stubs["mmdet.utils"].get_root_logger = _unused
    stubs["mmdet.models.builder"].BACKBONES = types.SimpleNamespace(register_module=lambda cls: cls)
    stubs["mmdet.models.utils"].ResLayer = _unused
    saved = {k: sys.modules.get(k) for k in list(stubs) + ["mmdet.models.backbones.resnet_react"]}
    sys.modules.update(stubs)
    try:
        react = _load_source_file("mmdet.models.backbones.resnet_react", react_path)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    class ResNetREACTClassifier(react.ResNet_REACT):
        def __init__(self, num_classes=1000, **kw):
            super().__init__(**kw)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * self.block.expansion, num_classes)

        def forward(self, x):
            x = super().forward(x)[-1]
            return self.fc(torch.flatten(self.avgpool(x), 1))

    assert arch == "resnet_react18"
    kw = dict(depth=18, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1, conv_cfg=dict(type="BiConv"),
              norm_cfg=dict(type="BN", requires_grad=True), style="pytorch")   # configs/voc_faster_rcnn_bir18.py backbone
    model = ResNetREACTClassifier(**kw)
    src = {os.path.relpath(f, repo): sha256_file(f) for f in (react_path, biconv_path)}
    cls = ("mmdet/models/backbones/resnet_react.py ResNet_REACT(depth=18, conv_cfg=BiConv, norm_cfg=BN, frozen_stages=1) "
           "+ AdaptiveAvgPool2d(1) + Linear(512, 1000) head [mmcv registry stubbed]")
    if weights is None:   # tests: random initialization
        model.eval()      # ResNet_REACT.train() returns None (resnet_react.py:561-568), so no chaining
        return model, cls, src
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in sd.values()), "checkpoint is not a flat state_dict"
    model.load_state_dict(sd, strict=True)
    model.eval()
    model._bitino_load_notes = {"head": "AdaptiveAvgPool2d(1) + Linear(512, 1000) added (not in the backbone source)",
                                  "mmcv_stubs": "build_conv_layer / build_norm_layer reimplemented from mmcv-0.5.1 (BN, Conv, BiConv)"}
    return model, cls, src


class _FProxy:
    """Stands in for the module-global ``F`` (torch.nn.functional) of a source binary-conv module.
    Inside a hooked conv module's forward, F.conv2d(input, weight, ...) counts exact zeros of the
    binarized input, optionally records the weight and optionally substitutes a given weight."""

    def __init__(self, real, state):
        self._real, self._state = real, state

    def __getattr__(self, k):
        return getattr(self._real, k)

    def conv2d(self, inp, weight, *a, **k):
        st = self._state
        name = st["current"]
        if name is not None:
            st["zeros"] += int((inp == 0).sum())
            st["n"] += int(inp.numel())
            if st["capture"] is not None:
                st["capture"][name] = weight.detach().clone()
            if st["override"] is not None:
                weight = st["override"][name].to(weight.dtype)
        return self._real.conv2d(inp, weight, *a, **k)


def _install_proxy(model, conv_cls: str, capture=None, override=None):
    state = {"current": None, "zeros": 0, "n": 0, "capture": capture, "override": override}
    hooks, globs = [], {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == conv_cls:
            hooks.append(mod.register_forward_pre_hook(lambda _m, _i, name=name: state.__setitem__("current", name)))
            hooks.append(mod.register_forward_hook(lambda _m, _i, _o: state.__setitem__("current", None)))
            gl = type(mod).forward.__globals__
            globs[id(gl)] = gl
    saved = []
    for gl in globs.values():
        saved.append((gl, gl["F"]))
        gl["F"] = _FProxy(gl["F"], state)

    def remove():
        for h in hooks:
            h.remove()
        for gl, f in saved:
            gl["F"] = f
    return state, remove


def reference_outputs(model, x_nhwc: np.ndarray, dtype, sign_module: str | None, conv_module: str | None = None,
                      capture: dict | None = None, override: dict | None = None, ge: bool = False) -> tuple[np.ndarray, dict]:
    """Logits of the official model plus exact-zero counts at every binarization input.

    Every sign module sees move(x) = x + bias. A forward hook also checks that the
    straight-through expression (sign - approx) + approx returns exactly torch.sign(input)."""
    import torch

    m = model.to(dtype)
    zeros = {"sign_input_exact_zero": 0, "n_binarized": 0, "sign_module_output_ne_torch_sign": 0}
    hooks = []
    proxy = None
    if conv_module and (sign_module is None or capture is not None or override is not None):
        proxy = _install_proxy(m, conv_module, capture, override)
    for name, mod in m.named_modules():
        if sign_module is not None and type(mod).__name__ == sign_module:
            def fwd(_m, inp, out):
                zeros["sign_input_exact_zero"] += int((inp[0] == 0).sum())
                zeros["n_binarized"] += int(inp[0].numel())
                want = torch.where(inp[0] >= 0, 1.0, -1.0).to(out.dtype) if ge else torch.sign(inp[0])
                zeros["sign_module_output_ne_torch_sign"] += int((out != want).sum())
            hooks.append(mod.register_forward_hook(fwd))
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), 16):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + 16].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).numpy())
    for h in hooks:
        h.remove()
    if proxy is not None:
        st, remove = proxy
        remove()
        if sign_module is None:   # binarization inside the conv module: zeros of the sign output fed to F.conv2d
            zeros = {"sign_input_exact_zero": st["zeros"], "n_binarized": st["n"]}
    model.to(torch.float32)
    return np.concatenate(out), zeros


# ------------------------------------------------------------- numpy executor workers
_G = {}
_EXACT: dict[str, dict] = {}   # variant key -> {constant name: float64 value} (set before forking)


def _init_worker(paths: dict) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
        if k in _EXACT:   # same graph with the folded constants kept in float64 (analysis only)
            gx = bfformat.read(p)
            for name, v in _EXACT[k].items():
                assert np.array_equal(v.astype(np.float32), gx.constants[name].data), f"{name}: bfmodel differs from adapter output"
                gx.constants[name].data = v
            _G[k + "#exact"] = gx


def _run_one(job):
    key, x, want_f64 = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    zeros = [int(np.sum(env[n.inputs[0]] == 0)) for n in g.nodes if n.op == "BConv2D"]
    out32 = env[g.outputs[0]].reshape(-1)
    out64, zeros64 = None, []
    if want_f64:
        env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        out64 = env64[g.outputs[0]].reshape(-1)
        zeros64 = [int(np.sum(env64[n.inputs[0]] == 0)) for n in g.nodes if n.op == "BConv2D"]
    outx = None
    if want_f64 and key + "#exact" in _G:
        outx = executor.run(_G[key + "#exact"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
    return out32, out64, zeros, zeros64, outx


def main(family: str) -> int:
    fam = FAMILIES[family]
    PREPROCESS = fam["preprocess"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(fam["archs"]))
    ap.add_argument("--repo", required=True, help=f"clone of {fam['repo_url']}")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"),
                    help="models.json holding the expected checkpoint SHA-256")
    ap.add_argument("--sha256", help="expected SHA-256 (overrides --registry)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16, help="worker processes for the numpy executor")
    ap.add_argument("--torch-threads", type=int, default=min(16, os.cpu_count() or 1), help="torch CPU threads for the references")
    args = ap.parse_args()

    expected = args.sha256
    if not expected:
        reg = json.load(open(args.registry))
        entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
        if entry is None:
            print(f"ERROR: {args.model_id} not in {args.registry}", file=sys.stderr)
            return 2
        expected = entry["checkpoint"]["sha256"]
    actual = sha256_file(args.weights)
    if actual != expected:
        print(f"ERROR: checkpoint hash mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}")
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    import torchvision

    torch.set_num_threads(args.torch_threads)
    model, cls, src_hashes = load_model(family, args.repo, args.model_id, args.weights)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    sign_module, conv_module = fam.get("sign_module"), fam.get("conv_module")
    cap32: dict | None = {} if conv_module else None
    cap64: dict | None = {} if conv_module else None
    modes = fam.get("binarize_modes", ("ge", "sign_zero"))
    ge_src = modes == ("ge",)   # source sign maps 0 -> +1 (the hook then checks against x >= 0 ? +1 : -1)
    ref32, zeros32 = reference_outputs(model, x, torch.float32, sign_module, conv_module, capture=cap32, ge=ge_src)
    ref64, zeros64 = reference_outputs(model, x, torch.float64, sign_module, conv_module, capture=cap64, ge=ge_src)
    weight_derivation = None
    if conv_module:
        # binary weights the source passes to F.conv2d, fp32 vs fp64 (sign level; weights carry a
        # positive per-channel scale for SNN, exactly +-1 for the IR-Net family)
        diff = {k: int((torch.sign(cap32[k]).double() != torch.sign(cap64[k])).sum()) for k in cap32}
        zeros_w = {k: int((cap32[k] == 0).sum()) for k in cap32}
        weight_derivation = {"fp64_vs_fp32_sign_differences_total": int(sum(diff.values())),
                             "fp64_vs_fp32_sign_differences_per_layer": {k: v for k, v in diff.items() if v},
                             "fp32_exact_zero_weights_total": int(sum(zeros_w.values())),
                             "n_weights": int(sum(v.numel() for v in cap32.values()))}
        _, info0 = convert_graph(family, model, args.model_id, "sign_zero")
        if "binary_weights" in info0:   # adapter weights == the tensor the fp32 source feeds to F.conv2d
            mism = {k: int((torch.from_numpy(v.astype(np.float32)) != torch.sign(cap32[k])).sum()) for k, v in info0["binary_weights"].items()}
            assert set(mism) == set(cap32), f"binary conv sets differ: {sorted(set(mism) ^ set(cap32))}"
            weight_derivation["adapter_bits_vs_source_fp32_sign_mismatches"] = int(sum(mism.values()))
            assert weight_derivation["adapter_bits_vs_source_fp32_sign_mismatches"] == 0, mism
        if "irnet" in fam and not fam.get("scaled_weights"):
            exact_pm1 = all(bool(torch.all(v.abs() == 1)) for v in cap32.values())
            assert exact_pm1, "IR-Net source weights are not exactly +-1"
            override = {k: v.double() for k, v in cap32.items()}
        else:
            override = {k: torch.sign(cap32[k]).double() * cap64[k].abs() for k in cap32}
        ref64_naive = ref64
        ref64, zeros64 = reference_outputs(model, x, torch.float64, sign_module, conv_module, override=override)
        ref64_naive.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))
        weight_derivation["primary_f64_reference"] = "official model in float64 with the fp32-derived binary weights (F.conv2d proxy)"
        weight_derivation["naive_f64_vs_primary_f64"] = compare_outputs(ref64, ref64_naive, labels)
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_preprocessing": PREPROCESS,
              "reference": {"framework": f"torch {torch.__version__} (CPU), torchvision {torchvision.__version__}",
                            "class": cls, "repo_commit": repo_commit(args.repo), "source_sha256": src_hashes,
                            "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)),
                            "top5_accuracy": float(np.mean([labels[i] in np.argsort(-ref32[i])[:5] for i in range(len(x))])),
                            "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                            "top5_accuracy_f64": float(np.mean([labels[i] in np.argsort(-ref64[i])[:5] for i in range(len(x))])),
                            "exact_zeros_at_binarization_f32": zeros32, "exact_zeros_at_binarization_f64": zeros64},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "variants": {}}
    if weight_derivation is not None:
        report["reference"]["weight_derivation"] = weight_derivation
    if getattr(model, "_bitino_load_notes", None):
        report["reference"]["load_notes"] = model._bitino_load_notes
    print(json.dumps(report["reference"], indent=1))

    if ge_src:
        report["reference"]["sign_semantics"] = ("source sign(x) = +1 for x >= 0 (incl. -0.0), -1 otherwise; "
                                                 "'sign_module_output_ne_torch_sign' counts deviations from that rule")
    files = {}
    for binarize in modes:
        for variant, fold in (("unfused", False), ("fused", True)):
            graph, info = convert_graph(family, model, args.model_id, binarize)
            applied = passes.optimize(graph, fold=fold)
            stem = "model" if binarize == "ge" else "model_sign_zero"
            key = variant if binarize == "ge" else f"sign_zero_{variant}"
            graph.manifest = {
                "schema": "bitino.manifest.v1",
                "model_id": args.model_id,
                "variant": f"{variant}, input_binarize={binarize}",
                "source": {"framework": "pytorch", "class": cls, "repo": fam["repo_url"],
                           "repo_commit": report["reference"]["repo_commit"], "source_sha256": src_hashes,
                           "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
                "conversion": {"tool": "tools/convert_irnet.py" if "irnet" in fam else f"tools/convert_{family}.py", "adapter": fam["adapter"].__name__,
                               "passes": applied, "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "python": platform.python_version(), "torch": torch.__version__,
                               "input_binarize": binarize,
                               "binarize_note": ("exact torch.sign semantics (sign(0)=0); reference kernels only" if binarize == "sign_zero"
                                                 else "deployable: 0 -> +1 at binarization; differs from torch.sign only for exact zeros")},
                "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                          "preprocessing": pre["definition"]},
                "outputs": ["logits"],
                "op_versions": {"format": 1},
                "layer_precision": info["layer_precision"],
                "weight_diagnostics": info["weight_diagnostics"],
                "unused_in_forward": info["unused_in_forward"],
            }
            if not fold:
                _EXACT[key] = info["float64_constants"]
            path = os.path.join(args.out_dir, f"{stem}.bitino" if fold else f"{stem}_unfused.bitino")
            file_sha = bfformat.write(graph, path)
            files[key] = path
            report["variants"][key] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": binarize, "op_counts": graph.op_counts(), "passes": applied}
            print(f"[{key}] wrote {path}: {graph.op_counts()} passes={applied}")

    # Level 3: numpy executor (float32 = runtime semantics; float64 = strict semantic check).
    jobs = [(k, x[i:i + 1], True) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
        results = pool.map(_run_one, jobs, chunksize=4)
    by_key = {k: results[j * len(x):(j + 1) * len(x)] for j, k in enumerate(files)}
    np_logits = {}
    for k, res in by_key.items():
        out32 = np.stack([r[0] for r in res]).astype(np.float32)
        out64 = np.stack([r[1] for r in res])
        zeros = np.array([r[2] for r in res])              # [samples, BConv2D]
        zeros64 = np.array([r[3] for r in res])
        np_logits[k] = out32
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        cmp = compare_outputs(ref32, out32, labels)
        cmp64 = compare_outputs(ref64, out32, labels)
        sem = compare_outputs(ref64, out64, labels)
        per_sample = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per_sample <= 1e-6))
        sem["n_samples_max_abs_le_1e-4"] = int(np.sum(per_sample <= 1e-4))
        sem["deviating_samples_gt_1e-4"] = [int(i) for i in np.where(per_sample > 1e-4)[0]]
        sem["per_sample_max_abs"] = [float(v) for v in per_sample]
        if res[0][4] is not None:
            outx = np.stack([r[4] for r in res])
            px = np.abs(outx - ref64).max(axis=1)
            sem["float64_constants"] = {"max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-6": int(np.sum(px <= 1e-6)),
                                        "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                                        "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1)))}
        report["variants"][k].update({
            "numpy_vs_reference_f32": cmp, "numpy_vs_reference_f64": cmp64,
            "semantic_check_ir_f64_vs_reference_f64": sem,
            "exact_zeros_at_bconv_inputs_numpy_f32": {"total": int(zeros.sum()), "samples_with_zeros": int(np.sum(zeros.sum(1) > 0))},
            "exact_zeros_at_bconv_inputs_numpy_f64": {"total": int(zeros64.sum()), "samples_with_zeros": int(np.sum(zeros64.sum(1) > 0))}})
        print(f"[{k}] vs f32 ref: top1 agree {cmp['top1_agreement']:.4f} max_abs {cmp['max_abs_error']:.3e} | vs f64 ref: top1 agree "
              f"{cmp64['top1_agreement']:.4f} max_abs {cmp64['max_abs_error']:.3e} | SEMANTIC (f64 IR vs f64 ref): max_abs "
              f"{sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} samples<=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{len(x)} "
              f"| exact zeros at BConv2D inputs (numpy f32): {int(zeros.sum())}"
              + (f" | f64 constants: max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else ""))
    if "sign_zero" in modes:
        report["ge_vs_sign_zero_numpy_f32"] = {v: compare_outputs(np_logits[f"sign_zero_{v}"], np_logits[v], labels) for v in ("unfused", "fused")}
        print("ge vs sign_zero (numpy f32) exact match:", {v: c["exact_match"] for v, c in report["ge_vs_sign_zero_numpy_f32"].items()})
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(np_logits["unfused"], np_logits["fused"], labels)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv.pop(1)))
