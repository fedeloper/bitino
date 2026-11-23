#!/usr/bin/env python3
"""Export a verified plain-PyTorch BNN checkpoint to a canonical QONNX file of the Bitino profile
(docs/qonnx-profile.md, ``bitino-qonnx/1``) with torch.onnx.export: the first step of the QONNX route for PyTorch
families (docs/qonnx-frontend.md step 1, docs/qonnx-step1.md; the second step is tools/import_qonnx.py in
.cache/venv-qonnx).

Runs in .cache/venv-qonnx2 (torch 2.6.0+cpu, torchvision 0.21.0+cpu, numpy 2.2.6, onnx 1.17.0: the torch / torchvision
versions of the recorded PyTorch conversions, plus the onnx package that torch.onnx.export needs), with python -I:

  .cache/venv-qonnx2/bin/python -I tools/export_torch_qonnx.py --model-id ab-bnn-imagenet-reactnet18 \
      --repo .cache/repos/AB-BNN --weights .cache/checkpoints/ab-bnn/imagenet_reactnet18_step2.h5 \
      --out-dir .cache/qonnx_step1/ab-bnn-imagenet-reactnet18/sign_zero [--eval-dir .cache/data/eval128] [--binarizer ge]

Families (export hooks; each mirrors the legacy adapter node for node and checks every module type it meets):
  abbnn   AB-BNN ReActNet-18 / -34 (ImageNet, CIFAR-10 / -100)                    bitino.adapters.torch_abbnn
  reacta  ReActNet-A: AB-BNN Reactnet; BatchNorm flavour BURN, S2-BNN              torch_abbnn._convert_reactnet_a
  rbonn   RBONN BiRealNet, SNN sub-bit (W1A1, W0.67A1), IDa-Det BiRes-18 backbone   torch_rbonn
  irnet   ReCU, BiPer, RBNN, SiMaN (ImageNet ResNet-18/34, CIFAR ResNet-18/20, VGG-Small), ShuaiZ Bi-Real-20  torch_irnet
  sabnn   SA-BNN ResNet-18                                                         torch_sabnn
  bnext   BNext-T / S / M / L                                                      torch_bnext
  bitorch bitorch 0.3.0 ResnetE18, DenseNet28, MeliusNet22 (ge only: sign(0) = +1)   torch_bitorch
tools/qonnx_step1_acceptance.py runs export + check-qonnx + import + comparison with the published files.

The model is loaded exactly as the family's converter loads it (tools/convert_*.py ``load_model``: the authors' class
definitions from a read-only clone, ``torch.load(weights_only=True)``, ``load_state_dict(strict=True)``), after its
checkpoint SHA-256 is checked against the registry. The hook builds an exportable copy of the network whose forward
is plain ONNX, in the adapter's node order:

* binarizers: ``torch.sign`` stays ``torch.sign`` (ONNX ``Sign``, sign(0) = 0); ``--binarizer ge`` replaces it by
  ``BipolarQuant(x, 1)`` (x >= 0 -> +1), the deployable choice of the legacy converters, recorded in export.json, in
  the model metadata (bitino.notes) and in the graph itself (never applied silently);
* learned sign thresholds / LearnableBias as ``Add(x, t)``; RPReLU as ``Add`` / ``PRelu`` / ``Add``;
* binary weights canonically (profile R4): ``Conv(Bin(x), BipolarQuant(W_latent, 1))`` followed by ``Mul(alpha[o])``
  (and ``Add(bias)``) in the conv's naming group, which the importer folds into post_mul / post_add; alpha keeps its
  sign (the legacy bit polarity); W_latent has no exact zeros and the sign of the source's binary weights (checked);
* BatchNorm (eval) as ``Mul`` + ``Add`` of float64 carriers (``Cast(DOUBLE initializer, to=FLOAT)``, R6 / R7);
  every constant the source derives from parameters (weight standardization, scales, b / beta, 1 - scale) is a float64
  carrier computed by the adapter's own helpers: the importer stores its float32 rounding (the legacy rule "float64
  then one rounding"), the float64 reference sees the float64 value.

Profile rewrite (``canonicalize``): IR version 10, Constant nodes -> initializers, de-duplicated buffers (Identity) ->
the initializer, explicit ``Pad(0)`` of every binarized Conv input (R2), naming metadata (bitino.name / order / attrs /
extra_names / consts, from the module paths or the hook's ``bitino_meta``), model metadata (§ 6). The file must pass
``check-qonnx`` (bitino.qonnx_profile), else the export fails. torch's JIT CSE pass is disabled during the export
so that the graph keeps the module structure (the legacy graphs compute a shared subexpression twice where the
source does).

Writes, into --out-dir:
  model.onnx            the canonical QONNX file (opset 17, input "x" NCHW, output "logits")
  export.json           checkpoint / ONNX SHA-256, exporter and versions, the export hook and its replacements,
                        the binarizer choice, the import arguments to use, the check-qonnx result, the logits files below
  source_logits.f32 / source_logits_f64.f64        the *source* model (authors' code, unmodified), float32 / float64
  exportable_logits.f32 / exportable_logits_f64.f64  the exportable copy in torch (checks the hook in isolation)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "python"))

OPSET = 17
# import arguments for the legacy PyTorch converters' file layout (graph names = module paths, data in first-use order,
# one ChannelAffine per affine source node); every computed value is the same with the importer defaults
IMPORT_ARGS_TORCH = ["--naming", "exporter", "--constant-order", "first-use", "--affine", "keep", "--cleanup", "none"]
QONNX_DOMAIN = "qonnx.custom_op.general"
# step 6: the import layout is model metadata (bitino.import {"mapping": "patterns", ...}, docs/qonnx-profile.md § 6),
# so the file alone determines its .bitino; an import argument that contradicts it is an error
_IMPORT_FLAGS = {"--naming": "naming", "--constant-order": "constant_order", "--affine": "affine", "--cleanup": "cleanup",
                 "--flatten-dense": "flatten_dense", "--bit-outputs": "bit_outputs"}


def import_meta(import_args: list) -> str:
    """bitino.import of a pattern-mapped file from the importer arguments of its layout (the values the import
    arguments of steps 1 / 1b gave; every option is written, the importer defaults included)."""
    opts = {"naming": "index", "affine": "fold", "constant_order": "walk", "cleanup": "qonnx", "bit_outputs": "set",
            "flatten_dense": "transpose"}
    it = iter(import_args)
    for flag in it:
        if flag not in _IMPORT_FLAGS:
            raise AssertionError(f"import argument {flag} has no bitino.import option")
        opts[_IMPORT_FLAGS[flag]] = next(it)
    return json.dumps({"mapping": "patterns", **opts}, sort_keys=True)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================================================ exportable modules
def _modules():
    """Exportable building blocks (defined lazily: torch is imported only here)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BipolarQuantFn(torch.autograd.Function):
        """qonnx BipolarQuant(x, scale): x >= 0 -> +scale, else -scale (forward only; export emits the QONNX op)."""

        @staticmethod
        def forward(ctx, x, scale):
            return torch.where(x >= 0, scale, -scale)

        @staticmethod
        def symbolic(g, x, scale):
            y = g.op(f"{QONNX_DOMAIN}::BipolarQuant", x, scale)
            y.setType(x.type())
            return y

    class PadFn(torch.autograd.Function):
        """Constant Pad with static pads (ONNX order [x1_begin, ..., x1_end, ...]) as one ONNX Pad node (F.pad exports a
        ConstantOfShape subgraph)."""

        @staticmethod
        def forward(ctx, x, pads, value):
            n = x.dim()
            tp = []
            for i in reversed(range(n)):
                tp += [pads[i], pads[i + n]]
            return F.pad(x, tuple(tp), value=value)

        @staticmethod
        def symbolic(g, x, pads, value):
            return g.op("Pad", x, g.op("Constant", value_t=torch.tensor(list(pads), dtype=torch.int64)),
                        g.op("Constant", value_t=torch.tensor(float(value), dtype=torch.float32)), mode_s="constant")

    class Binarize(nn.Module):
        """Activation binarizer: mode 'sign_zero' = torch.sign (ONNX Sign), 'ge' = BipolarQuant(x, 1)."""

        def __init__(self, mode: str):
            super().__init__()
            if mode not in ("sign_zero", "ge"):
                raise ValueError(mode)
            self.mode = mode
            self.register_buffer("one", torch.tensor(1.0))

        def forward(self, x):
            return torch.sign(x) if self.mode == "sign_zero" else BipolarQuantFn.apply(x, self.one)

    def carrier(arr, shape) -> "torch.Tensor":
        """A constant as a torch tensor: a float64 array stays float64 (a float64 carrier, docs/qonnx-profile.md R7:
        exported as a DOUBLE initializer read through Cast(to=FLOAT)); a float32 array stays float32."""
        a = np.asarray(arr)
        if a.dtype not in (np.float32, np.float64):
            raise TypeError(a.dtype)
        return torch.from_numpy(np.ascontiguousarray(a.reshape(shape)))

    class _Consts(nn.Module):
        """Constants as buffers in their own dtype: a float64 carrier is a DOUBLE initializer, cast to the activation
        dtype at use (Cast(DOUBLE -> FLOAT) in the export, a no-op in the float64 check run). The module must not be
        converted with .to(float32) (run_torch evaluates a deep copy)."""

        def c(self, name, x):
            t = getattr(self, "c_" + name)
            return t if t.dtype == x.dtype else t.to(x.dtype)

        def setc(self, name, arr, shape):
            self.register_buffer("c_" + name, carrier(arr, shape))

    class Bias(_Consts):
        """x + t[c] (LearnableBias, RSign threshold, RPReLU shifts). t: a float32 parameter as is, or a float64 value
        (e.g. bias / beta) as a float64 carrier (Bitino stores its float32 rounding)."""

        def __init__(self, t: np.ndarray):
            super().__init__()
            self.setc("t", t, (1, -1, 1, 1))

        def forward(self, x):
            return x + self.c("t", x)

    class Affine(_Consts):
        """x * s + t (BatchNorm in eval mode): s, t folded in float64 and exported as float64 carriers
        (docs/qonnx-profile.md R6 / R7)."""

        def __init__(self, s64: np.ndarray, t64: np.ndarray, rank: int):
            super().__init__()
            shape = (1, -1) + (1,) * (rank - 2)
            self.setc("s", np.asarray(s64, np.float64), shape)
            self.setc("t", np.asarray(t64, np.float64), shape)

        def forward(self, x):
            return x * self.c("s", x) + self.c("t", x)

    class PReLU(nn.Module):
        def __init__(self, slope: np.ndarray):
            super().__init__()
            self.register_buffer("slope", torch.from_numpy(np.asarray(slope, np.float32).reshape(-1).copy()))

        def forward(self, x):
            return F.prelu(x, self.slope)

    class FloatConv(nn.Module):
        """Conv with a float32 weight (OIHW) computed in float64 and rounded once."""

        def __init__(self, w: np.ndarray, stride, padding, dilation=1, groups=1, bias: np.ndarray | None = None):
            super().__init__()
            self.register_buffer("weight", torch.from_numpy(np.asarray(w, np.float32).copy()))
            self.register_buffer("bias", None if bias is None else torch.from_numpy(np.asarray(bias, np.float32).copy()))
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

        def forward(self, x):
            return F.conv2d(x, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

    class BinaryWeightConv(_Consts):
        """Canonical binary-weight conv (docs/qonnx-profile.md R4): Conv(x, BipolarQuant(W_latent, 1)) -> Mul(alpha[o]).
        W_latent = fl32 of the latent (no exact zeros and no sign change in float32: checked, so the ±1 weights are
        the source's torch.sign(W)); alpha = the per-output-channel scale with its sign (the source's bit polarity),
        a float64 carrier when computed in float64, else the float32 parameter. The importer folds the Mul of the
        same naming group into the binary operator's post_mul. Zero padding (Conv pads; the profile rewrite turns
        it into an explicit Pad of the binarized input)."""

        def __init__(self, sign: np.ndarray, latent: np.ndarray, scale: np.ndarray, stride, padding, dilation=1, groups=1,
                     alpha_inside: bool = False, bias: np.ndarray | None = None):
            super().__init__()
            self.alpha_inside = alpha_inside      # a float (not binarized) input: Conv(x, BipolarQuant(W, fl32(alpha)))
            lat = np.asarray(latent).astype(np.float32)
            if np.any(lat == 0) or not np.array_equal(np.sign(lat).astype(np.int8), np.asarray(sign, np.int8)):
                raise NotImplementedError("binary weight latent has exact zeros or a sign change in float32")
            self.register_buffer("w_latent", torch.from_numpy(lat.copy()))
            self.register_buffer("one", torch.tensor(1.0))
            self.has_alpha = scale is not None     # None: no scaling factor (BNext), no Mul
            if alpha_inside:   # the weights of a float conv: sign(W) * fl32(alpha) per output channel (the legacy kernel)
                self.register_buffer("alpha_w", torch.from_numpy(np.asarray(scale).astype(np.float32).reshape(-1, 1, 1, 1).copy()))
            elif self.has_alpha:
                self.setc("alpha", np.asarray(scale), (1, -1, 1, 1))
            self.has_bias = bias is not None
            if self.has_bias:
                if alpha_inside:
                    raise NotImplementedError("bias with alpha_inside")
                self.setc("bias", np.asarray(bias), (1, -1, 1, 1))
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

        def forward(self, x):
            if self.alpha_inside:
                return F.conv2d(x, BipolarQuantFn.apply(self.w_latent, self.alpha_w), None, self.stride, self.padding,
                                self.dilation, self.groups)
            w = BipolarQuantFn.apply(self.w_latent, self.one)
            y = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
            if self.has_alpha:
                y = y * self.c("alpha", y)
            return y + self.c("bias", y) if self.has_bias else y

    return dict(BipolarQuantFn=BipolarQuantFn, PadFn=PadFn, Binarize=Binarize, Bias=Bias, PReLU=PReLU, FloatConv=FloatConv,
                BinaryWeightConv=BinaryWeightConv, Affine=Affine, Consts=_Consts, carrier=carrier)


def bn_affine64(bn) -> tuple[np.ndarray, np.ndarray]:
    """BatchNorm1d/2d (eval, running statistics) as y = s*x + t in float64, exactly as the legacy adapters fold it
    (torch_rbonn / torch_irnet ``bn``): s = w / sqrt(var + eps), t = b - mean * s, eps the module's double."""
    if type(bn).__name__ not in ("BatchNorm1d", "BatchNorm2d") or not bn.affine or bn.running_mean is None:
        raise NotImplementedError(f"unsupported norm layer {bn}")
    f = lambda t: t.detach().cpu().double().reshape(-1).numpy()   # noqa: E731
    s = f(bn.weight) / np.sqrt(f(bn.running_var) + float(bn.eps))
    return s, f(bn.bias) - f(bn.running_mean) * s


def export_copy(m):
    """Deep copy of a standard module with every BatchNorm replaced by Affine (x * s + t, float64 fold): ONNX's
    BatchNormalization epsilon is a float32 attribute and cannot carry PyTorch's double eps (1e-5) exactly."""
    import torch.nn as nn

    if m is None:
        return None
    A = _modules()["Affine"]
    if type(m).__name__ in ("BatchNorm1d", "BatchNorm2d"):
        return A(*bn_affine64(m), 2 if type(m).__name__ == "BatchNorm1d" else 4)
    if type(m).__name__ == "Sequential":
        return nn.Sequential(*[export_copy(c) for c in m])
    if any(type(c).__name__.startswith("BatchNorm") for c in m.modules()):
        raise NotImplementedError(f"BatchNorm nested in {type(m).__name__}")
    return copy.deepcopy(m)


# ================================================================================================ families
class Family:
    """Export hook of one PyTorch family. Subclasses: model_ids(), load(), preprocess(), source_logits(),
    exportable()."""
    name = ""
    converter = ""
    source_binarizer = "sign_zero"      # what the authors' forward computes at every activation binarizer

    def model_ids(self) -> list:
        raise NotImplementedError

    def load(self, model_id: str, repo: str, weights: str):
        raise NotImplementedError

    def preprocess(self, model_id: str) -> str:
        raise NotImplementedError

    def source_logits(self, model, x_nhwc: np.ndarray, dtype) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def exportable(self, model, binarizer: str):
        raise NotImplementedError

    def input_domain(self, model_id: str) -> str:
        return "float32"

    def labels(self, model_id: str, em) -> str:
        """bitino.labels: the class set of the model's check set (registry/eval_sets.json name)."""
        name = (em or {}).get("name", "")
        for key, n, label_set in (("imagenet", 1000, "imagenet-1k"), ("cifar100", 100, "cifar100"), ("cifar10", 10, "cifar10")):
            if key in name:
                return json.dumps({"num_classes": n, "set": label_set})
        return "none"


class AbBnn(Family):
    """AB-BNN (Ruichen0424/AB-BNN) ReActNet-18 / ReActNet-34 (Network/ReActNet_18_Qaw.py, BNN.py), as
    tools/convert_abbnn.py loads them. Mirrors bitino.adapters.torch_abbnn.convert (ResNet path):

      BasicBlock.forward  out = conv1(sign(move0(x, beta) * beta)); out = out * alpha + residual
                          out = move2(prelu(move1(out)))
      exportable          out = conv1(Bin(x + fl32(bias / beta)))   [beta > 0: sign(y * beta) = sign(y), exact for
                          every y except |y| < 2^-126 * 2^-23/beta (float32 underflow of y * beta); the legacy adapter
                          drops beta the same way]
                          conv1 = Conv(., BipolarQuant(fl32(W_std), fl32(gain * mean|W_std| * alpha)))  [the legacy post_mul]
                          out = out + residual;  downsample = AvgPool2d + Conv(., BipolarQuant(fl32(W_std), fl32(gain * mean|W_std|)))
                          move1 / move2 = Add(x, bias);  prelu = PRelu(x, 2^round(L_alpha))
      stem                Conv(x, fl32(gain * scale * (W - mean) / (std + eps)))  [float64 fold]
      head                AdaptiveAvgPool2d(1) (GlobalAveragePool) + flatten + Linear (Gemm)
    """
    name = "abbnn"
    converter = "tools/convert_abbnn.py"

    def _cv(self):
        import convert_abbnn
        return convert_abbnn

    def model_ids(self):
        return [m for m, a in self._cv().ARCHS.items() if a[1] in ("ResNet18", "ResNet34")]

    def load(self, model_id, repo, weights):
        model, cls, src = self._cv().load_model(repo, model_id, weights)
        return model, {"class": cls, "source_sha256": src, "repo": "https://github.com/Ruichen0424/AB-BNN",
                       "repo_commit": self._cv().repo_commit(repo)}

    def preprocess(self, model_id):
        return self._cv().preprocess_of(model_id)

    def source_logits(self, model, x_nhwc, dtype):
        return self._cv().reference_outputs(model, x_nhwc, dtype)

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn

        from bitino.adapters import torch_abbnn as A

        M = _modules()
        cls = type(model).__name__
        if cls != "ResNet":
            raise NotImplementedError(f"AB-BNN export hook: only the ReActNet-18/34 ResNet class (got {cls})")
        pair = A._pair
        rec = []

        def binary_conv(bc, extra_scale: float, path: str, float_input: bool = False):
            parts = A.binary_conv_parts(bc)                     # raises on exact zeros of W_std in float64
            lat64 = A.standardized_weight(bc, torch.float64).numpy()
            s = parts["scale64"] * extra_scale
            rec.append({"module": path, "source": "HardBinaryScaledStdConv2d",
                        "export": "Conv(x, BipolarQuant(fl32(W_std), fl32(gain*mean|W_std|" + ("*alpha" if extra_scale != 1.0 else "") + ")))",
                        **{k: parts[k] for k in ("zeros_f32", "zeros_f64", "sign_mismatch_f32_vs_f64")}})
            return M["BinaryWeightConv"](parts["sign"], lat64, s, int(bc.stride), int(bc.padding), alpha_inside=float_input)

        class Block(nn.Module):
            def __init__(self, blk, path):
                super().__init__()
                beta, alpha = float(blk.beta), float(blk.alpha)
                if not beta > 0:
                    raise NotImplementedError(f"{path}: beta must be > 0")
                for m, want in (("act1", "BinaryActivation"), ("conv1", "HardBinaryScaledStdConv2d"), ("prelu", "Q_PReLU"),
                                ("move0", "LearnableBias"), ("move1", "LearnableBias"), ("move2", "LearnableBias")):
                    if type(getattr(blk, m)).__name__ != want:
                        raise NotImplementedError(f"{path}.{m}: expected {want}")
                b64 = lambda lb, d: (lb.bias.detach().to(torch.float64).reshape(-1) / d).numpy()   # noqa: E731
                self.move0 = M["Bias"](b64(blk.move0, beta))
                self.act1 = M["Binarize"](binarizer)
                self.conv1 = binary_conv(blk.conv1, alpha, f"{path}.conv1")
                self.downsample = None
                if blk.downsample is not None:
                    ap, dc = blk.downsample[0], blk.downsample[1]
                    if type(ap).__name__ != "AvgPool2d" or ap.ceil_mode or ap.divisor_override is not None or pair(ap.padding) != (0, 0):
                        raise NotImplementedError(f"{path}.downsample.0: unsupported {ap}")
                    self.downsample = nn.Sequential(copy.deepcopy(ap), binary_conv(dc, 1.0, f"{path}.downsample.1", float_input=True))
                self.move1 = M["Bias"](b64(blk.move1, 1.0))
                self.prelu = M["PReLU"](A.qprelu_slope(blk.prelu))
                self.move2 = M["Bias"](b64(blk.move2, 1.0))
                rec.append({"module": path, "source": "BasicBlock (move0 -> sign(.*beta) -> conv1 * alpha + residual -> move1 -> Q_PReLU -> move2)",
                            "export": f"Add(t=fl32(bias/beta)) -> {'Sign' if binarizer == 'sign_zero' else 'BipolarQuant'} -> Conv(±scale*alpha) "
                                      "+ residual -> Add -> PRelu(2^round(L_alpha)) -> Add", "beta": beta, "alpha": alpha})

            def forward(self, x):
                out = self.conv1(self.act1(self.move0(x)))
                residual = x if self.downsample is None else self.downsample(x)
                out = out + residual
                return self.move2(self.prelu(self.move1(out)))

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                c1 = src.conv1
                if type(c1).__name__ != "ScaledStdConv2d" or c1.bias is not None or c1.groups != 1:
                    raise NotImplementedError("conv1: expected ScaledStdConv2d without bias / groups")
                self.conv1 = M["FloatConv"](A.fold_scaled_std_conv(c1), c1.stride, c1.padding, c1.dilation)
                rec.append({"module": "conv1", "source": "ScaledStdConv2d", "export": "Conv(x, fl32(gain*scale*(W-mean)/(std+eps)))"})
                self.maxpool = copy.deepcopy(src.maxpool)
                for ln in ("layer1", "layer2", "layer3", "layer4"):
                    setattr(self, ln, nn.Sequential(*[Block(b, f"{ln}.{i}") for i, b in enumerate(getattr(src, ln))]))
                self.avgpool = copy.deepcopy(src.avgpool)
                self.fc = copy.deepcopy(src.fc)

            def forward(self, x):
                x = self.maxpool(self.conv1(x))
                x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
                return self.fc(torch.flatten(self.avgpool(x), 1))

        # every module of the source must be of a known type (the hook mirrors the adapter's coverage check)
        known = {"ResNet", "Sequential", "BasicBlock", "ScaledStdConv2d", "HardBinaryScaledStdConv2d", "LearnableBias",
                 "BinaryActivation", "Q_PReLU", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "Linear", "Identity"}
        for n, m in model.named_modules():
            if type(m).__name__ not in known:
                raise NotImplementedError(f"module {n or '<root>'}: unsupported type {type(m).__name__}")
        net = Net(model).eval()
        return net, {"replacements": rec, "import_args": IMPORT_ARGS_TORCH}


class TorchCnnFamily(Family):
    """A family loaded by tools/convert_torch_cnn.py (FAMILIES[key]): load / preprocess / source logits as there."""
    name = ""
    keys: tuple = ()

    def _cv(self):
        import convert_torch_cnn
        return convert_torch_cnn

    def _key(self, model_id: str) -> str:
        for k in self.keys:
            if model_id in self._cv().FAMILIES[k]["archs"]:
                return k
        raise KeyError(model_id)

    def model_ids(self):
        return sorted(m for k in self.keys for m in self._cv().FAMILIES[k]["archs"])

    def load(self, model_id, repo, weights):
        cv = self._cv()
        self._mid = model_id
        k = self._key(model_id)
        model, cls, src = cv.load_model(k, repo, model_id, weights)
        return model, {"class": cls, "source_sha256": src, "repo": cv.FAMILIES[k]["repo_url"], "repo_commit": cv.repo_commit(repo)}

    def preprocess(self, model_id):
        return self._cv().FAMILIES[self._key(model_id)]["preprocess"]

    def source_logits(self, model, x_nhwc, dtype):
        """tools/convert_torch_cnn.py main: float32 with the binary weights the source feeds to F.conv2d captured;
        float64 with the fp32-derived binary weights substituted (IR-Net: exactly ±1; others: sign(fp32) x |fp64|)."""
        import torch

        cv = self._cv()
        fam = cv.FAMILIES[self._key(self._mid)]
        sm, cm = fam.get("sign_module"), fam.get("conv_module")
        ge = fam.get("binarize_modes", ("ge", "sign_zero")) == ("ge",)
        if not cm:
            return cv.reference_outputs(model, x_nhwc, dtype, sm, cm, ge=ge)
        if dtype == torch.float32:
            self._cap32 = {}
            return cv.reference_outputs(model, x_nhwc, dtype, sm, cm, capture=self._cap32, ge=ge)
        cap32 = getattr(self, "_cap32", None)
        if not cap32:
            raise RuntimeError("source_logits(float64) needs the float32 run first (captured binary weights)")
        if "irnet" in fam and not fam.get("scaled_weights"):
            override = {k: v.double() for k, v in cap32.items()}
        else:
            cap64: dict = {}
            cv.reference_outputs(model, x_nhwc[:1], dtype, sm, cm, capture=cap64, ge=ge)
            override = {k: torch.sign(cap32[k]).double() * cap64[k].abs() for k in cap32}
        return cv.reference_outputs(model, x_nhwc, dtype, sm, cm, override=override, ge=ge)


class Irnet(TorchCnnFamily):
    """IR-Net family (ReCU, RBNN, SiMaN, BiPer: one shared model zoo; ShuaiZ Bi-Real-20), as tools/convert_irnet.py
    (convert_torch_cnn families recu / rbnn / siman / biper, *_cifar, shuaiz) loads it. Mirrors
    bitino.adapters.torch_irnet.convert node for node, in its creation order (the IR order and the Graph.unique()
    counter), for every topology it accepts:

      ImageNet ResNet-18/34   stem Conv 7x7 + BN + act + MaxPool; BasicBlock out = act1(BN1(bconv1(x)) + downsample(x)),
                              out = act2(BN2(bconv2(out)) + out); head AvgPool2d(7) -> GlobalAveragePool + BN1d + Linear
      CIFAR ResNet-20 / -18   standard / bireal / bireal_pre blocks, shortcuts A (x[:, :, ::2, ::2] + channel zero pad
                              -> AveragePool(1, stride 2) + Pad), avgpad (AvgPool2d(2) + Pad), bconv (BinarizeConv2d 1x1 + BN)
      VGG-Small               conv0 + BN + act; conv1..5 (pool after 1, 3, 5) + BN + act; view(N, -1) + Linear
      Bi-Real-20 (ShuaiZ)     BiRealConv = Conv(sign(x), sign(W)) * mean|W| + bias; out = BN(conv(x)) + shortcut A

      BinarizeConv2d eval     F.conv2d(sign(a'), bw) * alpha -> Conv(Bin(a'), BipolarQuant(bw, 1)) -> Mul(alpha[o])
                              (bw: the adapter's binary_weight(), the source's own fp32 expression, exactly ±1, checked;
                              alpha: the fp32 parameter)
      RBNN / SiMaN centering  a' = (a - mean_CHW a) / std (RBNN) or / sqrt(var + 1e-5) (SiMaN): the positive divisor does
                              not change the sign (the adapter drops it) -> ReduceMean(a, [1, 2, 3]) -> Sub(a, mean)
                              (IR: Reshape + ReduceMean + ChannelAffine(-1) + AddBroadcast, the adapter's four nodes)
      BatchNorm               Mul + Add of float64 carriers (the adapter's float64 fold)
      Hardtanh / F.hardtanh   Clip(-1, 1);  PReLU -> PRelu;  F.relu -> Relu

    Names: every node carries the adapter's IR name (module path, ``<path>/flatten`` ... for the centering, and
    Graph.unique() names ``nonlinear_1``, ``layer1.0.hardtanh_2``, ... drawn in the adapter's order); the counter's
    final value goes to ``bitino.ir_unique_counter`` (the fold pass's constant names continue it). Every binary conv
    carries ``bitino.attrs`` pad_value 0.0 (the adapter writes the float form also for unpadded 1x1 convs)."""
    name = "irnet"
    converter = "tools/convert_irnet.py (tools/convert_torch_cnn.py IR-Net families)"
    keys = ("recu", "rbnn", "siman", "biper", "recu_cifar", "rbnn_cifar", "siman_cifar", "biper_cifar", "shuaiz")

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from bitino.adapters import torch_irnet as I

        M = _modules()
        BQ = M["BipolarQuantFn"]
        cv = self._cv()
        fam = cv.FAMILIES[self._key(self._mid)]
        family = fam["irnet"]
        spec = I.FAMILIES[family]
        hw = tuple(fam.get("input_hw", (224, 224)))
        cls = type(model).__name__
        pair = I._pair
        rec = []
        counter = [0]

        def unique(prefix: str) -> str:            # Graph.unique, in the adapter's call order
            counter[0] += 1
            return f"{prefix}_{counter[0]}"

        def named(mod, name, **kw):
            mod.bitino_meta = {"name": name, **kw}
            return mod

        class Act(nn.Module):
            def __init__(self, kind, name, slope=None):
                super().__init__()
                self.kind = kind
                self.bitino_meta = {"name": name}
                if slope is not None:
                    self.register_buffer("slope", torch.from_numpy(np.asarray(slope, np.float32).reshape(-1).copy()))

            def forward(self, x):
                if self.kind == "hardtanh":
                    return F.hardtanh(x, -1.0, 1.0)
                if self.kind == "relu":
                    return F.relu(x)
                return F.prelu(x, self.slope)

        def act(m, name, channels, expect=None):     # adapter act(): Hardtanh -> unique name, PReLU -> fixed name
            want = expect or spec["block_act"]
            if type(m).__name__ != want:
                raise NotImplementedError(f"{name}: expected {want}, got {type(m).__name__}")
            if want == "Hardtanh":
                if float(m.min_val) != -1.0 or float(m.max_val) != 1.0:
                    raise NotImplementedError(f"{name}: expected Hardtanh(-1, 1)")
                return Act("hardtanh", unique(name))
            slope = m.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
            if slope.size not in (1, channels):
                raise NotImplementedError(f"{name}: {slope.size} slopes for {channels} channels")
            return Act("prelu", name, np.broadcast_to(slope, (channels,)).copy())

        def fact(kind, name):                       # adapter fact(): functional activation, unique name
            return Act(kind, unique(name))

        class Add2(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.bitino_meta = {"name": name}

            def forward(self, a, b):
                return a + b

        class BinConv(nn.Module):
            """[centering] -> Bin -> Conv(., BipolarQuant(bw, 1)) -> Mul(alpha) (-> Add(bias): BiRealConv)."""

            def __init__(self, m, name):
                super().__init__()
                self.center = bool(spec["center"])
                meta_conv = {"name": name, "attrs": {"pad_value": 0.0}}
                if family == "shuaiz":
                    if type(m).__name__ != "BiRealConv" or m.groups != 1 or m.padding_mode != "zeros":
                        raise NotImplementedError(f"{name}: unexpected BiRealConv")
                    w = m.weight.detach().cpu()
                    if int((w == 0).sum()):
                        raise NotImplementedError(f"{name}: latent weights exactly 0")
                    bw = torch.sign(w)
                    w64 = w.to(torch.float64)       # alpha = mean|W| (nested means, as the source), float64 carrier
                    alpha = torch.mean(torch.mean(torch.mean(abs(w64), dim=3, keepdim=True), dim=2, keepdim=True),
                                       dim=1, keepdim=True).reshape(1, -1, 1, 1)
                    self.register_buffer("bias", None if m.bias is None else m.bias.detach().reshape(1, -1, 1, 1).clone())
                    src = "BiRealConv: Conv(sign(x), sign(W)) * mean|W| (float64 carrier) + bias"
                else:
                    if type(m).__name__ != "BinarizeConv2d" or m.bias is not None or m.groups != 1 or pair(m.dilation) != (1, 1) \
                            or m.padding_mode != "zeros":
                        raise NotImplementedError(f"{name}: unexpected BinarizeConv2d settings")
                    if family == "rbnn":
                        a = I.source_args(m)
                        if a is None or getattr(a, "a32", False):
                            raise NotImplementedError(f"{name}: RBNN args.a32 is set (W1A32)")
                    bw = I.binary_weight(family, m).detach()
                    alpha = m.alpha.detach().reshape(1, -1, 1, 1).clone()
                    if alpha.numel() != bw.shape[0]:
                        raise NotImplementedError(f"{name}: alpha shape")
                    self.register_buffer("bias", None)
                    src = "BinarizeConv2d: F.conv2d(sign(a), bw) * alpha"
                if not bool(torch.all(bw.abs() == 1)):
                    raise NotImplementedError(f"{name}: binary weights not all +-1")
                self.register_buffer("bw", bw.clone().float())
                self.register_buffer("alpha", alpha)
                self.register_buffer("one", torch.tensor(1.0))
                self.stride, self.padding = pair(m.stride), pair(m.padding)
                by_op = {}
                if self.center:
                    by_op = {"ReduceMean": {"name": f"{name}/flatten", "extra_names": [f"{name}/mean"]},
                             "Sub": {"name": f"{name}/neg_mean", "extra_names": [f"{name}/center"],
                                     "consts": {"1": f"{name}/minus_one"}}}
                self.bitino_meta = {"by_op": by_op, "default": meta_conv}
                rec.append({"module": name, "source": src + (" on sign(a - mean_CHW a)" if self.center else ""),
                            "export": ("ReduceMean -> Sub -> " if self.center else "") +
                                      f"{'Sign' if binarizer == 'sign_zero' else 'BipolarQuant'} -> Conv(BipolarQuant(bw, 1)) -> Mul(alpha)"
                                      + (" -> Add(bias)" if self.bias is not None else "")})

            def forward(self, a):
                if self.center:
                    a = a - torch.mean(a, dim=(1, 2, 3), keepdim=True)
                ba = torch.sign(a) if binarizer == "sign_zero" else BQ.apply(a, self.one)
                y = F.conv2d(ba, BQ.apply(self.bw, self.one), None, self.stride, self.padding) * self.alpha.to(a.dtype)
                return y if self.bias is None else y + self.bias

        def bn(m, name, cls="BatchNorm2d"):
            if type(m).__name__ != cls:
                raise NotImplementedError(f"{name}: expected {cls}")
            return named(export_copy(m), name)

        def fconv(m, name):
            if type(m).__name__ != "Conv2d" or m.bias is not None or m.groups != 1 or pair(m.dilation) != (1, 1) \
                    or m.padding_mode != "zeros":
                raise NotImplementedError(f"{name}: unexpected Conv2d")
            return named(copy.deepcopy(m), name)

        class Shortcut(nn.Module):
            """Parameter-free shortcut A (x[:, :, ::2, ::2] + channel zero pad) / avgpad (AvgPool2d(2) + pad),
            checked numerically against the module as the adapter does."""

            def __init__(self, sc, kind, pth, cin, cout, h, w):
                super().__init__()
                probe = torch.randn(1, cin, h, w, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
                self.pad = (cout - cin) // 2
                with torch.no_grad():
                    got = copy.deepcopy(sc).double()(probe) if kind == "avgpad" else sc(probe)
                    want = F.pad((probe[:, :, ::2, ::2] if kind == "A" else F.avg_pool2d(probe, 2)), (0, 0, 0, 0, self.pad, self.pad))
                if tuple(got.shape) != tuple(want.shape) or not bool(torch.equal(got, want)):
                    raise NotImplementedError(f"{pth}.shortcut does not match option {kind}")
                k = 1 if kind == "A" else 2
                self.pool = named(nn.AvgPool2d(k, 2, 0, count_include_pad=True),
                                  f"{pth}.shortcut.{'subsample' if kind == 'A' else 'avgpool'}")
                self.padm = named(PadC(self.pad), f"{pth}.shortcut.pad")
                rec.append({"module": f"{pth}.shortcut", "source": "x[:, :, ::2, ::2] + F.pad channels" if kind == "A" else
                            "AvgPool2d(2) + F.pad channels", "export": f"AveragePool({k}, stride 2) + Pad(channels {self.pad})"})

            def forward(self, x):
                y = self.pool(x)
                return self.padm(y) if self.pad else y

        class PadC(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def forward(self, x):
                return M["PadFn"].apply(x, (0, self.p, 0, 0, 0, self.p, 0, 0), 0.0)

        class Ident(nn.Module):
            def forward(self, x):
                return x

        def shortcut(blk, pth, cin, cout, stride, h, w, kind):
            sc = blk.shortcut if hasattr(blk, "shortcut") else blk.downsample
            if sc is None or (type(sc).__name__ == "Sequential" and len(sc) == 0):
                if not (stride == 1 and cin == cout):
                    raise NotImplementedError(f"{pth}: identity shortcut with a shape change")
                return Ident()
            if kind == "bconv":
                if not (type(sc).__name__ == "Sequential" and len(sc) == 2 and type(sc[0]).__name__ == "BinarizeConv2d"
                        and pair(sc[0].kernel_size) == (1, 1) and pair(sc[0].padding) == (0, 0)):
                    raise NotImplementedError(f"{pth}.shortcut: expected BinarizeConv2d 1x1 + BN")
                return nn.Sequential(BinConv(sc[0], f"{pth}.shortcut.0"), bn(sc[1], f"{pth}.shortcut.1"))
            return Shortcut(sc, kind, pth, cin, cout, h, w)

        class Seq(nn.Module):
            """Modules called in registration order; forward given by a closure (the adapter's dataflow)."""

            def __init__(self, fwd, **mods):
                super().__init__()
                for k, v in mods.items():
                    setattr(self, k, v)
                self._fwd = fwd

            def forward(self, x):
                return self._fwd(self, x)

        cifar_spec, _ = I._cifar_spec(family, model) if cls == "ResNet" and not hasattr(model, "maxpool") else (None, None)
        body = []      # (attribute name, module) in creation order
        if family == "shuaiz":
            if cls != "BiRealNetCifar10":
                raise NotImplementedError(f"unexpected model {cls}")
            conv1, bn1 = fconv(model.conv1, "conv1"), bn(model.bn1, "bn1")
            h, w_, c = hw[0], hw[1], model.conv1.out_channels
            blocks = []
            for lname in ("layer1", "layer2", "layer3"):
                for bi, blk in enumerate(getattr(model, lname)):
                    pth = f"{lname}.{bi}"
                    if type(blk).__name__ != "BasicBlock":
                        raise NotImplementedError(f"{pth}: unexpected block")
                    cout, st = blk.binary_conv.out_channels, pair(blk.binary_conv.stride)[0]
                    mods = dict(binary_conv=BinConv(blk.binary_conv, f"{pth}.binary_conv"), bn1=bn(blk.bn1, f"{pth}.bn1"),
                                shortcut=shortcut(blk, pth, c, cout, st, h, w_, "A"), add=Add2(f"{pth}.add"))
                    blocks.append(Seq(lambda s, x: s.add(s.bn1(s.binary_conv(x)), s.shortcut(x)), **mods))
                    h, w_, c = (h - 1) // st + 1, (w_ - 1) // st + 1, cout
            if type(model.avgpool).__name__ != "AdaptiveAvgPool2d" or pair(model.avgpool.output_size) != (1, 1):
                raise NotImplementedError("avgpool")
            head = dict(avgpool=named(nn.AdaptiveAvgPool2d(1), "avgpool"), fc=named(copy.deepcopy(model.fc), "fc"))
            net = Seq(lambda s, x: s.fc(torch.flatten(s.avgpool(s.blocks(s.bn1(s.conv1(x)))), 1)),
                      conv1=conv1, bn1=bn1, blocks=nn.Sequential(*blocks), **head)
        elif cifar_spec is not None:
            cs = cifar_spec
            stem = [fconv(model.conv1, "conv1"), bn(model.bn1, "bn1")]
            if cs["stem"]:
                stem.append(fact(cs["stem"], "stem_" + cs["stem"]))
            h, w_, c = hw[0], hw[1], model.conv1.out_channels
            blocks = []
            layers = [n for n in ("layer1", "layer2", "layer3", "layer4") if hasattr(model, n)]
            for lname in layers:
                for bi, blk in enumerate(getattr(model, lname)):
                    pth = f"{lname}.{bi}"
                    if type(blk).__name__ not in ("BasicBlock_1w1a", "BasicBlock") or type(blk.conv1).__name__ != "BinarizeConv2d":
                        raise NotImplementedError(f"{pth}: unexpected block {type(blk).__name__}")
                    cout, st = blk.conv1.out_channels, pair(blk.conv1.stride)[0]
                    ho, wo = (h - 1) // st + 1, (w_ - 1) // st + 1
                    m1 = dict(conv1=BinConv(blk.conv1, f"{pth}.conv1"), bn1=bn(blk.bn1, f"{pth}.bn1"),
                              shortcut=shortcut(blk, pth, c, cout, st, h, w_, cs["shortcut"]))
                    if cs["block"] == "standard":
                        m1["a1"] = fact("hardtanh", f"{pth}.hardtanh1")
                        m1.update(conv2=BinConv(blk.conv2, f"{pth}.conv2"), bn2=bn(blk.bn2, f"{pth}.bn2"), add=Add2(f"{pth}.add"))
                        m1["a2"] = fact("hardtanh", f"{pth}.hardtanh2")

                        def fwd(s, x):
                            u = s.bn1(s.conv1(x))
                            res = s.shortcut(x)
                            t = s.a1(u)
                            return s.a2(s.add(s.bn2(s.conv2(t)), res))
                    elif cs["block"] == "bireal_pre":
                        m1["add1"] = Add2(f"{pth}.add1")
                        m1["a1"] = fact("hardtanh", f"{pth}.hardtanh")
                        m1.update(conv2=BinConv(blk.conv2, f"{pth}.conv2"), bn2=bn(blk.bn2, f"{pth}.bn2"), add2=Add2(f"{pth}.add2"))
                        m1["a2"] = act(blk.prelu, f"{pth}.prelu", cout, expect="PReLU")

                        def fwd(s, x):
                            u = s.add1(s.bn1(s.conv1(x)), s.shortcut(x))
                            t = s.a1(u)
                            return s.a2(s.add2(s.bn2(s.conv2(t)), u))
                    else:   # bireal
                        m1["add1"] = Add2(f"{pth}.add1")
                        m1["a1"] = act(blk.nonlinear1, f"{pth}.nonlinear1", cout) if cs["act"] == "prelu" else fact("hardtanh", f"{pth}.hardtanh1")
                        m1.update(conv2=BinConv(blk.conv2, f"{pth}.conv2"), bn2=bn(blk.bn2, f"{pth}.bn2"), add2=Add2(f"{pth}.add2"))
                        m1["a2"] = act(blk.nonlinear2, f"{pth}.nonlinear2", cout) if cs["act"] == "prelu" else fact("hardtanh", f"{pth}.hardtanh2")

                        def fwd(s, x):
                            u = s.a1(s.add1(s.bn1(s.conv1(x)), s.shortcut(x)))
                            return s.a2(s.add2(s.bn2(s.conv2(u)), u))
                    blocks.append(Seq(fwd, **m1))
                    h, w_, c = ho, wo, cout
            if len(layers) == 4 and (h, w_) != (4, 4):
                raise NotImplementedError("ResNet-18 head: F.avg_pool2d(out, 4) expects a 4x4 map")
            fc = model.linear
            if type(fc).__name__ != "Linear" or fc.bias is None:
                raise NotImplementedError("linear")
            rec.append({"module": "avgpool", "source": "F.avg_pool2d(out, out.size()[3]) on the full map",
                        "export": "AdaptiveAvgPool2d(1) -> GlobalAveragePool (the adapter's GlobalAvgPool)"})
            net = Seq(lambda s, x: s.linear(s.bn2(torch.flatten(s.avgpool(s.blocks(s.stem(x))), 1))),
                      stem=nn.Sequential(*stem), blocks=nn.Sequential(*blocks), avgpool=named(nn.AdaptiveAvgPool2d(1), "avgpool"),
                      bn2=bn(model.bn2, "bn2", "BatchNorm1d"), linear=named(copy.deepcopy(fc), "linear"))
        elif cls in I.VGG_CLASSES:
            acts_src = ([(model.nonlinear, "nonlinear")] * 6 if family == "rbnn" else
                        [(getattr(model, f"nonlinear{i}"), f"nonlinear{i}") for i in range(6)])
            seq = [fconv(model.conv0, "conv0"), bn(model.bn0, "bn0"), act(acts_src[0][0], acts_src[0][1], model.conv0.out_channels)]
            pool = model.pooling
            if type(pool).__name__ != "MaxPool2d" or pair(pool.kernel_size) != (2, 2) or pair(pool.stride) != (2, 2) \
                    or pair(pool.padding) != (0, 0) or pool.ceil_mode:
                raise NotImplementedError("pooling: expected MaxPool2d(2, 2)")
            h, w_, c = hw[0], hw[1], model.conv0.out_channels
            for i in range(1, 6):
                m = getattr(model, f"conv{i}")
                if pair(m.stride) != (1, 1) or pair(m.padding) != (1, 1):
                    raise NotImplementedError(f"conv{i}: unexpected stride / padding")
                seq.append(BinConv(m, f"conv{i}"))
                c = m.out_channels
                if i in (1, 3, 5):     # VGG-Small forward: conv -> pooling -> bn -> nonlinear
                    seq.append(named(nn.MaxPool2d(2, 2), f"pooling_{i}"))
                    h, w_ = h // 2, w_ // 2
                seq.append(bn(getattr(model, f"bn{i}"), f"bn{i}"))
                seq.append(act(acts_src[i][0], acts_src[i][1], c))
            fc = model.fc
            if type(fc).__name__ != "Linear" or fc.bias is None or fc.in_features != c * h * w_:
                raise NotImplementedError("fc")
            rec.append({"module": "flatten", "source": "x.view(N, -1) (NCHW order) + Linear",
                        "export": "Flatten + Gemm; imported with --flatten-dense permute (the adapter permutes the fc columns to NHWC)"})
            net = Seq(lambda s, x: s.fc(torch.flatten(s.features(x), 1)), features=nn.Sequential(*seq), fc=named(copy.deepcopy(fc), "fc"))
        elif cls == "ResNet" and hasattr(model, "maxpool"):
            stem = [fconv(model.conv1, "conv1"), bn(model.bn1, "bn1")]
            if spec["stem"] == "relu":
                stem.append(Act("relu", "stem_relu"))
            else:
                stem.append(act(getattr(model, spec["stem"]), spec["stem"], 64))
            mp = model.maxpool
            if type(mp).__name__ != "MaxPool2d" or pair(mp.dilation) != (1, 1) or mp.ceil_mode:
                raise NotImplementedError("maxpool")
            stem.append(named(copy.deepcopy(mp), "maxpool"))
            blocks = []
            for lname in ("layer1", "layer2", "layer3", "layer4"):
                for bi, blk in enumerate(getattr(model, lname)):
                    pth = f"{lname}.{bi}"
                    if type(blk).__name__ != "BasicBlock":
                        raise NotImplementedError(f"{pth}: unexpected block")
                    if family == "rbnn":
                        a1, a2, n1, n2 = blk.nonlinear, blk.nonlinear, f"{pth}.nonlinear", f"{pth}.nonlinear"
                    else:
                        a1, a2, n1, n2 = blk.nonlinear1, blk.nonlinear2, f"{pth}.nonlinear1", f"{pth}.nonlinear2"
                    cout = blk.conv1.out_channels
                    m1 = dict(conv1=BinConv(blk.conv1, f"{pth}.conv1"), bn1=bn(blk.bn1, f"{pth}.bn1"))
                    if blk.downsample is not None:
                        ds = blk.downsample
                        if type(ds).__name__ != "Sequential" or len(ds) != 2:
                            raise NotImplementedError(f"{pth}.downsample: unexpected structure (avg_down not supported)")
                        m1["downsample"] = nn.Sequential(fconv(ds[0], f"{pth}.downsample.0"), bn(ds[1], f"{pth}.downsample.1"))
                    else:
                        m1["downsample"] = Ident()
                    m1["add1"] = Add2(f"{pth}.add1")
                    m1["a1"] = act(a1, n1, cout)
                    m1.update(conv2=BinConv(blk.conv2, f"{pth}.conv2"), bn2=bn(blk.bn2, f"{pth}.bn2"), add2=Add2(f"{pth}.add2"))
                    m1["a2"] = act(a2, n2, cout)

                    def fwd(s, x):
                        t = s.bn1(s.conv1(x))
                        t = s.a1(s.add1(t, s.downsample(x)))
                        return s.a2(s.add2(s.bn2(s.conv2(t)), t))
                    blocks.append(Seq(fwd, **m1))
            ap = model.avgpool
            if type(ap).__name__ != "AvgPool2d" or pair(ap.kernel_size) != (7, 7) or ap.ceil_mode or pair(ap.padding) != (0, 0):
                raise NotImplementedError("avgpool: expected AvgPool2d(7) over the full 7x7 map")
            rec.append({"module": "avgpool", "source": "AvgPool2d(7, stride 1) on the 7x7 map (224x224 input)",
                        "export": "AdaptiveAvgPool2d(1) -> GlobalAveragePool (same window, the adapter's GlobalAvgPool)"})
            net = Seq(lambda s, x: s.fc(s.bn2(torch.flatten(s.avgpool(s.blocks(s.stem(x))), 1))),
                      stem=nn.Sequential(*stem), blocks=nn.Sequential(*blocks), avgpool=named(nn.AdaptiveAvgPool2d(1), "avgpool"),
                      bn2=bn(model.bn2, "bn2", "BatchNorm1d"), fc=named(copy.deepcopy(model.fc), "fc"))
        else:
            raise NotImplementedError(f"IR-Net export hook: unsupported top-level model {cls}")
        known = {"ResNet", *I.VGG_CLASSES, "BiRealNetCifar10", "BiRealConv", "AdaptiveAvgPool2d", "Sequential", "BasicBlock",
                 "BasicBlock_1w1a", "LambdaLayer", "Conv2d", "BatchNorm2d", "BatchNorm1d", "MaxPool2d", "AvgPool2d",
                 "BinarizeConv2d", "Hardtanh", "PReLU", "Linear"}
        for n, mm in model.named_modules():
            if type(mm).__name__ not in known:
                raise NotImplementedError(f"module {n or '<root>'}: unsupported type {type(mm).__name__}")
        args = list(IMPORT_ARGS_TORCH) + (["--flatten-dense", "permute"] if cls in I.VGG_CLASSES else [])
        return net.eval(), {"replacements": rec, "import_args": args, "ir_unique_counter": counter[0]}


class Rbonn(TorchCnnFamily):
    """Bi-Real / ReActNet ResNet-18 backbones converted by bitino.adapters.torch_rbonn: RBONN (SteveTsui/RBONN
    2_step2_rbonn/rbonn.py BiRealNet / rbonn18), SNN sub-bit (yikaiw/SNN birealnet.py, HardBinaryConv with or without
    the 2^bit_num pattern decode) and the IDa-Det BiRes-18 backbone (ResNet_REACT + BiConv, ResNetREACTClassifier), as
    tools/convert_rbonn.py / convert_snn.py / convert_idadet.py (convert_torch_cnn families rbonn / snn / idadet) load
    them. Mirrors torch_rbonn.convert:

      stem          Conv2d 7x7 + BatchNorm2d + MaxPool2d
      BasicBlock    out = bn1(binary_conv(binary_activation(move0(x)))) + residual; move2(prelu(move1(out)))
        move*       LearnableBias: x + bias                          -> Add(x, bias)
        binarizer   sign(x).detach() - out3.detach() + out3 == torch.sign(x) in eval (adapter docstring)
                                                                     -> Sign (or BipolarQuant with --binarizer ge)
        binary conv RBOConv: conv(x, sign(W) * 0.02|sf|); BiConv (IDa-Det): + conv bias; HardBinaryConv (SNN):
                    conv(x, mean|W| * sign(W)) (or the decoded ±1 patterns)
                    -> Conv(Bin(x), BipolarQuant(W, 1)) -> Mul(scale[o] float64 carrier) [-> Add(bias)]
        downsample  [AvgPool2d(2, 2) (IDa-Det ceil_mode: = floor mode on the even maps, checked)] + conv1x1 + BN
        prelu       nn.PReLU(C)                                      -> PRelu
      head          AdaptiveAvgPool2d(1) + flatten + Linear
    """
    name = "rbonn"
    converter = "tools/convert_rbonn.py, convert_snn.py, convert_idadet.py (tools/convert_torch_cnn.py families rbonn / snn / idadet)"
    keys = ("rbonn", "snn", "idadet")

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn

        from bitino.adapters import torch_rbonn as R

        M = _modules()
        if type(model).__name__ not in ("BiRealNet", "ResNetREACTClassifier"):
            raise NotImplementedError(f"RBONN export hook: only BiRealNet / ResNetREACTClassifier (got {type(model).__name__})")
        rec = []
        f64 = lambda t: t.detach().cpu().to(torch.float64)   # noqa: E731
        pair = R._pair

        def bias(lb, path):
            if type(lb).__name__ != "LearnableBias":
                raise NotImplementedError(f"{path}: expected LearnableBias")
            return M["Bias"](lb.bias.detach().numpy().reshape(-1))

        def out_hw(hw, k, s, p):
            return tuple((v + 2 * pp - kk) // ss + 1 for v, kk, ss, pp in zip(hw, k, s, p))

        hw0 = out_hw((224, 224), pair(model.conv1.kernel_size), pair(model.conv1.stride), pair(model.conv1.padding))
        mp = model.maxpool
        hw_state = [out_hw(hw0, pair(mp.kernel_size), pair(mp.stride), pair(mp.padding))]

        class Block(nn.Module):
            def __init__(self, blk, path):
                super().__init__()
                idadet = type(getattr(blk, "conv", None)).__name__ == "BiConv"
                act = blk.conv.binfunc if idadet else blk.binary_activation
                if type(act).__name__ != "BinaryActivation" or type(blk.bn1).__name__ != "BatchNorm2d" or type(blk.prelu).__name__ != "PReLU":
                    raise NotImplementedError(f"{path}: unexpected block modules")
                bc = blk.conv if idadet else blk.binary_conv
                bc_name = f"{path}.conv" if idadet else f"{path}.binary_conv"
                w = bc.weight.detach().cpu()
                zeros = int((w == 0).sum())
                bias_c = None
                if idadet:
                    if bc.bias is None or bc.groups != 1 or tuple(bc.dilation) != (1, 1) or bc.padding_mode != "zeros" or bc.transposed:
                        raise NotImplementedError(f"{bc_name}: unexpected BiConv settings")
                    if zeros:
                        raise NotImplementedError(f"{bc_name}: latent weights exactly 0")
                    sign, latent = w.sign().numpy().astype(np.int8), f64(w).numpy()
                    scale64 = 0.02 * np.abs(f64(bc.scale).reshape(-1).numpy())
                    bias_c = bc.bias.detach().cpu().numpy().astype(np.float32).reshape(-1)
                    src = "BiConv: conv(sign(x), sign(W) * 0.02|scale|) + bias"
                elif type(bc).__name__ == "RBOConv":
                    if bc.bias is not None or bc.groups != 1 or tuple(bc.dilation) != (1, 1) or bc.padding_mode != "zeros" or bc.transposed:
                        raise NotImplementedError(f"{bc_name}: unexpected RBOConv settings")
                    if zeros:
                        raise NotImplementedError(f"{bc_name}: latent weights exactly 0 (torch.sign gives a 0 weight)")
                    sign, latent = w.sign().numpy().astype(np.int8), f64(w).numpy()
                    scale64 = (0.02 * f64(bc.scale_factor).abs()).reshape(-1).numpy()       # rbonn.py:57-63, 125-135
                    src = "RBOConv: conv(sign(x), sign(W) * 0.02|scale_factor|)"
                else:
                    if type(bc).__name__ != "HardBinaryConv" or tuple(w.shape[2:]) != (3, 3):
                        raise NotImplementedError(f"{bc_name}: unexpected {type(bc).__name__}")
                    sign, extra = R._hard_binary_conv_bits(bc, bc_name, zeros)   # the source's own pattern decode, fp32
                    w64 = f64(w)
                    scale64 = torch.mean(torch.mean(torch.mean(abs(w64), dim=3, keepdim=True), dim=2, keepdim=True),
                                         dim=1, keepdim=True).reshape(-1).numpy()
                    latent = w64.numpy() if np.array_equal(np.sign(w.numpy()).astype(np.int8), sign) else sign.astype(np.float64)
                    src = "HardBinaryConv: conv(sign(x), mean|W| * " + ("sign(W))" if not hasattr(bc, "binarize") else
                                                                         "nearest binary 3x3 pattern) (decoded ±1 as the latent)")
                self.move0 = bias(blk.move0, f"{path}.move0")
                self.binary_activation = named_(M["Binarize"](binarizer), bc_name)
                self.binary_conv = named_(M["BinaryWeightConv"](sign, latent, scale64, tuple(pair(bc.stride)), tuple(pair(bc.padding)),
                                                                bias=bias_c), bc_name, attrs={"pad_value": 0.0})
                self.bn1 = named_(export_copy(blk.bn1), f"{path}.bn1")
                hw_in = hw_state[0]
                hw_state[0] = out_hw(hw_in, tuple(w.shape[2:]), pair(bc.stride), pair(bc.padding))
                self.downsample = None
                if blk.downsample is not None:
                    ds = blk.downsample
                    if type(ds).__name__ != "Sequential" or len(ds) not in (2, 3):
                        raise NotImplementedError(f"{path}.downsample: unexpected structure")
                    mods = []
                    if len(ds) == 3:
                        ap = ds[0]
                        if type(ap).__name__ != "AvgPool2d" or ap.divisor_override is not None or pair(ap.padding) != (0, 0):
                            raise NotImplementedError(f"{path}.downsample.0: unexpected {ap}")
                        ck, cs = pair(ap.kernel_size), pair(ap.stride)
                        if ap.ceil_mode and not all((v - kk) % ss == 0 for v, kk, ss in zip(hw_in, ck, cs)):
                            raise NotImplementedError(f"{path}.downsample.0: ceil_mode AvgPool2d on {hw_in} differs from floor mode")
                        mods.append(nn.AvgPool2d(ck, cs, 0, ceil_mode=False, count_include_pad=ap.count_include_pad))
                    mods += [export_copy(ds[-2]), export_copy(ds[-1])]
                    self.downsample = nn.Sequential(*mods)
                self.move1 = bias(blk.move1, f"{path}.move1")
                c = sign.shape[0]
                sl = blk.prelu.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
                if sl.size not in (1, c):
                    raise NotImplementedError(f"{path}.prelu: {sl.size} slopes for {c} channels")
                self.prelu = M["PReLU"](np.broadcast_to(sl, (c,)).copy())
                self.move2 = bias(blk.move2, f"{path}.move2")
                rec.append({"module": bc_name, "source": src, "export": f"{'Sign' if binarizer == 'sign_zero' else 'BipolarQuant'} "
                            "-> Conv(BipolarQuant(W, 1)) -> Mul(scale float64 carrier)" + (" -> Add(bias)" if bias_c is not None else ""),
                            "latent_weight_zeros": zeros})

            def forward(self, x):
                out = self.bn1(self.binary_conv(self.binary_activation(self.move0(x))))
                residual = x if self.downsample is None else self.downsample(x)
                out = out + residual
                return self.move2(self.prelu(self.move1(out)))

        def named_(mod, name, **kw):
            mod.bitino_meta = {"name": name, **kw}
            return mod

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                if type(mp).__name__ != "MaxPool2d" or pair(mp.dilation) != (1, 1) or mp.ceil_mode:
                    raise NotImplementedError("maxpool: unexpected")
                self.conv1, self.bn1, self.maxpool = (export_copy(getattr(src, k)) for k in ("conv1", "bn1", "maxpool"))
                for ln in ("layer1", "layer2", "layer3", "layer4"):
                    setattr(self, ln, nn.Sequential(*[Block(b, f"{ln}.{i}") for i, b in enumerate(getattr(src, ln))]))
                if type(src.avgpool).__name__ != "AdaptiveAvgPool2d" or pair(src.avgpool.output_size) != (1, 1):
                    raise NotImplementedError("avgpool: expected AdaptiveAvgPool2d(1)")
                self.avgpool = copy.deepcopy(src.avgpool)
                self.fc = copy.deepcopy(src.fc)

            def forward(self, x):
                x = self.maxpool(self.bn1(self.conv1(x)))
                x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
                return self.fc(torch.flatten(self.avgpool(x), 1))

        known = {"BiRealNet", "Sequential", "BasicBlock", "Conv2d", "BatchNorm2d", "MaxPool2d", "AvgPool2d", "LearnableBias",
                 "BinaryActivation", "RBOConv", "HardBinaryConv", "Binarize", "PReLU", "AdaptiveAvgPool2d", "Linear",
                 "ResNetREACTClassifier", "BiConv"}
        for n, m in model.named_modules():
            if type(m).__name__ not in known:
                raise NotImplementedError(f"module {n or '<root>'}: unsupported type {type(m).__name__}")
        net = Net(model).eval()
        unused = sorted(n for n, _ in model.named_parameters() if n.endswith("binary_conv.u"))
        if unused:
            rec.append({"module": "*", "unused_in_eval_forward": unused, "note": "RBOConv.u: recurrent-module step size, training only"})
        return net, {"replacements": rec, "import_args": IMPORT_ARGS_TORCH}


class Sabnn(TorchCnnFamily):
    """SA-BNN ResNet-18 (anonymous1443/SA-BNN Drive release, birealnet.py.txt BiRealNet), as tools/convert_sabnn.py
    (convert_torch_cnn family sabnn) loads it. Mirrors bitino.adapters.torch_sabnn.convert:

      stem   Conv2d 7x7 + BN + PReLU(64) + MaxPool2d
      block  a = bn2(x); r = downsample(a) (AvgPool2d + conv1x1 + BN) or x; h = prelu(bn1(conv1(sign(a))) + r);
             out = prelu2(bn4(conv2(sign(bn3(h)))) + r)    (downsample2 is the same module, never applied: checked)
      binary conv  HardBinaryConv: conv(x, mean|W| * sign(W)) -> Conv(Bin(x), BipolarQuant(W, 1)) -> Mul(mean|W| carrier)
      head   AdaptiveAvgPool2d(1) + Linear
    """
    name = "sabnn"
    converter = "tools/convert_sabnn.py (tools/convert_torch_cnn.py family sabnn)"
    keys = ("sabnn",)

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn

        from bitino.adapters import torch_sabnn as S

        M = _modules()
        if type(model).__name__ != "BiRealNet":
            raise NotImplementedError("SA-BNN export hook: only BiRealNet")
        pair = S._pair
        rec = []

        def named_(mod, name, **kw):
            mod.bitino_meta = {"name": name, **kw}
            return mod

        def prelu(m, name, c):
            if type(m).__name__ != "PReLU":
                raise NotImplementedError(f"{name}: expected PReLU")
            sl = m.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
            if sl.size not in (1, c):
                raise NotImplementedError(f"{name}: {sl.size} slopes for {c} channels")
            return named_(M["PReLU"](np.broadcast_to(sl, (c,)).copy()), name)

        def bconv(blk, attr, act_attr, p):
            bc = getattr(blk, attr)
            name = f"{p}.{attr}"
            if type(getattr(blk, act_attr)).__name__ != "BinaryActivation":
                raise NotImplementedError(f"{p}.{act_attr}: unexpected")
            sign, post_mul, d = S._binary_bits(bc, name)
            w64 = bc.weights.detach().cpu().view(bc.shape).to(torch.float64).numpy()
            s = int(bc.stride)
            rec.append({"module": name, "source": "BN -> BinaryActivation + HardBinaryConv (flat weights)",
                        "export": f"{'Sign' if binarizer == 'sign_zero' else 'BipolarQuant'} -> Conv(BipolarQuant(W, 1)) -> Mul(mean|W| carrier)",
                        **d})
            return nn.Sequential(named_(M["Binarize"](binarizer), name),
                                 named_(M["BinaryWeightConv"](sign, w64, post_mul, (s, s), (1, 1)), name, attrs={"pad_value": 0.0}))

        class Block(nn.Module):
            def __init__(self, blk, p):
                super().__init__()
                if type(blk).__name__ != "BasicBlock" or not hasattr(blk, "binary_conv2"):
                    raise NotImplementedError(f"{p}: unexpected block")
                cin, cout = int(blk.binary_conv.shape[1]), int(blk.binary_conv.shape[0])
                stride = int(blk.binary_conv.stride)
                if int(blk.binary_conv2.stride) != 1 or tuple(blk.binary_conv2.shape[:2]) != (cout, cout):
                    raise NotImplementedError(f"{p}.binary_conv2: unexpected")
                if blk.downsample2 is not blk.downsample:
                    raise NotImplementedError(f"{p}.downsample2 is not the shared downsample module")
                self.bn2 = named_(export_copy(blk.bn2), f"{p}.bn2")
                self.conv1 = bconv(blk, "binary_conv", "binary_activation", p)
                self.bn1 = named_(export_copy(blk.bn1), f"{p}.bn1")
                self.downsample = None
                if blk.downsample is not None:
                    ds = blk.downsample
                    ap = ds[0]
                    if type(ds).__name__ != "Sequential" or len(ds) != 3 or type(ap).__name__ != "AvgPool2d" or ap.ceil_mode \
                            or ap.divisor_override is not None or pair(ap.padding) != (0, 0):
                        raise NotImplementedError(f"{p}.downsample: unexpected structure")
                    self.downsample = nn.Sequential(named_(copy.deepcopy(ap), f"{p}.downsample.0"),
                                                    named_(export_copy(ds[1]), f"{p}.downsample.1"),
                                                    named_(export_copy(ds[2]), f"{p}.downsample.2"))
                elif not (stride == 1 and cin == cout):
                    raise NotImplementedError(f"{p}: identity shortcut with a shape change")
                self.add = named_(Add2(), f"{p}.add")
                self.prelu = prelu(blk.prelu, f"{p}.prelu", cout)
                self.bn3 = named_(export_copy(blk.bn3), f"{p}.bn3")
                self.conv2 = bconv(blk, "binary_conv2", "binary_activation2", p)
                self.bn4 = named_(export_copy(blk.bn4), f"{p}.bn4")
                self.add2 = named_(Add2(), f"{p}.add2")
                self.prelu2 = prelu(blk.prelu2, f"{p}.prelu2", cout)

            def forward(self, x):
                a = self.bn2(x)
                t = self.bn1(self.conv1(a))
                r = x if self.downsample is None else self.downsample(a)
                t = self.prelu(self.add(t, r))
                u = self.bn4(self.conv2(self.bn3(t)))
                return self.prelu2(self.add2(u, r))

        class Add2(nn.Module):
            def forward(self, a, b):
                return a + b

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                mp = src.maxpool
                if type(mp).__name__ != "MaxPool2d" or pair(mp.dilation) != (1, 1) or mp.ceil_mode:
                    raise NotImplementedError("maxpool: unexpected")
                self.conv1, self.bn1 = export_copy(src.conv1), export_copy(src.bn1)
                self.prelu = prelu(src.prelu, "prelu", 64)
                self.maxpool = copy.deepcopy(mp)
                for ln in ("layer1", "layer2", "layer3", "layer4"):
                    setattr(self, ln, nn.Sequential(*[Block(b, f"{ln}.{i}") for i, b in enumerate(getattr(src, ln))]))
                if type(src.avgpool).__name__ != "AdaptiveAvgPool2d" or pair(src.avgpool.output_size) != (1, 1):
                    raise NotImplementedError("avgpool")
                self.avgpool = copy.deepcopy(src.avgpool)
                self.fc = copy.deepcopy(src.fc)

            def forward(self, x):
                x = self.maxpool(self.prelu(self.bn1(self.conv1(x))))
                x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
                return self.fc(torch.flatten(self.avgpool(x), 1))

        known = {"BiRealNet", "Sequential", "BasicBlock", "Conv2d", "BatchNorm2d", "MaxPool2d", "AvgPool2d", "BinaryActivation",
                 "HardBinaryConv", "PReLU", "AdaptiveAvgPool2d", "Linear"}
        for n, m in model.named_modules(remove_duplicate=False):
            if type(m).__name__ not in known:
                raise NotImplementedError(f"module {n or '<root>'}: unsupported type {type(m).__name__}")
        return Net(model).eval(), {"replacements": rec, "import_args": IMPORT_ARGS_TORCH}


class ReactA(Family):
    """ReActNet-A (MobileNet-v1-like) in the two flavours of bitino.adapters.torch_abbnn._convert_reactnet_a:
    AB-BNN ``Reactnet`` (Network/ReActNet_A_Qaw.py, no BatchNorm; tools/convert_abbnn.py) and the BatchNorm flavour of
    BURN (naver-ai/burn ReActnet, use_mlp head) and S2-BNN (szq0214/S2-BNN ReActNet, fc = Identity) as
    tools/convert_reacta_bn.py loads them. Mirrors the adapter node for node:

      stem   AB-BNN: Conv(x, fl32(ScaledStdConv2d fold)); BN flavour: Conv2d + BN (float64 carriers)
      block  t = Bin(move11(x)) -> Conv(BipolarQuant(W, 1)) -> Mul(post_mul) [-> Add(post_add)]   (binary_3x3)
                 AB-BNN post_mul = gain*mean|W_std|*alpha; BN flavour post_mul = mean|W|*s_bn, post_add = t_bn
                 (the BN after each binary conv folded into its epilogue in float64, as the adapter does)
             out1 = move13(prelu1(move12(t + (AvgPool2d(2)(x) if stride 2 else x))))
             b = Bin(move21(out1)); out2 = binary_pw(b) + out1, or concat_j(binary_pw_down_j(b) + out1)
             out = move23(prelu2(move22(out2)))
      head   AdaptiveAvgPool2d(1) + Linear (AB-BNN, BiMatting) / Linear + ReLU (BURN MLP) / none (S2-BNN features)
    BiMatting backbone (htqin/BiMatting BiMobileNet, step 1b; tools/convert_reacta_bn.py's BiMobileNetClassifier):
      BiConv2d       centred weights W' = W - nested_mean(W) (float64, the adapter's bn_binary_conv_parts): the latent
                     of BipolarQuant(W', 1), post_mul = mean|W'| * s_bn
      planes = 4 C   four binary_pw_down_j + out1, concatenated (as the 2 C blocks)
      planes = C / 2 binary_pw_down1(b) + (out1[:, :P] + out1[:, P:]) / 2, the half sum written as the adapter's
                     Reshape [1, 2, P, H*W] + Transpose [0, 2, 3, 1] (= IR Reshape [1, H*W, 2, P]) -> AveragePool
                     (1 x 2, count_include_pad 1: one rounding of a + b, halved exactly) -> Reshape [1, P, H, W]
      macro          forward_single_frame: after block `last` of each macro block, + AvgPool2d(2, 2)(block `first`
                     output) when the map size changed (ceil_mode = floor mode on the even maps, checked); the next
                     block binarizes the macro sum literally (no PReLU tail: the adapter's rule)
    Thresholds (BN flavour, the adapter's documented choice): a binarization that reads a PReLU tail binarizes
    q + fl64(b_prev + b_next) (q the PReLU output, b_prev the tail's LearnableBias, b_next the move before the sign),
    exported as Add(q, float64 carrier) beside the residual path's Add(q, b_prev). AB-BNN: sign((x + b)·beta) as
    Bin(x + b/beta) (beta > 0), as the ReActNet-18 hook. Every binary conv carries bitino.attrs pad_value 0.0 (the
    adapter writes the float form also for the unpadded 1x1 convs)."""
    name = "reacta"
    converter = "tools/convert_abbnn.py (Reactnet), tools/convert_reacta_bn.py (BURN, S2-BNN)"
    BN_IDS = ("burn-reacta-stage2", "s2bnn-reacta-distill-only", "bimatting-bimobilenet-backbone")

    def _ab(self):
        import convert_abbnn
        return convert_abbnn

    def _bn(self):
        import convert_reacta_bn
        return convert_reacta_bn

    def model_ids(self):
        return sorted([m for m, a in self._ab().ARCHS.items() if a[1] == "Reactnet"] + list(self.BN_IDS))

    def load(self, model_id, repo, weights):
        self._mid = model_id
        if model_id in self.BN_IDS:
            cv = self._bn()
            model, info = cv.load_source(model_id, repo, weights)
            return model, {"class": type(model).__name__, "source_sha256": info["source_sha256"], "repo": cv.MODELS[model_id][1],
                           "repo_commit": cv.repo_commit(repo), "load_repairs": info.get("repairs")}
        model, cls, src = self._ab().load_model(repo, model_id, weights)
        return model, {"class": cls, "source_sha256": src, "repo": "https://github.com/Ruichen0424/AB-BNN",
                       "repo_commit": self._ab().repo_commit(repo)}

    def preprocess(self, model_id):
        return self._bn().PRE_KEY if model_id in self.BN_IDS else self._ab().preprocess_of(model_id)

    def labels(self, model_id, em):
        return "none" if model_id in self.BN_IDS[:2] else super().labels(model_id, em)

    def source_logits(self, model, x_nhwc, dtype):
        if self._mid in self.BN_IDS:
            return self._bn().run_reference(model, x_nhwc, dtype, [])
        return self._ab().reference_outputs(model, x_nhwc, dtype)

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn

        from bitino.adapters import torch_abbnn as A

        M = _modules()
        top = type(model).__name__
        bn_flavour = top in ("ReActnet", "ReActNet", "BiMobileNetClassifier")
        bimatting = top == "BiMobileNetClassifier"
        if not (bn_flavour or top == "Reactnet"):
            raise NotImplementedError(f"ReActNet-A export hook: unsupported class {top}")
        pair = A._pair
        rec = []
        f64 = lambda t: t.detach().to(torch.float64).reshape(-1).numpy()   # noqa: E731

        def named(mod, name, **kw):
            mod.bitino_meta = {"name": name, **kw}
            return mod

        class Add2(nn.Module):
            def forward(self, a, b):
                return a + b

        class Cat(nn.Module):
            def forward(self, *xs):
                return torch.cat(xs, 1)

        class SplitK(nn.Module):
            """[1, C, H, W] -> Reshape [1, k, C/k, H*W] -> Transpose [0, 2, 3, 1] = [1, C/k, H*W, k]."""

            def __init__(self, c, k, hw):
                super().__init__()
                self.c, self.k, self.hw = c, k, hw

            def forward(self, x):
                return x.reshape(1, self.k, self.c // self.k, self.hw).permute(0, 2, 3, 1)

        class Reshape4(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = tuple(int(d) for d in shape)

            def forward(self, x):
                return x.reshape(self.shape)

        def lbias(lb, name):
            if type(lb).__name__ != "LearnableBias":
                raise NotImplementedError(f"{name}: expected LearnableBias")
            return lb

        def bconv(conv, alpha, name, bn=None):
            if not bn_flavour:
                if type(conv).__name__ != "HardBinaryScaledStdConv2d":
                    raise NotImplementedError(f"{name}: unexpected {type(conv).__name__}")
                parts = A.binary_conv_parts(conv)
                s, pd = int(conv.stride), int(conv.padding)
                latent = A.standardized_weight(conv, torch.float64).numpy()
                post_mul, post_add = parts["scale64"] * alpha, None
            else:
                parts = A.bn_binary_conv_parts(conv)
                (s, s2), (pd, pd2) = parts["stride"], parts["padding"]
                if s != s2 or pd != pd2:
                    raise NotImplementedError(f"{name}: non-square stride / padding")
                sbn, tbn = A.bn_affine64(bn)
                w64 = conv.weight.detach().to(torch.float64)
                if parts["kind"] == "BiConv2d":       # centred weights (float64, as bn_binary_conv_parts)
                    w64 = w64 - A._nested_mean(w64)
                latent = w64.numpy()
                post_mul, post_add = parts["scale64"] * sbn, tbn
            rec.append({"module": name, "source": parts["kind"] if bn_flavour else "HardBinaryScaledStdConv2d",
                        "export": f"{'Sign' if binarizer == 'sign_zero' else 'BipolarQuant'} -> Conv(BipolarQuant(W, 1)) -> Mul(post_mul c64)"
                                  + (" -> Add(post_add c64)" if post_add is not None else ""),
                        **{k: parts[k] for k in ("zeros_f32", "zeros_f64", "sign_mismatch_f32_vs_f64")}})
            return nn.Sequential(named(M["Binarize"](binarizer), name),
                                 named(M["BinaryWeightConv"](parts["sign"], latent, post_mul, (s, s), (pd, pd), bias=post_add), name,
                                       attrs={"pad_value": 0.0}))

        class Block(nn.Module):
            """forward(x, q_prev) -> (out, q): q is the prelu2 output (the PReLU tail the next block's move11 may read)."""

            def __init__(self, blk, p, hw, refold_in):
                super().__init__()
                if bn_flavour:
                    alpha = beta1 = beta2 = 1.0
                    for an in ("binary_activation", "binary_activation1", "binary_activation2"):
                        if hasattr(blk, an) and type(getattr(blk, an)).__name__ != "BinaryActivation":
                            raise NotImplementedError(f"{p}.{an}: unexpected type")
                else:
                    if type(blk.binary_activation).__name__ != "BinaryActivation":
                        raise NotImplementedError(f"{p}.binary_activation: unexpected type")
                    alpha, beta1, beta2 = float(blk.alpha), float(blk.beta1), float(blk.beta2)
                    if not (beta1 > 0 and beta2 > 0):
                        raise NotImplementedError(f"{p}: beta must be > 0")
                self.stride, cin, cout = int(blk.stride), int(blk.inplanes), int(blk.planes)
                self.refold_in = refold_in and bn_flavour
                b11 = f64(lbias(blk.move11, f"{p}.move11").bias)
                if self.refold_in:       # q_prev + fl64(b_prev + b_next)
                    self.b_prev_next = None
                    self._b11 = b11
                else:
                    self.move11 = named(M["Bias"](b11 / beta1), f"{p}.move11")
                self.conv3 = bconv(blk.binary_3x3, alpha, f"{p}.binary_3x3", getattr(blk, "bn1", None) if bn_flavour else None)
                self.pool = None
                if self.stride == 2:
                    ap = blk.pooling
                    if type(ap).__name__ != "AvgPool2d" or ap.divisor_override is not None or pair(ap.padding) != (0, 0):
                        raise NotImplementedError(f"{p}.pooling: unexpected {ap}")
                    k, st = pair(ap.kernel_size), pair(ap.stride)
                    if ap.ceil_mode and not ((hw[0] - k[0]) % st[0] == 0 and (hw[1] - k[1]) % st[1] == 0):
                        raise NotImplementedError(f"{p}.pooling: ceil_mode with a partial window at {hw}")
                    self.pool = named(nn.AvgPool2d(k, st, 0, ceil_mode=False, count_include_pad=ap.count_include_pad), f"{p}.pooling")
                elif self.stride != 1:
                    raise NotImplementedError(f"{p}: stride {self.stride}")
                self.add1 = named(Add2(), f"{p}.add1")
                self.move12, self.prelu1, self.move13 = self._tail(blk, "move12", "prelu1", "move13", p, cin)
                b21 = f64(lbias(blk.move21, f"{p}.move21").bias)
                if bn_flavour:          # always a PReLU tail (prelu1 output)
                    self.move21 = named(M["Bias"](f64(blk.move13.bias) + b21), f"{p}.move21")
                else:
                    self.move21 = named(M["Bias"](b21 / beta2), f"{p}.move21")
                if cin == cout:
                    self.kind = "pw"
                    self.pw = bconv(blk.binary_pw, alpha, f"{p}.binary_pw", blk.bn2 if bn_flavour else None)
                    self.add2 = named(Add2(), f"{p}.add2")
                elif bimatting and 2 * cout == cin and type(blk.binary_pw_down1).__name__ == "BiConv2d":
                    self.kind = "half"            # mobilenet.py:166-172: + (out1[:, :P] + out1[:, P:]) / 2
                    ohw = (hw[0] // 2, hw[1] // 2) if self.stride == 2 else hw
                    self.split = named(SplitK(cin, 2, ohw[0] * ohw[1]), f"{p}.split")
                    self.split_sum_half = named(nn.AvgPool2d((1, 2), (1, 2), 0, ceil_mode=False, count_include_pad=True),
                                                f"{p}.split_sum_half")
                    self.split_sum = named(Reshape4((1, cout) + tuple(ohw)), f"{p}.split_sum")
                    self.pw = bconv(blk.binary_pw_down1, alpha, f"{p}.binary_pw_down1", blk.bn2_1)
                    self.add2 = named(Add2(), f"{p}.add2")
                elif cout in (2 * cin, 4 * cin) and (bn_flavour or cout == 2 * cin):
                    self.kind = "down"
                    self.n = cout // cin
                    self.downs = nn.ModuleList([bconv(getattr(blk, f"binary_pw_down{j}"), alpha, f"{p}.binary_pw_down{j}",
                                                      getattr(blk, f"bn2_{j}") if bn_flavour else None) for j in range(1, self.n + 1)])
                    self.adds = nn.ModuleList([named(Add2(), f"{p}.add2_{j}") for j in range(1, self.n + 1)])
                    self.cat = named(Cat(), f"{p}.cat")
                else:
                    raise NotImplementedError(f"{p}: planes {cout} vs inplanes {cin} not supported")
                self.move22, self.prelu2, self.move23 = self._tail(blk, "move22", "prelu2", "move23", p, cout)
                self.b23 = f64(blk.move23.bias)
                rec.append({"module": p, "source": "ReActNet-A block", "refolded_move11": self.refold_in, "refolded_move21": bn_flavour})

            def _tail(self, blk, a, b, c, p, ch):
                pr = getattr(blk, b)
                if type(pr).__name__ == "Q_PReLU" and not bn_flavour:
                    slope = A.qprelu_slope(pr)
                elif bn_flavour and type(pr).__name__ == "PReLU":
                    slope = pr.weight.detach().to(torch.float32).numpy().reshape(-1).copy()
                    if slope.size != ch:
                        raise NotImplementedError(f"{p}.{b}: expected one slope per channel")
                else:
                    raise NotImplementedError(f"{p}.{b}: unexpected {type(pr).__name__}")
                return (named(M["Bias"](f64(lbias(getattr(blk, a), f"{p}.{a}").bias)), f"{p}.{a}"),
                        named(M["PReLU"](slope), f"{p}.{b}"),
                        named(M["Bias"](f64(lbias(getattr(blk, c), f"{p}.{c}").bias)), f"{p}.{c}"))

            def forward(self, x, q_prev):
                t = self.move11(q_prev) if self.refold_in else self.move11(x)
                t = self.conv3(t)
                res = self.pool(x) if self.pool is not None else x
                t = self.add1(t, res)
                q1 = self.prelu1(self.move12(t))
                out1 = self.move13(q1)
                b = self.move21(q1) if bn_flavour else self.move21(out1)
                if self.kind == "pw":
                    t = self.add2(self.pw(b), out1)
                elif self.kind == "half":
                    half = self.split_sum(self.split_sum_half(self.split(out1)))
                    t = self.add2(self.pw(b), half)
                else:
                    t = self.cat(*[ad(dn(b), out1) for dn, ad in zip(self.downs, self.adds)])
                q2 = self.prelu2(self.move22(t))
                return self.move23(q2), q2

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                if top == "ReActnet":
                    oi = list(src.out_indices) if isinstance(src.out_indices, (list, tuple)) else [src.out_indices]
                    if oi != [len(src.feature) - 1] or not bool(src.use_mlp):
                        raise NotImplementedError("ReActnet: out_indices / use_mlp")
                feats = src.backbone.feature if bimatting else src.feature
                if bimatting and type(src.backbone).__name__ != "BiMobileNet":
                    raise NotImplementedError("backbone: expected BiMobileNet")
                if type(feats).__name__ != "ModuleList" or type(feats[0]).__name__ != "firstconv3x3":
                    raise NotImplementedError("feature: expected ModuleList with firstconv3x3")
                c1 = feats[0].conv1
                if type(c1).__name__ != ("Conv2d" if bn_flavour else "ScaledStdConv2d") or c1.bias is not None or c1.groups != 1:
                    raise NotImplementedError(f"feature.0.conv1: unexpected {type(c1).__name__}")
                (sh, sw), (ph, pw), (kh, kw) = pair(c1.stride), pair(c1.padding), pair(c1.kernel_size)
                if bn_flavour:
                    self.conv1 = named(M["FloatConv"](c1.weight.detach().numpy(), (sh, sw), (ph, pw)), "feature.0.conv1")
                    s0, t0 = A.bn_affine64(feats[0].bn1)
                    self.bn1 = named(M["Affine"](s0, t0, 4), "feature.0.bn1")
                else:
                    self.conv1 = named(M["FloatConv"](A.fold_scaled_std_conv(c1), (sh, sw), (ph, pw)), "feature.0.conv1")
                    self.bn1 = None
                in_hw = 224 if (bn_flavour or self_ab.ARCHS[self_mid][3]) else 32
                hw = ((in_hw + 2 * ph - kh) // sh + 1,) * 2
                blocks = []
                macro = A.BIMOBILENET_MACRO_BLOCKS if bimatting else ()
                macro_first = {a: b for a, b in macro}
                macro_hw: dict = {}
                self.macro = nn.ModuleDict()          # block index (str) -> (pool, add) after that block
                after_macro = set()
                for bi in range(1, len(feats)):
                    blk = feats[bi]
                    if type(blk).__name__ not in (("BasicBlock", "ReActNetBlock") if bn_flavour else ("BasicBlock",)):
                        raise NotImplementedError(f"feature.{bi}: unexpected block {type(blk).__name__}")
                    blocks.append(Block(blk, f"feature.{bi}", hw, refold_in=bi > 1 and (bi - 1) not in after_macro))
                    if int(blk.stride) == 2:
                        hw = (hw[0] // 2, hw[1] // 2)
                    if bi in macro_first:
                        macro_hw[macro_first[bi]] = (bi, hw)
                    elif bi in macro_hw:
                        first, fhw = macro_hw.pop(bi)
                        if fhw[0] != hw[0]:           # mobilenet.py:206-217: + pooling(feature) when the size changed
                            ap = src.backbone.pooling
                            k, st = pair(ap.kernel_size), pair(ap.stride)
                            if type(ap).__name__ != "AvgPool2d" or ap.divisor_override is not None or pair(ap.padding) != (0, 0) \
                                    or (ap.ceil_mode and not ((fhw[0] - k[0]) % st[0] == 0 and (fhw[1] - k[1]) % st[1] == 0)) \
                                    or (fhw[0] // 2, fhw[1] // 2) != hw:
                                raise NotImplementedError(f"macro residual into feature.{bi}: unexpected pooling / sizes")
                            self.macro[str(bi)] = nn.ModuleList([
                                named(nn.AvgPool2d(k, st, 0, ceil_mode=False, count_include_pad=ap.count_include_pad), f"macro{bi}.pooling"),
                                named(Add2(), f"macro{bi}.add")])
                            self.macro_src = getattr(self, "macro_src", {})
                            self.macro_src[bi] = first
                            after_macro.add(bi)
                self.blocks = nn.ModuleList(blocks)
                if type(src.pool1).__name__ != "AdaptiveAvgPool2d" or pair(src.pool1.output_size) != (1, 1):
                    raise NotImplementedError("pool1: expected AdaptiveAvgPool2d(1)")
                self.pool1 = named(nn.AdaptiveAvgPool2d(1), "pool1")
                fc = src.fc
                if top == "ReActnet":
                    if type(fc).__name__ != "Sequential" or len(fc) != 2 or type(fc[0]).__name__ != "Linear" or type(fc[1]).__name__ != "ReLU":
                        raise NotImplementedError("fc: expected Linear + ReLU")
                    self.head = nn.Sequential(named(copy.deepcopy(fc[0]), "fc.0"), named(nn.ReLU(), "fc.1"))
                elif type(fc).__name__ == "Identity":
                    if not bn_flavour:
                        raise NotImplementedError("fc: Identity")
                    self.head = None
                else:
                    if type(fc).__name__ != "Linear" or fc.bias is None:
                        raise NotImplementedError(f"fc: unexpected {type(fc).__name__}")
                    self.head = named(copy.deepcopy(fc), "fc")

            def forward(self, x):
                x = self.conv1(x)
                if self.bn1 is not None:
                    x = self.bn1(x)
                q = x
                outs = {}
                for i, b in enumerate(self.blocks):
                    x, q = b(x, q)
                    outs[i + 1] = x
                    if str(i + 1) in self.macro:
                        pool, add = self.macro[str(i + 1)]
                        x = add(x, pool(outs[self.macro_src[i + 1]]))
                        outs[i + 1] = x
                x = torch.flatten(self.pool1(x), 1)
                return x if self.head is None else self.head(x)

        self_mid, self_ab = self._mid, self._ab()
        known = ({"ReActnet", "ReActNet", "ModuleList", "firstconv3x3", "BasicBlock", "ReActNetBlock", "Conv2d", "BatchNorm2d",
                  "BinaryConv2d", "HardBinaryConv", "LearnableBias", "BinaryActivation", "PReLU", "AvgPool2d", "AdaptiveAvgPool2d",
                  "Linear", "ReLU", "Sequential", "Identity", "BiMobileNetClassifier", "BiMobileNet", "BiConv2d"} if bn_flavour else
                 {"Reactnet", "ModuleList", "firstconv3x3", "BasicBlock", "ScaledStdConv2d", "HardBinaryScaledStdConv2d",
                  "LearnableBias", "BinaryActivation", "Q_PReLU", "AvgPool2d", "AdaptiveAvgPool2d", "Linear"})
        for n, m in model.named_modules():
            if type(m).__name__ not in known:
                raise NotImplementedError(f"module {n or '<root>'}: unsupported type {type(m).__name__}")
        net = Net(model).eval()
        prev = None             # move11 of a block after a PReLU tail: q_prev + fl64(b_prev + b_next)
        for i, b in enumerate(net.blocks):
            if b.refold_in:
                b.move11 = named(M["Bias"](prev.b23 + b._b11), f"feature.{i + 1}.move11")
            prev = b
        return net, {"replacements": rec, "import_args": IMPORT_ARGS_TORCH, "batch1": bimatting}


class Bnext(TorchCnnFamily):
    """BNext (hpi-xnor/BNext src/bnext.py, BNext-T/S/M/L), as tools/convert_bnext.py (convert_torch_cnn family bnext)
    loads it. Mirrors bitino.adapters.torch_bnext.convert node for node:

      stem      firstconv3x3: Conv2d 3x3 + BN (its PReLU is unused in forward)
      unit      (Attention / FFN_3x3 / FFN_1x1)  residual = PReLU1(x) (Attention) or x, AvgPool2d(2) if stride 2
                t = norm1(Conv(Bin(move(x)), BipolarQuant(W, 1)))   [HardSign = torch.sign in eval; no scaling factor]
                t = PReLU2(t); info recoupling: mix = Mul(x', scale) + Mul(t, 1 - scale)   (float64 carriers;
                FFN_1x1 clamps scale to [0, 1] in eval, as the source)
                squeeze-and-expand: g = Conv1x1(ReLU(Conv1x1(GlobalAveragePool(mix)))) ; t = Mul(t, Sigmoid(g))
                Attention: t = Mul(t, residual) (ELM); norm2; out = t + residual
      block     unit3x3 then FFN, or concat(FFN_1, FFN_2) when planes = 2 * inplanes
      head      PReLU + AdaptiveAvgPool2d(1) + Linear
    Names: the adapter's (module paths, ``<unit>.mix_input`` / ``.mix_x`` / ``.mix`` / ``.se_mul`` / ``.elm_mul`` /
    ``.add`` / ``.pooling.residual`` / ``.pooling.input``; the mix constants ``<unit>.scale`` / ``<unit>.one_minus_scale``
    through bitino.consts); every binary conv carries bitino.attrs pad_value 0.0."""
    name = "bnext"
    converter = "tools/convert_bnext.py (tools/convert_torch_cnn.py family bnext)"
    keys = ("bnext",)

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from bitino.adapters import torch_bnext as B

        M = _modules()
        if type(model).__name__ != "BNext":
            raise NotImplementedError("BNext export hook: only class BNext")
        pair = B._pair
        rec = []

        def named(mod, name, **kw):
            mod.bitino_meta = {"name": name, **kw}
            return mod

        def prelu(m, name, c):
            if type(m).__name__ != "PReLU":
                raise NotImplementedError(f"{name}: expected PReLU")
            w = m.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
            if w.size not in (1, c):
                raise NotImplementedError(f"{name}: {w.size} slopes for {c} channels")
            return named(M["PReLU"](np.broadcast_to(w, (c,)).copy()), name)

        def bn(m, name):
            if type(m).__name__ != "BatchNorm2d" or not (m.affine and m.track_running_stats):
                raise NotImplementedError(f"{name}: expected BatchNorm2d with running stats")
            return named(export_copy(m), name)

        def avgpool2(m, name):
            if type(m).__name__ != "AvgPool2d" or pair(m.kernel_size) != (2, 2) or pair(m.stride) != (2, 2) or pair(m.padding) != (0, 0) \
                    or m.ceil_mode or m.divisor_override is not None:
                raise NotImplementedError(f"{name}: expected AvgPool2d(2, 2)")
            return named(nn.AvgPool2d(2, 2, 0, count_include_pad=m.count_include_pad), name)

        def conv1x1(m, name):
            if type(m).__name__ != "Conv2d" or m.kernel_size != (1, 1) or pair(m.stride) != (1, 1) or m.groups != 1 \
                    or pair(m.padding) != (0, 0) or m.bias is None:
                raise NotImplementedError(f"{name}: expected 1x1 Conv2d with bias")
            return named(copy.deepcopy(m), name)

        class Scale(M["Consts"]):
            def __init__(self, s64, name, cname):
                super().__init__()
                self.setc("s", np.asarray(s64, np.float64), (1, -1, 1, 1))
                self.bitino_meta = {"name": name, "consts": {"1": cname}}

            def forward(self, x):
                return x * self.c("s", x)

        class Op2(nn.Module):
            def __init__(self, kind, name):
                super().__init__()
                self.kind = kind
                self.bitino_meta = {"name": name}

            def forward(self, a, b):
                return a + b if self.kind == "add" else (a * b if self.kind == "mul" else a * torch.sigmoid(b))

        class Cat(nn.Module):
            def forward(self, a, b):
                return torch.cat([a, b], 1)

        class Unit(nn.Module):
            def __init__(self, u, p, c):
                super().__init__()
                cls = type(u).__name__
                if cls not in B.UNIT_CLASSES:
                    raise NotImplementedError(f"{p}: unsupported unit {cls}")
                self.stride = int(u.stride)
                if self.stride not in (1, 2) or (cls == "FFN_1x1" and self.stride != 1) or not (u.inplanes == u.planes == c):
                    raise NotImplementedError(f"{p}: unexpected stride / planes")
                self.cls = cls
                self.act1 = prelu(u.activation1, f"{p}.activation1", c) if cls == "Attention" else None
                self.pool_r = avgpool2(u.pooling, f"{p}.pooling.residual") if self.stride == 2 else None
                if type(u.move).__name__ != "LearnableBias":
                    raise NotImplementedError(f"{p}.move: unexpected")
                self.move = named(M["Bias"](u.move.bias.detach().cpu().numpy().astype(np.float32).reshape(-1)), f"{p}.move")
                ha = u.binary_activation
                if type(ha).__name__ != "HardSign" or ha.progressive or not float(ha.temperature.clamp(min=1e-8)) > 0 \
                        or list(ha.range) != [-1.5, 1.5]:
                    raise NotImplementedError(f"{p}.binary_activation: expected HardSign(progressive=False, range [-1.5, 1.5])")
                bc = u.binary_conv
                if type(bc).__name__ != "HardBinaryConv" or bc.groups != 1 or not bool(bc.temperature < 1e-7):
                    raise NotImplementedError(f"{p}.binary_conv: expected HardBinaryConv(groups=1) with temperature < 1e-7")
                w = bc.weight.detach().cpu()
                o, i, kh, kw = w.shape
                if kh != kw or i != c or o != c or int(bc.stride) != self.stride or int((w == 0).sum()):
                    raise NotImplementedError(f"{p}.binary_conv: unexpected weight shape / stride / exact zeros")
                sign = w.sign().numpy().astype(np.int8)
                self.binact = named(M["Binarize"](binarizer), f"{p}.binary_conv")
                self.bconv = named(M["BinaryWeightConv"](sign, w.to(torch.float64).numpy(), None,
                                                         (self.stride, self.stride), (kh // 2, kh // 2)), f"{p}.binary_conv",
                                   attrs={"pad_value": 0.0})
                self.norm1 = bn(u.norm1, f"{p}.norm1")
                self.act2 = prelu(u.activation2, f"{p}.activation2", c)
                self.recoupling = bool(u.infor_recoupling)
                if self.recoupling:
                    sc = u.scale.detach().cpu().double().reshape(-1)
                    if cls == "FFN_1x1":
                        sc = sc.clamp(0, 1)
                    sc = sc.numpy()
                    self.pool_i = avgpool2(u.pooling, f"{p}.pooling.input") if self.stride == 2 else None
                    self.mix_input = Scale(sc, f"{p}.mix_input", f"{p}.scale")
                    self.mix_x = Scale(1.0 - sc, f"{p}.mix_x", f"{p}.one_minus_scale")
                    self.mix = Op2("add", f"{p}.mix")
                    se = u.se
                    if type(se).__name__ != "SqueezeAndExpand" or type(se.attention).__name__ != "Sigmoid":
                        raise NotImplementedError(f"{p}.se: only attention_mode='sigmoid'")
                    seq = se.se
                    if type(seq).__name__ != "Sequential" or len(seq) != 4 or type(seq[0]).__name__ != "AdaptiveAvgPool2d" \
                            or pair(seq[0].output_size) != (1, 1) or type(seq[2]).__name__ != "ReLU":
                        raise NotImplementedError(f"{p}.se: unexpected structure")
                    self.se0 = named(nn.AdaptiveAvgPool2d(1), f"{p}.se.se.0")
                    self.se1 = conv1x1(seq[1], f"{p}.se.se.1")
                    self.se2 = named(nn.ReLU(), f"{p}.se.se.2")
                    self.se3 = conv1x1(seq[3], f"{p}.se.se.3")
                    self.se_mul = Op2("gate", f"{p}.se_mul")
                if cls == "Attention":
                    self.elm = Op2("mul", f"{p}.elm_mul")
                self.norm2 = bn(u.norm2, f"{p}.norm2") if (self.recoupling or cls == "Attention") else None
                self.add = Op2("add", f"{p}.add")
                rec.append({"module": p, "source": f"{cls} (HardSign + HardBinaryConv, infor_recoupling {self.recoupling})"})

            def forward(self, x):
                residual = self.act1(x) if self.act1 is not None else x
                if self.pool_r is not None:
                    residual = self.pool_r(residual)
                t = self.act2(self.norm1(self.bconv(self.binact(self.move(x)))))
                if self.recoupling:
                    mi = self.pool_i(x) if self.pool_i is not None else x
                    mix = self.mix(self.mix_input(mi), self.mix_x(t))
                    gt = self.se3(self.se2(self.se1(self.se0(mix))))
                    t = self.se_mul(t, gt)
                if self.cls == "Attention":
                    t = self.elm(t, residual)
                if self.norm2 is not None:
                    t = self.norm2(t)
                return self.add(t, residual)

        class Block(nn.Module):
            def __init__(self, blk, p, c):
                super().__init__()
                if type(blk).__name__ not in B.BLOCK_CLASSES or blk.inplanes != c or blk.planes not in (c, 2 * c):
                    raise NotImplementedError(f"{p}: unsupported block")
                first = "Attention" if hasattr(blk, "Attention") else "FFN_3x3"
                self.u1 = Unit(getattr(blk, first), f"{p}.{first}", c)
                self.split = blk.inplanes != blk.planes
                if not self.split:
                    self.u2 = Unit(blk.FFN, f"{p}.FFN", c)
                else:
                    self.f1, self.f2 = Unit(blk.FFN_1, f"{p}.FFN_1", c), Unit(blk.FFN_2, f"{p}.FFN_2", c)
                    self.cat = named(Cat(), f"{p}.cat")

            def forward(self, x):
                x = self.u1(x)
                return self.u2(x) if not self.split else self.cat(self.f1(x), self.f2(x))

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                feats = src.feature
                if type(feats).__name__ != "ModuleList" or type(feats[0]).__name__ != "firstconv3x3":
                    raise NotImplementedError("feature: expected ModuleList + firstconv3x3")
                cv = feats[0].conv1
                if type(cv).__name__ != "Conv2d" or cv.bias is not None or cv.groups != 1 or pair(cv.dilation) != (1, 1):
                    raise NotImplementedError("feature.0.conv1: unexpected conv")
                self.conv1 = named(copy.deepcopy(cv), "feature.0.conv1")
                self.bn1 = bn(feats[0].bn1, "feature.0.bn1")
                c = int(cv.out_channels)
                blocks = []
                for bi in range(1, len(feats)):
                    blocks.append(Block(feats[bi], f"feature.{bi}", c))
                    c = int(feats[bi].planes)
                self.blocks = nn.Sequential(*blocks)
                self.prelu = prelu(src.prelu, "prelu", c)
                if type(src.pool1).__name__ != "AdaptiveAvgPool2d" or pair(src.pool1.output_size) != (1, 1):
                    raise NotImplementedError("pool1")
                self.pool1 = named(nn.AdaptiveAvgPool2d(1), "pool1")
                fc = src.fc
                if type(fc).__name__ != "Linear" or fc.bias is None or fc.in_features != c:
                    raise NotImplementedError("fc")
                self.fc = named(copy.deepcopy(fc), "fc")

            def forward(self, x):
                x = self.blocks(self.bn1(self.conv1(x)))
                return self.fc(torch.flatten(self.pool1(self.prelu(x)), 1))

        known = {"BNext", "ModuleList", "Sequential", "firstconv3x3", *B.BLOCK_CLASSES, *B.UNIT_CLASSES, "Conv2d", "BatchNorm2d",
                 "PReLU", "LearnableBias", "HardSign", "HardBinaryConv", "AvgPool2d", "SqueezeAndExpand", "AdaptiveAvgPool2d",
                 "ReLU", "Sigmoid", "Linear"}
        for n, m in model.named_modules():
            if type(m).__name__ not in known:
                raise NotImplementedError(f"module {n or '<root>'}: unsupported type {type(m).__name__}")
        return Net(model).eval(), {"replacements": rec, "import_args": IMPORT_ARGS_TORCH}


class Bitorch(TorchCnnFamily):
    """bitorch 0.3.0 model hub (ResnetE18, DenseNet28, MeliusNet22; Lightning checkpoints), as tools/convert_bitorch.py
    (convert_torch_cnn family bitorch) loads it. Mirrors bitino.adapters.torch_bitorch.convert (its recursive layer
    mapping, names = the wrapper's module paths):

      QConv2d      a = Sign(x) (0 -> +1); conv2d(pad(a, value=-1), Sign(W) (0 -> +1), padding 0), no scale
                   -> BipolarQuant(x, 1) -> Pad(-1) -> Conv(., BipolarQuant(W, 1), pads 0)   [ge is the source
                   semantics: no sign_zero file]
      BatchNorm2d  Mul + Add of float64 carriers (affine or not);  Conv2d / ReLU / MaxPool2d / AvgPool2d as is
      DenseLayer   cat([x, features(x)], channels) -> Concat
      ImprovementBlock  x + y, or the sliced addition x + pad(y, C - c channels in front, 0) -> Pad + Add
      BasicBlock   body(x) + (downsample(x) or x)
      head         AdaptiveAvgPool2d(1) + Flatten + Linear"""
    name = "bitorch"
    converter = "tools/convert_bitorch.py (tools/convert_torch_cnn.py family bitorch)"
    keys = ("bitorch",)
    source_binarizer = "ge"

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from bitino.adapters import torch_bitorch as T

        M = _modules()
        BQ = M["BipolarQuantFn"]
        if binarizer != "ge":
            raise NotImplementedError("bitorch binarizes with sign(0) = +1: only the ge export exists")
        prefix = "_model." if hasattr(model, "_model") else ""
        net = model._model if prefix else model
        top = type(net).__name__
        if top not in ("_ResnetE", "_DenseNet", "_MeliusNet"):
            raise NotImplementedError(f"unsupported bitorch network class {top}")
        pair = T._pair
        rec = []
        n_q = [0]

        def named(mod, name, **kw):
            mod.bitino_meta = {"name": name, **kw}
            return mod

        class QConv(nn.Module):
            def __init__(self, m, name):
                super().__init__()
                act = m.activation
                if type(act).__name__ != "QActivation" or type(act.activation_function).__name__ not in T.SIGN_CLASSES \
                        or type(m._weight_quantize).__name__ not in T.SIGN_CLASSES or m._weight_quantize.training:
                    raise NotImplementedError(f"{name}: input / weight quantization is not sign")
                if m.bias is not None or m.padding_mode != "zeros" or float(m._pad_value) != -1.0:
                    raise NotImplementedError(f"{name}: bias / padding_mode / pad_value not supported")
                w = m.weight.detach()
                with torch.no_grad():
                    if not bool(torch.equal(m._weight_quantize(w), torch.where(w >= 0, 1.0, -1.0))):
                        raise NotImplementedError(f"{name}: weight quantizer output is not (W >= 0 ? +1 : -1)")
                    probe = torch.tensor([-2.0, -1e-30, -0.0, 0.0, 1e-30, 3.0])
                    if not torch.equal(act(probe), torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])):
                        raise NotImplementedError(f"{name}: input quantizer is not (x >= 0 ? +1 : -1)")
                self.register_buffer("w", w.clone())
                self.register_buffer("one", torch.tensor(1.0))
                self.stride, self.pad, self.dilation, self.groups = pair(m.stride), pair(m.padding), pair(m.dilation), int(m.groups)
                self.bitino_meta = {"name": name}
                n_q[0] += 1

            def forward(self, x):
                a = BQ.apply(x, self.one)
                ph, pw = self.pad
                if ph or pw:
                    a = M["PadFn"].apply(a, (0, 0, ph, pw, 0, 0, ph, pw), -1.0)
                return F.conv2d(a, BQ.apply(self.w, self.one), None, self.stride, 0, self.dilation, self.groups)

        class BN(M["Consts"]):
            def __init__(self, m, name):
                super().__init__()
                if type(m).__name__ != "BatchNorm2d" or not m.track_running_stats or m.running_mean is None:
                    raise NotImplementedError(f"{name}: unexpected {m}")
                f = lambda t: t.detach().to(torch.float64)   # noqa: E731
                gamma = f(m.weight) if m.affine else torch.ones_like(m.running_mean, dtype=torch.float64)
                beta = f(m.bias) if m.affine else torch.zeros_like(m.running_mean, dtype=torch.float64)
                inv = 1.0 / torch.sqrt(f(m.running_var) + m.eps)
                self.setc("s", (gamma * inv).numpy(), (1, -1, 1, 1))
                self.setc("t", (beta - f(m.running_mean) * gamma * inv).numpy(), (1, -1, 1, 1))
                self.bitino_meta = {"name": name}

            def forward(self, x):
                return x * self.c("s", x) + self.c("t", x)

        class Seq(nn.Module):
            def __init__(self, mods):
                super().__init__()
                self.mods = nn.ModuleList(mods)

            def forward(self, x):
                for m in self.mods:
                    x = m(x)
                return x

        class Cat2(nn.Module):
            def forward(self, a, b):
                return torch.cat([a, b], 1)

        class Add2(nn.Module):
            def forward(self, a, b):
                return a + b

        class PadFront(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.n = n

            def forward(self, x):
                return M["PadFn"].apply(x, (0, self.n, 0, 0, 0, 0, 0, 0), 0.0)

        class Dense(nn.Module):
            def __init__(self, feats, name):
                super().__init__()
                self.features, self.cat = feats, named(Cat2(), f"{name}.cat")

            def forward(self, x):
                return self.cat(x, self.features(x))

        class Improve(nn.Module):
            def __init__(self, body, name, pad):
                super().__init__()
                self.body, self.pad = body, (named(PadFront(pad), f"{name}.pad") if pad else None)
                self.add = named(Add2(), f"{name}.add")

            def forward(self, x):
                y = self.body(x)
                return self.add(x, self.pad(y) if self.pad is not None else y)

        class Basic(nn.Module):
            def __init__(self, body, ds, name):
                super().__init__()
                self.body, self.ds, self.add = body, ds, named(Add2(), f"{name}.add")

            def forward(self, x):
                res = self.ds(x) if self.ds is not None else x     # the adapter creates the downsample nodes first
                return self.add(self.body(x), res)

        def seq(m, name):
            return Seq([build(sub, f"{name}.{k}") for k, sub in m.named_children()])

        def build(m, name):
            if type(m).__name__ == "LayerContainer":
                m = m._layer_implementation
            c = type(m).__name__
            if c == "Sequential":
                return seq(m, name)
            if T._is_qconv(m):
                return QConv(m, name)
            if c == "Conv2d":
                if m.padding_mode != "zeros":
                    raise NotImplementedError(f"{name}: padding_mode")
                return named(copy.deepcopy(m), name)
            if c == "BatchNorm2d":
                return BN(m, name)
            if c == "ReLU":
                return named(nn.ReLU(), name)
            if c in ("MaxPool2d", "AvgPool2d"):
                if m.ceil_mode or (c == "MaxPool2d" and pair(m.dilation) != (1, 1)) or (c == "AvgPool2d" and m.divisor_override is not None):
                    raise NotImplementedError(f"{name}: unsupported {c} settings")
                return named(copy.deepcopy(m), name)
            if c == "AdaptiveAvgPool2d":
                if pair(m.output_size) != (1, 1):
                    raise NotImplementedError(f"{name}: output_size")
                return named(nn.AdaptiveAvgPool2d(1), name)
            if c == "Flatten":
                return named(nn.Flatten(), name)
            if c == "DenseLayer":
                if m.dropout:
                    raise NotImplementedError(f"{name}: dropout")
                return Dense(seq(m.features, f"{name}.features"), name)
            if c == "ImprovementBlock":
                body = seq(m.body, f"{name}.body")
                if not m.use_sliced_addition:
                    return Improve(body, name, 0)
                c_in = int(m.slices[-1])
                c_y = c_in - int(m.slices[1])
                if list(m.slices) != [0, c_in - c_y, c_in] or list(m.slices_add_x) != [False, True]:
                    raise NotImplementedError(f"{name}: unexpected slices")
                return Improve(body, name, c_in - c_y)
            if c == "BasicBlock":
                ds = seq(m.downsample, f"{name}.downsample") if m.shall_downsample else None
                return Basic(seq(m.body, f"{name}.body"), ds, name)
            raise NotImplementedError(f"{name}: unsupported module type {c}")

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = build(net.features, f"{prefix}features")
                if top == "_ResnetE":
                    head, hname = net.output_layer, f"{prefix}output_layer"
                    self.finalize = None
                else:
                    self.finalize = build(net.finalize, f"{prefix}finalize")
                    head, hname = net.output, f"{prefix}output"
                if type(head).__name__ != "Linear" or head.bias is None:
                    raise NotImplementedError(f"{hname}: unexpected head")
                self.head = named(copy.deepcopy(head), hname)

            def forward(self, x):
                x = self.features(x)
                if self.finalize is not None:
                    x = self.finalize(x)
                return self.head(x)

        out = Net().eval()
        rec.append({"module": "*", "source": f"bitorch {top}", "binary_convs": n_q[0],
                    "export": "BipolarQuant(x, 1) -> Pad(-1) -> Conv(BipolarQuant(W, 1)) (ge = the source semantics)"})
        return out, {"replacements": rec, "import_args": IMPORT_ARGS_TORCH}


# ================================================================================================ step 1b: image-to-image
IMPORT_ARGS_IMAGE = ["--naming", "exporter", "--affine", "keep", "--cleanup", "none"]   # data in the importer's walk order


def _named(mod, name: str, **kw):
    """Naming metadata of an exportable module (canonicalize: bitino.name / extra_names / consts / attrs)."""
    mod.bitino_meta = {"name": name, **kw}
    return mod


def _restoration_modules():
    """Exportable data-movement and element-wise blocks of the image-to-image hooks (docs/qonnx-profile.md § 4.6):
    PixelShuffle and the row gather of bitino.adapters.restoration_common, spelled out on NCHW tensors with
    standard ONNX operators (``DepthToSpace`` / ``Resize`` are not profile operators)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    M = _modules()

    class GatherAxis2Fn(torch.autograd.Function):
        """x [1, C, H*W], idx [OH, OW] int64 -> x[:, :, idx] = [1, C, OH, OW], exported as one ONNX Gather(axis 2)."""

        @staticmethod
        def forward(ctx, x, idx):
            return x[:, :, idx]

        @staticmethod
        def symbolic(g, x, idx):
            return g.op("Gather", x, idx, axis_i=2)

    class RowGather(nn.Module):
        """Reshape [1, C, H, W] -> [1, C, H*W] + Gather(axis 2, constant indices): nearest upsampling / replicate
        padding as a gather of pixels (restoration_common._row_gather, the same index map)."""

        def __init__(self, idx: np.ndarray, c: int, h: int, w: int):
            super().__init__()
            idx = np.asarray(idx)
            if idx.ndim != 2 or idx.min() < 0 or idx.max() >= h * w:
                raise ValueError("row gather indices")
            self.register_buffer("idx", torch.from_numpy(idx.astype(np.int64)))
            self.c, self.h, self.w = c, h, w

        def forward(self, x):
            return GatherAxis2Fn.apply(x.reshape(1, self.c, self.h * self.w), self.idx)

    class PixelShuffle(nn.Module):
        """torch.nn.PixelShuffle(r) on [1, C*r*r, H, W] as Reshape [1, C, r, r, H, W] -> Transpose [0, 1, 4, 2, 5, 3]
        -> Reshape [1, C, H*r, W*r] (exact data movement, = torch's pixel_shuffle)."""

        def __init__(self, c_in: int, r: int, h: int, w: int):
            super().__init__()
            if c_in % (r * r):
                raise ValueError("PixelShuffle channels")
            self.c, self.r, self.h, self.w = c_in // (r * r), r, h, w

        def forward(self, x):
            c, r, h, w = self.c, self.r, self.h, self.w
            return x.reshape(1, c, r, r, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, c, h * r, w * r)

    class MulConst(M["Consts"]):
        """x * c[1, C, 1, 1]: c float32 as is, or a float64 carrier (Bitino stores fl32(c))."""

        def __init__(self, c: np.ndarray):
            super().__init__()
            self.setc("k", np.asarray(c), (1, -1, 1, 1))

        def forward(self, x):
            return x * self.c("k", x)

    class Affine32(M["Consts"]):
        """x * s + t with float32 parameters as is (BSRC Scale / Normalize inverse: the source's two fp32 operations)."""

        def __init__(self, s: np.ndarray, t: np.ndarray):
            super().__init__()
            self.setc("s", np.asarray(s), (1, -1, 1, 1))
            self.setc("t", np.asarray(t), (1, -1, 1, 1))

        def forward(self, x):
            return x * self.c("s", x) + self.c("t", x)

    class AddOp(nn.Module):
        def forward(self, a, b):
            return a + b

    class SubOp(nn.Module):
        def forward(self, a, b):
            return a - b

    class LeakyReLU(nn.Module):
        def __init__(self, slope: float):
            super().__init__()
            self.slope = float(slope)

        def forward(self, x):
            return F.leaky_relu(x, self.slope)

    class ReLU(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    return dict(RowGather=RowGather, PixelShuffle=PixelShuffle, MulConst=MulConst, Affine32=Affine32, AddOp=AddOp,
                SubOp=SubOp, LeakyReLU=LeakyReLU, ReLU=ReLU)


class Restoration(Family):
    """Image-to-image families converted by tools/convert_restoration.py (BBCU: convert_bbcu.py, BSRC-2023:
    convert_bsrc.py). Outputs are images (NHWC files, as the legacy reference files); the source references are the
    converter's own (float64: sign bits x float64 scale substituted at F.conv2d, the legacy primary reference)."""
    output_kind = "image"
    key = ""

    def _cv(self):
        import convert_restoration
        return convert_restoration

    def model_ids(self):
        return sorted(self._cv().FAMILIES[self.key]["archs"])

    def load(self, model_id, repo, weights):
        cv = self._cv()
        self._mid = model_id
        model, cls, src = cv.load_model(self.key, repo, model_id, weights)
        return model, {"class": cls, "source_sha256": src, "repo": cv.FAMILIES[self.key]["repo_url"],
                       "repo_commit": cv.repo_commit(repo)}

    def preprocess(self, model_id):
        return self._cv().preprocess_key(self.key, model_id)

    def legacy_graph(self, model, binarize: str, hw):
        """The legacy adapter's graph (structure and coverage checks, the float64 constants, the index maps)."""
        return self._cv().convert_graph(self.key, model, self._mid, binarize, hw)

    def source_logits(self, model, x_nhwc, dtype):
        import torch

        cv = self._cv()
        fam = cv.FAMILIES[self.key]
        override = None
        if dtype == torch.float64:
            _, info0 = self.legacy_graph(model, "ge", tuple(x_nhwc.shape[1:3]))
            override = {n: torch.from_numpy(b.astype(np.float64) * info0["float64_constants"][f"{n}/post_mul"][:, None, None, None])
                        for n, b in info0["binary_weights"].items()}
        y, stats = cv.reference_outputs(model, self.key, x_nhwc, dtype, fam["conv_module"], override=override)
        return y, stats

    def labels(self, model_id, em):
        return "none"

    def task(self, em) -> str:
        it = (em or {}).get("image_task") or {}
        return f"image-to-image ({it.get('kind', 'restoration')}; output NCHW image, range {it.get('output_range')})"


class Bbcu(Restoration):
    """BBCU (Zj-BinXia/BBCU) DnCNN denoise / deblocking, BBCU-L / BBCU-M SR x2 / x4, as tools/convert_bbcu.py loads
    them. Mirrors bitino.adapters.torch_bbcu.convert node for node (the legacy adapter runs first: its module and
    parameter coverage checks, its float64 post_mul values and index maps are reused):

      x * k                 Mul(x, k)                                         (x_times_k, constant k/scale)
      BinaryConv2d          Add(move0) -> Bin -> Pad(0) -> Conv(BipolarQuant(W, 1)) -> Mul(mean|W| c64) [-> Add(bias)]
                            -> Add(pr_bias0) -> PRelu -> Add(pr_bias1) -> Add(x)    (binarizer: torch.sign = Sign, or
                            BipolarQuant with --binarizer ge)
      DnCNN tail            Sub(xk, n) (neg_n + x_minus_n), Mul(fl64(1 / k) c64) (div_k: the constant Bitino
                            stores; torch divides, <= 1 ulp, after the last binarization)
      PixelShuffle          Reshape rank 6 + Transpose + Reshape (§ 4.6)
      nearest x r           Reshape [1, C, H*W] + Gather(axis 2) with the legacy index map
      bilinear x r          replicate pad (row gather) + Conv(3x3 phase kernel, exact dyadic weights) + PixelShuffle
                            (restoration_common.bilinear_upsample; = torch's interpolate up to rounding order)
      LeakyReLU(0.1)        LeakyRelu (alpha's float64 value via bitino.attrs)
    """
    name = "bbcu"
    converter = "tools/convert_bbcu.py (tools/convert_restoration.py family bbcu)"
    key = "bbcu"

    def exportable(self, model, binarizer):
        import torch.nn as nn

        from bitino.adapters import torch_bbcu as TB

        M, R = _modules(), _restoration_modules()
        arch = self._cv().BBCU[self._mid][0]
        n_, c_, h, w = self.input_nchw
        g_leg, info = TB.convert(model, arch, self._mid, input_hw=(h, w), binarize=binarizer)
        f64 = info["float64_constants"]
        lc = g_leg.constants
        rec = []
        k = float(model.k)

        def conv(m, name):
            return _named(M["FloatConv"](m.weight.detach().numpy(), 1, tuple(m.padding),
                                         bias=None if m.bias is None else m.bias.detach().numpy()), name)

        def bias(m, name):
            return _named(M["Bias"](m.bias.detach().numpy().reshape(-1)), name)

        class Core(nn.Module):
            """move0 -> Bin -> HardBinaryConv -> RPReLU (BinaryConv2d / BinaryUpConv2d without the residual)."""

            def __init__(self, m, name):
                super().__init__()
                bc = m.binary_conv
                bn = f"{name}.binary_conv"
                self.move0 = bias(m.move0, f"{name}.move0")
                self.binarize = _named(M["Binarize"](binarizer), bn)
                sign = info["binary_weights"][bn]
                self.binary_conv = _named(M["BinaryWeightConv"](sign, bc.weight.detach().double().numpy(), f64[f"{bn}/post_mul"],
                                                                1, tuple(bc.padding),
                                                                bias=None if bc.bias is None else bc.bias.detach().numpy()), bn)
                rp = m.relu
                self.pr_bias0 = bias(rp.pr_bias0, f"{name}.relu.pr_bias0")
                c = sign.shape[0]
                sl = rp.pr_prelu.weight.detach().numpy().astype(np.float32).reshape(-1)
                self.pr_prelu = _named(M["PReLU"](np.broadcast_to(sl, (c,)).copy()), f"{name}.relu.pr_prelu")
                self.pr_bias1 = bias(rp.pr_bias1, f"{name}.relu.pr_bias1")
                rec.append({"module": bn, "source": "LearnableBias -> BinaryActivation (torch.sign) -> HardBinaryConv (mean|W| x sign(W))"
                                                    + (" + bias" if bc.bias is not None else "") + " -> RPReLU",
                            "export": f"Add -> {'Sign' if binarizer == 'sign_zero' else 'BipolarQuant'} -> Pad(0) -> Conv(BipolarQuant(W, 1)) "
                                      "-> Mul(mean|W| float64 carrier)" + (" -> Add(bias)" if bc.bias is not None else "")
                                      + " -> Add -> PRelu -> Add"})

            def forward(self, x):
                t = self.binary_conv(self.binarize(self.move0(x)))
                return self.pr_bias1(self.pr_prelu(self.pr_bias0(t)))

        class BinBlock(nn.Module):
            def __init__(self, m, name):
                super().__init__()
                self.core = Core(m, name)
                self.add = _named(R["AddOp"](), f"{name}.add")

            def forward(self, x):
                return self.add(self.core(x), x)

        def row_gather(name, c, hh, ww):
            return _named(R["RowGather"](lc[f"{name}/indices"].data, c, hh, ww), name, extra_names=[f"{name}/rows", f"{name}/gather"])

        def shuffle(name, c_in, r, hh, ww):
            return _named(R["PixelShuffle"](c_in, r, hh, ww), name, extra_names=[f"{name}/reshape_in", f"{name}/transpose"])

        class Stage(nn.Module):
            """One upsampling stage: BBCU-L conv -> PixelShuffle -> LeakyReLU; BBCU-M BinaryUpConv2d: core ->
            PixelShuffle, + nearest(x)."""

            def forward(self, t):
                if arch == "bbcul":
                    return self.act(self.shuffle(self.conv(t)))
                return self.add(self.shuffle(self.core(t)), self.nearest(t))

        class Bilinear(nn.Module):
            def __init__(self, c, r, hh, ww):
                super().__init__()
                pk = lc["base_bilinear/phase_kernel"].data                     # OHWI, exact dyadic weights
                self.pad = row_gather("base_bilinear/replicate_pad", c, hh, ww)
                self.phases = _named(M["FloatConv"](np.ascontiguousarray(pk.transpose(0, 3, 1, 2)), 1, 0), "base_bilinear/phases",
                                     consts={"1": "base_bilinear/phase_kernel"})
                self.shuffle = shuffle("base_bilinear", c * r * r, r, hh, ww)

            def forward(self, x):
                return self.shuffle(self.phases(self.pad(x)))

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                c0 = int(c_)
                self.x_times_k = _named(R["MulConst"](np.full(c0, k, np.float32)), "x_times_k", consts={"1": "k/scale"})
                self.inv = _named(R["MulConst"](np.full(c0, 1.0 / k)), "div_k", consts={"1": "inv_k/scale"})   # float64 carrier
                if arch == "dncnn":
                    mods = list(src.model.children())
                    self.head = conv(mods[0], "model.0")
                    self.head_relu = _named(R["ReLU"](), "model.1")
                    self.blocks = nn.ModuleList([BinBlock(mm, f"model.{i}") for i, mm in enumerate(mods) if 2 <= i < 20])
                    self.tail = conv(mods[20], "model.20")
                    self.minus = _named(R["SubOp"](), "neg_n", extra_names=["x_minus_n"], consts={"1": "neg/scale"})
                    return
                lr = float(src.lrelu.negative_slope)
                lrelu = lambda nm: _named(R["LeakyReLU"](lr), nm, attrs={"alpha": lr})   # noqa: E731
                self.conv_first = conv(src.conv_first, "conv_first")
                self.lrelu_first = lrelu("lrelu.conv_first")
                self.blocks = nn.ModuleList([BinBlock(cm, f"body.{bi}.body.{ci}") for bi, blk in enumerate(src.body)
                                             for ci, cm in enumerate(blk.body)])
                up = int(src.upscale)
                r = 2 if up == 4 else up
                nf = int(src.conv_first.weight.shape[0])
                self.stages = nn.ModuleList()
                ch, cw = h, w
                for st in ["upconv1"] + (["upconv2"] if up == 4 else []):
                    m = getattr(src, st)
                    s = Stage()
                    if arch == "bbcul":
                        s.conv = conv(m, st)
                        s.shuffle = shuffle(f"{st}.pixel_shuffle", int(m.weight.shape[0]), r, ch, cw)
                        s.act = lrelu(f"lrelu.{st}")
                    else:
                        s.core = Core(m, st)
                        s.shuffle = shuffle(f"{st}.pixel_shuffle", int(m.binary_conv.weight.shape[0]), r, ch, cw)
                        s.nearest = row_gather(f"{st}.nearest", nf, ch, cw)
                        s.add = _named(R["AddOp"](), f"{st}.add")
                    self.stages.append(s)
                    ch, cw = ch * r, cw * r
                self.hr = conv(src.conv_hr, "conv_hr") if arch == "bbcul" else BinBlock(src.conv_hr, "conv_hr")
                self.lrelu_hr = lrelu("lrelu.conv_hr")
                self.conv_last = conv(src.conv_last, "conv_last")
                self.base = Bilinear(c0, up, h, w)
                self.plus_base = _named(R["AddOp"](), "out_plus_base")

            def forward(self, x):
                xk = self.x_times_k(x)
                if arch == "dncnn":
                    t = self.head_relu(self.head(xk))
                    for b in self.blocks:
                        t = b(t)
                    return self.inv(self.minus(xk, self.tail(t)))
                t = self.lrelu_first(self.conv_first(xk))
                for b in self.blocks:
                    t = b(t)
                for s in self.stages:
                    t = s(t)
                t = self.conv_last(self.lrelu_hr(self.hr(t)))
                return self.inv(self.plus_base(t, self.base(xk)))

        net = Net(model).eval()
        rec.append({"module": "*", "legacy_adapter_op_counts": info["op_counts"], "output_shape_nhwc": info["output_shape"]})
        return net, {"replacements": rec, "import_args": IMPORT_ARGS_IMAGE}


class Bsrc(Restoration):
    """BSRC-2023 binary SR baselines (ilyazhara/BSRC-2023 @62bbcf2) x2 / x4, v0 / v1, as tools/convert_bsrc.py loads
    them. Mirrors bitino.adapters.torch_bsrc.convert node for node (the legacy adapter runs first: coverage checks,
    float64 constants, binary weights):

      BinaryConv2d    Bin(x) = sign(sign(x) + 0.1) (x >= 0 -> +1: BipolarQuant, the source semantics) -> Pad(0) ->
                      Conv(BipolarQuant(W, 1)) -> Mul(mean|W| c64) [-> Add(bias)]
      Scale           Mul(variance) + Add(mean) (one ChannelAffine, the source's two fp32 operations)
      LAB (v1)        depthwise Conv (C -> 2C) -> Conv 1x1 (groups C, weights -1 / +1: d = x1 - x0, exact sign)
                      -> Where(Greater(d, 0), 1, -1) (+1 iff x1 > x0, ties -> -1: argmax picks index 0) -> BConv2D gt
      MeanShift       Add(mean_shift); Normalize (x4 v0) Add(-127.5) + Mul(fl64(1/51) c64) (torch divides, <= 1 ulp);
                      inverse Mul(51) + Add(127.5)
      BatchNorm2d     Mul(s c64) + Add(t c64) (R6)
      PixelShuffle    Reshape rank 6 + Transpose + Reshape (§ 4.6)
    """
    name = "bsrc"
    converter = "tools/convert_bsrc.py (tools/convert_restoration.py family bsrc)"
    key = "bsrc"
    source_binarizer = "ge"

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn

        from bitino.adapters import torch_bsrc as TS

        if binarizer != "ge":
            raise NotImplementedError("BSRC binarizes with sign(sign(x) + 0.1): ge is the source semantics (one file)")
        M, R = _modules(), _restoration_modules()
        spec = TS.MODELS[self._mid]
        arch = spec["arch"]
        _, _, h, w = self.input_nchw
        _, info = TS.convert(model, self._mid, input_hw=(h, w), binarize="ge")
        f64 = info["float64_constants"]
        rec = []
        vec = lambda t: t.detach().cpu().numpy().astype(np.float32).reshape(-1)   # noqa: E731

        def conv(m, name):
            return _named(M["FloatConv"](m.weight.detach().numpy(), 1, tuple(m.padding), groups=int(m.groups),
                                         bias=None if m.bias is None else m.bias.detach().numpy()), name)

        def prelu(m, name, c):
            sl = vec(m.weight)
            return _named(M["PReLU"](np.broadcast_to(sl, (c,)).copy()), name)

        def scale(m, name):
            return _named(R["Affine32"](vec(m.variance), vec(m.mean)), name, consts={"2": f"{name}/bias"})

        class GtBinarize(nn.Module):
            """LAB decision: Where(Greater(d, 0), 1, -1)."""

            def __init__(self):
                super().__init__()
                self.register_buffer("zero", torch.tensor(0.0))
                self.register_buffer("one", torch.tensor(1.0))
                self.register_buffer("mone", torch.tensor(-1.0))

            def forward(self, d):
                return torch.where(d > self.zero, self.one, self.mone)

        class BConv(nn.Module):
            def __init__(self, m, name, gt: bool):
                super().__init__()
                bits = info["binary_weights"][name]
                lat = m.weight.detach().double().numpy()
                if np.any(lat == 0):                 # sign(sign(W) + 0.1): 0 -> +1, the decoded ±1 as the latent
                    lat = bits.astype(np.float64)
                self.binarize = _named(GtBinarize() if gt else M["Binarize"]("ge"), name)
                self.conv = _named(M["BinaryWeightConv"](bits, lat, f64[f"{name}/post_mul"], 1, tuple(m.padding),
                                                         bias=None if m.bias is None else m.bias.detach().numpy()), name)
                rec.append({"module": name, "source": "BinaryConv2d (input / weight Binarizer, L1Scaling)",
                            "export": ("Where(Greater(d, 0), 1, -1)" if gt else "BipolarQuant") + " -> Pad(0) -> "
                                      "Conv(BipolarQuant(W, 1)) -> Mul(mean|W| float64 carrier)" + (" -> Add(bias)" if m.bias is not None else "")})

            def forward(self, x):
                return self.conv(self.binarize(x))

        class Lab(nn.Module):
            def __init__(self, m, name, c):
                super().__init__()
                self.dw = conv(m.fp_depthwise_conv, f"{name}.fp_depthwise_conv")
                k = np.zeros((c, 2, 1, 1), np.float32)
                k[:, 0], k[:, 1] = -1.0, 1.0
                self.diff = _named(M["FloatConv"](k, 1, 0, groups=c), f"{name}.diff")

            def forward(self, x):
                return self.diff(self.dw(x))

        class ConvBlock(nn.Module):
            def __init__(self, m, name, c):
                super().__init__()
                self.add = _named(R["AddOp"](), f"{name}.add")
                if arch.endswith("v0"):
                    cp = m.conv_path
                    self.b = BConv(cp[0], f"{name}.conv_path.0", False)
                    self.sc = scale(cp[1], f"{name}.conv_path.1")
                    self.act = prelu(cp[2], f"{name}.conv_path.2", c)
                    return
                bp = m.bin_path
                self.lab = Lab(bp[0], f"{name}.bin_path.0", c)
                self.b = BConv(bp[1], f"{name}.bin_path.1", True)
                self.sc = scale(bp[2], f"{name}.bin_path.2")
                self.p1 = conv(m.fp_1x1_conv, f"{name}.fp_1x1_conv")
                self.p2 = conv(m.fp_depthwise_conv, f"{name}.fp_depthwise_conv")
                self.add01 = _named(R["AddOp"](), f"{name}.add01")
                if arch == "x2v1":
                    self.add012 = _named(R["AddOp"](), f"{name}.add012")
                self.act = prelu(m.act, f"{name}.act", c)

            def forward(self, x):
                if arch.endswith("v0"):
                    return self.add(self.act(self.sc(self.b(x))), x)
                p0 = self.sc(self.b(self.lab(x)))
                p1 = self.p1(x)
                p2 = self.p2(p1)
                s = self.add012(self.add01(p0, p1), p2) if arch == "x2v1" else self.add01(p0, p2)
                return self.add(self.act(s), x)

        class ResBlock(nn.Module):
            def __init__(self, m, name, c):
                super().__init__()
                self.fp = type(m).__name__ == "ResidualBlockFP"
                if self.fp:
                    cb = m.conv_block
                    self.c0 = conv(cb[0], f"{name}.conv_block.0")
                    self.relu = _named(R["ReLU"](), f"{name}.conv_block.1")
                    self.c2 = conv(cb[2], f"{name}.conv_block.2")
                    self.add = _named(R["AddOp"](), f"{name}.add")
                else:
                    self.blocks = nn.ModuleList([ConvBlock(m.res_block[i], f"{name}.res_block.{i}", c) for i in range(2)])

            def forward(self, x):
                if self.fp:
                    return self.add(x, self.c2(self.relu(self.c0(x))))
                for b in self.blocks:
                    x = b(x)
                return x

        class Up(nn.Module):
            """Upsampler / upsampling: (Conv2d(C -> 4C) + PixelShuffle(2) [+ PReLU]) x log2(scale)."""

            def __init__(self, seq, name, c, prelus):
                super().__init__()
                mods = list(seq.named_children())
                step = 3 if prelus else 2
                self.parts = nn.ModuleList()
                ch, cw = h, w
                for i in range(0, len(mods), step):
                    (cn, cm), (pn, _pm) = mods[i], mods[i + 1]
                    self.parts.append(conv(cm, f"{name}.{cn}"))
                    nm = f"{name}.{pn}"
                    self.parts.append(_named(R["PixelShuffle"](int(cm.weight.shape[0]), 2, ch, cw), nm,
                                             extra_names=[f"{nm}/reshape_in", f"{nm}/transpose"]))
                    ch, cw = ch * 2, cw * 2
                    if prelus:
                        self.parts.append(prelu(mods[i + 2][1], f"{name}.{mods[i + 2][0]}", c))

            def forward(self, x):
                for p in self.parts:
                    x = p(x)
                return x

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                if arch.startswith("x2"):
                    self.sub_mean = _named(M["Bias"](vec(src.sub_mean.mean_shift)), "sub_mean")
                    self.head = conv(src.head[0], "head.0")
                    c = int(src.head[0].weight.shape[0])
                    body = list(src.body.named_children())
                    self.body = nn.ModuleList([ResBlock(m, f"body.{nm}", c) for nm, m in body[:-1]])
                    self.body_last = conv(body[-1][1], f"body.{body[-1][0]}")
                    self.body_residual = _named(R["AddOp"](), "body_residual")
                    self.up = Up(src.tail[0], "tail.0", c, prelus=False)
                    self.tail = conv(src.tail[1], "tail.1")
                    self.add_mean = _named(M["Bias"](vec(src.add_mean.mean_shift)), "add_mean")
                    return
                if arch == "x4v0":
                    tr, it = src.transform, src.inverse_transform
                    self.t_sub = _named(M["Bias"](np.full(3, -float(tr.mean_shift), np.float32)), "transform/sub")
                    self.t_div = _named(R["MulConst"](f64["transform/div/scale"]), "transform/div")   # float64 carrier
                    self.inv = _named(R["Affine32"](np.full(3, float(it.scale_factor), np.float32),
                                                    np.full(3, float(it.mean_shift), np.float32)), "inverse_transform",
                                      consts={"2": "inverse_transform/bias"})
                c = int(src.conv1[0].weight.shape[0])
                self.conv1 = conv(src.conv1[0], "conv1.0")
                self.act1 = prelu(src.conv1[1], "conv1.1", c)
                self.res = nn.ModuleList([ResBlock(m, f"res_blocks.{nm}", c) for nm, m in src.res_blocks.named_children()])
                self.conv2 = conv(src.conv2[0], "conv2.0")
                bn = src.conv2[1]
                self.bn2 = _named(M["Affine"](f64["conv2.1/scale"], f64["conv2.1/shift"], 4), "conv2.1")
                if not (np.array_equal(f64["conv2.1/scale"], bn_affine64(bn)[0]) and np.array_equal(f64["conv2.1/shift"], bn_affine64(bn)[1])):
                    raise AssertionError("conv2.1: BatchNorm fold differs from the adapter's")
                self.add12 = _named(R["AddOp"](), "add_out1_out2")
                self.up = Up(src.upsampling, "upsampling", c, prelus=True)
                self.conv3 = conv(src.conv3[0], "conv3.0")

            def forward(self, x):
                if arch.startswith("x2"):
                    t = self.head(self.sub_mean(x))
                    b = t
                    for blk in self.body:
                        b = blk(b)
                    t = self.body_residual(t, self.body_last(b))
                    return self.add_mean(self.tail(self.up(t)))
                t = self.t_div(self.t_sub(x)) if arch == "x4v0" else x
                out1 = self.act1(self.conv1(t))
                r = out1
                for blk in self.res:
                    r = blk(r)
                t = self.add12(out1, self.bn2(self.conv2(r)))
                t = self.conv3(self.up(t))
                return self.inv(t) if arch == "x4v0" else t

        net = Net(model).eval()
        rec.append({"module": "*", "legacy_adapter_op_counts": info["op_counts"], "output_shape_nhwc": info["output_shape"],
                    "unused_in_forward": info["unused_in_forward"]})
        return net, {"replacements": rec, "import_args": IMPORT_ARGS_IMAGE}


class Bnas(Family):
    """BNAS cell networks (yonseivnl/bnas @ cdb58f7, the first-release BinOp the checkpoints were evaluated with):
    BNAS-A (CIFAR-10), BNAS-D / E / H (ImageNet), as tools/convert_bnas.py loads them (official model, checkpoint,
    BinOp.binarization()). The adapter (bitino.adapters.torch_bnas) is not a module-by-module mirror: it makes
    graph-level decisions (BN sign folded into the next conv's weights, the ge zero-channel correction, the BinOp
    partial-binarization quirk convs, shared identical pools, dropped dead branches, all-zero cell states). This hook
    therefore translates the adapter's own IR graph (before the graph passes) node by node into exportable modules,
    so every one of those decisions is reproduced by construction; the export check compares the canonical file with
    the authors' unmodified model (fp32 / fp64):

      ChannelAffine(s, t)        Mul(x, s) / Add(x, t), float64 carriers where the adapter folded in float64 (BN,
                                 b/s thresholds), the zero state Mul(x, 0)
      BConv2D (sign(s) folded)   Bin(x) -> Pad(0) -> Conv(BipolarQuant(bits, 1)) -> Mul(post_mul c64) [-> Relu]
                                 (bits with the BN-sign flip; post_mul = the exact fp32 XNOR scale)
      ge zero-channel correction BConv2D -> Add(map c64 [1, O, H, W]) -> Relu  (profile § 4.3: Add of a constant map)
      quirk conv                 Sign -> Pad(0) -> Conv(float latent weights, flipped columns) -> Relu, declared
                                 bitino.float_layer (profile § 7): Activation(sign_zero) + Conv2D in both variants
      pools / Concat / Add       MaxPool / AveragePool(count_include_pad 0) / Concat / Add; GlobalAveragePool +
                                 Flatten + Gemm (the classifier rows in the standard ImageNet class order)
    """
    name = "bnas"
    converter = "tools/convert_bnas.py"

    def _cv(self):
        import convert_bnas
        return convert_bnas

    def model_ids(self):
        return sorted(self._cv().MODELS)

    def load(self, model_id, repo, weights):
        cv = self._cv()
        self._mid = model_id
        model, targets, info = cv.load_source(model_id, repo, weights)
        self._targets = targets
        self._perm = cv.imagenet_class_perm() if cv.MODELS[model_id][0] == "imagenet" else None
        return model, {"class": info["class"], "source_sha256": info["source_sha256"], "repo": cv.REPO_URL,
                       "repo_commit": cv.repo_commit(repo), "binop": info["binop"]}

    def preprocess(self, model_id):
        return self._cv().MODELS[model_id][2]

    def _legacy(self, model, binarize):
        from bitino.adapters import torch_bnas as TB
        return TB.convert(model, self._targets, self._mid, binarize=binarize, class_perm=self._perm)

    def source_logits(self, model, x_nhwc, dtype):
        if getattr(self, "_points", None) is None:        # computed once (--chunk calls this per chunk)
            self._points = self._legacy(model, "sign_zero")[1]["binarization_points"]
        return self._cv().run_reference(model, x_nhwc, dtype, self._points, self._perm)

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn

        from bitino.ir import unpack_bits

        M, R = _modules(), _restoration_modules()
        g_leg, info = self._legacy(model, binarizer)
        f64 = info["float64_constants"]
        C = g_leg.constants
        rec = []

        def arr(name):            # the adapter's float64 value (a float64 carrier) where it folded one, else as stored
            return np.asarray(f64[name], np.float64) if name in f64 else np.asarray(C[name].data)

        def bits_oihw(name):
            k = C[name]
            return np.ascontiguousarray(unpack_bits(k.data, k.shape[-1]).transpose(0, 3, 1, 2))

        def tpad(at):
            t, b, l, r = at["pads"]
            if t != b or l != r:
                raise NotImplementedError(f"asymmetric pads {at['pads']}")
            return (t, l)

        class ChAff(M["Consts"]):
            def __init__(self, s, t):
                super().__init__()
                self.has_s, self.has_t = s is not None, t is not None
                if self.has_s:
                    self.setc("s", s, (1, -1, 1, 1))
                if self.has_t:
                    self.setc("t", t, (1, -1, 1, 1))

            def forward(self, x):
                if self.has_s:
                    x = x * self.c("s", x)
                return x + self.c("t", x) if self.has_t else x

        class AddMap(M["Consts"]):
            def __init__(self, m_nchw):
                super().__init__()
                self.setc("m", m_nchw, tuple(m_nchw.shape))

            def forward(self, x):
                return x + self.c("m", x)

        class Cat(nn.Module):
            def forward(self, *xs):
                return torch.cat(xs, 1)

        class Gap(nn.Module):
            def forward(self, x):
                return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)

        class Seq(nn.Module):
            def __init__(self, *mods):
                super().__init__()
                self.parts = nn.ModuleList(mods)

            def forward(self, x):
                for p in self.parts:
                    x = p(x)
                return x

        producer = {o: n for n in g_leg.nodes for o in n.outputs}
        consumers: dict = {}
        for n in g_leg.nodes:
            for i in n.inputs:
                if i:
                    consumers.setdefault(i, []).append(n)
        ops, plan = nn.ModuleList(), []
        n_quirk = n_corr = 0
        for n in g_leg.nodes:
            at, ins = n.attrs, n.inputs
            act = at.get("activation", "none")
            if act not in ("none", "relu"):
                raise NotImplementedError(f"{n.name}: activation {act}")
            relu = [_named(R["ReLU"](), f"{n.name}/act")] if act == "relu" else []
            if n.op == "ChannelAffine":
                sc, sh = (ins + ["", ""])[1:3]
                consts = {k: v for k, v in (("1", sc), ("2", sh)) if v}
                mod = _named(ChAff(arr(sc) if sc else None, arr(sh) if sh else None), n.name, consts=consts)
                src = [ins[0]]
            elif n.op == "BConv2D":
                kb, pm, pa = ins[1], ins[2], ins[3]
                if (len(ins) > 4 and ins[4]) or at["groups"] != 1:
                    raise NotImplementedError(f"{n.name}: threshold / groups")
                bits = bits_oihw(kb)
                consts = {"1": kb, "2": pm, **({"3": pa} if pa else {})}
                conv = _named(M["BinaryWeightConv"](bits, bits.astype(np.float64), arr(pm), tuple(at["strides"]), tpad(at),
                                                    dilation=tuple(at["dilations"]), bias=arr(pa) if pa else None),
                              n.name, consts=consts, attrs={"pad_value": 0.0})
                mod = Seq(_named(M["Binarize"](at["input_binarize"]), n.name), conv, *relu)
                src = [ins[0]]
            elif n.op == "Conv2D":
                k, b = ins[1], ins[2] if len(ins) > 2 else ""
                kern = np.ascontiguousarray(np.asarray(C[k].data).transpose(0, 3, 1, 2))
                consts = {"1": k, **({"2": b} if b else {})}
                p = producer.get(ins[0])
                quirk = p is not None and p.op == "Activation" and p.attrs.get("type") == "sign_zero"
                conv = _named(M["FloatConv"](kern, tuple(at["strides"]), tpad(at), tuple(at["dilations"]), at["groups"],
                                             bias=np.asarray(C[b].data) if b else None), n.name, consts=consts,
                              **({"extra_names": [p.name], "float_layer": True} if quirk else {}))
                if quirk:             # Activation(sign_zero) + float conv: Sign -> Pad(0) -> Conv, declared float layer
                    n_quirk += 1
                    mod = Seq(_named(M["Binarize"]("sign_zero"), n.name), conv, *relu)
                    src = [p.inputs[0]]
                else:
                    mod = Seq(conv, *relu)
                    src = [ins[0]]
            elif n.op == "Activation" and at.get("type") == "sign_zero":
                if [c.op for c in consumers.get(n.outputs[0], [])] != ["Conv2D"]:
                    raise NotImplementedError(f"{n.name}: sign_zero activation not feeding one Conv2D")
                continue                                       # emitted by its Conv2D (quirk conv)
            elif n.op == "Activation" and at.get("type") == "relu":
                mod, src = _named(R["ReLU"](), n.name), [ins[0]]
            elif n.op == "Add":
                if len(ins) == 2 and all(i not in C for i in ins):
                    mod, src = _named(R["AddOp"](), n.name), list(ins)
                elif len(ins) == 2 and ins[1] in C and ins[0] not in C:
                    m4 = arr(ins[1])
                    mod = Seq(_named(AddMap(np.ascontiguousarray(m4.transpose(0, 3, 1, 2))), n.name, consts={"1": ins[1]}), *relu)
                    src = [ins[0]]
                    n_corr += 1
                else:
                    raise NotImplementedError(f"{n.name}: Add inputs {ins}")
            elif n.op == "Concat":
                mod, src = _named(Cat(), n.name), list(ins)
            elif n.op == "MaxPool2D":
                mod = _named(nn.MaxPool2d(tuple(at["kernel"]), tuple(at["strides"]), tpad(at)), n.name)
                src = [ins[0]]
            elif n.op == "AvgPool2D":
                mod = _named(nn.AvgPool2d(tuple(at["kernel"]), tuple(at["strides"]), tpad(at),
                                          count_include_pad=bool(at["count_include_pad"])), n.name)
                src = [ins[0]]
            elif n.op == "GlobalAvgPool" and at.get("keepdims") == 0:
                mod, src = _named(Gap(), n.name), [ins[0]]
            elif n.op == "Dense":
                lin = nn.Linear(C[ins[1]].shape[1], C[ins[1]].shape[0])
                with torch.no_grad():
                    lin.weight.copy_(torch.from_numpy(np.asarray(C[ins[1]].data)))
                    lin.bias.copy_(torch.from_numpy(np.asarray(C[ins[2]].data)))
                mod = Seq(_named(lin, n.name, consts={"1": ins[1], "2": ins[2]}), *relu)
                src = [ins[0]]
            else:
                raise NotImplementedError(f"{n.name}: IR op {n.op} {at}")
            plan.append((len(ops), src, n.outputs[0]))
            ops.append(mod)

        out_name = g_leg.outputs[0]
        in_name = g_leg.inputs[0]["name"]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.ops = ops

            def forward(self, x):
                env = {in_name: x}
                for j, src, dst in plan:
                    env[dst] = self.ops[j](*[env[s] for s in src])
                return env[out_name]

        net = Net().eval()
        rec.append({"module": "*", "translation": "legacy adapter IR graph (before the passes) -> exportable modules",
                    "legacy_adapter_op_counts": info["op_counts"], "quirk_float_layers": n_quirk, "zero_channel_corrections": n_corr,
                    "bn_sign_flip_channels": int(sum(int(v.sum()) for v in info["bn_sign_flips"].values())),
                    "unused_in_forward (dead branches)": info["unused_in_forward"]})
        return net, {"replacements": rec, "import_args": IMPORT_ARGS_TORCH}


class Adabin(Family):
    """AdaBin CIFAR-10 ResNet-20 (huawei-noah/Efficient-Computing BinaryNetworks/AdaBin/cifar10, utils/binarylib.py,
    nets/resnet20.py), as tools/convert_adabin.py (tools/convert_s12_cifar.py family adabin) loads it. Mirrors
    bitino.adapters.torch_adabin.convert (which runs first: coverage checks, the weight levels):

      BinaryActivation   a = alpha_a * (Bin(x - beta_a) + beta_a): Sub -> Sign (or BipolarQuant with --binarizer ge)
                         -> Add -> Mul, the source's fp32 operations on the sign; the source binarizes
                         (x - beta_a) / alpha_a, the same sign for alpha_a > 0 except where the fp32 quotient underflows
                         to 0 (|x - beta_a| < alpha_a * 2^-150), dropped as the adapter does (the AB-BNN beta precedent)
      AdaBin_Conv2d      Conv(a, W_b) with W_b = sign(wn) * alpha_w + beta_w the source's fp32 two-level weights
                         (adabin_weight_levels), zero padding (the Conv's pads: a is not a binarized tensor)
      BatchNorm2d        Mul(s c64) + Add(t c64) (R6); the importer folds it into the AdaBin decomposition (profile § 4.1)
      Maxout             Relu(x) * pos + Relu(x * -1) * (-neg) (bit-exact with pos*relu(x) - neg*relu(-x))
      shortcut (A)       AveragePool(1, stride 2) + channel Pad; head AdaptiveAvgPool + BatchNorm1d + Linear
    """
    name = "adabin"
    converter = "tools/convert_adabin.py (tools/convert_s12_cifar.py family adabin)"

    def _cv(self):
        import convert_s12_cifar
        return convert_s12_cifar

    def model_ids(self):
        return sorted(self._cv().FAMILIES["adabin"]["archs"])

    def load(self, model_id, repo, weights):
        cv = self._cv()
        self._mid = model_id
        model, cls, src, _notes = cv.load_source("adabin", repo, weights)
        return model, {"class": cls, "source_sha256": src, "repo": cv.FAMILIES["adabin"]["repo_url"], "repo_commit": cv.repo_commit(repo)}

    def preprocess(self, model_id):
        return self._cv().FAMILIES["adabin"]["preprocess"]

    def source_logits(self, model, x_nhwc, dtype):
        """float32: the source as is (binary weights captured at F.conv2d); float64: the source with the fp32 weight
        levels substituted (convert_s12_cifar's primary float64 reference)."""
        import torch

        cv = self._cv()
        if dtype == torch.float32:
            self._cap32 = {}
            return cv.run_reference("adabin", model, x_nhwc, dtype, capture=self._cap32)
        return cv.run_reference("adabin", model, x_nhwc, dtype, override={k: v.double() for k, v in self._cap32.items()})

    def exportable(self, model, binarizer):
        import torch
        import torch.nn as nn

        from bitino.adapters import torch_adabin as TA

        M, R = _modules(), _restoration_modules()
        g_leg, info = TA.convert(model, self._mid, binarize=binarizer)      # coverage checks, constant order
        rec = []
        f32 = lambda t: t.detach().cpu().numpy().astype(np.float32)   # noqa: E731

        def scal(t):
            v = np.unique(f32(t).reshape(-1))
            if v.size != 1:
                raise NotImplementedError("AdaBin: alpha_a / beta_a must be scalars")
            return torch.tensor(float(v[0]), dtype=torch.float32)

        def conv(m, name):
            return _named(M["FloatConv"](f32(m.weight), tuple(m.stride), tuple(m.padding)), name)

        def bn(m, name, rank=4):
            return _named(M["Affine"](*bn_affine64(m), rank), name)

        class AdaConv(nn.Module):
            def __init__(self, m, name):
                super().__init__()
                alpha, beta = scal(m.binary_a.alpha_a), scal(m.binary_a.beta_a)
                if not float(alpha) > 0:
                    raise NotImplementedError(f"{name}: alpha_a <= 0")
                self.register_buffer("alpha", alpha)
                self.register_buffer("beta", beta)
                sign, wp, wm = TA.adabin_weight_levels(m)
                wb = np.where(sign > 0, wp.reshape(-1, 1, 1, 1), wm.reshape(-1, 1, 1, 1)).astype(np.float32)
                if not np.array_equal(sign, info["binary_weights"][name]):
                    raise AssertionError(f"{name}: weight signs differ from the adapter's")
                self.bin = M["Binarize"](binarizer)
                self.conv = M["FloatConv"](wb, tuple(m.stride), tuple(m.padding))
                rec.append({"module": name, "source": "AdaBin_Conv2d (BinaryActivation alpha*(sign((x-beta)/alpha)+beta), "
                                                      "two-level weights)", "alpha_a": float(alpha), "beta_a": float(beta),
                            "export": f"Sub(beta) -> {'Sign' if binarizer == 'sign_zero' else 'BipolarQuant'} -> Add(beta) -> "
                                      "Mul(alpha) -> Conv(W_b two-level fp32, zero pads)"})

            def forward(self, x):
                return self.conv((self.bin(x - self.beta) + self.beta) * self.alpha)

        def adaconv(m, name):
            return _named(AdaConv(m, name), name, extra_names=[f"{name}/sum_s", f"{name}/sum_s_broadcast", f"{name}/add_sum_s"])

        class Maxout(nn.Module):
            def __init__(self, m, name):
                super().__init__()
                pos, neg = f32(m.pos_scale).reshape(-1), f32(m.neg_scale).reshape(-1)
                c = pos.size
                self.relu_pos = _named(R["ReLU"](), f"{name}/relu_pos")
                self.pos = _named(R["MulConst"](pos), f"{name}/pos", consts={"1": f"{name}/pos_scale"})
                self.negate = _named(R["MulConst"](-np.ones(c, np.float32)), f"{name}/negate", consts={"1": f"{name}/minus_one"})
                self.relu_neg = _named(R["ReLU"](), f"{name}/relu_neg")
                self.neg = _named(R["MulConst"](-neg), f"{name}/neg", consts={"1": f"{name}/minus_neg_scale"})
                self.add = _named(R["AddOp"](), name)

            def forward(self, x):
                return self.add(self.pos(self.relu_pos(x)), self.neg(self.relu_neg(self.negate(x))))

        class Shortcut(nn.Module):
            def __init__(self, blk, pth, cin, cout, stride):
                super().__init__()
                sc = blk.shortcut
                self.ident = type(sc).__name__ == "Sequential" and len(sc) == 0
                if self.ident:
                    if stride != 1 or cin != cout:
                        raise NotImplementedError(f"{pth}: identity shortcut with a shape change")
                    return
                if type(sc).__name__ != "LambdaLayer":
                    raise NotImplementedError(f"{pth}.shortcut: only option A")
                self.pad = (cout - cin) // 2
                self.sub = _named(nn.AvgPool2d(1, 2), f"{pth}.shortcut.subsample")
                self.padm = _named(_PadMod(self.pad), f"{pth}.shortcut.pad")
                probe = torch.randn(1, cin, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
                with torch.no_grad():
                    if not torch.equal(sc(probe), self.padm(self.sub(probe))):
                        raise AssertionError(f"{pth}.shortcut: AveragePool(1, 2) + Pad differs from the option-A lambda")

            def forward(self, x):
                return x if self.ident else self.padm(self.sub(x))

        class _PadMod(nn.Module):
            def __init__(self, pad):
                super().__init__()
                self.p = pad

            def forward(self, x):
                return M["PadFn"].apply(x, (0, self.p, 0, 0, 0, self.p, 0, 0), 0.0)

        class Block(nn.Module):
            def __init__(self, blk, pth, cin):
                super().__init__()
                cout, st = blk.conv1.out_channels, int(tuple(blk.conv1.stride)[0])
                self.conv1, self.bn1 = adaconv(blk.conv1, f"{pth}.conv1"), bn(blk.bn1, f"{pth}.bn1")
                self.shortcut = Shortcut(blk, pth, cin, cout, st)
                self.add1 = _named(R["AddOp"](), f"{pth}.add1")
                self.nonlinear1 = Maxout(blk.nonlinear1, f"{pth}.nonlinear1")
                self.conv2, self.bn2 = adaconv(blk.conv2, f"{pth}.conv2"), bn(blk.bn2, f"{pth}.bn2")
                self.add2 = _named(R["AddOp"](), f"{pth}.add2")
                self.nonlinear2 = Maxout(blk.nonlinear2, f"{pth}.nonlinear2")

            def forward(self, x):
                u = self.nonlinear1(self.add1(self.bn1(self.conv1(x)), self.shortcut(x)))
                return self.nonlinear2(self.add2(self.bn2(self.conv2(u)), u))

        class Gap(nn.Module):
            def forward(self, x):
                return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)

        class Net(nn.Module):
            def __init__(self, src):
                super().__init__()
                self.conv1, self.bn1 = conv(src.conv1, "conv1"), bn(src.bn1, "bn1")
                self.nonlinear1 = Maxout(src.nonlinear1, "nonlinear1")
                blocks, c = [], src.conv1.out_channels
                for ln in ("layer1", "layer2", "layer3"):
                    for bi, blk in enumerate(getattr(src, ln)):
                        blocks.append(Block(blk, f"{ln}.{bi}", c))
                        c = blk.conv1.out_channels
                self.blocks = nn.ModuleList(blocks)
                self.avgpool = _named(Gap(), "avgpool")
                self.bn2 = bn(src.bn2, "bn2", rank=2)
                self.linear = _named(copy.deepcopy(src.linear), "linear")

            def forward(self, x):
                x = self.nonlinear1(self.bn1(self.conv1(x)))
                for b in self.blocks:
                    x = b(x)
                return self.linear(self.bn2(self.avgpool(x)))

        net = Net(model).eval()
        # data-section order of the legacy file (creation order of the adapter: threshold before ones_bits, Dense kernel
        # before bias): neither the importer's walk nor first-use order; bitino.const_order (profile § 7) is the final
        # data order, after the graph passes (the legacy converter's passes on the adapter's graph)
        from bitino import passes
        passes.optimize(g_leg, fold=True)
        return net, {"replacements": rec, "import_args": IMPORT_ARGS_TORCH, "const_order": list(g_leg.constants)}


FAMILIES = {f.name: f for f in (AbBnn(), Rbonn(), Irnet(), Sabnn(), ReactA(), Bnext(), Bitorch(), Bbcu(), Bsrc(), Bnas(), Adabin())}


def family_of(model_id: str) -> Family:
    for f in FAMILIES.values():
        if model_id in f.model_ids():
            return f
    raise SystemExit(f"no export hook covers {model_id} (families: {sorted(FAMILIES)})")


# ================================================================================================ ONNX
PROFILE = "bitino-qonnx/1"
IR_VERSION = 10                      # NodeProto.metadata_props (docs/qonnx-profile.md § 1)
_MOVE = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Identity", "Pad", "MaxPool", "Concat", "Slice", "Gather",
         "Cast"}


def _node_names(m, net, legacy_names=None) -> list:
    """Naming metadata of every node (dict: name, extra_names, consts, attrs). The deepest module on the node's
    scope that has a ``bitino_meta`` attribute names it: ``{"name": ..., "extra_names": [...], "consts": {...},
    "attrs": {...}}`` for all of the module's nodes, or ``{"by_op": {op type: {...}}}`` per ONNX op type (a module
    that emits several IR groups, e.g. the IR-Net activation centering). Otherwise: the module path for the single
    computing op of a leaf module, else ``<module path>.<op type lower>``; a hook's ``legacy_names`` maps those onto
    the legacy converter's names."""
    mods = dict(net.named_modules())
    leaf = {n for n, mm in mods.items() if n and not any(True for _ in mm.children())}
    out = []
    for nd in m.graph.node:
        # "/layer2/layer2.0/downsample/downsample.0/AveragePool": one scope per module, named by its attribute name;
        # a Sequential child is "<sequential attribute>.<index>" -> module path layer2.0.downsample.0
        seg: list = []
        for p in [p for p in nd.name.split("/") if p][:-1]:
            if seg and p.startswith(seg[-1] + "."):
                seg[-1] = p
            else:
                seg.append(p)
        path = ".".join(seg)
        if path and path not in mods:
            raise AssertionError(f"node {nd.name}: scope {path} is not a module path")
        meta = None
        q = path
        while True:
            mm = mods.get(q)
            bm = getattr(mm, "bitino_meta", None) if mm is not None else None
            if bm:
                if "by_op" in bm:
                    sub = bm["by_op"].get(nd.op_type) or bm.get("default")
                    meta = dict(sub) if sub else None
                else:
                    meta = dict(bm)
                if meta is not None:
                    break
            if not q:
                break
            q = q.rsplit(".", 1)[0] if "." in q else ""
        if meta is None:
            name = path if path in leaf else (f"{path}.{nd.op_type.lower()}" if path else nd.op_type.lower())
            if legacy_names is not None:
                name = legacy_names(path, nd.op_type, name, nd.name)
            meta = {"name": name}
        out.append(meta)
    return out


def _set_props(nd, props: dict) -> None:
    import onnx

    keep = [p for p in nd.metadata_props if p.key not in props]
    del nd.metadata_props[:]
    nd.metadata_props.extend(keep)
    for k, v in props.items():
        if v is not None:
            nd.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))


def canonicalize(onnx_path: str, net, hook: dict, model_meta: dict) -> dict:
    """Rewrite torch.onnx.export's file into a canonical file of the Bitino QONNX profile (docs/qonnx-profile.md):

    * ONNX IR version 10 (node metadata_props), ai.onnx 17 + qonnx.custom_op.general 1;
    * Constant nodes -> initializers; Identity of an initializer (torch's de-duplicated buffers) -> the initializer;
      unused initializers dropped;
    * R2: every Conv that pads a binarized tensor gets an explicit Pad(value 0) of that tensor (the PyTorch zero
      padding after the binarizer) and pads 0;
    * naming metadata (§ 7) on every activation-path node: ``bitino.name`` (module path / the legacy converter's
      name), ``bitino.order`` (the same on every node of a group: the exporter position of the group's first node),
      ``bitino.attrs`` from the hook (the legacy attribute JSON form, e.g. pad_value 0.0 of an unpadded binary conv);
      constant-only nodes carry none;
    * model metadata (§ 6) from ``model_meta`` (+ ``bitino.ir_unique_counter`` from the hook's legacy names).
    Returns a summary for export.json."""
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(onnx_path)
    g = m.graph
    metas = _node_names(m, net, hook.get("legacy_names"))
    names = [mt["name"] for mt in metas]
    named = []
    meta_of: dict = {}
    for nd, nm, mt in zip(g.node, names, metas):   # detached copies: g.node is rebuilt below
        c = onnx.NodeProto()
        c.CopyFrom(nd)
        named.append((c, nm))
        meta_of[id(c)] = mt
    # ---- constants: Constant nodes -> initializers, Identity(initializer) -> alias
    inits = {i.name for i in g.initializer}
    alias: dict = {}
    keep_nodes = []
    for nd, nm in named:
        ins = [alias.get(i, i) for i in nd.input]
        del nd.input[:]
        nd.input.extend(ins)
        if nd.op_type == "Constant" and len(nd.attribute) == 1 and nd.attribute[0].name == "value":
            t = onnx.TensorProto()
            t.CopyFrom(nd.attribute[0].t)
            t.name = nd.output[0]
            g.initializer.append(t)
            inits.add(t.name)
            continue
        if nd.op_type == "Identity" and nd.input[0] in inits and nd.output[0] not in {o.name for o in g.output}:
            alias[nd.output[0]] = nd.input[0]
            continue
        keep_nodes.append((nd, nm))
    for o in g.output:
        if o.name in alias:
            raise AssertionError(f"graph output {o.name} is a constant")
    # ---- dynamic (input-dependent) tensors and binarized tensors
    dyn = {i.name for i in g.input if i.name not in inits}
    binarized: set = set()
    producer = {o: nd for nd, _ in keep_nodes for o in nd.output if o}
    for nd, _ in keep_nodes:
        if any(i in dyn for i in nd.input if i):
            dyn.update(o for o in nd.output if o)
            if nd.op_type in ("Sign", "BipolarQuant") and nd.input[0] in dyn:
                binarized.update(nd.output)
            elif nd.op_type == "Where" and nd.input[0] in producer and \
                    producer[nd.input[0]].op_type in ("Greater", "GreaterOrEqual"):      # R1 Where(Greater(x, t), 1, -1)
                binarized.update(nd.output)
            elif nd.op_type in _MOVE and nd.input[0] in binarized:
                binarized.update(nd.output)
    # ---- R2: explicit Pad of a binarized Conv input
    new_nodes = []
    n_pad = 0
    for nd, nm in keep_nodes:
        if nd.op_type == "Conv" and nd.input[0] in binarized:
            at = {a.name: a for a in nd.attribute}
            pads = list(at["pads"].ints) if "pads" in at else [0, 0, 0, 0]
            if any(pads):
                t, l, b, r = pads
                pin = f"{nd.name}/pads"
                vin = f"{nd.name}/pad_value"
                g.initializer.extend([numpy_helper.from_array(np.array([0, 0, t, l, 0, 0, b, r], np.int64), pin),
                                      numpy_helper.from_array(np.array(0.0, np.float32), vin)])
                po = f"{nd.name}/padded"
                pad = helper.make_node("Pad", [nd.input[0], pin, vin], [po], name=f"{nd.name}/Pad", mode="constant")
                nd.input[0] = po
                at["pads"].ints[:] = [0, 0, 0, 0]
                dyn.add(po)
                binarized.add(po)
                new_nodes.append((pad, nm))
                n_pad += 1
        new_nodes.append((nd, nm))
    # ---- naming metadata: groups and their order
    first: dict = {}
    k = 0
    for nd, nm in new_nodes:
        if any(o in dyn for o in nd.output if o):
            if nm not in first:
                first[nm] = k
            k += 1
    attrs = dict(hook.get("group_attrs", {}))
    for nd, nm in new_nodes:
        mt = meta_of.get(id(nd), {})
        if mt.get("attrs") and any(o in dyn for o in nd.output if o):
            attrs[nm] = {**attrs.get(nm, {}), **mt["attrs"]}
    done_attrs: set = set()
    for nd, nm in new_nodes:
        del nd.metadata_props[:]
        if any(o in dyn for o in nd.output if o):
            mt = meta_of.get(id(nd), {})
            at = None
            if nm in attrs and nm not in done_attrs:       # once per group
                at = json.dumps(attrs[nm], sort_keys=True)
                done_attrs.add(nm)
            _set_props(nd, {"bitino.name": nm, "bitino.order": str(first[nm]), "bitino.attrs": at,
                            "bitino.extra_names": json.dumps(mt["extra_names"]) if mt.get("extra_names") else None,
                            "bitino.consts": json.dumps(mt["consts"], sort_keys=True) if mt.get("consts") else None,
                            "bitino.float_layer": "1" if mt.get("float_layer") and nd.op_type in ("Conv", "Gemm", "MatMul") else None})
    del g.node[:]
    g.node.extend([nd for nd, _ in new_nodes])
    used = {i for nd in g.node for i in nd.input}
    for t in [t for t in g.initializer if t.name not in used]:
        g.initializer.remove(t)
    missing_attr_groups = sorted(set(attrs) - set(first))
    if missing_attr_groups:
        raise AssertionError(f"hook attributes for unknown groups {missing_attr_groups[:5]}")
    # ---- container + model metadata
    m.ir_version = IR_VERSION
    del m.metadata_props[:]
    meta = dict(model_meta)
    meta["bitino.profile"] = PROFILE
    ln = hook.get("legacy_names")
    counter = hook.get("ir_unique_counter") or (getattr(ln, "counter", 0) if ln is not None else 0)
    if counter:
        meta["bitino.ir_unique_counter"] = str(counter)
    for kk, vv in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=kk, value=vv))
    m.producer_name, m.producer_version = "bitino tools/export_torch_qonnx.py", PROFILE
    onnx.save(m, onnx_path)        # onnx.checker runs in profile_check (E-ONNX-CHECKER)
    return {"nodes": len(g.node), "groups": len(first), "explicit_binary_pads": n_pad, "ir_version": IR_VERSION,
            "ir_unique_counter": meta.get("bitino.ir_unique_counter")}


def profile_check(path: str) -> dict:
    from bitino.qonnx_profile import check_file

    rep = check_file(path)
    d = rep.to_dict() if hasattr(rep, "to_dict") else {"passed": rep.passed}
    return d


def export(net, shape, path: str) -> None:
    import torch

    # no common-subexpression elimination: the exported graph keeps the module structure (e.g. the IR-Net shortcut
    # conv centers its input itself, the legacy graph has that mean twice; CSE would merge the two subgraphs)
    cse = torch._C._jit_pass_cse
    torch._C._jit_pass_cse = lambda graph: False
    try:
        with torch.no_grad():
            torch.onnx.export(net, torch.zeros(shape), path, opset_version=OPSET, do_constant_folding=False,
                              input_names=["x"], output_names=["logits"], dynamo=False,
                              custom_opsets={QONNX_DOMAIN: 1})
    finally:
        torch._C._jit_pass_cse = cse


def run_torch(net, x_nhwc: np.ndarray, dtype, batch: int = 16) -> np.ndarray:
    import torch

    m = net if dtype == torch.float32 else copy.deepcopy(net).to(dtype)   # float64 carriers stay float64 either way
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--repo", required=True, help="clone of the authors' repository (read only)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-dir", help="eval set (manifest.json): writes the source / exportable logits")
    ap.add_argument("--binarizer", choices=("source", "ge"), default="source",
                    help="source: keep the authors' binarizer (torch.sign -> ONNX Sign, sign(0) = 0); ge: export "
                         "BipolarQuant (0 -> +1), the deployable choice of the legacy converters (recorded)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=0, help="evaluate the source / exportable logits on this many samples at "
                                                         "a time (memory; 0 = the converters' batch of 16)")
    ap.add_argument("--no-source-logits", action="store_true",
                    help="skip the source / exportable logits (identity builds: tools/build_from_qonnx.py); --eval-dir "
                         "still gives the input shape and the preprocessing metadata")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    reg = json.load(open(args.registry))
    entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
    if entry is None:
        print(f"ERROR: {args.model_id} not in {args.registry}", file=sys.stderr)
        return 2
    ck_sha = sha256_file(args.weights)
    if ck_sha != entry["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch: expected {entry['checkpoint']['sha256']}, got {ck_sha}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {ck_sha}")
    os.makedirs(args.out_dir, exist_ok=True)

    import onnx
    import torch
    import torchvision

    torch.set_num_threads(max(1, args.threads))
    fam = family_of(args.model_id)
    model, prov = fam.load(args.model_id, args.repo, args.weights)
    mode = fam.source_binarizer if args.binarizer == "source" else "ge"
    pre_key = fam.preprocess(args.model_id)

    shape = None
    em = x = None
    if args.eval_dir:
        em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
        pre = em["preprocessing"][pre_key]
        x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
        shape = (1, pre["shape"][3], pre["shape"][1], pre["shape"][2])
    if shape is None:
        shape = (1, 3, 224, 224)
    fam.input_nchw = tuple(int(d) for d in shape)     # image-to-image hooks build shape-specific index maps
    image = getattr(fam, "output_kind", "logits") == "image"
    net, hook = fam.exportable(model, mode)
    out = os.path.join(args.out_dir, "model.onnx")
    export(net, shape, out)
    # source / exportable logits first, then free the torch objects (memory: the canonical rewrite and the profile check
    # of a 1.3 GB file, BNext-L, run after them)
    logit_rec: dict = {}
    if x is not None and not args.no_source_logits:
        if args.chunk:     # memory: the source / exportable models in float64 on a few samples at a time
            def chunked(dtype):
                parts = [fam.source_logits(model, x[i:i + args.chunk], dtype) for i in range(0, len(x), args.chunk)]
                z = {}
                for _, zz in parts:
                    for k, val in zz.items():
                        z[k] = z.get(k, 0) + val if isinstance(val, (int, float)) else val
                return np.concatenate([p[0] for p in parts]), z
            s32, z32 = chunked(torch.float32)
            s64, z64 = chunked(torch.float64)
        else:
            s32, z32 = fam.source_logits(model, x, torch.float32)
            s64, z64 = fam.source_logits(model, x, torch.float64)
        bt = 1 if (image or hook.get("batch1")) else (args.chunk or 16)   # shape-specific reshapes of a batch-1 graph
        e32, e64 = run_torch(net, x, torch.float32, bt), run_torch(net, x, torch.float64, bt)
        if image:      # images: the legacy reference files' NHWC layout (the source references are NHWC)
            e32, e64 = (np.ascontiguousarray(e.transpose(0, 2, 3, 1)) for e in (e32, e64))
        files = {"source_logits.f32": s32.astype(np.float32), "source_logits_f64.f64": s64.astype(np.float64),
                 "exportable_logits.f32": e32.astype(np.float32), "exportable_logits_f64.f64": e64.astype(np.float64)}
        for k, v in files.items():
            v.tofile(os.path.join(args.out_dir, k))
        labels = np.array([s["label"] for s in em["samples"]][: len(x)])
        logit_rec.update({"eval_set": em["name"], "n_samples": int(len(x)),
                    "logits_f32": os.path.abspath(os.path.join(args.out_dir, "source_logits.f32")),
                    "logits_f64": os.path.abspath(os.path.join(args.out_dir, "source_logits_f64.f64")),
                    "source_exact_zeros": {"f32": z32, "f64": z64},
                    "source_top1": None if image else float(np.mean(s32.argmax(1) == labels)),
                    "source_f32_vs_f64_top1_disagreements": None if image else int(np.sum(s32.argmax(1) != s64.argmax(1))),
                    "exportable_vs_source": {
                        "f64_max_abs": float(np.abs(e64 - s64).max()),
                        "f64_top1_disagreements": None if image else int(np.sum(e64.argmax(1) != s64.argmax(1))),
                        "f32_max_abs": float(np.abs(e32 - s32).max()),
                        "f32_top1_disagreements": None if image else int(np.sum(e32.argmax(1) != s32.argmax(1)))}})
        if image:
            logit_rec.update({"output_kind": "image", "output_layout": "NHWC (files); the ONNX output is NCHW",
                              "source_f32_vs_f64_max_abs": float(np.abs(s32.astype(np.float64) - s64).max())})
        print(f"source top-1 {logit_rec['source_top1']}; exportable vs source: {logit_rec['exportable_vs_source']}")
    # profile metadata (docs/qonnx-profile.md § 6)
    commit = _git_commit()
    lic = (entry.get("license") or {}).get("weights") or "unknown"
    pre_def = em["preprocessing"][pre_key]["definition"] if em else ""
    exporter = (f"tools/export_torch_qonnx.py@{commit or 'unknown'} hook '{fam.name}'; torch {torch.__version__} "
                f"torch.onnx.export (TorchScript, opset {OPSET}); onnx {onnx.__version__}")
    meta = {"bitino.model_id": args.model_id, "bitino.variant": "float", "bitino.layout": "NCHW",
            "bitino.input_domain": fam.input_domain(args.model_id),
            "bitino.preprocessing": json.dumps({"key": pre_key, "definition": pre_def}),
            "bitino.labels": fam.labels(args.model_id, em),
            "bitino.source.framework": "pytorch", "bitino.source.framework_version": torch.__version__,
            "bitino.source.checkpoint_sha256": ck_sha, "bitino.weights_license": lic, "bitino.exporter": exporter,
            "bitino.notes": f"binarizer {mode} ({'ONNX Sign, sign(0) = 0' if mode == 'sign_zero' else 'BipolarQuant, 0 -> +1'}); "
                            f"published file {'model_sign_zero.bitino' if mode == 'sign_zero' else 'model.bitino'}"}
    if image and hasattr(fam, "task"):
        meta["bitino.task"] = fam.task(em)
    meta["bitino.import"] = import_meta(hook["import_args"])     # step 6: the file determines its import
    if hook.get("const_order"):                       # profile § 7: data-section order of the importer's constants
        meta["bitino.const_order"] = json.dumps(hook["const_order"])
    if prov.get("repo") and prov.get("repo_commit") and re.fullmatch(r"[0-9a-f]{7,40}", str(prov["repo_commit"])):
        meta["bitino.source.code"] = f"{prov['repo'].split(' ')[0]}@{prov['repo_commit']}"
    names = canonicalize(out, net, hook, meta)
    del model, net
    import gc
    gc.collect()
    prof = profile_check(out)
    if not prof.get("passed"):
        print(json.dumps(prof, indent=1)[:4000], file=sys.stderr)
        raise SystemExit("ERROR: the exported file does not pass check-qonnx (profile bitino-qonnx/1)")
    m = onnx.load(out)
    rec = {"model_id": args.model_id, "family": fam.name, "legacy_converter": fam.converter,
           "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": ck_sha, **prov,
           "framework": f"torch {torch.__version__}, torchvision {torchvision.__version__}, onnx {onnx.__version__}, "
                        f"numpy {np.__version__}, python {platform.python_version()}",
           "exporter": f"torch {torch.__version__} torch.onnx.export (TorchScript, opset {OPSET}) + tools/export_torch_qonnx.py "
                       f"hook '{fam.name}'",
           "export_options": {"opset_version": OPSET, "do_constant_folding": False, "dynamo": False, "input": f"zeros{list(shape)}",
                              "input_names": ["x"], "output_names": ["logits"]},
           "binarizer": {"option": args.binarizer, "source": fam.source_binarizer, "exported": mode,
                         "onnx_op": "Sign" if mode == "sign_zero" else "BipolarQuant",
                         "note": ("the authors' binarizer (exact)" if mode == fam.source_binarizer else
                                  "deployable choice (0 -> +1) of the legacy converter; differs from the source only for "
                                  "activations exactly 0 at a binarization point (counted in source_exact_zeros)")},
           "hook": hook["replacements"], "import_args": hook["import_args"], "canonical": names,
           "profile_check": {k: prof.get(k) for k in ("profile", "passed", "summary")} | {
               "warnings": [f for f in prof.get("findings", []) if f.get("level") == "warning"]},
           "onnx_file": "model.onnx", "onnx_sha256": sha256_file(out), "onnx_producer": f"{m.producer_name} {m.producer_version}",
           "opsets": [[o.domain or "ai.onnx", int(o.version)] for o in m.opset_import], "eval_preprocessing": pre_key,
           "venv": ".cache/venv-qonnx2 (python -m venv, no system site packages; freeze in .cache/venv-qonnx2/freeze.txt)"}
    rec.update(logit_rec)
    json.dump(rec, open(os.path.join(args.out_dir, "export.json"), "w"), indent=1)
    print(f"exported {args.model_id} -> {out} (sha256 {rec['onnx_sha256'][:16]}), {len(m.graph.node)} nodes, binarizer {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
