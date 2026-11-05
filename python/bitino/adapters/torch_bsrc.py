"""Adapter: BSRC-2023 binary super-resolution baselines (ilyazhara/BSRC-2023 @62bbcf2, branch dev; challenge
baselines, no paper) -> Bitino IR. Image-to-image: input LR RGB in [0, 255], output SR RGB in [0, 255] (NHWC).

Checkpoints (bnn_competition/checkpoints/scalex{2,4}_baseline_v{0,1}.ckpt) are ``torch.save(model)`` pickles of
the whole nn.Module. ``load_checkpoint`` reads them with ``torch.load(weights_only=True)``: every non-tensor
global of the pickle (bnn_competition.* classes and torch.nn module classes) is allow-listed as an *inert stub*
class carrying the pickled module name, whose instances only store their pickled ``__dict__`` (no source code
of the pickle is executed, no real torch.nn class is instantiated from it). The tensors and the pickled
hyper-parameters are then read from the stub tree; the official model class (models/bnn/*.py, executed from
the clone) is built with the YAML arguments (examples/bnn/*.yaml), checked attribute by attribute against the
stub tree, and loaded with load_state_dict(strict=True).

Semantics (models/bnn/binary_scalex*_baseline_v*.py, libs/*):
  * BinaryConv2d (libs/binary_modules/binary_conv2d.py:26-38): ``F.conv2d(input_binarizer(x),
    weight_binarizer(W), bias, padding=1)``. Input binarizer (libs/binarizers/binarizer.py) with
    ParabolaSignApproximator (v0) or STESignApproximator (v1): forward ``sign(sign(x) + 0.1)`` -> +1 iff
    x >= 0 (``ge``, sign(0) = +1: the exact source semantics, no sign_zero variant needed). The STE terms
    ``(s - b).detach() + b`` return s exactly when |b| <= 1 and b has the sign of s or is 0 (Parabola: b in
    [-1, 1]; STE in v1: b = the LAB output, exactly +-1), checked with forward hooks. Weight binarizer:
    STEWithClipSignApproximator (exact by the same argument, |clamp(W)| <= 1) times L1Scaling(dim=[1,2,3]):
    ``sign(sign(W)+0.1) * mean|W|`` per output channel -> sign bits (W >= 0 -> +1), post_mul = mean|W|
    (float64, rounded once), post_add = conv bias. Zero padding of the binarized input -> pad_value 0.
  * Scale (``variance * x + mean``) -> ChannelAffine(scale, bias) (same two fp32 operations).
  * MeanShift (``x + mean_shift``, buffer) -> ChannelAffine(bias). Normalize (x4 v0): ``(x - 127.5) / 51`` ->
    ChannelAffine(bias -127.5) + ChannelAffine(scale fl(1/51)) (<= 1 ulp from torch's division);
    inverse ``x * 51 + 127.5`` -> ChannelAffine(51, 127.5) (identical operations).
  * LAB (v1, ``bin_path[0]``): depthwise Conv2d(C -> 2C, groups=C, bias) reshaped (n, C, 2, h, w); forward value
    ``argmax(dim=2) * 2 - 1`` (STE with the softmax; exact as above) = +1 iff x1 > x0 (ties -> index 0 -> -1).
    Mapped to Conv2D(groups=C) followed by a 1x1 Conv2D(groups=C) with weights (-1, +1) (d = x1 - x0, computed
    from the two rounded conv outputs: the sign of a two-term float sum is exact and d = 0 iff x1 = x0) feeding
    the BConv2D with input_binarize ``gt`` (+1 iff d > 0).
  * ConvBlock v0: ``conv_path(x) + x`` (BinaryConv2d, Scale, PReLU). ConvBlock v1 x2: ``act(bin_path(x) + p1 +
    dw(p1)) + x`` with p1 = fp_1x1_conv(x) (no bias) and dw = grouped 3x3 conv (groups = C/16, no bias);
    v1 x4: ``act(bin_path(x) + dw(fp_1x1_conv(x))) + x`` (groups = C, depthwise).
  * ResidualBlockFP (x2 v1, first/last n_blocks_skip blocks): ``x + conv(relu(conv(x)))``.
  * BatchNorm2d (eval, running stats) -> ChannelAffine folded in float64. PixelShuffle -> Reshape/Transpose.
  * Upsampler / upsampling: Conv2d(C -> 4C) + PixelShuffle(2) (+ PReLU(1) in x4), x log2(scale).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ..ir import Graph
from . import restoration_common as rc
from .torch_rbonn import _cls, _expect, _f64, _pair, _to_ohwi

MODELS = {
    "bsrc2023-binary-sr-x2-v0": {"arch": "x2v0", "file": "binary_scalex2_baseline_v0.py", "cls": "BinaryScalex2BaselineV0",
                                 "kwargs": {}, "scale": 2, "preprocess": "sr_x2_255", "yaml": "examples/bnn/scalex2_baseline-0.yaml"},
    "bsrc2023-binary-sr-x2-v1": {"arch": "x2v1", "file": "binary_scalex2_baseline_v1.py", "cls": "BinaryScalex2BaselineV1",
                                 "kwargs": {"group_size": 16, "n_blocks_skip": 2}, "scale": 2, "preprocess": "sr_x2_255",
                                 "yaml": "examples/bnn/scalex2_baseline-1.yaml"},
    "bsrc2023-binary-sr-x4-v0": {"arch": "x4v0", "file": "binary_scalex4_baseline_v0.py", "cls": "BinaryScalex4BaselineV0",
                                 "kwargs": {}, "scale": 4, "preprocess": "sr_x4_255", "yaml": "examples/bnn/scalex4_baseline-0.yaml"},
    "bsrc2023-binary-sr-x4-v1": {"arch": "x4v1", "file": "binary_scalex4_baseline_v1.py", "cls": "BinaryScalex4BaselineV1",
                                 "kwargs": {"group_size": 1}, "scale": 4, "preprocess": "sr_x4_255", "yaml": "examples/bnn/scalex4_baseline-1.yaml"},
}
_TORCH_SAFE = {"collections.OrderedDict", "torch.FloatStorage", "torch.LongStorage", "torch._utils._rebuild_parameter", "torch._utils._rebuild_tensor_v2"}


# ------------------------------------------------------------------ checkpoint (whole-module pickle)
class _Stub:
    """Inert stand-in for a pickled class: keeps the pickled state, executes nothing."""

    def __setstate__(self, state):
        self.__dict__["_bf_state"] = state

    def __repr__(self):
        return f"<stub {type(self).__module__}.{type(self).__qualname__}>"


def pickle_globals(path: str) -> list[tuple[str, str]]:
    """(module, name) of every GLOBAL / STACK_GLOBAL in the checkpoint's data.pkl (static scan, nothing executed)."""
    import pickletools
    import zipfile

    z = zipfile.ZipFile(path)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")]
    assert len(pk) == 1, pk
    ops = list(pickletools.genops(z.read(pk[0])))
    out, memo, strings = set(), {}, []
    for op, arg, _ in ops:
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        if op.name == "GLOBAL":
            mod, name = arg.split(" ", 1)
            out.add((mod, name))
        elif op.name == "STACK_GLOBAL":
            out.add((strings[-2], strings[-1]))
        elif op.name in ("BINPUT", "LONG_BINPUT", "MEMOIZE") and strings:
            memo[len(memo)] = strings[-1]
        elif op.name in ("BINGET", "LONG_BINGET"):
            if arg in memo:
                strings.append(memo[arg])
    return sorted(out)


def load_checkpoint(path: str):
    """torch.load(weights_only=True) with inert stubs allow-listed for the pickle's class globals.
    Returns the root stub (its tree mirrors the pickled nn.Module)."""
    import torch

    stubs = []
    for mod, name in pickle_globals(path):
        full = f"{mod}.{name}"
        if full in _TORCH_SAFE or full == "__builtin__.set" or full == "builtins.set":
            continue
        _expect(mod.startswith("bnn_competition.") or mod.startswith("torch.nn.modules."), f"unexpected pickle global {full}")
        cls = type(name, (_Stub,), {"__module__": mod, "__qualname__": name})
        stubs.append((cls, full))
    with torch.serialization.safe_globals(stubs + [set]):
        root = torch.load(path, map_location="cpu", weights_only=True)
    _expect(isinstance(root, _Stub), f"checkpoint root is {type(root)}, not a pickled module")
    return root


def stub_tree(root) -> dict[str, tuple[str, dict]]:
    """module path -> (class name, pickled __dict__) for the stub tree."""
    out = {}

    def walk(s, p):
        st = s._bf_state
        out[p] = (type(s).__qualname__, st)
        for k, ch in st.get("_modules", {}).items():
            if ch is not None:
                walk(ch, f"{p}.{k}" if p else k)
    walk(root, "")
    return out


def stub_state_dict(root) -> dict:
    import torch

    sd = {}
    for p, (_, st) in stub_tree(root).items():
        nonpersist = st.get("_non_persistent_buffers_set", set())
        for kind in ("_parameters", "_buffers"):
            for k, v in st.get(kind, {}).items():
                if v is None or (kind == "_buffers" and k in nonpersist):
                    continue
                _expect(isinstance(v, torch.Tensor), f"{p}.{k}: not a tensor")
                sd[f"{p}.{k}" if p else k] = v.detach().clone()
    return sd


def _simple(v):
    return isinstance(v, (bool, int, float, str)) or (isinstance(v, (tuple, list)) and all(isinstance(e, (bool, int, float, str)) for e in v))


def load_model(repo: str, model_id: str, weights: str | None):
    """Official class from the clone (bnn_competition/__init__.py, which imports nip/torchmetrics, is replaced by a
    namespace stub; everything below it is executed from the files) + the checkpoint's tensors."""
    import hashlib
    import importlib
    import types

    import torch

    spec = MODELS[model_id]
    root = os.path.join(os.path.abspath(repo), "bnn_competition")
    sys.dont_write_bytecode = True
    saved = {k: v for k, v in sys.modules.items() if k == "bnn_competition" or k.startswith("bnn_competition.")}
    pkg = types.ModuleType("bnn_competition")
    pkg.__path__ = [root]
    sys.modules["bnn_competition"] = pkg
    try:
        mod = importlib.import_module(f"bnn_competition.models.bnn.{spec['file'][:-3]}")
        model = getattr(mod, spec["cls"])(**spec["kwargs"])
    finally:
        for k in [k for k in sys.modules if k == "bnn_competition" or k.startswith("bnn_competition.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    files = [f"models/bnn/{spec['file']}", "libs/binarizers/binarizer.py", "libs/binary_modules/binary_conv2d.py",
             "libs/sign_approximators/parabola_sign_approximator.py", "libs/sign_approximators/ste_sign_approximator.py",
             "libs/sign_approximators/ste_with_clip_sign_approximator.py", "libs/transformations/l1_scaling.py"]
    src = {f"bnn_competition/{f}": hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest() for f in files}
    cls = f"bnn_competition/models/bnn/{spec['file']} {spec['cls']}({', '.join(f'{k}={v!r}' for k, v in spec['kwargs'].items())})"
    if weights is None:
        return model.eval(), cls, src
    rootstub = load_checkpoint(weights)
    tree = stub_tree(rootstub)
    mods = dict(model.named_modules())
    _expect(set(tree) == set(mods), f"module trees differ: {sorted(set(tree) ^ set(mods))[:10]}")
    checked = 0
    for p, (cname, st) in tree.items():
        m = mods[p]
        _expect(cname == type(m).__name__, f"{p}: pickled {cname} vs built {type(m).__name__}")
        for k, v in st.items():
            if k.startswith("_") or not _simple(v) or k == "training":
                continue
            _expect(hasattr(m, k), f"{p}.{k}: pickled attribute missing on the built module")
            mv = getattr(m, k)
            mv = list(mv) if isinstance(mv, tuple) else mv
            vv = list(v) if isinstance(v, tuple) else v
            _expect(mv == vv, f"{p}.{k}: pickled {v!r} vs built {getattr(m, k)!r}")
            checked += 1
    sd = stub_state_dict(rootstub)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model._bitino_load_notes = {"checkpoint": "whole-module pickle read with torch.load(weights_only=True) + inert stub classes",
                                  "pickled_hyperparameters_checked": checked, "tensors_loaded": len(sd),
                                  "pickled_training_flag": bool(tree[""][1].get("training", False))}
    return model, cls, src


# ------------------------------------------------------------------ IR
def convert(model, model_id: str, input_hw: tuple[int, int] = (48, 48), binarize: str = "ge") -> tuple[Graph, dict]:
    _expect(binarize == "ge", "BSRC binarizes with sign(sign(x)+0.1) (0 -> +1): only 'ge' is the source semantics")
    spec = MODELS[model_id]
    arch = spec["arch"]
    _expect(_cls(model) == spec["cls"], f"unsupported top-level class {_cls(model)} (expected {spec['cls']})")
    _expect(not model.training, "model must be in eval mode")
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    binary_weights: dict[str, np.ndarray] = {}
    visited: set[str] = set()
    unused: list[str] = []
    h, w = input_hw

    def vec(t) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def folded(name, v64):
        exact[name] = np.asarray(v64, np.float64)
        return g.add_constant(name, np.asarray(v64).astype(np.float32))

    def affine(x, name, scale=None, bias=None, c=None):
        s = g.add_constant(f"{name}/scale", np.broadcast_to(np.asarray(scale, np.float32), (c,)).copy()) if scale is not None else ""
        b = g.add_constant(f"{name}/bias", np.broadcast_to(np.asarray(bias, np.float32), (c,)).copy()) if bias is not None else ""
        return g.add_node("ChannelAffine", [x, s, b], name=name)

    def conv(m, x, name, act="none"):
        _expect(_cls(m) == "Conv2d" and _pair(m.dilation) == (1, 1) and _pair(m.stride) == (1, 1) and m.padding_mode == "zeros",
                f"{name}: unexpected {_cls(m)}")
        kh, kw = _pair(m.kernel_size)
        ph, pw = _pair(m.padding)
        _expect((ph, pw) == (kh // 2, kw // 2), f"{name}: not a 'same' conv")
        visited.add(name)
        b = g.add_constant(f"{name}/bias", vec(m.bias)) if m.bias is not None else ""
        precision.append({"layer": name, "op": "Conv2D", "source": f"Conv2d {kh}x{kw} groups={m.groups}", "weights": "fp32", "activations": "fp32"})
        return g.add_node("Conv2D", [x, g.add_constant(f"{name}/kernel", _to_ohwi(m.weight.detach().cpu().numpy().astype(np.float32)), layout="OHWI"), b],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [ph, ph, pw, pw], "groups": int(m.groups), "pad_value": 0.0,
                           "activation": act}, name=name)

    def prelu(m, x, name, c):
        _expect(_cls(m) == "PReLU", f"{name}: unexpected {_cls(m)}")
        sl = vec(m.weight)
        _expect(sl.size in (1, c), f"{name}: {sl.size} slopes for {c} channels")
        visited.add(name)
        return g.add_node("Activation", [x, g.add_constant(f"{name}/slope", np.broadcast_to(sl, (c,)).copy())], {"type": "prelu"}, name=name)

    def scale_mod(m, x, name, c):
        _expect(_cls(m) == "Scale", f"{name}: unexpected {_cls(m)}")
        visited.add(name)
        return affine(x, name, vec(m.variance), vec(m.mean), c)

    def bn(m, x, name):
        _expect(_cls(m) == "BatchNorm2d" and m.affine and m.running_mean is not None, f"{name}: expected BatchNorm2d with running stats")
        sc = _f64(m.weight) / np.sqrt(_f64(m.running_var) + float(m.eps))
        sh = _f64(m.bias) - _f64(m.running_mean) * sc
        visited.add(name)
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", sc), folded(f"{name}/shift", sh)], name=name)

    def bconv(m, x, name, mode):
        import torch
        _expect(_cls(m) == "BinaryConv2d" and m.groups == 1 and _pair(m.stride) == (1, 1) and _pair(m.padding) == (1, 1)
                and _pair(m.dilation) == (1, 1) and _pair(m.kernel_size) == (3, 3) and m.padding_mode == "zeros", f"{name}: unexpected conv")
        ib, wb = m.input_binarizer, m.weight_binarizer
        _expect(_cls(ib) == "Binarizer" and ib.transformation is None
                and _cls(ib.sign_approximator) in ("ParabolaSignApproximator", "STESignApproximator"), f"{name}.input_binarizer: unexpected")
        _expect(_cls(wb) == "Binarizer" and _cls(wb.sign_approximator) == "STEWithClipSignApproximator"
                and _cls(wb.transformation) == "L1Scaling" and list(wb.transformation.dim) == [1, 2, 3], f"{name}.weight_binarizer: unexpected")
        visited.add(name)
        wt = m.weight.detach().cpu()
        bits = torch.where(wt >= 0, 1, -1).numpy().astype(np.int8)       # sign(sign(W) + 0.1): 0 -> +1
        s64 = wt.double().abs().mean(dim=[1, 2, 3]).numpy()
        binary_weights[name] = bits
        diag.append({"layer": name, "latent_weight_zeros": int((wt == 0).sum()), "n_weights": int(wt.numel()),
                     "latent_abs_gt_1": int((wt.abs() > 1).sum())})
        pa = g.add_constant(f"{name}/post_add", vec(m.bias)) if m.bias is not None else ""
        t = g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", _to_ohwi(bits), dtype="bit", layout="OHWI_bit"),
                                   folded(f"{name}/post_mul", s64), pa, ""],
                       {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
                        "input_binarize": mode, "activation": "none", "output": "f32"}, name=name)
        precision.append({"layer": name, "op": "BConv2D", "source": "BinaryConv2d (input/weight Binarizer)",
                          "weights": "binary sign(W) (0 -> +1), per-channel mean|W| (L1Scaling)" + (" + fp bias" if m.bias is not None else ""),
                          "activations": "binary " + ("(sign(x), 0 -> +1)" if mode == "ge" else "(LAB: +1 iff x1 > x0)"),
                          "input_binarize": mode, "pad_value": 0.0, "kernel": [3, 3]})
        return t

    def lab(m, x, name, c):
        _expect(_cls(m) == "LAB" and float(m.temperature) > 0 and _cls(m.softmax) == "Softmax" and int(m.softmax.dim) == 2, f"{name}: unexpected LAB")
        dw = m.fp_depthwise_conv
        _expect(dw.groups == c and int(dw.weight.shape[0]) == 2 * c, f"{name}.fp_depthwise_conv: unexpected")
        visited.update({name, f"{name}.softmax"})
        t = conv(dw, x, f"{name}.fp_depthwise_conv")
        k = np.zeros((c, 1, 1, 2), np.float32)
        k[:, 0, 0, 0], k[:, 0, 0, 1] = -1.0, 1.0
        precision.append({"layer": f"{name}.diff", "op": "Conv2D", "source": "LAB argmax over (x0, x1): d = x1 - x0 (grouped 1x1, weights -1/+1)",
                          "weights": "exact", "activations": "fp32 (sign exact)"})
        return g.add_node("Conv2D", [t, g.add_constant(f"{name}.diff/kernel", k, layout="OHWI"), ""],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": c, "pad_value": 0.0,
                           "activation": "none"}, name=f"{name}.diff")

    def conv_block(m, x, name, c):
        _expect(_cls(m) == "ConvBlock", f"{name}: unexpected {_cls(m)}")
        visited.add(name)
        if arch.endswith("v0"):
            cp = m.conv_path
            _expect(_cls(cp) == "Sequential" and len(cp) == 3, f"{name}.conv_path: unexpected")
            t = bconv(cp[0], x, f"{name}.conv_path.0", "ge")
            t = scale_mod(cp[1], t, f"{name}.conv_path.1", c)
            t = prelu(cp[2], t, f"{name}.conv_path.2", c)
            return g.add_node("Add", [t, x], name=f"{name}.add")
        bp = m.bin_path
        _expect(_cls(bp) == "Sequential" and len(bp) == 3, f"{name}.bin_path: unexpected")
        d = lab(bp[0], x, f"{name}.bin_path.0", c)
        p0 = bconv(bp[1], d, f"{name}.bin_path.1", "gt")
        p0 = scale_mod(bp[2], p0, f"{name}.bin_path.2", c)
        _expect(m.fp_1x1_conv.bias is None and m.fp_depthwise_conv.bias is None, f"{name}: unexpected conv bias")
        p1 = conv(m.fp_1x1_conv, x, f"{name}.fp_1x1_conv")
        p2 = conv(m.fp_depthwise_conv, p1, f"{name}.fp_depthwise_conv")
        if arch == "x2v1":   # path_0 + path_1 + path_2
            s = g.add_node("Add", [g.add_node("Add", [p0, p1], name=f"{name}.add01"), p2], name=f"{name}.add012")
        else:                # x4 v1: path_0 + dw(1x1(x))
            s = g.add_node("Add", [p0, p2], name=f"{name}.add01")
        t = prelu(m.act, s, f"{name}.act", c)
        return g.add_node("Add", [t, x], name=f"{name}.add")

    def residual_block(m, x, name, c):
        if _cls(m) == "ResidualBlockFP":
            cb = m.conv_block
            _expect(_cls(cb) == "Sequential" and len(cb) == 3 and _cls(cb[1]) == "ReLU", f"{name}.conv_block: unexpected")
            visited.update({name, f"{name}.conv_block.1"})
            t = conv(cb[0], x, f"{name}.conv_block.0", act="relu")
            t = conv(cb[2], t, f"{name}.conv_block.2")
            return g.add_node("Add", [x, t], name=f"{name}.add")
        _expect(_cls(m) == "ResidualBlock" and len(m.res_block) == 2, f"{name}: unexpected {_cls(m)}")
        visited.add(name)
        t = conv_block(m.res_block[0], x, f"{name}.res_block.0", c)
        return conv_block(m.res_block[1], t, f"{name}.res_block.1", c)

    def upsample(seq, x, name, c, ch, cw, prelus: bool):
        mods = list(seq.named_children())
        step = 3 if prelus else 2
        _expect(len(mods) % step == 0, f"{name}: unexpected length")
        for i in range(0, len(mods), step):
            (cn, cm), (pn, pm) = mods[i], mods[i + 1]
            _expect(_cls(pm) == "PixelShuffle" and int(pm.upscale_factor) == 2, f"{name}.{pn}: unexpected")
            visited.add(f"{name}.{pn}")
            t = conv(cm, x, f"{name}.{cn}")
            x = rc.pixel_shuffle(g, t, ch, cw, int(cm.weight.shape[0]), 2, f"{name}.{pn}")
            ch, cw = ch * 2, cw * 2
            if prelus:
                x = prelu(mods[i + 2][1], x, f"{name}.{mods[i + 2][0]}", c)
        return x, ch, cw

    x = g.add_input("input", [1, h, w, 3])
    if arch.startswith("x2"):
        for nm in ("sub_mean", "add_mean"):
            _expect(_cls(getattr(model, nm)) == "MeanShift", f"{nm}: unexpected")
        visited.update({"sub_mean", "add_mean", "head", "body", "tail", "tail.0"})
        t = affine(x, "sub_mean", bias=vec(model.sub_mean.mean_shift), c=3)
        _expect(len(model.head) == 1, "head: unexpected")
        t = conv(model.head[0], t, "head.0")
        c = int(model.head[0].weight.shape[0])
        body = list(model.body.named_children())
        b = t
        for nm, m in body[:-1]:
            b = residual_block(m, b, f"body.{nm}", c)
        b = conv(body[-1][1], b, f"body.{body[-1][0]}")
        t = g.add_node("Add", [t, b], name="body_residual")
        _expect(_cls(model.tail[0]) == "Upsampler" and len(model.tail) == 2, "tail: unexpected")
        t, oh, ow = upsample(model.tail[0], t, "tail.0", c, h, w, prelus=False)
        t = conv(model.tail[1], t, "tail.1")
        t = affine(t, "add_mean", bias=vec(model.add_mean.mean_shift), c=3)
    else:
        visited.update({"conv1", "res_blocks", "conv2", "upsampling", "conv3"})
        if arch == "x4v0":
            tr, it = model.transform, model.inverse_transform
            _expect(_cls(tr) == "Normalize" and tr.sign == -1 and _cls(it) == "Normalize" and it.sign == 1, "Normalize: unexpected")
            visited.update({"transform", "inverse_transform"})
            t = affine(x, "transform/sub", bias=-float(tr.mean_shift), c=3)
            t = g.add_node("ChannelAffine", [t, folded("transform/div/scale", np.full(3, 1.0 / float(tr.scale_factor))), ""], name="transform/div")
        else:
            unused.append("transform / inverse_transform (Normalize modules constructed but not called in forward)")
            visited.update({"transform", "inverse_transform"})
            t = x
        _expect(len(model.conv1) == 2, "conv1: unexpected")
        c = int(model.conv1[0].weight.shape[0])
        out1 = prelu(model.conv1[1], conv(model.conv1[0], t, "conv1.0"), "conv1.1", c)
        r = out1
        for nm, m in model.res_blocks.named_children():
            r = residual_block(m, r, f"res_blocks.{nm}", c)
        _expect(len(model.conv2) == 2, "conv2: unexpected")
        out2 = bn(model.conv2[1], conv(model.conv2[0], r, "conv2.0"), "conv2.1")
        t = g.add_node("Add", [out1, out2], name="add_out1_out2")
        t, oh, ow = upsample(model.upsampling, t, "upsampling", c, h, w, prelus=True)
        _expect(len(model.conv3) == 1, "conv3: unexpected")
        t = conv(model.conv3[0], t, "conv3.0")
        if arch == "x4v0":
            t = affine(t, "inverse_transform", scale=float(it.scale_factor), bias=float(it.mean_shift), c=3)
    g.set_outputs(t)

    known = {spec["cls"], "Sequential", "Conv2d", "ReLU", "PReLU", "PixelShuffle", "BatchNorm2d", "BinaryConv2d", "Binarizer",
             "ParabolaSignApproximator", "STESignApproximator", "STEWithClipSignApproximator", "L1Scaling", "ConvBlock",
             "ResidualBlock", "ResidualBlockFP", "Scale", "MeanShift", "Upsampler", "LAB", "Softmax", "Normalize"}
    for n_, m in model.named_modules(remove_duplicate=False):
        _expect(_cls(m) in known, f"module {n_ or '<root>'}: unsupported type {_cls(m)}")
    for n_, _ in list(model.named_parameters()) + list(model.named_buffers()):
        ok = any(n_ == v or n_.startswith(v + ".") for v in visited)
        _expect(ok, f"parameter/buffer {n_} not covered by the adapter")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": unused, "float64_constants": exact, "binary_weights": binary_weights,
            "output_shape": [1, oh, ow, 3]}
    return g, info
