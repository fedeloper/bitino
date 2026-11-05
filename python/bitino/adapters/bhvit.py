"""BHViT (IMRL/BHViT, CVPR 2025) -> Bitino IR.

Reference implementation: official repo IMRL/BHViT (transformer/BHViT.py, transformer/utils_quant.py,
models.py), executed read-only. Every submodule is mapped explicitly; a module type or config flag
outside the verified set raises. Every parameter and buffer of the checkpoint must be consumed by the
mapping (checked at the end of convert()), so a silently dropped layer is impossible.

Semantics (file:line refer to the official repo at the commit recorded by tools/convert_bhvit.py):
  * BatchNormT (models.py:9-57) in eval mode = per-channel affine over the last (channel) dim
    -> ChannelAffine, scale = w/sqrt(var+eps), shift = b - mean*scale (fp64 fold, stored f32).
  * QuantizeLinear (utils_quant.py:214-248), W1A1, recu=False, no bias:
      out = linear(sign(x), mean(|W|, dim=1) * sign(W - mean(W, dim=-1)))
    Its input is always (h + move) (BHViT.py:192-194, 260, 303, 325) -> BDense with threshold = -move
    and input_binarize 'ge': sign(fl(h + move)) == sign(h - (-move)) exactly in IEEE arithmetic except
    for the exact tie h == -move, where torch.sign gives 0 (counted by the converter).
    binarize='sign_zero' gives the exact torch.sign semantics (reference kernels only).
  * QuantizeConv2d (utils_quant.py:338-373): scale = mean|W| over [1,2,3], sign(W - mean_[1,2,3] W),
    F.conv2d(sign(x)) with zero padding of the binarized input -> BConv2D pad_value 0, groups and
    dilation as in the module (token mixer: g=4, dilation 1/3/5, padding 'same'; downsample: 2x2 s2).
  * Binary weight signs and scales are computed in float64 from the fp32 latent weights. In fp64 no
    centred weight is exactly 0 (checked: raises otherwise); the fp32 torch expression rounds a few to
    exactly 0 (sign 0) on CPU; those differences are reported in info["weight_diagnostics"].
  * Q/K/V (BHViT.py:192-209): sign(RPReLU(BN(binlinear(h + move)) + h) + move2), torch.sign (sign(0)=0).
    Q.K^T -> BMatMul(sign_zero x sign_zero, scale 1/sqrt(dh)) (exact, dh a power of 4).
    V -> Activation 'sign_zero' (float {-1,0,+1}) for the 2-bit x ternary AV product (MatMul).
  * Softmax over keys (fp), then BinaryActivation_Attention(num_heads, 3) applied to P*3
    (BHViT.py:159, 222; utils_quant.py:540-571), Qn=0, Qp=3:
      q = clamp(round((3P)/alpha_h + zp_h), 0, 3) * alpha_h     -> op FakeQuant (axis 1 = heads)
    alpha_h/zp_h are the values grad_scale() returns in the forward pass, (a - a*g) + a*g with
    g = heads/sqrt(numel*Qp), numel at batch size 1, evaluated with the fp32 torch expression
    (identical to the parameter for both checkpoints; recorded in info).
  * Token_for_Attention (BHViT.py:87-113): 7x7 windows (Reshape/Transpose), merge tokens
    a1*MaxPool(7,7,p3) + (1-a1)*AvgPool(7,7,p3,count_include_pad) repeated per window (Concat axis 0),
    token Concat (axis 1), real nn.LayerNorm -> LayerNorm.
  * token_split (BHViT.py:172-185): window tokens -> window_reverse, merge tokens -> mean over windows
    + nearest upsampling; both are index selections, mapped to Gather with constant row indices that are
    traced through the OFFICIAL functions (window_reverse, F.interpolate) on index tensors.
  * Shift / Shift2 / Shift_channel_mix (BHViT.py:343-379): torch.roll -> Gather with constant row
    indices traced through torch.roll / the official Shift_channel_mix module; sum -> Add; the 1/3, 1/5
    factors are folded into the following LayerScale alpha (fp64 product, stored f32).
  * ViTIntermediate shortcut concat([h]*ratio) -> Concat; ViTOutput AvgPool1d(ratio) over channels ->
    Reshape [..., C, ratio] + ReduceMean(axis -1).
  * Stem: fp Conv2d k4 s4 + BN + GELU (Activation gelu, evaluated in double) + position embedding.
  * Head: final BN, mean over tokens (GlobalAvgPool), fp Linear -> Dense.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from ..ir import Graph

# Checkpoint args the adapter has been verified for (some_fp is handled: BinaryPatchEmbed or PatchEmbed).
EXPECTED_ARGS = {"model_type": "dbhvit", "weight_bits": 1, "input_bits": 1, "replace_ln_bn": True, "shift3": True,
                 "shift5": True, "gsb": False, "recu": False, "disable_layerscale": False, "enable_cls_token": False,
                 "nb_classes": 1000, "input_size": 224}
CONFIGS = {"bhvit-tiny-binary": "BHViT_tiny", "bhvit-tiny-somefp": "BHViT_tiny",
           "bhvit-small-binary": "BHViT", "bhvit-small-somefp": "BHViT"}


# --------------------------------------------------------------------------------------------
# Reference model (official code, imported read-only from a local clone)
# --------------------------------------------------------------------------------------------

def load_checkpoint(path: str):
    import torch
    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(path, map_location="cpu", weights_only=True)


def build_reference(repo_dir: str, model_id: str, ckpt: dict):
    """Instantiates BHViTForImageClassification exactly like main_new.py (models.get_model) and loads the
    checkpoint's 'model' state dict with strict=True. Returns (model, args_dict)."""
    import torch
    a = vars(ckpt["args"])
    for k, v in EXPECTED_ARGS.items():
        if a.get(k) != v:
            raise ValueError(f"checkpoint arg {k}={a.get(k)!r}, adapter verified only for {v!r}")
    if a.get("model_ema"):
        raise ValueError("checkpoint was trained with model EMA; adapter verified for model_ema=False")
    sys.dont_write_bytecode = True  # never write __pycache__ into the source clone
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import models as bhvit_models  # official models.py (get_model, BatchNormT)
    # args.model is the author's absolute path ('.../configs/BHVIT_tiny'); the repo's config dir is
    # spelled 'BHViT_tiny' / 'BHViT'.
    model = bhvit_models.get_model(ckpt["args"], os.path.join(repo_dir, "configs", CONFIGS[model_id]), a["model_type"],
                                   a["weight_bits"], a["input_bits"])
    res = model.load_state_dict(ckpt["model"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    model.eval()
    torch.set_grad_enabled(False)
    return model, {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in a.items()}


# --------------------------------------------------------------------------------------------
# IR building
# --------------------------------------------------------------------------------------------

def _np(t) -> np.ndarray:
    return t.detach().cpu().double().numpy()


class _Builder:
    """Graph builder that tracks consumed state_dict entries, per-layer precision, binarization sites
    and the float64 value of every float constant (analysis only, never serialized)."""

    def __init__(self, model, binarize: str):
        self.g = Graph()
        self.model = model
        self.sd = dict(model.state_dict())
        self.used: set[str] = set()
        self.precision: list[dict] = []
        self.weight_diag: list[dict] = []
        self.sites: list[dict] = []           # binarization / quantization points, execution order
        self.f64: dict[str, np.ndarray] = {}  # constant name -> exact float64 value
        self.quant_params: list[dict] = []
        self.prefix_of = {id(m): n for n, m in model.named_modules()}
        self.binarize = binarize
        self._idx_cache: dict = {}

    # ---------------------------------------------------------------- bookkeeping
    def name(self, module) -> str:
        return self.prefix_of[id(module)]

    def p(self, module, attr: str) -> np.ndarray:
        pre = self.name(module)
        full = f"{pre}.{attr}" if pre else attr
        if full not in self.sd:
            raise KeyError(full)
        self.used.add(full)
        return _np(self.sd[full])

    def mark(self, module, attr: str) -> None:
        full = f"{self.name(module)}.{attr}"
        if full not in self.sd:
            raise KeyError(full)
        self.used.add(full)

    def const(self, tag: str, arr, dtype="f32", layout="", f64=None) -> str:
        name = self.g.add_constant(self.g.unique(tag), np.asarray(arr), dtype=dtype, layout=layout)
        if dtype == "f32":
            self.f64[name] = np.asarray(arr if f64 is None else f64, np.float64)
        return name

    def node(self, op, inputs, attrs=None, tag=None) -> str:
        return self.g.add_node(op, inputs, attrs or {}, name=self.g.unique(tag or op.lower()))

    # ---------------------------------------------------------------- elementary layers
    def bn(self, x: str, m) -> str:
        """BatchNormT in eval mode -> ChannelAffine over the last dim."""
        if type(m).__name__ != "BatchNormT":
            raise TypeError(f"{self.name(m)}: expected BatchNormT, got {type(m).__name__}")
        if m.training or not m.track_running_stats or not m.affine or m.running_mean is None:
            raise ValueError(f"{self.name(m)}: unsupported BN configuration")
        w, b, mu, var = self.p(m, "weight"), self.p(m, "bias"), self.p(m, "running_mean"), self.p(m, "running_var")
        self.mark(m, "num_batches_tracked")
        scale = w / np.sqrt(var + float(m.eps))
        shift = b - mu * scale
        n = self.name(m)
        self.precision.append({"layer": n, "type": "BatchNormT(eval)->ChannelAffine", "weights": "fp32", "activations": "fp32"})
        return self.node("ChannelAffine", [x, self.const(n + "/scale", scale), self.const(n + "/shift", shift)], tag=n)

    def affine(self, x: str, scale=None, bias=None, tag="affine", scale64=None) -> str:
        ins = [x, self.const(tag + "/scale", scale, f64=scale64) if scale is not None else "",
               self.const(tag + "/bias", bias) if bias is not None else ""]
        while ins[-1] == "":
            ins.pop()
        return self.node("ChannelAffine", ins, tag=tag)

    def rprelu(self, x: str, m) -> str:
        """RPReLU (BHViT.py:24-33): prelu(x - move1) + move2 with a per-channel slope."""
        if type(m).__name__ != "RPReLU" or type(m.prelu).__name__ != "PReLU":
            raise TypeError(f"{self.name(m)}: expected RPReLU")
        n = self.name(m)
        mv1, mv2, slope = self.p(m, "move1"), self.p(m, "move2"), self.p(m.prelu, "weight")
        if slope.shape != mv1.shape:
            raise ValueError(f"{n}: PReLU must be per-channel")
        t = self.affine(x, bias=-mv1, tag=n + "/move1")  # x + (-m) == x - m exactly in IEEE
        t = self.node("Activation", [t, self.const(n + "/slope", slope)], {"type": "prelu"}, tag=n + "/prelu")
        self.precision.append({"layer": n, "type": "RPReLU->ChannelAffine+prelu+ChannelAffine", "weights": "fp32", "activations": "fp32"})
        return self.affine(t, bias=mv2, tag=n + "/move2")

    def layerscale(self, x: str, m, pre_scale: float = 1.0) -> str:
        """LayerScale (BHViT.py:36-47): x*alpha + move. pre_scale folds the preceding /3 or /5 of
        Shift/Shift2 into alpha (fp64 product, stored f32)."""
        if type(m).__name__ != "LayerScale":
            raise TypeError(f"{self.name(m)}: expected LayerScale, got {type(m).__name__}")
        a = self.p(m, "alpha") * pre_scale
        return self.affine(x, scale=a, bias=self.p(m, "move"), tag=self.name(m))

    def add(self, a: str, b: str, tag="add") -> str:
        return self.node("Add", [a, b], tag=tag)

    def reshape(self, x: str, shape, tag="reshape") -> str:
        return self.node("Reshape", [x], {"shape": [int(s) for s in shape]}, tag=tag)

    def concat(self, xs, axis: int, tag="concat") -> str:
        return self.node("Concat", list(xs), {"axis": int(axis)}, tag=tag)

    def gather_rows(self, x: str, n_rows_in: int, d: int, idx: np.ndarray, key, out_shape, tag) -> str:
        """y = rows of x (viewed as [n_rows_in, d]) selected by constant indices idx, reshaped."""
        idx = np.asarray(idx).reshape(-1)
        if idx.min() < 0 or idx.max() >= n_rows_in:
            raise ValueError(f"{tag}: gather index out of range")
        if key not in self._idx_cache:
            self._idx_cache[key] = self.const("idx/" + "_".join(str(k) for k in key), idx.astype(np.int32), dtype="i32")
        t = self.reshape(x, [n_rows_in, d], tag=tag + "/rows")
        t = self.node("Gather", [t, self._idx_cache[key]], tag=tag)
        return self.reshape(t, out_shape, tag=tag + "/out")

    # ---------------------------------------------------------------- binary layers
    def _wdiag(self, n: str, w64: np.ndarray, axes, torch_centred32) -> np.ndarray:
        centred = w64 - np.mean(w64, axis=axes, keepdims=True)
        if np.any(centred == 0):
            raise ValueError(f"{n}: exact-zero centred weight in fp64 (sign would be 0)")
        s64 = np.where(centred > 0, 1, -1)
        s32 = np.sign(torch_centred32)
        self.weight_diag.append({"layer": n, "n_weights": int(w64.size), "fp32_exact_zero_centred": int(np.sum(s32 == 0)),
                                 "fp32_sign_differs_from_fp64_nonzero": int(np.sum((s32 != 0) & (s32 != s64)))})
        return s64.astype(np.int8)

    def binlinear(self, x: str, m, move: np.ndarray) -> str:
        """QuantizeLinear(x + move), W1A1, no bias -> BDense(threshold = -move)."""
        import torch
        if type(m).__name__ != "QuantizeLinear":
            raise TypeError(f"{self.name(m)}: expected QuantizeLinear")
        if m.weight_bits != 1 or m.input_bits != 1 or m.recu or m.bias is not None:
            raise ValueError(f"{self.name(m)}: only W1A1 without bias / recu is verified")
        n = self.name(m)
        w = self.p(m, "weight")  # [O, K]
        if move.shape != (w.shape[1],):
            raise ValueError(f"{n}: move shape {move.shape} != K {w.shape[1]}")
        wt = m.weight.detach()
        c32 = (wt - torch.mean(wt, dim=-1, keepdim=True)).numpy()   # utils_quant.py:218, fp32 as in torch
        wb = self._wdiag(n, w, -1, c32)
        scale = np.mean(np.abs(w), axis=1)                          # utils_quant.py:216 (un-centred)
        self.precision.append({"layer": n, "type": "QuantizeLinear->BDense", "weights": "binary(+-1)*mean|W| per out-row",
                               "activations": f"binary sign(x+move) [{self.binarize}]", "shape": list(w.shape)})
        node = self.node("BDense", [x, self.const(n + "/w", wb, dtype="bit"), self.const(n + "/scale", scale), "",
                                    self.const(n + "/thr", -move)], {"input_binarize": self.binarize}, tag=n)
        self.sites.append({"kind": "bdense", "node": node.rsplit(":", 1)[0], "torch_module": n, "ir_input": x})
        return node

    def binconv(self, x: str, m, move: np.ndarray, has_bias: bool) -> str:
        """QuantizeConv2d(x + move) W1A1 -> BConv2D(threshold = -move, pad_value 0)."""
        import torch
        if type(m).__name__ != "QuantizeConv2d":
            raise TypeError(f"{self.name(m)}: expected QuantizeConv2d")
        if m.weight_bits != 1 or m.input_bits != 1:
            raise ValueError(f"{self.name(m)}: only W1A1 is verified")
        if m.padding_mode != "zeros":
            raise ValueError(f"{self.name(m)}: only zero padding is verified")
        n = self.name(m)
        w = self.p(m, "weight")  # [O, I/g, KH, KW]
        o, kic, kh, kw = w.shape
        wt = m.weight.detach()
        c32 = (wt - wt.mean([1, 2, 3], keepdim=True)).numpy()      # utils_quant.py:343, fp32 as in torch
        wb = self._wdiag(n, w, (1, 2, 3), c32).transpose(0, 2, 3, 1)  # OHWI
        scale = np.mean(np.abs(w), axis=(1, 2, 3))
        sh, sw = m.stride
        dh, dw = m.dilation
        if m.padding == "same":
            if (sh, sw) != (1, 1):
                raise ValueError(f"{n}: padding='same' needs stride 1")
            th, tw = dh * (kh - 1), dw * (kw - 1)
            pads = [th // 2, th - th // 2, tw // 2, tw - tw // 2]  # torch 'same': extra pad at the end
        else:
            ph, pw = m.padding
            pads = [ph, ph, pw, pw]
        ins = [x, self.const(n + "/w", wb, dtype="bit", layout="OHWI_bit"), self.const(n + "/scale", scale)]
        if (m.bias is not None) != has_bias:
            raise ValueError(f"{n}: unexpected bias configuration")
        ins.append(self.const(n + "/bias", self.p(m, "bias")) if has_bias else "")
        ins.append(self.const(n + "/thr", -move))
        self.precision.append({"layer": n, "type": "QuantizeConv2d->BConv2D", "weights": "binary(+-1)*mean|W| per out-channel",
                               "activations": f"binary sign(x+move) [{self.binarize}]", "shape": list(w.shape),
                               "groups": int(m.groups), "dilation": [int(dh), int(dw)], "stride": [int(sh), int(sw)], "pads": pads})
        node = self.node("BConv2D", ins, {"strides": [int(sh), int(sw)], "dilations": [int(dh), int(dw)], "pads": pads,
                                          "groups": int(m.groups), "pad_value": 0, "input_binarize": self.binarize}, tag=n)
        self.sites.append({"kind": "bconv", "node": node.rsplit(":", 1)[0], "torch_module": n, "ir_input": x})
        return node

    # ---------------------------------------------------------------- index tracing
    def traced(self, fn, key):
        """Constant row indices obtained by running an official index-moving function on an index tensor."""
        if key not in self._idx_cache:
            self._idx_cache[("np",) + key] = fn()
        return self._idx_cache[("np",) + key]

    # ---------------------------------------------------------------- blocks
    def shifts(self, lo: str, hsn: str, m, h: int, w: int, c: int) -> str:
        """shift3 / shift5 branches (BHViT.py:483-490 / 563-570), in the source's order."""
        import torch
        hw = h * w

        def roll_idx(shift: int, dim: int) -> np.ndarray:  # dim 2 = H, 3 = W of a BCHW tensor
            def f():
                ids = torch.arange(hw, dtype=torch.float64).reshape(1, 1, h, w)
                return torch.roll(ids, shift, dims=dim).reshape(-1).numpy().astype(np.int64)
            return self.traced(f, ("roll", h, w, shift, dim))

        def shift_sum(mod, dim: int, tag: str):
            tn = type(mod).__name__
            shifts = {"Shift": (1, -1), "Shift2": (1, -1, 2, -2)}[tn]
            s = hsn
            for k in shifts:
                r = self.gather_rows(hsn, hw, c, roll_idx(k, dim), ("roll", h, w, k, dim), [1, h, w, c], tag=f"{tag}/roll{k:+d}")
                s = self.add(s, r, tag=tag + "/add")
            return s, 1.0 / (len(shifts) + 1)

        def chmix(mod, tag: str):
            if type(mod).__name__ != "Shift_channel_mix" or c % 4:
                raise TypeError(f"{tag}: expected Shift_channel_mix with C % 4 == 0")
            ss = int(mod.shift_size)

            def f():  # row r = (pixel*4 + group) of x viewed as [H*W*4, C/4]
                ids = (torch.arange(hw, dtype=torch.float64).reshape(1, h, w) * 4)[None].repeat(1, 4, 1, 1)
                ids = ids + torch.arange(4, dtype=torch.float64).reshape(1, 4, 1, 1)
                out = mod(ids)                                    # official module on the index tensor
                return out[0].permute(1, 2, 0).reshape(-1).numpy().astype(np.int64)
            idx = self.traced(f, ("chmix", h, w, ss))
            return self.gather_rows(hsn, hw * 4, c // 4, idx, ("chmix", h, w, ss), [1, h, w, c], tag=tag)

        for enabled, sfx in ((m.shift3, "3"), (m.shift5, "5")):
            if not enabled:
                continue
            for kind, dim in (("h", 2), ("w", 3)):
                sm, ls = getattr(m, f"shift_{kind}{sfx}"), getattr(m, f"layerscale_{kind}{sfx}")
                s, k = shift_sum(sm, dim, self.name(sm))
                lo = self.add(lo, self.layerscale(s, ls, pre_scale=k), tag=self.name(sm) + "/acc")
            sm, ls = getattr(m, f"shift_ch{sfx}"), getattr(m, f"layerscale_ch{sfx}")
            lo = self.add(lo, self.layerscale(chmix(sm, self.name(sm)), ls), tag=self.name(sm) + "/acc")
        return lo

    def intermediate(self, x: str, m) -> str:
        """ViTIntermediate (BHViT.py:290-307)."""
        if type(m).__name__ != "ViTIntermediate":
            raise TypeError(self.name(m))
        y = self.bn(self.binlinear(x, m.dense, self.p(m, "move")), m.norm)
        rep = self.concat([x] * int(m.expansion_ratio), axis=-1, tag=self.name(m) + "/repeat")
        return self.rprelu(self.add(y, rep, tag=self.name(m) + "/shortcut"), m.rprelu)

    def output(self, x: str, m, shape_in, c_out: int) -> str:
        """ViTOutput (BHViT.py:310-333): BN(binlinear(x + move)) + AvgPool1d(ratio)(x) -> RPReLU -> LayerScale."""
        if type(m).__name__ != "ViTOutput" or type(m.layerscale).__name__ != "LayerScale":
            raise TypeError(self.name(m))
        pool = m.pooling
        k = int(pool.kernel_size[0])
        if (type(pool).__name__ != "AvgPool1d" or tuple(pool.stride) != (k,) or tuple(pool.padding) != (0,) or pool.ceil_mode
                or shape_in[-1] != k * c_out):
            raise ValueError(f"{self.name(m)}: unexpected AvgPool1d configuration")
        y = self.bn(self.binlinear(x, m.dense, self.p(m, "move")), m.norm)
        pooled = self.node("ReduceMean", [self.reshape(x, list(shape_in[:-1]) + [c_out, k], tag=self.name(pool) + "/split")],
                           {"axis": -1, "keepdims": 0}, tag=self.name(pool))
        y = self.rprelu(self.add(y, pooled, tag=self.name(m) + "/shortcut"), m.rprelu)
        return self.layerscale(y, m.layerscale)

    def mlp_and_shifts(self, h1: str, m, hh: int, ww: int, c: int) -> str:
        hsn2 = self.bn(h1, m.layernorm_after)
        inter = self.intermediate(hsn2, m.intermediate)
        ci = int(m.intermediate.dense.out_features)
        out = self.output(inter, m.output, [1, hh, ww, ci], c)
        lo = self.add(out, h1, tag=self.name(m) + "/residual2")
        return self.shifts(lo, hsn2, m, hh, ww, c)

    def gclayer(self, h: str, m, hh: int, ww: int, c: int) -> str:
        """GCLayer (BHViT.py:432-493). h: NHWC [1, H, W, C]."""
        if type(m).__name__ != "GCLayer" or type(m.GC).__name__ != "token_mixer":
            raise TypeError(self.name(m))
        hsn = self.bn(h, m.layernorm_before)
        gc = m.GC
        mv = self.p(gc.move, "bias").reshape(-1)
        branches = [self.rprelu(self.binconv(hsn, conv, mv, has_bias=True), act)
                    for conv, act in ((gc.cov1, gc.act1), (gc.cov2, gc.act2), (gc.cov3, gc.act3))]
        s = self.add(self.add(branches[0], branches[1], tag=self.name(gc) + "/sum"), branches[2], tag=self.name(gc) + "/sum")
        h1 = self.add(self.bn(s, gc.norm), h, tag=self.name(m) + "/residual1")
        return self.mlp_and_shifts(h1, m, hh, ww, c)

    def self_attention(self, x: str, sa, hh: int, ww: int, c: int) -> str:
        """BHViTSelfAttention (BHViT.py:117-238) on NHWC [1, H, W, C]; returns NHWC [1, H, W, C]."""
        import torch
        if type(sa).__name__ != "BHViTSelfAttention" or type(sa.token_FA).__name__ != "Token_for_Attention":
            raise TypeError(self.name(sa))
        if sa.att_prob_clip2 is not None or type(sa.att_prob_quantizer).__name__ != "BinaryActivation_Attention":
            raise ValueError(f"{self.name(sa)}: only the BinaryActivation_Attention (non-GSB) path is verified")
        n = self.name(sa)
        tfa, ws = sa.token_FA, int(sa.windows_size)
        if tfa.window_size != ws or hh % ws or ww % ws or hh != ww:
            raise ValueError(f"{n}: window size {ws} must divide square H=W ({hh}x{ww})")
        nwh, nww = hh // ws, ww // ws
        nw, l0 = nwh * nww, ws * ws
        # --- Token_for_Attention
        win = self.reshape(x, [1, nwh, ws, nww, ws, c], tag=n + "/win")
        win = self.node("Transpose", [win], {"perm": [0, 1, 3, 2, 4, 5]}, tag=n + "/win")
        win = self.reshape(win, [nw, l0, c], tag=n + "/win")
        pa, pm = tfa.merge_avg, tfa.merge_max
        for p_ in (pa, pm):
            if (tuple(np.atleast_1d(p_.kernel_size)) not in ((ws,), (ws, ws)) or tuple(np.atleast_1d(p_.stride)) not in ((ws,), (ws, ws))
                    or p_.ceil_mode):
                raise ValueError(f"{n}: unexpected merge pooling {p_}")
        if not pa.count_include_pad or pm.dilation != 1:
            raise ValueError(f"{n}: unexpected merge pooling flags")
        pad = int(pa.padding) if np.isscalar(pa.padding) else int(pa.padding[0])
        if int(np.atleast_1d(pm.padding)[0]) != pad:
            raise ValueError(f"{n}: avg/max merge padding differ")
        pool_attrs = {"kernel": [ws, ws], "strides": [ws, ws], "pads": [pad] * 4}
        avg = self.node("AvgPool2D", [x], dict(pool_attrs, count_include_pad=1), tag=self.name(pa))
        mx = self.node("MaxPool2D", [x], pool_attrs, tag=self.name(pm))
        mh = (hh + 2 * pad - ws) // ws + 1
        mw = (ww + 2 * pad - ws) // ws + 1
        n1 = mh * mw
        if n1 != nw:
            raise ValueError(f"{n}: window_reverse_high needs #merge tokens ({n1}) == #windows ({nw})")
        a1 = self.p(tfa, "a1").reshape(-1)
        one_minus = (np.float32(1.0) - a1.astype(np.float32)).astype(np.float64)  # torch fp32 (1.0 - a1)
        merge = self.add(self.affine(mx, scale=a1, tag=n + "/a1"), self.affine(avg, scale=one_minus, tag=n + "/1-a1", scale64=1.0 - a1),
                         tag=n + "/merge")
        merge = self.reshape(merge, [1, n1, c], tag=n + "/merge")
        merge = self.concat([merge] * nw, axis=0, tag=n + "/repeat_interleave")
        tok = self.concat([win, merge], axis=1, tag=n + "/tokens")
        ln = tfa.norm
        if type(ln).__name__ != "LayerNorm" or not ln.elementwise_affine or tuple(ln.normalized_shape) != (c,):
            raise TypeError(f"{self.name(ln)}: expected nn.LayerNorm({c})")
        tok = self.node("LayerNorm", [tok, self.const(self.name(ln) + "/gamma", self.p(ln, "weight")),
                                      self.const(self.name(ln) + "/beta", self.p(ln, "bias"))], {"eps": float(ln.eps)}, tag=self.name(ln))
        self.precision.append({"layer": self.name(ln), "type": "nn.LayerNorm->LayerNorm", "weights": "fp32", "activations": "fp32"})
        L = l0 + n1
        # --- Q, K, V
        nh, dh = int(sa.num_attention_heads), int(sa.attention_head_size)
        if nh * dh != c or sa.all_head_size != c:
            raise ValueError(f"{n}: heads {nh} x {dh} != {c}")
        mixed, pre = {}, {}
        for t in ("q", "k", "v"):
            lin = {"q": sa.query, "k": sa.key, "v": sa.value}[t]
            y = self.bn(self.binlinear(tok, lin, self.p(sa, f"move{t}")), getattr(sa, f"norm{t}"))
            y = self.rprelu(self.add(y, tok, tag=f"{n}/mixed_{t}"), getattr(sa, f"rprelu{t}"))
            mixed[t] = y
            pre[t] = self.affine(y, bias=self.p(sa, f"move{t}2"), tag=f"{n}/move{t}2")
            self.sites.append({"kind": "qkv_sign", "node": pre[t].rsplit(":", 1)[0], "torch_module": self.name(getattr(sa, f"rprelu{t}")),
                               "torch_add": f"{n}.move{t}2", "ir_input": pre[t]})

        def heads(t: str) -> str:
            r = self.reshape(t, [nw, L, nh, dh], tag=n + "/heads")
            return self.node("Transpose", [r], {"perm": [0, 2, 1, 3]}, tag=n + "/heads")

        vs = self.node("Activation", [pre["v"]], {"type": "sign_zero"}, tag=n + "/sign_v")
        if 4 ** round(math.log(dh, 4)) != dh:
            raise ValueError(f"{n}: 1/sqrt({dh}) is not exact; BMatMul scale would not match x / sqrt(dh)")
        scores = self.node("BMatMul", [heads(pre["q"]), heads(pre["k"])],
                           {"transpose_b": 1, "a_binarize": "sign_zero", "b_binarize": "sign_zero", "scale": 1.0 / math.sqrt(dh)},
                           tag=n + "/scores")
        probs = self.node("Softmax", [scores], tag=n + "/softmax")
        # --- 2-bit attention probabilities
        qz = sa.att_prob_quantizer
        qn, qp = 0, 2 ** (qz.nbits - 1) - 1
        self.mark(qz, "init_state")
        alpha, zp = self.p(qz, "alpha"), self.p(qz, "zero_point")
        g_ = qz.num_head / math.sqrt(nw * nh * L * L * qp)       # utils_quant.py:560 at batch size 1
        eff32 = {k: qz.grad_scale(getattr(qz, k).detach().float(), g_).numpy().astype(np.float64) for k in ("alpha", "zero_point")}
        eff64 = {k: qz.grad_scale(getattr(qz, k).detach().double(), g_).numpy() for k in ("alpha", "zero_point")}
        self.quant_params.append({"module": self.name(qz), "num_heads": nh, "Qn": qn, "Qp": qp, "g_batch1": g_,
                                  "alpha": alpha.tolist(), "zero_point": zp.tolist(),
                                  "grad_scale_fp32_changes_alpha": bool(np.any(eff32["alpha"] != alpha)),
                                  "grad_scale_fp32_changes_zero_point": bool(np.any(eff32["zero_point"] != zp))})
        pq = self.node("FakeQuant", [probs, self.const(self.name(qz) + "/alpha", eff32["alpha"], f64=eff64["alpha"]),
                                     self.const(self.name(qz) + "/zero_point", eff32["zero_point"], f64=eff64["zero_point"])],
                       {"axis": 1, "pre_scale": 3.0, "qmin": qn, "qmax": qp}, tag=self.name(qz))
        self.sites.append({"kind": "attn_quant", "node": pq.rsplit(":", 1)[0], "torch_module": self.name(qz), "ir_input": probs})
        self.precision.append({"layer": n + "/attention", "type": "BMatMul(sign x sign)/sqrt(dh) -> Softmax -> FakeQuant(2-bit, 4 levels) -> MatMul(x sign(V))",
                               "weights": "-", "activations": "Q,K,V ternary torch.sign; probs 2-bit (q*alpha_h, q in 0..3)"})
        ctx = self.node("MatMul", [pq, heads(vs)], {"transpose_b": 0, "scale": 1.0}, tag=n + "/context")
        ctx = self.node("Transpose", [ctx], {"perm": [0, 2, 1, 3]}, tag=n + "/context")
        ctx = self.reshape(ctx, [nw, L, c], tag=n + "/context")
        ctx = self.bn(ctx, sa.norm_context)
        for t in ("q", "k", "v"):
            ctx = self.add(ctx, mixed[t], tag=f"{n}/context+{t}")
        ctx = self.rprelu(ctx, sa.rprelu_context)
        # --- token_split: window tokens -> window_reverse; merge tokens -> mean over windows + nearest upsampling
        def f_rev():
            ids = torch.arange(nw * l0, dtype=torch.float64).reshape(nw, l0, 1)
            return sa.window_reverse(ids, ws, hh, ww, 1).reshape(-1).numpy()  # official function, [1,1,H,W]

        rev = self.traced(f_rev, ("winrev", hh, ws))
        rows_rev = (rev // l0) * L + (rev % l0)                  # row in the [nw*L, C] context of window token
        x1 = self.gather_rows(ctx, nw * L, c, rows_rev.astype(np.int64), ("winrev", hh, ws, L), [1, hh, ww, c], tag=n + "/window_reverse")
        rows_m = (np.arange(nw)[:, None] * L + l0 + np.arange(n1)[None, :]).reshape(-1)
        x2 = self.gather_rows(ctx, nw * L, c, rows_m, ("merge_rows", nw, L, l0), [1, nw, n1, c], tag=n + "/merge_tokens")
        x2 = self.node("ReduceMean", [x2], {"axis": 1, "keepdims": 0}, tag=n + "/window_reverse_high_mean")

        def f_up():
            ids = torch.arange(n1, dtype=torch.float64).reshape(1, 1, mh, mw)
            return torch.nn.functional.interpolate(ids, size=hh, mode="nearest").reshape(-1).numpy().astype(np.int64)  # BHViT.py:179
        x2 = self.gather_rows(x2, n1, c, self.traced(f_up, ("up", mh, hh)), ("up", mh, hh), [1, hh, ww, c], tag=n + "/upsample_nearest")
        parm = self.p(sa, "parm").reshape(-1)
        one_minus = (np.float32(1.0) - parm.astype(np.float32)).astype(np.float64)
        return self.add(self.affine(x1, scale=parm, tag=n + "/parm"), self.affine(x2, scale=one_minus, tag=n + "/1-parm", scale64=1.0 - parm),
                        tag=n + "/token_split")

    def vitlayer(self, h: str, m, hh: int, ww: int, c: int) -> str:
        """BHViTLayer (BHViT.py:496-574)."""
        if type(m).__name__ != "BHViTLayer" or type(m.attention).__name__ != "BHViTAttention":
            raise TypeError(self.name(m))
        hsn = self.bn(h, m.layernorm_before)
        att = self.self_attention(hsn, m.attention.attention, hh, ww, c)
        so = m.attention.output
        if type(so).__name__ != "BHViTSelfOutput":
            raise TypeError(self.name(so))
        y = self.bn(self.binlinear(att, so.dense, self.p(so, "move")), so.norm)
        y = self.layerscale(self.rprelu(self.add(y, att, tag=self.name(so) + "/shortcut"), so.rprelu), so.layerscale)
        h1 = self.add(y, h, tag=self.name(m) + "/residual1")
        return self.mlp_and_shifts(h1, m, hh, ww, c)

    def patch_embed(self, x: str, m, hh: int, ww: int, c: int):
        """BinaryPatchEmbed (BHViT.py:576-625) or fp PatchEmbed (some_fp, BHViT.py:680-716)."""
        tn = type(m).__name__
        if m.H != hh or m.W != ww:
            raise ValueError(f"{self.name(m)}: built for {m.H}x{m.W}, input {hh}x{ww}")
        hs = self.bn(x, m.norm0)
        pos = self.p(m, "position_embeddings")                  # [1, (H/2)^2, C2]
        if tn == "BinaryPatchEmbed":
            pool = m.pool
            ph, pw = m.patch_size
            if tuple(np.atleast_1d(pool.kernel_size)) not in ((ph,), (ph, pw)) or int(np.atleast_1d(pool.padding)[0]) != 0 or pool.ceil_mode:
                raise ValueError(f"{self.name(pool)}: unexpected pooling")
            res = self.node("AvgPool2D", [hs], {"kernel": [ph, pw], "strides": [ph, pw], "pads": [0, 0, 0, 0], "count_include_pad": 1},
                            tag=self.name(pool))
            y = self.binconv(hs, m.proj, self.p(m, "move").reshape(-1), has_bias=False)
            c2 = int(m.proj.out_channels)
            rep = self.concat([res] * (c2 // c), axis=-1, tag=self.name(m) + "/repeat")
            y = self.rprelu(self.add(self.bn(y, m.norm), rep, tag=self.name(m) + "/shortcut"), m.rprelu)
        elif tn == "PatchEmbed":
            proj = m.proj
            if type(proj).__name__ != "Conv2d" or proj.bias is not None or proj.groups != 1 or tuple(proj.padding) != (0, 0):
                raise ValueError(f"{self.name(proj)}: unexpected fp downsample conv")
            w = self.p(proj, "weight").transpose(0, 2, 3, 1)
            sh, sw = proj.stride
            y = self.node("Conv2D", [hs, self.const(self.name(proj) + "/w", w, layout="OHWI")],
                          {"strides": [int(sh), int(sw)], "pads": [0, 0, 0, 0]}, tag=self.name(proj))
            self.precision.append({"layer": self.name(proj), "type": "Conv2d->Conv2D", "weights": "fp32", "activations": "fp32",
                                   "shape": list(proj.weight.shape)})
            c2 = int(proj.out_channels)
            y = self.bn(y, m.norm)
        else:
            raise TypeError(f"{self.name(m)}: unexpected downsample {tn}")
        h2, w2 = hh // 2, ww // 2
        y = self.add(y, self.const(self.name(m) + "/pos", pos.reshape(1, h2, w2, c2)), tag=self.name(m) + "/pos")
        return y, h2, w2, c2


def convert(model, model_id: str, binarize: str = "ge"):
    """Builds the IR graph for an official BHViTForImageClassification in eval mode.
    binarize: 'ge' (deployable, 0 -> +1 at the BDense/BConv2D binarizations) or 'sign_zero'
    (exact torch.sign, reference kernels only). Q/K/V always use sign_zero (exact)."""
    if binarize not in ("ge", "sign_zero"):
        raise ValueError(binarize)
    if type(model).__name__ != "BHViTForImageClassification" or model.training:
        raise TypeError("expected BHViTForImageClassification in eval mode")
    b = _Builder(model, binarize)
    g = b.g
    cfg = model.config
    vit = model.vit
    x = g.add_input("input", [1, cfg.image_size, cfg.image_size, 3])
    # ---- stem (BHViT.py:50-85)
    emb = vit.embeddings
    pe = emb.patch_embeddings
    proj = pe.proj
    if type(proj).__name__ != "Conv2d" or proj.groups != 1 or tuple(proj.padding) != (0, 0) or proj.padding_mode != "zeros":
        raise ValueError("unexpected stem conv")
    if type(pe.act).__name__ != "GELU" or pe.act.approximate != "none":
        raise ValueError("stem activation must be exact (erf) GELU")
    sh, sw = proj.stride
    y = b.node("Conv2D", [x, b.const(b.name(proj) + "/w", b.p(proj, "weight").transpose(0, 2, 3, 1), layout="OHWI"),
                          b.const(b.name(proj) + "/b", b.p(proj, "bias"))], {"strides": [int(sh), int(sw)], "pads": [0, 0, 0, 0]},
               tag=b.name(proj))
    b.precision.append({"layer": b.name(proj), "type": "Conv2d->Conv2D", "weights": "fp32", "activations": "fp32", "shape": list(proj.weight.shape)})
    y = b.node("Activation", [b.bn(y, pe.norm)], {"type": "gelu"}, tag=b.name(pe.act))
    hh = ww = int(emb.num_patches)
    c = int(cfg.hidden_size[0])
    y = b.add(y, b.const("vit.embeddings.position_embeddings", b.p(emb, "position_embeddings").transpose(0, 2, 3, 1)), tag="vit.embeddings/pos")
    # ---- encoder (BHViT.py:739-787)
    enc = vit.encoder
    dep = list(enc.depths)
    for i, layer in enumerate(enc.layerA):
        y = b.gclayer(y, layer, hh, ww, c)
        if i == dep[0] - 1:
            y, hh, ww, c = b.patch_embed(y, enc.patch_embed1, hh, ww, c)
        elif i == dep[0] + dep[1] - 1:
            y, hh, ww, c = b.patch_embed(y, enc.patch_embed2, hh, ww, c)
    for i, layer in enumerate(enc.layerB):
        y = b.vitlayer(y, layer, hh, ww, c)
        if i == dep[2] - 1:
            y, hh, ww, c = b.patch_embed(y, enc.patch_embed3, hh, ww, c)
    # ---- head (BHViT.py:826, 903)
    y = b.bn(y, vit.layernorm)
    y = b.node("GlobalAvgPool", [y], {"keepdims": 0}, tag="token_mean")
    cl = model.classifier
    logits = b.node("Dense", [y, b.const("classifier/w", b.p(cl, "weight")), b.const("classifier/b", b.p(cl, "bias"))], tag="classifier")
    b.precision.append({"layer": "classifier", "type": "Linear->Dense", "weights": "fp32", "activations": "fp32", "shape": list(cl.weight.shape)})
    g.set_outputs(logits)
    unused = sorted(set(b.sd) - b.used)
    if unused:
        raise ValueError(f"{len(unused)} checkpoint tensors not consumed by the mapping: {unused[:10]}")
    info = {"layer_precision": b.precision, "weight_diagnostics": b.weight_diag, "sites": b.sites,
            "float64_constants": b.f64, "attention_quantizers": b.quant_params,
            "n_state_dict_tensors": len(b.sd)}
    return g, info
