"""Bi-ViT (YanjingLi0202/Bi-ViT, AAAI 2024) binary DeiT -> Bitino IR.

Reference implementation: the official repo (bi_vision_transformer.py, Bi_Quant.py, _binary_base_plus.py),
imported read-only. Two repairs are needed to import / run it at all and are recorded by the converter:
  * bi_vision_transformer.py:23-27 imports Binary_plus, lsq_plus, _quan_base_plus, which are not in the
    repo. Nothing the Bi-ViT DeiT models execute comes from them (Conv2dBi is only used by the unused
    bi_PatchEmbed), so they are provided as empty modules.
  * Bi_Quant.ActBi.forward calls grad_scale() but Bi_Quant.py does not import it (NameError). The repo
    defines grad_scale identically in _binary_base_plus.py:20 and utils_quant.py:11
    (y.detach() - (x*s).detach() + x*s); that function is injected into Bi_Quant's namespace.
The model is built like main_1bit.py (create_model(bi_deit_*_patch16_224) = twobits_VisionTransformer
with patch 16, depth 12, mlp_ratio 4, qkv_bias, LayerNorm eps 1e-6; the factory's unconditional
torch.load of the fp DeiT weights is skipped because main_1bit.py then load_state_dict()s the resumed
checkpoint with strict=True, which overwrites every tensor).

Semantics mapped (file:line of the official repo at the commit recorded by tools/convert_bivit.py):
  * PatchEmbed: fp nn.Conv2d 16x16 s16 + bias -> Conv2D; flatten -> Reshape [196, C];
    cat(cls, dist, x) -> Concat axis 0; + pos_embed -> Add (constant).
  * BiLinearBiReal (Bi_Quant.py:77-101): F.linear(sign(x), mean(|W|, dim=1) * sign(W)), NO bias (the
    stored biases are unused by the source and therefore by the IR). The activation STE
    sign(x) - p(x) + p(x) returns torch.sign(x) exactly in fp32/fp64 (|p| <= 1, same sign). The weight STE
    (alpha*sign(W) - clip(W)) + clip(W) is NOT exact in fp32: entries with |W| > 2*alpha are perturbed by a
    few ulp (counted in info["weight_ste_fp32"]); the IR carries the intended alpha*sign(W) (no exact-zero
    weight in either checkpoint).  -> BDense(post_mul = alpha_row). qkv is split per (q|k|v, head) into
    3*H BDense nodes with the corresponding weight rows.
  * ActBi (Bi_Quant.py:162-200), per head h (alpha, zero point along the head axis):
      r = round(zp) (via (zp.round() - zp) + zp, exact), alpha_e = grad_scale(alpha, g) (fp32 value at
      batch size 1), s = sign(fl(x / alpha_e) + r) in {-1,0,1}, code = (s - r) * alpha_e.
    s is reproduced exactly by binarizing an operand u with sign(u) = s:
      r == 0: u = x, s = sign(alpha) * sign(x) (exact unless |x| < |alpha|*2^-149);
      r != 0: x' = sign(alpha)*x, fl(x'/|alpha|) is monotone in x', so {s = +1} = {x' >= T_hi} and
              {s = 0} = {T_lo <= x' < T_hi} for float32 constants found by bisection over the float32
              ordering; u = x' - T (ChannelAffine, sign exact). 0 floats in the tie set -> BMatMul 'ge'
              with T = T_hi; 1 float -> 'sign_zero' with T = T_lo (both exact); >= 2 -> 'ge' (recorded).
    Products of codes (q@k^T, attn@v) are expanded exactly into integer BMatMul counts:
      sum (s_a - r_a)(s_b - r_b) = sa*sb*B(u_a,u_b) - r_b*sa*B(u_a,1) - r_a*sb*B(1,u_b) + K*r_a*r_b
    (sa, sb = sign relation of each operand), combined with AddBroadcast (exact integers) and scaled once
    by ChannelAffine(alpha_a*alpha_b [*1/sqrt(dh)]). Per head, so every scale is a scalar.
  * per-head LayerNorm on q, k (nn.LayerNorm(64), eps 1e-5) -> LayerNorm; Softmax -> Softmax.
  * bi_Mlp: fc1 -> nn.GELU (erf) -> clip(-10, 10) -> fc2 (sign): the clip does not change the sign and is
    dropped; GELU -> Activation gelu (double).
  * Head: final LayerNorm on tokens 0 and 1 (Gather), head / head_dist Dense, (a + b) / 2 ->
    Add + ChannelAffine(0.5) (exact halving).
Binarization modes: BDense inputs fed by a LayerNorm (qkv, fc1) use `binarize` ('ge' deployable or
'sign_zero' exact); proj and fc2 inputs are always 'sign_zero' because their exact zeros are structural
(attn@v and fc1 are integer counts times a scale, no bias, so 0 occurs often and torch.sign gives 0).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import types

import numpy as np

from ..ir import Graph

CONFIGS = {
    "bi-vit-deit-tiny": {"factory": "bi_deit_tiny_patch16_224", "embed_dim": 192, "num_heads": 3},
    "bi-vit-deit-small": {"factory": "bi_deit_small_patch16_224", "embed_dim": 384, "num_heads": 6},
}
MISSING_MODULES = ("Binary_plus", "lsq_plus", "_quan_base_plus")


# --------------------------------------------------------------------------------------------
# float32 helpers: exact thresholds of monotone float32 predicates
# --------------------------------------------------------------------------------------------

_F32_INF_ORD = 0x7F800000


def f32_to_ord(x) -> np.ndarray:
    """Monotone map float32 -> int64 (+0 and -0 both map to 0)."""
    i = np.asarray(x, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ord_to_f32(o) -> np.ndarray:
    o = np.asarray(o, np.int64)
    i = np.where(o < 0, (-o) | 0x80000000, o).astype(np.uint32)
    return i.view(np.float32)


def min_true(pred, shape=()) -> np.ndarray:
    """Smallest float32 x (per element, vectorized) with pred(x) True, for a predicate that is monotone
    non-decreasing over the float32 ordering [-inf, +inf]. Returns +inf+1 ordinal encoded as NaN-free:
    the result is an int64 ordinal array; ordinal _F32_INF_ORD + 1 means 'never true'."""
    lo = np.full(shape, -_F32_INF_ORD, np.int64)
    hi = np.full(shape, _F32_INF_ORD + 1, np.int64)   # sentinel: pred(hi) treated as True
    while np.any(lo < hi):
        mid = (lo + hi) // 2
        act = lo < hi
        t = pred(ord_to_f32(np.minimum(mid, _F32_INF_ORD)))
        t = np.where(mid > _F32_INF_ORD, True, t)
        hi = np.where(act & t, mid, hi)
        lo = np.where(act & ~t, mid + 1, lo)
    return lo


def actbi_threshold(alpha: float, r: int) -> dict:
    """For s = sign(fl32(fl32(x' / a) + r)) with a = |alpha| > 0 and integer r != 0:
    {s = +1} = {x' >= T_hi}, {s = 0} = {T_lo <= x' < T_hi} (float32 ordering)."""
    a = np.float32(abs(alpha))
    nr = np.float32(-r)

    def q(x):
        return (np.asarray(x, np.float32) / a).astype(np.float32)
    o_lo = int(min_true(lambda x: q(x) >= nr))
    o_hi = int(min_true(lambda x: q(x) > nr))
    if not (-_F32_INF_ORD < o_lo <= o_hi < _F32_INF_ORD):
        raise ValueError(f"actbi threshold out of range (alpha={alpha}, r={r})")
    t_lo, t_hi = float(ord_to_f32(o_lo)), float(ord_to_f32(o_hi))
    # verify against the exact predicate around the thresholds
    probe = ord_to_f32(np.arange(o_lo - 3, o_hi + 4))
    s = np.sign((q(probe) + np.float32(r)).astype(np.float32))
    exp = np.where(probe >= np.float32(t_hi), 1.0, np.where(probe >= np.float32(t_lo), 0.0, -1.0))
    assert np.array_equal(s, exp), (alpha, r)
    return {"t_lo": t_lo, "t_hi": t_hi, "n_tie_floats": o_hi - o_lo}


# --------------------------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------------------------

def load_checkpoint(path: str):
    import torch
    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(path, map_location="cpu", weights_only=True)


def timm_legacy_aliases() -> list[str]:
    """The repos target timm 0.4.12; timm >= 0.9 moved timm.models.layers.{drop,helpers,weight_init} to
    timm.layers.*. Alias the moved submodules (DropPath is the identity in eval; to_2tuple / trunc_normal_
    are only used at construction / init). Returns the list of aliases made."""
    import importlib
    import timm  # noqa: F401
    made = []
    for sub in ("drop", "helpers", "weight_init"):
        name = f"timm.models.layers.{sub}"
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            sys.modules[name] = importlib.import_module(f"timm.layers.{sub}")
            made.append(f"alias {name} -> timm.layers.{sub} (timm {timm.__version__}; the repo targets timm 0.4.12)")
    return made


def build_reference(repo_dir: str, model_id: str, ckpt: dict):
    """twobits_VisionTransformer as created by main_1bit.py, checkpoint 'model' loaded with strict=True.
    Returns (model, args_dict, repairs)."""
    import torch
    from functools import partial
    cfg = CONFIGS[model_id]
    a = vars(ckpt["args"])
    if a.get("model") != cfg["factory"] or a.get("nb_classes") != 1000 or a.get("input_size") != 224:
        raise ValueError(f"checkpoint args model={a.get('model')!r} nb_classes={a.get('nb_classes')} do not match {model_id}")
    sys.dont_write_bytecode = True  # never write __pycache__ into the source clone
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    repairs = []
    for name in MISSING_MODULES:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            repairs.append(f"empty stub module {name} (imported by bi_vision_transformer.py, absent from the repo, unused by the DeiT models)")
    repairs += timm_legacy_aliases()
    import _binary_base_plus  # noqa: E402  official
    import Bi_Quant  # noqa: E402  official
    import bi_vision_transformer as bvt  # noqa: E402  official
    if not hasattr(Bi_Quant, "grad_scale"):
        Bi_Quant.grad_scale = _binary_base_plus.grad_scale
        repairs.append("Bi_Quant.grad_scale = _binary_base_plus.grad_scale (ActBi.forward calls it without importing it)")
    model = bvt.twobits_VisionTransformer(patch_size=16, embed_dim=cfg["embed_dim"], depth=12, num_heads=cfg["num_heads"],
                                          mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                          num_classes=1000, drop_rate=float(a.get("drop", 0.0)),
                                          drop_path_rate=float(a.get("drop_path", 0.0)))
    res = model.load_state_dict(ckpt["model"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    model.eval()
    torch.set_grad_enabled(False)
    return model, {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in a.items()}, repairs


def copy_model(model):
    import copy
    return copy.deepcopy(model)


def fp64_exact_binary(model32, dtype=None):
    """float64 copy of the source whose BiLinearBiReal layers use the exact binary weights
    mean_fp32(|W|, dim=1) * sign(W) (the tensor the fp32 source intends; its weight STE
    (bw - clip(W)) + clip(W) perturbs ~12% of the entries by a few ulp, which turns the structural
    exact-zero integer counts of F.linear into rounding noise of random sign). The activation
    expression (Bi_Quant.py:83-91) is kept verbatim; everything else is the official module in float64."""
    import copy
    import torch
    import torch.nn.functional as F
    m64 = copy.deepcopy(model32).to(dtype or torch.float64)
    mods32 = dict(model32.named_modules())
    n = 0
    for name, mod in m64.named_modules():
        if type(mod).__name__ != "BiLinearBiReal":
            continue
        w32 = mods32[name].weight.detach()
        wb = (torch.mean(abs(w32), dim=1, keepdim=True) * torch.sign(w32)).to(dtype or torch.float64)   # exact fp32 products
        mod.register_buffer("_bf_wb", wb)

        def fwd(self, input):
            x = input
            out_forward = torch.sign(input)
            mask1 = x < -1
            mask2 = x < 0
            mask3 = x < 1
            out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
            input = out_forward.detach() - out3.detach() + out3
            return F.linear(input, self._bf_wb)
        mod.forward = types.MethodType(fwd, mod)
        n += 1
    return m64, n


def extra_references(model, x_nhwc, labels, out_dir) -> dict:
    """Diagnostics of the fp32 source (official code, CPU, batch 1):
      * exact_binary_weights_f32: exact binary weights (no weight-STE perturbation);
      * gelu_f64_f32: only the MLP GELU evaluated in float64 (torch erf formula), rest fp32 as released. The fp32
        erf formula x*0.5*(1+erf(x/sqrt2)) cancels to exactly 0 for x <= -5.54 (torch.sign -> 0), the float64 one
        for x <= -8.37 (the IR's double GELU has the same cut-off); PyTorch 1.7.1 (the repo's pinned version)
        computed CUDA GELU as x*normcdf(x), without that cancellation."""
    import torch
    from ..validation.metrics import compare_outputs
    ref32 = np.fromfile(os.path.join(out_dir, "reference_logits.f32"), np.float32).reshape(len(x_nhwc), -1)
    ref64 = np.fromfile(os.path.join(out_dir, "reference_logits_f64.f64"), np.float64).reshape(len(x_nhwc), -1)
    m32x, _ = fp64_exact_binary(model, torch.float32)
    mg = copy_model(model)
    for blk in mg.blocks:
        blk.mlp.act.forward = types.MethodType(lambda self, x: torch.nn.functional.gelu(x.double()).to(x.dtype), blk.mlp.act)
    res = {}
    for key, m, fname in (("exact_binary_weights_f32", m32x, "reference_logits_exact_binary_f32.f32"),
                          ("gelu_f64_f32", mg, "reference_logits_gelu_f64_f32.f32")):
        out = []
        with torch.no_grad():
            for i in range(len(x_nhwc)):
                out.append(m(torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2)))).double().numpy())
        out = np.concatenate(out)
        out.astype(np.float32).tofile(os.path.join(out_dir, fname))
        res[key] = {"file": fname, "top1_accuracy": float(np.mean(out.argmax(1) == labels)),
                    "top5_accuracy": float(np.mean([labels[i] in np.argsort(-out[i])[:5] for i in range(len(out))])),
                    "vs_reference_f32": compare_outputs(ref32, out, labels), "vs_reference_f64": compare_outputs(ref64, out, labels)}
    # GELU zero cut-offs (exact zero output for a negative input)
    xs = torch.linspace(-9, -4, 50001, dtype=torch.float64)
    res["gelu_exact_zero_for_x_le"] = {"float32": float(xs[torch.nn.functional.gelu(xs.float()) == 0].max()),
                                      "float64": float(xs[torch.nn.functional.gelu(xs) == 0].max())}
    return res


# --------------------------------------------------------------------------------------------
# IR builder (shared with torch_gsb)
# --------------------------------------------------------------------------------------------

def _np(t) -> np.ndarray:
    return t.detach().cpu().double().numpy()


class Builder:
    """Graph builder tracking consumed state_dict entries, per-layer precision and the float64 value of
    every float constant (for the float64-constants check; never serialized)."""

    def __init__(self, model, binarize: str):
        self.g = Graph()
        self.model = model
        self.sd = dict(model.state_dict())
        self.used: set[str] = set()
        self.unused_by_source: set[str] = set()
        self.precision: list[dict] = []
        self.f64: dict[str, np.ndarray] = {}
        self.prefix_of = {id(m): n for n, m in model.named_modules()}
        self.binarize = binarize
        self.notes: dict = {}
        self._ones: dict = {}

    def name(self, module) -> str:
        return self.prefix_of[id(module)]

    def key(self, module, attr: str) -> str:
        pre = self.name(module)
        full = f"{pre}.{attr}" if pre else attr
        if full not in self.sd:
            raise KeyError(full)
        return full

    def p(self, module, attr: str) -> np.ndarray:
        k = self.key(module, attr)
        self.used.add(k)
        return _np(self.sd[k])

    def p32(self, module, attr: str):
        k = self.key(module, attr)
        self.used.add(k)
        return self.sd[k].detach().float()

    def mark(self, module, attr: str, unused_by_source: bool = False) -> None:
        k = self.key(module, attr)
        self.used.add(k)
        if unused_by_source:
            self.unused_by_source.add(k)

    def const(self, tag: str, arr, dtype="f32", layout="", f64=None) -> str:
        name = self.g.add_constant(self.g.unique(tag), np.asarray(arr), dtype=dtype, layout=layout)
        if dtype == "f32":
            self.f64[name] = np.asarray(arr if f64 is None else f64, np.float64).reshape(np.shape(arr))
        return name

    def node(self, op, inputs, attrs=None, tag=None) -> str:
        return self.g.add_node(op, inputs, attrs or {}, name=self.g.unique(tag or op.lower()))

    def affine(self, x: str, n: int, scale=None, bias=None, tag="affine", scale64=None, bias64=None) -> str:
        """ChannelAffine over a last dim of size n with scalar or per-channel scale / bias."""
        ins = [x,
               self.const(tag + "/scale", np.broadcast_to(np.asarray(scale, np.float64), (n,)),
                          f64=None if scale64 is None else np.broadcast_to(np.asarray(scale64, np.float64), (n,))) if scale is not None else "",
               self.const(tag + "/bias", np.broadcast_to(np.asarray(bias, np.float64), (n,)),
                          f64=None if bias64 is None else np.broadcast_to(np.asarray(bias64, np.float64), (n,))) if bias is not None else ""]
        while ins[-1] == "":
            ins.pop()
        return self.node("ChannelAffine", ins, tag=tag)

    def reshape(self, x: str, shape, tag="reshape") -> str:
        return self.node("Reshape", [x], {"shape": [int(s) for s in shape]}, tag=tag)

    def ones(self, shape) -> str:
        shape = tuple(int(s) for s in shape)
        if shape not in self._ones:
            self._ones[shape] = self.const("ones", np.ones(shape))
        return self._ones[shape]

    def layernorm(self, x: str, ln, tag=None) -> str:
        if type(ln).__name__ != "LayerNorm" or not ln.elementwise_affine:
            raise TypeError(f"{self.name(ln)}: expected nn.LayerNorm with affine")
        n = tag or self.name(ln)
        return self.node("LayerNorm", [x, self.const(n + "/gamma", self.p(ln, "weight")), self.const(n + "/beta", self.p(ln, "bias"))],
                         {"eps": float(ln.eps)}, tag=n)


# --------------------------------------------------------------------------------------------
# Bi-ViT mapping
# --------------------------------------------------------------------------------------------

class _BiViT(Builder):
    def binlinear(self, x: str, m, rows: slice | None = None, binarize: str | None = None, tag: str | None = None,
                  input_kind: str = "") -> str:
        """BiLinearBiReal (Bi_Quant.py:77-101) on output rows `rows` -> BDense(post_mul = mean|W| per row)."""
        import torch
        if type(m).__name__ != "BiLinearBiReal":
            raise TypeError(f"{self.name(m)}: expected BiLinearBiReal, got {type(m).__name__}")
        n = self.name(m)
        w32 = self.p32(m, "weight")
        if m.bias is not None:
            self.mark(m, "bias", unused_by_source=True)   # F.linear(input, binary_weights): bias never added
        sc32 = torch.mean(abs(w32), dim=1, keepdim=True)  # Bi_Quant.py:92, fp32 as in the source
        if bool((w32 == 0).any()):
            raise ValueError(f"{n}: exact-zero latent weight (torch.sign gives 0); w_zero not implemented")
        # fp32 weight STE: count entries where the source's F.linear weight differs from sc*sign(W) (once per module)
        seen = self.notes.setdefault("_ste_seen", set())
        if n not in seen:
            seen.add(n)
            bw = sc32 * torch.sign(w32)
            c = torch.clamp(w32, -1.0, 1.0)
            ste = (bw - c) + c
            d = self.notes.setdefault("weight_ste_fp32", {"n_weights": 0, "n_perturbed": 0, "max_rel": 0.0})
            d["n_weights"] += int(w32.numel())
            d["n_perturbed"] += int((ste != bw).sum())
            d["max_rel"] = max(d["max_rel"], float(((ste - bw).abs() / sc32).max()))
        r = rows or slice(0, w32.shape[0])
        sgn = np.where(w32[r].numpy() > 0, 1, -1).astype(np.int8)
        mode = binarize or self.binarize
        tg = tag or n
        self.precision.append({"layer": tg, "type": "BiLinearBiReal->BDense", "weights": "binary(+-1)*mean|W| per out-row (fp32)",
                               "activations": f"binary torch.sign(x) [{mode}]{input_kind}", "shape": list(sgn.shape), "bias": "unused by the source"})
        return self.node("BDense", [x, self.const(tg + "/w", sgn, dtype="bit"),
                                    self.const(tg + "/scale", sc32[r, 0].double().numpy())],   # fp32-derived (= fp64 reference)
                         {"input_binarize": mode}, tag=tg)

    def actbi(self, x: str, m, h: int, d_last: int, n_tokens_numel: int, tag: str) -> dict:
        """Operand for ActBi head h. Returns {u, sigma, mode, r, alpha32, alpha64}: sign(u) = sigma * s."""
        import torch
        if type(m).__name__ != "ActBi":
            raise TypeError(f"{self.name(m)}: expected ActBi")
        for attr in ("init_state", "signed"):
            self.mark(m, attr)
        a32 = self.p32(m, "alpha")
        z32 = self.p32(m, "zero_point")
        g = 1.0 / math.sqrt(n_tokens_numel * 1)                      # Bi_Quant.py:184 at batch size 1, Qp = 1
        zr = (z32.round() - z32) + z32                                 # Bi_Quant.py:187 forward value
        gs = sys.modules["Bi_Quant"].grad_scale
        ae32, ze32 = gs(a32, g), gs(zr, g)
        ae64, ze64 = gs(a32.double(), g), gs(((z32.double().round() - z32.double()) + z32.double()), g)
        alpha, r = float(ae32[h]), float(ze32[h])
        if r != round(r) or float(ze64[h]) != r:
            raise ValueError(f"{tag}: effective zero point {r} is not an integer")
        r = int(round(r))
        rec = {"module": self.name(m), "head": h, "alpha_param": float(a32[h]), "alpha_eff_fp32": alpha,
               "zero_point_param": float(z32[h]), "r": r, "g_batch1": g, "grad_scale_changes_alpha": bool(alpha != float(a32[h]))}
        self.notes.setdefault("actbi", []).append(rec)
        sig = 1 if alpha > 0 else -1
        if alpha == 0:
            raise ValueError(f"{tag}: alpha == 0")
        if r == 0:
            rec.update(mode="sign_zero", operand="x")
            return {"u": x, "sigma": sig, "mode": "sign_zero", "r": 0, "alpha32": alpha, "alpha64": float(ae64[h])}
        t = actbi_threshold(alpha, r)
        mode, tt = ("sign_zero", t["t_lo"]) if t["n_tie_floats"] == 1 else ("ge", t["t_hi"])
        rec.update(mode=mode, threshold=tt, n_tie_floats=t["n_tie_floats"], operand="sign(alpha)*x - T")
        t64 = -r * abs(float(ae64[h]))                                  # exact real threshold of fl64(x'/|a|) + r
        u = self.affine(x, d_last, scale=float(sig) if sig < 0 else None, bias=-tt, bias64=-t64, tag=tag)
        return {"u": u, "sigma": 1, "mode": mode, "r": r, "alpha32": alpha, "alpha64": float(ae64[h])}

    def code_product(self, a: dict, b: dict, k: int, out_shape, transpose_b: int, extra32: float, extra64: float, tag: str) -> str:
        """sum_k code_a * code_b (codes alpha*(s - r)) as exact integer BMatMul counts, scaled once."""
        terms = []
        sa, sb = a["sigma"], b["sigma"]
        bm = {"a_binarize": a["mode"], "b_binarize": b["mode"], "transpose_b": transpose_b}
        terms.append(self.node("BMatMul", [a["u"], b["u"]], dict(bm, scale=float(sa * sb)), tag=tag + "/ab"))
        m_, n_ = out_shape
        if b["r"]:   # - r_b * sa * sum_k s_a  -> [M, 1]
            ones_b = self.ones((1, k)) if transpose_b else self.ones((k, 1))
            terms.append(self.node("BMatMul", [a["u"], ones_b], {"a_binarize": a["mode"], "b_binarize": "sign_zero", "transpose_b": transpose_b,
                                                               "scale": float(-b["r"] * sa)}, tag=tag + "/rowsum"))
        if a["r"]:   # - r_a * sb * sum_k s_b  -> [1, N]
            terms.append(self.node("BMatMul", [self.ones((1, k)), b["u"]], {"a_binarize": "sign_zero", "b_binarize": b["mode"],
                                                                          "transpose_b": transpose_b, "scale": float(-a["r"] * sb)}, tag=tag + "/colsum"))
        acc = terms[0]
        for t in terms[1:]:
            acc = self.node("AddBroadcast", [acc, t], tag=tag + "/sum")
        if a["r"] and b["r"]:
            acc = self.node("AddBroadcast", [acc, self.const(tag + "/krr", np.array([float(k * a["r"] * b["r"])]))], tag=tag + "/sum")
        s32 = float(np.float32(np.float32(a["alpha32"]) * np.float32(b["alpha32"])) * np.float32(extra32))
        s64 = a["alpha64"] * b["alpha64"] * extra64
        return self.affine(acc, n_, scale=s32, scale64=s64, tag=tag + "/scale")

    def attention(self, xn: str, at, n_tok: int, c: int) -> str:
        """bi_Attention (bi_vision_transformer.py:185-230) on xn [N, C]; returns [N, C] (before the residual)."""
        if type(at).__name__ != "bi_Attention":
            raise TypeError(self.name(at))
        n = self.name(at)
        nh = int(at.num_heads)
        dh = c // nh
        if dh * nh != c or at.scale != dh ** -0.5 or 2.0 ** round(math.log2(at.scale)) != at.scale:
            raise ValueError(f"{n}: head geometry / scale (1/sqrt(dh) must be a power of two to scale exactly)")
        for t in ("q", "k"):
            if tuple(getattr(at, f"norm_{t}").normalized_shape) != (dh,):
                raise ValueError(f"{n}: norm_{t} shape")
        ctx = []
        for h in range(nh):
            ops = {}
            for i, t in enumerate(("q", "k", "v")):
                rows = slice(i * c + h * dh, i * c + (h + 1) * dh)       # reshape(B, N, 3, H, dh) order
                y = self.binlinear(xn, at.qkv, rows=rows, tag=f"{n}.qkv/{t}{h}", input_kind=" (LayerNorm output)")
                if t in ("q", "k"):
                    y = self.layernorm(y, getattr(at, f"norm_{t}"), tag=f"{n}.norm_{t}/h{h}")
                numel = nh * n_tok * dh
                ops[t] = self.actbi(y, getattr(at, f"{t}_act"), h, dh, numel, tag=f"{n}.{t}_act/h{h}")
            logits = self.code_product(ops["q"], ops["k"], dh, (n_tok, n_tok), 1, float(at.scale), float(at.scale), tag=f"{n}/qk{h}")
            p = self.node("Softmax", [logits], tag=f"{n}/softmax{h}")
            pa = self.actbi(p, at.attn_act, h, n_tok, nh * n_tok * n_tok, tag=f"{n}.attn_act/h{h}")
            ctx.append(self.code_product(pa, ops["v"], n_tok, (n_tok, dh), 0, 1.0, 1.0, tag=f"{n}/av{h}"))
        self.precision.append({"layer": n + "/attention", "type": "per-head: LayerNorm(q,k) -> ActBi codes -> BMatMul counts -> Softmax -> ActBi -> BMatMul",
                               "weights": "-", "activations": "Q, K, V, attention probs: 1-bit affine codes alpha_h*(s - r_h) (ActBi)"})
        cat = self.node("Concat", ctx, {"axis": -1}, tag=n + "/heads")
        return self.binlinear(cat, at.proj, binarize="sign_zero", input_kind=" (attn@v counts: exact zeros are structural)")

    def mlp(self, xn: str, m) -> str:
        if type(m).__name__ != "bi_Mlp" or type(m.act).__name__ != "GELU" or m.act.approximate != "none":
            raise TypeError(self.name(m))
        y = self.binlinear(xn, m.fc1, input_kind=" (LayerNorm output)")
        y = self.node("Activation", [y], {"type": "gelu"}, tag=self.name(m.act))
        self.precision.append({"layer": self.name(m.act), "type": "nn.GELU (erf) -> Activation gelu (double); clip(-10,10) dropped (sign-preserving)",
                               "weights": "-", "activations": "fp32"})
        return self.binlinear(y, m.fc2, binarize="sign_zero", input_kind=" (GELU of integer counts: exact zeros are structural)")


def convert(model, model_id: str, binarize: str = "ge"):
    """IR for an official twobits_VisionTransformer (eval). binarize: 'ge' (deployable) or 'sign_zero'
    for the LayerNorm-fed BDense inputs (qkv, fc1); proj / fc2 / attention are always exact."""
    if binarize not in ("ge", "sign_zero"):
        raise ValueError(binarize)
    if type(model).__name__ != "twobits_VisionTransformer" or model.training:
        raise TypeError("expected twobits_VisionTransformer in eval mode")
    b = _BiViT(model, binarize)
    g = b.g
    pe = model.patch_embed
    proj = pe.proj
    if type(pe).__name__ != "PatchEmbed" or type(proj).__name__ != "Conv2d" or proj.groups != 1 or tuple(proj.padding) != (0, 0) \
            or tuple(proj.stride) != tuple(proj.kernel_size) or proj.bias is None:
        raise ValueError("unexpected patch embedding")
    hh, ww = pe.img_size
    x = g.add_input("input", [1, hh, ww, 3])
    c = int(model.embed_dim)
    gh, gw = hh // proj.stride[0], ww // proj.stride[1]
    y = b.node("Conv2D", [x, b.const("patch_embed.proj/w", b.p(proj, "weight").transpose(0, 2, 3, 1), layout="OHWI"),
                          b.const("patch_embed.proj/b", b.p(proj, "bias"))],
               {"strides": [int(s) for s in proj.stride], "pads": [0, 0, 0, 0]}, tag="patch_embed.proj")
    b.precision.append({"layer": "patch_embed.proj", "type": "Conv2d->Conv2D", "weights": "fp32", "activations": "fp32",
                        "shape": list(proj.weight.shape)})
    y = b.reshape(y, [gh * gw, c], tag="patch_embed/flatten")
    if model.dist_token is None or model.head_dist is None:
        raise ValueError("expected the distilled DeiT layout (cls + dist token, head + head_dist)")
    tok = b.node("Concat", [b.const("cls_token", b.p(model, "cls_token").reshape(1, c)),
                            b.const("dist_token", b.p(model, "dist_token").reshape(1, c)), y], {"axis": 0}, tag="tokens")
    n_tok = gh * gw + 2
    y = b.node("Add", [tok, b.const("pos_embed", b.p(model, "pos_embed").reshape(n_tok, c))], tag="pos_embed")
    for blk in model.blocks:
        if type(blk).__name__ != "bi_Block" or type(blk.drop_path).__name__ not in ("Identity", "DropPath"):
            raise TypeError(b.name(blk))
        a = b.attention(b.layernorm(y, blk.norm1), blk.attn, n_tok, c)
        y = b.node("Add", [y, a], tag=b.name(blk) + "/residual1")
        mo = b.mlp(b.layernorm(y, blk.norm2), blk.mlp)
        y = b.node("Add", [y, mo], tag=b.name(blk) + "/residual2")
    outs = []
    for i, hd in ((0, model.head), (1, model.head_dist)):
        r = b.node("Gather", [y, b.const(f"token{i}", np.array([i], np.int32), dtype="i32")], tag=f"token{i}")
        r = b.layernorm(r, model.norm, tag=f"norm/token{i}")
        outs.append(b.node("Dense", [r, b.const(b.name(hd) + "/w", b.p(hd, "weight")), b.const(b.name(hd) + "/b", b.p(hd, "bias"))],
                           tag=b.name(hd)))
        b.precision.append({"layer": b.name(hd), "type": "Linear->Dense", "weights": "fp32", "activations": "fp32",
                            "shape": list(hd.weight.shape)})
    logits = b.affine(b.node("Add", outs, tag="heads/sum"), int(model.num_classes), scale=0.5, tag="heads/mean")
    g.set_outputs(logits)
    if type(model.pre_logits).__name__ != "Identity":
        raise TypeError("pre_logits must be Identity")
    unused = sorted(set(b.sd) - b.used)
    if unused:
        raise ValueError(f"{len(unused)} checkpoint tensors not consumed by the mapping: {unused[:10]}")
    b.notes.pop("_ste_seen", None)
    info = {"layer_precision": b.precision, "float64_constants": b.f64, "notes": b.notes,
            "state_dict_unused_by_source": sorted(b.unused_by_source), "n_state_dict_tensors": len(b.sd)}
    return g, info
