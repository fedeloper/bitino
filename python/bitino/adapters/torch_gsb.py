"""GSB (IMRL/GSB-Vision-Transformer, Neural Networks 2024) binary DeiT-S -> Bitino IR.

Reference implementation: the official repo (One_bit_vit_new.py, quant_new_new.py, test.py), imported read-only;
the model is built like test.py (OQ_VitConfig + args from the checkpoint, onebits_deit_small_patch16_224,
load_state_dict(checkpoint['model']) strict). timm >= 0.9 needs aliases for timm.models.layers.{drop,helpers,
weight_init} (torch_bivit.timm_legacy_aliases).

Semantics (file:line of the official repo at the commit recorded by tools/convert_bivit.py):
  * Q_PatchEmbed(nbits=8) -> QuantizeConv2dQ (quant_new_new.py:525-551): 8-bit per-out-channel LSQ weights
    round_pass(clamp(W/alpha_e, -128, 127)) * alpha_e with alpha_e = grad_scale(alpha, g); fp input, bias.
    The exact fp32 weight tensor the source passes to F.conv2d is captured (hooked call) -> Conv2D.
  * Q_OVitLayerNorm (One_bit_vit_new.py, TF style, eps 1e-12 inside the sqrt) -> LayerNorm.
  * QuantizeLinear W1A1 (quant_new_new.py:481-510): weight BwnQuantizer layerwise sign(W - mean W) * mean|W|
    (one scalar per layer; captured fp32 tensor, checked to be exactly +-m), input x + move, then
      symmetric (qkv, proj, fc1): ElasticQuantBinarizerSigned 1-bit: sign(x + move) * alpha
        -> BDense(threshold -move, post_mul alpha*m, post_add bias); sign(fl(x + move)) == sign(x - (-move)).
      fc2 (symmetric=False): ElasticQuantBinarizerUnsigned 1-bit: round(fl(x + move)/alpha).clamp(0, 1) * alpha
        of x = clip(gelu(.), -10, 10): {0,1} input = [x > T_k] with T_k the largest float32 for which the
        source's float32 expression gives 0 (bisection over the float32 ordering, clip included)
        -> BDense('gt', threshold T, post_mul alpha*m/2, post_add alpha*m*sum_k w/2 + bias).
  * Attention (One_bit_vit_new.py:Q_Attention.forward), heads batched [H, N, dh] (all scales are per layer):
      q, k: sign(q + move_q) * alpha_q, sign(k + move_k) * alpha_k
        -> BMatMul(sign_zero x sign_zero) * (alpha_q*alpha_k/8) (ChannelAffine).
      attn = softmax(.) + move_att (constant [H, N, N]) -> Softmax, Add.
      ElasticQuantAttention (quant_new_new.py:22-75): alpha*[round(P'/alpha) clamped to 1] + alpha2*[P' > 0.7 max]
        + alpha3*[P' > 0.9 max], max over the whole tensor -> three {0,1} operands: P' - T0 (T0 exact float32
        threshold), P' - fl(0.7*max), P' - fl(0.9*max); max = MaxPool2D reduction (exact).
      ElasticQuantQKV (quant_new_new.py:101-166) on v' = v + move_v: a1*sign(v') + a2*sign(v')*T1 + a3*sign(v')*T2,
        T1 = [v' > 0.7 max] + [v' < 0.7 min] (T2 with 0.9) -> sign(v')*T1 = [v' > c+] - [v' < c-] (exact for any
        data: c- <= c+, and v' > c+ implies v' > 0 unless the mask is empty). min = -max(-v').
      attn_code @ v_code = sum_{i,j} alpha_i * a_j * (H_i @ S_j): 15 exact BMatMul counts (gt0 x {sign_zero, gt0}),
        integer differences combined exactly (BMatMul scale +-1, Add), scaled per (i, j) and summed.
    The whole-tensor max/min are taken per sample (the deployable semantics); the official test.py evaluates
    with batch size 96, where they are taken over the batch (reported separately by the converter).
  * Head / head_dist: QuantizeLinear 8-bit (weight_quant_method = input_quant_method = 'elastic', symmetric):
    weights round(W/alpha_w).clamp(-128, 127) * alpha_w (captured fp32 tensor) -> Dense; input x + move ->
    ChannelAffine, then round(x'/alpha).clamp(-128, 127) * alpha -> FakeQuant (same float32 steps); (a + b) / 2.
  * GELU (One_bit_vit_new.gelu, erf form) -> Activation gelu (double).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import types

import numpy as np

from .torch_bivit import Builder, min_true, ord_to_f32, _F32_INF_ORD, timm_legacy_aliases

CONFIGS = {"gsb-deit-s-cifar100-stage2": {"factory": "onebits_deit_small_patch16_224", "num_classes": 100}}
EXPECTED_ARGS = {"model": "onebits_deit_small_patch16_224", "data_set": "CIFAR", "nb_classes": 100, "input_size": 224,
                 "weight_quant_method": "bwn", "input_quant_method": "elastic", "sym_quant_qkvo": True,
                 "weight_layerwise": True, "input_layerwise": True, "learnable_scaling": True,
                 "not_quantize_attention": False, "clip_init_val": 2.5, "model_ema": False}


def load_checkpoint(path: str):
    import torch
    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(path, map_location="cpu", weights_only=True)


def _config(a: dict):
    """test.py:30-48 (OQ_VitConfig defaults) + test.py main() attribute assignments from args."""
    c = types.SimpleNamespace(clip_init_val=2.5, hidden_size=768 // 2, layer_norm_eps=1e-12, weight_bits=1, input_bits=1,
                              num_attention_heads=6)
    c.input_size = a["input_size"]
    c.weight_quant_method = a["weight_quant_method"]
    c.input_quant_method = a["input_quant_method"]
    c.clip_init_val = a["clip_init_val"]
    c.learnable_scaling = a["learnable_scaling"]
    c.sym_quant_qkvo = a["sym_quant_qkvo"]
    c.weight_layerwise = a["weight_layerwise"]
    c.input_layerwise = a["input_layerwise"]
    c.not_quantize_attention = a["not_quantize_attention"]
    return c


def build_reference(repo_dir: str, model_id: str, ckpt: dict):
    import torch
    a = vars(ckpt["args"])
    for k, v in EXPECTED_ARGS.items():
        if a.get(k) != v:
            raise ValueError(f"checkpoint arg {k}={a.get(k)!r}, adapter verified only for {v!r}")
    sys.dont_write_bytecode = True
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    repairs = timm_legacy_aliases()
    import One_bit_vit_new as obv  # noqa: E402  official
    cfg = CONFIGS[model_id]
    model = obv.onebits_deit_small_patch16_224(_config(a), num_classes=cfg["num_classes"])
    res = model.load_state_dict(ckpt["model"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    model.eval()
    torch.set_grad_enabled(False)
    model._bf_rebuild = lambda: build_reference(repo_dir, model_id, ckpt)[0]
    return model, {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in a.items()}, repairs


def copy_model(model):
    """A fresh instance with the same weights (AlphaInit parameters cannot be deep-copied: torch's
    Parameter.__deepcopy__ passes requires_grad to AlphaInit.__init__, which does not accept it)."""
    return model._bf_rebuild()


def capture_quantized_weights(model) -> dict:
    """Runs the fp32 source once and records the weight argument of every F.linear / F.conv2d call made by a
    QuantizeLinear / QuantizeConv2dQ, keyed by module name (exactly the tensors the source uses)."""
    import torch
    import torch.nn.functional as F
    cur, rec = [None], {}
    names = {id(m): n for n, m in model.named_modules()}
    hooks = []
    for m in model.modules():
        if type(m).__name__ in ("QuantizeLinear", "QuantizeConv2dQ"):
            hooks.append(m.register_forward_pre_hook(lambda mod, inp: cur.__setitem__(0, names[id(mod)])))
            hooks.append(m.register_forward_hook(lambda mod, inp, out: cur.__setitem__(0, None)))
    lin, conv = F.linear, F.conv2d

    def plin(x, w, *a, **k):
        if cur[0] is not None:
            rec.setdefault(cur[0], w.detach().clone())
        return lin(x, w, *a, **k)

    def pconv(x, w, *a, **k):
        if cur[0] is not None:
            rec.setdefault(cur[0], w.detach().clone())
        return conv(x, w, *a, **k)
    F.linear, F.conv2d = plin, pconv
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, 224, 224))
    finally:
        F.linear, F.conv2d = lin, conv
        for h in hooks:
            h.remove()
    return rec


def fp64_exact_binary(model32, dtype=None):
    """float64 copy of the source whose QuantizeLinear / QuantizeConv2dQ layers use the quantized weights the
    fp32 source computes (BWN signs of W - mean(W) in fp32, 8-bit LSQ / elastic levels in fp32), substituted at
    F.linear / F.conv2d; activation quantizers and everything else are the official code in float64."""
    import copy
    import torch
    import torch.nn.functional as F
    qn = sys.modules["quant_new_new"]
    rec = capture_quantized_weights(model32)
    m64 = copy_model(model32).to(dtype or torch.float64)
    n = 0
    for name, mod in m64.named_modules():
        kind = type(mod).__name__
        if kind not in ("QuantizeLinear", "QuantizeConv2dQ"):
            continue
        mod.register_buffer("_bf_wq", rec[name].to(dtype or torch.float64))
        if kind == "QuantizeLinear":
            def fwd(self, input):   # quant_new_new.py:497-510 with the fp32-derived weight
                input = self.move(input)
                input = qn.act_quant_fn(input, self.input_clip_val, num_bits=self.input_bits, symmetric=self.symmetric,
                                        quant_method=self.input_quant_method, layerwise=self.input_layerwise)
                out = F.linear(input, self._bf_wq)
                if self.bias is not None:
                    out += self.bias.view(1, -1).expand_as(out)
                return out
        else:
            def fwd(self, x):   # quant_new_new.py:550-551
                return F.conv2d(x, self._bf_wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        mod.forward = types.MethodType(fwd, mod)
        n += 1
    return m64, n


def extra_references(model, x_nhwc, labels, out_dir, batch: int = 96) -> dict:
    """The official fp32 model evaluated in batches of 96 (test.py: batch_size int(1.5 * 64)), where the
    attention / V thresholds use the max / min over the whole batch."""
    import torch
    from ..validation.metrics import compare_outputs
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), batch):
            out.append(model(torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch].transpose(0, 3, 1, 2)))).double().numpy())
    out = np.concatenate(out)
    out.astype(np.float32).tofile(os.path.join(out_dir, f"reference_logits_batch{batch}.f32"))
    ref32 = np.fromfile(os.path.join(out_dir, "reference_logits.f32"), np.float32).reshape(out.shape)
    return {f"official_fp32_batch{batch}": {"file": f"reference_logits_batch{batch}.f32", "top1_accuracy": float(np.mean(out.argmax(1) == labels)),
                                            "vs_reference_f32_batch1": compare_outputs(ref32, out, labels)}}


def _f32(v) -> float:
    return float(np.float32(v))


class _GSB(Builder):
    def eps_clip(self, t) -> float:
        """torch.where(alpha > eps, alpha, eps) with eps = float32(1e-5) (quant_new_new.py)."""
        a = np.float32(float(t))
        e = np.float32(0.00001)
        return float(a if a > e else e)

    def gmax(self, x: str, rows: int, cols: int, tag: str) -> str:
        """Max over every element of x viewed as [rows, cols] -> [1, 1, 1] (two MaxPool2D stages, exact)."""
        t = self.reshape(x, [1, 1, rows, cols], tag=tag + "/view")
        t = self.node("MaxPool2D", [t], {"kernel": [1, rows], "strides": [1, rows], "pads": [0, 0, 0, 0]}, tag=tag + "/rows")
        t = self.reshape(t, [1, 1, cols, 1], tag=tag + "/view2")
        t = self.node("MaxPool2D", [t], {"kernel": [1, cols], "strides": [1, cols], "pads": [0, 0, 0, 0]}, tag=tag + "/all")
        return t   # [1, 1, 1, 1]

    def scaled_scalar(self, m: str, factor: float, tag: str) -> str:
        """fl32(factor * m) for a [1,1,1,1] scalar, reshaped to [1, 1, 1] (factor as torch casts it: float32)."""
        y = self.affine(m, 1, scale=_f32(factor), scale64=factor, tag=tag)
        return self.reshape(y, [1, 1, 1], tag=tag + "/r")

    def qlinear(self, x: str, m, wq, rows: slice | None = None, tag: str | None = None, binarize: str | None = None,
                fc2_input_clip: tuple | None = None) -> str:
        """QuantizeLinear W1A1 (BWN weights, elastic 1-bit input)."""
        import torch
        if type(m).__name__ != "QuantizeLinear" or m.weight_bits != 1 or m.input_bits != 1 or m.weight_quant_method != "bwn" \
                or m.input_quant_method != "elastic" or not m.weight_layerwise or not m.input_layerwise:
            raise TypeError(f"{self.name(m)}: unexpected QuantizeLinear configuration")
        n = self.name(m)
        w32 = self.p32(m, "weight")
        mval = float((w32.norm(p=1) / w32.nelement()).item())          # BwnQuantizer m (fp32)
        e = w32.mean()
        sg = torch.sign(w32 - e)
        if bool((sg == 0).any()):
            raise ValueError(f"{n}: exact-zero centred weight (w_zero not implemented)")
        if not torch.equal(wq, sg * mval):
            raise ValueError(f"{n}: captured weight is not sign(W - mean) * m")
        r = rows or slice(0, w32.shape[0])
        sgn = sg[r].numpy().astype(np.int8)
        mv = self.p(m.move, "bias")
        a_in = self.eps_clip(self.p32(m, "input_clip_val"))
        bias = self.p(m, "bias")[r]
        tg = tag or n
        wc = self.const(tg + "/w", sgn, dtype="bit")
        if m.symmetric:
            pm32 = _f32(np.float32(a_in) * np.float32(mval))
            mode = binarize or self.binarize
            self.precision.append({"layer": tg, "type": "QuantizeLinear(BWN, elastic signed 1-bit)->BDense", "shape": list(sgn.shape),
                                   "weights": "binary sign(W - mean W) * mean|W| (layerwise scalar)",
                                   "activations": f"binary sign(x + move) * alpha [{mode}]"})
            return self.node("BDense", [x, wc, self.const(tg + "/post_mul", np.full(sgn.shape[0], pm32), f64=np.full(sgn.shape[0], a_in * mval)),
                                        self.const(tg + "/post_add", bias), self.const(tg + "/thr", -mv)],
                             {"input_binarize": mode}, tag=tg)
        # unsigned 1-bit input (fc2): s = [round(fl(fl(clip(x) + move) / alpha)) clamped >= 1]
        lo_c, hi_c = fc2_input_clip
        a32, mv32 = np.float32(a_in), mv.astype(np.float32)

        def pred(xv):
            xv = np.asarray(xv, np.float32)
            c = np.clip(xv, np.float32(lo_c), np.float32(hi_c)).astype(np.float32)
            t = ((c + mv32).astype(np.float32) / a32).astype(np.float32)
            return np.clip(np.rint(t), 0, 1) >= 1
        o_true = min_true(pred, shape=mv32.shape)
        thr = np.where(o_true > _F32_INF_ORD, np.inf, ord_to_f32(np.maximum(o_true - 1, -_F32_INF_ORD)).astype(np.float64))
        thr = np.where(o_true <= -_F32_INF_ORD, -np.inf, thr)
        t64 = 0.5 * a_in - mv                                          # exact real threshold (inside the clip range)
        inside = (thr > lo_c) & (thr < hi_c)
        t64 = np.where(inside, t64, thr)
        self.notes.setdefault("fc2_thresholds", []).append({"layer": tg, "alpha": a_in, "n_channels": int(thr.size),
                                                            "min": float(thr.min()), "max": float(thr.max()),
                                                            "n_outside_clip": int((~inside).sum())})
        sw = sgn.astype(np.float64).sum(axis=1)
        pm32 = _f32(np.float32(np.float32(a_in) * np.float32(mval)) * np.float32(0.5))
        pa32 = (np.float32(pm32) * sw.astype(np.float32) + bias.astype(np.float32)).astype(np.float64)
        pm64, pa64 = a_in * mval / 2, a_in * mval / 2 * sw + bias
        self.precision.append({"layer": tg, "type": "QuantizeLinear(BWN, elastic unsigned 1-bit)->BDense(gt)", "shape": list(sgn.shape),
                               "weights": "binary sign(W - mean W) * mean|W| (layerwise scalar)",
                               "activations": "{0,1}*alpha: [clip(gelu)+move)/alpha rounds to 1] = [x > T_k] (exact float32 threshold)"})
        return self.node("BDense", [x, wc, self.const(tg + "/post_mul", np.full(sgn.shape[0], pm32), f64=np.full(sgn.shape[0], pm64)),
                                    self.const(tg + "/post_add", pa32, f64=pa64), self.const(tg + "/thr", thr, f64=t64)],
                         {"input_binarize": "gt"}, tag=tg)

    def layernorm_tf(self, x: str, ln, tag=None) -> str:
        if type(ln).__name__ != "Q_OVitLayerNorm":
            raise TypeError(f"{self.name(ln)}: expected Q_OVitLayerNorm")
        n = tag or self.name(ln)
        return self.node("LayerNorm", [x, self.const(n + "/gamma", self.p(ln, "weight")), self.const(n + "/beta", self.p(ln, "bias"))],
                         {"eps": float(ln.variance_epsilon)}, tag=n)

    def heads(self, x: str, n_tok: int, nh: int, dh: int, tag: str) -> str:
        t = self.reshape(x, [n_tok, nh, dh], tag=tag + "/split")
        return self.node("Transpose", [t], {"perm": [1, 0, 2]}, tag=tag + "/heads")

    def attention(self, xn: str, at, wq: dict, n_tok: int, c: int) -> str:
        if type(at).__name__ != "Q_Attention" or not at.quantize_attn or at.input_bits != 1 or at.input_quant_method != "elastic":
            raise TypeError(self.name(at))
        n = self.name(at)
        nh = int(at.num_heads)
        dh = c // nh
        if dh * nh != c or at.scale != dh ** -0.5 or 2.0 ** round(math.log2(at.scale)) != at.scale:
            raise ValueError(f"{n}: head geometry / scale")
        if tuple(at.move_att.bias.shape) != (nh, n_tok, n_tok):
            raise ValueError(f"{n}: move_att shape {tuple(at.move_att.bias.shape)} needs {n_tok} tokens")
        wqkv = wq[self.name(at.qkv)]
        qkv = {}
        for i, t in enumerate(("q", "k", "v")):
            rows = slice(i * c, (i + 1) * c)
            y = self.qlinear(xn, at.qkv, wqkv, rows=rows, tag=f"{n}.qkv/{t}")
            mv = self.p(getattr(at, f"move_{t}"), "bias").reshape(-1)   # [H,1,dh] -> channel h*dh + d
            qkv[t] = self.affine(y, c, bias=mv, tag=f"{n}.move_{t}")
        aq, ak = self.eps_clip(self.p32(at, "clip_query")), self.eps_clip(self.p32(at, "clip_key"))
        a1, a2, a3 = (self.eps_clip(self.p32(at, k)) for k in ("clip_value", "clip_value2", "clip_value3"))
        al = [self.eps_clip(self.p32(at, k)) for k in ("clip_attn", "clip_attn2", "clip_attn3")]
        self.notes.setdefault("attention_scales", []).append({"module": n, "alpha_q": aq, "alpha_k": ak, "v": [a1, a2, a3], "attn": al})
        qh, kh, vh = (self.heads(qkv[t], n_tok, nh, dh, f"{n}/{t}") for t in ("q", "k", "v"))
        # ---- logits = sign(q') sign(k')^T * alpha_q * alpha_k / sqrt(dh)
        cnt = self.node("BMatMul", [qh, kh], {"a_binarize": "sign_zero", "b_binarize": "sign_zero", "transpose_b": 1, "scale": 1.0}, tag=n + "/qk")
        s32 = _f32(np.float32(np.float32(aq) * np.float32(ak)) * np.float32(at.scale))
        logits = self.affine(cnt, n_tok, scale=s32, scale64=aq * ak * float(at.scale), tag=n + "/qk_scale")
        p = self.node("Softmax", [logits], tag=n + "/softmax")
        p = self.node("Add", [p, self.const(n + ".move_att", self.p(at.move_att, "bias"))], tag=n + "/move_att")
        # ---- attention operands H0, H1, H2 ({0,1})
        a32 = np.float32(al[0])
        o_true = int(min_true(lambda xv: np.clip(np.rint((np.asarray(xv, np.float32) / a32).astype(np.float32)), 0, 1) >= 1))
        t0 = float(ord_to_f32(o_true - 1))
        self.notes.setdefault("attn_round_thresholds", []).append({"module": n, "alpha": al[0], "T": t0, "T_f64": 0.5 * al[0]})
        mp_ = self.gmax(p, nh * n_tok, n_tok, tag=n + "/attn_max")
        u = [self.node("AddBroadcast", [p, self.const(n + "/T0", np.array([-t0]), f64=np.array([-0.5 * al[0]]))], tag=n + "/H0"),
             self.node("AddBroadcast", [p, self.scaled_scalar(mp_, -0.7, n + "/-0.7max")], tag=n + "/H1"),
             self.node("AddBroadcast", [p, self.scaled_scalar(mp_, -0.9, n + "/-0.9max")], tag=n + "/H2")]
        # ---- V operands
        vneg = self.affine(qkv["v"], c, scale=-1.0, tag=n + "/-v")
        mv_ = self.gmax(qkv["v"], n_tok, c, tag=n + "/v_max")
        mvn = self.gmax(vneg, n_tok, c, tag=n + "/v_negmin")
        vnh = self.heads(vneg, n_tok, nh, dh, f"{n}/-v")
        s_ops = {1: [(vh, "sign_zero", 1.0)]}
        for j, f in ((2, 0.7), (3, 0.9)):
            pos = self.node("AddBroadcast", [vh, self.scaled_scalar(mv_, -f, f"{n}/-{f}vmax")], tag=f"{n}/S{j}+")
            neg = self.node("AddBroadcast", [vnh, self.scaled_scalar(mvn, -f, f"{n}/-{f}vnegmin")], tag=f"{n}/S{j}-")
            s_ops[j] = [(pos, "gt0", 1.0), (neg, "gt0", -1.0)]
        av = [a1, a2, a3]
        acc = None
        for i in range(3):
            for j in (1, 2, 3):
                parts = [self.node("BMatMul", [u[i], op], {"a_binarize": "gt0", "b_binarize": md, "transpose_b": 0, "scale": sgn},
                                   tag=f"{n}/H{i}S{j}") for op, md, sgn in s_ops[j]]
                cnt = parts[0] if len(parts) == 1 else self.node("Add", parts, tag=f"{n}/H{i}S{j}/diff")
                coef32 = _f32(np.float32(al[i]) * np.float32(av[j - 1]))
                term = self.affine(cnt, dh, scale=coef32, scale64=al[i] * av[j - 1], tag=f"{n}/H{i}S{j}/scale")
                acc = term if acc is None else self.node("Add", [acc, term], tag=f"{n}/av_sum")
        self.precision.append({"layer": n + "/attention", "type": "BMatMul(sign x sign) -> Softmax -> +move_att -> 3 {0,1} masks x "
                               "(sign(V), V 0.7/0.9 masks) -> 15 BMatMul counts", "weights": "-",
                               "activations": "Q, K binary; V = 3 binary bases; attention = 3 binary bases (GSB)"})
        ctx = self.node("Transpose", [acc], {"perm": [1, 0, 2]}, tag=n + "/merge")
        ctx = self.reshape(ctx, [n_tok, c], tag=n + "/merge")
        return self.qlinear(ctx, at.proj, wq[self.name(at.proj)])

    def mlp(self, xn: str, m, wq: dict) -> str:
        if type(m).__name__ != "Q_Mlp" or getattr(m.act, "__name__", "") != "gelu":
            raise TypeError(self.name(m))
        y = self.qlinear(xn, m.fc1, wq[self.name(m.fc1)])
        y = self.node("Activation", [y], {"type": "gelu"}, tag=self.name(m) + "/gelu")
        self.precision.append({"layer": self.name(m) + "/gelu", "type": "gelu (erf) -> Activation gelu (double); clip(-10,10) folded into fc2 thresholds",
                               "weights": "-", "activations": "fp32"})
        return self.qlinear(y, m.fc2, wq[self.name(m.fc2)], fc2_input_clip=(-10.0, 10.0))

    def head8(self, x: str, m, wq) -> str:
        """QuantizeLinear 8-bit weights / 8-bit elastic input -> ChannelAffine(move) -> FakeQuant -> Dense."""
        if type(m).__name__ != "QuantizeLinear" or m.weight_bits != 8 or m.input_bits != 8 or m.input_quant_method != "elastic" \
                or m.weight_quant_method != "elastic" or not m.symmetric:
            raise TypeError(self.name(m))
        n = self.name(m)
        self.mark(m, "weight")
        self.mark(m, "weight_clip_val")
        c = int(m.in_features)
        a_in = self.eps_clip(self.p32(m, "input_clip_val"))
        y = self.affine(x, c, bias=self.p(m.move, "bias"), tag=n + ".move")
        y = self.node("FakeQuant", [y, self.const(n + "/in_scale", np.full(c, a_in)), self.const(n + "/in_zp", np.zeros(c))],
                      {"axis": -1, "pre_scale": 1.0, "qmin": -128, "qmax": 127}, tag=n + "/input_q")
        self.precision.append({"layer": n, "type": "QuantizeLinear(8-bit elastic weights and input)->FakeQuant+Dense",
                               "weights": "8-bit levels * alpha_w (fp32 tensor)", "activations": "8-bit levels * alpha_in", "shape": list(wq.shape)})
        return self.node("Dense", [y, self.const(n + "/w", wq.double().numpy()), self.const(n + "/b", self.p(m, "bias"))], tag=n)


def convert(model, model_id: str, binarize: str = "ge"):
    """IR for the official lowbit_VisionTransformer (eval). binarize applies to the sign(x + move) BDense inputs
    (qkv, proj, fc1): 'ge' (deployable, tie -> +1) or 'sign_zero' (exact torch.sign)."""
    if binarize not in ("ge", "sign_zero"):
        raise ValueError(binarize)
    if type(model).__name__ != "lowbit_VisionTransformer" or model.training:
        raise TypeError("expected lowbit_VisionTransformer in eval mode")
    wq = capture_quantized_weights(model)
    b = _GSB(model, binarize)
    g = b.g
    pe = model.patch_embed
    proj = pe.proj
    if type(pe).__name__ != "Q_PatchEmbed" or type(proj).__name__ != "QuantizeConv2dQ" or proj.groups != 1 or tuple(proj.padding) != (0, 0) \
            or tuple(proj.stride) != tuple(proj.kernel_size) or proj.bias is None or proj.nbits != 8:
        raise ValueError("unexpected patch embedding")
    b.mark(proj, "weight")
    b.mark(proj, "alpha")
    b.mark(proj, "init_state")
    hh, ww = pe.img_size
    x = g.add_input("input", [1, hh, ww, 3])
    c = int(model.embed_dim)
    gh, gw = hh // proj.stride[0], ww // proj.stride[1]
    wpe = wq[b.name(proj)].double().numpy()
    y = b.node("Conv2D", [x, b.const("patch_embed.proj/w", wpe.transpose(0, 2, 3, 1), layout="OHWI"),
                          b.const("patch_embed.proj/b", b.p(proj, "bias"))],
               {"strides": [int(s) for s in proj.stride], "pads": [0, 0, 0, 0]}, tag="patch_embed.proj")
    b.precision.append({"layer": "patch_embed.proj", "type": "QuantizeConv2dQ(8-bit LSQ weights)->Conv2D", "weights": "8-bit levels * alpha_c (fp32 tensor)",
                        "activations": "fp32", "shape": list(proj.weight.shape)})
    y = b.reshape(y, [gh * gw, c], tag="patch_embed/flatten")
    tok = b.node("Concat", [b.const("cls_token", b.p(model, "cls_token").reshape(1, c)),
                            b.const("dist_token", b.p(model, "dist_token").reshape(1, c)), y], {"axis": 0}, tag="tokens")
    n_tok = gh * gw + 2
    y = b.node("Add", [tok, b.const("pos_embed", b.p(model, "pos_embed").reshape(n_tok, c))], tag="pos_embed")
    for blk in model.blocks:
        if type(blk).__name__ != "Q_Block":
            raise TypeError(b.name(blk))
        a = b.attention(b.layernorm_tf(y, blk.norm1), blk.attn, wq, n_tok, c)
        y = b.node("Add", [y, a], tag=b.name(blk) + "/residual1")
        mo = b.mlp(b.layernorm_tf(y, blk.norm2), blk.mlp, wq)
        y = b.node("Add", [y, mo], tag=b.name(blk) + "/residual2")
    outs = []
    for i, hd in ((0, model.head), (1, model.head_dist)):
        r = b.node("Gather", [y, b.const(f"token{i}", np.array([i], np.int32), dtype="i32")], tag=f"token{i}")
        r = b.layernorm_tf(r, model.norm, tag=f"norm/token{i}")
        outs.append(b.head8(r, hd, wq[b.name(hd)]))
    logits = b.affine(b.node("Add", outs, tag="heads/sum"), int(model.num_classes), scale=0.5, tag="heads/mean")
    g.set_outputs(logits)
    unused = sorted(set(b.sd) - b.used)
    if unused:
        raise ValueError(f"{len(unused)} checkpoint tensors not consumed by the mapping: {unused[:10]}")
    info = {"layer_precision": b.precision, "float64_constants": b.f64, "notes": b.notes,
            "state_dict_unused_by_source": sorted(b.unused_by_source), "n_state_dict_tensors": len(b.sd),
            "binarize_note": "attention, Q/K/V and fc2 binarizations are exact in every variant; binarize applies to qkv/proj/fc1 inputs"}
    return g, info
