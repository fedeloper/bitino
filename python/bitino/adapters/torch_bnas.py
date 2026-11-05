"""Adapter: official BNAS cell networks (yonseivnl/bnas, "Learning Architectures for Binary Networks", ECCV 2020)
-> Bitino IR. NetworkCIFAR (src/model.py, BNAS-A) and NetworkImageNet (src/model_imagenet.py, BNAS-D/E/F/G/H).

Source semantics (read only; commit 675ecc7f846c162329c8c4c4ed8a30c423df0363):
  * src/operations.py
      - BinReLUConvBN.forward (l.40-44) / BinDilConv.forward (l.73-76): ``x = bn(x); x, _ = BinActive()(x);
        return relu(conv(x))``. BinActive.forward (l.19-24) returns ``input.sign()`` (torch.sign: sign(0) = 0) and a
        channel mean that the callers discard. The conv zero-pads the *binarized* input (pad value 0), has no bias,
        dilation 2 for bin_dil_conv_*.
      - ReLUConvBN (l.47-59, CIFAR stem): ``relu(conv(bn(x)))`` without binarization.
      - Zero (l.88-97): ``x[:, :, ::s, ::s].mul(0.)``; AvgPool2d(3, s, 1, count_include_pad=False); MaxPool2d(3, s, 1).
  * src/bin_utils.py BinOp: before inference (test.py / test_imagenet.py ``infer``: ``bin_op.binarization()``) the
    stored latent weight of every *selected* nn.Conv2d is replaced by ``sign(w') * mean|w'|`` per output channel, with
    ``w' = clamp(w - mean_over_input_channels(w), -1, 1)`` (l.38-61). Selection (l.6-26): modules are counted in
    ``named_modules()`` order over ALL nn.Conv2d (``index``), but ``bin_range`` only spans ``1..#convs whose name has
    no 'res'``; a conv is selected iff ``index in bin_range and 'res' not in name and index >= num_skip + 1``. The
    preprocess_res convs advance ``index`` without being counted in ``bin_range``, so the LAST (number of preprocess_res
    convs) non-res convs are NOT weight-binarized (the "partial binarization quirk"): they keep their fp32 latent
    weights, while their inputs are still sign()-binarized by BinReLUConvBN/BinDilConv.
  * BinOp version: the released checkpoints were trained/evaluated with the first-release BinOp (commit cdb58f7), whose
    'res' exclusion is void (stale loop variable, see binop_targets): the preprocess_res convs are weight-binarized as
    well. Their input is the real-valued BN output (no sign), so they map to Conv2D with the binarized weight values.
    The adapter follows whatever target list the converter passes (official BinOp of the given source tree).
  * src/model.py / model_imagenet.py Cell.forward: ``states = [pre0(s0), pre1(s1)]``; step i adds
    ``op(states[idx])`` pairs; output ``cat(states[concat])`` plus (layer > 0) ``res1`` (normal cell) or
    ``preprocess_res(res1)`` = BN + Conv2d(2x2, s2, bias) (reduction cell). Drop-path is training-only.

Mapping (existing ops only):
  * the BN in front of every sign(): ``sign(s*x + b) = sign(s) * sign(x + b/s)``: ``ChannelAffine(bias b/s)`` (float64,
    one f32 rounding; ``fl(x + c)`` has the exact sign of ``x + c``), the per-channel ``sign(s)`` is moved into the
    weights of the following conv (bit flip / exact negation). This avoids the near-ties of a two-rounding
    ``fl(fl(s*x) + b)`` (BNAS-D has a channel where a recurring input value lands within one ulp of the threshold).
  * binarized BinReLUConvBN / BinDilConv: ``ChannelAffine`` (above) ->
    ``BConv2D(threshold none, pad_value 0, dilations, post_mul = mean|w'| per output channel (the exact fp32 value the
    source multiplies by), activation relu)``; weight bits = sign of the BinOp-binarized weight (|w| is checked to be
    exactly ``mean|w'|`` on every tap, no zero weight). ``sign_zero`` = exact source semantics (BN output exactly 0
    contributes 0); ``ge`` = deployable (0 -> +1).
  * quirk convs (not BinOp-selected, sign()-binarized input): ``ChannelAffine`` -> ``Activation(sign_zero)`` ->
    ``Conv2D(fp32 latent weights, pad_value 0, relu)`` in both variants (exact: the activation is the same {-1,0,+1}
    tensor, zero padded after binarization; the float sign_zero kernels are bit-exact).
  * structurally zero channels: channels that are exactly 0 for every input (all-zero states of layer-0 / normal
    cells, carried along the residual chain, and kept exactly 0 by a BN whose folded shift is 0) are tracked; where
    they reach a binarized conv, ``sign_zero`` is exact, and the deployable ``ge`` variant (0 -> +1) gets the exact,
    data-independent correction ``BConv2D(act none) + Add(-m[o] * sum_{zero channels, in-bounds taps} w) + relu``.
  * Zero ops: ``h + Zero(x) = h`` (the edge is dropped); a state that is the sum of two Zero ops is all zeros: if it is
    concatenated, it is materialized as ``ChannelAffine(any state of the cell, scale 0)`` (the source's ``x.mul(0.)``
    for finite x: +-0); dead branches (e.g. the reduction-cell preprocess1 of BNAS-A, which only feeds Zero ops) are
    not emitted.
  * pools: MaxPool2D (out-of-bounds taps ignored = torch -inf padding), AvgPool2D(count_include_pad 0); identical
    pools of the same state are shared.
  * concat -> Concat (channel axis), residual -> Add, preprocess_res -> ChannelAffine + Conv2D(2x2 s2, bias), stems ->
    ChannelAffine/Conv2D/Activation, AdaptiveAvgPool2d(1) / AvgPool2d(7) on the 7x7 map -> GlobalAvgPool, Linear ->
    Dense (optionally with a constant row permutation ``class_perm``: output c = source logit class_perm[c]).
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

BINARIZE_MODES = ("sign_zero", "ge")
BIN_OPS = ("BinReLUConvBN", "BinDilConv")


def _cls(m) -> str:
    return type(m).__name__


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)


def _pair(v) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _f64(t) -> np.ndarray:
    import torch
    return t.detach().cpu().to(torch.float64).reshape(-1).numpy()


def binop_targets(model, num_skip: int, first_release: bool = False) -> list[str]:
    """Names of the nn.Conv2d modules bin_utils.BinOp selects (re-implementation of src/bin_utils.py:6-26 for a
    cross-check; the converter compares it with the official BinOp's target list).

    ``first_release``: the BinOp of the first release (commit cdb58f7, 2020-08-12; the code the released checkpoints
    were trained and evaluated with) iterates ``for m in model.modules()`` in its selection loop while testing
    ``'res' not in n``, where ``n`` is the stale loop variable of the counting loop, i.e. the name of the model's LAST
    module ('classifier' / 'dropout'). The 'res' exclusion is therefore void: the preprocess_res convs are binarized
    too, and still only ``index <= #non-res convs`` qualifies. Commit 675ecc7 (2020-11-15, "typo fixes") changed the
    loop to ``named_modules()``, which makes the preprocess_res convs float."""
    import torch.nn as nn

    names = [n for n, _ in model.named_modules()]
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    count_nonres = sum(1 for n, _ in convs if "res" not in n)
    bin_range = set(range(1, count_nonres + 1))
    stale = names[-1]
    return [n for index, (n, _) in enumerate(convs, start=1)
            if index in bin_range and "res" not in (stale if first_release else n) and index >= num_skip + 1]


def xnor_binarize(w):
    """BinOp.binarization() of one latent weight in its dtype (meancenter, clamp, sign * mean|w'|), as the source."""
    s = w.size()
    w = w.add(w.mean(1, keepdim=True).mul(-1).expand_as(w))
    w = w.clamp(-1.0, 1.0)
    n = w[0].nelement()
    m = w.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
    return w.sign().mul(m.expand(s))


def convert(model, binarized: set[str] | list[str], model_id: str = "", input_name: str = "input",
            input_hw: tuple[int, int] | None = None, binarize: str = "sign_zero",
            class_perm: np.ndarray | None = None) -> tuple[Graph, dict]:
    """``model``: the official NetworkCIFAR / NetworkImageNet in eval mode AFTER ``BinOp.binarization()`` (the selected
    convs hold ``sign(w')*mean|w'|``); ``binarized``: names of those convs (BinOp target modules). Node names are the
    PyTorch module paths."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}")
    _expect(not model.training, "model must be in eval mode")
    kind = _cls(model)
    _expect(kind in ("NetworkCIFAR", "NetworkImageNet"), f"unsupported top-level model {kind}")
    if input_hw is None:
        input_hw = (32, 32) if kind == "NetworkCIFAR" else (224, 224)
    binarized = set(binarized)
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    visited: set[str] = set()
    bin_points: list[str] = []          # module paths of the sign() binarizations, in emission order
    weight_bits: dict[str, np.ndarray] = {}
    float_quirk: list[str] = []
    res_binarized: list[str] = []
    bn_flips: dict[str, np.ndarray] = {}
    zmask: dict[str, np.ndarray] = {}   # tensor -> channels that are structurally exactly zero (from Zero ops)
    zero_corr: list[dict] = []
    in_hw: dict[str, tuple[int, int]] = {}

    def zm(t: str, c: int) -> np.ndarray:
        return zmask.get(t, np.zeros(c, bool))

    def set_zm(t: str, mask: np.ndarray) -> str:
        if mask.any():
            zmask[t] = mask
        return t

    def bin_input_hw(path: str) -> tuple[int, int]:
        """Spatial size of a binarization's input, from one float32 forward of the source with hooks (only needed for
        the zero-channel correction)."""
        if not in_hw:
            import torch
            mods = {n: mm for n, mm in model.named_modules() if _cls(mm) in BIN_OPS}
            hs = [mm.bn.register_forward_hook(lambda _m, i, _o, n=n: in_hw.__setitem__(n, tuple(i[0].shape[2:]))) for n, mm in mods.items()]
            with torch.no_grad():
                model(torch.zeros(1, 3, *_pair(input_hw), dtype=next(model.parameters()).dtype))
            for h in hs:
                h.remove()
        return in_hw[path]

    def folded(name: str, value64) -> str:
        v = np.asarray(value64, np.float64)
        exact[name] = v
        return g.add_constant(name, v.astype(np.float32))

    def bn(m, x: str, name: str) -> str:
        _expect(_cls(m) == "BatchNorm2d" and m.running_mean is not None and m.track_running_stats, f"{name}: expected BatchNorm2d with running stats")
        inv = 1.0 / np.sqrt(_f64(m.running_var) + float(m.eps))
        scale = _f64(m.weight) * inv if m.affine else inv
        shift = (_f64(m.bias) if m.affine else 0.0) - _f64(m.running_mean) * scale
        visited.add(name)
        y = g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", shift)], name=name)
        # x = +-0 on a structurally zero channel: x*scale + shift is exactly 0 iff shift == 0 (f32 and f64 constants)
        return set_zm(y, zm(x, len(scale)) & (np.broadcast_to(shift, scale.shape) == 0) & np.isfinite(scale))

    def bn_sign(m, x: str, name: str) -> tuple[str, np.ndarray]:
        """BN whose output only feeds sign(): sign(s*x + b) = sign(s) * sign(x + b/s) for s != 0. Emits the bias-only
        ChannelAffine(x + b/s) (b/s in float64, one f32 rounding of the threshold; fl(x + c) has the exact sign of
        x + c, so no double-rounding near-ties of fl(fl(s*x) + b)), and returns the per-channel sign flip, which the
        caller moves into the following conv's weights. Channels with s == 0 are not supported (none in the
        checkpoints)."""
        _expect(_cls(m) == "BatchNorm2d" and m.running_mean is not None and m.track_running_stats, f"{name}: expected BatchNorm2d with running stats")
        inv = 1.0 / np.sqrt(_f64(m.running_var) + float(m.eps))
        scale = _f64(m.weight) * inv if m.affine else inv
        _expect(bool(np.all(scale != 0)) and bool(np.all(np.isfinite(scale))), f"{name}: BN scale 0 / non-finite (sign folding impossible)")
        beta = _f64(m.bias) if m.affine else np.zeros_like(scale)
        bias = beta / scale - _f64(m.running_mean)          # (s*x + beta - s*mean) / s
        visited.add(name)
        y = g.add_node("ChannelAffine", [x, "", folded(f"{name}/bias_over_scale", bias)], name=name)
        bn_flips[name] = scale < 0
        set_zm(y, zm(x, len(scale)) & (bias == 0))      # +-0 + 0 = 0: still structurally zero
        return y, scale < 0

    def conv_attrs(m, act: str) -> dict:
        (sh, sw), (ph, pw), (dh, dw) = _pair(m.stride), _pair(m.padding), _pair(m.dilation)
        _expect(m.groups == 1 and m.padding_mode == "zeros", f"unexpected Conv2d {m}")
        return {"strides": [sh, sw], "dilations": [dh, dw], "pads": [ph, ph, pw, pw], "groups": 1, "pad_value": 0.0, "activation": act}

    def fconv(m, x: str, name: str, act: str = "none", allow_binarized: bool = False, in_flip=None) -> str:
        _expect(_cls(m) == "Conv2d", f"{name}: expected Conv2d, got {_cls(m)}")
        _expect(allow_binarized or name not in binarized, f"{name}: a float conv is in the BinOp target list")
        visited.add(name)
        w = m.weight.detach().cpu().numpy().astype(np.float32).copy()
        if in_flip is not None:
            w[:, in_flip] = -w[:, in_flip]                    # exact negation
        w = np.ascontiguousarray(w.transpose(0, 2, 3, 1))
        ins = [x, g.add_constant(f"{name}/kernel", w, layout="OHWI"),
               g.add_constant(f"{name}/bias", m.bias.detach().cpu().numpy().astype(np.float32)) if m.bias is not None else ""]
        return g.add_node("Conv2D", ins, conv_attrs(m, act), name=name)

    def relu(x: str, name: str) -> str:
        return g.add_node("Activation", [x], {"type": "relu"}, name=name)

    def bin_op(m, x: str, path: str) -> str:
        """BinReLUConvBN / BinDilConv: bn -> sign -> conv (zero pad) -> relu."""
        _expect(_cls(m) in BIN_OPS and len(m.op) == 2 and _cls(m.op[0]) == "Conv2d" and _cls(m.op[1]) == "ReLU",
                f"{path}: unexpected {_cls(m)} layout")
        c = m.op[0]
        _expect(c.bias is None, f"{path}.op.0: unexpected conv bias")
        y, flip = bn_sign(m.bn, x, f"{path}.bn")
        bin_points.append(path)
        cname = f"{path}.op.0"
        kh, kw = _pair(c.kernel_size)
        if cname in binarized:
            w = c.weight.detach().cpu()
            o = w.shape[0]
            a = w.abs().reshape(o, -1)
            scale = a[:, 0].clone()
            _expect(bool((a == scale[:, None]).all()), f"{cname}: weight is not sign(w)*mean|w| (BinOp not applied?)")
            _expect(bool((scale > 0).all()), f"{cname}: zero XNOR scale / zero weights")
            bits = np.where(w.numpy() > 0, 1, -1).astype(np.int8)
            weight_bits[cname] = bits.copy()
            bits[:, flip] *= -1                               # sign(s) < 0 moved from the activation into the weights
            visited.add(cname)
            zin = zm(y, c.in_channels)
            correct = binarize == "ge" and bool(zin.any())
            attrs = conv_attrs(c, "none" if correct else "relu")
            attrs.update({"input_binarize": binarize, "output": "f32"})
            out = g.add_node("BConv2D", [y, g.add_constant(f"{cname}/kernel_bits", np.ascontiguousarray(bits.transpose(0, 2, 3, 1)),
                                                           dtype="bit", layout="OHWI_bit"),
                                         folded(f"{cname}/post_mul", scale.numpy().astype(np.float64)), "", ""], attrs,
                             name=f"{path}/bconv" if correct else path)
            if correct:
                # ge maps the structurally zero inputs (source: sign(0) = 0) to +1: subtract their exact, data-independent
                # contribution m[o] * sum_{zero channels, in-bounds taps} w (float64 conv of the channel indicator)
                import torch
                import torch.nn.functional as F
                hh, ww = bin_input_hw(path)
                ind = torch.zeros(1, c.in_channels, hh, ww, dtype=torch.float64)
                ind[0, torch.from_numpy(zin)] = 1.0
                corr = F.conv2d(ind, torch.from_numpy(bits.astype(np.float64)), stride=c.stride, padding=c.padding,
                                dilation=c.dilation)[0].permute(1, 2, 0).numpy()
                cmap = -scale.numpy().astype(np.float64) * corr
                out = g.add_node("Add", [out, folded(f"{cname}/zero_channel_correction", cmap[None])], name=f"{path}/zero_channel_correction")
                out = relu(out, path)
                zero_corr.append({"layer": path, "zero_input_channels": int(zin.sum()), "of": int(c.in_channels)})
            precision.append({"layer": path, "op": "ChannelAffine(bias b/s) + BConv2D(relu)", "source": f"{_cls(m)} (BN, sign, Conv2d, ReLU)",
                              "weights": "binary sign(clamp(w - mean_c w)) x mean|w'| per output channel (BinOp)",
                              "activations": "binary sign(BN(x)) (torch.sign, sign(0)=0) = sign(s) sign(x + b/s), sign(s) folded "
                                             "into the weight bits", "input_binarize": binarize, "n_negative_bn_scale": int(flip.sum()),
                              "kernel": [kh, kw], "stride": _pair(c.stride)[0], "dilation": _pair(c.dilation)[0], "pad_value": 0.0})
            diag.append({"layer": cname, "n_weights": int(bits.size), "n_plus": int(np.sum(bits > 0)),
                         "xnor_scale_min": float(scale.min()), "xnor_scale_max": float(scale.max())})
        else:
            float_quirk.append(cname)
            # float Activation(sign_zero) is supported by every float kernel (bit-exact): exact torch.sign in both variants
            s = g.add_node("Activation", [y], {"type": "sign_zero"}, name=f"{path}/sign")
            out = fconv(c, s, path, act="relu", in_flip=flip)
            precision.append({"layer": path, "op": "ChannelAffine + Activation(sign) + Conv2D(relu)",
                              "source": f"{_cls(m)} whose conv BinOp does not binarize (counting quirk)",
                              "weights": "fp32 latent weights (not binarized at eval by the official code)",
                              "activations": "binary sign(BN(x)) as float {-1,0,+1} (Activation sign_zero in both variants)",
                              "kernel": [kh, kw], "stride": _pair(c.stride)[0], "dilation": _pair(c.dilation)[0]})
        visited.add(path)
        return out

    def pool(m, x: str, name: str, cache: dict) -> str:
        key = (_cls(m), x, _pair(m.kernel_size), _pair(m.stride), _pair(m.padding), getattr(m, "count_include_pad", None))
        if key in cache:
            return cache[key]
        k, s, p = _pair(m.kernel_size), _pair(m.stride), _pair(m.padding)
        _expect(not m.ceil_mode, f"{name}: ceil_mode")
        if _cls(m) == "MaxPool2d":
            _expect(_pair(m.dilation) == (1, 1), f"{name}: dilated max pool")
            y = g.add_node("MaxPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [p[0], p[0], p[1], p[1]]}, name=name)
        else:
            _expect(m.divisor_override is None, f"{name}: divisor_override")
            y = g.add_node("AvgPool2D", [x], {"kernel": list(k), "strides": list(s), "pads": [p[0], p[0], p[1], p[1]],
                                              "count_include_pad": int(bool(m.count_include_pad))}, name=name)
        cache[key] = y
        if x in zmask:
            set_zm(y, zmask[x].copy())
        return y

    ZERO = None   # marker: an all-zero state (shape implied by the cell)

    def cell_graph(cell, ci: int, s0: str, s1: str, h: int, w: int) -> tuple[str, int, int, int]:
        pth = f"cells.{ci}"
        red = bool(cell.reduction)
        red_names = red and hasattr(cell, "preprocess0_red")
        p0n, p1n = ("preprocess0_red", "preprocess1_red") if red_names else ("preprocess0", "preprocess1")
        p0, p1 = getattr(cell, p0n), getattr(cell, p1n)
        c = p1.op[0].out_channels
        memo: dict[int, str | None] = {}
        cache: dict = {}

        def state(i: int):
            if i in memo:
                return memo[i]
            if i == 0:
                v = bin_op(p0, s0, f"{pth}.{p0n}")
            elif i == 1:
                v = bin_op(p1, s1, f"{pth}.{p1n}")
            else:
                k = i - 2
                hs = [apply(2 * k + j) for j in (0, 1)]
                nz = [v for v in hs if v is not ZERO]
                if not nz:
                    v = ZERO
                elif len(nz) == 1:
                    v = nz[0]
                else:
                    v = set_zm(g.add_node("Add", nz, name=f"{pth}.step{k}"), zm(nz[0], c) & zm(nz[1], c))
            memo[i] = v
            return v

        def apply(j: int):
            op = cell._ops[j]
            idx = cell._indices[j]
            name = f"{pth}._ops.{j}"
            if _cls(op) == "Zero":
                visited.add(name)
                return ZERO
            x = state(idx)
            _expect(x is not ZERO, f"{name}: op applied to an all-zero state")
            if _cls(op) in BIN_OPS:
                return bin_op(op, x, name)
            if _cls(op) in ("MaxPool2d", "AvgPool2d"):
                visited.add(name)
                return pool(op, x, name, cache)
            if _cls(op) == "Identity":
                return x
            raise NotImplementedError(f"{name}: unsupported op {_cls(op)}")

        outs = [state(i) for i in cell._concat]
        ref = next((v for v in outs if v is not ZERO), None)
        _expect(ref is not None, f"{pth}: every concatenated state is zero")
        if any(v is ZERO for v in outs):
            z = set_zm(g.add_node("ChannelAffine", [ref, g.add_constant(f"{pth}/zero_scale", np.zeros(c, np.float32)), ""], name=f"{pth}/zeros"),
                       np.ones(c, bool))
            outs = [z if v is ZERO else v for v in outs]
            precision.append({"layer": f"{pth}/zeros", "op": "ChannelAffine(scale 0)", "source": "state = Zero(.) + Zero(.) (x.mul(0.))",
                              "weights": "none", "activations": "fp32 zeros"})
        y = set_zm(g.add_node("Concat", outs, {"axis": -1}, name=f"{pth}/concat"), np.concatenate([zm(v, c) for v in outs]))
        st = 2 if red else 1
        ho, wo = (h - 1) // st + 1, (w - 1) // st + 1
        if cell.layer != 0:
            if red:
                pr = cell.preprocess_res
                _expect(_cls(pr) == "Sequential" and len(pr) == 2 and _cls(pr[1]) == "Conv2d", f"{pth}.preprocess_res: unexpected layout")
                r = bn(pr[0], s1, f"{pth}.preprocess_res.0")
                rname = f"{pth}.preprocess_res.1"
                if rname in binarized:   # first-release BinOp: binary weights x real-valued BN output (no sign) + bias
                    w = pr[1].weight.detach().cpu()
                    a = w.abs().reshape(w.shape[0], -1)
                    _expect(bool((a == a[:, :1]).all()) and bool((a[:, 0] > 0).all()), f"{rname}: weight is not sign(w)*mean|w|")
                    res_binarized.append(rname)
                    precision.append({"layer": rname, "op": "ChannelAffine + Conv2D", "source": "preprocess_res BN + Conv2d(2x2, s2, bias), "
                                      "weight-binarized by the first-release BinOp", "weights": "binary sign(w')*mean|w'| per output channel "
                                      "(stored as the exact fp32 values)", "activations": "fp32 (not binarized in the source)"})
                r = fconv(pr[1], r, rname, allow_binarized=True)
                visited.add(f"{pth}.preprocess_res")
            else:
                r = s1
            y = set_zm(g.add_node("Add", [y, r], name=f"{pth}/residual"), zm(y, c * len(outs)) & zm(r, c * len(outs)))
        return y, ho, wo, c * len(outs)

    hw = _pair(input_hw)
    x = g.add_input(input_name, [1, hw[0], hw[1], 3])
    if kind == "NetworkCIFAR":
        st = model.stem
        _expect(_cls(st) == "ReLUConvBN", "stem: expected ReLUConvBN")
        y = bn(st.bn, x, "stem.bn")
        y = fconv(st.op[0], y, "stem.op.0", act="relu")
        visited.add("stem")
        precision.append({"layer": "stem", "op": "ChannelAffine + Conv2D(relu)", "source": "ReLUConvBN: BN, Conv2d 3x3, ReLU",
                          "weights": "fp32", "activations": "fp32"})
        s0 = s1 = y
        h, w = hw
    else:
        st0, st1 = model.stem0, model.stem1
        _expect([_cls(m) for m in st0] == ["Conv2d", "BatchNorm2d", "ReLU", "Conv2d", "BatchNorm2d"]
                and [_cls(m) for m in st1] == ["ReLU", "Conv2d", "BatchNorm2d"], "stem: unexpected layout")
        y = fconv(st0[0], x, "stem0.0")
        y = relu(bn(st0[1], y, "stem0.1"), "stem0.2")
        y = fconv(st0[3], y, "stem0.3")
        s0 = bn(st0[4], y, "stem0.4")
        y = relu(s0, "stem1.0")
        y = fconv(st1[1], y, "stem1.1")
        s1 = bn(st1[2], y, "stem1.2")
        precision.append({"layer": "stem0/stem1", "op": "Conv2D + ChannelAffine (+ relu) x3", "source": "3 FP convs (num_skip = 3)",
                          "weights": "fp32", "activations": "fp32"})
        h, w = (hw[0] + 7) // 8, (hw[1] + 7) // 8      # stem1 output (s1); s0 is 2x larger, cell 0 reduces it (stride 2)
    for ci, cell in enumerate(model.cells):
        _expect(_cls(cell) == "Cell", f"cells.{ci}: unexpected {_cls(cell)}")
        y, h, w, _c = cell_graph(cell, ci, s0, s1, h, w)
        s0, s1 = s1, y
    gp = model.global_pooling
    if _cls(gp) == "AdaptiveAvgPool2d":
        _expect(_pair(gp.output_size) == (1, 1), "global_pooling: output size != 1")
    else:
        _expect(_cls(gp) == "AvgPool2d" and _pair(gp.kernel_size) == (h, w) and _pair(gp.stride) == (h, w) and _pair(gp.padding) == (0, 0),
                f"global_pooling: AvgPool2d must cover the full {h}x{w} map")
    y = g.add_node("GlobalAvgPool", [s1], {"keepdims": 0}, name="global_pooling")
    fc = model.classifier
    _expect(_cls(fc) == "Linear" and fc.bias is not None, "classifier: unexpected layer")
    wfc = fc.weight.detach().cpu().numpy().astype(np.float32)
    bfc = fc.bias.detach().cpu().numpy().astype(np.float32)
    if class_perm is not None:
        class_perm = np.asarray(class_perm, np.int64)
        _expect(sorted(class_perm.tolist()) == list(range(wfc.shape[0])), "class_perm is not a permutation")
        wfc, bfc = wfc[class_perm], bfc[class_perm]
    visited.update({"classifier", "global_pooling"})
    y = g.add_node("Dense", [y, g.add_constant("classifier/kernel", np.ascontiguousarray(wfc)), g.add_constant("classifier/bias", bfc)],
                   {"activation": "none"}, name="classifier")
    precision.append({"layer": "head", "op": "GlobalAvgPool + Dense", "source": f"{_cls(gp)} + Linear"
                      + (" (rows permuted to the standard class order)" if class_perm is not None else ""), "weights": "fp32", "activations": "fp32"})
    g.set_outputs(y)

    known = {"NetworkCIFAR", "NetworkImageNet", "Cell", "ModuleList", "Sequential", "Conv2d", "BatchNorm2d", "ReLU", "Linear",
             "AdaptiveAvgPool2d", "AvgPool2d", "MaxPool2d", "Zero", "Identity", "Dropout", "ReLUConvBN", *BIN_OPS}
    for n, m in model.named_modules():
        _expect(_cls(m) in known, f"module {n or '<root>'}: unsupported type {_cls(m)}")
    unused = sorted({n.rsplit(".", 1)[0] for n, _ in list(model.named_parameters()) + list(model.named_buffers())
                     if not any(n == v or n.startswith(v + ".") for v in visited)})
    # the only parameters allowed outside the graph: branches whose output only reaches Zero ops (dead code)
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": binarize,
            "unused_in_forward": unused, "float64_constants": exact, "binary_weights": weight_bits,
            "binarization_points": bin_points, "float_weight_convs_with_binary_input": float_quirk,
            "zero_channel_corrections": zero_corr, "bn_sign_flips": bn_flips, "binarized_float_input_convs": res_binarized,
            "structurally_zero_channels": {t: int(mk.sum()) for t, mk in zmask.items()}}
    return g, info
