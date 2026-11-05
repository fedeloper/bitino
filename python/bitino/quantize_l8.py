"""int8_full variant with an 8-bit log2 ("l8") residual stream and fused residual blocks (docs/int8.md,
"log2 residual stream").

Post-pass over the int8_full graph emitted by quantize_int8 (same calibration, same rewrites). Every residual block

    I8BConv2D -> I8AffineSum / I8Add (+ its binarize-aware "/bin" copy) -> I8Lut (ChannelAffine bias, PReLU,
    optional ChannelAffine bias)

becomes one I8BConv2DRes: the binary convolution's fp32 value, plus the decoded residual, through the PReLU chain,
encoded once as an l8 code (int8_ops._bconv_res). Its second output is the bit tensor the next binary layer reads:
the binarization of the unrounded pre-activation sum against that layer's thresholds (mapped by R3b), so the
binarize-aware int8 copies of the sums disappear. Residual inputs: l8 outputs of other fused blocks, l8 outputs of
1x1 downsample I8Conv2D (output "l8"), the stem's max pool (l8 by default: the stem convolution writes l8 and is computed
once; opts["stem"]), or linear int8 tensors. Other readers of an l8 tensor
(average pooling of the downsample path, the global pool before the classifier) read it through a per-tensor I8Lut
back to the tensor's linear int8 grid.

The l8 exponent of a tensor is ceil(log2(max |x|)) over the calibration min / max (power of two, so residual decoding
across tensors and the folding of 2^-E into the constants are exact).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import int8_ops
from .ir import Constant, Graph, Node
from . import quantize_int8 as QI


def l8_exp(st: dict, rule: str = "minmax") -> int:
    """E = ceil(log2 max|x|) over the calibration min / max (default), or over the clipped range (rule "lohi": the
    percentile range of the int8_full calibration; values beyond 2^E saturate to +-127), minus an optional extra
    shift (rule "lohi-k" / "minmax-k": k octaves more resolution near 0, more saturation)."""
    base, _, k = rule.partition("-")
    lo, hi = ("lo", "hi") if base == "lohi" else ("min", "max")
    a = max(abs(float(st[lo])), abs(float(st[hi])), 2.0 ** -60)
    return int(math.ceil(math.log2(a))) - (int(k) if k else 0)


def _vec(Q, t, n, default):
    if t:
        return np.zeros(n) + np.asarray(Q.const_f(t), np.float64)
    return np.full(n, float(default))


def _chain(Q, ops, c):
    """(beta, slope, b2) of a unary chain [ChannelAffine(bias)]* PReLU [ChannelAffine(bias)]* with unit scales, else None."""
    beta, b2 = np.zeros(c), np.zeros(c)
    slope, has_b2 = None, False
    for op, attrs, ins in ops:
        if op == "ChannelAffine":
            if len(ins) > 1 and ins[1] and not np.all(np.asarray(Q.const_f(ins[1])) == 1.0):
                return None
            b = _vec(Q, ins[2] if len(ins) > 2 else "", c, 0.0)
            if slope is None:
                beta = beta + b
            else:
                b2 = b2 + b
                has_b2 = True
        elif op == "Activation" and attrs.get("type") == "prelu" and slope is None:
            if len(ins) < 2 or not ins[1] or not Q.is_const(ins[1]):
                return None
            slope = _vec(Q, ins[1], c, 0.0)
        else:
            return None
    if slope is None:
        return None
    return beta, slope, (b2 if has_b2 else None)


def _same_binarization(Q, nodes, c) -> bool:
    """Binary layers that binarize their (common) input identically: same mode, same per-channel thresholds."""
    def key(n):
        t = n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else ""
        return n.attrs.get("input_binarize", "ge"), (np.zeros(c) + np.asarray(Q.const_f(t), np.float64)) if t else np.zeros(c)
    m0, t0 = key(nodes[0])
    return all(key(n)[0] == m0 and np.array_equal(key(n)[1], t0) for n in nodes[1:])
def acc_thresholds(t, mode: str, b, s) -> np.ndarray:
    """int32 T[o] with (a >= T[o]) == (s[o]*a + b[o] >= t[o]) (mode ge; > t[o] for gt) for every integer a (s > 0; s == 0:
    the constant b[o] against t[o]; a NaN threshold never binarizes to +1). Evaluated in float64: exact unless (t - b) / s
    rounds across an integer, i.e. within ~1e-16 relative of a tie."""
    t, b, s = (np.asarray(v, np.float64) for v in (t, b, s))
    t, b, s = np.broadcast_arrays(t, b, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (t - b) / s
    T = np.floor(q) + 1 if mode == "gt" else np.ceil(q)
    const_pos = (b > t) if mode == "gt" else (b >= t)
    T = np.where(s == 0, np.where(const_pos, 0.0, 1.0), T)
    T = np.nan_to_num(T, nan=2.0 ** 31 - 1, posinf=2.0 ** 31 - 1, neginf=-(2.0 ** 31))
    return np.clip(T, -(2 ** 31), 2 ** 31 - 1).astype(np.int32)


def _find_stems(blocks, yset, gemm_l8, fprod, fcons, qprod, qcons) -> dict:
    """Blocks whose residual r is a linear int8 tensor I8Conv2D -> I8MaxPool2D* that the block's binary convolution reads
    through a binarize-aware "/bin" twin of that whole chain (the ReActNet stem: conv1, max pool). {r: stem}."""
    out = {}
    for b in blocks:
        r, fB = b["res"], b["B"]
        if r in yset or r in gemm_l8 or fB.inputs[0] != r or r + ":bin" not in qprod:
            continue
        if sorted(u.name for u, _ in fcons[r]) != sorted((fB.name, b["A"].name)):
            continue
        pools, t = [], r
        while t in qprod and qprod[t].op == "I8MaxPool2D":
            pools.insert(0, qprod[t])
            t = qprod[t].inputs[0]
        conv = qprod.get(t)
        fconv = fprod.get(t)
        if conv is None or conv.op != "I8Conv2D" or conv.attrs.get("output", "i8") != "i8" or fconv is None or \
                fconv.op != "Conv2D" or fconv.attrs.get("activation", "none") != "none":
            continue
        chain = [conv] + pools
        twins = [qprod.get(n.outputs[0] + ":bin") for n in chain]
        if any(tw is None or tw.name != n.name + "/bin" for tw, n in zip(twins, chain)):
            continue
        if any(len(qcons[n.outputs[0]]) != 1 or len(qcons[n.outputs[0] + ":bin"]) != 1 for n in chain[:-1]):
            continue
        an = b["A"].name
        if len(qcons[r + ":bin"]) != 1 or any(u.name not in (an, an + "/bin") for u, _ in qcons[r]):
            continue
        out[r] = dict(conv=conv, fconv=fconv, pools=pools, twins=[tw.name for tw in twins], block=b)
    return out


def fuse(qg: Graph, Q, E, opts: dict | None = None) -> tuple[Graph, dict]:
    """qg: the int8_full graph emitted from Q (quantize_int8.rewrite) by emitter E. Returns (graph, report)."""
    stats = Q.stats
    qn = {n.name: n for n in qg.nodes}
    qprod = {o: n for n in qg.nodes for o in n.outputs}
    qcons = defaultdict(list)
    for n in qg.nodes:
        for j, t in enumerate(n.inputs):
            if t:
                qcons[t].append((n, j))
    fprod = {n.outputs[0]: n for n in Q.nodes}
    fcons = defaultdict(list)
    for n in Q.nodes:
        for j, t in enumerate(n.inputs):
            if t:
                fcons[t].append((n, j))

    res_pre = bool((opts or {}).get("res_pre", False))
    pre_of = {}  # binary convolution name -> its pre stage (opts["res_pre"])

    def walk_pre(x, c):
        """opts["res_pre"]: the summand x of a block sum reached from a BConv2D through single-reader _Unary (ChannelAffine,
        PReLU / ReLU) and ChannelGate (sigmoid, input 0) nodes. Returns (B, pre) with pre = dict(post_s, post_b (affines
        before the activation, folded into post_mul / post_add), slope, mul, add (after it), gate (the ChannelGate node)) in
        the order of the fused op's pre stage: activation, affine, gate. A gate before the activation commutes with it
        (G >= 0: prelu(G v) = G prelu(v)); an affine after the gate does not (None)."""
        chain, t = [], x
        while True:
            p = fprod.get(t)
            if p is None or len(fcons[t]) != 1 or p.name not in qn:
                return None
            if p.op == "BConv2D":
                B = p
                break
            if p.op == "_Unary" or (p.op == "ChannelGate" and fcons[p.inputs[0]] == [(p, 0)]):
                chain.insert(0, p)
                t = p.inputs[0]
                continue
            return None
        bact = B.attrs.get("activation", "none")
        if bact not in ("none", "relu"):
            return None
        post_s, post_b = np.ones(c), np.zeros(c)
        slope = np.zeros(c) if bact == "relu" else None
        mul, add, gate = np.ones(c), np.zeros(c), None
        for n in chain:
            if n.op == "ChannelGate":
                if gate is not None or n.attrs.get("gate", "sigmoid") != "sigmoid":
                    return None
                gate = n
                continue
            for op, attrs, ins in n.attrs["ops"]:
                if op == "ChannelAffine":
                    sv = _vec(Q, ins[1] if len(ins) > 1 else "", c, 1.0)
                    bv = _vec(Q, ins[2] if len(ins) > 2 else "", c, 0.0)
                    if slope is None:          # before the activation: into post_mul / post_add
                        if gate is not None:
                            return None
                        post_s, post_b = post_s * sv, post_b * sv + bv
                    else:
                        if gate is not None:
                            return None
                        mul, add = mul * sv, add * sv + bv
                elif op == "Activation" and attrs.get("type") in ("prelu", "relu") and slope is None:
                    if attrs.get("type") == "prelu":
                        if len(ins) < 2 or not ins[1] or not Q.is_const(ins[1]):
                            return None
                        slope = _vec(Q, ins[1], c, 0.0)
                    else:
                        slope = np.zeros(c)
                else:
                    return None
        if slope is None and gate is None:
            return None
        return B, dict(post_s=post_s, post_b=post_b, slope=slope, mul=mul, add=add, gate=gate, chain=[n.name for n in chain])

    def branch(A, acts=("none",)):
        """A = BConv2D (+) residual: (B, residual, c0, bias) or None. With opts["res_pre"] the summand may also be a binary
        convolution with a ReLU (QuickNet BConv[relu] -> BatchNorm folded into the sum's coefficients) or reach the sum through
        a chain walk_pre accepts (squeeze-excitation gates, PReLU / BatchNorm between the convolution and the add): the fused op
        then computes that chain in its pre stage (int8_ops.bconv_res_pre); pre_of[B.name] holds it."""
        if A is None or A.op not in ("_AffineSum", "Add") or A.attrs.get("activation", "none") not in acts or A.name not in qn:
            return None
        ins = [x for x in A.inputs if x and not Q.is_const(x)]
        if len(ins) != 2 or len(A.inputs) != 2:
            return None
        c = Q._shape(A.outputs[0])[-1]
        bconv = [x for x in ins if fprod.get(x) is not None and fprod[x].op == "BConv2D" and len(fcons[x]) == 1]
        pre = None
        if len(bconv) != 1 and res_pre:
            cand = [(x, walk_pre(x, c)) for x in ins]
            cand = [(x, w) for x, w in cand if w is not None]
            if len(cand) == 1:
                bconv = [cand[0][0]]
                pre = cand[0][1]
        if len(bconv) != 1:
            return None
        xb = bconv[0]
        res = ins[1] if ins[0] == xb else ins[0]
        B = fprod[xb] if pre is None else pre[0]
        bact = B.attrs.get("activation", "none")
        if pre is None and not (bact == "none" or (res_pre and bact == "relu")) or B.name not in qn:
            return None
        if A.op == "_AffineSum":
            coef = dict(zip(A.inputs, A.attrs["coef"]))
            c0, c1 = np.zeros(c) + coef[xb], np.zeros(c) + coef[res]
            bias = np.zeros(c) + np.asarray(A.attrs["bias"], np.float64)
        else:
            c0, c1, bias = np.ones(c), np.ones(c), np.zeros(c)
        if not np.all(c1 == 1.0):
            return None
        if pre is not None:
            pre_of[B.name] = pre[1]
        elif bact == "relu":   # QuickNet: BConv[relu], the sum's coefficient after the ReLU
            pre_of[B.name] = dict(post_s=np.ones(c), post_b=np.zeros(c), slope=np.zeros(c), mul=np.ones(c), add=np.zeros(c),
                                  gate=None, chain=[])
        return B, res, c0, bias

    # ---- 1. find the blocks (float level)
    blocks = []
    for U in Q.nodes:
        if U.op != "_Unary" or U.name not in qn:
            continue
        t = U.inputs[0]
        A = fprod.get(t)
        parts, cat = None, None
        if A is not None and A.op == "Concat" and len(A.inputs) == 2 and A.name in qn and \
                A.attrs.get("axis", -1) % len(Q._shape(t)) == len(Q._shape(t)) - 1 and all(len(fcons[a]) == 1 for a in A.inputs):
            # ReActNet-A downsample pair: Concat(B1(x) + r, B2(x) + r) with B1, B2 1x1-style binary convolutions of the
            # same input, binarized alike -> one binary convolution with both weight sets, residual Concat(r, r)
            p2 = [branch(fprod.get(a)) for a in A.inputs]
            if all(p is not None for p in p2) and p2[0][1] == p2[1][1]:
                B1, B2 = p2[0][0], p2[1][0]
                q1, q2 = qn[B1.name], qn[B2.name]
                same = (q1.inputs[0] == q2.inputs[0] and q1.attrs == q2.attrs and B1.inputs[0] == B2.inputs[0] and
                        _same_binarization(Q, [B1, B2], Q._shape(B1.inputs[0])[-1]))
                if same:
                    parts = [(fprod[A.inputs[0]],) + p2[0], (fprod[A.inputs[1]],) + p2[1]]
                    cat = A
        if parts is None:
            p1 = branch(A)
            if p1 is None:
                continue
            parts = [(A,) + p1]
        A0, B, res = parts[0][0], parts[0][1], parts[0][2]
        c0 = np.concatenate([p[3] for p in parts])
        bias = np.concatenate([p[4] for p in parts])
        A = cat if cat is not None else A0
        c = Q._shape(t)[-1]
        ch = _chain(Q, U.attrs["ops"], c)
        if ch is None:
            continue
        # the next binary layer(s): reading the sum t (through R3b-mapped thresholds; bits of u), or else reading the
        # block output y (a chain no threshold on u can express, e.g. PReLU slopes <= 0; bits of w, bit_src "out").
        # Several readers must binarize alike (same mode and thresholds): they share one bit output.
        others = [(u, j) for u, j in fcons[t] if u is not U]
        if any(not (u.op in ("BConv2D",) and j == 0) for u, j in others):
            continue
        src = "sum"
        if not others:
            src = "out"
            others = [(u, j) for u, j in fcons[U.outputs[0]] if u.op in ("BConv2D", "BDense") and j == 0]
            others = [(u, j) for u, j in others if u.op == "BConv2D"] if all(u.op == "BConv2D" for u, j in others) else None
            if others is None:
                continue
        if len(others) > 1 and not _same_binarization(Q, [u for u, _ in others], c):
            continue
        blocks.append(dict(U=U, A=A, B=B, t=t, y=U.outputs[0], res=res, c0=c0, bias=bias, chain=ch,
                           nxt=others[0][0] if others else None, src=src, parts=parts, cat=cat))
    # blocks without an activation chain (Bi-Real / ResNet-E style: BConv2D -> Add with the residual, the sum is the block
    # output): the fused op's output is the sum itself (PReLU slope 1, w = u exactly), its bit output the binarization of the
    # sum by the next binary convolution(s). opts["identity_blocks"]: "plain" (default; sums read only by binarizations and
    # non-unary operators: the next residual add, pools, 1x1 convolutions), "all" (also sums read by a unary chain the block
    # pattern above could not fuse, e.g. hardtanh: that LUT then indexes the l8 code, step 5), "off".
    idmode = (opts or {}).get("identity_blocks", "plain")
    if idmode not in ("plain", "all", "off"):
        raise ValueError("identity_blocks must be plain, all or off")
    if idmode != "off":
        claimed = {b["A"].name for b in blocks} | {p[0].name for b in blocks for p in b["parts"]}
        for A in Q.nodes:
            if A.op not in ("Add", "_AffineSum") or A.name in claimed:
                continue
            p1 = branch(A, ("none", "relu"))
            t = A.outputs[0]
            if p1 is None or t in Q.g.outputs:
                continue
            relu = A.attrs.get("activation", "none") == "relu"   # Add[relu] (NNCF / OMZ ResNets): PReLU slope 0
            readers = fcons[t]
            if not readers or (idmode == "plain" and any(u.op == "_Unary" for u, _ in readers)):
                continue
            bins = [(u, j) for u, j in readers if u.op in ("BConv2D", "BDense", "_BMatMulQ")]
            if any(not (u.op == "BConv2D" and j == 0) for u, j in bins):
                continue
            c = Q._shape(t)[-1]
            if len(bins) > 1 and not _same_binarization(Q, [u for u, _ in bins], c):
                continue
            B, res, c0, bias = p1
            # relu: the next layer binarizes relu(sum), the bits of w (bit_src "out")
            blocks.append(dict(U=None, A=A, B=B, t=t, y=t, res=res, c0=c0, bias=bias,
                               chain=(np.zeros(c), np.zeros(c) if relu else np.ones(c), None),
                               nxt=bins[0][0] if bins else None, src="out" if relu else "sum", parts=[(A,) + p1], cat=None))
    if not blocks:
        return qg, {"fused_blocks": 0}

    yset = {b["y"]: b for b in blocks}
    l8_rule = (opts or {}).get("l8_range", "minmax")
    E_of = {b["y"]: l8_exp(stats[b["y"]], l8_rule) for b in blocks}
    # downsample 1x1 convolutions feeding only a block's residual input -> l8 output
    gemm_l8 = {}
    for b in blocks:
        r = b["res"]
        p = fprod.get(r)
        if r in yset or p is None or p.op != "Conv2D" or len(fcons[r]) != 1 or p.attrs.get("activation", "none") != "none":
            continue
        if r not in qprod or qprod[r].op != "I8Conv2D" or r + ":bin" in qprod:
            continue
        gemm_l8[r] = p
        E_of[r] = l8_exp(stats[r], l8_rule)
    # the stem (docs/int8.md, "log2 residual stream", "The stem"), opts["stem"]: "l8" (default): one GEMM with l8 output,
    # max pooling on the codes (monotone), the block reads the pooled codes as its residual and binarizes them with the
    # integer thresholds of the code; "split": the prototype's linear stem with its binarize-aware twin (the GEMM computed
    # twice); "bits" / "l8_bits": one GEMM (linear int8 / l8 output) that also writes the binarization of its exact
    # accumulator (bit output), max pooled as bits
    stem_mode = (opts or {}).get("stem", "l8")
    if stem_mode not in ("split", "l8", "bits", "l8_bits"):
        raise ValueError("stem must be split, l8, bits or l8_bits")
    stems = {} if stem_mode == "split" else _find_stems(blocks, yset, gemm_l8, fprod, fcons, qprod, qcons)

    new_consts: dict[str, Constant] = {}

    def const(name, arr, dtype="f32"):
        a = np.ascontiguousarray(arr)
        new_consts[name] = Constant(name, dtype, tuple(a.shape), a)
        return name

    remove = set()
    replace = {}  # node name -> list of nodes emitted in its place
    bits_of = {}  # tensor the next binary layer binarizes (sum t, or block output y for bit_src "out") -> bit tensor name
    for b in blocks:
        if b["nxt"] is not None:
            b["bkey"] = b["t"] if b["src"] == "sum" else b["y"]
            bits_of[b["bkey"]] = b["bkey"] + ":bits"

    report = {"fused_blocks": len(blocks), "l8_tensors": {}, "gemm_l8": [], "lut_to_linear": [], "bit_inputs": []}

    def gemm_scales(gq, p):
        sx, _ = E.p(p.inputs[0])
        w = np.asarray(Q.const_f(p.inputs[1]), np.float64)
        wq, sw = QI.quantize_weights_per_channel(w)
        assert np.array_equal(wq, np.asarray(qg.constants[gq.inputs[1]].data)), "weights differ from the emitted ones"
        o = w.shape[0]
        bf = np.asarray(Q.const_f(p.inputs[2]), np.float64) if len(p.inputs) > 2 and p.inputs[2] else np.zeros(o)
        return sx * sw, bf

    def gemm_l8_node(gq, p, e):
        s_acc, bf = gemm_scales(gq, p)
        at = {k: v for k, v in gq.attrs.items() if k not in ("y_zero", "qmin", "qmax")}
        at["output"] = "l8"
        at["y_exp"] = e
        bn = const(gq.name + "/bias_l8", (bf * 2.0 ** -e).astype(np.float32))
        sn = const(gq.name + "/scale_l8", (s_acc * 2.0 ** -e).astype(np.float32))
        return Node("I8Conv2D", [gq.inputs[0], gq.inputs[1], bn, sn], list(gq.outputs), at, gq.name)

    # ---- 2. downsample GEMMs with l8 output
    for r, p in gemm_l8.items():
        gq = qprod[r]
        e = E_of[r]
        replace[gq.name] = [gemm_l8_node(gq, p, e)]
        report["gemm_l8"].append(gq.name)
        report["l8_tensors"][r] = e

    # ---- 2b. the stem: one GEMM instead of the GEMM and its binarize-aware twin
    if stem_mode != "split":
        report["stem"] = {"mode": stem_mode, "convs": [s["conv"].name for s in stems.values()]}
    for r, st in stems.items():
        gq, p, fB = st["conv"], st["fconv"], st["block"]["B"]
        remove.update(st["twins"])
        yc = gq.outputs[0]
        if stem_mode in ("l8", "l8_bits"):
            E_of[r] = l8_exp(stats[yc], l8_rule)
            m = gemm_l8_node(gq, p, E_of[r])
            report["l8_tensors"][yc] = report["l8_tensors"][r] = E_of[r]
        else:
            m = Node("I8Conv2D", list(gq.inputs), list(gq.outputs), dict(gq.attrs), gq.name)
        replace[gq.name] = [m]
        st["x"] = r
        if stem_mode in ("bits", "l8_bits"):
            # bit +1 iff the exact accumulator reaches the block's binarization threshold: maximum pooling commutes
            # (max >= t iff some element >= t), so the pooled bits are the binarization of the pooled exact values
            c = Q._shape(yc)[-1]
            thr = np.zeros(c) + (np.asarray(Q.const_f(fB.inputs[4]), np.float64) if len(fB.inputs) > 4 and fB.inputs[4] else 0.0)
            mode = fB.attrs.get("input_binarize", "ge")
            if mode not in ("ge", "gt"):
                raise NotImplementedError("stem bits for " + mode)
            s_acc, bf = gemm_scales(gq, p)
            while len(m.inputs) < 5:
                m.inputs.append("")
            m.inputs.append(const(gq.name + "/bit_thr", acc_thresholds(thr, mode, bf, s_acc), "i32"))
            m.outputs.append(yc + ":bits")
            prev = yc + ":bits"
            for pn in st["pools"]:
                bp = Node("I8MaxPool2D", [prev], [pn.outputs[0] + ":bits"], dict(pn.attrs), pn.name + "/bits")
                replace[pn.name] = [pn, bp]
                prev = bp.outputs[0]
            st["x"] = prev

    # ---- 3. fused blocks
    for b in blocks:
        B, A = qn[b["B"].name], qn[b["A"].name]
        U = qn[b["U"].name] if b["U"] is not None else None
        fB = b["B"]
        y, t, r = b["y"], b["t"], b["res"]
        e = E_of[y]
        report["l8_tensors"][y] = e
        o = Q._shape(y)[-1]
        pms, pas = [], []
        for part in b["parts"]:
            pB = part[1]
            ob = Q._shape(pB.outputs[0])[-1]
            pm = pB.attrs.get("_pm64")
            pa = pB.attrs.get("_pa64")
            pms.append(np.zeros(ob) + (pm if pm is not None else _vec(Q, pB.inputs[2] if len(pB.inputs) > 2 else "", ob, 1.0)))
            pas.append(np.zeros(ob) + (pa if pa is not None else _vec(Q, pB.inputs[3] if len(pB.inputs) > 3 else "", ob, 0.0)))
        pm, pa = np.concatenate(pms), np.concatenate(pas)
        beta, slope, b2 = b["chain"]
        wname = B.inputs[1]
        if b["cat"] is not None:   # merged pair: weights of both binary convolutions, residual Concat(r, r) (a copy)
            B2q = qn[b["parts"][1][1].name]
            w1, w2 = qg.constants[B.inputs[1]], qg.constants[B2q.inputs[1]]
            wname = const(B.name + "+" + B2q.name.split(".")[-1] + "/kernel_bits",
                          np.concatenate([np.asarray(w1.data), np.asarray(w2.data)], axis=0), w1.dtype)
            new_consts[wname] = Constant(wname, w1.dtype, (w1.shape[0] + w2.shape[0],) + tuple(w1.shape[1:]),
                                         new_consts[wname].data)
            rl8 = r in yset or r in gemm_l8
            zr = 0 if rl8 else int(E.params[r][1])
            rc = r + ":x2"
            pre_nodes = [Node("I8Concat", [r, r], [rc], {"axis": -1, "mults": [0, 0], "shifts": [0, 0], "y_zero": zr, "zeros": [zr, zr]},
                              b["cat"].name + "/res")]
        else:
            pre_nodes = []
            rc = r
        sc = 2.0 ** -e
        pre = pre_of.get(fB.name)
        if pre is not None:
            # pre stage (int8_ops.bconv_res_pre), all in units of 2^E (a power of two commutes with the activation and the
            # gate exactly): v = acc * pm' + pa' (pm' = pm * post_s, pa' = pa * post_s + post_b); v = act(v) (slope);
            # v = v * mul' + add'; v = v * G; u = v + R. The sum's coefficient c0 multiplies mul / add (or post_mul / post_add
            # when there is no activation); its bias and beta follow the gate only when there is none.
            if b["cat"] is not None:
                raise NotImplementedError("pre stage of a merged pair")
            c0, bb = b["c0"], b["bias"] + beta
            pm1, pa1 = pm * pre["post_s"], pa * pre["post_s"] + pre["post_b"]
            if pre["slope"] is None:
                pm1, pa1 = c0 * pm1, c0 * pa1
                mul, add = None, None
            else:
                mul, add = c0 * pre["mul"], c0 * pre["add"]
            if pre["gate"] is not None and np.any(bb != 0):
                raise NotImplementedError("bias after a squeeze-excitation gate")
            if pre["gate"] is None:
                if add is None:
                    pa1 = pa1 + bb
                else:
                    add = add + bb
            PM = const(B.name + "/pm_l8", (pm1 * sc).astype(np.float32))
            PA = const(B.name + "/pa_l8", (pa1 * sc).astype(np.float32))
            pre_ins = [""]
            pre_ins.append(const(B.name + "/pre_slope", pre["slope"].astype(np.float32)) if pre["slope"] is not None else "")
            pre_ins.append(const(B.name + "/pre_mul", mul.astype(np.float32)) if mul is not None and np.any(mul != 1.0) else "")
            pre_ins.append(const(B.name + "/pre_add", (add * sc).astype(np.float32)) if add is not None and np.any(add != 0) else "")
            if pre["gate"] is not None:
                gq = qn[pre["gate"].name]          # the I8Mul of the gate: input 1 is the [N, 1, 1, C] sigmoid code (1/255, -128)
                assert gq.op == "I8Mul" and int(gq.attrs["b_zero"]) == -128, gq
                pre_ins.append(gq.inputs[1])
                remove.add(gq.name)
            for nm in pre["chain"]:
                if nm in qn:
                    remove.add(nm)
            report.setdefault("pre_blocks", []).append(B.name)
        else:
            PM = const(B.name + "/pm_l8", (b["c0"] * pm * sc).astype(np.float32))
            PA = const(B.name + "/pa_l8", ((b["c0"] * pa + b["bias"] + beta) * sc).astype(np.float32))
            pre_ins = []
        SL = const(B.name + "/slope", slope.astype(np.float32))
        at = {k: v for k, v in B.attrs.items() if k not in ("qmin", "qmax", "output")}
        x = B.inputs[0]
        xin = x
        # the binarized input: the bit output of the block producing it (instead of its "/bin" int8 copy, or of the
        # sum's own int8 grid where no binarize-aware copy was kept)
        src = x[:-4] if x.endswith(":bin") else x
        thr_in = B.inputs[4] if len(B.inputs) > 4 else ""
        if src in bits_of:
            xin = bits_of[src]
            thr_in = ""
            for k in ("x_scale", "x_zero", "input_binarize"):
                at.pop(k, None)
        if r in stems and x == r + ":bin":
            xin = stems[r]["x"]
            for k in ("x_scale", "x_zero"):
                at.pop(k, None)
            if xin == r:  # stem "l8": the block binarizes the l8 code (integer thresholds T_ge / T_gt of the code)
                at.update(x_code="l8", x_exp=E_of[r])
            else:  # bit tensor
                thr_in = ""
                at.pop("input_binarize", None)
        if r in yset or r in gemm_l8 or (r in stems and stem_mode in ("l8", "l8_bits")):
            rq = rc
            at.update(r_code="l8", r_exp=E_of[r] - e)
        else:
            s_r, z_r = E.params[r]
            rq = rc
            at.update(r_code="linear", r_scale=float(np.float32(s_r * sc)), r_zero=int(z_r))
        at["y_exp"] = e
        if pre is not None and pre["gate"] is not None:
            at.update(g_scale=float(np.float32(1.0 / 255.0)), g_zero=-128)
        ins = [xin, wname, PM, PA, thr_in, rq, SL]
        ins.append(const(B.name + "/out_bias", (b2 * sc).astype(np.float32)) if b2 is not None else "")
        outs = [y]
        if b["nxt"] is not None:
            nx = b["nxt"]
            tn = np.zeros(o) + (np.asarray(Q.const_f(nx.inputs[4]), np.float64) if len(nx.inputs) > 4 and nx.inputs[4] else 0.0)
            if b["src"] == "sum":   # thresholds on t, mapped to u = t + beta
                ins.append(const(B.name + "/bit_thr", ((tn + beta) * sc).astype(np.float32)))
            else:                   # thresholds on y = w * 2^E
                ins.append(const(B.name + "/bit_thr", (tn * sc).astype(np.float32)))
                at["bit_src"] = "out"
                if y + ":bin" in qprod:  # the binarize-aware copy of y (a LUT on t) is replaced by the bits
                    remove.add(qprod[y + ":bin"].name)
            at["bit_binarize"] = nx.attrs.get("input_binarize", "ge")
            if at["bit_binarize"] not in ("ge", "gt"):
                raise NotImplementedError("bit output for " + at["bit_binarize"])
            outs.append(bits_of[b["bkey"]])
        if pre_ins:   # inputs 9 (out_lut: Session only) .. 12
            while len(ins) < 9:
                ins.append("")
            ins.extend(pre_ins)
        while ins and ins[-1] == "":
            ins.pop()
        name = B.name + "+res"
        replace[A.name] = pre_nodes + [Node("I8BConv2DRes", ins, outs, at, name)]
        tw = qprod.get(y + ":bin")
        if tw is not None and tw.op == "I8Lut" and tw.name not in remove and t in tw.inputs:
            # a remaining binarize-aware copy of y (read by a centering add, quantize_int8 center_split) reads the sum t,
            # which the fusion removes: index y's l8 code instead, the table maps the code to the copy's grid
            sb, zb = E.params[y + ":bin"]
            tab = int8_ops.quantize(int8_ops.l8_decode(np.arange(-128, 128), e), sb, zb).reshape(1, 256)
            replace[tw.name] = [Node("I8Lut", [y, const(tw.name + "/from_l8", tab.astype(np.int8), "i8")], list(tw.outputs), {}, tw.name)]
        remove.update({B.name, A.name + "/bin"} | ({U.name} if U is not None else set()))
        for part in b["parts"][1:] if b["cat"] is not None else []:
            remove.add(part[1].name)
        if b["cat"] is not None:
            remove.update({p[0].name for p in b["parts"]} | {p[0].name + "/bin" for p in b["parts"]})

    # ---- 4. plain binary layers reading a fused block's sum through its "/bin" copy -> bit input
    for n in qg.nodes:
        if n.op not in ("I8BConv2D", "I8BDense") or n.name in remove:
            continue
        x = n.inputs[0]
        xs = x[:-4] if x.endswith(":bin") else x
        if xs in bits_of:
            m = Node(n.op, [bits_of[xs]] + list(n.inputs[1:]), list(n.outputs), {k: v for k, v in n.attrs.items()
                                                                                   if k not in ("x_scale", "x_zero", "input_binarize")}, n.name)
            if len(m.inputs) > 4:
                m.inputs[4] = ""
            while m.inputs and m.inputs[-1] == "":
                m.inputs.pop()
            replace[n.name] = [m]
            report["bit_inputs"].append(n.name)

    # ---- 5. other readers of an l8 tensor -> per-tensor LUT back to its linear grid
    a_names = {b["A"].name for b in blocks}
    for yv in list(yset) + list(gemm_l8):
        readers = [(u, j) for u, j in qcons.get(yv, []) if u.name not in remove and u.name not in a_names]
        if not readers:
            continue
        s, z = E.params[yv]
        tab = int8_ops.quantize(int8_ops.l8_decode(np.arange(-128, 128), E_of[yv]), s, z).reshape(1, 256)
        ln = yv + ":lin"
        if (opts or {}).get("l8_compose_luts", True):
            # an I8Lut reader indexes the l8 code directly with its table composed with l8 -> linear (the same values
            # as the two LUTs in sequence, one pass less)
            rest = []
            for u, j in readers:
                tc = qg.constants.get(u.inputs[1]) if u.op == "I8Lut" and j == 0 and u.name not in replace else None
                if tc is None:
                    rest.append((u, j))
                    continue
                old = np.asarray(tc.data).reshape(-1, 256)
                comp = old[:, tab[0].astype(np.int64) + 128]
                cn = const(u.inputs[1] + "/from_l8", comp.reshape(tuple(tc.shape)).astype(old.dtype), tc.dtype)
                replace[u.name] = [Node("I8Lut", [yv, cn] + list(u.inputs[2:]), list(u.outputs), dict(u.attrs), u.name)]
                report.setdefault("lut_composed", []).append(u.name)
            readers = rest
            if not readers:
                continue
        tn = const(yv + "/l8_to_lin", tab.astype(np.int8), "i8")
        report["lut_to_linear"].append(yv)
        for u, j in readers:
            if u.name not in replace:
                replace[u.name] = [Node(u.op, list(u.inputs), list(u.outputs), dict(u.attrs), u.name)]
            for m in replace[u.name]:
                m.inputs = [ln if (k == j and v == yv) else v for k, v in enumerate(m.inputs)]
        replace.setdefault("__after__" + yv, []).append(Node("I8Lut", [yv, tn], [ln], {}, yv.split(":")[0] + "/l8_to_lin"))

    # ---- 6. assemble, toposort, drop unused constants
    nodes = []
    for n in qg.nodes:
        if n.name in remove:
            continue
        nodes.extend(replace.get(n.name, [n]))
    for k, v in replace.items():
        if k.startswith("__after__"):
            nodes.extend(v)
    out = Graph()
    out.inputs = [dict(i) for i in qg.inputs]
    out.outputs = list(qg.outputs)
    out.manifest = dict(qg.manifest)
    consts = dict(qg.constants)
    consts.update(new_consts)
    avail = {i["name"] for i in out.inputs} | set(consts)
    order, pending = [], list(nodes)
    while pending:
        prog = False
        for n in list(pending):
            if all((not t) or t in avail for t in n.inputs):
                order.append(n)
                avail.update(n.outputs)
                pending.remove(n)
                prog = True
        if not prog:
            raise RuntimeError("l8 fusion: graph not acyclic / missing tensor: " +
                               str([(n.name, [t for t in n.inputs if t and t not in avail]) for n in pending][:3]))
    out.nodes = order
    used = {t for n in order for t in n.inputs if t}
    out.constants = {k: v for k, v in consts.items() if k in used}
    ops = defaultdict(int)
    for n in order:
        if n.op.startswith("I8"):
            ops[n.op] += 1
    report["int8_ops"] = dict(ops)
    return out, report
