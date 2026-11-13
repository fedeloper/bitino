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


def _int_epilogue(node: Node, K: int, d, beta_u, ratio_r: float, z_r: int, slope, ob_u, thr_u, const, base: str,
                  pb_u=None, pmul=None):
    """I8BConv2DResI of a lin8 I8BConv2DRes block (output units = the output step): u = dot * d[o] + q * ratio_r +
    (beta_u[o] - z_r * ratio_r), w = PReLU(u, slope) + ob_u, y = round(w) + y_zero, bits of u (or w) against thr_u. In integer
    units 2^-S: m1 = round(d 2^S), m2 = round(ratio_r 2^S), b = round(beta_u 2^S) - z_r m2, PReLU slope as Q31 / 2^k, S the
    largest shift (<= 16) with |m1|, |m2| <= 32767 and every value below 2^31 (int8_ops.bconv_res_i_bound). None if S < 8.
    Pre stage (pb_u given: a ReLU between the binary convolution and an affine with per-channel multiplier pmul, QuickNet's
    BConv[relu] -> BatchNorm): d and pb_u are already multiplied by pmul, so relu(x) * pmul = max(pmul x, 0) for pmul >= 0
    and min(pmul x, 0) for pmul < 0, i.e. clamp(dot * m1 + pre_b, pre_lo, pre_hi) with (0, +inf) or (-inf, 0)."""
    a = dict(node.attrs)
    if int(a.get("groups", 1)) != 1:   # the integer epilogue runs in the two-phase NEON path of groups 1 only
        return None
    o = len(d)
    slope = None if slope is None or np.all(np.asarray(slope) == 1.0) else np.zeros(o) + np.asarray(slope, np.float64)
    k = 0
    P = None
    if slope is not None:
        mx = float(np.max(np.abs(slope)))
        k = 0 if mx < 1.0 else int(math.floor(math.log2(mx))) + 1
        P = np.clip(np.round(slope * 2.0 ** (31 - k)), -(2 ** 31 - 1), 2 ** 31 - 1).astype(np.int64)
    has_res = len(node.inputs) > 5 and bool(node.inputs[5])
    for S in range(16, 7, -1):
        m1 = np.round(np.asarray(d, np.float64) * 2.0 ** S).astype(np.int64)
        m2v = int(round(ratio_r * 2.0 ** S)) if has_res else 0
        if np.max(np.abs(m1)) > 32767 or abs(m2v) > 32767:
            continue
        m2 = np.full(o, m2v, np.int64)
        bb = np.round(np.asarray(beta_u, np.float64) * 2.0 ** S).astype(np.int64) - z_r * m2v
        ob = None if ob_u is None else np.round(np.asarray(ob_u, np.float64) * 2.0 ** S).astype(np.int64)
        pb = None if pb_u is None else np.round(np.asarray(pb_u, np.float64) * 2.0 ** S).astype(np.int64)
        if int8_ops.bconv_res_i_bound(K, m1, m2, bb, P, k, ob, pb) >= 2 ** 30:
            continue
        break
    else:
        return None
    ins = [node.inputs[0], node.inputs[1], "", "", node.inputs[4] if len(node.inputs) > 4 else "", node.inputs[5] if has_res else "",
           const(base + "/m1_i", m1.astype(np.int32), "i32"), const(base + "/b_i", bb.astype(np.int32), "i32"),
           const(base + "/m2_i", m2.astype(np.int32), "i32") if has_res else ""]
    at = {kk: v for kk, v in a.items() if kk not in ("output", "r_code", "r_scale", "r_zero", "bit_binarize", "y_exp")}
    at.update(shift=S, y_zero=int(a.get("y_zero", 0)), qmin=int(a.get("qmin", -128)), qmax=int(a.get("qmax", 127)))
    if thr_u is not None:
        mode = a.get("bit_binarize", "ge")
        T = acc_thresholds(np.asarray(thr_u, np.float64), mode, np.zeros(o), np.full(o, 2.0 ** -S))
        ins.append(const(base + "/bit_thr_i", T.astype(np.int32), "i32"))
    else:
        ins.append("")
    if P is not None:
        ins.append(const(base + "/slope_i", P.astype(np.int32), "i32"))
        at["slope_shift"] = k
    else:
        ins.append("")
        at.pop("slope_shift", None)
    ins.append(const(base + "/out_bias_i", ob.astype(np.int32), "i32") if ob is not None else "")
    if pb is not None:
        neg = np.asarray(pmul, np.float64) < 0
        ins.append(const(base + "/pre_b_i", pb.astype(np.int32), "i32"))
        ins.append(const(base + "/pre_lo_i", np.where(neg, -(2 ** 31), 0).astype(np.int32), "i32"))
        ins.append(const(base + "/pre_hi_i", np.where(neg, 0, 2 ** 31 - 1).astype(np.int32), "i32"))
    while ins and ins[-1] == "":
        ins.pop()
    return Node("I8BConv2DResI", ins, list(node.outputs), at, node.name)


def fuse_gated(qg: Graph, Q, E, opts: dict | None = None) -> tuple[Graph, dict]:
    """int8_full v2 post-pass (after fuse with code lin8) for BNext's blocks (docs/int8.md "int8_full v2"):

        p   = PReLU(BConv(x) [+ beta]) [+ b2]          I8BConv2D -> I8Lut           =>  I8BConv2DResI without a residual
        add = k * (p * sigmoid(se)) + x + bias        I8Mul + I8AffineSum (+ /bin)  =>  I8GatedAdd (bits for the /bin readers)

    The mix AffineSum and the squeeze-excitation path read p as before. A block is taken only when every piece matches
    (p read by the gate and other non-binarizing nodes only; the gated product read only by the add); otherwise its nodes
    stay. Returns (graph, report)."""
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
    new_consts: dict[str, Constant] = {}

    def const(name, arr, dtype="f32"):
        a = np.ascontiguousarray(arr)
        new_consts[name] = Constant(name, dtype, tuple(a.shape), a)
        return name

    replace, remove = {}, set()
    report = {"prelu_convs": [], "gated_adds": []}
    for B in Q.nodes:
        if B.op != "BConv2D" or B.attrs.get("activation", "none") != "none" or B.name not in qn or qn[B.name].op != "I8BConv2D" or \
                int(B.attrs.get("groups", 1)) != 1:
            continue
        tB = B.outputs[0]
        if len(fcons[tB]) != 1 or fcons[tB][0][0].op != "_Unary":
            continue
        U = fcons[tB][0][0]
        p = U.outputs[0]
        c = Q._shape(p)[-1]
        ch = _chain(Q, U.attrs["ops"], c)
        if ch is None or U.name not in qn or qn[U.name].op != "I8Lut" or p + ":bin" in qprod or p in Q.g.outputs:
            continue
        if any(u.op in ("BConv2D", "BDense", "_BMatMulQ") for u, _ in fcons[p]):
            continue
        qB = qn[B.name]
        if len(qcons[tB]) != 1:
            continue
        if qB.name in replace:   # already pointed at a gated add's bit output (an earlier block): build on that node
            qB = replace[qB.name][0]
        beta, slope, b2 = ch
        s_p, z_p = E.params[p]
        sc = 1.0 / float(s_p)
        pm = np.zeros(c) + (B.attrs.get("_pm64") if B.attrs.get("_pm64") is not None else _vec(Q, B.inputs[2] if len(B.inputs) > 2 else "", c, 1.0))
        pa = np.zeros(c) + (B.attrs.get("_pa64") if B.attrs.get("_pa64") is not None else _vec(Q, B.inputs[3] if len(B.inputs) > 3 else "", c, 0.0))
        at = {k: v for k, v in qB.attrs.items() if k not in ("qmin", "qmax", "output")}
        at.update(y_zero=int(z_p), qmin=-128, qmax=127)
        stub = Node("I8BConv2DRes", list(qB.inputs[:5]) + [""] * max(0, 5 - len(qB.inputs[:5])) + [""], [p], at, B.name + "+prelu")
        ws = qg.constants[qB.inputs[1]].shape
        ni = _int_epilogue(stub, int(np.prod(ws[1:])), pm * sc, (pa + beta) * sc, 0.0, 0, slope,
                           (b2 * sc) if b2 is not None else None, None, const, B.name)
        if ni is None:
            continue
        replace[qB.name] = [ni]
        remove.add(qn[U.name].name)
        report["prelu_convs"].append(ni.name)
        # the gated residual add fed by p
        gates = [u for u, j in fcons[p] if u.op == "ChannelGate" and j == 0 and u.attrs.get("gate", "sigmoid") == "sigmoid"]
        if len(gates) != 1:
            continue
        G = gates[0]
        se = G.outputs[0]
        if len(fcons[se]) != 1 or fcons[se][0][0].op != "_AffineSum" or G.name not in qn or qn[G.name].op != "I8Mul":
            continue
        A = fcons[se][0][0]
        if len(A.inputs) != 2 or A.attrs.get("activation", "none") != "none" or A.name not in qn or qn[A.name].op != "I8AffineSum":
            continue
        x = A.inputs[1] if A.inputs[0] == se else A.inputs[0]
        coef = dict(zip(A.inputs, A.attrs["coef"]))
        qA, qG = qn[A.name], qn[G.name]
        t_add = A.outputs[0]
        tw = qprod.get(t_add + ":bin")
        if t_add in Q.g.outputs or any(u.name not in (qA.name, tw.name if tw is not None else None) for u, _ in qcons[se]):
            continue
        xq = qA.inputs[A.inputs.index(x)]   # the int8 tensor the add read for x (its grid: E.params)
        s_x, z_x = E.params[xq]
        s_y, z_y = E.params[t_add]
        g_t = qG.inputs[1]
        z_g, s_g = int(qG.attrs["b_zero"]), 1.0 / 255.0
        assert z_g == -128
        kx_r = np.zeros(c) + np.asarray(coef[x], np.float64) * s_x / s_y
        kp_r = np.zeros(c) + np.asarray(coef[se], np.float64) * s_p * s_g / s_y
        b_r = np.zeros(c) + np.asarray(A.attrs["bias"], np.float64) / s_y
        for S in range(16, 7, -1):
            kx = np.round(kx_r * 2.0 ** S).astype(np.int64)
            kp = np.round(kp_r * 2.0 ** S).astype(np.int64)
            bb = np.round(b_r * 2.0 ** S).astype(np.int64)
            if np.abs(kx).max() <= 32767 and int8_ops.gated_add_bound(kx, kp, bb) < 2 ** 30:
                break
        else:
            continue
        # binarizing readers of the add's binarize-aware twin: one bit output per distinct (mode, thresholds)
        groups, readers = [], []
        ok = True
        if tw is not None:
            for u, j in qcons[t_add + ":bin"]:
                if u.op not in ("I8BConv2D", "I8BDense", "I8BConv2DRes", "I8BConv2DResI") or j != 0 or u.name in replace:
                    ok = False
                    break
                mode = u.attrs.get("input_binarize", "ge")
                if mode not in ("ge", "gt"):
                    ok = False
                    break
                tq = np.zeros(c) + (np.asarray(qg.constants[u.inputs[4]].data, np.float64) if len(u.inputs) > 4 and u.inputs[4] else 0.0)
                T = acc_thresholds(tq / s_y, mode, np.zeros(c), np.full(c, 2.0 ** -S))
                k = next((i for i, g in enumerate(groups) if np.array_equal(g, T)), None)
                if k is None:
                    groups.append(T)
                    k = len(groups) - 1
                readers.append((u, k))
        if not ok:
            continue
        name = A.name + "+gated"
        ins = [xq, p, g_t, const(name + "/kx", kx.astype(np.int32), "i32"), const(name + "/kp", kp.astype(np.int32), "i32"),
               const(name + "/b", bb.astype(np.int32), "i32")]
        outs = [t_add]
        if groups:
            ins.append(const(name + "/bit_thr", np.stack(groups).astype(np.int32), "i32"))
            outs += [f"{t_add}:bits{k}" for k in range(len(groups))]
        replace[qA.name] = [Node("I8GatedAdd", ins, outs, {"x_zero": int(z_x), "p_zero": int(z_p), "g_zero": z_g, "shift": S,
                                                            "y_zero": int(z_y), "qmin": -128, "qmax": 127}, name)]
        remove.add(qG.name)
        if tw is not None:
            remove.add(tw.name)
        for u, k in readers:
            m = Node(u.op, [f"{t_add}:bits{k}"] + list(u.inputs[1:]), list(u.outputs),
                     {kk: v for kk, v in u.attrs.items() if kk not in ("x_scale", "x_zero", "input_binarize", "x_code", "x_exp")}, u.name)
            if len(m.inputs) > 4:
                m.inputs[4] = ""
            while m.inputs and m.inputs[-1] == "":
                m.inputs.pop()
            prev = replace.get(u.name)
            replace[u.name] = [m] if prev is None else prev
            if prev is not None:   # the reader is itself rewritten (e.g. into an I8BConv2DResI): point it at the bits
                for nn in prev:
                    if nn.inputs and nn.inputs[0] == t_add + ":bin":
                        nn.inputs[0] = f"{t_add}:bits{k}"
                        if len(nn.inputs) > 4:
                            nn.inputs[4] = ""
                        for kk in ("x_scale", "x_zero", "input_binarize", "x_code", "x_exp"):
                            nn.attrs.pop(kk, None)
        report["gated_adds"].append(name)
    # GlobalAvgPool(AffineSum(x_1..x_n)) with the sum read only by the pool (BNext's GAP(mix)): one I8GapAffine, the pool of
    # the exact sum (no rounding of the full-resolution sum, no pass writing it)
    report["gap_affines"] = []
    for GP in Q.nodes:
        if GP.op != "GlobalAvgPool" or GP.name not in qn or qn[GP.name].op != "I8GlobalAvgPool":
            continue
        mix = GP.inputs[0]
        A2 = fprod.get(mix)
        if A2 is None or A2.op != "_AffineSum" or A2.attrs.get("activation", "none") != "none" or A2.name not in qn or \
                qn[A2.name].op != "I8AffineSum" or len(fcons[mix]) != 1 or mix + ":bin" in qprod or A2.name in replace:
            continue
        qA2 = qn[A2.name]
        k = len(A2.inputs)
        xs = qA2.inputs[:k]
        if any(t in qg.constants for t in xs):
            continue
        shp = Q._shape(mix)
        hw, c = int(shp[1] * shp[2]), int(shp[-1])
        s_y, z_y = E.params[GP.outputs[0]]
        rows = [np.zeros(c) + np.asarray(cf, np.float64) * float(E.params[t][0]) / (float(s_y) * hw) for t, cf in zip(xs, A2.attrs["coef"])]
        bias = np.zeros(c) + np.asarray(A2.attrs["bias"], np.float64) / float(s_y)
        cmax = max(max(float(np.abs(r).max()) for r in rows), 1e-30)
        bmax = max(float(np.abs(bias).max()), 1e-30)
        F = int(min(40, math.floor(math.log2((2.0 ** 30) / cmax)), math.floor(math.log2((2.0 ** 30) / bmax))))
        while F > 0 and 255.0 * hw * k * cmax * 2.0 ** F >= 2.0 ** 61:
            F -= 1
        K = np.stack([np.rint(r * 2.0 ** F) for r in rows]).astype(np.int64)
        Bv = np.rint(bias * 2.0 ** F).astype(np.int64)
        name = GP.name + "+gapaffine"
        lo, hi = int(qn[GP.name].attrs.get("qmin", -128)), int(qn[GP.name].attrs.get("qmax", 127))
        node = Node("I8GapAffine", list(xs) + [const(name + "/K", K.astype(np.int32), "i32"), const(name + "/B", Bv.astype(np.int32), "i32")],
                    [GP.outputs[0]], {"zeros": [int(E.params[t][1]) for t in xs], "shift": F, "y_zero": int(z_y), "qmin": lo, "qmax": hi}, name)
        replace[qn[GP.name].name] = [node]
        remove.add(qA2.name)
        report["gap_affines"].append(name)
    if not replace:
        return qg, {"prelu_convs": [], "gated_adds": [], "gap_affines": []}
    nodes = []
    for n in qg.nodes:
        if n.name in remove:
            continue
        nodes.extend(replace.get(n.name, [n]))
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
            raise RuntimeError("fuse_gated: graph not acyclic / missing tensor: " +
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


def _assemble(qg: Graph, replace: dict, remove: set, new_consts: dict, what: str) -> Graph:
    nodes = []
    for n in qg.nodes:
        if n.name in remove:
            continue
        nodes.extend(replace.get(n.name, [n]))
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
            raise RuntimeError(what + ": graph not acyclic / missing tensor: " +
                               str([(n.name, [t for t in n.inputs if t and t not in avail]) for n in pending][:3]))
    out.nodes = order
    used = {t for n in order for t in n.inputs if t}
    out.constants = {k: v for k, v in consts.items() if k in used}
    return out


def fuse_dense(qg: Graph, Q, E, opts: dict | None = None) -> tuple[Graph, dict]:
    """int8_full v2 post-pass for token-wise residual blocks (BHViT): a binary BDense on [N, H, W, C] tokens (a 1x1 binary
    convolution) summed with a residual, then a PReLU chain:

        y = PReLU(c0 * BDense(x) + r [+ bias + beta]) [+ b2],   r = x' or Concat(x', ..., x') (k copies: r_tile k)

    becomes one I8BConv2DResI (1x1, integer epilogue, r_tile) writing y on its int8_full grid, and the bits of y for the
    binary layers that binarize it (bit_src "out"); the BDense's int8 output, the repeat Concat, the affine sum, the PReLU
    LUT and its binarize-aware twin disappear."""
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
    new_consts: dict[str, Constant] = {}

    def const(name, arr, dtype="f32"):
        a = np.ascontiguousarray(arr)
        new_consts[name] = Constant(name, dtype, tuple(a.shape), a)
        return name

    replace, remove = {}, set()
    t4_made: set = set()
    report = {"dense_blocks": []}
    for U in Q.nodes:
        if U.op != "_Unary" or U.name not in qn or qn[U.name].op != "I8Lut":
            continue
        t, y = U.inputs[0], U.outputs[0]
        A = fprod.get(t)
        if A is None or A.op not in ("_AffineSum", "Add") or len(A.inputs) != 2 or A.attrs.get("activation", "none") != "none" or \
                A.name not in qn or qn[A.name].op not in ("I8AffineSum", "I8Add"):
            continue
        # the sum's other readers: binary layers binarizing it through R3b-mapped thresholds (bits of the sum, bit_src "sum")
        sum_bins = [(u, j) for u, j in fcons[t] if u is not U]
        if any(not (u.op in ("BDense", "BConv2D") and j == 0 and u.attrs.get("input_binarize", "ge") in ("ge", "gt"))
               for u, j in sum_bins):
            continue
        shp = Q._shape(y)
        if len(shp) not in (3, 4):   # rank 3 ([N, T, C] tokens): Reshape to [N, T, 1, C] around the fused op (aliases)
            continue
        r3 = len(shp) == 3
        c = shp[-1]
        ch = _chain(Q, U.attrs["ops"], c)
        if ch is None:
            continue
        cand = [x for x in A.inputs if fprod.get(x) is not None and fprod[x].op == "BDense" and len(fcons[x]) == 1]
        if len(cand) != 1:
            continue
        xb = cand[0]
        B = fprod[xb]
        r = A.inputs[1] if A.inputs[0] == xb else A.inputs[0]
        qB = qn.get(B.name)
        if qB is None or qB.op != "I8BDense" or B.attrs.get("activation", "none") != "none" or \
                B.attrs.get("input_binarize", "ge") not in ("ge", "gt") or len(Q._shape(B.inputs[0])) != len(shp):
            continue
        if qB.name in replace:   # already reading an earlier block's bits: build on that node
            qB = replace[qB.name][0]
        coef = dict(zip(A.inputs, A.attrs["coef"])) if A.op == "_AffineSum" else {xb: 1.0, r: 1.0}
        if not np.all(np.asarray(coef[r]) == 1.0):
            continue
        # the residual: r itself, or a channel Concat of k copies of one tensor (repeat)
        qA = qn[A.name]
        rq = qA.inputs[A.inputs.index(r)]
        rtile, cat = 1, None
        p = qprod.get(rq)
        if p is not None and p.op == "I8Concat" and len(set(p.inputs)) == 1 and len(p.inputs) > 1 and \
                int(p.attrs.get("axis", -1)) in (-1, 3) and all(int(m) == 0 for m in p.attrs.get("mults", [1])):
            rtile, cat = len(p.inputs), p
            rq = p.inputs[0]
        if c % rtile:
            continue
        # readers of y: binary layers binarizing it alike (through its binarize-aware twin) -> bits; others read y
        bins = [(u, j) for u, j in fcons[y] if u.op in ("BDense", "BConv2D") and j == 0]
        if any(u.attrs.get("input_binarize", "ge") not in ("ge", "gt") for u, _ in bins):
            continue
        if sum_bins and bins:
            continue
        src_sum = bool(sum_bins)
        if src_sum:
            bins = sum_bins
        if len(bins) > 1 and not _same_binarization(Q, [u for u, _ in bins], c):
            continue
        tw = qprod.get(y + ":bin")
        beta, slope, b2 = ch
        # y read (besides the binary layers) only by a per-channel bias (BHViT's move2): the block writes that tensor instead,
        # the bias joins the out_bias and the bit thresholds of y move with it (w + b >= t + b)
        others = [(u, j) for u, j in fcons[y] if src_sum or (u, j) not in bins]
        y_out, extra = y, None
        if len(others) == 1 and others[0][0].op == "_Unary" and y not in Q.g.outputs and others[0][0].name in qn and \
                qn[others[0][0].name].op == "I8Lut" and others[0][0].outputs[0] not in Q.g.outputs and \
                others[0][0].outputs[0] + ":bin" not in qprod:
            M = others[0][0]
            ops = M.attrs["ops"]
            if len(ops) == 1 and ops[0][0] == "ChannelAffine" and (len(ops[0][2]) < 2 or not ops[0][2][1] or
                                                                   np.all(np.asarray(Q.const_f(ops[0][2][1])) == 1.0)):
                extra = _vec(Q, ops[0][2][2] if len(ops[0][2]) > 2 else "", c, 0.0)
                y_out = M.outputs[0]
                remove.add(qn[M.name].name)
        if extra is not None:
            b2 = (b2 if b2 is not None else np.zeros(c)) + extra
        s_y, z_y = E.params[y_out]
        sc = 1.0 / float(s_y)
        s_r, z_r = E.params[rq]
        c0 = np.zeros(c) + np.asarray(coef[xb], np.float64)
        bias = np.zeros(c) + (np.asarray(A.attrs["bias"], np.float64) if A.op == "_AffineSum" else 0.0)
        pm = np.zeros(c) + (B.attrs.get("_pm64") if B.attrs.get("_pm64") is not None else _vec(Q, B.inputs[2] if len(B.inputs) > 2 else "", c, 1.0))
        pa = np.zeros(c) + (B.attrs.get("_pa64") if B.attrs.get("_pa64") is not None else _vec(Q, B.inputs[3] if len(B.inputs) > 3 else "", c, 0.0))
        wq = qg.constants[qB.inputs[1]]
        ic = int(wq.shape[-1])
        wn = B.name + "/kernel_bits_1x1"
        new_consts[wn] = Constant(wn, wq.dtype, (int(wq.shape[0]), 1, 1, ic), np.asarray(wq.data))
        at = {k: v for k, v in qB.attrs.items() if k not in ("qmin", "qmax", "output")}
        at.update(strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], groups=1, pad_value=0, y_zero=int(z_y), qmin=-128, qmax=127)
        if rtile > 1:
            at["r_tile"] = rtile
        xin4, rq4, pre4 = qB.inputs[0], rq, []
        if r3:   # rank 3: the fused op works on [N, T, 1, C] views (int8 / bit Reshape: same storage, aliased by the Session)
            n0, t0 = int(shp[0]), int(shp[1])
            xin4 = qB.inputs[0] + ":t4"
            if xin4 not in t4_made:   # q / k / v read one tensor: one view
                t4_made.add(xin4)
                pre4.append(Node("Reshape", [qB.inputs[0]], [xin4], {"shape": [n0, t0, 1, int(Q._shape(B.inputs[0])[-1])]}, B.name + "/x4"))
            rq4 = rq + ":t4"
            if rq4 not in t4_made:
                t4_made.add(rq4)
                pre4.append(Node("Reshape", [rq], [rq4], {"shape": [n0, t0, 1, int(c // rtile)]}, B.name + "/r4"))
        stub = Node("I8BConv2DRes", [xin4, wn, "", "", qB.inputs[4] if len(qB.inputs) > 4 else "", rq4], [y_out], at, B.name + "+res")
        thr_u = None
        if bins:
            nx = bins[0][0]
            tn = np.zeros(c) + (np.asarray(Q.const_f(nx.inputs[4]), np.float64) if len(nx.inputs) > 4 and nx.inputs[4] else 0.0)
            stub.attrs["bit_binarize"] = nx.attrs.get("input_binarize", "ge")
            if src_sum:   # thresholds on the sum t, mapped to u = t + beta (acc)
                thr_u = (tn + beta) * sc
            else:
                thr_u = (tn + (extra if extra is not None else 0.0)) * sc
                stub.attrs["bit_src"] = "out"
        ni = _int_epilogue(stub, ic, c0 * pm * sc, (c0 * pa + bias + beta) * sc, float(s_r) * sc, int(z_r), slope,
                           (b2 * sc) if b2 is not None else None, thr_u, const, B.name)
        if ni is None:
            continue
        if rtile > 1:
            ni.attrs["r_tile"] = rtile
        bits = y + ":bits"
        post4 = []
        if r3:   # the rank-4 views of the outputs exist now: a later block reads them directly (no second view)
            ni.outputs = [y_out + ":t4"]
            t4_made.add(y_out + ":t4")
            post4.append(Node("Reshape", [y_out + ":t4"], [y_out], {"shape": [int(v) for v in shp]}, B.name + "/y3"))
        if bins:
            ni.outputs = ni.outputs[:1] + [bits + (":t4" if r3 else "")]
            if r3:
                t4_made.add(bits + ":t4")
                post4.append(Node("Reshape", [bits + ":t4"], [bits], {"shape": [int(v) for v in shp]}, B.name + "/bits3"))
        replace[qA.name] = pre4 + [ni] + post4
        remove.update({qB.name, qn[U.name].name})
        if tw is not None and not src_sum:
            remove.add(tw.name)
        if qn.get(A.name + "/bin") is not None:
            remove.add(A.name + "/bin")
        for nm in (A.name + "/bin",):   # the sum's binarize-aware twin feeds only the sum's binary readers (now bits)
            pass
        if cat is not None and len(qcons[cat.outputs[0]]) == 1:
            remove.add(cat.name)
        for u, _ in bins:   # the binary readers read the bits (no thresholds, no input grid)
            qu = qn[u.name]
            m = Node(qu.op, [bits] + list(qu.inputs[1:]), list(qu.outputs),
                     {k: v for k, v in qu.attrs.items() if k not in ("x_scale", "x_zero", "input_binarize", "x_code", "x_exp")}, qu.name)
            if len(m.inputs) > 4:
                m.inputs[4] = ""
            while m.inputs and m.inputs[-1] == "":
                m.inputs.pop()
            replace[qu.name] = [m]
        report["dense_blocks"].append(ni.name)
    if not replace:
        return qg, report
    out = _assemble(qg, replace, remove, new_consts, "fuse_dense")
    ops = defaultdict(int)
    for n in out.nodes:
        if n.op.startswith("I8"):
            ops[n.op] += 1
    report["int8_ops"] = dict(ops)
    return out, report


def cse(qg: Graph) -> tuple[Graph, dict]:
    """Common subexpressions of the emitted graph: nodes with the same operator, inputs (after renaming) and attributes compute
    the same tensors (BHViT's shift branches gather the same rolled rows twice); the later one is dropped and its readers read
    the first one's outputs. Exact (every operator is a function of its inputs and attributes)."""
    import json as _json
    seen, rename, keep = {}, {}, []
    for n in qg.nodes:
        ins = [rename.get(t, t) for t in n.inputs]
        key = (n.op, tuple(ins), _json.dumps(n.attrs, sort_keys=True, default=str), len(n.outputs))
        if key in seen and not any(o in qg.outputs for o in n.outputs):
            for a, b in zip(n.outputs, seen[key].outputs):
                rename[a] = b
            continue
        m = Node(n.op, ins, list(n.outputs), n.attrs, n.name)
        seen.setdefault(key, m)
        keep.append(m)
    if not rename:
        return qg, {"cse_removed": 0}
    out = Graph()
    out.inputs = [dict(i) for i in qg.inputs]
    out.outputs = list(qg.outputs)
    out.manifest = dict(qg.manifest)
    out.nodes = keep
    used = {t for n in keep for t in n.inputs if t}
    out.constants = {k: v for k, v in qg.constants.items() if k in used}
    return out, {"cse_removed": len(qg.nodes) - len(keep)}


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
        if len(bconv) == 2 and (opts or {}).get("code", "l8") == "lin8" and \
                all(fprod[x].attrs.get("activation", "none") == "none" for x in bconv):
            # two binary convolutions summed (ResNet-18 CIFAR downsample blocks: 3x3 + 1x1 shortcut): the one with the larger
            # kernel is fused, the other keeps its int8 output on its own grid and is the block's (linear) residual
            area = [int(np.prod(Q._shape(fprod[x].inputs[1])[1:3])) if Q.is_const(fprod[x].inputs[1]) else 0 for x in bconv]
            bconv = [bconv[0] if area[0] >= area[1] else bconv[1]]
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
    # opts["code"]: "l8" (default, model_int8_full_log2) or "lin8" (int8_full v2, model_int8_full_v2): the fused blocks write
    # their output on its int8_full linear grid (I8BConv2DRes output "i8", constants in units of the output scale, linear
    # residual codes); downsample convolutions and the stem stay linear (stem "split": the Session computes the stem GEMM
    # and its binarize-aware twin once, I8Conv2DN), so no tensor changes its code and no l8 -> linear LUT is needed
    lin = (opts or {}).get("code", "l8") == "lin8"
    int_epi = bool((opts or {}).get("int_epilogue", False))   # lin8: I8BConv2DResI where the block has no pre stage
    sfx_pm, sfx_pa = ("/pm_lin8", "/pa_lin8") if lin else ("/pm_l8", "/pa_l8")
    l8_rule = (opts or {}).get("l8_range", "minmax")
    E_of = {} if lin else {b["y"]: l8_exp(stats[b["y"]], l8_rule) for b in blocks}
    # downsample 1x1 convolutions feeding only a block's residual input -> l8 output
    gemm_l8 = {}
    for b in blocks if not lin else []:
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
    # lin8: opts["lin_stem"] "bits" (default: one GEMM with its linear output and the exact binarization of its accumulator,
    # max pooled as int8 and as bits) or "split" (the binarize-aware twin GEMM)
    stem_mode = (opts or {}).get("stem", "l8") if not lin else (opts or {}).get("lin_stem", "bits")
    if lin and stem_mode not in ("split", "bits"):
        raise ValueError("lin_stem must be split or bits")
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
    if lin:
        report = {"fused_blocks": len(blocks), "code": "lin8", "lin8_tensors": {}, "bit_inputs": []}

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
        if lin:
            s_y, z_y = E.params[y]
            e = None
            report["lin8_tensors"][y] = [float(s_y), int(z_y)]
        else:
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
            rl8 = not lin and (r in yset or r in gemm_l8)
            zr = 0 if rl8 else int(E.params[r][1])
            rc = r + ":x2"
            pre_nodes = [Node("I8Concat", [r, r], [rc], {"axis": -1, "mults": [0, 0], "shifts": [0, 0], "y_zero": zr, "zeros": [zr, zr]},
                              b["cat"].name + "/res")]
        else:
            pre_nodes = []
            rc = r
        sc = 2.0 ** -e if not lin else 1.0 / float(s_y)
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
            PM = const(B.name + sfx_pm, (pm1 * sc).astype(np.float32))
            PA = const(B.name + sfx_pa, (pa1 * sc).astype(np.float32))
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
            PM = const(B.name + sfx_pm, (b["c0"] * pm * sc).astype(np.float32))
            PA = const(B.name + sfx_pa, ((b["c0"] * pa + b["bias"] + beta) * sc).astype(np.float32))
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
        if not lin and (r in yset or r in gemm_l8 or (r in stems and stem_mode in ("l8", "l8_bits"))):
            rq = rc
            at.update(r_code="l8", r_exp=E_of[r] - e)
        else:
            s_r, z_r = E.params[r]
            rq = rc
            at.update(r_code="linear", r_scale=float(np.float32(s_r * sc)), r_zero=int(z_r))
        if lin:
            at.update(output="i8", y_zero=int(z_y), qmin=-128, qmax=127)
        else:
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
        node = Node("I8BConv2DRes", ins, outs, at, name)
        int_ok = lin and int_epi and (pre is None or (pre["gate"] is None and (pre["slope"] is None or np.all(pre["slope"] == 0))))
        if int_ok:
            # integer epilogue (I8BConv2DResI, int8_ops._bconv_res_i): the same block in integer units 2^-S of the output
            # step; None when the constants do not fit (the float form above is kept). Pre stages: an affine only (folded),
            # or a ReLU then an affine (clamp pre stage); gates and PReLU pre stages keep the fp32 epilogue.
            ws = new_consts[wname].shape if wname in new_consts else qg.constants[wname].shape
            pb_u = pmul = None
            if pre is None:
                d_u, beta_u = b["c0"] * pm * sc, (b["c0"] * pa + b["bias"] + beta) * sc
            elif pre["slope"] is None:
                d_u, beta_u = pm1 * sc, pa1 * sc
            else:
                d_u, pb_u, pmul, beta_u = pm1 * mul * sc, pa1 * mul * sc, mul, add * sc
            ni = _int_epilogue(node, int(np.prod(ws[1:])), d_u, beta_u,
                               float(s_r) * sc, int(z_r), slope, (b2 * sc) if b2 is not None else None,
                               ((tn + beta) * sc if b["src"] == "sum" else tn * sc) if b["nxt"] is not None else None,
                               const, B.name, pb_u, pmul)
            if ni is not None:
                node = ni
                report.setdefault("int_epilogue", []).append(name)
        replace[A.name] = pre_nodes + [node]
        tw = qprod.get(y + ":bin")
        if tw is not None and tw.op == "I8Lut" and tw.name not in remove and t in tw.inputs:
            # a remaining binarize-aware copy of y (read by a centering add, quantize_int8 center_split) reads the sum t,
            # which the fusion removes: index y's l8 code instead, the table maps the code to the copy's grid
            sb, zb = E.params[y + ":bin"]
            dec = int8_ops.l8_decode(np.arange(-128, 128), e) if not lin else \
                int8_ops.deq(np.arange(-128, 128).astype(np.int8), float(s_y), int(z_y))
            tab = int8_ops.quantize(dec, sb, zb).reshape(1, 256)
            replace[tw.name] = [Node("I8Lut", [y, const(tw.name + ("/from_l8" if not lin else "/from_lin8"), tab.astype(np.int8), "i8")],
                                     list(tw.outputs), {}, tw.name)]
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
    for yv in (list(yset) + list(gemm_l8)) if not lin else []:
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
