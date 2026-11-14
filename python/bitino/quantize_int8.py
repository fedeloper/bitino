"""Static full-integer quantization of a float Bitino graph (stage `int8_full`).

    stats = calibrate(graph, samples)                 # float32 executor, per-tensor min/max (+ histograms)
    qgraph, report = quantize(graph, stats, opts)     # I8* operator graph, quantization params in node attrs

Scheme (TFLite-style static quantization, docs/int8.md "Full-integer execution: int8_full"):
  * activations: per-tensor asymmetric int8, x = s*(q - z); range = calibrated [min, max] (or percentiles)
    widened to contain 0; s = (hi - lo)/255, z = round(-128 - lo/s).
  * weights of Conv2D / Dense: per-output-channel symmetric int8 (max|w|/127), int32 bias at s_x*s_w,
    int32 accumulators, fixed-point requantization (int8_ops.requant).
  * binary operators keep their bit weights; their int8 inputs are binarized exactly (integer thresholds)
    and their fp32 epilogue (post_mul/post_add, folded with the output scale) is rounded to int8.
  * tensors whose every consumer binarizes them get a "binarize-aware" range around the thresholds.

Graph rewrites applied before quantization (all exact in real arithmetic, listed in the report):
  R1 sign(Softmax(x)) operand of a BMatMul -> constant +1 (softmax > 0; checked on the calibration data:
     the float softmax output never underflowed to 0), which makes the q.k branch of such heads dead.
  R2 Add(Dense(x1), Dense(x2)) [-> ChannelAffine] -> Dense(Concat(x1, x2)) (Bi-ViT/DeiT heads).
  R3 ChannelAffine feeding binarizations only -> threshold (and sign flip) of the binarizing operand.
  R4 Bi-ViT affine-code products: AddBroadcast trees of BMatMul counts (+ row/col sums and constants)
     followed by a scalar ChannelAffine -> one I8BMatMul with integer coefficients c_ab, c_rs, c_cs, c0.
  R5 ChannelAffine after a BDense/BConv2D (single consumer) -> folded into post_mul/post_add.
  R6 chains of unary elementwise ops (ChannelAffine, Activation, FakeQuant) -> one I8Lut (256-entry
     table per tensor or per channel, computed in float64 from the composed function).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

from . import executor, int8_ops
from .ir import Constant, Graph, Node, pack_bits

UNARY = {"ChannelAffine", "Activation", "FakeQuant"}
PASS_THROUGH = {"Reshape", "Transpose", "Gather", "MaxPool2D", "ChannelShift"}


# ============================================================================ calibration
_W = {}


def _worker_init(path):
    from . import format as F
    _W["g"] = F.read(path, verify=False)


def _feeds(g, record: bytes) -> dict:
    out, off = {}, 0
    for inp in g.inputs:
        dt = {"f32": np.float32, "i32": np.int32}[inp.get("dtype", "f32")]
        n = int(np.prod(inp["shape"]))
        out[inp["name"]] = np.frombuffer(record, dt, n, off).reshape(inp["shape"])
        off += n * 4
    return out


def _act_names(g) -> list[str]:
    names = [i["name"] for i in g.inputs if i.get("dtype", "f32") == "f32"]
    for n in g.nodes:
        names.extend(n.outputs)
    return names


def _sample_minmax(record):
    g = _W["g"]
    env = executor.run(g, _feeds(g, record), keep_intermediates=True)
    out = {}
    for t in _act_names(g):
        v = env.get(t)
        if v is None or v.dtype != np.float32 or v.size == 0:
            continue
        out[t] = (float(v.min()), float(v.max()))
    return out


def _sample_hist(args):
    record, edges = args
    g = _W["g"]
    env = executor.run(g, _feeds(g, record), keep_intermediates=True)
    out = {}
    for t, (lo, hi) in edges.items():
        v = env.get(t)
        if v is None:
            continue
        h, _ = np.histogram(v, bins=2048, range=(lo, hi))
        out[t] = h
    return out


def calibrate(model_path: str, records: list[bytes], percentile: float | None = None, jobs: int = 8) -> dict:
    """Per-tensor statistics of the float32 executor (exact Bitino semantics) over the calibration records.
    Returns {tensor: {"min", "max", "lo", "hi"}}; lo/hi are min/max or the (100-p, p) percentiles."""
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs, initializer=_worker_init, initargs=(model_path,)) as pool:
        per = pool.map(_sample_minmax, records, chunksize=1)
        stats = {}
        for d in per:
            for t, (lo, hi) in d.items():
                s = stats.setdefault(t, {"min": lo, "max": hi})
                s["min"] = min(s["min"], lo)
                s["max"] = max(s["max"], hi)
        for s in stats.values():
            s["lo"], s["hi"] = s["min"], s["max"]
        if percentile is not None:
            edges = {t: (s["min"], s["max"]) for t, s in stats.items() if s["max"] > s["min"]}
            hists = pool.map(_sample_hist, [(r, edges) for r in records], chunksize=1)
            for t, (lo, hi) in edges.items():
                h = sum(d[t] for d in hists if t in d)
                c = np.cumsum(h) / max(1, h.sum())
                w = (hi - lo) / 2048
                q = (100.0 - percentile) / 100.0
                i_lo = int(np.searchsorted(c, q, side="left"))
                i_hi = int(np.searchsorted(c, 1.0 - q, side="left"))
                stats[t]["lo"] = max(lo, lo + i_lo * w)
                stats[t]["hi"] = min(hi, lo + (i_hi + 1) * w)
    return stats


def improvement_slices(g) -> dict:
    """MeliusNet improvement blocks Add(x, Pad(b)) with b zero-padded on the channel axis only:
    {Add output: (c0, c1, b's producer output)} (the channel range of the updated slice)."""
    prod = {n.outputs[0]: n for n in g.nodes}
    out = {}
    for n in g.nodes:
        if n.op != "Add" or len(n.inputs) != 2 or n.attrs.get("activation", "none") != "none":
            continue
        p = prod.get(n.inputs[1])
        if p is None or p.op != "Pad" or float(p.attrs.get("value", 0.0)) != 0.0:
            continue
        pads = p.attrs.get("pads") or []
        r = len(pads) // 2
        if r < 2 or any(pads[k] for k in range(2 * r - 2)):
            continue
        out[n.outputs[0]] = (int(pads[2 * r - 2]), p.inputs[0])
    return out


def _order_stat(desc: np.ndarray, asc: np.ndarray, n: int, percentile: float):
    """np.percentile (linear) of n values per column from their k largest (desc, descending) and k smallest (asc)."""
    pos = (n - 1) * percentile / 100.0
    j = int(np.floor(pos))
    f = pos - j
    a0, a1 = desc[n - 1 - j], desc[max(n - 2 - j, 0)]   # ascending index j, j + 1 counted from the top
    hi = a0 + f * (a1 - a0)
    pos = (n - 1) * (100.0 - percentile) / 100.0
    j = int(np.floor(pos))
    f = pos - j
    lo = asc[j] + f * (asc[min(j + 1, asc.shape[0] - 1)] - asc[j])
    return lo, hi


def pc_slice_stats(g, records, stats, percentile):
    """Per-channel and per-slice calibration of MeliusNet's improvement slices (option --pc-slices): for every
    improvement Add output y (improvement_slices), over the same records and every position, the channel's
    (100-p, p) percentiles of the updated slice c0..c1 (np.percentile, linear interpolation; min / max without
    --percentile) -> stats[y]["slice_pc_lo"], ["slice_pc_hi"] (per channel of the slice), ["slice_lo"], ["slice_hi"]
    (the slice's values taken together), and the min / max (["slice_pc_min"], ["slice_pc_max"], ["slice_min"],
    ["slice_max"]). Exact order statistics from the k smallest / largest values per record."""
    sl = improvement_slices(g)
    if not sl:
        return
    shapes = _float_shapes(g, _feeds(g, records[0]))
    width = {y: (c0, c0 + int(shapes[b][-1])) for y, (c0, b) in sl.items()}
    npos = {y: int(np.prod(shapes[y][:-1])) for y in sl}
    p = percentile if percentile else 100.0
    kc = {y: int(np.ceil((npos[y] * len(records) - 1) * (100.0 - p) / 100.0)) + 2 for y in sl}
    ks = {y: int(np.ceil((npos[y] * (width[y][1] - width[y][0]) * len(records) - 1) * (100.0 - p) / 100.0)) + 2 for y in sl}
    acc = {y: [[], [], [], []] for y in sl}

    def hook(t, v):
        if t in sl:
            c0, c1 = width[t]
            x = np.asarray(v, np.float32).reshape(-1, v.shape[-1])[:, c0:c1]
            k = min(kc[t], x.shape[0])
            acc[t][0].append(-np.partition(-x, k - 1, axis=0)[:k])
            acc[t][1].append(np.partition(x, k - 1, axis=0)[:k])
            f = x.reshape(-1)
            k = min(ks[t], f.size)
            acc[t][2].append(-np.partition(-f, k - 1)[:k])
            acc[t][3].append(np.partition(f, k - 1)[:k])
            for j, kk in ((0, kc[t]), (1, kc[t])):   # keep only the k extreme values so far
                a = np.concatenate(acc[t][j])
                acc[t][j] = [(-np.sort(-a, axis=0) if j == 0 else np.sort(a, axis=0))[:kk]]
            for j, kk in ((2, ks[t]), (3, ks[t])):
                a = np.concatenate(acc[t][j])
                acc[t][j] = [(-np.sort(-a) if j == 2 else np.sort(a))[:kk]]
        return v

    for r in records:
        executor.run(g, _feeds(g, r), post_hook=hook)
    for y in sl:
        n = npos[y] * len(records)
        hi_d = np.asarray(acc[y][0][0], np.float64)
        lo_a = np.asarray(acc[y][1][0], np.float64)
        lo, hi = _order_stat(hi_d, lo_a, n, p)
        slo, shi = _order_stat(np.asarray(acc[y][2][0], np.float64), np.asarray(acc[y][3][0], np.float64),
                               n * (width[y][1] - width[y][0]), p)
        st = stats.setdefault(y, {})
        st["slice_pc_lo"] = [float(v) for v in lo]
        st["slice_pc_hi"] = [float(v) for v in hi]
        st["slice_lo"], st["slice_hi"] = float(slo), float(shi)
        st["slice_pc_min"] = [float(v) for v in lo_a[0]]
        st["slice_pc_max"] = [float(v) for v in hi_d[0]]
        st["slice_min"], st["slice_max"] = float(acc[y][3][0][0]), float(acc[y][2][0][0])


# ============================================================================ helpers
def _cons_map(nodes):
    cons = defaultdict(list)
    for n in nodes:
        for j, t in enumerate(n.inputs):
            if t:
                cons[t].append((n, j))
    return cons


def _prod_map(nodes):
    return {o: n for n in nodes for o in n.outputs}


def _uniform(a) -> float | None:
    a = np.asarray(a, np.float64).reshape(-1)
    return float(a[0]) if a.size and np.all(a == a[0]) else None


def qparams_range(lo: float, hi: float) -> tuple[float, int]:
    """Asymmetric int8 parameters for [lo, hi] widened to contain 0 (TFLite nudging of the zero point)."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-30:
        hi = lo + 1e-30
    s = (hi - lo) / 255.0
    z = int(round(-128 - lo / s))
    return float(np.float32(s)), max(-128, min(127, z))


def qparams_binarize(thresholds: list[float], absmax: float, data_floor: bool = True) -> tuple[float, int]:
    """Binarize-aware parameters: the tensor is only ever compared with `thresholds`, so the int8 grid only has
    to resolve values near them; values beyond [min t - mu, max t + mu] saturate to codes whose dequantized
    value still lies on the same side of every threshold (so their binarization is exact). data_floor=False: the
    2^-20 data-range term only when the thresholds alone give no margin (all thresholds 0)."""
    t = np.asarray(thresholds, np.float64)
    tmin, tmax = float(t.min()), float(t.max())
    span = tmax - tmin
    mu_t = max(span * 2.0 ** -8, max(abs(tmin), abs(tmax)) * 2.0 ** -12)
    mu = max(mu_t, absmax * 2.0 ** -20, 1e-30) if (data_floor or mu_t == 0.0) else mu_t
    lo, hi = tmin - mu, tmax + mu
    s = float(np.float32((hi - lo) / 255.0))
    z = int(round(-128 - lo / s))
    # check: every threshold strictly inside the grid's dequantized range
    d = int8_ops.deq(np.array([-128, 127]), s, z)
    assert d[0] < tmin and d[1] > tmax, (thresholds, s, z, d)
    return s, z


def quantize_weights_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel int8 (axis 0): s = max|w|/127 (1 if 0), q = clamp(rint(w/s), -127, 127)."""
    w64 = np.asarray(w, np.float64).reshape(w.shape[0], -1)
    m = np.abs(w64).max(axis=1)
    s = np.where(m > 0, m / 127.0, 1.0)
    q = np.clip(np.rint(w64 / s[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
    return q, s


def quantize_const(c: np.ndarray, s: float, z: int) -> np.ndarray:
    return np.clip(np.rint(np.asarray(c, np.float64) / s) + z, -128, 127).astype(np.int8)


def sign_preserving_codes(v: np.ndarray, codes: np.ndarray, s: float, z: int, row_thresholds, radius: int = 4) -> np.ndarray:
    """Table entries for a tensor that is (also) binarized: among the codes near round(v/s)+z, pick one whose
    dequantized value lies on the same side of (or on) every threshold of that table row as the float value v, so the
    binarization of the table output equals the binarization of the function value (e.g. GELU's tiny negative
    tail, which plain rounding maps onto the zero point). row_thresholds[r] lists the thresholds of table row r."""
    out = codes.copy()
    cand = np.arange(-128, 128)
    dq = int8_ops.deq(cand, s, z).astype(np.float64)
    for r in range(v.shape[0]):
        th = np.unique(np.asarray(row_thresholds[r], np.float32)).astype(np.float64)
        if th.size == 0:
            continue
        want = np.sign(v[r][:, None] - th[None, :])                       # [256, m]
        got = np.sign(dq[out[r].astype(np.int64) + 128][:, None] - th[None, :])
        for i in np.nonzero(np.any(want != got, axis=1))[0]:
            c0 = int(out[r, i])
            best, key = c0, None
            for c in range(max(-128, c0 - radius), min(127, c0 + radius) + 1):
                d = dq[c + 128]
                k = (int(np.sum(np.sign(d - th) != want[i])), abs(d - v[r, i]))
                if key is None or k < key:
                    best, key = c, k
            out[r, i] = best
    return out


def _erf(x):
    v = np.vectorize(math.erf)
    return v(x)


def unary_eval(ops: list, x: np.ndarray, consts: dict, table: bool = False) -> np.ndarray:
    """Composite of unary elementwise ops in float64 on x[..., C] (C = channel axis, last), or on a lookup table
    x[C, 256] (table=True; per-channel constants then have shape [C, 1] whatever their axis in the tensor)."""
    for op, attrs, ins in ops:
        if op == "ChannelAffine":
            if len(ins) > 1 and ins[1]:
                x = x * np.asarray(consts[ins[1]], np.float64)
            if len(ins) > 2 and ins[2]:
                x = x + np.asarray(consts[ins[2]], np.float64)
        elif op == "Activation":
            t = attrs["type"]
            if t == "relu":
                x = np.maximum(x, 0)
            elif t == "relu6":
                x = np.clip(x, 0, 6)
            elif t == "hardtanh":
                x = np.clip(x, attrs.get("min", -1.0), attrs.get("max", 1.0))
            elif t == "prelu":
                x = np.where(x >= 0, x, x * np.asarray(consts[ins[1]], np.float64))
            elif t == "leaky_relu":
                x = np.where(x >= 0, x, x * attrs.get("alpha", 0.01))
            elif t == "sigmoid":
                x = 1.0 / (1.0 + np.exp(-x))
            elif t == "tanh":
                x = np.tanh(x)
            elif t == "gelu":
                x = x * 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
            elif t == "hardswish":
                x = x * np.clip(x + 3, 0, 6) / 6
            elif t == "sign":
                x = np.where(x >= 0, 1.0, -1.0)
            elif t == "sign_zero":
                x = np.sign(x)
            else:
                raise NotImplementedError(t)
        elif op == "FakeQuant":
            if not table and attrs.get("axis", -1) not in (-1, x.ndim - 1):
                raise NotImplementedError("FakeQuant axis")
            s = np.asarray(consts[ins[1]], np.float64)
            z = np.asarray(consts[ins[2]], np.float64)
            t = np.clip(np.rint(x * attrs.get("pre_scale", 1.0) / s + z), attrs.get("qmin", 0), attrs.get("qmax", 1))
            x = t * s
        else:
            raise NotImplementedError(op)
    return x


def flip1(n, j):
    return int(n.attrs.get("_a_flip" if j == 0 else "_b_flip", 1))


def _is_ones(c: Constant | None) -> bool:
    return c is not None and c.dtype == "f32" and np.all(np.asarray(c.data) == 1.0)


# ============================================================================ rewrites
class _NoFuse(Exception):
    pass


class _Q:
    def __init__(self, g: Graph, stats: dict):
        self.g = g
        self.stats = stats
        self.nodes = [Node(n.op, list(n.inputs), list(n.outputs), dict(n.attrs), n.name) for n in g.nodes]
        self.consts = {k: np.asarray(c.data) for k, c in g.constants.items()}  # float/i32 arrays or packed bits
        self.cdtype = {k: c.dtype for k, c in g.constants.items()}
        self.cshape = {k: tuple(c.shape) for k, c in g.constants.items()}
        self.new_consts: dict[str, Constant] = {}
        self.rewrites = defaultdict(int)
        self.notes: list[str] = []
        self._uid = 0

    def uid(self, p):
        self._uid += 1
        return f"{p}__q{self._uid}"

    def is_const(self, t):
        return t in self.g.constants or t in self.new_consts

    def const_f(self, t):
        if t in self.new_consts:
            return np.asarray(self.new_consts[t].data)
        return self.consts[t]

    def add_const(self, name, arr, dtype):
        if dtype == "bit":
            self.new_consts[name] = Constant(name, "bit", tuple(arr.shape), pack_bits(arr))
        else:
            arr = np.ascontiguousarray(arr)
            self.new_consts[name] = Constant(name, dtype, tuple(arr.shape), arr)
        return name

    # --- R1: sign(softmax) == 1
    def r1_softmax_sign(self):
        prod = _prod_map(self.nodes)
        for n in self.nodes:
            if n.op != "BMatMul":
                continue
            for j, key in ((0, "a_binarize"), (1, "b_binarize")):
                p = prod.get(n.inputs[j])
                mode = n.attrs.get(key, "sign_zero")
                if p is None or p.op != "Softmax" or mode not in ("sign_zero", "gt0", "ge"):
                    continue
                st = self.stats.get(n.inputs[j])
                if mode != "ge" and (st is None or not st["min"] > 0.0):
                    self.notes.append(f"R1 skipped for {n.name}: softmax output min {st and st['min']} is not > 0")
                    continue
                shape = self._shape(n.inputs[j])
                name = self.add_const(self.uid(n.name + "/ones"), np.ones(shape, np.float32), "f32")
                n.inputs[j] = name
                self.rewrites["R1 sign(softmax) -> 1"] += 1

    # --- R2: Add(Dense(x1), Dense(x2)) [-> ChannelAffine]
    def r2_dense_heads(self):
        prod, cons = _prod_map(self.nodes), _cons_map(self.nodes)
        for n in list(self.nodes):
            if n.op != "Add" or n.attrs.get("activation", "none") != "none":
                continue
            d = [prod.get(t) for t in n.inputs]
            if not all(x is not None and x.op == "Dense" and len(cons[x.outputs[0]]) == 1 and x.attrs.get("activation", "none") == "none"
                       for x in d):
                continue
            if not all(self.is_const(x.inputs[1]) for x in d):
                continue
            w1, w2 = (np.asarray(self.const_f(x.inputs[1]), np.float64) for x in d)
            b = sum(np.asarray(self.const_f(x.inputs[2]), np.float64) if len(x.inputs) > 2 and x.inputs[2] else 0.0 for x in d)
            w = np.concatenate([w1, w2], axis=1)
            b = np.zeros(w.shape[0]) + b
            out = n.outputs[0]
            dead = [n, d[0], d[1]]
            nxt = cons[out]
            if len(nxt) == 1 and nxt[0][0].op == "ChannelAffine" and nxt[0][1] == 0:
                ca = nxt[0][0]
                sc = np.asarray(self.const_f(ca.inputs[1]), np.float64) if len(ca.inputs) > 1 and ca.inputs[1] else 1.0
                bi = np.asarray(self.const_f(ca.inputs[2]), np.float64) if len(ca.inputs) > 2 and ca.inputs[2] else 0.0
                w = w * (np.zeros(w.shape[0]) + sc)[:, None]
                b = b * sc + bi
                out = ca.outputs[0]
                dead.append(ca)
            cat_out = self.uid(n.name + "/cat") + ":out"
            cat = Node("Concat", [d[0].inputs[0], d[1].inputs[0]], [cat_out], {"axis": -1}, self.uid(n.name + "/cat"))
            wn = self.add_const(self.uid(n.name + "/w"), w.astype(np.float32), "f32")
            bn = self.add_const(self.uid(n.name + "/b"), b.astype(np.float32), "f32")
            dense = Node("Dense", [cat_out, wn, bn], [out], {}, self.uid(n.name + "/dense"))
            idx = self.nodes.index(n)
            for x in dead:
                self.nodes.remove(x)
            self.nodes.insert(min(idx, len(self.nodes)), dense)
            self.nodes.insert(min(idx, len(self.nodes)), cat)
            self.stats[cat_out] = {k: (min if k in ("min", "lo") else max)(self.stats[d[0].inputs[0]][k], self.stats[d[1].inputs[0]][k])
                                   for k in ("min", "max", "lo", "hi")}
            self.rewrites["R2 Add(Dense, Dense) -> Dense(Concat)"] += 1
            self._toposort()

    # --- R3: ChannelAffine -> binarizing operand
    def r3_affine_to_threshold(self):
        changed = True
        while changed:
            changed = False
            prod, cons = _prod_map(self.nodes), _cons_map(self.nodes)
            for ca in self.nodes:
                if ca.op != "ChannelAffine" or ca.outputs[0] in self.g.outputs:
                    continue
                users = cons[ca.outputs[0]]
                if not users:
                    continue
                sc = self.const_f(ca.inputs[1]) if len(ca.inputs) > 1 and ca.inputs[1] else None
                bi = self.const_f(ca.inputs[2]) if len(ca.inputs) > 2 and ca.inputs[2] else None
                ok = all((u.op == "BMatMul" and j in (0, 1)) or (u.op in ("BDense", "BConv2D") and j == 0) for u, j in users)
                if not ok:
                    continue
                if all(u.op == "BMatMul" for u, _ in users):
                    s = 1.0 if sc is None else _uniform(sc)
                    b = 0.0 if bi is None else _uniform(bi)
                    if s is None or b is None or s == 0.0:
                        continue
                    keys = [("a_binarize", "_a_thr", "_a_flip") if j == 0 else ("b_binarize", "_b_thr", "_b_flip") for _, j in users]
                    if any(u.attrs.get(k[1]) is not None for (u, _), k in zip(users, keys)):
                        continue
                    modes = {u.attrs.get(k[0], "sign_zero") for (u, _), k in zip(users, keys)}
                    if s < 0 and "gt0" in modes:
                        continue
                    t = float(np.float32(-b / s))
                    for (u, j), (km, kt, kf) in zip(users, keys):
                        m = u.attrs.get(km, "sign_zero")
                        if s < 0:
                            m = {"ge": "gt", "gt": "ge", "sign_zero": "sign_zero"}[m]
                            u.attrs[kf] = -1
                        u.attrs[km] = m
                        u.attrs[kt] = t
                        u.inputs[j] = ca.inputs[0]
                    self.rewrites["R3 ChannelAffine -> BMatMul operand threshold"] += 1
                    changed = True
                else:
                    if not all(u.op in ("BDense", "BConv2D") for u, _ in users):
                        continue
                    c = self._shape(ca.outputs[0])[-1]
                    s = np.ones(c) if sc is None else np.zeros(c) + np.asarray(sc, np.float64)
                    b = np.zeros(c) if bi is None else np.zeros(c) + np.asarray(bi, np.float64)
                    if not np.all(s > 0):
                        continue
                    for u, _ in users:
                        t0 = np.asarray(self.const_f(u.inputs[4]), np.float64) if len(u.inputs) > 4 and u.inputs[4] else np.zeros(c)
                        t = ((t0 - b) / s).astype(np.float32)
                        while len(u.inputs) < 5:
                            u.inputs.append("")
                        u.inputs[4] = self.add_const(self.uid(u.name + "/thr"), t, "f32")
                        u.inputs[0] = ca.inputs[0]
                    self.rewrites["R3 ChannelAffine -> binary layer threshold"] += 1
                    changed = True
                if changed:
                    self.nodes.remove(ca)
                    break

    # --- R3b: binary layers read through strictly increasing per-channel unary chains
    def _monotone_step(self, n, c):
        """Inverse map t -> f^-1(t) (vectors [c], float64) of a strictly increasing per-channel unary node, or None."""
        if n.op == "ChannelAffine" and all((not t) or self.is_const(t) for t in n.inputs[1:]):
            sc = np.zeros(c) + (np.asarray(self.const_f(n.inputs[1]), np.float64) if len(n.inputs) > 1 and n.inputs[1] else 1.0)
            bi = np.zeros(c) + (np.asarray(self.const_f(n.inputs[2]), np.float64) if len(n.inputs) > 2 and n.inputs[2] else 0.0)
            if not np.all(sc > 0):
                return None
            return lambda t: (t - bi) / sc
        if n.op == "Activation" and n.attrs.get("type") in ("prelu", "leaky_relu"):
            if n.attrs["type"] == "prelu":
                if len(n.inputs) < 2 or not self.is_const(n.inputs[1]):
                    return None
                sl = np.zeros(c) + np.asarray(self.const_f(n.inputs[1]), np.float64)
            else:
                sl = np.zeros(c) + float(n.attrs.get("alpha", 0.01))
            if not np.all(sl > 0):
                return None
            return lambda t: np.where(t >= 0, t, t / sl)
        return None

    def r3b_monotone_bypass(self):
        """binarize(f(x) >= t) == binarize(x >= f^-1(t)) for a strictly increasing per-channel f (ChannelAffine with
        positive scale, PReLU / leaky ReLU with positive slope, and chains of them, e.g. RPReLU): the BConv2D / BDense
        reads x with the mapped thresholds. The chain stays for its other consumers (residual paths), so x becomes a
        mixed tensor whose binarize-aware second version is computed directly from the producer's int32 / fp32 result
        instead of from the rounded chain output."""
        prod = _prod_map(self.nodes)
        for u in self.nodes:
            if u.op not in ("BConv2D", "BDense"):
                continue
            x = u.inputs[0]
            c = self._shape(x)[-1]
            chain = []
            p = prod.get(x)
            while p is not None and x not in self.g.outputs:
                inv = self._monotone_step(p, c)
                if inv is None or self._shape(p.inputs[0]) != self._shape(x):
                    break
                chain.append(inv)
                x = p.inputs[0]
                p = prod.get(x)
            if not chain:
                continue
            t = np.zeros(c) + (np.asarray(self.const_f(u.inputs[4]), np.float64) if len(u.inputs) > 4 and u.inputs[4] else 0.0)
            for inv in chain:  # closest node first
                t = inv(t)
            while len(u.inputs) < 5:
                u.inputs.append("")
            u.inputs[4] = self.add_const(self.uid(u.name + "/thr_mono"), t.astype(np.float32), "f32")
            u.inputs[0] = x
            self.rewrites["R3b binary layer reads through a monotone unary chain (mapped thresholds)"] += 1

    # --- R4: affine-code BMatMul products
    def r4_code_products(self):
        changed = True
        while changed:
            changed = False
            prod, cons = _prod_map(self.nodes), _cons_map(self.nodes)
            for top in self.nodes:
                if top.op != "ChannelAffine":
                    continue
                bi = self.const_f(top.inputs[2]) if len(top.inputs) > 2 and top.inputs[2] else None
                sc = self.const_f(top.inputs[1]) if len(top.inputs) > 1 and top.inputs[1] else None
                s = 1.0 if sc is None else _uniform(sc)
                if s is None or (bi is not None and np.any(np.asarray(bi) != 0)):
                    continue
                src = top.inputs[0]
                p = prod.get(src)
                if p is None or p.op not in ("BMatMul", "AddBroadcast") or len(cons[src]) != 1:
                    continue
                if self._fuse_tree(p, s, [top], prod, cons):
                    changed = True
                    break
        # every remaining BMatMul (not in a recognised tree) becomes a stand-alone I8BMatMul
        prod, cons = _prod_map(self.nodes), _cons_map(self.nodes)
        for n in list(self.nodes):
            if n.op == "BMatMul":
                self._fuse_tree(n, 1.0, [], prod, cons)

    def _fuse_tree(self, root, mul, absorbed, prod, cons) -> bool:
        leaves, inner = [], []
        if not self._collect(root, prod, cons, leaves, inner):
            return False
        res = self._combine(leaves)
        if res is None:
            return False
        main, coef, extra = res
        c_ab, c_rs, c_cs, c0 = coef
        k = self._shape(main.inputs[0])[-1]
        ma, mb = main.attrs.get("a_binarize", "sign_zero"), main.attrs.get("b_binarize", "sign_zero")
        if _is_ones(self._cobj(main.inputs[0])) and ma != "gt" and main.attrs.get("_a_thr") is None:
            c_cs, c0, c_ab, c_rs = c_cs + c_ab * flip1(main, 0), c0 + c_rs * k * flip1(main, 0), 0, 0   # A == 1: AB = CS, RS = K
        elif _is_ones(self._cobj(main.inputs[1])) and mb != "gt" and main.attrs.get("_b_thr") is None:
            c_rs, c0, c_ab, c_cs = c_rs + c_ab * flip1(main, 1), c0 + c_cs * k * flip1(main, 1), 0, 0   # B == 1: AB = RS, CS = K
        coef = (c_ab, c_rs, c_cs, c0)
        out = (absorbed[-1] if absorbed else root).outputs[0]
        q = Node("_BMatMulQ", [main.inputs[0], main.inputs[1]], [out],
                 dict(main.attrs, c_ab=coef[0], c_rs=coef[1], c_cs=coef[2], c0=coef[3], mul=float(mul) * extra), main.name)
        q.attrs.pop("scale", None)
        idx = min(self.nodes.index(x) for x in absorbed + inner)
        for x in absorbed + inner:
            if x in self.nodes:
                self.nodes.remove(x)
        self.nodes.insert(idx, q)
        self.rewrites["R4 BMatMul (+row/col sums, constants, scale) -> I8BMatMul" if (absorbed or len(leaves) > 1)
                      else "BMatMul -> I8BMatMul"] += 1
        return True

    def _collect(self, n, prod, cons, leaves, inner):
        inner.append(n)
        if n.op == "BMatMul":
            leaves.append(("bmm", n))
            return True
        if n.op != "AddBroadcast":
            return False
        for t in n.inputs:
            if self.is_const(t):
                c = np.asarray(self.const_f(t), np.float64).reshape(-1)
                if c.size != 1:
                    return False
                leaves.append(("const", float(c[0])))
                continue
            p = prod.get(t)
            if p is None or len(cons[t]) != 1 or p.op not in ("BMatMul", "AddBroadcast"):
                return False
            if not self._collect(p, prod, cons, leaves, inner):
                return False
        return True

    def _combine(self, leaves):
        """leaves -> (main BMatMul, (c_ab, c_rs, c_cs, c0), extra multiplier) or None."""
        bmms = [n for k, n in leaves if k == "bmm"]
        consts = [v for k, v in leaves if k == "const"]

        def flip(n, j):
            return int(n.attrs.get("_a_flip" if j == 0 else "_b_flip", 1))

        if len(bmms) == 1 and not consts:               # single product: any (also non-integer) scale
            m = bmms[0]
            return m, (flip(m, 0) * flip(m, 1), 0, 0, 0), float(m.attrs.get("scale", 1.0))
        mains = [n for n in bmms if not self.is_const(n.inputs[0]) and not self.is_const(n.inputs[1])]
        if len(mains) != 1:
            return None
        m = mains[0]
        sm = float(m.attrs.get("scale", 1.0))
        if not sm.is_integer() or not float(sum(consts)).is_integer():
            return None
        c_ab, c_rs, c_cs = int(sm) * flip(m, 0) * flip(m, 1), 0, 0
        for n in bmms:
            if n is m:
                continue
            s = float(n.attrs.get("scale", 1.0))
            if not s.is_integer():
                return None
            same_a = n.inputs[0] == m.inputs[0] and n.attrs.get("a_binarize") == m.attrs.get("a_binarize") and \
                n.attrs.get("_a_thr") == m.attrs.get("_a_thr")
            same_b = n.inputs[1] == m.inputs[1] and n.attrs.get("b_binarize") == m.attrs.get("b_binarize") and \
                n.attrs.get("_b_thr") == m.attrs.get("_b_thr")
            tb = int(n.attrs.get("transpose_b", 0)) == int(m.attrs.get("transpose_b", 0))
            if same_a and tb and _is_ones(self._cobj(n.inputs[1])) and n.attrs.get("b_binarize", "sign_zero") in ("sign_zero", "ge", "gt0"):
                c_rs += int(s) * flip(n, 0)                     # row sums: sum_k bin(a)
            elif same_b and tb and _is_ones(self._cobj(n.inputs[0])) and n.attrs.get("a_binarize", "sign_zero") in ("sign_zero", "ge", "gt0"):
                c_cs += int(s) * flip(n, 1)                     # column sums: sum_k bin(b)
            else:
                return None
        return m, (c_ab, c_rs, c_cs, int(sum(consts))), 1.0

    def _cobj(self, t):
        if t in self.new_consts:
            return self.new_consts[t]
        return self.g.constants.get(t)

    # --- R5: ChannelAffine after a binary layer
    def r5_affine_into_epilogue(self):
        changed = True
        while changed:
            changed = False
            prod, cons = _prod_map(self.nodes), _cons_map(self.nodes)
            for ca in self.nodes:
                if ca.op != "ChannelAffine":
                    continue
                p = prod.get(ca.inputs[0])
                if p is None or p.op not in ("BDense", "BConv2D") or len(cons[ca.inputs[0]]) != 1:
                    continue
                if p.attrs.get("activation", "none") != "none" or p.attrs.get("output", "f32") != "f32":
                    continue
                o = self._shape(p.outputs[0])[-1]
                sc = np.zeros(o) + (np.asarray(self.const_f(ca.inputs[1]), np.float64) if len(ca.inputs) > 1 and ca.inputs[1] else 1.0)
                bi = np.zeros(o) + (np.asarray(self.const_f(ca.inputs[2]), np.float64) if len(ca.inputs) > 2 and ca.inputs[2] else 0.0)
                pm = np.zeros(o) + (np.asarray(self.const_f(p.inputs[2]), np.float64) if len(p.inputs) > 2 and p.inputs[2] else 1.0)
                pa = np.zeros(o) + (np.asarray(self.const_f(p.inputs[3]), np.float64) if len(p.inputs) > 3 and p.inputs[3] else 0.0)
                while len(p.inputs) < 4:
                    p.inputs.append("")
                p.attrs["_pm64"] = pm * sc
                p.attrs["_pa64"] = pa * sc + bi
                p.inputs[2] = self.add_const(self.uid(p.name + "/pm"), (pm * sc).astype(np.float32), "f32")
                p.inputs[3] = self.add_const(self.uid(p.name + "/pa"), (pa * sc + bi).astype(np.float32), "f32")
                p.outputs[0] = ca.outputs[0]
                self.nodes.remove(ca)
                self.rewrites["R5 ChannelAffine -> binary layer epilogue"] += 1
                changed = True
                break

    # --- R7: trees of Add / ChannelAffine -> one integer affine sum
    def _affine_node(self, n):
        if n.op == "Add":
            return n.attrs.get("activation", "none") == "none"
        if n.op == "ChannelAffine":
            return all((not t) or self.is_const(t) for t in n.inputs[1:])
        return False

    def r7_affine_trees(self):
        """Maximal trees of Add / ChannelAffine whose inner tensors have one consumer become one I8AffineSum:
        y[c] = sum_leaf a_leaf[c] * leaf[c] + b[c], evaluated with int32 accumulation and one requantization
        (instead of one int8 rounding per Add / ChannelAffine). Trees with a single node stay I8Add / I8Lut."""
        prod, cons = _prod_map(self.nodes), _cons_map(self.nodes)

        def absorbable(t):
            p = prod.get(t)
            return (p is not None and self._affine_node(p) and len(cons[t]) == 1 and t not in self.g.outputs
                    and not self.is_const(t))

        roots = []
        for n in self.nodes:
            if n.op not in ("Add", "ChannelAffine"):
                continue
            if n.op == "Add" and n.attrs.get("activation", "none") not in ("none", "relu", "relu6"):
                continue
            if n.op == "ChannelAffine" and not self._affine_node(n):
                continue
            if absorbable(n.outputs[0]) and self._affine_node(cons[n.outputs[0]][0][0]) and \
                    self._shape(cons[n.outputs[0]][0][0].outputs[0]) == self._shape(n.outputs[0]):
                continue  # inner node of a larger tree (its only consumer is an Add / ChannelAffine)
            roots.append(n)
        done = set()
        for root in roots:
            c = self._shape(root.outputs[0])[-1]
            shape = self._shape(root.outputs[0])
            inner = []

            def expand(n):
                """-> (dict leaf -> coef vector [c], bias vector [c]) of node n's output."""
                inner.append(n)
                if n.op == "ChannelAffine":
                    sc = np.zeros(c) + (np.asarray(self.const_f(n.inputs[1]), np.float64) if len(n.inputs) > 1 and n.inputs[1] else 1.0)
                    bi = np.zeros(c) + (np.asarray(self.const_f(n.inputs[2]), np.float64) if len(n.inputs) > 2 and n.inputs[2] else 0.0)
                    terms, b = term(n.inputs[0])
                    return {k: v * sc for k, v in terms.items()}, b * sc + bi
                ta, ba = term(n.inputs[0])
                tb, bb = term(n.inputs[1])
                for k, v in tb.items():
                    ta[k] = ta.get(k, 0.0) + v
                return ta, ba + bb

            def term(t):
                if absorbable(t) and self._shape(t) == shape:
                    return expand(prod[t])
                if self.is_const(t):
                    cv = np.asarray(self.const_f(t), np.float64)
                    if cv.shape != tuple(shape):
                        raise _NoFuse()
                return {t: np.ones(c)}, np.zeros(c)

            try:
                if root.op == "Add" and any(self.is_const(t) and np.asarray(self.const_f(t)).shape != tuple(shape) for t in root.inputs):
                    continue
                terms, b = expand(root)
            except _NoFuse:
                continue
            if len(inner) < 2 or not any(x.op == "Add" for x in inner):
                continue
            if any(id(x) in done for x in inner):
                continue
            leaves = list(terms)
            q = Node("_AffineSum", leaves, [root.outputs[0]], {"coef": [terms[k] for k in leaves], "bias": b,
                                                                 "activation": root.attrs.get("activation", "none")},
                     root.name + ("+" + str(len(inner) - 1) if len(inner) > 1 else ""))
            idx = min(self.nodes.index(x) for x in inner)
            for x in inner:
                done.add(id(x))
                self.nodes.remove(x)
            self.nodes.insert(idx, q)
            self.rewrites["R7 Add/ChannelAffine tree -> I8AffineSum"] += 1
            self.rewrites["R7 nodes absorbed into affine sums"] += len(inner) - 1
        self._toposort()

    # --- R6: unary chains
    def r6_unary_chains(self):
        prod, cons = _prod_map(self.nodes), _cons_map(self.nodes)
        out_nodes = []
        absorbed = set()
        for n in self.nodes:
            if id(n) in absorbed:
                continue
            if n.op not in UNARY or (n.op == "Activation" and n.attrs.get("type") == "prelu" and not self.is_const(n.inputs[1])):
                out_nodes.append(n)
                continue
            if any(t and not self.is_const(t) for t in n.inputs[1:]):
                out_nodes.append(n)
                continue
            chain = [n]
            cur = n
            axis = self._unary_axis(n)
            while True:
                t = cur.outputs[0]
                us = cons[t]
                if t in self.g.outputs or len(us) != 1 or us[0][1] != 0:
                    break
                u = us[0][0]
                if u.op not in UNARY or any(x and not self.is_const(x) for x in u.inputs[1:]):
                    break
                ua = self._unary_axis(u)
                if ua is not None and axis is not None and ua != axis:
                    break  # per-channel along different axes: separate tables
                axis = ua if axis is None else axis
                chain.append(u)
                absorbed.add(id(u))
                cur = u
            q = Node("_Unary", [n.inputs[0]], [chain[-1].outputs[0]], {"ops": [(c.op, c.attrs, c.inputs) for c in chain]},
                     chain[0].name if len(chain) == 1 else chain[0].name + "+" + str(len(chain) - 1))
            out_nodes.append(q)
            self.rewrites["R6 unary chain -> I8Lut"] += 1
            if len(chain) > 1:
                self.rewrites["R6 ops absorbed into chains"] += len(chain) - 1
        self.nodes = out_nodes

    def _unary_axis(self, n):
        """Channel axis (non-negative) of a per-channel unary op, None if its constants are uniform."""
        rank = len(self._shape(n.inputs[0]))
        nonuni = any(t and self.is_const(t) and np.asarray(self.const_f(t)).size > 1 and _uniform(self.const_f(t)) is None
                     for t in n.inputs[1:])
        if not nonuni:
            return None
        if n.op == "FakeQuant":
            return n.attrs.get("axis", -1) % rank
        return rank - 1

    def dce(self):
        live = set(self.g.outputs)
        keep = []
        for n in reversed(self.nodes):
            if any(o in live for o in n.outputs):
                keep.append(n)
                live.update(t for t in n.inputs if t)
            else:
                self.rewrites["dead nodes removed"] += 1
        self.nodes = list(reversed(keep))

    def _toposort(self):
        avail = {i["name"] for i in self.g.inputs} | set(self.g.constants) | set(self.new_consts)
        out, pending = [], list(self.nodes)
        while pending:
            progressed = False
            for n in list(pending):
                if all((not t) or t in avail for t in n.inputs):
                    out.append(n)
                    avail.update(n.outputs)
                    pending.remove(n)
                    progressed = True
            if not progressed:
                raise RuntimeError("graph is not acyclic after rewrite")
        self.nodes = out

    def _shape(self, t):
        if self.is_const(t):
            return tuple(self._cobj(t).shape)
        if t.endswith(":bin"):
            t = t[:-4]
        return self.shapes[t]


# ============================================================================ quantization
def _float_shapes(g: Graph, record_feeds) -> dict:
    env = executor.run(g, record_feeds, keep_intermediates=True)
    return {k: tuple(v.shape) for k, v in env.items()}


def quantize(g: Graph, stats: dict, shapes: dict, opts: dict | None = None):
    """Returns (int8 Graph, report dict)."""
    opts = dict(opts or {})
    return _Emitter(rewrite(g, stats, shapes, opts), opts).emit()


def rewrite(g: Graph, stats: dict, shapes: dict, opts: dict | None = None) -> "_Q":
    """The rewritten float-level graph (R1-R7) that the emitter lowers to I8 operators."""
    opts = dict(opts or {})
    Q = _Q(g, dict(stats))
    Q.shapes = dict(shapes)
    Q.r1_softmax_sign()
    Q.dce()
    Q.r2_dense_heads()
    Q.r3_affine_to_threshold()
    Q.r4_code_products()
    Q.r5_affine_into_epilogue()
    if (opts or {}).get("monotone_bypass", True):
        Q.r3b_monotone_bypass()
        Q.dce()
    if (opts or {}).get("affine_trees", True):
        Q.r7_affine_trees()
    Q.r6_unary_chains()
    Q.dce()
    Q._toposort()
    for n in Q.nodes:
        if n.op == "Concat" and n.outputs[0] not in Q.shapes:
            sh = [Q._shape(t) for t in n.inputs]
            ax = n.attrs.get("axis", -1) % len(sh[0])
            s = list(sh[0])
            s[ax] = sum(x[ax] for x in sh)
            Q.shapes[n.outputs[0]] = tuple(s)
    return Q


class _Emitter:
    def __init__(self, Q: _Q, opts: dict):
        self.Q = Q
        self.opts = opts
        self.g = Graph()
        self.g.manifest = dict(Q.g.manifest)
        self.params: dict[str, tuple[float, int]] = {}
        self.left_float: dict[str, list] = defaultdict(list)
        self.int8_ops: dict[str, int] = defaultdict(int)
        self.f32_out: set[str] = set()
        self._bt_memo: dict = {}

    # ---------------------------------------------------------------- parameters
    def stat(self, t):
        s = self.Q.stats.get(t)
        if s is None:
            raise KeyError(f"no calibration statistics for tensor {t}")
        return s

    def binarize_thresholds(self, t, cons, prod, seen=None) -> list | None:
        """Thresholds of every binarization that consumes t (through pass-through ops), or None if some
        consumer is not a binarization."""
        users = cons.get(t, [])
        if not users or t in self.Q.g.outputs:
            return None
        th = []
        for u, j in users:
            if u.op in ("BDense", "BConv2D") and j == 0:
                if len(u.inputs) > 4 and u.inputs[4]:
                    th.extend(np.asarray(self.Q.const_f(u.inputs[4]), np.float64).reshape(-1).tolist())
                else:
                    th.append(0.0)
            elif u.op == "_BMatMulQ" and j in (0, 1):
                th.append(float(u.attrs.get("_a_thr" if j == 0 else "_b_thr", 0.0) or 0.0))
            elif u.op in ("Reshape", "Transpose", "Concat", "ChannelShift") or (u.op == "Gather" and j == 0):
                sub = self._bt_memo.get(u.outputs[0], 0)
                if sub == 0:
                    sub = self.binarize_thresholds(u.outputs[0], cons, prod)
                    self._bt_memo[u.outputs[0]] = sub
                if sub is None:
                    return None
                th.extend(sub)
            elif u.op == "Mul":
                # sign(a*b) = sign(a)*sign(b): an operand of a product that is only compared with 0 is itself
                # only compared with 0 (BiBERT's GELU * keep-mask before a sign_zero BDense)
                sub = self._bt_memo.get(u.outputs[0], 0)
                if sub == 0:
                    sub = self.binarize_thresholds(u.outputs[0], cons, prod)
                    self._bt_memo[u.outputs[0]] = sub
                if sub is None or any(v != 0.0 for v in sub):
                    return None
                th.append(0.0)
            else:
                return None
        return sorted(set(th))

    def row_thresholds(self, t, rows: int, last_axis: bool):
        """Per table row thresholds of the binarizing consumers of t: a direct BDense/BConv2D threshold vector applies
        channel-wise to a per-channel table on the last axis; anything else contributes its values to every row.
        None when there are no binarizing consumers or too many distinct values to check per row."""
        per_row = [[] for _ in range(rows)]
        common = []
        found = False
        for u, j in self.cons.get(t, []):
            if u.op in ("BDense", "BConv2D") and j == 0:
                found = True
                vec = np.asarray(self.Q.const_f(u.inputs[4]), np.float64).reshape(-1) if len(u.inputs) > 4 and u.inputs[4] else None
                if vec is None:
                    common.append(0.0)
                elif rows > 1 and last_axis and vec.size == rows:
                    for r in range(rows):
                        per_row[r].append(float(vec[r]))
                else:
                    common.extend(np.unique(vec).tolist())
        rest = self.partial_thresholds(t, direct=False)
        if rest:
            found = True
            common.extend(rest)
        if not found:
            return None
        common = sorted(set(common))
        if len(common) > 64:
            return None
        return [per_row[r] + common for r in range(rows)]

    def partial_thresholds(self, t, seen=None, direct=True) -> list:
        """Thresholds of the binarizing consumers of t (directly or through pass-through ops), ignoring the others."""
        seen = set() if seen is None else seen
        if t in seen:
            return []
        seen.add(t)
        th = []
        for u, j in self.cons.get(t, []):
            if u.op in ("BDense", "BConv2D") and j == 0:
                if not direct:
                    continue
                if len(u.inputs) > 4 and u.inputs[4]:
                    th.extend(np.asarray(self.Q.const_f(u.inputs[4]), np.float64).reshape(-1).tolist())
                else:
                    th.append(0.0)
            elif u.op == "_BMatMulQ" and j in (0, 1):
                th.append(float(u.attrs.get("_a_thr" if j == 0 else "_b_thr", 0.0) or 0.0))
            elif u.op in ("Reshape", "Transpose", "Concat", "ChannelShift") or (u.op == "Gather" and j == 0):
                th.extend(self.partial_thresholds(u.outputs[0], seen))
        return sorted(set(th))

    def bin_absmax(self, st) -> float:
        """Data magnitude for the 2^-20 margin floor of a binarize-aware grid (qparams_binarize): the calibration min / max
        (default), or option bin_margin "lohi": the clipped (percentile) range, so that rare giant values (BNAS-D: -7e9
        in a binarize-only BatchNorm output) do not widen the grid around the thresholds."""
        if self.opts.get("bin_margin", "minmax") == "lohi":
            return max(abs(st["lo"]), abs(st["hi"]))
        return max(abs(st["min"]), abs(st["max"]))

    def qbin(self, th, absmax, t=None):
        """qparams_binarize with option bin_margin "thr": no data-range margin when the thresholds give one; option
        bin_relu_clip: thresholds below 0 of a ReLU output t (producer activation relu / relu6) are dropped (every value
        is >= 0 > t: such a binarization is constant and exact on any grid that holds 0; BNAS-D: dead BatchNorm channels
        put thresholds near -7e5 next to ones in [-31, 9], so the grid of a binary convolution output had a step of 2703)."""
        if t is not None and self.opts.get("bin_relu_clip", False):
            p = self.prod.get(self.root_of(t) if hasattr(self, "split") else t)
            if p is not None and p.attrs.get("activation", "none") in ("relu", "relu6"):
                th = [v for v in th if v >= 0.0] or [0.0]
        if t is not None and self.opts.get("bin_clip_far", False):
            # thresholds more than twice the calibrated span beyond the calibrated min / max (dead BatchNorm channels:
            # +4.7e4 against data in [0, 403]) are dropped: those binarizations are constant on every calibration sample, a
            # value beyond such a threshold saturates on the grid and binarizes as if below it (an approximation)
            st = self.Q.stats.get(self.root_of(t)) or self.Q.stats.get(t)
            if st is not None:
                lo, hi = float(st["min"]), float(st["max"])
                w = 2.0 * max(hi - lo, 1e-30)
                kept = [v for v in th if lo - w <= v <= hi + w]
                th = kept or [min(max(0.0, lo), hi)]
        return qparams_binarize(th, absmax, data_floor=self.opts.get("bin_margin", "minmax") != "thr")

    def assign_params(self):
        Q = self.Q
        cons, prod = _cons_map(Q.nodes), _prod_map(Q.nodes)
        self.cons, self.prod = cons, prod
        # tensors produced in f32 (graph outputs of ops with an f32-output mode)
        f32_capable = {"Dense", "Conv2D", "MatMul", "BDense", "_BMatMulQ", "Softmax", "_Unary"}
        for n in Q.nodes:
            if n.outputs[0] in Q.g.outputs and n.op in f32_capable and len(cons[n.outputs[0]]) == 0:
                self.f32_out.add(n.outputs[0])
        acts = [i["name"] for i in Q.g.inputs if i.get("dtype", "f32") == "f32"]
        for n in Q.nodes:
            acts.extend(o for o in n.outputs if o not in self.f32_out)
        self.binarize_only = {}
        for t in acts:
            th = self.binarize_thresholds(t, cons, prod)
            st = Q.stats.get(t)
            if th is not None and st is not None:
                self.binarize_only[t] = th
                self.params[t] = self.qbin(th, self.bin_absmax(st), t)
            elif st is not None:
                p = prod.get(t)
                if p is not None and p.op == "Softmax":
                    self.params[t] = qparams_range(0.0, st["hi"])
                else:
                    self.params[t] = qparams_range(st["lo"], st["hi"])
        self._own_params = dict(self.params)   # before the Concat unification (per-part grids, plan_parts)
        # Concat: unify single-consumer inputs with the output (plain copy instead of a requantization)
        for n in reversed(Q.nodes):
            if n.op != "Concat":
                continue
            out = n.outputs[0]
            ins = [t for t in n.inputs]
            if out in self.binarize_only:
                th = list(self.binarize_only[out])
                am = max(self.bin_absmax(Q.stats[t]) for t in ins if t in Q.stats)
                self.params[out] = self.qbin(th, am)
            else:
                los = [Q.stats[t]["lo"] for t in ins if t in Q.stats] + [Q.stats.get(out, {"lo": 0})["lo"]]
                his = [Q.stats[t]["hi"] for t in ins if t in Q.stats] + [Q.stats.get(out, {"hi": 0})["hi"]]
                self.params[out] = qparams_range(min(los), max(his))
            for t in ins:
                p = prod.get(t)
                if Q.is_const(t) or p is None or p.op in PASS_THROUGH or len(cons[t]) != 1:
                    continue
                self.params[t] = self.params[out]
        # pass-through ops keep the parameters of their input. Option pool_grid_fix (default on since 2026-11-04):
        # AvgPool2D / Pad outputs too, here and not only at emission (emit_node sets them to the input grid), so that a
        # pass-through op after them (AvgPool2D -> Reshape -> Dense in the Larq heads and SE gates) gets the grid its
        # codes are actually on. Without it (pool_grid_fix=False: the int8_full models shipped before 2026-11-04) the
        # Reshape keeps the AvgPool output's calibrated grid and its consumers misread the codes (docs/int8.md,
        # "log2 residual stream beyond ReActNet-18", Larq collapses).
        keep = PASS_THROUGH | ({"AvgPool2D", "Pad"} if self.opts.get("pool_grid_fix", True) else set())
        # option maxpool_range: a tensor read only by a MaxPool2D (whose output keeps its grid) gets the union of its own
        # range and the pool output's (the max pool's percentile tail is higher: bitorch ResNet-E18's stem, conv 0..108 vs
        # pool 0..152 at p99.99, clipped at 108 before)
        # option stem_minmax: convolutions reading the graph input use the calibration min / max instead of the percentile
        # range (bitorch models: the ReLU stem's p99.99 is 84-135 against a max of 270-433, and 20 % of a dev image's stem
        # outputs saturated)
        if self.opts.get("stem_minmax", False):
            ins0 = {i["name"] for i in Q.g.inputs}
            for n in Q.nodes:
                x, y = n.inputs[0], n.outputs[0]
                if n.op == "Conv2D" and x in ins0 and y in Q.stats and y not in self.binarize_only:
                    self.params[y] = qparams_range(Q.stats[y]["min"], Q.stats[y]["max"])
        if self.opts.get("maxpool_range", False):
            for n in Q.nodes:
                x, y = n.inputs[0], n.outputs[0]
                if n.op == "MaxPool2D" and len(cons.get(x, [])) == 1 and x not in self.binarize_only and \
                        x in Q.stats and y in Q.stats and prod.get(x) is not None and prod[x].op not in keep:
                    sx, sy = Q.stats[x], Q.stats[y]
                    self.params[x] = qparams_range(min(sx["lo"], sy["lo"]), max(sx["hi"], sy["hi"]))
        for n in Q.nodes:
            if n.op in keep and not Q.is_const(n.inputs[0]) and n.inputs[0] in self.params:
                self.params[n.outputs[0]] = self.params[n.inputs[0]]
        # a product compared only with 0: output grid = product of the input grids (multiplier 1, zero point 0),
        # so every non-zero integer product keeps its sign
        for n in Q.nodes:
            y = n.outputs[0]
            if n.op == "Mul" and y in self.binarize_only and all(v == 0.0 for v in self.binarize_only[y]) and \
                    not any(Q.is_const(t) for t in n.inputs):
                sa, sb = self.params[n.inputs[0]][0], self.params[n.inputs[1]][0]
                self.params[y] = (float(np.float32(sa * sb)), 0)
        self.pc: dict[str, tuple] = {}
        if self.opts.get("concat_parts", False) or self.opts.get("pc_stem", False) or self.opts.get("pc_bin", False):
            self.plan_parts()
        self.plan_splits()

    def plan_parts(self):
        """Per-part grids (option concat_parts; DenseNet / MeliusNet): a channel Concat whose readers can all absorb
        per-channel input grids keeps every input's codes on that input's own grid (plain copies, no requantization),
        so the tensor has one (scale, zero) per channel (self.pc[t] = (s[C], z[C])). Absorbing readers: per-channel
        I8Lut (one table row per channel, computed with that channel's grid), binarizations ge / gt (per-channel integer
        code thresholds: x_scale 1, x_zero 0, threshold T[c]), Concat / MaxPool2D (copies; their outputs are per-part
        too) and the MeliusNet improvement Add(x, Pad(y)) with y padded on the channel axis (I8AffineSum with
        per-channel coefficients: the channels outside y are exact copies of x's codes)."""
        Q, cons, prod = self.Q, self.cons, self.prod

        def last_axis(t):
            return len(Q._shape(t)) - 1

        def pad_slice(n):
            """(c0, c1): channel range of the padded tensor holding its input (Pad only on the last axis), else None."""
            pads = n.attrs.get("pads")
            r = len(Q._shape(n.inputs[0]))
            # pads: (before, after) per axis
            if pads is None or len(pads) != 2 * r or any(pads[k] for k in range(2 * r - 2)) or \
                    float(n.attrs.get("value", 0.0)) != 0.0:
                return None
            c = Q._shape(n.inputs[0])[-1]
            return int(pads[2 * r - 2]), int(pads[2 * r - 2]) + c

        cand = set()
        for n in Q.nodes:
            y = n.outputs[0]
            if y in Q.g.outputs or y in self.f32_out:
                continue
            if n.op == "Concat" and n.attrs.get("axis", -1) % len(Q._shape(y)) == last_axis(y) and \
                    not any(Q.is_const(t) for t in n.inputs):
                cand.add(y)
            elif n.op == "MaxPool2D" and n.inputs[0] in cand:
                cand.add(y)
            elif n.op == "BConv2D" and self.opts.get("pc_bin") and n.attrs.get("activation", "none") == "relu" and \
                    y in self.binarize_only and cons.get(y) and \
                    all(u.op in ("BConv2D", "BDense") and j == 0 for u, j in cons[y]):
                cand.add(y)   # per-channel binarize-aware grids of a ReLU binary convolution read only by binarizations
            elif n.op == "Conv2D" and self.opts.get("pc_stem") and n.attrs.get("activation", "none") == "relu" and \
                    "pc_hi" in Q.stats.get(y, {}) and y not in self.binarize_only:
                cand.add(y)   # per-channel ReLU stem grid (option pc_stem: per-channel calibration of the stem output)
            elif n.op == "Add" and n.attrs.get("activation", "none") == "none" and n.inputs[0] in cand and \
                    prod.get(n.inputs[1]) is not None and prod[n.inputs[1]].op == "Pad" and pad_slice(prod[n.inputs[1]]) and \
                    len(cons[n.inputs[1]]) == 1:
                cand.add(y)

        def absorbs(u, j):
            if u.op == "_Unary" and j == 0:
                axes = {Q._unary_axis(Node(op, list(ins), ["_"], attrs, "_")) for op, attrs, ins in u.attrs["ops"]}
                return axes <= {None, last_axis(u.inputs[0])} and u.outputs[0] not in self.f32_out
            if u.op in ("BConv2D", "BDense") and j == 0:
                return u.attrs.get("input_binarize", "ge") in ("ge", "gt")
            if u.op in ("Concat", "MaxPool2D"):
                return u.outputs[0] in cand
            if u.op == "Add":
                return j == 0 and u.outputs[0] in cand
            if u.op == "_AffineSum":   # per-channel coefficients absorb the input grids (uniform output)
                return u.outputs[0] not in cand and u.outputs[0] not in self.f32_out and \
                    Q._shape(u.inputs[j])[-1] == Q._shape(u.outputs[0])[-1]
            return False

        changed = True
        while changed:
            changed = False
            for t in sorted(cand):
                if not all(absorbs(u, j) for u, j in cons.get(t, [])):
                    cand.discard(t)
                    changed = True
            for n in Q.nodes:   # an Add / MaxPool needs its per-part input
                if n.outputs[0] in cand and n.op in ("MaxPool2D", "Add") and n.inputs[0] not in cand:
                    cand.discard(n.outputs[0])
                    changed = True
        if not cand:
            return

        def vec(t):
            if t in self.pc:
                return self.pc[t]
            if t in cand:
                raise RuntimeError("per-part grid of " + t + " used before it is planned")
            s, z = self.params[t]
            c = Q._shape(t)[-1]
            return np.full(c, s, np.float64), np.full(c, z, np.int64)

        for n in Q.nodes:
            y = n.outputs[0]
            if y not in cand:
                continue
            if n.op == "Concat":
                for t in n.inputs:   # parts keep their own grid (undo the Concat unification; pass-through outputs keep
                    # their input's grid, which the unification never changed)
                    pp = prod.get(t)
                    if t not in cand and t in self._own_params and self.params.get(t) != self._own_params[t] and \
                            pp is not None and pp.op not in PASS_THROUGH | {"AvgPool2D", "Pad"}:
                        self.params[t] = self._own_params[t]
                vs = [vec(t) for t in n.inputs]
                self.pc[y] = (np.concatenate([v[0] for v in vs]), np.concatenate([v[1] for v in vs]))
            elif n.op == "MaxPool2D":
                self.pc[y] = self.pc[n.inputs[0]]
            elif n.op == "BConv2D":   # grid [0, max threshold of the channel + margin], zero point -128 shared
                c = Q._shape(y)[-1]
                tc = [[] for _ in range(c)]
                for u, j in cons[y]:
                    t4 = u.inputs[4] if len(u.inputs) > 4 and u.inputs[4] else ""
                    tv = np.zeros(c) + (np.asarray(Q.const_f(t4), np.float64).reshape(-1) if t4 else 0.0)
                    for k in range(c):
                        tc[k].append(float(tv[k]))
                st = Q.stats[y]
                am = max(abs(st["min"]), abs(st["max"]))
                hi = np.zeros(c)
                for k in range(c):
                    tp = [v for v in tc[k] if v >= 0.0]
                    tmax = max(tp) if tp else 0.0
                    mu = max((max(tp) - min(tp)) * 2.0 ** -8 if tp else 0.0, tmax * 2.0 ** -12, am * 2.0 ** -20, 1e-30)
                    hi[k] = tmax + mu
                self.pc[y] = ((hi / 255.0).astype(np.float32).astype(np.float64), np.full(c, -128, np.int64))
            elif n.op == "Conv2D":   # ReLU output: zero point -128 shared, scale per channel
                hi = np.asarray(Q.stats[y]["pc_hi"], np.float64)
                hi = np.maximum(hi, max(float(hi.max()), 1e-30) * 2.0 ** -12)   # dead channels: a small grid, bounded multipliers
                self.pc[y] = ((hi / 255.0).astype(np.float32).astype(np.float64), np.full(hi.size, -128, np.int64))
            else:   # Add(x, Pad(b)): x's grids outside the padded slice, the output's own grid inside
                c0, c1 = pad_slice(prod[n.inputs[1]])
                sx, zx = self.pc[n.inputs[0]]
                sy, zy = self.params[y]
                s, z = sx.copy(), zx.copy()
                s[c0:c1], z[c0:c1] = sy, zy
                st = Q.stats.get(y, {})
                mode = self.opts.get("pc_slices")
                if mode and "slice_pc_hi" in st:
                    # option pc_slices: the updated slice on grids from a per-channel (channel) or per-slice (slice)
                    # calibration of the slice alone (pc_slice_stats; percentile range, channel_minmax / slice_minmax: the
                    # min / max) instead of the whole Add output's range; a channel's span is floored at 2^-12 of the
                    # widest channel's (dead channels: bounded I8AffineSum coefficients)
                    mm = mode.endswith("_minmax")   # the calibration min / max instead of the percentile range
                    if mode.startswith("slice"):
                        s[c0:c1], z[c0:c1] = qparams_range(st["slice_min" if mm else "slice_lo"], st["slice_max" if mm else "slice_hi"])
                    else:
                        lo = np.minimum(np.asarray(st["slice_pc_min" if mm else "slice_pc_lo"], np.float64), 0.0)
                        hi = np.maximum(np.asarray(st["slice_pc_max" if mm else "slice_pc_hi"], np.float64), 0.0)
                        floor = max(float((hi - lo).max()), 1e-30) * 2.0 ** -12
                        for k in range(c0, c1):
                            l, h = lo[k - c0], hi[k - c0]
                            if h - l < floor:
                                h = l + floor
                            s[k], z[k] = qparams_range(l, h)
                if self.opts.get("concat_parts") == "sum":
                    # the slice's grid per channel from the input grids' ranges plus the padded tensor's calibrated range
                    # (tighter than the whole output's range when the slice is one dense layer's output)
                    bst = Q.stats.get(prod[n.inputs[1]].inputs[0])
                    if bst is not None:
                        for k in range(c0, c1):
                            xl, xh = sx[k] * (-128 - zx[k]), sx[k] * (127 - zx[k])
                            s[k], z[k] = qparams_range(xl + bst["lo"], xh + bst["hi"])
                self.pc[y] = (s, z)
        Q.rewrites["per-part grids of Concat trees (concat_parts)"] = len(self.pc)

    SPLITTABLE = {"_Unary", "Add", "AddBroadcast", "_AffineSum", "Conv2D", "Dense", "MatMul", "LayerNorm", "BDense", "BConv2D",
                  "_BMatMulQ", "Mul"}
    SPLIT_PASS = {"MaxPool2D", "Reshape", "Transpose", "Gather", "ChannelShift"}

    def plan_splits(self):
        """Tensors read both by binarizations and by other operators ("mixed") get a second int8 version on a
        binarize-aware grid for the binarizing consumers: the producer is emitted twice (same inputs, the second
        time with the binarize-aware output parameters); a pass-through producer (MaxPool, Reshape, Transpose, Gather)
        is duplicated on the split version of its input. Both versions are int8 tensors."""
        Q = self.Q
        self.split: dict[str, list] = {}       # tensor -> thresholds of its binarizing consumers
        self.center_users: set = set()         # AddBroadcast centering nodes reading the split version (center_split)
        if not self.opts.get("split_mixed", True):
            return
        prod = self.prod

        def can(t, depth=0):
            p = prod.get(t)
            if p is None or t in self.f32_out or depth > 8:
                return False
            if p.op in self.SPLITTABLE:
                return True
            if p.op in self.SPLIT_PASS and not Q.is_const(p.inputs[0]):
                return can(p.inputs[0], depth + 1)
            return False

        for n in Q.nodes:
            for t in n.outputs:
                if t in self.binarize_only or t not in self.params or t in self.pc:
                    continue
                th = self.direct_bin_thresholds(t)
                if th and can(t):
                    self.mark_split(t, th)
        for t, th in list(self.split.items()):
            st = Q.stats.get(t) or Q.stats.get(self.root_of(t))
            am = self.bin_absmax(st) if st else 1.0
            self.params[t + ":bin"] = self.qbin(th, am, t)
        # keep a split only where the binarize-aware grid is clearly finer than the tensor's own grid (thresholds
        # spread over the whole range gain nothing); a pass-through split needs its input split
        changed = True
        while changed:
            changed = False
            for n in Q.nodes:
                t = n.outputs[0]
                if t not in self.split:
                    continue
                coarse = self.params[t + ":bin"][0] > 0.5 * self.params[t][0]
                orphan = n.op in self.SPLIT_PASS and n.inputs[0] not in self.split
                if coarse or orphan:
                    del self.split[t]
                    changed = True
        for n in Q.nodes:  # pass-through copies carry their input's grid (no requantization in those kernels)
            if n.op in self.SPLIT_PASS and n.outputs[0] in self.split:
                self.params[n.outputs[0] + ":bin"] = self.params[n.inputs[0] + ":bin"]

    def root_of(self, t):
        p = self.prod.get(t)
        while p is not None and p.op in self.SPLIT_PASS:
            t = p.inputs[0]
            p = self.prod.get(t)
        return t

    def mark_split(self, t, th):
        old = self.split.get(t, [])
        self.split[t] = sorted(set(old) | set(th))
        p = self.prod.get(t)
        if p is not None and p.op in self.SPLIT_PASS:
            self.mark_split(p.inputs[0], self.split[t])

    def direct_bin_thresholds(self, t):
        """Thresholds of the direct binarizing consumers of t (BDense/BConv2D input, BMatMul operands)."""
        th = []
        for u, j in self.cons.get(t, []):
            if u.op in ("BDense", "BConv2D") and j == 0:
                if len(u.inputs) > 4 and u.inputs[4]:
                    th.extend(np.unique(np.asarray(self.Q.const_f(u.inputs[4]), np.float64)).tolist())
                else:
                    th.append(0.0)
            elif u.op == "_BMatMulQ" and j in (0, 1):
                th.append(float(u.attrs.get("_a_thr" if j == 0 else "_b_thr", 0.0) or 0.0))
            elif u.op == "AddBroadcast" and j == 0 and self.opts.get("center_split", False) and \
                    u.outputs[0] in self.binarize_only and not self.Q.is_const(u.inputs[1]) and u.inputs[1] in self.Q.stats:
                # per-sample centering (siman / RBNN: sign(x + (-mean x))): x only has to be resolved where it can meet
                # -n, n in the calibrated range of the other operand (widened by half its span); beyond, the saturated
                # code keeps x + n on the same side of every threshold of the sum
                st = self.Q.stats[u.inputs[1]]
                w = 0.5 * (st["max"] - st["min"]) + 1e-6 * max(abs(st["max"]), abs(st["min"]), 1e-30)
                for ty in self.binarize_only[u.outputs[0]]:
                    th.extend([ty - st["max"] - w, ty - st["min"] + w])
                self.center_users.add(u.name)
        return sorted(set(th))

    def bin_in(self, t):
        """The tensor a binarizing consumer reads for float tensor t (its binarize-aware split version if any)."""
        return t + ":bin" if t in self.split else t

    def emit_split_copy(self, n):
        """Second emission of node n producing <out>:bin on the binarize-aware grid."""
        t = n.outputs[0]
        m = Node(n.op, list(n.inputs), [t + ":bin"], dict(n.attrs), n.name + "/bin")
        if n.op in self.SPLIT_PASS:
            m.inputs[0] = n.inputs[0] + ":bin"
            self.ren_bin = True
        self.binarize_only[t + ":bin"] = self.split[t]
        self.emit_node(m)

    # ---------------------------------------------------------------- emission helpers
    def const(self, name, arr, dtype):
        if name in self.g.constants:
            return name
        if dtype == "bit":
            c = self.Q._cobj(name) if (name in self.Q.g.constants or name in self.Q.new_consts) else None
            self.g.constants[name] = Constant(name, "bit", tuple(c.shape), np.asarray(c.data), c.layout)
        else:
            a = np.ascontiguousarray(arr)
            self.g.constants[name] = Constant(name, dtype, tuple(a.shape), a)
        return name

    def keep_const(self, t):
        c = self.Q._cobj(t)
        if t not in self.g.constants:
            self.g.constants[t] = Constant(t, c.dtype, tuple(c.shape), np.asarray(c.data), c.layout)
        return t

    def node(self, op, ins, outs, attrs, name):
        self.g.nodes.append(Node(op, list(ins), list(outs), attrs, name))
        if op.startswith("I8") and op not in ("I8Quantize", "I8Dequantize"):
            self.int8_ops[op] += 1
        return outs[0]

    def qin(self, t):
        """int8 version of tensor t (constants quantized with their own range)."""
        if self.Q.is_const(t):
            qn = t + ":q8"
            if qn not in self.g.constants:
                c = np.asarray(self.Q.const_f(t), np.float64)
                if np.all(c == 1.0):
                    s, z = 1.0, 0
                else:
                    s, z = qparams_range(float(c.min()), float(c.max()))
                self.params[qn] = (s, z)
                self.const(qn, quantize_const(c, s, z), "i8")
            return qn
        return t

    def p(self, t):
        return self.params[self.qin(t)] if self.Q.is_const(t) else self.params[t]

    def qminmax(self, act, s, z):
        lo, hi = -128, 127
        if act in ("relu", "relu6"):
            lo = max(lo, z)
        if act == "relu6":
            hi = min(hi, z + int(round(6.0 / s)))
        return lo, hi

    # ---------------------------------------------------------------- emission
    def emit(self):
        Q = self.Q
        self.assign_params()
        g = self.g
        for inp in Q.g.inputs:
            g.inputs.append(dict(inp))
        # quantize f32 graph inputs
        for inp in Q.g.inputs:
            t = inp["name"]
            if inp.get("dtype", "f32") != "f32" or t not in self.params:
                continue
            s, z = self.params[t]
            qn = t + ":q8"
            self.node("I8Quantize", [t], [qn], {"scale": s, "zero": z}, "quantize_" + t)
            self.params[qn] = (s, z)
            self.rename_in = getattr(self, "rename_in", {})
            self.rename_in[t] = qn
        ren = getattr(self, "rename_in", {})
        self.ren = ren
        for n in Q.nodes:
            self.emit_node(n)
            if n.outputs[0] in self.split:
                self.emit_split_copy(n)
        # graph outputs
        outs = []
        for o in Q.g.outputs:
            if o in self.f32_out:
                outs.append(o)
            else:
                s, z = self.params[o]
                self.node("I8Dequantize", [o + ":i8"], [o], {"scale": s, "zero": z}, "dequantize_" + o)
                outs.append(o)
        g.outputs = outs
        if self.split:
            Q.rewrites["binarize-aware second version of mixed tensors (producer emitted twice)"] = len(self.split)
        report = {"rewrites": dict(Q.rewrites), "notes": Q.notes, "int8_ops": dict(self.int8_ops),
                  "left_float": {k: v for k, v in self.left_float.items()},
                  "binarize_aware_tensors": len(self.binarize_only),
                  "tensor_params": {t: [float(s), int(z)] for t, (s, z) in sorted(self.params.items())}}
        return g, report

    def T(self, t):
        """name of the emitted int8 tensor for float tensor t (input renames, output renames)."""
        t = self.ren.get(t, t)
        if t in self.Q.g.outputs and t not in self.f32_out:
            return t + ":i8"
        return t

    def out_name(self, t):
        return t if t in self.f32_out else self.T(t)

    def emit_node(self, n):
        Q, op, a = self.Q, n.op, n.attrs
        y = n.outputs[0]
        yo = self.out_name(y)
        f32o = y in self.f32_out
        if not f32o:
            sy, zy = self.params[y]
        if op == "Reshape":
            self.node("Reshape", [self.T(n.inputs[0])], [yo], {"shape": a["shape"]}, n.name)
        elif op == "Transpose":
            self.node("I8Transpose", [self.T(n.inputs[0])], [yo], {"perm": a["perm"]}, n.name)
        elif op == "Gather":
            self.emit_gather(n, yo)
        elif op == "MaxPool2D":
            self.node("I8MaxPool2D", [self.T(n.inputs[0])], [yo], {k: a[k] for k in ("kernel", "strides", "pads") if k in a}, n.name)
        elif op == "AvgPool2D":
            sx, zx = self.p(n.inputs[0])
            assert (sx, zx) == (sy, zy) or True
            self.params[y] = (sx, zx)
            self.node("I8AvgPool2D", [self.T(n.inputs[0])], [yo], dict({k: a[k] for k in ("kernel", "strides", "pads", "count_include_pad") if k in a},
                                                                      x_zero=zx), n.name)
        elif op in ("GlobalAvgPool", "ReduceMean"):
            sx, zx = self.p(n.inputs[0])
            shp = Q._shape(n.inputs[0])
            cnt = shp[1] * shp[2] if op == "GlobalAvgPool" else shp[a.get("axis", -1) % len(shp)]
            m, sh = int8_ops.quantize_multiplier(sx / (sy * cnt))
            at = dict({k: a[k] for k in ("keepdims", "axis") if k in a}, x_zero=zx, y_zero=zy, mult=m, shift=sh)
            self.node("I8" + op, [self.T(n.inputs[0])], [yo], at, n.name)
        elif op == "Concat" and y in self.pc:   # per-part grids: plain copies
            self.node("I8Concat", [self.T(t) for t in n.inputs], [yo], {"axis": a.get("axis", -1), "y_zero": 0,
                      "zeros": [0] * len(n.inputs), "mults": [0] * len(n.inputs), "shifts": [0] * len(n.inputs)}, n.name)
        elif op == "Add" and y in self.pc:
            self.emit_pc_add(n, yo)
        elif op == "Concat":
            zs, ms, shs = [], [], []
            ins = []
            for t in n.inputs:
                ins.append(self.T(self.qin(t)) if Q.is_const(t) else self.T(t))
                if Q.is_const(t):
                    c = np.asarray(Q.const_f(t), np.float64)
                    qn = t + ":q8y"
                    self.const(qn, quantize_const(c, sy, zy), "i8")
                    ins[-1] = qn
                    zs.append(zy), ms.append(0), shs.append(0)
                    continue
                st, zt = self.params[t]
                if (st, zt) == (sy, zy):
                    zs.append(zt), ms.append(0), shs.append(0)
                else:
                    m, sh = int8_ops.quantize_multiplier(st / sy)
                    zs.append(zt), ms.append(m), shs.append(sh)
            self.node("I8Concat", ins, [yo], {"axis": a.get("axis", -1), "y_zero": zy, "zeros": zs, "mults": ms, "shifts": shs}, n.name)
        elif op in ("Add", "AddBroadcast"):
            self.emit_add(n, yo, sy, zy)
        elif op == "Mul":
            sa, za = self.p(n.inputs[0])
            sb, zb = self.p(n.inputs[1])
            m, sh = int8_ops.quantize_multiplier(sa * sb / sy)
            self.node("I8Mul", [self.T(self.qin(n.inputs[0])), self.T(self.qin(n.inputs[1]))], [yo],
                      {"a_zero": za, "b_zero": zb, "mult": m, "shift": sh, "y_zero": zy}, n.name)
        elif op == "LayerNorm":
            sx, zx = self.p(n.inputs[0])
            d = Q._shape(n.inputs[0])[-1]
            gm = np.asarray(Q.const_f(n.inputs[1]), np.float64) if len(n.inputs) > 1 and n.inputs[1] else np.ones(d)
            bt = np.asarray(Q.const_f(n.inputs[2]), np.float64) if len(n.inputs) > 2 and n.inputs[2] else np.zeros(d)
            G = self.const(n.name + "/G", (gm / sy).astype(np.float32), "f32")
            B = self.const(n.name + "/B", (bt / sy + zy).astype(np.float32), "f32")
            self.node("I8LayerNorm", [self.T(n.inputs[0]), G, B], [yo], {"x_zero": zx, "x_scale": sx, "eps": a.get("eps", 1e-5)}, n.name)
        elif op == "Softmax":
            sx, zx = self.p(n.inputs[0])
            nn = Q._shape(n.inputs[0])[-1]
            bits = min(22, 30 - int(math.ceil(math.log2(max(nn, 2)))))
            tab = np.array([int(round(math.ldexp(math.exp(-sx * d), bits))) for d in range(256)], np.int32)
            E = self.const(n.name + "/exp", tab, "i32")
            at = {"output": "f32"} if f32o else {"inv_scale": 1.0 / sy, "y_zero": zy}
            self.node("I8Softmax", [self.T(n.inputs[0]), E], [yo], at, n.name)
        elif op == "_Unary":
            self.emit_unary(n, yo, f32o)
        elif op == "_AffineSum":
            self.emit_affine_sum(n, yo, sy, zy)
        elif op in ("Conv2D", "Dense"):
            self.emit_gemm(n, yo, f32o)
        elif op == "MatMul":
            sa, za = self.p(n.inputs[0])
            sb, zb = self.p(n.inputs[1])
            at = {"a_zero": za, "b_zero": zb, "transpose_b": a.get("transpose_b", 0)}
            if f32o:
                at.update(output="f32", scale=float(np.float32(sa * sb * a.get("scale", 1.0))))
            else:
                m, sh = int8_ops.quantize_multiplier(sa * sb * a.get("scale", 1.0) / sy)
                at.update(mult=m, shift=sh, y_zero=zy)
            self.node("I8MatMul", [self.T(self.qin(n.inputs[0])), self.T(self.qin(n.inputs[1]))], [yo], at, n.name)
        elif op in ("BDense", "BConv2D"):
            self.emit_binary(n, yo, f32o)
        elif op == "_BMatMulQ":
            self.emit_bmm(n, yo, f32o)
        elif op == "ChannelGate":
            self.emit_channel_gate(n, yo, sy, zy)
        elif op == "Pad":
            sx, zx = self.p(n.inputs[0])
            self.params[y] = (sx, zx)
            self.node("I8Pad", [self.T(n.inputs[0])], [yo], {"pads": a["pads"], "pad_q": int(quantize_const(np.array([a.get("value", 0.0)]), sx, zx)[0])},
                      n.name)
        elif op == "ChannelShift":
            sx, zx = self.p(n.inputs[0])
            self.params[y] = (sx, zx)
            gs = [v for t in executor.shift_groups(a.get("groups", [])) for v in t]
            self.node("I8ChannelShift", [self.T(n.inputs[0])], [yo],
                      {"groups": gs, "fill_q": int(quantize_const(np.array([0.0]), sx, zx)[0])}, n.name)
        elif op == "InstanceThreshold":
            self.emit_instance_threshold(n)
        elif op == "SimAM":
            self.emit_simam(n, yo, sy, zy)
        else:
            self.emit_float_fallback(n, f"no int8 lowering for {op}")

    # ---------------------------------------------------------------- glue ops (docs/int8.md "int8_full ops for accuracy-carrying glue")
    def emit_instance_threshold(self, n):
        """InstanceThreshold(x, t0, a, b[, off]) read only by binary layers: one I8BinarizeDyn per distinct binarization
        (mode, thresholds) of its readers, writing the bit tensor they read (emit_binary, self.dyn_in). bit = +1 iff
        x - (t0 + a mean + b std + off) >= t (gt: >), with x = s (q - z): (q - z) HW 2^F >= T0 + A S1 + B isqrt(V 4^G) + Ko (o - z_o),
        T0 = (t0 + t) HW 2^F / s (+1 for gt, the integer form of >), A = a 2^F, B = b 2^(F - G), Ko = s_o HW 2^F / s."""
        Q = self.Q
        x, y = n.inputs[0], n.outputs[0]
        sx, zx = self.p(x)
        shp = Q._shape(x)
        hw, c = int(shp[1] * shp[2]), int(shp[3])
        vec = lambda t: np.zeros(c) + (np.asarray(Q.const_f(t), np.float64).reshape(-1) if t else 0.0)  # noqa: E731
        t0, av, bv = vec(n.inputs[1]), vec(n.inputs[2]), vec(n.inputs[3])
        off = n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else ""
        if off:
            so, zo = self.p(off)
        readers = self.cons.get(y, [])
        if not readers or y in Q.g.outputs or any(not (u.op in ("BConv2D", "BDense") and j == 0) for u, j in readers):
            raise NotImplementedError("InstanceThreshold must be read only by binary layers (input 0)")
        G = 0
        while G < 20 and hw * hw * 65025 << (2 * (G + 1)) < 1 << 61:
            G += 1
        G = min(G, 16)
        self.dyn_in = getattr(self, "dyn_in", {})
        groups = []
        for u, _ in readers:
            mode = u.attrs.get("input_binarize", "ge")
            if mode not in ("ge", "gt"):
                raise NotImplementedError("InstanceThreshold read by a binarization " + mode)
            tr = vec(u.inputs[4] if len(u.inputs) > 4 and u.inputs[4] else "")
            key = (mode, tr.tobytes())
            k = next((i for i, g in enumerate(groups) if g[0] == key), None)
            if k is None:
                groups.append((key, mode, tr))
                k = len(groups) - 1
            self.dyn_in[u.name] = f"{y}:dyn{k}"
        for k, (_, mode, tr) in enumerate(groups):
            for F in range(40, -1, -1):
                sc = float(hw) * 2.0 ** F / sx
                T0 = np.rint((t0 + tr) * sc) + (1 if mode == "gt" else 0)
                A = np.rint(av * 2.0 ** F)
                B = np.rint(bv * 2.0 ** (F - G))
                Ko = np.rint(np.full(c, so * sc)) if off else None
                big = max(np.abs(T0).max(), np.abs(A).max(), np.abs(B).max(), np.abs(Ko).max() if off else 0.0)
                if big < 2 ** 31 - 1 and int8_ops.binarize_dyn_bound_ok(hw, T0.astype(np.int64), A.astype(np.int64), B.astype(np.int64),
                                                                       Ko.astype(np.int64) if off else None, F, G):
                    break
            else:
                raise NotImplementedError("InstanceThreshold: constants do not fit")
            base = f"{n.name}/dyn{k}"
            ins = [self.T(x), self.const(base + "/T0", T0.astype(np.int32), "i32"), self.const(base + "/A", A.astype(np.int32), "i32"),
                   self.const(base + "/B", B.astype(np.int32), "i32")]
            at = {"x_zero": int(zx), "shift": int(F), "sqrt_shift": int(G)}
            if off:
                ins += [self.T(off), self.const(base + "/Ko", Ko.astype(np.int32), "i32")]
                at["o_zero"] = int(zo)
            self.node("I8BinarizeDyn", ins, [f"{y}:dyn{k}"], at, base)

    def emit_simam(self, n, yo, sy, zy):
        """SimAM(x): I8SimAM with the gate table tab[k] = sigmoid(1/2 + (k + 1/2) / 16) on the grid (1/255, -128),
        lam_q = round(lambda HW^2 / s^2) and the output requantization s / (255 s_y)."""
        x = n.inputs[0]
        sx, zx = self.p(x)
        shp = self.Q._shape(x)
        hw = int(shp[1] * shp[2])
        k = np.arange(256, dtype=np.float64)
        gate = 1.0 / (1.0 + np.exp(-(0.5 + (k + 0.5) / 16.0)))
        gate[255] = 1.0
        tab = (np.clip(np.rint(gate * 255.0), 0, 255) - 128).astype(np.int8)[None, :]
        lam = float(n.attrs.get("lambda", 1e-4))
        L = int(min(max(1, round(lam * hw * hw / (sx * sx))), 1 << 40))
        m, sh = int8_ops.quantize_multiplier(sx / (255.0 * sy))
        self.node("I8SimAM", [self.T(x), self.const(n.name + "/gate_tab", tab, "i8")], [yo],
                  {"x_zero": int(zx), "lam_q": L, "g_zero": -128, "mult": m, "shift": sh, "y_zero": int(zy)}, n.name)

    def emit_float_fallback(self, n, why):
        """Dequantize -> float op -> Quantize (recorded in the report as left in float)."""
        self.left_float[n.op].append({"node": n.name, "reason": why})
        ins = []
        for t in n.inputs:
            if not t or self.Q.is_const(t):
                if t:
                    self.keep_const(t)
                ins.append(t)
                continue
            s, z = self.params[t]
            dn = self.Q.uid(t + "/deq") + ":out"
            self.node("I8Dequantize", [self.T(t)], [dn], {"scale": s, "zero": z}, self.Q.uid("deq"))
            ins.append(dn)
        y = n.outputs[0]
        if y in self.f32_out:
            self.node(n.op, ins, [y], n.attrs, n.name)
        else:
            fo = self.Q.uid(y + "/f32") + ":out"
            self.node(n.op, ins, [fo], n.attrs, n.name)
            s, z = self.params[y]
            self.node("I8Quantize", [fo], [self.out_name(y)], {"scale": s, "zero": z}, self.Q.uid("q"))

    def emit_gather(self, n, yo):
        Q = self.Q
        tab, idx = n.inputs[0], n.inputs[1]
        y = n.outputs[0]
        sy, zy = self.params[y]
        if Q.is_const(idx):
            self.keep_const(idx)
        ins = [None, self.T(idx) if not Q.is_const(idx) else idx]
        if Q.is_const(tab):
            c = Q._cobj(tab)
            if c.dtype == "bit":
                self.keep_const(tab)
                sc = np.asarray(Q.const_f(n.inputs[2]), np.float64) if len(n.inputs) > 2 and n.inputs[2] else np.ones(c.shape[0])
                qpm = np.stack([quantize_const(sc, sy, zy), quantize_const(-sc, sy, zy)], axis=1)
                ins[0] = tab
                ins.append(self.const(n.name + "/qpm", qpm, "i8"))
                if len(n.inputs) > 3 and n.inputs[3]:
                    ins.append(self.keep_const(n.inputs[3]))
            else:
                ins[0] = self.const(n.name + "/table", quantize_const(np.asarray(Q.const_f(tab), np.float64), sy, zy), "i8")
        else:
            ins[0] = self.T(tab)
        self.node("I8Gather", ins, [yo], {"fill_q": zy}, n.name)

    def emit_add(self, n, yo, sy, zy):
        a0, b0 = n.inputs
        if n.name in self.center_users and a0 in self.split:
            a0 = a0 + ":bin"
        sa, za = self.p(a0)
        sb, zb = self.p(b0)
        tm = 2.0 * max(sa, sb)
        ma, sha = int8_ops.quantize_multiplier(sa / tm)
        mb, shb = int8_ops.quantize_multiplier(sb / tm)
        mo, sho = int8_ops.quantize_multiplier(tm / ((1 << 20) * sy))
        lo, hi = self.qminmax(n.attrs.get("activation", "none"), sy, zy)
        self.node("I8Add", [self.T(self.qin(a0)), self.T(self.qin(b0))], [yo],
                  {"a_zero": za, "b_zero": zb, "left_shift": 20, "a_mult": ma, "a_shift": sha, "b_mult": mb, "b_shift": shb,
                   "out_mult": mo, "out_shift": sho, "y_zero": zy, "qmin": lo, "qmax": hi}, n.name)

    def emit_unary(self, n, yo, f32o):
        Q = self.Q
        sx, zx = self.p(n.inputs[0])
        shp = Q._shape(n.inputs[0])
        axis = None
        for op, attrs, ins in n.attrs["ops"]:
            nd = Node(op, list(ins), ["_"], attrs, "_")
            ax = Q._unary_axis(nd)
            axis = ax if ax is not None else axis
        codes = np.arange(-128, 128, dtype=np.float64)
        x = sx * (codes - zx)
        if n.inputs[0] in self.pc:   # per-part input grids: one table row per channel (last axis)
            ps, pz = self.pc[n.inputs[0]]
            x = ps[:, None] * (codes[None, :] - pz[:, None])
            axis = len(shp) - 1
        consts = {}
        for op, attrs, ins in n.attrs["ops"]:
            for t in ins[1:]:
                if t:
                    v = np.asarray(Q.const_f(t), np.float64).reshape(-1)
                    consts[t] = v[0] if v.size == 1 or np.all(v == v[0]) else v[:, None]
        if x.ndim == 2:
            yv = unary_eval(n.attrs["ops"], x, consts, table=True)
        elif axis is not None:
            yv = unary_eval(n.attrs["ops"], np.broadcast_to(x[None, :], (shp[axis], 256)), consts, table=True)
        else:
            yv = unary_eval(n.attrs["ops"], x, consts, table=True)[None, :]
        lut_attrs = {} if axis is None or axis == len(shp) - 1 else {"axis": int(axis)}
        if f32o:
            tab = self.const(n.name + "/lut", yv.astype(np.float32), "f32")
            self.node("I8Lut", [self.T(n.inputs[0]), tab], [yo], dict(lut_attrs, output="f32"), n.name)
        else:
            sy, zy = self.params[n.outputs[0]]
            codes_out = np.clip(np.rint(yv / sy) + zy, -128, 127)
            if n.outputs[0].endswith(":bin"):  # per-row thresholds of the original tensor's binarizing consumers
                rows_th = self.row_thresholds(n.outputs[0][:-4], yv.shape[0], axis is None or axis == len(shp) - 1)
            else:
                rows_th = self.row_thresholds(n.outputs[0], yv.shape[0], axis is None or axis == len(shp) - 1)
            if rows_th is not None:
                # only binarized: up to 4 codes away; mixed consumers: at most one code (<= 1 LSB for the arithmetic ones)
                radius = 4 if (n.outputs[0] in self.binarize_only or n.outputs[0].endswith(":bin")) else 1
                codes_out = sign_preserving_codes(yv, codes_out, sy, zy, rows_th, radius=radius)
            tab = self.const(n.name + "/lut", codes_out.astype(np.int8), "i8")
            self.node("I8Lut", [self.T(n.inputs[0]), tab], [yo], lut_attrs, n.name)

    def emit_channel_gate(self, n, yo, sy, zy):
        """ChannelGate(x, g) = x * sigmoid(g) per (sample, channel): I8Lut (sigmoid on g's grid, output grid [0, 1]),
        Reshape to [N, 1, 1, C], I8Mul with broadcasting (gate 'none': the gate tensor itself)."""
        x, gt = n.inputs[0], n.inputs[1]
        sx, zx = self.p(x)
        sg, zg = self.p(gt)
        shp = self.Q._shape(x)
        g_in = self.T(self.qin(gt))
        if n.attrs.get("gate", "sigmoid") == "sigmoid":
            ss, zs = 1.0 / 255.0, -128
            codes = np.arange(-128, 128, dtype=np.float64)
            tab = np.clip(np.rint(1.0 / (1.0 + np.exp(-sg * (codes - zg))) / ss) + zs, -128, 127).astype(np.int8)[None, :]
            sig = self.Q.uid(n.name + "/sigmoid") + ":out"
            self.node("I8Lut", [g_in, self.const(n.name + "/sigmoid_lut", tab, "i8")], [sig], {}, n.name + "/sigmoid")
            g_in, sg, zg = sig, ss, zs
        rs = self.Q.uid(n.name + "/gate4") + ":out"
        self.node("Reshape", [g_in], [rs], {"shape": [shp[0], 1, 1, shp[3]]}, n.name + "/gate4")
        m, sh = int8_ops.quantize_multiplier(sx * sg / sy)
        self.node("I8Mul", [self.T(self.qin(x)), rs], [yo], {"a_zero": zx, "b_zero": zg, "mult": m, "shift": sh, "y_zero": zy}, n.name)

    def emit_affine_sum(self, n, yo, sy, zy):
        """K[i][c] = round(a_i[c] * s_i / s_y * 2^F), B[c] = round(b[c] / s_y * 2^F) (|B| <= 2^30), requant by 2^-F;
        F <= 21 and N*255*max|K| <= 2^28 keep the int32 accumulator exact and make a clamped B saturate."""
        leaves = n.inputs
        ins, zs, rows, pcz = [], [], [], []
        c = self.Q._shape(n.outputs[0])[-1]
        for t, a in zip(leaves, n.attrs["coef"]):
            qt = self.qin(t)
            s_i, z_i = self.params[qt]
            pcz.append(None)
            if t in self.pc:   # per-part input grids (plan_parts): per-channel scale, zero point folded into B
                s_i = self.pc[t][0]
                pcz[-1] = self.pc[t][1]
                z_i = 0
            ins.append(self.T(qt))
            zs.append(int(z_i))
            rows.append(np.zeros(c) + np.asarray(a, np.float64) * s_i / sy if pcz[-1] is not None else np.asarray(a, np.float64) * s_i / sy)
        if any(z is not None for z in pcz):
            rows = [np.zeros(c) + r for r in rows]
        coef = np.stack(rows)
        cmax = max(float(np.abs(coef).max()), 1e-30)
        f = int(min(21, math.floor(math.log2(2.0 ** 28 / (len(leaves) * 255 * cmax)))))
        K = np.rint(coef * 2.0 ** f).astype(np.int64)
        B = np.clip(np.rint(np.asarray(n.attrs["bias"], np.float64) / sy * 2.0 ** f), -(2 ** 30), 2 ** 30).astype(np.int64)
        for i, z in enumerate(pcz):
            if z is not None:
                B = np.zeros(c, np.int64) + B - z.astype(np.int64) * K[i]
        assert np.abs(B).max() <= 2 ** 30
        if np.all(K == K[:, :1]) and np.all(B == B[0]):
            K, B = K[:, :1], B[:1]
        m, sh = int8_ops.quantize_multiplier(2.0 ** -f)
        lo, hi = self.qminmax(n.attrs.get("activation", "none"), sy, zy)
        Kn = self.const(n.name + "/K", K.astype(np.int32), "i32")
        Bn = self.const(n.name + "/B", B.astype(np.int32), "i32")
        self.node("I8AffineSum", ins + [Kn, Bn], [yo], {"zeros": zs, "mult": m, "shift": sh, "y_zero": zy, "qmin": lo, "qmax": hi}, n.name)

    def emit_pc_add(self, n, yo):
        """Add(x, Pad(b)) on per-part grids (plan_parts): I8AffineSum with per-channel coefficients, input zero points
        and output zero points folded into B (zeros 0, y_zero 0): y[c] = requant(sum_i q_i[c] K[i][c] + B[c]) with
        K[i][c] = round(s_i[c] / s_y[c] 2^F), B[c] = z_y[c] 2^F - sum_i z_i[c] K[i][c] (the channels outside b's slice:
        K = 2^F exactly for x, so their codes are copied)."""
        x, pb = n.inputs
        sy, zy = self.pc[n.outputs[0]]
        sx, zx = self.pc[x]
        sb, zb = self.p(pb)
        c = sy.size
        rows = [sx / sy, np.full(c, sb) / sy]
        coef = np.stack(rows)
        cmax = max(float(np.abs(coef).max()), 1e-30)
        f = int(min(21, math.floor(math.log2(2.0 ** 28 / (2 * 255 * cmax)))))
        K = np.rint(coef * 2.0 ** f).astype(np.int64)
        B = zy.astype(np.int64) * (1 << f) - zx.astype(np.int64) * K[0] - int(zb) * K[1]   # exact integers
        assert np.abs(B).max() < 2 ** 30
        m, sh = int8_ops.quantize_multiplier(2.0 ** -f)
        Kn = self.const(n.name + "/K", K.astype(np.int32), "i32")
        Bn = self.const(n.name + "/B", B.astype(np.int32), "i32")
        self.node("I8AffineSum", [self.T(x), self.T(pb), Kn, Bn], [yo], {"zeros": [0, 0], "mult": m, "shift": sh, "y_zero": 0,
                                                                           "qmin": -128, "qmax": 127}, n.name)

    def emit_gemm(self, n, yo, f32o):
        Q, a = self.Q, n.attrs
        sx, zx = self.p(n.inputs[0])
        w = np.asarray(Q.const_f(n.inputs[1]), np.float64)
        wq, sw = quantize_weights_per_channel(w)
        o = w.shape[0]
        b = np.asarray(Q.const_f(n.inputs[2]), np.float64) if len(n.inputs) > 2 and n.inputs[2] else np.zeros(o)
        wn = self.const(n.name + "/wq", wq, "i8")
        at = {"x_zero": zx}
        if n.op == "Conv2D":
            at.update({k: a[k] for k in ("strides", "dilations", "pads", "groups") if k in a})
            at["pad_q"] = int(quantize_const(np.array([a.get("pad_value", 0.0)]), sx, zx)[0])
        if f32o:
            if a.get("activation", "none") != "none":
                raise NotImplementedError("f32 output with fused activation")
            bn = self.const(n.name + "/bias_f", b.astype(np.float32), "f32")
            scn = self.const(n.name + "/scale_f", (sx * sw).astype(np.float32), "f32")
            at["output"] = "f32"
            ins = [self.T(n.inputs[0]), wn, bn, scn]
        else:
            sy, zy = self.params[n.outputs[0]]
            sys_ = np.full(o, sy)
            if n.outputs[0] in self.pc:   # per-channel output scales, shared zero point (plan_parts, pc_stem)
                sys_, zv = self.pc[n.outputs[0]]
                assert np.all(zv == zv[0])
                zy = int(zv[0])
            bq = np.clip(np.rint(b / (sx * sw)), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
            mm = [int8_ops.quantize_multiplier(sx * s / sc) for s, sc in zip(sw, sys_)]
            bn = self.const(n.name + "/bias_q", bq, "i32")
            mn = self.const(n.name + "/mult", np.array([m for m, _ in mm], np.int32), "i32")
            sn = self.const(n.name + "/shift", np.array([s for _, s in mm], np.int32), "i32")
            lo, hi = self.qminmax(a.get("activation", "none"), float(sys_[0]), zy)
            at.update(y_zero=zy, qmin=lo, qmax=hi)
            ins = [self.T(n.inputs[0]), wn, bn, mn, sn]
        self.node("I8" + n.op, ins, [yo], at, n.name)

    def emit_binary(self, n, yo, f32o):
        Q, a = self.Q, n.attrs
        n = Node(n.op, [self.bin_in(n.inputs[0])] + list(n.inputs[1:]), n.outputs, n.attrs, n.name)
        sx, zx = self.p(n.inputs[0])
        w = n.inputs[1]
        self.keep_const(w)
        o = Q._cobj(w).shape[0]
        pm = a.get("_pm64")
        pa = a.get("_pa64")
        if pm is None:
            pm = np.zeros(o) + (np.asarray(Q.const_f(n.inputs[2]), np.float64) if len(n.inputs) > 2 and n.inputs[2] else 1.0)
        if pa is None:
            pa = np.zeros(o) + (np.asarray(Q.const_f(n.inputs[3]), np.float64) if len(n.inputs) > 3 and n.inputs[3] else 0.0)
        at = {k: v for k, v in a.items() if not k.startswith("_") and k not in ("activation", "output")}
        at.update(x_scale=sx, x_zero=zx)
        pc_thr = None
        if n.inputs[0] in self.pc:   # per-part input grids: integer code thresholds per channel on the identity grid
            ps, pz = self.pc[n.inputs[0]]
            c = ps.size
            thr0 = n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else ""
            tv = np.zeros(c) + (np.asarray(Q.const_f(thr0), np.float64).reshape(-1) if thr0 else 0.0)
            qf = int8_ops.qthr_gt if a.get("input_binarize", "ge") == "gt" else int8_ops.qthr_ge
            T = np.array([int(qf(np.float32(tv[k]), float(ps[k]), int(pz[k]))) for k in range(c)], np.float64)
            pc_thr = self.const(n.name + "/thr_pc", T.astype(np.float32), "f32")
            at.update(x_scale=1.0, x_zero=0, input_binarize="ge")
        if f32o:
            if a.get("activation", "none") != "none":
                raise NotImplementedError("f32 output with fused activation")
            pmn = self.const(n.name + "/pm_f", pm.astype(np.float32), "f32")
            pan = self.const(n.name + "/pa_f", pa.astype(np.float32), "f32")
            at["output"] = "f32"
        else:
            sy, zy = self.params[n.outputs[0]]
            if n.outputs[0] in self.pc:   # per-channel output grids (plan_parts pc_bin), shared zero point
                sv, zv = self.pc[n.outputs[0]]
                assert np.all(zv == zv[0])
                sy, zy = sv, int(zv[0])
            pmn = self.const(n.name + "/pm_q", (pm / sy).astype(np.float32), "f32")
            pan = self.const(n.name + "/pa_q", (pa / sy + zy).astype(np.float32), "f32")
            lo, hi = self.qminmax(a.get("activation", "none"), float(np.max(sy)), zy)
            at.update(qmin=lo, qmax=hi)
        ins = [self.T(n.inputs[0]), w, pmn, pan]
        thr = n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else ""
        dyn = getattr(self, "dyn_in", {}).get(n.name)
        if dyn is not None:   # reads the bit tensor of an I8BinarizeDyn (InstanceThreshold): its thresholds are folded there
            ins[0] = dyn
            thr, pc_thr = "", None
            for k in ("x_scale", "x_zero", "input_binarize"):
                at.pop(k, None)
        if pc_thr is not None:
            ins.append(pc_thr)
        else:
            ins.append(self.const(thr, np.asarray(Q.const_f(thr), np.float32), "f32") if thr else "")
        if len(n.inputs) > 5 and n.inputs[5]:
            ins.append(self.keep_const(n.inputs[5]))
        while ins and ins[-1] == "":
            ins.pop()
        self.node("I8" + n.op, ins, [yo], at, n.name)

    def emit_bmm(self, n, yo, f32o):
        Q, a = self.Q, n.attrs
        ins = []
        at = {"a_binarize": a.get("a_binarize", "sign_zero"), "b_binarize": a.get("b_binarize", "sign_zero"),
              "transpose_b": int(a.get("transpose_b", 0)), "c_ab": int(a["c_ab"]), "c_rs": int(a["c_rs"]), "c_cs": int(a["c_cs"]),
              "c0": int(a["c0"])}
        for j, k in ((0, "a"), (1, "b")):
            t = n.inputs[j] if self.Q.is_const(n.inputs[j]) else self.bin_in(n.inputs[j])
            qt = self.qin(t)
            s, z = self.params[qt]
            at[k + "_scale"], at[k + "_zero"] = s, z
            thr = a.get(f"_{k}_thr")
            if thr is not None:
                at[k + "_threshold"] = float(thr)
            ins.append(self.T(qt))
        mul = float(a["mul"])
        if f32o:
            ins += [self.const(n.name + "/mul", np.array([mul], np.float32), "f32")]
            at["output"] = "f32"
        else:
            sy, zy = self.params[n.outputs[0]]
            ins += [self.const(n.name + "/mul", np.array([mul / sy], np.float32), "f32"),
                    self.const(n.name + "/add", np.array([zy], np.float32), "f32")]
        self.node("I8BMatMul", ins, [yo], at, n.name)
