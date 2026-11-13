"""Reference semantics (numpy oracle) of the full-integer `I8*` operators (stage `int8_full`).

The C++ reference kernels (runtime/kernels/int8_full/) must be bit-identical to this file; the NEON
kernels must be bit-identical to the C++ reference. The contracts are documented in docs/int8.md
("Full-integer execution: int8_full") and summarized here:

* Activation tensors are int8 with a per-tensor affine map  x = s * (q - z)  (s > 0, z an int32,
  normally in [-128, 127]). The parameters travel as node attributes (`x_scale`, `x_zero`, `y_zero`, ...).
* Fixed-point requantization (`requant`): a real multiplier m > 0 is stored as (M, shift) with
  M in [2^30, 2^31) and m ~= M * 2^shift / 2^31. requant(x) = rshr(mulhi(sat32(x << max(shift, 0)), M),
  max(-shift, 0)) with mulhi(x, M) = (x*M + 2^30) >> 31 (AArch64 SQRDMULH) and
  rshr(x, n) = (x + 2^(n-1)) >> n (AArch64 SRSHL by -n). Both roundings are round-half-up (towards
  +inf); >> is an arithmetic (floor) shift.
* fp32 epilogues are evaluated one IEEE fp32 operation at a time in the order written (numpy float32
  element-wise ops; the C++ build uses -ffp-contract=off), and rounded to an integer with
  round-half-to-even (np.rint / nearbyint / FCVTNS), clamped in float before the conversion.
* Binarization of an int8 operand is binarization of its dequantized value deq(q) = fp32(fp32(q - z) * s):
  deq is non-decreasing in q, so `deq(q) >= t` <=> `q >= T_ge(t)` with T_ge(t) = min{q : deq(q) >= t}
  (128 if none) and `deq(q) > t` <=> `q >= T_gt(t)`. qthr_ge / qthr_gt compute these by exhaustive
  search over the 256 codes, i.e. exactly by definition.
"""
from __future__ import annotations

import math

import numpy as np

from .ir import unpack_bits

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


# ----------------------------------------------------------------------------- fixed point
def quantize_multiplier(real: float) -> tuple[int, int]:
    """(M, shift) with M in [2^30, 2^31) and real ~= M * 2^shift / 2^31 (M = 0 for real == 0).
    M = round_half_away(frac * 2^31) where real = frac * 2^shift, frac in [0.5, 1)."""
    if real < 0:
        raise ValueError("multiplier must be >= 0")
    if real == 0.0:
        return 0, 0
    frac, exp = math.frexp(real)
    m = int(math.floor(frac * (1 << 31) + 0.5))
    if m == (1 << 31):
        m //= 2
        exp += 1
    if exp < -62:  # the product underflows any int32 input: represent as 0
        return 0, 0
    if exp > 30:
        raise ValueError(f"multiplier {real} too large")
    return m, exp


def sat32(x: np.ndarray) -> np.ndarray:
    return np.clip(x, I32_MIN, I32_MAX)


def mulhi(x: np.ndarray, m: int | np.ndarray) -> np.ndarray:
    """(x*M + 2^30) >> 31 in int64 (= SQRDMULH for M > 0; x, M int32)."""
    return (np.asarray(x, np.int64) * np.asarray(m, np.int64) + (1 << 30)) >> 31


def rshr(x: np.ndarray, n: int | np.ndarray) -> np.ndarray:
    """Rounding arithmetic shift right, half up: (x + 2^(n-1)) >> n; identity for n == 0."""
    x = np.asarray(x, np.int64)
    n = np.asarray(n, np.int64)
    rnd = np.where(n > 0, np.left_shift(np.int64(1), np.maximum(n - 1, 0)), 0)
    return (x + rnd) >> n


def requant(x: np.ndarray, m, shift) -> np.ndarray:
    """rshr(mulhi(sat32(x << max(shift, 0)), M), max(-shift, 0)); x int32-valued. Returns int64 values."""
    shift = np.asarray(shift, np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    xs = sat32(np.left_shift(np.asarray(x, np.int64), left))
    return rshr(mulhi(xs, m), right)


def rdiv(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Integer division rounding half away from zero (TFLite average pooling)."""
    a = np.asarray(a, np.int64)
    c = np.asarray(c, np.int64)
    return np.where(a >= 0, (a + c // 2) // c, -((-a + c // 2) // c))


def clamp_q(v: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    return np.clip(v, qmin, qmax).astype(np.int8)


def round_f32(v: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    """fp32 value -> int8: clamp(rne(v)) (clamping in float, then conversion: exact for every finite v)."""
    return np.clip(np.rint(np.asarray(v, np.float32)), np.float32(qmin), np.float32(qmax)).astype(np.int8)


# ----------------------------------------------------------------------------- quantization
def f32(x) -> np.float32:
    return np.float32(x)


def deq(q, s: float, z: int) -> np.ndarray:
    """Dequantized value in fp32: fp32(fp32(q - z) * s)."""
    return (np.asarray(np.asarray(q, np.int64) - z, np.float32) * f32(s)).astype(np.float32)


_CODES = np.arange(-128, 128, dtype=np.int64)


def qthr_ge(t, s: float, z: int) -> np.ndarray:
    """min{q in [-128,127] : deq(q) >= t}, 128 if none (elementwise over t)."""
    t = np.asarray(t, np.float32)
    d = deq(_CODES, s, z)                                   # non-decreasing
    return (np.searchsorted(d, t.reshape(-1), side="left") - 128).reshape(t.shape).astype(np.int32)


def qthr_gt(t, s: float, z: int) -> np.ndarray:
    """min{q : deq(q) > t}, 128 if none."""
    t = np.asarray(t, np.float32)
    d = deq(_CODES, s, z)
    return (np.searchsorted(d, t.reshape(-1), side="right") - 128).reshape(t.shape).astype(np.int32)


def binarize_q(q: np.ndarray, mode: str, t, s: float, z: int) -> np.ndarray:
    """Binarization of int8 codes = binarization of deq(q) against threshold t (scalar or per last-dim channel)."""
    q = np.asarray(q, np.int32)
    t = np.zeros(1, np.float32) if t is None else np.asarray(t, np.float32)
    tge, tgt = qthr_ge(t, s, z), qthr_gt(t, s, z)
    if mode == "ge":
        return np.where(q >= tge, 1, -1).astype(np.int32)
    if mode == "gt":
        return np.where(q >= tgt, 1, -1).astype(np.int32)
    if mode == "sign_zero":
        return np.where(q >= tgt, 1, np.where(q < tge, -1, 0)).astype(np.int32)
    if mode == "gt0":
        return np.where(q >= tgt, 1, 0).astype(np.int32)
    raise ValueError(mode)


# ----------------------------------------------------------------------------- 8-bit log2 code ("l8")
# A tensor with code l8 and exponent E holds int8 codes c whose value is
#   val(c) = 0 for c == 0, sign(c) * 2^(E + (j - 127)/8) with j = min(|c|, 127)
# (c = -128 decodes like -127; the codes are non-decreasing in value, so binarization thresholds are integer
# compares on the codes, exactly as for the linear grid). In fp32: with k = j + 1,
#   val(c) = sign(c) * ldexp(L8_D[k & 7], E + (k >> 3) - 16)
# where L8_D[r] = fp32(2^(r/8)) rounded to nearest (fixed bit patterns below); ldexp is exact.
# Encoding of an fp32 value w (in units of 2^E): with Ew the biased exponent of |w| and t its top 7 mantissa bits,
#   n = 8 (Ew - 127) + #{i : t >= L8_B[i]},  j = clamp(n + 127, 0, 127),  code = sign(w) * j
# i.e. nearest level in the log domain with the rounding boundaries 2^e * 2^((2i+1)/16) snapped to 2^e * (1 + B_i/128)
# (at most 0.34/128 of a mantissa away: < 3 % of a code step); subnormals and 0 -> 0, |w| >= 2^(1/16)-ish -> 127.
L8_D = np.array([0x3F800000, 0x3F8B95C2, 0x3F9837F0, 0x3FA5FED7, 0x3FB504F3, 0x3FC5672A, 0x3FD744FD, 0x3FEAC0C7],
                np.uint32).view(np.float32)
L8_B = (6, 18, 31, 45, 61, 78, 97, 117)
_L8_KF = np.array([sum(t >= b for b in L8_B) for t in range(128)], np.int64)


def l8_decode(c, e: int) -> np.ndarray:
    """fp32 value of l8 codes c with exponent e (exact)."""
    c = np.asarray(c, np.int64)
    k = np.minimum(np.abs(c), 127) + 1
    ex = e + (k >> 3) - 16
    if np.any((ex < -126) | (ex > 127)):
        raise ValueError("l8 exponent out of the fp32 normal range")
    v = np.ldexp(L8_D[k & 7].astype(np.float64), ex).astype(np.float32)
    return np.where(c == 0, np.float32(0), np.where(c < 0, -v, v)).astype(np.float32)


def l8_encode(w) -> np.ndarray:
    """l8 codes (exponent 0) of fp32 values w."""
    b = np.ascontiguousarray(np.asarray(w, np.float32)).view(np.uint32).astype(np.int64)
    ew = (b >> 23) & 0xFF
    t = (b >> 16) & 0x7F
    j = np.clip(8 * (ew - 127) + _L8_KF[t] + 127, 0, 127)
    return np.where((b >> 31) != 0, -j, j).astype(np.int8)


def code_table(a: dict, p: str = "x") -> np.ndarray:
    """Dequantized fp32 value of each of the 256 codes (index q + 128) of an operand described by attributes
    {p}_code ('linear' default: {p}_scale, {p}_zero; 'l8': {p}_exp)."""
    if a.get(p + "_code", "linear") == "l8":
        return l8_decode(_CODES, int(a[p + "_exp"]))
    return deq(_CODES, a[p + "_scale"], int(a[p + "_zero"]))


def qthr_ge_d(t, d: np.ndarray) -> np.ndarray:
    """min{q : d[q + 128] >= t} (128 if none) for a non-decreasing code table d."""
    t = np.asarray(t, np.float32)
    return (np.searchsorted(d, t.reshape(-1), side="left") - 128).reshape(t.shape).astype(np.int32)


def qthr_gt_d(t, d: np.ndarray) -> np.ndarray:
    t = np.asarray(t, np.float32)
    return (np.searchsorted(d, t.reshape(-1), side="right") - 128).reshape(t.shape).astype(np.int32)


def binarize_d(q: np.ndarray, mode: str, t, d: np.ndarray) -> np.ndarray:
    """binarize_q with an explicit code table (linear or l8 operand)."""
    q = np.asarray(q, np.int32)
    t = np.zeros(1, np.float32) if t is None else np.asarray(t, np.float32)
    tge, tgt = qthr_ge_d(t, d), qthr_gt_d(t, d)
    if mode == "ge":
        return np.where(q >= tge, 1, -1).astype(np.int32)
    if mode == "gt":
        return np.where(q >= tgt, 1, -1).astype(np.int32)
    if mode == "sign_zero":
        return np.where(q >= tgt, 1, np.where(q < tge, -1, 0)).astype(np.int32)
    if mode == "gt0":
        return np.where(q >= tgt, 1, 0).astype(np.int32)
    raise ValueError(mode)


def quantize(x: np.ndarray, s: float, z: int) -> np.ndarray:
    """I8Quantize: clamp(rne(fp32(x / s)) + z, -128, 127) with an IEEE fp32 division."""
    v = np.rint((np.asarray(x, np.float32) / f32(s)).astype(np.float32)).astype(np.float32)
    return np.clip(v + f32(z), np.float32(-128), np.float32(127)).astype(np.int8)


# ----------------------------------------------------------------------------- helpers
def _bcast_b(a_shape, b: np.ndarray) -> np.ndarray:
    return np.broadcast_to(b, a_shape) if b.shape != tuple(a_shape) else b


def _im2col_i(x: np.ndarray, kh, kw, sh, sw, dh, dw, pads, padq):
    n, h, w, c = x.shape
    pt, pb, pl, pr = pads
    xp = np.full((n, h + pt + pb, w + pl + pr, c), padq, dtype=np.int32)
    xp[:, pt:pt + h, pl:pl + w, :] = x
    oh = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    ow = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    cols = np.empty((n, oh, ow, kh, kw, c), dtype=np.int32)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, :, ky, kx, :] = xp[:, ky * dh: ky * dh + sh * (oh - 1) + 1: sh, kx * dw: kx * dw + sw * (ow - 1) + 1: sw, :]
    return cols, oh, ow


def _int_conv(xc: np.ndarray, w: np.ndarray, attrs: dict, padv: int) -> np.ndarray:
    """Exact integer convolution of int32 values (float64 BLAS: every partial sum < 2^53)."""
    sh, sw = attrs.get("strides", [1, 1])
    dh, dw = attrs.get("dilations", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    groups = attrs.get("groups", 1)
    o, kh, kw, kic = w.shape
    cols, oh, ow = _im2col_i(xc, kh, kw, sh, sw, dh, dw, pads, padv)
    n = xc.shape[0]
    ocg = o // groups
    out = np.empty((n, oh, ow, o), dtype=np.int64)
    for g in range(groups):
        cg = cols[..., g * kic:(g + 1) * kic].reshape(n * oh * ow, kh * kw * kic).astype(np.float64)
        wg = w[g * ocg:(g + 1) * ocg].reshape(ocg, kh * kw * kic).astype(np.float64)
        out[..., g * ocg:(g + 1) * ocg] = np.rint(cg @ wg.T).astype(np.int64).reshape(n, oh, ow, ocg)
    return out


def _matmul_exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact integer product of small-integer arrays (float64 BLAS; |partial sums| < 2^53)."""
    return np.rint(np.matmul(a.astype(np.float64), b.astype(np.float64))).astype(np.int64)


def _gemm_out(acc: np.ndarray, ins: list, a: dict) -> np.ndarray:
    """Epilogue of I8Conv2D / I8Dense / I8MatMul (per-channel along the last axis). output 'l8': the f32 value
    (scale / bias already in units of 2^y_exp) encoded with l8_encode."""
    if a.get("output", "i8") == "l8":
        y = acc.astype(np.float32) * np.asarray(ins[3], np.float32)
        if ins[2] is not None:
            y = (y.astype(np.float32) + np.asarray(ins[2], np.float32)).astype(np.float32)
        return l8_encode(y)
    if a.get("output", "i8") == "f32":
        scale = np.asarray(ins[3], np.float32)
        y = acc.astype(np.float32) * scale
        if ins[2] is not None:
            y = (y.astype(np.float32) + np.asarray(ins[2], np.float32)).astype(np.float32)
        return y.astype(np.float32)
    if ins[2] is not None:
        acc = acc + np.asarray(ins[2], np.int64)
    acc = sat32(acc)
    y = int(a.get("y_zero", 0)) + requant(acc, np.asarray(ins[3], np.int64), np.asarray(ins[4], np.int64))
    return clamp_q(y, int(a.get("qmin", -128)), int(a.get("qmax", 127)))


# ----------------------------------------------------------------------------- ops
def run_op(op: str, ins: list, a: dict, cache: dict, name: str) -> np.ndarray:
    if op == "I8Quantize":
        return quantize(ins[0], a["scale"], int(a["zero"]))
    if op == "I8Dequantize":
        return deq(ins[0], a["scale"], int(a["zero"]))
    if op == "I8Lut":
        q = np.asarray(ins[0], np.int64) + 128
        tab = np.asarray(ins[1])
        if tab.shape[0] == 1:
            return tab[0][q]
        ax = a.get("axis", -1) % q.ndim
        shp = [1] * q.ndim
        shp[ax] = q.shape[ax]
        ch = np.arange(q.shape[ax]).reshape(shp)
        return tab[ch, q]
    if op in ("I8Add", "I8Mul"):
        qa = np.asarray(ins[0], np.int64) - int(a.get("a_zero", 0))
        qb = _bcast_b(ins[0].shape, np.asarray(ins[1], np.int64) - int(a.get("b_zero", 0)))
        if op == "I8Add":
            ls = int(a.get("left_shift", 20))
            xa = requant(qa << ls, int(a["a_mult"]), int(a["a_shift"]))
            xb = requant(qb << ls, int(a["b_mult"]), int(a["b_shift"]))
            y = int(a["y_zero"]) + requant(xa + xb, int(a["out_mult"]), int(a["out_shift"]))
        else:
            y = int(a["y_zero"]) + requant(qa * qb, int(a["mult"]), int(a["shift"]))
        return clamp_q(y, int(a.get("qmin", -128)), int(a.get("qmax", 127)))
    if op == "I8Conv2D":
        xz = int(a.get("x_zero", 0))
        padq = int(a.get("pad_q", xz))
        acc = _int_conv(np.asarray(ins[0], np.int32) - xz, np.asarray(ins[1], np.int32), a, padq - xz)
        y = _gemm_out(acc, ins, a)
        if len(ins) > 5 and ins[5] is not None:  # bit output: +1 iff acc >= bit_thr[o] (the accumulator before bias)
            return y, np.where(acc >= np.asarray(ins[5], np.int64), 1.0, -1.0).astype(np.float32)
        return y
    if op == "I8Dense":
        x = np.asarray(ins[0], np.int64) - int(a.get("x_zero", 0))
        acc = _matmul_exact(x, np.asarray(ins[1], np.int64).T)
        return _gemm_out(acc, ins, a)
    if op == "I8MatMul":
        qa = np.asarray(ins[0], np.int64) - int(a.get("a_zero", 0))
        qb = np.asarray(ins[1], np.int64) - int(a.get("b_zero", 0))
        if a.get("transpose_b", 0):
            qb = np.swapaxes(qb, -1, -2)
        acc = _matmul_exact(qa, qb)
        if a.get("output", "i8") == "f32":
            return (acc.astype(np.float32) * f32(a["scale"])).astype(np.float32)
        y = int(a.get("y_zero", 0)) + requant(sat32(acc), int(a["mult"]), int(a["shift"]))
        return clamp_q(y, int(a.get("qmin", -128)), int(a.get("qmax", 127)))
    if op == "I8AffineSum":
        # y = clamp(y_zero + requant(sum_i (q_i - z_i) * K[i, c] + B[c], mult, shift)); inputs x_0..x_{N-1}, K, B
        n = len(ins) - 2
        K = np.asarray(ins[n], np.int64)
        B = np.asarray(ins[n + 1], np.int64)
        acc = np.zeros(ins[0].shape, np.int64) + B
        for i in range(n):
            acc = acc + (np.asarray(ins[i], np.int64) - int(a["zeros"][i])) * K[i]
        y = int(a["y_zero"]) + requant(acc, int(a["mult"]), int(a["shift"]))
        return clamp_q(y, int(a.get("qmin", -128)), int(a.get("qmax", 127)))
    if op == "I8LayerNorm":
        return _layernorm(ins, a)
    if op == "I8Softmax":
        return _softmax(ins, a)
    if op == "I8Concat":
        return _concat(ins, a)
    if op == "I8Transpose":
        return np.ascontiguousarray(np.transpose(ins[0], a["perm"]))
    if op == "I8Gather":
        return _gather(ins, a)
    if op == "I8MaxPool2D":
        if np.asarray(ins[0]).dtype != np.int8:  # bit tensor (+-1 view): +1 iff some element of the window is +1
            return _pool(np.asarray(ins[0]).astype(np.int8), a, "max").astype(np.float32)
        return _pool(ins[0], a, "max")
    if op == "I8AvgPool2D":
        return _pool(ins[0], a, "avg")
    if op in ("I8GlobalAvgPool", "I8ReduceMean"):
        return _mean(op, ins[0], a)
    if op == "I8Pad":
        p = a["pads"]
        return np.pad(ins[0], [(p[2 * d], p[2 * d + 1]) for d in range(ins[0].ndim)], constant_values=int(a.get("pad_q", 0))).astype(np.int8)
    if op in ("I8BDense", "I8BConv2D"):
        return _binary_layer(op, ins, a, cache, name)
    if op == "I8BConv2DRes":
        return _bconv_res(ins, a)
    if op == "I8BConv2DResI":
        return _bconv_res_i(ins, a)
    if op == "I8GatedAdd":
        return _gated_add(ins, a)
    if op == "I8GapAffine":
        return _gap_affine(ins, a)
    if op == "I8BMatMul":
        return _bmatmul(ins, a)
    raise NotImplementedError(f"numpy executor: op {op}")


def _layernorm(ins, a):
    x = np.asarray(ins[0], np.int64)
    d = x.shape[-1]
    t = (x - int(a["x_zero"])).reshape(-1, d)
    s1 = t.sum(axis=1)
    s2 = (t * t).sum(axis=1)
    vn = d * s2 - s1 * s1                                    # exact, >= 0
    xs = float(a["x_scale"])
    eps = float(a.get("eps", 1e-5))
    r = np.empty(t.shape[0], np.float32)
    c2 = (eps * d) * d
    for i in range(t.shape[0]):                              # per-row double scalar math, IEEE
        r[i] = np.float32(xs / math.sqrt((xs * xs) * float(vn[i]) + c2))
    u = (d * t - s1[:, None]).astype(np.float32)
    v = (u * r[:, None]).astype(np.float32)
    w = (v * np.asarray(ins[1], np.float32)).astype(np.float32)
    zz = (w + np.asarray(ins[2], np.float32)).astype(np.float32)
    return round_f32(zz, int(a.get("qmin", -128)), int(a.get("qmax", 127))).reshape(x.shape)


def _softmax(ins, a):
    x = np.asarray(ins[0], np.int64)
    n = x.shape[-1]
    tab = np.asarray(ins[1], np.int64)
    q = x.reshape(-1, n)
    d = q.max(axis=1, keepdims=True) - q
    e = tab[d]
    s = e.sum(axis=1)
    f32_out = a.get("output", "i8") == "f32"
    num = 1.0 if f32_out else float(a["inv_scale"])
    r = np.array([np.float32(num / float(si)) for si in s], np.float32)
    v = (e.astype(np.float32) * r[:, None]).astype(np.float32)
    if f32_out:
        return v.reshape(x.shape)
    y = np.rint(v) + np.float32(int(a["y_zero"]))
    return np.clip(y, np.float32(a.get("qmin", -128)), np.float32(a.get("qmax", 127))).astype(np.int8).reshape(x.shape)


def _concat(ins, a):
    yz = int(a["y_zero"])
    parts = []
    for k, x in enumerate(ins):
        m = int(a["mults"][k])
        if m == 0:      # same parameters as the output: plain copy
            parts.append(np.asarray(x, np.int8))
        else:
            y = yz + requant(np.asarray(x, np.int64) - int(a["zeros"][k]), m, int(a["shifts"][k]))
            parts.append(clamp_q(y, -128, 127))
    return np.concatenate(parts, axis=a.get("axis", -1))


def _gather(ins, a):
    table, idx = ins[0], np.asarray(ins[1], np.int64)
    fill = int(a.get("fill_q", 0))
    v = table.shape[0]
    ok = (idx >= 0) & (idx < v)
    safe = np.where(ok, idx, 0)
    if len(ins) > 2 and ins[2] is not None:
        # bit table: row value +1 -> qpos[row], -1 -> qneg[row]; zero_mask bit set -> fill
        bits = table  # unpacked {-1,+1} view
        qpm = np.asarray(ins[2], np.int8)                    # [V, 2] = (qpos, qneg)
        rows = np.where(bits[safe] > 0, qpm[safe][..., 0:1], qpm[safe][..., 1:2]).astype(np.int8)
        if len(ins) > 3 and ins[3] is not None:
            rows = np.where(ins[3][safe] < 0, np.int8(fill), rows)
    else:
        rows = np.asarray(table, np.int8)[safe]
    return np.where(ok[..., None], rows, np.int8(fill)).astype(np.int8)


def _pool(x, a, kind):
    kh, kw = a["kernel"]
    sh, sw = a.get("strides", [kh, kw])
    pt, pb, pl, pr = a.get("pads", [0, 0, 0, 0])
    n, h, w, c = x.shape
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    out = np.empty((n, oh, ow, c), np.int8)
    z = int(a.get("x_zero", 0))
    inc = bool(a.get("count_include_pad", 0))
    for oy in range(oh):
        for ox in range(ow):
            y0, x0 = oy * sh - pt, ox * sw - pl
            ys, ye, xs, xe = max(y0, 0), min(y0 + kh, h), max(x0, 0), min(x0 + kw, w)
            win = np.asarray(x[:, ys:ye, xs:xe, :], np.int64)
            if kind == "max":
                out[:, oy, ox, :] = win.max(axis=(1, 2))
            else:
                cnt = kh * kw if inc else (ye - ys) * (xe - xs)
                acc = (win - z).sum(axis=(1, 2))
                out[:, oy, ox, :] = clamp_q(z + rdiv(acc, max(cnt, 1)), -128, 127)
    return out


def _mean(op, x, a):
    x64 = np.asarray(x, np.int64) - int(a.get("x_zero", 0))
    if op == "I8GlobalAvgPool":
        acc = x64.reshape(x.shape[0], -1, x.shape[3]).sum(axis=1)
        cnt = x.shape[1] * x.shape[2]
        keep = (x.shape[0], 1, 1, x.shape[3]) if a.get("keepdims", 0) else None
    else:
        ax = a.get("axis", -1) % x.ndim
        acc = x64.sum(axis=ax)
        cnt = x.shape[ax]
        keep = tuple(1 if i == ax else s for i, s in enumerate(x.shape)) if a.get("keepdims", 0) else None
    if "mult" in a:
        y = int(a["y_zero"]) + requant(acc, int(a["mult"]), int(a["shift"]))
    else:
        y = int(a.get("x_zero", 0)) + rdiv(acc, cnt)
    y = clamp_q(y, -128, 127)
    return y.reshape(keep) if keep else y


def binary_input(x, a: dict, thr) -> np.ndarray:
    """Binarized operand of I8BDense / I8BConv2D / I8BConv2DRes: int8 codes (linear or l8) against the thresholds, or a
    bit tensor (I8BConv2DRes output 1; the executor's unpacked +-1 view) taken as is."""
    if np.asarray(x).dtype != np.int8:
        return np.where(np.asarray(x) >= 0, 1, -1).astype(np.int32)
    return binarize_d(x, a.get("input_binarize", "ge"), thr, code_table(a, "x"))


def _binary_layer(op, ins, a, cache, name):
    thr = ins[4] if len(ins) > 4 else None
    xb = binary_input(ins[0], a, thr)
    w = ins[1].astype(np.int32)
    if op == "I8BDense":
        if len(ins) > 5 and ins[5] is not None:
            w = np.where(ins[5] < 0, 0, w)
        acc = _matmul_exact(xb, w.T)
    else:
        acc = _int_conv(xb, w, a, int(a.get("pad_value", 0)))
    v = acc.astype(np.float32)
    if len(ins) > 2 and ins[2] is not None:
        v = (v * np.asarray(ins[2], np.float32)).astype(np.float32)
    if len(ins) > 3 and ins[3] is not None:
        v = (v + np.asarray(ins[3], np.float32)).astype(np.float32)
    if a.get("output", "i8") == "f32":
        return v
    return round_f32(v, int(a.get("qmin", -128)), int(a.get("qmax", 127)))


def residual_decode(r: np.ndarray, a: dict) -> np.ndarray:
    """I8BConv2DRes residual in fp32 units of the output (2^y_exp): r_code 'l8' -> l8_decode(r, r_exp) (r_exp relative
    to the output exponent, exact); 'linear' -> fp32(fp32(r - r_zero) * r_scale)."""
    if a.get("r_code", "l8") == "l8":
        return l8_decode(r, int(a["r_exp"]))
    return deq(r, float(a["r_scale"]), int(a.get("r_zero", 0)))


def _opt_in(ins, i):
    return ins[i] if len(ins) > i and ins[i] is not None else None


def bconv_res_pre(v, ins, a):
    """Optional pre stage of I8BConv2DRes between the binary layer's value v and the residual add (2026-11-05; absent
    inputs: no operation): v = v >= 0 ? v : fp32(v * pre_slope[o]) (in(10)); v = fp32(v * pre_mul[o]) (in(11));
    v = fp32(v + pre_add[o]) (in(12)); v = fp32(v * G[n, o]) with G = fp32(fp32(gate - g_zero) * g_scale), gate the int8
    tensor in(13) of shape [N, 1, 1, O] (a per-sample, per-channel squeeze-excitation multiplier)."""
    ps, pm, pa, g = (_opt_in(ins, i) for i in (10, 11, 12, 13))
    if ps is not None:
        v = np.where(v >= 0, v, (v * np.asarray(ps, np.float32)).astype(np.float32)).astype(np.float32)
    if pm is not None:
        v = (v * np.asarray(pm, np.float32)).astype(np.float32)
    if pa is not None:
        v = (v + np.asarray(pa, np.float32)).astype(np.float32)
    if g is not None:
        gv = deq(g, float(a["g_scale"]), int(a.get("g_zero", 0))).reshape(v.shape[0], 1, 1, v.shape[-1])
        v = (v * gv).astype(np.float32)
    return v


def lin_encode(w, a: dict) -> np.ndarray:
    """I8BConv2DRes output "i8" (2026-11-13, int8_full v2: the fused block writes the tensor's linear int8 grid; constants
    in units of its scale): y = clamp(rne(w) + y_zero, qmin, qmax), rne the round-half-to-even conversion to an integer
    (FCVTNS; saturating, so values beyond +-2^20 give the same y)."""
    r = np.clip(np.rint(np.asarray(w, np.float32)), np.float32(-2.0 ** 20), np.float32(2.0 ** 20))
    y = r.astype(np.int64) + int(a.get("y_zero", 0))
    return clamp_q(y, int(a.get("qmin", -128)), int(a.get("qmax", 127)))


def _bconv_res(ins, a):
    """I8BConv2DRes: binary convolution + residual + per-channel PReLU (+ bias) -> l8 code (or, output "i8", the linear
    code lin_encode(w)), and optionally the
    binarization of the pre-activation sum for the next binary layer (output 1, bit tensor):
    v = fp32(fp32(acc) * post_mul[o]) + post_add[o];  v = bconv_res_pre(v) (optional inputs 10-13);  u = v + R(res);  w = u >= 0 ? u : fp32(u * slope[o]);
    w = w + out_bias[o] (if given);  y = l8_encode(w);  bit: +1 iff s >= bit_thr[o] (bit_binarize ge) / s > (gt), with
    s = u (bit_src "sum", default) or s = w (bit_src "out": the next layer binarizes the block output, e.g. a PReLU
    with slopes <= 0 that no threshold on u can express) (every operation one IEEE fp32 operation).
    Returns y or (y, bits as the +-1 view)."""
    thr = ins[4] if len(ins) > 4 else None
    xb = binary_input(ins[0], a, thr)
    acc = _int_conv(xb, ins[1].astype(np.int32), a, int(a.get("pad_value", 0)))
    v = (acc.astype(np.float32) * np.asarray(ins[2], np.float32)).astype(np.float32)
    v = (v + np.asarray(ins[3], np.float32)).astype(np.float32)
    v = bconv_res_pre(v, ins, a)
    u = (v + residual_decode(ins[5], a)).astype(np.float32)
    w = np.where(u >= 0, u, (u * np.asarray(ins[6], np.float32)).astype(np.float32)).astype(np.float32)
    if len(ins) > 7 and ins[7] is not None:
        w = (w + np.asarray(ins[7], np.float32)).astype(np.float32)
    y = l8_encode(w) if a.get("output", "l8") == "l8" else lin_encode(w, a)
    if len(ins) > 8 and ins[8] is not None:
        t = np.asarray(ins[8], np.float32)
        s = w if a.get("bit_src", "sum") == "out" else u
        pos = (s > t) if a.get("bit_binarize", "ge") == "gt" else (s >= t)
        return y, np.where(pos, 1.0, -1.0).astype(np.float32)
    return y


def bconv_res_i_bound(K: int, m1, m2, b, slope=None, k: int = 0, ob=None, pb=None) -> int:
    """Largest |value| the I8BConv2DResI epilogue can reach (pre-stage a, acc and w; int32 arithmetic is exact iff < 2^31)."""
    pre = K * int(np.max(np.abs(np.asarray(m1, np.int64))))
    if pb is not None:
        pre += int(np.max(np.abs(np.asarray(pb, np.int64))))
    acc = pre + 128 * int(np.max(np.abs(np.asarray(m2, np.int64)))) + int(np.max(np.abs(np.asarray(b, np.int64))))
    w = acc
    if slope is not None:
        w = max(acc, ((acc * int(np.max(np.abs(np.asarray(slope, np.int64)))) >> 31) + 1) << k)
    if ob is not None:
        w += int(np.max(np.abs(np.asarray(ob, np.int64))))
    return max(acc, w)


def _bconv_res_i(ins, a):
    """I8BConv2DResI (int8_full v2, 2026-11-13): binary convolution + residual block with an integer epilogue. With dot the
    exact binary dot product (int8_ops._int_conv) and q the residual code (in(5), int8 [N, OH, OW, O]):
        acc = dot * m1[o] + q * m2[o] + b[o]                                     in(6) m1, in(8) m2, in(7) b (int32, exact)
              (with the pre stage in(12) pre_b, in(13) pre_lo, in(14) pre_hi:
               acc = clamp(dot * m1[o] + pre_b[o], pre_lo[o], pre_hi[o]) + q * m2[o] + b[o]; e.g. a ReLU and the sign of the
               batch norm after it, QuickNet's BConv[relu] -> BatchNorm -> Add)
        w   = acc < 0 ? (mulhi(acc, slope[o]) << slope_shift) : acc                in(10) slope (optional; mulhi = SQRDMULH)
        w   = w + out_bias[o]                                                       in(11) (optional)
        y   = clamp(rshr(w, shift) + y_zero, qmin, qmax)                            output 0 (int8, linear)
        bit = +1 iff s >= bit_thr[o], s = acc (bit_src "sum", default) or w ("out")  in(9) (optional), output 1
    Attribute r_tile (default 1): the residual has O / r_tile channels, channel o reads residual channel o mod (O / r_tile).
    The residual in(5) (and then m2) may be absent (q * m2 = 0: a binary convolution with an integer PReLU epilogue, BNext's
    BConv -> PReLU). Inputs 2 / 3 are unused (empty); in(4) the input-binarization thresholds as for I8BConv2D. The quantizer bounds every
    term (bconv_res_i_bound < 2^31), so every value is an exact int32."""
    thr = ins[4] if len(ins) > 4 else None
    xb = binary_input(ins[0], a, thr)
    w1 = ins[1].astype(np.int32)
    dot = _int_conv(xb, w1, a, int(a.get("pad_value", 0))).astype(np.int64)
    m1, b = (np.asarray(ins[i], np.int64) for i in (6, 7))
    m2 = np.asarray(ins[8], np.int64) if _opt_in(ins, 5) is not None else np.zeros(1, np.int64)   # no residual: q * m2 = 0
    sl, ob, pb = _opt_in(ins, 10), _opt_in(ins, 11), _opt_in(ins, 12)
    K = int(np.prod(w1.shape[1:]))
    if bconv_res_i_bound(K, m1, m2, b, sl, int(a.get("slope_shift", 0)), ob, pb) >= 2 ** 31:
        raise ValueError("I8BConv2DResI: constants exceed the int32 bound")
    pre = dot * m1
    if pb is not None:
        pre = np.clip(pre + np.asarray(pb, np.int64), np.asarray(ins[13], np.int64), np.asarray(ins[14], np.int64))
    acc = pre + b
    if _opt_in(ins, 5) is not None:
        r = np.asarray(ins[5], np.int64)
        rt = int(a.get("r_tile", 1))
        if rt > 1:   # residual of O / r_tile channels, repeated along the channels (BHViT's repeat)
            r = np.concatenate([r] * rt, axis=-1)
        acc = acc + r * m2
    w = acc
    if sl is not None:
        neg = mulhi(acc, np.asarray(sl, np.int64)) << int(a.get("slope_shift", 0))
        w = np.where(acc < 0, neg, acc)
    if ob is not None:
        w = w + np.asarray(ob, np.int64)
    y = clamp_q(rshr(w, int(a.get("shift", 0))) + int(a.get("y_zero", 0)), int(a.get("qmin", -128)), int(a.get("qmax", 127)))
    bt = _opt_in(ins, 9)
    if bt is not None:
        s = w if a.get("bit_src", "sum") == "out" else acc
        return y, np.where(s >= np.asarray(bt, np.int64), 1.0, -1.0).astype(np.float32)
    return y


def gated_add_bound(kx, kp, b) -> int:
    """Largest |acc| of I8GatedAdd (also of its folded form q_x kx + q_p (g - z_g) kp + b'), < 2^31 required."""
    return 256 * int(np.max(np.abs(np.asarray(kx, np.int64)))) + 65280 * int(np.max(np.abs(np.asarray(kp, np.int64)))) + \
        int(np.max(np.abs(np.asarray(b, np.int64))))


def _gated_add(ins, a):
    """I8GatedAdd (int8_full v2, 2026-11-13): a residual add of a squeeze-excitation-gated branch (BNext's
    x + norm(sigmoid(se) * p)), in integers. x, p int8 [N, H, W, C], g int8 [N, 1, 1, C] (the gate's code):
        acc = (x - x_zero) * kx[c] + (p - p_zero) * (g[n, c] - g_zero) * kp[c] + b[c]      (exact int32, gated_add_bound)
        y   = clamp(rshr(acc, shift) + y_zero, qmin, qmax)
        bit_j = +1 iff acc >= bit_thr[j, c]          (in(6) i32 [nb, C], optional: outputs 1 .. nb)"""
    x, p, g = (np.asarray(ins[i], np.int64) for i in (0, 1, 2))
    kx, kp, b = (np.asarray(ins[i], np.int64) for i in (3, 4, 5))
    if gated_add_bound(kx, kp, b) >= 2 ** 31:
        raise ValueError("I8GatedAdd: constants exceed the int32 bound")
    acc = (x - int(a.get("x_zero", 0))) * kx + (p - int(a.get("p_zero", 0))) * (g - int(a.get("g_zero", 0))) * kp + b
    y = clamp_q(rshr(acc, int(a.get("shift", 0))) + int(a.get("y_zero", 0)), int(a.get("qmin", -128)), int(a.get("qmax", 127)))
    bt = _opt_in(ins, 6)
    if bt is None:
        return y
    T = np.asarray(bt, np.int64).reshape(-1, x.shape[-1])
    return (y,) + tuple(np.where(acc >= T[j], 1.0, -1.0).astype(np.float32) for j in range(T.shape[0]))


def _gap_affine(ins, a):
    """I8GapAffine (int8_full v2, 2026-11-13): the global average pool of an affine sum of int8 tensors, without the sum's
    own rounding (BNext's GAP(mix), mix = s * x + (1 - s) * p). x_1..x_n int8 [N, H, W, C] (inputs 0 .. n-1), then
    K i32 [n, C] and B i32 [C]:
        S_i = sum over H, W of (x_i - zeros[i])                                 (exact integers)
        acc = sum_i S_i * K[i, c] + B[c]                                         (int64, exact)
        y   = clamp(rshr(acc, shift) + y_zero, qmin, qmax)                       [N, 1, 1, C]"""
    n = len(ins) - 2
    K = np.asarray(ins[n], np.int64)
    B = np.asarray(ins[n + 1], np.int64)
    zs = a.get("zeros", [0] * n)
    acc = B.reshape(1, 1, 1, -1).astype(np.int64)
    for i in range(n):
        S = (np.asarray(ins[i], np.int64) - int(zs[i])).sum(axis=(1, 2), keepdims=True)
        acc = acc + S * K[i]
    return clamp_q(rshr(acc, int(a.get("shift", 0))) + int(a.get("y_zero", 0)), int(a.get("qmin", -128)), int(a.get("qmax", 127)))


def _bmatmul(ins, a):
    A = binarize_q(ins[0], a.get("a_binarize", "sign_zero"), a.get("a_threshold", 0.0), a["a_scale"], int(a["a_zero"]))
    B = binarize_q(ins[1], a.get("b_binarize", "sign_zero"), a.get("b_threshold", 0.0), a["b_scale"], int(a["b_zero"]))
    if a.get("transpose_b", 0):
        B = np.swapaxes(B, -1, -2)
    k = A.shape[-1]
    ab = _matmul_exact(A, B)
    acc = int(a.get("c_ab", 1)) * ab
    if int(a.get("c_rs", 0)):
        acc = acc + int(a["c_rs"]) * A.astype(np.int64).sum(axis=-1, keepdims=True)
    if int(a.get("c_cs", 0)):
        acc = acc + int(a["c_cs"]) * B.astype(np.int64).sum(axis=-2, keepdims=True)
    acc = acc + int(a.get("c0", 0))
    del k
    v = acc.astype(np.float32)
    if len(ins) > 2 and ins[2] is not None:
        v = (v * np.asarray(ins[2], np.float32)).astype(np.float32)
    if len(ins) > 3 and ins[3] is not None:
        v = (v + np.asarray(ins[3], np.float32)).astype(np.float32)
    if a.get("output", "i8") == "f32":
        return v
    return round_f32(v, int(a.get("qmin", -128)), int(a.get("qmax", 127)))
