"""Image-feature operators (docs/operators_image_features.md): numpy executor counterparts.

ImageGrayU8, FftLogMagnitude and LocalBinaryPattern compute the hand-crafted input channels of the
"Faster Than Lies" deepfake detector (FFT magnitude + local binary pattern next to RGB) inside the
model, so the preprocessing runs in the runtime on the phone.

Every float operation below is an IEEE double (or float32) +, -, *, /, sqrt in a fixed order, with
no library transcendental: the twiddle factors come from a fixed polynomial sin/cos (``sincos_2pi``)
and the logarithm from a fixed polynomial (``log_f64``). The C++ kernels
(runtime/kernels/reference/image_features_reference.cc, image_features_opt.cc, shared arithmetic in
runtime/kernels/image_features_common.h) perform the same operations in the same order, so the
results are bit-identical on every platform (no dependence on libm, numpy's FFT or BLAS).
Hex constants are shared literally with the C++ header.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial

import numpy as np

# --------------------------------------------------------------------------- polynomial sin/cos
# sin x = x + x*(z*P_s(z)), cos x = 1 + z*P_c(z), z = x*x, |x| <= pi/4; P by Horner from the
# highest coefficient. Coefficients are (-1)^k/(2k+1)! and (-1)^k/(2k)!, correctly rounded.
SIN_C = [float(Fraction((-1) ** k, factorial(2 * k + 1))) for k in range(1, 9)]   # k = 1..8
COS_C = [float(Fraction((-1) ** k, factorial(2 * k))) for k in range(1, 10)]      # k = 1..9
PI_4 = float.fromhex("0x1.921fb54442d18p-1")
# log(m) = 2s + 2s*R(z), s = (m-1)/(m+1), z = s*s, R = z*(L1 + z*(L2 + ...)), L_k = 1/(2k+1)
LOG_C = [float(Fraction(1, 2 * k + 1)) for k in range(1, 12)]                      # k = 1..11
LN2_HI = float.fromhex("0x1.62e42fee00000p-1")   # fdlibm ln2_hi (trailing zeros: k*LN2_HI exact for |k| < 2^11)
LN2_LO = float.fromhex("0x1.a39ef35793c76p-33")  # fdlibm ln2_lo
SQRT2 = float.fromhex("0x1.6a09e667f3bcdp+0")


def _sin_cos_poly(x: np.ndarray):
    z = x * x
    p = np.full_like(x, SIN_C[-1])
    for c in SIN_C[-2::-1]:
        p = c + z * p
    s = x + x * (z * p)
    q = np.full_like(x, COS_C[-1])
    for c in COS_C[-2::-1]:
        q = c + z * q
    co = 1.0 + z * q
    return s, co


def sincos_2pi(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(cos(2 pi t / n), sin(2 pi t / n)) for t = 0..n-1 by octant reduction (exact integer arithmetic)
    and the polynomials above: phi = (PI_4 * r) / n with r in [0, n]."""
    t = np.arange(n, dtype=np.int64)
    o = (8 * t) // n
    r = 8 * t - o * n
    r = np.where(o % 2 == 1, n - r, r)
    phi = (PI_4 * r.astype(np.float64)) / float(n)
    s, c = _sin_cos_poly(phi)
    cos_t = np.select([o == 0, o == 1, o == 2, o == 3, o == 4, o == 5, o == 6], [c, s, -s, -c, -c, -s, s], c)
    sin_t = np.select([o == 0, o == 1, o == 2, o == 3, o == 4, o == 5, o == 6], [s, c, c, s, -s, -c, -c], -s)
    return cos_t, sin_t


def log_f64(x: np.ndarray) -> np.ndarray:
    """Natural logarithm of positive normal finite doubles (fixed polynomial, <= ~2 ulp)."""
    x = np.ascontiguousarray(x, np.float64)
    bits = x.view(np.uint64)
    e = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64) - 1023
    m = ((bits & np.uint64(0x000FFFFFFFFFFFFF)) | np.uint64(0x3FF0000000000000)).view(np.float64)
    big = m > SQRT2
    m = np.where(big, m * 0.5, m)
    e = np.where(big, e + 1, e)
    f = m - 1.0
    s = f / (2.0 + f)
    z = s * s
    p = np.full_like(z, LOG_C[-1])
    for c in LOG_C[-2::-1]:
        p = c + z * p
    r = z * p
    t = s + s
    k = e.astype(np.float64)
    return k * LN2_HI + (t + (t * r + k * LN2_LO))


# --------------------------------------------------------------------------- FFT
def fft_factors(n: int) -> list[int]:
    """Radix order of the 1-D transform: 4s, then one 2, then odd primes ascending."""
    f = []
    while n % 4 == 0:
        f.append(4)
        n //= 4
    if n % 2 == 0:
        f.append(2)
        n //= 2
    p = 3
    while n > 1:
        while n % p == 0:
            f.append(p)
            n //= p
        p += 2
    return f


def _cmul(ar, ai, wr, wi):
    return ar * wr - ai * wi, ar * wi + ai * wr


def fft_axis0(re: np.ndarray, im: np.ndarray, tw_c: np.ndarray, tw_s: np.ndarray):
    """Complex DFT (forward, exp(-2 pi i k n / N)) along axis 0 of [N, L] arrays, every column (lane)
    independently: mixed-radix Stockham decimation in frequency (docs/operators_image_features.md)."""
    N, L = re.shape
    wr_all, wi_all = tw_c, -tw_s            # W_N^t = exp(-2 pi i t / N)
    n, s = N, 1
    for p in fft_factors(N):
        m = n // p
        x_r = re.reshape(p, m, s, L)
        x_i = im.reshape(p, m, s, L)
        y_r = np.empty((m, p, s, L))
        y_i = np.empty((m, p, s, L))
        step = N // p
        if p == 2:
            b = [(x_r[0] + x_r[1], x_i[0] + x_i[1]), (x_r[0] - x_r[1], x_i[0] - x_i[1])]
        elif p == 4:
            t0r, t0i = x_r[0] + x_r[2], x_i[0] + x_i[2]
            t1r, t1i = x_r[0] - x_r[2], x_i[0] - x_i[2]
            t2r, t2i = x_r[1] + x_r[3], x_i[1] + x_i[3]
            t3r, t3i = x_r[1] - x_r[3], x_i[1] - x_i[3]
            b = [(t0r + t2r, t0i + t2i), (t1r + t3i, t1i - t3r), (t0r - t2r, t0i - t2i), (t1r - t3i, t1i + t3r)]
        else:
            # odd prime p, conjugate-pair form: s_q = a_q + a_{p-q}, d_q = a_q - a_{p-q} (q = 1..h, h = (p-1)/2);
            # b_0 = a_0 + s_1 + ... + s_h; for r = 1..h, A = a_0 + sum_q s_q cos(2 pi qr/p), B = sum_q d_q sin(2 pi qr/p)
            # (q order, real coefficients): b_r = (A.re + B.im, A.im - B.re), b_{p-r} = (A.re - B.im, A.im + B.re)
            h = (p - 1) // 2
            sr = [x_r[q] + x_r[p - q] for q in range(1, h + 1)]
            si = [x_i[q] + x_i[p - q] for q in range(1, h + 1)]
            dr = [x_r[q] - x_r[p - q] for q in range(1, h + 1)]
            di = [x_i[q] - x_i[p - q] for q in range(1, h + 1)]
            b0r, b0i = x_r[0], x_i[0]
            for q in range(h):
                b0r, b0i = b0r + sr[q], b0i + si[q]
            b = [None] * p
            b[0] = (b0r, b0i)
            for r in range(1, h + 1):
                Ar, Ai = x_r[0], x_i[0]
                Br = Bi = None
                for q in range(1, h + 1):
                    e = (q * r) % p
                    c, sn = wr_all[e * step], -wi_all[e * step]
                    Ar, Ai = Ar + sr[q - 1] * c, Ai + si[q - 1] * c
                    pr_, pi_ = dr[q - 1] * sn, di[q - 1] * sn
                    Br, Bi = (pr_, pi_) if Br is None else (Br + pr_, Bi + pi_)
                b[r] = (Ar + Bi, Ai - Br)
                b[p - r] = (Ar - Bi, Ai + Br)
        tstep = N // n
        j = np.arange(m)
        for r in range(p):
            br, bi = b[r]
            if r == 0:
                y_r[:, 0], y_i[:, 0] = br, bi
                continue
            idx = (j * r * tstep) % N
            wr = wr_all[idx][:, None, None]
            wi = wi_all[idx][:, None, None]
            pr, pi = _cmul(br, bi, wr, wi)
            y_r[:, r] = np.where((j == 0)[:, None, None], br, pr)
            y_i[:, r] = np.where((j == 0)[:, None, None], bi, pi)
        re, im = y_r.reshape(N, L), y_i.reshape(N, L)
        n, s = m, s * p
    return re, im


def fft2_magnitude(g: np.ndarray) -> np.ndarray:
    """|FFT2(g)| of a real [H, W] image: transform along H (all columns), then along W for rows
    k1 = 0..H//2; rows k1 > H//2 are mirrored, |X[k1, k2]| = |X[H-k1, (W-k2) % W]|."""
    h, w = g.shape
    ch, sh = sincos_2pi(h)
    cw, sw = sincos_2pi(w)
    yr, yi = fft_axis0(np.asarray(g, np.float64), np.zeros((h, w)), ch, sh)
    hh = h // 2 + 1
    zr, zi = fft_axis0(np.ascontiguousarray(yr[:hh].T), np.ascontiguousarray(yi[:hh].T), cw, sw)   # [W, hh]
    mag_half = np.sqrt(zr * zr + zi * zi).T                                                     # [hh, W]
    mag = np.empty((h, w))
    mag[:hh] = mag_half
    for k1 in range(hh, h):
        mag[k1] = mag_half[h - k1][(-np.arange(w)) % w]
    return mag


def fft_log_magnitude(x: np.ndarray, attrs: dict) -> np.ndarray:
    """FftLogMagnitude: x [N, H, W, 1] -> float32 [N, H, W, 1],
    y = f32((mul * log(|FFT2(x)| + eps)) / div), fftshift-ed when shift = 1."""
    eps, mul, div = float(attrs.get("eps", 1e-9)), float(attrs.get("mul", 20.0)), float(attrs.get("div", 255.0))
    shift = int(attrs.get("shift", 1))
    n, h, w, c = x.shape
    assert c == 1
    out = np.empty((n, h, w, 1), np.float32)
    for b in range(n):
        mag = fft2_magnitude(np.asarray(x[b, :, :, 0], np.float64))
        v = ((mul * log_f64(mag + eps)) / div).astype(np.float32)
        if shift:
            v = np.roll(v, (h // 2, w // 2), axis=(0, 1))
        out[b, :, :, 0] = v
    return out


# --------------------------------------------------------------------------- gray, LBP
def image_gray_u8(x: np.ndarray, attrs: dict) -> np.ndarray:
    """ImageGrayU8: x [N, H, W, 3] float -> float32 [N, H, W, 1] holding integers 0..255:
    u_c = trunc(clamp(f32(x_c * scale), 0, 255)), gray = (sum_c u_c * coeffs[c] + 2^(shift-1)) >> shift."""
    scale = np.float32(attrs.get("scale", 255.0))
    coeffs = [int(v) for v in attrs["coeffs"]]
    shift = int(attrs["shift"])
    v = np.asarray(x, np.float32) * scale
    v = np.where(v >= np.float32(0), v, np.float32(0))
    v = np.where(v < np.float32(255), v, np.float32(255))
    u = v.astype(np.int64)
    acc = u[..., 0] * coeffs[0] + u[..., 1] * coeffs[1] + u[..., 2] * coeffs[2] + (1 << (shift - 1))
    return (acc >> shift).astype(np.float32)[..., None]


def local_binary_pattern(x: np.ndarray, attrs: dict) -> np.ndarray:
    """LocalBinaryPattern (scikit-image local_binary_pattern semantics, 'default' / 'uniform'):
    x [N, H, W, 1] -> float32 [N, H, W, 1] = f32(lbp / div). Sample i at (r + rp[i], c + cp[i]),
    bilinear with constant 0 outside, bit_i = (texture_i - x[r, c] >= 0), all in double."""
    rp = [float(v) for v in attrs["rp"]]
    cp = [float(v) for v in attrs["cp"]]
    method = attrs.get("method", "uniform")
    div = float(attrs.get("div", 1.0))
    P = len(rp)
    n, h, w, _ = x.shape
    out = np.empty((n, h, w, 1), np.float32)
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    for b in range(n):
        img = np.asarray(x[b, :, :, 0], np.float64)
        bits = []
        for i in range(P):
            rr = np.broadcast_to(rows + rp[i], (h, w))
            cc = np.broadcast_to(cols + cp[i], (h, w))
            minr, maxr = np.floor(rr), np.ceil(rr)
            minc, maxc = np.floor(cc), np.ceil(cc)
            dr, dc = rr - minr, cc - minc

            def px(ri, ci):
                ri, ci = ri.astype(np.int64), ci.astype(np.int64)
                ok = (ri >= 0) & (ri < h) & (ci >= 0) & (ci < w)
                return np.where(ok, img[np.clip(ri, 0, h - 1), np.clip(ci, 0, w - 1)], 0.0)
            top = (1 - dc) * px(minr, minc) + dc * px(minr, maxc)
            bottom = (1 - dc) * px(maxr, minc) + dc * px(maxr, maxc)
            tex = (1 - dr) * top + dr * bottom
            bits.append((tex - img) >= 0)
        if method == "uniform":
            changes = sum((bits[i] != bits[i + 1]).astype(np.int64) for i in range(P - 1))
            ones = sum(bt.astype(np.int64) for bt in bits)
            lbp = np.where(changes <= 2, ones, P + 1).astype(np.float64)
        elif method == "default":
            lbp = sum(bt.astype(np.int64) << i for i, bt in enumerate(bits)).astype(np.float64)
        else:
            raise ValueError(f"LocalBinaryPattern: unsupported method {method}")
        out[b, :, :, 0] = (lbp / div).astype(np.float32)
    return out


def lbp_offsets(points: int, radius: float) -> tuple[list[float], list[float]]:
    """Sample offsets exactly as scikit-image 0.25 _local_binary_pattern computes them."""
    rr = -radius * np.sin(2 * np.pi * np.arange(points, dtype=np.float64) / points)
    cc = radius * np.cos(2 * np.pi * np.arange(points, dtype=np.float64) / points)
    return [float(v) for v in np.round(rr, 5)], [float(v) for v in np.round(cc, 5)]
