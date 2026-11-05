"""Image-quality metrics of the restoration / super-resolution sources, re-implemented exactly.

Three protocols are used by the converted image-to-image models (docs/numerical-correctness.md, S8):

* ``kair``: BBCU-denoiseAndblocking (KAIR) utils/utils_image.py ``calculate_psnr`` / ``calculate_ssim`` on
  uint8 images (``tensor2uint``: clamp to [0, 1], x255, round), RGB (SSIM averaged over channels) or gray.
* ``basicsr_y``: BBCU-SR (BasicSR) bbcu/metrics/psnr_ssim.py with ``test_y_channel=True``: the uint8 output is
  converted to Y (``bgr2ycbcr(img / 255, y_only)`` x 255, float, no rounding) after removing ``crop_border``.
* ``bsrc_y``: BSRC-2023 tools/metrics (torchmetrics PSNR, data_range 255) on the luminance
  ``(65.481 R + 128.553 G + 24.966 B) / 255 + 16`` of the *unrounded* output clamped to [0, 255]
  (``convert_to_default_image_range``), border ``scale`` removed.

SSIM is the Wang et al. Gaussian-window (11, 1.5) formula of KAIR / BasicSR (cv2.filter2D, valid region).
All functions take HWC arrays.
"""
from __future__ import annotations

import math

import numpy as np


def to_uint8_unit(img: np.ndarray) -> np.ndarray:
    """KAIR tensor2uint / BasicSR tensor2img for [0, 1] float images: clamp, x255, round (half to even, numpy)."""
    x = np.clip(np.asarray(img, np.float32), 0, 1)
    return np.uint8(np.round(x * np.float32(255.0)))


def psnr_kair(a: np.ndarray, b: np.ndarray, border: int = 0) -> float:
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    h, w = a.shape[:2]
    a = a[border:h - border, border:w - border].astype(np.float64)
    b = b[border:h - border, border:w - border].astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))


def _ssim_1ch(a: np.ndarray, b: np.ndarray) -> float:
    import cv2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.transpose())
    mu1 = cv2.filter2D(a, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, win)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.filter2D(a ** 2, -1, win)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(b ** 2, -1, win)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(a * b, -1, win)[5:-5, 5:-5] - mu12
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(m.mean())


def ssim_channels(a: np.ndarray, b: np.ndarray, border: int = 0) -> float:
    """KAIR calculate_ssim / BasicSR calculate_ssim (mean over channels)."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    h, w = a.shape[:2]
    a = a[border:h - border, border:w - border]
    b = b[border:h - border, border:w - border]
    if a.ndim == 2:
        return _ssim_1ch(a, b)
    return float(np.mean([_ssim_1ch(a[..., i], b[..., i]) for i in range(a.shape[2])]))


def _basicsr_y(img_rgb_u8: np.ndarray) -> np.ndarray:
    # bgr2ycbcr(img, y_only=True): img float32 in [0,1]; out = dot(img, [24.966, 128.553, 65.481]) + 16 (BGR order),
    # then _convert_output_type_range(out, float32) = out / 255; to_y_channel multiplies by 255.
    x = img_rgb_u8.astype(np.float32) / np.float32(255.0)
    out = np.dot(x[..., ::-1], [24.966, 128.553, 65.481]) + 16.0
    out = (out / 255.0).astype(np.float32)
    return out[..., None] * np.float32(255.0)


def psnr_ssim_basicsr_y(out_rgb_u8: np.ndarray, gt_rgb_u8: np.ndarray, crop_border: int) -> tuple[float, float]:
    a, b = out_rgb_u8, gt_rgb_u8
    if crop_border:
        a = a[crop_border:-crop_border, crop_border:-crop_border]
        b = b[crop_border:-crop_border, crop_border:-crop_border]
    ya, yb = _basicsr_y(a).astype(np.float64), _basicsr_y(b).astype(np.float64)
    mse = np.mean((ya - yb) ** 2)
    p = float("inf") if mse == 0 else float(10.0 * np.log10(255.0 * 255.0 / mse))
    return p, ssim_channels(ya, yb)


def psnr_bsrc_y(out_rgb_255: np.ndarray, gt_rgb_255: np.ndarray, boundary: int) -> float:
    """BSRC-2023 PSNR(min_val=0, max_val=255, boundary_size=scale) on one image (float32 arithmetic as torch)."""
    def lum(x):
        x = np.clip((np.asarray(x, np.float32) - np.float32(0)) / np.float32(255), 0, 1) * np.float32(255)
        x = x[boundary:-boundary, boundary:-boundary]
        s = np.asarray([65.481, 128.553, 24.966], np.float32)
        t = (x * s) / np.float32(255.0) + np.float32(16.0)
        return t[..., 0] + t[..., 1] + t[..., 2]
    d = (lum(out_rgb_255) - lum(gt_rgb_255)).astype(np.float32)
    mse = float(np.mean((d * d).astype(np.float64)))   # torchmetrics sums in float32; float64 here (<1e-5 dB apart)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


METRICS = ("kair_rgb", "kair_gray", "basicsr_y", "bsrc_y")


def image_quality(out: np.ndarray, gt_u8: np.ndarray, metric: str, border: int = 0, ssim: bool = True) -> tuple[float, float | None]:
    """(PSNR, SSIM or None) of one HWC model output against its uint8 ground truth with the source's protocol.
    kair_* and basicsr_y expect [0, 1] outputs (quantized like tensor2uint); bsrc_y expects [0, 255] outputs."""
    if metric in ("kair_rgb", "kair_gray"):
        o = to_uint8_unit(out)
        g = gt_u8
        if metric == "kair_gray":
            o, g = o[..., 0], gt_u8[..., 0]
        return psnr_kair(o, g, border), (ssim_channels(o, g, border) if ssim else None)
    if metric == "basicsr_y":
        p, s = psnr_ssim_basicsr_y(to_uint8_unit(out), gt_u8, border)
        return p, (s if ssim else None)
    if metric == "bsrc_y":
        return psnr_bsrc_y(out, gt_u8.astype(np.float32), border), None
    raise ValueError(metric)


def psnr_vs(reference: np.ndarray, candidate: np.ndarray, data_range: float) -> float:
    """Unquantized PSNR of a candidate output against a reference output (e.g. Bitino vs the float64 source)."""
    d = np.asarray(reference, np.float64) - np.asarray(candidate, np.float64)
    mse = float(np.mean(d * d))
    return float("inf") if mse == 0 else float(10.0 * np.log10(data_range * data_range / mse))


def compare_images(reference: np.ndarray, candidate: np.ndarray, data_range: float) -> dict:
    """Image-output analogue of metrics.compare_outputs: [N, ...] outputs compared per sample (no argmax)."""
    ref = np.asarray(reference, np.float64).reshape(len(reference), -1)
    cand = np.asarray(candidate, np.float64).reshape(len(candidate), -1)
    assert ref.shape == cand.shape, (ref.shape, cand.shape)
    d = np.abs(ref - cand)
    per_max = d.max(axis=1)
    per_psnr = [psnr_vs(r, c, data_range) for r, c in zip(ref, cand)]
    finite = [p for p in per_psnr if np.isfinite(p)]
    return {"n": int(len(ref)), "max_abs_error": float(d.max()), "mean_abs_error": float(d.mean()),
            "per_sample_max_abs": [float(v) for v in per_max],
            "psnr_vs_reference_min_db": float(min(finite)) if finite else float("inf"),
            "psnr_vs_reference_overall_db": psnr_vs(ref, cand, data_range),
            "n_samples_identical": int(np.sum(per_max == 0)), "exact_match": bool(np.array_equal(ref, cand)),
            "data_range": float(data_range)}
