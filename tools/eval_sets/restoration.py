#!/usr/bin/env python3
"""Fixed-size crop eval sets for the image-restoration / super-resolution models (slice S8), plus the degradation
functions shared with tools/restoration_eval.py (full-image PSNR/SSIM).

Degradations follow the official test scripts exactly:

* ``denoise`` (BBCU main_test_dncnn_denoising_b.py:148-165): ``img = imread_uint(path, 3)`` (cv2, RGB),
  ``img_L = uint2single(img)`` (``np.float32(img / 255.)``), ``np.random.seed(0)`` then
  ``img_L += np.random.normal(0, sigma / 255., img_L.shape)`` (float32 += float64 noise, unclipped) per image.
* ``jpeg_y`` (BBCU main_test_dncnn3_deblocking_b.py:146-177): ``imread_uint(path, 3)`` -> KAIR ``rgb2ycbcr``
  (uint8 in, Y = round(dot(img, [65.481, 128.553, 24.966]) / 255 + 16), uint8) -> ``cv2.imencode('.jpg', Y,
  [IMWRITE_JPEG_QUALITY, q])`` -> ``cv2.imdecode(enc, 0)`` -> ``uint2tensor4`` (float32 / 255). Target: Y.
* ``sr`` (BasicSR / EDSR benchmark convention): HR read with cv2 IMREAD_COLOR (gray -> 3 channels),
  ``modcrop`` to a multiple of the scale, LR = MATLAB-style bicubic ``imresize(HR / 255, 1 / scale)`` (BBCU-SR
  bbcu/utils/matlab_functions.py, executed from the clone) rounded to uint8 as MATLAB ``imwrite`` does. BBCU
  input: ``LR / 255`` (float32, BasicSR ``img.astype(np.float32) / 255.``); BSRC input: ``float(LR)`` in
  [0, 255] (bnn_competition SuperResolutionDataset). The official LR_bicubic PNGs of the EDSR benchmark
  package were not reachable; the HF mirror's LR images (eugenesiow/*) are not MATLAB bicubic (up to 14 levels
  off on Set5), so they are not used (HR files only, identical to cszn/DnCNN's Set14).

The crop sets hold one center crop per image (fixed graph input shape), the matching ground-truth crop
(``targets``, uint8) and the metric definition. Images are never committed.

  python3 -I tools/eval_sets/restoration.py denoise --images <CBSD68 dir> --sigma 25 --crop 96 --out .cache/data/eval68_cbsd68_sigma25_c96
  python3 -I tools/eval_sets/restoration.py jpeg_y --images <LIVE1 dir> <classic5 dir> --qualities 10 40 --crop 96 \
      --out .cache/data/eval68_jpeg_y_q10q40_c96
  python3 -I tools/eval_sets/restoration.py sr --images <Set5 HR> <Set14 HR> <B100 HR> --limit 0 0 45 --scale 2 --crop 48 \
      --matlab-functions <BBCU>/BBCU-SR/bbcu/utils/matlab_functions.py --out .cache/data/eval64_sr_x2_lr48
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

IMG_EXT = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP", ".tif")


def image_paths(d: str) -> list[str]:
    """KAIR get_image_paths / BasicSR scandir order: sorted file names."""
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.endswith(IMG_EXT))


def sha256_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def imread_uint_rgb(path: str) -> np.ndarray:
    """KAIR utils_image.imread_uint(path, 3)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def noisy(img_u8: np.ndarray, sigma: float) -> np.ndarray:
    img_l = np.float32(img_u8 / 255.)
    np.random.seed(seed=0)
    img_l += np.random.normal(0, sigma / 255., img_l.shape)
    return img_l


def rgb2y_kair(img_u8: np.ndarray) -> np.ndarray:
    """KAIR utils_image.rgb2ycbcr(uint8, only_y=True)."""
    rlt = np.dot(img_u8, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt.round().astype(np.uint8)


def jpeg_gray(y_u8: np.ndarray, q: int) -> np.ndarray:
    import cv2
    ok, enc = cv2.imencode(".jpg", y_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    assert ok
    return cv2.imdecode(enc, 0)


def u8_to_unit_f32(x_u8: np.ndarray) -> np.ndarray:
    """torch ``.float().div(255.)`` / BasicSR ``astype(float32) / 255.`` (one float32 division)."""
    return (x_u8.astype(np.float32) / np.float32(255.0)).astype(np.float32)


def imread_rgb_color(path: str) -> np.ndarray:
    """cv2 IMREAD_COLOR (gray replicated to 3 channels) -> RGB uint8 (BasicSR imfrombytes + bgr2rgb)."""
    import cv2
    return cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def modcrop(img: np.ndarray, s: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % s, : w - w % s]


_MF = {}


def matlab_bicubic_lr(hr_u8: np.ndarray, s: int, matlab_functions: str) -> np.ndarray:
    """LR = round(clip(imresize(HR / 255, 1 / s) * 255)) with BasicSR's MATLAB-compatible imresize."""
    if matlab_functions not in _MF:
        import importlib.machinery
        import importlib.util
        sys.dont_write_bytecode = True
        loader = importlib.machinery.SourceFileLoader("bbcu_matlab_functions", matlab_functions)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _MF[matlab_functions] = mod
    r = _MF[matlab_functions].imresize(hr_u8.astype(np.float64) / 255.0, 1 / s)
    return np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)


def center_crop(img: np.ndarray, size: int, y0: int | None = None, x0: int | None = None) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    assert h >= size and w >= size, (img.shape, size)
    y0 = (h - size) // 2 if y0 is None else y0
    x0 = (w - size) // 2 if x0 is None else x0
    return img[y0:y0 + size, x0:x0 + size], y0, x0


def _write(out: str, name: str, inputs: dict, targets: np.ndarray, samples: list, manifest_extra: dict) -> None:
    os.makedirs(out, exist_ok=True)
    pre = {}
    for key, (arr, definition) in inputs.items():
        fn = f"{key}.f32"
        a = np.ascontiguousarray(arr.astype(np.float32))
        a.tofile(os.path.join(out, fn))
        pre[key] = {"shape": list(a.shape), "dtype": "float32", "layout": "NHWC", "file": fn,
                    "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "definition": definition,
                    "script": "tools/eval_sets/restoration.py"}
    t = np.ascontiguousarray(targets.astype(np.uint8))
    t.tofile(os.path.join(out, "targets.u8"))
    manifest = {"schema": "bitino.eval_set.v1", "name": name, "preprocessing": pre,
                "targets": {"file": "targets.u8", "shape": list(t.shape), "dtype": "uint8", "layout": "NHWC",
                            "sha256": hashlib.sha256(t.tobytes()).hexdigest()},
                "samples": samples, **manifest_extra}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"wrote {len(samples)} samples -> {out}: " + ", ".join(f"{k} {v['shape']}" for k, v in pre.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=("denoise", "jpeg_y", "sr"))
    ap.add_argument("--images", nargs="+", required=True, help="image directories (one per test set)")
    ap.add_argument("--limit", nargs="+", type=int, help="per directory: first N sorted images (0 = all)")
    ap.add_argument("--sigma", type=float, default=25)
    ap.add_argument("--qualities", nargs="+", type=int, default=[10, 40])
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--crop", type=int, required=True, help="crop size (SR: LR crop)")
    ap.add_argument("--matlab-functions", help="BBCU-SR/bbcu/utils/matlab_functions.py (sr)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lim = a.limit or [0] * len(a.images)
    files = []
    for d, n in zip(a.images, lim):
        p = image_paths(d)
        files += [(os.path.basename(os.path.normpath(d)), f) for f in (p[:n] if n else p)]
    common = {"source_files": {f"{s}/{os.path.basename(f)}": sha256_file(f) for s, f in files},
              "limitations": "One center crop per test image (fixed graph input shape); PSNR/SSIM on crops are smoke/regression "
                             "numbers. Full-image metrics: tools/restoration_eval.py."}
    if a.task == "denoise":
        xs, ts, samples = [], [], []
        for s, f in files:
            img = imread_uint_rgb(f)
            nz = noisy(img, a.sigma)
            c, y0, x0 = center_crop(img, a.crop)
            xs.append(nz[y0:y0 + a.crop, x0:x0 + a.crop])
            ts.append(c)
            samples.append({"file": f"{s}/{os.path.basename(f)}", "label": -1, "crop_yx": [y0, x0], "image_hw": list(img.shape[:2])})
        key = f"bbcu_denoise_s{int(a.sigma)}"
        _write(a.out, f"cbsd68-sigma{int(a.sigma)}-center{a.crop}", {key: (np.stack(xs), (
            f"BBCU main_test_dncnn_denoising_b.py: imread_uint(RGB) -> uint2single -> np.random.seed(0); += N(0, {a.sigma}/255) "
            f"(unclipped, per image, full image), then center crop {a.crop}x{a.crop}"))}, np.stack(ts), samples,
            {**common, "image_task": {"kind": "denoise", "output_range": [0.0, 1.0], "metric": "kair_rgb", "crop_border": 0,
                                      "sigma": a.sigma}})
    elif a.task == "jpeg_y":
        xs, ts, samples = [], [], []
        for q in a.qualities:
            for s, f in files:
                y = rgb2y_kair(imread_uint_rgb(f))
                yj = jpeg_gray(y, q)
                c, y0, x0 = center_crop(y, a.crop)
                xs.append(u8_to_unit_f32(yj[y0:y0 + a.crop, x0:x0 + a.crop])[..., None])
                ts.append(c[..., None])
                samples.append({"file": f"{s}/{os.path.basename(f)}", "label": -1, "quality": q, "crop_yx": [y0, x0],
                                "image_hw": list(y.shape[:2])})
        import cv2
        _write(a.out, f"live1-classic5-jpeg-y-q{'q'.join(map(str, a.qualities))}-center{a.crop}", {"bbcu_jpeg_y": (np.stack(xs), (
            f"BBCU main_test_dncnn3_deblocking_b.py: imread_uint(RGB) -> KAIR rgb2ycbcr (Y, uint8) -> cv2 {cv2.__version__} JPEG "
            f"q in {a.qualities} -> imdecode gray -> float32/255, center crop {a.crop}x{a.crop}"))}, np.stack(ts), samples,
            {**common, "image_task": {"kind": "jpeg_deblocking", "output_range": [0.0, 1.0], "metric": "kair_gray", "crop_border": 0,
                                      "qualities": a.qualities}})
    else:
        assert a.matlab_functions, "--matlab-functions required for sr"
        s_ = a.scale
        lrs, ts, samples = [], [], []
        for s, f in files:
            hr = modcrop(imread_rgb_color(f), s_)
            lr = matlab_bicubic_lr(hr, s_, a.matlab_functions)
            c, y0, x0 = center_crop(lr, a.crop)
            lrs.append(c)
            ts.append(hr[y0 * s_:(y0 + a.crop) * s_, x0 * s_:(x0 + a.crop) * s_])
            samples.append({"file": f"{s}/{os.path.basename(f)}", "label": -1, "lr_crop_yx": [y0, x0], "hr_hw": list(hr.shape[:2])})
        lr = np.stack(lrs)
        d = (f"HR (cv2 IMREAD_COLOR, RGB) modcrop {s_} -> MATLAB-style bicubic imresize(1/{s_}) (BBCU-SR matlab_functions.py) "
             f"-> uint8 -> center LR crop {a.crop}x{a.crop}")
        _write(a.out, f"set5-set14-b100-x{s_}-lr{a.crop}", {
            f"sr_x{s_}_unit": (u8_to_unit_f32(lr), d + " -> float32 / 255 (BBCU / BasicSR)"),
            f"sr_x{s_}_255": (lr.astype(np.float32), d + " -> float32 in [0, 255] (BSRC-2023)")}, np.stack(ts), samples,
            {**common, "image_task": {"kind": "super_resolution", "scale": s_, "metric": "basicsr_y", "crop_border": s_,
                                      "output_range_by_preprocessing": {f"sr_x{s_}_unit": [0.0, 1.0], f"sr_x{s_}_255": [0.0, 255.0]}}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
