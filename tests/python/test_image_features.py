"""Image-feature operators (ImageGrayU8, FftLogMagnitude, LocalBinaryPattern; docs/operators_image_features.md).

* numpy helpers vs independent references: polynomial sin/cos and log vs libm, the mixed-radix FFT vs numpy.fft;
* numpy executor vs the C++ runtime, bit-exact, at stages reference and neon (the optimized kernels), 1 and 3
  threads, for several image sizes (radices 2, 3, 4, 5, 7, odd lengths), LBP 'uniform' / 'default' with integral
  and fractional column offsets, out-of-range and NaN pixels;
* when OpenCV and scikit-image are importable: ImageGrayU8 / LocalBinaryPattern equal cv2.cvtColor(BGR2GRAY) /
  skimage.feature.local_binary_pattern exactly, FftLogMagnitude equals 20*log(|fftshift(fft2)| + 1e-9)/255 in float32
  on natural-like images (the "Faster Than Lies" preprocessing).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, image_ops  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
GRAY = {"coeffs": [3735, 19235, 9798], "shift": 15, "scale": 255.0}
FFT = {"eps": 1e-9, "mul": 20.0, "div": 255.0, "shift": 1}


def test_sincos_and_log_accuracy():
    for n in (1, 2, 7, 12, 30, 224, 105):
        c, s = image_ops.sincos_2pi(n)
        for t in range(n):
            assert abs(c[t] - math.cos(2 * math.pi * t / n)) < 1.2e-15
            assert abs(s[t] - math.sin(2 * math.pi * t / n)) < 1.2e-15
    c, s = image_ops.sincos_2pi(8)
    assert list(c[[0, 2, 4, 6]]) == [1.0, 0.0, -1.0, 0.0] and list(s[[0, 2, 4, 6]]) == [0.0, 1.0, 0.0, -1.0]
    x = np.exp(np.random.default_rng(0).uniform(-25, 25, 100000))
    ref = np.log(x)
    assert np.max(np.abs(image_ops.log_f64(x) - ref) / np.spacing(np.abs(ref))) <= 2.0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 12, 30, 35, 105, 224])
def test_fft_matches_numpy(n):
    rng = np.random.default_rng(n)
    a, b = rng.standard_normal((n, 5)), rng.standard_normal((n, 5))
    c, s = image_ops.sincos_2pi(n)
    yr, yi = image_ops.fft_axis0(a, b, c, s)
    ref = np.fft.fft(a + 1j * b, axis=0)
    assert np.max(np.abs(yr + 1j * yi - ref)) <= 1e-14 * np.max(np.abs(ref))


def feature_graph(h: int, w: int, lbp: dict) -> Graph:
    g = Graph()
    x = g.add_input("input", [1, h, w, 3])
    gray = g.add_node("ImageGrayU8", [x], GRAY)
    f = g.add_node("FftLogMagnitude", [gray], FFT)
    lb = g.add_node("LocalBinaryPattern", [gray], lbp)
    g.add_node("Concat", [x, gray, f, lb], {"axis": -1}, name="cat")
    g.set_outputs("cat:out")
    return g


def lbp_attrs(points, radius, method="uniform"):
    rp, cp = image_ops.lbp_offsets(points, radius)
    return {"rp": rp, "cp": cp, "method": method, "div": 255.0}


def images(h, w, n, seed):
    rng = np.random.default_rng(seed)
    xs = []
    for i in range(n):
        if i % 3 == 0:
            x = rng.random((1, h, w, 3), dtype=np.float32)
        else:   # torchvision ToDtype(scale=True) values k * fl32(1/255), smooth with noise
            yy, xx = np.mgrid[0:h, 0:w]
            base = 128 + 100 * np.sin(yy[..., None] / (3 + i) + xx[..., None] / 5 + np.arange(3))
            k = np.clip(base + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)
            x = (k.astype(np.float32) * np.float32(1 / 255))[None]
        if i == 1:
            x[0, 0, 0] = [-0.5, 1.7, np.nan]
            x[0, -1, -1] = [np.inf, -np.inf, 1.0]
        xs.append(np.ascontiguousarray(x, np.float32))
    return xs


CASES = [((224, 224), lbp_attrs(3, 6)), ((30, 42), lbp_attrs(3, 6)), ((15, 21), lbp_attrs(4, 2.5, "default")),
         ((12, 7), lbp_attrs(8, 1.0)), ((5, 35), lbp_attrs(3, 6)), ((1, 1), lbp_attrs(3, 6))]


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,threads", [("reference", 1), ("neon", 1), ("neon", 3), ("all", 2)])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_numpy_executor_bit_exact_with_cpp(tmp_path, stage, threads, case):
    (h, w), lbp = CASES[case]
    g = feature_graph(h, w, lbp)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    xs = images(h, w, 4, case)
    np.concatenate(xs).tofile(tmp_path / "in.f32")
    r = subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                        "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", str(threads),
                        "--json", str(tmp_path / "r.json")], check=True, capture_output=True, text=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(len(xs), -1)
    for i, xin in enumerate(xs):
        want = executor.run(g2, {"input": xin})[g2.outputs[0]].ravel()
        np.testing.assert_array_equal(got[i].view(np.uint32), want.view(np.uint32))
    if stage != "reference":
        import json
        kernels = json.dumps(json.load(open(tmp_path / "r.json")))
        assert "fft_log_magnitude.opt" in kernels
        assert "local_binary_pattern.opt" in kernels   # fractional column offsets included (bilinear in both axes)


def test_matches_opencv_skimage_numpy():
    try:   # a broken install (e.g. skimage built against another numpy) raises ValueError, not ImportError
        import cv2
        from skimage import feature
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"OpenCV / scikit-image unavailable: {e}")
    rp, cp = image_ops.lbp_offsets(3, 6)
    for i, x in enumerate(images(224, 224, 6, 11)[2:]):
        hwc = x[0]
        gray_src = cv2.cvtColor((hwc * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = image_ops.image_gray_u8(x, GRAY)
        np.testing.assert_array_equal(gray[0, :, :, 0], gray_src)
        lbp_src = (feature.local_binary_pattern(gray_src, 3, 6, method="uniform") / 255).astype(np.float32)
        np.testing.assert_array_equal(image_ops.local_binary_pattern(gray, {"rp": rp, "cp": cp, "method": "uniform", "div": 255.0})[0, :, :, 0],
                                      lbp_src)
        fft_src = (20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_src))) + 1e-9) / 255).astype(np.float32)
        f = image_ops.fft_log_magnitude(gray, FFT)[0, :, :, 0]
        assert np.max(np.abs(f - fft_src)) <= 2 * np.spacing(np.float32(np.max(np.abs(fft_src))))
