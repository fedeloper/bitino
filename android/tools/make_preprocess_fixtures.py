#!/usr/bin/env python3
"""Generates the preprocessing fixtures of the Android library's unit tests (PreprocessingTest).

Synthetic RGB images (xorshift32 noise + gradients + a checkerboard, the same generator as the Kotlin test) are
written as PNG and preprocessed with the Python pipelines the models' evaluation tensors were made with:

  torchvision_imagenet  tools/build_eval_set.py make_preprocessor("torchvision_imagenet") (PIL bilinear Resize 256,
                        CenterCrop 224, ToTensor, ImageNet Normalize)
  omz_bgr               tools/eval_sets/omz_bgr.py preprocess() (OMZ Accuracy Checker: scale_greater bilinear 256,
                        center crop 224, BGR, 0..255)
  torchvision_bicubic   the same torchvision transform with InterpolationMode.BICUBIC (BHViT / Bi-ViT family)
  cifar_norm            ToTensor + Normalize on 32x32 images (no resize)

Each line of the output: <spec> <width> <height> <seed> <sha256 of the float32 NHWC output> <sum (float64)>.
Run with the environment that built the eval sets (torchvision 0.21.0, Pillow 12.2.0):

  python3 android/tools/make_preprocess_fixtures.py > android/bitino/src/test/resources/preprocess_fixtures.txt
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "eval_sets"))

IMAGES = [(500, 375, 1), (333, 500, 2), (640, 480, 3), (480, 640, 4), (256, 256, 5), (224, 300, 6), (150, 100, 7),
          (1000, 77, 8), (257, 1023, 9), (299, 299, 10)]
CIFAR_IMAGES = [(32, 32, 11), (32, 32, 12)]


def synth(w: int, h: int, seed: int) -> np.ndarray:
    """Must match PreprocessingTest.synth (Kotlin)."""
    s = (seed * 0x9E3779B9 + 1) & 0xFFFFFFFF
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            s ^= (s << 13) & 0xFFFFFFFF
            s ^= s >> 17
            s ^= (s << 5) & 0xFFFFFFFF
            r = (x * 255 // max(w - 1, 1)) ^ (s & 0x0F)
            g = (y * 255 // max(h - 1, 1)) ^ ((s >> 4) & 0x0F)
            b = ((x // 8 + y // 8) % 2) * 200 + ((s >> 24) & 0x37)
            img[y, x] = (r, g, b)
    return img


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    import PIL
    import torchvision
    from PIL import Image
    from torchvision import transforms

    from build_eval_set import make_preprocessor
    import omz_bgr

    tv_fn, _ = make_preprocessor("torchvision_imagenet")
    bicubic = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    cifar = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    def emit(spec, w, h, seed, arr):
        arr = np.ascontiguousarray(arr.astype(np.float32))
        print(f"{spec} {w} {h} {seed} {hashlib.sha256(arr.astype('<f4').tobytes()).hexdigest()} "
              f"{float(arr.astype(np.float64).sum())!r}")

    print(f"# torchvision {torchvision.__version__}, Pillow {PIL.__version__}; android/tools/make_preprocess_fixtures.py")
    with tempfile.TemporaryDirectory() as d:
        for w, h, seed in IMAGES:
            img = synth(w, h, seed)
            raw = png_bytes(img)
            emit("torchvision_imagenet", w, h, seed, tv_fn(raw))
            p = os.path.join(d, f"{seed}.png")
            open(p, "wb").write(raw)
            emit("omz_bgr", w, h, seed, omz_bgr.preprocess(p))
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            emit("torchvision_bicubic", w, h, seed, bicubic(pil).numpy().transpose(1, 2, 0))
        for w, h, seed in CIFAR_IMAGES:
            pil = Image.open(io.BytesIO(png_bytes(synth(w, h, seed)))).convert("RGB")
            emit("cifar_norm", w, h, seed, cifar(pil).numpy().transpose(1, 2, 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
