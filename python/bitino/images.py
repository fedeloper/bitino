"""Image preprocessing for the zoo's image classifiers, with Pillow + numpy only (no torch / TensorFlow).

Each zoo model records the preprocessing it was validated with (registry/recipes.json ``eval.preprocess``, defined
in registry/eval_sets.json). The keys below are re-implemented from those definitions: the resize, crop and
normalisation are the same operations in the same order (Pillow does the resampling, as in torchvision's PIL path
and in OpenVINO's Accuracy Checker with use_pillow), so results match the eval sets up to float rounding in the
normalisation. Models whose preprocessing needs the source framework (``larq_zoo``: TensorFlow image ops) are not
covered: pass an already preprocessed array to Model.predict() instead.

    x = bitino.images.load_image("cat.jpg", "torchvision_imagenet")   # float32, 1 x 224 x 224 x 3 (NHWC)
"""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# key -> (resize short side, interpolation, crop size, normalisation, channel order)
#   normalisation "unit": x/255 then (x - mean) / std (torchvision ToTensor + Normalize)
#                 "u8":   (x - 255 mean) / (255 std) on the uint8 values (timm PrefetchLoader)
#                 "raw":  0..255 as float32 (mean subtraction inside the model)
SPECS = {
    "torchvision_imagenet": dict(resize=256, interp="bilinear", crop=224, norm="unit", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "bitorch_imagenet": dict(resize=256, interp="bilinear", crop=224, norm="unit", mean=IMAGENET_MEAN, std=(0.229, 0.224, 0.255)),
    "timm_bicubic_crop095": dict(resize=235, interp="bicubic", crop=224, norm="u8", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "omz_bgr": dict(resize=256, interp="bilinear", crop=224, norm="raw", bgr=True, omz=True),
    # CIFAR-style classifiers take 32 x 32 inputs: any photo is reduced (short side 32, centre crop).
    "cifar10_imagenet_norm": dict(resize=32, interp="bilinear", crop=32, norm="unit", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "cifar100_imagenet_norm": dict(resize=32, interp="bilinear", crop=32, norm="unit", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "cifar10_bireal_norm": dict(resize=32, interp="bilinear", crop=32, norm="unit", mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2023, 0.1994, 0.201)),
}


def supported() -> list[str]:
    return sorted(SPECS)


def _pil():
    try:
        from PIL import Image
    except ImportError as e:   # pragma: no cover - depends on the environment
        raise ImportError("image decoding needs Pillow: pip install 'bitino[image]' (or pip install pillow)") from e
    return Image


def preprocess(img, key: str) -> np.ndarray:
    """A PIL image (any mode) -> float32 array 1 x H x W x C in the layout and scale the model expects."""
    if key not in SPECS:
        raise ValueError(f"no built-in preprocessing {key!r} (built in: {', '.join(supported())}); "
                         f"pass a preprocessed array to predict() instead")
    Image = _pil()
    sp = SPECS[key]
    img = img.convert("RGB")
    w, h = img.size
    s = sp["resize"]
    if sp.get("omz"):
        # Accuracy Checker resize (aspect_ratio_scale: greater): the short side becomes s, the other is truncated.
        if w < h:
            nw, nh = s, int(h * s / w)
        else:
            nw, nh = int(w * s / h), s
    else:
        # torchvision.transforms.Resize(int): short side s, long side int(s * long / short)
        if w <= h:
            nw, nh = s, int(s * h / w)
        else:
            nw, nh = int(s * w / h), s
    resample = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[sp["interp"]]
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), resample)
    c = sp["crop"]
    top, left = int(round((nh - c) / 2.0)), int(round((nw - c) / 2.0))
    img = img.crop((left, top, left + c, top + c))
    x = np.asarray(img, dtype=np.uint8)
    if sp["norm"] == "unit":
        x = x.astype(np.float32) / np.float32(255.0)
        x = (x - np.asarray(sp["mean"], np.float32)) / np.asarray(sp["std"], np.float32)
    elif sp["norm"] == "u8":
        x = (x.astype(np.float32) - np.asarray([m * 255 for m in sp["mean"]], np.float32)) / \
            np.asarray([v * 255 for v in sp["std"]], np.float32)
    else:
        x = x.astype(np.float32)
    if sp.get("bgr"):
        x = x[..., ::-1]
    return np.ascontiguousarray(x[None], dtype=np.float32)


def load_image(path_or_file, key: str) -> np.ndarray:
    """Decodes an image file (path or file object) and applies preprocess()."""
    Image = _pil()
    with Image.open(path_or_file) as im:
        im.load()
        return preprocess(im, key)
