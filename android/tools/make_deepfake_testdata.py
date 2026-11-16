#!/usr/bin/env python3
"""Test data for the deepfake app's instrumented test (android/deepfake/src/androidTest), from a development checkout.

  python3 android/tools/make_deepfake_testdata.py [--n 8] [models...]

Writes android/deepfake/build/testdata/ (build output; never committed: the images come from the CIFAKE eval subset):
  input.f32                      first N images of .cache/data/eval128_cifake_ftl (ftl_rgb01, N x 224 x 224 x 3)
  expected_<model>.f32           the OnePlus Open's stage-optimized outputs for them (N x 2; scripts/device_final_suite.sh)
  screen_images.u8, screen_labels.txt   2 ImageNet eval images (u8) and their labels, for the classification-mode test
Default models: ftl-deepfake-bnext-t-cocofake ftl-deepfake-bnext-t-dffd (the ones bundled by -Pbitino.deepfakeModels).
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["ftl-deepfake-bnext-t-cocofake", "ftl-deepfake-bnext-t-dffd"])
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    ev = os.path.join(ROOT, ".cache/data/eval128_cifake_ftl")
    pp = json.load(open(os.path.join(ev, "manifest.json")))["preprocessing"]["ftl_rgb01"]
    x = np.fromfile(os.path.join(ev, pp["file"]), np.float32).reshape(pp["shape"])[: a.n]
    out = os.path.join(ROOT, "android/deepfake/build/testdata")
    os.makedirs(out, exist_ok=True)
    x.tofile(os.path.join(out, "input.f32"))
    for m in a.models:
        ref = np.fromfile(os.path.join(ROOT, "artifacts", m, "device", "cph2551", f"{m}_optimized.f32"), np.float32)
        ref.reshape(-1, 2)[: a.n].tofile(os.path.join(out, f"expected_{m}.f32"))
    # classification mode: two ImageNet eval images (u8, 224 x 224 x 3) and their ImageNet labels
    iv = os.path.join(ROOT, ".cache/data/eval128_imagenet_ftl")
    im = json.load(open(os.path.join(iv, "manifest.json")))
    ip = im["preprocessing"]["ftl_rgb01"]
    xi = np.fromfile(os.path.join(iv, ip["file"]), np.float32).reshape(ip["shape"])[:2]
    (xi * 255 + 0.5).clip(0, 255).astype(np.uint8).tofile(os.path.join(out, "screen_images.u8"))
    with open(os.path.join(out, "screen_labels.txt"), "w") as f:
        f.write("\n".join(str(s["imagenet_label"]) for s in im["samples"][:2]) + "\n")
    print(f"{out}: {a.n} images, models {' '.join(a.models)}; 2 ImageNet images for the classification test")


if __name__ == "__main__":
    main()
