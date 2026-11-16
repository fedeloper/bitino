#!/usr/bin/env python3
"""Reference outputs for android/deepfake's Resample (the detectors' test preprocessing on a rectangle of the screen).

  python3 android/tools/make_resample_fixture.py   # -> android/deepfake/src/test/resources/resample_fixture.txt

The frame is defined by a formula (no image data): v(x, y, c) = (x*x // 7 + y*3 + c*60 + (x*y) % 17) % 256. For each
rectangle: PIL crop -> resize((ow, oh), BILINEAR) with torchvision's Resize(252) sizes -> CenterCrop(224) offsets
(Python round) -> u8, sampled every 9th row and column.
"""
import os

import numpy as np
from PIL import Image

W, H = 900, 1200
RECTS = [(40, 100, 340, 600), (500, 700, 620, 790), (0, 200, 756, 625), (100, 100, 352, 352), (37, 41, 811, 1180)]
STEP = 9


def frame():
    y, x = np.mgrid[0:H, 0:W].astype(np.int64)
    return np.stack([(x * x // 7 + y * 3 + c * 60 + (x * y) % 17) % 256 for c in range(3)], -1).astype(np.uint8)


def main() -> None:
    im = Image.fromarray(frame())
    cases = []
    for (x0, y0, x1, y1) in RECTS:
        w, h = x1 - x0, y1 - y0
        ow, oh = (252, int(252 * h / w)) if w <= h else (int(252 * w / h), 252)
        r = np.asarray(im.crop((x0, y0, x1, y1)).resize((ow, oh), Image.BILINEAR))
        top, left = int(round((oh - 224) / 2.0)), int(round((ow - 224) / 2.0))
        c = r[top:top + 224, left:left + 224]
        cases.append({"rect": [x0, y0, x1, y1], "resized": [ow, oh], "step": STEP,
                      "samples": c[::STEP, ::STEP].reshape(-1).tolist()})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../deepfake/src/test/resources/resample_fixture.txt")
    with open(out, "w") as f:
        f.write(f"# frame {W}x{H}, v(x, y, c) = (x*x // 7 + y*3 + c*60 + (x*y) % 17) % 256; PIL BILINEAR; every {STEP}th row/col\n")
        for c in cases:
            f.write("rect " + " ".join(map(str, c["rect"])) + " resized " + " ".join(map(str, c["resized"])) + "\n")
            f.write(" ".join(map(str, c["samples"])) + "\n")
    print(out, len(cases), "cases")


if __name__ == "__main__":
    main()
