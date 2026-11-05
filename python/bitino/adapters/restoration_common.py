"""IR building blocks shared by the image-restoration / super-resolution adapters (torch_bbcu, torch_bsrc).

No new operator: every construct is an exact composition of existing IR ops (docs/operators_torch_cnn.md, S8).

* ``pixel_shuffle`` (torch.nn.PixelShuffle(r), NCHW channel ``c*r*r + i*r + j`` -> pixel ``(h*r+i, w*r+j)``):
  in NHWC, ``Reshape [1,H,W,C,r,r] -> Transpose [0,1,4,2,5,3] -> Reshape [1,H*r,W*r,C]`` (pure data movement).
* ``nearest_upsample`` (F.interpolate(mode='nearest', scale_factor=r), source index ``floor(dst / r)``):
  one row ``Gather`` from the input viewed as ``[H*W, C]`` with constant i32 indices ``(y//r)*W + x//r`` (copy).
* ``bilinear_upsample`` (F.interpolate(mode='bilinear', align_corners=False, scale_factor=r), integer r):
  replicate-pad by one pixel (row ``Gather``, indices ``clamp(y-1)*W + clamp(x-1)``), then a 3x3 ``Conv2D``
  without padding whose ``C*r*r`` outputs are the ``r x r`` sub-pixel phases, then ``pixel_shuffle``. Phase ``i``
  of an axis reads the source position ``h + d_i``, ``d_i = (i + 0.5)/r - 0.5``: taps ``(h-1, h)`` with weights
  ``(-d_i, 1+d_i)`` for ``d_i < 0``, ``(h, h+1)`` with ``(1-d_i, d_i)`` otherwise; the 2-D weight is the product.
  torch clamps a negative source coordinate to 0 and the upper neighbour index to ``H-1``; with replicate
  padding both cases read the border pixel twice, and ``(-d)*x + (1+d)*x = x`` exactly (dyadic weights, the
  products and their sum are exact in the reference's double accumulator). The interior arithmetic is the
  same weighted sum as torch's ``h0l*(w0l*x00 + w1l*x01) + h1l*(w0l*x10 + w1l*x11)`` up to rounding order.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph


def pixel_shuffle(g: Graph, x: str, h: int, w: int, c_in: int, r: int, name: str) -> str:
    assert c_in % (r * r) == 0, (c_in, r)
    c = c_in // (r * r)
    t = g.add_node("Reshape", [x], {"shape": [1, h, w, c, r, r]}, name=name + "/reshape_in")
    t = g.add_node("Transpose", [t], {"perm": [0, 1, 4, 2, 5, 3]}, name=name + "/transpose")
    return g.add_node("Reshape", [t], {"shape": [1, h * r, w * r, c]}, name=name)


def _row_gather(g: Graph, x: str, h: int, w: int, c: int, idx: np.ndarray, name: str) -> str:
    oh, ow = idx.shape
    t = g.add_node("Reshape", [x], {"shape": [h * w, c]}, name=name + "/rows")
    ids = g.add_constant(name + "/indices", idx.astype(np.int32), dtype="i32")
    t = g.add_node("Gather", [t, ids], name=name + "/gather")          # -> [oh, ow, c]
    return g.add_node("Reshape", [t], {"shape": [1, oh, ow, c]}, name=name)


def nearest_upsample(g: Graph, x: str, h: int, w: int, c: int, r: int, name: str) -> str:
    yy, xx = np.meshgrid(np.arange(h * r) // r, np.arange(w * r) // r, indexing="ij")
    return _row_gather(g, x, h, w, c, yy * w + xx, name)


def replicate_pad1(g: Graph, x: str, h: int, w: int, c: int, name: str) -> str:
    yy, xx = np.meshgrid(np.clip(np.arange(h + 2) - 1, 0, h - 1), np.clip(np.arange(w + 2) - 1, 0, w - 1), indexing="ij")
    return _row_gather(g, x, h, w, c, yy * w + xx, name)


def bilinear_phase_weights(r: int) -> np.ndarray:
    """[r, 3] weights over the taps (h-1, h, h+1) for each output phase (float64, dyadic)."""
    out = np.zeros((r, 3))
    for i in range(r):
        d = (i + 0.5) / r - 0.5
        if d < 0:
            out[i, 0], out[i, 1] = -d, 1.0 + d
        else:
            out[i, 1], out[i, 2] = 1.0 - d, d
    return out


def bilinear_kernel(c: int, r: int) -> np.ndarray:
    """OHWI 3x3 kernel [c*r*r, 3, 3, c]: output channel c*r*r + i*r + j = phase (i, j) of input channel c."""
    p = bilinear_phase_weights(r)
    k = np.zeros((c * r * r, 3, 3, c))
    for ch in range(c):
        for i in range(r):
            for j in range(r):
                k[ch * r * r + i * r + j, :, :, ch] = np.outer(p[i], p[j])
    assert np.array_equal(k.astype(np.float32).astype(np.float64), k)   # exactly representable
    return k


def bilinear_upsample(g: Graph, x: str, h: int, w: int, c: int, r: int, name: str) -> str:
    p = replicate_pad1(g, x, h, w, c, name + "/replicate_pad")
    k = g.add_constant(name + "/phase_kernel", bilinear_kernel(c, r).astype(np.float32), layout="OHWI")
    t = g.add_node("Conv2D", [p, k, ""], {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1,
                                          "pad_value": 0.0, "activation": "none"}, name=name + "/phases")
    return pixel_shuffle(g, t, h, w, c * r * r, r, name)
