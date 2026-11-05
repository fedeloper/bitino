#!/usr/bin/env python3
"""PyTorch port of the official tensorpack DoReFa-Net AlexNet (examples/DoReFa-Net/alexnet-dorefa.py, --dorefa 1,1,32),
used as the float64 reference of slice S12 (the official TF graph, run by tools/tensorpack_dorefa_reference.py,
is float32 only). Input: the normalized NHWC BGR tensor of tools/eval_sets/tensorpack_bgr.py (= the output of the
graph's image_preprocess, checked bit for bit by the official runner).

Ported from (``.cache/repos/tensorpack``, read only):
  * dorefa.py get_dorefa(1, 1, 32) (l.8-64): fw (bitW == 1, l.25-32): E = mean(|W|) over the whole tensor,
    ``where(W == 0, 1, sign(W/E)) * E``; fa (l.38-41): quantize(x, 1) = round(x * 1)/1 (tf.round: half to even);
    fg is the identity in the forward pass (bitG == 32).
  * alexnet-dorefa.py Model.get_logits (l.66-133): weights of every layer whose name ends with W are quantized except
    conv0 and fct (l.74-81); nonlin = clip(x, 0, 1) (l.83-86); activate = fa(nonlin(x)) (l.88-89); argscope
    data_format channels_first, BatchNorm(momentum 0.9, epsilon 1e-4), Conv2D use_bias=False (l.91-94); layer list
    l.95-130: conv0 12x12/4 VALID (+bias) -> activate -> conv1 5x5 SAME split 2 -> bn1 -> pool1 3/2 SAME -> activate
    -> conv2 3x3 SAME -> bn2 -> pool2 3/2 SAME -> activate -> conv3 3x3 split 2 -> bn3 -> activate -> conv4 3x3 split 2
    -> bn4 -> pool4 3/2 VALID -> activate -> fc0 4096 (+bias) -> bnfc0 -> activate -> fc1 4096 (no bias) -> bnfc1 ->
    nonlin -> fct 1000 (+bias).
  * tensorpack layers: Conv2D(split) = grouped conv (input channels and kernel output channels split in `split`
    groups, outputs concatenated); MaxPooling SAME = TF SAME (extra pad at the bottom/right, padding ignored);
    FullyConnected flattens its NCHW input (C, H, W order); BatchNorm at inference uses the EMA statistics
    ``bn*/mean/EMA``, ``bn*/variance/EMA``. Kernels are HWIO (conv) / [in, out] (FC) in the npz.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-4
CONVS = [  # name, kernel, stride, padding (TF), groups, quantized
    ("conv0", 12, 4, "VALID", 1, False), ("conv1", 5, 1, "SAME", 2, True), ("conv2", 3, 1, "SAME", 1, True),
    ("conv3", 3, 1, "SAME", 2, True), ("conv4", 3, 1, "SAME", 2, True)]


def fw(w: torch.Tensor) -> torch.Tensor:
    e = torch.mean(torch.abs(w))
    return torch.where(w == 0, torch.ones_like(w), torch.sign(w / e)) * e


def activate(x: torch.Tensor) -> torch.Tensor:
    return torch.round(torch.clamp(x, 0.0, 1.0) * 1.0) / 1.0


def same_pool(x: torch.Tensor) -> torch.Tensor:
    """MaxPooling(3, 2, padding='SAME') in TF semantics (padding never selected)."""
    h, w = x.shape[2:]
    oh, ow = -(-h // 2), -(-w // 2)
    ph, pw = max((oh - 1) * 2 + 3 - h, 0), max((ow - 1) * 2 + 3 - w, 0)
    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=float("-inf"))
    return F.max_pool2d(x, 3, 2)


class DoReFaAlexNet(torch.nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.p = {k[:-2] if k.endswith(":0") else k: torch.from_numpy(np.asarray(v, np.float32).copy()) for k, v in params.items()}

    def to_dtype(self, dtype):
        self.p = {k: v.to(dtype) for k, v in self.p.items()}
        return self

    def bn(self, x, name):
        p = self.p
        return F.batch_norm(x, p[f"{name}/mean/EMA"], p[f"{name}/variance/EMA"], p[f"{name}/gamma"], p[f"{name}/beta"], False, 0.0, EPS)

    def conv(self, x, name, k, s, pad, groups, quant):
        w = self.p[f"{name}/W"]
        w = fw(w) if quant else w
        w = w.permute(3, 2, 0, 1)        # HWIO -> OIHW (group-major output channels, as tf.split + concat)
        p = (k - 1) // 2 if pad == "SAME" else 0
        assert pad == "VALID" or (k % 2 == 1 and s == 1)
        return F.conv2d(x, w, self.p.get(f"{name}/b"), s, p, 1, groups)

    def forward(self, x_nchw, capture: dict | None = None):
        """capture: {consuming binary layer: value handed to the quantizer (before clip), NHWC / [N, K]}."""
        p = self.p

        def act(v, consumer):
            if capture is not None:
                capture[consumer] = (v.permute(0, 2, 3, 1) if v.dim() == 4 else v).detach().numpy()
            return activate(v)

        x = act(self.conv(x_nchw, *CONVS[0]), "conv1")
        x = act(same_pool(self.bn(self.conv(x, *CONVS[1]), "bn1")), "conv2")
        x = act(same_pool(self.bn(self.conv(x, *CONVS[2]), "bn2")), "conv3")
        x = act(self.bn(self.conv(x, *CONVS[3]), "bn3"), "conv4")
        x = F.max_pool2d(self.bn(self.conv(x, *CONVS[4]), "bn4"), 3, 2)
        if capture is not None:   # fc0 sees the NCHW flatten; record NHWC-flattened, as the IR's Reshape
            capture["fc0"] = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1).detach().numpy()
        x = torch.flatten(activate(x), 1)          # NCHW -> (C, H, W) order, as tensorpack batch_flatten
        x = act(self.bn(x @ fw(p["fc0/W"]) + p["fc0/b"], "bnfc0"), "fc1")
        x = torch.clamp(self.bn(x @ fw(p["fc1/W"]), "bnfc1"), 0.0, 1.0)
        return x @ p["fct/W"] + p["fct/b"]


def load(npz_path: str) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
