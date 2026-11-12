"""Adapter: "Faster Than Lies" deepfake detector (Lanzino, Fontana, Diko, Marini, Cinque, CVPRW 2024,
arXiv:2406.04932; https://github.com/fedeloper/binary_deepfake_detection, model.py BNext4DFR) -> Bitino IR.

The adapter reads the *instantiated* BNext4DFR module (authors' class, weights loaded with
torch.load(weights_only=True)) and mirrors BNext4DFR.forward (model.py at commit 668db9a):

  forward(x):  x = RGB in [0, 1], NCHW (the test loaders do not normalise)
    x = add_new_channels(x)        concat on channels: RGB, then [magnitude], [fft], [lbp] (enabled ones, this order)
    x = adapter(x)                 nn.Conv2d(3 + k, 3, kernel 3, stride 1, padding 1, bias)
    x = (x - mean) / std           timm IMAGENET_DEFAULT_MEAN / _STD as float32 tensors
    f = base_model(x)              BNext with fc = nn.Identity (pooled features)
    logit = fc(f)                  nn.Linear(C, 1) for the binary task (label is_real: logit > 0 -> real)

  _add_new_channels_worker(image HWC):
    gray = cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
         -> ImageGrayU8(coeffs [3735, 19235, 9798], shift 15, scale 255): OpenCV 4.x's 15-bit fixed-point
            BGR2GRAY applied to RGB data (channel 0 = R gets the blue weight), verified on all 2^24 pixels
            (OpenCV 4.1.2 and 4.12.0); (x * 255) is float32 and astype(uint8) truncates
    fft  = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9)        -> FftLogMagnitude
    lbp  = skimage.feature.local_binary_pattern(gray, 3, 6, method='uniform')   -> LocalBinaryPattern
           (P = 3, R = 6: offsets rp = round(-6 sin(2 pi i / 3), 5), cp = round(6 cos(2 pi i / 3), 5))
    channels / 255 in float64, then .float()                                     (div = 255, f32 outputs)
    The magnitude (Sobel) channel is not supported (no released checkpoint uses it).

  adapter + normalisation are folded into one Conv2D: W' = W / std[o], b' = (b - mean[o]) / std[o] in float64
  (mean / std = the float32 values torch uses), rounded once; zero padding is applied to the adapter's input,
  so folding is exact in real arithmetic. The float64 constants are returned in info["float64_constants"].
  The BNext backbone is converted by adapters/torch_bnext.py (same semantics as the ImageNet models); the
  single-logit head becomes Dense with a prepended zero row: outputs [0, logit], argmax = (logit > 0) = real.
"""
from __future__ import annotations

import numpy as np

from .. import image_ops
from ..ir import Graph
from . import torch_bnext

GRAY = {"coeffs": [3735, 19235, 9798], "shift": 15, "scale": 255.0}
FFT = {"eps": 1e-9, "mul": 20.0, "div": 255.0, "shift": 1}
LBP_POINTS, LBP_RADIUS = 3, 6
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)   # timm.data.constants
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)


def convert(model, model_id: str = "", input_hw: tuple[int, int] = (224, 224), binarize: str = "sign_zero") -> tuple[Graph, dict]:
    import torch

    _expect(type(model).__name__ == "BNext4DFR", f"unsupported model class {type(model).__name__}")
    _expect(not model.training, "model must be in eval mode")
    _expect(not model.add_magnitude_channel, "the Sobel magnitude channel is not supported")
    _expect(model.add_fft_channel and model.add_lbp_channel, "only the FFT + LBP configuration is supported")
    ad = model.adapter
    _expect(type(ad).__name__ == "Conv2d" and ad.in_channels == 5 and ad.out_channels == 3 and tuple(ad.kernel_size) == (3, 3)
            and tuple(ad.stride) == (1, 1) and tuple(ad.padding) == (1, 1) and tuple(ad.dilation) == (1, 1) and ad.groups == 1
            and ad.bias is not None and ad.padding_mode == "zeros", "adapter: expected Conv2d(5, 3, 3, padding=1) with bias")
    fc = model.fc
    _expect(type(fc).__name__ == "Linear" and fc.out_features == 1 and fc.bias is not None, "fc: expected Linear(C, 1)")
    _expect(type(model.base_model.fc).__name__ == "Identity", "base_model.fc must be nn.Identity")

    g = Graph()
    h, w = input_hw
    x = g.add_input("input", [1, h, w, 3])
    gray = g.add_node("ImageGrayU8", [x], dict(GRAY), name="features.gray")
    fft = g.add_node("FftLogMagnitude", [gray], dict(FFT), name="features.fft")
    rp, cp = image_ops.lbp_offsets(LBP_POINTS, LBP_RADIUS)
    lbp = g.add_node("LocalBinaryPattern", [gray], {"rp": rp, "cp": cp, "method": "uniform", "div": 255.0}, name="features.lbp")
    cat = g.add_node("Concat", [x, fft, lbp], {"axis": -1}, name="features.concat")

    mean = np.asarray(torch.as_tensor(IMAGENET_DEFAULT_MEAN), np.float64)   # float32 values, as in forward()
    std = np.asarray(torch.as_tensor(IMAGENET_DEFAULT_STD), np.float64)
    wt = ad.weight.detach().cpu().double().numpy()                         # OIHW
    bt = ad.bias.detach().cpu().double().numpy()
    w64 = np.transpose(wt / std[:, None, None, None], (0, 2, 3, 1)).copy()  # OHWI
    b64 = (bt - mean) / std
    exact = {"adapter/kernel": w64, "adapter/bias": b64}
    xa = g.add_node("Conv2D", [cat, g.add_constant("adapter/kernel", w64.astype(np.float32), layout="OHWI"),
                               g.add_constant("adapter/bias", b64.astype(np.float32))],
                    {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0, "activation": "none"},
                    name="adapter")
    g, info = torch_bnext.convert(model.base_model, model_id, binarize=binarize, graph=g, input_tensor=xa, head=fc, zero_logit=True)
    info["float64_constants"] = {**exact, **info["float64_constants"]}
    info["layer_precision"] = [
        {"layer": "features", "op": "ImageGrayU8 + FftLogMagnitude + LocalBinaryPattern + Concat",
         "source": "_add_new_channels_worker (cv2 BGR2GRAY on RGB, 20*log(|fftshift(fft2)|+1e-9)/255, LBP(3, 6, uniform)/255)",
         "weights": "none", "activations": "float64 internally, float32 outputs"},
        {"layer": "adapter", "op": "Conv2D", "source": "adapter Conv2d(5, 3, 3, padding 1) + ImageNet mean/std normalisation (folded)",
         "weights": "fp32", "activations": "fp32"},
        *info["layer_precision"]]
    info["preprocessing_ops"] = {"gray": GRAY, "fft": FFT, "lbp": {"points": LBP_POINTS, "radius": LBP_RADIUS, "rp": rp, "cp": cp,
                                                                     "method": "uniform", "div": 255.0}}
    return g, info
