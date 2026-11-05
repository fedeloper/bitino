"""Adapter: tensorpack DoReFa-Net AlexNet W1A1 (``AlexNet-1,1,32.npz``, examples/DoReFa-Net/alexnet-dorefa.py
--dorefa 1,1,32) -> Bitino IR, from the npz tensors (no TensorFlow).

Semantics (dorefa.py, alexnet-dorefa.py; cited line by line in tools/dorefa_alexnet_reference.py):
  * binary weights (every layer except conv0 / fct): ``where(W == 0, 1, sign(W/E)) * E``, E = mean|W| over the
    whole tensor (one scalar per layer, also for the grouped convs); 0 -> +1.
  * activations: ``round(clip(x, 0, 1))`` (tf.round half to even) = u in {0, 1}, and u = 1 exactly when x > 0.5.
  * SAME zero padding of the quantized input (u = 0 is a legal level).

Mapping (existing ops; packed BConv2D/BDense kernels incl. NEON):
  s = +1 if x > 0.5 else -1 (BConv2D/BDense ``gt``, threshold 0.5), u = (1 + s)/2, so for one output channel o with
  binary weights w (+-1) over the K taps of its group
      sum_taps u*w = (sum_taps w + sum_taps s*w) / 2,
  where an out-of-bounds tap must contribute u = 0, i.e. s = -1: BConv2D ``pad_value = -1``. Hence
      conv = E*sum u*w = BConv2D(gt, threshold 0.5, pad_value -1) * (E/2) + (E/2)*sum_taps w   (post_mul / post_add),
  plus the FC bias in post_add for fc0. Grouped convs (split=2) -> BConv2D groups 2. BatchNorm (eval, EMA stats,
  eps 1e-4) -> ChannelAffine (float64 fold), MaxPooling SAME -> MaxPool2D with TF's asymmetric pads (padding ignored),
  conv0 -> Conv2D (fp, bias, VALID), fc1 -> BN -> clip(0, 1) -> Activation hardtanh(0, 1) -> fct Dense (fp).
  The NCHW flatten before fc0 is absorbed by permuting fc0's rows to the NHWC order.
Only one variant: the source binarization is exactly ``x > 0.5`` (no sign(0) ambiguity), so ``gt`` is exact.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

EPS = 1e-4


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)


def _tf_same_pads(n: int, k: int, s: int) -> tuple[int, int]:
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def convert(params: dict, model_id: str = "", input_name: str = "input", input_hw: tuple[int, int] = (224, 224)) -> tuple[Graph, dict]:
    p = {k[:-2] if k.endswith(":0") else k: v for k, v in params.items()}
    used: set[str] = set()
    g = Graph()
    precision: list[dict] = []
    diag: list[dict] = []
    exact: dict[str, np.ndarray] = {}
    binary_weights: dict[str, np.ndarray] = {}

    def t(name: str) -> np.ndarray:
        _expect(name in p, f"npz has no {name}")
        used.add(name)
        v = p[name]
        _expect(v.dtype == np.float32, f"{name}: dtype {v.dtype}")
        return v

    def folded(name: str, v64) -> str:
        v = np.asarray(v64, np.float64)
        exact[name] = v
        return g.add_constant(name, v.astype(np.float32))

    def bn(x: str, name: str, c: int) -> str:
        gamma, beta = t(f"{name}/gamma").astype(np.float64), t(f"{name}/beta").astype(np.float64)
        mean, var = t(f"{name}/mean/EMA").astype(np.float64), t(f"{name}/variance/EMA").astype(np.float64)
        _expect(gamma.shape == (c,), f"{name}: {gamma.shape} vs {c}")
        scale = gamma / np.sqrt(var + EPS)
        return g.add_node("ChannelAffine", [x, folded(f"{name}/scale", scale), folded(f"{name}/shift", beta - mean * scale)], name=name)

    def binary_levels(name: str, w: np.ndarray):
        e = float(np.mean(np.abs(w.astype(np.float64))))     # E = mean|W| (float64, rounded once into post_mul/post_add)
        sign = np.where(w >= 0, 1, -1).astype(np.int8)        # where(W == 0, 1, sign(W/E)), E > 0
        _expect(e > 0, f"{name}: E = {e}")
        binary_weights[name] = sign
        diag.append({"layer": name, "E": e, "n_weights": int(sign.size), "n_plus": int(np.sum(sign > 0)), "exact_zero_latent": int(np.sum(w == 0))})
        return sign, e

    def pool(x: str, name: str, hw: tuple[int, int], same: bool):
        (pt, pb), (pl, pr) = (_tf_same_pads(hw[0], 3, 2), _tf_same_pads(hw[1], 3, 2)) if same else ((0, 0), (0, 0))
        y = g.add_node("MaxPool2D", [x], {"kernel": [3, 3], "strides": [2, 2], "pads": [pt, pb, pl, pr]}, name=name)
        return y, ((hw[0] + pt + pb - 3) // 2 + 1, (hw[1] + pl + pr - 3) // 2 + 1)

    def bconv(x: str, name: str, cin: int, cout: int, k: int, groups: int, hw):
        w = t(f"{name}/W")                                     # HWIO, I = cin / groups
        _expect(w.shape == (k, k, cin // groups, cout), f"{name}: {w.shape}")
        sign, e = binary_levels(name, w)
        ohwi = np.ascontiguousarray(sign.transpose(3, 0, 1, 2))
        wsum = ohwi.reshape(cout, -1).sum(1).astype(np.float64)
        pad = (k - 1) // 2
        precision.append({"layer": name, "op": "BConv2D", "source": "Conv2D (fw binary, split=%d) on fa({0,1}) activations" % groups,
                          "weights": "binary sign(W) (0 -> +1), one scale E = mean|W|", "activations": "{0,1} = (1 + s)/2, s = gt(x, 0.5)",
                          "input_binarize": "gt", "threshold": 0.5, "pad_value": -1.0, "post": "E/2 * acc + E/2 * sum_taps w",
                          "kernel": [k, k], "groups": groups})
        return g.add_node("BConv2D", [x, g.add_constant(f"{name}/kernel_bits", ohwi, dtype="bit", layout="OHWI_bit"),
                                      folded(f"{name}/post_mul", np.full(cout, e / 2)), folded(f"{name}/post_add", e / 2 * wsum),
                                      g.add_constant(f"{name}/threshold", np.full(cin, 0.5, np.float32))],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [pad, pad, pad, pad], "groups": groups, "pad_value": -1.0,
                           "input_binarize": "gt", "activation": "none", "output": "f32"}, name=name)

    def bdense(x: str, name: str, w_rows: np.ndarray, bias):
        k, o = w_rows.shape
        sign, e = binary_levels(name, w_rows)
        w_ok = np.ascontiguousarray(sign.T)                    # [O, K]
        wsum = w_ok.sum(1).astype(np.float64)
        add = e / 2 * wsum + (bias.astype(np.float64) if bias is not None else 0.0)
        precision.append({"layer": name, "op": "BDense", "source": "FullyConnected (fw binary) on fa({0,1}) activations" + (" + bias" if bias is not None else ""),
                          "weights": "binary sign(W) (0 -> +1), one scale E = mean|W|", "activations": "{0,1} = (1 + s)/2, s = gt(x, 0.5)",
                          "input_binarize": "gt", "threshold": 0.5, "post": "E/2 * acc + E/2 * sum w (+ bias)"})
        return g.add_node("BDense", [x, g.add_constant(f"{name}/kernel_bits", w_ok, dtype="bit"), folded(f"{name}/post_mul", np.full(o, e / 2)),
                                     folded(f"{name}/post_add", add), g.add_constant(f"{name}/threshold", np.full(k, 0.5, np.float32))],
                          {"input_binarize": "gt", "activation": "none"}, name=name)

    h, w_ = input_hw
    x = g.add_input(input_name, [1, h, w_, 3])
    w0, b0 = t("conv0/W"), t("conv0/b")
    _expect(w0.shape == (12, 12, 3, 96), f"conv0: {w0.shape}")
    x = g.add_node("Conv2D", [x, g.add_constant("conv0/kernel", np.ascontiguousarray(w0.transpose(3, 0, 1, 2)), layout="OHWI"),
                              g.add_constant("conv0/bias", b0)],
                   {"strides": [4, 4], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0, "activation": "none"}, name="conv0")
    precision.append({"layer": "conv0", "op": "Conv2D", "source": "Conv2D 12x12/4 VALID + bias (fp, not quantized)", "weights": "fp32", "activations": "fp32"})
    hw = ((h - 12) // 4 + 1, (w_ - 12) // 4 + 1)
    x = bn(bconv(x, "conv1", 96, 256, 5, 2, hw), "bn1", 256)
    x, hw = pool(x, "pool1", hw, True)
    x = bn(bconv(x, "conv2", 256, 384, 3, 1, hw), "bn2", 384)
    x, hw = pool(x, "pool2", hw, True)
    x = bn(bconv(x, "conv3", 384, 384, 3, 2, hw), "bn3", 384)
    x = bn(bconv(x, "conv4", 384, 256, 3, 2, hw), "bn4", 256)
    x, hw = pool(x, "pool4", hw, False)
    _expect(hw == (6, 6), f"pool4 output {hw}")
    precision.append({"layer": "pool1/pool2/pool4", "op": "MaxPool2D", "source": "MaxPooling 3/2 SAME (TF pads, bottom/right) / VALID on the BN output",
                      "weights": "none", "activations": "fp32 (before the {0,1} quantizer, as the source)"})
    wf0 = t("fc0/W")
    _expect(wf0.shape == (256 * 36, 4096), f"fc0: {wf0.shape}")
    wf0 = np.ascontiguousarray(wf0.reshape(256, 6, 6, 4096).transpose(1, 2, 0, 3).reshape(9216, 4096))   # rows (c,h,w) -> (h,w,c)
    x = g.add_node("Reshape", [x], {"shape": [1, 9216]}, name="flatten")
    x = bn(bdense(x, "fc0", wf0, t("fc0/b")), "bnfc0", 4096)
    x = bn(bdense(x, "fc1", t("fc1/W"), None), "bnfc1", 4096)
    x = g.add_node("Activation", [x], {"type": "hardtanh", "min": 0.0, "max": 1.0}, name="nonlin")
    wt, bt = t("fct/W"), t("fct/b")
    x = g.add_node("Dense", [x, g.add_constant("fct/kernel", np.ascontiguousarray(wt.T)), g.add_constant("fct/bias", bt)], {"activation": "none"}, name="fct")
    precision.append({"layer": "head", "op": "Activation(hardtanh 0..1)+Dense", "source": "bnfc1 -> clip(0, 1) -> fct (fp + bias)",
                      "weights": "fp32", "activations": "fp32"})
    g.set_outputs(x)
    unused = sorted(set(p) - used)
    _expect(not unused, f"npz tensors not used: {unused}")
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "weight_diagnostics": diag, "binarize": "gt",
            "unused_in_forward": [], "float64_constants": exact, "binary_weights": binary_weights}
    return g, info
