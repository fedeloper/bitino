"""Reader: original BNN-PYNQ Theano/Lasagne npz checkpoints (CNV, LFC) -> ``finn_bnn`` layer specs.

Source: Xilinx/BNN-PYNQ (archived) bnn/src/training/{cnv,lfc,quantization,quantized_net}.py; the npz files
are ``np.savez(path, *lasagne.layers.get_all_param_values(net))`` (``arr_0..arr_N``, opened with
allow_pickle=False). Param order (Lasagne ``get_all_params``):
  * CNV: 6 x [conv W (O,I,3,3), b (O), BN beta, gamma, mean, inv_std]; 2 x [dense W (I,O), b, BN x4];
    dense W (512, n_out) without bias; BN with axes=(0,1) -> 4 scalars.
  * LFC: 4 x [dense W (I,O), b, BN beta, gamma, mean, inv_std] (last BN per class).

Semantics (verified in the code; the Theano run is the trusted reference, tools/bnn_pynq_theano_reference.py):
  * Weights: ``T.switch(T.round(hard_sigmoid(W)), +1, -1)``, hard_sigmoid(W) = clip((W+1)/2, 0, 1), scale 1.
    In float32, (W+1)/2 == 0.5 exactly for |W| below ~6e-8, and Theano 1.0's ``T.round`` default is
    half-to-even, so such tiny weights map to -1 even when positive. The adapter therefore takes the binary
    weights from the reference run itself (``wq``: Theano's own quantizeWeights output in float32) and checks
    them against an explicit mirror of that float32 expression; the float64 Theano run is fed the same
    binary weights (``--binary-weights``), as for the IR-Net family. finnthesizer.py (the FPGA flow) uses
    ``w >= 0`` instead; the count of weights where the two differ is reported.
  * Conv: ``lasagne.layers.Conv2DLayer`` with pad='valid', flip_filters=False (cross-correlation), bias.
  * Dense: ``lasagne.layers.DenseLayer`` (flattens its input in C order: NCHW, here at 1x1 or C=1), bias
    except the last CNV layer.
  * BatchNormLayer (deterministic): ``(x - mean) * (gamma * inv_std) + beta`` with the stored inv_std.
  * Activation ``FixedHardTanH(QuantizationBinary)``: clip(x, -1, 1), then ``2*round3(hard_sigmoid(x)) - 1``
    with round3 = C ``round`` (half away from zero): +1 iff fl(fl(x+1)/2) >= 0.5, i.e. ge-sign up to a
    rounding band |x| < ~3e-8 (fp32; ~6e-17 in fp64) below 0; the integer-threshold margins (reported) show
    whether any BN output can come that close to 0.
  * CNV order conv -> [MaxPool2D(2, ignore_border=False) on the float conv output] -> BN -> sign. MaxPool
    comes BEFORE BN here (unlike Brevitas): ``finn_bnn`` handles it as pool="before_bn" (integer thresholds
    + input-channel polarity in the next VALID conv).
  * Input: CNV ``FixedHardTanH(QuantizationFixed(8, 7, narrow_range=False))`` on the [-1, 1] input:
    round3((clip(x, -1, 1 - 2^-7) + 1) * 128) / 128 - 1 = int8 n/128 with n = round3(128 x) in [-128, 127];
    mapped to ChannelAffine(1, 0) + FakeQuant(1/128, [-128, 127]) after checking that no input of the declared
    domain (2k/255 - 1, k = 0..255, float32) sits on a rounding tie (where half-away and half-even differ).
    LFC: the dataset is binarized outside the network (np.where(2x-1 < 0, -1, 1)); the network input is +-1.
"""
from __future__ import annotations

import numpy as np

from .finn_bnn import InputSpec, Layer


def theano_binary_weights_f32(w: np.ndarray) -> np.ndarray:
    """Explicit float32 mirror of QuantizationBinary.quantizeWeights with Theano's half-to-even T.round."""
    w = np.asarray(w, np.float32)
    hs = np.clip((w + np.float32(1.0)) / np.float32(2.0), np.float32(0), np.float32(1)).astype(np.float32)
    return np.where(np.round(hs) != 0, 1, -1).astype(np.int8)    # np.round = half to even


def _bn(arrs, i, b):
    beta, gamma, mean, inv_std = (np.asarray(arrs[i + k], np.float64) for k in range(4))
    scale = gamma * inv_std
    shift = (b - mean) * scale + beta
    return scale, shift, {"gamma": gamma, "beta": beta, "mean": mean, "inv_std": inv_std}


def read(arrs: list, wq: list, arch: str, n_out: int):
    """arrs: npz arrays in order; wq: binary weights per quantized layer from the float32 reference run.
    Returns (InputSpec, [Layer], info)."""
    info = {"weights_vs_mirror": [], "weights_vs_finnthesizer_ge0": [], "exact_zero_latent_weights": []}

    def wbin(k, w_latent, conv: bool):
        q = np.asarray(wq[k])
        mir = theano_binary_weights_f32(w_latent)
        if not np.all(np.abs(q) == 1) or not np.array_equal(q.astype(np.int8), mir):
            raise NotImplementedError(f"quantized layer {k}: Theano wq differs from the float32 mirror")
        info["weights_vs_mirror"].append(True)
        info["weights_vs_finnthesizer_ge0"].append(int(np.sum(mir != np.where(np.asarray(w_latent) >= 0, 1, -1))))
        info["exact_zero_latent_weights"].append(int(np.sum(np.asarray(w_latent) == 0)))
        return mir if conv else np.ascontiguousarray(mir.T)          # dense: (I, O) -> [O, I]

    layers = []
    if arch == "cnv":
        if len(arrs) != 53:
            raise NotImplementedError(f"CNV npz: expected 53 arrays, got {len(arrs)}")
        pools = [False, True, False, True, False, False]
        i = 0
        for c in range(6):
            w, b = arrs[i], np.asarray(arrs[i + 1], np.float64)
            if w.ndim != 4 or w.shape[2:] != (3, 3):
                raise NotImplementedError(f"conv{c}: W shape {w.shape}")
            scale, shift, bnr = _bn(arrs, i + 2, b)
            layers.append(Layer(f"conv{c}", "conv", wbin(c, w, True), scale, shift, pool="before_bn" if pools[c] else None,
                                w1a8=(c == 0), source="Conv2DLayer (valid, flip_filters=False, +bias)" + (" -> MaxPool2D 2x2" if pools[c] else "")
                                + " -> BatchNormLayer -> FixedHardTanH(QuantizationBinary)",
                                extra={"negative_scale_channels": int(np.sum(scale < 0))}))
            i += 6
        for d in range(2):
            w, b = arrs[i], np.asarray(arrs[i + 1], np.float64)
            scale, shift, _ = _bn(arrs, i + 2, b)
            layers.append(Layer(f"dense{d}", "dense", wbin(6 + d, w, False), scale, shift,
                                source="DenseLayer (+bias) -> BatchNormLayer -> FixedHardTanH(QuantizationBinary)",
                                extra={"negative_scale_channels": int(np.sum(scale < 0))}))
            i += 6
        w = arrs[i]
        if w.shape[1] != n_out:
            raise NotImplementedError(f"last dense has {w.shape[1]} outputs, expected {n_out}")
        beta, gamma, mean, inv_std = (float(np.asarray(arrs[i + 1 + k], np.float64)) for k in range(4))
        s = gamma * inv_std
        layers.append(Layer("dense2", "dense", wbin(8, w, False), np.full(n_out, s), np.full(n_out, beta - mean * s), final=True,
                            source="DenseLayer (no bias) -> BatchNormLayer(axes=(0,1)) scalar affine",
                            extra={"scalar_bn": {"beta": beta, "gamma": gamma, "mean": mean, "inv_std": inv_std}}))
        inp = InputSpec("int8", [1, 32, 32, 3], affine=(1.0, 0.0), q={"scale": 1.0 / 128, "qmin": -128, "qmax": 127},
                        source="FixedHardTanH(QuantizationFixed(8, 7, narrow_range=False)): round3((clip(x)+1)*128)/128 - 1")
        # tie check of the input quantizer on the declared domain (float32 2k/255 - 1)
        k = np.arange(256, dtype=np.float32)
        xin = (np.float32(2.0 / 255.0) * k - np.float32(1.0)).astype(np.float32)
        t = ((np.clip(xin, np.float32(-1), np.float32(1 - 2 ** -7)) + np.float32(1)) * np.float32(128)).astype(np.float32)
        half_away = np.sign(t) * np.floor(np.abs(t) + 0.5)
        info["input_quantizer_check"] = {"domain": "float32(2/255)*k - 1, k = 0..255", "n_values": 256,
                                         "half_away_equals_half_even": bool(np.array_equal(half_away, np.round(t))),
                                         "min_distance_to_tie": float(np.min(np.abs(np.abs(t - np.floor(t)) - 0.5))),
                                         "theano_levels_equal_fakequant": bool(np.array_equal(
                                             (half_away / 128 - 1).astype(np.float32),
                                             (np.clip(np.round(xin.astype(np.float64) * 128), -128, 127) / 128).astype(np.float32)))}
    elif arch == "lfc":
        if len(arrs) != 24:
            raise NotImplementedError(f"LFC npz: expected 24 arrays, got {len(arrs)}")
        for d in range(4):
            i = 6 * d
            w, b = arrs[i], np.asarray(arrs[i + 1], np.float64)
            scale, shift, _ = _bn(arrs, i + 2, b)
            final = d == 3
            if final and w.shape[1] != n_out:
                raise NotImplementedError("LFC output size")
            layers.append(Layer(f"dense{d}", "dense", wbin(d, w, False), scale, shift, final=final,
                                source="DenseLayer (+bias) -> BatchNormLayer" + ("" if final else " -> FixedHardTanH(QuantizationBinary) (Dropout = identity)"),
                                extra={"negative_scale_channels": int(np.sum(scale < 0))}))
        inp = InputSpec("pm1", [1, 28, 28, 1], source="dataset pre-binarized to +-1 (mnist.py np.where(2x-1 < 0, -1, 1)); DropoutLayer = identity")
    else:
        raise NotImplementedError(arch)
    return inp, layers, info
