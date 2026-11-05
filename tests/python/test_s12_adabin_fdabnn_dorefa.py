"""Slice S12: AdaBin ResNet-20 (adapter torch_adabin), FDA-BNN ResNet-20 (mindspore_ckpt + mindspore_fdabnn) and the
tensorpack DoReFa-Net AlexNet W1A1 (tensorpack_dorefa). No new operator: every test runs existing ops.

* MindSpore checkpoint reader: a synthetic protobuf file (incl. a scalar stored with dims [0]) round-trips; unknown
  fields are rejected.
* AdaBin decomposition (asymmetric binary sets with zero padding): BConv2D(sign bits) + one-channel all-ones BConv2D +
  1x1 Conv2D + constant border map equals the direct float64 convolution of the two-level activations/weights with zero
  padding (strides 1 and 2); the official model with random weights (if the clone is present) vs its float64 forward
  with float64 constants; a synthetic AdaBin-pattern graph with exact ties x == beta: numpy vs C++ bit-exact
  (reference sign_zero / ge, neon ge).
* FDA-BNN: adapter vs the cited PyTorch port in float64 with random checkpoint tensors, and numpy vs C++ bit-exact.
* DoReFa {0,1} activations: BConv2D gt(0.5) with pad_value -1 and post_mul/post_add E/2 equals the zero-padded {0,1}
  convolution (grouped), BDense likewise; numpy vs C++ bit-exact with ties at x == 0.5 (reference, neon).
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters import mindspore_ckpt, mindspore_fdabnn, tensorpack_dorefa  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
ADABIN_REPO = os.environ.get("BITINO_ADABIN_REPO", os.path.join(MAIN, ".cache", "repos", "Efficient-Computing"))


# ----------------------------------------------------------------------------- MindSpore checkpoint reader
def _varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _ld(field: int, payload: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _ms_value(tag: str, arr: np.ndarray, dtype: str, dims=None, packed=False) -> bytes:
    dims = list(arr.shape) if dims is None else dims
    if packed:
        t = _ld(1, b"".join(_varint(d) for d in dims))
    else:
        t = b"".join(_varint(1 << 3 | 0) + _varint(d) for d in dims)
    t += _ld(2, dtype.encode()) + _ld(3, arr.astype(arr.dtype.newbyteorder("<")).tobytes())
    return _ld(1, _ld(1, tag.encode()) + _ld(2, t))


def test_mindspore_ckpt_reader_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=(7,)).astype(np.float32)
    data = (_ms_value("conv.weight", w, "Float32") + _ms_value("bn.beta", b, "Float32", packed=True)
            + _ms_value("bn.num_batches_tracked", np.array([153660], np.int64), "Int64", dims=[0]))
    p = tmp_path / "x.ckpt"
    p.write_bytes(data)
    got = mindspore_ckpt.read(str(p))
    assert list(got) == ["conv.weight", "bn.beta", "bn.num_batches_tracked"]
    np.testing.assert_array_equal(got["conv.weight"], w)
    np.testing.assert_array_equal(got["bn.beta"], b)
    assert got["bn.num_batches_tracked"].shape == () and int(got["bn.num_batches_tracked"]) == 153660
    bad = tmp_path / "bad.ckpt"
    bad.write_bytes(_ld(2, b"x"))
    with pytest.raises(ValueError):
        mindspore_ckpt.read(str(bad))
    short = tmp_path / "short.ckpt"
    short.write_bytes(_ms_value("w", w, "Float32", dims=[5, 3, 3, 3]))
    with pytest.raises(ValueError, match="does not match"):
        mindspore_ckpt.read(str(short))


# ----------------------------------------------------------------------------- AdaBin decomposition
def _conv_f64(x_nhwc, w_ohwi, stride, pad):
    import torch
    import torch.nn.functional as F
    y = F.conv2d(torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2)), torch.from_numpy(w_ohwi.transpose(0, 3, 1, 2)), stride=stride, padding=pad)
    return y.numpy().transpose(0, 2, 3, 1)


def adabin_pattern(seed: int, binarize: str, h: int, cin: int, cout: int, stride: int, quant: bool):
    """One AdaBin conv as the adapter emits it (BN folded), built from random levels. Returns (graph, direct-f64 fn)."""
    pytest.importorskip("torch")
    from bitino.adapters.torch_adabin import border_maps
    rng = np.random.default_rng(seed)
    alpha, beta = 0.75, (0.25 if quant else float(rng.normal() * 0.3))
    a_plus, a_minus = alpha * (beta + 1), alpha * (beta - 1)
    p, q = (a_plus + a_minus) / 2, (a_plus - a_minus) / 2
    sign = np.where(rng.random((cout, cin, 3, 3)) < 0.5, -1, 1).astype(np.int8)
    wp, wm = rng.normal(size=cout) * 0.1 + 0.3, rng.normal(size=cout) * 0.1 - 0.2
    c, d = (wp + wm) / 2, (wp - wm) / 2
    gsc, gsh = rng.random(cout) + 0.5, rng.normal(size=cout)
    sw_map, k_map = border_maps(sign, (h, h), stride, 1)
    ho = sw_map.shape[0]
    g = Graph()
    x = g.add_input("input", [1, h, h, cin])
    attrs = {"strides": [stride, stride], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
             "input_binarize": binarize, "activation": "none", "output": "f32"}
    beta = float(np.float32(beta))
    thr = g.add_constant("thr", np.full(cin, beta, np.float32))
    cnt = g.add_node("BConv2D", [x, g.add_constant("ones", np.ones((1, 3, 3, cin), np.int8), "bit", "OHWI_bit"), "", "", thr], attrs, name="c/sum_s")
    bsum = g.add_node("Conv2D", [cnt, g.add_constant("bw", (gsc * q * c).reshape(cout, 1, 1, 1).astype(np.float32)), ""],
                      {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0], "groups": 1, "pad_value": 0.0, "activation": "none"})
    main = g.add_node("BConv2D", [x, g.add_constant("bits", np.ascontiguousarray(sign.transpose(0, 2, 3, 1)), "bit", "OHWI_bit"),
                                  g.add_constant("pm", (gsc * q * d).astype(np.float32)), "", thr], attrs, name="c")
    y = g.add_node("Add", [g.add_node("Add", [main, bsum]),
                           g.add_constant("map", (gsc * p * (d * sw_map + c * k_map[..., None]) + gsh).reshape(1, ho, ho, cout).astype(np.float32))])
    g.set_outputs(y)
    exact = {"bw": (gsc * q * c).reshape(cout, 1, 1, 1), "pm": gsc * q * d,
             "map": (gsc * p * (d * sw_map + c * k_map[..., None]) + gsh).reshape(1, ho, ho, cout)}

    def direct(xin: np.ndarray) -> np.ndarray:
        s = np.sign(xin.astype(np.float64) - np.float32(beta))
        if binarize == "ge":
            s = np.where(s == 0, 1.0, s)
        a = p + q * s                                     # activation levels (s = 0 -> alpha*beta = p)
        w = np.where(sign > 0, wp[:, None, None, None], wm[:, None, None, None]).transpose(0, 2, 3, 1)
        return _conv_f64(a, np.ascontiguousarray(w), stride, 1) * gsc + gsh
    return g, direct, exact


@pytest.mark.parametrize("stride", [1, 2])
def test_adabin_decomposition_exact_float64(stride):
    g, direct, exact = adabin_pattern(3, "sign_zero", 9, 20, 12, stride, quant=False)
    for name, v in exact.items():
        g.constants[name].data = v
    x = np.random.default_rng(1).normal(size=(1, 9, 9, 20))
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    want = direct(x)
    assert np.abs(out - want).max() < 1e-12 * max(1.0, np.abs(want).max())


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("reference", "ge"), ("neon", "ge")])
def test_adabin_pattern_numpy_bit_exact_with_cpp(tmp_path, stage, binarize):
    g, _, _ = adabin_pattern(5, binarize, 8, 16, 32, 2, quant=True)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    rng = np.random.default_rng(2)
    xs = [(np.round(rng.normal(size=(1, 8, 8, 16)) * 4 * s) / 8).astype(np.float32) for s in (0.5, 1.0, 2.0)]
    np.concatenate(xs).tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(len(xs), -1)
    ties = 0
    for i, xin in enumerate(xs):
        ties += int(np.sum(xin == np.float32(0.25)))
        np.testing.assert_array_equal(got[i], executor.run(g2, {"input": xin})[g2.outputs[0]].ravel())
    assert ties > 0, "inputs should hit the threshold exactly (sign_zero vs ge)"


def _adabin_source():
    torch = pytest.importorskip("torch")
    if not os.path.exists(os.path.join(ADABIN_REPO, "BinaryNetworks", "AdaBin", "cifar10", "utils", "binarylib.py")):
        pytest.skip("AdaBin clone not available")
    from convert_s12_cifar import load_source
    model, _, _, _ = load_source("adabin", ADABIN_REPO, None)
    torch.manual_seed(0)
    with torch.no_grad():
        for n, prm in model.named_parameters():
            if n.endswith("alpha_a"):
                prm.copy_(torch.rand(()) + 0.5)
            elif n.endswith("beta_a"):
                prm.copy_(torch.randn(()) * 0.3)
            elif n.endswith(("pos_scale", "neg_scale")):
                prm.copy_(torch.randn_like(prm))
            elif n.endswith("weight") and prm.dim() == 4:
                prm.copy_(torch.randn_like(prm) * 0.1 + 0.02)
        for m in model.modules():
            if type(m).__name__.startswith("BatchNorm"):
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
                m.weight.copy_(torch.randn_like(m.weight))
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)
    return model.eval(), torch


def test_adabin_source_random_weights_float64_exact():
    model, torch = _adabin_source()
    from bitino.adapters import torch_adabin
    g, info = torch_adabin.convert(model, binarize="sign_zero")
    assert g.op_counts()["BConv2D"] == 36
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v.reshape(g.constants[name].data.shape)
    # float64 source with the fp32 weight levels (what the adapter carries)
    x = np.random.default_rng(0).normal(size=(2, 32, 32, 3))
    levels = info["weight_levels"]
    names = {m: n for n, m in model.named_modules()}
    gl = type(next(m for m in model.modules() if type(m).__name__ == "AdaBin_Conv2d")).forward.__globals__
    real = gl["F"]

    class FP:
        def __getattr__(self, k):
            return getattr(real, k)

        @staticmethod
        def conv2d(a, w, *args, **kw):
            n = FP.current
            s = torch.from_numpy(info["binary_weights"][n].astype(np.float64))
            wl = torch.where(s > 0, torch.from_numpy(levels[n][0].astype(np.float64)).view(-1, 1, 1, 1),
                             torch.from_numpy(levels[n][1].astype(np.float64)).view(-1, 1, 1, 1))
            return real.conv2d(a, wl, *args, **kw)
    hooks = [m.register_forward_pre_hook(lambda m, i: setattr(FP, "current", names[m])) for m in model.modules()
             if type(m).__name__ == "AdaBin_Conv2d"]
    gl["F"] = FP()
    try:
        with torch.no_grad():
            ref = model.double()(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy()
    finally:
        gl["F"] = real
        for h in hooks:
            h.remove()
        model.float()
    for i in range(len(x)):
        out = executor.run(g, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]]
        assert np.abs(out - ref[i]).max() < 1e-9 * max(1.0, np.abs(ref).max())


def test_adabin_rejects_nonpositive_alpha():
    model, torch = _adabin_source()
    from bitino.adapters import torch_adabin
    with torch.no_grad():
        model.layer2[1].conv1.binary_a.alpha_a.fill_(-0.5)
    with pytest.raises(NotImplementedError, match="alpha_a"):
        torch_adabin.convert(model)


# ----------------------------------------------------------------------------- FDA-BNN
def _fdabnn_params(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    p = {"conv1.weight": rng.normal(size=(16, 3, 3, 3)).astype(np.float32)}

    def bn(name, c):
        p[f"{name}.moving_mean"] = (rng.normal(size=c) * 0.1).astype(np.float32)
        p[f"{name}.moving_variance"] = (rng.random(c) + 0.5).astype(np.float32)
        p[f"{name}.gamma"] = rng.normal(size=c).astype(np.float32)
        p[f"{name}.beta"] = (rng.normal(size=c) * 0.1).astype(np.float32)
    bn("bn1", 16)
    cin = 16
    for li, planes in enumerate((16, 32, 64), start=1):
        for bi in range(3):
            pth = f"layer{li}.{bi}"
            p[f"{pth}.conv1.weight"] = (rng.normal(size=(planes, cin, 3, 3)) * 0.05).astype(np.float32)
            p[f"{pth}.conv2.weight"] = (rng.normal(size=(planes, planes, 3, 3)) * 0.05).astype(np.float32)
            bn(f"{pth}.bn1", planes)
            bn(f"{pth}.bn2", planes)
            p[f"{pth}.relu1.w"] = np.array([rng.random() * 0.5], np.float32)
            p[f"{pth}.relu2.w"] = np.array([rng.random() * 0.5], np.float32)
            cin = planes
    p["linear.weight"] = rng.normal(size=(10, 64)).astype(np.float32)
    p["linear.bias"] = rng.normal(size=10).astype(np.float32)
    return p


def test_fdabnn_adapter_vs_port_float64():
    torch = pytest.importorskip("torch")
    import fdabnn_reference
    params = _fdabnn_params(0)
    g, info = mindspore_fdabnn.convert(params)
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    model = fdabnn_reference.build(params).double()
    x = np.random.default_rng(1).normal(size=(2, 32, 32, 3))
    with torch.no_grad():
        ref = model(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy()
    for i in range(2):
        out = executor.run(g, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]]
        assert np.abs(out - ref[i]).max() < 1e-9 * max(1.0, np.abs(ref).max())
    bad = dict(params)
    bad["layer1.0.conv1.weight"] = bad["layer1.0.conv1.weight"].copy()
    bad["layer1.0.conv1.weight"][0, 0, 0, 0] = 0.0
    with pytest.raises(NotImplementedError, match="exactly 0"):
        mindspore_fdabnn.convert(bad)
    with pytest.raises(NotImplementedError, match="not used"):
        mindspore_fdabnn.convert({**params, "extra.tensor": np.zeros(1, np.float32)})


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("neon", "ge")])
def test_fdabnn_numpy_bit_exact_with_cpp(tmp_path, stage, binarize):
    g, _ = mindspore_fdabnn.convert(_fdabnn_params(4), binarize=binarize)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    xs = np.random.default_rng(3).normal(size=(2, 32, 32, 3)).astype(np.float32)
    xs.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(2, -1)
    for i in range(2):
        np.testing.assert_array_equal(got[i], executor.run(g2, {"input": xs[i:i + 1]})[g2.outputs[0]].ravel())


# ----------------------------------------------------------------------------- DoReFa {0,1} activations
def dorefa_pattern(seed: int):
    """BConv2D (grouped, SAME) and BDense in the tensorpack_dorefa form. Returns (graph, direct-f64 fn)."""
    rng = np.random.default_rng(seed)
    cin, cout, groups, h, k_fc, o_fc = 32, 16, 2, 6, 6 * 6 * 16, 24
    wc = np.where(rng.random((cout, 3, 3, cin // groups)) < 0.5, -1, 1).astype(np.int8)
    wd = np.where(rng.random((o_fc, k_fc)) < 0.5, -1, 1).astype(np.int8)
    ec, ed, bias = 2.0 ** -4, 2.0 ** -5, np.round(rng.normal(size=o_fc) * 8) / 8   # exactly representable constants
    g = Graph()
    x = g.add_input("input", [1, h, h, cin])
    y = g.add_node("BConv2D", [x, g.add_constant("wc", wc, "bit", "OHWI_bit"), g.add_constant("pmc", np.full(cout, ec / 2, np.float32)),
                               g.add_constant("pac", (ec / 2 * wc.reshape(cout, -1).sum(1)).astype(np.float32)),
                               g.add_constant("thc", np.full(cin, 0.5, np.float32))],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": groups, "pad_value": -1.0,
                    "input_binarize": "gt", "activation": "none", "output": "f32"})
    r = g.add_node("Reshape", [y], {"shape": [1, k_fc]})
    z = g.add_node("BDense", [r, g.add_constant("wd", wd, "bit"), g.add_constant("pmd", np.full(o_fc, ed / 2, np.float32)),
                              g.add_constant("pad", (ed / 2 * wd.sum(1) + bias).astype(np.float32)), g.add_constant("thd", np.full(k_fc, 0.5, np.float32))],
                   {"input_binarize": "gt", "activation": "none"})
    g.set_outputs(z, y)

    def direct(xin):
        import torch
        import torch.nn.functional as F
        u = (np.asarray(xin, np.float64) > 0.5).astype(np.float64)           # round(clip(x, 0, 1)), half to even
        yc = F.conv2d(torch.from_numpy(u.transpose(0, 3, 1, 2)), torch.from_numpy(wc.astype(np.float64).transpose(0, 3, 1, 2) * ec),
                      padding=1, groups=groups).numpy().transpose(0, 2, 3, 1)
        ud = (yc.reshape(1, -1) > 0.5).astype(np.float64)
        return ud @ (wd.astype(np.float64).T * ed) + bias, yc
    return g, direct


def test_dorefa_zero_one_mapping_exact_float64():
    pytest.importorskip("torch")
    g, direct = dorefa_pattern(0)
    x = np.random.default_rng(1).random((1, 6, 6, 32)) * 1.5 - 0.25
    env = executor.run(g, {"input": x}, float_dtype=np.float64)
    want_z, want_y = direct(x)
    np.testing.assert_allclose(env[g.outputs[1]], want_y, rtol=0, atol=1e-9)
    np.testing.assert_allclose(env[g.outputs[0]], want_z, rtol=0, atol=1e-9)


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage", ["reference", "neon"])
def test_dorefa_pattern_numpy_bit_exact_with_cpp(tmp_path, stage):
    g, _ = dorefa_pattern(2)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    rng = np.random.default_rng(5)
    xs = [(np.round(rng.random((1, 6, 6, 32)) * 8) / 8).astype(np.float32) for _ in range(3)]   # many x == 0.5
    np.concatenate(xs).tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(len(xs), -1)
    for i, xin in enumerate(xs):
        res = executor.run(g2, {"input": xin})
        np.testing.assert_array_equal(got[i], np.concatenate([res[g2.outputs[0]].ravel(), res[g2.outputs[1]].ravel()]))
    assert sum(int(np.sum(xin == np.float32(0.5))) for xin in xs) > 0


def test_tensorpack_dorefa_adapter_guards():
    rng = np.random.default_rng(0)
    with pytest.raises(NotImplementedError, match="npz has no conv0/W"):
        tensorpack_dorefa.convert({"fc0/W:0": rng.normal(size=(2, 2)).astype(np.float32)})


def test_tensorpack_bgr_normalize_matches_tf_rewrite():
    from eval_sets.tensorpack_bgr import MEAN, STD, normalize
    u8 = np.arange(256, dtype=np.uint8).reshape(1, 1, 256, 1).repeat(3, axis=3)
    got = normalize(u8)
    mean = np.asarray(MEAN, np.float32) * np.float32(255.0)
    recip = np.float32(1.0) / (np.asarray(STD, np.float32) * np.float32(255.0))
    np.testing.assert_array_equal(got, (u8.astype(np.float32) - mean) * recip)
