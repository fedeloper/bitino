"""BHViT operators and adapter.

* A small synthetic graph with every op on the BHViT path (grouped dilated BConv2D with thresholds,
  BDense, Gather-based roll / window reverse / nearest upsampling, Concat on axis 0 and 1, AvgPool2D
  count_include_pad with padding, MaxPool2D with padding, LayerNorm, BMatMul sign_zero, Softmax,
  FakeQuant, Activation sign_zero / prelu / gelu, MatMul, ReduceMean, GlobalAvgPool): numpy executor
  vs C++ runtime, bit-exact at stages reference and neon.
* If the official IMRL/BHViT clone and its dependencies are importable (BITINO_BHVIT_REPO, default
  .cache/repos/IMRL_BHViT; e.g. run with the 'bhvit' conda env), the official BHViT-Tiny with random
  weights and random BN statistics is converted and compared with the float64 source using exact
  float64 constants (must agree to ~1e-12).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
H = W = 14
C = 16
WS = 7


def tiny_bhvit(seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, H, W, C])

    def cst(name, arr, dtype="f32", layout=""):
        return g.add_constant(name, np.asarray(arr), dtype, layout)

    def pm1(shape):
        return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)

    def rs(t, shape):
        return g.add_node("Reshape", [t], {"shape": list(shape)})

    # token mixer branch: grouped dilated binary conv with per-channel threshold, zero padding
    br = []
    for i, d in enumerate((1, 3)):
        y = g.add_node("BConv2D", [x, cst(f"cw{i}", pm1((C, 3, 3, C // 4)), "bit", "OHWI_bit"), cst(f"cs{i}", rng.random(C) * 0.1),
                                   cst(f"cb{i}", rng.normal(size=C) * 0.1), cst(f"ct{i}", rng.normal(size=C) * 0.1)],
                       {"strides": [1, 1], "dilations": [d, d], "pads": [d, d, d, d], "groups": 4, "pad_value": 0, "input_binarize": "ge"})
        br.append(g.add_node("Activation", [y, cst(f"sl{i}", rng.random(C) * 0.5)], {"type": "prelu"}))
    h = g.add_node("Add", [x, g.add_node("Add", br)])
    # roll along H by +1 and channel-group roll via Gather
    idx = np.roll(np.arange(H * W).reshape(H, W), 1, axis=0).reshape(-1).astype(np.int32)
    h = g.add_node("Add", [h, rs(g.add_node("Gather", [rs(h, [H * W, C]), cst("roll", idx, "i32")]), [1, H, W, C])])
    # window tokens + pooled merge tokens
    nw = (H // WS) * (W // WS)
    win = rs(g.add_node("Transpose", [rs(h, [1, H // WS, WS, W // WS, WS, C])], {"perm": [0, 1, 3, 2, 4, 5]}), [nw, WS * WS, C])
    pa = {"kernel": [WS, WS], "strides": [WS, WS], "pads": [3, 3, 3, 3]}
    av = g.add_node("AvgPool2D", [h], dict(pa, count_include_pad=1))
    mx = g.add_node("MaxPool2D", [h], pa)
    mg = rs(g.add_node("Add", [g.add_node("ChannelAffine", [mx, cst("a1", rng.random(C))]),
                               g.add_node("ChannelAffine", [av, cst("a2", rng.random(C))])]), [1, nw, C])
    tok = g.add_node("Concat", [win, g.add_node("Concat", [mg] * nw, {"axis": 0})], {"axis": 1})
    L = WS * WS + nw
    tok = g.add_node("LayerNorm", [tok, cst("lg", rng.random(C) + 0.5), cst("lb", rng.normal(size=C))], {"eps": 1e-5})
    nh, dh = 2, C // 2

    def proj(name):
        y = g.add_node("BDense", [tok, cst(name + "w", pm1((C, C)), "bit"), cst(name + "s", rng.random(C)), "",
                                  cst(name + "t", rng.normal(size=C) * 0.1)], {"input_binarize": "ge"})
        return g.add_node("ChannelAffine", [g.add_node("Add", [y, tok]), "", cst(name + "m", rng.normal(size=C) * 0.3)])

    def heads(t):
        return g.add_node("Transpose", [rs(t, [nw, L, nh, dh])], {"perm": [0, 2, 1, 3]})

    q, k, v = proj("q"), proj("k"), proj("v")
    s = g.add_node("BMatMul", [heads(q), heads(k)], {"transpose_b": 1, "a_binarize": "sign_zero", "b_binarize": "sign_zero", "scale": 0.25})
    p = g.add_node("Softmax", [s])
    pq = g.add_node("FakeQuant", [p, cst("qa", rng.random(nh) * 0.05 + 0.02), cst("qz", rng.normal(size=nh))],
                    {"axis": 1, "pre_scale": 3.0, "qmin": 0, "qmax": 3})
    vs = g.add_node("Activation", [v], {"type": "sign_zero"})
    ctx = rs(g.add_node("Transpose", [g.add_node("MatMul", [pq, heads(vs)])], {"perm": [0, 2, 1, 3]}), [nw * L, C])
    # window reverse + merge-token mean + nearest upsampling (Gather with constant rows)
    rows = np.array([((hh // WS) * (W // WS) + ww // WS) * L + (hh % WS) * WS + ww % WS for hh in range(H) for ww in range(W)], np.int32)
    x1 = rs(g.add_node("Gather", [ctx, cst("rev", rows, "i32")]), [1, H, W, C])
    mrows = np.array([w_ * L + WS * WS + j for w_ in range(nw) for j in range(nw)], np.int32)
    x2 = g.add_node("ReduceMean", [rs(g.add_node("Gather", [ctx, cst("mrows", mrows, "i32")]), [1, nw, nw, C])], {"axis": 1})
    up = np.array([(hh // WS) * (W // WS) + ww // WS for hh in range(H) for ww in range(W)], np.int32)
    x2 = rs(g.add_node("Gather", [rs(x2, [nw, C]), cst("up", up, "i32")]), [1, H, W, C])
    h = g.add_node("Add", [h, g.add_node("Add", [x1, x2])])
    # MLP shortcuts: channel repeat (Concat) and channel AvgPool1d (Reshape + ReduceMean)
    f1 = g.add_node("BDense", [h, cst("f1w", pm1((4 * C, C)), "bit"), cst("f1s", rng.random(4 * C)), "", cst("f1t", rng.normal(size=C) * 0.1)],
                    {"input_binarize": "ge"})
    f1 = g.add_node("Activation", [g.add_node("Add", [f1, g.add_node("Concat", [h] * 4, {"axis": -1})])], {"type": "gelu"})
    f2 = g.add_node("BDense", [f1, cst("f2w", pm1((C, 4 * C)), "bit"), cst("f2s", rng.random(C)), "", cst("f2t", rng.normal(size=4 * C) * 0.1)],
                    {"input_binarize": "ge"})
    h = g.add_node("Add", [f2, g.add_node("ReduceMean", [rs(f1, [1, H, W, C, 4])], {"axis": -1})])
    logits = g.add_node("Dense", [g.add_node("GlobalAvgPool", [h]), cst("cw", rng.normal(size=(10, C))), cst("cb", rng.normal(size=10))])
    g.set_outputs(logits, h)
    return g


def test_executor_float64_mode_and_shapes():
    g = tiny_bhvit(0)
    xin = np.random.default_rng(1).normal(size=(1, H, W, C)).astype(np.float32)
    r32 = executor.run(g, {"input": xin})
    r64 = executor.run(g, {"input": xin}, float_dtype=np.float64)
    assert r32[g.outputs[0]].shape == (1, 10) and r32[g.outputs[1]].shape == (1, H, W, C)
    assert r64[g.outputs[0]].dtype == np.float64


def test_fake_quant_numpy_half_even_and_clamp():
    x = np.array([[[0.125, 0.625, 5.0, -1.0]]], np.float32)  # (3x)/0.75 = 4x: 0.5, 2.5, 20, -4
    y = executor._fake_quant(x, np.array([0.75], np.float32), np.array([0.0], np.float32),
                             {"axis": 0, "pre_scale": 3.0, "qmin": 0, "qmax": 3}, np.float32)
    np.testing.assert_array_equal(y, np.array([[[0.0, 1.5, 2.25, 0.0]]], np.float32))


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage", ["reference", "neon"])
def test_numpy_executor_bit_exact_with_cpp(tmp_path, stage):
    g = tiny_bhvit(3)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    rng = np.random.default_rng(5)
    xs = [rng.normal(size=(1, H, W, C)).astype(np.float32) * s for s in (0.5, 1.0, 2.0, 4.0)]
    np.concatenate(xs).tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(len(xs), -1)
    for i, xin in enumerate(xs):
        res = executor.run(g2, {"input": xin})
        want = np.concatenate([res[g2.outputs[0]].ravel(), res[g2.outputs[1]].ravel()])
        np.testing.assert_array_equal(got[i], want)


def _official_tiny():
    repo = os.environ.get("BITINO_BHVIT_REPO", os.path.join(ROOT, ".cache", "repos", "IMRL_BHViT"))
    if not os.path.isdir(os.path.join(repo, "transformer")):
        pytest.skip("official IMRL/BHViT clone not available")
    for mod in ("torch", "timm", "transformers"):
        pytest.importorskip(mod)
    import argparse

    import torch
    sys.dont_write_bytecode = True
    sys.path.insert(0, repo)
    import models as bhvit_models
    a = argparse.Namespace(drop_path=0.0, nb_classes=1000, shift3=True, shift5=True, replace_ln_bn=True, disable_layerscale=False,
                           enable_cls_token=False, gsb=False, recu=False, some_fp=False)
    torch.manual_seed(0)
    try:
        model = bhvit_models.get_model(a, os.path.join(repo, "configs", "BHViT_tiny"), "dbhvit", 1, 1)
    except Exception as e:  # e.g. transformers >= 5 rejects the list-valued hidden_size of the official config
        pytest.skip(f"official BHViT model not constructible in this environment (use the 'bhvit' env): {type(e).__name__}")
    with torch.no_grad():  # non-trivial BN statistics, moves and quantizer parameters
        for n, p in model.named_parameters():
            if n.endswith(("move", "move1", "move2", "moveq", "movek", "movev", "moveq2", "movek2", "movev2", ".bias")):
                p.copy_(torch.randn_like(p) * 0.05)
            elif n.endswith(("alpha",)) and "att_prob_quantizer" in n:
                p.copy_(torch.rand_like(p) * 0.05 + 0.02)
            elif n.endswith("zero_point"):
                p.copy_(torch.randn_like(p))
        for mod in model.modules():
            if type(mod).__name__ == "BatchNormT":
                mod.running_mean.copy_(torch.randn_like(mod.running_mean) * 0.1)
                mod.running_var.copy_(torch.rand_like(mod.running_var) + 0.5)
    model.eval()
    return model, torch


def test_official_bhvit_tiny_random_weights_float64_exact():
    model, torch = _official_tiny()
    from bitino.adapters import bhvit
    g, info = bhvit.convert(model, "bhvit-tiny-binary")
    for name, v in info["float64_constants"].items():
        g.constants[name].data = v
    x = np.random.default_rng(0).normal(size=(1, 224, 224, 3))
    with torch.no_grad():
        ref = model.double()(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).logits.numpy()
    out = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]]
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())
