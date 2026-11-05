"""Slice S8: BBCU restoration / SR (adapter torch_bbcu) and BSRC-2023 SR baselines (adapter torch_bsrc).

* restoration_common: PixelShuffle (Reshape/Transpose), nearest upsampling (row Gather) and bilinear upsampling
  (replicate-pad Gather + phase Conv2D + PixelShuffle) equal torch's F.pixel_shuffle / F.interpolate exactly in
  float64 (r = 2, 4; odd and non-square sizes; borders).
* A synthetic graph with every op of both adapters (BConv2D sign_zero / ge / gt with post_add, LearnableBias
  ChannelAffine, PReLU, LeakyReLU, grouped Conv2D with multiplier 2 and the grouped 1x1 (-1, +1) difference of
  LAB, row Gather, 6-D Transpose, Reshape, Add): numpy executor vs C++ runtime bit-exact (reference, neon), and
  the optimized stage within float tolerance.
* If the clones are available (BBCU: .cache/repos/BBCU or BITINO_BBCU_REPO; BSRC: .cache/repos/BSRC-2023-62bbcf2
  or BITINO_BSRC_REPO), every source architecture with random weights is converted and compared with the
  float64 source using exact float64 constants (<= 1e-9 relative), and the BSRC stub-unpickler is exercised on
  a synthetic whole-module pickle.
* validate_model.py --output-kind image on a small synthetic artifact (exact stage and --explain).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters import restoration_common as rc  # noqa: E402
from bitino.ir import Graph  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
BBCU_REPO = os.environ.get("BITINO_BBCU_REPO", os.path.join(MAIN, ".cache", "repos", "BBCU"))
BSRC_REPO = os.environ.get("BITINO_BSRC_REPO", os.path.join(MAIN, ".cache", "repos", "BSRC-2023-62bbcf2"))


@pytest.mark.parametrize("h,w,c,r", [(5, 7, 3, 2), (6, 4, 3, 4), (1, 3, 2, 2), (3, 3, 16, 2), (2, 5, 4, 4)])
def test_resampling_matches_torch_float64(h, w, c, r):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    # float32-valued inputs (the runtime's): the dyadic phase products are then exact in float64, so even the
    # bilinear sums equal torch's float64 result bit for bit
    x = (np.random.default_rng(h * 31 + w).normal(size=(1, h, w, c)) * 40).astype(np.float32).astype(np.float64)
    xt = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
    cases = [("nearest", lambda g, i: rc.nearest_upsample(g, i, h, w, c, r, "n"), F.interpolate(xt, scale_factor=r, mode="nearest")),
             ("bilinear", lambda g, i: rc.bilinear_upsample(g, i, h, w, c, r, "b"),
              F.interpolate(xt, scale_factor=r, mode="bilinear", align_corners=False))]
    if c % (r * r) == 0:
        cases.append(("shuffle", lambda g, i: rc.pixel_shuffle(g, i, h, w, c, r, "p"), F.pixel_shuffle(xt, r)))
    for kind, build, ref in cases:
        g = Graph()
        o = build(g, g.add_input("input", [1, h, w, c]))
        g.set_outputs(o)
        y = executor.run(g, {"input": x}, float_dtype=np.float64)[o]
        np.testing.assert_array_equal(y, ref.numpy().transpose(0, 2, 3, 1), err_msg=kind)


def test_bilinear_rejects_non_dyadic_scale():
    with pytest.raises(AssertionError):
        rc.bilinear_kernel(3, 3)


H, W, C = 6, 5, 16


def restoration_like(seed: int, binarize: str) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, H, W, 3])

    def cst(name, arr, dtype="f32", layout=""):
        return g.add_constant(name, np.asarray(arr), dtype, layout)

    def pm1(shape):
        return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int8)

    def conv(t, tag, cin, cout, k, groups=1, act="none", w=None):
        w = rng.normal(size=(cout, k, k, cin // groups)) * 0.3 if w is None else w
        return g.add_node("Conv2D", [t, cst(tag + "w", w, layout="OHWI"), cst(tag + "b", rng.normal(size=cout) * 0.1)],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [k // 2] * 4, "groups": groups, "pad_value": 0.0, "activation": act})

    def bconv(t, tag, cin, cout, mode, bias):
        return g.add_node("BConv2D", [t, cst(tag + "w", pm1((cout, 3, 3, cin)), "bit", "OHWI_bit"), cst(tag + "pm", rng.random(cout) * 0.1),
                                      cst(tag + "pa", rng.normal(size=cout)) if bias else "", ""],
                          {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "pad_value": 0.0,
                           "input_binarize": mode, "activation": "none", "output": "f32"})

    def affine(t, tag, c, scale=True, quant=False):
        b = np.round(rng.normal(size=c) * 4) / 8 if quant else rng.normal(size=c)
        return g.add_node("ChannelAffine", [t, cst(tag + "s", rng.random(c) + 0.5) if scale else "", cst(tag + "b", b)])

    xk = g.add_node("ChannelAffine", [x, cst("k", np.full(3, 130.0)), ""])
    t = g.add_node("Activation", [conv(xk, "first", 3, C, 3)], {"type": "leaky_relu", "alpha": 0.1})
    # BBCU BinaryConv2d: move0 -> sign -> bconv (+bias) -> RPReLU -> + x
    u = bconv(affine(t, "move0", C, scale=False, quant=True), "b1", C, C, binarize, True)
    u = affine(g.add_node("Activation", [affine(u, "pr0", C, scale=False), cst("sl", rng.random(C) * 0.5)], {"type": "prelu"}), "pr1", C, scale=False)
    t = g.add_node("Add", [u, t])
    # LAB: grouped conv C -> 2C (multiplier 2), grouped 1x1 difference (-1, +1), BConv2D gt, Scale
    d = conv(t, "lab", C, 2 * C, 3, groups=C)
    k = np.zeros((C, 1, 1, 2))
    k[:, 0, 0, 0], k[:, 0, 0, 1] = -1, 1
    d = g.add_node("Conv2D", [d, cst("diff", k, layout="OHWI"), ""], {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0],
                                                                    "groups": C, "pad_value": 0.0, "activation": "none"})
    p0 = affine(bconv(d, "b2", C, C, "gt", True), "scale", C)
    p1 = conv(t, "p1", C, C, 1)
    p2 = conv(p1, "dw", C, C, 3, groups=C // 4)
    t = g.add_node("Add", [g.add_node("Activation", [g.add_node("Add", [g.add_node("Add", [p0, p1]), p2]), cst("sl2", rng.random(C) * 0.5)],
                                      {"type": "prelu"}), t])
    # BBCUM upsampler: bconv C -> 4C, pixel shuffle, + nearest(t); bilinear global skip
    u = bconv(affine(t, "move1", C, scale=False), "b3", C, 4 * C, "ge", False)
    u = rc.pixel_shuffle(g, u, H, W, 4 * C, 2, "ps")
    t = g.add_node("Add", [u, rc.nearest_upsample(g, t, H, W, C, 2, "nn")])
    out = conv(t, "last", C, 3, 3)
    out = g.add_node("Add", [out, rc.bilinear_upsample(g, xk, H, W, 3, 2, "bl")])
    g.set_outputs(out)
    return g


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("reference", "ge"), ("neon", "ge"), ("optimized", "ge")])
def test_numpy_executor_vs_cpp(tmp_path, stage, binarize):
    g = restoration_like(3, binarize)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    rng = np.random.default_rng(5)
    # inputs on a 1/1024 grid: x*130 and the quantized move0 give exact zeros at the first binarization
    xs = [(np.round(rng.random(size=(1, H, W, 3)) * 1024) / 1024).astype(np.float32) for _ in range(3)]
    np.concatenate(xs).tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(len(xs), -1)
    for i, xin in enumerate(xs):
        want = executor.run(g2, {"input": xin})[g2.outputs[0]].ravel()
        if stage == "optimized":
            np.testing.assert_allclose(got[i], want, rtol=1e-3, atol=1e-2 * np.abs(want).max())
        else:
            np.testing.assert_array_equal(got[i], want)


def _randomize(model, torch):
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 4:
                p.copy_(torch.randn_like(p) * 0.05)
            elif "prelu" in n or n.endswith(("act.weight", "conv_path.2.weight")) or (n.endswith(".weight") and p.dim() == 1 and "conv" not in n):
                p.copy_(torch.rand_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1)
        for m in model.modules():
            if type(m).__name__ == "BatchNorm2d":
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
    model.eval()


def _float64_check(model, graph, info, x, torch):
    for name, v in info["float64_constants"].items():
        graph.constants[name].data = v
    with torch.no_grad():
        ref = model.double()(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy().transpose(0, 2, 3, 1)
    model.float()
    out = executor.run(graph, {"input": x}, float_dtype=np.float64)[graph.outputs[0]]
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize("model_id", ["bbcu-denoise", "bbcu-deblocking", "bbcu-sr-l-x2", "bbcu-sr-l-x4", "bbcu-sr-m-x2", "bbcu-sr-m-x4"])
def test_bbcu_source_random_weights_float64_exact(model_id):
    torch = pytest.importorskip("torch")
    if not os.path.exists(os.path.join(BBCU_REPO, "BBCU-SR", "quant_convs_bbcu.py")):
        pytest.skip("BBCU clone not available")
    import convert_restoration as cr
    model, _, _ = cr.load_model("bbcu", BBCU_REPO, model_id, None)
    _randomize(model, torch)
    hw = (10, 7)
    g, info = cr.convert_graph("bbcu", model, model_id, "sign_zero", hw)
    c = 1 if model_id == "bbcu-deblocking" else 3
    x = np.random.default_rng(1).random(size=(1,) + hw + (c,))
    _float64_check(model, g, info, x, torch)


@pytest.mark.parametrize("model_id", ["bsrc2023-binary-sr-x2-v0", "bsrc2023-binary-sr-x2-v1", "bsrc2023-binary-sr-x4-v0", "bsrc2023-binary-sr-x4-v1"])
def test_bsrc_source_random_weights_float64_exact(model_id):
    torch = pytest.importorskip("torch")
    if not os.path.exists(os.path.join(BSRC_REPO, "bnn_competition", "models", "bnn")):
        pytest.skip("BSRC-2023 clone not available")
    from bitino.adapters import torch_bsrc
    model, _, _ = torch_bsrc.load_model(BSRC_REPO, model_id, None)
    _randomize(model, torch)
    hw = (9, 6)
    g, info = torch_bsrc.convert(model, model_id, input_hw=hw)
    x = np.random.default_rng(2).random(size=(1,) + hw + (3,)) * 255
    _float64_check(model, g, info, x, torch)


def test_bsrc_stub_unpickler_roundtrip(tmp_path):
    """A whole-module pickle of the official class is read back through the inert stubs (torch.load weights_only=True)
    and gives the same tensors and the same model outputs; no pickled class is ever instantiated for real."""
    torch = pytest.importorskip("torch")
    if not os.path.exists(os.path.join(BSRC_REPO, "bnn_competition", "models", "bnn")):
        pytest.skip("BSRC-2023 clone not available")
    import types
    from bitino.adapters import torch_bsrc
    model, _, _ = torch_bsrc.load_model(BSRC_REPO, "bsrc2023-binary-sr-x2-v1", None)
    _randomize(model, torch)
    # pickling needs the classes importable under their bnn_competition.* names: register the already-built
    # module objects under those names only for the duration of torch.save
    leaves = {type(m).__module__ for m in model.modules() if type(m).__module__.startswith("bnn_competition")}
    names = {".".join(n.split(".")[:i]) for n in leaves for i in range(1, n.count(".") + 2)}
    saved = {k: sys.modules.get(k) for k in names}
    try:
        for k in sorted(names):
            sys.modules[k] = types.ModuleType(k)
        for m in model.modules():
            mod = type(m).__module__
            if mod.startswith("bnn_competition"):
                setattr(sys.modules[mod], type(m).__qualname__, type(m))
        path = tmp_path / "whole.ckpt"
        torch.save(model, str(path))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    root = torch_bsrc.load_checkpoint(str(path))
    assert type(root).__module__ == "bnn_competition.models.bnn.binary_scalex2_baseline_v1"
    assert isinstance(root, torch_bsrc._Stub) and not isinstance(root, torch.nn.Module)
    tree = torch_bsrc.stub_tree(root)
    assert set(tree) == {n for n, _ in model.named_modules()}
    sd = torch_bsrc.stub_state_dict(root)
    ref = model.state_dict()
    assert set(sd) == set(ref) and all(torch.equal(sd[k], ref[k]) for k in ref)


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
def test_validate_model_image_mode(tmp_path):
    """--output-kind image on a synthetic artifact: neon bit-exact with numpy (PASS via --accept-exact-numpy), the
    default PSNR criterion, and --explain at the optimized stage."""
    g = restoration_like(9, "ge")
    art, ev = tmp_path / "art", tmp_path / "ev"
    art.mkdir()
    ev.mkdir()
    bfformat.write(g, str(art / "model.bitino"))
    rng = np.random.default_rng(4)
    n = 4
    x = rng.random(size=(n, H, W, 3)).astype(np.float32)
    x.tofile(ev / "in.f32")
    outs = np.stack([executor.run(g, {"input": x[i:i + 1]})[g.outputs[0]][0] for i in range(n)]).astype(np.float32)
    outs.tofile(art / "numpy_logits_fused.f32")
    outs.tofile(art / "reference_logits.f32")
    outs.astype(np.float64).tofile(art / "reference_logits_f64.f64")
    lo, hi = float(outs.min()), float(outs.max())
    tg = np.clip((outs - lo) / (hi - lo) * 255 + rng.normal(size=outs.shape) * 4, 0, 255).astype(np.uint8)
    tg.tofile(ev / "targets.u8")
    man = {"name": "synthetic", "limitations": "test", "samples": [{"label": -1}] * n,
           "preprocessing": {"p": {"file": "in.f32", "shape": [n, H, W, 3], "definition": "random"}},
           "targets": {"file": "targets.u8", "shape": list(tg.shape)},
           "image_task": {"metric": "bsrc_y", "crop_border": 2, "output_range": [lo, hi]}}
    json.dump(man, open(ev / "manifest.json", "w"))
    base = [sys.executable, os.path.join(ROOT, "tools", "validate_model.py"), "--artifacts", str(art), "--eval-dir", str(ev),
            "--preprocess", "p", "--bench", BENCH, "--output-kind", "image", "--threads", "2"]
    r = subprocess.run(base + ["--stage", "neon", "--accept-exact-numpy", "--json", str(tmp_path / "a.json")], capture_output=True, text=True)
    assert r.returncode == 0 and "exact=True" in r.stdout, r.stdout + r.stderr
    rep = json.load(open(tmp_path / "a.json"))
    assert rep["vs_numpy_executor_f32"]["exact_match"] and rep["output_kind"] == "image"
    r = subprocess.run(base + ["--stage", "neon", "--json", str(tmp_path / "b.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr        # identical to the references -> PSNR criterion holds
    r = subprocess.run(base + ["--stage", "optimized", "--explain", "--image-tol", "0", "--json", str(tmp_path / "c.json")],
                       capture_output=True, text=True)
    rep = json.load(open(tmp_path / "c.json"))
    assert "explain" in rep and rep["explain"]["n_samples_traced"] == n - rep["explain"]["n_samples_identical_to_numpy"]
