"""Python API of the native runtime (bitino.load / Model, ctypes over libbitino_c).

* a small random graph: binding vs numpy executor (close) and vs bitino_bench (bit-identical, same runtime);
* a converted model (artifacts, when present): predict() at stage neon bit-exact with the numpy executor's
  recorded outputs, 1 and 4 threads;
* API surface: shapes, batching, named inputs, dtype conversion, errors, benchmark, zoo-id resolution.

Needs the native library: built with -DBITINO_BUILD_SHARED=ON (found under build/*/) or an installed wheel. With
BITINO_TEST_INSTALLED=1 the installed package is tested instead of python/ of this checkout.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
if os.environ.get("BITINO_TEST_INSTALLED") != "1":
    sys.path.insert(0, os.path.join(ROOT, "python"))

import bitino  # noqa: E402
from bitino import _native, executor, format as bfformat, passes, zoo  # noqa: E402
from bitino.ir import Graph  # noqa: E402

pytestmark = pytest.mark.skipif(not _native.available(), reason="native library libbitino_c not built / installed")

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
if not os.path.exists(BENCH):
    BENCH = zoo.find_bench() or BENCH


def small_graph(seed: int = 3) -> Graph:
    """Float conv + two binary convs (ge / gt, padding values 0 / 1) + pooling + dense + softmax."""
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 9, 9, 3])
    y = g.add_node("Conv2D", [x, g.add_constant("w0", rng.normal(size=(16, 3, 3, 3)).astype(np.float32)),
                              g.add_constant("b0", rng.normal(size=16))], {"strides": [2, 2], "pads": [0, 1, 0, 1]})
    y = g.add_node("ChannelAffine", [y, g.add_constant("s0", rng.random(16) + 0.5), g.add_constant("t0", rng.normal(size=16))])
    wb = np.where(rng.random((32, 3, 3, 16)) < 0.5, -1, 1).astype(np.int8)
    b = g.add_node("BConv2D", [y, g.add_constant("wb", wb, "bit"), g.add_constant("m", rng.random(32)),
                               g.add_constant("a", rng.normal(size=32)), ""],
                   {"pads": [1, 1, 1, 1], "pad_value": 0.0, "input_binarize": "ge"})
    wb2 = np.where(rng.random((32, 3, 3, 32)) < 0.5, -1, 1).astype(np.int8)
    b2 = g.add_node("BConv2D", [b, g.add_constant("wb2", wb2, "bit"), "", "", g.add_constant("thr", rng.normal(size=32) * 0.1)],
                    {"pads": [1, 1, 1, 1], "pad_value": 1.0, "input_binarize": "gt", "activation": "relu"})
    s = g.add_node("Add", [b, b2])
    gp = g.add_node("GlobalAvgPool", [s])
    logits = g.add_node("Dense", [gp, g.add_constant("fc", rng.normal(size=(10, 32))), g.add_constant("fcb", rng.normal(size=10))])
    probs = g.add_node("Softmax", [logits])
    g.set_outputs(logits, probs)
    passes.optimize(g, fold=True)
    return g


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    g = small_graph()
    path = str(tmp_path_factory.mktemp("m") / "small.bitino")
    bfformat.write(g, path)
    return g, path


@pytest.mark.parametrize("stage", ["reference", "neon", "all"])
def test_small_model_vs_numpy_executor(small_model, stage):
    g, path = small_model
    xs = np.random.default_rng(7).normal(size=(4, 9, 9, 3)).astype(np.float32)
    with bitino.load(path, stage=stage) as m:
        logits, probs = m.predict(xs)
    assert logits.shape == (4, 10) and probs.shape == (4, 10)
    for i in range(4):
        res = executor.run(g, {"input": xs[i:i + 1]})
        np.testing.assert_allclose(logits[i], res[g.outputs[0]].ravel(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(probs[i], res[g.outputs[1]].ravel(), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not os.path.exists(BENCH), reason="bitino_bench not built")
@pytest.mark.parametrize("stage", ["neon", "all"])
def test_small_model_bit_identical_to_bench(small_model, tmp_path, stage):
    _, path = small_model
    xs = np.random.default_rng(11).normal(size=(5, 1, 9, 9, 3)).astype(np.float32)
    xs.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", path, "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage], check=True, capture_output=True)
    want = np.fromfile(tmp_path / "out.f32", np.float32).reshape(5, 20)
    with bitino.load(path, stage=stage, threads=2) as m:
        logits, probs = m.predict(xs.reshape(5, 9, 9, 3))
    got = np.concatenate([logits, probs], axis=1)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


# ---------------------------------------------------------------------------------------- converted model
ART = os.path.join(MAIN, "artifacts", "ab-bnn-cifar10-reactnet18")
EVAL = os.path.join(MAIN, ".cache", "data", "eval128_cifar10_norm")


@pytest.mark.skipif(not (os.path.exists(os.path.join(ART, "model.bitino")) and os.path.isdir(EVAL)),
                    reason="artifacts / eval set of ab-bnn-cifar10-reactnet18 not present")
@pytest.mark.parametrize("threads", [1, 4])
def test_converted_model_bit_exact_with_numpy_executor(threads):
    man = json.load(open(os.path.join(EVAL, "manifest.json")))
    pre = man["preprocessing"]["cifar10_imagenet_norm"]
    x = np.fromfile(os.path.join(EVAL, pre["file"]), np.float32).reshape(pre["shape"])
    want = np.fromfile(os.path.join(ART, "numpy_logits_fused.f32"), np.float32).reshape(len(x), -1)
    with bitino.load(os.path.join(ART, "model.bitino"), stage="neon", threads=threads) as m:
        assert m.inputs[0].shape == (1, 32, 32, 3)
        y = m.predict(x)
    assert y.shape == want.shape
    assert np.array_equal(y.view(np.uint32), want.view(np.uint32))


# ---------------------------------------------------------------------------------------- API surface
def test_specs_and_shapes(small_model):
    _, path = small_model
    m = bitino.load(path)
    assert [s.name for s in m.inputs] == ["input"]
    assert m.inputs[0].shape == (1, 9, 9, 3) and m.inputs[0].dtype == np.float32
    assert [s.shape for s in m.outputs] == [(1, 10), (1, 10)]
    x = np.random.default_rng(0).normal(size=(9, 9, 3))          # float64, no batch dimension
    a = m.predict(x)[0]
    b = m.predict(x[None].astype(np.float32))[0]
    c = m.predict({"input": x})[0]
    d = m.predict(input=x)[0]
    assert a.shape == (1, 10) and np.array_equal(a, b) and np.array_equal(a, c) and np.array_equal(a, d)
    batch = np.stack([x, -x, 2 * x]).astype(np.float32)
    lg = m.predict(batch)[0]
    assert lg.shape == (3, 10) and np.array_equal(lg[:1], a)
    assert np.array_equal(m(batch)[0], lg)                      # __call__
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 8, 8, 3), np.float32))
    with pytest.raises(TypeError):
        m.predict(x, x)
    with pytest.raises(KeyError):
        m.predict(nope=x)
    assert "kernels" in m.describe() and m.describe()["stage"] == "all"
    assert isinstance(m.manifest, dict)
    m.close()
    with pytest.raises(ValueError):
        m.predict(x)


def test_outputs_are_copies(small_model):
    _, path = small_model
    with bitino.load(path) as m:
        x = np.random.default_rng(1).normal(size=(1, 9, 9, 3)).astype(np.float32)
        y1 = m.predict(x)[0].copy()
        y_keep = m.predict(x)[0]
        m.predict(-x)
        assert np.array_equal(y_keep, y1)                       # later runs do not overwrite returned arrays


def test_stage_errors_and_threads(small_model):
    _, path = small_model
    with pytest.raises(_native.NativeError, match="unknown stage"):
        bitino.load(path, stage="warp_speed")
    with pytest.raises(ValueError):
        bitino.load(path, threads=0)
    with pytest.raises(_native.NativeError):
        bitino.load(__file__)                                 # not a model file
    x = np.random.default_rng(2).normal(size=(2, 9, 9, 3)).astype(np.float32)
    outs = {s: bitino.load(path, stage=s, threads=t).predict(x)[0] for s, t in (("neon", 1), ("all", 3))}
    np.testing.assert_allclose(outs["neon"], outs["all"], rtol=1e-4, atol=1e-4)


def test_benchmark(small_model):
    _, path = small_model
    with bitino.load(path, threads=1) as m:
        med = m.benchmark(iterations=5, warmup=1)
        assert isinstance(med, float) and med > 0
        r = m.benchmark(iterations=4, warmup=0, threads=2, stage="neon", full=True)
        assert r["threads"] == 2 and r["stage"] == "neon" and len(r["samples_ms"]) == 4
        assert r["session"]["threads"] == 2
        assert m.threads == 1 and m.stage == "all"              # the model's own session is unchanged
        x = np.ones((1, 9, 9, 3), np.float32)
        assert m.benchmark(iterations=2, warmup=0, inputs=(x,)) > 0


def test_cpu_info():
    ci = bitino.cpu_info()
    assert ci["arch"] in ("x86_64", "aarch64", "arm", "unknown") and isinstance(ci["features"], list)
    assert ci["runtime_version"] == bitino.version()


def test_zoo_resolution(small_model, tmp_path, monkeypatch):
    _, path = small_model
    monkeypatch.setenv("BITINO_HOME", str(tmp_path))
    rel = sorted(k for k in (zoo._load(os.path.join("registry", "release_models.json")).get("models") or {})
                 if zoo.release_entry(k)[0])
    nonrel = next((k for k in sorted(zoo.recipes()) if not zoo.release_entry(k)[0]), None)
    if rel:
        mid = rel[0]
        with pytest.raises(FileNotFoundError):
            bitino.load(mid, download=False)                  # redistributable, not downloaded
        d = os.path.join(tmp_path, "models", mid)
        os.makedirs(d)
        shutil.copy(path, os.path.join(d, "model" + zoo.MODEL_EXT))
        m = bitino.load(mid)                                  # already in BITINO_HOME: no download
        assert m.model_id == mid and m.preprocess == zoo.preprocess_key(mid)
        m.close()
    if nonrel:
        with pytest.raises(PermissionError, match="weights licence"):
            bitino.load(nonrel, download=False)
    with pytest.raises(KeyError):
        bitino.load("no-such-model-id")
    with pytest.raises(FileNotFoundError):
        bitino.load("missing/file" + zoo.MODEL_EXT)


def test_image_preprocessing_shapes(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    from bitino import images
    img = PIL.fromarray((np.arange(300 * 200 * 3) % 251).astype(np.uint8).reshape(300, 200, 3))
    for key in images.supported():
        x = images.preprocess(img, key)
        side = images.SPECS[key]["crop"]
        assert x.shape == (1, side, side, 3) and x.dtype == np.float32
    img.save(tmp_path / "a.png")
    assert np.array_equal(images.load_image(str(tmp_path / "a.png"), "omz_bgr"), images.preprocess(img, "omz_bgr"))
    with pytest.raises(ValueError):
        images.preprocess(img, "larq_zoo")
