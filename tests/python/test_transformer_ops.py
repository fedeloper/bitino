"""Transformer operators: numpy executor vs C++ runtime (bit-exact), multi-input bench records."""
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
L, D, H, V = 8, 64, 2, 50


def tiny_bert(seed: int) -> Graph:
    """One BiBERT-style layer with every new op (Gather bit+f32, LayerNorm, ternary BDense with
    zero weights, heads Reshape/Transpose, BMatMul sign_zero / gt0, mask AddBroadcast, GELU, tanh,
    MatMul)."""
    rng = np.random.default_rng(seed)
    g = Graph()
    ids = g.add_input("input_ids", [1, L], "i32")
    mask = g.add_input("attention_mask", [1, L], "f32")
    pm = np.where(rng.random((V, D)) < 0.5, -1, 1).astype(np.int8)
    zm = np.where(rng.random((V, D)) < 0.05, -1, 1).astype(np.int8)
    we = g.add_node("Gather", [g.add_constant("wtab", pm, "bit"), ids, g.add_constant("wscale", rng.random(V) + 0.1),
                               g.add_constant("wzero", zm, "bit")])
    pe = g.add_node("Gather", [g.add_constant("ptab", rng.normal(size=(L, D))), g.add_constant("pids", np.arange(L, dtype=np.int32)[None], "i32")])
    h = g.add_node("LayerNorm", [g.add_node("Add", [we, pe]), g.add_constant("g0", rng.random(D) + 0.5), g.add_constant("b0", rng.normal(size=D))],
                   {"eps": 1e-12})

    def bdense(x, o, k, name, zeros=False):
        w = np.where(rng.random((o, k)) < 0.5, -1, 1).astype(np.int8)
        ins = [x, g.add_constant(name + "_w", w, "bit"), g.add_constant(name + "_a", rng.random(o) * 0.05),
               g.add_constant(name + "_b", rng.normal(size=o) * 0.1), ""]
        if zeros:
            ins.append(g.add_constant(name + "_z", np.where(rng.random((o, k)) < 0.02, -1, 1).astype(np.int8), "bit"))
        return g.add_node("BDense", ins, {"input_binarize": "sign_zero"})

    def heads(x):
        return g.add_node("Transpose", [g.add_node("Reshape", [x], {"shape": [1, L, H, D // H]})], {"perm": [0, 2, 1, 3]})

    q, k, v = (heads(bdense(h, D, D, n, zeros=(n == "k"))) for n in ("q", "k", "v"))
    s = g.add_node("BMatMul", [q, k], {"transpose_b": 1, "a_binarize": "sign_zero", "b_binarize": "sign_zero", "scale": 0.125})
    m = g.add_node("ChannelAffine", [mask, g.add_constant("ms", np.full(L, 10000.0)), g.add_constant("mb", np.full(L, -10000.0))])
    s = g.add_node("AddBroadcast", [s, g.add_node("Reshape", [m], {"shape": [1, 1, 1, L]})])
    c = g.add_node("BMatMul", [s, v], {"transpose_b": 0, "a_binarize": "gt0", "b_binarize": "sign_zero", "scale": 1.0})
    c = g.add_node("Reshape", [g.add_node("Transpose", [c], {"perm": [0, 2, 1, 3]})], {"shape": [1, L, D]})
    a1 = g.add_node("LayerNorm", [g.add_node("Add", [bdense(c, D, D, "o"), h]), g.add_constant("g1", rng.random(D) + 0.5), ""], {"eps": 1e-12})
    f = g.add_node("Activation", [bdense(a1, 4 * D, D, "f1")], {"type": "gelu"})
    h2 = g.add_node("LayerNorm", [g.add_node("Add", [bdense(f, D, 4 * D, "f2", zeros=True), a1])], {"eps": 1e-12})
    cls = g.add_node("Gather", [g.add_node("Reshape", [h2], {"shape": [L, D]}), g.add_constant("cls", np.zeros(1, np.int32), "i32")])
    p = g.add_node("Activation", [bdense(cls, D, D, "pool")], {"type": "tanh"})
    mm = g.add_node("MatMul", [p, g.add_constant("mmw", rng.normal(size=(D, 3)))], {"scale": 0.5})
    logits = g.add_node("Dense", [mm, g.add_constant("cw", rng.normal(size=(2, 3))), g.add_constant("cb", rng.normal(size=2))])
    g.set_outputs(logits, h2)
    return g


def feeds(n: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ln = int(rng.integers(3, L + 1))
        ids = np.zeros((1, L), np.int32)
        ids[0, :ln] = rng.integers(0, V, ln)
        ids[0, 1] = V + 3 if ln > 4 else ids[0, 1]  # out-of-range id -> zero row (documented)
        mask = np.zeros((1, L), np.float32)
        mask[0, :ln] = 1
        out.append({"input_ids": ids, "attention_mask": mask})
    return out


def test_ops_shapes_and_float64_mode():
    g = tiny_bert(0)
    f = feeds(1, 1)[0]
    r32 = executor.run(g, f)
    r64 = executor.run(g, f, float_dtype=np.float64)
    assert r32[g.outputs[0]].shape == (1, 2) and r32[g.outputs[1]].shape == (1, L, D)
    assert r64[g.outputs[0]].dtype == np.float64


def test_gather_out_of_range_and_zero_mask():
    t = np.array([[1, -1, 1], [-1, -1, 1]], np.int8)
    z = np.array([[1, 1, -1], [1, 1, 1]], np.int8)
    y = executor._gather(t, np.array([[1, 0, 5]]), np.array([2.0, 0.5], np.float32), z, np.float32)
    assert np.array_equal(y, np.array([[[-0.5, -0.5, 0.5], [2.0, -2.0, 0.0], [0, 0, 0]]], np.float32))


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage", ["reference", "neon"])
def test_numpy_executor_bit_exact_with_cpp_multi_input(tmp_path, stage):
    g = tiny_bert(3)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    fs = feeds(6, 5)
    with open(tmp_path / "in.bin", "wb") as fh:  # per-sample record: inputs in graph order
        for f in fs:
            fh.write(f["input_ids"].tobytes() + f["attention_mask"].tobytes())
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.bin"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(len(fs), -1)
    for i, f in enumerate(fs):
        res = executor.run(g2, f)
        want = np.concatenate([res[g2.outputs[0]].ravel(), res[g2.outputs[1]].ravel()])
        np.testing.assert_array_equal(got[i], want)
