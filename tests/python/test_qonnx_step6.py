"""QONNX route step 6 (docs/qonnx-step6.md): QONNX is the build path.

Covered: every canonical file determines its .bitino alone (tools/import_qonnx.py with no import argument = with the
file's own layout; an argument that contradicts bitino.import, an option's default or bitino.input_domain is an error;
a group-mapped file takes no import argument), the step-1 / 1b / 2 writers' bitino.import (the values of their former
import arguments), the *_unfused derivation of a pattern-mapped file (--no-fold = the graph before the fold passes),
the L8ENC bisection's numpy mirror, and the route table registry/qonnx_routes.json against the recipes.

  .cache/venv-qonnx/bin/python -I -m pytest tests/python/test_qonnx_step6.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

onnx = pytest.importorskip("onnx")

from bitino import format as bfformat, int8_ops, passes, qonnx_int8 as qi8  # noqa: E402
from bitino import qonnx_groups as qgr, qonnx_import as qi  # noqa: E402
from bitino.fetch import data_sha256, graph_sha256  # noqa: E402
from qonnx_ir_writer import IRWriter, ir_shapes  # noqa: E402
import test_qonnx_step4 as t4  # noqa: E402


def _cli(path, out, *extra):
    return qi.main([str(path), "--model-id", "t", "--out-dir", str(out), *extra])


def _hashes(p):
    return data_sha256(str(p)), graph_sha256(str(p))


def test_pattern_file_determines_its_import(tmp_path):
    rng = np.random.default_rng(5)
    g0, c64 = t4._dabnn_like(rng)
    m = t4.to_model(IRWriter(g0, ir_shapes(g0), c64).write(), t4.KEEP)
    onnx.save(m, str(tmp_path / "m.onnx"))
    assert _cli(tmp_path / "m.onnx", tmp_path / "a") == 0
    # the same arguments the file records: the same file
    assert _cli(tmp_path / "m.onnx", tmp_path / "b", "--naming", "exporter", "--affine", "keep", "--constant-order", "first-use",
                "--cleanup", "none", "--bit-outputs", "none", "--input-domain", "float32") == 0
    assert _hashes(tmp_path / "a" / "model.bitino") == _hashes(tmp_path / "b" / "model.bitino")
    gp = copy.deepcopy(g0)
    passes.optimize(gp, fold=True)
    gp.manifest = {}
    bfformat.write(gp, str(tmp_path / "legacy.bitino"))
    assert _hashes(tmp_path / "a" / "model.bitino") == _hashes(tmp_path / "legacy.bitino")
    # contradictions: a listed option, an option the file leaves at its default (flatten_dense transpose), the domain
    for bad in (["--naming", "index"], ["--affine", "fold"], ["--bit-outputs", "set"], ["--flatten-dense", "permute"],
                ["--input-domain", "uint8"]):
        with pytest.raises(SystemExit):
            _cli(tmp_path / "m.onnx", tmp_path / "c", *bad)


def test_unfused_derivation_is_the_pre_pass_graph(tmp_path):
    """--no-fold: the derived *_unfused.bitino of a pattern-mapped file is the importer's graph before the fold passes
    (the legacy converters' unfused file is their graph before passes.optimize(fold=True))."""
    rng = np.random.default_rng(6)
    g0, c64 = t4._dabnn_like(rng)
    m = t4.to_model(IRWriter(g0, ir_shapes(g0), c64).write(), t4.KEEP)
    onnx.save(m, str(tmp_path / "m.onnx"))
    assert _cli(tmp_path / "m.onnx", tmp_path / "u", "--no-fold", "--file-name", "model_unfused.bitino") == 0
    gu = copy.deepcopy(g0)
    passes.optimize(gu, fold=False)
    gu.manifest = {}
    bfformat.write(gu, str(tmp_path / "legacy_unfused.bitino"))
    assert _hashes(tmp_path / "u" / "model_unfused.bitino") == _hashes(tmp_path / "legacy_unfused.bitino")
    g = bfformat.read(str(tmp_path / "u" / "model_unfused.bitino"))
    assert "ChannelAffine" in g.op_counts()          # the affines are still separate nodes


def test_group_file_takes_no_import_argument(tmp_path):
    from bitino.ir import Graph
    g = Graph()
    x = g.add_input("input", [1, 4, 4, 3])
    y = g.add_node("ChannelAffine", [x, g.add_constant("a/scale", np.array([1.5, 2, 3], np.float32)),
                                     g.add_constant("a/shift", np.array([0.5, 0, -1], np.float32))], name="a")
    y = g.add_node("GlobalAvgPool", [y], {"keepdims": 0}, name="gap")
    g.set_outputs(y)
    spec, _ = qgr.write_spec(g)
    spec.meta.update({**t4.META, "bitino.import": json.dumps({"mapping": "groups"})})
    qgr.save_spec(spec, str(tmp_path / "spec"))
    qgr.assemble(str(tmp_path / "spec"), str(tmp_path / "g.onnx"), "test")
    assert _cli(tmp_path / "g.onnx", tmp_path / "ok") == 0
    for bad in (["--naming", "exporter"], ["--no-fold"], ["--cleanup", "none"], ["--input-domain", "uint8"]):
        with pytest.raises(SystemExit):
            _cli(tmp_path / "g.onnx", tmp_path / "x", *bad)


def test_group_writer_unfused_constructs(tmp_path):
    """The step-6 group-writer constructs of the legacy pre-pass graphs (tools/export_unfused_qonnx.py): a standalone
    Activation(sign) (BipolarQuant), a float Conv2D on a binarized tensor (explicit Pad(0) + bitino.float_layer, R2),
    the IR Pad (AdaBin's channel padding), BDense with activation "none" written by the converter (an added default) and
    with a fused ReLU; check-qonnx passes without error and the import is data + graph identical."""
    from bitino.ir import Graph
    from bitino.qonnx_profile import check_model
    rng = np.random.default_rng(8)
    g = Graph()
    x = g.add_input("input", [1, 6, 6, 3])
    y = g.add_node("Conv2D", [x, g.add_constant("c0/kernel", rng.normal(0, 1, (4, 3, 3, 3)).astype(np.float32), layout="OHWI"), ""],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "activation": "none", "pad_value": 0.0}, name="c0")
    y = g.add_node("ChannelAffine", [y, g.add_constant("bn/scale", rng.uniform(0.5, 2, 4).astype(np.float32)),
                                     g.add_constant("bn/shift", rng.normal(0, 1, 4).astype(np.float32))], name="bn")
    y = g.add_node("Activation", [y], {"type": "sign"}, name="sign")
    y = g.add_node("Conv2D", [y, g.add_constant("c1/kernel", rng.normal(0, 1, (4, 3, 3, 4)).astype(np.float32), layout="OHWI"), ""],
                   {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1], "groups": 1, "activation": "relu", "pad_value": 0.0}, name="c1")
    y = g.add_node("Pad", [y], {"pads": [0, 0, 0, 0, 0, 0, 2, 2], "value": 0.0}, name="pad")
    y = g.add_node("GlobalAvgPool", [y], {"keepdims": 0}, name="gap")
    bits = np.where(rng.random((5, 8)) < 0.5, -1, 1).astype(np.int8)
    y = g.add_node("BDense", [y, g.add_constant("fc/kernel_bits", bits, dtype="bit", layout="OI_bit"), "", "", "", ""],
                   {"input_binarize": "ge", "activation": "none"}, name="fc")
    bits2 = np.where(rng.random((3, 5)) < 0.5, -1, 1).astype(np.int8)
    y = g.add_node("BDense", [y, g.add_constant("fc2/kernel_bits", bits2, dtype="bit", layout="OI_bit"),
                              g.add_constant("fc2/post_mul", np.full(3, 0.5, np.float32)), "", "", ""],
                   {"input_binarize": "ge", "activation": "relu"}, name="fc2")
    g.set_outputs(y)
    spec, _ = qgr.write_spec(g)
    spec.meta.update({**t4.META, "bitino.labels": "none"})
    qgr.save_spec(spec, str(tmp_path / "spec"))
    m = qgr.assemble(str(tmp_path / "spec"), str(tmp_path / "u.onnx"), "test")
    rep = check_model(m)
    assert not rep.codes("error") and "I-FLOAT-LAYER" in rep.codes("info"), rep.to_text()
    ops = [n.op_type for n in m.graph.node]
    assert ops.count("Pad") == 2 and "BipolarQuant" in ops
    g2, _ = qgr.import_groups(onnx.load(str(tmp_path / "u.onnx")))
    g.manifest, g2.manifest = {}, {}
    bfformat.write(g, str(tmp_path / "a.bitino"))
    bfformat.write(g2, str(tmp_path / "b.bitino"))
    assert _hashes(tmp_path / "a.bitino") == _hashes(tmp_path / "b.bitino")


def test_writers_record_the_import_layout():
    import export_keras_qonnx as ekq
    import export_torch_qonnx as etq
    t = json.loads(etq.import_meta(etq.IMPORT_ARGS_TORCH))
    assert t == {"mapping": "patterns", "naming": "exporter", "constant_order": "first-use", "affine": "keep", "cleanup": "none",
                 "bit_outputs": "set", "flatten_dense": "transpose"}
    assert json.loads(etq.import_meta(etq.IMPORT_ARGS_TORCH + ["--flatten-dense", "permute"]))["flatten_dense"] == "permute"
    assert json.loads(etq.import_meta(etq.IMPORT_ARGS_IMAGE))["constant_order"] == "walk"     # the importer default
    k = json.loads(ekq.IMPORT_META_KERAS)
    assert k["bit_outputs"] == "none" and k["constant_order"] == "first-use" and k["affine"] == "keep"
    assert set(t) - {"mapping"} == set(qi.PATTERN_IMPORT_OPTIONS) == set(k) - {"mapping"}
    with pytest.raises(AssertionError):
        etq.import_meta(["--no-fold"])


def test_l8enc_bisection_mirror():
    rng = np.random.default_rng(7)
    w = np.concatenate([rng.normal(0, 1, 20000), rng.normal(0, 1e-4, 20000), [0.0, -0.0, np.inf, -np.inf]]).astype(np.float32)
    b = qi8.beta()
    w = np.concatenate([w, b, np.nextafter(b, np.float32(0)), -b])
    assert np.array_equal(qi8.l8enc_bisect(w), int8_ops.l8_encode(w))


def test_routes_cover_every_recipe():
    routes = json.load(open(os.path.join(ROOT, "registry", "qonnx_routes.json")))
    recs = json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]
    assert set(routes["routes"]) == set(recs)
    n_pub = 0
    for mid, r in routes["routes"].items():
        assert r["front_end"] in ("pytorch", "keras", "groups", "ir")
        for fn, f in r["files"].items():
            if f["role"] == "published":
                n_pub += 1
                assert f["canonical"]["writer"].startswith("tools/export_") and os.path.exists(os.path.join(ROOT, f["canonical"]["writer"]))
                assert not f["canonical"]["writer"].startswith("tools/convert_")
            else:
                assert f.get("derived_from") in r["files"] or f.get("canonical")
        for fn in recs[mid].get("expected", {}):
            assert fn in r["files"] and r["files"][fn]["role"] == "published"
    assert n_pub == routes["counts"]["published_files"] == 296
