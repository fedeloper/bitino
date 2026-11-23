"""`bitino fetch` builds through QONNX: the recipe format (registry/recipes.json, tools/build_recipes.py) and the
step runner of python/bitino/fetch.py.

No network, no framework: the recipe data is checked as data, and fetch() runs fake writer / importer / quantizer
tools (python -I scripts written here) on tiny .bitino files, so the moves, the records and the identity checks
are exercised without the reference environments. The real end-to-end runs are logged in
registry/recipes_verified.json -> verified_qonnx.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import fetch as F, format as bfformat, zoo  # noqa: E402
from bitino.ir import Graph  # noqa: E402

RECIPES = zoo.recipe_doc()
ROUTES = json.load(open(os.path.join(ROOT, "registry", "qonnx_routes.json")))["routes"]
LAYOUT_FLAGS = {"--naming", "--constant-order", "--affine", "--cleanup", "--flatten-dense", "--bit-outputs", "--input-domain"}


# ------------------------------------------------------------------------------------------------ recipe data
@pytest.mark.skipif(not RECIPES, reason="registry/recipes.json not generated")
def test_recipes_build_through_qonnx():
    envs = RECIPES["envs"]
    for mid, r in RECIPES["recipes"].items():
        route = ROUTES[mid]
        assert r["build"] == "qonnx" and r["qonnx"]["front_end"] == route["front_end"], mid
        tools = [s["tool"] for s in r["steps"]]
        # the legacy converter is a cross-check only, never a build step
        assert not any(t.startswith("tools/convert_") for t in tools), (mid, tools)
        assert r["converter"] in [s["tool"] for s in r["crosscheck"]["steps"]], mid
        assert all(s["env"] in envs for s in r["steps"] + r["crosscheck"]["steps"]), mid
        assert set(r["envs"]) <= set(envs), mid
        # every file of the route has recorded hashes and is built by the steps
        assert set(r["expected"]) == set(route["files"]), mid
        canon = {}
        for s in r["steps"]:
            if s.get("role") == "qonnx-writer":
                assert s["tool"].startswith("tools/export_") and os.path.exists(os.path.join(ROOT, s["tool"])), mid
                dst = [d for d in s["outputs"].values() if d.endswith(".qonnx.onnx")]
                assert dst == [s["file"][: -len(".bitino")] + ".qonnx.onnx"], (mid, s)
                assert not any("{cache}" in a or "{artifacts}" in a for a in s["args"]), (mid, s["args"])
                canon[s["file"]] = dst[0]
            if s["tool"] == "tools/import_qonnx.py":
                # no import argument: the canonical file determines its .bitino; --no-fold only as a derivation
                assert not (LAYOUT_FLAGS & set(s["args"])), (mid, s["args"])
                assert s["args"][0].startswith("{out_dir}/") and s["args"][0].endswith(".qonnx.onnx")
                assert s["args"][0][len("{out_dir}/"):] in canon.values(), (mid, s["args"][0])
                if "--no-fold" in s["args"]:
                    assert s["role"] == "derivation" and s["file"].endswith("_unfused.bitino")
                assert list(s["outputs"].values())[0] == s["file"]
        built = {s["file"] for s in r["steps"] if s["tool"] == "tools/import_qonnx.py"}
        if any(s["tool"] == "tools/ftl_deepfake_submodels.py" for s in r["steps"]):
            built |= {"model_features.bitino", "model_network.bitino"}
        assert built == {f for f in route["files"] if not f.startswith("model_int8")}, mid
        assert set(canon) == {f for f, s in route["files"].items() if s.get("canonical") and not f.startswith("model_int8")}
        for q in r["int8"]:
            a = q["args"]
            assert a[a.index("--out") + 1] == "{work}/int8/" + q["file"]           # the quantizer output stays in work
            w, imp = q["qonnx_steps"]
            assert w["tool"] == "tools/export_int8_qonnx.py" and "--source" in w["args"]
            assert w["args"][w["args"].index("--source") + 1] == "{work}/int8/" + q["file"]
            assert imp["tool"] == "tools/import_qonnx.py" and not (LAYOUT_FLAGS | {"--no-fold"}) & set(imp["args"])
            assert q["file"] in r["expected"]


# ------------------------------------------------------------------------------------------------ fetch() mechanics
def tiny_model(path: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    g = Graph()
    x = g.add_input("input", [1, 4])
    y = g.add_node("Dense", [x, g.add_constant("w", rng.normal(size=(3, 4))), g.add_constant("b", rng.normal(size=3))])
    g.set_outputs(y)
    bfformat.write(g, path)
    return path


def script(d, name: str, body: str) -> str:
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write("import json, os, shutil, sys\n" + textwrap.dedent(body))
    return p


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A recipe whose steps are fake tools: the 'writer' writes model.onnx holding the path of a prebuilt .bitino,
    the 'importer' copies that file (its conversion_report.json beside it), the legacy 'converter' copies the prebuilt
    file into {out_dir}."""
    tdir = tmp_path / "tools"
    tdir.mkdir()
    pre = {k: tiny_model(str(tmp_path / f"{k}.bitino"), s) for k, s in (("float", 0), ("unfused", 1), ("int8", 2), ("other", 3))}
    writer = script(tdir, "writer.py", """
        out, src = sys.argv[sys.argv.index("--out-dir") + 1], sys.argv[sys.argv.index("--prebuilt") + 1]
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "model.onnx"), "w").write(src)
        json.dump({"writer": "fake"}, open(os.path.join(out, "export.json"), "w"))
    """)
    importer = script(tdir, "importer.py", """
        src = open(sys.argv[1]).read()
        out, name = sys.argv[sys.argv.index("--out-dir") + 1], sys.argv[sys.argv.index("--file-name") + 1]
        if "--no-fold" in sys.argv:
            src = src.replace("float.bitino", "unfused.bitino")
        os.makedirs(out, exist_ok=True)
        shutil.copyfile(src, os.path.join(out, name))
        json.dump({"file": name}, open(os.path.join(out, "conversion_report.json"), "w"))
    """)
    converter = script(tdir, "converter.py", """
        out = sys.argv[sys.argv.index("--out-dir") + 1]
        shutil.copyfile(sys.argv[sys.argv.index("--prebuilt") + 1], os.path.join(out, "model.bitino"))
    """)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"checkpoint")

    def h(p):
        return {"data_sha256": F.data_sha256(p), "graph_sha256": F.graph_sha256(p)}

    def imp(fn, canonical, extra=()):
        stem = fn[: -len(".bitino")]
        d = "{work}/import/" + stem
        return {"env": "base", "tool": importer, "role": "qonnx-import", "file": fn,
                "args": ["{out_dir}/" + canonical, "--model-id", "{model_id}", "--out-dir", d, "--file-name", fn, *extra],
                "outputs": {d + "/" + fn: fn, d + "/conversion_report.json": stem + ".import.json"}}

    rec = {"model_id": "fake-net", "registry_model_id": "fake-net", "state": "verified", "build": "qonnx", "env": "base",
           "envs": ["base"], "converter": "tools/convert_fake.py", "notes": [],
           "downloads": [{"name": "weights", "url": "https://example.invalid/w", "file": "weights.bin",
                          "sha256": F.sha256_file(str(weights))}],
           "sources": [], "eval": {"dir": "x", "preprocess": "x", "synthetic_ok": True},
           "qonnx": {"front_end": "pytorch", "files": {"model.bitino": {"role": "published", "canonical": "model.qonnx.onnx"},
                                                      "model_unfused.bitino": {"role": "derived", "canonical": "model.qonnx.onnx",
                                                                                "derivation": "--no-fold"}}},
           "steps": [{"env": "base", "tool": writer, "role": "qonnx-writer", "file": "model.bitino",
                      "args": ["--weights", "{weights}", "--prebuilt", pre["float"], "--out-dir", "{work}/qonnx/model"],
                      "outputs": {"{work}/qonnx/model/model.onnx": "model.qonnx.onnx", "{work}/qonnx/model/export.json": "model.qonnx.json"}},
                     imp("model.bitino", "model.qonnx.onnx"),
                     dict(imp("model_unfused.bitino", "model.qonnx.onnx", ["--no-fold"]), role="derivation")],
           "crosscheck": {"steps": [{"env": "base", "tool": converter, "args": ["--prebuilt", pre["float"], "--out-dir", "{out_dir}"]}]},
           "expected": {"model.bitino": h(pre["float"]), "model_unfused.bitino": h(pre["unfused"]),
                        "model_int8_full.bitino": h(pre["int8"])},
           "int8": [{"file": "model_int8_full.bitino",
                     "args": ["--model", "{out_dir}/model.bitino", "--calib-dir", "{calib_dir}", "--out", "{work}/int8/model_int8_full.bitino",
                              "--prebuilt", pre["int8"]],
                     "qonnx_steps": [{"env": "base", "tool": writer, "role": "qonnx-writer", "file": "model_int8_full.bitino",
                                      "args": ["--prebuilt", "{work}/int8/model_int8_full.bitino", "--out-dir", "{work}/qonnx/model_int8_full"],
                                      "outputs": {"{work}/qonnx/model_int8_full/model.onnx": "model_int8_full.qonnx.onnx"}},
                                     imp("model_int8_full.bitino", "model_int8_full.qonnx.onnx")]}]}
    monkeypatch.setenv("BITINO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(zoo, "recipes", lambda: {"fake-net": rec})
    monkeypatch.setattr(zoo, "source_checkout", lambda: True)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    return {"rec": rec, "pre": pre, "weights": str(weights), "eval": str(eval_dir), "tmp": tmp_path, "writer": writer}


def run_fetch(fake, **kw):
    return F.fetch("fake-net", files={"weights": fake["weights"]}, eval_dir=fake["eval"], **kw)


def test_fetch_qonnx_build_records_and_compares(fake):
    r = run_fetch(fake)
    d = zoo.model_dir("fake-net")
    for fn in ("model.bitino", "model_unfused.bitino"):
        assert r["results"][fn]["identical_data"] and r["results"][fn]["identical_graph"], fn
        assert r["results"][fn]["canonical_qonnx"] == "model.qonnx.onnx"
    assert set(os.listdir(d)) >= {"model.bitino", "model_unfused.bitino", "model.qonnx.onnx", "model.qonnx.json",
                                  "model.import.json", "model_unfused.import.json", "fetch_record.json"}
    q = r["qonnx_files"]["model.qonnx.onnx"]
    assert q["sha256"] == F.sha256_file(os.path.join(d, "model.qonnx.onnx")) == r["results"]["model.bitino"]["canonical_qonnx_sha256"]
    assert r["build"] == "qonnx" and "model_int8_full.bitino" not in r["results"]
    assert json.load(open(os.path.join(d, "fetch_record.json")))["results"] == json.loads(json.dumps(r["results"]))


def test_fetch_reports_a_differing_file(fake):
    fake["rec"]["expected"]["model_unfused.bitino"] = {"data_sha256": "0" * 64, "graph_sha256": "1" * 64}
    r = run_fetch(fake)
    assert r["results"]["model.bitino"]["identical_data"]
    assert r["results"]["model_unfused.bitino"]["identical_data"] is False
    assert r["results"]["model_unfused.bitino"]["identical_graph"] is False


def test_fetch_fails_when_a_step_writes_nothing(fake):
    run_fetch(fake)                                         # a complete earlier run: its files must not be reused
    fake["rec"]["steps"][0]["args"][fake["rec"]["steps"][0]["args"].index("--out-dir") + 1] = "{work}/elsewhere"
    with pytest.raises(F.FetchError, match="did not write"):
        run_fetch(fake)
    assert not os.path.exists(os.path.join(zoo.model_dir("fake-net"), "model.bitino"))


def test_fetch_legacy_crosscheck(fake):
    r = run_fetch(fake, legacy=True)
    c = r["crosscheck"]
    assert c["dir"] == os.path.join(zoo.model_dir("fake-net"), "legacy")
    assert c["files"]["model.bitino"]["equals_qonnx_build"] and c["files"]["model.bitino"]["identical_data"]
    assert c["agree"]


def test_fetch_int8_reimport_must_equal_quantizer_output(fake, monkeypatch):
    # tools/quantize_int8_full.py of a fake checkout: copies the prebuilt int8 file to --out
    root = fake["tmp"] / "root"
    (root / "tools").mkdir(parents=True)
    script(root / "tools", "quantize_int8_full.py", """
        a = sys.argv
        shutil.copyfile(a[a.index("--prebuilt") + 1], a[a.index("--out") + 1])
    """)
    monkeypatch.setattr(zoo, "ROOT", str(root))
    calib = fake["tmp"] / "calib"
    calib.mkdir()
    r = run_fetch(fake, int8=True, calib_dir=str(calib))
    q = r["results"]["model_int8_full.bitino"]
    assert q["reimport_equals_quantizer_output"] and q["identical_data"] and q["identical_graph"]
    assert q["canonical_qonnx"] == "model_int8_full.qonnx.onnx" and "model_int8_full.qonnx.onnx" in r["qonnx_files"]
    # a canonical file whose import is not the quantizer output: the fetch fails, the file is not left as the result
    w = fake["rec"]["int8"][0]["qonnx_steps"][0]["args"]
    w[w.index("--prebuilt") + 1] = fake["pre"]["other"]
    with pytest.raises(F.FetchError, match="differs from the quantizer output"):
        run_fetch(fake, int8=True, calib_dir=str(calib))
    assert not os.path.exists(os.path.join(zoo.model_dir("fake-net"), "model_int8_full.bitino"))
    # other calibration images: the re-import still equals the quantizer output; the published hash is only reported
    w[w.index("--prebuilt") + 1] = "{work}/int8/model_int8_full.bitino"
    qa = fake["rec"]["int8"][0]["args"]
    qa[qa.index("--prebuilt") + 1] = fake["pre"]["other"]
    q = run_fetch(fake, int8=True, calib_dir=str(calib))["results"]["model_int8_full.bitino"]
    assert q["reimport_equals_quantizer_output"] and q["identical_data"] is False


def test_validate_uses_the_legacy_reference_outputs(fake, monkeypatch, capsys):
    """The QONNX build checks identity only; `bitino validate` compares the runtime with the source framework's
    outputs, which the legacy cross-check run (fetch --legacy) writes into <model dir>/legacy/."""
    from bitino.cli import main as cli
    run_fetch(fake)
    assert cli.main(["validate", "fake-net"]) == 3
    assert "--legacy" in capsys.readouterr().err
    seen = {}

    def call(argv):
        seen["argv"] = argv
        return 0
    monkeypatch.setattr(cli.subprocess, "call", call)
    monkeypatch.setattr(cli, "bench_or_die", lambda b: "bench")
    ld = os.path.join(zoo.model_dir("fake-net"), "legacy")
    os.makedirs(ld)
    np.zeros(3, np.float32).tofile(os.path.join(ld, "reference_logits.f32"))
    assert cli.main(["validate", "fake-net"]) == 0
    assert seen["argv"][seen["argv"].index("--artifacts") + 1] == ld
