"""Model-zoo data and CLI: registry / recipes / results consistency, licence rule, list / info / fetch --dry-run.

No network, no framework: only the JSON data and the command-line entry points.
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import zoo  # noqa: E402
from bitino.cli import main as cli  # noqa: E402

RECIPES = zoo.recipe_doc()
pytestmark = pytest.mark.skipif(not RECIPES, reason="registry/recipes.json not generated")


def run_cli(*argv) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(list(argv))
    assert rc == 0, argv
    return buf.getvalue()


def test_recipes_schema():
    assert RECIPES["schema"].endswith(".recipes.v1")
    for mid, r in RECIPES["recipes"].items():
        assert r["model_id"] == mid
        assert r["state"] in ("verified", "needs-manual-step", "broken", "reconstructed")
        assert r["env"] in RECIPES["envs"], (mid, r["env"])
        assert all(s["env"] in RECIPES["envs"] for s in r["steps"]), mid
        assert r["steps"] and all(os.path.exists(os.path.join(ROOT, s["tool"])) for s in r["steps"]), mid
        for d in r["downloads"]:
            assert d.get("url"), mid
            assert d.get("sha256"), (mid, d.get("name"))   # every download is checked
            assert " " not in d["url"], (mid, d["url"])
        exp = r["expected"].get("model" + zoo.MODEL_EXT)
        assert exp and len(exp["data_sha256"]) == 64 and len(exp["graph_sha256"]) == 64, mid
        if r["state"] == "verified":
            assert r.get("verified", {}).get("date"), mid
        if r["state"] == "broken":
            assert r.get("broken", {}).get("reason"), mid
        if r["state"] == "needs-manual-step":
            assert r.get("manual_step", {}).get("instructions"), mid


def test_results_exist_for_every_recipe():
    for mid in RECIPES["recipes"]:
        res = zoo.results(mid)
        assert res.get("model_id") == mid
        for dev, runs in res.get("speed", {}).items():
            for run in runs:
                assert run["validity"] in ("clean", "vendor-policy", "thermal-throttled", "throttled", "unknown-caps",
                                           "superseded-runtime"), (mid, dev)
                assert run["median_ms"] and run["median_ms"] > 0


@pytest.mark.parametrize("weights,ok", [
    ("MIT (HF card)", True),
    ("Apache-2.0 (model.yml license field points to the OMZ LICENSE)", True),
    ("BSD-3-Clause", True),
    ("No explicit weights licence (inherits repo licence? unverified)", False),
    ("MIT by co-location in the same directory (no separate weights statement)", False),
    ("BSD-3-Clause by inclusion in the repo tree (no separate weights licence)", False),
    ("Not stated separately. Presumably Apache-2.0 (unverified).", False),
    ("non-commercial research only", False),
    ("unspecified", False),
    ("", False),
])
def test_weights_licence_rule(weights, ok):
    assert zoo.weights_licence_decision({"license": {"weights": weights}})[0] is ok


def test_list_and_info():
    out = run_cli("list", "--converted", "--json")
    rows = json.loads(out)
    assert len(rows) == len(RECIPES["recipes"])
    rel = json.loads(run_cli("list", "--release", "--json"))
    assert rel and all(r["release"] for r in rel)
    mid = sorted(RECIPES["recipes"])[0]
    info = json.loads(run_cli("info", mid))
    assert info["model_id"] == mid and info["recipe"]["model_id"] == mid


def test_fetch_dry_run():
    mid = sorted(RECIPES["recipes"])[0]
    out = run_cli("fetch", mid, "--dry-run")
    assert json.loads(out)["model_id"] == mid


def test_fetch_drive_links():
    from bitino import fetch as F
    fid = "1CJ0XOEhoHuNe-tDYJaAOd1j4YyNXuyas"
    for u in (f"https://drive.google.com/file/d/{fid}/view?usp=sharing", f"https://drive.google.com/uc?export=download&id={fid}",
              f"https://drive.google.com/uc?id={fid}"):
        assert F.drive_file_id(u) == fid
        assert F.direct_url(u) == f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    assert F.drive_file_id("https://huggingface.co/x/y/resolve/abc/m.h5?id=" + fid) is None
    page = ('<html><form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
            f'<input type="hidden" name="id" value="{fid}"><input type="hidden" name="export" value="download">'
            '<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="u-1"></form></html>')
    nxt = F._drive_form(page)
    assert nxt.startswith("https://drive.usercontent.google.com/download?") and f"id={fid}" in nxt and "uuid=u-1" in nxt


def test_synthetic_eval_inputs(tmp_path):
    np = pytest.importorskip("numpy")
    from bitino import fetch as F
    rec = {"eval": {"dir": "sst2/eval128", "preprocess": "bibert", "input_shape": [1, 64]}}
    d = F.synthetic_eval(rec, str(tmp_path), 4)
    ids = np.fromfile(os.path.join(d, "input_ids.i32"), np.int32).reshape(4, 64)
    assert (ids[:, 0] == 101).all() and (ids[:, -1] == 102).all()
    rec = {"eval": {"dir": "eval128_cifar10_finn", "preprocess": "cifar10_pixels", "input_shape": [1, 32, 32, 3]}}
    x = np.fromfile(os.path.join(F.synthetic_eval(rec, str(tmp_path), 4), "cifar10_pixels.f32"), np.float32)
    assert x.min() >= 0 and x.max() <= 255 and (x == np.round(x)).all()
