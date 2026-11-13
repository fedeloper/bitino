"""tools/accuracy_estimator.py: the fitted curves reproduce their anchors, the op-combination rule behaves as documented
(error-proportional transfer, float discount, group exclusivity, 0.5 stacking weight), and the API returns what the
architecture search needs. Uses only the committed research/accuracy_estimator/*.json (no model files)."""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import accuracy_estimator as AE  # noqa: E402


def test_points_and_macs():
    pts = AE.points()
    by = {p["id"]: p for p in pts}
    # binary MACs from the zoo graphs match the papers' BOPs
    assert by["bnext-tiny"]["gmacs"] == pytest.approx(4.82, abs=0.01)
    assert by["bnext-large"]["gmacs"] == pytest.approx(52.15, abs=0.01)
    assert by["rbonn-reactnet-resnet18-imagenet"]["gmacs"] == pytest.approx(1.68, abs=0.01)
    assert by["rbonn-reactnet-resnet18-imagenet"]["first_M"] == pytest.approx(118.0, abs=0.1)
    for p in pts:
        assert p["family"] in AE.FAMILIES and p["recipe"] in AE.RECIPES | {"other": ""}
        for k in p["ops"]:
            assert k in AE.OPS or k == "real_downsample_1x1"
    assert sum(p["fit"] for p in pts) >= 50


def test_fit_reproduces_anchors():
    # in-sample: each family curve passes near its anchors (LOO is the honest error; fit.json holds it)
    for mid, fam, g, top1 in [("bnext-small", "bnext", 10.84, 76.1), ("PokeBNN-1.0x", "pokebnn", 3.6095, 73.4),
                              ("QuickNet", "quicknet", 1.85, 63.3)]:
        e = AE.estimate({"family": fam, "gmacs": g})
        assert abs(e["estimate"] - top1) < 1.0, mid
        assert e["low"] < e["estimate"] < e["high"] and e["low90"] < e["low"]
        assert e["anchors"] and e["anchors"][0]["gmacs"] == pytest.approx(g, abs=0.01)


def test_loo_summary_present():
    m = AE.model()
    unc = m["unc"]
    assert 0.3 < unc["loo_rmse_pooled"] < 3.0
    assert unc["lofo_rmse"] > unc["loo_rmse_pooled"]   # a new family is less certain than a known one
    assert set(unc["loo_rmse_family"]) == set(m["fit"].fams)


def test_monotone_in_macs_and_recipe():
    a = AE.estimate({"family": "resnet", "gmacs": 2.0, "recipe": "kd"})["estimate"]
    b = AE.estimate({"family": "resnet", "gmacs": 8.0, "recipe": "kd"})["estimate"]
    c = AE.estimate({"family": "resnet", "gmacs": 8.0, "recipe": "scratch"})["estimate"]
    d = AE.estimate({"family": "resnet", "gmacs": 8.0, "recipe": "bnext"})["estimate"]
    assert a < b and c < b < d


def test_extrapolation_widens_interval():
    inside = AE.estimate({"family": "resnet", "gmacs": 2.5, "recipe": "kd"})
    outside = AE.estimate({"family": "resnet", "gmacs": 14.0, "recipe": "kd"})
    assert outside["sigma"] > inside["sigma"] + 0.5


def test_error_proportional_transfer_and_float_discount():
    # insta_th: +1.1 at base 60.0; at base 80 the error ratio is 20/40
    r = AE.ops_effect(["insta_th"], 80.0)
    assert r["gain"] == pytest.approx(1.1 * 0.5, abs=1e-6)
    # float-only evidence counts at FLOAT_DISCOUNT
    s = AE.ops_effect(["simam"], 70.33)
    assert s["gain"] == pytest.approx(0.98 * AE.FLOAT_DISCOUNT, abs=1e-6)
    # an op whose FCs are real in the evidence is halved again unless kept real (published-model check only)
    f = AE.ops_effect(["dynamic_relu"], 73.3)
    assert f["gain"] == pytest.approx(2.9 * AE.FLOAT_DISCOUNT * AE.BINARY_FC_FACTOR, abs=1e-6)
    assert AE.ops_effect(["dynamic_relu"], 73.3, fc_real=True)["gain"] == pytest.approx(2.9 * AE.FLOAT_DISCOUNT, abs=1e-6)


def test_combination_rule():
    # measured pair replaces its members
    r = AE.ops_effect(["insta_th", "insta_prelu"], 60.0)
    assert r["gain"] == pytest.approx(2.1, abs=1e-6)
    # same group: only the best counts
    g = AE.ops_effect(["insta_th", "dysign"], 60.0)
    assert g["gain"] == pytest.approx(1.1, abs=1e-6)
    # different groups: second at STACK_WEIGHT; costs add fully
    s = AE.ops_effect(["insta_th", "simam", "patchify4_stem"], 60.0)
    simam = 0.98 * 40 / 29.67 * AE.FLOAT_DISCOUNT
    stem = -0.4 * 40 / 26.5
    assert s["gain"] == pytest.approx(1.1 + AE.STACK_WEIGHT * simam + stem, abs=1e-3)
    # a family's standard op blocks the same group unless the added op is an upgrade measured over it
    assert AE.ops_effect(["se_binary_net"], 70.0, std_ops=["info_rcp"])["gain"] == 0.0
    assert AE.ops_effect(["insta_th"], 60.0, std_ops=["rsign"])["gain"] == pytest.approx(1.1, abs=1e-6)
    with pytest.raises(ValueError):
        AE.ops_effect(["rsign"], 60.0, std_ops=["rsign"])
    with pytest.raises(KeyError):
        AE.ops_effect(["no_such_op"], 60.0)


def test_estimate_api_for_search():
    e = AE.estimate({"family": "bnext", "gmacs": 10.0, "ops": ["binary_se_fc"]})
    assert 73.0 < e["estimate"] < 77.0 and e["rule_compliant_ops"]
    n = AE.estimate({"family": "new", "like": "reacta", "gmacs": 10.0, "recipe": "kd"})
    k = AE.estimate({"family": "reacta", "gmacs": 10.0, "recipe": "kd"})
    assert n["estimate"] == pytest.approx(k["estimate"], abs=1e-6) and n["sigma"] > k["sigma"]
    assert not AE.estimate({"family": "reacta", "gmacs": 4.8, "ops": ["real_downsample"]})["rule_compliant_ops"]
    g = AE.gmacs_for({"family": "bnext"}, 78.0)
    assert g is not None and 12 < g < 30
    assert math.isfinite(AE.estimate({"family": "new", "gmacs": 5.0})["estimate"])
