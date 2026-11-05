"""Model-zoo data access: registry, recipes, results, licence qualification, binary discovery.

Everything is plain JSON in the source tree (``registry/``, ``results/``); nothing here imports numpy or a
framework, so ``bitino list`` works in a bare Python install.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_EXT = ".bitino"
BENCH_NAME = "bitino_bench"


def _load(rel: str) -> dict:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def registry() -> dict:
    return _load(os.path.join("registry", "models.json"))


def recipes() -> dict:
    return _load(os.path.join("registry", "recipes.json")).get("recipes", {})


def recipe_doc() -> dict:
    return _load(os.path.join("registry", "recipes.json"))


def eval_sets() -> dict:
    return _load(os.path.join("registry", "eval_sets.json")).get("eval_sets", {})


def results(model_id: str) -> dict:
    return _load(os.path.join("results", "models", f"{model_id}.json"))


def all_results() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "models", "*.json"))):
        with open(p) as f:
            d = json.load(f)
        out[d["model_id"]] = d
    return out


def home() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("BITINO_HOME", "~/.cache/bitino")))


def model_dir(model_id: str) -> str:
    return os.path.join(home(), "models", model_id)


# Weights licence qualification for redistribution of converted files (GitHub Releases).
# Only an explicit permissive licence stated for the weights themselves qualifies; inherited, co-located,
# presumed or unverified licences do not.
PERMISSIVE = re.compile(r"^\s*(MIT|Apache[- ]2\.0|BSD-[23]-Clause)\b", re.I)
DISQUALIFY = re.compile(r"inherit|unverified|presumabl|by co-location|by inclusion|not stated|unspecified|no explicit|"
                        r"none|non-commercial|NC\b|research only", re.I)


def weights_licence_decision(model: dict) -> tuple[bool, str]:
    lic = model.get("license", {})
    w = (lic.get("weights") or "").strip()
    if not w:
        return False, "no weights licence recorded"
    if DISQUALIFY.search(w):
        return False, f"weights licence not explicit: {w}"
    m = PERMISSIVE.match(w)
    if not m:
        return False, f"weights licence not permissive or not explicit: {w}"
    return True, f"{w} (evidence: {lic.get('evidence') or 'registry'})"


def registry_entry(model_id: str) -> dict | None:
    reg = registry()
    rec = recipes().get(model_id, {})
    rid = rec.get("registry_model_id", model_id)
    return next((m for m in reg.get("models", []) if m["model_id"] == rid), None)


def find_bench(explicit: str | None = None) -> str | None:
    cands = [explicit, os.environ.get("BITINO_BENCH")]
    cands += [os.path.join(ROOT, "build", "host", BENCH_NAME), os.path.join(ROOT, "build", "release", BENCH_NAME)]
    cands += sorted(glob.glob(os.path.join(ROOT, "build", "*", BENCH_NAME)))
    cands.append(shutil.which(BENCH_NAME))
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def resolve_model_file(spec: str, variant: str = "float") -> str:
    """A model id (fetched into BITINO_HOME) or a path to a model file."""
    if os.path.isfile(spec):
        return spec
    name = {"float": "model", "int8": "model_int8_full", "int8_log2": "model_int8_full_log2"}[variant] + MODEL_EXT
    p = os.path.join(model_dir(spec), name)
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"{spec}: not a file and not fetched ({p} missing); run `bitino fetch {spec}` first")
