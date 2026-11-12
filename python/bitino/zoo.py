"""Model-zoo data access: registry, recipes, results, licence qualification, binary discovery.

Everything is plain JSON in the source tree (``registry/``, ``results/``); nothing here imports numpy or a
framework, so ``bitino list`` works in a bare Python install. An installed wheel carries a copy of that data in
``bitino/_data/`` (installed by CMakeLists.txt); the conversion tools (``tools/``) and the reference
environments of the fetch recipes exist only in a source checkout (``source_checkout()``).
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
MODEL_EXT = ".bitino"
BENCH_NAME = "bitino_bench"
REPO_URL = "https://github.com/fedeloper/bitino"


def source_checkout() -> bool:
    """True when running from a clone of the repository (converters and fetch recipes available)."""
    return os.path.isfile(os.path.join(ROOT, "registry", "models.json")) and os.path.isdir(os.path.join(ROOT, "tools"))


def data_root() -> str:
    """Directory holding registry/ and results/: the source checkout, else the copy inside the installed package."""
    return ROOT if os.path.isfile(os.path.join(ROOT, "registry", "models.json")) else PKG_DATA


def _path(rel: str) -> str:
    p = os.path.join(data_root(), rel)
    if not os.path.exists(p) and data_root() == ROOT:
        alt = os.path.join(ROOT, "public", rel)   # development tree: public-only files (release manifest)
        if os.path.exists(alt):
            return alt
    return p


def _load(rel: str) -> dict:
    p = _path(rel)
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
    for p in sorted(glob.glob(os.path.join(data_root(), "results", "models", "*.json"))):
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


VARIANT_FILES = {"float": "model", "int8": "model_int8_full", "int8_log2": "model_int8_full_log2"}


def fetched_file(model_id: str, variant: str = "float") -> str:
    if variant not in VARIANT_FILES:
        raise ValueError(f"unknown variant {variant!r} (use {', '.join(VARIANT_FILES)})")
    return os.path.join(model_dir(model_id), VARIANT_FILES[variant] + MODEL_EXT)


def resolve_model_file(spec: str, variant: str = "float") -> str:
    """A model id (fetched into BITINO_HOME) or a path to a model file."""
    if os.path.isfile(spec):
        return spec
    p = fetched_file(spec, variant)
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"{spec}: not a file and not fetched ({p} missing); run `bitino fetch {spec}` first")


def release_entry(model_id: str) -> tuple[dict | None, str]:
    """(release manifest entry, reason) for a model whose converted files may be downloaded from the GitHub release.

    Both conditions are required: the release manifest lists the model, and its registry entry passes
    weights_licence_decision() (the rule of `bitino list --release`)."""
    man = _load(os.path.join("registry", "release_models.json"))
    ent = (man.get("models") or {}).get(model_id)
    ok, why = weights_licence_decision(registry_entry(model_id) or {})
    if not ok:
        return None, why
    if not ent:
        return None, "not in the release manifest (registry/release_models.json)"
    return ent, why


def preprocess_key(model_id: str) -> str | None:
    """The input preprocessing a zoo model was validated with (registry/recipes.json eval.preprocess)."""
    return ((recipes().get(model_id) or {}).get("eval") or {}).get("preprocess")
