"""Shared helper for the eval_sets preprocessing scripts' --add-to-manifest mode.

Instead of writing a derived single-preprocessing eval dir, the new preprocessing entry is added to the
existing manifest.json in --out (as tools/build_eval_set.py does when it adds a second preprocessing):
the manifest there must list exactly the same samples (file, JPEG SHA-256, label) as the reference
manifest the tensors were computed from. Other entries and fields are kept unchanged. Used for the
calibration sets (e.g. .cache/data/calib512_imagenet100train), which hold all preprocessings in one dir.
"""
from __future__ import annotations

import json
import os


def add_preprocessing(out_dir: str, ref_manifest: str, name: str, entry: dict) -> str:
    man_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"ERROR: --add-to-manifest needs an existing {man_path}")
    manifest = json.load(open(man_path))
    ref = json.load(open(ref_manifest))
    key = lambda s: (s["file"], s["jpeg_sha256"], s["label"])  # noqa: E731
    if [key(s) for s in manifest["samples"]] != [key(s) for s in ref["samples"]]:
        raise SystemExit(f"ERROR: {man_path} lists different samples than {ref_manifest}; refusing to mix sets")
    manifest["preprocessing"][name] = entry
    json.dump(manifest, open(man_path, "w"), indent=1)
    return man_path
