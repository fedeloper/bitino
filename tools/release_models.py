#!/usr/bin/env python3
"""Prepares the redistributable converted models for a GitHub release.

  python3 tools/release_models.py --artifacts <dir holding artifacts/<model>/> --out ../bitino-release/models \\
      [--tag v0.1.0] [--write-registry]

Rule (bitino.zoo.weights_licence_decision): a model qualifies only if the licence recorded for its WEIGHTS is an
explicit permissive one (MIT / Apache-2.0 / BSD-2/3-Clause stated for the weights, e.g. on the model card or in a
model.yml licence field). "Inherits the repository licence", "by inclusion in the repo tree", "presumably",
"unverified", non-commercial and unspecified licences do not qualify.

For each qualifying model and each shipped file (exact float model, int8_full, int8_full_log2):
  * <model-id>[_int8_full[_log2]].bitino: the converted file with its JSON manifest strings updated to the Bitino
    names (the data section is copied byte for byte; the script checks data_sha256 before and after);
  * <model-id>.LICENSE.txt: the weights licence text with the attribution and a statement of the changes made;
then SHA256SUMS and manifest.json (which models qualified, with the licence evidence, and which did not, with the
reason). --write-registry also writes registry/release_models.json, used by `bitino fetch --prebuilt`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from bitino import zoo  # noqa: E402

FILES = ("model", "model_int8_full", "model_int8_full_log2")
DEV_EXT = ".bf" + "model"          # extension of files produced before the rename
OLD = "bit" + "forge"
# Local path prefixes recorded by the converters / quantizer in manifest strings (relative checkouts named
# <name>-<long hex id>, absolute home paths) are reduced to the repository-relative part.
SCRUB = [(re.compile(r"(?:\.\./)+[\w.-]+-[0-9a-f]{12,}/"), ""),
         (re.compile(r"/home/[^/\"\s]+/[^\"]*?/(?=artifacts/|\.cache/)"), "")]

MIT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def rewrite_strings(x):
    if isinstance(x, dict):
        return {rewrite_strings(k): rewrite_strings(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rewrite_strings(v) for v in x]
    if isinstance(x, str):
        s = x
        for rx, rep in SCRUB:
            s = rx.sub(rep, s)
        return (s.replace("Bitino", "Bitino").replace(OLD.upper(), "BITINO").replace(OLD, "bitino")
                .replace(DEV_EXT, ".bitino"))
    return x


def read(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"BFMODEL1":
        raise SystemExit(f"{path}: not a model file")
    ver, hs, jo, js, do, ds = struct.unpack("<IIQQQQ", blob[8:48])
    return json.loads(blob[jo:jo + js]), blob[do:do + ds], ver


def write(doc, data, ver, path) -> str:
    js = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()
    data_off = (64 + len(js) + 63) // 64 * 64
    hdr = b"BFMODEL1" + struct.pack("<IIQQQQ", ver, 64, 64, len(js), data_off, len(data))
    hdr += b"\0" * (64 - len(hdr))
    blob = hdr + js + b"\0" * (data_off - 64 - len(js)) + data
    with open(path, "wb") as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def licence_text(m: dict, mid: str, files: list[str]) -> tuple[str, str]:
    lic = m.get("license", {})
    w = lic.get("weights", "")
    head = (f"Converted model files: {', '.join(files)}\n"
            f"Model: {m.get('canonical_name') or mid} ({mid})\n"
            f"Original weights: {m.get('checkpoint', {}).get('url', '')}\n"
            f"Original weights SHA-256: {m.get('checkpoint', {}).get('sha256', '')}\n"
            f"Authors / publisher: {m.get('authors') or m.get('provenance') or 'see the model card'}\n"
            f"Source repository: {m.get('repo_url', '')}\n"
            f"Paper: {(m.get('paper') or {}).get('title', '')} {(m.get('paper') or {}).get('url', '')}\n"
            f"Weights licence: {w}\nLicence evidence: {lic.get('evidence', '')}\n\n"
            "Changes: the original weights were converted by Bitino (https://github.com/fedeloper/bitino) into the\n"
            "Bitino model format: binary weights bit-packed, batch normalization and scales folded into per-channel\n"
            "constants, optionally quantized to int8 (files named *_int8_full*). No retraining.\n"
            "This licence covers the weights only; it is the licence of the original authors, not Bitino's.\n\n")
    if w.upper().startswith("MIT"):
        return "MIT", head + MIT.replace("MIT License\n", "MIT License\n\n(copyright: the authors of the original weights, "
                                         "as published on the model card above; no copyright line is given there)\n")
    if w.lower().startswith("apache"):
        with open(os.path.join(ROOT, "LICENSE")) as f:
            return "Apache-2.0", head + f.read()
    return w.split()[0], head + "See the licence text at the evidence link above.\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="v0.1.0")
    ap.add_argument("--repo", default="fedeloper/bitino")
    ap.add_argument("--write-registry", action="store_true")
    a = ap.parse_args()
    adir = a.artifacts if os.path.basename(os.path.normpath(a.artifacts)) == "artifacts" else os.path.join(a.artifacts, "artifacts")
    os.makedirs(a.out, exist_ok=True)
    recipes = zoo.recipes()
    reg = {m["model_id"]: m for m in zoo.registry()["models"]}
    qualified, excluded, sums, reg_out = [], [], [], {}
    for mid, rec in sorted(recipes.items()):
        m = reg.get(rec.get("registry_model_id", mid), {})
        ok, why = zoo.weights_licence_decision(m)
        if not ok:
            excluded.append({"model_id": mid, "weights_licence": m.get("license", {}).get("weights", ""), "reason": why})
            continue
        assets = []
        for stem in FILES:
            src = os.path.join(adir, mid, stem + DEV_EXT)
            if not os.path.exists(src):
                src = os.path.join(adir, mid, stem + ".bitino")
                if not os.path.exists(src):
                    continue
            doc, data, ver = read(src)
            want = doc.get("data_sha256")
            if hashlib.sha256(data).hexdigest() != want:
                raise SystemExit(f"{src}: data section does not match its data_sha256")
            doc["manifest"] = rewrite_strings(doc.get("manifest", {}))
            asset = mid + stem[len("model"):] + ".bitino"
            dst = os.path.join(a.out, asset)
            sha = write(doc, data, ver, dst)
            d2, data2, _ = read(dst)
            assert data2 == data and d2["data_sha256"] == want
            assets.append({"asset": asset, "file": stem + ".bitino", "sha256": sha, "bytes": os.path.getsize(dst),
                           "data_sha256": want})
            sums.append(f"{sha}  {asset}")
        if not assets:
            excluded.append({"model_id": mid, "reason": "qualifies, but no converted file found in --artifacts"})
            continue
        spdx, text = licence_text(m, mid, [x["asset"] for x in assets])
        lic_name = f"{mid}.LICENSE.txt"
        with open(os.path.join(a.out, lic_name), "w") as f:
            f.write(text)
        sums.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {lic_name}")
        attribution = f"{m.get('canonical_name') or mid}; weights by {m.get('authors') or m.get('provenance') or 'the original authors'} ({m.get('repo_url', '')})"
        qualified.append({"model_id": mid, "licence": spdx, "weights_licence": m["license"]["weights"],
                          "evidence": m["license"].get("evidence"), "decision": why, "attribution": attribution,
                          "licence_file": lic_name, "assets": assets})
        reg_out[mid] = {"licence": spdx, "attribution": attribution, "assets": assets}
    sums.sort(key=lambda s: s.split("  ")[1])
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    man = {"schema": "bitino.release_models.v1", "tag": a.tag, "repo": a.repo,
           "rule": "weights licence explicitly MIT / Apache-2.0 / BSD for the weights themselves; inherited, presumed, "
                   "unverified, non-commercial or unspecified licences do not qualify",
           "qualified": qualified, "excluded": excluded,
           "counts": {"qualified": len(qualified), "excluded": len(excluded),
                      "files": sum(len(q["assets"]) for q in qualified)}}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
        f.write("\n")
    if a.write_registry:
        with open(os.path.join(ROOT, "registry", "release_models.json"), "w") as f:
            json.dump({"schema": "bitino.release_models.v1", "tag": a.tag, "repo": a.repo, "models": reg_out}, f, indent=1)
            f.write("\n")
    print(json.dumps(man["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
