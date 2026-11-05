#!/usr/bin/env python3
"""Derive a model variant from an existing .bitino by applying conversion-time IR passes (python/bitino/passes.py).

No framework or checkpoint is needed: the pass rewrites the converted graph, and the result is the same graph the
converter produces with the corresponding option (e.g. tools/convert_bivit.py / tools/convert_bibert.py
--drop-gelu-before-sign). The manifest records the pass, its rewrite count and the SHA-256 of the source file.

  python3 tools/derive_variant.py --pass drop_gelu_before_sign \
      --in artifacts/bi-vit-deit-tiny/model.bitino --out artifacts/bi-vit-deit-tiny/no_gelu/model.bitino
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import format as bfformat, passes  # noqa: E402

PASSES = {"drop_gelu_before_sign": passes.drop_gelu_before_sign}
ZERO_BELOW = {"float64-gelu": passes.F64_GELU_NONZERO_FROM}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="passes", action="append", required=True, choices=sorted(PASSES))
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", help="manifest 'variant' suffix (default: the pass names)")
    ap.add_argument("--allow-noop", action="store_true", help="write the file even if no pass rewrote anything")
    ap.add_argument("--keep-zero-below", metavar="T",
                    help="drop_gelu_before_sign: keep the GELU's zero set {x < T} with an exact mask instead of dropping it "
                         "('float64-gelu' = the IR's double GELU cut-off: bit-identical model)")
    a = ap.parse_args()
    g = bfformat.read(a.src)
    kw = {}
    if a.keep_zero_below is not None:
        kw["keep_zero_below"] = ZERO_BELOW.get(a.keep_zero_below) or float(a.keep_zero_below)
    applied = {p: PASSES[p](g, **kw) for p in a.passes}
    if kw:
        applied["keep_zero_below"] = kw["keep_zero_below"]
    applied["prune_unused_constants"] = passes.prune_unused_constants(g)
    if not any(applied[p] for p in a.passes) and not a.allow_noop:
        print(f"{a.src}: no rewrite ({applied}); nothing written", file=sys.stderr)
        return 1
    m = copy.deepcopy(g.manifest)
    m["variant"] = f"{m.get('variant', '')}{', ' if m.get('variant') else ''}{a.variant or '+'.join(a.passes)}"
    conv = m.setdefault("conversion", {})
    conv.setdefault("derived", []).append({"tool": "tools/derive_variant.py", "passes": applied,
                                           "source_file": os.path.basename(a.src), "source_sha256": sha256_file(a.src),
                                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    g.manifest = m
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sha = bfformat.write(g, a.out)
    print(f"{a.out}: {applied}, {len(g.nodes)} nodes, sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
