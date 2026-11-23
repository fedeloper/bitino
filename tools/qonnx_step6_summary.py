#!/usr/bin/env python3
"""Merges the shard results of the step-6 sweep (tools/build_from_qonnx.py --out reports/qonnx_step6/shard_*.json) into
reports/qonnx_step6/summary.json with the counts of docs/qonnx-step6.md, and prints the per-front-end table.

  python3 tools/qonnx_step6_summary.py [--shards reports/qonnx_step6/shard_*.json] [--out reports/qonnx_step6/summary.json]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="*", default=sorted(glob.glob(os.path.join(ROOT, "reports", "qonnx_step6", "shard_*.json"))))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_step6", "summary.json"))
    args = ap.parse_args()
    routes = json.load(open(os.path.join(ROOT, "registry", "qonnx_routes.json")))["routes"]
    rows = {}
    for p in args.shards:            # later shards win (re-runs)
        rows.update(json.load(open(p))["files"])
    want = {f"{m}/{f}": (r["front_end"], s["role"]) for m, r in routes.items() for f, s in r["files"].items()}
    missing = sorted(set(want) - set(rows))
    by = collections.defaultdict(lambda: collections.Counter())
    for k, (fe, role) in want.items():
        r = rows.get(k)
        st = "missing" if r is None else ("PASS" if str(r.get("status", "")).startswith("PASS") else "FAIL")
        kind = ("int8_full" if "int8_full" in k else "float") if role == "published" else "derived"
        by[(fe, kind)][st] += 1
    tot = collections.Counter()
    for c in by.values():
        tot.update(c)
    pub = {k: v for k, v in want.items() if v[1] == "published"}
    der = {k: v for k, v in want.items() if v[1] == "derived"}
    ok = lambda ks: sum(1 for k in ks if str(rows.get(k, {}).get("status", "")).startswith("PASS"))  # noqa: E731
    recipe_rows = [r for k, r in rows.items() if k in want and (r.get("identical") or {}).get("recipe") is not None]
    doc = {"schema": "bitino.qonnx_step6.v1", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "identity_bar": "data SHA-256 + graph SHA-256 of the published file (artifacts/<id>/<file>; recipe expected hashes "
                           "where listed), names byte for byte; manifest not compared",
           "entry_point": "tools/build_from_qonnx.py (routes: registry/qonnx_routes.json)",
           "counts": {"networks": len(routes), "published_files": len(pub), "published_identical": ok(pub),
                      "published_float": sum(1 for k in pub if "int8_full" not in k),
                      "published_float_identical": ok([k for k in pub if "int8_full" not in k]),
                      "published_int8_full": sum(1 for k in pub if "int8_full" in k),
                      "published_int8_full_identical": ok([k for k in pub if "int8_full" in k]),
                      "derived_files": len(der), "derived_identical": ok(der),
                      "recipe_expected_checked": len(recipe_rows),
                      "recipe_expected_identical": sum(1 for r in recipe_rows if r["identical"]["recipe"]),
                      "missing": len(missing)},
           "by_front_end": {f"{fe}/{kind}": dict(c) for (fe, kind), c in sorted(by.items())},
           "failed": sorted(k for k in want if k in rows and not str(rows[k].get("status", "")).startswith("PASS")),
           "missing": missing, "files": {k: rows[k] for k in sorted(rows) if k in want}}
    json.dump(doc, open(args.out, "w"), indent=1)
    print(json.dumps(doc["counts"], indent=1))
    print("| front end | files | identical |\n|---|---:|---:|")
    for (fe, kind), c in sorted(by.items()):
        print(f"| {fe} {kind} | {sum(c.values())} | {c['PASS']} |")
    if doc["failed"]:
        print("failed:", doc["failed"])
    return 0 if not doc["failed"] and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
