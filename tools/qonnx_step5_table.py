#!/usr/bin/env python3
"""Renders the per-file acceptance table of docs/qonnx-step5.md from reports/qonnx_step5/summary.json (between the
markers <!-- step5-table --> and <!-- /step5-table -->), and prints the counts.

  python3 tools/qonnx_step5_table.py [--summary reports/qonnx_step5/summary.json] [--doc docs/qonnx-step5.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def yn(v) -> str:
    return "yes" if v is True else ("no" if v is False else "-")


def row_md(key: str, r: dict) -> str:
    mid, fn = key.split("/")
    f = fn.replace(".bitino", "")
    if "error" in r and "import" not in r:
        return f"| `{mid}` | {f} | - | - | - | - | - | - | **not done**: {r['error'][:120]} |"
    ex, im, idn = r["export"], r["import"], r["identical"]
    b = r.get("bit_exact") or {}
    if b:
        n = b["samples"]
        xc = b.get("qonnx_execute_onnx_vs_ort", {})
        xs = yn(xc.get("bit_identical")) if "bit_identical" in xc else "skip"
        bits = (f"{yn(b['ort_vs_cpp_int8_full'])} ({n}) / {yn(b['ort_vs_cpp_int8_full_reference'])} ({b.get('cpp_reference_samples', n)})"
                f" / {yn(b['numpy_oracle_vs_ort']['bit_identical'])} ({b['numpy_oracle_vs_ort']['samples']}) / {xs}")
    else:
        bits = "not run"
    return (f"| `{mid}` | {f} | {r['check_qonnx']} | `{im['data_sha256'][:16]}` | `{im['graph_sha256'][:16]}` | "
            f"{yn(idn['data'])} / {yn(idn['graph'])} ({yn(idn['recipe_data'] and idn['recipe_graph'])}) | "
            f"{ex['onnx_bytes'] / 1e6:.1f} MB, {ex['onnx_nodes']} nodes, {ex['writer']['groups']} groups | {bits} | **{r['status']}** |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=os.path.join(ROOT, "reports", "qonnx_step5", "summary.json"))
    ap.add_argument("--doc", default=os.path.join(ROOT, "docs", "qonnx-step5.md"))
    a = ap.parse_args()
    s = json.load(open(a.summary))
    rows = s["rows"]
    head = ("| network | file | check-qonnx | data SHA-256 | graph SHA-256 | data / graph = published (recipe) | canonical file | "
            "bit-identical: ORT vs C++ int8_full (samples) / vs C++ int8_full_reference (samples) / numpy oracle (samples) / "
            "qonnx execute_onnx (1 sample) | status |\n|---|---|---|---|---|---|---|---|---|")
    body = "\n".join(row_md(k, rows[k]) for k in sorted(rows))
    doc = open(a.doc).read()
    i, j = doc.index("<!-- step5-table -->"), doc.index("<!-- /step5-table -->")
    doc = doc[:i] + "<!-- step5-table -->\n" + head + "\n" + body + "\n" + doc[j:]
    open(a.doc, "w").write(doc)
    n = len(rows)
    ok = sum(1 for r in rows.values() if r.get("status") == "PASS")
    ident = sum(1 for r in rows.values() if r.get("identical") and all(r["identical"].values()))
    bits = sum(1 for r in rows.values() if (r.get("bit_exact") or {}).get("all_bit_identical"))
    print(f"rows {n}, PASS {ok}, identical {ident}, bit-exact {bits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
