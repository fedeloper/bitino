#!/usr/bin/env python3
"""Promotes int8_full v2 files (docs/int8.md "int8_full v2") to the shipped model_int8_full.bitino.

  python3 tools/promote_int8_full_v2.py --table <v2 table json> --v2-root <dir holding artifacts/<m>/model_int8_full_v2.bitino
          and reports/{accuracy,int8_full_v2}> [--apply]

A model is promoted iff tools/int8_full_v2_table.py gave its (model, int8_full) pair the verdict "propose": the v2 file
is on par with the current file (top-1 >= current - 0.3 pt on the model's large eval set) and faster on every phone
that timed it. Only model_int8_full.bitino is replaced; model_int8_full_log2.bitino files are kept.
Per promoted model (--apply; without it, a dry run):
  * the current file moves to artifacts/<m>/superseded/model_int8_full.<data sha8>.bitino, the v2 file takes its place;
  * the emulated int8_full reference (artifacts/<m>/emulated/<m>_int8_full.f32[.commit]) moves to
    artifacts/<m>/emulated/superseded/ (it belongs to the old file; the suites then record the new outputs);
  * reports/int8_full_v2/build/<m>.json and the paired accuracy reports are copied from --v2-root;
  * registry/bitino_status.json int8_full_variants["model_int8_full.bitino"] is rewritten (accuracy of the v2 file
    vs the exact model, the previous file's accuracy, the promotion date); the previous text is kept under
    "model_int8_full.bitino (superseded 2026-11-13)".
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {"eval5000": "eval5000", "cifar10test10000": "cifar10-test-10000", "cifar100test10000": "cifar100-test-10000"}


def sha8(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    rows = json.load(open(a.table))
    today = datetime.date.today().isoformat()
    sp = os.path.join(ROOT, "registry", "bitino_status.json")
    status = json.load(open(sp))
    done = []
    for r in rows:
        if r.get("current") != "int8_full" or r.get("verdict") != "propose":
            continue
        m = r["model"]
        src = os.path.join(a.v2_root, "artifacts", m, "model_int8_full_v2.bitino")
        dst = os.path.join(ROOT, "artifacts", m, "model_int8_full.bitino")
        if not (os.path.exists(src) and os.path.exists(dst)):
            print(f"skip {m}: missing {src if not os.path.exists(src) else dst}", file=sys.stderr)
            continue
        full = hashlib.sha256(open(src, "rb").read()).hexdigest()
        if r.get("v2_sha") and r["v2_sha"] != full:
            print(f"skip {m}: v2 file sha {full[:12]} != table {r['v2_sha'][:12]}", file=sys.stderr)
            continue
        tag = SETS.get(r["eval"], r["eval"])
        text = (f"{tag}: {r['acc_v2']:.2f}% vs exact {r['acc_float']:.2f}%; int8_full v2 (lin8 residual code, "
                f"tools/int8_full_v2.py), promoted {today}: previous int8_full file {r['acc_current']:.2f}% "
                f"(v2 - previous {r['diff']:+.2f} pt, paired p={r['p']:.2g}, n={r['n']}), faster on every phone timed")
        done.append(m)
        print(f"{'promote' if a.apply else 'would promote'} {m}: {text}")
        if not a.apply:
            continue
        sup = os.path.join(ROOT, "artifacts", m, "superseded")
        os.makedirs(sup, exist_ok=True)
        shutil.move(dst, os.path.join(sup, f"model_int8_full.{sha8(dst)}.bitino"))
        shutil.copy2(src, dst)
        for e in glob.glob(os.path.join(ROOT, "artifacts", m, "emulated", f"{m}_int8_full.f32*")):
            os.makedirs(os.path.join(os.path.dirname(e), "superseded"), exist_ok=True)
            shutil.move(e, os.path.join(os.path.dirname(e), "superseded", os.path.basename(e)))
        for rel in [f"reports/int8_full_v2/build/{m}.json", f"reports/int8_full_v2/build/{m}.quantize.json"] + \
                [os.path.relpath(p, a.v2_root) for p in glob.glob(os.path.join(a.v2_root, "reports", "accuracy", f"{m}_int8_full*_vs_int8_full_v2_*.json"))]:
            s = os.path.join(a.v2_root, rel)
            if os.path.exists(s):
                os.makedirs(os.path.dirname(os.path.join(ROOT, rel)), exist_ok=True)
                shutil.copy2(s, os.path.join(ROOT, rel))
        v = status[m].setdefault("int8_full_variants", {})
        if "model_int8_full.bitino" in v:
            v[f"model_int8_full.bitino (superseded {today})"] = v["model_int8_full.bitino"]
        v["model_int8_full.bitino"] = text
    if a.apply:
        with open(sp, "w") as f:
            json.dump(status, f, indent=1, ensure_ascii=False)
            f.write("\n")
    print(f"{len(done)} models {'promoted' if a.apply else 'to promote'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
