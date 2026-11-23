#!/usr/bin/env python3
"""THE build path of Bitino's networks (QONNX step 6, docs/qonnx-standard.md, docs/qonnx-step6.md):

    source checkpoint --(writer of the network's front end)--> canonical QONNX file --(bitino.qonnx_import)--> .bitino
                                                                                     \\--> derived files

Every published file of every recipe (registry/recipes.json; the float files model.bitino / model_sign_zero.bitino
and the int8_full files) is built from a canonical QONNX file of the Bitino profile (docs/qonnx-profile.md) written by
the network's front-end writer; the importer runs with NO import argument (the file determines its .bitino: steps
3-5 always did, steps 1 / 1b / 2 since step 6). The derived files are re-derived from the QONNX route too:

  *_unfused.bitino      pattern-mapped canonical file: the same file imported without the fold passes
                         (tools/import_qonnx.py --no-fold), i.e. the importer's graph before passes.optimize; group-mapped
                         canonical files hold the post-pass graph (no pass runs after a group import), so their unfused
                         file has a canonical file of its own, written by the same writer from the converter's graph
                         before its fold passes (tools/export_groups_qonnx.py --file *_unfused.bitino)
  model_features / model_network.bitino   the two stages of a "Faster Than Lies" detector, split from the
                         QONNX-built model.bitino by tools/ftl_deepfake_submodels.split (the derivation tool)

Which writer, environment and arguments produce each file is recorded per recipe in registry/qonnx_routes.json
(written by --write-routes from the recipes, the published file list and the writers' own tables).

Front ends (writer, stage-1 environment; stage 2 / the importer run in .cache/venv-qonnx):
  pytorch (steps 1, 1b)   tools/export_torch_qonnx.py   .cache/venv-qonnx2   torch.onnx.export + profile rewrite
  keras (step 2)          tools/export_keras_qonnx.py   .cache/venv-larq     direct writer from the Keras model
  groups (step 3)         tools/export_groups_qonnx.py  the family's conda env  one ONNX group per IR node
  ir (step 4)             tools/export_ir_qonnx.py      the front end's venv  direct writer from the source parse
  int8 (step 5)           tools/export_int8_qonnx.py    .cache/venv-qonnx    the published int8_full graph (hash-checked)

Identity bar (user decision 2026-11-20): data SHA-256 + graph SHA-256 of the published file (artifacts/<id>/<file>;
for the files listed in registry/recipes.json -> expected also the recipe's hashes), names byte for byte; the manifest
is not compared. Nothing under artifacts/ is written. Canonical files and rebuilt files are deleted after each row
(their SHA-256 are kept in the summary) unless --keep-files.

  scripts/capped.sh 5G nice -n 10 .cache/venv-qonnx/bin/python -I tools/build_from_qonnx.py --model-id ID [--file F]
      | --all [--shard I/N] [--front-end F] [--cache-dir DIR] [--artifacts-dir DIR] [--work DIR] [--out JSON]
  .cache/venv-qonnx/bin/python -I tools/build_from_qonnx.py --write-routes        # regenerate registry/qonnx_routes.json
  .cache/venv-qonnx/bin/python -I tools/build_from_qonnx.py --list [--model-id ID]  # print the routes
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ROUTES = os.path.join(ROOT, "registry", "qonnx_routes.json")
FLOAT_FILES = ("model.bitino", "model_sign_zero.bitino")
INT8_FILES = ("model_int8_full.bitino", "model_int8_full_log2.bitino")
UNFUSED = {"model_unfused.bitino": "model.bitino", "model_sign_zero_unfused.bitino": "model_sign_zero.bitino"}
SUBMODELS = {"model_features.bitino": "features", "model_network.bitino": "network"}
QONNX_PY = "{cache}/venv-qonnx/bin/python"


# ================================================================================================ routes
def _recipes() -> dict:
    return json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]


def front_end_of(mid: str, rec: dict) -> str:
    import export_groups_qonnx as egq
    import export_ir_qonnx as eiq
    import export_keras_qonnx as ekq
    conv = rec["converter"]
    if conv in egq.CONVERTERS:
        return "groups"
    if conv in eiq.FAMILIES:
        return "ir"
    if conv in ekq.CONVERTERS:
        return "keras"
    import qonnx_step1_acceptance as s1
    import qonnx_step1b_acceptance as s1b
    if mid in s1.NETWORKS or mid in s1b.NETWORKS:
        return "pytorch"
    raise SystemExit(f"{mid}: no QONNX front end for converter {conv}")


def build_routes(artifacts: str) -> dict:
    """Per recipe: the front end and, per published / derived file, the writer that produces its canonical QONNX file
    (or the derivation). Paths are templates: {cache} = the checkout's .cache, {weights} = registry/models.json
    checkpoint.local_cache_path under it, {eval_dir} = <cache>/data/<recipe eval dir>."""
    import export_groups_qonnx as egq
    import export_ir_qonnx as eiq
    import export_unfused_qonnx as euq
    import qonnx_step1_acceptance as s1
    import qonnx_step1b_acceptance as s1b
    import qonnx_step2_acceptance as s2
    recs = _recipes()
    out = {}
    for mid, rec in sorted(recs.items()):
        fe = front_end_of(mid, rec)
        present = {os.path.basename(p) for p in glob.glob(os.path.join(artifacts, mid, "model*.bitino"))
                   if os.path.basename(p).count(".") == 1}
        files = {}
        for fn in FLOAT_FILES:
            if fn not in present:
                continue
            if fe == "pytorch":
                if mid in s1.NETWORKS:
                    repo = s1.repo_of(mid)
                    chunk = s1.CHUNK.get(mid)
                else:
                    repo = s1b.NETWORKS[mid][1]
                    chunk = s1b.CHUNK.get(mid)
                w = {"writer": "tools/export_torch_qonnx.py", "env": "{cache}/venv-qonnx2/bin/python",
                     "args": ["--model-id", mid, "--repo", "{weights_dir}" if repo == "@checkpoint" else "{cache}/repos/" + repo,
                              "--weights", "{weights}", "--eval-dir", "{eval_dir}",
                              "--binarizer", "ge" if fn == "model.bitino" else "source", "--no-source-logits"],
                     "mapping": "patterns"}
                if chunk:
                    w["note"] = f"acceptance runs use --chunk {chunk} for the source logits (not computed here)"
            elif fe == "keras":
                w = {"writer": "tools/export_keras_qonnx.py", "env": "{cache}/venv-larq/bin/python", "stage2_env": QONNX_PY,
                     "args": ["--model-id", mid, "--weights", "{weights}", "--eval-dir", "{eval_dir}", "--cache-dir", "{cache}",
                              "--qonnx-python", QONNX_PY] + (["--repo", "{cache}/repos/" + s2.REPOS[mid]] if mid in s2.REPOS else []),
                     "mapping": "patterns"}
            elif fe == "groups":
                fam = egq.family_of(mid)
                w = {"writer": "tools/export_groups_qonnx.py", "env": egq.ENVS[fam], "stage2_env": QONNX_PY,
                     "args": ["--model-id", mid, "--file", fn, "--cache-dir", "{cache}", "--artifacts-dir", "{artifacts}",
                              "--qonnx-python", QONNX_PY], "mapping": "groups",
                     "note": "stage 1 rebuilds the deployed graph from the checkpoint with the legacy adapter and requires "
                             "it to equal the published file before writing"}
            else:   # ir
                fam = eiq.FAMILIES[rec["converter"]]
                w = {"writer": "tools/export_ir_qonnx.py", "env": "{cache}/" + eiq.VENVS[fam] + "/bin/python", "stage2_env": QONNX_PY,
                     "args": ["--model-id", mid, "--eval-dir", "{eval_dir}", "--cache-dir", "{cache}", "--qonnx-python", QONNX_PY,
                              "--threads", "4", "--no-source-logits", "--only-file", fn], "mapping": "patterns",
                     "family": fam}
            files[fn] = {"role": "published", "canonical": w}
        for fn in INT8_FILES:
            if fn in rec.get("expected", {}):
                files[fn] = {"role": "published", "canonical": {
                    "writer": "tools/export_int8_qonnx.py", "env": QONNX_PY,
                    "args": ["--model-id", mid, "--file", fn, "--artifacts-dir", "{artifacts}", "--cache-dir", "{cache}"],
                    "mapping": "groups", "source": "the published int8_full graph, data + graph SHA-256 checked against the "
                                                   "recipe (the quantizer's calibration images are not redistributed)"}}
        for fn, parent in UNFUSED.items():
            if fn not in present:
                continue
            if euq.own_route(mid, rec["converter"]) is not None:
                kind, env, why = euq.own_route(mid, rec["converter"])
                pargs = files[parent]["canonical"]["args"]
                repo = ["--repo", pargs[pargs.index("--repo") + 1]] if "--repo" in pargs else []
                files[fn] = {"role": "derived", "canonical": {
                    "writer": "tools/export_unfused_qonnx.py", "env": env, "stage2_env": QONNX_PY,
                    "args": ["--model-id", mid, "--file", fn, "--cache-dir", "{cache}", "--artifacts-dir", "{artifacts}",
                             "--qonnx-python", QONNX_PY] + repo, "mapping": "groups",
                    "note": "stage 1 rebuilds the converter's graph before its fold passes from the checkpoint with the legacy "
                            "adapter and requires it to equal the published derived file before writing"},
                    "derivation": f"own canonical file (the importer's pre-pass graph of {parent}'s canonical file has another "
                                  f"layout: {why})"}
            elif fe == "groups":
                w = dict(files[parent]["canonical"])
                w["args"] = [fn if a == parent else a for a in w["args"]]
                files[fn] = {"role": "derived", "canonical": w,
                             "derivation": f"own canonical file: the converter's graph before its fold passes (a group-mapped "
                                           f"canonical file holds the post-pass graph of {parent})"}
            else:
                files[fn] = {"role": "derived", "derived_from": parent,
                             "derivation": "import the canonical file of the parent without the fold passes "
                                           "(tools/import_qonnx.py --no-fold)"}
        for fn, stage in SUBMODELS.items():
            if fn in present:
                files[fn] = {"role": "derived", "derived_from": "model.bitino",
                             "derivation": f"tools/ftl_deepfake_submodels.split of the QONNX-built model.bitino ({stage} stage)"}
        unknown = sorted(present - set(files))
        if unknown:
            raise SystemExit(f"{mid}: published files without a route: {unknown}")
        out[mid] = {"front_end": fe, "legacy_converter": rec["converter"],
                    "legacy_role": "cross-check only (tools/qonnx_legacy_crosscheck.py); not the build path",
                    "files": files}
    return out


def write_routes(artifacts: str) -> dict:
    routes = build_routes(artifacts)
    n_pub = sum(1 for r in routes.values() for f in r["files"].values() if f["role"] == "published")
    n_der = sum(1 for r in routes.values() for f in r["files"].values() if f["role"] == "derived")
    doc = {"schema": "bitino.qonnx_routes.v1", "generator": "tools/build_from_qonnx.py --write-routes",
           "date": datetime.date.today().isoformat(),
           "note": "per recipe: the writer of each published file's canonical QONNX file (docs/qonnx-standard.md) and "
                   "the derivation of each derived file; {cache} / {artifacts} / {weights} / {weights_dir} / {eval_dir} "
                   "are filled in by tools/build_from_qonnx.py",
           "counts": {"networks": len(routes), "published_files": n_pub, "derived_files": n_der,
                      "front_ends": {fe: sum(1 for r in routes.values() if r["front_end"] == fe)
                                     for fe in ("pytorch", "keras", "groups", "ir")}},
           "routes": routes}
    json.dump(doc, open(ROUTES, "w"), indent=1)
    return doc


# ================================================================================================ build
def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Ctx:
    def __init__(self, args):
        self.cache = os.path.abspath(args.cache_dir)
        self.artifacts = os.path.abspath(args.artifacts_dir)
        self.work = os.path.abspath(args.work or os.path.join(self.cache, "qonnx_step6"))
        self.keep = args.keep_files
        self.models = {m["model_id"]: m for m in json.load(open(os.path.join(ROOT, "registry", "models.json")))["models"]}
        self.recipes = _recipes()
        self.qpy = QONNX_PY.format(cache=self.cache)
        self.threads = args.threads

    def fill(self, s: str, mid: str) -> str:
        lc = self.models[self.recipes[mid].get("registry_model_id", mid)]["checkpoint"]["local_cache_path"]
        weights = os.path.join(self.cache, lc[len(".cache/"):]) if lc.startswith(".cache/") else os.path.join(ROOT, lc)
        ev = os.path.join(self.cache, "data", self.recipes[mid]["eval"]["dir"])
        return (s.replace("{cache}", self.cache).replace("{artifacts}", self.artifacts).replace("{weights_dir}", os.path.dirname(weights))
                .replace("{weights}", weights).replace("{eval_dir}", ev))

    def env(self) -> dict:
        t = str(self.threads)
        return dict(os.environ, OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t, TF_CPP_MIN_LOG_LEVEL="2",
                    TF_NUM_INTRAOP_THREADS=t, TF_NUM_INTEROP_THREADS="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")


def run(cmd: list, log: str, env: dict) -> int:
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        # cwd: the log directory (some front ends write files into the working directory, e.g. NNCF graph dumps)
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(os.path.abspath(log))).returncode


def published_hashes(ctx: Ctx, mid: str, fn: str) -> dict:
    from bitino.fetch import data_sha256, graph_sha256
    p = os.path.join(ctx.artifacts, mid, fn)
    out = {"data_sha256": data_sha256(p), "graph_sha256": graph_sha256(p)}
    e = ctx.recipes[mid].get("expected", {}).get(fn)
    return {"published": out, "recipe_expected": ({"data_sha256": e["data_sha256"], "graph_sha256": e["graph_sha256"]} if e else None)}


def compare(ctx: Ctx, mid: str, fn: str, built: str, row: dict) -> None:
    from bitino.fetch import data_sha256, graph_sha256
    h = published_hashes(ctx, mid, fn)
    got = {"data_sha256": data_sha256(built), "graph_sha256": graph_sha256(built)}
    row["built"] = got | {"file_sha256": sha256_file(built), "bytes": os.path.getsize(built)}
    row.update(h)
    row["identical"] = {"data": got["data_sha256"] == h["published"]["data_sha256"],
                        "graph": got["graph_sha256"] == h["published"]["graph_sha256"]}
    if h["recipe_expected"]:
        row["identical"]["recipe"] = got == h["recipe_expected"]
    ok = all(row["identical"].values())
    row["status"] = "PASS" if ok else "FAIL: " + ", ".join(k for k, v in row["identical"].items() if not v) + " differ"


def write_canonical(ctx: Ctx, mid: str, fn: str, spec: dict, d: str, log: str) -> tuple[str | None, dict]:
    """Runs the writer of one canonical file; returns (model.onnx path or None, export record)."""
    w = spec["canonical"]
    args = [ctx.fill(a, mid) for a in w["args"]]
    py = os.path.expanduser(ctx.fill(w["env"], mid))
    os.makedirs(d, exist_ok=True)
    tool = os.path.join(ROOT, w["writer"])
    if w["writer"].endswith("export_int8_qonnx.py"):
        onnx_path = os.path.join(d, "model.onnx")
        rc = run([py, "-I", tool] + args + ["--out", onnx_path, "--json", os.path.join(d, "export.json")], log, ctx.env())
    elif w["writer"].endswith("export_ir_qonnx.py"):
        tmpd = os.path.join(ctx.work, "tmp")      # Theano's compile cache: no spaces in the path is not required here
        os.makedirs(tmpd, exist_ok=True)
        rc = run([py, "-I", tool] + args + ["--out-dir", d, "--tmp-dir", tmpd], log, ctx.env())
        onnx_path = os.path.join(d, fn[: -len(".bitino")], "model.onnx")
    else:
        rc = run([py, "-I", tool] + args + ["--out-dir", d], log, ctx.env())
        onnx_path = os.path.join(d, "model.onnx")
    exp_json = os.path.join(os.path.dirname(onnx_path), "export.json")
    if os.path.exists(os.path.join(d, "spec", "export.json")) and not os.path.exists(exp_json):
        exp_json = os.path.join(d, "spec", "export.json")
    rec = json.load(open(exp_json)) if os.path.exists(exp_json) else {}
    if rc or not os.path.exists(onnx_path):
        return None, {"writer_returncode": rc, "error": rec.get("error")}
    return onnx_path, {"writer_returncode": rc, "onnx_sha256": sha256_file(onnx_path), "onnx_bytes": os.path.getsize(onnx_path),
                       "profile_check": _profile_summary(rec)}


def _profile_summary(rec: dict):
    pc = rec.get("profile_check")
    if isinstance(pc, dict):
        return {k: pc.get(k) for k in ("passed", "summary")} | ({"warnings": len(pc["warnings"])} if "warnings" in pc else {})
    return pc


def import_canonical(ctx: Ctx, mid: str, onnx_path: str, d: str, fn: str, log: str, extra: list | None = None) -> str | None:
    """bitino.qonnx_import with no import argument (the file determines its .bitino); extra = a derivation flag."""
    out_dir = os.path.join(d, "import_" + fn[: -len(".bitino")])
    os.makedirs(out_dir, exist_ok=True)
    rc = run([ctx.qpy, "-I", os.path.join(ROOT, "tools", "import_qonnx.py"), onnx_path, "--model-id", mid, "--out-dir", out_dir,
              "--file-name", fn] + (extra or []), log, ctx.env())
    p = os.path.join(out_dir, fn)
    return p if rc == 0 and os.path.exists(p) else None


def derive_submodels(ctx: Ctx, mid: str, built_model: str, d: str) -> dict:
    from bitino import format as bff
    from ftl_deepfake_submodels import split
    g = bff.read(built_model)
    gf, gn = split(g)
    out = {}
    for sub, fn in ((gf, "model_features.bitino"), (gn, "model_network.bitino")):
        p = os.path.join(d, "derived_" + fn)
        bff.write(sub, p)
        out[fn] = p
    return out


def build_model(ctx: Ctx, mid: str, route: dict, only_files: list | None, log_dir: str) -> dict:
    rows = {}
    d0 = os.path.join(ctx.work, mid)
    os.makedirs(d0, exist_ok=True)
    log = os.path.join(log_dir, f"{mid}.log")
    files = route["files"]
    want = set(only_files or files)
    # a derived file needs its parent's canonical file / built file
    need = set(want) | {files[f]["derived_from"] for f in want if files[f].get("derived_from")}
    order = [f for f in files if files[f].get("canonical") and f in need]
    for fn in order:
        spec = files[fn]
        t0 = time.time()
        d = os.path.join(d0, fn[: -len(".bitino")])
        row = {"model_id": mid, "file": fn, "role": spec["role"], "front_end": route["front_end"],
               "writer": spec["canonical"]["writer"], "mapping": spec["canonical"].get("mapping")}
        onnx_path, wrec = write_canonical(ctx, mid, fn, spec, d, log)
        row["canonical"] = wrec
        children = [f for f in files if files[f].get("derived_from") == fn and f in want]
        if onnx_path is None:
            row["status"] = f"FAIL: writer failed (rc {wrec['writer_returncode']}) {wrec.get('error') or ''}".strip()
            for c in children:
                rows[f"{mid}/{c}"] = {"model_id": mid, "file": c, "role": "derived", "derived_from": fn,
                                      "status": "FAIL: parent canonical file not written"}
            if fn in want:
                rows[f"{mid}/{fn}"] = row | {"seconds": round(time.time() - t0, 1)}
            shutil.rmtree(d, ignore_errors=True) if not ctx.keep else None
            continue
        built = import_canonical(ctx, mid, onnx_path, d, fn, log)
        if built is None:
            row["status"] = "FAIL: import failed"
        else:
            compare(ctx, mid, fn, built, row)
        row["seconds"] = round(time.time() - t0, 1)
        if fn in want:
            rows[f"{mid}/{fn}"] = row
        print(f"[{mid}/{fn}] {row['status']} ({row['seconds']} s)", flush=True)
        for c in children:
            t1 = time.time()
            cs = files[c]
            crow = {"model_id": mid, "file": c, "role": "derived", "derived_from": fn, "derivation": cs["derivation"],
                    "front_end": route["front_end"], "parent_canonical_sha256": wrec.get("onnx_sha256")}
            if c in UNFUSED:
                p = import_canonical(ctx, mid, onnx_path, d, c, log, ["--no-fold"])
            elif c in SUBMODELS:
                p = derive_submodels(ctx, mid, built, d)[c] if built else None
            else:
                raise AssertionError(c)
            if p is None:
                crow["status"] = "FAIL: derivation failed"
            else:
                compare(ctx, mid, c, p, crow)
            crow["seconds"] = round(time.time() - t1, 1)
            rows[f"{mid}/{c}"] = crow
            print(f"[{mid}/{c}] {crow['status']} ({crow['seconds']} s)", flush=True)
        if not ctx.keep:
            shutil.rmtree(d, ignore_errors=True)
    if not ctx.keep:
        shutil.rmtree(d0, ignore_errors=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", action="append", help="recipe id (repeatable)")
    ap.add_argument("--file", action="append", help="only this published / derived file name (repeatable)")
    ap.add_argument("--all", action="store_true", help="every recipe")
    ap.add_argument("--front-end", action="append", choices=("pytorch", "keras", "groups", "ir"))
    ap.add_argument("--shard", help="I/N: the I-th of N interleaved shards of the selected recipes (1-based)")
    ap.add_argument("--skip-done", action="store_true", help="skip files already PASS in --out")
    ap.add_argument("--exclude", action="append", default=[], help="recipe id to leave out (repeatable; e.g. the largest "
                                                                    "networks, run alone with a larger memory cap)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"), help="checkout .cache (checkpoints, repos, venvs, data)")
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"), help="published files (read only)")
    ap.add_argument("--work", help="scratch directory (default <cache>/qonnx_step6)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qonnx_step6", "summary.json"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep-files", action="store_true")
    ap.add_argument("--write-routes", action="store_true", help="regenerate registry/qonnx_routes.json and exit")
    ap.add_argument("--list", action="store_true", help="print the routes and exit")
    args = ap.parse_args()
    if args.write_routes:
        doc = write_routes(os.path.abspath(args.artifacts_dir))
        print(f"wrote {ROUTES}: {doc['counts']}")
        return 0
    routes = json.load(open(ROUTES))["routes"]
    sel = sorted(routes) if args.all else (args.model_id or [])
    if not sel:
        ap.error("--model-id or --all")
    bad = [m for m in sel if m not in routes]
    if bad:
        raise SystemExit(f"no route for {bad}")
    if args.front_end:
        sel = [m for m in sel if routes[m]["front_end"] in args.front_end]
    if args.shard:                       # before --exclude: the shards do not depend on the exclusions
        i, n = (int(x) for x in args.shard.split("/"))
        sel = sel[i - 1::n]
    sel = [m for m in sel if m not in args.exclude]
    if args.list:
        for m in sel:
            print(m, json.dumps(routes[m], indent=1))
        return 0
    ctx = Ctx(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    doc = json.load(open(args.out)) if os.path.exists(args.out) else {"schema": "bitino.qonnx_step6.v1", "files": {}}
    log_dir = os.path.join(ctx.work, "logs")
    os.makedirs(log_dir, exist_ok=True)
    failed = []
    for mid in sel:
        files = [f for f in routes[mid]["files"] if not args.file or f in args.file]
        if args.skip_done:
            files = [f for f in files if not str(doc["files"].get(f"{mid}/{f}", {}).get("status", "")).startswith("PASS")]
        if not files:
            continue
        rows = build_model(ctx, mid, routes[mid], files, log_dir)
        doc["files"].update(rows)
        failed += [k for k, r in rows.items() if not str(r.get("status", "")).startswith("PASS")]
        doc["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json.dump(doc, open(args.out, "w"), indent=1)
    print(f"done: {len(sel)} recipe(s); failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
