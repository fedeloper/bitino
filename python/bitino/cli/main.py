"""`bitino` command-line interface.

Model zoo:
  bitino list [--task T] [--family F] [--framework F] [--converted] [--strict] [--release] [--json]
  bitino info MODEL_ID
  bitino fetch MODEL_ID [--eval-dir D] [--python ENV=PY] [--file NAME=PATH] [--source NAME=DIR]
                          [--int8 --calib-dir D] [--dry-run]
  bitino run MODEL_ID|FILE [--input X.f32 | --random N] [--int8] [--stage S] [--threads N] [--top K]
  bitino bench MODEL_ID|FILE [--stage S] [--threads N] [--iterations I] [--warmup W] [--affinity C] [--json P]
  bitino validate MODEL_ID [--eval-dir D] [--stage S]

Registry / device (research tooling):
  bitino registry list [--family F] [--task T] [--precision P] [--license L] [--status S] [--strict] [--json]
  bitino registry inspect MODEL_ID
  bitino checkpoint verify MODEL_ID
  bitino device inspect [--adb PATH] [--json]

Exit codes: 0 success, 1 failure (e.g. hash mismatch), 2 usage error, 3 not found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

from .. import zoo

ROOT = zoo.ROOT
REGISTRY = os.path.join(ROOT, "registry", "models.json")


def load_registry() -> dict:
    with open(REGISTRY) as f:
        return json.load(f)


def find(reg: dict, model_id: str) -> dict | None:
    return next((m for m in reg["models"] if m["model_id"] == model_id), None)


# ------------------------------------------------------------------------------------------------ zoo
def zoo_rows() -> list[dict]:
    reg = load_registry()
    recs = zoo.recipes()
    by_reg = {m["model_id"]: m for m in reg["models"]}
    res = zoo.all_results()
    rows = []
    seen = set()
    for mid, rec in sorted(recs.items()):
        m = by_reg.get(rec.get("registry_model_id", mid), {})
        seen.add(rec.get("registry_model_id", mid))
        ok, why = zoo.weights_licence_decision(m)
        r = res.get(mid, {})
        head = (r.get("accuracy") or {}).get("headline") or {}
        t1 = {dev: min((x["median_ms"] for x in runs if x.get("threads") == 1 and x.get("stage") in ("int8_full", "optimized", "all")),
                       default=None)
              for dev, runs in (r.get("speed") or {}).items() if dev != "x86"}   # phones: results/devices.json
        rows.append({"model_id": mid, "task": m.get("task", ""), "family": m.get("family", ""), "framework": rec.get("framework") or "",
                     "precision": ",".join(m.get("precision", {}).get("categories", [])), "strict": m.get("strict_bnn", False),
                     "converted": True, "recipe": rec["state"], "release": ok, "licence_reason": why,
                     "metric": head.get("metric"), "reference": head.get("reference"), "runtime": head.get("runtime"),
                     "eval_set": head.get("eval_set"),
                     "s22_t1_ms": t1.get("s22"), "phone_t1_ms": {k: v for k, v in t1.items() if v is not None}})
    for m in reg["models"]:
        if m["model_id"] in seen:
            continue
        rows.append({"model_id": m["model_id"], "task": m.get("task", ""), "family": m.get("family", ""), "framework": "",
                     "precision": ",".join(m.get("precision", {}).get("categories", [])), "strict": m.get("strict_bnn", False),
                     "converted": False, "recipe": None, "release": False,
                     "licence_reason": "", "status": m.get("registry_status"),
                     "conversion": m.get("bitino", {}).get("conversion_status")})
    return rows


def cmd_list(a) -> int:
    rows = zoo_rows()
    out = []
    for r in rows:
        if a.converted and not r["converted"]:
            continue
        if a.release and not r["release"]:
            continue
        if a.strict and not r["strict"]:
            continue
        if a.task and a.task.lower() not in r["task"].lower():
            continue
        if a.family and a.family.lower() not in (r["family"] + " " + r["model_id"]).lower():
            continue
        if a.framework and a.framework.lower() not in (r["framework"] or "").lower():
            continue
        out.append(r)
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f"{'model_id':44s} {'task':28s} {'recipe':13s} {'release':7s} {'accuracy (ref -> bitino, set)':40s}")
    for r in out:
        if r["converted"]:
            acc = ""
            if isinstance(r.get("reference"), (int, float)) and isinstance(r.get("runtime"), (int, float)):
                pct = "PSNR" not in (r.get("metric") or "")
                f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.2f}")
                acc = f"{f(r['reference'])} -> {f(r['runtime'])} ({(r.get('eval_set') or '').split(' ')[0].rstrip(',')})"
            print(f"{r['model_id'][:44]:44s} {r['task'][:28]:28s} {r['recipe']:13s} {'yes' if r['release'] else '-':7s} {acc[:60]}")
        else:
            print(f"{r['model_id'][:44]:44s} {r['task'][:28]:28s} {'-':13s} {'-':7s} not converted ({r.get('conversion') or r.get('status')})")
    n_conv = sum(1 for r in out if r["converted"])
    print(f"{len(out)} models ({n_conv} converted)", file=sys.stderr)
    return 0


def cmd_info(a) -> int:
    rec = zoo.recipes().get(a.model_id)
    m = zoo.registry_entry(a.model_id) or find(load_registry(), a.model_id)
    if not rec and not m:
        print(f"unknown model id {a.model_id}", file=sys.stderr)
        return 3
    ok, why = zoo.weights_licence_decision(m or {})
    res = zoo.results(a.model_id)
    print(json.dumps({"model_id": a.model_id, "name": (m or {}).get("canonical_name"), "task": (m or {}).get("task"),
                      "paper": (m or {}).get("paper"), "repo": (m or {}).get("repo_url"),
                      "licence": (m or {}).get("license"), "release_permitted": ok, "release_reason": why,
                      "recipe": rec, "accuracy": res.get("accuracy", {}).get("headline"),
                      "instructions_M": (res.get("instructions_M") or {}).get("values")}, indent=1, ensure_ascii=False))
    return 0


def kv_list(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def cmd_fetch(a) -> int:
    from .. import fetch as F
    try:
        if a.prebuilt:
            rec = F.fetch_prebuilt(a.model_id)
        else:
            rec = F.fetch(a.model_id, eval_dir=a.eval_dir, files=kv_list(a.file), sources=kv_list(a.source),
                          python=kv_list(a.python), int8=a.int8, calib_dir=a.calib_dir, synthetic_n=a.synthetic_n,
                          dry_run=a.dry_run, threads=a.threads)
    except F.FetchError as e:
        print(f"fetch failed: {e}", file=sys.stderr)
        return 1
    if a.dry_run:
        return 0
    res = rec.get("results", {})
    print(json.dumps({"model_id": a.model_id, "dir": zoo.model_dir(a.model_id), "files": res}, indent=1))
    bad = [k for k, v in res.items() if not v.get("identical_data")]
    if bad:
        print(f"note: data section differs from the published conversion for {bad}", file=sys.stderr)
    return 0 if res else 1


def bench_or_die(explicit: str | None) -> str:
    b = zoo.find_bench(explicit)
    if not b:
        raise SystemExit(f"{zoo.BENCH_NAME} not found: build it (cmake -S . -B build/host && cmake --build build/host), "
                         f"download a release binary, or pass --bench PATH / set BITINO_BENCH")
    return b


def model_inputs(model: str) -> list[dict]:
    import struct
    with open(model, "rb") as f:
        h = f.read(64)
        jo, js = struct.unpack("<QQ", h[16:32])
        f.seek(jo)
        return json.loads(f.read(js)).get("inputs", [])


def cmd_run(a) -> int:
    import numpy as np
    model = zoo.resolve_model_file(a.model, "int8" if a.int8 else "float")
    bench = bench_or_die(a.bench)
    shape = model_inputs(model)[0]["shape"]
    per = int(np.prod(shape[1:]))
    with tempfile.TemporaryDirectory(prefix="bitino-run-") as td:
        if a.input:
            x = np.fromfile(a.input, np.float32)
            if x.size % per:
                print(f"{a.input}: {x.size} floats is not a multiple of the input size {shape[1:]}", file=sys.stderr)
                return 2
            inp = a.input
        else:
            x = np.random.default_rng(0).standard_normal([a.random] + list(shape[1:])).astype(np.float32)
            inp = os.path.join(td, "x.f32")
            x.tofile(inp)
        n = x.size // per
        outp = os.path.join(td, "y.f32")
        stage = a.stage or ("int8_full" if a.int8 else "all")
        r = subprocess.run([bench, "--model", model, "--inputs", inp, "--outputs", outp, "--stage", stage,
                            "--threads", str(a.threads), "--warmup", "0", "--iterations", "1"], capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            return 1
        y = np.fromfile(outp, np.float32).reshape(n, -1)
    if a.output:
        y.tofile(a.output)
    print(f"model {model}\ninput {n} x {shape[1:]} ({'file ' + a.input if a.input else 'random, seed 0'}), stage {stage}, "
          f"output {n} x {y.shape[1]} floats")
    if y.shape[1] > 1 and y.shape[1] <= 100000:
        for i in range(min(n, 8)):
            top = np.argsort(-y[i])[: a.top]
            print(f"  sample {i}: top-{a.top} " + ", ".join(f"{int(k)}:{y[i, k]:.4g}" for k in top))
    return 0


def cmd_bench(a) -> int:
    model = zoo.resolve_model_file(a.model, "int8" if a.int8 else "float")
    bench = bench_or_die(a.bench)
    stage = a.stage or ("int8_full" if a.int8 else "all")
    with tempfile.TemporaryDirectory(prefix="bitino-bench-") as td:
        jp = a.json or os.path.join(td, "bench.json")
        argv = [bench, "--model", model, "--stage", stage, "--threads", str(a.threads), "--warmup", str(a.warmup),
                "--iterations", str(a.iterations), "--json", jp] + (["--affinity", a.affinity] if a.affinity else [])
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            return 1
        with open(jp) as f:
            rep = json.load(f)
    lat = rep.get("latency", {})
    print(f"{os.path.basename(model)} stage {stage}, {a.threads} thread(s): median {lat.get('median_ms', float('nan')):.3f} ms, "
          f"p90 {lat.get('p90_ms', float('nan')):.3f} ms over {lat.get('n')} iterations (warmup {a.warmup})")
    print("note: timings are only meaningful on real hardware; never under an emulator", file=sys.stderr)
    return 0


def cmd_validate(a) -> int:
    rec = zoo.recipes().get(a.model_id)
    if not rec:
        print(f"no recipe for {a.model_id}", file=sys.stderr)
        return 3
    d = zoo.model_dir(a.model_id)
    if not os.path.isdir(d):
        print(f"{a.model_id} not fetched; run `bitino fetch {a.model_id}`", file=sys.stderr)
        return 3
    bench = bench_or_die(a.bench)
    eval_dir = a.eval_dir
    if not eval_dir:
        with open(os.path.join(d, "fetch_record.json")) as f:
            fr = json.load(f)
        eval_dir = next((c[c.index("--eval-dir") + 1] for c in fr["commands"] if "--eval-dir" in c), None)
    argv = [sys.executable, "-I", os.path.join(ROOT, "tools", "validate_model.py"), "--artifacts", d, "--eval-dir", eval_dir,
            "--preprocess", rec["eval"]["preprocess"], "--bench", bench, "--stage", a.stage, "--threads", str(a.threads),
            "--label", f"cli_{a.stage}"] + rec["eval"].get("validate_args", [])
    if a.stage in ("neon", "reference", "binary_scalar"):
        argv.append("--accept-exact-numpy")
    else:
        argv.append("--explain")
    return subprocess.call(argv)


# ------------------------------------------------------------------------------------------------ registry
def cmd_registry_list(a) -> int:
    reg = load_registry()
    out = []
    for m in reg["models"]:
        lic = (m["license"]["code"] + " " + m["license"]["weights"]).lower()
        if a.family and a.family.lower() not in (m["family"] + " " + m["research"]["discovered_in"]).lower():
            continue
        if a.task and a.task.lower() not in m["task"].lower():
            continue
        if a.precision and not any(a.precision.lower() in c.lower() for c in m["precision"]["categories"]):
            continue
        if a.license and a.license.lower() not in lic:
            continue
        if a.status and m["registry_status"] != a.status:
            continue
        if a.strict and not m["strict_bnn"]:
            continue
        out.append(m)
    if a.json:
        print(json.dumps([{k: m[k] for k in ("model_id", "family", "task", "registry_status", "strict_bnn", "precision", "license", "bitino")}
                          | {"size_bytes": m["checkpoint"].get("size_bytes")} for m in out], indent=1))
    else:
        for m in out:
            size = m["checkpoint"].get("size_bytes") or 0
            print(f"{m['model_id']:48s} {m['task'][:28]:28s} {','.join(m['precision']['categories'])[:24]:24s} "
                  f"{size / 1e6:8.1f}MB {m['registry_status']:20s} {m['bitino'].get('conversion_status', '')}")
        print(f"{len(out)} models", file=sys.stderr)
    return 0


def cmd_registry_inspect(a) -> int:
    m = find(load_registry(), a.model_id)
    if not m:
        print(f"unknown model id {a.model_id}", file=sys.stderr)
        return 3
    print(json.dumps(m, indent=1, ensure_ascii=False))
    return 0


def cmd_checkpoint_verify(a) -> int:
    m = find(load_registry(), a.model_id)
    if not m:
        print(f"unknown model id {a.model_id}", file=sys.stderr)
        return 3
    ck = m["checkpoint"]
    rel = ck.get("local_cache_path")
    if not rel or not ck.get("sha256"):
        print(json.dumps({"model_id": a.model_id, "ok": False, "reason": "no local cache path or recorded sha256", "url": ck.get("url")}))
        return 1
    path = os.path.join(ROOT, rel)
    if os.path.isdir(path):
        path = os.path.join(path, os.path.basename(ck["url"].split("?")[0]))
    if not os.path.isfile(path):
        print(json.dumps({"model_id": a.model_id, "ok": False, "reason": f"missing file {path}; download from {ck.get('url')}"}))
        return 1
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    ok = h.hexdigest() == ck["sha256"].lower()
    print(json.dumps({"model_id": a.model_id, "ok": ok, "path": path, "expected": ck["sha256"], "actual": h.hexdigest()}))
    return 0 if ok else 1


def cmd_device_inspect(a) -> int:
    script = os.path.join(ROOT, "tools", "collect_device_info.py")
    return subprocess.call([sys.executable, script] + (["--adb", a.adb] if a.adb else []) + (["--json"] if a.json else [])
                           + (["--out", a.out] if a.out else []))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bitino", description="Binary neural networks: model zoo, conversion, runtime.")
    sub = p.add_subparsers(dest="group", required=True)

    ls = sub.add_parser("list", help="model zoo (converted models first)")
    for f in ("task", "family", "framework"):
        ls.add_argument(f"--{f}")
    ls.add_argument("--converted", action="store_true", help="only models with a conversion recipe")
    ls.add_argument("--strict", action="store_true", help="strict BNNs only (binary weights and activations)")
    ls.add_argument("--release", action="store_true", help="only models whose weights licence permits redistribution")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(fn=cmd_list)

    inf = sub.add_parser("info", help="recipe, licence and results of one model")
    inf.add_argument("model_id")
    inf.set_defaults(fn=cmd_info)

    fe = sub.add_parser("fetch", help="download the original weights, verify, convert locally")
    fe.add_argument("model_id")
    fe.add_argument("--eval-dir", help="real eval set (default: a small synthetic one)")
    fe.add_argument("--python", action="append", metavar="ENV=PATH", help="python of a reference env, e.g. larq=/venv/bin/python")
    fe.add_argument("--file", action="append", metavar="NAME=PATH", help="use an already downloaded file (SHA-256 still checked)")
    fe.add_argument("--source", action="append", metavar="NAME=DIR", help="use a local source tree instead of downloading it")
    fe.add_argument("--int8", action="store_true", help="also quantize to int8_full with the recorded flags")
    fe.add_argument("--calib-dir", help="calibration set for --int8 (training split; not redistributed)")
    fe.add_argument("--synthetic-n", type=int, default=8)
    fe.add_argument("--threads", type=int, default=4)
    fe.add_argument("--dry-run", action="store_true", help="print the recipe only")
    fe.add_argument("--prebuilt", action="store_true",
                    help="download the converted files from the GitHub release instead (permissive weights licence only)")
    fe.set_defaults(fn=cmd_fetch)

    for name, fn, hlp in (("run", cmd_run, "run a model on an input file or random inputs"),
                          ("bench", cmd_bench, "latency on this machine")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("model", help="model id (fetched) or model file")
        q.add_argument("--int8", action="store_true", help="use the int8_full file of a fetched model")
        q.add_argument("--stage")
        q.add_argument("--threads", type=int, default=1)
        q.add_argument("--bench", help=f"path to {zoo.BENCH_NAME}")
        if name == "run":
            q.add_argument("--input", help="raw float32 file, N x input shape")
            q.add_argument("--random", type=int, default=1, help="N random inputs if --input is not given")
            q.add_argument("--output", help="write the raw float32 outputs here")
            q.add_argument("--top", type=int, default=5)
        else:
            q.add_argument("--iterations", type=int, default=50)
            q.add_argument("--warmup", type=int, default=10)
            q.add_argument("--affinity")
            q.add_argument("--json")
        q.set_defaults(fn=fn)

    va = sub.add_parser("validate", help="check the runtime against the numpy executor and the source references")
    va.add_argument("model_id")
    va.add_argument("--eval-dir")
    va.add_argument("--stage", default="neon")
    va.add_argument("--threads", type=int, default=4)
    va.add_argument("--bench")
    va.set_defaults(fn=cmd_validate)

    r = sub.add_parser("registry", help="research registry of investigated BNNs").add_subparsers(dest="cmd", required=True)
    rl = r.add_parser("list")
    for f in ("family", "task", "precision", "license", "status"):
        rl.add_argument(f"--{f}")
    rl.add_argument("--strict", action="store_true", help="strict BNNs only (binary weights and activations)")
    rl.add_argument("--json", action="store_true")
    rl.set_defaults(fn=cmd_registry_list)
    ri = r.add_parser("inspect")
    ri.add_argument("model_id")
    ri.set_defaults(fn=cmd_registry_inspect)
    c = sub.add_parser("checkpoint").add_subparsers(dest="cmd", required=True)
    cv = c.add_parser("verify")
    cv.add_argument("model_id")
    cv.set_defaults(fn=cmd_checkpoint_verify)
    d = sub.add_parser("device").add_subparsers(dest="cmd", required=True)
    di = d.add_parser("inspect")
    di.add_argument("--adb")
    di.add_argument("--json", action="store_true")
    di.add_argument("--out", help="also write the JSON to this file")
    di.set_defaults(fn=cmd_device_inspect)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
