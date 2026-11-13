#!/usr/bin/env python3
"""int8_full v2 files (artifacts/<m>/model_int8_full_v2.bitino; docs/int8.md "int8_full v2"): build, bit-identity check
against the numpy oracle, paired accuracy against the current int8_full files, tables.

    python3 tools/int8_full_v2.py build --models m1 m2 ...            # quantize (same calibration as the shipped files)
    python3 tools/int8_full_v2.py check --models ... [--n 128]        # numpy oracle == host C++ (reference / int8_full) on eval128
    python3 tools/int8_full_v2.py accuracy --models ...               # large eval set, current int8_full file(s) vs v2, paired

build: tools/quantize_int8_full.py with the shipped file's calibration (set, preprocessing, n, percentile; the cached
statistics .cache/int8_full_stats/<m>.stats.json) and its quantizer options (manifest.quantization.options: per-part
grids, per-channel stems, BNAS clipping, ...), then `--residual-code lin8` plus the v2 options of the model (V2_OPTS,
default `--identity-blocks all --res-pre`). The shipped files are never written. Reports: reports/int8_full_v2/build/<m>.json.
check: the v2 file on the model's eval128 set (scripts/eval_sets.sh) in the numpy int8 oracle (python/bitino/int8_ops.py)
and in build/host/bitino_bench at stages int8_full_reference and int8_full (byte comparison); the oracle's output is
kept as artifacts/<m>/v2/oracle_eval128_n<N>.f32 (its sha256 in reports/int8_full_v2/oracle/<m>_eval128_n<N>.json) for
the phone comparison.
accuracy: tools/accuracy_eval.py on the family's large eval set (scripts/int8_accuracy_all.sh), A = the current
model_int8_full.bitino (and model_int8_full_log2.bitino), B = model_int8_full_v2.bitino, both stage int8_full, host
runtime; reports/accuracy/<m>_int8_full[_log2]_vs_int8_full_v2_<tag>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, HERE)
import int8_full_suite as S  # noqa: E402

# quantizer flags reproducing a shipped file's options (manifest.quantization.options)
OPT_FLAGS = {"concat_parts": lambda v: ["--concat-parts"] + ([] if v is True else [v]),
             "pc_stem": lambda v: ["--pc-stem"] + ([] if v is True else [v]),
             "pc_slices": lambda v: ["--pc-slices", v], "bin_relu_clip": lambda v: ["--bin-relu-clip"],
             "bin_clip_far": lambda v: ["--bin-clip-far"], "pc_bin": lambda v: ["--pc-bin"], "center_split": lambda v: ["--center-split"],
             "maxpool_range": lambda v: ["--maxpool-range"], "stem_minmax": lambda v: ["--stem-minmax"],
             "bin_margin": lambda v: ["--bin-margin", v]}
DEFAULT_V2 = ["--identity-blocks", "all", "--res-pre", "--int-epilogue", "--gated", "--dense"]
V2_OPTS: dict[str, list[str]] = {}   # per-model overrides of DEFAULT_V2 (choices recorded in docs/int8.md)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def art(m, f="model_int8_full_v2.bitino"):
    return os.path.join(ROOT, "artifacts", m, f)


def shipped_manifest(m):
    from bitino import format as F
    g = F.read(art(m, "model_int8_full.bitino"))
    return g.manifest.get("quantization", {})


def build_one(m, a):
    q = shipped_manifest(m)
    cal = q["calibration"]
    cdir, key = S.calib_of(m)
    assert key == cal["preprocessing"], (m, key, cal["preprocessing"])
    fm = art(m, "model.bitino")
    if sha(fm) != q["source_model"]["sha256"]:
        return {"model": m, "status": "float model changed since the shipped int8_full file"}
    cmd = [sys.executable, os.path.join(HERE, "quantize_int8_full.py"), "--model", fm, "--calib-dir", os.path.join(ROOT, cdir),
           "--preprocess", key, "--n", str(cal["n"]), "--residual-code", "lin8"]
    st = os.path.join(ROOT, ".cache", "int8_full_stats", f"{m}.stats.json")
    if os.path.exists(st):
        cmd += ["--stats", st]
    if cal["range"].startswith("percentile"):
        cmd += ["--percentile", cal["range"].split()[1]]
    opts = q.get("options", {})
    for k, v in opts.items():
        if k in OPT_FLAGS and v:
            cmd += OPT_FLAGS[k](v)
    if opts.get("pool_grid_fix") is False:
        cmd += ["--no-pool-grid-fix"]
    v2 = a.v2_opts if a.v2_opts is not None else V2_OPTS.get(m, DEFAULT_V2)
    cmd += list(v2)
    out = a.out or art(m)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd += ["--out", out, "--report", os.path.join(ROOT, "reports", "int8_full_v2", "build", f"{m}.quantize.json")]
    os.makedirs(os.path.join(ROOT, "reports", "int8_full_v2", "build"), exist_ok=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        return {"model": m, "status": "FAILED", "stderr": r.stderr[-2000:]}
    j = json.load(open(os.path.join(ROOT, "reports", "int8_full_v2", "build", f"{m}.quantize.json")))
    rc = j.get("residual_code", {})
    res = {"model": m, "status": "built", "out": os.path.relpath(out, ROOT), "sha256": sha(out), "nodes": j["nodes"],
           "fused_blocks": rc.get("fused_blocks"), "int8_ops": j["int8_ops"], "args": cmd[cmd.index("--n"):cmd.index("--out")],
           "seconds": round(time.time() - t0, 1), "shipped_nodes": None}
    json.dump(res, open(os.path.join(ROOT, "reports", "int8_full_v2", "build", f"{m}.json"), "w"), indent=1)
    return res


def eval128(m):
    out = subprocess.run(["bash", "-c", 'source scripts/eval_sets.sh; eval_of "$0"', m], cwd=ROOT, capture_output=True,
                         text=True).stdout.split()
    edir, pp = out[0], out[1]
    man = json.load(open(os.path.join(ROOT, edir, "manifest.json")))
    e = man["preprocessing"][pp]
    return os.path.join(ROOT, edir, e["file"]), e


def check_one(m, a):
    from bitino import executor, format as F
    f = art(m)
    g = F.read(f)
    path, e = eval128(m)
    ins = g.inputs
    per = sum(int(np.prod(i["shape"])) for i in ins)
    dt = np.int32 if ins[0].get("dtype") == "i32" else np.float32
    raw = np.fromfile(path, dt)
    n_all = raw.size // per
    n = min(a.n, n_all)
    t0 = time.time()
    outs = []
    for i in range(n):
        rec = raw[i * per:(i + 1) * per]
        feeds, off = {}, 0
        for inp in ins:
            k = int(np.prod(inp["shape"]))
            feeds[inp["name"]] = rec[off:off + k].reshape(inp["shape"])
            off += k
        env = executor.run(g, feeds)
        outs.append(b"".join(np.ascontiguousarray(env[o]).tobytes() for o in g.outputs))
    ref = b"".join(outs)
    t_np = time.time() - t0
    od = os.path.join(ROOT, "reports", "int8_full_v2", "oracle")
    os.makedirs(od, exist_ok=True)
    ad = os.path.join(ROOT, "artifacts", m, "v2")   # the oracle's output bytes stay next to the file (not in git)
    os.makedirs(ad, exist_ok=True)
    op = os.path.join(ad, f"oracle_eval128_n{n}.f32")
    open(op, "wb").write(ref)
    import tempfile
    os.makedirs(os.path.join(ROOT, "build", "tmp"), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"v2chk_{m}_", dir=os.path.join(ROOT, "build", "tmp"))   # private: checks may run concurrently
    inp = os.path.join(tmp, "in.bin")
    raw[:n * per].tofile(inp)
    res = {"model": m, "n": n, "oracle_sha256": hashlib.sha256(ref).hexdigest(), "oracle_seconds": round(t_np, 1),
           "file_sha256": sha(f), "runners": {}}
    for st in a.stages:
        o = os.path.join(tmp, f"out_{st}.bin")
        p = subprocess.run([os.path.join(ROOT, a.bench), "--model", f, "--inputs", inp, "--outputs", o, "--stage", st, "--threads",
                            str(a.threads), "--warmup", "0", "--iterations", "1"], capture_output=True, text=True, cwd=ROOT)
        got = open(o, "rb").read() if p.returncode == 0 and os.path.exists(o) else None
        res["runners"][f"host:{st}"] = "IDENTICAL" if got == ref else ("DIFF" if got is not None else f"rc={p.returncode}")
    json.dump(res, open(os.path.join(od, f"{m}_eval128_n{n}.json"), "w"), indent=1)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def accuracy_one(m, a):
    edir, pp, ncls, tag = S.big_eval_of(m)
    outs = []
    for cur in ("model_int8_full.bitino", "model_int8_full_log2.bitino"):
        if not os.path.exists(art(m, cur)) or (a.only_linear and "log2" in cur):
            continue
        name = "int8_full" if "log2" not in cur else "int8_full_log2"
        js = os.path.join(ROOT, "reports", "accuracy", f"{m}_{name}_vs_int8_full_v2_{tag}.json")
        if os.path.exists(js) and not a.force:
            d = json.load(open(js))
            if d.get("files_sha256", {}).get("b") == sha(art(m)):
                outs.append(js)
                continue
        cmd = [sys.executable, os.path.join(HERE, "accuracy_eval.py"), "--eval-dir", edir, "--preprocess", pp,
               "--a", f"{os.path.relpath(art(m, cur), ROOT)}:int8_full", "--b", f"{os.path.relpath(art(m), ROOT)}:int8_full",
               "--bench", a.bench, "--classes", str(ncls), "--threads", str(a.threads), "--cache-dir", a.cache_dir, "--json", js]
        if a.shards > 1:   # concurrent sample chunks of one run (the int8 element-wise reference kernels are single-threaded on x86)
            cmd += ["--shards", str(a.shards)]
        v2sha = sha(art(m))
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if r.returncode:
            print(f"{m} {name}: FAILED {r.stderr[-800:]}", flush=True)
            continue
        d = json.load(open(js))
        d["files_sha256"] = {"a": sha(art(m, cur)), "b": v2sha}
        if sha(art(m)) != v2sha:
            d["warning"] = "the v2 file changed during the run"
        json.dump(d, open(js, "w"), indent=1)
        outs.append(js)
    for js in outs:
        d = json.load(open(js))
        A, B, P = d["configs"]["a"], d["configs"]["b"], d["paired"]
        print(f"{m} {os.path.basename(js)}: current {A['top1'] * 100:.2f} v2 {B['top1'] * 100:.2f} "
              f"diff {P['top1_diff_b_minus_a'] * 100:+.2f} pt p={P['mcnemar_exact_p']:.2g} agree {P['top1_prediction_agreement'] * 100:.1f}% n={d['n']}",
              flush=True)


def compare(a):
    """Phone outputs (tools/int8_full_v2_device.py outputs) vs the numpy oracle (check) and across phones: byte identity of
    the oracle's samples (a prefix of the eval128 output) and of the whole output between phones."""
    import glob
    rows = {}
    for m in a.models:
        fsha = sha(art(m))

        def current(p):   # an oracle output of this very file (its check JSON records the file's sha256)
            js = os.path.join(ROOT, "reports", "int8_full_v2", "oracle", f"{m}_eval128_n{p.rsplit('_n', 1)[1][:-4]}.json")
            return os.path.exists(js) and json.load(open(js)).get("file_sha256") == fsha
        orc = sorted((p for p in glob.glob(os.path.join(ROOT, "artifacts", m, "v2", "oracle_eval128_n*.f32")) if current(p)),
                     key=lambda p: int(p.rsplit("_n", 1)[1][:-4]))
        ref = open(orc[-1], "rb").read() if orc else None
        n_or = int(orc[-1].rsplit("_n", 1)[1][:-4]) if orc else 0
        r = {"oracle_n": n_or}
        outs = {}
        for d in a.dirs:
            f = os.path.join(ROOT, d, f"{m}_int8_full_v2.f32")
            if not os.path.exists(f):
                continue
            b = open(f, "rb").read()
            outs[d] = b
            r[os.path.basename(d.rstrip("/"))] = ("NO ORACLE" if ref is None else ("IDENTICAL" if b[:len(ref)] == ref else "DIFF")) + \
                f" (oracle prefix {len(ref) if ref else 0} of {len(b)} bytes)"
        vals = list(outs.values())
        r["phones_identical"] = all(v == vals[0] for v in vals) if vals else None
        rows[m] = r
        print(m, json.dumps(r), flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "check", "accuracy", "compare"))
    ap.add_argument("--dirs", nargs="*", default=[], help="(compare) directories of phone outputs")
    ap.add_argument("--json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--v2-opts", nargs=argparse.REMAINDER, default=None, help="(build) quantizer flags instead of the defaults (last)")
    ap.add_argument("--out", help="(build, one model) output path instead of artifacts/<m>/model_int8_full_v2.bitino")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--bench", default="build/host/bitino_bench")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cache-dir", default="build/tmp/acc_cache")
    ap.add_argument("--only-linear", action="store_true")
    ap.add_argument("--shards", type=int, default=1, help="(accuracy) concurrent sample chunks per run, --threads split among them")
    ap.add_argument("--stages", nargs="*", default=["int8_full_reference", "int8_full"], help="(check) host stages compared")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "compare":
        compare(a)
        return 0
    for m in a.models:
        if a.cmd == "build":
            r = build_one(m, a)
            print(f"{m}: {r['status']}" + (f" ({r['nodes']} nodes, {r['fused_blocks']} fused, {r['seconds']} s)" if "nodes" in r else "")
                  + (f"\n{r['stderr']}" if "stderr" in r else ""), flush=True)
        elif a.cmd == "check":
            r = check_one(m, a)
            print(f"{m}: n={r['n']} " + " ".join(f"{k} {v}" for k, v in r["runners"].items()) + f" (oracle {r['oracle_seconds']} s)", flush=True)
        else:
            accuracy_one(m, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
