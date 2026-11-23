"""`bitino` command-line interface.

Model zoo:
  bitino list [--task T] [--family F] [--framework F] [--converted] [--strict] [--release] [--json]
  bitino info MODEL_ID
  bitino fetch MODEL_ID [--eval-dir D] [--python ENV=PY] [--file NAME=PATH] [--source NAME=DIR]
                          [--int8 --calib-dir D] [--legacy] [--dry-run]
  bitino run MODEL_ID|FILE [--image F.jpg ... | --input X.f32 | --random N] [--bench] [--int8] [--stage S]
                             [--threads N] [--top K] [--labels FILE]
  bitino bench MODEL_ID|FILE [--stage S] [--threads N] [--iterations I] [--warmup W] [--affinity C] [--json P]
  bitino validate MODEL_ID [--eval-dir D] [--stage S]
  bitino import-qonnx MODEL.onnx --model-id ID --out-dir D [--eval-dir D --preprocess K] [--input-domain uint8|unit8|float32]
                          [--bench PATH] [--python PY]   (QONNX route, docs/qonnx-frontend.md; needs onnx + qonnx,
                          e.g. --python .cache/venv-qonnx/bin/python; `bitino import-qonnx --help` for all options)
  bitino check-qonnx MODEL.onnx [...] [--json OUT|-] [--strict] [--python PY]
                          (validate against the Bitino QONNX profile v1, docs/qonnx-profile.md; needs onnx)
  bitino deepfake MODEL_ID|FILE IMAGE|DIR... [--stage S] [--threads N] [--affinity C] [--repeat R]
                          ("Faster Than Lies" detectors: real/fake score and per-frame time, via bitino_deepfake)

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


def load_registry() -> dict:
    reg = zoo.registry()
    if not reg:
        raise FileNotFoundError(f"registry/models.json not found under {zoo.data_root()}")
    return reg


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
    print(f"{'model_id':44s} {'task':28s} {'recipe':17s} {'release':7s} {'accuracy (ref -> bitino, set)':40s}")
    for r in out:
        if r["converted"]:
            acc = ""
            if isinstance(r.get("reference"), (int, float)) and isinstance(r.get("runtime"), (int, float)):
                pct = "PSNR" not in (r.get("metric") or "")
                f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.2f}")
                acc = f"{f(r['reference'])} -> {f(r['runtime'])} ({(r.get('eval_set') or '').split(' ')[0].rstrip(',')})"
            print(f"{r['model_id'][:44]:44s} {r['task'][:28]:28s} {r['recipe']:17s} {'yes' if r['release'] else '-':7s} {acc[:60]}")
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
                          dry_run=a.dry_run, threads=a.threads, legacy=a.legacy)
    except F.FetchError as e:
        print(f"fetch failed: {e}", file=sys.stderr)
        return 1
    if a.dry_run:
        return 0
    res = rec.get("results", {})
    out = {"model_id": a.model_id, "dir": zoo.model_dir(a.model_id), "build": rec.get("build"), "files": res}
    if rec.get("qonnx_files"):
        out["qonnx_files"] = rec["qonnx_files"]
    if rec.get("crosscheck"):
        out["crosscheck"] = rec["crosscheck"]
    print(json.dumps(out, indent=1))
    bad = [k for k, v in res.items() if not (v.get("identical_data") and v.get("identical_graph") is not False)]
    if bad:
        print(f"note: data section or graph differs from the published conversion for {bad}", file=sys.stderr)
    if rec.get("crosscheck") and not rec["crosscheck"].get("agree"):
        print("note: the legacy converter's files differ from the QONNX-built ones (see crosscheck)", file=sys.stderr)
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


def native_ok() -> bool:
    try:
        from .. import _native
        return _native.available()
    except Exception:
        return False


def model_for_run(spec: str, variant: str) -> tuple[str, str | None]:
    """(file, zoo id): a path, a fetched zoo model, or a redistributable one downloaded from the release now."""
    if os.path.isfile(spec):
        return spec, None
    p = zoo.fetched_file(spec, variant)
    if not os.path.isfile(p) and zoo.release_entry(spec)[0]:
        from .. import fetch as F
        F.fetch_prebuilt(spec)
    return zoo.resolve_model_file(spec, variant), spec


def load_labels(path: str | None) -> list[str] | None:
    """Class names: a text file with one name per line, or a JSON list / {"0": [wnid, name], ...} map."""
    if not path:
        return None
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        d = json.loads(text)
        if isinstance(d, dict):
            return [(lambda v: v[-1] if isinstance(v, list) else str(v))(d[k]) for k in sorted(d, key=int)]
        return [str(v) for v in d]
    return [ln.strip() for ln in text.splitlines()]


def cmd_run(a) -> int:
    import numpy as np
    variant = "int8" if a.int8 else "float"
    stage = a.stage or ("int8_full" if a.int8 else "all")
    bench_bin = a.bench or None                  # --bench PATH: run through that bitino_bench binary
    native = bench_bin is None and native_ok()
    try:
        model, mid = model_for_run(a.model, variant)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 3
    shape = model_inputs(model)[0]["shape"]
    per = int(np.prod(shape[1:]))
    if a.image:
        from .. import images
        key = a.preprocess or (zoo.preprocess_key(mid) if mid else None)
        if not key:
            print(f"--image: preprocessing unknown for {a.model}; pass --preprocess ({', '.join(images.supported())})",
                  file=sys.stderr)
            return 2
        try:
            x = np.concatenate([images.load_image(p, key) for p in a.image])
        except (ImportError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 2
        src = f"{len(a.image)} image(s), preprocessing {key}"
    elif a.input:
        x = np.fromfile(a.input, np.float32)
        if x.size % per:
            print(f"{a.input}: {x.size} floats is not a multiple of the input size {shape[1:]}", file=sys.stderr)
            return 2
        x = x.reshape([-1] + list(shape[1:]))
        src = "file " + a.input
    else:
        x = np.random.default_rng(0).standard_normal([a.random] + list(shape[1:])).astype(np.float32)
        src = "random, seed 0"
    n = x.shape[0]
    timing = None
    if native:
        from .. import runtime
        with runtime.Model(model, stage=stage, threads=a.threads, model_id=mid) as m:
            ys = m.predict(x)
            ys = ys if isinstance(ys, list) else [ys]
            y = np.concatenate([v.reshape(n, -1) for v in ys], axis=1)   # bitino_bench --outputs record layout
            first = ys[0].reshape(n, -1).shape[1]
            if a.bench == "":
                timing = m.benchmark(iterations=a.iterations, warmup=a.warmup, full=True)
    else:
        bench = bench_or_die(bench_bin)
        with tempfile.TemporaryDirectory(prefix="bitino-run-") as td:
            inp, outp = os.path.join(td, "x.f32"), os.path.join(td, "y.f32")
            x.astype(np.float32).tofile(inp)
            r = subprocess.run([bench, "--model", model, "--inputs", inp, "--outputs", outp, "--stage", stage,
                                "--threads", str(a.threads), "--warmup", "0", "--iterations", "1"], capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                return 1
            y = np.fromfile(outp, np.float32).reshape(n, -1)
        first = y.shape[1]
    if a.output:
        y.astype(np.float32).tofile(a.output)
    labels = load_labels(a.labels)
    print(f"model {model}\ninput {n} x {list(shape[1:])} ({src}), stage {stage}, output {n} x {y.shape[1]} values"
          + ("" if native else " (via " + os.path.basename(bench) + ")"))
    if first > 1 and first <= 100000:
        for i in range(min(n, 8)):
            yi = y[i, :first]
            top = np.argsort(-yi)[: a.top]
            name = (lambda k: f" {labels[k]}" if labels and k < len(labels) else "")
            print(f"  sample {i}: top-{a.top} " + ", ".join(f"{int(k)}{name(int(k))}:{yi[k]:.4g}" for k in top))
    if timing:
        print(f"latency on this machine ({timing['cpu']['arch']}: {' '.join(timing['cpu']['features'])}): median "
              f"{timing['median_ms']:.3f} ms, min {timing['min_ms']:.3f} ms over {timing['iterations']} runs "
              f"(warmup {timing['warmup']}, {timing['threads']} thread(s))")
    return 0


def cmd_bench(a) -> int:
    model = zoo.resolve_model_file(a.model, "int8" if a.int8 else "float")
    stage = a.stage or ("int8_full" if a.int8 else "all")
    if not a.bench and not zoo.find_bench() and native_ok():
        # No bitino_bench binary: time with the runtime of the Python package (same timing loop: run() only).
        from .. import runtime
        aff = [int(c) for c in a.affinity.split(",")] if a.affinity else None
        with runtime.Model(model, stage=stage, threads=a.threads, affinity=aff) as m:
            r = m.benchmark(iterations=a.iterations, warmup=a.warmup, full=True)
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"schema": "bitino.python_benchmark.v1", "model": model, **r}, f, indent=1)
        print(f"{os.path.basename(model)} stage {stage}, {a.threads} thread(s): median {r['median_ms']:.3f} ms, "
              f"p90 {r['p90_ms']:.3f} ms over {a.iterations} iterations (warmup {a.warmup}; Python package runtime)")
        print("note: timings are only meaningful on real hardware; never under an emulator", file=sys.stderr)
        return 0
    bench = bench_or_die(a.bench)
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


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def deepfake_tensor(paths: list[str]) -> tuple["object", list[str]]:
    """The authors' test-loader preprocessing (tools/eval_sets/ftl_deepfake.py): RGB -> Resize(252, bilinear,
    antialias) -> CenterCrop(224) -> u8 * fl32(1/255); torchvision when importable, else PIL (same steps)."""
    import numpy as np
    from PIL import Image
    try:
        import torch
        import torchvision.transforms.v2 as T
        tf = T.Compose([T.Resize(252, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224), T.ToImage(),
                        T.ToDtype(torch.float32, scale=True)])
        conv = lambda im: tf(im).permute(1, 2, 0).contiguous().numpy()  # noqa: E731
    except ImportError:
        def conv(im):
            w, h = im.size
            s = 252 / min(w, h)
            im = im.resize((252 if w <= h else int(w * s), 252 if h < w else int(h * s)), Image.BILINEAR)
            w, h = im.size
            l, t = int(round((w - 224) / 2.0)), int(round((h - 224) / 2.0))
            return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8).astype(np.float32) * np.float32(1 / 255)
    xs = [conv(Image.open(p).convert("RGB"))[None] for p in paths]
    return np.ascontiguousarray(np.concatenate(xs), np.float32), [os.path.basename(p) for p in paths]


def cmd_deepfake(a) -> int:
    model = zoo.resolve_model_file(a.model)
    tool = a.tool or zoo.find_bench(None)
    tool = os.path.join(os.path.dirname(tool), "bitino_deepfake") if tool else None
    if not tool or not os.path.isfile(tool):
        print("bitino_deepfake not found (build it with cmake --build build/host, or pass --tool)", file=sys.stderr)
        return 3
    images, passthrough = [], []
    for p in a.inputs:
        if os.path.isdir(p):
            names = sorted(os.listdir(p))
            images += [os.path.join(p, n) for n in names if n.lower().endswith(IMAGE_EXT)]
            if any(n.lower().endswith(".ppm") for n in names):
                passthrough.append(p)
        elif p.lower().endswith(IMAGE_EXT):
            images.append(p)
        else:
            passthrough.append(p)   # .ppm / .f32: read by bitino_deepfake
    with tempfile.TemporaryDirectory(prefix="bitino-deepfake-") as td:
        args = list(passthrough)
        if images:
            x, names = deepfake_tensor(images)
            f = os.path.join(td, "images.f32")
            x.tofile(f)
            print("images.f32[i]: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)), flush=True)
            args.append(f)
        argv = [tool, "--model", model, "--stage", a.stage, "--threads", str(a.threads), "--repeat", str(a.repeat)]
        argv += (["--affinity", a.affinity] if a.affinity else []) + args
        return subprocess.run(argv).returncode


def cmd_validate(a) -> int:
    rec = zoo.recipes().get(a.model_id)
    if not rec:
        print(f"no recipe for {a.model_id}", file=sys.stderr)
        return 3
    d = zoo.model_dir(a.model_id)
    if not os.path.isdir(d):
        print(f"{a.model_id} not fetched; run `bitino fetch {a.model_id}`", file=sys.stderr)
        return 3
    if not os.path.exists(os.path.join(d, "reference_logits.f32")):
        # the QONNX build checks identity with the published file; the source framework's reference outputs come from
        # the legacy converter's cross-check run (fetch --legacy), whose model files are compared with the QONNX ones
        ld = os.path.join(d, "legacy")
        if not os.path.exists(os.path.join(ld, "reference_logits.f32")):
            print(f"{a.model_id}: no reference outputs of the source framework in {d}; the QONNX build checks identity "
                  f"with the published file only. Run `bitino fetch {a.model_id} --legacy` (the legacy converter computes "
                  f"them into {ld})", file=sys.stderr)
            return 3
        d = ld
    bench = bench_or_die(a.bench)
    eval_dir = a.eval_dir
    if not eval_dir:
        with open(os.path.join(zoo.model_dir(a.model_id), "fetch_record.json")) as f:
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


def _onnx_python(py: str | None) -> str | None:
    """Interpreter for the onnx-based subcommands: --python, else None when onnx is importable here, else
    .cache/venv-qonnx/bin/python of the checkout when it exists."""
    if py:
        return py
    try:
        import onnx  # noqa: F401
        return None
    except ImportError:
        cand = os.path.join(ROOT, ".cache", "venv-qonnx", "bin", "python")
        return cand if os.path.exists(cand) else ""


def cmd_check_qonnx(a) -> int:
    """Profile check of QONNX files (bitino.qonnx_profile). In-process when onnx is importable, else through
    tools/check_qonnx.py in --python or .cache/venv-qonnx/bin/python (python -I)."""
    rest = list(a.args)
    if rest and rest[0] == "--":
        rest = rest[1:]
    py = _onnx_python(a.python)
    if py:
        return subprocess.call([py, "-I", os.path.join(ROOT, "tools", "check_qonnx.py")] + rest)
    if py == "":
        print("check-qonnx needs onnx; pass --python .cache/venv-qonnx/bin/python", file=sys.stderr)
        return 2
    from .. import qonnx_profile
    return qonnx_profile.main(rest)


def cmd_import_qonnx(a) -> int:
    """QONNX -> .bitino (bitino.qonnx_import). In-process when onnx is importable, else (or with --python) through
    tools/import_qonnx.py in the given interpreter (python -I)."""
    rest = list(a.args)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if a.python:
        return subprocess.call([a.python, "-I", os.path.join(ROOT, "tools", "import_qonnx.py")] + rest)
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("import-qonnx needs onnx (+ qonnx for cleanup and the float32 reference); pass --python "
              ".cache/venv-qonnx/bin/python", file=sys.stderr)
        return 2
    from .. import qonnx_import
    return qonnx_import.main(rest)


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

    fe = sub.add_parser("fetch", help="download the original weights, verify, build locally through QONNX")
    fe.add_argument("model_id")
    fe.add_argument("--eval-dir", help="real eval set (default: a small synthetic one)")
    fe.add_argument("--python", action="append", metavar="ENV=PATH", help="python of a reference env, e.g. larq=/venv/bin/python")
    fe.add_argument("--file", action="append", metavar="NAME=PATH", help="use an already downloaded file (SHA-256 still checked)")
    fe.add_argument("--source", action="append", metavar="NAME=DIR", help="use a local source tree instead of downloading it")
    fe.add_argument("--int8", action="store_true", help="also quantize to int8_full with the recorded flags (your calibration "
                    "images), write its canonical QONNX file and re-import it (must equal the quantizer output)")
    fe.add_argument("--legacy", action="store_true", help="also run the legacy converter (tools/convert_*.py) as a cross-check "
                    "into <model dir>/legacy/ (and its source-framework reference outputs, used by `bitino validate`)")
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
        if name == "run":
            q.add_argument("--bench", nargs="?", const="", metavar="PATH",
                           help="without a value: also time the model (native runtime); with PATH: run through that "
                                f"{zoo.BENCH_NAME} binary instead of the Python runtime")
            q.add_argument("--image", nargs="+", metavar="FILE",
                           help="image file(s), decoded and preprocessed as the model was validated (needs Pillow)")
            q.add_argument("--preprocess", help="preprocessing key for --image (default: the zoo model's)")
            q.add_argument("--labels", help="class names: text file (one per line) or JSON list / index map")
            q.add_argument("--input", help="raw float32 file, N x input shape")
            q.add_argument("--random", type=int, default=1, help="N random inputs if neither --image nor --input is given")
            q.add_argument("--output", help="write the raw float32 outputs here")
            q.add_argument("--top", type=int, default=5)
            q.add_argument("--iterations", type=int, default=50, help="timed runs with --bench")
            q.add_argument("--warmup", type=int, default=10, help="untimed runs before timing with --bench")
        else:
            q.add_argument("--bench", help=f"path to {zoo.BENCH_NAME}")
            q.add_argument("--iterations", type=int, default=50)
            q.add_argument("--warmup", type=int, default=10)
            q.add_argument("--affinity")
            q.add_argument("--json")
        q.set_defaults(fn=fn)

    dfk = sub.add_parser("deepfake", help="run a 'Faster Than Lies' deepfake detector on images / frames (score + time per frame)")
    dfk.add_argument("model", help="model id (fetched) or model file")
    dfk.add_argument("inputs", nargs="+", help="JPEG/PNG files, .ppm frames, directories of them, or .f32 tensors")
    dfk.add_argument("--stage", default="all")
    dfk.add_argument("--threads", type=int, default=1)
    dfk.add_argument("--affinity")
    dfk.add_argument("--repeat", type=int, default=1)
    dfk.add_argument("--tool", help="path to bitino_deepfake (default: next to the bitino_bench found)")
    dfk.set_defaults(fn=cmd_deepfake)

    va = sub.add_parser("validate", help="check the runtime against the numpy executor and the source references")
    va.add_argument("model_id")
    va.add_argument("--eval-dir")
    va.add_argument("--stage", default="neon")
    va.add_argument("--threads", type=int, default=4)
    va.add_argument("--bench")
    va.set_defaults(fn=cmd_validate)

    iq = sub.add_parser("import-qonnx", add_help=False,
                        help="import a QONNX / FINN-ONNX graph (docs/qonnx-frontend.md); `import-qonnx --help` for options")
    iq.add_argument("--python", help="run the importer in this interpreter (e.g. .cache/venv-qonnx/bin/python)")
    iq.add_argument("args", nargs=argparse.REMAINDER)
    iq.set_defaults(fn=cmd_import_qonnx)

    cq = sub.add_parser("check-qonnx", add_help=False,
                        help="check QONNX files against the Bitino QONNX profile v1 (docs/qonnx-profile.md)")
    cq.add_argument("--python", help="run the checker in this interpreter (default: here if onnx is importable, "
                                     "else .cache/venv-qonnx/bin/python)")
    cq.add_argument("args", nargs=argparse.REMAINDER)
    cq.set_defaults(fn=cmd_check_qonnx)

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
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "import-qonnx":   # pass every option through (incl. --help) to bitino.qonnx_import
        rest, py = argv[1:], None
        if "--python" in rest:
            k = rest.index("--python")
            py, rest = rest[k + 1], rest[:k] + rest[k + 2:]
        return cmd_import_qonnx(argparse.Namespace(python=py, args=rest))
    if argv and argv[0] == "check-qonnx":    # pass every option through (incl. --help) to bitino.qonnx_profile
        rest, py = argv[1:], None
        if "--python" in rest:
            k = rest.index("--python")
            py, rest = rest[k + 1], rest[:k] + rest[k + 2:]
        return cmd_check_qonnx(argparse.Namespace(python=py, args=rest))
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
