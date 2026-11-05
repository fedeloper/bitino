#!/usr/bin/env python3
"""Generates the static website and the README tables from the same data (registry/, results/).

  python3 tools/build_site.py --out ../bitino-site        # website (static, relative links only, no build step to serve)
  python3 tools/build_site.py --readme-only               # rewrite the generated blocks of README.md and docs/MODEL_ZOO.md

Inputs: registry/models.json, registry/recipes.json, results/models/*.json, results/speed_s22.json,
results/speed_x86.json, release.json (+ the release manifest, if --release-manifest is given).
Outputs (site): index.html, models.html, benchmarks.html, docs.html, about.html, assets/{style.css,app.js,logo.svg},
data/models.json. Everything is self-contained: no external fonts, scripts or trackers.

Speed numbers come only from measured runs on the Galaxy S22 and on the x86-64 host, with their date, thread
configuration and throttle state. Emulated runs contribute instruction counts only, never timings.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITE_SRC = os.path.join(ROOT, "site")
REPO_URL = "https://github.com/fedeloper/bitino"
AUTHOR_URL = "https://federicofontana.dev"


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


# ----------------------------------------------------------------------------------------------- data
def pick_speed(runs: list[dict]) -> dict:
    """Representative S22 numbers: latest measured runs of the shipped files, 1 thread on the X2 and 4 threads."""
    out = {}
    if not runs:
        return out
    latest_date = max(r.get("date") or "" for r in runs)

    def best(pred):
        c = [r for r in runs if pred(r) and r.get("median_ms")]
        if not c:
            return None
        c.sort(key=lambda r: (r.get("date") or "", r.get("plan") or ""), reverse=True)
        return c[0]

    for key, thr, aff in (("t1", 1, "7"), ("t4", 4, None)):
        for stage in ("int8_full", "all", "optimized"):
            r = best(lambda r, st=stage: r.get("stage") == st and r.get("threads") == thr
                     and (aff is None and r.get("affinity") in ("unpinned", "") or aff is not None and r.get("affinity") == aff)
                     and "nospin" not in (r.get("config") or "") and "spinrun" not in (r.get("config") or ""))
            if r:
                out[key] = {"ms": round(r["median_ms"], 3), "stage": "float" if stage in ("all", "optimized") else stage,
                            "date": r.get("date"), "validity": r.get("validity"),
                            "file": r.get("model_file"), "config": r.get("config"), "caps": r.get("caps_seen")}
                break
    out["latest"] = latest_date
    return out


def collect() -> dict:
    reg = {m["model_id"]: m for m in load(os.path.join(ROOT, "registry", "models.json"), {}).get("models", [])}
    recipes = load(os.path.join(ROOT, "registry", "recipes.json"), {}).get("recipes", {})
    release = load(os.path.join(ROOT, "release.json"), {})
    rel_manifest = load(os.path.join(ROOT, "registry", "release_models.json"), {}) or {}
    s22 = load(os.path.join(ROOT, "results", "speed_s22.json"), {})
    x86 = load(os.path.join(ROOT, "results", "speed_x86.json"), {})
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from bitino import zoo   # noqa: E402  (licence rule shared with `bitino list --release`)
    models = []
    for mid, rec in sorted(recipes.items()):
        res = load(os.path.join(ROOT, "results", "models", f"{mid}.json"), {})
        m = reg.get(rec.get("registry_model_id", mid), {})
        ok, why = zoo.weights_licence_decision(m)
        acc = res.get("accuracy", {})
        head = acc.get("headline", {})
        big = None
        for st in acc.get("studies", []):
            if st.get("a", "").endswith(":neon") and (st.get("n") or 0) > (big or {}).get("n", 0):
                big = {"set": st.get("eval_set"), "n": st.get("n"), "top1": st.get("a_top1")}
        i8 = None
        for v in acc.get("int8_full", []):
            if "top1" in v:
                i8 = {"file": v["file"], "set": v.get("eval_set"), "n": v.get("n"), "top1": v.get("top1"),
                      "delta": v.get("delta_vs_exact_pt"), "p": v.get("mcnemar_p")}
                if "log2" in v["file"]:
                    break
        if i8 is None:
            for st in sorted(acc.get("studies", []), key=lambda s: s.get("date") or ""):
                if st.get("b", "").endswith(":int8_full") and "poolbug" not in st.get("study", "") and "prev" not in st.get("study", ""):
                    i8 = {"file": st["b"].split(":")[0], "set": st.get("eval_set"), "n": st.get("n"), "top1": st.get("b_top1"),
                          "delta": st.get("delta_pt"), "p": st.get("mcnemar_p")}
        insn = (res.get("instructions_M") or {}).get("values") or {}
        sp = pick_speed((res.get("speed") or {}).get("s22", []))
        xr = [r for r in (res.get("speed") or {}).get("x86", []) if r.get("config") == "t1"]
        assets = []
        rel = (rel_manifest.get("models") or {}).get(mid)
        if ok and rel:   # files listed in the release manifest (tools/release_models.py)
            tag = rel_manifest.get("tag", release.get("tag", "v0.1.0"))
            for x in rel["assets"]:
                assets.append({"file": x["file"], "asset": x["asset"], "sha256": x["sha256"], "bytes": x["bytes"],
                               "url": f"{REPO_URL}/releases/download/{tag}/{x['asset']}"})
        models.append({
            "id": mid, "name": m.get("canonical_name") or mid, "task": m.get("task", ""), "family": m.get("family", ""),
            "framework": rec.get("framework") or "", "precision": ", ".join(m.get("precision", {}).get("categories", [])),
            "strict": bool(m.get("strict_bnn")), "paper": (m.get("paper") or {}).get("title"),
            "paper_url": (m.get("paper") or {}).get("url"), "repo": m.get("repo_url"),
            "metric": head.get("metric"), "eval_set": re.split(r",| \(", head.get("eval_set") or "")[0], "n": head.get("n"),
            "reference_kind": head.get("reference_kind"),
            "reference": head.get("reference"), "runtime": head.get("runtime"), "delta": head.get("delta_vs_reference"),
            "delta_unit": head.get("delta_unit"), "big": big, "int8": i8,
            "insn_opt": insn.get("optimized"), "insn_int8": insn.get("int8_full_log2") or insn.get("int8_full"),
            "s22": sp, "x86_t1": round(xr[0]["median_ms"], 4) if xr else None,
            "licence_code": (m.get("license") or {}).get("code", ""), "licence_weights": (m.get("license") or {}).get("weights", ""),
            "release": ok, "release_reason": why, "assets": assets, "recipe": rec.get("state"),
            "env": rec.get("env"), "files": list((rec.get("expected") or {}).keys()),
        })
    n_reg = len(reg)
    n_weights = sum(1 for m in reg.values() if m.get("registry_status") == "verified-pretrained")
    return {"models": models, "registry_total": n_reg, "verified_weights": n_weights, "s22": s22, "x86": x86,
            "release": release, "release_manifest": rel_manifest}


# ----------------------------------------------------------------------------------------------- formatting
def fmt_acc(v, metric):
    if not isinstance(v, (int, float)):
        return "–"
    if metric and ("PSNR" in metric):
        return f"{v:.2f} dB"
    if metric and "cosine" in metric:
        return f"{v:.5f}"
    return f"{100 * v:.1f}%"


def fmt_ms(v):
    if v is None:
        return "–"
    return f"{v:.3g}" if v < 10 else f"{v:.1f}"


VALIDITY_MARK = {"clean": "", "throttled": " ⚠", "unknown-caps": " †", "superseded-runtime": " ‡"}


# ----------------------------------------------------------------------------------------------- README
def readme_tables(d: dict) -> dict[str, str]:
    ms = d["models"]
    n_conv, n_rel = len(ms), sum(1 for m in ms if m["release"])
    n_ver = sum(1 for m in ms if m["recipe"] == "verified")
    stats = (f"**{d['registry_total']}** binary networks investigated, **{d['verified_weights']}** with verified public "
             f"weights, **{n_conv}** converted and validated against their original framework, **{n_rel}** redistributable "
             f"as converted files (explicit permissive weights licence). Fetch recipes: {n_ver} verified end to end, "
             f"{n_conv - n_ver} reconstructed from the conversion records.")
    hi = [m for m in ms if m["s22"].get("t1") or m["s22"].get("t4")]
    hi.sort(key=lambda m: m["id"])
    lines = ["| model | task | accuracy (reference → Bitino) | S22 1 thread (X2) | S22 4 threads | measured | model file |",
             "|---|---|---|---|---|---|---|"]
    for m in hi:
        t1, t4 = m["s22"].get("t1"), m["s22"].get("t4")
        acc = f"{fmt_acc(m['reference'], m['metric'])} → {fmt_acc(m['runtime'], m['metric'])} ({m['eval_set']}, n={m['n']})"
        def cell(x):
            return f"{fmt_ms(x['ms'])} ms{VALIDITY_MARK.get(x.get('validity'), '')} ({x['stage']})" if x else "–"
        dates = sorted({x["date"] for x in (t1, t4) if x and x.get("date")})
        lines.append(f"| `{m['id']}` | {m['task']} | {acc} | {cell(t1)} | {cell(t4)} | {', '.join(dates)} | "
                     f"{(t1 or t4 or {}).get('file') or ''} |")
    speed = "\n".join(lines) + ("\n\n⚠ = at least one repeat ran with a thermal CPU-frequency cap (phone charging over USB; "
                                "caps are recorded per run in `results/speed_s22.json`); † = run made before the frequency-cap "
                                "check existed (caps unknown). Every published S22 number so far is one of the two: the full "
                                "device suite has not been re-run on the current runtime yet, so each number is for the runtime of its date. "
                                "\"float\" = exact binary layers with float glue (stage all); int8_full = full-integer file. "
                                "Emulated timings are never published.")
    zl = ["| model | task | framework | accuracy (reference → Bitino) | eval set (n) | int8_full Δ vs exact | S22 ms (t1) | weights licence | release |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        i8 = m["int8"]
        i8s = f"{i8['delta']:+.2f} pt ({i8['set']}, n={i8['n']})" if i8 and isinstance(i8.get("delta"), (int, float)) else "–"
        t1 = m["s22"].get("t1")
        zl.append(f"| `{m['id']}` | {m['task']} | {m['framework']} | {fmt_acc(m['reference'], m['metric'])} → "
                  f"{fmt_acc(m['runtime'], m['metric'])} | {m['eval_set']} ({m['n']}) | {i8s} | "
                  f"{(fmt_ms(t1['ms']) + VALIDITY_MARK.get(t1.get('validity'), '')) if t1 else '–'} | "
                  f"{(m['licence_weights'] or '?')[:48].replace('|', '/')} | {'yes' if m['release'] else '–'} |")
    zoo = "\n".join(zl)
    return {"stats": stats, "speed": speed, "zoo_top": "\n".join(zl[:2] + [r for r in zl[2:] if any(f"`{m['id']}`" in r for m in hi or [])]),
            "zoo": zoo}


def replace_block(text: str, name: str, body: str) -> str:
    a, b = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
    if a not in text:
        return text
    pre, rest = text.split(a, 1)
    _, post = rest.split(b, 1)
    return pre + a + "\n" + body.strip() + "\n" + b + post


def write_readme(d: dict) -> None:
    t = readme_tables(d)
    p = os.path.join(ROOT, "README.md")
    text = open(p).read()
    for k in ("stats", "speed", "zoo_top"):
        text = replace_block(text, k, t[k])
    open(p, "w").write(text)
    z = os.path.join(ROOT, "docs", "MODEL_ZOO.md")
    zoo_doc = ("# Model zoo\n\nGenerated by `tools/build_site.py --readme-only` from `registry/` and `results/`. Do not edit by hand.\n\n"
               + t["stats"] + "\n\nAccuracy: the headline numbers are on small smoke-test subsets (n = 128 for most "
               "classifiers; PSNR on crops for image-to-image models), comparing the original model run by its own framework (float64 where recorded, else float32) with "
               "Bitino's exact stage. They show conversion fidelity, not an estimate of full-dataset accuracy. The int8_full "
               "column is the paired difference of the shipped full-integer file against the exact stage on a larger "
               "set (McNemar test in `results/models/<id>.json`).\n\n" + t["zoo"] + "\n")
    open(z, "w").write(zoo_doc)


# ----------------------------------------------------------------------------------------------- site
NAV = [("index.html", "Home"), ("models.html", "Model zoo"), ("benchmarks.html", "Benchmarks"), ("docs.html", "Getting started"),
       ("about.html", "About")]


def page(title: str, active: str, body: str, extra_head: str = "", scripts: str = "") -> str:
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == active else ""}>{esc(n)}</a>' for h, n in NAV)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="Bitino: 1-bit neural networks that keep your phone cool. A binary-neural-network runtime for ARM and x86, model converters and a curated zoo of BNNs with published weights.">
<link rel="icon" href="assets/logo.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/style.css">
<script>try{{var t=localStorage.getItem('bitino-theme');if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
{extra_head}
</head>
<body>
<header class="top">
  <a class="brand" href="index.html"><img src="assets/logo.svg" alt="" width="28" height="28"><span>Bitino</span></a>
  <nav>{nav}</nav>
  <button class="theme" type="button" aria-label="Toggle light or dark theme" title="Light / dark">◐</button>
</header>
<main>
{body}
</main>
<footer>
  <p>Bitino · by Fred · <a href="{AUTHOR_URL}">federicofontana.dev</a> · <a href="{REPO_URL}">GitHub</a> · Apache-2.0 code; model weights keep their own licences.</p>
</footer>
<script src="assets/app.js"></script>
{scripts}
</body>
</html>
"""


def site_index(d: dict) -> str:
    ms = d["models"]
    n_rel = sum(1 for m in ms if m["release"])
    tasks = sorted({m["task"].split(" (")[0] for m in ms})
    best = [m for m in ms if m["s22"].get("t4")]
    fast = min(best, key=lambda m: m["s22"]["t4"]["ms"]) if best else None
    qs = """<pre><code><span class="c"># install the tools (Python 3.10+) and build the runtime</span>
git clone https://github.com/fedeloper/bitino && cd bitino
pip install -e .
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build/host

<span class="c"># a model: the converted file from GitHub Releases (permissive licence) ...</span>
bitino fetch ab-bnn-imagenet-reactnet18 --prebuilt
<span class="c"># ... or the original weights + code: SHA-256 check, local conversion (needs torch)</span>
bitino fetch ab-bnn-imagenet-reactnet18

<span class="c"># run and time it on this machine</span>
bitino run ab-bnn-imagenet-reactnet18 --random 2
bitino bench ab-bnn-imagenet-reactnet18 --threads 4

<span class="c"># deploy on an Android phone (arm64) with the prebuilt binary</span>
adb push bitino_bench-android-arm64 /data/local/tmp/bitino_bench
adb push ~/.cache/bitino/models/ab-bnn-imagenet-reactnet18/model.bitino /data/local/tmp/
adb shell /data/local/tmp/bitino_bench --model /data/local/tmp/model.bitino --threads 4</code></pre>"""
    body = f"""
<section class="hero">
  <img class="hero-logo" src="assets/logo.svg" alt="Bitino logo" width="96" height="96">
  <div>
    <h1>Bitino</h1>
    <p class="tagline">1-bit neural networks that keep your phone cool.</p>
    <p class="by">by Fred · <a href="{AUTHOR_URL}">federicofontana.dev</a></p>
    <p class="cta"><a class="btn" href="models.html">Browse the model zoo</a> <a class="btn ghost" href="{REPO_URL}">Source on GitHub</a></p>
  </div>
</section>
<section class="numbers">
  <div><b>{d['registry_total']}</b><span>binary networks investigated</span></div>
  <div><b>{d['verified_weights']}</b><span>with verified public weights</span></div>
  <div><b>{len(ms)}</b><span>converted &amp; validated</span></div>
  <div><b>{n_rel}</b><span>redistributable files</span></div>
</section>
<section class="pitch">
  <div>
    <h2>Why binary networks</h2>
    <p>A binary layer replaces 32-bit multiply-adds with XNOR and popcount on packed bits: 32 weights per word, a
    fraction of the memory traffic and energy. Phones run them on the CPU they already have, without a hot NPU
    driver or a cloud round trip.</p>
  </div>
  <div>
    <h2>Why Bitino</h2>
    <p>Most published BNNs live in research code for one framework. Bitino converts them from their
    <em>original</em> checkpoints (Larq, PyTorch, OpenVINO, ONNX, Brevitas, MindSpore, …) into one small file format,
    checks every conversion against the source framework in float64, and runs them with hand-written NEON
    (ARMv8.2 dotprod / i8mm) and AVX2 kernels, plus a full-integer int8 path.</p>
  </div>
  <div>
    <h2>Honest numbers</h2>
    <p>Speed comes only from a real phone (Galaxy S22, Snapdragon 8 Gen 1) and a real x86 host, with the date,
    thread setting and thermal state of every run. Emulators are used for correctness and instruction counts,
    never for timings.</p>
  </div>
</section>
<section>
  <h2>Five-minute quickstart</h2>
  {qs}
  <p class="note">Reference environments for the converters (TensorFlow/Larq, PyTorch, OpenVINO, …) are listed in
  <code>envs/</code>; <code>bitino fetch</code> tells you which one a model needs. Details: <a href="docs.html">Getting started</a>.</p>
</section>
<section>
  <h2>Tasks covered</h2>
  <p class="chips">{''.join(f'<span class="chip">{esc(t)}</span>' for t in tasks)}</p>
</section>
"""
    return page("Bitino: 1-bit neural networks that keep your phone cool", "index.html", body)


def site_models(d: dict) -> str:
    data = json.dumps(d["models"], ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = f"""
<section>
  <h1>Model zoo</h1>
  <p>Every converted network, with its accuracy against the original framework, emulated instruction counts and
  measured speed. <b>Accuracy</b> here is the conversion check on a small subset (mostly n = 128): the original model
  run by its own framework (float64 where recorded, else float32) versus Bitino's exact stage. <b>int8 Δ</b> is the shipped full-integer file against the exact stage on a
  larger set. <b>Release</b> means the weights licence is explicitly permissive, so the converted file is attached to
  the GitHub release; every other model is one <code>bitino fetch</code> away from its original source.</p>
  <div class="filters">
    <input id="q" type="search" placeholder="Search models, tasks, frameworks…" aria-label="Search">
    <select id="task" aria-label="Task"><option value="">All tasks</option></select>
    <select id="fw" aria-label="Framework"><option value="">All frameworks</option></select>
    <label><input id="rel" type="checkbox"> release files only</label>
    <label><input id="spd" type="checkbox"> measured on S22</label>
    <span id="count" class="muted"></span>
  </div>
  <div class="table-wrap">
  <table id="zoo">
    <thead><tr>
      <th data-k="id">Model</th><th data-k="task">Task</th><th data-k="framework">Framework</th>
      <th data-k="runtime" class="num">Accuracy (ref → Bitino)</th><th data-k="int8d" class="num">int8 Δ</th>
      <th data-k="insn_opt" class="num">M instr. (exact / int8)</th><th data-k="s22t1" class="num">S22 t1 ms</th>
      <th data-k="s22t4" class="num">S22 t4 ms</th><th data-k="x86" class="num">x86 t1 ms</th>
      <th data-k="release">Licence / get it</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  </div>
  <p class="note">⚠ throttled (thermal frequency cap during at least one repeat) · † caps unknown (run predates the cap check)
  · ‡ x86 number from an older runtime (2026-11-01). M instr. = exact ARM instructions per inference counted under
  qemu (1 thread): a cost measure, not a timing.</p>
</section>
<script id="zoo-data" type="application/json">{data}</script>
"""
    return page("Bitino · Model zoo", "models.html", body, scripts='<script>BitinoZoo.init()</script>')


def worst_cap(r: dict) -> str:
    """Lowest X2 (big core) frequency cap seen during the run, e.g. 'X2 ≤ 1.61 of 3.00 GHz'."""
    caps = []
    for c in r.get("caps_seen") or []:
        m = re.search(r"X2 ([\d.]+)/([\d.]+) GHz", c)
        if m:
            caps.append((float(m.group(1)), float(m.group(2))))
    if not caps:
        return "" if r.get("validity") == "clean" else ("unknown" if r.get("validity") == "unknown-caps" else "")
    lo, mx = min(caps)
    return f"X2 down to {lo:.2f} of {mx:.2f} GHz"


def site_benchmarks(d: dict) -> str:
    s22, x86 = d["s22"], d["x86"]
    dev = s22.get("device", {})
    cores = ", ".join(f"{v['count']}× {k} ({v['max_ghz']:.2f} GHz)" for k, v in (dev.get("cores") or {}).items())
    rows = []
    for r in sorted(s22.get("runs", []), key=lambda r: (r.get("date") or "", r["model_id"], r.get("config") or ""), reverse=True):
        rows.append(f"<tr><td>{esc(r.get('date'))}</td><td><code>{esc(r['model_id'])}</code></td><td>{esc(r.get('model_file'))}</td>"
                    f"<td>{esc(r.get('stage'))}</td><td class=num>{esc(r.get('threads'))}</td><td>{esc(r.get('affinity'))}</td>"
                    f"<td class=num>{fmt_ms(r.get('median_ms'))}</td><td class=num>{esc(r.get('repeats'))}×{esc(r.get('iterations'))}</td>"
                    f"<td><span class='v v-{esc(r.get('validity'))}'>{esc(r.get('validity'))}</span></td>"
                    f"<td class=small title='{esc('; '.join(r.get('caps_seen') or []))}'>{esc(worst_cap(r))}</td></tr>")
    host = x86.get("host", {})
    xrows = []
    for r in sorted(x86.get("runs", []), key=lambda r: r["model_id"]):
        if r.get("config") != "t1":
            continue
        xrows.append(f"<tr><td><code>{esc(r['model_id'])}</code></td><td>{esc(r.get('stage'))}</td><td class=num>{esc(r.get('threads'))}</td>"
                     f"<td class=num>{fmt_ms(r.get('median_ms'))}</td><td>{esc(r.get('date'))}</td><td><span class='v v-{esc(r.get('validity'))}'>"
                     f"{esc(r.get('validity'))}</span></td></tr>")
    body = f"""
<section>
  <h1>Benchmarks</h1>
  <h2>Method</h2>
  <ul>
    <li><b>Device:</b> {esc(dev.get('name'))}, {esc(dev.get('soc'))}, Android {esc(dev.get('android'))}; {esc(cores)}; {esc(dev.get('isa'))}.</li>
    <li><b>Measurement:</b> <code>bitino_bench</code> pushed over adb, one inference (batch 1) per iteration; median per repeat, median of repeats; repeats interleaved across configurations so thermal drift spreads evenly. Each run records CPU frequency caps, Android thermal status and skin temperature before and after.</li>
    <li><b>Validity labels:</b> <span class="v v-clean">clean</span> every repeat ran at full clocks · <span class="v v-throttled">throttled</span> at least one repeat ran under a thermal frequency cap (listed) · <span class="v v-unknown-caps">unknown-caps</span> run made before the cap check existed.</li>
    <li><b>Never published:</b> timings from emulators (qemu) or from CI machines. The emulator provides exact instruction counts, which are shown in the model zoo as a cost measure.</li>
  </ul>
  <h2>Status, honestly</h2>
  <p>Every S22 run published so far is either <span class="v v-throttled">throttled</span> (the phone charged over USB
  and the vendor thermal engine capped the clocks, the X2 core down to 1.61 of 3.00 GHz) or
  <span class="v v-unknown-caps">unknown-caps</span> (the first runs, before the cap check existed). They are real
  measurements on the runtime of their date, labelled as such, and they are not the phone's best case. The full device
  suite has not been re-run since the latest runtime merges (status: {esc((s22.get('next_suite') or {}).get('status', 'pending'))});
  its runs will be added to <code>results/speed_s22.json</code> with the same schema.</p>
  <h2>Galaxy S22 runs ({len(s22.get('runs', []))})</h2>
  <div class="table-wrap"><table class="bench">
  <thead><tr><th>Date</th><th>Model</th><th>File</th><th>Stage</th><th class=num>Threads</th><th>Affinity</th><th class=num>Median ms</th><th class=num>Repeats×iters</th><th>Validity</th><th>Caps seen</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>
  <h2>x86-64 host</h2>
  <p>{esc(host.get('cpu'))}, governor {esc(host.get('governor'))}, turbo {esc(host.get('turbo'))}, max {esc(host.get('scaling_max_freq_ghz'))} GHz,
  load average {esc(host.get('loadavg_start'))} at start. These runs used the runtime of 2026-11-01 (before the int8_full
  and kernel-tuning merges): they are kept, labelled <span class="v v-superseded-runtime">superseded-runtime</span>, until the
  host suite is re-run. Slot: <code>results/speed_x86.json</code> → <code>next_suite</code>.</p>
  <div class="table-wrap"><table class="bench">
  <thead><tr><th>Model</th><th>Stage</th><th class=num>Threads</th><th class=num>Median ms</th><th>Date</th><th>Validity</th></tr></thead>
  <tbody>{''.join(xrows)}</tbody></table></div>
</section>
"""
    return page("Bitino · Benchmarks", "benchmarks.html", body)


def md_to_html(md: str) -> str:
    """Tiny Markdown subset (headings, paragraphs, lists, code blocks, inline code, links, bold) for docs pages."""
    out, in_code, in_list, para = [], False, False, []

    def inline(s):
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', s)
        return s

    def flush():
        nonlocal para
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []

    for line in md.splitlines():
        if line.startswith("```"):
            flush()
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(esc(line))
            continue
        if re.match(r"^#{1,4} ", line):
            flush()
            lvl = len(line.split(" ")[0])
            out.append(f"<h{lvl + 1 if lvl == 1 else lvl}>{inline(line[lvl + 1:])}</h{lvl + 1 if lvl == 1 else lvl}>")
            continue
        if re.match(r"^\s*[-*] ", line):
            flush()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + inline(re.sub(r"^\s*[-*] ", "", line)) + "</li>")
            continue
        if not line.strip():
            flush()
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if in_list and line.startswith("  "):
            out[-1] = out[-1][:-5] + " " + inline(line.strip()) + "</li>"
            continue
        para.append(line.strip())
    flush()
    if in_list:
        out.append("</ul>")
    return "\n".join(out).replace("\n</code></pre>", "</code></pre>")


def site_docs(d: dict) -> str:
    p = os.path.join(ROOT, "docs", "getting-started.md")
    md = open(p).read() if os.path.exists(p) else "# Getting started\n"
    md = re.sub(r"\]\((?!https?://)([^)]+)\)", lambda m: f"]({REPO_URL}/blob/main/docs/{m.group(1)})"
                if not m.group(1).startswith(("../", "#")) else f"]({REPO_URL}/blob/main/{m.group(1)[3:]})", md)
    return page("Bitino · Getting started", "docs.html", "<section class='doc'>" + md_to_html(md) + "</section>")


def site_about(d: dict) -> str:
    body = f"""
<section class="doc">
  <h1>About</h1>
  <p><b>Bitino</b> is made by Fred (Federico Fontana) · <a href="{AUTHOR_URL}">federicofontana.dev</a>.</p>
  <p>The name is an Italian pun: <i>un bitino</i> is “a little bit”. A binary network is exactly that: one little bit
  per weight and per activation, and a phone that stays cool while it thinks.</p>
  <h2>What is in the repository</h2>
  <ul>
    <li>A C++17 runtime with no dependencies (NEON, dotprod, i8mm on ARM; AVX2 / AVX-VNNI on x86-64; scalar reference kernels for every operator).</li>
    <li>Converters from the original frameworks, an independent numpy executor and a five-level validation chain.</li>
    <li>A full-integer int8 quantizer with exact binarization thresholds.</li>
    <li>A registry of {d['registry_total']} investigated binary networks with licences and provenance, fetch recipes for every converted one, and the measured results shown on this site.</li>
  </ul>
  <h2>Licences</h2>
  <p>Code: Apache-2.0. Model weights keep the licence of their authors; converted files are redistributed only when
  that licence is explicitly permissive for the weights themselves, with attribution. Datasets are never redistributed.</p>
  <h2>Links</h2>
  <p><a href="{REPO_URL}">GitHub: fedeloper/bitino</a> · <a href="{AUTHOR_URL}">federicofontana.dev</a></p>
</section>
"""
    return page("Bitino · About", "about.html", body)


def build_site(d: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        p = os.path.join(out, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    shutil.copytree(os.path.join(SITE_SRC, "assets"), os.path.join(out, "assets"))
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    with open(os.path.join(out, "data", "models.json"), "w") as f:
        json.dump(d["models"], f, ensure_ascii=False, indent=0)
    pages = {"index.html": site_index(d), "models.html": site_models(d), "benchmarks.html": site_benchmarks(d),
             "docs.html": site_docs(d), "about.html": site_about(d)}
    for name, text in pages.items():
        with open(os.path.join(out, name), "w") as f:
            f.write(text)
    # Relative links only: the site is served under a sub-path (e.g. /bitino/).
    for name, text in pages.items():
        bad = re.findall(r'(?:href|src)="(/[^"]*)"', text)
        if bad:
            raise SystemExit(f"{name}: absolute site links {bad[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="website output directory")
    ap.add_argument("--readme-only", action="store_true")
    a = ap.parse_args()
    d = collect()
    write_readme(d)
    if not a.readme_only:
        if not a.out:
            raise SystemExit("--out is required (or --readme-only)")
        build_site(d, os.path.abspath(a.out))
        print(f"site: {a.out} ({len(d['models'])} models)")
    else:
        print("README.md and docs/MODEL_ZOO.md tables regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
