#!/usr/bin/env python3
"""Generates the static website and the README tables from the same data (registry/, results/).

  python3 tools/build_site.py --out ../bitino-site        # website (static, relative links only, no build step to serve)
  python3 tools/build_site.py --readme-only               # rewrite the generated blocks of README.md and docs/MODEL_ZOO.md

Inputs: registry/models.json, registry/recipes.json, results/models/*.json, results/devices.json,
results/speed_<device>.json (one per phone), results/speed_x86.json, release.json (+ the release manifest,
registry/release_models.json, when present).
Outputs (site): index.html, models.html, benchmarks.html, docs.html, about.html, assets/{style.css,app.js,logo.svg},
data/models.json. Everything is self-contained: no external fonts, scripts or trackers.

Speed numbers come only from measured runs on real phones (results/devices.json) and on the x86-64 host, with their
date, thread configuration and throttle state. Emulated runs contribute instruction counts only, never timings.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pareto_plot as PP  # noqa: E402

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
SPEED_KEYS = (("t1", 1, False), ("t4", 4, False), ("t1_int8", 1, True), ("t4_int8", 4, True))


def norm_validity(v):
    return "thermal-throttled" if v == "throttled" else v   # files written before the vendor-policy label existed


def pick_speed(runs: list[dict], fastest_cpu: str = "7") -> dict:
    """Representative numbers of one device: latest measured runs, 1 thread pinned to the fastest core and 4 threads
    unpinned, for the float file (stage all: binary layers exact, float32 elsewhere) and for int8 (the full-integer file where measured, else the int8
    kernels on the float file)."""
    out = {}
    if not runs:
        return out

    def best(pred):
        c = [r for r in runs if pred(r) and r.get("median_ms")]
        if not c:
            return None
        c.sort(key=lambda r: (r.get("date") or "", r.get("plan") or ""), reverse=True)
        return c[0]

    for key, thr, int8 in SPEED_KEYS:
        for stage in (("int8_full", "optimized_int8") if int8 else ("all", "optimized")):
            r = best(lambda r, st=stage: r.get("stage") == st and r.get("threads") == thr
                     and (r.get("affinity") in ("unpinned", "") if thr != 1 else r.get("affinity") == fastest_cpu)
                     and "nospin" not in (r.get("config") or "") and "spinrun" not in (r.get("config") or ""))
            if r:
                out[key] = {"ms": round(r["median_ms"], 3), "stage": "float" if stage in ("all", "optimized") else stage,
                            "date": r.get("date"), "validity": norm_validity(r.get("validity")),
                            "file": r.get("model_file"), "config": r.get("config"), "caps": r.get("caps_seen")}
                break
    i8f = {}
    for r in sorted((r for r in runs if r.get("stage") == "int8_full" and r.get("threads") == 1
                     and r.get("affinity") == fastest_cpu and r.get("median_ms")),
                    key=lambda r: (r.get("date") or "", r.get("plan") or "")):
        i8f[os.path.basename(r.get("model_file") or "")] = {"ms": round(r["median_ms"], 3), "date": r.get("date"),
                                                             "validity": norm_validity(r.get("validity"))}
    if i8f:
        out["t1_int8_full_files"] = i8f   # latest run per file
    if out:
        out["latest"] = max(r.get("date") or "" for r in runs)
    return out


def load_devices() -> dict:
    """results/devices.json; without it (older trees) the S22 entry is rebuilt from results/speed_s22.json."""
    devs = (load(os.path.join(ROOT, "results", "devices.json"), {}) or {}).get("devices")
    if devs:
        return devs
    d = (load(os.path.join(ROOT, "results", "speed_s22.json"), {}) or {}).get("device") or {}
    cores = d.get("cores") or {}
    if isinstance(cores, dict):
        cores = [{"name": k, **v} for k, v in sorted(cores.items(), key=lambda kv: -kv[1].get("max_ghz", 0))]
    return {"s22": {**d, "kind": "phone", "short": "Galaxy S22", "cores": cores, "fastest_cpu": "7",
                    "policies": {"7": "X2", "4": "A710", "0": "A510"}}}


def collect() -> dict:
    reg = {m["model_id"]: m for m in load(os.path.join(ROOT, "registry", "models.json"), {}).get("models", [])}
    recipes = load(os.path.join(ROOT, "registry", "recipes.json"), {}).get("recipes", {})
    release = load(os.path.join(ROOT, "release.json"), {})
    rel_manifest = load(os.path.join(ROOT, "registry", "release_models.json"), {}) or {}
    devices = load_devices()
    speed = {}   # device id -> speed_<id>.json, phones with at least one run, in devices.json order
    for did, dev in devices.items():
        doc = load(os.path.join(ROOT, "results", f"speed_{did}.json"), {}) or {}
        if dev.get("kind", "phone") == "phone" and doc.get("runs"):
            speed[did] = doc
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
                b = st.get("b", "")
                if b.endswith(":int8_full") and "/" not in b and st.get("a", "").endswith(":neon") \
                        and "poolbug" not in st.get("study", "") and "prev" not in st.get("study", ""):
                    i8 = {"file": st["b"].split(":")[0], "set": st.get("eval_set"), "n": st.get("n"), "top1": st.get("b_top1"),
                          "delta": st.get("delta_pt"), "p": st.get("mcnemar_p")}
        insn = (res.get("instructions_M") or {}).get("values") or {}
        phones = {}
        for did in speed:
            p = pick_speed((res.get("speed") or {}).get(did, []), devices[did].get("fastest_cpu", "7"))
            if p:
                phones[did] = p
        best = {}
        for key in ("t1", "t4"):   # fastest phone (float file, stage all), for the zoo table
            c = [(p[key]["ms"], did) for did, p in phones.items() if p.get(key)]
            if c:
                ms_, did = min(c)
                best[key] = {**phones[did][key], "device": did, "device_short": devices[did].get("short") or did}
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
            "phones": phones, "best": best, "x86_t1": round(xr[0]["median_ms"], 4) if xr else None,
            "licence_code": (m.get("license") or {}).get("code", ""), "licence_weights": (m.get("license") or {}).get("weights", ""),
            "release": ok, "release_reason": why, "assets": assets, "recipe": rec.get("state"),
            "env": rec.get("env"), "files": list((rec.get("expected") or {}).keys()),
        })
    n_reg = len(reg)
    n_weights = sum(1 for m in reg.values() if m.get("registry_status") == "verified-pretrained")
    return {"models": models, "registry_total": n_reg, "verified_weights": n_weights, "devices": devices, "speed": speed,
            "x86": x86, "release": release, "release_manifest": rel_manifest}


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


VALIDITY_MARK = {"clean": "", "vendor-policy": " ◇", "thermal-throttled": " ⚠", "throttled": " ⚠", "unknown-caps": " †",
                 "superseded-runtime": " ‡"}
VALIDITY_TEXT = {
    "vendor-policy": "the phone's own power policy held the clock ceiling below the maximum with no thermal event "
                     "(stock settings, screen off; the ceiling is recorded per run)",
    "thermal-throttled": "at least one repeat ran under a thermal CPU-frequency cap (e.g. a phone charging over USB)",
    "unknown-caps": "run made before the frequency-cap check existed (caps unknown)",
    "superseded-runtime": "measured on an older runtime",
}


def mark(x: dict | None) -> str:
    return VALIDITY_MARK.get((x or {}).get("validity"), "")


def device_pin(dev: dict) -> str:
    """Name of the core the 1-thread configurations pin to, e.g. 'Cortex-X2'."""
    fc = int(dev.get("fastest_cpu", "7"))
    for c in dev.get("cores") or []:
        lo, _, hi = str(c.get("cpus", "")).partition("-")
        if lo.isdigit() and int(lo) <= fc <= int(hi or lo):
            return c.get("name") or f"cpu{fc}"
    return f"cpu{fc}"


def validity_counts(runs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in runs:
        v = norm_validity(r.get("validity"))
        out[v] = out.get(v, 0) + 1
    return out


def speed_legend(d: dict, html_out: bool = False) -> str:
    used = {norm_validity(r.get("validity")) for doc in d["speed"].values() for r in doc.get("runs", [])}
    parts = [f"{VALIDITY_MARK[v].strip()} = {VALIDITY_TEXT[v]}" for v in VALIDITY_TEXT if v in used]
    status = "; ".join(f"{d['devices'][did].get('short') or did}: " +
                       ", ".join(f"{n} {v}" for v, n in sorted(validity_counts(doc['runs']).items()))
                       for did, doc in d["speed"].items())
    text = ("Unmarked = clean (every repeat at full clocks). " + "; ".join(parts) + ". " if parts else "") + \
        (f"Runs per device and validity: {status}. " if status else "")
    return esc(text) if html_out else text


# ----------------------------------------------------------------------------------------------- README
# Models of the headline speed table and the README zoo excerpt: the released files plus one or two per family.
FEATURED = ("larq-zoo-quicknet-small", "larq-zoo-quicknet", "larq-zoo-quicknet-large", "larq-zoo-birealnet",
            "ab-bnn-imagenet-reactnet18", "ab-bnn-imagenet-reactnet34", "ab-bnn-imagenet-reactneta",
            "omz-resnet18-xnor-binary-onnx-0001", "omz-resnet50-binary-0001", "bnext-tiny", "bi-vit-deit-tiny",
            "bhvit-tiny-binary", "bibert-bert-base-sst2", "ab-bnn-cifar10-reactnet18", "brevitas-bnn-pynq-cnv-1w1a",
            "bbcu-sr-l-x4")


def primary_device(d: dict) -> str | None:
    """The phone of the headline tables: the most recent full arm64 suite (latest run date, then most runs)."""
    c = [(max((r.get("date") or "") for r in doc.get("runs", [])), len(doc.get("runs", [])), did)
         for did, doc in d["speed"].items() if d["devices"][did].get("kernels") != "portable-fallback"]
    return max(c)[2] if c else None


def short_task(t: str) -> str:
    return re.split(r" \(|/", t or "")[0].strip().replace("-classification", "")


def task_label(m: dict) -> str:
    return "mnist" if MNIST_ID.search(m["id"]) else short_task(m["task"])


def ms_cell(x: dict | None) -> str:
    return f"{fmt_ms(x['ms'])}{mark(x)}" if x else "–"


MNIST_ID = re.compile(r"(?:^|-)[tsl]fc(?:-|$)")   # BNN-PYNQ TFC/SFC/LFC: MNIST MLPs (some registry tasks say "cifar10 / mnist")
GROUPS = (("cifar10", "CIFAR-10", lambda m: m["task"].startswith("cifar10-classification") and not MNIST_ID.search(m["id"])),
          ("imagenet", "ImageNet", lambda m: m["task"] == "imagenet-classification"),
          ("others", "Other tasks", lambda m: True))


def short_label(mid: str) -> str:
    """Compact model name for plot labels."""
    s = mid
    for a, b in (("larq-zoo-", ""), ("ab-bnn-imagenet-", "AB-BNN "), ("ab-bnn-cifar10-", "AB-BNN "), ("omz-", "OMZ "),
                 ("bitorch-", "bitorch "), ("-imagenet", ""), ("-cifar10", ""), ("-binary-onnx-0001", ""), ("-binary-0001", ""),
                 ("brevitas-bnn-pynq-", "Brevitas "), ("snn-subbit-", "SNN "), ("tensorpack-", "")):
        s = s.replace(a, b)
    return s


INT8_BROKEN_PT = 10.0   # int8_full files losing more than this are left out of the int8_full plot (named in its note)


def pareto_data(d: dict, dataset: str, int8: bool = False) -> tuple[str | None, list[dict], list[str]]:
    """(device, points, left-out ids) for the accuracy-vs-latency plot of `dataset` on the headline phone, 1 thread.
    Float file (stage all): published full-test-set top-1 (registry/published_accuracy.json); the binary layers are
    exact and the rest is float32 as in the source framework, so the published accuracy applies.
    int8=True, the shipped full-integer file (stage int8_full): published top-1 plus the measured paired difference
    of that file against the exact stage (int8_full Δ, on the larger eval subset), i.e. an estimate; files losing more
    than INT8_BROKEN_PT are left out. Duplicate files (-fp16) are left out."""
    pid = primary_device(d)
    acc = (load(os.path.join(ROOT, "registry", "published_accuracy.json"), {}) or {}).get("models", {})
    pts, dropped = [], []
    for m in d["models"]:
        a = acc.get(m["id"])
        p = (m["phones"].get(pid) or {}) if pid else {}
        if not a or a.get("dataset") != dataset or m["id"].endswith("-fp16"):
            continue
        if not int8:
            if p.get("t1"):
                pts.append({"id": m["id"], "label": short_label(m["id"]), "x": p["t1"]["ms"], "y": a["top1"],
                            "extra": "published full test-set top-1"})
            continue
        i8 = m["int8"]
        t = (p.get("t1_int8_full_files") or {}).get(os.path.basename((i8 or {}).get("file") or ""))
        if not t or not isinstance((i8 or {}).get("delta"), (int, float)):
            continue
        if i8["delta"] < -INT8_BROKEN_PT:
            dropped.append(m["id"])
            continue
        pts.append({"id": m["id"], "label": short_label(m["id"]), "x": t["ms"], "y": round(a["top1"] + i8["delta"], 2),
                    "extra": f"published {a['top1']:.1f} % {i8['delta']:+.2f} pt int8_full Δ ({i8.get('set') or '?'}, n={i8.get('n')})"})
    return pid, pts, dropped


def pareto_figure(d: dict, dataset: str, title: str, int8: bool = False, interactive: bool = False) -> str | None:
    pid, pts, dropped = pareto_data(d, dataset, int8)
    if not pid or len(pts) < 3:
        return None
    dev = d["devices"][pid]
    if int8:
        sub = (f"{dev.get('short') or dev.get('name')}, 1 thread on the {device_pin(dev)}, int8_full file; "
               "top-1 = published + measured Δ (estimate)")
        if dropped:
            sub += f"; {len(dropped)} losing > {INT8_BROKEN_PT:g} pt not shown"
    else:
        sub = (f"{dev.get('name')}, 1 thread on the {device_pin(dev)}, float file (binary layers exact, "
               "float32 elsewhere); published full test-set top-1")
    return PP.pareto_svg(pts, f"{title}{' int8_full' if int8 else ''}: accuracy vs latency ({len(pts)} models)", sub,
                         f"latency per image, ms (log scale), {dev.get('short') or pid} 1 thread",
                         f"{title} top-1 (%)" + (" (estimate)" if int8 else ""), interactive=interactive)


def pareto_files(d: dict, key: str, title: str, interactive: bool = False) -> list[tuple[str, str]]:
    """[(file name under docs/img, svg)] of the plots of one group: float file, then int8_full."""
    out = []
    for int8, suffix in ((False, ""), (True, "_int8_full")):
        svg = pareto_figure(d, key, title, int8, interactive) if any(k == key for k, _ in PARETO) else None
        if svg:
            out.append((f"pareto_{key}{suffix}.svg", svg))
    return out


PARETO = (("imagenet", "ImageNet"), ("cifar10", "CIFAR-10"))


def group_rows(d: dict) -> tuple[str | None, list[str], list[tuple[str, str, list[dict]]]]:
    """Speed tables by task: CIFAR-10 classifiers, ImageNet classifiers, everything else; fastest first on the
    headline phone."""
    pid = primary_device(d)
    others = [did for did in d["speed"] if did != pid]
    left = [m for m in d["models"] if m["phones"]]
    out = []
    big = 1e18
    for key, title, pred in GROUPS:
        g = [m for m in left if pred({**m, "task": m["task"] or ""})]
        left = [m for m in left if m not in g]
        g.sort(key=lambda m: (((m["phones"].get(pid) or {}).get("t1") or {}).get("ms", big), m["id"]))
        if g:
            out.append((key, title, g))
    return pid, others, out


def int8_delta(m: dict) -> str:
    i8 = m["int8"]
    return f"{i8['delta']:+.2f} pt" if i8 and isinstance(i8.get("delta"), (int, float)) else "–"


def readme_tables(d: dict) -> dict[str, str]:
    ms = d["models"]
    devs = d["devices"]
    n_conv, n_rel = len(ms), sum(1 for m in ms if m["release"])
    n_ver = sum(1 for m in ms if m["recipe"] == "verified")
    stats = (f"**{d['registry_total']}** binary networks investigated, **{d['verified_weights']}** with verified public "
             f"weights, **{n_conv}** converted and validated against their original framework, **{n_rel}** redistributable "
             f"as converted files (explicit permissive weights licence). Fetch recipes: {n_ver} verified end to end, "
             f"{n_conv - n_ver} reconstructed from the conversion records.")
    hi = sorted((m for m in ms if m["phones"]), key=lambda m: m["id"])
    pids = list(d["speed"])

    def cell(p: dict | None) -> str:
        if not p:
            return "–"
        def pair(a, b):
            return f"{fmt_ms(a['ms']) if a else '–'}{mark(a)} / {fmt_ms(b['ms']) if b else '–'}{mark(b)}"
        out = pair(p.get("t1"), p.get("t4")) if p.get("t1") or p.get("t4") else ""
        if p.get("t1_int8") or p.get("t4_int8"):
            out += ("<br>" if out else "") + "int8 " + pair(p.get("t1_int8"), p.get("t4_int8"))
        return out or "–"

    head = " | ".join(f"{devs[did].get('short') or did} ms" + (" (32-bit fallback)" if devs[did].get("kernels") == "portable-fallback" else "")
                      for did in pids)
    lines = [f"| model | task | accuracy (reference → Bitino) | {head} |", "|---|---|---|" + "---|" * len(pids)]
    for m in hi:
        acc = f"{fmt_acc(m['reference'], m['metric'])} → {fmt_acc(m['runtime'], m['metric'])} ({m['eval_set']}, n={m['n']})"
        lines.append(f"| `{m['id']}` | {m['task']} | {acc} | " + " | ".join(cell(m["phones"].get(did)) for did in pids) + " |")
    devline = "; ".join(f"{devs[did].get('short') or did}: {devs[did].get('soc')}, 1 thread on the {device_pin(devs[did])}"
                        for did in pids)
    speed_full = "\n".join(lines) + (
        "\n\nEach cell: median latency of one inference, 1 thread pinned to the fastest core / 4 threads; "
        "first line exact binary layers with float glue (stage all), \"int8\" line the full-integer file (int8_full) where "
        "measured, else the int8 kernels on the exact file. " + (devline + ". " if devline else "") + speed_legend(d) +
        "Per-run details (date, runtime, clock ceilings): `results/speed_<device>.json`. Emulated timings are never published.")
    ver = "; ".join(f"{devs[did].get('short') or did}: {verified_text(devs[did])}" for did in devs if verified_text(devs[did]))
    if ver:
        speed_full += ("\n\nOn-device correctness (every model, every stage, `results/models/<id>.json` → device_verification): "
                  + ver + ".")
    pid, others, groups = group_rows(d)
    speed = ""
    if pid:
        pdev, ps = devs[pid], devs[pid].get("short") or pid
        speed += (f"Median latency of one inference (batch 1) in ms. Main columns: **{pdev.get('name')}** ({pdev.get('soc')}); "
                  "1 thread = pinned to the fastest core; int8 = the full-integer file (int8_full) where measured, else the "
                  "int8 kernels on the exact file. Accuracy: the original model in its own framework → Bitino's exact stage, "
                  "on the small check subset; int8_full Δ: the full-integer file's paired difference against the exact stage. "
                  "Fastest first.\n")
        oh = "".join(f" {devs[o].get('short') or o} 1 thread" + (" (32-bit fallback)" if devs[o].get("kernels") == "portable-fallback" else "") + " |"
                     for o in others)
        for key, title, ms_ in groups:
            task_col = key == "others"
            img = "".join(f"\n![{title}{' int8_full' if 'int8' in fn else ''}: accuracy vs latency, Pareto frontier](docs/img/{fn})\n"
                          for fn, _ in pareto_files(d, key, title))
            speed += (f"\n### {title} ({len(ms_)} models)\n{img}\n| model |" + (" task |" if task_col else "") +
                      f" accuracy (reference → Bitino) | int8_full Δ | {ps} 1 thread | {ps} 4 threads | {ps} int8 1 thread |{oh}\n"
                      "|---|" + ("---|" if task_col else "") + "---|---:|---:|---:|---:|" + "---:|" * len(others) + "\n")
            for m in ms_:
                p = m["phones"].get(pid) or {}
                speed += (f"| `{m['id']}` |" + (f" {task_label(m)} |" if task_col else "") +
                          f" {fmt_acc(m['reference'], m['metric'])} → {fmt_acc(m['runtime'], m['metric'])} | "
                          f"{int8_delta(m)} | {ms_cell(p.get('t1'))} | {ms_cell(p.get('t4'))} | {ms_cell(p.get('t1_int8'))} |"
                          + "".join(f" {ms_cell((m['phones'].get(o) or {}).get('t1'))} |" for o in others) + "\n")
        speed += ("\n" + speed_legend(d) + "\n\nEvery phone with 1 and 4 threads and int8: "
                  "[docs/SPEED.md](docs/SPEED.md)." + (" On-device correctness: " + ver + "." if ver else ""))
    else:
        speed = speed_full
    zl = ["| model | task | framework | accuracy (reference → Bitino) | eval set (n) | int8_full Δ vs exact | fastest phone ms (1 thread) | weights licence | release |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        i8 = m["int8"]
        i8s = f"{i8['delta']:+.2f} pt ({i8['set']}, n={i8['n']})" if i8 and isinstance(i8.get("delta"), (int, float)) else "–"
        t1 = m["best"].get("t1")
        zl.append(f"| `{m['id']}` | {m['task']} | {m['framework']} | {fmt_acc(m['reference'], m['metric'])} → "
                  f"{fmt_acc(m['runtime'], m['metric'])} | {m['eval_set']} ({m['n']}) | {i8s} | "
                  f"{(fmt_ms(t1['ms']) + mark(t1) + ' (' + t1['device_short'] + ')') if t1 else '–'} | "
                  f"{(m['licence_weights'] or '?')[:48].replace('|', '/')} | {'yes' if m['release'] else '–'} |")
    zoo = "\n".join(zl)
    feat = [f"`{x}`" for x in FEATURED]
    return {"stats": stats, "speed": speed, "speed_full": speed_full,
            "zoo_top": "\n".join(zl[:2] + [r for r in zl[2:] if r.split(" | ")[0][2:] in feat]),
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
    os.makedirs(os.path.join(ROOT, "docs", "img"), exist_ok=True)
    for key, title in PARETO:
        for fn, svg in pareto_files(d, key, title):
            with open(os.path.join(ROOT, "docs", "img", fn), "w") as f:
                f.write(svg + "\n")
    with open(os.path.join(ROOT, "docs", "SPEED.md"), "w") as f:
        f.write("# Speed: all models on all phones\n\nGenerated by `tools/build_site.py --readme-only` from `results/`. "
                "Do not edit by hand.\n\n" + t["speed_full"] + "\n")
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
    phones = [f"{dev.get('short') or did} ({re.sub(r' [(].*', '', dev.get('soc') or '')})"
              for did, dev in d["devices"].items() if did in d["speed"]]
    phones_txt = ", ".join(phones) if phones else "a real phone"
    qs ="""<pre><code><span class="c"># install the tools (Python 3.10+) and build the runtime</span>
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
    <p>Speed comes only from real phones ({esc(phones_txt)}) and a real x86 host, with the date,
    thread setting and clock state of every run. Emulators are used for correctness and instruction counts,
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
    <label><input id="spd" type="checkbox"> measured on a phone</label>
    <span id="count" class="muted"></span>
  </div>
  <div class="table-wrap">
  <table id="zoo">
    <thead><tr>
      <th data-k="id">Model</th><th data-k="task">Task</th><th data-k="framework">Framework</th>
      <th data-k="runtime" class="num">Accuracy (ref → Bitino)</th><th data-k="int8d" class="num">int8 Δ</th>
      <th data-k="insn_opt" class="num">M instr. (exact / int8)</th><th data-k="p1" class="num">Phone 1 thread ms</th>
      <th data-k="p4" class="num">Phone 4 threads ms</th><th data-k="x86" class="num">x86 t1 ms</th>
      <th data-k="release">Licence / get it</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  </div>
  <p class="note">Phone columns: the fastest measured phone for the float file, stage all (1 thread pinned to its fastest core,
  4 threads unpinned); every device's numbers, int8 included, are on the <a href="benchmarks.html">benchmarks</a> page.
  ◇ vendor power policy held the clock ceiling below the maximum (no thermal event) · ⚠ thermal frequency cap during
  at least one repeat · † caps unknown (run predates the cap check) · ‡ x86 number from an older runtime (2026-11-01).
  M instr. = exact ARM instructions per inference counted under qemu (1 thread): a cost measure, not a timing.</p>
</section>
<script id="zoo-data" type="application/json">{data}</script>
"""
    return page("Bitino · Model zoo", "models.html", body, scripts='<script>BitinoZoo.init()</script>')


def worst_cap(r: dict) -> str:
    """Lowest ceiling of the fastest cpufreq policy seen during the run, e.g. 'X2 down to 1.61 of 3.00 GHz'
    (caps text lists the fastest policy first)."""
    caps = []
    for c in r.get("caps_seen") or []:
        m = re.match(r"\s*([^,]+?) ([\d.]+)/([\d.]+) GHz", c)
        if m:
            caps.append((float(m.group(2)), float(m.group(3)), m.group(1)))
    v = norm_validity(r.get("validity"))
    if not caps:
        return "unknown" if v == "unknown-caps" else ""
    lo, mx, name = min(caps)
    return f"{name} {'held at' if v == 'vendor-policy' else 'down to'} {lo:.2f} of {mx:.2f} GHz"


def vbadge(v) -> str:
    v = norm_validity(v)
    return f"<span class='v v-{esc(v)}'>{esc(v)}</span>"


def verified_text(dev: dict) -> str:
    """On-device correctness summary of results/devices.json "verified" (scripts/device_final_suite.sh)."""
    v = dev.get("verified") or {}
    if not v.get("stages"):
        return ""
    ds = v.get("dates") or []
    when = (f", {ds[0]}" if len(ds) == 1 else f", {ds[0]} to {ds[-1]}") if ds else ""
    how = ("exact stage bit-identical to the numpy executor, float stage within tolerance of the float64 reference"
           if dev.get("kernels") == "portable-fallback" else "outputs byte-identical to the emulated-ARM run of the same code")
    return f"{v['stages_passed']}/{v['stages']} correctness checks passed on {v['models']} models{when} ({how})"


def device_card(did: str, dev: dict, runs: list[dict]) -> str:
    cores = ", ".join(f"{c.get('count')}× {c.get('name')} ({c.get('max_ghz', 0):.2f} GHz)" for c in dev.get("cores") or [])
    counts = " · ".join(f"{n} {vbadge(v)}" for v, n in sorted(validity_counts(runs).items()))
    dates = sorted({r.get("date") for r in runs if r.get("date")})
    span = (dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}") if dates else "–"
    fb = dev.get("kernels") == "portable-fallback"
    notes = "".join(f"<p class=small>{esc(n)}</p>" for n in dev.get("notes") or [])
    return f"""<div class="device" id="dev-{esc(did)}">
    <h3>{esc(dev.get('name') or did)}</h3>
    <p class=small>{esc(dev.get('soc'))} · Android {esc(dev.get('android'))} · {esc(dev.get('abi'))}{' · <b>32-bit fallback build</b>' if fb else ''}</p>
    <p class=small>{esc(cores)}</p>
    <p class=small>{esc(dev.get('isa'))}; 1-thread runs pinned to the {esc(device_pin(dev))} (cpu{esc(dev.get('fastest_cpu', '7'))})</p>
    <p class=small>{len(runs)} runs, {esc(span)}: {counts}</p>
    {f"<p class=small>{esc(verified_text(dev))}</p>" if verified_text(dev) else ""}
    {notes}
  </div>"""


def run_rows(runs: list[dict]) -> str:
    rows = []
    for r in sorted(runs, key=lambda r: (r.get("date") or "", r["model_id"], r.get("config") or ""), reverse=True):
        rows.append(f"<tr><td>{esc(r.get('date'))}</td><td><code>{esc(r['model_id'])}</code></td><td>{esc(r.get('model_file'))}</td>"
                    f"<td>{esc(r.get('stage'))}</td><td class=num>{esc(r.get('threads'))}</td><td>{esc(r.get('affinity'))}</td>"
                    f"<td class=num>{fmt_ms(r.get('median_ms'))}</td><td class=num>{esc(r.get('repeats'))}×{esc(r.get('iterations'))}</td>"
                    f"<td>{vbadge(r.get('validity'))}</td>"
                    f"<td class=small title='{esc('; '.join(r.get('caps_seen') or []))}'>{esc(worst_cap(r))}</td></tr>")
    return "".join(rows)


def site_benchmarks(d: dict) -> str:
    devs, speed, x86 = d["devices"], d["speed"], d["x86"]
    pids = list(speed)
    cards = "".join(device_card(did, devs[did], speed[did].get("runs", [])) for did in pids)

    def cell(x):
        if not x:
            return "<td class=num>–</td>"
        return (f"<td class=num title='{esc(x.get('stage'))}, {esc(x.get('date'))}, {esc(norm_validity(x.get('validity')))}'>"
                f"{fmt_ms(x['ms'])}{esc(mark(x))}</td>")

    hdr1 = "".join(f"<th colspan=4 class=num><a href='#dev-{esc(did)}'>{esc(devs[did].get('short') or did)}</a></th>" for did in pids)
    hdr2 = "".join("<th class=num>1T</th><th class=num>4T</th><th class=num>int8 1T</th><th class=num>int8 4T</th>" for _ in pids)
    xrows = []
    for m in sorted((m for m in d["models"] if m["phones"]), key=lambda m: m["id"]):
        tds = "".join(cell((m["phones"].get(did) or {}).get(k)) for did in pids for k in ("t1", "t4", "t1_int8", "t4_int8"))
        xrows.append(f"<tr><td><code>{esc(m['id'])}</code></td>{tds}</tr>")
    per_dev = "".join(
        f"""<details class="runs"><summary>{esc(devs[did].get('name') or did)}: all runs ({len(speed[did].get('runs', []))})</summary>
  <div class="table-wrap"><table class="bench">
  <thead><tr><th>Date</th><th>Model</th><th>File</th><th>Stage</th><th class=num>Threads</th><th>Affinity</th><th class=num>Median ms</th><th class=num>Repeats×iters</th><th>Validity</th><th>Clock ceiling</th></tr></thead>
  <tbody>{run_rows(speed[did].get('runs', []))}</tbody></table></div></details>"""
        for did in pids)

    gpid, gothers, groups = group_rows(d)
    group_html = ""
    if gpid:
        pdev, ps = devs[gpid], esc(devs[gpid].get("short") or gpid)
        oth = "".join(f"<th class=num>{esc(devs[o].get('short') or o)} 1T{' (32-bit)' if devs[o].get('kernels') == 'portable-fallback' else ''}</th>" for o in gothers)
        group_html = (f"<p class=small>Median latency of one inference (batch 1) in ms. Main columns: {esc(pdev.get('name'))} "
                      f"({esc(pdev.get('soc'))}); 1T = 1 thread pinned to the fastest core, 4T = 4 threads, int8 = the "
                      "full-integer file where measured. Accuracy on the small check subset (reference → Bitino); int8_full Δ = "
                      "paired difference of the full-integer file against the exact stage. Fastest first.</p>")
        for key, title, ms_ in groups:
            task_col = key == "others"
            trs = "".join(f"<tr id=\"row-{esc(m['id'])}\"><td><code>{esc(m['id'])}</code></td>" + (f"<td>{esc(task_label(m))}</td>" if task_col else "") +
                          f"<td>{esc(fmt_acc(m['reference'], m['metric']))} → {esc(fmt_acc(m['runtime'], m['metric']))}</td>"
                          f"<td class=num>{esc(int8_delta(m))}</td>"
                          + "".join(cell((m['phones'].get(gpid) or {}).get(k)) for k in ("t1", "t4", "t1_int8"))
                          + "".join(cell((m['phones'].get(o) or {}).get('t1')) for o in gothers) + "</tr>" for m in ms_)
            figs = [svg for _, svg in pareto_files(d, key, title, interactive=True)]
            group_html += f"""<h2 id="speed-{key}">{esc(title)} ({len(ms_)} models)</h2>
  {''.join('<figure class="pareto">' + fig + '<figcaption class=small>Pareto-optimal: no other model is both faster and at least as accurate. Point at a dot for its numbers; click it to jump to its row.</figcaption></figure>' for fig in figs)}
  <div class="table-wrap"><table class="bench">
  <thead><tr><th>Model</th>{'<th>Task</th>' if task_col else ''}<th>Accuracy (reference → Bitino)</th><th class=num>int8_full Δ</th><th class=num>{ps} 1T</th><th class=num>{ps} 4T</th><th class=num>{ps} int8 1T</th>{oth}</tr></thead>
  <tbody>{trs}</tbody></table></div>"""
    pending = [devs[did].get("short") or did for did in pids if (speed[did].get("next_suite") or {}).get("status") == "pending"]
    host = x86.get("host", {})
    hrows = []
    for r in sorted(x86.get("runs", []), key=lambda r: r["model_id"]):
        if r.get("config") != "t1":
            continue
        hrows.append(f"<tr><td><code>{esc(r['model_id'])}</code></td><td>{esc(r.get('stage'))}</td><td class=num>{esc(r.get('threads'))}</td>"
                     f"<td class=num>{fmt_ms(r.get('median_ms'))}</td><td>{esc(r.get('date'))}</td><td>{vbadge(r.get('validity'))}</td></tr>")
    body = f"""
<section>
  <h1>Benchmarks</h1>
  {group_html}
  <p class=small>{speed_legend(d, html_out=True)}</p>
  <h2>Method</h2>
  <ul>
    <li><b>Measurement:</b> <code>bitino_bench</code> pushed over adb, one inference (batch 1) per iteration; median per repeat, median of repeats; repeats interleaved across configurations so thermal drift spreads evenly. Each run records the CPU frequency ceilings, Android thermal status and skin temperature before and after. 1-thread runs are pinned to the phone's fastest core; 4- and 8-thread runs are left to the scheduler.</li>
    <li><b>Validity labels:</b> <span class="v v-clean">clean</span> every repeat ran with every readable frequency ceiling at its maximum · <span class="v v-vendor-policy">vendor-policy</span> the phone's own power policy held a ceiling below the maximum while Android reported no thermal event (stock settings, screen off; steady, recorded per run) · <span class="v v-thermal-throttled">thermal-throttled</span> at least one repeat ran under a thermal frequency cap · <span class="v v-unknown-caps">unknown-caps</span> run made before the cap check existed.</li>
    <li><b>Never published:</b> timings from emulators (qemu) or from CI machines. The emulator provides exact instruction counts, which are shown in the model zoo as a cost measure.</li>
  </ul>
  <h2>Devices</h2>
  <div class="devices">{cards or '<p>No phone runs yet.</p>'}</div>
  <h2>Status, honestly</h2>
  <p>{speed_legend(d, html_out=True)}Each number is for the runtime of its date (listed per run below).{(' The full device suite has not been re-run on the current runtime on: ' + esc(', '.join(pending)) + '.') if pending else ''}</p>
  <h2>All phones, per model (ms)</h2>
  <p class=small>Median latency of one inference. 1T = 1 thread on the fastest core, 4T = 4 threads; float file, stage all
  (binary layers exact, float32 elsewhere) and int8 (full-integer file where measured, else int8 kernels on the exact file).
  ◇ vendor-policy · ⚠ thermal-throttled · † unknown-caps; hover a number for its stage, date and label.</p>
  <div class="table-wrap"><table class="bench cross">
  <thead><tr><th rowspan=2>Model</th>{hdr1}</tr><tr>{hdr2}</tr></thead>
  <tbody>{''.join(xrows)}</tbody></table></div>
  <h2>Every run</h2>
  {per_dev}
  <h2>x86-64 host</h2>
  <p>{esc(host.get('cpu'))}, governor {esc(host.get('governor'))}, turbo {esc(host.get('turbo'))}, max {esc(host.get('scaling_max_freq_ghz'))} GHz,
  load average {esc(host.get('loadavg_start'))} at start. These runs used the runtime of 2026-11-01 (before the int8_full
  and kernel-tuning merges): they are kept, labelled <span class="v v-superseded-runtime">superseded-runtime</span>, until the
  host suite is re-run. Slot: <code>results/speed_x86.json</code> → <code>next_suite</code>.</p>
  <div class="table-wrap"><table class="bench">
  <thead><tr><th>Model</th><th>Stage</th><th class=num>Threads</th><th class=num>Median ms</th><th>Date</th><th>Validity</th></tr></thead>
  <tbody>{''.join(hrows)}</tbody></table></div>
</section>
"""
    return page("Bitino · Benchmarks", "benchmarks.html", body, scripts='<script>BitinoPareto.init()</script>')


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
