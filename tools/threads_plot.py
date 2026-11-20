#!/usr/bin/env python3
"""Thread scaling chart (self-contained SVG, no dependencies): median speed-up over 1 thread at 1, 2, 4 and 8
threads, one line per phone, from the latest thread suite of each phone in results/speed_<device>.json.

A thread suite is a plan named threads_<date>_<tag>; its configurations t1_fast, t2_fast, t4_fast, t8_fast run the
float file (stage all) pinned to the N fastest cores. Per model, speed-up = t1 / tN of the same suite; models under
0.05 ms at 1 thread are left out (timer resolution). Used by tools/build_site.py.
"""
from __future__ import annotations

import json
import os
import statistics
from html import escape

from pareto_plot import DARK, LIGHT

THREADS = (1, 2, 4, 8)
# Categorical hues in fixed order (blue, orange, purple); identity is also carried by direct labels and markers.
SERIES_LIGHT = ("#2a78d6", "#c8641a", "#7b4fc2")
SERIES_DARK = ("#3987e5", "#e07a2e", "#9a72d9")
MARKERS = ("circle", "square", "diamond")


def suite_medians(runs: list[dict], suffix: str = "_fast") -> tuple[str, dict[int, float], int] | None:
    """(plan, {threads: median speed-up}, models) for the latest threads_* plan with a t1 run, float file."""
    plans = sorted({r["plan"] for r in runs if (r.get("plan") or "").startswith("threads_")
                    and r.get("config") == "t1" + suffix})
    if not plans:
        return None
    plan = max(plans, key=lambda p: (p.split("_")[1:3], p))
    by = {}
    for r in runs:
        if r.get("plan") == plan and isinstance(r.get("median_ms"), (int, float)):
            by.setdefault(r["model_id"], {})[r["config"]] = r["median_ms"]
    out, n = {1: 1.0}, 0
    for t in THREADS[1:]:
        v = [c["t1" + suffix] / c[f"t{t}{suffix}"] for c in by.values()
             if c.get("t1" + suffix, 0) >= 0.05 and c.get(f"t{t}{suffix}")]
        if v:
            out[t] = statistics.median(v)
            n = max(n, len(v))
    return plan, out, n


def collect(root: str, devices: dict) -> list[dict]:
    series = []
    for did, dev in devices.items():
        p = os.path.join(root, "results", f"speed_{did}.json")
        if not os.path.exists(p):
            continue
        s = suite_medians(json.load(open(p)).get("runs", []))
        if s and len(s[1]) == len(THREADS):
            series.append({"id": did, "name": dev.get("short") or did, "soc": dev.get("soc") or "", "plan": s[0],
                           "medians": s[1], "n": s[2]})
    return series


def _marker(kind: str, x: float, y: float, cls: str) -> str:
    if kind == "square":
        return f'<rect class="{cls}" x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1.5"/>'
    if kind == "diamond":
        return f'<path class="{cls}" d="M{x:.1f} {y - 6:.1f}L{x + 6:.1f} {y:.1f}L{x:.1f} {y + 6:.1f}L{x - 6:.1f} {y:.1f}Z"/>'
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'


def threads_svg(series: list[dict], title: str = "Speed-up over 1 thread (median network, float file)") -> str:
    W, H, L, R, T, B = 640, 340, 52, 150, 52, 44
    pw, ph = W - L - R, H - T - B
    ymax = max(8.0, max(max(s["medians"].values()) for s in series)) if series else 8.0
    ymax = 8.0 if ymax <= 8 else float(int(ymax) + 1)
    xs = {t: L + pw * i / (len(THREADS) - 1) for i, t in enumerate(THREADS)}
    def Y(v):
        return T + ph * (1 - (v - 1) / (ymax - 1))
    def rules(c, sc):
        r = (f".bg{{fill:{c['surface']}}} .grid{{stroke:{c['grid']};stroke-width:1}} .axis{{stroke:{c['axis']};stroke-width:1}} "
             f".tick{{fill:{c['muted']}}} .title{{fill:{c['ink']}}} .sub{{fill:{c['ink2']}}} .lab{{fill:{c['ink']}}} "
             f".ideal{{stroke:{c['axis']};stroke-width:1.5;stroke-dasharray:4 4;fill:none}} .idl{{fill:{c['muted']}}}")
        for i, col in enumerate(sc):
            r += f" .s{i}{{stroke:{col};stroke-width:2;fill:none;stroke-linejoin:round}} .m{i}{{fill:{col};stroke:{c['surface']};stroke-width:2}}"
        return r
    style = ("svg{font-family:system-ui,-apple-system,'Segoe UI',sans-serif} .tick{font-size:11px;font-variant-numeric:tabular-nums} "
             ".title{font-size:14px;font-weight:600} .sub,.lab,.idl{font-size:11px} .lab{font-weight:600} "
             + rules(LIGHT, SERIES_LIGHT) + " @media (prefers-color-scheme: dark){" + rules(DARK, SERIES_DARK) + "}")
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{escape(title)}"><style>{style}</style><rect class="bg" width="{W}" height="{H}" rx="6"/>',
         f'<text class="title" x="{L}" y="22">{escape(title)}</text>',
         f'<text class="sub" x="{L}" y="38">Bitino, batch 1, pinned to the N fastest cores; dashed: perfect scaling</text>']
    v = 1
    while v <= ymax:
        o.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="tick" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}×</text>')
        v += 1
    for t, x in xs.items():
        o.append(f'<text class="tick" x="{x:.1f}" y="{T + ph + 18}" text-anchor="middle">{t} thread{"s" if t > 1 else ""}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    ideal = " ".join(f"{xs[t]:.1f},{Y(min(t, ymax)):.1f}" for t in THREADS if t <= ymax)
    o.append(f'<polyline class="ideal" points="{ideal}"/>')
    ends = []
    for i, s in enumerate(series[:3]):
        pts = [(xs[t], Y(s["medians"][t])) for t in THREADS]
        o.append(f'<polyline class="s{i}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for t, (x, y) in zip(THREADS, pts):
            o.append(f'<g>{_marker(MARKERS[i], x, y, f"m{i}")}<title>{escape(s["name"])}, {t} thread{"s" if t > 1 else ""}: '
                     f'{s["medians"][t]:.2f}× (median of {s["n"]} networks)</title></g>')
        ends.append([pts[-1][1], s, i])
    ends.sort(key=lambda e: e[0])   # direct labels at the right end, nudged apart
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 30)
    for y, s, i in ends:
        o.append(f'<text class="lab" x="{L + pw + 12}" y="{y + 4:.1f}">{escape(s["name"])} {s["medians"][8]:.2f}×</text>'
                 f'<text class="tick" x="{L + pw + 12}" y="{y + 17:.1f}">{escape(s["soc"].split(" (")[0][:26])}</text>')
    o.append("</svg>")
    return "".join(o)
