#!/usr/bin/env python3
"""Accuracy-vs-latency scatter with the Pareto frontier, as a self-contained SVG (no dependencies).

Used by tools/build_site.py for the README and the website. A model is Pareto-optimal when no other model is both
faster (lower latency) and at least as accurate. Latency is on a log axis (models span two orders of magnitude);
the frontier is drawn as a staircase: from each optimal model, the best accuracy available at that latency or less.
"""
from __future__ import annotations

import math
from html import escape

# Reference palette roles, light and dark; plain colours (no CSS variables) so every SVG renderer draws them.
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "accent": "#2a78d6", "other": "#b5b3ab"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "accent": "#3987e5", "other": "#6b6a65"}


def _rules(c: dict) -> str:
    return (f".bg{{fill:{c['surface']}}} .grid{{stroke:{c['grid']};stroke-width:1}} .axis{{stroke:{c['axis']};stroke-width:1}} "
            f".tick{{fill:{c['muted']}}} .title{{fill:{c['ink']}}} .sub,.alabel{{fill:{c['ink2']}}} .lab{{fill:{c['ink']}}} "
            f".front{{fill:none;stroke:{c['accent']};stroke-width:2;stroke-linejoin:round}} "
            f".dot-f{{fill:{c['accent']};stroke:{c['surface']};stroke-width:2}} .dot-o{{fill:{c['other']};stroke:{c['surface']};stroke-width:2}} "
            f".hit:hover + .dot-o,.hit:hover + .dot-f{{stroke:{c['ink']}}}")


STYLE = ("svg{font-family:system-ui,-apple-system,'Segoe UI',sans-serif} .tick{font-size:11px;font-variant-numeric:tabular-nums} "
         ".title{font-size:14px;font-weight:600} .sub,.alabel,.lab{font-size:11px} .hit{fill:#000;fill-opacity:0} "
         + _rules(LIGHT) + " @media (prefers-color-scheme: dark){" + _rules(DARK) + "}")



def frontier(points: list[dict]) -> list[dict]:
    """Pareto-optimal points (lower x and higher y are better), sorted by x."""
    best, out = -math.inf, []
    for p in sorted(points, key=lambda p: (p["x"], -p["y"])):
        if p["y"] > best:
            out.append(p)
            best = p["y"]
    return out


def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.0001:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo <= v <= hi:
                ticks.append(v)
        e += 1
    return ticks


def _fmt(v: float) -> str:
    return f"{v:g}" if v >= 1 else f"{v:.2g}"


def dominator(p: dict, front: list[dict]) -> dict | None:
    """The slowest frontier point that is at least as fast and at least as accurate as p (its closest better model)."""
    c = [q for q in front if q["id"] != p["id"] and q["x"] <= p["x"] and q["y"] >= p["y"]]
    return max(c, key=lambda q: q["x"]) if c else None


def pareto_svg(points: list[dict], title: str, subtitle: str, xlabel: str, ylabel: str,
               width: int = 760, height: int = 460, interactive: bool = False) -> str:
    """points: [{"id", "label", "x" (ms > 0), "y" (accuracy %), optional "extra" (tooltip line)}]. Returns the SVG
    text. interactive=True (website): no embedded <style> (the page's stylesheet colours it, light and dark theme
    toggle included), no native tooltips, and data-* attributes on every point for assets/app.js (BitinoPareto):
    nearest-point hover with crosshair and tooltip card, click to jump to the model's table row."""
    pts = [p for p in points if p.get("x") and p["x"] > 0 and p.get("y") is not None]
    front = frontier(pts)
    fids = {p["id"] for p in front}
    ml, mr, mt, mb = 56, 24, 58, 46
    pw, ph = width - ml - mr, height - mt - mb
    xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
    xlo, xhi = min(xs) / 1.25, max(xs) * 1.25
    ylo = math.floor((min(ys) - 2) / 5) * 5
    yhi = math.ceil((max(ys) + 2) / 5) * 5

    def X(v):
        return ml + (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) * pw

    def Y(v):
        return mt + (yhi - v) / (yhi - ylo) * ph

    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" ')
    if interactive:
        head += (f'class="pareto-svg" data-ml="{ml}" data-mt="{mt}" data-pw="{pw}" data-ph="{ph}" '
                 f'data-ylo="{ylo}" data-yhi="{yhi}" data-xlo="{xlo:.6g}" data-xhi="{xhi:.6g}" '
                 f'role="group" aria-label="{escape(title)}">')
    else:
        head += f'role="img" aria-label="{escape(title)}"><style>{STYLE}</style>'
    o = [head,
         f'<title>{escape(title)}</title>',
         f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="8"/>',
         f'<text class="title" x="{ml}" y="24">{escape(title)}</text>',
         f'<text class="sub" x="{ml}" y="42">{escape(subtitle)}</text>']
    ystep = 5 if yhi - ylo <= 40 else 10
    for v in range(int(ylo), int(yhi) + 1, ystep):
        o.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="tick" x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for v in _log_ticks(xlo, xhi):
        o.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{mt}" y2="{mt + ph}"/>'
                 f'<text class="tick" x="{X(v):.1f}" y="{mt + ph + 16}" text-anchor="middle">{_fmt(v)}</text>')
    o.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    o.append(f'<text class="alabel" x="{ml + pw / 2:.1f}" y="{height - 10}" text-anchor="middle">{escape(xlabel)}</text>')
    o.append(f'<text class="alabel" transform="translate(14 {mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>')
    # staircase frontier: horizontal at each optimum's accuracy until the next (faster-to-slower) optimum
    if front:
        d = f"M{X(front[0]['x']):.1f},{Y(front[0]['y']):.1f}"
        for a, b in zip(front, front[1:]):
            d += f" H{X(b['x']):.1f} V{Y(b['y']):.1f}"
        d += f" H{ml + pw:.1f}"
        o.append(f'<path class="front" d="{d}"/>')
    if interactive:   # crosshair (shown by the page script), beneath the dots
        o.append('<g class="xh" visibility="hidden"><line class="xh-l xh-v"/><line class="xh-l xh-h"/>'
                 '<text class="xh-t xh-tx" text-anchor="middle"/><text class="xh-t xh-ty" text-anchor="start"/></g>')
    # dots: dominated first (beneath), then the frontier; each with a larger invisible hit target and a tooltip
    for p in sorted(pts, key=lambda p: p["id"] in fids):
        cls = "dot-f" if p["id"] in fids else "dot-o"
        cx, cy = X(p["x"]), Y(p["y"])
        dot = f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{5 if cls == "dot-f" else 4}"/>'
        if interactive:
            dom = None if p["id"] in fids else dominator(p, front)
            note = ("Pareto-optimal: nothing is both faster and more accurate" if p["id"] in fids else
                    f"beaten by {dom['label']}: {_fmt(dom['x'])} ms, {dom['y']:.1f} %" if dom else "")
            o.append(f'<g class="pt" tabindex="0" data-id="{escape(p["id"])}" data-label="{escape(p["label"])}" '
                     f'data-x="{p["x"]:.4g}" data-y="{p["y"]:.2f}" data-front="{1 if p["id"] in fids else 0}" '
                     f'data-note="{escape(note)}" data-extra="{escape(p.get("extra") or "")}">'
                     f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>{dot}</g>')
        else:
            tip = f"{p['label']}: {p['y']:.1f} % top-1, {_fmt(p['x'])} ms" + (" (Pareto-optimal)" if p["id"] in fids else "")
            o.append(f'<g><title>{escape(tip)}</title><circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>{dot}</g>')
    # direct labels on the frontier only: the first candidate position that overlaps no dot, no label and stays
    # inside the plot (fewest overlaps otherwise), with a leader line when the label sits away from its dot
    dots = [(X(p["x"]), Y(p["y"])) for p in pts]
    boxes: list[tuple[float, float, float, float]] = []   # x0, y0 (top), x1, y1

    fdots = [(X(p["x"]), Y(p["y"])) for p in front]

    def rect_dist(bx, px, py):
        x0, y0, x1, y1 = bx
        return math.hypot(max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1))

    def overlaps(bx, own):
        x0, y0, x1, y1 = bx
        n = sum(1 for cx, cy in dots if x0 - 6 < cx < x1 + 6 and y0 - 6 < cy < y1 + 6)
        mine = rect_dist(bx, *own)
        n += 4 * any(rect_dist(bx, fx, fy) < mine for fx, fy in fdots if (fx, fy) != own)   # reads as another dot's label
        n += sum(1 for b in boxes if x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1) * 3
        n += 5 * (x0 < ml or x1 > ml + pw or y0 < mt or y1 > mt + ph)
        return n

    for p in front:
        cx, cy = X(p["x"]), Y(p["y"])
        w, h = 6.1 * len(p["label"]), 12
        cands = []
        for dy in (-10, 14, -24, 28, -38, 42):
            cands += [(cx + 9, cy + dy), (cx - 9 - w, cy + dy)]
        cands += [(cx - w / 2, cy - 14), (cx - w / 2, cy + 22)]
        best = min(cands, key=lambda c: (overlaps((c[0], c[1] - h + 2, c[0] + w, c[1] + 3), (cx, cy)), abs(c[1] - cy)))
        x, y = best
        boxes.append((x, y - h + 2, x + w, y + 3))
        bx = boxes[-1]
        if rect_dist(bx, cx, cy) > 10:   # detached label: leader line to the nearest point of the label box
            nx, ny = min(max(cx, bx[0]), bx[2]), min(max(cy, bx[1]), bx[3])
            o.append(f'<line class="axis" x1="{cx:.1f}" y1="{cy:.1f}" x2="{nx:.1f}" y2="{ny:.1f}"/>')
        o.append(f'<text class="lab" x="{x:.1f}" y="{y:.1f}">{escape(p["label"])}</text>')
    # legend (two marks): right end of the title line
    lx = ml + pw - 220
    o.append(f'<circle class="dot-f" cx="{lx}" cy="20" r="5"/><text class="sub" x="{lx + 10}" y="24">Pareto-optimal</text>'
             f'<circle class="dot-o" cx="{lx + 118}" cy="20" r="4"/><text class="sub" x="{lx + 128}" y="24">other models</text>')
    o.append("</svg>")
    return "\n".join(o)
