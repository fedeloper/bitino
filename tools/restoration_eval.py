#!/usr/bin/env python3
"""Full-image PSNR / SSIM of the converted restoration / super-resolution models on the standard test sets (slice S8):
the official PyTorch model (float32, CPU) against Bitino (bitino_bench, host) with each source's own protocol.

Bitino graphs have a fixed input shape, so the model is converted once per distinct test-image size with the same
adapter and settings as the artifact (deployable ``ge`` binarization, fused passes = ``model.bitino``). The graph
built at the artifact's crop size is checked to equal ``artifacts/<id>/model.bitino`` (constants and nodes), so
the per-size graphs differ from the artifact only in shape attributes. Every image of one size goes through one
``bitino_bench --inputs`` call per stage.

Protocols (tools/eval_sets/restoration.py for the degradations, bitino.validation.image_metrics for metrics):
  bbcu-denoise      CBSD68 / Kodak24, AWGN sigma 15/25/50 (seed 0 per image), RGB PSNR/SSIM on uint8 (KAIR)
  bbcu-deblocking   LIVE1 / Classic5, OpenCV JPEG q 10/20/30/40 on the KAIR Y channel, PSNR/SSIM on uint8 Y (KAIR)
  bbcu-sr-*         Set5 / Set14 / B100, MATLAB-style bicubic LR, Y PSNR/SSIM with crop_border = scale (BasicSR)
  bsrc2023-*        Set5 / Set14 / B100, same LR, BSRC tools/metrics PSNR (Y of the unrounded output, border =
                    scale), single forward and the evaluator's 4-way self-ensemble (identity, hflip, vflip,
                    transpose; outputs mapped back and averaged in float32 as bnn_competition/evaluation/evaluator.py)

  python3 -I tools/restoration_eval.py --model-id bbcu-denoise --repo .cache/repos/BBCU \
      --weights .cache/checkpoints/bbcu-denoise/BBCU-denoise.pth --artifacts artifacts/bbcu-denoise \
      --set CBSD68=<dir> --bench build/host/bitino_bench --json reports/restoration/bbcu-denoise_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "eval_sets"))

from bitino import format as bfformat, passes  # noqa: E402
from bitino.validation import image_metrics as im  # noqa: E402
import convert_restoration as cr  # noqa: E402
import restoration as rd  # noqa: E402


def family_of(model_id: str) -> str:
    return "bbcu" if model_id.startswith("bbcu") else "bsrc"


def build_items(model_id: str, sets: dict, args) -> list[dict]:
    """Degraded full images: [{set, file, cond, x (HWC float32), gt (HWC uint8)}]."""
    items = []
    for sname, d in sets.items():
        files = rd.image_paths(d)[: args.limit or None]
        for f in files:
            base = os.path.basename(f)
            if model_id == "bbcu-denoise":
                img = rd.imread_uint_rgb(f)
                for s in args.sigmas:
                    items.append({"set": sname, "file": base, "cond": f"sigma{s:g}", "x": rd.noisy(img, s), "gt": img})
            elif model_id == "bbcu-deblocking":
                y = rd.rgb2y_kair(rd.imread_uint_rgb(f))
                for q in args.qualities:
                    items.append({"set": sname, "file": base, "cond": f"q{q}", "x": rd.u8_to_unit_f32(rd.jpeg_gray(y, q))[..., None],
                                  "gt": y[..., None]})
            else:
                sc = int(model_id.split("-x")[1][0])
                hr = rd.modcrop(rd.imread_rgb_color(f), sc)
                lr = rd.matlab_bicubic_lr(hr, sc, args.matlab_functions)
                x = rd.u8_to_unit_f32(lr) if model_id.startswith("bbcu") else lr.astype(np.float32)
                items.append({"set": sname, "file": base, "cond": f"x{sc}", "x": x, "gt": hr})
    return items


def metric_for(model_id: str) -> tuple[str, int]:
    if model_id == "bbcu-denoise":
        return "kair_rgb", 0
    if model_id == "bbcu-deblocking":
        return "kair_gray", 0
    sc = int(model_id.split("-x")[1][0])
    return ("basicsr_y" if model_id.startswith("bbcu") else "bsrc_y"), sc


def graph_signature(g) -> tuple:
    return (tuple((n.op, tuple(n.inputs), json.dumps(n.attrs, sort_keys=True)) for n in g.nodes),
            tuple(sorted((k, c.dtype, c.shape, c.data.tobytes()) for k, c in g.constants.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--artifacts", required=True, help="artifacts/<id> (model.bitino + conversion_report.json)")
    ap.add_argument("--set", action="append", required=True, help="NAME=DIR (repeatable)")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[15, 25, 50])
    ap.add_argument("--qualities", nargs="+", type=int, default=[10, 20, 30, 40])
    ap.add_argument("--matlab-functions", help="BBCU-SR/bbcu/utils/matlab_functions.py (SR)")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--stages", nargs="+", default=["neon", "optimized"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--self-ensemble", action="store_true", help="BSRC: also the evaluator's 4-way self-ensemble")
    ap.add_argument("--limit", type=int, default=0, help="first N images per set (smoke runs)")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    fam = family_of(args.model_id)
    model, cls, _ = cr.load_model(fam, args.repo, args.model_id, args.weights)
    rep = json.load(open(os.path.join(args.artifacts, "conversion_report.json")))
    art = bfformat.read(os.path.join(args.artifacts, "model.bitino"))
    hw = tuple(art.inputs[0]["shape"][1:3])

    def build(h, w):
        g, _ = cr.convert_graph(fam, model, args.model_id, "ge", (h, w))
        passes.optimize(g, fold=True)
        return g

    same = graph_signature(build(*hw)) == graph_signature(art)
    assert same, "graph rebuilt at the artifact size differs from model.bitino"
    sets = dict(s.split("=", 1) for s in args.set)
    items = build_items(args.model_id, sets, args)
    metric, border = metric_for(args.model_id)
    variants = [("single", None)]
    if args.self_ensemble:
        variants += [("hflip", "h"), ("vflip", "v"), ("transpose", "t")]

    def tf(x, op):            # HWC
        return x if op is None else (x[:, ::-1] if op == "h" else x[::-1] if op == "v" else np.transpose(x, (1, 0, 2)))

    # source fp32 (and its ensemble members)
    t0 = time.time()
    src = {}
    with torch.no_grad():
        for i, it in enumerate(items):
            for vname, op in variants:
                xi = np.ascontiguousarray(tf(it["x"], op))
                y = model(torch.from_numpy(xi.transpose(2, 0, 1)[None])).numpy()[0].transpose(1, 2, 0)
                src[(i, vname)] = np.ascontiguousarray(tf(y, op))
    t_src = time.time() - t0
    # Bitino, grouped by input size
    bf = {st: {} for st in args.stages}
    groups: dict = {}
    for i, it in enumerate(items):
        for vname, op in variants:
            xi = np.ascontiguousarray(tf(it["x"], op))
            groups.setdefault(xi.shape[:2], []).append((i, vname, op, xi))
    tmp = tempfile.mkdtemp(prefix="bf_restoration_")
    t0 = time.time()
    for (h, w), members in sorted(groups.items()):
        mp = os.path.join(tmp, f"m_{h}x{w}.bitino")
        bfformat.write(build(h, w), mp)
        ip = os.path.join(tmp, "in.f32")
        np.stack([m[3] for m in members]).astype(np.float32).tofile(ip)
        for st in args.stages:
            op_ = os.path.join(tmp, "out.f32")
            subprocess.run([args.bench, "--model", mp, "--inputs", ip, "--outputs", op_, "--stage", st, "--threads", str(args.threads),
                            "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
            ys = np.fromfile(op_, np.float32).reshape(len(members), -1)
            for (i, vname, op, xi), y in zip(members, ys):
                oh, ow = items[i]["gt"].shape[:2]
                if op == "t":
                    oh, ow = ow, oh
                bf[st][(i, vname)] = np.ascontiguousarray(tf(y.reshape(oh, ow, -1), op))
        os.remove(mp)
    t_bf = time.time() - t0
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    def ensemble(store, i):
        ys = torch.from_numpy(np.stack([store[(i, v)].transpose(2, 0, 1) for v, _ in variants]))
        return ys.mean(dim=0).numpy().transpose(1, 2, 0)          # torch.cat(...).mean(dim=0) in float32

    def evaluate(get):
        rows = []
        for i, it in enumerate(items):
            p, s = im.image_quality(get(i), it["gt"], metric, border)
            rows.append({"set": it["set"], "file": it["file"], "cond": it["cond"], "psnr": p, "ssim": s})
        return rows

    def agg(rows):
        out = {}
        for r in rows:
            out.setdefault(f"{r['set']}/{r['cond']}", []).append(r)
        return {k: {"n": len(v), "psnr": float(np.mean([r["psnr"] for r in v])),
                    "ssim": float(np.mean([r["ssim"] for r in v])) if v[0]["ssim"] is not None else None} for k, v in sorted(out.items())}

    res = {"model_id": args.model_id, "source_class": cls, "metric": metric, "crop_border": border, "n_images": len(items),
           "graph_at_artifact_size_equals_model_bfmodel": same, "sizes": len(groups), "stages": args.stages,
           "seconds": {"source_fp32": round(t_src, 1), "bitino_all_stages": round(t_bf, 1)},
           "source_fp32": {"single": agg(evaluate(lambda i: src[(i, "single")]))}, "bitino": {}}
    per_image = {"source_fp32_single": evaluate(lambda i: src[(i, "single")])}
    if args.self_ensemble:
        res["source_fp32"]["self_ensemble"] = agg(evaluate(lambda i: ensemble(src, i)))
    for st in args.stages:
        rows = evaluate(lambda i, st=st: bf[st][(i, "single")])
        per_image[f"bitino_{st}_single"] = rows
        res["bitino"][st] = {"single": agg(rows)}
        d = [np.abs(bf[st][(i, "single")].astype(np.float64) - src[(i, "single")]).max() for i in range(len(items))]
        res["bitino"][st]["max_abs_vs_source_fp32"] = float(max(d))
        if args.self_ensemble:
            res["bitino"][st]["self_ensemble"] = agg(evaluate(lambda i, st=st: ensemble(bf[st], i)))
    res["per_image"] = per_image
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=1)
    for k, v in res["source_fp32"]["single"].items():
        line = f"{args.model_id} {k}: source fp32 {v['psnr']:.4f}" + (f"/{v['ssim']:.4f}" if v["ssim"] is not None else "")
        for st in args.stages:
            b = res["bitino"][st]["single"][k]
            line += f" | bitino {st} {b['psnr']:.4f}" + (f"/{b['ssim']:.4f}" if b["ssim"] is not None else "")
        if args.self_ensemble:
            line += f" || ensemble: source {res['source_fp32']['self_ensemble'][k]['psnr']:.4f}" + "".join(
                f" bitino {st} {res['bitino'][st]['self_ensemble'][k]['psnr']:.4f}" for st in args.stages)
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
