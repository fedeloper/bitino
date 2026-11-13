#!/usr/bin/env python3
"""Builds the full-integer model (stage `int8_full`) of a converted float model.

    python3 tools/quantize_int8_full.py --model artifacts/bi-vit-deit-tiny/model.bitino \\
        --calib-dir .cache/data/calib512_imagenet100train --preprocess bhvit --n 256 \\
        --out artifacts/bi-vit-deit-tiny/model_int8_full.bitino

Calibration runs the float32 numpy executor (bit-identical to the exact Bitino stage) on the first --n
records of a calibration set that is disjoint from every evaluation set (tools/build_eval_set.py
--split train records the disjointness check in its manifest). Ranges are min/max, or percentiles with
--percentile. The quantization parameters live in the node attributes of the I8* operators
(docs/int8.md "Full-integer execution: int8_full"); the manifest records the calibration source, the
sample ids, the options, the rewrites and the per-tensor parameters (`manifest.quantization`).
A JSON report (--report, default <out>.report.json) lists the int8 operators and anything left in float.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as F, quantize_int8 as QI, quantize_l8 as QL  # noqa: E402


def load_records(calib_dir: str, preprocess: str, n: int, g) -> tuple[list[bytes], dict]:
    man = json.load(open(os.path.join(calib_dir, "manifest.json")))
    pre = man["preprocessing"][preprocess]
    raw = open(os.path.join(calib_dir, pre["file"]), "rb").read()
    per = sum(int(np.prod(i["shape"])) * 4 for i in g.inputs)
    total = len(raw) // per
    if len(raw) % per:
        raise SystemExit(f"{pre['file']}: size {len(raw)} is not a multiple of the record size {per}")
    n = min(n, total)
    return [raw[i * per:(i + 1) * per] for i in range(n)], man


def sample_ids(man: dict, n: int) -> list:
    out = []
    for s in man.get("samples", [])[:n]:
        out.append(s.get("file") or s.get("guid") or s.get("id") or s.get("index"))
    return out


def pc_stem_stats(g, records, stats, percentile, which="input"):
    """Per-channel calibration of ReLU stem convolutions (Conv2D reading a graph input; which="pooled": the ReLU
    Conv2D before the first max pool instead, e.g. Larq MeliusNet's third stem convolution) and the max pool after them:
    stats[conv out]["pc_hi"][c] = max over the two tensors of the channel's percentile (or max) over the same records.
    Only the stem nodes run (float numpy executor)."""
    from bitino.ir import Graph
    ins = {i["name"] for i in g.inputs}
    cons = {}
    for n in g.nodes:
        for t in n.inputs:
            cons.setdefault(t, []).append(n)
    first_pool = next((m for m in g.nodes if m.op == "MaxPool2D"), None)
    for k, n in enumerate(g.nodes):
        if n.op != "Conv2D" or n.attrs.get("activation", "none") != "relu":
            continue
        if which == "input" and n.inputs[0] not in ins:
            continue
        if which == "pooled" and (first_pool is None or first_pool.inputs[0] != n.outputs[0]):
            continue
        y = n.outputs[0]
        nodes = [n] + [u for u in cons.get(y, []) if u.op == "MaxPool2D"]
        outs = [m.outputs[0] for m in nodes]
        pre = []   # the stem nodes before n (pooled: the convolutions between the input and n)
        if n.inputs[0] not in ins:
            need = {n.inputs[0]}
            for m in reversed(g.nodes[:k]):
                if m.outputs[0] in need:
                    pre.insert(0, m)
                    need.update(t for t in m.inputs if t and t not in g.constants and t not in ins)
        sub = Graph()
        sub.inputs = [dict(i) for i in g.inputs]
        nodes = pre + nodes
        sub.nodes = nodes
        sub.constants = {c: v for c, v in g.constants.items() if any(c in m.inputs for m in nodes)}
        sub.outputs = outs
        sub.manifest = {}
        vals = {t: [] for t in sub.outputs}
        for r in records:
            env = executor.run(sub, QI._feeds(g, r))
            for t in sub.outputs:
                v = np.asarray(env[t], np.float32)
                vals[t].append(v.reshape(-1, v.shape[-1]))
        hi = None
        for t in sub.outputs:
            v = np.concatenate(vals[t])
            h = np.percentile(v, percentile, axis=0) if percentile else v.max(axis=0)
            hi = h if hi is None else np.maximum(hi, h)
        stats.setdefault(y, {})["pc_hi"] = [float(x) for x in hi]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--preprocess", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--percentile", type=float, help="e.g. 99.99: clip ranges to the (100-p, p) percentiles of each tensor")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--no-split", action="store_true", help="no binarize-aware second version of mixed tensors")
    ap.add_argument("--no-monotone", action="store_true", help="no R3b (binary layers reading through monotone chains)")
    ap.add_argument("--no-affine-trees", action="store_true", help="no R7 (Add/ChannelAffine trees as one affine sum)")
    ap.add_argument("--stats", help="reuse calibration statistics (JSON written by a previous run as <out>.stats.json)")
    ap.add_argument("--residual-code", choices=("linear", "l8", "lin8"), default="linear",
                    help="l8: 8-bit log2 residual stream with fused I8BConv2DRes blocks (python/bitino/quantize_l8.py); lin8: "
                         "int8_full v2, the same fused blocks writing the int8_full linear grids (I8BConv2DRes output i8; "
                         "model_int8_full_v2.bitino, docs/int8.md)")
    ap.add_argument("--pool-grid-fix", dest="pool_grid_fix", action="store_true", default=True,
                    help="(default) pass-through ops after AvgPool2D / Pad get the pool's actual (input) grid (fixes a stale-grid "
                         "bug of the Larq heads / SE gates; quantize_int8._Emitter.assign_params)")
    ap.add_argument("--no-pool-grid-fix", dest="pool_grid_fix", action="store_false",
                    help="the int8_full models shipped before 2026-11-04: a Reshape after an AvgPool2D keeps the pool output's "
                         "calibrated grid (wrong in 13 of 87 models, tools/int8_full_grid_check.py)")
    ap.add_argument("--l8-range", default="minmax", help="l8 exponent rule: minmax (default), lohi (the percentile range), "
                    "minmax-k / lohi-k (k octaves lower; quantize_l8.l8_exp)")
    ap.add_argument("--stem", choices=("split", "l8", "bits", "l8_bits"), default="l8",
                    help="with --residual-code l8: how the block after the stem convolution binarizes it (split: binarize-aware "
                         "second GEMM; l8: l8 stem output, binarize the code; bits: bit output of the stem GEMM, max pooled as "
                         "bits; l8_bits: both; docs/int8.md)")
    ap.add_argument("--concat-parts", nargs="?", const="on", choices=("on", "sum"),
                    help="per-part grids of channel Concat trees whose readers absorb per-channel grids (DenseNet / MeliusNet; "
                         "quantize_int8._Emitter.plan_parts; sum: MeliusNet improvement slices on the sum of the input ranges)")
    ap.add_argument("--bin-margin", choices=("minmax", "lohi", "thr"), default="minmax",
                    help="the 2^-20 data-range margin of binarize-aware grids: from the calibration min/max (default), from the "
                         "percentile range, or only where the thresholds give none (thr; quantize_int8._Emitter.qbin)")
    ap.add_argument("--bin-relu-clip", action="store_true",
                    help="binarize-aware grids of ReLU outputs ignore thresholds below 0 (constant binarizations; "
                         "quantize_int8._Emitter.qbin)")
    ap.add_argument("--center-split", action="store_true",
                    help="per-sample mean centering x - mean(x) before a binarization (siman / RBNN): the centering add reads a "
                         "second version of x on a grid around the calibrated means (quantize_int8._Emitter.direct_bin_thresholds)")
    ap.add_argument("--maxpool-range", action="store_true",
                    help="a tensor read only by a max pool gets the union of its range and the pool output's (quantize_int8)")
    ap.add_argument("--stem-minmax", action="store_true",
                    help="convolutions reading the graph input use the calibration min / max range (quantize_int8)")
    ap.add_argument("--pc-stem", nargs="?", const="input", choices=("input", "pooled"),
                    help="per-channel output grids of a ReLU stem convolution (and its max pool) from a per-channel calibration "
                         "of the same samples; their readers absorb them (quantize_int8._Emitter.plan_parts). input (default): the "
                         "convolutions reading the graph input; pooled: the ReLU convolution before the first max pool (stems of "
                         "several convolutions, Larq MeliusNet)")
    ap.add_argument("--pc-slices", nargs="?", const="channel", choices=("channel", "slice", "channel_minmax", "slice_minmax"),
                    help="with --concat-parts: MeliusNet improvement slices Add(x, Pad(b)) keep grids from a calibration of the "
                         "updated slice alone, per channel (channel, default) or one per slice (slice), percentile range or (_minmax) the "
                         "calibration min / max, instead of the whole Add "
                         "output's grid (same records; pc_slice_stats, quantize_int8._Emitter.plan_parts)")
    ap.add_argument("--bin-clip-far", action="store_true",
                    help="binarize-aware grids ignore thresholds more than twice the calibrated span beyond the calibrated range "
                         "(quantize_int8._Emitter.qbin)")
    ap.add_argument("--pc-bin", action="store_true",
                    help="per-channel binarize-aware grids [0, max threshold] of ReLU binary convolutions read only by "
                         "binarizations (BNAS cells; quantize_int8._Emitter.plan_parts)")
    ap.add_argument("--identity-blocks", choices=("plain", "all", "off"), default="plain",
                    help="with --residual-code l8: also fuse BConv2D -> Add blocks without an activation chain (Bi-Real / ResNet-E "
                         "style; plain: sums with no unary reader, all: also sums read by an unfusable unary chain; "
                         "quantize_l8.fuse)")
    ap.add_argument("--res-pre", action="store_true",
                    help="with --residual-code l8: also fuse blocks whose binary convolution has a ReLU (QuickNet BConv[relu] -> "
                         "BatchNorm -> Add): I8BConv2DRes pre stage (quantize_l8.fuse, int8_ops.bconv_res_pre); off by default")
    ap.add_argument("--dense", action="store_true",
                    help="with --residual-code lin8: token-wise BDense residual blocks (BHViT) as 1x1 I8BConv2DResI "
                         "(quantize_l8.fuse_dense)")
    ap.add_argument("--gated", action="store_true",
                    help="with --residual-code lin8: BNext blocks, BConv -> PReLU as I8BConv2DResI and the squeeze-excitation "
                         "gated residual add as I8GatedAdd (quantize_l8.fuse_gated)")
    ap.add_argument("--int-epilogue", action="store_true",
                    help="with --residual-code lin8: fused blocks without a pre stage use the integer epilogue (I8BConv2DResI, "
                         "int8_ops._bconv_res_i) instead of the fp32 one")
    a = ap.parse_args()

    g = F.read(a.model)
    man = json.load(open(os.path.join(a.calib_dir, "manifest.json")))
    if man.get("split") != "train" and man.get("purpose") != "calibration":
        raise SystemExit(f"{a.calib_dir} is not a calibration set (manifest split/purpose); refusing to calibrate on eval data")
    records, man = load_records(a.calib_dir, a.preprocess, a.n, g)
    t0 = time.time()
    stats = json.load(open(a.stats)) if a.stats else None
    if stats is not None and any(t not in stats for t in QI._act_names(g) if t in {o for n in g.nodes for o in n.outputs}):
        print(f"{a.stats}: does not cover this model's tensors (model changed); recalibrating", file=sys.stderr)
        stats = None
    if stats is None:
        stats = QI.calibrate(os.path.abspath(a.model), records, a.percentile, a.jobs)
        json.dump(stats, open(a.out + ".stats.json", "w"))
    if a.pc_stem:
        pc_stem_stats(g, records, stats, a.percentile, a.pc_stem)
    if a.pc_slices:
        QI.pc_slice_stats(g, records, stats, a.percentile)
    t_cal = time.time() - t0
    shapes = QI._float_shapes(g, QI._feeds(g, records[0]))
    qopts = {"split_mixed": not a.no_split, "monotone_bypass": not a.no_monotone, "affine_trees": not a.no_affine_trees}
    qopts["pool_grid_fix"] = bool(a.pool_grid_fix)
    if a.concat_parts:
        qopts["concat_parts"] = True if a.concat_parts == "on" else a.concat_parts
    if a.bin_margin != "minmax":
        qopts["bin_margin"] = a.bin_margin
    if a.bin_relu_clip:
        qopts["bin_relu_clip"] = True
    if a.center_split:
        qopts["center_split"] = True
    if a.maxpool_range:
        qopts["maxpool_range"] = True
    if a.stem_minmax:
        qopts["stem_minmax"] = True
    if a.pc_stem:
        qopts["pc_stem"] = True if a.pc_stem == "input" else a.pc_stem
    if a.pc_slices:
        qopts["pc_slices"] = a.pc_slices
    if a.bin_clip_far:
        qopts["bin_clip_far"] = True
    if a.pc_bin:
        qopts["pc_bin"] = True
    Q = QI.rewrite(g, stats, shapes, qopts)
    E = QI._Emitter(Q, qopts)
    qg, rep = E.emit()  # == QI.quantize(g, stats, shapes, qopts)
    l8rep = None
    if a.residual_code in ("l8", "lin8"):
        l8o = dict(qopts, l8_range=a.l8_range, stem=a.stem, identity_blocks=a.identity_blocks)
        if a.residual_code == "lin8":
            l8o["code"] = "lin8"
            if a.int_epilogue:
                l8o["int_epilogue"] = True
        if a.res_pre:
            l8o["res_pre"] = True
        qg, l8rep = QL.fuse(qg, Q, E, l8o)
        if a.residual_code == "lin8" and a.dense:   # BHViT: token-wise BDense residual blocks as 1x1 I8BConv2DResI
            qg, drep = QL.fuse_dense(qg, Q, E, l8o)
            l8rep["dense"] = {k: v for k, v in drep.items() if k != "int8_ops"}
            if drep.get("int8_ops"):
                l8rep["int8_ops"] = drep["int8_ops"]
        if a.residual_code == "lin8" and a.gated:   # BNext: PReLU binary convolutions and gated residual adds
            qg, grep_ = QL.fuse_gated(qg, Q, E, l8o)
            l8rep["gated"] = {k: v for k, v in grep_.items() if k != "int8_ops"}
            if grep_.get("int8_ops"):
                l8rep["int8_ops"] = grep_["int8_ops"]
        if a.residual_code == "lin8":   # common subexpressions (BHViT's repeated rolls)
            qg, crep = QL.cse(qg)
            l8rep["cse_removed"] = crep["cse_removed"]
            if crep["cse_removed"]:
                from collections import Counter
                l8rep["int8_ops"] = dict(Counter(n.op for n in qg.nodes if n.op.startswith("I8")))
        if not l8rep.get("fused_blocks") and not l8rep.get("gated", {}).get("prelu_convs") and \
                not l8rep.get("dense", {}).get("dense_blocks"):
            # e.g. MeliusNet: no BConv2D -> Add(residual) block (the improvement adds are padded)
            raise SystemExit("--residual-code l8: no residual block to fuse (quantize_l8.fuse: fused_blocks 0); no file written")
        if "int8_ops" not in l8rep:
            l8rep["int8_ops"] = rep["int8_ops"]
        if a.residual_code == "l8":
            l8rep["l8_range"] = a.l8_range
        if a.int_epilogue:
            l8rep["int_epilogue_option"] = True
        l8rep["identity_blocks"] = a.identity_blocks
        if a.res_pre:
            l8rep["res_pre"] = True
        rep["int8_ops"] = l8rep["int8_ops"]
    pre = man["preprocessing"][a.preprocess]
    quant = {
        "scheme": "int8_full v1: per-tensor asymmetric int8 activations (static, calibrated), per-output-channel symmetric int8 "
                  "weights, int32 accumulators, fixed-point requantization (SQRDMULH + SRSHL rounding); binary operators with "
                  "exact integer binarization thresholds and fp32 epilogues rounded half-to-even (docs/int8.md)",
        "source_model": {"path": os.path.relpath(os.path.abspath(a.model), ROOT), "sha256": hashlib.sha256(open(a.model, "rb").read()).hexdigest()},
        "calibration": {"set": man.get("name"), "dir": os.path.relpath(os.path.abspath(a.calib_dir), ROOT), "split": man.get("split"),
                        "purpose": man.get("purpose"), "preprocessing": a.preprocess, "tensor_sha256": pre.get("sha256"),
                        "n": len(records), "sample_ids": sample_ids(man, len(records)), "disjointness": man.get("disjointness"),
                        "range": f"percentile {a.percentile}" if a.percentile else "min/max", "seconds": round(t_cal, 1)},
        "options": qopts,
        "rewrites": rep["rewrites"], "notes": rep["notes"], "int8_ops": rep["int8_ops"], "left_float": rep["left_float"],
        "binarize_aware_tensors": rep["binarize_aware_tensors"],
        "tensor_params": rep["tensor_params"],
        **({"residual_code": {"code": "l8", "scheme": "8-bit log2 residual stream (sign, 0, 2^(E + (j-127)/8)), fused "
                              "I8BConv2DRes blocks with bit outputs; docs/int8.md", **{k: v for k, v in l8rep.items() if k != "int8_ops"}}}
           if l8rep and a.residual_code == "l8" else {}),
        **({"residual_code": {"code": "lin8", "scheme": "int8_full v2: fused I8BConv2DRes blocks writing the int8_full linear grid "
                              "of the block output (output i8: clamp(rne(w) + y_zero)), linear residual codes, bit outputs of the "
                              "unrounded sums; docs/int8.md", **{k: v for k, v in l8rep.items() if k != "int8_ops"}}}
           if l8rep and a.residual_code == "lin8" else {}),
        "date": datetime.date.today().isoformat(),
    }
    qg.manifest = dict(qg.manifest)
    qg.manifest["variant"] = {"linear": "int8_full", "l8": "int8_full_l8", "lin8": "int8_full_v2"}[a.residual_code]
    qg.manifest["quantization"] = quant
    sha = F.write(qg, a.out)
    # sanity: the quantized graph runs in the numpy oracle and agrees with the float model on a calibration record
    feeds = QI._feeds(g, records[0])
    yf = executor.run(g, feeds)[g.outputs[0]].reshape(-1)
    yq = executor.run(F.read(a.out), feeds)[qg.outputs[0]].reshape(-1)
    report = {k: v for k, v in quant.items() if k != "tensor_params"}
    report.update(out=os.path.relpath(os.path.abspath(a.out), ROOT), out_sha256=sha, nodes=len(qg.nodes), float_nodes=len(g.nodes),
                  sanity={"argmax_float": int(yf.argmax()), "argmax_int8": int(yq.argmax()),
                          "max_abs_diff": float(np.abs(yf - yq).max())})
    rp = a.report or a.out + ".report.json"
    json.dump(report, open(rp, "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("rewrites", "int8_ops", "left_float", "nodes", "float_nodes", "sanity")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
