#!/usr/bin/env python3
"""First binarization disagreement between an official PyTorch binary CNN (BNext, RBONN, SNN,
IR-Net family) and its converted IR, both evaluated with float64 intermediates (generic version of
tools/first_flip_analysis_torch.py for the families in tools/convert_torch_cnn.py).

Node names of the converted graph are PyTorch module paths, and the node feeding every BConv2D
is the LearnableBias ChannelAffine (BNext "<unit>.move", RBONN "<block>.move0"). A forward
hook on that module captures the value handed to torch.sign. A flip is a position where the
sign differs (incl. exact zeros). Each sample is also evaluated with the folded constants
replaced by their float64 values (adapter info["float64_constants"]; never serialized).

IR-Net family (ReCU/RBNN/SiMaN/BiPer) and bitorch (QConv2d): binarization happens inside the conv
module, whose module path is the BConv2D node name. A forward pre-hook captures its input a; the value handed to
torch.sign is a (ReCU, BiPer, bitorch) or, up to a positive per-sample factor, a - mean_CHW(a) (RBNN, SiMaN).
bitorch binarizes with 0 -> +1, so its flips compare (a >= 0) instead of sign(a).
For families whose binary weights are derived from parameters, the float64 source runs with the
fp32-derived weights (same F.conv2d proxy as tools/convert_torch_cnn.py).

  python3 tools/first_flip_analysis_torch_cnn.py --family bnext --model-id bnext-tiny \\
      --repo .cache/repos/BNext --weights .cache/checkpoints/bnext-tiny/checkpoint.pth.tar \\
      --bfmodel artifacts/bnext-tiny/model_sign_zero_unfused.bitino \\
      --eval-dir .cache/data/eval128_timm_bicubic_crop095 --report artifacts/bnext-tiny/conversion_report.json \\
      --json artifacts/bnext-tiny/first_flip_analysis.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitino import executor, format as bfformat  # noqa: E402
from convert_torch_cnn import FAMILIES, _install_proxy, convert_graph, load_model  # noqa: E402

_S: dict = {}


def _analyze(idx: int) -> dict:
    import torch

    torch.set_num_threads(1)
    g, g_exact, model, bnodes, feeders, x_all = (_S[k] for k in ("g", "gx", "model", "bnodes", "feeders", "x"))
    captured: dict[str, np.ndarray] = {}
    modules = dict(model.named_modules())
    if _S["center"] is None:
        hooks = [modules[f].register_forward_hook(lambda m, i, o, f=f: captured.__setitem__(f, o.detach().numpy().transpose(0, 2, 3, 1)))
                 for f in feeders]
    else:   # IR-Net: value binarized inside the conv module
        def pre(m, i, f):
            a = i[0].detach()
            if _S["center"]:
                a = a - a.mean([1, 2, 3], keepdim=True)
            captured[f] = a.numpy().transpose(0, 2, 3, 1)
        hooks = [modules[f].register_forward_pre_hook(lambda m, i, f=f: pre(m, i, f)) for f in feeders]
    x = x_all[idx:idx + 1].astype(np.float64)
    with torch.no_grad():
        logits_t = model(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))).numpy()
    for h in hooks:
        h.remove()
    env = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    env_x = executor.run(g_exact, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    sgn = (lambda v: v >= 0) if _S["ge"] else np.sign
    flips_x = sum(int(np.sum(sgn(captured[f]) != sgn(env_x[n.inputs[0]]))) for n, f in zip(bnodes, feeders))
    rec = {"sample": idx, "first_flip_node": None,
           "logits_max_abs_diff": float(np.max(np.abs(env[g.outputs[0]].reshape(1, -1) - logits_t))),
           "top1_equal": bool(env[g.outputs[0]].argmax() == logits_t.argmax()),
           "exact_constants": {"logits_max_abs_diff": float(np.max(np.abs(env_x[g.outputs[0]].reshape(1, -1) - logits_t))),
                               "n_flips_all_bconv": flips_x}}
    for n, f in zip(bnodes, feeders):
        tv, iv = captured[f], env[n.inputs[0]]
        flips = sgn(tv) != sgn(iv)
        if flips.any():
            rec.update({"first_flip_node": n.name, "n_flips": int(flips.sum()), "n_binarized": int(flips.size),
                        "max_abs_pre_at_flip_torch": float(np.max(np.abs(tv[flips]))),
                        "max_abs_pre_at_flip_ir": float(np.max(np.abs(iv[flips]))),
                        "max_abs_diff_feeding_tensor": float(np.max(np.abs(tv - iv))),
                        "typical_abs_pre": float(np.median(np.abs(tv)))})
            break
        rec["max_abs_diff_before_first_flip"] = max(rec.get("max_abs_diff_before_first_flip", 0.0), float(np.max(np.abs(tv - iv))))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bfmodel", required=True, help="UNFUSED converted model (node names = module paths)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--indices", help="comma-separated sample indices (default: deviating samples from --report)")
    ap.add_argument("--report", help="conversion_report.json (to pick samples with semantic deviation > --tol)")
    ap.add_argument("--variant", default="sign_zero_unfused")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--all", action="store_true", help="analyze every sample of the eval set")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    import torch

    fam = FAMILIES[args.family]
    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][fam["preprocess"]]
    if args.all:
        indices = list(range(pre["shape"][0]))
    elif args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        rep = json.load(open(args.report))
        per = rep["variants"][args.variant]["semantic_check_ir_f64_vs_reference_f64"]["per_sample_max_abs"]
        indices = [i for i, v in enumerate(per) if v > args.tol]
    model, _cls, _ = load_model(args.family, args.repo, args.model_id, args.weights)
    g = bfformat.read(args.bitino)
    binarize = next(n.attrs["input_binarize"] for n in g.nodes if n.op == "BConv2D")
    _, info = convert_graph(args.family, model, args.model_id, binarize)
    x_all = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if fam.get("conv_module"):   # float64 source with the fp32-derived binary weights
        cap32, cap64 = {}, {}
        for dt, cap in ((torch.float32, cap32), (torch.float64, cap64)):
            st, remove = _install_proxy(model.to(dt), fam["conv_module"], capture=cap)
            with torch.no_grad():
                model(torch.from_numpy(np.ascontiguousarray(x_all[:1].transpose(0, 3, 1, 2))).to(dt))
            remove()
        override = {k: (v.double() if "irnet" in fam and not fam.get("scaled_weights") else torch.sign(v).double() * cap64[k].abs()) for k, v in cap32.items()}
        model.to(torch.float64)
        _install_proxy(model, fam["conv_module"], override=override)   # kept for the forked workers
    g_exact = bfformat.read(args.bitino)
    for name, v in info["float64_constants"].items():
        assert np.array_equal(v.astype(np.float32), g_exact.constants[name].data), f"{name}: bfmodel differs from adapter output"
        g_exact.constants[name].data = v
    producers = {o: n for n in g.nodes for o in n.outputs}
    bnodes = [n for n in g.nodes if n.op == "BConv2D"]
    irnet = "irnet" in fam
    inside = irnet or fam.get("conv_binarizes", False)
    _S.update(g=g, gx=g_exact, model=model.to(torch.float64), bnodes=bnodes,
              feeders=[n.name for n in bnodes] if inside else [producers[n.inputs[0]].name for n in bnodes],
              center=(fam["adapter"].FAMILIES[fam["irnet"]]["center"] if irnet else (False if inside else None)), x=x_all,
              ge=fam.get("binarize_modes") == ("ge",))
    with mp.get_context("fork").Pool(max(1, min(args.jobs, len(indices)))) as pool:
        results = pool.map(_analyze, indices, chunksize=1)
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "exact_constants"}))
    summary = {"model_id": args.model_id, "bfmodel": os.path.basename(args.bitino), "n_samples_analyzed": len(results),
               "n_with_flip": sum(r["first_flip_node"] is not None for r in results),
               "n_top1_changes": sum(not r["top1_equal"] for r in results),
               "n_with_more_than_one_flip_at_first_flip_node": sum(r.get("n_flips", 0) > 1 for r in results),
               "max_abs_pre_at_first_flip": max([r.get("max_abs_pre_at_flip_torch", 0.0) for r in results], default=0.0),
               "max_abs_pre_at_first_flip_ir": max([r.get("max_abs_pre_at_flip_ir", 0.0) for r in results], default=0.0),
               "min_typical_abs_pre_at_first_flip": min([r["typical_abs_pre"] for r in results if "typical_abs_pre" in r], default=0.0),
               "logits_max_abs_diff_samples_without_flip": max([r["logits_max_abs_diff"] for r in results if r["first_flip_node"] is None], default=0.0),
               "exact_constants": {"logits_max_abs_diff": max([r["exact_constants"]["logits_max_abs_diff"] for r in results], default=0.0),
                                   "n_samples_with_flips": sum(r["exact_constants"]["n_flips_all_bconv"] > 0 for r in results)},
               "samples": results}
    json.dump(summary, open(args.json, "w"), indent=1)
    print(f"analyzed {len(results)} samples; with a flip: {summary['n_with_flip']} (top-1 changes {summary['n_top1_changes']}); "
          f"max |pre| at first flip: {summary['max_abs_pre_at_first_flip']:.3e} (IR {summary['max_abs_pre_at_first_flip_ir']:.3e}); "
          f"with exact float64 constants: max |dlogit| {summary['exact_constants']['logits_max_abs_diff']:.3e}, "
          f"samples with flips {summary['exact_constants']['n_samples_with_flips']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
