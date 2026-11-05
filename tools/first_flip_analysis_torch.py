#!/usr/bin/env python3
"""Locates the FIRST binarization disagreement between the official AB-BNN PyTorch model and
its converted IR, both evaluated with float64 intermediates (torch analogue of
tools/first_flip_analysis.py).

For each sample, walks the BConv2D nodes in topological order and compares the value fed
to binarization: BasicBlock.move0 output (x + bias/beta) in PyTorch vs the IR tensor named
"<block>.move0" (node names of the converted graph are the PyTorch module paths). A flip is
a position where torch.sign differs (incl. exact zeros). If the flipped positions have
|pre-binarization value| at float32-constant rounding scale (<< typical |value|), the
downstream divergence is a rounding cascade, not a semantic conversion error: the IR
stores folded constants (standardized weights, scales, bias/beta) in float32.

The same samples are also evaluated with the IR's folded constants replaced by their exact
float64 values (adapter info["float64_constants"]; this graph is never serialized). If that
IR matches the float64 source to ~1e-9 with no flip, the conversion is semantically exact
and the residual deviation is entirely due to float32 constant storage.

  python3 tools/first_flip_analysis_torch.py --model-id ab-bnn-imagenet-reactnet18 \
      --repo .cache/repos/AB-BNN --weights .cache/checkpoints/ab-bnn/imagenet_reactnet18_step2.h5 \
      --bfmodel artifacts/ab-bnn-imagenet-reactnet18/model_sign_zero_unfused.bitino \
      --eval-dir .cache/data/eval128 --report artifacts/ab-bnn-imagenet-reactnet18/conversion_report.json \
      --variant sign_zero_unfused --json artifacts/ab-bnn-imagenet-reactnet18/first_flip_analysis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters import torch_abbnn  # noqa: E402
from convert_abbnn import ARCHS, load_model, preprocess_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(ARCHS))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bfmodel", required=True, help="UNFUSED converted model (node names = module paths)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--indices", help="comma-separated sample indices (default: deviating samples from --report)")
    ap.add_argument("--report", help="conversion_report.json (to pick samples with semantic deviation > --tol)")
    ap.add_argument("--variant", default="sign_zero_unfused")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--all", action="store_true", help="analyze every sample of the eval set")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    import torch

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][preprocess_of(args.model_id)]
    if args.all:
        indices = list(range(pre["shape"][0]))
    elif args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        rep = json.load(open(args.report))
        per = rep["variants"][args.variant]["semantic_check_ir_f64_vs_reference_f64"]["per_sample_max_abs"]
        indices = [i for i, v in enumerate(per) if v > args.tol]
    model, _cls, _ = load_model(args.repo, args.model_id, args.weights)
    g = bfformat.read(args.bitino)
    binarize = next(n.attrs["input_binarize"] for n in g.nodes if n.op == "BConv2D")
    _, info = torch_abbnn.convert(model, args.model_id, binarize=binarize, input_hw=tuple(pre["shape"][1:3]))
    g_exact = bfformat.read(args.bitino)
    for name, v in info["float64_constants"].items():
        assert np.array_equal(v.astype(np.float32), g_exact.constants[name].data), f"{name}: bfmodel differs from adapter output"
        g_exact.constants[name].data = v
    model = model.to(torch.float64)
    producers = {o: n for n in g.nodes for o in n.outputs}
    bnodes = [n for n in g.nodes if n.op == "BConv2D"]
    feeders = [producers[n.inputs[0]].name for n in bnodes]          # "<block>.move0" (ReActNet-A: move11 / move21)
    modules = dict(model.named_modules())
    captured: dict[str, np.ndarray] = {}
    for f in feeders:
        modules[f].register_forward_hook(lambda m, i, o, f=f: captured.__setitem__(f, o.detach().numpy().transpose(0, 2, 3, 1)))

    x_all = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    results = []
    for idx in indices:
        x = x_all[idx:idx + 1].astype(np.float64)
        captured.clear()
        with torch.no_grad():
            logits_t = model(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))).numpy()
        env = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        env_x = executor.run(g_exact, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        flips_x = sum(int(np.sum(np.sign(captured[f]) != np.sign(env_x[n.inputs[0]]))) for n, f in zip(bnodes, feeders))
        rec = {"sample": idx, "first_flip_node": None,
               "logits_max_abs_diff": float(np.max(np.abs(env[g.outputs[0]].reshape(1, -1) - logits_t))),
               "exact_constants": {"logits_max_abs_diff": float(np.max(np.abs(env_x[g.outputs[0]].reshape(1, -1) - logits_t))),
                                   "n_flips_all_bconv": flips_x}}
        for n, f in zip(bnodes, feeders):
            tv, iv = captured[f], env[n.inputs[0]]
            flips = np.sign(tv) != np.sign(iv)
            if flips.any():
                rec.update({"first_flip_node": n.name, "n_flips": int(flips.sum()), "n_binarized": int(flips.size),
                            "max_abs_pre_at_flip_torch": float(np.max(np.abs(tv[flips]))),
                            "max_abs_pre_at_flip_ir": float(np.max(np.abs(iv[flips]))),
                            "max_abs_diff_feeding_tensor": float(np.max(np.abs(tv - iv))),
                            "typical_abs_pre": float(np.median(np.abs(tv)))})
                break
            rec.setdefault("max_abs_diff_before_first_flip", 0.0)
            rec["max_abs_diff_before_first_flip"] = max(rec["max_abs_diff_before_first_flip"], float(np.max(np.abs(tv - iv))))
        results.append(rec)
        print(json.dumps(rec))
    summary = {"model_id": args.model_id, "bfmodel": os.path.basename(args.bitino), "n_samples_analyzed": len(results),
               "n_with_flip": sum(r["first_flip_node"] is not None for r in results),
               "n_with_more_than_one_flip_at_first_flip_node": sum(r.get("n_flips", 0) > 1 for r in results),
               "max_abs_pre_at_first_flip": max([r.get("max_abs_pre_at_flip_torch", 0.0) for r in results], default=0.0),
               "max_abs_pre_at_first_flip_ir": max([r.get("max_abs_pre_at_flip_ir", 0.0) for r in results], default=0.0),
               "logits_max_abs_diff_samples_without_flip": max([r["logits_max_abs_diff"] for r in results if r["first_flip_node"] is None], default=0.0),
               "exact_constants": {"logits_max_abs_diff": max([r["exact_constants"]["logits_max_abs_diff"] for r in results], default=0.0),
                                   "n_samples_with_flips": sum(r["exact_constants"]["n_flips_all_bconv"] > 0 for r in results)},
               "samples": results}
    json.dump(summary, open(args.json, "w"), indent=1)
    print(f"analyzed {len(results)} samples; with a flip: {summary['n_with_flip']}; "
          f"max |pre| at first flip: {summary['max_abs_pre_at_first_flip']:.3e} (IR {summary['max_abs_pre_at_first_flip_ir']:.3e}); "
          f"with exact float64 constants: max |dlogit| {summary['exact_constants']['logits_max_abs_diff']:.3e}, "
          f"samples with flips {summary['exact_constants']['n_samples_with_flips']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
