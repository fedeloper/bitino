#!/usr/bin/env python3
"""First binarization / quantization disagreement between the official BHViT model and its IR,
both evaluated with float64 intermediates (BHViT analogue of tools/first_flip_analysis_torch.py).

Sites, in IR node order (adapter info["sites"]):
  bdense / bconv : value fed to torch.sign in QuantizeLinear / QuantizeConv2d, i.e. (h + move);
                   IR: BDense/BConv2D input minus its threshold (-move). |pre| = |h + move|.
  qkv_sign       : RPReLU output + move2 before torch.sign (Q, K, V). |pre| = |value|.
  attn_quant     : argument u = (3P)/alpha + zp of the 2-bit attention quantizer; a flip is a different
                   clamp(round(u)) level. |pre| = distance of u to the nearest rounding boundary k + 0.5.
A deviation is explained as rounding when the first flip happens at |pre| far below the typical |pre|
of that site (default rule: < --flip-rel x median). The same samples are also run with every float
constant of the IR replaced by its exact float64 value (analysis only); if that matches the float64
source to ~1e-12 with no flip, the conversion is semantically exact and all deviation comes from
storing the constants in float32.

  ~/anaconda3/envs/bhvit/bin/python tools/first_flip_bhvit.py --model-id bhvit-tiny-binary \
      --repo .cache/repos/IMRL_BHViT --weights .cache/checkpoints/bhvit-tiny-binary/best64.0.pth \
      --bfmodel artifacts/bhvit-tiny-binary/model_unfused.bitino --eval-dir .cache/data/eval128_bhvit \
      --all --json artifacts/bhvit-tiny-binary/first_flip.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters import bhvit  # noqa: E402

PREPROCESS = "bhvit"
_S: dict = {}


def _levels(u, qmin, qmax):
    return np.clip(np.rint(u), qmin, qmax)


def _dist(u):
    return np.abs(u - (np.floor(u) + 0.5))


def _analyze(idx: int) -> dict:
    import torch
    torch.set_num_threads(1)
    model, g, gx, sites, x_all = _S["model"], _S["g"], _S["gx"], _S["sites"], _S["x"]
    nodes = {n.name: n for n in g.nodes}
    mods = dict(model.named_modules())
    params = dict(model.named_parameters())
    cap: dict[str, np.ndarray] = {}
    hooks = []
    for s in sites:
        m = mods[s["torch_module"]]
        key = s["node"]
        if s["kind"] in ("bdense", "bconv"):
            hooks.append(m.register_forward_pre_hook(lambda _m, i, key=key: cap.__setitem__(key, i[0].detach().numpy())))
        elif s["kind"] == "qkv_sign":
            p = params[s["torch_add"]]
            hooks.append(m.register_forward_hook(lambda _m, i, o, key=key, p=p: cap.__setitem__(key, (o + p).detach().numpy())))
        else:  # attention quantizer: u = x/alpha_eff + zp_eff exactly as BinaryActivation_Attention.forward
            def qhook(qm, i, key=key):
                xx = i[0]
                qp = 2 ** (qm.nbits - 1) - 1
                gg = qm.num_head / np.sqrt(xx.numel() * qp)
                a = qm.grad_scale(qm.alpha, gg).reshape(1, -1, 1, 1)
                z = qm.grad_scale(qm.zero_point, gg).reshape(1, -1, 1, 1)
                cap[key] = (xx / a + z).detach().numpy()
            hooks.append(m.register_forward_pre_hook(qhook))
    x = x_all[idx:idx + 1].astype(np.float64)
    with torch.no_grad():
        lt = model(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))).logits.numpy()
    for h in hooks:
        h.remove()
    env = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    envx = executor.run(gx, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    rec = {"sample": idx, "logits_max_abs_diff": float(np.abs(env[g.outputs[0]].reshape(1, -1) - lt).max()),
           "exact_constants": {"logits_max_abs_diff": float(np.abs(envx[g.outputs[0]].reshape(1, -1) - lt).max()), "n_flips_all_sites": 0},
           "first_flip": None, "n_sites_with_flips": 0, "n_flips_total": 0, "max_abs_diff_before_first_flip": 0.0}
    for s in sites:
        n = nodes[s["node"]]
        tv = cap[s["node"]]
        if s["kind"] == "bconv":
            tv = tv.transpose(0, 2, 3, 1)
        tv = tv.reshape(-1)
        res = {}
        for tag, e in (("ir", env), ("x", envx)):
            if s["kind"] in ("bdense", "bconv"):
                pre = (e[n.inputs[0]].astype(np.float64) - e[n.inputs[4]].astype(np.float64)).reshape(-1)
                mode = n.attrs.get("input_binarize", "ge")
                b_ir = np.where(pre >= 0, 1, -1) if mode == "ge" else np.sign(pre)
                b_t = np.sign(tv)
                flips = b_ir != b_t
                abs_t, abs_ir = np.abs(tv), np.abs(pre)
            elif s["kind"] == "qkv_sign":
                pre = e[s["ir_input"]].astype(np.float64).reshape(-1)
                flips = np.sign(pre) != np.sign(tv)
                abs_t, abs_ir = np.abs(tv), np.abs(pre)
            else:
                a = n.attrs
                shp = [1, -1, 1, 1]
                u = (e[n.inputs[0]] * a.get("pre_scale", 1.0)) / e[n.inputs[1]].reshape(shp) + e[n.inputs[2]].reshape(shp)
                u = u.reshape(-1)
                flips = _levels(u, a["qmin"], a["qmax"]) != _levels(tv, a["qmin"], a["qmax"])
                abs_t, abs_ir = _dist(tv), _dist(u)
                pre = u
            res[tag] = (flips, abs_t, abs_ir, pre)
        fx = res["x"][0]
        rec["exact_constants"]["n_flips_all_sites"] += int(fx.sum())
        flips, abs_t, abs_ir, pre = res["ir"]
        if flips.any():
            rec["n_sites_with_flips"] += 1
            rec["n_flips_total"] += int(flips.sum())
            if rec["first_flip"] is None:
                rec["first_flip"] = {"node": n.name, "kind": s["kind"], "torch_module": s["torch_module"], "n_flips": int(flips.sum()),
                                     "n_values": int(flips.size), "max_abs_pre_at_flip_torch": float(abs_t[flips].max()),
                                     "max_abs_pre_at_flip_ir": float(abs_ir[flips].max()), "typical_abs_pre": float(np.median(abs_t)),
                                     "max_abs_diff_feeding_value": float(np.abs(pre - tv).max())}
        elif rec["first_flip"] is None:
            rec["max_abs_diff_before_first_flip"] = max(rec["max_abs_diff_before_first_flip"], float(np.abs(pre - tv).max()))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(bhvit.CONFIGS))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bfmodel", required=True, help="UNFUSED converted model")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--indices", help="comma-separated sample indices")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--flip-rel", type=float, default=1e-4)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    import torch
    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    indices = list(range(len(x))) if args.all or not args.indices else [int(i) for i in args.indices.split(",")]
    model, _ = bhvit.build_reference(args.repo, args.model_id, bhvit.load_checkpoint(args.weights))
    g = bfformat.read(args.bitino)
    binarize = next(n.attrs["input_binarize"] for n in g.nodes if n.op == "BDense")
    _, info = bhvit.convert(model, args.model_id, binarize=binarize)
    gx = bfformat.read(args.bitino)
    for name, v in info["float64_constants"].items():
        assert np.allclose(np.asarray(v, np.float32), gx.constants[name].data, rtol=2.0 ** -23, atol=0), name
        gx.constants[name].data = np.asarray(v, np.float64)
    order = {n.name: i for i, n in enumerate(g.nodes)}
    sites = sorted(info["sites"], key=lambda s: order[s["node"]])
    _S.update({"model": model.to(torch.float64), "g": g, "gx": gx, "sites": sites, "x": x})
    with mp.get_context("fork").Pool(args.jobs) as pool:
        results = pool.map(_analyze, indices, chunksize=1)
    for r in results:
        ff = r["first_flip"]
        r["explained_as_rounding"] = bool(ff is None or max(ff["max_abs_pre_at_flip_torch"], ff["max_abs_pre_at_flip_ir"]) < args.flip_rel * ff["typical_abs_pre"])
    flips = [r["first_flip"] for r in results if r["first_flip"]]
    kinds: dict[str, int] = {}
    for f in flips:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    summary = {"model_id": args.model_id, "bfmodel": os.path.basename(args.bitino), "n_samples_analyzed": len(results),
               "rule": f"first flip at |pre| < {args.flip_rel} x median |pre| of that site",
               "n_with_flip": len(flips), "first_flip_kinds": kinds,
               "n_explained_as_rounding": sum(r["explained_as_rounding"] for r in results),
               "max_abs_pre_at_first_flip_torch": max([f["max_abs_pre_at_flip_torch"] for f in flips], default=0.0),
               "max_abs_pre_at_first_flip_ir": max([f["max_abs_pre_at_flip_ir"] for f in flips], default=0.0),
               "max_ratio_pre_to_typical": max([max(f["max_abs_pre_at_flip_torch"], f["max_abs_pre_at_flip_ir"]) / f["typical_abs_pre"] for f in flips], default=0.0),
               "n_with_more_than_one_flip_at_first_site": sum(f["n_flips"] > 1 for f in flips),
               "logits_max_abs_diff_samples_without_flip": max([r["logits_max_abs_diff"] for r in results if r["first_flip"] is None], default=0.0),
               "exact_constants": {"logits_max_abs_diff": max(r["exact_constants"]["logits_max_abs_diff"] for r in results),
                                   "n_samples_with_flips": sum(r["exact_constants"]["n_flips_all_sites"] > 0 for r in results)},
               "samples": results}
    json.dump(summary, open(args.json, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=1))
    return 0 if summary["n_explained_as_rounding"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
