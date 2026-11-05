#!/usr/bin/env python3
"""Convert the S10 binary DeiT checkpoints (Bi-ViT tiny/small, GSB DeiT-S CIFAR-100) to .bitino and run the
Level-3 float64 semantic check.

Runs with the base python (torch 2.6, timm 1.0; CPU). The official model code is imported read-only from a
local clone; checkpoints are loaded with torch.load(weights_only=True) (argparse.Namespace allow-listed).

  python3 -I tools/convert_bivit.py --model-id bi-vit-deit-tiny --repo .cache/repos/YanjingLi0202_Bi-ViT \
      --weights .cache/checkpoints/bi-vit-deit-tiny/best_checkpoint_tiny.pth \
      --eval-dir .cache/data/eval128_bhvit --out-dir artifacts/bi-vit-deit-tiny
  python3 -I tools/convert_bivit.py --model-id gsb-deit-s-cifar100-stage2 --repo .cache/repos/IMRL_GSB-Vision-Transformer \
      --weights .cache/checkpoints/gsb-deit-s-cifar100-stage2/1W1A.pth \
      --eval-dir .cache/data/eval128_cifar100_deit --out-dir artifacts/gsb-deit-s-cifar100-stage2

Produces in --out-dir:
  model.bitino / model_unfused.bitino   input_binarize 'ge' at the BDense inputs fed by a LayerNorm (deployable)
  model_sign_zero.bitino                 'sign_zero' there too (exact torch.sign; ternary BDense kernels)
  reference_logits.f32                    official model, float32, batch size 1, as released
  reference_logits_f64.f64                official model in float64 with the fp32-derived quantized weights
                                          substituted at F.linear / F.conv2d (primary semantic reference)
  reference_logits_f64_naive.f64          official model .double() as is
  numpy_logits_<variant>.f32              numpy executor, float32 (numpy_logits_fused.f32 = model.bitino)
  conversion_report.json                  float64 semantic check per variant, float64-constants check, first
                                          binarization difference of every deviating sample, tie statistics
--drop-gelu-before-sign (Bi-ViT): variant without the MLP GELU (passes.drop_gelu_before_sign, the same graph as
tools/derive_variant.py on model.bitino); every file above then describes that variant and the references come from the
official model with the GELU replaced by the identity. Use a variant --out-dir (artifacts/<model>/no_gelu).
Batch size 1 everywhere: Bi-ViT's grad_scale() forward value and GSB's whole-tensor max/min thresholds
depend on the batch (GSB is evaluated per sample, i.e. the deployable semantics; see the report).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # numpy workers: one BLAS thread
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import torch_bivit, torch_gsb  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

FAMILIES = {
    "bivit": {"adapter": torch_bivit, "preprocess": "bhvit", "repo_url": "https://github.com/YanjingLi0202/Bi-ViT",
              "sources": ("bi_vision_transformer.py", "Bi_Quant.py", "_binary_base_plus.py", "main_1bit.py", "datasets.py"),
              "class": "bi_vision_transformer.twobits_VisionTransformer (create_model bi_deit_*_patch16_224)"},
    "gsb": {"adapter": torch_gsb, "preprocess": "cifar100_deit", "repo_url": "https://github.com/IMRL/GSB-Vision-Transformer",
            "sources": ("One_bit_vit_new.py", "quant_new_new.py", "test.py", "datasets.py", "weight_init.py"),
            "class": "One_bit_vit_new.lowbit_VisionTransformer (onebits_deit_small_patch16_224, test.py OQ_VitConfig)"},
}


def family_of(model_id: str) -> str:
    if model_id in torch_bivit.CONFIGS:
        return "bivit"
    if model_id in torch_gsb.CONFIGS:
        return "gsb"
    raise KeyError(model_id)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit(repo: str) -> str:
    head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    p = os.path.join(repo, ".git", ref)
    if os.path.exists(p):
        return open(p).read().strip()
    for line in open(os.path.join(repo, ".git", "packed-refs")):
        if line.strip().endswith(ref):
            return line.split()[0]
    return "unknown"


def run_reference(model, x_nhwc: np.ndarray, dtype) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for i in range(len(x_nhwc)):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2))).to(dtype)
            out.append(model(xb).double().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------- numpy executor workers
_G: dict = {}
_SITE_OPS = ("BDense", "BConv2D", "BMatMul")


def _init_worker(paths: dict, exact_key: str, f64_consts: dict) -> None:
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
    g = bfformat.read(paths[exact_key])
    for name, v in f64_consts.items():
        assert np.allclose(np.asarray(v, np.float32), g.constants[name].data, rtol=2.0 ** -20, atol=0) or \
            np.array_equal(np.isinf(v), np.isinf(g.constants[name].data)), f"{name}: bfmodel differs from adapter"
        g.constants[name].data = np.asarray(v, np.float64)
    _G["__exact__"] = g


def _site_values(g, env):
    """Binarization sites in node order: (node, input index, value relative to its threshold, mode)."""
    out = []
    for n in g.nodes:
        if n.op in ("BDense", "BConv2D"):
            x = env[n.inputs[0]].astype(np.float64)
            t = env[n.inputs[4]].astype(np.float64) if len(n.inputs) > 4 and n.inputs[4] else 0.0
            out.append((n.name, 0, x - t, n.attrs.get("input_binarize", "ge")))
        elif n.op == "BMatMul":
            for j, key in ((0, "a_binarize"), (1, "b_binarize")):
                if n.inputs[j] not in g.constants:
                    out.append((n.name, j, env[n.inputs[j]].astype(np.float64), n.attrs.get(key, "sign_zero")))
    return out


def _bin(v, mode):
    if mode == "sign_zero":
        return np.sign(v)
    if mode in ("gt", "gt0"):
        return v > 0
    return v >= 0


def _ties(g, env) -> dict:
    """Exact zeros at sign_zero / gt sites (where torch.sign gives 0 or a tie decides) and at 'ge' sites."""
    z = {}
    for name, j, v, mode in _site_values(g, env):
        k = f"{mode}"
        d = z.setdefault(k, [0, 0])
        d[0] += int(np.sum(v == 0))
        d[1] += int(v.size)
    return z


def _first_diff(g, env_a, env_b) -> dict | None:
    """First binarization site whose binarized values differ between two runs of the same graph."""
    sa, sb = _site_values(g, env_a), _site_values(g, env_b)
    for (na, ja, va, mode), (_, _, vb, _) in zip(sa, sb):
        if va.shape != vb.shape:
            continue
        flips = _bin(va, mode) != _bin(vb, mode)
        if flips.any():
            return {"node": na, "operand": ja, "mode": mode, "n_flips": int(flips.sum()), "n_values": int(flips.size),
                    "max_abs_pre_a": float(np.abs(va[flips]).max()), "max_abs_pre_b": float(np.abs(vb[flips]).max()),
                    "typical_abs_pre": float(np.median(np.abs(va)))}
    return None


def _run_one(job):
    key, i, x = job
    g = _G[key]
    env32 = executor.run(g, {"input": x}, keep_intermediates=True)
    out32 = env32[g.outputs[0]].reshape(-1)
    t32 = _ties(g, env32)
    del env32
    env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    out64 = env64[g.outputs[0]].reshape(-1)
    outx, ff = None, None
    if key == "unfused":
        gx = _G["__exact__"]
        envx = executor.run(gx, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        outx = envx[gx.outputs[0]].reshape(-1)
        ff = _first_diff(g, env64, envx)
    return out32, out64, outx, t32, ff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(list(torch_bivit.CONFIGS) + list(torch_gsb.CONFIGS)))
    ap.add_argument("--repo", required=True, help="clone of the official repo (read only)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--sha256", help="expected SHA-256 (overrides --registry)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--drop-gelu-before-sign", action="store_true",
                    help="variant: the MLP GELU, used only through torch.sign by fc2, is removed (passes.drop_gelu_before_sign; "
                         "sign(GELU(x)) = sign(x), so it differs only where the evaluated GELU underflows to 0, x <= -8.37 in "
                         "float64). The source references are computed with the GELU replaced by the identity (same semantics); "
                         "--out-dir should be a variant directory (e.g. artifacts/<model>/no_gelu)")
    args = ap.parse_args()
    fam = FAMILIES[family_of(args.model_id)]
    ad = fam["adapter"]

    expected = args.sha256
    if not expected:
        reg = json.load(open(args.registry))
        entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
        if entry is None:
            print(f"ERROR: {args.model_id} not in {args.registry}", file=sys.stderr)
            return 2
        expected = entry["checkpoint"]["sha256"]
    actual = sha256_file(args.weights)
    if actual != expected:
        print(f"ERROR: checkpoint hash mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}")
    os.makedirs(args.out_dir, exist_ok=True)

    import timm
    import torch

    torch.set_num_threads(args.torch_threads)
    ckpt = ad.load_checkpoint(args.weights)
    model, ck_args, repairs = ad.build_reference(args.repo, args.model_id, ckpt)
    src_hashes = {f: sha256_file(os.path.join(args.repo, f)) for f in fam["sources"]}

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][fam["preprocess"]]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    src = model
    if args.drop_gelu_before_sign:
        # The variant's source semantics: the official model with every MLP GELU replaced by the identity
        # (fc2 sees torch.sign(clip(fc1 output)) = sign(fc1 output)); the IR is converted from the official model.
        if not hasattr(model, "blocks"):
            raise ValueError("--drop-gelu-before-sign: no transformer blocks with an MLP GELU")
        src = ad.copy_model(model)
        for blk in src.blocks:
            if type(blk.mlp.act).__name__ != "GELU":
                raise TypeError(f"--drop-gelu-before-sign: MLP activation is {type(blk.mlp.act).__name__}")
            blk.mlp.act = torch.nn.Identity()
    m64, n_sub = ad.fp64_exact_binary(src)
    ref32 = run_reference(src, x, torch.float32).astype(np.float32)
    ref64 = run_reference(m64, x, torch.float64)
    ref64n = run_reference(ad.copy_model(src).double(), x, torch.float64)
    del m64
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    ref64n.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))
    extra_refs = {}
    if hasattr(ad, "extra_references") and not args.drop_gelu_before_sign:
        extra_refs = ad.extra_references(model, x, labels, args.out_dir)

    def acc(lg, k=1):
        return float(np.mean([labels[i] in np.argsort(-lg[i])[:k] for i in range(len(lg))]))

    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_preprocessing": fam["preprocess"],
              "eval_preprocessing_definition": pre["definition"],
              "reference": {"framework": f"torch {torch.__version__} (CPU), timm {timm.__version__}", "class": fam["class"],
                            "repo": fam["repo_url"], "repo_commit": repo_commit(args.repo), "source_sha256": src_hashes,
                            "repairs_to_run_official_code": repairs, "checkpoint_args": ck_args,
                            "checkpoint_epoch": int(ckpt.get("epoch", -1)), "batch_size": 1,
                            "torch_threads": args.torch_threads,
                            "torch_threads_note": "fp32 reductions in the source's weight quantizers (mean|W|, ||W||_1/n, mean W) can differ "
                                                  "by an ulp with the CPU thread count; the IR constants and every reference come from "
                                                  "this one process, so they are mutually consistent",
                            "f64_reference": f"official model in float64 with the fp32-derived quantized weights substituted in {n_sub} layers",
                            "top1_accuracy": acc(ref32), "top5_accuracy": acc(ref32, 5),
                            "top1_accuracy_f64": acc(ref64), "top5_accuracy_f64": acc(ref64, 5),
                            "top1_accuracy_f64_naive": acc(ref64n)},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "reference_f64_naive_vs_f64": compare_outputs(ref64, ref64n, labels),
              "reference_f32_vs_f64_naive": compare_outputs(ref64n, ref32, labels),
              "extra_references": extra_refs, "variants": {}}
    print(json.dumps({k: v for k, v in report["reference"].items() if k not in ("checkpoint_args", "source_sha256")}, indent=1))
    print("source fp32 vs fp64:", {k: report["reference_f32_vs_f64"][k] for k in ("max_abs_error", "top1_agreement", "cosine_min")})

    files, info_ge = {}, None
    for key, binarize, fold in (("unfused", "ge", False), ("fused", "ge", True), ("sign_zero", "sign_zero", True)):
        graph, info = ad.convert(model, args.model_id, binarize=binarize)
        if key == "unfused":
            info_ge = info
        applied = passes.optimize(graph, fold=fold)
        if args.drop_gelu_before_sign:
            applied["drop_gelu_before_sign"] = passes.drop_gelu_before_sign(graph)
            if not applied["drop_gelu_before_sign"]:
                raise ValueError("--drop-gelu-before-sign: no GELU used only through a sign binarization")
            applied["prune_unused_constants"] += passes.prune_unused_constants(graph)
        graph.manifest = {
            "schema": "bitino.manifest.v1",
            "model_id": args.model_id,
            "variant": f"{'fused' if fold else 'unfused'}, input_binarize={binarize}"
                       + (", drop_gelu_before_sign" if args.drop_gelu_before_sign else ""),
            "source": {"framework": "pytorch", "class": fam["class"], "repo": fam["repo_url"],
                       "repo_commit": report["reference"]["repo_commit"], "source_sha256": src_hashes,
                       "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
            "conversion": {"tool": "tools/convert_bivit.py", "adapter": ad.__name__, "passes": applied,
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "torch": torch.__version__, "input_binarize": binarize,
                           "binarize_note": info.get("binarize_note", "")},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                      "preprocessing": pre["definition"]},
            "outputs": ["logits"],
            "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"],
            "notes": info["notes"],
        }
        name = {"unfused": "model_unfused.bitino", "fused": "model.bitino", "sign_zero": "model_sign_zero.bitino"}[key]
        path = os.path.join(args.out_dir, name)
        file_sha = bfformat.write(graph, path)
        files[key] = path
        report["variants"][key] = {"file": name, "file_sha256": file_sha, "file_bytes": os.path.getsize(path), "input_binarize": binarize,
                                   "op_counts": graph.op_counts(), "n_nodes": len(graph.nodes), "passes": applied,
                                   "constant_bytes": graph.constant_bytes()}
        print(f"[{key}] wrote {path} ({os.path.getsize(path)} B): {len(graph.nodes)} nodes, passes={applied}")
    report["conversion_notes"] = info_ge["notes"]
    report["state_dict_tensors"] = info_ge["n_state_dict_tensors"]
    report["state_dict_unused_by_source"] = info_ge["state_dict_unused_by_source"]

    jobs = [(k, i, x[i:i + 1]) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files, "unfused", info_ge["float64_constants"])) as pool:
        results = pool.map(_run_one, jobs, chunksize=1)
    for j, k in enumerate(files):
        res = results[j * len(x):(j + 1) * len(x)]
        out32 = np.stack([r[0] for r in res]).astype(np.float32)
        out64 = np.stack([r[1] for r in res])
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        sem = compare_outputs(ref64, out64, labels)
        per_sample = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per_sample <= 1e-6))
        sem["n_samples_max_abs_le_1e-4"] = int(np.sum(per_sample <= 1e-4))
        sem["deviating_samples_gt_1e-6"] = [int(i) for i in np.where(per_sample > 1e-6)[0]]
        sem["per_sample_max_abs"] = [float(v) for v in per_sample]
        ties = {}
        for r in res:
            for mode, (zz, tot) in r[3].items():
                t = ties.setdefault(mode, [0, 0])
                t[0] += zz
                t[1] += tot
        v = report["variants"][k]
        v.update({"numpy_vs_reference_f32": compare_outputs(ref32, out32, labels),
                  "numpy_vs_reference_f64": compare_outputs(ref64, out32, labels),
                  "numpy_vs_reference_f64_naive": compare_outputs(ref64n, out32, labels),
                  "semantic_check_ir_f64_vs_reference_f64": sem,
                  "exact_zeros_at_binarization_sites_numpy_f32": {m: {"zeros": a, "values": b} for m, (a, b) in ties.items()}})
        if k == "unfused":
            outx = np.stack([r[2] for r in res])
            px = np.abs(outx - ref64).max(axis=1)
            v["float64_constants_check"] = {
                "description": "unfused IR with every float constant replaced by its exact float64 value vs float64 source",
                "max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1)))}
            ffs = {int(i): res[i][4] for i in sem["deviating_samples_gt_1e-6"]}
            rel = [f["max_abs_pre_a"] / max(f["typical_abs_pre"], 1e-300) for f in ffs.values() if f]
            v["first_binarization_difference_f32_constants_vs_f64_constants"] = {
                "description": "for each sample deviating > 1e-6 from the f64 source: first binarization site where the IR with "
                               "float32 constants and the IR with float64 constants (both evaluated in float64) binarize differently",
                "n_deviating": len(ffs), "n_without_flip": int(sum(1 for f in ffs.values() if f is None)),
                "max_abs_pre": float(max([f["max_abs_pre_a"] for f in ffs.values() if f] or [0.0])),
                "max_rel_pre": float(max(rel or [0.0])), "per_sample": ffs}
        c32, c64 = v["numpy_vs_reference_f32"], v["numpy_vs_reference_f64"]
        print(f"[{k}] numpy f32: top1 {c64['candidate_top1_accuracy']:.4f}, agree vs f32 ref {c32['top1_agreement']:.4f}, "
              f"vs f64 ref {c64['top1_agreement']:.4f} (max_abs {c64['max_abs_error']:.3e}) | SEMANTIC f64: max_abs {sem['max_abs_error']:.3e} "
              f"top1 {sem['top1_agreement']:.4f} samples<=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{len(x)}"
              + (f" | f64-constants: max_abs {v['float64_constants_check']['max_abs_error']:.3e}" if k == "unfused" else ""))
    np32 = {k: np.fromfile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"), np.float32).reshape(len(x), -1) for k in files}
    report["ge_vs_sign_zero_numpy_f32"] = compare_outputs(np32["sign_zero"], np32["fused"], labels)
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(np32["unfused"], np32["fused"], labels)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    print("source fp32 vs fp64 top1 agreement", report["reference_f32_vs_f64"]["top1_agreement"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
