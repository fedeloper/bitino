#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert a verified BHViT (IMRL/BHViT, CVPR'25) ImageNet checkpoint to .bitino and validate it (Level 3).

Runs in the read-only 'bhvit' conda env (torch 2.5, timm, transformers 4.39; CPU). The official model
definition is imported from a clone of https://github.com/IMRL/BHViT (read only, no bytecode written);
the checkpoint is loaded with torch.load(weights_only=True).

  ~/anaconda3/envs/bhvit/bin/python tools/convert_bhvit.py --model-id bhvit-tiny-binary \
      --repo .cache/repos/IMRL_BHViT --weights .cache/checkpoints/bhvit-tiny-binary/best64.0.pth \
      --eval-dir .cache/data/eval128_bhvit --out-dir artifacts/bhvit-tiny-binary

Produces in --out-dir:
  model.bitino / model_unfused.bitino     input_binarize 'ge' at BDense/BConv2D (deployable, 0 -> +1)
  model_sign_zero.bitino                   input_binarize 'sign_zero' (exact torch.sign; reference kernels only)
  reference_logits.f32 / reference_logits_f64.f64   official model, float32 / float64, batch size 1
  numpy_logits_<variant>.f32                numpy executor (float32) per variant (numpy_logits_fused.f32 = model.bitino)
  conversion_report.json                    strict float64 semantic check, float64-constant check,
                                            exact ties at binarization, GELU ablation, weight diagnostics
The reference runs at batch size 1: BinaryActivation_Attention.grad_scale uses x.numel(), so its
forward value depends on the batch size through fp32 rounding (it does not change the parameters of
the shipped checkpoints, see report "attention_quantizers").
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
from bitino.adapters import bhvit  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

PREPROCESS = "bhvit"
SOURCE_FILES = ("transformer/BHViT.py", "transformer/utils_quant.py", "models.py", "datasets.py", "main_new.py")


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


def reference_outputs(model, x_nhwc: np.ndarray, dtype, gelu_double: bool = False) -> tuple[np.ndarray, dict]:
    """Logits at batch size 1 plus exact-tie counts at every binarization / rounding point:
    QuantizeLinear/QuantizeConv2d inputs (torch.sign(0) = 0), Q/K/V before sign, and attention-quantizer
    arguments exactly at a .5 rounding boundary. gelu_double: stem GELU evaluated in float64 (ablation)."""
    import torch

    m = model.to(dtype)
    z = {"binary_layer_inputs_exact_zero": 0, "binary_layer_inputs_total": 0, "qkv_exact_zero": 0, "qkv_total": 0,
         "attn_quant_half_ties": 0, "attn_quant_total": 0, "stem_gelu_inputs_lt_-5.5": 0, "stem_gelu_fp_zero_for_negative_input": 0}
    hooks = []
    for name, mod in m.named_modules():
        tn = type(mod).__name__
        if tn in ("QuantizeLinear", "QuantizeConv2d"):
            def pre(_m, inp):
                z["binary_layer_inputs_exact_zero"] += int((inp[0] == 0).sum())
                z["binary_layer_inputs_total"] += int(inp[0].numel())
            hooks.append(mod.register_forward_pre_hook(pre))
        elif tn == "BHViTSelfAttention":
            for t in ("q", "k", "v"):
                def post(_m, _i, o, sa=mod, t=t):
                    v = o + getattr(sa, f"move{t}2")
                    z["qkv_exact_zero"] += int((v == 0).sum())
                    z["qkv_total"] += int(v.numel())
                hooks.append(getattr(mod, f"rprelu{t}").register_forward_hook(post))
        elif tn == "BinaryActivation_Attention":
            def qpre(qm, inp):
                a = qm.alpha.reshape(1, -1, 1, 1)
                u = inp[0] / a + qm.zero_point.reshape(1, -1, 1, 1)
                z["attn_quant_half_ties"] += int(((u - torch.floor(u)) == 0.5).sum())
                z["attn_quant_total"] += int(u.numel())
            hooks.append(mod.register_forward_pre_hook(qpre))
    act = m.vit.embeddings.patch_embeddings.act

    def gpre(_m, inp):
        x = inp[0]
        z["stem_gelu_inputs_lt_-5.5"] += int((x < -5.5).sum())
        z["stem_gelu_fp_zero_for_negative_input"] += int(((torch.nn.functional.gelu(x) == 0) & (x < 0)).sum())
    hooks.append(act.register_forward_pre_hook(gpre))
    if gelu_double:
        hooks.append(act.register_forward_hook(lambda _m, i, o: torch.nn.functional.gelu(i[0].double()).to(o.dtype)))
    out = []
    with torch.no_grad():
        for i in range(len(x_nhwc)):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + 1].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).logits.numpy())
    for h in hooks:
        h.remove()
    model.to(torch.float32)
    return np.concatenate(out), z


# ------------------------------------------------------------- numpy executor workers
_G: dict = {}


def _init_worker(paths: dict, exact_key: str, f64_consts: dict) -> None:
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
    if exact_key:
        g = bfformat.read(paths[exact_key])
        for name, v in f64_consts.items():
            # The fp32 grad_scale() value of a quantizer alpha/zero_point may differ from the f64 one by 1 ulp.
            assert np.allclose(np.asarray(v, np.float32), g.constants[name].data, rtol=2.0 ** -23, atol=0), f"{name}: bfmodel differs from adapter"
            g.constants[name].data = np.asarray(v, np.float64)
        _G["__exact__"] = g


def _zeros(g, env) -> int:
    tot = 0
    for n in g.nodes:
        if n.op in ("BConv2D", "BDense"):
            x = env[n.inputs[0]]
            t = env[n.inputs[4]] if len(n.inputs) > 4 and n.inputs[4] else 0.0
            tot += int(np.sum(x == t))
    return tot


def _run_one(job):
    key, x = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    out32 = env[g.outputs[0]].reshape(-1)
    z32 = _zeros(g, env)
    del env
    env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    out64 = env64[g.outputs[0]].reshape(-1)
    z64 = _zeros(g, env64)
    del env64
    outx = None
    if key == "unfused" and "__exact__" in _G:
        outx = executor.run(_G["__exact__"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
    return out32, out64, outx, z32, z64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(bhvit.CONFIGS))
    ap.add_argument("--repo", required=True, help="clone of github.com/IMRL/BHViT (read only)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--sha256", help="expected SHA-256 (overrides --registry)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=16)
    args = ap.parse_args()

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
    import transformers

    torch.set_num_threads(args.torch_threads)
    ckpt = bhvit.load_checkpoint(args.weights)
    model, ck_args = bhvit.build_reference(args.repo, args.model_id, ckpt)
    cls = "transformer.BHViT.BHViTForImageClassification (models.get_model)"
    src_hashes = {f: sha256_file(os.path.join(args.repo, f)) for f in SOURCE_FILES}

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    ref32, ties32 = reference_outputs(model, x, torch.float32)
    ref64, ties64 = reference_outputs(model, x, torch.float64)
    ref32_gd, _ = reference_outputs(model, x, torch.float32, gelu_double=True)
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    def acc(lg, k=1):
        return float(np.mean([labels[i] in np.argsort(-lg[i])[:k] for i in range(len(lg))]))

    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_preprocessing": PREPROCESS,
              "eval_preprocessing_definition": pre["definition"],
              "reference": {"framework": f"torch {torch.__version__} (CPU), timm {timm.__version__}, transformers {transformers.__version__}",
                            "class": cls, "repo": "https://github.com/IMRL/BHViT", "repo_commit": repo_commit(args.repo),
                            "source_sha256": src_hashes, "checkpoint_args": ck_args, "checkpoint_epoch": int(ckpt.get("epoch", -1)),
                            "batch_size": 1,
                            "top1_accuracy": acc(ref32), "top5_accuracy": acc(ref32, 5),
                            "top1_accuracy_f64": acc(ref64), "top5_accuracy_f64": acc(ref64, 5),
                            "exact_ties_f32": ties32, "exact_ties_f64": ties64,
                            "note": "the authors' evaluate() runs under torch.cuda.amp.autocast (fp16 GEMMs on GPU); this reference is fp32/fp64 CPU"},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "gelu_ablation": {"description": "fp32 source with only the stem GELU evaluated in float64 (the IR evaluates GELU in double)",
                                "vs_reference_f32": compare_outputs(ref32, ref32_gd, labels),
                                "vs_reference_f64": compare_outputs(ref64, ref32_gd, labels)},
              "variants": {}}
    print(json.dumps({k: v for k, v in report["reference"].items() if k not in ("checkpoint_args", "source_sha256")}, indent=1))
    print("source fp32 vs fp64:", {k: report["reference_f32_vs_f64"][k] for k in ("max_abs_error", "top1_agreement", "cosine_min")})

    files, info_ge = {}, None
    for key, binarize, fold in (("unfused", "ge", False), ("fused", "ge", True), ("sign_zero", "sign_zero", True)):
        graph, info = bhvit.convert(model, args.model_id, binarize=binarize)
        if key == "unfused":
            info_ge = info
        applied = passes.optimize(graph, fold=fold)
        graph.manifest = {
            "schema": "bitino.manifest.v1",
            "model_id": args.model_id,
            "variant": f"{'fused' if fold else 'unfused'}, input_binarize={binarize}",
            "source": {"framework": "pytorch", "class": cls, "repo": "https://github.com/IMRL/BHViT",
                       "repo_commit": report["reference"]["repo_commit"], "source_sha256": src_hashes,
                       "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
            "conversion": {"tool": "tools/convert_bhvit.py", "adapter": "bitino.adapters.bhvit", "passes": applied,
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "torch": torch.__version__, "input_binarize": binarize,
                           "binarize_note": ("exact torch.sign semantics (sign(0)=0) at BDense/BConv2D; reference kernels only" if binarize == "sign_zero"
                                             else "deployable: 0 -> +1 at BDense/BConv2D binarization; differs from torch.sign only for exact ties"),
                           "qkv_note": "Q/K/V always use torch.sign semantics (BMatMul sign_zero, Activation sign_zero)"},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                      "preprocessing": pre["definition"]},
            "outputs": ["logits"],
            "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"],
            "weight_diagnostics": info["weight_diagnostics"],
            "attention_quantizers": info["attention_quantizers"],
        }
        name = {"unfused": "model_unfused.bitino", "fused": "model.bitino", "sign_zero": "model_sign_zero.bitino"}[key]
        path = os.path.join(args.out_dir, name)
        file_sha = bfformat.write(graph, path)
        files[key] = path
        report["variants"][key] = {"file": name, "file_sha256": file_sha, "file_bytes": os.path.getsize(path), "input_binarize": binarize,
                                   "op_counts": graph.op_counts(), "n_nodes": len(graph.nodes), "passes": applied,
                                   "constant_bytes": graph.constant_bytes()}
        print(f"[{key}] wrote {path} ({os.path.getsize(path)} B): {len(graph.nodes)} nodes, passes={applied}")
    wd = info_ge["weight_diagnostics"]
    report["weight_binarization"] = {
        "rule": "signs and scales computed in float64 from the fp32 latent weights; no exact zero in fp64",
        "n_binary_layers": len(wd), "n_binary_weights": int(sum(d["n_weights"] for d in wd)),
        "fp32_torch_exact_zero_centred": int(sum(d["fp32_exact_zero_centred"] for d in wd)),
        "fp32_torch_sign_differs_from_fp64_nonzero": int(sum(d["fp32_sign_differs_from_fp64_nonzero"] for d in wd)),
        "layers_with_differences": [d for d in wd if d["fp32_exact_zero_centred"] or d["fp32_sign_differs_from_fp64_nonzero"]]}
    report["attention_quantizers"] = info_ge["attention_quantizers"]
    report["state_dict_tensors_consumed"] = info_ge["n_state_dict_tensors"]

    # Level 3: numpy executor in float32 (runtime semantics) and float64 (strict semantic check), plus the
    # unfused IR with every float constant replaced by its exact float64 value (analysis only).
    jobs = [(k, x[i:i + 1]) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files, "unfused", info_ge["float64_constants"])) as pool:
        results = pool.map(_run_one, jobs, chunksize=2)
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
        v = report["variants"][k]
        v.update({"numpy_vs_reference_f32": compare_outputs(ref32, out32, labels),
                  "numpy_vs_reference_f64": compare_outputs(ref64, out32, labels),
                  "semantic_check_ir_f64_vs_reference_f64": sem,
                  "exact_ties_at_bdense_bconv_inputs_numpy_f32": int(sum(r[3] for r in res)),
                  "exact_ties_at_bdense_bconv_inputs_numpy_f64": int(sum(r[4] for r in res))})
        if k == "unfused":
            outx = np.stack([r[2] for r in res])
            px = np.abs(outx - ref64).max(axis=1)
            v["float64_constants_check"] = {
                "description": "unfused IR with every float constant replaced by its exact float64 value vs float64 source",
                "max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1)))}
        c32, c64 = v["numpy_vs_reference_f32"], v["numpy_vs_reference_f64"]
        print(f"[{k}] numpy f32: top1 {c64['candidate_top1_accuracy']:.4f}, agree vs f32 ref {c32['top1_agreement']:.4f}, "
              f"vs f64 ref {c64['top1_agreement']:.4f} (max_abs {c64['max_abs_error']:.3e}) | SEMANTIC f64: max_abs {sem['max_abs_error']:.3e} "
              f"top1 {sem['top1_agreement']:.4f} samples<=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{len(x)}"
              + (f" | f64-constants: max_abs {v['float64_constants_check']['max_abs_error']:.3e}" if k == "unfused" else ""))
    np32 = {k: np.fromfile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"), np.float32).reshape(len(x), -1) for k in files}
    report["ge_vs_sign_zero_numpy_f32"] = compare_outputs(np32["sign_zero"], np32["fused"], labels)
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(np32["unfused"], np32["fused"], labels)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    print("source fp32 vs fp64 top1 agreement", report["reference_f32_vs_f64"]["top1_agreement"],
          "| gelu ablation (fp32 + double GELU) vs fp32:", report["gelu_ablation"]["vs_reference_f32"]["max_abs_error"],
          "top1 vs f64", report["gelu_ablation"]["vs_reference_f64"]["top1_agreement"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
