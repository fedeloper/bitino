#!/usr/bin/env python3
"""Convert a verified AB-BNN (A&B BNN, CVPR'24) checkpoint (ImageNet / CIFAR-10 / CIFAR-100) to .bitino and validate it.

Runs in a plain PyTorch environment (CPU). The official model definition is imported from
a clone of https://github.com/Ruichen0424/AB-BNN (read only, no bytecode written into it);
the checkpoint is loaded with torch.load(weights_only=True).

  python3 tools/convert_abbnn.py --model-id ab-bnn-imagenet-reactnet18 \
      --repo .cache/repos/AB-BNN --weights .cache/checkpoints/ab-bnn/imagenet_reactnet18_step2.h5 \
      --registry registry/models.json --eval-dir .cache/data/eval128 \
      --out-dir artifacts/ab-bnn-imagenet-reactnet18

CIFAR models (ResNet18 / Reactnet with imagenet=False, Inference/CIFAR10.ipynb, CIFAR100.ipynb) use the
ImageNet-mean/std normalised CIFAR eval sets (tools/eval_sets/cifar10_normalized.py key cifar10_imagenet_norm,
tools/eval_sets/cifar100_normalized.py key cifar100_imagenet_norm); ReActNet-A (Network/ReActNet_A_Qaw.py) is
handled by torch_abbnn._convert_reactnet_a.

Produces in --out-dir:
  model.bitino / model_unfused.bitino                      input_binarize "ge" (deployable, 0 -> +1)
  model_sign_zero.bitino / model_sign_zero_unfused.bitino  input_binarize "sign_zero" (exact torch.sign,
                                                             reference kernels only)
  reference_logits.f32 / reference_logits_f64.f64            official PyTorch model, float32 and float64
  numpy_logits_<variant>.f32                                 numpy executor (float32) per variant
  conversion_report.json                                     Level-3 comparison, exact-zero counts
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import torch_abbnn  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

ARCHS = {  # model_id -> (module, constructor, num_classes, imagenet stem, eval preprocessing key)
    "ab-bnn-imagenet-reactnet18": ("Network.ReActNet_18_Qaw", "ResNet18", 1000, True, "torchvision_imagenet"),
    "ab-bnn-imagenet-reactnet34": ("Network.ReActNet_18_Qaw", "ResNet34", 1000, True, "torchvision_imagenet"),
    "ab-bnn-imagenet-reactneta": ("Network.ReActNet_A_Qaw", "Reactnet", 1000, True, "torchvision_imagenet"),
    "ab-bnn-cifar10-reactnet18": ("Network.ReActNet_18_Qaw", "ResNet18", 10, False, "cifar10_imagenet_norm"),
    "ab-bnn-cifar10-reactneta": ("Network.ReActNet_A_Qaw", "Reactnet", 10, False, "cifar10_imagenet_norm"),
    "ab-bnn-cifar100-reactnet18": ("Network.ReActNet_18_Qaw", "ResNet18", 100, False, "cifar100_imagenet_norm"),
    "ab-bnn-cifar100-reactneta": ("Network.ReActNet_A_Qaw", "Reactnet", 100, False, "cifar100_imagenet_norm"),
}
PREPROCESS = "torchvision_imagenet"   # ImageNet models; preprocess_of(model_id) in general


def preprocess_of(model_id: str) -> str:
    return ARCHS[model_id][4]


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


def load_model(repo: str, model_id: str, weights: str):
    import importlib

    import torch

    sys.dont_write_bytecode = True   # never write __pycache__ into the source clone
    sys.path.insert(0, os.path.abspath(repo))
    mod_name, ctor, n_cls, imagenet, _ = ARCHS[model_id]
    module = importlib.import_module(mod_name)
    model = getattr(module, ctor)(num_classes=n_cls, imagenet=imagenet)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    src = {os.path.relpath(f, repo): sha256_file(f) for f in (module.__file__, os.path.join(os.path.dirname(module.__file__), "BNN.py"))}
    return model, f"{mod_name}.{ctor}", src


def reference_outputs(model, x_nhwc: np.ndarray, dtype) -> tuple[np.ndarray, dict]:
    """Logits of the official model plus exact-zero counts at every binarization input.

    BasicBlock binarizes sign(move0(x) * beta); a forward hook on move0 sees x + bias/beta
    and one on act1 sees the value handed to torch.sign."""
    import torch

    m = model.to(dtype)
    zeros = {"move0_out_exact_zero": 0, "sign_input_exact_zero": 0, "n_binarized": 0}
    hooks = []
    for name, mod in m.named_modules():
        if type(mod).__name__ == "BasicBlock":
            # ResNet: move0 -> act1; ReActNet-A: move11 / move21 -> binary_activation (called twice per block)
            for mv in ("move0", "move11", "move21"):
                if hasattr(mod, mv):
                    hooks.append(getattr(mod, mv).register_forward_hook(
                        lambda _m, _i, o: zeros.__setitem__("move0_out_exact_zero", zeros["move0_out_exact_zero"] + int((o == 0).sum()))))

            def pre(_m, inp):
                zeros["sign_input_exact_zero"] += int((inp[0] == 0).sum())
                zeros["n_binarized"] += int(inp[0].numel())
            hooks.append((mod.act1 if hasattr(mod, "act1") else mod.binary_activation).register_forward_pre_hook(pre))
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), 16):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + 16].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).numpy())
    for h in hooks:
        h.remove()
    model.to(torch.float32)
    return np.concatenate(out), zeros


# ------------------------------------------------------------- numpy executor workers
_G = {}


def _init_worker(paths: dict) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    for k, p in paths.items():
        _G[k] = bfformat.read(p)


def _run_one(job):
    key, x, want_f64 = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    zeros = [int(np.sum(env[n.inputs[0]] == 0)) for n in g.nodes if n.op == "BConv2D"]
    out32 = env[g.outputs[0]].reshape(-1)
    out64, zeros64 = None, []
    if want_f64:
        env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        out64 = env64[g.outputs[0]].reshape(-1)
        zeros64 = [int(np.sum(env64[n.inputs[0]] == 0)) for n in g.nodes if n.op == "BConv2D"]
    return out32, out64, zeros, zeros64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(ARCHS))
    ap.add_argument("--repo", required=True, help="clone of github.com/Ruichen0424/AB-BNN")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"),
                    help="models.json holding the expected checkpoint SHA-256")
    ap.add_argument("--sha256", help="expected SHA-256 (overrides --registry)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16, help="worker processes for the numpy executor")
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

    import torch
    import torchvision

    torch.set_num_threads(min(16, os.cpu_count() or 1))
    model, cls, src_hashes = load_model(args.repo, args.model_id, args.weights)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][preprocess_of(args.model_id)]
    input_hw = tuple(pre["shape"][1:3])
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    ref32, zeros32 = reference_outputs(model, x, torch.float32)
    ref64, zeros64 = reference_outputs(model, x, torch.float64)
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_preprocessing": preprocess_of(args.model_id),
              "eval_set": em["name"],
              "reference": {"framework": f"torch {torch.__version__} (CPU), torchvision {torchvision.__version__}",
                            "class": cls, "repo_commit": repo_commit(args.repo), "source_sha256": src_hashes,
                            "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)),
                            "top5_accuracy": float(np.mean([labels[i] in np.argsort(-ref32[i])[:5] for i in range(len(x))])),
                            "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                            "top5_accuracy_f64": float(np.mean([labels[i] in np.argsort(-ref64[i])[:5] for i in range(len(x))])),
                            "exact_zeros_at_binarization_f32": zeros32, "exact_zeros_at_binarization_f64": zeros64},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "variants": {}}
    print(json.dumps(report["reference"], indent=1))

    files = {}
    for binarize in ("ge", "sign_zero"):
        for variant, fold in (("unfused", False), ("fused", True)):
            graph, info = torch_abbnn.convert(model, args.model_id, binarize=binarize, input_hw=input_hw)
            applied = passes.optimize(graph, fold=fold)
            stem = "model" if binarize == "ge" else "model_sign_zero"
            key = variant if binarize == "ge" else f"sign_zero_{variant}"
            graph.manifest = {
                "schema": "bitino.manifest.v1",
                "model_id": args.model_id,
                "variant": f"{variant}, input_binarize={binarize}",
                "source": {"framework": "pytorch", "class": cls, "repo": "https://github.com/Ruichen0424/AB-BNN",
                           "repo_commit": report["reference"]["repo_commit"], "source_sha256": src_hashes,
                           "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
                "conversion": {"tool": "tools/convert_abbnn.py", "adapter": "bitino.adapters.torch_abbnn",
                               "passes": applied, "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "python": platform.python_version(), "torch": torch.__version__,
                               "input_binarize": binarize,
                               "binarize_note": ("exact torch.sign semantics (sign(0)=0); reference kernels only" if binarize == "sign_zero"
                                                 else "deployable: 0 -> +1 at binarization; differs from torch.sign only for exact zeros")},
                "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32",
                          "preprocessing": pre["definition"]},
                "outputs": ["logits"],
                "op_versions": {"format": 1},
                "layer_precision": info["layer_precision"],
                "weight_diagnostics": info["weight_diagnostics"],
            }
            path = os.path.join(args.out_dir, f"{stem}.bitino" if fold else f"{stem}_unfused.bitino")
            file_sha = bfformat.write(graph, path)
            files[key] = path
            report["variants"][key] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": binarize, "op_counts": graph.op_counts(), "passes": applied}
            print(f"[{key}] wrote {path}: {graph.op_counts()} passes={applied}")

    # Level 3: numpy executor (float32 = runtime semantics; float64 = strict semantic check).
    jobs = [(k, x[i:i + 1], True) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
        results = pool.map(_run_one, jobs, chunksize=4)
    by_key = {k: results[j * len(x):(j + 1) * len(x)] for j, k in enumerate(files)}
    np_logits = {}
    for k, res in by_key.items():
        out32 = np.stack([r[0] for r in res]).astype(np.float32)
        out64 = np.stack([r[1] for r in res])
        zeros = np.array([r[2] for r in res])              # [samples, BConv2D]
        zeros64 = np.array([r[3] for r in res])
        np_logits[k] = out32
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        cmp = compare_outputs(ref32, out32, labels)
        cmp64 = compare_outputs(ref64, out32, labels)
        sem = compare_outputs(ref64, out64, labels)
        per_sample = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per_sample <= 1e-6))
        sem["n_samples_max_abs_le_1e-4"] = int(np.sum(per_sample <= 1e-4))
        sem["deviating_samples_gt_1e-4"] = [int(i) for i in np.where(per_sample > 1e-4)[0]]
        sem["per_sample_max_abs"] = [float(v) for v in per_sample]
        report["variants"][k].update({
            "numpy_vs_reference_f32": cmp, "numpy_vs_reference_f64": cmp64,
            "semantic_check_ir_f64_vs_reference_f64": sem,
            "exact_zeros_at_bconv_inputs_numpy_f32": {"total": int(zeros.sum()), "samples_with_zeros": int(np.sum(zeros.sum(1) > 0))},
            "exact_zeros_at_bconv_inputs_numpy_f64": {"total": int(zeros64.sum()), "samples_with_zeros": int(np.sum(zeros64.sum(1) > 0))}})
        print(f"[{k}] vs f32 ref: top1 agree {cmp['top1_agreement']:.4f} max_abs {cmp['max_abs_error']:.3e} | vs f64 ref: top1 agree "
              f"{cmp64['top1_agreement']:.4f} max_abs {cmp64['max_abs_error']:.3e} | SEMANTIC (f64 IR vs f64 ref): max_abs "
              f"{sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} samples<=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{len(x)} "
              f"| exact zeros at BConv2D inputs (numpy f32): {int(zeros.sum())}")
    report["ge_vs_sign_zero_numpy_f32"] = {v: compare_outputs(np_logits[f"sign_zero_{v}"], np_logits[v], labels) for v in ("unfused", "fused")}
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    print("ge vs sign_zero (numpy f32) exact match:", {v: c["exact_match"] for v, c in report["ge_vs_sign_zero_numpy_f32"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
