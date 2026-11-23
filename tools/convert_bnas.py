#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Converter for the BNAS checkpoints (yonseivnl/bnas, ECCV 2020; adapter bitino.adapters.torch_bnas):
BNAS-A (CIFAR-10, src/model.py NetworkCIFAR) and BNAS-D/E/H (ImageNet, src/model_imagenet.py NetworkImageNet).
The reference is the official model code (read only, run with python3 -I) with the official BinOp
(src/bin_utils.py) applied exactly as test.py / test_imagenet.py ``infer`` does before the forward pass.

  python3 -I tools/convert_bnas.py --model-id bnas-a-cifar10 --repo .cache/repos/bnas-s13/src-cdb58f7 \
      --weights .cache/checkpoints/bnas-a-cifar10/bnas_a_best.pt \
      --eval-dir .cache/data/eval128_cifar10_darts --big-eval-dir .cache/data/eval10000_cifar10_darts \
      --out-dir artifacts/bnas-a-cifar10
  python3 -I tools/convert_bnas.py --model-id bnas-d-imagenet --repo .cache/repos/bnas-s13/src-cdb58f7 \
      --weights .cache/checkpoints/bnas-d-imagenet/bnas_d_best.pth.tar \
      --eval-dir .cache/data/eval128 --big-eval-dir .cache/data/eval5000 --out-dir artifacts/bnas-d-imagenet

Source version: the released checkpoints (files dated 2020-07-20) were trained and evaluated with the FIRST release
(commit cdb58f7, 2020-08-12). Its BinOp also binarizes the preprocess_res convs (the 'res' test reads a stale loop
variable); the later "typo fixes" commit 675ecc7 (2020-11-15, current HEAD) keeps them float, which does not match the
checkpoints (official eval on CPU: BNAS-A 81.6% instead of ~92.4%, BNAS-H 0% on eval128). Evidence that the first
release is the right reference: the preprocess_res latent weights in every checkpoint are mean-centred over the input
channels (|mean| <= 3e-8) and clamped to [-1, 1] (BinOp.restore() writes back the centred/clamped tensor), exactly
like the binarized convs and unlike the never-binarized quirk convs; and the first-release code reproduces the
README accuracies. Pass the first-release tree as --repo (tarball of cdb58f7, with a COMMIT_SHA file).

Repairs of the official code (documented in conversion_report.json ``reference.repairs``):
  * ``operations.BinActive`` is a legacy autograd.Function with a non-static forward, which modern PyTorch rejects
    at call time; it is replaced by a callable with the same forward (``input.sign()``, channel mean of |input|).
    Inference only; nothing else is changed.
ImageNet class order: prepare_imgnet.py renames the synset folders to the integers 0..999 (synset_to_label.txt = the
standard order) and torchvision ImageFolder then sorts the folder NAMES as strings ("0", "1", "10", "100", ...), so
output j of the official model is standard class int(sorted_names[j]). Reference logits and the IR's classifier rows
are put into the standard order (constant permutation, exact).

Outputs (layout of tools/convert_s12_cifar.py): model{,_unfused,_sign_zero,_sign_zero_unfused}.bitino,
reference_logits.f32 / reference_logits_f64.f64 (official model in float32 / float64 after BinOp in float32),
numpy_logits_<variant>.f32, conversion_report.json (Level 3, float64-constants check, first-flip analysis of every
sample deviating by more than 1e-6), with --big-eval-dir reference_logits_eval<N>.f32 (official fp32).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import io
import json
import multiprocessing as mp
import os
import platform
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import torch_bnas  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402
from convert_torch_cnn import repo_commit, sha256_file  # noqa: E402

REPO_URL = "https://github.com/yonseivnl/bnas"
SOURCE_MODULES = ("operations", "utils", "model", "model_imagenet", "genotypes", "bin_utils")
MODELS = {
    # model_id: (kind, model_config key in test_imagenet.py configs / None for CIFAR, eval preprocessing key)
    "bnas-a-cifar10": ("cifar", None, "cifar10_darts_norm"),
    "bnas-d-imagenet": ("imagenet", "bnas-d", "torchvision_imagenet"),
    "bnas-e-imagenet": ("imagenet", "bnas-e", "torchvision_imagenet"),
    "bnas-h-imagenet": ("imagenet", "bnas-h", "torchvision_imagenet"),
}


def _argparse_defaults(path: str) -> dict:
    """Defaults of the one-line ``parser.add_argument('--x', ..., default=V)`` calls and the ``configs = {...}`` line,
    each line parsed with ast (the scripts are not executed: they need CUDA / apex, and test_imagenet.py is not valid
    Python >= 3.7 syntax as a whole: ``target.cuda(async=True)``)."""
    out: dict = {}
    for line in open(path):
        t = line.strip()
        if t.startswith("parser.add_argument("):
            node = ast.parse(t).body[0].value
            flag = ast.literal_eval(node.args[0])
            for kw in node.keywords:
                if kw.arg == "default":
                    out[flag.lstrip("-").replace("-", "_")] = ast.literal_eval(kw.value)
        elif t.startswith("configs ="):
            out["configs"] = ast.literal_eval(ast.parse(t).body[0].value)
    return out


def imagenet_class_perm() -> np.ndarray:
    """perm[c] = index of standard class c among the official model's outputs (ImageFolder over folders "0".."999")."""
    names = sorted(str(i) for i in range(1000))
    return np.array([names.index(str(c)) for c in range(1000)], np.int64)


def load_source(model_id: str, repo: str, weights: str | None, cfg_override: dict | None = None, prepare=None):
    """Returns (model after BinOp.binarization() in eval mode, BinOp target names, info dict). Tests: ``weights=None``
    keeps the random initialization, ``cfg_override`` changes init_channels/layers, ``prepare(model)`` runs before BinOp."""
    import importlib

    import torch

    kind, cfg_key, _ = MODELS[model_id]
    sys.dont_write_bytecode = True
    sub = os.path.join(os.path.abspath(repo), "src")
    for k in list(sys.modules):
        if k.split(".")[0] in SOURCE_MODULES:
            del sys.modules[k]
    saved = list(sys.path)
    # generic module names (utils, model, operations, ...): drop every sys.path entry that has one, so only the clone wins
    sys.path[:] = [sub] + [q for q in sys.path if not any(os.path.exists(os.path.join(q or ".", x + s))
                                                          for x in SOURCE_MODULES for s in ("", ".py"))]
    try:
        ops = importlib.import_module("operations")
        genotypes = importlib.import_module("genotypes")
        bin_utils = importlib.import_module("bin_utils")
        net_mod = importlib.import_module("model" if kind == "cifar" else "model_imagenet")
    finally:
        sys.path[:] = saved
    files = ["src/operations.py", "src/bin_utils.py", "src/genotypes.py", "src/utils.py",
             "src/model.py" if kind == "cifar" else "src/model_imagenet.py", "src/test.py" if kind == "cifar" else "src/test_imagenet.py"]
    src_hashes = {f: sha256_file(os.path.join(repo, f)) for f in files}

    # repair: legacy autograd.Function (non-static forward) -> plain callable with the same forward
    legacy_error = None
    try:
        ops.BinActive()(torch.zeros(1, 2, 1, 1))
    except Exception as e:  # noqa: BLE001 - recorded verbatim
        legacy_error = f"{type(e).__name__}: {str(e).splitlines()[0]}"

    class BinActive:
        """Inference-only replacement of operations.BinActive (src/operations.py:14-29): same forward values."""
        def __call__(self, input):
            mean = torch.mean(input.abs(), 1, keepdim=True)
            return input.sign(), mean
    ops.BinActive = BinActive

    if kind == "cifar":
        d = _argparse_defaults(os.path.join(repo, "src", "test.py"))
        cfg = {"init_channels": d["init_channels"], "layers": d["layers"], "arch": d["arch"], "num_skip": d["num_skip"], "num_classes": 10}
    else:
        d = _argparse_defaults(os.path.join(repo, "src", "test_imagenet.py"))
        layers, ch, opt_level = d["configs"][cfg_key]
        cfg = {"init_channels": ch, "layers": layers, "arch": d["arch"], "num_skip": d["num_skip"], "num_classes": 1000,
               "apex_opt_level_official": opt_level}
    cfg.update(cfg_override or {})
    genotype = getattr(genotypes, cfg["arch"])
    with contextlib.redirect_stdout(io.StringIO()):          # Cell.__init__ prints the channel counts
        cls = net_mod.NetworkCIFAR if kind == "cifar" else net_mod.NetworkImageNet
        model = cls(cfg["init_channels"], cfg["num_classes"], cfg["layers"], genotype)
    model.drop_path_prob = 0.0                                # training-only (Cell.forward checks self.training first)
    ck = torch.load(weights, map_location="cpu", weights_only=True) if weights is not None else {"state_dict": None}
    meta = {}
    if weights is None:
        sd = None
    elif "state_dict" in ck:
        meta = {k: (float(v) if isinstance(v, (int, float)) or (hasattr(v, "numel") and v.numel() == 1) else "<present>")
                for k, v in ck.items() if k not in ("state_dict", "optimizer", "scheduler", "amp")}
        meta["other_keys"] = sorted(k for k in ck if k != "state_dict")
        sd = ck["state_dict"]
    else:
        sd = ck
    if sd is not None:
        assert all(k.startswith("module.") for k in sd), "expected the DataParallel 'module.' prefix on every key"
        model.load_state_dict({k[7:]: v for k, v in sd.items()}, strict=True)
    del ck, sd
    model.eval()
    if prepare is not None:
        prepare(model)

    latent = {n: m.weight.detach().clone() for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)}
    bin_op = bin_utils.BinOp(model, types.SimpleNamespace(num_skip=cfg["num_skip"]))
    by_param = {id(m.weight): n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)}
    targets = [by_param[id(p)] for p in bin_op.target_modules]
    first_release = "for m in model.modules():" in open(bin_utils.__file__).read()
    re_targets = torch_bnas.binop_targets(model, cfg["num_skip"], first_release=first_release)
    assert targets == re_targets, "torch_bnas.binop_targets differs from the official BinOp selection"
    bin_op.binarization()
    re_mism = sum(int((torch_bnas.xnor_binarize(latent[n]) != dict(model.named_modules())[n].weight.detach()).sum()) for n in targets)
    zeros = sum(int((dict(model.named_modules())[n].weight == 0).sum()) for n in targets)
    def signature(names):   # training-time binarization leaves latent weights mean-centred (dim 1) and clamped
        names = list(names)
        if not names:
            return None
        return {"n": len(names), "max_abs_mean_over_input_channels": max(float(latent[n].mean(1).abs().max()) for n in names),
                "max_abs_weight": max(float(latent[n].abs().max()) for n in names)}
    res_names = [n for n in latent if "res" in n]
    nonres_bin = [n for n in targets if "res" not in n]
    quirk = [n for n in latent if "res" not in n and n not in targets]
    latent_sig = {"binarized_nonres": signature(nonres_bin), "preprocess_res": signature(res_names),
                  "not_binarized_nonres (stem + quirk)": signature(quirk)}
    nonres = [n for n in latent if "res" not in n]
    not_bin = [n for n in nonres if n not in targets]
    info = {"config": cfg, "genotype": str(genotype), "source_sha256": src_hashes, "checkpoint_meta": meta,
            "class": f"{'model.NetworkCIFAR' if kind == 'cifar' else 'model_imagenet.NetworkImageNet'}({cfg['init_channels']}, "
                     f"{cfg['num_classes']}, {cfg['layers']}, genotypes.{cfg['arch']})",
            "repairs": [{"what": "operations.BinActive replaced by a callable with the identical forward (sign, |x| channel mean)",
                         "why": "legacy autograd.Function with non-static forward",
                         "original_error_on_torch_" + torch.__version__: legacy_error}],
            "binop": {"version": "first release (cdb58f7: preprocess_res convs binarized)" if first_release
                      else "head (675ecc7: preprocess_res convs float)", "num_skip": cfg["num_skip"],
                      "res_convs_binarized": [n for n in targets if "res" in n], "n_conv2d": len(latent), "n_nonres_conv2d": len(nonres),
                      "n_binarized": len(targets), "nonres_not_binarized": not_bin,
                      "reimplementation_matches_official_selection": True,
                      "reimplementation_weight_mismatches": re_mism, "exact_zero_binarized_weights": zeros,
                      "n_binary_weights": int(sum(latent[n].numel() for n in targets)),
                      "checkpoint_latent_weight_signature": latent_sig}}
    return model, targets, info


def binarization_modules(model) -> dict:
    return {n: m for n, m in model.named_modules() if type(m).__name__ in torch_bnas.BIN_OPS}


def run_reference(model, x_nhwc: np.ndarray, dtype, points: list[str], perm=None, batch: int = 16, collect_stats: bool = True):
    """Logits (standard class order) and exact zeros at the sign() inputs (BN outputs) of the emitted binarizations."""
    import torch

    m = model.to(dtype)
    stats = {"sign_input_exact_zero": 0, "n_binarized": 0}
    hooks = []
    mods = binarization_modules(m)
    if collect_stats:
        def count(_m, _i, out):
            stats["sign_input_exact_zero"] += int((out == 0).sum())
            stats["n_binarized"] += int(out.numel())
        for p in points:
            hooks.append(mods[p].bn.register_forward_hook(count))
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).numpy())
    for h in hooks:
        h.remove()
    model.to(torch.float32)
    y = np.concatenate(out)
    return (y[:, perm] if perm is not None else y), stats


def capture_binarization_inputs(model, x_nhwc: np.ndarray, points: list[str]) -> dict:
    """float64 forward of one sample; {binarization path: BN output handed to sign(), NHWC}."""
    import torch

    m = model.to(torch.float64)
    cap = {}
    mods = binarization_modules(m)
    hooks = [mods[p].bn.register_forward_hook(lambda _m, _i, o, p=p: cap.__setitem__(p, o.detach().numpy().transpose(0, 2, 3, 1)))
             for p in points]
    with torch.no_grad():
        m(torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))).to(torch.float64))
    for h in hooks:
        h.remove()
    model.to(torch.float32)
    return cap


# ------------------------------------------------------------- numpy executor workers
_G: dict = {}
_EXACT: dict = {}
_POINTS: list = []
_FLIPS: dict = {}   # BN path -> channels with negative scale (IR binarizes x + b/s, the source s*x + b)


def _init_worker(paths: dict) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
        if k in _EXACT:
            gx = bfformat.read(p)
            for name, v in _EXACT[k].items():
                assert np.array_equal(v.astype(np.float32), gx.constants[name].data.reshape(v.shape)), f"{name}: bfmodel differs from adapter output"
                gx.constants[name].data = v.reshape(gx.constants[name].data.shape)
            _G[k + "#exact"] = gx


def _run_one(job):
    key, x = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    zeros = [int(np.sum(env[f"{p}.bn:out"] == 0)) for p in _POINTS]
    out32 = env[g.outputs[0]].reshape(-1)
    del env
    env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    out64 = env64[g.outputs[0]].reshape(-1)
    pre64 = ({p: env64[f"{p}.bn:out"] * np.where(_FLIPS[f"{p}.bn"], -1.0, 1.0) for p in _POINTS}
             if key == "sign_zero_unfused" else None)
    del env64
    outx = executor.run(_G[key + "#exact"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1) if key + "#exact" in _G else None
    return out32, out64, zeros, outx, pre64


def first_flip(src_pre: dict, ir_pre: dict, order: list[str]) -> dict | None:
    """First binarization (forward order) whose sign differs between the float64 source and the float64 IR."""
    for name in order:
        a, b = src_pre[name], ir_pre[name].reshape(src_pre[name].shape)
        flips = np.sign(a) != np.sign(b)
        if flips.any():
            typical = float(np.median(np.abs(a)))
            mx = float(np.abs(a[flips]).max())
            return {"layer": name, "n_flips": int(flips.sum()), "max_abs_pre_at_flip": mx, "typical_abs_pre": typical,
                    "relative_to_typical": mx / typical if typical else None}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--repo", required=True, help="yonseivnl/bnas source tree (contains src/)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--big-eval-dir", help="large eval set (official fp32 logits -> reference_logits_eval<N>.f32)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--big-batch", type=int, default=100)
    ap.add_argument("--skip-numpy", action="store_true", help="write models + references only (no Level-3 numpy runs)")
    args = ap.parse_args()

    kind, _, pre_key = MODELS[args.model_id]
    reg = json.load(open(args.registry))
    entry = next(m for m in reg["models"] if m.get("model_id") == args.model_id)
    actual = sha256_file(args.weights)
    if actual != entry["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch: expected {entry['checkpoint']['sha256']}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    torch.set_num_threads(args.torch_threads)
    model, targets, src_info = load_source(args.model_id, args.repo, args.weights)
    perm = imagenet_class_perm() if kind == "imagenet" else None
    print(json.dumps({k: v for k, v in src_info.items() if k not in ("source_sha256", "genotype")}, indent=1, default=str), flush=True)

    def convert(binarize: str):
        return torch_bnas.convert(model, targets, args.model_id, binarize=binarize, class_perm=perm)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][pre_key]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    _, info0 = convert("sign_zero")
    points = info0["binarization_points"]
    global _POINTS, _FLIPS
    _POINTS = points
    _FLIPS = info0["bn_sign_flips"]
    unused = info0["unused_in_forward"]
    ref32, zeros32 = run_reference(model, x, torch.float32, points, perm)
    ref64, zeros64 = run_reference(model, x, torch.float64, points, perm)
    perm_check = None
    if perm is not None:   # the class-order hypothesis, checked on the eval set: permuted vs raw official outputs
        raw = ref32[:, np.argsort(perm)]
        perm_check = {"top1_standard_order": float(np.mean(ref32.argmax(1) == labels)),
                      "top1_raw_output_index_as_label": float(np.mean(raw.argmax(1) == labels))}
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    import torchvision
    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_preprocessing": pre_key,
              "reference": {"framework": f"torch {torch.__version__} (CPU), torchvision {torchvision.__version__}", "class": src_info["class"],
                            "repo": REPO_URL, "repo_commit": repo_commit(args.repo), "source_sha256": src_info["source_sha256"],
                            "config": src_info["config"], "genotype": src_info["genotype"], "repairs": src_info["repairs"],
                            "binop": src_info["binop"], "checkpoint_meta": src_info["checkpoint_meta"],
                            "class_order": ("standard ImageNet index (official outputs permuted: ImageFolder over folders named 0..999)"
                                            if perm is not None else "CIFAR-10 standard"), "class_order_check": perm_check,
                            "reference_f32_is": "official model fp32 (this process, CPU)",
                            "reference_f64_is": "official model in float64 after BinOp in float32 (model.double(): binarized weights exact)",
                            "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)),
                            "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                            "top5_accuracy_f64": float(np.mean([labels[i] in np.argsort(-ref64[i])[:5] for i in range(len(x))])),
                            "exact_zeros_at_binarization_f32": zeros32, "exact_zeros_at_binarization_f64": zeros64},
              "adapter": {"binarization_points": len(points),
                          "negative_bn_scale_channels (sign folded into weights)": int(sum(int(v.sum()) for v in _FLIPS.values())),
                          "binarized_float_input_convs": info0["binarized_float_input_convs"], "float_weight_convs_with_binary_input": info0["float_weight_convs_with_binary_input"],
                          "unused_in_forward (dead branches feeding only Zero ops)": unused},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels), "variants": {}}
    print(json.dumps({k: v for k, v in report["reference"].items() if k not in ("source_sha256", "genotype", "binop", "repairs")}, indent=1, default=str), flush=True)
    print("adapter:", json.dumps(report["adapter"]), flush=True)

    files = {}
    for binarize in ("ge", "sign_zero"):
        for variant, fold in (("unfused", False), ("fused", True)):
            graph, info = convert(binarize)
            applied = passes.optimize(graph, fold=fold)
            stem = "model" if binarize == "ge" else "model_sign_zero"
            key = variant if binarize == "ge" else f"sign_zero_{variant}"
            graph.manifest = {
                "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": f"{variant}, input_binarize={binarize}",
                "source": {"framework": "pytorch", "class": src_info["class"], "repo": REPO_URL, "repo_commit": report["reference"]["repo_commit"],
                           "source_sha256": src_info["source_sha256"], "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
                "conversion": {"tool": "tools/convert_bnas.py", "adapter": torch_bnas.__name__, "passes": applied,
                               "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": platform.python_version(),
                               "torch": torch.__version__, "input_binarize": binarize,
                               "binarize_note": ("exact sign semantics (sign(0)=0); reference kernels only" if binarize == "sign_zero"
                                                 else "deployable: 0 -> +1 at binarization; differs from sign only for exact ties"),
                               "class_order": report["reference"]["class_order"]},
                "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32", "preprocessing": pre["definition"]},
                "outputs": ["logits"], "op_versions": {"format": 1},
                "layer_precision": info["layer_precision"], "weight_diagnostics": info["weight_diagnostics"], "unused_in_forward": unused,
            }
            if not fold:
                _EXACT[key] = info["float64_constants"]
            path = os.path.join(args.out_dir, f"{stem}.bitino" if fold else f"{stem}_unfused.bitino")
            file_sha = bfformat.write(graph, path)
            files[key] = path
            report["variants"][key] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": binarize, "op_counts": graph.op_counts(), "passes": applied,
                                       "zero_channel_corrections": info["zero_channel_corrections"]}
            print(f"[{key}] wrote {path}: {graph.op_counts()} passes={applied}", flush=True)
            del graph, info

    if not args.skip_numpy:
        jobs = [(k, x[i:i + 1]) for k in files for i in range(len(x))]
        with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
            results = pool.map(_run_one, jobs, chunksize=2)
        by_key = {k: results[j * len(x):(j + 1) * len(x)] for j, k in enumerate(files)}
        np_logits = {}
        for k, res in by_key.items():
            out32 = np.stack([r[0] for r in res]).astype(np.float32)
            out64 = np.stack([r[1] for r in res])
            zeros = np.array([r[2] for r in res])
            np_logits[k] = out32
            out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
            sem = compare_outputs(ref64, out64, labels)
            per_sample = np.abs(out64 - ref64).max(axis=1)
            sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per_sample <= 1e-6))
            sem["n_samples_max_abs_le_1e-4"] = int(np.sum(per_sample <= 1e-4))
            sem["per_sample_max_abs"] = [float(v) for v in per_sample]
            if res[0][3] is not None:
                outx = np.stack([r[3] for r in res])
                px = np.abs(outx - ref64).max(axis=1)
                sem["float64_constants"] = {"max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-6": int(np.sum(px <= 1e-6)),
                                            "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                                            "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1)))}
            if k == "sign_zero_unfused":   # first-flip analysis of every deviating sample (float64 IR, float32 constants)
                ff = []
                for i in np.where(per_sample > 1e-6)[0]:
                    src_pre = capture_binarization_inputs(model, x[i:i + 1], points)
                    f = first_flip(src_pre, res[i][4], points)
                    ff.append({"sample": int(i), "max_abs_logit_diff": float(per_sample[i]),
                               "top1_changed": bool(out64[i].argmax() != ref64[i].argmax()), "first_flip": f})
                no_flip = [d for d in ff if d["first_flip"] is None]
                with_flip = [d for d in ff if d["first_flip"] is not None]
                sem["first_flip_analysis"] = {
                    "n_analyzed": len(ff), "n_without_flip": len(no_flip),
                    "max_abs_logit_diff_without_flip": max([d["max_abs_logit_diff"] for d in no_flip], default=0.0),
                    "n_with_flip": len(with_flip),
                    "max_abs_pre_at_first_flip": max([d["first_flip"]["max_abs_pre_at_flip"] for d in with_flip], default=None),
                    "max_relative_to_typical": max([d["first_flip"]["relative_to_typical"] for d in with_flip], default=None),
                    "first_flip_layers": sorted({d["first_flip"]["layer"] for d in with_flip}),
                    "n_top1_changes": int(sum(d["top1_changed"] for d in ff)), "samples": ff}
            report["variants"][k].update({
                "numpy_vs_reference_f32": compare_outputs(ref32, out32, labels), "numpy_vs_reference_f64": compare_outputs(ref64, out32, labels),
                "semantic_check_ir_f64_vs_reference_f64": sem,
                "exact_zeros_at_binarization_numpy_f32": {"total": int(zeros.sum()), "samples_with_zeros": int(np.sum(zeros.sum(1) > 0))}})
            print(f"[{k}] vs f32 ref: top1 agree {report['variants'][k]['numpy_vs_reference_f32']['top1_agreement']:.4f} | SEMANTIC f64: max_abs "
                  f"{sem['max_abs_error']:.3e} <=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{len(x)} <=1e-4: {sem['n_samples_max_abs_le_1e-4']} | zeros(np f32) {int(zeros.sum())}"
                  + (f" | f64 consts max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else "")
                  + (f" | first-flip: {sem['first_flip_analysis']['n_with_flip']} with flip (max rel {sem['first_flip_analysis']['max_relative_to_typical']})"
                     if "first_flip_analysis" in sem else ""), flush=True)
        report["ge_vs_sign_zero_numpy_f32"] = {v: compare_outputs(np_logits[f"sign_zero_{v}"], np_logits[v], labels) for v in ("unfused", "fused")}
        report["fused_vs_unfused_numpy_f32"] = compare_outputs(np_logits["unfused"], np_logits["fused"], labels)
        json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)

    if args.big_eval_dir:
        bm = json.load(open(os.path.join(args.big_eval_dir, "manifest.json")))
        bp = bm["preprocessing"][pre_key]
        n = int(bp["shape"][0])
        xb = np.memmap(os.path.join(args.big_eval_dir, bp["file"]), np.float32, "r", shape=tuple(bp["shape"]))
        yb = np.array([s["label"] for s in bm["samples"]][:n])
        lb, zb = run_reference(model, xb, torch.float32, points, perm, batch=args.big_batch)
        lb.astype(np.float32).tofile(os.path.join(args.out_dir, f"reference_logits_eval{n}.f32"))
        big = {"eval_set": bm["name"], "n": n, "source_fp32_top1": float(np.mean(lb.argmax(1) == yb)),
               "source_fp32_top5": float(np.mean([yb[i] in np.argsort(-lb[i])[:5] for i in range(n)])),
               "exact_zeros_at_binarization": zb}
        report["large_eval_source"] = big
        print("large eval:", json.dumps(big), flush=True)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
