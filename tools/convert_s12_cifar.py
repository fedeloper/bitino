#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Converter for the slice-S12 CIFAR-10 ResNet-20s: AdaBin (official PyTorch, huawei-noah/Efficient-Computing)
and FDA-BNN (MindSpore model zoo checkpoint; reference = the cited PyTorch port tools/fdabnn_reference.py,
cross-checked against the official MindSpore code run by tools/fdabnn_mindspore_reference.py when available).

  python3 -I tools/convert_s12_cifar.py adabin --model-id adabin-resnet20-cifar10 \
      --repo .cache/repos/Efficient-Computing --weights .cache/checkpoints/adabin-resnet20-cifar10/model.th \
      --eval-dir .cache/data/eval128_cifar10_norm --big-eval-dir .cache/data/eval10000_cifar10_norm \
      --out-dir artifacts/adabin-resnet20-cifar10
  python3 -I tools/convert_s12_cifar.py fdabnn --model-id fda-bnn-resnet20-cifar10-mindspore \
      --repo .cache/repos/mindspore-models-FDA-BNN --weights .cache/checkpoints/fda-bnn-resnet20-cifar10-mindspore/fdabnn.ckpt \
      --eval-dir .cache/data/eval128_cifar10_norm --big-eval-dir .cache/data/eval10000_cifar10_norm \
      --mindspore-logits artifacts/fda-bnn-resnet20-cifar10-mindspore/mindspore_logits_eval128.f32 \
      --out-dir artifacts/fda-bnn-resnet20-cifar10-mindspore

Outputs (same layout as tools/convert_torch_cnn.py): model{,_unfused,_sign_zero,_sign_zero_unfused}.bitino,
reference_logits.f32 / reference_logits_f64.f64 (primary float64 reference, see below), reference_logits_f64_naive.f64,
numpy_logits_<variant>.f32, conversion_report.json (Level 3, float64-constants check, first-flip analysis of every
sample deviating by more than 1e-6) and, with --big-eval-dir, reference_logits_eval10000.f32.

Float64 references. AdaBin: the weight levels W+/W- = sign*alpha_w + beta_w are a function of the parameters; the
primary float64 reference runs the official model in float64 with the fp32 levels substituted at its F.conv2d call
(the converter checks they equal the adapter's levels bit for bit); ``model.double()`` as is = the naive reference.
FDA-BNN: the port in float64 (scale mean|W| in float64, as the IR's exact constants).
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import platform
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import mindspore_ckpt, mindspore_fdabnn, torch_adabin  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402
from convert_torch_cnn import _install_proxy, repo_commit, sha256_file  # noqa: E402

FAMILIES = {
    "adabin": {"archs": {"adabin-resnet20-cifar10": "resnet20_1w1a"},
               "repo_url": "https://github.com/huawei-noah/Efficient-Computing (BinaryNetworks/AdaBin/cifar10)",
               "adapter": torch_adabin, "tool": "tools/convert_adabin.py",
               # cifar10/main.py:124-141: ToTensor + Normalize(ImageNet mean/std)
               "preprocess": "cifar10_imagenet_norm"},
    "fdabnn": {"archs": {"fda-bnn-resnet20-cifar10-mindspore": "resnet20"},
               "repo_url": "https://gitee.com/mindspore/models (research/cv/FDA-BNN)",
               "adapter": mindspore_fdabnn, "tool": "tools/convert_fdabnn.py",
               # src/dataset.py create_dataset_cifar10: Rescale(1/255) + Normalize(ImageNet mean/std), HWC2CHW
               "preprocess": "cifar10_imagenet_norm"},
}


def load_source(family: str, repo: str, weights: str | None):
    """Returns (torch model in eval mode, class string, {source file: sha256}, load notes)."""
    import importlib

    import torch

    sys.dont_write_bytecode = True
    if family == "adabin":
        sub = os.path.join(os.path.abspath(repo), "BinaryNetworks", "AdaBin", "cifar10")
        for k in list(sys.modules):
            if k.split(".")[0] in ("utils", "nets"):
                del sys.modules[k]
        saved = list(sys.path)
        # AdaBin's utils/ and nets/ have no __init__.py (namespace packages), and a regular utils.py / nets.py anywhere on
        # sys.path wins over a namespace package even when listed later, so drop every entry that has one.
        sys.path[:] = [sub] + [q for q in sys.path if not any(os.path.exists(os.path.join(q or ".", x))
                                                              for x in ("utils", "utils.py", "nets", "nets.py"))]
        try:
            zoo = importlib.import_module("nets.resnet20")
            bl = importlib.import_module("utils.binarylib")
        finally:
            sys.path[:] = saved
        model = zoo.resnet20_1w1a()
        src = {os.path.relpath(f, repo): sha256_file(f) for f in (zoo.__file__, bl.__file__)}
        notes = {}
        if weights is not None:
            ck = torch.load(weights, map_location="cpu", weights_only=True)
            assert set(ck) == {"state_dict", "best_prec1"}, sorted(ck)
            model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["state_dict"].items()}, strict=True)
            notes = {"checkpoint_meta": {"best_prec1": float(ck["best_prec1"])}}
        return model.eval(), "cifar10/nets/resnet20.resnet20_1w1a()", src, notes
    import fdabnn_reference
    params = mindspore_ckpt.read(weights)
    model = fdabnn_reference.build(params)
    src = {f: sha256_file(os.path.join(repo, f)) for f in ("quan.py", "resnet.py")}
    src["tools/fdabnn_reference.py (port)"] = sha256_file(fdabnn_reference.__file__)
    return model, "tools/fdabnn_reference.ResNet20 (PyTorch port of src/resnet.py + src/quan.py)", src, {"params": params}


def binarization_modules(family: str, model):
    """{module path: module} of the binary convs, in forward order."""
    cls = "AdaBin_Conv2d" if family == "adabin" else "QuanConv"
    return {n: m for n, m in model.named_modules() if type(m).__name__ == cls}


def threshold_of(family: str, mod) -> float:
    return float(mod.binary_a.beta_a.detach()) if family == "adabin" else 0.0


def run_reference(family: str, model, x_nhwc: np.ndarray, dtype, override=None, capture=None, batch: int = 16):
    """Logits plus exact-zero counts at binarization (x == threshold, where the source sign gives 0) and, for
    AdaBin, a check that gradient_approx returns torch.sign exactly."""
    import torch

    m = model.to(dtype)
    stats = {"sign_input_exact_zero": 0, "n_binarized": 0}
    hooks = []
    convs = binarization_modules(family, m)
    def count(_m, i, thr):
        stats["sign_input_exact_zero"] += int((i[0] == thr).sum())
        stats["n_binarized"] += int(i[0].numel())

    for name, mod in convs.items():
        hooks.append(mod.register_forward_pre_hook(lambda _m, i, thr=threshold_of(family, mod): count(_m, i, thr)))
    if family == "adabin":
        stats["gradient_approx_ne_torch_sign"] = 0
        for mod in convs.values():
            ba = mod.binary_a
            orig = ba.gradient_approx

            def ga(xx, orig=orig):
                out = orig(xx)
                stats["gradient_approx_ne_torch_sign"] += int((out != torch.sign(xx)).sum())
                return out
            ba.gradient_approx = ga
    proxy = _install_proxy(m, "AdaBin_Conv2d", capture, override) if family == "adabin" and (capture is not None or override is not None) else None
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).numpy())
    for h in hooks:
        h.remove()
    if family == "adabin":
        for mod in convs.values():
            del mod.binary_a.gradient_approx
    if proxy is not None:
        proxy[1]()
    model.to(torch.float32)
    return np.concatenate(out), stats


def capture_binarization_inputs(family: str, model, x_nhwc: np.ndarray, override=None) -> dict:
    """float64 forward of one sample; {conv path: value handed to sign (x - threshold), NHWC}."""
    import torch

    m = model.to(torch.float64)
    cap = {}
    hooks = []
    for name, mod in binarization_modules(family, m).items():
        thr = threshold_of(family, mod)
        hooks.append(mod.register_forward_pre_hook(
            lambda _m, i, name=name, thr=thr: cap.__setitem__(name, (i[0].detach() - thr).numpy().transpose(0, 2, 3, 1))))
    proxy = _install_proxy(m, "AdaBin_Conv2d", None, override) if family == "adabin" else None
    with torch.no_grad():
        m(torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))).to(torch.float64))
    for h in hooks:
        h.remove()
    if proxy is not None:
        proxy[1]()
    model.to(torch.float32)
    return cap


# ------------------------------------------------------------- numpy executor workers
_G: dict = {}
_EXACT: dict = {}


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


def _main_bconvs(g):
    return [n for n in g.nodes if n.op == "BConv2D" and not n.name.endswith("/sum_s")]


def _run_one(job):
    key, x = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    zeros = [int(np.sum(env[n.inputs[0]] == (g.constants[n.inputs[4]].data if len(n.inputs) > 4 and n.inputs[4] else 0)))
             for n in _main_bconvs(g)]
    out32 = env[g.outputs[0]].reshape(-1)
    env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    out64 = env64[g.outputs[0]].reshape(-1)
    pre64 = {}
    for n in _main_bconvs(g):
        t = g.constants[n.inputs[4]].data.astype(np.float64) if len(n.inputs) > 4 and n.inputs[4] else 0.0
        pre64[n.name] = env64[n.inputs[0]] - t
    outx = executor.run(_G[key + "#exact"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1) if key + "#exact" in _G else None
    return out32, out64, zeros, outx, pre64


def first_flip(src_pre: dict, ir_pre: dict, order: list[str]) -> dict | None:
    """First binary conv (forward order) whose binarization differs between the float64 source and the float64 IR."""
    for name in order:
        a, b = src_pre[name], ir_pre[name].reshape(src_pre[name].shape)
        flips = np.sign(a) != np.sign(b)
        if flips.any():
            typical = float(np.median(np.abs(a)))
            mx = float(np.abs(a[flips]).max())
            return {"layer": name, "n_flips": int(flips.sum()), "max_abs_pre_at_flip": mx, "typical_abs_pre": typical,
                    "relative_to_typical": mx / typical if typical else None}
    return None


def main(family: str) -> int:
    fam = FAMILIES[family]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(fam["archs"]))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--big-eval-dir", help="large eval set (source fp32 logits -> reference_logits_eval<N>.f32)")
    ap.add_argument("--mindspore-logits", help="FDA-BNN: official MindSpore fp32 logits on --eval-dir (becomes reference_logits.f32)")
    ap.add_argument("--mindspore-logits-big", help="FDA-BNN: official MindSpore fp32 logits on --big-eval-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=4)
    args = ap.parse_args()

    reg = json.load(open(args.registry))
    entry = next(m for m in reg["models"] if m.get("model_id") == args.model_id)
    actual = sha256_file(args.weights)
    if actual != entry["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch: expected {entry['checkpoint']['sha256']}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}")
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    torch.set_num_threads(args.torch_threads)
    model, cls, src_hashes, notes = load_source(family, args.repo, args.weights)
    params = notes.pop("params", None)

    def convert(binarize: str):
        if family == "adabin":
            return torch_adabin.convert(model, args.model_id, binarize=binarize)
        return mindspore_fdabnn.convert(params, args.model_id, binarize=binarize)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][fam["preprocess"]]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    _, info0 = convert("sign_zero")
    weight_derivation = None
    override = None
    if family == "adabin":
        cap32, cap64 = {}, {}
        ref32, zeros32 = run_reference(family, model, x, torch.float32, capture=cap32)
        ref64_naive, _ = run_reference(family, model, x, torch.float64, capture=cap64)
        # the adapter's levels == the tensor the fp32 source passes to F.conv2d (bit for bit)
        mism = 0
        for k, (wp, wm) in info0["weight_levels"].items():
            s = torch.from_numpy(info0["binary_weights"][k])
            want = torch.where(s > 0, torch.from_numpy(wp).view(-1, 1, 1, 1), torch.from_numpy(wm).view(-1, 1, 1, 1))
            mism += int((want != cap32[k]).sum())
        assert mism == 0, f"adapter weight levels differ from the source F.conv2d weights at {mism} positions"
        override = {k: v.double() for k, v in cap32.items()}
        ref64, zeros64 = run_reference(family, model, x, torch.float64, override=override)
        lv_diff = {k: float((cap64[k] - cap32[k].double()).abs().max()) for k in cap32}
        sign_diff = int(sum(int(((cap64[k] - cap64[k].mean((1, 2, 3), keepdim=True)).sign() !=
                                 torch.from_numpy(info0["binary_weights"][k]).double()).sum()) for k in cap32))
        weight_derivation = {"adapter_levels_vs_source_fp32_weight_mismatches": mism,
                             "fp64_vs_fp32_level_max_abs_difference": max(lv_diff.values()),
                             "fp64_vs_fp32_weight_sign_differences": sign_diff,
                             "primary_f64_reference": "official model in float64 with the fp32 weight levels substituted (F.conv2d proxy)",
                             "naive_f64_vs_primary_f64": compare_outputs(ref64, ref64_naive, labels)}
        ref64_naive.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))
    else:
        ref32, zeros32 = run_reference(family, model, x, torch.float32)
        ref64, zeros64 = run_reference(family, model, x, torch.float64)
        ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))
        weight_derivation = {"primary_f64_reference": "PyTorch port in float64 (scale mean|W| in float64 from the fp32 weights)"}
    port32 = ref32
    official = None
    if family == "fdabnn" and args.mindspore_logits:
        ms32 = np.fromfile(args.mindspore_logits, np.float32).reshape(ref32.shape)
        official = {"mindspore_fp32_vs_port_fp32": compare_outputs(ms32, port32, labels),
                    "mindspore_fp32_vs_port_fp64": compare_outputs(ref64, ms32, labels),
                    "file": os.path.basename(args.mindspore_logits)}
        port32.astype(np.float32).tofile(os.path.join(args.out_dir, "reference_logits_port_f32.f32"))
        ref32 = ms32
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))

    import torchvision
    report = {"model_id": args.model_id, "n_samples": int(len(x)), "eval_preprocessing": fam["preprocess"],
              "reference": {"framework": f"torch {torch.__version__} (CPU), torchvision {torchvision.__version__}", "class": cls,
                            "repo_commit": repo_commit(args.repo), "source_sha256": src_hashes,
                            "reference_f32_is": "official MindSpore fp32 (tools/fdabnn_mindspore_reference.py)" if official else "source fp32 (this process)",
                            "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)),
                            "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                            "top5_accuracy_f64": float(np.mean([labels[i] in np.argsort(-ref64[i])[:5] for i in range(len(x))])),
                            "exact_zeros_at_binarization_f32": zeros32, "exact_zeros_at_binarization_f64": zeros64,
                            "weight_derivation": weight_derivation, "load_notes": notes},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels), "variants": {}}
    if official:
        report["reference"]["official_mindspore"] = official
        report["reference"]["port_f32_top1_accuracy"] = float(np.mean(port32.argmax(1) == labels))
    print(json.dumps({k: v for k, v in report["reference"].items() if k != "source_sha256"}, indent=1, default=str))

    files = {}
    for binarize in ("ge", "sign_zero"):
        for variant, fold in (("unfused", False), ("fused", True)):
            graph, info = convert(binarize)
            applied = passes.optimize(graph, fold=fold)
            stem = "model" if binarize == "ge" else "model_sign_zero"
            key = variant if binarize == "ge" else f"sign_zero_{variant}"
            graph.manifest = {
                "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": f"{variant}, input_binarize={binarize}",
                "source": {"framework": "pytorch" if family == "adabin" else "mindspore (checkpoint read without MindSpore)", "class": cls,
                           "repo": fam["repo_url"], "repo_commit": report["reference"]["repo_commit"], "source_sha256": src_hashes,
                           "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
                "conversion": {"tool": fam["tool"], "adapter": fam["adapter"].__name__, "passes": applied,
                               "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": platform.python_version(),
                               "torch": torch.__version__, "input_binarize": binarize,
                               "binarize_note": ("exact sign semantics (sign(0)=0); reference kernels only" if binarize == "sign_zero"
                                                 else "deployable: 0 -> +1 at binarization; differs from sign only for exact ties")},
                "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32", "preprocessing": pre["definition"]},
                "outputs": ["logits"], "op_versions": {"format": 1},
                "layer_precision": info["layer_precision"], "weight_diagnostics": info["weight_diagnostics"], "unused_in_forward": [],
            }
            if not fold:
                _EXACT[key] = info["float64_constants"]
            path = os.path.join(args.out_dir, f"{stem}.bitino" if fold else f"{stem}_unfused.bitino")
            file_sha = bfformat.write(graph, path)
            files[key] = path
            report["variants"][key] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": binarize, "op_counts": graph.op_counts(), "passes": applied}
            print(f"[{key}] wrote {path}: {graph.op_counts()} passes={applied}")

    jobs = [(k, x[i:i + 1]) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
        results = pool.map(_run_one, jobs, chunksize=4)
    by_key = {k: results[j * len(x):(j + 1) * len(x)] for j, k in enumerate(files)}
    np_logits = {}
    order = list(binarization_modules(family, model))
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
                src_pre = capture_binarization_inputs(family, model, x[i:i + 1], override)
                f = first_flip(src_pre, res[i][4], order)
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
            "exact_zeros_at_bconv_inputs_numpy_f32": {"total": int(zeros.sum()), "samples_with_zeros": int(np.sum(zeros.sum(1) > 0))}})
        print(f"[{k}] vs f32 ref: top1 agree {report['variants'][k]['numpy_vs_reference_f32']['top1_agreement']:.4f} | SEMANTIC f64: max_abs "
              f"{sem['max_abs_error']:.3e} <=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{len(x)} | zeros(np f32) {int(zeros.sum())}"
              + (f" | f64 consts max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else "")
              + (f" | first-flip: {sem['first_flip_analysis']['n_with_flip']} with flip (max rel {sem['first_flip_analysis']['max_relative_to_typical']})"
                 if "first_flip_analysis" in sem else ""))
    report["ge_vs_sign_zero_numpy_f32"] = {v: compare_outputs(np_logits[f"sign_zero_{v}"], np_logits[v], labels) for v in ("unfused", "fused")}
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(np_logits["unfused"], np_logits["fused"], labels)

    if args.big_eval_dir:
        bm = json.load(open(os.path.join(args.big_eval_dir, "manifest.json")))
        bp = bm["preprocessing"][fam["preprocess"]]
        xb = np.fromfile(os.path.join(args.big_eval_dir, bp["file"]), np.float32).reshape(bp["shape"])
        yb = np.array([s["label"] for s in bm["samples"]][: len(xb)])
        lb, zb = run_reference(family, model, xb, torch.float32, batch=500)
        n = len(xb)
        lb.astype(np.float32).tofile(os.path.join(args.out_dir, f"reference_logits_eval{n}.f32"))
        big = {"eval_set": bm["name"], "n": n, "source_fp32_top1": float(np.mean(lb.argmax(1) == yb)),
               "source_fp32_top5": float(np.mean([yb[i] in np.argsort(-lb[i])[:5] for i in range(n)])),
               "source_fp32_top1_first_9984 (drop_last batches of the official eval)": float(np.mean(lb[:9984].argmax(1) == yb[:9984])),
               "exact_zeros_at_binarization": zb}
        if family == "fdabnn":
            big["source_is"] = "PyTorch port fp32"
            if args.mindspore_logits_big:
                msb = np.fromfile(args.mindspore_logits_big, np.float32).reshape(lb.shape)
                big["mindspore_fp32_top1"] = float(np.mean(msb.argmax(1) == yb))
                big["mindspore_fp32_top1_first_9984"] = float(np.mean(msb[:9984].argmax(1) == yb[:9984]))
                big["mindspore_vs_port_fp32"] = compare_outputs(msb, lb, yb)
        report["large_eval_source"] = big
        print("large eval:", json.dumps({k: v for k, v in big.items() if k != "mindspore_vs_port_fp32"}))
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv.pop(1)))
