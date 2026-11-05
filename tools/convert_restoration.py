#!/usr/bin/env python3
"""Shared converter for PyTorch binary image-to-image models (slice S8): BBCU denoising / JPEG deblocking /
super-resolution (Zj-BinXia/BBCU, adapter torch_bbcu) and the BSRC-2023 binary SR baselines
(ilyazhara/BSRC-2023, adapter torch_bsrc). Entry points: tools/convert_bbcu.py, tools/convert_bsrc.py.

  python3 -I tools/convert_bbcu.py --model-id bbcu-denoise --repo .cache/repos/BBCU \
      --weights .cache/checkpoints/bbcu-denoise/BBCU-denoise.pth \
      --eval-dir .cache/data/eval68_cbsd68_sigma25_c96 --out-dir artifacts/bbcu-denoise
  python3 -I tools/convert_bsrc.py --model-id bsrc2023-binary-sr-x2-v0 --repo .cache/repos/BSRC-2023-62bbcf2 \
      --weights .cache/checkpoints/bsrc2023-binary-sr/scalex2_baseline_v0.ckpt \
      --eval-dir .cache/data/eval64_sr_x2_lr48 --out-dir artifacts/bsrc2023-binary-sr-x2-v0

The official model definitions are executed from the clones (read only, no bytecode written); checkpoints are
loaded with torch.load(weights_only=True) (BSRC whole-module pickles: only inert stub classes are allow-listed,
see torch_bsrc.load_checkpoint). The graph input shape is the eval set's crop size; outputs are images (NHWC).

The primary float64 reference runs the official model in float64 with the binary weights replaced at the
F.conv2d call of each binary conv by exactly ``sign bits * float64 scale`` (the adapter's float64 constants;
the fp32 STE expressions perturb the binary weights at the rounding level). ``model.double()`` as is gives
reference_logits_f64_naive.f64.

Produces in --out-dir (file names shared with the classifier converters; "logits" = output image, NHWC):
  model.bitino / model_unfused.bitino                      deployable binarization (BBCU: "ge", 0 -> +1;
                                                             BSRC: the exact source semantics)
  model_sign_zero.bitino / model_sign_zero_unfused.bitino  BBCU only: exact torch.sign (reference kernels only)
  reference_logits.f32 / reference_logits_f64.f64            official model outputs, float32 and float64
  numpy_logits_<variant>.f32                                 numpy executor (float32) per variant
  conversion_report.json                                     Level 3, exact-zero counts, PSNR/SSIM on the crops
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
from bitino.adapters import torch_bbcu, torch_bsrc  # noqa: E402
from bitino.validation.image_metrics import compare_images, image_quality  # noqa: E402
from convert_torch_cnn import _install_proxy, _load_source_file, _purge, repo_commit, sha256_file  # noqa: E402

BBCU_URL = "https://github.com/Zj-BinXia/BBCU"
BSRC_URL = "https://github.com/ilyazhara/BSRC-2023"
# model_id -> (arch, source file, constructor kwargs, eval preprocessing key, metric override, output range)
BBCU = {
    "bbcu-denoise": ("dncnn", "BBCU-denoiseAndblocking/models/network_dncnn_bin.py",
                     dict(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R", k=130.0), "bbcu_denoise_s25"),
    "bbcu-deblocking": ("dncnn", "BBCU-denoiseAndblocking/models/network_dncnn3_bin.py",
                        dict(in_nc=1, out_nc=1, nc=64, nb=20, act_mode="R", k=130.0), "bbcu_jpeg_y"),
    "bbcu-sr-l-x2": ("bbcul", "BBCU-SR/bbcu/archs/bbcul_arch.py", dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=2, k=130.0), "sr_x2_unit"),
    "bbcu-sr-l-x4": ("bbcul", "BBCU-SR/bbcu/archs/bbcul_arch.py", dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4, k=130.0), "sr_x4_unit"),
    "bbcu-sr-m-x2": ("bbcum", "BBCU-SR/bbcu/archs/bbcum_arch.py", dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=2, k=130.0), "sr_x2_unit"),
    "bbcu-sr-m-x4": ("bbcum", "BBCU-SR/bbcu/archs/bbcum_arch.py", dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4, k=130.0), "sr_x4_unit"),
}
FAMILIES = {
    "bbcu": {"archs": {k: v[0] for k, v in BBCU.items()}, "repo_url": BBCU_URL, "adapter": torch_bbcu,
             "binarize_modes": ("ge", "sign_zero"), "sign_module": "BinaryActivation", "conv_module": "HardBinaryConv",
             "output_range": 1.0},
    "bsrc": {"archs": {k: v["arch"] for k, v in torch_bsrc.MODELS.items()}, "repo_url": BSRC_URL, "adapter": torch_bsrc,
             "binarize_modes": ("ge",), "sign_module": None, "conv_module": "BinaryConv2d", "output_range": 255.0},
}


def preprocess_key(family: str, model_id: str) -> str:
    return BBCU[model_id][3] if family == "bbcu" else torch_bsrc.MODELS[model_id]["preprocess"]


def metric_of(family: str, manifest: dict) -> tuple[str, int]:
    it = manifest["image_task"]
    if family == "bsrc":
        return "bsrc_y", int(it["crop_border"])
    return it["metric"], int(it["crop_border"])


# ------------------------------------------------------------------ source loading
def _load_bbcu(repo: str, model_id: str, weights: str | None):
    """Executes the official files from the clone under stub package names (no sys.path change):
    denoise/deblock: models/basicblock.py (as ``models.basicblock``) and quants/quant_convs_bbcu.py (as
    ``quants.quant_convs_bbcu``), then models/network_dncnn{,3}_bin.py. SR: BBCU-SR/quant_convs_bbcu.py (as
    ``quant_convs_bbcu``) and bbcu/archs/bbcu{l,m}_arch.py with ``bbcu.utils.registry.ARCH_REGISTRY.register()``
    stubbed to the identity and ``bbcu.archs.arch_util`` providing ``make_layer`` (arch_util.py:46-59: an
    nn.Sequential of ``num_basic_block`` blocks built with the same kwargs) and ``default_init_weights`` as a
    no-op (initialization only; every tensor is overwritten by the strict load). The real arch_util.py is not
    executed: it imports bbcu.utils (the whole BasicSR training stack) and distutils (removed in Python 3.12)."""
    import types

    import torch
    import torch.nn as nn

    sys.dont_write_bytecode = True
    arch, rel, kwargs, _ = BBCU[model_id]
    root = os.path.abspath(repo)
    path = os.path.join(root, rel)
    src = {rel: sha256_file(path)}
    stubs = []

    def pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        stubs.append(name)
        return m

    try:
        if arch == "dncnn":
            base = os.path.join(root, "BBCU-denoiseAndblocking")
            pkg("models")
            pkg("quants")
            for name, f in (("models.basicblock", "models/basicblock.py"), ("quants.quant_convs_bbcu", "quants/quant_convs_bbcu.py")):
                _load_source_file(name, os.path.join(base, f))
                src[os.path.join("BBCU-denoiseAndblocking", f)] = sha256_file(os.path.join(base, f))
            mod = _load_source_file("bbcu_" + os.path.basename(rel)[:-3], path)
            model = mod.DnCNN(**kwargs)
            cls = f"{rel} DnCNN({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
        else:
            base = os.path.join(root, "BBCU-SR")
            _load_source_file("quant_convs_bbcu", os.path.join(base, "quant_convs_bbcu.py"))
            stubs.append("quant_convs_bbcu")
            src["BBCU-SR/quant_convs_bbcu.py"] = sha256_file(os.path.join(base, "quant_convs_bbcu.py"))
            pkg("bbcu")
            pkg("bbcu.utils")
            reg = types.ModuleType("bbcu.utils.registry")
            reg.ARCH_REGISTRY = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
            sys.modules["bbcu.utils.registry"] = reg
            stubs.append("bbcu.utils.registry")
            pkg("bbcu.archs")
            au = types.ModuleType("bbcu.archs.arch_util")

            def make_layer(basic_block, num_basic_block, **kwarg):   # arch_util.py:46-59
                return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])

            au.make_layer = make_layer
            au.default_init_weights = lambda *a, **k: None
            sys.modules["bbcu.archs.arch_util"] = au
            stubs.append("bbcu.archs.arch_util")
            name = "bbcu.archs." + os.path.basename(rel)[:-3]
            mod = _load_source_file(name, path)
            stubs.append(name)
            model = getattr(mod, {"bbcul": "BBCUL", "bbcum": "BBCUM"}[arch])(**kwargs)
            cls = f"{rel} {type(model).__name__}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())}) [arch_util/registry stubbed]"
    finally:
        _purge({s.split('.')[0] for s in stubs})
    if weights is None:
        return model.eval(), cls, src
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    if arch != "dncnn":
        assert set(sd) == {"params"}, f"unexpected checkpoint keys {sorted(sd)}"   # BasicSR param_key_g: params
        sd = sd["params"]
    assert all(isinstance(v, torch.Tensor) for v in sd.values()), "checkpoint is not a flat state_dict"
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cls, src


def load_model(family: str, repo: str, model_id: str, weights: str | None):
    if family == "bbcu":
        return _load_bbcu(repo, model_id, weights)
    return torch_bsrc.load_model(repo, model_id, weights)


def convert_graph(family: str, model, model_id: str, binarize: str, input_hw):
    if family == "bbcu":
        arch = BBCU[model_id][0]
        return torch_bbcu.convert(model, arch, model_id, input_hw=input_hw, binarize=binarize)
    return torch_bsrc.convert(model, model_id, input_hw=input_hw, binarize=binarize)


# ------------------------------------------------------------------ references
def _sign_hooks(model, family: str, stats: dict):
    """Exact-zero counts at every binarization input and a check that the source's binarizer returns exactly the
    claimed function (BBCU: torch.sign; BSRC input binarizers: +1 iff x >= 0; BSRC LAB: +1 iff x1 > x0)."""
    import torch

    hooks = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if family == "bbcu" and cls == "BinaryActivation":
            def fwd(_m, inp, out):
                x = inp[0]
                stats["sign_input_exact_zero"] += int((x == 0).sum())
                stats["n_binarized"] += int(x.numel())
                stats["binarizer_output_ne_claimed"] += int((out != torch.sign(x)).sum())
            hooks.append(mod.register_forward_hook(fwd))
        elif family == "bsrc" and name.endswith("input_binarizer"):
            def fwd(_m, inp, out):
                x = inp[0]
                stats["sign_input_exact_zero"] += int((x == 0).sum())
                stats["n_binarized"] += int(x.numel())
                stats["binarizer_output_ne_claimed"] += int((out != torch.where(x >= 0, 1.0, -1.0).to(out.dtype)).sum())
            hooks.append(mod.register_forward_hook(fwd))
        elif family == "bsrc" and cls == "LAB":
            def fwd(m, inp, out):
                x = m.fp_depthwise_conv(inp[0])
                n, c2, h, w = x.shape
                x = x.reshape(n, c2 // 2, 2, h, w)
                stats["lab_exact_ties"] = stats.get("lab_exact_ties", 0) + int((x[:, :, 1] == x[:, :, 0]).sum())
                want = torch.where(x[:, :, 1] > x[:, :, 0], 1.0, -1.0).to(out.dtype)
                stats["binarizer_output_ne_claimed"] += int((out != want).sum())
            hooks.append(mod.register_forward_hook(fwd))
    return hooks


def reference_outputs(model, family: str, x_nhwc: np.ndarray, dtype, conv_module: str, override=None, batch: int = 4):
    import torch

    m = model.to(dtype)
    stats = {"sign_input_exact_zero": 0, "n_binarized": 0, "binarizer_output_ne_claimed": 0}
    hooks = _sign_hooks(m, family, stats)
    proxy = _install_proxy(m, conv_module, None, override) if override is not None else None
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).numpy().transpose(0, 2, 3, 1))
    for h in hooks:
        h.remove()
    if proxy is not None:
        proxy[1]()
    model.to(torch.float32)
    return np.ascontiguousarray(np.concatenate(out)), stats


# ------------------------------------------------------------------ numpy executor workers
_G = {}
_EXACT: dict[str, dict] = {}


def _init_worker(paths: dict) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
        if k in _EXACT:
            gx = bfformat.read(p)
            for name, v in _EXACT[k].items():
                assert np.array_equal(v.astype(np.float32), gx.constants[name].data), f"{name}: bfmodel differs from adapter output"
                gx.constants[name].data = v
            _G[k + "#exact"] = gx


def _bin_sites(g):
    return [n.inputs[0] for n in g.nodes if n.op == "BConv2D"]


def _zeros_at(g, env) -> int:
    tot = 0
    for n in g.nodes:
        if n.op != "BConv2D":
            continue
        x = env[n.inputs[0]].astype(np.float64)
        t = env[n.inputs[4]].astype(np.float64) if len(n.inputs) > 4 and n.inputs[4] else 0.0
        tot += int(np.sum(x - t == 0))
    return tot


def _first_flip(g, env_a, env_b):
    """First BConv2D input (node order) whose binarization differs between two float64 runs of the same graph
    (float32 vs float64 constants): node, number of flipped values, |pre| at the flips (in the float64-constants
    run, which equals the source to ~1e-15) and the typical |pre| of that input."""
    for n in g.nodes:
        if n.op != "BConv2D":
            continue
        mode = n.attrs.get("input_binarize", "ge")
        thr = n.inputs[4] if len(n.inputs) > 4 and n.inputs[4] else None
        pa = env_a[n.inputs[0]] - (env_a[thr] if thr else 0.0)
        pb = env_b[n.inputs[0]] - (env_b[thr] if thr else 0.0)
        f = {"ge": lambda v: v >= 0, "gt": lambda v: v > 0, "sign_zero": np.sign}[mode]
        flips = f(pa) != f(pb)
        if flips.any():
            return {"node": n.name, "mode": mode, "n_flips": int(flips.sum()), "max_abs_pre": float(np.abs(pb[flips]).max()),
                    "typical_abs_pre": float(np.median(np.abs(pb)))}
    return None


def _run_one(job):
    key, x, want_f64 = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    out32 = env[g.outputs[0]].reshape(-1)
    z32 = _zeros_at(g, env)
    del env
    out64, z64, outx = None, 0, None
    if want_f64:
        env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
        out64 = env64[g.outputs[0]].reshape(-1)
        z64 = _zeros_at(g, env64)
        ff = None
        if key + "#exact" in _G:
            envx = executor.run(_G[key + "#exact"], {"input": x}, keep_intermediates=True, float_dtype=np.float64)
            outx = envx[g.outputs[0]].reshape(-1)
            if not np.array_equal(out64, outx):
                ff = _first_flip(g, env64, envx) or {"node": None, "max_abs_diff_vs_f64_constants": float(np.abs(out64 - outx).max())}
            del envx
        del env64
        return out32, out64, z32, z64, outx, ff
    return out32, out64, z32, z64, outx, None


def crop_quality(outs: np.ndarray, targets: np.ndarray, metric: str, border: int) -> dict:
    ps, ss = [], []
    for o, t in zip(outs, targets):
        p, s = image_quality(o, t, metric, border)
        ps.append(p)
        if s is not None:
            ss.append(s)
    return {"metric": metric, "crop_border": border, "psnr_mean": float(np.mean(ps)), "ssim_mean": float(np.mean(ss)) if ss else None,
            "psnr_per_sample": [float(v) for v in ps]}


def main(family: str) -> int:
    fam = FAMILIES[family]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(fam["archs"]))
    ap.add_argument("--repo", required=True, help=f"clone of {fam['repo_url']}")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--sha256", help="expected SHA-256 (overrides --registry)")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4, help="worker processes for the numpy executor")
    ap.add_argument("--torch-threads", type=int, default=4)
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

    torch.set_num_threads(args.torch_threads)
    model, cls, src_hashes = load_model(family, args.repo, args.model_id, args.weights)
    key = preprocess_key(family, args.model_id)
    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][key]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    tg = em["targets"]
    targets = np.fromfile(os.path.join(args.eval_dir, tg["file"]), np.uint8).reshape(tg["shape"])
    if args.max_samples:
        x, targets = x[: args.max_samples], targets[: args.max_samples]
    n, h, w, _ = x.shape
    metric, border = metric_of(family, em)
    rng_out = fam["output_range"]

    conv_module = fam["conv_module"]
    _, info0 = convert_graph(family, model, args.model_id, "ge", (h, w))
    override = {}
    for name, bits in info0["binary_weights"].items():
        s64 = info0["float64_constants"][f"{name}/post_mul"]
        override[name] = torch.from_numpy(bits.astype(np.float64) * s64[:, None, None, None])
    ref32, z32 = reference_outputs(model, family, x, torch.float32, conv_module)
    ref64n, z64n = reference_outputs(model, family, x, torch.float64, conv_module)
    ref64, z64 = reference_outputs(model, family, x, torch.float64, conv_module, override=override)
    # adapter bits == sign of the weight tensor the fp32 source passes to F.conv2d
    cap: dict = {}
    st, remove = _install_proxy(model, conv_module, cap, None)
    with torch.no_grad():
        model(torch.from_numpy(np.ascontiguousarray(x[:1].transpose(0, 3, 1, 2))))
    remove()
    mism = {k: int((torch.from_numpy(v.astype(np.float32)) != torch.where(cap[k] >= 0, 1.0, -1.0)).sum()) for k, v in info0["binary_weights"].items()}
    assert set(mism) == set(cap), f"binary conv sets differ: {sorted(set(mism) ^ set(cap))}"
    assert sum(mism.values()) == 0, mism
    rel = {k: float(((cap[k].double().abs() - torch.from_numpy(info0['float64_constants'][f'{k}/post_mul'])[:, None, None, None]).abs()
                     / torch.from_numpy(info0['float64_constants'][f'{k}/post_mul'])[:, None, None, None]).max()) for k in cap}
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    ref64n.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))

    report = {"model_id": args.model_id, "n_samples": int(n), "eval_set": em["name"], "eval_preprocessing": key,
              "output": {"layout": "NHWC", "shape": list(ref32.shape[1:]), "range": [0.0, rng_out], "kind": "image"},
              "reference": {"framework": f"torch {torch.__version__} (CPU, {args.torch_threads} threads)", "class": cls,
                            "repo_commit": repo_commit(args.repo), "source_sha256": src_hashes,
                            "exact_zeros_at_binarization_f32": z32, "exact_zeros_at_binarization_f64": z64,
                            "exact_zeros_at_binarization_f64_naive": z64n,
                            "binary_weights": {"adapter_bits_vs_source_fp32_sign_mismatches": int(sum(mism.values())),
                                               "n_binary_convs": len(mism),
                                               "source_fp32_weight_vs_exact_scale_max_rel": max(rel.values()),
                                               "primary_f64_reference": "official model in float64 with sign bits * float64 scale substituted at F.conv2d"},
                            "quality_on_crops_f32": crop_quality(ref32, targets, metric, border),
                            "quality_on_crops_f64": crop_quality(ref64, targets, metric, border)},
              "reference_f32_vs_f64": compare_images(ref64, ref32, rng_out),
              "naive_f64_vs_primary_f64": compare_images(ref64, ref64n, rng_out),
              "variants": {}}
    if getattr(model, "_bitino_load_notes", None):
        report["reference"]["load_notes"] = model._bitino_load_notes
    print(json.dumps({k: v for k, v in report["reference"].items() if not k.startswith("quality")}, indent=1))
    print("source quality on crops: f32", report["reference"]["quality_on_crops_f32"]["psnr_mean"],
          "f64", report["reference"]["quality_on_crops_f64"]["psnr_mean"])

    files = {}
    for binarize in fam["binarize_modes"]:
        for variant, fold in (("unfused", False), ("fused", True)):
            graph, info = convert_graph(family, model, args.model_id, binarize, (h, w))
            applied = passes.optimize(graph, fold=fold)
            stem = "model" if binarize == "ge" else "model_sign_zero"
            vkey = variant if binarize == "ge" else f"sign_zero_{variant}"
            graph.manifest = {
                "schema": "bitino.manifest.v1", "model_id": args.model_id,
                "variant": f"{variant}, input_binarize={binarize}",
                "source": {"framework": "pytorch", "class": cls, "repo": fam["repo_url"], "repo_commit": report["reference"]["repo_commit"],
                           "source_sha256": src_hashes, "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
                "conversion": {"tool": f"tools/convert_{family}.py", "adapter": fam["adapter"].__name__, "passes": applied,
                               "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "python": platform.python_version(), "torch": torch.__version__, "input_binarize": binarize,
                               "binarize_note": ("exact torch.sign semantics (sign(0)=0); reference kernels only" if binarize == "sign_zero"
                                                 else ("deployable: 0 -> +1 at binarization; differs from torch.sign only for exact zeros"
                                                       if family == "bbcu" else "exact source semantics (sign(sign(x)+0.1): 0 -> +1; LAB: +1 iff x1 > x0)"))},
                "input": {"name": "input", "shape": [1, h, w, int(x.shape[3])], "layout": "NHWC", "dtype": "f32",
                          "preprocessing": pre["definition"]},
                "outputs": ["image"],
                "output": {"layout": "NHWC", "shape": info["output_shape"], "range": [0.0, rng_out],
                           "note": "image-to-image model: the output is the restored / super-resolved image"},
                "op_versions": {"format": 1},
                "layer_precision": info["layer_precision"],
                "weight_diagnostics": info["weight_diagnostics"],
                "unused_in_forward": info["unused_in_forward"],
            }
            if not fold:
                _EXACT[vkey] = info["float64_constants"]
            path = os.path.join(args.out_dir, f"{stem}.bitino" if fold else f"{stem}_unfused.bitino")
            file_sha = bfformat.write(graph, path)
            files[vkey] = path
            report["variants"][vkey] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                        "input_binarize": binarize, "op_counts": graph.op_counts(), "passes": applied}
            print(f"[{vkey}] wrote {path}: {graph.op_counts()} passes={applied}")

    torch.set_num_threads(1)     # the references are done: no idle intra-op threads next to the --jobs workers
    jobs = [(k, x[i:i + 1], True) for k in files for i in range(n)]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
        results = pool.map(_run_one, jobs, chunksize=1)
    by_key = {k: results[j * n:(j + 1) * n] for j, k in enumerate(files)}
    np_out = {}
    for k, res in by_key.items():
        out32 = np.stack([r[0] for r in res]).astype(np.float32).reshape(ref32.shape)
        out64 = np.stack([r[1] for r in res]).reshape(ref32.shape)
        np_out[k] = out32
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        sem = compare_images(ref64, out64, rng_out)
        pmax = np.array(sem["per_sample_max_abs"])
        sem["n_samples_max_abs_le_1e-6_rel"] = int(np.sum(pmax <= 1e-6 * rng_out))
        sem["n_samples_max_abs_le_1e-4_rel"] = int(np.sum(pmax <= 1e-4 * rng_out))
        sem["deviating_samples_gt_1e-6_rel"] = [int(i) for i in np.where(pmax > 1e-6 * rng_out)[0]]
        if res[0][4] is not None:
            outx = np.stack([r[4] for r in res]).reshape(ref32.shape)
            sx = compare_images(ref64, outx, rng_out)
            px = np.array(sx["per_sample_max_abs"])
            sem["float64_constants"] = {"max_abs_error": sx["max_abs_error"], "n_samples_max_abs_le_1e-6_rel": int(np.sum(px <= 1e-6 * rng_out)),
                                        "n_samples_max_abs_le_1e-9_rel": int(np.sum(px <= 1e-9 * rng_out)),
                                        "per_sample_max_abs": sx["per_sample_max_abs"]}
            # first-flip analysis of every sample deviating by > 1e-6 x range: float32 vs float64 constants (both float64
            # arithmetic); the float64-constants IR equals the source, so this is the IR-vs-source first flip
            dev = sem["deviating_samples_gt_1e-6_rel"]
            ffs = {i: res[i][5] for i in dev}
            with_flip = {i: f for i, f in ffs.items() if f and f.get("node")}
            sem["first_flip_analysis"] = {
                "n_analyzed": len(dev), "n_with_flip": len(with_flip), "n_without_flip": len(dev) - len(with_flip),
                "max_abs_without_flip": float(max([pmax[i] for i in dev if i not in with_flip], default=0.0)),
                "max_abs_pre_at_first_flip": float(max([f["max_abs_pre"] for f in with_flip.values()], default=0.0)),
                "max_pre_over_typical": float(max([f["max_abs_pre"] / f["typical_abs_pre"] for f in with_flip.values()], default=0.0)),
                "first_flip_nodes": sorted({f["node"] for f in with_flip.values()}),
                "per_sample": {str(i): f for i, f in ffs.items()}}
        report["variants"][k].update({
            "numpy_vs_reference_f32": {kk: vv for kk, vv in compare_images(ref32, out32, rng_out).items() if kk != "per_sample_max_abs"},
            "numpy_vs_reference_f64": {kk: vv for kk, vv in compare_images(ref64, out32, rng_out).items() if kk != "per_sample_max_abs"},
            "semantic_check_ir_f64_vs_reference_f64": sem,
            "quality_on_crops_numpy_f32": crop_quality(out32, targets, metric, border),
            "exact_zeros_at_bconv_inputs_numpy_f32": int(sum(r[2] for r in res)),
            "exact_zeros_at_bconv_inputs_numpy_f64": int(sum(r[3] for r in res))})
        q = report["variants"][k]["quality_on_crops_numpy_f32"]["psnr_mean"]
        print(f"[{k}] SEMANTIC (f64 IR vs f64 ref): max_abs {sem['max_abs_error']:.3e} samples<=1e-6rel "
              f"{sem['n_samples_max_abs_le_1e-6_rel']}/{n} psnr_min {sem['psnr_vs_reference_min_db']:.1f} dB"
              + (f" | f64 constants: max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else "")
              + (" | first flips: {n_with_flip}/{n_analyzed} at |pre| <= {max_abs_pre_at_first_flip:.2e} (<= {max_pre_over_typical:.2e} x typical), "
                 "no-flip max {max_abs_without_flip:.2e}".format(**sem["first_flip_analysis"]) if "first_flip_analysis" in sem else "")
              + f" | numpy f32 vs ref f32 max_abs {report['variants'][k]['numpy_vs_reference_f32']['max_abs_error']:.3e}"
              + f" | crop PSNR numpy {q:.4f} (src f32 {report['reference']['quality_on_crops_f32']['psnr_mean']:.4f})"
              + f" | zeros numpy f32 {report['variants'][k]['exact_zeros_at_bconv_inputs_numpy_f32']}")
    if "sign_zero" in fam["binarize_modes"]:
        report["ge_vs_sign_zero_numpy_f32"] = {v: {kk: vv for kk, vv in compare_images(np_out[f"sign_zero_{v}"], np_out[v], rng_out).items()
                                                   if kk != "per_sample_max_abs"} for v in ("unfused", "fused")}
    report["fused_vs_unfused_numpy_f32"] = {kk: vv for kk, vv in compare_images(np_out["unfused"], np_out["fused"], rng_out).items()
                                            if kk != "per_sample_max_abs"}
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv.pop(1)))
