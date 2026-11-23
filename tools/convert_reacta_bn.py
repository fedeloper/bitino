#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Converter for the ReActNet-A-with-BatchNorm checkpoints of slice S11 (adapter bitino.adapters.torch_abbnn,
BN flavour of _convert_reactnet_a):

  bimatting-bimobilenet-backbone  htqin/BiMatting model/mobilenet.py BiMobileNet (ImageNet classifier backbone)
  burn-reacta-stage2              naver-ai/burn openselfsup/models/backbones/reactnet.py ReActnet(out_indices=[13],
                                  use_mlp=True) -- self-supervised backbone, NO classifier released (feature-level only)
  s2bnn-reacta-distill-only       szq0214/S2-BNN Distillation_only/step2/ReActNet.py ReActNet (encoder_q, fc = Identity:
                                  the main_lincls.py feature) -- self-supervised backbone, NO classifier released

The references are the official model classes (sources read only, executed from their files with python3 -I, no
bytecode written; checkpoints with torch.load(weights_only=True)), in float32 and float64 on the CPU.

  python3 -I tools/convert_reacta_bn.py --model-id bimatting-bimobilenet-backbone --repo .cache/repos/BiMatting \\
      --weights .cache/checkpoints/bimatting-bimobilenet-backbone/bimobilenet_best.pth.tar \\
      --eval-dir .cache/data/eval128 --big-eval-dir .cache/data/eval5000 --out-dir artifacts/bimatting-bimobilenet-backbone
  python3 -I tools/convert_reacta_bn.py --model-id burn-reacta-stage2 --repo .cache/repos/burn \\
      --weights .cache/checkpoints/burn-reacta-stage2/burn_keep_fc_backbone_cls_tch_cls_ep_200.pth ...
  python3 -I tools/convert_reacta_bn.py --model-id s2bnn-reacta-distill-only --repo .cache/repos/S2-BNN \\
      --weights .cache/checkpoints/s2bnn-reacta-distill-only/checkpoint_0199.pth.tar ...

Source-specific notes (details in the research notes (not published)):
  * BiMatting: no classification script was released; BiMobileNet deletes pool1/fc in __init__. The reference is
    BiMobileNetClassifier = official BiMobileNet.forward_single_frame (macro residuals) -> f4 -> AdaptiveAvgPool2d(1)
    -> Linear(1024, 1000) with the checkpoint's fc.* (the ReActNet-A head). The ImageNet normalisation that
    forward_single_frame applies (torchvision normalize) is hoisted into the eval preprocessing (torchvision_imagenet,
    same mean/std): the module-global ``normalize`` of the loaded copy is replaced by the identity (in this process;
    the file is not modified). The plain sequential forward gives 0% top-1 with this checkpoint (checked by the
    converter: reference.forward_check), the macro-residual forward the expected accuracy.
  * BURN: openselfsup's mmcv / registry imports are satisfied by stub modules (init / checkpoint helpers raise if
    called; ReActnet's constructor and forward never call them); binaryfunction.py is executed from its file.
    Output = the backbone output that linear evaluation consumes (configs/benchmarks/linear_classification/imagenet/
    react_last_full.py: ClsHead on the 1024-d fc.0 MLP + ReLU output).
  * S2-BNN: keys module.encoder_q.* without encoder_q.fc (main_lincls.py:166-169), strict load into ReActNet with fc
    replaced by nn.Identity (the 1024-d pooled feature that main_lincls.py's linear classifier consumes).

Outputs: model{,_unfused,_sign_zero,_sign_zero_unfused}.bitino, reference_logits.f32 / reference_logits_f64.f64
(official model fp32 / fp64; embeddings for BURN / S2-BNN), numpy_logits_<variant>.f32, conversion_report.json (Level 3,
float64-constants check, first-flip analysis of every sample deviating by more than 1e-6, source-semantics checks),
with --big-eval-dir reference_logits_eval<N>.f32 (official fp32).
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
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import torch_abbnn  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

PRE_KEY = "torchvision_imagenet"   # Resize 256 / CenterCrop 224 / ImageNet mean-std (all three sources)
MODELS = {  # model_id -> (kind, repo url, source file(s) relative to the repo)
    "bimatting-bimobilenet-backbone": ("classifier", "https://github.com/htqin/BiMatting", ("model/mobilenet.py",)),
    "burn-reacta-stage2": ("embedding", "https://github.com/naver-ai/burn",
                           ("openselfsup/models/backbones/reactnet.py", "openselfsup/models/backbones/binaryfunction.py")),
    "s2bnn-reacta-distill-only": ("embedding", "https://github.com/szq0214/S2-BNN", ("Distillation_only/step2/ReActNet.py",)),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit(repo: str) -> str:
    gd = os.path.join(repo, ".git")
    if not os.path.exists(gd):
        return "none (not a git checkout; see source_sha256)"
    head = open(os.path.join(gd, "HEAD")).read().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    if os.path.exists(os.path.join(gd, ref)):
        return open(os.path.join(gd, ref)).read().strip()
    for line in open(os.path.join(gd, "packed-refs")):
        if line.strip().endswith(ref):
            return line.split()[0]
    return "unknown"


def _load_source_file(name: str, path: str):
    """Executes one source file as module ``name`` without putting its directory on sys.path (no bytecode)."""
    import importlib.machinery
    import importlib.util

    sys.dont_write_bytecode = True
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def _strip(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_source(model_id: str, repo: str, weights: str | None):
    """Returns (model, info). weights=None: random initialisation (tests)."""
    import torch
    import torch.nn as nn

    repo = os.path.abspath(repo)
    files = MODELS[model_id][2]
    src = {f: sha256_file(os.path.join(repo, f)) for f in files}
    info: dict = {"source_sha256": src, "repairs": []}
    ck = None
    if weights is not None:
        ck = torch.load(weights, map_location="cpu", weights_only=True)
    if model_id == "bimatting-bimobilenet-backbone":
        mod = _load_source_file("bimatting_mobilenet", os.path.join(repo, files[0]))
        mod.normalize = lambda x, mean, std: x   # hoisted into the eval preprocessing (same mean/std)
        info["repairs"].append({"what": "module-global torchvision normalize -> identity in this process",
                                "why": "forward_single_frame normalises its input; the eval set is already normalised "
                                       "(torchvision_imagenet, mean (0.485,0.456,0.406), std (0.229,0.224,0.225))"})

        class BiMobileNetClassifier(nn.Module):
            """Official BiMobileNet.forward_single_frame -> f4 (= output of the last block) -> pool -> fc (the head the
            checkpoint carries; BiMobileNet.__init__ deletes its pool1 / fc)."""

            def __init__(self):
                super().__init__()
                self.backbone = mod.BiMobileNet(pretrained=False)
                self.pool1 = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(1024, 1000)

            def forward(self, x):
                f4 = self.backbone.forward_single_frame(x)[3]
                return self.fc(torch.flatten(self.pool1(f4), 1))

            def forward_plain(self, x):   # sequential blocks, no macro residuals (diagnostic only)
                for b in self.backbone.feature:
                    x = b(x)
                return self.fc(torch.flatten(self.pool1(x), 1))

        model = BiMobileNetClassifier()
        info["class"] = ("model/mobilenet.py BiMobileNet(pretrained=False).forward_single_frame -> f4 -> AdaptiveAvgPool2d(1) "
                         "-> Linear(1024, 1000) [checkpoint fc.*; head and wiring not in the released code]")
        if ck is not None:
            assert set(ck) == {"epoch", "state_dict", "best_top1_acc", "optimizer"}, sorted(ck)
            sd = _strip(ck["state_dict"], "module.")
            assert len(sd) == len(ck["state_dict"]), "expected a DataParallel 'module.' prefix on every key"
            model.backbone.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=True)
            model.fc.load_state_dict(_strip(sd, "fc."), strict=True)
            info["checkpoint_meta"] = {"epoch": int(ck["epoch"]), "best_top1_acc": float(ck["best_top1_acc"]),
                                       "optimizer_state_ignored": True}
    elif model_id == "burn-reacta-stage2":
        def _unused(*a, **k):
            raise RuntimeError("stubbed mmcv / openselfsup helper called")
        stubs = {}
        for name in ("mmcv", "mmcv.cnn", "mmcv.runner", "openselfsup", "openselfsup.utils", "openselfsup.models",
                     "openselfsup.models.registry", "openselfsup.models.utils", "openselfsup.models.backbones"):
            m = types.ModuleType(name)
            m.__path__ = []
            stubs[name] = m
        stubs["mmcv.cnn"].__dict__.update(constant_init=_unused, kaiming_init=_unused)
        stubs["mmcv.runner"].load_checkpoint = _unused
        stubs["openselfsup.utils"].get_root_logger = _unused
        stubs["openselfsup.models.registry"].BACKBONES = types.SimpleNamespace(register_module=lambda cls: cls)
        stubs["openselfsup.models.utils"].__dict__.update(build_conv_layer=_unused, build_norm_layer=_unused)
        names = list(stubs) + ["openselfsup.models.backbones.binaryfunction", "openselfsup.models.backbones.reactnet"]
        saved = {k: sys.modules.get(k) for k in names}
        sys.modules.update(stubs)
        try:
            _load_source_file("openselfsup.models.backbones.binaryfunction", os.path.join(repo, files[1]))
            mod = _load_source_file("openselfsup.models.backbones.reactnet", os.path.join(repo, files[0]))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        # configs/selfsup/burn/react_burn_stage2_bs2048_accumulate2_ep200_fp16.py backbone (and the linear-eval config)
        model = mod.ReActnet(out_indices=[13], use_mlp=True)
        info["class"] = "openselfsup/models/backbones/reactnet.py ReActnet(out_indices=[13], use_mlp=True) [mmcv registry stubbed]"
        info["output"] = "backbone output = ReLU(fc.0(AdaptiveAvgPool2d(feature.13))), 1024-d (the linear-eval ClsHead input)"
        if ck is not None:
            assert set(ck) == {"state_dict", "author"}, sorted(ck)
            model.load_state_dict(ck["state_dict"], strict=True)
            info["checkpoint_meta"] = {"author": ck["author"]}
    elif model_id == "s2bnn-reacta-distill-only":
        mod = _load_source_file("s2bnn_reactnet_step2", os.path.join(repo, files[0]))
        model = mod.ReActNet()
        model.fc = nn.Identity()   # main_lincls.py: features before the (re-initialised) linear classifier
        info["class"] = "Distillation_only/step2/ReActNet.py ReActNet() with fc = nn.Identity (encoder_q backbone)"
        info["output"] = "AdaptiveAvgPool2d(feature.13) flattened, 1024-d (the main_lincls.py linear-classifier input)"
        if ck is not None:
            assert set(ck) == {"epoch", "arch", "state_dict", "optimizer"}, sorted(ck)
            sd = {k: v for k, v in _strip(ck["state_dict"], "module.encoder_q.").items() if not k.startswith("fc.")}
            model.load_state_dict(sd, strict=True)
            info["checkpoint_meta"] = {"epoch": int(ck["epoch"]), "arch_field": ck["arch"],
                                       "dropped": "encoder_k.*, queue, queue_ptr, encoder_q.fc.* (MoCo projection), optimizer"}
    else:
        raise KeyError(model_id)
    model.eval()
    return model, info


def module_prefix(model) -> str:
    return "backbone." if type(model).__name__ == "BiMobileNetClassifier" else ""


def forward(model, xb):
    y = model(xb)
    return y[0] if isinstance(y, tuple) else y   # BURN ReActnet returns a 1-tuple


def run_reference(model, x_nhwc: np.ndarray, dtype, points: list[str], batch: int = 16, plain: bool = False):
    """Outputs, exact zeros at the binarization inputs (move11 / move21 outputs) and STE checks: every
    BinaryActivation output must equal torch.sign(input) exactly."""
    import torch

    m = model.to(dtype)
    stats = {"sign_input_exact_zero": 0, "n_binarized": 0, "binary_activation_ne_torch_sign": 0}
    mods = dict(m.named_modules())
    hooks = []
    pre = module_prefix(model)

    def count(_m, _i, out):
        stats["sign_input_exact_zero"] += int((out == 0).sum())
        stats["n_binarized"] += int(out.numel())

    def ste(_m, inp, out):
        stats["binary_activation_ne_torch_sign"] += int((out != torch.sign(inp[0])).sum())
    for p in points:
        hooks.append(mods[pre + p].register_forward_hook(count))
    for n, mm in m.named_modules():
        if type(mm).__name__ == "BinaryActivation" and not type(model).__name__ == "ReActnet":   # BURN: unused module
            hooks.append(mm.register_forward_hook(ste))
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch].transpose(0, 3, 1, 2))).to(dtype)
            out.append((m.forward_plain(xb) if plain else forward(m, xb)).numpy())
    for h in hooks:
        h.remove()
    model.to(torch.float32)
    return np.concatenate(out), stats


def capture_binarization_inputs(model, x_nhwc: np.ndarray, points: list[str]) -> dict:
    """float64 forward of one sample; {point: value handed to sign() (move11 / move21 output), NHWC}."""
    import torch

    m = model.to(torch.float64)
    cap = {}
    mods = dict(m.named_modules())
    pre = module_prefix(model)
    hooks = [mods[pre + p].register_forward_hook(lambda _m, _i, o, p=p: cap.__setitem__(p, o.detach().numpy().transpose(0, 2, 3, 1)))
             for p in points]
    with torch.no_grad():
        forward(m, torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))).to(torch.float64))
    for h in hooks:
        h.remove()
    model.to(torch.float32)
    return cap


def ste_weight_noise(model) -> dict:
    """HardBinaryConv / BiConv2d use scale*sign(W) - clamp(W) + clamp(W) in float32: count the taps where the tensor the
    float32 source hands to F.conv2d differs from scale*sign(W) (the float32 noise floor; the IR uses scale*sign(W))."""
    import torch

    n_diff, n_tot, max_rel = 0, 0, 0.0
    kinds = set()
    for _, m in model.named_modules():
        k = type(m).__name__
        if k not in ("HardBinaryConv", "BiConv2d"):
            continue
        kinds.add(k)
        w = m.weight.detach().to(torch.float32)
        if k == "BiConv2d":
            w = w - torch_abbnn._nested_mean(w)
        sf = torch_abbnn._nested_mean(w.abs())
        a = sf * torch.sign(w)
        c = torch.clamp(w, -1.0, 1.0)
        eff = a - c + c
        d = (eff != a)
        n_diff += int(d.sum())
        n_tot += int(w.numel())
        if d.any():
            max_rel = max(max_rel, float(((eff - a).abs() / sf.expand_as(a))[d].max()))
    return {"kinds": sorted(kinds), "n_weights": n_tot, "n_taps_differing_from_scale_sign_f32": n_diff,
            "max_rel_deviation_vs_scale": max_rel} if kinds else {"kinds": [], "note": "BinaryConv2d: sign(W)*scale exact"}


# ------------------------------------------------------------- numpy executor workers
_G: dict = {}
_EXACT: dict = {}
_POINTS: list = []


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
    zeros = [int(np.sum(env[f"{p}:out"] == 0)) for p in _POINTS]
    out32 = env[g.outputs[0]].reshape(-1)
    del env
    env64 = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=np.float64)
    out64 = env64[g.outputs[0]].reshape(-1)
    pre64 = {p: env64[f"{p}:out"] for p in _POINTS} if key == "sign_zero_unfused" else None
    del env64
    outx = executor.run(_G[key + "#exact"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1) if key + "#exact" in _G else None
    return out32, out64, zeros, outx, pre64


def _run_f32(job):
    key, x = job
    g = _G[key]
    env = executor.run(g, {"input": x}, keep_intermediates=True)
    return env[g.outputs[0]].reshape(-1), [int(np.sum(env[f"{p}:out"] == 0)) for p in _POINTS]


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


def cosines(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)


def embedding_stats(ref, cand) -> dict:
    c = cosines(cand, ref)
    rel = np.linalg.norm(np.asarray(cand, np.float64) - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-300)
    return {"cosine_min": float(c.min()), "cosine_mean": float(c.mean()), "cosine_median": float(np.median(c)),
            "n_cosine_lt_0.99999": int(np.sum(c < 0.99999)), "n_cosine_lt_0.9999": int(np.sum(c < 0.9999)),
            "n_cosine_lt_0.999": int(np.sum(c < 0.999)), "rel_l2_max": float(rel.max()), "rel_l2_mean": float(rel.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--big-eval-dir", help="large eval set (official fp32 outputs -> reference_logits_eval<N>.f32)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--big-batch", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=0, help="debug: first N eval samples only")
    args = ap.parse_args()

    kind, repo_url, _ = MODELS[args.model_id]
    reg = json.load(open(args.registry))
    entry = next(m for m in reg["models"] if m.get("model_id") == args.model_id)
    actual = sha256_file(args.weights)
    if actual != entry["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch: expected {entry['checkpoint']['sha256']}, got {actual}", file=sys.stderr)
        return 3
    print(f"checkpoint SHA-256 OK: {actual}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    import torchvision
    torch.set_num_threads(args.torch_threads)
    model, src_info = load_source(args.model_id, args.repo, args.weights)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PRE_KEY]
    input_hw = tuple(pre["shape"][1:3])
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    def convert(binarize: str, literal: bool = False):
        return torch_abbnn.convert(model, args.model_id, binarize=binarize, input_hw=input_hw, literal_thresholds=literal)

    _, info0 = convert("sign_zero")
    points = info0["binarization_points"]
    global _POINTS
    _POINTS = points
    ref32, zeros32 = run_reference(model, x, torch.float32, points)
    ref64, zeros64 = run_reference(model, x, torch.float64, points)
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    is_cls = kind == "classifier"
    forward_check = None
    if args.model_id == "bimatting-bimobilenet-backbone":   # which forward does the checkpoint expect?
        plain32, _ = run_reference(model, x, torch.float32, points, plain=True)
        forward_check = {"top1_forward_single_frame_macro_residuals": float(np.mean(ref32.argmax(1) == labels)),
                         "top1_plain_sequential_blocks": float(np.mean(plain32.argmax(1) == labels))}

    rinfo = {"framework": f"torch {torch.__version__} (CPU), torchvision {torchvision.__version__}", "class": src_info["class"],
             "repo": repo_url, "repo_commit": repo_commit(args.repo), "source_sha256": src_info["source_sha256"],
             "repairs": src_info["repairs"], "checkpoint_meta": src_info.get("checkpoint_meta"), "output": src_info.get("output", "logits"),
             "reference_f32_is": "official model fp32 (this process, CPU)", "reference_f64_is": "official model in float64 (model.double())",
             "exact_zeros_at_binarization_f32": zeros32, "exact_zeros_at_binarization_f64": zeros64,
             "ste_weight_noise_f32": ste_weight_noise(model), "forward_check": forward_check}
    if is_cls:
        rinfo.update({"top1_accuracy": float(np.mean(ref32.argmax(1) == labels)),
                      "top5_accuracy": float(np.mean([labels[i] in np.argsort(-ref32[i])[:5] for i in range(len(x))])),
                      "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)),
                      "top5_accuracy_f64": float(np.mean([labels[i] in np.argsort(-ref64[i])[:5] for i in range(len(x))]))})
    report = {"model_id": args.model_id, "kind": kind, "n_samples": int(len(x)), "eval_preprocessing": PRE_KEY, "eval_set": em["name"],
              "reference": rinfo,
              "adapter": {"binarization_points": len(points), "unused_in_forward": info0["unused_in_forward"],
                          "refolded_thresholds": info0["refolded_thresholds"],
                          "weight_diagnostics_totals": {k: int(sum(d[k] for d in info0["weight_diagnostics"]))
                                                        for k in ("zeros_f32", "zeros_f64", "sign_mismatch_f32_vs_f64")}},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels if is_cls else None),
              "variants": {}}
    if not is_cls:
        report["reference_f32_vs_f64_embedding"] = embedding_stats(ref64, ref32)
    print(json.dumps({k: v for k, v in rinfo.items() if k not in ("source_sha256",)}, indent=1, default=str), flush=True)
    print("adapter:", json.dumps({k: v for k, v in report["adapter"].items() if k != "unused_in_forward"}),
          f"unused_in_forward: {len(info0['unused_in_forward'])}", flush=True)

    files = {}
    for binarize in ("ge", "sign_zero"):
        for variant, fold in (("unfused", False), ("fused", True)):
            graph, info = convert(binarize)
            applied = passes.optimize(graph, fold=fold)
            stem = "model" if binarize == "ge" else "model_sign_zero"
            key = variant if binarize == "ge" else f"sign_zero_{variant}"
            graph.manifest = {
                "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": f"{variant}, input_binarize={binarize}",
                "source": {"framework": "pytorch", "class": src_info["class"], "repo": repo_url, "repo_commit": rinfo["repo_commit"],
                           "source_sha256": src_info["source_sha256"], "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
                "conversion": {"tool": "tools/convert_reacta_bn.py", "adapter": "bitino.adapters.torch_abbnn (ReActNet-A, BN flavour)",
                               "passes": applied, "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "python": platform.python_version(), "torch": torch.__version__, "input_binarize": binarize,
                               "binarize_note": ("exact torch.sign semantics (sign(0)=0); reference kernels only" if binarize == "sign_zero"
                                                 else "deployable: 0 -> +1 at binarization; differs from torch.sign only for exact zeros")},
                "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32", "preprocessing": pre["definition"]},
                "outputs": ["logits"] if is_cls else ["embedding"], "op_versions": {"format": 1},
                "layer_precision": info["layer_precision"], "weight_diagnostics": info["weight_diagnostics"],
                "unused_in_forward": info["unused_in_forward"],
            }
            if not is_cls:
                graph.manifest["output_note"] = src_info["output"] + "; no ImageNet classifier was released with this checkpoint"
            if not fold:
                _EXACT[key] = info["float64_constants"]
            path = os.path.join(args.out_dir, f"{stem}.bitino" if fold else f"{stem}_unfused.bitino")
            file_sha = bfformat.write(graph, path)
            files[key] = path
            report["variants"][key] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": binarize, "op_counts": graph.op_counts(), "passes": applied}
            print(f"[{key}] wrote {path}: {graph.op_counts()} passes={applied}", flush=True)

    jobs = [(k, x[i:i + 1]) for k in files for i in range(len(x))]
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
        results = pool.map(_run_one, jobs, chunksize=2)
    by_key = {k: results[j * len(x):(j + 1) * len(x)] for j, k in enumerate(files)}
    np_logits = {}
    lab = labels if is_cls else None
    for k, res in by_key.items():
        out32 = np.stack([r[0] for r in res]).astype(np.float32)
        out64 = np.stack([r[1] for r in res])
        zeros = np.array([r[2] for r in res])
        np_logits[k] = out32
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        sem = compare_outputs(ref64, out64, lab)
        per_sample = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per_sample <= 1e-6))
        sem["n_samples_max_abs_le_1e-4"] = int(np.sum(per_sample <= 1e-4))
        sem["per_sample_max_abs"] = [float(v) for v in per_sample]
        if not is_cls:
            sem["embedding"] = embedding_stats(ref64, out64)
        if res[0][3] is not None:
            outx = np.stack([r[3] for r in res])
            px = np.abs(outx - ref64).max(axis=1)
            sem["float64_constants"] = {"max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-6": int(np.sum(px <= 1e-6)),
                                        "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                                        "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1))),
                                        "cosine_min": float(cosines(outx, ref64).min())}
        if k == "sign_zero_unfused":   # first-flip analysis of every deviating sample (float64 IR, float32 constants)
            ff = []
            for i in np.where(per_sample > 1e-6)[0]:
                src_pre = capture_binarization_inputs(model, x[i:i + 1], points)
                f = first_flip(src_pre, res[i][4], points)
                ff.append({"sample": int(i), "max_abs_diff": float(per_sample[i]),
                           "cosine_vs_f64": float(cosines(out64[i:i + 1], ref64[i:i + 1])[0]),
                           "top1_changed": bool(out64[i].argmax() != ref64[i].argmax()) if is_cls else None, "first_flip": f})
            no_flip = [d for d in ff if d["first_flip"] is None]
            with_flip = [d for d in ff if d["first_flip"] is not None]
            sem["first_flip_analysis"] = {
                "n_analyzed": len(ff), "n_without_flip": len(no_flip),
                "max_abs_diff_without_flip": max([d["max_abs_diff"] for d in no_flip], default=0.0),
                "n_with_flip": len(with_flip),
                "max_abs_pre_at_first_flip": max([d["first_flip"]["max_abs_pre_at_flip"] for d in with_flip], default=None),
                "max_relative_to_typical": max([d["first_flip"]["relative_to_typical"] for d in with_flip], default=None),
                "first_flip_layers": sorted({d["first_flip"]["layer"] for d in with_flip}),
                "n_top1_changes": int(sum(bool(d["top1_changed"]) for d in ff)) if is_cls else None,
                "min_cosine_vs_f64": min([d["cosine_vs_f64"] for d in ff], default=1.0), "samples": ff}
        vr = {"numpy_vs_reference_f32": compare_outputs(ref32, out32, lab), "numpy_vs_reference_f64": compare_outputs(ref64, out32, lab),
              "semantic_check_ir_f64_vs_reference_f64": sem,
              "exact_zeros_at_binarization_numpy_f32": {"total": int(zeros.sum()), "samples_with_zeros": int(np.sum(zeros.sum(1) > 0))}}
        if not is_cls:
            vr["numpy_vs_reference_f32_embedding"] = embedding_stats(ref32, out32)
            vr["numpy_vs_reference_f64_embedding"] = embedding_stats(ref64, out32)
        report["variants"][k].update(vr)
        print(f"[{k}] vs f32 ref: cos_min {vr['numpy_vs_reference_f32']['cosine_min']:.6f} top1 agree {vr['numpy_vs_reference_f32']['top1_agreement']:.4f}"
              f" | SEMANTIC f64: max_abs {sem['max_abs_error']:.3e} <=1e-6: {sem['n_samples_max_abs_le_1e-6']}/{len(x)} "
              f"<=1e-4: {sem['n_samples_max_abs_le_1e-4']} cos_min {sem['cosine_min']:.9f} | zeros(np f32) {int(zeros.sum())}"
              + (f" | f64 consts max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else "")
              + (f" | first-flip: {sem['first_flip_analysis']['n_analyzed']} analyzed, {sem['first_flip_analysis']['n_with_flip']} with flip "
                 f"(max rel {sem['first_flip_analysis']['max_relative_to_typical']})" if "first_flip_analysis" in sem else ""), flush=True)
    report["ge_vs_sign_zero_numpy_f32"] = {v: compare_outputs(np_logits[f"sign_zero_{v}"], np_logits[v], lab) for v in ("unfused", "fused")}
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(np_logits["unfused"], np_logits["fused"], lab)

    # Comparison only (not written to the artifacts): thresholds as the source's literal fl(fl(q + b_prev) + b_next).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        lit = {}
        for binarize in ("ge", "sign_zero"):
            graph, _ = convert(binarize, literal=True)
            passes.optimize(graph, fold=True)
            lit[f"literal_{binarize}_fused"] = os.path.join(td, f"literal_{binarize}.bitino")
            bfformat.write(graph, lit[f"literal_{binarize}_fused"])
        _EXACT.clear()
        with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker, initargs=(lit,)) as pool:
            res_l = pool.map(_run_f32, [(k, x[i:i + 1]) for k in lit for i in range(len(x))], chunksize=2)
    cmp_l = {}
    for j, k in enumerate(lit):
        rr = res_l[j * len(x):(j + 1) * len(x)]
        o = np.stack([r[0] for r in rr]).astype(np.float32)
        cmp_l[k] = {"numpy_vs_reference_f32": compare_outputs(ref32, o, lab), "numpy_vs_reference_f64": compare_outputs(ref64, o, lab),
                    "exact_zeros_at_binarization_numpy_f32": int(sum(sum(r[1]) for r in rr))}
        if not is_cls:
            cmp_l[k]["numpy_vs_reference_f64_embedding"] = embedding_stats(ref64, o)
        print(f"[{k}] (comparison) vs f64 ref: top1 agree {cmp_l[k]['numpy_vs_reference_f64']['top1_agreement']:.4f} cos_min "
              f"{cmp_l[k]['numpy_vs_reference_f64']['cosine_min']:.6f} zeros {cmp_l[k]['exact_zeros_at_binarization_numpy_f32']}", flush=True)
    report["literal_thresholds_comparison"] = cmp_l
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)

    if args.big_eval_dir:
        bm = json.load(open(os.path.join(args.big_eval_dir, "manifest.json")))
        bp = bm["preprocessing"][PRE_KEY]
        n = int(bp["shape"][0])
        xb = np.memmap(os.path.join(args.big_eval_dir, bp["file"]), np.float32, "r", shape=tuple(bp["shape"]))
        yb = np.array([s["label"] for s in bm["samples"]][:n])
        lb, zb = run_reference(model, xb, torch.float32, points, batch=args.big_batch)
        lb.astype(np.float32).tofile(os.path.join(args.out_dir, f"reference_logits_eval{n}.f32"))
        big = {"eval_set": bm["name"], "n": n, "exact_zeros_at_binarization": zb, "file": f"reference_logits_eval{n}.f32"}
        if is_cls:
            big.update({"source_fp32_top1": float(np.mean(lb.argmax(1) == yb)),
                        "source_fp32_top5": float(np.mean([yb[i] in np.argsort(-lb[i])[:5] for i in range(n)]))})
        report["large_eval_source"] = big
        print("large eval:", json.dumps(big), flush=True)
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
