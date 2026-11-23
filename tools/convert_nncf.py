#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert an NNCF-binarized ResNet-18 (OpenVINO NNCF zoo, XNOR / DoReFa) to .bitino and validate it.

Runs in the reference venv .cache/venv-nncf (Python 3.10, nncf 2.9.0, torch 2.1.2+cpu = NNCF 2.9's
recommended torch, torchvision 0.16.2; envs/nncf-freeze.txt). The trusted reference
is official NNCF: torchvision ``resnet18`` + the published binarization config
(examples/torch/classification/configs/binarization/resnet18_imagenet_binarization_<mode>.json) through
``create_compressed_model(model, config, compression_state)`` and ``load_state(model, state_dict,
is_resume=True)``, exactly as examples/torch/classification/main.py evaluates a resumed checkpoint.
The checkpoint itself is a training checkpoint with pickled non-tensor objects; it is first reduced to a
tensors-only state with tools/nncf_extract_state.py (restricted unpickler + inert stubs, base env), and that
file is loaded here with torch.load(weights_only=True). The SHA-256 of the original .pth is checked
against registry/models.json and against the hash recorded inside the derived file.

  python3 -I tools/nncf_extract_state.py .cache/checkpoints/nncf-resnet18-binarization-xnor/resnet18_imagenet_binarization_xnor.pth \
      .cache/derived-nncf/resnet18_xnor.state.pt
  .cache/venv-nncf/bin/python -I tools/convert_nncf.py --model-id nncf-resnet18-binarization-xnor \
      --weights .cache/checkpoints/nncf-resnet18-binarization-xnor/resnet18_imagenet_binarization_xnor.pth \
      --state .cache/derived-nncf/resnet18_xnor.state.pt \
      --config .cache/repos/nncf-v2.5.0/examples/torch/classification/configs/binarization/resnet18_imagenet_binarization_xnor.json \
      --eval-dir .cache/data/eval128 --out-dir artifacts/nncf-resnet18-binarization-xnor --jobs 4

DoReFa weight scale. NNCF's CPU forward (DOREFABinarizeFn) scales the binary weights by mean|W| of the whole
tensor, but NNCF's CUDA kernel (wb_cuda_forward, per_channel=false) uses mean|W| of output filter 0 only: it
sets elements_per_scale = numel / size(0) regardless of per_channel. The zoo checkpoint was trained and
evaluated with the CUDA kernel; on the CPU path it scores 8.6% top-1 on eval128, with the CUDA-kernel
scale 62.5%. Default --dorefa-weight-scale cuda: the weight binarizer call is replaced by a transliteration
of the CUDA kernel ((w > 0) ? s : -s with s = mean|W[0]|); the unmodified CPU-path logits are kept in
reference_logits_nncf_cpu_path.f32. XNOR (per_channel=true) is identical on both paths.

float64 references. The activation binarizer compares x with t = sigmoid(theta) * exp(log_s); the fp32 source
compares with t32 = fl32(sigmoid32 * s32). ``reference_logits_f64.f64`` (primary) runs the same NNCF model in
float64 with the binarizer constants the fp32 model uses (s32, sigmoid32 and the fp32 binarized weight
wb32 = sign * mean|W|, substituted at the binarizer call sites; NNCF's own functions do the rest, so its
threshold is the exact product sigmoid32 * s32). ``reference_logits_f64_naive.f64`` is model.double() as is
(binarizer constants re-derived in float64).

Produces in --out-dir: model.bitino (fused BN), model_unfused.bitino, reference_logits.f32,
reference_logits_f64.f64, reference_logits_f64_naive.f64, numpy_logits_{fused,unfused}.f32,
conversion_report.json (Level-3 float64 semantic check incl. float64 constants, threshold ties, first flips).
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
from bitino.adapters import torch_nncf  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402

MODELS = {   # model_id -> mode; the v2.7.0 records are tensor-identical re-serialisations (see the notes)
    "nncf-resnet18-binarization-xnor": "xnor",
    "nncf-resnet18-binarization-dorefa": "dorefa",
    "nncf-resnet18-binarization-xnor-v2.7.0": "xnor",
    "nncf-resnet18-binarization-dorefa-v2.7.0": "dorefa",
}
PREPROCESS = "torchvision_imagenet"   # examples/torch/classification/main.py:469-476 (Resize 256, CenterCrop 224, ImageNet mean/std)
REPO = "https://github.com/openvinotoolkit/nncf (PyPI nncf 2.9.0; checkpoints published with v2.5.0)"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(state_path: str, config_path: str):
    import torch
    import torchvision
    from nncf import NNCFConfig
    from nncf.torch import create_compressed_model
    from nncf.torch.checkpoint_loading import load_state

    st = torch.load(state_path, map_location="cpu", weights_only=True)
    meta = json.loads(st["meta_json"])
    cfg = NNCFConfig.from_json(config_path)
    model = torchvision.models.resnet18(weights=None, num_classes=1000)
    ctrl, model = create_compressed_model(model, cfg, compression_state=meta["compression_state"])
    n_loaded = load_state(model, dict(st["state_dict"]), is_resume=True)
    model.eval()
    for name in torch_nncf.BINARIZED:
        conv = model.get_submodule(name)
        wb, ab = torch_nncf.binarizer_modules(conv)
        assert wb is not None, f"{name}: not binarized"
    for name in ("conv1", "layer2.0.downsample.0", "layer3.0.downsample.0", "layer4.0.downsample.0", "fc"):
        assert not list(model.get_submodule(name).pre_ops.values()), f"{name}: unexpectedly binarized"
    return model, meta, n_loaded, len(st["state_dict"])


class Instrument:
    """Wraps the 16 weight / activation binarizers of the NNCF model (instance attributes only; NNCF code
    is called unchanged). capture: record the fp32 constants; override: substitute given constants;
    counts exact threshold ties; optionally records every binarizer input (x, t) of a forward."""

    def __init__(self, model, dorefa_cuda: bool = False):
        self.model = model
        self.dorefa_cuda = dorefa_cuda
        self.capture: dict = {}
        self.override: dict | None = None
        self.record: dict | None = None
        self.ties = 0
        self.n = 0
        self.min_rel = np.inf
        for name in torch_nncf.BINARIZED:
            wbm, abm = torch_nncf.binarizer_modules(model.get_submodule(name))
            self._wrap(name, wbm, abm)

    def _wrap(self, name, wbm, abm):
        import torch
        orig_w, orig_a = wbm.binarize, abm.bin

        def wfn(x):
            if self.dorefa_cuda:
                # NNCF's CUDA kernel (extensions/src/binarization/cuda/functions_cuda_impl.cu, wb_cuda_forward with
                # per_channel=false): scale_count = 1 but elements_per_scale = numel / size(0), so the single scale
                # is mean|W| over output filter 0 only; output = (w > 0) ? scale : -scale.
                out = ((x > 0).type(x.dtype) * 2 - 1) * x[0].abs().mean()
            else:
                out = orig_w(x)
            if x.dtype == torch.float32 and self.override is None:
                self.capture.setdefault(name, {})["wb"] = out.detach().clone()
            if self.override is not None:
                return self.override[name]["wb"].to(x.dtype)
            return out

        def afn(x, scale, threshold):
            if x.dtype == torch.float32 and self.override is None:
                c = self.capture.setdefault(name, {})
                c["scale"], c["sigmoid"] = scale.detach().clone(), threshold.detach().clone()
            if self.override is not None:
                scale = self.override[name]["scale"].to(x.dtype)
                threshold = self.override[name]["sigmoid"].to(x.dtype)
            shape = [1] * x.dim()
            shape[1] = x.shape[1]
            t = (threshold * scale).view(shape)        # same expression as ActivationBinarizationScaleThresholdFn
            self.ties += int((x == t).sum())
            self.n += int(x.numel())
            d = (x - t).abs() / t
            self.min_rel = min(self.min_rel, float(d.min()))
            if self.record is not None:
                self.record[name] = (x.detach().to(torch.float64).numpy().copy(), t.detach().to(torch.float64).numpy().reshape(-1).copy())
            return orig_a(x, scale, threshold)

        wbm.binarize = wfn
        abm.bin = afn

    def stats(self):
        r = {"activations_exactly_at_threshold": self.ties, "n_binarized": self.n, "min_rel_distance_to_threshold": self.min_rel}
        self.ties, self.n, self.min_rel = 0, 0, np.inf
        return r


def reference_outputs(model, ins: Instrument, x_nhwc: np.ndarray, dtype, override=None) -> tuple[np.ndarray, dict]:
    import torch
    m = model.to(dtype)
    ins.override = override
    out = []
    with torch.no_grad():
        for i in range(0, len(x_nhwc), 16):
            xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + 16].transpose(0, 3, 1, 2))).to(dtype)
            out.append(m(xb).numpy())
    ins.override = None
    model.to(torch.float32)
    return np.concatenate(out), ins.stats()


# ------------------------------------------------------------- numpy executor workers
_G: dict = {}
_EXACT: dict[str, dict] = {}
_SRC: dict = {}


def _init_worker(paths: dict) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    for k, p in paths.items():
        _G[k] = bfformat.read(p)
        if k in _EXACT:
            gx = bfformat.read(p)
            for name, v in _EXACT[k].items():
                assert np.array_equal(v.astype(np.float32), gx.constants[name].data), f"{name}: bfmodel differs from adapter output"
                gx.constants[name].data = v
            _G[k + "#exact"] = gx


def _run_one(job):
    key, x = job
    g = _G[key]
    out32 = executor.run(g, {"input": x})[g.outputs[0]].reshape(-1)
    out64 = executor.run(g, {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
    outx = None
    if key + "#exact" in _G:
        outx = executor.run(_G[key + "#exact"], {"input": x}, float_dtype=np.float64)[g.outputs[0]].reshape(-1)
    return out32, out64, outx


def _first_flip(src_rec: dict, env: dict, g, consts64: bool) -> dict | None:
    """First binarized conv (forward order) whose decisions differ between source (x > t) and IR (x > thr)."""
    for name in torch_nncf.BINARIZED:
        node = next(n for n in g.nodes if n.name == name)
        x_ir = np.asarray(env[node.inputs[0]], np.float64)[0]                  # HWC
        thr_ir = np.asarray(env[node.inputs[4]], np.float64)
        xs, ts = src_rec[name]
        xs = xs[0].transpose(1, 2, 0)                                          # CHW -> HWC
        d = (xs > ts) != (x_ir > thr_ir)
        if d.any():
            rel = np.abs(xs - ts)
            return {"layer": name, "n_flips": int(d.sum()), "max_abs_pre_source": float(rel[d].max()),
                    "max_abs_pre_ir": float(np.abs(x_ir - thr_ir)[d].max()),
                    "typical_abs_pre_source": float(np.median(rel)), "note": "pre = x - threshold"}
    return None


def _flip_job(job):
    """Per deviating sample: source binarizer inputs (fp64 primary and fp32) vs IR intermediates (fused graph)."""
    import torch
    i, x = job
    model, ins, ovr = _SRC["model"], _SRC["ins"], _SRC["override"]
    g = _G["fused"]
    res = {"sample": int(i)}
    for tag, dtype, fdt in (("ir_f64_vs_source_f64", torch.float64, np.float64), ("ir_f32_vs_source_f32", torch.float32, np.float32)):
        ins.record = {}
        reference_outputs(model, ins, x, dtype, override=ovr if dtype == torch.float64 else None)
        rec, ins.record = ins.record, None
        env = executor.run(g, {"input": x}, keep_intermediates=True, float_dtype=fdt)
        res[tag] = _first_flip(rec, env, g, fdt is np.float64)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--weights", required=True, help="original NNCF .pth (hash check only; never unpickled here)")
    ap.add_argument("--state", required=True, help="tensors-only state written by tools/nncf_extract_state.py")
    ap.add_argument("--config", required=True, help="NNCF binarization config JSON of the model")
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--sha256")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dorefa-weight-scale", choices=("cuda", "cpu"), default="cuda",
                    help="DoReFa only: 'cuda' = the scale NNCF's CUDA kernel computes (mean|W| of output filter 0; the "
                         "checkpoint was trained and evaluated with it), 'cpu' = NNCF's CPU path (mean|W| of the whole tensor)")
    args = ap.parse_args()
    mode = MODELS[args.model_id]

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

    import nncf
    import torch
    import torchvision

    torch.set_num_threads(args.threads)
    model, meta, n_loaded, n_state = build_model(args.state, args.config)
    if meta["source_sha256"] != actual:
        print("ERROR: derived state was not extracted from --weights", file=sys.stderr)
        return 3
    dorefa_cuda = mode == "dorefa" and args.dorefa_weight_scale == "cuda"
    ins = Instrument(model, dorefa_cuda)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][PREPROCESS]
    x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
    if args.max_samples:
        x = x[: args.max_samples]
    labels = np.array([s["label"] for s in em["samples"]][: len(x)])

    cpu_path = None
    if dorefa_cuda:   # the unmodified NNCF CPU path, for the record
        ins.dorefa_cuda = False
        cpu_path, ties_cpu = reference_outputs(model, ins, x, torch.float32)
        ins.capture, ins.dorefa_cuda = {}, True
        cpu_path.astype(np.float32).tofile(os.path.join(args.out_dir, "reference_logits_nncf_cpu_path.f32"))
    ref32, ties32 = reference_outputs(model, ins, x, torch.float32)
    cap = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in ins.capture.items()}
    override = {k: {"wb": v["wb"].double(), "scale": v["scale"].double(), "sigmoid": v["sigmoid"].double()} for k, v in cap.items()}
    ref64, ties64 = reference_outputs(model, ins, x, torch.float64, override=override)
    ref64n, ties64n = reference_outputs(model, ins, x, torch.float64)
    # how the binarizers behave in float64 (naive): re-derived constants vs fp32 ones
    m64 = model.to(torch.float64)
    with torch.no_grad():
        deriv = {"max_rel_diff_weight_scale": 0.0, "max_rel_diff_threshold": 0.0, "weight_sign_differences": 0}
        for name in torch_nncf.BINARIZED:
            conv = m64.get_submodule(name)
            wbm, abm = torch_nncf.binarizer_modules(conv)
            wb64 = wbm.binarize(conv.weight)
            wb64 = wb64.to(torch.float64)
            s64, sig64 = abm.scale.exp(), abm.threshold.sigmoid().reshape(-1)
            c = cap[name]
            deriv["weight_sign_differences"] += int((torch.sign(wb64) != torch.sign(c["wb"].double())).sum())
            deriv["max_rel_diff_weight_scale"] = max(deriv["max_rel_diff_weight_scale"], float(((wb64.abs() - c["wb"].double().abs()).abs() / wb64.abs()).max()))
            t64 = sig64 * s64
            t32 = (c["sigmoid"].reshape(-1) * c["scale"]).double()
            deriv["max_rel_diff_threshold"] = max(deriv["max_rel_diff_threshold"], float(((t64 - t32).abs() / t64).max()))
    model.to(torch.float32)
    ins.capture = {}
    ref32 = ref32.astype(np.float32)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    ref64n.astype(np.float64).tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))

    nncf_dir = os.path.dirname(nncf.__file__)
    src_files = ["torch/binarization/binarize_functions.py", "torch/binarization/layers.py", "torch/binarization/algo.py", "torch/layers.py"]
    src_hashes = {f"nncf/{f}": sha256_file(os.path.join(nncf_dir, f)) for f in src_files}
    tv_file = torchvision.models.resnet.__file__
    src_hashes["torchvision/models/resnet.py"] = sha256_file(tv_file)
    cls = f"torchvision.models.resnet18(num_classes=1000) + nncf.torch.create_compressed_model({os.path.basename(args.config)})"

    def top5(r):
        return float(np.mean([labels[i] in np.argsort(-r[i])[:5] for i in range(len(x))]))

    report = {"model_id": args.model_id, "mode": mode, "n_samples": int(len(x)), "eval_preprocessing": PREPROCESS,
              "reference": {"framework": f"nncf {nncf.__version__}, torch {torch.__version__} (CPU), torchvision {torchvision.__version__}",
                            "class": cls, "config_file": os.path.basename(args.config), "config_sha256": sha256_file(args.config),
                            "source_sha256": src_hashes,
                            "checkpoint_meta": {k: meta[k] for k in ("epoch", "arch", "best_acc1", "compression_state", "top_keys", "key_prefix_example")},
                            "derived_state": {"file": os.path.basename(args.state), "sha256": sha256_file(args.state),
                                              "tool": "tools/nncf_extract_state.py", "torch": meta.get("torch_used_for_extraction"),
                                              "n_tensors": n_state, "n_loaded_by_nncf_load_state": n_loaded},
                            "top1_accuracy": float(np.mean(ref32.argmax(1) == labels)), "top5_accuracy": top5(ref32),
                            "top1_accuracy_f64": float(np.mean(ref64.argmax(1) == labels)), "top5_accuracy_f64": top5(ref64),
                            "top1_accuracy_f64_naive": float(np.mean(ref64n.argmax(1) == labels)),
                            "threshold_ties_f32": ties32, "threshold_ties_f64": ties64, "threshold_ties_f64_naive": ties64n,
                            "binarizer_constants_float64_naive_vs_fp32": deriv,
                            "dorefa_weight_scale": (args.dorefa_weight_scale if mode == "dorefa" else None),
                            "primary_f64_reference": "NNCF model in float64 with the fp32 binarizer constants (s32, sigmoid32, wb32) substituted at the binarizer call sites"},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "nncf_cpu_path_f32": (None if cpu_path is None else {
                  "note": "unmodified NNCF 2.9.0 CPU forward (DoReFa scale = mean|W| of the whole tensor); NOT the trained network",
                  "top1_accuracy": float(np.mean(cpu_path.argmax(1) == labels)), "threshold_ties": ties_cpu,
                  "vs_reference_f32": compare_outputs(ref32, cpu_path.astype(np.float32), labels)}),
              "reference_f64_naive_vs_f64": compare_outputs(ref64, ref64n, labels),
              "reference_f32_vs_f64_naive": compare_outputs(ref64n, ref32, labels),
              "variants": {}}
    print(json.dumps({k: report["reference"][k] for k in ("top1_accuracy", "top1_accuracy_f64", "top1_accuracy_f64_naive", "threshold_ties_f32",
                                                          "binarizer_constants_float64_naive_vs_fp32")}, indent=1))

    capnp = {k: {"wb": v["wb"].numpy(), "scale": float(v["scale"].reshape(-1)[0]), "sigmoid": v["sigmoid"].numpy()} for k, v in cap.items()}
    files = {}
    for variant, fold in (("unfused", False), ("fused", True)):
        graph, info = torch_nncf.convert(model, capnp, mode, args.model_id)
        applied = passes.optimize(graph, fold=fold)
        graph.manifest = {
            "schema": "bitino.manifest.v1", "model_id": args.model_id, "variant": f"{variant}, input_binarize=gt (exact)",
            "source": {"framework": "pytorch+nncf", "nncf": nncf.__version__, "class": cls, "repo": REPO, "source_sha256": src_hashes,
                       "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": actual},
            "conversion": {"tool": "tools/convert_nncf.py", "adapter": "bitino.adapters.torch_nncf", "passes": applied,
                           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "python": platform.python_version(), "torch": torch.__version__, "input_binarize": "gt",
                           "binarize_note": "NNCF {0,s} activations as gt-binarized +-1 with pad_value -1; exact (ties map to 0 / -1 in both)"},
            "input": {"name": "input", "shape": [1] + pre["shape"][1:], "layout": "NHWC", "dtype": "f32", "preprocessing": pre["definition"]},
            "outputs": ["logits"], "op_versions": {"format": 1},
            "layer_precision": info["layer_precision"], "weight_diagnostics": info["weight_diagnostics"],
        }
        if not fold:
            _EXACT[variant] = info["float64_constants"]
        path = os.path.join(args.out_dir, "model.bitino" if fold else "model_unfused.bitino")
        file_sha = bfformat.write(graph, path)
        files[variant] = path
        report["variants"][variant] = {"file": os.path.basename(path), "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                                       "input_binarize": "gt", "op_counts": graph.op_counts(), "passes": applied}
        print(f"[{variant}] wrote {path}: {graph.op_counts()} passes={applied}")

    jobs = [(k, x[i:i + 1]) for k in files for i in range(len(x))]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(files,)) as pool:
        results = pool.map(_run_one, jobs, chunksize=2)
    by_key = {k: results[j * len(x):(j + 1) * len(x)] for j, k in enumerate(files)}
    np_logits = {}
    for k, res in by_key.items():
        out32 = np.stack([r[0] for r in res]).astype(np.float32)
        out64 = np.stack([r[1] for r in res])
        np_logits[k] = out32
        out32.tofile(os.path.join(args.out_dir, f"numpy_logits_{k}.f32"))
        sem = compare_outputs(ref64, out64, labels)
        per_sample = np.abs(out64 - ref64).max(axis=1)
        sem["n_samples_max_abs_le_1e-6"] = int(np.sum(per_sample <= 1e-6))
        sem["n_samples_max_abs_le_1e-4"] = int(np.sum(per_sample <= 1e-4))
        sem["deviating_samples_gt_1e-4"] = [int(i) for i in np.where(per_sample > 1e-4)[0]]
        sem["per_sample_max_abs"] = [float(v) for v in per_sample]
        if res[0][2] is not None:
            outx = np.stack([r[2] for r in res])
            px = np.abs(outx - ref64).max(axis=1)
            sem["float64_constants"] = {"max_abs_error": float(px.max()), "n_samples_max_abs_le_1e-6": int(np.sum(px <= 1e-6)),
                                        "n_samples_max_abs_le_1e-9": int(np.sum(px <= 1e-9)),
                                        "top1_agreement": float(np.mean(outx.argmax(1) == ref64.argmax(1)))}
        report["variants"][k].update({"numpy_vs_reference_f32": compare_outputs(ref32, out32, labels),
                                      "numpy_vs_reference_f64": compare_outputs(ref64, out32, labels),
                                      "semantic_check_ir_f64_vs_reference_f64": sem})
        c32 = report["variants"][k]["numpy_vs_reference_f32"]
        print(f"[{k}] numpy f32 vs ref f32: top1 agree {c32['top1_agreement']:.4f} max_abs {c32['max_abs_error']:.3e} | SEMANTIC f64: max_abs "
              f"{sem['max_abs_error']:.3e} top1 {sem['top1_agreement']:.4f} samples<=1e-6 {sem['n_samples_max_abs_le_1e-6']}/{len(x)}"
              + (f" | f64 constants: max_abs {sem['float64_constants']['max_abs_error']:.3e}" if "float64_constants" in sem else ""))
    report["fused_vs_unfused_numpy_f32"] = compare_outputs(np_logits["unfused"], np_logits["fused"], labels)

    # First-flip analysis (fused graph) for every sample deviating by > 1e-6 in fp64 or changing top-1 in fp32.
    sem = report["variants"]["fused"]["semantic_check_ir_f64_vs_reference_f64"]
    dev = sorted(set(i for i, v in enumerate(sem["per_sample_max_abs"]) if v > 1e-6)
                 | set(np.where(np_logits["fused"].argmax(1) != ref32.argmax(1))[0].tolist()))
    _SRC.update({"model": model, "ins": ins, "override": override})
    with ctx.Pool(args.jobs, initializer=_init_worker, initargs=({"fused": files["fused"]},)) as pool:
        flips = pool.map(_flip_job, [(i, x[i:i + 1]) for i in dev], chunksize=1)
    f64 = [f for f in flips if f["ir_f64_vs_source_f64"]]
    f32 = [f for f in flips if f["ir_f32_vs_source_f32"]]
    no_flip = [f["sample"] for f in flips if not f["ir_f64_vs_source_f64"]]
    summ = {"n_analyzed": len(flips), "n_with_flip_f64": len(f64), "n_with_flip_f32": len(f32),
            "max_abs_pre_at_first_flip_f64": max([f["ir_f64_vs_source_f64"]["max_abs_pre_source"] for f in f64], default=None),
            "max_abs_pre_at_first_flip_f32": max([max(f["ir_f32_vs_source_f32"]["max_abs_pre_source"], f["ir_f32_vs_source_f32"]["max_abs_pre_ir"]) for f in f32], default=None),
            "max_dlogit_without_flip_f64": float(max([sem["per_sample_max_abs"][i] for i in no_flip], default=0.0)),
            "top1_changes_f64": int(np.sum(np.stack([r[1] for r in by_key["fused"]]).argmax(1) != ref64.argmax(1))),
            "samples": flips}
    report["first_flip_analysis_fused"] = summ
    print("first-flip:", {k: v for k, v in summ.items() if k != "samples"})
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
