#!/usr/bin/env python3
"""Export a verified Brevitas BNN-PYNQ checkpoint to QONNX (brevitas.export.export_qonnx), the first step of the QONNX
route (docs/qonnx-frontend.md phase 1; the second step is tools/import_qonnx.py in .cache/venv-qonnx).

Runs in .cache/venv-brevitas (brevitas 0.11.0, torch 2.4.1, onnx 1.16.2), with python -I. The model is loaded
exactly as tools/convert_brevitas.py loads it (``convert_brevitas.load_model``: official brevitas_examples model
definition, ``torch.load(weights_only=True)``, ``load_state_dict(strict=True)``), after the checkpoint SHA-256 is
checked against the registry.

  .cache/venv-brevitas/bin/python -I tools/export_brevitas_qonnx.py --model-id brevitas-bnn-pynq-tfc-1w1a \
      --weights .cache/checkpoints/brevitas-bnn-pynq-tfc-1w1a/tfc_1w1a-45185b4d.pth \
      --out-dir .cache/qonnx_phase1/brevitas-bnn-pynq-tfc-1w1a [--eval-dir .cache/data/eval128_mnist_brevitas]

Writes model.onnx and export.json (checkpoint and ONNX SHA-256, exporter and versions, export options). With
--eval-dir it also writes the source model's own logits on that eval set, float32 and float64
(source_logits.f32 / source_logits_f64.f64), which the importer compares with the exported graph (export check).

Note: brevitas' ONNX managers import ``onnxoptimizer``, which .cache/venv-brevitas does not have (installing it is not
allowed here). The QONNX manager uses it for two passes only, ``extract_constant_to_initializer`` and
``eliminate_unused_initializer``; this script supplies an equivalent local implementation of exactly those two
(any other pass raises), recorded in export.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "python"))

from convert_brevitas import MODELS, load_model, reference_run, sha256_file  # noqa: E402

_SHIM_PASSES = ("extract_constant_to_initializer", "eliminate_unused_initializer")


def _onnxoptimizer_shim():
    import onnx

    def optimize(model, passes):
        unknown = [p for p in passes if p not in _SHIM_PASSES]
        if unknown:
            raise NotImplementedError(f"onnxoptimizer shim: passes {unknown} not implemented")
        g = model.graph
        if "extract_constant_to_initializer" in passes:
            keep = []
            for n in g.node:
                if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
                    t = onnx.TensorProto()
                    t.CopyFrom(n.attribute[0].t)
                    t.name = n.output[0]
                    g.initializer.append(t)
                else:
                    keep.append(n)
            del g.node[:]
            g.node.extend(keep)
        if "eliminate_unused_initializer" in passes:
            used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
            dead = [t for t in g.initializer if t.name not in used]
            names = {t.name for t in dead}
            for t in dead:
                g.initializer.remove(t)
            for i in [i for i in g.input if i.name in names]:   # keep_initializers_as_inputs=True lists them as inputs
                g.input.remove(i)
        return model

    return types.SimpleNamespace(optimize=optimize, __name__="onnxoptimizer-shim")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(MODELS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--constant-folding", choices=("torch-default", "off"), default="off",
                    help="torch.onnx.export do_constant_folding (off keeps e.g. TensorNorm's var + eps in the graph, so the "
                         "importer folds it in float64)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    reg = json.load(open(args.registry))
    entry = next((m for m in reg["models"] if m.get("model_id") == args.model_id), None)
    if entry is None:
        print(f"ERROR: {args.model_id} not in {args.registry}", file=sys.stderr)
        return 2
    ck_sha = sha256_file(args.weights)
    if ck_sha != entry["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch: expected {entry['checkpoint']['sha256']}, got {ck_sha}", file=sys.stderr)
        return 3
    os.makedirs(args.out_dir, exist_ok=True)

    import brevitas
    import onnx
    import torch
    import brevitas.export.onnx.manager as bman
    from brevitas.export import export_qonnx

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    shim = None
    if getattr(bman, "opt", None) is None:
        bman.opt = _onnxoptimizer_shim()
        bman.onnx = onnx
        shim = {"onnxoptimizer": "not installed in the env; QONNXManager.onnx_passes "
                                 f"{list(_SHIM_PASSES)} run by an equivalent local implementation (tools/export_brevitas_qonnx.py)"}
    cfg_name, pre_key, _ = MODELS[args.model_id]
    model, src_hashes, cfg_dict, n_tensors = load_model(cfg_name, args.weights)
    in_ch = int({k.lower(): v for k, v in cfg_dict["MODEL"].items()}["in_channels"])
    hw = 32 if cfg_name.startswith("cnv") else 28
    shape = (1, in_ch, hw, hw)
    out = os.path.join(args.out_dir, "model.onnx")
    kw = {} if args.constant_folding == "torch-default" else {"do_constant_folding": False}
    export_qonnx(model, input_t=torch.zeros(shape), export_path=out, **kw)
    m = onnx.load(out)
    rec = {"model_id": args.model_id, "checkpoint_file": os.path.basename(args.weights), "checkpoint_sha256": ck_sha,
           "framework": f"brevitas {brevitas.__version__}, torch {torch.__version__}, onnx {onnx.__version__}",
           "exporter": f"brevitas {brevitas.__version__} export_qonnx (torch {torch.__version__} torch.onnx.export)",
           "export_options": {"input_t": f"zeros{list(shape)}", **({"do_constant_folding": False} if kw else {"do_constant_folding": "torch default (True)"})},
           "class": f"brevitas_examples.bnn_pynq.models.model_with_cfg('{cfg_name}')", "source_sha256": src_hashes,
           "checkpoint_tensors": n_tensors, "onnx_file": "model.onnx", "onnx_sha256": sha256_file(out),
           "onnx_producer": f"{m.producer_name} {m.producer_version}", "opsets": [[o.domain or "ai.onnx", o.version] for o in m.opset_import],
           "onnxoptimizer_shim": shim, "eval_preprocessing": pre_key}
    if args.eval_dir:
        em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
        pre = em["preprocessing"][pre_key]
        x = np.fromfile(os.path.join(args.eval_dir, pre["file"]), np.float32).reshape(pre["shape"])
        ref32, _ = reference_run(model, x, torch.float32)
        ref64, _ = reference_run(model, x, torch.float64)
        p32, p64 = os.path.join(args.out_dir, "source_logits.f32"), os.path.join(args.out_dir, "source_logits_f64.f64")
        ref32.astype(np.float32).tofile(p32)
        ref64.astype(np.float64).tofile(p64)
        rec.update({"eval_set": em["name"], "logits_f32": os.path.abspath(p32), "logits_f64": os.path.abspath(p64),
                    "source_f32_vs_f64_top1_disagreements": int(np.sum(ref32.argmax(1) != ref64.argmax(1)))})
    json.dump(rec, open(os.path.join(args.out_dir, "export.json"), "w"), indent=1)
    print(f"exported {args.model_id} -> {out} (sha256 {rec['onnx_sha256'][:16]}), {len(m.graph.node)} nodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
