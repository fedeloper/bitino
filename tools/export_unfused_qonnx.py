#!/usr/bin/env python3
"""Writer of the derived *_unfused.bitino files whose layout the importer's pre-pass graph does not reproduce (QONNX
step 6, docs/qonnx-step6.md): a verified checkpoint -> the converter's graph before its fold passes -> canonical QONNX
file of the Bitino profile, group-mapped (one ONNX group per IR node: python/bitino/qonnx_groups.py, the step-3
writer), so that tools/import_qonnx.py rebuilds the derived file byte for byte with no import argument.

Why a canonical file of its own: model_unfused.bitino / model_sign_zero_unfused.bitino are the legacy converters'
graphs before passes.optimize(fold=True). For most networks that is exactly the pattern importer's graph before the
same passes, so the derived file is the parent's canonical file imported with --no-fold (tools/build_from_qonnx.py).
For the families below the legacy adapter's own pre-pass graph has another layout (UNFUSED_OWN: ReLUs fused by the
adapter, the data section in the adapter's creation order, the literal BatchNorm + sign mirror of the FINN-style
adapters instead of their integer thresholds), which the parent's canonical file does not carry; the fold passes are
lossy, so it cannot be recovered from the fused .bitino either.

Two stages, one file:

  stage 1 (the legacy converter's reference environment, python -I; numpy + the framework, no onnx)
      PY -I tools/export_unfused_qonnx.py --model-id bbcu-denoise --file model_unfused.bitino --out-dir DIR \\
          [--repo SRC] [--cache-dir .cache] [--artifacts-dir artifacts] [--qonnx-python .cache/venv-qonnx/bin/python]
    loads the checkpoint (SHA-256 checked against registry/models.json) with the legacy converter's own load function,
    builds the graph with the legacy adapter and passes.optimize(fold=False) exactly as the converter does for its
    unfused file, requires it to have the published derived file's data and graph SHA-256 (the legacy route reproduces
    from the checkpoint; artifacts/ read only), and writes its group spec with the model metadata (§ 6);
  stage 2 (.cache/venv-qonnx: onnx) assembles model.onnx and refuses a file that fails check-qonnx
      (tools/export_groups_qonnx.py --assemble DIR).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "python"))

PROFILE = "bitino-qonnx/1"
TORCH_PY = "~/anaconda3/bin/python"          # the recorded conversions' torch 2.6.0 environment (recipe env "torch")
# legacy converter -> (rebuild kind, stage-1 interpreter, why the parent's canonical file does not give the layout)
UNFUSED_OWN = {
    "tools/convert_bbcu.py": ("restoration:bbcu", TORCH_PY, "the adapter fuses the first conv's ReLU itself; data in the adapter's creation order"),
    "tools/convert_bsrc.py": ("restoration:bsrc", TORCH_PY, "data section in the adapter's creation order"),
    "tools/convert_bnas.py": ("bnas", TORCH_PY, "the adapter fuses the ReLUs of its convs itself"),
    "tools/convert_adabin.py": ("adabin", TORCH_PY, "data section in the adapter's creation order (bitino.const_order of the "
                                                    "canonical file lists the fused file's constants)"),
    "tools/convert_bnext.py": ("torch_cnn", TORCH_PY, "data section in the adapter's creation order"),
    "tools/convert_brevitas.py": ("brevitas", "{cache}/venv-brevitas/bin/python",
                                  "the unfused file is the literal BatchNorm + sign mirror (ChannelAffine + Activation(sign)); "
                                  "the canonical file holds the source chain the importer folds into integer thresholds"),
    "tools/convert_finn_onnx.py": ("finn", "{cache}/venv-qonnx/bin/python", "literal BatchNorm + sign mirror (as Brevitas)"),
    "tools/convert_bnn_pynq_npz.py": ("bnnpynq", "{cache}/venv-qonnx/bin/python", "literal BatchNorm + sign mirror (as Brevitas)"),
    "tools/convert_omz.py": ("omz", "{cache}/venv-openvino/bin/python", "the unfused file is the OpenVINO parse before "
                             "merge_affine_chains; the canonical file writes the merged affine chains (and, fp16, carries the "
                             "fused file's bitino.const_order)"),
    "tools/convert_larq_zoo.py": ("keras", "{cache}/venv-larq/bin/python", "the adapter fuses Larq's activation=relu into the "
                                  "binary conv itself (the canonical file writes it as a Relu group the importer folds)"),
    "tools/convert_keras_embedding.py": ("keras", "{cache}/venv-larq/bin/python", "the adapter fuses Larq's activation=relu "
                                         "into the binary conv itself"),
}
# networks of an UNFUSED_OWN family whose unfused file stays the --no-fold derivation (which reproduces it): an IR
# operator the group writer does not have (QBConv2D / QBDense: Larq DoReFaNet; Activation leaky_relu: the BBCU x4
# super-resolution networks)
UNFUSED_NO_FOLD = {"larq-zoo-dorefanet", "bbcu-sr-l-x2", "bbcu-sr-l-x4", "bbcu-sr-m-x2", "bbcu-sr-m-x4"}
# per-model routes (a family member whose layout differs while its family stays on the --no-fold derivation)
UNFUSED_OWN_MODELS = {"shuaiz-bireal20-cifar10": ("torch_cnn", TORCH_PY, "data section in the adapter's creation order (the other "
                                                                    "IR-Net networks have first-use order)")}


def own_route(model_id: str, converter: str):
    """(kind, interpreter, why) when the model's *_unfused files get a canonical file of their own, else None."""
    if model_id in UNFUSED_OWN_MODELS:
        return UNFUSED_OWN_MODELS[model_id]
    if converter in UNFUSED_OWN and model_id not in UNFUSED_NO_FOLD:
        return UNFUSED_OWN[converter]
    return None
FILES = ("model_unfused.bitino", "model_sign_zero_unfused.bitino")


def recipes() -> dict:
    return json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]


def models() -> dict:
    return {m["model_id"]: m for m in json.load(open(os.path.join(ROOT, "registry", "models.json")))["models"]}


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def rebuild(args) -> tuple:
    """(the converter's graph before its fold passes, provenance), rebuilt from the checkpoint."""
    from bitino import passes
    rec = recipes()[args.model_id]
    kind = own_route(args.model_id, rec["converter"])[0]
    entry = models()[rec.get("registry_model_id", args.model_id)]
    lc = entry["checkpoint"]["local_cache_path"]
    weights = os.path.join(args.cache_dir, lc[len(".cache/"):]) if lc.startswith(".cache/") else os.path.join(ROOT, lc)
    weights = args.weights or weights
    ck = sha256_file(weights)
    if ck != entry["checkpoint"]["sha256"]:
        raise SystemExit(f"checkpoint hash mismatch: {ck} vs registry {entry['checkpoint']['sha256']}")
    binarize = "sign_zero" if args.file.startswith("model_sign_zero") else "ge"
    ev = rec["eval"]
    eval_dir = args.eval_dir or os.path.join(args.cache_dir, "data", ev["dir"])
    prov = {"legacy_converter": rec["converter"], "kind": kind, "checkpoint_file": os.path.basename(weights), "checkpoint_sha256": ck,
            "binarize": binarize}
    if kind in ("torch_cnn", "bnas", "adabin") or kind.startswith("restoration"):
        import torch
        torch.set_num_threads(args.threads)
        prov["framework"] = f"pytorch {torch.__version__}"
    if kind == "torch_cnn":
        import convert_torch_cnn as ct
        key = next(k for k, f in ct.FAMILIES.items() if args.model_id in f["archs"])
        model, cls, _ = ct.load_model(key, args.repo, args.model_id, weights)
        g, _ = ct.convert_graph(key, model, args.model_id, binarize)
        prov["adapter"] = f"{ct.FAMILIES[key]['adapter'].__name__} (tools/convert_torch_cnn.py family {key})"
    elif kind.startswith("restoration:"):
        import convert_restoration as cr
        fam = kind.split(":")[1]
        model, cls, _ = cr.load_model(fam, args.repo, args.model_id, weights)
        em = json.load(open(os.path.join(eval_dir, "manifest.json")))
        shp = em["preprocessing"][ev["preprocess"]]["shape"]
        g, _ = cr.convert_graph(fam, model, args.model_id, binarize, (int(shp[1]), int(shp[2])))
        prov["adapter"] = f"bitino.adapters.torch_{fam} (tools/convert_restoration.py family {fam})"
    elif kind == "bnas":
        import convert_bnas as cb
        from bitino.adapters import torch_bnas
        mk = cb.MODELS[args.model_id][0]
        model, targets, _ = cb.load_source(args.model_id, args.repo, weights)
        g, _ = torch_bnas.convert(model, targets, args.model_id, binarize=binarize,
                                  class_perm=cb.imagenet_class_perm() if mk == "imagenet" else None)
        prov["adapter"] = "bitino.adapters.torch_bnas (tools/convert_bnas.py)"
    elif kind == "adabin":
        import convert_s12_cifar as cs
        from bitino.adapters import torch_adabin
        model, cls, _, _ = cs.load_source("adabin", args.repo, weights)
        g, _ = torch_adabin.convert(model, args.model_id, binarize=binarize)
        prov["adapter"] = "bitino.adapters.torch_adabin (tools/convert_s12_cifar.py family adabin)"
    elif kind == "brevitas":
        import brevitas
        import torch
        import convert_brevitas as cv
        from bitino.adapters import brevitas_bnn
        torch.set_num_threads(args.threads)
        model, _, _, _ = cv.load_model(cv.MODELS[args.model_id][0], weights)
        g, _ = brevitas_bnn.convert(model, args.model_id, fused=False)
        prov.update({"adapter": "bitino.adapters.brevitas_bnn (fused=False)", "framework": f"brevitas {brevitas.__version__}"})
    elif kind == "keras":
        import export_keras_qonnx as ekq
        from bitino.adapters import keras_larq
        em = json.load(open(os.path.join(eval_dir, "manifest.json")))
        pre = em["preprocessing"][ev["preprocess"]]
        x = np.fromfile(os.path.join(eval_dir, pre["file"]), np.float32).reshape(pre["shape"])[:1]
        model, _, _, _, _ = ekq.build_source(args.model_id, rec, weights, args.repo, x)
        g, _ = keras_larq.convert(model, args.model_id)
        if ekq.CONVERTERS[rec["converter"]] == "zoo":   # tools/convert_larq_zoo.py: logits (softmax input) + probabilities
            sm = g.nodes[-1]
            assert sm.op == "Softmax"
            g.set_outputs(sm.inputs[0], sm.outputs[0])
        import larq
        import tensorflow as tf
        prov.update({"adapter": "bitino.adapters.keras_larq", "framework": f"tensorflow {tf.__version__}, larq {larq.__version__}"})
    elif kind == "omz":
        import convert_omz as co
        from bitino.adapters import openvino_ir as oi
        sub, prec, stem = co.MODELS[args.model_id]
        d = os.path.join(args.checkpoint_dir or os.path.join(args.cache_dir, "checkpoints"), sub, prec)
        xml, binf = os.path.join(d, stem + ".xml"), os.path.join(d, stem + ".bin")
        prov["checkpoint_sha256"] = {os.path.basename(binf): sha256_file(binf), os.path.basename(xml): sha256_file(xml)}
        ir = oi.IRModel(xml, binf)
        g, _ = oi.convert(ir, args.model_id)      # tools/convert_omz.py build_graphs: the unfused file is the parse + prune
        prov.update({"adapter": "bitino.adapters.openvino_ir", "framework": f"openvino-ir v{ir.version} ({prec})"})
    elif kind in ("finn", "bnnpynq"):
        import export_ir_qonnx as eiq
        from bitino.adapters import finn_bnn
        ns = types.SimpleNamespace(model_id=args.model_id, cache_dir=args.cache_dir, eval_dir=eval_dir, no_source_logits=True,
                                   tmp_dir=args.tmp_dir, threads=args.threads, max_samples=0, weights=args.weights,
                                   theano_python=args.theano_python, lasagne_dir=args.lasagne_dir, bnnpynq_dir=args.bnnpynq_dir)
        em = json.load(open(os.path.join(eval_dir, "manifest.json")))
        pre = em["preprocessing"][ev["preprocess"]]
        x = np.fromfile(os.path.join(eval_dir, pre["file"]), np.float32).reshape(pre["shape"])[:1]
        var = eiq.SOURCES[kind](ns, rec, entry, x)["variants"][0]
        g, _ = finn_bnn.emit(var["inp"], var["layers"], fused=False)
        prov.update({"adapter": f"bitino.adapters.finn_bnn.emit(fused=False) ({'finn_onnx' if kind == 'finn' else 'bnn_pynq_npz'} reader)",
                     "framework": "onnx (FINN release)" if kind == "finn" else "theano+lasagne (BNN-PYNQ npz)"})
    else:
        raise AssertionError(kind)
    prov["passes"] = passes.optimize(g, fold=False)
    return g, prov


def stage1(args) -> int:
    from bitino import format as bfformat, qonnx_groups as qg
    from bitino.fetch import data_sha256, graph_sha256
    rec = recipes()[args.model_id]
    if own_route(args.model_id, rec["converter"]) is None:
        raise SystemExit(f"{args.model_id}: {rec['converter']} is not an UNFUSED_OWN family (its unfused file is the "
                         "parent canonical file imported with --no-fold)")
    os.makedirs(args.out_dir, exist_ok=True)
    g, prov = rebuild(args)
    tmp = os.path.join(args.out_dir, "rebuilt.bitino")
    g.manifest = {}
    bfformat.write(g, tmp)
    rebuilt = {"data_sha256": data_sha256(tmp), "graph_sha256": graph_sha256(tmp)}
    from export_groups_qonnx import expected_hashes
    published = expected_hashes(args.model_id, args.file, args.artifacts_dir, args.expect_recipe)
    os.remove(tmp)
    prov["rebuilt_vs_published"] = {"rebuilt": rebuilt, "published": published, "identical": rebuilt == published}
    print(f"rebuilt {args.model_id}/{args.file}: {'= published' if rebuilt == published else '!= published'}", flush=True)
    if rebuilt != published:
        json.dump({"model_id": args.model_id, "file": args.file, "error": "rebuilt graph != published", **prov},
                  open(os.path.join(args.out_dir, "export.json"), "w"), indent=1)
        print("ERROR: the legacy adapter does not reproduce the published derived file from the checkpoint", file=sys.stderr)
        return 6
    spec, rep = qg.write_spec(g)
    entry = models()[rec.get("registry_model_id", args.model_id)]
    ev = rec["eval"]
    em = json.load(open(os.path.join(args.eval_dir or os.path.join(args.cache_dir, "data", ev["dir"]), "manifest.json")))
    pre = em["preprocessing"][ev["preprocess"]]
    va = ev.get("validate_args", [])
    if "--output-kind" in va and va[va.index("--output-kind") + 1] == "image":
        labels = "none"
    else:
        nc = int(va[va.index("--num-classes") + 1]) if "--num-classes" in va else 1000
        name = em.get("name", "")
        lset = next((s for k, s in (("imagenet", "imagenet-1k"), ("cifar100", "cifar100"), ("cifar10", "cifar10"), ("mnist", "mnist"),
                                    ("gtsrb", "gtsrb")) if k in name), f"{nc}-classes")
        labels = json.dumps({"num_classes": nc, "set": lset})
    src = (rec.get("sources") or [{}])[0]
    lic = (entry.get("license") or {}).get("weights") or "unknown"
    why = own_route(args.model_id, rec["converter"])[2]
    meta = {"bitino.profile": PROFILE, "bitino.model_id": args.model_id, "bitino.variant": "float", "bitino.layout": "NCHW",
            "bitino.input_domain": "float32",
            "bitino.preprocessing": json.dumps({"key": ev["preprocess"], "definition": pre["definition"]}),
            "bitino.labels": labels, "bitino.source.framework": prov.get("framework", "").split(" ")[0] or "unknown",
            "bitino.source.framework_version": prov.get("framework", ""), "bitino.source.checkpoint_sha256": (prov["checkpoint_sha256"] if isinstance(prov["checkpoint_sha256"], str)
                                                else json.dumps(prov["checkpoint_sha256"], sort_keys=True)),
            "bitino.weights_license": lic,
            "bitino.exporter": f"tools/export_unfused_qonnx.py@{_git_commit() or 'unknown'} (direct writer of the converter's graph "
                               f"before its fold passes, {prov['adapter']}); numpy {np.__version__}",
            "bitino.notes": f"derived file {args.file} (QONNX step 6): the legacy converter's graph before passes.optimize(fold=True), "
                            f"group mapping; a canonical file of its own because {why}"}
    if src.get("repo") and src.get("commit"):
        meta["bitino.source.code"] = f"{src['repo']}@{src['commit']}"
    spec.meta.update(meta)
    qg.save_spec(spec, os.path.join(args.out_dir, "spec"))
    exp = {"model_id": args.model_id, "file": args.file, **prov, "writer": rep, "ir_nodes": len(g.nodes), "ir_constants": len(g.constants),
           "import_args": [], "exporter": meta["bitino.exporter"], "framework": f"{prov.get('framework', '')}, numpy {np.__version__}, "
                                                                              f"python {platform.python_version()}"}
    json.dump(exp, open(os.path.join(args.out_dir, "spec", "export.json"), "w"), indent=1)
    del g, spec
    py = args.qonnx_python or os.path.join(args.cache_dir, "venv-qonnx", "bin", "python")
    r = subprocess.run([py, "-I", os.path.join(ROOT, "tools", "export_groups_qonnx.py"), "--assemble", args.out_dir])
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--file", required=True, choices=FILES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", help="the authors' source tree (PyTorch families; read only)")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"), help="published files (read only)")
    ap.add_argument("--qonnx-python", help="interpreter of stage 2 (default <cache>/venv-qonnx/bin/python)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/tmp"), help="scratch (the BNN-PYNQ Theano run)")
    # bitino fetch: the recipe's own downloads and source trees instead of the checkout's cache layout
    ap.add_argument("--weights", help="checkpoint (default: the registry's local_cache_path under --cache-dir)")
    ap.add_argument("--eval-dir", help="eval set (default <cache>/data/<recipe eval dir>)")
    ap.add_argument("--checkpoint-dir", help="OpenVINO IR root <dir>/<model>/<precision>/ (default <cache>/checkpoints)")
    ap.add_argument("--theano-python", help="BNN-PYNQ: Python 2.7 Theano interpreter (default <cache>/env-theano27/bin/python)")
    ap.add_argument("--lasagne-dir", help="BNN-PYNQ: Lasagne source tree")
    ap.add_argument("--bnnpynq-dir", help="BNN-PYNQ: bnn/src/training of the BNN-PYNQ source tree")
    ap.add_argument("--expect-recipe", action="store_true",
                    help="compare the rebuilt graph with the recipe's recorded hashes instead of artifacts/ (bitino fetch)")
    args = ap.parse_args()
    return stage1(args)


if __name__ == "__main__":
    sys.exit(main())
