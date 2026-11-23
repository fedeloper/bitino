#!/usr/bin/env python3
"""Direct writer of step 3 of the QONNX route (docs/qonnx-frontend.md § Plan, docs/qonnx-step3.md): a verified
PyTorch checkpoint of a transformer (BiBERT, Bi-ViT, GSB, BHViT) or of a "Faster Than Lies" detector (BNext with the
ImageGrayU8 / FftLogMagnitude / LocalBinaryPattern front end) -> canonical QONNX file of the Bitino profile
(docs/qonnx-profile.md, ``bitino-qonnx/1``), one ONNX group per IR node (python/bitino/qonnx_groups.py).

Two stages, one file:

  stage 1 (the family's reference environment, python -I; numpy + torch, no onnx)
      PY -I tools/export_groups_qonnx.py --model-id bibert-bert-base-sst2 --file model.bitino \
          --out-dir .cache/qonnx_step3/bibert-bert-base-sst2/model [--cache-dir .cache] [--artifacts-dir artifacts]
    loads the checkpoint exactly as the legacy converter does (its own load functions: tools/convert_bibert.py,
    tools/convert_bivit.py, tools/convert_bhvit.py, tools/convert_ftl_deepfake.py; checkpoint SHA-256 checked against
    the registry), builds the deployed graph with the legacy adapter, the converter's passes and the recipe's derive
    step (tools/derive_variant.py drop_gelu_before_sign for BiBERT and Bi-ViT model.bitino), checks that this graph has
    the published file's data and graph SHA-256 (the legacy route reproduces from the checkpoint), and writes the
    group spec of that graph (python/bitino/qonnx_groups.write_spec: every IR node as its standard-op pattern,
    values from the rebuilt graph, i.e. from the checkpoint) with the model metadata (§ 6);
  stage 2 (.cache/venv-qonnx: onnx; started by stage 1, or ``--assemble DIR``)
      assembles model.onnx (IR 10, ai.onnx 17 [+ qonnx.custom_op.general 1]) and refuses a file that does not pass
      check-qonnx (bitino.qonnx_profile).

Families (recipe converter -> how the deployed graph is rebuilt):
  tools/convert_bibert.py      torch_bibert.load_official + convert(seq_len of the eval set) + passes.optimize(fold=False)
  tools/convert_bivit.py       torch_bivit / torch_gsb load_checkpoint + build_reference + convert(binarize) + optimize(fold=True)
  tools/convert_bhvit.py       bhvit.load_checkpoint + build_reference + convert(binarize) + optimize(fold=True)
  tools/convert_ftl_deepfake.py convert_ftl_deepfake.load_model + torch_ftl_deepfake.convert(binarize) + optimize(fold=True)
  model.bitino: binarize ge; model_sign_zero.bitino: binarize sign_zero; a tools/derive_variant.py step of the
  recipe is applied to model.bitino as the recipe does.

Writes into --out-dir: model.onnx, export.json (checkpoint / ONNX SHA-256, versions, the rebuilt graph's hashes vs
the published ones, the writer report, check-qonnx result, the source logits files used for the export check).
The source framework's own outputs on the check set are the published conversion's reference files
(artifacts/<id>/reference_logits.f32 / reference_logits_f64.f64: the same converter, checkpoint and eval set), read
only; they are copied to --out-dir as source_logits.f32 / source_logits_f64.f64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "python"))

PROFILE = "bitino-qonnx/1"
CONVERTERS = {"tools/convert_bibert.py": "bibert", "tools/convert_bivit.py": "bivit", "tools/convert_bhvit.py": "bhvit",
              "tools/convert_ftl_deepfake.py": "ftl"}
# reference environments of the recorded conversions (read only; python -I)
ENVS = {"bibert": "~/anaconda3/envs/bibert/bin/python", "bivit": "~/anaconda3/bin/python",
        "bhvit": "~/anaconda3/envs/bhvit/bin/python", "ftl": "~/anaconda3/envs/capitello/bin/python"}
# torch CPU threads of the recorded conversions (the fp32 weight-quantizer reductions can depend on it)
THREADS = {"bibert": 8, "bivit": 8, "bhvit": 16, "ftl": 8}
REPOS = {"https://github.com/htqin/BiBERT": "BiBERT", "https://github.com/YanjingLi0202/Bi-ViT": "YanjingLi0202_Bi-ViT",
         "https://github.com/IMRL/GSB-Vision-Transformer": "IMRL_GSB-Vision-Transformer", "https://github.com/IMRL/BHViT": "IMRL_BHViT",
         "https://github.com/hpi-xnor/BNext": "BNext"}
# converter flags of the recipes that the rebuild reproduces (any other flag stops the export)
_COMMON = {"--model-id", "--repo", "--eval-dir", "--out-dir"}
KNOWN_FLAGS = {"bibert": _COMMON | {"--ckpt-dir", "--threads", "--drop-gelu-before-sign", "--gelu-fp32-zero-set"},
               "bivit": _COMMON | {"--weights", "--jobs", "--torch-threads"},
               "bhvit": _COMMON | {"--weights", "--jobs", "--torch-threads"},
               "ftl": _COMMON | {"--bnext-repo", "--bnext-weights", "--weights", "--jobs", "--torch-threads"}}
GLUE = {"bibert-bert-base-sst2": ("sst-2", 2), "bibert-bert-base-mrpc": ("mrpc", 2), "bibert-bert-base-qqp": ("qqp", 2),
        "bibert-bert-base-mnli": ("mnli", 3)}


def sha256_file(path: str) -> str:
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


def recipes() -> dict:
    return json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]


def family_of(model_id: str) -> str:
    rec = recipes().get(model_id)
    if rec is None or rec.get("converter") not in CONVERTERS:
        raise SystemExit(f"{model_id}: not a step-3 recipe (converters {sorted(CONVERTERS)})")
    return CONVERTERS[rec["converter"]]


def step3_models() -> list:
    return [k for k, v in recipes().items() if v.get("converter") in CONVERTERS]


def published_files(model_id: str, artifacts: str) -> list:
    return [f for f in ("model.bitino", "model_sign_zero.bitino") if os.path.exists(os.path.join(artifacts, model_id, f))]


def legacy_steps(rec: dict) -> list:
    """The recipe's legacy converter steps (registry/recipes.json -> crosscheck.steps; before the QONNX build path
    became the recipes' steps they were the recipe's steps)."""
    return (rec.get("crosscheck") or {}).get("steps") or rec["steps"]


def local_paths(model_id: str, cache: str, over: dict | None = None) -> dict:
    """Checkpoint, source trees and extra downloads of a recipe in the checkout's cache (the converters' inputs).
    over (bitino fetch: the recipe's own downloads and source trees): {"weights": path, "sources": {name: dir},
    "downloads": {name: path}} replace the cache layout."""
    over = over or {}
    rec = recipes()[model_id]
    reg = {m["model_id"]: m for m in json.load(open(os.path.join(ROOT, "registry", "models.json")))["models"]}[model_id]
    lc = reg["checkpoint"]["local_cache_path"]
    weights = os.path.join(cache, lc[len(".cache/"):]) if lc.startswith(".cache/") else os.path.join(ROOT, lc)
    weights = over.get("weights") or weights
    srcs = {}
    for s in rec.get("sources", []):
        name = REPOS.get(s.get("repo", ""))
        if s["name"] in (over.get("sources") or {}):
            srcs[s["name"]] = over["sources"][s["name"]]
        elif name and os.path.isdir(os.path.join(cache, "repos", name)):
            srcs[s["name"]] = os.path.join(cache, "repos", name)
        else:    # fetched into its own directory by tools/qonnx_step3_acceptance.py (bitino.fetch.get_source)
            nm = s["repo"].rstrip("/").split("github.com/")[-1].replace("/", "-") + "-" + s["commit"][:12]
            srcs[s["name"]] = os.path.join(cache, "qonnx_step3", "_src", "src", nm)
    extra = {}
    for d in rec.get("downloads", []):
        if d["name"] == "bnext_weights":
            size = {"t": "tiny", "s": "small", "m": "middle"}[model_id.split("-bnext-")[1][0]]
            extra["bnext_weights"] = ((over.get("downloads") or {}).get("bnext_weights")
                                      or os.path.join(cache, "checkpoints", f"bnext-{size}", "checkpoint.pth.tar"))
            extra["bnext_weights_sha256"] = d["sha256"]
    return {"weights": weights, "checkpoint_sha256": reg["checkpoint"]["sha256"], "sources": srcs, "license": reg.get("license", {}),
            **extra}


# ================================================================================================ stage 1: rebuild
def rebuild(model_id: str, file: str, cache: str, threads: int, over: dict | None = None, eval_dir: str | None = None):
    """(deployed IR graph, provenance) rebuilt from the checkpoint exactly as the recipe's legacy steps do."""
    import torch
    from bitino import passes
    fam = family_of(model_id)
    rec = recipes()[model_id]
    paths = local_paths(model_id, cache, over)
    ck = sha256_file(paths["weights"])
    if ck != paths["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint hash mismatch: {ck} vs registry {paths['checkpoint_sha256']}")
    torch.set_num_threads(threads)
    binarize = "sign_zero" if file.startswith("model_sign_zero") else "ge"
    # step 6: the derived *_unfused.bitino files are the converter's graph before its fold passes (no derive step);
    # a group-mapped canonical file holds the post-pass graph, so the unfused file gets a canonical file of its own
    fold = not file.endswith("_unfused.bitino")
    prov = {"family": fam, "legacy_converter": rec["converter"], "checkpoint_file": os.path.basename(paths["weights"]),
            "checkpoint_sha256": ck, "torch": torch.__version__, "torch_threads": threads, "binarize": binarize}
    ev = rec["eval"]
    eval_dir = eval_dir or os.path.join(cache, "data", ev["dir"])
    steps = legacy_steps(rec)
    flags = {a for a in steps[0]["args"] if a.startswith("--")}
    unknown = flags - KNOWN_FLAGS[fam]
    if unknown:
        raise SystemExit(f"{model_id}: converter flags {sorted(unknown)} are not reproduced by this exporter")
    prov["converter_flags"] = sorted(flags & {"--drop-gelu-before-sign", "--gelu-fp32-zero-set"})
    if fam == "bibert":
        import convert_bibert as cb
        from bitino.adapters import torch_bibert
        if binarize != "ge":
            raise SystemExit("BiBERT publishes model.bitino only")
        ckdir = os.path.dirname(paths["weights"])
        model = torch_bibert.load_official(paths["sources"]["repo"], ckdir)
        _, ids, _, _, _ = cb.load_eval(eval_dir)
        tz = None
        if "--gelu-fp32-zero-set" in flags:      # as tools/convert_bibert.py: the fp32 GELU's zero set of the source
            import transformer.modeling_quant as mq
            tz = cb.fp32_gelu_zero_threshold(mq.gelu)["t"]
            prov["gelu_fp32_zero_below"] = tz
        g, info = torch_bibert.convert(model, model_id, seq_len=int(ids.shape[1]), gelu_zero_below=tz)
        applied = passes.optimize(g, fold=False)
        if "--drop-gelu-before-sign" in flags:   # in the converter process (not a re-read file)
            applied["drop_gelu_before_sign"] = passes.drop_gelu_before_sign(g)
            applied["prune_unused_constants"] += passes.prune_unused_constants(g)
        prov.update({"adapter": "bitino.adapters.torch_bibert", "seq_len": int(ids.shape[1]), "passes": applied,
                     "class": "transformer.modeling_quant.BertForSequenceClassification"})
    elif fam == "bivit":
        import convert_bivit as cv
        f = cv.FAMILIES[cv.family_of(model_id)]
        ad = f["adapter"]
        ckpt = ad.load_checkpoint(paths["weights"])
        model, _, _ = ad.build_reference(paths["sources"]["repo"], model_id, ckpt)
        g, info = ad.convert(model, model_id, binarize=binarize)
        applied = passes.optimize(g, fold=fold)
        prov.update({"adapter": ad.__name__, "passes": applied, "class": f["class"]})
    elif fam == "bhvit":
        from bitino.adapters import bhvit
        ckpt = bhvit.load_checkpoint(paths["weights"])
        model, _ = bhvit.build_reference(paths["sources"]["repo"], model_id, ckpt)
        g, info = bhvit.convert(model, model_id, binarize=binarize)
        applied = passes.optimize(g, fold=fold)
        prov.update({"adapter": "bitino.adapters.bhvit", "passes": applied,
                     "class": "transformer.BHViT.BHViTForImageClassification (models.get_model)"})
    elif fam == "ftl":
        import convert_ftl_deepfake as cf
        from bitino.adapters import torch_ftl_deepfake
        if sha256_file(paths["bnext_weights"]) != paths["bnext_weights_sha256"]:
            raise SystemExit("BNext initialisation checkpoint hash mismatch")
        try:
            import torchinfo  # noqa: F401
        except ImportError:
            # BNext/src/bnext.py imports torchinfo.summary (model summaries) and never calls it; an inert stand-in,
            # like the converter's own stand-ins for lightning / fvcore / torchmetrics
            import types

            def _unused(*a, **k):
                raise RuntimeError("torchinfo.summary is not available in the export process")
            sys.modules["torchinfo"] = types.SimpleNamespace(summary=_unused)
            prov["stand_ins"] = ["torchinfo.summary (imported by BNext/src/bnext.py, never called)"]
        model, _, meta = cf.load_model(paths["sources"]["repo"], paths["sources"]["bnext"], paths["bnext_weights"], paths["weights"])
        g, info = torch_ftl_deepfake.convert(model, model_id, binarize=binarize)
        applied = passes.optimize(g, fold=fold)
        prov.update({"adapter": "bitino.adapters.torch_ftl_deepfake", "passes": applied, "class": meta["class"],
                     "bnext_init_checkpoint_sha256": paths["bnext_weights_sha256"]})
    else:
        raise AssertionError(fam)
    derived = []
    if file == "model.bitino" and steps[1:]:
        # tools/derive_variant.py reads the converter's file: the same graph with a fresh unique-name counter
        import tempfile
        from bitino import format as bfformat
        with tempfile.TemporaryDirectory() as td:
            bfformat.write(g, os.path.join(td, "m.bitino"))
            g = bfformat.read(os.path.join(td, "m.bitino"))
        for st in steps[1:]:
            if st["tool"] != "tools/derive_variant.py":
                raise SystemExit(f"recipe step {st['tool']} is not reproduced by this exporter")
            a = st["args"]
            names = [a[i + 1] for i in range(len(a)) if a[i] == "--pass"]
            kw = {}
            if "--keep-zero-below" in a:
                kw["keep_zero_below"] = float(a[a.index("--keep-zero-below") + 1])
            done = {p: {"drop_gelu_before_sign": passes.drop_gelu_before_sign}[p](g, **kw) for p in names}
            done["prune_unused_constants"] = passes.prune_unused_constants(g)
            derived.append({"tool": "tools/derive_variant.py", "passes": done, **kw})
    prov["derived"] = derived
    del model
    return g, prov


def stage1(args) -> int:
    from bitino import format as bfformat, qonnx_groups as qg
    from bitino.fetch import data_sha256, graph_sha256
    import torch
    fam = family_of(args.model_id)
    rec = recipes()[args.model_id]
    os.makedirs(args.out_dir, exist_ok=True)
    threads = args.threads or THREADS[fam]
    over = fetch_overrides(args)
    g, prov = rebuild(args.model_id, args.file, args.cache_dir, threads, over, args.eval_dir)
    tmp = os.path.join(args.out_dir, "rebuilt.bitino")
    g.manifest = {}
    bfformat.write(g, tmp)
    rebuilt = {"data_sha256": data_sha256(tmp), "graph_sha256": graph_sha256(tmp)}
    published = expected_hashes(args.model_id, args.file, args.artifacts_dir, args.expect_recipe)
    os.remove(tmp)
    prov["rebuilt_vs_published"] = {"rebuilt": rebuilt, "published": published, "identical": rebuilt == published}
    print(f"rebuilt {args.model_id}/{args.file}: data {rebuilt['data_sha256'][:16]} graph {rebuilt['graph_sha256'][:16]} "
          f"-> {'= published' if rebuilt == published else '!= published ' + str(published)}", flush=True)
    if rebuilt != published and not args.allow_mismatch:
        json.dump({"model_id": args.model_id, "file": args.file, "error": "rebuilt graph != published", **prov},
                  open(os.path.join(args.out_dir, "export.json"), "w"), indent=1)
        print("ERROR: the legacy route does not reproduce the published file from the checkpoint", file=sys.stderr)
        return 6
    spec, rep = qg.write_spec(g)
    reg = {m["model_id"]: m for m in json.load(open(os.path.join(ROOT, "registry", "models.json")))["models"]}[args.model_id]
    ev = rec["eval"]
    em = json.load(open(os.path.join(args.eval_dir or os.path.join(args.cache_dir, "data", ev["dir"]), "manifest.json")))
    pre = em["preprocessing"][ev["preprocess"]]
    if fam == "bibert":
        word = next(n for n in g.nodes if n.op == "Gather" and n.inputs[1] == "input_ids")
        tt = next(n for n in g.nodes if n.op == "Gather" and n.inputs[1] == "token_type_ids")
        dom = json.dumps({"input_ids": f"ids:{g.constants[word.inputs[0]].shape[0]}",
                          "token_type_ids": f"ids:{g.constants[tt.inputs[0]].shape[0]}", "attention_mask": "float32"})
        task, ncls = GLUE[args.model_id]
        labels = json.dumps({"num_classes": ncls, "set": f"glue-{task}"})
    else:
        dom = "float32"
        name = em.get("name", "")
        if fam == "ftl":
            labels = json.dumps({"num_classes": 2, "set": "ftl-real-fake (0 = fake, 1 = real)"})
        elif "cifar100" in name:
            labels = json.dumps({"num_classes": 100, "set": "cifar100"})
        else:
            labels = json.dumps({"num_classes": 1000, "set": "imagenet-1k"})
    ck = {prov["checkpoint_file"]: prov["checkpoint_sha256"]}
    if "bnext_init_checkpoint_sha256" in prov:
        ck["checkpoint.pth.tar (BNext ImageNet initialisation, overwritten by the detector checkpoint)"] = prov["bnext_init_checkpoint_sha256"]
    fw = f"torch {torch.__version__}"
    for mod in ("timm", "transformers", "torchvision"):
        try:
            fw += f", {mod} {__import__(mod).__version__}"
        except Exception:
            pass
    commit = _git_commit()
    src = (rec.get("sources") or [{}])[0]
    lic = (reg.get("license") or {}).get("weights") or "unknown"
    role = (f"derived file {args.file} (the converter's graph before its fold passes, step 6)"
            if args.file.endswith("_unfused.bitino") else f"published file {args.file}")
    notes = (f"{role}; activation binarizer of BDense / BConv2D: {prov['binarize']} "
             f"({'BipolarQuant(x, 1): 0 -> +1' if prov['binarize'] == 'ge' else 'Sign: sign(0) = 0, torch.sign'}); "
             f"group mapping (one ONNX group per IR node of the deployed graph, bitino.import)")
    if prov["derived"]:
        notes += f"; derived variant: {prov['derived']}"
    meta = {"bitino.profile": PROFILE, "bitino.model_id": args.model_id, "bitino.variant": "float", "bitino.layout": "NCHW",
            "bitino.input_domain": dom, "bitino.preprocessing": json.dumps({"key": ev["preprocess"], "definition": pre["definition"]}),
            "bitino.labels": labels, "bitino.source.framework": "pytorch", "bitino.source.framework_version": fw,
            "bitino.source.checkpoint_sha256": json.dumps(ck) if len(ck) > 1 else prov["checkpoint_sha256"],
            "bitino.weights_license": lic,
            "bitino.exporter": f"tools/export_groups_qonnx.py@{commit or 'unknown'} (direct writer of the deployed graph, "
                               f"{prov['adapter']} + {rec['converter']} passes); {fw}",
            "bitino.notes": notes}
    if src.get("repo") and src.get("commit"):
        meta["bitino.source.code"] = f"{src['repo']}@{src['commit']}"
    spec.meta.update(meta)
    qg.save_spec(spec, os.path.join(args.out_dir, "spec"))
    art = os.path.join(args.artifacts_dir, args.model_id)
    logits = {}
    if args.expect_recipe:      # bitino fetch: no published conversion to read source logits from
        art = os.path.join(args.out_dir, "_no_published_files")
    for a, b in (("reference_logits.f32", "source_logits.f32"), ("reference_logits_f64.f64", "source_logits_f64.f64")):
        if os.path.exists(os.path.join(art, a)):
            shutil.copyfile(os.path.join(art, a), os.path.join(args.out_dir, b))
            logits["logits_f32" if b.endswith("f32") else "logits_f64"] = os.path.abspath(os.path.join(args.out_dir, b))
    crep_pub = os.path.join(art, "conversion_report.json")
    if os.path.exists(crep_pub):      # the published references cover a subset of the check set (FTL --max-samples)
        sel = json.load(open(crep_pub)).get("sample_indices")
        if sel is not None:
            logits["source_sample_indices"] = [int(i) for i in sel]
    exp = {"model_id": args.model_id, "file": args.file, **prov, "writer": rep, "ir_nodes": len(g.nodes),
           "ir_constants": len(g.constants), "import_args": ["--naming", "exporter", "--cleanup", "none"],
           "exporter": meta["bitino.exporter"], "framework": f"{fw}, numpy {np.__version__}, python {platform.python_version()}",
           "eval_set": em.get("name"), "eval_dir": ev["dir"], "eval_preprocessing": ev["preprocess"],
           "source_logits": "published conversion's reference files (same converter, checkpoint and eval set), read only",
           **logits, "venv": f"stage 1 {ENVS[fam]} (read only, python -I), stage 2 .cache/venv-qonnx"}
    json.dump(exp, open(os.path.join(args.out_dir, "spec", "export.json"), "w"), indent=1)
    del g, spec
    py = args.qonnx_python or os.path.join(args.cache_dir, "venv-qonnx", "bin", "python")
    r = subprocess.run([py, "-I", os.path.abspath(__file__), "--assemble", args.out_dir] + (["--keep-spec"] if args.keep_spec else []))
    return r.returncode


# ================================================================================================ stage 2: assemble
def assemble(out_dir: str, keep_spec: bool) -> int:
    import onnx
    from bitino import qonnx_groups as qg
    from bitino.qonnx_profile import check_file
    sdir = os.path.join(out_dir, "spec")
    exp = json.load(open(os.path.join(sdir, "export.json")))
    path = os.path.join(out_dir, "model.onnx")
    m = qg.assemble(sdir, path, "bitino tools/export_groups_qonnx.py",
                    extra_meta={"bitino.exporter": json.load(open(os.path.join(sdir, "spec.json")))["meta"]["bitino.exporter"]
                                + f"; onnx {onnx.__version__}"})
    del m
    rep = check_file(path).to_dict()
    if not rep.get("passed"):
        print(json.dumps(rep, indent=1)[:4000], file=sys.stderr)
        print("ERROR: the written file does not pass check-qonnx (profile bitino-qonnx/1)", file=sys.stderr)
        return 5
    exp.update({"onnx_file": "model.onnx", "onnx_sha256": sha256_file(path), "onnx_bytes": os.path.getsize(path),
                "onnx": onnx.__version__, "profile_check": {k: rep.get(k) for k in ("profile", "passed", "summary")} | {
                    "warnings": [f for f in rep.get("findings", []) if f.get("level") == "warning"]}})
    json.dump(exp, open(os.path.join(out_dir, "export.json"), "w"), indent=1)
    if not keep_spec:
        shutil.rmtree(sdir)
    print(f"exported {exp['model_id']}/{exp['file']} -> {path} ({exp['onnx_bytes']} bytes, sha256 {exp['onnx_sha256'][:16]}); "
          f"check-qonnx {rep.get('summary', {}).get('nodes')} nodes, passed", flush=True)
    return 0


def fetch_overrides(args) -> dict:
    """--weights / --source NAME=DIR / --download NAME=PATH (bitino fetch) -> local_paths overrides."""
    def kv(items):
        return dict(i.split("=", 1) for i in items or [])
    return {"weights": args.weights, "sources": kv(args.source), "downloads": kv(args.download)}


def expected_hashes(model_id: str, file: str, artifacts: str, from_recipe: bool) -> dict:
    """The data + graph SHA-256 the rebuilt graph must have: the published file's (artifacts/, read only) or, with
    --expect-recipe (bitino fetch, no published files), the recipe's recorded hashes (registry/recipes.json ->
    expected[file])."""
    from bitino.fetch import data_sha256, graph_sha256
    if from_recipe:
        exp = recipes()[model_id].get("expected", {}).get(file)
        if not exp:
            raise SystemExit(f"{model_id}/{file}: no recorded hashes in registry/recipes.json -> expected")
        return {"data_sha256": exp["data_sha256"], "graph_sha256": exp["graph_sha256"]}
    pub = os.path.join(artifacts, model_id, file)
    return {"data_sha256": data_sha256(pub), "graph_sha256": graph_sha256(pub)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id")
    ap.add_argument("--file", default="model.bitino", choices=("model.bitino", "model_sign_zero.bitino", "model_unfused.bitino",
                                                                 "model_sign_zero_unfused.bitino"))
    ap.add_argument("--out-dir")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"), help="published files (read only)")
    ap.add_argument("--qonnx-python")
    ap.add_argument("--weights", help="checkpoint (default: the registry's local_cache_path under --cache-dir)")
    ap.add_argument("--source", action="append", metavar="NAME=DIR", help="source tree of the recipe (default: under --cache-dir)")
    ap.add_argument("--download", action="append", metavar="NAME=PATH", help="other download of the recipe (bnext_weights)")
    ap.add_argument("--eval-dir", help="eval set (default <cache>/data/<recipe eval dir>)")
    ap.add_argument("--expect-recipe", action="store_true",
                    help="compare the rebuilt graph with the recipe's recorded hashes instead of artifacts/ (bitino fetch)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (default: the recorded conversion's)")
    ap.add_argument("--allow-mismatch", action="store_true", help="write the spec even if the rebuilt graph != published (debug)")
    ap.add_argument("--keep-spec", action="store_true")
    ap.add_argument("--assemble", metavar="DIR")
    ap.add_argument("--list", action="store_true", help="print the step-3 model ids of the registry")
    args = ap.parse_args()
    if args.list:
        print("\n".join(step3_models()))
        return 0
    if args.assemble:
        return assemble(args.assemble, args.keep_spec)
    if not (args.model_id and args.out_dir):
        ap.error("--model-id and --out-dir are required")
    return stage1(args)


if __name__ == "__main__":
    sys.exit(main())
